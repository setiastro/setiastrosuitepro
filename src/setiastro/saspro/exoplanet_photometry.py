# src/setiastro/saspro/exoplanet_photometry.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Time-series differential photometry for the exoplanet / variable-star tool.

No Qt here; ExoPlanetWindow drives it.  The pieces, in pipeline order:

  saturation      level from MAXLIN/SATURATE (scaled into loaded units) or
                  a flat-topped plateau at the frame maximum; saturated
                  pixels kept as a sparse coordinate list per frame.
  tracking        per-frame affine (reference -> frame) from SEP catalogues:
                  offset voting, then least-squares affine on matched pairs,
                  180-degree (meridian flip) retry, astroalign last.  The
                  apertures follow the stars -- frames are never resampled
                  for photometry.
  measurement     per star per frame: windowed recentring, sigma-clipped
                  annulus median sky (0.0 = no-data pixels excluded), exact-
                  overlap aperture sums at a grid of radii, CCD-equation
                  errors, and EDGE / SATURATED / SKY flags.
  ensemble        inverse-variance weighted artificial comparison star with
                  iterative rejection of variable comparisons (Broeg, Fernandez
                  & Neuhauser 2005), leave-one-out for the comparison stars
                  themselves.
  error bars      CCD equation plus the per-frame excess noise (scintillation,
                  thin cloud, seeing) measured from the comparison stars'
                  own scatter -- frame_excess_noise.
  screening       per-star: white and red noise, von Neumann eta (smooth
                  variability), and a single-event box search with a
                  simultaneous polynomial / airmass baseline -- the detection
                  statistic for one transit in one night.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from setiastro.saspro.exoplanet_transit import header_float, header_str, robust_std, binned_rms

# ── per-(star, frame) flag bits ──────────────────────────────────────────
FLAG_EDGE = 1        # aperture touches the frame edge or no-data (0.0) pixels
FLAG_SAT = 2         # a saturated / non-linear pixel inside the aperture
FLAG_BADFRAME = 4    # whole frame rejected (tracking failed, clouded out)
FLAG_OUTLIER = 8     # isolated outlier in the differential light curve
FLAG_INVALID = 16    # non-finite or non-positive flux
FLAG_SKY = 32        # too few usable sky pixels in the annulus

FLAG_NAMES = {
    FLAG_EDGE: "edge/no-data", FLAG_SAT: "saturated", FLAG_BADFRAME: "bad frame",
    FLAG_OUTLIER: "outlier", FLAG_INVALID: "invalid flux", FLAG_SKY: "bad sky",
}

# Aperture radii, in units of the median FWHM.  EXOTIC searches 1.5-6 Gaussian
# sigma (0.64-2.55 FWHM); this grid spans the same range.
APERTURE_MULTS = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


# ════════════════════════════ saturation ══════════════════════════════════

_UNIT_SCALE = {
    "8-bit": 255.0,
    "16-bit": 65535.0,
    "32-bit unsigned": 4294967295.0,
    "32-bit floating point": 1.0,
    "64-bit floating point": 1.0,
}


def saturation_level(img: np.ndarray, hdr, bit_depth) -> tuple:
    """
    (level in loaded-image units or None, description).

    Header first: MAXLIN (end of the linear range) is preferred to SATURATE
    when both exist.  Integer data were divided by the type maximum on load,
    float data are untouched -- LCO BANZAI frames are float electrons and
    their SATURATE/MAXLIN are in electrons too, so the scale is 1.

    Without a usable keyword, look for a plateau: several pixels sharing the
    frame maximum is what a clipped ADC looks like.

    Calibrated, flat-fielded, normalised or registered frames have neither:
    a clipped core sits at (full well - dark) / flat, which varies across
    the field with the flat.  For those, find the flat-topped star cores
    themselves (_core_plateau_level) and return a level just under the
    lowest of them.  This used to return None, leaving saturated stars free
    to become comparison stars -- on real flat-fielded data every comparison
    was saturated and almost every star in the field was flagged variable.
    """
    scale = _UNIT_SCALE.get(str(bit_depth or "").strip())
    lin = header_float(hdr, ("MAXLIN",))
    sat = header_float(hdr, ("SATURATE", "SATLEVEL", "DATASAT"))
    if scale is not None:
        if lin is not None and lin > 0:
            return 0.995 * lin / scale, "MAXLIN"
        if sat is not None and sat > 0:
            return 0.98 * sat / scale, "SATURATE"
    try:
        a = np.asarray(img)
        if a.ndim == 3:
            a = a.max(axis=2)
        m = float(np.nanmax(a))
        if not np.isfinite(m) or m <= 0:
            return None, "none"
        tol = max(abs(m) * 1e-6, 1e-12)
        n_at = int(np.count_nonzero(a >= m - tol))
        if n_at >= 5:
            return 0.98 * m, "plateau"
        lv = _core_plateau_level(a)
        if lv is not None:
            # 10% under the lowest clipped core: CMOS response is already
            # non-linear in the last few percent below full well.
            return 0.90 * lv, "flat-topped cores"
    except Exception:
        pass
    return None, "none"


def _core_plateau_level(a: np.ndarray, rel_tol: float = 0.003, n_min: int = 6):
    """
    Clipped level of saturated star cores in calibrated data, or None.

    A saturated core is flat-topped: its clipped pixels share one value
    ((full well - dark) / flat, and the flat barely changes across a core),
    so several pixels sit within rel_tol of the core maximum.  An unsaturated
    PSF is peaked -- at FWHM >= 3 px the pixels next to the peak are already
    percents below it -- so it can't pass.  Returns the 10th percentile of the
    clipped levels found (the lowest-lying cores, where the flat is
    brightest), or None when no core is flat-topped.
    """
    from scipy import ndimage
    v = a[np.isfinite(a)]
    if v.size < 100:
        return None
    step = max(1, v.size // 4_000_000)
    top = float(np.percentile(v[::step], 99.999))
    if not np.isfinite(top) or top <= 0:
        return None
    m = np.isfinite(a) & (a >= 0.5 * top)
    lab, n = ndimage.label(m)
    if n == 0 or n > 50_000:
        return None
    levels = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        vals = a[sl][lab[sl] == k]
        if vals.size < n_min:
            continue
        vmax = float(vals.max())
        if np.count_nonzero(vals >= vmax * (1.0 - rel_tol)) >= n_min:
            levels.append(vmax)
    if not levels:
        return None
    return float(np.percentile(levels, 10)) if len(levels) >= 5 else float(min(levels))


def saturated_coords_binned(img: np.ndarray, level: Optional[float],
                            bin_factor: int = 2, cap: int = 200000) -> np.ndarray:
    """(N,2) float32 [x, y] of saturated pixels on the binned grid."""
    if level is None or not np.isfinite(level):
        return np.zeros((0, 2), np.float32)
    a = np.asarray(img)
    m = a >= level
    if m.ndim == 3:
        m = m.any(axis=2)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return np.zeros((0, 2), np.float32)
    b = max(1, int(bin_factor))
    key = np.unique((ys // b).astype(np.int64) * 1_000_003 + (xs // b).astype(np.int64))
    if key.size > cap:
        key = key[:: int(math.ceil(key.size / cap))]
    yb = (key // 1_000_003).astype(np.float32)
    xb = (key % 1_000_003).astype(np.float32)
    return np.stack([xb, yb], axis=1)


def transform_coords(coords: np.ndarray, M) -> np.ndarray:
    """Apply a 2x3 affine (src -> dst) to (N,2) [x, y] coordinates."""
    if coords is None or len(coords) == 0 or M is None:
        return coords
    A = np.asarray(M, dtype=np.float64).reshape(2, 3)
    c = np.asarray(coords, dtype=np.float64)
    out = c @ A[:, :2].T + A[:, 2]
    return out.astype(np.float32)


# ═════════════════════════════ tracking ═══════════════════════════════════

def _vote_offset(ref_xy, xy, L, tol):
    """Most common offset d with xy ~= L @ ref + d.  Returns (d, votes)."""
    r = ref_xy @ np.asarray(L, dtype=np.float64).T
    d = (xy[None, :, :] - r[:, None, :]).reshape(-1, 2)
    # Bias the bin indices non-negative before packing them into one key:
    # floor-division decoding of a packed key is wrong for negative indices
    # (-1 // N == -1), which sent every frame with a negative offset
    # component to the astroalign fallback.
    BIAS, BASE = 1_000_000, 2_000_003
    q = np.floor(d / tol).astype(np.int64) + BIAS
    key = q[:, 0] * BASE + q[:, 1]
    uniq, cnt = np.unique(key, return_counts=True)
    if uniq.size == 0:
        return None, 0
    best = int(uniq[int(np.argmax(cnt))])
    qx = best // BASE - BIAS
    qy = best % BASE - BIAS
    centre = (np.array([qx, qy], dtype=np.float64) + 0.5) * tol
    near = np.hypot(d[:, 0] - centre[0], d[:, 1] - centre[1]) <= 1.5 * tol
    if not np.any(near):
        return None, 0
    dd = np.median(d[near], axis=0)
    near = np.hypot(d[:, 0] - dd[0], d[:, 1] - dd[1]) <= tol
    return np.median(d[near], axis=0), int(np.count_nonzero(near))


def _fit_affine(src, dst, n_iter: int = 3):
    """Least-squares affine src -> dst with 3-sigma clipping.  (M 2x3, rms, n)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    keep = np.ones(len(src), bool)
    M = None
    rms = float("inf")
    for _ in range(n_iter):
        if np.count_nonzero(keep) < 6:
            break
        A = np.column_stack([src[keep], np.ones(np.count_nonzero(keep))])
        sol, *_ = np.linalg.lstsq(A, dst[keep], rcond=None)
        M = sol.T
        pred = src @ M[:, :2].T + M[:, 2]
        res = np.hypot(*(pred - dst).T)
        rms = float(np.sqrt(np.mean(res[keep] ** 2)))
        s = max(1.4826 * float(np.median(res[keep])), 0.05)
        new = res <= max(3.0 * s, 0.3)
        if np.array_equal(new, keep):
            break
        keep = new
    return M, rms, int(np.count_nonzero(keep))


def estimate_transform(ref_xy: np.ndarray, frame_xy: np.ndarray, tol: float,
                       allow_astroalign: bool = True):
    """
    Affine (2x3) mapping REFERENCE pixel coords to FRAME pixel coords, from
    two brightness-sorted star lists.  Returns (M, info) or (None, reason).
    """
    ref_xy = np.asarray(ref_xy, dtype=np.float64)
    frame_xy = np.asarray(frame_xy, dtype=np.float64)
    if len(ref_xy) < 3 or len(frame_xy) < 3:
        return None, "too few stars"
    tol = max(float(tol), 1.0)
    nv = min(60, len(ref_xy)), min(60, len(frame_xy))
    min_votes = max(4, int(0.15 * min(nv)))

    from scipy.spatial import cKDTree
    tree = cKDTree(frame_xy)
    for L, tag in ((np.eye(2), "shift"), (-np.eye(2), "flip180")):
        d, votes = _vote_offset(ref_xy[:nv[0]], frame_xy[:nv[1]], L, tol)
        if d is None or votes < min_votes:
            continue
        pred = ref_xy @ L.T + d
        dist, j = tree.query(pred, k=1, distance_upper_bound=tol)
        ok = np.isfinite(dist)
        if np.count_nonzero(ok) >= 6:
            M, rms, n = _fit_affine(ref_xy[ok], frame_xy[j[ok]])
            if M is not None and n >= 6 and rms < max(1.0, 0.5 * tol):
                return M, f"{tag}+affine ({n} stars, rms {rms:.2f}px)"
        M = np.column_stack([L, d])
        return M, f"{tag} ({votes} votes)"

    if allow_astroalign:
        try:
            from setiastro.saspro import astroalign as _aa
            T, _pairs = _aa.find_transform(ref_xy[:80], frame_xy[:80],
                                           max_control_points=80)
            P = np.asarray(T.params, dtype=np.float64)
            M = P[:2, :]
            pred = ref_xy @ M[:, :2].T + M[:, 2]
            dist, j = tree.query(pred, k=1, distance_upper_bound=tol)
            ok = np.isfinite(dist)
            if np.count_nonzero(ok) >= 6:
                M2, rms, n = _fit_affine(ref_xy[ok], frame_xy[j[ok]])
                if M2 is not None and n >= 6:
                    return M2, f"astroalign+affine ({n} stars)"
            return M, "astroalign"
        except Exception as e:
            return None, f"no match ({type(e).__name__})"
    return None, "no match"


# ═══════════════════════════ per-frame measurement ════════════════════════

@dataclass
class ApertureSetup:
    radii: np.ndarray            # (n_ap,) pixels on the binned grid
    r_in: float
    r_out: float
    fwhm: float
    recenter: bool = True
    centroid_radius: float = 3.0
    max_shift: float = 3.0
    min_sky_pixels: int = 12


def aperture_setup(fwhm_px: float, mults: Sequence[float] = APERTURE_MULTS,
                   recenter: bool = True) -> ApertureSetup:
    f = float(max(fwhm_px, 1.2))
    radii = np.array(sorted({max(1.5, m * f) for m in mults}), dtype=np.float64)
    r_in = float(radii[-1] + max(2.0, 0.5 * f))
    r_out = float(r_in + max(5.0, 2.5 * f))
    return ApertureSetup(radii=radii, r_in=r_in, r_out=r_out, fwhm=f,
                         recenter=recenter,
                         centroid_radius=float(max(2.0, 1.2 * f)),
                         max_shift=float(max(1.5, 1.0 * f)))


def _row_nanmedian(V: np.ndarray):
    """
    Row-wise median ignoring NaN.  np.nanmedian(axis=1) runs a Python-level
    loop per row (apply_along_axis) and was 75% of the photometry time;
    np.sort puts NaN last, so the median is two take_along_axis reads.
    """
    S = np.sort(V, axis=1)
    n = np.sum(np.isfinite(V), axis=1)
    lo = np.clip((n - 1) // 2, 0, None)[:, None]
    hi = np.clip(n // 2, 0, None)[:, None]
    hi = np.minimum(hi, V.shape[1] - 1)
    med = 0.5 * (np.take_along_axis(S, lo, 1)[:, 0] + np.take_along_axis(S, hi, 1)[:, 0])
    return np.where(n > 0, med, np.nan), n


def _sigma_clipped_median(V: np.ndarray, n_iter: int = 4, k: float = 3.0):
    """Row-wise sigma-clipped median / robust sigma / count of a NaN-padded array."""
    V = V.copy()
    for _ in range(n_iter):
        med, n = _row_nanmedian(V)
        mad, _ = _row_nanmedian(np.abs(V - med[:, None]))
        s = 1.4826 * mad
        out = np.abs(V - med[:, None]) > k * s[:, None]
        out &= (s > 0)[:, None]
        if not np.any(out):
            break
        V[out] = np.nan
    med, n = _row_nanmedian(V)
    mad, _ = _row_nanmedian(np.abs(V - med[:, None]))
    return med, 1.4826 * mad, n


def measure_frame(plane: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                  ap: ApertureSetup, *, gain: Optional[float] = None,
                  sat_xy: Optional[np.ndarray] = None,
                  snr_recenter: float = 15.0) -> dict:
    """
    Aperture photometry of every star on one frame.

    plane: 2-D float image (NOT background-subtracted -- the annulus does
           that locally).  xs, ys: predicted positions on this frame.

    Returns arrays of length n_stars: x, y (used), sky, sky_sigma, n_sky,
    peak, flags, and flux/err of shape (n_ap, n_stars).
    """
    import sep

    plane = np.ascontiguousarray(plane, dtype=np.float32)
    H, W = plane.shape
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    S = xs.size
    n_ap = ap.radii.size
    flags = np.zeros(S, np.int16)

    R = int(math.ceil(ap.r_out)) + 1
    off = np.arange(-R, R + 1)
    cx = np.round(xs).astype(np.int64)
    cy = np.round(ys).astype(np.int64)
    gx = cx[:, None, None] + off[None, None, :]          # (S,1,P)
    gy = cy[:, None, None] + off[None, :, None]          # (S,P,1)
    inb = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)    # (S,P,P)
    pix = plane[np.clip(gy, 0, H - 1), np.clip(gx, 0, W - 1)].astype(np.float64)
    # 0.0 is SASpro's universal no-data value (registration borders, masked
    # regions); treat it like off-frame.
    nodata = (~inb) | (~np.isfinite(pix)) | (pix == 0.0)
    pix = np.where(nodata, np.nan, pix)

    # ---- sky: sigma-clipped annulus median around the predicted position ----
    dx = gx - xs[:, None, None]
    dy = gy - ys[:, None, None]
    d = np.sqrt(dx * dx + dy * dy)
    ann = (d >= ap.r_in) & (d < ap.r_out)
    V = np.where(ann, pix, np.nan).reshape(S, -1)
    sky, sky_sig, n_sky = _sigma_clipped_median(V)
    bad_sky = ~(np.isfinite(sky) & (n_sky >= ap.min_sky_pixels))
    sky = np.where(np.isfinite(sky), sky, 0.0)
    sky_sig = np.where(np.isfinite(sky_sig) & (sky_sig > 0), sky_sig,
                       np.nan)
    flags[bad_sky] |= FLAG_SKY

    # ---- recentring ------------------------------------------------------------
    x_use = xs.copy()
    y_use = ys.copy()
    if ap.recenter:
        net = np.nan_to_num(pix - sky[:, None, None], nan=0.0)
        xc, yc = xs.copy(), ys.copy()
        for _ in range(3):
            dc = np.hypot(gx - xc[:, None, None], gy - yc[:, None, None])
            w = np.where(dc <= ap.centroid_radius, np.clip(net, 0.0, None), 0.0)
            sw = w.sum(axis=(1, 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                nx = (w * gx).sum(axis=(1, 2)) / sw
                ny = (w * gy).sum(axis=(1, 2)) / sw
            okc = np.isfinite(nx) & np.isfinite(ny) & (sw > 0)
            xc = np.where(okc, nx, xc)
            yc = np.where(okc, ny, yc)
        # Accept only where the star is bright enough for its centroid to beat
        # the frame transform, and it hasn't wandered onto a neighbour.
        npx = np.pi * ap.centroid_radius ** 2
        noise = np.sqrt(np.nan_to_num(sky_sig, nan=np.inf) ** 2 * npx)
        dc = np.hypot(gx - xs[:, None, None], gy - ys[:, None, None])
        sig_c = np.where(dc <= ap.centroid_radius, net, 0.0).sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            snr_c = sig_c / noise
        shift = np.hypot(xc - xs, yc - ys)
        acc = (snr_c >= snr_recenter) & (shift <= ap.max_shift) & np.isfinite(shift)
        x_use = np.where(acc, xc, xs)
        y_use = np.where(acc, yc, ys)

    # ---- aperture sums (exact overlap) -----------------------------------------
    flux = np.full((n_ap, S), np.nan, np.float64)
    err = np.full((n_ap, S), np.nan, np.float64)
    xa = np.ascontiguousarray(x_use, dtype=np.float64)
    ya = np.ascontiguousarray(y_use, dtype=np.float64)
    var_sky = np.nan_to_num(sky_sig, nan=0.0) ** 2
    nsky = np.maximum(n_sky, 1).astype(np.float64)
    trunc = np.zeros(S, bool)
    for k, r in enumerate(ap.radii):
        try:
            s_, _e, fl = sep.sum_circle(plane, xa, ya, float(r), subpix=0)
        except Exception:
            continue
        area = math.pi * r * r
        net_f = s_ - sky * area
        var = area * var_sky * (1.0 + area / nsky)
        if gain:
            var = var + np.clip(net_f, 0.0, None) / float(gain)
        flux[k] = net_f
        err[k] = np.sqrt(var)
        trunc |= (np.asarray(fl) & 0x0010) != 0      # SEP_APER_TRUNC

    # ---- edge / no-data inside the largest aperture -----------------------------
    da = np.hypot(gx - x_use[:, None, None], gy - y_use[:, None, None])
    in_ap = da <= ap.radii[-1] + 0.5
    nd_in = np.sum(nodata & in_ap, axis=(1, 2))
    flags[(nd_in > 0) | trunc] |= FLAG_EDGE
    pk = np.where(in_ap & np.isfinite(pix), pix, -np.inf).reshape(S, -1).max(axis=1)
    peak = np.where(np.isfinite(pk), pk, np.nan)

    # ---- saturation ---------------------------------------------------------------
    if sat_xy is not None and len(sat_xy):
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(sat_xy, dtype=np.float64))
        cnt = tree.query_ball_point(np.column_stack([x_use, y_use]),
                                    r=float(ap.radii[-1]) + 1.0,
                                    return_length=True)
        flags[np.asarray(cnt) > 0] |= FLAG_SAT

    bad = ~np.isfinite(flux) | (flux <= 0)
    return {
        "x": x_use, "y": y_use, "sky": sky, "sky_sigma": sky_sig,
        "n_sky": n_sky, "peak": peak, "flags": flags,
        "flux": flux, "err": err, "nonpos": bad,
    }


@dataclass
class PhotometryCube:
    radii: np.ndarray                 # (n_ap,)
    flux: np.ndarray                  # (n_ap, n_stars, n_frames)
    err: np.ndarray                   # (n_ap, n_stars, n_frames)
    nonpos: np.ndarray                # (n_ap, n_stars, n_frames) bool
    flags: np.ndarray                 # (n_stars, n_frames) int16
    x: np.ndarray                     # (n_stars, n_frames)
    y: np.ndarray
    sky: np.ndarray
    peak: np.ndarray
    frame_ok: np.ndarray              # (n_frames,) bool -- tracking succeeded
    frame_info: list = field(default_factory=list)
    setup: ApertureSetup = None


def run_photometry(images: Sequence[np.ndarray], transforms: Sequence,
                   ref_xy: np.ndarray, ap: ApertureSetup, *,
                   gain: Optional[float] = None,
                   sat_coords: Optional[Sequence] = None,
                   n_workers: int = 4,
                   progress: Optional[Callable[[int, int], None]] = None,
                   executor=None, store: Optional[dict] = None,
                   slots: Optional[Sequence] = None) -> PhotometryCube:
    """
    Measure every reference star on every frame.  transforms[f] is the 2x3
    reference -> frame affine (None = tracking failed; the reference
    positions are used and the frame is flagged BADFRAME).

    With `executor` (a process pool) the frames are measured in worker
    processes: frame f is read from FrameStore `store` slot slots[f], or sent
    as an array when it has no slot.  sep.sum_circle holds the GIL, so a
    thread pool only overlaps the numpy parts; processes use every core.
    """
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    n_f = len(images)
    S = len(ref_xy)
    n_ap = ap.radii.size
    flux = np.full((n_ap, S, n_f), np.nan, np.float32)
    err = np.full((n_ap, S, n_f), np.nan, np.float32)
    nonpos = np.ones((n_ap, S, n_f), bool)
    flags = np.zeros((S, n_f), np.int16)
    X = np.full((S, n_f), np.nan, np.float32)
    Y = np.full((S, n_f), np.nan, np.float32)
    SKY = np.full((S, n_f), np.nan, np.float32)
    PK = np.full((S, n_f), np.nan, np.float32)
    frame_ok = np.ones(n_f, bool)
    ref_xy = np.asarray(ref_xy, dtype=np.float64)

    def _one(fi):
        img = images[fi]
        if img is None:
            return fi, None
        plane = img.mean(axis=2) if img.ndim == 3 else img
        M = transforms[fi]
        if M is None:
            pred = ref_xy
        else:
            M = np.asarray(M, dtype=np.float64).reshape(2, 3)
            pred = ref_xy @ M[:, :2].T + M[:, 2]
        sat = sat_coords[fi] if sat_coords is not None and fi < len(sat_coords) else None
        return fi, measure_frame(plane, pred[:, 0], pred[:, 1], ap,
                                 gain=gain, sat_xy=sat)

    def _results():
        if executor is None:
            with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
                yield from ex.map(_one, range(n_f))
            return
        from setiastro.saspro.exoplanet_frame_worker import photometry_task
        futs = {}
        for fi in range(n_f):
            img = images[fi]
            if img is None:
                yield fi, None
                continue
            slot = slots[fi] if slots is not None and fi < len(slots) else None
            sat = sat_coords[fi] if sat_coords is not None and fi < len(sat_coords) else None
            futs[executor.submit(photometry_task, (slot if slot is not None else fi),
                                 (store if slot is not None else None),
                                 (None if slot is not None else np.asarray(img)),
                                 transforms[fi], ref_xy, ap, gain, sat)] = fi
        pending = set(futs)
        while pending:
            fin, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for f in fin:
                yield futs[f], f.result()[1]
            if not fin:
                yield None, None          # heartbeat: lets the caller pump the GUI

    done = 0
    for fi, m in _results():
        if fi is None:
            if progress is not None:
                progress(done, n_f)
            continue
        done += 1
        if m is None:
            frame_ok[fi] = False
            flags[:, fi] |= FLAG_BADFRAME
        else:
            flux[:, :, fi] = m["flux"]
            err[:, :, fi] = m["err"]
            nonpos[:, :, fi] = m["nonpos"]
            flags[:, fi] = m["flags"]
            X[:, fi], Y[:, fi] = m["x"], m["y"]
            SKY[:, fi] = m["sky"]
            PK[:, fi] = m["peak"]
            if transforms[fi] is None:
                frame_ok[fi] = False
                flags[:, fi] |= FLAG_BADFRAME
        if progress is not None:
            progress(done, n_f)
    return PhotometryCube(radii=ap.radii.copy(), flux=flux, err=err,
                          nonpos=nonpos, flags=flags, x=X, y=Y, sky=SKY,
                          peak=PK, frame_ok=frame_ok, setup=ap)


# ═════════════════════════════ ensemble ═══════════════════════════════════

_BAD_FOR_ENSEMBLE = FLAG_EDGE | FLAG_SAT | FLAG_BADFRAME | FLAG_SKY | FLAG_INVALID


@dataclass
class Ensemble:
    comps: np.ndarray                  # selected comparison star indices
    weights: np.ndarray                # weight per comp (same order)
    sigma: dict                        # comp idx -> leave-one-out robust sigma
    rejected: list                     # comps rejected as variable / noisy
    artificial: np.ndarray             # (n_f,) normalised artificial comparison
    rel: np.ndarray                    # (n_s, n_f) relative flux, median 1
    rel_err: np.ndarray                # (n_s, n_f)
    valid: np.ndarray                  # (n_s, n_f) bool used to build rel
    n_comp_frame: np.ndarray           # (n_f,) comps contributing per frame


def build_ensemble(F: np.ndarray, E: np.ndarray, valid: np.ndarray, *,
                   max_comps: int = 10, exclude: Sequence[int] = (),
                   eligible: Optional[np.ndarray] = None,
                   min_valid_frac: float = 0.9, n_iter: int = 8) -> Ensemble:
    """
    Weighted artificial comparison star.

    Each comparison j is normalised to its own median (so weight, not
    brightness, decides its influence) and weighted 1/sigma_j^2, where sigma_j
    is the robust scatter of j against the ensemble of the OTHERS.  Comps whose
    scatter is far in excess of their photometric noise -- variables, blends,
    stars near a bad column -- are rejected and the weights recomputed until
    the set settles (Broeg et al. 2005).  The old code paired each star with
    the k stars of most similar median flux and took the plain median of
    their RAW fluxes, which lets one variable or one cosmic ray in a small
    group set the comparison level.

    Relative flux of a comp star is taken against the ensemble WITHOUT that
    star (leave-one-out), so a comp can never cancel its own variability.
    """
    F = np.asarray(F, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    v = np.asarray(valid, bool) & np.isfinite(F) & (F > 0)
    n_s, n_f = F.shape

    med = np.full(n_s, np.nan)
    for i in range(n_s):
        if np.count_nonzero(v[i]) >= 3:
            med[i] = np.median(F[i, v[i]])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = F / med[:, None]
        e = E / med[:, None]
    f = np.where(v, f, np.nan)
    e_med, _ = _row_nanmedian(np.where(v & np.isfinite(e) & (e > 0), e, np.nan))

    cand_mask = np.isfinite(med) & (med > 0)
    cand_mask &= v.mean(axis=1) >= min_valid_frac
    if eligible is not None:
        cand_mask &= np.asarray(eligible, bool)
    for j in exclude:
        if 0 <= int(j) < n_s:
            cand_mask[int(j)] = False
    cand = np.where(cand_mask)[0]
    if cand.size < 2:
        # Too strict for a sparse field: drop the coverage requirement.
        cand_mask = np.isfinite(med) & (med > 0) & (v.mean(axis=1) >= 0.5)
        for j in exclude:
            if 0 <= int(j) < n_s:
                cand_mask[int(j)] = False
        cand = np.where(cand_mask)[0]
    snr = np.where(np.isfinite(e_med) & (e_med > 0), 1.0 / e_med, 0.0)
    pool_n = max(3 * int(max_comps), 20)
    cand = cand[np.argsort(-snr[cand])][:pool_n]

    rejected = []
    if cand.size == 0:
        art = np.ones(n_f)
        return Ensemble(comps=np.array([], int), weights=np.array([]), sigma={},
                        rejected=[], artificial=art,
                        rel=np.where(v, f, np.nan), rel_err=np.where(v, e, np.nan),
                        valid=v, n_comp_frame=np.zeros(n_f, int))

    fz = np.nan_to_num(f[cand], nan=0.0)
    vz = v[cand].astype(np.float64)
    w = np.where(np.isfinite(e_med[cand]) & (e_med[cand] > 0),
                 1.0 / np.maximum(e_med[cand], 1e-6) ** 2, 1.0)
    active = np.ones(cand.size, bool)
    sig = np.full(cand.size, np.nan)

    def _loo_sigma(wv, act):
        num = (wv[act, None] * fz[act] * vz[act]).sum(axis=0)
        den = (wv[act, None] * vz[act]).sum(axis=0)
        out = np.full(cand.size, np.nan)
        for jj in range(cand.size):
            n_ = num - (wv[jj] * fz[jj] * vz[jj] if act[jj] else 0.0)
            d_ = den - (wv[jj] * vz[jj] if act[jj] else 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                A = n_ / d_
                r = f[cand[jj]] / A
            r = r[np.isfinite(r) & v[cand[jj]] & (d_ > 0)]
            if r.size >= 5:
                out[jj] = robust_std(r / np.median(r))
        return out

    for _ in range(int(n_iter)):
        if np.count_nonzero(active) < 2:
            break
        sig = _loo_sigma(w, active)
        good = np.isfinite(sig) & (sig > 0)
        w_new = np.where(good, 1.0 / np.maximum(sig, 1e-6) ** 2, 0.0)
        # excess scatter over the photometric error, brightness-independent
        rho = np.where(good, sig / np.maximum(e_med[cand], 1e-9), np.inf)
        a_rho = rho[active & good]
        changed = False
        if a_rho.size >= 4:
            lr = np.log(a_rho)
            cut = math.exp(float(np.median(lr)) + 3.0 * max(robust_std(lr), 0.15))
            s_cut = 5.0 * float(np.median(sig[active & good]))
            for jj in range(cand.size):
                if active[jj] and (not good[jj] or (rho[jj] > max(cut, 1.5))
                                   or sig[jj] > s_cut):
                    active[jj] = False
                    rejected.append(int(cand[jj]))
                    changed = True
        w = np.where(active, w_new, 0.0)
        if not changed and _ > 1:
            break

    order = np.argsort(-w)
    sel_local = [jj for jj in order if active[jj] and w[jj] > 0][: int(max_comps)]
    if len(sel_local) < 2:
        sel_local = [jj for jj in order if w[jj] > 0][: max(2, int(max_comps))]
    if not sel_local:
        sel_local = list(range(min(cand.size, int(max_comps))))
        w = np.ones(cand.size)
    sel_local = np.asarray(sel_local, int)
    act = np.zeros(cand.size, bool)
    act[sel_local] = True
    sig_final = _loo_sigma(np.where(act, w, 0.0), act)
    wf = np.where(act & np.isfinite(sig_final) & (sig_final > 0),
                  1.0 / np.maximum(sig_final, 1e-6) ** 2, 0.0)
    if not np.any(wf > 0):
        wf = np.where(act, 1.0, 0.0)

    comps = cand[sel_local]
    wc = wf[sel_local]
    fc = np.nan_to_num(f[comps], nan=0.0)
    vc = v[comps].astype(np.float64)
    ec = np.nan_to_num(e[comps], nan=0.0)
    num = (wc[:, None] * fc * vc).sum(axis=0)
    den = (wc[:, None] * vc).sum(axis=0)
    var_num = ((wc[:, None] ** 2) * (ec ** 2) * vc).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        art = num / den
        art_err = np.sqrt(var_num) / den
    art = np.where(den > 0, art, np.nan)
    n_comp_frame = vc.sum(axis=0).astype(int)

    rel = np.full((n_s, n_f), np.nan)
    rel_err = np.full((n_s, n_f), np.nan)
    comp_pos = {int(c): k for k, c in enumerate(comps)}
    for i in range(n_s):
        if not np.isfinite(med[i]):
            continue
        k = comp_pos.get(i)
        if k is None:
            A, Ae = art, art_err
        else:
            n_ = num - wc[k] * fc[k] * vc[k]
            d_ = den - wc[k] * vc[k]
            with np.errstate(invalid="ignore", divide="ignore"):
                A = np.where(d_ > 0, n_ / d_, np.nan)
                Ae = np.sqrt(np.clip(var_num - (wc[k] ** 2) * (ec[k] ** 2) * vc[k], 0, None)) / d_
        with np.errstate(invalid="ignore", divide="ignore"):
            r = f[i] / A
            re = np.abs(r) * np.sqrt((e[i] / f[i]) ** 2 + (Ae / A) ** 2)
        ok = v[i] & np.isfinite(r) & (r > 0)
        if np.count_nonzero(ok) >= 3:
            m_ = np.median(r[ok])
            r = r / m_
            re = re / m_
        rel[i] = np.where(ok, r, np.nan)
        rel_err[i] = np.where(ok, re, np.nan)

    sig_map = {int(cand[jj]): float(sig_final[jj]) for jj in sel_local}
    return Ensemble(comps=comps.astype(int), weights=wc, sigma=sig_map,
                    rejected=sorted(set(rejected)), artificial=art, rel=rel,
                    rel_err=rel_err, valid=v, n_comp_frame=n_comp_frame)


CHI2_1_MEDIAN = 0.45493642311957283     # median of a chi-square variable, 1 dof


def frame_excess_noise(F: np.ndarray, E: np.ndarray, valid: np.ndarray,
                       comps: Sequence[int], weights: Sequence[float], *,
                       t: Optional[np.ndarray] = None,
                       group: Optional[np.ndarray] = None,
                       pool: int = 2, min_comps: int = 3) -> np.ndarray:
    """
    Per-frame fractional noise beyond the CCD equation, measured from the
    comparison stars themselves: scintillation, thin cloud, seeing-driven
    aperture losses, flat-field error under drift, a missing Poisson term
    when the gain is unknown.  The CCD equation sees none of these; the
    comparison stars see all of them, frame by frame.

    Model: in frame j every star carries an extra fractional variance s_j^2,
    independent from star to star (a transparency change common to the whole
    field cancels in the differential and never shows up).  For each comp's
    leave-one-out curve r, the second difference d_j = r_j - (r_prev+r_next)/2
    removes any slow trend (colour-dependent extinction, a slow variable),
    and has variance
        V_ccd + 1.5 * s_j^2 * (1 + q),   q = sum w_k^2 / (sum w_k)^2
    over the OTHER comps (their averaged excess).  s_j^2 >= 0 is solved so
    that the median of d^2 / variance over the comps in a (2*pool+1)-frame
    window equals the chi-square(1) median -- a median, so one variable comp
    or one cosmic ray can't set it.  Windows never cross a `group` (night)
    boundary.  Frames with too few samples take their night's median.

    Returns s (n_f,) as a fraction; 0 = nothing beyond the CCD equation, or
    too few comps to tell.
    """
    F = np.asarray(F, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    n_f = F.shape[1]
    s_out = np.zeros(n_f)
    comps = np.asarray(comps, dtype=int)
    w = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(w) & (w > 0)
    comps, w = comps[keep], w[keep]
    nc = comps.size
    if nc < int(min_comps) or n_f < 5:
        return s_out
    v = np.asarray(valid, bool)[comps] & np.isfinite(F[comps]) & (F[comps] > 0)
    meds = np.array([np.median(F[c, v[k]]) if np.count_nonzero(v[k]) >= 3 else np.nan
                     for k, c in enumerate(comps)])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = F[comps] / meds[:, None]
        e = E[comps] / meds[:, None]
    v &= np.isfinite(f) & np.isfinite(meds)[:, None]
    fz = np.where(v, f, 0.0)
    ez = np.where(v & np.isfinite(e), e, 0.0)
    W = w[:, None] * v.astype(np.float64)
    num = (W * fz).sum(axis=0)
    den = W.sum(axis=0)
    var = (W ** 2 * ez ** 2).sum(axis=0)
    w2 = (W ** 2).sum(axis=0)
    d_i = den[None, :] - W
    with np.errstate(invalid="ignore", divide="ignore"):
        A = (num[None, :] - W * fz) / d_i
        r = f / A
        Ae2 = (var[None, :] - W ** 2 * ez ** 2) / d_i ** 2
        ve = (e / f) ** 2 + Ae2 / A ** 2
        q = (w2[None, :] - W ** 2) / d_i ** 2
    ok = v & (d_i > 0) & np.isfinite(r) & (r > 0) & np.isfinite(ve) & np.isfinite(q)
    for k in range(nc):
        if np.count_nonzero(ok[k]) >= 3:
            r[k] = r[k] / np.median(r[k, ok[k]])
        else:
            ok[k] = False

    if t is not None and len(t) == n_f and np.all(np.isfinite(np.asarray(t, dtype=np.float64))):
        order = np.argsort(np.asarray(t, dtype=np.float64), kind="stable")
    else:
        order = np.arange(n_f)
    g = (np.asarray(group)[order] if group is not None and len(group) == n_f
         else np.zeros(n_f, dtype=int))
    r, ve, q, ok = r[:, order], ve[:, order], q[:, order], ok[:, order]

    # previous / next valid frame of the same comp, in time order
    idx = np.arange(n_f)
    prev = np.maximum.accumulate(np.where(ok, idx[None, :], -1), axis=1)
    prev = np.concatenate([np.full((nc, 1), -1), prev[:, :-1]], axis=1)
    nxt = np.minimum.accumulate(np.where(ok, idx[None, :], n_f)[:, ::-1], axis=1)[:, ::-1]
    nxt = np.concatenate([nxt[:, 1:], np.full((nc, 1), n_f)], axis=1)
    pc = np.clip(prev, 0, n_f - 1)
    nn = np.clip(nxt, 0, n_f - 1)
    has = ok & (prev >= 0) & (nxt < n_f) & (g[pc] == g[None, :]) & (g[nn] == g[None, :])
    rows = np.arange(nc)[:, None]
    with np.errstate(invalid="ignore"):
        d = r - 0.5 * (r[rows, pc] + r[rows, nn])
        vd = ve + 0.25 * (ve[rows, pc] + ve[rows, nn])
    D2 = np.where(has, d * d, np.nan)
    VD = np.where(has, vd, np.nan)
    CQ = np.where(has, 1.5 * (1.0 + q), np.nan)

    # pool the comps over a short window of frames within the same night
    PD, PV, PC = [], [], []
    for o in range(-int(pool), int(pool) + 1):
        j = idx + o
        jc = np.clip(j, 0, n_f - 1)
        same = ((j >= 0) & (j < n_f) & (g[jc] == g))[None, :]
        PD.append(np.where(same, D2[:, jc], np.nan))
        PV.append(np.where(same, VD[:, jc], np.nan))
        PC.append(np.where(same, CQ[:, jc], np.nan))
    PD = np.concatenate(PD, axis=0).T
    PV = np.concatenate(PV, axis=0).T
    PC = np.concatenate(PC, axis=0).T
    n_ok = np.sum(np.isfinite(PD), axis=1)

    def _med_ratio(s2):
        with np.errstate(invalid="ignore", divide="ignore"):
            R = PD / (PV + PC * s2[:, None])
        return _row_nanmedian(R)[0]

    s2 = np.zeros(n_f)
    measured = n_ok >= 6
    m0 = _med_ratio(s2)
    need = measured & np.isfinite(m0) & (m0 > CHI2_1_MEDIAN)
    if np.any(need):
        with np.errstate(invalid="ignore", divide="ignore"):
            hi = np.nanmax(np.where(np.isfinite(PD), PD / PC, -np.inf), axis=1) / CHI2_1_MEDIAN
        hi = np.where(need & np.isfinite(hi) & (hi > 0), hi, 0.0)
        lo = np.zeros(n_f)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            above = _med_ratio(mid) > CHI2_1_MEDIAN
            lo = np.where(need & above, mid, lo)
            hi = np.where(need & ~above, mid, hi)
        s2 = np.where(need, 0.5 * (lo + hi), 0.0)
    if np.any(measured) and not np.all(measured):
        glob = float(np.median(s2[measured]))
        for lab in np.unique(g):
            m = g == lab
            mm = m & measured
            s2[m & ~measured] = float(np.median(s2[mm])) if np.any(mm) else glob
    s_out[order] = np.sqrt(np.clip(s2, 0.0, None))
    return s_out


def with_frame_noise(rel: np.ndarray, rel_err: np.ndarray, s: Optional[np.ndarray],
                     comps: Sequence[int], weights: Sequence[float],
                     valid: np.ndarray) -> np.ndarray:
    """
    Fold the per-frame excess noise s (frame_excess_noise) into relative-flux
    errors: var = var_ccd + rel^2 s^2 (1 + q).  The target carries its own
    s^2; the weighted comparison carries s^2 * q, q = sum w^2 / (sum w)^2 of
    the comps present in that frame (leave-one-out for a comp star itself).
    """
    out = np.asarray(rel_err, dtype=np.float64).copy()
    if s is None or len(comps) == 0:
        return out
    s2 = np.asarray(s, dtype=np.float64) ** 2
    if not np.any(s2 > 0):
        return out
    rel = np.asarray(rel, dtype=np.float64)
    comps = np.asarray(comps, dtype=int)
    W = np.asarray(weights, dtype=np.float64)[:, None] * np.asarray(valid, bool)[comps]
    den = W.sum(axis=0)
    w2 = (W ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(den > 0, w2 / den ** 2, 1.0)
        out = np.sqrt(out ** 2 + rel ** 2 * (s2 * (1.0 + q))[None, :])
        base = np.asarray(rel_err, dtype=np.float64)
        for k, c in enumerate(comps):
            d = den - W[k]
            qc = np.where(d > 0, (w2 - W[k] ** 2) / d ** 2, 1.0)
            out[c] = np.sqrt(base[c] ** 2 + rel[c] ** 2 * s2 * (1.0 + qc))
    return out


def target_lightcurve(F: np.ndarray, E: np.ndarray, valid: np.ndarray,
                      target: int, comps: Sequence[int], weights: Sequence[float],
                      frame_noise: Optional[np.ndarray] = None):
    """
    Relative flux of one target against a chosen weighted comparison set.
    Returns (rel, rel_err, ok) with rel normalised to its median.
    `frame_noise` (frame_excess_noise) adds the comp-measured per-frame
    excess to the CCD-equation errors.
    """
    comps = np.asarray(comps, int)
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(valid, bool) & np.isfinite(F) & (F > 0)
    meds = np.array([np.median(F[c, v[c]]) if np.count_nonzero(v[c]) >= 3 else np.nan
                     for c in comps])
    good = np.isfinite(meds) & (meds > 0) & (w > 0)
    comps, w, meds = comps[good], w[good], meds[good]
    n_f = F.shape[1]
    if comps.size == 0:
        return np.full(n_f, np.nan), np.full(n_f, np.nan), np.zeros(n_f, bool)
    fc = np.nan_to_num(F[comps] / meds[:, None], nan=0.0)
    ec = np.nan_to_num(E[comps] / meds[:, None], nan=0.0)
    vc = v[comps].astype(np.float64)
    num = (w[:, None] * fc * vc).sum(axis=0)
    den = (w[:, None] * vc).sum(axis=0)
    var_num = ((w[:, None] ** 2) * ec ** 2 * vc).sum(axis=0)
    # require most of the comparison weight present in each frame
    full_w = float(w.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        A = num / den
        Ae = np.sqrt(var_num) / den
        r = F[target] / A
        re = np.abs(r) * np.sqrt((E[target] / F[target]) ** 2 + (Ae / A) ** 2)
        if frame_noise is not None and len(frame_noise) == n_f:
            w2 = ((w[:, None] ** 2) * vc).sum(axis=0)
            q = np.where(den > 0, w2 / den ** 2, 1.0)
            s2 = np.asarray(frame_noise, dtype=np.float64) ** 2
            re = np.sqrt(re ** 2 + r ** 2 * s2 * (1.0 + q))
    ok = v[target] & (den >= 0.5 * full_w) & np.isfinite(r) & (r > 0)
    if np.count_nonzero(ok) >= 3:
        m = np.median(r[ok])
        r, re = r / m, re / m
    return np.where(ok, r, np.nan), np.where(ok, re, np.nan), ok


def outlier_flags(t: np.ndarray, rel: np.ndarray, ok: np.ndarray,
                  sigma: float = 5.0, group: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Isolated-outlier mask for one light curve: residual from the median of
    the two neighbours on each side (the point itself excluded).  A linear
    ramp -- ingress, egress, a steep eclipse branch -- has zero residual
    against that median, so only points that disagree with BOTH sides are
    flagged.  Cosmic rays, satellites in the aperture or annulus.

    `group` (e.g. the observing-night index) keeps neighbours inside their
    own night: across a night boundary the level can step, and a median
    straddling the step would flag the last and first points of each night.
    """
    out = np.zeros(rel.shape, bool)
    idx = np.where(ok & np.isfinite(rel))[0]
    if idx.size < 8:
        return out
    g = (np.zeros(rel.shape, dtype=np.int64) if group is None
         else np.asarray(group, dtype=np.int64))
    o = idx[np.lexsort((t[idx], g[idx]))]
    y = rel[o]
    go = g[o]
    n = y.size
    loc = np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - 2), min(n, i + 3)
        nb = np.concatenate([y[a:i][go[a:i] == go[i]], y[i + 1:b][go[i + 1:b] == go[i]]])
        if nb.size:
            loc[i] = np.median(nb)
    d = y - loc
    fin = np.isfinite(d)
    s = robust_std(d[fin])
    if not (np.isfinite(s) and s > 0):
        return out
    bad = fin & (np.abs(d - np.median(d[fin])) > sigma * s)
    out[o[bad]] = True
    return out


# ════════════════════════════ detrending ══════════════════════════════════

def segment_labels(session: Optional[np.ndarray], side: Optional[np.ndarray], n: int) -> np.ndarray:
    """Per-frame segment = (night, pier side), as one integer label."""
    s = np.zeros(n, dtype=np.int64) if session is None else np.asarray(session, dtype=np.int64)
    if side is None:
        return s
    p = np.asarray(side, dtype=np.int64)
    if p.shape != s.shape:
        return s
    return s * 2 + (p != 0)


def session_design(t: np.ndarray, mode: str, airmass: Optional[np.ndarray] = None,
                   session: Optional[np.ndarray] = None, side: Optional[np.ndarray] = None):
    """
    Baseline design matrix over one or more observing nights.

    Returns (B, n_offset).  The first n_offset columns are level indicators
    -- one per night, or one per (night, pier side) segment when `side` is
    given: a meridian flip puts every star on a different part of the sensor,
    so any flat-field or vignetting error turns into a per-star step at the
    flip (measured 8-20 ppt on flat-fielded data, 80-190 ppt without a flat).
    The rest describe the shape WITHIN nights:

      linear / quadratic  a slope (and curvature) per night, centred on that
                          night -- a time trend running across days means
                          nothing for systematics.
      airmass             ONE coefficient shared by every night: differential
                          extinction depends on the target-comparison colour
                          difference, which doesn't change from night to
                          night.  Falls back to per-night linear when the
                          frames carry no airmass variation.
      none                levels only.
    """
    t = np.asarray(t, dtype=np.float64)
    n = t.size
    s = np.zeros(n, dtype=np.int64) if session is None else np.asarray(session, dtype=np.int64)
    labels = np.unique(s)
    seg = segment_labels(s, side, n)
    cols = [(seg == lab).astype(np.float64) for lab in np.unique(seg)]
    n_off = len(cols)
    X = np.asarray(airmass, dtype=np.float64) if airmass is not None else None
    use_X = (mode == "airmass" and X is not None and X.shape == t.shape
             and np.all(np.isfinite(X)) and np.ptp(X) > 1e-4)
    if mode in ("linear", "quadratic") or (mode == "airmass" and not use_X):
        for lab in labels:
            m = s == lab
            if np.count_nonzero(m) < 4:
                continue
            tt = t[m]
            span = max(float(np.ptp(tt)), 1e-6)
            tau = np.zeros(n)
            tau[m] = (tt - tt.mean()) / span
            cols.append(tau)
            if mode == "quadratic" and np.count_nonzero(m) >= 6:
                q = np.zeros(n)
                q[m] = tau[m] ** 2 - np.mean(tau[m] ** 2)
                cols.append(q)
    elif use_X:
        cols.append(X - float(np.mean(X)))
    return np.column_stack(cols), n_off


def baseline_basis(t: np.ndarray, mode: str, airmass: Optional[np.ndarray] = None,
                   session: Optional[np.ndarray] = None, side: Optional[np.ndarray] = None):
    """Design matrix only -- see session_design."""
    return session_design(t, mode, airmass, session, side)[0]


def robust_trend(t: np.ndarray, y: np.ndarray, ok: np.ndarray, mode: str,
                 airmass: Optional[np.ndarray] = None,
                 session: Optional[np.ndarray] = None,
                 side: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Within-night trend to divide out, fitted jointly over every night.

    The model is  y = level[night] + shape  (session_design), fitted in TIME
    with iterative 3-sigma clipping -- the old code fitted against frame
    index, so a gap bent the polynomial.  What comes back is the shape
    relative to each night's level, (level + shape) / level, so dividing by
    it removes the within-night systematics but KEEPS the level of every
    night: for a long-period variable the change from night to night is the
    signal.  The levels are still fitted -- as nuisance terms -- so a variable
    brightening across nights can't drag the shared airmass coefficient.

    With `side` (pier side per frame) each (night, side) segment has its own
    level and the step between them is divided out too -- it is
    instrumental -- while the night's frame-weighted mean level is kept.

    A night whose trend would leave [0.5, 2] gets no correction (that is how
    a polynomial divided into a broken curve produced the "flux 0 to 3"
    U-shapes).
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.ones_like(y)
    two_sided = side is not None and np.unique(segment_labels(session, side, y.size)).size >         np.unique(segment_labels(session, None, y.size)).size
    if mode in (None, "none") and not two_sided:
        return out
    B, n_off = session_design(t, mode if mode not in (None, "none") else "none",
                              airmass, session, side)
    if B.shape[1] <= n_off and not two_sided:
        return out
    m = np.asarray(ok, bool) & np.isfinite(y) & np.all(np.isfinite(B), axis=1)
    if np.count_nonzero(m) < B.shape[1] + 5:
        return out
    use = m.copy()
    c = None
    for _ in range(6):
        c, *_ = np.linalg.lstsq(B[use], y[use], rcond=None)
        r = y - B @ c
        sd = robust_std(r[use])
        if not (np.isfinite(sd) and sd > 0):
            break
        new = m & (np.abs(r) <= 3.0 * sd)
        if np.array_equal(new, use) or np.count_nonzero(new) < B.shape[1] + 5:
            break
        use = new
    if c is None:
        return out
    level = B[:, :n_off] @ c[:n_off]
    s = np.zeros(y.size, dtype=np.int64) if session is None else np.asarray(session, dtype=np.int64)
    if two_sided:
        # the night's level = frame-weighted mean of its segments' levels
        night_level = level.copy()
        for lab in np.unique(s):
            k = (s == lab) & m
            if np.any(k):
                night_level[s == lab] = float(np.mean(level[k]))
        level = night_level
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (B @ c) / level
    good = np.isfinite(rel) & (level > 0)
    rel = np.where(good, rel, 1.0)
    for lab in np.unique(s):
        k = (s == lab) & m
        if np.any(k) and (np.any(rel[k] <= 0.5) or np.any(rel[k] >= 2.0)):
            rel[s == lab] = 1.0
    return rel


# ════════════════════════════ screening ═══════════════════════════════════

def von_neumann_eta(r: np.ndarray):
    """(eta, z) -- eta ~ 2 for white noise, << 2 for smooth variability."""
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 10:
        return float("nan"), 0.0
    s2 = np.var(r, ddof=1)
    if not (s2 > 0):
        return float("nan"), 0.0
    d2 = np.sum(np.diff(r) ** 2) / (n - 1)
    eta = float(d2 / s2)
    z = (2.0 - eta) / (2.0 / math.sqrt(n))
    return eta, float(z)


def box_search(t: np.ndarray, y: np.ndarray, B: np.ndarray, sigma_w: float,
               durations_d: Sequence[float], segment: Optional[np.ndarray] = None):
    """
    Single-event box search with a simultaneous linear baseline.

    For every trial (centre, duration) the model  y = B c + delta*box  is
    solved by weighted least squares.  All trials are solved at once from
    cumulative sums, so the cost is a batch of tiny (P+1)x(P+1) solves.
    Fitting the baseline together with the box is what stops a slope or an
    airmass curve from posing as a dip -- or from hiding one.

    SNR uses white noise plus the red-noise floor measured at the trial
    duration (Pont et al. 2006 "pink noise"):
        snr = depth / sqrt(var_white(delta) + sigma_r(D)^2)
    sigma_r(D) is measured on the residuals WITH the candidate box modelled
    out.  Measuring it on the plain baseline residuals lets a real transit
    inflate its own noise estimate: a clean 10 ppt dip in 2 ppt noise scored
    3.1 sigma that way instead of ~20.  So: best box per duration by white-
    noise SNR, then red noise from that model's residuals, then rank by the
    pink-noise SNR.

    `segment` (night / pier-side label per point, the groups B gives their
    own level): every segment a box touches must keep at least
    max(6, half its in-box count) points OUTSIDE the box.  A box that swallows
    a short segment is the same thing as that segment's level term, so its
    "depth" was measured against the one or two points left over -- on real
    data that produced 120-180 ppt "dips" at 8-12 sigma in flat light curves.
    Returns the best negative box: dict(depth, snr, t_center, duration, n_in).
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, P = B.shape
    best = {"depth": 0.0, "snr": 0.0, "t_center": float("nan"),
            "duration": float("nan"), "n_in": 0}
    if n < P + 8 or not (sigma_w > 0):
        return best
    cB = np.vstack([np.zeros((1, P)), np.cumsum(B, axis=0)])
    cy = np.concatenate([[0.0], np.cumsum(y)])
    G0 = B.T @ B
    r0 = B.T @ y
    seg_c = seg_n = None
    if segment is not None and np.shape(segment) == t.shape:
        _labs, _inv = np.unique(np.asarray(segment), return_inverse=True)
        if _labs.size > 1:
            ind = np.zeros((_labs.size, n))
            ind[_inv, np.arange(n)] = 1.0
            seg_c = np.hstack([np.zeros((_labs.size, 1)), np.cumsum(ind, axis=1)])
            seg_n = seg_c[:, -1]
    span = t[-1] - t[0]
    cad = float(np.median(np.diff(t))) if n > 1 else 0.0
    for D in durations_d:
        if D <= 0 or D > 0.75 * span or D < 2.5 * cad:
            continue
        step = max(cad, D / 10.0)
        centres = np.arange(t[0] + 0.25 * D, t[-1] - 0.25 * D + 1e-12, step)
        if centres.size == 0:
            continue
        i0 = np.searchsorted(t, centres - 0.5 * D, side="left")
        i1 = np.searchsorted(t, centres + 0.5 * D, side="left")
        nin = i1 - i0
        # At least half the box must hold data. Across multi-night sets the
        # centres also step through the daytime gaps, and a box hanging off
        # the end of a night would otherwise fit the night's edge as a "dip".
        need = max(3, int(math.ceil(0.5 * D / cad))) if cad > 0 else 3
        keep = (nin >= need) & ((n - nin) >= P + 3)
        if seg_c is not None and np.any(keep):
            s_in = seg_c[:, i1] - seg_c[:, i0]                  # (n_seg, K)
            s_out = seg_n[:, None] - s_in
            bad = (s_in > 0) & (s_out < np.maximum(6.0, 0.5 * s_in))
            keep &= ~np.any(bad, axis=0)
        if not np.any(keep):
            continue
        centres, i0, i1, nin = centres[keep], i0[keep], i1[keep], nin[keep]
        K = centres.size
        sB = cB[i1] - cB[i0]
        sy = cy[i1] - cy[i0]
        G = np.empty((K, P + 1, P + 1))
        G[:, :P, :P] = G0
        G[:, :P, P] = sB
        G[:, P, :P] = sB
        G[:, P, P] = nin
        rhs = np.empty((K, P + 1))
        rhs[:, :P] = r0
        rhs[:, P] = sy
        try:
            sol = np.linalg.solve(G, rhs[..., None])[..., 0]
            eP = np.zeros((K, P + 1, 1))
            eP[:, P, 0] = 1.0
            ginv_last = np.linalg.solve(G, eP)[:, P, 0]
        except np.linalg.LinAlgError:
            continue
        delta = sol[:, P]
        var_w = sigma_w ** 2 * np.clip(ginv_last, 0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            snr_w = -delta / np.sqrt(var_w)
        if not np.any(np.isfinite(snr_w)):
            continue
        k = int(np.nanargmax(snr_w))
        if not (np.isfinite(snr_w[k]) and snr_w[k] > 0):
            continue
        box = np.zeros(n)
        box[i0[k]:i1[k]] = 1.0
        resid = y - (B @ sol[k, :P] + delta[k] * box)
        sr2 = 0.0
        sb, nb = binned_rms(t, resid, D)
        if np.isfinite(sb) and nb > 0:
            sr2 = max(0.0, sb * sb - sigma_w ** 2 / nb)
        snr = float(-delta[k] / math.sqrt(var_w[k] + sr2))
        if snr > best["snr"]:
            best = {"depth": float(-delta[k]), "snr": snr,
                    "t_center": float(centres[k]), "duration": float(D),
                    "n_in": int(nin[k])}
    return best


def box_depth(t: np.ndarray, y: np.ndarray, box: np.ndarray, baseline: str = "linear",
              airmass: Optional[np.ndarray] = None,
              side: Optional[np.ndarray] = None) -> tuple:
    """
    Depth (fraction) and 1-sigma error of a FIXED box, fitted jointly with the
    same baseline box_search uses on one night.  For re-measuring a found dip
    on another aperture: a depth taken from medians instead disagrees with
    the box search whenever the baseline has a slope -- a box at the start
    of a night is then mostly trend.  (nan, nan) when it can't be fitted.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    box = np.asarray(box, dtype=bool)
    B, _ = session_design(t, baseline if baseline else "none", airmass, None, side)
    A = np.column_stack([B, box.astype(np.float64)])
    if A.shape[0] < A.shape[1] + 3 or box.sum() < 3 or (~box).sum() < 6:
        return float("nan"), float("nan")
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    s = robust_std(y - A @ c)
    g = float(np.linalg.pinv(A.T @ A)[-1, -1])
    return float(-c[-1]), float(s * math.sqrt(max(g, 0.0)))


DIP_DURATIONS_H = (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def screen_star(t_d: np.ndarray, rel: np.ndarray, err: np.ndarray, ok: np.ndarray, *,
                baseline: str = "linear", airmass: Optional[np.ndarray] = None,
                session: Optional[np.ndarray] = None,
                side: Optional[np.ndarray] = None,
                search_dip: bool = True) -> dict:
    """
    Detection statistics for one differential light curve.

    `session` (observing-night index per frame) gives every night its own
    level in the baseline -- also inside the dip search -- so the step
    between two nights is neither a "dip" nor "smooth variability".
    `search_dip=False` skips the box search (the caller searches each night
    on its own instead -- see ExoPlanetWindow._refresh_detrend_and_screen).

      rms_ppt       robust scatter about the baseline
      white_ppt     point-to-point (white) noise
      chi2_red      against the baseline, photometric errors
      eta, eta_z    von Neumann ratio and its significance (smooth variability)
      amp_ppt       5-95% range of the 10-minute-binned residual curve
      dip_*         best single box (transit-like dip), depth in ppt and SNR
    """
    out = {"n": 0, "rms_ppt": np.nan, "white_ppt": np.nan, "chi2_red": np.nan,
           "eta": np.nan, "eta_z": 0.0, "amp_ppt": 0.0,
           "dip_depth_ppt": 0.0, "dip_snr": 0.0, "dip_t": np.nan,
           "dip_dur_h": np.nan, "dip_n": 0}
    t = np.asarray(t_d, dtype=np.float64)
    y = np.asarray(rel, dtype=np.float64)
    e = np.asarray(err, dtype=np.float64) if err is not None else np.full_like(y, np.nan)
    X = np.asarray(airmass, dtype=np.float64) if airmass is not None else None
    m = np.asarray(ok, bool) & np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(m) < 12:
        return out
    o = np.where(m)[0]
    o = o[np.argsort(t[o])]
    t, y, e = t[o], y[o], e[o]
    Xo = X[o] if X is not None and X.shape == np.asarray(t_d).shape else None
    S = np.asarray(session) if session is not None else None
    So = S[o] if S is not None and S.shape == np.asarray(t_d).shape else None
    P = np.asarray(side) if side is not None else None
    Po = P[o] if P is not None and P.shape == np.asarray(t_d).shape else None
    out["n"] = int(t.size)

    B, n_off = session_design(t, baseline if baseline else "none", Xo, So, Po)
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    r = y - B @ c
    if Po is not None:
        # take the pier-side steps out of y for the night levels below
        seg_lv = B[:, :n_off] @ c[:n_off]
        s0 = So if So is not None else np.zeros(y.size, dtype=np.int64)
        y_lv = y.copy()
        for lab in np.unique(s0):
            k = s0 == lab
            y_lv[k] = y[k] - (seg_lv[k] - float(np.mean(seg_lv[k])))
    else:
        y_lv = y
    s_w = robust_std(np.diff(y)) / math.sqrt(2.0)
    if not (np.isfinite(s_w) and s_w > 0):
        s_w = robust_std(r)
    out["white_ppt"] = float(s_w * 1e3)
    out["rms_ppt"] = float(robust_std(r) * 1e3)
    ee = np.where(np.isfinite(e) & (e > 0), e, np.nan)
    if np.any(np.isfinite(ee)):
        ee = np.where(np.isfinite(ee), ee, np.nanmedian(ee))
        ee = np.sqrt(ee ** 2 + (1e-4) ** 2)          # 0.1 ppt floor
        out["chi2_red"] = float(np.sum((r / ee) ** 2) / max(1, t.size - B.shape[1]))
    eta, z = von_neumann_eta(r)
    out["eta"], out["eta_z"] = eta, z

    width = 10.0 / 1440.0
    if t.size > 1 and np.median(np.diff(t)) * 1440.0 > 5.0:
        width = 3.0 * float(np.median(np.diff(t)))
    idx = np.floor((t - t[0]) / width).astype(np.int64)
    uniq, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
    bm = np.bincount(inv, weights=r) / cnt
    bm = bm[cnt >= max(1, int(0.5 * np.median(cnt)))]
    if bm.size >= 4:
        out["amp_ppt"] = float((np.percentile(bm, 95) - np.percentile(bm, 5)) * 1e3)

    if search_dip:
        seg = segment_labels(So, Po, y.size) if (So is not None or Po is not None) else None
        best = box_search(t, y, B, s_w, [h / 24.0 for h in DIP_DURATIONS_H], segment=seg)
        out["dip_depth_ppt"] = float(best["depth"] * 1e3)
        out["dip_snr"] = float(best["snr"])
        out["dip_t"] = best["t_center"]
        out["dip_dur_h"] = best["duration"] * 24.0 if np.isfinite(best["duration"]) else np.nan
        out["dip_n"] = best["n_in"]

    # Night-to-night levels.  The baseline above gives every night its own
    # level, which is right for dips and within-night variability but hides
    # a long-period variable's change BETWEEN nights; these let the caller
    # test that against the comparison stars' own night-to-night scatter.
    if So is not None:
        lv, le = [], []
        for lab in np.unique(So):
            yy = y_lv[So == lab]
            if yy.size >= 5:
                lv.append(float(np.median(yy)))
                # standard error of a median
                le.append(float(1.2533 * robust_std(yy) / math.sqrt(yy.size)))
        if len(lv) >= 2:
            out["night_levels"] = lv
            out["night_err"] = le
            out["night_amp_ppt"] = float((max(lv) - min(lv)) * 1e3)
    return out


def night_level_floor(screens: dict, stars: Sequence[int]) -> float:
    """
    Instrumental night-to-night scatter, from the comparison stars' night
    levels (they're constant by selection): robust std of every comp's
    per-night level about its own mean.  0 when it can't be measured.
    """
    dev = []
    for i in stars:
        m = screens.get(int(i), {})
        lv = m.get("night_levels")
        if lv and len(lv) >= 2:
            a = np.asarray(lv, dtype=np.float64)
            dev.extend((a - a.mean()).tolist())
    if len(dev) < 4:
        return 0.0
    s = robust_std(np.asarray(dev))
    return float(s) if np.isfinite(s) else 0.0


def night_variability_z(m: dict, floor: float) -> float:
    """
    Significance that a star's night levels differ by more than their errors
    plus the instrumental floor: chi^2 of the levels about their weighted
    mean, as a z-score (chi^2 - dof) / sqrt(2 dof).
    """
    lv, le = m.get("night_levels"), m.get("night_err")
    if not lv or len(lv) < 2:
        return 0.0
    L = np.asarray(lv, dtype=np.float64)
    E = np.asarray(le, dtype=np.float64)
    w = 1.0 / np.maximum(E ** 2 + floor ** 2, 1e-12)
    mean = float(np.sum(w * L) / np.sum(w))
    chi2 = float(np.sum(w * (L - mean) ** 2))
    dof = L.size - 1
    return float((chi2 - dof) / math.sqrt(2.0 * dof))


def field_outlier_z(values, mags, half_width: float = 0.5, min_n: int = 30) -> np.ndarray:
    """
    How far each star's statistic stands above stars of similar brightness:
    (value - median) / (1.4826 MAD) over the stars within +/- half_width mag,
    the window widened to at least min_n stars (Sokolovsky et al. 2017, the
    VaST approach).  Field-wide systematics -- seeing-dependent aperture
    loss, flat-field error, a meridian flip -- raise every star's statistic
    together, so a fixed cut flags the whole field; this flags the stars that
    stand out from it.  NaN where it can't be judged (fewer than 10 stars).
    """
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(mags, dtype=np.float64)
    z = np.full(v.size, np.nan)
    ok = np.isfinite(v) & np.isfinite(m)
    idx = np.where(ok)[0]
    if idx.size < 10:
        return z
    order = idx[np.argsort(m[idx], kind="stable")]
    ms, vs = m[order], v[order]
    n = order.size
    need = min(int(min_n), n)
    for j in range(n):
        lo = int(np.searchsorted(ms, ms[j] - half_width, side="left"))
        hi = int(np.searchsorted(ms, ms[j] + half_width, side="right"))
        if hi - lo < need:
            lo = max(0, min(j - need // 2, n - need))
            hi = lo + need
        w = vs[lo:hi]
        med = float(np.median(w))
        s = 1.4826 * float(np.median(np.abs(w - med)))
        if not (np.isfinite(s) and s > 0):
            s = float(np.std(w)) or 1e-12
        z[order[j]] = (vs[j] - med) / s
    return z


def bin_curve(t: np.ndarray, y: np.ndarray, width: float, min_n: int = 2,
              e: Optional[np.ndarray] = None):
    """
    Fixed-width bins.  (tb, yb, eb).

    Without errors: mean and standard error of the mean.  With per-point
    errors `e`: inverse-variance weighted mean, error 1/sqrt(sum 1/e^2)
    scaled up by sqrt(chi2_red) of the bin when the points scatter more than
    their errors say (Birge ratio) -- never smaller than the propagated
    error, never smaller than what the bin's own scatter shows.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(y)
    if e is not None:
        e = np.asarray(e, dtype=np.float64)
        if e.shape != y.shape or not np.any(m & np.isfinite(e) & (e > 0)):
            e = None
    if e is not None:
        fill = float(np.median(e[m & np.isfinite(e) & (e > 0)]))
        e = np.where(np.isfinite(e) & (e > 0), e, fill)[m]
    t, y = t[m], y[m]
    if t.size < 2 or width <= 0:
        return np.array([]), np.array([]), np.array([])
    idx = np.floor((t - t.min()) / width).astype(np.int64)
    uniq, inv, cnt = np.unique(idx, return_inverse=True, return_counts=True)
    keep = cnt >= min_n
    if e is not None:
        wt = 1.0 / e ** 2
        sw = np.bincount(inv, weights=wt)
        mean = np.bincount(inv, weights=wt * y) / sw
        tb = np.bincount(inv, weights=t) / cnt
        chi2 = np.bincount(inv, weights=wt * (y - mean[inv]) ** 2)
        birge = np.where(cnt > 1, chi2 / np.maximum(cnt - 1, 1), 1.0)
        eb = np.sqrt(np.maximum(birge, 1.0) / sw)
        return tb[keep], mean[keep], eb[keep]
    sy = np.bincount(inv, weights=y)
    st = np.bincount(inv, weights=t)
    sy2 = np.bincount(inv, weights=y * y)
    mean = sy / cnt
    tb = st / cnt
    var = np.clip(sy2 / cnt - mean ** 2, 0, None) * cnt / np.maximum(cnt - 1, 1)
    eb = np.sqrt(var / cnt)
    return tb[keep], mean[keep], eb[keep]


# ═══════════════════════════ pipeline orchestration ═══════════════════════

def extract_catalog(plane: np.ndarray, thresh: float = 5.0, minarea: int = 9) -> dict:
    """
    SEP source catalogue of one frame, brightest first.  No-data (0.0) and
    non-finite pixels are masked out of both the background and detection.
    Returns dict(x, y, flux, a, b, fwhm (median of the brighter half), n).
    """
    import sep
    try:
        sep.set_extract_pixstack(3_000_000)
        sep.set_sub_object_limit(4096)
    except Exception:
        pass
    data = np.ascontiguousarray(plane, dtype=np.float32)
    mask = (~np.isfinite(data)) | (data == 0.0)
    if mask.any():
        data = np.where(mask, np.float32(0.0), data)
    empty = {"x": np.zeros(0), "y": np.zeros(0), "flux": np.zeros(0),
             "a": np.zeros(0), "b": np.zeros(0), "fwhm": float("nan"), "n": 0,
             "flag": np.zeros(0, int), "bkg_rms": float("nan")}
    try:
        bkg = sep.Background(data, mask=mask)
        sub = data - bkg.back()
        objs = sep.extract(sub, float(thresh), err=bkg.globalrms, mask=mask,
                           minarea=int(minarea), deblend_nthresh=32, clean=True)
    except Exception:
        return empty
    if objs is None or len(objs) == 0:
        return empty
    o = np.argsort(-np.asarray(objs["flux"], dtype=np.float64))
    objs = objs[o]
    a = np.clip(np.asarray(objs["a"], np.float64), 1e-3, None)
    b = np.clip(np.asarray(objs["b"], np.float64), 1e-3, None)
    fw = 2.3548 * np.sqrt(a * b)
    good = (np.asarray(objs["flag"]) == 0) & np.isfinite(fw)
    top = fw[good][: max(5, int(0.5 * np.count_nonzero(good)))]
    fwhm = float(np.median(top)) if top.size else float(np.median(fw))
    return {"x": np.asarray(objs["x"], np.float64), "y": np.asarray(objs["y"], np.float64),
            "flux": np.asarray(objs["flux"], np.float64), "a": a, "b": b,
            "fwhm": fwhm, "n": int(len(objs)), "flag": np.asarray(objs["flag"], int),
            "bkg_rms": float(bkg.globalrms)}


def compute_transforms(catalogs: Sequence[dict], ref_idx: int, tol: float):
    """Reference -> frame affine for every frame.  Returns (list[M|None], list[info])."""
    ref = catalogs[ref_idx]
    ref_xy = np.column_stack([ref["x"], ref["y"]])
    Ms, infos = [], []
    for i, c in enumerate(catalogs):
        if i == ref_idx:
            Ms.append(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
            infos.append("reference")
            continue
        if c is None or c.get("n", 0) < 3:
            Ms.append(None)
            infos.append("no stars detected")
            continue
        M, info = estimate_transform(ref_xy, np.column_stack([c["x"], c["y"]]), tol)
        Ms.append(M)
        infos.append(info)
    return Ms, infos


@dataclass
class DiffResult:
    ap_index: int
    rel: np.ndarray            # (n_s, n_f) relative flux, median 1 (NaN = unusable)
    rel_err: np.ndarray        # CCD equation + comp-measured per-frame excess
    flags: np.ndarray          # (n_s, n_f) int16, 0 = good
    ensemble: Ensemble
    transparency: np.ndarray   # (n_f,) artificial comparison (1 = clear)
    aperture_scores: list = field(default_factory=list)
    rel_err_phot: Optional[np.ndarray] = None   # CCD equation alone
    frame_noise: Optional[np.ndarray] = None    # (n_f,) excess fractional sigma


def _base_valid(cube: PhotometryCube, k: int, extra_bad: Optional[np.ndarray] = None):
    bad = (cube.flags & _BAD_FOR_ENSEMBLE) != 0
    bad |= cube.nonpos[k]
    if extra_bad is not None:
        bad |= extra_bad
    return ~bad


def differential_photometry(cube: PhotometryCube, *, max_comps: int = 10,
                            exclude: Sequence[int] = (), ap_index: Optional[int] = None,
                            times: Optional[np.ndarray] = None,
                            sessions: Optional[np.ndarray] = None,
                            outlier_sigma: float = 5.0) -> DiffResult:
    """
    Aperture choice + ensemble + frame/outlier flags for the whole field.

    Aperture: the radius that minimises the median leave-one-out scatter of
    the ensemble comparison stars -- one global radius for screening (the
    transit fit re-optimises it for its own target).
    """
    n_ap, n_s, n_f = cube.flux.shape
    sat_frac = ((cube.flags & FLAG_SAT) != 0).mean(axis=1)
    eligible = sat_frac <= 0.0          # a comp that saturates anywhere is out

    scores = []
    if ap_index is None:
        for k in range(n_ap):
            v = _base_valid(cube, k)
            ens = build_ensemble(cube.flux[k], cube.err[k], v, max_comps=max_comps,
                                 exclude=exclude, eligible=eligible)
            sig = [s for s in ens.sigma.values() if np.isfinite(s)]
            scores.append(float(np.median(sig)) if sig else float("inf"))
        ap_index = int(np.argmin(scores)) if scores and np.isfinite(np.min(scores)) \
            else int(min(3, n_ap - 1))
    k = int(ap_index)

    flags = cube.flags.copy()
    flags[cube.nonpos[k]] |= FLAG_INVALID

    v = _base_valid(cube, k)
    ens = build_ensemble(cube.flux[k], cube.err[k], v, max_comps=max_comps,
                         exclude=exclude, eligible=eligible)
    # Frames where the comparison light collapsed (cloud, dome, mis-track) or
    # most comps are unusable are rejected for every star.
    art = ens.artificial
    n_c = max(1, len(ens.comps))
    bad_frame = (~np.isfinite(art)) | (art < 0.25) | (ens.n_comp_frame < max(1, int(0.5 * n_c)))
    if np.any(bad_frame) and np.count_nonzero(~bad_frame) >= 10:
        flags[:, bad_frame] |= FLAG_BADFRAME
        v = _base_valid(cube, k, extra_bad=np.broadcast_to(bad_frame, v.shape))
        ens = build_ensemble(cube.flux[k], cube.err[k], v, max_comps=max_comps,
                             exclude=exclude, eligible=eligible)

    rel, rel_err_phot = ens.rel, ens.rel_err
    t = np.asarray(times, dtype=np.float64) if times is not None else np.arange(n_f, dtype=np.float64)
    # Error bars: CCD equation plus the per-frame excess the comparison stars
    # actually show (scintillation, thin cloud, seeing) -- see frame_excess_noise.
    s = frame_excess_noise(cube.flux[k], cube.err[k], v, ens.comps, ens.weights,
                           t=t, group=sessions)
    rel_err = with_frame_noise(rel, rel_err_phot, s, ens.comps, ens.weights, v)
    ok_all = np.isfinite(rel) & (flags == 0)
    for i in range(n_s):
        ol = outlier_flags(t, rel[i], ok_all[i], sigma=outlier_sigma, group=sessions)
        if np.any(ol):
            flags[i, ol] |= FLAG_OUTLIER
    return DiffResult(ap_index=k, rel=rel, rel_err=rel_err, flags=flags,
                      ensemble=ens, transparency=ens.artificial,
                      aperture_scores=scores, rel_err_phot=rel_err_phot,
                      frame_noise=s)
