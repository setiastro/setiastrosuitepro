# src/setiastro/saspro/aperture_flux_normalization.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Aperture Flux Normalization (AFN) -- SASpro's local normalization.

WHAT IT CORRECTS
----------------
After registration every frame sits on the reference grid, and at each pixel

    frame_i(x, y) = T_i(x, y) * S(x, y) + B_i(x, y)

S is the sky and the target, the same in every frame. T_i is how much of it
reached the sensor -- thin cloud, haze, the extinction gradient across a wide
field, dew, a flat that no longer quite matches -- and it MULTIPLIES. B_i is
light pollution, moon, airglow, scattered light, and it ADDS. Both vary across
the field and from frame to frame.

Measured on real data (403 registered L frames over three weeks, Newtonian):
39 of 40 sampled frames had star brightness ratios that varied across the
field far beyond the photometric noise, by a median of ~7% peak to peak and
up to 11%. A single global scale per frame cannot represent that.

HOW
---
The two fields are solved from DIFFERENT evidence, never jointly:

1. Stars give the multiplicative ratio R_i = T_i / T_set. Aperture photometry
   with a local sky annulus cancels B_i at each star, so the measured flux
   ratio sees only the dimming. Stars are the one thing in the frame whose
   true relative brightness is known: it is the same in every frame.

2. Background gives the additive difference, AFTER dividing by R_i. Once the
   dimming is taken out, frame_i / R_i and the set differ only by sky, so the
   target S cancels in the difference -- no need to find "pure background"
   around a galaxy or nebula, which is the hard part of ABE.

Fitting a gain and an offset jointly from the same pixels is degenerate on
real data: the solver trades one against the other, flattens frames to grey
and lifts the sky. Two fields from two independent kinds of evidence is what
keeps this well posed.

The reference is THE WHOLE SET: each star's median flux and each background
cell's median level across all frames of the group. No single frame's noise
or cloud is copied into the others.

How flexible each frame's maps are is decided per frame by cross-validation
on held-out stars (or held-out background cells), over a ladder that starts at
a constant. A clear frame lands on the constant, which is plain global
normalization. A frame with a cloud edge across it earns a flexible map
because that map predicts stars it never saw. Structure the held-out data
does not support is not added -- which is what keeps this from mottling. On
the real data above, the most flexible candidate predicted worse than no
correction at all, so the selection is doing real work.

0.0 IS NO-DATA
--------------
Registration leaves 0.0 borders. A star is used only if no 0.0 (or non-finite)
pixel falls inside its outer annulus; a background cell only from non-zero
samples; and when the correction is applied, 0.0 stays exactly 0.0 while a
corrected real pixel is floored just above zero, so it can never turn into
no-data.

COST
----
The correction is a small grid per frame, applied as each tile is read during
integration. No corrected frames are written, which on a write-bound drive is
most of the time a normalization phase would otherwise take. Measurements are
cached per group, so re-integrating with different rejection settings does
not measure again.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = [
    "AFNFrameModel",
    "build_models_for_group",
    "apply_tile_inplace",
    "AFN_VERSION",
]

AFN_VERSION = "afn-v2"   # v2: deterministic background object mask
# Versions the FITTED maps separately from the measurements, so a change in
# how maps are fitted refits from cached measurements instead of re-reading
# every frame. v2: reference balanced across meridian-flip orientations.
FIT_VERSION = "fit-v2"
# An orientation group helps define the reference only if it is a real
# share of the set; a lone flipped frame is corrected, it does not vote.
MIN_GROUP_FRAC = 0.10

# Photometry geometry, in units of the set's measured FWHM. Measured on real
# Newtonian data: at 1.5 x FWHM the flux-ratio map came out at ~11% peak to
# peak, inflated by coma and focus changes leaking light out of the aperture;
# at 2.5x and 4.0x it agreed at ~7%. 2.5x is the smallest aperture whose
# answer no longer depends on the star's shape, and it is quieter than 4.0x.
AP_FWHM = 2.5
ANN_IN_FWHM = 4.0
ANN_OUT_FWHM = 6.5

MAX_STARS = 800           # kernel smoothers cost ~stars^2 per frame
MIN_STARS_FLEX = 40       # fewer than this and only polynomials are tried
MIN_STARS_ANY = 12        # fewer than this and only a constant is fitted
ENSEMBLE_N = 5            # frames medianed to find star positions
BG_STRIDE = 4             # background cells sample every 4th pixel
GRID_TARGET = 128         # ~cells along the long axis
MIN_FRAMES = 3            # a median of fewer is not a reference
LOG_CLAMP = math.log(2.0)  # never scale a frame by more than 2x anywhere
FLOOR = 1e-7              # corrected real pixels never fall to 0.0 (no-data)
CV_FOLDS = 5


# ════════════════════════════════════════════════════════════════════════
#  Per-frame model and its application
# ════════════════════════════════════════════════════════════════════════

@dataclass
class AFNFrameModel:
    """out = in * gain - offset, both sampled on a coarse grid of cells.

    gain and offset are (C, gh, gw), evaluated at cell centres; a tile gets
    them by bilinear interpolation. The grid is the only thing kept per frame,
    whatever kind of model produced it.
    """
    cell: int
    gain: np.ndarray
    offset: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def is_identity(self) -> bool:
        return bool(np.all(self.gain == 1.0) and np.all(self.offset == 0.0))


_map_cache: dict = {}
_map_lock = threading.Lock()


def _tile_maps(y0: int, y1: int, x0: int, x1: int, cell: int):
    """Grid coordinates of every pixel centre in a tile, shared by all frames.

    Every frame's grid has the same geometry, so the remap coordinates for a
    tile are computed once and reused N times. Pixel centre x maps to grid
    coordinate (x + 0.5) / cell - 0.5, which puts cell centres on integers.
    """
    key = (y0, y1, x0, x1, cell)
    with _map_lock:
        hit = _map_cache.get(key)
        if hit is not None:
            return hit
    xs = (np.arange(x0, x1, dtype=np.float32) + 0.5) / cell - 0.5
    ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) / cell - 0.5
    mx = np.ascontiguousarray(np.broadcast_to(xs[None, :], (y1 - y0, x1 - x0)))
    my = np.ascontiguousarray(np.broadcast_to(ys[:, None], (y1 - y0, x1 - x0)))
    with _map_lock:
        if len(_map_cache) > 6:          # only a few tiles are ever in flight
            _map_cache.clear()
        _map_cache[key] = (mx, my)
    return mx, my


def apply_tile_inplace(view: np.ndarray, model: Optional[AFNFrameModel],
                       y0: int, y1: int, x0: int, x1: int) -> None:
    """Correct one frame's tile in place. `view` is (th, tw, C) float32.

    0.0 stays exactly 0.0 (no-data), NaN stays NaN, and a real pixel the
    correction would push to or below zero is floored at FLOOR instead, so
    the integration never mistakes it for a registration border.
    """
    if model is None or model.info.get("identity"):
        return
    import cv2
    th, tw = y1 - y0, x1 - x0
    if th <= 0 or tw <= 0:
        return
    mx, my = _tile_maps(y0, y1, x0, x1, int(model.cell))
    C = view.shape[-1] if view.ndim == 3 else 1
    for c in range(C):
        gc = model.gain[min(c, model.gain.shape[0] - 1)]
        oc = model.offset[min(c, model.offset.shape[0] - 1)]
        g = cv2.remap(gc, mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        t = view[..., c] if view.ndim == 3 else view
        nodata = (t == 0.0)
        np.multiply(t, g, out=t)
        if model.info.get("has_offset", True):
            o = cv2.remap(oc, mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
            np.subtract(t, o, out=t)
        np.maximum(t, np.float32(FLOOR), out=t)     # NaN propagates unchanged
        t[nodata] = 0.0


# ════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ════════════════════════════════════════════════════════════════════════

def _grid_dims(H: int, W: int):
    cell = int(round(max(H, W) / float(GRID_TARGET)))
    cell = max(32, cell)
    cell = int(math.ceil(cell / BG_STRIDE) * BG_STRIDE)   # whole strides per cell
    gh = max(1, H // cell)
    gw = max(1, W // cell)
    return cell, gh, gw


def _uv(x, y, W, H):
    return 2.0 * np.asarray(x, float) / max(W - 1, 1) - 1.0, \
        2.0 * np.asarray(y, float) / max(H - 1, 1) - 1.0


def _grid_uv(cell, gh, gw, W, H):
    gx = (np.arange(gw) + 0.5) * cell
    gy = (np.arange(gh) + 0.5) * cell
    GX, GY = np.meshgrid(gx, gy)
    u, v = _uv(GX.ravel(), GY.ravel(), W, H)
    return u, v


def _fast_nanmedian(a, axis=-1):
    """Exactly np.nanmedian along `axis`, without numpy's slow NaN path.

    np.nanmedian falls back to a per-slice Python path once NaNs are present;
    on real 8288x5644 frames it cost 171 ms per frame for the background
    cells and 15 s for the 5-frame star-detection median -- more than reading
    the frames. Here NaN (and +/-inf) sort to the end as +inf, and the median
    is read off by each slice's count of real values. Same rule as numpy:
    odd count -> middle value, even -> mean of the two middles; an empty
    slice gives NaN.
    """
    a = np.moveaxis(np.asarray(a), axis, -1)
    v = np.where(np.isfinite(a), a, np.inf)
    v.sort(axis=-1)
    n = np.isfinite(v).sum(-1)
    lo = np.maximum((n - 1) // 2, 0)[..., None]
    hi = np.maximum(n // 2, 0)[..., None]
    if n.ndim == 0:
        lo, hi = lo.reshape(1), hi.reshape(1)
    vlo = np.take_along_axis(v, lo, axis=-1)[..., 0]
    vhi = np.take_along_axis(v, hi, axis=-1)[..., 0]
    med = (vlo + vhi) / 2
    return np.where(n > 0, med, np.nan).astype(v.dtype, copy=False)


def _rsig(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


# ════════════════════════════════════════════════════════════════════════
#  Star detection on the ensemble
# ════════════════════════════════════════════════════════════════════════

@dataclass
class _Stars:
    x: np.ndarray
    y: np.ndarray
    fwhm: float
    r_ap: float
    r_in: float
    r_out: float
    H: int
    W: int
    C: int
    cell: int
    gh: int
    gw: int
    bgmask_s: np.ndarray       # strided background mask, True = object/star


def _sep_extract_seg(sep, sub, thresh, err, mask):
    from setiastro.saspro.sep_tiled import (
        PIXSTACK_FIRST, ensure_pixstack, pixstack_full)
    ensure_pixstack(sep, PIXSTACK_FIRST)
    try:
        return sep.extract(sub, thresh, err=err, mask=mask, minarea=9,
                           segmentation_map=True)
    except Exception as e:                           # noqa: BLE001
        if not pixstack_full(e):
            raise
        ensure_pixstack(sep, int(sub.size))
        return sep.extract(sub, thresh, err=err, mask=mask, minarea=9,
                           segmentation_map=True)


def _detect_stars(lum: np.ndarray, valid: np.ndarray, C: int, log) -> Optional[_Stars]:
    import cv2
    import sep
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    H, W = lum.shape
    data = np.ascontiguousarray(lum, dtype=np.float32)
    mask = ~valid
    bkg = sep.Background(data, mask=mask, bw=128, bh=128)
    sub = data - bkg.back()
    sub[mask] = 0.0
    rms = float(bkg.globalrms)
    cat, seg = _sep_extract_seg(sep, sub, 5.0, rms, mask)
    if len(cat) == 0:
        log("⚠️ AFN: no sources detected on the set's median frame.")
        return None

    fw_all = 2.3548 * np.sqrt(np.clip(cat["a"] * cat["b"], 1e-6, None))
    strong = (cat["flag"] == 0) & (cat["peak"] > 15.0 * rms)
    if strong.sum() < 5:
        log("⚠️ AFN: too few clean stars to measure the FWHM.")
        return None
    fwhm = float(np.clip(np.median(fw_all[strong]), 1.0, 30.0))
    r_ap, r_in, r_out = AP_FWHM * fwhm, ANN_IN_FWHM * fwhm, ANN_OUT_FWHM * fwhm

    good = strong.copy()
    good &= cat["peak"] < 0.6 * float(np.max(cat["peak"]))          # unsaturated
    good &= (fw_all > 0.6 * fwhm) & (fw_all < 2.0 * fwhm)            # not galaxies / hot px
    good &= (cat["b"] / np.maximum(cat["a"], 1e-6)) > 0.5            # not blends
    xy = np.c_[cat["x"], cat["y"]]
    if len(xy) > 1:
        nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
        good &= nn > (r_out + fwhm)                                  # isolated
    m = int(math.ceil(r_out + 2))
    good &= (cat["x"] > m) & (cat["x"] < W - m - 1) & \
            (cat["y"] > m) & (cat["y"] < H - m - 1)
    ver = binary_erosion(valid, iterations=m)
    xi = np.clip(np.round(cat["x"]).astype(int), 0, W - 1)
    yi = np.clip(np.round(cat["y"]).astype(int), 0, H - 1)
    good &= ver[yi, xi]

    idx = np.flatnonzero(good)
    if idx.size > MAX_STARS:                        # keep the brightest usable
        idx = idx[np.argsort(cat["flux"][idx])[::-1][:MAX_STARS]]
    if idx.size < MIN_STARS_ANY:
        log(f"⚠️ AFN: only {idx.size} usable photometry stars; "
            f"need at least {MIN_STARS_ANY}.")
        return None

    # Background mask: everything above the 5-sigma detection threshold,
    # grown by an aperture radius so star wings and halos stay out of the
    # background cells.
    #
    # A plain threshold, NOT SEP's segmentation map. SEP's deblending is not
    # bit-repeatable call to call: two identical runs on the same frames
    # disagreed on 72 of 241,032 background cells (by up to 1.7e-4), which
    # changes the fitted maps. The same frames must always give the same
    # maps -- for the cache, and for anyone comparing two stacks. Above-
    # threshold pixels are the region the segmentation covers anyway.
    del seg
    objmask = ((sub > 5.0 * rms) & valid).astype(np.uint8)
    rr = max(1, int(round(r_ap)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))
    objmask = cv2.dilate(objmask, k) > 0
    cell, gh, gw = _grid_dims(H, W)
    bgmask_s = objmask[::BG_STRIDE, ::BG_STRIDE]
    bgmask_s = np.ascontiguousarray(
        bgmask_s[: gh * cell // BG_STRIDE, : gw * cell // BG_STRIDE])

    log(f"🔆 AFN: {idx.size} photometry stars (of {len(cat)} detections), "
        f"FWHM {fwhm:.2f}px, aperture r={r_ap:.1f}px, "
        f"annulus {r_in:.1f}-{r_out:.1f}px, grid {gw}x{gh} @ {cell}px")
    return _Stars(x=cat["x"][idx].astype(np.float64),
                  y=cat["y"][idx].astype(np.float64),
                  fwhm=fwhm, r_ap=r_ap, r_in=r_in, r_out=r_out,
                  H=H, W=W, C=C, cell=cell, gh=gh, gw=gw, bgmask_s=bgmask_s)


# ════════════════════════════════════════════════════════════════════════
#  Per-frame measurement
# ════════════════════════════════════════════════════════════════════════

def _circle_offsets(r: float):
    R = int(math.ceil(r))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    return R, (xx * xx + yy * yy) <= (r * r)


def _measure_frame(frame: np.ndarray, st: _Stars):
    """Star fluxes (C, K) + usable flags, and background cells (C, gh, gw)."""
    import sep
    H, W, C = frame.shape
    K = st.x.size
    flux = np.zeros((C, K), np.float32)
    ok = np.zeros((C, K), bool)
    bg = np.full((C, st.gh, st.gw), np.nan, np.float32)
    R, disk = _circle_offsets(st.r_out)
    xi = np.round(st.x).astype(int)
    yi = np.round(st.y).astype(int)
    cs = st.cell // BG_STRIDE

    for c in range(C):
        d = np.ascontiguousarray(frame[..., c], dtype=np.float32)

        # A star counts only if no no-data (0.0) or non-finite pixel falls
        # inside its outer annulus. Checked on a box per star rather than a
        # full-frame mask: the borders are large and contiguous, and this is
        # a few tens of milliseconds instead of hundreds of MB.
        clean = np.ones(K, bool)
        for k in range(K):
            box = d[yi[k] - R: yi[k] + R + 1, xi[k] - R: xi[k] + R + 1]
            if box.shape != disk.shape:
                clean[k] = False
                continue
            vals = box[disk]
            if not np.all(np.isfinite(vals)) or np.any(vals == 0.0):
                clean[k] = False

        f, _fe, fl = sep.sum_circle(d, st.x, st.y, st.r_ap,
                                    bkgann=(st.r_in, st.r_out), subpix=5)
        good = clean & (fl == 0) & np.isfinite(f) & (f > 0)
        flux[c] = np.where(good, f, 0.0)
        ok[c] = good

        # Background cells: every BG_STRIDE-th pixel, objects and no-data out.
        s = d[::BG_STRIDE, ::BG_STRIDE][: st.gh * cs, : st.gw * cs]
        use = (s != 0.0) & np.isfinite(s) & ~st.bgmask_s
        vals = np.where(use, s, np.nan).reshape(st.gh, cs, st.gw, cs)
        vals = vals.transpose(0, 2, 1, 3).reshape(st.gh, st.gw, cs * cs)
        cnt = np.isfinite(vals).sum(-1)
        # _fast_nanmedian, not np.nanmedian: this was 171 ms of a ~390 ms
        # frame. It also needs no warnings filter (an empty cell just comes
        # back NaN), which matters here -- this runs in worker threads, and
        # warnings.catch_warnings() swaps a process-global filter list.
        med = _fast_nanmedian(vals, axis=-1)
        med[cnt < max(20, (cs * cs) // 8)] = np.nan
        bg[c] = med.astype(np.float32)

    return flux, ok, bg


# ════════════════════════════════════════════════════════════════════════
#  Model ladders and cross-validated selection
# ════════════════════════════════════════════════════════════════════════

def _poly_design(u, v, order):
    cols = [np.ones_like(u)]
    for d in range(1, order + 1):
        for i in range(d + 1):
            cols.append(u ** (d - i) * v ** i)
    return np.stack(cols, 1)


def _fit_poly(u, v, z, w, order):
    """Robust weighted polynomial; returns a predictor (uq, vq) -> values."""
    A = _poly_design(u, v, order)
    sw = np.sqrt(np.maximum(w, 0.0))
    keep = np.ones(z.size, bool)
    coef = np.zeros(A.shape[1])
    coef[0] = float(np.median(z)) if z.size else 0.0
    for _ in range(4):
        if keep.sum() < A.shape[1] + 1:
            break
        coef, *_ = np.linalg.lstsq(A[keep] * sw[keep, None], z[keep] * sw[keep],
                                   rcond=None)
        r = (z - A @ coef) * sw
        s = _rsig(r[keep])
        if s <= 0:
            break
        nk = np.abs(r) < 3.0 * s
        if nk.sum() == keep.sum():
            break
        keep = nk
    c = coef.copy()
    return lambda uq, vq: _poly_design(np.asarray(uq, float),
                                       np.asarray(vq, float), order) @ c


def _kernel_eval(uq, vq, u, v, z, w, h):
    out = np.empty(np.size(uq))
    uq = np.asarray(uq, float).ravel()
    vq = np.asarray(vq, float).ravel()
    inv = -0.5 / (h * h)
    for a in range(0, uq.size, 2048):
        du = uq[a:a + 2048, None] - u[None, :]
        dv = vq[a:a + 2048, None] - v[None, :]
        kw = np.exp((du * du + dv * dv) * inv) * w[None, :]
        out[a:a + 2048] = (kw @ z) / np.maximum(kw.sum(1), 1e-300)
    return out


def _fit_kernel(u, v, z, w, h):
    """Robust Gaussian-weighted local mean with bandwidth h (in u-v units)."""
    keep = np.ones(z.size, bool)
    sw = np.sqrt(np.maximum(w, 0.0))
    for _ in range(3):
        r = (z - _kernel_eval(u, v, u[keep], v[keep], z[keep], w[keep], h)) * sw
        s = _rsig(r[keep])
        if s <= 0:
            break
        nk = np.abs(r) < 3.0 * s
        if nk.sum() == keep.sum():
            break
        keep = nk
    uk, vk, zk, wk = u[keep], v[keep], z[keep], w[keep]
    return lambda uq, vq: _kernel_eval(uq, vq, uk, vk, zk, wk, h)


# Ladder order IS the simplicity order the one-standard-error rule uses.
def _star_ladder(n: int):
    lad = [("constant", lambda u, v, z, w: _fit_poly(u, v, z, w, 0))]
    if n >= MIN_STARS_ANY:
        lad.append(("plane", lambda u, v, z, w: _fit_poly(u, v, z, w, 1)))
    if n >= 24:
        lad.append(("quadratic", lambda u, v, z, w: _fit_poly(u, v, z, w, 2)))
    if n >= MIN_STARS_FLEX:
        lad.append(("smooth 0.60", lambda u, v, z, w: _fit_kernel(u, v, z, w, 0.60)))
        lad.append(("cubic", lambda u, v, z, w: _fit_poly(u, v, z, w, 3)))
        lad.append(("smooth 0.35", lambda u, v, z, w: _fit_kernel(u, v, z, w, 0.35)))
    if n >= 120:
        lad.append(("smooth 0.20", lambda u, v, z, w: _fit_kernel(u, v, z, w, 0.20)))
    if n >= 300:
        lad.append(("smooth 0.12", lambda u, v, z, w: _fit_kernel(u, v, z, w, 0.12)))
    return lad


# Held-out loss is the squared standardized residual, capped at WINSOR sigma.
#
# NOT a median/MAD. A score that is robust to the minority is blind to
# exactly the case local normalization exists for: stars under a passing
# cloud ARE a minority of the field, so a model that ignores the cloud scores
# almost as well on the MAD as one that captures it. On synthetic data,
# MAD-scored selection recovered 5% of a 12% cloud dip. The cap keeps a truly
# wild star (variable, cosmic ray, satellite) from deciding anything -- and
# because candidates are compared PAIRED per star (_pick_paired), what every
# model misses equally cancels outright.
WINSOR = 5.0


def _losses_from_resid(R):
    """Standardized out-of-fold residuals (n_cand, n) -> capped losses.

    The cap is relative to the residual scale the data can actually reach,
    not to the photometric noise alone. When noise dominates (real data:
    ~2.5% per star) the typical residual is ~1 sigma and nothing changes.
    When model misfit dominates -- very quiet stars, or structure no
    candidate captures exactly -- even the best candidate sits at several
    sigma, half of all residuals hit a fixed 5-sigma cap, their losses
    saturate, and the comparison washes out: a quadratic three times worse
    under a cloud was judged "not significantly worse" that way. Scaling by
    the best candidate's typical residual (a MAD, so the cloud minority does
    not set it) keeps the cap meaning "wild", not "ordinary".
    """
    R = np.asarray(R, float)
    s = max(1.0, min(_rsig(R[c]) for c in range(R.shape[0])))
    t = R / s
    return np.minimum(t * t, WINSOR * WINSOR)


def _pick_paired(losses):
    """Simplest candidate not significantly worse than the best, PAIRED.

    losses: (n_candidates, n_samples) out-of-fold losses, every candidate
    scored on the same held-out samples. The comparison is on per-sample
    DIFFERENCES from the best candidate, so anything every model misses
    equally -- a variable star, a cosmic ray in an aperture, a satellite
    across a sky cell -- cancels instead of drowning the comparison.

    Comparing fold-level scores instead (the usual one-standard-error rule)
    failed on real data: a few such outliers land unevenly across folds, the
    fold-to-fold scatter balloons, and every model looks within noise of the
    constant -- 282 of 403 real frames were sent to "constant" that way,
    against strong spatial structure measured in the same frames.
    Returns (index, cv_gain) where cv_gain is the held-out RMS reduction of
    the pick versus the first (simplest) candidate.
    """
    L = np.asarray(losses, float)
    means = L.mean(axis=1)
    b = int(np.argmin(means))
    n = L.shape[1]
    pick = b
    for c in range(L.shape[0]):
        d = L[c] - L[b]
        se = float(d.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
        if d.mean() <= se:
            pick = c
            break
    gain = 1.0 - math.sqrt(max(means[pick], 0.0) / max(means[0], 1e-300))
    return pick, gain


def _cv_choose(u, v, z, w, ladder, seed):
    """Cross-validated model choice over the ladder, decided per star, paired.

    `w` must be inverse variances, so residual * sqrt(w) is in sigma units.
    Each star is held out exactly once; every candidate predicts it from the
    same other stars.
    """
    if len(ladder) == 1 or z.size < 2 * CV_FOLDS:
        name, maker = ladder[0]
        return name, maker(u, v, z, w), 0.0
    rng = np.random.default_rng(seed)
    fold = rng.permutation(z.size) % CV_FOLDS
    sw = np.sqrt(np.maximum(w, 0.0))
    resid = np.empty((len(ladder), z.size))
    for ci, (_name, maker) in enumerate(ladder):
        for f in range(CV_FOLDS):
            tr, te = fold != f, fold == f
            pred = maker(u[tr], v[tr], z[tr], w[tr])(u[te], v[te])
            resid[ci, te] = (z[te] - pred) * sw[te]
    pick, gain = _pick_paired(_losses_from_resid(resid))
    name, maker = ladder[pick]
    return name, maker(u, v, z, w), gain


def _grid_smooth_fit(d, m, sigma):
    """Robust normalized convolution of masked grid data d."""
    from scipy.ndimage import gaussian_filter
    keep = m.copy()
    out = np.zeros_like(d)
    for _ in range(3):
        num = gaussian_filter(np.where(keep, d, 0.0), sigma, mode="nearest")
        den = gaussian_filter(keep.astype(float), sigma, mode="nearest")
        out = num / np.maximum(den, 1e-12)
        r = np.where(keep, d - out, 0.0)
        s = _rsig(r[keep])
        if s <= 0:
            break
        nk = m & (np.abs(d - out) < 3.0 * s)
        if nk.sum() == keep.sum():
            break
        keep = nk
    return out


def _grid_ladder(gh, gw):
    base = max(gh, gw)
    lad = []
    for name, order in (("constant", 0), ("plane", 1), ("quadratic", 2)):
        lad.append((name, ("poly", order)))
    lad.append(("smooth wide", ("smooth", base / 6.0)))
    lad.append(("cubic", ("poly", 3)))
    lad.append(("smooth", ("smooth", base / 10.0)))
    lad.append(("smooth fine", ("smooth", base / 16.0)))
    return lad


def _grid_predict(kind, d, m, U, V):
    if kind[0] == "poly":
        uu, vv = U[m], V[m]
        f = _fit_poly(uu, vv, d[m], np.ones(uu.size), kind[1])
        return f(U.ravel(), V.ravel()).reshape(d.shape)
    return _grid_smooth_fit(d, m, kind[1])


def _cv_choose_grid(d, m, U, V, seed):
    ladder = _grid_ladder(*d.shape)
    n = int(m.sum())
    if n < 2 * CV_FOLDS:
        return "constant", np.full(d.shape, float(np.median(d[m])) if n else 0.0), 0.0
    rng = np.random.default_rng(seed)
    fold = np.full(d.shape, -1)
    fold[m] = rng.permutation(n) % CV_FOLDS
    # Cell noise from differences between neighbouring cells: a smooth
    # gradient barely changes from one cell to the next, so this measures
    # the scatter without the structure we are trying to fit.
    dx = (d[:, 1:] - d[:, :-1])[m[:, 1:] & m[:, :-1]]
    dy = (d[1:, :] - d[:-1, :])[m[1:, :] & m[:-1, :]]
    nd = np.concatenate([dx, dy])
    s_noise = _rsig(nd) / math.sqrt(2.0) if nd.size >= 10 else _rsig(d[m])
    s_noise = max(s_noise, 1e-12)
    idx = np.flatnonzero(m.ravel())
    resid = np.empty((len(ladder), idx.size))
    fl = fold.ravel()[idx]
    for ci, (_name, kind) in enumerate(ladder):
        for f in range(CV_FOLDS):
            tr = m & (fold != f)
            pred = _grid_predict(kind, d, tr, U, V).ravel()[idx]
            te = fl == f
            resid[ci, te] = (d.ravel()[idx][te] - pred[te]) / s_noise
    pick, gain = _pick_paired(_losses_from_resid(resid))
    name, kind = ladder[pick]
    return name, _grid_predict(kind, d, m, U, V), gain


# ════════════════════════════════════════════════════════════════════════
#  Fitting a whole group
# ════════════════════════════════════════════════════════════════════════

def _star_noise_by_brightness(Z, ref, star_ok, su, sv):
    """Per-star noise as a function of BRIGHTNESS only, never of position.

    The obvious estimate -- each star's own scatter across the set -- folds
    the signal into the noise: a star near the edge, where transmission
    varies most between frames, scatters more BECAUSE of the very structure
    we are trying to fit, and gets down-weighted for it (measured on real
    data: ~2.7% at the field edge vs ~2.1% once each frame's own smooth trend
    is removed). So each frame's robust quadratic is removed first, the
    per-star scatter of what is left is taken, and that is smoothed against
    flux: stars of similar brightness share one noise level wherever they sit.
    Returns (sigma per star, outlier mask of stars far noisier than their
    brightness explains -- variables, blends).
    """
    N, K = Z.shape
    E = np.full_like(Z, np.nan)
    for i in range(N):
        s = star_ok & np.isfinite(Z[i])
        n = int(s.sum())
        if n < 12:
            continue
        p = _fit_poly(su[s], sv[s], Z[i, s], np.ones(n), 2 if n >= 24 else 1)
        E[i, s] = Z[i, s] - p(su[s], sv[s])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per = 1.4826 * np.nanmedian(np.abs(E - np.nanmedian(E, 0, keepdims=True)), 0)
    good = star_ok & np.isfinite(per) & (per > 0)
    if good.sum() < 3:
        fill = float(np.nanmedian(per)) if np.isfinite(per).any() else 0.01
        return np.full(K, max(fill, 1e-3)), np.zeros(K, bool)
    order = np.argsort(ref[good])
    lf = np.log(ref[good][order])
    pv = per[good][order]
    nb = max(1, min(12, int(good.sum()) // 25))
    edges = np.linspace(0, lf.size, nb + 1).astype(int)
    pairs = [(np.median(lf[a:b]), np.median(pv[a:b]))
             for a, b in zip(edges[:-1], edges[1:]) if b > a]
    cx = np.array([p[0] for p in pairs])
    cy = np.array([p[1] for p in pairs])
    if cy.size > 1:
        cy = np.minimum.accumulate(cy[::-1])[::-1]   # brighter is never noisier
    safe_ref = np.where(np.isfinite(ref) & (ref > 0), ref, np.nanmedian(ref[good]))
    lr = np.log(safe_ref)
    sigma = np.interp(lr, cx, cy) if cx.size > 1 else np.full(K, float(cy[0]))
    sigma = np.maximum(sigma, 0.5 * float(np.median(cy)))
    outlier = np.isfinite(per) & (per > 4.0 * sigma)
    return sigma, outlier


def _orientation_groups(groups, N):
    """Index arrays of the orientation groups big enough to shape the reference.

    None when there is nothing to balance: no labels, one orientation, or
    only one orientation with a real share of the frames.
    """
    if groups is None:
        return None
    g = np.asarray(list(groups))
    if g.size != N:
        return None
    labels, counts = np.unique(g, return_counts=True)
    need = max(MIN_FRAMES, int(math.ceil(MIN_GROUP_FRAC * N)))
    # Frames whose orientation could not be read are still corrected; they
    # just do not get a vote in what the reference is.
    big = [lab for lab, cnt in zip(labels, counts)
           if cnt >= need and str(lab) not in ("None", "")]
    if len(big) < 2:
        return None
    return [np.flatnonzero(g == lab) for lab in big]


def _balanced_ref(A, eligible, axis_min_count):
    """Reference over frames (axis 0), giving each orientation equal weight.

    Each orientation's median is taken separately, then their GEOMETRIC mean
    (mean of logs) for star fluxes / arithmetic mean for sky levels is the
    reference. A brightness pattern fixed to the optics -- a flat that does
    not quite match -- appears mirrored after a meridian flip, so the two
    orientations sit either side of the truth. The plain median picks the
    larger orientation and hands its whole error to every frame (measured on
    real IC410 Ha: 34 pre-flip vs 21 post-flip frames, a ~15% left-right
    split, inherited in full). Equal weight cancels the part that flips.
    Entries a group cannot support come back NaN.
    """
    meds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for idx in eligible:
            sub = A[idx]
            m = _fast_nanmedian(sub, axis=0)
            cnt = np.isfinite(sub).sum(0)
            m = np.where(cnt >= max(2, int(axis_min_count * idx.size)), m, np.nan)
            meds.append(m)
    return np.stack(meds)                 # (G, ...) -- caller combines


def _fit_mult_one(zrow, star_ok, w_all, su, sv, gu, gv, gh, gw, seed):
    """One frame's brightness map. Module-level so a worker process can run it.

    Returns (gain grid or None, model name, held-out gain, p2p %, clamped?).
    """
    sel = star_ok & np.isfinite(zrow)
    n = int(sel.sum())
    if n < 3:
        return None, "none (too few stars)", 0.0, 0.0, False
    name, pred, g = _cv_choose(su[sel], sv[sel], zrow[sel], w_all[sel],
                               _star_ladder(n), seed=seed)
    logR = pred(gu, gv).reshape(gh, gw)
    at_stars = pred(su[sel], sv[sel])
    lo = max(float(at_stars.min()) - 0.02, -LOG_CLAMP)
    hi = min(float(at_stars.max()) + 0.02, LOG_CLAMP)
    clamped = bool(np.any(logR < lo) or np.any(logR > hi))
    logR = np.clip(logR, lo, hi)
    p2p = 100.0 * (math.exp(float(logR.max() - logR.min())) - 1.0)
    return np.exp(-logR).astype(np.float32), name, float(g), p2p, clamped


def _fit_add_one(d, m, U, V, seed):
    """One frame's sky map. Module-level so a worker process can run it.

    Returns (offset grid or None, model name, held-out gain).
    """
    if m.sum() < 2 * CV_FOLDS:
        return None, "none (no clean sky)", 0.0
    dd = np.where(m, d, 0.0)
    name, fit, g = _cv_choose_grid(dd, m, U, V, seed=seed)
    lo, hi = float(np.min(fit[m])), float(np.max(fit[m]))
    pad = 0.05 * max(hi - lo, 1e-12)
    return np.clip(fit, lo - pad, hi + pad).astype(np.float32), name, float(g)


def _fit_mult_batch(task):
    idx, zrows, star_ok, w_all, su, sv, gu, gv, gh, gw, c = task
    return [(i,) + _fit_mult_one(z, star_ok, w_all, su, sv, gu, gv, gh, gw,
                                 seed=1000 * c + i)
            for i, z in zip(idx, zrows)]


def _fit_add_batch(task):
    idx, drows, mrows, U, V, c = task
    return [(i,) + _fit_add_one(d, m, U, V, seed=7000 * c + i)
            for i, d, m in zip(idx, drows, mrows)]


def _run_batches(fn, tasks, executor, workers):
    """Every frame's result from batched tasks -- in a process pool if given.

    A process pool is what actually uses the machine: per-frame fitting is
    many small numpy calls, so threads contend for the GIL (8 threads ran
    x1.3 on a 36-core machine). Results do not depend on which path ran:
    every frame carries its own seed.
    """
    out = []
    if executor is not None:
        for res in executor.map(fn, tasks):
            out.extend(res)
        return out
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, int(workers)),
                            thread_name_prefix="afnFit") as ex:
        for res in ex.map(fn, tasks):
            out.extend(res)
    return out


def _batches(N, n_slots):
    """Frame index batches: a few per worker, so a slow frame does not stall."""
    size = max(1, int(math.ceil(N / max(1, 4 * n_slots))))
    return [list(range(a, min(N, a + size))) for a in range(0, N, size)]


def _fit_group(st: _Stars, flux, ok, bg, *, additive: bool, log, workers=None,
               groups=None, executor=None):
    """flux (N,C,K), ok (N,C,K), bg (N,C,gh,gw) -> list[AFNFrameModel].

    groups: optional orientation label per frame (e.g. which side of the
    meridian flip). With two or more sizeable orientations the reference is
    balanced across them; otherwise it is the plain median of the set.
    executor: optional process pool for the per-frame fits; threads otherwise.
    """
    N, C, K = flux.shape
    su, sv = _uv(st.x, st.y, st.W, st.H)
    gu, gv = _grid_uv(st.cell, st.gh, st.gw, st.W, st.H)
    U = gu.reshape(st.gh, st.gw)
    V = gv.reshape(st.gh, st.gw)

    gains = np.ones((N, C, st.gh, st.gw), np.float32)
    offsets = np.zeros((N, C, st.gh, st.gw), np.float32)
    mult_names = [[None] * C for _ in range(N)]
    add_names = [[None] * C for _ in range(N)]
    mult_gain = np.zeros((N, C))
    add_gain = np.zeros((N, C))
    p2p = np.zeros((N, C))
    clamped = 0
    if workers is None:
        workers = max(1, min(8, _cpu_workers()))
    n_slots = getattr(executor, "_max_workers", None) or workers
    eligible = _orientation_groups(groups, N)
    batches = _batches(N, n_slots)

    for c in range(C):
        F = np.where(ok[:, c, :], flux[:, c, :].astype(np.float64), np.nan)
        cnt = np.isfinite(F).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            if eligible is None:
                ref = _fast_nanmedian(F, axis=0)                # the whole set
            else:
                lm = np.log(_balanced_ref(F, eligible, 0.3))
                ok_all = np.all(np.isfinite(lm), axis=0)
                ref = np.where(ok_all, np.exp(np.mean(np.where(ok_all, lm, 0.0), 0)),
                               np.nan)
            Z = np.log(F / ref[None, :])
        star_ok = (cnt >= max(MIN_FRAMES, int(0.3 * N))) & np.isfinite(ref) & (ref > 0)
        sig, outl = _star_noise_by_brightness(Z, ref, star_ok, su, sv)
        star_ok &= ~outl                                         # variables, blends
        w_all = 1.0 / np.maximum(sig, 1e-6) ** 2                 # inverse variances

        tasks = [(b, Z[b], star_ok, w_all, su, sv, gu, gv, st.gh, st.gw, c)
                 for b in batches]
        for i, gain, name, g, pp, cl in _run_batches(_fit_mult_batch, tasks,
                                                     executor, workers):
            mult_names[i][c] = name
            if gain is not None:
                gains[i, c] = gain
                mult_gain[i, c] = g
                p2p[i, c] = pp
                clamped += int(cl)

        if not additive:
            continue

        # Additive part, AFTER the gain: the target cancels in the difference.
        Bc = bg[:, c] * gains[:, c]
        if eligible is None:
            ens = _fast_nanmedian(Bc, axis=0)
        else:
            # sky fixed to the sensor (amp glow, a light leak) flips too
            bm = _balanced_ref(Bc, eligible, 0.3)
            ens = np.where(np.all(np.isfinite(bm), axis=0), np.mean(bm, axis=0),
                           np.nan)
        fin = np.isfinite(ens)
        if fin.sum() >= 2 * CV_FOLDS:
            e = ens[fin]
            bright = ens > (np.median(e) + 5.0 * max(_rsig(e), 1e-12))
        else:
            bright = np.zeros_like(fin)
        D = Bc - ens[None]
        M = np.isfinite(D) & fin[None] & ~bright[None]

        tasks = [(b, D[b], M[b], U, V, c) for b in batches]
        for i, off, name, g in _run_batches(_fit_add_batch, tasks, executor, workers):
            add_names[i][c] = name
            if off is not None:
                offsets[i, c] = off
                add_gain[i, c] = g

    models = []
    for i in range(N):
        models.append(AFNFrameModel(
            cell=st.cell,
            gain=np.ascontiguousarray(gains[i]),
            offset=np.ascontiguousarray(offsets[i]),
            info={"mult": mult_names[i], "add": add_names[i],
                  "p2p_pct": [float(x) for x in p2p[i]],
                  "mult_cv_gain": [float(x) for x in mult_gain[i]],
                  "add_cv_gain": [float(x) for x in add_gain[i]],
                  "has_offset": bool(additive),
                  "hw": (int(st.H), int(st.W))},
        ))
    return models, p2p, clamped, mult_gain, add_gain


# ════════════════════════════════════════════════════════════════════════
#  Group driver: measure (cached) -> fit
# ════════════════════════════════════════════════════════════════════════

def _signature(file_list, additive) -> str:
    h = hashlib.sha1()
    h.update(f"{AFN_VERSION}|{AP_FWHM}|{ANN_IN_FWHM}|{ANN_OUT_FWHM}|"
             f"{MAX_STARS}|{BG_STRIDE}|{GRID_TARGET}".encode())
    for p in file_list:
        try:
            s = os.stat(p)
            h.update(f"{os.path.normcase(os.path.abspath(p))}|{s.st_size}|"
                     f"{s.st_mtime_ns}\n".encode())
        except OSError:
            h.update(f"{p}|missing\n".encode())
    return h.hexdigest()


def _fit_with_pool(st, flux, ok, bg, *, additive, log, groups, processes, N,
                   pool=None):
    """_fit_group in a spawn process pool when asked for, threads otherwise.

    Spawn, not fork: safe alongside Qt on every OS. A pool passed in (the one
    that just did the measuring) is used and left for its owner to close;
    otherwise one is made here and lives only for the fit. Any failure falls
    back to threads; the maps are the same either way.
    """
    if pool is not None:
        try:
            return _fit_group(st, flux, ok, bg, additive=additive, log=log,
                              groups=groups, executor=pool)
        except Exception as e:                       # noqa: BLE001
            log(f"⚠️ AFN: process pool failed ({type(e).__name__}: {e}); "
                f"fitting on threads.")
            return _fit_group(st, flux, ok, bg, additive=additive, log=log,
                              groups=groups, executor=None)
    if processes and int(processes) > 1 and N >= 16:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            n = min(int(processes), N)
            pool = ProcessPoolExecutor(max_workers=n,
                                       mp_context=mp.get_context("spawn"))
            log(f"🔆 AFN: fitting {N} frames across {n} processes")
        except Exception as e:                       # noqa: BLE001
            log(f"⚠️ AFN: process pool unavailable ({type(e).__name__}: {e}); "
                f"fitting on threads.")
            pool = None
    try:
        if pool is not None:
            try:
                return _fit_group(st, flux, ok, bg, additive=additive, log=log,
                                  groups=groups, executor=pool)
            except Exception as e:                   # noqa: BLE001
                log(f"⚠️ AFN: process pool failed ({type(e).__name__}: {e}); "
                    f"fitting on threads.")
        return _fit_group(st, flux, ok, bg, additive=additive, log=log,
                          groups=groups, executor=None)
    finally:
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception:                        # noqa: BLE001
                pass


def _cpu_workers() -> int:
    return max(1, os.cpu_count() or 4)


def _measure_workers(frame_bytes: int) -> int:
    """Threads for the measure pass: enough to keep the drive busy.

    Measuring is mostly reading (187 MB per 8288x5644 mono frame), and file
    reads release the GIL, so threads overlap well; the CPU part per frame is
    tens of milliseconds. Each thread holds one whole frame, so the count is
    capped by free memory (a quarter of it) as well as by cores, and by 16 --
    past that a single drive only queues, it does not read faster.
    """
    n = min(16, _cpu_workers())
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
        per = max(1, int(frame_bytes * 1.6))        # frame + working copies
        n = min(n, max(1, (avail // 4) // per))
    except Exception:                                # noqa: BLE001
        n = min(n, 4)
    return max(1, n)


def _save_fit(path, sig, models, p2p, clamped, mult_gain, add_gain, log):
    import json
    try:
        tmp = path + ".tmp.npz"
        np.savez(tmp, sig=np.array(sig),
                 cell=np.array(models[0].cell),
                 gain=np.stack([m.gain for m in models]),
                 offset=np.stack([m.offset for m in models]),
                 info=np.array(json.dumps([m.info for m in models])),
                 p2p=p2p, clamped=np.array(clamped),
                 mult_gain=mult_gain, add_gain=add_gain)
        os.replace(tmp, path)
    except Exception as e:                           # noqa: BLE001
        log(f"⚠️ AFN: could not cache maps ({e}).")


def _load_fit(path, sig, st, N, C):
    import json
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if str(z["sig"]) != sig:
            return None
        gain, offset = z["gain"], z["offset"]
        if gain.shape != (N, C, st.gh, st.gw):
            return None
        infos = json.loads(str(z["info"]))
        cell = int(z["cell"])
        models = []
        for i in range(N):
            info = dict(infos[i])
            if "hw" in info:
                info["hw"] = tuple(info["hw"])
            models.append(AFNFrameModel(cell=cell,
                                        gain=np.ascontiguousarray(gain[i]),
                                        offset=np.ascontiguousarray(offset[i]),
                                        info=info))
        return (models, z["p2p"], int(z["clamped"]), z["mult_gain"], z["add_gain"])
    except Exception:                                # noqa: BLE001
        return None


class _FitsFrameReader:
    """GUI-free mirror of stacking_suite._MMImage's FITS path, for workers.

    Measurement has to read EXACTLY what integration reads, or the maps are
    measured on different pixels than they are applied to. _MMImage lives in
    stacking_suite, which drags in Qt -- far too heavy to import in 32 spawn
    workers -- so its FITS behaviour is mirrored here call for call: same HDU
    choice, same read-only memmap with a non-memmap fallback, same colour-axis
    handling, same integer normalization. Bit-identity with _MMImage on real
    frames is checked by test; anything that is not FITS stays on the
    threaded _MMImage path.
    """

    def __init__(self, path: str):
        from astropy.io import fits

        def _do_open(memmap_flag: bool):
            extra = {"mode": "denywrite"} if memmap_flag else {}
            hdul = fits.open(path, memmap=memmap_flag, **extra)
            try:
                hdu = None
                for h in hdul:
                    if getattr(h, "data", None) is not None:
                        hdu = h
                        break
                if hdu is None or hdu.data is None:
                    hdul.close()
                    raise ValueError(f"Empty FITS: {path}")
                return hdul, hdu
            except Exception:
                try:
                    hdul.close()
                except Exception:                    # noqa: BLE001
                    pass
                raise

        try:
            hdul, hdu = _do_open(True)
        except Exception:                            # noqa: BLE001
            hdul, hdu = _do_open(False)
        self._hdul = hdul
        d = hdu.data
        self._d = d
        self.ndim = d.ndim
        self._orig_dtype = d.dtype
        try:
            self._bitpix = int(hdu.header.get("BITPIX", 0))
        except Exception:                            # noqa: BLE001
            self._bitpix = 0
        if self.ndim == 2:
            self._color_axis = None
            self._spat = (0, 1)
        elif self.ndim == 3:
            self._color_axis = next((i for i, s in enumerate(d.shape) if s == 3), None)
            self._spat = ((0, 1) if self._color_axis is None
                          else tuple(i for i in range(3) if i != self._color_axis))
        else:
            raise ValueError(f"Unsupported ndim={self.ndim} for {path}")

    def read_tile_into(self, dst, y0, y1, x0, x1, channels):
        d = self._d
        if self.ndim == 2:
            tile = d[y0:y1, x0:x1]
        else:
            sl = [slice(None)] * 3
            sl[self._spat[0]] = slice(y0, y1)
            sl[self._spat[1]] = slice(x0, x1)
            tile = d[tuple(sl)]
            if self._color_axis is not None and self._color_axis != 2:
                tile = np.moveaxis(tile, self._color_axis, -1)
        if tile.ndim == 2:
            tile = tile[..., None]
        if tile.shape[-1] == 1 and channels == 3:
            np.copyto(dst, tile, casting="unsafe")
        else:
            np.copyto(dst, tile[..., :channels], casting="unsafe")
        if self._orig_dtype.kind != "f":
            divisor = {8: 255.0, 16: 65535.0, -16: 65535.0,
                       32: 4294967295.0}.get(self._bitpix)
            if divisor is not None:
                np.divide(dst, np.float32(divisor), out=dst)

    def close(self):
        self._d = None
        try:
            self._hdul.close()
        except Exception:                            # noqa: BLE001
            pass


_WSTATE: dict = {}


def _pool_init(st, H, W, C):
    """Per-worker state: the star list, sent once instead of with every frame."""
    _WSTATE.update(st=st, H=H, W=W, C=C)


def _measure_worker_one(task):
    i, path = task
    st, H, W, C = _WSTATE["st"], _WSTATE["H"], _WSTATE["W"], _WSTATE["C"]
    fr = _read_full(_FitsFrameReader, path, H, W, C)
    f, o, b = _measure_frame(fr, st)
    return i, f, o, b


def _measure_in_pool(pool, file_list, flux, ok, bg, N, log, cancelled):
    """Measure every frame in the process pool, filling flux/ok/bg in place."""
    from concurrent.futures import as_completed
    futs = [pool.submit(_measure_worker_one, (i, file_list[i])) for i in range(N)]
    done = 0
    try:
        for fut in as_completed(futs):
            if cancelled():
                raise InterruptedError("AFN cancelled")
            i, f, o, b = fut.result()
            flux[i], ok[i], bg[i] = f, o, b
            done += 1
            if done == N or done % 10 == 0:
                log(f"🔆 AFN measuring {done}/{N}")
    except BaseException:
        for f in futs:
            f.cancel()
        raise


def _make_pool(processes, N, frame_bytes, st, H, W, C, log):
    """A spawn pool for measure + fit, sized to cores AND free memory.

    Each measuring worker holds one whole frame plus working copies, so the
    count is capped by a quarter of free RAM as well as by `processes`.
    None when a pool is not wanted or cannot start (threads take over).
    """
    if not processes or int(processes) <= 1 or N < 16:
        return None
    n = min(int(processes), N)
    try:
        import psutil
        avail = int(psutil.virtual_memory().available)
        n = min(n, max(1, (avail // 4) // max(1, int(frame_bytes * 2.2))))
    except Exception:                                # noqa: BLE001
        pass
    if n <= 1:
        return None
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"),
                                   initializer=_pool_init, initargs=(st, H, W, C))
        log(f"🔆 AFN: using {n} processes")
        return pool
    except Exception as e:                           # noqa: BLE001
        log(f"⚠️ AFN: process pool unavailable ({type(e).__name__}: {e}); "
            f"using threads.")
        return None


def _read_full(open_reader, path, H, W, C):
    src = open_reader(path)
    try:
        buf = np.empty((H, W, C), np.float32)
        src.read_tile_into(buf, 0, H, 0, W, C)
        return buf
    finally:
        try:
            src.close()
        except Exception:
            pass


def build_models_for_group(
    file_list,
    open_reader: Callable,
    *,
    shape_hwc,
    cache_dir: Optional[str] = None,
    additive: bool = True,
    log: Callable[[str], None] = print,
    cancelled: Callable[[], bool] = lambda: False,
    workers: Optional[int] = None,
    orientation=None,
    processes: int = 0,
):
    """Measure every frame of one integration group and fit its models.

    `orientation`, optional, one label per frame (which side of the meridian
    flip it was taken on). With two sizeable orientations the reference is
    balanced across them, so a flat error that flips with the mount cancels
    instead of being copied from the larger side to every frame.

    `processes` > 1 fits the frames in a spawn process pool of that size
    (the caller decides whether process pools work on this machine); any
    pool trouble falls back to threads with the same result.

    `open_reader(path)` must return the SAME reader the integration uses
    (read_tile_into(dst, y0, y1, x0, x1, channels) + close()), so the maps
    are measured in exactly the orientation they are applied in.

    Returns a list of AFNFrameModel aligned with file_list, or None when the
    group cannot be normalized this way (too few frames or stars). None means
    "integrate as before", never a partial result.
    """
    N = len(file_list)
    if N < MIN_FRAMES:
        log(f"🔆 AFN: {N} frame(s) — a set median needs at least {MIN_FRAMES}; "
            f"skipped for this group.")
        return None
    H, W, C = (int(shape_hwc[0]), int(shape_hwc[1]), int(shape_hwc[2]))
    t0 = time.perf_counter()

    sig = _signature(file_list, additive)
    cache_path = os.path.join(cache_dir, f"{sig[:24]}.npz") if cache_dir else None
    st = flux = ok = bg = None
    pool = None          # made after star detection, shared by measure + fit

    if cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path, allow_pickle=False)
            if str(z["sig"]) == sig:
                cell, gh, gw = (int(z["cell"]), int(z["gh"]), int(z["gw"]))
                st = _Stars(x=z["x"], y=z["y"], fwhm=float(z["fwhm"]),
                            r_ap=float(z["r_ap"]), r_in=float(z["r_in"]),
                            r_out=float(z["r_out"]), H=H, W=W, C=C, cell=cell,
                            gh=gh, gw=gw, bgmask_s=np.zeros((1, 1), bool))
                flux, ok, bg = z["flux"], z["ok"], z["bg"]
                log(f"🔆 AFN: reusing cached measurements for {N} frames.")
        except Exception as e:                       # noqa: BLE001
            log(f"⚠️ AFN: cache unreadable ({e}); measuring again.")
            st = None

    if st is None:
        # ---- star positions from a median of a few frames -------------------
        pick = np.unique(np.linspace(0, N - 1, min(ENSEMBLE_N, N)).astype(int))
        from concurrent.futures import ThreadPoolExecutor

        def _lum(j):
            if cancelled():
                raise InterruptedError("AFN cancelled")
            fr = _read_full(open_reader, file_list[j], H, W, C)
            lum = fr.mean(axis=2) if C > 1 else fr[..., 0]
            return np.where((lum != 0.0) & np.isfinite(lum), lum,
                            np.nan).astype(np.float32)

        with ThreadPoolExecutor(max_workers=len(pick),
                                thread_name_prefix="afnEnsemble") as ex:
            lums = np.stack(list(ex.map(_lum, pick)))
        # The median of the stack, in row bands across threads. This was 15 s
        # with np.nanmedian on 8288x5644 -- seven times the reads around it.
        ens = np.empty((H, W), np.float32)
        band = 256

        def _med(r0):
            ens[r0:r0 + band] = _fast_nanmedian(lums[:, r0:r0 + band], axis=0)

        with ThreadPoolExecutor(max_workers=_cpu_workers(),
                                thread_name_prefix="afnEnsemble") as ex:
            list(ex.map(_med, range(0, H, band)))
        del lums
        valid = np.isfinite(ens)
        ens = np.where(valid, ens, 0.0).astype(np.float32)
        st = _detect_stars(ens, valid, C, log)
        del ens, valid
        if st is None:
            return None

        # ---- measure every frame at the same positions ----------------------
        K = st.x.size
        flux = np.zeros((N, C, K), np.float32)
        ok = np.zeros((N, C, K), bool)
        bg = np.full((N, C, st.gh, st.gw), np.nan, np.float32)
        # Processes first: measuring holds the GIL in SEP, the per-star loop
        # and FITS header parsing, so threads topped out at ~6 of 36 cores.
        # FITS only -- the worker reader mirrors _MMImage's FITS path; XISF
        # and anything else measure on threads through _MMImage itself.
        pool = _make_pool(processes, N, H * W * C * 4, st, H, W, C, log)
        measured = False
        if pool is not None and all(not str(p).lower().endswith(".xisf")
                                    for p in file_list):
            try:
                _measure_in_pool(pool, file_list, flux, ok, bg, N, log, cancelled)
                measured = True
            except InterruptedError:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:                   # noqa: BLE001
                log(f"⚠️ AFN: multi-process measuring failed "
                    f"({type(e).__name__}: {e}); using threads.")
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:                    # noqa: BLE001
                    pass
                pool = None

        if not measured:
            if workers is None:
                workers = _measure_workers(H * W * C * 4)
            done = [0]
            lock = threading.Lock()

            def _one(i):
                if cancelled():
                    raise InterruptedError("AFN cancelled")
                fr = _read_full(open_reader, file_list[i], H, W, C)
                f, o, b = _measure_frame(fr, st)
                del fr
                flux[i], ok[i], bg[i] = f, o, b
                with lock:
                    done[0] += 1
                    k = done[0]
                if k == N or k % 10 == 0:
                    log(f"🔆 AFN measuring {k}/{N}")

            with ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                    thread_name_prefix="afnMeasure") as ex:
                list(ex.map(_one, range(N)))

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                tmp = cache_path + ".tmp.npz"
                np.savez(tmp, sig=np.array(sig), x=st.x, y=st.y,
                         fwhm=st.fwhm, r_ap=st.r_ap, r_in=st.r_in,
                         r_out=st.r_out, cell=st.cell, gh=st.gh, gw=st.gw,
                         flux=flux, ok=ok, bg=bg)
                os.replace(tmp, cache_path)
            except Exception as e:                   # noqa: BLE001
                log(f"⚠️ AFN: could not cache measurements ({e}).")

    t_meas = time.perf_counter() - t0
    if cancelled():
        raise InterruptedError("AFN cancelled")

    t1 = time.perf_counter()
    # The fitted maps are cached too, beside the measurements: on 403 real
    # frames the fit took ~80 s, and re-integrating the same frames with a
    # different rejection setting should not pay that again.
    fit_path = (cache_path[:-4] + f"_fit{'A' if additive else 'M'}.npz"
                if cache_path else None)
    # The maps depend on how they are fitted and on which frames are which
    # orientation, not only on the measurements -- key the cache on all three.
    labels = None if orientation is None else [str(o) for o in orientation]
    fit_sig = hashlib.sha1(
        f"{sig}|{FIT_VERSION}|{labels!r}".encode()).hexdigest()
    elig = _orientation_groups(labels, N)
    if elig is not None:
        sizes = " / ".join(str(len(e)) for e in elig)
        log(f"🔆 AFN: reference balanced across {len(elig)} meridian-flip "
            f"orientations ({sizes} frames), so a flat error that flips "
            f"with the mount cancels instead of being copied to every frame.")
    fitted = _load_fit(fit_path, fit_sig, st, N, C) if fit_path else None
    if fitted is not None:
        models, p2p, clamped, mult_gain, add_gain = fitted
        log(f"🔆 AFN: reusing cached maps for {N} frames.")
    else:
        try:
            models, p2p, clamped, mult_gain, add_gain = _fit_with_pool(
                st, flux, ok, bg, additive=additive, log=log, groups=labels,
                processes=processes, N=N, pool=pool)
        finally:
            if pool is not None:
                try:
                    pool.shutdown(wait=True, cancel_futures=True)
                except Exception:                    # noqa: BLE001
                    pass
                pool = None
        if fit_path:
            _save_fit(fit_path, fit_sig, models, p2p, clamped, mult_gain, add_gain, log)
    if pool is not None:              # measured in a pool, maps came from cache
        try:
            pool.shutdown(wait=True, cancel_futures=True)
        except Exception:                            # noqa: BLE001
            pass
    t_fit = time.perf_counter() - t1

    # ---- summary ------------------------------------------------------------
    from collections import Counter
    mult = Counter(m.info["mult"][0] for m in models)
    add = Counter(m.info["add"][0] for m in models) if additive else None
    pp = p2p[:, 0]
    log(f"🔆 AFN fitted {N} frames: brightness maps "
        + ", ".join(f"{k} {v}" for k, v in mult.most_common())
        + f"; field variation median {np.median(pp):.1f}%, "
          f"max {np.max(pp):.1f}%")
    # Held-out gain: how much better each frame's chosen map predicts stars
    # (or sky cells) it was NOT fitted to, than a single global value does.
    # Independent evidence that the maps describe the frames, not the noise.
    mg = 100.0 * mult_gain[:, 0]
    log(f"🔆 AFN held-out star prediction vs a global scale: "
        f"median {np.median(mg):.1f}% better, 90th {np.percentile(mg, 90):.1f}%")
    if add is not None:
        log("🔆 AFN sky maps: " + ", ".join(f"{k} {v}" for k, v in add.most_common()))
        ag = 100.0 * add_gain[:, 0]
        log(f"🔆 AFN held-out sky prediction vs a global offset: "
            f"median {np.median(ag):.1f}% better")
    if clamped:
        log(f"⚠️ AFN: {clamped} map(s) were held to the range the stars "
            f"support (or to 2x) where they would have extrapolated further.")
    log(f"🔆 AFN timing: measure {t_meas:.1f}s, fit {t_fit:.1f}s")
    return models
