# src/setiastro/saspro/seestar_mosaic.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Seestar / smart-telescope mosaics: plate-solve every frame, place it on one
canvas by its own plate solution, and average-stack.

A Seestar mosaic is hundreds of short frames from an alt-az mount: the field
rotates through the night and the scope steps between panel positions. Every
frame is plate-solved against the local Gaia catalogue, seeded by the pointing
in its header (about 1-2 s a frame, run in parallel worker processes; a blind
quad-index solve is the fallback). Each frame is then mapped onto a canvas
built around all of them by the projective transform between its WCS and the
canvas WCS -- the shift and rotation between frames, exact for TAN-to-TAN --
and the frames are averaged, with an optional sigma-clip against satellites.

GUI-free; Mosaic Master's window drives it.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

FITS_EXTS = (".fit", ".fits", ".fts", ".fz")
FEATHER_PX = 48           # frame-edge weight ramp, hides coverage steps
EDGE_ERODE_PX = 3         # interpolation that touched the frame edge
MAX_CANVAS_SIDE = 40000


# ─────────────────────────────────────────────────────────────────────────────
# Worker processes
# ─────────────────────────────────────────────────────────────────────────────
def init_worker(org: str = "", app: str = ""):
    """Give QSettings the app's identity, so a worker finds the user's Gaia
    library directory (a QSettings value) the way the GUI does."""
    try:
        from PyQt6.QtCore import QCoreApplication
        if org:
            QCoreApplication.setOrganizationName(org)
        if app:
            QCoreApplication.setApplicationName(app)
    except Exception:
        pass


def bayer_of(header):
    try:
        bp = header.get("BAYERPAT")
    except Exception:
        return None
    if not bp:
        return None
    bp = str(bp[0] if isinstance(bp, tuple) else bp).strip().upper()
    return bp if bp in ("RGGB", "BGGR", "GRBG", "GBRG") else None


def _as_header(h):
    from astropy.io.fits import Header
    if isinstance(h, Header):
        return h
    out = Header()
    try:
        for k, v in dict(h or {}).items():
            try:
                out[str(k)[:8]] = v
            except Exception:
                pass
    except Exception:
        pass
    return out


def load_frame(path):
    """(image float32 2-D or HWC, header) as SASpro loads it, or (None, None)."""
    from setiastro.saspro.legacy.image_manager import load_image
    img, hdr, _bd, _mono = load_image(path)
    if img is None:
        return None, None
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return a, _as_header(hdr)


def luminance_for_solve(img, header):
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        return a[..., :3].mean(axis=2)
    bp = bayer_of(header)
    if bp:
        from setiastro.saspro.legacy.numba_utils import debayer_fits_fast
        return debayer_fits_fast(a, bp, method="edge").mean(axis=2).astype(np.float32)
    return a


def solve_frame(path, image=None, header_str=None):
    """Plate-solve one frame. Seeded by the RA/DEC and pixel scale in its
    header; a blind quad-index solve if that fails. The user's plate-solver
    dialog settings (manual seed, force blind, ...) are deliberately not
    used: settings=None. Returns a dict (picklable)."""
    t0 = time.time()
    from astropy.io.fits import Header
    from setiastro.saspro.plate_solver import _solve_with_GAIA
    out = {"path": path, "ok": False, "header": None, "error": None,
           "shape": None, "mode": None, "pid": os.getpid()}
    try:
        if image is None:
            image, header = load_frame(path)
            if image is None:
                out["error"] = "could not be read"
                return out
        else:
            header = Header.fromstring(header_str) if header_str else Header()
        lum = luminance_for_solve(image, header)
        out["shape"] = tuple(lum.shape[:2])
        quiet = (lambda *_: None)
        ok, res = _solve_with_GAIA(lum, header, parent=None, settings=None,
                                   status_cb=quiet, force_blind=False)
        mode = "seeded"
        if not ok:
            ok, res = _solve_with_GAIA(lum, header, parent=None, settings=None,
                                       status_cb=quiet, force_blind=True)
            mode = "blind"
        if ok and isinstance(res, Header):
            out.update(ok=True, header=res.tostring(), mode=mode)
        else:
            out["error"] = str(res)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["elapsed"] = time.time() - t0
    return out


def warm():
    from setiastro.saspro import plate_solver  # noqa: F401
    from setiastro.saspro.legacy.image_manager import load_image  # noqa: F401
    return os.getpid()


# ─────────────────────────────────────────────────────────────────────────────
# Solution cache (so a rebuild doesn't re-solve; raw files are never rewritten)
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path():
    try:
        from PyQt6.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)
    except Exception:
        base = ""
    base = base or os.path.join(os.path.expanduser("~"), ".saspro_cache")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "seestar_wcs_cache.json")


def _cache_key(path):
    try:
        st = os.stat(path)
        return f"{os.path.normcase(os.path.abspath(path))}|{st.st_size}|{st.st_mtime_ns}"
    except Exception:
        return None


def load_cache():
    try:
        with open(_cache_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: dict, max_entries: int = 20000):
    try:
        if len(cache) > max_entries:
            for k in list(cache)[: len(cache) - max_entries]:
                cache.pop(k, None)
        tmp = _cache_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, _cache_path())
    except Exception as e:
        print(f"[Seestar] could not save the plate-solve cache: {e}")


def cache_get(cache, path):
    k = _cache_key(path)
    return cache.get(k) if k else None


def cache_put(cache, path, header_str):
    k = _cache_key(path)
    if k:
        cache[k] = header_str


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def _radec_to_vec(ra, dec):
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    cd = np.cos(d)
    return np.stack([cd * np.cos(r), cd * np.sin(r), np.sin(d)], axis=-1)


def frame_outline(wcs, shape, n=8):
    """(ra, dec) along a frame's border, n points per side."""
    H, W = shape[:2]
    xs = np.linspace(0, W - 1, n); ys = np.linspace(0, H - 1, n)
    px = np.concatenate([xs, np.full(n, W - 1), xs[::-1], np.zeros(n)])
    py = np.concatenate([np.zeros(n), ys, np.full(n, H - 1), ys[::-1]])
    ra, dec = wcs.all_pix2world(px, py, 0)
    return np.asarray(ra, float), np.asarray(dec, float)


def make_canvas(wcs_list, shapes, pad: int = 4):
    """Canvas WCS around every frame: TAN at the mean pointing, the frames'
    median pixel scale, north up and east left as displayed (row 0 at the
    top). Returns (canvas WCS, (H, W))."""
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    vecs, scales, outlines = [], [], []
    for w, sh in zip(wcs_list, shapes):
        ra, dec = frame_outline(w, sh)
        g = np.isfinite(ra) & np.isfinite(dec)
        outlines.append((ra[g], dec[g]))
        vecs.append(_radec_to_vec(ra[g], dec[g]))
        try:
            scales.append(float(np.mean(proj_plane_pixel_scales(w.celestial))))
        except Exception:
            pass
    c = np.concatenate(vecs, axis=0).mean(axis=0)
    c /= max(float(np.linalg.norm(c)), 1e-12)
    ra0 = float(np.degrees(np.arctan2(c[1], c[0]))) % 360.0
    dec0 = float(np.degrees(np.arcsin(np.clip(c[2], -1.0, 1.0))))
    s = float(np.median(scales)) if scales else 1.0 / 3600.0

    cw = WCS(naxis=2)
    cw.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    cw.wcs.crval = [ra0, dec0]
    cw.wcs.crpix = [1.0, 1.0]
    # x grows west (east left), y grows south (north up with row 0 at the top)
    cw.wcs.cd = np.array([[-s, 0.0], [0.0, -s]])
    xs, ys = [], []
    for ra, dec in outlines:
        x, y = cw.wcs_world2pix(ra, dec, 0)
        xs.append(x); ys.append(y)
    x = np.concatenate(xs); y = np.concatenate(ys)
    x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
    y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
    W = int(math.ceil(x1 - x0)) + 2 * pad + 1
    H = int(math.ceil(y1 - y0)) + 2 * pad + 1
    if max(W, H) > MAX_CANVAS_SIDE:
        raise ValueError(f"the frames span {W} x {H} px, more than the {MAX_CANVAS_SIDE} px "
                         f"limit; are some plate solutions wrong?")
    cw.wcs.crpix = [1.0 - x0 + pad, 1.0 - y0 + pad]
    return cw, (H, W)


def frame_homography(wcs, shape, canvas_wcs, n: int = 7):
    """3x3 projective transform from frame pixels to canvas pixels, fitted to
    an n x n grid mapped exactly through both WCS (SIP included). TAN to TAN
    is a projective map, so the fit is exact without SIP and sub-pixel with it."""
    import cv2
    H, W = shape[:2]
    gx, gy = np.meshgrid(np.linspace(0, W - 1, n), np.linspace(0, H - 1, n))
    gx = gx.ravel(); gy = gy.ravel()
    ra, dec = wcs.all_pix2world(gx, gy, 0)
    cx, cy = canvas_wcs.wcs_world2pix(ra, dec, 0)
    src = np.column_stack([gx, gy]).astype(np.float64)
    dst = np.column_stack([cx, cy]).astype(np.float64)
    Hm, _ = cv2.findHomography(src, dst, 0)
    return Hm


def canvas_corners(Hm, shape):
    H, W = shape[:2]
    pts = np.array([[0, 0, 1], [W - 1, 0, 1], [W - 1, H - 1, 1], [0, H - 1, 1]], float).T
    q = Hm @ pts
    return (q[:2] / q[2]).T                     # (4, 2) canvas x, y


def canvas_bbox(Hm, shape, canvas_shape, pad: int = 2):
    c = canvas_corners(Hm, shape)
    Hc, Wc = canvas_shape
    x0 = max(0, int(math.floor(c[:, 0].min())) - pad)
    y0 = max(0, int(math.floor(c[:, 1].min())) - pad)
    x1 = min(Wc, int(math.ceil(c[:, 0].max())) + pad + 1)
    y1 = min(Hc, int(math.ceil(c[:, 1].max())) + pad + 1)
    return x0, y0, x1, y1


# ─────────────────────────────────────────────────────────────────────────────
# One frame onto the canvas
# ─────────────────────────────────────────────────────────────────────────────
def frame_sky(img):
    """Per-channel sky level (darkest tiles, sigma-clipped), as Mosaic Master
    measures a panel's sky."""
    from setiastro.saspro.mosaic_master import estimate_background_level
    a = np.asarray(img)
    step = max(1, int(max(a.shape[:2]) // 512))       # a sky level needs no full resolution
    lev, _sig = estimate_background_level(a[::step, ::step])
    return np.asarray(lev, np.float32)


def prepare_frame(path, Hm, bbox, *, image=None, header=None, debayer_method="rcd",
                  crop=(0, 0, 0, 0), sky=None):
    """Load, debayer, subtract the frame's own sky, and warp onto its canvas
    box. Returns dict: data (h, w, C) sky-subtracted, weight (h, w) feathered
    (0 = not covered), sky (C,), bbox. None if unreadable."""
    import cv2
    from setiastro.saspro.star_alignment import _warp_antiring
    if image is None:
        image, header = load_frame(path)
        if image is None:
            return None
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 2:
        bp = bayer_of(header)
        if bp:
            from setiastro.saspro.legacy.numba_utils import debayer_fits_fast
            a = debayer_fits_fast(a, bp, method=str(debayer_method or "rcd").lower())
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    a = a[..., :3]
    Hs, Ws = a.shape[:2]
    valid = np.isfinite(a).all(axis=2) & (a != 0).any(axis=2)
    l, r, t, b = (int(v) for v in crop)
    if l: valid[:, :l] = False
    if r: valid[:, Ws - r:] = False
    if t: valid[:t, :] = False
    if b: valid[Hs - b:, :] = False
    if sky is None:
        sky = frame_sky(a)
    sky = np.resize(np.asarray(sky, np.float32).ravel(), a.shape[2])
    # Warp on a pedestal: invalid pixels and the outside border sit at the
    # same level as the sky-subtracted data, so Lanczos has no edge step to
    # ring on, and _warp_antiring's zero floor never touches the noise.
    PED = 1.0
    src = np.where(valid[..., None], a - sky[None, None, :], 0.0).astype(np.float32) + PED
    if src.shape[2] == 1:
        src = src[..., 0]

    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ Hm

    def _wp(img, flags, border=0.0):
        bv = (border,) * 4
        return cv2.warpPerspective(img, T, (bw, bh), flags=flags,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    data = _warp_antiring(lambda: _wp(src, cv2.INTER_LANCZOS4, PED),
                          lambda: _wp(src, cv2.INTER_LINEAR, PED))
    data = np.asarray(data, np.float32) - PED
    if data.ndim == 2:
        data = data[..., None]

    cov = _wp(valid.astype(np.float32), cv2.INTER_LINEAR) >= 0.999
    if EDGE_ERODE_PX > 0 and cov.any():
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * EDGE_ERODE_PX + 1,) * 2)
        cov = cv2.erode(cov.astype(np.uint8), k).astype(bool)
    d = cv2.distanceTransform(cov.astype(np.uint8), cv2.DIST_L2, 3)
    w = np.clip((d + 1.0) / float(FEATHER_PX), 0.0, 1.0).astype(np.float32)
    w[~cov] = 0.0
    return {"data": data, "weight": w, "sky": sky, "bbox": bbox}


# ─────────────────────────────────────────────────────────────────────────────
# Average stack over the canvas, with one-pass outlier rejection
# ─────────────────────────────────────────────────────────────────────────────
# A sigma clip needs each pixel's mean and spread before it can judge a
# sample, so it costs a second read, debayer and warp of every frame -- half
# the stacking time. The outliers that matter here (satellite and aircraft
# trails, hot pixels that wander with field rotation, cosmic rays) are a few
# samples per pixel. So one pass keeps, per pixel and channel, the sums plus
# its few most extreme samples (REJECT_HIGH brightest, REJECT_LOW faintest)
# with their weights. At the end those candidates go through a generalized
# ESD (Rosner): repeatedly take the most extreme remaining candidate, measured
# against the mean and sigma of the samples left WITHOUT it (so a trail
# can't hide in the spread it inflates), with the Bonferroni Student-t
# critical value Live Stacking uses; the outliers are the largest prefix of
# that sequence that passes, and they are subtracted from the sums. Ordinary
# samples are never trimmed, unlike a k-sigma clip, which on pure noise drops
# 0.27% of good samples at 3 sigma.
REJECT_HIGH = 3
REJECT_LOW = 1
REJECT_ALPHA = 1e-3          # false-removal rate per pixel, split between the two ends
REJECT_MIN_REST = 3          # samples that must remain to judge one

try:
    from numba import njit as _njit
except Exception:                                    # numba missing: plain mean below
    _njit = None

if _njit is not None:
    @_njit(cache=True)
    def _acc_one(S, S2, Wt, N, Hv, Hw, Lv, Lw, v, w, y0, x0, track):
        h, ww, C = v.shape
        KH = Hv.shape[3]
        KL = Lv.shape[3]
        for y in range(h):
            for x in range(ww):
                wt = w[y, x]
                if wt <= 0.0:
                    continue
                Y = y0 + y; X = x0 + x
                Wt[Y, X] += wt
                N[Y, X] += 1.0
                for c in range(C):
                    val = v[y, x, c]
                    S[Y, X, c] += wt * val
                    if not track:
                        continue
                    S2[Y, X, c] += wt * val * val
                    if val > Hv[Y, X, c, KH - 1]:            # into the brightest few
                        j = KH - 1
                        while j > 0 and val > Hv[Y, X, c, j - 1]:
                            Hv[Y, X, c, j] = Hv[Y, X, c, j - 1]
                            Hw[Y, X, c, j] = Hw[Y, X, c, j - 1]
                            j -= 1
                        Hv[Y, X, c, j] = val
                        Hw[Y, X, c, j] = wt
                    if val < Lv[Y, X, c, KL - 1]:            # into the faintest few
                        j = KL - 1
                        while j > 0 and val < Lv[Y, X, c, j - 1]:
                            Lv[Y, X, c, j] = Lv[Y, X, c, j - 1]
                            Lw[Y, X, c, j] = Lw[Y, X, c, j - 1]
                            j -= 1
                        Lv[Y, X, c, j] = val
                        Lw[Y, X, c, j] = wt
        return 0

    @_njit(cache=True)
    def _finish_one(S, S2, Wt, N, Hv, Hw, Lv, Lw, crit, min_frames, min_rest, out):
        H, W, C = S.shape
        KH = Hv.shape[3]
        KL = Lv.shape[3]
        K = KH + KL
        cs = np.empty(K + 1); cs2 = np.empty(K + 1); cw = np.empty(K + 1)
        removed = 0
        considered = 0
        for Y in range(H):
            for X in range(W):
                w0 = Wt[Y, X]
                n0 = int(N[Y, X])
                for c in range(C):
                    if w0 <= 0.0:
                        out[Y, X, c] = 0.0
                        continue
                    s = float(S[Y, X, c]); s2 = float(S2[Y, X, c]); w = float(w0)
                    if n0 < min_frames:
                        out[Y, X, c] = s / w
                        continue
                    considered += n0
                    cs[0] = s; cs2[0] = s2; cw[0] = w
                    n = n0
                    ih = 0; il = 0
                    best = 0
                    for step in range(K):
                        if n - 1 < min_rest:
                            break
                        th = -1.0; tl = -1.0
                        if ih < KH:
                            xv = Hv[Y, X, c, ih]; xw = Hw[Y, X, c, ih]
                            wr = w - xw
                            if np.isfinite(xv) and wr > 1e-9:
                                mr = (s - xw * xv) / wr
                                vr = (s2 - xw * xv * xv) / wr - mr * mr
                                if vr > 0.0:
                                    th = (xv - mr) / np.sqrt(vr)
                        if il < KL:
                            xv = Lv[Y, X, c, il]; xw = Lw[Y, X, c, il]
                            wr = w - xw
                            if np.isfinite(xv) and wr > 1e-9:
                                mr = (s - xw * xv) / wr
                                vr = (s2 - xw * xv * xv) / wr - mr * mr
                                if vr > 0.0:
                                    tl = (mr - xv) / np.sqrt(vr)
                        if th <= 0.0 and tl <= 0.0:
                            break
                        if th >= tl:
                            xv = Hv[Y, X, c, ih]; xw = Hw[Y, X, c, ih]; ih += 1; t = th
                        else:
                            xv = Lv[Y, X, c, il]; xw = Lw[Y, X, c, il]; il += 1; t = tl
                        s -= xw * xv; s2 -= xw * xv * xv; w -= xw; n -= 1
                        cs[step + 1] = s; cs2[step + 1] = s2; cw[step + 1] = w
                        if t > crit[min(n, crit.shape[0] - 1)]:
                            best = step + 1           # Rosner: the largest passing prefix
                    removed += best
                    out[Y, X, c] = cs[best] / cw[best] if cw[best] > 1e-12 else 0.0
        return removed, considered


class CanvasStack:
    """Weighted mean of the frames over the canvas, in one pass. With
    rejection (clip_k truthy), each pixel's few most extreme samples are kept
    and tested at the end (generalized ESD, see above) where at least
    `min_frames` frames cover it; outliers are subtracted from the sums."""

    def __init__(self, shape, channels, clip_k=None, min_frames=4):
        from setiastro.saspro.memory_utils import smart_zeros
        H, W = shape
        C = int(channels)
        self.shape, self.C = (H, W), C
        self.k = clip_k
        self.min_frames = int(min_frames)
        self.S, _ = smart_zeros((H, W, C), dtype=np.float32)
        self.Wt, _ = smart_zeros((H, W), dtype=np.float32)
        self.N, _ = smart_zeros((H, W), dtype=np.float32)
        self.track = bool(clip_k) and _njit is not None
        if bool(clip_k) and _njit is None:
            print("[Seestar] numba unavailable: frames are averaged without outlier rejection")
        self.S2 = self.Hv = self.Hw = self.Lv = self.Lw = None
        if self.track:
            self.S2, _ = smart_zeros((H, W, C), dtype=np.float32)
            self.Hv, _ = smart_zeros((H, W, C, REJECT_HIGH), dtype=np.float32)
            self.Hw, _ = smart_zeros((H, W, C, REJECT_HIGH), dtype=np.float32)
            self.Lv, _ = smart_zeros((H, W, C, REJECT_LOW), dtype=np.float32)
            self.Lw, _ = smart_zeros((H, W, C, REJECT_LOW), dtype=np.float32)
            self.Hv[...] = -np.inf
            self.Lv[...] = np.inf
        self.rejected = 0
        self.considered = 0

    def add(self, prep):
        x0, y0, x1, y1 = prep["bbox"]
        v = prep["data"]; w = prep["weight"]
        if v.shape[2] < self.C:
            v = np.repeat(v[..., :1], self.C, axis=2)
        if _njit is not None:
            # np.asarray: a plain ndarray view even when the canvas is memory-mapped
            A = np.asarray
            v = np.ascontiguousarray(v[..., :self.C], dtype=np.float32)
            w = np.ascontiguousarray(w, dtype=np.float32)
            S = A(self.S)
            if self.track:
                _acc_one(S, A(self.S2), A(self.Wt), A(self.N), A(self.Hv), A(self.Hw),
                         A(self.Lv), A(self.Lw), v, w, int(y0), int(x0), True)
            else:
                d4 = np.zeros((1, 1, 1, 1), np.float32)
                _acc_one(S, S, A(self.Wt), A(self.N), d4, d4, d4, d4,
                         v, w, int(y0), int(x0), False)
            return
        sl = (slice(y0, y1), slice(x0, x1))
        self.S[sl] += w[..., None] * v[..., :self.C]
        self.Wt[sl] += w
        self.N[sl] += (w > 0)

    def result(self):
        """The stack: the weighted mean, with the outliers taken out."""
        H, W = self.shape
        if not self.track:
            out = np.zeros((H, W, self.C), np.float32)
            for y in range(0, H, 512):
                ys = slice(y, min(H, y + 512))
                wt = self.Wt[ys][..., None]
                out[ys] = np.where(wt > 0, self.S[ys] / np.maximum(wt, 1e-12), 0.0)
            return out
        from setiastro.saspro.live_stack_worker import max_outlier_crit
        nmax = int(np.max(self.N)) if self.N.size else 0
        crit = np.full(max(nmax, 2) + 1, np.inf)
        for n in range(max(2, REJECT_MIN_REST), nmax + 1):
            crit[n] = max_outlier_crit(n, REJECT_ALPHA / 2.0)
        A = np.asarray
        out = np.zeros((H, W, self.C), np.float32)
        r, c = _finish_one(A(self.S), A(self.S2), A(self.Wt), A(self.N), A(self.Hv), A(self.Hw),
                           A(self.Lv), A(self.Lw), crit, int(self.min_frames),
                           int(REJECT_MIN_REST), out)
        self.rejected, self.considered = int(r), int(c)
        return out
