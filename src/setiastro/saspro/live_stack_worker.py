# src/setiastro/saspro/live_stack_worker.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
GUI-free per-frame preparation for Live Stacking.

    load -> calibrate on the mosaic -> debayer -> star metrics (SEP, on a
    statistical-stretch copy) -> cull thresholds -> linear per-channel
    normalization -> star alignment to the reference frame

prepare_frame() runs in ProcessPoolExecutor children (spawn). A thread is not
enough: SEP, astroalign's Python parts and other steps hold the GIL, and the
GUI shares that lock with every thread of its process. In a worker process
they don't touch the GUI at all, and several frames can be prepared at once.
When worker processes are unavailable, Live Stacking's watch thread calls the
very same function in-process, so both paths give the same frame.

The window keeps the stacks. A worker only returns the aligned, normalized
frame and its measurements; the window weights and accumulates it, in file
order.

Import-light at module level (numpy only); a spawned child pulls in only
what preparing a frame needs. No PyQt widgets, no pyqtgraph.
"""
from __future__ import annotations

import os

import numpy as np

RAW_EXTS = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')


# ─────────────────────────────────────────────────────────────────────────────
# Small array helpers (also used by the window)
# ─────────────────────────────────────────────────────────────────────────────
def as_hwc(img) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    return a[..., None] if a.ndim == 2 else a


def linear_channel_stats(img):
    """Per-channel (floor, level) over valid pixels: floor = min, level =
    median(x - floor). Same semantics as the Stacking Suite's per-channel
    normalization targets. 0.0 / non-finite = no-data."""
    a = as_hwc(img)
    floors = np.zeros(a.shape[2], np.float32)
    levels = np.zeros(a.shape[2], np.float32)
    for c in range(a.shape[2]):
        band = a[..., c]
        v = band[np.isfinite(band) & (band != 0.0)]
        if v.size == 0:
            continue
        mn = float(v.min())
        sample = v[::7] if v.size > 2_000_000 else v   # a gain doesn't need the exact median
        floors[c] = mn
        levels[c] = float(np.median(sample)) - mn
    return floors, levels


def normalize_linear(img, targets):
    """Linear per-channel match to `targets` (the stack reference's levels):
    out = (x - floor) * target / level. No clipping, no stretch.
    Returns (out (H,W,C) float32, gains)."""
    a = as_hwc(img)
    floors, levels = linear_channel_stats(a)
    out = np.empty_like(a)
    gains = np.ones(a.shape[2], np.float32)
    for c in range(a.shape[2]):
        t = float(targets[c]) if c < len(targets) else 0.0
        g = t / float(levels[c]) if (levels[c] > 1e-12 and t > 0.0) else 1.0
        gains[c] = g
        band = a[..., c]
        o = (band - floors[c]) * np.float32(g)
        o[~(np.isfinite(band) & (band != 0.0))] = 0.0      # no-data stays no-data
        out[..., c] = o
    return out, gains


def max_outlier_crit(n_rest: int, alpha: float = 1e-3) -> float:
    """Critical deviation (in sigmas of the other samples) for the MAXIMUM
    of n_rest + 1 samples to be an outlier, at a false-removal rate of
    `alpha` per pixel.

    Testing a pixel's brightest sample against plain k sigma is a multiple
    comparison with sigma estimated from few frames: on pure noise it removed
    a sample from ~13% of pixels at 8 frames, k = 3, biasing the stack low.
    Bonferroni over the n + 1 candidates, Student t with n_rest - 1 degrees of
    freedom, scaled for predicting a new sample. Shared by Live Stacking and
    Mosaic Master's Seestar stacking."""
    n_rest = max(2, int(n_rest))
    df = n_rest - 1
    p = 1.0 - alpha / (n_rest + 1)
    try:
        from scipy.stats import t as _t
        q = float(_t.ppf(p, df))
    except Exception:
        from statistics import NormalDist
        z = NormalDist().inv_cdf(p)
        q = z + (z ** 3 + z) / (4.0 * df)          # Cornish-Fisher, first term
    return q * float(np.sqrt(1.0 + 1.0 / n_rest))


def match_master_to_image(master, img):
    """Coerce a master to the frame's dimensionality: a mono master on an RGB
    frame broadcasts (H,W,1); an RGB master on a mono frame is averaged."""
    if master is None:
        return None
    if img.ndim == 3 and master.ndim == 2:
        return master[..., None]
    if img.ndim == 2 and master.ndim == 3:
        return master.mean(axis=2)
    return master


def filter_key(header):
    """FITS FILTER -> 'L','R','G','B','H' (Ha),'O' (OIII),'S' (SII), or None."""
    try:
        fn = str(header.get('FILTER', '')).strip().upper()
    except Exception:
        return None
    if not fn:
        return None
    if fn in ('H', 'HA', 'HALPHA', 'H-ALPHA'):
        return 'H'
    if fn in ('O', 'O3', 'OIII'):
        return 'O'
    if fn in ('S', 'S2', 'SII'):
        return 'S'
    if fn in ('R', 'RED', 'RD'):
        return 'R'
    if fn in ('G', 'GREEN', 'GRN'):
        return 'G'
    if fn in ('B', 'BLUE', 'BL'):
        return 'B'
    if fn in ('L', 'LUM', 'LUMI', 'LUMINANCE'):
        return 'L'
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Camera RAW
# ─────────────────────────────────────────────────────────────────────────────
def raw_cfa(path: str):
    """('BAYER', 'RGGB'|'BGGR'|'GRBG'|'GBRG') | ('XTRANS', None) |
    ('UNKNOWN', None) for a camera RAW. The 2x2 token is read from
    raw_colors_visible, i.e. for exactly the mosaic load_image returns."""
    try:
        import rawpy
        from setiastro.saspro.stacking_suite import (
            _rawpy_is_xtrans_or_bayer, _rawpy_pattern_to_token)
        with rawpy.imread(path) as rp:
            fam = _rawpy_is_xtrans_or_bayer(rp)
            if fam == "XTRANS":
                return fam, None
            token = None
            try:
                cv = np.asarray(rp.raw_colors_visible)[:2, :2].ravel()
                desc = rp.color_desc
                letters = []
                for v in cv:
                    ch = desc[int(v)]
                    ch = chr(ch) if isinstance(ch, int) else str(ch)
                    letters.append(ch.upper()[:1])
                t = "".join(letters)
                token = t if t in ("RGGB", "BGGR", "GRBG", "GBRG") else None
            except Exception:
                token = None
            if token is None:
                token = _rawpy_pattern_to_token(rp)
            return ("BAYER" if token else fam), token
    except Exception:
        return "UNKNOWN", None


def xtrans_linear(path: str):
    """LibRaw's linear demosaic for X-Trans (as the Stacking Suite does)."""
    try:
        import rawpy
        with rawpy.imread(path) as rp:
            rgb16 = rp.postprocess(
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                no_auto_bright=True, gamma=(1.0, 1.0), output_bps=16,
                use_camera_wb=True,
                highlight_mode=rawpy.HighlightMode.Clip)
        return rgb16.astype(np.float32) / 65535.0
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Load + calibrate
# ─────────────────────────────────────────────────────────────────────────────
def load_calibrated(path, *, mono_color_mode=False, dark=None, flat=None, flats=None,
                    debayer_method: str = "rcd"):
    """Load one frame LINEAR, calibrated on the mosaic, then debayered.

    Returns (img, header, mono_key, why, notes): img (H,W) or (H,W,3)
    float32, or None with `why` when the frame can't be used. notes is a list
    of (tag, text) for the window to show once. Camera RAW goes through
    load_image like everywhere else (the Bayer mosaic), not LibRaw's
    gamma-encoded, white-balanced RGB."""
    from astropy.io import fits
    from setiastro.saspro.legacy.image_manager import load_image
    from setiastro.saspro.legacy.numba_utils import (
        apply_flat_division_numba, debayer_fits_fast)

    notes = []
    img, header, bit_depth, _is_mono = load_image(path)
    if img is None:
        return None, None, None, "could not be loaded", notes
    if "preview" in str(bit_depth).lower():
        return None, None, None, ("RAW decode failed; only the embedded JPEG "
                                  "preview was readable, which is not linear"), notes
    if header is None:
        header = fits.Header()
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]

    bayer = None
    calibrate = True
    if str(path).lower().endswith(RAW_EXTS) and img.ndim == 2:
        fam, token = raw_cfa(path)
        if fam == "XTRANS":
            rgb = xtrans_linear(path)
            if rgb is None:
                return None, None, None, "X-Trans RAW could not be demosaiced", notes
            img, calibrate = rgb, False
            notes.append(("xtrans", "X-Trans RAW uses LibRaw's linear demosaic; "
                                    "master dark/flat are not applied to X-Trans frames."))
        else:
            bayer = token
    elif img.ndim == 2:
        bp = header.get("BAYERPAT")
        if bp:
            bayer = str(bp[0] if isinstance(bp, tuple) else bp).strip().upper() or None

    mono_key = None
    if mono_color_mode and img.ndim == 2 and bayer is None and header.get("FILTER"):
        mono_key = filter_key(header)

    if calibrate:
        if dark is not None:
            md = match_master_to_image(dark, img).astype(np.float32, copy=False)
            img = img - md
        flats = flats or {}
        if mono_key and mono_key in flats:
            mf = match_master_to_image(flats[mono_key], img).astype(np.float32, copy=False)
            img = apply_flat_division_numba(img, mf)
        elif flat is not None:
            mf = match_master_to_image(flat, img).astype(np.float32, copy=False)
            img = apply_flat_division_numba(img, mf)

    if bayer:
        img = debayer_fits_fast(img, bayer, method=str(debayer_method or "rcd").lower())
    return np.asarray(img, dtype=np.float32), header, mono_key, None, notes


# ─────────────────────────────────────────────────────────────────────────────
# Star metrics (Blink's SEP settings)
# ─────────────────────────────────────────────────────────────────────────────
def star_metrics(img, thresh_sigma: float = 7.0, minarea: int = 16,
                 deblend_nthresh: int = 32):
    """(star_count, fwhm, ecc) with Blink's measurement (measure_stars_sep):
    SEP background, global RMS as the error, the same threshold, minimum area
    and retry ladder, median FWHM = 2.3548*sqrt(a*b), ecc = sqrt(1-(b/a)^2).
    Extraction goes through sep_tiled.extract_auto, which strips a dense
    field (same catalogue, faster) in a worker process.
    (0, nan, nan) when no stars; (0, 30, 1) if SEP fails, as Blink."""
    try:
        import sep
        from setiastro.saspro.sep_tiled import extract_auto

        data = np.asarray(img, dtype=np.float32)
        if data.ndim != 2 or data.size == 0:
            return 0, np.nan, np.nan
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
        data = np.ascontiguousarray(data)
        bkg = sep.Background(data)
        sub = np.ascontiguousarray(data - bkg.back())
        grms = float(getattr(bkg, "globalrms", np.nan))
        if not np.isfinite(grms) or grms <= 0:
            try:
                grms = float(np.nanmedian(np.asarray(bkg.rms(), dtype=np.float32)))
            except Exception:
                grms = np.nan
        if not np.isfinite(grms) or grms <= 0:
            grms = 1e-6

        ladder = [
            (float(thresh_sigma), int(minarea)),
            (5.0, max(2, int(minarea * 0.75))),
            (4.0, max(2, int(minarea * 0.50))),
            (3.5, max(2, int(minarea * 0.35))),
        ]
        cat = None
        for thr, ma in ladder:
            try:
                cat = extract_auto(sep, sub, thr, err=grms, minarea=ma, clean=True,
                                   deblend_nthresh=int(deblend_nthresh))
            except Exception:
                cat = None
            if cat is not None and len(cat) > 0:
                break
        if cat is None or len(cat) == 0:
            return 0, np.nan, np.nan

        a = np.clip(cat["a"].astype(np.float32, copy=False), 1e-6, None)
        b = np.clip(cat["b"].astype(np.float32, copy=False), 1e-6, None)
        fwhm_vals = 2.3548 * np.sqrt(a * b)
        ratios = np.clip(b / a, 0.0, 1.0)
        ecc_vals = np.sqrt(np.maximum(0.0, 1.0 - ratios * ratios))
        return int(len(cat)), float(np.nanmedian(fwhm_vals)), float(np.nanmedian(ecc_vals))
    except Exception:
        return 0, 30, 1


# Thresholds the star-count waterfall tries, highest first, with the minimum
# area at each: Blink's 16 px down to 7 sigma (these are measured on a
# statistically stretched copy), then smaller areas as the old retry did.
STAR_LADDER = ((100.0, 16), (50.0, 16), (25.0, 16), (12.0, 16), (7.0, 16),
               (5.0, 12), (4.0, 8), (3.5, 6))
STAR_TARGET = 100


def _shape_stats(cat):
    if cat is None or len(cat) == 0:
        return 0, float("nan"), float("nan")
    a = np.clip(cat["a"].astype(np.float32, copy=False), 1e-6, None)
    b = np.clip(cat["b"].astype(np.float32, copy=False), 1e-6, None)
    ratios = np.clip(b / a, 0.0, 1.0)
    return (int(len(cat)), float(np.nanmedian(2.3548 * np.sqrt(a * b))),
            float(np.nanmedian(np.sqrt(np.maximum(0.0, 1.0 - ratios * ratios)))))


def star_metrics_auto(img, sigma=None, target: int = STAR_TARGET,
                      deblend_nthresh: int = 32) -> dict:
    """Star count, median FWHM and eccentricity with the detection threshold
    chosen by count: the highest threshold of STAR_LADDER that finds at least
    `target` stars (sep_tiled.waterfall_extract). Given `sigma`, measured at
    that threshold only, so frames of one stack are measured alike and the
    star count stays a fair comparison between them (a cloudy frame then
    shows FEWER stars instead of a lower threshold).

    Returns {sc, fwhm, ecc, sigma, rungs: {sigma: (sc, fwhm, ecc)}} -- rungs
    holds every threshold tried. sigma is None if SEP failed (sc 0)."""
    out = {"sc": 0, "fwhm": float("nan"), "ecc": float("nan"), "sigma": None, "rungs": {}}
    try:
        import sep
        from setiastro.saspro.sep_tiled import waterfall_extract

        data = np.asarray(img, dtype=np.float32)
        if data.ndim != 2 or data.size == 0:
            return out
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
        r = waterfall_extract(sep, data, int(target), ladder=STAR_LADDER, fixed=sigma,
                              clean=True, deblend_nthresh=int(deblend_nthresh))
        if r is None:
            return out
        out["rungs"] = {float(s_): _shape_stats(c_) for s_, c_ in r["rungs"]}
        sc, fwhm, ecc = out["rungs"][float(r["sigma"])]
        out.update(sc=sc, fwhm=fwhm, ecc=ecc, sigma=float(r["sigma"]))
    except Exception:
        pass
    return out


def cull_reason(sc, fwhm, ecc, max_fwhm, max_ecc, min_stars):
    """Why the user's cull thresholds reject a frame, or None."""
    if fwhm > float(max_fwhm):
        return f"FWHM {fwhm:.1f} px > {float(max_fwhm):.1f}"
    if ecc > float(max_ecc):
        return f"eccentricity {ecc:.2f} > {float(max_ecc):.2f}"
    if sc < int(min_stars):
        return f"{sc} stars < {int(min_stars)}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker process plumbing
# ─────────────────────────────────────────────────────────────────────────────
# Masters and the reference frame reach a worker as .npy files written once
# by the window; each worker loads a file once and keeps it (keyed by path,
# mtime and size). Loaded whole, not memory-mapped, so the window can delete
# a replaced file at any time.
_ARR_CACHE: dict = {}
_ARR_CACHE_MAX = 8


def _arr(v):
    if v is None or isinstance(v, np.ndarray):
        return v
    st = os.stat(v)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _ARR_CACHE.get(v)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    a = np.load(v)
    _ARR_CACHE[v] = (stamp, a)
    while len(_ARR_CACHE) > _ARR_CACHE_MAX:
        _ARR_CACHE.pop(next(iter(_ARR_CACHE)))
    return a


def init_worker(org: str = "", app: str = ""):
    """Process initializer: give QSettings the app's identity so settings read
    in a worker (e.g. the GPU star-detection switches) match the GUI's."""
    try:
        from PyQt6.QtCore import QCoreApplication
        if org:
            QCoreApplication.setOrganizationName(org)
        if app:
            QCoreApplication.setApplicationName(app)
    except Exception:
        pass


def warm():
    """Import the heavy modules up front so the first frame isn't slow."""
    import sep  # noqa: F401
    from setiastro.saspro.legacy.image_manager import load_image  # noqa: F401
    from setiastro.saspro.imageops.stretch import stretch_mono_image  # noqa: F401
    from setiastro.saspro.star_alignment import StarRegistrationWorker  # noqa: F401
    return os.getpid()


# ─────────────────────────────────────────────────────────────────────────────
# One frame
# ─────────────────────────────────────────────────────────────────────────────
def prepare_frame(task: dict, should_stop=None) -> dict:
    """Prepare one frame for the stack. `task` (all picklable):

      path, mono_color_mode, debayer_method, star_target, star_sigmas {key:
      the stack's detection threshold, once chosen}, max_fwhm, max_ecc,
      min_star_count, dark, flat, flats {key: master}, ref (the reference
      plane), targets {key: per-channel levels}. Masters and ref are arrays
      or .npy paths.

    Returns a dict with status:
      "error"   - not usable (why)
      "cull"    - measured, rejected by a threshold or alignment (why)
      "ok"      - frame: aligned, normalized (H,W,C) float32; targets_used
                  and targets_own (True if this stack had no levels yet and the
                  frame was normalized to its own; the window rescales it to
                  the stack's levels if another frame set them first);
                  aligned False when there was no reference yet (this frame
                  becomes the reference)
      "stopped" - should_stop() was set (in-process only)
    plus path, pid, notes, and when measured: key, header, sc, fwhm, ecc, bg,
    star_sigma (the threshold used) and star_rungs (see star_metrics_auto).
    """
    stop = should_stop or (lambda: False)
    path = task["path"]
    res = {"path": path, "pid": os.getpid(), "notes": []}

    img, header, mono_key, why, notes = load_calibrated(
        path, mono_color_mode=bool(task.get("mono_color_mode")),
        dark=_arr(task.get("dark")), flat=_arr(task.get("flat")),
        flats={k: _arr(v) for k, v in (task.get("flats") or {}).items()},
        debayer_method=task.get("debayer_method", "rcd"))
    res["notes"] = notes
    if img is None:
        res.update(status="error", why=why)
        return res
    if stop():
        res["status"] = "stopped"
        return res

    from setiastro.saspro.imageops.stretch import stretch_mono_image

    lin = as_hwc(img)
    del img
    key = mono_key or "__main__"
    plane = lin[..., 0] if lin.shape[2] == 1 else lin.mean(axis=2)

    # Star metrics measured as Blink measures them: on a statistical-stretch
    # COPY (SEP's 16 px area is tuned on stretched data; on linear data small
    # stars fall under it). The stack stays linear. The threshold is found
    # by star count on the stack's first frames, then held (star_sigmas).
    bg = float(np.median(plane[::4, ::4]))
    m = star_metrics_auto(stretch_mono_image(plane, target_median=0.3),
                          sigma=(task.get("star_sigmas") or {}).get(key),
                          target=int(task.get("star_target", STAR_TARGET)))
    sc, fwhm, ecc = m["sc"], m["fwhm"], m["ecc"]
    del plane
    res.update(key=key, header=header, sc=int(sc), fwhm=float(fwhm), ecc=float(ecc),
               bg=bg, shape=tuple(lin.shape), star_sigma=m["sigma"], star_rungs=m["rungs"])
    if stop():
        res["status"] = "stopped"
        return res

    # The user's cull thresholds (safety net)
    reason = cull_reason(sc, fwhm, ecc, task.get("max_fwhm", 15.0),
                         task.get("max_ecc", 0.9), task.get("min_star_count", 5))
    if reason is not None:
        res.update(status="cull", why=reason)
        return res

    # Linear per-channel normalization to this stack's levels
    targets = (task.get("targets") or {}).get(key)
    own = targets is None
    if own:
        _floors, targets = linear_channel_stats(lin)
    norm, _gains = normalize_linear(lin, targets)
    del lin
    res["targets_used"] = np.asarray(targets, np.float32)
    res["targets_own"] = bool(own)
    if stop():
        res["status"] = "stopped"
        return res

    # Align to the reference frame. A frame that can't be placed is culled:
    # stacking it unaligned would ghost every star.
    ref = _arr(task.get("ref"))
    if ref is None:
        res.update(status="ok", frame=norm, aligned=False)
        return res
    nplane = norm[..., 0] if norm.shape[2] == 1 else norm.mean(axis=2)
    if ref.shape != nplane.shape:
        res.update(status="cull", why="size differs from the reference frame")
        return res
    from setiastro.saspro.star_alignment import StarRegistrationWorker, StarRegistrationThread
    delta = StarRegistrationWorker.compute_affine_transform_astroalign(nplane, ref)
    if delta is None:
        res.update(status="cull", why="could not be aligned to the reference frame")
        return res
    aligned = as_hwc(StarRegistrationThread.apply_affine_transform_static(
        norm[..., 0] if norm.shape[2] == 1 else norm, delta))
    res.update(status="ok", frame=np.ascontiguousarray(aligned, dtype=np.float32), aligned=True)
    return res
