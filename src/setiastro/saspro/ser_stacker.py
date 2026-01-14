# src/setiastro/saspro/ser_stacker.py
from __future__ import annotations
import os
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

import cv2
cv2.setNumThreads(1)

from setiastro.saspro.imageops.serloader import SERReader
from setiastro.saspro.ser_stack_config import SERStackConfig
from setiastro.saspro.ser_tracking import PlanetaryTracker, SurfaceTracker, _to_mono01
from setiastro.saspro.imageops.serloader import open_planetary_source, PlanetaryFrameSource

_BAYER_TO_CV2 = {
    "RGGB": cv2.COLOR_BayerRG2RGB,
    "BGGR": cv2.COLOR_BayerBG2RGB,
    "GRBG": cv2.COLOR_BayerGR2RGB,
    "GBRG": cv2.COLOR_BayerGB2RGB,
}

def _cfg_bayer_pattern(cfg) -> str | None:
    # cfg.bayer_pattern might be missing in older saved projects; be defensive
    return getattr(cfg, "bayer_pattern", None)


def _get_frame(src, idx: int, *, roi, debayer: bool, to_float01: bool, force_rgb: bool, bayer_pattern: str | None):
    """
    Drop-in wrapper:
    - passes cfg.bayer_pattern down to sources that support it
    - stays compatible with sources whose get_frame() doesn't accept bayer_pattern yet
    """
    try:
        return src.get_frame(
            int(idx),
            roi=roi,
            debayer=debayer,
            to_float01=to_float01,
            force_rgb=force_rgb,
            bayer_pattern=bayer_pattern,
        )
    except TypeError:
        # Back-compat: older PlanetaryFrameSource implementations
        return src.get_frame(
            int(idx),
            roi=roi,
            debayer=debayer,
            to_float01=to_float01,
            force_rgb=force_rgb,
        )


@dataclass
class AnalyzeResult:
    frames_total: int
    roi_used: Optional[Tuple[int, int, int, int]]
    track_mode: str
    quality: np.ndarray        # (N,) float32 higher=better
    dx: np.ndarray             # (N,) float32
    dy: np.ndarray             # (N,) float32
    conf: np.ndarray           # (N,) float32 0..1  (final conf used by stacking)
    order: np.ndarray          # (N,) int indices sorted by quality desc
    ref_mode: str              # "best_frame" | "best_stack"
    ref_count: int
    ref_image: np.ndarray      # float32 [0..1], ROI-sized
    ap_centers: Optional[np.ndarray] = None  # (M,2) int32 in ROI coords
    ap_size: int = 64
    ap_multiscale: bool = False

    # ✅ NEW: surface anchor confidence (coarse tracker)
    coarse_conf: Optional[np.ndarray] = None  # (N,) float32 0..1
 

@dataclass
class FrameEval:
    idx: int
    score: float
    dx: float
    dy: float
    conf: float

def _print_surface_debug(
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    conf: np.ndarray,
    coarse_conf: np.ndarray | None,
    floor: float = 0.05,
    prefix: str = "[SER][Surface]"
) -> None:
    try:
        dx = np.asarray(dx, dtype=np.float32)
        dy = np.asarray(dy, dtype=np.float32)
        conf = np.asarray(conf, dtype=np.float32)

        dx_min = float(np.min(dx)) if dx.size else 0.0
        dx_max = float(np.max(dx)) if dx.size else 0.0
        dy_min = float(np.min(dy)) if dy.size else 0.0
        dy_max = float(np.max(dy)) if dy.size else 0.0

        conf_mean = float(np.mean(conf)) if conf.size else 0.0
        conf_min = float(np.min(conf)) if conf.size else 0.0

        msg = (
            f"{prefix} dx[min,max]=({dx_min:.2f},{dx_max:.2f})  "
            f"dy[min,max]=({dy_min:.2f},{dy_max:.2f})  "
            f"conf[mean,min]=({conf_mean:.3f},{conf_min:.3f})"
        )

        if coarse_conf is not None:
            cc = np.asarray(coarse_conf, dtype=np.float32)
            cc_mean = float(np.mean(cc)) if cc.size else 0.0
            cc_min = float(np.min(cc)) if cc.size else 0.0
            cc_bad = float(np.mean(cc < 0.2)) if cc.size else 0.0
            msg += f"  coarse_conf[mean,min]=({cc_mean:.3f},{cc_min:.3f})  frac<0.2={cc_bad:.2%}"

        if conf_mean <= floor + 1e-6:
            msg += f"  ⚠ conf.mean near floor ({floor}); alignment likely failing"
        print(msg)
    except Exception as e:
        print(f"{prefix} debug print failed: {e}")


def _clamp_roi_in_bounds(roi: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, rw, rh = [int(v) for v in roi]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    rw = max(1, min(w - x, rw))
    rh = max(1, min(h - y, rh))
    return x, y, rw, rh

def _grad_img(m: np.ndarray) -> np.ndarray:
    """Simple, robust edge image for SSD refine."""
    m = m.astype(np.float32, copy=False)
    if cv2 is None:
        # fallback: finite differences
        gx = np.zeros_like(m); gx[:, 1:] = m[:, 1:] - m[:, :-1]
        gy = np.zeros_like(m); gy[1:, :] = m[1:, :] - m[:-1, :]
        g = np.abs(gx) + np.abs(gy)
        g -= float(g.mean())
        s = float(g.std()) + 1e-6
        return g / s

    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    g -= float(g.mean())
    s = float(g.std()) + 1e-6
    return (g / s).astype(np.float32, copy=False)

def _ssd_prepare_ref(ref_m: np.ndarray, crop: float = 0.80):
    """
    Precompute reference gradient + crop window once.

    Returns:
      rg   : full reference gradient image (float32)
      rgc  : cropped view of rg
      sl   : (y0,y1,x0,x1) crop slices
    """
    ref_m = ref_m.astype(np.float32, copy=False)
    rg = _grad_img(ref_m)  # compute ONCE

    H, W = rg.shape[:2]
    cfx = max(8, int(W * (1.0 - float(crop)) * 0.5))
    cfy = max(8, int(H * (1.0 - float(crop)) * 0.5))
    x0, x1 = cfx, W - cfx
    y0, y1 = cfy, H - cfy

    rgc = rg[y0:y1, x0:x1]  # view
    return rg, rgc, (y0, y1, x0, x1)

def _subpixel_quadratic_1d(vm: float, v0: float, vp: float) -> float:
    """
    Given SSD at (-1,0,+1): (vm, v0, vp), return vertex offset in [-0.5,0.5]-ish.
    Works for minimizing SSD.
    """
    denom = (vm - 2.0 * v0 + vp)
    if abs(denom) < 1e-12:
        return 0.0
    # vertex of parabola fit through -1,0,+1
    t = 0.5 * (vm - vp) / denom
    return float(np.clip(t, -0.75, 0.75))


def _ssd_confidence_prepared(
    rgc: np.ndarray,
    cgc0: np.ndarray,
    dx_i: int,
    dy_i: int,
) -> float:
    """
    Compute SSD between rgc and cgc0 shifted by (dx_i,dy_i) using slicing overlap.
    Returns SSD (lower is better).

    NOTE: This is integer-only and extremely fast (no warps).
    """
    H, W = rgc.shape[:2]

    # Overlap slices for rgc and shifted cgc0
    x0r = max(0, dx_i)
    x1r = min(W, W + dx_i)
    y0r = max(0, dy_i)
    y1r = min(H, H + dy_i)

    x0c = max(0, -dx_i)
    x1c = min(W, W - dx_i)
    y0c = max(0, -dy_i)
    y1c = min(H, H - dy_i)

    rr = rgc[y0r:y1r, x0r:x1r]
    cc = cgc0[y0c:y1c, x0c:x1c]

    d = rr - cc
    return float(np.mean(d * d))


def _ssd_confidence(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    dx: float,
    dy: float,
    *,
    crop: float = 0.80,
) -> float:
    """
    Original API: confidence from gradient SSD, higher=better (0..1).

    Optimized:
      - computes ref grad once per call (still OK if used standalone)
      - uses one warp for (dx,dy)
      - no extra work beyond necessary

    For iterative search, use _refine_shift_ssd() which avoids redoing work.
    """
    ref_m = ref_m.astype(np.float32, copy=False)
    cur_m = cur_m.astype(np.float32, copy=False)

    # shift current by the proposed shift
    cur_s = _shift_image(cur_m, float(dx), float(dy))

    rg, rgc, sl = _ssd_prepare_ref(ref_m, crop=crop)
    y0, y1, x0, x1 = sl

    cg = _grad_img(cur_s)
    cgc = cg[y0:y1, x0:x1]

    d = rgc - cgc
    ssd = float(np.mean(d * d))

    scale = 0.002
    conf = float(np.exp(-ssd / max(1e-12, scale)))
    return float(np.clip(conf, 0.0, 1.0))


def _refine_shift_ssd(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    dx0: float,
    dy0: float,
    *,
    radius: int = 10,
    crop: float = 0.80,
    bruteforce: bool = False,
    max_steps: int | None = None,
) -> tuple[float, float, float]:
    """
    Returns (dx_refine, dy_refine, conf) where you ADD refine to (dx0,dy0).

    CPU-optimized:
      - precompute ref gradient crop once
      - apply (dx0,dy0) shift ONCE
      - compute gradient ONCE for shifted cur
      - evaluate integer candidates via slicing overlap SSD (no warps)

    If bruteforce=True, does full window scan in [-r,r]^2 (fast).
    Otherwise does 8-neighbor hill-climb over integer offsets (very fast).

    Optional subpixel polish:
      - after choosing best integer (best_dx,best_dy), do a tiny separable quadratic
        fit along x and y using SSD at +/-1 around the best integer.
      - does NOT require any new gradients/warps (just 4 extra SSD evals).
    """
    r = int(max(0, radius))
    if r == 0:
        # nothing to do; just compute confidence at dx0/dy0
        c = _ssd_confidence(ref_m, cur_m, dx0, dy0, crop=crop)
        return 0.0, 0.0, float(c)

    # Prepare ref grad crop ONCE
    _, rgc, sl = _ssd_prepare_ref(ref_m, crop=crop)
    y0, y1, x0, x1 = sl

    # Shift cur by the current estimate ONCE, then gradient ONCE
    cur_m = cur_m.astype(np.float32, copy=False)
    cur0 = _shift_image(cur_m, float(dx0), float(dy0))
    cg0 = _grad_img(cur0)
    cgc0 = cg0[y0:y1, x0:x1]

    # Helper: parabola vertex for minimizing SSD, using (-1,0,+1) samples
    def _quad_min_offset(vm: float, v0: float, vp: float) -> float:
        denom = (vm - 2.0 * v0 + vp)
        if abs(denom) < 1e-12:
            return 0.0
        t = 0.5 * (vm - vp) / denom
        return float(np.clip(t, -0.75, 0.75))

    if bruteforce:
        # NOTE: your bruteforce path currently includes a subpixel step already.
        # If you want to keep using that exact implementation, just call it:
        dxr, dyr, conf = _refine_shift_ssd_bruteforce(ref_m, cur_m, dx0, dy0, radius=r, crop=crop)
        return float(dxr), float(dyr), float(conf)

    # Hill-climb in integer space minimizing SSD
    if max_steps is None:
        max_steps = max(1, min(r, 6))  # small cap helps speed; tune if you want

    best_dx = 0
    best_dy = 0
    best_ssd = _ssd_confidence_prepared(rgc, cgc0, 0, 0)

    neigh = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))

    for _ in range(int(max_steps)):
        improved = False
        for sx, sy in neigh:
            cand_dx = best_dx + sx
            cand_dy = best_dy + sy
            if abs(cand_dx) > r or abs(cand_dy) > r:
                continue

            ssd = _ssd_confidence_prepared(rgc, cgc0, cand_dx, cand_dy)
            if ssd < best_ssd:
                best_ssd = ssd
                best_dx = cand_dx
                best_dy = cand_dy
                improved = True

        if not improved:
            break

    # ---- subpixel quadratic polish around best integer (cheap) ----
    # Uses SSD at +/-1 around best integer in X and Y (separable).
    dx_sub = 0.0
    dy_sub = 0.0
    if r >= 1:
        # X samples at (best_dx-1, best_dy), (best_dx, best_dy), (best_dx+1, best_dy)
        if abs(best_dx - 1) <= r:
            s_xm = _ssd_confidence_prepared(rgc, cgc0, best_dx - 1, best_dy)
        else:
            s_xm = best_ssd
        s_x0 = best_ssd
        if abs(best_dx + 1) <= r:
            s_xp = _ssd_confidence_prepared(rgc, cgc0, best_dx + 1, best_dy)
        else:
            s_xp = best_ssd
        dx_sub = _quad_min_offset(s_xm, s_x0, s_xp)

        # Y samples at (best_dx, best_dy-1), (best_dx, best_dy), (best_dx, best_dy+1)
        if abs(best_dy - 1) <= r:
            s_ym = _ssd_confidence_prepared(rgc, cgc0, best_dx, best_dy - 1)
        else:
            s_ym = best_ssd
        s_y0 = best_ssd
        if abs(best_dy + 1) <= r:
            s_yp = _ssd_confidence_prepared(rgc, cgc0, best_dx, best_dy + 1)
        else:
            s_yp = best_ssd
        dy_sub = _quad_min_offset(s_ym, s_y0, s_yp)

    best_dx_f = float(best_dx) + float(dx_sub)
    best_dy_f = float(best_dy) + float(dy_sub)

    # Confidence: keep based on best *integer* SSD (no subpixel warp needed)
    scale = 0.002
    conf = float(np.exp(-best_ssd / max(1e-12, scale)))
    conf = float(np.clip(conf, 0.0, 1.0))

    return float(best_dx_f), float(best_dy_f), float(conf)



def _refine_shift_ssd_bruteforce(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    dx0: float,
    dy0: float,
    *,
    radius: int = 2,
    crop: float = 0.80,
) -> tuple[float, float, float]:
    """
    Full brute-force scan in [-radius,+radius]^2, but optimized:
      - shift by (dx0,dy0) ONCE
      - compute gradients ONCE
      - evaluate candidates via slicing overlap SSD (no warps)
      - keep your separable quadratic subpixel fit
    """
    ref_m = ref_m.astype(np.float32, copy=False)
    cur_m = cur_m.astype(np.float32, copy=False)

    r = int(max(0, radius))
    if r == 0:
        c = _ssd_confidence(ref_m, cur_m, dx0, dy0, crop=crop)
        return 0.0, 0.0, float(c)

    # Apply current estimate once
    cur0 = _shift_image(cur_m, float(dx0), float(dy0))

    # Gradients once
    rg = _grad_img(ref_m)
    cg0 = _grad_img(cur0)

    H, W = rg.shape[:2]
    cfx = max(8, int(W * (1.0 - float(crop)) * 0.5))
    cfy = max(8, int(H * (1.0 - float(crop)) * 0.5))
    x0, x1 = cfx, W - cfx
    y0, y1 = cfy, H - cfy

    rgc = rg[y0:y1, x0:x1]
    cgc0 = cg0[y0:y1, x0:x1]

    # brute-force integer search
    best = (0, 0)
    best_ssd = float("inf")
    ssds: dict[tuple[int, int], float] = {}

    for j in range(-r, r + 1):
        for i in range(-r, r + 1):
            ssd = _ssd_confidence_prepared(rgc, cgc0, i, j)
            ssds[(i, j)] = ssd
            if ssd < best_ssd:
                best_ssd = ssd
                best = (i, j)

    bx, by = best

    # Subpixel quadratic fit (separable) if neighbors exist
    def _quad_peak(vm, v0, vp):
        denom = (vm - 2.0 * v0 + vp)
        if abs(denom) < 1e-12:
            return 0.0
        return 0.5 * (vm - vp) / denom

    dx_sub = 0.0
    dy_sub = 0.0
    if (bx - 1, by) in ssds and (bx + 1, by) in ssds:
        dx_sub = _quad_peak(ssds[(bx - 1, by)], ssds[(bx, by)], ssds[(bx + 1, by)])
    if (bx, by - 1) in ssds and (bx, by + 1) in ssds:
        dy_sub = _quad_peak(ssds[(bx, by - 1)], ssds[(bx, by)], ssds[(bx, by + 1)])

    dxr = float(bx + np.clip(dx_sub, -0.75, 0.75))
    dyr = float(by + np.clip(dy_sub, -0.75, 0.75))

    # Confidence: use your “sharpness” idea (median neighbor vs best)
    neigh = [v for (k, v) in ssds.items() if k != (bx, by)]
    neigh_med = float(np.median(np.asarray(neigh, np.float32))) if neigh else best_ssd
    sharp = max(0.0, neigh_med - best_ssd)
    conf = float(np.clip(sharp / max(1e-6, neigh_med), 0.0, 1.0))

    return dxr, dyr, conf

def _bandpass(m: np.ndarray) -> np.ndarray:
    """Illumination-robust image for tracking (float32)."""
    m = m.astype(np.float32, copy=False)

    # remove large-scale illumination (terminator gradient)
    lo = cv2.GaussianBlur(m, (0, 0), 6.0)
    hi = cv2.GaussianBlur(m, (0, 0), 1.2)
    bp = hi - lo

    # normalize
    bp -= float(bp.mean())
    s = float(bp.std()) + 1e-6
    bp = bp / s

    # window to reduce FFT edge artifacts
    hann_y = np.hanning(bp.shape[0]).astype(np.float32)
    hann_x = np.hanning(bp.shape[1]).astype(np.float32)
    bp *= (hann_y[:, None] * hann_x[None, :])

    return bp

def _reject_ap_outliers(ap_dx: np.ndarray, ap_dy: np.ndarray, ap_cf: np.ndarray, *, z: float = 3.5) -> np.ndarray:
    """
    Return a boolean mask of APs to keep based on MAD distance from median.
    """
    dx = np.asarray(ap_dx, np.float32)
    dy = np.asarray(ap_dy, np.float32)
    cf = np.asarray(ap_cf, np.float32)

    good = cf > 0.15
    if not np.any(good):
        return good

    dxg = dx[good]
    dyg = dy[good]

    mx = float(np.median(dxg))
    my = float(np.median(dyg))

    rx = np.abs(dxg - mx)
    ry = np.abs(dyg - my)

    madx = float(np.median(rx)) + 1e-6
    mady = float(np.median(ry)) + 1e-6

    zx = rx / madx
    zy = ry / mady

    keep_g = (zx < z) & (zy < z)
    keep = np.zeros_like(good)
    keep_idx = np.where(good)[0]
    keep[keep_idx] = keep_g
    return keep


def _coarse_surface_ref_locked(
    source_obj,
    *,
    n: int,
    roi,
    roi_used=None,
    debayer: bool,
    to_rgb: bool,
    bayer_pattern: Optional[str] = None,
    progress_cb=None,
    progress_every: int = 25,
    # tuning:
    down: int = 2,
    template_size: int = 256,
    search_radius: int = 96,
    bandpass: bool = True,
    # ✅ NEW: parallel coarse
    workers: int | None = None,
    stride: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Surface coarse tracking that DOES NOT DRIFT:
    - Locks to frame0 reference (in roi=roi_track coords).
    - Uses NCC + subpixel phaseCorr.
    - Optional parallelization by chunking time into segments of length=stride.
      Each segment runs sequentially (keeps pred window), segments run in parallel.
    """
    if cv2 is None:
        dx = np.zeros((n,), np.float32)
        dy = np.zeros((n,), np.float32)
        cc = np.ones((n,), np.float32)
        return dx, dy, cc

    dx = np.zeros((n,), dtype=np.float32)
    dy = np.zeros((n,), dtype=np.float32)
    cc = np.zeros((n,), dtype=np.float32)

    def _downN(m: np.ndarray) -> np.ndarray:
        if down <= 1:
            return m.astype(np.float32, copy=False)
        H, W = m.shape[:2]
        return cv2.resize(
            m,
            (max(2, W // down), max(2, H // down)),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32, copy=False)

    def _pick_anchor_center_ds(W: int, H: int) -> tuple[int, int]:
        cx = W // 2
        cy = H // 2
        if roi_used is None or roi is None:
            return int(cx), int(cy)
        try:
            xt, yt, wt, ht = [int(v) for v in roi]
            xu, yu, wu, hu = [int(v) for v in roi_used]
            cux = xu + (wu * 0.5)
            cuy = yu + (hu * 0.5)
            cx_full = cux - xt
            cy_full = cuy - yt
            cx = int(round(cx_full / max(1, int(down))))
            cy = int(round(cy_full / max(1, int(down))))
            cx = max(0, min(W - 1, cx))
            cy = max(0, min(H - 1, cy))
        except Exception:
            pass
        return int(cx), int(cy)

    # ---------------------------
    # Prep ref/template once
    # ---------------------------
    src0, owns0 = _ensure_source(source_obj, cache_items=2)
    try:
        img0 = _get_frame(src0, 0, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb), bayer_pattern=bayer_pattern)

        ref0 = _to_mono01(img0).astype(np.float32, copy=False)
        ref0 = _downN(ref0)
        ref0p = _bandpass(ref0) if bandpass else (ref0 - float(ref0.mean()))

        H, W = ref0p.shape[:2]
        ts = int(max(64, min(template_size, min(H, W) - 4)))
        half = ts // 2

        cx0, cy0 = _pick_anchor_center_ds(W, H)
        rx0 = max(0, min(W - ts, cx0 - half))
        ry0 = max(0, min(H - ts, cy0 - half))
        ref_t = ref0p[ry0:ry0 + ts, rx0:rx0 + ts].copy()
    finally:
        if owns0:
            try:
                src0.close()
            except Exception:
                pass

    dx[0] = 0.0
    dy[0] = 0.0
    cc[0] = 1.0

    if progress_cb:
        progress_cb(0, n, "Surface: coarse (ref-locked NCC+subpix)…")

    # If no workers requested (or too small), fall back to sequential
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = max(1, min(cpu, 48))
    workers = int(max(1, workers))
    stride = int(max(4, stride))

    # ---------------------------
    # Core "one frame" matcher
    # ---------------------------
    def _match_one(curp: np.ndarray, pred_x: float, pred_y: float, r: int) -> tuple[float, float, float, float, float]:
        # returns (mx_ds, my_ds, dx_full, dy_full, conf)
        x0 = int(max(0, min(W - 1, pred_x - r)))
        y0 = int(max(0, min(H - 1, pred_y - r)))
        x1 = int(min(W, pred_x + r + ts))
        y1 = int(min(H, pred_y + r + ts))

        win = curp[y0:y1, x0:x1]
        if win.shape[0] < ts or win.shape[1] < ts:
            return float(pred_x), float(pred_y), 0.0, 0.0, 0.0

        res = cv2.matchTemplate(win, ref_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        conf_ncc = float(np.clip(max_val, 0.0, 1.0))

        mx_ds = float(x0 + max_loc[0])
        my_ds = float(y0 + max_loc[1])

        # subpix refine on the matched patch
        mx_i = int(round(mx_ds))
        my_i = int(round(my_ds))
        cur_t = curp[my_i:my_i + ts, mx_i:mx_i + ts]
        if cur_t.shape == ref_t.shape:
            (sdx, sdy), resp = cv2.phaseCorrelate(ref_t.astype(np.float32), cur_t.astype(np.float32))
            sub_dx = float(sdx)
            sub_dy = float(sdy)
            conf_pc = float(np.clip(resp, 0.0, 1.0))
        else:
            sub_dx = 0.0
            sub_dy = 0.0
            conf_pc = 0.0

        dx_ds = float(rx0 - mx_ds) + sub_dx
        dy_ds = float(ry0 - my_ds) + sub_dy
        dx_full = float(dx_ds * down)
        dy_full = float(dy_ds * down)

        conf = float(np.clip(0.65 * conf_ncc + 0.35 * conf_pc, 0.0, 1.0))
        return float(mx_ds), float(my_ds), dx_full, dy_full, conf

    # ---------------------------
    # Keyframe boundary pass (sequential)
    # ---------------------------
    boundaries = list(range(0, n, stride))
    start_pred = {}  # b -> (pred_x, pred_y)
    start_pred[0] = (float(rx0), float(ry0))

    # We use a slightly larger radius for boundary frames to be extra safe
    r_key = int(max(16, int(search_radius) * 2))

    srck, ownsk = _ensure_source(source_obj, cache_items=2)
    try:
        pred_x, pred_y = float(rx0), float(ry0)
        for b in boundaries[1:]:
            img = _get_frame(srck, b, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb), bayer_pattern=bayer_pattern)

            cur = _to_mono01(img).astype(np.float32, copy=False)
            cur = _downN(cur)
            curp = _bandpass(cur) if bandpass else (cur - float(cur.mean()))

            mx_ds, my_ds, dx_b, dy_b, conf_b = _match_one(curp, pred_x, pred_y, r_key)

            # store boundary predictor (template top-left in this frame)
            start_pred[b] = (mx_ds, my_ds)

            # update for next boundary
            pred_x, pred_y = mx_ds, my_ds

            # also fill boundary output immediately (optional but nice)
            dx[b] = dx_b
            dy[b] = dy_b
            cc[b] = conf_b
            if conf_b < 0.15 and b > 0:
                dx[b] = dx[b - 1]
                dy[b] = dy[b - 1]
    finally:
        if ownsk:
            try:
                srck.close()
            except Exception:
                pass

    # ---------------------------
    # Parallel per-chunk scan (each chunk sequential)
    # ---------------------------
    r = int(max(16, search_radius))

    def _run_chunk(b: int, e: int) -> int:
        src, owns = _ensure_source(source_obj, cache_items=0)
        try:
            pred_x, pred_y = start_pred.get(b, (float(rx0), float(ry0)))
            # if boundary already computed above, keep it; start after b
            i0 = b
            if b in start_pred and b != 0:
                i0 = b + 1   # boundary already solved with r_key

            if i0 == 0:
                i0 = 1
            for i in range(i0, e):
                if i in start_pred:
                    pred_x, pred_y = start_pred[i]
                    continue

                img = _get_frame(src, i, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb), bayer_pattern=bayer_pattern)
                cur = _to_mono01(img).astype(np.float32, copy=False)
                cur = _downN(cur)
                curp = _bandpass(cur) if bandpass else (cur - float(cur.mean()))

                mx_ds, my_ds, dx_i, dy_i, conf_i = _match_one(curp, pred_x, pred_y, r)

                dx[i] = dx_i
                dy[i] = dy_i
                cc[i] = conf_i

                pred_x, pred_y = mx_ds, my_ds

                if conf_i < 0.15 and i > 0:
                    dx[i] = dx[i - 1]
                    dy[i] = dy[i - 1]
            return (e - b)
        finally:
            if owns:
                try:
                    src.close()
                except Exception:
                    pass

    if workers <= 1 or n <= stride * 2:
        # small job: just do sequential scan exactly like before
        src, owns = _ensure_source(source_obj, cache_items=2)
        try:
            pred_x, pred_y = float(rx0), float(ry0)
            for i in range(1, n):
                img = _get_frame(src, i, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb), bayer_pattern=bayer_pattern)
                cur = _to_mono01(img).astype(np.float32, copy=False)
                cur = _downN(cur)
                curp = _bandpass(cur) if bandpass else (cur - float(cur.mean()))

                mx_ds, my_ds, dx_i, dy_i, conf_i = _match_one(curp, pred_x, pred_y, r)
                dx[i] = dx_i
                dy[i] = dy_i
                cc[i] = conf_i
                pred_x, pred_y = mx_ds, my_ds

                if conf_i < 0.15:
                    dx[i] = dx[i - 1]
                    dy[i] = dy[i - 1]

                if progress_cb and (i % int(max(1, progress_every)) == 0 or i == n - 1):
                    progress_cb(i, n, "Surface: coarse (ref-locked NCC+subpix)…")
        finally:
            if owns:
                try:
                    src.close()
                except Exception:
                    pass
        return dx, dy, cc

    # Parallel chunks
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for b in boundaries:
            e = min(n, b + stride)
            futs.append(ex.submit(_run_chunk, b, e))

        for fut in as_completed(futs):
            done += int(fut.result())
            if progress_cb:
                # best-effort: done is "frames processed" not exact index
                progress_cb(min(done, n - 1), n, "Surface: coarse (ref-locked NCC+subpix)…")

    return dx, dy, cc


def _shift_image(img01: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Shift image by (dx,dy) in pixel units. Positive dx shifts right, positive dy shifts down.
    Uses cv2.warpAffine if available; else nearest-ish roll (wrap) fallback.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img01

    if cv2 is not None:
        # border replicate is usually better than constant black for planetary
        h, w = img01.shape[:2]
        M = np.array([[1.0, 0.0, dx],
                      [0.0, 1.0, dy]], dtype=np.float32)
        if img01.ndim == 2:
            return cv2.warpAffine(img01, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            return cv2.warpAffine(img01, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # very rough fallback (wraps!)
    rx = int(round(dx))
    ry = int(round(dy))
    out = np.roll(img01, shift=ry, axis=0)
    out = np.roll(out, shift=rx, axis=1)
    return out

def _downsample_mono01(img01: np.ndarray, max_dim: int = 512) -> np.ndarray:
    """
    Convert to mono and downsample for analysis/tracking. Returns float32 in [0,1].
    """
    m = _to_mono01(img01).astype(np.float32, copy=False)
    H, W = m.shape[:2]
    mx = int(max(1, max_dim))
    if max(H, W) <= mx:
        return m

    if cv2 is None:
        # crude fallback
        scale = mx / float(max(H, W))
        nh = max(2, int(round(H * scale)))
        nw = max(2, int(round(W * scale)))
        # nearest-ish
        ys = (np.linspace(0, H - 1, nh)).astype(np.int32)
        xs = (np.linspace(0, W - 1, nw)).astype(np.int32)
        return m[ys[:, None], xs[None, :]].astype(np.float32)

    scale = mx / float(max(H, W))
    nh = max(2, int(round(H * scale)))
    nw = max(2, int(round(W * scale)))
    return cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)


def _phase_corr_shift(ref_m: np.ndarray, cur_m: np.ndarray) -> tuple[float, float, float]:
    """
    Returns (dx, dy, response) such that shifting cur by (dx,dy) aligns to ref.
    Uses cv2.phaseCorrelate if available.
    """
    if cv2 is None:
        return 0.0, 0.0, 1.0

    # phaseCorrelate expects float32/float64
    ref = ref_m.astype(np.float32, copy=False)
    cur = cur_m.astype(np.float32, copy=False)
    (dx, dy), resp = cv2.phaseCorrelate(ref, cur)  # shift cur -> ref
    return float(dx), float(dy), float(resp)

def _ensure_source(source, cache_items: int = 10) -> tuple[PlanetaryFrameSource, bool]:
    """
    Returns (src, owns_src)

    Accepts:
      - PlanetaryFrameSource-like object (duck typed: get_frame/meta/close)
      - path string
      - list/tuple of paths
    """
    # Already an opened source-like object
    if source is not None and hasattr(source, "get_frame") and hasattr(source, "meta") and hasattr(source, "close"):
        return source, False

    # allow tuple -> list
    if isinstance(source, tuple):
        source = list(source)

    src = open_planetary_source(source, cache_items=cache_items)
    return src, True

def stack_ser(
    source: str | list[str] | PlanetaryFrameSource,
    *,
    roi=None,
    debayer: bool = True,
    keep_percent: float = 20.0,
    track_mode: str = "planetary",
    surface_anchor=None,
    to_rgb: bool = False,                 # ✅ add this
    bayer_pattern: Optional[str] = None,  # ✅ strongly recommended since dialog passes it    
    analysis: AnalyzeResult | None = None,
    local_warp: bool = True,
    max_dim: int = 512,
    progress_cb=None,
    cache_items: int = 10,
    workers: int | None = None,
    chunk_size: int | None = None,
    # ✅ NEW drizzle knobs
    drizzle_scale: float = 1.0,
    drizzle_pixfrac: float = 0.80,
    drizzle_kernel: str = "gaussian",
    drizzle_sigma: float = 0.0,

) -> tuple[np.ndarray, dict]:
    source_obj = source

    # ---- Worker count ----
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = max(1, min(cpu, 48))

    if cv2 is not None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    drizzle_scale = float(drizzle_scale)
    drizzle_on = drizzle_scale > 1.0001
    drizzle_pixfrac = float(drizzle_pixfrac)
    drizzle_kernel = str(drizzle_kernel).strip().lower()
    if drizzle_kernel not in ("square", "circle", "gaussian"):
        drizzle_kernel = "gaussian"
    drizzle_sigma = float(drizzle_sigma)

    # ---- Open once to get meta + first frame shape ----
    src0, owns0 = _ensure_source(source_obj, cache_items=cache_items)
    try:
        n = int(src0.meta.frames)
        keep_percent = max(0.1, min(100.0, float(keep_percent)))
        k = max(1, int(round(n * (keep_percent / 100.0))))

        if analysis is None or analysis.ref_image is None or analysis.ap_centers is None:
            raise ValueError("stack_ser expects analysis with ref_image + ap_centers (run Analyze first).")

        order = np.asarray(analysis.order, np.int32)
        keep_idx = order[:k].astype(np.int32, copy=False)

        # reference / APs
        ref_img = analysis.ref_image.astype(np.float32, copy=False)
        ref_m = _to_mono01(ref_img).astype(np.float32, copy=False)
        ap_centers_all = np.asarray(analysis.ap_centers, np.int32)
        ap_size = int(getattr(analysis, "ap_size", 64) or 64)

        # frame shape for accumulator
        first = _get_frame(src0, int(keep_idx[0]), roi=roi, debayer=debayer, to_float01=True, force_rgb=False, bayer_pattern=bayer_pattern)
        acc_shape = first.shape  # (H,W) or (H,W,3)
    finally:
        if owns0:
            try:
                src0.close()
            except Exception:
                pass

    # ---- Progress aggregation (thread-safe) ----
    done_lock = threading.Lock()
    done_ct = 0
    total_ct = int(len(keep_idx))

    def _bump_progress(delta: int, phase: str = "Stack"):
        nonlocal done_ct
        if progress_cb is None:
            return
        with done_lock:
            done_ct += int(delta)
            d = done_ct
        progress_cb(d, total_ct, phase)

    # ---- Chunking ----
    idx_list = keep_idx.tolist()
    if chunk_size is None:
        chunk_size = max(8, int(np.ceil(len(idx_list) / float(workers * 2))))
    chunks: list[list[int]] = [idx_list[i:i + chunk_size] for i in range(0, len(idx_list), chunk_size)]

    if progress_cb:
        progress_cb(0, total_ct, "Stack")

    # ---- drizzle helpers ----
    if drizzle_on:
        from setiastro.saspro.legacy.numba_utils import (
            drizzle_deposit_numba_kernel_mono,
            drizzle_deposit_color_kernel,
            finalize_drizzle_2d,
            finalize_drizzle_3d,
        )

        # map kernel string -> code used by your numba
        kernel_code = {"square": 0, "circle": 1, "gaussian": 2}[drizzle_kernel]

        # If gaussian sigma isn't provided, use something tied to pixfrac.
        # Your numba interprets gaussian sigma as "sigma_out", and also enforces >= drop_shrink*0.5.
        if drizzle_sigma <= 1e-9:
            # a good practical default: sigma ~ pixfrac*0.5
            drizzle_sigma_eff = max(1e-3, float(drizzle_pixfrac) * 0.5)
        else:
            drizzle_sigma_eff = drizzle_sigma

        H, W = int(acc_shape[0]), int(acc_shape[1])
        outH = int(round(H * drizzle_scale))
        outW = int(round(W * drizzle_scale))

        # Identity transform from input pixels -> aligned/reference pixel coords
        # drizzle_factor applies the scale.
        T = np.zeros((2, 3), dtype=np.float32)
        T[0, 0] = 1.0
        T[1, 1] = 1.0

    # ---- Worker: accumulate its own sum OR its own drizzle buffers ----
    def _stack_chunk(chunk: list[int]):
        src, owns = _ensure_source(source_obj, cache_items=0)
        try:
            if drizzle_on:
                if len(acc_shape) == 2:
                    dbuf = np.zeros((outH, outW), dtype=np.float32)
                    cbuf = np.zeros((outH, outW), dtype=np.float32)
                else:
                    dbuf = np.zeros((outH, outW, acc_shape[2]), dtype=np.float32)
                    cbuf = np.zeros((outH, outW, acc_shape[2]), dtype=np.float32)
            else:
                acc = np.zeros(acc_shape, dtype=np.float32)
                wacc = 0.0

            for i in chunk:
                img = _get_frame(src, int(i), roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb), bayer_pattern=bayer_pattern).astype(np.float32, copy=False)

                # Global prior (from Analyze)
                gdx = float(analysis.dx[int(i)]) if (analysis.dx is not None) else 0.0
                gdy = float(analysis.dy[int(i)]) if (analysis.dy is not None) else 0.0

                # Global prior always first
                warped_g = _shift_image(img, gdx, gdy)

                if cv2 is None or (not local_warp):
                    warped = warped_g
                else:
                    cur_m_g = _to_mono01(warped_g).astype(np.float32, copy=False)

                    ap_rdx, ap_rdy, ap_resp = _ap_phase_shifts_per_ap(
                        ref_m, cur_m_g,
                        ap_centers=ap_centers_all,
                        ap_size=ap_size,
                        max_dim=max_dim,
                    )
                    ap_cf = np.clip(ap_resp.astype(np.float32, copy=False), 0.0, 1.0)

                    keep = _reject_ap_outliers(ap_rdx, ap_rdy, ap_cf, z=3.5)
                    if np.any(keep):
                        ap_centers = ap_centers_all[keep]
                        ap_dx_k = ap_rdx[keep]
                        ap_dy_k = ap_rdy[keep]
                        ap_cf_k = ap_cf[keep]

                        dx_field, dy_field = _dense_field_from_ap_shifts(
                            warped_g.shape[0], warped_g.shape[1],
                            ap_centers, ap_dx_k, ap_dy_k, ap_cf_k,
                            grid=32, power=2.0, conf_floor=0.15,
                            radius=float(ap_size) * 3.0,
                        )
                        warped = _warp_by_dense_field(warped_g, dx_field, dy_field)
                    else:
                        warped = warped_g

                if drizzle_on:
                    # deposit aligned frame into drizzle buffers
                    fw = 1.0  # frame_weight (could later use quality weights)
                    if warped.ndim == 2:
                        drizzle_deposit_numba_kernel_mono(
                            warped, T, dbuf, cbuf,
                            drizzle_factor=drizzle_scale,
                            drop_shrink=drizzle_pixfrac,
                            frame_weight=fw,
                            kernel_code=kernel_code,
                            gaussian_sigma_or_radius=drizzle_sigma_eff,
                        )
                    else:
                        drizzle_deposit_color_kernel(
                            warped, T, dbuf, cbuf,
                            drizzle_factor=drizzle_scale,
                            drop_shrink=drizzle_pixfrac,
                            frame_weight=fw,
                            kernel_code=kernel_code,
                            gaussian_sigma_or_radius=drizzle_sigma_eff,
                        )
                else:
                    acc += warped
                    wacc += 1.0

            _bump_progress(len(chunk), "Stack")

            if drizzle_on:
                return dbuf, cbuf
            return acc, wacc

        finally:
            if owns:
                try:
                    src.close()
                except Exception:
                    pass

    # ---- Parallel run + reduce ----
    if drizzle_on:
        # reduce drizzle buffers
        if len(acc_shape) == 2:
            dbuf_total = np.zeros((outH, outW), dtype=np.float32)
            cbuf_total = np.zeros((outH, outW), dtype=np.float32)
        else:
            dbuf_total = np.zeros((outH, outW, acc_shape[2]), dtype=np.float32)
            cbuf_total = np.zeros((outH, outW, acc_shape[2]), dtype=np.float32)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_stack_chunk, c) for c in chunks if c]
            for fut in as_completed(futs):
                db, cb = fut.result()
                dbuf_total += db
                cbuf_total += cb

        # finalize
        if len(acc_shape) == 2:
            out = np.zeros((outH, outW), dtype=np.float32)
            finalize_drizzle_2d(dbuf_total, cbuf_total, out)
        else:
            out = np.zeros((outH, outW, acc_shape[2]), dtype=np.float32)
            finalize_drizzle_3d(dbuf_total, cbuf_total, out)

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    else:
        acc_total = np.zeros(acc_shape, dtype=np.float32)
        wacc_total = 0.0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_stack_chunk, c) for c in chunks if c]
            for fut in as_completed(futs):
                acc_c, w_c = fut.result()
                acc_total += acc_c
                wacc_total += float(w_c)

        out = np.clip(acc_total / max(1e-6, wacc_total), 0.0, 1.0).astype(np.float32, copy=False)

    diag = {
        "frames_total": int(n),
        "frames_kept": int(len(keep_idx)),
        "roi_used": roi,
        "track_mode": track_mode,
        "local_warp": bool(local_warp),
        "workers": int(workers),
        "chunk_size": int(chunk_size),
        "drizzle_scale": float(drizzle_scale),
        "drizzle_pixfrac": float(drizzle_pixfrac),
        "drizzle_kernel": str(drizzle_kernel),
        "drizzle_sigma": float(drizzle_sigma),
    }
    return out, diag

def _build_reference(
    src: PlanetaryFrameSource,
    *,
    order: np.ndarray,
    roi,
    debayer: bool,
    to_rgb: bool,
    ref_mode: str,
    ref_count: int,
    bayer_pattern=None,
) -> np.ndarray:
    """
    ref_mode:
      - "best_frame": return best single frame
      - "best_stack": return mean of best ref_count frames
    """
    best_idx = int(order[0])
    f0 = _get_frame(src, best_idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb), bayer_pattern=bayer_pattern)
    if ref_mode != "best_stack" or ref_count <= 1:
        return f0.astype(np.float32, copy=False)

    k = int(max(2, min(ref_count, len(order))))
    acc = np.zeros_like(f0, dtype=np.float32)
    for j in range(k):
        idx = int(order[j])
        fr = _get_frame(src, idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb), bayer_pattern=bayer_pattern)
        acc += fr.astype(np.float32, copy=False)
    ref = acc / float(k)
    return np.clip(ref, 0.0, 1.0).astype(np.float32, copy=False)

def _cfg_get_source(cfg) -> Any:
    """
    Back-compat: prefer cfg.source (new), else cfg.ser_path (old).
    cfg.source may be:
      - path string (ser/avi/mp4/etc)
      - list of image paths
      - PlanetaryFrameSource
    """
    src = getattr(cfg, "source", None)
    if src is not None and src != "":
        return src
    return getattr(cfg, "ser_path", None)

def analyze_ser(
    cfg: SERStackConfig,
    *,
    debayer: bool = True,
    to_rgb: bool = False,
    smooth_sigma: float = 1.5,   # kept for API compat
    thresh_pct: float = 92.0,    # kept for API compat
    ref_mode: str = "best_frame",    # "best_frame" or "best_stack"
    bayer_pattern: Optional[str] = None,
    ref_count: int = 5,
    max_dim: int = 512,
    progress_cb=None,
    workers: Optional[int] = None,
) -> AnalyzeResult:
    """
    Parallel analyze for *any* PlanetaryFrameSource (SER/AVI/MP4/images/sequence).
    - Pass 1: quality for every frame
    - Build reference:
        - planetary: best frame or best-N stack
        - surface: frame 0 (chronological anchor)
    - Autoplace APs (always)
    - Pass 2:
        - planetary: AP-based shift directly
        - surface:
            (A) coarse drift stabilization via ref-locked NCC+subpix (on a larger tracking ROI),
            (B) AP search+refine that follows coarse, with outlier rejection,
            (C) robust median -> final dx/dy/conf
    """

    source_obj = _cfg_get_source(cfg)
    bpat = bayer_pattern or _cfg_bayer_pattern(cfg)

    if not source_obj:
        raise ValueError("SERStackConfig.source/ser_path is empty")

    # ---- open source + meta (single open) ----
    src0, owns0 = _ensure_source(source_obj, cache_items=2)
    try:
        meta = src0.meta
        base_roi = cfg.roi
        if base_roi is not None:
            base_roi = _clamp_roi_in_bounds(base_roi, meta.width, meta.height)
        n = int(meta.frames)
        if n <= 0:
            raise ValueError("Source contains no frames")
        src_w = int(meta.width)
        src_h = int(meta.height)
    finally:
        if owns0:
            try:
                src0.close()
            except Exception:
                pass

    # ---- Worker count ----
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = max(1, min(cpu, 48))

    if cv2 is not None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    # ---- Surface tracking ROI (IMPORTANT for big drift) ----
    def _surface_tracking_roi() -> Optional[Tuple[int, int, int, int]]:
        if base_roi is None:
            return None  # full frame
        margin = int(getattr(cfg, "surface_track_margin", 256))
        x, y, w, h = [int(v) for v in base_roi]
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(src_w, x + w + margin)
        y1 = min(src_h, y + h + margin)
        return _clamp_roi_in_bounds((x0, y0, x1 - x0, y1 - y0), src_w, src_h)

    roi_track = _surface_tracking_roi() if cfg.track_mode == "surface" else base_roi
    roi_used = base_roi  # APs and final ref are in this coordinate system

    # -------------------------------------------------------------------------
    # Pass 1: quality (use roi_used)
    # -------------------------------------------------------------------------
    quality = np.zeros((n,), dtype=np.float32)
    idxs = np.arange(n, dtype=np.int32)
    n_chunks = max(5, int(workers) * int(getattr(cfg, "progress_chunk_factor", 5)))
    n_chunks = max(1, min(int(n), n_chunks))
    chunks = np.array_split(idxs, n_chunks)

    if progress_cb:
        progress_cb(0, n, "Quality")

    def _q_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_i: list[int] = []
        out_q: list[float] = []
        src, owns = _ensure_source(source_obj, cache_items=0)
        try:
            for i in chunk.tolist():
                img = _get_frame(
                    src, int(i),
                    roi=roi_used,
                    debayer=debayer,
                    to_float01=True,
                    force_rgb=bool(to_rgb),
                    bayer_pattern=bpat,
                )
                m = _downsample_mono01(img, max_dim=max_dim)

                if cv2 is not None:
                    lap = cv2.Laplacian(m, cv2.CV_32F, ksize=3)
                    q = float(np.mean(np.abs(lap)))
                else:
                    q = float(
                        np.abs(m[:, 1:] - m[:, :-1]).mean() +
                        np.abs(m[1:, :] - m[:-1, :]).mean()
                    )
                out_i.append(int(i))
                out_q.append(q)
        finally:
            if owns:
                try:
                    src.close()
                except Exception:
                    pass
        return np.asarray(out_i, np.int32), np.asarray(out_q, np.float32)

    done_ct = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_q_chunk, c) for c in chunks if c.size > 0]
        for fut in as_completed(futs):
            ii, qq = fut.result()
            quality[ii] = qq
            done_ct += int(ii.size)
            if progress_cb:
                progress_cb(done_ct, n, "Quality")

    order = np.argsort(-quality).astype(np.int32, copy=False)

    # -------------------------------------------------------------------------
    # Build reference
    # -------------------------------------------------------------------------
    ref_count = int(max(1, min(int(ref_count), n)))
    ref_mode = "best_stack" if ref_mode == "best_stack" else "best_frame"

    src_ref, owns_ref = _ensure_source(source_obj, cache_items=2)
    if progress_cb:
        progress_cb(0, n, f"Building reference ({ref_mode}, N={ref_count})…")
    try:
        if cfg.track_mode == "surface":
            # Surface ref must be frame 0 in roi_used coords
            ref_img = _get_frame(
                src_ref, 0,
                roi=roi_used, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb),
                bayer_pattern=bpat,
            ).astype(np.float32, copy=False)

            ref_mode = "first_frame"
            ref_count = 1
        else:
            ref_img = _build_reference(
                src_ref,
                order=order,
                roi=roi_used,
                debayer=debayer,
                to_rgb=to_rgb,
                ref_mode=ref_mode,
                ref_count=ref_count,
                bayer_pattern=bpat,   # ✅ add this
            ).astype(np.float32, copy=False)

    finally:
        if owns_ref:
            try:
                src_ref.close()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Autoplace APs (always)
    # -------------------------------------------------------------------------
    if progress_cb:
        progress_cb(0, n, "Placing alignment points…")

    ap_size = int(getattr(cfg, "ap_size", 64) or 64)
    ap_centers = _autoplace_aps(
        ref_img,
        ap_size=ap_size,
        ap_spacing=int(getattr(cfg, "ap_spacing", 48)),
        ap_min_mean=float(getattr(cfg, "ap_min_mean", 0.03)),
    )

    # -------------------------------------------------------------------------
    # Pass 2: shifts/conf
    # -------------------------------------------------------------------------
    dx = np.zeros((n,), dtype=np.float32)
    dy = np.zeros((n,), dtype=np.float32)
    conf = np.ones((n,), dtype=np.float32)
    coarse_conf: Optional[np.ndarray] = None

    if cfg.track_mode == "off" or cv2 is None:
        return AnalyzeResult(
            frames_total=n,
            roi_used=roi_used,
            track_mode=cfg.track_mode,
            quality=quality,
            dx=dx,
            dy=dy,
            conf=conf,
            order=order,
            ref_mode=ref_mode,
            ref_count=ref_count,
            ref_image=ref_img,
            ap_centers=ap_centers,
            ap_size=ap_size,
            ap_multiscale=bool(getattr(cfg, "ap_multiscale", False)),
            coarse_conf=None,
        )

    ref_m_full = _to_mono01(ref_img).astype(np.float32, copy=False)
    use_multiscale = bool(getattr(cfg, "ap_multiscale", False))

    # ---- surface coarse drift (ref-locked) ----
    if cfg.track_mode == "surface":
        coarse_conf = np.zeros((n,), dtype=np.float32)
        if progress_cb:
            progress_cb(0, n, "Surface: coarse drift (ref-locked NCC+subpix)…")

        dx_chain, dy_chain, cc_chain = _coarse_surface_ref_locked(
            source_obj,
            n=n,
            roi=roi_track,
            roi_used=roi_used,   # ✅ NEW
            debayer=debayer,
            to_rgb=to_rgb,
            bayer_pattern=bpat,
            progress_cb=progress_cb,
            progress_every=25,
            down=2,
            template_size=256,
            search_radius=96,
            bandpass=True,
            workers=min(workers, 8),   # coarse doesn’t need 48; 4–8 is usually ideal
            stride=16,                 # 8–32 typical            
        )
        dx[:] = dx_chain
        dy[:] = dy_chain
        coarse_conf[:] = cc_chain

    # ---- chunked refine ----
    idxs2 = np.arange(n, dtype=np.int32)

    # More/smaller chunks => progress updates sooner (futures complete more frequently)
    chunk_factor = int(getattr(cfg, "progress_chunk_factor", 5))  # optional knob
    min_chunks = 5
    n_chunks2 = max(min_chunks, int(workers) * chunk_factor)
    n_chunks2 = max(1, min(int(n), n_chunks2))

    chunks2 = np.array_split(idxs2, n_chunks2)

    if progress_cb:
        progress_cb(0, n, "SSD Refine")

    if cfg.track_mode == "surface":
        # FAST surface refine:
        #  - use coarse dx/dy from ref-locked tracker
        #  - apply coarse shift to current mono frame
        #  - compute residual per-AP phase shifts (NO SEARCH)
        #  - final dx/dy = coarse + median(residual)
        def _shift_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []

            src, owns = _ensure_source(source_obj, cache_items=0)
            try:
                for i in chunk.tolist():
                    img = _get_frame(
                        src, int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                        bayer_pattern=bpat,
                    )
                    cur_m = _to_mono01(img).astype(np.float32, copy=False)

                    coarse_dx = float(dx[int(i)])
                    coarse_dy = float(dy[int(i)])

                    # Apply coarse shift FIRST (so APs line up without any searching)
                    cur_m_g = _shift_image(cur_m, coarse_dx, coarse_dy)

                    if use_multiscale:
                        s2, s1, s05 = _scaled_ap_sizes(ap_size)

                        def _one_scale(s_ap: int):
                            rdx, rdy, resp = _ap_phase_shifts_per_ap(
                                ref_m_full, cur_m_g,
                                ap_centers=ap_centers,
                                ap_size=s_ap,
                                max_dim=max_dim,
                            )
                            cf = np.clip(resp.astype(np.float32, copy=False), 0.0, 1.0)
                            keep = _reject_ap_outliers(rdx, rdy, cf, z=3.5)
                            if not np.any(keep):
                                return 0.0, 0.0, 0.25
                            dx_r = float(np.median(rdx[keep]))
                            dy_r = float(np.median(rdy[keep]))
                            cf_r = float(np.median(cf[keep]))
                            return dx_r, dy_r, cf_r

                        dx2, dy2, cf2 = _one_scale(s2)
                        dx1, dy1, cf1 = _one_scale(s1)
                        dx0, dy0, cf0 = _one_scale(s05)

                        w2 = max(1e-3, float(cf2)) * 1.25
                        w1 = max(1e-3, float(cf1)) * 1.00
                        w0 = max(1e-3, float(cf0)) * 0.85
                        wsum = (w2 + w1 + w0)

                        dx_res = (w2 * dx2 + w1 * dx1 + w0 * dx0) / wsum
                        dy_res = (w2 * dy2 + w1 * dy1 + w0 * dy0) / wsum
                        cf_ap = float(np.clip((w2 * cf2 + w1 * cf1 + w0 * cf0) / wsum, 0.0, 1.0))
                    else:
                        rdx, rdy, resp = _ap_phase_shifts_per_ap(
                            ref_m_full, cur_m_g,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                        )
                        cf = np.clip(resp.astype(np.float32, copy=False), 0.0, 1.0)
                        keep = _reject_ap_outliers(rdx, rdy, cf, z=3.5)
                        if np.any(keep):
                            dx_res = float(np.median(rdx[keep]))
                            dy_res = float(np.median(rdy[keep]))
                            cf_ap = float(np.median(cf[keep]))
                        else:
                            dx_res, dy_res, cf_ap = 0.0, 0.0, 0.25

                    # Final = coarse + residual (residual is relative to coarse-shifted frame)
                    # Final = coarse + residual (residual is relative to coarse-shifted frame)
                    dx_i = float(coarse_dx + dx_res)
                    dy_i = float(coarse_dy + dy_res)

                    # Final lock-in refinement: minimize (ref-cur)^2 on gradients in a tiny window
                    # NOTE: pass *unshifted* cur_m with the current dx_i/dy_i estimate
                    dxr, dyr, c_ssd = _refine_shift_ssd(
                        ref_m_full, cur_m, dx_i, dy_i,
                        radius=5, crop=0.80,
                        bruteforce=bool(getattr(cfg, "ssd_refine_bruteforce", False)),
                    )
                    dx_i += float(dxr)
                    dy_i += float(dyr)

                    # Confidence: combine coarse + AP, then optionally nudge with SSD
                    cc = float(coarse_conf[int(i)]) if coarse_conf is not None else 0.5
                    cf_i = float(np.clip(0.60 * cc + 0.40 * float(cf_ap), 0.0, 1.0))
                    cf_i = float(np.clip(0.85 * cf_i + 0.15 * float(c_ssd), 0.05, 1.0))

                    out_i.append(int(i))
                    out_dx.append(dx_i)
                    out_dy.append(dy_i)
                    out_cf.append(cf_i)
            finally:
                if owns:
                    try:
                        src.close()
                    except Exception:
                        pass

            return (
                np.asarray(out_i, np.int32),
                np.asarray(out_dx, np.float32),
                np.asarray(out_dy, np.float32),
                np.asarray(out_cf, np.float32),
            )

    else:
        # planetary: centroid tracking (same as viewer) for GLOBAL dx/dy/conf
        # APs are still computed and used later by stack_ser for local_warp residuals.
        tracker = PlanetaryTracker(
            smooth_sigma=float(getattr(cfg, "planet_smooth_sigma", smooth_sigma)),
            thresh_pct=float(getattr(cfg, "planet_thresh_pct", thresh_pct)),
        )

        # IMPORTANT: reference center is computed from the SAME reference image that Analyze chose
        ref_cx, ref_cy, ref_cc = tracker.compute_center(ref_img)
        if ref_cc <= 0.0:
            # fallback: center of ROI
            mref = _to_mono01(ref_img)
            ref_cx = float(mref.shape[1] * 0.5)
            ref_cy = float(mref.shape[0] * 0.5)

        ref_center = (float(ref_cx), float(ref_cy))
        ref_m_full = _to_mono01(ref_img).astype(np.float32, copy=False)

        def _shift_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []

            src, owns = _ensure_source(source_obj, cache_items=0)
            try:
                for i in chunk.tolist():
                    img = _get_frame(
                        src, int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                        bayer_pattern=bpat,
                    )

                    dx_i, dy_i, cf_i = tracker.shift_to_ref(img, ref_center)

                    if float(cf_i) >= 0.25:
                        cur_m = _to_mono01(img).astype(np.float32, copy=False)
                        dxr, dyr, c_ssd = _refine_shift_ssd(
                            ref_m_full, cur_m, dx_i, dy_i,
                            radius=5, crop=0.80,
                            bruteforce=bool(getattr(cfg, "ssd_refine_bruteforce", False)),
                        )

                        dx_i = float(dx_i) + dxr
                        dy_i = float(dy_i) + dyr
                        cf_i = float(np.clip(0.85 * float(cf_i) + 0.15 * c_ssd, 0.05, 1.0))
                    out_i.append(int(i))
                    out_dx.append(float(dx_i))
                    out_dy.append(float(dy_i))
                    out_cf.append(float(cf_i))
            finally:
                if owns:
                    try:
                        src.close()
                    except Exception:
                        pass

            return (
                np.asarray(out_i, np.int32),
                np.asarray(out_dx, np.float32),
                np.asarray(out_dy, np.float32),
                np.asarray(out_cf, np.float32),
            )


    done_ct = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_shift_chunk, c) for c in chunks2 if c.size > 0]
        for fut in as_completed(futs):
            ii, ddx, ddy, ccf = fut.result()
            dx[ii] = ddx
            dy[ii] = ddy
            conf[ii] = np.clip(ccf, 0.05, 1.0).astype(np.float32, copy=False)

            done_ct += int(ii.size)
            if progress_cb:
                progress_cb(done_ct, n, "SSD Refine")

    if cfg.track_mode == "surface":
        _print_surface_debug(dx=dx, dy=dy, conf=conf, coarse_conf=coarse_conf, floor=0.05, prefix="[SER][Surface]")

    return AnalyzeResult(
        frames_total=n,
        roi_used=roi_used,
        track_mode=cfg.track_mode,
        quality=quality,
        dx=dx,
        dy=dy,
        conf=conf,
        order=order,
        ref_mode=ref_mode,
        ref_count=ref_count,
        ref_image=ref_img,
        ap_centers=ap_centers,
        ap_size=ap_size,
        ap_multiscale=use_multiscale,
        coarse_conf=coarse_conf,
    )


def realign_ser(
    cfg: SERStackConfig,
    analysis: AnalyzeResult,
    *,
    debayer: bool = True,
    to_rgb: bool = False,
    max_dim: int = 512,
    progress_cb=None,
    workers: Optional[int] = None,
) -> AnalyzeResult:
    """
    Recompute dx/dy/conf only using analysis.ref_image and analysis.ap_centers.
    Keeps quality/order/ref_image unchanged.

    Surface mode:
      - recompute coarse drift (ref-locked) on roi_track
      - refine via AP search+refine FOLLOWING coarse + outlier rejection
    """
    bpat = bayer_pattern or _cfg_bayer_pattern(cfg)

    if analysis is None:
        raise ValueError("analysis is None")
    if analysis.ref_image is None:
        raise ValueError("analysis.ref_image is missing")

    source_obj = _cfg_get_source(cfg)
    if not source_obj:
        raise ValueError("SERStackConfig.source/ser_path is empty")

    n = int(analysis.frames_total)
    roi_used = analysis.roi_used
    ref_img = analysis.ref_image

    if cfg.track_mode == "off" or cv2 is None:
        analysis.dx = np.zeros((n,), dtype=np.float32)
        analysis.dy = np.zeros((n,), dtype=np.float32)
        analysis.conf = np.ones((n,), dtype=np.float32)
        if hasattr(analysis, "coarse_conf"):
            analysis.coarse_conf = None
        return analysis

    # Ensure AP centers exist
    ap_centers = getattr(analysis, "ap_centers", None)
    if ap_centers is None or np.asarray(ap_centers).size == 0:
        ap_centers = _autoplace_aps(
            ref_img,
            ap_size=int(getattr(cfg, "ap_size", 64)),
            ap_spacing=int(getattr(cfg, "ap_spacing", 48)),
            ap_min_mean=float(getattr(cfg, "ap_min_mean", 0.03)),
        )
        analysis.ap_centers = ap_centers

    if workers is None:
        cpu = os.cpu_count() or 4
        workers = max(1, min(cpu, 48))

    if cv2 is not None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    # Need meta for ROI expansion (surface tracking)
    src0, owns0 = _ensure_source(source_obj, cache_items=2)
    try:
        meta = src0.meta
        src_w = int(meta.width)
        src_h = int(meta.height)
    finally:
        if owns0:
            try:
                src0.close()
            except Exception:
                pass

    def _surface_tracking_roi() -> Optional[Tuple[int, int, int, int]]:
        if roi_used is None:
            return None
        margin = int(getattr(cfg, "surface_track_margin", 256))
        x, y, w, h = [int(v) for v in roi_used]
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(src_w, x + w + margin)
        y1 = min(src_h, y + h + margin)
        return _clamp_roi_in_bounds((x0, y0, x1 - x0, y1 - y0), src_w, src_h)

    roi_track = _surface_tracking_roi() if cfg.track_mode == "surface" else roi_used

    # ---- chunked refine ----
    idxs2 = np.arange(n, dtype=np.int32)

    # More/smaller chunks => progress updates sooner (futures complete more frequently)
    chunk_factor = int(getattr(cfg, "progress_chunk_factor", 5))  # optional knob
    min_chunks = 5
    n_chunks2 = max(min_chunks, int(workers) * chunk_factor)
    n_chunks2 = max(1, min(int(n), n_chunks2))

    chunks2 = np.array_split(idxs2, n_chunks2)

    dx = np.zeros((n,), dtype=np.float32)
    dy = np.zeros((n,), dtype=np.float32)
    conf = np.ones((n,), dtype=np.float32)

    ref_m = _to_mono01(ref_img).astype(np.float32, copy=False)

    ap_size = int(getattr(cfg, "ap_size", 64) or 64)
    use_multiscale = bool(getattr(cfg, "ap_multiscale", False))

    coarse_conf: Optional[np.ndarray] = None
    if cfg.track_mode == "surface":
        coarse_conf = np.zeros((n,), dtype=np.float32)
        if progress_cb:
            progress_cb(0, n, "Surface: coarse drift (ref-locked NCC+subpix)…")

        dx_chain, dy_chain, cc_chain = _coarse_surface_ref_locked(
            source_obj,
            n=n,
            roi=roi_track,
            roi_used=roi_used,   # ✅ NEW
            debayer=debayer,
            to_rgb=to_rgb,
            bayer_pattern=bpat,
            progress_cb=progress_cb,
            progress_every=25,
            down=2,
            template_size=256,
            search_radius=96,
            bandpass=True,
            workers=min(workers, 8),   # coarse doesn’t need 48; 4–8 is usually ideal
            stride=16,                 # 8–32 typical            
        )

        dx[:] = dx_chain
        dy[:] = dy_chain
        coarse_conf[:] = cc_chain

    if progress_cb:
        progress_cb(0, n, "SSD Refine")

    if cfg.track_mode == "surface":
        def _shift_chunk(chunk: np.ndarray):
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []
            out_cc: list[float] = []

            src, owns = _ensure_source(source_obj, cache_items=0)
            try:
                for i in chunk.tolist():
                    img = _get_frame(
                        src, int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                        bayer_pattern=bpat,
                    )
                    cur_m = _to_mono01(img).astype(np.float32, copy=False)

                    coarse_dx = float(dx[int(i)])
                    coarse_dy = float(dy[int(i)])
                    cc = float(coarse_conf[int(i)]) if coarse_conf is not None else 0.5

                    # Apply coarse shift first
                    cur_m_g = _shift_image(cur_m, coarse_dx, coarse_dy)

                    if use_multiscale:
                        s2, s1, s05 = _scaled_ap_sizes(ap_size)

                        def _one_scale(s_ap: int):
                            rdx, rdy, resp = _ap_phase_shifts_per_ap(
                                ref_m, cur_m_g,
                                ap_centers=ap_centers,
                                ap_size=s_ap,
                                max_dim=max_dim,
                            )
                            cf = np.clip(resp.astype(np.float32, copy=False), 0.0, 1.0)
                            keep = _reject_ap_outliers(rdx, rdy, cf, z=3.5)
                            if not np.any(keep):
                                return 0.0, 0.0, 0.25
                            return (
                                float(np.median(rdx[keep])),
                                float(np.median(rdy[keep])),
                                float(np.median(cf[keep])),
                            )

                        dx2, dy2, cf2 = _one_scale(s2)
                        dx1, dy1, cf1 = _one_scale(s1)
                        dx0, dy0, cf0 = _one_scale(s05)

                        w2 = max(1e-3, float(cf2)) * 1.25
                        w1 = max(1e-3, float(cf1)) * 1.00
                        w0 = max(1e-3, float(cf0)) * 0.85
                        wsum = (w2 + w1 + w0)

                        dx_res = (w2 * dx2 + w1 * dx1 + w0 * dx0) / wsum
                        dy_res = (w2 * dy2 + w1 * dy1 + w0 * dy0) / wsum
                        cf_ap = float(np.clip((w2 * cf2 + w1 * cf1 + w0 * cf0) / wsum, 0.0, 1.0))
                    else:
                        rdx, rdy, resp = _ap_phase_shifts_per_ap(
                            ref_m, cur_m_g,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                        )
                        cf = np.clip(resp.astype(np.float32, copy=False), 0.0, 1.0)
                        keep = _reject_ap_outliers(rdx, rdy, cf, z=3.5)
                        if np.any(keep):
                            dx_res = float(np.median(rdx[keep]))
                            dy_res = float(np.median(rdy[keep]))
                            cf_ap = float(np.median(cf[keep]))
                        else:
                            dx_res, dy_res, cf_ap = 0.0, 0.0, 0.25

                    # Final = coarse + residual (residual is relative to coarse-shifted frame)
                    dx_i = float(coarse_dx + dx_res)
                    dy_i = float(coarse_dy + dy_res)

                    # Final lock-in refinement: minimize (ref-cur)^2 on gradients in a tiny window
                    # NOTE: pass *unshifted* cur_m with the current dx_i/dy_i estimate
                    dxr, dyr, c_ssd = _refine_shift_ssd(ref_m, cur_m, dx_i, dy_i, radius=5, crop=0.80, bruteforce=bool(getattr(cfg, "ssd_refine_bruteforce", False)))
                    dx_i += float(dxr)
                    dy_i += float(dyr)

                    # Confidence: combine coarse + AP, then optionally nudge with SSD
                    cc = float(coarse_conf[int(i)]) if coarse_conf is not None else 0.5
                    cf_i = float(np.clip(0.60 * cc + 0.40 * float(cf_ap), 0.0, 1.0))
                    cf_i = float(np.clip(0.85 * cf_i + 0.15 * float(c_ssd), 0.05, 1.0))


                    out_i.append(int(i))
                    out_dx.append(dx_i)
                    out_dy.append(dy_i)
                    out_cf.append(cf_i)
                    out_cc.append(float(cc))
            finally:
                if owns:
                    try:
                        src.close()
                    except Exception:
                        pass

            return (
                np.asarray(out_i, np.int32),
                np.asarray(out_dx, np.float32),
                np.asarray(out_dy, np.float32),
                np.asarray(out_cf, np.float32),
                np.asarray(out_cc, np.float32),
            )

    else:
        # planetary: centroid tracking (same as viewer)
        tracker = PlanetaryTracker(
            smooth_sigma=float(getattr(cfg, "planet_smooth_sigma", 1.5)),
            thresh_pct=float(getattr(cfg, "planet_thresh_pct", 92.0)),
        )

        # Reference center comes from analysis.ref_image (same anchor as analyze_ser)
        ref_cx, ref_cy, ref_cc = tracker.compute_center(ref_img)
        if ref_cc <= 0.0:
            mref = _to_mono01(ref_img)
            ref_cx = float(mref.shape[1] * 0.5)
            ref_cy = float(mref.shape[0] * 0.5)

        ref_center = (float(ref_cx), float(ref_cy))
        ref_m_full = _to_mono01(ref_img).astype(np.float32, copy=False)

        def _shift_chunk(chunk: np.ndarray):
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []

            src, owns = _ensure_source(source_obj, cache_items=0)
            try:
                for i in chunk.tolist():
                    img = _get_frame(
                        src, int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                        bayer_pattern=bpat,
                    )

                    dx_i, dy_i, cf_i = tracker.shift_to_ref(img, ref_center)
                    
                    if float(cf_i) >= 0.25:
                        cur_m = _to_mono01(img).astype(np.float32, copy=False)
                        dxr, dyr, c_ssd = _refine_shift_ssd(ref_m_full, cur_m, float(dx_i), float(dy_i), radius=2, crop=0.80, bruteforce=bool(getattr(cfg, "ssd_refine_bruteforce", False)))
                        dx_i = float(dx_i) + dxr
                        dy_i = float(dy_i) + dyr
                        cf_i = float(np.clip(0.85 * float(cf_i) + 0.15 * c_ssd, 0.05, 1.0))
                    out_i.append(int(i))
                    out_dx.append(float(dx_i))
                    out_dy.append(float(dy_i))
                    out_cf.append(float(cf_i))

            finally:
                if owns:
                    try:
                        src.close()
                    except Exception:
                        pass

            return (
                np.asarray(out_i, np.int32),
                np.asarray(out_dx, np.float32),
                np.asarray(out_dy, np.float32),
                np.asarray(out_cf, np.float32),
            )


    done_ct = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_shift_chunk, c) for c in chunks2 if c.size > 0]
        for fut in as_completed(futs):
            if cfg.track_mode == "surface":
                ii, ddx, ddy, ccf, ccc = fut.result()
                if coarse_conf is not None:
                    coarse_conf[ii] = ccc
            else:
                ii, ddx, ddy, ccf = fut.result()

            dx[ii] = ddx
            dy[ii] = ddy
            conf[ii] = np.clip(ccf, 0.05, 1.0).astype(np.float32, copy=False)

            done_ct += int(ii.size)
            if progress_cb:
                progress_cb(done_ct, n, "SSD Refine")

    analysis.dx = dx
    analysis.dy = dy
    analysis.conf = conf
    if hasattr(analysis, "coarse_conf"):
        analysis.coarse_conf = coarse_conf

    if cfg.track_mode == "surface":
        _print_surface_debug(dx=dx, dy=dy, conf=conf, coarse_conf=coarse_conf, floor=0.05, prefix="[SER][Surface][realign]")

    return analysis

def _autoplace_aps(ref_img01: np.ndarray, ap_size: int, ap_spacing: int, ap_min_mean: float) -> np.ndarray:
    """
    Return AP centers as int32 array of shape (M,2) with columns (cx, cy) in ROI coords.
    We grid-scan by spacing and keep patches whose mean brightness exceeds ap_min_mean.
    """
    m = _to_mono01(ref_img01).astype(np.float32, copy=False)
    H, W = m.shape[:2]
    s = int(max(16, ap_size))
    step = int(max(4, ap_spacing))

    half = s // 2
    xs = list(range(half, max(half + 1, W - half), step))
    ys = list(range(half, max(half + 1, H - half), step))

    pts = []
    for cy in ys:
        y0 = cy - half
        y1 = y0 + s
        if y0 < 0 or y1 > H:
            continue
        for cx in xs:
            x0 = cx - half
            x1 = x0 + s
            if x0 < 0 or x1 > W:
                continue
            patch = m[y0:y1, x0:x1]
            if float(patch.mean()) >= float(ap_min_mean):
                pts.append((cx, cy))

    if not pts:
        # absolute fallback: a single center point (behaves like single-point)
        pts = [(W // 2, H // 2)]

    return np.asarray(pts, dtype=np.int32)

def _scaled_ap_sizes(base: int) -> tuple[int, int, int]:
    b = int(base)
    s2 = int(round(b * 2.0))
    s1 = int(round(b * 1.0))
    s05 = int(round(b * 0.5))
    # clamp to sane limits
    s2 = max(16, min(256, s2))
    s1 = max(16, min(256, s1))
    s05 = max(16, min(256, s05))
    return s2, s1, s05

def _dense_field_from_ap_shifts(
    H: int, W: int,
    ap_centers: np.ndarray,        # (M,2)
    ap_dx: np.ndarray,             # (M,)
    ap_dy: np.ndarray,             # (M,)
    ap_cf: np.ndarray,             # (M,)
    *,
    grid: int = 32,                # coarse grid resolution (32 or 48 are good)
    power: float = 2.0,
    conf_floor: float = 0.15,
    radius: float | None = None,   # optional clamp in pixels (ROI coords)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns dense (dx_field, dy_field) as float32 arrays (H,W) in ROI pixels.
    Computed on coarse grid then upsampled.
    """
    # coarse grid points
    gh = max(4, int(grid))
    gw = max(4, int(round(grid * (W / max(1, H)))))

    ys = np.linspace(0, H - 1, gh, dtype=np.float32)
    xs = np.linspace(0, W - 1, gw, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)  # (gh,gw)

    pts = ap_centers.astype(np.float32)
    px = pts[:, 0].reshape(-1, 1, 1)  # (M,1,1)
    py = pts[:, 1].reshape(-1, 1, 1)  # (M,1,1)

    cf = np.maximum(ap_cf.astype(np.float32), 0.0)
    good = cf >= float(conf_floor)

    if not np.any(good):
        dxg = np.zeros((gh, gw), np.float32)
        dyg = np.zeros((gh, gw), np.float32)
    else:
        px = px[good]
        py = py[good]
        dx = ap_dx[good].astype(np.float32).reshape(-1, 1, 1)
        dy = ap_dy[good].astype(np.float32).reshape(-1, 1, 1)
        cw = cf[good].astype(np.float32).reshape(-1, 1, 1)

        dxp = px - gx[None, :, :]   # (M,gh,gw)
        dyp = py - gy[None, :, :]   # (M,gh,gw)
        d2 = dxp * dxp + dyp * dyp  # (M,gh,gw)

        if radius is not None:
            r2 = float(radius) * float(radius)
            far = d2 > r2
        else:
            far = None

        w = 1.0 / np.maximum(d2, 1.0) ** (power * 0.5)
        w *= cw

        if far is not None:
            w = np.where(far, 0.0, w)

        wsum = np.sum(w, axis=0)  # (gh,gw)

        dxg = np.sum(w * dx, axis=0) / np.maximum(wsum, 1e-6)
        dyg = np.sum(w * dy, axis=0) / np.maximum(wsum, 1e-6)


    # upsample to full res
    dx_field = cv2.resize(dxg, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32, copy=False)
    dy_field = cv2.resize(dyg, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32, copy=False)
    return dx_field, dy_field

def _warp_by_dense_field(img01: np.ndarray, dx_field: np.ndarray, dy_field: np.ndarray) -> np.ndarray:
    """
    img01 (H,W) or (H,W,3)
    dx_field/dy_field are (H,W) in pixels: shifting cur by (dx,dy) aligns to ref.
    """
    H, W = dx_field.shape
    # remap wants map_x/map_y = source sampling coordinates
    # If we want output aligned-to-ref, we sample from cur at (x - dx, y - dy)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    map_x = xs - dx_field
    map_y = ys - dy_field

    if img01.ndim == 2:
        return cv2.remap(img01, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        return cv2.remap(img01, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def _ap_phase_shift(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    ap_centers: np.ndarray,
    ap_size: int,
    max_dim: int,
) -> tuple[float, float, float]:
    """
    Compute a robust global shift from multiple local AP shifts.
    Returns (dx, dy, conf) in ROI pixel units.
    conf is median of per-AP phase correlation responses.
    """
    s = int(max(16, ap_size))
    half = s // 2

    H, W = ref_m.shape[:2]
    dxs = []
    dys = []
    resps = []

    # downsample reference patches once per AP? (fast enough as-is; M is usually modest)
    for (cx, cy) in ap_centers.tolist():
        x0 = cx - half
        y0 = cy - half
        x1 = x0 + s
        y1 = y0 + s
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue

        ref_patch = ref_m[y0:y1, x0:x1]
        cur_patch = cur_m[y0:y1, x0:x1]

        rp = _downsample_mono01(ref_patch, max_dim=max_dim)
        cp = _downsample_mono01(cur_patch, max_dim=max_dim)

        if rp.shape != cp.shape:
            cp = cv2.resize(cp, (rp.shape[1], rp.shape[0]), interpolation=cv2.INTER_AREA)

        sdx, sdy, resp = _phase_corr_shift(rp, cp)

        # scale back to ROI pixels (patch pixels -> ROI pixels)
        sx = float(s) / float(rp.shape[1])
        sy = float(s) / float(rp.shape[0])

        dxs.append(float(sdx * sx))
        dys.append(float(sdy * sy))
        resps.append(float(resp))

    if not dxs:
        return 0.0, 0.0, 0.5

    dx_med = float(np.median(np.asarray(dxs, np.float32)))
    dy_med = float(np.median(np.asarray(dys, np.float32)))
    conf = float(np.median(np.asarray(resps, np.float32)))

    return dx_med, dy_med, conf

def _ap_phase_shifts_per_ap(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    ap_centers: np.ndarray,
    ap_size: int,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-AP phase correlation shifts (NO SEARCH).
    Returns arrays (ap_dx, ap_dy, ap_resp) in ROI pixels, where shifting cur by (dx,dy)
    aligns it to ref for each AP.
    """
    s = int(max(16, ap_size))
    half = s // 2

    H, W = ref_m.shape[:2]
    M = int(ap_centers.shape[0])

    ap_dx = np.zeros((M,), np.float32)
    ap_dy = np.zeros((M,), np.float32)
    ap_resp = np.zeros((M,), np.float32)

    if cv2 is None or M == 0:
        ap_resp[:] = 0.5
        return ap_dx, ap_dy, ap_resp

    for j, (cx, cy) in enumerate(ap_centers.tolist()):
        x0 = int(cx - half)
        y0 = int(cy - half)
        x1 = x0 + s
        y1 = y0 + s
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            ap_resp[j] = 0.0
            continue

        ref_patch = ref_m[y0:y1, x0:x1]
        cur_patch = cur_m[y0:y1, x0:x1]

        rp = _downsample_mono01(ref_patch, max_dim=max_dim)
        cp = _downsample_mono01(cur_patch, max_dim=max_dim)

        if rp.shape != cp.shape and cv2 is not None:
            cp = cv2.resize(cp, (rp.shape[1], rp.shape[0]), interpolation=cv2.INTER_AREA)

        sdx, sdy, resp = _phase_corr_shift(rp, cp)

        # scale to ROI pixels
        sx = float(s) / float(rp.shape[1])
        sy = float(s) / float(rp.shape[0])

        ap_dx[j] = float(sdx * sx)
        ap_dy[j] = float(sdy * sy)
        ap_resp[j] = float(resp)

    return ap_dx, ap_dy, ap_resp


def _ap_phase_shift_multiscale(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    ap_centers: np.ndarray,
    base_ap_size: int,
    max_dim: int,
) -> tuple[float, float, float]:
    """
    Multi-scale AP shift:
    - compute shifts at 2×, 1×, ½× AP sizes using same centers
    - combine using confidence weights (favoring coarser slightly)
    Returns (dx, dy, conf) in ROI pixels.
    """
    s2, s1, s05 = _scaled_ap_sizes(base_ap_size)

    dx2, dy2, cf2 = _ap_phase_shift(ref_m, cur_m, ap_centers, s2, max_dim)
    dx1, dy1, cf1 = _ap_phase_shift(ref_m, cur_m, ap_centers, s1, max_dim)
    dx0, dy0, cf0 = _ap_phase_shift(ref_m, cur_m, ap_centers, s05, max_dim)

    # weights: confidence * slight preference for larger scale (stability)
    w2 = max(1e-3, float(cf2)) * 1.25
    w1 = max(1e-3, float(cf1)) * 1.00
    w0 = max(1e-3, float(cf0)) * 0.85

    wsum = (w2 + w1 + w0)
    dx = (w2 * dx2 + w1 * dx1 + w0 * dx0) / wsum
    dy = (w2 * dy2 + w1 * dy1 + w0 * dy0) / wsum
    conf = float(np.clip((w2 * cf2 + w1 * cf1 + w0 * cf0) / wsum, 0.0, 1.0))

    return float(dx), float(dy), float(conf)
