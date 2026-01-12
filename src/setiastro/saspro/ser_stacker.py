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
    debayer: bool,
    to_rgb: bool,
    progress_cb=None,
    progress_every: int = 25,
    # tuning:
    down: int = 2,                 # 2 is usually better than 4 for the Moon
    template_size: int = 256,       # template patch on reference (in downsampled pixels!)
    search_radius: int = 96,        # search radius around predicted pos (downsampled pixels)
    bandpass: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Surface coarse tracking that DOES NOT DRIFT:
    - Locks to frame0 reference.
    - Uses predicted shift (from previous frame) + local search window.
    - NCC match for integer shift + phase-corr for subpixel refine.
    Returns dx,dy in FULL-RES pixels (ROI coords), and conf in [0,1].
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
        return cv2.resize(m, (max(2, W // down), max(2, H // down)), interpolation=cv2.INTER_AREA).astype(np.float32, copy=False)

    src, owns = _ensure_source(source_obj, cache_items=2)
    try:
        img0 = src.get_frame(0, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
        ref0 = _to_mono01(img0).astype(np.float32, copy=False)
        ref0 = _downN(ref0)
        if bandpass:
            ref0p = _bandpass(ref0)
        else:
            ref0p = ref0 - float(ref0.mean())

        H, W = ref0p.shape[:2]
        ts = int(max(64, min(template_size, min(H, W) - 4)))
        half = ts // 2

        # choose a “good” template center:
        # default: center of ROI (better: pick highest-variance tile, but keep it simple first)
        cx0 = W // 2
        cy0 = H // 2

        # reference template (fixed)
        rx0 = max(0, min(W - ts, cx0 - half))
        ry0 = max(0, min(H - ts, cy0 - half))
        ref_t = ref0p[ry0:ry0 + ts, rx0:rx0 + ts].copy()

        dx[0] = 0.0
        dy[0] = 0.0
        cc[0] = 1.0

        if progress_cb:
            progress_cb(0, n, "Surface: coarse (ref-locked NCC+subpix)…")

        # predicted location in current frame (downsampled coords)
        pred_x = float(rx0)
        pred_y = float(ry0)

        for i in range(1, n):
            img = src.get_frame(i, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
            cur = _to_mono01(img).astype(np.float32, copy=False)
            cur = _downN(cur)
            curp = _bandpass(cur) if bandpass else (cur - float(cur.mean()))

            # search window around predicted top-left
            r = int(max(16, search_radius))
            x0 = int(max(0, min(W - 1, pred_x - r)))
            y0 = int(max(0, min(H - 1, pred_y - r)))
            x1 = int(min(W, pred_x + r + ts))
            y1 = int(min(H, pred_y + r + ts))

            win = curp[y0:y1, x0:x1]
            if win.shape[0] < ts or win.shape[1] < ts:
                # fallback: keep previous shift
                dx[i] = dx[i - 1]
                dy[i] = dy[i - 1]
                cc[i] = 0.0
                continue

            # NCC match for integer (downsampled) location
            res = cv2.matchTemplate(win, ref_t, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            conf_ncc = float(np.clip(max_val, 0.0, 1.0))

            mx_ds = x0 + max_loc[0]
            my_ds = y0 + max_loc[1]

            # subpixel refine using phase correlation on the matched patches
            cur_t = curp[my_ds:my_ds + ts, mx_ds:mx_ds + ts]
            if cur_t.shape == ref_t.shape:
                (sdx, sdy), resp = cv2.phaseCorrelate(ref_t.astype(np.float32), cur_t.astype(np.float32))
                # dx/dy that shift CUR to REF: dx = (ref_pos - cur_pos) + subpix
                sub_dx = float(sdx)
                sub_dy = float(sdy)
                conf_pc = float(np.clip(resp, 0.0, 1.0))
            else:
                sub_dx = 0.0
                sub_dy = 0.0
                conf_pc = 0.0

            dx_ds = float(rx0 - mx_ds) + sub_dx
            dy_ds = float(ry0 - my_ds) + sub_dy

            dx[i] = float(dx_ds * down)
            dy[i] = float(dy_ds * down)

            # update prediction for next frame
            pred_x = float(mx_ds)
            pred_y = float(my_ds)

            cc[i] = float(np.clip(0.65 * conf_ncc + 0.35 * conf_pc, 0.0, 1.0))

            # reliability fallback: if confidence is awful, freeze motion
            if cc[i] < 0.15:
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
    analysis: AnalyzeResult | None = None,
    local_warp: bool = True,
    max_dim: int = 512,
    progress_cb=None,
    cache_items: int = 10,
    workers: int | None = None,
    chunk_size: int | None = None,
) -> tuple[np.ndarray, dict]:
    source_obj = source

    # ---- Worker count ----
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = max(1, min(cpu, 24))

    if cv2 is not None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

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
        first = src0.get_frame(int(keep_idx[0]), roi=roi, debayer=debayer, to_float01=True, force_rgb=False)
        acc_shape = first.shape
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

    # ---- Worker: accumulate its own sum ----
    def _stack_chunk(chunk: list[int]) -> tuple[np.ndarray, float]:
        src, owns = _ensure_source(source_obj, cache_items=0)
        try:
            acc = np.zeros(acc_shape, dtype=np.float32)
            wacc = 0.0

            for i in chunk:
                img = src.get_frame(int(i), roi=roi, debayer=debayer, to_float01=True, force_rgb=False).astype(np.float32, copy=False)
                cur_m = _to_mono01(img).astype(np.float32, copy=False)

                # Global prior (from Analyze)
                gdx = float(analysis.dx[int(i)]) if (analysis.dx is not None) else 0.0
                gdy = float(analysis.dy[int(i)]) if (analysis.dy is not None) else 0.0

                # Global prior (from Analyze) is ALWAYS applied first
                warped_g = _shift_image(img, gdx, gdy)

                # If no OpenCV or local_warp disabled -> just global shift
                if cv2 is None or (not local_warp):
                    warped = warped_g
                else:
                    # FAST path: NO SEARCH.
                    # Compute per-AP *residual* shifts using phase correlation on the globally-shifted frame.
                    cur_m_g = _to_mono01(warped_g).astype(np.float32, copy=False)

                    ap_rdx, ap_rdy, ap_resp = _ap_phase_shifts_per_ap(
                        ref_m, cur_m_g,
                        ap_centers=ap_centers_all,
                        ap_size=ap_size,
                        max_dim=max_dim,
                    )

                    # Use phase response as confidence
                    ap_cf = np.clip(ap_resp.astype(np.float32, copy=False), 0.0, 1.0)

                    # Reject outliers BEFORE dense field fit
                    keep = _reject_ap_outliers(ap_rdx, ap_rdy, ap_cf, z=3.5)
                    if np.any(keep):
                        ap_centers = ap_centers_all[keep]
                        ap_dx_k = ap_rdx[keep]
                        ap_dy_k = ap_rdy[keep]
                        ap_cf_k = ap_cf[keep]

                        # Dense residual field (residuals are relative to warped_g)
                        dx_field, dy_field = _dense_field_from_ap_shifts(
                            warped_g.shape[0], warped_g.shape[1],
                            ap_centers, ap_dx_k, ap_dy_k, ap_cf_k,
                            grid=32, power=2.0, conf_floor=0.15,
                            radius=float(ap_size) * 3.0,
                        )
                        # Warp the globally aligned frame by residual field
                        warped = _warp_by_dense_field(warped_g, dx_field, dy_field)
                    else:
                        # fallback: no good APs, just global shift
                        warped = warped_g


                acc += warped
                wacc += 1.0

            _bump_progress(len(chunk), "Stack")
            return acc, wacc
        finally:
            if owns:
                try:
                    src.close()
                except Exception:
                    pass

    # ---- Parallel run + reduce ----
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
) -> np.ndarray:
    """
    ref_mode:
      - "best_frame": return best single frame
      - "best_stack": return mean of best ref_count frames
    """
    best_idx = int(order[0])
    f0 = src.get_frame(best_idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
    if ref_mode != "best_stack" or ref_count <= 1:
        return f0.astype(np.float32, copy=False)

    k = int(max(2, min(ref_count, len(order))))
    acc = np.zeros_like(f0, dtype=np.float32)
    for j in range(k):
        idx = int(order[j])
        fr = src.get_frame(idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
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
        workers = max(1, min(cpu, 24))

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
    n_chunks = max(1, min(int(n), int(workers) * 2))
    chunks = np.array_split(idxs, n_chunks)

    if progress_cb:
        progress_cb(0, n, "Quality")

    def _q_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_i: list[int] = []
        out_q: list[float] = []
        src, owns = _ensure_source(source_obj, cache_items=0)
        try:
            for i in chunk.tolist():
                img = src.get_frame(
                    int(i),
                    roi=roi_used,
                    debayer=debayer,
                    to_float01=True,
                    force_rgb=bool(to_rgb),
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
            ref_img = src_ref.get_frame(
                0, roi=roi_used, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb)
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
            debayer=debayer,
            to_rgb=to_rgb,
            progress_cb=progress_cb,
            progress_every=25,
            down=2,
            template_size=256,
            search_radius=96,
            bandpass=True,
        )
        dx[:] = dx_chain
        dy[:] = dy_chain
        coarse_conf[:] = cc_chain

    # ---- chunked refine ----
    idxs2 = np.arange(n, dtype=np.int32)
    n_chunks2 = max(1, min(int(n), int(workers) * 2))
    chunks2 = np.array_split(idxs2, n_chunks2)

    if progress_cb:
        progress_cb(0, n, "Align")

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
                    img = src.get_frame(
                        int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
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
                    dx_i = float(coarse_dx + dx_res)
                    dy_i = float(coarse_dy + dy_res)

                    cc = float(coarse_conf[int(i)]) if coarse_conf is not None else 0.5
                    cf_i = float(np.clip(0.60 * cc + 0.40 * float(cf_ap), 0.0, 1.0))

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
        # planetary (unchanged): AP global phase-corr median shift
        def _shift_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []

            src, owns = _ensure_source(source_obj, cache_items=0)
            try:
                for i in chunk.tolist():
                    img = src.get_frame(
                        int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                    )
                    cur_m_full = _to_mono01(img).astype(np.float32, copy=False)

                    if use_multiscale:
                        dx_i, dy_i, cf_i = _ap_phase_shift_multiscale(
                            ref_m_full, cur_m_full,
                            ap_centers=ap_centers,
                            base_ap_size=ap_size,
                            max_dim=max_dim,
                        )
                    else:
                        dx_i, dy_i, cf_i = _ap_phase_shift(
                            ref_m_full, cur_m_full,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                        )

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
                progress_cb(done_ct, n, "Align")

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
        workers = max(1, min(cpu, 24))

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

    idxs = np.arange(n, dtype=np.int32)
    n_chunks2 = max(1, min(int(n), int(workers) * 2))
    chunks2 = np.array_split(idxs, n_chunks2)

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
            debayer=debayer,
            to_rgb=to_rgb,
            progress_cb=progress_cb,
            progress_every=25,
            down=2,
            template_size=256,
            search_radius=96,
            bandpass=True,
        )
        dx[:] = dx_chain
        dy[:] = dy_chain
        coarse_conf[:] = cc_chain

    if progress_cb:
        progress_cb(0, n, "Align")

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
                    img = src.get_frame(
                        int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
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

                    dx_i = float(coarse_dx + dx_res)
                    dy_i = float(coarse_dy + dy_res)

                    cf_i = float(np.clip(0.60 * float(cc) + 0.40 * float(cf_ap), 0.0, 1.0))

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
        def _shift_chunk(chunk: np.ndarray):
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []

            src, owns = _ensure_source(source_obj, cache_items=0)
            try:
                for i in chunk.tolist():
                    img = src.get_frame(
                        int(i),
                        roi=roi_used,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                    )
                    cur_m = _to_mono01(img).astype(np.float32, copy=False)

                    if use_multiscale:
                        dx_i, dy_i, cf_i = _ap_phase_shift_multiscale(
                            ref_m, cur_m,
                            ap_centers=ap_centers,
                            base_ap_size=ap_size,
                            max_dim=max_dim,
                        )
                    else:
                        dx_i, dy_i, cf_i = _ap_phase_shift(
                            ref_m, cur_m,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                        )

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
                progress_cb(done_ct, n, "Align")

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
