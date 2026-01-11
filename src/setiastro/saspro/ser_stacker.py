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


def _quality_laplacian(img01: np.ndarray) -> float:
    """
    Simple sharpness proxy.
    Returns higher for sharper frames.
    """
    m = _to_mono01(img01)
    if cv2 is not None:
        lap = cv2.Laplacian(m, cv2.CV_32F, ksize=3)
        return float(np.mean(np.abs(lap)))
    # fallback: finite differences (very simple)
    dx = np.abs(m[:, 1:] - m[:, :-1]).mean()
    dy = np.abs(m[1:, :] - m[:-1, :]).mean()
    return float(dx + dy)

def _subpix_patch(img_mono: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray | None:
    """
    Extract a size×size patch centered at (cx,cy) with subpixel accuracy.
    Returns None if patch would go out of bounds.
    """
    if cv2 is None:
        # fallback: integer crop
        half = size // 2
        ix = int(round(cx))
        iy = int(round(cy))
        x0, y0 = ix - half, iy - half
        x1, y1 = x0 + size, y0 + size
        H, W = img_mono.shape[:2]
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            return None
        return img_mono[y0:y1, x0:x1]

    H, W = img_mono.shape[:2]
    half = size / 2.0
    if (cx - half) < 0 or (cy - half) < 0 or (cx + half) >= W or (cy + half) >= H:
        return None
    return cv2.getRectSubPix(img_mono, (int(size), int(size)), (float(cx), float(cy)))


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

def _anchor_match_shift(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    anchor: tuple[int, int, int, int],
    *,
    search_radius: int = 96,
    max_dim: int = 512,
) -> tuple[float, float, float]:
    """
    Find ref anchor patch inside a search window of cur using normalized cross-correlation.
    Returns (dx, dy, conf) where shifting cur by (dx,dy) aligns cur->ref in ROI pixels.

    - anchor is (ax,ay,aw,ah) in ROI coords, defined on reference.
    - search_radius expands a window around the anchor location to allow drift.
    """
    if cv2 is None:
        return 0.0, 0.0, 0.0

    ax, ay, aw, ah = [int(v) for v in anchor]
    H, W = ref_m.shape[:2]

    # clamp anchor to bounds
    ax = max(0, min(W - 2, ax))
    ay = max(0, min(H - 2, ay))
    aw = max(8, min(W - ax, aw))
    ah = max(8, min(H - ay, ah))

    ref_t = ref_m[ay:ay + ah, ax:ax + aw]
    if ref_t.size == 0:
        return 0.0, 0.0, 0.0

    # search window in current frame
    r = int(max(8, search_radius))
    x0 = max(0, ax - r)
    y0 = max(0, ay - r)
    x1 = min(W, ax + aw + r)
    y1 = min(H, ay + ah + r)

    cur_win = cur_m[y0:y1, x0:x1]
    if cur_win.shape[0] < ah or cur_win.shape[1] < aw:
        return 0.0, 0.0, 0.0

    # downsample both for speed if needed (keep same scale factor)
    def _ds(img: np.ndarray) -> np.ndarray:
        return _downsample_mono01(img, max_dim=max_dim)

    ref_t_ds = _ds(ref_t)
    cur_w_ds = _ds(cur_win)

    # ensure template fits
    th, tw = ref_t_ds.shape[:2]
    wh, ww = cur_w_ds.shape[:2]
    if wh < th or ww < tw:
        return 0.0, 0.0, 0.0

    # stabilize illumination differences
    ref_t_ds = ref_t_ds.astype(np.float32, copy=False) - float(ref_t_ds.mean())
    cur_w_ds = cur_w_ds.astype(np.float32, copy=False) - float(cur_w_ds.mean())

    # NCC match
    res = cv2.matchTemplate(cur_w_ds, ref_t_ds, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)  # max_loc = (x,y) in ds coords
    conf = float(np.clip(max_val, 0.0, 1.0))

    # convert ds match location back to full-res ROI coords
    # scale factors:
    sx = float((x1 - x0)) / float(cur_w_ds.shape[1])
    sy = float((y1 - y0)) / float(cur_w_ds.shape[0])

    mx_ds, my_ds = max_loc
    mx = float(x0) + float(mx_ds) * sx  # top-left of matched template in full-res
    my = float(y0) + float(my_ds) * sy

    # shift needed so matched top-left aligns to reference anchor top-left
    dx = float(ax) - mx
    dy = float(ay) - my
    return dx, dy, conf


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
    workers: int | None = None,          # ✅ add
    chunk_size: int | None = None,       # ✅ add (optional tuning)
) -> tuple[np.ndarray, dict]:
    source_obj = source  # keep name consistent with analyze

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
            raise ValueError("stack_ser now expects analysis with ref_image + ap_centers (run Analyze first).")

        order = np.asarray(analysis.order, np.int32)
        keep_idx = order[:k].astype(np.int32, copy=False)

        # reference / APs
        ref_img = analysis.ref_image.astype(np.float32, copy=False)
        ref_m = _to_mono01(ref_img).astype(np.float32, copy=False)
        ap_centers = np.asarray(analysis.ap_centers, np.int32)

        ap_size = int(getattr(analysis, "ap_size", getattr(getattr(analysis, "cfg", None), "ap_size", 64)) or 64)

        # frame shape (mono or RGB) for accumulator
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
        # heuristic: enough chunks so each worker stays busy
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

                if cv2 is None or (not local_warp):
                    dx = float(analysis.dx[int(i)]) if analysis.dx is not None else 0.0
                    dy = float(analysis.dy[int(i)]) if analysis.dy is not None else 0.0
                    warped = _shift_image(img, dx, dy)
                else:
                    # per-AP shifts for this frame
                    ap_dx, ap_dy, ap_cf = _ap_phase_shifts(ref_m, cur_m, ap_centers, ap_size=ap_size, max_dim=max_dim)

                    # ✅ OPTIONAL global-prior add (recommended)
                    gdx = float(analysis.dx[int(i)]) if analysis.dx is not None else 0.0
                    gdy = float(analysis.dy[int(i)]) if analysis.dy is not None else 0.0
                    ap_dx = ap_dx + gdx
                    ap_dy = ap_dy + gdy

                    dx_field, dy_field = _dense_field_from_ap_shifts(
                        img.shape[0], img.shape[1],
                        ap_centers, ap_dx, ap_dy, ap_cf,
                        grid=32, power=2.0, conf_floor=0.15,
                        radius=float(ap_size) * 3.0,
                    )
                    warped = _warp_by_dense_field(img, dx_field, dy_field)

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
    - Build reference (best frame or mean of best-N)
    - Autoplace APs (always)
    - Pass 2: dx/dy/conf using AP multi-point (planetary) or anchor+AP (surface)
    """
    source_obj = _cfg_get_source(cfg)
    if not source_obj:
        raise ValueError("SERStackConfig.source/ser_path is empty")

    # ---- open source + meta (single open) ----
    src0, owns0 = _ensure_source(source_obj, cache_items=2)
    try:
        meta = src0.meta
        roi = cfg.roi
        if roi is not None:
            roi = _clamp_roi_in_bounds(roi, meta.width, meta.height)
        n = int(meta.frames)
        if n <= 0:
            raise ValueError("Source contains no frames")
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

    # Avoid OpenCV oversubscription when we are already parallelizing at Python level
    if cv2 is not None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Pass 1: quality
    # -------------------------------------------------------------------------
    quality = np.zeros((n,), dtype=np.float32)
    idxs = np.arange(n, dtype=np.int32)
    chunks = np.array_split(idxs, max(1, workers))

    done_ct = 0
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
                    roi=roi,
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

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_q_chunk, c) for c in chunks if c.size > 0]
        if progress_cb:
            progress_cb(0, n, "Quality: opening source / decoding first frames…")
        for fut in as_completed(futs):
            ii, qq = fut.result()
            quality[ii] = qq
            done_ct += int(ii.size)
            if progress_cb:
                progress_cb(done_ct, n, "Quality")

    order = np.argsort(-quality).astype(np.int32, copy=False)

    # -------------------------------------------------------------------------
    # Build reference (single thread)
    # -------------------------------------------------------------------------
    ref_count = int(max(1, min(int(ref_count), n)))
    ref_mode = "best_stack" if ref_mode == "best_stack" else "best_frame"

    src_ref, owns_ref = _ensure_source(source_obj, cache_items=2)
    if progress_cb:
        progress_cb(0, n, f"Building reference ({ref_mode}, N={ref_count})…")    
    try:
        ref_img = _build_reference(
            src_ref,
            order=order,
            roi=roi,
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
    # Autoplace APs (ALWAYS; at least one center is returned)
    # -------------------------------------------------------------------------
    if progress_cb:
        progress_cb(0, n, "Placing alignment points…")    
    ap_centers = _autoplace_aps(
        ref_img,
        ap_size=int(getattr(cfg, "ap_size", 64)),
        ap_spacing=int(getattr(cfg, "ap_spacing", 48)),
        ap_min_mean=float(getattr(cfg, "ap_min_mean", 0.03)),
    )

    # -------------------------------------------------------------------------
    # Pass 2: shifts/conf
    # -------------------------------------------------------------------------
    dx = np.zeros((n,), dtype=np.float32)
    dy = np.zeros((n,), dtype=np.float32)
    conf = np.ones((n,), dtype=np.float32)

    # ✅ track coarse confidence separately (surface only)
    coarse_conf = None

    if cfg.track_mode == "off" or cv2 is None:
        return AnalyzeResult(
            frames_total=n,
            roi_used=roi,
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
        )

    ref_m_full = _to_mono01(ref_img).astype(np.float32, copy=False)
    ap_size = int(getattr(cfg, "ap_size", 64))
    use_multiscale = bool(getattr(cfg, "ap_multiscale", False))

    if cfg.track_mode == "surface":
        if cfg.surface_anchor is None:
            raise ValueError("track_mode='surface' requires cfg.surface_anchor (ROI-space)")

        coarse_conf = np.zeros((n,), dtype=np.float32)

        ax, ay, aw, ah = [int(v) for v in cfg.surface_anchor]
        H, W = ref_img.shape[:2]
        ax = max(0, min(W - 2, ax))
        ay = max(0, min(H - 2, ay))
        aw = max(8, min(W - ax, aw))
        ah = max(8, min(H - ay, ah))

        # reasonable default search radius: 2× anchor size, but clamp
        base_r = int(max(aw, ah) * 2)
        search_r = int(max(32, min(base_r, max(W, H) // 2)))

        def _shift_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                        roi=roi,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                    )
                    cur_m_full = _to_mono01(img).astype(np.float32, copy=False)

                    # ✅ Coarse: anchor template match inside search window (tracks drift!)
                    coarse_dx, coarse_dy, cc = _anchor_match_shift(
                        ref_m_full, cur_m_full,
                        (ax, ay, aw, ah),
                        search_radius=search_r,
                        max_dim=max_dim,
                    )

                    # ✅ Refine using APs, sampling cur at (cx+coarse_dx, cy+coarse_dy) with subpixel patch
                    if use_multiscale:
                        s2, s1, s05 = _scaled_ap_sizes(ap_size)
                        dx2, dy2, cf2 = _ap_phase_shift_with_coarse(ref_m_full, cur_m_full, ap_centers, s2, max_dim, coarse_dx, coarse_dy)
                        dx1, dy1, cf1 = _ap_phase_shift_with_coarse(ref_m_full, cur_m_full, ap_centers, s1, max_dim, coarse_dx, coarse_dy)
                        dx0, dy0, cf0 = _ap_phase_shift_with_coarse(ref_m_full, cur_m_full, ap_centers, s05, max_dim, coarse_dx, coarse_dy)

                        w2 = max(1e-3, float(cf2)) * 1.25
                        w1 = max(1e-3, float(cf1)) * 1.00
                        w0 = max(1e-3, float(cf0)) * 0.85
                        wsum = (w2 + w1 + w0)

                        dx_i = (w2 * dx2 + w1 * dx1 + w0 * dx0) / wsum
                        dy_i = (w2 * dy2 + w1 * dy1 + w0 * dy0) / wsum
                        cf_ap = float(np.clip((w2 * cf2 + w1 * cf1 + w0 * cf0) / wsum, 0.0, 1.0))
                    else:
                        dx_i, dy_i, cf_ap = _ap_phase_shift_with_coarse(
                            ref_m_full, cur_m_full,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                            coarse_dx=coarse_dx,
                            coarse_dy=coarse_dy,
                        )

                    # ✅ Final confidence blends anchor confidence + AP confidence
                    # anchor match tends to be very indicative of success/failure
                    cf_i = float(np.clip(0.55 * float(cc) + 0.45 * float(cf_ap), 0.0, 1.0))

                    out_i.append(int(i))
                    out_dx.append(float(dx_i))
                    out_dy.append(float(dy_i))
                    out_cf.append(float(cf_i))
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
        # planetary: AP-based shift across full ROI
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
                        roi=roi,
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

    # ---- run pass 2 chunked ----
    done_ct = 0
    if progress_cb:
        progress_cb(0, n, "Align")
        progress_cb(0, n, f"Align: starting ({workers} workers)…")

    chunks2 = np.array_split(idxs, max(1, workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_shift_chunk, c) for c in chunks2 if c.size > 0]
        if progress_cb:
            progress_cb(0, n, "Align: computing shifts…")
        for fut in as_completed(futs):
            if cfg.track_mode == "surface":
                ii, ddx, ddy, ccf, ccc = fut.result()
                coarse_conf[ii] = ccc
            else:
                ii, ddx, ddy, ccf = fut.result()

            dx[ii] = ddx
            dy[ii] = ddy
            conf[ii] = np.clip(ccf, 0.05, 1.0).astype(np.float32, copy=False)

            done_ct += int(ii.size)
            if progress_cb:
                progress_cb(done_ct, n, "Align")

    # ✅ DEBUG PRINTS (surface only)
    if cfg.track_mode == "surface":
        _print_surface_debug(dx=dx, dy=dy, conf=conf, coarse_conf=coarse_conf, floor=0.05, prefix="[SER][Surface]")

        # Also warn in-progress channel if you want (optional)
        if progress_cb is not None:
            try:
                dx_min = float(np.min(dx)); dx_max = float(np.max(dx))
                dy_min = float(np.min(dy)); dy_max = float(np.max(dy))
                cc_mean = float(np.mean(coarse_conf)) if coarse_conf is not None else 0.0
                progress_cb(n, n, f"Surface debug: dx[{dx_min:.1f},{dx_max:.1f}] dy[{dy_min:.1f},{dy_max:.1f}] coarse_conf_mean={cc_mean:.2f}")
            except Exception:
                pass

    return AnalyzeResult(
        frames_total=n,
        roi_used=roi,
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
        coarse_conf=coarse_conf,  # ✅ new field (or remove if you didn’t edit dataclass)
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

    Works for SER/AVI/MP4/image sequences via cfg.source (or cfg.ser_path back-compat).
    """
    if analysis is None:
        raise ValueError("analysis is None")
    if analysis.ref_image is None:
        raise ValueError("analysis.ref_image is missing")

    source_obj = _cfg_get_source(cfg)
    if not source_obj:
        raise ValueError("SERStackConfig.source/ser_path is empty")

    n = int(analysis.frames_total)
    roi = analysis.roi_used
    ref_img = analysis.ref_image

    if cfg.track_mode == "off" or cv2 is None:
        analysis.dx = np.zeros((n,), dtype=np.float32)
        analysis.dy = np.zeros((n,), dtype=np.float32)
        analysis.conf = np.ones((n,), dtype=np.float32)
        # optional
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

    # Worker count
    if workers is None:
        cpu = os.cpu_count() or 4
        workers = max(1, min(cpu, 24))

    if cv2 is not None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    idxs = np.arange(n, dtype=np.int32)
    chunks2 = np.array_split(idxs, max(1, workers))

    dx = np.zeros((n,), dtype=np.float32)
    dy = np.zeros((n,), dtype=np.float32)
    conf = np.ones((n,), dtype=np.float32)

    # Precompute reference mono once
    ref_m_full = _to_mono01(ref_img).astype(np.float32, copy=False)

    ap_size = int(getattr(cfg, "ap_size", 64))
    use_multiscale = bool(getattr(cfg, "ap_multiscale", False))

    # ✅ surface-only coarse confidence (anchor NCC)
    coarse_conf = None
    if cfg.track_mode == "surface":
        coarse_conf = np.zeros((n,), dtype=np.float32)

    if progress_cb:
        progress_cb(0, n, "Align")

    if cfg.track_mode == "surface":
        if cfg.surface_anchor is None:
            raise ValueError("track_mode='surface' requires cfg.surface_anchor (ROI-space)")

        ax, ay, aw, ah = [int(v) for v in cfg.surface_anchor]
        H, W = ref_img.shape[:2]
        ax = max(0, min(W - 2, ax))
        ay = max(0, min(H - 2, ay))
        aw = max(8, min(W - ax, aw))
        ah = max(8, min(H - ay, ah))

        # ✅ define a sane search radius (2× anchor size, clamped)
        base_r = int(max(aw, ah) * 2)
        search_r = int(max(32, min(base_r, max(W, H) // 2)))

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
                        roi=roi,
                        debayer=debayer,
                        to_float01=True,
                        force_rgb=bool(to_rgb),
                    )
                    cur_m_full = _to_mono01(img).astype(np.float32, copy=False)

                    # ✅ coarse: anchor template match inside window
                    coarse_dx, coarse_dy, cc = _anchor_match_shift(
                        ref_m_full, cur_m_full,
                        (ax, ay, aw, ah),
                        search_radius=search_r,
                        max_dim=max_dim,
                    )

                    # ✅ refine: APs follow drift using subpixel patch sampling
                    if use_multiscale:
                        s2, s1, s05 = _scaled_ap_sizes(ap_size)

                        dx2, dy2, cf2 = _ap_phase_shift_with_coarse(ref_m_full, cur_m_full, ap_centers, s2, max_dim, coarse_dx, coarse_dy)
                        dx1, dy1, cf1 = _ap_phase_shift_with_coarse(ref_m_full, cur_m_full, ap_centers, s1, max_dim, coarse_dx, coarse_dy)
                        dx0, dy0, cf0 = _ap_phase_shift_with_coarse(ref_m_full, cur_m_full, ap_centers, s05, max_dim, coarse_dx, coarse_dy)

                        w2 = max(1e-3, float(cf2)) * 1.25
                        w1 = max(1e-3, float(cf1)) * 1.00
                        w0 = max(1e-3, float(cf0)) * 0.85
                        wsum = (w2 + w1 + w0)

                        dx_i = (w2 * dx2 + w1 * dx1 + w0 * dx0) / wsum
                        dy_i = (w2 * dy2 + w1 * dy1 + w0 * dy0) / wsum
                        cf_ap = float(np.clip((w2 * cf2 + w1 * cf1 + w0 * cf0) / wsum, 0.0, 1.0))
                    else:
                        dx_i, dy_i, cf_ap = _ap_phase_shift_with_coarse(
                            ref_m_full, cur_m_full,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                            coarse_dx=coarse_dx,
                            coarse_dy=coarse_dy,
                        )

                    # ✅ final conf: blend coarse anchor match + AP confidence
                    cf_i = float(np.clip(0.55 * float(cc) + 0.45 * float(cf_ap), 0.0, 1.0))

                    out_i.append(int(i))
                    out_dx.append(float(dx_i))
                    out_dy.append(float(dy_i))
                    out_cf.append(float(cf_i))
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
        # planetary: AP-based shift across full ROI
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
                        roi=roi,
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
            if cfg.track_mode == "surface":
                ii, ddx, ddy, ccf, ccc = fut.result()
                coarse_conf[ii] = ccc
            else:
                ii, ddx, ddy, ccf = fut.result()

            dx[ii] = ddx
            dy[ii] = ddy
            conf[ii] = np.clip(ccf, 0.05, 1.0).astype(np.float32, copy=False)

            done_ct += int(ii.size)
            if progress_cb:
                progress_cb(done_ct, n, "Align")

    # ✅ store results
    analysis.dx = dx
    analysis.dy = dy
    analysis.conf = conf

    # ✅ store coarse_conf if your AnalyzeResult has it
    if hasattr(analysis, "coarse_conf"):
        analysis.coarse_conf = coarse_conf

    # ✅ DEBUG PRINTS (surface only)
    if cfg.track_mode == "surface":
        _print_surface_debug(
            dx=dx, dy=dy, conf=conf, coarse_conf=coarse_conf,
            floor=0.05, prefix="[SER][Surface][realign]"
        )

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

def _ap_phase_shifts(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    ap_centers: np.ndarray,
    ap_size: int,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-AP shifts.
    Returns:
      ap_dx: (M,) float32  ROI pixels
      ap_dy: (M,) float32
      ap_cf: (M,) float32 0..1 response
    """
    s = int(max(16, ap_size))
    half = s // 2
    H, W = ref_m.shape[:2]

    M = int(ap_centers.shape[0])
    ap_dx = np.zeros((M,), np.float32)
    ap_dy = np.zeros((M,), np.float32)
    ap_cf = np.zeros((M,), np.float32)

    for j, (cx, cy) in enumerate(ap_centers.tolist()):
        x0 = cx - half; y0 = cy - half
        x1 = x0 + s;   y1 = y0 + s
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            ap_cf[j] = 0.0
            continue

        ref_patch = ref_m[y0:y1, x0:x1]
        cur_patch = cur_m[y0:y1, x0:x1]

        rp = _downsample_mono01(ref_patch, max_dim=max_dim)
        cp = _downsample_mono01(cur_patch, max_dim=max_dim)
        if rp.shape != cp.shape:
            cp = cv2.resize(cp, (rp.shape[1], rp.shape[0]), interpolation=cv2.INTER_AREA)

        sdx, sdy, resp = _phase_corr_shift(rp, cp)

        sx = float(s) / float(rp.shape[1])
        sy = float(s) / float(rp.shape[0])

        ap_dx[j] = float(sdx * sx)
        ap_dy[j] = float(sdy * sy)
        ap_cf[j] = float(resp)

    return ap_dx, ap_dy, ap_cf

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


def _ap_phase_shift_with_coarse(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    ap_centers: np.ndarray,
    ap_size: int,
    max_dim: int,
    coarse_dx: float,
    coarse_dy: float,
) -> tuple[float, float, float]:
    s = int(max(16, ap_size))
    half = s // 2
    H, W = ref_m.shape[:2]

    dxs, dys, resps = [], [], []

    for (cx_i, cy_i) in ap_centers.tolist():
        cx = float(cx_i)
        cy = float(cy_i)

        # ref patch fixed (integer crop is fine)
        rx0 = int(cx_i - half)
        ry0 = int(cy_i - half)
        rx1 = rx0 + s
        ry1 = ry0 + s
        if rx0 < 0 or ry0 < 0 or rx1 > W or ry1 > H:
            continue
        ref_patch = ref_m[ry0:ry1, rx0:rx1]

        # cur patch follows drift SUBPIXEL
        cur_patch = _subpix_patch(cur_m, cx + coarse_dx, cy + coarse_dy, s)
        if cur_patch is None or cur_patch.shape != ref_patch.shape:
            continue

        rp = _downsample_mono01(ref_patch, max_dim=max_dim)
        cp = _downsample_mono01(cur_patch, max_dim=max_dim)
        if rp.shape != cp.shape:
            cp = cv2.resize(cp, (rp.shape[1], rp.shape[0]), interpolation=cv2.INTER_AREA)

        sdx, sdy, resp = _phase_corr_shift(rp, cp)

        sx = float(s) / float(rp.shape[1])
        sy = float(s) / float(rp.shape[0])

        dxs.append(float(sdx * sx))
        dys.append(float(sdy * sy))
        resps.append(float(resp))

    if not dxs:
        return float(coarse_dx), float(coarse_dy), 0.3

    dx_med = float(np.median(np.asarray(dxs, np.float32)))
    dy_med = float(np.median(np.asarray(dys, np.float32)))
    conf = float(np.median(np.asarray(resps, np.float32)))

    return float(coarse_dx + dx_med), float(coarse_dy + dy_med), conf



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
