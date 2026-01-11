# src/setiastro/saspro/ser_stacker.py
from __future__ import annotations
import os
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
    conf: np.ndarray           # (N,) float32 0..1
    order: np.ndarray          # (N,) int indices sorted by quality desc
    ref_mode: str              # "best_frame" | "best_stack"
    ref_count: int
    ref_image: np.ndarray      # float32 [0..1], ROI-sized
    ap_centers: Optional[np.ndarray] = None  # (M,2) int32 in ROI coords

@dataclass
class FrameEval:
    idx: int
    score: float
    dx: float
    dy: float
    conf: float


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
    roi: tuple[int, int, int, int] | None = None,     # full-frame coords
    debayer: bool = True,
    keep_percent: float = 20.0,
    track_mode: str = "planetary",                    # "planetary" | "surface" | "off"
    surface_anchor: tuple[int, int, int, int] | None = None,   # ROI-space
    # ... keep your other existing params here ...
    cache_items: int = 10,
) -> tuple[np.ndarray, dict]:
    """
    Stack frames from any supported planetary source (SER/AVI/video/images/sequence).

    source:
      - path to SER/AVI/MP4/etc
      - list of image paths (sequence)
      - PlanetaryFrameSource (already opened)
    """
    src, owns = _ensure_source(source, cache_items=cache_items)

    try:
        m = src.meta
        n = int(m.frames)

        # ---- validate inputs ----
        keep_percent = float(keep_percent)
        keep_percent = max(0.1, min(100.0, keep_percent))
        k = max(1, int(round(n * (keep_percent / 100.0))))

        # ---- decide ROI used for reading ----
        read_roi = roi  # full-frame coords (your loader handles cropping)

        # ---- frame selection / quality scoring ----
        # (keep your existing logic here; below is only a placeholder)
        # Example: choose first k frames if you don't have ranking here:
        keep_idx = list(range(n))[:k]

        # ---- read frames ----
        frames = []
        for i in keep_idx:
            fr = src.get_frame(
                i,
                roi=read_roi,
                debayer=debayer,
                to_float01=True,
                force_rgb=False,   # keep as loader decides; stacker can convert as needed
            )
            frames.append(fr)

        if not frames:
            raise RuntimeError("No frames read from source.")

        # ---- stack (replace this with your existing alignment/stack code) ----
        # For now: simple mean stack placeholder
        stack = np.mean(np.stack(frames, axis=0).astype(np.float32), axis=0)

        diag = {
            "source_kind": getattr(m, "source_kind", "unknown"),
            "source_path": getattr(m, "path", None),
            "frames_total": n,
            "frames_kept": len(frames),
            "keep_percent": keep_percent,
            "debayer": bool(debayer),
            "roi": roi,
            "track_mode": track_mode,
            "surface_anchor": surface_anchor,
            "width": int(getattr(m, "width", 0)),
            "height": int(getattr(m, "height", 0)),
        }

        return stack, diag

    finally:
        if owns:
            try:
                src.close()
            except Exception:
                pass

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

    # Precompute reference mono full-res once
    ref_m_full = _to_mono01(ref_img).astype(np.float32, copy=False)

    ap_size = int(getattr(cfg, "ap_size", 64))
    use_multiscale = bool(getattr(cfg, "ap_multiscale", False))

    # ---- surface vs planetary setup ----
    if cfg.track_mode == "surface":
        if cfg.surface_anchor is None:
            raise ValueError("track_mode='surface' requires cfg.surface_anchor (ROI-space)")

        ax, ay, aw, ah = [int(v) for v in cfg.surface_anchor]
        H, W = ref_img.shape[:2]
        ax = max(0, min(W - 1, ax))
        ay = max(0, min(H - 1, ay))
        aw = max(2, min(W - ax, aw))
        ah = max(2, min(H - ay, ah))

        # Reference anchor patch (full-res)
        ref_patch_full = ref_m_full[ay:ay + ah, ax:ax + aw]
        ref_patch_ds = _downsample_mono01(ref_patch_full, max_dim=max_dim)
        refPH, refPW = ref_patch_ds.shape[:2]

        # Scale downsampled-patch shifts back to ROI pixels
        sx = float(aw) / float(refPW)
        sy = float(ah) / float(refPH)

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

                    # --- coarse shift from anchor patch (follows drift) ---
                    cur_patch_full = cur_m_full[ay:ay + ah, ax:ax + aw]
                    cur_patch_ds = _downsample_mono01(cur_patch_full, max_dim=max_dim)
                    if cur_patch_ds.shape != ref_patch_ds.shape:
                        cur_patch_ds = cv2.resize(cur_patch_ds, (refPW, refPH), interpolation=cv2.INTER_AREA)

                    sdx, sdy, coarse_resp = _phase_corr_shift(ref_patch_ds, cur_patch_ds)
                    coarse_dx = float(sdx * sx)
                    coarse_dy = float(sdy * sy)

                    # --- refine using APs, sampling cur at (cx+coarse_dx, cy+coarse_dy) ---
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
                        cf_i = float(np.clip((w2 * cf2 + w1 * cf1 + w0 * cf0) / wsum, 0.0, 1.0))
                    else:
                        dx_i, dy_i, cf_i = _ap_phase_shift_with_coarse(
                            ref_m_full, cur_m_full,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                            coarse_dx=coarse_dx,
                            coarse_dy=coarse_dy,
                        )
                        # combine coarse response into confidence a bit (helps when APs are weak)
                        cf_i = float(np.clip(0.5 * cf_i + 0.5 * float(coarse_resp), 0.0, 1.0))

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

    else:
        # planetary: ALWAYS multi-point AP-based shift across ROI
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
    if progress_cb:
        progress_cb(0, n, f"Align: starting ({workers} workers)…")
    chunks2 = np.array_split(idxs, max(1, workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_shift_chunk, c) for c in chunks2 if c.size > 0]
        if progress_cb:
            progress_cb(0, n, "Align: opening source / computing initial shifts…")        
        for fut in as_completed(futs):
            ii, ddx, ddy, ccf = fut.result()
            dx[ii] = ddx
            dy[ii] = ddy
            conf[ii] = np.clip(ccf, 0.05, 1.0).astype(np.float32, copy=False)

            done_ct += int(ii.size)
            if progress_cb:
                progress_cb(done_ct, n, "Align")

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

    if progress_cb:
        progress_cb(0, n, "Align")

    # Precompute reference mono once
    ref_m_full = _to_mono01(ref_img).astype(np.float32, copy=False)

    ap_size = int(getattr(cfg, "ap_size", 64))
    use_multiscale = bool(getattr(cfg, "ap_multiscale", False))

    if cfg.track_mode == "surface":
        if cfg.surface_anchor is None:
            raise ValueError("track_mode='surface' requires cfg.surface_anchor (ROI-space)")

        ax, ay, aw, ah = [int(v) for v in cfg.surface_anchor]
        H, W = ref_img.shape[:2]
        ax = max(0, min(W - 1, ax))
        ay = max(0, min(H - 1, ay))
        aw = max(2, min(W - ax, aw))
        ah = max(2, min(H - ay, ah))

        ref_patch_full = ref_m_full[ay:ay + ah, ax:ax + aw]
        ref_patch_ds = _downsample_mono01(ref_patch_full, max_dim=max_dim)
        refPH, refPW = ref_patch_ds.shape[:2]
        sx = float(aw) / float(refPW)
        sy = float(ah) / float(refPH)

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

                    # coarse from anchor patch
                    cur_patch_full = cur_m_full[ay:ay + ah, ax:ax + aw]
                    cur_patch_ds = _downsample_mono01(cur_patch_full, max_dim=max_dim)
                    if cur_patch_ds.shape != ref_patch_ds.shape:
                        cur_patch_ds = cv2.resize(cur_patch_ds, (refPW, refPH), interpolation=cv2.INTER_AREA)

                    sdx, sdy, coarse_resp = _phase_corr_shift(ref_patch_ds, cur_patch_ds)
                    coarse_dx = float(sdx * sx)
                    coarse_dy = float(sdy * sy)

                    # refine using APs following drift
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
                        cf_i = float(np.clip((w2 * cf2 + w1 * cf1 + w0 * cf0) / wsum, 0.0, 1.0))
                    else:
                        dx_i, dy_i, cf_i = _ap_phase_shift_with_coarse(
                            ref_m_full, cur_m_full,
                            ap_centers=ap_centers,
                            ap_size=ap_size,
                            max_dim=max_dim,
                            coarse_dx=coarse_dx,
                            coarse_dy=coarse_dy,
                        )
                        cf_i = float(np.clip(0.5 * cf_i + 0.5 * float(coarse_resp), 0.0, 1.0))

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

def _ap_phase_shift_with_coarse(
    ref_m: np.ndarray,
    cur_m: np.ndarray,
    ap_centers: np.ndarray,
    ap_size: int,
    max_dim: int,
    coarse_dx: float,
    coarse_dy: float,
) -> tuple[float, float, float]:
    """
    Same as _ap_phase_shift, but each AP patch in cur is sampled at (cx+coarse_dx, cy+coarse_dy)
    so APs follow drift.
    Returns total (dx, dy, conf) in ROI pixels.
    """
    s = int(max(16, ap_size))
    half = s // 2
    H, W = ref_m.shape[:2]

    dxs = []
    dys = []
    resps = []

    for (cx, cy) in ap_centers.tolist():
        # ref patch fixed
        rx0 = cx - half
        ry0 = cy - half
        rx1 = rx0 + s
        ry1 = ry0 + s

        # cur patch shifted by coarse drift
        ccx = int(round(cx + coarse_dx))
        ccy = int(round(cy + coarse_dy))
        cx0 = ccx - half
        cy0 = ccy - half
        cx1 = cx0 + s
        cy1 = cy0 + s

        # bounds check
        if rx0 < 0 or ry0 < 0 or rx1 > W or ry1 > H:
            continue
        if cx0 < 0 or cy0 < 0 or cx1 > W or cy1 > H:
            continue

        ref_patch = ref_m[ry0:ry1, rx0:rx1]
        cur_patch = cur_m[cy0:cy1, cx0:cx1]

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

    # total shift = coarse + AP refinement
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
