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


def stack_ser(
    cfg: SERStackConfig,
    *,
    debayer: bool = True,
    to_rgb: bool = False,
    smooth_sigma: float = 1.5,
    thresh_pct: float = 92.0,
    keep_min: int = 8,
    max_keep: Optional[int] = None,
    analysis: Optional[AnalyzeResult] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    if analysis is None:
        analysis = analyze_ser(
            cfg,
            debayer=debayer,
            to_rgb=to_rgb,
            smooth_sigma=smooth_sigma,
            thresh_pct=thresh_pct,
            ref_mode="best_frame",
            ref_count=1,
        )

    n = int(analysis.frames_total)
    keep_pct = float(cfg.keep_percent)
    keep_pct = max(0.1, min(100.0, keep_pct))

    k = int(round(n * (keep_pct / 100.0)))
    k = max(min(n, int(keep_min)), k)
    if max_keep is not None:
        k = min(k, int(max_keep))
    k = max(1, min(n, k))

    winners = analysis.order[:k]
    used = [int(i) for i in winners]

    with SERReader(cfg.ser_path, cache_items=2) as r:
        roi = analysis.roi_used
        # Determine output shape by reading first winner
        w0 = int(winners[0])
        img0 = r.get_frame(w0, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))

        if cfg.track_mode != "off":
            img0 = _shift_image(img0, float(analysis.dx[w0]), float(analysis.dy[w0]))

        acc = np.zeros_like(img0, dtype=np.float32)
        wsum = 0.0

        for idx in winners:
            i = int(idx)
            img = r.get_frame(i, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))

            if cfg.track_mode != "off":
                img = _shift_image(img, float(analysis.dx[i]), float(analysis.dy[i]))

            wt = float(max(0.05, min(1.0, float(analysis.conf[i]))))
            acc += img.astype(np.float32) * wt
            wsum += wt

        out = acc / max(1e-8, wsum)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        diag: Dict[str, Any] = {
            "frames_total": n,
            "frames_kept": k,
            "kept_indices": used,
            "roi_used": roi,
            "track_mode": cfg.track_mode,
            "ref_mode": analysis.ref_mode,
            "ref_count": analysis.ref_count,
            "quality_top5": [(int(analysis.order[i]), float(analysis.quality[analysis.order[i]])) for i in range(min(5, n))],
        }
        return out, diag


def _build_reference(
    r: SERReader,
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
    f0 = r.get_frame(best_idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
    if ref_mode != "best_stack" or ref_count <= 1:
        return f0.astype(np.float32, copy=False)

    k = int(max(2, min(ref_count, len(order))))
    acc = np.zeros_like(f0, dtype=np.float32)
    for j in range(k):
        idx = int(order[j])
        fr = r.get_frame(idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
        acc += fr.astype(np.float32, copy=False)
    ref = acc / float(k)
    return np.clip(ref, 0.0, 1.0).astype(np.float32, copy=False)

def analyze_ser(
    cfg: SERStackConfig,
    *,
    debayer: bool = True,
    to_rgb: bool = False,
    smooth_sigma: float = 1.5,   # kept for API compat (unused in fast path)
    thresh_pct: float = 92.0,    # kept for API compat (unused in fast path)
    ref_mode: str = "best_frame",    # "best_frame" or "best_stack"
    ref_count: int = 5,
    max_dim: int = 512,
    progress_cb=None,
    workers: Optional[int] = None,
) -> AnalyzeResult:
    """
    Parallel analyze:
    - Pass 1 (parallel, chunked): compute quality for every frame.
    - Build reference from best frame or best-N mean stack (single-thread).
    - Pass 2 (parallel, chunked): compute dx/dy/conf via phase correlation against reference.
      (planetary: full ROI frame; surface: anchor patch)
    """
    if not cfg.ser_path:
        raise ValueError("SERStackConfig.ser_path is empty")

    # ---- Meta + ROI (single reader) ----
    with SERReader(cfg.ser_path, cache_items=2) as r0:
        meta = r0.meta
        roi = cfg.roi
        if roi is not None:
            roi = _clamp_roi_in_bounds(roi, meta.width, meta.height)
        n = int(meta.frames)
        if n <= 0:
            raise ValueError("SER contains no frames")

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
    # Pass 1: quality (chunked; each worker opens SERReader once)
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
        with SERReader(cfg.ser_path, cache_items=0) as r:
            for i in chunk.tolist():
                img = r.get_frame(int(i), roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
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

        return np.asarray(out_i, np.int32), np.asarray(out_q, np.float32)

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
    # Build reference (single thread)
    # -------------------------------------------------------------------------
    ref_count = int(max(1, min(ref_count, n)))
    ref_mode = "best_stack" if ref_mode == "best_stack" else "best_frame"

    with SERReader(cfg.ser_path, cache_items=2) as r:
        ref_img = _build_reference(
            r,
            order=order,
            roi=roi,
            debayer=debayer,
            to_rgb=to_rgb,
            ref_mode=ref_mode,
            ref_count=ref_count,
        ).astype(np.float32, copy=False)

    # -------------------------------------------------------------------------
    # Pass 2: shifts/conf (skip if tracking off or no cv2)
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
        )

    # Prepare reference mono used by phase correlation + scaling back to ROI pixels
    if cfg.track_mode == "surface":
        if cfg.surface_anchor is None:
            raise ValueError("track_mode='surface' requires cfg.surface_anchor (ROI-space)")

        ax, ay, aw, ah = [int(v) for v in cfg.surface_anchor]
        H, W = ref_img.shape[:2]
        ax = max(0, min(W - 1, ax))
        ay = max(0, min(H - 1, ay))
        aw = max(2, min(W - ax, aw))
        ah = max(2, min(H - ay, ah))

        ref_patch = ref_img[ay:ay + ah, ax:ax + aw]
        ref_m = _downsample_mono01(ref_patch, max_dim=max_dim)
        refH, refW = ref_m.shape[:2]

        # scale back to ROI pixels based on patch geometry
        sx = float(aw) / float(refW)
        sy = float(ah) / float(refH)

        def _shift_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []

            with SERReader(cfg.ser_path, cache_items=0) as r:
                for i in chunk.tolist():
                    img = r.get_frame(int(i), roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))

                    H2, W2 = img.shape[:2]
                    ax2 = max(0, min(W2 - 1, ax))
                    ay2 = max(0, min(H2 - 1, ay))
                    aw2 = max(2, min(W2 - ax2, aw))
                    ah2 = max(2, min(H2 - ay2, ah))

                    patch = img[ay2:ay2 + ah2, ax2:ax2 + aw2]
                    cur_m = _downsample_mono01(patch, max_dim=max_dim)

                    if cur_m.shape != ref_m.shape:
                        cur_m = cv2.resize(cur_m, (refW, refH), interpolation=cv2.INTER_AREA)

                    sdx, sdy, resp = _phase_corr_shift(ref_m, cur_m)

                    out_i.append(int(i))
                    out_dx.append(float(sdx * sx))
                    out_dy.append(float(sdy * sy))
                    out_cf.append(float(resp))

            return (
                np.asarray(out_i, np.int32),
                np.asarray(out_dx, np.float32),
                np.asarray(out_dy, np.float32),
                np.asarray(out_cf, np.float32),
            )

    else:
        # planetary: correlate full ROI frame
        ref_m = _downsample_mono01(ref_img, max_dim=max_dim)
        refH, refW = ref_m.shape[:2]
        fullH, fullW = ref_img.shape[:2]

        sx = float(fullW) / float(refW)
        sy = float(fullH) / float(refH)

        def _shift_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            out_i: list[int] = []
            out_dx: list[float] = []
            out_dy: list[float] = []
            out_cf: list[float] = []

            with SERReader(cfg.ser_path, cache_items=0) as r:
                for i in chunk.tolist():
                    img = r.get_frame(int(i), roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
                    cur_m = _downsample_mono01(img, max_dim=max_dim)

                    if cur_m.shape != ref_m.shape:
                        cur_m = cv2.resize(cur_m, (refW, refH), interpolation=cv2.INTER_AREA)

                    sdx, sdy, resp = _phase_corr_shift(ref_m, cur_m)

                    out_i.append(int(i))
                    out_dx.append(float(sdx * sx))
                    out_dy.append(float(sdy * sy))
                    out_cf.append(float(resp))

            return (
                np.asarray(out_i, np.int32),
                np.asarray(out_dx, np.float32),
                np.asarray(out_dy, np.float32),
                np.asarray(out_cf, np.float32),
            )

    # Run pass 2 chunked
    done_ct = 0
    if progress_cb:
        progress_cb(0, n, "Align")

    chunks2 = np.array_split(idxs, max(1, workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_shift_chunk, c) for c in chunks2 if c.size > 0]
        for fut in as_completed(futs):
            ii, ddx, ddy, ccf = fut.result()
            dx[ii] = ddx
            dy[ii] = ddy
            # clamp response to 0.05..1.0 for stability
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
    )
