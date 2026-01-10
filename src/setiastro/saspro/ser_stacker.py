# src/setiastro/saspro/ser_stacker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from setiastro.saspro.imageops.serloader import SERReader
from setiastro.saspro.ser_stack_config import SERStackConfig
from setiastro.saspro.ser_tracking import PlanetaryTracker, SurfaceTracker, _to_mono01


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


def stack_ser(
    cfg: SERStackConfig,
    *,
    debayer: bool = True,
    to_rgb: bool = False,
    smooth_sigma: float = 1.5,
    thresh_pct: float = 92.0,
    keep_min: int = 8,
    max_keep: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    V1 SER stacker:
    - Two-pass: evaluate (score + shifts), then re-read & accumulate winners
    - Output: float32 image in [0..1], ROI-sized if ROI enabled else full frame

    Params:
      debayer: if SER is Bayer, debayer frames
      to_rgb: force mono->RGB output (usually False; keep native)
      keep_min: always keep at least this many frames (unless frames < keep_min)
      max_keep: optional cap on how many frames to keep
    """
    if not cfg.ser_path:
        raise ValueError("SERStackConfig.ser_path is empty")

    with SERReader(cfg.ser_path, cache_items=2) as r:
        meta = r.meta

        roi = cfg.roi
        if roi is not None:
            roi = _clamp_roi_in_bounds(roi, meta.width, meta.height)

        n = int(meta.frames)
        if n <= 0:
            raise ValueError("SER contains no frames")

        # ----------------------------
        # Pass 1: evaluate frames
        # ----------------------------
        evals: List[FrameEval] = []

        tracker_planet: Optional[PlanetaryTracker] = None
        tracker_surface: Optional[SurfaceTracker] = None

        # For surface mode we need the anchor patch defined in ROI-space
        anchor_roi = cfg.surface_anchor

        # Pre-load first frame if needed (for surface ref patch)
        if cfg.track_mode == "surface":
            if anchor_roi is None:
                raise ValueError("track_mode='surface' requires cfg.surface_anchor (ROI-space)")
            # Create tracker from first frame's anchor patch
            f0 = r.get_frame(0, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
            ax, ay, aw, ah = [int(v) for v in anchor_roi]
            # clamp anchor patch to ROI frame bounds
            H, W = f0.shape[:2]
            ax = max(0, min(W - 1, ax))
            ay = max(0, min(H - 1, ay))
            aw = max(1, min(W - ax, aw))
            ah = max(1, min(H - ay, ah))
            ref_patch = f0[ay:ay + ah, ax:ax + aw]
            tracker_surface = SurfaceTracker(ref_patch, hann_window=True)

        elif cfg.track_mode == "planetary":
            tracker_planet = PlanetaryTracker(smooth_sigma=smooth_sigma, thresh_pct=thresh_pct)

        # Evaluate all frames
        for i in range(n):
            img = r.get_frame(i, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))

            dx = dy = 0.0
            conf = 1.0

            if cfg.track_mode == "planetary" and tracker_planet is not None:
                dx, dy, conf = tracker_planet.step(img)

            elif cfg.track_mode == "surface" and tracker_surface is not None:
                ax, ay, aw, ah = [int(v) for v in cfg.surface_anchor]  # ROI-space
                # clamp per-frame (in case dimensions differ due to ROI changes, etc.)
                H, W = img.shape[:2]
                ax2 = max(0, min(W - 1, ax))
                ay2 = max(0, min(H - 1, ay))
                aw2 = max(1, min(W - ax2, aw))
                ah2 = max(1, min(H - ay2, ah))
                patch = img[ay2:ay2 + ah2, ax2:ax2 + aw2]
                dx, dy, conf = tracker_surface.step(patch)

            # quality
            score = _quality_laplacian(img)

            # optionally weight by confidence (prevents garbage)
            score_w = float(score * max(0.05, min(1.0, conf)))

            evals.append(FrameEval(idx=i, score=score_w, dx=dx, dy=dy, conf=conf))

        # ----------------------------
        # Select top K
        # ----------------------------
        keep_pct = float(cfg.keep_percent)
        keep_pct = max(0.1, min(100.0, keep_pct))

        k = int(round(n * (keep_pct / 100.0)))
        k = max(min(n, int(keep_min)), k)

        if max_keep is not None:
            k = min(k, int(max_keep))
        k = max(1, min(n, k))

        evals_sorted = sorted(evals, key=lambda e: e.score, reverse=True)
        winners = evals_sorted[:k]
        winner_indices = [w.idx for w in winners]

        # ----------------------------
        # Pass 2: accumulate winners
        # ----------------------------
        # Determine output shape by reading first winner
        w0 = winners[0]
        img0 = r.get_frame(w0.idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))
        img0a = _shift_image(img0, w0.dx, w0.dy) if cfg.track_mode != "off" else img0

        acc = np.zeros_like(img0a, dtype=np.float32)
        wsum = 0.0

        used: List[int] = []
        for w in winners:
            img = r.get_frame(w.idx, roi=roi, debayer=debayer, to_float01=True, force_rgb=bool(to_rgb))

            if cfg.track_mode != "off":
                img = _shift_image(img, w.dx, w.dy)

            # weight by score or confidence — v1: use conf only (keeps it stable)
            wt = float(max(0.05, min(1.0, w.conf)))
            acc += img.astype(np.float32) * wt
            wsum += wt
            used.append(w.idx)

        out = acc / max(1e-8, wsum)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        diag: Dict[str, Any] = {
            "frames_total": n,
            "frames_kept": k,
            "kept_indices": used,
            "scores_top5": [(w.idx, w.score, w.conf) for w in winners[:5]],
            "roi_used": roi,
            "track_mode": cfg.track_mode,
        }
        return out, diag
