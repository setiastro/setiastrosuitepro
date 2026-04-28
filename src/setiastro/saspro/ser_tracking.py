# src/setiastro/saspro/ser_tracking.py
from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def _to_mono01(img: np.ndarray) -> np.ndarray:
    """Convert frame float [0..1] mono/RGB -> mono float32 [0..1]."""
    if img.ndim == 2:
        m = img
    else:
        # simple luma; keep fast
        m = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return np.asarray(m, dtype=np.float32)


class PlanetaryTracker:
    def __init__(
        self,
        *,
        smooth_sigma: float = 1.5,
        thresh_pct: float = 92.0,
        min_val: float = 0.02,
        use_norm: bool = True,
        norm_hi_pct: float = 99.5,
        norm_lo_pct: float = 1.0,
    ):
        self.smooth_sigma = float(smooth_sigma)
        self.thresh_pct = float(thresh_pct)
        self.min_val = float(min_val)
        self.use_norm = bool(use_norm)
        self.norm_hi_pct = float(norm_hi_pct)
        self.norm_lo_pct = float(norm_lo_pct)
        self._ref_center = None

    def reset(self):
        self._ref_center = None

    def _blur(self, m: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return m
        sigma = max(0.0, self.smooth_sigma)
        if sigma <= 0:
            return m
        k = int(max(3, (sigma * 6) // 2 * 2 + 1))
        return cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma)

    def _normalize_for_detect(self, m: np.ndarray) -> np.ndarray:
        if not self.use_norm:
            return m
        lo = float(np.percentile(m, self.norm_lo_pct))
        hi = float(np.percentile(m, self.norm_hi_pct))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo + 1e-12):
            return m
        det = (m - lo) / (hi - lo)
        return np.clip(det, 0.0, 1.0).astype(np.float32, copy=False)

    def _prep_mono01(self, img01: np.ndarray) -> np.ndarray:
        m = _to_mono01(img01).astype(np.float32, copy=False)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        if self.smooth_sigma > 0.0 and cv2 is not None:
            m = cv2.GaussianBlur(m, (0, 0), float(self.smooth_sigma))
        m = self._normalize_for_detect(m)
        return np.clip(m, 0.0, 1.0).astype(np.float32, copy=False)

    def _find_brightest_blob_center(self, m: np.ndarray) -> tuple[float, float, float]:
        """
        For a planetary disk, geometric center of the brightest blob is always
        more accurate than brightness-weighted centroid (which biases toward
        the brighter limb/core).
        """
        if cv2 is None:
            return self._fallback_geometric_center(m)

        for pct in [99, 97, 95, 93, 90, 85, 80, 70, 60, 50]:
            thr = float(np.percentile(m, pct))
            if thr < self.min_val:
                continue
            mask = (m >= thr).astype(np.uint8)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            n_blobs = num - 1
            if n_blobs < 1:
                continue

            # Find best blob: largest area × mean brightness
            best_k = -1
            best_score = -1.0
            for k in range(1, num):
                area = stats[k, cv2.CC_STAT_AREA]
                if area < 4:
                    continue
                mean_val = float(m[labels == k].mean())
                score = mean_val * (float(area) ** 0.3)
                if score > best_score:
                    best_score = score
                    best_k = k

            if best_k < 1:
                continue

            # Use minEnclosingCircle for geometric center — immune to brightness bias
            blob_mask = (labels == best_k).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # Merge all contours (shouldn't be more than one, but be safe)
            all_pts = np.concatenate(contours, axis=0)
            (cx, cy), radius = cv2.minEnclosingCircle(all_pts)

            # Confidence: how round is the blob (circularity)
            area = stats[best_k, cv2.CC_STAT_AREA]
            circularity = float(area) / max(1.0, np.pi * radius * radius)
            conf = float(np.clip(circularity * float(m[labels == best_k].mean()), 0.0, 1.0))

            return (float(cx), float(cy), conf)

        return self._fallback_geometric_center(m)


    def _fallback_geometric_center(self, m: np.ndarray) -> tuple[float, float, float]:
        """Bounding box center of top-5% brightest pixels."""
        thr = float(np.percentile(m, 95))
        thr = max(thr, self.min_val)
        ys, xs = np.where(m >= thr)
        if len(xs) < 4:
            idx = np.argmax(m)
            cy, cx = np.unravel_index(idx, m.shape)
            return (float(cx), float(cy), 0.1)
        # Bounding box center — not brightness-weighted
        cx = float(xs.min() + xs.max()) / 2.0
        cy = float(ys.min() + ys.max()) / 2.0
        conf = float(np.clip(float(m[ys, xs].mean()) / max(1e-6, float(m.max())), 0.0, 1.0))
        return (cx, cy, conf)

    def _weighted_centroid_in_mask(
        self, m: np.ndarray, mask: np.ndarray
    ) -> tuple[float, float, float]:
        """Brightness-weighted centroid within a binary mask."""
        ys, xs = np.nonzero(mask)
        if len(xs) < 4:
            return (m.shape[1] * 0.5, m.shape[0] * 0.5, 0.0)
        w = m[ys, xs].astype(np.float64)
        sw = float(w.sum()) + 1e-12
        cx = float((xs * w).sum() / sw)
        cy = float((ys * w).sum() / sw)
        conf = float(np.clip(float(w.mean()) / max(1e-6, float(m.max())), 0.0, 1.0))
        return (cx, cy, conf)

    def _fallback_weighted_centroid(self, m: np.ndarray) -> tuple[float, float, float]:
        """Top-5% brightness weighted centroid, no connectivity needed."""
        thr = float(np.percentile(m, 95))
        thr = max(thr, self.min_val)
        ys, xs = np.where(m >= thr)
        if len(xs) < 4:
            # Last resort: just use the single brightest pixel
            idx = np.argmax(m)
            cy, cx = np.unravel_index(idx, m.shape)
            return (float(cx), float(cy), 0.1)
        w = m[ys, xs].astype(np.float64)
        sw = float(w.sum()) + 1e-12
        cx = float((xs * w).sum() / sw)
        cy = float((ys * w).sum() / sw)
        conf = float(np.clip(float(w.mean()) / max(1e-6, float(m.max())), 0.0, 1.0))
        return (cx, cy, conf)

    def compute_center(self, img01: np.ndarray) -> tuple[float, float, float]:
        m = self._prep_mono01(img01)
        return self._find_brightest_blob_center(m)

    def step(self, img01: np.ndarray) -> tuple[float, float, float]:
        cx, cy, conf = self.compute_center(img01)
        if conf <= 0.0:
            return 0.0, 0.0, 0.0
        if self._ref_center is None:
            self._ref_center = (cx, cy)
            return 0.0, 0.0, conf
        rx, ry = self._ref_center
        return float(rx - cx), float(ry - cy), float(conf)

    def shift_to_ref(
        self, img01: np.ndarray, ref_center: tuple[float, float]
    ) -> tuple[float, float, float]:
        cx, cy, conf = self.compute_center(img01)
        if conf <= 0.0:
            return 0.0, 0.0, 0.0
        rx, ry = ref_center
        return float(rx - cx), float(ry - cy), float(conf)
    
class SurfaceTracker:
    """
    Tracks by translation using phase correlation between anchor patch and current patch.
    Good for: lunar/solar/planetary surface close-ups.
    """
    def __init__(self, anchor_patch: np.ndarray, hann_window: bool = True):
        self.ref = np.asarray(anchor_patch, dtype=np.float32)
        self.ref = _to_mono01(self.ref)
        self.hann_window = bool(hann_window)
        self._ref_fft = None
        self._win = None
        self._prep()

    def _prep(self):
        h, w = self.ref.shape
        if self.hann_window:
            wy = np.hanning(h).astype(np.float32)
            wx = np.hanning(w).astype(np.float32)
            self._win = (wy[:, None] * wx[None, :]).astype(np.float32)
        else:
            self._win = None

        a = self.ref.copy()
        a -= float(np.mean(a))
        if self._win is not None:
            a *= self._win
        self._ref_fft = np.fft.rfft2(a)

    def _phase_corr_shift(ref_m: np.ndarray, cur_m: np.ndarray) -> tuple[float, float, float]:
        """
        Returns (dx, dy, response) such that shifting cur by (dx,dy) aligns to ref.
        Uses cv2.phaseCorrelate with mean subtraction + Hann window.
        """
        if cv2 is None:
            return 0.0, 0.0, 0.0

        ref = ref_m.astype(np.float32, copy=False)
        cur = cur_m.astype(np.float32, copy=False)

        # stabilize
        ref = ref - float(ref.mean())
        cur = cur - float(cur.mean())

        # hann window (OpenCV expects float32)
        try:
            win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(ref, cur, win)
        except Exception:
            (dx, dy), resp = cv2.phaseCorrelate(ref, cur)

        return float(dx), float(dy), float(resp)


    @staticmethod
    def _phase_corr(ref_m: np.ndarray, cur_m: np.ndarray) -> tuple[float, float, float]:
        if cv2 is None:
            return 0.0, 0.0, 0.0

        ref = ref_m.astype(np.float32, copy=False)
        cur = cur_m.astype(np.float32, copy=False)

        ref = ref - float(ref.mean())
        cur = cur - float(cur.mean())

        try:
            win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(ref, cur, win)
        except Exception:
            (dx, dy), resp = cv2.phaseCorrelate(ref, cur)

        return float(dx), float(dy), float(resp)

    def step(self, cur_patch: np.ndarray) -> tuple[float, float, float]:
        cur = _to_mono01(np.asarray(cur_patch, dtype=np.float32))
        ref = self.ref
        return self._phase_corr(ref, cur)
    

def compute_planet_center(img01: np.ndarray, *, smooth_sigma: float = 1.5,
                          min_val: float = 0.02, use_norm: bool = True,
                          norm_hi_pct: float = 99.5, norm_lo_pct: float = 1.0,
                          thresh_pct: float = 92.0) -> tuple[float, float] | None:
    """
    Single source of truth for planetary disk center detection.
    Used by the viewer overlay, analyze_ser, and realign_ser.
    Returns (cx, cy) in image pixel coords, or None on failure.
    """
    try:
        tracker = PlanetaryTracker(
            smooth_sigma=smooth_sigma,
            thresh_pct=thresh_pct,
            min_val=min_val,
            use_norm=use_norm,
            norm_hi_pct=norm_hi_pct,
            norm_lo_pct=norm_lo_pct,
        )
        cx, cy, conf = tracker.compute_center(img01)
        if conf <= 0.0:
            return None
        return (float(cx), float(cy))
    except Exception:
        return None    