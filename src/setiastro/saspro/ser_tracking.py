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
        min_val: float = 0.02,          # ✅ NEW
        use_norm: bool = True,
        norm_hi_pct: float = 99.5,
        norm_lo_pct: float = 1.0,
    ):
        self.smooth_sigma = float(smooth_sigma)
        self.thresh_pct = float(thresh_pct)
        self.min_val = float(min_val)  # ✅ store
        self.use_norm = bool(use_norm)
        self.norm_hi_pct = float(norm_hi_pct)
        self.norm_lo_pct = float(norm_lo_pct)

    def reset(self):
        self._ref_center = None

    def _blur(self, m: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return m
        # sigma->ksize
        sigma = max(0.0, self.smooth_sigma)
        if sigma <= 0:
            return m
        k = int(max(3, (sigma * 6) // 2 * 2 + 1))
        return cv2.GaussianBlur(m, (k, k), sigmaX=sigma, sigmaY=sigma)

    def _largest_component_mask(self, mask: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return mask
        # mask uint8 0/255
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return mask
        # skip background 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
        out = (labels == k).astype(np.uint8) * 255
        return out

    def _centroid(self, m: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        if cv2 is None:
            # fallback: simple average of mask pixels
            ys, xs = np.nonzero(mask > 0)
            if len(xs) < 10:
                return (0.0, 0.0, 0.0)
            return (float(xs.mean()), float(ys.mean()), 1.0)

        mm = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
        if mm["m00"] <= 0:
            return (0.0, 0.0, 0.0)
        cx = float(mm["m10"] / mm["m00"])
        cy = float(mm["m01"] / mm["m00"])
        # confidence: area fraction
        conf = float(np.clip(mm["m00"] / float(mask.size), 0.0, 1.0))
        return (cx, cy, conf)

    def _normalize_for_detect(self, m: np.ndarray) -> np.ndarray:
        if not self.use_norm:
            return m

        lo = float(np.percentile(m, self.norm_lo_pct))
        hi = float(np.percentile(m, self.norm_hi_pct))
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo + 1e-12):
            return m

        det = (m - lo) / (hi - lo)
        return np.clip(det, 0.0, 1.0).astype(np.float32, copy=False)

    def step(self, img01: np.ndarray) -> tuple[float, float, float]:
        cx, cy, conf = self.compute_center(img01)
        if conf <= 0.0:
            return 0.0, 0.0, 0.0

        if self._ref_center is None:
            self._ref_center = (cx, cy)
            return 0.0, 0.0, conf

        rx, ry = self._ref_center
        dx = rx - cx
        dy = ry - cy
        return float(dx), float(dy), float(conf)
    
    def _prep_mono01(self, img01: np.ndarray) -> np.ndarray:
        m = _to_mono01(img01).astype(np.float32, copy=False)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

        # match step(): blur then lo/hi percentile normalize
        if self.smooth_sigma > 0.0 and cv2 is not None:
            m = cv2.GaussianBlur(m, (0, 0), float(self.smooth_sigma))

        m = self._normalize_for_detect(m)
        return np.clip(m, 0.0, 1.0).astype(np.float32, copy=False)

    def compute_center(self, img01: np.ndarray):
        m = self._prep_mono01(img01)  # already blurred + normalized

        thr = float(np.percentile(m, np.clip(self.thresh_pct, 0.0, 100.0)))
        if not np.isfinite(thr):
            return (m.shape[1] * 0.5), (m.shape[0] * 0.5), 0.0

        # ✅ critical: threshold cannot be below min_val (same domain: normalized [0..1])
        thr = max(thr, float(self.min_val))

        mask = (m >= thr).astype(np.uint8)
        if int(mask.sum()) < 10:
            return (m.shape[1] * 0.5), (m.shape[0] * 0.5), 0.0

        ys, xs = np.nonzero(mask)
        w = m[ys, xs]
        sw = float(w.sum()) + 1e-12
        cx = float((xs * w).sum() / sw)
        cy = float((ys * w).sum() / sw)

        conf = float(np.clip((float(np.mean(w)) - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0))
        return cx, cy, conf


    def shift_to_ref(self, img01: np.ndarray, ref_center: tuple[float, float]) -> tuple[float, float, float]:
        """
        Returns (dx, dy, conf) shifting FROM current frame TO the provided reference center.
        """
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