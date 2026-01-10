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
    """
    Tracks by centroid of the brightest connected component inside ROI.
    Good for: planets, full disk objects.
    """
    def __init__(self, smooth_sigma: float = 1.5, thresh_pct: float = 92.0):
        self.smooth_sigma = float(smooth_sigma)
        self.thresh_pct = float(thresh_pct)
        self._ref_center = None  # (cx, cy)

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
        """
        Weighted centroid using intensity. Returns (cx, cy, confidence).
        """
        w = m * (mask > 0)
        s = float(w.sum())
        if s <= 1e-8:
            return (0.0, 0.0, 0.0)
        ys, xs = np.indices(m.shape, dtype=np.float32)
        cx = float((w * xs).sum() / s)
        cy = float((w * ys).sum() / s)
        # confidence: fraction of bright energy captured
        conf = float(np.clip(s / (float(m.sum()) + 1e-8), 0.0, 1.0))
        return cx, cy, conf

    def step(self, img01: np.ndarray) -> tuple[float, float, float]:
        """
        Returns (dx, dy, conf) where dx/dy shifts FROM current frame TO reference.
        """
        m = _to_mono01(img01)
        m2 = self._blur(m)

        # adaptive threshold by percentile
        t = float(np.percentile(m2, self.thresh_pct))
        if not np.isfinite(t):
            return 0.0, 0.0, 0.0

        mask = (m2 >= t).astype(np.uint8) * 255
        mask = self._largest_component_mask(mask)

        cx, cy, conf = self._centroid(m2, mask)
        if conf <= 0.0:
            return 0.0, 0.0, 0.0

        if self._ref_center is None:
            self._ref_center = (cx, cy)
            return 0.0, 0.0, conf

        rx, ry = self._ref_center
        dx = rx - cx
        dy = ry - cy
        return float(dx), float(dy), conf


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

    def _phase_corr(self, cur_patch: np.ndarray) -> tuple[float, float, float]:
        b = _to_mono01(cur_patch).astype(np.float32)
        if b.shape != self.ref.shape:
            return 0.0, 0.0, 0.0

        b = b - float(np.mean(b))
        if self._win is not None:
            b *= self._win

        B = np.fft.rfft2(b)
        R = self._ref_fft * np.conj(B)
        denom = np.maximum(np.abs(R), 1e-8)
        R /= denom
        cc = np.fft.irfft2(R)
        # peak
        y, x = np.unravel_index(np.argmax(cc), cc.shape)
        peak = float(cc[y, x])

        h, w = cc.shape
        # wrap to signed shift
        if x > w // 2:
            x = x - w
        if y > h // 2:
            y = y - h

        # dx/dy to move current -> reference
        dx = -float(x)
        dy = -float(y)

        # confidence from peak (rough)
        conf = float(np.clip((peak - 0.05) / 0.95, 0.0, 1.0))
        return dx, dy, conf

    def step(self, cur_patch: np.ndarray) -> tuple[float, float, float]:
        return self._phase_corr(cur_patch)
