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
        simple_thresh: float = 0.5,
    ):
        self.smooth_sigma = float(smooth_sigma)
        self.simple_thresh = float(simple_thresh)
        self._ref_center = None

    def reset(self):
        self._ref_center = None

    def compute_center(self, img01: np.ndarray) -> tuple[float, float, float]:
        m = _to_mono01(img01).astype(np.float32, copy=False)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

        if self.smooth_sigma > 0.0 and cv2 is not None:
            m = cv2.GaussianBlur(m, (0, 0), float(self.smooth_sigma))

        mask = m >= self.simple_thresh
        ys, xs = np.where(mask)

        if len(xs) < 4:
            # fallback: brightest pixel
            idx = np.argmax(m)
            cy, cx = np.unravel_index(idx, m.shape)
            return (float(cx), float(cy), 0.1)

        cx = float(xs.mean())
        cy = float(ys.mean())
        conf = float(np.clip(float(len(xs)) / max(1.0, float(m.size)), 0.0, 1.0))
        return (cx, cy, conf)

    def step(self, img01: np.ndarray) -> tuple[float, float, float]:
        cx, cy, conf = self.compute_center(img01)
        if conf <= 0.0:
            return 0.0, 0.0, 0.0
        if self._ref_center is None:
            self._ref_center = (cx, cy)
            return 0.0, 0.0, conf
        rx, ry = self._ref_center
        return float(rx - cx), float(ry - cy), float(conf)

    def shift_to_ref(self, img01: np.ndarray, ref_center: tuple[float, float]) -> tuple[float, float, float]:
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
    

def compute_planet_center(img01: np.ndarray, *,
                          smooth_sigma: float = 1.5,
                          simple_thresh: float = 0.5,
                          ) -> tuple[float, float] | None:
    """Single source of truth for planetary disk center detection."""
    try:
        tracker = PlanetaryTracker(
            smooth_sigma=smooth_sigma,
            simple_thresh=simple_thresh,
        )
        cx, cy, conf = tracker.compute_center(img01)
        if conf <= 0.0:
            return None
        return (float(cx), float(cy))
    except Exception:
        return None