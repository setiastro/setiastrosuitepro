# pro/autostretch.py
import numpy as np

_MAX_STATS_PIXELS = 1_000_000
_DEFAULT_SIGMA = 3
_U8_MAX  = 4095
_U16_MAX = 65535

# ---------- helpers (generic N-level pipeline) ----------
def _to_uN(a: np.ndarray, maxv: int) -> np.ndarray:
    """Convert to uint8/uint16 [0..maxv] for cheap hist/LUT work."""
    tgt_dtype = np.uint16 if maxv > _U8_MAX else np.uint8
    if a.dtype == tgt_dtype:
        return a
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        if info.max <= 0:
            return np.zeros_like(a, dtype=tgt_dtype)
        scaled = np.clip(a.astype(np.float32), 0, info.max) * (maxv / float(info.max))
        return (scaled + 0.5).astype(tgt_dtype)
    # float-ish
    af = np.clip(a.astype(np.float32), 0.0, 1.0)
    return (af * maxv + 0.5).astype(tgt_dtype)

def _choose_stride(h: int, w: int, max_pixels: int) -> tuple[int, int]:
    n = h * w
    if n <= max_pixels:
        return 1, 1
    s = max(1, int(np.sqrt(n / float(max_pixels))))
    return s, s

def _stats_from_hist_generic(uN: np.ndarray, maxv: int) -> tuple[float, float, int, int]:
    """(median, std, minv, maxv) using a histogram of levels [0..maxv]."""
    hist = np.bincount(uN.ravel(), minlength=maxv + 1)
    total = int(hist.sum())
    if total == 0:
        return 0.0, 1.0, 0, 0

    # min / max (first/last nonzero bin)
    minv = int(np.argmax(hist > 0))
    maxv_found = int(maxv - np.argmax(hist[::-1] > 0))

    cdf = np.cumsum(hist)
    med_idx = int(np.searchsorted(cdf, (total + 1) // 2))

    bins = np.arange(hist.size, dtype=np.float64)
    s1 = float((hist * bins).sum())
    s2 = float((hist * (bins * bins)).sum())
    mean = s1 / total
    var = max(0.0, s2 / total - mean * mean)
    std = float(np.sqrt(var))
    return float(med_idx), std, minv, maxv_found

def _build_lut_generic(bp: int, target_median: float, med_uN: float, maxv: int) -> np.ndarray:
    """Build N-level → float32 LUT using SASv2-style rational formula."""
    denom_bp = max(1, maxv - int(bp))
    median_rescaled = (med_uN - bp) / float(denom_bp)
    median_rescaled = float(np.clip(median_rescaled, 1e-9, 1.0))

    x = np.arange(maxv + 1, dtype=np.float32)
    r = (x - bp) / float(denom_bp)
    denom = median_rescaled * (target_median + r - 1.0) - target_median * r
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    out = ((median_rescaled - 1.0) * target_median * r) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _fast_channel_autostretch_uN(ch_uN: np.ndarray, target: float, sigma: float, maxv: int) -> np.ndarray:
    h, w = ch_uN.shape
    sy, sx = _choose_stride(h, w, _MAX_STATS_PIXELS)
    sample = ch_uN[::sy, ::sx]
    med, std, minv, _ = _stats_from_hist_generic(sample, maxv)
    bp = int(max(minv, med - sigma * std))
    lut = _build_lut_generic(bp, target, med, maxv)
    return lut[ch_uN]  # vectorized lookup

# ---------- public API ----------
def autostretch(
    img: np.ndarray,
    target_median: float = 0.25,
    linked: bool = False,
    sigma: float = _DEFAULT_SIGMA,
    *,
    use_16bit: bool | None = None,
) -> np.ndarray:
    """
    High-quality autostretch that can operate in 16-bit (HQ, default) or 8-bit (fast) mode.

    • 16-bit mode: smooth gradients, minimal posterization (recommended).
    • 8-bit mode: slightly faster on very large images, lower fidelity.

    If use_16bit is None, we try to read QSettings("display/autostretch_16bit") and
    default to True on failure (no Qt in context).
    """
    if img is None:
        return None

    # Optional auto-read from QSettings if caller didn’t pass a flag.
    if use_16bit is None:
        try:
            from PyQt6.QtCore import QSettings
            use_16bit = QSettings().value("display/autostretch_16bit", True, type=bool)
        except Exception:
            use_16bit = True

    maxv = _U16_MAX if use_16bit else _U8_MAX
    a = np.asarray(img)

    # MONO (or pseudo-mono)
    if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):
        u = _to_uN(a.squeeze(), maxv)
        out = _fast_channel_autostretch_uN(u, target_median, sigma, maxv)
        return out.astype(np.float32, copy=False)

    # COLOR
    u = _to_uN(a, maxv)
    if linked:
        # one set of stats over all channels (subsampled)
        h, w, _ = u.shape
        sy, sx = _choose_stride(h, w, max(1, _MAX_STATS_PIXELS // 3))
        sample = u[::sy, ::sx].reshape(-1)  # H*W*C subsample flattened
        med, std, minv, _ = _stats_from_hist_generic(sample, maxv)
        bp = int(max(minv, med - sigma * std))
        lut = _build_lut_generic(bp, target_median, med, maxv)
        out = lut[u]
        return out.astype(np.float32, copy=False)
    else:
        # per-channel stats/LUTs
        out = np.empty_like(u, dtype=np.float32)
        C = u.shape[2]
        for c in range(min(3, C)):
            out[..., c] = _fast_channel_autostretch_uN(u[..., c], target_median, sigma, maxv)
        # if image has >3 channels, just copy remaining
        if C > 3:
            out[..., 3:] = u[..., 3:] / float(maxv)
        return out
