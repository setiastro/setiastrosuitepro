# pro/autostretch.py
import numpy as np
from imageops.stretch import stretch_mono_image, stretch_color_image

# --- knobs ---
_MAX_STATS_PIXELS = 1_000_000   # subsample so stats use ~1M px max
_DEFAULT_SIGMA     = 3
_U16_MAX           = 4096 #65535

def _to_u16(a: np.ndarray) -> np.ndarray:
    """
    Convert any image to uint16 [0..65535] for fast histogram/LUT work.
    Assumes input is linear (not gamma) and nonnegative-ish (clips to [0..1] for float).
    """
    if a.dtype == np.uint16:
        return a
    if a.dtype == np.uint8:
        # expand 8->16 (keep full 0..65535 span)
        return (a.astype(np.uint16) * 257)
    if np.issubdtype(a.dtype, np.integer):
        # generic ints: scale by max of dtype if known; fallback clip
        info = np.iinfo(a.dtype)
        if info.max > 0:
            return np.clip((a.astype(np.float32) / info.max) * _U16_MAX, 0, _U16_MAX).astype(np.uint16)
    # floats
    af = np.clip(a.astype(np.float32), 0.0, 1.0)
    return (af * _U16_MAX + 0.5).astype(np.uint16)

def _choose_stride(h: int, w: int, max_pixels: int) -> tuple[int, int]:
    n = h * w
    if n <= max_pixels:
        return 1, 1
    ratio = np.sqrt(n / float(max_pixels))
    s = max(1, int(ratio))
    return s, s

def _stats_from_hist(u16: np.ndarray) -> tuple[float, float, int, int]:
    """
    Compute (median, std, minv, maxv) using a uint16 histogram for accuracy.
    """
    # Fast path: bincount on flattened array
    hist = np.bincount(u16.ravel(), minlength=_U16_MAX + 1)
    total = int(hist.sum())
    if total == 0:
        return 0.0, 1.0, 0, 0

    # min/max
    minv = int(np.argmax(hist > 0))
    maxv = int(_U16_MAX - np.argmax(hist[::-1] > 0))

    # median via cumulative sum
    cdf = np.cumsum(hist)
    med_idx = int(np.searchsorted(cdf, (total + 1) // 2))

    # mean & variance from histogram
    bins = np.arange(hist.size, dtype=np.float64)
    s1 = float((hist * bins).sum())
    s2 = float((hist * (bins * bins)).sum())
    mean = s1 / total
    var = max(0.0, s2 / total - mean * mean)
    std = float(np.sqrt(var))
    return float(med_idx), std, minv, maxv

def _build_lut(bp: int, target_median: float, med_u16: float) -> np.ndarray:
    """
    Build a 16-bit → float32 LUT using the final rational formula.
    We avoid recomputing med(rescaled) by noting that linear rescale preserves median:
        median_rescaled = (median_u16 - bp) / (65535 - bp)
    """
    denom_bp = max(1, _U16_MAX - int(bp))
    median_rescaled = (med_u16 - bp) / float(denom_bp)
    median_rescaled = float(np.clip(median_rescaled, 1e-9, 1.0))

    x = np.arange(_U16_MAX + 1, dtype=np.float32)
    r = (x - bp) / float(denom_bp)           # linear rescale
    # final mapping (same as your SASv2 formula)
    denom = median_rescaled * (target_median + r - 1.0) - target_median * r
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    out = ((median_rescaled - 1.0) * target_median * r) / denom
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _fast_channel_autostretch_u16(ch_u16: np.ndarray, target: float, sigma: float) -> np.ndarray:
    """
    One channel in uint16 → return float32 [0..1] using LUT.
    Stats are computed on a subsample for speed, histogram on that subsample only.
    """
    h, w = ch_u16.shape
    sy, sx = _choose_stride(h, w, _MAX_STATS_PIXELS)
    sample = ch_u16[::sy, ::sx]

    med, std, minv, _ = _stats_from_hist(sample)
    bp = int(max(minv, med - sigma * std))
    lut = _build_lut(bp, target, med)
    return lut[ch_u16]  # vectorized lookup

def autostretch(img: np.ndarray,
                target_median: float = 0.25,
                linked: bool = False,
                sigma: float = _DEFAULT_SIGMA) -> np.ndarray:
    """
    Very fast display autostretch:
      • Converts to uint16 (quantized) for cheap hist + LUT.
      • Uses black_point = max(min, median - sigma*std).
      • Applies final formula via a precomputed LUT.
    Returns float32 in [0..1], same shape as input.
    """
    if img is None:
        return None

    a = np.asarray(img)
    # MONO (or pseudo-mono)
    if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):
        u = _to_u16(a.squeeze())
        out = _fast_channel_autostretch_u16(u, target_median, sigma)
        return out.astype(np.float32, copy=False)

    # COLOR
    if linked:
        # one set of stats over all channels (flattened)
        u = _to_u16(a)
        # build sample on luminance-like flatten
        # Fast approach: just use channel 0 for stats proxy to avoid big flatten costs
        # or, better: take subsample on all channels concatenated cheaply
        h, w, _ = u.shape
        sy, sx = _choose_stride(h, w, _MAX_STATS_PIXELS // 3)
        sample = u[::sy, ::sx].reshape(-1)  # flatten H*W*3 subsample
        med, std, minv, _ = _stats_from_hist(sample)
        bp = int(max(minv, med - sigma * std))
        lut = _build_lut(bp, target_median, med)
        # apply same LUT per channel
        out = lut[u]
        return out.astype(np.float32, copy=False)
    else:
        # per-channel stats/LUTs
        u = _to_u16(a)
        out = np.empty_like(u, dtype=np.float32)
        for c in range(3):
            out[..., c] = _fast_channel_autostretch_u16(u[..., c], target_median, sigma)
        return out