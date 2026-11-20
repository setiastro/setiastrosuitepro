#saspro/autostretch.py
import os
import threading
import weakref
from collections import OrderedDict

import numpy as np

_MAX_STATS_PIXELS = 1_000_000
_DEFAULT_SIGMA = 3
_U8_MAX  = 4095  # 12-bit output for better gradations than 255
_U24_MAX = 16777215  # 24-bit output for better gradations

# ---------- block-parallel helpers ----------
# The whole-array work below (the 16.7M-entry 24-bit LUT build, float -> level
# conversion, LUT gathers) is element-wise, so it runs in blocks on a small
# thread pool. numpy releases the GIL inside these loops, and a block's
# temporaries stay in cache instead of streaming a full-size intermediate
# through RAM per operation. Every element still goes through the very same
# numpy operations, so the results are bit-identical to the one-shot forms.
_BLOCK_ELEMS = 1 << 17
_POOL = None
_POOL_LOCK = threading.Lock()
_WORKER = threading.local()     # .busy is set on the pool's own threads


def _mark_worker():
    _WORKER.busy = True


def _pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                from concurrent.futures import ThreadPoolExecutor
                n = max(2, min(8, (os.cpu_count() or 2) - 1))
                _POOL = ThreadPoolExecutor(max_workers=n,
                                           thread_name_prefix="saspro-autostretch",
                                           initializer=_mark_worker)
    return _POOL


def _run_blocks(n: int, block: int, fn) -> None:
    """fn(lo, hi) over [0, n) in blocks of `block`, on the pool when there is
    more than one block. fn may only write its own slice. Called from one of
    the pool's own threads it runs inline, so the pool never waits on itself."""
    block = max(1, int(block))
    if n <= 2 * block or getattr(_WORKER, "busy", False):
        fn(0, n)                    # tile-sized job, or already on the pool
        return
    bounds = [(lo, min(n, lo + block)) for lo in range(0, n, block)]
    try:
        futs = [_pool().submit(fn, lo, hi) for lo, hi in bounds]
    except RuntimeError:            # interpreter shutting down: run inline
        for lo, hi in bounds:
            fn(lo, hi)
        return
    for f in futs:
        f.result()


def parallel_map(fn, items) -> list:
    """[fn(x) for x in items], side by side on the pool. For GIL-releasing
    work (numpy, cv2) on independent items; inline for a single item or when
    already on a pool thread."""
    items = list(items)
    if len(items) < 2 or getattr(_WORKER, "busy", False):
        return [fn(x) for x in items]
    try:
        futs = [_pool().submit(fn, x) for x in items]
    except RuntimeError:            # interpreter shutting down: run inline
        return [fn(x) for x in items]
    return [f.result() for f in futs]


def _row_block(a: np.ndarray) -> int:
    """Rows per block when splitting `a` along axis 0."""
    per_row = max(1, int(a.size) // max(1, int(a.shape[0])))
    return max(1, _BLOCK_ELEMS // per_row)


def array_max(a: np.ndarray) -> float:
    """float(a.max()) of a non-empty array, reduced in row blocks on the pool.
    A max is exact and NaN propagates through the block maxima, so the value
    is the one a.max() gives."""
    a = np.asarray(a)
    if a.ndim == 0 or a.size <= 4 * _BLOCK_ELEMS:
        return float(a.max())
    rb = 4 * _row_block(a)
    part = np.empty(-(-a.shape[0] // rb), dtype=a.dtype)

    def _blk(r0, r1):
        part[r0 // rb] = a[r0:r1].max()

    _run_blocks(a.shape[0], rb, _blk)
    return float(part.max())

# ---------- helpers (generic N-level pipeline) ----------
def _to_uN(a: np.ndarray, maxv: int) -> np.ndarray:
    """Convert to uint8/uint16/uint32 [0..maxv] for cheap hist/LUT work."""
    # uint8 for maxv <= 255, uint16 for 256..65535, uint32 for larger (24-bit)
    if maxv > 65535:
        tgt_dtype = np.uint32
    elif maxv > 255:
        tgt_dtype = np.uint16
    else:
        tgt_dtype = np.uint8
    if a.dtype == tgt_dtype:
        return a
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        if info.max <= 0:
            return np.zeros_like(a, dtype=tgt_dtype)
        scaled = np.clip(a.astype(np.float32), 0, info.max) * (maxv / float(info.max))
        # Clamp to maxv to avoid index-out-of-bounds when used as LUT indices
        return np.minimum((scaled + 0.5).astype(tgt_dtype), maxv)
    # float-ish
    if a.ndim == 0 or a.size <= _BLOCK_ELEMS:
        return _float_to_uN(a, maxv, tgt_dtype)
    out = np.empty(a.shape, dtype=tgt_dtype)

    def _blk(r0, r1):
        out[r0:r1] = _float_to_uN(a[r0:r1], maxv, tgt_dtype)

    _run_blocks(a.shape[0], _row_block(a), _blk)
    return out

def _float_to_uN(a: np.ndarray, maxv: int, tgt_dtype) -> np.ndarray:
    af = np.clip(a.astype(np.float32), 0.0, 1.0)
    # Clamp to maxv: when af=1.0, af*maxv+0.5 can exceed maxv
    return np.minimum((af * maxv + 0.5).astype(tgt_dtype), maxv)

def _choose_stride(h: int, w: int, max_pixels: int) -> tuple[int, int]:
    n = h * w
    if n <= max_pixels:
        return 1, 1
    s = max(1, int(np.sqrt(n / float(max_pixels))))
    return s, s

def _bincount_nonzero(uN: np.ndarray, maxv: int) -> np.ndarray:
    """Histogram of levels [0..maxv], excluding level-0 (true black / dropout) pixels."""
    hist = np.bincount(uN.ravel(), minlength=maxv + 1)
    hist[0] = 0  # ignore zero-valued pixels (stacking border dropouts, padding, etc.)
    return hist

_SPARSE_STATS_MAX = 4_000_000   # samples above this use the dense histogram

def _sparse_bp_med(sample_uN: np.ndarray, sigma: float, maxv: int,
                   qfloor: float, no_black_clip: bool):
    """(bp, med) exactly as the dense code in _compute_lut_from_sample finds
    them, from the sorted nonzero levels of a SMALL sample: the view stretches
    from ~65k pixels, and a (maxv+1)-bin histogram plus its cumsum is 16.7M
    entries at 24 bit -- ~0.25 s per channel to read off a handful of numbers.

    The median and the floor quantile are order statistics of the sorted
    levels. The lower-half mean is an exact integer sum. The lower-half
    variance is the same dense float64 expression over [0, med] the original
    evaluates, so its rounding (pairwise summation) matches too.
    None = not applicable, use the dense path."""
    v = np.asarray(sample_uN)
    if (v.size == 0 or v.size > _SPARSE_STATS_MAX
            or not np.issubdtype(v.dtype, np.integer)):
        return None
    s = np.sort(v[v != 0], axis=None)      # level 0 = no-data, excluded
    total = int(s.size)
    if total == 0 or int(s[-1]) > maxv:
        return None
    med = int(s[(total + 1) // 2 - 1])
    if no_black_clip:
        return int(s[0]), med
    n_low = int(np.searchsorted(s, med, side="right"))
    if n_low * med >= 2 ** 53:             # integer sum no longer exact in float64
        return None
    low = s[:n_low]
    total_low = n_low or 1
    mean_low = float(np.float64(int(low.sum(dtype=np.uint64))) / total_low)
    hist_low = np.bincount(low, minlength=med + 1)
    bins = np.arange(med + 1, dtype=np.float64)
    var_low = float((hist_low * (bins - mean_low)**2).sum() / total_low)
    std_low = float(np.sqrt(max(1e-12, var_low)))
    k = int(qfloor * total)
    if k > total:
        return None
    floor_idx = int(s[k - 1]) if k > 0 else 0
    bp = int(max(floor_idx, med - sigma * std_low))
    bp = int(np.clip(bp, 0, maxv - 1))
    return bp, med

def _compute_lut_from_sample(sample_uN: np.ndarray, target: float, sigma: float, maxv: int,
                             qfloor: float = 0.001, no_black_clip: bool = False) -> np.ndarray:
    fast = _sparse_bp_med(sample_uN, sigma, maxv, qfloor, no_black_clip)
    if fast is not None:
        return _build_lut_generic(fast[0], target, fast[1], maxv)

    hist = _bincount_nonzero(sample_uN, maxv)
    total = int(hist.sum()) or 1

    cdf = np.cumsum(hist)
    med = int(np.searchsorted(cdf, (total + 1) // 2))

    if no_black_clip:
        # Use true image minimum (excluding zero dropout) as black point — no sigma clipping
        nz = np.argmax(hist > 0)
        bp = int(nz) if hist[nz] > 0 else 0
    else:
        # only bins[:med + 1] is ever used; same values as slicing a full arange
        bins      = np.arange(min(med + 1, maxv + 1), dtype=np.float64)
        hist_low  = hist[:med + 1]
        total_low = int(hist_low.sum()) or 1
        mean_low  = float((hist_low * bins).sum() / total_low)
        var_low   = float((hist_low * (bins - mean_low)**2).sum() / total_low)
        std_low   = float(np.sqrt(max(1e-12, var_low)))
        floor_idx = int(np.searchsorted(cdf, int(qfloor * total)))
        bp = int(max(floor_idx, med - sigma * std_low))
        bp = int(np.clip(bp, 0, maxv - 1))

    return _build_lut_generic(bp, target, med, maxv)

def _stats_from_hist_generic(uN: np.ndarray, maxv: int) -> tuple[float, float, int, int]:
    """(median, std, minv, maxv) using a histogram of levels [0..maxv], excluding zero pixels."""
    hist = _bincount_nonzero(uN, maxv)
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

# Built LUTs, keyed by everything that decides their bits: a LUT is a pure
# function of (bp, target, median, maxv), so toggling the display stretch off
# and on, flipping a profile back, or undoing to an earlier image finds the
# same four numbers and reuses the table instead of rebuilding 16.7M entries.
# Bounded by bytes (a 24-bit LUT is 64 MB); views hold their own references,
# so this only keeps recently dropped tables alive.
_LUT_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_LUT_CACHE_BYTES = 256 * 1024 * 1024
_LUT_CACHE_LOCK = threading.Lock()

def _lut_key(bp, target_median, med_uN, maxv):
    """Exact cache key, or None when an argument is not a plain Python number
    (a numpy scalar can change the dtype the curve is computed in)."""
    def k(v):
        if type(v) is int:
            return ("i", v)
        if type(v) is float:
            return ("f", v.hex())
        raise TypeError
    try:
        return (k(bp), k(target_median), k(med_uN), k(maxv))
    except TypeError:
        return None

def _build_lut_generic(bp: int, target_median: float, med_uN: float, maxv: int) -> np.ndarray:
    key = _lut_key(bp, target_median, med_uN, maxv)
    if key is not None:
        with _LUT_CACHE_LOCK:
            lut = _LUT_CACHE.get(key)
            if lut is not None:
                _LUT_CACHE.move_to_end(key)
                return lut
    lut = _build_lut_uncached(bp, target_median, med_uN, maxv)
    if key is not None:
        with _LUT_CACHE_LOCK:
            _LUT_CACHE[key] = lut
            _LUT_CACHE.move_to_end(key)
            total = sum(v.nbytes for v in _LUT_CACHE.values())
            while total > _LUT_CACHE_BYTES and len(_LUT_CACHE) > 1:
                _k, old = _LUT_CACHE.popitem(last=False)
                total -= old.nbytes
    return lut

def _build_lut_uncached(bp: int, target_median: float, med_uN: float, maxv: int) -> np.ndarray:
    denom_bp = max(1, maxv - int(bp))
    median_rescaled = (med_uN - bp) / float(denom_bp)
    median_rescaled = float(np.clip(median_rescaled, 1e-9, 1.0))

    def _curve(x):
        # ✅ KEY FIX: clamp r to [0,1] so x<=bp -> r=0 (maps to 0), x>=max -> r=1
        r = np.clip((x - bp) / float(denom_bp), 0.0, 1.0)

        denom = median_rescaled * (target_median + r - 1.0) - target_median * r
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        out = ((median_rescaled - 1.0) * target_median * r) / denom
        return np.clip(out, 0.0, 1.0)

    n = maxv + 1
    if n <= _BLOCK_ELEMS:
        lut = _curve(np.arange(n, dtype=np.float32)).astype(np.float32)
    elif (_lut_key(bp, target_median, med_uN, maxv) is not None
          and _float32_chain(bp, float(denom_bp), median_rescaled, target_median,
                             (median_rescaled - 1.0) * target_median)):
        # 24 bit, and every step of _curve is a float32 op under this numpy:
        # the same steps done in place in reused buffers
        lut = _curve_f32_blocks(bp, float(denom_bp), median_rescaled, target_median, n)
    else:
        # 24 bit: 16.7M entries, built in cache-sized blocks on the pool
        # (x is the same exact-integer float32 ramp, block by block)
        lut = np.empty(n, dtype=np.float32)

        def _blk(lo, hi):
            lut[lo:hi] = _curve(np.arange(lo, hi, dtype=np.float32))

        _run_blocks(n, _BLOCK_ELEMS, _blk)
    lut[0] = 0.0  # keep true zero pixels mapped to 0 regardless of stretch
    # Shared, never edited in place: lets _lut8_of() cache its uint8 twin.
    lut.setflags(write=False)
    return lut

_LUT_BLOCK = 1 << 18
_SCRATCH = threading.local()

def _float32_chain(*scalars) -> bool:
    """True when a float32 array combined with each of these Python scalars
    stays float32 under the running numpy's promotion: always on numpy 2 (a
    Python scalar is weak, NEP 50), but numpy 1's value-based casting turns
    e.g. `x - 5592405` (an int above 65535) into float64 -- and then the
    original expression runs in float64, which the float32 in-place path
    would not reproduce."""
    z = np.zeros(1, dtype=np.float32)
    try:
        return all(np.result_type(z, s) == np.float32 for s in scalars)
    except Exception:
        return False

def _curve_f32_blocks(bp: int, fdenom: float, m: float, t: float, n: int) -> np.ndarray:
    """_curve(np.arange(n, dtype=float32)) of _build_lut_uncached for plain
    Python bp/target/median, where every step of the expression is a float32
    operation: the same steps, in the same order, with each commuted operand
    pair exact (IEEE + and * commute), run in place in per-thread buffers. The
    one-shot expression's ~12 fresh temporaries per block were most of the
    build time. Bit-identical to _curve; checked over a parameter sweep."""
    c = (m - 1.0) * t                   # Python evaluates this pair first
    lut = np.empty(n, dtype=np.float32)

    def _blk(lo, hi):
        k = hi - lo
        bufs = getattr(_SCRATCH, "bufs", None)
        if bufs is None or bufs[0].shape[0] < k:
            size = max(k, _LUT_BLOCK)
            bufs = _SCRATCH.bufs = (
                np.arange(size, dtype=np.float32),          # ramp 0..size-1
                np.empty(size, dtype=np.float32),
                np.empty(size, dtype=np.float32),
                np.empty(size, dtype=np.float32),
                np.empty(size, dtype=np.bool_),
            )
        ramp, x, r, d, tiny = (b[:k] for b in bufs)
        np.add(ramp, lo, out=x)          # == np.arange(lo, hi): exact integers
        np.subtract(x, bp, out=r)        # r = clip((x - bp) / float(denom_bp), 0, 1)
        np.divide(r, fdenom, out=r)
        np.clip(r, 0.0, 1.0, out=r)
        np.add(r, t, out=d)              # denom = m * (t + r - 1.0) - t * r
        np.subtract(d, 1.0, out=d)
        np.multiply(d, m, out=d)
        np.multiply(r, t, out=x)
        np.subtract(d, x, out=d)
        np.abs(d, out=x)                 # denom = where(|denom| < 1e-12, 1e-12, denom)
        np.less(x, 1e-12, out=tiny)
        np.copyto(d, 1e-12, where=tiny)
        np.multiply(r, c, out=r)         # out = ((m - 1.0) * t * r) / denom
        np.divide(r, d, out=r)
        np.clip(r, 0.0, 1.0, out=lut[lo:hi])

    _run_blocks(n, _LUT_BLOCK, _blk)
    return lut

def _fast_channel_autostretch_uN(ch_uN: np.ndarray, target: float, sigma: float, maxv: int,
                                 qfloor: float = 0.001) -> np.ndarray:
    """
    qfloor: low-percentile floor (e.g. 0.1%) so BP doesn't peg to minv=0.
    """
    # Subsample
    h, w = ch_uN.shape
    sy, sx = _choose_stride(h, w, _MAX_STATS_PIXELS)
    sample = ch_uN[::sy, ::sx]

    # Histogram on the sample, excluding zero-valued (dropout/border) pixels
    hist = _bincount_nonzero(sample, maxv)
    total = int(hist.sum()) or 1

    # CDF + median index
    cdf = np.cumsum(hist)
    med = int(np.searchsorted(cdf, (total + 1)//2))

    # Robust std: only use bins <= median (avoids bright-tail inflation)
    bins      = np.arange(maxv + 1, dtype=np.float64)
    hist_low  = hist[:med + 1]
    total_low = int(hist_low.sum()) or 1
    mean_low  = float((hist_low * bins[:med + 1]).sum() / total_low)
    var_low   = float((hist_low * (bins[:med + 1] - mean_low)**2).sum() / total_low)
    std_low   = float(np.sqrt(max(1e-12, var_low)))

    # Percentile floor (avoid pegging to 0)
    floor_idx = int(np.searchsorted(cdf, int(qfloor * total)))

    # Black point driven by sigma, but never below the floor
    bp = int(max(floor_idx, med - sigma * std_low))
    bp = int(np.clip(bp, 0, maxv - 1))

    # Build LUT and map
    lut = _build_lut_generic(bp, target, med, maxv)
    return lut[ch_uN]

def _apply_maxv(lut_cache, use_24bit) -> int:
    """Level count to apply lut_cache at. With use_24bit=None the LUT's own
    length decides (4096 or 16777216 entries): the display/autostretch_24bit
    setting can change after a LUT was built, and applying a 24-bit LUT at 12
    bit paints the view near-black (the reverse raises IndexError)."""
    if use_24bit is None:
        lut0 = lut_cache[0] if isinstance(lut_cache, (tuple, list)) else lut_cache
        n = int(lut0.shape[0]) if isinstance(lut0, np.ndarray) and lut0.ndim == 1 else -1
        if n == _U24_MAX + 1:
            return _U24_MAX
        if n == _U8_MAX + 1:
            return _U8_MAX
        try:
            from PyQt6.QtCore import QSettings
            use_24bit = QSettings().value("display/autostretch_24bit", True, type=bool)
        except Exception:
            use_24bit = True
    return _U24_MAX if use_24bit else _U8_MAX

def _build_maxv(use_24bit, use_16bit) -> int:
    """Level count to build a LUT at (autostretch_with_lut's rule)."""
    if use_16bit is not None and use_24bit is None:
        use_24bit = True

    if use_24bit is None:
        try:
            from PyQt6.QtCore import QSettings
            use_24bit = QSettings().value("display/autostretch_24bit", True, type=bool)
        except Exception:
            use_24bit = True

    return _U24_MAX if use_24bit else _U8_MAX

def _lut_take(lut, u: np.ndarray) -> np.ndarray:
    """lut[u].astype(np.float32, copy=False), gathered in row blocks."""
    if not isinstance(lut, np.ndarray) or u.ndim == 0 or u.size <= _BLOCK_ELEMS:
        return lut[u].astype(np.float32, copy=False)
    out = np.empty(u.shape, dtype=np.float32)

    def _blk(r0, r1):
        out[r0:r1] = lut[u[r0:r1]]

    _run_blocks(u.shape[0], _row_block(u), _blk)
    return out

def _fill_color(out: np.ndarray, u: np.ndarray, lut_cache, linked: bool, maxv: int) -> None:
    """The colour body of apply_autostretch_lut, in row blocks: out is
    float32 shaped like u (H, W, C)."""
    C = u.shape[2]

    def _blk(r0, r1):
        ub = u[r0:r1]
        if linked:
            out[r0:r1, :, :3] = lut_cache[ub[..., :3]]
        else:
            for c in range(min(3, C)):
                out[r0:r1, :, c] = lut_cache[c][ub[..., c]]
        if C > 3:
            out[r0:r1, :, 3:] = ub[..., 3:] / float(maxv)

    _run_blocks(u.shape[0], _row_block(u), _blk)

def apply_autostretch_lut(
    img: np.ndarray,
    lut_cache,
    *,
    linked: bool = False,
    use_24bit: bool | None = None,
) -> np.ndarray:
    if img is None:
        return None

    maxv = _apply_maxv(lut_cache, use_24bit)
    a = np.asarray(img)

    if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):
        u = _to_uN(a.squeeze(), maxv)
        return _lut_take(lut_cache, u)

    u = _to_uN(a, maxv)
    out = np.empty_like(u, dtype=np.float32)
    _fill_color(out, u, lut_cache, linked, maxv)
    return out

_LUT8_CACHE: dict = {}

def _lut8_of(lut) -> np.ndarray:
    """uint8 twin of a float display LUT: the exact conversion to_uint8_rgb()
    applies to apply_autostretch_lut()'s float32 output, done once per LUT
    entry, so lut8[u] == to_uint8(lut[u]) element for element. Kept for the
    LUT's lifetime when the LUT is read-only (every LUT built here is)."""
    if not isinstance(lut, np.ndarray) or lut.ndim != 1:
        raise TypeError("a display LUT is a 1-D array")
    key = id(lut)
    hit = _LUT8_CACHE.get(key)
    if hit is not None and hit[0]() is lut:
        return hit[1]
    lut8 = np.empty(lut.shape, dtype=np.uint8)

    def _blk(lo, hi):
        lut8[lo:hi] = (np.clip(lut[lo:hi].astype(np.float32, copy=False), 0.0, 1.0) * 255.0).astype(np.uint8)

    _run_blocks(lut.shape[0], _BLOCK_ELEMS, _blk)
    if not lut.flags.writeable:
        def _gone(ref, key=key):
            cur = _LUT8_CACHE.get(key)
            if cur is not None and cur[0] is ref:
                _LUT8_CACHE.pop(key, None)
        _LUT8_CACHE[key] = (weakref.ref(lut, _gone), lut8)
    return lut8

def apply_autostretch_lut_u8(
    img: np.ndarray,
    lut_cache,
    *,
    linked: bool = False,
    use_24bit: bool | None = None,
) -> np.ndarray:
    """to_uint8_rgb(apply_autostretch_lut(img, lut_cache, ...)) in one pass:
    (H, W, 3) C-contiguous uint8, bit-identical, without the float32 image in
    between. Levels go straight to display bytes through the LUT's uint8 twin
    (16 MB at 24 bit against the float LUT's 64 MB, so the gather mostly hits
    cache). Unusual shapes take the two-step path."""
    if img is None:
        return None
    a = np.asarray(img)
    mono = a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1)
    m = a.squeeze() if mono else None
    if (m.ndim != 2) if mono else not (a.ndim == 3 and a.shape[2] >= 3):
        from setiastro.saspro.imageops.viewport_render import to_uint8_rgb
        return to_uint8_rgb(apply_autostretch_lut(img, lut_cache, linked=linked,
                                                  use_24bit=use_24bit))
    maxv = _apply_maxv(lut_cache, use_24bit)

    if mono:
        lut8 = _lut8_of(lut_cache)
        out = np.empty(m.shape + (3,), dtype=np.uint8)

        def _blk(r0, r1):
            out[r0:r1] = lut8[_to_uN(m[r0:r1], maxv)][..., None]

        _run_blocks(m.shape[0], _row_block(m), _blk)
        return out

    src = a if a.shape[2] == 3 else a[..., :3]
    if linked:
        lut8 = _lut8_of(lut_cache)
    else:
        lut8s = [_lut8_of(lut_cache[c]) for c in range(3)]
    out = np.empty(a.shape[:2] + (3,), dtype=np.uint8)

    def _blk(r0, r1):
        ub = _to_uN(src[r0:r1], maxv)
        if linked:
            out[r0:r1] = lut8[ub]
        else:
            for c in range(3):
                out[r0:r1, :, c] = lut8s[c][ub[..., c]]

    _run_blocks(src.shape[0], _row_block(src), _blk)
    return out

def _luts_from_levels(u: np.ndarray, mono: bool, target_median, linked: bool,
                      sigma, maxv: int, no_black_clip: bool):
    """The LUT(s) autostretch_with_lut derives from the levelled image u."""
    if mono:
        return _compute_lut_from_sample(u, target_median, sigma, maxv, no_black_clip=no_black_clip)

    if linked:
        h, w, _ = u.shape
        sy, sx = _choose_stride(h, w, max(1, _MAX_STATS_PIXELS // 3))
        sample = u[::sy, ::sx]
        lum = (0.2126 * sample[..., 0] + 0.7152 * sample[..., 1] + 0.0722 * sample[..., 2]).astype(u.dtype)
        return _compute_lut_from_sample(lum, target_median, sigma, maxv, no_black_clip=no_black_clip)

    return tuple(
        _compute_lut_from_sample(u[..., c], target_median, sigma, maxv, no_black_clip=no_black_clip)
        for c in range(min(3, u.shape[2]))
    )

def autostretch_lut(
    img: np.ndarray,
    target_median: float = 0.25,
    linked: bool = False,
    sigma: float = _DEFAULT_SIGMA,
    *,
    use_24bit: bool | None = None,
    use_16bit: bool | None = None,
    no_black_clip: bool = False,
    **_ignored_kwargs,
):
    """Only the LUT(s) autostretch_with_lut(img, ...) returns, without
    stretching img itself (the view measures a sample and applies the LUT to
    what it displays)."""
    if img is None:
        return None
    maxv = _build_maxv(use_24bit, use_16bit)
    a = np.asarray(img)
    mono = a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1)
    u = _to_uN(a.squeeze() if mono else a, maxv)
    return _luts_from_levels(u, mono, target_median, linked, sigma, maxv, no_black_clip)

def autostretch_with_lut(
    img: np.ndarray,
    target_median: float = 0.25,
    linked: bool = False,
    sigma: float = _DEFAULT_SIGMA,
    *,
    use_24bit: bool | None = None,
    use_16bit: bool | None = None,
    no_black_clip: bool = False,
    **_ignored_kwargs,
):
    if img is None:
        return None, None

    maxv = _build_maxv(use_24bit, use_16bit)
    a = np.asarray(img)

    if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):
        u = _to_uN(a.squeeze(), maxv)
        lut = _luts_from_levels(u, True, target_median, linked, sigma, maxv, no_black_clip)
        return _lut_take(lut, u), lut

    u = _to_uN(a, maxv)
    lut = _luts_from_levels(u, False, target_median, linked, sigma, maxv, no_black_clip)
    out = np.empty_like(u, dtype=np.float32)
    _fill_color(out, u, lut, linked, maxv)
    return out, lut

# ---------- public API ----------
def autostretch(
    img: np.ndarray,
    target_median: float = 0.25,
    linked: bool = False,
    sigma: float = _DEFAULT_SIGMA,
    *,
    use_24bit: bool | None = None,
    use_16bit: bool | None = None,
    no_black_clip: bool = False,
    **_ignored_kwargs,
) -> np.ndarray:
    out, _lut = autostretch_with_lut(
        img,
        target_median=target_median,
        linked=linked,
        sigma=sigma,
        use_24bit=use_24bit,
        use_16bit=use_16bit,
        no_black_clip=no_black_clip,
        **_ignored_kwargs,
    )
    return out