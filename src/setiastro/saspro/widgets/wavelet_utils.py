#saspro/widgets/wavelet_utils.py
"""
Shared wavelet utilities for à-trous decomposition and reconstruction.

This module provides centralized implementations for wavelet operations
used across wavescale_hdr.py, wavescalede.py, and other modules.
"""
from __future__ import annotations
import os
import threading
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Convolution helpers (SciPy if available; otherwise a separable fallback)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.ndimage import convolve as _nd_convolve
    from scipy.ndimage import gaussian_filter as _nd_gauss
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    _nd_convolve = None
    _nd_gauss = None

try:
    from scipy.ndimage._filters import _gaussian_kernel1d as _nd_gauss_kernel1d
except Exception:
    _nd_gauss_kernel1d = None


# ─────────────────────────────────────────────────────────────────────────────
# Fast exact paths: numba kernels that release the GIL, run on row bands
#
# Bit-identical stand-ins for the two SciPy filters above, 2-D float32 only.
# They accumulate in float64 in exactly SciPy's order:
#   conv_sep_reflect -> ndimage.convolve (N-D correlate): kernel flipped, taps
#       in kernel order, zero taps skipped, acc = 0.0 then acc += w * x.
#   gauss_blur -> ndimage.gaussian_filter: correlate1d on axis 0 then axis 1,
#       float32 in between; symmetric-kernel path acc = x * w[c], then
#       acc += (left + right) * w[d], outermost pair first.
# Border: 'reflect' (d c b a | a b c d | d c b a).  Anything outside that
# envelope (other dtypes, kernel radius > image side, no numba) goes to SciPy.
# Speed: ~10x per pass, and a worker thread using them no longer holds the
# GIL, so the GUI keeps painting.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from numba import njit as _njit
except Exception:
    _njit = None

_FAST_OK = bool(_HAVE_SCIPY and _njit is not None)

if _FAST_OK:
    @_njit(nogil=True, cache=True)
    def _nb_corr_x(src, out, taps, ws, xmap, r0, r1):
        # taps index the reflect-padded row: tap t reads pad[x + taps[t]]
        W = src.shape[1]
        P = xmap.shape[0]
        T = taps.shape[0]
        pad = np.empty(P, dtype=np.float64)
        acc = np.empty(W, dtype=np.float64)
        for y in range(r0, r1):
            for p in range(P):
                pad[p] = np.float64(src[y, xmap[p]])
            for x in range(W):
                acc[x] = 0.0
            for t in range(T):
                w = ws[t]
                o = taps[t]
                for x in range(W):
                    acc[x] += w * pad[x + o]
            for x in range(W):
                out[y, x] = np.float32(acc[x])

    @_njit(nogil=True, cache=True)
    def _nb_corr_y(src, out, taps, ws, ymap, r0, r1):
        W = src.shape[1]
        T = taps.shape[0]
        acc = np.empty(W, dtype=np.float64)
        for y in range(r0, r1):
            for x in range(W):
                acc[x] = 0.0
            for t in range(T):
                w = ws[t]
                yy = ymap[y + taps[t]]
                for x in range(W):
                    acc[x] += w * np.float64(src[yy, x])
            for x in range(W):
                out[y, x] = np.float32(acc[x])

    @_njit(nogil=True, cache=True)
    def _nb_corr_y_detail(src, cur, out, det, taps, ws, ymap, r0, r1):
        # _nb_corr_y, plus det = cur - out (float32, as numpy's current - smooth)
        W = src.shape[1]
        T = taps.shape[0]
        acc = np.empty(W, dtype=np.float64)
        for y in range(r0, r1):
            for x in range(W):
                acc[x] = 0.0
            for t in range(T):
                w = ws[t]
                yy = ymap[y + taps[t]]
                for x in range(W):
                    acc[x] += w * np.float64(src[yy, x])
            for x in range(W):
                s = np.float32(acc[x])
                out[y, x] = s
                det[y, x] = cur[y, x] - s

    @_njit(nogil=True, cache=True)
    def _nb_sym_x(src, out, w, xmap, r0, r1):
        W = src.shape[1]
        P = xmap.shape[0]
        R = w.shape[0] // 2
        wc = w[R]
        pad = np.empty(P, dtype=np.float64)
        acc = np.empty(W, dtype=np.float64)
        for y in range(r0, r1):
            for p in range(P):
                pad[p] = np.float64(src[y, xmap[p]])
            for x in range(W):
                acc[x] = pad[x + R] * wc
            for d in range(R, 0, -1):
                wd = w[R - d]
                for x in range(W):
                    acc[x] += (pad[x + R - d] + pad[x + R + d]) * wd
            for x in range(W):
                out[y, x] = np.float32(acc[x])

    @_njit(nogil=True, cache=True)
    def _nb_sym_y(src, out, w, ymap, r0, r1):
        W = src.shape[1]
        R = w.shape[0] // 2
        wc = w[R]
        acc = np.empty(W, dtype=np.float64)
        for y in range(r0, r1):
            for x in range(W):
                acc[x] = np.float64(src[y, x]) * wc
            for d in range(R, 0, -1):
                wd = w[R - d]
                ya = ymap[y + R - d]
                yb = ymap[y + R + d]
                for x in range(W):
                    acc[x] += (np.float64(src[ya, x]) + np.float64(src[yb, x])) * wd
            for x in range(W):
                out[y, x] = np.float32(acc[x])


_POOL = None
_POOL_LOCK = threading.Lock()
_POOL_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _row_pool():
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                from concurrent.futures import ThreadPoolExecutor
                _POOL = ThreadPoolExecutor(max_workers=_POOL_WORKERS,
                                           thread_name_prefix="wavelet")
    return _POOL


def _run_row_bands(kern, H: int, W: int, args: tuple) -> None:
    """kern(*args, r0, r1) over row bands; bands only write their own rows."""
    if _POOL_WORKERS <= 1 or H * W < (1 << 18) or H < 2:
        kern(*args, 0, H)
        return
    nb = min(H, _POOL_WORKERS * 4)
    step = -(-H // nb)
    pool = _row_pool()
    futs = [pool.submit(kern, *args, r0, min(H, r0 + step)) for r0 in range(0, H, step)]
    for f in futs:
        f.result()


def run_row_bands(fn, H: int) -> None:
    """Call fn(slice(r0, r1)) for row bands covering 0..H, in the shared pool.

    For elementwise numpy work on large images: numpy releases the GIL in its
    loops, so the bands run in parallel and the result is the same as one call
    over all rows. Each band must write only its own rows, and fn must not call
    anything that uses this pool itself (conv_sep_reflect, gauss_blur).
    """
    H = int(H)
    if _POOL_WORKERS <= 1 or H < 64:
        fn(slice(0, H))
        return
    nb = min(H // 16, _POOL_WORKERS * 2)
    step = -(-H // nb)
    pool = _row_pool()
    futs = [pool.submit(fn, slice(r0, min(H, r0 + step))) for r0 in range(0, H, step)]
    for f in futs:
        f.result()


def _reflect_index_map(n: int, R: int) -> np.ndarray:
    """Source index for padded positions -R .. n-1+R (ndimage 'reflect'; R <= n)."""
    idx = np.arange(-R, n + R, dtype=np.int64) % (2 * n)
    return np.ascontiguousarray(np.where(idx >= n, 2 * n - 1 - idx, idx).astype(np.int64))


_SELF_CHECK = None      # None: not run yet; then True/False


def _fast_paths_verified() -> bool:
    """One-time check that the fast paths reproduce the installed SciPy bit for
    bit (guards against a SciPy whose filters or private kernel helper changed).
    On any mismatch or error the fast paths stay off and SciPy is used."""
    global _SELF_CHECK, _FAST_OK
    if _SELF_CHECK is None:
        ok = False
        try:
            rng = np.random.default_rng(12345)
            x = rng.random((67, 71), dtype=np.float32)
            x[::7, ::5] = 0.0
            ok = True
            for s in range(3):
                k = build_spaced_kernel(B3_KERNEL, s)
                for ax in (0, 1):
                    kk = k.reshape(1, -1) if ax == 1 else k.reshape(-1, 1)
                    a = _nd_convolve(x, kk, mode="reflect")
                    b = _conv_sep_reflect_fast_unchecked(x, k, ax)
                    ok = ok and b is not None and np.array_equal(a.view(np.uint32), b.view(np.uint32))
                t = _nd_convolve(x, k.reshape(1, -1), mode="reflect")
                a = _nd_convolve(t, k.reshape(-1, 1), mode="reflect")
                sd = _smooth_and_detail_fast_unchecked(t, x, k)
                ok = (ok and sd is not None
                      and np.array_equal(a.view(np.uint32), sd[0].view(np.uint32))
                      and np.array_equal((x - a).view(np.uint32), sd[1].view(np.uint32)))
            if _nd_gauss_kernel1d is not None:
                for sg in (3.0, 1.3):
                    a = _nd_gauss(x, sigma=sg, mode="reflect")
                    b = _gauss_blur_fast_unchecked(x, sg)
                    ok = ok and b is not None and np.array_equal(a.view(np.uint32), b.view(np.uint32))
        except Exception:
            ok = False
        _SELF_CHECK = bool(ok)
        if not ok:
            _FAST_OK = False
    return bool(_SELF_CHECK)


def _fast_2d_f32(image2d) -> bool:
    return (_FAST_OK and isinstance(image2d, np.ndarray) and image2d.ndim == 2
            and image2d.dtype == np.float32 and image2d.size > 0)


def _conv_sep_reflect_fast(image2d: np.ndarray, k1d: np.ndarray, axis: int):
    """Exact ndimage.convolve(image2d, k1d along axis, mode='reflect'), or None."""
    if not _fast_2d_f32(image2d) or not _fast_paths_verified():
        return None
    return _conv_sep_reflect_fast_unchecked(image2d, k1d, axis)


def _conv_sep_reflect_fast_unchecked(image2d: np.ndarray, k1d: np.ndarray, axis: int):
    global _FAST_OK
    if not _fast_2d_f32(image2d) or axis not in (0, 1):
        return None
    k = np.asarray(k1d)
    if k.ndim != 1 or (k.shape[0] & 1) == 0 or k.dtype.kind not in "fiu":
        return None
    w = np.asarray(k, dtype=np.float64)[::-1]      # convolve = correlate, kernel flipped
    R = w.shape[0] // 2
    H, W = image2d.shape
    if R > (H if axis == 0 else W):
        return None
    keep = np.abs(w) > np.finfo(np.float64).eps   # SciPy's footprint (zero taps skipped)
    taps = np.ascontiguousarray(np.nonzero(keep)[0].astype(np.int64))
    ws = np.ascontiguousarray(w[keep])
    if taps.size == 0:
        return None
    src = np.ascontiguousarray(image2d)
    out = np.empty((H, W), dtype=np.float32)
    try:
        if axis == 1:
            _run_row_bands(_nb_corr_x, H, W, (src, out, taps, ws, _reflect_index_map(W, R)))
        else:
            _run_row_bands(_nb_corr_y, H, W, (src, out, taps, ws, _reflect_index_map(H, R)))
    except Exception:
        _FAST_OK = False                              # e.g. numba cannot compile here
        return None
    return out


def _smooth_and_detail_fast(tmp: np.ndarray, cur: np.ndarray, k1d: np.ndarray):
    """(conv_sep_reflect(tmp, k1d, 0), cur - that) in one pass, or None."""
    if not _fast_2d_f32(tmp) or not _fast_paths_verified():
        return None
    return _smooth_and_detail_fast_unchecked(tmp, cur, k1d)


def _smooth_and_detail_fast_unchecked(tmp: np.ndarray, cur: np.ndarray, k1d: np.ndarray):
    global _FAST_OK
    if not (_fast_2d_f32(tmp) and _fast_2d_f32(cur) and tmp.shape == cur.shape):
        return None
    k = np.asarray(k1d)
    if k.ndim != 1 or (k.shape[0] & 1) == 0 or k.dtype.kind not in "fiu":
        return None
    w = np.asarray(k, dtype=np.float64)[::-1]
    R = w.shape[0] // 2
    H, W = tmp.shape
    if R > H:
        return None
    keep = np.abs(w) > np.finfo(np.float64).eps
    taps = np.ascontiguousarray(np.nonzero(keep)[0].astype(np.int64))
    ws = np.ascontiguousarray(w[keep])
    if taps.size == 0:
        return None
    src = np.ascontiguousarray(tmp)
    c = np.ascontiguousarray(cur)
    out = np.empty((H, W), dtype=np.float32)
    det = np.empty((H, W), dtype=np.float32)
    try:
        _run_row_bands(_nb_corr_y_detail, H, W, (src, c, out, det, taps, ws, _reflect_index_map(H, R)))
    except Exception:
        _FAST_OK = False
        return None
    return out, det


def _gauss_blur_fast(image2d: np.ndarray, sigma: float):
    """Exact ndimage.gaussian_filter(image2d, sigma, mode='reflect'), or None."""
    if not _fast_2d_f32(image2d) or not _fast_paths_verified():
        return None
    return _gauss_blur_fast_unchecked(image2d, sigma)


def _gauss_blur_fast_unchecked(image2d: np.ndarray, sigma: float):
    global _FAST_OK
    if not _fast_2d_f32(image2d) or _nd_gauss_kernel1d is None:
        return None
    try:
        sd = float(sigma)
    except Exception:
        return None
    if not (sd > 1e-15):
        return None
    lw = int(4.0 * sd + 0.5)                          # gaussian_filter1d, truncate=4.0
    if lw < 1:
        return None
    w = np.ascontiguousarray(np.asarray(_nd_gauss_kernel1d(sigma, 0, lw)[::-1], dtype=np.float64))
    R = w.shape[0] // 2
    eps = np.finfo(np.float64).eps
    for ii in range(1, R + 1):                       # correlate1d's symmetric-path test
        if abs(w[R + ii] - w[R - ii]) > eps:
            return None
    H, W = image2d.shape
    if R > H or R > W:
        return None
    src = np.ascontiguousarray(image2d)
    tmp = np.empty((H, W), dtype=np.float32)
    out = np.empty((H, W), dtype=np.float32)
    try:
        _run_row_bands(_nb_sym_y, H, W, (src, tmp, w, _reflect_index_map(H, R)))
        _run_row_bands(_nb_sym_x, H, W, (tmp, out, w, _reflect_index_map(W, R)))
    except Exception:
        _FAST_OK = False
        return None
    return out


def conv_sep_reflect(image2d: np.ndarray, k1d: np.ndarray, axis: int) -> np.ndarray:
    """
    Separable 1D convolution along a given axis with reflect padding.
    
    Uses scipy.ndimage.convolve if available, otherwise falls back to numpy.
    
    Args:
        image2d: 2D input array
        k1d: 1D kernel
        axis: 0 for vertical (y), 1 for horizontal (x)
        
    Returns:
        Convolved array, same shape as input
    """
    if _HAVE_SCIPY:
        fast = _conv_sep_reflect_fast(image2d, k1d, axis)
        if fast is not None:
            return fast
        if axis == 1:  # x
            # Use 'reflect' mode to match original behavior
            return _nd_convolve(image2d, k1d.reshape(1, -1), mode="reflect")
        else:          # y
            return _nd_convolve(image2d, k1d.reshape(-1, 1), mode="reflect")
    else:
        # Optimization: Use OpenCV if available, falling back to NumPy only if strictly necessary.
        # cv2.sepFilter2D can tackle 1D separable/separable passes easily, 
        # but here we need a specific generic 1D convolution.
        # cv2.filter2D is generic 2D but fast.
        try:
            import cv2
            # cv2.filter2D(src, ddepth, kernel, anchor, delta, borderType)
            # kernel needs to be created correctly.
            if axis == 1: # horizontal
                k = k1d.reshape(1, -1)
            else: # vertical
                k = k1d.reshape(-1, 1)
            
            # Using ddepth=-1 to keep input depth (float32)
            # borderType=cv2.BORDER_REFLECT to match mode="reflect"
            return cv2.filter2D(image2d, -1, k, borderType=cv2.BORDER_REFLECT)
        except ImportError:
            # Fallback numpy implementation (slow!)
            image2d = np.asarray(image2d, dtype=np.float32)
            k1d = np.asarray(k1d, dtype=np.float32)
            r = len(k1d) // 2
            if axis == 1:  # horizontal
                pad = np.pad(image2d, ((0, 0), (r, r)), mode="reflect")
                out = np.empty_like(image2d, dtype=np.float32)
                for i in range(image2d.shape[0]):
                    out[i] = np.convolve(pad[i], k1d, mode="valid")
                return out
            else:          # vertical
                pad = np.pad(image2d, ((r, r), (0, 0)), mode="reflect")
                out = np.empty_like(image2d, dtype=np.float32)
                for j in range(image2d.shape[1]):
                    out[:, j] = np.convolve(pad[:, j], k1d, mode="valid")
                return out


def gauss1d(sigma: float) -> np.ndarray:
    """
    Create a 1D Gaussian kernel.
    
    Args:
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Normalized 1D Gaussian kernel
    """
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma)**2)
    k /= np.sum(k)
    return k.astype(np.float32)


def gauss_blur(image2d: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to a 2D image.
    
    Uses scipy.ndimage.gaussian_filter if available, otherwise separable convolution.
    
    Args:
        image2d: 2D input array
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Blurred array
    """
    if _HAVE_SCIPY and _nd_gauss is not None:
        fast = _gauss_blur_fast(image2d, sigma)
        if fast is not None:
            return fast
        return _nd_gauss(image2d, sigma=sigma, mode="reflect")
    
    # Try OpenCV GaussianBlur first -> much faster
    try:
        import cv2
        # kernel size: sigma*3 or auto (0)
        # cv2 uses (ksize_w, ksize_h), ksize must be odd.
        # To strictly match sigma, simple let cv2 compute it with ksize=(0,0)
        # BUT 'reflect' border is not default (BORDER_DEFAULT is Reflect_101).
        # mode="reflect" in scipy is usually dcb a | abcd | dc
        # cv2.BORDER_REFLECT is fedcba | abcdef | gfedcb
        # They are compatible enough for this application.
        return cv2.GaussianBlur(image2d, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    except ImportError:
        pass
        
    k = gauss1d(float(sigma))
    tmp = conv_sep_reflect(image2d, k, axis=1)
    return conv_sep_reflect(tmp, k, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# À-trous wavelet transform (B3 spline kernel)
# ─────────────────────────────────────────────────────────────────────────────

# Standard B3-spline kernel for à-trous
B3_KERNEL = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


def build_spaced_kernel(kernel: np.ndarray, scale_idx: int) -> np.ndarray:
    """
    Build a spaced (à-trous) kernel for a given scale.
    
    Args:
        kernel: Base 1D kernel
        scale_idx: Scale index (0 = no spacing, 1 = step 2, 2 = step 4, etc.)
        
    Returns:
        Spaced kernel with zeros inserted
    """
    if scale_idx == 0:
        return kernel.astype(np.float32, copy=False)
    step = 2 ** scale_idx
    spaced_len = len(kernel) + (len(kernel) - 1) * (step - 1)
    spaced = np.zeros(spaced_len, dtype=np.float32)
    spaced[0::step] = kernel
    return spaced


def atrous_decompose(img2d: np.ndarray, n_scales: int, 
                     base_kernel: np.ndarray | None = None) -> list[np.ndarray]:
    """
    Perform à-trous (undecimated) wavelet decomposition.
    
    Args:
        img2d: 2D input image
        n_scales: Number of detail scales to extract
        base_kernel: Base kernel (default: B3 spline)
        
    Returns:
        List of [detail_0, detail_1, ..., detail_n-1, residual]
        where detail_i is the wavelet plane at scale i
    """
    if base_kernel is None:
        base_kernel = B3_KERNEL
    
    current = img2d.astype(np.float32, copy=True)
    planes: list[np.ndarray] = []
    
    for s in range(n_scales):
        k = build_spaced_kernel(base_kernel, s)
        tmp = conv_sep_reflect(current, k, axis=1)
        fused = _smooth_and_detail_fast(tmp, current, k)
        if fused is not None:
            smooth, detail = fused
        else:
            smooth = conv_sep_reflect(tmp, k, axis=0)
            detail = current - smooth
        planes.append(detail)  # detail = current - smoothed
        current = smooth
    
    planes.append(current)  # residual (lowest frequency)
    return planes


def atrous_reconstruct(planes: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct image from à-trous wavelet planes.
    
    Args:
        planes: List of [detail_0, ..., detail_n-1, residual]
        
    Returns:
        Reconstructed image
    """
    out = planes[-1].astype(np.float32, copy=True)  # start with residual
    for detail in planes[:-1]:
        out += detail
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Color space utilities (with optional Numba acceleration)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from setiastro.saspro.legacy.numba_utils import (
        rgb_to_xyz_numba, xyz_to_lab_numba,
        lab_to_xyz_numba, xyz_to_rgb_numba,
    )
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# sRGB -> XYZ transformation matrix
_RGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float32)

# XYZ -> sRGB transformation matrix (inverse)
_XYZ_TO_RGB_MATRIX = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252]
], dtype=np.float32)

# D65 illuminant reference white
_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

# The numba Lab kernels hold the GIL for a whole call. Feeding them ~2 MP row
# chunks (the kernels are per-pixel, per-row) lets the GUI thread in between
# when a worker converts a large image; output is identical.
_LAB_CHUNK_PX = 1 << 21


def _lab_chunk_rows(arr: np.ndarray) -> int:
    if arr.ndim != 3:
        return max(1, arr.shape[0])
    return max(1, _LAB_CHUNK_PX // max(1, arr.shape[1]))


# numpy fallback = the numba kernels (legacy/numba_utils) step for step: RGB is
# taken as LINEAR (sRGB primaries, D65, no transfer curve), XYZ in float32, the
# Lab non-linearity in float64 with the white as plain floats. The fallback used
# to apply the sRGB curve, so without numba every WaveScale result looked
# different (grey 0.5 -> L* 53 instead of 76).
_LAB_DELTA = 6.0 / 29.0
_XN, _YN, _ZN = 0.95047, 1.00000, 1.08883


def _rgb_to_lab_np(rgb: np.ndarray) -> np.ndarray:
    M = _RGB_TO_XYZ_MATRIX
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = _LAB_DELTA

    def f(t):
        return np.where(t > d ** 3, np.cbrt(t), t / (3.0 * d * d) + 4.0 / 29.0)

    fx = f((M[0, 0] * r + M[0, 1] * g + M[0, 2] * b).astype(np.float64) / _XN)
    fy = f((M[1, 0] * r + M[1, 1] * g + M[1, 2] * b).astype(np.float64) / _YN)
    fz = f((M[2, 0] * r + M[2, 1] * g + M[2, 2] * b).astype(np.float64) / _ZN)
    out = np.empty(rgb.shape[:-1] + (3,), dtype=np.float32)
    out[..., 0] = 116.0 * fy - 16.0
    out[..., 1] = 500.0 * (fx - fy)
    out[..., 2] = 200.0 * (fy - fz)
    return out


def _lab_to_rgb_np(lab: np.ndarray) -> np.ndarray:
    d = _LAB_DELTA
    fy = (lab[..., 0].astype(np.float64) + 16.0) / 116.0
    fx = fy + lab[..., 1].astype(np.float64) / 500.0
    fz = fy - lab[..., 2].astype(np.float64) / 200.0

    def finv(t):
        return np.where(t > d, t * t * t, 3.0 * d * d * (t - 4.0 / 29.0))

    X = (_XN * finv(fx)).astype(np.float32)
    Y = (_YN * finv(fy)).astype(np.float32)
    Z = (_ZN * finv(fz)).astype(np.float32)
    M = _XYZ_TO_RGB_MATRIX
    out = np.empty(lab.shape[:-1] + (3,), dtype=np.float32)
    for c in range(3):
        np.clip(M[c, 0] * X + M[c, 1] * Y + M[c, 2] * Z, 0.0, 1.0, out=out[..., c])
    return out


def _lab_np_chunked(fn, arr: np.ndarray) -> np.ndarray:
    """fn on row chunks (bounded float64 temporaries on big images)."""
    if arr.ndim != 3:
        return fn(arr)
    out = np.empty(arr.shape[:2] + (3,), dtype=np.float32)
    step = _lab_chunk_rows(arr)
    for y0 in range(0, arr.shape[0], step):
        out[y0:y0 + step] = fn(arr[y0:y0 + step])
    return out


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to CIE L*a*b*. RGB is taken as linear (sRGB primaries,
    D65 white, no transfer curve), in both the numba path and the numpy
    fallback, which match to float32 rounding.

    Args:
        rgb: RGB image (H, W, 3) float32 in [0, 1]

    Returns:
        Lab image (H, W, 3) where L is [0, 100], a/b are roughly [-128, 127]
    """
    if _HAVE_NUMBA:
        rgb32 = np.ascontiguousarray(rgb, dtype=np.float32)
        step = _lab_chunk_rows(rgb32)
        if step >= rgb32.shape[0]:
            return xyz_to_lab_numba(rgb_to_xyz_numba(rgb32))
        lab = np.empty(rgb32.shape[:2] + (3,), dtype=np.float32)
        for y0 in range(0, rgb32.shape[0], step):
            lab[y0:y0 + step] = xyz_to_lab_numba(rgb_to_xyz_numba(rgb32[y0:y0 + step]))
        return lab
    
    # numpy fallback: the same arithmetic as the numba kernels
    return _lab_np_chunked(_rgb_to_lab_np, np.asarray(rgb, dtype=np.float32))


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE L*a*b* back to linear RGB (the inverse of rgb_to_lab), clipped
    to [0, 1]. Numba path and numpy fallback match to float32 rounding.

    Args:
        lab: Lab image (H, W, 3)

    Returns:
        RGB image (H, W, 3) float32 in [0, 1]
    """
    if _HAVE_NUMBA:
        lab32 = np.ascontiguousarray(lab, dtype=np.float32)
        step = _lab_chunk_rows(lab32)
        if step >= lab32.shape[0]:
            return np.clip(xyz_to_rgb_numba(lab_to_xyz_numba(lab32)), 0.0, 1.0)
        rgb = np.empty(lab32.shape[:2] + (3,), dtype=np.float32)
        for y0 in range(0, lab32.shape[0], step):
            part = xyz_to_rgb_numba(lab_to_xyz_numba(lab32[y0:y0 + step]))
            np.clip(part, 0.0, 1.0, out=rgb[y0:y0 + step])
        return rgb
    
    # numpy fallback: the same arithmetic as the numba kernels
    return _lab_np_chunked(_lab_to_rgb_np, np.asarray(lab, dtype=np.float32))
