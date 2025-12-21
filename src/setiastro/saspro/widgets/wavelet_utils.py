# pro/widgets/wavelet_utils.py
"""
Shared wavelet utilities for à-trous decomposition and reconstruction.

This module provides centralized implementations for wavelet operations
used across wavescale_hdr.py, wavescalede.py, and other modules.
"""
from __future__ import annotations
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
        if axis == 1:  # x
            return _nd_convolve(image2d, k1d.reshape(1, -1), mode="reflect")
        else:          # y
            return _nd_convolve(image2d, k1d.reshape(-1, 1), mode="reflect")
    else:
        # Fallback numpy implementation
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
        return _nd_gauss(image2d, sigma=sigma, mode="reflect")
    else:
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
        smooth = conv_sep_reflect(tmp, k, axis=0)
        planes.append(current - smooth)  # detail = current - smoothed
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


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB image to CIE L*a*b* color space.
    
    Uses Numba-accelerated version if available.
    
    Args:
        rgb: RGB image (H, W, 3) float32 in [0, 1]
        
    Returns:
        Lab image (H, W, 3) where L is [0, 100], a/b are roughly [-128, 127]
    """
    if _HAVE_NUMBA:
        rgb32 = np.ascontiguousarray(rgb.astype(np.float32))
        xyz = rgb_to_xyz_numba(rgb32)
        lab = xyz_to_lab_numba(xyz)
        return lab
    
    # Numpy fallback
    rgb = np.asarray(rgb, dtype=np.float32)
    
    # sRGB gamma linearization
    linear = np.where(rgb > 0.04045, 
                      np.power((rgb + 0.055) / 1.055, 2.4),
                      rgb / 12.92)
    
    # RGB -> XYZ
    xyz = np.einsum('ij,...j->...i', _RGB_TO_XYZ_MATRIX, linear)
    
    # XYZ -> Lab
    xyz_n = xyz / _D65_WHITE
    
    def f(t):
        return np.where(t > 0.008856, 
                       np.power(t, 1/3), 
                       7.787 * t + 16/116)
    
    fx, fy, fz = f(xyz_n[..., 0]), f(xyz_n[..., 1]), f(xyz_n[..., 2])
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIE L*a*b* image to sRGB color space.
    
    Uses Numba-accelerated version if available.
    
    Args:
        lab: Lab image (H, W, 3)
        
    Returns:
        RGB image (H, W, 3) float32 in [0, 1]
    """
    if _HAVE_NUMBA:
        lab32 = np.ascontiguousarray(lab.astype(np.float32))
        xyz = lab_to_xyz_numba(lab32)
        rgb = xyz_to_rgb_numba(xyz)
        return np.clip(rgb, 0.0, 1.0)
    
    # Numpy fallback
    lab = np.asarray(lab, dtype=np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    
    # Lab -> XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inv(t):
        return np.where(t > 0.206893,
                       np.power(t, 3),
                       (t - 16/116) / 7.787)
    
    xyz = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1) * _D65_WHITE
    
    # XYZ -> linear RGB
    linear = np.einsum('ij,...j->...i', _XYZ_TO_RGB_MATRIX, xyz)
    
    # sRGB gamma correction
    rgb = np.where(linear > 0.0031308,
                   1.055 * np.power(np.maximum(linear, 0), 1/2.4) - 0.055,
                   12.92 * linear)
    
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)
