# pro/rgb_extract.py
from __future__ import annotations
import numpy as np

def _to_float01(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a)
    if x.dtype.kind in "ui":
        x = x.astype(np.float32)
    elif x.dtype.kind == "f" and x.dtype != np.float32:
        x = x.astype(np.float32)
    if x.size:
        m = float(np.nanmax(x))
        if m > 1.0 and np.isfinite(m):
            x = x / m
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)

def extract_rgb_channels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (R, G, B) as mono float32 2D arrays in [0,1].
    Raises if input is not 3-channel RGB.
    """
    x = _to_float01(img)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("RGB Extract requires a 3-channel RGB image.")
    r = x[..., 0].copy()
    g = x[..., 1].copy()
    b = x[..., 2].copy()
    return r, g, b
