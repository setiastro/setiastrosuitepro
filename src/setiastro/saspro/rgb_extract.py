# pro/rgb_extract.py
from __future__ import annotations
import numpy as np

# Shared utilities
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01

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
