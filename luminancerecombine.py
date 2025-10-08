# pro/luminancerecombine.py
from __future__ import annotations
import numpy as np
import cv2

from skimage.color import rgb2lab, lab2rgb

# destination-mask helper (optional but nice to have for masked blends)
from pro.add_stars import _active_mask_array_from_doc

# Rec.709 linear luma weights
_LUMA_W = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _to_float01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.float32 and a.dtype != np.float64:
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        elif a.dtype == np.uint16:
            a = a.astype(np.float32) / 65535.0
        else:
            a = a.astype(np.float32)
    # If anything >1, assume linear scale and normalize to max
    m = float(np.nanmax(a))
    if m > 1.0 and np.isfinite(m):
        a = a / m
    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def compute_luminance(img: np.ndarray) -> np.ndarray:
    """Return 2-D luminance in [0..1]. Accepts mono (passes through) or RGB."""
    f = _to_float01(img)
    if f.ndim == 2:
        return f
    if f.ndim == 3 and f.shape[2] == 1:
        return f[..., 0]
    if f.ndim == 3 and f.shape[2] == 3:
        return f[..., 0]*_LUMA_W[0] + f[..., 1]*_LUMA_W[1] + f[..., 2]*_LUMA_W[2]
    raise ValueError("compute_luminance: expected mono or RGB image.")


def recombine_luminance_rgb(target_rgb: np.ndarray, new_L: np.ndarray) -> np.ndarray:
    """
    Replace target L* with new L (in [0..1]) using CIE Lab,
    then convert back to RGB, clip to [0..1].
    """
    rgb = _to_float01(target_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Recombine Luminance requires an RGB target image.")

    H, W, _ = rgb.shape
    if new_L.shape[:2] != (H, W):
        new_L = cv2.resize(new_L.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # Lab expects L* in [0..100]
    lab = rgb2lab(rgb)
    lab[..., 0] = np.clip(new_L, 0.0, 1.0) * 100.0
    out = lab2rgb(lab).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def apply_recombine_to_doc(target_doc, luminance_source_img: np.ndarray):
    """
    Overwrite target_doc.image by recombining with luminance_source_img (RGB or mono).
    Honors destination mask if present on the doc.
    """
    base = _to_float01(np.asarray(target_doc.image))
    newL = compute_luminance(luminance_source_img)

    replaced = recombine_luminance_rgb(base, newL)

    # destination-mask blend if active
    m = _active_mask_array_from_doc(target_doc)
    if m is not None:
        if replaced.ndim == 3:
            m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32)
        else:
            m3 = m.astype(np.float32)
        replaced = base * (1.0 - m3) + replaced * m3

    target_doc.apply_edit(
        replaced.astype(np.float32, copy=False),
        metadata={"step_name": "Recombine Luminance"},
        step_name="Recombine Luminance",
    )
