# pro/masks_core.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

@dataclass
class MaskLayer:
    id: str
    name: str
    data: np.ndarray          # HxW float32 in [0..1]
    invert: bool = False
    opacity: float = 1.0      # 0..1
    mode: str = "affect"      # "affect" or "protect"
    visible: bool = True

def _resize_mask(mask: np.ndarray, shape_hw: tuple[int,int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape[:2] == (h, w):
        return mask.astype(np.float32, copy=False)
    if cv2 is not None:
        return cv2.resize(mask.astype(np.float32, copy=False), (w, h), interpolation=cv2.INTER_LINEAR)
    # Pure-numpy fallback (nearest)
    y = (np.linspace(0, mask.shape[0]-1, h)).astype(np.int32)
    x = (np.linspace(0, mask.shape[1]-1, w)).astype(np.int32)
    return mask[y][:, x].astype(np.float32, copy=False)

def blend_with_mask(original: np.ndarray,
                    edited: np.ndarray,
                    layer: MaskLayer | None) -> np.ndarray:
    if layer is None:
        return edited
    m = _resize_mask(layer.data, original.shape[:2])
    m = np.clip(m, 0.0, 1.0)
    if layer.mode == "protect":
        m = 1.0 - m
    if layer.invert:
        m = 1.0 - m
    m = m * float(max(0.0, min(1.0, layer.opacity)))

    # Shape/broadcast safety
    o = original
    e = edited
    if e.ndim == 2 and o.ndim == 3:
        e = np.repeat(e[..., None], o.shape[2], axis=2)
    if o.ndim == 2 and e.ndim == 3:
        o = np.repeat(o[..., None], e.shape[2], axis=2)
    if m.ndim == 2 and e.ndim == 3:
        m = m[..., None]

    return (e.astype(np.float32, copy=False) * m +
            o.astype(np.float32, copy=False) * (1.0 - m)).astype(e.dtype, copy=False)
