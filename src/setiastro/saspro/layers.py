# pro/layers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

BLEND_MODES = [
    "Normal",
    "Multiply",
    "Screen",
    "Overlay",
    "Soft Light",
    "Hard Light",
    "Color Dodge",
    "Color Burn",
    "Pin Light",
    "Add",
    "Lighten",
    "Darken",
    "Sigmoid",
]

@dataclass
class ImageLayer:
    name: str
    src_doc: Optional[object] = None          # ImageDocument (can be None for baked raster)
    pixels: Optional[np.ndarray] = None       # NEW: baked raster pixels in float32 or any dtype

    visible: bool = True
    opacity: float = 1.0
    mode: str = "Normal"
    mask_doc: Optional[object] = None
    mask_invert: bool = False
    mask_feather: float = 0.0
    mask_use_luma: bool = False

    sigmoid_center: float = 0.5
    sigmoid_strength: float = 10.0

def _float01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind in "ui":
        info = np.iinfo(a.dtype)
        if info.max == 0:
            return a.astype(np.float32)
        return (a.astype(np.float32) / float(info.max))
    return np.clip(a.astype(np.float32), 0.0, 1.0)

def _ensure_3c(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[2] == 1:
        return np.repeat(a, 3, axis=2)
    return a

def _resize_like(src: np.ndarray, tgt_shape_hw: tuple[int, int]) -> np.ndarray:
    """Nearest resize without dependencies. src: (H,W[,C]), target: (H,W)."""
    Ht, Wt = int(tgt_shape_hw[0]), int(tgt_shape_hw[1])
    Hs, Ws = src.shape[0], src.shape[1]
    if (Hs, Ws) == (Ht, Wt):
        return src
    yi = (np.linspace(0, Hs - 1, Ht)).astype(np.int32)
    xi = (np.linspace(0, Ws - 1, Wt)).astype(np.int32)
    return src[yi][:, xi, ...] if src.ndim == 3 else src[yi][:, xi]

def _luminance01(img: np.ndarray) -> np.ndarray:
    a = _float01(img)
    a = _ensure_3c(a)
    # Rec. 709 luma
    y = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return np.clip(y.astype(np.float32, copy=False), 0.0, 1.0)

def _mask_from_doc(doc, *, use_luma: bool = False) -> Optional[np.ndarray]:
    if doc is None:
        return None
    if use_luma:
        img = getattr(doc, "image", None)
        if img is None:
            return None
        return _luminance01(img)

    # existing active-mask path
    masks = getattr(doc, "masks", {}) or {}
    mid   = getattr(doc, "active_mask_id", None)
    layer = masks.get(mid) if mid else None
    data  = getattr(layer, "data", None) if layer is not None else None
    if data is None:
        return None
    m = np.asarray(data)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        return None
    return np.clip(m.astype(np.float32, copy=False), 0.0, 1.0)

# ---- blend ops (src over base) ---------------------------------------
def _apply_mode(base: np.ndarray, src: np.ndarray, layer: ImageLayer) -> np.ndarray:
    """
    base, src: float32, [0..1], same shape.
    Uses layer.mode and optional extra params (e.g. sigmoid_center/strength).
    """
    mode = getattr(layer, "mode", "Normal") or "Normal"

    if mode == "Multiply":
        return base * src

    if mode == "Screen":
        return 1.0 - (1.0 - base) * (1.0 - src)

    if mode == "Overlay":
        return np.where(
            base <= 0.5,
            2.0 * base * src,
            1.0 - 2.0 * (1.0 - base) * (1.0 - src),
        )

    if mode == "Soft Light":
        # SVG / W3C-style soft light
        return (1.0 - 2.0 * src) * (base * base) + 2.0 * src * base

    if mode == "Hard Light":
        # Overlay, but conditioned on src
        return np.where(
            src <= 0.5,
            2.0 * base * src,
            1.0 - 2.0 * (1.0 - base) * (1.0 - src),
        )

    if mode == "Color Dodge":
        eps = 1e-6
        denom = np.maximum(1.0 - src, eps)
        out = base / denom
        return np.clip(out, 0.0, 1.0)

    if mode == "Color Burn":
        eps = 1e-6
        denom = np.maximum(src, eps)
        out = 1.0 - (1.0 - base) / denom
        return np.clip(out, 0.0, 1.0)

    if mode == "Pin Light":
        hi = np.maximum(base, 2.0 * src - 1.0)
        lo = np.minimum(base, 2.0 * src)
        return np.where(src > 0.5, hi, lo)

    if mode == "Add":
        return np.clip(base + src, 0.0, 1.0)

    if mode == "Lighten":
        return np.maximum(base, src)

    if mode == "Darken":
        return np.minimum(base, src)

    if mode == "Sigmoid":
        # Per-layer sigmoid blend:
        # dark base → stay closer to base
        # bright base → move towards src
        luma = _luminance01(base)  # (H, W)

        center = float(getattr(layer, "sigmoid_center", 0.5) or 0.5)
        strength = float(getattr(layer, "sigmoid_strength", 10.0) or 10.0)

        # weight in [0..1]
        w = 1.0 / (1.0 + np.exp(-strength * (luma - center)))
        w = w[..., None]  # broadcast over channels

        return base * (1.0 - w) + src * w

    # Normal
    return src


def composite_stack(base_img: np.ndarray, layers: List[ImageLayer]) -> np.ndarray:
    if base_img is None:
        return None
    out = _float01(base_img)
    out = _ensure_3c(out)

    H, W = out.shape[0], out.shape[1]

    # iterate bottom → top so the top-most layer renders last
    for L in reversed(layers or []):
        if not L.visible:
            continue
        src = getattr(L, "pixels", None)
        if src is None:
            src_doc = getattr(L, "src_doc", None)
            src = getattr(src_doc, "image", None) if src_doc is not None else None
        if src is None:
            continue
        s = _ensure_3c(_float01(src))
        s = _resize_like(s, (H, W))

        if getattr(L, "mode", None) not in BLEND_MODES:
            L.mode = "Normal"

        blended = _apply_mode(out, s, L)

        alpha = float(L.opacity if 0.0 <= L.opacity <= 1.0 else 1.0)
        if L.mask_doc is not None:
            m = _mask_from_doc(L.mask_doc, use_luma=bool(L.mask_use_luma))
            if m is not None:
                m = _resize_like(m, (H, W))
                if L.mask_invert:
                    m = 1.0 - m
                alpha_map = np.clip(alpha * m, 0.0, 1.0)[..., None]
                out = out * (1.0 - alpha_map) + blended * alpha_map
                continue
        out = out * (1.0 - alpha) + blended * alpha

    return out
