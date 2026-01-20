# pro/layers.py
from __future__ import annotations
from dataclasses import dataclass, field
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
class LayerTransform:
    tx: float = 0.0
    ty: float = 0.0
    rot_deg: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    pivot_x: float | None = None
    pivot_y: float | None = None

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
    transform: LayerTransform = field(default_factory=LayerTransform)
    selected: bool = False  # optional, or store selected layer elsewhere
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

def _affine_matrix_from_transform(t: LayerTransform, H: int, W: int) -> np.ndarray:
    """
    Returns 3x3 matrix mapping source -> destination in pixel coords.
    We will invert it when sampling (dest -> src).
    """
    tx, ty = float(t.tx), float(t.ty)
    sx, sy = float(t.sx), float(t.sy)
    th = np.deg2rad(float(t.rot_deg))

    # Default pivot: image center in pixel coords
    px = (W - 1) * 0.5 if t.pivot_x is None else float(t.pivot_x)
    py = (H - 1) * 0.5 if t.pivot_y is None else float(t.pivot_y)

    c = float(np.cos(th))
    s = float(np.sin(th))

    # Build: T(tx,ty) * T(pivot) * R * S * T(-pivot)
    T1 = np.array([[1, 0, -px],
                   [0, 1, -py],
                   [0, 0,  1]], dtype=np.float32)

    S  = np.array([[sx, 0,  0],
                   [0, sy,  0],
                   [0,  0,  1]], dtype=np.float32)

    R  = np.array([[ c, -s, 0],
                   [ s,  c, 0],
                   [ 0,  0, 1]], dtype=np.float32)

    T2 = np.array([[1, 0, px],
                   [0, 1, py],
                   [0, 0,  1]], dtype=np.float32)

    Tt = np.array([[1, 0, tx],
                   [0, 1, ty],
                   [0, 0,  1]], dtype=np.float32)

    M = (Tt @ T2 @ R @ S @ T1).astype(np.float32, copy=False)
    return M


def _warp_affine_nearest(src: np.ndarray, out_hw: tuple[int, int], M_src_to_dst: np.ndarray,
                         *, fill: float = 0.0) -> np.ndarray:
    """
    Nearest-neighbor warp without deps.
    src: (Hs,Ws) or (Hs,Ws,C)
    out: (Ht,Wt) or (Ht,Wt,C)
    M_src_to_dst: 3x3 matrix mapping src -> dst.
    We sample by inverting: src = inv(M) * dst.
    """
    Ht, Wt = int(out_hw[0]), int(out_hw[1])
    Hs, Ws = int(src.shape[0]), int(src.shape[1])

    if src.ndim == 2:
        C = None
    else:
        C = int(src.shape[2])

    # Compute inverse mapping
    try:
        Minv = np.linalg.inv(M_src_to_dst).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        # Singular matrix (e.g. sx=0). Return fill.
        if C is None:
            return np.full((Ht, Wt), fill, dtype=src.dtype)
        return np.full((Ht, Wt, C), fill, dtype=src.dtype)

    # Build destination grid
    yy, xx = np.mgrid[0:Ht, 0:Wt]
    ones = np.ones_like(xx, dtype=np.float32)

    # (3, Ht*Wt)
    dst = np.stack([xx.astype(np.float32), yy.astype(np.float32), ones], axis=0).reshape(3, -1)
    src_xyw = (Minv @ dst)

    xs = np.rint(src_xyw[0]).astype(np.int32)
    ys = np.rint(src_xyw[1]).astype(np.int32)

    valid = (xs >= 0) & (xs < Ws) & (ys >= 0) & (ys < Hs)

    if C is None:
        out = np.full((Ht * Wt,), fill, dtype=src.dtype)
        out[valid] = src[ys[valid], xs[valid]]
        return out.reshape(Ht, Wt)

    out = np.full((Ht * Wt, C), fill, dtype=src.dtype)
    out[valid] = src[ys[valid], xs[valid], :]
    return out.reshape(Ht, Wt, C)


def _apply_transform_to_layer_image(img01_3c: np.ndarray, t: LayerTransform, H: int, W: int) -> np.ndarray:
    """
    img01_3c is float32 [0..1], shape (Hs,Ws,3).
    Returns float32 [0..1], shape (H,W,3) transformed into base canvas size.
    """
    M = _affine_matrix_from_transform(t, img01_3c.shape[0], img01_3c.shape[1])

    # IMPORTANT:
    # Your transform parameters (tx,ty,pivot) are in *layer pixel space*.
    # We want to composite in base space (H,W).
    #
    # Since you already resize layers to (H,W) before compositing, easiest is:
    # 1) resize to (H,W)
    # 2) build transform using H,W
    # 3) warp in that same canvas
    #
    # So caller should pass already-resized (H,W,3) and we rebuild M with H,W.
    M = _affine_matrix_from_transform(t, H, W)
    warped = _warp_affine_nearest(img01_3c, (H, W), M, fill=0.0).astype(np.float32, copy=False)
    return np.clip(warped, 0.0, 1.0)

def _apply_transform_to_mask(mask01: np.ndarray, t: LayerTransform, H: int, W: int) -> np.ndarray:
    m = np.clip(mask01.astype(np.float32, copy=False), 0.0, 1.0)
    if m.ndim != 2:
        m = m[..., 0] if (m.ndim == 3 and m.shape[2] == 1) else m
    m = _resize_like(m, (H, W))
    M = _affine_matrix_from_transform(t, H, W)
    warped = _warp_affine_nearest(m, (H, W), M, fill=0.0).astype(np.float32, copy=False)
    return np.clip(warped, 0.0, 1.0)


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
    """
    Composite a base image with a stack of ImageLayer objects.

    Notes:
    - Works in float32 [0..1] internally.
    - Ensures 3-channel output.
    - Applies per-layer transform (translate/rotate/scale about pivot) in canvas space
      AFTER resizing the layer to the base canvas (H,W).
    - Applies the SAME transform to the layer's mask (if present) so alpha lines up.
    """
    if base_img is None:
        return None

    out = _ensure_3c(_float01(base_img))
    H, W = int(out.shape[0]), int(out.shape[1])

    # iterate bottom → top so the top-most layer renders last
    for L in reversed(layers or []):
        if not getattr(L, "visible", True):
            continue

        # ----- fetch source pixels -----
        src = getattr(L, "pixels", None)
        if src is None:
            src_doc = getattr(L, "src_doc", None)
            src = getattr(src_doc, "image", None) if src_doc is not None else None
        if src is None:
            continue

        # ----- normalize to float01 + 3 channels + canvas size -----
        s = _ensure_3c(_float01(src))
        s = _resize_like(s, (H, W))

        # ----- apply layer transform in canvas space -----
        t = getattr(L, "transform", None)
        if t is not None:
            try:
                s = _apply_transform_to_layer_image(s, t, H, W)
            except Exception:
                # If transform fails for any reason, fall back to untransformed
                pass

        # ----- validate blend mode -----
        if getattr(L, "mode", None) not in BLEND_MODES:
            L.mode = "Normal"

        # ----- blend result (still full-strength; opacity/mask applied after) -----
        blended = _apply_mode(out, s, L)

        # ----- compute alpha -----
        try:
            alpha = float(getattr(L, "opacity", 1.0))
        except Exception:
            alpha = 1.0
        if not (0.0 <= alpha <= 1.0):
            alpha = 1.0

        # ----- optional mask (transformed to match the layer) -----
        if getattr(L, "mask_doc", None) is not None:
            m = _mask_from_doc(L.mask_doc, use_luma=bool(getattr(L, "mask_use_luma", False)))
            if m is not None:
                m = _resize_like(m, (H, W))

                # Apply SAME transform to mask so it stays registered with the layer pixels
                if t is not None:
                    try:
                        m = _apply_transform_to_mask(m, t, H, W)
                    except Exception:
                        pass

                if getattr(L, "mask_invert", False):
                    m = 1.0 - m

                alpha_map = np.clip(alpha * m, 0.0, 1.0)[..., None]
                out = out * (1.0 - alpha_map) + blended * alpha_map
                continue

        # ----- no mask: uniform opacity -----
        out = out * (1.0 - alpha) + blended * alpha

    return out
