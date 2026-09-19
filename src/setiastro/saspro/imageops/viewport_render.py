"""Viewport-sized display rasterization.

Science images stay as float32 in the document. Display code should call
rasterize_for_display() so 8-bit / QImage / QPixmap buffers are only as large
as the on-screen image, never a second full-resolution copy of the source.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def resize_for_display(
    arr: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    src_x: int = 0,
    src_y: int = 0,
    src_w: int | None = None,
    src_h: int | None = None,
) -> np.ndarray:
    """Resize a source crop to ``(out_h, out_w)`` keeping the source dtype/channels."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3):
        raise ValueError(f"Unsupported array ndim={a.ndim}")

    H, W = int(a.shape[0]), int(a.shape[1])
    x0 = max(0, int(src_x))
    y0 = max(0, int(src_y))
    cw = int(W - x0 if src_w is None else src_w)
    ch = int(H - y0 if src_h is None else src_h)
    cw = max(1, min(cw, W - x0))
    ch = max(1, min(ch, H - y0))
    crop = a[y0:y0 + ch, x0:x0 + cw]

    ow = max(1, int(out_w))
    oh = max(1, int(out_h))
    if crop.shape[0] != oh or crop.shape[1] != ow:
        crop = _resize(crop, ow, oh)
    return crop


def rasterize_for_display(
    arr: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    src_x: int = 0,
    src_y: int = 0,
    src_w: int | None = None,
    src_h: int | None = None,
) -> np.ndarray:
    """Resize a source crop to ``(out_h, out_w)`` uint8 RGB.

    Peak extra allocation is proportional to the *output* size (plus a view
    of the crop), not a full-resolution 8-bit copy of ``arr``.
    """
    crop = resize_for_display(
        arr, out_w, out_h, src_x=src_x, src_y=src_y, src_w=src_w, src_h=src_h
    )
    return _to_uint8_rgb(crop)


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Public wrapper used by the view after optional display stretch."""
    return _to_uint8_rgb(img)


def _resize(crop: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    src_h, src_w = crop.shape[:2]
    if cv2 is not None:
        interp = cv2.INTER_AREA if (out_w < src_w or out_h < src_h) else cv2.INTER_NEAREST
        return cv2.resize(crop, (out_w, out_h), interpolation=interp)

    # Fallback: sample with nearest-neighbour (no full-res 8-bit buffer).
    ys = (np.linspace(0, src_h - 1, out_h)).astype(np.int32)
    xs = (np.linspace(0, src_w - 1, out_w)).astype(np.int32)
    return crop[ys][:, xs]


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        buf8 = a
    elif a.dtype == np.uint16:
        buf8 = (a.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    else:
        buf8 = (np.clip(a.astype(np.float32, copy=False), 0.0, 1.0) * 255.0).astype(np.uint8)

    if buf8.ndim == 2:
        buf8 = np.stack([buf8, buf8, buf8], axis=-1)
    elif buf8.ndim == 3:
        c = buf8.shape[2]
        if c == 1:
            buf8 = np.repeat(buf8, 3, axis=2)
        elif c > 3:
            buf8 = buf8[..., :3]
    else:
        buf8 = np.stack([np.squeeze(buf8)] * 3, axis=-1)

    return np.ascontiguousarray(buf8, dtype=np.uint8)
