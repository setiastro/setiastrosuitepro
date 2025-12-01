# pro/widgets/image_utils.py
"""
Centralized image conversion utilities for Seti Astro Suite Pro.

Provides common numpy <-> QImage conversion functions.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def numpy_to_qimage(arr: np.ndarray, normalize: bool = True) -> QImage:
    """
    Convert a numpy array to QImage.
    
    Args:
        arr: Image array. Can be:
            - 2D grayscale (H, W)
            - 3D grayscale (H, W, 1)  
            - 3D RGB (H, W, 3)
            - float32 [0, 1] or uint8 [0, 255]
        normalize: If True, clip and scale float arrays to [0, 1]
        
    Returns:
        QImage in appropriate format
    """
    if arr is None:
        raise ValueError("Input array is None")
    
    arr = np.ascontiguousarray(arr)
    
    # Handle float vs uint8
    if arr.dtype in (np.float32, np.float64):
        if normalize:
            arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        # Convert other integer types
        arr = arr.astype(np.uint8)
    
    # Ensure contiguous
    arr = np.ascontiguousarray(arr)
    
    # Handle dimensions
    if arr.ndim == 2:
        # Grayscale
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    
    elif arr.ndim == 3:
        h, w, c = arr.shape
        
        if c == 1:
            # Grayscale with channel dim
            arr = arr.squeeze()
            return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        
        elif c == 3:
            # RGB
            bytes_per_line = 3 * w
            return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        
        elif c == 4:
            # RGBA
            bytes_per_line = 4 * w
            return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()
        
        else:
            raise ValueError(f"Unsupported number of channels: {c}")
    
    else:
        raise ValueError(f"Unsupported array dimensions: {arr.ndim}")


def numpy_to_qpixmap(arr: np.ndarray, normalize: bool = True) -> QPixmap:
    """
    Convert a numpy array to QPixmap.
    
    Args:
        arr: Image array (see numpy_to_qimage for formats)
        normalize: If True, clip and scale float arrays to [0, 1]
        
    Returns:
        QPixmap
    """
    return QPixmap.fromImage(numpy_to_qimage(arr, normalize))


def float_to_qimage_rgb8(arr: np.ndarray) -> QImage:
    """
    Convert float32 [0, 1] array to QImage RGB888 format.
    
    This is a shared implementation replacing duplicates in:
    - pro/pixelmath.py (_float_to_qimage_rgb8)
    - pro/curve_editor_pro.py (_float_to_qimage_rgb8)
    
    Args:
        arr: float32 array in [0, 1], can be:
            - 2D grayscale (H, W) - expanded to RGB
            - 3D with 1 channel (H, W, 1) - expanded to RGB
            - 3D RGB (H, W, 3)
            
    Returns:
        QImage in RGB888 format
    """
    f = np.asarray(arr, dtype=np.float32)
    if f.ndim == 2:
        f = np.stack([f, f, f], axis=-1)
    elif f.ndim == 3 and f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)
    
    buf8 = (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf8 = np.ascontiguousarray(buf8)
    h, w, _ = buf8.shape
    img = QImage(buf8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    # Keep reference so bytes stay alive
    img._buf = buf8
    return img


def qimage_to_numpy(qimg: QImage) -> np.ndarray:
    """
    Convert a QImage to numpy array.
    
    Args:
        qimg: QImage to convert
        
    Returns:
        numpy array in RGB format (H, W, 3) or grayscale (H, W)
    """
    # Convert to a standard format
    fmt = qimg.format()
    
    if fmt == QImage.Format.Format_Grayscale8:
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w))
        return arr.copy()
    
    # Convert to RGB888 for other formats
    if fmt != QImage.Format.Format_RGB888:
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 3)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
    return arr.copy()


def create_preview_image(arr: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Create a preview-sized version of an image.
    
    Args:
        arr: Full resolution image array
        max_size: Maximum dimension for preview
        
    Returns:
        Downscaled image if needed, original otherwise
    """
    if arr is None:
        return arr
    
    h = arr.shape[0]
    w = arr.shape[1]
    
    if max(h, w) <= max_size:
        return arr
    
    # Calculate scale factor
    scale = max_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Use simple slicing for speed (subsampling)
    step_h = max(1, h // new_h)
    step_w = max(1, w // new_w)
    
    if arr.ndim == 2:
        return arr[::step_h, ::step_w]
    else:
        return arr[::step_h, ::step_w, :]


def normalize_image(arr: np.ndarray, target_max: float = 1.0) -> np.ndarray:
    """
    Normalize image to [0, target_max] range.
    
    Args:
        arr: Image array
        target_max: Maximum value after normalization
        
    Returns:
        Normalized float32 array
    """
    arr = np.asarray(arr, dtype=np.float32)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin) * target_max
    else:
        arr = np.zeros_like(arr)
    
    return arr


# ---------------------------------------------------------------------------
# Shared float normalization (replaces 10+ duplicate implementations)
# ---------------------------------------------------------------------------

def to_float01(arr: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 in [0, 1] range.
    
    Handles:
    - uint8: divides by 255
    - uint16: divides by 65535
    - float: clips to [0, 1]
    - Already float32 in [0, 1]: returns as-is
    
    Args:
        arr: Input image array
        
    Returns:
        float32 array normalized to [0, 1]
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32, copy=False)
        if arr.max() > 1.0 or arr.min() < 0.0:
            return np.clip(arr, 0.0, 1.0)
        return arr


def to_float01_strict(arr: np.ndarray) -> np.ndarray:
    """
    Strictly convert image to float32 in [0, 1] with explicit scaling.
    
    Always clips output to ensure [0, 1] range.
    
    Args:
        arr: Input image array
        
    Returns:
        float32 array strictly in [0, 1]
    """
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    else:
        return np.clip(arr.astype(np.float32), 0.0, 1.0)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Ensure image is 3-channel RGB format.
    
    Args:
        img: Input image (grayscale or RGB)
        
    Returns:
        3-channel RGB array (H, W, 3)
    """
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 3:
        return img
    elif img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3]  # Drop alpha
    else:
        raise ValueError(f"Cannot convert shape {img.shape} to RGB")


def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Ensure image is grayscale format.
    
    Args:
        img: Input image (grayscale or RGB)
        
    Returns:
        2D grayscale array (H, W)
    """
    if img is None:
        return None
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    elif img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0]
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        # Luminance weights: 0.2126 R + 0.7152 G + 0.0722 B
        return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(img.dtype)
    else:
        raise ValueError(f"Cannot convert shape {img.shape} to grayscale")


# ---------------------------------------------------------------------------
# Mask extraction helper (replaces 4+ duplicate implementations)
# ---------------------------------------------------------------------------

try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None


# ---------------------------------------------------------------------------
# Resize utilities (replaces 4+ duplicate implementations)
# ---------------------------------------------------------------------------

def nearest_resize_2d(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Resize a 2D array to (H, W) using nearest neighbor interpolation.
    
    This is a shared implementation replacing duplicates in:
    - pro/clahe.py
    - pro/morphology.py
    - pro/pixelmath.py
    - pro/stacking_suite.py
    
    Args:
        m: 2D input array
        H: Target height
        W: Target width
        
    Returns:
        Resized float32 array (H, W)
    """
    m = np.asarray(m, dtype=np.float32)
    if m.shape == (H, W):
        return m
    if _cv2 is not None:
        try:
            return _cv2.resize(m, (W, H), interpolation=_cv2.INTER_NEAREST)
        except Exception:
            pass
    # Fallback without cv2
    yi = np.linspace(0, m.shape[0] - 1, H).astype(np.int32)
    xi = np.linspace(0, m.shape[1] - 1, W).astype(np.int32)
    return m[yi][:, xi].astype(np.float32, copy=False)


def extract_mask_resized(doc, H: int, W: int) -> np.ndarray | None:
    """
    Extract active mask from document and resize to (H, W).
    
    This is a shared implementation replacing duplicates in:
    - pro/clahe.py (_get_active_mask_resized)
    - pro/morphology.py (_get_active_mask_resized)
    
    Args:
        doc: Document object with active_mask_id and masks attributes
        H: Target height
        W: Target width
        
    Returns:
        Resized mask (H, W) float32 in [0, 1], or None if not found
    """
    if doc is None:
        return None
    mid = getattr(doc, "active_mask_id", None)
    if not mid:
        return None
    
    masks = getattr(doc, "masks", {}) or {}
    layer = masks.get(mid)
    if layer is None:
        return None
    
    # Extract data from layer (object, dict, or raw ndarray)
    data = None
    for attr in ("data", "mask", "image", "array"):
        if hasattr(layer, attr):
            val = getattr(layer, attr)
            if val is not None:
                data = val
                break
    if data is None and isinstance(layer, dict):
        for key in ("data", "mask", "image", "array"):
            if key in layer and layer[key] is not None:
                data = layer[key]
                break
    if data is None and isinstance(layer, np.ndarray):
        data = layer
    if data is None:
        return None
    
    m = np.asarray(data)
    if m.ndim == 3:  # collapse RGB(A) → gray
        m = m.mean(axis=2)
    m = m.astype(np.float32, copy=False)
    
    # Normalize to [0, 1]
    mx = float(m.max()) if m.size else 1.0
    if mx > 1.0:
        m = m / mx
    m = np.clip(m, 0.0, 1.0)
    
    return nearest_resize_2d(m, H, W)


def blend_with_mask(base: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Blend base and out arrays using mask weights.
    
    result = base * (1 - mask) + out * mask
    
    This is a shared implementation replacing duplicates in:
    - pro/morphology.py (_blend_with_mask)
    - pro/ghs_preset.py (_blend_with_mask)
    
    Args:
        base: Original image (2D or 3D)
        out: Processed image (2D or 3D)
        mask: 2D mask in [0, 1] range
        
    Returns:
        Blended image in [0, 1]
    """
    base = np.asarray(base, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)
    mask = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    
    if out.ndim == 3:
        # Ensure base is 3D
        if base.ndim == 2:
            base = base[:, :, None].repeat(out.shape[2], axis=2)
        elif base.ndim == 3 and base.shape[2] == 1:
            base = base.repeat(out.shape[2], axis=2)
        # Expand mask to 3D
        M = mask[:, :, None].repeat(out.shape[2], axis=2)
        return np.clip(base * (1.0 - M) + out * M, 0.0, 1.0)
    
    # 2D output
    if base.ndim == 3 and base.shape[2] == 1:
        base = base.squeeze(axis=2)
    return np.clip(base * (1.0 - mask) + out * mask, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Mask extraction helper (replaces 4+ duplicate implementations)
# ---------------------------------------------------------------------------

def extract_mask_from_document(doc) -> np.ndarray | None:
    """
    Extract active mask (H, W) float32 in [0, 1] from a document.
    
    This is a shared implementation replacing duplicates in:
    - pro/add_stars.py
    - pro/remove_stars.py
    - pro/remove_green.py
    - pro/image_combine.py
    
    Args:
        doc: Document object with active_mask_id and masks attributes
        
    Returns:
        Mask array (H, W) float32 in [0, 1], or None if not found
    """
    try:
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        data = getattr(layer, "data", None) if layer is not None else None
        if data is None:
            return None
        a = np.asarray(data)
        if a.ndim == 3:
            if _cv2 is not None:
                a = _cv2.cvtColor(a, _cv2.COLOR_BGR2GRAY)
            else:
                a = a.mean(axis=2)
        a = a.astype(np.float32, copy=False)
        return np.clip(a, 0.0, 1.0)
    except Exception:
        return None
