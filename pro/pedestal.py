# pro/pedestal.py
from __future__ import annotations
import numpy as np

try:
    import cv2  # not required, just for mask helpers if you later want it
except Exception:
    cv2 = None


def _as_float01(img: np.ndarray) -> np.ndarray:
    """Return float32 image; compress crazy-high float ranges to ~[0..1]."""
    a = np.asarray(img)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    # If upstream keeps images normalized, this is a noop.
    # If someone slipped a >1.0 range, compress gently so 'min-subtract' behaves.
    if a.size:
        mx = float(a.max())
        if mx > 5.0:
            a = a / mx
    return a


def _remove_pedestal_array(a: np.ndarray) -> np.ndarray:
    """
    Subtract per-channel minimum. Preserves shape (2D or 3D).
    Result is clipped to [0,1] for safety.
    """
    a = _as_float01(a)

    if a.ndim == 2:
        mn = float(a.min()) if a.size else 0.0
        out = a - mn
    else:
        # H, W, C
        out = np.empty_like(a, dtype=np.float32)
        for c in range(a.shape[2]):
            ch = a[..., c]
            mn = float(ch.min()) if ch.size else 0.0
            out[..., c] = ch - mn

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def remove_pedestal(main, target_doc=None):
    """
    Headless: subtract per-channel minimum on the target document (or active doc).
    Creates a single undo step via doc.apply_edit, like the rest of Pro.
    """
    # Resolve document
    doc = target_doc
    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        # Quiet exit if nothing to do (no popups by design).
        return

    try:
        src = np.asarray(doc.image)
        out = _remove_pedestal_array(src)

        meta = {
            "step_name": "Pedestal Removal",
            "bit_depth": "32-bit floating point",
            "is_mono": (out.ndim == 2),
        }
        doc.apply_edit(out, metadata=meta, step_name="Pedestal Removal")
        if hasattr(main, "_log"):
            main._log("Pedestal Removal")
    except Exception as e:
        # Keep it lightweight; surface as a message box if you prefer
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(main, "Pedestal Removal", f"Failed:\n{e}")
        except Exception:
            pass
