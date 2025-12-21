# pro/pedestal.py
from __future__ import annotations
import numpy as np

try:
    import cv2  # not required
    # just for mask helpers if you later want it
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
    Subtract per-channel minimum on the *currently active* content:
    - If a Preview/ROI tab is active, operate on that ROI document.
    - Otherwise, operate on the full base document.
    Creates a single undo step via doc.apply_edit (ROI docs manage their
    own preview undo; base docs go into full undo stack).
    """
    doc = target_doc

    # Prefer DocManager's view-aware resolution
    dm = getattr(main, "doc_manager", None) or getattr(main, "docman", None)

    if doc is None and dm is not None:
        # 1) Try to resolve from the currently active view (ROI-aware)
        vw = None
        try:
            if hasattr(dm, "_active_view_widget") and callable(dm._active_view_widget):
                vw = dm._active_view_widget()
        except Exception:
            vw = None

        if vw is not None and hasattr(dm, "get_document_for_view"):
            try:
                candidate = dm.get_document_for_view(vw)
                if candidate is not None:
                    doc = candidate
            except Exception:
                doc = None

        # 2) If that failed, fall back to whatever DM thinks is active
        if doc is None and hasattr(dm, "get_active_document"):
            try:
                doc = dm.get_active_document()
            except Exception:
                doc = None

    # 3) Last resort: legacy _active_doc on the main window
    if doc is None:
        ad = getattr(main, "_active_doc", None)
        if callable(ad):
            doc = ad()
        else:
            doc = ad

    if doc is None or getattr(doc, "image", None) is None:
        # Quiet exit if nothing to do.
        return

    # Optional debug so you can confirm it hits ROI when Preview is active
    try:
        is_roi = bool(getattr(doc, "_roi", None)) or bool(getattr(doc, "_roi_info", None))
        print(f"[Pedestal] target_doc={doc!r}, type={type(doc)}, is_roi={is_roi}")
    except Exception:
        pass

    # ---- actual operation ----
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
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(main, "Pedestal Removal", f"Failed:\n{e}")
        except Exception:
            pass
