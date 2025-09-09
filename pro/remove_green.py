# pro/remove_green.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QSlider, QHBoxLayout, QPushButton, QMessageBox
try:
    import cv2
except Exception:
    cv2 = None

# use your SCNR implementation
from imageops.scnr import apply_average_neutral_scnr


# ---------- utils ----------
def _to_float01(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype.kind in "ui":
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32, copy=False)
        if arr.size:
            mx = float(arr.max())
            if mx > 5.0:
                arr = arr / mx
    return np.clip(arr, 0.0, 1.0)

def _ensure_rgb(arr: np.ndarray) -> np.ndarray | None:
    """Return float32 RGB [0..1] or None if impossible."""
    a = _to_float01(arr)
    if a.ndim == 2:
        # grayscale → SCNR is a no-op conceptually; return None so we skip
        return None
    if a.ndim == 3 and a.shape[2] == 1:
        return None
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., :3].astype(np.float32, copy=False)
    return None

def _active_mask_array_from_doc(doc) -> np.ndarray | None:
    try:
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        data = getattr(layer, "data", None) if layer is not None else None
        if data is None:
            return None
        m = np.asarray(data)
        if m.ndim == 3:
            if cv2 is not None:
                m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            else:
                m = m.mean(axis=2)
        m = m.astype(np.float32, copy=False)
        return np.clip(m, 0.0, 1.0)
    except Exception:
        return None


# ---------- headless core ----------
def remove_green_headless(doc, amount: float = 1.0):
    """
    Run your SCNR on doc.image (RGB only), blend with active mask if present,
    push as an undoable edit.
    """
    if doc is None or getattr(doc, "image", None) is None:
        return

    src = np.asarray(doc.image)
    rgb = _ensure_rgb(src)
    if rgb is None:
        # Not RGB → nothing to do; just inform/log silently
        try:
            doc.apply_edit(src.astype(np.float32, copy=False), metadata={"step_name": "Remove Green (no-op non-RGB)"}, step_name="Remove Green")
        except Exception:
            pass
        return

    amt = float(max(0.0, min(1.0, amount)))
    processed = apply_average_neutral_scnr(rgb, amount=amt)  # uses your function

    # put processed back into original shape if source had >=3 channels
    if src.ndim == 3 and src.shape[2] > 3:
        out = src.astype(np.float32, copy=False).copy()
        out[..., :3] = processed
    else:
        out = processed

    # mask-aware blend (mask from destination doc)
    m = _active_mask_array_from_doc(doc)
    if m is not None:
        h, w = out.shape[:2]
        if m.shape != (h, w):
            if cv2 is not None:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                yi = (np.linspace(0, m.shape[0]-1, h)).astype(np.int32)
                xi = (np.linspace(0, m.shape[1]-1, w)).astype(np.int32)
                m = m[yi][:, xi]
        if out.ndim == 3:
            m = np.repeat(m[:, :, None], out.shape[2], axis=2)
        src_f = _to_float01(src)
        out = np.clip(src_f * (1.0 - m) + out * m, 0.0, 1.0)

    meta = {
        "step_name": "Remove Green",
        "remove_green": {"amount": amt},
        "bit_depth": "32-bit floating point",
        "is_mono": (out.ndim == 2),
    }
    doc.apply_edit(out.astype(np.float32, copy=False), metadata=meta, step_name="Remove Green")


# ---------- dialog (click / background drop) ----------
class RemoveGreenDialog(QDialog):
    def __init__(self, main, doc, parent=None):
        super().__init__(parent)
        self.main = main
        self.doc = doc
        self.setWindowTitle("Remove Green (SCNR)")
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Select the amount to remove green noise:"))

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(100)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.value_label = QLabel("Amount: 1.00")
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(f"Amount: {v/100.0:.2f}"))

        lay.addWidget(self.slider)
        lay.addWidget(self.value_label)

        row = QHBoxLayout()
        btn_apply = QPushButton("Apply"); btn_apply.clicked.connect(self._apply)
        btn_cancel= QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(btn_apply); row.addWidget(btn_cancel)
        lay.addLayout(row)

        self.resize(420, 140)

    def set_amount(self, amt: float):
        try:
            self.slider.setValue(int(round(max(0.0, min(1.0, float(amt))) * 100)))
        except Exception:
            pass

    def _apply(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            QMessageBox.warning(self, "Remove Green", "No image.")
            return
        amount = self.slider.value() / 100.0
        remove_green_headless(self.doc, amount)
        if hasattr(self.main, "_log"):
            self.main._log(f"Remove Green: amount={amount:.2f}")
        self.accept()


# ---------- entry points used by main ----------
def open_remove_green_dialog(main, preset: dict | None = None):
    doc = getattr(main, "_active_doc", None)
    if callable(doc): doc = doc()
    if doc is None or getattr(doc, "image", None) is None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(main, "Remove Green", "Open an image first.")
        return
    dlg = RemoveGreenDialog(main, doc, parent=main)
    if preset:
        amt = preset.get("amount", preset.get("strength", preset.get("value", None)))
        if amt is not None:
            dlg.set_amount(float(amt))
    dlg.show()

def apply_remove_green_preset_to_doc(main, doc, preset: dict):
    amt = float(preset.get("amount", preset.get("strength", preset.get("value", 1.0))))
    remove_green_headless(doc, amt)
    if hasattr(main, "_log"):
        name = doc.display_name() if hasattr(doc, "display_name") else "Image"
        main._log(f"Remove Green (headless) on '{name}'; amount={amt:.2f}")
