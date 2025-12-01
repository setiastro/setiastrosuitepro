# pro/remove_green.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QSlider, QHBoxLayout,
    QPushButton, QMessageBox, QCheckBox, QComboBox
)
try:
    import cv2
except Exception:
    cv2 = None

# ---------- utils (now imported from shared location) ----------
from pro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_from_document as _active_mask_array_from_doc
)

def _ensure_rgb(arr: np.ndarray) -> np.ndarray | None:
    """Return float32 RGB [0..1] or None if impossible."""
    a = _to_float01(arr)
    if a.ndim == 2:
        return None
    if a.ndim == 3 and a.shape[2] == 1:
        return None
    if a.ndim == 3 and a.shape[2] >= 3:
        return a[..., :3].astype(np.float32, copy=False)
    return None

# ---------- SCNR core (with modes + preserve lightness) ----------
_SCNR_MODE_LABELS = {
    "avg": "Average(R,B)",
    "max": "Max(R,B)",
    "min": "Min(R,B)",
}

def _compute_neutral(r: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return np.maximum(r, b)
    elif mode == "min":
        return np.minimum(r, b)
    # default "avg"
    return (r + b) * 0.5

def _apply_scnr_rgb(rgb: np.ndarray, amount: float, mode: str = "avg", preserve_lightness: bool = True) -> np.ndarray:
    """
    SCNR green suppression:
      G' = G - amount * max(0, G - neutral)
    where neutral is avg/max/min of (R,B) depending on mode.

    If preserve_lightness=True:
      compute per-pixel scale s = Y_before / Y_after (Rec.709 luma),
      cap s so that no channel exceeds 1.0, and multiply ALL channels by s.
    """
    rgb = np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # choose neutral comparator (avg/max/min of R,B)
    neutral = _compute_neutral(R, B, mode)
    excess  = np.maximum(0.0, G - neutral)
    G_new   = np.clip(G - float(np.clip(amount, 0.0, 1.0)) * excess, 0.0, 1.0)

    out = np.stack([R, G_new, B], axis=-1)

    if not preserve_lightness:
        return out

    # ---- preserve perceived lightness (scale ALL channels equally) ----
    wR, wG, wB = 0.2126, 0.7152, 0.0722
    Y_before = wR * R + wG * G + wB * B
    Y_after  = wR * out[..., 0] + wG * out[..., 1] + wB * out[..., 2]

    eps   = 1e-8
    scale = Y_before / np.maximum(Y_after, eps)

    # highlight safety: prevent any channel from exceeding 1.0 after scaling
    maxc  = np.max(out, axis=2)  # current per-pixel max channel
    cap   = np.where(maxc > 0.0, 1.0 / np.maximum(maxc, eps), 1.0)
    scale = np.minimum(scale, cap)

    out = np.clip(out * scale[..., None], 0.0, 1.0)
    return out

# ---------- headless core ----------
def remove_green_headless(
    doc,
    amount: float = 1.0,
    mode: str = "avg",                  # "avg" | "max" | "min"
    preserve_lightness: bool = True,
):
    """
    Run SCNR on doc.image (RGB only), blend with active mask if present, push as undoable edit.
    """
    if doc is None or getattr(doc, "image", None) is None:
        return

    src = np.asarray(doc.image)
    rgb = _ensure_rgb(src)
    if rgb is None:
        try:
            doc.apply_edit(src.astype(np.float32, copy=False),
                           metadata={"step_name": "Remove Green (no-op non-RGB)"},
                           step_name="Remove Green")
        except Exception:
            pass
        return

    amt = float(max(0.0, min(1.0, amount)))
    mode = (mode or "avg").lower()
    if mode not in ("avg", "max", "min"):
        mode = "avg"

    processed = _apply_scnr_rgb(rgb, amt, mode=mode, preserve_lightness=preserve_lightness)

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
        "remove_green": {
            "amount": amt,
            "mode": mode,
            "preserve_lightness": bool(preserve_lightness),
            "mode_label": _SCNR_MODE_LABELS.get(mode, "Average(R,B)"),
        },
        "bit_depth": "32-bit floating point",
        "is_mono": (out.ndim == 2),
    }
    doc.apply_edit(out.astype(np.float32, copy=False), metadata=meta, step_name="Remove Green")

# ---------- dialog ----------
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

        # amount
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(100)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.value_label = QLabel("Amount: 1.00")
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(f"Amount: {v/100.0:.2f}"))
        lay.addWidget(self.slider)
        lay.addWidget(self.value_label)

        # mode dropdown
        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Neutral mode:"))
        self.mode_box = QComboBox()
        # order: avg (default), max, min
        self.mode_box.addItem(_SCNR_MODE_LABELS["avg"], userData="avg")
        self.mode_box.addItem(_SCNR_MODE_LABELS["max"], userData="max")
        self.mode_box.addItem(_SCNR_MODE_LABELS["min"], userData="min")
        self.mode_box.setCurrentIndex(0)
        row_mode.addWidget(self.mode_box)
        row_mode.addStretch(1)
        lay.addLayout(row_mode)

        # preserve lightness
        self.cb_preserve = QCheckBox("Preserve lightness")
        self.cb_preserve.setChecked(True)
        lay.addWidget(self.cb_preserve)

        # buttons
        row = QHBoxLayout()
        btn_apply = QPushButton("Apply"); btn_apply.clicked.connect(self._apply)
        btn_cancel= QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(btn_apply); row.addWidget(btn_cancel)
        lay.addLayout(row)

        self.resize(460, 200)

    def set_amount(self, amt: float):
        try:
            self.slider.setValue(int(round(max(0.0, min(1.0, float(amt))) * 100)))
        except Exception:
            pass

    def set_mode(self, mode: str | None):
        m = (mode or "avg").lower()
        idx = {"avg":0, "max":1, "min":2}.get(m, 0)
        try:
            self.mode_box.setCurrentIndex(idx)
        except Exception:
            pass

    def set_preserve_lightness(self, preserve: bool | None):
        try:
            self.cb_preserve.setChecked(True if preserve is None else bool(preserve))
        except Exception:
            pass

    def _apply(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            QMessageBox.warning(self, "Remove Green", "No image.")
            return

        amount   = self.slider.value() / 100.0
        mode     = self.mode_box.currentData() or "avg"
        preserve = self.cb_preserve.isChecked()

        # Build a preset dict so headless + replay use the same schema
        preset = {
            "amount": float(amount),
            "mode":   str(mode),
            "preserve_lightness": bool(preserve),
        }

        # Apply to this doc
        remove_green_headless(self.doc, amount=amount, mode=mode, preserve_lightness=preserve)

        # Log + record for Replay Last Action (if main supports it)
        if hasattr(self.main, "_log"):
            self.main._log(
                f"Remove Green: amount={amount:.2f}, mode={mode}, "
                f"preserve_lightness={preserve}"
            )

        try:
            # stash last headless-style command
            self.main._last_headless_command = {
                "command_id": "remove_green",
                "preset": dict(preset),
            }
            if hasattr(self.main, "_log"):
                self.main._log(
                    f"[Replay] Recorded Remove Green preset "
                    f"(amount={amount:.2f}, mode={mode}, "
                    f"preserve_lightness={preserve})"
                )
        except Exception:
            # Never let replay bookkeeping kill the dialog
            pass

        self.accept()



# ---------- entry points used by main ----------
def open_remove_green_dialog(main, doc=None, preset: dict | None = None):
    """
    Open the Remove Green dialog for a specific document.

    If doc is None, we fall back to main._active_doc for legacy callers.
    """
    from PyQt6.QtWidgets import QMessageBox

    if doc is None:
        doc = getattr(main, "_active_doc", None)
        if callable(doc):
            doc = doc()

    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(main, "Remove Green", "Open an image first.")
        return

    dlg = RemoveGreenDialog(main, doc, parent=main)

    if preset:
        amt = preset.get("amount", preset.get("strength", preset.get("value", None)))
        if amt is not None:
            dlg.set_amount(float(amt))

        mode = preset.get("mode", preset.get("neutral_mode"))
        if mode is not None:
            dlg.set_mode(str(mode))

        preserve = preset.get("preserve_lightness", preset.get("preserve", True))
        dlg.set_preserve_lightness(bool(preserve))

    dlg.show()


def apply_remove_green_preset_to_doc(main, doc, preset: dict):
    amt = float(preset.get("amount", preset.get("strength", preset.get("value", 1.0))))
    mode = str(preset.get("mode", preset.get("neutral_mode", "avg"))).lower()
    preserve = bool(preset.get("preserve_lightness", preset.get("preserve", True)))
    remove_green_headless(doc, amount=amt, mode=mode, preserve_lightness=preserve)
    if hasattr(main, "_log"):
        name = doc.display_name() if hasattr(doc, "display_name") else "Image"
        main._log(
            f"Remove Green (headless) on '{name}'; amount={amt:.2f}, mode={mode}, preserve_lightness={preserve}"
        )
