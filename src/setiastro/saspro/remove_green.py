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
from setiastro.saspro.widgets.image_utils import (
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
    "avg": "Average of other two",
    "max": "Max of other two",
    "min": "Min of other two",
}
def _compute_neutral(r: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return np.maximum(r, b)
    elif mode == "min":
        return np.minimum(r, b)
    # default "avg"
    return (r + b) * 0.5

_CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}
_CHANNEL_LABELS = {
    "R": "Red",   "G": "Green",   "B": "Blue",
    "C": "Cyan",  "M": "Magenta", "Y": "Yellow",
}
# CMY channels are just RGB SCNR applied in inverted space:
#   Cyan    = inverted Red
#   Magenta = inverted Green
#   Yellow  = inverted Blue
_CMY_TO_RGB = {"C": "R", "M": "G", "Y": "B"}
_VALID_CHANNELS = ("R", "G", "B", "C", "M", "Y")

def _apply_scnr_rgb(rgb: np.ndarray, amount: float, mode: str = "avg",
                    preserve_lightness: bool = True, channel: str = "G") -> np.ndarray:
    """
    SCNR channel suppression (default target: Green):
      C' = C - amount * max(0, C - neutral)
    where C is the target channel and neutral is avg/max/min of the OTHER
    two channels, per `mode`.

    If preserve_lightness=True:
      compute per-pixel scale s = Y_before / Y_after (Rec.709 luma),
      cap s so that no channel exceeds 1.0, and multiply ALL channels by s.
    """
    rgb = np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    ch = (channel or "G").upper()
    if ch not in _VALID_CHANNELS:
        ch = "G"

    # For CMY targets: invert → suppress complementary RGB channel → invert back
    inverted = ch in _CMY_TO_RGB
    if inverted:
        rgb_work = 1.0 - rgb
        eff_ch = _CMY_TO_RGB[ch]
    else:
        rgb_work = rgb
        eff_ch = ch

    ci = _CHANNEL_INDEX[eff_ch]
    others = [i for i in (0, 1, 2) if i != ci]
    A  = rgb_work[..., others[0]]
    Bc = rgb_work[..., others[1]]
    C  = rgb_work[..., ci]

    # neutral comparator from the other two channels
    neutral = _compute_neutral(A, Bc, mode)
    excess  = np.maximum(0.0, C - neutral)
    C_new   = np.clip(C - float(np.clip(amount, 0.0, 1.0)) * excess, 0.0, 1.0)

    out = rgb_work.astype(np.float32, copy=True)
    out[..., ci] = C_new

    if inverted:
        out = np.clip(1.0 - out, 0.0, 1.0)

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
    mode: str = "avg",
    preserve_lightness: bool = True,
    channel: str = "G",
):
    _dbg = np.asarray(getattr(doc, "image", None))
    print(f"[RG headless] shape={getattr(_dbg, 'shape', None)} "
          f"amount={amount} mode={mode} channel={channel} preserve={preserve_lightness}")
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
    ch = (channel or "G").upper()
    if ch not in _VALID_CHANNELS:
        ch = "G"

    processed = _apply_scnr_rgb(rgb, amt, mode=mode,
                                preserve_lightness=preserve_lightness,
                                channel=ch)

    # put processed back into original shape if source had >=3 channels
    if src.ndim == 3 and src.shape[2] > 3:
        out = src.astype(np.float32, copy=True)
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

    step_label = f"SCNR (Remove {_CHANNEL_LABELS[ch]})"
    meta = {
        "step_name": step_label,
        "remove_green": {
            "amount": amt,
            "mode": mode,
            "preserve_lightness": bool(preserve_lightness),
            "mode_label": _SCNR_MODE_LABELS.get(mode, "Average"),
            "channel": ch,
            "channel_label": _CHANNEL_LABELS[ch],
        },
        "bit_depth": "32-bit floating point",
        "is_mono": (out.ndim == 2),
    }
    doc.apply_edit(out.astype(np.float32, copy=False), metadata=meta, step_name=step_label)

# ---------- dialog ----------
class RemoveGreenDialog(QDialog):
    def __init__(self, main, doc, parent=None):
        super().__init__(parent)
        self.main = main
        self.doc = doc
        self.setWindowTitle(self.tr("Remove Green (SCNR)"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)

        # target channel
        row_ch = QHBoxLayout()
        row_ch.addWidget(QLabel(self.tr("Target channel:")))
        self.channel_box = QComboBox()
        self.channel_box.addItem(self.tr("Green"),   userData="G")
        self.channel_box.addItem(self.tr("Red"),     userData="R")
        self.channel_box.addItem(self.tr("Blue"),    userData="B")
        self.channel_box.addItem(self.tr("Magenta"), userData="M")
        self.channel_box.addItem(self.tr("Cyan"),    userData="C")
        self.channel_box.addItem(self.tr("Yellow"),  userData="Y")
        self.channel_box.setCurrentIndex(0)
        row_ch.addWidget(self.channel_box)
        row_ch.addStretch(1)
        lay.addLayout(row_ch)

        lay.addWidget(QLabel(self.tr("Select the amount to suppress:")))

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
        row_mode.addWidget(QLabel(self.tr("Neutral mode:")))
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
        self.cb_preserve = QCheckBox(self.tr("Preserve lightness"))
        self.cb_preserve.setChecked(True)
        lay.addWidget(self.cb_preserve)

        # status label (cleared on any parameter change, set on Apply)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.status_label)

        # clear status whenever the user changes anything
        self.slider.valueChanged.connect(lambda _=None: self.status_label.clear())
        self.mode_box.currentIndexChanged.connect(lambda _=None: self.status_label.clear())
        self.channel_box.currentIndexChanged.connect(lambda _=None: self.status_label.clear())
        self.cb_preserve.toggled.connect(lambda _=None: self.status_label.clear())

        # buttons
        row = QHBoxLayout()
        btn_apply = QPushButton(self.tr("Apply")); btn_apply.clicked.connect(self._apply)
        btn_cancel= QPushButton(self.tr("Cancel")); btn_cancel.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(btn_apply); row.addWidget(btn_cancel)
        lay.addLayout(row)

        # --- preset drag handle (grip) ---
        try:
            from PyQt6.QtGui import QIcon
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import green_path
                _grip_icon = QIcon(green_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "remove_green", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a Remove Green (SCNR) shortcut "
                    "with these exact settings.\n"
                    "Drop directly on an image to apply them headlessly."
                ),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            drag_row.addStretch(1)
            lay.addLayout(drag_row)
        except Exception:
            pass

        self.resize(460, 220)

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

    def set_channel(self, channel: str | None):
        c = (channel or "G").upper()
        idx = {"G": 0, "R": 1, "B": 2, "M": 3, "C": 4, "Y": 5}.get(c, 0)
        try:
            self.channel_box.setCurrentIndex(idx)
        except Exception:
            pass

    def set_preserve_lightness(self, preserve: bool | None):
        try:
            self.cb_preserve.setChecked(True if preserve is None else bool(preserve))
        except Exception:
            pass

    # -------- preset emit (grip) --------
    def get_preset(self) -> dict:
        """Emit current state as the SCNR preset schema (inverse of seed_from_preset).
        Mirrors exactly what _apply records for Replay Last."""
        return {
            "amount": float(self.slider.value() / 100.0),
            "mode": str(self.mode_box.currentData() or "avg"),
            "preserve_lightness": bool(self.cb_preserve.isChecked()),
            "channel": str(self.channel_box.currentData() or "G"),
        }

    # -------- preset seed (double-click open) --------
    def seed_from_preset(self, p: dict | None):
        """Inverse of get_preset. Reuses the set_* helpers (which handle the /100
        amount divisor). Accepts legacy key aliases like the existing opener does."""
        p = dict(p or {})
        amt = p.get("amount", p.get("strength", p.get("value", None)))
        if amt is not None:
            self.set_amount(float(amt))
        self.set_mode(str(p.get("mode", p.get("neutral_mode", "avg"))))
        self.set_channel(str(p.get("channel", p.get("target_channel", "G"))))
        self.set_preserve_lightness(p.get("preserve_lightness", p.get("preserve", True)))
        try:
            self.status_label.clear()
        except Exception:
            pass

    def _apply(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            QMessageBox.warning(self, "Remove Green", "No image.")
            return

        amount   = self.slider.value() / 100.0
        mode     = self.mode_box.currentData() or "avg"
        preserve = self.cb_preserve.isChecked()
        channel  = self.channel_box.currentData() or "G"

        # Build a preset dict so headless + replay use the same schema
        preset = {
            "amount": float(amount),
            "mode":   str(mode),
            "preserve_lightness": bool(preserve),
            "channel": str(channel),
        }

        # Apply to this doc
        remove_green_headless(self.doc, amount=amount, mode=mode,
                              preserve_lightness=preserve, channel=channel)

        # Log + record for Replay Last Action (if main supports it)
        if hasattr(self.main, "_log"):
            self.main._log(
                f"SCNR (channel={channel}): amount={amount:.2f}, mode={mode}, "
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
                    f"[Replay] Recorded SCNR preset "
                    f"(channel={channel}, amount={amount:.2f}, mode={mode}, "
                    f"preserve_lightness={preserve})"
                )
        except Exception:
            # Never let replay bookkeeping kill the dialog
            pass

        # Show applied status (channel name + doc name)
        try:
            ch_label = _CHANNEL_LABELS.get(str(channel).upper(), str(channel))
            doc_name = ""
            try:
                if hasattr(self.doc, "display_name"):
                    doc_name = self.doc.display_name() or ""
            except Exception:
                pass
            if doc_name:
                self.status_label.setText(
                    self.tr(f"✓ Removed {ch_label} from “{doc_name}”")
                )
            else:
                self.status_label.setText(self.tr(f"✓ Removed {ch_label}"))
        except Exception:
            self.status_label.setText(self.tr("✓ Applied"))

        # Dialog stays open so user can apply to other images
        # Refresh document reference for next operation
        self._refresh_document_from_active()

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            if self.main and hasattr(self.main, "_active_doc"):
                new_doc = self.main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
        except Exception:
            pass


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

        channel = preset.get("channel", preset.get("target_channel", "G"))
        dlg.set_channel(str(channel))

    dlg.show()

def apply_remove_green_preset_to_doc(main, doc, preset: dict):
    amt = float(preset.get("amount", preset.get("strength", preset.get("value", 1.0))))
    mode = str(preset.get("mode", preset.get("neutral_mode", "avg"))).lower()
    preserve = bool(preset.get("preserve_lightness", preset.get("preserve", True)))
    channel = str(preset.get("channel", preset.get("target_channel", "G"))).upper()
    remove_green_headless(doc, amount=amt, mode=mode,
                          preserve_lightness=preserve, channel=channel)
    if hasattr(main, "_log"):
        name = doc.display_name() if hasattr(doc, "display_name") else "Image"
        main._log(
            f"SCNR (headless) on '{name}'; channel={channel}, amount={amt:.2f}, "
            f"mode={mode}, preserve_lightness={preserve}"
        )

def open_remove_green_with_preset(main_window, preset: dict | None = None):
    """Double-click a Remove Green shortcut -> open the dialog seeded.
    Subwindow-first doc resolution (harmonized with _open_<tool>_tool)."""
    from PyQt6.QtGui import QIcon

    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    if doc is None or getattr(doc, "image", None) is None:
        return None

    dlg = RemoveGreenDialog(main_window, doc, parent=main_window)
    try:
        from setiastro.saspro.resources import green_path
        dlg.setWindowIcon(QIcon(green_path))
    except Exception:
        pass

    # No on-show reset -> seed directly before show.
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass

    try:
        main_window._remove_green_dialog = dlg
    except Exception:
        pass

    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg