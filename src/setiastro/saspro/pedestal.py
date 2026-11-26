# saspro/pedestal.py
from __future__ import annotations
import os
import numpy as np

from PyQt6.QtCore import Qt, QEvent, QPointF, QSettings, QTimer, QThread, pyqtSignal, QCoreApplication, QT_TRANSLATE_NOOP
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QScrollArea, QMessageBox, QWidget, QFrame
)
from PyQt6.QtGui import QPixmap, QImage, QIcon

# Shared utilities (same helpers Star Stretch relies on)
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01
from setiastro.saspro.shortcuts import PresetDragHandle
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)

# Icon is optional; fall back gracefully if resources doesn't expose one yet.
try:
    from setiastro.saspro.resources import pedestal_icon_path
except Exception:
    pedestal_icon_path = None


def _tr(text: str) -> str:
    return QCoreApplication.translate("Pedestal", text)

# Target median for the preview-only auto display stretch (linear data).
_DISPLAY_TARGET_MEDIAN = 0.25


# ─────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────
# "min"           : subtract the channel minimum (classic pedestal removal)
# "percentile"    : subtract a low percentile (e.g. P0.1) so a handful of
#                   grossly-low cold pixels can't peg the floor near zero and
#                   render min-subtract a no-op
# "first_nonzero" : subtract the smallest strictly-positive value — useful when
#                   an image has a hard zero floor but real signal begins at a
#                   small offset
MODE_LABELS = {
    "min":           QT_TRANSLATE_NOOP("Pedestal", "Image minimum"),
    "percentile":    QT_TRANSLATE_NOOP("Pedestal", "Low percentile (P{pct})"),
    "first_nonzero": QT_TRANSLATE_NOOP("Pedestal", "First non-zero value"),
}
_MODE_ORDER = ["min", "percentile", "first_nonzero"]


# ─────────────────────────────────────────────────────────────────────────
# Pure math (importable / testable, no Qt)
# ─────────────────────────────────────────────────────────────────────────
def _prep_float01(img: np.ndarray) -> np.ndarray:
    """float32 image; gently compress absurd (>1) ranges so 'min-subtract'
    behaves. No-op for already-normalized data."""
    a = np.asarray(img)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if a.size:
        mx = float(a.max())
        if mx > 5.0:
            a = a / mx
    return a


def _pedestal_scalar(ch: np.ndarray, mode: str, pct: float) -> float:
    """The value to subtract for a single channel (or the whole array in
    global mode)."""
    if ch.size == 0:
        return 0.0
    if mode == "percentile":
        return float(np.percentile(ch, float(pct)))
    if mode == "first_nonzero":
        pos = ch[ch > 0.0]
        return float(pos.min()) if pos.size else 0.0
    # "min" (default / fallback)
    return float(ch.min())


def compute_pedestal(a: np.ndarray, mode: str = "min", pct: float = 0.1,
                     per_channel: bool = True) -> list[float]:
    """Return the per-channel (or single global) values that WOULD be
    subtracted — used for the readout without mutating the image."""
    a = _prep_float01(a)
    if a.ndim == 2 or not per_channel:
        return [_pedestal_scalar(a, mode, pct)]
    return [_pedestal_scalar(a[..., c], mode, pct) for c in range(a.shape[2])]


def remove_pedestal_array(a: np.ndarray, mode: str = "min", pct: float = 0.1,
                          per_channel: bool = True):
    """
    Subtract the chosen pedestal. Preserves shape (2D or 3D), clips to [0,1].
    Returns (out_float32, subtracted_values, clipped_counts) where
    clipped_counts[c] is the number of pixels in channel c driven below zero
    (i.e. clamped to black) by the subtraction.
    """
    a = _prep_float01(a)

    if a.ndim == 2:
        v = _pedestal_scalar(a, mode, pct)
        out = a - v
        vals = [v]
        clipped = [int(np.count_nonzero(a < v))]
    elif per_channel:
        out = np.empty_like(a, dtype=np.float32)
        vals: list[float] = []
        clipped: list[int] = []
        for c in range(a.shape[2]):
            ch = a[..., c]
            v = _pedestal_scalar(ch, mode, pct)
            out[..., c] = ch - v
            vals.append(v)
            clipped.append(int(np.count_nonzero(ch < v)))
    else:
        v = _pedestal_scalar(a, mode, pct)   # one value across all channels
        out = a - v
        vals = [v] * a.shape[2]
        clipped = [int(np.count_nonzero(a[..., c] < v)) for c in range(a.shape[2])]

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False), vals, clipped


# Back-compat alias for anything importing the old private name.
def _remove_pedestal_array(a: np.ndarray) -> np.ndarray:
    out, _vals, _clip = remove_pedestal_array(a, mode="min", per_channel=True)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Preview helpers
# ─────────────────────────────────────────────────────────────────────────
def _as_qimage_rgb8(float01: np.ndarray) -> QImage:
    f = np.asarray(float01, dtype=np.float32)
    if f.ndim == 3 and f.shape[2] == 1:
        f = f[..., 0]

    # mono: 8-bit first, then replicate the bytes (same bytes as stacking floats)
    buf8 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    if buf8.ndim == 2:
        try:   # same bytes as np.repeat, which holds the GIL ~0.4 s on 26 MP
            import cv2
            buf8 = cv2.cvtColor(np.ascontiguousarray(buf8), cv2.COLOR_GRAY2RGB)
        except Exception:
            buf8 = np.repeat(buf8[..., None], 3, axis=2)
    buf8 = np.ascontiguousarray(buf8)
    h, w, _ = buf8.shape
    bpl = int(buf8.strides[0])
    try:
        from PyQt6 import sip
        qimg = QImage(sip.voidptr(buf8.ctypes.data), w, h, bpl, QImage.Format.Format_RGB888)
        qimg._keepalive = buf8
        return qimg.copy()
    except Exception:
        data = buf8.tobytes()
        qimg = QImage(data, w, h, bpl, QImage.Format.Format_RGB888)
        return qimg.copy()


def _display_stretch(img: np.ndarray) -> np.ndarray:
    """Preview-only screen stretch for linear data. no_black_clip=True so
    the pedestal's effect on the floor stays visible. Never mutates the
    data that gets applied."""
    a = np.asarray(img, dtype=np.float32)
    try:
        from setiastro.saspro.imageops.stretch import (
            stretch_mono_image, stretch_color_image,
        )
    except Exception:
        return a
    try:
        if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):
            return stretch_mono_image(a.squeeze(), _DISPLAY_TARGET_MEDIAN,
                                      no_black_clip=True)
        return stretch_color_image(a, _DISPLAY_TARGET_MEDIAN, linked=True,
                                   no_black_clip=True)
    except Exception:
        return a


def _resample_mask(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    mh, mw = mask.shape[:2]
    th, tw = out_hw
    if (mh, mw) == (th, tw):
        return mask
    yi = np.linspace(0, mh - 1, th).astype(np.int32)
    xi = np.linspace(0, mw - 1, tw).astype(np.int32)
    return mask[yi][:, xi]


def _blend_mask(processed: np.ndarray, src_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """m*out + (1-m)*src with the active mask (float01, any size)."""
    src = _to_float01(src_image)
    out = processed.astype(np.float32, copy=False)

    th, tw = out.shape[:2]
    m = _resample_mask(mask, (th, tw))
    if out.ndim == 3 and out.shape[2] == 3:
        m = m[..., None]

    if src.ndim == 2 and out.ndim == 3 and out.shape[2] == 3:
        src = np.stack([src] * 3, axis=-1)
    elif src.ndim == 3 and src.shape[2] == 3 and out.ndim == 2:
        src = src[..., 0]

    return (m * out + (1.0 - m) * src).astype(np.float32, copy=False)


class _PedestalPreviewWorker(QThread):
    """One preview off the GUI thread: subtract, mask blend, display stretch
    and the 8-bit image. On a 22 MP RGB frame that is ~2-2.5 s, all of which
    used to run on the GUI thread on every control change. A QImage, unlike a
    QPixmap, may be built here."""
    ready = pyqtSignal(object)    # dict, see run()

    def __init__(self, doc, src, params: dict, mask, autostretch: bool):
        super().__init__()        # no parent: thread_keeper owns it
        self.doc = doc
        self.src = src
        self.params = dict(params)
        self.mask = mask
        self.autostretch = bool(autostretch)

    def run(self):
        res = {"doc": self.doc, "params": self.params}
        try:
            src = self.src
            out, vals, clipped = remove_pedestal_array(
                src,
                mode=self.params["mode"],
                pct=float(self.params["percentile"]),
                per_channel=bool(self.params["per_channel"]),
            )
            if self.mask is not None:
                out = _blend_mask(out, src, self.mask)
            disp = _display_stretch(out) if self.autostretch else out
            res.update(out=out, vals=vals, clipped=clipped,
                       total=(int(src.shape[0] * src.shape[1]) if src.ndim >= 2 else int(src.size)),
                       qimg=_as_qimage_rgb8(_to_float01(disp)))
        except Exception as e:
            res["error"] = str(e)
        self.ready.emit(res)


# ─────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────
class RemovePedestalDialog(QDialog):
    """
    Remove Pedestal for SASpro.
    - Works on the active ImageDocument (passed in).
    - Live preview (op is cheap, computed on the UI thread).
    - 'Apply to Document' records history via doc.apply_edit(step_name="Pedestal Removal").
    - Carries a PI-style drag grip so settings can be dropped onto the canvas
      (make a shortcut) or onto an image (headless apply).
    """
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Remove Pedestal"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._main = parent
        self.doc = document
        self._preview: np.ndarray | None = None
        self._preview_params: dict | None = None   # settings _preview was computed with
        self._preview_doc = None
        self._preview_vals: list[float] | None = None
        self._pv_thread = None
        self._pv_pending = False
        # One debounce for every control: a burst of changes (holding a spin
        # arrow, typing 0.25) is one recompute, not one per keystroke.
        self._pv_timer = QTimer(self)
        self._pv_timer.setSingleShot(True)
        self._pv_timer.setInterval(150)
        self._pv_timer.timeout.connect(self._start_preview_job)
        self._headless = False
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        # Track active-document changes (Fabio hook) so the dialog can be reused.
        self._connected_current_doc_changed = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_current_doc_changed = True
            except Exception:
                self._connected_current_doc_changed = False

        self._pix: QPixmap | None = None
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()

        # ── UI ────────────────────────────────────────────────────────────
        main = QHBoxLayout(self)

        # Left column (controls)
        left = QVBoxLayout()
        info = QLabel(
            self.tr("Subtract a pedestal (a constant offset) from the image.\n\n"
            "• Image minimum — classic; removes the darkest value per channel.\n"
            "• Low percentile — ignores a few grossly-low cold pixels that would\n"
            "  otherwise peg the floor and make min-subtract do nothing.\n"
            "• First non-zero value — subtracts the smallest value above zero,\n"
            "  for data sitting on a hard zero floor with a small real offset.")
        )
        info.setWordWrap(True)
        left.addWidget(info)

        # Mode
        self.cmb_mode = QComboBox()
        for m in _MODE_ORDER:
            self.cmb_mode.addItem(self._mode_label(m), userData=m)
        self.cmb_mode.setCurrentIndex(0)  # "min" default
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        left.addWidget(QLabel(self.tr("Subtract:")))
        left.addWidget(self.cmb_mode)

        # Percentile value (only relevant for percentile mode)
        self.lbl_pct = QLabel(self.tr("Percentile:"))
        self.spin_pct = QDoubleSpinBox()
        self.spin_pct.setRange(0.0, 50.0)
        self.spin_pct.setDecimals(3)
        self.spin_pct.setSingleStep(0.05)
        self.spin_pct.setValue(0.1)                 # P0.1 default
        self.spin_pct.setSuffix(" %")
        self.spin_pct.setToolTip(
            self.tr("Percentile of pixel values to subtract.\n"
            "0.1 % is a good starting point — it steps past a handful of\n"
            "cold/hot outliers without eating real signal."))
        self.spin_pct.valueChanged.connect(lambda *_: self._schedule_preview())
        left.addWidget(self.lbl_pct)
        left.addWidget(self.spin_pct)

        # Per-channel
        self.chk_per_channel = QCheckBox(self.tr("Per-channel (RGB)"))
        self.chk_per_channel.setChecked(True)
        self.chk_per_channel.setToolTip(
            self.tr("On: compute a separate pedestal for R, G and B (also removes a\n"
            "background color cast).\nOff: one pedestal across all channels."))
        self.chk_per_channel.toggled.connect(lambda *_: self._schedule_preview())
        left.addWidget(self.chk_per_channel)

        # Auto display stretch — preview only; the applied data stays linear.
        self.chk_autostretch = QCheckBox(self.tr("Auto display stretch (preview only)"))
        self.chk_autostretch.setChecked(True)
        self.chk_autostretch.setToolTip(
            self.tr("Linear data looks nearly black. This applies a temporary screen\n"
            "stretch to the PREVIEW only so you can see what the pedestal is\n"
            "doing — it does not change the data that gets applied."))
        self.chk_autostretch.toggled.connect(self._on_autostretch_toggled)
        left.addWidget(self.chk_autostretch)

        # Readout of what's being subtracted
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); sep.setFrameShadow(QFrame.Shadow.Sunken)
        left.addWidget(sep)
        self.lbl_vals = QLabel(self.tr("Subtracting: —"))
        self.lbl_vals.setWordWrap(True)
        self.lbl_vals.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        left.addWidget(self.lbl_vals)

        # Buttons row
        rowb = QHBoxLayout()
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_apply = QPushButton(self.tr("Apply to Document"))
        self.btn_reset = make_reset_button(self)
        rowb.addWidget(self.btn_preview)
        rowb.addWidget(self.btn_apply)
        rowb.addWidget(self.btn_reset)
        left.addLayout(rowb)

        left.addStretch(1)

        # ── Drag-to-canvas grip (PI-style "new instance") ─────────────────
        drag_row = QHBoxLayout()
        drag_row.setContentsMargins(0, 0, 0, 0)
        self.preset_drag_handle = PresetDragHandle(
            "pedestal",
            self._pedestal_params,
            icon=QIcon(pedestal_icon_path) if pedestal_icon_path else None,
            tooltip=self.tr(
                "Drag to the canvas to create a Remove Pedestal shortcut\n"
                "with these exact settings.\n"
                "Drop directly on an image to apply them headlessly."
            ),
            parent=self,
        )
        drag_row.addWidget(self.preset_drag_handle)
        drag_row.addStretch(1)
        self.btn_help = make_help_button("pedestal", self)
        drag_row.addWidget(self.btn_help)
        left.addLayout(drag_row)

        main.addLayout(left, 0)

        # Right column (preview with zoom/pan)
        right = QVBoxLayout()
        zoombar = QHBoxLayout()
        b_out = QPushButton(self.tr("Zoom Out"))
        b_in = QPushButton(self.tr("Zoom In"))
        b_fit = QPushButton(self.tr("Fit to Preview"))
        b_out.clicked.connect(self._zoom_out)
        b_in.clicked.connect(self._zoom_in)
        b_fit.clicked.connect(self._fit)
        zoombar.addWidget(b_out); zoombar.addWidget(b_in); zoombar.addWidget(b_fit)
        right.addLayout(zoombar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)

        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)

        right.addWidget(self.scroll, 1)
        main.addLayout(right, 1)

        # signals
        self.btn_preview.clicked.connect(self._start_preview_job)
        self.btn_apply.clicked.connect(self._apply_to_doc)
        self.btn_reset.clicked.connect(self._reset_to_factory)

        # Factory defaults for Reset (the display stretch is preview-only)
        self._factory = snapshot_controls(self, exclude=(self.chk_autostretch,))

        # initial state (the first worker preview replaces this text)
        self._sync_pct_enabled()
        self.label.setText(self.tr("Computing preview…"))
        self._schedule_preview()

    # --- mode label helper -------------------------------------------------
    def _mode_label(self, mode: str) -> str:
        lbl = _tr(MODE_LABELS[mode]) if mode in MODE_LABELS else mode
        if "{pct}" in lbl:
            lbl = lbl.replace("{pct}", f"{self.spin_pct.value():g}" if hasattr(self, "spin_pct") else "0.1")
        return lbl

    def _current_mode(self) -> str:
        return self.cmb_mode.currentData() or "min"

    def _sync_pct_enabled(self):
        is_pct = (self._current_mode() == "percentile")
        self.lbl_pct.setEnabled(is_pct)
        self.spin_pct.setEnabled(is_pct)

    def _reset_to_factory(self):
        restore_controls(self._factory)
        # the percentile entry's text carries the value; refresh it
        i = self.cmb_mode.findData("percentile")
        if i >= 0:
            self.cmb_mode.setItemText(i, self._mode_label("percentile"))

    # --- active document change -------------------------------------------
    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._preview = None
        self._schedule_preview()

    # --- UI change handlers ------------------------------------------------
    def _on_mode_changed(self, *_):
        # keep the percentile label live in the combo item text
        idx = self.cmb_mode.currentIndex()
        m = self.cmb_mode.itemData(idx)
        if m == "percentile":
            self.cmb_mode.setItemText(idx, self._mode_label("percentile"))
        self._sync_pct_enabled()
        self._schedule_preview()

    # --- preview / processing ---------------------------------------------
    def _schedule_preview(self):
        """Coalesce rapid control changes into a single recompute."""
        self._pv_timer.start()

    def _start_preview_job(self):
        """Preview on a worker; one at a time. A change made while one runs
        re-runs it with the latest settings when it lands."""
        self._pv_timer.stop()
        if self.doc is None or getattr(self.doc, "image", None) is None:
            return
        if thread_running(self._pv_thread):
            self._pv_pending = True
            return
        if self._current_mode() == "percentile":
            idx = self.cmb_mode.currentIndex()
            self.cmb_mode.setItemText(idx, self._mode_label("percentile"))
        mask, _mid, _name = self._active_mask_layer()
        thr = _PedestalPreviewWorker(
            self.doc, np.asarray(self.doc.image), self._pedestal_params(), mask,
            bool(self.chk_autostretch.isChecked()),
        )
        thr.ready.connect(self._on_preview_ready)
        thr.finished.connect(self._on_preview_finished)
        self._pv_thread = thr
        start_owned_thread(thr)

    def _on_preview_ready(self, res: dict):
        if res.get("doc") is not self.doc:
            return                       # the active document changed meanwhile
        if "error" in res:
            self.lbl_vals.setText(self.tr("Subtracting: (error: {0})").format(res['error']))
            return
        self._preview = res["out"]
        self._preview_params = res["params"]
        self._preview_doc = res["doc"]
        self._preview_vals = list(res["vals"])
        self._update_readout(res["vals"], res["clipped"], res["total"])
        first = self._pix is None
        self._pix = QPixmap.fromImage(res["qimg"])
        if first:
            self._fit()
        else:
            self._apply_zoom()

    def _on_preview_finished(self):
        if self._pv_pending:
            self._pv_pending = False
            self._pv_timer.start()

    def _recompute_preview(self):
        if self.doc is None or getattr(self.doc, "image", None) is None:
            return
        # refresh the percentile label if it's showing
        if self._current_mode() == "percentile":
            idx = self.cmb_mode.currentIndex()
            self.cmb_mode.setItemText(idx, self._mode_label("percentile"))

        try:
            src = np.asarray(self.doc.image)
            out, vals, clipped = remove_pedestal_array(
                src,
                mode=self._current_mode(),
                pct=float(self.spin_pct.value()),
                per_channel=self.chk_per_channel.isChecked(),
            )
        except Exception as e:
            self.lbl_vals.setText(self.tr("Subtracting: (error: {0})").format(e))
            return

        total = int(src.shape[0] * src.shape[1]) if src.ndim >= 2 else int(src.size)
        out = self._blend_with_mask(out)
        self._preview = out
        self._preview_params = self._pedestal_params()
        self._preview_doc = self.doc
        self._preview_vals = list(vals)
        self._update_readout(vals, clipped, total)
        self._update_preview_pix(out)

    def _update_readout(self, vals: list[float], clipped: list[int] | None = None,
                        total: int = 0):
        if not vals:
            self.lbl_vals.setText(self.tr("Subtracting: —"))
            return
        clipped = clipped or []
        names = (["R", "G", "B", "A"][:len(vals)]) if len(vals) > 1 else [self.tr("Value")]
        lines = []
        for i, (n, v) in enumerate(zip(names, vals)):
            cnt = clipped[i] if i < len(clipped) else 0
            pct = (100.0 * cnt / total) if total else 0.0
            lines.append(self.tr("{0}: {1:.6f}   →  clipped {2:,} px ({3:.3f}%)").format(n, v, cnt, pct))
        self.lbl_vals.setText(self.tr("Subtracting / pixels clamped to black:\n{0}").format("\n".join(lines)))

    def _apply_to_doc(self):
        if not self._headless and self.isVisible():
            try:
                self._save_window_geometry()
            except Exception:
                pass
        # The preview is what gets applied; recompute it here if it is missing
        # or was made with other settings / for another document (a worker
        # preview may still be in flight).
        if (self._preview is None or self._preview_params != self._pedestal_params()
                or self._preview_doc is not self.doc):
            self._recompute_preview()
        if self._preview is None:
            QMessageBox.information(self, self.tr("No image"), self.tr("Open an image first."))
            return
        self._finish_apply()

    def _pedestal_params(self) -> dict:
        """Canonical preset — single source of truth for apply + the drag handle."""
        return {
            "mode":        self._current_mode(),               # "min"|"percentile"|"first_nonzero"
            "percentile":  float(self.spin_pct.value()),       # only meaningful for percentile mode
            "per_channel": bool(self.chk_per_channel.isChecked()),
        }

    def seed_from_preset(self, p: dict | None) -> None:
        """Load a preset dict (same schema as _pedestal_params) into the live
        controls, so a double-clicked shortcut opens the dialog seeded."""
        if not p:
            return
        widgets = [self.cmb_mode, self.spin_pct, self.chk_per_channel]
        for w in widgets:
            try:
                w.blockSignals(True)
            except Exception:
                pass
        try:
            mode = str(p.get("mode", "min")).lower()
            i = self.cmb_mode.findData(mode)
            self.cmb_mode.setCurrentIndex(i if i >= 0 else 0)

            pct = float(p.get("percentile", p.get("pct", 0.1)))
            self.spin_pct.setValue(pct)

            self.chk_per_channel.setChecked(bool(p.get("per_channel", True)))

            # refresh percentile label text now that value is set
            idx = self.cmb_mode.currentIndex()
            if self.cmb_mode.itemData(idx) == "percentile":
                self.cmb_mode.setItemText(idx, self._mode_label("percentile"))
        finally:
            for w in widgets:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass
        self._sync_pct_enabled()
        self._schedule_preview()

    def _finish_apply(self):
        try:
            _marr, mid, mname = self._active_mask_layer()
            if self._preview_vals is not None:
                # what the preview subtracted (same settings, see _apply_to_doc);
                # compute_pedestal reports one value in global mode
                vals = (list(self._preview_vals) if self.chk_per_channel.isChecked()
                        else list(self._preview_vals[:1]))
            else:
                vals = compute_pedestal(
                    np.asarray(self.doc.image),
                    mode=self._current_mode(),
                    pct=float(self.spin_pct.value()),
                    per_channel=self.chk_per_channel.isChecked(),
                )
            meta = {
                "step_name": "Pedestal Removal",
                "command_id": "pedestal",
                "preset": self._pedestal_params(),
                "bit_depth": "32-bit floating point",
                "is_mono": (self._preview.ndim == 2),
                "pedestal": {
                    "mode": self._current_mode(),
                    "percentile": float(self.spin_pct.value()),
                    "per_channel": bool(self.chk_per_channel.isChecked()),
                    "subtracted": [float(v) for v in vals],
                },
                # mask bookkeeping (matches Star Stretch)
                "masked": bool(mid),
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            }
            self.doc.apply_edit(self._preview.copy(), metadata=meta, step_name="Pedestal Removal")

            mw = self._find_main_window()
            if mw and hasattr(mw, "_log"):
                mw._log("Pedestal Removal: applied to document.")

            # Record as last headless-style command for Replay
            try:
                if mw and hasattr(mw, "_remember_last_headless_command"):
                    mw._remember_last_headless_command(
                        "pedestal",
                        self._pedestal_params(),
                        description="Pedestal Removal",
                    )
            except Exception:
                pass

        except Exception as e:
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))
            return

        if not self._headless and self.isVisible():
            try:
                self._save_window_geometry()
            except Exception:
                pass
        self.close()

    # --- preview rendering -------------------------------------------------
    def _autostretch_for_display(self, img: np.ndarray) -> np.ndarray:
        return _display_stretch(img)

    def _on_autostretch_toggled(self, *_):
        self._schedule_preview()        # the display stretch runs on the worker

    def _update_preview_pix(self, img: np.ndarray | None):
        if img is None:
            self.label.clear(); self._pix = None; return
        disp = img
        if getattr(self, "chk_autostretch", None) is not None and self.chk_autostretch.isChecked():
            disp = self._autostretch_for_display(img)
        qimg = _as_qimage_rgb8(_to_float01(disp))
        self._pix = QPixmap.fromImage(qimg)
        self._apply_zoom()

    def _apply_zoom(self):
        if self._pix is None:
            return
        scaled = self._pix.scaled(self._pix.size() * self._zoom,
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

    # --- zoom/pan ----------------------------------------------------------
    def _zoom_at(self, factor: float, vp_pos: QPointF):
        """Zoom by `factor`, keeping the content point under `vp_pos`
        (viewport coordinates) fixed on screen."""
        if self._pix is None:
            return
        old = self._zoom
        new = float(max(0.05, min(old * factor, 8.0)))
        if abs(new - old) < 1e-9:
            return

        h = self.scroll.horizontalScrollBar()
        v = self.scroll.verticalScrollBar()
        vp = self.scroll.viewport().size()

        # Content coord under the cursor now (accounting for centering margin
        # when the image is smaller than the viewport), as a fraction 0..1.
        old_w = self._pix.width() * old
        old_h = self._pix.height() * old
        margin_x = max(0.0, (vp.width() - old_w) / 2.0)
        margin_y = max(0.0, (vp.height() - old_h) / 2.0)
        cx = h.value() + float(vp_pos.x()) - margin_x
        cy = v.value() + float(vp_pos.y()) - margin_y
        fx = (cx / old_w) if old_w > 0 else 0.5
        fy = (cy / old_h) if old_h > 0 else 0.5

        self._zoom = new
        self._apply_zoom()

        # Defer so the scrollbars' ranges reflect the new content size first.
        def _anchor():
            new_w = self._pix.width() * self._zoom
            new_h = self._pix.height() * self._zoom
            h.setValue(int(round(fx * new_w - float(vp_pos.x()))))
            v.setValue(int(round(fy * new_h - float(vp_pos.y()))))
        QTimer.singleShot(0, _anchor)

    def _viewport_center(self) -> QPointF:
        r = self.scroll.viewport().rect().center()
        return QPointF(r)

    def _zoom_in(self):  self._zoom_at(1.25, self._viewport_center())
    def _zoom_out(self): self._zoom_at(1.0 / 1.25, self._viewport_center())
    def _fit(self):
        if self._pix is None: return
        vp = self.scroll.viewport().size()
        if self._pix.width() == 0 or self._pix.height() == 0: return
        s = min(vp.width() / self._pix.width(), vp.height() / self._pix.height())
        self._set_zoom(max(0.05, s))

    def _set_zoom(self, z: float):
        self._zoom = float(max(0.05, min(z, 8.0)))
        self._apply_zoom()

    # --- event filter (wheel zoom + panning) ------------------------------
    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel:
                dy = ev.pixelDelta().y()
                if dy != 0:
                    abs_dy = abs(dy)
                    ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if abs_dy <= 3:
                        base_factor = 1.012 if ctrl_down else 1.010
                    elif abs_dy <= 10:
                        base_factor = 1.025 if ctrl_down else 1.020
                    else:
                        base_factor = 1.040 if ctrl_down else 1.030
                    factor = base_factor if dy > 0 else 1.0 / base_factor
                else:
                    dy = ev.angleDelta().y()
                    if dy == 0:
                        ev.accept(); return True
                    ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    step = 1.25 if ctrl_down else 1.15
                    factor = step if dy > 0 else 1.0 / step
                self._zoom_at(factor, ev.position())
                ev.accept(); return True

            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept(); return True

            if ev.type() == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                h = self.scroll.horizontalScrollBar()
                v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - int(d.x()))
                v.setValue(v.value() - int(d.y()))
                self._pan_start = ev.position()
                ev.accept(); return True

            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True

        return super().eventFilter(obj, ev)

    # --- helper ------------------------------------------------------------
    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    # --- mask helpers ------------------------------------------------------
    def _active_mask_layer(self):
        """Return (mask_array_float01, mask_id, mask_name) or (None, None, None)."""
        doc = self.doc
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None, None, None
        layer = getattr(doc, "masks", {}).get(mid)
        if layer is None:
            return None, None, None
        m = np.asarray(getattr(layer, "data", None), dtype=np.float32)
        if m is None or m.size == 0:
            return None, None, None
        if m.dtype.kind in "ui":
            m = m / float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0:
                m = m / mx
        m = np.clip(m, 0.0, 1.0)
        return m, mid, getattr(layer, "name", "Mask")

    def _resample_mask_if_needed(self, mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        return _resample_mask(mask, out_hw)

    def _blend_with_mask(self, processed: np.ndarray) -> np.ndarray:
        mask, _mid, _name = self._active_mask_layer()
        if mask is None:
            return processed
        return _blend_mask(processed, self.doc.image, mask)

    # --- lifecycle / geometry ---------------------------------------------
    def closeEvent(self, ev):
        if not self._headless and self.isVisible():
            try:
                self._save_window_geometry()
            except Exception:
                pass
        try:
            if self._connected_current_doc_changed and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._connected_current_doc_changed = False
        super().closeEvent(ev)

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("remove_pedestal_ui/window_geometry", None)
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("remove_pedestal_ui/window_geometry", self.saveGeometry())
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True

            def _after_restore_refit():
                self._restore_window_geometry()
                self._fit()
            QTimer.singleShot(0, _after_restore_refit)
            return
        QTimer.singleShot(0, self._fit)


# ─────────────────────────────────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────────────────────────────────
def _resolve_active_doc(main_window):
    """Active document: prefer the visible MDI subwindow, then the doc manager."""
    doc = None
    try:
        sw = main_window.mdi.activeSubWindow()
        if sw is not None:
            w = sw.widget()
            doc = getattr(w, "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main_window, "doc_manager", getattr(main_window, "docman", None))
        if dm is not None:
            doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                   else getattr(dm, "active_document", None))
    return doc


def open_remove_pedestal_dialog(main_window):
    """Toolbar / menu entry — open the live dialog on the active document."""
    doc = _resolve_active_doc(main_window)
    if doc is None or getattr(doc, "image", None) is None:
        return
    dlg = RemovePedestalDialog(main_window, doc)
    try:
        if pedestal_icon_path:
            dlg.setWindowIcon(QIcon(pedestal_icon_path))
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg


def open_remove_pedestal_with_preset(main_window, preset: dict | None = None):
    """Double-click preset-open path (ShortcutManager.trigger_with_preset)."""
    doc = _resolve_active_doc(main_window)
    if doc is None or getattr(doc, "image", None) is None:
        return
    dlg = RemovePedestalDialog(main_window, doc)
    try:
        if pedestal_icon_path:
            dlg.setWindowIcon(QIcon(pedestal_icon_path))
    except Exception:
        pass
    try:
        dlg.seed_from_preset(preset or {})
    except Exception:
        pass
    dlg.show(); dlg.raise_(); dlg.activateWindow()
    return dlg


def remove_pedestal(main, target_doc=None, preset: dict | None = None):
    """
    Headless / one-shot pedestal removal on the *currently active* content
    (ROI-aware), honoring an optional preset. Called by:
      • the headless drop-on-image path (main window dispatch), and
      • shortcut "Run" without opening the UI.
    With no preset it reproduces the classic per-channel image-minimum subtract,
    so any legacy caller keeps its old behavior.
    """
    doc = target_doc
    dm = getattr(main, "doc_manager", None) or getattr(main, "docman", None)

    if doc is None and dm is not None:
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
        if doc is None and hasattr(dm, "get_active_document"):
            try:
                doc = dm.get_active_document()
            except Exception:
                doc = None

    if doc is None:
        ad = getattr(main, "_active_doc", None)
        doc = ad() if callable(ad) else ad

    if doc is None or getattr(doc, "image", None) is None:
        return

    p = dict(preset or {})
    mode = str(p.get("mode", "min")).lower()
    pct = float(p.get("percentile", p.get("pct", 0.1)))
    per_channel = bool(p.get("per_channel", True))

    try:
        src = np.asarray(doc.image)
        out, vals, _clip = remove_pedestal_array(src, mode=mode, pct=pct, per_channel=per_channel)
        meta = {
            "step_name": "Pedestal Removal",
            "command_id": "pedestal",
            "preset": {"mode": mode, "percentile": pct, "per_channel": per_channel},
            "bit_depth": "32-bit floating point",
            "is_mono": (out.ndim == 2),
            "pedestal": {
                "mode": mode,
                "percentile": pct,
                "per_channel": per_channel,
                "subtracted": [float(v) for v in vals],
            },
        }
        doc.apply_edit(out, metadata=meta, step_name="Pedestal Removal")
        if hasattr(main, "_log"):
            main._log(f"Pedestal Removal ({mode})")
    except Exception as e:
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(main, _tr("Pedestal Removal"), _tr("Failed:\n{0}").format(e))
        except Exception:
            pass