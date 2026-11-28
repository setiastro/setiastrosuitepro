# src/setiastro/saspro/banding_blaster.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Banding Blaster: row or column banding reduction for the whole image.

The algorithm is CanonBandingReduction.js v0.9.1 by Georg Viehoever
(PixInsight JavaScript, 2009-2013, GPLv3), first ported to SASpro as the
Banding mode of Blemish Blaster. All credit for the method goes to Georg.

Per channel: shift every row so its median sits at the channel's global
median, scaled by Amount. With Protect highlights on, pixels brighter than
global median + sigma * robust sigma are left out of each row's median, so
stars and bright nebulae don't bias it. Vertical runs the same thing on
columns.

This is the one copy of the math: Blemish Blaster's Banding mode calls
reduce_banding too. It gives the same numbers as that mode's original port,
bit for bit, and is laid out for speed: rows are sorted in parallel, vertical
works on a contiguous transposed copy, and the per-row offsets are kept apart
from Amount so the preview can change Amount without measuring the image
again.

Standard tool shape (as pedestal.py / remove_green.py):
  - banding_offsets / apply_banding_offsets / reduce_banding: pure math, no Qt.
  - banding_blaster_headless(doc, preset): apply to a document (mask aware,
    one undo step). Used by Apply, preset drops, replay, scripts and the CLI.
  - BandingBlasterDialog: live preview with before/after, follows the active
    view, preset grip.
  - open_banding_blaster_dialog / open_banding_blaster_with_preset /
    apply_banding_blaster_preset_to_doc: entry points used by the main window.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from PyQt6.QtCore import (
    Qt, QCoreApplication, QSettings, QThread, QTimer, pyqtSignal,
)
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QRadioButton, QVBoxLayout, QWidget,
)

from setiastro.saspro.widgets.image_utils import (
    to_float01 as _to_float01,
    extract_mask_resized as _mask_resized,
    blend_with_mask as _blend_with_mask,
)
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)


def _tr(text: str) -> str:
    return QCoreApplication.translate("BandingBlaster", text)


COMMAND_ID = "banding_blaster"
STEP_NAME = "Banding Blaster"

DEFAULTS = {
    "direction": "horizontal",      # rows (Canon DSLR); "vertical" = columns
    "amount": 1.0,                  # 0..4; 1 = each row's median to the global
    "protect_highlights": True,
    "sigma": 1.0,                   # 0.01..5, robust sigmas above the median
}
AMOUNT_RANGE = (0.0, 4.0)
SIGMA_RANGE = (0.01, 5.0)


# ─────────────────────────────────────────────────────────────────────────
# Presets
# ─────────────────────────────────────────────────────────────────────────
def _as_bool(v, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _clamped(v, default: float, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f):
        return default
    return float(min(max(f, lo), hi))


def normalize_preset(p: dict | None) -> dict:
    """Any preset (shortcut, bundle, script, CLI) as the four canonical keys,
    with defaults filled in and values clamped to the dialog's ranges."""
    p = dict(p or {})
    d = str(p.get("direction", p.get("dir", DEFAULTS["direction"])) or "").strip().lower()
    return {
        "direction": "vertical" if d.startswith(("v", "col")) else "horizontal",
        "amount": _clamped(p.get("amount", DEFAULTS["amount"]), DEFAULTS["amount"], *AMOUNT_RANGE),
        "protect_highlights": _as_bool(p.get("protect_highlights", p.get("protect")),
                                       DEFAULTS["protect_highlights"]),
        "sigma": _clamped(p.get("sigma", DEFAULTS["sigma"]), DEFAULTS["sigma"], *SIGMA_RANGE),
    }


def describe_preset(p: dict) -> str:
    p = normalize_preset(p)
    prot = f"protect sigma={p['sigma']:.2f}" if p["protect_highlights"] else "no protection"
    return f"{p['direction']}, amount={p['amount']:.2f}, {prot}"


# ─────────────────────────────────────────────────────────────────────────
# Pure math (no Qt)
# ─────────────────────────────────────────────────────────────────────────
def _workers() -> int:
    return max(1, min(os.cpu_count() or 1, 16))


def _masked_row_medians(rows: np.ndarray, threshold: float, fallback: float) -> np.ndarray:
    """Median of each row over the pixels <= threshold; rows with none get
    fallback. The same expressions as the original port, so the same numbers:
    sort each row, survivors are its first n_below values."""
    H, W = rows.shape
    n_below = (rows <= threshold).sum(axis=1)
    srt = np.sort(rows, axis=1)
    half = n_below // 2
    r = np.arange(H)
    idx_hi = np.clip(half, 1, W - 1)
    idx_lo = idx_hi - 1
    med_odd = srt[r, np.clip(half, 0, W - 1)]
    med_even = 0.5 * (srt[r, idx_lo] + srt[r, idx_hi])
    odd = (n_below & 1).astype(bool)
    zero = n_below == 0
    med = np.where(zero, fallback, np.where(odd, med_odd, med_even))
    return med.astype(np.float32, copy=False)


def _row_medians(rows: np.ndarray, threshold: float | None, fallback: float,
                 pool: ThreadPoolExecutor | None) -> np.ndarray:
    """Per-row medians of a C-contiguous 2-D array, in blocks of rows on the
    pool (np.sort and np.median release the GIL)."""
    n = rows.shape[0]
    if threshold is None:
        def one(blk):
            return np.median(blk, axis=1).astype(np.float32, copy=False)
    else:
        def one(blk):
            return _masked_row_medians(blk, threshold, fallback)
    if pool is None or n < 64:
        return one(rows)
    step = max(16, -(-n // (_workers() * 4)))
    blocks = [rows[i:i + step] for i in range(0, n, step)]
    return np.concatenate(list(pool.map(one, blocks)))


def _channels(a: np.ndarray) -> int:
    return 1 if a.ndim == 2 else a.shape[2]


def banding_offsets(image: np.ndarray, *, protect_highlights: bool = True,
                    sigma: float = 1.0, direction: str = "horizontal",
                    channels=None) -> np.ndarray:
    """What Amount 1.0 adds to each row (column, for vertical) of each
    channel: global median minus that row's median. float32, shape (C, N).

    channels: the colour channels to correct (None = all); the others get
    zero offsets. A mono image is always corrected.
    The input is read as float32 clipped to [0, 1], as the correction is."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape: {img.shape}")
    vertical = str(direction).lower().startswith("v")
    n_ch = _channels(img)
    todo = (list(range(n_ch)) if (channels is None or img.ndim == 2)
            else [c for c in channels if 0 <= c < n_ch])

    def measure(c):
        # The channel in its original orientation, so its sums run in the
        # same order as the original's and give the same float32 result.
        src = img if img.ndim == 2 else img[..., c]
        ch = np.ascontiguousarray(np.clip(src, 0.0, 1.0))
        gmed = float(np.median(ch))
        # Robust sigma as Georg's script has it: 1.4826 * MEAN |x - median|
        robust_sigma = float(1.4826 * np.mean(np.abs(ch - gmed)))
        thr = gmed + sigma * robust_sigma if (protect_highlights and robust_sigma > 0.0) else None
        rows = np.ascontiguousarray(ch.T) if vertical else ch
        return rows, thr, gmed

    out = np.zeros((n_ch, img.shape[1] if vertical else img.shape[0]), np.float32)
    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        measured = list(pool.map(measure, todo))                    # channels at once
        for c, (rows, thr, gmed) in zip(todo, measured):
            out[c] = gmed - _row_medians(rows, thr, gmed, pool)     # float32, as the original has it
    return out


def apply_banding_offsets(image: np.ndarray, offsets: np.ndarray, amount: float,
                          direction: str = "horizontal") -> np.ndarray:
    """image clipped to [0, 1], plus offsets * amount along rows (or columns),
    clipped again. Same shape as image, float32."""
    out = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    vertical = str(direction).lower().startswith("v")
    for c in range(_channels(out)):
        fix = (offsets[c] * amount).astype(np.float32, copy=False)
        fix = fix[None, :] if vertical else fix[:, None]
        if out.ndim == 2:
            out += fix
        else:
            out[..., c] += fix
    np.clip(out, 0.0, 1.0, out=out)
    return out


def reduce_banding(image: np.ndarray, *, amount: float = 1.0,
                   protect_highlights: bool = True, sigma: float = 1.0,
                   direction: str = "horizontal", channels=None) -> np.ndarray:
    """The whole correction in one call (Blemish Blaster's Banding mode
    calls this too)."""
    off = banding_offsets(image, protect_highlights=protect_highlights,
                          sigma=sigma, direction=direction, channels=channels)
    return apply_banding_offsets(image, off, amount, direction)


def _offsets_for(src: np.ndarray, p: dict) -> np.ndarray:
    return banding_offsets(src, protect_highlights=p["protect_highlights"],
                           sigma=p["sigma"], direction=p["direction"])


# ─────────────────────────────────────────────────────────────────────────
# Headless apply
# ─────────────────────────────────────────────────────────────────────────
def banding_blaster_headless(doc, preset: dict | None = None, *,
                             offsets: np.ndarray | None = None) -> bool:
    """Correct doc's image with preset, blended through its active mask, as
    one undo step. offsets: banding_offsets() of this very image with these
    settings, when the caller already has them (the dialog's preview).
    Returns False when there was nothing to do."""
    if doc is None or getattr(doc, "image", None) is None:
        return False
    p = normalize_preset(preset)
    if p["amount"] <= 0.0:
        return False
    src = _to_float01(np.asarray(doc.image))
    if offsets is None:
        offsets = _offsets_for(src, p)
    out = apply_banding_offsets(src, offsets, p["amount"], p["direction"])
    mask = _mask_resized(doc, out.shape[0], out.shape[1])
    if mask is not None:
        out = _blend_with_mask(src, out, mask).astype(np.float32, copy=False)
    meta = {
        "step_name": STEP_NAME,
        "command_id": COMMAND_ID,
        "preset": dict(p),
        "banding_blaster": dict(p),
        "bit_depth": "32-bit floating point",
        "is_mono": (out.ndim == 2),
        "masked": mask is not None,
    }
    doc.apply_edit(out, metadata=meta, step_name=STEP_NAME)
    return True


def apply_banding_blaster_preset_to_doc(main, doc, preset: dict | None = None) -> bool:
    """Preset drops, replay, Function Bundles, scripts and the CLI."""
    p = normalize_preset(preset)
    done = banding_blaster_headless(doc, p)
    if done and hasattr(main, "_log"):
        try:
            name = doc.display_name() if hasattr(doc, "display_name") else "image"
            main._log(f"Banding Blaster on '{name}': {describe_preset(p)}")
        except Exception:
            pass
    return done


# ─────────────────────────────────────────────────────────────────────────
# Preview
# ─────────────────────────────────────────────────────────────────────────
_PREVIEW_TARGET_MEDIAN = 0.25


def _qimage_rgb8(a: np.ndarray) -> QImage:
    """float [0, 1] (mono or colour) as an RGB888 QImage that owns its bytes."""
    f = np.asarray(a, dtype=np.float32)
    if f.ndim == 3 and f.shape[2] == 1:
        f = f[..., 0]
    u8 = (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if u8.ndim == 2:
        try:            # the same bytes as np.repeat, without holding the GIL
            import cv2
            u8 = cv2.cvtColor(np.ascontiguousarray(u8), cv2.COLOR_GRAY2RGB)
        except Exception:
            u8 = np.repeat(u8[..., None], 3, axis=2)
    else:
        u8 = u8[..., :3]
    u8 = np.ascontiguousarray(u8)
    h, w = u8.shape[:2]
    q = QImage(u8.data, w, h, int(u8.strides[0]), QImage.Format.Format_RGB888).copy()
    try:
        from setiastro.saspro.color_space_manager import tag_qimage_with_working_color_space
        q = tag_qimage_with_working_color_space(q)
    except Exception:
        pass
    return q


class _PreviewWorker(QThread):
    """One preview off the GUI thread. Measuring the rows is the slow part
    (0.5-1 s on 24 MP colour), so the offsets come back to be reused while
    only Amount changes. A QImage, unlike a QPixmap, may be built here."""
    ready = pyqtSignal(object)

    def __init__(self, job: dict):
        super().__init__()          # no parent: thread_keeper owns it
        self.job = job

    def run(self):
        j = self.job
        res = {"gen": j["gen"], "img": j["img"], "key": j["key"]}
        try:
            from setiastro.saspro.widgets.before_after import (
                apply_display_stretch, display_stretch_params,
            )
            src, p = j["src"], j["preset"]
            off = j["offsets"]
            if off is None:
                off = _offsets_for(src, p)
            res["offsets"] = off
            out = apply_banding_offsets(src, off, p["amount"], p["direction"])
            if j["mask"] is not None:
                out = _blend_with_mask(src, out, j["mask"])
            stretch = j["stretch"]
            if stretch is None:
                # One stretch for both sides, measured on the original
                stretch = display_stretch_params(src, target_median=_PREVIEW_TARGET_MEDIAN,
                                                 linked=True, ignore_zeros=True)
                res["stretch"] = stretch
                res["before"] = _qimage_rgb8(apply_display_stretch(src, stretch))
            res["after"] = _qimage_rgb8(apply_display_stretch(out, stretch))
            res["largest"] = float(np.abs(off).max() * p["amount"]) if off.size else 0.0
            res["size"] = (int(src.shape[1]), int(src.shape[0]))
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
        self.job = None             # drop the image references now
        self.ready.emit(res)


class _PreviewView(QGraphicsView):
    """Wheel zooms about the cursor, drag pans. Smooth when zoomed out,
    nearest when zoomed in, so single-pixel rows stay sharp."""

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def _sync_smooth(self):
        # A pixmap item's own mode overrides the view's render hint, so set
        # it on every image, the compare's "before" copies included.
        mode = (Qt.TransformationMode.SmoothTransformation if self.transform().m11() < 1.0
                else Qt.TransformationMode.FastTransformation)
        for it in self.scene().items():
            if isinstance(it, QGraphicsPixmapItem):
                it.setTransformationMode(mode)

    def wheelEvent(self, ev):
        dy = ev.angleDelta().y()
        if dy == 0:
            ev.ignore()
            return
        f = 1.25 if dy > 0 else 0.8
        z = self.transform().m11() * f
        if 0.01 <= z <= 32.0:
            self.scale(f, f)
            self._sync_smooth()
        ev.accept()

    def fit(self):
        r = self.scene().sceneRect()
        if r.isEmpty():
            return
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
        self._sync_smooth()


# ─────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────
class BandingBlasterDialog(QDialog):
    """Banding Blaster as its own tool: preview with before/after, follows
    the active view, Apply is one undo step. WA_DeleteOnClose is set by the
    opener (open_banding_blaster_dialog)."""

    _GEOMETRY_KEY = "banding_blaster/window_geometry"

    def __init__(self, main, doc, parent=None):
        super().__init__(parent if parent is not None else main)
        self._main = main
        self.doc = doc
        self.setWindowTitle(self.tr("Banding Blaster"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)

        self._closed = False
        self._gen = 0                 # newest preview request
        self._offsets = None          # (image, key, offsets) of the last measurement
        self._stretch = None          # (image, stretch params, before pixmap)
        self._pv_thread = None
        self._pv_pending = False
        self._fit_pending = True
        self._geom_restored = False
        # One debounce for every control: holding a spin arrow is one preview
        self._pv_timer = QTimer(self)
        self._pv_timer.setSingleShot(True)
        self._pv_timer.setInterval(150)
        self._pv_timer.timeout.connect(self._start_preview)

        self._build_ui()

        # Follow the active view; disconnected in _shutdown
        self._follow_conn = None
        if hasattr(main, "currentDocumentChanged"):
            try:
                main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._follow_conn = True
            except Exception:
                self._follow_conn = None

        self._sync_sigma_enabled()
        self._schedule_preview()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        from setiastro.saspro.widgets.before_after import BeforeAfterCompare
        lay = QVBoxLayout(self)

        self.scene = QGraphicsScene(self)
        self.view = _PreviewView(self.scene, self)
        self._pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pix_item)
        self._cmp = BeforeAfterCompare(self.view, self.scene, [self._pix_item],
                                       settings_key="banding_blaster/compare", host=self)
        self._cmp.fitRequested.connect(self.view.fit)

        top = QHBoxLayout()
        top.addWidget(self._cmp.toolbar, 1)
        self.btn_fit = QPushButton(self.tr("Fit"))
        self.btn_fit.setToolTip(self.tr("Fit the whole image in the preview"))
        self.btn_fit.clicked.connect(self.view.fit)
        top.addWidget(self.btn_fit)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)

        # Controls: the same ranges, defaults and words as Blemish Blaster's
        # Banding mode
        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Direction:")))
        self.rb_horiz = QRadioButton(self.tr("Horizontal"))
        self.rb_vert = QRadioButton(self.tr("Vertical"))
        self.rb_horiz.setToolTip(self.tr("Row banding, as on Canon DSLRs"))
        self.rb_vert.setToolTip(self.tr("Column banding, as on some CMOS cameras"))
        self.rb_horiz.setChecked(True)
        self._dir_group = QButtonGroup(self)
        self._dir_group.addButton(self.rb_horiz, 0)
        self._dir_group.addButton(self.rb_vert, 1)
        row.addWidget(self.rb_horiz)
        row.addWidget(self.rb_vert)
        row.addSpacing(16)

        row.addWidget(QLabel(self.tr("Amount:")))
        self.sb_amount = QDoubleSpinBox(self)
        self.sb_amount.setRange(*AMOUNT_RANGE)
        self.sb_amount.setSingleStep(0.05)
        self.sb_amount.setDecimals(2)
        self.sb_amount.setValue(DEFAULTS["amount"])
        self.sb_amount.setToolTip(self.tr(
            "Strength of the correction. 1.0 fully equalises the per-row "
            "median to the global median. <1.0 = partial correction, "
            ">1.0 = over-correction (rarely useful)."))
        row.addWidget(self.sb_amount)
        row.addSpacing(16)

        self.cb_protect = QCheckBox(self.tr("Protect highlights"))
        self.cb_protect.setChecked(DEFAULTS["protect_highlights"])
        self.cb_protect.setToolTip(self.tr(
            "Exclude bright pixels (stars, DSO cores) from the per-row "
            "median so they don't bias the correction. Recommended ON "
            "for astronomical images."))
        row.addWidget(self.cb_protect)
        self.lbl_sigma = QLabel(self.tr("Sigma:"))
        row.addWidget(self.lbl_sigma)
        self.sb_sigma = QDoubleSpinBox(self)
        self.sb_sigma.setRange(*SIGMA_RANGE)
        self.sb_sigma.setSingleStep(0.1)
        self.sb_sigma.setDecimals(2)
        self.sb_sigma.setValue(DEFAULTS["sigma"])
        self.sb_sigma.setToolTip(self.tr(
            "Rejection threshold in robust-sigma units above the global "
            "median. Lower = more aggressive rejection of bright pixels. "
            "Only used when 'Protect highlights' is on."))
        row.addWidget(self.sb_sigma)
        row.addStretch(1)
        lay.addLayout(row)

        info = QHBoxLayout()
        self.lbl_status = QLabel(self.tr("Computing preview…"))
        self.lbl_status.setStyleSheet("color:#9a9a9a;")
        info.addWidget(self.lbl_status, 1)
        credit = QLabel(self.tr(
            "Algorithm: CanonBandingReduction by Georg Viehoever (GPLv3, 2009-2013)"))
        credit.setStyleSheet("color:#888; font-size:10px;")
        info.addWidget(credit, 0, Qt.AlignmentFlag.AlignRight)
        lay.addLayout(info)

        bottom = QHBoxLayout()
        # Grip + help: their own row under the buttons
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            from setiastro.saspro.help_support import make_help_button
            self.preset_drag_handle = PresetDragHandle(
                COMMAND_ID, self.get_preset, icon=_tool_icon(),
                tooltip=self.tr(
                    "Drag to the canvas to create a Banding Blaster shortcut "
                    "with these exact settings.\n"
                    "Drop directly on an image to apply them headlessly."),
                parent=self,
            )
            grip_row.addWidget(self.preset_drag_handle)
            grip_row.addStretch(1)
            grip_row.addWidget(make_help_button(COMMAND_ID, self))
        except Exception:
            pass
        bottom.addStretch(1)
        self.btn_apply = QPushButton(self.tr("Apply"))
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset = make_reset_button(self)
        self.btn_reset.clicked.connect(self._reset_to_factory)
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.clicked.connect(self.reject)
        bottom.addWidget(self.btn_apply)
        bottom.addWidget(self.btn_reset)
        bottom.addWidget(self.btn_close)
        lay.addLayout(bottom)
        lay.addLayout(grip_row)

        for sig in (self.rb_horiz.toggled, self.cb_protect.toggled):
            sig.connect(lambda *_: self._on_controls_changed())
        for sb in (self.sb_amount, self.sb_sigma):
            sb.valueChanged.connect(lambda *_: self._on_controls_changed())

        self.resize(1000, 720)
        # Factory defaults, before the opener seeds a preset. The before/after
        # compare toolbar is view state and keeps its mode.
        self._factory = snapshot_controls(self, exclude=(self._cmp.toolbar,))

    def _reset_to_factory(self):
        """Every setting back to its factory default; each control schedules
        the (debounced) preview as for a user change."""
        restore_controls(self._factory)

    def _sync_sigma_enabled(self):
        on = self.cb_protect.isChecked()
        self.sb_sigma.setEnabled(on)
        self.lbl_sigma.setEnabled(on)

    def _on_controls_changed(self):
        self._sync_sigma_enabled()
        self._schedule_preview()

    # ── presets ─────────────────────────────────────────────────────────
    def get_preset(self) -> dict:
        """The current controls as a preset: what Apply records for replay
        and what the grip carries."""
        return normalize_preset({
            "direction": "vertical" if self.rb_vert.isChecked() else "horizontal",
            "amount": self.sb_amount.value(),
            "protect_highlights": self.cb_protect.isChecked(),
            "sigma": self.sb_sigma.value(),
        })

    def seed_from_preset(self, preset: dict | None):
        p = normalize_preset(preset)
        widgets = (self.rb_horiz, self.rb_vert, self.sb_amount, self.cb_protect, self.sb_sigma)
        for w in widgets:
            w.blockSignals(True)
        try:
            (self.rb_vert if p["direction"] == "vertical" else self.rb_horiz).setChecked(True)
            self.sb_amount.setValue(p["amount"])
            self.cb_protect.setChecked(p["protect_highlights"])
            self.sb_sigma.setValue(p["sigma"])
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._on_controls_changed()

    @staticmethod
    def _key(p: dict) -> tuple:
        """What the offsets depend on (Amount is applied afterwards)."""
        return (p["direction"], p["protect_highlights"], round(p["sigma"], 6))

    def _cached_offsets(self, img, key):
        c = self._offsets
        return c[2] if (c is not None and c[0] is img and c[1] == key) else None

    # ── preview ─────────────────────────────────────────────────────────
    def _schedule_preview(self):
        if not self._closed:
            self._pv_timer.start()

    def _start_preview(self):
        from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
        self._pv_timer.stop()
        if self._closed:
            return
        img = getattr(self.doc, "image", None)
        if img is None:
            return
        if thread_running(self._pv_thread):
            self._pv_pending = True       # rerun with the newest settings when it lands
            return
        p = self.get_preset()
        key = self._key(p)
        src = _to_float01(np.asarray(img))
        st = self._stretch
        self._gen += 1
        job = {
            "gen": self._gen, "img": img, "key": key, "src": src, "preset": p,
            "offsets": self._cached_offsets(img, key),
            "mask": _mask_resized(self.doc, src.shape[0], src.shape[1]),
            "stretch": st[1] if (st is not None and st[0] is img) else None,
        }
        thr = _PreviewWorker(job)
        thr.ready.connect(self._on_preview_ready)
        thr.finished.connect(self._on_preview_finished)
        self._pv_thread = thr
        if job["offsets"] is None:
            self.lbl_status.setText(self.tr("Measuring the bands…"))
        start_owned_thread(thr)

    def _on_preview_ready(self, res: dict):
        if self._closed:
            return
        img = res.get("img")
        current = img is not None and img is getattr(self.doc, "image", None)
        if current and res.get("offsets") is not None:
            self._offsets = (img, res["key"], res["offsets"])
        if current and "stretch" in res:
            self._stretch = (img, res["stretch"], QPixmap.fromImage(res["before"]))
        if not current or res.get("gen") != self._gen:
            return                        # a newer request is queued or running
        if "error" in res:
            self.lbl_status.setText(self.tr("Preview failed: {0}").format(res["error"]))
            return
        w, h = res["size"]
        self._pix_item.setPixmap(QPixmap.fromImage(res["after"]))
        self._cmp.set_image_size(w, h)
        self._cmp.set_before(self._stretch[2] if self._stretch is not None else None)
        if self._fit_pending:
            self._fit_pending = False
            self.view.fit()
        what = (self.tr("Largest column shift: {0:.5f}") if self.rb_vert.isChecked()
                else self.tr("Largest row shift: {0:.5f}"))
        self.lbl_status.setText(what.format(res.get("largest", 0.0)))

    def _on_preview_finished(self):
        if self._pv_pending and not self._closed:
            self._pv_pending = False
            self._pv_timer.start()

    def _on_active_doc_changed(self, doc):
        if self._closed or doc is None or doc is self.doc:
            return
        if getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._offsets = None
        self._stretch = None
        self._fit_pending = True
        self._schedule_preview()

    # ── apply ───────────────────────────────────────────────────────────
    def _apply(self):
        doc = self.doc
        img = getattr(doc, "image", None)
        if img is None:
            QMessageBox.information(self, self.tr("Banding Blaster"), self.tr("Open an image first."))
            return
        p = self.get_preset()
        if p["amount"] <= 0.0:
            QMessageBox.information(self, self.tr("Banding Blaster"),
                                    self.tr("Amount is 0, so there is nothing to apply."))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            done = banding_blaster_headless(doc, p, offsets=self._cached_offsets(img, self._key(p)))
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, self.tr("Banding Blaster"), self.tr("Apply failed:\n{0}").format(e))
            return
        QApplication.restoreOverrideCursor()
        mw = self._main
        if done:
            try:
                if hasattr(mw, "_log"):
                    name = doc.display_name() if hasattr(doc, "display_name") else "image"
                    mw._log(f"Banding Blaster on '{name}': {describe_preset(p)}")
                if hasattr(mw, "_remember_last_headless_command"):
                    mw._remember_last_headless_command(COMMAND_ID, p, description=STEP_NAME)
            except Exception:
                pass
        self.accept()

    # ── life cycle ──────────────────────────────────────────────────────
    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._geom_restored:
            self._geom_restored = True
            try:
                g = QSettings().value(self._GEOMETRY_KEY, None)
                if g is not None:
                    self.restoreGeometry(g)
            except Exception:
                pass
        QTimer.singleShot(0, self.view.fit)

    def _shutdown(self):
        """Once, however the dialog goes (Close, Esc, Apply, the title bar)."""
        if self._closed:
            return
        self._closed = True
        self._pv_timer.stop()
        if self._follow_conn:
            try:
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
            except Exception:
                pass
        self._follow_conn = None
        try:
            QSettings().setValue(self._GEOMETRY_KEY, self.saveGeometry())
        except Exception:
            pass
        try:
            self._cmp.detach()
        except Exception:
            pass
        # The pixmaps and offsets are the big buffers: let them go now
        self._pix_item.setPixmap(QPixmap())
        self._offsets = None
        self._stretch = None

    def done(self, r):
        self._shutdown()
        super().done(r)

    def closeEvent(self, ev):
        self._shutdown()
        super().closeEvent(ev)


# ─────────────────────────────────────────────────────────────────────────
# Preset editor (desktop shortcut > Edit Preset)
# ─────────────────────────────────────────────────────────────────────────
class _BandingBlasterPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Banding Blaster — Preset"))
        p = normalize_preset(initial)

        self.rb_horiz = QRadioButton(self.tr("Horizontal"))
        self.rb_vert = QRadioButton(self.tr("Vertical"))
        (self.rb_vert if p["direction"] == "vertical" else self.rb_horiz).setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_horiz)
        grp.addButton(self.rb_vert)
        dir_w = QWidget(self)
        dir_row = QHBoxLayout(dir_w)
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(self.rb_horiz)
        dir_row.addWidget(self.rb_vert)
        dir_row.addStretch(1)

        self.sb_amount = QDoubleSpinBox(self)
        self.sb_amount.setRange(*AMOUNT_RANGE)
        self.sb_amount.setSingleStep(0.05)
        self.sb_amount.setDecimals(2)
        self.sb_amount.setValue(p["amount"])

        self.cb_protect = QCheckBox(self.tr("Protect highlights"))
        self.cb_protect.setChecked(p["protect_highlights"])

        self.sb_sigma = QDoubleSpinBox(self)
        self.sb_sigma.setRange(*SIGMA_RANGE)
        self.sb_sigma.setSingleStep(0.1)
        self.sb_sigma.setDecimals(2)
        self.sb_sigma.setValue(p["sigma"])
        self.sb_sigma.setEnabled(p["protect_highlights"])
        self.cb_protect.toggled.connect(self.sb_sigma.setEnabled)

        form = QFormLayout(self)
        form.addRow(self.tr("Direction:"), dir_w)
        form.addRow(self.tr("Amount:"), self.sb_amount)
        form.addRow("", self.cb_protect)
        form.addRow(self.tr("Sigma:"), self.sb_sigma)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return normalize_preset({
            "direction": "vertical" if self.rb_vert.isChecked() else "horizontal",
            "amount": self.sb_amount.value(),
            "protect_highlights": self.cb_protect.isChecked(),
            "sigma": self.sb_sigma.value(),
        })


# ─────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────
def _tool_icon() -> QIcon:
    try:
        from setiastro.saspro.resources import bandingblaster_path
        return QIcon(bandingblaster_path)
    except Exception:
        return QIcon()


def _resolve_active_doc(main):
    """The active document: the active MDI subwindow first, then the doc manager."""
    doc = None
    try:
        sw = main.mdi.activeSubWindow()
        if sw is not None:
            doc = getattr(sw.widget(), "document", None)
    except Exception:
        doc = None
    if doc is None:
        dm = getattr(main, "doc_manager", None) or getattr(main, "docman", None)
        if dm is not None:
            try:
                doc = (dm.get_active_document() if hasattr(dm, "get_active_document")
                       else getattr(dm, "active_document", None))
            except Exception:
                doc = None
    return doc


def open_banding_blaster_dialog(main, doc=None, preset: dict | None = None):
    """Menu / toolbar: the dialog on the active image (or doc), seeded from
    preset when one is given."""
    if doc is None:
        doc = _resolve_active_doc(main)
    if doc is None or getattr(doc, "image", None) is None:
        QMessageBox.information(main, _tr("Banding Blaster"), _tr("Open an image first."))
        return None
    dlg = BandingBlasterDialog(main, doc)
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.setWindowIcon(_tool_icon())
    if preset:
        dlg.seed_from_preset(preset)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg


def open_banding_blaster_with_preset(main_window, preset: dict | None = None):
    """Double-clicking a Banding Blaster shortcut: the dialog, seeded."""
    return open_banding_blaster_dialog(main_window, None, preset or {})
