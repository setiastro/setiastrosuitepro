# pro/stat_stretch.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QSize, QEvent
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QDoubleSpinBox,
    QCheckBox, QPushButton, QScrollArea, QWidget, QMessageBox, QSlider, QToolBar, QToolButton, QComboBox,QProgressBar, QApplication
)
from PyQt6.QtGui import QImage, QPixmap, QMouseEvent, QCursor
import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt, QSize, QEvent, QTimer
from PyQt6.QtWidgets import QProgressDialog, QApplication
from .doc_manager import ImageDocument
# use your existing stretch code
from setiastro.saspro.imageops.stretch import (
    stretch_mono_image,
    stretch_color_image,
    _compute_blackpoint_sigma,
    _compute_blackpoint_sigma_per_channel,
)

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.luminancerecombine import LUMA_PROFILES

class _StretchWorker(QObject):
    finished = pyqtSignal(object, str)  # (out_array_or_None, error_message_or_empty)

    def __init__(self, dialog_ref):
        super().__init__()
        self._dlg = dialog_ref

    @pyqtSlot()
    def run(self):
        try:
            # dialog might be closing; guard
            if self._dlg is None or sip.isdeleted(self._dlg):
                self.finished.emit(None, "Dialog was closed.")
                return

            out = self._dlg._run_stretch()
            self.finished.emit(out, "")
        except Exception as e:
            self.finished.emit(None, str(e))


class StatisticalStretchDialog(QDialog):
    """
    Non-destructive preview; Apply commits to the document image.
    """
    def __init__(self, parent, document: ImageDocument):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Statistical Stretch"))

        # --- Top-level non-modal tool window (Linux WM friendly) ---
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        # --- State / refs ---
        self._main = parent
        self.doc = document

        self._last_preview = None
        self._preview_qimg = None
        self._preview_scale = 1.0
        self._fit_mode = True

        self._panning = False
        self._pan_last = None  # QPoint

        self._hdr_knee_user_locked = False
        self._pending_close = False
        self._suppress_replay_record = False

        # ---- Clip-stats scheduling (define EARLY so init callbacks can't crash) ----
        self._clip_timer = None

        def _schedule_clip_stats():
            # Safe early stub; once timer exists it will debounce
            if getattr(self, "_job_running", False):
                return
            if sip.isdeleted(self):
                return
            t = getattr(self, "_clip_timer", None)
            if t is None:
                return
            t.start()

        self._schedule_clip_stats = _schedule_clip_stats

        self._thread = None
        self._worker = None
        self._follow_conn = None
        self._job_running = False
        self._job_mode = ""

        # --- Follow active document changes (optional) ---
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._follow_conn = True
            except Exception:
                self._follow_conn = None

        # ------------------------------------------------------------------
        # Controls
        # ------------------------------------------------------------------

        # Target median
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(0.01, 0.99)
        self.spin_target.setSingleStep(0.01)
        self.spin_target.setValue(0.25)
        self.spin_target.setDecimals(3)

        # Linked channels
        self.chk_linked = QCheckBox(self.tr("Linked channels"))
        self.chk_linked.setChecked(False)

        # Normalize
        self.chk_normalize = QCheckBox(self.tr("Normalize to [0..1]"))
        self.chk_normalize.setChecked(False)

        # --- Black point sigma row ---
        self.row_bp = QWidget()
        bp_lay = QHBoxLayout(self.row_bp)
        bp_lay.setContentsMargins(0, 0, 0, 0)
        bp_lay.setSpacing(8)

        bp_lay.addWidget(QLabel(self.tr("Black point σ:")))

        self.sld_bp = QSlider(Qt.Orientation.Horizontal)
        self.sld_bp.setRange(50, 600)   # 0.50 .. 6.00
        self.sld_bp.setValue(500)       # 5.00 default (matches your label)
        bp_lay.addWidget(self.sld_bp, 1)

        self.lbl_bp = QLabel(f"{self.sld_bp.value()/100:.2f}")
        bp_lay.addWidget(self.lbl_bp)

        bp_tip = self.tr(
            "Black point (σ) controls how aggressively the dark background is clipped.\n"
            "Higher values clip more (darker background, more contrast), but can crush faint dust.\n"
            "Lower values preserve faint background, but may leave the image gray.\n"
            "Tip: start around 2.7–5.0 depending on gradient/noise."
        )
        self.row_bp.setToolTip(bp_tip)
        self.sld_bp.setToolTip(bp_tip)
        self.lbl_bp.setToolTip(bp_tip)

        # No black clipping
        self.chk_no_black_clip = QCheckBox(self.tr("No black clipping (Old Stat Stretch behavior)"))
        self.chk_no_black_clip.setChecked(False)
        self.chk_no_black_clip.setToolTip(self.tr(
            "Disables black-point clipping.\n"
            "Uses the image minimum as the black point (preserves faint background),\n"
            "but the result may look flatter / hazier."
        ))

        # --- HDR compress ---
        self.chk_hdr = QCheckBox(self.tr("HDR highlight compress"))
        self.chk_hdr.setChecked(False)
        self.chk_hdr.setToolTip(self.tr(
            "Compresses bright highlights after the stretch.\n"
            "Use lightly: high values can flatten the image and create star ringing."
        ))

        self.hdr_row = QWidget()
        hdr_lay = QVBoxLayout(self.hdr_row)
        hdr_lay.setContentsMargins(0, 0, 0, 0)
        hdr_lay.setSpacing(6)

        # HDR amount row
        row_a = QHBoxLayout()
        row_a.setContentsMargins(0, 0, 0, 0)
        row_a.setSpacing(8)
        row_a.addWidget(QLabel(self.tr("Amount:")))

        self.sld_hdr_amt = QSlider(Qt.Orientation.Horizontal)
        self.sld_hdr_amt.setRange(0, 100)
        self.sld_hdr_amt.setValue(15)
        row_a.addWidget(self.sld_hdr_amt, 1)

        self.lbl_hdr_amt = QLabel(f"{self.sld_hdr_amt.value()/100:.2f}")
        row_a.addWidget(self.lbl_hdr_amt)

        self.sld_hdr_amt.setToolTip(self.tr(
            "Compression strength (0–1).\n"
            "Start low (0.10–0.15). Too much can flatten the image and ring stars."
        ))
        self.lbl_hdr_amt.setToolTip(self.sld_hdr_amt.toolTip())

        # HDR knee row
        row_k = QHBoxLayout()
        row_k.setContentsMargins(0, 0, 0, 0)
        row_k.setSpacing(8)
        row_k.addWidget(QLabel(self.tr("Knee:")))

        self.sld_hdr_knee = QSlider(Qt.Orientation.Horizontal)
        self.sld_hdr_knee.setRange(10, 95)
        self.sld_hdr_knee.setValue(75)
        row_k.addWidget(self.sld_hdr_knee, 1)

        self.lbl_hdr_knee = QLabel(f"{self.sld_hdr_knee.value()/100:.2f}")
        row_k.addWidget(self.lbl_hdr_knee)

        self.sld_hdr_knee.setToolTip(self.tr(
            "Where compression begins (0–1).\n"
            "Good starting point: knee ≈ target median + 0.10 to + 0.20.\n"
            "Example: target 0.25 → knee 0.35–0.45."
        ))
        self.lbl_hdr_knee.setToolTip(self.sld_hdr_knee.toolTip())

        hdr_lay.addLayout(row_a)
        hdr_lay.addLayout(row_k)

        self.hdr_row.setEnabled(False)

        # --- Luma-only row (checkbox + dropdown on one line) ---
        self.luma_row = QWidget()
        lr = QHBoxLayout(self.luma_row)
        lr.setContentsMargins(0, 0, 0, 0)
        lr.setSpacing(8)

        self.chk_luma_only = QCheckBox(self.tr("Luminance-only"))
        self.chk_luma_only.setChecked(False)

        self.cmb_luma = QComboBox()
        keys = list(LUMA_PROFILES.keys())

        def _cat(k):
            return str(LUMA_PROFILES.get(k, {}).get("category", ""))

        keys.sort(key=lambda k: (_cat(k), k.lower()))
        self.cmb_luma.addItems(keys)
        self.cmb_luma.setCurrentText("rec709")
        self.cmb_luma.setEnabled(False)

        lr.addWidget(self.chk_luma_only)
        lr.addWidget(QLabel(self.tr("Mode:")))
        lr.addWidget(self.cmb_luma, 1)

        # --- Luma blend row (only meaningful when Luma-only is enabled) ---
        self.luma_blend_row = QWidget()
        lbr = QHBoxLayout(self.luma_blend_row)
        lbr.setContentsMargins(0, 0, 0, 0)
        lbr.setSpacing(8)

        lbr.addWidget(QLabel(self.tr("Luma blend:")))

        self.sld_luma_blend = QSlider(Qt.Orientation.Horizontal)
        self.sld_luma_blend.setRange(0, 100)   # 0=normal linked, 100=luma-only
        self.sld_luma_blend.setValue(60)       # nice default: “mostly luma” but tame
        lbr.addWidget(self.sld_luma_blend, 1)

        self.lbl_luma_blend = QLabel(f"{self.sld_luma_blend.value()/100:.2f}")
        lbr.addWidget(self.lbl_luma_blend)

        tip = self.tr(
            "Blend between a normal linked RGB stretch (0.00) and a luminance-only stretch (1.00).\n"
            "Use this to tame the saturation punch of luma-only."
        )
        self.luma_blend_row.setToolTip(tip)
        self.sld_luma_blend.setToolTip(tip)
        self.lbl_luma_blend.setToolTip(tip)

        self.luma_blend_row.setEnabled(False)

        # --- Curves boost ---
        self.chk_curves = QCheckBox(self.tr("Curves boost"))
        self.chk_curves.setChecked(False)

        self.curves_row = QWidget()
        cr_lay = QHBoxLayout(self.curves_row)
        cr_lay.setContentsMargins(0, 0, 0, 0)
        cr_lay.setSpacing(8)

        cr_lay.addWidget(QLabel(self.tr("Strength:")))
        self.sld_curves = QSlider(Qt.Orientation.Horizontal)
        self.sld_curves.setRange(0, 100)
        self.sld_curves.setValue(20)
        cr_lay.addWidget(self.sld_curves, 1)

        self.lbl_curves_val = QLabel(f"{self.sld_curves.value()/100:.2f}")
        cr_lay.addWidget(self.lbl_curves_val)

        self.curves_row.setEnabled(False)

        # ------------------------------------------------------------------
        # Preview UI
        # ------------------------------------------------------------------
        self.preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(QSize(320, 240))
        self.preview_label.setScaledContents(False)
        self.preview_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.viewport().installEventFilter(self)

        # Zoom buttons
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_100 = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit")

        zoom_row.addStretch(1)
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_100, self.btn_zoom_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)

        # Buttons
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_apply   = QPushButton(self.tr("Apply"))
        self.btn_close   = QPushButton(self.tr("Close"))
        self.btn_reset = QPushButton(self.tr("Reset ⟳"))  

        self.btn_clipstats = QPushButton(self.tr("Clip stats"))
        self.lbl_clipstats = QLabel("")
        self.lbl_clipstats.setWordWrap(True)
        self.lbl_clipstats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_clipstats.setMinimumHeight(38)
        self.lbl_clipstats.setFrameShape(QLabel.Shape.StyledPanel)
        self.lbl_clipstats.setFrameShadow(QLabel.Shadow.Sunken)
        self.lbl_clipstats.setContentsMargins(6, 4, 6, 4)

        # --- In-UI busy indicator (Wayland-friendly) ---
        self.busy_row = QWidget()
        br = QHBoxLayout(self.busy_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(8)

        self.lbl_busy = QLabel(self.tr("Processing…"))
        self.lbl_busy.setStyleSheet("color:#888;")
        self.pbar_busy = QProgressBar()
        self.pbar_busy.setRange(0, 0)          # indeterminate
        self.pbar_busy.setTextVisible(False)
        self.pbar_busy.setFixedHeight(10)

        br.addWidget(self.lbl_busy)
        br.addWidget(self.pbar_busy, 1)

        self.busy_row.setVisible(False)        # hidden until needed

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        form = QFormLayout()
        form.addRow(self.tr("Target median:"), self.spin_target)
        form.addRow("", self.chk_linked)
        form.addRow("", self.row_bp)
        form.addRow("", self.chk_no_black_clip)
        form.addRow("", self.chk_hdr)
        form.addRow("", self.hdr_row)
        form.addRow("", self.luma_row)
        form.addRow("", self.luma_blend_row)
        form.addRow("", self.chk_normalize)
        form.addRow("", self.chk_curves)
        form.addRow("", self.curves_row)

        left = QVBoxLayout()
        left.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_preview)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_clipstats)
        btn_row.addStretch(1)        
        btn_row.addWidget(self.btn_reset)  
        btn_row.addStretch(1)
        left.addLayout(btn_row)

        left.addWidget(self.lbl_clipstats)
        left.addWidget(self.busy_row)
        left.addStretch(1)

        right = QVBoxLayout()
        right.addLayout(zoom_row)
        right.addWidget(self.preview_scroll, 1)

        main = QHBoxLayout(self)
        main.addLayout(left, 0)
        main.addLayout(right, 1)

        # ------------------------------------------------------------------
        # Behavior / wiring
        # ------------------------------------------------------------------

        # Blackpoint slider -> label + debounced clip stats
        def _on_bp_changed(v: int):
            self.lbl_bp.setText(f"{v/100:.2f}")
            self._schedule_clip_stats()

        self.sld_bp.valueChanged.connect(_on_bp_changed)

        # No-black-clip toggles blackpoint UI + triggers stats
        def _on_no_black_clip_toggled(on: bool):
            self.row_bp.setEnabled(not on)
            self._schedule_clip_stats()

        self.chk_no_black_clip.toggled.connect(_on_no_black_clip_toggled)
        _on_no_black_clip_toggled(self.chk_no_black_clip.isChecked())

        # Curves
        self.chk_curves.toggled.connect(self.curves_row.setEnabled)
        self.sld_curves.valueChanged.connect(lambda v: self.lbl_curves_val.setText(f"{v/100:.2f}"))

        # HDR enable toggles HDR row
        self.chk_hdr.toggled.connect(self.hdr_row.setEnabled)
        self.sld_hdr_amt.valueChanged.connect(lambda v: self.lbl_hdr_amt.setText(f"{v/100:.2f}"))
        self.sld_hdr_knee.valueChanged.connect(lambda v: self.lbl_hdr_knee.setText(f"{v/100:.2f}"))
        self.sld_hdr_knee.sliderPressed.connect(lambda: setattr(self, "_hdr_knee_user_locked", True))

        # Auto-suggest HDR knee from target (unless user locked)
        def _suggest_hdr_knee_from_target():
            if getattr(self, "_hdr_knee_user_locked", False):
                return
            t = float(self.spin_target.value())
            knee = float(np.clip(t + 0.10, 0.10, 0.95))
            self.sld_hdr_knee.blockSignals(True)
            self.sld_hdr_knee.setValue(int(round(knee * 100)))
            self.sld_hdr_knee.blockSignals(False)
            self.lbl_hdr_knee.setText(f"{knee:.2f}")

        self.spin_target.valueChanged.connect(_suggest_hdr_knee_from_target)

        # Zoom buttons
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_by(1/1.25))
        self.btn_zoom_100.clicked.connect(self._zoom_reset_100)
        self.btn_zoom_fit.clicked.connect(self._fit_preview)

        # Main buttons
        self.btn_preview.clicked.connect(self._do_preview)
        self.btn_apply.clicked.connect(self._do_apply)
        self.btn_reset.clicked.connect(self._reset_defaults)   
        self.btn_close.clicked.connect(self.close)
        self.btn_clipstats.clicked.connect(self._do_clip_stats)

        # Debounced clip stats timer
        self._clip_timer = QTimer(self)
        self._clip_timer.setSingleShot(True)
        self._clip_timer.setInterval(500)
        self._clip_timer.timeout.connect(self._do_clip_stats)

        # Initialize UI state
        _suggest_hdr_knee_from_target()
        self.sld_luma_blend.valueChanged.connect(
            lambda v: self.lbl_luma_blend.setText(f"{v/100:.2f}")
        )

        # Luma-only: one unified handler for all dependent UI state
        def _on_luma_only_toggled(on: bool):
            # enable luma mode dropdown only when luma-only is on
            self.cmb_luma.setEnabled(on)

            # linked channels doesn't make sense in luma-only mode
            self.chk_linked.setEnabled(not on)

            # luma blend row only meaningful when luma-only is enabled
            self.luma_blend_row.setEnabled(on)

            # mode-affecting => refresh clip stats
            self._schedule_clip_stats()

        self.chk_luma_only.toggled.connect(_on_luma_only_toggled)
        _on_luma_only_toggled(self.chk_luma_only.isChecked())


        # Initial preview + clip stats
        self._populate_initial_preview()


    # ----- helpers -----
    def _show_busy(self, title: str, text: str):
        # title kept for signature compatibility; not shown
        try:
            self.lbl_busy.setText(text or self.tr("Processing…"))
            self.busy_row.setVisible(True)
            # make sure UI repaints before thread work starts
            QApplication.processEvents()
        except Exception:
            pass

    def _hide_busy(self):
        try:
            if getattr(self, "busy_row", None) is not None:
                self.busy_row.setVisible(False)
        except Exception:
            pass


    def _set_controls_enabled(self, enabled: bool):
        try:
            self.btn_preview.setEnabled(enabled)
            self.btn_apply.setEnabled(enabled)
            if getattr(self, "btn_reset", None) is not None:
                self.btn_reset.setEnabled(enabled)  # <-- NEW
            if getattr(self, "btn_clipstats", None) is not None:
                self.btn_clipstats.setEnabled(enabled)
        except Exception:
            pass

    def _reset_defaults(self):
        """Reset all controls back to factory defaults."""
        if getattr(self, "_job_running", False):
            return

        # Defaults (must match your __init__ setValue/setChecked calls)
        DEFAULT_TARGET = 0.25
        DEFAULT_LINKED = False
        DEFAULT_NORMALIZE = False
        DEFAULT_BP_SLIDER = 500          # 5.00
        DEFAULT_NO_BLACK_CLIP = False

        DEFAULT_HDR_ON = False
        DEFAULT_HDR_AMT = 15             # 0.15
        DEFAULT_HDR_KNEE = 75            # 0.75

        DEFAULT_LUMA_ONLY = False
        DEFAULT_LUMA_MODE = "rec709"
        DEFAULT_LUMA_BLEND = 60          # 0.60

        DEFAULT_CURVES_ON = False
        DEFAULT_CURVES_STRENGTH = 20     # 0.20

        # Avoid cascading signal storms while we set everything
        widgets = [
            self.spin_target,
            self.chk_linked,
            self.chk_normalize,
            self.sld_bp,
            self.chk_no_black_clip,
            self.chk_hdr,
            self.sld_hdr_amt,
            self.sld_hdr_knee,
            self.chk_luma_only,
            self.cmb_luma,
            self.sld_luma_blend,
            self.chk_curves,
            self.sld_curves,
        ]

        old_blocks = []
        for w in widgets:
            try:
                old_blocks.append((w, w.blockSignals(True)))
            except Exception:
                pass

        try:
            # Reset “user locked” HDR knee behavior
            self._hdr_knee_user_locked = False

            # Core controls
            self.spin_target.setValue(DEFAULT_TARGET)
            self.chk_linked.setChecked(DEFAULT_LINKED)
            self.chk_normalize.setChecked(DEFAULT_NORMALIZE)

            # Black point
            self.chk_no_black_clip.setChecked(DEFAULT_NO_BLACK_CLIP)
            self.sld_bp.setValue(DEFAULT_BP_SLIDER)
            self.lbl_bp.setText(f"{DEFAULT_BP_SLIDER/100:.2f}")

            # HDR
            self.chk_hdr.setChecked(DEFAULT_HDR_ON)
            self.sld_hdr_amt.setValue(DEFAULT_HDR_AMT)
            self.lbl_hdr_amt.setText(f"{DEFAULT_HDR_AMT/100:.2f}")
            self.sld_hdr_knee.setValue(DEFAULT_HDR_KNEE)
            self.lbl_hdr_knee.setText(f"{DEFAULT_HDR_KNEE/100:.2f}")

            # Luma-only + mode + blend
            self.chk_luma_only.setChecked(DEFAULT_LUMA_ONLY)
            if DEFAULT_LUMA_MODE:
                self.cmb_luma.setCurrentText(DEFAULT_LUMA_MODE)
            self.sld_luma_blend.setValue(DEFAULT_LUMA_BLEND)
            self.lbl_luma_blend.setText(f"{DEFAULT_LUMA_BLEND/100:.2f}")

            # Curves
            self.chk_curves.setChecked(DEFAULT_CURVES_ON)
            self.sld_curves.setValue(DEFAULT_CURVES_STRENGTH)
            self.lbl_curves_val.setText(f"{DEFAULT_CURVES_STRENGTH/100:.2f}")

        finally:
            # Restore signal states
            for w, _prev in old_blocks:
                try:
                    w.blockSignals(False)
                except Exception:
                    pass

        # Re-apply dependent enable/disable states exactly like normal interactions
        try:
            # no-black-clip disables BP row
            self.row_bp.setEnabled(not self.chk_no_black_clip.isChecked())
        except Exception:
            pass

        try:
            # HDR enables HDR row
            self.hdr_row.setEnabled(self.chk_hdr.isChecked())
        except Exception:
            pass

        try:
            # Curves enables curves row
            self.curves_row.setEnabled(self.chk_curves.isChecked())
        except Exception:
            pass

        try:
            # Luma-only enables dropdown + blend row, disables linked
            luma_on = self.chk_luma_only.isChecked()
            self.cmb_luma.setEnabled(luma_on)
            self.luma_blend_row.setEnabled(luma_on)
            self.chk_linked.setEnabled(not luma_on)
        except Exception:
            pass

        # Auto-suggest HDR knee from target (since we cleared lock)
        try:
            t = float(self.spin_target.value())
            knee = float(np.clip(t + 0.10, 0.10, 0.95))
            self.sld_hdr_knee.setValue(int(round(knee * 100)))
            self.lbl_hdr_knee.setText(f"{knee:.2f}")
        except Exception:
            pass

        # Refresh baseline preview + clip stats
        self._populate_initial_preview()


    def _clip_mode_label(self, imgf: np.ndarray) -> str:
        # Mono image
        if imgf.ndim == 2 or (imgf.ndim == 3 and imgf.shape[2] == 1):
            return self.tr("Mono")

        # RGB image
        luma_only = bool(getattr(self, "chk_luma_only", None) and self.chk_luma_only.isChecked())
        if luma_only:
            return self.tr("Luma-only (L ≤ bp)")

        linked = bool(getattr(self, "chk_linked", None) and self.chk_linked.isChecked())
        if linked:
            return self.tr("Linked (L ≤ bp)")

        return self.tr("Unlinked (any channel ≤ bp)")


    def _do_clip_stats(self):
        imgf = self._get_source_float()
        if imgf is None or imgf.size == 0:
            self.lbl_clipstats.setText(self.tr("No image loaded."))
            return

        sig = float(self.sld_bp.value()) / 100.0
        no_black_clip = bool(self.chk_no_black_clip.isChecked())

        # Modes that affect how we count / threshold
        luma_only = bool(getattr(self, "chk_luma_only", None) and self.chk_luma_only.isChecked())
        linked = bool(getattr(self, "chk_linked", None) and self.chk_linked.isChecked())

        # Outputs we’ll fill
        bp = None            # float threshold (mono / L-based modes)
        bp3 = None           # per-channel thresholds (unlinked RGB)
        clipped = None       # [H,W] bool

        # --- Compute blackpoint threshold(s) exactly like stretch.py ---
        if imgf.ndim == 2 or (imgf.ndim == 3 and imgf.shape[2] == 1):
            mono = imgf.squeeze().astype(np.float32, copy=False)
            if no_black_clip:
                bp = float(mono.min())
            else:
                bp, _ = _compute_blackpoint_sigma(mono, sig)

            clipped = (mono <= bp)

        else:
            rgb = imgf.astype(np.float32, copy=False)

            if luma_only or linked:
                # One threshold for the pixel: use luminance proxy
                L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                if no_black_clip:
                    bp = float(L.min())
                else:
                    bp, _ = _compute_blackpoint_sigma(L, sig)

                clipped = (L <= bp)

            else:
                # Unlinked: per-channel thresholds
                if no_black_clip:
                    bp3 = np.array(
                        [float(rgb[..., 0].min()),
                        float(rgb[..., 1].min()),
                        float(rgb[..., 2].min())],
                        dtype=np.float32
                    )
                else:
                    bp3 = _compute_blackpoint_sigma_per_channel(rgb, sig).astype(np.float32, copy=False)

                # Pixel considered clipped if ANY channel would clip
                clipped = np.any(rgb <= bp3.reshape((1, 1, 3)), axis=2)

        # --- Count pixels (NOT rgb elements) ---
        clipped_count = int(np.count_nonzero(clipped))
        total = int(clipped.size)
        pct = 100.0 * clipped_count / max(1, total)

        # --- Optional masked-area stats ---
        masked_note = ""
        m = self._active_mask_array()
        if m is not None:
            affected = (m > 0.01)
            aff_total = int(np.count_nonzero(affected))
            aff_clip = int(np.count_nonzero(clipped & affected))
            aff_pct = 100.0 * aff_clip / max(1, aff_total)
            masked_note = self.tr(f" | masked area: {aff_clip:,}/{aff_total:,} ({aff_pct:.4f}%)")

        mode_lbl = self._clip_mode_label(imgf)

        # --- No-black-clip message (must be mode-aware) ---
        if no_black_clip:
            if imgf.ndim == 2 or (imgf.ndim == 3 and imgf.shape[2] == 1):
                bp_text = self.tr(f"min={float(bp):.6f}")
            else:
                if luma_only or linked:
                    bp_text = self.tr(f"L min={float(bp):.6f}")
                else:
                    # bp3 exists here
                    bp_text = self.tr(
                        f"R min={float(bp3[0]):.6f}, G min={float(bp3[1]):.6f}, B min={float(bp3[2]):.6f}"
                    )

            self.lbl_clipstats.setText(
                self.tr(f"Black clipping disabled ({mode_lbl}). Threshold={bp_text}: "
                        f"{clipped_count:,}/{total:,} pixels ({pct:.4f}%)") + masked_note
            )
            return

        # --- Normal message: show correct threshold(s) ---
        if (imgf.ndim == 3 and imgf.shape[2] == 3) and not (luma_only or linked):
            # Unlinked RGB: show per-channel thresholds
            bp_disp = self.tr(
                f"R={float(bp3[0]):.6f}, G={float(bp3[1]):.6f}, B={float(bp3[2]):.6f}"
            )
        else:
            # Mono or L-based: single threshold
            bp_disp = self.tr(f"{float(bp):.6f}")

        self.lbl_clipstats.setText(
            self.tr(f"Black clip ({mode_lbl}) @ {bp_disp}: "
                    f"{clipped_count:,}/{total:,} pixels ({pct:.4f}%)") + masked_note
        )


    def _start_stretch_job(self, mode: str):
        """
        mode: 'preview' or 'apply'
        """
        if getattr(self, "_job_running", False):
            return

        self._job_running = True
        self._job_mode = mode

        self._set_controls_enabled(False)
        self._show_busy("Statistical Stretch", "Processing preview…" if mode == "preview" else "Applying stretch…")


        self._thread = QThread(self._main)
        self._worker = _StretchWorker(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_stretch_done)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()


    def _get_source_float(self) -> np.ndarray:
        """
        Return a float32 array scaled into ~[0..1] for stretching.
        """
        src = np.asarray(self.doc.image)
        if src is None or src.size == 0:
            return None

        if np.issubdtype(src.dtype, np.integer):
            # Assume 16-bit astro sources by default; adjust if you prefer
            scale = 65535.0 if src.dtype.itemsize >= 2 else 255.0
            return (src.astype(np.float32) / scale).clip(0, 1)
        else:
            f = src.astype(np.float32)
            # If values are way above 1 (linear calibrated data), compress softly
            mx = float(f.max()) if f.size else 1.0
            if mx > 5.0:
                f = f / mx
            return f

    def _apply_current_zoom(self):
        """Apply the current zoom mode (fit or manual) to the preview image."""
        if self._preview_qimg is None:
            return
        if self._fit_mode:
            self._fit_preview()
        else:
            self._update_preview_scaled()

    def _fit_preview(self):
        """Fit the image into the visible scroll viewport."""
        if self._preview_qimg is None:
            return
        vp = self.preview_scroll.viewport().size()
        if vp.width() <= 1 or vp.height() <= 1:
            return
        iw, ih = self._preview_qimg.width(), self._preview_qimg.height()
        if iw <= 0 or ih <= 0:
            return
        # compute scale to fit
        sx = vp.width()  / iw
        sy = vp.height() / ih
        self._preview_scale = max(0.05, min(sx, sy))
        self._fit_mode = True
        self._update_preview_scaled()

    def _zoom_reset_100(self):
        """Set zoom to 100% (1:1)."""
        self._fit_mode = False
        self._preview_scale = 1.0
        self._update_preview_scaled()

    def _zoom_by(self, factor: float):
        vp = self.preview_scroll.viewport()
        center = vp.rect().center()
        self._zoom_at(factor, center)

    def _zoom_at(self, factor: float, vp_pos):
        """Zoom keeping the image point under vp_pos (viewport coords) stationary."""
        if self._preview_qimg is None:
            return

        old_scale = float(self._preview_scale)

        # Content coords (in scaled-image pixels) currently under the mouse
        hsb = self.preview_scroll.horizontalScrollBar()
        vsb = self.preview_scroll.verticalScrollBar()
        cx = hsb.value() + int(vp_pos.x())
        cy = vsb.value() + int(vp_pos.y())

        # Convert to image-space coords (unscaled)
        ix = cx / old_scale
        iy = cy / old_scale

        # Apply zoom
        self._fit_mode = False
        new_scale = max(0.05, min(old_scale * float(factor), 8.0))
        self._preview_scale = new_scale

        # Rebuild pixmap/label size
        self._update_preview_scaled()

        # New content coords for same image-space point
        ncx = int(ix * new_scale)
        ncy = int(iy * new_scale)

        # Set scrollbars so that point stays under the mouse
        hsb.setValue(ncx - int(vp_pos.x()))
        vsb.setValue(ncy - int(vp_pos.y()))


    # --- MASK helpers ----------------------------------------------------
    def _active_mask_array(self) -> np.ndarray | None:
        """Return active mask as float32 [H,W] in 0..1, resized to doc image."""
        try:
            mid = getattr(self.doc, "active_mask_id", None)
            if not mid:
                return None
            layer = getattr(self.doc, "masks", {}).get(mid)
            if layer is None:
                return None

            m = np.asarray(getattr(layer, "data", None))
            if m is None or m.size == 0:
                return None

            # squeeze to 2D
            if m.ndim == 3 and m.shape[2] == 1:
                m = m[..., 0]
            elif m.ndim == 3:  # RGB/whatever → luminance
                m = (0.2126*m[...,0] + 0.7152*m[...,1] + 0.0722*m[...,2])

            orig = m
            # normalize if integer
            if orig.dtype.kind in "ui":
                m = orig.astype(np.float32) / float(np.iinfo(orig.dtype).max)
            else:
                m = orig.astype(np.float32, copy=False)

            m = np.clip(m, 0.0, 1.0)

            th, tw = self.doc.image.shape[:2]
            sh, sw = m.shape[:2]
            if (sh, sw) != (th, tw):
                yi = (np.linspace(0, sh-1, th)).astype(np.int32)
                xi = (np.linspace(0, sw-1, tw)).astype(np.int32)
                m = m[yi][:, xi]

            # honor opacity if present
            opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
            if opacity < 1.0:
                m *= opacity
            return m
        except Exception:
            return None

    def _blend_with_mask(self, base: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """base/out can be mono or 3ch; mask is [H,W] in 0..1."""
        if out.ndim == 3 and out.shape[2] == 3:
            m = mask[..., None]
        else:
            m = mask
        return base * (1.0 - m) + out * m


    def _run_stretch(self) -> np.ndarray | None:
        imgf = self._get_source_float()
        if imgf is None:
            return None
        blackpoint_sigma = float(self.sld_bp.value()) / 100.0
        hdr_on = bool(self.chk_hdr.isChecked())
        hdr_amount = float(self.sld_hdr_amt.value()) / 100.0
        hdr_knee = float(self.sld_hdr_knee.value()) / 100.0
        luma_only = bool(getattr(self, "chk_luma_only", None) and self.chk_luma_only.isChecked())
        luma_mode = str(self.cmb_luma.currentText()) if getattr(self, "cmb_luma", None) else "rec709"
        no_black_clip = bool(self.chk_no_black_clip.isChecked())
        luma_blend = float(self.sld_luma_blend.value()) / 100.0 if getattr(self, "sld_luma_blend", None) else 1.0

        target = float(self.spin_target.value())
        linked = bool(self.chk_linked.isChecked())
        normalize = bool(self.chk_normalize.isChecked())
        apply_curves = bool(self.chk_curves.isChecked())
        curves_boost = float(self.sld_curves.value()) / 100.0

        if imgf.ndim == 2 or (imgf.ndim == 3 and imgf.shape[2] == 1):
            out = stretch_mono_image(
                imgf.squeeze(),
                target_median=target,
                normalize=normalize,
                apply_curves=apply_curves,
                curves_boost=curves_boost,
                blackpoint_sigma=blackpoint_sigma,
                no_black_clip=no_black_clip,
                hdr_compress=hdr_on,
                hdr_amount=hdr_amount,
                hdr_knee=hdr_knee,
            )
        else:
            out = stretch_color_image(
                imgf,
                target_median=target,
                linked=linked,
                normalize=normalize,
                apply_curves=apply_curves,
                curves_boost=curves_boost,
                blackpoint_sigma=blackpoint_sigma,
                no_black_clip=no_black_clip,
                hdr_compress=hdr_on,
                hdr_amount=hdr_amount,
                hdr_knee=hdr_knee,
                luma_only=luma_only,
                luma_mode=luma_mode,
                luma_blend=luma_blend,   # <-- NEW
            )

        # ✅ If a mask is active, blend stretched result with original
        m = self._active_mask_array()
        if m is not None:
            base = imgf.astype(np.float32, copy=False)
            out = self._blend_with_mask(base, out, m)

        return out            


    def _set_preview_pixmap(self, arr: np.ndarray):
        vis = arr
        if vis is None or vis.size == 0:
            self.preview_label.clear()
            return

        # Ensure 3 channels for display
        if vis.ndim == 2:
            vis3 = np.stack([vis] * 3, axis=-1)
        elif vis.ndim == 3 and vis.shape[2] == 1:
            vis3 = np.repeat(vis, 3, axis=2)
        else:
            vis3 = vis

        # Convert to 8-bit RGB
        if vis3.dtype == np.uint8:
            buf8 = vis3
        elif vis3.dtype == np.uint16:
            buf8 = (vis3.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            buf8 = (np.clip(vis3, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Must be C-contiguous for QImage
        buf8 = np.ascontiguousarray(buf8)
        h, w, _ = buf8.shape
        bytes_per_line = buf8.strides[0]

        # Build QImage from raw pointer; keep references alive
        self._last_preview = buf8  # keep backing store alive
        ptr = sip.voidptr(self._last_preview.ctypes.data)
        qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self._preview_qimg = qimg
        self._apply_current_zoom() 

    # ----- active document change -----
    def _on_active_doc_changed(self, doc):
        """Called when user clicks a different image window."""
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._populate_initial_preview()
        try:
            self._schedule_clip_stats()
        except Exception:
            pass        

    # ----- slots -----
    def _populate_initial_preview(self):
        # show the current (unstretched) image as baseline
        src = self._get_source_float()
        if src is not None:
            self._set_preview_pixmap(np.clip(src, 0, 1))
        try:
            self.lbl_clipstats.setText(self.tr("Calculating clip stats…"))
        except Exception:
            pass
        try:
            self._schedule_clip_stats()
        except Exception:
            pass


    def _do_preview(self):
        self._start_stretch_job("preview")


    def _do_apply(self):
        self._start_stretch_job("apply")

    def _apply_out_to_doc(self, out: np.ndarray):
        # Preserve mono vs color shape
        if out.ndim == 3 and out.shape[2] == 3 and (self.doc.image.ndim == 2 or self.doc.image.shape[-1] == 1):
            out = out[..., 0]

        # --- Gather current UI state ------------------------------------
        target = float(self.spin_target.value())
        linked = bool(self.chk_linked.isChecked())
        normalize = bool(self.chk_normalize.isChecked())
        apply_curves = bool(getattr(self, "chk_curves", None) and self.chk_curves.isChecked())
        curves_boost = float(self.sld_curves.value()) / 100.0 if getattr(self, "sld_curves", None) is not None else 0.0
        blackpoint_sigma = float(self.sld_bp.value()) / 100.0
        hdr_on = bool(self.chk_hdr.isChecked())
        hdr_amount = float(self.sld_hdr_amt.value()) / 100.0
        hdr_knee = float(self.sld_hdr_knee.value()) / 100.0
        luma_only = bool(getattr(self, "chk_luma_only", None) and self.chk_luma_only.isChecked())
        luma_mode = str(self.cmb_luma.currentText()) if getattr(self, "cmb_luma", None) else "rec709"
        no_black_clip = bool(self.chk_no_black_clip.isChecked())
        luma_blend = float(self.sld_luma_blend.value()) / 100.0 if getattr(self, "sld_luma_blend", None) else 1.0

        parts = [f"target={target:.2f}", "linked" if linked else "unlinked"]
        if normalize:
            parts.append("norm")
        if apply_curves:
            parts.append(f"curves={curves_boost:.2f}")
        if self._active_mask_array() is not None:
            parts.append("masked")
        parts.append(f"bpσ={blackpoint_sigma:.2f}")
        if hdr_on and hdr_amount > 0:
            parts.append(f"hdr={hdr_amount:.2f}@{hdr_knee:.2f}")
        if luma_only:
            parts.append(f"luma={luma_mode}")
            parts.append(f"blend={luma_blend:.2f}")
        if no_black_clip:
            parts.append("no_black_clip")

        step_name = f"Statistical Stretch ({', '.join(parts)})"
        self.doc.apply_edit(out.astype(np.float32, copy=False), step_name=step_name)

        # Turn off display stretch on the active view, if any
        mw = self.parent()
        if hasattr(mw, "mdi") and mw.mdi.activeSubWindow():
            view = mw.mdi.activeSubWindow().widget()
            if getattr(view, "autostretch_enabled", False):
                view.set_autostretch(False)

            # Existing logging, now using the same values as above
            if hasattr(mw, "_log"):
                curves_on = apply_curves
                boost_val = curves_boost if curves_on else 0.0
                mw._log(
                    "Applied Statistical Stretch "
                    f"(target={target:.3f}, linked={linked}, normalize={normalize}, "
                    f"bp_sigma={blackpoint_sigma:.2f}, "
                    f"hdr={'ON' if hdr_on else 'OFF'}"
                    f"{', amt='+str(round(hdr_amount,2))+' knee='+str(round(hdr_knee,2)) if hdr_on else ''}, "
                    f"luma={'ON' if luma_only else 'OFF'}{', mode='+luma_mode if luma_only else ''}, "
                    f"curves={'ON' if curves_on else 'OFF'}"
                    f"{', boost='+str(round(boost_val,2)) if curves_on else ''}, "
                    f"mask={'ON' if self._active_mask_array() is not None else 'OFF'})"
                )

            # --- Build preset for headless replay ---------------------------
            # --- Build preset for headless replay ---------------------------
            preset = {
                "target_median": target,
                "linked": linked,
                "normalize": normalize,
                "apply_curves": apply_curves,
                "curves_boost": curves_boost,
                "blackpoint_sigma": blackpoint_sigma,
                "no_black_clip": no_black_clip,
                "hdr_compress": hdr_on,
                "hdr_amount": hdr_amount,
                "hdr_knee": hdr_knee,
                "luma_only": luma_only,
                "luma_mode": luma_mode,
                "luma_blend": luma_blend,
            }

            # ✅ Remember this as the last headless-style command
            #    (unless we are in a headless/suppressed call)
            suppress = bool(getattr(self, "_suppress_replay_record", False))
            if not suppress:
                from PyQt6.QtWidgets import QMainWindow
                try:
                    mw2 = self.parent()
                    while mw2 is not None and not isinstance(mw2, QMainWindow):
                        mw2 = mw2.parent()

                    if mw2 is not None and hasattr(mw2, "remember_last_headless_command"):
                        mw2.remember_last_headless_command(
                            command_id="stat_stretch",
                            preset=preset,
                            description="Statistical Stretch",
                        )
                        print(f"Remembered Statistical Stretch last headless command: {preset}")
                    else:
                        print("No main window with remember_last_headless_command; cannot store stat_stretch preset")
                except Exception as e:
                    print(f"Failed to remember Statistical Stretch last headless command: {e}")
            else:
                # optional debug
                print("Statistical Stretch: replay recording suppressed for this apply()")

        self.close()


    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
                    # Reset preview state for new document
                    self._last_preview = None
                    self._preview_qimg = None
        except Exception:
            pass

    @pyqtSlot(object, str)
    def _on_stretch_done(self, out, err: str):
        # dialog might be closing; guard
        if sip.isdeleted(self):
            return

        self._hide_busy()
        self._set_controls_enabled(True)
        self._job_running = False

        if err:
            QMessageBox.warning(self, "Stretch failed", err)
            return

        if out is None:
            QMessageBox.information(self, "No image", "No image is loaded in the active document.")
            return

        if getattr(self, "_job_mode", "") == "preview":
            self._set_preview_pixmap(out)
            return

        # apply mode: reuse your existing apply logic, but using `out` we already computed
        self._apply_out_to_doc(out)

        if getattr(self, "_pending_close", False):
            self._pending_close = False
            self.close()

    def closeEvent(self, ev):
        # If a job is running, DO NOT close (WA_DeleteOnClose would delete the QThread)
        if getattr(self, "_job_running", False):
            self._pending_close = True
            try:
                self._hide_busy()
            except Exception:
                pass
            try:
                self.hide()
            except Exception:
                pass
            ev.ignore()
            return

        # disconnect follow behavior
        try:
            if self._follow_conn and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass

        super().closeEvent(ev)


    def _update_preview_scaled(self):
        if self._preview_qimg is None:
            self.preview_label.clear()
            return
        sw = max(1, int(self._preview_qimg.width()  * self._preview_scale))
        sh = max(1, int(self._preview_qimg.height() * self._preview_scale))
        scaled = self._preview_qimg.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(QPixmap.fromImage(scaled))
        self.preview_label.resize(scaled.size())            # <- crucial for scrollbars

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit_mode:
            self._fit_preview()

    def eventFilter(self, obj, ev):
        # Ctrl+wheel zoom
        if ev.type() == QEvent.Type.Wheel and obj is self.preview_scroll.viewport():
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else (1/1.25)
                self._zoom_at(factor, ev.position())
                return True
            return False

        # Click+drag pan (left or middle mouse)
        if obj is self.preview_scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    self._panning = True
                    self._pan_last = ev.globalPosition().toPoint()
                    self.preview_scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

            elif ev.type() == QEvent.Type.MouseMove and self._panning:
                pos = ev.globalPosition().toPoint()
                delta = pos - self._pan_last
                self._pan_last = pos

                hsb = self.preview_scroll.horizontalScrollBar()
                vsb = self.preview_scroll.verticalScrollBar()
                hsb.setValue(hsb.value() - delta.x())
                vsb.setValue(vsb.value() - delta.y())
                return True

            elif ev.type() == QEvent.Type.MouseButtonRelease and self._panning:
                if ev.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                    self._panning = False
                    self._pan_last = None
                    self.preview_scroll.viewport().unsetCursor()
                    return True

        return super().eventFilter(obj, ev)
     