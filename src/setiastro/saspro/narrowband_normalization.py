# src/setiastro/saspro/narrowband_normalization.py
from __future__ import annotations

import os
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
import traceback
from PyQt6.QtCore import Qt, QSize, QEvent, QTimer, QPoint, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFileDialog, QInputDialog, QMessageBox, QCheckBox, QSizePolicy,
    QComboBox, QGroupBox, QFormLayout, QDoubleSpinBox, QSlider
)
from PyQt6.QtGui import QPixmap, QImage, QCursor, QIcon
from PyQt6 import sip
# legacy loader (same one DocManager uses)
from setiastro.saspro.legacy.image_manager import load_image as legacy_load_image

# your statistical stretch (mono + color) like SASv2 (for DISPLAY only)
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

from setiastro.saspro.linear_fit import linear_fit_mono_to_ref, _nanmedian

from setiastro.saspro.imageops.narrowband_normalization import normalize_narrowband, NBNParams

from setiastro.saspro.backgroundneutral import background_neutralize_rgb, auto_rect_50x50
from setiastro.saspro.widgets.image_utils import extract_mask_from_document as _active_mask_array_from_doc
from setiastro.saspro.help_support import make_help_button

# Real SCNR core (green / colour channel suppression)
from setiastro.saspro.remove_green import _apply_scnr_rgb


# Flip to True to log the params going into the engine and a fingerprint of
# the array coming back. Useful when a control appears to do nothing.
_NBN_DEBUG = False


@dataclass
class _NBNJob:
    ha: np.ndarray | None
    oiii: np.ndarray | None
    sii: np.ndarray | None
    params: NBNParams
    step_name: str

class _NBNWorker(QThread):
    progress = pyqtSignal(int, str)
    failed = pyqtSignal(str)
    done = pyqtSignal(object, str)  # (np.ndarray, step_name)

    def __init__(self, job: _NBNJob):
        super().__init__()
        self.job = job

    def run(self):
        try:
            def cb(pct: int, msg: str = ""):
                self.progress.emit(int(pct), str(msg))

            out = normalize_narrowband(
                self.job.ha, self.job.oiii, self.job.sii,
                self.job.params,
                progress_cb=cb,
            )
            self.progress.emit(99, "Rendering Preview...")
            self.done.emit(out, self.job.step_name)

        except Exception:
            self.failed.emit(traceback.format_exc())

class NarrowbandNormalization(QWidget):
    def __init__(self, doc_manager=None, parent=None):
        super().__init__(parent)

        # Force top-level floating window behavior even if parent is main window
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass    

        self.doc_manager = doc_manager
        self.setWindowTitle("Narrowband Normalization")

        # raw channels (float32 [0..1])
        self.ha: np.ndarray | None = None
        self.oiii: np.ndarray | None = None
        self.sii: np.ndarray | None = None
        self.osc1: np.ndarray | None = None   # (Ha/OIII)
        self.osc2: np.ndarray | None = None   # (SII/OIII)

        # Track the source document so Apply writes back to the correct
        # image even when channels were split from an OSC (colour) view.
        self._source_doc = None

        self._dim_mismatch_accepted = False

        # result
        self.final: np.ndarray | None = None  # RGB float32 [0..1]
        self._base_pm: QPixmap | None = None

        # preview state
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 6.0
        self._panning = False
        self._pan_last: QPoint | None = None

        # debounce
        self._debounce = QTimer(self)
        self._debounce.setInterval(250)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._kick_preview_compute)
        # async compute control
        self._calc_seq = 0          # increments on every requested recompute
        self._active_seq = 0        # seq of currently running worker (optional)
        self._worker = None         # current worker ref
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        # Create all widgets FIRST (fixes btn_ha missing, etc.)
        self._init_widgets()
        self._connect_signals()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        root = QHBoxLayout()
        root.setSpacing(10)
        outer.addLayout(root, 1)

        # ---------------- LEFT PANEL ----------------
        # Deliberately NOT in a scroll area. The columns are a fixed set of
        # controls; wrapping them meant the scrollbars ate width from the
        # columns rather than the panel just being as wide as it needs to be.
        left_host = QWidget(self)

        left_row = QHBoxLayout(left_host)
        left_row.setContentsMargins(0, 0, 0, 0)
        left_row.setSpacing(10)

        colA = QVBoxLayout(); colA.setSpacing(8)
        colB = QVBoxLayout(); colB.setSpacing(8)

        # Column A: loaders
        colA.addWidget(self.grp_import)

        colA.addWidget(QLabel("<b>Load channels</b>"))

        self.grp_nb = QGroupBox("Narrowband channels", self)
        nbv = QVBoxLayout(self.grp_nb); nbv.setSpacing(4)
        for btn, lab in (
            (self.btn_ha, self.lbl_ha),
            (self.btn_oiii, self.lbl_oiii),
            (self.btn_sii, self.lbl_sii),
        ):
            nbv.addWidget(btn)
            nbv.addWidget(lab)

        self.grp_osc = QGroupBox("OSC extractions", self)
        oscv = QVBoxLayout(self.grp_osc); oscv.setSpacing(4)
        for btn, lab in (
            (self.btn_osc1, self.lbl_osc1),
            (self.btn_osc2, self.lbl_osc2),
        ):
            oscv.addWidget(btn)
            oscv.addWidget(lab)

        colA.addWidget(self.grp_nb)
        colA.addWidget(self.grp_osc)

        # extras sections referenced by _refresh_visibility
        colA.addWidget(self.grp_hoo_extras)
        colA.addWidget(self.grp_sho_extras)
        colA.addStretch(1)

        # Column B: normalization + actions
        colB.addWidget(self.grp_norm)

        actions = QGroupBox("Actions", self)
        actv = QVBoxLayout(actions); actv.setSpacing(6)
        for b in (self.btn_clear, self.btn_preview, self.btn_apply, self.btn_push):
            b.setMinimumHeight(28)
            actv.addWidget(b)
        colB.addWidget(actions)
        colB.addStretch(1)

        left_row.addLayout(colA, 1)
        left_row.addLayout(colB, 1)

        # Minimum horizontally so the columns can't be squeezed below their
        # content; the preview takes all remaining width.
        left_host.setSizePolicy(QSizePolicy.Policy.Minimum,
                                QSizePolicy.Policy.Preferred)
        root.addWidget(left_host, 0)

        # ---------------- RIGHT PANEL (Preview) ----------------
        right = QVBoxLayout()
        right.setSpacing(8)

        tools = QHBoxLayout()
        tools.setSpacing(6)

        self.btn_zoom_in = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_fit = themed_toolbtn("zoom-fit-best", "Fit to Preview")

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_at(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_at(0.8))
        self.btn_fit.clicked.connect(self._fit_to_preview)

        tools.addStretch(1)
        tools.addWidget(self.btn_zoom_out)
        tools.addWidget(self.btn_zoom_in)
        tools.addWidget(self.btn_fit)
        tools.addStretch(1)
        self.btn_doc_help = make_help_button("nb_normalization", self)
        tools.addWidget(self.btn_doc_help)
        right.addLayout(tools)

        self.scroll = QScrollArea(self)
        # MUST be False: widgetResizable(True) re-sizes the label to the
        # viewport on every layout pass, overriding the explicit
        # preview.resize(scaled.size()) in _update_preview_pixmap. The label
        # then never exceeds the viewport, so there is nothing to scroll and
        # zoom appears to crop instead of pan.
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 240)
        self.scroll.setWidget(self.preview)

        self.preview.setMouseTracking(True)
        self.preview.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)
        self.scroll.installEventFilter(self)
        self.scroll.horizontalScrollBar().installEventFilter(self)
        self.scroll.verticalScrollBar().installEventFilter(self)

        right.addWidget(self.scroll, 1)

        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#888;")
        right.addWidget(self.status, 0)

        right_host = QWidget(self)
        right_host.setLayout(right)
        root.addWidget(right_host, 1)

        # ---------------- FOOTER ----------------
        self.lbl_credits = QLabel(
            """
            <div style="text-align:center;">
            <span style="font-size:12px; color:#b8b8b8;">
                PixelMath narrowband normalization concept &amp; SHO/HOS/HSO/HOO formulas by
                <b>Bill Blanshan</b> and <b>Mike Cranfield</b><br>
                <a style="color:#9ecbff;" href="https://www.youtube.com/@anotherastrochannel2173">Bill Blanshan (YouTube)</a>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <a style="color:#9ecbff;" href="https://cosmicphotons.com/">Mike Cranfield (cosmicphotons.com)</a>
            </span>
            </div>
            """
        )
        self.lbl_credits.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_credits.setOpenExternalLinks(True)
        self.lbl_credits.setWordWrap(True)
        self.lbl_credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_credits.setStyleSheet("margin-top:6px; padding:6px 8px;")

        # Key: don't let it be clipped—allow it to take minimum height
        self.lbl_credits.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        # Wrap footer so it can scroll if the window is too short
        outer.addWidget(self.lbl_credits, 0)

        self.setLayout(outer)
        self.setMinimumSize(1080, 720)

        # Initial state
        self._refresh_visibility()


    def _init_widgets(self):
        # -------- Import mapped RGB (Perfect Palette / existing composites) --------
        self.grp_import = QGroupBox("Import mapped RGB view", self)
        impv = QVBoxLayout(self.grp_import)
        impv.setSpacing(6)

        # Use themed buttons for consistency
        self.btn_imp_sho = QPushButton("Load SHO View…", self)
        self.btn_imp_hso = QPushButton("Load HSO View…", self)
        self.btn_imp_hos = QPushButton("Load HOS View…", self)
        self.btn_imp_hoo = QPushButton("Load HOO View…", self)

        for b in (self.btn_imp_sho, self.btn_imp_hso, self.btn_imp_hos, self.btn_imp_hoo):
            b.setMinimumHeight(28)
            impv.addWidget(b)

        # -------- Channel load buttons + labels --------
        self.btn_ha = QPushButton("Load Ha…", self)
        self.btn_oiii = QPushButton("Load OIII…", self)
        self.btn_sii = QPushButton("Load SII…", self)
        self.btn_osc1 = QPushButton("Load OSC1 (Ha/OIII)…", self)
        self.btn_osc2 = QPushButton("Load OSC2 (SII/OIII)…", self)

        self.lbl_ha = QLabel("No Ha loaded.", self)
        self.lbl_oiii = QLabel("No OIII loaded.", self)
        self.lbl_sii = QLabel("No SII loaded.", self)
        self.lbl_osc1 = QLabel("No OSC1 loaded.", self)
        self.lbl_osc2 = QLabel("No OSC2 loaded.", self)

        # -------- Actions --------
        self.btn_clear = QPushButton("Clear", self)
        self.btn_preview = QPushButton("Preview", self)
        self.btn_apply = QPushButton("Apply to Current View", self)
        self.btn_push = QPushButton("Push as New View", self)

        # -------- Preview options --------
        self.chk_preview_autostretch = QCheckBox("Autostretch preview", self)
        self.chk_preview_autostretch.setChecked(False)

        # -------- Normalization controls (built in helper) --------
        self.grp_norm, self._norm_form = self._build_norm_group()

        # -------- Extras groups referenced by _refresh_visibility --------
        self.grp_hoo_extras = QGroupBox("HOO Extras", self)
        self.grp_hoo_extras.setLayout(QVBoxLayout())
        self.grp_hoo_extras.layout().addWidget(QLabel("Reserved for future HOO-specific options.", self))

        self.grp_sho_extras = QGroupBox("SHO / HSO / HOS Extras", self)
        self.grp_sho_extras.setLayout(QVBoxLayout())
        self.grp_sho_extras.layout().addWidget(QLabel("Reserved for future SHO-family options.", self))

        for lab in (self.lbl_ha, self.lbl_oiii, self.lbl_sii, self.lbl_osc1, self.lbl_osc2):
            lab.setWordWrap(False)
            lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def _ui_alive(self) -> bool:
        if sip is None:
            return True
        try:
            return (not sip.isdeleted(self)) and (not sip.isdeleted(self.status))
        except Exception:
            return False

    def _safe_set_status(self, txt: str):
        try:
            if self._ui_alive():
                self.status.setText(str(txt))
        except RuntimeError:
            # QLabel already deleted
            pass

    def _build_norm_group(self) -> tuple[QGroupBox, QFormLayout]:
        grp = QGroupBox("Normalization", self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        # ------------------------------------------------------------------
        # Engine selector
        #
        # V1 and V2 are entirely separate algorithms with disjoint parameter
        # sets. normalize_narrowband() dispatches on params.use_v2 and the
        # V2 path reads NONE of the V1 fields, so the two control blocks
        # below are mutually exclusive — see _refresh_visibility().
        # ------------------------------------------------------------------
        self.cmb_engine = QComboBox(self)
        self.cmb_engine.addItems([
            "V2 — v2.23",
            "V1 — legacy PixelMath",
        ])
        self.cmb_engine.setToolTip(
            "V2 is NarrowbandNormalizationV2 (v2.23).\n"
            "V1 is the original PixelMath port.\n\n"
            "Each engine has its own parameters; switching changes which\n"
            "controls below are available."
        )

        # Scenario / mode / lightness
        self.cmb_scenario = QComboBox(self)
        self.cmb_scenario.addItems(["SHO", "HSO", "HOS", "HOO"])

        self.cmb_mode = QComboBox(self)
        self.cmb_mode.addItems(["Non-linear (Mode=1)", "Linear (Mode=0)"])

        self.cmb_lightness = QComboBox(self)
        self.cmb_lightness.addItems(["Off (0)", "Original (1)", "Ha (2)", "SII (3)", "OIII (4)"])

        # ==================== V1 (legacy) controls ====================
        # Blackpoint: [0..1] step 0.01
        self.row_blackpoint, self.spin_blackpoint, self.sld_blackpoint = self._slider_spin_row(
            lo=0.0, hi=1.0, step=0.01, val=0.25, decimals=3
        )

        # HL Recover / Reduction: [0.5..2.0] default 1.0
        self.row_hlrecover, self.spin_hlrecover, self.sld_hlrecover = self._slider_spin_row(
            lo=0.5, hi=2.0, step=0.01, val=1.0, decimals=3
        )
        self.row_hlreduct, self.spin_hlreduct, self.sld_hlreduct = self._slider_spin_row(
            lo=0.5, hi=2.0, step=0.01, val=1.0, decimals=3
        )

        # Brightness: [0.5..2.0] default 1.0
        self.row_brightness, self.spin_brightness, self.sld_brightness = self._slider_spin_row(
            lo=0.5, hi=2.0, step=0.01, val=1.0, decimals=3
        )

        # Ha Blend (HOO only)
        self.row_hablend, self.spin_hablend, self.sld_hablend = self._slider_spin_row(
            lo=0.0, hi=1.0, step=0.01, val=0.6, decimals=3
        )

        # Boosts: [0.5..2.0] default 1.0
        self.row_oiiiboost, self.spin_oiiiboost, self.sld_oiiiboost = self._slider_spin_row(
            lo=0.5, hi=2.0, step=0.01, val=1.0, decimals=3
        )
        self.row_siiboost, self.spin_siiboost, self.sld_siiboost = self._slider_spin_row(
            lo=0.5, hi=2.0, step=0.01, val=1.0, decimals=3
        )
        self.row_oiii_sho, self.spin_oiiiboost2, self.sld_oiiiboost2 = self._slider_spin_row(
            lo=0.5, hi=2.0, step=0.01, val=1.0, decimals=3
        )

        # Blend Mode (V1, HOO only)
        self.cmb_blendmode = QComboBox(self)
        self.cmb_blendmode.addItems(["Screen", "Add", "Linear Dodge", "Normal"])

        # ==================== V2 controls ====================
        # Background equalization — maps to NBNParams.bgn. This is the
        # engine's own per-channel pedestal equalization, NOT the same thing
        # as the gradient-descent background neutralization post-step below.
        self.chk_v2_bgn = QCheckBox("Background equalization (engine)", self)
        self.chk_v2_bgn.setChecked(True)
        self.chk_v2_bgn.setToolTip(
            "Equalizes each channel's sky pedestal to their common mean\n"
            "before normalization (Bill's BGN step). Distinct from the\n"
            "∇-descent Background Neutralization post-step further down."
        )

        # Background noise: MAD units above the measured sky level
        self.row_v2_bgnoise, self.spin_v2_bgnoise, self.sld_v2_bgnoise = self._slider_spin_row(
            lo=0.0, hi=5.0, step=0.05, val=1.0, decimals=2
        )

        # Per-channel midtone boosts. 0.5 is neutral; the engine clips to
        # [0.001, 0.999] and uses (1 - value) as the MTF midpoint.
        self.row_v2_red, self.spin_v2_red, self.sld_v2_red = self._slider_spin_row(
            lo=0.0, hi=1.0, step=0.01, val=0.5, decimals=3
        )
        self.row_v2_green, self.spin_v2_green, self.sld_v2_green = self._slider_spin_row(
            lo=0.0, hi=1.0, step=0.01, val=0.5, decimals=3
        )
        self.row_v2_blue, self.spin_v2_blue, self.sld_v2_blue = self._slider_spin_row(
            lo=0.0, hi=1.0, step=0.01, val=0.5, decimals=3
        )

        self.cmb_v2_lumhold = QComboBox(self)
        self.cmb_v2_lumhold.addItems(["Off", "Preserve", "Red", "Green", "Blue"])
        self.cmb_v2_lumhold.setToolTip(
            "Rescale RGB so Rec.709 luma matches a chosen source.\n"
            "'Preserve' holds the pre-normalization luma; the channel\n"
            "options drive luma from that single normalized channel."
        )

        # OSC HOO synthetic green (HOO only)
        self.chk_v2_osc = QCheckBox("OSC HOO (synthetic green)", self)
        self.chk_v2_osc.setChecked(False)
        self.chk_v2_osc.setToolTip(
            "For one-shot-colour HOO data: build the green channel by\n"
            "blending rather than reusing OIII directly."
        )

        self.cmb_v2_blendmode = QComboBox(self)
        self.cmb_v2_blendmode.addItems(["Mode 1", "Mode 2", "Mode 3"])

        self.row_v2_blendamt, self.spin_v2_blendamt, self.sld_v2_blendamt = self._slider_spin_row(
            lo=0.0, hi=1.0, step=0.01, val=0.5, decimals=3
        )

        self.chk_v2_showbg = QCheckBox("Show background map (diagnostic)", self)
        self.chk_v2_showbg.setChecked(False)
        self.chk_v2_showbg.setToolTip(
            "Paints everything at or below the computed sky level white.\n"
            "Use it to check the Background Noise setting, then turn it off.\n"
            "SCNR and the BN post-step are skipped while this is on."
        )

        # ==================== shared post-steps ====================
        self.chk_scnr = QCheckBox("SCNR (reduce green cast)", self)
        self.chk_scnr.setChecked(True)

        self.row_scnr_amount, self.spin_scnr_amount, self.sld_scnr_amount = self._slider_spin_row(
            lo=0.0, hi=1.0, step=0.01, val=0.80, decimals=2
        )
        self.chk_scnr_preserve = QCheckBox("Preserve luminance", self)
        self.chk_scnr_preserve.setChecked(True)

        self.chk_linear_fit = QCheckBox("Linear Fit (highest signal)", self)
        self.chk_linear_fit.setChecked(False)

        self.chk_bg_neutral = QCheckBox("Background Neutralization (∇-descent)", self)
        self.chk_bg_neutral.setChecked(True)
        self.chk_bg_neutral.setToolTip(
            "Post-process step applied to the finished RGB result.\n"
            "With the V2 engine this runs IN ADDITION to the engine's own\n"
            "background equalization — you usually want one or the other."
        )

        # ---------------- Layout ----------------
        form.addRow("Engine:", self.cmb_engine)
        form.addRow("Scenario:", self.cmb_scenario)
        form.addRow("Mode:", self.cmb_mode)
        form.addRow("Lightness:", self.cmb_lightness)

        # V1 labels
        self._lbl_blackpoint = QLabel("Blackpoint\n(Min → Med):", self)
        self._lbl_blackpoint.setWordWrap(True)
        self._lbl_blackpoint.setToolTip(
            "Controls the blackpoint reference M used by the normalization.\n\n"
            "M = min + Blackpoint × (median − min)\n"
            "• 0.0 = use Min\n"
            "• 1.0 = use Median\n\n"
            "Higher values lift the baseline (brighter background); lower values preserve darker blacks."
        )
        self._lbl_hlrecover  = QLabel("HL Recover:", self)
        self._lbl_hlreduct   = QLabel("HL Reduction:", self)
        self._lbl_brightness = QLabel("Brightness:", self)
        self._lbl_blendmode  = QLabel("Blend Mode:", self)
        self._lbl_hablend    = QLabel("Ha Blend:", self)
        self._lbl_oiiiboost  = QLabel("OIII Boost:", self)
        self._lbl_siiboost   = QLabel("SII Boost:", self)
        self._lbl_oiiiboost2 = QLabel("OIII Boost:", self)

        form.addRow(self._lbl_blackpoint, self.row_blackpoint)
        form.addRow(self._lbl_hlrecover,  self.row_hlrecover)
        form.addRow(self._lbl_hlreduct,   self.row_hlreduct)
        form.addRow(self._lbl_brightness, self.row_brightness)

        form.addRow(self._lbl_blendmode,  self.cmb_blendmode)
        form.addRow(self._lbl_hablend,    self.row_hablend)
        form.addRow(self._lbl_oiiiboost,  self.row_oiiiboost)
        form.addRow(self._lbl_siiboost,   self.row_siiboost)
        form.addRow(self._lbl_oiiiboost2, self.row_oiii_sho)

        # V2 labels + rows
        self._lbl_v2_bgnoise  = QLabel("Background Noise:", self)
        self._lbl_v2_bgnoise.setToolTip(
            "Sky pedestal height, in MAD units above the measured background.\n"
            "Raise it if the background still shows colour mottling."
        )
        self._lbl_v2_red      = QLabel("Red Boost:", self)
        self._lbl_v2_green    = QLabel("Green Boost:", self)
        self._lbl_v2_blue     = QLabel("Blue Boost:", self)
        self._lbl_v2_lumhold  = QLabel("Luminance Hold:", self)
        self._lbl_v2_blendmode = QLabel("OSC Blend Mode:", self)
        self._lbl_v2_blendamt = QLabel("OSC Blend Amount:", self)

        form.addRow("", self.chk_v2_bgn)
        form.addRow(self._lbl_v2_bgnoise,  self.row_v2_bgnoise)
        form.addRow(self._lbl_v2_red,      self.row_v2_red)
        form.addRow(self._lbl_v2_green,    self.row_v2_green)
        form.addRow(self._lbl_v2_blue,     self.row_v2_blue)
        form.addRow(self._lbl_v2_lumhold,  self.cmb_v2_lumhold)
        form.addRow("", self.chk_v2_osc)
        form.addRow(self._lbl_v2_blendmode, self.cmb_v2_blendmode)
        form.addRow(self._lbl_v2_blendamt,  self.row_v2_blendamt)
        form.addRow("", self.chk_v2_showbg)

        # Shared post-steps
        form.addRow("", self.chk_scnr)
        self._lbl_scnr_amount = QLabel("SCNR Amount:", self)
        form.addRow(self._lbl_scnr_amount, self.row_scnr_amount)
        form.addRow("", self.chk_scnr_preserve)
        form.addRow("", self.chk_linear_fit)
        form.addRow("", self.chk_bg_neutral)
        form.addRow("", self.chk_preview_autostretch)

        grp.setLayout(form)
        return grp, form

    def _use_v2(self) -> bool:
        return self.cmb_engine.currentIndex() == 0

    def _connect_signals(self):
        # Loaders
        self.btn_imp_sho.clicked.connect(lambda: self._import_mapped_view("SHO"))
        self.btn_imp_hso.clicked.connect(lambda: self._import_mapped_view("HSO"))
        self.btn_imp_hos.clicked.connect(lambda: self._import_mapped_view("HOS"))
        self.btn_imp_hoo.clicked.connect(lambda: self._import_mapped_view("HOO"))        
        self.btn_ha.clicked.connect(lambda: self._load_channel("Ha"))
        self.btn_oiii.clicked.connect(lambda: self._load_channel("OIII"))
        self.btn_sii.clicked.connect(lambda: self._load_channel("SII"))
        self.btn_osc1.clicked.connect(lambda: self._load_channel("OSC1"))
        self.btn_osc2.clicked.connect(lambda: self._load_channel("OSC2"))

        # Actions
        self.btn_clear.clicked.connect(self._clear_channels)
        self.btn_preview.clicked.connect(self._schedule_preview)  # debounced compute
        self.btn_apply.clicked.connect(self._apply_to_current_view)
        self.btn_push.clicked.connect(self._push_result)

        # Structural changes -> re-evaluate which controls apply, then recompute
        self.cmb_engine.currentIndexChanged.connect(self._refresh_visibility)
        self.cmb_scenario.currentIndexChanged.connect(self._refresh_visibility)
        self.cmb_mode.currentIndexChanged.connect(self._refresh_visibility)
        self.cmb_lightness.currentIndexChanged.connect(self._schedule_preview)
        self.chk_bg_neutral.toggled.connect(self._schedule_preview)

        # V1 value controls
        for w in (
            self.spin_blackpoint, self.spin_hlrecover, self.spin_hlreduct, self.spin_brightness,
            self.cmb_blendmode, self.spin_hablend, self.spin_oiiiboost, self.spin_siiboost,
            self.spin_oiiiboost2
        ):
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._schedule_preview)
            if hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self._schedule_preview)

        for s in (
            self.sld_blackpoint, self.sld_hlrecover, self.sld_hlreduct, self.sld_brightness,
            self.sld_hablend, self.sld_oiiiboost, self.sld_siiboost, self.sld_oiiiboost2
        ):
            s.valueChanged.connect(self._schedule_preview)

        # V2 value controls
        for w in (
            self.spin_v2_bgnoise, self.spin_v2_red, self.spin_v2_green,
            self.spin_v2_blue, self.spin_v2_blendamt,
        ):
            w.valueChanged.connect(self._schedule_preview)

        for s in (
            self.sld_v2_bgnoise, self.sld_v2_red, self.sld_v2_green,
            self.sld_v2_blue, self.sld_v2_blendamt,
        ):
            s.valueChanged.connect(self._schedule_preview)

        self.cmb_v2_lumhold.currentIndexChanged.connect(self._schedule_preview)
        self.cmb_v2_blendmode.currentIndexChanged.connect(self._schedule_preview)
        self.chk_v2_bgn.toggled.connect(self._schedule_preview)
        # These two change which other rows make sense, so they re-run visibility.
        self.chk_v2_osc.toggled.connect(self._refresh_visibility)
        self.chk_v2_showbg.toggled.connect(self._refresh_visibility)

        # Shared post-steps
        self.chk_scnr.toggled.connect(self._on_scnr_toggled)
        self.spin_scnr_amount.valueChanged.connect(self._schedule_preview)
        self.sld_scnr_amount.valueChanged.connect(self._schedule_preview)
        self.chk_scnr_preserve.toggled.connect(self._schedule_preview)
        self.chk_linear_fit.toggled.connect(self._schedule_preview)
        self.chk_preview_autostretch.toggled.connect(self._schedule_preview)

    def _slider_spin_row(self, lo: float, hi: float, step: float, val: float, decimals: int):
        """
        Returns (row_widget, spinbox, slider).
        Slider is int-mapped: int_value = round(x / step)
        """
        w = QWidget(self)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        sp = QDoubleSpinBox(self)
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setValue(val)
        # Fixed, compact numeric field: a form row should never demand more
        # width than the column can give it.
        sp.setFixedWidth(84)

        s = QSlider(Qt.Orientation.Horizontal, self)
        s.setMinimumWidth(80)
        s.setTracking(False)  # don't spam recompute while dragging; fires on release
        imin = int(round(lo / step))
        imax = int(round(hi / step))
        s.setRange(imin, imax)
        s.setValue(int(round(val / step)))

        # sync both ways (block signals to avoid loops)
        def slider_to_spin(iv: int):
            sp.blockSignals(True)
            sp.setValue(iv * step)
            sp.blockSignals(False)

        def spin_to_slider(v: float):
            s.blockSignals(True)
            s.setValue(int(round(v / step)))
            s.blockSignals(False)

        s.valueChanged.connect(slider_to_spin)
        sp.valueChanged.connect(spin_to_slider)

        lay.addWidget(s, 1)
        lay.addWidget(sp, 0)

        return w, sp, s


    def _schedule_preview(self):
        """Call this on ANY UI change that should recompute preview."""
        self._calc_seq += 1
        self.status.setText("Updating preview...")
        self._debounce.start()

    def _dbg(self, msg: str):
        # Change this to logging if you prefer
        print(f"[NBN] {msg}")
        self.status.setText(msg)

    def _preview_scale(self) -> float:
        """
        Choose a downsample factor so preview compute stays fast.
        Target: keep preview processing under ~2 MP and cap max dimension.
        """
        # pick a reference shape from whatever is loaded
        ha, oo, si = self._prepared_channels()
        ref = ha if ha is not None else (oo if oo is not None else si)
        if ref is None:
            return 1.0

        h, w = ref.shape[:2]
        mp = (w * h) / 1e6

        # Hard caps (tweak to taste)
        max_dim = 1800          # keep longest side ~<= 1800px
        target_mp = 2.0         # keep total pixels ~<= 2MP

        s_dim = min(1.0, max_dim / float(max(h, w)))
        s_mp = min(1.0, (target_mp / max(mp, 1e-6)) ** 0.5)

        s = min(s_dim, s_mp)

        # Don't micro-scale; prefer a few stable buckets
        if s >= 0.90: return 1.0
        if s >= 0.65: return 0.75
        if s >= 0.45: return 0.50
        if s >= 0.30: return 0.33
        return 0.25


    def _downsample_mono(self, ch: np.ndarray | None, s: float) -> np.ndarray | None:
        if ch is None or s >= 0.999:
            return ch
        h, w = ch.shape[:2]
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        return cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)


    def _downsample_rgb(self, img: np.ndarray | None, s: float) -> np.ndarray | None:
        if img is None or s >= 0.999:
            return img
        h, w = img.shape[:2]
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _kick_preview_compute(self):
        # If nothing loaded, don't compute
        try:
            ha, oo, si = self._prepared_channels()
            if ha is None and oo is None and si is None:
                self.status.setText("Load channels to preview.")
                return
        except Exception as e:
            print(f"[NBN] _prepared_channels failed:\n{traceback.format_exc()}")
            self._safe_set_status(f"Preview error: {e}")
            return

        def on_done(out: np.ndarray, step_name: str):
            out2 = self._maybe_apply_scnr(out)
            out2 = self._maybe_background_neutralize_rgb(out2, doc_for_mask=None)
            self.final = out2

            disp = out2
            if self.chk_preview_autostretch.isChecked():
                disp = np.clip(stretch_color_image(disp, target_median=0.25, linked=True), 0.0, 1.0)

            qimg = self._to_qimage(disp)
            first = (self._base_pm is None)
            self._set_preview_image(qimg, fit=first, preserve_view=True)
            self.status.setText("Done (100%)")

        def on_fail(err: str):
            # Don't spam modal dialogs for "missing channels" type errors.
            # Test only the LAST line: err is a full traceback, and the file
            # paths in its frames would trip a substring match on the whole blob
            # (e.g. any frame in image_manager.load_image).
            last = ""
            for line in reversed((err or "").strip().splitlines()):
                if line.strip():
                    last = line.strip()
                    break
            low = last.lower()
            if "requires" in low or low.startswith("load") or "load " in low:
                self._safe_set_status(last)
                return
            print(f"[NBN] preview failed:\n{err}")
            QMessageBox.critical(self, "Narrowband Normalization", err)
            self._safe_set_status("Preview failed.")

        self._start_job(
            downsample=True,
            step_name="NBN Preview",
            on_done=on_done,
            on_fail=on_fail,
        )

    def _maybe_apply_scnr(self, rgb: np.ndarray) -> np.ndarray:
        """
        Apply full SCNR green suppression as a post-step if enabled.
        Uses the real SCNR core from remove_green with the user's amount
        and preserve-luminance settings.
        """
        # The background map is a binary diagnostic; post-processing it would
        # make it unreadable.
        if self._use_v2() and self.chk_v2_showbg.isChecked():
            return rgb
        if not self.chk_scnr.isChecked():
            return rgb
        # V1 does SCNR natively, before the Lab lightness replacement. Running
        # this post-step on top of it would leave no excess for the amount
        # slider to act on — which is exactly the all-or-nothing behavior.
        if not self._use_v2():
            return rgb
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            return rgb

        amount = float(self.spin_scnr_amount.value())
        preserve = bool(self.chk_scnr_preserve.isChecked())

        if amount <= 0.0:
            return rgb

        out = _apply_scnr_rgb(
            rgb, amount, mode="avg",
            preserve_lightness=preserve, channel="G",
        )
        if _NBN_DEBUG:
            d = np.abs(np.asarray(out, np.float64) - np.asarray(rgb, np.float64))
            print(f"[NBN] scnr: engine={'V2' if self._use_v2() else 'V1'} "
                  f"amount={amount} maxdelta={d.max():.6f} meandelta={d.mean():.8f}")
        return out

    def _maybe_background_neutralize_rgb(self, rgb: np.ndarray, *, doc_for_mask=None) -> np.ndarray:
        """
        Apply BN to an RGB float image in [0,1] if the checkbox is enabled.
        If doc_for_mask is provided, blend result using destination active mask (headless behavior).
        """
        if self._use_v2() and self.chk_v2_showbg.isChecked():
            return rgb
        if not getattr(self, "chk_bg_neutral", None) or not self.chk_bg_neutral.isChecked():
            return rgb

        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            return rgb

        # auto rect + neutralize (same logic as headless BN default)
        rect = auto_rect_50x50(rgb)
        out = background_neutralize_rgb(rgb.astype(np.float32, copy=False), rect)

        # destination active-mask blend (same as apply_background_neutral_to_doc)
        if doc_for_mask is not None:
            m = _active_mask_array_from_doc(doc_for_mask)
            if m is not None:
                m3 = np.repeat(m[..., None], 3, axis=2).astype(np.float32, copy=False)
                base_for_blend = rgb.astype(np.float32, copy=False)
                out = base_for_blend * (1.0 - m3) + out * m3

        return out.astype(np.float32, copy=False)

    def _requirements_met(self, ha, oo, si) -> tuple[bool, str]:
        scen = self._scenario()
        if scen == "HOO":
            if ha is None or oo is None:
                return False, "Load Ha + OIII to preview HOO."
            return True, ""
        else:
            missing = []
            if ha is None: missing.append("Ha")
            if oo is None: missing.append("OIII")
            if si is None: missing.append("SII")
            if missing:
                return False, f"Load {', '.join(missing)} to preview {scen}."
            return True, ""


    def _form_set_row_visible(self, form: QFormLayout, row: int, visible: bool):
        """Hide/show both label and field for a QFormLayout row."""
        label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)

        for it in (label_item, field_item):
            if it is None:
                continue
            w = it.widget()
            if w is not None:
                w.setVisible(visible)
            else:
                # sometimes the field is a layout
                lay = it.layout()
                if lay is not None:
                    for i in range(lay.count()):
                        ww = lay.itemAt(i).widget()
                        if ww is not None:
                            ww.setVisible(visible)

    def _form_find_row(self, form: QFormLayout, field_widget: QWidget) -> int:
        """Return row index where field_widget is the FieldRole."""
        for r in range(form.rowCount()):
            it = form.itemAt(r, QFormLayout.ItemRole.FieldRole)
            if it and it.widget() is field_widget:
                return r
        return -1

    def _scenario(self) -> str:
        return self.cmb_scenario.currentText().split()[0].upper()

    def _mode_value(self) -> int:
        # your combo is ["Non-linear (Mode=1)", "Linear (Mode=0)"]
        return 1 if self.cmb_mode.currentIndex() == 0 else 0

    def _set_lightness_items(self, items: list[str]):
        self.cmb_lightness.blockSignals(True)
        cur = self.cmb_lightness.currentText()
        self.cmb_lightness.clear()
        self.cmb_lightness.addItems(items)
        # try to preserve selection if possible
        idx = self.cmb_lightness.findText(cur)
        if idx >= 0:
            self.cmb_lightness.setCurrentIndex(idx)
        self.cmb_lightness.blockSignals(False)

    def _refresh_visibility(self, *_):
        scen = self._scenario()
        mode = self._mode_value()
        v2 = self._use_v2()

        is_hoo = (scen == "HOO")
        self.btn_sii.setVisible(not is_hoo)
        self.lbl_sii.setVisible(not is_hoo)

        def show_row(field_widget, visible: bool):
            r = self._form_find_row(self._norm_form, field_widget)
            if r >= 0:
                self._form_set_row_visible(self._norm_form, r, visible)

        # ---- V1-only rows ----
        # None of these are read by the V2 code path, so showing them under V2
        # would be the bug this release fixes.
        show_row(self.row_blackpoint, not v2)
        show_row(self.row_hlrecover,  not v2)
        show_row(self.row_hlreduct,   not v2)
        show_row(self.row_brightness, not v2)
        show_row(self.cmb_blendmode,  (not v2) and is_hoo)
        show_row(self.row_hablend,    (not v2) and is_hoo)
        show_row(self.row_oiiiboost,  (not v2) and is_hoo)
        show_row(self.row_siiboost,   (not v2) and (not is_hoo))
        show_row(self.row_oiii_sho,   (not v2) and (not is_hoo))

        # Lightness is V1-only and only meaningful in non-linear mode.
        show_row(self.cmb_lightness, (not v2) and (mode == 1))
        if (not v2) and (mode == 1):
            if is_hoo:
                self._set_lightness_items(["Off (0)", "Original (1)", "Ha (2)", "OIII (3)"])
            else:
                self._set_lightness_items(["Off (0)", "Original (1)", "Ha (2)", "SII (3)", "OIII (4)"])

        # ---- V2-only rows ----
        showbg = v2 and self.chk_v2_showbg.isChecked()
        osc_on = v2 and is_hoo and self.chk_v2_osc.isChecked()

        show_row(self.chk_v2_bgn,      v2)
        show_row(self.row_v2_bgnoise,  v2)
        # When the background map is up, the tone controls have no effect on
        # what's displayed — the engine returns early.
        show_row(self.row_v2_red,      v2 and not showbg)
        show_row(self.row_v2_green,    v2 and not showbg)
        show_row(self.row_v2_blue,     v2 and not showbg)
        show_row(self.cmb_v2_lumhold,  v2 and not showbg)
        show_row(self.chk_v2_osc,      v2 and is_hoo and not showbg)
        show_row(self.cmb_v2_blendmode, osc_on and not showbg)
        show_row(self.row_v2_blendamt,  osc_on and not showbg)
        show_row(self.chk_v2_showbg,    v2)

        # ---- shared post-steps ----
        # SCNR stays SHO-family only, and is pointless over the background map.
        show_row(self.chk_scnr, (not is_hoo) and not showbg)
        scnr_subs = (not is_hoo) and self.chk_scnr.isChecked() and not showbg
        show_row(self.row_scnr_amount,   scnr_subs)
        # Preserve-luminance is a property of the remove_green post-step, which
        # only runs under V2; V1's native stage has no equivalent.
        show_row(self.chk_scnr_preserve, scnr_subs and v2)
        show_row(self.chk_bg_neutral,    not showbg)

        self.grp_hoo_extras.setVisible(is_hoo)
        self.grp_sho_extras.setVisible(not is_hoo)

        self.chk_linear_fit.setEnabled(True)

        self._schedule_preview()

    def _on_scnr_toggled(self, checked: bool):
        """Show/hide the SCNR sub-controls and recompute preview."""
        self._refresh_visibility()

    def _make_dspin(self, lo, hi, step, val, _debounce_timer_unused) -> QDoubleSpinBox:
        sp = QDoubleSpinBox(self)
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(3)
        sp.setValue(val)
        sp.valueChanged.connect(lambda *_: self._schedule_preview())
        return sp

    def _on_mode_changed(self):
        # Lightness only meaningful for non-linear (Mode=1) per Bill notes.
        non_linear = (self.cmb_mode.currentIndex() == 0)
        self.cmb_lightness.setEnabled(non_linear)
        self._schedule_preview()

    # ---------------- loaders ----------------
    def _gather_params(self) -> NBNParams:
        """Collect EVERY NBNParams field from the UI.

        Both engines' parameters are filled in on every call. The engine
        ignores the block it doesn't use, and keeping both populated means a
        mid-session engine switch doesn't silently fall back to dataclass
        defaults for the newly-active block.
        """
        scenario = self.cmb_scenario.currentText()
        mode = 1 if self.cmb_mode.currentIndex() == 0 else 0
        lightness = self.cmb_lightness.currentIndex()

        hlrecover = max(float(self.spin_hlrecover.value()), 0.25)
        hlreduct  = max(float(self.spin_hlreduct.value()), 0.25)
        brightness = max(float(self.spin_brightness.value()), 0.25)

        return NBNParams(
            # ---- shared ----
            scenario=scenario,
            mode=mode,
            use_v2=self._use_v2(),

            # ---- V1 (legacy) ----
            lightness=lightness,
            blackpoint=float(self.spin_blackpoint.value()),
            hlrecover=hlrecover,
            hlreduct=hlreduct,
            brightness=brightness,
            blendmode=self.cmb_blendmode.currentIndex(),
            hablend=float(self.spin_hablend.value()),
            oiiiboost=float(self.spin_oiiiboost.value()),
            siiboost=float(self.spin_siiboost.value()),
            oiiiboost2=float(self.spin_oiiiboost2.value()),
            # V1 has a native SCNR stage that runs on the normalized channels
            # before the Lab lightness replacement — a different (and better)
            # result than suppressing green at the very end. V2 ignores this
            # field, so it's only meaningful on the V1 path.
            scnr=bool(self.chk_scnr.isChecked()) and not self._use_v2(),
            scnr_amount=float(self.spin_scnr_amount.value()),

            # ---- V2 (Bill Blanshan v2.23) ----
            bgn=bool(self.chk_v2_bgn.isChecked()),
            background_noise=float(self.spin_v2_bgnoise.value()),
            red_boost=float(self.spin_v2_red.value()),
            green_boost=float(self.spin_v2_green.value()),
            blue_boost=float(self.spin_v2_blue.value()),
            v2_blend_mode=str(self.cmb_v2_blendmode.currentText()),
            v2_blend_amount=float(self.spin_v2_blendamt.value()),
            luminance_hold=str(self.cmb_v2_lumhold.currentText()),
            osc_hoo=bool(self.chk_v2_osc.isChecked()),
            show_background=bool(self.chk_v2_showbg.isChecked()),
        )

    def _set_status_label(self, which: str, text: str | None):
        lab = getattr(self, f"lbl_{which.lower()}")

        if text:
            full = str(text)
            shown = self._elide_label_text(full, max_chars=52)

            lab.setText(shown)
            lab.setToolTip(full)  # full path/view name accessible on hover
            lab.setStyleSheet("color:#2a7; font-weight:600; margin-left:8px;")
        else:
            lab.setText(f"No {which} loaded.")
            lab.setToolTip("")
            lab.setStyleSheet("color:#888; margin-left:8px;")


    def _load_channel(self, which: str):
        src, ok = QInputDialog.getItem(
            self, f"Load {which}", "Source:", ["From View", "From File"], 0, False
        )
        if not ok:
            return

        out = self._load_from_view(which) if src == "From View" else self._load_from_file(which)
        if out is None:
            return

        img, header, bit_depth, is_mono, path, label, src_doc = out

        # Track source document for apply-back (first loaded view wins)
        if src_doc is not None and self._source_doc is None:
            self._source_doc = src_doc

        # NB channels → mono; OSC → RGB
        if which in ("Ha", "OIII", "SII"):
            if img.ndim == 3:
                img = img[:, :, 0]
        else:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

        setattr(self, which.lower(), self._as_float01(img))
        self._set_status_label(which, label)
        self.status.setText(f"{which} loaded ({'mono' if img.ndim==2 else 'RGB'}) shape={img.shape}")

        self._schedule_preview()

    def _import_mapped_view(self, scenario: str):
        """
        Import an already-mapped RGB composite (e.g. from Perfect Palette Picker)
        and split it into Ha/OIII/SII channels according to scenario mapping.
        """
        # Force scenario selection to match the mapping the user chose
        idx = self.cmb_scenario.findText(scenario)
        if idx >= 0:
            self.cmb_scenario.setCurrentIndex(idx)

        views = self._list_open_views()
        if not views:
            QMessageBox.warning(self, "No Views", "No open image views were found.")
            return

        labels = [lab for lab, _ in views]
        choice, ok = QInputDialog.getItem(
            self, f"Select {scenario} View", "Choose a mapped RGB view:", labels, 0, False
        )
        if not ok or not choice:
            return

        sw = dict(views)[choice]
        doc = getattr(sw, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Empty View", "Selected view has no image.")
            return

        img = doc.image
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            QMessageBox.warning(
                self, "Not RGB",
                "That view is mono. Import requires an RGB mapped composite (3-channel)."
            )
            return

        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.warning(
                self, "Unsupported Shape",
                f"Expected RGB (H,W,3). Got {img.shape}."
            )
            return

        rgb = self._as_float01(img)

        ha, oiii, sii = self._split_mapped_rgb(rgb, scenario)

        # Store as mono float [0..1]
        self.ha = ha
        self.oiii = oiii
        self.sii = sii

        # Clear OSC helpers (we're now using direct NB channels)
        self.osc1 = None
        self.osc2 = None

        # Track source document so Apply writes back to the correct view
        # (not the temp split channels that only exist in memory)
        self._source_doc = doc

        # Labels
        src = f"From View: {choice}"
        if scenario == "HOO":
            self._set_status_label("Ha",   f"(Ha←R)")
            self._set_status_label("OIII", f"(OIII←G/B)")
            self._set_status_label("SII",  None)
        else:
            # indicate mapping
            map_txt = {
                "SHO": "(SII←R, Ha←G, OIII←B)",
                "HSO": "(Ha←R, SII←G, OIII←B)",
                "HOS": "(Ha←R, OIII←G, SII←B)",
            }.get(scenario, "")
            self._set_status_label("Ha",   f"{map_txt}")
            self._set_status_label("OIII", f"{map_txt}")
            self._set_status_label("SII",  f"{map_txt}")

        self.status.setText(f"Imported mapped {scenario} view → channels loaded.")
        self._schedule_preview()


    def _split_mapped_rgb(self, rgb: np.ndarray, scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Given an RGB mapped composite in [0..1], return (Ha, OIII, SII) mono channels.
        For HOO, returns (Ha, OIII, None).
        """
        r = rgb[..., 0].astype(np.float32, copy=False)
        g = rgb[..., 1].astype(np.float32, copy=False)
        b = rgb[..., 2].astype(np.float32, copy=False)

        scen = scenario.upper().strip()

        if scen == "SHO":
            # R=SII, G=Ha, B=OIII
            sii = r
            ha = g
            oiii = b
            return ha, oiii, sii

        if scen == "HSO":
            # R=Ha, G=SII, B=OIII
            ha = r
            sii = g
            oiii = b
            return ha, oiii, sii

        if scen == "HOS":
            # R=Ha, G=OIII, B=SII
            ha = r
            oiii = g
            sii = b
            return ha, oiii, sii

        if scen == "HOO":
            # Common mapping: R=Ha, G/B = OIII-ish
            ha = r
            oiii = 0.5 * (g + b)
            return ha, oiii.astype(np.float32, copy=False), None

        # Fallback: treat as HOS-ish
        ha = r
        oiii = g
        sii = b
        return ha, oiii, sii


    def _load_from_view(self, which):
        views = self._list_open_views()
        if not views:
            QMessageBox.warning(self, "No Views", "No open image views were found.")
            return None

        labels = [lab for lab, _ in views]
        choice, ok = QInputDialog.getItem(
            self, f"Select View for {which}", "Choose a view (by name):", labels, 0, False
        )
        if not ok or not choice:
            return None

        sw = dict(views)[choice]
        doc = getattr(sw, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, "Empty View", "Selected view has no image.")
            return None

        img = doc.image
        meta = getattr(doc, "metadata", {}) or {}
        header = meta.get("original_header", None)
        bit_depth = meta.get("bit_depth", "Unknown")
        is_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)
        path = meta.get("file_path", None)
        return img, header, bit_depth, is_mono, path, f"From View: {choice}", doc

    def _load_from_file(self, which):
        filt = "Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"
        path, _ = QFileDialog.getOpenFileName(self, f"Select {which} File", "", filt)
        if not path:
            return None
        img, header, bit_depth, is_mono = legacy_load_image(path)
        if img is None:
            QMessageBox.critical(self, "Load Error", f"Could not load {os.path.basename(path)}")
            return None
        label = f"From File: {os.path.basename(path)}"
        return img, header, bit_depth, is_mono, path, label, None

    # ---------------- channel prep ----------------
    def _as_float01(self, arr):
        a = np.asarray(arr)
        if a.dtype == np.uint8:
            return a.astype(np.float32) / 255.0
        if a.dtype == np.uint16:
            return a.astype(np.float32) / 65535.0
        return np.clip(a.astype(np.float32), 0.0, 1.0)

    def _resize_to(self, arr: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
        """Resize np array to (w,h). Keeps dtype/scale. Uses INTER_AREA for downsizing."""
        if arr is None:
            return None
        w, h = size
        if arr.ndim == 2:
            src_h, src_w = arr.shape
        else:
            src_h, src_w = arr.shape[:2]
        if (src_w, src_h) == (w, h):
            return arr
        interp = cv2.INTER_AREA if (w < src_w or h < src_h) else cv2.INTER_LINEAR
        return cv2.resize(arr, (w, h), interpolation=interp)

    def _prepared_channels(self):
        """
        Build Ha/OIII/SII bases from inputs.
        Strategy (strict, safer for normalization):
        - If NB channels are present, prefer them.
        - Else synthesize from OSC inputs:
            OSC1: R≈Ha, mean(G,B)≈OIII
            OSC2: R≈SII, mean(G,B)≈OIII
        - If dimensions differ, prompt once and resize to reference.
        """
        ha = self.ha
        oo = self.oiii
        si = self.sii
        o1 = self.osc1
        o2 = self.osc2

        # If NB present, keep them; else synth from OSC.
        if ha is None and o1 is not None:
            ha = o1[..., 0]
        if oo is None and o1 is not None:
            oo = o1[..., 1:3].mean(axis=2)

        if si is None and o2 is not None:
            si = o2[..., 0]
        # If OIII still missing, try OSC2 too
        if oo is None and o2 is not None:
            oo = o2[..., 1:3].mean(axis=2)

        # Basic requirements for scenarios:
        # HOO: needs Ha + OIII
        # others: ideally need Ha + SII + OIII (but we can allow missing and warn)
        shapes = [x.shape[:2] for x in (ha, oo, si) if x is not None]
        if len(shapes) and len(set(shapes)) > 1:
            # choose reference (prefer Ha, then OIII, then SII)
            ref = ha if ha is not None else (oo if oo is not None else si)
            ref_name = "Ha" if ha is not None else ("OIII" if oo is not None else "SII")
            ref_h, ref_w = ref.shape[:2]

            if not self._dim_mismatch_accepted:
                msg = (
                    "The loaded channels have different image dimensions.\n\n"
                    f"• Ha:   {None if ha is None else ha.shape}\n"
                    f"• OIII: {None if oo is None else oo.shape}\n"
                    f"• SII:  {None if si is None else si.shape}\n\n"
                    f"SASpro can resize (warp) the channels to match the reference frame:\n"
                    f"• Reference: {ref_name}\n"
                    f"• Target size: ({ref_w} × {ref_h})\n\n"
                    "Proceed and resize mismatched channels?"
                )
                ret = QMessageBox.question(
                    self,
                    "Channel Size Mismatch",
                    msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if ret != QMessageBox.StandardButton.Yes:
                    return None, None, None
                self._dim_mismatch_accepted = True

            ha = self._resize_to(ha, (ref_w, ref_h)) if ha is not None else None
            oo = self._resize_to(oo, (ref_w, ref_h)) if oo is not None else None
            si = self._resize_to(si, (ref_w, ref_h)) if si is not None else None

        return ha, oo, si

    def _linear_fit_channels(self, ha, oo, si, ref="Ha"):
        """
        Use the shared Linear Fit engine to median-match each mono channel
        to a chosen reference channel.

        Uses rescale_mode_idx=2 (leave as-is) so we don't normalize/clip here;
        the normalization algorithm should decide what to do later.
        """
        if ha is None and oo is None and si is None:
            return ha, oo, si

        # pick reference array
        ref_arr = None
        if ref == "Ha" and ha is not None:
            ref_arr = ha
        elif ref == "OIII" and oo is not None:
            ref_arr = oo
        elif ref == "SII" and si is not None:
            ref_arr = si
        else:
            # fallback: first available
            ref_arr = ha if ha is not None else (oo if oo is not None else si)

        if ref_arr is None:
            return ha, oo, si

        # rescale_mode_idx:
        # 0 clip, 1 normalize if needed, 2 leave as-is
        rescale_mode_idx = 2

        def fit(ch):
            if ch is None:
                return None
            out, _, _ = linear_fit_mono_to_ref(ch, ref_arr, rescale_mode_idx=rescale_mode_idx)
            return out.astype(np.float32, copy=False)

        return fit(ha), fit(oo), fit(si)

    # ---------------- Single Launch Job Function -----------------------
    def _set_busy(self, busy: bool, msg: str = ""):
        # Optional UX: disable buttons while processing
        for w in (self.btn_preview, self.btn_apply, self.btn_push, self.btn_clear,
                  self.btn_ha, self.btn_oiii, self.btn_sii, self.btn_osc1, self.btn_osc2):
            try:
                w.setEnabled(not busy)
            except Exception:
                pass
        if msg:
            self.status.setText(msg)

    def _prepare_inputs_for_job(self, *, downsample: bool) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
        """
        Returns (ha, oo, si, scale_used). If downsample=True, returns downsampled channels.
        Applies the SAME channel derivation + optional linear fit as the full-res path,
        just on the preview-resolution data.
        """
        ha, oo, si = self._prepared_channels()
        ok, msg = self._requirements_met(ha, oo, si)
        if not ok:
            raise RuntimeError(msg)

        # Downsample first (preview path)
        s = 1.0
        if downsample:
            s = self._preview_scale()
            ha = self._downsample_mono(ha, s)
            oo = self._downsample_mono(oo, s)
            si = self._downsample_mono(si, s)

        # Optional linear fit (apply it on whatever resolution we're running at)
        if self.chk_linear_fit.isChecked():
            meds = {}
            if ha is not None: meds["Ha"] = _nanmedian(ha)
            if oo is not None: meds["OIII"] = _nanmedian(oo)
            if si is not None: meds["SII"] = _nanmedian(si)
            ref = max(meds, key=meds.get) if meds else "Ha"
            ha, oo, si = self._linear_fit_channels(ha, oo, si, ref=ref)

        return ha, oo, si, s

    def _start_job(
        self,
        *,
        downsample: bool,
        step_name: str,
        on_done,                 # (out: np.ndarray, step_name: str) -> None
        on_fail=None,
    ):
        """
        One job launcher for preview/apply/push.
        Uses seq gating so stale workers can't overwrite newer results.
        """
        self._calc_seq += 1
        seq = int(self._calc_seq)
        self._active_seq = seq

        try:
            ha, oo, si, s = self._prepare_inputs_for_job(downsample=downsample)
        except Exception as e:
            if _NBN_DEBUG:
                print(f"[NBN] _prepare_inputs_for_job failed:\n{traceback.format_exc()}")
            self._safe_set_status(str(e))
            return

        if downsample and s < 0.999:
            self._set_busy(True, f"Computing preview… (downsample {s:.2f}×)")
        else:
            self._set_busy(True, "Computing…")

        def _done(out, _step):
            if seq != self._calc_seq:
                return
            try:
                self.final = out
                on_done(out, _step)
            finally:
                self._set_busy(False)

        def _fail(err: str):
            if seq != self._calc_seq:
                return
            try:
                if on_fail is not None:
                    on_fail(err)
                else:
                    print(f"[NBN] job failed:\n{err}")
                    QMessageBox.critical(self, "Narrowband Normalization", err)
                    self._safe_set_status("Failed.")
            finally:
                self._set_busy(False)

        # Always go through worker so progress emits (preview + full-res)
        self._start_nbn_worker(ha, oo, si, step_name=step_name, on_done=_done, on_fail=_fail)



    # ---------------- normalization core (STUBS for now) ----------------
    def _run_normalization(self, ha, oo, si) -> np.ndarray:
        """
        Placeholder implementation:
        - Applies optional quick linear fit
        - Produces a basic RGB mapping for the selected scenario so UI works today        
        """
        scenario = self.cmb_scenario.currentText()

        if self.chk_linear_fit.isChecked():
            # auto pick highest-median reference among available channels
            meds = {}
            if ha is not None: meds["Ha"] = _nanmedian(ha)
            if oo is not None: meds["OIII"] = _nanmedian(oo)
            if si is not None: meds["SII"] = _nanmedian(si)
            ref = max(meds, key=meds.get) if meds else "Ha"
            ha, oo, si = self._linear_fit_channels(ha, oo, si, ref=ref)

        # Basic sanity
        if scenario == "HOO":
            if ha is None or oo is None:
                raise RuntimeError("HOO requires Ha + OIII (or OSC1 providing both).")
            r = ha
            g = oo
            b = oo
        else:
            if ha is None or oo is None or si is None:
                raise RuntimeError(f"{scenario} requires Ha + OIII + SII (or OSC1+OSC2).")
            if scenario == "SHO":
                r, g, b = si, ha, oo
            elif scenario == "HSO":
                r, g, b = ha, si, oo
            elif scenario == "HOS":
                r, g, b = ha, oo, si
            else:
                r, g, b = ha, oo, si

        rgb = np.stack([r, g, b], axis=2).astype(np.float32)
        mx = float(rgb.max()) or 1.0
        rgb = np.clip(rgb / mx, 0.0, 1.0)
        return rgb

    def _on_worker_progress(self, p: int, m: str):
        self._safe_set_status(f"{m} ({p}%)" if m else f"{p}%")

    def _start_nbn_worker(self, ha, oo, si, *, step_name: str, on_done, on_fail=None):
        """
        Start a background normalization job.
        Keeps a strong reference to the worker and routes signals safely.
        """
        params = self._gather_params()

        if _NBN_DEBUG:
            try:
                import dataclasses as _dc
                print(f"[NBN] params in: {_dc.asdict(params)}")
            except Exception:
                print(f"[NBN] params in: {params!r}")

        job = _NBNJob(ha=ha, oiii=oo, sii=si, params=params, step_name=step_name)

        # Keep strong ref, and parent it so Qt owns lifetime correctly
        w = _NBNWorker(job)
        try:
            w.setParent(self)
        except Exception:
            pass
        self._worker = w

        # progress → safe UI update
        w.progress.connect(self._on_worker_progress)

        # done/fail wrappers → always safe + always cleanup
        def _done_wrapper(out, _step):
            try:
                if _NBN_DEBUG:
                    try:
                        a = np.asarray(out, dtype=np.float64)
                        print(f"[NBN] out: shape={a.shape} sum={a.sum():.6f} "
                              f"mean={a.mean():.8f} min={a.min():.6f} max={a.max():.6f}")
                    except Exception:
                        pass
                if self._ui_alive():
                    on_done(out, _step)
            finally:
                # stop late emissions touching us
                try:
                    w.progress.disconnect(self._on_worker_progress)
                except Exception:
                    pass
                try:
                    w.done.disconnect(_done_wrapper)
                except Exception:
                    pass
                try:
                    w.failed.disconnect(_fail_wrapper)
                except Exception:
                    pass
                try:
                    w.deleteLater()
                except Exception:
                    pass
                if self._worker is w:
                    self._worker = None

        def _fail_wrapper(err: str):
            try:
                if on_fail is not None:
                    if self._ui_alive():
                        on_fail(err)
                else:
                    print(f"[NBN] worker failed:\n{err}")
                    if self._ui_alive():
                        QMessageBox.critical(self, "Narrowband Normalization", err)
            finally:
                try:
                    w.progress.disconnect()
                except Exception:
                    pass
                try:
                    w.done.disconnect()
                except Exception:
                    pass
                try:
                    w.failed.disconnect()
                except Exception:
                    pass
                try:
                    w.deleteLater()
                except Exception:
                    pass
                if self._worker is w:
                    self._worker = None

        w.done.connect(_done_wrapper)
        w.failed.connect(_fail_wrapper)

        w.start()


    # ---------------- preview ----------------
    def _update_preview(self):
        ha, oo, si = self._prepared_channels()
        ok, msg = self._requirements_met(ha, oo, si)
        if not ok:
            self.status.setText(msg)
            return

        # optional linear fit
        if self.chk_linear_fit.isChecked():
            meds = {}
            if ha is not None: meds["Ha"] = _nanmedian(ha)
            if oo is not None: meds["OIII"] = _nanmedian(oo)
            if si is not None: meds["SII"] = _nanmedian(si)
            ref = max(meds, key=meds.get) if meds else "Ha"
            ha, oo, si = self._linear_fit_channels(ha, oo, si, ref=ref)

        params = self._gather_params()
        out = normalize_narrowband(ha, oo, si, params, progress_cb=None)
        out = self._maybe_apply_scnr(out)
        self.final = out

        disp = out
        if self.chk_preview_autostretch.isChecked():
            disp = np.clip(stretch_color_image(disp, target_median=0.25, linked=True), 0.0, 1.0)

        first = (self._base_pm is None)
        self._set_preview_image(self._to_qimage(disp), fit=first, preserve_view=True)
        self.status.setText(f"Preview updated ({self.cmb_scenario.currentText()}).")


    def _capture_view_state(self):
        if self._base_pm is None:
            return None
        vp = self.scroll.viewport()

        anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)
        anchor_lbl = self.preview.mapFrom(vp, anchor_vp)

        base_x = anchor_lbl.x() / max(self._zoom, 1e-6)
        base_y = anchor_lbl.y() / max(self._zoom, 1e-6)

        pm = self._base_pm.size()
        fx = 0.5 if pm.width() <= 0 else (base_x / pm.width())
        fy = 0.5 if pm.height() <= 0 else (base_y / pm.height())

        return {"zoom": float(self._zoom), "fx": float(fx), "fy": float(fy)}

    def _restore_view_state(self, state):
        if not state or self._base_pm is None:
            return

        self._zoom = max(self._min_zoom, min(self._max_zoom, float(state["zoom"])))
        self._update_preview_pixmap()

        pm = self._base_pm.size()
        fx = float(state.get("fx", 0.5))
        fy = float(state.get("fy", 0.5))
        base_x = fx * pm.width()
        base_y = fy * pm.height()

        lbl_x = int(base_x * self._zoom)
        lbl_y = int(base_y * self._zoom)

        vp = self.scroll.viewport()
        anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), lbl_x - anchor_vp.x())))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), lbl_y - anchor_vp.y())))

    def _set_preview_image(self, qimg: QImage, *, fit: bool = False, preserve_view: bool = True):
        state = None
        if preserve_view and (not fit) and (self._base_pm is not None):
            state = self._capture_view_state()

        self._base_pm = QPixmap.fromImage(qimg)

        if fit or state is None:
            self._zoom = 1.0
            self._update_preview_pixmap()
            if fit:
                QTimer.singleShot(0, self._fit_to_preview)
            else:
                QTimer.singleShot(0, self._center_scrollbars)
            return

        self._restore_view_state(state)

    def _update_preview_pixmap(self):
        if self._base_pm is None:
            return

        base_sz = self._base_pm.size()
        w = max(1, int(base_sz.width() * self._zoom))
        h = max(1, int(base_sz.height() * self._zoom))

        # Heuristic:
        # - Fast when zoomed out (lots of pixels squeezed) or when scaled image is huge
        # - Smooth when zoomed in (user wants quality)
        scaled_pixels = w * h
        huge = scaled_pixels >= 6_000_000        # ~6MP threshold (tweak)
        zoomed_out = self._zoom < 1.0

        mode = Qt.TransformationMode.FastTransformation if (huge or zoomed_out) else Qt.TransformationMode.SmoothTransformation

        scaled = self._base_pm.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            mode
        )
        self.preview.setPixmap(scaled)
        self.preview.resize(scaled.size())


    def _set_zoom(self, new_zoom: float):
        self._zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))
        self._update_preview_pixmap()

    def _zoom_at(self, factor: float = 1.25, anchor_vp: QPoint | None = None):
        if self._base_pm is None:
            return

        vp = self.scroll.viewport()
        if anchor_vp is None:
            anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)

        lbl_before = self.preview.mapFrom(vp, anchor_vp)

        old_zoom = self._zoom
        new_zoom = max(self._min_zoom, min(self._max_zoom, old_zoom * factor))
        ratio = new_zoom / max(old_zoom, 1e-6)
        if abs(ratio - 1.0) < 1e-6:
            return

        self._zoom = new_zoom
        self._update_preview_pixmap()

        lbl_after_x = int(lbl_before.x() * ratio)
        lbl_after_y = int(lbl_before.y() * ratio)

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(max(hbar.minimum(), min(hbar.maximum(), lbl_after_x - anchor_vp.x())))
        vbar.setValue(max(vbar.minimum(), min(vbar.maximum(), lbl_after_y - anchor_vp.y())))

    def _fit_to_preview(self):
        if self._base_pm is None:
            return
        vp = self.scroll.viewport().size()
        pm = self._base_pm.size()
        if pm.width() == 0 or pm.height() == 0:
            return
        k = min(vp.width() / pm.width(), vp.height() / pm.height())
        self._set_zoom(max(self._min_zoom, min(self._max_zoom, k)))
        self._center_scrollbars()

    def _center_scrollbars(self):
        h = self.scroll.horizontalScrollBar()
        v = self.scroll.verticalScrollBar()
        h.setValue((h.maximum() + h.minimum()) // 2)
        v.setValue((v.maximum() + v.minimum()) // 2)

    # ---------------- actions ----------------
    def _clear_channels(self):
        self.ha = self.oiii = self.sii = self.osc1 = self.osc2 = None
        self._source_doc = None
        self._dim_mismatch_accepted = False
        self.final = None
        self._base_pm = None
        self.preview.clear()
        for which in ("Ha", "OIII", "SII", "OSC1", "OSC2"):
            self._set_status_label(which, None)
        self.status.setText("Cleared all loaded channels.")

    def _resolve_target_doc(self):
        """
        Return the document that "Apply to Current View" should write to.

        Priority:
          1. _source_doc  – the view the user loaded channels from (especially
             important for OSC imports where the temp channel arrays are in
             memory only).
          2. mw.current_document()  – whatever is focused right now.
          3. None  → caller falls back to Push.
        """
        # Try the source doc we tracked at load time
        if self._source_doc is not None:
            img = getattr(self._source_doc, "image", None)
            if img is not None:
                return self._source_doc

        mw = self._find_main_window()
        doc = getattr(mw, "current_document", None)
        if callable(doc):
            doc = doc()
        return doc

    def _apply_to_current_view(self):
        if self._use_v2() and self.chk_v2_showbg.isChecked():
            QMessageBox.information(
                self, "Background Map",
                "The background map is a diagnostic view.\n"
                "Turn off 'Show background map' before applying."
            )
            return

        doc = self._resolve_target_doc()
        if doc is None:
            QMessageBox.information(self, "No Active Doc", "Couldn't find an active document; pushing to new view instead.")
            self._push_result()
            return

        def on_done(out: np.ndarray, step_name: str):
            try:
                out2 = self._maybe_apply_scnr(out)
                out2 = self._maybe_background_neutralize_rgb(out2, doc_for_mask=doc)

                # Prefer apply_edit if your doc supports it (history + metadata)
                if hasattr(doc, "apply_edit"):
                    meta = {
                        "step_name": "Narrowband Normalization",
                        "nbn_engine": "V2" if self._use_v2() else "V1",
                    }
                    if self.chk_bg_neutral.isChecked():
                        meta["post_step"] = "Background Neutralization (auto)"
                    doc.apply_edit(out2.astype(np.float32, copy=False), metadata=meta, step_name="Narrowband Normalization")
                elif hasattr(doc, "set_image"):
                    doc.set_image(out2, step_name="Narrowband Normalization")
                else:
                    doc.image = out2

                self.status.setText("Applied normalization to current view.")
            except Exception as e:
                print(f"[NBN] apply failed:\n{traceback.format_exc()}")
                QMessageBox.critical(self, "Apply Error", f"Failed to apply:\n{e}")


        self._start_job(
            downsample=False,                 # FULL RES
            step_name="NBN Apply",
            on_done=on_done,
        )

    def _get_doc_manager(self):
        if self.doc_manager is not None:
            return self.doc_manager
        mw = self._find_main_window()
        if mw is None:
            return None
        return getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)

    def _push_result(self):
        dm = self._get_doc_manager()
        if dm is None:
            QMessageBox.warning(self, "DocManager Missing", "DocManager not found; can't push to a new view.")
            return

        engine = "V2" if self._use_v2() else "V1"
        title = f"NBN {self.cmb_scenario.currentText()} ({engine})"

        def on_done(out: np.ndarray, step_name: str):
            try:
                # Apply optional SCNR + headless BN to the RESULT before pushing
                out2 = self._maybe_apply_scnr(out)
                out2 = self._maybe_background_neutralize_rgb(out2, doc_for_mask=None)

                meta = {"is_mono": False, "nbn_engine": engine}
                if getattr(self, "chk_bg_neutral", None) and self.chk_bg_neutral.isChecked():
                    meta["post_step"] = "Background Neutralization (auto)"

                if hasattr(dm, "open_array"):
                    dm.open_array(out2, metadata=meta, title=title)
                elif hasattr(dm, "create_document"):
                    dm.create_document(image=out2, metadata=meta, name=title)
                else:
                    raise RuntimeError("DocManager lacks open_array/create_document")

                self.status.setText("Opened result in a new view.")
            except Exception as e:
                print(f"[NBN] push failed:\n{traceback.format_exc()}")
                QMessageBox.critical(self, "Push Error", f"Failed to open new view:\n{e}")

        self._start_job(
            downsample=False,                 # FULL RES
            step_name="NBN Push",
            on_done=on_done,
        )

    # ---------------- utilities ----------------
    def _to_qimage(self, arr):
        # ascontiguousarray matters: astype() defaults to order='K', which
        # preserves a non-contiguous input's layout, and QImage would then
        # read it against the wrong stride.
        a = np.clip(arr, 0, 1)
        if a.ndim == 2:
            u = np.ascontiguousarray((a * 255).astype(np.uint8))
            h, w = u.shape
            return QImage(u.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        if a.ndim == 3 and a.shape[2] == 3:
            u = np.ascontiguousarray((a * 255).astype(np.uint8))
            h, w, _ = u.shape
            return QImage(u.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        raise ValueError(f"Unexpected image shape: {a.shape}")

    def _find_main_window(self):
        w = self
        from PyQt6.QtWidgets import QMainWindow, QApplication
        while w is not None and not isinstance(w, QMainWindow):
            w = w.parentWidget()
        if w:
            return w
        for tlw in QApplication.topLevelWidgets():
            if isinstance(tlw, QMainWindow):
                return tlw
        return None

    def _list_open_views(self):
        mw = self._find_main_window()
        if not mw:
            return []
        try:
            from setiastro.saspro.subwindow import ImageSubWindow
            subs = mw.findChildren(ImageSubWindow)
        except Exception:
            subs = []
        out = []
        for sw in subs:
            title = getattr(sw, "view_title", None) or sw.windowTitle() or getattr(sw.document, "display_name", lambda: "Untitled")()
            out.append((str(title), sw))
        return out

    def _elide_label_text(self, s: str, *, max_chars: int = 48) -> str:
        """
        Cap how much we show in the green status labels so long paths/view names
        don't resize the whole left panel and cut off sliders.
        """
        s = (s or "").strip()
        if len(s) <= max_chars:
            return s
        # keep the important end of the string (usually filename)
        head = max(0, max_chars - 1 - 18)
        return f"{s[:head]}…{s[-18:]}"

    # ---------------- event filter (zoom/pan) ----------------
    def eventFilter(self, obj, ev):
        # Wheel = zoom at mouse (plain wheel + Ctrl+wheel). No scrolling.
        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview
            or obj is self.scroll
            or obj is self.scroll.viewport()
            or obj is self.scroll.horizontalScrollBar()
            or obj is self.scroll.verticalScrollBar()
        ):
            ev.accept()

            vp = self.scroll.viewport()
            anchor_vp = vp.mapFromGlobal(ev.globalPosition().toPoint())

            r = vp.rect()
            if not r.contains(anchor_vp):
                anchor_vp.setX(max(r.left(), min(r.right(), anchor_vp.x())))
                anchor_vp.setY(max(r.top(), min(r.bottom(), anchor_vp.y())))

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
                    return True

                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                step = 1.25 if ctrl_down else 1.15
                factor = step if dy > 0 else 1.0 / step

            self._zoom_at(factor, anchor_vp)
            return True

        # click-drag pan on viewport
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_last = ev.position().toPoint()
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                cur = ev.position().toPoint()
                delta = cur - (self._pan_last or cur)
                self._pan_last = cur
                h = self.scroll.horizontalScrollBar()
                v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - delta.x())
                v.setValue(v.value() - delta.y())
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self._pan_last = None
                self.scroll.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                return True

        return super().eventFilter(obj, ev)

    def closeEvent(self, e):
        # bump seq so any in-flight callbacks are ignored
        try:
            self._calc_seq += 1
        except Exception:
            pass
        super().closeEvent(e)