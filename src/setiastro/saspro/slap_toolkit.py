# ============================================================
#  ____  _         _____           _ _    _ _
# / ___|| |    __ |_   _|__   ___ | | | _(_) |_
# \___ \| |   / _` || |/ _ \ / _ \| | |/ / | __|
#  ___) | |__| (_| || | (_) | (_) | |   <| | |_
# |____/|_____\__,_||_|\___/ \___/|_|_|\_\_|\__|
#
#  Solar, Lunar, and Planetary Toolkit  (SLaP Toolkit)
#  src/setiastro/saspro/slap_toolkit.py
#
#  Part of Seti Astro Suite Pro
#  Copyright © 2025 Franklin Marek  |  www.setiastro.com
#  All rights reserved.
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFrame,
)

# ─────────────────────────────────────────────────────────────
# SLaP modes
# ─────────────────────────────────────────────────────────────

SLAP_MODES = [
    "Solar — Surface",
    "Solar — Surface + Prominence",
    "Solar — Prominence Only",
    "Lunar",
    "Planetary",
]

# ─────────────────────────────────────────────────────────────
# Canned workflow definitions (imported into WorkflowAssistant)
# ─────────────────────────────────────────────────────────────

def slap_canned_workflows():
    """
    Returns a list of WorkflowDefinition objects for each SLaP mode.
    Called by WorkflowDialog to populate its canned workflow menu.
    """
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    return [
        WorkflowDefinition(
            name="SLaP — Solar Surface",
            description="Full solar surface processing pipeline: crop, pedestal, deconvolution, "
                        "multiscale sharpening, LHE contrast, denoise, colorize.",
            lanes=["main"],
            steps=[
                WorkflowStep(command_id="crop",
                             note="Crop to the solar disc — remove any black border from stacking.",
                             lane="main"),
                WorkflowStep(command_id="pedestal",
                             note="Remove the pedestal / minimum black point.",
                             lane="main"),
                WorkflowStep(command_id="multiscale_decomp",
                             note="Multiscale sharpening: boost layers 1–3, suppress layer 4+ "
                                  "to control noise. Use Linear mode for a clean sharpen.",
                             lane="main"),
                WorkflowStep(command_id="convo",
                             note="Optional: Richardson-Lucy deconvolution with a small PSF radius "
                                  "(1.5–2 px) to restore fine surface detail.",
                             lane="main"),
                WorkflowStep(command_id="clahe",
                             note="CLAHE / LHE to bring out granulation and sunspot structure.",
                             lane="main"),
                WorkflowStep(command_id="cosmicclarity",
                             note="Cosmic Clarity denoise — keep it light, surface detail is precious.",
                             lane="main"),
                WorkflowStep(command_id="curves",
                             note="Inverted-V curves: lift midtones, keep limb darkening natural.",
                             lane="main"),
                WorkflowStep(command_id="mono_to_rgb",
                             note="Colorize: apply a solar palette (warm yellow-orange for white "
                                  "light, red for Hα, violet for Ca-K).",
                             lane="main"),
                WorkflowStep(command_id="save_as",
                             note="Save your result.",
                             lane="main"),
            ],
        ),

        WorkflowDefinition(
            name="SLaP — Solar Surface + Prominence",
            description="Solar processing with prominence enhancement using WaveScale Dark Enhancer "
                        "and a limb mask to protect the disc.",
            lanes=["main"],
            steps=[
                WorkflowStep(command_id="crop",
                             note="Crop to the solar disc, leaving a margin of space around the limb "
                                  "to preserve prominence data.",
                             lane="main"),
                WorkflowStep(command_id="pedestal",
                             note="Remove pedestal.",
                             lane="main"),
                WorkflowStep(command_id="wavescale_dark_enhance",
                             note="WaveScale Dark Enhancer: boost faint prominence signal near the "
                                  "solar limb. Set Mask Gamma high (5–8) so the effect stays outside "
                                  "the bright disc.",
                             lane="main"),
                WorkflowStep(command_id="multiscale_decomp",
                             note="Multiscale: sharpen surface detail on layers 1–3. "
                                  "Leave prominence layers (4–5) with Gain ≤ 1 to avoid boosting noise.",
                             lane="main"),
                WorkflowStep(command_id="clahe",
                             note="LHE / CLAHE on the disc area. Consider masking out space regions.",
                             lane="main"),
                WorkflowStep(command_id="cosmicclarity",
                             note="Light denoise — protect fine prominence filaments.",
                             lane="main"),
                WorkflowStep(command_id="curves",
                             note="S-curve for surface; inverted-V or gentle lift for the limb.",
                             lane="main"),
                WorkflowStep(command_id="mono_to_rgb",
                             note="Colorize with an Hα or white-light solar palette.",
                             lane="main"),
                WorkflowStep(command_id="save_as",
                             note="Save your result.",
                             lane="main"),
            ],
        ),

        WorkflowDefinition(
            name="SLaP — Solar Prominence Only",
            description="Processing for over-exposed prominence-only data where the disc is clipped.",
            lanes=["main"],
            steps=[
                WorkflowStep(command_id="crop",
                             note="Crop to the area of interest around the prominences.",
                             lane="main"),
                WorkflowStep(command_id="pedestal",
                             note="Remove pedestal.",
                             lane="main"),
                WorkflowStep(command_id="wavescale_dark_enhance",
                             note="WaveScale Dark Enhancer: use a low Mask Gamma (1–2) so the "
                                  "effect spreads into faint prominence regions.",
                             lane="main"),
                WorkflowStep(command_id="multiscale_decomp",
                             note="Multiscale: gentle gain on fine layers to reveal prominence "
                                  "structure without amplifying background noise.",
                             lane="main"),
                WorkflowStep(command_id="freqsep",
                             note="Optional Frequency Separation: work on the LF component to "
                                  "improve prominence gradient smoothness.",
                             lane="main"),
                WorkflowStep(command_id="cosmicclarity",
                             note="Denoise the background space aggressively.",
                             lane="main"),
                WorkflowStep(command_id="curves",
                             note="Gentle S-curve — protect the faint tips.",
                             lane="main"),
                WorkflowStep(command_id="mono_to_rgb",
                             note="Colorize: deep red / orange palette for Hα prominences.",
                             lane="main"),
                WorkflowStep(command_id="save_as",
                             note="Save your result.",
                             lane="main"),
            ],
        ),

        WorkflowDefinition(
            name="SLaP — Lunar",
            description="Lunar surface processing: high-contrast terrain detail with controlled "
                        "noise and optional false-color.",
            lanes=["main"],
            steps=[
                WorkflowStep(command_id="crop",
                             note="Crop to the lunar disc.",
                             lane="main"),
                WorkflowStep(command_id="pedestal",
                             note="Remove pedestal.",
                             lane="main"),
                WorkflowStep(command_id="convo",
                             note="Richardson-Lucy deconvolution — PSF 1.5–2.5 px, 20–40 iterations. "
                                  "Bring out crater rims and rilles.",
                             lane="main"),
                WorkflowStep(command_id="multiscale_decomp",
                             note="Multiscale: boost layers 1–4 to enhance fine surface texture. "
                                  "Apply NR on the highest layer to suppress grain.",
                             lane="main"),
                WorkflowStep(command_id="clahe",
                             note="CLAHE to balance the huge dynamic range from bright highlands "
                                  "to dark mare.",
                             lane="main"),
                WorkflowStep(command_id="wavescale_dark_enhance",
                             note="Optional WaveScale: compress the very bright limb regions so "
                                  "terminator detail isn't lost.",
                             lane="main"),
                WorkflowStep(command_id="cosmicclarity",
                             note="Light denoise — preserve crater rim sharpness.",
                             lane="main"),
                WorkflowStep(command_id="curves",
                             note="Contrast S-curve. Lunar images respond well to strong midtone "
                                  "contrast.",
                             lane="main"),
                WorkflowStep(command_id="save_as",
                             note="Save your result.",
                             lane="main"),
            ],
        ),

        WorkflowDefinition(
            name="SLaP — Planetary",
            description="Planetary processing pipeline optimised for small-disc high-magnification "
                        "RGB or narrowband data.",
            lanes=["main", "RGB", "L"],
            steps=[
                WorkflowStep(command_id="crop",
                             note="Crop tightly around the planet to speed up processing.",
                             lane="main",
                             outputs=["cropped"]),
                WorkflowStep(command_id="pedestal",
                             note="Remove pedestal.",
                             lane="main",
                             inputs=["cropped"],
                             outputs=["linear"]),
                WorkflowStep(kind="split",
                             note="Split into RGB colour processing and Luminance sharpening lanes.",
                             lane="main",
                             outputs=["rgb_path", "lum_path"]),

                # ── RGB lane ─────────────────────────────────────────
                WorkflowStep(command_id="white_balance",
                             note="White balance the RGB data before combining with L.",
                             lane="RGB",
                             inputs=["rgb_path"],
                             outputs=["rgb_balanced"]),
                WorkflowStep(command_id="cosmicclarity",
                             note="Light denoise on RGB — keep chroma noise under control.",
                             lane="RGB",
                             inputs=["rgb_balanced"],
                             outputs=["rgb_clean"]),

                # ── Luminance lane ────────────────────────────────────
                WorkflowStep(command_id="convo",
                             note="Richardson-Lucy deconvolution on the luminance channel. "
                                  "PSF 1–2 px, 30–60 iterations. This is where planetary detail lives.",
                             lane="L",
                             inputs=["lum_path"],
                             outputs=["lum_decon"]),
                WorkflowStep(command_id="multiscale_decomp",
                             note="Multiscale sharpening on L: strong gain on layers 1–3, "
                                  "NR on layers 4+. Use Linear mode.",
                             lane="L",
                             inputs=["lum_decon"],
                             outputs=["lum_sharp"]),
                WorkflowStep(command_id="clahe",
                             note="LHE / CLAHE on L to boost belt, zone, and polar cap contrast.",
                             lane="L",
                             inputs=["lum_sharp"],
                             outputs=["lum_final"]),

                # ── Merge ─────────────────────────────────────────────
                WorkflowStep(kind="merge",
                             note="Merge the sharpened L back with the clean RGB.",
                             lane="main",
                             inputs=["rgb_clean", "lum_final"],
                             outputs=["merged"]),
                WorkflowStep(command_id="curves",
                             note="Final contrast and colour curves.",
                             lane="main",
                             inputs=["merged"],
                             outputs=["final"]),
                WorkflowStep(command_id="save_as",
                             note="Save your result.",
                             lane="main",
                             inputs=["final"]),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────
# Palette presets
# ─────────────────────────────────────────────────────────────

SLAP_PALETTES = {
    "White Light Solar (warm yellow)": {
        "description": "Classic warm yellow-orange for white-light solar images.",
        "r": 1.00, "g": 0.85, "b": 0.55,
    },
    "Hα Solar (deep red)": {
        "description": "Rich red for hydrogen-alpha solar images.",
        "r": 1.00, "g": 0.25, "b": 0.10,
    },
    "Ca-K Solar (violet)": {
        "description": "Blue-violet for calcium K-line solar images.",
        "r": 0.55, "g": 0.55, "b": 1.00,
    },
    "He I 10830 (teal)": {
        "description": "Teal-green for helium 10830 Å infrared solar images.",
        "r": 0.30, "g": 0.90, "b": 0.80,
    },
    "Lunar (natural grey)": {
        "description": "Near-neutral grey with a very slight warm cast — natural lunar look.",
        "r": 1.00, "g": 0.97, "b": 0.90,
    },
    "Lunar (false colour)": {
        "description": "False-colour blue-grey to bring out mare / highland contrast.",
        "r": 0.75, "g": 0.85, "b": 1.00,
    },
    "Planetary (natural)": {
        "description": "Neutral — no colour shift, rely on the RGB data.",
        "r": 1.00, "g": 1.00, "b": 1.00,
    },
    "Jupiter (warm)": {
        "description": "Slightly warm tone to flatter Jupiter's belt colours.",
        "r": 1.00, "g": 0.92, "b": 0.78,
    },
    "Mars (rust)": {
        "description": "Rust-red palette for Mars.",
        "r": 1.00, "g": 0.60, "b": 0.35,
    },
}

# ─────────────────────────────────────────────────────────────
# Per-mode recommended tool settings
# ─────────────────────────────────────────────────────────────

_MODE_TIPS: Dict[str, Dict[str, str]] = {
    "Solar — Surface": {
        "Deconvolution": "Richardson-Lucy, PSF 1.5–2 px, 20–30 iterations, no regularization.",
        "Multiscale": "4 layers, Base σ 1.0. Gain layers 1–3 → 1.3–2.0. Layer 4 → 0.8. NR 0.1–0.2 on layer 4.",
        "LHE / CLAHE": "Kernel 64–128 px. Contrast limit 2.0–3.0. Brings out granulation beautifully.",
        "WaveScale": "Not typically needed for surface-only. Use for limb brightening correction.",
        "Colorize": "White Light: warm yellow.  Hα: deep red.  Ca-K: violet.",
        "Curves": "Gentle inverted-V: lift midtones, keep shadows and bright limb natural.",
    },
    "Solar — Surface + Prominence": {
        "Deconvolution": "Richardson-Lucy, PSF 1.5 px, 20 iterations. Apply only to a surface mask.",
        "Multiscale": "5 layers. Boost layers 1–3 for surface; keep layers 4–5 at Gain 1.0 for prominence.",
        "LHE / CLAHE": "Use Surface mask to limit contrast boost to the disc only.",
        "WaveScale": "Key tool here. Mask Gamma 5–8 keeps effect outside the bright disc. "
                     "Compression 1.5–2.0 to lift faint prominences.",
        "Colorize": "Hα palette. Consider separate colourisation of disc vs. prominence.",
        "Curves": "Two-zone: S-curve for disc, gentle lift for prominence region.",
    },
    "Solar — Prominence Only": {
        "Deconvolution": "Not usually needed — prominence data is diffuse. Skip or use very light RL.",
        "Multiscale": "3–4 layers. Very gentle gain (1.1–1.3) on layers 1–2. Strong NR on layers 3–4.",
        "LHE / CLAHE": "Small kernel (32–64 px), low contrast limit (1.5). Avoid crushing faint tips.",
        "WaveScale": "Low Mask Gamma (1–2) to spread the enhancement into faint regions. "
                     "Compression 2.0–3.0.",
        "Colorize": "Deep red / orange for Hα. Optionally invert for a dramatic black-disc look.",
        "Curves": "Gentle — protect faint prominence tips from clipping.",
    },
    "Lunar": {
        "Deconvolution": "Richardson-Lucy is very effective on lunar data. PSF 1.5–3 px, "
                         "20–50 iterations. Use de-ring to suppress ringing at crater rims.",
        "Multiscale": "5–6 layers. Strong gain (1.5–3.0) on layers 1–4 for fine surface texture. "
                      "Heavy NR (0.3–0.5) on layer 5+.",
        "LHE / CLAHE": "Large kernel (128–256 px), moderate contrast (2.0–4.0). "
                        "Balances highland/mare dynamic range.",
        "WaveScale": "Useful for compressing the very bright limb or terminator zone. "
                     "Keep Mask Gamma moderate (3–5).",
        "Colorize": "Natural grey or false-colour blue. Colour adds little scientifically "
                    "but can be visually striking.",
        "Curves": "Strong S-curve — lunar data tolerates aggressive contrast well.",
    },
    "Planetary": {
        "Deconvolution": "Critical for planetary. PSF 1–2 px, 30–80 iterations. "
                         "RL is best. Deconvolve the L channel before blending with RGB.",
        "Multiscale": "3–4 layers. Very aggressive gain (2.0–4.0) on layers 1–2, "
                      "Gain ≤ 1 on layers 3–4. NR 0.2–0.4 on upper layers. Use Linear mode.",
        "LHE / CLAHE": "Small kernel (16–64 px), contrast limit 2.0–3.5. "
                        "Brings out belt detail and polar cap structure.",
        "WaveScale": "Can help with large-disc targets like Jupiter. "
                     "Compress highlights to prevent the disc from washing out.",
        "Colorize": "Use the natural RGB data. Fine-tune with white balance, not a palette.",
        "Curves": "Targeted S-curve: boost midtones strongly, clip highlights carefully.",
    },
}


# ─────────────────────────────────────────────────────────────
# Section card widget
# ─────────────────────────────────────────────────────────────

class _SectionCard(QGroupBox):
    """
    Collapsible section card for the SLaP panel.
    Header has a collapse toggle; body is any widget.
    """
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFlat(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ──
        header = QWidget(self)
        header.setObjectName("SLaPSectionHeader")
        header.setStyleSheet("""
            #SLaPSectionHeader {
                background: rgba(255,255,255,18);
                border-radius: 4px;
                padding: 2px;
            }
        """)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(6, 4, 6, 4)

        self._toggle = QToolButton(header)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.setFixedSize(18, 18)
        self._toggle.setStyleSheet("border: none; background: transparent;")

        self._title_lbl = QLabel(f"<b>{title}</b>", header)

        h_lay.addWidget(self._toggle)
        h_lay.addWidget(self._title_lbl)
        h_lay.addStretch(1)

        outer.addWidget(header)

        # ── Body ──
        self._body = QWidget(self)
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(8, 6, 8, 8)
        body_lay.setSpacing(6)
        self._body_lay = body_lay

        outer.addWidget(self._body)

        self._toggle.toggled.connect(self._body.setVisible)

    def body_layout(self) -> QVBoxLayout:
        return self._body_lay

    def add_widget(self, w: QWidget):
        self._body_lay.addWidget(w)

    def set_collapsed(self, collapsed: bool):
        self._toggle.setChecked(not collapsed)
        self._body.setVisible(not collapsed)


# ─────────────────────────────────────────────────────────────
# Tool launch button
# ─────────────────────────────────────────────────────────────

class _LaunchButton(QPushButton):
    """
    Styled launch button for a SLaP tool section.
    Shows tool name + a brief one-line description beneath.
    """
    def __init__(self, label: str, tip: str, parent=None):
        super().__init__(parent)
        self.setToolTip(tip)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(1)

        lbl_name = QLabel(f"<b>{label}</b>", self)
        lbl_name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        lbl_tip = QLabel(f"<span style='font-size:10px;color:#aaa;'>{tip}</span>", self)
        lbl_tip.setWordWrap(True)
        lbl_tip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        lay.addWidget(lbl_name)
        lay.addWidget(lbl_tip)


# ─────────────────────────────────────────────────────────────
# Main SLaP Toolkit Dialog
# ─────────────────────────────────────────────────────────────

class SLaPToolkitDialog(QDialog):
    """
    Solar, Lunar, and Planetary (SLaP) Toolkit

    A floating non-modal panel that groups all planetary/solar/lunar
    processing tools in one place with mode-appropriate tips, palette
    presets, and one-click access to every relevant SASpro tool.
    """

    def __init__(self, main, parent=None):
        super().__init__(parent or main)
        self.main = main
        self.settings = getattr(main, "settings", None)

        self.setWindowTitle("SLaP Toolkit  —  Solar, Lunar, and Planetary")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.resize(520, 820)

        self._build_ui()
        self._restore_geometry()
        self._on_mode_changed(0)

    # ─────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Mode selector ────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("<b>Mode:</b>"))
        self.combo_mode = QComboBox(self)
        self.combo_mode.addItems(SLAP_MODES)
        self.combo_mode.setToolTip(
            "Select the type of solar-system target you are processing. "
            "Tips and recommended settings will update automatically."
        )
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.combo_mode, 1)

        self.btn_open_workflow = QPushButton("Open in Workflow Assistant", self)
        self.btn_open_workflow.setToolTip(
            "Load the matching canned workflow for this mode into the Workflow Assistant."
        )
        self.btn_open_workflow.clicked.connect(self._open_workflow)
        mode_row.addWidget(self.btn_open_workflow)

        root.addLayout(mode_row)

        # ── Mode tip banner ──────────────────────────────────
        self.lbl_mode_tip = QLabel("", self)
        self.lbl_mode_tip.setWordWrap(True)
        self.lbl_mode_tip.setStyleSheet(
            "background: rgba(80,160,255,30); border-radius: 4px; "
            "padding: 6px 8px; font-size: 11px; color: #cde;"
        )
        root.addWidget(self.lbl_mode_tip)

        # ── Scrollable tool sections ─────────────────────────
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget(self)
        self._scroll_lay = QVBoxLayout(scroll_widget)
        self._scroll_lay.setContentsMargins(0, 0, 0, 0)
        self._scroll_lay.setSpacing(6)
        scroll.setWidget(scroll_widget)
        root.addWidget(scroll, 1)

        # Build all sections (they store references so we can show/hide tips)
        self._build_sections()
        self._scroll_lay.addStretch(1)

        # ── Bottom bar ───────────────────────────────────────
        bot = QHBoxLayout()
        self.lbl_status = QLabel("Ready.", self)
        self.lbl_status.setStyleSheet("color: #888; font-size: 10px;")

        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.close)

        bot.addWidget(self.lbl_status, 1)
        bot.addWidget(btn_close)
        root.addLayout(bot)

    def _build_sections(self):
        """Build all collapsible tool sections once; tips update on mode change."""

        # ── 1. Pre-processing ────────────────────────────────
        sec_pre = _SectionCard("1 · Pre-processing", self)
        self._tip_pre = QLabel("", self)
        self._tip_pre.setWordWrap(True)
        self._tip_pre.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_pre.add_widget(self._tip_pre)

        row_pre = QHBoxLayout()
        btn_crop = _LaunchButton("Crop", "Remove stacking edges and border artefacts.")
        btn_crop.clicked.connect(lambda: self._trigger("crop"))
        btn_pedestal = _LaunchButton("Pedestal", "Remove the minimum black pedestal value.")
        btn_pedestal.clicked.connect(lambda: self._trigger("pedestal"))
        row_pre.addWidget(btn_crop)
        row_pre.addWidget(btn_pedestal)
        sec_pre.body_layout().addLayout(row_pre)

        self._scroll_lay.addWidget(sec_pre)

        # ── 2. Deconvolution ─────────────────────────────────
        sec_decon = _SectionCard("2 · Deconvolution", self)
        self._tip_decon = QLabel("", self)
        self._tip_decon.setWordWrap(True)
        self._tip_decon.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_decon.add_widget(self._tip_decon)

        btn_decon = _LaunchButton(
            "Convolution / Deconvolution",
            "Richardson-Lucy deconvolution to restore diffraction-limited detail."
        )
        btn_decon.clicked.connect(lambda: self._trigger("convo"))
        sec_decon.add_widget(btn_decon)

        self._scroll_lay.addWidget(sec_decon)

        # ── 3. Multiscale Sharpening ─────────────────────────
        sec_ms = _SectionCard("3 · Multiscale Sharpening", self)
        self._tip_ms = QLabel("", self)
        self._tip_ms.setWordWrap(True)
        self._tip_ms.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_ms.add_widget(self._tip_ms)

        btn_ms = _LaunchButton(
            "Multiscale Decomposition",
            "Per-layer gain, threshold, and NR — the most powerful sharpening tool in SASpro."
        )
        btn_ms.clicked.connect(lambda: self._trigger("multiscale_decomp"))
        sec_ms.add_widget(btn_ms)

        self._scroll_lay.addWidget(sec_ms)

        # ── 4. Contrast Enhancement ──────────────────────────
        sec_ce = _SectionCard("4 · Contrast Enhancement", self)
        self._tip_ce = QLabel("", self)
        self._tip_ce.setWordWrap(True)
        self._tip_ce.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_ce.add_widget(self._tip_ce)

        row_ce = QHBoxLayout()
        btn_clahe = _LaunchButton("CLAHE / LHE", "Local histogram equalisation — essential for solar and lunar.")
        btn_clahe.clicked.connect(lambda: self._trigger("clahe"))
        btn_wavescale = _LaunchButton("WaveScale Dark Enhancer", "Wavelet-based local contrast + prominence boost.")
        btn_wavescale.clicked.connect(lambda: self._trigger("wavescale_dark_enhance"))
        row_ce.addWidget(btn_clahe)
        row_ce.addWidget(btn_wavescale)
        sec_ce.body_layout().addLayout(row_ce)

        btn_freqsep = _LaunchButton(
            "Frequency Separation",
            "Split LF/HF for targeted processing of large-scale gradients and fine detail separately."
        )
        btn_freqsep.clicked.connect(lambda: self._trigger("freqsep"))
        sec_ce.add_widget(btn_freqsep)

        self._scroll_lay.addWidget(sec_ce)

        # ── 5. Noise Reduction ───────────────────────────────
        sec_nr = _SectionCard("5 · Noise Reduction", self)
        self._tip_nr = QLabel("", self)
        self._tip_nr.setWordWrap(True)
        self._tip_nr.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_nr.add_widget(self._tip_nr)

        btn_cc = _LaunchButton(
            "Cosmic Clarity",
            "AI-powered denoise and sharpen — use denoise mode for solar/planetary noise reduction."
        )
        btn_cc.clicked.connect(lambda: self._trigger("cosmicclarity"))
        sec_nr.add_widget(btn_cc)

        self._scroll_lay.addWidget(sec_nr)

        # ── 6. Tone & Colour ─────────────────────────────────
        sec_tone = _SectionCard("6 · Tone and Colour", self)
        self._tip_tone = QLabel("", self)
        self._tip_tone.setWordWrap(True)
        self._tip_tone.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_tone.add_widget(self._tip_tone)

        btn_curves = _LaunchButton(
            "Curves",
            "S-curve for contrast, inverted-V for solar midtone lift. "
            "Works identically to Photoshop curves."
        )
        btn_curves.clicked.connect(lambda: self._trigger("curves"))
        sec_tone.add_widget(btn_curves)

        btn_wb = _LaunchButton("White Balance", "Balance RGB channels before or after palette application.")
        btn_wb.clicked.connect(lambda: self._trigger("white_balance"))
        sec_tone.add_widget(btn_wb)

        btn_sellum = _LaunchButton(
            "Selective Luminance",
            "Apply colour/contrast adjustments only to a specific luminance band — "
            "ideal for taming clipped solar disc vs. dark space."
        )
        btn_sellum.clicked.connect(lambda: self._trigger("selective_lum"))
        sec_tone.add_widget(btn_sellum)

        self._scroll_lay.addWidget(sec_tone)

        # ── 7. Colorization ──────────────────────────────────
        sec_col = _SectionCard("7 · Colorization", self)
        self._tip_col = QLabel("", self)
        self._tip_col.setWordWrap(True)
        self._tip_col.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_col.add_widget(self._tip_col)

        # Palette picker
        pal_row = QHBoxLayout()
        pal_row.addWidget(QLabel("Palette preset:"))
        self.combo_palette = QComboBox(self)
        self.combo_palette.addItems(list(SLAP_PALETTES.keys()))
        self.combo_palette.currentTextChanged.connect(self._on_palette_changed)
        pal_row.addWidget(self.combo_palette, 1)
        sec_col.body_layout().addLayout(pal_row)

        self.lbl_palette_desc = QLabel("", self)
        self.lbl_palette_desc.setWordWrap(True)
        self.lbl_palette_desc.setStyleSheet("font-size: 10px; color: #aaa;")
        sec_col.add_widget(self.lbl_palette_desc)

        btn_mono_rgb = _LaunchButton(
            "Mono to RGB / Colorize",
            "Convert mono image to RGB, then apply curves to colorize with the selected palette."
        )
        btn_mono_rgb.clicked.connect(self._do_mono_to_rgb)
        sec_col.add_widget(btn_mono_rgb)

        self._scroll_lay.addWidget(sec_col)

        # ── 8. Masks ─────────────────────────────────────────
        sec_mask = _SectionCard("8 · Masks", self)
        self._tip_mask = QLabel("", self)
        self._tip_mask.setWordWrap(True)
        self._tip_mask.setStyleSheet("font-size: 10px; color: #aaa; margin: 0 0 4px 0;")
        sec_mask.add_widget(self._tip_mask)

        # Mask creation — opens MaskCreationDialog directly on the active doc
        btn_mask = _LaunchButton(
            "Create Mask",
            "Draw freehand, ellipse, or range-selection masks on the active image."
        )
        btn_mask.clicked.connect(self._do_create_mask)
        sec_mask.add_widget(btn_mask)

        # Disc mask — convenience: full-image ellipse mask pre-sized to the solar/lunar disc
        btn_disc = _LaunchButton(
            "Quick Disc Mask",
            "Open Mask Creation with an ellipse pre-fitted to the image bounds — "
            "ideal for isolating the solar or lunar disc."
        )
        btn_disc.clicked.connect(self._do_disc_mask)
        sec_mask.add_widget(btn_disc)

        # Planetary stacker shortcut
        btn_planet_stack = _LaunchButton(
            "Planetary Stacker",
            "Open the planetary stacker for lucky-imaging stacking of video frames."
        )
        btn_planet_stack.clicked.connect(lambda: self._trigger("planetary_stacker"))
        sec_mask.add_widget(btn_planet_stack)

        self._scroll_lay.addWidget(sec_mask)

        # ── 9. Output ────────────────────────────────────────
        sec_out = _SectionCard("9 · Output", self)
        btn_save = _LaunchButton("Save As", "Save the processed result to disk.")
        btn_save.clicked.connect(lambda: self._trigger("save_as"))
        sec_out.add_widget(btn_save)
        self._scroll_lay.addWidget(sec_out)

        # Store section refs for show/hide logic
        self._sec_decon = sec_decon

    # ─────────────────────────────────────────────────────────
    # Inline tool handlers
    # ─────────────────────────────────────────────────────────

    def _active_doc(self):
        """Mirror of main_window._active_doc() — get the current document."""
        try:
            if hasattr(self.main, "_active_doc"):
                return self.main._active_doc()
            if hasattr(self.main, "mdi") and self.main.mdi.activeSubWindow():
                sw = self.main.mdi.activeSubWindow().widget()
                doc = getattr(sw, "document", None)
                if doc is not None:
                    return doc
            if getattr(self.main, "docman", None) and self.main.docman._docs:
                return self.main.docman._docs[-1]
        except Exception:
            pass
        return None

    def _do_mono_to_rgb(self):
        """
        Convert the active mono image to RGB via the registered action,
        then immediately open Curves so the user can apply a colour palette
        using the selected SLaP palette as a reference tip.
        """
        # Step 1: convert mono → RGB using the registered action
        from setiastro.saspro.workflows import _find_action_by_command_id
        act = _find_action_by_command_id(self.main, "mono_to_rgb")
        if act is None:
            QMessageBox.warning(
                self, "SLaP Toolkit",
                "Could not find 'Convert Mono to RGB' action.\n"
                "Make sure act_mono_to_rgb is registered with command_id 'mono_to_rgb'."
            )
            return
        try:
            act.trigger()
            self._status("Converted to RGB — open Curves to apply palette colours.")
        except Exception as e:
            QMessageBox.critical(self, "SLaP Toolkit", f"Mono to RGB failed:\n{e}")
            return

        # Step 2: show a tip about the currently selected palette
        pal_name = self.combo_palette.currentText()
        pal = SLAP_PALETTES.get(pal_name, {})
        r, g, b = pal.get("r", 1.0), pal.get("g", 1.0), pal.get("b", 1.0)
        tip = (
            f"Palette tip — {pal_name}:\n"
            f"In Curves, set per-channel output midpoints to approx:\n"
            f"  R = {r:.2f}   G = {g:.2f}   B = {b:.2f}\n\n"
            f"{pal.get('description', '')}"
        )
        QMessageBox.information(self, "SLaP Toolkit — Palette Reference", tip)

    def _do_create_mask(self):
        """Open MaskCreationDialog on the active document."""
        doc = self._active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "SLaP Toolkit", "Open an image first.")
            return
        try:
            from setiastro.saspro.mask_creation import MaskCreationDialog
            dlg = MaskCreationDialog(doc.image, parent=self.main, auto_push_on_ok=True)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dlg.show()
            self._status("Mask creation opened.")
        except Exception as e:
            QMessageBox.critical(self, "SLaP Toolkit", f"Could not open Mask Creation:\n{e}")

    def _do_disc_mask(self):
        """
        Open MaskCreationDialog with an ellipse pre-fitted to the full image bounds.
        Perfect starting point for a solar or lunar disc mask — the user can then
        resize and reposition the ellipse to match the actual disc.
        """
        doc = self._active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, "SLaP Toolkit", "Open an image first.")
            return
        try:
            from setiastro.saspro.mask_creation import MaskCreationDialog
            from PyQt6.QtCore import QRectF, QTimer as _QTimer

            dlg = MaskCreationDialog(doc.image, parent=self.main, auto_push_on_ok=True)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

            def _pre_fit_ellipse():
                """
                After the canvas is visible, switch to Ellipse mode and add an
                ellipse covering ~90 % of the image — a good starting disc mask.
                """
                try:
                    from PyQt6.QtCore import QRectF, QPointF
                    from setiastro.saspro.mask_creation import InteractiveEllipseItem
                    from PyQt6.QtGui import QBrush
                    from PyQt6.QtCore import Qt as _Qt

                    canvas = dlg.canvas
                    canvas.set_mode("ellipse")
                    dlg.ellipse_btn.setChecked(True)

                    h, w = doc.image.shape[:2]
                    margin_x = w * 0.05
                    margin_y = h * 0.05
                    rect = QRectF(margin_x, margin_y,
                                  w - 2 * margin_x, h - 2 * margin_y)

                    ell = InteractiveEllipseItem(
                        QRectF(0, 0, rect.width(), rect.height())
                    )
                    ell.setBrush(QBrush(_Qt.BrushStyle.NoBrush))
                    ell.setZValue(1)
                    ell.setPos(rect.topLeft())
                    canvas.scene.addItem(ell)
                    canvas.shapes.append(ell)
                    canvas.fit_to_view()
                    dlg._invalidate_base_mask()
                except Exception:
                    pass  # non-fatal — user can draw manually

            dlg.show()
            _QTimer.singleShot(150, _pre_fit_ellipse)
            self._status("Disc mask opened — resize the ellipse to match your disc.")
        except Exception as e:
            QMessageBox.critical(self, "SLaP Toolkit", f"Could not open Disc Mask:\n{e}")

    # ─────────────────────────────────────────────────────────
    # Mode logic
    # ─────────────────────────────────────────────────────────

    def _on_mode_changed(self, idx: int):
        mode = SLAP_MODES[idx]
        tips = _MODE_TIPS.get(mode, {})

        # Banner
        mode_summaries = {
            "Solar — Surface":
                "Surface-only solar processing. Optimised for granulation, sunspots, "
                "and active region detail. Deconvolution and multiscale are the key tools.",
            "Solar — Surface + Prominence":
                "Full-disc solar with prominence enhancement. WaveScale Dark Enhancer "
                "lifts faint filament structure outside the solar limb.",
            "Solar — Prominence Only":
                "Over-exposed prominence data where the disc is clipped. "
                "Emphasis on gentle enhancement of faint extremities without noise amplification.",
            "Lunar":
                "High-contrast lunar surface detail. Richardson-Lucy deconvolution and "
                "strong multiscale sharpening are the workhorses here.",
            "Planetary":
                "Small-disc high-magnification planetary data. L+RGB split workflow — "
                "deconvolve and sharpen L hard, denoise RGB gently, then blend.",
        }
        self.lbl_mode_tip.setText(mode_summaries.get(mode, ""))

        # Per-section tips
        self._tip_pre.setText(
            "Crop first to remove stacking edge artefacts. "
            "Then remove the pedestal to set a clean black point."
        )
        self._tip_decon.setText(tips.get("Deconvolution", ""))
        self._tip_ms.setText(tips.get("Multiscale", ""))
        self._tip_ce.setText(tips.get("LHE / CLAHE", "") + "\n" + tips.get("WaveScale", ""))
        self._tip_nr.setText(tips.get("Cosmicclarity",
            "Cosmic Clarity denoise — keep it light to preserve fine detail."))
        self._tip_tone.setText(tips.get("Curves", ""))
        self._tip_col.setText(tips.get("Colorize", ""))
        self._tip_mask.setText(
            "Use masks to constrain tools to specific image zones — "
            "disc vs. limb vs. space — especially important for prominence work."
        )

        # Auto-select a sensible palette default for the mode
        palette_defaults = {
            "Solar — Surface":              "White Light Solar (warm yellow)",
            "Solar — Surface + Prominence": "Hα Solar (deep red)",
            "Solar — Prominence Only":      "Hα Solar (deep red)",
            "Lunar":                        "Lunar (natural grey)",
            "Planetary":                    "Planetary (natural)",
        }
        default_pal = palette_defaults.get(mode)
        if default_pal and default_pal in SLAP_PALETTES:
            self.combo_palette.blockSignals(True)
            self.combo_palette.setCurrentText(default_pal)
            self.combo_palette.blockSignals(False)
            self._on_palette_changed(default_pal)

        # Deconvolution is less relevant for prominence-only — keep visible
        # but flag in the tip that it's optional (already done via tip text above)

        self._save_mode()

    def _on_palette_changed(self, name: str):
        pal = SLAP_PALETTES.get(name, {})
        self.lbl_palette_desc.setText(pal.get("description", ""))

    # ─────────────────────────────────────────────────────────
    # Tool triggering
    # ─────────────────────────────────────────────────────────

    def _trigger(self, command_id: str):
        """
        Fire a SASpro tool by command_id — same mechanism as the Workflow
        Assistant's _run_selected_step() so the two systems stay in sync.
        """
        from setiastro.saspro.workflows import _find_action_by_command_id
        action = _find_action_by_command_id(self.main, command_id)
        if action is None:
            self._status(f"Could not find tool: {command_id}")
            QMessageBox.warning(
                self,
                "SLaP Toolkit",
                f"Could not find the tool action for command_id:\n\n{command_id}\n\n"
                "Make sure the tool is registered in the SASpro toolbar.",
            )
            return
        try:
            action.trigger()
            self._status(f"Opened: {action.text() or command_id}")
        except Exception as e:
            self._status(f"Error: {e}")
            QMessageBox.critical(self, "SLaP Toolkit", f"Error opening tool:\n\n{e}")

    def _status(self, msg: str):
        self.lbl_status.setText(msg)
        QTimer.singleShot(4000, lambda: self.lbl_status.setText("Ready."))

    # ─────────────────────────────────────────────────────────
    # Workflow integration
    # ─────────────────────────────────────────────────────────

    def _open_workflow(self):
        """
        Load the matching canned workflow for the current mode into
        the Workflow Assistant (full dialog) and bring it to the front.
        """
        mode = self.combo_mode.currentText()
        mode_to_wf_name = {
            "Solar — Surface":              "SLaP — Solar Surface",
            "Solar — Surface + Prominence": "SLaP — Solar Surface + Prominence",
            "Solar — Prominence Only":      "SLaP — Solar Prominence Only",
            "Lunar":                        "SLaP — Lunar",
            "Planetary":                    "SLaP — Planetary",
        }
        wf_name = mode_to_wf_name.get(mode)
        if not wf_name:
            return

        workflows = slap_canned_workflows()
        target = next((w for w in workflows if w.name == wf_name), None)
        if target is None:
            return

        from setiastro.saspro.workflows import show_workflow_dialog
        show_workflow_dialog(self.main)

        dlg = getattr(self.main, "_workflow_dialog", None)
        if dlg is not None:
            dlg.load_workflow_definition(target)
            self._status(f"Loaded workflow: {wf_name}")

    # ─────────────────────────────────────────────────────────
    # Geometry / state persistence
    # ─────────────────────────────────────────────────────────

    def _save_mode(self):
        if self.settings is None:
            return
        try:
            self.settings.setValue(
                "slap_toolkit/mode",
                self.combo_mode.currentText()
            )
        except Exception:
            pass

    def _restore_geometry(self):
        if self.settings is None:
            return
        try:
            g = self.settings.value("slap_toolkit/geometry")
            if g:
                self.restoreGeometry(g)
            mode = self.settings.value("slap_toolkit/mode", "")
            if mode in SLAP_MODES:
                self.combo_mode.setCurrentText(mode)
        except Exception:
            pass

    def _save_geometry(self):
        if self.settings is None:
            return
        try:
            self.settings.setValue("slap_toolkit/geometry", self.saveGeometry())
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_geometry()
        try:
            if getattr(self.main, "_slap_dialog", None) is self:
                self.main._slap_dialog = None
        except Exception:
            pass
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def show_slap_toolkit(main):
    """
    Open (or raise) the SLaP Toolkit panel.
    Call this from main_window.py:

        from setiastro.saspro.slap_toolkit import show_slap_toolkit
        show_slap_toolkit(self)

    And wire it to a menu/toolbar action with command_id = "slap_toolkit".
    """
    dlg = getattr(main, "_slap_dialog", None)
    if dlg is None:
        dlg = SLaPToolkitDialog(main, parent=main)
        main._slap_dialog = dlg

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()