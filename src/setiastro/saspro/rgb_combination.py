#saspro/rgb_combination.py
from __future__ import annotations
import os
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QGroupBox, QFileDialog, QComboBox, QButtonGroup, QMessageBox, QFormLayout,
    QCheckBox, QDoubleSpinBox, QFrame, QToolButton, QWidget, QTextBrowser,
    QDialogButtonBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize

try:
    from setiastro.saspro.legacy.image_manager import load_image
except Exception:
    load_image = None

from setiastro.saspro.help_support import make_help_button

# Luminance machinery is shared with the standalone Recombine Luminance tool,
# so the Advanced section here exposes exactly the same knobs and produces
# bit-identical results for the same settings.
from setiastro.saspro.luminancerecombine import (
    recombine_luminance_linear_scale,
    compute_luminance,
    resolve_luma_profile_weights,
    iter_luma_profiles_by_category,
    _estimate_noise_sigma_per_channel,
    _boost_saturation,
    _chrominance_nr,
    _LUMA_REC601,
    _LUMA_REC2020,
    _LUMA_REC709,
)

# The house Linear Fit engine, so the "fit L first" option here behaves
# exactly like running Linear Fit from the menu rather than being a second,
# subtly different implementation.
from setiastro.saspro.linear_fit import linear_fit_mono_to_ref


def _to_f01(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind in "ui":
        a = a.astype(np.float32)
    elif a.dtype.kind == "f" and a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.size:
        m = float(np.nanmax(a))
        if m > 1.0 and np.isfinite(m):
            a = a / m
    return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)


def _mono_from_any(arr: np.ndarray) -> np.ndarray:
    a = _to_f01(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] >= 1:
        return a[:, :, 0]
    raise ValueError("Expected mono (HxW) or RGB (HxWx3) image.")


def _as_is_from_any(arr: np.ndarray) -> np.ndarray:
    """Normalise to float32 [0,1] but KEEP the channel count.

    Used for the luminance source: if the user points L at a colour image we
    want to run it through compute_luminance with the chosen profile, not
    silently take channel 0 the way _mono_from_any would.
    """
    a = _to_f01(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] >= 1:
        return a
    raise ValueError("Expected mono (HxW) or multi-channel (HxWxC) image.")


# ── Warning copy ──────────────────────────────────────────────────────────
# Kept at module scope so the short box and the long explanation can't drift
# apart, and so it's translatable in one place.
_L_WARNING_SHORT = (
    "Do not combine a LINEAR L with LINEAR RGB."
)
_L_WARNING_BODY = (
    "Unless you have already matched the two stacks — total integration, "
    "sub-exposure length, pedestal/offset, filter throughput and camera "
    "response — the scale factor this tool applies is meaningless. You will "
    "get blown highlights or crushed shadows, and badly desaturated colour.\n\n"
    "Normal workflow: combine RGB here WITHOUT L, stretch it, stretch L "
    "separately, then use Recombine Luminance on the stretched pair."
)

_L_WARNING_HTML = """
<h3 style="margin-top:0;">Why linear LRGB goes wrong</h3>

<p>Recombination computes the RGB image's own luminance</p>
<pre>    Y = wR&middot;R + wG&middot;G + wB&middot;B</pre>
<p>and then rescales every pixel by</p>
<pre>    s = L / Y</pre>
<p>so the result carries L's detail with the RGB's colour.</p>

<p><b>That only works if L and Y sit on the same intensity scale.</b> For
stretched data they usually do, because stretching pushes both into roughly
the display range. For linear data they almost never do:</p>

<ul>
<li><b>Total integration</b> &mdash; an L stack is often several times deeper
    than any single colour channel. More signal, higher mean level.</li>
<li><b>Sub-exposure length</b> &mdash; different sub lengths scale per-pixel
    counts directly.</li>
<li><b>Filter throughput</b> &mdash; an L or clear filter passes far more
    photons than any one of R, G or B over the same period.</li>
<li><b>Camera response</b> &mdash; QE varies with wavelength, so identical
    photon flux yields different electron counts per channel.</li>
<li><b>Pedestal and offset</b> &mdash; calibration pedestals differ between
    stacks. Near the background, <code>s = L/Y</code> is dominated by the
    difference in offsets rather than by any real signal.</li>
<li><b>Stacking normalisation</b> &mdash; most integration routines rescale
    each stack to its own reference frame, destroying whatever cross-stack
    relationship survived calibration.</li>
</ul>

<p>The result is that <code>s</code> is roughly a constant <i>wrong</i> number
across the whole frame. Too high and the highlights blow and the colour washes
out; too low and everything crushes. Either way chroma suffers, because colour
lives in the <i>ratios</i> between R, G and B and those only survive when the
overall scale is right.</p>

<h3>What to do instead</h3>
<ol>
<li>Combine R, G and B here with the L option switched off.</li>
<li>Stretch the RGB result.</li>
<li>Stretch the L image separately, to a similar visual brightness.</li>
<li>Run <b>Recombine Luminance</b> on the stretched pair.</li>
</ol>

<p>If you must work linear, fit L to the RGB luminance first. The Advanced
section has a <b>Linear-fit L to RGB luminance</b> option that runs the same
Linear Fit engine as the menu command, scaling L so its median matches the
combined RGB's luminance before recombining.</p>

<p>That handles the <i>scale</i> error &mdash; the first four causes above.
It does not handle the last two. The fit is a pure multiply with no offset
term, so a <b>pedestal difference survives it</b>; subtract your pedestals
before combining if the two stacks were calibrated differently. And no scalar
fit can correct <b>spectral response</b>: an L filter sees a wider band than R,
G or B, so even a perfectly fitted L carries structure the colour channels
never recorded, which is what produces the flat, washed-out look people blame
on "too much L".</p>

<h3>If your data is already stretched</h3>
<p>Then none of the above applies and this is a perfectly ordinary LRGB
combine. The warning is about linear data specifically.</p>
"""


class _WrapLabel(QLabel):
    """A word-wrapped QLabel that reports an honest height.

    QLabel.sizeHint() for wrapped text is computed against a width Qt guesses
    (roughly a golden-ratio box), not the width the label actually gets. In a
    fixed-width column that guess is too wide, the hint comes back too short,
    and the tail of the text is silently clipped — which is exactly what
    happened to the second paragraph of the luminance warning once it moved
    into a side column.

    Re-asserting minimumHeight from heightForWidth() on every resize makes the
    layout allocate the room the text really needs, at whatever width and in
    whatever language it ends up.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.MinimumExpanding)

    def minimumSizeHint(self) -> QSize:
        s = super().minimumSizeHint()
        w = self.width()
        if w > 0:
            return QSize(s.width(), max(s.height(), self.heightForWidth(w)))
        return s

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        w = self.width()
        if w > 0:
            need = self.heightForWidth(w)
            if need > 0 and need != self.minimumHeight():
                self.setMinimumHeight(need)


class _LWarningDialog(QDialog):
    """Scrollable long-form explanation behind the warning button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Luminance in linear data"))
        self.resize(640, 560)
        v = QVBoxLayout(self)
        tb = QTextBrowser(self)
        tb.setOpenExternalLinks(True)
        tb.setHtml(_L_WARNING_HTML)
        v.addWidget(tb, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        v.addWidget(bb)


class RGBCombinationDialogPro(QDialog):
    """
    Combine R/G/B (and optionally L) from open views or files and ALWAYS
    create a new document.
    """

    # Explicit fg AND bg so the warning reads the same under light and dark
    # themes rather than inheriting whatever the palette happens to be.
    _WARN_CSS = (
        "QFrame#lWarnBox {"
        "  background:#fff8d0;"
        "  border:2px solid #d4a017;"
        "  border-radius:4px;"
        "}"
        "QFrame#lWarnBox QLabel { color:#4a3800; background:transparent; }"
    )

    def __init__(self, parent, list_open_docs_fn=None, doc_manager=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("LRGB Combination"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        #self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._list_open_docs = list_open_docs_fn or (lambda: [])
        self._docman = doc_manager

        # file-mode state
        self.r_path = None
        self.g_path = None
        self.b_path = None
        self.l_path = None

        # ── mode choose
        self.load_files_radio = QRadioButton(self.tr("Load Individual Files"))
        self.use_views_radio  = QRadioButton(self.tr("Use Open Views"))
        self.use_views_radio.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.load_files_radio)
        self.mode_group.addButton(self.use_views_radio)

        mode_box = QGroupBox(self.tr("Select LRGB Combination Mode"))
        ml = QVBoxLayout(mode_box)
        ml.addWidget(self.use_views_radio)
        ml.addWidget(self.load_files_radio)

        # ── file mode widgets
        self.r_label = QLabel(self.tr("Red: Not selected"))
        self.g_label = QLabel(self.tr("Green: Not selected"))
        self.b_label = QLabel(self.tr("Blue: Not selected"))
        self.l_label = QLabel(self.tr("Luminance: Not selected"))
        self.btn_load_r = QPushButton(self.tr("Load Red…"));   self.btn_load_r.clicked.connect(lambda: self._pick_file("R"))
        self.btn_load_g = QPushButton(self.tr("Load Green…")); self.btn_load_g.clicked.connect(lambda: self._pick_file("G"))
        self.btn_load_b = QPushButton(self.tr("Load Blue…"));  self.btn_load_b.clicked.connect(lambda: self._pick_file("B"))
        self.btn_load_l = QPushButton(self.tr("Load Luminance…")); self.btn_load_l.clicked.connect(lambda: self._pick_file("L"))

        file_box = QGroupBox(self.tr("Files"))
        fl = QVBoxLayout(file_box)
        for lab, btn in [(self.l_label, self.btn_load_l),
                         (self.r_label, self.btn_load_r),
                         (self.g_label, self.btn_load_g),
                         (self.b_label, self.btn_load_b)]:
            row = QHBoxLayout(); row.addWidget(lab); row.addStretch(); row.addWidget(btn)
            fl.addLayout(row)

        # ── open-views widgets
        views_box = QGroupBox(self.tr("Select Open Views for L / R / G / B"))
        vf = QFormLayout(views_box)

        self.cmb_r = QComboBox()
        self.cmb_g = QComboBox()
        self.cmb_b = QComboBox()
        self.cmb_l = QComboBox()

        # optional: make long titles usable without widening the dialog
        for cmb in (self.cmb_r, self.cmb_g, self.cmb_b, self.cmb_l):
            cmb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(24)  # tweak if you want

        # L first, so the rows read in the conventional LRGB order.
        self._lum_view_row_label = QLabel(self.tr("Luminance:"))
        vf.addRow(self._lum_view_row_label, self.cmb_l)
        vf.addRow(self.tr("Red:"), self.cmb_r)
        vf.addRow(self.tr("Green:"), self.cmb_g)
        vf.addRow(self.tr("Blue:"), self.cmb_b)

        # ── luminance section
        lum_box = self._build_luminance_box()

        # ── buttons
        self.btn_combine = QPushButton(self.tr("Combine"))
        self.btn_cancel  = QPushButton(self.tr("Cancel"))
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_combine.clicked.connect(self._combine)

        btns = QHBoxLayout()
        self.btn_help = make_help_button("rgb_combination", self)
        btns.addWidget(self.btn_help)
        btns.addStretch(); btns.addWidget(self.btn_combine); btns.addWidget(self.btn_cancel)

        # ── layout: two columns, so the luminance warning + advanced options
        # sit beside the source pickers instead of stacking the dialog to
        # ~1000px tall. Buttons span the full width underneath.
        col_left = QVBoxLayout()
        col_left.setContentsMargins(0, 0, 0, 0)
        col_left.addWidget(mode_box)
        col_left.addWidget(views_box)
        col_left.addWidget(file_box)
        col_left.addStretch(1)

        col_right = QVBoxLayout()
        col_right.setContentsMargins(0, 0, 0, 0)
        col_right.addWidget(lum_box)
        col_right.addStretch(1)

        cols = QHBoxLayout()
        cols.setSpacing(10)
        # Left takes its natural width (the combos already set a minimum
        # contents length); the right column absorbs any extra, which is where
        # the wrapped warning text wants it.
        cols.addLayout(col_left, 0)
        cols.addLayout(col_right, 1)

        L = QVBoxLayout(self)
        L.addLayout(cols, 1)
        L.addLayout(btns)

        self.mode_group.buttonClicked.connect(self._refresh_mode)
        self._populate_views()
        self._refresh_mode()
        self._refresh_luminance_enabled()

    # ── luminance UI construction ─────────────────────────────────────

    def _build_luminance_box(self) -> QGroupBox:
        box = QGroupBox(self.tr("Luminance (L)"))
        # Wide enough that the warning paragraphs wrap to a readable measure
        # rather than a narrow ribbon, and that the Advanced rows don't squeeze
        # the profile combo.
        box.setMinimumWidth(400)
        v = QVBoxLayout(box)

        self.chk_use_l = QCheckBox(self.tr("Add a Luminance (L) channel"))
        self.chk_use_l.setToolTip(self.tr(
            "Replace the combined RGB's luminance with a separate L image.\n"
            "Read the warning before using this on LINEAR data."
        ))
        self.chk_use_l.toggled.connect(self._on_use_l_toggled)
        v.addWidget(self.chk_use_l)

        # ── the yellow box ────────────────────────────────────────────
        self.warn_box = QFrame(box)
        self.warn_box.setObjectName("lWarnBox")
        self.warn_box.setFrameShape(QFrame.Shape.StyledPanel)
        self.warn_box.setStyleSheet(self._WARN_CSS)
        wv = QVBoxLayout(self.warn_box)
        wv.setContentsMargins(12, 10, 12, 10)
        wv.setSpacing(6)

        head = _WrapLabel("⚠  " + self.tr(_L_WARNING_SHORT), self.warn_box)
        f = head.font()
        f.setBold(True)
        f.setPointSize(max(11, f.pointSize() + 2))
        head.setFont(f)
        wv.addWidget(head)

        body = _WrapLabel(self.tr(_L_WARNING_BODY), self.warn_box)
        wv.addWidget(body)

        wrow = QHBoxLayout()
        self.btn_l_warning = QPushButton(self.tr("⚠  What this means…"), self.warn_box)
        self.btn_l_warning.setToolTip(self.tr("Full explanation of linear LRGB scaling"))
        self.btn_l_warning.clicked.connect(self._show_l_warning)
        wrow.addWidget(self.btn_l_warning)
        wrow.addStretch(1)
        wv.addLayout(wrow)

        self.warn_box.setVisible(False)
        v.addWidget(self.warn_box)

        # ── advanced disclosure ───────────────────────────────────────
        self.btn_adv = QToolButton(box)
        self.btn_adv.setText(self.tr("Advanced"))
        self.btn_adv.setCheckable(True)
        self.btn_adv.setChecked(False)
        self.btn_adv.setArrowType(Qt.ArrowType.RightArrow)
        self.btn_adv.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn_adv.toggled.connect(self._toggle_advanced)
        arow = QHBoxLayout()
        arow.addWidget(self.btn_adv)
        arow.addStretch(1)
        v.addLayout(arow)

        self.adv_widget = QWidget(box)
        af = QFormLayout(self.adv_widget)
        af.setContentsMargins(18, 4, 0, 0)

        self.cmb_luma_profile = self._build_profile_combo()
        af.addRow(self.tr("Luma profile:"), self.cmb_luma_profile)

        self.spn_blend = QDoubleSpinBox()
        self.spn_blend.setRange(0.0, 1.0); self.spn_blend.setSingleStep(0.05)
        self.spn_blend.setDecimals(2); self.spn_blend.setValue(1.00)
        self.spn_blend.setToolTip(self.tr(
            "1.00 = fully replace the luminance; 0.00 = leave the RGB untouched."))
        af.addRow(self.tr("Blend:"), self.spn_blend)

        self.spn_knee = QDoubleSpinBox()
        self.spn_knee.setRange(0.0, 1.0); self.spn_knee.setSingleStep(0.05)
        self.spn_knee.setDecimals(2); self.spn_knee.setValue(0.00)
        self.spn_knee.setToolTip(self.tr(
            "Soft-knee compression on the per-pixel scale factor, to hold back\n"
            "highlights where L is much brighter than the RGB luminance."))
        af.addRow(self.tr("Highlight soft knee:"), self.spn_knee)

        self.spn_pedestal = QDoubleSpinBox()
        self.spn_pedestal.setRange(0.0, 0.5); self.spn_pedestal.setSingleStep(0.01)
        self.spn_pedestal.setDecimals(3); self.spn_pedestal.setValue(0.050)
        self.spn_pedestal.setToolTip(self.tr(
            "Noise-floor lift applied to both images before scaling, to stop\n"
            "hue skew near zero. 0.000 disables it (pure linear scaling)."))
        af.addRow(self.tr("Pedestal:"), self.spn_pedestal)

        self.spn_sat = QDoubleSpinBox()
        self.spn_sat.setRange(-1.0, 2.0); self.spn_sat.setSingleStep(0.05)
        self.spn_sat.setDecimals(2); self.spn_sat.setValue(0.00)
        self.spn_sat.setToolTip(self.tr(
            "Saturation boost applied to the RGB BEFORE the luminance is\n"
            "measured, so it cannot skew the luma being placed.\n"
            "0 = no change, 1.0 = double saturation."))
        af.addRow(self.tr("Saturation boost:"), self.spn_sat)

        self.spn_cnr = QDoubleSpinBox()
        self.spn_cnr.setRange(0.0, 10.0); self.spn_cnr.setSingleStep(0.5)
        self.spn_cnr.setDecimals(1); self.spn_cnr.setValue(0.0)
        self.spn_cnr.setSuffix(" px")
        self.spn_cnr.setToolTip(self.tr(
            "Chrominance noise reduction: blurs Cb/Cr only, leaving luma\n"
            "untouched. 0 = off."))
        af.addRow(self.tr("Chrominance NR:"), self.spn_cnr)

        self.chk_linfit = QCheckBox(self.tr("Linear-fit L to RGB luminance"))
        self.chk_linfit.setToolTip(self.tr(
            "Runs the standard Linear Fit engine: scales L so its median matches\n"
            "the combined RGB's own luminance, before recombining.\n\n"
            "This is the minimum pre-work for linear data. It removes the gross\n"
            "scale mismatch between the two stacks, but it is a pure multiply —\n"
            "it does NOT subtract a pedestal difference, and it cannot correct\n"
            "for the L filter covering a wider band than R, G or B."))
        af.addRow("", self.chk_linfit)

        self.adv_widget.setVisible(False)
        v.addWidget(self.adv_widget)

        return box

    def _build_profile_combo(self) -> QComboBox:
        """Flat combo with disabled category headers, built from the same
        LUMA_PROFILES registry the standalone tool uses."""
        cmb = QComboBox()
        cmb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        cmb.setMinimumContentsLength(28)

        last_cat = None
        for category_path, key, description, info in iter_luma_profiles_by_category():
            cat = " / ".join(category_path)
            if cat != last_cat:
                cmb.addItem(f"— {cat} —")
                try:
                    cmb.model().item(cmb.count() - 1).setEnabled(False)
                except Exception:
                    pass
                last_cat = cat
            cmb.addItem(f"    {description}", key)
            if info:
                cmb.setItemData(cmb.count() - 1, info, Qt.ItemDataRole.ToolTipRole)

        # Default to Rec.709
        for i in range(cmb.count()):
            if cmb.itemData(i) == "rec709":
                cmb.setCurrentIndex(i)
                break
        cmb.setToolTip(self.tr(
            "Weights used to measure the RGB image's own luminance. If the L\n"
            "source is a colour image, this also selects how L is extracted\n"
            "from it."
        ))
        return cmb

    def _current_profile_key(self) -> str:
        key = self.cmb_luma_profile.currentData()
        return str(key) if key else "rec709"

    # ── luminance UI behaviour ────────────────────────────────────────

    def _on_use_l_toggled(self, on: bool):
        self.warn_box.setVisible(bool(on))
        self._refresh_luminance_enabled()
        self.adjustSize()

    def _toggle_advanced(self, on: bool):
        self.adv_widget.setVisible(bool(on))
        self.btn_adv.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.adjustSize()

    def _refresh_luminance_enabled(self):
        use_l = self.chk_use_l.isChecked()
        use_views = self.use_views_radio.isChecked()
        self.cmb_l.setEnabled(use_l and use_views)
        self._lum_view_row_label.setEnabled(use_l and use_views)
        self.l_label.setEnabled(use_l and not use_views)
        self.btn_load_l.setEnabled(use_l and not use_views)
        self.btn_adv.setEnabled(use_l)
        self.adv_widget.setEnabled(use_l)

    def _show_l_warning(self):
        dlg = _LWarningDialog(self)
        dlg.exec()

    # --- helpers

    # ─── Filter → channel map ──────────────────────────────────────────
    # Used by _filter_bonus below to score the FITS FILTER keyword, which is
    # the DEFINITIVE signal for which channel a stack belongs to — unlike
    # filename patterns, a FILTER card can't masquerade as something else.
    #
    # Weights: direct broadband R/G/B/L get 100, Sloan variants 90, narrowband
    # aliases 50-60. Lower narrowband weight means "if both an R and an Ha
    # file exist, prefer R for the red role" — the Ha file still beats a
    # file with no FILTER card, but a true R beats it.
    #
    # OIII is intentionally NOT mapped: users split it G or B depending on
    # palette (HOO, SHO), so we leave it to the filename regex / user choice.
    @staticmethod
    def _map_filter(f: str):
        """
        Return (channel, weight) for a FITS FILTER keyword value, or None if
        it maps to nothing we can act on. Case/space/underscore-insensitive.
        """
        if not f:
            return None
        s = str(f).strip().upper().replace(" ", "").replace("_", "").replace("-", "")
        EXACT = {
            "R":   ("R", 100), "RED":   ("R", 100),
            "G":   ("G", 100), "GREEN": ("G", 100),
            "B":   ("B", 100), "BLUE":  ("B", 100),
            # L is a real role now that this dialog can take a luminance
            # channel. Mapping it also means an L stack gets PENALISED for
            # the R/G/B roles, which is what we want — it shouldn't win red
            # just because nothing else had a FILTER card.
            "L":   ("L", 100), "LUM":   ("L", 100), "LUMINANCE": ("L", 100),
            "CLEAR": ("L", 80), "C":    ("L", 70), "NONE": ("L", 60),
            # Sloan short and prime notations
            "SR":  ("R", 90),  "SG": ("G", 90),  "SB": ("B", 90),
            "R'":  ("R", 90),  "G'": ("G", 90),  "B'": ("B", 90),
            # Common narrowband mappings (weaker than broadband so true R wins)
            "HA":  ("R", 60),  "HALPHA": ("R", 60),
            "SII": ("R", 55),  "S2":     ("R", 55),
        }
        if s in EXACT:
            return EXACT[s]
        # Filter-wheel entries like "Ha_7nm", "Halpha6", "S2narrow"
        if s.startswith(("HALPHA", "HA")):
            return ("R", 55)
        if s.startswith(("SII", "S2")):
            return ("R", 50)
        if s.startswith(("LUMINANCE", "LUM")):
            return ("L", 90)
        return None

    @staticmethod
    def _filter_from_doc(doc) -> str:
        """
        Pull the FITS FILTER keyword value off a doc's metadata. Returns the
        raw string (not uppercased — _map_filter normalises). Checks the
        common storage patterns SASpro uses: a flat metadata key, or a
        nested astropy Header under one of a few names.
        """
        if doc is None:
            return ""
        try:
            meta = getattr(doc, "metadata", None) or {}
        except Exception:
            return ""
        # Direct flat key
        for key in ("FILTER", "filter", "Filter"):
            v = meta.get(key)
            if v:
                return str(v)
        # Nested FITS/WCS header
        for hkey in ("original_header", "fits_header", "header", "wcs_header"):
            h = meta.get(hkey)
            if h is None:
                continue
            try:
                v = h.get("FILTER") if hasattr(h, "get") else h["FILTER"]
                if v:
                    return str(v)
            except Exception:
                continue
        return ""

    def _guess_channel(self, titles: list[str], docs: list, channel: str) -> int:
        """
        Score each (title, doc) pair for how likely it is to be the given
        channel (R/G/B/L), and return the index of the best match. Falls back
        to 0 only if nothing scored positive.

        Scoring is two-layer:
          1. FITS FILTER keyword — definitive when present. A card reading
             "G" gets +100 for green and -50 for red/blue.
          2. Filename regex — fallback for files whose FILTER card is
             missing or non-standard.

        The filename regex used to have a bug: `[_\\-\\s\\.]g[_\\-\\s\\.\\d]`
        allowed `\\d` as the trailing boundary, which let the SASpro gain
        prefix (`..._G121_...` on every master-light file) false-match as a
        green channel. Every file in a field tied at the same score, and the
        first one in the list won — which is why Green rarely auto-assigned
        correctly while R and B almost always did (there's no `_R<n>_` or
        `_B<n>_` convention to collide with). Removing `\\d` fixes it: only
        the real `_G_` separator matches now.
        """
        import re
        channel = channel.upper()

        def filter_bonus(doc) -> int:
            entry = self._map_filter(self._filter_from_doc(doc))
            if entry is None:
                return 0
            mapped, weight = entry
            if mapped == channel:
                return int(weight)
            # Mapped to a DIFFERENT channel — penalise so it doesn't win
            # the role it isn't right for. Half-weight: a FILTER=Ha file
            # still scores -30 for green, which the real _G_ filename
            # pattern (+8) will overcome, so the right file still wins.
            return -int(weight) // 2

        def filename_score(title: str) -> int:
            t = title.lower()
            s = 0

            if channel == "R":
                if re.search(r'\bred\b', t):               s += 10
                if re.search(r'[_\-\s\.]r[_\-\s\.]', t):   s += 8   # was [...\d] — see docstring
                if re.search(r'[_\-\s\.]ha[_\-\s\.]', t):  s += 6
                if re.search(r'[_\-\s\.]sii[_\-\s\.]', t): s += 5
                if t.endswith('_r') or t.endswith('-r'):   s += 7
                if re.search(r'[_\-\s\.]r$', t):           s += 7
                if re.search(r'\bgreen\b|\bblue\b', t):    s -= 10
                if re.search(r'\blum(inance)?\b', t):      s -= 8
                if re.search(r'[_\-\s\.]g[_\-\s\.]|[_\-\s\.]b[_\-\s\.]', t): s -= 5

            elif channel == "G":
                if re.search(r'\bgreen\b', t):             s += 10
                if re.search(r'[_\-\s\.]g[_\-\s\.]', t):   s += 8   # dropped \d — see docstring
                if t.endswith('_g') or t.endswith('-g'):   s += 7
                if re.search(r'[_\-\s\.]g$', t):           s += 7
                if re.search(r'[_\-\s\.]oiii[_\-\s\.]', t): s += 5
                if re.search(r'\bred\b|\bblue\b', t):      s -= 10
                if re.search(r'\blum(inance)?\b', t):      s -= 8
                if re.search(r'[_\-\s\.]r[_\-\s\.]|[_\-\s\.]b[_\-\s\.]', t): s -= 5

            elif channel == "B":
                if re.search(r'\bblue\b', t):              s += 10
                if re.search(r'[_\-\s\.]b[_\-\s\.]', t):   s += 8   # dropped \d — see docstring
                if t.endswith('_b') or t.endswith('-b'):   s += 7
                if re.search(r'[_\-\s\.]b$', t):           s += 7
                if re.search(r'[_\-\s\.]oiii[_\-\s\.]', t): s += 4
                if re.search(r'\bred\b|\bgreen\b', t):     s -= 10
                if re.search(r'\blum(inance)?\b', t):      s -= 8
                if re.search(r'[_\-\s\.]r[_\-\s\.]|[_\-\s\.]g[_\-\s\.]', t): s -= 5

            elif channel == "L":
                if re.search(r'\blum(inance)?\b', t):      s += 10
                if re.search(r'[_\-\s\.]l[_\-\s\.]', t):   s += 8
                if t.endswith('_l') or t.endswith('-l'):   s += 7
                if re.search(r'[_\-\s\.]l$', t):           s += 7
                if re.search(r'\bred\b|\bgreen\b|\bblue\b', t): s -= 10
                if re.search(r'[_\-\s\.][rgb][_\-\s\.]', t):    s -= 5

            return s

        # If docs weren't supplied (older callers), fall back to filename-only
        # with the regex fix still in place.
        if not docs or len(docs) != len(titles):
            scores = [filename_score(t) for t in titles]
        else:
            scores = [filename_score(t) + filter_bonus(d)
                      for t, d in zip(titles, docs)]

        if not scores:
            return 0
        best_score = max(scores)
        if best_score <= 0:
            return 0
        return scores.index(best_score)

    def _labeled(self, text, w):
        box = QVBoxLayout()
        box.addWidget(QLabel(text))
        box.addWidget(w)
        return box

    def _populate_views(self):
        self.cmb_r.clear(); self.cmb_g.clear(); self.cmb_b.clear(); self.cmb_l.clear()
        titles = []
        docs = []
        for title, doc in self._list_open_docs():
            self.cmb_r.addItem(title, doc)
            self.cmb_g.addItem(title, doc)
            self.cmb_b.addItem(title, doc)
            self.cmb_l.addItem(title, doc)
            titles.append(title)
            docs.append(doc)

        if len(titles) >= 3:
            # Pass docs through so _guess_channel can read the FITS FILTER
            # keyword — the definitive signal when it's present. Filename
            # patterns are the fallback.
            self.cmb_r.setCurrentIndex(self._guess_channel(titles, docs, "R"))
            self.cmb_g.setCurrentIndex(self._guess_channel(titles, docs, "G"))
            self.cmb_b.setCurrentIndex(self._guess_channel(titles, docs, "B"))
        if titles:
            self.cmb_l.setCurrentIndex(self._guess_channel(titles, docs, "L"))

    def _refresh_mode(self):
        use_views = self.use_views_radio.isChecked()
        for w in (self.cmb_r, self.cmb_g, self.cmb_b):
            w.setEnabled(use_views)
        for w in (self.r_label, self.g_label, self.b_label, self.btn_load_r, self.btn_load_g, self.btn_load_b):
            w.setEnabled(not use_views)
        if not load_image:
            self.load_files_radio.setEnabled(False)
            self.use_views_radio.setChecked(True)
        self._refresh_luminance_enabled()

    def _pick_file(self, which):
        titles = {"R": "Red", "G": "Green", "B": "Blue", "L": "Luminance"}
        fp, _ = QFileDialog.getOpenFileName(
            self, f"Select {titles.get(which, which)} Image", "",
            "Images (*.png *.tif *.tiff *.fits *.fit *.xisf *.jpg *.jpeg);;All Files (*)"
        )
        if not fp: return
        base = os.path.basename(fp)
        if which == "R":
            self.r_path = fp; self.r_label.setText(f"Red: {base}")
        elif which == "G":
            self.g_path = fp; self.g_label.setText(f"Green: {base}")
        elif which == "B":
            self.b_path = fp; self.b_label.setText(f"Blue: {base}")
        else:
            self.l_path = fp; self.l_label.setText(f"Luminance: {base}")

    def _load_file_mono(self, path: str) -> np.ndarray:
        if load_image is None:
            raise RuntimeError("File loading not available.")
        arr, _, _, _ = load_image(path)
        if arr is None:
            raise RuntimeError(f"Failed to load: {path}")
        return _mono_from_any(arr)

    def _load_file_raw(self, path: str) -> np.ndarray:
        """Like _load_file_mono but keeps the channel count (for the L source)."""
        if load_image is None:
            raise RuntimeError("File loading not available.")
        arr, _, _, _ = load_image(path)
        if arr is None:
            raise RuntimeError(f"Failed to load: {path}")
        return _as_is_from_any(arr)

    # ── luminance maths ───────────────────────────────────────────────

    def _recombine_weights(self, profile_key: str):
        """(resolved_method, weights3, profile_name) for the chosen profile.

        Mirrors apply_recombine_to_doc so a given profile behaves identically
        in both tools. Methods with no weight vector of their own (max,
        median, snr) fall back to Rec.709 for measuring the RGB's luminance —
        they only change how L is EXTRACTED when the L source is colour.
        """
        resolved_method, w, profile_name = resolve_luma_profile_weights(profile_key)
        if w is not None:
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            if w.size != 3:
                w = None
        if w is not None:
            recombine_w = w
        elif resolved_method == "equal":
            recombine_w = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float32)
        elif resolved_method == "rec601":
            recombine_w = _LUMA_REC601
        elif resolved_method == "rec2020":
            recombine_w = _LUMA_REC2020
        else:
            recombine_w = _LUMA_REC709
        return resolved_method, np.asarray(recombine_w, dtype=np.float32), profile_name

    # Out-of-range handling passed to linear_fit_mono_to_ref. Index 0 =
    # "Clip to [0..1]". The module's own default is 1 ("normalize if needed"),
    # but normalising subtracts the minimum and rescales the whole frame,
    # which undoes the median match we just made. Everything in this dialog
    # already lives in [0,1] and recombine_luminance_linear_scale clips its
    # output anyway, so clipping is the faithful choice here.
    _LINFIT_RESCALE_IDX = 0

    def _apply_luminance(self, rgb: np.ndarray, l_src: np.ndarray, md: dict) -> np.ndarray:
        """Run the same pipeline as the standalone Recombine Luminance tool."""
        profile_key = self._current_profile_key()
        resolved_method, recombine_w, profile_name = self._recombine_weights(profile_key)

        sat = float(self.spn_sat.value())
        cnr = float(self.spn_cnr.value())

        # Pre-recombine colour processing, in the same order as the standalone
        # tool: saturation first, then chrominance NR. Both act on the RGB
        # before luma is measured so they can't skew the luma being placed.
        rgb_pre = _boost_saturation(rgb, sat)
        rgb_pre = _chrominance_nr(rgb_pre, cnr)

        # Build L from the source. Mono passes through; colour goes through
        # compute_luminance with the selected profile.
        if l_src.ndim == 2 or (l_src.ndim == 3 and l_src.shape[2] == 1):
            L = l_src if l_src.ndim == 2 else l_src[..., 0]
            L = np.asarray(L, dtype=np.float32)
        else:
            ns = None
            if resolved_method == "snr":
                ns = _estimate_noise_sigma_per_channel(l_src)
            w_for_extract = recombine_w if profile_name else None
            L = compute_luminance(l_src, method=resolved_method,
                                  weights=w_for_extract, noise_sigma=ns)

        if self.chk_linfit.isChecked():
            # Reference is the RGB's own luminance under the SELECTED profile,
            # computed here rather than letting linear_fit_mono_to_ref derive
            # its internal Rec.709 proxy — otherwise a sensor profile would be
            # fitted against the wrong Y.
            Y = (rgb_pre[..., 0] * recombine_w[0]
                 + rgb_pre[..., 1] * recombine_w[1]
                 + rgb_pre[..., 2] * recombine_w[2]).astype(np.float32)

            # linear_fit_mono_to_ref divides by max(median, 1e-12), so a blank
            # or near-black L yields an astronomical scale factor rather than
            # an error. Nothing useful can come of fitting that, so skip it
            # and say so instead of recording a nonsense number.
            m_src_pre = float(np.nanmedian(L)) if np.isfinite(L).any() else 0.0
            m_ref_pre = float(np.nanmedian(Y)) if np.isfinite(Y).any() else 0.0
            if m_src_pre <= 1e-8 or m_ref_pre <= 1e-8:
                md["l_linear_fit"] = False
                md["l_linear_fit_skipped"] = "median too close to zero to fit"
            else:
                L, m_src, m_ref = linear_fit_mono_to_ref(L, Y, self._LINFIT_RESCALE_IDX)
                L = np.asarray(L, dtype=np.float32)
                md["l_linear_fit"] = True
                md["l_linear_fit_median_src"] = float(m_src)
                md["l_linear_fit_median_ref"] = float(m_ref)
                md["l_linear_fit_scale"] = float(m_ref / max(m_src, 1e-12))

        out = recombine_luminance_linear_scale(
            rgb_pre,
            L,
            weights=recombine_w,
            blend=float(self.spn_blend.value()),
            highlight_soft_knee=float(self.spn_knee.value()),
            pedestal=float(self.spn_pedestal.value()),
        )

        md["luma_method"] = resolved_method
        md["luma_profile_key"] = profile_key
        if profile_name:
            md["luma_profile"] = profile_name
        md["luma_weights"] = np.asarray(recombine_w, dtype=np.float32).tolist()
        md["l_blend"] = float(self.spn_blend.value())
        md["l_soft_knee"] = float(self.spn_knee.value())
        md["l_pedestal"] = float(self.spn_pedestal.value())
        if sat != 0.0:
            md["saturation_boost"] = sat
        if cnr > 0.0:
            md["chrominance_nr_sigma"] = cnr

        return out

    # --- combine → ALWAYS new document
    def _combine(self):
        try:
            use_l = self.chk_use_l.isChecked()
            l_src = None

            if self.use_views_radio.isChecked():
                rdoc = self.cmb_r.currentData()
                gdoc = self.cmb_g.currentData()
                bdoc = self.cmb_b.currentData()
                if not (rdoc and gdoc and bdoc):
                    QMessageBox.information(self, "RGB Combination", "Please select three views.")
                    return
                ldoc = self.cmb_l.currentData() if use_l else None
                if use_l and not ldoc:
                    QMessageBox.information(self, "RGB Combination",
                                            "Please select a view for the luminance channel.")
                    return
                r = _mono_from_any(getattr(rdoc, "image", None))
                g = _mono_from_any(getattr(gdoc, "image", None))
                b = _mono_from_any(getattr(bdoc, "image", None))
                title = "RGB_Combined"
                wcs_donor_docs = [gdoc, rdoc, bdoc]  # G → R → B priority
                if use_l:
                    l_src = _as_is_from_any(getattr(ldoc, "image", None))
            else:
                if not (self.r_path and self.g_path and self.b_path):
                    QMessageBox.information(self, "RGB Combination", "Please choose three files.")
                    return
                if use_l and not self.l_path:
                    QMessageBox.information(self, "RGB Combination",
                                            "Please choose a luminance file.")
                    return
                r = self._load_file_mono(self.r_path)
                g = self._load_file_mono(self.g_path)
                b = self._load_file_mono(self.b_path)
                title = f"RGB_{os.path.splitext(os.path.basename(self.r_path))[0]}"
                wcs_donor_docs = []  # file mode: no open docs to pull WCS from
                if use_l:
                    l_src = self._load_file_raw(self.l_path)

            if r.shape != g.shape or r.shape != b.shape:
                raise ValueError("All three images must have identical dimensions.")

            if use_l and l_src is not None and l_src.shape[:2] != r.shape[:2]:
                raise ValueError(
                    "The luminance image must have the same dimensions as R/G/B "
                    f"({l_src.shape[1]}x{l_src.shape[0]} vs {r.shape[1]}x{r.shape[0]}). "
                    "Register it to the colour channels first."
                )

            rgb = np.stack([r, g, b], axis=2).astype(np.float32, copy=False)
            rgb = np.clip(rgb, 0.0, 1.0)

            md = {"step_name": "RGB Combination"}

            if use_l and l_src is not None:
                md["step_name"] = "LRGB Combination"
                md["luminance_applied"] = True
                rgb = self._apply_luminance(rgb, l_src, md)
                title = title.replace("RGB", "LRGB", 1)

            if not self._docman:
                raise RuntimeError("No document manager available.")

            step = md["step_name"]
            if hasattr(self._docman, "open_array"):
                newdoc = self._docman.open_array(rgb, metadata=md, title=title)
            elif hasattr(self._docman, "open_numpy"):
                newdoc = self._docman.open_numpy(rgb, metadata=md, title=title)
            else:
                newdoc = self._docman.create_document(image=rgb, metadata=md, name=title)

            # Inherit WCS from G → R → B, whichever has one first
            mw = self.parent()
            if wcs_donor_docs and hasattr(mw, "_extract_wcs_dict") and hasattr(mw, "_apply_wcs_dict_to_doc"):
                for donor in wcs_donor_docs:
                    try:
                        wcs = mw._extract_wcs_dict(donor)
                        if wcs:
                            mw._apply_wcs_dict_to_doc(newdoc, dict(wcs))
                            break
                    except Exception:
                        continue

            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(newdoc)

            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "RGB Combination", f"Failed to combine:\n{e}")