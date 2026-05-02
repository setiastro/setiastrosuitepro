# pro/ghs_dialog_pro.py
from PyQt6.QtCore import Qt, QEvent, QPointF, QTimer, QSettings, QByteArray
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QComboBox, QSlider, QToolButton, QWidget,
                             QMessageBox, QDoubleSpinBox)
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor
import numpy as np



# Reuse the engine from curves_editor_pro
from .curve_editor_pro import (
    CurveEditor, _CurvesWorker, _apply_mode_any, build_curve_lut,
    _float_to_qimage_rgb8, _downsample_for_preview, ImageLabel
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# ---------------------------------------------------------------------------
# Stretch function implementations
# ---------------------------------------------------------------------------

def _build_ghs_lut(a, b, g, lp, hp, sym_u, N=65536):
    """
    Generalized Hyperbolic Stretch → returns a float32 LUT of length N,
    indexed 0..N-1 → output 0..1.
    """
    us = np.linspace(0.0, 1.0, N, dtype=np.float64)
    SP  = float(sym_u)
    eps = 1e-9

    left  = us <= 0.5
    right = ~left

    rawL = us**a / (us**a + b*(1.0-us)**a + eps)
    rawR = us**a / (us**a + (1.0/b)*(1.0-us)**a + eps)

    midL = (0.5**a) / (0.5**a + b*(0.5)**a + eps)
    midR = (0.5**a) / (0.5**a + (1.0/b)*(0.5)**a + eps)

    up = np.empty_like(us)
    vp = np.empty_like(us)

    up[left]  = 2.0 * SP * us[left]
    vp[left]  = rawL[left] * (SP / max(midL, eps))

    up[right] = SP + 2.0*(1.0 - SP)*(us[right] - 0.5)
    vp[right] = SP + (rawR[right] - midR) * ((1.0 - SP) / max(1.0 - midR, eps))

    if lp > 0:
        m = up <= SP
        vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
    if hp > 0:
        m = up >= SP
        vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

    if abs(g - 1.0) > 1e-6:
        vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

    vp = np.clip(vp, 0.0, 1.0)
    vp = np.maximum.accumulate(vp)
    return vp.astype(np.float32)


def _build_arcsinh_lut(strength, g, lp, hp, sym_u, N=65536):
    """
    Pivot-aware ArcSinh stretch so that f(SP) == SP.
    Each half is normalised independently:
      left  [0..SP]  → output [0..SP]
      right [SP..1]  → output [SP..1]
    LP/HP and gamma applied afterward.
    """
    us  = np.linspace(0.0, 1.0, N, dtype=np.float64)
    SP  = float(sym_u)
    s   = max(float(strength), 1e-6)
    eps = 1e-15

    shifted = us - SP
    raw     = np.arcsinh(s * shifted)

    raw_at_sp_left  = np.arcsinh(s * (0.0 - SP))
    raw_at_zero     = np.arcsinh(s * 0.0)          # == 0
    span_left       = raw_at_zero - raw_at_sp_left + eps

    raw_at_sp_right = np.arcsinh(s * 0.0)          # == 0
    raw_at_one      = np.arcsinh(s * (1.0 - SP))
    span_right      = raw_at_one - raw_at_sp_right + eps

    vp    = np.empty_like(us)
    left  = us <= SP
    right = ~left
    vp[left]  = SP * (raw[left]  - raw_at_sp_left)  / span_left
    vp[right] = SP + (1.0 - SP) * (raw[right] - raw_at_sp_right) / span_right

    up = us.copy()
    if lp > 0:
        m = up <= SP
        vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
    if hp > 0:
        m = up >= SP
        vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

    if abs(g - 1.0) > 1e-6:
        vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

    vp = np.clip(vp, 0.0, 1.0)
    vp = np.maximum.accumulate(vp)
    return vp.astype(np.float32)

def _build_log_lut(strength, g, lp, hp, sym_u, N=65536):
    """
    Pivot-aware Logarithmic stretch: raw(x) = log(1 + s*(x-SP)).
    Same two-half normalisation as ArcSinh so f(0)=0, f(SP)=SP, f(1)=1.
    Compresses highlights more aggressively than ArcSinh.
    """
    us  = np.linspace(0.0, 1.0, N, dtype=np.float64)
    SP  = float(sym_u)
    s   = max(float(strength), 1e-6)
    eps = 1e-15

    shifted = us - SP

    # log(1 + s*x) is only real for x >= -1/s; for the left half x-SP can be
    # negative, so we use the signed form: sign(d)*log(1 + s*|d|)
    def _signed_log(d):
        return np.sign(d) * np.log1p(s * np.abs(d))

    raw = _signed_log(shifted)

    raw_at_left  = _signed_log(0.0 - SP)       # raw at x=0
    raw_at_pivot = _signed_log(0.0)             # == 0
    raw_at_right = _signed_log(1.0 - SP)        # raw at x=1

    span_left  = raw_at_pivot - raw_at_left  + eps
    span_right = raw_at_right - raw_at_pivot + eps

    vp    = np.empty_like(us)
    left  = us <= SP
    right = ~left
    vp[left]  = SP * (raw[left]  - raw_at_left)  / span_left
    vp[right] = SP + (1.0 - SP) * (raw[right] - raw_at_pivot) / span_right

    up = us.copy()
    if lp > 0:
        m = up <= SP;  vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
    if hp > 0:
        m = up >= SP;  vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

    if abs(g - 1.0) > 1e-6:
        vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

    vp = np.clip(vp, 0.0, 1.0)
    vp = np.maximum.accumulate(vp)
    return vp.astype(np.float32)


def _build_exp_lut(strength, g, lp, hp, sym_u, N=65536):
    """
    Pivot-aware Exponential stretch, two-half normalisation:
      right half [SP..1]: raw = expm1(s*(x-SP))          — lifts upward
      left  half [0..SP]: raw = -expm1(s*(SP-x))         — mirror across diagonal
    Guarantees f(0)=0, f(SP)=SP, f(1)=1.
    LP/HP and gamma applied afterward.
    """
    us  = np.linspace(0.0, 1.0, N, dtype=np.float64)
    SP  = float(sym_u)
    s   = max(float(strength), 1e-6)
    eps = 1e-15

    # right half: exp(s*(x-SP))-1, zero at SP, positive going right
    raw_right_at_one  = np.expm1(s * (1.0 - SP))
    # left half: -(exp(s*(SP-x))-1), zero at SP, positive going left (toward 0)
    raw_left_at_zero  = np.expm1(s * SP)           # magnitude at x=0

    vp = np.empty_like(us)
    left  = us <= SP
    right = ~left

    # right: normalise [SP..1] → output [SP..1]
    raw_r = np.expm1(s * (us[right] - SP))
    vp[right] = SP + (1.0 - SP) * raw_r / (raw_right_at_one + eps)

    # left: flip — mirrors the right shape across the diagonal
    raw_l = np.expm1(s * (SP - us[left]))
    vp[left] = SP - SP * raw_l / (raw_left_at_zero + eps)

    up = us.copy()
    if lp > 0:
        m = up <= SP;  vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
    if hp > 0:
        m = up >= SP;  vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

    if abs(g - 1.0) > 1e-6:
        vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

    vp = np.clip(vp, 0.0, 1.0)
    vp = np.maximum.accumulate(vp)
    return vp.astype(np.float32)


def _build_pip_lut(strength, g, lp, hp, sym_u, N=65536):
    """
    Power of Inverted Pixels:
      f(x) = x ^ (1 - strength*x)
    'strength' ∈ [0..2], where 1.0 is "x^(1-x)" and 0 is identity.
    LP/HP and gamma applied afterward.
    """
    us  = np.linspace(0.0, 1.0, N, dtype=np.float64)
    SP  = float(sym_u)
    s   = float(strength)

    # avoid 0^negative: clamp base above tiny eps
    base = np.clip(us, 1e-9, 1.0)
    exp  = 1.0 - s * us          # exponent varies per pixel
    vp   = base ** exp            # x^(1-s*x)

    # Apply LP/HP protection
    up = us.copy()
    if lp > 0:
        m = up <= SP
        vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
    if hp > 0:
        m = up >= SP
        vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

    if abs(g - 1.0) > 1e-6:
        vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

    vp = np.clip(vp, 0.0, 1.0)
    vp = np.maximum.accumulate(vp)
    return vp.astype(np.float32)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class GhsDialogPro(QDialog):
    """
    Hyperbolic Stretch dialog (multi-function):
    - Function selector: Hyperbolic Stretch | ArcSinh Stretch | Power of Inverted Pixels
    - Left: parameter sliders (context-sensitive) + LP/HP + channel selector
    - Right: same preview/zoom/pan as CurvesDialogPro
    """
    # ---- stretch-mode constants ----
    MODE_GHS     = "Hyperbolic Stretch"
    MODE_ARCSINH = "ArcSinh Stretch"
    MODE_LOG     = "Logarithmic Stretch"
    MODE_EXP     = "Exponential Stretch"
    MODE_PIP     = "Power of Inverted Pixels"

    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Hyperbolic Stretch"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.doc = document
        self._preview_img = None
        self._full_img    = None
        self._pix         = None
        self._zoom        = 0.25
        self._panning     = False
        self._pan_start   = QPointF()
        self._sym_u       = 0.5   # pivot in [0..1]
        self._cached_processed_pix = None

        # ---------- layout ----------
        main = QHBoxLayout(self)

        # ── Left controls ────────────────────────────────────────────────
        left = QVBoxLayout()
        self.editor = CurveEditor(self)
        left.addWidget(self.editor)

        hint = QLabel(self.tr("Tip: Ctrl+Click (or double-click) the image to set the symmetry pivot"))
        hint.setStyleSheet("color: #888; font-size: 11px;")
        left.addWidget(hint)

        # ── Stretch function selector ────────────────────────────────────
        fn_row = QHBoxLayout()
        fn_row.addWidget(QLabel(self.tr("Function:")))
        self.cmb_fn = QComboBox(self)
        self.cmb_fn.addItems([self.MODE_GHS, self.MODE_ARCSINH, self.MODE_LOG, self.MODE_EXP, self.MODE_PIP])
        fn_row.addWidget(self.cmb_fn)
        left.addLayout(fn_row)

        # ── Channel selector ─────────────────────────────────────────────
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel(self.tr("Channel:")))
        self.cmb_ch = QComboBox(self)
        self.cmb_ch.addItems(["K (Brightness)", "R", "G", "B"])
        ch_row.addWidget(self.cmb_ch)
        left.addLayout(ch_row)

        # ── Parameter sliders ────────────────────────────────────────────
        # We build ALL sliders up front and show/hide the relevant ones.

        def _mk_row(name, lo, hi, val, divisor, fmt="{:.2f}"):
            row   = QHBoxLayout()
            lab   = QLabel(name); lab.setFixedWidth(28); row.addWidget(lab)
            s     = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi); s.setValue(val)
            row.addWidget(s)
            disp_val = val / divisor
            v     = QLabel(fmt.format(disp_val)); v.setFixedWidth(40)
            row.addWidget(v)
            # store divisor on the slider for convenience
            s._divisor = divisor
            s._fmt     = fmt
            return row, s, v

        # GHS: α, β
        self._row_A, self.sA, self.labA = _mk_row("α", 1, 500, 50,  50.0)
        self._row_B, self.sB, self.labB = _mk_row("β", 1, 500, 50,  50.0)
        # ArcSinh: Strength (1..1000 → /10 → 0.1..100)
        self._row_S, self.sS, self.labS = _mk_row("Str", 1, 1000, 50, 10.0)
        # PIP: Strength (0..200 → /100 → 0.0..2.0)
        self._row_P, self.sP, self.labP = _mk_row("Str", 0, 200,  100, 100.0)
        # Shared: γ
        self._row_G, self.sG, self.labG = _mk_row("γ",  1, 500, 100, 100.0)

        for row in (self._row_A, self._row_B, self._row_S, self._row_P, self._row_G):
            left.addLayout(row)

        # LP / HP
        rowLP = QHBoxLayout()
        rowLP.addWidget(QLabel("LP"))
        self.sLP = QSlider(Qt.Orientation.Horizontal); self.sLP.setRange(0, 360)
        rowLP.addWidget(self.sLP)
        self.labLP = QLabel("0.00"); rowLP.addWidget(self.labLP)

        rowHP = QHBoxLayout()
        rowHP.addWidget(QLabel("HP"))
        self.sHP = QSlider(Qt.Orientation.Horizontal); self.sHP.setRange(0, 360)
        rowHP.addWidget(self.sHP)
        self.labHP = QLabel("0.00"); rowHP.addWidget(self.labHP)

        left.addLayout(rowLP)
        left.addLayout(rowHP)
        # SP (Symmetry Point) — slider + editable spinbox, both synced
        rowSP = QHBoxLayout()
        rowSP.addWidget(QLabel("SP"))
        self.sSP = QSlider(Qt.Orientation.Horizontal)
        self.sSP.setRange(0, 1000)        # 0..1000 → 0.000..1.000
        self.sSP.setValue(500)
        rowSP.addWidget(self.sSP)
        self.spSP = QDoubleSpinBox()
        self.spSP.setRange(0.0, 1.0)
        self.spSP.setDecimals(3)
        self.spSP.setSingleStep(0.001)
        self.spSP.setValue(0.5)
        self.spSP.setFixedWidth(68)
        rowSP.addWidget(self.spSP)
        left.addLayout(rowSP)
        self._sp_syncing = False          # guard against feedback loops


        # ── Buttons ──────────────────────────────────────────────────────
        left.addLayout(rowSP)
        self._sp_syncing = False

        # ── Toggle Preview button ────────────────────────────────────────
        toggle_row = QHBoxLayout()
        self.btn_toggle_preview = QPushButton(self.tr("Preview: ON"))
        self.btn_toggle_preview.setCheckable(True)
        self.btn_toggle_preview.setChecked(True)
        self.btn_toggle_preview.setFixedWidth(140)
        toggle_row.addWidget(self.btn_toggle_preview)
        toggle_row.addStretch(1)
        left.addLayout(toggle_row)

        # ── Buttons ──────────────────────────────────────────────────────
        rowb = QHBoxLayout()
        self.btn_apply = QPushButton(self.tr("Apply"))
        self.btn_reset = QToolButton(); self.btn_reset.setText(self.tr("Reset"))
        self.btn_hist  = QToolButton(); self.btn_hist.setText(self.tr("Histogram"))
        self.btn_hist.setToolTip(self.tr("Open a Histogram for this image.\n"
                                         "Ctrl+Click on the histogram to set the pivot."))
        rowb.addWidget(self.btn_apply)
        rowb.addWidget(self.btn_reset)
        rowb.addWidget(self.btn_hist)
        left.addLayout(rowb)
        left.addStretch(1)

        main.addLayout(left, 0)

        # ── Right preview panel ──────────────────────────────────────────
        right   = QVBoxLayout()
        zoombar = QHBoxLayout()
        zoombar.addStretch(1)

        b_out = themed_toolbtn("zoom-out",      self.tr("Zoom Out"))
        b_in  = themed_toolbtn("zoom-in",       self.tr("Zoom In"))
        b_fit = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))

        zoombar.addWidget(b_out)
        zoombar.addWidget(b_in)
        zoombar.addWidget(b_fit)
        right.addLayout(zoombar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = ImageLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.mouseMoved.connect(self._on_preview_mouse_moved)
        self.label.installEventFilter(self)

        self.scroll.setWidget(self.label)
        self.scroll.viewport().installEventFilter(self)

        right.addWidget(self.scroll, 1)
        main.addLayout(right, 1)

        # ---------- wiring ----------
        self._suppress_editor_preview = False
        self.editor.setPreviewCallback(lambda _lut8: self._on_editor_preview())
        self.editor.setSymmetryCallback(self._on_symmetry_pick)

        # All value-bearing sliders (sSP drives _sym_u via its own handler)
        self._all_sliders = (self.sA, self.sB, self.sS, self.sP, self.sG, self.sLP, self.sHP, self.sSP)
        for s in self._all_sliders:
            s.sliderPressed.connect(self._on_any_slider_pressed)
            s.sliderReleased.connect(self._on_any_slider_released)
        # non-SP sliders drive rebuild directly
        for s in (self.sA, self.sB, self.sS, self.sP, self.sG, self.sLP, self.sHP):
            s.valueChanged.connect(self._schedule_rebuild_from_params)

        # SP slider <-> spinbox bidirectional sync (each also triggers rebuild)
        self.sSP.valueChanged.connect(self._on_sp_slider_changed)
        self.spSP.valueChanged.connect(self._on_sp_spin_changed)

        self.cmb_ch.currentTextChanged.connect(self._recolor_curve)
        self.cmb_fn.currentTextChanged.connect(self._on_function_changed)

        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self._reset)
        self._hist_dlg = None
        self.btn_hist.clicked.connect(self._open_histogram)

        b_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        b_in .clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        b_fit.clicked.connect(self._fit)

        # debounce timer
        self._rebuild_debounce = QTimer(self)
        self._rebuild_debounce.setSingleShot(True)
        self._rebuild_debounce.setInterval(75)
        self._rebuild_debounce.timeout.connect(self._rebuild_from_params_now)

        self._slider_dragging = False

        # seed image data
        self._load_from_doc()

        # show/hide sliders for initial mode, then draw first curve
        self._update_slider_visibility()
        self._rebuild_from_params()

        self.btn_toggle_preview.toggled.connect(self._on_toggle_preview)

        QTimer.singleShot(0, self._fit)

    # -----------------------------------------------------------------------
    # Function-mode helpers
    # -----------------------------------------------------------------------
    def _on_toggle_preview(self, checked: bool):
        self.btn_toggle_preview.setText("Preview: ON" if checked else "Preview: OFF")
        if self._pix is None:
            return
        if checked:
            # show processed
            if self._cached_processed_pix is not None:
                self._pix = self._cached_processed_pix
                self._apply_zoom()
        else:
            # show original
            if self._preview_img is not None:
                orig = _float_to_qimage_rgb8(self._preview_img)
                self._pix = QPixmap.fromImage(orig)
                self._apply_zoom()

    def _update_preview_pix(self, img01):
        if img01 is None:
            self.label.clear(); self._pix = None; return
        qimg = _float_to_qimage_rgb8(img01)
        pm = QPixmap.fromImage(qimg)
        self._cached_processed_pix = pm
        # only update display if preview is ON
        if getattr(self, "btn_toggle_preview", None) is None or self.btn_toggle_preview.isChecked():
            self._pix = pm
            self._apply_zoom()

    def _current_mode(self) -> str:
        return self.cmb_fn.currentText()

    def _on_function_changed(self, _text: str):
        self._update_slider_visibility()
        self._rebuild_from_params()

    def _update_slider_visibility(self):
        """Show/hide parameter rows based on the active stretch function."""
        mode = self._current_mode()

        ghs_only  = mode == self.MODE_GHS
        asi_only  = mode in (self.MODE_ARCSINH, self.MODE_LOG, self.MODE_EXP)
        pip_only  = mode == self.MODE_PIP

        def _set_row_visible(layout, visible: bool):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                w = item.widget() if item else None
                if w is not None:
                    w.setVisible(visible)

        _set_row_visible(self._row_A, ghs_only)
        _set_row_visible(self._row_B, ghs_only)
        _set_row_visible(self._row_S, asi_only)
        _set_row_visible(self._row_P, pip_only)
        # γ, LP, HP always visible

        # SP is meaningless for PIP — disable and grey it out
        sp_active = not pip_only
        self.sSP.setEnabled(sp_active)
        self.spSP.setEnabled(sp_active)
        # γ, LP, HP, SP always visible

    # -----------------------------------------------------------------------
    # Histogram support
    # -----------------------------------------------------------------------

    def _open_histogram(self):
        try:
            from .histogram import HistogramDialog
        except Exception as e:
            QMessageBox.warning(self, self.tr("Histogram"),
                                self.tr("Could not import histogram module:\n{0}").format(e))
            return

        if self._hist_dlg is not None:
            try:
                if self._hist_dlg.isVisible():
                    self._hist_dlg.raise_()
                    self._hist_dlg.activateWindow()
                    return
            except RuntimeError:
                self._hist_dlg = None

        dlg = HistogramDialog(self, self.doc)
        self._hist_dlg = dlg
        try:
            dlg.pivotPicked.connect(self._on_hist_pivot)
        except Exception:
            pass
        dlg.show()

    def _on_hist_pivot(self, u: float):
        self._set_sym_u(float(u))
        self.editor.setSymmetryPoint(self._sym_u * 360.0, 0.0)
        self._rebuild_from_params()

    # -----------------------------------------------------------------------
    # Slider housekeeping
    # -----------------------------------------------------------------------

    def _on_editor_preview(self):
        if getattr(self, "_suppress_editor_preview", False):
            return
        self._quick_preview()

    def _on_any_slider_pressed(self):
        self._slider_dragging = True

    def _on_any_slider_released(self):
        self._slider_dragging = False
        if self._rebuild_debounce.isActive():
            self._rebuild_debounce.stop()
        self._rebuild_from_params_now()

    def _set_sym_u(self, u: float):
        """Single source of truth for updating _sym_u + syncing SP widgets."""
        u = float(np.clip(u, 0.0, 1.0))
        self._sym_u = u
        if self._sp_syncing:
            return
        self._sp_syncing = True
        try:
            self.sSP.setValue(int(round(u * 1000)))
            self.spSP.setValue(u)
        finally:
            self._sp_syncing = False

    def _on_sp_slider_changed(self, int_val: int):
        if self._sp_syncing:
            return
        u = int_val / 1000.0
        self._sp_syncing = True
        try:
            self.spSP.setValue(u)
        finally:
            self._sp_syncing = False
        self._sym_u = u
        self.editor.setSymmetryPoint(u * 360.0, 0.0)
        self._schedule_rebuild_from_params()

    def _on_sp_spin_changed(self, val: float):
        if self._sp_syncing:
            return
        u = float(np.clip(val, 0.0, 1.0))
        self._sp_syncing = True
        try:
            self.sSP.setValue(int(round(u * 1000)))
        finally:
            self._sp_syncing = False
        self._sym_u = u
        self.editor.setSymmetryPoint(u * 360.0, 0.0)
        self._rebuild_from_params()

    def _schedule_rebuild_from_params(self):
        """Lightweight label update then debounced heavy rebuild."""
        self._update_labels_fast()
        self._rebuild_debounce.start()

    def _update_labels_fast(self):
        self.labA.setText(f"{self.sA.value()/50.0:.2f}")
        self.labB.setText(f"{self.sB.value()/50.0:.2f}")
        self.labS.setText(f"{self.sS.value()/10.0:.1f}")
        self.labP.setText(f"{self.sP.value()/100.0:.2f}")
        self.labG.setText(f"{self.sG.value()/100.0:.2f}")
        self.labLP.setText(f"{self.sLP.value()/360.0:.2f}")
        self.labHP.setText(f"{self.sHP.value()/360.0:.2f}")

    def _rebuild_from_params(self):
        """Immediate rebuild (for non-slider callers)."""
        if self._rebuild_debounce.isActive():
            self._rebuild_debounce.stop()
        self._rebuild_from_params_now()

    def _on_symmetry_pick(self, u, v):
        self._set_sym_u(float(u))
        self._rebuild_from_params()

    # -----------------------------------------------------------------------
    # Core rebuild dispatcher
    # -----------------------------------------------------------------------

    def _rebuild_from_params_now(self):
        self._update_labels_fast()
        mode = self._current_mode()

        if mode == self.MODE_GHS:
            self._rebuild_ghs()
        elif mode == self.MODE_ARCSINH:
            self._rebuild_arcsinh()
        elif mode == self.MODE_LOG:
            self._rebuild_log()
        elif mode == self.MODE_EXP:
            self._rebuild_exp()
        elif mode == self.MODE_PIP:
            self._rebuild_pip()

    # ---- GHS ---------------------------------------------------------------

    def _rebuild_ghs(self):
        a  = self.sA.value() / 50.0
        b  = self.sB.value() / 50.0
        g  = self.sG.value() / 100.0
        lp = self.sLP.value() / 360.0
        hp = self.sHP.value() / 360.0
        SP = float(self._sym_u)
        eps = 1e-6

        N = len(self.editor.control_points) or 20
        if len(self.editor.control_points) == 0:
            for _ in range(N):
                self.editor.addControlPoint(0, 0)

        us    = np.linspace(0.0, 1.0, N)
        left  = us <= 0.5
        right = ~left

        rawL = us**a / (us**a + b*(1.0-us)**a + eps)
        rawR = us**a / (us**a + (1.0/b)*(1.0-us)**a + eps)

        midL = (0.5**a) / (0.5**a + b*(0.5)**a + eps)
        midR = (0.5**a) / (0.5**a + (1.0/b)*(0.5)**a + eps)

        up = np.empty_like(us); vp = np.empty_like(us)
        up[left]  = 2.0 * SP * us[left]
        vp[left]  = rawL[left] * (SP / max(midL, eps))
        up[right] = SP + 2.0*(1.0 - SP)*(us[right] - 0.5)
        vp[right] = SP + (rawR[right] - midR) * ((1.0 - SP) / max(1.0 - midR, eps))

        if lp > 0:
            m = up <= SP; vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
        if hp > 0:
            m = up >= SP; vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

        if abs(g - 1.0) > 1e-6:
            vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

        vp = np.clip(vp, 0.0, 1.0)
        vp = np.maximum.accumulate(vp)

        self._push_handles_to_editor(up, vp)

    # ---- ArcSinh -----------------------------------------------------------

    def _rebuild_arcsinh(self):
        s  = max(self.sS.value() / 10.0, 1e-6)   # 0.1 .. 100
        g  = self.sG.value() / 100.0
        lp = self.sLP.value() / 360.0
        hp = self.sHP.value() / 360.0
        SP = float(self._sym_u)

        N = len(self.editor.control_points) or 20
        if len(self.editor.control_points) == 0:
            for _ in range(N):
                self.editor.addControlPoint(0, 0)

        us = np.linspace(0.0, 1.0, N, dtype=np.float64)

        # Pivot-aware ArcSinh: stretch x relative to SP so that f(SP) == SP.
        # We shift the input by SP, apply asinh, then normalise each half
        # independently so the curve passes through (0,0), (SP,SP), (1,1).
        eps = 1e-15
        shifted = us - SP                               # centred on pivot

        raw = np.arcsinh(s * shifted)                  # unbounded, centred on 0

        # normalise left half  [0..SP] → output [0..SP]
        raw_at_sp_left  = np.arcsinh(s * (0.0 - SP))  # raw value at x=0
        raw_at_zero     = np.arcsinh(s * 0.0)          # raw value at x=SP (== 0)
        span_left       = raw_at_zero - raw_at_sp_left + eps

        # normalise right half [SP..1] → output [SP..1]
        raw_at_sp_right = np.arcsinh(s * 0.0)          # raw value at x=SP (== 0)
        raw_at_one      = np.arcsinh(s * (1.0 - SP))   # raw value at x=1
        span_right      = raw_at_one - raw_at_sp_right + eps

        vp = np.empty_like(us)
        left  = us <= SP
        right = ~left

        # left half maps raw ∈ [raw(0), 0] → output ∈ [0, SP]
        vp[left]  = SP * (raw[left]  - raw_at_sp_left)  / span_left
        # right half maps raw ∈ [0, raw(1)] → output ∈ [SP, 1]
        vp[right] = SP + (1.0 - SP) * (raw[right] - raw_at_sp_right) / span_right

        up = us.copy()
        if lp > 0:
            m = up <= SP; vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
        if hp > 0:
            m = up >= SP; vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

        if abs(g - 1.0) > 1e-6:
            vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

        vp = np.clip(vp, 0.0, 1.0)
        vp = np.maximum.accumulate(vp)

        self._push_handles_to_editor(up, vp)


    # ---- Logarithmic Stretch -----------------------------------------------

    def _rebuild_log(self):
        s  = max(self.sS.value() / 10.0, 1e-6)
        g  = self.sG.value() / 100.0
        lp = self.sLP.value() / 360.0
        hp = self.sHP.value() / 360.0
        SP = float(self._sym_u)

        N = len(self.editor.control_points) or 20
        if len(self.editor.control_points) == 0:
            for _ in range(N):
                self.editor.addControlPoint(0, 0)

        us  = np.linspace(0.0, 1.0, N, dtype=np.float64)
        eps = 1e-15

        def _signed_log(d):
            return np.sign(d) * np.log1p(s * np.abs(d))

        raw = _signed_log(us - SP)

        raw_at_left  = _signed_log(0.0 - SP)
        raw_at_pivot = _signed_log(0.0)        # == 0
        raw_at_right = _signed_log(1.0 - SP)

        span_left  = raw_at_pivot - raw_at_left  + eps
        span_right = raw_at_right - raw_at_pivot + eps

        vp    = np.empty_like(us)
        left  = us <= SP
        right = ~left
        vp[left]  = SP * (raw[left]  - raw_at_left)  / span_left
        vp[right] = SP + (1.0 - SP) * (raw[right] - raw_at_pivot) / span_right

        up = us.copy()
        if lp > 0:
            m = up <= SP; vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
        if hp > 0:
            m = up >= SP; vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

        if abs(g - 1.0) > 1e-6:
            vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

        vp = np.clip(vp, 0.0, 1.0)
        vp = np.maximum.accumulate(vp)
        self._push_handles_to_editor(up, vp)

    # ---- Exponential Stretch ------------------------------------------------

    def _rebuild_exp(self):
        s  = max(self.sS.value() / 10.0, 1e-6)
        g  = self.sG.value() / 100.0
        lp = self.sLP.value() / 360.0
        hp = self.sHP.value() / 360.0
        SP = float(self._sym_u)

        N = len(self.editor.control_points) or 20
        if len(self.editor.control_points) == 0:
            for _ in range(N):
                self.editor.addControlPoint(0, 0)

        us  = np.linspace(0.0, 1.0, N, dtype=np.float64)
        eps = 1e-15

        raw_right_at_one = np.expm1(s * (1.0 - SP))
        raw_left_at_zero = np.expm1(s * SP)

        vp    = np.empty_like(us)
        left  = us <= SP
        right = ~left

        raw_r = np.expm1(s * (us[right] - SP))
        vp[right] = SP + (1.0 - SP) * raw_r / (raw_right_at_one + eps)

        raw_l = np.expm1(s * (SP - us[left]))
        vp[left] = SP - SP * raw_l / (raw_left_at_zero + eps)

        up = us.copy()
        if lp > 0:
            m = up <= SP; vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
        if hp > 0:
            m = up >= SP; vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

        if abs(g - 1.0) > 1e-6:
            vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

        vp = np.clip(vp, 0.0, 1.0)
        vp = np.maximum.accumulate(vp)
        self._push_handles_to_editor(up, vp)

    # ---- Power of Inverted Pixels ------------------------------------------

    def _rebuild_pip(self):
        s  = self.sP.value() / 100.0   # 0.0 .. 2.0
        g  = self.sG.value() / 100.0
        lp = self.sLP.value() / 360.0
        hp = self.sHP.value() / 360.0
        SP = float(self._sym_u)

        N = len(self.editor.control_points) or 20
        if len(self.editor.control_points) == 0:
            for _ in range(N):
                self.editor.addControlPoint(0, 0)

        us   = np.linspace(0.0, 1.0, N, dtype=np.float64)
        base = np.clip(us, 1e-9, 1.0)
        exp  = 1.0 - s * us        # exponent = 1 - strength*x
        vp   = base ** exp         # x^(1 - s*x)

        up = us.copy()
        if lp > 0:
            m = up <= SP; vp[m] = (1.0 - lp)*vp[m] + lp*up[m]
        if hp > 0:
            m = up >= SP; vp[m] = (1.0 - hp)*vp[m] + hp*up[m]

        if abs(g - 1.0) > 1e-6:
            vp = np.clip(vp, 0.0, 1.0) ** (1.0 / g)

        vp = np.clip(vp, 0.0, 1.0)
        vp = np.maximum.accumulate(vp)

        self._push_handles_to_editor(up, vp)

    # ---- shared handle writer ----------------------------------------------

    def _push_handles_to_editor(self, up: np.ndarray, vp: np.ndarray):
        """Write normalised (up, vp) ∈ [0,1]² into the CurveEditor handles."""
        xs  = up * 360.0
        ys  = (1.0 - vp) * 360.0
        pts = list(zip(xs.astype(float), ys.astype(float)))

        self._suppress_editor_preview = True
        try:
            cps_sorted = sorted(self.editor.control_points, key=lambda p: p.scenePos().x())
            for p, (x, y) in zip(cps_sorted, pts):
                p.setPos(x, y)

            self._recolor_curve()
            self.editor.updateCurve()
        finally:
            self._suppress_editor_preview = False

        self._quick_preview()

    # -----------------------------------------------------------------------
    # LUT building (used by preview & apply)
    # -----------------------------------------------------------------------

    def _build_lut01(self) -> np.ndarray | None:
        """Build a float32 LUT[0..65535] → [0..1] for the active function."""
        mode = self._current_mode()
        g    = self.sG.value() / 100.0
        lp   = self.sLP.value() / 360.0
        hp   = self.sHP.value() / 360.0
        SP   = float(self._sym_u)

        try:
            if mode == self.MODE_GHS:
                # GHS: always derive the LUT from the curve editor itself,
                # so the applied result exactly matches what the curve shows.
                fn = getattr(self.editor, "getCurveFunction", None)
                if not fn:
                    return None
                f = fn()
                if f is None:
                    return None
                return build_curve_lut(f, size=65536)
            elif mode == self.MODE_ARCSINH:
                s = max(self.sS.value() / 10.0, 1e-6)
                return _build_arcsinh_lut(s, g, lp, hp, SP)
            elif mode == self.MODE_LOG:
                s = max(self.sS.value() / 10.0, 1e-6)
                return _build_log_lut(s, g, lp, hp, SP)
            elif mode == self.MODE_EXP:
                s = max(self.sS.value() / 10.0, 1e-6)
                return _build_exp_lut(s, g, lp, hp, SP)
            elif mode == self.MODE_PIP:
                s = self.sP.value() / 100.0
                return _build_pip_lut(s, g, lp, hp, SP)
        except Exception:
            pass
        return None

    # -----------------------------------------------------------------------
    # Curve colour
    # -----------------------------------------------------------------------

    def _recolor_curve(self):
        color_map = {
            "K (Brightness)": Qt.GlobalColor.white,
            "R": Qt.GlobalColor.red,
            "G": Qt.GlobalColor.green,
            "B": Qt.GlobalColor.blue,
        }
        ch = self.cmb_ch.currentText()
        if getattr(self.editor, "curve_item", None):
            pen = QPen(color_map[ch]); pen.setWidth(3)
            self.editor.curve_item.setPen(pen)
        self._quick_preview()

    # -----------------------------------------------------------------------
    # Preview / Apply
    # -----------------------------------------------------------------------

    def _quick_preview(self):
        if self._preview_img is None:
            return
        lut01 = self._build_lut01()
        if lut01 is None:
            return
        # apply_mode_any expects an 8-bit LUT via build_curve_lut
        # but _build_lut01 already returns a full float LUT we can use directly
        mode = self.cmb_ch.currentText()
        out  = _apply_mode_any(self._preview_img, mode, lut01)
        out  = self._blend_with_mask(out)
        self._update_preview_pix(out)

    def _apply(self):
        if self._full_img is None:
            return

        luts = self._build_all_active_luts()
        self.btn_apply.setEnabled(False)
        self._thr = _CurvesWorker(self._full_img, luts, self)
        self._thr.done.connect(self._on_apply_ready)
        self._thr.finished.connect(lambda: self.btn_apply.setEnabled(True))
        self._thr.start()

    def _build_all_active_luts(self) -> dict[str, np.ndarray]:
        lut = self._build_lut01()
        if lut is None:
            return {}
        ch = self.cmb_ch.currentText()
        ui2key = {"K (Brightness)": "K", "R": "R", "G": "G", "B": "B"}
        key = ui2key.get(ch, "K")
        return {key: lut}

    def _apply_all_curves_once(self, img: np.ndarray, luts: dict[str, np.ndarray]) -> np.ndarray:
        """Called by _CurvesWorker on the background thread."""
        if not luts:
            return img
        (key, lut), = luts.items()
        key2mode = {"K": "K (Brightness)", "R": "R", "G": "G", "B": "B"}
        mode = key2mode.get(key, "K (Brightness)")
        out = _apply_mode_any(img, mode, lut)
        return out.astype(np.float32, copy=False)

    def _on_apply_ready(self, out01: np.ndarray):
        try:
            out_masked = self._blend_with_mask(out01)

            mode = self._current_mode()

            # Build params dict common to all modes
            ghs_params = {
                "function":  mode,
                "gamma":     self.sG.value() / 100.0,
                "lp":        self.sLP.value() / 360.0,
                "hp":        self.sHP.value() / 360.0,
                "pivot":     float(self._sym_u),
                "channel":   self.cmb_ch.currentText(),
            }
            if mode == self.MODE_GHS:
                ghs_params["alpha"] = self.sA.value() / 50.0
                ghs_params["beta"]  = self.sB.value() / 50.0
            elif mode in (self.MODE_ARCSINH, self.MODE_LOG, self.MODE_EXP):
                ghs_params["strength"] = self.sS.value() / 10.0
            elif mode == self.MODE_PIP:
                ghs_params["strength"] = self.sP.value() / 100.0

            _marr, mid, mname = self._active_mask_layer()
            meta = {
                "step_name": f"Hyperbolic Stretch ({mode})",
                "ghs":        ghs_params,
                "masked":     bool(mid),
                "mask_id":    mid,
                "mask_name":  mname,
                "mask_blend": "m*out + (1-m)*src",
            }

            # Propagate to replay system
            mw = self.parent()
            while mw is not None and not (
                hasattr(mw, "_remember_last_action_from_dialog")
                or hasattr(mw, "_remember_last_headless_command")
            ):
                mw = mw.parent()

            if mw is not None:
                if hasattr(mw, "_remember_last_action_from_dialog"):
                    try:
                        mw._remember_last_action_from_dialog("ghs", ghs_params)
                    except Exception:
                        pass
                if hasattr(mw, "_remember_last_headless_command"):
                    try:
                        mw._remember_last_headless_command(
                            "ghs", ghs_params,
                            description=f"Hyperbolic Stretch ({mode})",
                        )
                        try:
                            mw._log(f"[Replay] GHS stored: fn={mode}, "
                                    f"keys={list(ghs_params.keys())}")
                        except Exception:
                            print("[Replay] GHS stored, keys=", list(ghs_params.keys()))
                    except Exception as e:
                        print("[Replay] GHS remember_last_headless_command failed:", e)

            self.doc.apply_edit(out_masked.copy(),
                                metadata=meta,
                                step_name=f"Hyperbolic Stretch ({mode})")

            self._load_from_doc()

            # Reset to defaults for next pass
            self._set_sym_u(0.5)
            self.editor.clearSymmetryLine()
            self.editor.initCurve()
            self.sA.setValue(50);  self.sB.setValue(50);  self.sG.setValue(100)
            self.sS.setValue(50);  self.sP.setValue(100)
            self.sLP.setValue(0);  self.sHP.setValue(0)
            self._rebuild_from_params()

        except Exception as e:
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))

    # -----------------------------------------------------------------------
    # Image plumbing / zoom / pan
    # -----------------------------------------------------------------------

    def _load_from_doc(self):
        img = self.doc.image
        if img is None:
            QMessageBox.information(self, self.tr("No image"), self.tr("Open an image first."))
            return
        arr = np.asarray(img).astype(np.float32)
        if arr.dtype.kind in "ui":
            arr = arr / np.iinfo(img.dtype).max
        self._full_img    = arr
        self._preview_img = _downsample_for_preview(arr, 1200)
        self._update_preview_pix(self._preview_img)

    def _apply_zoom(self):
        if self._pix is None: return
        scaled = self._pix.scaled(
            self._pix.size() * self._zoom,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

    def _set_zoom(self, z):
        self._zoom = float(max(0.05, min(z, 8.0)))
        self._apply_zoom()

    def _fit(self):
        if self._pix is None: return
        vp = self.scroll.viewport().size()
        if self._pix.width() == 0 or self._pix.height() == 0: return
        s = min(vp.width() / self._pix.width(), vp.height() / self._pix.height())
        self._set_zoom(max(0.05, s))

    def _k_from_label_point(self, lbl_pt):
        if self._preview_img is None or self.label.pixmap() is None:
            return None
        pix = self.label.pixmap()
        pw, ph = pix.width(), pix.height()
        x, y   = int(lbl_pt.x()), int(lbl_pt.y())
        if not (0 <= x < pw and 0 <= y < ph):
            return None
        ih, iw = self._preview_img.shape[:2]
        ix = int(x * iw / pw); iy = int(y * ih / ph)
        ix = max(0, min(iw - 1, ix)); iy = max(0, min(ih - 1, iy))
        px = self._preview_img[iy, ix]
        k  = float(np.mean(px)) if self._preview_img.ndim == 3 else float(px)
        return max(0.0, min(1.0, k))

    def eventFilter(self, obj, ev):
        lbl = getattr(self, "label", None)
        if lbl is None:
            return False

        if obj is self.label or obj is self.scroll.viewport():
            # Double-click → set pivot
            if (ev.type() == QEvent.Type.MouseButtonDblClick
                    and ev.button() == Qt.MouseButton.LeftButton):
                lbl_pt = (ev.position().toPoint() if obj is self.label
                          else self.label.mapFrom(self.scroll.viewport(),
                                                  ev.position().toPoint()))
                k = self._k_from_label_point(lbl_pt)
                if k is not None:
                    self._set_sym_u(k)
                    self.editor.setSymmetryPoint(self._sym_u * 360.0, 0)
                    self._rebuild_from_params()
                    ev.accept(); return True

            # Ctrl+click → set pivot
            if (ev.type() == QEvent.Type.MouseButtonPress
                    and ev.button() == Qt.MouseButton.LeftButton
                    and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)):
                lbl_pt = (ev.position().toPoint() if obj is self.label
                          else self.label.mapFrom(self.scroll.viewport(),
                                                  ev.position().toPoint()))
                k = self._k_from_label_point(lbl_pt)
                if k is not None:
                    self._set_sym_u(k)
                    self.editor.setSymmetryPoint(self._sym_u * 360.0, 0)
                    self._rebuild_from_params()
                    ev.accept(); return True

        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel:
                dy = ev.pixelDelta().y()
                if dy != 0:
                    abs_dy   = abs(dy)
                    ctrl_dn  = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if abs_dy <= 3:
                        base = 1.012 if ctrl_dn else 1.010
                    elif abs_dy <= 10:
                        base = 1.025 if ctrl_dn else 1.020
                    else:
                        base = 1.040 if ctrl_dn else 1.030
                    factor = base if dy > 0 else 1.0 / base
                else:
                    dy = ev.angleDelta().y()
                    if dy == 0:
                        ev.accept(); return True
                    ctrl_dn = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    step    = 1.25 if ctrl_dn else 1.15
                    factor  = step if dy > 0 else 1.0 / step
                self._set_zoom(self._zoom * factor)
                ev.accept(); return True

            if (ev.type() == QEvent.Type.MouseButtonPress
                    and ev.button() == Qt.MouseButton.LeftButton):
                self._panning  = True; self._pan_start = ev.position()
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

            if (ev.type() == QEvent.Type.MouseButtonRelease
                    and ev.button() == Qt.MouseButton.LeftButton):
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True

        return super().eventFilter(obj, ev)

    def _on_preview_mouse_moved(self, x: float, y: float):
        if self._panning or self._preview_img is None or self._pix is None:
            return
        ix = int(x / max(self._zoom, 1e-6))
        iy = int(y / max(self._zoom, 1e-6))
        ix = max(0, min(self._pix.width()  - 1, ix))
        iy = max(0, min(self._pix.height() - 1, iy))

        img = self._preview_img
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            v = float(img[iy, ix] if img.ndim == 2 else img[iy, ix, 0])
            v = float(np.clip(v, 0.0, 1.0))
            self.editor.updateValueLines(v, 0.0, 0.0, grayscale=True)
        else:
            r, g, b = img[iy, ix, 0], img[iy, ix, 1], img[iy, ix, 2]
            r = float(np.clip(r, 0.0, 1.0))
            g = float(np.clip(g, 0.0, 1.0))
            b = float(np.clip(b, 0.0, 1.0))
            self.editor.updateValueLines(r, g, b, grayscale=False)

    # -----------------------------------------------------------------------
    # Mask helpers
    # -----------------------------------------------------------------------

    def _active_mask_layer(self):
        mid = getattr(self.doc, "active_mask_id", None)
        if not mid: return None, None, None
        layer = getattr(self.doc, "masks", {}).get(mid)
        if layer is None: return None, None, None
        m = np.asarray(getattr(layer, "data", None))
        if m is None or m.size == 0: return None, None, None
        m = m.astype(np.float32, copy=False)
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0: m /= mx
        return np.clip(m, 0.0, 1.0), mid, getattr(layer, "name", "Mask")

    def _resample_mask_if_needed(self, mask: np.ndarray,
                                  out_hw: tuple[int, int]) -> np.ndarray:
        mh, mw = mask.shape[:2]
        th, tw = out_hw
        if (mh, mw) == (th, tw): return mask
        yi = np.linspace(0, mh - 1, th).astype(np.int32)
        xi = np.linspace(0, mw - 1, tw).astype(np.int32)
        return mask[yi][:, xi]

    def _blend_with_mask(self, processed: np.ndarray) -> np.ndarray:
        mask, _mid, _mname = self._active_mask_layer()
        if mask is None:
            return processed

        out = processed.astype(np.float32, copy=False)
        if (hasattr(self, "_full_img") and self._full_img is not None
                and out.shape[:2] == self._full_img.shape[:2]):
            src = self._full_img
        else:
            src = self._preview_img

        m = self._resample_mask_if_needed(mask, out.shape[:2])
        if out.ndim == 3 and out.shape[2] == 3:
            m = m[..., None]

        if src.ndim == 2 and out.ndim == 3:
            src = np.stack([src]*3, axis=-1)
        elif src.ndim == 3 and out.ndim == 2:
            src = src[..., 0]

        return (m * out + (1.0 - m) * src).astype(np.float32, copy=False)

    # -----------------------------------------------------------------------
    # Window geometry persistence
    # -----------------------------------------------------------------------

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("ghs/window_geometry", None)
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("ghs/window_geometry", self.saveGeometry())
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

    # -----------------------------------------------------------------------
    # Reset / Close
    # -----------------------------------------------------------------------

    def _reset(self):
        for s in self._all_sliders:
            s.blockSignals(True)
        try:
            self.sA.setValue(50);  self.sB.setValue(50);  self.sG.setValue(100)
            self.sS.setValue(50);  self.sP.setValue(100)
            self.sLP.setValue(0);  self.sHP.setValue(0)
        finally:
            for s in self._all_sliders:
                s.blockSignals(False)

        self._set_sym_u(0.5)
        self.editor.clearSymmetryLine()
        self._rebuild_from_params()

    def closeEvent(self, ev):
        self._save_window_geometry()
        super().closeEvent(ev)