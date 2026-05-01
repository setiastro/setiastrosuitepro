#saspro/narrowbandintegration.py

from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QComboBox, QGroupBox,
    QVBoxLayout, QHBoxLayout, QMessageBox, QDoubleSpinBox,
    QScrollArea, QFrame, QSplitter, QSizePolicy, QCheckBox, QLineEdit,
    QInputDialog
)

try:
    from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
except Exception:
    def themed_toolbtn(name, tooltip=""):
        btn = QPushButton(name)
        btn.setToolTip(tooltip)
        btn.setFixedSize(28, 28)
        return btn


# ---------------------------------------------------------------------------
# Wavelength → approximate HSV hue (degrees)
# ---------------------------------------------------------------------------
_DEFAULT_HUE = {
    "Ha":   4.0,    # 656nm — warm red
    "SII":  0.0,    # 672nm — pure/deep red
    "OIII": 156.0,  # 496nm — green-cyan
}

_NB_LABELS = ["Ha", "SII", "OIII"]

# Preset filter library for the dynamic "Add Filter" feature
# name → (wavelength_nm, hue_deg, description)
_FILTER_PRESETS = {
    "Ha"      : (656, 4.0,   "Hydrogen Alpha 656 nm"),
    "SII"     : (672, 0.0,   "Sulfur II 672 nm"),
    "OIII"    : (496, 156.0, "Oxygen III 496/501 nm"),
    "Neon"    : (585, 52.0,  "Neon 585 nm"),
    "Argon"   : (488, 196.0, "Argon 488 nm"),
    "Nitrogen": (658, 2.0,   "Nitrogen 658 nm"),
    "Helium"  : (587, 51.0,  "Helium 587 nm"),
    "Custom"  : (None, 0.0,  "Custom wavelength"),
}

_FILTER_PRESET_NAMES = ["Ha", "SII", "OIII", "Neon", "Argon", "Nitrogen", "Helium", "Custom"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_f32(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return np.clip(a, 0.0, 1.0)


def _ensure_mono(arr: np.ndarray) -> np.ndarray:
    a = _to_f32(arr)
    if a.ndim == 3:
        return (0.2989 * a[..., 0] + 0.5870 * a[..., 1] + 0.1140 * a[..., 2]).astype(np.float32)
    return a


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    a = _to_f32(arr)
    if a.ndim == 2:
        return np.repeat(a[..., None], 3, axis=2)
    return a


def _colorize(mono: np.ndarray, hue_deg: float) -> np.ndarray:
    h  = float(hue_deg) % 360.0
    hi = int(h / 60.0) % 6
    f  = (h / 60.0) - int(h / 60.0)
    v  = mono.astype(np.float32)
    p  = np.zeros_like(v)
    q  = v * (1.0 - f)
    t  = v * f
    sectors = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]
    r_ch, g_ch, b_ch = sectors[hi]
    return np.clip(np.stack([r_ch, g_ch, b_ch], axis=-1), 0.0, 1.0).astype(np.float32)


def _screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a + b - a * b, 0.0, 1.0).astype(np.float32)


def _to_pixmap(img01: np.ndarray) -> QPixmap:
    a  = np.clip(_ensure_rgb(img01), 0.0, 1.0)
    u8 = (a * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = u8.shape
    qimg = QImage(u8.data, w, h, u8.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _downsample(img: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_dim:
        return img
    k = max_dim / float(s)
    nh, nw = int(h * k), int(w * k)
    if cv2 is not None:
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img[::int(1.0/k), ::int(1.0/k)]


# ---------------------------------------------------------------------------
# Hue gradient slider widget
# ---------------------------------------------------------------------------

class HueGradientSlider(QWidget):
    hueChanged = pyqtSignal(float)

    def __init__(self, default_hue: float = 0.0, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setMinimumWidth(200)
        self._hue = float(default_hue) % 360.0
        self._dragging = False
        self._gradient_img = None
        self.setMouseTracking(True)

    def hue(self) -> float:
        return self._hue

    def setHue(self, hue: float, notify: bool = True):
        h = float(hue) % 360.0
        if abs(h - self._hue) < 0.01:
            return
        self._hue = h
        self.update()
        if notify:
            self.hueChanged.emit(self._hue)

    def _build_gradient(self, w: int, h: int):
        if (self._gradient_img is not None
                and self._gradient_img.width() == w
                and self._gradient_img.height() == h):
            return
        hues    = np.linspace(0, 179, w, dtype=np.uint8)
        sat     = np.full(w, 220, dtype=np.uint8)
        val     = np.full(w, 230, dtype=np.uint8)
        hsv_row = np.stack([hues, sat, val], axis=-1)[None, :, :]
        if cv2 is not None:
            rgb_row = cv2.cvtColor(hsv_row, cv2.COLOR_HSV2RGB)
        else:
            r = np.clip(np.abs(hues.astype(float)/180*6 - 3) - 1, 0, 1)
            g = np.clip(2 - np.abs(hues.astype(float)/180*6 - 2), 0, 1)
            b = np.clip(2 - np.abs(hues.astype(float)/180*6 - 4), 0, 1)
            rgb_row = (np.stack([r, g, b], axis=-1)[None] * 230).astype(np.uint8)
        row  = np.repeat(rgb_row, h, axis=0)
        qimg = QImage(row.data, w, h, row.strides[0], QImage.Format.Format_RGB888).copy()
        self._gradient_img = qimg

    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter, QPen, QColor, QBrush
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track_h      = 14
        track_y      = (self.height() - track_h) // 2
        track_rect_x = 10
        track_rect_w = self.width() - 20
        self._build_gradient(track_rect_w, track_h)
        p.drawImage(track_rect_x, track_y, self._gradient_img)
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(track_rect_x, track_y, track_rect_w, track_h)
        tx = int(track_rect_x + (self._hue / 360.0) * track_rect_w)
        ty = self.height() // 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 60)))
        p.drawEllipse(tx - 9, ty - 9, 18, 18)
        hsv_pix = np.array([[[int(self._hue / 2), 220, 230]]], dtype=np.uint8)
        if cv2 is not None:
            rgb_pix    = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2RGB)[0, 0]
            fill_color = QColor(int(rgb_pix[0]), int(rgb_pix[1]), int(rgb_pix[2]))
        else:
            fill_color = QColor(200, 200, 200)
        p.setBrush(QBrush(fill_color))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawEllipse(tx - 8, ty - 8, 16, 16)

    def _hue_from_x(self, x: int) -> float:
        track_rect_x = 10
        track_rect_w = self.width() - 20
        frac = (x - track_rect_x) / max(track_rect_w, 1)
        return max(0.0, min(360.0, frac * 360.0))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.setHue(self._hue_from_x(int(ev.position().x())))

    def mouseMoveEvent(self, ev):
        if self._dragging:
            self.setHue(self._hue_from_x(int(ev.position().x())))

    def mouseReleaseEvent(self, ev):
        self._dragging = False


# ---------------------------------------------------------------------------
# Dynamic filter row widget
# ---------------------------------------------------------------------------

class DynamicFilterRow(QWidget):
    """
    A self-contained widget representing one user-added narrowband filter row.
    Emits changed() whenever any control value changes.
    Emits removeRequested() when the ✕ button is clicked.
    """
    changed         = pyqtSignal()
    removeRequested = pyqtSignal(object)  # passes self

    def __init__(self, doc_items: list[tuple], default_name: str = "Custom",
                 default_hue: float = 0.0, parent=None):
        """
        doc_items: list of (display_name, doc_object) for populating the dropdown.
                   First entry should be ("— None —", None).
        """
        super().__init__(parent)
        self._doc_items = doc_items
        self._build(default_name, default_hue)

    def _build(self, default_name: str, default_hue: float):
        # Outer group box — title will be the filter name
        self._gb = QGroupBox(default_name)
        gb_layout = QVBoxLayout(self._gb)
        gb_layout.setSpacing(6)

        # Row 0: enable + name + preset dropdown + remove button
        row0 = QHBoxLayout()

        self.cb_enable = QCheckBox("Enable")
        self.cb_enable.setChecked(False)
        self.cb_enable.toggled.connect(self._on_enable_toggled)
        row0.addWidget(self.cb_enable)

        self.cmb_doc = QComboBox()
        self.cmb_doc.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_doc.setMinimumContentsLength(20)
        self.cmb_doc.setEnabled(False)
        for name, doc in self._doc_items:
            self.cmb_doc.addItem(name, doc)
        self.cmb_doc.currentIndexChanged.connect(self.changed)
        row0.addWidget(self.cmb_doc, 1)

        # Preset snap button
        self.cmb_preset = QComboBox()
        self.cmb_preset.setToolTip("Snap hue to a known filter wavelength")
        for pname in _FILTER_PRESET_NAMES:
            _, hue, desc = _FILTER_PRESETS[pname]
            nm = _FILTER_PRESETS[pname][0]
            label = f"{pname}" if nm is None else f"{pname} ({nm}nm)"
            self.cmb_preset.addItem(label, pname)
        self.cmb_preset.setCurrentIndex(_FILTER_PRESET_NAMES.index("Custom"))
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_selected)
        self.cmb_preset.setFixedWidth(130)
        row0.addWidget(self.cmb_preset)

        # Name edit
        self.name_edit = QLineEdit(default_name)
        self.name_edit.setPlaceholderText("Filter name")
        self.name_edit.setFixedWidth(90)
        self.name_edit.textChanged.connect(self._on_name_changed)
        row0.addWidget(self.name_edit)

        # Remove button
        btn_remove = QPushButton("✕")
        btn_remove.setFixedSize(24, 24)
        btn_remove.setToolTip("Remove this filter")
        btn_remove.clicked.connect(lambda: self.removeRequested.emit(self))
        row0.addWidget(btn_remove)

        gb_layout.addLayout(row0)

        # Row 1: hue label
        gb_layout.addWidget(QLabel("Colorize hue (°):"))

        # Row 2: hue gradient slider
        self.hue_slider = HueGradientSlider(default_hue=default_hue)
        self.hue_slider.setEnabled(False)
        self.hue_slider.hueChanged.connect(self._on_hue_slider_changed)
        gb_layout.addWidget(self.hue_slider)

        # Row 3: hue spin + amount
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Hue:"))
        self.hue_spin = QDoubleSpinBox()
        self.hue_spin.setRange(0.0, 359.9)
        self.hue_spin.setSingleStep(1.0)
        self.hue_spin.setDecimals(1)
        self.hue_spin.setValue(default_hue)
        self.hue_spin.setEnabled(False)
        self.hue_spin.valueChanged.connect(self._on_hue_spin_changed)
        row3.addWidget(self.hue_spin)

        row3.addSpacing(16)
        row3.addWidget(QLabel("Amount:"))
        self.amt_spin = QDoubleSpinBox()
        self.amt_spin.setRange(0.0, 2.0)
        self.amt_spin.setSingleStep(0.05)
        self.amt_spin.setDecimals(2)
        self.amt_spin.setValue(1.0)
        self.amt_spin.setEnabled(False)
        self.amt_spin.setToolTip("0 = no contribution, 1 = full screen, 2 = double weight")
        self.amt_spin.valueChanged.connect(self.changed)
        row3.addWidget(self.amt_spin)
        gb_layout.addLayout(row3)

        # Wrap group box in this widget's layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._gb)
        self.setMinimumWidth(340)

    # --- internal helpers ---

    def _on_enable_toggled(self, checked: bool):
        self.cmb_doc.setEnabled(checked)
        self.hue_slider.setEnabled(checked)
        self.hue_spin.setEnabled(checked)
        self.amt_spin.setEnabled(checked)
        self.changed.emit()

    def _on_name_changed(self, text: str):
        self._gb.setTitle(text or "Custom")

    def _on_preset_selected(self, _):
        key = self.cmb_preset.currentData()
        if key and key != "Custom":
            _, hue, _ = _FILTER_PRESETS[key]
            self.hue_slider.setHue(hue, notify=False)
            self.hue_slider.update()
            self.hue_spin.blockSignals(True)
            self.hue_spin.setValue(hue)
            self.hue_spin.blockSignals(False)
            # Also update name to match preset
            self.name_edit.setText(key)
            self.changed.emit()

    def _on_hue_slider_changed(self, hue: float):
        self.hue_spin.blockSignals(True)
        self.hue_spin.setValue(hue % 360.0)
        self.hue_spin.blockSignals(False)
        self.changed.emit()

    def _on_hue_spin_changed(self, hue: float):
        self.hue_slider.setHue(hue, notify=False)
        self.hue_slider.update()
        self.changed.emit()

    # --- public API ---

    def filter_name(self) -> str:
        return self.name_edit.text() or "Custom"

    def is_enabled(self) -> bool:
        return self.cb_enable.isChecked()

    def current_doc(self):
        return self.cmb_doc.currentData()

    def hue(self) -> float:
        return self.hue_slider.hue()

    def amount(self) -> float:
        return self.amt_spin.value()

    def reset(self):
        self.cb_enable.setChecked(False)
        key = self.cmb_preset.currentData()
        if key and key != "Custom":
            _, hue, _ = _FILTER_PRESETS[key]
        else:
            hue = 0.0
        self.hue_slider.setHue(hue, notify=False)
        self.hue_slider.update()
        self.hue_spin.blockSignals(True)
        self.hue_spin.setValue(hue)
        self.hue_spin.blockSignals(False)
        self.amt_spin.setValue(1.0)


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class NarrowbandIntegrationDialog(QDialog):
    """
    Screen narrowband (Ha/SII/OIII + user-defined) data into an RGB broadband image.
    Screen formula: out = rgb + colorized_nb - rgb * colorized_nb
    """

    def __init__(self, doc_manager=None, parent=None, window_icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("Narrowband Integration")
        if window_icon:
            self.setWindowIcon(window_icon)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        self.docman          = doc_manager
        self._zoom           = 1.0
        self._panning        = False
        self._pan_start_pos_vp  = None
        self._pan_start_scroll  = (0, 0)
        self._pan_deadzone   = 1
        self._dynamic_rows: list[DynamicFilterRow] = []
        self._state_restored = False
        self._cached_result_pm = None

        # Timer must exist before _build_ui
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()
        self._populate_views()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        root.addWidget(self.splitter)

        # ── LEFT PANE ──────────────────────────────────────────────────────
        left_widget = QWidget()
        left_widget.setMinimumWidth(380)
        left_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.splitter.addWidget(left_widget)

        left_outer = QVBoxLayout(left_widget)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(8)

        # Scrollable controls
        self._controls_container = QWidget()
        self._controls_layout    = QVBoxLayout(self._controls_container)
        self._controls_layout.setContentsMargins(4, 4, 4, 4)
        self._controls_layout.setSpacing(10)

        # ── RGB source ──────────────────────────────────────────────────────
        gb_rgb = QGroupBox("Broadband RGB Image")
        gl_rgb = QVBoxLayout(gb_rgb)
        rgb_row = QHBoxLayout()
        self.cmb_rgb = QComboBox()
        self.cmb_rgb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_rgb.setMinimumContentsLength(28)
        self.cmb_rgb.currentIndexChanged.connect(self._schedule_preview)
        rgb_row.addWidget(self.cmb_rgb, 1)
        btn_refresh = QPushButton("↺ Refresh")
        btn_refresh.setToolTip("Refresh dropdowns from currently open images")
        btn_refresh.setFixedWidth(80)
        btn_refresh.clicked.connect(self._refresh_views)
        rgb_row.addWidget(btn_refresh)
        gl_rgb.addLayout(rgb_row)
        self._controls_layout.addWidget(gb_rgb)

        # ── Fixed narrowband channels ───────────────────────────────────────
        self._nb_combos      = {}
        self._nb_hue_sliders = {}
        self._nb_hue_spins   = {}
        self._nb_amount      = {}
        self._nb_enabled     = {}

        nb_descriptions = {
            "Ha":   "Hα  656 nm  (hydrogen alpha)",
            "SII":  "SII  672 nm  (sulfur II)",
            "OIII": "OIII  496/501 nm  (oxygen III)",
        }

        for label in _NB_LABELS:
            gb = QGroupBox(nb_descriptions[label])
            g  = QVBoxLayout(gb)
            g.setSpacing(6)

            row0 = QHBoxLayout()
            cb_en = QCheckBox("Enable")
            cb_en.setChecked(False)
            cb_en.toggled.connect(lambda checked, lbl=label: self._on_nb_enable_toggled(lbl, checked))
            self._nb_enabled[label] = cb_en
            row0.addWidget(cb_en)

            cmb = QComboBox()
            cmb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(24)
            cmb.setEnabled(False)
            cmb.currentIndexChanged.connect(self._schedule_preview)
            self._nb_combos[label] = cmb
            row0.addWidget(cmb, 1)
            g.addLayout(row0)

            g.addWidget(QLabel("Colorize hue (°):"))

            hue_slider = HueGradientSlider(default_hue=_DEFAULT_HUE[label])
            hue_slider.setEnabled(False)
            hue_slider.hueChanged.connect(lambda h, lbl=label: self._on_hue_changed(lbl, h))
            self._nb_hue_sliders[label] = hue_slider
            g.addWidget(hue_slider)

            row3 = QHBoxLayout()
            row3.addWidget(QLabel("Hue:"))
            hue_spin = QDoubleSpinBox()
            hue_spin.setRange(0.0, 359.9)
            hue_spin.setSingleStep(1.0)
            hue_spin.setDecimals(1)
            hue_spin.setValue(_DEFAULT_HUE[label])
            hue_spin.setEnabled(False)
            hue_spin.valueChanged.connect(lambda v, lbl=label: self._on_hue_spin_changed(lbl, v))
            self._nb_hue_spins[label] = hue_spin
            row3.addWidget(hue_spin)

            row3.addSpacing(16)
            row3.addWidget(QLabel("Amount:"))
            amt_spin = QDoubleSpinBox()
            amt_spin.setRange(0.0, 2.0)
            amt_spin.setSingleStep(0.05)
            amt_spin.setDecimals(2)
            amt_spin.setValue(1.0)
            amt_spin.setEnabled(False)
            amt_spin.setToolTip("0 = no contribution, 1 = full screen, 2 = double weight")
            amt_spin.valueChanged.connect(self._schedule_preview)
            self._nb_amount[label] = amt_spin
            row3.addWidget(amt_spin)
            g.addLayout(row3)

            self._controls_layout.addWidget(gb)

        # ── Dynamic rows placeholder — inserted before the add button ───────
        # We'll keep a reference to insert dynamic rows above the add button
        self._dynamic_rows_layout = QVBoxLayout()
        self._dynamic_rows_layout.setSpacing(10)
        self._controls_layout.addLayout(self._dynamic_rows_layout)

        # ── Add Filter button ───────────────────────────────────────────────
        add_row = QHBoxLayout()
        self.btn_add_filter = QPushButton("＋  Add Filter")
        self.btn_add_filter.setToolTip("Add a custom narrowband filter channel")
        self.btn_add_filter.clicked.connect(self._add_dynamic_filter)
        add_row.addWidget(self.btn_add_filter)
        add_row.addStretch(1)
        self._controls_layout.addLayout(add_row)

        self._controls_layout.addStretch(1)

        # Scroll wrapper
        left_scroll = QScrollArea()
        left_scroll.setWidget(self._controls_container)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        left_outer.addWidget(left_scroll, 1)

        self.cb_small = QCheckBox("Small preview (fast)")
        self.cb_small.setChecked(True)
        self.cb_small.toggled.connect(self._schedule_preview)
        left_outer.addWidget(self.cb_small)

        btn_row = QHBoxLayout()
        self.btn_apply     = QPushButton("Apply to Current View")
        self.btn_apply_new = QPushButton("Apply as New Document")
        self.btn_reset     = QPushButton("↺ Reset")
        self.btn_apply.clicked.connect(self._apply_to_current)
        self.btn_apply_new.clicked.connect(self._apply_as_new)
        self.btn_reset.clicked.connect(self._reset_controls)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_apply_new)
        btn_row.addWidget(self.btn_reset)
        left_outer.addLayout(btn_row)

        # ── RIGHT PANE ─────────────────────────────────────────────────────
        right_widget = QWidget()
        right_widget.setMinimumWidth(420)
        right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.splitter.addWidget(right_widget)

        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)

        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out",      "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in",       "Zoom In")
        self.btn_zoom_1   = themed_toolbtn("zoom-original", "1:1")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit")
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_1)
        zoom_row.addWidget(self.btn_fit)
        zoom_row.addStretch(1)
        right.addLayout(zoom_row)

        self.btn_zoom_in.clicked.connect(lambda: self._apply_zoom(self._zoom * 1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._apply_zoom(self._zoom / 1.25))
        self.btn_zoom_1.clicked.connect(lambda: self._apply_zoom(1.0))
        self.btn_fit.clicked.connect(self._fit_to_preview)

        lbl_help = QLabel("🖱️  <b>Drag</b>: pan  &nbsp;•&nbsp;  <b>Wheel</b>: zoom")
        lbl_help.setTextFormat(Qt.TextFormat.RichText)
        lbl_help.setStyleSheet("color: #888; font-size: 11px;")
        right.addWidget(lbl_help)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_preview.setMinimumSize(10, 10)
        self.scroll.setWidget(self.lbl_preview)
        right.addWidget(self.scroll, 1)

        vp = self.scroll.viewport()
        vp.setMouseTracking(True)
        vp.installEventFilter(self)
        self.lbl_preview.setMouseTracking(True)
        self.lbl_preview.installEventFilter(self)

        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 900])

        self.setSizeGripEnabled(True)
        try:
            g = QGuiApplication.primaryScreen().availableGeometry()
            max_h = int(g.height() * 0.92)
            self.resize(1100, min(720, max_h))
            self.setMaximumHeight(max_h)
        except Exception:
            self.resize(1100, 720)

    # -----------------------------------------------------------------------
    # Dynamic filter management
    # -----------------------------------------------------------------------

    def _refresh_views(self):
        # Remember current selections by doc identity so we can restore them
        current_rgb = self.cmb_rgb.currentData()
        current_nb  = {lbl: self._nb_combos[lbl].currentData() for lbl in _NB_LABELS}

        self._populate_views()

        # Restore RGB selection if still open
        if current_rgb is not None:
            for i in range(self.cmb_rgb.count()):
                if self.cmb_rgb.itemData(i) is current_rgb:
                    self.cmb_rgb.setCurrentIndex(i)
                    break

        # Restore NB selections if still open
        for lbl in _NB_LABELS:
            prev = current_nb[lbl]
            if prev is not None:
                cmb = self._nb_combos[lbl]
                for i in range(cmb.count()):
                    if cmb.itemData(i) is prev:
                        cmb.setCurrentIndex(i)
                        break

    def _build_doc_items(self) -> list[tuple]:
        """Build the (name, doc) list used by dynamic row dropdowns."""
        items = [("— None —", None)]
        if self.docman is None:
            return items
        try:
            docs = self.docman.all_documents() or []
        except Exception:
            return items
        for d in docs:
            if hasattr(d, "image") and d.image is not None:
                items.append((self._doc_name(d), d))
        return items

    def _add_dynamic_filter(self):
        doc_items = self._build_doc_items()
        row = DynamicFilterRow(doc_items=doc_items, default_name="Custom",
                            default_hue=0.0, parent=None)
        row.changed.connect(self._schedule_preview)
        row.removeRequested.connect(self._remove_dynamic_filter)
        self._dynamic_rows.append(row)
        self._dynamic_rows_layout.addWidget(row)

    def _remove_dynamic_filter(self, row: DynamicFilterRow):
        if row in self._dynamic_rows:
            self._dynamic_rows.remove(row)
        self._dynamic_rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self._schedule_preview()

    # -----------------------------------------------------------------------
    # View population
    # -----------------------------------------------------------------------

    def _populate_views(self):
        if self.docman is None:
            return
        try:
            docs = self.docman.all_documents() or []
        except Exception:
            docs = []
        img_docs = [(d, self._doc_name(d)) for d in docs
                    if hasattr(d, "image") and d.image is not None]

        self.cmb_rgb.blockSignals(True)
        self.cmb_rgb.clear()
        self.cmb_rgb.addItem("— Select RGB image —", None)   # blank default
        for d, name in img_docs:
            self.cmb_rgb.addItem(name, d)
        self.cmb_rgb.setCurrentIndex(0)   # force blank
        self.cmb_rgb.blockSignals(False)

        for label in _NB_LABELS:
            cmb = self._nb_combos[label]
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem("— None —", None)
            for d, name in img_docs:
                cmb.addItem(name, d)
            cmb.blockSignals(False)

        titles = [name for _, name in img_docs]
        self._autofill_combos(img_docs, titles)

    def _doc_name(self, doc) -> str:
        try:
            return doc.display_name()
        except Exception:
            return "Untitled"

    def _autofill_combos(self, img_docs, titles):
        import re

        def score_nb(t, label):
            t = t.lower(); s = 0
            if label == 'ha':
                if re.search(r'\bha\b|hα|h.alpha|halpha', t): s += 10
                if re.search(r'[_\-\.]ha[_\-\.\s]|[_\-\.]ha$', t): s += 8
            elif label == 'sii':
                if re.search(r'\bsii\b|s2\b|sulfur', t): s += 10
                if re.search(r'[_\-\.]sii[_\-\.\s]|[_\-\.]sii$|[_\-\.]s2[_\-\.]', t): s += 8
            elif label == 'oiii':
                if re.search(r'\boiii\b|o3\b|oxygen', t): s += 10
                if re.search(r'[_\-\.]oiii[_\-\.\s]|[_\-\.]oiii$|[_\-\.]o3[_\-\.]', t): s += 8
            return s

        if not titles:
            return

        for label in _NB_LABELS:
            scores = [score_nb(t, label.lower()) for t in titles]
            best   = max(range(len(scores)), key=lambda i: scores[i])
            if scores[best] > 0:
                cmb = self._nb_combos[label]
                cmb.blockSignals(True)
                cmb.setCurrentIndex(best + 1)  # +1 for "— None —"
                cmb.blockSignals(False)
                # enable the checkbox and controls — signals blocked so no preview fires
                self._nb_enabled[label].blockSignals(True)
                self._nb_enabled[label].setChecked(True)
                self._nb_enabled[label].blockSignals(False)
                self._nb_combos[label].setEnabled(True)
                self._nb_hue_sliders[label].setEnabled(True)
                self._nb_hue_spins[label].setEnabled(True)
                self._nb_amount[label].setEnabled(True)
    # -----------------------------------------------------------------------
    # Fixed channel enable/hue sync
    # -----------------------------------------------------------------------

    def _on_nb_enable_toggled(self, label: str, checked: bool):
        self._nb_combos[label].setEnabled(checked)
        self._nb_hue_sliders[label].setEnabled(checked)
        self._nb_hue_spins[label].setEnabled(checked)
        self._nb_amount[label].setEnabled(checked)
        self._schedule_preview()

    def _on_hue_changed(self, label: str, hue: float):
        spin = self._nb_hue_spins[label]
        spin.blockSignals(True)
        spin.setValue(hue % 360.0)
        spin.blockSignals(False)
        self._schedule_preview()

    def _on_hue_spin_changed(self, label: str, hue: float):
        slider = self._nb_hue_sliders[label]
        slider.setHue(hue, notify=False)
        slider.update()
        self._schedule_preview()

    # -----------------------------------------------------------------------
    # Preview
    # -----------------------------------------------------------------------

    def _schedule_preview(self, *_):
        self._preview_timer.stop()
        self._preview_timer.start(300)

    def _get_rgb_image(self) -> np.ndarray | None:
        doc = self.cmb_rgb.currentData()
        if doc is None or not hasattr(doc, "image") or doc.image is None:
            return None
        return _ensure_rgb(_to_f32(doc.image))

    def _get_nb_image_fixed(self, label: str) -> np.ndarray | None:
        if not self._nb_enabled[label].isChecked():
            return None
        doc = self._nb_combos[label].currentData()
        if doc is None or not hasattr(doc, "image") or doc.image is None:
            return None
        return _ensure_mono(_to_f32(doc.image))

    def _compute_result(self, rgb: np.ndarray) -> np.ndarray:
        out = rgb.copy()

        # Fixed channels
        for label in _NB_LABELS:
            nb_mono = self._get_nb_image_fixed(label)
            if nb_mono is None:
                continue
            hue    = float(self._nb_hue_sliders[label].hue())
            amount = float(self._nb_amount[label].value())
            if amount < 1e-4:
                continue
            out = self._apply_nb_screen(out, nb_mono, hue, amount)

        # Dynamic channels
        for row in self._dynamic_rows:
            if not row.is_enabled():
                continue
            doc = row.current_doc()
            if doc is None or not hasattr(doc, "image") or doc.image is None:
                continue
            nb_mono = _ensure_mono(_to_f32(doc.image))
            hue     = row.hue()
            amount  = row.amount()
            if amount < 1e-4:
                continue
            out = self._apply_nb_screen(out, nb_mono, hue, amount)

        return np.clip(out, 0.0, 1.0)

    def _apply_nb_screen(self, out: np.ndarray, nb_mono: np.ndarray,
                         hue: float, amount: float) -> np.ndarray:
        rh, rw = out.shape[:2]
        mh, mw = nb_mono.shape[:2]
        if (mh, mw) != (rh, rw):
            if cv2 is not None:
                nb_mono = cv2.resize(nb_mono, (rw, rh), interpolation=cv2.INTER_AREA)
            else:
                nb_mono = nb_mono[
                    np.linspace(0, mh-1, rh).astype(int)[:, None],
                    np.linspace(0, mw-1, rw).astype(int)[None, :]
                ]
        colorized = _colorize(nb_mono, hue)
        colorized_scaled = np.clip(colorized * amount, 0.0, 1.0)
        return _screen(out, colorized_scaled)

    def _update_preview(self):
        if not self.isVisible():
            return
        rgb = self._get_rgb_image()
        if rgb is None:
            self.lbl_preview.setText("Select a broadband RGB image.")
            self._cached_result_pm = None
            return
        base   = _downsample(rgb, 1200) if self.cb_small.isChecked() else rgb
        result = self._compute_result(base)
        self._cached_result_pm = _to_pixmap(result)
        self._cached_result_size = (result.shape[1], result.shape[0])  # w, h
        self._redisplay_preview()

    def _redisplay_preview(self):
        if self._cached_result_pm is None:
            return
        w, h  = self._cached_result_size
        zw    = max(1, int(round(w * self._zoom)))
        zh    = max(1, int(round(h * self._zoom)))
        pm_scaled = self._cached_result_pm.scaled(
            zw, zh,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview.setPixmap(pm_scaled)
        self.lbl_preview.resize(zw, zh)

    # -----------------------------------------------------------------------
    # Zoom / pan
    # -----------------------------------------------------------------------

    def _current_scroll(self):
        hb = self.scroll.horizontalScrollBar()
        vb = self.scroll.verticalScrollBar()
        return hb.value(), vb.value(), hb.maximum(), vb.maximum()

    def _set_scroll(self, x, y):
        hb = self.scroll.horizontalScrollBar()
        vb = self.scroll.verticalScrollBar()
        hb.setValue(int(max(0, min(x, hb.maximum()))))
        vb.setValue(int(max(0, min(y, vb.maximum()))))

    def _apply_zoom(self, new_zoom: float, anchor_label_pos=None):
        old_zoom = self._zoom
        new_zoom = max(0.05, min(16.0, float(new_zoom)))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        sx, sy, _, _ = self._current_scroll()
        vp = self.scroll.viewport().rect()
        if anchor_label_pos is None:
            cx = (sx + vp.width()  / 2.0) / max(old_zoom, 1e-9)
            cy = (sy + vp.height() / 2.0) / max(old_zoom, 1e-9)
        else:
            cx = float(anchor_label_pos.x())
            cy = float(anchor_label_pos.y())
        pvx = cx * old_zoom - sx
        pvy = cy * old_zoom - sy
        self._zoom = new_zoom
        self._redisplay_preview()   # ← was _update_preview
        self._set_scroll(cx * new_zoom - pvx, cy * new_zoom - pvy)

    def _fit_to_preview(self):
        pm = self.lbl_preview.pixmap()
        if pm is None or pm.isNull():
            return
        vp    = self.scroll.viewport().size()
        orig_w = max(1, int(round(pm.width()  / self._zoom)))
        orig_h = max(1, int(round(pm.height() / self._zoom)))
        k = min(vp.width() / orig_w, vp.height() / orig_h)
        self._apply_zoom(k)

    def eventFilter(self, obj, ev):
        from PyQt6.QtCore import QEvent

        def _vp_pos(o, e):
            if o is self.scroll.viewport():
                return e.position()
            return self.lbl_preview.mapTo(self.scroll.viewport(), e.position().toPoint())

        if obj in (self.scroll.viewport(), self.lbl_preview) and ev.type() == QEvent.Type.Wheel:
            anchor = ev.position() if obj is self.lbl_preview else \
                     self.lbl_preview.mapFrom(self.scroll.viewport(), ev.position().toPoint())
            dy = ev.pixelDelta().y()
            if dy != 0:
                factor = 1.03 if dy > 0 else 1.0 / 1.03
            else:
                dy = ev.angleDelta().y()
                factor = 1.15 if dy > 0 else 1.0 / 1.15
            self._apply_zoom(self._zoom * factor, anchor_label_pos=anchor)
            ev.accept()
            return True

        if obj in (self.scroll.viewport(), self.lbl_preview):
            if ev.type() == QEvent.Type.MouseButtonPress:
                if ev.button() == Qt.MouseButton.LeftButton:
                    self._panning = True
                    self._pan_start_pos_vp = _vp_pos(obj, ev)
                    hb = self.scroll.horizontalScrollBar()
                    vb = self.scroll.verticalScrollBar()
                    self._pan_start_scroll = (hb.value(), vb.value())
                    self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            elif ev.type() == QEvent.Type.MouseMove and self._panning:
                cur = _vp_pos(obj, ev)
                dx  = cur.x() - self._pan_start_pos_vp.x()
                dy  = cur.y() - self._pan_start_pos_vp.y()
                hb  = self.scroll.horizontalScrollBar()
                vb  = self.scroll.verticalScrollBar()
                hb.setValue(int(self._pan_start_scroll[0] - dx))
                vb.setValue(int(self._pan_start_scroll[1] - dy))
                return True
            elif ev.type() in (QEvent.Type.MouseButtonRelease, QEvent.Type.Leave):
                if self._panning:
                    self._panning = False
                    self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                    return True

        return super().eventFilter(obj, ev)


    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._state_restored and self.isVisible():
            self._redisplay_preview() 

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_controls(self):
        for label in _NB_LABELS:
            self._nb_enabled[label].setChecked(False)
            self._nb_hue_sliders[label].setHue(_DEFAULT_HUE[label], notify=False)
            self._nb_hue_sliders[label].update()
            self._nb_hue_spins[label].blockSignals(True)
            self._nb_hue_spins[label].setValue(_DEFAULT_HUE[label])
            self._nb_hue_spins[label].blockSignals(False)
            self._nb_amount[label].setValue(1.0)
        for row in self._dynamic_rows:
            row.reset()
        self._schedule_preview()

    # -----------------------------------------------------------------------
    # Apply
    # -----------------------------------------------------------------------

    def _apply_fullres(self) -> np.ndarray | None:
        doc = self.cmb_rgb.currentData()
        if doc is None or not hasattr(doc, "image") or doc.image is None:
            QMessageBox.information(self, "No image", "Please select a broadband RGB image.")
            return None
        return self._compute_result(_ensure_rgb(_to_f32(doc.image)))

    def _apply_to_current(self):
        result = self._apply_fullres()
        if result is None:
            return
        doc = self.cmb_rgb.currentData()
        try:
            if hasattr(doc, "set_image"):
                doc.set_image(result)
                return
        except Exception:
            pass
        self._open_result(result, doc, "[NB Integration]")

    def _apply_as_new(self):
        result = self._apply_fullres()
        if result is None:
            return
        self._open_result(result, self.cmb_rgb.currentData(), "[NB Integration]")

    def _open_result(self, result: np.ndarray, source_doc, suffix: str):
        if self.docman is None:
            QMessageBox.warning(self, "No document manager", "Cannot create new document.")
            return
        title = f"{self._doc_name(source_doc)} {suffix}" if source_doc else suffix
        try:
            if hasattr(self.docman, "open_array"):
                self.docman.open_array(result, title=title)
            elif hasattr(self.docman, "open_numpy"):
                self.docman.open_numpy(result, title=title)
            else:
                self.docman.create_document(image=result, name=title)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create document:\n{e}")

    # -----------------------------------------------------------------------
    # Window state
    # -----------------------------------------------------------------------

    def _save_state(self):
        try:
            s = QSettings()
            s.setValue("nbintegration/geometry",      self.saveGeometry())
            s.setValue("nbintegration/splitter",       self.splitter.saveState())
            s.setValue("nbintegration/small_preview",  self.cb_small.isChecked())
            s.sync()
        except Exception:
            pass

    def _restore_state(self):
        try:
            s   = QSettings()
            geo = s.value("nbintegration/geometry")
            if geo is not None and len(geo) > 0:
                self.restoreGeometry(geo)
            sp  = s.value("nbintegration/splitter")
            if sp is not None and len(sp) > 0:
                self.splitter.restoreState(sp)
            self.cb_small.setChecked(bool(s.value("nbintegration/small_preview", True, type=bool)))
        except Exception:
            pass
        # Always trigger first preview here, after geometry is settled
        self._schedule_preview()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._state_restored:
            return
        self._state_restored = True
        QTimer.singleShot(0, self._restore_state)


    def closeEvent(self, ev):
        self._save_state()
        super().closeEvent(ev)