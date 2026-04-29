#saspro/narrowbandintegration.py

from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QImage, QPixmap, QIcon, QGuiApplication
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QComboBox, QSlider, QGroupBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QDoubleSpinBox,
    QScrollArea, QFrame, QSplitter, QSizePolicy, QCheckBox
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
# Wavelength → approximate RGB hue (degrees)
# Ha  656nm  →  ~0°   (deep red)
# SII 672nm  →  ~345° (slightly crimson-red, just below red)
# OIII 496nm →  ~185° (cyan-teal)
# ---------------------------------------------------------------------------
_DEFAULT_HUE = {
    "Ha":   4.0,    # 656nm — warm red, slight orange shift
    "SII":  0.0,    # 672nm — pure/deep red (redder than Ha, at the spectrum edge)
    "OIII": 156.0,  # 496nm — green-cyan (NOT pure cyan — this was the main error)
}

_NB_LABELS = ["Ha", "SII", "OIII"]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_f32(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    return np.clip(a, 0.0, 1.0)


def _ensure_mono(arr: np.ndarray) -> np.ndarray:
    """Return HxW float32."""
    a = _to_f32(arr)
    if a.ndim == 3:
        # luminance
        return (0.2989 * a[..., 0] + 0.5870 * a[..., 1] + 0.1140 * a[..., 2]).astype(np.float32)
    return a


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Return HxWx3 float32."""
    a = _to_f32(arr)
    if a.ndim == 2:
        return np.repeat(a[..., None], 3, axis=2)
    return a


def _colorize(mono: np.ndarray, hue_deg: float) -> np.ndarray:
    """
    Colorize a mono image (HxW, 0..1) with a given hue (degrees).
    Returns HxWx3 float32.
    Strategy: HSV with S=1, V=pixel value, H=hue_deg.
    """
    h = float(hue_deg) % 360.0
    hi = int(h / 60.0) % 6
    f  = (h / 60.0) - int(h / 60.0)
    v  = mono.astype(np.float32)  # HxW

    p = np.zeros_like(v)            # v*(1-s)=0
    q = v * (1.0 - f)
    t = v * f

    rgb_sectors = [
        (v, t, p),  # 0
        (q, v, p),  # 1
        (p, v, t),  # 2
        (p, q, v),  # 3
        (t, p, v),  # 4
        (v, p, q),  # 5
    ]
    r_ch, g_ch, b_ch = rgb_sectors[hi]
    return np.clip(np.stack([r_ch, g_ch, b_ch], axis=-1), 0.0, 1.0).astype(np.float32)


def _screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Screen blend: a + b - a*b  (both HxWx3, result HxWx3)."""
    return np.clip(a + b - a * b, 0.0, 1.0).astype(np.float32)


def _to_pixmap(img01: np.ndarray) -> QPixmap:
    a = np.clip(_ensure_rgb(img01), 0.0, 1.0)
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
# Hue gradient widget
# ---------------------------------------------------------------------------

class HueGradientSlider(QWidget):
    """
    A slider with a hue-spectrum gradient track (0..360 degrees).
    Emits hueChanged(float) when value changes.
    """
    from PyQt6.QtCore import pyqtSignal
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
        # Build hue row
        hues = np.linspace(0, 179, w, dtype=np.uint8)
        sat  = np.full(w, 220, dtype=np.uint8)
        val  = np.full(w, 230, dtype=np.uint8)
        hsv_row = np.stack([hues, sat, val], axis=-1)[None, :, :]  # 1 x W x 3
        if cv2 is not None:
            rgb_row = cv2.cvtColor(hsv_row, cv2.COLOR_HSV2RGB)
        else:
            # fallback: just a rainbow using hue directly
            r = np.clip(np.abs(hues.astype(float)/180*6 - 3) - 1, 0, 1)
            g = np.clip(2 - np.abs(hues.astype(float)/180*6 - 2), 0, 1)
            b = np.clip(2 - np.abs(hues.astype(float)/180*6 - 4), 0, 1)
            rgb_row = np.stack([r, g, b], axis=-1)[None] * 230
            rgb_row = rgb_row.astype(np.uint8)
        row = np.repeat(rgb_row, h, axis=0)
        qimg = QImage(row.data, w, h, row.strides[0], QImage.Format.Format_RGB888).copy()
        self._gradient_img = qimg

    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        track_h = 14
        track_y = (self.height() - track_h) // 2
        track_rect_x = 10
        track_rect_w = self.width() - 20

        # Draw gradient track
        self._build_gradient(track_rect_w, track_h)
        p.drawImage(track_rect_x, track_y, self._gradient_img)

        # Track border
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(track_rect_x, track_y, track_rect_w, track_h)

        # Thumb position
        tx = int(track_rect_x + (self._hue / 360.0) * track_rect_w)
        ty = self.height() // 2

        # Shadow
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(0, 0, 0, 60)))
        p.drawEllipse(tx - 9, ty - 9, 18, 18)

        # Thumb fill — use the hue color itself
        hsv_pix = np.array([[[int(self._hue / 2), 220, 230]]], dtype=np.uint8)
        if cv2 is not None:
            rgb_pix = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2RGB)[0, 0]
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
# Main dialog
# ---------------------------------------------------------------------------

class NarrowbandIntegrationDialog(QDialog):
    """
    Screen narrowband (Ha/SII/OIII) data into an RGB broadband image.
    Each narrowband channel gets a colorize hue slider and an amount knob.
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

        self.docman = doc_manager
        self._zoom = 1.0
        self._panning = False
        self._pan_start_pos_vp = None
        self._pan_start_scroll = (0, 0)
        self._pan_deadzone = 1

        # Timer must exist before _build_ui() so signal connections during
        # widget construction can't fire _schedule_preview() on a missing attribute
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)

        self._build_ui()
        self._populate_views()

        QTimer.singleShot(0, self._update_preview)

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
        left_widget.setMinimumWidth(360)
        left_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.splitter.addWidget(left_widget)

        left_outer = QVBoxLayout(left_widget)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(8)

        # Scroll area for controls
        controls_container = QWidget()
        left = QVBoxLayout(controls_container)
        left.setContentsMargins(4, 4, 4, 4)
        left.setSpacing(10)

        # ── RGB source ──────────────────────────────────────────────────────
        gb_rgb = QGroupBox("Broadband RGB Image")
        gl_rgb = QVBoxLayout(gb_rgb)
        self.cmb_rgb = QComboBox()
        self.cmb_rgb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_rgb.setMinimumContentsLength(28)
        self.cmb_rgb.currentIndexChanged.connect(self._schedule_preview)
        gl_rgb.addWidget(self.cmb_rgb)
        left.addWidget(gb_rgb)

        # ── Narrowband channels ─────────────────────────────────────────────
        self._nb_combos   = {}   # label → QComboBox
        self._nb_hue_sliders = {}  # label → HueGradientSlider
        self._nb_hue_spins   = {}  # label → QDoubleSpinBox
        self._nb_amount      = {}  # label → QDoubleSpinBox
        self._nb_enabled     = {}  # label → QCheckBox

        nb_descriptions = {
            "Ha":   "Hα  656 nm  (hydrogen alpha)",
            "SII":  "SII  672 nm  (sulfur II)",
            "OIII": "OIII  496/501 nm  (oxygen III)",
        }

        for label in _NB_LABELS:
            gb = QGroupBox(nb_descriptions[label])
            g  = QVBoxLayout(gb)
            g.setSpacing(6)

            # Row 0: enabled checkbox + dropdown
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

            # Row 1: hue label
            hue_label_row = QHBoxLayout()
            hue_label_row.addWidget(QLabel("Colorize hue (°):"))
            hue_label_row.addStretch()
            g.addLayout(hue_label_row)

            # Row 2: gradient slider
            hue_slider = HueGradientSlider(default_hue=_DEFAULT_HUE[label])
            hue_slider.setEnabled(False)
            hue_slider.hueChanged.connect(lambda h, lbl=label: self._on_hue_changed(lbl, h))
            self._nb_hue_sliders[label] = hue_slider
            g.addWidget(hue_slider)

            # Row 3: spin for precise hue entry + amount
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

            left.addWidget(gb)

        left.addStretch(1)

        # Scroll wrapper
        left_scroll = QScrollArea()
        left_scroll.setWidget(controls_container)
        left_scroll.setWidgetResizable(False)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        left_outer.addWidget(left_scroll, 1)

        # Small preview toggle (outside scroll)
        self.cb_small = QCheckBox("Small preview (fast)")
        self.cb_small.setChecked(True)
        self.cb_small.toggled.connect(self._schedule_preview)
        left_outer.addWidget(self.cb_small)

        # Buttons
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

        # Zoom toolbar
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

        # Preview scroll area
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

        # Splitter config
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([400, 900])

        self.setSizeGripEnabled(True)
        try:
            g = QGuiApplication.primaryScreen().availableGeometry()
            max_h = int(g.height() * 0.92)
            self.resize(1100, min(720, max_h))
            self.setMaximumHeight(max_h)
        except Exception:
            self.resize(1100, 720)

    # -----------------------------------------------------------------------
    # View population
    # -----------------------------------------------------------------------

    def _populate_views(self):
        """Populate all combo boxes from open documents."""
        if self.docman is None:
            return

        docs = []
        try:
            docs = self.docman.all_documents() or []
        except Exception:
            pass
        img_docs = [(d, self._doc_name(d)) for d in docs
                    if hasattr(d, "image") and d.image is not None]

        none_item = ("— None —", None)

        # RGB combo: no None option (must pick something)
        self.cmb_rgb.blockSignals(True)
        self.cmb_rgb.clear()
        for d, name in img_docs:
            self.cmb_rgb.addItem(name, d)
        self.cmb_rgb.blockSignals(False)

        # NB combos: include None
        for label in _NB_LABELS:
            cmb = self._nb_combos[label]
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItem(none_item[0], none_item[1])
            for d, name in img_docs:
                cmb.addItem(name, d)
            cmb.blockSignals(False)

        # Auto-guess from names
        titles = [name for _, name in img_docs]
        self._autofill_combos(img_docs, titles)

    def _doc_name(self, doc) -> str:
        try:
            return doc.display_name()
        except Exception:
            return "Untitled"

    def _autofill_combos(self, img_docs, titles):
        """Guess which doc is RGB, which are Ha/SII/OIII using name heuristics."""
        import re

        def score_rgb(t: str) -> int:
            t = t.lower()
            s = 0
            if re.search(r'\brgb\b', t): s += 10
            if re.search(r'\bbroadband\b', t): s += 8
            if re.search(r'\blight\b', t): s += 4
            # penalise obvious NB names
            for kw in ('ha', 'hα', 'sii', 'oiii', 'nb', 'narrowband'):
                if kw in t: s -= 8
            return s

        def score_nb(t: str, label: str) -> int:
            t = t.lower()
            s = 0
            lbl = label.lower()
            if lbl == 'ha':
                if re.search(r'\bha\b|hα|h.alpha|halpha', t): s += 10
                if re.search(r'[_\-\.]ha[_\-\.\s]|[_\-\.]ha$', t): s += 8
            elif lbl == 'sii':
                if re.search(r'\bsii\b|s2\b|sulfur', t): s += 10
                if re.search(r'[_\-\.]sii[_\-\.\s]|[_\-\.]sii$|[_\-\.]s2[_\-\.]', t): s += 8
            elif lbl == 'oiii':
                if re.search(r'\boiii\b|o3\b|oxygen', t): s += 10
                if re.search(r'[_\-\.]oiii[_\-\.\s]|[_\-\.]oiii$|[_\-\.]o3[_\-\.]', t): s += 8
            return s

        if not titles:
            return

        # Best RGB guess
        rgb_scores = [score_rgb(t) for t in titles]
        best_rgb = max(range(len(rgb_scores)), key=lambda i: rgb_scores[i])
        if rgb_scores[best_rgb] > 0:
            self.cmb_rgb.setCurrentIndex(best_rgb)

        # Best NB guesses
        for label in _NB_LABELS:
            scores = [score_nb(t, label) for t in titles]
            best = max(range(len(scores)), key=lambda i: scores[i])
            if scores[best] > 0:
                # +1 because index 0 is "— None —"
                self._nb_combos[label].setCurrentIndex(best + 1)
                # auto-enable if a good match was found
                self._nb_enabled[label].setChecked(True)

    # -----------------------------------------------------------------------
    # Enable/disable per-channel controls
    # -----------------------------------------------------------------------

    def _on_nb_enable_toggled(self, label: str, checked: bool):
        self._nb_combos[label].setEnabled(checked)
        self._nb_hue_sliders[label].setEnabled(checked)
        self._nb_hue_spins[label].setEnabled(checked)
        self._nb_amount[label].setEnabled(checked)
        self._schedule_preview()

    # -----------------------------------------------------------------------
    # Hue sync between slider and spin
    # -----------------------------------------------------------------------

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
        self._preview_timer.start(150)

    def _get_rgb_image(self) -> np.ndarray | None:
        doc = self.cmb_rgb.currentData()
        if doc is None or not hasattr(doc, "image") or doc.image is None:
            return None
        return _ensure_rgb(_to_f32(doc.image))

    def _get_nb_image(self, label: str) -> np.ndarray | None:
        if not self._nb_enabled[label].isChecked():
            return None
        doc = self._nb_combos[label].currentData()
        if doc is None or not hasattr(doc, "image") or doc.image is None:
            return None
        return _ensure_mono(_to_f32(doc.image))

    def _compute_result(self, rgb: np.ndarray) -> np.ndarray:
        """Apply all enabled NB screens onto a copy of rgb."""
        out = rgb.copy()
        for label in _NB_LABELS:
            nb_mono = self._get_nb_image(label)
            if nb_mono is None:
                continue
            hue    = float(self._nb_hue_sliders[label].hue())
            amount = float(self._nb_amount[label].value())
            if amount < 1e-4:
                continue

            # Resize NB to match RGB if needed
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

            # Scale colorized by amount (amount > 1 → brighten before screen for stronger effect)
            colorized_scaled = np.clip(colorized * amount, 0.0, 1.0)

            out = _screen(out, colorized_scaled)

        return np.clip(out, 0.0, 1.0)

    def _update_preview(self):
        rgb = self._get_rgb_image()
        if rgb is None:
            self.lbl_preview.setText("Select a broadband RGB image.")
            return

        if self.cb_small.isChecked():
            rgb_small = _downsample(rgb, 1200)
        else:
            rgb_small = rgb

        result = self._compute_result(rgb_small)
        pm = _to_pixmap(result)
        h, w = result.shape[:2]
        zw = max(1, int(round(w * self._zoom)))
        zh = max(1, int(round(h * self._zoom)))
        pm_scaled = pm.scaled(zw, zh,
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
        self._update_preview()
        nx = cx * new_zoom - pvx
        ny = cy * new_zoom - pvy
        self._set_scroll(nx, ny)

    def _fit_to_preview(self):
        pm = self.lbl_preview.pixmap()
        if pm is None or pm.isNull():
            return
        vp = self.scroll.viewport().size()
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

        # Wheel zoom
        if obj in (self.scroll.viewport(), self.lbl_preview) and ev.type() == QEvent.Type.Wheel:
            if obj is self.lbl_preview:
                anchor = ev.position()
            else:
                anchor = self.lbl_preview.mapFrom(self.scroll.viewport(), ev.position().toPoint())
            dy = ev.pixelDelta().y()
            if dy != 0:
                factor = 1.03 if dy > 0 else 1.0 / 1.03
            else:
                dy = ev.angleDelta().y()
                factor = 1.15 if dy > 0 else 1.0 / 1.15
            self._apply_zoom(self._zoom * factor, anchor_label_pos=anchor)
            ev.accept()
            return True

        # Pan (Ctrl + drag)
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
                dx = cur.x() - self._pan_start_pos_vp.x()
                dy = cur.y() - self._pan_start_pos_vp.y()
                hb = self.scroll.horizontalScrollBar()
                vb = self.scroll.verticalScrollBar()
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
        QTimer.singleShot(0, self._update_preview)

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def _reset_controls(self):
        for label in _NB_LABELS:
            self._nb_enabled[label].setChecked(False)
            self._nb_hue_sliders[label].setHue(_DEFAULT_HUE[label], notify=False)
            self._nb_hue_spins[label].blockSignals(True)
            self._nb_hue_spins[label].setValue(_DEFAULT_HUE[label])
            self._nb_hue_spins[label].blockSignals(False)
            self._nb_amount[label].setValue(1.0)
        self._schedule_preview()

    # -----------------------------------------------------------------------
    # Apply
    # -----------------------------------------------------------------------

    def _apply_fullres(self) -> np.ndarray | None:
        doc = self.cmb_rgb.currentData()
        if doc is None or not hasattr(doc, "image") or doc.image is None:
            QMessageBox.information(self, "No image", "Please select a broadband RGB image.")
            return None
        rgb = _ensure_rgb(_to_f32(doc.image))
        return self._compute_result(rgb)

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
        # fallback: open as new
        self._open_result(result, doc, suffix="[NB Integration]")

    def _apply_as_new(self):
        result = self._apply_fullres()
        if result is None:
            return
        doc = self.cmb_rgb.currentData()
        self._open_result(result, doc, suffix="[NB Integration]")

    def _open_result(self, result: np.ndarray, source_doc, suffix: str):
        if self.docman is None:
            QMessageBox.warning(self, "No document manager", "Cannot create new document.")
            return
        name = self._doc_name(source_doc) if source_doc else "Result"
        title = f"{name} {suffix}"
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
    # Window state persistence
    # -----------------------------------------------------------------------

    def _save_state(self):
        try:
            s = QSettings()
            s.setValue("nbintegration/geometry", self.saveGeometry())
            s.setValue("nbintegration/splitter", self.splitter.saveState())
            s.setValue("nbintegration/small_preview", self.cb_small.isChecked())
            s.sync()
        except Exception:
            pass

    def _restore_state(self):
        try:
            s = QSettings()
            geo = s.value("nbintegration/geometry")
            if geo is not None and len(geo) > 0:
                self.restoreGeometry(geo)
            sp = s.value("nbintegration/splitter")
            if sp is not None and len(sp) > 0:
                self.splitter.restoreState(sp)
            self.cb_small.setChecked(bool(s.value("nbintegration/small_preview", True, type=bool)))
        except Exception:
            pass

    def showEvent(self, ev):
        super().showEvent(ev)
        if getattr(self, "_state_restored", False):
            return
        self._state_restored = True
        QTimer.singleShot(0, self._restore_state)

    def closeEvent(self, ev):
        self._save_state()
        super().closeEvent(ev)