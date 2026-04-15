#src/setiastro/saspro/selective_luma.py
from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PyQt6.QtCore import Qt, QTimer, QSettings, QByteArray
from PyQt6.QtGui import QImage, QPixmap, QIcon, QGuiApplication, QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient, QConicalGradient
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QGroupBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QScrollArea, QFrame, QTabWidget, QSplitter
)
from PyQt6.QtWidgets import QSizePolicy
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# ---------------------------------------------------------------------
# Small helpers  (identical to SelectiveColor)
# ---------------------------------------------------------------------

def _to_uint8_rgb(img01: np.ndarray) -> np.ndarray:
    a = np.clip(img01, 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return (a * 255.0 + 0.5).astype(np.uint8)

def _to_pixmap(img01: np.ndarray) -> QPixmap:
    a = _to_uint8_rgb(img01)
    h, w, _ = a.shape
    qimg = QImage(a.data, w, h, a.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

def _luminance01(img01: np.ndarray) -> np.ndarray:
    if img01.ndim == 2:
        return np.clip(img01, 0.0, 1.0).astype(np.float32)
    r, g, b = img01[..., 0], img01[..., 1], img01[..., 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)

def _softstep(x, edge0, edge1):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def _ensure_rgb01(img: np.ndarray) -> np.ndarray:
    a = np.clip(img.astype(np.float32), 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return a

# ---------------------------------------------------------------------
# Luminance-band mask  (replaces hue-band mask)
# ---------------------------------------------------------------------

# Named luminance presets (0..1 range, lo/hi)
_LUM_PRESETS = {
    "Shadows":     (0.00, 0.25),
    "Dark Mids":   (0.20, 0.45),
    "Midtones":    (0.35, 0.65),
    "Bright Mids": (0.55, 0.80),
    "Highlights":  (0.75, 1.00),
}

def _lum_band_mask(L: np.ndarray, lo: float, hi: float,
                   smooth: float, invert: bool = False) -> np.ndarray:
    """
    Soft luminance band mask on L in [0..1].
    `smooth` is feather width expressed as a fraction of the 0..1 range (e.g. 0.05).
    """
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    if lo > hi:
        lo, hi = hi, lo

    s = float(max(smooth, 0.0))

    # Hard band
    mask = ((L >= lo) & (L <= hi)).astype(np.float32)

    if s > 1e-6:
        # Lower feather: fade in just above lo
        lower = (L >= lo - s) & (L < lo)
        mask[lower] = np.maximum(mask[lower], (L[lower] - (lo - s)) / s)

        # Upper feather: fade out just above hi
        upper = (L > hi) & (L <= hi + s)
        mask[upper] = np.maximum(mask[upper], 1.0 - (L[upper] - hi) / s)

    if invert:
        mask = 1.0 - mask

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _build_lum_mask(img01: np.ndarray,
                    lo: float, hi: float,
                    smooth: float,
                    invert: bool,
                    blur_px: int) -> np.ndarray:
    L = _luminance01(img01)
    mask = _lum_band_mask(L, lo, hi, smooth, invert)

    if blur_px > 0 and cv2 is not None:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(blur_px))

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------
# Color / luminance adjustments  (same as SelectiveColor, unchanged)
# ---------------------------------------------------------------------
def _band_contrast(out: np.ndarray, m: np.ndarray,
                   con: float, band_lo: float, band_hi: float) -> np.ndarray:
    """
    Anchored sigmoid S-curve contrast within [band_lo, band_hi].

      - con = 0  → perfect identity (returns out unchanged).
      - con > 0  → expanding S-curve (more contrast).
      - con < 0  → compressing curve (less contrast, bows toward midpoint).
      - band_lo and band_hi are the sigmoid anchors (always map to themselves).
      - Spatial blending is handled entirely by the luminance mask m —
        no hard in_band boolean gate, which would create a seam at the edges.

    The sigmoid is computed for every pixel relative to the band, but pixels
    outside the band have m≈0 so the blend leaves them untouched. Pixels
    near the band edges are already feathered by m from _build_lum_mask.
    """
    import math

    span = float(band_hi) - float(band_lo)
    if span < 1e-6 or abs(con) < 1e-4:
        return out

    k = abs(float(con)) * 6.0   # steepness, always positive

    # Sigmoid normalisation anchors so expand(0,k)=0 and expand(1,k)=1
    s_neg = 1.0 / (1.0 + math.exp( k))   # sig(-k)
    s_pos = 1.0 / (1.0 + math.exp(-k))   # sig(+k)
    s_rng = s_pos - s_neg                  # always > 0

    out_f = out.astype(np.float32)

    # Compute t for ALL pixels relative to the band.
    # Out-of-band pixels will have t outside [0,1] but they are masked out
    # by m≈0 so they contribute nothing to the blend.
    t = (out_f - band_lo) / span          # HxWx3 (or HxW), may be outside [0,1]
    u = 2.0 * t - 1.0                     # centred on [-1, 1]

    raw = 1.0 / (1.0 + np.exp(-k * u))   # sigmoid(k*u), shape matches out_f
    e   = (raw - s_neg) / s_rng           # expand(t,k), anchored to [0,1] within band

    if con > 0:
        anchored = e                       # bow above diagonal
    else:
        anchored = 2.0 * t - e            # flip deviation: bow below diagonal

    # Remap back to pixel values — no clamp here so OOB pixels remap cleanly
    # (they'll be zeroed out by m anyway)
    result = anchored * span + band_lo

    # Blend using the luminance mask which already carries all the feathering
    m3      = m[..., None] if out_f.ndim == 3 else m
    blended = out_f * (1.0 - m3) + result * m3

    return np.clip(blended, 0.0, 1.0).astype(np.float32)

def _apply_selective_adjustments(img01: np.ndarray,
                                  mask01: np.ndarray,
                                  cyan: float, magenta: float, yellow: float,
                                  r: float, g: float, b: float,
                                  lum: float, chroma: float, sat: float, con: float,
                                  intensity: float,
                                  use_chroma_mode: bool,
                                  band_lo: float = 0.0,
                                  band_hi: float = 1.0) -> np.ndarray:
    a = img01.astype(np.float32, copy=True)
    m = np.clip(mask01.astype(np.float32) * float(intensity), 0.0, 1.0)

    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)

    R = a[..., 0]; G = a[..., 1]; B = a[..., 2]

    R = np.clip(R + (-cyan)    * m, 0.0, 1.0)
    G = np.clip(G + (-magenta) * m, 0.0, 1.0)
    B = np.clip(B + (-yellow)  * m, 0.0, 1.0)

    R = np.clip(R + r * m, 0.0, 1.0)
    G = np.clip(G + g * m, 0.0, 1.0)
    B = np.clip(B + b * m, 0.0, 1.0)

    out = np.stack([R, G, B], axis=-1)

    if any(abs(x) > 1e-6 for x in (lum, chroma, sat, con)):
        if abs(lum) > 0:
            out = np.clip(out + lum * m[..., None], 0.0, 1.0)

        if abs(con) > 0:
            out = _band_contrast(out, m, con, band_lo, band_hi)

        if use_chroma_mode:
            if abs(chroma) > 0:
                out = _apply_chroma_boost(out, m, chroma)
        else:
            if abs(sat) > 0:
                try:
                    import cv2 as _cv2
                    u8 = _to_uint8_rgb(out)
                    hsv8 = _cv2.cvtColor(u8, _cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv8[..., 1] = np.clip(hsv8[..., 1] * (1.0 + sat * m), 0.0, 255.0)
                    hsv8 = hsv8.astype(np.uint8)
                    out = _cv2.cvtColor(hsv8, _cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
                except Exception:
                    pass

    return np.clip(out, 0.0, 1.0)


def _apply_chroma_boost(rgb01: np.ndarray, m01: np.ndarray, chroma: float) -> np.ndarray:
    rgb = _ensure_rgb01(rgb01).astype(np.float32)
    m   = np.clip(m01.astype(np.float32), 0.0, 1.0)[..., None]
    Y   = _luminance01(rgb)[..., None]
    d   = rgb - Y
    k   = 1.0 + float(chroma) * m
    return np.clip(Y + d * k, 0.0, 1.0)


# ---------------------------------------------------------------------
# Luminance Wheel widget
# ---------------------------------------------------------------------

class LuminanceWheel(QWidget):
    """
    A circular grayscale gradient ring with two draggable handles
    for selecting a luminance band (0.0 .. 1.0).

    Convention: 0 (black) is at the bottom (270°), 1 (white) is at the top (90°).
    The handle angle maps linearly: angle → lum = (angle - 270) % 360 / 360.
    """
    from PyQt6.QtCore import pyqtSignal
    rangeChanged = pyqtSignal(float, float)   # lo, hi  both in [0..1]

    def __init__(self, lo=0.0, hi=0.35, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self._lo = float(np.clip(lo, 0.0, 1.0))
        self._hi = float(np.clip(hi, 0.0, 1.0))
        self._dragging = None   # "lo" | "hi" | None
        self._ring_img = None
        self._picked = None    # luminance value or None (for marker)

    # --- public ---
    def setRange(self, lo: float, hi: float, notify=True):
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if abs(lo - self._lo) < 1e-6 and abs(hi - self._hi) < 1e-6:
            return
        self._lo, self._hi = lo, hi
        self.update()
        if notify:
            self.rangeChanged.emit(self._lo, self._hi)

    def range(self):
        return self._lo, self._hi

    def setPickedLum(self, lum: float | None):
        self._picked = None if lum is None else float(np.clip(lum, 0.0, 1.0))
        self.update()

    # --- angle <-> lum conversion (0 lum = 270°, 1 lum = 90°) ---
    @staticmethod
    def _lum_to_angle(lum: float) -> float:
        """lum 0..1  →  angle 0..360  (0→270°, 1→90°, increases CCW)"""
        return (270.0 - lum * 360.0) % 360.0

    @staticmethod
    def _angle_to_lum(angle: float) -> float:
        """angle 0..360  →  lum 0..1"""
        return float(np.clip((270.0 - angle) % 360.0 / 360.0, 0.0, 1.0))

    @staticmethod
    def _pos_to_angle(cx, cy, x, y) -> float:
        import math
        a = math.degrees(math.atan2(y - cy, x - cx))
        return (a + 360.0) % 360.0

    # --- ring image cache ---
    def _ensure_ring(self, side: int):
        if (self._ring_img is not None
                and self._ring_img.width() == side
                and self._ring_img.height() == side):
            return
        side    = int(side)
        cx = cy = side // 2
        r_outer = int(side * 0.48)
        r_inner = int(side * 0.33)   # ring thickness = 15% of side

        img = np.zeros((side, side, 3), np.uint8)

        # Vectorised — no Python loops, correct radius math
        ys, xs = np.mgrid[0:side, 0:side]
        dx = (xs - cx).astype(np.float32)
        dy = (ys - cy).astype(np.float32)
        d  = np.sqrt(dx * dx + dy * dy)

        in_ring = (d >= r_inner) & (d <= r_outer)

        # angle → luminance:  270° = black (lum 0), 90° = white (lum 1)
        ang = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        lum = np.clip((270.0 - ang) % 360.0 / 360.0, 0.0, 1.0)
        v   = (lum * 255.0 + 0.5).astype(np.uint8)

        img[in_ring, 0] = v[in_ring]
        img[in_ring, 1] = v[in_ring]
        img[in_ring, 2] = v[in_ring]

        h, w, _ = img.shape
        self._ring_img = QImage(img.data, w, h, img.strides[0],
                                QImage.Format.Format_RGB888).copy()

    # --- paint ---
    def paintEvent(self, ev):
        import math
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height())
        self._ensure_ring(side)
        x0 = (self.width()  - side) // 2
        y0 = (self.height() - side) // 2
        p.drawImage(x0, y0, self._ring_img)

        cx = x0 + side // 2
        cy = y0 + side // 2
        r  = int(side * 0.48)

        def pt(lum_val):
            ang = math.radians(self._lum_to_angle(lum_val))
            return int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))

        # --- draw arc between lo and hi (always short arc) ---
        lo_ang = self._lum_to_angle(self._lo)
        hi_ang = self._lum_to_angle(self._hi)

        # Walk from lo_ang to hi_ang in the direction that matches increasing lum
        # i.e. decreasing angle (CCW on standard math axes)
        # arc length in "angle decreasing" sense
        arc_len = (lo_ang - hi_ang) % 360.0
        steps = 60
        if arc_len > 0.1:
            p.setPen(QPen(QColor(255, 200, 0, 200), 4))
            px, py = pt(self._lo)
            for k in range(1, steps + 1):
                lv = self._lo + (self._hi - self._lo) * k / steps
                qx, qy = pt(float(np.clip(lv, 0.0, 1.0)))
                p.drawLine(px, py, qx, qy)
                px, py = qx, qy

        # handles
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        for lv in (self._lo, self._hi):
            xh, yh = pt(lv)
            p.drawEllipse(xh - 5, yh - 5, 10, 10)

        # picked-lum marker
        if self._picked is not None:
            ang = math.radians(self._lum_to_angle(self._picked))
            px2 = int(cx + r * math.cos(ang))
            py2 = int(cy + r * math.sin(ang))
            p.setBrush(QBrush(QColor(255, 100, 0)))
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawEllipse(px2 - 6, py2 - 6, 12, 12)

        # center label
        p.setPen(QPen(QColor(180, 180, 180)))
        p.drawText(x0, y0, side, side, Qt.AlignmentFlag.AlignCenter, "L")

    # --- mouse ---
    def mousePressEvent(self, ev):
        x, y = ev.position().x(), ev.position().y()
        side = min(self.width(), self.height())
        x0 = (self.width()  - side) // 2
        y0 = (self.height() - side) // 2
        cx = x0 + side // 2; cy = y0 + side // 2
        ang = self._pos_to_angle(cx, cy, x, y)
        lum = self._angle_to_lum(ang)

        def ang_dist(a1, a2):
            return abs((a1 - a2 + 180) % 360 - 180)

        d_lo = ang_dist(ang, self._lum_to_angle(self._lo))
        d_hi = ang_dist(ang, self._lum_to_angle(self._hi))

        if d_lo <= d_hi:
            self._dragging = "lo"
            self._lo = float(np.clip(lum, 0.0, 1.0))
        else:
            self._dragging = "hi"
            self._hi = float(np.clip(lum, 0.0, 1.0))

        self.update()
        self.rangeChanged.emit(self._lo, self._hi)

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return
        x, y = ev.position().x(), ev.position().y()
        side = min(self.width(), self.height())
        x0 = (self.width()  - side) // 2
        y0 = (self.height() - side) // 2
        cx = x0 + side // 2; cy = y0 + side // 2
        ang = self._pos_to_angle(cx, cy, x, y)
        lum = float(np.clip(self._angle_to_lum(ang), 0.0, 1.0))

        if self._dragging == "lo":
            self._lo = lum
        else:
            self._hi = lum
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)

    def mouseReleaseEvent(self, ev):
        self._dragging = None


# ---------------------------------------------------------------------
# Main Dialog
# ---------------------------------------------------------------------

class SelectiveLuminanceCorrection(QDialog):
    """
    Selective Luminance Correction v1.0
    Target pixels by luminance band (instead of hue) and apply
    CMY / RGB / Luminance / Chroma / Saturation / Contrast adjustments
    only to that band.  Ideal for taming galaxy cores, lifting faint nebulosity,
    desaturating noise in dark regions, etc.
    """
    _CONTRAST_MONOTONIC_LIMIT = -2.0 / 3.0   # ≈ −0.6667

    def __init__(self, doc_manager=None, document=None, parent=None,
                 window_icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Selective Luminance Correction"))
        if window_icon:
            self.setWindowIcon(window_icon)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.docman   = doc_manager
        self.document = document
        if self.document is None or getattr(self.document, "image", None) is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            self.close(); return

        self.img = np.clip(self.document.image.astype(np.float32), 0.0, 1.0)

        self._imported_mask_full = None
        self._imported_mask_name = None
        self._use_imported_mask  = False
        self._mask_delay_ms = 200
        self._adj_delay_ms  = 200
        self._syncing_lum   = False
        self._region_mask_full = None
        self._lasso_points     = []
        self._lasso_drawing    = False
        # ── Must be set BEFORE _build_ui() because _build_ui ends with
        #    _update_preview_pixmap(), which reads _zoom, _panning, etc. ──
        self._zoom              = 1.0
        self._panning           = False
        self._pan_start_pos_vp  = None
        self._pan_start_scroll  = (0, 0)
        self._pan_deadzone      = 1

        self._build_ui()

        # Timers live after _build_ui so the slots they connect to exist,
        # but they're created here rather than inside _build_ui to keep
        # _build_ui purely about layout.
        self._mask_timer = QTimer(self)
        self._mask_timer.setSingleShot(True)
        self._mask_timer.timeout.connect(self._recompute_mask_and_preview)

        self._adj_timer = QTimer(self)
        self._adj_timer.setSingleShot(True)
        self._adj_timer.timeout.connect(self._update_preview_pixmap)

        self.dd_preset.setCurrentText("Shadows")
        self._recompute_mask_and_preview()

    # =================================================================
    # UI BUILD
    # =================================================================
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        root.addWidget(self.splitter)

        # ------------------------------------------------------------------
        # LEFT PANE
        # ------------------------------------------------------------------
        left_widget = QWidget()
        left_widget.setMinimumWidth(360)
        left_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.splitter.addWidget(left_widget)

        left_outer = QVBoxLayout(left_widget)
        left_outer.setContentsMargins(0, 0, 0, 0)
        left_outer.setSpacing(8)

        # Target label
        try:
            disp = getattr(self.document, "display_name", lambda: "Image")()
        except Exception:
            disp = "Image"
        self.lbl_target = QLabel(f"Target View:  <b>{disp}</b>")
        left_outer.addWidget(self.lbl_target)

        # Small preview toggle
        self.cb_small_preview = QCheckBox("Small-sized Preview (fast)")
        self.cb_small_preview.setChecked(True)
        self.cb_small_preview.toggled.connect(self._recompute_mask_and_preview)
        left_outer.addWidget(self.cb_small_preview)

        # ---------- scrollable controls container ----------
        controls_container = QWidget()
        left = QVBoxLayout(controls_container)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(8)

        # ===== MASK group =====
        gb_mask = QGroupBox(self.tr("Luminance Mask"))
        gl = QGridLayout(gb_mask)
        gl.setContentsMargins(8, 8, 8, 8)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(8)

        # Row 0: Preset
        gl.addWidget(QLabel("Preset:"), 0, 0)
        self.dd_preset = QComboBox()
        self.dd_preset.addItems(["Custom"] + list(_LUM_PRESETS.keys()))
        self.dd_preset.currentTextChanged.connect(self._on_preset_change)
        gl.addWidget(self.dd_preset, 0, 1, 1, 4)

        # Luminance wheel (rows 1–6, cols 0–1)
        self.lum_wheel = LuminanceWheel(lo=0.0, hi=0.35)
        self.lum_wheel.setMinimumSize(130, 130)
        self.lum_wheel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        gl.addWidget(self.lum_wheel, 1, 0, 6, 2)

        # Wheel → sliders/spins
        def _wheel_to_controls(lo: float, hi: float):
            if not self._syncing_lum and self.dd_preset.currentText() != "Custom":
                self.dd_preset.blockSignals(True)
                self.dd_preset.setCurrentText("Custom")
                self.dd_preset.blockSignals(False)
            self._syncing_lum = True
            try:
                for w, val in (
                    (self.ds_lo, lo), (self.sl_lo, int(round(lo * 255))),
                    (self.ds_hi, hi), (self.sl_hi, int(round(hi * 255))),
                ):
                    w.blockSignals(True)
                    w.setValue(val)
                    w.blockSignals(False)
            finally:
                self._syncing_lum = False
            self._schedule_mask()

        self.lum_wheel.rangeChanged.connect(_wheel_to_controls)

        # Helper: 0-255 integer slider + 0.0-1.0 double spin (linked)
        def _lum_pair(grid, label, row):
            grid.addWidget(QLabel(label), row, 2)
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(0, 255); sld.setSingleStep(1); sld.setPageStep(10)
            spn = QDoubleSpinBox()
            spn.setRange(0.0, 1.0); spn.setSingleStep(0.01); spn.setDecimals(3)

            def _s2d(v):
                spn.blockSignals(True)
                spn.setValue(v / 255.0)
                spn.blockSignals(False)

            def _d2s(v):
                sld.blockSignals(True)
                sld.setValue(int(round(v * 255)))
                sld.blockSignals(False)

            sld.valueChanged.connect(_s2d)
            spn.valueChanged.connect(_d2s)
            grid.addWidget(sld, row, 3, 1, 3)
            grid.addWidget(spn, row, 6, 1, 1)
            return sld, spn

        # Rows 1–2: Lo / Hi
        self.sl_lo, self.ds_lo = _lum_pair(gl, "Lum. low (0–255):",  1)
        self.sl_hi, self.ds_hi = _lum_pair(gl, "Lum. high (0–255):", 2)
        self.sl_lo.setValue(0);   self.ds_lo.setValue(0.0)
        self.sl_hi.setValue(89);  self.ds_hi.setValue(89/255)

        # Row 3: smoothness + invert
        gl.addWidget(QLabel("Smoothness:"), 3, 2)
        self.ds_smooth = QDoubleSpinBox()
        self.ds_smooth.setRange(0.0, 0.5); self.ds_smooth.setSingleStep(0.01)
        self.ds_smooth.setDecimals(3); self.ds_smooth.setValue(0.05)
        self.ds_smooth.valueChanged.connect(self._schedule_mask)
        gl.addWidget(self.ds_smooth, 3, 3)

        self.cb_invert = QCheckBox("Invert range")
        self.cb_invert.setChecked(False)
        self.cb_invert.toggled.connect(self._schedule_mask)
        gl.addWidget(self.cb_invert, 3, 4, 1, 3)

        # Row 4: intensity + blur
        gl.addWidget(QLabel("Intensity:"), 4, 2)
        self.ds_int = QDoubleSpinBox()
        self.ds_int.setRange(0.0, 2.0); self.ds_int.setSingleStep(0.05)
        self.ds_int.setValue(1.0)
        self.ds_int.valueChanged.connect(self._schedule_adjustments)
        gl.addWidget(self.ds_int, 4, 3)

        gl.addWidget(QLabel("Edge blur (px):"), 4, 4)
        self.sb_blur = QSpinBox()
        self.sb_blur.setRange(0, 150); self.sb_blur.setValue(5)
        self.sb_blur.valueChanged.connect(self._schedule_mask)
        gl.addWidget(self.sb_blur, 4, 5)

        # Row 5: show mask + use imported
        self.cb_show_mask = QCheckBox("Show mask overlay")
        self.cb_show_mask.setChecked(False)
        self.cb_show_mask.toggled.connect(self._update_preview_pixmap)
        gl.addWidget(self.cb_show_mask, 5, 2, 1, 2)

        self.cb_use_imported = QCheckBox("Use imported mask")
        self.cb_use_imported.setChecked(False)
        self.cb_use_imported.toggled.connect(self._on_use_imported_mask_toggled)
        gl.addWidget(self.cb_use_imported, 5, 4, 1, 2)

        # Row 6: import mask
        self.btn_import_mask = QPushButton("Pick mask from view…")
        self.btn_import_mask.clicked.connect(self._import_mask_from_view)
        gl.addWidget(self.btn_import_mask, 6, 2, 1, 2)

        self.lbl_imported_mask = QLabel("No imported mask")
        gl.addWidget(self.lbl_imported_mask, 6, 4, 1, 3)

        # column stretch
        gl.setColumnStretch(0, 0); gl.setColumnStretch(1, 0)
        for c in (2, 3, 4, 5, 6, 7):
            gl.setColumnStretch(c, 1)

        left.addWidget(gb_mask)

        # ===== CMY =====
        gb_cmy = QGroupBox(self.tr("Complementary Colors (CMY)"))
        glc = QGridLayout(gb_cmy)
        self.sl_c, self.ds_c = self._slider_pair(glc, "Cyan:",    0)
        self.sl_m, self.ds_m = self._slider_pair(glc, "Magenta:", 1)
        self.sl_y, self.ds_y = self._slider_pair(glc, "Yellow:",  2)
        left.addWidget(gb_cmy)

        # ===== RGB =====
        gb_rgb = QGroupBox(self.tr("RGB Colors"))
        glr = QGridLayout(gb_rgb)
        self.sl_r, self.ds_r = self._slider_pair(glr, "Red:",   0)
        self.sl_g, self.ds_g = self._slider_pair(glr, "Green:", 1)
        self.sl_b, self.ds_b = self._slider_pair(glr, "Blue:",  2)
        left.addWidget(gb_rgb)

        # ===== L / Chroma / Sat / Contrast =====
        gb_lsc = QGroupBox(self.tr("Luminance, Chroma/Saturation, Contrast"))
        gll = QGridLayout(gb_lsc)
        self.sl_l,      self.ds_l      = self._slider_pair(gll, "Luminance:",             0)
        self.sl_chroma, self.ds_chroma = self._slider_pair(gll, "Chroma (L-preserving):", 1)
        self.sl_s,      self.ds_s      = self._slider_pair(gll, "Saturation (HSV S):",    2)
        self.sl_c2,     self.ds_c2     = self._slider_pair(gll, "Contrast:",               3)
        gll.addWidget(QLabel("Color boost mode:"), 4, 0)
        self.dd_color_mode = QComboBox()
        self.dd_color_mode.addItems(["Chroma (L-preserving)", "Saturation (HSV S)"])
        self.dd_color_mode.setCurrentIndex(0)
        self.dd_color_mode.currentIndexChanged.connect(self._update_color_mode_enabled)
        gll.addWidget(self.dd_color_mode, 4, 1, 1, 2)
        left.addWidget(gb_lsc)
 
        self.lbl_contrast_warning = QLabel(
            "⚠️  Contrast below −0.67: curve is no longer monotonically increasing"
        )
        self.lbl_contrast_warning.setStyleSheet(
            "color: #e07020; font-size: 10px; padding: 2px 4px;"
        )
        self.lbl_contrast_warning.setWordWrap(True)
        self.lbl_contrast_warning.setVisible(False)
        left.addWidget(self.lbl_contrast_warning)
 
        # wrap in scroll area
        left_scroll = QScrollArea()
        left_scroll.setWidget(controls_container)
        left_scroll.setWidgetResizable(False)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        left_outer.addWidget(left_scroll, 1)

        # Live toggle
        self.cb_live = QCheckBox("Preview changed image")
        self.cb_live.setChecked(True)
        self.cb_live.toggled.connect(self._update_preview_pixmap)
        left_outer.addWidget(self.cb_live)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_apply        = QPushButton("Apply")
        self.btn_push         = QPushButton("Apply as New Document")
        self.btn_export_mask  = QPushButton("Export Mask")
        self.btn_reset        = QPushButton("↺ Reset")
        self.btn_apply.clicked.connect(self._apply_to_document)
        self.btn_push.clicked.connect(self._apply_as_new_doc)
        self.btn_export_mask.clicked.connect(self._export_mask_doc)
        self.btn_reset.clicked.connect(self._reset_controls)
        for b in (self.btn_apply, self.btn_push, self.btn_export_mask, self.btn_reset):
            btn_row.addWidget(b)

        self.btn_clear_region = QPushButton("✕ Clear Region Mask")
        self.btn_clear_region.setToolTip("Remove the freehand region constraint")
        self.btn_clear_region.clicked.connect(self._clear_region_mask)
        self.btn_clear_region.setEnabled(False)
        btn_row.addWidget(self.btn_clear_region)

        left_outer.addLayout(btn_row)

        # Footer
        footer_label = QLabel("""
            Written by Franklin Marek<br>
            <a href='http://www.setiastro.com'>www.setiastro.com</a>
        """)
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setStyleSheet("font-size: 10px;")
        left_outer.addWidget(footer_label)

        # ------------------------------------------------------------------
        # RIGHT PANE
        # ------------------------------------------------------------------
        right_widget = QWidget()
        right_widget.setMinimumWidth(420)
        right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.splitter.addWidget(right_widget)

        right = QVBoxLayout(right_widget)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(8)

        # Zoom toolbar
        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in",  "Zoom In")
        self.btn_zoom_1   = themed_toolbtn("zoom-original", "1:1")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit")
        for b in (self.btn_zoom_out, self.btn_zoom_in, self.btn_zoom_1, self.btn_fit):
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        right.addLayout(zoom_row)

        self.lbl_help = QLabel(
            "🖱️ <b>Click</b>: show luminance &nbsp;•&nbsp; "
            "<b>Shift+Click</b>: select that band &nbsp;•&nbsp; "
            "<b>Ctrl+Drag</b>: pan &nbsp;•&nbsp; "
            "<b>Alt+Drag</b>: draw region mask &nbsp;•&nbsp; "
            "<b>Wheel</b>: zoom"
        )
        self.lbl_help.setWordWrap(True)
        self.lbl_help.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_help.setStyleSheet("color: #888; font-size: 11px;")
        right.addWidget(self.lbl_help)

        # Preview scroller
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

        # Luminance readout
        self.lbl_lum_readout = QLabel("Picked luminance: —")
        right.addWidget(self.lbl_lum_readout)
        self.lbl_region_readout = QLabel("Region mask: none  (Alt+Drag to draw)")
        self.lbl_region_readout.setStyleSheet("color: #888; font-size: 11px;")
        right.addWidget(self.lbl_region_readout)
        # Splitter stretch
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([420, 900])

        self.setSizeGripEnabled(True)
        try:
            g = QGuiApplication.primaryScreen().availableGeometry()
            max_h = int(g.height() * 0.9)
            self.resize(1080, min(680, max_h))
            self.setMaximumHeight(max_h)
        except Exception:
            self.resize(1080, 680)

        # --- wiring ---
        self._update_color_mode_enabled()

        for w in (self.ds_c, self.ds_m, self.ds_y,
                  self.ds_r, self.ds_g, self.ds_b,
                  self.ds_l, self.ds_s, self.ds_c2,
                  self.ds_chroma, self.ds_int):
            w.valueChanged.connect(self._schedule_adjustments)

        def _controls_to_wheel(_=None):
            if self._syncing_lum:
                return
            if self.dd_preset.currentText() != "Custom":
                self.dd_preset.blockSignals(True)
                self.dd_preset.setCurrentText("Custom")
                self.dd_preset.blockSignals(False)
            lo = self.ds_lo.value()
            hi = self.ds_hi.value()
            self.lum_wheel.setRange(lo, hi, notify=False)
            self._schedule_mask()

        self.ds_lo.valueChanged.connect(_controls_to_wheel)
        self.ds_hi.valueChanged.connect(_controls_to_wheel)
        self.sl_lo.valueChanged.connect(_controls_to_wheel)
        self.sl_hi.valueChanged.connect(_controls_to_wheel)
        self.ds_c2.valueChanged.connect(self._update_contrast_warning)
        self.sl_c2.valueChanged.connect(
            lambda v: self._update_contrast_warning(v / 100.0)
        )
        # Zoom buttons
        self.btn_zoom_in.clicked.connect(lambda: self._apply_zoom(self._zoom * 1.25, None))
        self.btn_zoom_out.clicked.connect(lambda: self._apply_zoom(self._zoom / 1.25, None))
        self.btn_zoom_1.clicked.connect(lambda: self._apply_zoom(1.0, None))
        self.btn_fit.clicked.connect(self._fit_to_preview)

        try:
            self.splitter.splitterMoved.connect(lambda *_: self._save_window_state())
        except Exception:
            pass

        self._update_preview_pixmap()

    # =================================================================
    # Helpers
    # =================================================================

    def _clear_region_mask(self):
        self._region_mask_full = None
        self._lasso_points = []
        self.lbl_region_readout.setText("Region mask: none  (Alt+Drag to draw)")
        self.btn_clear_region.setEnabled(False)
        self._recompute_mask_and_preview()

    def _lasso_to_region_mask(self, points_label: list, base_shape: tuple):
        if len(points_label) < 3:
            return None
        bh, bw = base_shape[:2]
        z = max(self._zoom, 1e-6)
        pts_img = [(x / z, y / z) for x, y in points_label]
        if cv2 is not None:
            pts_arr = np.array(pts_img, dtype=np.int32)
            mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_arr], 255)
            return mask.astype(np.float32) / 255.0
        else:
            from matplotlib.path import Path
            path = Path(pts_img)
            yy, xx = np.mgrid[0:bh, 0:bw]
            coords = np.column_stack([xx.ravel(), yy.ravel()])
            inside = path.contains_points(coords)
            return inside.reshape(bh, bw).astype(np.float32)

    def _draw_lasso_overlay(self):
        """Redraw the preview with the in-progress lasso on top."""
        base = getattr(self, "_last_base", None)
        if base is None:
            return

        mask = getattr(self, "_mask", np.zeros(base.shape[:2], np.float32))
        if self.cb_live.isChecked():
            out = _apply_selective_adjustments(
                base, mask,
                cyan=float(self.ds_c.value()),
                magenta=float(self.ds_m.value()),
                yellow=float(self.ds_y.value()),
                r=float(self.ds_r.value()),
                g=float(self.ds_g.value()),
                b=float(self.ds_b.value()),
                lum=float(self.ds_l.value()),
                chroma=float(self.ds_chroma.value()),
                sat=float(self.ds_s.value()),
                con=float(self.ds_c2.value()),
                intensity=float(self.ds_int.value()),
                use_chroma_mode=(self.dd_color_mode.currentIndex() == 0),
                band_lo=float(self.ds_lo.value()),
                band_hi=float(self.ds_hi.value()),
            )
            out = _ensure_rgb01(out)
        else:
            out = _ensure_rgb01(base)

        if self.cb_show_mask.isChecked():
            out = self._overlay_mask(out, mask * float(self.ds_int.value()))

        pm = _to_pixmap(out)
        h, w = out.shape[:2]
        zw = max(1, int(round(w * self._zoom)))
        zh = max(1, int(round(h * self._zoom)))
        pm_scaled = pm.scaled(zw, zh, Qt.AspectRatioMode.IgnoreAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)

        # Paint lasso on top
        from PyQt6.QtGui import QPainterPath
        overlay = pm_scaled.copy()
        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if len(self._lasso_points) > 2:
            path = QPainterPath()
            path.moveTo(*self._lasso_points[0])
            for px, py in self._lasso_points[1:]:
                path.lineTo(px, py)
            path.closeSubpath()
            painter.fillPath(path, QColor(255, 165, 0, 35))

        pen = QPen(QColor(255, 165, 0), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for i in range(1, len(self._lasso_points)):
            x0, y0 = self._lasso_points[i-1]
            x1, y1 = self._lasso_points[i]
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
        if len(self._lasso_points) > 2:
            x0, y0 = self._lasso_points[-1]
            x1, y1 = self._lasso_points[0]
            painter.setPen(QPen(QColor(255, 165, 0, 120), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))

        painter.end()
        self.lbl_preview.setPixmap(overlay)
        self.lbl_preview.resize(zw, zh)

    def _draw_region_outline_on_pixmap(self, pm: QPixmap) -> QPixmap:
        """Draw the persistent dashed orange outline of the active region mask."""
        if self._region_mask_full is None or cv2 is None:
            return pm
        pw, ph = pm.width(), pm.height()
        try:
            rm_preview = cv2.resize(
                self._region_mask_full, (pw, ph),
                interpolation=cv2.INTER_NEAREST)
            rm_u8 = (rm_preview * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                rm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return pm

        from PyQt6.QtGui import QPainterPath
        result = pm.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Outer glow
        pen_glow = QPen(QColor(255, 165, 0, 80), 4)
        pen_glow.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen_glow)
        for contour in contours:
            pts = contour.reshape(-1, 2)
            for i in range(len(pts)):
                x0, y0 = int(pts[i][0]), int(pts[i][1])
                x1, y1 = int(pts[(i+1) % len(pts)][0]), int(pts[(i+1) % len(pts)][1])
                painter.drawLine(x0, y0, x1, y1)

        # Dashed outline
        pen_dash = QPen(QColor(255, 165, 0), 2)
        pen_dash.setStyle(Qt.PenStyle.DashLine)
        pen_dash.setDashPattern([6, 4])
        painter.setPen(pen_dash)
        for contour in contours:
            pts = contour.reshape(-1, 2)
            for i in range(len(pts)):
                x0, y0 = int(pts[i][0]), int(pts[i][1])
                x1, y1 = int(pts[(i+1) % len(pts)][0]), int(pts[(i+1) % len(pts)][1])
                painter.drawLine(x0, y0, x1, y1)

        painter.end()
        return result

    def _update_contrast_warning(self, value: float):
        """Show warning and recolour slider when contrast is non-monotonic."""
        non_mono = value < self._CONTRAST_MONOTONIC_LIMIT
        self.lbl_contrast_warning.setVisible(non_mono)
 
        if non_mono:
            self.sl_c2.setStyleSheet("""
                QSlider::groove:horizontal {
                    height: 4px;
                    background: #5a1a1a;
                    border-radius: 2px;
                }
                QSlider::sub-page:horizontal {
                    background: #c0392b;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background: #e74c3c;
                    border: 1px solid #922b21;
                    width: 12px; height: 12px;
                    margin: -4px 0;
                    border-radius: 6px;
                }
            """)
        else:
            self.sl_c2.setStyleSheet("")   # restore default theme styling

    def _slider_pair(self, grid, name, row, minv=-1.0, maxv=1.0, step=0.05):
        import math

        def _to_slider(v):
            s = abs(v) * 100.0
            s = math.floor(s + 0.5)
            return int(-s if v < 0 else s)

        grid.addWidget(QLabel(name), row, 0)
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(int(minv * 100), int(maxv * 100))
        sld.setSingleStep(int(step * 100))
        sld.setPageStep(int(5 * step * 100))
        sld.setValue(0)

        box = QDoubleSpinBox()
        box.setRange(minv, maxv); box.setSingleStep(step)
        box.setDecimals(2); box.setValue(0.0)
        box.setKeyboardTracking(False)

        def _s2b(v_int):
            box.blockSignals(True); box.setValue(v_int / 100.0); box.blockSignals(False)

        def _b2s(v_float):
            sld.blockSignals(True); sld.setValue(_to_slider(v_float)); sld.blockSignals(False)

        sld.valueChanged.connect(_s2b)
        box.valueChanged.connect(_b2s)
        sld.valueChanged.connect(self._schedule_adjustments)
        box.valueChanged.connect(self._schedule_adjustments)
        sld.sliderReleased.connect(self._update_preview_pixmap)
        box.editingFinished.connect(self._update_preview_pixmap)

        grid.addWidget(sld, row, 1)
        grid.addWidget(box, row, 2)
        return sld, box

    def _set_pair(self, sld, box, value):
        sld.blockSignals(True); box.blockSignals(True)
        sld.setValue(int(round(value * 100)))
        box.setValue(float(value))
        sld.blockSignals(False); box.blockSignals(False)

    def _update_color_mode_enabled(self):
        use_chroma = (self.dd_color_mode.currentIndex() == 0)
        self.ds_chroma.setEnabled(use_chroma); self.sl_chroma.setEnabled(use_chroma)
        self.ds_s.setEnabled(not use_chroma);  self.sl_s.setEnabled(not use_chroma)
        self._schedule_adjustments()

    def _schedule_mask(self, *_, delay_ms=None):
        if not hasattr(self, "_mask_timer"): return
        ms = max(1, int(delay_ms or self._mask_delay_ms))
        self._mask_timer.stop(); self._mask_timer.start(ms)

    def _schedule_adjustments(self, *_, delay_ms=None):
        if not hasattr(self, "_adj_timer"): return
        ms = max(1, int(delay_ms or self._adj_delay_ms))
        self._adj_timer.stop(); self._adj_timer.start(ms)

    # =================================================================
    # Zoom / Pan
    # =================================================================
    def _current_scroll(self):
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        return hbar.value(), vbar.value(), hbar.maximum(), vbar.maximum()

    def _set_scroll(self, x, y):
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(int(max(0, min(x, hbar.maximum()))))
        vbar.setValue(int(max(0, min(y, vbar.maximum()))))

    def _apply_zoom(self, new_zoom, anchor_label_pos=None):
        old_zoom = getattr(self, "_zoom", 1.0)
        new_zoom = max(0.05, min(16.0, float(new_zoom)))
        if abs(new_zoom - old_zoom) < 1e-6: return

        if anchor_label_pos is None:
            sx, sy, _, _ = self._current_scroll()
            vp = self.scroll.viewport().rect()
            cx = (sx + vp.width()  / 2.0) / max(old_zoom, 1e-9)
            cy = (sy + vp.height() / 2.0) / max(old_zoom, 1e-9)
        else:
            cx = float(anchor_label_pos.x())
            cy = float(anchor_label_pos.y())

        sx, sy, _, _ = self._current_scroll()
        vp = self.scroll.viewport().rect()
        pvx = cx * old_zoom - sx
        pvy = cy * old_zoom - sy

        self._zoom = new_zoom
        self._update_preview_pixmap()

        nx = cx * new_zoom - pvx
        ny = cy * new_zoom - pvy
        self._set_scroll(nx, ny)

    def _fit_to_preview(self):
        base = getattr(self, "_last_base", None)
        if base is None: return
        h, w = base.shape[:2]
        vp = self.scroll.viewport().size()
        if w <= 0 or h <= 0: return
        k = min(vp.width() / w, vp.height() / h)
        self._apply_zoom(k)

    # =================================================================
    # Event filter (zoom + pan + luminance pick)
    # =================================================================
    def eventFilter(self, obj, ev):
        from PyQt6.QtCore import QEvent

        def _vp_pos(o, e):
            if o is self.scroll.viewport():
                return e.position()
            return self.lbl_preview.mapTo(self.scroll.viewport(), e.position().toPoint())

        # Wheel zoom
        if obj in (self.scroll.viewport(), self.lbl_preview) and ev.type() == QEvent.Type.Wheel:
            anchor = (ev.position() if obj is self.lbl_preview
                      else self.lbl_preview.mapFrom(self.scroll.viewport(), ev.position().toPoint()))

            dy = ev.pixelDelta().y()
            if dy != 0:
                abs_dy = abs(dy)
                ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                base_f = (1.040 if ctrl else 1.030) if abs_dy > 10 else (1.025 if ctrl else 1.020) if abs_dy > 3 else (1.012 if ctrl else 1.010)
                factor = base_f if dy > 0 else 1.0 / base_f
            else:
                dy = ev.angleDelta().y()
                if dy == 0: ev.accept(); return True
                ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                step = 1.25 if ctrl else 1.15
                factor = step if dy > 0 else 1.0 / step

            self._apply_zoom(self._zoom * factor, anchor_label_pos=anchor)
            ev.accept(); return True
        # --- LASSO REGION MASK (Alt + LMB) ---
        if obj in (self.scroll.viewport(), self.lbl_preview):
            if ev.type() == QEvent.Type.MouseButtonPress:
                if (ev.button() == Qt.MouseButton.LeftButton and
                        ev.modifiers() & Qt.KeyboardModifier.AltModifier and
                        not ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    lpos = (ev.position() if obj is self.lbl_preview
                            else self.lbl_preview.mapFrom(
                                self.scroll.viewport(), ev.position().toPoint()))
                    self._lasso_drawing = True
                    self._lasso_points = [(float(lpos.x()), float(lpos.y()))]
                    self.scroll.viewport().setCursor(Qt.CursorShape.CrossCursor)
                    return True

            elif ev.type() == QEvent.Type.MouseMove and self._lasso_drawing:
                lpos = (ev.position() if obj is self.lbl_preview
                        else self.lbl_preview.mapFrom(
                            self.scroll.viewport(), ev.position().toPoint()))
                self._lasso_points.append((float(lpos.x()), float(lpos.y())))
                if len(self._lasso_points) % 8 == 0:
                    self._draw_lasso_overlay()
                return True

            elif ev.type() == QEvent.Type.MouseButtonRelease and self._lasso_drawing:
                self._lasso_drawing = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)

                if len(self._lasso_points) >= 3 and hasattr(self, "_last_base"):
                    region = self._lasso_to_region_mask(
                        self._lasso_points, self._last_base.shape)
                    if region is not None:
                        fh, fw = self.img.shape[:2]
                        bh, bw = self._last_base.shape[:2]
                        if (bh, bw) != (fh, fw) and cv2 is not None:
                            region_full = cv2.resize(
                                region, (fw, fh), interpolation=cv2.INTER_LINEAR)
                        else:
                            region_full = region
                        self._region_mask_full = np.clip(
                            region_full.astype(np.float32), 0.0, 1.0)
                        n_pts = len(self._lasso_points)
                        self.lbl_region_readout.setText(
                            f"Region mask: active ({n_pts} points) "
                            f"— click '✕ Clear Region Mask' to remove")
                        self.btn_clear_region.setEnabled(True)

                self._lasso_points = []
                self._recompute_mask_and_preview()
                return True
        # Pan (Ctrl + LMB)
        if obj in (self.scroll.viewport(), self.lbl_preview):
            if ev.type() == QEvent.Type.MouseButtonPress:
                if (ev.button() == Qt.MouseButton.LeftButton and
                        ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                    self._panning = True
                    self._pan_start_pos_vp = _vp_pos(obj, ev)
                    hbar = self.scroll.horizontalScrollBar()
                    vbar = self.scroll.verticalScrollBar()
                    self._pan_start_scroll = (hbar.value(), vbar.value())
                    self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

            elif ev.type() == QEvent.Type.MouseMove and self._panning:
                cur = _vp_pos(obj, ev)
                dx = cur.x() - self._pan_start_pos_vp.x()
                dy = cur.y() - self._pan_start_pos_vp.y()
                if abs(dx) > self._pan_deadzone or abs(dy) > self._pan_deadzone:
                    hbar = self.scroll.horizontalScrollBar()
                    vbar = self.scroll.verticalScrollBar()
                    hbar.setValue(int(self._pan_start_scroll[0] - dx))
                    vbar.setValue(int(self._pan_start_scroll[1] - dy))
                return True

            elif ev.type() in (QEvent.Type.MouseButtonRelease, QEvent.Type.Leave):
                if self._panning:
                    self._panning = False
                    self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                    return True

        # Luminance pick (plain / shift click on label)
        if obj is self.lbl_preview and ev.type() == QEvent.Type.MouseButtonPress:
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                return True   # let pan handle it
            pt = self._map_label_point_to_image_xy(ev.position())
            if pt is not None:
                x, y = pt
                lum = self._sample_lum_from_base(x, y)
                if lum is not None:
                    self.lum_wheel.setPickedLum(lum)
                    self.lbl_lum_readout.setText(
                        f"Picked luminance: {lum:.3f}  ({int(round(lum * 255))})"
                    )
                    if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        half = 0.05
                        lo = float(np.clip(lum - half, 0.0, 1.0))
                        hi = float(np.clip(lum + half, 0.0, 1.0))
                        self.lum_wheel.setRange(lo, hi)
            return True

        return super().eventFilter(obj, ev)

    def _map_label_point_to_image_xy(self, ev_pos):
        base = getattr(self, "_last_base", None)
        if base is None: return None
        bh, bw = base.shape[:2]
        x = int(round(ev_pos.x() / max(self._zoom, 1e-6)))
        y = int(round(ev_pos.y() / max(self._zoom, 1e-6)))
        if x < 0 or y < 0 or x >= bw or y >= bh: return None
        return (x, y)

    def _sample_lum_from_base(self, x, y):
        base = getattr(self, "_last_base", None)
        if base is None: return None
        h, w = base.shape[:2]
        if not (0 <= x < w and 0 <= y < h): return None
        pix = base[y:y+1, x:x+1]
        return float(_luminance01(pix)[0, 0])

    # =================================================================
    # Preset
    # =================================================================
    def _on_preset_change(self, txt):
        if txt == "Custom":
            return
        lo, hi = _LUM_PRESETS[txt]
        self._syncing_lum = True
        try:
            self.lum_wheel.setRange(lo, hi, notify=False)
            for w, val in (
                (self.sl_lo, int(round(lo * 255))), (self.ds_lo, lo),
                (self.sl_hi, int(round(hi * 255))), (self.ds_hi, hi),
            ):
                w.blockSignals(True); w.setValue(val); w.blockSignals(False)
            self.lum_wheel.update()
        finally:
            self._syncing_lum = False
        self._recompute_mask_and_preview()

    # =================================================================
    # Mask & Preview
    # =================================================================
    def _downsample(self, img, max_dim=1200):
        h, w = img.shape[:2]
        s = max(h, w)
        if s <= max_dim: return img
        k = max_dim / float(s)
        if cv2 is not None:
            return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        return img[::int(1/k), ::int(1/k)]

    def _recompute_mask_and_preview(self):
        if self.img is None: return

        base = self._downsample(self.img, 1200) if self.cb_small_preview.isChecked() else self.img
        self._last_base = base

        if self._use_imported_mask and self._imported_mask_full is not None:
            imp = self._imported_mask_full
            bh, bw = base.shape[:2]
            mh, mw = imp.shape[:2]
            if (mh, mw) != (bh, bw):
                mask = cv2.resize(imp, (bw, bh), interpolation=cv2.INTER_LINEAR) if cv2 is not None else imp
            else:
                mask = imp
        else:
            lo = self.ds_lo.value()
            hi = self.ds_hi.value()
            if lo > hi:
                lo, hi = hi, lo
            mask = _build_lum_mask(
                base,
                lo=lo, hi=hi,
                smooth=float(self.ds_smooth.value()),
                invert=self.cb_invert.isChecked(),
                blur_px=int(self.sb_blur.value()),
            )

        self._mask = np.clip(mask, 0.0, 1.0)

        # AND with region mask if one is active
        if self._region_mask_full is not None:
            bh, bw = base.shape[:2]
            rm = self._region_mask_full
            mh, mw = rm.shape[:2]
            if (mh, mw) != (bh, bw):
                rm = cv2.resize(rm, (bw, bh), interpolation=cv2.INTER_LINEAR) if cv2 is not None else rm
            self._mask = self._mask * np.clip(rm.astype(np.float32), 0.0, 1.0)
        self._mask = np.clip(self._mask, 0.0, 1.0)

        self._update_preview_pixmap()

    def _update_preview_pixmap(self):
        if not hasattr(self, "_last_base"):
            self._recompute_mask_and_preview(); return

        base = self._last_base
        mask = getattr(self, "_mask", np.zeros(base.shape[:2], np.float32))

        if self.cb_live.isChecked():
            out = _apply_selective_adjustments(
                base, mask,
                cyan=float(self.ds_c.value()),
                magenta=float(self.ds_m.value()),
                yellow=float(self.ds_y.value()),
                r=float(self.ds_r.value()),
                g=float(self.ds_g.value()),
                b=float(self.ds_b.value()),
                lum=float(self.ds_l.value()),
                chroma=float(self.ds_chroma.value()),
                sat=float(self.ds_s.value()),
                con=float(self.ds_c2.value()),
                intensity=float(self.ds_int.value()),
                use_chroma_mode=(self.dd_color_mode.currentIndex() == 0),
                band_lo=float(self.ds_lo.value()), 
                band_hi=float(self.ds_hi.value()),           
            )
            out = _ensure_rgb01(out)
        else:
            out = _ensure_rgb01(base)

        if self.cb_show_mask.isChecked():
            show = self._overlay_mask(out, mask * float(self.ds_int.value()))
        else:
            show = out

        pm = _to_pixmap(show)
        h, w = show.shape[:2]
        zw = max(1, int(round(w * self._zoom)))
        zh = max(1, int(round(h * self._zoom)))
        pm_scaled = pm.scaled(zw, zh,
                              Qt.AspectRatioMode.IgnoreAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        pm_scaled = self._draw_region_outline_on_pixmap(pm_scaled)
        self.lbl_preview.setPixmap(pm_scaled)
        self.lbl_preview.resize(zw, zh)

    def _overlay_mask(self, base, mask):
        base = _ensure_rgb01(base)
        alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None] * 0.6
        overlay = base.copy()
        overlay[..., 0] = np.clip(base[..., 0] * (1 - alpha[..., 0]) + 1.0 * alpha[..., 0], 0.0, 1.0)
        overlay[..., 1] = np.clip(base[..., 1] * (1 - alpha[..., 0]) + 0.0 * alpha[..., 0], 0.0, 1.0)
        overlay[..., 2] = np.clip(base[..., 2] * (1 - alpha[..., 0]) + 0.0 * alpha[..., 0], 0.0, 1.0)
        return overlay

    # =================================================================
    # Build full-res mask
    # =================================================================
    def _build_mask(self, base):
        if self._use_imported_mask and self._imported_mask_full is not None:
            imp = self._imported_mask_full
            bh, bw = base.shape[:2]
            mh, mw = imp.shape[:2]
            if (mh, mw) != (bh, bw):
                return cv2.resize(imp, (bw, bh), interpolation=cv2.INTER_LINEAR) if cv2 is not None else imp
            return np.clip(imp.astype(np.float32), 0.0, 1.0)

        lo = self.ds_lo.value()
        hi = self.ds_hi.value()
        if lo > hi:
            lo, hi = hi, lo
        mask = _build_lum_mask(
            base, lo=lo, hi=hi,
            smooth=float(self.ds_smooth.value()),
            invert=self.cb_invert.isChecked(),
            blur_px=int(self.sb_blur.value()),
        )

        # AND with region mask
        if self._region_mask_full is not None:
            bh, bw = base.shape[:2]
            rm = self._region_mask_full
            mh, mw = rm.shape[:2]
            if (mh, mw) != (bh, bw):
                rm = cv2.resize(rm, (bw, bh), interpolation=cv2.INTER_LINEAR) if cv2 is not None else rm
            mask = mask * np.clip(rm.astype(np.float32), 0.0, 1.0)

        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def _apply_fullres(self):
        base = self.img
        mask = self._build_mask(base)
        return _apply_selective_adjustments(
            base, mask,
            cyan=float(self.ds_c.value()),
            magenta=float(self.ds_m.value()),
            yellow=float(self.ds_y.value()),
            r=float(self.ds_r.value()),
            g=float(self.ds_g.value()),
            b=float(self.ds_b.value()),
            lum=float(self.ds_l.value()),
            chroma=float(self.ds_chroma.value()),
            sat=float(self.ds_s.value()),
            con=float(self.ds_c2.value()),
            intensity=float(self.ds_int.value()),
            use_chroma_mode=(self.dd_color_mode.currentIndex() == 0),
            band_lo=float(self.ds_lo.value()),
            band_hi=float(self.ds_hi.value()),       
        )

    # =================================================================
    # Apply / Export / Import mask
    # =================================================================
    def _apply_to_document(self):
        try:
            result = self._apply_fullres()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e)); return
        try: self._save_window_state()
        except Exception: pass
        try:
            if hasattr(self.document, "set_image"):
                self.document.set_image(result)
        except Exception:
            name = getattr(self.document, "display_name", lambda: "Image")()
            if hasattr(self.docman, "open_array"):
                self.docman.open_array(result, title=f"{name} [SelLum]")
        self.img = np.clip(result.astype(np.float32), 0.0, 1.0)
        self._last_base = None
        self._reset_adjustments_only()

    def _apply_as_new_doc(self):
        try:
            result = self._apply_fullres()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e)); return
        try: self._save_window_state()
        except Exception: pass
        name = getattr(self.document, "display_name", lambda: "Image")()
        new_doc = None
        if hasattr(self.docman, "open_array"):
            new_doc = self.docman.open_array(result, title=f"{name} [SelLum]")
        if new_doc is not None:
            self.document = new_doc
            try:
                disp = getattr(self.document, "display_name", lambda: "Image")()
            except Exception:
                disp = "Image"
            self.lbl_target.setText(f"Target View:  <b>{disp}</b>")
        self.img = np.clip(result.astype(np.float32), 0.0, 1.0)
        self._last_base = None
        self._reset_adjustments_only()

    def _export_mask_doc(self):
        if self.docman is None:
            QMessageBox.information(self, "No document manager", "Cannot export mask without a document manager."); return
        base = self.img
        if base is None:
            QMessageBox.information(self, "No image", "Open an image first."); return
        mask = self._build_mask(base)
        mask_rgb = np.repeat(mask[..., None], 3, axis=2).astype(np.float32)
        name = getattr(self.document, "display_name", lambda: "Image")()
        try:
            self.docman.open_array(mask_rgb, title=f"{name} [SelLum MASK]")
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    def _import_mask_from_view(self):
        if self.docman is None:
            QMessageBox.information(self, "No document manager", "Cannot import without a document manager."); return
        docs = self.docman.all_documents() or []
        img_docs = [d for d in docs if hasattr(d, "image") and d.image is not None]
        if not img_docs:
            QMessageBox.information(self, "No views", "There are no image views to import a mask from."); return
        items = []
        for d in img_docs:
            try:
                nm = d.display_name()
            except Exception:
                nm = "Untitled"
            items.append(nm)
        from PyQt6.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(self, "Pick mask view", "Open image views:", items, 0, False)
        if not ok: return
        sel_doc = next((d for d, nm in zip(img_docs, items) if nm == choice), None)
        if sel_doc is None or getattr(sel_doc, "image", None) is None:
            QMessageBox.warning(self, "Import failed", "Selected view has no image."); return
        mask_img = np.clip(sel_doc.image.astype(np.float32), 0.0, 1.0)
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        dst_h, dst_w = self.img.shape[:2]
        src_h, src_w = mask_img.shape[:2]
        if (src_h, src_w) != (dst_h, dst_w):
            mask_full = cv2.resize(mask_img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR) if cv2 is not None else mask_img
        else:
            mask_full = mask_img
        self._imported_mask_full = np.clip(mask_full.astype(np.float32), 0.0, 1.0)
        self._imported_mask_name = choice
        self.lbl_imported_mask.setText(f"Imported: {choice}")
        self.cb_use_imported.setChecked(True)
        self._use_imported_mask = True
        self._recompute_mask_and_preview()

    def _on_use_imported_mask_toggled(self, on):
        self._use_imported_mask = bool(on)
        if self._use_imported_mask and self._imported_mask_full is None:
            self._use_imported_mask = False
            self.cb_use_imported.setChecked(False)
            QMessageBox.information(self, "No imported mask", "Pick a mask view first.")
            return
        self._recompute_mask_and_preview()

    # =================================================================
    # Reset
    # =================================================================

    def _reset_adjustments_only(self):
        """
        Called after Apply / Apply as New Document.
        Zeroes only the CMY/RGB/L/Chroma/Sat/Contrast/Intensity sliders.
        Leaves the luminance range, preset, smoothness, blur, and invert
        exactly as the user left them so they can keep tweaking the same band.
        """
        self._mask_timer.stop()
        self._adj_timer.stop()

        for sld, box in (
            (self.sl_c,      self.ds_c),
            (self.sl_m,      self.ds_m),
            (self.sl_y,      self.ds_y),
            (self.sl_r,      self.ds_r),
            (self.sl_g,      self.ds_g),
            (self.sl_b,      self.ds_b),
            (self.sl_l,      self.ds_l),
            (self.sl_s,      self.ds_s),
            (self.sl_c2,     self.ds_c2),
            (self.sl_chroma, self.ds_chroma),
        ):
            self._set_pair(sld, box, 0.0)

        self.ds_int.blockSignals(True)
        self.ds_int.setValue(1.0)
        self.ds_int.blockSignals(False)

        self.dd_color_mode.blockSignals(True)
        self.dd_color_mode.setCurrentIndex(0)
        self.dd_color_mode.blockSignals(False)
        self._update_color_mode_enabled()

        # Rebuild preview with the freshly-applied image but same mask range
        self._last_base = None
        self._recompute_mask_and_preview()

    def _reset_controls(self):
        """
        Full reset — called by the ↺ Reset button.
        Resets everything including the luminance range back to Shadows.
        """
        self._mask_timer.stop()
        self._adj_timer.stop()

        self._syncing_lum = True
        try:
            self.dd_preset.blockSignals(True)
            self.dd_preset.setCurrentText("Shadows")
            self.dd_preset.blockSignals(False)
            lo, hi = _LUM_PRESETS["Shadows"]
            self.lum_wheel.setRange(lo, hi, notify=False)
            for w, val in (
                (self.sl_lo, int(round(lo * 255))), (self.ds_lo, lo),
                (self.sl_hi, int(round(hi * 255))), (self.ds_hi, hi),
            ):
                w.blockSignals(True); w.setValue(val); w.blockSignals(False)
        finally:
            self._syncing_lum = False

        def setv(w, val):
            w.blockSignals(True)
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                w.setValue(val)
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QComboBox):
                idx = w.findText(val)
                if idx >= 0: w.setCurrentIndex(idx)
            elif isinstance(w, QSlider):
                w.setValue(int(val))
            w.blockSignals(False)

        setv(self.ds_smooth, 0.05)
        setv(self.cb_invert, False)
        setv(self.sb_blur, 0)
        setv(self.cb_show_mask, False)

        for sld, box in (
            (self.sl_c,      self.ds_c),
            (self.sl_m,      self.ds_m),
            (self.sl_y,      self.ds_y),
            (self.sl_r,      self.ds_r),
            (self.sl_g,      self.ds_g),
            (self.sl_b,      self.ds_b),
            (self.sl_l,      self.ds_l),
            (self.sl_s,      self.ds_s),
            (self.sl_c2,     self.ds_c2),
            (self.sl_chroma, self.ds_chroma),
        ):
            self._set_pair(sld, box, 0.0)

        self.dd_color_mode.blockSignals(True)
        self.dd_color_mode.setCurrentIndex(0)
        self.dd_color_mode.blockSignals(False)
        self._update_color_mode_enabled()

        self.ds_int.blockSignals(True); self.ds_int.setValue(1.0); self.ds_int.blockSignals(False)
        self.lum_wheel.setPickedLum(None)
        self._recompute_mask_and_preview()

    # =================================================================
    # Window state persistence
    # =================================================================
    def _save_window_state(self):
        try:
            s = QSettings()
            s.setValue("sellum/window_geometry", self.saveGeometry())
            if hasattr(self, "splitter"):
                s.setValue("sellum/splitter_state", self.splitter.saveState())
            s.setValue("sellum/small_preview", bool(self.cb_small_preview.isChecked()))
            s.setValue("sellum/live_preview",  bool(self.cb_live.isChecked()))
            s.setValue("sellum/show_mask",     bool(self.cb_show_mask.isChecked()))
            s.setValue("sellum/zoom",          float(getattr(self, "_zoom", 1.0)))
            try: s.sync()
            except Exception: pass
        except Exception: pass

    def _restore_window_state(self):
        try:
            s = QSettings()
            g = s.value("sellum/window_geometry", None)
            if g is not None: self.restoreGeometry(g)
            sp = s.value("sellum/splitter_state", None)
            if sp is not None and hasattr(self, "splitter"):
                self.splitter.restoreState(sp)
            self.cb_small_preview.setChecked(bool(s.value("sellum/small_preview", True, type=bool)))
            self.cb_live.setChecked(bool(s.value("sellum/live_preview", True, type=bool)))
            self.cb_show_mask.setChecked(bool(s.value("sellum/show_mask", False, type=bool)))
            z = float(s.value("sellum/zoom", 1.0))
            self._zoom = max(0.05, min(16.0, z))
        except Exception: pass

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._update_preview_pixmap)

    def showEvent(self, ev):
        super().showEvent(ev)
        if getattr(self, "_geom_restored", False): return
        self._geom_restored = True
        def _after():
            self._restore_window_state()
            self._last_base = None
            self._recompute_mask_and_preview()
        QTimer.singleShot(0, _after)

    def closeEvent(self, ev):
        try: self._save_window_state()
        except Exception: pass
        super().closeEvent(ev)