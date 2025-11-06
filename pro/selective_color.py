from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QGroupBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QSpinBox, QDoubleSpinBox,
    QFileDialog
)
from PyQt6.QtWidgets import QSizePolicy
# ---------------------------------------------------------------------
# Small helpers
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

def _rgb_to_hsv01(img01: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for Selective Color.")
    # expects 8-bit for best speed
    u8 = _to_uint8_rgb(img01)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)  # H in [0,180], S,V in [0,255]
    out = np.empty_like(hsv, dtype=np.float32)
    out[...,0] = hsv[...,0].astype(np.float32) / 180.0  # 0..1
    out[...,1] = hsv[...,1].astype(np.float32) / 255.0
    out[...,2] = hsv[...,2].astype(np.float32) / 255.0
    return out

def _luminance01(img01: np.ndarray) -> np.ndarray:
    if img01.ndim == 2:
        return np.clip(img01, 0.0, 1.0).astype(np.float32)
    r, g, b = img01[...,0], img01[...,1], img01[...,2]
    return (0.2989*r + 0.5870*g + 0.1140*b).astype(np.float32)

def _softstep(x, edge0, edge1):
    # smoothstep
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3 - 2*t)

# ---------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------

_PRESETS = {
    "Red":     [(340, 360), (0, 15)],
    "Orange":  [(15, 40)],
    "Yellow":  [(40, 70)],
    "Green":   [(70, 170)],
    "Cyan":    [(170, 200)],
    "Blue":    [(200, 270)],
    "Magenta": [(270, 340)],
}

def _hue_band(Hdeg: np.ndarray, lo: float, hi: float, smooth_deg: float) -> np.ndarray:
    """
    Soft band on the hue circle (degrees 0..360), but with *local* feathering:
    - core band is the forward arc lo → hi
    - smooth_deg only adds a ramp *right after hi* and *right before lo*
    - never balloons into the whole hue wheel
    """
    H = Hdeg.astype(np.float32)

    lo = float(lo) % 360.0
    hi = float(hi) % 360.0

    # length of the forward arc
    L = (hi - lo) % 360.0
    if L <= 1e-6:
        return np.zeros_like(H, dtype=np.float32)

    s = float(max(smooth_deg, 0.0))

    # forward distance from lo → hue (always 0..360)
    fwd = (H - lo) % 360.0
    # backward distance from hue → lo (always 0..360)
    bwd = (lo - H) % 360.0

    # start with zeros
    band = np.zeros_like(H, dtype=np.float32)

    # 1) core: strictly inside the band
    inside = (fwd <= L)
    band[inside] = 1.0

    if s > 1e-6:
        # 2) upper feather: just after hi
        upper = (fwd > L) & (fwd < L + s)
        band[upper] = np.maximum(
            band[upper],
            1.0 - (fwd[upper] - L) / s
        )

        # 3) lower feather: just before lo (going backwards)
        lower = (bwd > 0) & (bwd < s)
        band[lower] = np.maximum(
            band[lower],
            1.0 - bwd[lower] / s
        )

    return np.clip(band, 0.0, 1.0).astype(np.float32)



def _hue_mask(img01: np.ndarray,
              ranges_deg: list[tuple[float,float]],
              min_chroma: float,
              min_light: float,
              max_light: float,
              smooth_deg: float,
              invert_range: bool = False) -> np.ndarray:
    """
    Return mask in 0..1 for the UNION of hue bands in ranges_deg (degrees).
    Handles wrap-around without recursion. If invert_range=True, selects the
    COMPLEMENT of the union on the hue circle (before chroma/light gating).
    """
    hsv  = _rgb_to_hsv01(img01)                 # H in [0..1)
    Hdeg = (np.mod(hsv[..., 0] * 360.0, 360.0)).astype(np.float32)
    S    = hsv[..., 1].astype(np.float32)
    V    = hsv[..., 2].astype(np.float32)

    m = np.zeros_like(Hdeg, dtype=np.float32)
    for lo, hi in ranges_deg:
        m = np.maximum(m, _hue_band(Hdeg, lo, hi, smooth_deg))

    # Invert selection on the hue circle if requested
    if invert_range:
        m = 1.0 - m

    # chroma/light gating
    if min_chroma > 0:
        chroma = (S * V).astype(np.float32)
        m *= _softstep(chroma, float(min_chroma)*0.7, float(min_chroma))
    if min_light > 0:
        m *= (V >= float(min_light)).astype(np.float32)
    if max_light < 1:
        m *= (V <= float(max_light)).astype(np.float32)

    return np.clip(m, 0.0, 1.0)




def _weight_shadows_highlights(mask: np.ndarray,
                               img01: np.ndarray,
                               shadows: float,
                               highlights: float,
                               balance: float) -> np.ndarray:
    """
    New behavior:
    - `shadows` in [0..1]: pixels BELOW this luminance get faded OUT.
    - `highlights` in [0..1]: pixels ABOVE this luminance get faded OUT.
    - `balance` just tweaks feather width (optional).
    """
    L = _luminance01(img01).astype(np.float32)
    w = np.ones_like(L, dtype=np.float32)

    # feather size ~ 8% of range, you can tune this
    feather = 0.08 + 0.12 * balance  # 0.08..0.2

    # 1) shadow gate: fade OUT below `shadows`
    if shadows > 1e-3:
        s0 = max(0.0, shadows - feather)
        s1 = min(1.0, shadows + 1e-6)
        # below s0 → 0, above s1 → 1
        w *= _softstep(L, s0, s1)

    # 2) highlight gate: fade OUT above `highlights`
    if highlights < 0.999:
        h0 = max(0.0, highlights - 1e-6)
        h1 = min(1.0, highlights + feather)
        # below h0 → 1, above h1 → 0
        w *= (1.0 - _softstep(L, h0, h1))

    # apply to mask
    return np.clip(mask * w, 0.0, 1.0)


# ---------------------------------------------------------------------
# Color adjustments
# ---------------------------------------------------------------------

def _apply_selective_adjustments(img01: np.ndarray,
                                 mask01: np.ndarray,
                                 cyan: float, magenta: float, yellow: float,
                                 r: float, g: float, b: float,
                                 lum: float, chroma: float, sat: float, con: float,
                                 intensity: float,
                                 use_chroma_mode: bool) -> np.ndarray:

    """
    CMY/RGB sliders in [-1..+1] range (we’ll clamp).
    L/S/C also in [-1..+1].
    """
    a = img01.astype(np.float32).copy()
    m = np.clip(mask01.astype(np.float32) * float(intensity), 0.0, 1.0)

    # RGB base
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)

    R = a[...,0]; G = a[...,1]; B = a[...,2]

    # CMY = reduce the complementary primary
    # Positive Cyan -> reduce Red; negative Cyan -> increase Red.
    R = np.clip(R + (-cyan) * m, 0.0, 1.0)
    G = np.clip(G + (-magenta) * m, 0.0, 1.0)
    B = np.clip(B + (-yellow) * m, 0.0, 1.0)

    # Primary boosts
    R = np.clip(R + r * m, 0.0, 1.0)
    G = np.clip(G + g * m, 0.0, 1.0)
    B = np.clip(B + b * m, 0.0, 1.0)

    out = np.stack([R,G,B], axis=-1)

    # L / Chroma-or-Sat / Contrast
    if any(abs(x) > 1e-6 for x in (lum, chroma, sat, con)):
        if abs(lum) > 0:
            out = np.clip(out + lum * m[..., None], 0.0, 1.0)

        if abs(con) > 0:
            out = np.clip((out - 0.5) * (1.0 + con * m[..., None]) + 0.5, 0.0, 1.0)

        if use_chroma_mode:
            if abs(chroma) > 0:
                out = _apply_chroma_boost(out, m, chroma)
        else:
            if abs(sat) > 0:
                hsv = _rgb_to_hsv01(out)
                hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat * m), 0.0, 1.0)
                # HSV->RGB using cv2 (expects 8-bit)
                hv = (hsv[..., 0] * 180.0).astype(np.uint8)
                sv = (hsv[..., 1] * 255.0).astype(np.uint8)
                vv = (hsv[..., 2] * 255.0).astype(np.uint8)
                hsv8 = np.stack([hv, sv, vv], axis=-1)
                rgb8 = cv2.cvtColor(hsv8, cv2.COLOR_HSV2RGB)
                out = rgb8.astype(np.float32) / 255.0

    return np.clip(out, 0.0, 1.0)

def _apply_chroma_boost(rgb01: np.ndarray, m01: np.ndarray, chroma: float) -> np.ndarray:
    """
    L-preserving chroma change:
      rgb' = Y + (rgb - Y) * (1 + chroma * m)
    where Y is luminance and m is the 0..1 mask (with intensity applied upstream).
    Positive chroma -> more colorfulness; negative -> less.
    """
    rgb = _ensure_rgb01(rgb01).astype(np.float32)
    m   = np.clip(m01.astype(np.float32), 0.0, 1.0)[..., None]
    Y   = _luminance01(rgb)[..., None]                 # HxWx1
    d   = rgb - Y                                      # chroma direction
    k   = (1.0 + float(chroma) * m)                    # scale per-pixel with mask
    out = Y + d * k
    return np.clip(out, 0.0, 1.0)


def _ensure_rgb01(img: np.ndarray) -> np.ndarray:
    """Return an RGB float image in [0,1]."""
    a = np.clip(img.astype(np.float32), 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return a

class HueWheel(QWidget):
    """
    A compact HSV hue wheel with two draggable handles for start/end (degrees 0..360).
    Emits rangeChanged(start_deg, end_deg) when either handle moves.
    """
    from PyQt6.QtCore import pyqtSignal
    rangeChanged = pyqtSignal(int, int)

    def __init__(self, start_deg=65, end_deg=158, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 160)
        self._start = int(start_deg) % 360
        self._end   = int(end_deg)   % 360
        self._dragging = None  # "start" | "end" | None
        self._ring_img = None
        self._picked = None  # degrees or None

    # --- public API
    def setRange(self, start_deg: int, end_deg: int, notify=True):
        s = int(start_deg) % 360
        e = int(end_deg)   % 360
        if s == self._start and e == self._end:
            return
        self._start, self._end = s, e
        self.update()
        if notify:
            self.rangeChanged.emit(self._start, self._end)

    def range(self):
        return self._start, self._end

    def setPickedHue(self, deg: float | int | None):
        """Show a small marker on the wheel at the sampled hue (degrees)."""
        if deg is None:
            self._picked = None
        else:
            self._picked = int(deg) % 360
        self.update()

    # --- util
    @staticmethod
    def _ang_from_pos(cx, cy, x, y):
        import math
        a = math.degrees(math.atan2(y - cy, x - cx))
        a = (a + 360.0) % 360.0
        return a

    def _ensure_ring(self, side):
        # cache a color wheel image to paint fast
        if self._ring_img is not None and self._ring_img.width() == side and self._ring_img.height() == side:
            return
        import math
        side = int(side)
        img = np.zeros((side, side, 3), np.uint8)
        cx = cy = side // 2
        r  = int(side*0.48)
        rr2 = r*r
        for y in range(side):
            dy = y - cy
            for x in range(side):
                dx = x - cx
                d2 = dx*dx + dy*dy
                if rr2 - r*12 <= d2 <= rr2:  # thin ring
                    ang = self._ang_from_pos(cx, cy, x, y)
                    hsv = np.array([ang/2, 255, 255], np.uint8)   # H 0..180 in OpenCV
                    rgb = cv2.cvtColor(hsv[None,None,:], cv2.COLOR_HSV2RGB)[0,0]
                    img[y, x] = rgb
        h, w, _ = img.shape
        self._ring_img = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGB888).copy()

    # --- events
    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
        p = QPainter(self)
        side = min(self.width(), self.height())
        self._ensure_ring(side)

        # center ring
        x0 = (self.width() - side)//2
        y0 = (self.height() - side)//2
        p.drawImage(x0, y0, self._ring_img)

        # draw handles & arc
        cx = x0 + side//2
        cy = y0 + side//2
        r  = int(side*0.48)

        def pt(ang_deg):
            import math
            th = math.radians(ang_deg)
            return int(cx + r*math.cos(th)), int(cy + r*math.sin(th))

        # --- RANGE ARC (match mask logic) ---
        # Mask defines band as positive arc from start -> end with L = (end - start) % 360.
        s, e = int(self._start) % 360, int(self._end) % 360
        L = (e - s) % 360  # arc length in degrees (0..359)
        steps = 60
        if L > 0:
            p.setPen(QPen(QColor(255, 255, 255, 140), 4))
            px, py = pt(s)
            for k in range(1, steps + 1):
                a = (s + (L * k) / steps) % 360  # move forward along the positive arc
                qx, qy = pt(a)
                p.drawLine(px, py, qx, qy)
                px, py = qx, qy

        # handles
        p.setBrush(QBrush(QColor(255,255,255)))
        p.setPen(QPen(QColor(0,0,0), 1))
        for ang in (self._start, self._end):
            xh, yh = pt(ang)
            p.drawEllipse(xh-5, yh-5, 10, 10)

        # sampled hue marker
        if self._picked is not None:
            import math
            th = math.radians(self._picked)
            px = int(cx + r*math.cos(th)); py = int(cy + r*math.sin(th))
            p.setBrush(QBrush(QColor(0, 0, 0)))
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawEllipse(px-6, py-6, 12, 12)


    def mousePressEvent(self, ev):
        x, y = ev.position().x(), ev.position().y()
        side = min(self.width(), self.height())
        x0 = (self.width() - side)//2
        y0 = (self.height() - side)//2
        cx = x0 + side//2
        cy = y0 + side//2
        a = self._ang_from_pos(cx, cy, x, y)
        # pick the nearest handle
        def d(a0, a1):
            dd = abs((a0 - a1 + 180) % 360 - 180)
            return dd
        if d(a, self._start) <= d(a, self._end):
            self._dragging = "start"
            self._start = int(a)
        else:
            self._dragging = "end"
            self._end = int(a)
        self.update()
        self.rangeChanged.emit(self._start, self._end)

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return
        x, y = ev.position().x(), ev.position().y()
        side = min(self.width(), self.height())
        x0 = (self.width() - side)//2
        y0 = (self.height() - side)//2
        cx = x0 + side//2
        cy = y0 + side//2
        a = int(self._ang_from_pos(cx, cy, x, y)) % 360
        if self._dragging == "start":
            self._start = a
        else:
            self._end = a
        self.update()
        self.rangeChanged.emit(self._start, self._end)

    def mouseReleaseEvent(self, ev):
        self._dragging = None


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

class SelectiveColorCorrection(QDialog):
    """
    v1.0 — live preview, mask overlay, presets + custom hue range,
    CMY/RGB + L/S/C sliders. Loads active document's image.
    """
    def __init__(self, doc_manager=None, document=None, parent=None, window_icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("Selective Color Correction")
        if window_icon:
            self.setWindowIcon(window_icon)

        self.docman = doc_manager
        self.document = document
        if self.document is None or getattr(self.document, "image", None) is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            self.close(); return

        self.img = np.clip(self.document.image.astype(np.float32), 0.0, 1.0)
        self.preview_img = self.img.copy()

        self._imported_mask_full = None     # full-res mask (H x W) float32 0..1
        self._imported_mask_name = None     # nice label to show in UI
        self._use_imported_mask = False     # checkbox state mirror
        self._mask_delay_ms = 200
        self._adj_delay_ms  = 200
        self._build_ui()
        self._mask_delay_ms = 200  # 0.2s idle before recomputing mask
        self._mask_timer = QTimer(self)
        self._mask_timer.setSingleShot(True)
        self._mask_timer.timeout.connect(self._recompute_mask_and_preview)   
        self._adj_delay_ms = 200
        self._adj_timer = QTimer(self)
        self._adj_timer.setSingleShot(True)
        self._adj_timer.timeout.connect(self._update_preview_pixmap)    
        self.dd_preset.setCurrentText("Red")
        self._setting_preset = False
        self._recompute_mask_and_preview()

    # ------------- UI -------------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # LEFT: controls
        left = QVBoxLayout()
        root.addLayout(left, 0)

        # Target view
        lbl_t = QLabel(f"Target View:  <b>{getattr(self.document, 'display_name', lambda: 'Image')()}</b>")
        left.addWidget(lbl_t)

        # Preview mode (downsample switch)
        self.cb_small_preview = QCheckBox("Small-sized Preview (fast)")
        self.cb_small_preview.setChecked(True)
        self.cb_small_preview.toggled.connect(self._recompute_mask_and_preview)
        left.addWidget(self.cb_small_preview)

        # --- Mask block
        gb_mask = QGroupBox("Mask")
        gl = QGridLayout(gb_mask)
        gl.setContentsMargins(8, 8, 8, 8)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(8)

        # Row 0: preset
        gl.addWidget(QLabel("Preset:"), 0, 0)
        self.dd_preset = QComboBox()
        self.dd_preset.addItems(["Custom"] + list(_PRESETS.keys()))
        self.dd_preset.currentTextChanged.connect(self._on_preset_change)
        gl.addWidget(self.dd_preset, 0, 1, 1, 4)  # span across right side too

        # Left column: hue wheel (own vertical band, no overlap)
        self.hue_wheel = HueWheel(start_deg=65, end_deg=158)
        self.hue_wheel.setMinimumSize(170, 170)

        self.hue_wheel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        gl.addWidget(self.hue_wheel, 1, 0, 7, 2)

        # remember defaults here (on the instance actually in the UI)
        self._default_h_start, self._default_h_end = self.hue_wheel.range()
        # Give it fixed-ish behavior so it stays square and doesn't get crushed


        # Right side controls (start at col=2, never col 0/1)

        # Row 1: start/end sliders
        def make_deg_slider_row(caption, row):
            gl.addWidget(QLabel(caption), row, 2)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 360); s.setSingleStep(1); s.setPageStep(10)
            s.setValue(65 if "Start" in caption else 158)
            sp = QSpinBox(); sp.setRange(0, 360); sp.setValue(s.value())
            s.valueChanged.connect(sp.setValue); sp.valueChanged.connect(s.setValue)
            s.valueChanged.connect(self._recompute_mask_and_preview)
            sp.valueChanged.connect(self._recompute_mask_and_preview)
            gl.addWidget(s,  row, 3, 1, 2)
            gl.addWidget(sp, row, 5)
            return s, sp

        self.sl_h1, self.sp_h1 = make_deg_slider_row("Start hue", 1)
        self.sl_h2, self.sp_h2 = make_deg_slider_row("End hue",   2)

        # Keep wheel and sliders in sync
        def _wheel_to_sliders(s: int, e: int):
            # user moved the wheel → flip to Custom unless we're applying a preset
            if not self._setting_preset and self.dd_preset.currentText() != "Custom":
                self.dd_preset.setCurrentText("Custom")

            # update sliders/spinboxes without triggering recursion
            self.sl_h1.blockSignals(True); self.sp_h1.blockSignals(True)
            self.sl_h2.blockSignals(True); self.sp_h2.blockSignals(True)
            self.sl_h1.setValue(int(s));   self.sp_h1.setValue(int(s))
            self.sl_h2.setValue(int(e));   self.sp_h2.setValue(int(e))
            self.sl_h1.blockSignals(False); self.sp_h1.blockSignals(False)
            self.sl_h2.blockSignals(False); self.sp_h2.blockSignals(False)

            # debounce the mask recompute
            self._schedule_mask()

        self.hue_wheel.rangeChanged.connect(_wheel_to_sliders)

        def _sliders_to_wheel(_=None):
            # user moved sliders/spinboxes → flip to Custom unless preset-driving
            if not self._setting_preset and self.dd_preset.currentText() != "Custom":
                self.dd_preset.setCurrentText("Custom")

            s = int(self.sp_h1.value())
            e = int(self.sp_h2.value())

            # update wheel but don't re-emit rangeChanged (prevents ping-pong)
            self.hue_wheel.setRange(s, e, notify=False)

            # debounce the mask recompute
            self._schedule_mask()

        self.sp_h1.valueChanged.connect(_sliders_to_wheel)
        self.sp_h2.valueChanged.connect(_sliders_to_wheel)
        self.sl_h1.valueChanged.connect(_sliders_to_wheel)
        self.sl_h2.valueChanged.connect(_sliders_to_wheel)

        # Row 3: chroma + lightness
        gl.addWidget(QLabel("Min chroma:"), 3, 2)
        self.ds_minC = QDoubleSpinBox(); self.ds_minC.setRange(0,1); self.ds_minC.setSingleStep(0.05); self.ds_minC.setValue(0.0)
        self.ds_minC.valueChanged.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.ds_minC, 3, 3)

        gl.addWidget(QLabel("Lightness min/max:"), 3, 4)
        self.ds_minL = QDoubleSpinBox(); self.ds_minL.setRange(0,1); self.ds_minL.setSingleStep(0.05); self.ds_minL.setValue(0.0)
        self.ds_maxL = QDoubleSpinBox(); self.ds_maxL.setRange(0,1); self.ds_maxL.setSingleStep(0.05); self.ds_maxL.setValue(1.0)
        self.ds_minL.valueChanged.connect(self._recompute_mask_and_preview)
        self.ds_maxL.valueChanged.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.ds_minL, 3, 5)
        gl.addWidget(QLabel("to"), 3, 6)
        gl.addWidget(self.ds_maxL, 3, 7)

        # Row 4: smoothness
        gl.addWidget(QLabel("Smoothness (deg):"), 4, 2)
        self.ds_smooth = QDoubleSpinBox(); self.ds_smooth.setRange(0,60); self.ds_smooth.setSingleStep(1.0); self.ds_smooth.setValue(10.0)
        self.ds_smooth.valueChanged.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.ds_smooth, 4, 3)

        self.cb_invert = QCheckBox("Invert hue range")
        self.cb_invert.setChecked(False)
        self.cb_invert.toggled.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.cb_invert, 4, 4, 1, 3)  # place to the right of Smoothness

        # Row 5: shadows/highlights  (Balance is hidden; Intensity moved here)
        gl.addWidget(QLabel("Shadows:"),   5, 2)
        self.ds_sh = QDoubleSpinBox(); self.ds_sh.setRange(0,1); self.ds_sh.setSingleStep(0.05); self.ds_sh.setValue(0.0)
        self.ds_sh.valueChanged.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.ds_sh, 5, 3)

        gl.addWidget(QLabel("Highlights:"), 5, 4)
        self.ds_hi = QDoubleSpinBox(); self.ds_hi.setRange(0,1); self.ds_hi.setSingleStep(0.05); self.ds_hi.setValue(1.0)
        self.ds_hi.valueChanged.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.ds_hi, 5, 5)

        # --- hidden balance control (still used in math) ---
        self.ds_bal = QDoubleSpinBox()
        self.ds_bal.setRange(0,1)
        self.ds_bal.setSingleStep(0.05)
        self.ds_bal.setValue(0.5)
        self.ds_bal.valueChanged.connect(self._recompute_mask_and_preview)
        self.ds_bal.setVisible(False)  # not added to layout, or you can add+hide

        # put Intensity where Balance was
        gl.addWidget(QLabel("Intensity:"), 5, 6)
        self.ds_int = QDoubleSpinBox(); self.ds_int.setRange(0, 2.0); self.ds_int.setSingleStep(0.05); self.ds_int.setValue(1.0)
        self.ds_int.valueChanged.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.ds_int, 5, 7)

        # Row 6: blur + overlay
        gl.addWidget(QLabel("Edge blur (px):"), 6, 2)
        self.sb_blur = QSpinBox(); self.sb_blur.setRange(0, 150); self.sb_blur.setValue(0)
        self.sb_blur.valueChanged.connect(self._recompute_mask_and_preview)
        gl.addWidget(self.sb_blur, 6, 3)

        self.cb_show_mask = QCheckBox("Show mask overlay")
        self.cb_show_mask.setChecked(False)
        self.cb_show_mask.toggled.connect(self._update_preview_pixmap)
        gl.addWidget(self.cb_show_mask, 6, 4, 1, 2)

        # Row 7: imported mask controls
        self.cb_use_imported = QCheckBox("Use imported mask")
        self.cb_use_imported.setChecked(False)
        self.cb_use_imported.toggled.connect(self._on_use_imported_mask_toggled)
        gl.addWidget(self.cb_use_imported, 7, 2, 1, 2)

        self.btn_import_mask = QPushButton("Pick mask from view…")
        self.btn_import_mask.clicked.connect(self._import_mask_from_view)
        gl.addWidget(self.btn_import_mask, 7, 4, 1, 2)

        self.lbl_imported_mask = QLabel("No imported mask")
        gl.addWidget(self.lbl_imported_mask, 7, 6, 1, 2)

        # Column sizing: wheel column fixed, right side stretchy
        gl.setColumnStretch(0, 0)
        gl.setColumnStretch(1, 0)
        for c in (2,3,4,5,6,7):
            gl.setColumnStretch(c, 1)

        left.addWidget(gb_mask)

        # --- Adjustments
        # CMY group
        gb_cmy = QGroupBox("Complementary colors (CMY)")
        glc = QGridLayout(gb_cmy)
        self.sl_c, self.ds_c = self._slider_pair(glc, "Cyan:",    0)
        self.sl_m, self.ds_m = self._slider_pair(glc, "Magenta:", 1)
        self.sl_y, self.ds_y = self._slider_pair(glc, "Yellow:",  2)
        left.addWidget(gb_cmy)

        # RGB group
        gb_rgb = QGroupBox("RGB Colors")
        glr = QGridLayout(gb_rgb)
        self.sl_r, self.ds_r = self._slider_pair(glr, "Red:",   0)
        self.sl_g, self.ds_g = self._slider_pair(glr, "Green:", 1)
        self.sl_b, self.ds_b = self._slider_pair(glr, "Blue:",  2)
        left.addWidget(gb_rgb)

        # LSC group
        # LSC group
        gb_lsc = QGroupBox("Luminance, Chroma/Saturation, Contrast")
        gll = QGridLayout(gb_lsc)

        # Row 0: Luminance
        self.sl_l, self.ds_l  = self._slider_pair(gll, "Luminance:",  0)

        # Row 1: Chroma (L-preserving)
        self.sl_chroma, self.ds_chroma = self._slider_pair(gll, "Chroma (L-preserving):", 1)

        # Row 2: Saturation (HSV S)
        self.sl_s, self.ds_s  = self._slider_pair(gll, "Saturation (HSV S):", 2)

        # Row 3: Contrast
        self.sl_c2, self.ds_c2 = self._slider_pair(gll, "Contrast:",  3)

        # Row 4: Mode selector (which one to apply)
        gll.addWidget(QLabel("Color boost mode:"), 4, 0)
        self.dd_color_mode = QComboBox()
        self.dd_color_mode.addItems(["Chroma (L-preserving)", "Saturation (HSV S)"])
        self.dd_color_mode.setCurrentIndex(0)  # default to Chroma for astro
        self.dd_color_mode.currentIndexChanged.connect(self._update_color_mode_enabled)
        gll.addWidget(self.dd_color_mode, 4, 1, 1, 2)

        left.addWidget(gb_lsc)


        # Preview + actions
        self.cb_live = QCheckBox("Preview changed image")
        self.cb_live.setChecked(True)
        self.cb_live.toggled.connect(self._update_preview_pixmap)
        left.addWidget(self.cb_live)

        row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._apply_to_document)

        self.btn_push = QPushButton("Apply as New Document")
        self.btn_push.clicked.connect(self._apply_as_new_doc)

        self.btn_export_mask = QPushButton("Export Mask")
        self.btn_export_mask.clicked.connect(self._export_mask_doc)

        # NEW: Reset
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.clicked.connect(self._reset_controls)

        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_push)
        row.addWidget(self.btn_export_mask)
        row.addWidget(self.btn_reset)
        left.addLayout(row)

        self.lbl_target = QLabel(f"Target View:  <b>{getattr(self.document, 'display_name', lambda: 'Image')()}</b>")
        left.addWidget(self.lbl_target)

        # RIGHT: image preview
        # RIGHT: image preview + zoom bar
        right = QVBoxLayout()
        root.addLayout(right, 1)

        # zoom toolbar
        zoom_row = QHBoxLayout()
        self.btn_zoom_in  = QPushButton("Zoom +")
        self.btn_zoom_out = QPushButton("Zoom –")
        self.btn_zoom_1   = QPushButton("1:1")
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_1)
        zoom_row.addStretch(1)
        right.addLayout(zoom_row)

        # scrollable image
        from PyQt6.QtWidgets import QScrollArea
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.lbl_preview = QLabel()
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_preview.setMinimumSize(10, 10)
        self.scroll.setWidget(self.lbl_preview)
        right.addWidget(self.scroll, 1)

        # sampled hue readout
        self.lbl_hue_readout = QLabel("Picked hue: —")
        right.addWidget(self.lbl_hue_readout)

        # zoom state
        self._zoom = 1.0
        def _apply_zoom(z):
            self._zoom = max(0.05, min(16.0, float(z)))
            self._update_preview_pixmap()

        self.btn_zoom_in.clicked.connect(lambda: _apply_zoom(self._zoom * 1.25))
        self.btn_zoom_out.clicked.connect(lambda: _apply_zoom(self._zoom / 1.25))
        self.btn_zoom_1.clicked.connect(lambda: _apply_zoom(1.0))

        # optional: ctrl+wheel zoom over the label
        def _wheel(ev):
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                _apply_zoom(self._zoom * (1.25 if ev.angleDelta().y() > 0 else 1/1.25))
                ev.accept()
                return True
            return False
        self.lbl_preview.wheelEvent = lambda ev: (_wheel(ev) or QLabel.wheelEvent(self.lbl_preview, ev))

        # enable hue sampling on the label itself
        self.lbl_preview.setMouseTracking(True)
        self.lbl_preview.installEventFilter(self)

        # allow clicking on the preview to sample hue
        self.lbl_preview.setMouseTracking(True)
        self.lbl_preview.installEventFilter(self)
        self._update_preview_pixmap()

        # tweak
        self.resize(1080, 680)

        self._update_color_mode_enabled()

        # any slider change should refresh preview
        for w in (self.ds_c, self.ds_m, self.ds_y, self.ds_r, self.ds_g, self.ds_b, self.ds_l, self.ds_s, self.ds_c2, self.ds_int):
            w.valueChanged.connect(self._schedule_adjustments)

    def _update_color_mode_enabled(self):
        use_chroma = (self.dd_color_mode.currentIndex() == 0)
        # enable Chroma controls when chroma mode; disable Sat controls, and vice versa
        self.ds_chroma.setEnabled(use_chroma); self.sl_chroma.setEnabled(use_chroma)
        self.ds_s.setEnabled(not use_chroma);  self.sl_s.setEnabled(not use_chroma)
        # refresh preview
        self._schedule_adjustments()


    def _set_pair(self, sld: QSlider, box: QDoubleSpinBox, value: float):
        # block both sides to avoid ping-pong and callbacks
        sld.blockSignals(True); box.blockSignals(True)
        sld.setValue(int(round(value * 100)))  # because slider units are *100
        box.setValue(float(value))
        sld.blockSignals(False); box.blockSignals(False)


    def _reset_controls(self):
        """Reset all UI controls to defaults and rebuild mask/preview on current self.img."""
        # pause timers while resetting
        self._mask_timer.stop()
        self._adj_timer.stop()

        # --- Preset: make 'Red' the default and let _on_preset_change drive the wheel/sliders ---
        # IMPORTANT: do NOT overwrite with 'Custom' afterwards.
        self._setting_preset = True
        try:
            # This emits currentTextChanged -> _on_preset_change(), which:
            #  - sets the hue_wheel to the preset range (notify=False)
            #  - sets sp_h1/sp_h2 to the preset lo/hi
            #  - calls _recompute_mask_and_preview()
            self.dd_preset.setCurrentText("Red")
        finally:
            self._setting_preset = False

        # --- Mask gating defaults (won't change the preset/wheel) ---
        def setv(w, val):
            w.blockSignals(True)
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                w.setValue(val)
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QComboBox):
                idx = w.findText(val)
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QSlider):
                w.setValue(int(val))
            w.blockSignals(False)

        setv(self.ds_minC, 0.0)
        setv(self.ds_minL, 0.0)
        setv(self.ds_maxL, 1.0)
        setv(self.ds_smooth, 10.0)
        setv(self.cb_invert, False)

        # Shadows/Highlights/Balance
        setv(self.ds_sh, 0.0)
        setv(self.ds_hi, 1.0)
        setv(self.ds_bal, 0.5)

        # Blur / overlays / preview
        setv(self.sb_blur, 0)
        setv(self.cb_show_mask, False)
        # keep user’s small/large preview choice & zoom as-is

        # CMY/RGB/LSC back to 0, intensity to 1.0
        self._set_pair(self.sl_c,  self.ds_c,  0.0)
        self._set_pair(self.sl_m,  self.ds_m,  0.0)
        self._set_pair(self.sl_y,  self.ds_y,  0.0)
        self._set_pair(self.sl_r,  self.ds_r,  0.0)
        self._set_pair(self.sl_g,  self.ds_g,  0.0)
        self._set_pair(self.sl_b,  self.ds_b,  0.0)
        self._set_pair(self.sl_l,  self.ds_l,  0.0)
        self._set_pair(self.sl_s,  self.ds_s,  0.0)
        self._set_pair(self.sl_c2, self.ds_c2, 0.0)

        self._set_pair(self.sl_chroma, self.ds_chroma, 0.0)
        # default to Chroma mode
        self.dd_color_mode.blockSignals(True)
        self.dd_color_mode.setCurrentIndex(0)
        self.dd_color_mode.blockSignals(False)
        self._update_color_mode_enabled()

        self.ds_int.blockSignals(True)
        self.ds_int.setValue(1.0)
        self.ds_int.blockSignals(False)

        # Clear any sampled hue marker on the wheel
        self.hue_wheel.setPickedHue(None)

        # Rebuild preview (preset handler already recomputed the mask, but this is safe)
        self._recompute_mask_and_preview()


    def _schedule_adjustments(self, delay_ms: int | None = None):
        if delay_ms is None:
            delay_ms = getattr(self, "_adj_delay_ms", 200)
        # if called very early, just no-op safely
        if not hasattr(self, "_adj_timer"):
            return
        self._adj_timer.stop()
        self._adj_timer.start(int(delay_ms))


    def _schedule_mask(self, delay_ms: int | None = None):
        """Debounce mask recomputation for hue changes."""
        if delay_ms is None:
            delay_ms = self._mask_delay_ms
        # restart the timer on every change
        self._mask_timer.stop()
        self._mask_timer.start(int(delay_ms))


    def _sample_hue_deg_from_base(self, x: int, y: int) -> float | None:
        """Return hue in degrees at (x,y) in _last_base (float RGB in [0,1])."""
        base = getattr(self, "_last_base", None)
        if base is None:
            return None
        h, w = base.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        pix = base[y:y+1, x:x+1, :] if base.ndim == 3 else np.repeat(base[y:y+1, x:x+1][...,None], 3, axis=2)
        hsv = _rgb_to_hsv01(pix)  # 1x1x3, H in [0,1]
        return float(hsv[0,0,0] * 360.0)

    def _map_label_point_to_image_xy(self, ev_pos):
        """Map a click on the *label* to base image (x,y), accounting for zoom."""
        base = getattr(self, "_last_base", None)
        if base is None:
            return None
        bh, bw = base.shape[:2]
        # ev_pos is in the label's local coordinates
        x = int(round(ev_pos.x() / max(self._zoom, 1e-6)))
        y = int(round(ev_pos.y() / max(self._zoom, 1e-6)))
        if x < 0 or y < 0 or x >= bw or y >= bh:
            return None
        return (x, y)


    def eventFilter(self, obj, ev):
        from PyQt6.QtCore import QEvent, Qt
        if obj is self.lbl_preview and ev.type() == QEvent.Type.MouseButtonPress:
            pt = self._map_label_point_to_image_xy(ev.position())
            if pt is not None:
                x, y = pt
                hue = self._sample_hue_deg_from_base(x, y)
                if hue is not None:
                    # show marker on wheel and text readout
                    self.hue_wheel.setPickedHue(hue)
                    self.lbl_hue_readout.setText(f"Picked hue: {hue:.1f}°")

                    # Optional: Shift-click to set range centered on the hue (±15°)
                    if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        half = 15
                        start = int((hue - half) % 360)
                        end   = int((hue + half) % 360)
                        # this updates sliders & preview via our sync wiring
                        self.hue_wheel.setRange(start, end)
            return True
        return super().eventFilter(obj, ev)


    def _slider_row(self, grid: QGridLayout, name: str, row: int) -> QDoubleSpinBox:
        grid.addWidget(QLabel(name), row, 0)
        s = QDoubleSpinBox()
        s.setRange(-1.0, 1.0); s.setSingleStep(0.05); s.setDecimals(2); s.setValue(0.0)
        s.valueChanged.connect(self._recompute_mask_and_preview)
        grid.addWidget(s, row, 1)
        return s

    def _slider_pair(self, grid: QGridLayout, name: str, row: int, minv=-1.0, maxv=1.0, step=0.05):
        import math

        def _to_slider(v: float) -> int:
            # Symmetric rounding away from zero at half-steps; no banker’s rounding.
            s = abs(v) * 100.0
            s = math.floor(s + 0.5)
            return int(-s if v < 0 else s)

        def _to_spin(v_int: int) -> float:
            return float(v_int) / 100.0

        grid.addWidget(QLabel(name), row, 0)

        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(int(minv*100), int(maxv*100))   # e.g., -100..100
        sld.setSingleStep(int(step*100))             # e.g., 5
        sld.setPageStep(int(5*step*100))             # e.g., 25
        sld.setValue(0)

        box = QDoubleSpinBox()
        box.setRange(minv, maxv)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setValue(0.0)
        box.setKeyboardTracking(False)  # only fire on committed changes

        # Two-way binding without ping-pong
        def _sld_to_box(v_int: int):
            box.blockSignals(True)
            box.setValue(_to_spin(v_int))
            box.blockSignals(False)

        def _box_to_sld(v_float: float):
            sld.blockSignals(True)
            sld.setValue(_to_slider(v_float))
            sld.blockSignals(False)

        sld.valueChanged.connect(_sld_to_box)
        box.valueChanged.connect(_box_to_sld)

        # Debounced preview updates (adjustments don’t rebuild mask)
        sld.valueChanged.connect(self._schedule_adjustments)
        box.valueChanged.connect(self._schedule_adjustments)
        # Nice UX: force one final refresh on release
        sld.sliderReleased.connect(self._update_preview_pixmap)
        box.editingFinished.connect(self._update_preview_pixmap)

        grid.addWidget(sld, row, 1)
        grid.addWidget(box, row, 2)
        return sld, box


    # ------------- Logic -------------
    def _on_preset_change(self, txt: str):
        self._setting_preset = True
        try:
            if txt != "Custom":
                intervals = _PRESETS.get(txt, [])
                if intervals:
                    lo, hi = (intervals[0][0], intervals[-1][1]) if len(intervals) > 1 else intervals[0]
                    self.hue_wheel.setRange(int(lo), int(hi), notify=False)  # update wheel silently
                    self.hue_wheel.update()  # ensure repaint
                    self.sp_h1.blockSignals(True); self.sp_h2.blockSignals(True)
                    self.sp_h1.setValue(int(lo));  self.sp_h2.setValue(int(hi))
                    self.sp_h1.blockSignals(False); self.sp_h2.blockSignals(False)
            self._recompute_mask_and_preview()
        finally:
            self._setting_preset = False


    def _downsample(self, img, max_dim=1024):
        h, w = img.shape[:2]
        s = max(h, w)
        if s <= max_dim: return img
        k = max_dim / float(s)
        if cv2 is None:
            return cv2.resize(img, (int(w*k), int(h*k))) if False else img[::int(1/k), ::int(1/k)]
        return cv2.resize(img, (int(w*k), int(h*k)), interpolation=cv2.INTER_AREA)

    def _recompute_mask_and_preview(self):
        if self.img is None:
            return

        base = self._downsample(self.img, 1200) if self.cb_small_preview.isChecked() else self.img
        self._last_base = base

        # if user wants imported mask and we have one → use it
        if self._use_imported_mask and self._imported_mask_full is not None:
            imp = self._imported_mask_full
            bh, bw = base.shape[:2]
            mh, mw = imp.shape[:2]
            if (mh, mw) != (bh, bw):
                if cv2 is not None:
                    mask = cv2.resize(imp, (bw, bh), interpolation=cv2.INTER_LINEAR)
                else:
                    yy = (np.linspace(0, mh - 1, bh)).astype(int)
                    xx = (np.linspace(0, mw - 1, bw)).astype(int)
                    mask = imp[yy[:, None], xx[None, :]]
            else:
                mask = imp
            mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
        else:
            # your original hue-based build
            preset = self.dd_preset.currentText()
            if preset == "Custom":
                ranges = [(float(self.sp_h1.value()), float(self.sp_h2.value()))]
            else:
                ranges = _PRESETS[preset]

            mask = _hue_mask(
                base,
                ranges_deg=ranges,
                min_chroma=float(self.ds_minC.value()),
                min_light=float(self.ds_minL.value()),
                max_light=float(self.ds_maxL.value()),
                smooth_deg=float(self.ds_smooth.value()),
                invert_range=self.cb_invert.isChecked(),
            )

            mask = _weight_shadows_highlights(
                mask, base,
                shadows=float(self.ds_sh.value()),
                highlights=float(self.ds_hi.value()),
                balance=float(self.ds_bal.value()),
            )

            k = int(self.sb_blur.value())
            if k > 0 and cv2 is not None:
                mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(k))

        self._mask = np.clip(mask, 0.0, 1.0)
        self._update_preview_pixmap()

    def _on_use_imported_mask_toggled(self, on: bool):
        self._use_imported_mask = bool(on)
        # if we don't have an imported mask yet, turn it off again
        if self._use_imported_mask and self._imported_mask_full is None:
            self._use_imported_mask = False
            self.cb_use_imported.setChecked(False)
            QMessageBox.information(self, "No imported mask", "Pick a mask view first.")
            return

        # just rebuild preview with the external mask
        self._recompute_mask_and_preview()

    def _import_mask_from_view(self):
        if self.docman is None:
            QMessageBox.information(self, "No document manager", "Cannot import without a document manager.")
            return

        # get ALL docs user currently has open (renamed, FITS layers, XISF layers, duplicates, etc.)
        docs = self.docman.all_documents() or []
        # only image docs
        img_docs = [d for d in docs if hasattr(d, "image") and d.image is not None]

        if not img_docs:
            QMessageBox.information(self, "No views", "There are no image views to import a mask from.")
            return

        # build names as the user sees them
        items = []
        for d in img_docs:
            try:
                nm = d.display_name()
            except Exception:
                nm = "Untitled"
            items.append(nm)

        from PyQt6.QtWidgets import QInputDialog
        choice, ok = QInputDialog.getItem(
            self,
            "Pick mask view",
            "Open image views:",
            items,
            0,
            False
        )
        if not ok:
            return

        # find selected document
        sel_doc = None
        for d, nm in zip(img_docs, items):
            if nm == choice:
                sel_doc = d
                break

        if sel_doc is None or getattr(sel_doc, "image", None) is None:
            QMessageBox.warning(self, "Import failed", "Selected view has no image.")
            return

        mask_img = np.clip(sel_doc.image.astype(np.float32), 0.0, 1.0)

        # if it's RGB, take channel 0 — that’s how your exported mask would look (3 equal channels)
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]

        # resize to current image size if needed
        dst_h, dst_w = self.img.shape[:2]
        src_h, src_w = mask_img.shape[:2]
        if (src_h, src_w) != (dst_h, dst_w):
            if cv2 is not None:
                mask_full = cv2.resize(mask_img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
            else:
                yy = (np.linspace(0, src_h - 1, dst_h)).astype(int)
                xx = (np.linspace(0, src_w - 1, dst_w)).astype(int)
                mask_full = mask_img[yy[:, None], xx[None, :]]
        else:
            mask_full = mask_img

        mask_full = np.clip(mask_full.astype(np.float32), 0.0, 1.0)

        # store
        self._imported_mask_full = mask_full
        self._imported_mask_name = choice
        self.lbl_imported_mask.setText(f"Imported: {choice}")

        # auto-enable
        self.cb_use_imported.setChecked(True)
        self._use_imported_mask = True

        # refresh preview
        self._recompute_mask_and_preview()


    def _overlay_mask(self, base: np.ndarray, mask: np.ndarray) -> np.ndarray:
        base = _ensure_rgb01(base)
        # mask is HxW -> expand to HxWx1 for broadcasting
        alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None] * 0.6
        # red overlay
        overlay = base.copy()
        overlay[..., 0] = np.clip(base[..., 0]*(1 - alpha[..., 0]) + 1.0*alpha[..., 0], 0.0, 1.0)
        overlay[..., 1] = np.clip(base[..., 1]*(1 - alpha[..., 0]) + 0.0*alpha[..., 0], 0.0, 1.0)
        overlay[..., 2] = np.clip(base[..., 2]*(1 - alpha[..., 0]) + 0.0*alpha[..., 0], 0.0, 1.0)
        return overlay

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
            )

            out = _ensure_rgb01(out)
        else:
            out = _ensure_rgb01(base)

        if self.cb_show_mask.isChecked():
            # fade overlay by intensity too
            mask_vis = mask * float(self.ds_int.value())
            show = self._overlay_mask(out, mask_vis)
        else:
            show = out

        pm = _to_pixmap(show)
        h, w = show.shape[:2]
        zw, zh = max(1, int(round(w * self._zoom))), max(1, int(round(h * self._zoom)))
        pm_scaled = pm.scaled(zw, zh, Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview.setPixmap(pm_scaled)
        self.lbl_preview.resize(zw, zh)


    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        QTimer.singleShot(0, self._update_preview_pixmap)

    # ------------- Apply -------------
    def _apply_fullres(self) -> np.ndarray:
        base = self.img

        if self._use_imported_mask and self._imported_mask_full is not None:
            mask = np.clip(self._imported_mask_full.astype(np.float32), 0.0, 1.0)
        else:
            mask = self._build_mask(base)

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
        )

        return out

    def _export_mask_doc(self):
        if self.docman is None:
            QMessageBox.information(self, "No document manager", "Cannot export mask without a document manager.")
            return

        base = self.img
        if base is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            return

        mask = self._build_mask(base)          # H x W, float32, 0..1
        mask_rgb = np.repeat(mask[..., None], 3, axis=2).astype(np.float32)

        name = getattr(self.document, "display_name", lambda: "Image")()
        title = f"{name} [SelectiveColor MASK]"
        try:
            self.docman.open_array(mask_rgb, title=title)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))


    def _build_mask(self, base: np.ndarray) -> np.ndarray:
        """
        Build the full-res mask using the *current UI settings*.
        This is exactly what your old _apply_fullres did, just pulled out.
        """
        preset = self.dd_preset.currentText()
        ranges = (
            [(float(self.sp_h1.value()), float(self.sp_h2.value()))]
            if preset == "Custom"
            else _PRESETS[preset]
        )

        # 1) hue / chroma / light / smooth / invert
        mask = _hue_mask(
            base,
            ranges_deg=ranges,
            min_chroma=float(self.ds_minC.value()),
            min_light=float(self.ds_minL.value()),
            max_light=float(self.ds_maxL.value()),
            smooth_deg=float(self.ds_smooth.value()),
            invert_range=self.cb_invert.isChecked(),
        )

        # 2) shadows / highlights weighting
        mask = _weight_shadows_highlights(
            mask, base,
            shadows=float(self.ds_sh.value()),
            highlights=float(self.ds_hi.value()),
            balance=float(self.ds_bal.value()),
        )

        # 3) optional blur
        k = int(self.sb_blur.value())
        if k > 0 and cv2 is not None:
            mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(k))

        return np.clip(mask, 0.0, 1.0).astype(np.float32)



    def _apply_to_document(self):
        try:
            result = self._apply_fullres()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e)); return

        # write back to the same document (preferred)
        try:
            if hasattr(self.document, "set_image"):
                self.document.set_image(result)
        except Exception:
            # fallback: if set_image fails, at least open it as a new view (but keep dialog open)
            name = getattr(self.document, "display_name", lambda: "Image")()
            if hasattr(self.docman, "open_array"):
                self.docman.open_array(result, title=f"{name} [SelectiveColor]")

        # make the processed image the new working base for further tweaks
        self.img = np.clip(result.astype(np.float32), 0.0, 1.0)
        self._last_base = None   # force rebuild from current self.img
        self._reset_controls()   # reset knobs; dialog remains open


    def _apply_as_new_doc(self):
        try:
            result = self._apply_fullres()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e)); return

        name = getattr(self.document, "display_name", lambda: "Image")()
        new_doc = None
        if hasattr(self.docman, "open_array"):
            new_doc = self.docman.open_array(result, title=f"{name} [SelectiveColor]")

        # continue editing the new doc if we got a handle; otherwise just keep editing current
        if new_doc is not None:
            self.document = new_doc
            # refresh label
            try:
                disp = getattr(self.document, "display_name", lambda: "Image")()
            except Exception:
                disp = "Image"
            self.lbl_target.setText(f"Target View:  <b>{disp}</b>")

        # new working base is the processed pixels either way
        self.img = np.clip(result.astype(np.float32), 0.0, 1.0)
        self._last_base = None
        self._reset_controls()

