# src/setiastro/saspro/histogram_transform_pro.py
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSettings, QByteArray, QEvent
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
from PyQt6.QtWidgets import (
    QDialog, QWidget, QLabel, QPushButton, QCheckBox, QDoubleSpinBox, QSlider,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QScrollArea, QMessageBox,
    QSizePolicy, QComboBox
)

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


# ---------------- math ----------------

def _ensure_rgb01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img).astype(np.float32, copy=False)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return np.clip(a, 0.0, 1.0)

def _luma01(rgb01: np.ndarray) -> np.ndarray:
    if rgb01.ndim == 2:
        return np.clip(rgb01, 0, 1).astype(np.float32)
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (0.2126*r + 0.7152*g + 0.0722*b).astype(np.float32)

def _is_mono(img: np.ndarray) -> bool:
    a = np.asarray(img)
    return (a.ndim == 2) or (a.ndim == 3 and a.shape[2] == 1)

def _ensure_mono01(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img).astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return np.clip(a, 0.0, 1.0)

def _ensure_rgb01_keep(img: np.ndarray) -> np.ndarray:
    """
    Like _ensure_rgb01 but DOES NOT expand mono -> 3ch.
    Returns either HxW (mono) or HxWx3 (RGB).
    """
    a = np.asarray(img).astype(np.float32, copy=False)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim == 2:
        return np.clip(a, 0.0, 1.0)
    if a.ndim == 3 and a.shape[2] >= 3:
        return np.clip(a[..., :3], 0.0, 1.0)
    # fallback
    return np.clip(a, 0.0, 1.0)

def _mtf_vect(x: np.ndarray, m: float) -> np.ndarray:
    """
    Midtones transfer function M(x;m), vectorized.
    Special cases:
      M(0)=0, M(m)=0.5, M(1)=1
      otherwise: ((m-1)*x) / ((2m-1)*x - m)
    """
    x = np.asarray(x, dtype=np.float32)
    m = float(m)

    # clamp domain
    x = np.clip(x, 0.0, 1.0)

    # handle degenerate m
    if m <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    if m >= 1.0:
        return np.ones_like(x, dtype=np.float32)

    out = np.empty_like(x, dtype=np.float32)

    # endpoints
    out[x <= 0.0] = 0.0
    out[x >= 1.0] = 1.0

    # general formula
    mid = (x > 0.0) & (x < 1.0)
    xm = x[mid]
    denom = ((2.0*m - 1.0) * xm - m)

    # safe division
    num = (m - 1.0) * xm
    y = np.zeros_like(xm, dtype=np.float32)
    ok = np.abs(denom) > 1e-10
    y[ok] = num[ok] / denom[ok]
    y = np.clip(y, 0.0, 1.0)

    out[mid] = y

    # force exact at x==m (floating compare with tolerance)
    if np.isfinite(m):
        out[np.isclose(x, m, atol=1e-6)] = 0.5

    return out

def apply_histogram_transform(img01: np.ndarray, black: float, mid: float, white: float) -> np.ndarray:
    """
    Linked histogram transform:
      1) normalize by black/white: t = clamp((x - b)/(w - b), 0..1)
      2) apply PI MTF: y = M(t; mid)
    Applies same b/m/w to all channels.
    """
    a = _ensure_rgb01(img01)
    b = float(black)
    w = float(white)
    m = float(mid)

    # enforce sane
    if w <= b + 1e-8:
        w = b + 1e-8

    t = (a - b) / (w - b)
    t = np.clip(t, 0.0, 1.0)

    # apply per-channel
    out = np.empty_like(t, dtype=np.float32)
    out[..., 0] = _mtf_vect(t[..., 0], m)
    out[..., 1] = _mtf_vect(t[..., 1], m)
    out[..., 2] = _mtf_vect(t[..., 2], m)
    return np.clip(out, 0.0, 1.0)

def clipping_stats_channel(img01: np.ndarray, black: float, white: float, channel: str) -> tuple[int, int, int]:
    """
    Returns (n_low, n_high, n_total) for selected channel.
    Counts pixels that would be clipped by black/white.
    """
    a = _ensure_rgb01(img01)
    b = float(black)
    w = float(white)
    if w <= b + 1e-8:
        w = b + 1e-8

    ch = str(channel).upper().strip()

    if ch == "L":
        v = _rgb_to_luma(a)
    elif ch == "K":
        # linked RGB brightness mode: use luminance proxy for stats display
        v = _rgb_to_luma(a)
    else:
        idx = {"R": 0, "G": 1, "B": 2}.get(ch, 0)
        v = a[..., idx]

    v = np.asarray(v, dtype=np.float32)
    n_total = int(v.size)
    n_low  = int(np.count_nonzero(v <= b))
    n_high = int(np.count_nonzero(v >= w))
    return n_low, n_high, n_total

def apply_histogram_transform_channel(img01: np.ndarray, black: float, mid: float, white: float, channel: str) -> np.ndarray:
    """
    Shape-preserving histogram transform.
    - If input is mono (HxW or HxWx1): output stays mono (HxW).
    - If input is RGB (HxWx3): output stays RGB (HxWx3).

    channel: 'L','R','G','B'
      - mono input:
          * L/R/G/B all behave the same (operate on the mono plane)
      - RGB input:
          * L: apply to luma then recombine (preserve chroma)
          * R/G/B: apply only that channel
    """
    b = float(black); w = float(white); m = float(mid)
    if w <= b + 1e-8:
        w = b + 1e-8
    ch = str(channel).upper().strip()

    # --- MONO PATH (preserve mono) ---
    if _is_mono(img01):
        x = _ensure_mono01(img01)
        t = np.clip((x - b) / (w - b), 0.0, 1.0)
        y = _mtf_vect(t, m)
        return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)

    # --- RGB PATH ---
    a = _ensure_rgb01(img01)

    if ch == "L":
        Y = _rgb_to_luma(a)
        t = np.clip((Y - b) / (w - b), 0.0, 1.0)
        Y2 = _mtf_vect(t, m)
        return _recombine_luma_into_rgb(Y2, a)

    if ch == "K":
        # linked RGB brightness mode: apply same levels curve to each channel directly
        return apply_histogram_transform(a, b, m, w).astype(np.float32, copy=False)

    idx = {"R": 0, "G": 1, "B": 2}.get(ch, 0)
    out = a.copy()
    t = np.clip((out[..., idx] - b) / (w - b), 0.0, 1.0)
    out[..., idx] = _mtf_vect(t, m)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

def _rgb_to_luma(rgb01: np.ndarray) -> np.ndarray:
    a = _ensure_rgb01(rgb01)
    return (0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]).astype(np.float32)

def _recombine_luma_into_rgb(Y: np.ndarray, RGB: np.ndarray) -> np.ndarray:
    rgb = _ensure_rgb01(RGB)
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    orig_Y = (rgb[...,0]*w[0] + rgb[...,1]*w[1] + rgb[...,2]*w[2]).astype(np.float32)
    chroma = rgb / (orig_Y[...,None] + 1e-6)
    return np.clip(chroma * Y[...,None], 0.0, 1.0)

# ---------------- small Qt helpers ----------------

def _to_qimage_rgb(img01: np.ndarray) -> QImage:
    a = np.clip(img01, 0.0, 1.0)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    u8 = (a * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = u8.shape
    q = QImage(u8.data, w, h, u8.strides[0], QImage.Format.Format_RGB888)
    return q.copy()

def _to_pixmap(img01: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(_to_qimage_rgb(img01))


# ---------------- widgets ----------------
class CurveWidget(QWidget):
    """Draw the histogram transform curve defined by black/mid/white — with draggable handles."""
    from PyQt6.QtCore import pyqtSignal
    paramsChanged = pyqtSignal(float, float, float)  # black, mid, white

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.black = 0.0
        self.mid   = 0.5
        self.white = 1.0
        self._dragging = None   # "black" | "mid" | "white" | None
        self.setMouseTracking(True)

    def set_params(self, black: float, mid: float, white: float, notify=False):
        self.black = float(black)
        self.mid   = float(mid)
        self.white = float(white)
        self.update()
        if notify:
            self.paramsChanged.emit(self.black, self.mid, self.white)

    def _plot_rect(self):
        return self.rect().adjusted(10, 10, -10, -10)

    def _x_to_val(self, px: int) -> float:
        r = self._plot_rect()
        return float(np.clip((px - r.left()) / max(r.width(), 1), 0.0, 1.0))

    def _val_to_x(self, val: float) -> int:
        r = self._plot_rect()
        return r.left() + int(np.clip(val, 0.0, 1.0) * r.width())

    def _handle_at(self, x: int, y: int) -> str | None:
        """Return which handle (if any) is close to (x, y)."""
        r = self._plot_rect()
        tol = 8
        for name, val in (("black", self.black), ("mid", self.mid), ("white", self.white)):
            hx = self._val_to_x(val)
            # handle is at the bottom tick mark
            hy = r.bottom() - 5
            if abs(x - hx) <= tol and abs(y - hy) <= tol + 10:
                return name
        return None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            hit = self._handle_at(int(ev.position().x()), int(ev.position().y()))
            if hit:
                self._dragging = hit
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        x = int(ev.position().x())
        y = int(ev.position().y())

        if self._dragging:
            val = self._x_to_val(x)
            if self._dragging == "black":
                self.black = min(val, self.white - 0.001)
            elif self._dragging == "white":
                self.white = max(val, self.black + 0.001)
            elif self._dragging == "mid":
                self.mid = float(np.clip(val, 0.0, 1.0))
            self.update()
            self.paramsChanged.emit(self.black, self.mid, self.white)
            ev.accept()
            return

        # cursor hint
        hit = self._handle_at(x, y)
        self.setCursor(Qt.CursorShape.SizeHorCursor if hit else Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = None
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 25))

        r = self._plot_rect()
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.drawRect(r)

        # grid
        p.setPen(QPen(QColor(45, 45, 45), 1))
        for k in range(1, 4):
            x = r.left() + int(k * r.width() / 4)
            y = r.top() + int(k * r.height() / 4)
            p.drawLine(x, r.top(), x, r.bottom())
            p.drawLine(r.left(), y, r.right(), y)

        # curve
        xs = np.linspace(0.0, 1.0, 512, dtype=np.float32)
        b = self.black; w = self.white
        if w <= b + 1e-8: w = b + 1e-8
        t = np.clip((xs - b) / (w - b), 0.0, 1.0)
        ys = _mtf_vect(t, self.mid)

        p.setPen(QPen(QColor(220, 220, 220), 2))
        last = None
        for x, y in zip(xs, ys):
            px = r.left() + int(x * r.width())
            py = r.bottom() - int(y * r.height())
            if last is not None:
                p.drawLine(last[0], last[1], px, py)
            last = (px, py)

        # handle definitions: (name, value, color, label)
        handles = [
            ("black", self.black, QColor(100, 160, 255), "B"),
            ("mid",   self.mid,   QColor(100, 220, 100), "M"),
            ("white", self.white, QColor(255, 180,  80), "W"),
        ]

        for name, val, color, label in handles:
            hx = self._val_to_x(val)
            is_dragging = (self._dragging == name)

            # vertical dashed line from top to bottom
            pen = QPen(color, 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(hx, r.top(), hx, r.bottom())

            # filled triangle handle at the bottom
            from PyQt6.QtGui import QPolygon, QBrush
            from PyQt6.QtCore import QPoint
            tri_h = 9 if is_dragging else 7
            tri_w = 6 if is_dragging else 5
            tri = QPolygon([
                QPoint(hx,          r.bottom() + 2),
                QPoint(hx - tri_w,  r.bottom() + 2 + tri_h),
                QPoint(hx + tri_w,  r.bottom() + 2 + tri_h),
            ])
            p.setBrush(QBrush(color))
            p.setPen(QPen(color.darker(140), 1))
            p.drawPolygon(tri)

            # small label below the triangle
            p.setPen(QPen(color, 1))
            p.setFont(self.font())
            p.drawText(hx - 4, r.bottom() + 2 + tri_h + 11, label)

        p.end()


class HistogramWidget(QWidget):
    """RGB histogram: original vs preview (R/G/B) + optional luma overlay."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._orig = None  # dict: {"R":h, "G":h, "B":h, "L":h}
        self._prev = None

    def set_histograms(self, orig: dict | None, prev: dict | None):
        self._orig = orig
        self._prev = prev
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(20, 20, 20))
        r = self.rect().adjusted(10, 10, -10, -10)
        p.setPen(QPen(QColor(70, 70, 70), 1))
        p.drawRect(r)

        def _norm(h):
            if h is None or getattr(h, "size", 0) == 0:
                return None
            h = h.astype(np.float32, copy=False)
            mx = float(h.max()) if float(h.max()) > 0 else 1.0
            return h / mx

        def draw(h, color: QColor, width: int):
            h = _norm(h)
            if h is None:
                return
            p.setPen(QPen(color, width))
            n = int(h.shape[0])
            last = None
            for i in range(n):
                x = r.left() + int(i * r.width() / max(1, n - 1))
                y = r.bottom() - int(h[i] * r.height())
                if last is not None:
                    p.drawLine(last[0], last[1], x, y)
                last = (x, y)

        # Original: thinner/dimmer
        if isinstance(self._orig, dict):
            draw(self._orig.get("L"), QColor(120, 120, 120), 1)
            draw(self._orig.get("R"), QColor(160, 80, 80), 1)
            draw(self._orig.get("G"), QColor(80, 160, 80), 1)
            draw(self._orig.get("B"), QColor(80, 120, 200), 1)

        # Preview: thicker/brighter
        if isinstance(self._prev, dict):
            draw(self._prev.get("L"), QColor(190, 190, 190), 2)
            draw(self._prev.get("R"), QColor(255, 90, 90), 2)
            draw(self._prev.get("G"), QColor(90, 255, 120), 2)
            draw(self._prev.get("B"), QColor(100, 160, 255), 2)

        p.end()


# ---------------- dialog ----------------

class HistogramTransformDialogPro(QDialog):
    """
    Old-school histogram transform (black / midtones / white) with PI MTF.
    Left: curve + histogram. Right: preview.
    """
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Levels (Histogram Transform)"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        self.document = document
        if self.document is None or getattr(self.document, "image", None) is None:
            QMessageBox.information(self, "No image", "Open an image first.")
            self.close()
            return

        self._settings = QSettings()
        self._persist_prefix = "histogram_transform"
        self._geom_restored = False
        self._restoring_ui = True

        # --- performance knobs ---
        self._preview_max_dim = 1400       # preview downsample cap (fast)
        self._hist_max_pixels = 600_000    # histogram sample cap (fast)

        # --- base image + caches ---
        self._img = np.clip(np.asarray(self.document.image, dtype=np.float32), 0.0, 1.0)
        self._base = _ensure_rgb01(self._img)

        # cached preview base + original histogram (computed ONCE)
        self._preview_base = self._make_preview_base(self._base)
        self._h0_rgb = self._hist_rgb_256(self._preview_base)
        self._h0 = self._hist_luma_256(self._preview_base)

        self._out = self._preview_base.copy()

        # preview state
        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 16.0
        self._panning = False
        self._pan_last = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)   # feels snappy but avoids spam
        self._debounce.timeout.connect(self._recompute)

        self._build_ui()
        self._update_channel_ui_for_image()
        # IMPORTANT: do not restore black/mid/white. Always start defaults.
        # But we do restore geometry + live checkbox.
        self._recompute()

    def _k(self, key: str) -> str:
        return f"{self._persist_prefix}/{key}"
        
    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        # LEFT: controls + curve + hist
        left = QVBoxLayout()
        left_host = QWidget(self)
        left_host.setLayout(left)
        left_host.setFixedWidth(360)

        # controls group
        gb = QGroupBox("Histogram Transform", self)
        gl = QGridLayout(gb)

        self.cb_live = QCheckBox("Live preview", self)
        self.cb_live.setChecked(True)
        gl.addWidget(self.cb_live, 0, 0, 1, 2)
        gl.addWidget(QLabel("Channel:"), 1, 0)
        self.cb_channel = QComboBox(self)
        self.cb_channel.addItem("L (Luminance)", "L")
        self.cb_channel.addItem("K (Brightness / RGB linked)", "K")
        self.cb_channel.addItem("R", "R")
        self.cb_channel.addItem("G", "G")
        self.cb_channel.addItem("B", "B")
        idx_k = self.cb_channel.findData("K")
        if idx_k >= 0:
            self.cb_channel.setCurrentIndex(idx_k)

        gl.addWidget(self.cb_channel, 1, 1, 1, 2)

        self.cb_channel.currentIndexChanged.connect(self._schedule)
        # black / mid / white controls
        self.sp_black = QDoubleSpinBox(self); self.sp_black.setRange(0.0, 1.0); self.sp_black.setDecimals(5); self.sp_black.setSingleStep(0.001); self.sp_black.setValue(0.0)
        self.sp_mid   = QDoubleSpinBox(self); self.sp_mid.setRange(0.0, 1.0);   self.sp_mid.setDecimals(5);   self.sp_mid.setSingleStep(0.001);   self.sp_mid.setValue(0.5)
        self.sp_white = QDoubleSpinBox(self); self.sp_white.setRange(0.0, 1.0); self.sp_white.setDecimals(5); self.sp_white.setSingleStep(0.001); self.sp_white.setValue(1.0)

        self.sl_black = QSlider(Qt.Orientation.Horizontal, self); self.sl_black.setRange(0, 100000); self.sl_black.setValue(0)
        self.sl_mid   = QSlider(Qt.Orientation.Horizontal, self); self.sl_mid.setRange(0, 100000);   self.sl_mid.setValue(50000)
        self.sl_white = QSlider(Qt.Orientation.Horizontal, self); self.sl_white.setRange(0, 100000); self.sl_white.setValue(100000)

        def bind(sl: QSlider, sp: QDoubleSpinBox):
            sl.valueChanged.connect(lambda v: (sp.blockSignals(True), sp.setValue(v / 100000.0), sp.blockSignals(False)))
            sp.valueChanged.connect(lambda x: (sl.blockSignals(True), sl.setValue(int(round(float(x) * 100000.0))), sl.blockSignals(False)))

        bind(self.sl_black, self.sp_black)
        bind(self.sl_mid, self.sp_mid)
        bind(self.sl_white, self.sp_white)
        # wiring - IMPORTANT: sliders must schedule too because spinbox signals are blocked during slider drag
        for w in (self.sp_black, self.sp_mid, self.sp_white):
            w.valueChanged.connect(self._schedule)

        for s in (self.sl_black, self.sl_mid, self.sl_white):
            s.valueChanged.connect(self._schedule)       # live update (debounced)
            s.sliderReleased.connect(self._schedule)     # ensure final recompute when user lets go

        self.cb_live.toggled.connect(self._schedule)
        gl.addWidget(QLabel("Black point:"), 2, 0); gl.addWidget(self.sl_black, 2, 1); gl.addWidget(self.sp_black, 2, 2)
        gl.addWidget(QLabel("Midtones:"),    3, 0); gl.addWidget(self.sl_mid,   3, 1); gl.addWidget(self.sp_mid,   3, 2)
        gl.addWidget(QLabel("White point:"), 4, 0); gl.addWidget(self.sl_white, 4, 1); gl.addWidget(self.sp_white, 4, 2)

        self.lbl_clip = QLabel("Clipped: low 0 (0.00%) | high 0 (0.00%)", self)
        self.lbl_clip.setStyleSheet("color:#aaa;")
        gl.addWidget(self.lbl_clip, 5, 0, 1, 3)
        left.addWidget(gb)
        # min/max stats label
        self.lbl_minmax = QLabel("Min: —   Max: —", self)
        self.lbl_minmax.setStyleSheet("color:#aaa; font-size: 11px;")
        self.lbl_minmax.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self.lbl_minmax)

        # curve + histogram
        self.curve = CurveWidget(self)
        left.addWidget(self.curve, 0)

        self.hist = HistogramWidget(self)
        left.addWidget(self.hist, 0)

        # buttons
        brow = QHBoxLayout()
        self.btn_apply = QPushButton("Apply", self)
        self.btn_new = QPushButton("Apply as New", self)
        self.btn_reset = QPushButton("Reset", self)
        brow.addWidget(self.btn_apply)
        brow.addWidget(self.btn_new)
        brow.addWidget(self.btn_reset)
        left.addLayout(brow)

        left.addStretch(1)
        root.addWidget(left_host, 0)

        # RIGHT: preview w/ zoom
        right = QVBoxLayout()

        zrow = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_fit      = themed_toolbtn("zoom-fit-best", "Fit")
        zrow.addWidget(self.btn_zoom_out)
        zrow.addWidget(self.btn_zoom_in)
        zrow.addWidget(self.btn_fit)
        zrow.addStretch(1)
        right.addLayout(zrow)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.lbl_preview)
        right.addWidget(self.scroll, 1)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setStyleSheet("color:#888;")
        right.addWidget(self.lbl_status, 0)

        right_host = QWidget(self)
        right_host.setLayout(right)
        root.addWidget(right_host, 1)

        # wiring
        for w in (self.sp_black, self.sp_mid, self.sp_white):
            w.valueChanged.connect(self._schedule)
        self.cb_live.toggled.connect(self._schedule)

        self.btn_zoom_in.clicked.connect(lambda: self._zoom_at(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_at(0.8))
        self.btn_fit.clicked.connect(self._fit_to_preview)

        # panning
        self.scroll.viewport().installEventFilter(self)

        # actions
        self.btn_apply.clicked.connect(self._apply_inplace)
        self.btn_new.clicked.connect(self._apply_new)
        self.btn_reset.clicked.connect(self._reset)
        self.curve.paramsChanged.connect(self._on_curve_drag)
        # initial
        self.resize(1100, 720)

    def _on_curve_drag(self, black: float, mid: float, white: float):
        """Called when user drags a handle on the CurveWidget."""
        for w in (self.sp_black, self.sp_mid, self.sp_white,
                  self.sl_black, self.sl_mid, self.sl_white):
            w.blockSignals(True)
        try:
            self.sp_black.setValue(black)
            self.sp_mid.setValue(mid)
            self.sp_white.setValue(white)
            self.sl_black.setValue(int(round(black * 100000)))
            self.sl_mid.setValue(int(round(mid   * 100000)))
            self.sl_white.setValue(int(round(white * 100000)))
        finally:
            for w in (self.sp_black, self.sp_mid, self.sp_white,
                      self.sl_black, self.sl_mid, self.sl_white):
                w.blockSignals(False)
        self._schedule()

    def _update_channel_ui_for_image(self):
        """
        Disable channel selection for mono sources (HxW or HxWx1).
        Forces channel to L and prevents choosing R/G/B.
        """
        try:
            # NOTE: use the ORIGINAL doc image to decide mono vs RGB
            src = getattr(self.document, "image", None)
            is_mono = _is_mono(src) if src is not None else False
        except Exception:
            is_mono = False

        try:
            # If mono: force L and disable dropdown
            if is_mono:
                self.cb_channel.blockSignals(True)
                try:
                    idx = self.cb_channel.findData("L")
                    if idx >= 0:
                        self.cb_channel.setCurrentIndex(idx)
                finally:
                    self.cb_channel.blockSignals(False)

                self.cb_channel.setEnabled(False)
                self.cb_channel.setToolTip("Mono image: channel selection disabled (L only).")
            else:
                self.cb_channel.setEnabled(True)
                self.cb_channel.setToolTip("")
        except Exception:
            pass

    # ---------- persistence ----------
    def showEvent(self, e):
        super().showEvent(e)
        if not self._geom_restored:
            self._geom_restored = True
            QTimer.singleShot(0, self._restore_ui_state)

    def closeEvent(self, e):
        try:
            self._save_ui_state()
        except Exception:
            pass
        super().closeEvent(e)

    def _save_ui_state(self):
        if getattr(self, "_restoring_ui", False) or not getattr(self, "_geom_restored", False):
            return
        s = self._settings
        s.setValue(self._k("window_geometry"), self.saveGeometry())
        s.setValue(self._k("live"), bool(self.cb_live.isChecked()))
        try:
            s.sync()
        except Exception:
            pass

    def _restore_ui_state(self):
        self._restoring_ui = True
        try:
            s = self._settings

            g = s.value(self._k("window_geometry"), None)
            if g is not None:
                self.restoreGeometry(g)

            live = bool(s.value(self._k("live"), True, type=bool))

            # reset points ALWAYS (no persistence)
            for ww in (self.sp_black, self.sp_mid, self.sp_white, self.cb_live, self.cb_channel):
                ww.blockSignals(True)
            try:
                self.sp_black.setValue(0.0)
                self.sp_mid.setValue(0.5)
                self.sp_white.setValue(1.0)

                self.sl_black.setValue(0)
                self.sl_mid.setValue(50000)
                self.sl_white.setValue(100000)

                self.cb_live.setChecked(bool(live))

                if _is_mono(getattr(self.document, "image", None)):
                    idx = self.cb_channel.findData("L")
                else:
                    idx = self.cb_channel.findData("K")
                if idx >= 0:
                    self.cb_channel.setCurrentIndex(idx)

            finally:
                for ww in (self.sp_black, self.sp_mid, self.sp_white, self.cb_live, self.cb_channel):
                    ww.blockSignals(False)
        finally:
            self._restoring_ui = False
            self._update_channel_ui_for_image()
            self._schedule()

    def _make_preview_base(self, rgb01: np.ndarray) -> np.ndarray:
        """
        Downsample (by striding) the base image for realtime preview/histogram.
        Keeps aspect ratio, no cv2 required, very fast.
        """
        a = _ensure_rgb01(rgb01)
        h, w = a.shape[:2]
        max_dim = int(getattr(self, "_preview_max_dim", 1400))
        if max(h, w) <= max_dim:
            return a

        step = int(np.ceil(max(h, w) / max_dim))
        step = max(1, step)
        return a[::step, ::step, :]

    def _hist_rgb_256(self, rgb01: np.ndarray) -> dict:
        """
        Fast 256-bin histograms for R/G/B + L on strided sampling.
        Returns dict {"R","G","B","L"}.
        """
        a = _ensure_rgb01(rgb01)
        h, w = a.shape[:2]

        max_px = int(getattr(self, "_hist_max_pixels", 600_000))
        npx = h * w
        if npx > max_px:
            step = int(np.ceil(np.sqrt(npx / max_px)))
            step = max(1, step)
            a = a[::step, ::step, :]

        out = {}
        for key, idx in (("R", 0), ("G", 1), ("B", 2)):
            hist, _ = np.histogram(a[..., idx].ravel(), bins=256, range=(0.0, 1.0))
            out[key] = hist.astype(np.int64, copy=False)

        Y = _rgb_to_luma(a)
        hist, _ = np.histogram(Y.ravel(), bins=256, range=(0.0, 1.0))
        out["L"] = hist.astype(np.int64, copy=False)
        return out

    def _hist_luma_256(self, rgb01: np.ndarray) -> np.ndarray:
        """
        Fast 256-bin histogram on luma using strided sampling.
        """
        a = _ensure_rgb01(rgb01)
        h, w = a.shape[:2]

        # Additional stride for histogram only (cheap)
        max_px = int(getattr(self, "_hist_max_pixels", 600_000))  # ~0.6MP
        npx = h * w
        if npx > max_px:
            step = int(np.ceil(np.sqrt(npx / max_px)))
            step = max(1, step)
            a = a[::step, ::step, :]

        l = _luma01(a)
        # np.histogram on a smaller array is fast enough
        hist, _ = np.histogram(l.ravel(), bins=256, range=(0.0, 1.0))
        return hist.astype(np.int64, copy=False)




    def _reload_base_from_document(self):
        img = np.clip(np.asarray(self.document.image, dtype=np.float32), 0.0, 1.0)
        self._img = img
        self._base = _ensure_rgb01(img)

        self._preview_base = self._make_preview_base(self._base)

        # Cache ORIGINAL RGB histograms once
        self._h0_rgb = self._hist_rgb_256(self._preview_base)

    # ---------- core ----------
    def _schedule(self, *_):
        if getattr(self, "_restoring_ui", False):
            return
        self._debounce.stop()
        self._debounce.start()

    def _sanitize_points(self) -> tuple[float, float, float]:
        b = float(self.sp_black.value())
        m = float(self.sp_mid.value())
        w = float(self.sp_white.value())

        # ensure ordering b < w; keep m inside [0,1] but doesn't need to be between b/w
        if w <= b + 1e-6:
            w = min(1.0, b + 1e-6)
            self.sp_white.blockSignals(True)
            self.sp_white.setValue(w)
            self.sp_white.blockSignals(False)
        m = np.clip(m, 0.0, 1.0)
        return b, m, w

    def _active_mask_array(self) -> np.ndarray | None:
        """Return active mask as float32 [H,W] in 0..1, resized to doc image."""
        try:
            mid = getattr(self.document, "active_mask_id", None)
            if not mid:
                return None
            layer = getattr(self.document, "masks", {}).get(mid)
            if layer is None:
                return None
 
            m = np.asarray(getattr(layer, "data", None))
            if m is None or m.size == 0:
                return None
 
            # squeeze to 2D
            if m.ndim == 3 and m.shape[2] == 1:
                m = m[..., 0]
            elif m.ndim == 3:
                m = (0.2126 * m[..., 0] + 0.7152 * m[..., 1] + 0.0722 * m[..., 2])
 
            if m.dtype.kind in "ui":
                m = m.astype(np.float32) / float(np.iinfo(m.dtype).max)
            else:
                m = m.astype(np.float32, copy=False)
 
            m = np.clip(m, 0.0, 1.0)
 
            th, tw = self.document.image.shape[:2]
            sh, sw = m.shape[:2]
            if (sh, sw) != (th, tw):
                yi = np.linspace(0, sh - 1, th).astype(np.int32)
                xi = np.linspace(0, sw - 1, tw).astype(np.int32)
                m = m[yi][:, xi]
 
            opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
            if opacity < 1.0:
                m = m * opacity
 
            return m
        except Exception:
            return None
 
    def _blend_with_mask(self, base: np.ndarray, out: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
        """Blend base and out using mask [H,W] in 0..1."""
        if out.ndim == 3:
            m = mask[..., None]
        else:
            m = mask
        return (base * (1.0 - m) + out * m).astype(np.float32, copy=False)
 
    def _resize_mask_to(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Nearest-neighbour resize of a 2-D mask to (h, w) — no cv2 needed."""
        mh, mw = mask.shape[:2]
        if (mh, mw) == (h, w):
            return mask
        yi = np.linspace(0, mh - 1, h).astype(np.int32)
        xi = np.linspace(0, mw - 1, w).astype(np.int32)
        return mask[yi][:, xi]

    def _recompute(self):
        b, m, w = self._sanitize_points()

        # update min/max stats label
        try:
            img_flat = self._preview_base.ravel()
            mn = float(img_flat.min())
            mx = float(img_flat.max())
            self.lbl_minmax.setText(
                f"Min: {mn:.5f}  ({int(mn*65535)})   "
                f"Max: {mx:.5f}  ({int(mx*65535)})"
            )
        except Exception:
            pass

        self.curve.set_params(b, m, w)
 
        chan = "L"
        try:
            chan = str(self.cb_channel.currentData() or "L")
        except Exception:
            chan = "L"
 
        if self.cb_live.isChecked():
            out = apply_histogram_transform_channel(self._preview_base, b, m, w, chan)
 
            # --- mask blend (preview) ---
            mask_full = self._active_mask_array()
            if mask_full is not None:
                ph, pw = self._preview_base.shape[:2]
                mask_prev = self._resize_mask_to(mask_full, ph, pw)
                out = self._blend_with_mask(self._preview_base, out, mask_prev)
        else:
            out = self._preview_base
 
        self._out = out
 
        # clipping stats
        try:
            lo, hi, tot = clipping_stats_channel(self._preview_base, b, w, chan)
            plo = 100.0 * lo / max(1, tot)
            phi = 100.0 * hi / max(1, tot)
            clip_txt = f"Clipped: low {lo} ({plo:.2f}%) | high {hi} ({phi:.2f}%)"
            if self._active_mask_array() is not None:
                clip_txt += "  [mask active]"
            self.lbl_clip.setText(clip_txt)
        except Exception:
            pass
 
        h1_rgb = self._hist_rgb_256(self._out)
        self.hist.set_histograms(self._h0_rgb, h1_rgb)
 
        self._base_pm = _to_pixmap(self._out)
        self._apply_zoom()
 
        mask_note = "  [masked]" if self._active_mask_array() is not None else ""
        self.lbl_status.setText(
            f"Ch={chan}  Black={b:.5f}  Mid={m:.5f}  White={w:.5f}{mask_note}"
        )

    # ---------- preview zoom/pan ----------
    def _apply_zoom(self):
        if not hasattr(self, "_base_pm") or self._base_pm is None:
            return
        base_sz = self._base_pm.size()
        w = max(1, int(base_sz.width() * self._zoom))
        h = max(1, int(base_sz.height() * self._zoom))
        scaled = self._base_pm.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview.setPixmap(scaled)
        self.lbl_preview.resize(scaled.size())

    def _zoom_at(self, factor: float):
        self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * float(factor)))
        self._apply_zoom()

    def _fit_to_preview(self):
        if not hasattr(self, "_base_pm") or self._base_pm is None:
            return
        vp = self.scroll.viewport().size()
        pm = self._base_pm.size()
        if pm.width() <= 0 or pm.height() <= 0:
            return
        k = min(vp.width() / pm.width(), vp.height() / pm.height())
        self._zoom = max(self._min_zoom, min(self._max_zoom, float(k)))
        self._apply_zoom()

    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_last = ev.position().toPoint()
                obj.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
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
                obj.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                return True
        return super().eventFilter(obj, ev)

    # ---------- apply ----------
    def _apply_result_fullres(self) -> np.ndarray:
        b, m, w = self._sanitize_points()
        img = np.clip(np.asarray(self.document.image, dtype=np.float32), 0.0, 1.0)
 
        chan = "L"
        try:
            chan = str(self.cb_channel.currentData() or "L")
        except Exception:
            chan = "L"
 
        out = apply_histogram_transform_channel(img, b, m, w, chan)
        out = np.clip(out, 0.0, 1.0)
 
        # --- mask blend (full res) ---
        mask_full = self._active_mask_array()
        if mask_full is not None:
            out = self._blend_with_mask(img, out, mask_full)
 
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def _apply_inplace(self):
        try:
            self.lbl_status.setText("Processing full image…")
            self.btn_apply.setEnabled(False)
            self.btn_new.setEnabled(False)
            self.btn_reset.setEnabled(False)
            self.repaint()
            try:
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents()
            except Exception:
                pass

            result = self._apply_result_fullres()
        except Exception as e:
            self.btn_apply.setEnabled(True)
            self.btn_new.setEnabled(True)
            self.btn_reset.setEnabled(True)
            self.lbl_status.setText("")
            QMessageBox.critical(self, "Levels", str(e))
            return

        try:
            if hasattr(self.document, "apply_edit"):
                self.document.apply_edit(
                    result.astype(np.float32, copy=False),
                    metadata={"step_name": "Levels"},
                    step_name="Levels"
                )
            elif hasattr(self.document, "set_image"):
                self.document.set_image(result, step_name="Levels")
            else:
                self.document.image = result
        except Exception as e:
            self.btn_apply.setEnabled(True)
            self.btn_new.setEnabled(True)
            self.btn_reset.setEnabled(True)
            self.lbl_status.setText("")
            QMessageBox.critical(self, "Levels", f"Failed to apply:\n{e}")
            return

        # Reload base image + cached hist0, KEEP UI OPEN and KEEP slider values
        self._reload_base_from_document()

        # immediate refresh
        self._recompute()

        self.btn_apply.setEnabled(True)
        self.btn_new.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.lbl_status.setText("Finished and applied to the active document.")
        QTimer.singleShot(2000, self._recompute)

    def _apply_new(self):
        mw = self.parent()
        dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
        if dm is None:
            QMessageBox.warning(self, "Histogram Transform", "DocManager not available.")
            return

        try:
            result = self._apply_result_fullres()
        except Exception as e:
            QMessageBox.critical(self, "Histogram Transform", str(e))
            return

        title = "Histogram Transform"
        try:
            if hasattr(dm, "open_array"):
                doc = dm.open_array(result, metadata={"is_mono": False}, title=title)
            elif hasattr(dm, "create_document"):
                doc = dm.create_document(image=result, metadata={"is_mono": False}, name=title)
            else:
                raise RuntimeError("DocManager lacks open_array/create_document")

            if hasattr(mw, "_spawn_subwindow_for"):
                mw._spawn_subwindow_for(doc)
        except Exception as e:
            QMessageBox.critical(self, "Histogram Transform", f"Failed to open new view:\n{e}")

    def _reset(self):
        # reset numeric controls
        self.sp_black.blockSignals(True)
        self.sp_mid.blockSignals(True)
        self.sp_white.blockSignals(True)
        self.cb_channel.blockSignals(True)
        self.cb_live.blockSignals(True)

        try:
            self.sp_black.setValue(0.0)
            self.sp_mid.setValue(0.5)
            self.sp_white.setValue(1.0)

            # keep sliders in sync explicitly
            self.sl_black.setValue(0)
            self.sl_mid.setValue(50000)
            self.sl_white.setValue(100000)

            self.cb_live.setChecked(True)

            # mono stays L, color defaults to K
            if _is_mono(getattr(self.document, "image", None)):
                idx = self.cb_channel.findData("L")
            else:
                idx = self.cb_channel.findData("K")

            if idx >= 0:
                self.cb_channel.setCurrentIndex(idx)

        finally:
            self.sp_black.blockSignals(False)
            self.sp_mid.blockSignals(False)
            self.sp_white.blockSignals(False)
            self.cb_channel.blockSignals(False)
            self.cb_live.blockSignals(False)

        self._update_channel_ui_for_image()
        self._schedule()

#   ---   headless operations ----
def levels_array(
    img: np.ndarray,
    *,
    black: float = 0.0,
    mid: float = 0.5,
    white: float = 1.0,
    channel: str = "L",
) -> np.ndarray:
    """
    Headless levels on a numpy array. Preserves mono vs RGB shape.
    Returns float32 in [0,1].
    """
    a = _ensure_rgb01_keep(img)  # keeps mono as mono
    out = apply_histogram_transform_channel(a, float(black), float(mid), float(white), str(channel))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def levels_doc(
    doc,
    *,
    black: float = 0.0,
    mid: float = 0.5,
    white: float = 1.0,
    channel: str = "L",
    step_name: str = "Levels",
    apply: bool = True,
):
    if doc is None or getattr(doc, "image", None) is None:
        raise ValueError("levels_doc: doc has no image")

    src = np.asarray(doc.image)
    was_mono = _is_mono(src)

    result = levels_array(src, black=black, mid=mid, white=white, channel=channel)

    if not apply:
        return result

    meta = {"step_name": step_name, "levels": {"black": black, "mid": mid, "white": white, "channel": channel}}
    if was_mono:
        meta["is_mono"] = True

    if hasattr(doc, "apply_edit"):
        doc.apply_edit(result.astype(np.float32, copy=False), metadata=meta, step_name=step_name)
    elif hasattr(doc, "set_image"):
        doc.set_image(result, step_name=step_name)
    else:
        doc.image = result

    return result