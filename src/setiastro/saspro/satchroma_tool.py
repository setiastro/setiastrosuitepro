# ============================================================
#  ____  _         _____           _ _    _ _
# / ___|| |    __ |_   _|__   ___ | | | _(_) |_
# \___ \| |   / _` || |/ _ \ / _ \| | |/ / | __|
#  ___) | |__| (_| || | (_) | (_) | |   <| | |_
# |____/|_____\__,_||_|\___/ \___/|_|_|\_\_|\__|
#
#  Saturation / Chroma Tool  (SatChroma)
#  src/setiastro/saspro/satchroma_tool.py
#
#  Hue-selective saturation (HSV) and chroma (Lab) adjustment
#  using a draggable PCHIP spline curve over the hue wheel.
#  End control points are linked to wrap seamlessly at 0°/360°.
#  Preview uses an embedded split-view widget — never touches
#  the document image until Apply is clicked.
#
#  Part of Seti Astro Suite Pro
#  Copyright © 2025 Franklin Marek  |  www.setiastro.com
#  All rights reserved.
# ============================================================
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtGui import (
    QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath,
    QPen, QBrush, QPixmap,
)
from PyQt6.QtCore import (
    QPoint, QPointF, QRectF, QSettings, QSize, Qt, QTimer, QThread, QObject, pyqtSignal,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFrame,
    QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

Y_MIN      = 0.0
Y_MAX      = 3.0
Y_NEUT     = 1.0
N_CURVE    = 4096
DEBOUNCE_MS = 180
MAX_PREVIEW = 1200    # max pixel dimension for the preview ROI

_DEFAULT_POINTS: List[Tuple[float, float]] = [
    (0.0,  1.0),   # red  (linked to tail)
    (1/6,  1.0),   # yellow
    (2/6,  1.0),   # green
    (3/6,  1.0),   # cyan
    (4/6,  1.0),   # blue
    (5/6,  1.0),   # magenta
    (1.0,  1.0),   # red again (linked to head)
]

_HUE_STOPS = [
    (0.000, "#ff0000"), (0.167, "#ffff00"), (0.333, "#00ff00"),
    (0.500, "#00ffff"), (0.667, "#0000ff"), (0.833, "#ff00ff"),
    (1.000, "#ff0000"),
]


# ─────────────────────────────────────────────────────────────
# PCHIP spline
# ─────────────────────────────────────────────────────────────

def _pchip_lut(points: List[Tuple[float, float]], n: int = N_CURVE) -> np.ndarray:
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    xi = np.linspace(0.0, 1.0, n, dtype=np.float64)
    try:
        from scipy.interpolate import PchipInterpolator
        lut = PchipInterpolator(xs, ys, extrapolate=True)(xi)
    except ImportError:
        lut = np.interp(xi, xs, ys)
    return np.clip(lut, Y_MIN, Y_MAX).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────

def _apply_saturation_hsv(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    try:
        import cv2
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_norm = hsv[:, :, 0] / 360.0
        idx  = np.clip((hue_norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * lut[idx], 0.0, 1.0)
        out  = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    except ImportError:
        import colorsys
        flat = img.reshape(-1, 3)
        out_flat = np.empty_like(flat)
        for i, (r, g, b) in enumerate(flat):
            h_f, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
            idx = int(h_f * (len(lut) - 1))
            s2  = min(1.0, s * float(lut[idx]))
            out_flat[i] = colorsys.hsv_to_rgb(h_f, s2, v)
        out = out_flat.reshape(img.shape)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_chroma_lab(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    def _rgb2lab(rgb):
        lin = np.where(rgb <= 0.04045, rgb / 12.92,
                       ((rgb + 0.055) / 1.055) ** 2.4)
        M = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
        xyz = lin @ M.T
        xyz /= np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        f = np.where(xyz > 0.008856, xyz ** (1/3), 7.787 * xyz + 16/116)
        L = 116 * f[..., 1] - 16
        a = 500 * (f[..., 0] - f[..., 1])
        b = 200 * (f[..., 1] - f[..., 2])
        return np.stack([L, a, b], axis=-1)

    def _lab2rgb(lab):
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16) / 116; fx = a / 500 + fy; fz = fy - b / 200
        x = np.where(fx**3 > 0.008856, fx**3, (fx - 16/116) / 7.787) * 0.95047
        y = np.where(fy**3 > 0.008856, fy**3, (fy - 16/116) / 7.787) * 1.00000
        z = np.where(fz**3 > 0.008856, fz**3, (fz - 16/116) / 7.787) * 1.08883
        xyz = np.stack([x, y, z], axis=-1)
        M_inv = np.array([[ 3.2404542,-1.5371385,-0.4985314],
                           [-0.9692660, 1.8760108, 0.0415560],
                           [ 0.0556434,-0.2040259, 1.0572252]], dtype=np.float32)
        lin  = xyz @ M_inv.T
        rgb  = np.where(lin <= 0.0031308,
                        12.92 * lin,
                        1.055 * np.clip(lin, 0, None) ** (1/2.4) - 0.055)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    try:
        from skimage import color as sk
        lab = sk.rgb2lab(img.astype(np.float32))
    except ImportError:
        lab = _rgb2lab(img.astype(np.float32))

    a_ch = lab[..., 1]; b_ch = lab[..., 2]
    hue_norm = (np.arctan2(b_ch, a_ch) / (2 * math.pi)) % 1.0
    idx  = np.clip((hue_norm * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
    mult = lut[idx]
    lab[..., 1] = a_ch * mult
    lab[..., 2] = b_ch * mult

    try:
        from skimage import color as sk
        out = sk.lab2rgb(lab).astype(np.float32)
    except ImportError:
        out = _lab2rgb(lab)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# numpy → QPixmap helper  (32-bit float RGB → 8-bit QPixmap)
# ─────────────────────────────────────────────────────────────

def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert float32 H×W×3 [0,1] array to QPixmap."""
    rgb8 = np.clip(arr[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    h, w = rgb8.shape[:2]
    img  = QImage(rgb8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# ─────────────────────────────────────────────────────────────
# Split-preview widget
# ─────────────────────────────────────────────────────────────

class _SplitPreviewWidget(QWidget):
    """
    Side-by-side before/after preview with a draggable vertical split line.
    Left  = before (original)
    Right = after  (processed)
    Supports mouse-wheel zoom and click-drag pan. Auto-fits to the widget
    size until the user manually zooms or pans, after which the view is
    left alone across preview refreshes (via set_after()). Double-click
    anywhere off the split line to snap back to fit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._before: Optional[QPixmap] = None
        self._after:  Optional[QPixmap] = None
        self._split   = 0.5
        self._dragging_split = False
        self._zoom    = 1.0
        self._offset  = QPoint(0, 0)
        self._panning = False
        self._pan_start_mouse  = QPoint()
        self._pan_start_offset = QPoint()
        self._fit_mode = True   # True until the user manually zooms/pans

    def set_images(self, before: QPixmap, after: QPixmap):
        """Load a new baseline image pair — resets to fit."""
        self._before = before
        self._after  = after
        self._fit_mode = True
        self._fit()
        self.update()

    def set_after(self, after: QPixmap):
        """Update only the after pixmap (before + current view stay put)."""
        self._after = after
        self.update()

    def reset_view(self):
        """Explicitly re-fit to the widget (e.g. from a 'Fit' button)."""
        self._fit_mode = True
        self._fit()
        self.update()

    def _fit(self):
        if self._before is None:
            return
        pw, ph = self._before.width(), self._before.height()
        ww, wh = self.width(), self.height()
        if pw <= 0 or ph <= 0 or ww <= 0 or wh <= 0:
            return
        scale = min(ww / pw, wh / ph)
        self._zoom   = scale
        self._offset = QPoint(
            int((ww - pw * scale) / 2),
            int((wh - ph * scale) / 2),
        )

    def _img_rect(self) -> QRectF:
        if self._before is None:
            return QRectF()
        pw = self._before.width()  * self._zoom
        ph = self._before.height() * self._zoom
        return QRectF(self._offset.x(), self._offset.y(), pw, ph)

    def _near_split(self, x: int) -> bool:
        r = self._img_rect()
        sx = r.left() + self._split * r.width()
        return abs(x - sx) < 6

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 28))

        if self._before is None:
            p.setPen(QColor(100, 100, 110))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Open an image to see the preview.")
            p.end()
            return

        r = self._img_rect()
        split_x = r.left() + self._split * r.width()

        # Before (left)
        src_w = self._before.width()
        src_split = int(self._split * src_w)
        p.drawPixmap(
            QRectF(r.left(), r.top(), split_x - r.left(), r.height()),
            self._before,
            QRectF(0, 0, src_split, self._before.height()),
        )

        # After (right)
        if self._after is not None:
            after_src_w = self._after.width()
            after_split  = int(self._split * after_src_w)
            p.drawPixmap(
                QRectF(split_x, r.top(), r.right() - split_x, r.height()),
                self._after,
                QRectF(after_split, 0,
                       after_src_w - after_split, self._after.height()),
            )

        # Split line
        pen = QPen(QColor(255, 220, 60), 2)
        p.setPen(pen)
        p.drawLine(QPointF(split_x, r.top()), QPointF(split_x, r.bottom()))

        # Labels
        p.setFont(QFont("Segoe UI", 9))
        lbl_r = 60
        if split_x - r.left() > lbl_r + 4:
            p.setPen(QColor(200, 200, 200, 180))
            p.fillRect(QRectF(r.left() + 4, r.top() + 4, lbl_r, 18),
                       QColor(0, 0, 0, 100))
            p.drawText(QRectF(r.left() + 4, r.top() + 4, lbl_r, 18),
                       Qt.AlignmentFlag.AlignCenter, "Before")
        if r.right() - split_x > lbl_r + 4:
            p.setPen(QColor(200, 200, 200, 180))
            p.fillRect(QRectF(r.right() - lbl_r - 4, r.top() + 4, lbl_r, 18),
                       QColor(0, 0, 0, 100))
            p.drawText(QRectF(r.right() - lbl_r - 4, r.top() + 4, lbl_r, 18),
                       Qt.AlignmentFlag.AlignCenter, "After")
        p.end()

    def mousePressEvent(self, ev):
        x = int(ev.position().x())
        if self._near_split(x):
            self._dragging_split = True
        else:
            self._panning = True
            self._pan_start_mouse  = ev.position().toPoint()
            self._pan_start_offset = QPoint(self._offset)

    def mouseMoveEvent(self, ev):
        x = int(ev.position().x())
        if self._dragging_split:
            r = self._img_rect()
            if r.width() > 0:
                self._split = max(0.02, min(0.98,
                    (ev.position().x() - r.left()) / r.width()))
            self.update()
        elif self._panning:
            delta        = ev.position().toPoint() - self._pan_start_mouse
            self._offset = self._pan_start_offset + delta
            if delta.manhattanLength() > 3:
                self._fit_mode = False
            self.update()
        else:
            if self._near_split(x):
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, ev):
        self._dragging_split = False
        self._panning        = False

    def mouseDoubleClickEvent(self, ev):
        x = int(ev.position().x())
        if not self._near_split(x):
            self.reset_view()

    def wheelEvent(self, ev):
        dy = ev.pixelDelta().y() or ev.angleDelta().y()
        if dy == 0:
            return
        factor = 1.15 if dy > 0 else 1.0 / 1.15
        old_zoom = self._zoom
        self._zoom = max(0.05, min(32.0, self._zoom * factor))
        self._fit_mode = False
        # zoom toward cursor
        pos = ev.position().toPoint()
        self._offset = QPoint(
            int(pos.x() - (pos.x() - self._offset.x()) * self._zoom / old_zoom),
            int(pos.y() - (pos.y() - self._offset.y()) * self._zoom / old_zoom),
        )
        self.update()
        ev.accept()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._before is not None and self._fit_mode:
            self._fit()
            self.update()

# ─────────────────────────────────────────────────────────────
# Hue-curve canvas widget
# ─────────────────────────────────────────────────────────────

class HueCurveCanvas(QWidget):
    curve_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._pts: List[List[float]] = [list(p) for p in _DEFAULT_POINTS]
        self._drag_idx: Optional[int] = None
        self._hover_idx: Optional[int] = None
        self._pad = (38, 14, 12, 34)

    def _plot_rect(self) -> QRectF:
        pl, pt, pr, pb = self._pad
        return QRectF(pl, pt, self.width() - pl - pr, self.height() - pt - pb)

    def _to_px(self, h: float, m: float) -> QPointF:
        r = self._plot_rect()
        return QPointF(
            r.left() + h * r.width(),
            r.bottom() - (m - Y_MIN) / (Y_MAX - Y_MIN) * r.height(),
        )

    def _from_px(self, px: QPointF) -> Tuple[float, float]:
        r = self._plot_rect()
        h = max(0.0, min(1.0, (px.x() - r.left()) / r.width()))
        m = Y_MIN + (r.bottom() - px.y()) / r.height() * (Y_MAX - Y_MIN)
        return h, max(Y_MIN, min(Y_MAX, m))

    def _hit_test(self, pos: QPointF) -> Optional[int]:
        for i, (h, m) in enumerate(self._pts):
            if (self._to_px(h, m) - pos).manhattanLength() < 12:
                return i
        return None

    def get_points(self) -> List[Tuple[float, float]]:
        return [(p[0], p[1]) for p in self._pts]

    def set_points(self, pts: List[Tuple[float, float]]):
        self._pts = [list(p) for p in pts]
        self.update()

    def reset_points(self):
        self._pts = [list(p) for p in _DEFAULT_POINTS]
        self.curve_changed.emit()
        self.update()

    def build_lut(self) -> np.ndarray:
        return _pchip_lut(self.get_points())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._plot_rect()
        p.fillRect(self.rect(), QColor(30, 30, 35))

        # Hue gradient strip
        strip_h = 12
        strip_r = QRectF(r.left(), r.bottom() + 4, r.width(), strip_h)
        grad = QLinearGradient(strip_r.left(), 0, strip_r.right(), 0)
        for stop, col in _HUE_STOPS:
            grad.setColorAt(stop, QColor(col))
        p.fillRect(strip_r, grad)
        p.setPen(QPen(QColor(70, 70, 70), 1))
        p.drawRect(strip_r)

        # Grid
        p.setPen(QPen(QColor(55, 55, 60), 1, Qt.PenStyle.DotLine))
        for m in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            py = self._to_px(0, m).y()
            p.drawLine(QPointF(r.left(), py), QPointF(r.right(), py))
        for i in range(7):
            px = self._to_px(i / 6.0, Y_MIN).x()
            p.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))

        # Axis labels
        p.setPen(QColor(130, 130, 140))
        p.setFont(QFont("Segoe UI", 8))
        hue_lbls = ["0°", "60°", "120°", "180°", "240°", "300°", "360°"]
        for i, lbl in enumerate(hue_lbls):
            px = self._to_px(i / 6.0, Y_MIN).x()
            p.drawText(QRectF(px - 18, r.bottom() + 17, 36, 14),
                       Qt.AlignmentFlag.AlignHCenter, lbl)
        for m in [0.0, 1.0, 2.0, 3.0]:
            py = self._to_px(0, m).y()
            p.drawText(QRectF(0, py - 8, r.left() - 3, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{m:.0f}×")

        # Neutral line
        p.setPen(QPen(QColor(90, 160, 90, 120), 1, Qt.PenStyle.DashLine))
        ny = self._to_px(0, Y_NEUT).y()
        p.drawLine(QPointF(r.left(), ny), QPointF(r.right(), ny))

        # Border
        p.setPen(QPen(QColor(75, 75, 85), 1))
        p.drawRect(r)

        # Spline
        lut  = self.build_lut()
        path = QPainterPath()
        first = True
        step  = max(1, len(lut) // 512)
        for i in range(0, len(lut), step):
            pt = self._to_px(i / (len(lut) - 1), float(lut[i]))
            if first:
                path.moveTo(pt); first = False
            else:
                path.lineTo(pt)
        p.setPen(QPen(QColor(255, 220, 60), 2))
        p.drawPath(path)

        # Control points
        for i, (h, m) in enumerate(self._pts):
            pt      = self._to_px(h, m)
            linked  = (i == 0 or i == len(self._pts) - 1)
            hovered = (i == self._hover_idx)
            drag    = (i == self._drag_idx)
            if drag:
                oc, ic, ro, ri = QColor(255,255,80), QColor(255,200,0), 9, 5
            elif hovered:
                oc, ic, ro, ri = QColor(200,200,255), QColor(160,160,255), 8, 4
            elif linked:
                oc, ic, ro, ri = QColor(255,120,80), QColor(220,80,40), 7, 3
            else:
                oc, ic, ro, ri = QColor(200,200,200), QColor(255,255,255), 7, 3
            p.setPen(QPen(oc, 2)); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(pt, ro, ro)
            p.setBrush(QBrush(ic)); p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(pt, ri, ri)
            if drag or hovered:
                lbl = f"{h*360:.0f}°  ×{m:.2f}"
                p.setPen(QColor(255,240,180))
                p.setFont(QFont("Segoe UI", 8))
                tx = min(pt.x() + 10, self.width() - 72)
                ty = max(pt.y() - 14, 4)
                p.drawText(QRectF(tx, ty, 70, 14), Qt.AlignmentFlag.AlignLeft, lbl)
        p.end()

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = QPointF(ev.position())
        idx = self._hit_test(pos)
        if idx is not None:
            self._drag_idx = idx
        else:
            h, m = self._from_px(pos)
            ins  = 1
            for i, (ph, _) in enumerate(self._pts[1:-1], start=1):
                if ph < h: ins = i + 1
            self._pts.insert(ins, [h, m])
            self._drag_idx = ins
            self.curve_changed.emit()
        self.update()

    def mouseMoveEvent(self, ev):
        pos = QPointF(ev.position())
        if self._drag_idx is not None:
            h, m = self._from_px(pos)
            idx  = self._drag_idx
            if idx == 0:
                self._pts[0][1] = m; self._pts[-1][1] = m
            elif idx == len(self._pts) - 1:
                self._pts[-1][1] = m; self._pts[0][1] = m
            else:
                xlo = self._pts[idx-1][0] + 0.001
                xhi = self._pts[idx+1][0] - 0.001
                self._pts[idx][0] = max(xlo, min(xhi, h))
                self._pts[idx][1] = m
            self.curve_changed.emit()
            self.update()
        else:
            new_hover = self._hit_test(pos)
            if new_hover != self._hover_idx:
                self._hover_idx = new_hover
                self.update()
            r = self._plot_rect()
            if new_hover is not None:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif r.contains(pos):
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.unsetCursor()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_idx = None
            self.update()

    def mouseDoubleClickEvent(self, ev):
        pos = QPointF(ev.position())
        idx = self._hit_test(pos)
        if idx is not None and 0 < idx < len(self._pts) - 1:
            self._pts.pop(idx)
            self._hover_idx = None
            self.curve_changed.emit()
            self.update()

    def contextMenuEvent(self, ev):
        from PyQt6.QtWidgets import QMenu
        pos = QPointF(ev.pos())
        idx = self._hit_test(pos)
        menu = QMenu(self)
        if idx is not None and 0 < idx < len(self._pts) - 1:
            menu.addAction("Delete point").triggered.connect(
                lambda: self._delete_point(idx))
        menu.addAction("Reset curve").triggered.connect(self.reset_points)
        menu.exec(ev.globalPos())

    def _delete_point(self, idx: int):
        if 0 < idx < len(self._pts) - 1:
            self._pts.pop(idx)
            self.curve_changed.emit()
            self.update()

class _SatChromaApplyWorker(QObject):
    """Runs the full-resolution SatChroma processing off the GUI thread."""
    finished = pyqtSignal(bool, str, object)  # ok, error_message, result_array

    def __init__(self, source: np.ndarray, mode_index: int, lut: np.ndarray,
                 mask: Optional[np.ndarray] = None):
        super().__init__()
        self._source     = source
        self._mode_index = mode_index
        self._lut        = lut
        self._mask       = mask

    def run(self):
        try:
            img3 = self._source[:, :, :3] if self._source.ndim == 3 else self._source
            if self._mode_index == 0:
                out = _apply_saturation_hsv(img3.copy(), self._lut)
            else:
                out = _apply_chroma_lab(img3.copy(), self._lut)

            if self._mask is not None:
                m = self._mask
                if m.ndim == 2:
                    m = m[:, :, None]
                out = (m * out + (1.0 - m) * img3).astype(np.float32)

            out = np.clip(out, 0.0, 1.0).astype(np.float32)

            if self._source.ndim == 3 and self._source.shape[2] == 4:
                out = np.concatenate([out, self._source[:, :, 3:4]], axis=2)

            self.finished.emit(True, "", out)
        except Exception as e:
            import traceback
            self.finished.emit(False, f"{e}\n\n{traceback.format_exc()}", None)

# ─────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────

class SatChromaTool(QDialog):
    """
    Hue-selective Saturation / Chroma adjustment tool.

    Preview is rendered into an embedded split-view widget using a
    downsampled ROI of the source image — the document is never
    touched until Apply is clicked.
    """

    def __init__(self, doc_manager=None, document=None, parent=None):
        super().__init__(parent)
        self.docman   = doc_manager
        self.document = document
        self.settings = QSettings("SetiAstro", "SASpro")

        self.setWindowTitle("Saturation / Chroma Tool")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(1000, 620)

        # Debounce timer for live preview
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._update_preview)

        # Source image: full-res float32 RGB, never modified here
        self._source: Optional[np.ndarray] = None
        # Downsampled ROI for preview (keeps things fast)
        self._preview_src: Optional[np.ndarray] = None
        self._before_px:   Optional[QPixmap]    = None
        self._apply_thread: Optional[QThread] = None
        self._apply_worker: Optional[_SatChromaApplyWorker] = None
        self._build_ui()
        self._restore_geometry()

        if document is not None and getattr(document, "image", None) is not None:
            self._load_source(np.asarray(document.image, dtype=np.float32).copy())

    # ── source setup ─────────────────────────────────────────

    def _load_source(self, img: np.ndarray):
        """Store full-res source and build the downsampled preview buffer."""
        self._source = img

        # Ensure RGB
        if img.ndim == 2:
            img3 = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img3 = np.repeat(img, 3, axis=2)
        elif img.ndim == 3 and img.shape[2] >= 3:
            img3 = img[:, :, :3]
        else:
            img3 = img

        # Downsample for preview
        h, w = img3.shape[:2]
        scale = min(1.0, MAX_PREVIEW / max(h, w, 1))
        if scale < 1.0:
            nh, nw = int(h * scale), int(w * scale)
            try:
                import cv2
                prev = cv2.resize(img3, (nw, nh), interpolation=cv2.INTER_AREA)
            except ImportError:
                yi = np.linspace(0, h-1, nh).astype(np.int32)
                xi = np.linspace(0, w-1, nw).astype(np.int32)
                prev = img3[yi][:, xi]
        else:
            prev = img3.copy()

        self._preview_src = prev.astype(np.float32)
        self._before_px   = _arr_to_pixmap(self._preview_src)

        # Reset zoom/pan exactly once, for this new baseline image.
        self.preview.set_images(self._before_px, self._before_px)
        self._update_preview()

    # ── UI ────────────────────────────────────────────────────
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # ── Left column: mode, curve, controls ────────────────────
        left = QWidget(self)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(5)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox(self)
        self.combo_mode.addItems(["Saturation  (HSV)", "Chroma  (CIE Lab)"])
        self.combo_mode.setToolTip(
            "Saturation (HSV): adjusts colour purity in HSV.\n"
            "Chroma (Lab): adjusts perceptual colour strength — less hue shift."
        )
        self.combo_mode.currentIndexChanged.connect(self._on_changed)
        mode_row.addWidget(self.combo_mode, 1)
        left_lay.addLayout(mode_row)

        left_lay.addWidget(QLabel(
            "<span style='color:#777; font-size:10px;'>"
            "Drag · Click to add · Dbl-click to delete · Endpoints linked</span>"
        ))

        self.canvas = HueCurveCanvas(self)
        self.canvas.curve_changed.connect(self._on_changed)
        left_lay.addWidget(self.canvas, 1)

        str_row = QHBoxLayout()
        str_row.addWidget(QLabel("Global strength:"))
        self.sld_strength = QSlider(Qt.Orientation.Horizontal, self)
        self.sld_strength.setRange(0, 300)
        self.sld_strength.setValue(100)
        self.spin_strength = QDoubleSpinBox(self)
        self.spin_strength.setRange(0.0, 3.0)
        self.spin_strength.setSingleStep(0.05)
        self.spin_strength.setValue(1.0)
        self.spin_strength.setDecimals(2)
        self.spin_strength.setFixedWidth(64)
        self.sld_strength.valueChanged.connect(
            lambda v: self.spin_strength.setValue(v / 100.0))
        self.spin_strength.valueChanged.connect(
            lambda v: self.sld_strength.setValue(int(v * 100)))
        self.spin_strength.valueChanged.connect(self._on_changed)

        btn_reset = QPushButton("Reset Curve")
        btn_reset.setFixedWidth(88)
        btn_reset.clicked.connect(self.canvas.reset_points)

        str_row.addWidget(self.sld_strength, 1)
        str_row.addWidget(self.spin_strength)
        str_row.addWidget(btn_reset)
        left_lay.addLayout(str_row)

        opt_row = QHBoxLayout()
        self.chk_mask = QCheckBox("Respect active mask", self)
        self.chk_mask.setChecked(True)
        self.chk_live = QCheckBox("Live preview", self)
        self.chk_live.setChecked(True)
        self.chk_live.toggled.connect(self._on_live_toggled)

        btn_prev = QPushButton("Preview")
        btn_prev.clicked.connect(self._update_preview)

        opt_row.addWidget(self.chk_mask)
        opt_row.addStretch(1)
        opt_row.addWidget(self.chk_live)
        opt_row.addWidget(btn_prev)
        left_lay.addLayout(opt_row)

        self.lbl_status = QLabel("Ready.", self)
        self.lbl_status.setStyleSheet("color: #777; font-size: 10px;")
        left_lay.addWidget(self.lbl_status)

        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: rgba(255,255,255,18);")
        left_lay.addWidget(line)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to Document")
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.close)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)
        left_lay.addLayout(btn_row)

        splitter.addWidget(left)

        # ── Right column: preview ─────────────────────────────────
        right = QWidget(self)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        prev_hdr_row = QHBoxLayout()
        prev_hdr = QLabel("Preview", self)
        prev_hdr.setStyleSheet("color: #999; font-size: 10px; font-weight: bold;")
        btn_fit = QPushButton("Fit")
        btn_fit.setFixedWidth(50)
        btn_fit.setToolTip("Reset zoom/pan to fit the preview.\n(Double-click the preview also does this.)")
        btn_fit.clicked.connect(self._fit_preview_view)
        prev_hdr_row.addWidget(prev_hdr)
        prev_hdr_row.addStretch(1)
        prev_hdr_row.addWidget(btn_fit)
        right_lay.addLayout(prev_hdr_row)

        self.preview = _SplitPreviewWidget(self)
        right_lay.addWidget(self.preview, 1)

        splitter.addWidget(right)
        splitter.setSizes([380, 340])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
    # ── change / debounce ─────────────────────────────────────

    def _on_changed(self):
        if self.chk_live.isChecked():
            self._preview_timer.start()

    def _on_live_toggled(self, checked: bool):
        if checked:
            self._preview_timer.start()

    # ── LUT + processing ──────────────────────────────────────

    def _build_lut(self) -> np.ndarray:
        lut    = self.canvas.build_lut()
        scale  = float(self.spin_strength.value())
        scaled = (lut - 1.0) * scale + 1.0
        return np.clip(scaled, Y_MIN, Y_MAX).astype(np.float32)

    def _process(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] < 3:
            return img
        lut = self._build_lut()
        if self.combo_mode.currentIndex() == 0:
            return _apply_saturation_hsv(img[:, :, :3].copy(), lut)
        else:
            return _apply_chroma_lab(img[:, :, :3].copy(), lut)

    def _blend_mask(self, out: np.ndarray, src: np.ndarray,
                    use_full_res: bool = False) -> np.ndarray:
        """Blend with active mask. use_full_res=True for Apply path."""
        if not self.chk_mask.isChecked() or self.document is None:
            return out
        mid = getattr(self.document, "active_mask_id", None)
        if not mid:
            return out
        masks = getattr(self.document, "masks", {}) or {}
        layer = masks.get(mid)
        m = np.asarray(getattr(layer, "data", None)) if layer else None
        if m is None or m.size == 0:
            return out
        m = np.clip(m.astype(np.float32), 0.0, 1.0)

        # For the preview ROI, downsample the mask to match preview_src size
        if not use_full_res and m.shape[:2] != out.shape[:2]:
            nh, nw = out.shape[:2]
            yi = np.linspace(0, m.shape[0]-1, nh).astype(np.int32)
            xi = np.linspace(0, m.shape[1]-1, nw).astype(np.int32)
            m  = m[yi][:, xi]

        if m.ndim == 2 and out.ndim == 3:
            m = m[:, :, None]
        return (m * out + (1.0 - m) * src).astype(np.float32)

    # ── Preview (never touches document) ──────────────────────

    def _update_preview(self):
        if self._preview_src is None or self._before_px is None:
            return
        try:
            self.lbl_status.setText("Updating preview…")
            out   = self._process(self._preview_src)
            out   = self._blend_mask(out, self._preview_src, use_full_res=False)
            after = _arr_to_pixmap(out)
            self.preview.set_after(after)
            self.lbl_status.setText("Preview updated.")
        except Exception as e:
            self.lbl_status.setText(f"Preview error: {e}")

    def _fit_preview_view(self):
        self.preview.reset_view()

    # ── Apply ─────────────────────────────────────────────────

    def _get_full_res_mask(self) -> Optional[np.ndarray]:
        if not self.chk_mask.isChecked() or self.document is None:
            return None
        mid = getattr(self.document, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(self.document, "masks", {}) or {}
        layer = masks.get(mid)
        m = np.asarray(getattr(layer, "data", None)) if layer else None
        if m is None or m.size == 0:
            return None
        return np.clip(m.astype(np.float32), 0.0, 1.0)

    def _set_apply_controls_enabled(self, enabled: bool):
        self.btn_apply.setEnabled(enabled)
        self.btn_cancel.setEnabled(enabled)
        self.canvas.setEnabled(enabled)
        self.combo_mode.setEnabled(enabled)
        self.spin_strength.setEnabled(enabled)
        self.sld_strength.setEnabled(enabled)

    def _apply(self):
        if self.document is None or self._source is None:
            QMessageBox.information(self, "SatChroma", "No document loaded.")
            return
        if self._apply_thread is not None:
            return  # already running

        try:
            lut        = self._build_lut()
            mask       = self._get_full_res_mask()
            mode_index = self.combo_mode.currentIndex()

            self.lbl_status.setText("Applying — processing full-resolution image…")
            self._set_apply_controls_enabled(False)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            self._apply_thread = QThread(self)
            self._apply_worker = _SatChromaApplyWorker(
                self._source.copy(), mode_index, lut, mask
            )
            self._apply_worker.moveToThread(self._apply_thread)
            self._apply_thread.started.connect(
                self._apply_worker.run, Qt.ConnectionType.QueuedConnection)
            self._apply_worker.finished.connect(
                self._on_apply_finished, Qt.ConnectionType.QueuedConnection)
            self._apply_thread.start()
        except Exception:
            import traceback
            QMessageBox.critical(self, "SatChroma", traceback.format_exc())
            self._teardown_apply_thread()
            self._set_apply_controls_enabled(True)
            QApplication.restoreOverrideCursor()

    def _teardown_apply_thread(self):
        if self._apply_thread is not None:
            try:
                self._apply_thread.quit()
                self._apply_thread.wait()
            except Exception:
                pass
        self._apply_thread = None
        self._apply_worker = None

    def _on_apply_finished(self, ok: bool, message: str, out):
        self._teardown_apply_thread()
        self._set_apply_controls_enabled(True)
        QApplication.restoreOverrideCursor()

        if not ok:
            self.lbl_status.setText("Apply failed.")
            QMessageBox.critical(self, "SatChroma", message)
            return

        try:
            mode_name = self.combo_mode.currentText().split("(")[0].strip()
            mid = getattr(self.document, "active_mask_id", None) \
                  if self.chk_mask.isChecked() else None

            meta = {
                "step_name": f"SatChroma — {mode_name}",
                "mode":      self.combo_mode.currentText(),
                "strength":  self.spin_strength.value(),
                "points":    self.canvas.get_points(),
                "masked":    bool(mid),
                "mask_id":   mid,
            }
            self._remember_last_headless_command()
            self.document.apply_edit(out, metadata=meta,
                                     step_name=f"SatChroma — {mode_name}")
            self._load_source(out.copy())
            self.lbl_status.setText("Applied.")
        except Exception:
            import traceback
            self.lbl_status.setText("Apply failed.")
            QMessageBox.critical(self, "SatChroma", traceback.format_exc())

    # ── preset save / load ───────────────────────────────────

    def get_preset(self) -> dict:
        """Return the current UI state as a serialisable preset dict."""
        return {
            "command_id": "satchroma",
            "mode":       self.combo_mode.currentIndex(),
            "strength":   self.spin_strength.value(),
            "points":     self.canvas.get_points(),
            "use_mask":   self.chk_mask.isChecked(),
        }

    def load_preset(self, preset: dict):
        """Restore UI state from a preset dict (e.g. from replay)."""
        if "mode" in preset:
            self.combo_mode.setCurrentIndex(int(preset["mode"]))
        if "strength" in preset:
            self.spin_strength.setValue(float(preset["strength"]))
        if "points" in preset:
            pts = [(float(x), float(y)) for x, y in preset["points"]]
            self.canvas.set_points(pts)
        if "use_mask" in preset:
            self.chk_mask.setChecked(bool(preset["use_mask"]))
        self._update_preview()

    def _remember_last_headless_command(self):
        """
        Register this tool's current state with the main window so the
        workflow replay / history system can re-run it headlessly.
        """
        try:
            from PyQt6.QtWidgets import QApplication
            for w in QApplication.instance().topLevelWidgets():
                if hasattr(w, "_last_headless_command"):
                    w._last_headless_command = {
                        "command_id": "satchroma",
                        "preset":     self.get_preset(),
                    }
                    break
        except Exception:
            pass

    # ── geometry persistence ──────────────────────────────────

    def _restore_geometry(self):
        try:
            g = self.settings.value("satchroma/geometry")
            if g:
                self.restoreGeometry(g)
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self.settings.setValue("satchroma/geometry", self.saveGeometry())
        except Exception:
            pass
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────
# Headless runner  (called by replay / workflow / history)
# ─────────────────────────────────────────────────────────────

def apply_satchroma_headless(doc, preset: dict, main_window=None) -> bool:
    """
    Apply SatChroma to *doc* from a preset dict without opening any dialog.
    Used by the workflow assistant replay system and command-drop.

    Returns True on success, False on failure.

    preset keys (all optional, fall back to defaults):
        mode      : int   0=HSV saturation, 1=Lab chroma
        strength  : float global multiplier scale (default 1.0)
        points    : list  of [hue_norm, multiplier] pairs
        use_mask  : bool  respect active mask (default True)
    """
    import numpy as np

    if doc is None or getattr(doc, "image", None) is None:
        return False

    try:
        img = np.asarray(doc.image, dtype=np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        # Build LUT from preset
        pts = preset.get("points", _DEFAULT_POINTS)
        pts = [(float(x), float(y)) for x, y in pts]
        if not pts:
            pts = list(_DEFAULT_POINTS)

        lut    = _pchip_lut(pts)
        scale  = float(preset.get("strength", 1.0))
        lut    = np.clip((lut - 1.0) * scale + 1.0, Y_MIN, Y_MAX).astype(np.float32)

        mode   = int(preset.get("mode", 0))
        src3   = img[:, :, :3].copy()

        if mode == 0:
            out = _apply_saturation_hsv(src3, lut)
        else:
            out = _apply_chroma_lab(src3, lut)

        # Mask blend
        use_mask = bool(preset.get("use_mask", True))
        if use_mask:
            mid = getattr(doc, "active_mask_id", None)
            if mid:
                masks = getattr(doc, "masks", {}) or {}
                layer = masks.get(mid)
                m = np.asarray(getattr(layer, "data", None)) if layer else None
                if m is not None and m.size > 0:
                    m = np.clip(m.astype(np.float32), 0.0, 1.0)
                    if m.ndim == 2:
                        m = m[:, :, None]
                    if m.shape[:2] != out.shape[:2]:
                        yi = np.linspace(0, m.shape[0]-1, out.shape[0]).astype(np.int32)
                        xi = np.linspace(0, m.shape[1]-1, out.shape[1]).astype(np.int32)
                        m  = m[yi][:, xi]
                        if m.ndim == 2:
                            m = m[:, :, None]
                    out = (m * out + (1.0 - m) * src3).astype(np.float32)

        out = np.clip(out, 0.0, 1.0).astype(np.float32)

        # Re-attach alpha
        if img.shape[2] == 4:
            out = np.concatenate([out, img[:, :, 3:4]], axis=2)

        mode_names = {0: "Saturation (HSV)", 1: "Chroma (Lab)"}
        mode_name  = mode_names.get(mode, "Saturation")
        mid        = getattr(doc, "active_mask_id", None) if use_mask else None

        meta = {
            "command_id": "satchroma",
            "step_name":  f"SatChroma — {mode_name}",
            "mode":       mode,
            "strength":   scale,
            "points":     pts,
            "masked":     bool(mid),
            "mask_id":    mid,
            "preset":     dict(preset),
        }

        # Register with main window replay system
        if main_window is not None and hasattr(main_window, "_last_headless_command"):
            main_window._last_headless_command = {
                "command_id": "satchroma",
                "preset":     preset,
            }

        doc.apply_edit(out, metadata=meta, step_name=f"SatChroma — {mode_name}")
        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────
# Replay entry point  (called by main_window.replay_last_action_on_base)
# ─────────────────────────────────────────────────────────────

def replay_satchroma(main_window, target_sw=None):
    """
    Re-apply the last SatChroma operation headlessly on the base document.
    Mirrors the replay pattern used by Curves, Cosmic Clarity, etc.
    """
    last = getattr(main_window, "_last_headless_command", None) or {}
    if last.get("command_id") != "satchroma":
        return False

    preset = dict(last.get("preset") or {})

    # Resolve the target document
    doc = None
    try:
        if target_sw is not None:
            sw_widget = target_sw.widget() if hasattr(target_sw, "widget") else target_sw
            doc = getattr(sw_widget, "document", None)
        if doc is None and hasattr(main_window, "_active_doc"):
            doc = main_window._active_doc()
        if doc is None and getattr(main_window, "docman", None):
            doc = main_window.docman.get_active_document()
    except Exception:
        pass

    if doc is None:
        return False

    return apply_satchroma_headless(doc, preset, main_window=main_window)