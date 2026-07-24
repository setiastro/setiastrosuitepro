#saspro/curve_editor_pro.py
# ============================================================
#  Curves Editor  (curve_editor_pro.py)
#  Part of Seti Astro Suite Pro
#  Copyright © 2026 Franklin Marek  |  www.setiastro.com
#  All rights reserved.
# ============================================================
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QEvent, QPointF, QPoint, QTimer,
                          QSettings, QByteArray, QRectF, QSize)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QLineEdit,
    QWidget, QMessageBox, QRadioButton, QButtonGroup, QToolButton, QInputDialog, QMenu,
    QSizePolicy, QCheckBox, QScrollBar
)
from PyQt6.QtGui import (
    QPixmap, QImage, QWheelEvent, QPainter, QPainterPath, QPen, QColor, QBrush,
    QIcon, QKeyEvent, QCursor, QFont, QLinearGradient
)
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

from setiastro.saspro.widgets.image_utils import float_to_qimage_rgb8 as _float_to_qimage_rgb8

from setiastro.saspro.curves_preset import (
    list_custom_presets, save_custom_preset, _points_norm_to_scene, _norm_mode,
    _shape_points_norm, open_curves_with_preset, _lut_from_preset
)
from PyQt6.QtWidgets import QFrame
from scipy.interpolate import PchipInterpolator
from setiastro.saspro.curves_preset import _sanitize_scene_points, _norm_mode
from setiastro.saspro.histogram_transform_pro import HistogramWidget

try:
    from setiastro.saspro.legacy.numba_utils import (
        apply_lut_gray          as _nb_apply_lut_gray,
        apply_lut_color         as _nb_apply_lut_color,
        apply_lut_mono_inplace  as _nb_apply_lut_mono_inplace,
        apply_lut_color_inplace as _nb_apply_lut_color_inplace,
        rgb_to_xyz_numba, xyz_to_rgb_numba,
        xyz_to_lab_numba, lab_to_xyz_numba,
        rgb_to_hsv_numba, hsv_to_rgb_numba,
    )
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def _compute_hist_rgb(img01: np.ndarray) -> dict:
    from setiastro.saspro.histogram_transform_pro import _ensure_rgb01, _rgb_to_luma
    a = _ensure_rgb01(img01)
    out = {}
    for key, idx in (("R", 0), ("G", 1), ("B", 2)):
        hist, _ = np.histogram(a[..., idx].ravel(), bins=256, range=(0.0, 1.0))
        out[key] = hist.astype(np.int64)
    Y = _rgb_to_luma(a)
    hist, _ = np.histogram(Y.ravel(), bins=256, range=(0.0, 1.0))
    out["L"] = hist.astype(np.int64)
    return out


def _warm_numba_once():
    if not _HAS_NUMBA:
        return
    dummy = np.zeros((2, 2), np.float32)
    lut   = np.linspace(0, 1, 16).astype(np.float32)
    try:
        _nb_apply_lut_mono_inplace(dummy, lut)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Native-paint curve editor  (replaces QGraphicsView version)
# ─────────────────────────────────────────────────────────────

class CurveEditor(QWidget):
    """
    PCHIP curve editor drawn with QPainter.

    Public API is identical to the old QGraphicsView version so
    CurvesDialogPro requires zero changes.

    Points are stored as [(x, y)] in *scene* space [0..360] with
    Y increasing downward (matches the old convention: bottom-left
    is black, top-right is white).

    Endpoints:  index 0 (black/bottom-left) and -1 (white/top-right).
                They can slide along their respective edges.
    Controls:   everything in between — freely draggable, right-click
                or double-click to delete.
    """

    # ── constants ────────────────────────────────────────────
    _PAD   = (38, 12, 12, 28)   # left, top, right, bottom
    _SZ    = 360                 # logical scene size
    _STEPS = 512                 # spline sample count for painting
    _HIT_R = 10                  # hit-test radius (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 425)
        self.setFixedSize(380, 425)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        # Public callbacks (same as old version)
        self.preview_callback   = None
        self.symmetry_callback  = None

        # Points: list of [x, y] in scene space [0..360 × 0..360]
        # index 0 = black endpoint, index -1 = white endpoint
        self._pts: list[list[float]] = []
        self._drag_idx: int | None   = None
        self._hover_idx: int | None  = None

        # Spline state
        self.curve_function: PchipInterpolator | None = None
        self.curve_points:   list[tuple[float, float]] = []

        # Overlay curves from inactive channels {key: [(x,y) scene]}
        self._overlays: dict[str, list[tuple[float, float]]] = {}
        self._overlay_active_key: str = ""

        # Symmetry / inflection line (scene X, or None)
        self._sym_x: float | None = None

        # Value-indicator lines  {key: value_in_01}
        self._val_lines: dict[str, float | None] = {}

        # CDF (unused by canvas but kept for compat)
        self._cdf = None
        self._cdf_bins = 1024
        self._cdf_total = 0

        self._init_default_points()
        _warm_numba_once()

    # ── geometry helpers ─────────────────────────────────────

    def _plot_rect(self) -> QRectF:
        pl, pt, pr, pb = self._PAD
        return QRectF(pl, pt,
                      self.width()  - pl - pr,
                      self.height() - pt - pb)

    def _to_px(self, sx: float, sy: float) -> QPointF:
        """Scene coords [0..360] → widget pixel coords."""
        r = self._plot_rect()
        x = r.left() + (sx / self._SZ) * r.width()
        y = r.top()  + (sy / self._SZ) * r.height()   # Y down = dark at bottom
        return QPointF(x, y)

    def _from_px(self, px: QPointF) -> tuple[float, float]:
        """Widget pixel → scene coords, clamped to [0..360]."""
        r = self._plot_rect()
        sx = (px.x() - r.left()) / r.width()  * self._SZ
        sy = (px.y() - r.top())  / r.height() * self._SZ
        return float(np.clip(sx, 0, self._SZ)), float(np.clip(sy, 0, self._SZ))

    def _hit_test(self, pos: QPointF) -> int | None:
        best_d, best_i = self._HIT_R + 1, None
        for i, (sx, sy) in enumerate(self._pts):
            d = (self._to_px(sx, sy) - pos).manhattanLength()
            if d < best_d:
                best_d, best_i = d, i
        return best_i if best_d <= self._HIT_R else None

    # ── point initialisation ──────────────────────────────────

    def _init_default_points(self):
        self._pts = [
            [0.0,   self._SZ],   # black endpoint  (bottom-left)
            [self._SZ, 0.0],     # white endpoint  (top-right)
        ]
        self._rebuild_spline()

    # ── public API (matches old QGraphicsView version exactly) ─

    # --- scene / end-point introspection (used by dialog) ---

    @property
    def end_points(self):
        """Compat shim: return fake objects with .scenePos()."""
        class _FakePt:
            def __init__(self, x, y):
                self._x, self._y = x, y
            def scenePos(self):
                return QPointF(self._x, self._y)
        return [_FakePt(*self._pts[0]), _FakePt(*self._pts[-1])]

    @property
    def control_points(self):
        """
        Compat shim. Returns fake point objects whose .setPos(x, y) writes
        back into self._pts so GhsDialogPro can reposition handles in-place.
        """
        editor = self
        pts    = self._pts

        class _FakePt:
            def __init__(self, idx, x, y):
                self._idx = idx
                self._x, self._y = x, y
            def scenePos(self):
                return QPointF(self._x, self._y)
            def setPos(self, x, y):
                pts[self._idx][0] = float(x)
                pts[self._idx][1] = float(y)
                self._x, self._y  = float(x), float(y)
                editor._rebuild_spline()
                editor.update()

        return [_FakePt(i + 1, p[0], p[1]) for i, p in enumerate(self._pts[1:-1])]

    # --- callbacks ---

    def setPreviewCallback(self, callback):
        self.preview_callback = callback

    def setSymmetryCallback(self, fn):
        self.symmetry_callback = fn

    # --- curve access ---

    def getCurveFunction(self) -> PchipInterpolator | None:
        return self.curve_function

    def getCurvePoints(self) -> list[tuple[float, float]]:
        return list(self.curve_points)

    def get8bitLUT(self) -> np.ndarray:
        return self._make_lut(256).astype(np.uint8)

    def getLUT(self) -> np.ndarray:
        return (self._make_lut(65536) * 65535.0).clip(0, 65535).astype(np.uint16)

    def _make_lut(self, size: int) -> np.ndarray:
        pts = self.curve_points
        if not pts:
            return np.linspace(0, 1, size, dtype=np.float32)
        arr = np.array(pts, dtype=np.float64)
        xs  = arr[:, 0]
        ys  = self._SZ - arr[:, 1]          # flip Y → output goes up
        inp = np.linspace(0, self._SZ, size, dtype=np.float64)
        out = np.interp(inp, xs, ys)
        if size == 256:
            return np.clip(out / self._SZ * 255, 0, 255).astype(np.float32)
        return np.clip(out / self._SZ, 0.0, 1.0).astype(np.float32)

    # --- control handles (used by dialog for store/recall) ---

    def getControlHandles(self) -> list[tuple[float, float]]:
        return [(p[0], p[1]) for p in self._pts[1:-1]]

    def setControlHandles(self, handles: list[tuple[float, float]]):
        ep0 = self._pts[0]
        ep1 = self._pts[-1]
        self._pts = [ep0] + [[float(x), float(y)] for x, y in handles] + [ep1]
        self._rebuild_spline()
        self.update()

    # --- init / reset ---

    def initCurve(self):
        self._init_default_points()
        self._sym_x = None
        self._val_lines.clear()
        self.update()

    def updateCurve(self):
        self._rebuild_spline()
        self.update()

    # --- overlay curves ---

    def setOverlayCurves(self, overlays: dict[str, list[tuple[float, float]]], active_key: str):
        self._overlays = overlays
        self._overlay_active_key = active_key
        self.update()

    # --- value indicator lines ---

    def updateValueLines(self, r: float, g: float, b: float, grayscale: bool = False):
        if grayscale:
            self._val_lines = {"gray": r}
        else:
            self._val_lines = {"r": r, "g": g, "b": b}
        self.update()

    def clearValueLines(self):
        self._val_lines.clear()
        self.update()

    # --- symmetry / inflection ---

    def setSymmetryPoint(self, x: float, y: float):
        self._sym_x = float(x)
        self.update()

    def clearSymmetryLine(self):
        self._sym_x = None
        self.update()

    def redistributeHandlesByPivot(self, u: float):
        u = float(np.clip(u, 0.0, 1.0))
        controls = self._pts[1:-1]
        N = len(controls)
        if N == 0:
            return
        nL = N // 2
        nR = N - nL
        xL = np.linspace(0.0, u * self._SZ, nL + 2, dtype=np.float32)[1:-1]
        xR = np.linspace(u * self._SZ, self._SZ, nR + 2, dtype=np.float32)[1:-1]
        xs = np.concatenate([xL, xR]) if (nL and nR) else (xR if nL == 0 else xL)
        fn = self.curve_function
        ys = np.clip(fn(xs), 0.0, self._SZ) if callable(fn) else (self._SZ - xs)
        ep0 = self._pts[0]
        ep1 = self._pts[-1]
        self._pts = [ep0] + [[float(x), float(y)] for x, y in zip(xs, ys)] + [ep1]
        self._rebuild_spline()
        self.update()

    # --- black / white threshold readback (used by dialog) ---

    def current_black_white_thresholds(self) -> tuple[float | None, float | None]:
        bx = wx = None
        eps = 1.0
        p0, p1 = self._pts[0], self._pts[-1]
        for p in (p0, p1):
            x, y = p
            if abs(y - self._SZ) <= eps:   # bottom edge → black
                bx = float(np.clip(x / self._SZ, 0, 1))
            if abs(y - 0.0) <= eps:         # top edge → white
                wx = float(np.clip(x / self._SZ, 0, 1))
        return bx, wx

    # ── spline rebuild ───────────────────────────────────────

    def _rebuild_spline(self):
        pts = self._pts
        if len(pts) < 2:
            self.curve_function = None
            self.curve_points   = []
            return

        # sort by X, dedupe
        sorted_pts = sorted(pts, key=lambda p: p[0])
        xs, ys = [], []
        last_x = -1e9
        for p in sorted_pts:
            x = float(p[0])
            if x <= last_x:
                x = last_x + 1e-4
            xs.append(x); ys.append(float(p[1]))
            last_x = x

        try:
            interp = PchipInterpolator(xs, ys, extrapolate=True)
            self.curve_function = interp
            sx = np.linspace(xs[0], xs[-1], self._STEPS, dtype=np.float64)
            sy = np.clip(interp(sx), 0.0, self._SZ)
            self.curve_points = [(float(x), float(y)) for x, y in zip(sx, sy)]
        except Exception:
            self.curve_function = None
            self.curve_points   = list(zip(xs, ys))

        if self.preview_callback:
            lut = self.get8bitLUT()
            self.preview_callback(lut)

    # ── painting ─────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor(32, 32, 36))

        r = self._plot_rect()

        # Grid
        grid_pen = QPen(QColor(62, 62, 72), 1, Qt.PenStyle.DashLine)
        p.setPen(grid_pen)
        for i in range(11):
            t = i / 10.0
            px = r.left() + t * r.width()
            py = r.top()  + t * r.height()
            p.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))
            p.drawLine(QPointF(r.left(), py), QPointF(r.right(), py))

        # Diagonal identity reference
        p.setPen(QPen(QColor(70, 70, 80), 1, Qt.PenStyle.DotLine))
        p.drawLine(self._to_px(0, self._SZ), self._to_px(self._SZ, 0))

        # Border
        p.setPen(QPen(QColor(75, 75, 85), 1))
        p.drawRect(r)

        # X-axis labels
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(140, 140, 150))
        for i in range(11):
            val = i / 10.0
            sx  = val * self._SZ
            px  = self._to_px(sx, self._SZ)
            p.drawText(QRectF(px.x() - 16, r.bottom() + 3, 32, 14),
                       Qt.AlignmentFlag.AlignHCenter,
                       f"{val:.1f}")

        # Y-axis labels
        for i in range(11):
            val = i / 10.0
            sy  = (1.0 - val) * self._SZ   # scene Y (0=top=white)
            py  = self._to_px(0, sy)
            p.drawText(QRectF(0, py.y() - 7, r.left() - 3, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f"{val:.1f}")

        # ── Overlay curves (inactive channels) ───────────────
        OV_COLORS = {
            "K":"#FFFFFF", "R":"#FF4A4A", "G":"#5CC45C", "B":"#4AA0FF",
            "L*":"#FFFFFF", "a*":"#FF8AB2", "b*":"#A6C8FF",
            "Chroma":"#FFD866", "Saturation":"#66FFD8",
        }
        for key, scene_pts in self._overlays.items():
            if key == self._overlay_active_key or len(scene_pts) < 2:
                continue
            pts_sorted = sorted(scene_pts, key=lambda t: t[0])
            ov_xs = np.array([t[0] for t in pts_sorted], dtype=np.float64)
            ov_ys = np.array([t[1] for t in pts_sorted], dtype=np.float64)
            if np.any(np.diff(ov_xs) <= 0):
                ov_xs += np.linspace(0, 1e-3, len(ov_xs))
            sx = np.linspace(ov_xs[0], ov_xs[-1], self._STEPS, dtype=np.float64)
            try:
                sy = PchipInterpolator(ov_xs, ov_ys, extrapolate=True)(sx)
            except Exception:
                sy = np.interp(sx, ov_xs, ov_ys)
            col = QColor(OV_COLORS.get(key, "#BBBBBB"))
            col.setAlpha(80)
            pen = QPen(col, 1)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            path = QPainterPath()
            first = True
            for x, y in zip(sx, sy):
                pt = self._to_px(float(x), float(np.clip(y, 0, self._SZ)))
                if first:
                    path.moveTo(pt); first = False
                else:
                    path.lineTo(pt)
            p.drawPath(path)

        # ── Symmetry / inflection line ───────────────────────
        if self._sym_x is not None:
            sym_pen = QPen(QColor(255, 220, 60), 1, Qt.PenStyle.DashLine)
            p.setPen(sym_pen)
            top = self._to_px(self._sym_x, 0)
            bot = self._to_px(self._sym_x, self._SZ)
            p.drawLine(top, bot)

        # ── Value-indicator lines ─────────────────────────────
        LINE_COLORS = {
            "r": QColor(255, 70, 70),
            "g": QColor(70, 220, 70),
            "b": QColor(70, 130, 255),
            "gray": QColor(180, 180, 180),
        }
        for key, val in self._val_lines.items():
            if val is None:
                continue
            col = LINE_COLORS.get(key, QColor(200, 200, 200))
            col.setAlpha(200)
            p.setPen(QPen(col, 1))
            sx = float(val) * self._SZ
            top = self._to_px(sx, 0)
            bot = self._to_px(sx, self._SZ)
            p.drawLine(top, bot)

        # ── Active curve ──────────────────────────────────────
        if len(self.curve_points) >= 2:
            # Shadow
            sh_pen = QPen(QColor(0, 0, 0, 160), 5)
            sh_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            sh_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            path = QPainterPath()
            first = True
            for sx, sy in self.curve_points:
                pt = self._to_px(sx, float(np.clip(sy, 0, self._SZ)))
                if first:
                    path.moveTo(pt); first = False
                else:
                    path.lineTo(pt)
            p.setPen(sh_pen)
            p.drawPath(path)
            # Foreground
            fg_pen = QPen(QColor(255, 255, 255), 2)
            fg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            fg_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(fg_pen)
            p.drawPath(path)

        # ── Control points ────────────────────────────────────
        for i, (sx, sy) in enumerate(self._pts):
            is_endpoint = (i == 0 or i == len(self._pts) - 1)
            is_hover    = (i == self._hover_idx)
            is_drag     = (i == self._drag_idx)

            pt = self._to_px(sx, sy)

            if is_drag:
                outer_c, inner_c, ro, ri = QColor(255,255,80),  QColor(255,200,0),   9, 5
            elif is_hover:
                outer_c, inner_c, ro, ri = QColor(200,200,255), QColor(160,160,255), 8, 4
            elif is_endpoint:
                outer_c, inner_c, ro, ri = QColor(255,120,80),  QColor(220,80,40),   7, 3
            else:
                outer_c, inner_c, ro, ri = QColor(140,230,140), QColor(80,200,80),   7, 3

            p.setPen(QPen(outer_c, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(pt, ro, ro)
            p.setBrush(QBrush(inner_c))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(pt, ri, ri)

            if is_drag or is_hover:
                x_norm = sx / self._SZ
                y_norm = 1.0 - sy / self._SZ
                lbl = f"in {x_norm:.3f}  out {y_norm:.3f}"
                p.setFont(QFont("Segoe UI", 8))
                p.setPen(QColor(255, 240, 180))
                tx = min(pt.x() + 12, self.width() - 110)
                ty = max(pt.y() - 14, 4)
                p.drawText(QRectF(tx, ty, 105, 14), Qt.AlignmentFlag.AlignLeft, lbl)

        p.end()

    # ── mouse events ─────────────────────────────────────────

    def mousePressEvent(self, event):
        pos = QPointF(event.position())

        # Ctrl+left → inflection/symmetry pick
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            sx, _ = self._from_px(pos)
            self._sym_x = sx
            u = sx / self._SZ
            self.update()
            if self.symmetry_callback:
                self.symmetry_callback(u, 1.0 - u)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._hit_test(pos)
            if idx is not None:
                self._drag_idx = idx
            else:
                sx, sy = self._from_px(pos)
                r = self._plot_rect()
                # only add a point if click is inside the plot area
                if r.contains(pos):
                    self._insert_control(sx, sy)
            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            idx = self._hit_test(pos)
            if idx is not None and 0 < idx < len(self._pts) - 1:
                self._pts.pop(idx)
                self._hover_idx = None
                self._rebuild_spline()
                self.update()

    def mouseMoveEvent(self, event):
        pos = QPointF(event.position())

        if self._drag_idx is not None:
            idx = self._drag_idx
            sx, sy = self._from_px(pos)

            if idx == 0:
                # black endpoint: slides along bottom or left edges
                self._move_black_endpoint(sx, sy, pos)
            elif idx == len(self._pts) - 1:
                # white endpoint: slides along top or right edges
                self._move_white_endpoint(sx, sy, pos)
            else:
                # interior control: free within X neighbours
                xlo = self._pts[idx - 1][0] + 0.5
                xhi = self._pts[idx + 1][0] - 0.5
                self._pts[idx][0] = float(np.clip(sx, xlo, xhi))
                self._pts[idx][1] = float(np.clip(sy, 0, self._SZ))

            self._rebuild_spline()
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

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_idx = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        pos = QPointF(event.position())
        idx = self._hit_test(pos)
        if idx is not None and 0 < idx < len(self._pts) - 1:
            self._pts.pop(idx)
            self._hover_idx = None
            self._rebuild_spline()
            self.update()
        elif idx is None:
            # double-click on empty space → add point (same as single click,
            # but single already added it; this just prevents a ghost second add)
            pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            # delete hovered/selected interior point
            if self._hover_idx is not None and 0 < self._hover_idx < len(self._pts) - 1:
                self._pts.pop(self._hover_idx)
                self._hover_idx = None
                self._rebuild_spline()
                self.update()
        super().keyPressEvent(event)

    # ── endpoint constrained movement ────────────────────────

    def _move_black_endpoint(self, sx: float, sy: float, pos: QPointF):
        """Black point slides on bottom edge (y=360) or left edge (x=0)."""
        r   = self._plot_rect()
        # which edge is pos closer to?
        d_bottom = abs(pos.y() - r.bottom())
        d_left   = abs(pos.x() - r.left())
        right_neighbor_x = self._pts[1][0] if len(self._pts) > 1 else self._SZ
        if d_left <= d_bottom:
            self._pts[0] = [0.0, float(np.clip(sy, 0, self._SZ))]
        else:
            self._pts[0] = [float(np.clip(sx, 0, right_neighbor_x - 0.5)), self._SZ]

    def _move_white_endpoint(self, sx: float, sy: float, pos: QPointF):
        """White point slides on top edge (y=0) or right edge (x=360)."""
        r   = self._plot_rect()
        d_top   = abs(pos.y() - r.top())
        d_right = abs(pos.x() - r.right())
        left_neighbor_x = self._pts[-2][0] if len(self._pts) > 1 else 0.0
        if d_right <= d_top:
            self._pts[-1] = [self._SZ, float(np.clip(sy, 0, self._SZ))]
        else:
            self._pts[-1] = [float(np.clip(sx, left_neighbor_x + 0.5, self._SZ)), 0.0]

    # ── insert control point ──────────────────────────────────

    def _insert_control(self, sx: float, sy: float, start_drag: bool = True):
        # find insertion index (keep X-sorted)
        ins = 1
        for i, (px, _) in enumerate(self._pts[1:-1], start=1):
            if px < sx:
                ins = i + 1
        self._pts.insert(ins, [sx, sy])
        if start_drag:
            self._drag_idx = ins
        self._rebuild_spline()

    # ── compatibility shims (no-ops or delegating) ────────────

    def addControlPoint(self, x: float, y: float, lock_axis=None):
        """
        Add a single interior control point.

        GHS pre-population pattern: it calls addControlPoint(0, 0) N times
        when control_points is empty, then immediately overwrites positions
        via setPos(). To avoid stacking N points at x=0 (which collides with
        the black endpoint), we spread them evenly across the input range on
        the identity line when x==0 and y==0.
        """
        sx, sy = float(x), float(y)
        if sx == 0.0 and sy == 0.0:
            # GHS sentinel: insert a placeholder evenly spaced on identity line
            n_existing = len(self._pts) - 2  # current interior count
            # place at equally-spaced x, on the identity diagonal (y = SZ - x)
            frac = (n_existing + 1) / 21.0   # 20 points → fracs 1/21..20/21
            frac = float(np.clip(frac, 0.01, 0.99))
            sx = frac * self._SZ
            sy = self._SZ - sx               # identity: output == input
        self._insert_control(sx, sy, start_drag=False)
        self.update()

    def addEndPoint(self, *args, **kwargs):
        pass  # endpoints are always at index 0 and -1

    # curve_item is referenced by GhsDialogPro._recolor_curve(); return a
    # no-op shim so it doesn't crash (colour is ignored on native-paint canvas).
    @property
    def curve_item(self):
        class _NullItem:
            def setPen(self, *a, **kw): pass
            def isValid(self): return False
            def __bool__(self): return False
        return _NullItem()


# ── ImageLabel (unchanged) ────────────────────────────────────────────────────

class ImageLabel(QLabel):
    mouseMoved = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.position().x(), event.position().y())
        super().mouseMoveEvent(event)

# ─────────────────────────────────────────────────────────────
# Histogram + transfer-curve view
#
# Shared by CurvesDialogPro and GhsDialogPro. Draws per-channel histograms
# with x-zoom/pan and, optionally, the exact transfer curve that will be
# applied — no spline, no resampled control points, so the drawing cannot
# drift from the result.
#
# Two ways to feed it the "after" state:
#   set_result(lut=...)          - push the before-histogram through a
#                                  monotone LUT (cheap; GHS uses this)
#   set_result(result_image=...) - histogram an already-processed image
#                                  (Curves uses this, because it composites
#                                  several LUTs plus Lab/HSV round-trips and
#                                  there is no single LUT to push through)
# ─────────────────────────────────────────────────────────────

_HIST_BINS          = 8192      # useful when zoomed into the first 1-2%
_CURVE_POINTS       = 1024
_SMOOTH_PX          = 2.5       # Gaussian kernel width in SCREEN pixels
_HIST_SAMPLES_REF   = 2_000_000 # reference pass: once per image load
_HIST_SAMPLES_LIVE  = 600_000   # live pass: every debounce tick

_HV_CH_COLORS = {
    "K": QColor(220, 220, 220),
    "R": QColor(255,  80,  80),
    "G": QColor(110, 220,  95),
    "B": QColor( 80, 145, 255),
}
_HV_MARKER_COLORS = {
    "SP": QColor(255, 255, 255),
    "LP": QColor( 90, 135, 255),
    "HP": QColor(255, 210,  80),
    "BP": QColor(255, 120, 120),
}

# "b*" must not resolve to the blue channel, and Chroma/Saturation/L*/a*
# have no channel of their own — they all fall back to K.
_HV_CHANNEL_ALIASES = {
    "K": "K", "K (BRIGHTNESS)": "K", "BRIGHTNESS": "K", "LUMA": "K",
    "R": "R", "RED": "R",
    "G": "G", "GREEN": "G",
    "B": "B", "BLUE": "B",
}


def _hist_of(a, max_n=_HIST_SAMPLES_REF):
    flat = np.asarray(a, dtype=np.float32).ravel()
    if flat.size > max_n:
        flat = flat[:: max(1, flat.size // max_n)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        flat = np.zeros(1, dtype=np.float32)
    h, _ = np.histogram(np.clip(flat, 0.0, 1.0), bins=_HIST_BINS, range=(0.0, 1.0))
    return h.astype(np.float32)


def _channel_hists(img, max_n=_HIST_SAMPLES_REF):
    arr = np.asarray(img, dtype=np.float32)
    out = {}
    if arr.ndim == 3 and arr.shape[2] >= 3:
        for i, name in enumerate(("R", "G", "B")):
            out[name] = _hist_of(arr[..., i], max_n)
        out["K"] = (out["R"] + out["G"] + out["B"]) / 3.0
        return True, out
    h = _hist_of(arr if arr.ndim == 2 else arr[..., 0], max_n)
    for name in ("K", "R", "G", "B"):
        out[name] = h
    return False, out


def _remap_hist(hist, lut):
    """
    Push a histogram through a monotone LUT, treating each bin as an INTERVAL
    rather than a point.

    Mapping bin centres and bincount-ing leaves gaps wherever the transform
    expands the input — adjacent centres land 2-3 output bins apart and
    everything between gets nothing, which reads as a comb. Since the LUT is
    monotone the correct answer is the input CDF resampled onto the output
    axis: exact, mass-preserving, no gaps, and it degrades gracefully where
    the LUT is flat because the image is clipping.
    """
    n = int(hist.size)
    edges = np.linspace(0.0, 1.0, n + 1)
    idx = np.clip((edges * (lut.size - 1)).astype(np.int64), 0, lut.size - 1)
    edges_out = np.asarray(lut, dtype=np.float64)[idx]
    cdf = np.concatenate(([0.0], np.cumsum(hist, dtype=np.float64)))
    out = np.diff(np.interp(edges, edges_out, cdf))
    return np.maximum(out, 0.0).astype(np.float32)


def _clip_pct(hist, lut):
    centers = (np.arange(_HIST_BINS, dtype=np.float64) + 0.5) / _HIST_BINS
    idx = np.clip((centers * (lut.size - 1)).astype(np.int64), 0, lut.size - 1)
    mapped = np.asarray(lut, dtype=np.float64)[idx]
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0, 0.0
    return (float(hist[mapped <= 0.0].sum()) / total * 100.0,
            float(hist[mapped >= 1.0].sum()) / total * 100.0)


def _smooth_path(pts):
    """
    Quadratic Beziers through segment midpoints: every sample becomes a control
    point and the result is C1 continuous, so the outline reads as a curve
    instead of a polyline.
    """
    path = QPainterPath()
    if not pts:
        return path
    path.moveTo(pts[0][0], pts[0][1])
    if len(pts) < 3:
        for x, y in pts[1:]:
            path.lineTo(x, y)
        return path
    for i in range(1, len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        path.quadTo(x0, y0, (x0 + x1) * 0.5, (y0 + y1) * 0.5)
    path.lineTo(pts[-1][0], pts[-1][1])
    return path


class _HistCanvas(QWidget):
    pivotPicked  = pyqtSignal(float)
    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None, min_h=190):
        super().__init__(parent)
        self.setMinimumSize(240, int(min_h))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._is_rgb = False
        self._channel = "K"
        self._before = {}
        self._after = {}
        self._curve = None
        self._markers = {}
        self._log = False
        self._x0, self._x1 = 0.0, 1.0
        self._vlines = None

    # -- data
    def set_histograms(self, is_rgb, before, after):
        self._is_rgb, self._before, self._after = bool(is_rgb), before or {}, after or {}
        self.update()

    def set_curve(self, c):
        self._curve = c; self.update()

    def set_markers(self, m):
        self._markers = dict(m or {}); self.update()

    def set_channel(self, ch):
        self._channel = ch if ch in _HV_CH_COLORS else "K"; self.update()

    def set_log(self, on):
        self._log = bool(on); self.update()

    def set_value_lines(self, r, g, b, gray):
        self._vlines = (r, g, b, bool(gray)); self.update()

    def clear_value_lines(self):
        self._vlines = None; self.update()

    # -- x range
    def view_range(self):
        return self._x0, self._x1

    def set_view_range(self, x0, x1, emit=True):
        x0 = float(np.clip(x0, 0.0, 1.0)); x1 = float(np.clip(x1, 0.0, 1.0))
        if x1 <= x0:
            x0, x1 = 0.0, 1.0
        self._x0, self._x1 = x0, x1
        self.update()
        if emit:
            self.rangeChanged.emit(x0, x1)

    def reset_view(self):
        self.set_view_range(0.0, 1.0)

    def zoom_x(self, factor, center=0.5):
        span = max(1e-6, self._x1 - self._x0)
        new = float(np.clip(span * factor, 0.0005, 1.0))
        if new >= 0.999999:
            self.reset_view(); return
        c = float(np.clip(center, 0.0, 1.0))
        x0 = self._x0 + c * span - new * c
        x1 = x0 + new
        if x0 < 0.0: x1 -= x0; x0 = 0.0
        if x1 > 1.0: x0 -= (x1 - 1.0); x1 = 1.0
        self.set_view_range(max(0.0, x0), min(1.0, x1))

    def pan_to_fraction(self, f):
        span = max(1e-9, self._x1 - self._x0)
        if span >= 0.999999:
            self.reset_view(); return
        x0 = float(np.clip(f, 0.0, 1.0)) * (1.0 - span)
        self.set_view_range(x0, x0 + span)

    # -- geometry
    def _plot_rect(self):
        return self.rect().adjusted(8, 6, -8, -6)

    def _px(self, v, rect):
        span = max(1e-9, self._x1 - self._x0)
        return rect.left() + rect.width() * (float(v) - self._x0) / span

    def _vis(self, v):
        return self._x0 <= float(v) <= self._x1

    def _names(self):
        if self._channel == "K":
            return ("R", "G", "B") if self._is_rgb else ("K",)
        return (self._channel,)

    # -- events
    def _emit_pivot(self, ev):
        rect = self._plot_rect(); p = ev.position()
        if not rect.contains(int(p.x()), int(p.y())):
            return False
        nx = (float(p.x()) - rect.left()) / max(1.0, float(rect.width() - 1))
        self.pivotPicked.emit(float(np.clip(self._x0 + nx * (self._x1 - self._x0), 0.0, 1.0)))
        ev.accept()
        return True

    def mouseDoubleClickEvent(self, ev):
        if not self._emit_pivot(ev):
            super().mouseDoubleClickEvent(ev)

    def mousePressEvent(self, ev):
        if (ev.modifiers() & Qt.KeyboardModifier.ControlModifier) and self._emit_pivot(ev):
            return
        super().mousePressEvent(ev)

    def wheelEvent(self, ev):
        rect = self._plot_rect(); p = ev.position()
        if rect.contains(int(p.x()), int(p.y())):
            nx = (float(p.x()) - rect.left()) / max(1.0, float(rect.width() - 1))
            d = ev.angleDelta().y()
            if d > 0:   self.zoom_x(0.80, nx)
            elif d < 0: self.zoom_x(1.25, nx)
            ev.accept(); return
        super().wheelEvent(ev)

    # -- painting
    def _draw_hists(self, p, rect, coll, fill_a, line_a):
        w = max(1, rect.width())
        n = _HIST_BINS

        i0 = int(np.floor(self._x0 * (n - 1)))
        i1 = int(np.ceil(self._x1 * (n - 1)))
        i0 = max(0, min(n - 2, i0))
        i1 = max(i0 + 1, min(n - 1, i1))
        n_vis = i1 - i0 + 1

        # Kernel in screen pixels, so it tracks the zoom: wide in bins when
        # zoomed out (anti-aliases bins sharing a pixel), narrow when zoomed in
        # (keeps real structure). The Bezier outline covers the rest.
        sigma = max(0.6, _SMOOTH_PX * n_vis / float(w))
        rad = int(min(128, max(1, round(3.0 * sigma))))
        kern = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
        kern /= kern.sum()

        lo = max(0, i0 - rad)
        hi = min(n - 1, i1 + rad)
        src_x = np.arange(lo, hi + 1, dtype=np.float64)

        m_out = int(max(2, min(n_vis, 2 * w)))
        pos = np.linspace(float(i0), float(i1), m_out)

        for name in self._names():
            h = coll.get(name)
            if h is None or h.size != n:
                continue

            seg = np.asarray(h[lo:hi + 1], dtype=np.float64)
            if self._log:
                seg = np.log10(seg + 1.0)
            seg = np.convolve(np.pad(seg, rad, mode="edge"), kern, mode="valid")

            vals = np.interp(pos, src_x, seg)
            mx = float(vals.max())
            if mx <= 0.0:
                continue
            vals = vals / mx

            bottom = float(rect.bottom())
            pts = []
            for xb, v in zip(pos, vals):
                x = self._px(xb / (n - 1), rect)
                y = bottom - rect.height() * float(v)
                pts.append((float(x), float(y)))

            outline = _smooth_path(pts)
            filled = QPainterPath(outline)
            filled.lineTo(pts[-1][0], bottom)
            filled.lineTo(pts[0][0], bottom)
            filled.closeSubpath()

            col = QColor(_HV_CH_COLORS[name])
            f = QColor(col); f.setAlpha(fill_a)
            l = QColor(col); l.setAlpha(line_a)
            p.fillPath(filled, f)
            p.setPen(QPen(l, 1))
            p.drawPath(outline)

    def _draw_curve(self, p, rect):
        p.setPen(QPen(QColor(120, 120, 120, 110), 1, Qt.PenStyle.DashLine))
        p.drawLine(rect.left(),  int(rect.bottom() - rect.height() * self._x0),
                   rect.right(), int(rect.bottom() - rect.height() * self._x1))
        if self._curve is None:
            return
        c = np.asarray(self._curve, dtype=np.float32)
        if c.size < 2:
            return
        # sample the VISIBLE range, otherwise a zoomed view keeps only a
        # handful of samples and the curve vanishes at the zoom floor
        m = int(max(2, min(2 * max(1, rect.width()), _CURVE_POINTS)))
        xs = np.linspace(self._x0, self._x1, m)
        idx = np.clip((xs * (c.size - 1)).astype(np.int64), 0, c.size - 1)
        ys = np.clip(c[idx], 0.0, 1.0)

        bottom = float(rect.bottom()); h = rect.height()
        path = QPainterPath()
        path.moveTo(float(self._px(xs[0], rect)), float(bottom - h * float(ys[0])))
        for i in range(1, m):
            path.lineTo(float(self._px(xs[i], rect)), float(bottom - h * float(ys[i])))
        col = QColor(_HV_CH_COLORS[self._channel]); col.setAlpha(235)
        p.setPen(QPen(col, 2)); p.drawPath(path)

    def _draw_markers(self, p, rect):
        for name, val in self._markers.items():
            if val is None or not self._vis(val):
                continue
            col = _HV_MARKER_COLORS.get(name, QColor(200, 200, 200))
            x = self._px(float(np.clip(val, 0.0, 1.0)), rect)
            p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
            p.drawLine(int(x), rect.top(), int(x), rect.bottom())
            p.setPen(col); p.drawText(int(x) + 3, rect.top() + 12, name)

    def _draw_vlines(self, p, rect):
        if self._vlines is None:
            return
        r, g, b, gray = self._vlines
        items = [(r, _HV_CH_COLORS["K"])] if gray else [
            (r, _HV_CH_COLORS["R"]), (g, _HV_CH_COLORS["G"]), (b, _HV_CH_COLORS["B"])]
        for val, col in items:
            if not self._vis(val):
                continue
            c = QColor(col); c.setAlpha(170)
            x = self._px(float(np.clip(val, 0.0, 1.0)), rect)
            p.setPen(QPen(c, 1, Qt.PenStyle.DotLine))
            p.drawLine(int(x), rect.top(), int(x), rect.bottom())

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self._plot_rect()
        p.fillRect(self.rect(), QColor(7, 7, 7))
        p.fillRect(rect, QColor(0, 0, 0))

        major = QPen(QColor(65, 65, 65), 1, Qt.PenStyle.SolidLine)
        minor = QPen(QColor(42, 42, 42), 1, Qt.PenStyle.DotLine)
        for i in range(11):
            x = rect.left() + rect.width() * i / 10.0
            y = rect.top() + rect.height() * i / 10.0
            p.setPen(major if i in (0, 5, 10) else minor)
            p.drawLine(int(x), rect.top(), int(x), rect.bottom())
            p.drawLine(rect.left(), int(y), rect.right(), int(y))

        self._draw_hists(p, rect, self._before, 18, 90)
        self._draw_hists(p, rect, self._after,  55, 175)
        self._draw_curve(p, rect)
        self._draw_markers(p, rect)
        self._draw_vlines(p, rect)

        if self._x0 > 0.0 or self._x1 < 1.0:
            p.setPen(QColor(190, 190, 190, 190))
            p.drawText(rect.left() + 6, rect.bottom() - 5,
                       f"x: {self._x0:.5f} \u2013 {self._x1:.5f}")
        p.setPen(QPen(QColor(115, 115, 115), 1))
        p.drawRect(rect)
        p.end()


class StretchCurveView(QWidget):
    """
    Histogram + optional transfer curve.

    compact=True  drops the clip readout — use it where the host dialog
                  already reports clipping (CurvesDialogPro does, per channel
                  with counts, in its status line).
    show_curve=False draws histograms only.

    Also exposes a small CurveEditor-shaped facade (setSymmetryPoint /
    clearSymmetryLine / updateValueLines / control_points / curve_item / ...)
    so GhsDialogPro can use it in place of the editor unchanged.
    """

    pivotPicked = pyqtSignal(float)

    control_points = ()
    curve_item = None

    def __init__(self, parent=None, compact=False, show_curve=True, min_canvas_h=None):
        super().__init__(parent)
        self._sym_cb = None
        self._preview_cb = None
        self._is_rgb = False
        self._before = {}
        self._after_direct = None
        self._lut = None
        self._channel = "K"
        self._compact = bool(compact)
        self._show_curve = bool(show_curve)

        if min_canvas_h is None:
            min_canvas_h = 120 if compact else 190
        self.canvas = _HistCanvas(self, min_h=min_canvas_h)

        self.chk_log  = QCheckBox(self.tr("Log"))
        self.chk_log.setToolTip(self.tr("Logarithmic histogram scaling (display only)"))
        self.btn_zout = QPushButton("-");   self.btn_zout.setFixedWidth(24)
        self.btn_zin  = QPushButton("+");   self.btn_zin.setFixedWidth(24)
        self.btn_1to1 = QPushButton("1:1"); self.btn_1to1.setFixedWidth(38)
        self.btn_zout.setToolTip(self.tr("Zoom out (x axis)"))
        self.btn_zin.setToolTip(self.tr("Zoom in (x axis)"))
        self.btn_1to1.setToolTip(self.tr("Reset x axis to full range"))
        if compact:
            for _w in (self.btn_zout, self.btn_zin, self.btn_1to1):
                _w.setMaximumHeight(20)

        head = QHBoxLayout(); head.setContentsMargins(0, 0, 0, 0); head.setSpacing(4)
        head.addWidget(self.chk_log); head.addStretch(1)
        head.addWidget(self.btn_zout); head.addWidget(self.btn_zin); head.addWidget(self.btn_1to1)

        self.pan = QScrollBar(Qt.Orientation.Horizontal)
        self.pan.setRange(0, 10000); self.pan.setPageStep(10000)
        self.pan.setSingleStep(100); self.pan.setEnabled(False)
        self.pan.setToolTip(self.tr("Pan the visible histogram range after zooming"))
        if compact:
            self.pan.setFixedHeight(12)

        self.lbl_clip = QLabel("Clip %: 0.0000 shadows / 0.0000 highlights")
        self.lbl_clip.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        self.lbl_clip.setVisible(not compact)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        lay.addLayout(head)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.pan)
        if not compact:
            lay.addWidget(self.lbl_clip)

        self._pan_syncing = False
        self.chk_log.toggled.connect(self.canvas.set_log)
        self.btn_zin.clicked.connect(lambda: self.canvas.zoom_x(0.80, 0.5))
        self.btn_zout.clicked.connect(lambda: self.canvas.zoom_x(1.25, 0.5))
        self.btn_1to1.clicked.connect(self.canvas.reset_view)
        self.canvas.rangeChanged.connect(self._sync_pan)
        self.pan.valueChanged.connect(self._on_pan)
        self.canvas.pivotPicked.connect(self._on_pivot)

    def sizeHint(self):
        return QSize(340, 160 if self._compact else 300)

    def _sync_pan(self, x0, x1):
        span = max(0.0, float(x1) - float(x0))
        self._pan_syncing = True
        try:
            if span >= 0.999999:
                self.pan.setEnabled(False); self.pan.setPageStep(10000); self.pan.setValue(0)
            else:
                self.pan.setEnabled(True)
                self.pan.setPageStep(max(1, int(round(span * 10000))))
                self.pan.setValue(int(np.clip(round(x0 / max(1e-9, 1.0 - span) * 10000), 0, 10000)))
        finally:
            self._pan_syncing = False

    def _on_pan(self, v):
        if not self._pan_syncing:
            self.canvas.pan_to_fraction(v / 10000.0)

    def _on_pivot(self, u):
        self.pivotPicked.emit(float(u))
        if callable(self._sym_cb):
            try:
                self._sym_cb(float(u), 0.0)
            except Exception:
                pass

    # -- primary API
    def set_reference_image(self, img):
        """The unmodified image. Call once whenever the document changes."""
        if img is None:
            self._is_rgb, self._before = False, {}
        else:
            self._is_rgb, self._before = _channel_hists(img, _HIST_SAMPLES_REF)
        self._refresh()

    def set_result(self, result_image=None, lut=None, markers=None, clear_result=False):
        """
        Update the 'after' state in one call, so a caller that has both a LUT
        and a processed image doesn't trigger two repaints per tick.

        result_image wins over lut for the histogram; lut still drives the
        transfer curve and the clip readout.
        """
        if result_image is not None:
            _rgb, self._after_direct = _channel_hists(result_image, _HIST_SAMPLES_LIVE)
        elif clear_result:
            self._after_direct = None
        if lut is not None:
            self._lut = np.asarray(lut, dtype=np.float32)
        elif clear_result:
            self._lut = None
        if markers is not None:
            self.canvas.set_markers(markers)
        self._refresh()

    def set_lut(self, lut01, markers=None):
        """LUT-only path (GHS)."""
        self._lut = None if lut01 is None else np.asarray(lut01, dtype=np.float32)
        self._after_direct = None
        if markers is not None:
            self.canvas.set_markers(markers)
        self._refresh()

    def set_markers(self, m):
        self.canvas.set_markers(m)

    def set_curve_visible(self, on: bool):
        self._show_curve = bool(on)
        self._refresh()

    def set_channel(self, ch):
        """Accepts 'K', 'K (Brightness)', 'R', 'G', 'B'. Anything else -> K."""
        self._channel = _HV_CHANNEL_ALIASES.get(str(ch or "K").strip().upper(), "K")
        self.canvas.set_channel(self._channel)
        self._refresh()

    def _refresh(self):
        if not self._before:
            self.canvas.set_histograms(self._is_rgb, {}, {})
            self.canvas.set_curve(None)
            return

        self.canvas.set_curve(self._lut if (self._show_curve and self._lut is not None) else None)

        if self._after_direct is not None:
            after = dict(self._after_direct)
        elif self._lut is not None:
            touched = ("R", "G", "B") if self._channel == "K" else (self._channel,)
            after = {}
            for name in ("R", "G", "B", "K"):
                h = self._before.get(name)
                if h is None:
                    continue
                after[name] = _remap_hist(h, self._lut) if (name in touched or name == "K") else h
        else:
            after = {}

        self.canvas.set_histograms(self._is_rgb, self._before, after)

        if not self._compact and self._lut is not None:
            ref = self._before.get("K", self._before.get("R"))
            if ref is not None:
                lo, hi = _clip_pct(ref, self._lut)
                self.lbl_clip.setText(f"Clip %: {lo:.4f} shadows / {hi:.4f} highlights")

    # -- CurveEditor compatibility facade
    def setPreviewCallback(self, cb):   self._preview_cb = cb
    def setSymmetryCallback(self, cb):  self._sym_cb = cb
    def initCurve(self):                pass
    def updateCurve(self):              pass
    def addControlPoint(self, *a, **k): pass
    def getCurveFunction(self):         return None

    def setSymmetryPoint(self, x360, _y=0.0):
        m = dict(self.canvas._markers)
        m["SP"] = float(np.clip(float(x360) / 360.0, 0.0, 1.0))
        self.canvas.set_markers(m)

    def clearSymmetryLine(self):
        m = dict(self.canvas._markers); m.pop("SP", None)
        self.canvas.set_markers(m)

    def updateValueLines(self, r, g, b, grayscale=False):
        self.canvas.set_value_lines(r, g, b, grayscale)

    def clearValueLines(self):
        self.canvas.clear_value_lines()

# ── misc helpers (unchanged from original) ────────────────────────────────────

def _downsample_for_preview(img01: np.ndarray, max_w: int = 1200) -> np.ndarray:
    h, w = img01.shape[:2]
    if w <= max_w:
        return img01.copy()
    s = max_w / float(w)
    new_w, new_h = max_w, int(round(h * s))
    u8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    try:
        import cv2
        out = cv2.resize(u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        y_idx = np.linspace(0, h-1, new_h).astype(np.int32)
        x_idx = np.linspace(0, w-1, new_w).astype(np.int32)
        out = u8[y_idx][:, x_idx]
    return out.astype(np.float32) / 255.0


def build_curve_lut(curve_func, size=65536):
    x = np.linspace(0.0, 360.0, size, dtype=np.float32)
    y = 360.0 - curve_func(x)
    return (y / 360.0).clip(0.0, 1.0).astype(np.float32)


def _apply_lut_float01_channel(ch, lut01):
    idx = np.clip((ch * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
    return lut01[idx]

def _apply_lut_rgb(img01, lut01):
    try:
        return _nb_apply_lut_color(img01.astype(np.float32, copy=False), lut01.astype(np.float32, copy=False))
    except Exception:
        idx = np.clip((img01 * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
        return lut01[idx]

def _np_apply_lut_channel(ch, lut01):
    idx = np.clip((ch * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
    return lut01[idx]

def _np_apply_lut_rgb(img01, lut01):
    idx = np.clip((img01 * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
    return lut01[idx]

# color-space fallbacks (vectorized NumPy) — identical to original
_M_rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_M_xyz2rgb = np.array([[ 3.2404542,-1.5371385,-0.4985314],
                       [-0.9692660, 1.8760108, 0.0415560],
                       [ 0.0556434,-0.2040259, 1.0572252]], dtype=np.float32)
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883
_delta  = 6.0/29.0
_delta3 = _delta**3
_kappa  = 24389.0/27.0
_eps    = 216.0/24389.0

def _np_rgb_to_xyz(rgb01):
    return (rgb01.reshape(-1,3) @ _M_rgb2xyz.T).reshape(rgb01.shape)

def _np_xyz_to_rgb(xyz):
    return np.clip((xyz.reshape(-1,3) @ _M_xyz2rgb.T).reshape(xyz.shape), 0.0, 1.0)

def _f_lab_np(t):
    return np.where(t > _delta3, np.cbrt(t), (t / (3*_delta*_delta)) + (4.0/29.0))

def _f_lab_inv_np(ft):
    return np.where(ft > _delta, ft**3, 3*_delta*_delta*(ft - 4.0/29.0))

def _np_xyz_to_lab(xyz):
    fx = _f_lab_np(xyz[...,0]/_Xn)
    fy = _f_lab_np(xyz[...,1]/_Yn)
    fz = _f_lab_np(xyz[...,2]/_Zn)
    return np.stack([116*fy-16, 500*(fx-fy), 200*(fy-fz)], axis=-1).astype(np.float32)

def _np_lab_to_xyz(lab):
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    fy = (L+16)/116.0
    X = _Xn * _f_lab_inv_np(fy + a/500.0)
    Y = _Yn * _f_lab_inv_np(fy)
    Z = _Zn * _f_lab_inv_np(fy - b/200.0)
    return np.stack([X,Y,Z], axis=-1).astype(np.float32)

def _np_rgb_to_hsv(rgb01):
    r,g,b = rgb01[...,0], rgb01[...,1], rgb01[...,2]
    cmax = np.maximum.reduce([r,g,b])
    cmin = np.minimum.reduce([r,g,b])
    delta = cmax - cmin
    H = np.zeros_like(cmax, dtype=np.float32)
    mask = delta != 0
    mr = mask & (cmax==r); mg = mask & (cmax==g); mb = mask & (cmax==b)
    H[mr] = ((g[mr]-b[mr])/delta[mr]) % 6.0
    H[mg] = (b[mg]-r[mg])/delta[mg] + 2.0
    H[mb] = (r[mb]-g[mb])/delta[mb] + 4.0
    H = (H * 60.0).astype(np.float32)
    S = np.zeros_like(cmax, dtype=np.float32)
    nz = cmax != 0
    S[nz] = (delta[nz]/cmax[nz]).astype(np.float32)
    return np.stack([H, S, cmax.astype(np.float32)], axis=-1)

def _np_hsv_to_rgb(hsv):
    H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]
    C = V*S; hh = (H/60.0) % 6.0; X = C*(1-np.abs(hh%2-1)); m = V-C
    z = np.zeros_like(H, dtype=np.float32)
    def _sel(a,b,c,d,e,f):
        return np.where((0<=hh)&(hh<1),a,np.where((1<=hh)&(hh<2),b,
               np.where((2<=hh)&(hh<3),c,np.where((3<=hh)&(hh<4),d,
               np.where((4<=hh)&(hh<5),e,f)))))
    r = _sel(C,X,z,z,X,C); g = _sel(X,C,C,X,z,z); b = _sel(z,z,X,C,C,X)
    return np.clip(np.stack([r+m,g+m,b+m],axis=-1), 0,1).astype(np.float32)


# ── Worker thread (unchanged) ─────────────────────────────────────────────────

class _CurvesWorker(QThread):
    done = pyqtSignal(object)

    def __init__(self, image01, luts, invoker=None):
        super().__init__()
        self.image01 = np.ascontiguousarray(image01.astype(np.float32, copy=False))
        self._legacy_single = False
        self._invoker = None

        if isinstance(luts, str):
            mode_str = luts
            lut01 = np.ascontiguousarray(invoker.astype(np.float32, copy=False))
            key = {"K (Brightness)":"K","K":"K","R":"R","G":"G","B":"B"}.get(mode_str, "K")
            self.luts = {key: lut01}
            self._legacy_single = True
            return

        if isinstance(luts, np.ndarray):
            self.luts = {"K": np.ascontiguousarray(luts.astype(np.float32, copy=False))}
            self._legacy_single = True
            return

        self.luts = {k: np.ascontiguousarray(v.astype(np.float32, copy=False))
                     for k, v in luts.items()}
        self._invoker = invoker

    def run(self):
        if self._legacy_single:
            out = self.image01
            if out.ndim == 2 or (out.ndim == 3 and out.shape[2] == 1):
                lut = (self.luts.get("K") or self.luts.get("R")
                       or self.luts.get("G") or self.luts.get("B"))
                if lut is not None:
                    idx = np.clip((out*(len(lut)-1)).astype(np.int32), 0, len(lut)-1)
                    out = lut[idx]
                self.done.emit(out.astype(np.float32, copy=False))
                return
            out = out.copy()
            lutK = self.luts.get("K")
            for ci, k in enumerate(("R","G","B")):
                lut = self.luts.get(k, lutK)
                if lut is not None:
                    idx = np.clip((out[...,ci]*(len(lut)-1)).astype(np.int32), 0, len(lut)-1)
                    out[...,ci] = lut[idx]
            self.done.emit(out.astype(np.float32, copy=False))
            return

        if self._invoker is None:
            self.done.emit(self.image01)
            return
        self.done.emit(self._invoker._apply_all_curves_once(self.image01, self.luts))


# ── CurvesDialogPro (unchanged except CommaToDotLineEdit removed — unused) ────

class CommaToDotLineEdit(QLineEdit):
    def keyPressEvent(self, event: QKeyEvent):
        if event.text() == "," and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Period, event.modifiers(), ".")
        super().keyPressEvent(event)


class CurvesDialogPro(QDialog):
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Curves Editor"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._main = parent
        self.doc = document

        self._follow_conn = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._follow_conn = True
            except Exception:
                pass
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.finished.connect(self._cleanup_connections)
        self._preview_img  = None
        self._full_img     = None
        self._pix          = None
        self._zoom         = 0.25
        self._panning      = False
        self._pan_start    = QPointF()
        self._did_initial_fit  = False
        self._apply_when_ready = False
        self._preview_orig = None
        self._preview_proc = None
        self._show_proc    = False
        self._cdf          = None
        self._cdf_bins     = 1024
        self._cdf_total    = 0
        self._curve_debounce_ms = 120
        self._curve_debounce    = QTimer(self)
        self._curve_debounce.setSingleShot(True)
        self._curve_debounce.timeout.connect(self._rebuild_preview_from_curve_debounced)
        self._curve_gen        = 0
        self._clip_scale       = 1.0
        self._cdf_total_full   = 0
        self._cdf_total_preview = 0

        # --- UI ---
        main = QVBoxLayout(self)
        top  = QHBoxLayout()

        left = QVBoxLayout()
        self.editor = CurveEditor(self)
        left.addWidget(self.editor)
        # compact: CurvesDialogPro already reports clipping per channel with
        # counts in lbl_status, so the widget's own readout is redundant.
        self.hist = StretchCurveView(self, compact=True)
        self.hist.setMinimumHeight(190)
        self.hist.setMaximumHeight(240)
        left.addWidget(self.hist, 0)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        row1 = QHBoxLayout()
        for m in ("K (Brightness)", "R", "G", "B"):
            rb = QRadioButton(m, self)
            if m == "K (Brightness)":
                rb.setChecked(True)
            self.mode_group.addButton(rb)
            row1.addWidget(rb)

        row2 = QHBoxLayout()
        for m in ("L*", "a*", "b*", "Chroma", "Saturation"):
            rb = QRadioButton(m, self)
            self.mode_group.addButton(rb)
            row2.addWidget(rb)

        left.addLayout(row1)
        left.addLayout(row2)

        self._mode_key_map = {
            "K (Brightness)":"K", "R":"R", "G":"G", "B":"B",
            "L*":"L*", "a*":"a*", "b*":"b*", "Chroma":"Chroma", "Saturation":"Saturation"
        }
        self._curves_store     = {k: [(0.0,0.0),(1.0,1.0)] for k in self._mode_key_map.values()}
        self._current_mode_key = "K"

        for b in self.mode_group.buttons():
            b.toggled.connect(self._on_mode_toggled)

        rowp = QHBoxLayout()
        self.btn_presets = QToolButton(self)
        self.btn_presets.setText(self.tr("Presets"))
        self.btn_presets.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        rowp.addWidget(self.btn_presets)

        self.btn_save_preset = QToolButton(self)
        self.btn_save_preset.setText(self.tr("Save as Preset..."))
        self.btn_save_preset.clicked.connect(self._save_current_as_preset)
        rowp.addWidget(self.btn_save_preset)
        left.addLayout(rowp)

        self.lbl_status = QLabel("", self)
        self.lbl_status.setStyleSheet("color: gray;")

        rowb = QHBoxLayout()
        self.btn_preview = QToolButton(self)
        self.btn_preview.setText(self.tr("Toggle Preview"))
        self.btn_preview.setCheckable(True)
        self.btn_apply   = QPushButton(self.tr("Apply to Document"))
        self.btn_reset   = QToolButton()
        self.btn_reset.setText(self.tr("Reset"))
        rowb.addWidget(self.btn_preview)
        rowb.addWidget(self.btn_apply)
        rowb.addWidget(self.btn_reset)
        left.addLayout(rowb)
        left.addStretch(1)

        # ── Drag-to-canvas grip (PI-style "new instance") ─────────────────
        # After the stretch → pins to the lower-left corner.
        # Deferred import avoids any shortcuts.py <-> curve_editor_pro cycle.
        from setiastro.saspro.shortcuts import PresetDragHandle
        try:
            from setiastro.saspro.resources import curves_path
            _cv_icon = QIcon(curves_path)
        except Exception:
            _cv_icon = QIcon()

        drag_row = QHBoxLayout()
        drag_row.setContentsMargins(0, 0, 0, 0)
        self.preset_drag_handle = PresetDragHandle(
            "curves",
            self._curves_params,
            icon=_cv_icon,
            tooltip=self.tr(
                "Drag to the canvas to create a Curves shortcut with ALL current\n"
                "channel curves baked in.\n"
                "Drop directly on an image to apply them headlessly."
            ),
            parent=self,
        )
        drag_row.addWidget(self.preset_drag_handle)
        drag_row.addStretch(1)
        left.addLayout(drag_row)

        top.addLayout(left, 0)

        right = QVBoxLayout()
        zoombar = QHBoxLayout()
        zoombar.addStretch(1)
        self.btn_zoom_out = themed_toolbtn("zoom-out",      self.tr("Zoom Out"))
        self.btn_zoom_in  = themed_toolbtn("zoom-in",       self.tr("Zoom In"))
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))
        zoombar.addWidget(self.btn_zoom_out)
        zoombar.addWidget(self.btn_zoom_in)
        zoombar.addWidget(self.btn_zoom_fit)
        right.addLayout(zoombar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.label = ImageLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.mouseMoved.connect(self._on_preview_mouse_moved)
        self.label.installEventFilter(self)
        self.scroll.setWidget(self.label)
        right.addWidget(self.scroll, 1)
        top.addLayout(right, 1)
        main.addLayout(top, 1)

        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        main.addWidget(sep)

        status_row = QHBoxLayout()
        self.lbl_status = QLabel("", self)
        self.lbl_status.setObjectName("curvesStatus")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_status.setStyleSheet("color: #bbb;")
        line_h = self.fontMetrics().height()
        self.lbl_status.setMaximumHeight(int(line_h * 2.2))
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        status_row.addWidget(self.lbl_status, 1)
        main.addLayout(status_row, 0)

        # wire
        self.btn_preview.clicked.connect(self._run_preview)
        self.btn_preview.toggled.connect(self._toggle_preview)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self._reset_curve)
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_fit.clicked.connect(self._fit)

        self.editor.setPreviewCallback(self._on_editor_curve_changed)
        self._load_from_doc()
        QTimer.singleShot(0, self._fit_after_load)
        self.editor.setSymmetryCallback(self._on_symmetry_pick)
        self.btn_preview.setChecked(True)

        self.main_window = self._find_main_window()
        self.source_view = None
        try:
            if hasattr(self.parent(), "view"):
                self.source_view = self.parent().view
            elif hasattr(self.parent(), "doc_view"):
                self.source_view = self.parent().doc_view
        except Exception:
            pass

        self._rebuild_presets_menu()

    # ── all methods below are identical to the original ──────────────────────
    # (only the CurveEditor widget changed; dialog logic is untouched)

    def _on_editor_curve_changed(self, _lut8=None):
        try:
            self._curves_store[self._current_mode_key] = self._editor_points_norm()
        except Exception:
            pass
        self._refresh_overlays()
        self._curve_gen += 1
        self._curve_debounce.start(self._curve_debounce_ms)

    def _rebuild_preview_from_curve_debounced(self):
        if self._preview_orig is None and self._preview_img is None:
            return
        if not getattr(self, "btn_preview", None) or not self.btn_preview.isChecked():
            return
        self._quick_preview()

    def _active_mode_key(self) -> str:
        for b in self.mode_group.buttons():
            if b.isChecked():
                return self._mode_key_map.get(b.text(), "K")
        return "K"

    def _editor_points_norm(self):
        return self._collect_points_norm_from_editor()

    def _editor_set_from_norm(self, ptsN):
        pts_scene = _points_norm_to_scene(ptsN)
        # separate endpoints from interior handles
        eps_thresh = 1e-6
        # the two scene endpoints should be closest to (0,360) and (360,0)
        # Just pass interior handles to setControlHandles; endpoints are fixed
        filt = [(x, y) for (x, y) in pts_scene if x > eps_thresh and x < 360 - eps_thresh]
        # Also push endpoints if they've been moved (black/white clipping)
        # Find the scene points nearest to x=0 and x=360
        sorted_pts = sorted(pts_scene, key=lambda t: t[0])
        if sorted_pts:
            bx, by = sorted_pts[0]
            wx, wy = sorted_pts[-1]
            # Update editor endpoints
            self.editor._pts[0]  = [float(bx), float(by)]
            self.editor._pts[-1] = [float(wx), float(wy)]
        self.editor.setControlHandles(filt)
        self.editor.updateCurve()

    def _on_mode_toggled(self, checked: bool):
        if not checked:
            return
        prev = self._current_mode_key
        try:
            self._curves_store[prev] = self._editor_points_norm()
        except Exception:
            pass
        key = self._active_mode_key()
        self._current_mode_key = key
        try:
            self.hist.set_channel(key)
        except Exception:
            pass
        self._editor_set_from_norm(self._curves_store.get(key, [(0.0,0.0),(1.0,1.0)]))
        self._refresh_overlays()
        self._quick_preview()

    def _refresh_overlays(self):
        overlays = {}
        for key, ptsN in self._curves_store.items():
            if not ptsN:
                continue
            pts_scene = _points_norm_to_scene(ptsN)
            overlays[key] = pts_scene
        self.editor.setOverlayCurves(overlays, self._current_mode_key)

    _GEOM_KEY = "ui/curves_dialog/geometry"

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value(self._GEOM_KEY, None, type=QByteArray)
            if g and isinstance(g, QByteArray) and not g.isEmpty():
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue(self._GEOM_KEY, self.saveGeometry())
        except Exception:
            pass

    def _lut01_from_points_norm(self, ptsN, size=65536):
        pts_scene = _points_norm_to_scene(ptsN)
        if len(pts_scene) < 2:
            return np.linspace(0.0, 1.0, size, dtype=np.float32)
        xs = np.array([p[0] for p in pts_scene], dtype=np.float64)
        ys = np.array([p[1] for p in pts_scene], dtype=np.float64)
        if np.any(np.diff(xs) <= 0):
            xs += np.linspace(0, 1e-3, len(xs), dtype=np.float64)
        ys = 360.0 - ys
        inp = np.linspace(0.0, 360.0, size, dtype=np.float64)
        try:
            f   = PchipInterpolator(xs, ys, extrapolate=True)
            out = f(inp)
        except Exception:
            out = np.interp(inp, xs, ys)
        return np.clip(out / 360.0, 0.0, 1.0).astype(np.float32)

    def _build_all_active_luts(self):
        luts: dict[str, np.ndarray] = {}
        active_key = self._current_mode_key
        fn = getattr(self.editor, "getCurveFunction", None)
        if callable(fn):
            f = fn()
            if f is not None:
                luts[active_key] = build_curve_lut(f, size=65536)
        for key, pts in self._curves_store.items():
            if key == active_key:
                continue
            if isinstance(pts, (list, tuple)) and len(pts) == 2 and pts[0] == (0.0,0.0) and pts[1] == (1.0,1.0):
                continue
            luts[key] = self._lut01_from_points_norm(pts, size=65536)
        return luts

    def _remember_as_last_action(self):
        mw = self._find_main_window()
        if mw is None:
            return
        btn = self.mode_group.checkedButton() if hasattr(self, "mode_group") else None
        mode_label = btn.text() if btn is not None else "K (Brightness)"
        mode_label = _norm_mode(mode_label)
        handles = self.editor.getControlHandles()
        pts_scene = [(float(x), float(y)) for x, y in handles]
        if not pts_scene:
            pts_scene = [(0.0, 360.0), (360.0, 0.0)]
        pts_scene = _sanitize_scene_points(pts_scene)
        core_preset = {"mode": mode_label, "shape": "custom", "amount": 1.0, "points_scene": pts_scene}
        op = None
        try:
            op = self.export_preview_ops()
        except Exception:
            pass
        if op:
            core_preset["_ops"] = op
        try:
            mw._remember_last_headless_command("curves", core_preset, description="Curves")
            source_view = getattr(self, "source_view", None)
            if hasattr(mw, "_update_replay_button"):
                mw._update_replay_button(source_view)
        except Exception as e:
            print("Curves: failed to remember last action:", e)

    def _apply_all_curves_once(self, img01, luts):
        out = img01
        if out.ndim == 2:
            lutK = luts.get("K")
            if lutK is not None:
                out = _np_apply_lut_channel(out, lutK)
            return np.clip(out, 0.0, 1.0).astype(np.float32)

        def _compose(a, b):
            if a is None: return b
            if b is None: return a
            N = len(a)
            return b[np.clip((a*(N-1)).astype(np.int32), 0, N-1)]

        lutK = luts.get("K")
        lutR = _compose(lutK, luts.get("R"))
        lutG = _compose(lutK, luts.get("G"))
        lutB = _compose(lutK, luts.get("B"))

        if lutR is None and lutG is None and lutB is None and lutK is not None:
            out = _np_apply_lut_rgb(out, lutK)
        else:
            out = out.copy()
            for ci, lut in ((0,lutR),(1,lutG),(2,lutB)):
                if lut is not None:
                    out[...,ci] = _np_apply_lut_channel(out[...,ci], lut)
                elif lutK is not None:
                    out[...,ci] = _np_apply_lut_channel(out[...,ci], lutK)

        need_lab = any(k in luts for k in ("L*","a*","b*","Chroma"))
        if need_lab:
            xyz = _np_rgb_to_xyz(out); lab = _np_xyz_to_lab(xyz)
            if "L*" in luts:
                L = np.clip(lab[...,0]/100.0, 0.0, 1.0)
                L = _np_apply_lut_channel(L, luts["L*"]); lab[...,0] = L*100.0
            if "a*" in luts:
                a = lab[...,1]; an = np.clip((a+128.0)/255.0, 0.0, 1.0)
                an = _np_apply_lut_channel(an, luts["a*"]); lab[...,1] = an*255.0-128.0
            if "b*" in luts:
                b = lab[...,2]; bn = np.clip((b+128.0)/255.0, 0.0, 1.0)
                bn = _np_apply_lut_channel(bn, luts["b*"]); lab[...,2] = bn*255.0-128.0
            if "Chroma" in luts:
                a = lab[...,1]; b = lab[...,2]
                C = np.sqrt(a*a+b*b); Cn = np.clip(C/200.0, 0.0, 1.0)
                Cn = _np_apply_lut_channel(Cn, luts["Chroma"]); Cnew = Cn*200.0
                ratio = np.divide(Cnew, C, out=np.ones_like(Cnew), where=(C>0))
                lab[...,1] = a*ratio; lab[...,2] = b*ratio
            out = _np_xyz_to_rgb(_np_lab_to_xyz(lab))

        if "Saturation" in luts:
            hsv = _np_rgb_to_hsv(out)
            S = np.clip(hsv[...,1], 0.0, 1.0)
            hsv[...,1] = _np_apply_lut_channel(S, luts["Saturation"])
            out = _np_hsv_to_rgb(hsv)

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _fit_after_load(self, tries=0):
        if self._did_initial_fit:
            return
        if not self.isVisible():
            QTimer.singleShot(0, lambda: self._fit_after_load(tries))
            return
        pm = self.label.pixmap()
        vp = self.scroll.viewport() if hasattr(self, "scroll") else None
        have_pm    = bool(pm and not pm.isNull())
        have_sizes = bool(vp and vp.width() > 0 and vp.height() > 0)
        if not (self._pix and have_pm and have_sizes):
            if tries < 20:
                QTimer.singleShot(15, lambda: self._fit_after_load(tries+1))
            return
        self._did_initial_fit = True
        self._fit()

    def _capture_view(self):
        try:
            vp = self.scroll.viewport()
            h  = self.scroll.horizontalScrollBar()
            v  = self.scroll.verticalScrollBar()
            lw = max(1, self.label.width())
            lh = max(1, self.label.height())
            cx = h.value() + vp.width()  / 2.0
            cy = v.value() + vp.height() / 2.0
            return float(cx)/float(lw), float(cy)/float(lh), float(self._zoom)
        except Exception:
            return 0.5, 0.5, float(self._zoom)

    def _restore_view(self, fx, fy, zoom):
        self._set_zoom(zoom)
        vp = self.scroll.viewport()
        h  = self.scroll.horizontalScrollBar()
        v  = self.scroll.verticalScrollBar()
        cx = int(round(fx * max(1, self.label.width())))
        cy = int(round(fy * max(1, self.label.height())))
        h.setValue(max(h.minimum(), min(h.maximum(), cx - vp.width()//2)))
        v.setValue(max(v.minimum(), min(v.maximum(), cy - vp.height()//2)))

    def _build_preview_luma_cdf(self):
        img  = self._preview_img
        bins = int(getattr(self, "_cdf_bins", 1024))
        self._cdf_bins = bins
        self._cdf = None; self._cdf_total = 0
        self._cdf_total_preview = 0; self._cdf_total_full = 0; self._clip_scale = 1.0
        if img is None:
            return
        luma = img if img.ndim == 2 else (
            img[...,0] if img.ndim == 3 and img.shape[2] == 1 else
            (0.2126*img[...,0]+0.7152*img[...,1]+0.0722*img[...,2]).astype(np.float32)
        )
        luma = np.clip(luma, 0.0, 1.0)
        hist, _ = np.histogram(luma, bins=bins, range=(0.0,1.0))
        self._cdf = np.cumsum(hist).astype(np.int64)
        self._cdf_total_preview = int(luma.size)
        self._cdf_total = self._cdf_total_preview
        full_pixels = int(self._full_img.shape[0]*self._full_img.shape[1]) \
                      if isinstance(getattr(self,"_full_img",None), np.ndarray) else 0
        if full_pixels <= 0:
            full_pixels = self._cdf_total_preview
        self._cdf_total_full = full_pixels
        self._clip_scale = full_pixels / float(self._cdf_total_preview) if self._cdf_total_preview else 1.0

    def _build_preview_rgb_cdfs(self):
        self._cdf_rgb = None
        img = self._preview_img
        if img is None or not (img.ndim == 3 and img.shape[2] >= 3):
            return
        bins = int(getattr(self, "_cdf_bins", 1024))
        cdfs = {}
        for ci, k in enumerate(("r","g","b")):
            ch = np.clip(img[...,ci].astype(np.float32), 0.0, 1.0)
            h, _ = np.histogram(ch, bins=bins, range=(0.0,1.0))
            cdfs[k] = np.cumsum(h).astype(np.int64)
        cdfs["total_preview"] = int(img[...,0].size)
        self._cdf_rgb = cdfs

    def _on_symmetry_pick(self, u, _v):
        self.editor.redistributeHandlesByPivot(u)
        self._set_status(self.tr("Inflection @ K={0:.3f}").format(u))
        self._quick_preview()

    def _fit_once(self):
        if not self._did_initial_fit:
            self._fit_after_load(0)

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True
            def _after():
                self._restore_window_geometry()
                self._fit()
            QTimer.singleShot(0, _after)
            return
        QTimer.singleShot(0, self._fit_after_load)

    def _on_preview_mouse_moved(self, x, y):
        if self._preview_img is None:
            return
        mapped = self._map_label_xy_to_image_ij(x, y)
        if not mapped:
            self.editor.clearValueLines()
            self.hist.clearValueLines()
            self._set_status("")
            return
        img = self._preview_img
        H, W = img.shape[:2]
        try:
            ix, iy = mapped
            ix = max(0, min(W-1, int(round(ix))))
            iy = max(0, min(H-1, int(round(iy))))
        except Exception:
            self.editor.clearValueLines(); self.hist.clearValueLines()
            self._set_status(""); return

        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            v = float(img[iy,ix] if img.ndim == 2 else img[iy,ix,0])
            v = float(np.clip(0.0 if not np.isfinite(v) else v, 0.0, 1.0))
            self.editor.updateValueLines(v, 0.0, 0.0, grayscale=True)
            self.hist.updateValueLines(v, 0.0, 0.0, grayscale=True)
            self._set_status(self.tr("Cursor ({0}, {1})  K: {2:.3f}").format(ix,iy,v))
        else:
            C = img.shape[2]
            r = g = b = 0.0
            if C >= 3:
                r,g,b = img[iy,ix,0], img[iy,ix,1], img[iy,ix,2]
            elif C >= 1:
                r = g = b = img[iy,ix,0]
            r = float(np.clip(r if np.isfinite(r) else 0.0, 0.0, 1.0))
            g = float(np.clip(g if np.isfinite(g) else 0.0, 0.0, 1.0))
            b = float(np.clip(b if np.isfinite(b) else 0.0, 0.0, 1.0))
            self.editor.updateValueLines(r, g, b, grayscale=False)
            self.hist.updateValueLines(r, g, b, grayscale=False)
            self._set_status(self.tr("Cursor ({0}, {1})  R: {2:.3f}  G: {3:.3f}  B: {4:.3f}").format(ix,iy,r,g,b))

    def _map_label_xy_to_image_ij(self, x, y):
        if self._pix is None:
            return None
        pm_disp = self.label.pixmap()
        if pm_disp is None or pm_disp.isNull():
            return None
        disp_w = pm_disp.width(); disp_h = pm_disp.height()
        lbl_w  = self.label.width(); lbl_h = self.label.height()
        off_x  = max(0, (lbl_w - disp_w) // 2)
        off_y  = max(0, (lbl_h - disp_h) // 2)
        px = float(x) - float(off_x)
        py = float(y) - float(off_y)
        if px < 0 or py < 0 or px >= disp_w or py >= disp_h:
            return None
        src_w = self._pix.width(); src_h = self._pix.height()
        if src_w <= 0 or src_h <= 0:
            return None
        ix = int(px / (disp_w / float(src_w)))
        iy = int(py / (disp_h / float(src_h)))
        if ix < 0 or iy < 0 or ix >= src_w or iy >= src_h:
            return None
        return ix, iy

    def _scene_to_norm_points(self, pts_scene):
        out = []
        lastx = -1e9
        for (x, y) in sorted(pts_scene, key=lambda t: t[0]):
            x = float(np.clip(x, 0.0, 360.0))
            y = float(np.clip(y, 0.0, 360.0))
            if x <= lastx:
                x = lastx + 1e-3
            lastx = x
            out.append((x/360.0, 1.0-(y/360.0)))
        if not any(abs(px-0.0) < 1e-6 for px,_ in out): out.insert(0,(0.0,0.0))
        if not any(abs(px-1.0) < 1e-6 for px,_ in out): out.append((1.0,1.0))
        return [(float(np.clip(x,0,1)), float(np.clip(y,0,1))) for (x,y) in out]

    def _collect_points_norm_from_editor(self):
        pts_scene = []
        for p in self.editor.end_points + self.editor.control_points:
            pos = p.scenePos()
            pts_scene.append((float(pos.x()), float(pos.y())))
        out = []
        lastx = -1e9
        for (x, y) in sorted(pts_scene, key=lambda t: t[0]):
            x = float(np.clip(x, 0.0, 360.0))
            y = float(np.clip(y, 0.0, 360.0))
            if x <= lastx:
                x = lastx + 1e-3
            lastx = x
            out.append((x/360.0, 1.0-(y/360.0)))
        return [(float(np.clip(x,0,1)), float(np.clip(y,0,1))) for (x,y) in out]

    def _save_current_as_preset(self):
        name, ok = QInputDialog.getText(self, self.tr("Save Curves Preset"), self.tr("Preset name:"))
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            self._curves_store[self._current_mode_key] = self._editor_points_norm()
        except Exception:
            pass
        modes = {}
        for k, pts in self._curves_store.items():
            if not isinstance(pts, (list,tuple)) or len(pts) < 2:
                continue
            modes[k] = [(float(x),float(y)) for (x,y) in pts]
        preset = {"name":name,"version":2,"kind":"curves_multi",
                  "active":self._current_mode_key,"modes":modes}
        if save_custom_preset(name, preset):
            self._set_status(self.tr('Saved preset "{0}".').format(name))
            self._rebuild_presets_menu()
        else:
            QMessageBox.warning(self, self.tr("Save failed"), self.tr("Could not save preset."))

    def _rebuild_presets_menu(self):
        m = QMenu(self)
        builtins = [
            ("Linear",              {"mode":"K (Brightness)","shape":"linear"}),
            ("S-Curve (mild)",      {"mode":"K (Brightness)","shape":"s_mild","amount":1.0}),
            ("S-Curve (medium)",    {"mode":"K (Brightness)","shape":"s_med","amount":1.0}),
            ("S-Curve (strong)",    {"mode":"K (Brightness)","shape":"s_strong","amount":1.0}),
            ("Lift Shadows",        {"mode":"K (Brightness)","shape":"lift_shadows","amount":1.0}),
            ("Crush Shadows",       {"mode":"K (Brightness)","shape":"crush_shadows","amount":1.0}),
            ("Fade Blacks",         {"mode":"K (Brightness)","shape":"fade_blacks","amount":1.0}),
            ("Rolloff Highlights",  {"mode":"K (Brightness)","shape":"rolloff_highlights","amount":1.0}),
            ("Flatten",             {"mode":"K (Brightness)","shape":"flatten","amount":1.0}),
        ]
        mb = m.addMenu(self.tr("Built-ins"))
        for label, preset in builtins:
            act = mb.addAction(label)
            act.triggered.connect(lambda _=False, p=preset: self._apply_preset_dict(p))
        customs = list_custom_presets()
        if customs:
            mc = m.addMenu(self.tr("Custom"))
            for pp in sorted(customs, key=lambda d: d.get("name","").lower()):
                act = mc.addAction(pp.get("name","(unnamed)"))
                act.triggered.connect(lambda _=False, pp=pp: self._apply_preset_dict(pp))
            mc.addSeparator()
            mc.addAction(self.tr("Manage…")).triggered.connect(self._open_manage_customs_dialog)
        else:
            m.addAction(self.tr("(No custom presets yet)")).setEnabled(False)
        self.btn_presets.setMenu(m)

    def _open_manage_customs_dialog(self):
        customs = list_custom_presets()
        if not customs:
            QMessageBox.information(self, self.tr("Manage Presets"), self.tr("No custom presets."))
            return
        names = [p.get("name","") for p in customs]
        name, ok = QInputDialog.getItem(self, self.tr("Delete Preset"),
                                        self.tr("Choose preset to delete:"), names, 0, False)
        if ok and name:
            from setiastro.saspro.curves_preset import delete_custom_preset
            if delete_custom_preset(name):
                self._rebuild_presets_menu()

    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._load_from_doc()
        QTimer.singleShot(0, self._fit_after_load)

    def _load_from_doc(self):
        img = self.doc.image
        if img is None:
            QMessageBox.information(self, self.tr("No image"), self.tr("Open an image first."))
            return
        arr = np.asarray(img)
        if arr.dtype.kind in "ui":
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        elif arr.dtype.kind == "f":
            mx = float(arr.max()) if arr.size else 1.0
            arr = (arr / (mx if mx > 1.0 else 1.0)).astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        self._full_img     = arr
        self._preview_img  = _downsample_for_preview(arr, 1200)
        self._preview_orig = self._preview_img.copy()
        self._preview_proc = None
        max_hist_dim = 1400
        h, w = arr.shape[:2]
        if max(h,w) > max_hist_dim:
            step = max(1, int(np.ceil(max(h,w)/max_hist_dim)))
            self._hist_base = arr[::step,::step,...]
        else:
            self._hist_base = arr
        self.hist.set_reference_image(self._hist_base)
        self.hist.set_channel(self._current_mode())
        self._show_proc = True
        self._quick_preview()
        self._update_preview_pix(
            self._preview_proc if self._preview_proc is not None else self._preview_orig,
            preserve_view=False)
        self._build_preview_luma_cdf()
        self._build_preview_rgb_cdfs()

    def _build_lut01(self):
        f = getattr(self.editor, "getCurveFunction", lambda: None)()
        return build_curve_lut(f, size=65536) if f is not None else None

    def _toggle_preview(self, on):
        self._show_proc = bool(on)
        if self._preview_proc is None:
            self._quick_preview()
        img = self._preview_proc if (self._show_proc and self._preview_proc is not None) else self._preview_orig
        self._update_preview_pix(img)
        try:
            if not on:
                self.hist.set_result(clear_result=True)
            else:
                self.hist.set_result(result_image=self._preview_proc)
        except Exception:
            pass
        self._set_status(self.tr("Preview ON") if self._show_proc else self.tr("Preview OFF"))

    def _quick_preview(self):
        if self._preview_img is None:
            return
        luts = self._build_all_active_luts()
        proc = self._apply_all_curves_once(self._preview_img, luts)
        proc = self._blend_with_mask(proc)
        self._preview_proc = proc
        if self._show_proc:
            self._update_preview_pix(self._preview_proc)
        try:
            hist_proc = self._apply_all_curves_once(self._hist_base, luts)
            # result_image rather than a LUT: the composite spans several
            # per-channel LUTs plus Lab/HSV round-trips, so there is no single
            # monotone LUT to push the histogram through. The active channel's
            # LUT still drives the curve overlay.
            self.hist.set_result(result_image=hist_proc,
                                 lut=luts.get(self._current_mode_key))
        except Exception:
            pass
        try:
            bt, wt = self.editor.current_black_white_thresholds()
            if self._preview_img is not None and self._preview_img.ndim == 3 and self._preview_img.shape[2] >= 3:
                rgb = self._clip_counts_rgb_from_thresholds(bt, wt)
                def _fmt(pair):
                    cnt_b,cnt_w,fb,fw = pair
                    return self.tr("Bk {0:,} ({1:.2f}%)  Wt {2:,} ({3:.2f}%)").format(cnt_b,fb*100,cnt_w,fw*100)
                self._set_status(self.tr("Clipping —  R: {0}   G: {1}   B: {2}").format(
                    _fmt(rgb['r']), _fmt(rgb['g']), _fmt(rgb['b'])))
            else:
                below,above,f_below,f_above = self._clip_counts_from_thresholds(bt,wt)
                self._set_status(self.tr("Clipping —  Bk {0:,} ({1:.2f}%)   Wt {2:,} ({3:.2f}%)").format(
                    below,f_below*100,above,f_above*100))
        except Exception:
            pass

    def _run_preview(self):
        if self._full_img is None:
            return
        luts = self._build_all_active_luts()
        self.btn_apply.setEnabled(False)
        self._thr = _CurvesWorker(self._full_img, luts, self)
        self._thr.done.connect(self._on_preview_ready)
        self._thr.finished.connect(lambda: self.btn_apply.setEnabled(True))
        self._thr.start()

    def _on_preview_ready(self, out01):
        self._last_preview = self._blend_with_mask(out01)
        self._set_status(self.tr("Full-res ready (not shown)."))

    def _clip_counts_from_thresholds(self, black_t, white_t):
        if self._cdf is None or getattr(self,"_cdf_total_preview",0) <= 0:
            return 0,0,0.0,0.0
        bins  = int(getattr(self,"_cdf_bins",1024))
        scale = float(getattr(self,"_clip_scale",1.0))
        total_full = int(getattr(self,"_cdf_total_full",self._cdf_total_preview)) or 1
        total_prev = self._cdf_total_preview

        def _idx(t):
            return max(0, min(bins-1, int(np.floor(np.clip(float(t),0,1)*(bins-1)))))

        below_prev = int(self._cdf[_idx(black_t)]) if black_t is not None else 0
        above_prev = int(total_prev - self._cdf[_idx(white_t)]) if white_t is not None else 0
        below_full = max(0, min(total_full, int(round(below_prev*scale))))
        above_full = max(0, min(total_full, int(round(above_prev*scale))))
        return below_full, above_full, below_full/float(total_full), above_full/float(total_full)

    def _clip_counts_rgb_from_thresholds(self, black_t, white_t):
        out = {"r":(0,0,0.0,0.0),"g":(0,0,0.0,0.0),"b":(0,0,0.0,0.0)}
        if getattr(self,"_cdf_rgb",None) is None:
            return out
        bins  = int(getattr(self,"_cdf_bins",1024))
        scale = float(getattr(self,"_clip_scale",1.0))
        total_full = int(getattr(self,"_cdf_total_full",self._cdf_rgb["total_preview"])) or 1
        total_prev = int(self._cdf_rgb["total_preview"]) or 1

        def _idx(t):
            return max(0, min(bins-1, int(np.floor(np.clip(float(t),0,1)*(bins-1)))))

        for ch in ("r","g","b"):
            cdf = self._cdf_rgb[ch]
            below_prev = int(cdf[_idx(black_t)]) if black_t is not None else 0
            above_prev = int(total_prev - cdf[_idx(white_t)]) if white_t is not None else 0
            below_full = max(0, min(total_full, int(round(below_prev*scale))))
            above_full = max(0, min(total_full, int(round(above_prev*scale))))
            out[ch] = (below_full, above_full, below_full/float(total_full), above_full/float(total_full))
        return out

    def _curves_params(self) -> dict:
        """
        Canonical multi-curve preset for the drag handle — identical in shape to
        what _save_current_as_preset persists and what apply_curves_via_preset
        consumes (kind="curves_multi", modes keyed by K/R/G/B/L*/a*/b*/Chroma/
        Saturation, points normalized 0..1).

        CRITICAL: flush the active channel's live editor state into the store
        first, exactly as _save_current_as_preset / export_preview_ops do —
        otherwise mid-edit adjustments on the current channel are lost.
        """
        try:
            self._curves_store[self._current_mode_key] = self._editor_points_norm()
        except Exception:
            pass

        modes = {}
        for k, pts in self._curves_store.items():
            if not isinstance(pts, (list, tuple)) or len(pts) < 2:
                continue
            modes[k] = [(float(x), float(y)) for (x, y) in pts]

        return {
            "command_id": "curves",
            "kind": "curves_multi",
            "version": 2,
            "active": str(self._current_mode_key or "K"),
            "modes": modes,
        }

    def export_preview_ops(self):
        try:
            self._curves_store[self._current_mode_key] = self._editor_points_norm()
        except Exception:
            pass
        def _is_linear(pts):
            return isinstance(pts,(list,tuple)) and len(pts)==2 and pts[0]==(0.0,0.0) and pts[1]==(1.0,1.0)
        modes = {k:[(float(x),float(y)) for (x,y) in pts]
                 for k,pts in self._curves_store.items() if pts and not _is_linear(pts)}
        return {"version":1,"tool":"curves","modes":modes,"active":self._current_mode_key,
                "lut_size":65536,"mask":{"id":getattr(self.doc,"active_mask_id",None),
                                         "blend":"m*out+(1-m)*src"}}

    def _apply(self):
        if not hasattr(self,"_last_preview"):
            luts  = self._build_all_active_luts()
            out01 = self._apply_all_curves_once(self._full_img, luts)
            out01 = self._blend_with_mask(out01)
            self._last_preview = out01
        self._commit(self._last_preview)

    def _commit(self, out01):
        try:
            _marr,mid,mname = self._active_mask_layer()
            meta = {"step_name":"Curves","curves":{"mode":self._current_mode()},
                    "masked":bool(mid),"mask_id":mid,"mask_name":mname,
                    "mask_blend":"m*out + (1-m)*src"}
            try:
                self._remember_as_last_action()
            except Exception:
                pass
            mw = self._find_main_window()
            if mw is not None:
                last = getattr(mw,"_last_headless_command",None) or {}
                if last.get("command_id") == "curves":
                    meta["command_id"] = "curves"
                    meta["preset"] = dict(last.get("preset") or {})
            self.doc.apply_edit(out01.copy(), metadata=meta, step_name="Curves")
            try:
                self._remember_as_last_action()
            except Exception:
                pass
            self.__dict__.pop("_last_preview", None)
            self._full_img = None; self._preview_img = None
            self._load_from_doc()
            if hasattr(self.editor, "clearSymmetryLine"):
                self.editor.clearSymmetryLine()
            self.editor.initCurve()
            for k in list(self._curves_store.keys()):
                self._curves_store[k] = [(0.0,0.0),(1.0,1.0)]
            self._refresh_overlays()
            self._quick_preview()
            self._set_status(self.tr("Applied. Image reloaded. All curves reset — keep tweaking."))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))

    def _current_mode(self):
        for b in self.mode_group.buttons():
            if b.isChecked():
                return b.text()
        return "K (Brightness)"

    def _set_status(self, s):
        self.lbl_status.setText(s)

    def _update_preview_pix(self, img01, preserve_view=True):
        if img01 is None:
            self.label.clear(); self._pix = None; return
        state = self._capture_view() if preserve_view else None
        qimg  = _float_to_qimage_rgb8(img01)
        pm    = QPixmap.fromImage(qimg)
        self._pix = pm
        if preserve_view and state is not None:
            fx,fy,zoom = state
            self._restore_view(fx,fy,zoom)
        else:
            self._apply_zoom()
            if not self._did_initial_fit:
                QTimer.singleShot(0, self._fit_once)

    def _active_mask_layer(self):
        mid = getattr(self.doc,"active_mask_id",None)
        if not mid: return None,None,None
        layer = getattr(self.doc,"masks",{}).get(mid)
        if layer is None: return None,None,None
        m = np.asarray(getattr(layer,"data",None))
        if m is None or m.size == 0: return None,None,None
        m = m.astype(np.float32, copy=False)
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0: m /= mx
        return np.clip(m,0.0,1.0), mid, getattr(layer,"name","Mask")

    def _resample_mask_if_needed(self, mask, out_hw):
        mh,mw = mask.shape[:2]; th,tw = out_hw
        if (mh,mw)==(th,tw): return mask
        yi = np.linspace(0,mh-1,th).astype(np.int32)
        xi = np.linspace(0,mw-1,tw).astype(np.int32)
        return mask[yi][:,xi]

    def _blend_with_mask(self, processed):
        mask,_mid,_mname = self._active_mask_layer()
        if mask is None: return processed
        out = processed.astype(np.float32, copy=False)
        src = self._full_img if (hasattr(self,"_full_img") and self._full_img is not None
                                 and out.shape[:2]==self._full_img.shape[:2]) else self._preview_img
        m = self._resample_mask_if_needed(mask, out.shape[:2])
        if out.ndim == 3 and out.shape[2] == 3: m = m[...,None]
        if src.ndim == 2 and out.ndim == 3: src = np.stack([src]*3, axis=-1)
        elif src.ndim == 3 and out.ndim == 2: src = src[...,0]
        return (m*out + (1.0-m)*src).astype(np.float32, copy=False)

    def closeEvent(self, ev):
        try: self._save_window_geometry()
        except Exception: pass
        self._cleanup_connections()
        super().closeEvent(ev)

    def _cleanup_connections(self):
        try:
            if self._follow_conn and hasattr(self._main,"currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._follow_conn = False
        try:
            thr = getattr(self,"_thr",None)
            if thr is not None:
                for fn in (thr.requestInterruption, thr.quit, lambda: thr.wait(250)):
                    try: fn()
                    except Exception: pass
        except Exception:
            pass
        try: self._thr = None
        except Exception: pass

    def _apply_zoom(self):
        if self._pix is None: return
        scaled = self._pix.scaled(self._pix.size()*self._zoom,
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
        if self._pix.width()==0 or self._pix.height()==0: return
        s = min(vp.width()/self._pix.width(), vp.height()/self._pix.height())
        self._set_zoom(max(0.05, s))

    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel:
                dy = ev.pixelDelta().y()
                if dy != 0:
                    abs_dy = abs(dy)
                    ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    base = 1.012 if abs_dy<=3 else (1.025 if abs_dy<=10 else 1.040)
                    if ctrl: base += 0.015
                    factor = base if dy>0 else 1.0/base
                else:
                    dy = ev.angleDelta().y()
                    if dy == 0: ev.accept(); return True
                    ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    step = 1.25 if ctrl else 1.15
                    factor = step if dy>0 else 1.0/step
                self._set_zoom(self._zoom*factor)
                ev.accept(); return True

            if ev.type()==QEvent.Type.MouseButtonPress and ev.button()==Qt.MouseButton.LeftButton:
                self._panning=True; self._pan_start=ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept(); return True
            if ev.type()==QEvent.Type.MouseMove and self._panning:
                d = ev.position()-self._pan_start
                h=self.scroll.horizontalScrollBar(); v=self.scroll.verticalScrollBar()
                h.setValue(h.value()-int(d.x())); v.setValue(v.value()-int(d.y()))
                self._pan_start=ev.position(); ev.accept(); return True
            if ev.type()==QEvent.Type.MouseButtonRelease and ev.button()==Qt.MouseButton.LeftButton:
                self._panning=False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True

            if ev.type()==QEvent.Type.MouseMove and not self._panning:
                lp = self.label.mapFrom(self.scroll.viewport(),
                                        QPoint(int(ev.position().x()), int(ev.position().y())))
                if 0<=lp.x()<self.label.width() and 0<=lp.y()<self.label.height():
                    self._on_preview_mouse_moved(lp.x(), lp.y())
                else:
                    self.editor.clearValueLines(); self.hist.clearValueLines()
                    self._set_status("")
                return False

            if ev.type()==QEvent.Type.MouseButtonDblClick and ev.button()==Qt.MouseButton.LeftButton:
                if self._preview_img is None or self._pix is None:
                    return False
                pos = self.label.mapFrom(self.scroll.viewport(), ev.pos())
                ix  = int(pos.x() / max(self._zoom, 1e-6))
                iy  = int(pos.y() / max(self._zoom, 1e-6))
                ix  = max(0, min(self._pix.width()-1,  ix))
                iy  = max(0, min(self._pix.height()-1, iy))
                img = self._preview_img
                k   = float(img[iy,ix] if img.ndim==2 else
                            (img[iy,ix,0] if img.shape[2]==1 else np.mean(img[iy,ix,:3])))
                k   = float(np.clip(k, 0.0, 1.0))
                self.editor.setSymmetryPoint(k*360.0, 0.0)
                self._on_symmetry_pick(k, k)
                ev.accept(); return True

        if obj is self.label and ev.type()==QEvent.Type.Leave:
            self.editor.clearValueLines(); self.hist.clearValueLines()
            self._set_status(""); return False

        if obj is self.label and ev.type()==QEvent.Type.MouseButtonDblClick:
            if ev.button() != Qt.MouseButton.LeftButton:
                return False
            pos    = ev.position()
            mapped = self._map_label_xy_to_image_ij(pos.x(), pos.y())
            if not mapped or self._preview_img is None:
                return False
            ix,iy  = mapped
            img    = self._preview_img
            v = float(img[iy,ix] if img.ndim==2 else
                      (img[iy,ix,0] if img.ndim==3 and img.shape[2]==1 else np.mean(img[iy,ix,:3])))
            if np.isnan(v): return True
            v = float(np.clip(v, 0.0, 1.0))
            x = max(0.001, min(359.999, v*360.0))
            y = None
            try:
                f = self.editor.getCurveFunction()
                if f is not None: y = float(f(x))
            except Exception:
                pass
            if y is None: y = 360.0-x
            xs = [p.scenePos().x() for p in (self.editor.end_points+self.editor.control_points)]
            if any(abs(x-ex)<1e-3 for ex in xs):
                step = 0.002
                for ki in range(1,2000):
                    for cand in (x+ki*step, x-ki*step):
                        if 0.001<cand<359.999 and all(abs(cand-ex)>=1e-3 for ex in xs):
                            x=cand; break
                    else:
                        continue
                    break
            self.editor.addControlPoint(x, y)
            self._set_status(self.tr("Added point at x={0:.3f}").format(v))
            ev.accept(); return True

        return super().eventFilter(obj, ev)

    def _reset_curve(self):
        self.editor.initCurve()
        for k in list(self._curves_store.keys()):
            self._curves_store[k] = [(0.0,0.0),(1.0,1.0)]
        self._refresh_overlays()
        self._quick_preview()
        try:
            self.hist.set_result(clear_result=True)
        except Exception:
            pass
        self._set_status(self.tr("All curves reset."))

    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    def _apply_preset_dict(self, preset):
        preset = preset or {}
        if preset.get("kind")=="curves_multi" or ("modes" in preset and isinstance(preset.get("modes"),dict)):
            modes = preset.get("modes",{}) or {}
            for k in self._curves_store.keys():
                pts = modes.get(k)
                if isinstance(pts,(list,tuple)) and len(pts)>=2:
                    self._curves_store[k] = [(float(x),float(y)) for (x,y) in pts]
                else:
                    self._curves_store[k] = [(0.0,0.0),(1.0,1.0)]
            active = str(preset.get("active") or "K")
            if active not in self._curves_store: active = "K"
            self._current_mode_key = active
            key_to_label = {v:k for k,v in self._mode_key_map.items()}
            want_label = key_to_label.get(active,"K (Brightness)")
            for b in self.mode_group.buttons():
                if b.text()==want_label:
                    b.setChecked(True); break
            self._editor_set_from_norm(self._curves_store[active])
            self._refresh_overlays(); self._quick_preview()
            self._set_status(self.tr("Preset: {0}  [multi]").format(preset.get("name",self.tr("(built-in)"))))
            return

        want = _norm_mode(preset.get("mode"))
        for b in self.mode_group.buttons():
            if b.text().lower()==want.lower():
                b.setChecked(True); break
        ptsN  = preset.get("points_norm")
        shape = preset.get("shape")
        amount = float(preset.get("amount",1.0))
        if not (isinstance(ptsN,(list,tuple)) and len(ptsN)>=2):
            try: ptsN = _shape_points_norm(str(shape or "linear"), amount)
            except Exception: ptsN = [(0.0,0.0),(1.0,1.0)]
        self._editor_set_from_norm(ptsN)
        self._curves_store[self._current_mode_key] = self._editor_points_norm()
        self._refresh_overlays(); self._quick_preview()
        shape_tag = f"[{shape}]" if shape else "[custom]"
        self._set_status(self.tr("Preset: {0}  {1}").format(
            preset.get("name",self.tr("(built-in)")), shape_tag))


# ── Headless apply (unchanged) ────────────────────────────────────────────────

def apply_curves_ops(doc, op: dict):
    try:
        if op.get("tool") != "curves":
            return False
        lut_size = int(op.get("lut_size", 65536))
        modes    = dict(op.get("modes", {}))
        if not modes:
            return True

        def _lut01_from_ptsN(ptsN, size=65536):
            pts_scene = _points_norm_to_scene(ptsN)
            if len(pts_scene) < 2:
                return np.linspace(0.0, 1.0, size, dtype=np.float32)
            xs = np.array([p[0] for p in pts_scene], dtype=np.float64)
            ys = np.array([p[1] for p in pts_scene], dtype=np.float64)
            if np.any(np.diff(xs) <= 0):
                xs += np.linspace(0, 1e-3, len(xs), dtype=np.float64)
            ys = 360.0 - ys
            inp = np.linspace(0.0, 360.0, size, dtype=np.float64)
            try:
                out = PchipInterpolator(xs, ys, extrapolate=True)(inp)
            except Exception:
                out = np.interp(inp, xs, ys)
            return np.clip(out/360.0, 0.0, 1.0).astype(np.float32)

        luts = {k: _lut01_from_ptsN(pts, lut_size) for k,pts in modes.items()}

        img = np.asarray(doc.image)
        if img.dtype.kind in "ui":
            img01 = img.astype(np.float32) / np.iinfo(img.dtype).max
        elif img.dtype.kind == "f":
            mx = float(img.max()) if img.size else 1.0
            img01 = (img / (mx if mx > 1.0 else 1.0)).astype(np.float32)
        else:
            img01 = img.astype(np.float32)

        out01 = CurvesDialogPro._apply_all_curves_once(None, img01, luts)

        dlg_like = CurvesDialogPro.__new__(CurvesDialogPro)
        dlg_like.doc = doc
        dlg_like._full_img = img01
        out01 = CurvesDialogPro._blend_with_mask(dlg_like, out01)

        meta = {"step_name":"Curves (Replay)",
                "curves":{"modes":list(modes.keys()),"lut_size":lut_size},
                "masked":bool(op.get("mask",{}).get("id")),
                "mask_id":op.get("mask",{}).get("id")}
        doc.apply_edit(out01.copy(), metadata=meta, step_name="Curves (Replay)")
        return True
    except Exception as e:
        print("apply_curves_ops failed:", e)
        return False


def _apply_mode_any(img01: np.ndarray, mode: str, lut01: np.ndarray) -> np.ndarray:
    if img01.ndim == 2 or (img01.ndim == 3 and img01.shape[2] == 1):
        ch = img01 if img01.ndim == 2 else img01[...,0]
        if _HAS_NUMBA:
            out = ch.copy(); _nb_apply_lut_mono_inplace(out, lut01)
        else:
            out = _np_apply_lut_channel(ch, lut01)
        return out

    m = mode.lower()
    if m == "k (brightness)":
        if _HAS_NUMBA:
            out = img01.copy(); _nb_apply_lut_color_inplace(out, lut01); return out
        return _np_apply_lut_rgb(img01, lut01)

    if m in ("r","g","b"):
        out = img01.copy(); idx = {"r":0,"g":1,"b":2}[m]
        if _HAS_NUMBA: _nb_apply_lut_mono_inplace(out[...,idx], lut01)
        else: out[...,idx] = _np_apply_lut_channel(out[...,idx], lut01)
        return out

    if m in ("l*","a*","b*","chroma"):
        if _HAS_NUMBA:
            xyz = rgb_to_xyz_numba(img01); lab = xyz_to_lab_numba(xyz)
        else:
            xyz = _np_rgb_to_xyz(img01); lab = _np_xyz_to_lab(xyz)
        if m == "l*":
            L = np.clip(lab[...,0]/100.0, 0.0, 1.0)
            if _HAS_NUMBA: _nb_apply_lut_mono_inplace(L, lut01)
            else: L = _np_apply_lut_channel(L, lut01)
            lab[...,0] = L*100.0
        elif m == "a*":
            a = lab[...,1]; an = np.clip((a+128.0)/255.0, 0.0, 1.0)
            if _HAS_NUMBA: _nb_apply_lut_mono_inplace(an, lut01)
            else: an = _np_apply_lut_channel(an, lut01)
            lab[...,1] = an*255.0-128.0
        elif m == "b*":
            b = lab[...,2]; bn = np.clip((b+128.0)/255.0, 0.0, 1.0)
            if _HAS_NUMBA: _nb_apply_lut_mono_inplace(bn, lut01)
            else: bn = _np_apply_lut_channel(bn, lut01)
            lab[...,2] = bn*255.0-128.0
        else:
            a = lab[...,1]; b = lab[...,2]; C = np.sqrt(a*a+b*b)
            Cn = np.clip(C/200.0, 0.0, 1.0)
            if _HAS_NUMBA: _nb_apply_lut_mono_inplace(Cn, lut01)
            else: Cn = _np_apply_lut_channel(Cn, lut01)
            ratio = np.divide(Cn*200.0, C, out=np.zeros_like(Cn), where=(C>0))
            lab[...,1] = a*ratio; lab[...,2] = b*ratio
        if _HAS_NUMBA:
            return np.clip(xyz_to_rgb_numba(lab_to_xyz_numba(lab)), 0.0, 1.0).astype(np.float32)
        return np.clip(_np_xyz_to_rgb(_np_lab_to_xyz(lab)), 0.0, 1.0).astype(np.float32)

    if m == "saturation":
        if _HAS_NUMBA: hsv = rgb_to_hsv_numba(img01)
        else: hsv = _np_rgb_to_hsv(img01)
        S = np.clip(hsv[...,1], 0.0, 1.0)
        if _HAS_NUMBA: _nb_apply_lut_mono_inplace(S, lut01)
        else: S = _np_apply_lut_channel(S, lut01)
        hsv[...,1] = np.clip(S, 0.0, 1.0)
        if _HAS_NUMBA: return np.clip(hsv_to_rgb_numba(hsv), 0.0, 1.0).astype(np.float32)
        return np.clip(_np_hsv_to_rgb(hsv), 0.0, 1.0).astype(np.float32)

    if _HAS_NUMBA:
        out = img01.copy(); _nb_apply_lut_color_inplace(out, lut01); return out
    return _np_apply_lut_rgb(img01, lut01)