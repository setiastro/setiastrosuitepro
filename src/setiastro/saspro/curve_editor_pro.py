#saspro/curve_editor_pro.py
# ============================================================
#  ____  _         _____           _ _    _ _
# / ___|| |    __ |_   _|__   ___ | | | _(_) |_
# \___ \| |   / _` || |/ _ \ / _ \| | |/ / | __|
#  ___) | |__| (_| || | (_) | (_) | |   <| | |_
# |____/|_____\__,_||_|\___/ \___/|_|_|\_\_|\__|
#
#  Curves Editor  (curve_editor_pro.py)
#  Part of Seti Astro Suite Pro
#  Copyright © 2026 Franklin Marek  |  www.setiastro.com
#  All rights reserved.
# ============================================================
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QPointF, QPoint, QTimer, QSettings, QByteArray, QRectF
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QLineEdit,
    QWidget, QMessageBox, QRadioButton, QButtonGroup, QToolButton, QInputDialog, QMenu, QSizePolicy
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

    def _insert_control(self, sx: float, sy: float):
        # find insertion index (keep X-sorted)
        ins = 1
        for i, (px, _) in enumerate(self._pts[1:-1], start=1):
            if px < sx:
                ins = i + 1
        self._pts.insert(ins, [sx, sy])
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
        self._insert_control(sx, sy)
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
        self.hist = HistogramWidget(self)
        self.hist.setMinimumHeight(120)
        self.hist.setMaximumHeight(140)
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
            self._set_status("")
            return
        img = self._preview_img
        H, W = img.shape[:2]
        try:
            ix, iy = mapped
            ix = max(0, min(W-1, int(round(ix))))
            iy = max(0, min(H-1, int(round(iy))))
        except Exception:
            self.editor.clearValueLines(); self._set_status(""); return

        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            v = float(img[iy,ix] if img.ndim == 2 else img[iy,ix,0])
            v = float(np.clip(0.0 if not np.isfinite(v) else v, 0.0, 1.0))
            self.editor.updateValueLines(v, 0.0, 0.0, grayscale=True)
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
        self._h0_rgb = _compute_hist_rgb(self._hist_base)
        self.hist.set_histograms(self._h0_rgb, None)
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
                self.hist.set_histograms(self._h0_rgb, None)
            else:
                h1_rgb = _compute_hist_rgb(self._preview_proc)
                self.hist.set_histograms(self._h0_rgb, h1_rgb)
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
            self.hist.set_histograms(self._h0_rgb, _compute_hist_rgb(hist_proc))
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
                    self.editor.clearValueLines(); self._set_status("")
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
            self.editor.clearValueLines(); self._set_status(""); return False

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
            self.hist.set_histograms(self._h0_rgb, None)
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