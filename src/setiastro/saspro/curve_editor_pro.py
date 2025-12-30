# pro/curve_editor_pro.py
from __future__ import annotations
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QPointF, QPoint, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QGraphicsView, QLineEdit, QGraphicsScene, 
    QWidget, QMessageBox, QRadioButton, QButtonGroup, QToolButton, QGraphicsEllipseItem, QGraphicsItem, QGraphicsTextItem, QInputDialog, QMenu
)
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QPainterPath, QPen, QColor, QBrush, QIcon, QKeyEvent, QCursor
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# Import shared utilities
from setiastro.saspro.widgets.image_utils import float_to_qimage_rgb8 as _float_to_qimage_rgb8

from setiastro.saspro.curves_preset import (
    list_custom_presets, save_custom_preset, _points_norm_to_scene, _norm_mode,
    _shape_points_norm, open_curves_with_preset, _lut_from_preset
)
from PyQt6.QtWidgets import QFrame, QSizePolicy
from scipy.interpolate import PchipInterpolator
from setiastro.saspro.curves_preset import _sanitize_scene_points, _norm_mode

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

class DraggablePoint(QGraphicsEllipseItem):
    def __init__(self, curve_editor, x, y, color=Qt.GlobalColor.green, lock_axis=None, position_type=None):
        super().__init__(-5, -5, 10, 10)
        self.curve_editor = curve_editor
        self.lock_axis = lock_axis
        self.position_type = position_type
        self.setBrush(QBrush(color))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton | Qt.MouseButton.RightButton)
        self.setPos(x, y)
        outline = QColor(255, 255, 255) if QColor(color).lightnessF() < 0.5 else QColor(0, 0, 0)
        pen = QPen(outline)
        try:
            pen.setWidthF(1.5)  # PyQt6 supports float widths
        except AttributeError:
            pen.setWidth(2)     # fallback for builds missing setWidthF
        self.setPen(pen)        

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            if self in self.curve_editor.control_points:
                self.curve_editor.control_points.remove(self)
                self.curve_editor.scene.removeItem(self)
                self.curve_editor.updateCurve()
            return
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            new_pos = value
            x = new_pos.x()
            y = new_pos.y()

            if self.position_type == 'top_right':
                dist_to_top = abs(y-0)
                dist_to_right = abs(x-360)
                if dist_to_right<dist_to_top:
                    nx=360
                    ny=min(max(y,0),360)
                else:
                    ny=0
                    nx=min(max(x,0),360)
                x,y=nx,ny
            elif self.position_type=='bottom_left':
                dist_to_left=abs(x-0)
                dist_to_bottom=abs(y-360)
                if dist_to_left<dist_to_bottom:
                    nx=0
                    ny=min(max(y,0),360)
                else:
                    ny=360
                    nx=min(max(x,0),360)
                x,y=nx,ny

            all_points=self.curve_editor.end_points+self.curve_editor.control_points
            other_points=[p for p in all_points if p is not self]
            other_points_sorted=sorted(other_points,key=lambda p:p.scenePos().x())

            insert_index=0
            for i,p in enumerate(other_points_sorted):
                if p.scenePos().x()<x:
                    insert_index=i+1
                else:
                    break

            if insert_index>0:
                left_p=other_points_sorted[insert_index-1]
                left_x=left_p.scenePos().x()
                if x<=left_x:
                    x=left_x+0.0001

            if insert_index<len(other_points_sorted):
                right_p=other_points_sorted[insert_index]
                right_x=right_p.scenePos().x()
                if x>=right_x:
                    x=right_x-0.0001

            x=max(0,min(x,360))
            y=max(0,min(y,360))

            super().setPos(x,y)
            self.curve_editor.updateCurve()

        return super().itemChange(change, value)

class ImageLabel(QLabel):
    mouseMoved = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
    def mouseMoveEvent(self, event):
        self.mouseMoved.emit(event.position().x(), event.position().y())
        super().mouseMoveEvent(event)

def _warm_numba_once():
    if not _HAS_NUMBA:
        return
    dummy = np.zeros((2,2), np.float32)
    lut   = np.linspace(0,1,16).astype(np.float32)
    try:
        _nb_apply_lut_mono_inplace(dummy, lut)  # JIT compile path
    except Exception:
        pass

class CurveEditor(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setFixedSize(380, 425)
        self.preview_callback = None  # To trigger real-time updates
        self.symmetry_callback = None
        self._cdf = None
        self._cdf_bins = 1024
        self._cdf_total = 0        

        # Initialize control points and curve path
        self.end_points = []  # Start and end points with axis constraints
        self.control_points = []  # Dynamically added control points
        self.curve_path = QPainterPath()
        self.curve_item = None  # Stores the curve line
        self.sym_line = None

        # Set scene rectangle
        self.scene.setSceneRect(0, 0, 360, 360)
        self.scene.setBackgroundBrush(QColor(32, 32, 36))              # dark background
        self._grid_pen     = QPen(QColor(95, 95, 105), 0, Qt.PenStyle.DashLine)
        self._label_color  = QColor(210, 210, 210)                     # light grid labels
        self._curve_fg     = QColor(255, 255, 255)                     # bright curve
        self._curve_shadow = QColor(0, 0, 0, 190)                      # black halo under curve
        self.initGrid()
        self.initCurve()
        _warm_numba_once()



    def _on_symmetry_pick(self, u: float, _v: float):
        # editor already drew the yellow line; now redistribute handles
        self.redistributeHandlesByPivot(u)
        self._set_status(self.tr("Inflection @ K={0:.3f}").format(u))
        self._quick_preview()

    def initGrid(self):
        pen = self._grid_pen
        for i in range(0, 361, 36):  # grid lines
            self.scene.addLine(i, 0, i, 360, pen)
            self.scene.addLine(0, i, 360, i, pen)

        # X-axis labels (0..1 mapped to 0..360)
        for i in range(0, 361, 36):
            val = i / 360.0
            label = QGraphicsTextItem(f"{val:.3f}")
            label.setDefaultTextColor(self._label_color)
            label.setPos(i - 5, 365)
            self.scene.addItem(label)

    def initCurve(self):
        # Remove existing items from the scene
        # First remove control points
        for p in self.control_points:
            self.scene.removeItem(p)
        # Remove end points
        for p in self.end_points:
            self.scene.removeItem(p)
        # Remove the curve item if any
        if self.curve_item:
            self.scene.removeItem(self.curve_item)
            self.curve_item = None

        # Clear existing point lists
        self.end_points = []
        self.control_points = []

        # Add the default endpoints again
        self.addEndPoint(0, 360, lock_axis=None, position_type='bottom_left', color=Qt.GlobalColor.black)
        self.addEndPoint(360, 0, lock_axis=None, position_type='top_right', color=Qt.GlobalColor.white)

        # Redraw the initial line
        self.updateCurve()

    def getControlHandles(self):
        """Return just the user-added handles (not the endpoints)."""
        # control_points are your green, draggable handles:
        return [(p.scenePos().x(), p.scenePos().y()) for p in self.control_points]

    def setControlHandles(self, handles):
        """Clear existing controls (but keep endpoints), then re-add."""
        # remove any existing controls
        for p in list(self.control_points):
            self.scene.removeItem(p)
        self.control_points.clear()

        # now add back each one
        for x,y in handles:
            self.addControlPoint(x, y)

        # finally redraw spline once
        self.updateCurve()

    def clearSymmetryLine(self):
        """Remove any drawn symmetry line and reset."""
        if self.sym_line:
            self.scene.removeItem(self.sym_line)
            self.sym_line = None
            # redraw without symmetry aid
            self.updateCurve()

    def addEndPoint(self, x, y, lock_axis=None, position_type=None, color=Qt.GlobalColor.red):
        point = DraggablePoint(self, x, y, color=color, lock_axis=lock_axis, position_type=position_type)
        self.scene.addItem(point)
        self.end_points.append(point)

    def addControlPoint(self, x, y, lock_axis=None):

        point = DraggablePoint(self, x, y, color=Qt.GlobalColor.green, lock_axis=lock_axis, position_type=None)
        self.scene.addItem(point)
        self.control_points.append(point)
        self.updateCurve()

    def setSymmetryCallback(self, fn):
        """fn will be called with (u, v) in [0..1] when user ctrl+clicks the grid."""
        self.symmetry_callback = fn

    def setSymmetryPoint(self, x, y):
        pen = QPen(Qt.GlobalColor.yellow)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)
        if self.sym_line is None:
            # draw a vertical symmetry line at scene X==x
            self.sym_line = self.scene.addLine(x, 0, x, 360, pen)
        else:
            self.sym_line.setLine(x, 0, x, 360)
        # if you want to re-draw the curve mirrored around x,
        # you can trigger updateCurve() here or elsewhere
        self.updateCurve()

    def catmull_rom_spline(self, p0, p1, p2, p3, t):
        """
        Compute a point on a Catmull-Rom spline segment at parameter t (0<=t<=1).
        Each p is a QPointF.
        """
        t2 = t * t
        t3 = t2 * t

        x = 0.5 * (2*p1.x() + (-p0.x() + p2.x()) * t +
                    (2*p0.x() - 5*p1.x() + 4*p2.x() - p3.x()) * t2 +
                    (-p0.x() + 3*p1.x() - 3*p2.x() + p3.x()) * t3)
        y = 0.5 * (2*p1.y() + (-p0.y() + p2.y()) * t +
                    (2*p0.y() - 5*p1.y() + 4*p2.y() - p3.y()) * t2 +
                    (-p0.y() + 3*p1.y() - 3*p2.y() + p3.y()) * t3)

        # Clamp to bounding box
        x = max(0, min(360, x))
        y = max(0, min(360, y))

        return QPointF(x, y)

    def generateSmoothCurvePoints(self, points):
        """
        Given a sorted list of QGraphicsItems (endpoints + control points),
        generate a list of smooth points approximating a Catmull-Rom spline
        through these points.
        """
        if len(points) < 2:
            return []
        if len(points) == 2:
            # Just a straight line between two points
            p0 = points[0].scenePos()
            p1 = points[1].scenePos()
            return [p0, p1]

        # Extract scene positions
        pts = [p.scenePos() for p in points]

        # For Catmull-Rom, we need points before the first and after the last
        # We'll duplicate the first and last points.
        extended_pts = [pts[0]] + pts + [pts[-1]]

        smooth_points = []
        steps_per_segment = 20  # increase for smoother curve
        for i in range(len(pts) - 1):
            p0 = extended_pts[i]
            p1 = extended_pts[i+1]
            p2 = extended_pts[i+2]
            p3 = extended_pts[i+3]

            # Sample the spline segment between p1 and p2
            for step in range(steps_per_segment+1):
                t = step / steps_per_segment
                pos = self.catmull_rom_spline(p0, p1, p2, p3, t)
                smooth_points.append(pos)

        return smooth_points

    # Add a callback for the preview
    def setPreviewCallback(self, callback):
        self.preview_callback = callback

    def get8bitLUT(self):
        lut_size = 256

        curve_pts = self.getCurvePoints()
        if len(curve_pts) == 0:
            return np.linspace(0, 255, lut_size, dtype=np.uint8)

        curve_array = np.array(curve_pts, dtype=np.float64)
        xs = curve_array[:, 0]   # 0..360 (scene)
        ys = curve_array[:, 1]   # 0..360 (scene, down)

        ys_for_lut = 360.0 - ys

        input_positions = np.linspace(0, 360, lut_size, dtype=np.float64)
        output_values = np.interp(input_positions, xs, ys_for_lut)

        output_values = (output_values / 360.0) * 255.0
        return np.clip(output_values, 0, 255).astype(np.uint8)

    def updateCurve(self):
        """Update the curve by redrawing based on endpoints and control points."""
        
        all_points = self.end_points + self.control_points
        if not all_points:
            # No points, no curve
            if self.curve_item:
                self.scene.removeItem(self.curve_item)
                self.curve_item = None
            return

        # Sort points by X coordinate
        sorted_points = sorted(all_points, key=lambda p: p.scenePos().x())

        # Extract arrays of X and Y
        xs = [p.scenePos().x() for p in sorted_points]
        ys = [p.scenePos().y() for p in sorted_points]

        # Ensure X values are strictly increasing
        unique_xs, unique_ys = [], []
        for i in range(len(xs)):
            if i == 0 or xs[i] > xs[i - 1]:  # Skip duplicate X values
                unique_xs.append(xs[i])
                unique_ys.append(ys[i])

        # If there's only one point or none, we can't interpolate
        if len(unique_xs) < 2:
            if self.curve_item:
                self.scene.removeItem(self.curve_item)
                self.curve_item = None

            if len(unique_xs) == 1:
                # Optionally draw a single point
                single_path = QPainterPath()
                single_path.addEllipse(unique_xs[0]-2, unique_ys[0]-2, 4, 4)
                pen = QPen(Qt.GlobalColor.white)
                pen.setWidth(3)
                self.curve_item = self.scene.addPath(single_path, pen)
            return

        try:
            # Create a PCHIP interpolator
            interpolator = PchipInterpolator(unique_xs, unique_ys)
            self.curve_function = interpolator

            # Sample the curve
            sample_xs = np.linspace(unique_xs[0], unique_xs[-1], 361)
            sample_ys = interpolator(sample_xs)

        except ValueError as e:
            print(f"Interpolation Error: {e}")  # Log the error instead of crashing
            return  # Exit gracefully

        curve_points = [QPointF(float(x), float(y)) for x, y in zip(sample_xs, sample_ys)]
        self.curve_points = curve_points

        if not curve_points:
            if self.curve_item:
                self.scene.removeItem(self.curve_item)
                self.curve_item = None
            return

        self.curve_path = QPainterPath()
        self.curve_path.moveTo(curve_points[0])
        for pt in curve_points[1:]:
            self.curve_path.lineTo(pt)

        if self.curve_item:
            self.scene.removeItem(self.curve_item)
            self.curve_item = None
        if getattr(self, "curve_shadow_item", None):
            self.scene.removeItem(self.curve_shadow_item)
            self.curve_shadow_item = None

        # shadow (under)
        sh_pen = QPen(self._curve_shadow)
        sh_pen.setWidth(5)
        sh_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        sh_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.curve_shadow_item = self.scene.addPath(self.curve_path, sh_pen)

        # foreground (over)
        pen = QPen(self._curve_fg)
        pen.setWidth(3)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.curve_item = self.scene.addPath(self.curve_path, pen)

        # Trigger the preview callback
        if hasattr(self, 'preview_callback') and self.preview_callback:
            # Generate the 8-bit LUT and pass it to the callback
            lut = self.get8bitLUT()
            self.preview_callback(lut)

    def getCurveFunction(self):
        return self.curve_function

    def getCurvePoints(self):
        if not hasattr(self, 'curve_points') or not self.curve_points:
            return []
        return [(pt.x(), pt.y()) for pt in self.curve_points]

    def getLUT(self):
        lut_size = 65536

        curve_pts = self.getCurvePoints()
        if len(curve_pts) == 0:
            return np.linspace(0, 65535, lut_size, dtype=np.uint16)

        curve_array = np.array(curve_pts, dtype=np.float64)
        xs = curve_array[:, 0]   # 0..360
        ys = curve_array[:, 1]   # 0..360

        ys_for_lut = 360.0 - ys

        input_positions = np.linspace(0, 360, lut_size, dtype=np.float64)
        output_values = np.interp(input_positions, xs, ys_for_lut)

        output_values = (output_values / 360.0) * 65535.0
        return np.clip(output_values, 0, 65535).astype(np.uint16)

    
    def mousePressEvent(self, event):
        # ctrl+left click on the grid → pick inflection point
        if (event.button() == Qt.MouseButton.LeftButton
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            scene_pt = self.mapToScene(event.pos())
            # clamp into scene rect
            x = max(0, min(360, scene_pt.x()))
            y = max(0, min(360, scene_pt.y()))
            # draw the yellow symmetry line
            self.setSymmetryPoint(x, y)
            # compute normalized (u, v)
            u = x / 360.0
            v = 1.0 - (y / 360.0)
            # tell anyone who cares
            if self.symmetry_callback:
                self.symmetry_callback(u, v)
            return  # consume
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """
        Handle double-click events to add a new control point.
        """
        scene_pos = self.mapToScene(event.pos())

        self.addControlPoint(scene_pos.x(), scene_pos.y())
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        """Remove selected points on Delete key press."""
        if event.key() == Qt.Key.Key_Delete:
            for point in self.control_points[:]:
                if point.isSelected():
                    self.scene.removeItem(point)
                    self.control_points.remove(point)
            self.updateCurve()
        super().keyPressEvent(event)

    def clearValueLines(self):
        """Hide any temporary value indicator lines."""
        for attr in ("r_line", "g_line", "b_line", "gray_line"):
            ln = getattr(self, attr, None)
            if ln is not None:
                ln.setVisible(False)

    def _scene_to_norm_points(self, pts_scene: list[tuple[float,float]]) -> list[tuple[float,float]]:
        """(x:[0..360], y:[0..360] down) → (x,y in [0..1] up). Ensures endpoints present & strictly increasing x."""
        out = []
        lastx = -1e9
        for (x, y) in sorted(pts_scene, key=lambda t: t[0]):
            x = float(np.clip(x, 0.0, 360.0))
            y = float(np.clip(y, 0.0, 360.0))
            # strictly increasing X
            if x <= lastx:
                x = lastx + 1e-3
            lastx = x
            out.append((x / 360.0, 1.0 - (y / 360.0)))
        # ensure endpoints
        if not any(abs(px - 0.0)  < 1e-6 for px, _ in out): out.insert(0, (0.0, 0.0))
        if not any(abs(px - 1.0)  < 1e-6 for px, _ in out): out.append((1.0, 1.0))
        # clamp
        return [(float(np.clip(x,0,1)), float(np.clip(y,0,1))) for (x,y) in out]

    def _collect_points_norm_from_editor(self) -> list[tuple[float,float]]:
        """Take endpoints+handles from editor => normalized points."""
        pts_scene = []
        for p in (self.editor.end_points + self.editor.control_points):
            pos = p.scenePos()
            pts_scene.append((float(pos.x()), float(pos.y())))
        return self._scene_to_norm_points(pts_scene)


    def redistributeHandlesByPivot(self, u: float):
        """
        Re-space current control handles around a pivot u∈[0..1].
        Half the handles go in [0, u], the other half in [u, 1].
        Y is sampled from the current curve (fallback: identity).
        """
        u = float(max(0.0, min(1.0, u)))
        N = len(self.control_points)
        if N == 0:
            return

        nL = N // 2
        nR = N - nL
        xL = np.linspace(0.0, u * 360.0, nL + 2, dtype=np.float32)[1:-1]  # exclude endpoints
        xR = np.linspace(u * 360.0, 360.0, nR + 2, dtype=np.float32)[1:-1]
        xs = np.concatenate([xL, xR]) if (nL and nR) else (xR if nL == 0 else xL)

        fn = getattr(self, "curve_function", None)
        if callable(fn):
            try:
                ys = np.clip(fn(xs), 0.0, 360.0)
            except Exception:
                ys = 360.0 - xs  # identity fallback
        else:
            ys = 360.0 - xs     # identity fallback

        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        cps_sorted = sorted(self.control_points, key=lambda p: p.scenePos().x())
        for p, (x, y) in zip(cps_sorted, pairs):
            p.setPos(float(x), float(y))

        self.updateCurve()


    def updateValueLines(self, r, g, b, grayscale=False):
        """
        Update vertical lines on the curve scene.
        For color images (grayscale=False), three lines (red, green, blue) are drawn.
        For grayscale images (grayscale=True), a single gray line is drawn.
        
        Values are assumed to be in the range [0, 1] and mapped to 0–360.
        """
        if grayscale:
            # Map the 0–1 grayscale value to the scene's X coordinate (0–360)
            x = r * 360.0
            if not hasattr(self, "gray_line") or self.gray_line is None:
                self.gray_line = self.scene.addLine(x, 0, x, 360, QPen(Qt.GlobalColor.gray))
            else:
                self.gray_line.setLine(x, 0, x, 360)

            # 🔑 Make sure it’s visible again after Leave/clearValueLines()
            self.gray_line.setVisible(True)

            # Hide any color lines if present
            for attr in ("r_line", "g_line", "b_line"):
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    getattr(self, attr).setVisible(False)
        else:
            # Hide grayscale line if present
            if hasattr(self, "gray_line") and self.gray_line is not None:
                self.gray_line.setVisible(False)
            
            # Map each 0–1 value to X coordinate on scene (0–360)
            r_x = r * 360.0
            g_x = g * 360.0
            b_x = b * 360.0

            # Create or update the red line
            if not hasattr(self, "r_line") or self.r_line is None:
                self.r_line = self.scene.addLine(r_x, 0, r_x, 360, QPen(Qt.GlobalColor.red))
            else:
                self.r_line.setLine(r_x, 0, r_x, 360)
            self.r_line.setVisible(True)

            # Create or update the green line
            if not hasattr(self, "g_line") or self.g_line is None:
                self.g_line = self.scene.addLine(g_x, 0, g_x, 360, QPen(Qt.GlobalColor.green))
            else:
                self.g_line.setLine(g_x, 0, g_x, 360)
            self.g_line.setVisible(True)

            # Create or update the blue line
            if not hasattr(self, "b_line") or self.b_line is None:
                self.b_line = self.scene.addLine(b_x, 0, b_x, 360, QPen(Qt.GlobalColor.blue))
            else:
                self.b_line.setLine(b_x, 0, b_x, 360)
            self.b_line.setVisible(True)

    def current_black_white_thresholds(self) -> tuple[float|None, float|None]:
        """
        Return (black_t, white_t) in [0..1], derived from endpoints:
        - black_t is the X position of the endpoint that sits on the bottom edge (y≈360)
        - white_t is the X position of the endpoint that sits on the top edge (y≈0)
        If an endpoint is on the left/right edges instead, we return None for that side
        (i.e., no clipping for that side).
        """
        bx = None
        wx = None
        eps = 1.0
        for p in self.end_points:
            pos = p.scenePos()
            x, y = float(pos.x()), float(pos.y())
            if abs(y - 360.0) <= eps:
                bx = max(0.0, min(1.0, x / 360.0))
            if abs(y - 0.0) <= eps:
                wx = max(0.0, min(1.0, x / 360.0))
        return bx, wx

    # --- Overlay management (other channels) -----------------
    def setOverlayCurves(self, overlays: dict[str, list[tuple[float,float]]], active_key: str):
        # clear old
        if hasattr(self, "_overlay_items") and self._overlay_items:
            for it in self._overlay_items:
                try: self.scene.removeItem(it)
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        self._overlay_items = []

        colors = {
            "K":"#FFFFFF", "R":"#FF4A4A", "G":"#5CC45C", "B":"#4AA0FF",
            "L*":"#FFFFFF", "a*":"#FF8AB2", "b*":"#A6C8FF", "Chroma":"#FFD866", "Saturation":"#66FFD8"
        }
        faint = 120

        for key, pts in overlays.items():
            if key == active_key or not pts or len(pts) < 2:
                continue

            xs = np.array([p[0] for p in pts], dtype=np.float64)
            ys = np.array([p[1] for p in pts], dtype=np.float64)

            # strict increase on X
            if np.any(np.diff(xs) <= 0):
                xs = xs + np.linspace(0, 1e-3, len(xs), dtype=np.float64)

            # build smooth samples across the whole domain
            sample_x = np.linspace(0.0, 360.0, 361, dtype=np.float64)
            try:
                from scipy.interpolate import PchipInterpolator
                f = PchipInterpolator(xs, ys, extrapolate=True)
                sample_y = f(sample_x)
            except Exception:
                # straight fallback
                sample_y = np.interp(sample_x, xs, ys)

            pen = QPen(QColor(colors.get(key, "#BBBBBB"))); pen.setWidth(2)
            c = pen.color(); c.setAlpha(faint); pen.setColor(c)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

            path = QPainterPath(QPointF(float(sample_x[0]), float(sample_y[0])))
            for x, y in zip(sample_x[1:], sample_y[1:]):
                path.lineTo(QPointF(float(x), float(y)))

            it = self.scene.addPath(path, pen)
            it.setZValue(-5)
            self._overlay_items.append(it)




class CommaToDotLineEdit(QLineEdit):
    def keyPressEvent(self, event: QKeyEvent):
        print("C2D got:", event.key(), repr(event.text()), event.modifiers())
        # if they hit comma (and it's not a Ctrl+Comma shortcut), turn it into a dot
        if event.text() == "," and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            # synthesize a “.” keypress instead
            event = QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Period,
                event.modifiers(),
                "."
            )
        super().keyPressEvent(event)


# ---------- small utilities ----------
# _float_to_qimage_rgb8 imported from setiastro.saspro.widgets.image_utils

def _downsample_for_preview(img01: np.ndarray, max_w: int = 1200) -> np.ndarray:
    h, w = img01.shape[:2]
    if w <= max_w:
        return img01.copy()
    s = max_w / float(w)
    new_w, new_h = max_w, int(round(h * s))
    # resize via nearest/area using uint8 route for speed
    u8 = (np.clip(img01,0,1)*255).astype(np.uint8)
    try:
        import cv2
        out = cv2.resize(u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        # fallback: numpy stride trick (coarse)
        y_idx = (np.linspace(0, h-1, new_h)).astype(np.int32)
        x_idx = (np.linspace(0, w-1, new_w)).astype(np.int32)
        out = u8[y_idx][:, x_idx]
    return out.astype(np.float32)/255.0


# ---------- fallbacks ----------
def build_curve_lut(curve_func, size=65536):
    """Map v∈[0..1] → y∈[0..1] using your curve defined on x∈[0..360]."""
    x = np.linspace(0.0, 360.0, size, dtype=np.float32)
    y = 360.0 - curve_func(x)
    y = (y / 360.0).clip(0.0, 1.0).astype(np.float32)
    return y  # shape (65536,), float32 in [0..1]

def _apply_lut_float01_channel(ch: np.ndarray, lut01: np.ndarray) -> np.ndarray:
    """Apply 16-bit LUT (float [0..1]) to a single channel float image [0..1]."""
    idx = np.clip((ch * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
    return lut01[idx]

def _apply_lut_rgb(img01: np.ndarray, lut01: np.ndarray) -> np.ndarray:
    """Optimized: use Numba LUT for RGB images, falls back to NumPy otherwise."""
    try:
        # Use Numba-accelerated LUT application (parallel, cache-optimized)
        return _nb_apply_lut_color(img01.astype(np.float32, copy=False), lut01.astype(np.float32, copy=False))
    except Exception:
        # Fallback: vectorized NumPy (still faster than loop)
        idx = np.clip((img01 * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
        return lut01[idx]

def _np_apply_lut_channel(ch: np.ndarray, lut01: np.ndarray) -> np.ndarray:
    idx = np.clip((ch * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
    return lut01[idx]

def _np_apply_lut_rgb(img01: np.ndarray, lut01: np.ndarray) -> np.ndarray:
    """Optimized: vectorized LUT on all channels at once instead of per-channel loop."""
    # Vectorized: apply LUT to all channels simultaneously
    idx = np.clip((img01 * (len(lut01)-1)).astype(np.int32), 0, len(lut01)-1)
    return lut01[idx]

# ---- color-space fallbacks (vectorized NumPy) ----
# sRGB <-> XYZ (D65)
_M_rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
_M_xyz2rgb = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660,  1.8760108,  0.0415560],
                       [ 0.0556434, -0.2040259,  1.0572252]], dtype=np.float32)
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883
_delta = 6.0/29.0
_delta3 = _delta**3
_kappa = 24389.0/27.0
_eps   = 216.0/24389.0

def _np_rgb_to_xyz(rgb01: np.ndarray) -> np.ndarray:
    shp = rgb01.shape
    flat = rgb01.reshape(-1, 3)
    xyz = flat @ _M_rgb2xyz.T
    return xyz.reshape(shp)

def _np_xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    shp = xyz.shape
    flat = xyz.reshape(-1, 3)
    rgb = flat @ _M_xyz2rgb.T
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.reshape(shp)

def _f_lab_np(t):
    # f(t) for CIE Lab
    return np.where(t > _delta3, np.cbrt(t), (t / (3*_delta*_delta)) + (4.0/29.0))

def _f_lab_inv_np(ft):
    # inverse of f()
    return np.where(ft > _delta, ft**3, 3*_delta*_delta*(ft - 4.0/29.0))

def _np_xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    X = xyz[...,0] / _Xn
    Y = xyz[...,1] / _Yn
    Z = xyz[...,2] / _Zn
    fx, fy, fz = _f_lab_np(X), _f_lab_np(Y), _f_lab_np(Z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L,a,b], axis=-1).astype(np.float32)

def _np_lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L = lab[...,0]
    a = lab[...,1]
    b = lab[...,2]
    fy = (L + 16)/116.0
    fx = fy + a/500.0
    fz = fy - b/200.0
    X = _Xn * _f_lab_inv_np(fx)
    Y = _Yn * _f_lab_inv_np(fy)
    Z = _Zn * _f_lab_inv_np(fz)
    return np.stack([X,Y,Z], axis=-1).astype(np.float32)

def _np_rgb_to_hsv(rgb01: np.ndarray) -> np.ndarray:
    r,g,b = rgb01[...,0], rgb01[...,1], rgb01[...,2]
    cmax = np.maximum.reduce([r,g,b])
    cmin = np.minimum.reduce([r,g,b])
    delta = cmax - cmin
    H = np.zeros_like(cmax, dtype=np.float32)

    mask = delta != 0
    # where cmax == r
    mr = mask & (cmax == r)
    mg = mask & (cmax == g)
    mb = mask & (cmax == b)
    H[mr] = ( (g[mr]-b[mr]) / delta[mr] ) % 6.0
    H[mg] = ((b[mg]-r[mg]) / delta[mg]) + 2.0
    H[mb] = ((r[mb]-g[mb]) / delta[mb]) + 4.0
    H = (H * 60.0).astype(np.float32)

    S = np.zeros_like(cmax, dtype=np.float32)
    nz = cmax != 0
    S[nz] = (delta[nz] / cmax[nz]).astype(np.float32)
    V = cmax.astype(np.float32)
    return np.stack([H,S,V], axis=-1)

def _np_hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    C = V * S
    hh = (H / 60.0) % 6.0
    X = C * (1 - np.abs(hh % 2 - 1))
    m = V - C
    zeros = np.zeros_like(H, dtype=np.float32)
    r = np.where((0<=hh)&(hh<1), C, np.where((1<=hh)&(hh<2), X, np.where((2<=hh)&(hh<3), zeros, np.where((3<=hh)&(hh<4), zeros, np.where((4<=hh)&(hh<5), X, C)))))
    g = np.where((0<=hh)&(hh<1), X, np.where((1<=hh)&(hh<2), C, np.where((2<=hh)&(hh<3), C, np.where((3<=hh)&(hh<4), X, np.where((4<=hh)&(hh<5), zeros, zeros)))))
    b = np.where((0<=hh)&(hh<1), zeros, np.where((1<=hh)&(hh<2), zeros, np.where((2<=hh)&(hh<3), X, np.where((3<=hh)&(hh<4), C, np.where((4<=hh)&(hh<5), C, X)))))
    rgb = np.stack([r+m, g+m, b+m], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ---------- worker (full-res) ----------

# ---------- worker (full-res) ----------

class _CurvesWorker(QThread):
    done = pyqtSignal(object)

    def __init__(self, image01, luts, invoker=None):
        """
        Backward-compatible worker.

        Accepted call styles:

        1) NEW (multi-curve) style  ← what CurvesDialogPro now uses
           _CurvesWorker(img01, {"K": lutK, "R": lutR, ...}, invoker=self)

        2) OLD (single-curve) style  ← what GHS was doing
           _CurvesWorker(img01, "K (Brightness)", lut01)
           _CurvesWorker(img01, "R", lut01)
           _CurvesWorker(img01, "G", lut01)
           _CurvesWorker(img01, "B", lut01)

        3) Very old / emergency:
           _CurvesWorker(img01, lut01)   → assumes K
        """
        super().__init__()

        # always keep the image contiguous float32
        self.image01 = np.ascontiguousarray(image01.astype(np.float32, copy=False))

        # flags / placeholders
        self._legacy_single = False
        self._invoker = None   # only needed for the new multi-curve path

        # ─────────────────────────────────────────
        # CASE A: GHS / old-style call
        # ─────────────────────────────────────────
        # GHS called: _CurvesWorker(full_img, "K (Brightness)", lut01)
        if isinstance(luts, str):
            mode_str = luts
            lut01 = np.ascontiguousarray(invoker.astype(np.float32, copy=False))
            # map UI text to internal key
            mode_map = {
                "K (Brightness)": "K",
                "K": "K",
                "R": "R",
                "G": "G",
                "B": "B",
            }
            key = mode_map.get(mode_str, "K")
            self.luts = {key: lut01}
            self._legacy_single = True
            return

        # ─────────────────────────────────────────
        # CASE B: weird 2-arg legacy: (img, lut01)
        # ─────────────────────────────────────────
        # someone might have done _CurvesWorker(img, lut01)
        if isinstance(luts, np.ndarray):
            lut01 = np.ascontiguousarray(luts.astype(np.float32, copy=False))
            self.luts = {"K": lut01}
            self._legacy_single = True
            return

        # ─────────────────────────────────────────
        # CASE C: new style (what CurvesDialogPro uses now)
        # ─────────────────────────────────────────
        # here luts should be a dict: {"K": lutK, "R": lutR, ...}
        self.luts = {
            k: np.ascontiguousarray(v.astype(np.float32, copy=False))
            for k, v in luts.items()
        }
        # in the new path we expect an invoker that has _apply_all_curves_once()
        self._invoker = invoker
        self._legacy_single = False

    def run(self):
        # ─────────────────────────────────────────
        # LEGACY path: single channel / single LUT
        # ─────────────────────────────────────────
        if self._legacy_single:
            out = self.image01
            # mono / 2D
            if out.ndim == 2 or (out.ndim == 3 and out.shape[2] == 1):
                lut = (
                    self.luts.get("K")
                    or self.luts.get("R")
                    or self.luts.get("G")
                    or self.luts.get("B")
                )
                if lut is not None:
                    idx = np.clip((out * (len(lut) - 1)).astype(np.int32), 0, len(lut) - 1)
                    out = lut[idx]
                self.done.emit(out.astype(np.float32, copy=False))
                return

            # RGB
            out = out.copy()
            # prefer per-channel, fall back to K
            lutK = self.luts.get("K")
            lutR = self.luts.get("R", lutK)
            lutG = self.luts.get("G", lutK)
            lutB = self.luts.get("B", lutK)

            if lutR is not None:
                idx = np.clip((out[..., 0] * (len(lutR) - 1)).astype(np.int32), 0, len(lutR) - 1)
                out[..., 0] = lutR[idx]
            if lutG is not None:
                idx = np.clip((out[..., 1] * (len(lutG) - 1)).astype(np.int32), 0, len(lutG) - 1)
                out[..., 1] = lutG[idx]
            if lutB is not None:
                idx = np.clip((out[..., 2] * (len(lutB) - 1)).astype(np.int32), 0, len(lutB) - 1)
                out[..., 2] = lutB[idx]

            self.done.emit(out.astype(np.float32, copy=False))
            return

        # ─────────────────────────────────────────
        # NEW path: multi-curve, use dialog’s helper
        # ─────────────────────────────────────────
        if self._invoker is None:
            # extreme safety fallback
            self.done.emit(self.image01)
            return

        out = self._invoker._apply_all_curves_once(self.image01, self.luts)
        self.done.emit(out)


# ---------- dialog ----------

class CurvesDialogPro(QDialog):
    """
    Minimal, shippable Curves Editor for SASpro:
    - Uses your CurveEditor for handles/spline (PCHIP).
    - Live preview on a downsampled copy.
    - Apply writes to the ImageDocument history.
    - Multiple dialogs allowed (no global singletons).
    """
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Curves Editor"))
        self.setWindowFlag(Qt.WindowType.Window, True)
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
                self._follow_conn = False

        self.finished.connect(self._cleanup_connections)
        self._preview_img = None     # downsampled float01
        self._full_img = None        # full-res float01
        self._pix = None
        self._zoom = 0.25
        self._panning = False
        self._pan_start = QPointF()
        self._did_initial_fit = False
        self._apply_when_ready = False
        self._preview_orig = None     # downsampled original
        self._preview_proc = None     # downsampled processed (latest)
        self._show_proc = False       # A/B: False=show original, True=show processed
        self._cdf = None
        self._cdf_bins = 1024
        self._cdf_total = 0

        self._clip_scale = 1.0          # preview→full multiplier
        self._cdf_total_full = 0        # total pixels in full image (H*W)
        self._cdf_total_preview = 0     # total pixels in preview (H*W)

        # --- UI ---
        main = QVBoxLayout(self)          # ⬅️ root is now vertical
        top  = QHBoxLayout()              # ⬅️ holds the two columns

        # Left column: CurveEditor + mode + buttons
        left = QVBoxLayout()
        self.editor = CurveEditor(self)
        left.addWidget(self.editor)

        # mode radio
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        row1 = QHBoxLayout()
        for m in ("K (Brightness)", "R", "G", "B"):
            rb = QRadioButton(m, self)
            if m == "K (Brightness)":
                rb.setChecked(True)          # default selection
            self.mode_group.addButton(rb)
            row1.addWidget(rb)

        row2 = QHBoxLayout()
        for m in ("L*", "a*", "b*", "Chroma", "Saturation"):
            rb = QRadioButton(m, self)
            self.mode_group.addButton(rb)
            row2.addWidget(rb)              

        left.addLayout(row1)
        left.addLayout(row2)

        # Map UI label → internal key
        self._mode_key_map = {
            "K (Brightness)":"K", "R":"R", "G":"G", "B":"B",
            "L*":"L*", "a*":"a*", "b*":"b*", "Chroma":"Chroma", "Saturation":"Saturation"
        }

        # each entry holds points in *normalized* space [(x,y) in 0..1 up, endpoints included]
        self._curves_store = { k: [(0.0,0.0),(1.0,1.0)] for k in self._mode_key_map.values() }

        # remember current mode key
        self._current_mode_key = "K"

        # when user changes the radio, stash current points and load new
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

        # status
        self.lbl_status = QLabel("", self)
        self.lbl_status.setStyleSheet("color: gray;")


        # buttons
        rowb = QHBoxLayout()
        self.btn_preview = QToolButton(self)
        self.btn_preview.setText(self.tr("Toggle Preview"))
        self.btn_preview.setCheckable(True)                  # ⬅️ toggle
        self.btn_apply   = QPushButton(self.tr("Apply to Document"))
        self.btn_reset   = QToolButton(); self.btn_reset.setText(self.tr("Reset"))
        rowb.addWidget(self.btn_preview); rowb.addWidget(self.btn_apply); rowb.addWidget(self.btn_reset)
        left.addLayout(rowb)
        left.addStretch(1)
        top.addLayout(left, 0)

        # Right column: preview w/ zoom/pan
        right = QVBoxLayout()          
        zoombar = QHBoxLayout()
        zoombar.addStretch(1)

        self.btn_zoom_out = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        self.btn_zoom_in  = themed_toolbtn("zoom-in", self.tr("Zoom In"))
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

        # subtle separator line

        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        main.addWidget(sep)

        # bottom status row
        status_row = QHBoxLayout()
        self.lbl_status = QLabel("", self)             # ⬅️ re-create here at bottom
        self.lbl_status.setObjectName("curvesStatus")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_status.setStyleSheet("color: #bbb;")  # or keep your theme color

        # keep it from growing tall: ~2 lines max
        line_h = self.fontMetrics().height()
        self.lbl_status.setMaximumHeight(int(line_h * 2.2))
        self.lbl_status.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        status_row.addWidget(self.lbl_status, 1)

        main.addLayout(status_row, 0)

        # wire
        self.btn_preview.clicked.connect(self._run_preview)
        self.btn_preview.toggled.connect(self._toggle_preview)   # ⬅️ new
        self.btn_apply.clicked.connect(self._apply)
        self.btn_reset.clicked.connect(self._reset_curve)
        self.btn_zoom_out.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.btn_zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.btn_zoom_fit.clicked.connect(self._fit)

        # When curve changes, do a quick preview (non-blocking: downsampled in-UI)
        # You can switch to threaded small preview if images are huge.
        self.editor.setPreviewCallback(self._on_editor_curve_changed)

        # seed images
        self._load_from_doc()
        QTimer.singleShot(0, self._fit_after_load)
        self.editor.setSymmetryCallback(self._on_symmetry_pick)
        self.btn_preview.setChecked(True) 

        self.main_window = self._find_main_window()
        self.source_view = None
        try:
            # Common cases: parent is a subwindow or has a doc_view
            if hasattr(self.parent(), "view"):
                self.source_view = self.parent().view
            elif hasattr(self.parent(), "doc_view"):
                self.source_view = self.parent().doc_view
        except Exception:
            pass

        if self.main_window is not None:
            print(f"[Replay] CurvesDialogPro bound to main_window={id(self.main_window)}, "
                  f"source_view={getattr(self.source_view,'view_id',None)}")

        self._rebuild_presets_menu()

    def _on_editor_curve_changed(self, _lut8=None):
        """
        Called on every editor redraw/drag. Persist the currently edited curve
        into the store, refresh overlays, and do a realtime preview.
        """
        try:
            self._curves_store[self._current_mode_key] = self._editor_points_norm()
        except Exception:
            pass
        # show the true shapes of other channels too
        self._refresh_overlays()
        # now build from *all* current curves (including the just-edited one)
        self._quick_preview()


    def _active_mode_key(self) -> str:
        for b in self.mode_group.buttons():
            if b.isChecked():
                return self._mode_key_map.get(b.text(), "K")
        return "K"

    def _editor_points_norm(self) -> list[tuple[float,float]]:
        # uses your existing _collect_points_norm_from_editor()
        return self._collect_points_norm_from_editor()

    def _editor_set_from_norm(self, ptsN: list[tuple[float,float]]):
        # convert to scene and strip endpoints
        pts_scene = _points_norm_to_scene(ptsN)
        filt = [(x,y) for (x,y) in pts_scene if x > 1e-6 and x < 360-1e-6]
        self.editor.setControlHandles(filt)
        self.editor.updateCurve()

    def _on_mode_toggled(self, checked: bool):
        if not checked:
            return
        # 1) save the curve we were editing
        prev = self._current_mode_key
        try:
            self._curves_store[prev] = self._editor_points_norm()
        except Exception:
            pass

        # 2) load the newly selected curve
        key = self._active_mode_key()
        self._current_mode_key = key
        self._editor_set_from_norm(self._curves_store.get(key, [(0.0,0.0),(1.0,1.0)]))

        # 3) draw overlays for reference
        self._refresh_overlays()
        # 4) refresh preview immediately
        self._quick_preview()

    def _refresh_overlays(self):
        # Build overlay polylines in scene coords for all modes except the active one
        overlays = {}
        for key, ptsN in self._curves_store.items():
            if not ptsN: 
                continue
            pts_scene = _points_norm_to_scene(ptsN)
            # keep full polyline (including endpoints) to show exact shape
            overlays[key] = pts_scene
        self.editor.setOverlayCurves(overlays, self._current_mode_key)

    def _lut01_from_points_norm(self, ptsN: list[tuple[float,float]], size: int = 65536) -> np.ndarray:
        # ptsN are (x,y) in 0..1 (up). Convert to scene space and build a smooth monotone interpolator.
        pts_scene = _points_norm_to_scene(ptsN)  # [(x:[0..360], y:[0..360 down])]
        if len(pts_scene) < 2:
            return np.linspace(0.0, 1.0, size, dtype=np.float32)

        xs = np.array([p[0] for p in pts_scene], dtype=np.float64)
        ys = np.array([p[1] for p in pts_scene], dtype=np.float64)

        # Ensure strictly increasing X (protect against accidental ties)
        m = np.diff(xs) <= 0
        if np.any(m):
            xs = xs + np.linspace(0, 1e-3, len(xs), dtype=np.float64)

        ys = 360.0 - ys  # flip to “up”

        inp = np.linspace(0.0, 360.0, size, dtype=np.float64)
        try:
            from scipy.interpolate import PchipInterpolator
            f = PchipInterpolator(xs, ys, extrapolate=True)
            out = f(inp)
        except Exception:
            # Fallback to linear if SciPy missing or bad control set
            out = np.interp(inp, xs, ys)

        out = np.clip(out / 360.0, 0.0, 1.0).astype(np.float32)
        return out


    def _build_all_active_luts(self) -> dict[str, np.ndarray]:
        """
        Build LUTs for every curve.

        IMPORTANT:
        - The *active* curve (the one shown in the editor right now) must be built
          from the editor's actual spline so we DO NOT re-insert (0,0)/(1,1).
        - All *other* curves (stored in _curves_store) can still be built from
          their normalized points (those are allowed to have endpoints).
        """
        luts: dict[str, np.ndarray] = {}
        active_key = self._current_mode_key

        # 1) ACTIVE curve → from editor spline (no normalization, no auto endpoints)
        fn = getattr(self.editor, "getCurveFunction", None)
        if callable(fn):
            f = fn()
            if f is not None:
                luts[active_key] = build_curve_lut(f, size=65536)

        # 2) OTHER curves → from stored normalized points (old behavior)
        for key, pts in self._curves_store.items():
            if key == active_key:
                continue  # already done above
            # skip exact linear
            if isinstance(pts, (list, tuple)) and len(pts) == 2 and pts[0] == (0.0, 0.0) and pts[1] == (1.0, 1.0):
                continue
            luts[key] = self._lut01_from_points_norm(pts, size=65536)

        return luts

    def _remember_as_last_action(self):
        """
        Capture the current curve as a replayable headless command.

        We store it exactly like other tools:

            command_id = "curves"
            preset     = {mode, shape, amount, points_scene, _ops?}

        where `_ops` (if present) is a full, tool-agnostic op dict
        from export_preview_ops(), used for replay-on-base.
        """
        mw = self._find_main_window()
        if mw is None:
            print("[Replay] Curves: no main_window; not storing last action.")
            return

        # 1) mode label
        btn = self.mode_group.checkedButton() if hasattr(self, "mode_group") else None
        mode_label = btn.text() if btn is not None else "K (Brightness)"
        mode_label = _norm_mode(mode_label)

        # 2) collect control handles → scene points
        if hasattr(self.editor, "getControlHandles"):
            handles = self.editor.getControlHandles()
        elif hasattr(self.editor, "controlHandles"):
            handles = self.editor.controlHandles()
        else:
            handles = []

        pts_scene: list[tuple[float, float]] = []
        for h in handles:
            try:
                x = float(h.x()); y = float(h.y())
            except Exception:
                try:
                    x = float(h[0]); y = float(h[1])
                except Exception:
                    continue
            pts_scene.append((x, y))

        if not pts_scene:
            pts_scene = [(0.0, 360.0), (360.0, 0.0)]

        pts_scene = _sanitize_scene_points(pts_scene)

        core_preset = {
            "mode": mode_label,
            "shape": "custom",
            "amount": 1.0,
            "points_scene": pts_scene,
        }

        # 3) Attach a full op dict for exact replay on base, if possible
        op = None
        try:
            op = self.export_preview_ops()
        except Exception:
            op = None

        if op:
            core_preset["_ops"] = op

        try:
            # This is the same pattern used by Statistical Stretch etc.
            mw._remember_last_headless_command("curves", core_preset, description="Curves")

            # Enable/update the replay button for the originating view
            source_view = getattr(self, "source_view", None)
            if hasattr(mw, "_update_replay_button"):
                mw._update_replay_button(source_view)

            print(
                f"[Replay] Curves: stored last action; "
                f"has_ops={bool(op)} mode={mode_label}"
            )
        except Exception as e:
            print("Curves: failed to remember last action:", e)




    def _apply_all_curves_once(self, img01: np.ndarray, luts: dict[str, np.ndarray]) -> np.ndarray:
        # 1) RGB domain — K then per-channel compose
        out = img01
        if out.ndim == 2:  # mono → treat as K only
            lutK = luts.get("K")
            if lutK is not None:
                out = _np_apply_lut_channel(out, lutK)
            # nothing else applies meaningfully to mono
            return np.clip(out, 0.0, 1.0).astype(np.float32)

        # RGB image
        # compose helper: lut2(lut1(x))
        def _compose_lut(a: np.ndarray | None, b: np.ndarray | None):
            if a is None: return b
            if b is None: return a
            # fast index compose on [0..1] sampled arrays
            N = len(a)
            idx = np.clip((a * (N - 1)).astype(np.int32), 0, N - 1)
            return b[idx]

        lutK = luts.get("K")
        lutR = _compose_lut(lutK, luts.get("R"))
        lutG = _compose_lut(lutK, luts.get("G"))
        lutB = _compose_lut(lutK, luts.get("B"))

        # If no per-channel, still apply K uniformly
        if lutR is None and lutG is None and lutB is None and lutK is not None:
            out = _np_apply_lut_rgb(out, lutK)
        else:
            out = out.copy()
            if lutR is not None: out[...,0] = _np_apply_lut_channel(out[...,0], lutR)
            elif lutK is not None: out[...,0] = _np_apply_lut_channel(out[...,0], lutK)
            if lutG is not None: out[...,1] = _np_apply_lut_channel(out[...,1], lutG)
            elif lutK is not None: out[...,1] = _np_apply_lut_channel(out[...,1], lutK)
            if lutB is not None: out[...,2] = _np_apply_lut_channel(out[...,2], lutB)
            elif lutK is not None: out[...,2] = _np_apply_lut_channel(out[...,2], lutK)

        # 2) Lab family
        need_lab = any(k in luts for k in ("L*","a*","b*","Chroma"))
        if need_lab:
            xyz = _np_rgb_to_xyz(out); lab = _np_xyz_to_lab(xyz)
            if "L*" in luts:
                L = np.clip(lab[...,0]/100.0, 0.0, 1.0)
                L = _np_apply_lut_channel(L, luts["L*"]); lab[...,0] = L*100.0
            if "a*" in luts:
                a = lab[...,1]; an = np.clip((a+128.0)/255.0, 0.0, 1.0)
                an = _np_apply_lut_channel(an, luts["a*"]); lab[...,1] = an*255.0 - 128.0
            if "b*" in luts:
                b = lab[...,2]; bn = np.clip((b+128.0)/255.0, 0.0, 1.0)
                bn = _np_apply_lut_channel(bn, luts["b*"]); lab[...,2] = bn*255.0 - 128.0
            if "Chroma" in luts:
                a = lab[...,1]; b = lab[...,2]
                C = np.sqrt(a*a + b*b); Cn = np.clip(C/200.0, 0.0, 1.0)
                Cn = _np_apply_lut_channel(Cn, luts["Chroma"]); Cnew = Cn*200.0
                ratio = np.divide(Cnew, C, out=np.ones_like(Cnew), where=(C>0))
                lab[...,1] = a*ratio; lab[...,2] = b*ratio
            out = _np_xyz_to_rgb(_np_lab_to_xyz(lab))

        # 3) Saturation (HSV)
        if "Saturation" in luts:
            hsv = _np_rgb_to_hsv(out)
            S = np.clip(hsv[...,1], 0.0, 1.0)
            hsv[...,1] = _np_apply_lut_channel(S, luts["Saturation"])
            out = _np_hsv_to_rgb(hsv)

        return np.clip(out, 0.0, 1.0).astype(np.float32)


    def _fit_after_load(self, tries: int = 0):
        """
        Run Fit-to-Preview once the dialog is visible, the pixmap is ready,
        and the viewport knows its final size. Retries a few ticks if needed.
        """
        if self._did_initial_fit:
            return

        if not self.isVisible():
            QTimer.singleShot(0, lambda: self._fit_after_load(tries))
            return

        # need a pixmap and a live viewport size
        pm = self.label.pixmap()
        vp = self.scroll.viewport() if hasattr(self, "scroll") else None
        have_pm = bool(pm and not pm.isNull())
        have_sizes = bool(vp and vp.width() > 0 and vp.height() > 0)

        if not (self._pix and have_pm and have_sizes):
            if tries < 20:  # ~ a handful of event-loop turns
                QTimer.singleShot(15, lambda: self._fit_after_load(tries + 1))
            return

        # finally do the fit-once
        self._did_initial_fit = True
        self._fit()

    def _capture_view(self):
        """Return (fx, fy, zoom) where f* are fractional center coords in label space."""
        try:
            vp = self.scroll.viewport()
            h  = self.scroll.horizontalScrollBar()
            v  = self.scroll.verticalScrollBar()
            lw = max(1, self.label.width())
            lh = max(1, self.label.height())
            cx = h.value() + vp.width()  / 2.0
            cy = v.value() + vp.height() / 2.0
            fx = float(cx) / float(lw)
            fy = float(cy) / float(lh)
            return (fx, fy, float(self._zoom))
        except Exception:
            return (0.5, 0.5, float(self._zoom))

    def _restore_view(self, fx: float, fy: float, zoom: float):
        """Restore zoom and recenter viewport to previous fractional center."""
        self._set_zoom(zoom)  # calls _apply_zoom() internally
        vp = self.scroll.viewport()
        h  = self.scroll.horizontalScrollBar()
        v  = self.scroll.verticalScrollBar()
        cx = int(round(fx * max(1, self.label.width())))
        cy = int(round(fy * max(1, self.label.height())))
        hx = cx - vp.width()  // 2
        vy = cy - vp.height() // 2
        # clamp
        h.setValue(max(h.minimum(), min(h.maximum(), hx)))
        v.setValue(max(v.minimum(), min(v.maximum(), vy)))


    def _build_preview_luma_cdf(self):
        """Compute a luminance CDF once from the preview image for fast clipping lookups.
        Also derives a preview→full scaling factor so we can report full-image pixel counts.
        """
        img = self._preview_img
        # defaults / safety
        bins = int(getattr(self, "_cdf_bins", 1024))
        self._cdf_bins = bins  # remember for consistency

        # reset outputs
        self._cdf = None
        self._cdf_total = 0
        self._cdf_total_preview = 0
        self._cdf_total_full = 0
        self._clip_scale = 1.0

        if img is None:
            return

        # luminance (float32 [0..1])
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            luma = img if img.ndim == 2 else img[..., 0]
        else:
            luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)
        luma = np.clip(luma, 0.0, 1.0)

        # preview CDF
        hist, _edges = np.histogram(luma, bins=bins, range=(0.0, 1.0))
        self._cdf = np.cumsum(hist).astype(np.int64)
        self._cdf_total_preview = int(luma.size)
        self._cdf_total = self._cdf_total_preview  # backward-compat alias

        # compute full-image pixel count
        full_pixels = 0
        if isinstance(getattr(self, "_full_img", None), np.ndarray) and self._full_img.ndim >= 2:
            Hf, Wf = self._full_img.shape[:2]
            full_pixels = int(Hf * Wf)
        if full_pixels <= 0:
            full_pixels = self._cdf_total_preview  # fall back to preview size

        self._cdf_total_full = full_pixels
        self._clip_scale = (full_pixels / float(self._cdf_total_preview)) if self._cdf_total_preview else 1.0

    def _build_preview_rgb_cdfs(self):
        """Compute per-channel CDFs (R,G,B) from the preview image for clipping stats."""
        self._cdf_rgb = None
        img = self._preview_img
        if img is None or not (img.ndim == 3 and img.shape[2] >= 3):
            return

        bins = int(getattr(self, "_cdf_bins", 1024))
        r = np.clip(img[..., 0].astype(np.float32), 0.0, 1.0)
        g = np.clip(img[..., 1].astype(np.float32), 0.0, 1.0)
        b = np.clip(img[..., 2].astype(np.float32), 0.0, 1.0)

        hr, _ = np.histogram(r, bins=bins, range=(0.0, 1.0))
        hg, _ = np.histogram(g, bins=bins, range=(0.0, 1.0))
        hb, _ = np.histogram(b, bins=bins, range=(0.0, 1.0))

        self._cdf_rgb = {
            "r": np.cumsum(hr).astype(np.int64),
            "g": np.cumsum(hg).astype(np.int64),
            "b": np.cumsum(hb).astype(np.int64),
            "total_preview": int(r.size)  # same for each channel
        }


    def _on_symmetry_pick(self, u: float, _v: float):
        self.editor.redistributeHandlesByPivot(u)
        self._set_status(self.tr("Inflection @ K={0:.3f}").format(u))
        self._quick_preview()

    def _fit_once(self):
        if not self._did_initial_fit:
            self._fit_after_load(0)

    def showEvent(self, ev):
        super().showEvent(ev)
        # kick the fit after this show/layout pass
        QTimer.singleShot(0, self._fit_after_load)

    def _on_preview_mouse_moved(self, x: float, y: float):
        if self._preview_img is None:
            return

        mapped = self._map_label_xy_to_image_ij(x, y)
        if not mapped:
            # cursor is outside the actual pixmap area
            self.editor.clearValueLines()
            self._set_status("")
            return

        # --- clamp to edges so the last pixel is valid ---
        img = self._preview_img
        H, W = img.shape[:2]
        try:
            ix, iy = mapped
            ix = max(0, min(W - 1, int(round(ix))))
            iy = max(0, min(H - 1, int(round(iy))))
        except Exception:
            self.editor.clearValueLines()
            self._set_status("")
            return
        # -------------------------------------------------

        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            v = float(img[iy, ix] if img.ndim == 2 else img[iy, ix, 0])
            v = 0.0 if not np.isfinite(v) else float(np.clip(v, 0.0, 1.0))
            self.editor.updateValueLines(v, 0.0, 0.0, grayscale=True)
            self._set_status(self.tr("Cursor ({0}, {1})  K: {2:.3f}").format(ix, iy, v))
        else:
            C = img.shape[2]
            if C >= 3:
                r, g, b = img[iy, ix, 0], img[iy, ix, 1], img[iy, ix, 2]
            elif C == 2:
                r = g = b = img[iy, ix, 0]
            elif C == 1:
                r = g = b = img[iy, ix, 0]
            else:
                r = g = b = 0.0
            r = 0.0 if not np.isfinite(r) else float(np.clip(r, 0.0, 1.0))
            g = 0.0 if not np.isfinite(g) else float(np.clip(g, 0.0, 1.0))
            b = 0.0 if not np.isfinite(b) else float(np.clip(b, 0.0, 1.0))
            self.editor.updateValueLines(r, g, b, grayscale=False)
            self._set_status(self.tr("Cursor ({0}, {1})  R: {2:.3f}  G: {3:.3f}  B: {4:.3f}").format(ix, iy, r, g, b))


    # 1) Put this helper inside CurvesDialogPro (near other helpers)
    def _map_label_xy_to_image_ij(self, x: float, y: float):
        """Map label-local coords (x,y) to _preview_img pixel (i,j). Returns (ix, iy) or None."""
        if self._pix is None:
            return None
        pm_disp = self.label.pixmap()
        if pm_disp is None or pm_disp.isNull():
            return None

        src_w = self._pix.width()          # size of the *source* pixmap (preview image)
        src_h = self._pix.height()
        disp_w = pm_disp.width()           # size of the *displayed* pixmap on the label
        disp_h = pm_disp.height()
        if src_w <= 0 or src_h <= 0 or disp_w <= 0 or disp_h <= 0:
            return None

        sx = disp_w / float(src_w)
        sy = disp_h / float(src_h)

        ix = int(x / sx)
        iy = int(y / sy)
        if ix < 0 or iy < 0 or ix >= src_w or iy >= src_h:
            return None
        return ix, iy

    def _scene_to_norm_points(self, pts_scene: list[tuple[float,float]]) -> list[tuple[float,float]]:
        """(x:[0..360], y:[0..360] down) → (x,y in [0..1] up). Ensures endpoints present & strictly increasing x."""
        out = []
        lastx = -1e9
        for (x, y) in sorted(pts_scene, key=lambda t: t[0]):
            x = float(np.clip(x, 0.0, 360.0))
            y = float(np.clip(y, 0.0, 360.0))
            # strictly increasing X
            if x <= lastx:
                x = lastx + 1e-3
            lastx = x
            out.append((x / 360.0, 1.0 - (y / 360.0)))
        # ensure endpoints
        if not any(abs(px - 0.0)  < 1e-6 for px, _ in out): out.insert(0, (0.0, 0.0))
        if not any(abs(px - 1.0)  < 1e-6 for px, _ in out): out.append((1.0, 1.0))
        # clamp
        return [(float(np.clip(x,0,1)), float(np.clip(y,0,1))) for (x,y) in out]

    def _collect_points_norm_from_editor(self) -> list[tuple[float,float]]:
        """
        Take endpoints+handles from editor => normalized points.
        NOTE: we do NOT force-add (0,0) and (1,1) here, because that breaks
        manual black/white endpoints. Presets can still add them later.
        """
        pts_scene: list[tuple[float, float]] = []
        for p in (self.editor.end_points + self.editor.control_points):
            pos = p.scenePos()
            pts_scene.append((float(pos.x()), float(pos.y())))

        # convert WITHOUT forced endpoints
        out: list[tuple[float,float]] = []
        lastx = -1e9
        for (x, y) in sorted(pts_scene, key=lambda t: t[0]):
            x = float(np.clip(x, 0.0, 360.0))
            y = float(np.clip(y, 0.0, 360.0))
            if x <= lastx:
                x = lastx + 1e-3
            lastx = x
            out.append((x / 360.0, 1.0 - (y / 360.0)))
        # no auto (0,0)/(1,1) here
        return [(float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))) for (x, y) in out]

    def _save_current_as_preset(self):
        # get name
        name, ok = QInputDialog.getText(self, self.tr("Save Curves Preset"), self.tr("Preset name:"))
        if not ok or not name.strip():
            return
        pts_norm = self._collect_points_norm_from_editor()
        mode = self._current_mode()
        if save_custom_preset(name.strip(), mode, pts_norm):
            self._set_status(self.tr("Saved preset “{0}”.").format(name.strip()))
            self._rebuild_presets_menu()
        else:
            QMessageBox.warning(self, self.tr("Save failed"), self.tr("Could not save preset."))

    def _rebuild_presets_menu(self):
        m = QMenu(self)
        # Built-in shapes under K (Brightness)
        builtins = [
            ("Linear",              {"mode": "K (Brightness)", "shape": "linear"}),
            ("S-Curve (mild)",      {"mode": "K (Brightness)", "shape": "s_mild", "amount": 1.0}),
            ("S-Curve (medium)",    {"mode": "K (Brightness)", "shape": "s_med",  "amount": 1.0}),
            ("S-Curve (strong)",    {"mode": "K (Brightness)", "shape": "s_strong","amount": 1.0}),
            ("Lift Shadows",        {"mode": "K (Brightness)", "shape": "lift_shadows", "amount": 1.0}),
            ("Crush Shadows",       {"mode": "K (Brightness)", "shape": "crush_shadows","amount": 1.0}),
            ("Fade Blacks",         {"mode": "K (Brightness)", "shape": "fade_blacks",  "amount": 1.0}),
            ("Rolloff Highlights",  {"mode": "K (Brightness)", "shape": "rolloff_highlights","amount": 1.0}),
            ("Flatten",             {"mode": "K (Brightness)", "shape": "flatten", "amount": 1.0}),
        ]
        if builtins:
            mb = m.addMenu(self.tr("Built-ins"))
            for label, preset in builtins:
                act = mb.addAction(label)
                act.triggered.connect(lambda _=False, p=preset: self._apply_preset_dict(p))

        # Custom presets (from QSettings)
        customs = list_custom_presets()
        if customs:
            mc = m.addMenu(self.tr("Custom"))
            for p in sorted(customs, key=lambda d: d.get("name","").lower()):
                act = mc.addAction(p.get("name","(unnamed)"))
                act.triggered.connect(lambda _=False, pp=p: self._apply_preset_dict(pp))
            mc.addSeparator()
            act_manage = mc.addAction(self.tr("Manage…"))
            act_manage.triggered.connect(self._open_manage_customs_dialog)  # optional (see below)
        else:
            m.addAction(self.tr("(No custom presets yet)")).setEnabled(False)

        self.btn_presets.setMenu(m)

    def _open_manage_customs_dialog(self):
        # optional: quick-and-dirty remover
        customs = list_custom_presets()
        if not customs:
            QMessageBox.information(self, self.tr("Manage Presets"), self.tr("No custom presets."))
            return
        names = [p.get("name","") for p in customs]
        name, ok = QInputDialog.getItem(self, self.tr("Delete Preset"), self.tr("Choose preset to delete:"), names, 0, False)
        if ok and name:
            from setiastro.saspro.curves_preset import delete_custom_preset
            if delete_custom_preset(name):
                self._rebuild_presets_menu()


    # ----- active document change -----
    def _on_active_doc_changed(self, doc):
        """Called when user clicks a different image window."""
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._load_from_doc()
        QTimer.singleShot(0, self._fit_after_load)

    # ----- data -----
    def _load_from_doc(self):
        img = self.doc.image
        if img is None:
            QMessageBox.information(self, self.tr("No image"), self.tr("Open an image first."))
            return
        arr = np.asarray(img)
        # normalize to float01 gently
        if arr.dtype.kind in "ui":
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        elif arr.dtype.kind == "f":
            mx = float(arr.max()) if arr.size else 1.0
            arr = (arr / (mx if mx > 1.0 else 1.0)).astype(np.float32)
        else:
            arr = arr.astype(np.float32)
        self._full_img = arr
        self._preview_img  = _downsample_for_preview(arr, 1200)
        self._preview_orig = self._preview_img.copy()
        self._preview_proc = None

        self._show_proc = True                      # ⬅️ start with preview ON
        self._quick_preview()                       # ⬅️ build first processed DS frame
        self._update_preview_pix(                   # ⬅️ show processed immediately
            self._preview_proc if self._preview_proc is not None else self._preview_orig,
            preserve_view=False
        )
        self._build_preview_luma_cdf()
        self._build_preview_rgb_cdfs()

    # ----- building LUT from editor -----
    def _build_lut01(self) -> np.ndarray | None:
        get_fn = getattr(self.editor, "getCurveFunction", None)
        if not get_fn:
            return None
        curve_func = get_fn()
        if curve_func is None:
            return None
        # this is your old good helper from the file you pasted
        return build_curve_lut(curve_func, size=65536)


    def _toggle_preview(self, on: bool):
        self._show_proc = bool(on)
        # Ensure we have a processed frame ready
        if self._preview_proc is None:
            self._quick_preview()
        # Pick which buffer to show (both are downsampled)
        img = self._preview_proc if (self._show_proc and self._preview_proc is not None) else self._preview_orig
        self._update_preview_pix(img)
        self._set_status(self.tr("Preview ON") if self._show_proc else self.tr("Preview OFF"))


    # ----- quick (in-UI) preview on downsample -----
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
            bt, wt = self.editor.current_black_white_thresholds()

            if self._preview_img is not None and self._preview_img.ndim == 3 and self._preview_img.shape[2] >= 3:
                # Color image → only per-channel stats
                rgb = self._clip_counts_rgb_from_thresholds(bt, wt)
                def _fmt(pair):
                    cnt_b, cnt_w, fb, fw = pair
                    return self.tr("Bk {0:,} ({1:.2f}%)  Wt {2:,} ({3:.2f}%)").format(cnt_b, fb*100, cnt_w, fw*100)
                self._set_status(
                    self.tr("Clipping —  R: {0}   G: {1}   B: {2}").format(_fmt(rgb['r']), _fmt(rgb['g']), _fmt(rgb['b']))
                )
            else:
                # Grayscale/mono → K summary (unchanged behavior)
                below, above, f_below, f_above = self._clip_counts_from_thresholds(bt, wt)
                self._set_status(
                    self.tr("Clipping —  Bk {0:,} ({1:.2f}%)   Wt {2:,} ({3:.2f}%)").format(below, f_below*100, above, f_above*100)
                )
        except Exception:
            pass


    # ----- threaded full-res preview (also used for Apply path if needed) -----
    def _run_preview(self):
        if self._full_img is None:
            return
        luts = self._build_all_active_luts()
        self.btn_apply.setEnabled(False)
        self._thr = _CurvesWorker(self._full_img, luts, self)
        self._thr.done.connect(self._on_preview_ready)
        self._thr.finished.connect(lambda: self.btn_apply.setEnabled(True))
        self._thr.start()

    def _on_preview_ready(self, out01: np.ndarray):
        # NOTE: do not push full-res into the label
        out_masked = self._blend_with_mask(out01)
        self._last_preview = out_masked         # cache for Apply
        self._set_status(self.tr("Full-res ready (not shown)."))

    def _clip_counts_from_thresholds(self, black_t: float | None, white_t: float | None):
        """
        Return tuple: (below_count_full, above_count_full, below_frac, above_frac)
        using the precomputed *preview* luma CDF, scaled to full-image counts.
        """
        if self._cdf is None or getattr(self, "_cdf_total_preview", 0) <= 0:
            return 0, 0, 0.0, 0.0

        bins = int(getattr(self, "_cdf_bins", 1024))

        # blacks: values strictly < black_t
        if black_t is None:
            below_preview = 0
        else:
            i = int(np.floor(np.clip(float(black_t), 0.0, 1.0) * (bins - 1)))
            i = max(0, min(bins - 1, i))
            below_preview = int(self._cdf[i])

        # whites: values strictly > white_t
        if white_t is None:
            above_preview = 0
        else:
            j = int(np.floor(np.clip(float(white_t), 0.0, 1.0) * (bins - 1)))
            j = max(0, min(bins - 1, j))
            above_preview = int(self._cdf_total_preview - self._cdf[j])

        # scale preview counts to full-image counts
        scale = float(getattr(self, "_clip_scale", 1.0))
        total_full = int(getattr(self, "_cdf_total_full", self._cdf_total_preview)) or 1
        below_full = int(round(below_preview * scale))
        above_full = int(round(above_preview * scale))

        # clamp to valid range
        below_full = max(0, min(below_full, total_full))
        above_full = max(0, min(above_full, total_full))

        # fractions against full-image total
        f_below = below_full / float(total_full)
        f_above = above_full / float(total_full)

        return below_full, above_full, f_below, f_above

    def _clip_counts_rgb_from_thresholds(self, black_t: float | None, white_t: float | None):
        """
        Returns dict:
        {
            'r': (below_full, above_full, frac_below, frac_above),
            'g': (...),
            'b': (...)
        }
        using the precomputed preview RGB CDFs scaled to full-image counts.
        """
        out = {"r": (0,0,0.0,0.0), "g": (0,0,0.0,0.0), "b": (0,0,0.0,0.0)}
        if getattr(self, "_cdf_rgb", None) is None:
            return out

        bins = int(getattr(self, "_cdf_bins", 1024))
        scale = float(getattr(self, "_clip_scale", 1.0))
        total_full = int(getattr(self, "_cdf_total_full", self._cdf_rgb["total_preview"])) or 1
        total_prev = int(self._cdf_rgb["total_preview"]) or 1

        def _bin_idx(t):
            i = int(np.floor(np.clip(float(t), 0.0, 1.0) * (bins - 1)))
            return max(0, min(bins - 1, i))

        for ch in ("r", "g", "b"):
            cdf = self._cdf_rgb[ch]
            # blacks
            if black_t is None:
                below_prev = 0
            else:
                i = _bin_idx(black_t)
                below_prev = int(cdf[i])
            # whites
            if white_t is None:
                above_prev = 0
            else:
                j = _bin_idx(white_t)
                above_prev = int(total_prev - cdf[j])

            below_full = int(round(below_prev * scale))
            above_full = int(round(above_prev * scale))

            below_full = max(0, min(below_full, total_full))
            above_full = max(0, min(above_full, total_full))

            out[ch] = (
                below_full,
                above_full,
                below_full / float(total_full),
                above_full / float(total_full),
            )
        return out

    def export_preview_ops(self) -> dict:
        """
        Produce a deterministic, tool-agnostic op dict for Curves
        that can be replayed on the full image later.
        """
        # Make sure the store has the latest edit from the active editor
        try:
            self._curves_store[self._current_mode_key] = self._editor_points_norm()
        except Exception:
            pass

        # Only include modes that differ from linear
        def _is_linear(pts):
            return isinstance(pts, (list,tuple)) and len(pts)==2 and pts[0]==(0.0,0.0) and pts[1]==(1.0,1.0)

        modes = {}
        for k, pts in self._curves_store.items():
            if not pts or _is_linear(pts):
                continue
            modes[k] = [(float(x), float(y)) for (x,y) in pts]

        op = {
            "version": 1,
            "tool": "curves",
            "modes": modes,
            "active": self._current_mode_key,
            "lut_size": 65536,
            "mask": {
                "id": getattr(self.doc, "active_mask_id", None),
                "blend": "m*out+(1-m)*src",
            },
        }
        return op


    # ----- apply to document -----
    def _apply(self):
        if not hasattr(self, "_last_preview"):
            luts = self._build_all_active_luts()
            out01 = self._apply_all_curves_once(self._full_img, luts)
            out01 = self._blend_with_mask(out01)
            self._last_preview = out01
        self._commit(self._last_preview)

    def _commit(self, out01: np.ndarray):
        try:
            _marr, mid, mname = self._active_mask_layer()
            meta = {
                "step_name": "Curves",
                "curves": {"mode": self._current_mode()},
                "masked": bool(mid),
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            }

            # 1) Apply to the document (updates the active view)
            self.doc.apply_edit(out01.copy(), metadata=meta, step_name="Curves")

            try:
                self._remember_as_last_action()
            except Exception:
                pass

            # 2) Pull the NEW image back into the curves dialog
            #    (clear cached previews so we truly reload from the document)
            self.__dict__.pop("_last_preview", None)
            self._full_img = None
            self._preview_img = None
            self._load_from_doc()          # refresh preview from updated doc

            # 3) Reset the curve drawing so user can keep tweaking from scratch
            # --- after reloading the image from the document ---
            if hasattr(self.editor, "clearSymmetryLine"):
                self.editor.clearSymmetryLine()
            self.editor.initCurve()

            # Clear ALL curves, not just current
            for k in list(self._curves_store.keys()):
                self._curves_store[k] = [(0.0, 0.0), (1.0, 1.0)]

            self._refresh_overlays()
            self._quick_preview()
            self._set_status(self.tr("Applied. Image reloaded. All curves reset — keep tweaking."))



        except Exception as e:
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))


    # ----- helpers -----
    def _current_mode(self) -> str:
        for b in self.mode_group.buttons():
            if b.isChecked():
                return b.text()
        return "K (Brightness)"

    def _set_status(self, s: str):
        self.lbl_status.setText(s)

    # preview label drawing
    def _update_preview_pix(self, img01: np.ndarray | None, preserve_view: bool = True):
        if img01 is None:
            self.label.clear(); self._pix = None; return

        state = self._capture_view() if preserve_view else None

        qimg = _float_to_qimage_rgb8(img01)
        pm = QPixmap.fromImage(qimg)
        self._pix = pm

        if preserve_view and state is not None:
            fx, fy, zoom = state
            # Avoid any auto-fit when we explicitly preserve view
            self._restore_view(fx, fy, zoom)
        else:
            self._apply_zoom()
            if not self._did_initial_fit:
                QTimer.singleShot(0, self._fit_once)


    # --- mask helpers ---------------------------------------------------
    def _active_mask_layer(self):
        """Return (mask_float01, mask_id, mask_name) or (None, None, None)."""
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

    def _resample_mask_if_needed(self, mask: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
        """Nearest-neighbor resize via integer indexing."""
        mh, mw = mask.shape[:2]
        th, tw = out_hw
        if (mh, mw) == (th, tw): return mask
        yi = np.linspace(0, mh - 1, th).astype(np.int32)
        xi = np.linspace(0, mw - 1, tw).astype(np.int32)
        return mask[yi][:, xi]

    def _blend_with_mask(self, processed: np.ndarray) -> np.ndarray:
        """
        Blend processed image with original using active mask (if any).
        Chooses original from preview/full buffers to match shape.
        """
        mask, _mid, _mname = self._active_mask_layer()
        if mask is None:
            return processed

        out = processed.astype(np.float32, copy=False)
        # pick matching original
        if (hasattr(self, "_full_img") and self._full_img is not None
                and out.shape[:2] == self._full_img.shape[:2]):
            src = self._full_img
        else:
            src = self._preview_img

        m = self._resample_mask_if_needed(mask, out.shape[:2])
        if out.ndim == 3 and out.shape[2] == 3:
            m = m[..., None]

        # shape/channel reconcile
        if src.ndim == 2 and out.ndim == 3:
            src = np.stack([src]*3, axis=-1)
        elif src.ndim == 3 and out.ndim == 2:
            src = src[..., 0]

        return (m * out + (1.0 - m) * src).astype(np.float32, copy=False)

    def closeEvent(self, ev):
        self._cleanup_connections()
        super().closeEvent(ev)

    def _cleanup_connections(self):
        # disconnect the "follow active doc" hook
        try:
            if self._follow_conn and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._follow_conn = False

        # stop/kill any running worker thread(s)
        try:
            thr = getattr(self, "_thr", None)
            if thr is not None:
                try:
                    thr.requestInterruption()
                except Exception:
                    pass
                try:
                    thr.quit()
                except Exception:
                    pass
                try:
                    thr.wait(250)
                except Exception:
                    pass
        except Exception:
            pass

        # optional: drop refs that can keep things alive
        try:
            self._thr = None
        except Exception:
            pass


    # zoom/pan
    def _apply_zoom(self):
        if self._pix is None:
            return
        scaled = self._pix.scaled(self._pix.size()*self._zoom,
                                  Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

    def _set_zoom(self, z: float):
        self._zoom = float(max(0.05, min(z, 8.0)))
        self._apply_zoom()

    def _fit(self):
        if self._pix is None: return
        vp = self.scroll.viewport().size()
        if self._pix.width()==0 or self._pix.height()==0: return
        s = min(vp.width()/self._pix.width(), vp.height()/self._pix.height())
        self._set_zoom(max(0.05, s))

    # event filter: ctrl+wheel zoom + panning (like Star Stretch)
    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            # Ctrl+wheel zoom / panning (your existing code) ...
            if ev.type() == QEvent.Type.Wheel and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self._set_zoom(self._zoom * (1.25 if ev.angleDelta().y() > 0 else 0.8))
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True; self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                h = self.scroll.horizontalScrollBar(); v = self.scroll.verticalScrollBar()
                h.setValue(h.value() - int(d.x())); v.setValue(v.value() - int(d.y()))
                self._pan_start = ev.position()
                ev.accept(); return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                ev.accept(); return True

            # NEW: if just moving the mouse (not panning), forward to label coords
            if ev.type() == QEvent.Type.MouseMove and not self._panning:
                # map viewport point → label-local point
                lp = self.label.mapFrom(self.scroll.viewport(), QPoint(int(ev.position().x()), int(ev.position().y())))
                if 0 <= lp.x() < self.label.width() and 0 <= lp.y() < self.label.height():
                    self._on_preview_mouse_moved(lp.x(), lp.y())
                else:
                    self.editor.clearValueLines()
                    self._set_status("")
                return False  # don't consume

            if ev.type() == QEvent.Type.MouseButtonDblClick and ev.button() == Qt.MouseButton.LeftButton:
                if self._preview_img is None or self._pix is None:
                    return False
                pos = self.label.mapFrom(self.scroll.viewport(), ev.pos())
                ix = int(pos.x() / max(self._zoom, 1e-6))
                iy = int(pos.y() / max(self._zoom, 1e-6))
                ix = max(0, min(self._pix.width()  - 1, ix))
                iy = max(0, min(self._pix.height() - 1, iy))

                img = self._preview_img
                if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                    k = float(img[iy, ix] if img.ndim == 2 else img[iy, ix, 0])
                else:
                    k = float(np.mean(img[iy, ix, :3]))
                k = float(np.clip(k, 0.0, 1.0))

                # show the yellow bar + redistribute
                self.editor.setSymmetryPoint(k * 360.0, 0.0)
                self._on_symmetry_pick(k, k)
                ev.accept()
                return True

        # existing label Leave handler
        if obj is self.label and ev.type() == QEvent.Type.Leave:
            self.editor.clearValueLines()
            self._set_status("")
            return False

        # existing double-click handler: just swap in the same mapper
        if obj is self.label and ev.type() == QEvent.Type.MouseButtonDblClick:
            if ev.button() != Qt.MouseButton.LeftButton:
                return False
            pos = ev.position()
            mapped = self._map_label_xy_to_image_ij(pos.x(), pos.y())
            if not mapped or self._preview_img is None:
                return False
            ix, iy = mapped
            img = self._preview_img
            # mono or RGB-average
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                v = float(img[iy, ix] if img.ndim == 2 else img[iy, ix, 0])
            else:
                r, g, b = float(img[iy, ix, 0]), float(img[iy, ix, 1]), float(img[iy, ix, 2])
                v = (r + g + b) / 3.0
            if np.isnan(v):
                return True
            v = float(np.clip(v, 0.0, 1.0))
            x = max(0.001, min(359.999, v * 360.0))

            # place on current curve
            y = None
            try:
                f = self.editor.getCurveFunction()
                if f is not None:
                    y = float(f(x))
            except Exception:
                pass
            if y is None:
                y = 360.0 - x

            # avoid x-collisions
            xs = [p.scenePos().x() for p in (self.editor.end_points + self.editor.control_points)]
            if any(abs(x - ex) < 1e-3 for ex in xs):
                step = 0.002
                for k in range(1, 2000):
                    for cand in (x + k*step, x - k*step):
                        if 0.001 < cand < 359.999 and all(abs(cand - ex) >= 1e-3 for ex in xs):
                            x = cand; break
                    else:
                        continue
                    break

            self.editor.addControlPoint(x, y)
            self._set_status(self.tr("Added point at x={0:.3f}").format(v))
            ev.accept()
            return True

        return super().eventFilter(obj, ev)


    def _reset_curve(self):
        # 1) reset editor drawing to linear
        self.editor.initCurve()
        # 2) mark *every* stored curve linear
        for k in list(self._curves_store.keys()):
            self._curves_store[k] = [(0.0, 0.0), (1.0, 1.0)]
        # 3) refresh overlays & preview
        self._refresh_overlays()
        self._quick_preview()
        self._set_status(self.tr("All curves reset."))

    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    def _apply_preset_dict(self, preset: dict):
        preset = preset or {}

        # 1) set mode radio
        want = _norm_mode(preset.get("mode"))
        for b in self.mode_group.buttons():
            if b.text().lower() == want.lower():
                b.setChecked(True)
                break

        # 2) get points_norm — if absent, build from shape/amount (built-ins)
        ptsN = preset.get("points_norm")
        shape = preset.get("shape")  # may be None for custom presets
        amount = float(preset.get("amount", 1.0))

        if not (isinstance(ptsN, (list, tuple)) and len(ptsN) >= 2):
            try:
                # build from a named shape (built-ins); default to linear
                ptsN = _shape_points_norm(str(shape or "linear"), amount)
            except Exception:
                ptsN = [(0.0, 0.0), (1.0, 1.0)]  # safe fallback

        # 3) apply handles to the editor (strip exact endpoints)
        pts_scene = _points_norm_to_scene(ptsN)
        filt = [(x, y) for (x, y) in pts_scene if 1e-6 < x < 360.0 - 1e-6]

        if hasattr(self.editor, "clearSymmetryLine"):
            self.editor.clearSymmetryLine()

        self.editor.setControlHandles(filt)
        self.editor.updateCurve()   # ensure redraw

        # persist into store & refresh
        self._curves_store[self._current_mode_key] = self._editor_points_norm()
        self._refresh_overlays()
        self._quick_preview()

        # 4) status: don’t assume shape exists
        shape_tag = f"[{shape}]" if shape else "[custom]"
        self._set_status(self.tr("Preset: {0}  {1}").format(preset.get('name', self.tr('(built-in)')), shape_tag))


def apply_curves_ops(doc, op: dict):
    """
    Rebuild LUTs from normalized points and apply to doc.image (full-res).
    Uses the same math as the dialog path, but headless.
    """
    try:
        if op.get("tool") != "curves":
            return False

        # safety defaults
        lut_size = int(op.get("lut_size", 65536))
        modes = dict(op.get("modes", {}))
        if not modes:
            return True  # nothing to do (all linear)

        # Build LUTs exactly like the dialog does (_lut01_from_points_norm)
        def _lut01_from_ptsN(ptsN, size=65536):
            # local import: reuse your existing helper if you prefer
            pts_scene = _points_norm_to_scene(ptsN)
            if len(pts_scene) < 2:
                return np.linspace(0.0, 1.0, size, dtype=np.float32)
            xs = np.array([p[0] for p in pts_scene], dtype=np.float64)
            ys = np.array([p[1] for p in pts_scene], dtype=np.float64)
            if np.any(np.diff(xs) <= 0):
                xs = xs + np.linspace(0, 1e-3, len(xs), dtype=np.float64)
            ys = 360.0 - ys
            inp = np.linspace(0.0, 360.0, size, dtype=np.float64)
            try:
                from scipy.interpolate import PchipInterpolator
                f = PchipInterpolator(xs, ys, extrapolate=True)
                out = f(inp)
            except Exception:
                out = np.interp(inp, xs, ys)
            out = np.clip(out / 360.0, 0.0, 1.0).astype(np.float32)
            return out

        luts = {k: _lut01_from_ptsN(pts, lut_size) for k, pts in modes.items()}

        # Pull full-res, normalize to float01 (same as dialog)
        img = np.asarray(doc.image)
        if img.dtype.kind in "ui":
            img01 = img.astype(np.float32) / np.iinfo(img.dtype).max
        elif img.dtype.kind == "f":
            mx = float(img.max()) if img.size else 1.0
            img01 = (img / (mx if mx > 1.0 else 1.0)).astype(np.float32)
        else:
            img01 = img.astype(np.float32)

        # Apply using the same engine as the dialog
        # (reuse CurvesDialogPro._apply_all_curves_once logic via a tiny local copy)
        out01 = CurvesDialogPro._apply_all_curves_once(None, img01, luts)  # call as unbound

        # Blend with active mask if any
        # Reuse the dialog helper via a tiny shim:
        dlg_like = CurvesDialogPro.__new__(CurvesDialogPro)  # no init
        dlg_like.doc = doc
        dlg_like._full_img = img01
        out01 = CurvesDialogPro._blend_with_mask(dlg_like, out01)

        # Commit to doc history
        meta = {
            "step_name": "Curves (Replay)",
            "curves": {"modes": list(modes.keys()), "lut_size": lut_size},
            "masked": bool(op.get("mask", {}).get("id")),
            "mask_id": op.get("mask", {}).get("id"),
        }
        doc.apply_edit(out01.copy(), metadata=meta, step_name="Curves (Replay)")
        return True
    except Exception as e:
        print("apply_curves_ops failed:", e)
        return False


def _apply_mode_any(img01: np.ndarray, mode: str, lut01: np.ndarray) -> np.ndarray:
    """
    img01: float32 [0..1], mono(H,W) or RGB(H,W,3)
    mode: "K (Brightness)" | "R" | "G" | "B" | "L*" | "a*" | "b*" | "Chroma" | "Saturation"
    lut01: float32 [0..1] LUT
    """
    if img01.ndim == 2 or (img01.ndim == 3 and img01.shape[2] == 1):
        ch = img01 if img01.ndim == 2 else img01[...,0]
        # mono – just apply
        if _HAS_NUMBA:
            out = ch.copy()
            _nb_apply_lut_mono_inplace(out, lut01)
        else:
            out = _np_apply_lut_channel(ch, lut01)
        return out

    # RGB:
    m = mode.lower()
    if m == "k (brightness)":
        if _HAS_NUMBA:
            out = img01.copy()
            _nb_apply_lut_color_inplace(out, lut01)
            return out
        return _np_apply_lut_rgb(img01, lut01)

    if m in ("r","g","b"):
        out = img01.copy()
        idx = {"r":0, "g":1, "b":2}[m]
        if _HAS_NUMBA:
            _nb_apply_lut_mono_inplace(out[..., idx], lut01)
        else:
            out[..., idx] = _np_apply_lut_channel(out[..., idx], lut01)
        return out

    # L*, a*, b*, Chroma => Lab trip
    if m in ("l*", "a*", "b*", "chroma"):
        if _HAS_NUMBA:
            xyz = rgb_to_xyz_numba(img01)
            lab = xyz_to_lab_numba(xyz)
        else:
            xyz = _np_rgb_to_xyz(img01)
            lab = _np_xyz_to_lab(xyz)

        if m == "l*":
            L = lab[...,0] / 100.0
            L = np.clip(L, 0.0, 1.0)
            if _HAS_NUMBA:
                _nb_apply_lut_mono_inplace(L, lut01)
            else:
                L = _np_apply_lut_channel(L, lut01)
            lab[...,0] = L * 100.0

        elif m == "a*":
            a = lab[...,1]
            a_norm = np.clip((a + 128.0)/255.0, 0.0, 1.0)
            if _HAS_NUMBA:
                _nb_apply_lut_mono_inplace(a_norm, lut01)
            else:
                a_norm = _np_apply_lut_channel(a_norm, lut01)
            lab[...,1] = a_norm*255.0 - 128.0

        elif m == "b*":
            b = lab[...,2]
            b_norm = np.clip((b + 128.0)/255.0, 0.0, 1.0)
            if _HAS_NUMBA:
                _nb_apply_lut_mono_inplace(b_norm, lut01)
            else:
                b_norm = _np_apply_lut_channel(b_norm, lut01)
            lab[...,2] = b_norm*255.0 - 128.0

        else:  # chroma
            a = lab[...,1]; b = lab[...,2]
            C = np.sqrt(a*a + b*b)
            C_norm = np.clip(C / 200.0, 0.0, 1.0)
            if _HAS_NUMBA:
                _nb_apply_lut_mono_inplace(C_norm, lut01)
            else:
                C_norm = _np_apply_lut_channel(C_norm, lut01)
            C_new = C_norm * 200.0
            ratio = np.divide(C_new, C, out=np.zeros_like(C_new), where=(C>0))
            lab[...,1] = a * ratio
            lab[...,2] = b * ratio

        if _HAS_NUMBA:
            xyz2 = lab_to_xyz_numba(lab)
            out  = xyz_to_rgb_numba(xyz2)
        else:
            xyz2 = _np_lab_to_xyz(lab)
            out  = _np_xyz_to_rgb(xyz2)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # Saturation => HSV trip
    if m == "saturation":
        if _HAS_NUMBA:
            hsv = rgb_to_hsv_numba(img01)
        else:
            hsv = _np_rgb_to_hsv(img01)
        S = np.clip(hsv[...,1], 0.0, 1.0)
        if _HAS_NUMBA:
            _nb_apply_lut_mono_inplace(S, lut01)
        else:
            S = _np_apply_lut_channel(S, lut01)
        hsv[...,1] = np.clip(S, 0.0, 1.0)
        if _HAS_NUMBA:
            out = hsv_to_rgb_numba(hsv)
        else:
            out = _np_hsv_to_rgb(hsv)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # Unknown ⇒ fallback to brightness
    if _HAS_NUMBA:
        out = img01.copy()
        _nb_apply_lut_color_inplace(out, lut01)
        return out
    return _np_apply_lut_rgb(img01, lut01)