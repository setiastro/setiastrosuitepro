#saspro/gui/mixins/geometry_mixin.py
"""
Geometry operations mixin for AstroSuiteProMainWindow.

This mixin contains all geometry-related functionality: invert, flip,
rotate, rescale, and WCS transformation handling.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap, QPainter, QPen, QColor, QPalette
from PyQt6.QtWidgets import (
    QMessageBox, QInputDialog, QDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QDialogButtonBox,
    QGridLayout, QCheckBox, QStackedLayout, QWidget, QSlider, QSizePolicy
)

# Import numba-accelerated functions

from setiastro.saspro.legacy.numba_utils import (
    invert_image_numba,
    flip_horizontal_numba,
    flip_vertical_numba,
    rotate_90_clockwise_numba,
    rotate_90_counterclockwise_numba,
    rotate_180_numba,
    rescale_image_numba,
    _bin_NxN_numba,
    _upsample_NxN,
    bin2x2_numba,
)


from setiastro.saspro.wcs_update import update_wcs_after_crop

import cv2
import math

if TYPE_CHECKING:
    pass

class RescaleDialog(QDialog):
    def __init__(self, parent=None, *, cur_w=0, cur_h=0, icon=None):
        super().__init__(parent)
        self.setWindowTitle("Rescale Image")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        if icon:
            self.setWindowIcon(icon)

        self._cur_w = max(1, int(cur_w))
        self._cur_h = max(1, int(cur_h))
        self._updating = False

        lay = QVBoxLayout(self)

        # ── Method selector ──────────────────────────────────────────────
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.cmb_method = QComboBox(self)
        self.cmb_method.addItem("Bilinear (smooth, any factor)",  userData="bilinear")
        self.cmb_method.addItem("Integer Bin 2×2 (avg, ÷2)",     userData="bin2x2")
        self.cmb_method.addItem("Integer Bin 3×3 (avg, ÷3)",     userData="bin3x3")
        self.cmb_method.addItem("Integer Bin 4×4 (avg, ÷4)",     userData="bin4x4")
        self.cmb_method.addItem("Integer Upsample 2× (repeat)",  userData="up2x")
        self.cmb_method.addItem("Integer Upsample 3× (repeat)",  userData="up3x")
        self.cmb_method.addItem("Integer Upsample 4× (repeat)",  userData="up4x")
        method_row.addWidget(self.cmb_method, 1)
        lay.addLayout(method_row)

        # ── Factor / description stack ───────────────────────────────────
        self.stk = QStackedLayout()

        # Page 0 — bilinear: show factor spinner
        bilinear_widget = QWidget()
        bilinear_lay = QFormLayout(bilinear_widget)
        bilinear_lay.setContentsMargins(0, 0, 0, 0)
        self.spn_factor = QDoubleSpinBox()
        self.spn_factor.setRange(0.1, 10.0)
        self.spn_factor.setDecimals(3)
        self.spn_factor.setSingleStep(0.05)
        self.spn_factor.setValue(1.0)
        bilinear_lay.addRow("Scale factor:", self.spn_factor)
        self.stk.addWidget(bilinear_widget)

        # Page 1 — integer methods: just show a description label
        self.lbl_int_desc = QLabel()
        self.lbl_int_desc.setWordWrap(True)
        self.stk.addWidget(self.lbl_int_desc)

        stk_host = QWidget()
        stk_host.setLayout(self.stk)
        lay.addWidget(stk_host)

        # ── Result summary ───────────────────────────────────────────────
        self.lbl_summary = QLabel()
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)

        # ── Buttons ──────────────────────────────────────────────────────
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel, self)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.cmb_method.currentIndexChanged.connect(self._on_method_changed)
        self.spn_factor.valueChanged.connect(self._update_summary)
        self._on_method_changed()

    def _on_method_changed(self):
        method = self.cmb_method.currentData()
        if method == "bilinear":
            self.stk.setCurrentIndex(0)
        else:
            self.stk.setCurrentIndex(1)
            descriptions = {
                "bin2x2": "Averages every 2×2 block of pixels into one.\nOutput will be exactly half the width and height.\nIdeal for reducing noise while preserving signal (like CCD 2×2 binning).",
                "bin3x3": "Averages every 3×3 block of pixels into one.\nOutput will be exactly one-third the width and height.\nStrong noise reduction, significant resolution loss.",
                "bin4x4": "Averages every 4×4 block of pixels into one.\nOutput will be exactly one-quarter the width and height.\nMaximum noise reduction, use for very noisy data.",
                "up2x":   "Repeats each pixel into a 2×2 block.\nOutput will be exactly double the width and height.\nPixel-perfect integer upscale — no interpolation artifacts.",
                "up3x":   "Repeats each pixel into a 3×3 block.\nOutput will be exactly triple the width and height.",
                "up4x":   "Repeats each pixel into a 4×4 block.\nOutput will be exactly 4× the width and height.",
            }
            self.lbl_int_desc.setText(descriptions.get(method, ""))
        self._update_summary()

    def _effective_factor(self):
        method = self.cmb_method.currentData()
        factors = {
            "bilinear": self.spn_factor.value(),
            "bin2x2": 0.5, "bin3x3": 1/3, "bin4x4": 0.25,
            "up2x": 2.0,   "up3x": 3.0,   "up4x": 4.0,
        }
        return factors.get(method, 1.0)

    def _update_summary(self):
        f = self._effective_factor()
        new_w = max(1, int(round(self._cur_w * f)))
        new_h = max(1, int(round(self._cur_h * f)))
        method = self.cmb_method.currentData()
        method_label = self.cmb_method.currentText().split("(")[0].strip()
        self.lbl_summary.setText(
            f"Current: <b>{self._cur_w} × {self._cur_h}</b> px<br>"
            f"Result:  <b>{new_w} × {new_h}</b> px  ({f:.4g}×)<br>"
            f"Method:  <b>{method_label}</b>"
        )

    def values(self):
        return {
            "method": self.cmb_method.currentData(),
            "factor": self._effective_factor(),
        }

class ResizeCanvasDialog(QDialog):
    def __init__(self, parent=None, *, cur_w=0, cur_h=0, icon: QIcon | None = None):
        super().__init__(parent)
        self.setWindowTitle("Resize Canvas")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        if icon is not None:
            self.setWindowIcon(icon)

        self._cur_w = max(1, int(cur_w))
        self._cur_h = max(1, int(cur_h))
        self._updating = False
        self._anchor = "center"

        lay = QVBoxLayout(self)

        # -----------------------------
        # Units row
        # -----------------------------
        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Units:"))

        self.cmb_units = QComboBox(self)
        self.cmb_units.addItem("Pixels", userData="px")
        self.cmb_units.addItem("Percent", userData="percent")
        units_row.addWidget(self.cmb_units, 1)

        lay.addLayout(units_row)

        # -----------------------------
        # Size controls
        # -----------------------------
        form = QFormLayout()

        self.spn_w_px = QSpinBox(self)
        self.spn_w_px.setRange(1, 200000)
        self.spn_w_px.setValue(self._cur_w)

        self.spn_h_px = QSpinBox(self)
        self.spn_h_px.setRange(1, 200000)
        self.spn_h_px.setValue(self._cur_h)

        self.spn_w_pct = QDoubleSpinBox(self)
        self.spn_w_pct.setRange(1.0, 10000.0)
        self.spn_w_pct.setDecimals(2)
        self.spn_w_pct.setSingleStep(1.0)
        self.spn_w_pct.setSuffix(" %")
        self.spn_w_pct.setValue(100.0)

        self.spn_h_pct = QDoubleSpinBox(self)
        self.spn_h_pct.setRange(1.0, 10000.0)
        self.spn_h_pct.setDecimals(2)
        self.spn_h_pct.setSingleStep(1.0)
        self.spn_h_pct.setSuffix(" %")
        self.spn_h_pct.setValue(100.0)

        self.spn_fill = QDoubleSpinBox(self)
        self.spn_fill.setRange(0.0, 1.0)
        self.spn_fill.setDecimals(6)
        self.spn_fill.setSingleStep(0.01)
        self.spn_fill.setValue(0.0)

        # Width row container
        self.width_stack = QStackedLayout()
        self.width_stack.setContentsMargins(0, 0, 0, 0)
        self.width_stack.addWidget(self.spn_w_px)   # index 0
        self.width_stack.addWidget(self.spn_w_pct)  # index 1

        width_host = QWidget(self)
        width_host.setLayout(self.width_stack)

        # Height row container
        self.height_stack = QStackedLayout()
        self.height_stack.setContentsMargins(0, 0, 0, 0)
        self.height_stack.addWidget(self.spn_h_px)   # index 0
        self.height_stack.addWidget(self.spn_h_pct)  # index 1

        height_host = QWidget(self)
        height_host.setLayout(self.height_stack)

        form.addRow("New width:", width_host)
        form.addRow("New height:", height_host)
        form.addRow("Fill value:", self.spn_fill)
        lay.addLayout(form)
        self._form = form
        # -----------------------------
        # Current/result summary
        # -----------------------------
        self.lbl_summary = QLabel(self)
        self.lbl_summary.setWordWrap(True)
        lay.addWidget(self.lbl_summary)

        # -----------------------------
        # Anchor / pin controls
        # -----------------------------
        lay.addWidget(QLabel("Keep image pinned to:"))

        grid = QGridLayout()
        self._anchor_buttons = {}

        anchors = [
            ("top-left",     0, 0, "↖"),
            ("top",          0, 1, "↑"),
            ("top-right",    0, 2, "↗"),
            ("left",         1, 0, "←"),
            ("center",       1, 1, "•"),
            ("right",        1, 2, "→"),
            ("bottom-left",  2, 0, "↙"),
            ("bottom",       2, 1, "↓"),
            ("bottom-right", 2, 2, "↘"),
        ]

        for key, r, c, txt in anchors:
            b = QPushButton(txt, self)
            b.setCheckable(True)
            b.setFixedSize(48, 36)
            b.setToolTip(f"Pin to {self._anchor_pretty_name(key)}")
            b.clicked.connect(lambda checked, k=key: self._set_anchor(k))
            self._anchor_buttons[key] = b
            grid.addWidget(b, r, c)

        lay.addLayout(grid)

        self.lbl_anchor_help = QLabel(self)
        self.lbl_anchor_help.setWordWrap(True)
        lay.addWidget(self.lbl_anchor_help)

        self.chk_update_wcs = QCheckBox("Update WCS if present", self)
        self.chk_update_wcs.setChecked(True)
        lay.addWidget(self.chk_update_wcs)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        # Signals
        self.cmb_units.currentIndexChanged.connect(self._sync_units_mode)

        self.spn_w_px.valueChanged.connect(self._sync_from_pixels)
        self.spn_h_px.valueChanged.connect(self._sync_from_pixels)

        self.spn_w_pct.valueChanged.connect(self._sync_from_percent)
        self.spn_h_pct.valueChanged.connect(self._sync_from_percent)

        self._sync_from_pixels()
        self._set_anchor("center")
        self._sync_units_mode()
        self._update_summary()

    def _units(self) -> str:
        return str(self.cmb_units.currentData() or "px")

    def _sync_units_mode(self):
        mode = self._units()

        if mode == "percent":
            self.width_stack.setCurrentIndex(1)   # percent widgets
            self.height_stack.setCurrentIndex(1)
        else:
            self.width_stack.setCurrentIndex(0)   # pixel widgets
            self.height_stack.setCurrentIndex(0)

        self._update_summary()

    def _sync_from_pixels(self):
        if self._updating:
            return
        self._updating = True
        try:
            w = max(1, int(self.spn_w_px.value()))
            h = max(1, int(self.spn_h_px.value()))
            self.spn_w_pct.setValue((w / self._cur_w) * 100.0)
            self.spn_h_pct.setValue((h / self._cur_h) * 100.0)
        finally:
            self._updating = False
        self._update_summary()

    def _sync_from_percent(self):
        if self._updating:
            return
        self._updating = True
        try:
            pw = max(0.01, float(self.spn_w_pct.value()))
            ph = max(0.01, float(self.spn_h_pct.value()))
            w = max(1, int(round(self._cur_w * pw / 100.0)))
            h = max(1, int(round(self._cur_h * ph / 100.0)))
            self.spn_w_px.setValue(w)
            self.spn_h_px.setValue(h)
        finally:
            self._updating = False
        self._update_summary()

    def _anchor_pretty_name(self, key: str) -> str:
        names = {
            "top-left": "Top Left",
            "top": "Top",
            "top-right": "Top Right",
            "left": "Left",
            "center": "Center",
            "right": "Right",
            "bottom-left": "Bottom Left",
            "bottom": "Bottom",
            "bottom-right": "Bottom Right",
        }
        return names.get(key, key)

    def _anchor_rc(self, key: str) -> tuple[int, int]:
        mapping = {
            "top-left": (0, 0),
            "top": (0, 1),
            "top-right": (0, 2),
            "left": (1, 0),
            "center": (1, 1),
            "right": (1, 2),
            "bottom-left": (2, 0),
            "bottom": (2, 1),
            "bottom-right": (2, 2),
        }
        return mapping.get(key, (1, 1))

    def _growth_glyph_for_cell(self, pinned_key: str, cell_key: str) -> str:
        """
        Returns the glyph to show in a given 3x3 cell based on which anchor is pinned.
        The pinned cell is shown as a dot. All other cells show the direction that
        extra canvas expands relative to the pinned location.
        """
        if pinned_key == cell_key:
            return "•"

        pr, pc = self._anchor_rc(pinned_key)
        cr, cc = self._anchor_rc(cell_key)

        dr = cr - pr
        dc = cc - pc

        vr = 0 if dr == 0 else (1 if dr > 0 else -1)
        vc = 0 if dc == 0 else (1 if dc > 0 else -1)

        glyphs = {
            (-1, -1): "↖",
            (-1,  0): "↑",
            (-1,  1): "↗",
            ( 0, -1): "←",
            ( 0,  0): "•",
            ( 0,  1): "→",
            ( 1, -1): "↙",
            ( 1,  0): "↓",
            ( 1,  1): "↘",
        }
        return glyphs.get((vr, vc), "•")

    def _refresh_anchor_buttons(self):
        for cell_key, btn in self._anchor_buttons.items():
            # Dynamic directional glyph
            btn.setText(self._growth_glyph_for_cell(self._anchor, cell_key))

            # Tooltip
            btn.setToolTip(f"Pin to {self._anchor_pretty_name(cell_key)}")

            checked = (cell_key == self._anchor)
            btn.setChecked(checked)

            if checked:
                # Slightly larger selected button with filled background
                btn.setFixedSize(54, 42)
                btn.setStyleSheet("""
                    QPushButton {
                        font-weight: bold;
                        font-size: 20px;
                        color: palette(highlighted-text);
                        background-color: palette(highlight);
                        border: 2px solid palette(highlight);
                        border-radius: 7px;
                        padding: 2px;
                    }
                    QPushButton:hover {
                        border: 2px solid palette(highlight);
                    }
                    QPushButton:pressed {
                        background-color: palette(highlight);
                    }
                """)
            else:
                btn.setFixedSize(48, 36)
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 18px;
                        border: 1px solid palette(mid);
                        border-radius: 6px;
                        padding: 2px;
                    }
                    QPushButton:hover {
                        border: 1px solid palette(highlight);
                    }
                """)

    def _set_anchor(self, key: str):
        self._anchor = key
        self._refresh_anchor_buttons()

        pretty = self._anchor_pretty_name(key)
        self.lbl_anchor_help.setText(
            f"Pinned to: <b>{pretty}</b><br>"
            f"Extra canvas is added away from the pinned side. "
            f"If the canvas is made smaller, cropping also happens away from the pinned side."
        )
        self._update_summary()

    def _update_summary(self):
        new_w = int(self.spn_w_px.value())
        new_h = int(self.spn_h_px.value())
        pw = (new_w / self._cur_w) * 100.0
        ph = (new_h / self._cur_h) * 100.0

        mode = self._units()
        if mode == "percent":
            size_text = f"{pw:.2f}% × {ph:.2f}%"
        else:
            size_text = f"{new_w} × {new_h} px"

        self.lbl_summary.setText(
            f"Current: <b>{self._cur_w} × {self._cur_h}</b> px<br>"
            f"Result: <b>{new_w} × {new_h}</b> px ({pw:.2f}% × {ph:.2f}%)<br>"
            f"Pinned to: <b>{self._anchor_pretty_name(self._anchor)}</b><br>"
            f"Editing in: <b>{'Percent' if mode == 'percent' else 'Pixels'}</b> ({size_text})"
        )

    def values(self):
        return {
            "new_w": int(self.spn_w_px.value()),
            "new_h": int(self.spn_h_px.value()),
            "fill_value": float(self.spn_fill.value()),
            "anchor": self._anchor,
            "update_wcs": bool(self.chk_update_wcs.isChecked()),
            "units": self._units(),
            "width_percent": float(self.spn_w_pct.value()),
            "height_percent": float(self.spn_h_pct.value()),
        }

class _RotateCanvas(QWidget):
    """Live rotation preview with a grab handle.

    Drag anywhere in the view to rotate; the ring handle is there to show
    where the grip is and which way is zero. Positive is CCW, matching the
    rotation the apply step performs.
    """

    angleChanged = pyqtSignal(float)

    HANDLE_R = 9          # knob radius, px
    GRAB_SLOP = 16        # how close to the knob still counts as grabbing it

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base: QPixmap | None = None
        self._angle = 0.0
        self._drag_from = None          # (cursor_angle_ccw, angle_at_press)
        self.setMinimumSize(440, 340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── state ────────────────────────────────────────────────────────────
    def set_base(self, pm: QPixmap | None):
        self._base = pm
        self.update()

    def angle(self) -> float:
        return self._angle

    def set_angle(self, deg: float, *, emit: bool = True):
        deg = float(deg)
        # keep it in (-180, 180]: the same rotation, said the short way
        deg = ((deg + 180.0) % 360.0) - 180.0
        if abs(deg - self._angle) < 1e-9:
            return
        self._angle = deg
        self.update()
        if emit:
            self.angleChanged.emit(deg)

    # ── geometry helpers ─────────────────────────────────────────────────
    def _center(self) -> QPointF:
        return QPointF(self.width() * 0.5, self.height() * 0.5)

    def _ring_radius(self) -> float:
        return max(40.0, min(self.width(), self.height()) * 0.5 - 18.0)

    def _cursor_angle(self, pos) -> float:
        """Screen position -> CCW-positive degrees. Screen y runs down, so
        the visual CCW direction is the negative of the raw atan2."""
        c = self._center()
        dx = float(pos.x()) - c.x()
        dy = float(pos.y()) - c.y()
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return self._angle
        return -math.degrees(math.atan2(dy, dx))

    def _knob_pos(self) -> QPointF:
        c = self._center()
        r = self._ring_radius()
        a = math.radians(-self._angle)
        return QPointF(c.x() + r * math.cos(a), c.y() + r * math.sin(a))

    @staticmethod
    def _snap(deg: float, mods) -> float:
        if mods & Qt.KeyboardModifier.ControlModifier:
            return round(deg / 15.0) * 15.0          # common angles
        if mods & Qt.KeyboardModifier.ShiftModifier:
            return round(deg * 10.0) / 10.0          # fine
        return round(deg)                            # whole degrees by default

    # ── interaction ──────────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(ev)
            return
        self._drag_from = (self._cursor_angle(ev.position()), self._angle)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_from is None:
            return
        start_cursor, start_angle = self._drag_from
        delta = self._cursor_angle(ev.position()) - start_cursor
        # unwrap: a drag across the +/-180 seam must not jump a full turn
        delta = ((delta + 180.0) % 360.0) - 180.0
        self.set_angle(self._snap(start_angle + delta, ev.modifiers()))
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._drag_from = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        self.set_angle(0.0)
        ev.accept()

    def wheelEvent(self, ev):
        step = 0.1 if (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1.0
        notches = ev.angleDelta().y() / 120.0
        if notches:
            self.set_angle(self._angle + notches * step)
            ev.accept()
        else:
            super().wheelEvent(ev)

    def keyPressEvent(self, ev):
        step = 0.1 if (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1.0
        k = ev.key()
        if k in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.set_angle(self._angle + step)
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.set_angle(self._angle - step)
        elif k in (Qt.Key.Key_0, Qt.Key.Key_Home):
            self.set_angle(0.0)
        else:
            super().keyPressEvent(ev)
            return
        ev.accept()

    # ── paint ────────────────────────────────────────────────────────────
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        p.fillRect(self.rect(), QColor(24, 24, 28))

        if self._base is None or self._base.isNull():
            p.setPen(QColor(160, 160, 170))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No preview")
            return

        bw, bh = self._base.width(), self._base.height()
        a = math.radians(self._angle)
        ca, sa = abs(math.cos(a)), abs(math.sin(a))
        rot_w = bw * ca + bh * sa              # bounding box of the result
        rot_h = bw * sa + bh * ca

        margin = self.HANDLE_R + 10
        avail_w = max(1.0, self.width() - 2 * margin)
        avail_h = max(1.0, self.height() - 2 * margin)
        scale = min(avail_w / rot_w, avail_h / rot_h, 1.0e6)

        c = self._center()

        # the grown canvas the rotation will produce
        p.save()
        p.translate(c)
        box = QRectF(-rot_w * scale * 0.5, -rot_h * scale * 0.5,
                     rot_w * scale, rot_h * scale)
        p.setPen(QPen(QColor(90, 90, 100), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(box)
        p.restore()

        # the image itself
        p.save()
        p.translate(c)
        p.rotate(-self._angle)                 # screen y is down: negate for CCW
        p.scale(scale, scale)
        p.drawPixmap(QPointF(-bw * 0.5, -bh * 0.5), self._base)
        p.restore()

        # ── the grip ─────────────────────────────────────────────────────
        r = self._ring_radius()
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 36), 1, Qt.PenStyle.DotLine))
        p.drawEllipse(c, r, r)

        # every 15 deg, longer at the cardinals
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        for t in range(0, 360, 15):
            th = math.radians(t)
            inner = r - (7 if t % 45 == 0 else 4)
            p.drawLine(QPointF(c.x() + inner * math.cos(th), c.y() + inner * math.sin(th)),
                       QPointF(c.x() + r * math.cos(th), c.y() + r * math.sin(th)))

        # zero reference, pointing right
        p.setPen(QPen(QColor(150, 150, 160), 1, Qt.PenStyle.DashLine))
        p.drawLine(c, QPointF(c.x() + r, c.y()))

        accent = QColor(94, 183, 255)
        if abs(self._angle) > 1e-6:            # arc swept so far
            p.setPen(QPen(QColor(94, 183, 255, 120), 3))
            rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
            p.drawArc(rect, 0, int(round(self._angle * 16)))

        knob = self._knob_pos()
        p.setPen(QPen(accent, 2))
        p.drawLine(c, knob)
        p.setBrush(accent if self._drag_from is None else QColor(255, 255, 255))
        p.setPen(QPen(QColor(16, 16, 20), 2))
        p.drawEllipse(knob, self.HANDLE_R, self.HANDLE_R)

        # readout
        p.setPen(QColor(235, 235, 240))
        f = p.font()
        f.setPointSizeF(max(9.0, f.pointSizeF() + 1.0))
        f.setBold(True)
        p.setFont(f)
        txt = f"{self._angle:+.1f}°".replace("+0.0", "0.0")
        p.drawText(QRectF(8, 6, self.width() - 16, 22),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)


class RotateDialog(QDialog):
    """Rotate by eye: drag the image around, read the angle off the dial."""

    PREVIEW_MAX = 720     # longest preview edge, px

    def __init__(self, parent=None, *, image=None, icon: QIcon | None = None,
                 initial: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("Rotate")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.Window, True)
        if icon:
            self.setWindowIcon(icon)

        self._src = np.asarray(image) if image is not None else None
        self._src_h, self._src_w = (self._src.shape[:2] if self._src is not None
                                    else (0, 0))
        self._pm_raw: QPixmap | None = None
        self._pm_stretched: QPixmap | None = None

        lay = QVBoxLayout(self)

        self.canvas = _RotateCanvas(self)
        lay.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Angle"))
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(-1800, 1800)          # tenths of a degree
        self.slider.setSingleStep(10)
        self.slider.setPageStep(150)
        self.slider.setTickInterval(450)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        row.addWidget(self.slider, 1)
        self.spin = QDoubleSpinBox(self)
        self.spin.setRange(-180.0, 180.0)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(1.0)
        self.spin.setSuffix(" °")
        self.spin.setKeyboardTracking(False)
        row.addWidget(self.spin)
        self.btn_reset = QPushButton("Reset", self)
        row.addWidget(self.btn_reset)
        lay.addLayout(row)

        hint = QLabel(
            "Drag the image to rotate · whole degrees by default · "
            "Shift = 0.1° · Ctrl = snap to 15° · "
            "wheel or arrows nudge · double-click resets", self)
        hint.setWordWrap(True)
        try:
            pal = hint.palette()
            col = pal.color(QPalette.ColorRole.WindowText)
            col.setAlpha(170)
            pal.setColor(QPalette.ColorRole.WindowText, col)
            hint.setPalette(pal)
        except Exception:
            pass
        lay.addWidget(hint)

        bottom = QHBoxLayout()
        self.chk_stretch = QCheckBox("Autostretch preview", self)
        self.chk_stretch.setChecked(True)
        self.chk_stretch.setToolTip(
            "Display only — the rotation is applied to the linear data.")
        bottom.addWidget(self.chk_stretch)
        bottom.addStretch(1)
        self.lbl_size = QLabel("", self)
        bottom.addWidget(self.lbl_size)
        lay.addLayout(bottom)

        self._accepted_angle: float | None = None
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel, self)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.canvas.angleChanged.connect(self._on_canvas_angle)
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.btn_reset.clicked.connect(lambda: self._set_angle(0.0))
        self.chk_stretch.toggled.connect(self._refresh_preview)

        self._syncing = False
        self._build_previews()
        self._refresh_preview()
        self._set_angle(float(initial))
        self.resize(760, 680)

    # ── preview ──────────────────────────────────────────────────────────
    def _build_previews(self):
        src = self._src
        if src is None:
            return
        try:
            a = np.asarray(src, dtype=np.float32)
            if a.ndim == 3 and a.shape[2] == 1:
                a = a[..., 0]
            if a.ndim == 3 and a.shape[2] > 3:
                a = a[..., :3]

            h, w = a.shape[:2]
            longest = max(h, w)
            if longest > self.PREVIEW_MAX and cv2 is not None:
                f = self.PREVIEW_MAX / float(longest)
                a = cv2.resize(a, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                               interpolation=cv2.INTER_AREA)

            self._pm_raw = self._to_pixmap(np.clip(a, 0.0, 1.0))
            self._pm_stretched = self._to_pixmap(self._stretch(a))
        except Exception:
            self._pm_raw = self._pm_raw or None
            self._pm_stretched = self._pm_stretched or self._pm_raw

    @staticmethod
    def _stretch(a: np.ndarray) -> np.ndarray:
        """Display stretch only. Needs [0,1] in, so scale first if the data
        runs hot rather than clipping the highlights away."""
        x = np.asarray(a, dtype=np.float32)
        peak = float(np.nanmax(x)) if x.size else 1.0
        if np.isfinite(peak) and peak > 1.0:
            x = x / peak
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        try:
            from setiastro.saspro.imageops.stretch import histogram_style_autostretch
            if x.ndim == 3 and x.shape[2] == 3:
                return np.asarray(histogram_style_autostretch(x, 3.0), dtype=np.float32)
            if x.ndim == 2:
                return np.asarray(histogram_style_autostretch(x, 3.0), dtype=np.float32)
        except Exception:
            pass
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if hi - lo <= 1e-6:
            return np.zeros_like(x)
        return ((x - lo) / (hi - lo)).astype(np.float32)

    @staticmethod
    def _to_pixmap(a: np.ndarray) -> QPixmap | None:
        x = np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0)
        if x.ndim == 2:
            x = np.repeat(x[..., None], 3, axis=2)
        u8 = np.ascontiguousarray((x * 255.0 + 0.5).astype(np.uint8))
        h, w = u8.shape[:2]
        qi = QImage(u8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qi)

    def _refresh_preview(self):
        pm = self._pm_stretched if self.chk_stretch.isChecked() else self._pm_raw
        self.canvas.set_base(pm)

    # ── angle plumbing ───────────────────────────────────────────────────
    def _set_angle(self, deg: float):
        self.canvas.set_angle(deg)
        self._on_canvas_angle(self.canvas.angle())

    def _on_canvas_angle(self, deg: float):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.slider.setValue(int(round(deg * 10.0)))
            self.spin.setValue(float(deg))
        finally:
            self._syncing = False
        self._update_size_label(deg)

    def _on_slider(self, v: int):
        if self._syncing:
            return
        self._syncing = True
        try:
            deg = v / 10.0
            self.canvas.set_angle(deg, emit=False)
            self.spin.setValue(deg)
        finally:
            self._syncing = False
        self._update_size_label(self.canvas.angle())

    def _on_spin(self, v: float):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.canvas.set_angle(v, emit=False)
            self.slider.setValue(int(round(v * 10.0)))
        finally:
            self._syncing = False
        self._update_size_label(self.canvas.angle())

    def _update_size_label(self, deg: float):
        if not self._src_w or not self._src_h:
            self.lbl_size.setText("")
            return
        ow, oh = self.output_size(deg)
        self.lbl_size.setText(
            f"{self._src_w} × {self._src_h}  →  {ow} × {oh}")

    def output_size(self, deg: float | None = None) -> tuple[int, int]:
        """Same bounding-box math the apply step uses, so the label matches."""
        if deg is None:
            deg = self.angle()
        w, h = float(self._src_w), float(self._src_h)
        a = math.radians(float(deg))
        ca, sa = abs(math.cos(a)), abs(math.sin(a))
        return (int(math.ceil((w - 1) * ca + (h - 1) * sa + 1.0)),
                int(math.ceil((w - 1) * sa + (h - 1) * ca + 1.0)))

    def accept(self):
        # Latch the value before the widgets can go away, so the caller can
        # still read it from a dialog that has already been torn down.
        try:
            self._accepted_angle = float(self.spin.value())
        except RuntimeError:
            pass
        super().accept()

    def angle(self) -> float:
        if self._accepted_angle is not None:
            return float(self._accepted_angle)
        return float(self.spin.value())


class GeometryMixin:
    """
    Mixin for geometry operations.
    
    Provides methods for inverting, flipping, rotating, and rescaling images,
    with automatic WCS (World Coordinate System) updates when applicable.
    """

    def _apply_geom_with_wcs(self, doc, out_image: np.ndarray,
                            M_src_to_dst: np.ndarray | None,
                            step_name: str,
                            command_id: str | None = None,
                            preset: dict | None = None):
        out_h, out_w = out_image.shape[:2]
        meta = dict(getattr(doc, "metadata", {}) or {})

        # Only attempt WCS update if there is a real celestial WCS present
        should_update_wcs = False
        if update_wcs_after_crop is not None and M_src_to_dst is not None:
            wcs_obj = meta.get("wcs")
            if wcs_obj is not None:
                try:
                    from astropy.wcs import WCS as _WCS
                    if isinstance(wcs_obj, _WCS) and wcs_obj.has_celestial:
                        # Sanity check: CRVAL must be valid sky coords
                        crval = getattr(wcs_obj.wcs, "crval", None)
                        if (crval is not None and len(crval) >= 2
                                and -360.0 <= float(crval[0]) <= 360.0
                                and -90.0  <= float(crval[1]) <= 90.0):
                            should_update_wcs = True
                        else:
                            print(f"[WCS-GEOM] Skipping WCS update for {step_name}: "
                                f"CRVAL {crval} is not valid sky coordinates.")
                except Exception as e:
                    print(f"[WCS-GEOM] WCS validation failed for {step_name}: {e}")

        if should_update_wcs:
            try:
                meta = update_wcs_after_crop(
                    meta,
                    M_src_to_dst=M_src_to_dst,
                    out_w=out_w,
                    out_h=out_h,
                )
            except Exception as e:
                print(f"[WCS-GEOM] WCS update failed for {step_name}: {e}")

        # Push the image + updated metadata back into the document
        if hasattr(doc, "apply_edit"):
            _md = {**meta, "step_name": step_name}
            if command_id:
                _md["command_id"] = command_id
                _md["preset"] = dict(preset or {})
            doc.apply_edit(out_image, metadata=_md, step_name=step_name)
        else:
            doc.image = out_image
            try:
                setattr(doc, "metadata", {**meta, "step_name": step_name})
            except Exception:
                pass
            if hasattr(doc, "changed"):
                try:
                    doc.changed.emit()
                except Exception:
                    pass

        dbg = meta.get("__wcs_debug__")
        if isinstance(dbg, dict):
            try:
                self._show_wcs_update_popup(dbg, step_name=step_name)
            except Exception as e:
                print(f"[WCS-GEOM] Failed to show WCS popup for {step_name}: {e}")
                
    def _exec_geom_invert(self):
        """Execute invert operation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Invert"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_invert_to_doc(doc)
            self._log("Invert applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Invert", str(e))

    def _exec_geom_flip_h(self):
        """Execute horizontal flip on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Flip Horizontal"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_flip_h_to_doc(doc)
            self._log("Flip Horizontal applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Flip Horizontal", str(e))

    def _exec_geom_flip_v(self):
        """Execute vertical flip on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Flip Vertical"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_flip_v_to_doc(doc)
            self._log("Flip Vertical applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Flip Vertical", str(e))

    def _exec_geom_rot_cw(self):
        """Execute 90 degree clockwise rotation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate 90° CW"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_rot_cw_to_doc(doc)
            self._log("Rotate 90° CW applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 90° CW", str(e))

    def _exec_geom_rot_ccw(self):
        """Execute 90 degree counterclockwise rotation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate 90° CCW"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_rot_ccw_to_doc(doc)
            self._log("Rotate 90° CCW applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 90° CCW", str(e))

    def _exec_geom_rot_180(self):
        """Execute 180 degree rotation on active view."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate 180°"), self.tr("Active view has no image."))
            return
        try:
            self._apply_geom_rot_180_to_doc(doc)
            self._log("Rotate 180° applied to active view")
        except Exception as e:
            QMessageBox.critical(self, "Rotate 180°", str(e))

    def _exec_geom_rot_any(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rotate..."), self.tr("Active view has no image."))
            return

        if cv2 is None:
            QMessageBox.warning(self, self.tr("Rotate..."), self.tr("OpenCV (cv2) is required for arbitrary rotation."))
            return

        icon = None
        try:
            from setiastro.saspro.resources import rotatearbitrary_path
            icon = QIcon(rotatearbitrary_path)
        except Exception:
            pass

        dlg = RotateDialog(self, image=doc.image, icon=icon, initial=0.0)
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        angle = float(dlg.angle()) if accepted else 0.0
        dlg.deleteLater()
        if not accepted:
            return

        if abs(angle) < 1e-6:
            return
        try:
            self._apply_geom_rot_any_to_doc(doc, angle_deg=angle)
            self._log(f"Rotate ({angle:g}°) applied to active view")
        except Exception as e:
            QMessageBox.critical(self, self.tr("Rotate..."), str(e))


    def _exec_geom_rescale(self):
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Rescale Image"), self.tr("Active view has no image."))
            return

        arr = np.asarray(doc.image)
        h, w = arr.shape[:2]

        try:
            from setiastro.saspro.resources import rescale_path
            icon = QIcon(rescale_path)
        except Exception:
            icon = None

        dlg = RescaleDialog(self, cur_w=w, cur_h=h, icon=icon)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        vals = dlg.values()
        try:
            self._apply_geom_rescale_to_doc(doc, factor=vals["factor"], method=vals["method"])
            self._last_rescale_factor = vals["factor"]
            method_label = dlg.cmb_method.currentText().split("(")[0].strip()
            self._log(f"Rescale ({method_label}, {vals['factor']:g}×) applied to active view")

            if getattr(self, "_auto_fit_on_resize", False) and view is not None:
                if hasattr(view, "fit_to_window"):
                    view.fit_to_window()
                elif hasattr(self, "_zoom_active_fit"):
                    self._zoom_active_fit()

        except Exception as e:
            QMessageBox.critical(self, self.tr("Rescale Image"), str(e))

    def _exec_geom_resize_canvas(self):
        """Resize canvas without resampling pixels."""
        sw = self.mdi.activeSubWindow() if hasattr(self, "mdi") else None
        view = sw.widget() if sw else None
        doc = getattr(view, "document", None)
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.information(self, self.tr("Resize Canvas"), self.tr("Active view has no image."))
            return

        arr = np.asarray(doc.image)
        h, w = arr.shape[:2]

        try:
            from setiastro.saspro.resources import resizecanvas_path
            icon = QIcon(resizecanvas_path)
        except Exception:
            icon = None

        dlg = ResizeCanvasDialog(self, cur_w=w, cur_h=h, icon=icon)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        vals = dlg.values()

        try:
            self._apply_geom_resize_canvas_to_doc(
                doc,
                new_w=vals["new_w"],
                new_h=vals["new_h"],
                anchor=vals["anchor"],
                fill_value=vals["fill_value"],
                update_wcs=vals["update_wcs"],
            )
            self._log(
                f"Resize Canvas applied to active view "
                f"({w}x{h} -> {vals['new_w']}x{vals['new_h']}, "
                f"anchor={vals['anchor']}, units={vals.get('units', 'px')})"
            )
        except Exception as e:
            QMessageBox.critical(self, self.tr("Resize Canvas"), str(e))

    def _anchor_offsets(self, old_w: int, old_h: int, new_w: int, new_h: int, anchor: str):
        """
        Returns (dx, dy) where source pixel (0,0) lands in destination coordinates.
        Positive dx/dy means the old image is shifted right/down inside the new canvas.
        Negative dx/dy means the destination acts like a crop window into the source.
        """
        if anchor == "top-left":
            dx = 0
            dy = 0
        elif anchor == "top":
            dx = (new_w - old_w) // 2
            dy = 0
        elif anchor == "top-right":
            dx = new_w - old_w
            dy = 0
        elif anchor == "left":
            dx = 0
            dy = (new_h - old_h) // 2
        elif anchor == "center":
            dx = (new_w - old_w) // 2
            dy = (new_h - old_h) // 2
        elif anchor == "right":
            dx = new_w - old_w
            dy = (new_h - old_h) // 2
        elif anchor == "bottom-left":
            dx = 0
            dy = new_h - old_h
        elif anchor == "bottom":
            dx = (new_w - old_w) // 2
            dy = new_h - old_h
        elif anchor == "bottom-right":
            dx = new_w - old_w
            dy = new_h - old_h
        else:
            dx = (new_w - old_w) // 2
            dy = (new_h - old_h) // 2

        return int(dx), int(dy)

    # --- Geometry: headless apply-to-doc helpers ---

    def _apply_geom_invert_to_doc(self, doc):
        """Apply invert to document, respecting active mask if present."""
        arr = np.asarray(doc.image, dtype=np.float32)
        out = invert_image_numba(arr)

        # Blend with active mask if present
        mid = getattr(doc, "active_mask_id", None)
        if mid:
            masks = getattr(doc, "masks", {}) or {}
            layer = masks.get(mid)
            m = np.asarray(getattr(layer, "data", None)) if layer else None
            if m is not None and m.size > 0:
                m = np.clip(m.astype(np.float32), 0.0, 1.0)
                if m.ndim == 2 and out.ndim == 3:
                    m = m[:, :, None]
                # Resize mask if needed
                if m.shape[:2] != out.shape[:2]:
                    yi = np.linspace(0, m.shape[0] - 1, out.shape[0]).astype(np.int32)
                    xi = np.linspace(0, m.shape[1] - 1, out.shape[1]).astype(np.int32)
                    m = m[yi][:, xi]
                    if out.ndim == 3 and m.ndim == 2:
                        m = m[:, :, None]
                out = m * out + (1.0 - m) * arr

        meta = {
            "step_name": "Invert",
            "command_id": "geom_invert",
            "preset": {},
            "masked": bool(mid),
            "mask_id": mid,
            "mask_blend": "m*out+(1-m)*src",
        }
        if hasattr(doc, "apply_edit"):
            doc.apply_edit(out.astype(np.float32), metadata=meta, step_name="Invert")
        else:
            doc.image = out.astype(np.float32)

    def _apply_geom_flip_h_to_doc(self, doc):
        """Apply horizontal flip to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = flip_horizontal_numba(arr)

        M = np.array([
            [-1.0, 0.0, w - 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Flip Horizontal",
                                  command_id="geom_flip_horizontal")

    def _apply_geom_flip_v_to_doc(self, doc):
        """Apply vertical flip to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = flip_vertical_numba(arr)

        M = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, h - 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Flip Vertical",
                                  command_id="geom_flip_vertical")

    def _apply_geom_rot_cw_to_doc(self, doc):
        """Apply 90° clockwise rotation to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = rotate_90_clockwise_numba(arr)  # out shape: (w, h)

        M = np.array([
            [0.0, -1.0, h - 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Rotate 90° Clockwise",
                                  command_id="geom_rotate_clockwise")

    def _apply_geom_rot_ccw_to_doc(self, doc):
        """Apply 90° counterclockwise rotation to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = rotate_90_counterclockwise_numba(arr)  # out shape: (w, h)

        M = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, w - 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Rotate 90° Counterclockwise",
                                  command_id="geom_rotate_counterclockwise")

    def _apply_geom_rot_180_to_doc(self, doc):
        """Apply 180° rotation to document with WCS update."""
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]
        out = rotate_180_numba(arr)  # out shape: (h, w)

        # 180° rotation around the image center:
        # (x, y) -> (w-1 - x, h-1 - y)
        M = np.array([
            [-1.0, 0.0, w - 1.0],
            [0.0, -1.0, h - 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name="Rotate 180°",
                                  command_id="geom_rotate_180")

    def _apply_geom_rot_any_to_doc(self, doc, *, angle_deg: float):
        if cv2 is None:
            raise RuntimeError("cv2 is required for arbitrary rotation")

        src = np.asarray(doc.image, dtype=np.float32, order="C")
        h, w = src.shape[:2]

        # Rotation about center
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5

        # OpenCV uses CCW degrees
        A2 = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)  # 2x3

        # Convert to 3x3
        M = np.array([
            [A2[0,0], A2[0,1], A2[0,2]],
            [A2[1,0], A2[1,1], A2[1,2]],
            [0.0,     0.0,     1.0    ],
        ], dtype=np.float32)

        # Compute output bounds by rotating the four corners
        corners = np.array([
            [0.0, 0.0, 1.0],
            [w - 1.0, 0.0, 1.0],
            [w - 1.0, h - 1.0, 1.0],
            [0.0, h - 1.0, 1.0],
        ], dtype=np.float32).T  # 3x4

        rc = (M @ corners)  # 3x4
        xs = rc[0, :]
        ys = rc[1, :]

        min_x = float(xs.min())
        max_x = float(xs.max())
        min_y = float(ys.min())
        max_y = float(ys.max())

        out_w = int(math.ceil(max_x - min_x + 1.0))
        out_h = int(math.ceil(max_y - min_y + 1.0))
        if out_w <= 0 or out_h <= 0:
            raise RuntimeError("Invalid output size after rotation")

        # Shift so that min corner maps to (0,0)
        T = np.array([
            [1.0, 0.0, -min_x],
            [0.0, 1.0, -min_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        M = (T @ M).astype(np.float32)  # final src->dst 3x3

        # Warp
        # cv2.warpPerspective expects (W,H)
        flags = cv2.INTER_LANCZOS4
        if src.ndim == 2:
            out = cv2.warpPerspective(src, M, (out_w, out_h), flags=flags)
        else:
            # warpPerspective works on multi-channel too
            out = cv2.warpPerspective(src, M, (out_w, out_h), flags=flags)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M, step_name=f"Rotate ({angle_deg:g}°)",
                                  command_id="geom_rotate_any",
                                  preset={"angle_deg": float(angle_deg)})


    def _apply_geom_rescale_to_doc(self, doc, *, factor: float, method: str = "bilinear"):
        factor = float(factor)
        arr = np.asarray(doc.image, dtype=np.float32)
        h, w = arr.shape[:2]

        if method == "bilinear":
            factor = max(0.1, min(10.0, factor))
            out = rescale_image_numba(arr, factor)
        elif method == "bin2x2":
            out = bin2x2_numba(arr)
            factor = 0.5
        elif method == "bin3x3":
            out = _bin_NxN_numba(arr, 3)
            factor = 1/3
        elif method == "bin4x4":
            out = _bin_NxN_numba(arr, 4)
            factor = 0.25
        elif method == "up2x":
            out = _upsample_NxN(arr, 2)
            factor = 2.0
        elif method == "up3x":
            out = _upsample_NxN(arr, 3)
            factor = 3.0
        elif method == "up4x":
            out = _upsample_NxN(arr, 4)
            factor = 4.0
        else:
            factor = max(0.1, min(10.0, factor))
            out = rescale_image_numba(arr, factor)

        M = np.array([
            [factor, 0.0,    0.0],
            [0.0,    factor, 0.0],
            [0.0,    0.0,    1.0],
        ], dtype=float)

        self._apply_geom_with_wcs(doc, out, M_src_to_dst=M,
                                  step_name=f"Rescale ({method}, {factor:g}×)",
                                  command_id="geom_rescale",
                                  preset={"factor": float(factor), "method": str(method)})

    def _apply_geom_resize_canvas_to_doc(
        self,
        doc,
        *,
        new_w: int,
        new_h: int,
        anchor: str = "center",
        fill_value: float = 0.0,
        update_wcs: bool = True,
    ):
        """
        Resize canvas without resampling pixels. Supports both padding and cropping.
        WCS is updated by passing the equivalent src->dst translation matrix.
        """
        if new_w <= 0 or new_h <= 0:
            raise ValueError("Canvas size must be positive")

        src = np.asarray(doc.image, dtype=np.float32, order="C")
        old_h, old_w = src.shape[:2]

        if old_w == new_w and old_h == new_h:
            return

        dx, dy = self._anchor_offsets(old_w, old_h, new_w, new_h, anchor)

        if src.ndim == 2:
            out = np.full((new_h, new_w), fill_value, dtype=np.float32)
        else:
            out = np.full((new_h, new_w, src.shape[2]), fill_value, dtype=np.float32)

        # Source rectangle and destination rectangle intersection
        src_x0 = max(0, -dx)
        src_y0 = max(0, -dy)
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)

        copy_w = min(old_w - src_x0, new_w - dst_x0)
        copy_h = min(old_h - src_y0, new_h - dst_y0)

        if copy_w > 0 and copy_h > 0:
            if src.ndim == 2:
                out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
                    src[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
            else:
                out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w, :] = \
                    src[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w, :]

        M = None
        if update_wcs:
            M = np.array([
                [1.0, 0.0, float(dx)],
                [0.0, 1.0, float(dy)],
                [0.0, 0.0, 1.0],
            ], dtype=float)

        self._apply_geom_with_wcs(
            doc, out, M_src_to_dst=M,
            step_name=f"Resize Canvas ({old_w}×{old_h} → {new_w}×{new_h})",
            command_id="geom_resize_canvas",
            preset={"width": int(new_w), "height": int(new_h), "anchor": str(anchor),
                    "fill_value": float(fill_value), "update_wcs": bool(update_wcs)},
        )

    def _apply_geom_resize_canvas_preset_to_doc(self, doc, preset):
        """
        Flexible preset support:
        {
            "width": 4000,
            "height": 3000,
            "anchor": "center",
            "fill_value": 0.0,
            "update_wcs": True
        }
        """
        if not isinstance(preset, dict):
            raise ValueError("Resize Canvas preset must be a dict")

        new_w = int(preset.get("width", preset.get("new_w", 0)))
        new_h = int(preset.get("height", preset.get("new_h", 0)))
        anchor = str(preset.get("anchor", "center"))
        fill_value = float(preset.get("fill_value", 0.0))
        update_wcs = bool(preset.get("update_wcs", True))

        if new_w <= 0 or new_h <= 0:
            raise ValueError("Resize Canvas preset must include positive width and height")

        self._apply_geom_resize_canvas_to_doc(
            doc,
            new_w=new_w,
            new_h=new_h,
            anchor=anchor,
            fill_value=fill_value,
            update_wcs=update_wcs,
        )

    def _apply_geom_rescale_preset_to_doc(self, doc, preset):
        """
        Accepts flexible presets:
        - dict with 'factor' or 'scale'
        - a lone float/int
        - a '0.5x'/'2x' string
        - (factor, ...) tuple/list
        Falls back to 1.0 if unparsable.
        """
        factor = None
        try:
            if isinstance(preset, dict):
                factor = preset.get("factor", preset.get("scale", None))
            elif isinstance(preset, (float, int)):
                factor = float(preset)
            elif isinstance(preset, str):
                s = preset.strip().lower().replace("×", "x")
                if s.endswith("x"):
                    s = s[:-1]
                factor = float(s)
            elif isinstance(preset, (list, tuple)) and preset:
                factor = float(preset[0])
        except Exception:
            factor = None

        if factor is None:
            factor = getattr(self, "_last_rescale_factor", 1.0) or 1.0

        self._apply_geom_rescale_to_doc(doc, factor=factor)

    def _apply_rescale_preset_to_doc(self, doc, preset: dict):
        """
        Headless rescale for drag-and-drop / shortcut preset application.
        Expects preset like {"factor": 1.25}.
        """
        factor = float(preset.get("factor", 1.0))
        if not (0.10 <= factor <= 10.0):
            raise ValueError("Rescale factor must be between 0.10 and 10.0")

        if getattr(doc, "image", None) is None:
            raise RuntimeError("Target document has no image")

        src = np.asarray(doc.image, dtype=np.float32, order="C")
        out = rescale_image_numba(src, factor)

        if hasattr(doc, "set_image"):
            doc.set_image(out, step_name=f"Rescale ×{factor:.2f}")
        else:
            doc.image = out
