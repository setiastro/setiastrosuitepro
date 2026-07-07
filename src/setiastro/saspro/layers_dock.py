# ============================================================
#  Layers Dock  (src/setiastro/saspro/layers_dock.py)
#  Part of Seti Astro Suite Pro
#  Copyright © 2026 Franklin Marek  |  www.setiastro.com
#  All rights reserved.
# ============================================================
from __future__ import annotations
from typing import Optional
import json
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QByteArray, QTimer, QPoint, QPointF, QRectF, QThread, QObject
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QSlider, QCheckBox,
    QPushButton, QFrame, QMessageBox, QDialog, QFormLayout, QDoubleSpinBox,
    QDialogButtonBox, QSizePolicy, QToolButton, QScrollArea,
)
from PyQt6.QtGui import (
    QIcon, QDragEnterEvent, QDropEvent, QPixmap, QCursor, QImage, QPainter,
    QPen, QColor, QFont,
)

from setiastro.saspro.layers import LayerTransform
from setiastro.saspro.dnd_mime import MIME_VIEWSTATE, MIME_MASK
from setiastro.saspro.layers import composite_stack, ImageLayer, BLEND_MODES, _apply_levels, _ensure_3c, _float01


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
    """float32 H×W×3 [0,1] → QPixmap."""
    rgb8 = np.clip(arr[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    h, w = rgb8.shape[:2]
    img = QImage(rgb8.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img)


# ─────────────────────────────────────────────────────────────
# Background composite worker
# ─────────────────────────────────────────────────────────────

class _CompositeWorker(QObject):
    """
    Runs composite_stack off the GUI thread.
    Emits done(gen, result_array) — caller discards if gen < current_gen.
    """
    done = pyqtSignal(int, object)   # (generation, np.ndarray)

    def __init__(self, gen: int, working_base: np.ndarray,
                 layers: list, parent=None):
        super().__init__(parent)
        self._gen         = gen
        self._working_base = working_base
        self._layers      = layers

    def run(self):
        try:
            if self._layers:
                result = composite_stack(self._working_base, self._layers)
                out = result if result is not None else self._working_base
            else:
                out = _ensure_3c(_float01(self._working_base))
            self.done.emit(self._gen, out)
        except Exception as ex:
            print("[CompositeWorker] error:", ex)
            self.done.emit(self._gen, self._working_base)



# ─────────────────────────────────────────────────────────────
# Split-preview canvas  (before/after or after-only)
# ─────────────────────────────────────────────────────────────

class _PreviewCanvas(QWidget):
    """Zoom/pan canvas with optional vertical split before/after line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._before: Optional[QPixmap] = None   # raw base image
        self._after:  Optional[QPixmap] = None   # composited result
        self._split_mode = True                  # True = before/after split
        self._split = 0.5

        self._zoom   = 1.0
        self._offset = QPoint(0, 0)
        self._fit_mode = True

        self._panning = False
        self._pan_start_mouse  = QPoint()
        self._pan_start_offset = QPoint()
        self._dragging_split   = False

    # ── public API ───────────────────────────────────────────

    def set_images(self, before: QPixmap, after: QPixmap):
        self._before = before
        self._after  = after
        self._fit_mode = True
        self._fit()
        self.update()

    def set_after(self, after: QPixmap):
        self._after = after
        self.update()

    def set_split_mode(self, on: bool):
        self._split_mode = bool(on)
        self.update()

    def reset_view(self):
        self._fit_mode = True
        self._fit()
        self.update()

    # ── geometry ─────────────────────────────────────────────

    def _fit(self):
        src = self._before or self._after
        if src is None:
            return
        pw, ph = src.width(), src.height()
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
        src = self._before or self._after
        if src is None:
            return QRectF()
        return QRectF(
            self._offset.x(), self._offset.y(),
            src.width()  * self._zoom,
            src.height() * self._zoom,
        )

    def _near_split(self, x: int) -> bool:
        if not self._split_mode:
            return False
        r = self._img_rect()
        sx = r.left() + self._split * r.width()
        return abs(x - sx) < 7

    # ── paint ─────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 28))

        src = self._before or self._after
        if src is None:
            p.setPen(QColor(100, 100, 110))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "No image — add a layer or click Preview.")
            p.end()
            return

        r = self._img_rect()

        if not self._split_mode or self._before is None or self._after is None:
            # Single-view: show after (or before if after missing)
            pm = self._after if self._after is not None else self._before
            p.drawPixmap(r, pm, QRectF(0, 0, pm.width(), pm.height()))
        else:
            split_x = r.left() + self._split * r.width()
            src_w = self._before.width()
            src_split = int(self._split * src_w)

            # Before (left)
            p.drawPixmap(
                QRectF(r.left(), r.top(), split_x - r.left(), r.height()),
                self._before,
                QRectF(0, 0, src_split, self._before.height()),
            )
            # After (right)
            after_src_w = self._after.width()
            after_split = int(self._split * after_src_w)
            p.drawPixmap(
                QRectF(split_x, r.top(), r.right() - split_x, r.height()),
                self._after,
                QRectF(after_split, 0, after_src_w - after_split, self._after.height()),
            )

            # Split line
            p.setPen(QPen(QColor(255, 220, 60), 2))
            p.drawLine(QPointF(split_x, r.top()), QPointF(split_x, r.bottom()))

            # Labels
            p.setFont(QFont("Segoe UI", 9))
            lbl_w = 54
            if split_x - r.left() > lbl_w + 4:
                p.setPen(QColor(200, 200, 200, 180))
                p.fillRect(QRectF(r.left() + 4, r.top() + 4, lbl_w, 18), QColor(0, 0, 0, 120))
                p.drawText(QRectF(r.left() + 4, r.top() + 4, lbl_w, 18),
                           Qt.AlignmentFlag.AlignCenter, "Before")
            if r.right() - split_x > lbl_w + 4:
                p.setPen(QColor(200, 200, 200, 180))
                p.fillRect(QRectF(r.right() - lbl_w - 4, r.top() + 4, lbl_w, 18), QColor(0, 0, 0, 120))
                p.drawText(QRectF(r.right() - lbl_w - 4, r.top() + 4, lbl_w, 18),
                           Qt.AlignmentFlag.AlignCenter, "After")
        p.end()

    # ── mouse ─────────────────────────────────────────────────

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
        if not self._near_split(int(ev.position().x())):
            self.reset_view()

    def wheelEvent(self, ev):
        dy = ev.pixelDelta().y() or ev.angleDelta().y()
        if dy == 0:
            return
        factor = 1.15 if dy > 0 else 1.0 / 1.15
        old_zoom = self._zoom
        self._zoom = max(0.05, min(32.0, self._zoom * factor))
        self._fit_mode = False
        pos = ev.position().toPoint()
        self._offset = QPoint(
            int(pos.x() - (pos.x() - self._offset.x()) * self._zoom / old_zoom),
            int(pos.y() - (pos.y() - self._offset.y()) * self._zoom / old_zoom),
        )
        self.update()
        ev.accept()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._fit_mode:
            self._fit()
            self.update()


# ─────────────────────────────────────────────────────────────
# Layers Preview Window
# ─────────────────────────────────────────────────────────────

class LayersPreviewWindow(QDialog):
    """
    Floating non-modal preview window for the Layers dock.

    Shows the composited result against the raw base image.
    Never touches the base document — read-only display only.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layers Preview")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self._last_before_id: int = 0
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        except Exception:
            pass
        self.resize(760, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── toolbar ──────────────────────────────────────────
        bar = QHBoxLayout()

        self.btn_split = QPushButton("⧉  Split Before/After")
        self.btn_split.setCheckable(True)
        self.btn_split.setChecked(True)
        self.btn_split.setToolTip("Toggle before/after split view")
        self.btn_split.toggled.connect(self._on_split_toggled)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setFixedWidth(48)
        self.btn_fit.setToolTip("Reset zoom/pan to fit  (double-click preview also does this)")
        self.btn_fit.clicked.connect(self.canvas.reset_view if False else lambda: self.canvas.reset_view())

        self.lbl_info = QLabel("Ready.")
        self.lbl_info.setStyleSheet("color: #888; font-size: 10px;")

        bar.addWidget(self.btn_split)
        bar.addWidget(self.btn_fit)
        bar.addStretch(1)
        bar.addWidget(self.lbl_info)
        root.addLayout(bar)

        # ── canvas ───────────────────────────────────────────
        self.canvas = _PreviewCanvas(self)
        root.addWidget(self.canvas, 1)

        # Fix btn_fit now canvas exists
        self.btn_fit.clicked.disconnect()
        self.btn_fit.clicked.connect(self.canvas.reset_view)

    def _on_split_toggled(self, on: bool):
        self.canvas.set_split_mode(on)
        self.btn_split.setText("⧉  Split Before/After" if on else "□  After Only")

    def update_composite(self, before_arr: Optional[np.ndarray],
                          after_arr: Optional[np.ndarray]):
        """Called by the dock whenever a new composite is ready."""
        if before_arr is not None:
            before_px = _arr_to_pixmap(
                _ensure_3c(_float01(np.asarray(before_arr, dtype=np.float32)))
            )
        else:
            before_px = None

        if after_arr is not None:
            after_px = _arr_to_pixmap(
                _ensure_3c(_float01(np.asarray(after_arr, dtype=np.float32)))
            )
        else:
            after_px = None

        # set_images resets zoom/pan — only call when base actually changes.
        # For parameter tweaks id() of before_arr stays the same → set_after only.
        before_id = id(before_arr) if before_arr is not None else 0
        if before_id != self._last_before_id:
            self._last_before_id = before_id
            self.canvas.set_images(before_px, after_px)
        else:
            if before_px is not None and self.canvas._before is None:
                self.canvas._before = before_px
            if after_px is not None:
                self.canvas.set_after(after_px)

        n_px = after_arr.size if after_arr is not None else 0
        self.lbl_info.setText(f"Composite ready — {n_px:,} px")

    def closeEvent(self, ev):
        # Hide rather than destroy so the dock can reopen it cheaply
        ev.ignore()
        self.hide()


# ─────────────────────────────────────────────────────────────
# Transform dialog (unchanged from original)
# ─────────────────────────────────────────────────────────────

class _TransformDialog(QDialog):
    def __init__(self, parent=None, *, t: LayerTransform | None = None,
                 apply_cb=None, cancel_cb=None, debounce_ms: int = 200):
        super().__init__(parent)
        self.setWindowTitle("Layer Transform")
        self.setModal(True)

        self._apply_cb  = apply_cb
        self._cancel_cb = cancel_cb
        self._t0 = t or LayerTransform()
        self._orig = LayerTransform(
            tx=float(self._t0.tx), ty=float(self._t0.ty),
            rot_deg=float(self._t0.rot_deg),
            sx=float(self._t0.sx), sy=float(self._t0.sy),
            pivot_x=self._t0.pivot_x, pivot_y=self._t0.pivot_y,
        )
        self._tmr = QTimer(self)
        self._tmr.setSingleShot(True)
        self._tmr.timeout.connect(self._apply_live)
        self._debounce_ms = int(debounce_ms)

        lay  = QVBoxLayout(self)
        form = QFormLayout()
        lay.addLayout(form)

        def mk(lo, hi, step, dec, val):
            s = QDoubleSpinBox(self)
            s.setRange(lo, hi); s.setSingleStep(step)
            s.setDecimals(dec); s.setValue(float(val))
            return s

        self.tx  = mk(-100000, 100000, 1.0,  2, self._t0.tx)
        self.ty  = mk(-100000, 100000, 1.0,  2, self._t0.ty)
        self.rot = mk(-3600,   3600,   0.1,  3, self._t0.rot_deg)
        self.sx  = mk(0.001,   1000.0, 0.01, 4, self._t0.sx)
        self.sy  = mk(0.001,   1000.0, 0.01, 4, self._t0.sy)

        form.addRow("Translate X (px)", self.tx)
        form.addRow("Translate Y (px)", self.ty)
        form.addRow("Rotate (deg)",     self.rot)
        form.addRow("Scale X",          self.sx)
        form.addRow("Scale Y",          self.sy)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self)
        lay.addWidget(btns)
        self.btn_reset = QPushButton("Reset")
        btns.addButton(self.btn_reset, QDialogButtonBox.ButtonRole.ResetRole)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self.btn_reset.clicked.connect(self._reset)

        for w in (self.tx, self.ty, self.rot, self.sx, self.sy):
            w.valueChanged.connect(self._schedule_live)
        self._schedule_live()

    def _reset(self):
        for w, v in ((self.tx,0),(self.ty,0),(self.rot,0),(self.sx,1),(self.sy,1)):
            w.setValue(v)
        self._schedule_live()

    def _schedule_live(self, *_):
        self._tmr.start(self._debounce_ms)

    def _current_transform(self) -> LayerTransform:
        return LayerTransform(
            tx=float(self.tx.value()), ty=float(self.ty.value()),
            rot_deg=float(self.rot.value()),
            sx=float(self.sx.value()), sy=float(self.sy.value()),
            pivot_x=None, pivot_y=None,
        )

    def _apply_live(self):
        if callable(self._apply_cb):
            self._apply_cb(self._current_transform())

    def reject(self):
        if callable(self._cancel_cb):
            self._cancel_cb(self._orig)
        super().reject()

    def accept(self):
        try:
            if self._tmr.isActive():
                self._tmr.stop()
        except Exception:
            pass
        self._apply_live()
        super().accept()


# ─────────────────────────────────────────────────────────────
# Layer row widget
# ─────────────────────────────────────────────────────────────

class _LayerRow(QWidget):
    changed          = pyqtSignal()
    requestDelete    = pyqtSignal()
    moveUp           = pyqtSignal()
    moveDown         = pyqtSignal()
    requestTransform = pyqtSignal()

    def __init__(self, name: str, mode: str = "Normal", opacity: float = 1.0,
                 visible: bool = True, parent=None, *, is_base: bool = False):
        super().__init__(parent)
        self._name    = name
        self._is_base = bool(is_base)

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 2, 6, 2)

        # ── row 1: visibility / name / mode / opacity / buttons ──
        r1 = QHBoxLayout(); v.addLayout(r1)
        self.chk  = QCheckBox(); self.chk.setChecked(visible)
        self.lbl  = QLabel(name)
        self.mode = QComboBox(); self.mode.addItems(BLEND_MODES)
        try:
            self.mode.setCurrentIndex(max(0, BLEND_MODES.index(mode)))
        except Exception:
            self.mode.setCurrentIndex(0)

        # Opacity: slider + spinbox paired
        self.sld = QSlider(Qt.Orientation.Horizontal)
        self.sld.setRange(0, 100)
        self.sld.setValue(int(round(opacity * 100)))
        self.spin_opacity = QDoubleSpinBox()
        self.spin_opacity.setRange(0.0, 1.0)
        self.spin_opacity.setSingleStep(0.01)
        self.spin_opacity.setDecimals(2)
        self.spin_opacity.setFixedWidth(58)
        self.spin_opacity.setValue(float(opacity))

        self.btn_up = QPushButton("↑"); self.btn_up.setFixedWidth(28)
        self.btn_dn = QPushButton("↓"); self.btn_dn.setFixedWidth(28)
        self.btn_x  = QPushButton("✕"); self.btn_x.setFixedWidth(28)
        self.btn_tf = QPushButton("Transform…"); self.btn_tf.setFixedWidth(92)

        r1.addWidget(self.chk)
        r1.addWidget(self.lbl, 1)
        r1.addWidget(self.mode)
        r1.addWidget(QLabel("Opacity"))
        r1.addWidget(self.sld, 1)
        r1.addWidget(self.spin_opacity)
        r1.addWidget(self.btn_tf)
        r1.addWidget(self.btn_up)
        r1.addWidget(self.btn_dn)
        r1.addWidget(self.btn_x)

        # ── row 2: mask controls ──────────────────────────────────
        r2 = QHBoxLayout(); v.addLayout(r2)
        self.mask_combo  = QComboBox(); self.mask_combo.setMinimumWidth(140)
        self.mask_combo.setPlaceholderText("Mask: (none)")
        self.mask_invert    = QCheckBox("Invert")
        self.btn_clear_mask = QPushButton("Clear"); self.btn_clear_mask.setFixedWidth(52)
        r2.addWidget(QLabel("Mask"))
        r2.addWidget(self.mask_combo, 1)
        r2.addWidget(self.mask_invert)
        r2.addWidget(self.btn_clear_mask)

        # ── row 3: levels ─────────────────────────────────────────
        r3 = QHBoxLayout(); v.addLayout(r3)
        self.levels_enable = QCheckBox()
        self.levels_enable.setToolTip("Enable per-layer levels adjustment")

        def _mk_dspin(lo, hi, step, dec, val):
            s = QDoubleSpinBox()
            s.setRange(lo, hi); s.setSingleStep(step)
            s.setDecimals(dec); s.setValue(float(val))
            return s

        self.levels_bp_label = QLabel("Black")
        self.levels_bp       = _mk_dspin(0.0, 1.0, 0.01, 3, 0.0)
        self.levels_wp_label = QLabel("White")
        self.levels_wp       = _mk_dspin(0.0, 1.0, 0.01, 3, 1.0)
        self.levels_mt_label = QLabel("Midtones")
        self.levels_mt       = _mk_dspin(0.01, 1.0, 0.01, 3, 0.5)

        r3.addWidget(self.levels_enable)
        r3.addWidget(self.levels_bp_label); r3.addWidget(self.levels_bp)
        r3.addWidget(self.levels_mt_label); r3.addWidget(self.levels_mt)
        r3.addWidget(self.levels_wp_label); r3.addWidget(self.levels_wp)
        r3.addStretch(1)

        # ── row 4: sigmoid (non-base only) ────────────────────────
        self.sig_center_label = self.sig_center = None
        self.sig_strength_label = self.sig_strength = None
        if not self._is_base:
            r4 = QHBoxLayout(); v.addLayout(r4)
            self.sig_center_label = QLabel("Sigmoid center")
            self.sig_center       = _mk_dspin(0.0, 1.0, 0.01, 3, 0.5)
            self.sig_strength_label = QLabel("Strength")
            self.sig_strength     = _mk_dspin(0.1, 50.0, 0.5, 2, 10.0)
            r4.addWidget(self.sig_center_label); r4.addWidget(self.sig_center)
            r4.addWidget(self.sig_strength_label); r4.addWidget(self.sig_strength)
            r4.addStretch(1)

        # ── wire signals ──────────────────────────────────────────
        if self._is_base:
            for w in (self.chk, self.mode, self.sld, self.spin_opacity,
                      self.btn_up, self.btn_tf, self.btn_dn, self.btn_x,
                      self.mask_combo, self.mask_invert, self.btn_clear_mask):
                if w is not None:
                    w.setEnabled(False)
            self.lbl.setStyleSheet("color: palette(mid);")
            self.levels_enable.toggled.connect(self._update_levels_enabled_ui)
            self.levels_enable.toggled.connect(self._emit)
            self.levels_bp.valueChanged.connect(self._clamp_levels_ui)
            self.levels_wp.valueChanged.connect(self._clamp_levels_ui)
            self.levels_mt.valueChanged.connect(self._emit)
        else:
            # Opacity: slider ↔ spinbox bidirectional sync
            self.sld.valueChanged.connect(
                lambda v: (self.spin_opacity.blockSignals(True),
                           self.spin_opacity.setValue(v / 100.0),
                           self.spin_opacity.blockSignals(False),
                           self._emit()))
            self.spin_opacity.valueChanged.connect(
                lambda v: (self.sld.blockSignals(True),
                           self.sld.setValue(int(round(v * 100))),
                           self.sld.blockSignals(False),
                           self._emit()))

            self.chk.stateChanged.connect(self._emit)
            self.mode.currentIndexChanged.connect(self._on_mode_changed)
            self.mask_combo.currentIndexChanged.connect(self._emit)
            self.mask_invert.stateChanged.connect(self._emit)
            self.btn_clear_mask.clicked.connect(self._on_clear_mask)
            self.btn_x.clicked.connect(self.requestDelete.emit)
            self.btn_up.clicked.connect(self.moveUp.emit)
            self.btn_dn.clicked.connect(self.moveDown.emit)
            self.btn_tf.clicked.connect(self.requestTransform.emit)

            self.levels_enable.toggled.connect(self._update_levels_enabled_ui)
            self.levels_enable.toggled.connect(self._emit)
            self.levels_bp.valueChanged.connect(self._clamp_levels_ui)
            self.levels_wp.valueChanged.connect(self._clamp_levels_ui)
            self.levels_mt.valueChanged.connect(self._emit)

            if self.sig_center is not None:
                self.sig_center.valueChanged.connect(self._emit)
            if self.sig_strength is not None:
                self.sig_strength.valueChanged.connect(self._emit)

            self.mode.currentIndexChanged.connect(
                lambda _: self._update_extra_controls(self.mode.currentText()))

        self._update_levels_enabled_ui()
        if not self._is_base:
            self._update_extra_controls(self.mode.currentText())

    # ── public setters ────────────────────────────────────────

    def setName(self, name: str):
        self._name = name
        self.lbl.setText(name)

    def setTransformDirty(self, dirty: bool):
        if self.btn_tf is not None:
            self.btn_tf.setText("Transform… *" if dirty else "Transform…")

    def set_levels_enabled(self, enabled: bool):
        self.levels_enable.blockSignals(True)
        self.levels_enable.setChecked(bool(enabled))
        self.levels_enable.blockSignals(False)
        self._update_levels_enabled_ui()

    def set_levels_params(self, bp: float, wp: float, mt: float):
        for w, v in ((self.levels_bp, bp), (self.levels_wp, wp), (self.levels_mt, mt)):
            w.blockSignals(True); w.setValue(float(v)); w.blockSignals(False)

    def set_sigmoid_params(self, center: float, strength: float):
        if self.sig_center is None:
            return
        self.sig_center.blockSignals(True);   self.sig_center.setValue(float(center));   self.sig_center.blockSignals(False)
        self.sig_strength.blockSignals(True);  self.sig_strength.setValue(float(strength)); self.sig_strength.blockSignals(False)
        self._update_extra_controls(self.mode.currentText())

    # ── internal ─────────────────────────────────────────────

    def _update_levels_enabled_ui(self):
        enabled = bool(self.levels_enable.isChecked())
        for w in (self.levels_bp_label, self.levels_bp,
                  self.levels_mt_label, self.levels_mt,
                  self.levels_wp_label, self.levels_wp):
            if w is not None:
                w.setEnabled(enabled)

    def _update_extra_controls(self, mode_text: str):
        is_sig = (mode_text == "Sigmoid")
        for w in (self.sig_center_label, self.sig_center,
                  self.sig_strength_label, self.sig_strength):
            if w is not None:
                w.setVisible(is_sig)
        # Do NOT call adjustSize()/updateGeometry() — collapses the opacity slider

    def _on_mode_changed(self, _idx):
        self._update_extra_controls(self.mode.currentText())
        lay = self.layout()
        if lay:
            lay.invalidate(); lay.activate()
        self._emit()

    def _clamp_levels_ui(self, *_):
        bp = float(self.levels_bp.value())
        wp = float(self.levels_wp.value())
        if wp <= bp:
            if self.sender() is self.levels_bp:
                self.levels_wp.blockSignals(True)
                self.levels_wp.setValue(min(1.0, bp + 0.001))
                self.levels_wp.blockSignals(False)
            else:
                self.levels_bp.blockSignals(True)
                self.levels_bp.setValue(max(0.0, wp - 0.001))
                self.levels_bp.blockSignals(False)
        self._emit()

    def _on_clear_mask(self):
        self.mask_combo.setCurrentIndex(0)
        self._emit()

    def _emit(self, *_):
        self.changed.emit()

    def params(self) -> dict:
        out = {
            "visible":     self.chk.isChecked(),
            "mode":        self.mode.currentText(),
            "opacity":     self.spin_opacity.value(),
            "name":        self._name,
            "mask_index":  self.mask_combo.currentIndex(),
            "mask_invert": self.mask_invert.isChecked(),
            "levels_enabled": self.levels_enable.isChecked(),
            "black_point": self.levels_bp.value(),
            "white_point": self.levels_wp.value(),
            "midtones":    self.levels_mt.value(),
        }
        if self.sig_center is not None:
            out["sigmoid_center"]   = self.sig_center.value()
            out["sigmoid_strength"] = self.sig_strength.value()
        return out


# ─────────────────────────────────────────────────────────────
# Layers Dock
# ─────────────────────────────────────────────────────────────

class LayersDock(QDockWidget):
    def __init__(self, main_window):
        super().__init__("Layers", main_window)
        self.setObjectName("LayersDock")
        self.mw     = main_window
        self.docman = main_window.docman
        self._wired_title_sources = set()

        # ── Preview window (lazy-created) ─────────────────────
        self._preview_win: Optional[LayersPreviewWindow] = None

        # ── Cached composite (numpy float32) ─────────────────
        # Always kept up to date in the background; the base view is
        # NEVER updated by layers anymore — only push/merge touches it.
        self._cached_composite: Optional[np.ndarray] = None
        self._cached_before:    Optional[np.ndarray] = None   # raw base image

        # Cached converted before — only rebuilt when base_doc.image identity changes
        self._cached_before_display: Optional[np.ndarray] = None
        self._cached_before_img_id:  int = 0

        # Worker thread state
        self._composite_gen:    int             = 0   # incremented on every schedule
        self._composite_thread: Optional[QThread] = None
        self._composite_worker: Optional[_CompositeWorker] = None

        # ── Timers ────────────────────────────────────────────
        # Fast debounce → immediate downsampled preview on GUI thread.
        # Slow debounce → full-res composite on worker thread.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)    # ms — snappy fast preview
        self._preview_timer.timeout.connect(self._run_fast_preview)

        self._composite_timer = QTimer(self)
        self._composite_timer.setSingleShot(True)
        self._composite_timer.setInterval(350)  # ms — full-res after drag settles
        self._composite_timer.timeout.connect(self._run_composite_threaded)

        # ── UI ────────────────────────────────────────────────
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(8, 8, 8, 8)

        # Top bar: view selector + preview toggle
        top = QHBoxLayout(); v.addLayout(top)
        top.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        top.addWidget(self.view_combo, 1)

        self.btn_preview = QPushButton("👁  Preview")
        self.btn_preview.setCheckable(True)
        self.btn_preview.setChecked(False)
        self.btn_preview.setToolTip(
            "Open the composite preview window.\n"
            "The base image in the main view is never modified\n"
            "until you click Merge → Push to View."
        )
        self.btn_preview.toggled.connect(self._on_preview_toggled)
        top.addWidget(self.btn_preview)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setAlternatingRowColors(True)
        v.addWidget(self.list, 1)

        # Bottom buttons
        row = QHBoxLayout(); v.addLayout(row)
        self.btn_clear     = QPushButton("Clear All Layers")
        self.btn_merge     = QPushButton("Merge → Push to View")
        self.btn_merge_new = QPushButton("Merge → New Document")
        self.btn_merge_sel = QPushButton("Merge Selected → Single Layer")
        self.btn_merge.setToolTip(
            "Flatten the cached composite and push it to the current view as an undo step.")
        self.btn_merge_new.setToolTip(
            "Flatten the cached composite into a new document.")
        self.btn_merge_sel.setToolTip(
            "Merge selected layers into a single raster layer.")
        row.addWidget(self.btn_merge)
        row.addWidget(self.btn_merge_new)
        row.addWidget(self.btn_merge_sel)
        row.addStretch(1)
        row.addWidget(self.btn_clear)
        self.setWidget(w)

        self.setAcceptDrops(True)

        # Signals
        self.view_combo.currentIndexChanged.connect(self._on_pick_view)
        self.btn_clear.clicked.connect(self._clear_layers)
        self.mw.mdi.subWindowActivated.connect(lambda _sw: self._refresh_views())
        self.docman.documentAdded.connect(lambda _d: self._refresh_views())
        self.docman.documentRemoved.connect(lambda _d: self._refresh_views())
        self.btn_merge.clicked.connect(self._merge_and_push)
        self.btn_merge_new.clicked.connect(self._merge_to_new_doc)
        self.btn_merge_sel.clicked.connect(self._merge_selected_to_single_layer)

        self._refresh_views()

    # ─────────────────────────────────────────────────────────
    # Preview window management
    # ─────────────────────────────────────────────────────────

    def _ensure_preview_win(self) -> LayersPreviewWindow:
        if self._preview_win is None:
            self._preview_win = LayersPreviewWindow(parent=self.mw)
            self._preview_win.finished.connect(self._on_preview_win_closed)
        return self._preview_win

    def _on_preview_toggled(self, on: bool):
        if on:
            pw = self._ensure_preview_win()
            pw.show(); pw.raise_(); pw.activateWindow()
            # Immediately push whatever we have cached
            if self._cached_before is not None or self._cached_composite is not None:
                pw.update_composite(self._cached_before, self._cached_composite)
            else:
                # Trigger a fresh composite
                self._schedule_composite()
        else:
            if self._preview_win is not None:
                self._preview_win.hide()

    def _on_preview_win_closed(self, *_):
        self.btn_preview.blockSignals(True)
        self.btn_preview.setChecked(False)
        self.btn_preview.blockSignals(False)

    def _push_to_preview(self):
        """Send the current cached composite to the preview window (if open)."""
        if self._preview_win is None or not self._preview_win.isVisible():
            return
        self._preview_win.update_composite(self._cached_before, self._cached_composite)

    # ─────────────────────────────────────────────────────────
    # Composite engine
    # ─────────────────────────────────────────────────────────

    def _schedule_composite(self):
        """
        Kick off both tiers:
          - fast preview fires after 80 ms (GUI thread, downsampled)
          - full-res composite fires after 350 ms (worker thread)
        Both timers restart on every call so rapid changes coalesce.
        """
        self._preview_timer.start()
        self._composite_timer.start()

    def _get_working_base(self, vw) -> Optional[np.ndarray]:
        """
        Return the base image with levels applied, using a cached
        converted copy so we never re-convert on every composite call
        unless base_doc.image actually changes.
        """
        base_doc = getattr(vw, "document", None)
        if base_doc is None or getattr(base_doc, "image", None) is None:
            return None

        base_img = base_doc.image
        img_id   = id(base_img)

        # Cache raw before (for id() tracking in preview window)
        if img_id != self._cached_before_img_id:
            self._cached_before_img_id    = img_id
            self._cached_before           = np.asarray(base_img, dtype=np.float32)
            # Also invalidate the display-ready conversion
            self._cached_before_display   = _ensure_3c(_float01(self._cached_before))

        # Apply base levels on top of the cached 3c float version
        base_levels = getattr(vw, "_base_levels", None)
        if base_levels and base_levels.get("enabled", False):
            return _apply_levels(
                self._cached_before_display,
                float(base_levels.get("black_point", 0.0)),
                float(base_levels.get("white_point", 1.0)),
                float(base_levels.get("midtones",    0.5)),
            )
        return self._cached_before_display

    def _run_fast_preview(self):
        """
        Immediate downsampled composite on the GUI thread.
        Runs at preview-window resolution so it's cheap and fast.
        """
        if self._preview_win is None or not self._preview_win.isVisible():
            return

        vw = self.current_view()
        if vw is None:
            return

        working_base = self._get_working_base(vw)
        if working_base is None:
            return

        # Determine a sensible max_dim from the preview canvas size
        try:
            cw = self._preview_win.canvas.width()
            ch = self._preview_win.canvas.height()
            max_dim = max(cw, ch, 512)
        except Exception:
            max_dim = 900

        layers = list(getattr(vw, "_layers", []) or [])
        try:
            if layers:
                result = composite_stack(working_base, layers, max_dim=max_dim)
                fast_composite = result if result is not None else working_base
            else:
                fast_composite = _ensure_3c(_float01(working_base))
        except Exception as ex:
            print("[LayersDock] fast preview error:", ex)
            return

        # Push fast result — will be overwritten by full-res shortly
        self._preview_win.update_composite(self._cached_before, fast_composite)

    def _run_composite_threaded(self):
        """
        Full-resolution composite on a worker thread.
        Uses a generation counter to discard stale results.
        """
        vw = self.current_view()
        if vw is None:
            return

        working_base = self._get_working_base(vw)
        if working_base is None:
            return

        layers = list(getattr(vw, "_layers", []) or [])

        # Increment generation — any in-flight worker with an older gen
        # will emit its result and it will be silently dropped.
        self._composite_gen += 1
        gen = self._composite_gen

        # Cancel previous thread if still running (rare — 350 ms debounce)
        self._teardown_composite_thread()

        worker = _CompositeWorker(gen, working_base, layers)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run, Qt.ConnectionType.QueuedConnection)
        worker.done.connect(self._on_composite_done, Qt.ConnectionType.QueuedConnection)
        worker.done.connect(lambda *_: self._teardown_composite_thread())
        self._composite_worker = worker
        self._composite_thread = thread
        thread.start()

    def _teardown_composite_thread(self):
        try:
            if self._composite_thread is not None:
                self._composite_thread.quit()
                self._composite_thread.wait(500)
        except Exception:
            pass
        self._composite_thread = None
        self._composite_worker = None

    def _on_composite_done(self, gen: int, result: np.ndarray):
        """Receive full-res composite — discard if a newer render is pending."""
        if gen < self._composite_gen:
            return   # stale — a newer render is already in flight
        self._cached_composite = result
        self._push_to_preview()

    def _run_composite(self):
        """
        Synchronous full-res composite — used only by merge/push operations
        where we need the result immediately on the calling thread.
        """
        vw = self.current_view()
        if vw is None:
            return

        working_base = self._get_working_base(vw)
        if working_base is None:
            return

        layers = list(getattr(vw, "_layers", []) or [])
        if layers:
            result = composite_stack(working_base, layers)
            self._cached_composite = result if result is not None else working_base
        else:
            self._cached_composite = _ensure_3c(_float01(working_base))

        self._push_to_preview()

    # ─────────────────────────────────────────────────────────
    # Row → layer sync + scheduling
    # ─────────────────────────────────────────────────────────

    def _on_row_changed(self):
        """Called on every row widget change — sync layers then debounce composite."""
        vw = self.current_view()
        if vw:
            self._sync_layers_from_rows(vw)
        self._schedule_composite()
        self._refresh_row_heights()

    # ─────────────────────────────────────────────────────────
    # View management
    # ─────────────────────────────────────────────────────────

    def _mask_choices(self):
        return [(sw._effective_title() or "Untitled", sw.document)
                for sw in self._all_subwindows()]

    def _all_subwindows(self):
        from setiastro.saspro.subwindow import ImageSubWindow
        return [sw.widget() for sw in self.mw.mdi.subWindowList()
                if isinstance(sw.widget(), ImageSubWindow)]

    def _refresh_views(self):
        subs = self._all_subwindows()
        current_uid = self._current_view_uid()

        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        for w in subs:
            title = w._effective_title() or "Untitled"
            uid   = getattr(getattr(w, "document", None), "uid", None) or id(w)
            self.view_combo.addItem(title, userData=uid)
        self.view_combo.blockSignals(False)

        if current_uid is not None:
            for i in range(self.view_combo.count()):
                if self.view_combo.itemData(i) == current_uid:
                    self.view_combo.setCurrentIndex(i); break
            else:
                if subs: self.view_combo.setCurrentIndex(0)
        elif subs:
            self.view_combo.setCurrentIndex(0)

        self._wire_title_change_listeners(subs)
        self._rebuild_list()

    def _wire_title_change_listeners(self, subs):
        for sw in subs:
            if sw in self._wired_title_sources:
                continue
            if hasattr(sw, "viewTitleChanged"):
                try:
                    sw.viewTitleChanged.connect(lambda *_: self._refresh_titles_only())
                except Exception:
                    pass
            self._wired_title_sources.add(sw)

    def _refresh_titles_only(self):
        subs = self._all_subwindows()
        uid_to_title = {
            (getattr(getattr(sw, "document", None), "uid", None) or id(sw)):
            (sw._effective_title() or "Untitled")
            for sw in subs
        }
        self.view_combo.blockSignals(True)
        cur = self.view_combo.currentIndex()
        for i in range(self.view_combo.count()):
            uid = self.view_combo.itemData(i)
            if uid in uid_to_title:
                self.view_combo.setItemText(i, uid_to_title[uid])
        self.view_combo.blockSignals(False)
        if 0 <= cur < self.view_combo.count():
            self.view_combo.setCurrentIndex(cur)

    def _current_view_uid(self):
        idx = self.view_combo.currentIndex()
        return self.view_combo.itemData(idx) if idx >= 0 else None

    def current_view(self):
        uid = self._current_view_uid()
        if uid is None:
            return None
        for sw in self._all_subwindows():
            doc   = getattr(sw, "document", None)
            sw_uid = getattr(doc, "uid", None) or id(sw)
            if sw_uid == uid:
                return sw
        return None

    def _on_pick_view(self, _i):
        self._rebuild_list()
        self._schedule_composite()

    # ─────────────────────────────────────────────────────────
    # List building
    # ─────────────────────────────────────────────────────────

    def _rebuild_list(self):
        self.list.clear()
        vw = self.current_view()
        if not vw:
            return

        choices = self._mask_choices()
        docs    = [d for _, d in choices]

        for lyr in getattr(vw, "_layers", []):
            name = str(getattr(lyr, "name", "Layer"))
            try:
                src_doc = getattr(lyr, "src_doc", None)
                if src_doc is not None:
                    for sw in self._all_subwindows():
                        if getattr(sw, "document", None) is src_doc:
                            t = getattr(sw, "_effective_title", None)
                            if callable(t):
                                t = t()
                            if t:
                                name = t
                            break
            except Exception:
                pass

            roww = _LayerRow(
                name,
                getattr(lyr, "mode",    "Normal"),
                float(getattr(lyr, "opacity",  1.0)),
                bool(getattr(lyr,  "visible", True)),
            )
            # Mask combo
            roww.mask_combo.blockSignals(True)
            roww.mask_combo.clear()
            roww.mask_combo.addItem("(none)", userData=None)
            for title, doc in choices:
                roww.mask_combo.addItem(title, userData=doc)
            if getattr(lyr, "mask_doc", None) in docs:
                roww.mask_combo.setCurrentIndex(1 + docs.index(lyr.mask_doc))
            else:
                roww.mask_combo.setCurrentIndex(0)
            roww.mask_invert.setChecked(bool(getattr(lyr, "mask_invert", False)))
            roww.mask_combo.blockSignals(False)

            roww.set_levels_enabled(getattr(lyr, "levels_enabled", False))
            roww.set_levels_params(
                getattr(lyr, "black_point", 0.0),
                getattr(lyr, "white_point", 1.0),
                getattr(lyr, "midtones",    0.5),
            )
            roww.set_sigmoid_params(
                getattr(lyr, "sigmoid_center",   0.5),
                getattr(lyr, "sigmoid_strength", 10.0),
            )

            t = getattr(lyr, "transform", None)
            dirty = (t is not None and (
                abs(t.tx) > 1e-6 or abs(t.ty) > 1e-6 or abs(t.rot_deg) > 1e-6 or
                abs(t.sx - 1.0) > 1e-6 or abs(t.sy - 1.0) > 1e-6))
            roww.setTransformDirty(dirty)

            self._bind_row(roww)
            it = QListWidgetItem(self.list)
            it.setSizeHint(roww.sizeHint())
            self.list.addItem(it)
            self.list.setItemWidget(it, roww)

        # Base row (always last)
        base_name = "Current View"
        try:
            t = getattr(vw, "_effective_title", None)
            if callable(t): t = t()
            if t: base_name = t
        except Exception:
            pass

        base_row = _LayerRow(f"Base • {base_name}", "—", 1.0, True, is_base=True)
        base_levels = getattr(vw, "_base_levels", None)
        if base_levels:
            base_row.set_levels_enabled(base_levels.get("enabled", False))
            base_row.set_levels_params(
                base_levels.get("black_point", 0.0),
                base_levels.get("white_point", 1.0),
                base_levels.get("midtones",    0.5),
            )
        base_row.changed.connect(self._on_row_changed)
        itb = QListWidgetItem(self.list)
        itb.setSizeHint(base_row.sizeHint())
        self.list.addItem(itb)
        self.list.setItemWidget(itb, base_row)

        has = bool(getattr(vw, "_layers", []))
        self.btn_merge.setEnabled(has)
        self.btn_merge_new.setEnabled(has)
        self.btn_merge_sel.setEnabled(has)
        self.btn_clear.setEnabled(has)
        self._refresh_row_heights()

    def _bind_row(self, roww: _LayerRow):
        if getattr(roww, "_is_base", False):
            return
        roww.changed.connect(self._on_row_changed)
        roww.requestDelete.connect(lambda: self._delete_row(roww))
        roww.moveUp.connect(lambda: self._move_row(roww, -1))
        roww.moveDown.connect(lambda: self._move_row(roww, +1))
        roww.requestTransform.connect(lambda rw=roww: self._edit_transform_for_row(rw))

    def _refresh_row_heights(self):
        try:
            for i in range(self.list.count()):
                item = self.list.item(i)
                roww = self.list.itemWidget(item)
                if roww is not None:
                    roww.layout().activate()
                    item.setSizeHint(roww.minimumSizeHint())
        except Exception as ex:
            print("[LayersDock] _refresh_row_heights:", ex)

    # ─────────────────────────────────────────────────────────
    # Layer sync
    # ─────────────────────────────────────────────────────────

    def _sync_layers_from_rows(self, vw) -> None:
        layers = getattr(vw, "_layers", None) or []
        n = min(len(layers), self.list.count())
        for i in range(n):
            it   = self.list.item(i)
            roww = self.list.itemWidget(it) if it else None
            if roww is None or getattr(roww, "_is_base", False):
                continue
            try:
                p   = roww.params()
                lyr = layers[i]
                lyr.visible        = bool(p["visible"])
                lyr.mode           = p["mode"]
                lyr.opacity        = float(p["opacity"])
                lyr.levels_enabled = bool(p.get("levels_enabled", False))
                lyr.black_point    = float(p.get("black_point", 0.0))
                lyr.midtones       = float(p.get("midtones",    0.5))
                lyr.white_point    = float(p.get("white_point", 1.0))
                if "sigmoid_center" in p:
                    lyr.sigmoid_center   = float(p["sigmoid_center"])
                if "sigmoid_strength" in p:
                    lyr.sigmoid_strength = float(p["sigmoid_strength"])
                mi = p.get("mask_index")
                lyr.mask_doc    = roww.mask_combo.itemData(mi) if (mi and mi > 0) else None
                lyr.mask_use_luma = True
                lyr.mask_invert   = bool(p["mask_invert"])
            except Exception as ex:
                print("[LayersDock] sync row error:", ex)

        # Sync base levels
        n_layers = len(getattr(vw, "_layers", []) or [])
        base_item = self.list.item(n_layers)
        if base_item:
            base_roww = self.list.itemWidget(base_item)
            if isinstance(base_roww, _LayerRow) and getattr(base_roww, "_is_base", False):
                try:
                    p = base_roww.params()
                    vw._base_levels = {
                        "enabled":     bool(p.get("levels_enabled", False)),
                        "black_point": float(p.get("black_point", 0.0)),
                        "white_point": float(p.get("white_point", 1.0)),
                        "midtones":    float(p.get("midtones",    0.5)),
                    }
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────
    # Row manipulation
    # ─────────────────────────────────────────────────────────

    def _find_row_index(self, roww: _LayerRow) -> int:
        for i in range(self.list.count()):
            if self.list.itemWidget(self.list.item(i)) is roww:
                return i
        return -1

    def _layer_count(self) -> int:
        vw = self.current_view()
        return len(getattr(vw, "_layers", [])) if vw else 0

    def _delete_row(self, roww: _LayerRow):
        vw = self.current_view()
        if not vw: return
        idx = self._find_row_index(roww)
        if idx < 0 or idx >= self._layer_count(): return
        vw._layers.pop(idx)
        self.list.takeItem(idx)
        self._schedule_composite()

    def _move_row(self, roww: _LayerRow, delta: int):
        vw = self.current_view()
        if not vw: return
        i = self._find_row_index(roww)
        if i < 0 or i >= self._layer_count(): return
        j = i + delta
        if j < 0 or j >= self._layer_count(): return
        vw._layers[i], vw._layers[j] = vw._layers[j], vw._layers[i]
        self._rebuild_list()
        self._schedule_composite()

    def _selected_layer_indices(self) -> list[int]:
        vw = self.current_view()
        if not vw: return []
        n = len(getattr(vw, "_layers", []) or [])
        return sorted({self.list.row(it) for it in self.list.selectedItems()
                       if 0 <= self.list.row(it) < n})

    # ─────────────────────────────────────────────────────────
    # Transform editing
    # ─────────────────────────────────────────────────────────

    def _edit_transform_for_row(self, roww: _LayerRow):
        vw = self.current_view()
        if not vw: return
        idx = self._find_row_index(roww)
        if idx < 0 or idx >= self._layer_count(): return
        lyr = vw._layers[idx]
        t   = getattr(lyr, "transform", None) or LayerTransform()
        lyr.transform = t

        def _apply_t(new_t):
            lyr.transform = new_t
            dirty = (abs(new_t.tx)>1e-6 or abs(new_t.ty)>1e-6 or
                     abs(new_t.rot_deg)>1e-6 or abs(new_t.sx-1)>1e-6 or abs(new_t.sy-1)>1e-6)
            try: roww.setTransformDirty(dirty)
            except Exception: pass
            self._schedule_composite()

        def _cancel_t(orig_t):
            lyr.transform = orig_t
            dirty = (abs(orig_t.tx)>1e-6 or abs(orig_t.ty)>1e-6 or
                     abs(orig_t.rot_deg)>1e-6 or abs(orig_t.sx-1)>1e-6 or abs(orig_t.sy-1)>1e-6)
            try: roww.setTransformDirty(dirty)
            except Exception: pass
            self._schedule_composite()

        dlg = _TransformDialog(self, t=lyr.transform, apply_cb=_apply_t,
                               cancel_cb=_cancel_t, debounce_ms=120)
        dlg.exec()
        self._schedule_composite()

    # ─────────────────────────────────────────────────────────
    # Merge / push operations  (use cached composite)
    # ─────────────────────────────────────────────────────────

    def _get_composite_for_push(self) -> Optional[np.ndarray]:
        """Return cached composite, running a fresh sync+render if needed."""
        vw = self.current_view()
        if not vw:
            return None
        # Ensure the cache is fresh by syncing rows first
        self._sync_layers_from_rows(vw)
        self._run_composite()                   # synchronous — called directly
        return self._cached_composite

    def _merge_and_push(self):
        vw = self.current_view()
        if not vw: return
        layers = list(getattr(vw, "_layers", []) or [])
        if not layers:
            QMessageBox.information(self, "Layers", "There are no layers to merge.")
            return
        try:
            base_doc = getattr(vw, "document", None)
            if base_doc is None or getattr(base_doc, "image", None) is None:
                QMessageBox.warning(self, "Layers", "No base image for this view.")
                return
            merged = self._get_composite_for_push()
            if merged is None:
                QMessageBox.warning(self, "Layers", "Composite failed.")
                return
            meta = dict(getattr(base_doc, "metadata", {}) or {})
            meta["step_name"] = "Layers Merge"
            base_doc.apply_edit(merged.copy(), metadata=meta, step_name="Layers Merge")
            vw._layers      = []
            vw._base_levels = None
            if hasattr(vw, "_reinstall_layer_watchers"):
                vw._reinstall_layer_watchers()
            self._cached_composite = None
            self._rebuild_list()
            # Tell the view to re-render normally now (no composite active)
            try: vw._render(rebuild=True)
            except Exception: pass
            QMessageBox.information(self, "Layers",
                "Merged visible layers and pushed to the current view.")
        except Exception as ex:
            print("[LayersDock] merge error:", ex)
            QMessageBox.critical(self, "Layers", f"Merge failed:\n{ex}")

    def _merge_to_new_doc(self):
        vw = self.current_view()
        if not vw: return
        if not getattr(vw, "_layers", []):
            QMessageBox.information(self, "Layers", "There are no layers to merge.")
            return
        try:
            base_doc = getattr(vw, "document", None)
            if base_doc is None or getattr(base_doc, "image", None) is None:
                QMessageBox.warning(self, "Layers", "No base image for this view.")
                return
            merged = self._get_composite_for_push()
            if merged is None:
                QMessageBox.warning(self, "Layers", "Composite failed.")
                return
            self._push_merged_as_new_doc(base_doc, merged)
            QMessageBox.information(self, "Layers",
                "Merged visible layers and created a new document.")
        except Exception as ex:
            print("[LayersDock] merge_to_new_doc error:", ex)
            QMessageBox.critical(self, "Layers", f"Merge failed:\n{ex}")

    def _merge_selected_to_single_layer(self):
        vw = self.current_view()
        if not vw: return
        layers = list(getattr(vw, "_layers", []) or [])
        if not layers:
            QMessageBox.information(self, "Layers", "There are no layers to merge.")
            return
        sel = self._selected_layer_indices()
        if len(sel) < 2:
            QMessageBox.information(self, "Layers", "Select two or more layers to merge.")
            return
        try:
            base_doc = getattr(vw, "document", None)
            if base_doc is None or getattr(base_doc, "image", None) is None:
                QMessageBox.warning(self, "Layers", "No base image."); return
            base_img     = base_doc.image
            i0, i1       = sel[0], sel[-1]
            layers_above = layers[:i0]
            layers_sel   = layers[i0:i1+1]
            layers_below = layers[i1+1:]
            under        = composite_stack(base_img, layers_below)
            baked        = composite_stack(under, layers_sel)
            merged_layer = ImageLayer(
                name=f"Merged ({len(layers_sel)})",
                pixels=baked.astype(np.float32, copy=False),
                visible=True, opacity=1.0, mode="Normal",
            )
            merged_layer.mask_doc = None; merged_layer.mask_use_luma = True
            vw._layers = layers_above + [merged_layer] + layers_below
            if hasattr(vw, "_reinstall_layer_watchers"):
                vw._reinstall_layer_watchers()
            self._rebuild_list()
            self._schedule_composite()
            QMessageBox.information(self, "Layers",
                f"Merged {len(layers_sel)} layers into one.")
        except Exception as ex:
            print("[LayersDock] merge_selected error:", ex)
            QMessageBox.critical(self, "Layers", f"Merge Selected failed:\n{ex}")

    def _push_merged_as_new_doc(self, base_doc, arr: np.ndarray):
        dm = getattr(self.mw, "docman", None)
        if not dm or not hasattr(dm, "open_array"):
            return
        try:
            vw   = self.current_view()
            base = ""
            if vw and hasattr(vw, "_effective_title"):
                base = (vw._effective_title() or "").strip()
            if not base:
                dn = getattr(base_doc, "display_name", None)
                base = (dn() if callable(dn) else (dn or "Untitled"))
            title = base if base.endswith("_merged") else f"{base}_merged"
            meta  = dict(getattr(base_doc, "metadata", {}) or {})
            meta.update({"bit_depth": "32-bit floating point",
                         "is_mono": (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)),
                         "source": "Layers Merge", "step_name": "Layers Merge"})
            newdoc = dm.open_array(arr.astype(np.float32, copy=False), metadata=meta, title=title)
            if hasattr(self.mw, "_spawn_subwindow_for"):
                self.mw._spawn_subwindow_for(newdoc)
        except Exception as ex:
            print("[LayersDock] _push_merged_as_new_doc:", ex)

    def _clear_layers(self):
        vw = self.current_view()
        if not vw: return
        vw._layers = []
        if hasattr(vw, "_reinstall_layer_watchers"):
            vw._reinstall_layer_watchers()
        self._cached_composite = None
        self._rebuild_list()
        try: vw._render(rebuild=True)
        except Exception: pass

    # ─────────────────────────────────────────────────────────
    # Drag and drop
    # ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, e: QDragEnterEvent):
        md = e.mimeData()
        if md.hasFormat(MIME_VIEWSTATE) or md.hasFormat(MIME_MASK):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        self.dragEnterEvent(e)

    def dropEvent(self, e: QDropEvent):
        vw = self.current_view()
        if not vw: e.ignore(); return
        md = e.mimeData()
        try:
            if md.hasFormat(MIME_VIEWSTATE):
                st      = json.loads(bytes(md.data(MIME_VIEWSTATE)).decode("utf-8"))
                src_doc = self._resolve_doc_from_state(st)
                if src_doc is None:
                    raise RuntimeError("Source doc gone")
                layer_name = "Layer"
                for sw in self._all_subwindows():
                    if getattr(sw, "document", None) is src_doc:
                        t = getattr(sw, "_effective_title", None)
                        layer_name = (t() if callable(t) else t) or "Layer"
                        break
                new_layer = ImageLayer(
                    name=layer_name, src_doc=src_doc,
                    visible=True, opacity=1.0, mode="Normal",
                )
                if not hasattr(vw, "_layers") or vw._layers is None:
                    vw._layers = []
                vw._layers.insert(0, new_layer)
                if hasattr(vw, "_reinstall_layer_watchers"):
                    vw._reinstall_layer_watchers()
                self._rebuild_list()
                self._schedule_composite()
                e.acceptProposedAction(); return

            if md.hasFormat(MIME_MASK):
                payload  = json.loads(bytes(md.data(MIME_MASK)).decode("utf-8"))
                mask_doc = self._resolve_doc_from_state(payload)
                if mask_doc is None:
                    raise RuntimeError("Mask doc gone")
                if not getattr(vw, "_layers", None):
                    QMessageBox.information(self, "No Layers",
                        "Add a layer first, then drop a mask onto it.")
                    e.ignore(); return
                sel_row = self.list.currentRow()
                idx     = min(max(sel_row, 0), len(vw._layers) - 1)
                lyr     = vw._layers[idx]
                lyr.mask_doc    = mask_doc
                lyr.mask_invert = bool(payload.get("invert", False))
                try: lyr.mask_feather = float(payload.get("feather", 0.0) or 0.0)
                except Exception: lyr.mask_feather = 0.0
                if hasattr(vw, "_reinstall_layer_watchers"):
                    vw._reinstall_layer_watchers()
                self._rebuild_list()
                self._schedule_composite()
                e.acceptProposedAction(); return
        except Exception as ex:
            print("[LayersDock] drop error:", ex)
        e.ignore()

    # ─────────────────────────────────────────────────────────
    # Doc resolution helpers
    # ─────────────────────────────────────────────────────────

    def _resolve_doc_ptr(self, ptr: int):
        try:
            for d in self.docman.all_documents():
                if id(d) == ptr:
                    return d
        except Exception:
            pass
        return None

    def _resolve_doc_from_state(self, st):
        if isinstance(st, int):
            return self._resolve_doc_ptr(st)
        if not isinstance(st, dict):
            return None
        for uid_key in ("doc_uid", "base_doc_uid"):
            uid = st.get(uid_key)
            if uid and hasattr(self.docman, "get_document_by_uid"):
                d = self.docman.get_document_by_uid(uid)
                if d is not None:
                    return d
        ptr = st.get("doc_ptr") or st.get("mask_doc_ptr")
        if isinstance(ptr, int):
            d = self._resolve_doc_ptr(ptr)
            if d is not None:
                return d
        fp = (st.get("file_path") or "").strip()
        if fp:
            try:
                for d in self.docman.all_documents():
                    if (getattr(d, "metadata", {}) or {}).get("file_path") == fp:
                        return d
            except Exception:
                pass
        return None