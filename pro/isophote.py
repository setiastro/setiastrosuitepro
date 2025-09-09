#pro.isophote.py
from __future__ import annotations

# --- Stdlib ---
import time
import inspect
from types import SimpleNamespace
from typing import Optional

# --- Third-party ---
import numpy as np
from astropy.io import fits

# photutils is optional; we degrade gracefully if missing
try:
    from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
except Exception:  # pragma: no cover
    Ellipse = None
    EllipseGeometry = None
    build_ellipse_model = None

# --- Qt (PyQt6) ---
from PyQt6.QtCore import (
    pyqtSignal, QObject, Qt, QSize, QEvent, QThread, QPointF, QRectF
)
from PyQt6.QtGui import (
    QIcon, QPixmap, QImage, QPen, QBrush, QPainterPath, QCursor, QFontMetrics, QAction, QTransform
)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsEllipseItem, QGraphicsPathItem, QFormLayout, QHBoxLayout, QVBoxLayout,
    QLabel, QSlider, QPushButton, QCheckBox, QDoubleSpinBox, QSizePolicy, QSplitter,
    QToolButton, QMenu, QMessageBox, QStyle, QProgressDialog, QGraphicsItem, QFileDialog
)

from imageops.stretch import stretch_mono_image, stretch_color_image


# ===========================
#  UI Helpers / Components
# ===========================

class _SyncedView(QGraphicsView):
    """Zoom/pan-enabled view that mirrors BOTH transform and scrollbars to a peer.
       Shift+LeftClick emits image coords for 'pick center'."""
    viewChanged = pyqtSignal(QTransform, int, int)
    mousePosClicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._sync_block = False
        self._img_item: Optional[QGraphicsPixmapItem] = None
        self._img_shape = None
        self.horizontalScrollBar().valueChanged.connect(self._emit_view_changed)
        self.verticalScrollBar().valueChanged.connect(self._emit_view_changed)

    def setSceneImage(self, qpix: QPixmap, img_shape):
        scene = QGraphicsScene(self)
        self._img_item = QGraphicsPixmapItem(qpix)
        scene.addItem(self._img_item)
        self.setScene(scene)
        self._img_shape = img_shape
        self.fitInView(self._img_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._emit_view_changed()

    def wheelEvent(self, e):
        factor = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._emit_view_changed()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._emit_view_changed()

    def _emit_view_changed(self, *_):
        if self._sync_block:
            return
        self.viewChanged.emit(
            self.transform(),
            self.horizontalScrollBar().value(),
            self.verticalScrollBar().value()
        )

    def setPeerView(self, tr: QTransform, hval: int, vval: int):
        if self._sync_block:
            return
        self._sync_block = True
        try:
            self.setTransform(tr)
            self.horizontalScrollBar().setValue(hval)
            self.verticalScrollBar().setValue(vval)
        finally:
            self._sync_block = False

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.MouseButton.LeftButton and
            (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier) and
            self._img_item is not None):
            p = self.mapToScene(ev.position().toPoint())
            self.mousePosClicked.emit(p.x(), p.y())
            ev.accept()
            return
        super().mousePressEvent(ev)


class FloatSlider(QWidget):
    """Labeled horizontal slider that emits/accepts float values, with fixed-width value label."""
    valueChanged = pyqtSignal(float)

    def __init__(self, minimum: float, maximum: float, value: float,
                 decimals: int = 2, unit: str = "", tick: Optional[float] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._scale = 10 ** decimals
        self._decimals = decimals
        self._unit = unit

        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._label = QLabel(self)

        self._slider.setRange(int(round(minimum * self._scale)),
                              int(round(maximum * self._scale)))
        if tick:
            self._slider.setSingleStep(int(max(1, round(tick * self._scale))))

        fm = QFontMetrics(self._label.font())
        max_abs = max(abs(minimum), abs(maximum))
        sample_text = f"-{max_abs:.{self._decimals}f}{self._unit}"
        self._label.setMinimumWidth(fm.horizontalAdvance(sample_text) + 8)
        self._label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._slider, 1); lay.addWidget(self._label, 0)

        self._slider.valueChanged.connect(self._on_slider)
        self.setValue(value)

    def _on_slider(self, iv):
        val = iv / self._scale
        self._label.setText(f"{val:.{self._decimals}f}{self._unit}")
        self.valueChanged.emit(val)

    def value(self) -> float:
        return self._slider.value() / self._scale

    def setValue(self, v: float):
        self._slider.blockSignals(True)
        self._slider.setValue(int(round(v * self._scale)))
        self._slider.blockSignals(False)
        self._label.setText(f"{self.value():.{self._decimals}f}{self._unit}")


class DraggableEllipse(QGraphicsEllipseItem):
    """Seed ellipse: movable only while holding Ctrl."""
    def __init__(self, rect: QRectF, on_center_moved=None):
        super().__init__(rect)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        pen = QPen(Qt.GlobalColor.cyan); pen.setWidthF(1.5)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._drag_active = False
        self._drag_offset = QPointF(0, 0)
        self._on_center_moved = on_center_moved

    def center_scene(self) -> QPointF:
        return self.mapToScene(self.rect().center())

    def set_center_scene(self, p: QPointF):
        d = p - self.center_scene()
        self.moveBy(d.x(), d.y())

    def hoverMoveEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().hoverMoveEvent(ev)

    def mousePressEvent(self, ev):
        if (ev.button() == Qt.MouseButton.LeftButton and
            (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)):
            self._drag_active = True
            self._drag_offset = self.mapToScene(ev.pos()) - self.center_scene()
            ev.accept()
        else:
            self._drag_active = False
            ev.ignore()

    def mouseMoveEvent(self, ev):
        if self._drag_active:
            new_center = self.mapToScene(ev.pos()) - self._drag_offset
            self.set_center_scene(new_center)
            ev.accept()
        else:
            ev.ignore()

    def mouseReleaseEvent(self, ev):
        if self._drag_active:
            self._drag_active = False
            if self._on_center_moved:
                c = self.center_scene()
                self._on_center_moved(c.x(), c.y())
            ev.accept()
        else:
            ev.ignore()


# ===========================
#  Fitting Worker (threaded)
# ===========================

class _FitWorker(QObject):
    finished = pyqtSignal(object, object, object)  # model, resid, isolist_or_scaled
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, img, params, parent=None):
        super().__init__(parent)
        self.img = img
        self.p   = params

    @staticmethod
    def _downsample_mean(img, ds):
        ds = int(max(1, ds))
        H, W = img.shape
        if ds == 1:
            return img, (H, W)
        Hc, Wc = (H // ds) * ds, (W // ds) * ds
        if Hc == 0 or Wc == 0:
            return img, (H, W)
        crop = img[:Hc, :Wc]
        small = crop.reshape(Hc // ds, ds, Wc // ds, ds).mean(axis=(1, 3))
        return small.astype(img.dtype, copy=False), (H, W)

    @staticmethod
    def _upsample_nn(arr, ds, out_shape, pad_mode="edge"):
        ds = int(max(1, ds))
        if ds == 1:
            up = arr
        else:
            up = arr.repeat(ds, axis=0).repeat(ds, axis=1)
        H, W = out_shape
        uh, uw = up.shape
        if uh < H or uw < W:
            up = np.pad(up, ((0, max(0, H - uh)), (0, max(0, W - uw))), mode=pad_mode)
        return up[:H, :W].astype(arr.dtype, copy=False)

    def run(self):
        try:
            if Ellipse is None or EllipseGeometry is None or build_ellipse_model is None:
                raise RuntimeError("photutils.isophote not available")

            self.progress.emit(5,  "Preparing…")
            ds = int(max(1, self.p.get("downsample", 1)))

            # --- geometry at full-res ---
            pa_rad = np.deg2rad(self.p["pa_deg"])
            try:
                geom_full = EllipseGeometry(x0=self.p["cx"], y0=self.p["cy"],
                                            sma=self.p["sma0"], eps=self.p["eps"], pa=pa_rad)
            except TypeError:
                geom_full = EllipseGeometry(x0=self.p["cx"], y0=self.p["cy"],
                                            sma=self.p["sma0"], eps=self.p["eps"],
                                            position_angle=pa_rad)

            # --- build (possibly) downsampled image & geometry ---
            img_for_fit = self.img
            geom_for_fit = geom_full
            minsma = self.p["minsma"]; maxsma = self.p["maxsma"]; step = self.p["step"]
            if ds > 1:
                small, full_shape = self._downsample_mean(self.img, ds)
                img_for_fit = small
                geom_for_fit = EllipseGeometry(
                    x0=geom_full.x0/ds, y0=geom_full.y0/ds,
                    sma=geom_full.sma/ds, eps=geom_full.eps, pa=geom_full.pa
                )
                minsma = minsma/ds; maxsma = maxsma/ds; step = max(step/ds, 0.5)

            self.progress.emit(20, "Building mask…")
            h, w = img_for_fit.shape
            if self.p["use_wedge"]:
                yy, xx = np.mgrid[0:h, 0:w]
                cx_fit, cy_fit = geom_for_fit.x0, geom_for_fit.y0
                ang  = np.arctan2(yy - cy_fit, xx - cx_fit)
                pa   = np.deg2rad(self.p["wedge_pa"])
                half = np.deg2rad(self.p["wedge_width"] / 2.0)
                d = np.arctan2(np.sin(ang - pa), np.cos(ang - pa))
                wedge_mask = (np.abs(d) <= half)
            else:
                wedge_mask = np.zeros((h, w), dtype=bool)

            img_ma = np.ma.masked_array(img_for_fit, mask=wedge_mask)
            ell = Ellipse(img_ma, geometry=geom_for_fit)

            # --- fit kwargs (version-safe) ---
            fit_kwargs = dict(
                sma0=geom_for_fit.sma, minsma=minsma, maxsma=maxsma,
                step=step, sclip=self.p["sclip"], nclip=int(self.p["nclip"]),
                fix_center=self.p["fix_center"], fix_pa=self.p["fix_pa"], fix_eps=self.p["fix_eps"],
            )
            sig = inspect.signature(ell.fit_image).parameters
            mode_key = "integrmode" if "integrmode" in sig else ("integr_mode" if "integr_mode" in sig else None)
            if mode_key:
                fit_kwargs[mode_key] = "bilinear"

            self.progress.emit(40, "Fitting isophotes…")
            isolist = ell.fit_image(**fit_kwargs)
            if hasattr(isolist, "__len__") and len(isolist) == 0:
                raise ValueError("isolist must not be empty")

            self.progress.emit(60, "Building model…")
            model_fit = build_ellipse_model(img_for_fit.shape, isolist,
                                            high_harmonics=self.p["high_harm"])
            resid_fit = img_for_fit - model_fit

            self.progress.emit(95, "Upsampling / finalizing…")

            if ds > 1:
                model_full = self._upsample_nn(model_fit, ds, (self.img.shape[0], self.img.shape[1]))
                resid_full = self.img - model_full
                # scale isolist params to full-res
                scaled = []
                for iso in isolist:
                    x0  = float(getattr(iso, "x0",  getattr(iso, "x0_center", geom_for_fit.x0))) * ds
                    y0  = float(getattr(iso, "y0",  getattr(iso, "y0_center", geom_for_fit.y0))) * ds
                    sma = float(getattr(iso, "sma", getattr(iso, "sma0",  geom_for_fit.sma))) * ds
                    eps = float(getattr(iso, "eps", getattr(iso, "ellipticity", geom_for_fit.eps)))
                    pa  = float(getattr(iso, "pa",  getattr(iso, "position_angle", geom_for_fit.pa)))
                    scaled.append(SimpleNamespace(x0=x0, y0=y0, sma=sma, eps=eps, pa=pa))
                self.finished.emit(model_full.astype(np.float32),
                                   resid_full.astype(np.float32),
                                   scaled)
            else:
                self.finished.emit(model_fit.astype(np.float32),
                                   resid_fit.astype(np.float32),
                                   isolist)

        except Exception as e:
            self.error.emit(str(e))


# ===========================
#  Main Dialog
# ===========================

class IsophoteModelerDialog(QDialog):
    pushRequested = pyqtSignal(str, int, object)  # kept for legacy, not used with doc_manager

    def __init__(self, mono_image: np.ndarray, parent: Optional[QWidget] = None,
                 title_hint: Optional[str] = None, image_manager=None, doc_manager=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self.doc_manager   = doc_manager

        self._ellipse_item = None
        self._max_item = None
        self._min_item = None
        self._isolist = None
        self._last_fit_params = None
        self._preview_right01 = None

        self._perf = {1: 0.060714, 4: 0.004286}
        self._last_run_timer = None

        if Ellipse is None:
            QMessageBox.critical(self, "Photutils Missing",
                                 "photutils.isophote is required for GLIMR.")
            self.close(); return

        self.setWindowTitle(title_hint or "GLIMR — GaLaxy Isophote Modeler & Residual Revealer")
        self.setMinimumSize(1100, 700)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowMaximizeButtonHint
                            | Qt.WindowType.WindowMinimizeButtonHint)
        self.setSizeGripEnabled(True)

        self._img = mono_image.astype(np.float32, copy=False)
        self._model = None
        self._resid = None

        # ---- Views ----
        self.left = _SyncedView()
        self.right = _SyncedView()
        self.left.viewChanged.connect(self.right.setPeerView)
        self.right.viewChanged.connect(self.left.setPeerView)
        self.left.mousePosClicked.connect(self._on_left_click)

        self._in01 = self._compute_input01()
        lpix = self._np_to_qpix_linear01(self._in01)
        self.left.setSceneImage(lpix, self._in01.shape)
        self.right.setSceneImage(lpix, self._in01.shape)
        self.right.setPeerView(
            self.left.transform(),
            self.left.horizontalScrollBar().value(),
            self.left.verticalScrollBar().value()
        )

        # overlays
        self._wedge_item = None

        # ---- Controls ----
        ctl = QWidget(); form = QFormLayout(ctl)

        self.fix_center = QCheckBox("Fix Center")
        self.fix_pa = QCheckBox("Fix PA")
        self.fix_eps = QCheckBox("Fix Ellipticity")
        self.high_harm = QCheckBox("Add a3/b3/a4/b4 in model")

        h, w = self._img.shape
        max_rad = min(h, w) / 1.2

        self.sma0      = FloatSlider(1.0, max_rad, 20.0, decimals=1, unit=" px")
        self.minsma    = FloatSlider(0.0, max_rad, 0.0,  decimals=1, unit=" px")
        self.maxsma    = FloatSlider(1.0, max_rad, max_rad, decimals=1, unit=" px")
        self.step      = FloatSlider(0.01, 3.00, 1.00, decimals=2)
        self.sclip     = FloatSlider(1.0, 10.0, 3.0, decimals=2)
        self.nclip     = FloatSlider(0.0, 20.0, 1.0, decimals=0)

        self.eps       = FloatSlider(0.0, 0.95, 0.20, decimals=3)
        self.pa_deg    = FloatSlider(-180.0, 180.0, 90.0, decimals=1, unit="°")

        self.use_wedge   = QCheckBox("Exclude wedge (deg)")
        self.wedge_pa    = FloatSlider(-180.0, 180.0, 0.0, decimals=1, unit="°")
        self.wedge_width = FloatSlider(0.0, 180.0, 30.0, decimals=1, unit="°")

        self._cx, self._cy = w/2.0, h/2.0
        self.center_label = QLabel(f"Center: ({self._cx:.1f}, {self._cy:.1f})")
        pick_center_btn = QPushButton("Pick Center (Shift+click)  •  Move (Ctrl+drag ellipse)")

        self.hq_interp = QCheckBox("High-quality interpolation (slower)")
        self.hq_interp.setChecked(False)
        self.quick_preview = QCheckBox("Quick preview (4× downsample)")
        self.quick_preview.setChecked(True)

        run_btn = QPushButton("Fit Model"); run_btn.clicked.connect(self._run_fit)
        self.preview_blend = QCheckBox("Show original outside max ellipse")
        self.preview_blend.setChecked(True)

        self.normalize_input = QCheckBox("Normalize before fitting (Linear Data)")
        self.normalize_input.setToolTip(
            "Apply global statistical stretch to the input before fitting/preview.\n"
            "Uses mono median target of 0.25."
        )
        self.normalize_input.setChecked(False)

        form.addRow(self.normalize_input)
        form.addRow(QLabel("<b>Geometry & Start</b>"))
        form.addRow("sma0", self.sma0)
        form.addRow("min sma", self.minsma)
        form.addRow("max sma", self.maxsma)
        form.addRow("step", self.step)
        self.ring_est_label = QLabel("≈ 0 rings"); form.addRow(self.ring_est_label)
        form.addRow("σ-clip (sclip)", self.sclip)
        form.addRow("σ-clip iters", self.nclip)
        form.addRow(self._help_row(self.fix_center, "Fix (x0,y0) across radii."))
        form.addRow(self._help_row(self.fix_pa, "Fix PA across radii."))
        form.addRow(self._help_row(self.fix_eps, "Fix ellipticity ε across radii."))
        form.addRow(self._help_row(self.high_harm, "Include a3/b3/a4/b4 in model."))
        form.addRow(pick_center_btn)

        form.addRow(QLabel("<b>Center / Shape</b>"))
        form.addRow(self.center_label)
        form.addRow("ellipticity ε", self.eps)
        form.addRow("PA (deg)", self.pa_deg)

        form.addRow(QLabel("<b>Wedge Mask</b>"))
        wr = QWidget(); wr_l = QHBoxLayout(wr); wr_l.setContentsMargins(0,0,0,0)
        wr_l.addWidget(self.use_wedge); wr_l.addWidget(QLabel("PA0")); wr_l.addWidget(self.wedge_pa)
        wr_l.addWidget(QLabel("±width/2")); wr_l.addWidget(self.wedge_width)
        form.addRow(wr)
        form.addRow(self.hq_interp)
        form.addRow(self.preview_blend)
        form.addRow(self.quick_preview)
        form.addRow(run_btn)

        self.save_resid_shifted = QCheckBox("Shift residuals to ≥ 0 on save")
        self.save_resid_shifted.setChecked(True)
        form.addRow(self.save_resid_shifted)

        # Export rows: push to NEW documents via doc_manager
        model_row, self._model_lowres_hint = self._make_export_row("model")
        resid_row, self._resid_lowres_hint = self._make_export_row("resid")
        form.addRow(model_row); form.addRow(resid_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.left); split.addWidget(self.right)
        split.setSizes([700, 700])

        root = QHBoxLayout(self); root.addWidget(split, 4); root.addWidget(ctl, 0)

        # connections
        pick_center_btn.clicked.connect(lambda: QMessageBox.information(
            self, "Center Picking",
            "Shift+LeftClick in the left image to set the center.\n"
            "Hold Ctrl and drag the cyan ellipse to adjust."
        ))
        for s in (self.sma0, self.maxsma, self.eps, self.pa_deg, self.minsma):
            s.valueChanged.connect(lambda _=None: self._create_or_update_overlay())
        self.preview_blend.stateChanged.connect(lambda _=None: self._rebuild_right_preview())
        for s in (self.minsma, self.sma0, self.maxsma):
            s.valueChanged.connect(lambda _=None: self._enforce_sma_order())
        for s in (self.wedge_pa, self.wedge_width):
            s.valueChanged.connect(lambda _=None: self._update_wedge_overlay())
        self.use_wedge.stateChanged.connect(self._update_wedge_overlay)

        def _update_ring_estimate():
            mn = float(self.minsma.value()); mx = float(self.maxsma.value())
            st = max(1e-6, float(self.step.value()))
            n  = int(max(0, (mx - mn) / st))
            ds = 4 if self.quick_preview.isChecked() else 1
            spr = self._perf.get(ds)
            if spr is None:
                other = 4 if ds == 1 else 1
                other_spr = self._perf.get(other)
                if other_spr is not None:
                    spr = other_spr * ((other / ds) ** 2)
            if spr is not None and n > 0:
                eta_sec = spr * n
                s = max(0.0, float(eta_sec))
                if s < 1.0: eta = f"{s:.1f}s"
                else:
                    m, s = divmod(int(round(s)), 60)
                    if m < 1: eta = f"{s:d}s"
                    else:
                        h, m = divmod(m, 60)
                        eta = f"{m:d}m {s:d}s" if h < 1 else f"{h:d}h {m:d}m"
                tag = "quick" if ds > 1 else "full"
                self.ring_est_label.setText(f"≈ {n:,} rings • est: {eta} ({tag})")
            else:
                self.ring_est_label.setText(f"≈ {n:,} rings")
            if n > 10000:
                new_step = max(st, (mx - mn) / 10000.0)
                if abs(new_step - st) > 1e-12:
                    self.step.setValue(new_step)
        for s in (self.minsma, self.maxsma, self.step):
            s.valueChanged.connect(lambda _=None: self._update_ring_estimate())
        self._update_ring_estimate()

        self.quick_preview.stateChanged.connect(lambda _=None: self._update_ring_estimate())
        self._update_lowres_hints()
        self.quick_preview.stateChanged.connect(lambda _=None: self._update_lowres_hints())
        self.normalize_input.stateChanged.connect(lambda _=None: self._recompute_input_view())
        self._update_wedge_overlay()

    # ---------- event/utility ----------
    def _compute_input01(self) -> np.ndarray:
        x = self._img.astype(np.float32, copy=False)
        try:
            if self.normalize_input.isChecked():
                x = stretch_mono_image(
                    x, target_median=0.25, normalize=False, apply_curves=False, curves_boost=0.0
                ).astype(np.float32, copy=False)
            else:
                x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
        return x

    def _recompute_input_view(self):
        self._in01 = self._compute_input01()
        pix = self._np_to_qpix_linear01(self._in01)
        self.left.setSceneImage(pix, self._in01.shape)
        if self._resid is None:
            self.right.setSceneImage(pix, self._in01.shape)
        else:
            self._rebuild_right_preview()

    def _update_lowres_hints(self):
        on = self.quick_preview.isChecked()
        ds = 4 if on else 1
        text = f"low-res (fit at {ds}× downsample)" if on else ""
        for lbl in (self._model_lowres_hint, self._resid_lowres_hint):
            lbl.setText(text); lbl.setVisible(on)

    def _make_export_row(self, which: str):
        row = QWidget(self); lay = QHBoxLayout(row); lay.setContentsMargins(0,0,0,0)
        save_btn = QPushButton(f"Save {which.capitalize()} FITS…", row)
        save_btn.clicked.connect(lambda: self._save_fits(which=which))
        lay.addWidget(save_btn, 0)

        new_btn = QPushButton(f"New Doc: {which.capitalize()} (normalized)", row)
        new_btn.clicked.connect(lambda: self._push_product(which=which, variant="normalized"))
        lay.addWidget(new_btn, 0)

        if which == "resid":
            vis_btn = QPushButton("New Doc: Residual (visible)", row)
            vis_btn.setToolTip("Push exactly what you see in the right preview pane")
            vis_btn.clicked.connect(lambda: self._push_product(which="resid", variant="visible"))
            lay.addWidget(vis_btn, 0)

            stretch_btn = QPushButton("New Doc: Residual (stretched)", row)
            stretch_btn.setToolTip("Symmetric preview stretch (0 → gray)")
            stretch_btn.clicked.connect(lambda: self._push_product(which="resid", variant="stretched"))
            lay.addWidget(stretch_btn, 0)

        hint = QLabel("", row)
        hint.setStyleSheet("color:#b58900; font-style: italic;"); hint.setVisible(False)
        lay.addWidget(hint, 0, Qt.AlignmentFlag.AlignVCenter); lay.addStretch(1)
        return row, hint

    def _update_ring_estimate(self):
        """Update the '≈ N rings' label (and ETA if we have a profile)."""
        mn = float(self.minsma.value())
        mx = float(self.maxsma.value())
        st = max(1e-6, float(self.step.value()))
        rings = int(max(0, (mx - mn) / st))

        # Soft cap to ~10k rings by raising step if needed
        if rings > 10000:
            new_step = (mx - mn) / 10000.0
            if new_step > st:
                self.step.setValue(new_step)
                st = new_step
                rings = int(max(0, (mx - mn) / st))

        ds = 4 if self.quick_preview.isChecked() else 1
        txt = f"≈ {rings:,} rings"

        # Seconds-per-ring profile (EMA-learned); scale from the other ds if missing
        spr = self._perf.get(ds)
        if spr is None:
            other = 4 if ds == 1 else 1
            if self._perf.get(other) is not None:
                spr = self._perf[other] * ((other / ds) ** 2)

        if spr is not None and rings > 0:
            eta = self._humanize_secs(spr * rings)
            txt += f"  •  est: {eta}" + (" (quick preview)" if ds > 1 else "")

        self.ring_est_label.setText(txt)

    def _humanize_secs(self, secs: float) -> str:
        secs = max(0.0, float(secs))
        if secs < 1.0:
            return f"{secs:.2f}s"
        m, s = divmod(int(round(secs)), 60)
        if m == 0:
            return f"{s}s"
        h, m = divmod(m, 60)
        if h == 0:
            return f"{m}m {s}s"
        return f"{h}h {m}m"

    def _apply_geometry_to_overlay(self):
        if self._ellipse_item is None:
            return
        self._create_or_update_overlay()

    def _resid_to_disp01(self, resid, mask, pct=99.5):
        r = np.nan_to_num(resid, 0.0, 0.0, 0.0).astype(np.float32)
        abs_in = np.abs(r[mask])
        S = float(np.percentile(abs_in, pct)) if abs_in.size else 1.0
        if not np.isfinite(S) or S <= 0:
            S = float(np.max(np.abs(r)) or 1.0)
        disp = np.empty_like(self._img, dtype=np.float32)
        disp[mask]  = 0.5 + (r[mask] / (2.0 * S))
        disp[~mask] = np.clip(self._img[~mask], 0.0, 1.0)
        return np.clip(disp, 0.0, 1.0)

    def _auto_estimate_from_moments(self):
        img = np.nan_to_num(self._img, nan=0.0).astype(np.float64)
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        q = np.quantile(img, 0.80)
        mask = img >= q
        if not np.any(mask):
            QMessageBox.information(self, "Auto-estimate", "Could not find bright core pixels.")
            return
        I = img[mask]; x = xx[mask]; y = yy[mask]
        I_sum = I.sum()
        cx = float((I * x).sum() / I_sum)
        cy = float((I * y).sum() / I_sum)
        x0 = x - cx; y0 = y - cy
        cov_xx = float((I * x0 * x0).sum() / I_sum)
        cov_yy = float((I * y0 * y0).sum() / I_sum)
        cov_xy = float((I * x0 * y0).sum() / I_sum)
        cov = np.array([[cov_xx, cov_xy],[cov_xy, cov_yy]])
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]; evecs = evecs[:, order]
        sigma_a = np.sqrt(max(evals[0], 1e-6))
        sigma_b = np.sqrt(max(evals[1], 1e-6))
        axis_ratio = float(np.clip(sigma_b / sigma_a, 1e-3, 0.999))
        eps = 1.0 - axis_ratio
        vx, vy = evecs[0,0], evecs[1,0]
        pa_deg = float(np.rad2deg(np.arctan2(vy, vx)))
        sma = float(2.5 * sigma_a)
        self._set_center(cx, cy)
        self.eps.setValue(min(0.95, max(0.0, eps)))
        self.pa_deg.setValue(pa_deg)
        self.sma0.setValue(max(5.0, min(sma, min(h, w)/1.2)))


    def _normalize01_for_push(self, arr: np.ndarray):
        a = np.asarray(arr, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        vmin = float(a.min()); vmax = float(a.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin + 1e-12:
            return np.zeros_like(a, dtype=np.float32), vmin, vmax
        out = (a - vmin) / (vmax - vmin)
        return out.astype(np.float32, copy=False), vmin, vmax

    def _residual_preview_stretch01(self, resid: np.ndarray, pct: float = 99.5):
        r = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        cx, cy, sma, eps, pa_deg = self._fit_boundary()
        _, m = self._elliptical_alpha(r.shape, cx, cy, sma, eps, pa_deg, feather_frac=0.04)
        inside = (m <= 1.0)
        abs_in = np.abs(r[inside]) if inside.any() else np.abs(r)
        S = float(np.percentile(abs_in, pct)) if abs_in.size else 1.0
        if not np.isfinite(S) or S <= 0:
            S = float(np.max(np.abs(r)) or 1.0)
        out01 = np.clip(0.5 + (r / (2.0 * S)), 0.0, 1.0).astype(np.float32, copy=False)
        return out01, S

    def _pix_from_01(self, img01):
        vals = np.nan_to_num(img01, nan=0.0, posinf=1.0, neginf=0.0)
        u8 = (np.clip(vals, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = u8.shape
        qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    def _np_to_qpix_linear01(self, img: np.ndarray) -> QPixmap:
        img = np.nan_to_num(img, 0.0, 0.0, 0.0)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        h, w = u8.shape
        qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    def _enforce_sma_order(self):
        changed = False
        if self.minsma.value() > self.sma0.value():
            self.minsma.setValue(self.sma0.value()); changed = True
        if self.sma0.value() > self.maxsma.value():
            self.sma0.setValue(self.maxsma.value()); changed = True
        if changed:
            self._create_or_update_overlay()

    def _ellipse_mask(self, shape, cx, cy, sma, eps, pa_deg):
        h, w = shape
        a = float(max(1.0, sma))
        b = float(max(1.0, a * (1.0 - eps)))
        pa = np.deg2rad(pa_deg)
        yy, xx = np.mgrid[0:h, 0:w]
        x0 = xx - cx; y0 = yy - cy
        c, s = np.cos(pa), np.sin(pa)
        xr =  x0 * c + y0 * s
        yr = -x0 * s + y0 * c
        return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0

    def _create_or_update_overlay(self):
        if self.left.scene() is None:
            return
        a0 = max(1.0, float(self.sma0.value()))
        b0 = max(1.0, a0 * (1.0 - float(self.eps.value())))
        rect0 = QRectF(self._cx - a0, self._cy - b0, 2*a0, 2*b0)

        if getattr(self, "_ellipse_item", None) is None:
            self._ellipse_item = DraggableEllipse(rect0, on_center_moved=self._set_center)
            self._ellipse_item.setTransformOriginPoint(self._ellipse_item.rect().center())
            self._ellipse_item.setRotation(self.pa_deg.value())
            self.left.scene().addItem(self._ellipse_item)
        else:
            c = self._ellipse_item.center_scene()
            self._ellipse_item.setRect(rect0)
            self._ellipse_item.setTransformOriginPoint(self._ellipse_item.rect().center())
            self._ellipse_item.setRotation(self.pa_deg.value())
            self._ellipse_item.set_center_scene(c)

        aI = max(1.0, float(self.minsma.value()))
        bI = max(1.0, aI * (1.0 - float(self.eps.value())))
        rectI = QRectF(self._cx - aI, self._cy - bI, 2*aI, 2*bI)
        if self._min_item is None:
            self._min_item = QGraphicsEllipseItem(rectI)
            penI = QPen(Qt.GlobalColor.magenta); penI.setWidthF(1.0); penI.setStyle(Qt.PenStyle.DotLine)
            self._min_item.setPen(penI); self._min_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self._min_item.setZValue(8); self.left.scene().addItem(self._min_item)
        else:
            self._min_item.setRect(rectI)
        self._min_item.setTransformOriginPoint(self._min_item.rect().center())
        self._min_item.setRotation(self.pa_deg.value())

        aM = max(1.0, float(self.maxsma.value()))
        bM = max(1.0, aM * (1.0 - float(self.eps.value())))
        rectM = QRectF(self._cx - aM, self._cy - bM, 2*aM, 2*bM)
        if self._max_item is None:
            self._max_item = QGraphicsEllipseItem(rectM)
            penM = QPen(Qt.GlobalColor.yellow); penM.setWidthF(1.0); penM.setStyle(Qt.PenStyle.DashLine)
            self._max_item.setPen(penM); self._max_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self._max_item.setZValue(9); self.left.scene().addItem(self._max_item)
        else:
            self._max_item.setRect(rectM)
        self._max_item.setTransformOriginPoint(self._max_item.rect().center())
        self._max_item.setRotation(self.pa_deg.value())

        self._update_wedge_overlay()

    def eventFilter(self, obj, ev):
        if obj is self.left.scene() and getattr(self, "_ellipse_item", None) is not None:
            if ev.type() == QEvent.Type.GraphicsSceneMouseRelease:
                c = self._ellipse_item.center_scene()
                self._set_center(c.x(), c.y())
        return super().eventFilter(obj, ev)

    def _update_wedge_overlay(self):
        if getattr(self, "_wedge_item", None) and self.left.scene():
            self.left.scene().removeItem(self._wedge_item); self._wedge_item = None
        if not self.use_wedge.isChecked() or self.left.scene() is None:
            return
        cx, cy = self._cx, self._cy
        pa = np.deg2rad(self.wedge_pa.value())
        half = np.deg2rad(self.wedge_width.value()/2.0)
        h, w = self._img.shape
        R = float(np.hypot(w, h))
        path = QPainterPath(QPointF(cx, cy))
        path.arcTo(cx-R, cy-R, 2*R, 2*R, -np.rad2deg(pa-half), self.wedge_width.value())
        path.lineTo(QPointF(cx, cy))
        item = QGraphicsPathItem(path)
        item.setOpacity(0.2)
        item.setBrush(QBrush(Qt.BrushStyle.Dense4Pattern))
        item.setPen(QPen(Qt.PenStyle.NoPen))
        self.left.scene().addItem(item)
        self._wedge_item = item

    def _make_wedge_mask(self, h, w):
        if not self.use_wedge.isChecked():
            return np.zeros((h, w), dtype=bool)
        cx, cy = self._cx, self._cy
        pa = np.deg2rad(self.wedge_pa.value())
        half = np.deg2rad(self.wedge_width.value()/2.0)
        yy, xx = np.mgrid[0:h, 0:w]
        ang = np.arctan2(yy - cy, xx - cx)
        d = np.arctan2(np.sin(ang - pa), np.cos(ang - pa))
        return np.abs(d) <= half

    def _help_row(self, ctrl: QWidget, help_text: str) -> QWidget:
        row = QWidget(self); lay = QHBoxLayout(row); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(ctrl, 1)
        btn = QToolButton(row); btn.setAutoRaise(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxQuestion))
        btn.setFixedSize(QSize(18, 18)); btn.setIconSize(QSize(14, 14))
        btn.setToolTip(help_text); ctrl.setToolTip(help_text); ctrl.setWhatsThis(help_text)
        title = ctrl.text() if hasattr(ctrl, "text") else "Help"
        btn.clicked.connect(lambda: QMessageBox.information(self, title, help_text))
        lay.addWidget(btn, 0)
        return row

    def _set_center(self, x: float, y: float):
        h, w = self._img.shape
        self._cx = float(np.clip(x, 0, w-1)); self._cy = float(np.clip(y, 0, h-1))
        self.center_label.setText(f"Center: ({self._cx:.1f}, {self._cy:.1f})")
        self._update_wedge_overlay(); self._create_or_update_overlay()

    def _on_left_click(self, x, y):
        self._set_center(x, y)

    # ---------- fit / preview ----------
    def _run_fit(self):
        p = dict(
            cx=self._cx, cy=self._cy,
            sma0=float(self.sma0.value()),
            minsma=float(self.minsma.value()),
            maxsma=float(self.maxsma.value()),
            step=float(self.step.value()),
            sclip=float(self.sclip.value()),
            nclip=float(self.nclip.value()),
            eps=float(self.eps.value()),
            pa_deg=float(self.pa_deg.value()),
            fix_center=self.fix_center.isChecked(),
            fix_pa=self.fix_pa.isChecked(),
            fix_eps=self.fix_eps.isChecked(),
            high_harm=self.high_harm.isChecked(),
            use_wedge=self.use_wedge.isChecked(),
            wedge_pa=float(self.wedge_pa.value()),
            wedge_width=float(self.wedge_width.value()),
            hq_interp=self.hq_interp.isChecked(),
            downsample=4 if self.quick_preview.isChecked() else 1,
        )

        n_est = int(max(0, (p["maxsma"] - p["minsma"]) / max(1e-6, p["step"])))
        ds    = int(max(1, p["downsample"]))
        self._last_run_timer = (time.perf_counter(), n_est, ds)

        spr = self._perf.get(ds)
        if spr is None:
            other = 4 if ds == 1 else 1
            other_spr = self._perf.get(other)
            if other_spr is not None:
                spr = other_spr * ((other / ds) ** 2)
        busy_hint = ""
        if spr is not None and n_est > 0:
            eta_sec = spr * n_est
            s = max(0.0, float(eta_sec))
            if s < 1.0: eta = f"{s:.1f}s"
            else:
                m, s = divmod(int(round(s)), 60)
                if m < 1: eta = f"{s:d}s"
                else:
                    h, m = divmod(m, 60)
                    eta = f"{m:d}m {s:d}s" if h < 1 else f"{h:d}h {m:d}m"
            busy_hint = f" (~{n_est:,} rings, est {eta})"

        self._last_fit_params = dict(p)
        self._busy = QProgressDialog(f"Fitting…{busy_hint}", None, 0, 100, self)
        self._busy.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._busy.setAutoClose(False); self._busy.setAutoReset(False)
        self._busy.show()

        fit_img = self._in01
        self._thread = QThread(self)
        self._worker = _FitWorker(fit_img, p)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(lambda pct, msg: (self._busy.setValue(pct), self._busy.setLabelText(msg)))
        self._worker.finished.connect(self._on_fit_finished)
        self._worker.error.connect(self._on_fit_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _build_preview_cache_from_fit(self):
        cx, cy, sma, eps, pa_deg = self._fit_boundary()
        alpha, m = self._elliptical_alpha(self._resid.shape, cx, cy, sma, eps, pa_deg, feather_frac=0.04)
        r = np.nan_to_num(self._resid, 0.0, 0.0, 0.0).astype(np.float32, copy=False)
        inside = (m <= 1.0)
        abs_in = np.abs(r[inside]) if inside.any() else np.abs(r)
        S = float(np.percentile(abs_in, 99.5)) if abs_in.size else 1.0
        if not np.isfinite(S) or S <= 0:
            S = float(np.max(np.abs(r)) or 1.0)
        self._alpha_fit   = alpha
        self._resid01_fit = np.clip(0.5 + (r/(2.0*S)), 0.0, 1.0)
        self._S_fit       = S

    def _rebuild_right_preview(self):
        if self._resid is None:
            return
        orig01 = getattr(self, "_orig01", np.clip(self._img, 0.0, 1.0).astype(np.float32))
        if self.preview_blend.isChecked():
            disp01 = self._alpha_fit * self._resid01_fit + (1.0 - self._alpha_fit) * orig01
        else:
            disp01 = self._resid01_fit
        self._preview_right01 = disp01.astype(np.float32, copy=True)
        self.right.setSceneImage(self._pix_from_01(disp01), self._resid.shape)

    def _refresh_preview(self):
        if self._resid is None:
            return
        if getattr(self, "_in01", None) is None:
            self._in01 = self._compute_input01()
        cx, cy, sma, eps, pa_deg = self._fit_boundary()
        alpha, m = self._elliptical_alpha(self._resid.shape, cx, cy, sma, eps, pa_deg, feather_frac=0.04)
        r = np.nan_to_num(self._resid, 0.0, 0.0, 0.0).astype(np.float32)
        inside = (m <= 1.0)
        if self.use_wedge.isChecked():
            inside &= ~self._make_wedge_mask(*self._resid.shape)
        abs_in = np.abs(r[inside]) if inside.any() else np.abs(r)
        S = float(np.percentile(abs_in, 99.5)) if abs_in.size else 1.0
        if not np.isfinite(S) or S <= 0:
            S = float(np.max(np.abs(r)) or 1.0)
        self._last_preview_scale_S = S
        resid01 = np.clip(0.5 + (r / (2.0 * S)), 0.0, 1.0)
        orig01  = self._in01
        disp01 = alpha * resid01 + (1.0 - alpha) * orig01 if self.preview_blend.isChecked() else resid01
        self._preview_right01 = disp01.astype(np.float32, copy=True)
        self.right.setSceneImage(self._pix_from_01(disp01), self._resid.shape)

    def _fit_boundary(self):
        if getattr(self, "_isolist", None):
            for iso in reversed(self._isolist):
                cx  = float(getattr(iso, "x0",  getattr(iso, "x0_center", np.nan)))
                cy  = float(getattr(iso, "y0",  getattr(iso, "y0_center", np.nan)))
                sma = float(getattr(iso, "sma", getattr(iso, "sma0",      np.nan)))
                eps = float(getattr(iso, "eps", getattr(iso, "ellipticity", np.nan)))
                pa  = float(getattr(iso, "pa",  getattr(iso, "position_angle", np.nan)))
                if (np.isfinite([cx, cy, sma, eps, pa]).all() and sma > 0.0 and 0.0 <= eps < 1.0):
                    return cx, cy, sma, float(np.clip(eps, 0.0, 0.95)), float(np.rad2deg(pa))
        return (self._cx, self._cy,
                float(self.maxsma.value()),
                float(self.eps.value()),
                float(self.pa_deg.value()))

    def _elliptical_alpha(self, shape, cx, cy, sma, eps, pa_deg, feather_frac=0.04):
        if not np.isfinite(pa_deg):
            pa_deg = float(self.pa_deg.value())
        a = float(max(1.0, sma))
        b = float(max(1.0, a * (1.0 - float(eps))))
        th = np.deg2rad(pa_deg)
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]
        x = xx - cx; y = yy - cy
        c, s = np.cos(th), np.sin(th)
        xr =  x*c + y*s
        yr = -x*s + y*c
        m = np.sqrt((xr/a)**2 + (yr/b)**2)
        band = max(1e-3, float(feather_frac))
        alpha = np.clip((1.0 - m) / band, 0.0, 1.0)
        return alpha, m

    def _on_fit_finished(self, model, resid, isolist):
        if self._last_run_timer is not None:
            start_t, rings, ds = self._last_run_timer
            self._last_run_timer = None
            if rings > 0:
                elapsed = max(0.0, time.perf_counter() - start_t)
                spr_meas = elapsed / rings
                prev = self._perf.get(ds)
                alpha = 0.35
                self._perf[ds] = spr_meas if prev is None else (alpha * spr_meas + (1 - alpha) * prev)
        self._update_lowres_hints()
        self._model = model; self._resid = resid; self._isolist = isolist
        if hasattr(self, "_busy") and self._busy is not None:
            self._busy.close(); self._busy = None
        self._build_preview_cache_from_fit(); self._rebuild_right_preview()

    def _on_fit_error(self, msg):
        if hasattr(self, "_busy") and self._busy is not None:
            self._busy.close(); self._busy = None
        self._last_run_timer = None
        QMessageBox.critical(self, "Isophote Fit Error", msg)

    # ---------- export (push to doc_manager) ----------
    def _push_product(self, which: str, variant: str):
        if self.doc_manager is None:
            QMessageBox.information(self, "GLIMR", "No document manager available.")
            return
        arr = self._resid if which == "resid" else self._model
        if arr is None:
            QMessageBox.information(self, "GLIMR", f"Run the fit first to generate the {which}.")
            return

        step_name = f"Isophote {which.capitalize()}"
        ds = int(max(1, (self._last_fit_params or {}).get("downsample", 1)))
        if ds > 1: step_name += " (quick preview)"

        meta_common = {
            "from": "GLIMR",
            "product": which,
            "downsample": ds,
            "isophote_params": (
                {k: self._last_fit_params.get(k) for k in (
                    "cx","cy","sma0","minsma","maxsma","step","sclip","nclip",
                    "eps","pa_deg","fix_center","fix_pa","fix_eps",
                    "high_harm","use_wedge","wedge_pa","wedge_width","hq_interp","downsample"
                )} if self._last_fit_params else None
            )
        }

        title = ""
        if which == "resid" and variant == "visible":
            if self._preview_right01 is None:
                self._refresh_preview()
            data01 = np.clip(np.nan_to_num(self._preview_right01, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
            meta = {**meta_common, "push_variant": "visible_preview",
                    "preview_blend": bool(self.preview_blend.isChecked()),
                    "feather_frac": 0.04,
                    "note": "Exact right-pane display"}
            title = "GLIMR Residual (visible)"
        elif which == "resid" and variant == "stretched":
            data01, S = self._residual_preview_stretch01(arr, pct=99.5)
            meta = {**meta_common, "push_variant": "preview_stretch",
                    "stretch_pct": 99.5, "stretch_scale_S": float(S),
                    "zero_maps_to_gray_0p5": True}
            title = "GLIMR Residual (stretched)"
        else:
            # normalized (min→0, max→1), optionally shift residuals to ≥0 first
            data = np.asarray(arr, dtype=np.float32)
            if which == "resid" and self.save_resid_shifted.isChecked():
                mn = float(np.nanmin(data))
                if np.isfinite(mn) and mn < 0.0:
                    data = data - mn
            data01, vmin, vmax = self._normalize01_for_push(data)
            meta = {**meta_common, "push_variant": "normalized_01",
                    "source_range_min": float(vmin), "source_range_max": float(vmax)}
            title = f"GLIMR {which.capitalize()}"
            if ds > 1: title += " (quick)"

        ok = self._push_array_to_doc_manager(data01, title, meta)
        if not ok:
            QMessageBox.warning(self, "GLIMR", "Could not create a new document in doc manager.")

    def _push_array_to_doc_manager(self, arr01: np.ndarray, title: str, meta: dict) -> bool:
        """Create a brand-new document in your DocManager."""
        dm = self.doc_manager
        if dm is None:
            return False

        # ensure float32 [0..1]
        img = np.asarray(arr01, dtype=np.float32)

        # 1) Preferred: open_array(img, metadata=None, title=None)
        fn = getattr(dm, "open_array", None)
        if callable(fn):
            try:
                fn(img, metadata=dict(meta or {}), title=title)
                return True
            except Exception:
                pass

        # 2) Also supported in your manager: create_document(image, metadata=None, name=None)
        fn = getattr(dm, "create_document", None)
        if callable(fn):
            try:
                fn(image=img, metadata=dict(meta or {}), name=title)
                return True
            except Exception:
                pass

        # 3) Alias present: open_numpy == open_array (same signature)
        fn = getattr(dm, "open_numpy", None)
        if callable(fn):
            try:
                fn(img, metadata=dict(meta or {}), title=title)
                return True
            except Exception:
                pass

        # If none of the above worked, report failure
        return False


    # ---------- save ----------
    def _save_fits(self, which="resid"):
        arr = self._resid if which == "resid" else self._model
        if arr is None:
            QMessageBox.information(self, "Nothing to save",
                                    f"Run the fit first to generate the {which}.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, f"Save {which} FITS", f"{which}.fits", "FITS files (*.fits *.fit)")
        if not fn:
            return
        try:
            data = np.asarray(arr, dtype=np.float32)
            orig_min = float(np.nanmin(data)); orig_max = float(np.nanmax(data))
            pedestal = 0.0
            if which == "resid" and self.save_resid_shifted.isChecked():
                if np.isfinite(orig_min) and orig_min < 0.0:
                    pedestal = -orig_min; data = data + pedestal
            hdr = fits.Header()
            hdr["CREATOR"]  = ("GLIMR", "SASpro Isophote Modeler")
            hdr["PRODUCT"]  = (which, "model or resid")
            hdr["ORIGMIN"]  = (orig_min, "Min before any pedestal")
            hdr["ORIGMAX"]  = (orig_max, "Max before any pedestal")
            hdr["PEDESTAL"] = (float(pedestal), "Added so min(data)>=0 at save time")
            ds = 1
            if getattr(self, "_last_fit_params", None):
                ds = int(max(1, self._last_fit_params.get("downsample", 1)))
            hdr["DSFACTOR"] = (ds, "Downsample factor used for fit (then upsampled)")
            p = self._last_fit_params or {}
            hdr["ISO_EPS"]   = (float(p.get("eps", np.nan)),       "Seed ellipticity")
            hdr["ISO_PA"]    = (float(p.get("pa_deg", np.nan)),    "Seed PA (deg)")
            hdr["ISO_SMA0"]  = (float(p.get("sma0", np.nan)),      "Initial SMA (px)")
            hdr["ISO_MIN"]   = (float(p.get("minsma", np.nan)),    "Min SMA (px)")
            hdr["ISO_MAX"]   = (float(p.get("maxsma", np.nan)),    "Max SMA (px)")
            hdr["ISO_STEP"]  = (float(p.get("step", np.nan)),      "SMA step (px)")
            hdr["ISO_SCLIP"] = (float(p.get("sclip", np.nan)),     "Sigma clip")
            hdr["ISO_NCLIP"] = (int(p.get("nclip", 0)),            "Sigma clip iters")
            hdr["ISO_FXC"]   = (bool(p.get("fix_center", False)),  "Fix center")
            hdr["ISO_FPA"]   = (bool(p.get("fix_pa", False)),      "Fix PA")
            hdr["ISO_FEPS"]  = (bool(p.get("fix_eps", False)),     "Fix ellipticity")
            hdr["ISO_HARM"]  = (bool(p.get("high_harm", False)),   "Use a3/b3/a4/b4")
            hdr["ISO_WEDGE"] = (bool(p.get("use_wedge", False)),   "Exclude wedge")
            if p.get("use_wedge", False):
                hdr["ISO_WPA"]  = (float(p.get("wedge_pa", np.nan)),   "Wedge PA (deg)")
                hdr["ISO_WWID"] = (float(p.get("wedge_width", np.nan)),"Wedge width (deg)")
            fits.PrimaryHDU(data.astype(np.float32), header=hdr).writeto(fn, overwrite=True)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))