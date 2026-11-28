# saspro/convo.py
from __future__ import annotations

import os
import math
import numpy as np
from typing import Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import logging
import threading

# ── SciPy / scikit-image
from scipy.signal import fftconvolve
import scipy.fft as _sp_fft
from scipy.ndimage import laplace
from numpy.fft import fft2, ifft2, ifftshift

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.color import rgb2lab, lab2rgb
from skimage.util import img_as_float32
from skimage.transform import warp, AffineTransform
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means, denoise_wavelet, estimate_sigma

# ── Qt
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QImage, QPainter, QPen, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMessageBox,
    QDialog, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QSlider, QLineEdit,
    QFormLayout, QTabWidget, QComboBox, QCheckBox, QPushButton, QToolButton,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QWidget,
    QSpinBox, QDoubleSpinBox
)
import cv2
# Optional FITS export
from astropy.io import fits


# Import centralized widgets
from setiastro.saspro.widgets.spinboxes import CustomSpinBox
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.widgets.before_after import (
    BeforeAfterCompare, display_stretch_params, apply_display_stretch,
)
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.thread_keeper import start_owned_thread, thread_running
from setiastro.saspro.psf_utils import detect_stars_waterfall, median_star_stamp

from PyQt6.QtCore import QThread, pyqtSignal as _pyqtSignal
from PyQt6.QtCore import QCoreApplication
import numpy as np


def _tr(text: str) -> str:
    return QCoreApplication.translate("Convo", text)
 
 
class _Canceled(Exception):
    """Raised inside a preview job once its cancel flag is set."""


class _PreviewWorker(QThread):
    """
    Computes one Convo/Deconvo preview off the GUI thread.

    The job is a plain function that ConvoDeconvoDialog._build_job makes from a
    snapshot of the settings, so nothing here reads a widget. Started with
    thread_keeper.start_owned_thread (no parent): the dialog may close, or be
    deleted, at any point of a run, and its result is then dropped by job id.

    Signals
    -------
    progress(int, str)        — job id, short status text
    result(int, object, str)  — job id, payload dict | None, error
                                ("" ok, "__canceled__", else the traceback)
    QThread.finished is not shadowed: thread_keeper releases the thread on it.
    """
    progress = _pyqtSignal(int, str)
    result = _pyqtSignal(int, object, str)

    def __init__(self, job_id: int, fn):
        super().__init__(None)
        self._job_id = job_id
        self._fn = fn
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    def _report(self, text: str):
        self.progress.emit(self._job_id, str(text))

    def run(self):
        payload, err = None, ""
        try:
            payload = self._fn(self._report, self._stop.is_set)
            if self._stop.is_set():
                payload, err = None, "__canceled__"
        except _Canceled:
            err = "__canceled__"
        except Exception as exc:
            import traceback
            err = "__canceled__" if self._stop.is_set() else f"{exc}\n\n{traceback.format_exc()}"
        self._fn = None   # the job's arrays leave with the payload, not with the thread
        self.result.emit(self._job_id, payload, err)


class _JobSpec:
    """One preview request, read from the widgets on the GUI thread.

    compute_raw(report, canceled) gives the tool's output before Strength and
    mask; blend(raw) applies Strength exactly as the dialog always has. raw_key
    holds every setting compute_raw depends on, so equal keys over the same
    image give the same raw result.
    """
    __slots__ = ("doc", "img", "orig", "tab", "raw_key", "strength", "mask_id", "mask_data",
                 "compute_raw", "blend", "denoise", "label", "preset")

    def same_raw(self, other) -> bool:
        return other is not None and self.raw_key == other.raw_key and self.img is other.img

    def same_result(self, other) -> bool:
        return (self.same_raw(other) and self.strength == other.strength
                and self.orig is other.orig and self.mask_id == other.mask_id
                and self.mask_data is other.mask_data)

# --- GraphicsView with Shift+Click LS center + optional scene ctor -----------

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPen, QColor, QWheelEvent
from typing import Optional, Tuple
 
 
class InteractiveGraphicsView(QGraphicsView):
    """
    GraphicsView with:
    - Shift+click to set Larson–Sekanina center (crosshair)
    - Mouse-wheel zoom (smooth trackpad + click-wheel)
    """
    def __init__(self, scene: QGraphicsScene | None = None, parent=None):
        super().__init__(parent)
        if scene is not None:
            self.setScene(scene)
        self.ls_center: Optional[Tuple[float, float]] = None
        self.cross_items = []
        # Image bounds in scene coords. Side-by-side comparison puts a second
        # copy left of the image; a centre picked there would be off-image.
        self.image_rect: QRectF | None = None
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def mousePressEvent(self, event):
        if (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_pt = self.mapToScene(event.position().toPoint())
            if self.image_rect is not None and not self.image_rect.contains(scene_pt):
                return
            x, y = scene_pt.x(), scene_pt.y()
            self.ls_center = (x, y)
            self._draw_crosshair_at(x, y)
            return
        super().mousePressEvent(event)
 
    def _draw_crosshair_at(self, x: float, y: float):
        for item in self.cross_items:
            self.scene().removeItem(item)
        self.cross_items.clear()
        size = 10
        pen = QPen(QColor(255, 0, 0), 2)
        hline = self.scene().addLine(x - size, y, x + size, y, pen)
        vline = self.scene().addLine(x, y - size, x, y + size, pen)
        self.cross_items.extend([hline, vline])
 
    def wheelEvent(self, event: QWheelEvent):
        """
        Smooth zoom centred on cursor position.
        Trackpad: uses pixelDelta (fine-grained).
        Click-wheel: uses angleDelta (coarse steps).
        Ctrl held = faster zoom in both cases.
        """
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
 
        # ── Trackpad path (pixelDelta available) ──────────────
        dy_px = event.pixelDelta().y()
        if dy_px != 0:
            abs_dy = abs(dy_px)
            if abs_dy <= 3:
                base = 1.012 if ctrl else 1.010
            elif abs_dy <= 10:
                base = 1.025 if ctrl else 1.020
            else:
                base = 1.040 if ctrl else 1.030
            factor = base if dy_px > 0 else 1.0 / base
            self.scale(factor, factor)
            event.accept()
            return
 
        # ── Click-wheel path (angleDelta) ─────────────────────
        dy_ang = event.angleDelta().y()
        if dy_ang == 0:
            event.ignore()
            return
        step = 1.20 if ctrl else 1.15
        factor = step if dy_ang > 0 else 1.0 / step
        self.scale(factor, factor)
        event.accept()



class FloatSliderWithEdit(QWidget):
    """
    Integer slider + float line edit, mapped by fixed step; emits valueChanged(float)
    """
    valueChanged = pyqtSignal(float)

    def __init__(self, *, minimum: float, maximum: float, step: float, initial: float, suffix: str = "", parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._step = step
        self._suffix = suffix
        self._factor = 1.0 / step
        self._int_min = int(round(minimum * self._factor))
        self._int_max = int(round(maximum * self._factor))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(self._int_min, self._int_max)
        layout.addWidget(self.slider, stretch=1)

        self.edit = QLineEdit(self)
        self.edit.setFixedWidth(60)
        validator = QDoubleValidator(minimum, maximum, int(abs(np.log10(step))), self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.edit.setValidator(validator)
        layout.addWidget(self.edit)

        self.setValue(initial)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.edit.editingFinished.connect(self._on_edit_finished)

    def _on_slider_changed(self, int_val: int):
        f = int_val / self._factor
        f = min(max(f, self._min), self._max)
        text = f"{f:.{max(0, int(-np.log10(self._step)))}f}{self._suffix}"
        self.edit.blockSignals(True)
        self.edit.setText(text)
        self.edit.blockSignals(False)
        self.valueChanged.emit(f)

    def _on_edit_finished(self):
        txt = self.edit.text().rstrip(self._suffix)
        try:
            f = float(txt)
        except ValueError:
            f = self.slider.value() / self._factor
        f = min(max(f, self._min), self._max)
        int_val = int(round(f * self._factor))
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)

    def value(self) -> float:
        return self.slider.value() / self._factor

    def setValue(self, f: float):
        f = min(max(f, self._min), self._max)
        int_val = int(round(f * self._factor))
        self.slider.blockSignals(True)
        self.slider.setValue(int_val)
        self.slider.blockSignals(False)
        s = f"{(int_val / self._factor):.{max(0, int(-np.log10(self._step)))}f}{self._suffix}"
        self.edit.setText(s)
        self.valueChanged.emit(int_val / self._factor)


def _control_value(w):
    """Current value of a parameter control, for Reset snapshots."""
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    return w.value()          # FloatSliderWithEdit / CustomSpinBox


def _set_control_value(w, v):
    if isinstance(w, QComboBox):
        w.setCurrentText(v)
    elif isinstance(w, QCheckBox):
        w.setChecked(v)
    else:
        w.setValue(v)


# ============= Convo/Deconvo dialog (DocManager-powered) =====================
class ConvoDeconvoDialog(QDialog):
    """
    SASpro version: takes a DocManager, no ImageManager dependency.
    """
    def __init__(self, doc_manager, parent=None, doc=None):
        super().__init__(parent)
        self.doc_manager = doc_manager
        self._main = parent  # keep a ref to the main window (has _active_doc + signal)
        self._doc_override = doc  # ← explicit doc (ROI or full) from the MDI

        # Only follow global active-doc changes if we *weren't* given a doc
        if hasattr(self._main, "currentDocumentChanged") and self._doc_override is None:
            self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions
        self.setWindowTitle(self.tr("Convolution / Deconvolution"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.resize(1000, 650)
        self._use_custom_psf = False
        self._custom_psf: Optional[np.ndarray] = None
        self._last_stellar_psf: Optional[np.ndarray] = None
        self._psf_job = None        # running _PSFEstimateJob, if any
        self._original_image: Optional[np.ndarray] = None
        self._preview_result: Optional[np.ndarray] = None
        # Previews are computed on a worker thread, one job at a time (_start_job).
        self._preview_spec = None      # _JobSpec that _preview_result was computed for
        self._raw_cache = None         # (_JobSpec, result before Strength/mask): Strength-only changes re-blend
        self._job = None               # running job: {"id", "spec", "worker", "canceled"}
        self._job_seq = 0
        self._queued_preview = False   # Preview clicked mid-run: rerun with the latest settings
        self._push_spec = None         # Push clicked with no preview for its settings yet
        self._shown = None             # array the pane shows, and its stretch key (None = unstretched)
        self._shown_stretch_key = None
        self._auto_fit = False
        self._load_original_on_show = True
        self._pending_ls_center = None  # stash for double-click-open LS center seed

        # ── Layout: left controls / right preview
        main_layout = QHBoxLayout(self)
        # Left
        left_panel = QFrame(); left_panel.setFrameShape(QFrame.Shape.StyledPanel); left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel); main_layout.addWidget(left_panel)
        # Right
        preview_panel = QFrame()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setSpacing(4)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(preview_panel, stretch=1)

        # ── Autostretch toolbar (above zoom row) ─────────────────────
        stretch_bar = QHBoxLayout()
        stretch_bar.setSpacing(8)

        self.cb_autostretch = QCheckBox(self.tr("Auto-stretch preview"))
        self.cb_autostretch.setChecked(False)
        self.cb_autostretch.setToolTip(
            self.tr("Apply a statistical stretch to the preview only.\n"
            "The actual image data is NOT modified — this is display only.\n"
            "Useful for checking denoising / convolution results on linear data.")
        )
        stretch_bar.addWidget(self.cb_autostretch)

        stretch_bar.addWidget(QLabel(self.tr("Target median:")))
        self.s_target_median = QDoubleSpinBox()
        self.s_target_median.setRange(0.01, 0.60)
        self.s_target_median.setSingleStep(0.01)
        self.s_target_median.setDecimals(3)
        self.s_target_median.setValue(0.25)
        self.s_target_median.setFixedWidth(70)
        self.s_target_median.setEnabled(False)   # disabled until checkbox on
        stretch_bar.addWidget(self.s_target_median)

        self.cb_linked = QCheckBox(self.tr("Linked"))
        self.cb_linked.setChecked(True)
        self.cb_linked.setToolTip(self.tr("Stretch all RGB channels together (linked)."))
        self.cb_linked.setEnabled(False)
        stretch_bar.addWidget(self.cb_linked)

        stretch_bar.addStretch(1)
        preview_layout.addLayout(stretch_bar)

        # Wire stretch controls
        self.cb_autostretch.toggled.connect(self._on_autostretch_toggled)
        self.s_target_median.valueChanged.connect(self._refresh_display)
        self.cb_linked.toggled.connect(self._refresh_display)


        # Tabs
        self.tabs = QTabWidget(); left_layout.addWidget(self.tabs)
        self.deconv_param_stack: dict[str, QWidget] = {}
        self._build_convolution_tab()
        self._build_deconvolution_tab()
        self._build_psf_estimator_tab()
        self._build_tv_denoise_tab()

        # PSF preview chip
        self.conv_psf_label = QLabel(); self.conv_psf_label.setFixedSize(64, 64)
        self.conv_psf_label.setStyleSheet("border: 1px solid #888;")
        left_layout.addWidget(self.conv_psf_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Strength
        self.strength_slider = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=1.0, suffix="")
        srow = QHBoxLayout(); srow.addWidget(QLabel(self.tr("Strength:"))); srow.addWidget(self.strength_slider)
        left_layout.addLayout(srow)

        # Buttons
        row1 = QHBoxLayout()
        self.preview_btn = QPushButton(self.tr("Preview"))
        self.undo_btn    = QPushButton(self.tr("Undo"))
        self.reset_btn   = QPushButton(self.tr("Reset"))
        self.close_btn   = QPushButton(self.tr("Close"))
        self.undo_btn.setToolTip(self.tr("Discard the preview and show the original image"))
        self.reset_btn.setToolTip(self.tr("Restore this tab's settings and Strength to their defaults"))
        row1.addWidget(self.preview_btn); row1.addWidget(self.undo_btn); row1.addWidget(self.reset_btn)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.push_btn    = QPushButton(self.tr("Push"))
        self.push_btn.setToolTip(self.tr("Apply the current settings to the image. "
                                         "Computes them first when there is no preview for them."))
        row2.addWidget(self.push_btn); row2.addWidget(self.close_btn)
        left_layout.addLayout(row2)

        # --- preset drag handle (grip) ---
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            try:
                from setiastro.saspro.resources import convoicon_path
                _grip_icon = QIcon(convoicon_path)
            except Exception:
                _grip_icon = QIcon()
            drag_row = QHBoxLayout()
            drag_row.setContentsMargins(0, 0, 0, 0)
            self.preset_drag_handle = PresetDragHandle(
                "convo", self.get_preset, icon=_grip_icon,
                tooltip=self.tr(
                    "Drag to the canvas to create a Convolution/Deconvolution shortcut "
                    "with these exact settings (current tab).\n"
                    "Drop directly on an image to apply them headlessly."
                ),
                parent=self,
            )
            drag_row.addWidget(self.preset_drag_handle)
            self.btn_help = make_help_button("convo", self)
            drag_row.addStretch(1)
            drag_row.addWidget(self.btn_help)
        except Exception:
            drag_row = None

        left_layout.addStretch()
        self.rl_status_label = QLabel(""); self.rl_status_label.setStyleSheet("color:#fff;background:#333;padding:4px;")
        self.rl_status_label.setFixedHeight(24)
        # Cancel for a preview (or Push) still computing; hidden when idle.
        self.job_cancel_btn = QPushButton(self.tr("Cancel"))
        self.job_cancel_btn.setToolTip(self.tr("Stop the computation in progress"))
        self.job_cancel_btn.setFixedHeight(24)
        self.job_cancel_btn.setVisible(False)
        self.job_cancel_btn.clicked.connect(self._cancel_jobs)
        status_row = QHBoxLayout(); status_row.setSpacing(4)
        status_row.addWidget(self.rl_status_label, stretch=1)
        status_row.addWidget(self.job_cancel_btn)
        left_layout.addLayout(status_row)
        if drag_row is not None:
            left_layout.addLayout(drag_row)     # grip + help: the last row of the controls

        # Zoom & Preview
        zrow = QHBoxLayout(); zrow.addStretch()
        self.zoom_in_btn = QToolButton();  self.zoom_in_btn.setIcon(QIcon.fromTheme("zoom-in"));     self.zoom_in_btn.setToolTip(self.tr("Zoom In"))
        self.zoom_out_btn= QToolButton();  self.zoom_out_btn.setIcon(QIcon.fromTheme("zoom-out"));    self.zoom_out_btn.setToolTip(self.tr("Zoom Out"))
        self.fit_btn     = QToolButton();  self.fit_btn.setIcon(QIcon.fromTheme("zoom-fit-best"));    self.fit_btn.setToolTip(self.tr("Fit to Preview"))
        zrow.addWidget(self.zoom_in_btn); zrow.addWidget(self.zoom_out_btn); zrow.addWidget(self.fit_btn)
        preview_layout.addLayout(zrow)

        self.scene = QGraphicsScene()
        self.view = InteractiveGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pixmap_item = QGraphicsPixmapItem(); self.scene.addItem(self.pixmap_item)
        preview_layout.addWidget(self.view)

        # Before/after comparison: pixmap_item is the "after"; the untouched
        # image goes underneath through the same display stretch.
        self._cmp = BeforeAfterCompare(self.view, self.scene, [self.pixmap_item],
                                       settings_key="convo/compare", host=self)
        zrow.insertWidget(0, self._cmp.toolbar)
        self._cmp.fitRequested.connect(self._on_fit_clicked)
        self._stretch_cache = None     # (before array, target, linked, params)
        self._pm_before_cache = None   # (before array, stretch key, QPixmap)

        # Signals
        self.preview_btn.clicked.connect(self._on_preview)
        self.undo_btn.clicked.connect(self._on_undo)
        self.reset_btn.clicked.connect(self._on_reset)
        self.push_btn.clicked.connect(self._on_push_to_doc)
        self.close_btn.clicked.connect(self.close)
        # Esc / reject() closes without a closeEvent (Qt >= 6.3); both paths emit finished.
        self.finished.connect(self._teardown_jobs)

        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.fit_btn.clicked.connect(self._on_fit_clicked)

        self.tabs.currentChanged.connect(self._update_psf_preview)
        self.deconv_algo_combo.currentTextChanged.connect(self._update_psf_preview)

        self.sep_run_button.clicked.connect(self._on_run_sep)
        self.sep_use_button.clicked.connect(self._on_use_stellar_psf)
        self.sep_save_button.clicked.connect(self._on_save_stellar_psf)

        for s in (self.conv_radius_slider, self.conv_shape_slider, self.conv_aspect_slider, self.conv_rotation_slider):
            s.valueChanged.connect(self._update_psf_preview)
        for s in (self.rl_psf_radius_slider, self.rl_psf_shape_slider, self.rl_psf_aspect_slider, self.rl_psf_rotation_slider):
            s.valueChanged.connect(self._update_psf_preview)

        self._snapshot_reset_defaults()
        self._update_psf_preview()

    def _active_doc(self):
        # 1) If we were given a specific doc (ROI or full), always use that.
        if getattr(self, "_doc_override", None) is not None:
            return self._doc_override

        # 2) Otherwise fall back to the MDI's notion of active
        if self._main is not None and hasattr(self._main, "_active_doc") and callable(self._main._active_doc):
            try:
                return self._main._active_doc()
            except Exception:
                pass

        # 3) Last resort: DocManager's active doc
        if hasattr(self.doc_manager, "get_active_document"):
            return self.doc_manager.get_active_document()

        return None


    def _on_active_doc_changed(self, doc):
        # If this dialog is bound to a specific doc (ROI/full), ignore global changes
        if getattr(self, "_doc_override", None) is not None:
            return

        img = getattr(doc, "image", None)
        self._teardown_jobs()
        self._preview_result = None
        self._original_image = img.copy() if isinstance(img, np.ndarray) else None
        if self._original_image is not None:
            self._auto_fit = True
            self._refresh_display()

    def _on_autostretch_toggled(self, on: bool):
        """Enable/disable dependent controls and refresh display."""
        self.s_target_median.setEnabled(on)
        self.cb_linked.setEnabled(on)
        self._refresh_display()


    def _refresh_display(self, *_, prebuilt=None):
        """
        Render the preview pane. The preview result (or the original, when
        there is none) is the "after"; the original is the comparison's
        "before". Called whenever stretch parameters change, or after
        Preview/Undo/Push changes either image.

        Autostretch is measured ONCE, on the original, and applied to both
        sides. Stretching each on its own statistics would shift the result's
        brightness (a denoise lowers the noise, which moves the black point)
        and the before/after comparison would show a difference that is not
        in the data.

        prebuilt: the images a preview job already made off the GUI thread
        (_display_images). Used only while the display settings still match.
        """
        before = self._original_image
        after = self._preview_result
        shown = after if after is not None else before
        if shown is None:
            return

        auto = self.cb_autostretch.isChecked()
        tm = float(self.s_target_median.value())
        linked = bool(self.cb_linked.isChecked())
        pre = prebuilt
        if pre is not None and not (pre["shown"] is shown and pre["orig"] is before
                                    and pre["auto"] == auto
                                    and (not auto or (pre["tm"] == tm and pre["linked"] == linked))):
            pre = None      # the display settings moved while the job ran

        stretch = None
        if auto:
            if pre is not None and pre["params"] is not None and before is not None:
                self._stretch_cache = (before, tm, linked, pre["params"])
            stretch = self._display_stretch(before if before is not None else shown)

        def _disp(a):
            return apply_display_stretch(a, stretch) if stretch is not None else a

        if pre is not None:
            self.pixmap_item.setPixmap(QPixmap.fromImage(pre["after"][0]))
        else:
            self.pixmap_item.setPixmap(self._pixmap_from_array(_disp(shown)))
        self._shown = shown
        self._shown_stretch_key = None if stretch is None else (tm, linked)
        h, w = shown.shape[:2]
        self.view.image_rect = QRectF(0, 0, w, h)
        self._cmp.set_image_size(w, h)

        if after is not None and before is not None and before.shape[:2] == after.shape[:2]:
            key = None if stretch is None else (tm, linked)
            c = self._pm_before_cache
            if c is None or c[0] is not before or c[1] != key:
                if pre is not None and pre["before"] is not None:
                    pm = QPixmap.fromImage(pre["before"][0])
                else:
                    pm = self._pixmap_from_array(_disp(before))
                c = (before, key, pm)
                self._pm_before_cache = c
            self._cmp.set_before(c[2])
        else:
            self._pm_before_cache = None
            self._cmp.set_before(None)

        if self._auto_fit:
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._auto_fit = False

    def _display_stretch(self, ref: np.ndarray):
        """Stretch parameters for `ref`, measured once per image and setting.

        The cache holds the array itself and compares by identity: a new
        original always re-measures, with no risk of a recycled id() matching.
        """
        tm = float(self.s_target_median.value())
        linked = bool(self.cb_linked.isChecked())
        c = self._stretch_cache
        if c is None or c[0] is not ref or c[1] != tm or c[2] != linked:
            c = (ref, tm, linked, display_stretch_params(ref, target_median=tm, linked=linked))
            self._stretch_cache = c
        return c[3]

    # ---------------- DocManager IO helpers ----------------
    def _get_active_image_and_meta(self) -> tuple[Optional[np.ndarray], dict]:
        doc = self._active_doc()
        if doc is None or getattr(doc, "image", None) is None:
            return None, {}
        return doc.image, (getattr(doc, "metadata", {}) or {})

    # ---------------- Qt life-cycle ----------------
    def showEvent(self, ev):
        super().showEvent(ev)
        self._preview_result = None
        self._preview_spec = None
        if self._load_original_on_show:
            img, _ = self._get_active_image_and_meta()
            if img is not None:
                self._original_image = img.copy()
                self._auto_fit = True
                self._refresh_display()
            self._load_original_on_show = False
        self.conv_psf_label.clear()
        self.sep_psf_preview.clear() if hasattr(self, "sep_psf_preview") else None
        self._update_psf_preview()
        # Seed LS center now that the scene exists and has been fitted.
        from PyQt6.QtCore import QTimer as _QTimer
        _QTimer.singleShot(0, self._seed_ls_center)

    def closeEvent(self, ev):
        # Clear state so next open starts fresh
        if hasattr(self.view, "ls_center"):
            self.view.ls_center = None
        self._original_image = None
        self._preview_result = None
        self._last_stellar_psf = None
        self._custom_psf = None
        self._use_custom_psf = False
        self._psf_job = None        # a running estimate finishes unseen (thread_keeper holds it)
        self._set_psf_busy(False, "")
        self.conv_psf_label.clear() if hasattr(self, "conv_psf_label") else None
        self.sep_psf_preview.clear() if hasattr(self, "sep_psf_preview") else None
        self.rl_status_label.setText("") if hasattr(self, "rl_status_label") else None
        self.custom_psf_bar.setVisible(False) if hasattr(self, "custom_psf_bar") else None
        self._stretch_cache = None
        self._pm_before_cache = None
        self._cmp.detach()
        # thread_keeper owns the worker: cancel it and let it end on its own,
        # instead of blocking the GUI in wait().
        self._teardown_jobs()


        super().closeEvent(ev)

    # ---------------- Build tabs ----------------
    def _build_convolution_tab(self):
        conv_tab = QWidget()
        layout = QVBoxLayout(conv_tab)
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.conv_radius_slider = FloatSliderWithEdit(minimum=0.1, maximum=200.0, step=0.1, initial=5.0, suffix=" px")
        form.addRow(self.tr("Radius:"), self.conv_radius_slider)

        self.conv_shape_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=2.0, suffix="σ")
        form.addRow(self.tr("Kurtosis (σ):"), self.conv_shape_slider)

        self.conv_aspect_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=1.0, suffix="")
        form.addRow(self.tr("Aspect Ratio:"), self.conv_aspect_slider)

        self.conv_rotation_slider = FloatSliderWithEdit(minimum=0.0, maximum=360.0, step=1.0, initial=0.0, suffix="°")
        form.addRow(self.tr("Rotation:"), self.conv_rotation_slider)

        layout.addLayout(form); layout.addStretch()
        self.tabs.addTab(conv_tab, self.tr("Convolution"))

    def _build_deconvolution_tab(self):
        deconv_tab = QWidget()
        outer_layout = QVBoxLayout(deconv_tab)

        # Algo row
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel(self.tr("Algorithm:")))
        self.deconv_algo_combo = QComboBox()
        self.deconv_algo_combo.addItems(["Richardson-Lucy", "Wiener", "Larson-Sekanina", "Van Cittert"])
        self.deconv_algo_combo.currentTextChanged.connect(self._on_deconv_algo_changed)
        algo_layout.addWidget(self.deconv_algo_combo); algo_layout.addStretch()
        outer_layout.addLayout(algo_layout)

        # PSF sliders (shared for RL/Wiener)
        self.psf_param_group = QWidget()
        psf_group_layout = QFormLayout(self.psf_param_group)
        psf_group_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.rl_psf_radius_slider = FloatSliderWithEdit(minimum=0.1, maximum=100.0, step=0.1, initial=3.0, suffix=" px")
        psf_group_layout.addRow(self.tr("PSF Radius:"), self.rl_psf_radius_slider)
        self.rl_psf_shape_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=2.0, suffix="σ")
        psf_group_layout.addRow(self.tr("PSF Kurtosis (σ):"), self.rl_psf_shape_slider)
        self.rl_psf_aspect_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=1.0, suffix="")
        psf_group_layout.addRow(self.tr("PSF Aspect Ratio:"), self.rl_psf_aspect_slider)
        self.rl_psf_rotation_slider = FloatSliderWithEdit(minimum=0.0, maximum=360.0, step=1.0, initial=0.0, suffix="°")
        psf_group_layout.addRow(self.tr("PSF Rotation:"), self.rl_psf_rotation_slider)
        outer_layout.addWidget(self.psf_param_group)
        self.psf_param_group.setVisible(self.deconv_algo_combo.currentText() in ("Richardson-Lucy", "Wiener"))

        # “Using Stellar PSF” bar
        self.custom_psf_bar = QWidget()
        bar_layout = QHBoxLayout(self.custom_psf_bar); bar_layout.setContentsMargins(0, 0, 0, 0); bar_layout.setSpacing(4)
        self.rl_custom_label = QLabel(self.tr("Using Stellar PSF"))
        self.rl_custom_label.setStyleSheet("color:#fff;background-color:#007acc;padding:2px;")
        self.rl_custom_label.setVisible(False)
        self.rl_disable_custom_btn = QPushButton(self.tr("Disable Stellar PSF"))
        self.rl_disable_custom_btn.setToolTip(self.tr("Revert to PSF sliders"))
        self.rl_disable_custom_btn.setVisible(False)
        self.rl_disable_custom_btn.clicked.connect(self._clear_custom_psf_flag)
        bar_layout.addWidget(self.rl_custom_label); bar_layout.addWidget(self.rl_disable_custom_btn); bar_layout.addStretch()
        outer_layout.addWidget(self.custom_psf_bar)
        self.custom_psf_bar.setVisible(False)

        # Stacked parameter panels
        self.deconv_param_stack.clear()
        self.deconv_stack_container = QWidget(); self.deconv_stack_layout = QVBoxLayout(self.deconv_stack_container)

        # RL
        rl_widget = QWidget()
        rl_form = QFormLayout(rl_widget); rl_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.rl_iterations_slider = FloatSliderWithEdit(minimum=1.0, maximum=100.0, step=1.0, initial=30.0, suffix="")
        rl_form.addRow(self.tr("Iterations:"), self.rl_iterations_slider)
        self.rl_reg_combo = QComboBox(); self.rl_reg_combo.addItems(["None (Plain R–L)", "Tikhonov (L2)", "Total Variation (TV)"])
        rl_form.addRow(self.tr("Regularization:"), self.rl_reg_combo)
        self.rl_clip_checkbox = QCheckBox(self.tr("Enable de‐ring")); self.rl_clip_checkbox.setChecked(True)
        rl_form.addRow("", self.rl_clip_checkbox)
        self.rl_luminance_only_checkbox = QCheckBox(self.tr("Deconvolve L* Only")); self.rl_luminance_only_checkbox.setChecked(True)
        self.rl_luminance_only_checkbox.setToolTip(self.tr("If checked and the image is color, RL runs only on the L* channel."))
        rl_form.addRow("", self.rl_luminance_only_checkbox)
        rl_widget.setLayout(rl_form)
        self.deconv_param_stack["Richardson-Lucy"] = rl_widget

        # Wiener
        wiener_widget = QWidget(); wiener_layout = QVBoxLayout(wiener_widget)
        wiener_form = QFormLayout(); wiener_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.wiener_nsr_slider = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.001, initial=0.01, suffix="")
        wiener_form.addRow(self.tr("Noise/Signal (λ):"), self.wiener_nsr_slider)
        self.wiener_reg_combo = QComboBox(); self.wiener_reg_combo.addItems(["None (Classical Wiener)", "Tikhonov (L2)"])
        wiener_form.addRow(self.tr("Regularization:"), self.wiener_reg_combo)
        self.wiener_luminance_only_checkbox = QCheckBox(self.tr("Deconvolve L* Only")); self.wiener_luminance_only_checkbox.setChecked(True)
        self.wiener_luminance_only_checkbox.setToolTip(self.tr("If checked and the image is color, Wiener runs only on the L* channel."))
        wiener_form.addRow("", self.wiener_luminance_only_checkbox)
        self.wiener_dering_checkbox = QCheckBox(self.tr("Enable de-ring")); self.wiener_dering_checkbox.setChecked(True)
        self.wiener_dering_checkbox.setToolTip(self.tr("Applies a single bilateral pass after Wiener deconvolution"))
        wiener_form.addRow("", self.wiener_dering_checkbox)
        wiener_layout.addLayout(wiener_form)
        self.deconv_param_stack["Wiener"] = wiener_widget

        # Larson–Sekanina
        ls_widget = QWidget(); ls_form = QFormLayout(ls_widget); ls_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.ls_radial_slider  = FloatSliderWithEdit(minimum=0.0, maximum=50.0, step=0.1, initial=0.0, suffix=" px")
        self.ls_angular_slider = FloatSliderWithEdit(minimum=0.1, maximum=360.0, step=0.1, initial=1.0, suffix="°")
        self.ls_operator_combo = QComboBox(); self.ls_operator_combo.addItems(["Divide", "Subtract"])
        self.ls_blend_combo    = QComboBox(); self.ls_blend_combo.addItems(["SoftLight", "Screen"])
        ls_form.addRow(self.tr("Radial Step (px):"), self.ls_radial_slider)
        ls_form.addRow(self.tr("Angular Step (°):"), self.ls_angular_slider)
        ls_form.addRow(self.tr("LS Operator:"), self.ls_operator_combo)
        ls_form.addRow(self.tr("Blend Mode:"), self.ls_blend_combo)
        self.ls_operator_combo.currentTextChanged.connect(self._on_ls_operator_changed)
        self.deconv_param_stack["Larson-Sekanina"] = ls_widget

        # Van Cittert
        vc_widget = QWidget(); vc_form = QFormLayout(vc_widget); vc_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.vc_iterations_slider = FloatSliderWithEdit(minimum=1, maximum=1000, step=1, initial=10, suffix="")
        self.vc_relax_slider      = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=0.0, suffix="")
        vc_form.addRow(self.tr("Iterations:"), self.vc_iterations_slider)
        vc_form.addRow(self.tr("Relaxation (0–1):"), self.vc_relax_slider)
        self.deconv_param_stack["Van Cittert"] = vc_widget

        # Add all panels (hidden initially)
        for widget in self.deconv_param_stack.values():
            widget.setVisible(False)
            self.deconv_stack_layout.addWidget(widget)

        first_algo = self.deconv_algo_combo.currentText()
        if first_algo in self.deconv_param_stack:
            self.deconv_param_stack[first_algo].setVisible(True)

        outer_layout.addWidget(self.deconv_stack_container)
        outer_layout.addStretch()
        self.tabs.addTab(deconv_tab, self.tr("Deconvolution"))

        # Clear “custom PSF” if sliders change
        for s in (self.rl_psf_radius_slider, self.rl_psf_shape_slider, self.rl_psf_aspect_slider, self.rl_psf_rotation_slider):
            s.valueChanged.connect(self._clear_custom_psf_flag)

    def _build_psf_estimator_tab(self):
        psf_tab = QWidget(); layout = QVBoxLayout(psf_tab)

        h_image = QHBoxLayout()
        h_image.addWidget(QLabel(self.tr("Image for PSF Estimate:")))
        self.sep_image_label = QLabel(self.tr("(Current Active Image)"))
        h_image.addWidget(self.sep_image_label); layout.addLayout(h_image)

        form = QFormLayout(); form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Detection is the PSF Viewer's (psf_utils): Auto descends the sigma
        # ladder until ~1000 good stars are in hand; unchecked, one extraction
        # at the slider value. Same range and default as the PSF Viewer.
        self.sep_auto_checkbox = QCheckBox(self.tr("Auto (waterfall → ~{0} stars)").format(PSF_AUTO_TARGET))
        self.sep_auto_checkbox.setToolTip(
            self.tr("When checked, descend through σ = 100, 50, 25, 12, 6, 3 and stop "
            "as soon as ~{0} quality stars are detected, as the PSF Viewer does. "
            "Fast on rich fields. Uncheck to use the Detection σ value literally "
            "(low σ on a large, rich frame can take minutes).").format(PSF_AUTO_TARGET)
        )
        self.sep_auto_checkbox.setChecked(True)
        form.addRow("", self.sep_auto_checkbox)
        self.sep_threshold_slider = FloatSliderWithEdit(minimum=1.0, maximum=50.0, step=0.1, initial=15.0, suffix=" σ")
        self.sep_threshold_slider.setEnabled(False)
        self.sep_auto_checkbox.toggled.connect(lambda on: self.sep_threshold_slider.setEnabled(not on))
        form.addRow(self.tr("Detection σ:"), self.sep_threshold_slider)
        self.sep_minarea_spin = CustomSpinBox(minimum=1, maximum=100, initial=5, step=1)
        form.addRow(self.tr("Min Area (px²):"), self.sep_minarea_spin)
        self.sep_sat_slider = FloatSliderWithEdit(minimum=1000, maximum=100000, step=500, initial=50000, suffix=" ADU")
        self.sep_sat_slider.setToolTip(
            self.tr("Stars whose brightest pixel reaches this level are left out of the PSF. "
            "16-bit ADU: on a normalised (0–1) image 65535 is full scale.")
        )
        form.addRow(self.tr("Saturation Cutoff:"), self.sep_sat_slider)
        self.sep_maxstars_spin = CustomSpinBox(minimum=1, maximum=500, initial=500, step=1)
        self.sep_maxstars_spin.setToolTip(
            self.tr("Stars median-stacked into the PSF, spread evenly through the detections "
            "(the PSF Viewer uses 500).")
        )
        form.addRow(self.tr("Max Stars:"), self.sep_maxstars_spin)
        self.sep_stamp_spin = CustomSpinBox(minimum=5, maximum=50, initial=15, step=1)
        form.addRow(self.tr("Half‐Width (px):"), self.sep_stamp_spin)
        layout.addLayout(form)

        h_buttons = QHBoxLayout()
        self.sep_run_button = QPushButton(self.tr("Run SEP Extraction"))
        self.sep_save_button = QPushButton(self.tr("Save PSF…"))
        self.sep_use_button  = QPushButton(self.tr("Use as Current PSF"))
        h_buttons.addWidget(self.sep_run_button); h_buttons.addWidget(self.sep_save_button); h_buttons.addWidget(self.sep_use_button)
        layout.addLayout(h_buttons)

        self.psf_estimate_title = QLabel(self.tr("Estimated PSF (64×64):"))
        layout.addWidget(self.psf_estimate_title, alignment=Qt.AlignmentFlag.AlignLeft)
        self.sep_psf_preview = QLabel(); self.sep_psf_preview.setFixedSize(64, 64)
        self.sep_psf_preview.setStyleSheet("border: 1px solid #888;")
        layout.addWidget(self.sep_psf_preview, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.sep_status_label = QLabel("")
        self.sep_status_label.setWordWrap(True)
        layout.addWidget(self.sep_status_label)

        layout.addStretch()
        self.tabs.addTab(psf_tab, self.tr("PSF Estimator"))

    def _build_tv_denoise_tab(self):
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
            QLabel, QComboBox, QCheckBox,
        )
        from PyQt6.QtCore import Qt
    
        tvd_tab = QWidget()
        outer = QVBoxLayout(tvd_tab)
        outer.setSpacing(6)
    
        # ── Algorithm selector ───────────────────────────────────
        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel(self.tr("Method:")))
        self.denoise_algo_combo = QComboBox()
        self.denoise_algo_combo.addItems([
            "TV Chambolle",
            "Non-Local Means",
            "Bilateral",
            "Gaussian",
            "Median",
            "Wavelet",
        ])
        algo_row.addWidget(self.denoise_algo_combo)
        algo_row.addStretch(1)
        outer.addLayout(algo_row)
    
        # ── Shared: L*-only ──────────────────────────────────────
        self.denoise_lum_only_chk = QCheckBox(
            self.tr("Operate on L* only  (color images — preserves chroma)")
        )
        self.denoise_lum_only_chk.setChecked(True)
        self.denoise_lum_only_chk.setToolTip(
            self.tr("When checked, the denoiser runs only on the L* (luminance) channel "
            "of a Lab-converted image and chroma is left untouched.\n"
            "Uncheck to run on all channels independently.")
        )
        outer.addWidget(self.denoise_lum_only_chk)
    
        # ── Parameter stack ──────────────────────────────────────
        self._denoise_param_stack: dict[str, QWidget] = {}
    
        # TV Chambolle
        tv_w = QWidget()
        tv_f = QFormLayout(tv_w)
        tv_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.tv_weight_slider = FloatSliderWithEdit(
            minimum=0.0, maximum=1.0, step=0.01, initial=0.10, suffix=""
        )
        tv_f.addRow(self.tr("TV Weight:"), self.tv_weight_slider)
        self.tv_iter_slider = FloatSliderWithEdit(
            minimum=1, maximum=100, step=1, initial=10, suffix=""
        )
        tv_f.addRow(self.tr("Max Iterations:"), self.tv_iter_slider)
        self.tv_multichannel_checkbox = QCheckBox(self.tr("Multi-channel"))
        self.tv_multichannel_checkbox.setChecked(True)
        self.tv_multichannel_checkbox.setVisible(False)
        tv_f.addRow("", self.tv_multichannel_checkbox)
        self._denoise_param_stack["TV Chambolle"] = tv_w
    
        # Non-Local Means
        nlm_w = QWidget()
        nlm_f = QFormLayout(nlm_w)
        nlm_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.nlm_h_slider = FloatSliderWithEdit(
            minimum=0.001, maximum=0.30, step=0.001, initial=0.015, suffix=""
        )
        self.nlm_h_slider.setToolTip(
            self.tr("Cut-off distance (h). Controls denoising strength.\n"
            "Typical range: 0.01–0.05 for stretched astrophotos.")
        )
        nlm_f.addRow(self.tr("h (strength):"), self.nlm_h_slider)
        self.nlm_patch_slider = FloatSliderWithEdit(
            minimum=3, maximum=15, step=2, initial=5, suffix=" px"
        )
        self.nlm_patch_slider.setToolTip(
            self.tr("Patch size (pixels). Larger = more context per comparison, slower.")
        )
        nlm_f.addRow(self.tr("Patch size:"), self.nlm_patch_slider)
        self.nlm_dist_slider = FloatSliderWithEdit(
            minimum=3, maximum=21, step=2, initial=7, suffix=" px"
        )
        self.nlm_dist_slider.setToolTip(
            self.tr("Search distance. Larger = better quality, slower.")
        )
        nlm_f.addRow(self.tr("Search dist:"), self.nlm_dist_slider)
        nlm_hint = QLabel(self.tr("  ⓘ  Fast mode — skimage NLM with sigma estimation."))
        nlm_hint.setStyleSheet("color: #888; font-size: 10px;")
        nlm_f.addRow("", nlm_hint)
        self._denoise_param_stack["Non-Local Means"] = nlm_w
    
        # Bilateral
        bil_w = QWidget()
        bil_f = QFormLayout(bil_w)
        bil_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.bil_d_slider = FloatSliderWithEdit(
            minimum=1, maximum=25, step=2, initial=9, suffix=" px"
        )
        self.bil_d_slider.setToolTip(self.tr("Diameter of each pixel neighbourhood (d)."))
        bil_f.addRow(self.tr("Diameter:"), self.bil_d_slider)
        self.bil_sigma_color_slider = FloatSliderWithEdit(
            minimum=1.0, maximum=200.0, step=1.0, initial=50.0, suffix=""
        )
        self.bil_sigma_color_slider.setToolTip(
            self.tr("Sigma in colour space. Higher = pixels further in colour are mixed.")
        )
        bil_f.addRow(self.tr("Sigma colour:"), self.bil_sigma_color_slider)
        self.bil_sigma_space_slider = FloatSliderWithEdit(
            minimum=1.0, maximum=200.0, step=1.0, initial=50.0, suffix=""
        )
        self.bil_sigma_space_slider.setToolTip(
            self.tr("Sigma in coordinate space. Higher = pixels further apart are mixed.")
        )
        bil_f.addRow(self.tr("Sigma space:"), self.bil_sigma_space_slider)
        bil_hint = QLabel(self.tr("  ⓘ  Uses OpenCV bilateralFilter (8-bit internally)."))
        bil_hint.setStyleSheet("color: #888; font-size: 10px;")
        bil_f.addRow("", bil_hint)
        self._denoise_param_stack["Bilateral"] = bil_w
    
        # Gaussian
        gau_w = QWidget()
        gau_f = QFormLayout(gau_w)
        gau_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.gau_sigma_slider = FloatSliderWithEdit(
            minimum=0.1, maximum=10.0, step=0.1, initial=1.0, suffix=" px"
        )
        self.gau_sigma_slider.setToolTip(
            self.tr("Gaussian sigma. 1.0 = gentle, 3.0+ = heavy blurring.")
        )
        gau_f.addRow(self.tr("Sigma:"), self.gau_sigma_slider)
        self._denoise_param_stack["Gaussian"] = gau_w
    
        # Median
        med_w = QWidget()
        med_f = QFormLayout(med_w)
        med_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.med_size_slider = FloatSliderWithEdit(
            minimum=1, maximum=15, step=2, initial=3, suffix=" px"
        )
        self.med_size_slider.setToolTip(
            self.tr("Kernel size (odd). 3 = minimal, 5 = moderate.\n"
            "Good for cosmic ray speckles.")
        )
        med_f.addRow(self.tr("Kernel size:"), self.med_size_slider)
        self._denoise_param_stack["Median"] = med_w
    
        # Wavelet
        wav_w = QWidget()
        wav_f = QFormLayout(wav_w)
        wav_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.wav_sigma_slider = FloatSliderWithEdit(
            minimum=0.0, maximum=0.30, step=0.001, initial=0.0, suffix=""
        )
        self.wav_sigma_slider.setToolTip(
            self.tr("Noise sigma. 0.0 = auto-estimate (BayesShrink).")
        )
        wav_f.addRow(self.tr("Noise σ (0=auto):"), self.wav_sigma_slider)
        self.wav_wavelet_combo = QComboBox()
        self.wav_wavelet_combo.addItems(
            ["db1", "db2", "db4", "db8", "sym4", "sym8", "coif1", "bior1.3"]
        )
        self.wav_wavelet_combo.setCurrentText("db2")
        self.wav_wavelet_combo.setToolTip(
            self.tr("Wavelet family. db2–4 = good general purpose.\n"
            "sym4/sym8 = near-symmetric, good for nebulosity.")
        )
        wav_f.addRow(self.tr("Wavelet:"), self.wav_wavelet_combo)
        self.wav_mode_combo = QComboBox()
        self.wav_mode_combo.addItems(["soft", "hard"])
        self.wav_mode_combo.setToolTip(
            self.tr("soft = smoother.  hard = sharper edges, may ring.")
        )
        wav_f.addRow(self.tr("Threshold mode:"), self.wav_mode_combo)
        wav_hint = QLabel("  ⓘ  skimage.restoration.denoise_wavelet (BayesShrink).")
        wav_hint.setStyleSheet("color: #888; font-size: 10px;")
        wav_f.addRow("", wav_hint)
        self._denoise_param_stack["Wavelet"] = wav_w
    
        # ── Stack container ──────────────────────────────────────
        self._denoise_stack_container = QWidget()
        stack_layout = QVBoxLayout(self._denoise_stack_container)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        for key, w in self._denoise_param_stack.items():
            w.setVisible(key == "TV Chambolle")
            stack_layout.addWidget(w)
        outer.addWidget(self._denoise_stack_container)
    
        # ── Status label + cancel button (shown while running) ───
        status_row = QHBoxLayout()
        self.denoise_status_label = QLabel("")
        self.denoise_status_label.setStyleSheet(
            "color: #7ecfea; font-size: 10px; padding: 1px 2px;"
        )
        status_row.addWidget(self.denoise_status_label, stretch=1)
    
        self.denoise_cancel_btn = QPushButton(self.tr("Cancel"))
        self.denoise_cancel_btn.setFixedWidth(60)
        self.denoise_cancel_btn.setVisible(False)
        self.denoise_cancel_btn.setStyleSheet(
            "QPushButton { background: rgba(200,80,0,0.25); border: 1px solid rgba(200,80,0,0.5);"
            " border-radius: 3px; padding: 2px 6px; font-size: 10px; }"
            "QPushButton:hover { background: rgba(200,80,0,0.45); }"
        )
        self.denoise_cancel_btn.clicked.connect(self._cancel_jobs)
        status_row.addWidget(self.denoise_cancel_btn)
        outer.addLayout(status_row)
    
        outer.addStretch()
    
        # Wire combo → show/hide panels
        self.denoise_algo_combo.currentTextChanged.connect(self._on_denoise_algo_changed)
    
        self.tabs.addTab(tvd_tab, self.tr("Classical Denoise"))

    
    
    def _on_denoise_algo_changed(self, algo: str):
        """Show only the selected denoiser's parameter panel."""
        for key, w in self._denoise_param_stack.items():
            w.setVisible(key == algo)


    def _denoise_params(self) -> dict:
        """Classical Denoise settings as plain values (read on the GUI thread)."""
        return {
            "algo": self.denoise_algo_combo.currentText(),
            "lum_only": bool(self.denoise_lum_only_chk.isChecked()),
            "tv_weight": float(self.tv_weight_slider.value()),
            "tv_iter": int(round(float(self.tv_iter_slider.value()))),
            "nlm_h": float(self.nlm_h_slider.value()),
            "nlm_patch": int(round(float(self.nlm_patch_slider.value()))),
            "nlm_dist": int(round(float(self.nlm_dist_slider.value()))),
            "bil_d": int(round(float(self.bil_d_slider.value()))),
            "bil_sigma_color": float(self.bil_sigma_color_slider.value()),
            "bil_sigma_space": float(self.bil_sigma_space_slider.value()),
            "gau_sigma": float(self.gau_sigma_slider.value()),
            "med_size": int(round(float(self.med_size_slider.value()))),
            "wav_sigma": float(self.wav_sigma_slider.value()),
            "wav_wavelet": self.wav_wavelet_combo.currentText(),
            "wav_mode": self.wav_mode_combo.currentText(),
        }

    def _apply_classical_denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Dispatch to the selected classical denoiser.
        Handles L*-only mode for color images uniformly.
        Returns float32 [0,1].
        """
        return _classical_denoise(img, self._denoise_params())

    def _denoise_single_channel(self, algo: str, ch: np.ndarray) -> np.ndarray:
        """
        Run the selected algorithm on a single 2-D float32 [0,1] channel.
        Returns float32 [0,1].
        """
        return _denoise_channel(algo, ch, self._denoise_params())

    # ---------------- UI reactions ----------------
    def _on_deconv_algo_changed(self, selected: str):
        for w in self.deconv_param_stack.values():
            w.setVisible(False)
        if selected in self.deconv_param_stack:
            self.deconv_param_stack[selected].setVisible(True)

        # Show/hide PSF sliders & bar
        on_psf_algo = selected in ("Richardson-Lucy", "Wiener")
        self.psf_param_group.setVisible(on_psf_algo)
        self.custom_psf_bar.setVisible(on_psf_algo and self._use_custom_psf and (self._custom_psf is not None))

    def _on_ls_operator_changed(self, op_text: str):
        self.ls_blend_combo.setCurrentText("SoftLight" if op_text == "Divide" else "Screen")

    def _make_psf_pixmap(self, radius, kurtosis, aspect, rotation_deg) -> QPixmap:
        psf = make_elliptical_gaussian_psf(radius, kurtosis, aspect, rotation_deg)
        h, w = psf.shape
        img8 = ((psf / psf.max()) * 255.0).astype(np.uint8) if psf.max() > 0 else psf.astype(np.uint8)
        qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
        scaled = QPixmap.fromImage(qimg).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        final = QPixmap(64, 64); final.fill(Qt.GlobalColor.transparent)
        p = QPainter(final); p.drawPixmap((64 - scaled.width()) // 2, (64 - scaled.height()) // 2, scaled); p.end()
        return final

    def _make_stellar_psf_pixmap(self, psf_kernel: np.ndarray) -> QPixmap:
        h, w = psf_kernel.shape
        img8 = ((psf_kernel / max(psf_kernel.max(), 1e-12)) * 255.0).astype(np.uint8)
        qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
        scaled = QPixmap.fromImage(qimg).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        final = QPixmap(64, 64); final.fill(Qt.GlobalColor.transparent)
        p = QPainter(final); p.drawPixmap((64 - scaled.width()) // 2, (64 - scaled.height()) // 2, scaled); p.end()
        return final

    def _update_psf_preview(self):
        current_tab = self.tabs.tabText(self.tabs.currentIndex())
        algo = getattr(self, "deconv_algo_combo", None)
        algo_text = algo.currentText() if algo is not None else ""

        if current_tab in (self.tr("Convolution"), "Convolution"):
            r, k, a, rot = (self.conv_radius_slider.value(), self.conv_shape_slider.value(),
                            self.conv_aspect_slider.value(), self.conv_rotation_slider.value())
            self.conv_psf_label.setPixmap(self._make_psf_pixmap(r, k, a, rot))
        elif current_tab in (self.tr("Deconvolution"), "Deconvolution") and algo_text in ("Richardson-Lucy", "Wiener"):
            if self._use_custom_psf and (self._custom_psf is not None):
                self.conv_psf_label.setPixmap(self._make_stellar_psf_pixmap(self._custom_psf))
            else:
                r, k, a, rot = (self.rl_psf_radius_slider.value(), self.rl_psf_shape_slider.value(),
                                self.rl_psf_aspect_slider.value(), self.rl_psf_rotation_slider.value())
                self.conv_psf_label.setPixmap(self._make_psf_pixmap(r, k, a, rot))
        else:
            self.conv_psf_label.clear()

    # ---------------- Mask helper (from active document) ----------------
    def _active_mask_array_from_active_doc(self) -> np.ndarray | None:
        """
        Read the active mask from the active document:
        doc.active_mask_id -> doc.masks[mid].data
        Return a 2-D float32 mask in [0..1], or None.
        """
        try:
            doc = self._active_doc()
            if doc is None:
                return None
            mid = getattr(doc, "active_mask_id", None)
            if not mid:
                return None
            masks = getattr(doc, "masks", {}) or {}
            layer = masks.get(mid)
            data = getattr(layer, "data", None) if layer is not None else None
            if data is None:
                return None

            return _mask_float2d(data)
        except Exception:
            return None


    def _resize_mask_nearest(self, mask2d: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """Resize 2-D mask to (H, W) using nearest neighbor."""
        H, W = target_hw
        if mask2d.shape == (H, W):
            return mask2d
        if cv2 is not None:
            return cv2.resize(mask2d, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
        # NumPy fallback NN
        yi = (np.linspace(0, mask2d.shape[0] - 1, H)).astype(np.int32)
        xi = (np.linspace(0, mask2d.shape[1] - 1, W)).astype(np.int32)
        return mask2d[yi][:, xi].astype(np.float32, copy=False)


    def _get_active_mask_from_doc(self, target_shape) -> np.ndarray | None:
        """
        Return mask resized to `target_shape`; broadcast to channels if needed.
        """
        m = self._active_mask_array_from_active_doc()
        if m is None:
            return None

        H, W = target_shape[:2]
        m = self._resize_mask_nearest(m, (H, W))

        # If the processed image is RGB, expand mask to 3 channels
        if len(target_shape) == 3 and m.ndim == 2:
            m = np.repeat(m[:, :, None], target_shape[2], axis=2)

        return np.clip(m.astype(np.float32, copy=False), 0.0, 1.0)

    # ---------------- Core actions ----------------
    # Preview and Push run the tool on a worker thread (_PreviewWorker), one job
    # at a time. A Preview click while a job runs is never dropped: the job is
    # stopped if it was a preview for other settings, and a fresh one starts
    # from the settings current when it ends. Push without a preview for the
    # current settings computes it first, through the same steps, then pushes.
    def _on_preview(self):
        spec = self._build_job_spec()
        if spec is None:
            return
        job = self._job
        if job is None:
            self._start_job(spec)
            return
        if not job["canceled"] and spec.same_result(job["spec"]):
            return                          # already computing exactly this
        if self._push_spec is None:
            self._stop_job(job)             # superseded preview: stop it, drop its result
        self._queued_preview = True         # rerun with the settings current when it ends
        self._show_busy()

    def _active_mask_ref(self):
        """(active mask id, its data array) of the active doc, or (None, None)."""
        try:
            doc = self._active_doc()
            mid = getattr(doc, "active_mask_id", None) if doc is not None else None
            if not mid:
                return None, None
            layer = (getattr(doc, "masks", {}) or {}).get(mid)
            data = getattr(layer, "data", None) if layer is not None else None
            return (mid, data) if data is not None else (None, None)
        except Exception:
            return None, None

    def _build_job_spec(self):
        """Snapshot everything a preview of the current tab needs, on the GUI thread.

        Returns None, after telling the user why, when there is nothing to run.
        The compute and Strength steps are the dialog's Preview steps moved into
        functions the worker can run; their results are bit-identical.
        """
        doc = self._active_doc()
        if hasattr(self.doc_manager, "set_active_document"):
            self.doc_manager.set_active_document(doc)
        img, _ = self._get_active_image_and_meta()
        if img is None:
            self._show_message(self.tr("No active image to process."))
            return None

        if self._original_image is None:
            self._original_image = img.copy()
        orig = self._original_image

        current_tab_name = self.tabs.tabText(self.tabs.currentIndex())
        s = self.strength_slider.value()
        is_rgb = img.ndim == 3 and img.shape[2] == 3

        spec = _JobSpec()
        spec.doc, spec.img, spec.orig, spec.tab, spec.strength = doc, img, orig, current_tab_name, s
        spec.denoise = False
        spec.label = self.tr("Computing preview…")
        spec.blend = lambda raw: raw * s + (1 - s) * img

        if current_tab_name in (self.tr("Convolution"), "Convolution"):
            radius  = self.conv_radius_slider.value()
            kurtosis= self.conv_shape_slider.value()
            aspect  = self.conv_aspect_slider.value()
            rotation= self.conv_rotation_slider.value()
            psf_kernel = make_elliptical_gaussian_psf(radius, kurtosis, aspect, rotation).astype(np.float32)
            spec.raw_key = ("conv", radius, kurtosis, aspect, rotation)

            def compute_raw(report, canceled):
                return _convolve_image(img, psf_kernel)

        elif current_tab_name in (self.tr("Deconvolution"), "Deconvolution"):
            algo = self.deconv_algo_combo.currentText()
            if algo == "Richardson-Lucy":
                iters = int(round(self.rl_iterations_slider.value()))
                reg_type = self.rl_reg_combo.currentText()
                # "Use as Current PSF" selects RL for the stellar PSF, so RL
                # must use it (it used to fall through to the slider model).
                if self._use_custom_psf and (self._custom_psf is not None):
                    psf_kernel = self._custom_psf.astype(np.float32)
                    psf_key = ("stellar", psf_kernel.shape, psf_kernel.tobytes())
                else:
                    pr, ps, pa, pt = (self.rl_psf_radius_slider.value(), self.rl_psf_shape_slider.value(),
                                      self.rl_psf_aspect_slider.value(), self.rl_psf_rotation_slider.value())
                    psf_kernel = make_elliptical_gaussian_psf(pr, ps, pa, pt).astype(np.float32)
                    psf_key = ("model", pr, ps, pa, pt)
                clip_flag = self.rl_clip_checkbox.isChecked()
                lum_only = self.rl_luminance_only_checkbox.isChecked() and is_rgb
                spec.raw_key = ("rl", iters, reg_type, psf_key, clip_flag, lum_only)

                def compute_raw(report, canceled):
                    if lum_only:
                        lab = _rgb2lab_rows(img.astype(np.float32, copy=False))
                        L = (lab[:, :, 0] / 100.0).astype(np.float32)
                        deconv_L = _rl_deconvolve(L, psf_kernel, iters, reg_type, clip_flag,
                                                  progress=report, cancel=canceled)
                        lab[:, :, 0] = np.clip(deconv_L * 100.0, 0.0, 100.0)
                        rgb_deconv = _lab2rgb_rows(lab.astype(np.float32, copy=False))
                        return np.clip(rgb_deconv.astype(np.float32, copy=False), 0.0, 1.0)
                    return _rl_deconvolve(img.astype(np.float32, copy=False), psf_kernel, iters, reg_type,
                                          clip_flag, progress=report, cancel=canceled)

            elif algo == "Wiener":
                if self._use_custom_psf and (self._custom_psf is not None):
                    small_psf = self._custom_psf.astype(np.float32)
                    psf_key = ("stellar", small_psf.shape, small_psf.tobytes())
                else:
                    pr, ps, pa, pt = (self.rl_psf_radius_slider.value(), self.rl_psf_shape_slider.value(),
                                      self.rl_psf_aspect_slider.value(), self.rl_psf_rotation_slider.value())
                    small_psf = make_elliptical_gaussian_psf(pr, ps, pa, pt).astype(np.float32)
                    psf_key = ("model", pr, ps, pa, pt)

                nsr = self.wiener_nsr_slider.value()
                reg_type = "Wiener" if self.wiener_reg_combo.currentText() == "None (Classical Wiener)" else "Tikhonov"
                do_dering = self.wiener_dering_checkbox.isChecked()
                lum_only = self.wiener_luminance_only_checkbox.isChecked() and is_rgb
                spec.raw_key = ("wiener", psf_key, nsr, reg_type, do_dering, lum_only)

                def compute_raw(report, canceled):
                    if lum_only:
                        lab = _rgb2lab_rows(img.astype(np.float32, copy=False))
                        L = (lab[:, :, 0] / 100.0).astype(np.float32)
                        deconv_L = _wiener_deconvolve(L, small_psf, nsr, reg_type, do_dering, cancel=canceled)
                        lab[:, :, 0] = np.clip(deconv_L * 100.0, 0.0, 100.0)
                        rgb_deconv = _lab2rgb_rows(lab.astype(np.float32, copy=False))
                        return np.clip(rgb_deconv.astype(np.float32, copy=False), 0.0, 1.0)
                    processed = _wiener_deconvolve(img, small_psf, nsr, reg_type, do_dering, cancel=canceled)
                    return np.clip(processed, 0.0, 1.0)

            elif algo == "Larson-Sekanina":
                if not hasattr(self.view, "ls_center") or self.view.ls_center is None:
                    QMessageBox.information(self, self.tr("Hold Shift + Click"),
                        self.tr("To choose a Larson–Sekanina center, hold Shift and click on the preview."))
                    return None

                center = self.view.ls_center
                rstep  = self.ls_radial_slider.value()
                astep  = self.ls_angular_slider.value()
                operator = self.ls_operator_combo.currentText()
                blend_mode = self.ls_blend_combo.currentText()
                spec.raw_key = ("ls", tuple(center), rstep, astep, operator, blend_mode)

                def compute_raw(report, canceled):
                    B = larson_sekanina(image=img, center=center, radial_step=rstep, angular_step_deg=astep, operator=operator)
                    A = img
                    if A.ndim == 3 and A.shape[2] == 3:
                        # B broadcasts over the channels: the values of repeating it 3x, without the copy
                        B_rgb, A_rgb = B[:, :, None], A
                    else:
                        B_rgb, A_rgb = B[..., None], A[..., None]
                    C = (A_rgb + B_rgb - (A_rgb * B_rgb)) if blend_mode == "Screen" else ((1 - 2 * B_rgb) * (A_rgb**2) + 2 * B_rgb * A_rgb)
                    processed = np.clip(C, 0.0, 1.0)
                    return processed[..., 0] if img.ndim == 2 else processed

            elif algo == "Van Cittert":
                # The slider value is a float and range() needs an int (passing
                # the float raised TypeError, so this preview never worked).
                iters2 = int(round(self.vc_iterations_slider.value()))
                relax  = self.vc_relax_slider.value()
                spec.raw_key = ("vc", iters2, relax)

                def compute_raw(report, canceled):
                    if is_rgb:
                        chans = _run_parallel(
                            lambda c: van_cittert_deconv(img[:, :, c], iters2, relax, cancel=canceled),
                            range(3), workers=3)
                        processed = np.stack(chans, axis=2)
                    else:
                        processed = van_cittert_deconv(img, iters2, relax, cancel=canceled)
                    return np.clip(processed, 0.0, 1.0)

            else:
                self._show_message(self.tr("Unknown deconvolution algorithm"))
                return None

        elif current_tab_name in (self.tr("Classical Denoise"), "Classical Denoise"):
            p = self._denoise_params()
            spec.denoise = True
            spec.label = self.tr("Running {0}…").format(p['algo'])
            spec.raw_key = ("denoise", p["algo"], p["lum_only"] and is_rgb) + tuple(
                p[k] for k in _DENOISE_PARAM_KEYS.get(p["algo"], ()))
            # The denoise path has always blended against the dialog's copy of the image.
            spec.blend = lambda raw: np.clip(raw, 0.0, 1.0) * s + (1.0 - s) * orig

            def compute_raw(report, canceled):
                return _classical_denoise(img, p)

        else:
            self._show_message(self.tr("Unknown tab"))
            return None

        spec.compute_raw = compute_raw
        spec.mask_id, spec.mask_data = self._active_mask_ref()
        spec.preset = self._build_replay_preset()
        return spec

    def _build_job(self, spec, for_push: bool = False):
        """The function the worker runs for `spec`. It closes over arrays and plain
        values only (no widgets, no dialog), so it is safe on another thread."""
        c = self._raw_cache
        raw_hit = c[1] if (c is not None and spec.same_raw(c[0])) else None
        if not for_push:
            disp = self._display_snapshot(spec.orig)
        elif not self.cb_autostretch.isChecked():
            # After the push the pane shows these pixels unstretched as the new
            # baseline (_commit_push), so only that picture is worth making here.
            disp = {"auto": False, "tm": float(self.s_target_median.value()),
                    "linked": bool(self.cb_linked.isChecked()), "params": None, "need_before": False}
        else:
            disp = None     # stretched: the new baseline needs its own stretch, made after the push
        compute_raw, blend, orig, mask_data = spec.compute_raw, spec.blend, spec.orig, spec.mask_data

        def job(report, canceled):
            raw = raw_hit if raw_hit is not None else compute_raw(report, canceled)
            if canceled():
                raise _Canceled()
            processed = blend(raw)
            final = _apply_mask_blend(processed, orig, _mask_or_none(mask_data))
            del processed
            display = _display_images(final, orig, disp) if disp is not None else None
            return {"raw": raw, "final": final, "display": display}
        return job

    def _display_snapshot(self, orig):
        """The display state a new result will be drawn with, read on the GUI thread."""
        auto = bool(self.cb_autostretch.isChecked())
        tm = float(self.s_target_median.value())
        linked = bool(self.cb_linked.isChecked())
        params = None
        c = self._stretch_cache
        if auto and c is not None and c[0] is orig and c[1] == tm and c[2] == linked:
            params = c[3]
        key = (tm, linked) if auto else None
        pc = self._pm_before_cache
        need_before = pc is None or pc[0] is not orig or pc[1] != key
        return {"auto": auto, "tm": tm, "linked": linked, "params": params, "need_before": need_before}

    def _start_job(self, spec):
        self._job_seq += 1
        for_push = self._push_spec is not None and spec.same_result(self._push_spec)
        worker = _PreviewWorker(self._job_seq, self._build_job(spec, for_push=for_push))
        worker.progress.connect(self._on_job_progress)
        worker.result.connect(self._on_job_result)
        self._job = {"id": self._job_seq, "spec": spec, "worker": worker, "canceled": False,
                     "for_push": for_push}
        if spec.denoise:
            self.denoise_status_label.setText(spec.label)
        self._show_busy()
        start_owned_thread(worker)

    def _on_job_progress(self, job_id: int, text: str):
        job = self._job
        if job is None or job_id != job["id"] or job["canceled"]:
            return
        self.rl_status_label.setText(text or self._busy_text())

    def _on_job_result(self, job_id: int, payload, err: str):
        job = self._job
        if job is None or job_id != job["id"]:
            return                                  # superseded, or the dialog let go of it
        self._job = None
        spec = job["spec"]
        if job["canceled"]:
            payload, err = None, "__canceled__"
        ok = (not err) and payload is not None

        if ok:
            self._raw_cache = (spec, payload["raw"])
            self._preview_result = payload["final"]
            self._preview_spec = spec
            # A push job's picture is shown by _commit_push, as the new baseline.
            if payload.get("display") is not None and not job["for_push"]:
                self._refresh_display(prebuilt=payload["display"])
            if spec.denoise:
                self.denoise_status_label.setText(self.tr("Done."))
        elif err == "__canceled__":
            if spec.denoise:
                self.denoise_status_label.setText(self.tr("Canceled."))
        else:
            if spec.denoise:
                self.denoise_status_label.setText(self.tr("Error."))
            if self._push_spec is not None and self._push_spec.same_result(spec):
                self._push_spec = None
            logging.getLogger(__name__).error("Convo/Deconvo preview failed:\n%s", err)
            first = err.splitlines()[0] if err else self.tr("Unknown error")
            if spec.denoise:
                msg = self.tr("Denoising failed:\n\n{0}\n\nCheck the SASpro log for details.").format(first)
            else:
                msg = self.tr("The preview failed:\n\n{0}\n\nCheck the SASpro log for details.").format(first)
            QMessageBox.critical(
                self, self.tr("Denoise Error") if spec.denoise else self.tr("Preview Error"),
                msg
            )

        push = self._push_spec
        if push is not None:
            if ok and push.same_result(spec):
                self._push_spec = None
                self._commit_push(payload["final"], push,
                                  prebuilt=payload.get("display") if job["for_push"] else None)
            else:
                self._start_job(push)               # the push's own job, after the one it displaced
                return
        elif ok and job["for_push"]:
            self._refresh_display(prebuilt=payload.get("display"))   # its push went away: show it
        if self._queued_preview:
            self._queued_preview = False
            nxt = self._build_job_spec()
            if nxt is not None and not (self._preview_result is not None
                                        and nxt.same_result(self._preview_spec)):
                self._start_job(nxt)
                return
        self._show_idle()

    def _stop_job(self, job):
        job["canceled"] = True
        job["worker"].cancel()

    def _cancel_jobs(self, *_):
        """Cancel buttons / Undo: stop the running job and drop anything queued."""
        self._push_spec = None
        self._queued_preview = False
        job = self._job
        if job is not None and not job["canceled"]:
            self._stop_job(job)
            self.rl_status_label.setText(self.tr("Canceling…"))
            if job["spec"].denoise:
                self.denoise_status_label.setText(self.tr("Canceling…"))
        self.push_btn.setEnabled(True)

    def _teardown_jobs(self, *_):
        """Dialog closing, or its image changing: cancel and forget all work.
        The thread stays with thread_keeper until it returns."""
        self._push_spec = None
        self._queued_preview = False
        job, self._job = self._job, None
        if job is not None:
            w = job["worker"]
            for sig, slot in ((w.result, self._on_job_result), (w.progress, self._on_job_progress)):
                try:
                    sig.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
            w.cancel()
        self._raw_cache = None
        self._preview_spec = None
        try:
            self._show_idle()
        except RuntimeError:
            pass

    def _busy_text(self) -> str:
        if self._push_spec is not None:
            return self.tr("Computing, then pushing…")
        return self.tr("Computing preview…")

    def _show_busy(self):
        job = self._job
        self.rl_status_label.setText(self._busy_text())
        self.job_cancel_btn.setVisible(True)
        self.denoise_cancel_btn.setVisible(job is not None and job["spec"].denoise)
        self.push_btn.setEnabled(self._push_spec is None)
        self.setCursor(Qt.CursorShape.BusyCursor)

    def _show_idle(self):
        self.rl_status_label.setText("")
        self.job_cancel_btn.setVisible(False)
        self.denoise_cancel_btn.setVisible(False)
        self.push_btn.setEnabled(True)
        self.unsetCursor()
    


    def _on_undo(self):
        if self._original_image is not None:
            self._cancel_jobs()   # a preview still computing would land after the Undo
            self._preview_result = None
            self._preview_spec = None
            self._refresh_display()
        else:
            self._show_message(self.tr("Nothing to undo."))

    def _snapshot_reset_defaults(self):
        """Record each tab's controls as built, so Reset restores exactly those.

        Groups follow tab order (Convolution, Deconvolution, PSF Estimator,
        Classical Denoise). Combos lead each group because they drive which
        parameter panel is visible, and the LS operator precedes the LS blend
        because changing the operator rewrites the blend.
        """
        groups = (
            (self.conv_radius_slider, self.conv_shape_slider,
             self.conv_aspect_slider, self.conv_rotation_slider),
            (self.deconv_algo_combo, self.rl_reg_combo, self.wiener_reg_combo,
             self.ls_operator_combo, self.ls_blend_combo,
             self.rl_psf_radius_slider, self.rl_psf_shape_slider,
             self.rl_psf_aspect_slider, self.rl_psf_rotation_slider,
             self.rl_iterations_slider, self.rl_clip_checkbox,
             self.rl_luminance_only_checkbox,
             self.wiener_nsr_slider, self.wiener_luminance_only_checkbox,
             self.wiener_dering_checkbox,
             self.ls_radial_slider, self.ls_angular_slider,
             self.vc_iterations_slider, self.vc_relax_slider),
            (self.sep_auto_checkbox, self.sep_threshold_slider, self.sep_minarea_spin,
             self.sep_sat_slider, self.sep_maxstars_spin, self.sep_stamp_spin),
            (self.denoise_algo_combo, self.wav_wavelet_combo, self.wav_mode_combo,
             self.denoise_lum_only_chk,
             self.tv_weight_slider, self.tv_iter_slider, self.tv_multichannel_checkbox,
             self.nlm_h_slider, self.nlm_patch_slider, self.nlm_dist_slider,
             self.bil_d_slider, self.bil_sigma_color_slider, self.bil_sigma_space_slider,
             self.gau_sigma_slider, self.med_size_slider, self.wav_sigma_slider),
        )
        self._reset_defaults = [[(w, _control_value(w)) for w in g] for g in groups]
        self._strength_default = float(self.strength_slider.value())

    def _on_reset(self):
        """Restore the current tab's settings and Strength to their defaults.

        Settings only: a preview on screen stays until Undo or the next Preview.
        """
        idx = self.tabs.currentIndex()
        if not (0 <= idx < len(self._reset_defaults)):
            return
        for w, v in self._reset_defaults[idx]:
            _set_control_value(w, v)
        self.strength_slider.setValue(self._strength_default)
        if idx == 1:
            self._clear_custom_psf_flag()   # default PSF is the slider-built one
        self._update_psf_preview()

    def _build_replay_preset(self) -> dict | None:
        """
        Capture the current UI state as a preset-style dict so Replay Last Action
        can re-run the same Convo/Deconvo/TV operation on another document.
        Matches the schema used by ConvoPresetDialog.result_dict().
        """
        current_tab_name = self.tabs.tabText(self.tabs.currentIndex())
        strength = float(self.strength_slider.value())

        # ── Convolution tab ─────────────────────────────────────────────
        if current_tab_name in (self.tr("Convolution"), "Convolution"):
            return {
                "op": "convolution",
                "radius":   float(self.conv_radius_slider.value()),
                "kurtosis": float(self.conv_shape_slider.value()),
                "aspect":   float(self.conv_aspect_slider.value()),
                "rotation": float(self.conv_rotation_slider.value()),
                "strength": strength,
            }

        # ── Deconvolution tab ───────────────────────────────────────────
        if current_tab_name in (self.tr("Deconvolution"), "Deconvolution"):
            algo = self.deconv_algo_combo.currentText()
            p: dict[str, object] = {
                "op": "deconvolution",
                "algo": algo,
                # RL/Wiener PSF params
                "psf_radius":   float(self.rl_psf_radius_slider.value()),
                "psf_kurtosis": float(self.rl_psf_shape_slider.value()),
                "psf_aspect":   float(self.rl_psf_aspect_slider.value()),
                "psf_rotation": float(self.rl_psf_rotation_slider.value()),
                # RL options
                "rl_iter":   float(self.rl_iterations_slider.value()),
                "rl_reg":    self.rl_reg_combo.currentText(),
                "rl_dering": bool(self.rl_clip_checkbox.isChecked()),
                "luminance_only": self._deconv_lum_only(algo),
                # Wiener options
                "wiener_nsr":    float(self.wiener_nsr_slider.value()),
                "wiener_reg":    self.wiener_reg_combo.currentText(),
                "wiener_dering": bool(self.wiener_dering_checkbox.isChecked()),
                # Larson–Sekanina options
                "ls_rstep":    float(self.ls_radial_slider.value()),
                "ls_astep":    float(self.ls_angular_slider.value()),
                "ls_operator": self.ls_operator_combo.currentText(),
                "ls_blend":    self.ls_blend_combo.currentText(),
                # Van Cittert options
                "vc_iter":  float(self.vc_iterations_slider.value()),
                "vc_relax": float(self.vc_relax_slider.value()),
                # Global blend strength
                "strength": strength,
            }

            # If user actually picked an LS center, preserve it for replay.
            # Interactive view stores (x,y). apply_convo_via_preset expects [cx, cy].
            if hasattr(self.view, "ls_center") and self.view.ls_center is not None:
                cx, cy = self.view.ls_center  # (x, y)
                p["center"] = [float(cx), float(cy)]

            p.update(self._stellar_psf_preset(algo))
            return p

        # ── Classical Denoise tab ───────────────────────────────────────
        if current_tab_name == self.tr("Classical Denoise") or current_tab_name == "Classical Denoise":
            # Reuse the grip emitter so Replay and the shortcut stay identical.
            return self.get_preset()

        return None

    def _deconv_lum_only(self, algo: str) -> bool:
        """A preset has one luminance_only key: the running algorithm's L* box.
        (It used to be RL's box even for Wiener.)"""
        box = (self.wiener_luminance_only_checkbox if algo == "Wiener"
               else self.rl_luminance_only_checkbox)
        return bool(box.isChecked())

    def _stellar_psf_preset(self, algo: str) -> dict:
        """{"psf_kernel": [[...]]} when RL/Wiener runs on the stellar PSF, so
        Replay Last Action, the grip and History replay deconvolve with the same
        kernel (float32 -> list -> float32 is exact); {} for the slider PSF."""
        if (algo in ("Richardson-Lucy", "Wiener") and self._use_custom_psf
                and self._custom_psf is not None):
            return {"psf_kernel": np.asarray(self._custom_psf, dtype=np.float32).tolist()}
        return {}

    # -------- preset emit (grip) --------
    def get_preset(self) -> dict | None:
        """Emit current UI state as a Convo/Deconvo preset.
        Does NOT rely on _build_replay_preset's tab-name match (which is stale for
        the denoise tab); maps the real tab text to op directly so the tv grip works."""
        tab = self.tabs.tabText(self.tabs.currentIndex())
        strength = float(self.strength_slider.value())

        if tab == self.tr("Convolution") or tab == "Convolution":
            return {
                "op": "convolution",
                "radius":   float(self.conv_radius_slider.value()),
                "kurtosis": float(self.conv_shape_slider.value()),
                "aspect":   float(self.conv_aspect_slider.value()),
                "rotation": float(self.conv_rotation_slider.value()),
                "strength": strength,
            }

        if tab == self.tr("Deconvolution") or tab == "Deconvolution":
            algo = self.deconv_algo_combo.currentText()
            p = {
                "op": "deconvolution",
                "algo": algo,
                "psf_radius":   float(self.rl_psf_radius_slider.value()),
                "psf_kurtosis": float(self.rl_psf_shape_slider.value()),
                "psf_aspect":   float(self.rl_psf_aspect_slider.value()),
                "psf_rotation": float(self.rl_psf_rotation_slider.value()),
                "rl_iter":   float(self.rl_iterations_slider.value()),
                "rl_reg":    self.rl_reg_combo.currentText(),
                "rl_dering": bool(self.rl_clip_checkbox.isChecked()),
                "luminance_only": self._deconv_lum_only(algo),
                "wiener_nsr":    float(self.wiener_nsr_slider.value()),
                "wiener_reg":    self.wiener_reg_combo.currentText(),
                "wiener_dering": bool(self.wiener_dering_checkbox.isChecked()),
                "ls_rstep":    float(self.ls_radial_slider.value()),
                "ls_astep":    float(self.ls_angular_slider.value()),
                "ls_operator": self.ls_operator_combo.currentText(),
                "ls_blend":    self.ls_blend_combo.currentText(),
                "vc_iter":  float(self.vc_iterations_slider.value()),
                "vc_relax": float(self.vc_relax_slider.value()),
                "strength": strength,
            }
            if getattr(self.view, "ls_center", None) is not None:
                cx, cy = self.view.ls_center  # (x, y)
                p["center"] = [float(cx), float(cy)]
            p.update(self._stellar_psf_preset(algo))
            return p

        # Classical Denoise tab -> op:"denoise", carries the selected method + its params.
        algo = self.denoise_algo_combo.currentText()
        p = {
            "op": "denoise",
            "denoise_algo": algo,
            "lum_only": bool(self.denoise_lum_only_chk.isChecked()),
            "strength": strength,
        }
        if algo == "TV Chambolle":
            p["tv_weight"]       = float(self.tv_weight_slider.value())
            p["tv_iter"]         = int(round(float(self.tv_iter_slider.value())))
            p["tv_multichannel"] = bool(self.tv_multichannel_checkbox.isChecked())
        elif algo == "Non-Local Means":
            p["nlm_h"]     = float(self.nlm_h_slider.value())
            p["nlm_patch"] = int(round(float(self.nlm_patch_slider.value())))
            p["nlm_dist"]  = int(round(float(self.nlm_dist_slider.value())))
        elif algo == "Bilateral":
            p["bil_d"]           = int(round(float(self.bil_d_slider.value())))
            p["bil_sigma_color"] = float(self.bil_sigma_color_slider.value())
            p["bil_sigma_space"] = float(self.bil_sigma_space_slider.value())
        elif algo == "Gaussian":
            p["gau_sigma"] = float(self.gau_sigma_slider.value())
        elif algo == "Median":
            p["med_size"] = int(round(float(self.med_size_slider.value())))
        elif algo == "Wavelet":
            p["wav_sigma"]   = float(self.wav_sigma_slider.value())
            p["wav_wavelet"] = self.wav_wavelet_combo.currentText()
            p["wav_mode"]    = self.wav_mode_combo.currentText()
        return p

    # -------- preset seed (double-click open) --------
    def seed_from_preset(self, p: dict | None):
        """Inverse of get_preset. FloatSliderWithEdit stores true floats -> no divisors.
        LS center is deferred to _seed_ls_center (needs scene/fit to exist first)."""
        p = dict(p or {})
        op = str(p.get("op", "convolution")).lower()

        def _sv(widget, key, cast=float):
            if key in p and p[key] is not None:
                try:
                    widget.setValue(cast(p[key]))
                except Exception:
                    pass

        def _combo(widget, key):
            if p.get(key) is not None:
                try:
                    widget.setCurrentText(str(p[key]))
                except Exception:
                    pass

        def _chk(widget, key):
            if key in p and p[key] is not None:
                try:
                    widget.setChecked(bool(p[key]))
                except Exception:
                    pass

        if op == "convolution":
            _sv(self.conv_radius_slider,   "radius")
            _sv(self.conv_shape_slider,    "kurtosis")
            _sv(self.conv_aspect_slider,   "aspect")
            _sv(self.conv_rotation_slider, "rotation")
            _sv(self.strength_slider,      "strength")
            self._select_tab(self.tr("Convolution"), "Convolution")

        elif op == "deconvolution":
            _combo(self.deconv_algo_combo, "algo")
            _sv(self.rl_psf_radius_slider,   "psf_radius")
            _sv(self.rl_psf_shape_slider,    "psf_kurtosis")
            _sv(self.rl_psf_aspect_slider,   "psf_aspect")
            _sv(self.rl_psf_rotation_slider, "psf_rotation")
            _sv(self.rl_iterations_slider,   "rl_iter")
            _combo(self.rl_reg_combo,        "rl_reg")
            _chk(self.rl_clip_checkbox,      "rl_dering")
            _chk(self.rl_luminance_only_checkbox, "luminance_only")
            _chk(self.wiener_luminance_only_checkbox, "luminance_only")
            _sv(self.wiener_nsr_slider,      "wiener_nsr")
            _combo(self.wiener_reg_combo,    "wiener_reg")
            _chk(self.wiener_dering_checkbox,"wiener_dering")
            _sv(self.ls_radial_slider,       "ls_rstep")
            _sv(self.ls_angular_slider,      "ls_astep")
            _combo(self.ls_operator_combo,   "ls_operator")
            _combo(self.ls_blend_combo,      "ls_blend")
            _sv(self.vc_iterations_slider,   "vc_iter")
            _sv(self.vc_relax_slider,        "vc_relax")
            _sv(self.strength_slider,        "strength")
            # After the PSF sliders (moving one drops a stellar PSF): the preset's
            # stellar PSF becomes the active one; a preset without one leaves the
            # slider PSF in charge, as it will be when the preset is applied.
            try:
                kern = preset_psf_kernel(p)
            except ValueError as e:
                kern = None
                QMessageBox.warning(self, self.tr("Stellar PSF"), self.tr("{0}\nThe slider PSF is used instead.").format(e))
            if kern is not None:
                self._activate_custom_psf(kern)
            else:
                self._clear_custom_psf_flag()
            # Drive algo-dependent panel/PSF-group visibility.
            try:
                self._on_deconv_algo_changed(self.deconv_algo_combo.currentText())
            except Exception:
                pass
            # Stash LS center for after-show (crosshair needs the scene fitted).
            c = p.get("center")
            if c and len(c) == 2:
                self._pending_ls_center = (float(c[0]), float(c[1]))
            self._select_tab(self.tr("Deconvolution"), "Deconvolution")
            self._update_psf_preview()

        else:  # denoise (or legacy "tv")
            # Legacy tv presets have no denoise_algo -> map to TV Chambolle.
            algo = str(p.get("denoise_algo", "TV Chambolle"))
            try:
                self.denoise_algo_combo.setCurrentText(algo)
            except Exception:
                pass
            # Show the right param panel for the seeded method.
            try:
                self._on_denoise_algo_changed(self.denoise_algo_combo.currentText())
            except Exception:
                pass
            _chk(self.denoise_lum_only_chk, "lum_only")
            _sv(self.strength_slider, "strength")

            if algo == "TV Chambolle":
                _sv(self.tv_weight_slider, "tv_weight")
                _sv(self.tv_iter_slider,   "tv_iter")
                _chk(self.tv_multichannel_checkbox, "tv_multichannel")
            elif algo == "Non-Local Means":
                _sv(self.nlm_h_slider,     "nlm_h")
                _sv(self.nlm_patch_slider, "nlm_patch")
                _sv(self.nlm_dist_slider,  "nlm_dist")
            elif algo == "Bilateral":
                _sv(self.bil_d_slider,           "bil_d")
                _sv(self.bil_sigma_color_slider, "bil_sigma_color")
                _sv(self.bil_sigma_space_slider, "bil_sigma_space")
            elif algo == "Gaussian":
                _sv(self.gau_sigma_slider, "gau_sigma")
            elif algo == "Median":
                _sv(self.med_size_slider, "med_size")
            elif algo == "Wavelet":
                _sv(self.wav_sigma_slider, "wav_sigma")
                _combo(self.wav_wavelet_combo, "wav_wavelet")
                _combo(self.wav_mode_combo,    "wav_mode")

            self._select_tab(self.tr("Classical Denoise"), "Classical Denoise")

    def _select_tab(self, *names):
        for i in range(self.tabs.count()):
            t = self.tabs.tabText(i)
            if t in names:
                self.tabs.setCurrentIndex(i)
                return

    def _seed_ls_center(self):
        """Apply a stashed LS center after the scene is built and fitted."""
        c = getattr(self, "_pending_ls_center", None)
        if not c:
            return
        self._pending_ls_center = None
        try:
            cx, cy = c
            self.view.ls_center = (cx, cy)
            if hasattr(self.view, "_draw_crosshair_at"):
                self.view._draw_crosshair_at(cx, cy)
        except Exception:
            pass


    def _on_push_to_doc(self):
        doc = self._active_doc()
        if doc is None:
            QMessageBox.warning(self, self.tr("No Document"), self.tr("No active document to push into."))
            return
        if self._push_spec is not None:
            return                                  # a push is already on its way

        spec = self._build_job_spec()
        if spec is None:
            return
        # The preview on screen was computed for these exact settings: push it.
        if self._preview_result is not None and spec.same_result(self._preview_spec):
            self._commit_push(self._preview_result, spec)
            return

        # No preview for these settings (none yet, or they changed since): compute
        # it on the worker through the same steps as Preview, then push that.
        self._push_spec = spec
        job = self._job
        if job is None:
            self._start_job(spec)
            return
        if not job["canceled"] and not spec.same_result(job["spec"]):
            self._stop_job(job)        # a preview for other settings; the push job follows it
        self._show_busy()              # (a running job for these settings is pushed when it lands)

    def _show_pushed_pixels(self, result, img_after, prebuilt) -> bool:
        """After a push, show the new baseline without rebuilding its picture when
        the document kept `result` itself and the pane is unstretched: the pixels,
        and so the 8-bit picture, are the ones already made. False = redraw."""
        if img_after is not result or self.cb_autostretch.isChecked():
            return False
        if prebuilt is not None and prebuilt.get("shown") is result and not prebuilt["auto"]:
            self.pixmap_item.setPixmap(QPixmap.fromImage(prebuilt["after"][0]))
        elif not (self._shown is result and self._shown_stretch_key is None):
            return False
        self._shown, self._shown_stretch_key = self._original_image, None
        self._pm_before_cache = None
        self._cmp.set_before(None)
        h, w = self._original_image.shape[:2]
        self.view.image_rect = QRectF(0, 0, w, h)
        self._cmp.set_image_size(w, h)
        return True

    def _commit_push(self, result, spec, prebuilt=None):
        """Apply `result` (computed for `spec`) to the document; HEAD's Push body.
        prebuilt: an unstretched picture of `result` made by its push job."""
        doc = self._active_doc()
        if doc is None:
            QMessageBox.warning(self, self.tr("No Document"), self.tr("No active document to push into."))
            return
        if doc is not spec.doc:
            QMessageBox.warning(self, self.tr("Not pushed"),
                                self.tr("The active image changed while the result was computing; nothing was pushed."))
            self._refresh_display()
            return

        # Grab current metadata from this specific doc
        _, meta = self._get_active_image_and_meta()
        new_meta = dict(meta)
        new_meta["source"] = "ConvoDeconvo"

        # Settings as they were when Preview/Push was clicked, i.e. what made `result`.
        _cv_preset = spec.preset
        if _cv_preset:
            new_meta["command_id"] = "convo"
            new_meta["preset"] = dict(_cv_preset)
        # else: no replayable op on the current tab — leave the marker off;
        # the extractor still labels it 'convo' by step_name, just preset-less.

        try:
            # `result` is this dialog's own array and is let go just below, so it
            # goes in without the defensive copy (one image-sized copy less).
            if hasattr(doc, "apply_edit"):
                # Preferred: update this exact Document (ROI or full) so all views update
                doc.apply_edit(
                    result,
                    metadata=new_meta,
                    step_name="Convo/Deconvo",
                )
            else:
                # Fallback for older paths: go through DocManager active-doc API
                if hasattr(self.doc_manager, "set_active_document"):
                    self.doc_manager.set_active_document(doc)
                self.doc_manager.update_active_document(
                    result,
                    metadata=new_meta,
                    step_name="Convo/Deconvo",
                )
        except Exception as e:
            QMessageBox.critical(self, self.tr("Push failed"), str(e))
            self._refresh_display()
            return

        # Make the pushed image the new baseline so you can iterate
        img_after, _ = self._get_active_image_and_meta()
        if img_after is not None:
            self._original_image = img_after.copy()
            self._preview_result = None
            self._preview_spec = None
            self._raw_cache = None
            if not self._show_pushed_pixels(result, img_after, prebuilt):
                self._refresh_display()

        # Replay wiring (unchanged, just moved under try/except)
        try:
            if self._main is not None:
                preset = spec.preset
                if preset:
                    self._main._last_headless_command = {
                        "cid": "convo",
                        "preset": dict(preset),
                    }
                    if hasattr(self._main, "_log"):
                        op = preset.get("op", "convolution")
                        self._main._log(f"Replay: stored Convo/Deconvo ({op}) from dialog.")
        except Exception:
            # Replay wiring should never break the actual push
            pass

        QMessageBox.information(self, self.tr("Pushed"), self.tr("Result committed to the active document."))



    # ---------------- Utils ----------------
    def _show_message(self, text: str):
        # Reuse pixmap_item rather than clearing the scene: a clear would also
        # delete the comparison overlay and the LS crosshair items the view
        # still holds references to.
        temp_label = QLabel(text); temp_label.setStyleSheet("color: white; background-color: #222;"); temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = temp_label.grab()
        self._cmp.set_before(None)
        self._shown = None
        self.pixmap_item.setPixmap(pixmap)
        self._cmp.set_image_size(pixmap.width(), pixmap.height())
        self.view.resetTransform()
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _convolve_color(self, image: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
        """
        Convolve image with psf_kernel using reflect padding so we don't get
        dark borders from zero–padding. Returns same H×W (and channels) as input.
        """
        return _convolve_image(image, psf_kernel)

    def _rl_status_gui(self, text: str):
        """R-L status on the GUI thread (headless callers): label + repaint."""
        self.rl_status_label.setText(text); QApplication.processEvents()

    def _richardson_lucy_color(self, image: np.ndarray, psf_kernel: np.ndarray, iterations: int,
                               reg_type: str = "None (Plain R–L)", clip_flag: bool = True) -> np.ndarray:
        return _rl_deconvolve(image, psf_kernel, iterations, reg_type, clip_flag,
                              progress=self._rl_status_gui)

    def _wiener_deconv_with_kernel(self, image: np.ndarray, small_psf: np.ndarray, nsr: float,
                                   reg_type: str, do_dering: bool) -> np.ndarray:
        return _wiener_deconvolve(image, small_psf, nsr, reg_type, do_dering)

    @staticmethod
    def _pixmap_from_array(array: np.ndarray) -> QPixmap:
        """8-bit display pixmap: mono as Grayscale8, colour as RGB888."""
        qimg, _buf = _qimage_from_array(array)
        return QPixmap.fromImage(qimg)   # fromImage copies, so the buffer can go

    def zoom_in(self):  self.view.scale(1.2, 1.2)
    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)

    def _on_fit_clicked(self):
        self.view.resetTransform()
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ---------------- SEP PSF estimator ----------------
    def _on_run_sep(self):
        img, _ = self._get_active_image_and_meta()
        if img is None:
            QMessageBox.warning(self, self.tr("No Image"), self.tr("Please select an image before estimating PSF."))
            return
        if thread_running(self._psf_job):
            return

        params = dict(
            threshold_sigma=float(self.sep_threshold_slider.value()),
            min_area=int(self.sep_minarea_spin.value()),
            saturation_limit=float(self.sep_sat_slider.value()),
            max_stars=int(self.sep_maxstars_spin.value()),
            stamp_half_width=int(self.sep_stamp_spin.value()),
            auto=bool(self.sep_auto_checkbox.isChecked()),
        )
        job = _PSFEstimateJob(img, params)
        job.done.connect(self._on_psf_estimate_done)
        self._psf_job = job
        self._set_psf_busy(True, self.tr("Detecting stars…"))
        start_owned_thread(job)

    def _set_psf_busy(self, busy: bool, status: str):
        if not hasattr(self, "sep_run_button"):
            return
        self.sep_run_button.setEnabled(not busy)
        self.sep_run_button.setText(self.tr("Estimating…") if busy else self.tr("Run SEP Extraction"))
        self.sep_status_label.setText(status)

    def _on_psf_estimate_done(self, res):
        if self.sender() is not self._psf_job:
            return                    # dialog closed meanwhile
        self._psf_job = None
        if isinstance(res, Exception):   # sep raises plain Exception too (e.g. pixel buffer full)
            self._set_psf_busy(False, self.tr("PSF estimate failed."))
            QMessageBox.critical(self, self.tr("PSF Error"), str(res))
            return
        psf_kernel, info = res
        self._set_psf_busy(False, info["status"])
        self._last_stellar_psf = psf_kernel
        self._show_stellar_psf_preview(psf_kernel)


    def _show_stellar_psf_preview(self, psf_kernel: np.ndarray):
        h, w = psf_kernel.shape
        img8 = ((psf_kernel / max(psf_kernel.max(), 1e-12)) * 255.0).astype(np.uint8)
        qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
        scaled = QPixmap.fromImage(qimg).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        final = QPixmap(64, 64); final.fill(Qt.GlobalColor.transparent)
        p = QPainter(final); p.drawPixmap((64 - scaled.width()) // 2, (64 - scaled.height()) // 2, scaled); p.end()
        self.sep_psf_preview.setPixmap(final)

    def _on_use_stellar_psf(self):
        if self._last_stellar_psf is None:
            QMessageBox.warning(self, self.tr("No PSF"), self.tr("Run SEP extraction first."))
            return
        self._activate_custom_psf(self._last_stellar_psf)
        self.deconv_algo_combo.setCurrentText("Richardson-Lucy")
        self.rl_custom_label.setVisible(True)
        self.rl_disable_custom_btn.setVisible(True)
        self.custom_psf_bar.setVisible(True)
        QMessageBox.information(self, self.tr("PSF Selected"), self.tr("Stellar PSF is now active for Richardson–Lucy."))

    def _activate_custom_psf(self, kernel: np.ndarray):
        """Make `kernel` the stellar PSF RL/Wiener use (a private float32 copy)."""
        self._custom_psf = np.array(kernel, dtype=np.float32)
        self._use_custom_psf = True
        self.conv_psf_label.setPixmap(self._make_stellar_psf_pixmap(self._custom_psf))
        self.rl_custom_label.setVisible(True)
        self.rl_disable_custom_btn.setVisible(True)
        self.custom_psf_bar.setVisible(
            self.deconv_algo_combo.currentText() in ("Richardson-Lucy", "Wiener"))

    def _clear_custom_psf_flag(self, _=None):
        if self._use_custom_psf:
            self._use_custom_psf = False
            self._custom_psf = None
            self.rl_custom_label.setVisible(False)
            self.rl_disable_custom_btn.setVisible(False)
            self.custom_psf_bar.setVisible(False)

    def _on_save_stellar_psf(self):
        if self._last_stellar_psf is None:
            QMessageBox.warning(self, self.tr("No PSF"), self.tr("Run SEP extraction before saving."))
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save PSF as..."),
            "",
            "TIFF (*.tif);;FITS (*.fits)"
        )
        if not path:
            return

        ext = path.lower().split('.')[-1]

        if ext == 'fits':
            fits.PrimaryHDU(self._last_stellar_psf.astype(np.float32)).writeto(path, overwrite=True)

        elif ext in ('tif', 'tiff'):
            import tifffile
            tifffile.imwrite(path, self._last_stellar_psf.astype(np.float32))

        else:
            QMessageBox.warning(self, self.tr("Invalid Extension"), self.tr("Please choose .fits or .tif."))
            return

        QMessageBox.information(self, self.tr("Saved"), self.tr("PSF saved to:\n{0}").format(path))


# ─────────────────────────────────────────────────────────────────────────────
def preset_psf_kernel(p: dict) -> Optional[np.ndarray]:
    """The stellar PSF a Convo/Deconvo preset carries ("psf_kernel"), as the
    float32 array the dialog used, or None when it uses the slider PSF.
    Raises ValueError for a kernel that can't be one: a replay must not
    quietly deconvolve with a different PSF than the one recorded."""
    v = (p or {}).get("psf_kernel")
    if v is None:
        return None
    try:
        k = np.asarray(v, dtype=np.float32)
    except (TypeError, ValueError):
        k = None
    if (k is None or k.ndim != 2 or k.size == 0
            or not np.all(np.isfinite(k)) or float(k.sum()) <= 0.0):
        raise ValueError("This preset's stellar PSF (psf_kernel) is not a usable 2-D PSF.")
    return k


# The PSF Viewer's detection: same sigma ladder and target.
PSF_AUTO_LADDER = (100.0, 50.0, 25.0, 12.0, 6.0, 3.0)
PSF_AUTO_TARGET = 1000


def estimate_psf_from_image(image_array: np.ndarray,
                            threshold_sigma: float,
                            min_area: int,
                            saturation_limit: float,
                            max_stars: int,
                            stamp_half_width: int,
                            auto: bool = True,
                            target_stars: int = PSF_AUTO_TARGET) -> tuple[np.ndarray, dict]:
    """
    Stellar PSF as a deconvolution kernel ((2*stamp_half_width+1)^2, >= 0,
    sums to 1), plus a summary dict with a ready-made "status" line.

    Estimated the way the PSF Viewer does it, with its psf_utils code:
    - Detection: detect_stars_waterfall. Auto descends sigma 100 -> 3 and
      stops at ~target_stars well-formed stars (flag 0, npix >= min_area,
      sane ellipse); manual is one extraction at threshold_sigma. This used to
      be one extraction at the old 2.5 sigma default, which on a 26 MP master
      ran ~20 minutes inside SEP, holding the GIL. Extraction runs in strips
      (same catalogue) so the GUI stays responsive.
    - Shape: median_star_stamp, the median of border-subtracted,
      peak-normalised cutouts (the PSF Viewer's "true pixels" star), over up
      to max_stars of the stars whose brightest core pixel is under
      saturation_limit. That limit is 16-bit ADU, so on a normalised image it
      is taken as a fraction of 65535.
    The median stamp's floor is noise around zero: values within 3 sigma of
    that noise (measured on the stamp's outer ring) are set to zero so they
    don't add a flat pedestal to the kernel.
    """
    img = np.asarray(image_array)
    gray = img.mean(axis=2) if img.ndim == 3 else img
    gray = np.ascontiguousarray(gray, dtype=np.float32)
    H, W = gray.shape

    ladder = PSF_AUTO_LADDER if auto else (float(threshold_sigma),)
    cat = detect_stars_waterfall(gray, sigma_ladder=ladder,
                                 target_count=int(target_stars) if auto else 10**9,
                                 min_area=int(min_area), quality_filter=True, strips=True)
    if cat is None or cat["n"] == 0:
        where = "down to 3σ" if auto else f"at {float(threshold_sigma):g}σ"
        raise RuntimeError(f"No usable stars found {where}.")
    mode = "Auto" if auto else "Manual"
    found = (f"{mode}: {cat['n']} stars at σ={cat['sigma_stopped']:g} "
             f"(kept {cat['n']}/{cat['total_at_stop']} at that threshold)")

    # Saturation: brightest pixel of the 3x3 core, over all channels.
    xi = np.clip(np.rint(cat["x"]).astype(np.int64), 1, W - 2)
    yi = np.clip(np.rint(cat["y"]).astype(np.int64), 1, H - 2)
    peak = np.full(xi.shape, -np.inf, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            v = img[yi + dy, xi + dx]
            peak = np.maximum(peak, v.max(axis=1) if v.ndim == 2 else v)
    normalised = float(np.nanmax(gray)) <= 1.0 + 1e-3
    limit = float(saturation_limit) / 65535.0 if normalised else float(saturation_limit)
    keep = peak < limit
    if not np.any(keep):
        raise RuntimeError(f"{found}, but every one reaches the saturation cutoff "
                           f"({int(saturation_limit)} ADU). Raise the cutoff.")

    w = int(stamp_half_width)
    stamp = median_star_stamp(gray, cat["x"][keep], cat["y"][keep], w, int(max_stars))
    if stamp is None:
        raise RuntimeError("No valid postage stamps extracted (all were off-edge or zero).")

    ring = np.ones(stamp.shape, dtype=bool)
    ring[2:-2, 2:-2] = False
    floor = stamp[ring]
    noise = 1.4826 * float(np.median(np.abs(floor - np.median(floor))))
    kernel = np.where(stamp > 3.0 * noise, stamp, 0.0).astype(np.float32)
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    else:
        kernel[:] = 0; kernel[w, w] = 1.0

    used = int(min(int(np.count_nonzero(keep)), int(max_stars)))
    fwhm = float(np.median(2.3548 * cat["a"][keep]))
    info = dict(n=int(cat["n"]), sigma_stopped=float(cat["sigma_stopped"]),
                total_at_stop=int(cat["total_at_stop"]), saturated=int(np.count_nonzero(~keep)),
                used=used, fwhm=fwhm)
    # The status line is shown in the dialog; `found` above stays English (exception text).
    if auto:
        found_ui = _tr("Auto: {0} stars at σ={1:g} (kept {0}/{2} at that threshold)")
    else:
        found_ui = _tr("Manual: {0} stars at σ={1:g} (kept {0}/{2} at that threshold)")
    found_ui = found_ui.format(cat['n'], cat['sigma_stopped'], cat['total_at_stop'])
    if info["saturated"]:
        status = _tr("{0}. PSF from {1} stars, {2} saturated left out; median FWHM {3:.2f} px, kernel {4}×{4}.")
    else:
        status = _tr("{0}. PSF from {1} stars; median FWHM {3:.2f} px, kernel {4}×{4}.")
    info["status"] = status.format(found_ui, used, info["saturated"], fwhm, 2 * w + 1)
    return kernel, info


class _PSFEstimateJob(QThread):
    """estimate_psf_from_image off the GUI thread. SEP holds the GIL while it
    extracts, so detection runs in strips (strips=True) to let the GUI in
    between them."""
    done = _pyqtSignal(object)       # (kernel, info) or the Exception raised

    def __init__(self, image: np.ndarray, params: dict):
        super().__init__()           # no parent: thread_keeper owns it
        self.image = image
        self.params = dict(params)

    def run(self):
        try:
            self.done.emit(estimate_psf_from_image(self.image, **self.params))
        except Exception as e:
            self.done.emit(e)


# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=64)
def make_elliptical_gaussian_psf(radius: float, kurtosis: float, aspect: float, rotation_deg: float) -> np.ndarray:
    """Generate elliptical Gaussian PSF kernel. Results are cached."""
    sigma_x = radius
    sigma_y = radius / max(aspect, 1e-8)

    size = int(np.ceil(6 * sigma_x))
    size = size + 1 if size % 2 == 0 else size
    half = size // 2

    xs = np.linspace(-half, half, size)
    ys = np.linspace(-half, half, size)
    xv, yv = np.meshgrid(xs, ys)

    theta = np.deg2rad(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x_rot =  cos_t * xv + sin_t * yv
    y_rot = -sin_t * xv + cos_t * yv

    beta = kurtosis
    squared_sum = (x_rot / max(sigma_x, 1e-8))**2 + (y_rot / max(sigma_y, 1e-8))**2
    psf = np.exp(-(squared_sum ** beta))
    total = psf.sum()
    return (psf / total).astype(np.float32) if total != 0 else np.zeros_like(psf, dtype=np.float32)


def _rl_tile_process_reg(tile_and_meta: Tuple[int, np.ndarray], cancel=None) -> Tuple[int, np.ndarray]:
    (tile_index, padded_tile, psf, num_iter, clip_flag, pad, reg_type, y0_ext, y1_ext, L_ext) = tile_and_meta
    alpha_L2 = 0.01
    alpha_tv = 0.01
    f = np.clip(padded_tile.astype(np.float32), 1e-8, None)
    psf_flipped = psf[::-1, ::-1]
    # fftconvolve(..., mode="same") with each kernel transformed once per tile,
    # not twice per iteration; same values.
    blur = _same_convolver(f, psf)
    blur_t = _same_convolver(f, psf_flipped)

    for _ in range(num_iter):
        if cancel is not None and cancel():
            break
        estimate_blurred = blur(f)
        ratio = padded_tile / (estimate_blurred + 1e-8)
        correction = blur_t(ratio)
        f = f * correction
        if reg_type == "Tikhonov (L2)":
            f = f - alpha_L2 * laplace(f)
        elif reg_type == "Total Variation (TV)":
            f = denoise_tv_chambolle(f, weight=alpha_tv, channel_axis=None).astype(np.float32)
        f = np.clip(f, 0.0, 1.0)

    if cancel is not None and cancel():
        return (tile_index, None)   # canceled: the caller raises, skip the de-ring
    if clip_flag:
        f = denoise_bilateral(f, sigma_color=0.08, sigma_spatial=1).astype(np.float32)

    full_h, full_w = f.shape
    Wcore = full_w - 2 * pad
    deconv_core = f[pad: pad + L_ext, pad: pad + Wcore].astype(np.float32)
    return (tile_index, deconv_core)


# ─────────────────────────────────────────────────────────────────────────────
def van_cittert_deconv(image: np.ndarray, iterations: int, relaxation: float, cancel=None) -> np.ndarray:
    sigma = 3.0
    size = int(np.ceil(6 * sigma)); size = size + 1 if size % 2 == 0 else size
    xs = np.linspace(-size//2, size//2, size)
    kernel_1d = np.exp(-(xs**2) / (2*sigma**2)); kernel_1d = kernel_1d / kernel_1d.sum()
    psf = np.outer(kernel_1d, kernel_1d).astype(np.float32)

    f = image.copy().astype(np.float32)
    image32 = image.astype(np.float32)     # loop-invariant; was converted every iteration
    blur = _same_convolver(f, psf)         # == fftconvolve(f, psf, mode="same"), PSF transformed once
    for _ in range(iterations):
        if cancel is not None and cancel():
            raise _Canceled()
        conv = blur(f)
        f = f + relaxation * (image32 - conv)
    return np.clip(f, 0.0, 1.0)


def rotate_about_center(image: np.ndarray, angle_deg: float, center: Tuple[float, float]) -> np.ndarray:
    img_f = img_as_float32(image)
    H, W = img_f.shape[:2]
    y0, x0 = center
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    tx = x0 - ( x0 * cos_t - y0 * sin_t )
    ty = y0 - ( x0 * sin_t + y0 * cos_t )
    M3 = np.array([[ cos_t, -sin_t, tx ],
                   [ sin_t,  cos_t, ty ],
                   [  0.0 ,   0.0 , 1.0 ]], dtype=np.float32)
    tform = AffineTransform(matrix=np.linalg.inv(M3))
    rotated = warp(img_f, inverse_map=tform, order=1, mode='constant', cval=0.0, preserve_range=True)
    return rotated.astype(np.float32)


def _bilinear_interpolate_gray(gray: np.ndarray, y_coords: np.ndarray, x_coords: np.ndarray, cval: float = 0.0) -> np.ndarray:
    H, W = gray.shape
    x0 = np.floor(x_coords).astype(int); x1 = x0 + 1
    y0 = np.floor(y_coords).astype(int); y1 = y0 + 1
    dx = x_coords - x0; dy = y_coords - y0
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    Ia = gray[y0c, x0c]; Ib = gray[y0c, x1c]; Ic = gray[y1c, x0c]; Id = gray[y1c, x1c]
    wa = (1 - dx) * (1 - dy); wb = dx * (1 - dy); wc = (1 - dx) * dy; wd = dx * dy
    interp = (Ia * wa) + (Ib * wb) + (Ic * wc) + (Id * wd)
    oob = (x_coords < 0) | (x_coords >= W) | (y_coords < 0) | (y_coords >= H)
    interp[oob] = cval
    return interp.astype(np.float32)


def larson_sekanina(image: np.ndarray, center: Tuple[float, float], radial_step: Optional[float],
                    angular_step_deg: float, operator: str = "Divide") -> np.ndarray:
    if image.dtype != np.float32:
        raise ValueError("larson_sekanina: input must be float32 in [0..1]")
    if image.ndim == 3 and image.shape[2] == 3:
        from skimage.color import rgb2gray
        gray = rgb2gray(image)
    else:
        gray = image

    H, W = gray.shape
    y0, x0 = center
    dtheta = (angular_step_deg / 180.0) * np.pi

    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    dy = np.broadcast_to(ys - y0, (H, W))
    dx = np.broadcast_to(xs - x0, (H, W))
    r  = np.sqrt(dx*dx + dy*dy)
    theta = np.arctan2(dy, dx); theta[theta < 0] += 2*np.pi

    r2 = r if (radial_step is None or radial_step <= 0) else (r + radial_step)
    theta2 = (theta + dtheta) % (2*np.pi)

    x2 = x0 + r2 * np.cos(theta2)
    y2 = y0 + r2 * np.sin(theta2)

    J = _bilinear_interpolate_gray_par(gray, y2.ravel(), x2.ravel(), cval=0.0).reshape(H, W)

    if operator == "Divide":
        eps = 1e-6
        med_j = np.median(J)
        med = med_j if med_j > 0 else 1e-6
        B = np.clip(gray * (med / (J + eps)), 0.0, 1.0)
    else:
        diff = gray - J
        B = np.clip(diff, 0.0, None)
        maxv = B.max()
        B = (B / maxv) if maxv > 0 else np.zeros_like(B)

    return B.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Preview building blocks
#
# The heavy steps split work that is independent per row band, FFT line or
# colour channel across threads and reassemble it, so every output is
# byte-for-byte what the original serial call gives. Where a helper leans on
# library internals (fftconvolve's steps, skimage's bilateral kernel, per-line
# FFTs, per-pixel colour conversion), a small self-test against the public call
# runs once per process, and the helper uses the public call if it ever
# differs. The heavy calls release the GIL, so the threads overlap and a
# worker thread leaves the GUI responsive.
# ─────────────────────────────────────────────────────────────────────────────
_POOL_WORKERS = max(1, min(16, os.cpu_count() or 1))
_SELF_TESTS: dict = {}

# Classical Denoise settings each method reads (preview cache key).
_DENOISE_PARAM_KEYS = {
    "TV Chambolle": ("tv_weight", "tv_iter"),
    "Non-Local Means": ("nlm_h", "nlm_patch", "nlm_dist"),
    "Bilateral": ("bil_d", "bil_sigma_color", "bil_sigma_space"),
    "Gaussian": ("gau_sigma",),
    "Median": ("med_size",),
    "Wavelet": ("wav_sigma", "wav_wavelet", "wav_mode"),
}
# Methods that are serial inside, so per-channel mode runs the channels in parallel.
_CHANNEL_PARALLEL_DENOISE = ("TV Chambolle", "Non-Local Means", "Wavelet")


def _row_bands(n: int, parts: int, halo: int = 0) -> list:
    """Split range(n) into up to `parts` bands [(y0, y1, e0, e1)]; [e0, e1) is
    [y0, y1) grown by `halo` and clamped to [0, n)."""
    n = int(n)
    parts = max(1, min(int(parts), n))
    step = -(-n // parts)
    return [(y0, min(n, y0 + step), max(0, y0 - halo), min(n, y0 + step + halo))
            for y0 in range(0, n, step)]


def _run_parallel(fn, items, workers: int | None = None) -> list:
    """[fn(x) for x in items], on a thread pool when there is more than one."""
    items = list(items)
    if len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(len(items), workers or _POOL_WORKERS)) as ex:
        return list(ex.map(fn, items))


def _self_test(name: str, check) -> bool:
    """check() once per process; True only if it returned True without raising."""
    ok = _SELF_TESTS.get(name)
    if ok is None:
        try:
            ok = bool(check())
        except Exception:
            ok = False
        _SELF_TESTS[name] = ok
    return ok


def _same_bytes(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _test_rng():
    return np.random.default_rng(20261119)


# ── FFT convolution with the kernel transformed once ─────────────────────────
def _fft_same_geometry(s1, s2):
    """FFT size and 'same'-mode crop origin fftconvolve uses for 2-D shapes s1, s2."""
    shape = [int(s1[i]) + int(s2[i]) - 1 for i in range(2)]
    fshape = [_sp_fft.next_fast_len(n, True) for n in shape]
    start = [(shape[i] - int(s1[i])) // 2 for i in range(2)]
    return fshape, start


def _fft_conv_window(x, kernel_spec, fshape, y0, x0, h, w, out=None):
    """Window [y0:y0+h, x0:x0+w] of the full linear convolution of x with the
    kernel whose rfftn at fshape is kernel_spec (an array, or a Future of one,
    awaited after x is transformed). The rfftn -> product -> irfftn sequence of
    scipy.signal.fftconvolve, so the values are identical."""
    sp = _sp_fft.rfftn(x, fshape, axes=[0, 1])
    ks = kernel_spec.result() if hasattr(kernel_spec, "result") else kernel_spec
    np.multiply(sp, ks, out=sp)
    ret = _sp_fft.irfftn(sp, fshape, axes=[0, 1])
    del sp
    win = ret[y0:y0 + h, x0:x0 + w]
    if out is None:
        return win.copy()
    out[...] = win
    return out


def _fft_same_usable(x_shape, x_dtype, kernel) -> bool:
    """True when _fft_conv_window reproduces fftconvolve(x, kernel, mode='same'):
    2-D float32 operands with no length-1 axis (scipy skips the FFT there)."""
    if not (len(x_shape) == 2 and kernel.ndim == 2 and np.dtype(x_dtype) == np.float32
            and kernel.dtype == np.float32 and min(x_shape) > 1 and min(kernel.shape) > 1):
        return False

    def check():
        rng = _test_rng()
        x = rng.random((61, 83), dtype=np.float32)
        k = rng.random((9, 7), dtype=np.float32)
        k /= k.sum()
        fshape, start = _fft_same_geometry(x.shape, k.shape)
        for kk in (k, k[::-1, ::-1]):
            spec = _sp_fft.rfftn(kk, fshape, axes=[0, 1])
            mine = _fft_conv_window(x, spec, fshape, start[0], start[1], x.shape[0], x.shape[1])
            if not _same_bytes(mine, fftconvolve(x, kk, mode="same")):
                return False
        return True
    return _self_test("fft_same", check)


def _same_convolver(like: np.ndarray, kernel: np.ndarray):
    """A function equal to fftconvolve(y, kernel, mode="same") for arrays y shaped
    and typed like `like`, with the kernel transformed once instead of per call."""
    if not _fft_same_usable(like.shape, like.dtype, kernel):
        return lambda y: fftconvolve(y, kernel, mode="same")
    fshape, start = _fft_same_geometry(like.shape, kernel.shape)
    spec = _sp_fft.rfftn(kernel, fshape, axes=[0, 1])
    shape, dtype = like.shape, like.dtype

    def conv(y):
        if y.shape != shape or y.dtype != dtype:
            return fftconvolve(y, kernel, mode="same")
        return _fft_conv_window(y, spec, fshape, start[0], start[1], shape[0], shape[1])
    return conv


def _convolve_image(image: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
    """
    Convolve image with psf_kernel using reflect padding so we don't get
    dark borders from zero-padding. Returns same HxW (and channels) as input.

    The colour channels run in parallel and the PSF is transformed once, not
    once per channel; the values are those of the per-channel fftconvolve loop.
    """
    if image is None or psf_kernel is None:
        return image

    img = image.astype(np.float32, copy=False)
    if not (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3)):
        # Unknown layout; just return a copy to be safe
        return img.copy()
    kh, kw = psf_kernel.shape
    pad_y = kh // 2
    pad_x = kw // 2
    H, W = img.shape[:2]
    out = np.empty(img.shape, dtype=np.float32)
    src = [img] if img.ndim == 2 else [img[:, :, c] for c in range(3)]
    dst = [out] if img.ndim == 2 else [out[:, :, c] for c in range(3)]

    def _padded(c):
        if pad_y or pad_x:
            return np.pad(src[c], ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
        return src[c]

    padded_shape = (H + 2 * pad_y, W + 2 * pad_x)
    if _fft_same_usable(padded_shape, np.float32, psf_kernel):
        fshape, start = _fft_same_geometry(padded_shape, psf_kernel.shape)
        y0, x0 = start[0] + pad_y, start[1] + pad_x
        with ThreadPoolExecutor(max_workers=len(src) + 1) as ex:
            spec = ex.submit(_sp_fft.rfftn, psf_kernel, fshape, axes=[0, 1])

            def _one(c):
                _fft_conv_window(_padded(c), spec, fshape, y0, x0, H, W, out=dst[c])
            for fut in [ex.submit(_one, c) for c in range(len(src))]:
                fut.result()
    else:
        def _one(c):
            conv_full = fftconvolve(_padded(c), psf_kernel, mode="same")
            if pad_y or pad_x:
                conv_full = conv_full[pad_y:-pad_y or None, pad_x:-pad_x or None]
            dst[c][...] = conv_full
        _run_parallel(_one, range(len(src)), workers=len(src))

    # PSF is normalized, but clamp just in case of numeric noise
    np.clip(out, 0.0, 1.0, out=out)
    return out


# ── Richardson-Lucy ──────────────────────────────────────────────────────────
def _rl_deconvolve(image: np.ndarray, psf_kernel: np.ndarray, iterations: int,
                   reg_type: str = "None (Plain R–L)", clip_flag: bool = True,
                   progress=None, cancel=None) -> np.ndarray:
    """Tiled Richardson-Lucy (ConvoDeconvoDialog._richardson_lucy_color).

    progress(text) receives the status lines; cancel() returning True stops
    the run between iterations and raises _Canceled.
    """
    report = progress if progress is not None else (lambda _text: None)
    iters = int(round(iterations))
    psf = psf_kernel.astype(np.float32)

    def _deconv_2d_parallel(gray: np.ndarray) -> np.ndarray:
        H, W = gray.shape
        psf_h, psf_w = psf.shape
        half_psf = max(psf_h, psf_w) // 2
        extra = 15
        pad = half_psf + extra
        overlap = pad

        n_cores = min((os.cpu_count() or 1), H)
        tile_h = math.ceil(H / n_cores)
        tile_ranges = []
        for i in range(n_cores):
            y0 = i * tile_h; y1 = min((i + 1) * tile_h, H)
            if y0 >= H: break
            tile_ranges.append((y0, y1))

        accum_image = np.zeros((H, W), dtype=np.float32)
        accum_weight = np.zeros((H, W), dtype=np.float32)

        def _build_vertical_ramp(L: int, ov: int) -> np.ndarray:
            w = np.ones(L, dtype=np.float32)
            if ov <= 0: return w
            if 2 * ov >= L:
                for i in range(L):
                    w[i] = 1.0 - abs((i - (L - 1) / 2) / ((L - 1) / 2))
                return w
            for i in range(ov):
                w[i] = (i + 1) / float(ov)
                w[L - 1 - i] = (i + 1) / float(ov)
            return w

        tile_inputs = []
        for idx, (y0, y1) in enumerate(tile_ranges):
            y0_ext = max(0, y0 - overlap); y1_ext = min(H, y1 + overlap)
            core_tile = gray[y0_ext:y1_ext, :]
            padded = np.pad(core_tile, ((pad, pad), (pad, pad)), mode="reflect")
            L_ext = y1_ext - y0_ext
            tile_inputs.append((idx, padded, psf, iters, clip_flag, pad, reg_type, y0_ext, y1_ext, L_ext))

        results = [None] * len(tile_inputs)
        max_workers = min(len(tile_inputs), os.cpu_count() or 1)
        if max_workers < 1:
            max_workers = 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for tile_index, deconv_ext in executor.map(lambda t: _rl_tile_process_reg(t, cancel), tile_inputs):
                results[tile_index] = deconv_ext
        if cancel is not None and cancel():
            raise _Canceled()

        for idx, (y0, y1) in enumerate(tile_ranges):
            (_, _, _, _, _, _, _, y0_ext, y1_ext, L_ext) = tile_inputs[idx]
            deconv_ext = results[idx]
            w = _build_vertical_ramp(L_ext, overlap)[:, None]   # broadcasts across the row
            accum_image[y0_ext:y1_ext, :] += deconv_ext * w
            accum_weight[y0_ext:y1_ext, :] += w

        final_deconv = np.zeros_like(accum_image, dtype=np.float32)
        nz = accum_weight > 0
        np.divide(accum_image, accum_weight, out=final_deconv, where=nz)
        return final_deconv

    if image.ndim == 2:
        report(_tr("Running RL for {0} iterations").format(iters))
        deconv = _deconv_2d_parallel(image.astype(np.float32, copy=False))
        report("")
        return np.clip(deconv, 0.0, 1.0)
    elif image.ndim == 3 and image.shape[2] == 3:
        outs = []
        for c in range(3):
            report(_tr("Running RL on ch {0} for {1} iterations").format(c+1, iters))
            outs.append(np.clip(_deconv_2d_parallel(image[:, :, c].astype(np.float32)), 0.0, 1.0))
        report("")
        return np.stack(outs, axis=2)
    else:
        return image.copy()


# ── Wiener ───────────────────────────────────────────────────────────────────
def _fft2_lines_impl(a, inverse: bool, parts: int):
    f = np.fft.ifft if inverse else np.fft.fft
    H, W = a.shape
    rows = _row_bands(H, parts)
    pieces = _run_parallel(lambda b: f(a[b[0]:b[1]], axis=-1), rows, workers=parts)
    tmp = np.empty((H, W), dtype=pieces[0].dtype)
    for (y0, y1, _, _), piece in zip(rows, pieces):
        tmp[y0:y1] = piece
    del pieces
    out = np.empty_like(tmp)

    def _cols(b):
        out[:, b[0]:b[1]] = f(tmp[:, b[0]:b[1]], axis=0)
    _run_parallel(_cols, _row_bands(W, parts), workers=parts)
    return out


def _fft2_lines(a: np.ndarray, inverse: bool = False) -> np.ndarray:
    """numpy fft2 / ifft2 of a 2-D array with its 1-D line transforms (rows, then
    columns, fft2's own order) spread over threads. Every line goes through the
    same 1-D transform as inside fft2, so the result is identical."""
    def check():
        x = _test_rng().random((67, 131), dtype=np.float32)
        X = fft2(x)
        return (_same_bytes(_fft2_lines_impl(x, False, 5), X)
                and _same_bytes(_fft2_lines_impl(X, True, 7), ifft2(X)))
    if a.ndim != 2 or min(a.shape) < 2 * _POOL_WORKERS or not _self_test("fft2_lines", check):
        return ifft2(a) if inverse else fft2(a)
    return _fft2_lines_impl(a, inverse, _POOL_WORKERS)


def _bilateral_bands_impl(image, sigma_color, sigma_spatial, parts):
    # skimage.restoration.denoise_bilateral (channel_axis=None, defaults
    # bins=10000, mode='constant', cval=0), run on row bands with a window-radius
    # halo. The global terms (min, max, LUTs) come from the whole image.
    from skimage.restoration._denoise import _compute_color_lut, _compute_spatial_lut
    from skimage.restoration._denoise_cy import _denoise_bilateral
    from skimage.util import img_as_float
    bins, mode, cval = 10000, "constant", 0
    win_size = max(5, 2 * int(math.ceil(3 * sigma_spatial)) + 1)
    min_value = image.min()
    max_value = image.max()
    if min_value == max_value:
        return image
    im = np.ascontiguousarray(np.atleast_3d(img_as_float(image)))
    sigma_color = sigma_color or im.std()
    color_lut = _compute_color_lut(bins, sigma_color, max_value, dtype=im.dtype)
    range_lut = _compute_spatial_lut(win_size, sigma_spatial, dtype=im.dtype)
    if min_value < 0:
        im = im - min_value
        max_value -= min_value
    out = np.empty(im.shape, dtype=im.dtype)

    def _band(b):
        y0, y1, e0, e1 = b
        sub = np.ascontiguousarray(im[e0:e1])
        res = np.empty(sub.shape, dtype=sub.dtype)
        _denoise_bilateral(sub, max_value, win_size, sigma_color, sigma_spatial, bins, mode, cval,
                           color_lut, range_lut, np.empty(im.shape[2], dtype=im.dtype), res)
        out[y0:y1] = res[y0 - e0:y1 - e0]
    _run_parallel(_band, _row_bands(im.shape[0], parts, win_size // 2), workers=parts)
    out = np.squeeze(out)
    if min_value < 0:
        out += min_value
    return out


def _bilateral_bands(image: np.ndarray, sigma_color: float, sigma_spatial: float) -> np.ndarray:
    """denoise_bilateral(image, sigma_color=..., sigma_spatial=...) of a 2-D image in
    row bands across threads; identical output (the kernel is per pixel)."""
    def check():
        x = _test_rng().random((53, 71), dtype=np.float32)
        y = x - np.float32(0.4)
        return all(_same_bytes(_bilateral_bands_impl(a, 0.08, 1, n),
                               denoise_bilateral(a, sigma_color=0.08, sigma_spatial=1))
                   for a, n in ((x, 4), (y, 5)))
    if (image.ndim != 2 or image.shape[0] < 4 * _POOL_WORKERS
            or not _self_test("bilateral_bands", check)):
        return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    return _bilateral_bands_impl(image, sigma_color, sigma_spatial, _POOL_WORKERS)


def _wiener_deconvolve(image: np.ndarray, small_psf: np.ndarray, nsr: float,
                       reg_type: str, do_dering: bool, cancel=None) -> np.ndarray:
    """Wiener deconvolution (ConvoDeconvoDialog._wiener_deconv_with_kernel).

    The filter depends only on the image size, so it is built once and shared
    by the colour channels; FFT lines and de-ring bands run in parallel. Same
    values as building it per channel with numpy's fft2.
    """
    def _check_cancel():
        if cancel is not None and cancel():
            raise _Canceled()

    def _filter(H, W):
        psf_h, psf_w = small_psf.shape
        Hpsf = np.zeros((H, W), dtype=np.float32)
        cy, cx = H // 2, W // 2
        y0 = cy - psf_h // 2; x0 = cx - psf_w // 2
        Hpsf[y0:y0+psf_h, x0:x0+psf_w] = small_psf
        H_f = _fft2_lines(ifftshift(Hpsf)); H_f_conj = np.conj(H_f); mag2 = np.abs(H_f) ** 2
        K = nsr * nsr if reg_type == "Tikhonov" else nsr
        return H_f_conj / (mag2 + K)

    def _deconv_gray(im2d: np.ndarray, Wf: np.ndarray, do_dering_flag: bool) -> np.ndarray:
        X = _fft2_lines(im2d)
        if X.dtype == Wf.dtype:
            np.multiply(Wf, X, out=X)          # Wf * fft2(im2d), in place
        else:
            X = Wf * X
        _check_cancel()
        deconv = np.real(_fft2_lines(X, inverse=True)).astype(np.float32)
        del X
        _check_cancel()
        if do_dering_flag:
            deconv = _bilateral_bands(deconv, sigma_color=0.08, sigma_spatial=1)
        return deconv.clip(0.0, 1.0)

    if image.ndim == 2:
        return _deconv_gray(image.astype(np.float32), _filter(*image.shape), do_dering)
    elif image.ndim == 3 and image.shape[2] == 3:
        Wf = _filter(*image.shape[:2])
        return np.stack([_deconv_gray(image[:, :, c].astype(np.float32), Wf, do_dering)
                         for c in range(3)], axis=2)
    else:
        return image.copy()


# ── per-pixel / local helpers ────────────────────────────────────────────────
def _rows_map_impl(fn, a, parts):
    bands = _row_bands(a.shape[0], parts)
    pieces = _run_parallel(lambda b: fn(a[b[0]:b[1]]), bands, workers=parts)
    out = np.empty((a.shape[0],) + pieces[0].shape[1:], dtype=pieces[0].dtype)
    for (y0, y1, _, _), piece in zip(bands, pieces):
        out[y0:y1] = piece
    return out


def _rgb2lab_rows(rgb: np.ndarray) -> np.ndarray:
    """skimage rgb2lab in row bands across threads (per pixel, so identical)."""
    def check():
        x = _test_rng().random((37, 29, 3), dtype=np.float32)
        return _same_bytes(_rows_map_impl(rgb2lab, x, 4), rgb2lab(x))
    if rgb.ndim != 3 or rgb.shape[0] < 4 * _POOL_WORKERS or not _self_test("rgb2lab_rows", check):
        return rgb2lab(rgb)
    return _rows_map_impl(rgb2lab, rgb, _POOL_WORKERS)


def _lab2rgb_rows(lab: np.ndarray) -> np.ndarray:
    """skimage lab2rgb in row bands across threads (per pixel, so identical; an
    out-of-gamut warning comes once per band rather than once)."""
    def check():
        x = rgb2lab(_test_rng().random((37, 29, 3), dtype=np.float32))
        return _same_bytes(_rows_map_impl(lab2rgb, x, 4), lab2rgb(x))
    if lab.ndim != 3 or lab.shape[0] < 4 * _POOL_WORKERS or not _self_test("lab2rgb_rows", check):
        return lab2rgb(lab)
    return _rows_map_impl(lab2rgb, lab, _POOL_WORKERS)


def _bilinear_interpolate_gray_par(gray: np.ndarray, y_coords: np.ndarray, x_coords: np.ndarray,
                                   cval: float = 0.0) -> np.ndarray:
    """_bilinear_interpolate_gray over chunks of the coordinate list in parallel
    (elementwise arithmetic and gathers only, so identical)."""
    n = int(y_coords.size)
    if n < (1 << 16):
        return _bilinear_interpolate_gray(gray, y_coords, x_coords, cval=cval)
    out = np.empty(n, dtype=np.float32)

    def _chunk(b):
        out[b[0]:b[1]] = _bilinear_interpolate_gray(gray, y_coords[b[0]:b[1]], x_coords[b[0]:b[1]], cval=cval)
    _run_parallel(_chunk, _row_bands(n, _POOL_WORKERS))
    return out


def _local_filter_bands_impl(fn, a, halo, parts):
    bands = _row_bands(a.shape[0], parts, halo)

    def _band(b):
        y0, y1, e0, e1 = b
        return fn(a[e0:e1])[y0 - e0:y1 - e0]
    pieces = _run_parallel(_band, bands, workers=parts)
    out = np.empty(a.shape, dtype=pieces[0].dtype)
    for (y0, y1, _, _), piece in zip(bands, pieces):
        out[y0:y1] = piece
    return out


def _local_filter_bands(fn, a: np.ndarray, halo: int, kind: str) -> np.ndarray:
    """fn(a) for a 2-D local filter ('reflect' borders) whose output rows depend
    only on input rows within `halo`, in row bands across threads; identical."""
    parts = min(_POOL_WORKERS, a.shape[0] // max(64, 4 * halo)) if a.ndim == 2 else 0

    def check():
        x = _test_rng().random((97, 53), dtype=np.float32)
        return _same_bytes(_local_filter_bands_impl(fn, x, halo, 6), fn(x))
    if parts < 2 or not _self_test("bands_" + kind, check):
        return fn(a)
    return _local_filter_bands_impl(fn, a, halo, parts)


# ── Classical Denoise ────────────────────────────────────────────────────────
def _denoise_channel(algo: str, ch: np.ndarray, p: dict) -> np.ndarray:
    """
    Run the selected algorithm on a single 2-D float32 [0,1] channel.
    Returns float32 [0,1]. p: ConvoDeconvoDialog._denoise_params().
    """
    ch = np.asarray(ch, dtype=np.float32)

    if algo == "TV Chambolle":
        return denoise_tv_chambolle(
            ch, weight=p["tv_weight"], max_num_iter=p["tv_iter"], channel_axis=None
        ).astype(np.float32)

    elif algo == "Non-Local Means":
        h_val      = p["nlm_h"]
        patch_size = p["nlm_patch"]
        patch_dist = p["nlm_dist"]
        # Ensure odd
        if patch_size % 2 == 0:
            patch_size += 1
        if patch_dist % 2 == 0:
            patch_dist += 1
        sigma_est = float(np.mean(estimate_sigma(ch, channel_axis=None)))
        return denoise_nl_means(
            ch,
            h=h_val,
            fast_mode=True,
            patch_size=patch_size,
            patch_distance=patch_dist,
            sigma=sigma_est,
            channel_axis=None,
        ).astype(np.float32)

    elif algo == "Bilateral":
        d           = p["bil_d"]
        sigma_color = p["bil_sigma_color"]
        sigma_space = p["bil_sigma_space"]
        # cv2 bilateral works on 8-bit; convert, filter, convert back
        ch8 = np.clip(ch * 255.0, 0, 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(ch8, d, sigma_color, sigma_space)
        return (filtered.astype(np.float32) / 255.0)

    elif algo == "Gaussian":
        sigma = p["gau_sigma"]
        # Kernel radius int(4*sigma + 0.5) (scipy's truncate=4.0), +1 for margin.
        g = _local_filter_bands(lambda a: gaussian_filter(a, sigma=sigma), ch,
                                int(4.0 * sigma + 0.5) + 1, "gaussian")
        return np.clip(g.astype(np.float32), 0.0, 1.0)

    elif algo == "Median":
        size = p["med_size"]
        if size % 2 == 0:
            size += 1   # enforce odd
        m = _local_filter_bands(lambda a: median_filter(a, size=size), ch, size // 2 + 1, "median")
        return np.clip(m.astype(np.float32), 0.0, 1.0)

    elif algo == "Wavelet":
        sigma_val = p["wav_sigma"]
        wavelet   = p["wav_wavelet"]
        mode      = p["wav_mode"]
        # sigma=None triggers BayesShrink auto-estimate
        sigma_arg = None if sigma_val < 1e-6 else sigma_val
        return np.clip(
            denoise_wavelet(
                ch,
                sigma=sigma_arg,
                wavelet=wavelet,
                mode=mode,
                method="BayesShrink",
                channel_axis=None,
            ).astype(np.float32),
            0.0, 1.0,
        )

    else:
        return ch.copy()


def _classical_denoise(img: np.ndarray, p: dict) -> np.ndarray:
    """
    The selected classical denoiser (ConvoDeconvoDialog._apply_classical_denoise);
    L*-only or per channel for colour images. Returns float32 [0,1].
    """
    algo = p["algo"]
    lum_only = p["lum_only"]
    is_color = (img.ndim == 3 and img.shape[2] == 3)

    img_f = np.asarray(img, dtype=np.float32)

    # ── L*-only mode: extract L, denoise it, put back ────────
    if lum_only and is_color:
        lab = _rgb2lab_rows(img_f)
        L_norm = (lab[:, :, 0] / 100.0).astype(np.float32)
        L_denoised = _denoise_channel(algo, L_norm, p)
        lab[:, :, 0] = np.clip(L_denoised * 100.0, 0.0, 100.0)
        return np.clip(_lab2rgb_rows(lab).astype(np.float32), 0.0, 1.0)

    # ── Per-channel mode ────────────────────────────────────
    if is_color:
        workers = 3 if algo in _CHANNEL_PARALLEL_DENOISE else 1
        channels = _run_parallel(lambda c: _denoise_channel(algo, img_f[:, :, c], p),
                                 range(3), workers=workers)
        return np.clip(np.stack(channels, axis=2), 0.0, 1.0)

    # ── Mono ─────────────────────────────────────────────────
    mono = img_f.reshape(img_f.shape[0], img_f.shape[1])
    return np.clip(_denoise_channel(algo, mono, p), 0.0, 1.0)


# ── mask + display (worker side) ─────────────────────────────────────────────
def _mask_float2d(data) -> np.ndarray:
    """A mask layer's data as a 2-D float32 mask in [0..1] (RGB(A) -> gray,
    0..255 -> 0..1). Always a new array; the layer's own data is not touched
    (the old in-place /= 255 rescaled a float32 doc mask)."""
    m = np.asarray(data)
    # If RGB(A) mask, convert to gray
    if m.ndim == 3:
        if cv2 is not None:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        else:
            m = m.mean(axis=2)

    m = m.astype(np.float32, copy=False)
    if m.max() > 1.0:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def _mask_or_none(data):
    if data is None:
        return None
    try:
        return _mask_float2d(data)
    except Exception:
        return None


def _apply_mask_blend(processed: np.ndarray, orig: np.ndarray, mask2d) -> np.ndarray:
    """The dialog's masked blend: clip(processed*m + orig*(1-m)) with the active
    mask (nearest-resized to the result), or the result itself with no mask."""
    if mask2d is None:
        return processed
    H, W = processed.shape[:2]
    m = mask2d
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)
    m = np.clip(m.astype(np.float32, copy=False), 0.0, 1.0)
    if processed.ndim == 3:
        m = m[:, :, None]   # broadcasts: the values of repeating it per channel, without the copy
    return np.clip(processed * m + orig * (1.0 - m), 0.0, 1.0)


def _qimage_from_array(array: np.ndarray):
    """(QImage, buffer): 8-bit display image, mono as Grayscale8, colour as RGB888.
    The QImage points into the buffer, so keep both until QPixmap.fromImage has
    copied it. QImage is reentrant, so this runs on a worker (QPixmap may not)."""
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype in (np.float32, np.float64):
        arr8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr8 = (np.clip(arr, 0, 65535) // 257).astype(np.uint8)
    elif arr.dtype == np.uint8:
        arr8 = arr
    else:
        mn, mx = arr.min(), arr.max()
        arr8 = ((arr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(arr, dtype=np.uint8)
    arr8 = np.ascontiguousarray(arr8)

    h, w = arr8.shape[:2]
    if arr8.ndim == 2:
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(arr8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return qimg, arr8


def _display_images(final: np.ndarray, orig, d: dict) -> dict:
    """Off-thread half of _refresh_display for a new result: the shared display
    stretch (measured on the original, as _display_stretch does) and the 8-bit
    images of the result and, when the cached one is stale, of the original."""
    params = None
    shown = final
    if d["auto"]:
        params = d["params"]
        if params is None:
            params = display_stretch_params(orig if orig is not None else final,
                                            target_median=d["tm"], linked=d["linked"])
        shown = apply_display_stretch(final, params)
    after = _qimage_from_array(shown)
    del shown
    before = None
    if d["need_before"] and orig is not None and orig.shape[:2] == final.shape[:2]:
        before = _qimage_from_array(apply_display_stretch(orig, params) if params is not None else orig)
    out = dict(d)
    out.update(params=params, after=after, before=before, orig=orig, shown=final)
    return out


# Optional helper to open like SFCC:
def open_convo_deconvo(doc_manager, parent=None, doc=None) -> ConvoDeconvoDialog:
    dlg = ConvoDeconvoDialog(doc_manager=doc_manager, parent=parent, doc=doc)
    dlg.show()
    return dlg

