# saspro/convo.py
from __future__ import annotations

import os
import math
import numpy as np
from typing import Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# ── SciPy / scikit-image
from scipy.signal import fftconvolve
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

import sep  # PSF estimator

# Import centralized widgets
from setiastro.saspro.widgets.spinboxes import CustomSpinBox
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.imageops.stretch import stretch_color_image, stretch_mono_image


from PyQt6.QtCore import QThread, pyqtSignal as _pyqtSignal
import numpy as np
 
 
class _DenoiseWorker(QThread):
    """
    Runs ConvoDeconvoDialog._apply_classical_denoise() on a background thread.
 
    Signals
    -------
    progress(str)           — short status message for the UI label
    finished(object, str)   — (result_ndarray | None, error_msg)
    """
    progress = _pyqtSignal(str)
    finished = _pyqtSignal(object, str)   # ndarray | None, error
 
    def __init__(self, dialog, img: np.ndarray, parent=None):
        super().__init__(parent)
        self._dialog = dialog   # reference to the dialog (read-only during run)
        self._img    = img
        self._canceled = False
 
    def cancel(self):
        self._canceled = True
 
    def run(self):
        try:
            algo = self._dialog.denoise_algo_combo.currentText()
            self.progress.emit(f"Running {algo}…")
            result = self._dialog._apply_classical_denoise(self._img)
            if self._canceled:
                self.finished.emit(None, "__canceled__")
            else:
                self.finished.emit(result, "")
        except Exception as exc:
            import traceback
            if self._canceled:
                self.finished.emit(None, "__canceled__")
            else:
                self.finished.emit(None, f"{exc}\n\n{traceback.format_exc()}")

# --- GraphicsView with Shift+Click LS center + optional scene ctor -----------

from PyQt6.QtCore import Qt
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
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
 
    def mousePressEvent(self, event):
        if (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
            and event.button() == Qt.MouseButton.LeftButton
        ):
            scene_pt = self.mapToScene(event.position().toPoint())
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
        self._original_image: Optional[np.ndarray] = None
        self._preview_result: Optional[np.ndarray] = None
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

        self.cb_autostretch = QCheckBox("Auto-stretch preview")
        self.cb_autostretch.setChecked(False)
        self.cb_autostretch.setToolTip(
            "Apply a statistical stretch to the preview only.\n"
            "The actual image data is NOT modified — this is display only.\n"
            "Useful for checking denoising / convolution results on linear data."
        )
        stretch_bar.addWidget(self.cb_autostretch)

        stretch_bar.addWidget(QLabel("Target median:"))
        self.s_target_median = QDoubleSpinBox()
        self.s_target_median.setRange(0.01, 0.60)
        self.s_target_median.setSingleStep(0.01)
        self.s_target_median.setDecimals(3)
        self.s_target_median.setValue(0.25)
        self.s_target_median.setFixedWidth(70)
        self.s_target_median.setEnabled(False)   # disabled until checkbox on
        stretch_bar.addWidget(self.s_target_median)

        self.cb_linked = QCheckBox("Linked")
        self.cb_linked.setChecked(True)
        self.cb_linked.setToolTip("Stretch all RGB channels together (linked).")
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
        srow = QHBoxLayout(); srow.addWidget(QLabel("Strength:")); srow.addWidget(self.strength_slider)
        left_layout.addLayout(srow)

        # Buttons
        row1 = QHBoxLayout()
        self.preview_btn = QPushButton(self.tr("Preview"))
        self.undo_btn    = QPushButton(self.tr("Undo"))
        self.close_btn   = QPushButton(self.tr("Close"))
        row1.addWidget(self.preview_btn); row1.addWidget(self.undo_btn)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.push_btn    = QPushButton(self.tr("Push"))
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
            drag_row.addStretch(1)
            left_layout.addLayout(drag_row)
        except Exception:
            pass

        left_layout.addStretch()
        self.rl_status_label = QLabel(""); self.rl_status_label.setStyleSheet("color:#fff;background:#333;padding:4px;")
        self.rl_status_label.setFixedHeight(24)
        left_layout.addWidget(self.rl_status_label)

        # Zoom & Preview
        zrow = QHBoxLayout(); zrow.addStretch()
        self.zoom_in_btn = QToolButton();  self.zoom_in_btn.setIcon(QIcon.fromTheme("zoom-in"));     self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_out_btn= QToolButton();  self.zoom_out_btn.setIcon(QIcon.fromTheme("zoom-out"));    self.zoom_out_btn.setToolTip("Zoom Out")
        self.fit_btn     = QToolButton();  self.fit_btn.setIcon(QIcon.fromTheme("zoom-fit-best"));    self.fit_btn.setToolTip("Fit to Preview")
        zrow.addWidget(self.zoom_in_btn); zrow.addWidget(self.zoom_out_btn); zrow.addWidget(self.fit_btn)
        preview_layout.addLayout(zrow)

        self.scene = QGraphicsScene()
        self.view = InteractiveGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pixmap_item = QGraphicsPixmapItem(); self.scene.addItem(self.pixmap_item)
        preview_layout.addWidget(self.view)

        # Signals
        self.preview_btn.clicked.connect(self._on_preview)
        self.undo_btn.clicked.connect(self._on_undo)
        self.push_btn.clicked.connect(self._on_push_to_doc)
        self.close_btn.clicked.connect(self.close)

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
        # Autostretch state — mirrors BlemishBlaster pattern
        self._stretch_original: np.ndarray | None = None
        self._stretch_is_mono: bool = False

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
        self._preview_result = None
        self._original_image = img.copy() if isinstance(img, np.ndarray) else None
        if self._original_image is not None:
            self._auto_fit = True
            self._display_in_view(self._original_image)
            self._cache_stretch_original(self._original_image)

    def _on_autostretch_toggled(self, on: bool):
        """Enable/disable dependent controls and refresh display."""
        self.s_target_median.setEnabled(on)
        self.cb_linked.setEnabled(on)
        self._refresh_display()
    
    
    def _refresh_display(self, *_):
        """
        Re-render the preview view from the stored linear original,
        applying autostretch if enabled.  Called whenever stretch
        parameters change, or after Preview/Undo/Push updates the image.
    
        If no linear original is cached yet, falls back gracefully.
        """
        src = getattr(self, "_stretch_original", None)
        if src is None:
            # Nothing to redisplay yet — leave view as-is
            return
    
        if not self.cb_autostretch.isChecked():
            self._display_in_view(src)
            return
    
        tm = float(self.s_target_median.value())
        src_f = np.asarray(src, dtype=np.float32)
    
        is_mono = getattr(self, "_stretch_is_mono", False)
    
        if is_mono:
            # Mono: collapse to single plane, stretch, replicate for display
            if src_f.ndim == 3:
                plane = src_f[:, :, 0]
            else:
                plane = src_f
            stretched = stretch_mono_image(
                plane, target_median=tm, normalize=False,
                apply_curves=False, no_black_clip=False
            )
            disp = np.stack([stretched] * 3, axis=-1).astype(np.float32)
        else:
            # Color — linked or unlinked
            linked = self.cb_linked.isChecked()
            disp = stretch_color_image(
                src_f, target_median=tm, linked=linked,
                normalize=False, apply_curves=False, no_black_clip=False
            ).astype(np.float32)
    
        self._display_in_view(np.clip(disp, 0.0, 1.0))
    
    
    def _cache_stretch_original(self, img: np.ndarray):
        """
        Store a linear copy of `img` for the autostretch display path.
        Call this any time the working image changes:
        - on showEvent (initial load)
        - after Preview produces a result
        - after Undo
        - after Push (new baseline)
        """
        self._stretch_original = np.asarray(img, dtype=np.float32).copy()
        self._stretch_is_mono = (
            img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)
        )
    
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
        if self._load_original_on_show:
            img, _ = self._get_active_image_and_meta()
            if img is not None:
                self._original_image = img.copy()
                self._auto_fit = True
                self._display_in_view(img)
                self._cache_stretch_original(img)
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
        self.conv_psf_label.clear() if hasattr(self, "conv_psf_label") else None
        self.sep_psf_preview.clear() if hasattr(self, "sep_psf_preview") else None
        self.rl_status_label.setText("") if hasattr(self, "rl_status_label") else None
        self.custom_psf_bar.setVisible(False) if hasattr(self, "custom_psf_bar") else None
        self._stretch_original = None
        if getattr(self, "_denoise_worker", None) is not None:
            if self._denoise_worker.isRunning():
                self._denoise_worker.cancel()
                self._denoise_worker.wait(2000)
            self._denoise_worker = None


        super().closeEvent(ev)

    # ---------------- Build tabs ----------------
    def _build_convolution_tab(self):
        conv_tab = QWidget()
        layout = QVBoxLayout(conv_tab)
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.conv_radius_slider = FloatSliderWithEdit(minimum=0.1, maximum=200.0, step=0.1, initial=5.0, suffix=" px")
        form.addRow("Radius:", self.conv_radius_slider)

        self.conv_shape_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=2.0, suffix="σ")
        form.addRow("Kurtosis (σ):", self.conv_shape_slider)

        self.conv_aspect_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=1.0, suffix="")
        form.addRow("Aspect Ratio:", self.conv_aspect_slider)

        self.conv_rotation_slider = FloatSliderWithEdit(minimum=0.0, maximum=360.0, step=1.0, initial=0.0, suffix="°")
        form.addRow("Rotation:", self.conv_rotation_slider)

        layout.addLayout(form); layout.addStretch()
        self.tabs.addTab(conv_tab, self.tr("Convolution"))

    def _build_deconvolution_tab(self):
        deconv_tab = QWidget()
        outer_layout = QVBoxLayout(deconv_tab)

        # Algo row
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("Algorithm:"))
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
        psf_group_layout.addRow("PSF Radius:", self.rl_psf_radius_slider)
        self.rl_psf_shape_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=2.0, suffix="σ")
        psf_group_layout.addRow("PSF Kurtosis (σ):", self.rl_psf_shape_slider)
        self.rl_psf_aspect_slider = FloatSliderWithEdit(minimum=0.1, maximum=10.0, step=0.1, initial=1.0, suffix="")
        psf_group_layout.addRow("PSF Aspect Ratio:", self.rl_psf_aspect_slider)
        self.rl_psf_rotation_slider = FloatSliderWithEdit(minimum=0.0, maximum=360.0, step=1.0, initial=0.0, suffix="°")
        psf_group_layout.addRow("PSF Rotation:", self.rl_psf_rotation_slider)
        outer_layout.addWidget(self.psf_param_group)
        self.psf_param_group.setVisible(self.deconv_algo_combo.currentText() in ("Richardson-Lucy", "Wiener"))

        # “Using Stellar PSF” bar
        self.custom_psf_bar = QWidget()
        bar_layout = QHBoxLayout(self.custom_psf_bar); bar_layout.setContentsMargins(0, 0, 0, 0); bar_layout.setSpacing(4)
        self.rl_custom_label = QLabel("Using Stellar PSF")
        self.rl_custom_label.setStyleSheet("color:#fff;background-color:#007acc;padding:2px;")
        self.rl_custom_label.setVisible(False)
        self.rl_disable_custom_btn = QPushButton("Disable Stellar PSF")
        self.rl_disable_custom_btn.setToolTip("Revert to PSF sliders")
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
        rl_form.addRow("Iterations:", self.rl_iterations_slider)
        self.rl_reg_combo = QComboBox(); self.rl_reg_combo.addItems(["None (Plain R–L)", "Tikhonov (L2)", "Total Variation (TV)"])
        rl_form.addRow("Regularization:", self.rl_reg_combo)
        self.rl_clip_checkbox = QCheckBox("Enable de‐ring"); self.rl_clip_checkbox.setChecked(True)
        rl_form.addRow("", self.rl_clip_checkbox)
        self.rl_luminance_only_checkbox = QCheckBox("Deconvolve L* Only"); self.rl_luminance_only_checkbox.setChecked(True)
        self.rl_luminance_only_checkbox.setToolTip("If checked and the image is color, RL runs only on the L* channel.")
        rl_form.addRow("", self.rl_luminance_only_checkbox)
        rl_widget.setLayout(rl_form)
        self.deconv_param_stack["Richardson-Lucy"] = rl_widget

        # Wiener
        wiener_widget = QWidget(); wiener_layout = QVBoxLayout(wiener_widget)
        wiener_form = QFormLayout(); wiener_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.wiener_nsr_slider = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.001, initial=0.01, suffix="")
        wiener_form.addRow("Noise/Signal (λ):", self.wiener_nsr_slider)
        self.wiener_reg_combo = QComboBox(); self.wiener_reg_combo.addItems(["None (Classical Wiener)", "Tikhonov (L2)"])
        wiener_form.addRow("Regularization:", self.wiener_reg_combo)
        self.wiener_luminance_only_checkbox = QCheckBox("Deconvolve L* Only"); self.wiener_luminance_only_checkbox.setChecked(True)
        self.wiener_luminance_only_checkbox.setToolTip("If checked and the image is color, Wiener runs only on the L* channel.")
        wiener_form.addRow("", self.wiener_luminance_only_checkbox)
        self.wiener_dering_checkbox = QCheckBox("Enable de-ring"); self.wiener_dering_checkbox.setChecked(True)
        self.wiener_dering_checkbox.setToolTip("Applies a single bilateral pass after Wiener deconvolution")
        wiener_form.addRow("", self.wiener_dering_checkbox)
        wiener_layout.addLayout(wiener_form)
        self.deconv_param_stack["Wiener"] = wiener_widget

        # Larson–Sekanina
        ls_widget = QWidget(); ls_form = QFormLayout(ls_widget); ls_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.ls_radial_slider  = FloatSliderWithEdit(minimum=0.0, maximum=50.0, step=0.1, initial=0.0, suffix=" px")
        self.ls_angular_slider = FloatSliderWithEdit(minimum=0.1, maximum=360.0, step=0.1, initial=1.0, suffix="°")
        self.ls_operator_combo = QComboBox(); self.ls_operator_combo.addItems(["Divide", "Subtract"])
        self.ls_blend_combo    = QComboBox(); self.ls_blend_combo.addItems(["SoftLight", "Screen"])
        ls_form.addRow("Radial Step (px):", self.ls_radial_slider)
        ls_form.addRow("Angular Step (°):", self.ls_angular_slider)
        ls_form.addRow("LS Operator:", self.ls_operator_combo)
        ls_form.addRow("Blend Mode:", self.ls_blend_combo)
        self.ls_operator_combo.currentTextChanged.connect(self._on_ls_operator_changed)
        self.deconv_param_stack["Larson-Sekanina"] = ls_widget

        # Van Cittert
        vc_widget = QWidget(); vc_form = QFormLayout(vc_widget); vc_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.vc_iterations_slider = FloatSliderWithEdit(minimum=1, maximum=1000, step=1, initial=10, suffix="")
        self.vc_relax_slider      = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=0.0, suffix="")
        vc_form.addRow("Iterations:", self.vc_iterations_slider)
        vc_form.addRow("Relaxation (0–1):", self.vc_relax_slider)
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
        h_image.addWidget(QLabel("Image for PSF Estimate:"))
        self.sep_image_label = QLabel("(Current Active Image)")
        h_image.addWidget(self.sep_image_label); layout.addLayout(h_image)

        form = QFormLayout(); form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sep_threshold_slider = FloatSliderWithEdit(minimum=1.0, maximum=5.0, step=0.1, initial=2.5, suffix=" σ")
        form.addRow("Detection σ:", self.sep_threshold_slider)
        self.sep_minarea_spin = CustomSpinBox(minimum=1, maximum=100, initial=5, step=1)
        form.addRow("Min Area (px²):", self.sep_minarea_spin)
        self.sep_sat_slider = FloatSliderWithEdit(minimum=1000, maximum=100000, step=500, initial=50000, suffix=" ADU")
        form.addRow("Saturation Cutoff:", self.sep_sat_slider)
        self.sep_maxstars_spin = CustomSpinBox(minimum=1, maximum=500, initial=50, step=1)
        form.addRow("Max Stars:", self.sep_maxstars_spin)
        self.sep_stamp_spin = CustomSpinBox(minimum=5, maximum=50, initial=15, step=1)
        form.addRow("Half‐Width (px):", self.sep_stamp_spin)
        layout.addLayout(form)

        h_buttons = QHBoxLayout()
        self.sep_run_button = QPushButton("Run SEP Extraction")
        self.sep_save_button = QPushButton("Save PSF…")
        self.sep_use_button  = QPushButton("Use as Current PSF")
        h_buttons.addWidget(self.sep_run_button); h_buttons.addWidget(self.sep_save_button); h_buttons.addWidget(self.sep_use_button)
        layout.addLayout(h_buttons)

        self.psf_estimate_title = QLabel("Estimated PSF (64×64):")
        layout.addWidget(self.psf_estimate_title, alignment=Qt.AlignmentFlag.AlignLeft)
        self.sep_psf_preview = QLabel(); self.sep_psf_preview.setFixedSize(64, 64)
        self.sep_psf_preview.setStyleSheet("border: 1px solid #888;")
        layout.addWidget(self.sep_psf_preview, alignment=Qt.AlignmentFlag.AlignHCenter)

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
        algo_row.addWidget(QLabel("Method:"))
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
            "Operate on L* only  (color images — preserves chroma)"
        )
        self.denoise_lum_only_chk.setChecked(True)
        self.denoise_lum_only_chk.setToolTip(
            "When checked, the denoiser runs only on the L* (luminance) channel "
            "of a Lab-converted image and chroma is left untouched.\n"
            "Uncheck to run on all channels independently."
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
        tv_f.addRow("TV Weight:", self.tv_weight_slider)
        self.tv_iter_slider = FloatSliderWithEdit(
            minimum=1, maximum=100, step=1, initial=10, suffix=""
        )
        tv_f.addRow("Max Iterations:", self.tv_iter_slider)
        self.tv_multichannel_checkbox = QCheckBox("Multi-channel")
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
            "Cut-off distance (h). Controls denoising strength.\n"
            "Typical range: 0.01–0.05 for stretched astrophotos."
        )
        nlm_f.addRow("h (strength):", self.nlm_h_slider)
        self.nlm_patch_slider = FloatSliderWithEdit(
            minimum=3, maximum=15, step=2, initial=5, suffix=" px"
        )
        self.nlm_patch_slider.setToolTip(
            "Patch size (pixels). Larger = more context per comparison, slower."
        )
        nlm_f.addRow("Patch size:", self.nlm_patch_slider)
        self.nlm_dist_slider = FloatSliderWithEdit(
            minimum=3, maximum=21, step=2, initial=7, suffix=" px"
        )
        self.nlm_dist_slider.setToolTip(
            "Search distance. Larger = better quality, slower."
        )
        nlm_f.addRow("Search dist:", self.nlm_dist_slider)
        nlm_hint = QLabel("  ⓘ  Fast mode — skimage NLM with sigma estimation.")
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
        self.bil_d_slider.setToolTip("Diameter of each pixel neighbourhood (d).")
        bil_f.addRow("Diameter:", self.bil_d_slider)
        self.bil_sigma_color_slider = FloatSliderWithEdit(
            minimum=1.0, maximum=200.0, step=1.0, initial=50.0, suffix=""
        )
        self.bil_sigma_color_slider.setToolTip(
            "Sigma in colour space. Higher = pixels further in colour are mixed."
        )
        bil_f.addRow("Sigma colour:", self.bil_sigma_color_slider)
        self.bil_sigma_space_slider = FloatSliderWithEdit(
            minimum=1.0, maximum=200.0, step=1.0, initial=50.0, suffix=""
        )
        self.bil_sigma_space_slider.setToolTip(
            "Sigma in coordinate space. Higher = pixels further apart are mixed."
        )
        bil_f.addRow("Sigma space:", self.bil_sigma_space_slider)
        bil_hint = QLabel("  ⓘ  Uses OpenCV bilateralFilter (8-bit internally).")
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
            "Gaussian sigma. 1.0 = gentle, 3.0+ = heavy blurring."
        )
        gau_f.addRow("Sigma:", self.gau_sigma_slider)
        self._denoise_param_stack["Gaussian"] = gau_w
    
        # Median
        med_w = QWidget()
        med_f = QFormLayout(med_w)
        med_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.med_size_slider = FloatSliderWithEdit(
            minimum=1, maximum=15, step=2, initial=3, suffix=" px"
        )
        self.med_size_slider.setToolTip(
            "Kernel size (odd). 3 = minimal, 5 = moderate.\n"
            "Good for cosmic ray speckles."
        )
        med_f.addRow("Kernel size:", self.med_size_slider)
        self._denoise_param_stack["Median"] = med_w
    
        # Wavelet
        wav_w = QWidget()
        wav_f = QFormLayout(wav_w)
        wav_f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.wav_sigma_slider = FloatSliderWithEdit(
            minimum=0.0, maximum=0.30, step=0.001, initial=0.0, suffix=""
        )
        self.wav_sigma_slider.setToolTip(
            "Noise sigma. 0.0 = auto-estimate (BayesShrink)."
        )
        wav_f.addRow("Noise σ (0=auto):", self.wav_sigma_slider)
        self.wav_wavelet_combo = QComboBox()
        self.wav_wavelet_combo.addItems(
            ["db1", "db2", "db4", "db8", "sym4", "sym8", "coif1", "bior1.3"]
        )
        self.wav_wavelet_combo.setCurrentText("db2")
        self.wav_wavelet_combo.setToolTip(
            "Wavelet family. db2–4 = good general purpose.\n"
            "sym4/sym8 = near-symmetric, good for nebulosity."
        )
        wav_f.addRow("Wavelet:", self.wav_wavelet_combo)
        self.wav_mode_combo = QComboBox()
        self.wav_mode_combo.addItems(["soft", "hard"])
        self.wav_mode_combo.setToolTip(
            "soft = smoother.  hard = sharper edges, may ring."
        )
        wav_f.addRow("Threshold mode:", self.wav_mode_combo)
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
    
        self.denoise_cancel_btn = QPushButton("Cancel")
        self.denoise_cancel_btn.setFixedWidth(60)
        self.denoise_cancel_btn.setVisible(False)
        self.denoise_cancel_btn.setStyleSheet(
            "QPushButton { background: rgba(200,80,0,0.25); border: 1px solid rgba(200,80,0,0.5);"
            " border-radius: 3px; padding: 2px 6px; font-size: 10px; }"
            "QPushButton:hover { background: rgba(200,80,0,0.45); }"
        )
        self.denoise_cancel_btn.clicked.connect(self._cancel_denoise)
        status_row.addWidget(self.denoise_cancel_btn)
        outer.addLayout(status_row)
    
        outer.addStretch()
    
        # Wire combo → show/hide panels
        self.denoise_algo_combo.currentTextChanged.connect(self._on_denoise_algo_changed)
    
        # Worker slot (created fresh on each Preview click)
        self._denoise_worker = None
    
        self.tabs.addTab(tvd_tab, self.tr("Classical Denoise"))

    
    
    def _on_denoise_algo_changed(self, algo: str):
        """Show only the selected denoiser's parameter panel."""
        for key, w in self._denoise_param_stack.items():
            w.setVisible(key == algo)


    def _apply_classical_denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Dispatch to the selected classical denoiser.
        Handles L*-only mode for color images uniformly.
        Returns float32 [0,1].
        """
        from skimage.color import rgb2lab, lab2rgb
    
        algo = self.denoise_algo_combo.currentText()
        lum_only = self.denoise_lum_only_chk.isChecked()
        is_color = (img.ndim == 3 and img.shape[2] == 3)
    
        img_f = np.asarray(img, dtype=np.float32)

        # ── L*-only mode: extract L, denoise it, put back ────────
        if lum_only and is_color:
            lab = rgb2lab(img_f)
            L_norm = (lab[:, :, 0] / 100.0).astype(np.float32)
            L_denoised = self._denoise_single_channel(algo, L_norm)   # ← was _denoise_single_channel(self, ...)
            lab[:, :, 0] = np.clip(L_denoised * 100.0, 0.0, 100.0)
            result = np.clip(lab2rgb(lab).astype(np.float32), 0.0, 1.0)
            return result

        # ── Per-channel mode ────────────────────────────────────
        if is_color:
            channels = [
                self._denoise_single_channel(algo, img_f[:, :, c])   # ← same fix
                for c in range(3)
            ]
            return np.clip(np.stack(channels, axis=2), 0.0, 1.0)

        # ── Mono ─────────────────────────────────────────────────
        mono = img_f.reshape(img_f.shape[0], img_f.shape[1])
        return np.clip(self._denoise_single_channel(algo, mono), 0.0, 1.0) 
    
    
    def _denoise_single_channel(self, algo: str, ch: np.ndarray) -> np.ndarray:
        """
        Run the selected algorithm on a single 2-D float32 [0,1] channel.
        Returns float32 [0,1].
        """
        ch = np.asarray(ch, dtype=np.float32)
    
        if algo == "TV Chambolle":
            weight   = float(self.tv_weight_slider.value())
            max_iter = int(round(float(self.tv_iter_slider.value())))
            return denoise_tv_chambolle(
                ch, weight=weight, max_num_iter=max_iter, channel_axis=None
            ).astype(np.float32)
    
        elif algo == "Non-Local Means":
            h_val      = float(self.nlm_h_slider.value())
            patch_size = int(round(float(self.nlm_patch_slider.value())))
            patch_dist = int(round(float(self.nlm_dist_slider.value())))
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
            d           = int(round(float(self.bil_d_slider.value())))
            sigma_color = float(self.bil_sigma_color_slider.value())
            sigma_space = float(self.bil_sigma_space_slider.value())
            # cv2 bilateral works on 8-bit; convert, filter, convert back
            ch8 = np.clip(ch * 255.0, 0, 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(ch8, d, sigma_color, sigma_space)
            return (filtered.astype(np.float32) / 255.0)
    
        elif algo == "Gaussian":
            sigma = float(self.gau_sigma_slider.value())
            return np.clip(
                gaussian_filter(ch, sigma=sigma).astype(np.float32), 0.0, 1.0
            )
    
        elif algo == "Median":
            size = int(round(float(self.med_size_slider.value())))
            if size % 2 == 0:
                size += 1   # enforce odd
            return np.clip(
                median_filter(ch, size=size).astype(np.float32), 0.0, 1.0
            )
    
        elif algo == "Wavelet":
            sigma_val = float(self.wav_sigma_slider.value())
            wavelet   = self.wav_wavelet_combo.currentText()
            mode      = self.wav_mode_combo.currentText()
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

        if current_tab == "Convolution":
            r, k, a, rot = (self.conv_radius_slider.value(), self.conv_shape_slider.value(),
                            self.conv_aspect_slider.value(), self.conv_rotation_slider.value())
            self.conv_psf_label.setPixmap(self._make_psf_pixmap(r, k, a, rot))
        elif current_tab == "Deconvolution" and algo_text in ("Richardson-Lucy", "Wiener"):
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

            m = np.asarray(data)
            # If RGB(A) mask, convert to gray
            if m.ndim == 3:
                if cv2 is not None:
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                else:
                    m = m.mean(axis=2)

            m = m.astype(np.float32, copy=False)
            if m.max() > 1.0:
                m /= 255.0
            return np.clip(m, 0.0, 1.0)
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
    def _on_preview(self):
        doc = self._active_doc()
        if hasattr(self.doc_manager, "set_active_document"):
            self.doc_manager.set_active_document(doc)        
        img, _ = self._get_active_image_and_meta()
        if img is None:
            self._show_message("No active image to process.")
            return

        if self._original_image is None:
            self._original_image = img.copy()

        current_tab_name = self.tabs.tabText(self.tabs.currentIndex())

        if current_tab_name == "Convolution":
            radius  = self.conv_radius_slider.value()
            kurtosis= self.conv_shape_slider.value()
            aspect  = self.conv_aspect_slider.value()
            rotation= self.conv_rotation_slider.value()
            psf_kernel = make_elliptical_gaussian_psf(radius, kurtosis, aspect, rotation).astype(np.float32)
            processed = self._convolve_color(img, psf_kernel)
            processed = processed * self.strength_slider.value() + (1 - self.strength_slider.value()) * img

        elif current_tab_name == "Deconvolution":
            algo = self.deconv_algo_combo.currentText()
            if algo == "Richardson-Lucy":
                iters = int(round(self.rl_iterations_slider.value()))
                reg_type = self.rl_reg_combo.currentText()
                pr, ps, pa, pt = (self.rl_psf_radius_slider.value(), self.rl_psf_shape_slider.value(),
                                  self.rl_psf_aspect_slider.value(), self.rl_psf_rotation_slider.value())
                psf_kernel = make_elliptical_gaussian_psf(pr, ps, pa, pt).astype(np.float32)
                clip_flag = self.rl_clip_checkbox.isChecked()

                if self.rl_luminance_only_checkbox.isChecked() and img.ndim == 3 and img.shape[2] == 3:
                    lab = rgb2lab(img.astype(np.float32))
                    L = (lab[:, :, 0] / 100.0).astype(np.float32)
                    deconv_L = self._richardson_lucy_color(L, psf_kernel, iterations=iters, reg_type=reg_type, clip_flag=clip_flag)
                    lab[:, :, 0] = np.clip(deconv_L * 100.0, 0.0, 100.0)
                    rgb_deconv = lab2rgb(lab.astype(np.float32))
                    processed = np.clip(rgb_deconv.astype(np.float32), 0.0, 1.0)
                else:
                    processed = self._richardson_lucy_color(img.astype(np.float32), psf_kernel, iters, reg_type, clip_flag)

                processed = processed * self.strength_slider.value() + (1 - self.strength_slider.value()) * img

            elif algo == "Wiener":
                if self._use_custom_psf and (self._custom_psf is not None):
                    small_psf = self._custom_psf.astype(np.float32)
                else:
                    pr, ps, pa, pt = (self.rl_psf_radius_slider.value(), self.rl_psf_shape_slider.value(),
                                      self.rl_psf_aspect_slider.value(), self.rl_psf_rotation_slider.value())
                    small_psf = make_elliptical_gaussian_psf(pr, ps, pa, pt).astype(np.float32)

                nsr = self.wiener_nsr_slider.value()
                reg_type = "Wiener" if self.wiener_reg_combo.currentText() == "None (Classical Wiener)" else "Tikhonov"
                do_dering = self.wiener_dering_checkbox.isChecked()

                if self.wiener_luminance_only_checkbox.isChecked() and img.ndim == 3 and img.shape[2] == 3:
                    lab = rgb2lab(img.astype(np.float32))
                    L = (lab[:, :, 0] / 100.0).astype(np.float32)
                    deconv_L = self._wiener_deconv_with_kernel(L, small_psf, nsr, reg_type, do_dering)
                    lab[:, :, 0] = np.clip(deconv_L * 100.0, 0.0, 100.0)
                    rgb_deconv = lab2rgb(lab.astype(np.float32))
                    processed = np.clip(rgb_deconv.astype(np.float32), 0.0, 1.0)
                else:
                    processed = self._wiener_deconv_with_kernel(img, small_psf, nsr, reg_type, do_dering)
                    processed = np.clip(processed, 0.0, 1.0)

                processed = processed * self.strength_slider.value() + (1 - self.strength_slider.value()) * img

            elif algo == "Larson-Sekanina":
                if not hasattr(self.view, "ls_center") or self.view.ls_center is None:
                    QMessageBox.information(self, "Hold Shift + Click",
                        "To choose a Larson–Sekanina center, hold Shift and click on the preview.")
                    return

                center = self.view.ls_center
                rstep  = self.ls_radial_slider.value()
                astep  = self.ls_angular_slider.value()
                operator = self.ls_operator_combo.currentText()
                blend_mode = self.ls_blend_combo.currentText()

                B = larson_sekanina(image=img, center=center, radial_step=rstep, angular_step_deg=astep, operator=operator)
                A = img
                if A.ndim == 3 and A.shape[2] == 3:
                    B_rgb, A_rgb = np.repeat(B[:, :, None], 3, axis=2), A
                else:
                    B_rgb, A_rgb = B[..., None], A[..., None]
                C = (A_rgb + B_rgb - (A_rgb * B_rgb)) if blend_mode == "Screen" else ((1 - 2 * B_rgb) * (A_rgb**2) + 2 * B_rgb * A_rgb)
                processed = np.clip(C, 0.0, 1.0)
                processed = processed[..., 0] if img.ndim == 2 else processed
                processed = processed * self.strength_slider.value() + (1 - self.strength_slider.value()) * img

            elif algo == "Van Cittert":
                iters2 = self.vc_iterations_slider.value()
                relax  = self.vc_relax_slider.value()
                if img.ndim == 3 and img.shape[2] == 3:
                    chans = [van_cittert_deconv(img[:, :, c], iters2, relax) for c in range(3)]
                    processed = np.stack(chans, axis=2)
                else:
                    processed = van_cittert_deconv(img, iters2, relax)
                processed = np.clip(processed, 0.0, 1.0)
                processed = processed * self.strength_slider.value() + (1 - self.strength_slider.value()) * img

            else:
                self._show_message("Unknown deconvolution algorithm")
                return

        elif current_tab_name == "Classical Denoise":
            self._launch_classical_denoise(img)
            return   # worker calls _display_in_view when done


        else:
            self._show_message("Unknown tab")
            return

        # Masked blend if an active mask exists
        mask = self._get_active_mask_from_doc(processed.shape)
        if mask is not None:
            if processed.ndim == 3 and mask.ndim == 2:
                mask = mask[..., None]
            final_result = np.clip(processed * mask + self._original_image * (1.0 - mask), 0.0, 1.0)
        else:
            final_result = processed

        self._preview_result = final_result
        self._cache_stretch_original(final_result)
        self._refresh_display()


    def _launch_classical_denoise(self, img: np.ndarray):
        """
        Kick off _DenoiseWorker for the Classical Denoise tab.
        Returns immediately; result arrives via _on_denoise_finished.
        """
        # Cancel any previous run that somehow didn't finish
        if self._denoise_worker is not None and self._denoise_worker.isRunning():
            self._denoise_worker.cancel()
            self._denoise_worker.wait(2000)
    
        algo = self.denoise_algo_combo.currentText()
        self.denoise_status_label.setText(f"Running {algo}…")
        self._set_denoise_running(True)
    
        worker = _DenoiseWorker(self, img, parent=self)
        worker.progress.connect(
            lambda msg: self.denoise_status_label.setText(msg)
        )
        worker.finished.connect(self._on_denoise_finished)
        self._denoise_worker = worker
        worker.start()
    


    def _on_undo(self):
        if self._original_image is not None:
            self._preview_result = None
            self._display_in_view(self._original_image)
            self._cache_stretch_original(self._original_image)  # ← ADD
            self._refresh_display()                   
        else:
            self._show_message("Nothing to undo.")

    def _build_replay_preset(self) -> dict | None:
        """
        Capture the current UI state as a preset-style dict so Replay Last Action
        can re-run the same Convo/Deconvo/TV operation on another document.
        Matches the schema used by ConvoPresetDialog.result_dict().
        """
        current_tab_name = self.tabs.tabText(self.tabs.currentIndex())
        strength = float(self.strength_slider.value())

        # ── Convolution tab ─────────────────────────────────────────────
        if current_tab_name == "Convolution":
            return {
                "op": "convolution",
                "radius":   float(self.conv_radius_slider.value()),
                "kurtosis": float(self.conv_shape_slider.value()),
                "aspect":   float(self.conv_aspect_slider.value()),
                "rotation": float(self.conv_rotation_slider.value()),
                "strength": strength,
            }

        # ── Deconvolution tab ───────────────────────────────────────────
        if current_tab_name == "Deconvolution":
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
                "luminance_only": bool(self.rl_luminance_only_checkbox.isChecked()),
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

            return p

        # ── Classical Denoise tab ───────────────────────────────────────
        if current_tab_name == self.tr("Classical Denoise") or current_tab_name == "Classical Denoise":
            # Reuse the grip emitter so Replay and the shortcut stay identical.
            return self.get_preset()

        return None

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
            p = {
                "op": "deconvolution",
                "algo": self.deconv_algo_combo.currentText(),
                "psf_radius":   float(self.rl_psf_radius_slider.value()),
                "psf_kurtosis": float(self.rl_psf_shape_slider.value()),
                "psf_aspect":   float(self.rl_psf_aspect_slider.value()),
                "psf_rotation": float(self.rl_psf_rotation_slider.value()),
                "rl_iter":   float(self.rl_iterations_slider.value()),
                "rl_reg":    self.rl_reg_combo.currentText(),
                "rl_dering": bool(self.rl_clip_checkbox.isChecked()),
                "luminance_only": bool(self.rl_luminance_only_checkbox.isChecked()),
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
            QMessageBox.warning(self, "No Document", "No active document to push into.")
            return

        if self._preview_result is None:
            QMessageBox.warning(self, "No Preview", "No preview to push. Click Preview first.")
            return

        # Grab current metadata from this specific doc
        _, meta = self._get_active_image_and_meta()
        new_meta = dict(meta)
        new_meta["source"] = "ConvoDeconvo"

        try:
            if hasattr(doc, "apply_edit"):
                # ⭐ Preferred: update this exact Document (ROI or full) so all views update
                doc.apply_edit(
                    self._preview_result.copy(),
                    metadata=new_meta,
                    step_name="Convo/Deconvo",
                )
            else:
                # Fallback for older paths: go through DocManager active-doc API
                if hasattr(self.doc_manager, "set_active_document"):
                    self.doc_manager.set_active_document(doc)
                self.doc_manager.update_active_document(
                    self._preview_result.copy(),
                    metadata=new_meta,
                    step_name="Convo/Deconvo",
                )
        except Exception as e:
            QMessageBox.critical(self, "Push failed", str(e))
            return

        # Make the pushed image the new baseline so you can iterate
        img_after, _ = self._get_active_image_and_meta()
        if img_after is not None:
            self._original_image = img_after.copy()
            self._preview_result = None
            self._display_in_view(self._original_image)
            self._cache_stretch_original(self._original_image)
            self._refresh_display()   

        # 🔴 Replay wiring (unchanged, just moved under try/except)
        try:
            if self._main is not None:
                preset = self._build_replay_preset()
                if preset:
                    self._main._last_headless_command = {
                        "cid": "convo",
                        "preset": preset,
                    }
                    if hasattr(self._main, "_log"):
                        op = preset.get("op", "convolution")
                        self._main._log(f"Replay: stored Convo/Deconvo ({op}) from dialog.")
        except Exception:
            # Replay wiring should never break the actual push
            pass

        QMessageBox.information(self, "Pushed", "Result committed to the active document.")



    # ---------------- Utils ----------------
    def _show_message(self, text: str):
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.view.resetTransform()
        temp_label = QLabel(text); temp_label.setStyleSheet("color: white; background-color: #222;"); temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = temp_label.grab()
        self.pixmap_item.setPixmap(pixmap)
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _convolve_color(self, image: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
        """
        Convolve image with psf_kernel using reflect padding so we don't get
        dark borders from zero–padding. Returns same H×W (and channels) as input.
        """
        if image is None or psf_kernel is None:
            return image

        img = image.astype(np.float32, copy=False)
        kh, kw = psf_kernel.shape
        pad_y = kh // 2
        pad_x = kw // 2

        def _conv_single_channel(im2d: np.ndarray) -> np.ndarray:
            if pad_y or pad_x:
                padded = np.pad(
                    im2d,
                    ((pad_y, pad_y), (pad_x, pad_x)),
                    mode="reflect"
                )
            else:
                padded = im2d

            conv_full = fftconvolve(padded, psf_kernel, mode="same")

            if pad_y or pad_x:
                conv = conv_full[pad_y:-pad_y or None, pad_x:-pad_x or None]
            else:
                conv = conv_full

            return conv.astype(np.float32)

        if img.ndim == 2:
            out = _conv_single_channel(img)
        elif img.ndim == 3 and img.shape[2] == 3:
            chans = [_conv_single_channel(img[:, :, c]) for c in range(3)]
            out = np.stack(chans, axis=2)
        else:
            # Unknown layout; just return a copy to be safe
            return img.copy()

        # PSF is normalized, but clamp just in case of numeric noise
        return np.clip(out, 0.0, 1.0)


    def _richardson_lucy_color(self, image: np.ndarray, psf_kernel: np.ndarray, iterations: int,
                               reg_type: str = "None (Plain R–L)", clip_flag: bool = True) -> np.ndarray:
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
                for tile_index, deconv_ext in executor.map(_rl_tile_process_reg, tile_inputs):
                    results[tile_index] = deconv_ext

            for idx, (y0, y1) in enumerate(tile_ranges):
                (_, _, _, _, _, _, _, y0_ext, y1_ext, L_ext) = tile_inputs[idx]
                deconv_ext = results[idx]
                w = _build_vertical_ramp(L_ext, overlap)
                w2d = np.broadcast_to(w[:, None], (L_ext, W)).astype(np.float32)
                accum_image[y0_ext:y1_ext, :] += deconv_ext * w2d
                accum_weight[y0_ext:y1_ext, :] += w2d

            final_deconv = np.zeros_like(accum_image, dtype=np.float32)
            nz = accum_weight > 0
            final_deconv[nz] = accum_image[nz] / accum_weight[nz]
            return final_deconv

        if image.ndim == 2:
            self.rl_status_label.setText(f"Running RL for {iters} iterations"); QApplication.processEvents()
            deconv = _deconv_2d_parallel(image.astype(np.float32))
            self.rl_status_label.setText(""); QApplication.processEvents()
            return np.clip(deconv, 0.0, 1.0)
        elif image.ndim == 3 and image.shape[2] == 3:
            outs = []
            for c in range(3):
                self.rl_status_label.setText(f"Running RL on ch {c+1} for {iters} iterations"); QApplication.processEvents()
                outs.append(np.clip(_deconv_2d_parallel(image[:, :, c].astype(np.float32)), 0.0, 1.0))
            self.rl_status_label.setText(""); QApplication.processEvents()
            return np.stack(outs, axis=2)
        else:
            return image.copy()

    def _wiener_deconv_with_kernel(self, image: np.ndarray, small_psf: np.ndarray, nsr: float,
                                   reg_type: str, do_dering: bool) -> np.ndarray:
        def _deconv_gray(im2d: np.ndarray, do_dering_flag: bool) -> np.ndarray:
            H, W = im2d.shape
            psf_h, psf_w = small_psf.shape
            Hpsf = np.zeros((H, W), dtype=np.float32)
            cy, cx = H // 2, W // 2
            y0 = cy - psf_h // 2; x0 = cx - psf_w // 2
            Hpsf[y0:y0+psf_h, x0:x0+psf_w] = small_psf
            H_f = fft2(ifftshift(Hpsf)); H_f_conj = np.conj(H_f); mag2 = np.abs(H_f) ** 2
            K = nsr * nsr if reg_type == "Tikhonov" else nsr
            Wf = H_f_conj / (mag2 + K)
            deconv = np.real(ifft2(Wf * fft2(im2d))).astype(np.float32)
            if do_dering_flag:
                deconv = denoise_bilateral(deconv, sigma_color=0.08, sigma_spatial=1)
            return deconv.clip(0.0, 1.0)

        if image.ndim == 2:
            return _deconv_gray(image.astype(np.float32), do_dering)
        elif image.ndim == 3 and image.shape[2] == 3:
            return np.stack([_deconv_gray(image[:, :, c].astype(np.float32), do_dering) for c in range(3)], axis=2)
        else:
            return image.copy()

    def _display_in_view(self, array: np.ndarray):
        arr = array.copy()
        if arr.dtype in (np.float32, np.float64):
            arr = np.clip(arr, 0.0, 1.0); arr8 = (arr * 255).astype(np.uint8)
        elif arr.dtype == np.uint16:
            arr8 = (np.clip(arr, 0, 65535) // 257).astype(np.uint8)
        elif arr.dtype == np.uint8:
            arr8 = arr
        else:
            mn, mx = arr.min(), arr.max()
            arr8 = ((arr - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(arr, dtype=np.uint8)

        h, w = arr8.shape[:2]
        if arr8.ndim == 2:
            fmt = QImage.Format.Format_Grayscale8; bytespp = w
        else:
            fmt = QImage.Format.Format_RGB888; bytespp = 3 * w

        qimg = QImage(arr8.data, w, h, bytespp, fmt)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)

        if self._auto_fit:
            self.view.resetTransform()
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._auto_fit = False

    def zoom_in(self):  self.view.scale(1.2, 1.2)
    def zoom_out(self): self.view.scale(1/1.2, 1/1.2)

    def _on_fit_clicked(self):
        self._auto_fit = True
        if self._preview_result is not None:
            self._display_in_view(self._preview_result)
        elif self._original_image is not None:
            self._display_in_view(self._original_image)

    # ---------------- SEP PSF estimator ----------------
    def _on_run_sep(self):
        img, _ = self._get_active_image_and_meta()
        if img is None:
            QMessageBox.warning(self, "No Image", "Please select an image before estimating PSF.")
            return

        img_gray = img.mean(axis=2).astype(np.float32) if (img.ndim == 3) else img.astype(np.float32)

        sigma    = float(self.sep_threshold_slider.value())
        minarea  = int(self.sep_minarea_spin.value())     # ✅
        sat      = float(self.sep_sat_slider.value())
        maxstars = int(self.sep_maxstars_spin.value())    # ✅
        half_w   = int(self.sep_stamp_spin.value())       # ✅

        try:
            psf_kernel = estimate_psf_from_image(
                image_array=img_gray,
                threshold_sigma=sigma,
                min_area=minarea,
                saturation_limit=sat,
                max_stars=maxstars,
                stamp_half_width=half_w
            )
        except RuntimeError as e:
            QMessageBox.critical(self, "PSF Error", str(e))
            return

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
            QMessageBox.warning(self, "No PSF", "Run SEP extraction first.")
            return
        self._custom_psf = self._last_stellar_psf.copy()
        self._use_custom_psf = True
        self.conv_psf_label.setPixmap(self._make_stellar_psf_pixmap(self._custom_psf))
        self.deconv_algo_combo.setCurrentText("Richardson-Lucy")
        self.rl_custom_label.setVisible(True)
        self.rl_disable_custom_btn.setVisible(True)
        self.custom_psf_bar.setVisible(True)
        QMessageBox.information(self, "PSF Selected", "Stellar PSF is now active for Richardson–Lucy.")

    def _clear_custom_psf_flag(self, _=None):
        if self._use_custom_psf:
            self._use_custom_psf = False
            self._custom_psf = None
            self.rl_custom_label.setVisible(False)
            self.rl_disable_custom_btn.setVisible(False)
            self.custom_psf_bar.setVisible(False)

    def _on_save_stellar_psf(self):
        if self._last_stellar_psf is None:
            QMessageBox.warning(self, "No PSF", "Run SEP extraction before saving.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PSF as...",
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
            QMessageBox.warning(self, "Invalid Extension", "Please choose .fits or .tif.")
            return

        QMessageBox.information(self, "Saved", f"PSF saved to:\n{path}")


    def _cancel_denoise(self):
        """Cancel a running denoise worker."""
        if self._denoise_worker is not None and self._denoise_worker.isRunning():
            self._denoise_worker.cancel()
            self.denoise_status_label.setText("Canceling…")
    
    
    def _set_denoise_running(self, running: bool):
        """Lock/unlock UI around a running denoise job."""
        self.preview_btn.setEnabled(not running)
        self.push_btn.setEnabled(not running)
        self.denoise_cancel_btn.setVisible(running)
        if not running:
            # re-enable the algo combo and all param sliders
            self.denoise_algo_combo.setEnabled(True)
            self.denoise_lum_only_chk.setEnabled(True)
            self._denoise_stack_container.setEnabled(True)
        else:
            self.denoise_algo_combo.setEnabled(False)
            self.denoise_lum_only_chk.setEnabled(False)
            self._denoise_stack_container.setEnabled(False)
    
    
    def _on_denoise_finished(self, result, error_msg: str):
        """Slot called when _DenoiseWorker finishes."""
        self._set_denoise_running(False)
        self._denoise_worker = None
    
        if error_msg == "__canceled__":
            self.denoise_status_label.setText("Canceled.")
            return
    
        if error_msg or result is None:
            self.denoise_status_label.setText("Error.")
            QMessageBox.critical(
                self, "Denoise Error",
                f"Denoising failed:\n\n{error_msg.splitlines()[0] if error_msg else 'Unknown error'}\n\n"
                "Check the SASpro log for details."
            )
            return
    
        # Apply strength blend on GUI thread (fast — just math on arrays)
        img = self._original_image
        strength = self.strength_slider.value()
        processed = np.clip(result, 0.0, 1.0)
        processed = processed * strength + (1.0 - strength) * img
    
        # Masked blend
        mask = self._get_active_mask_from_doc(processed.shape)
        if mask is not None:
            if processed.ndim == 3 and mask.ndim == 2:
                mask = mask[..., None]
            final_result = np.clip(
                processed * mask + self._original_image * (1.0 - mask), 0.0, 1.0
            )
        else:
            final_result = processed
    
        self._preview_result = final_result
        self._cache_stretch_original(final_result)
        self._refresh_display()
        self.denoise_status_label.setText("Done.")



# ─────────────────────────────────────────────────────────────────────────────
def estimate_psf_from_image(image_array: np.ndarray,
                            threshold_sigma: float,
                            min_area: int,
                            saturation_limit: float,
                            max_stars: int,
                            stamp_half_width: int) -> np.ndarray:
    data = image_array.astype(np.float32)
    bkg = sep.Background(data)
    bkg_sub = data - bkg.back()
    sources = sep.extract(bkg_sub, thresh=threshold_sigma, err=bkg.globalrms, minarea=min_area)
    if len(sources) == 0:
        raise RuntimeError(f"No sources found with SEP threshold = {threshold_sigma:.1f} σ.")

    valid_sources = [s for s in sources if s['peak'] < saturation_limit]
    if len(valid_sources) == 0:
        raise RuntimeError(f"All detected sources exceed saturation limit {int(saturation_limit)}.")

    valid_sources.sort(key=lambda s: s['peak'], reverse=True)
    selected = valid_sources[:max_stars]

    w = stamp_half_width
    ksize = 2*w + 1
    psf_sum = np.zeros((ksize, ksize), dtype=np.float32)
    count = 0

    H, W = data.shape[:2]
    for src in selected:
        xi = int(round(src['x'])); yi = int(round(src['y']))
        y0, y1 = yi - w, yi + w + 1
        x0, x1 = xi - w, xi + w + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue
        stamp = bkg_sub[y0:y1, x0:x1].astype(np.float32)
        total_flux = float(np.sum(stamp))
        if total_flux <= 0:
            continue
        psf_sum += (stamp / total_flux)
        count += 1

    if count == 0:
        raise RuntimeError("No valid postage stamps extracted (all were off-edge or zero).")

    psf_kernel = (psf_sum / count).astype(np.float32)
    total = float(psf_kernel.sum())
    if total > 0:
        psf_kernel /= total
    else:
        psf_kernel[:] = 0; psf_kernel[w, w] = 1.0
    return psf_kernel


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


def _rl_tile_process_reg(tile_and_meta: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    (tile_index, padded_tile, psf, num_iter, clip_flag, pad, reg_type, y0_ext, y1_ext, L_ext) = tile_and_meta
    alpha_L2 = 0.01
    alpha_tv = 0.01
    f = np.clip(padded_tile.astype(np.float32), 1e-8, None)
    psf_flipped = psf[::-1, ::-1]

    for _ in range(num_iter):
        estimate_blurred = fftconvolve(f, psf, mode="same")
        ratio = padded_tile / (estimate_blurred + 1e-8)
        correction = fftconvolve(ratio, psf_flipped, mode="same")
        f = f * correction
        if reg_type == "Tikhonov (L2)":
            f = f - alpha_L2 * laplace(f)
        elif reg_type == "Total Variation (TV)":
            f = denoise_tv_chambolle(f, weight=alpha_tv, channel_axis=None).astype(np.float32)
        f = np.clip(f, 0.0, 1.0)

    if clip_flag:
        f = denoise_bilateral(f, sigma_color=0.08, sigma_spatial=1).astype(np.float32)

    full_h, full_w = f.shape
    Wcore = full_w - 2 * pad
    deconv_core = f[pad: pad + L_ext, pad: pad + Wcore].astype(np.float32)
    return (tile_index, deconv_core)


# ─────────────────────────────────────────────────────────────────────────────
def van_cittert_deconv(image: np.ndarray, iterations: int, relaxation: float) -> np.ndarray:
    sigma = 3.0
    size = int(np.ceil(6 * sigma)); size = size + 1 if size % 2 == 0 else size
    xs = np.linspace(-size//2, size//2, size)
    kernel_1d = np.exp(-(xs**2) / (2*sigma**2)); kernel_1d = kernel_1d / kernel_1d.sum()
    psf = np.outer(kernel_1d, kernel_1d).astype(np.float32)

    f = image.copy().astype(np.float32)
    for _ in range(iterations):
        conv = fftconvolve(f, psf, mode="same")
        f = f + relaxation * (image.astype(np.float32) - conv)
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

    J = _bilinear_interpolate_gray(gray, y2.ravel(), x2.ravel(), cval=0.0).reshape(H, W)

    if operator == "Divide":
        eps = 1e-6
        med = np.median(J) if np.median(J) > 0 else 1e-6
        B = np.clip(gray * (med / (J + eps)), 0.0, 1.0)
    else:
        diff = gray - J
        B = np.clip(diff, 0.0, None)
        maxv = B.max()
        B = (B / maxv) if maxv > 0 else np.zeros_like(B)

    return B.astype(np.float32)


# Optional helper to open like SFCC:
def open_convo_deconvo(doc_manager, parent=None, doc=None) -> ConvoDeconvoDialog:
    dlg = ConvoDeconvoDialog(doc_manager=doc_manager, parent=parent, doc=doc)
    dlg.show()
    return dlg

