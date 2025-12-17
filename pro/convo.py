# pro/convo.py
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

# ── Qt
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QImage, QPainter, QPen, QColor, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMessageBox,
    QDialog, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QSlider, QLineEdit,
    QFormLayout, QTabWidget, QComboBox, QCheckBox, QPushButton, QToolButton,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QWidget,
    QSpinBox
)
import cv2
# Optional FITS export
from astropy.io import fits

import sep  # PSF estimator

# Import centralized widgets
from pro.widgets.spinboxes import CustomSpinBox
from pro.widgets.themed_buttons import themed_toolbtn


# --- GraphicsView with Shift+Click LS center + optional scene ctor -----------
class InteractiveGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene | None = None, parent=None):
        super().__init__(parent)
        if scene is not None:
            self.setScene(scene)
        self.ls_center: Optional[Tuple[float, float]] = None
        self.cross_items = []

    def mousePressEvent(self, event):
        if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and event.button() == Qt.MouseButton.LeftButton:
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

        self.setWindowTitle("Convolution / Deconvolution")
        self.resize(1000, 650)
        self._use_custom_psf = False
        self._custom_psf: Optional[np.ndarray] = None
        self._last_stellar_psf: Optional[np.ndarray] = None
        self._original_image: Optional[np.ndarray] = None
        self._preview_result: Optional[np.ndarray] = None
        self._auto_fit = False
        self._load_original_on_show = True

        # ── Layout: left controls / right preview
        main_layout = QHBoxLayout(self)
        # Left
        left_panel = QFrame(); left_panel.setFrameShape(QFrame.Shape.StyledPanel); left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel); main_layout.addWidget(left_panel)
        # Right
        preview_panel = QFrame(); preview_layout = QVBoxLayout(preview_panel); main_layout.addWidget(preview_panel, stretch=1)

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
        self.preview_btn = QPushButton("Preview")
        self.undo_btn    = QPushButton("Undo")
        self.close_btn   = QPushButton("Close")
        row1.addWidget(self.preview_btn); row1.addWidget(self.undo_btn)
        left_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.push_btn    = QPushButton("Push")
        row2.addWidget(self.push_btn); row2.addWidget(self.close_btn)
        left_layout.addLayout(row2)

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
            self._load_original_on_show = False
        self.conv_psf_label.clear()
        self.sep_psf_preview.clear() if hasattr(self, "sep_psf_preview") else None
        self._update_psf_preview()

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
        self.tabs.addTab(conv_tab, "Convolution")

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
        self.tabs.addTab(deconv_tab, "Deconvolution")

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
        self.tabs.addTab(psf_tab, "PSF Estimator")

    def _build_tv_denoise_tab(self):
        tvd_tab = QWidget(); layout = QVBoxLayout(tvd_tab)
        form = QFormLayout(); form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.tv_weight_slider = FloatSliderWithEdit(minimum=0.0, maximum=1.0, step=0.01, initial=0.1, suffix="")
        form.addRow("TV Weight:", self.tv_weight_slider)
        self.tv_iter_slider   = FloatSliderWithEdit(minimum=1, maximum=100, step=1, initial=10, suffix="")
        form.addRow("Max Iterations:", self.tv_iter_slider)
        self.tv_multichannel_checkbox = QCheckBox("Multi‐channel"); self.tv_multichannel_checkbox.setChecked(True)
        self.tv_multichannel_checkbox.setToolTip("If checked and the image is color, run TV on all channels jointly")
        form.addRow("", self.tv_multichannel_checkbox)
        layout.addLayout(form); layout.addStretch()
        self.tabs.addTab(tvd_tab, "TV Denoise")

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

        elif current_tab_name == "TV Denoise":
            weight = self.tv_weight_slider.value()
            max_iter = int(self.tv_iter_slider.value())
            multichannel = self.tv_multichannel_checkbox.isChecked()

            if img.ndim == 3 and multichannel:
                processed = denoise_tv_chambolle(img.astype(np.float32), weight=weight, max_num_iter=max_iter, channel_axis=-1).astype(np.float32)
            else:
                if img.ndim == 3 and img.shape[2] == 3:
                    channels_out = [
                        denoise_tv_chambolle(img[:, :, c].astype(np.float32), weight=weight, max_num_iter=max_iter, channel_axis=None).astype(np.float32)
                        for c in range(3)
                    ]
                    processed = np.stack(channels_out, axis=2)
                else:
                    gray = img.astype(np.float32) if img.ndim == 2 else img
                    processed = denoise_tv_chambolle(gray, weight=weight, max_num_iter=max_iter, channel_axis=None).astype(np.float32)

            processed = np.clip(processed, 0.0, 1.0)
            processed = processed * self.strength_slider.value() + (1 - self.strength_slider.value()) * img

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
        self._display_in_view(final_result)

    def _on_undo(self):
        if self._original_image is not None:
            self._preview_result = None
            self._display_in_view(self._original_image)
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

        # ── TV Denoise tab ──────────────────────────────────────────────
        if current_tab_name == "TV Denoise":
            return {
                "op": "tv",
                "tv_weight":       float(self.tv_weight_slider.value()),
                "tv_iter":         int(round(float(self.tv_iter_slider.value()))),
                "tv_multichannel": bool(self.tv_multichannel_checkbox.isChecked()),
                "strength":        strength,
            }

        return None


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

        sigma   = self.sep_threshold_slider.value()
        minarea = self.sep_minarea_spin.value
        sat     = self.sep_sat_slider.value()
        maxstars= self.sep_maxstars_spin.value
        half_w  = self.sep_stamp_spin.value

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
            QMessageBox.critical(self, "PSF Error", str(e)); return

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
