from __future__ import annotations
import os, glob, shutil, tempfile, datetime as _dt
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox)

import pyqtgraph as pg
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# optional deps used in your code; guard if not installed
try:
    import rawpy
except Exception:
    rawpy = None
try:
    import exifread
except Exception:
    exifread = None

import sep
import exifread

# your helpers/utilities
from imageops.stretch import stretch_mono_image, stretch_color_image
from legacy.numba_utils import apply_flat_division_numba, debayer_fits_fast   # adjust names if different
from legacy.image_manager import load_image
from pro.star_alignment import StarRegistrationWorker, StarRegistrationThread, IDENTITY_2x3

class CustomDoubleSpinBox(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, minimum=0.0, maximum=10.0, initial=0.0, step=0.1,
                 suffix: str = "", parent=None):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self._value = initial
        self.suffix = suffix

        # Line edit
        self.lineEdit = QLineEdit(f"{initial:.3f}")
        self.lineEdit.setAlignment(Qt.AlignmentFlag.AlignRight)
        validator = QDoubleValidator(self.minimum, self.maximum, 3, self)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.lineEdit.setValidator(validator)
        self.lineEdit.editingFinished.connect(self.onEditingFinished)

        # Up/down buttons
        self.upButton = QToolButton(); self.upButton.setText("▲")
        self.downButton = QToolButton(); self.downButton.setText("▼")
        self.upButton.clicked.connect(self.increaseValue)
        self.downButton.clicked.connect(self.decreaseValue)

        # Buttons layout
        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.upButton)
        buttonLayout.addWidget(self.downButton)
        buttonLayout.setSpacing(0)
        buttonLayout.setContentsMargins(0, 0, 0, 0)

        # Optional suffix label
        self.suffixLabel = QLabel(self.suffix) if self.suffix else None
        if self.suffixLabel:
            self.suffixLabel.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Main layout
        mainLayout = QHBoxLayout()
        mainLayout.setSpacing(2)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(self.lineEdit)
        mainLayout.addLayout(buttonLayout)
        if self.suffixLabel:
            mainLayout.addWidget(self.suffixLabel)
        self.setLayout(mainLayout)

        self.updateButtonStates()

    def updateButtonStates(self):
        self.upButton.setEnabled(self._value < self.maximum)
        self.downButton.setEnabled(self._value > self.minimum)

    def increaseValue(self):
        self.setValue(self._value + self.step)

    def decreaseValue(self):
        self.setValue(self._value - self.step)

    def onEditingFinished(self):
        try:
            new_val = float(self.lineEdit.text())
        except ValueError:
            new_val = self._value
        self.setValue(new_val)

    def setValue(self, val: float):
        if val < self.minimum:
            val = self.minimum
        elif val > self.maximum:
            val = self.maximum
        if abs(val - self._value) > 1e-9:
            self._value = val
            self.lineEdit.setText(f"{val:.3f}")
            self.valueChanged.emit(val)
            self.updateButtonStates()

    def value(self) -> float:
        return self._value

class CustomSpinBox(QWidget):
    """
    A simple custom spin box that mimics QSpinBox functionality.
    Emits valueChanged(int) when the value changes.
    """
    valueChanged = pyqtSignal(int)

    def __init__(self, minimum=0, maximum=100, initial=0, step=1, parent=None):
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self._value = initial

        # Create a line edit to show the value.
        self.lineEdit = QLineEdit(str(initial))
        self.lineEdit.setAlignment(Qt.AlignmentFlag.AlignRight)
        # Optionally, restrict input to integers using a validator.
        
        self.lineEdit.setValidator(QIntValidator(self.minimum, self.maximum, self))
        self.lineEdit.editingFinished.connect(self.editingFinished)

        # Create up and down buttons with arrow text or icons.
        self.upButton = QToolButton()
        self.upButton.setText("▲")
        self.downButton = QToolButton()
        self.downButton.setText("▼")
        self.upButton.clicked.connect(self.increaseValue)
        self.downButton.clicked.connect(self.decreaseValue)

        # Arrange the buttons vertically.
        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.upButton)
        buttonLayout.addWidget(self.downButton)
        buttonLayout.setSpacing(0)
        buttonLayout.setContentsMargins(0, 0, 0, 0)

        # Arrange the line edit and buttons horizontally.
        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.lineEdit)
        mainLayout.addLayout(buttonLayout)
        mainLayout.setSpacing(0)
        mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(mainLayout)

        self.updateButtonStates()

    @property
    def value(self):
        return self._value

    def setValue(self, val):
        if val < self.minimum:
            val = self.minimum
        elif val > self.maximum:
            val = self.maximum
        if val != self._value:
            self._value = val
            self.lineEdit.setText(str(val))
            self.valueChanged.emit(val)
            self.updateButtonStates()

    def updateButtonStates(self):
        self.upButton.setEnabled(self._value < self.maximum)
        self.downButton.setEnabled(self._value > self.minimum)

    def increaseValue(self):
        self.setValue(self._value + self.step)

    def decreaseValue(self):
        self.setValue(self._value - self.step)

    def editingFinished(self):
        try:
            newVal = int(self.lineEdit.text())
        except ValueError:
            newVal = self._value
        self.setValue(newVal)


class LiveStackSettingsDialog(QDialog):
    """
    Combined dialog for:
      • Live‐stack parameters (bootstrap frames, σ‐clip threshold)
      • Culling thresholds (max FWHM, max eccentricity, min star count)
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Live Stack & Culling Settings")

        # — Live Stack Settings —
        # Bootstrap frames (int)
        self.bs_spin = CustomSpinBox(
            minimum=1,
            maximum=100,
            initial=parent.bootstrap_frames,
            step=1
        )
        self.bs_spin.valueChanged.connect(lambda v: None)

        # Sigma threshold (float)
        self.sigma_spin = CustomDoubleSpinBox(
            minimum=0.1,
            maximum=10.0,
            initial=parent.clip_threshold,
            step=0.1,
            suffix="σ"
        )
        self.sigma_spin.valueChanged.connect(lambda v: None)

        # — Culling Thresholds —
        # Max FWHM (float)
        self.fwhm_spin = CustomDoubleSpinBox(
            minimum=0.1,
            maximum=50.0,
            initial=parent.max_fwhm,
            step=0.1,
            suffix=" px"
        )
        self.fwhm_spin.valueChanged.connect(lambda v: None)

        # Max eccentricity (float)
        self.ecc_spin = CustomDoubleSpinBox(
            minimum=0.0,
            maximum=1.0,
            initial=parent.max_ecc,
            step=0.01
        )
        self.ecc_spin.valueChanged.connect(lambda v: None)

        # Min star count (int)
        self.star_spin = CustomSpinBox(
            minimum=0,
            maximum=5000,
            initial=parent.min_star_count,
            step=1
        )
        self.star_spin.valueChanged.connect(lambda v: None)

        # Build form layout
        form = QFormLayout()
        form.addRow("Switch to μ–σ clipping after:", self.bs_spin)
        form.addRow("Clip threshold:", self.sigma_spin)
        form.addRow(QLabel(""))  # blank row for separation
        form.addRow("Max FWHM (px):", self.fwhm_spin)
        form.addRow("Max Eccentricity:", self.ecc_spin)
        form.addRow("Min Star Count:", self.star_spin)

        self.mapping_combo = QComboBox()
        opts = ["Natural", "SHO", "HSO", "OSH", "SOH", "HOS", "OHS"]
        self.mapping_combo.addItems(opts)
        # preselect current
        idx = opts.index(parent.narrowband_mapping) \
              if parent.narrowband_mapping in opts else 0
        self.mapping_combo.setCurrentIndex(idx)
        form.addRow("Narrowband Mapping:", self.mapping_combo)

        # OK / Cancel buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        # Assemble dialog layout
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(btns)
        self.setLayout(layout)

    def getValues(self):
        """
        Returns a tuple of five values in order:
          (bootstrap_frames, clip_threshold,
           max_fwhm, max_ecc, min_star_count)
        """
        bs      = self.bs_spin.value
        sigma   = self.sigma_spin.value()
        fwhm    = self.fwhm_spin.value()
        ecc     = self.ecc_spin.value()
        stars   = self.star_spin.value
        mapping = self.mapping_combo.currentText()
        return bs, sigma, fwhm, ecc, stars, mapping



class LiveMetricsPanel(QWidget):
    """
    A simple 2×2 grid of PyQtGraph plots to show, in real time:
      [0,0] → FWHM (px) vs. frame index
      [0,1] → Eccentricity vs. frame index
      [1,0] → Star Count vs. frame index
      [1,1] → (μ–ν)/σ (∝SNR) vs. frame index
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        titles = ["FWHM (px)", "Eccentricity", "Star Count", "(μ–ν)/σ (∝SNR)"]

        layout = QVBoxLayout(self)
        grid = pg.GraphicsLayoutWidget()
        layout.addWidget(grid)

        self.plots = []
        self.scats = []
        self._data_x = [[], [], [], []]
        self._data_y = [[], [], [], []]
        self._flags  = [[], [], [], []]  # track if each point was “bad” (True) or “good” (False)

        for row in range(2):
            for col in range(2):
                pw = grid.addPlot(row=row, col=col)
                idx = row * 2 + col
                pw.setTitle(titles[idx])
                pw.showGrid(x=True, y=True, alpha=0.3)
                pw.setLabel('bottom', "Frame #")
                pw.setLabel('left', titles[idx])

                scat = pg.ScatterPlotItem(pen=pg.mkPen(None),
                                          brush=pg.mkBrush(100, 100, 255, 200),
                                          size=6)
                pw.addItem(scat)
                self.plots.append(pw)
                self.scats.append(scat)

    def add_point(self, frame_idx: int, fwhm: float, ecc: float, star_cnt: int, snr_val: float, flagged: bool):
        """
        Append one new data point to each metric.  
        If flagged == True, draw that single point in red; else blue.  
        But keep all previously-plotted points at their original colors.
        """
        values = [fwhm, ecc, star_cnt, snr_val]
        for i in range(4):
            self._data_x[i].append(frame_idx)
            self._data_y[i].append(values[i])
            self._flags[i].append(flagged)

            # Now build a brush list for *all* points up to index i,
            # coloring each point according to its own flag.
            brushes = [
                pg.mkBrush(255, 0, 0, 200) if self._flags[i][j]
                else pg.mkBrush(100, 100, 255, 200)
                for j in range(len(self._data_x[i]))
            ]

            self.scats[i].setData(
                self._data_x[i],
                self._data_y[i],
                brush=brushes,
                pen=pg.mkPen(None),
                size=6
            )

    def clear_all(self):
        """Clear data from all four plots."""
        for i in range(4):
            self._data_x[i].clear()
            self._data_y[i].clear()
            self._flags[i].clear()
            self.scats[i].clear()

class LiveMetricsWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Stack Metrics")
        self.resize(600, 400)

        layout = QVBoxLayout(self)
        self.metrics_panel = LiveMetricsPanel(self)
        layout.addWidget(self.metrics_panel)

from pro.star_metrics import measure_stars_sep

def compute_frame_star_metrics(image_2d):
    """
    Harmonized with Blink metrics:
      - SEP.Background() for back/noise
      - thresh = 7σ
      - median aggregation for FWHM & Ecc
    """
    # ensure float32 mono [0..1]
    data = np.asarray(image_2d)
    if data.ndim == 3:
        data = data.mean(axis=2)
    if data.dtype == np.uint8:
        data = data.astype(np.float32) / 255.0
    elif data.dtype == np.uint16:
        data = data.astype(np.float32) / 65535.0
    else:
        data = data.astype(np.float32, copy=False)

    star_count, fwhm, ecc = measure_stars_sep(
        data,
        thresh_sigma=7.0,
        minarea=16,
        deblend_nthresh=32,
        aggregate="median",
    )
    return star_count, fwhm, ecc

def estimate_global_snr(
    stack_image: np.ndarray,
    bkg_box_size: int = 200
) -> float:
    """
    “Hybrid” global SNR ≔ (μ_patch − median_patch) / σ_central,
    where:
      • μ_patch    and median_patch come from a small bkg_box_size×bkg_box_size patch
        centered inside the middle 50% of the image.
      • σ_central  is the standard deviation computed over the entire “middle 50%” region.

    Steps:
      1) Collapse to grayscale (H×W) if needed.
      2) Identify the middle 50% rectangle of the image.
      3) Within that, center a patch of size up to bkg_box_size×bkg_box_size.
      4) Compute μ_patch = mean(patch), median_patch = median(patch).
      5) Compute σ_central = std(middle50_region).
      6) If σ_central ≤ 0, return 0. Otherwise return (μ_patch − median_patch) / σ_central.
    """

    # 1) Collapse to simple 2D float array (grayscale)
    if stack_image.ndim == 3 and stack_image.shape[2] == 3:
        # RGB → grayscale by averaging channels
        gray = stack_image.mean(axis=2).astype(np.float32)
    else:
        # Already mono: just cast to float32
        gray = stack_image.astype(np.float32)

    H, W = gray.shape

    # 2) Compute coordinates of the “middle 50%” rectangle
    y0 = H // 4
    y1 = y0 + (H // 2)
    x0 = W // 4
    x1 = x0 + (W // 2)

    # Extract that central50 region as a view (no copy)
    central50 = gray[y0:y1, x0:x1]

    # 3) Within that central50, choose a patch of up to bkg_box_size×bkg_box_size, centered
    center_h = (y1 - y0)
    center_w = (x1 - x0)

    # Clamp box size so it does not exceed central50 dimensions
    box_h = min(bkg_box_size, center_h)
    box_w = min(bkg_box_size, center_w)

    # Compute top-left corner of that patch so it’s centered in central50
    cy0 = y0 + (center_h - box_h) // 2
    cx0 = x0 + (center_w - box_w) // 2

    patch = gray[cy0 : cy0 + box_h, cx0 : cx0 + box_w]

    # 4) Compute patch statistics
    mu_patch = float(np.mean(patch))
    med_patch = float(np.median(patch))
    min_patch = float(np.min(patch))

    # 5) Compute σ over the entire central50 region
    sigma_central = float(np.std(central50))
    if sigma_central <= 0.0:
        return 0.0

    nu = med_patch - 3.0 * sigma_central * med_patch

    # 6) Return (mean − nu) / σ
    return (mu_patch - nu) / sigma_central
    #return (mu_patch) / sigma_central

class LiveStackWindow(QDialog):
    def __init__(self, parent=None, doc_manager=None, wrench_path=None, spinner_path=None):
        super().__init__(parent)
        self.parent = parent
        self._docman = doc_manager
        self._wrench_path = wrench_path
        self._spinner_path = spinner_path
        self.setWindowTitle("Live Stacking")
        self.resize(900, 600)

        # ─── State Variables ─────────────────────────────────────
        self.watch_folder = None
        self.processed_files = set()
        self.master_dark = None
        self.master_flat = None
        self.master_flats  = {}

        self.filter_stacks = {}       # key → np.ndarray (float32)
        self.filter_counts = {}       # key → int
        self.filter_buffers  = {}  # key → list of bootstrap frames [H×W arrays]
        self.filter_mus      = {}  # key → µ array after bootstrap (H×W)
        self.filter_m2s      = {}  # key → M2 array after bootstrap (H×W)

        self.cull_folder = None

        self.is_running = False
        self.frame_count = 0
        self.current_stack = None

        # ── Load persisted settings ───────────────────────────────
        s = QSettings()
        self.bootstrap_frames    = s.value("LiveStack/bootstrap_frames",    24,     type=int)
        self.clip_threshold      = s.value("LiveStack/clip_threshold",      3.5,    type=float)
        self.max_fwhm            = s.value("LiveStack/max_fwhm",            15.0,   type=float)
        self.max_ecc             = s.value("LiveStack/max_ecc",             0.9,    type=float)
        self.min_star_count      = s.value("LiveStack/min_star_count",      5,      type=int)
        self.narrowband_mapping  = s.value("LiveStack/narrowband_mapping",  "Natural", type=str)
        self.star_trail_mode = s.value("LiveStack/star_trail_mode", False, type=bool)


        self.total_exposure = 0.0  # seconds
        self.exposure_label = QLabel("Total Exp: 00:00:00")
        self.exposure_label.setStyleSheet("color: #cccccc; font-weight: bold;")

        self.brightness = 0.0   # [-1.0..+1.0]
        self.contrast   = 1.0   # [0.1..3.0]


        self._buffer = []    # store up to bootstrap_frames normalized frames
        self._mu = None      # per-pixel mean (after bootstrap)
        self._m2 = None      # per-pixel sum of squares differences (for Welford)

        # ─── Create Separate Metrics Window (initially hidden) ─────
        # We do NOT embed this in the stacking dialog’s layout!
        self.metrics_window = LiveMetricsWindow(None)
        self.metrics_window.hide()

        # ─── UI ELEMENTS FOR STACKING DIALOG ───────────────────────
        # 1) Folder selection
        self.folder_label = QLabel("Folder: (none)")
        self.select_folder_btn = QPushButton("Select Folder…")
        self.select_folder_btn.clicked.connect(self.select_folder)

        # 2) Load master dark/flat
        self.load_darks_btn = QPushButton("Load Master Dark…")
        self.load_darks_btn.clicked.connect(self.load_masters)
        self.load_flats_btn = QPushButton("Load Master Flat…")
        self.load_flats_btn.clicked.connect(self.load_masters)
        self.load_filter_flats_btn = QPushButton("Load MonoFilter Flats…")
        self.load_filter_flats_btn.clicked.connect(self.load_filter_flats)        

        # 2b) Cull folder selection
        self.cull_folder_label = QLabel("Cull Folder: (none)")
        self.select_cull_btn = QPushButton("Select Cull Folder…")
        self.select_cull_btn.clicked.connect(self.select_cull_folder)

        self.dark_status_label = QLabel("Dark: ❌")
        self.flat_status_label = QLabel("Flat: ❌")
        for lbl in (self.dark_status_label, self.flat_status_label):
            lbl.setStyleSheet("color: #cccccc; font-weight: bold;")
        # 3) “Process & Monitor” / “Monitor Only” / “Stop” / “Reset”
        self.mono_color_checkbox = QCheckBox("Mono → Color Stacking")
        self.mono_color_checkbox.setToolTip(
            "When checked, bucket mono frames by FILTER and composite R, G, B, Ha, OIII, SII."
        )
        # **Connect the toggled(bool) signal** before we ever call it
        self.mono_color_checkbox.toggled.connect(self._on_mono_color_toggled)

        # ** new: Star-Trail mode checkbox **
        self.star_trail_checkbox = QCheckBox("★★ Star-Trail Mode ★★")
        self.star_trail_checkbox.setChecked(self.star_trail_mode)
        self.star_trail_checkbox.setToolTip("If checked, build a max-value trail instead of a running stack")
        self.star_trail_checkbox.toggled.connect(self._on_star_trail_toggled)

        self.process_and_monitor_btn = QPushButton("Process && Monitor")
        self.process_and_monitor_btn.clicked.connect(self.start_and_process)
        self.monitor_only_btn = QPushButton("Monitor Only")
        self.monitor_only_btn.clicked.connect(self.start_monitor_only)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_live)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_live)

        self.frame_count_label = QLabel("Frames: 0")

        # 4) Live‐stack preview area (QGraphicsView)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self._did_initial_fit = False

        # 5) Zoom toolbar + Settings icon
        tb = QToolBar()

        zi = QAction(QIcon.fromTheme("zoom-in"), "Zoom In", self)
        zo = QAction(QIcon.fromTheme("zoom-out"), "Zoom Out", self)
        fit = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit to Window", self)

        tb.addAction(zi)
        tb.addAction(zo)
        tb.addAction(fit)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        # — Replace the QAction “wrench” with a styled QToolButton —
        self.wrench_button = QToolButton()
        self.wrench_button.setIcon(QIcon(self._wrench_path))
        self.wrench_button.setToolTip("Settings")
        # Apply your stylesheet to the QToolButton
        self.wrench_button.setStyleSheet("""
            QToolButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #FF6347;
            }
        """)
        # Connect the clicked signal to open_settings()
        self.wrench_button.clicked.connect(self.open_settings)

        # Add the styled QToolButton into the toolbar
        tb.addWidget(self.wrench_button)

        zi.triggered.connect(self.zoom_in)
        zo.triggered.connect(self.zoom_out)
        fit.triggered.connect(self.fit_to_window)


        # 6) Brightness & Contrast sliders
        bright_slider = QSlider(Qt.Orientation.Horizontal)
        bright_slider.setRange(-100, 100)
        bright_slider.setValue(0)
        bright_slider.setToolTip("Brightness")
        bright_slider.valueChanged.connect(self.on_brightness_changed)

        contrast_slider = QSlider(Qt.Orientation.Horizontal)
        contrast_slider.setRange(10, 1000)
        contrast_slider.setValue(100)
        contrast_slider.setToolTip("Contrast")
        contrast_slider.valueChanged.connect(self.on_contrast_changed)

        bc_layout = QHBoxLayout()
        bc_layout.addWidget(QLabel("Brightness"))
        bc_layout.addWidget(bright_slider)
        bc_layout.addWidget(QLabel("Contrast"))
        bc_layout.addWidget(contrast_slider)

        # 7) “Send to Slot” button
        open_btn = QPushButton("Open in New View ▶")
        open_btn.clicked.connect(self.send_to_new_view)

        # 8) “Show Metrics” button
        self.show_metrics_btn = QPushButton("Show Metrics")
        self.show_metrics_btn.clicked.connect(self.show_metrics_window)

        # ─── ASSEMBLE MAIN LAYOUT (exactly one setLayout call!) ─────
        main_layout = QVBoxLayout()

        # A) Top‐row controls
        controls = QHBoxLayout()
        controls.addWidget(self.select_folder_btn)
        controls.addWidget(self.load_darks_btn)
        controls.addWidget(self.load_flats_btn)
        controls.addWidget(self.load_filter_flats_btn)
        controls.addWidget(self.select_cull_btn)
        controls.addStretch()
        controls.addWidget(self.mono_color_checkbox)
        controls.addWidget(self.star_trail_checkbox)
        controls.addWidget(self.process_and_monitor_btn)
        controls.addWidget(self.monitor_only_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.reset_btn)
        main_layout.addLayout(controls)

        # B) Status line: folder label + frame count
        status_line = QHBoxLayout()
        status_line.addWidget(self.folder_label)
        status_line.addWidget(self.dark_status_label)
        status_line.addWidget(self.flat_status_label)
        status_line.addWidget(self.cull_folder_label)
        status_line.addStretch()        
        status_line.addWidget(self.frame_count_label)
        status_line.addWidget(self.exposure_label)
        main_layout.addLayout(status_line)

        # C) Zoom toolbar
        main_layout.addWidget(tb)

        # D) Show Metrics button (separate window)
        main_layout.addWidget(self.show_metrics_btn)

        # E) Live‐stack preview area
        main_layout.addWidget(self.view)

        # F) Brightness/Contrast sliders
        main_layout.addLayout(bc_layout)

        # G) “Send to Slot” + mode/idle labels
        main_layout.addWidget(open_btn)
        self.mode_label = QLabel("Mode: Linear Average")
        self.mode_label.setStyleSheet("color: #a0a0a0;")
        main_layout.addWidget(self.mode_label)
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #a0a0a0;")
        main_layout.addWidget(self.status_label)

        # Finalize
        self.setLayout(main_layout)

        # Timer for polling new files
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(1500)
        self.poll_timer.timeout.connect(self.check_for_new_frames)
        self._on_mono_color_toggled(self.mono_color_checkbox.isChecked())


    # ─────────────────────────────────────────────────────────────────────────
    def _on_star_trail_toggled(self, checked: bool):
        """Enable/disable star-trail mode."""
        self.star_trail_mode = checked
        QSettings().setValue("LiveStack/star_trail_mode", checked)
        self.mode_label.setText("Mode: Star-Trail" if checked else "Mode: Linear Average")
        # if you want, disable mono/color checkbox when star-trail is on:
        self.mono_color_checkbox.setEnabled(not checked)

    def _on_mono_color_toggled(self, checked: bool):
        self.mono_color_mode = checked
        self.filter_stacks.clear()
        self.filter_counts.clear()

        msg = "Enabled" if checked else "Disabled"
        self.status_label.setText(f"Mono→Color Mode {msg}")

    def show_metrics_window(self):
        """Pop up the separate metrics window (never embed it here)."""
        self.metrics_window.show()
        self.metrics_window.raise_()


    def select_cull_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Cull Folder")
        if folder:
            self.cull_folder = folder
            self.cull_folder_label.setText(f"Cull: {os.path.basename(folder)}")

    def _cull_frame(self, path: str):
        """
        Move a flagged frame into the cull folder (if set), 
        or just update the status label if not.
        """
        name = os.path.basename(path)
        if self.cull_folder:
            try:
                os.makedirs(self.cull_folder, exist_ok=True)
                dst = os.path.join(self.cull_folder, name)
                shutil.move(path, dst)
                self.status_label.setText(f"⚠ Culled {name} → {self.cull_folder}")
            except Exception:
                self.status_label.setText(f"⚠ Failed to cull {name}")
        else:
            self.status_label.setText(f"⚠ Flagged (not stacked): {name}")
        QApplication.processEvents()

    def open_settings(self):
        dlg = LiveStackSettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            bs, sigma, fwhm, ecc, stars, mapping = dlg.getValues()

            # 1) Persist into QSettings
            s = QSettings()
            s.setValue("LiveStack/bootstrap_frames",   bs)
            s.setValue("LiveStack/clip_threshold",     sigma)
            s.setValue("LiveStack/max_fwhm",           fwhm)
            s.setValue("LiveStack/max_ecc",            ecc)
            s.setValue("LiveStack/min_star_count",     stars)
            s.setValue("LiveStack/narrowband_mapping", mapping)

            # 2) Apply to this live‐stack session
            self.bootstrap_frames   = bs
            self.clip_threshold     = sigma
            self.max_fwhm           = fwhm
            self.max_ecc            = ecc
            self.min_star_count     = stars
            self.narrowband_mapping = mapping

            self.status_label.setText(
                f"↺ Settings saved: BS={bs}, σ={sigma:.1f}, "
                f"FWHM≤{fwhm:.1f}, ECC≤{ecc:.2f}, Stars≥{stars}, "
                f"Mapping={mapping}"
            )
            QApplication.processEvents()

    def zoom_in(self):
        self.view.scale(1.2, 1.2)

    def zoom_out(self):
        self.view.scale(1/1.2, 1/1.2)

    def fit_to_window(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # — Brightness / Contrast —

    def _refresh_preview(self):
        """
        Recompute the current preview array (stack vs. composite)
        and call update_preview on it.
        """
        if self.mono_color_mode:
            # build the composite from filter_stacks
            preview = self._build_color_composite()
        else:
            # use the normal running stack
            preview = self.current_stack

        if preview is not None:
            self.update_preview(preview)

    def on_brightness_changed(self, val: int):
        self.brightness = val / 100.0  # map to [-1,1]
        self._refresh_preview()

    def on_contrast_changed(self, val: int):
        self.contrast = val / 100.0   # map to [0.1,10.0]
        self._refresh_preview()

    # — Sending out —

    def send_to_new_view(self):
        """
        Create a brand-new document/view from the current live stack or composite.
        Prefers using doc_manager's native numpy-open methods; otherwise falls back
        to writing a temp TIFF and asking the host window to open it.
        """
        # pick what to export
        if self.mono_color_mode:
            img = self._build_color_composite()
        else:
            img = self.current_stack

        if img is None:
            self.status_label.setText("⚠ Nothing to open")
            return

        # ensure float32 in [0,1]
        img = np.clip(img, 0.0, 1.0).astype(np.float32)

        title = f"LiveStack_{_dt.datetime.now():%Y%m%d_%H%M%S}_{self.frame_count}f"
        metadata = {"source": "LiveStack", "frames_stacked": int(self.frame_count)}

        # 1) Try doc_manager direct numpy APIs (several common names)
        dm = self._docman
        if dm is not None:
            for name in ("create_numpy_document",
                        "new_document_from_numpy",
                        "open_numpy",
                        "open_array",
                        "open_image_array",
                        "add_document_from_array"):
                fn = getattr(dm, name, None)
                if callable(fn):
                    try:
                        fn(img, title=title, metadata=metadata)
                        self.status_label.setText(f"Opened new view: {title}")
                        return
                    except TypeError:
                        # some variants might not accept title/metadata
                        try:
                            fn(img)
                            self.status_label.setText(f"Opened new view: {title}")
                            return
                        except Exception:
                            pass
                    except Exception:
                        pass

        # 2) Fallback: write a temp 16-bit TIFF and ask main window to open it
        try:
            import tifffile as tiff
            tmp = tempfile.NamedTemporaryFile(suffix=".tiff", delete=False)
            tmp_path = tmp.name
            tmp.close()
            # export as 16-bit so it's friendly to the rest of the app
            arr16 = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
            tiff.imwrite(tmp_path, arr16)

            host = self.parent
            for name in ("open_files", "open_file", "load_paths", "load_path"):
                fn = getattr(host, name, None)
                if callable(fn):
                    try:
                        fn([tmp_path]) if fn.__code__.co_argcount != 2 else fn(tmp_path)
                        self.status_label.setText(f"Opened new view from temp: {os.path.basename(tmp_path)}")
                        return
                    except Exception:
                        pass

            # ultimate fallback: let the user know where it went
            QMessageBox.information(self, "Saved Temp Image",
                                    f"Saved temp: {tmp_path}\nOpen it from File → Open.")
        except Exception as e:
            QMessageBox.warning(self, "Open Failed",
                                f"Could not open in new view:\n{e}")


    # ── New helper: map header["FILTER"] to a single letter key
    def _get_filter_key(self, header):
        """
        Map a FITS header FILTER string to one of:
        'L' (luminance),
        'R','G','B',
        'H' (H-alpha),
        'O' (OIII),
        'S' (SII),
        or return None if it doesn’t match.
        """
        raw = header.get('FILTER', '')
        fn = raw.strip().upper()
        if not fn:
            return None

        # H-alpha
        if fn in ('H', 'HA', 'HALPHA', 'H-ALPHA'):
            return 'H'
        # OIII
        if fn in ('O', 'O3', 'OIII'):
            return 'O'
        # SII
        if fn in ('S', 'S2', 'SII'):
            return 'S'
        # Red
        if fn in ('R', 'RED', 'RD'):
            return 'R'
        # Green
        if fn in ('G', 'GREEN', 'GRN'):
            return 'G'
        # Blue
        if fn in ('B', 'BLUE', 'BL'):
            return 'B'
        # Luminance
        if fn in ('L', 'LUM', 'LUMI', 'LUMINANCE'):
            return 'L'

        return None

    # ── New helper: stack a single mono frame under filter key
    def _stack_mono_channel(self, key, img, delta=None):
        # img: 2D or 3D array; we convert to 2D mono always
        mono = img if img.ndim==2 else np.mean(img,axis=2)
        # align if you need (use same logic as color branch)
        if hasattr(self, 'reference_image_2d'):
            d = delta or StarRegistrationWorker.compute_affine_transform_astroalign(
                        mono, self.reference_image_2d)
            if d is not None:
                mono = StarRegistrationThread.apply_affine_transform_static(mono, d)
        # normalize
        norm = stretch_mono_image(mono, target_median=0.3)
        # first frame?
        if key not in self.filter_stacks:
            self.filter_stacks[key] = norm.copy()
            self.filter_counts[key] = 1
            # set reference on first good channel frame
            if not hasattr(self, 'reference_image_2d'):
                self.reference_image_2d = norm.copy()
        else:
            cnt = self.filter_counts[key]
            self.filter_stacks[key] = (cnt/self.filter_counts[key]+1)*self.filter_stacks[key] \
                                      + (1.0/(cnt+1))*norm
            self.filter_counts[key] += 1

    # ── New helper: build an RGB preview from whatever channels we have
    def _build_color_composite(self):
        """
        Composite filters into an RGB preview according to self.narrowband_mapping:

        • "Natural":
            – If SII present:
                R = 0.5*(Ha + SII)
                G = 0.5*(SII + OIII)
                B = OIII
            – Elif any R/G/B loaded:
                R = R_filter
                G = G_filter + OIII
                B = B_filter + OIII
            – Else (no SII, no R/G/B):
                R = Ha
                G = OIII
                B = OIII

        • Any 3-letter code (e.g. "SHO", "OHS"):
            R = filter_stacks[mapping[0]]
            G = filter_stacks[mapping[1]]
            B = filter_stacks[mapping[2]]

        Missing channels default to zero.
        """
        # 1) Determine H, W
        if self.filter_stacks:
            first = next(iter(self.filter_stacks.values()))
            H, W = first.shape
        elif getattr(self, 'current_stack', None) is not None:
            H, W = self.current_stack.shape[:2]
        else:
            return None

        # helper: get stack or zeros
        def getf(k):
            return self.filter_stacks.get(k, np.zeros((H, W), np.float32))

        mode = self.narrowband_mapping.upper()
        if mode == "NATURAL":
            Ha = getf('H')
            O3 = getf('O')
            S2 = self.filter_stacks.get('S', None)
            Rf = self.filter_stacks.get('R', None)
            Gf = self.filter_stacks.get('G', None)
            Bf = self.filter_stacks.get('B', None)

            if S2 is not None:
                # narrowband SII branch
                R = 0.5 * (Ha + S2)
                G = 0.5 * (S2 + O3)
                B = O3.copy()

            elif any(x is not None for x in (Rf, Gf, Bf)):
                # broadband branch: Rf/Gf/Bf with OIII boost
                R = Rf if Rf is not None else np.zeros((H, W), np.float32)
                G = (Gf if Gf is not None else np.zeros((H, W), np.float32)) + O3
                B = (Bf if Bf is not None else np.zeros((H, W), np.float32)) + O3

            else:
                # fallback HOO
                R = Ha
                G = O3
                B = O3

        else:
            # direct mapping: e.g. "SHO" → R=S, G=H, B=O
            letters = list(mode)
            if len(letters) != 3 or any(l not in ("S","H","O") for l in letters):
                # invalid code → fallback to natural
                return self._build_color_composite.__wrapped__(self)

            R = getf(letters[0])
            G = getf(letters[1])
            B = getf(letters[2])

        return np.stack([R, G, B], axis=2)


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Watch")
        if folder:
            self.watch_folder = folder
            self.folder_label.setText(f"Folder: {os.path.basename(folder)}")

    def load_masters(self):
        """
        When the user picks “Load Master Dark…” or “Load Master Flat…”, we load exactly one file
        (the first in the dialog).  We simply store it in `self.master_dark` or `self.master_flat`,
        but we also check its dimensions against any existing master so that the user can’t load
        a 2D flat while the dark is 3D (for example).
        """
        sender = self.sender()
        dlg = QFileDialog(self, "Select Master Files",
                         filter="FITS TIFF or XISF (*.fit *.fits *.tif *.tiff *.xisf)")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if not dlg.exec():
            return

        chosen = dlg.selectedFiles()[0]
        img, hdr, bit_depth, is_mono = load_image(chosen)
        if img is None:
            QMessageBox.warning(self, "Load Error",
                                f"Failed to load master file:\n{chosen}")
            return

        # Convert everything to float32 for consistency
        img = img.astype(np.float32)

        if "Dark" in sender.text():
            # If a flat is already loaded, ensure shape‐compatibility
            if self.master_flat is not None:
                if not self._shapes_compatible(master=img, other=self.master_flat):
                    QMessageBox.warning(
                        self, "Shape Mismatch",
                        "Cannot load this master dark: it has incompatible shape "
                        "vs. the already‐loaded master flat."
                    )
                    return

            self.master_dark = img
            self.dark_status_label.setText("Dark: ✅")
            self.dark_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")            
            QMessageBox.information(
                self, "Master Dark Loaded",
                f"Loaded master dark:\n{os.path.basename(chosen)}"
            )
        else:
            # "Flat" was clicked
            if self.master_dark is not None:
                if not self._shapes_compatible(master=self.master_dark, other=img):
                    QMessageBox.warning(
                        self, "Shape Mismatch",
                        "Cannot load this master flat: it has incompatible shape "
                        "vs. the already‐loaded master dark."
                    )
                    return

            self.master_flat = img
            self.flat_status_label.setText("Flat: ✅")
            self.flat_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")            
            QMessageBox.information(
                self, "Master Flat Loaded",
                f"Loaded master flat:\n{os.path.basename(chosen)}"
            )

    def load_filter_flats(self):
        """
        Let the user pick one or more flat files.
        We try to read the FITS header FILTER key to decide which filter
        each flat belongs to; otherwise fall back to the filename.
        """
        dlg = QFileDialog(self, "Select Filter Flats",
                          filter="FITS or TIFF (*.fit *.fits *.tif *.tiff)")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if not dlg.exec():
            return

        files = dlg.selectedFiles()
        loaded = []
        for path in files:
            img, hdr, bit_depth, is_mono = load_image(path)
            if img is None:
                continue
            # guess filter key from header, else from filename
            key = None
            if hdr and hdr.get("FILTER"):
                key = self._get_filter_key(hdr)
            if not key:
                # fallback: basename before extension
                key = os.path.splitext(os.path.basename(path))[0]

            # store it
            self.master_flats[key] = img.astype(np.float32)
            loaded.append(key)

        # update the flat status label to list loaded filters
        if loaded:
            names = ", ".join(loaded)
            self.flat_status_label.setText(f"Flats: {names}")
            self.flat_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")
            QMessageBox.information(
                self, "Filter Flats Loaded",
                f"Loaded flats for filters: {names}"
            )
        else:
            QMessageBox.warning(self, "No Flats Loaded",
                                "No flats could be loaded.")

    def _shapes_compatible(self, master: np.ndarray, other: np.ndarray) -> bool:
        """
        Return True if `master` and `other` can be used together in calibration:
          - Exactly the same shape, OR
          - master is 2D (H×W) and other is 3D (H×W×3), OR
          - vice versa.
        """
        if master.shape == other.shape:
            return True

        # If one is 2D and the other is H×W×3, check the first two dims
        if master.ndim == 2 and other.ndim == 3 and other.shape[:2] == master.shape:
            return True
        if other.ndim == 2 and master.ndim == 3 and master.shape[:2] == other.shape:
            return True

        return False

    def _average_images(self, paths):
        # stub: load each via load_image(), convert to float32, accumulate & divide
        return None

    def _normalized_average(self, paths):
        # stub: load each, divide by its mean, average them, then renormalize
        return None

    def start_and_process(self):
        """Process everything currently in folder, then begin monitoring."""
        if not self.watch_folder:
            self.status_label.setText("❗ No folder selected")
            return
        # Clear any old record so existing files are re-processed
        self.processed_files.clear()
        # Process all current files once
        self.check_for_new_frames()
        # Now start monitoring
        self.is_running = True
        self.poll_timer.start()
        self.status_label.setText(f"▶ Processing & Monitoring: {os.path.basename(self.watch_folder)}")

    def start_monitor_only(self):
        """Mark existing files as seen and only process new arrivals."""
        if not self.watch_folder:
            self.status_label.setText("❗ No folder selected")
            return
        # Populate processed_files with all existing files so they won't be re-processed
        exts = (
            "*.fit", "*.fits", "*.tif", "*.tiff",
            "*.cr2", "*.cr3", "*.nef", "*.arw",
            "*.dng", "*.orf", "*.rw2", "*.pef", "*.xisf", "*.png", "*.jpg", "*.jpeg"
        )
        all_paths = []
        for ext in exts:
            all_paths += glob.glob(os.path.join(self.watch_folder, ext))
        self.processed_files = set(all_paths)

        # Start monitoring
        self.is_running = True
        self.poll_timer.start()
        self.status_label.setText(f"▶ Monitoring Only: {os.path.basename(self.watch_folder)}")

    def start_live(self):
        if not self.watch_folder:
            self.status_label.setText("❗ No folder selected")
            return
        self.is_running = True
        self.poll_timer.start()
        self.status_label.setText(f"▶ Monitoring: {os.path.basename(self.watch_folder)}")
        self.mode_label.setText("Mode: Linear Average")

    def stop_live(self):
        if self.is_running:
            self.is_running = False
            self.poll_timer.stop()
            self.status_label.setText("■ Stopped")
        else:
            self.status_label.setText("■ Already stopped")

    def reset_live(self):
        if self.is_running:
            self.is_running = False
            self.poll_timer.stop()
            self.status_label.setText("■ Stopped")
        else:
            self.status_label.setText("■ Already stopped")

        # Clear all state
        self.processed_files.clear()
        self.frame_count = 0
        self.current_stack = None

        self.total_exposure = 0.0
        self.exposure_label.setText("Total Exp: 00:00:00")

        self.filter_stacks.clear()
        self.filter_counts.clear()
        self.filter_buffers.clear()
        self.filter_mus.clear()
        self.filter_m2s.clear()

        if hasattr(self, 'reference_image_2d'):
            del self.reference_image_2d

        # Re-initialize bootstrapping stats
        self._buffer = []
        self._mu = None
        self._m2 = None

        # NEW: clear the metrics panel
        self.metrics_window.metrics_panel.clear_all()

        # Update labels
        self.frame_count_label.setText("Frames: 0")
        self.status_label.setText("↺ Reset")
        self.mode_label.setText("Mode: Linear Average")

        # Clear the displayed image
        self.pixmap_item.setPixmap(QPixmap())

        # Reset zoom/pan fit flag
        self._did_initial_fit = False
        #self.master_dark = None
        #self.master_flat = None
        #self.dark_status_label.setText("Dark: ❌")
        #self.flat_status_label.setText("Flat: ❌")
        #self.dark_status_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        #self.flat_status_label.setStyleSheet("color: #cccccc; font-weight: bold;")        




    def check_for_new_frames(self):
        if not self.is_running or not self.watch_folder:
            return

        # build the list of files with supported extensions
        exts = (
            "*.fit", "*.fits", "*.tif", "*.tiff",
            "*.cr2", "*.cr3", "*.nef", "*.arw",
            "*.dng", "*.orf", "*.rw2", "*.pef", "*.xisf",
            "*.png", "*.jpg", "*.jpeg"
        )
        all_paths = []
        for ext in exts:
            all_paths += glob.glob(os.path.join(self.watch_folder, ext))

        # only pick the ones we haven’t seen yet
        new = [p for p in sorted(all_paths) if p not in self.processed_files]
        if not new:
            return

        # update status
        first = os.path.basename(new[0])
        self.status_label.setText(f"➜ New frame: {first}")

        for path in new:
            self.processed_files.add(path)
            self.process_frame(path)

    def process_frame(self, path):
        # if star-trail mode is on, bypass the normal pipeline entirely:
        if self.star_trail_mode:
            return self._process_star_trail(path)
                
        # 1) Load
        # ─── 1) RAW‐file check ────────────────────────────────────────────
        lower = path.lower()
        raw_exts = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef')
        if lower.endswith(raw_exts):
            # Attempt to decode using rawpy
            try:
                with rawpy.imread(path) as raw:
                    # Postprocess into an 8‐bit RGB array
                    # (you could tweak postprocess params if desired)
                    img_rgb8 = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=True,
                        output_bps=16
                    )  # shape (H, W, 3), dtype=uint8

                # Convert to float32 [0..1] so it matches load_image() behavior
                img = img_rgb8.astype(np.float32) / 65535.0

                # Build a minimal FITS header and attempt to extract EXIF tags
                header = fits.Header()
                header["SIMPLE"] = True
                header["BITPIX"] = 16
                header["CREATOR"] = "LiveStack(RAW)"
                header["IMAGETYP"] = "RAW"
                # Default EXPTIME/ISO/DATE-OBS in case EXIF fails
                header["EXPTIME"] = "Unknown"
                header["ISO"]     = "Unknown"
                header["DATE-OBS"] = "Unknown"

                try:
                    with open(path, 'rb') as f:
                        tags = exifread.process_file(f, details=False)
                    # EXIF: ExposureTime
                    exp_tag = tags.get("EXIF ExposureTime") or tags.get("EXIF ShutterSpeedValue")
                    if exp_tag:
                        exp_str = str(exp_tag.values)
                        if '/' in exp_str:
                            top, bot = exp_str.split('/', 1)
                            header["EXPTIME"] = (float(top)/float(bot), "Exposure Time (s)")
                        else:
                            header["EXPTIME"] = (float(exp_str), "Exposure Time (s)")
                    # ISO
                    iso_tag = tags.get("EXIF ISOSpeedRatings")
                    if iso_tag:
                        header["ISO"] = str(iso_tag.values)
                    # Date/time original
                    date_obs = tags.get("EXIF DateTimeOriginal")
                    if date_obs:
                        header["DATE-OBS"] = str(date_obs.values)
                except Exception:
                    # If EXIF parsing fails, just leave defaults
                    pass

                bit_depth = 16
                is_mono = False

            except Exception as e:
                # If rawpy fails, bail out early
                self.status_label.setText(f"⚠ Failed to decode RAW: {os.path.basename(path)}")
                QApplication.processEvents()
                return

        else:
            # ─── 2) Not RAW → call your existing load_image()
            img, header, bit_depth, is_mono = load_image(path)
            if img is None:
                self.status_label.setText(f"⚠ Failed to load {os.path.basename(path)}")
                QApplication.processEvents()
                return

        # ——— 2) CALIBRATION (once) ————————————————————————
        # ——— 2a) DETECT MONO→COLOR MODE ————————————————————
        mono_key = None
        if self.mono_color_mode and is_mono and header.get('FILTER') and 'BAYERPAT' not in header:
            mono_key = self._get_filter_key(header)

        # ——— 2b) CALIBRATION (once) ————————————————————————
        if self.master_dark is not None:
            img = img.astype(np.float32) - self.master_dark
        # prefer per-filter flat if we’re in mono→color and have one
        if mono_key and mono_key in self.master_flats:
            img = apply_flat_division_numba(img, self.master_flats[mono_key])
        elif self.master_flat is not None:
            img = apply_flat_division_numba(img, self.master_flat)

        # ——— 3) DEBAYER if BAYERPAT ——————————————————————
        if is_mono and header.get('BAYERPAT'):
            pat = header['BAYERPAT'][0] if isinstance(header['BAYERPAT'], tuple) else header['BAYERPAT']
            img = debayer_fits_fast(img, pat)
            is_mono = False

        # ——— 5) PROMOTION TO 3-CHANNEL if NOT in mono-mode —————
        if mono_key is None and img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        # ——— 6) BUILD PLANE for alignment & metrics —————————
        plane = img if (mono_key and img.ndim == 2) else np.mean(img, axis=2)

        # ——— 7) ALIGN to reference_image_2d ——————————————————
        if hasattr(self, 'reference_image_2d'):
            delta = StarRegistrationWorker.compute_affine_transform_astroalign(
                plane, self.reference_image_2d
            )
            if delta is None:
                delta = IDENTITY_2x3
            # apply to full img (if color) and to plane
            if mono_key is None:
                img = StarRegistrationThread.apply_affine_transform_static(img, delta)
            plane = StarRegistrationThread.apply_affine_transform_static(
                plane if plane.ndim == 2 else plane[:, :, None], delta
            ).squeeze()

        # ——— 8) NORMALIZE —————————————————————————————
        if mono_key:
            norm_plane = stretch_mono_image(plane, target_median=0.3)
            norm_color = None
        else:
            norm_color = stretch_color_image(img, target_median=0.3, linked=False)
            norm_plane = np.mean(norm_color, axis=2)

        # ——— 9) METRICS & SNR —————————————————————————
        sc, fwhm, ecc = compute_frame_star_metrics(norm_plane)
        # instead, use the cumulative stack (or composite) for SNR:
        if mono_key:
            # once we have any filter_stacks, build the composite;
            # fall back to this frame’s plane if it’s the first one
            if self.filter_stacks:
                stack_img = self._build_color_composite()
            else:
                stack_img = norm_plane
        else:
            # for color‐only, use the running‐average stack once it exists,
            # else fall back to this frame’s normalized color
            if self.current_stack is not None:
                stack_img = self.current_stack
            else:
                stack_img = norm_color
        snr_val = estimate_global_snr(stack_img)

        # ——— 10) CULLING? ————————————————————————————
        flagged = (
            (fwhm > self.max_fwhm) or
            (ecc > self.max_ecc)     or
            (sc < self.min_star_count)
        )
        if flagged:
            self._cull_frame(path)
            self.metrics_window.metrics_panel.add_point(
                self.frame_count + 1, fwhm, ecc, sc, snr_val, True
            )
            return

        # ─── 11) FIRST-FRAME INITIALIZATION ──────────────────────────────
        if self.frame_count == 0:
            # set reference on the very first good frame
            self.reference_image_2d = norm_plane.copy()
            self.frame_count = 1
            self.frame_count_label.setText("Frames: 1")
            # always start in linear‐average mode
            if mono_key:
                self.mode_label.setText(f"Mode: Linear Average ({mono_key})")
                self.status_label.setText(f"Started {mono_key}-filter linear stack")
            else:
                self.mode_label.setText("Mode: Linear Average")
                self.status_label.setText("Started linear stack")
            QApplication.processEvents()

            if mono_key:
                # start the filter stack
                self.filter_stacks[mono_key]  = norm_plane.copy()
                self.filter_counts[mono_key]  = 1
                self.filter_buffers[mono_key] = [norm_plane.copy()]
            else:
                # start the normal running stack
                self.current_stack = norm_color.copy()
                self._buffer = [norm_color.copy()]
            # ─── accumulate exposure ─────────────────────
            exp_val = header.get("EXPOSURE", header.get("EXPTIME", None))
            if exp_val is not None:
                try:
                    secs = float(exp_val)
                    self.total_exposure += secs
                    hrs  = int(self.total_exposure // 3600)
                    mins = int((self.total_exposure % 3600) // 60)
                    secs_rem = int(self.total_exposure % 60)
                    self.exposure_label.setText(
                        f"Total Exp: {hrs:02d}:{mins:02d}:{secs_rem:02d}"
                    )
                except:
                    pass
            QApplication.processEvents()


        else:
            # ─── 12) RUNNING–AVERAGE or CLIP-σ UPDATE ────────────────────
            if mono_key is None:
                # — Color-only stacking —
                if self.frame_count < self.bootstrap_frames:
                    # 12a) Linear bootstrap
                    n = self.frame_count + 1
                    self.current_stack = (
                        (self.frame_count / n) * self.current_stack
                        + (1.0 / n) * norm_color
                    )
                    self._buffer.append(norm_color.copy())

                    # hit the bootstrap threshold?
                    if n == self.bootstrap_frames:
                        # init Welford stats
                        buf = np.stack(self._buffer, axis=0)
                        self._mu = np.mean(buf, axis=0)
                        diffs = buf - self._mu[np.newaxis, ...]
                        self._m2 = np.sum(diffs * diffs, axis=0)
                        self._buffer = None

                        # switch to clipping mode
                        self.mode_label.setText("Mode: μ-σ Clipping Average")
                        self.status_label.setText("Switched to μ–σ clipping (color)")
                        QApplication.processEvents()
                    else:
                        # still linear
                        self.mode_label.setText("Mode: Linear Average")
                        self.status_label.setText(f"Processed color frame #{n} (linear)")
                        QApplication.processEvents()
                else:
                    # 12b) μ–σ clipping
                    sigma = np.sqrt(self._m2 / (self.frame_count - 1))
                    mask = np.abs(norm_color - self._mu) <= (self.clip_threshold * sigma)
                    clipped = np.where(mask, norm_color, self._mu)

                    n = self.frame_count + 1
                    self.current_stack = (
                        (self.frame_count / n) * self.current_stack
                        + (1.0 / n) * clipped
                    )

                    # Welford update
                    delta_mu = clipped - self._mu
                    self._mu += delta_mu / n
                    delta2 = clipped - self._mu
                    self._m2 += delta_mu * delta2

                    # stay in clipping mode
                    self.mode_label.setText("Mode: μ-σ Clipping Average")
                    self.status_label.setText(f"Processed color frame #{n} (clipped)")
                    QApplication.processEvents()

                # bump global frame count
                self.frame_count = n
                # ─── accumulate exposure ─────────────────────
                exp_val = header.get("EXPOSURE", header.get("EXPTIME", None))
                if exp_val is not None:
                    try:
                        secs = float(exp_val)
                        self.total_exposure += secs
                        hrs  = int(self.total_exposure // 3600)
                        mins = int((self.total_exposure % 3600) // 60)
                        secs_rem = int(self.total_exposure % 60)
                        self.exposure_label.setText(
                            f"Total Exp: {hrs:02d}:{mins:02d}:{secs_rem:02d}"
                        )
                    except:
                        pass
                QApplication.processEvents()


            else:
                # — Mono→color (per-filter) stacking —
                count = self.filter_counts.get(mono_key, 0)
                buf   = self.filter_buffers.setdefault(mono_key, [])

                if count < self.bootstrap_frames:
                    # 12c) Linear bootstrap per-filter
                    new_count = count + 1
                    if count == 0:
                        self.filter_stacks[mono_key] = norm_plane.copy()
                    else:
                        self.filter_stacks[mono_key] = (
                            (count / new_count) * self.filter_stacks[mono_key]
                            + (1.0 / new_count) * norm_plane
                        )
                    buf.append(norm_plane.copy())
                    self.filter_counts[mono_key] = new_count

                    if new_count == self.bootstrap_frames:
                        # init Welford
                        stacked = np.stack(buf, axis=0)
                        mu    = np.mean(stacked, axis=0)
                        diffs = stacked - mu[np.newaxis, ...]
                        m2    = np.sum(diffs * diffs, axis=0)
                        self.filter_mus[mono_key] = mu
                        self.filter_m2s[mono_key] = m2

                        self.mode_label.setText(f"Mode: μ-σ Clipping Average ({mono_key})")
                        self.status_label.setText(f"Switched to μ–σ clipping ({mono_key})")
                        QApplication.processEvents()
                    else:
                        # still linear
                        self.mode_label.setText(f"Mode: Linear Average ({mono_key})")
                        self.status_label.setText(
                            f"Processed {mono_key}-filter frame #{new_count} (linear)"
                        )
                        QApplication.processEvents()

                else:
                    # 12d) μ–σ clipping per-filter
                    mu = self.filter_mus[mono_key]
                    m2 = self.filter_m2s[mono_key]
                    sigma = np.sqrt(m2 / (count - 1))
                    mask   = np.abs(norm_plane - mu) <= (self.clip_threshold * sigma)
                    clipped = np.where(mask, norm_plane, mu)

                    new_count = count + 1
                    self.filter_stacks[mono_key] = (
                        (count / new_count) * self.filter_stacks[mono_key]
                        + (1.0 / new_count) * clipped
                    )

                    # Welford update on µ and m2
                    delta   = clipped - mu
                    new_mu  = mu + delta / new_count
                    delta2  = clipped - new_mu
                    new_m2  = m2 + delta * delta2
                    self.filter_mus[mono_key] = new_mu
                    self.filter_m2s[mono_key] = new_m2
                    self.filter_counts[mono_key] = new_count

                    self.mode_label.setText(f"Mode: μ-σ Clipping Average ({mono_key})")
                    self.status_label.setText(
                        f"Processed {mono_key}-filter frame #{new_count} (clipped)"
                    )
                    QApplication.processEvents()

                # bump global frame count
                self.frame_count += 1
                self.frame_count_label.setText(f"Frames: {self.frame_count}")
                # ─── accumulate exposure ─────────────────────
                exp_val = header.get("EXPOSURE", header.get("EXPTIME", None))
                if exp_val is not None:
                    try:
                        secs = float(exp_val)
                        self.total_exposure += secs
                        hrs  = int(self.total_exposure // 3600)
                        mins = int((self.total_exposure % 3600) // 60)
                        secs_rem = int(self.total_exposure % 60)
                        self.exposure_label.setText(
                            f"Total Exp: {hrs:02d}:{mins:02d}:{secs_rem:02d}"
                        )
                    except:
                        pass
                QApplication.processEvents()

            # ─── 13) Update UI ─────────────────────────────────────────
            self.frame_count_label.setText(f"Frames: {self.frame_count}")
            QApplication.processEvents()

        # ——— 13) METRICS PANEL for good frame —————————————
        self.metrics_window.metrics_panel.add_point(
            self.frame_count, fwhm, ecc, sc, snr_val, False
        )

        # ——— 14) PREVIEW & STATUS LABEL —————————————————————
        if mono_key:
            preview = self._build_color_composite()
            self.status_label.setText(f"Stacked {mono_key}-filter frame {os.path.basename(path)}")
            QApplication.processEvents()
        else:
            preview = self.current_stack
            self.status_label.setText(f"✔ processed {os.path.basename(path)}")
            QApplication.processEvents()

        self.update_preview(preview)
        QApplication.processEvents()

    def _process_star_trail(self, path: str):
        """
        Load/calibrate a single frame (RAW or FITS/TIFF), debayer if needed,
        normalize, then build a max‐value “star trail” in self.current_stack.
        """
        # ─── 1) Load (RAW vs FITS) ─────────────────────────────
        lower = path.lower()
        raw_exts = ('.cr2', '.cr3', '.nef', '.arw', '.dng',
                    '.orf', '.rw2', '.pef')
        if lower.endswith(raw_exts):
            try:
                with rawpy.imread(path) as raw:
                    img_rgb8 = raw.postprocess(use_camera_wb=True,
                                               no_auto_bright=True,
                                               output_bps=16)
                img = img_rgb8.astype(np.float32) / 65535.0
                header = fits.Header()
                header["SIMPLE"]   = True
                header["BITPIX"]   = 16
                header["CREATOR"]  = "LiveStack(RAW)"
                header["IMAGETYP"] = "RAW"
                header["EXPTIME"]  = "Unknown"
                # attempt EXIF, same as process_frame…
                try:
                    with open(path,'rb') as f:
                        tags = exifread.process_file(f, details=False)
                    exp_tag = tags.get("EXIF ExposureTime") \
                              or tags.get("EXIF ShutterSpeedValue")
                    if exp_tag:
                        ev = str(exp_tag.values)
                        if '/' in ev:
                            n,d = ev.split('/',1)
                            header["EXPTIME"] = (float(n)/float(d),
                                                 "Exposure Time (s)")
                        else:
                            header["EXPTIME"] = (float(ev),
                                                 "Exposure Time (s)")
                except:
                    pass
                bit_depth = 16
                is_mono   = False
            except Exception:
                self.status_label.setText(
                    f"⚠ Failed to decode RAW: {os.path.basename(path)}"
                )
                QApplication.processEvents()
                return
        else:
            # FITS / TIFF / XISF
            img, header, bit_depth, is_mono = load_image(path)
            if img is None:
                self.status_label.setText(
                    f"⚠ Failed to load {os.path.basename(path)}"
                )
                QApplication.processEvents()
                return

        # ─── 2) Calibration ─────────────────────────────────────
        mono_key = None
        if (self.mono_color_mode
            and is_mono
            and header.get('FILTER')
            and 'BAYERPAT' not in header):
            mono_key = self._get_filter_key(header)

        if self.master_dark is not None:
            img = img.astype(np.float32) - self.master_dark

        if mono_key and mono_key in self.master_flats:
            img = apply_flat_division_numba(img,
                                            self.master_flats[mono_key])
        elif self.master_flat is not None:
            img = apply_flat_division_numba(img,
                                            self.master_flat)

        # ─── 3) Debayer ─────────────────────────────────────────
        if is_mono and header.get('BAYERPAT'):
            pat = (header['BAYERPAT'][0]
                   if isinstance(header['BAYERPAT'], tuple)
                   else header['BAYERPAT'])
            img = debayer_fits_fast(img, pat)
            is_mono = False

        # ─── 4) Force 3-channel if still mono ───────────────────
        if not mono_key and img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        # ─── 5) Normalize ───────────────────────────────────────
        # for star-trail we want a visible, stretched version:
        if img.ndim == 2:
            plane = stretch_mono_image(img, target_median=0.3)
            norm_color = np.stack([plane]*3, axis=2)
        else:
            norm_color = stretch_color_image(img,
                                             target_median=0.3,
                                             linked=False)

        # ─── 6) Build max-value stack ───────────────────────────
        if self.frame_count == 0:
            self.current_stack = norm_color.copy()
        else:
            # elementwise max over all frames so far
            self.current_stack = np.maximum(self.current_stack,
                                            norm_color)

        # ─── 7) Update counters and labels ──────────────────────
        self.frame_count += 1
        self.frame_count_label.setText(f"Frames: {self.frame_count}")

        exp_val = header.get("EXPOSURE", header.get("EXPTIME", None))
        if exp_val is not None:
            try:
                secs = float(exp_val)
                self.total_exposure += secs
                h = int(self.total_exposure // 3600)
                m = int((self.total_exposure % 3600)//60)
                s = int(self.total_exposure % 60)
                self.exposure_label.setText(
                    f"Total Exp: {h:02d}:{m:02d}:{s:02d}")
            except:
                pass

        self.status_label.setText(
            f"★ Star-Trail frame {self.frame_count}: "
            f"{os.path.basename(path)}"
        )
        self.update_preview(self.current_stack)
        QApplication.processEvents()



    def update_preview(self, array: np.ndarray):
        # 1) normalize, apply contrast/brightness
        arr = np.clip(array, 0.0, 1.0).astype(np.float32)
        pivot = 0.3
        arr = ((arr - pivot) * self.contrast + pivot) + self.brightness
        arr = np.clip(arr, 0.0, 1.0)

        # 2) convert to uint8 and KEEP a reference on self
        self._last_frame_bytes = (arr * 255).astype(np.uint8)
        h, w = self._last_frame_bytes.shape[:2]

        # 3) build QImage from the kept buffer
        if self._last_frame_bytes.ndim == 2:
            fmt = QImage.Format.Format_Grayscale8
            bytespp = w
        else:
            fmt = QImage.Format.Format_RGB888
            bytespp = 3 * w
        qimg = QImage(self._last_frame_bytes.data, w, h, bytespp, fmt)

        # 4) update scene
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)

        if not self._did_initial_fit:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._did_initial_fit = True
