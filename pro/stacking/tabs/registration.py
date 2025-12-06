# Tab module - imports from parent packages
from __future__ import annotations
import os
import sys
import platform
import math
import time
import numpy as np
import cv2
cv2.setNumThreads(0)

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QFileDialog, QAbstractItemView,
    QProgressDialog, QApplication, QMessageBox, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QToolButton, QLineEdit, QSlider,
    QInputDialog, QMenu
)
from astropy.io import fits
from datetime import datetime

# Import shared utilities from project
from legacy.image_manager import load_image, save_image
from legacy.numba_utils import debayer_raw_fast
from pro.accel_installer import current_backend
from pro.stacking.functions import compute_star_count_fast_preview
from pro.accel_workers import AccelInstallWorker


class RegistrationTab(QObject):
    """Extracted Registration tab functionality."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def create_image_registration_tab(self):
        """
        Image Registration tab with:
        - tree of calibrated lights
        - tolerance row (now includes Auto-crop)
        - add/remove buttons
        - global drizzle controls
        - reference selection + auto-accept toggle
        - MFDeconv (no 'beta')
        - Comet + Star-Trail checkboxes in one row BELOW MFDeconv
        - Backend (acceleration) row moved BELOW the comet+trail row
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ─────────────────────────────────────────
        # 1) QTreeWidget
        # ─────────────────────────────────────────
        self.main.reg_tree = QTreeWidget()
        self.main.reg_tree.setColumnCount(3)
        self.main.reg_tree.setHeaderLabels([
            "Filter - Exposure - Size",
            "Metadata",
            "Drizzle"
        ])
        self.main.reg_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.main.reg_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(QLabel("Calibrated Light Frames"))
        layout.addWidget(self.main.reg_tree)

        model = self.main.reg_tree.model()
        model.rowsInserted.connect(lambda *_: QTimer.singleShot(0, self.main._refresh_reg_tree_summaries))
        model.rowsRemoved.connect(lambda *_: QTimer.singleShot(0, self.main._refresh_reg_tree_summaries))

        # ─────────────────────────────────────────
        # 2) Exposure tolerance + Auto-crop + Split dual-band (same row)
        # ─────────────────────────────────────────
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Exposure Tolerance (sec):"))

        self.main.exposure_tolerance_spin = QSpinBox()
        self.main.exposure_tolerance_spin.setRange(0, 900)
        self.main.exposure_tolerance_spin.setValue(0)
        self.main.exposure_tolerance_spin.setSingleStep(5)
        tol_layout.addWidget(self.main.exposure_tolerance_spin)
        tol_layout.addStretch()

        # Auto-crop moved here
        self.main.autocrop_cb = QCheckBox("Auto-crop output")
        self.main.autocrop_cb.setToolTip("Crop final image to pixels covered by ≥ Coverage % of frames")
        tol_layout.addWidget(self.main.autocrop_cb)

        tol_layout.addWidget(QLabel("Coverage:"))
        self.main.autocrop_pct = QDoubleSpinBox()
        self.main.autocrop_pct.setRange(50.0, 100.0)
        self.main.autocrop_pct.setSingleStep(1.0)
        self.main.autocrop_pct.setSuffix(" %")
        self.main.autocrop_pct.setValue(self.main.settings.value("stacking/autocrop_pct", 95.0, type=float))
        self.main.autocrop_cb.setChecked(self.main.settings.value("stacking/autocrop_enabled", True, type=bool))
        tol_layout.addWidget(self.main.autocrop_pct)

        tol_layout.addStretch()

        self.main.split_dualband_cb = QCheckBox("Split dual-band OSC before integration")
        self.main.split_dualband_cb.setToolTip("For OSC dual-band data: SII/OIII → R=SII, G=OIII; Ha/OIII → R=Ha, G=OIII")
        tol_layout.addWidget(self.main.split_dualband_cb)

        layout.addLayout(tol_layout)
        self.main.exposure_tolerance_spin.valueChanged.connect(
                lambda _ : (self.main.populate_calibrated_lights(), self.main._refresh_reg_tree_summaries())
            )

        # ─────────────────────────────────────────
        # 3) Buttons for Managing Files
        # ─────────────────────────────────────────
        btn_layout = QHBoxLayout()
        self.main.add_reg_files_btn = QPushButton("Add Light Files")
        self.main.add_reg_files_btn.clicked.connect(self.main.add_light_files_to_registration)
        btn_layout.addWidget(self.main.add_reg_files_btn)

        self.main.clear_selection_btn = QPushButton("Remove Selected")
        self.main.clear_selection_btn.clicked.connect(lambda: self.main.clear_tree_selection_registration(self.main.reg_tree))
        btn_layout.addWidget(self.main.clear_selection_btn)

        layout.addLayout(btn_layout)

        # ─────────────────────────────────────────
        # 4) Global Drizzle Controls
        # ─────────────────────────────────────────
        drizzle_layout = QHBoxLayout()

        self.main.drizzle_checkbox = QCheckBox("Enable Drizzle")
        self.main.drizzle_checkbox.toggled.connect(self.main._on_drizzle_checkbox_toggled)
        drizzle_layout.addWidget(self.main.drizzle_checkbox)

        drizzle_layout.addWidget(QLabel("Scale:"))
        self.main.drizzle_scale_combo = QComboBox()
        self.main.drizzle_scale_combo.addItems(["1x", "2x", "3x"])
        self.main.drizzle_scale_combo.currentIndexChanged.connect(self.main._on_drizzle_param_changed)
        drizzle_layout.addWidget(self.main.drizzle_scale_combo)

        drizzle_layout.addWidget(QLabel("Drop Shrink:"))
        self.main.drizzle_drop_shrink_spin = QDoubleSpinBox()
        self.main.drizzle_drop_shrink_spin.setRange(0.0, 1.0)  # pixfrac is [0..1]
        self.main.drizzle_drop_shrink_spin.setDecimals(3)
        self.main.drizzle_drop_shrink_spin.setValue(self.main._get_drizzle_pixfrac())
        self.main.drizzle_drop_shrink_spin.valueChanged.connect(lambda v: self.main._set_drizzle_pixfrac(v))
        drizzle_layout.addWidget(self.main.drizzle_drop_shrink_spin)

        self.main.cfa_drizzle_cb = QCheckBox("CFA Drizzle")
        self.main.cfa_drizzle_cb.setChecked(self.main.settings.value("stacking/cfa_drizzle", False, type=bool))
        self.main.cfa_drizzle_cb.toggled.connect(self.main._on_cfa_drizzle_toggled)
        self.main.cfa_drizzle_cb.setToolTip("Requires 'Enable Drizzle'. Maps R/G/B CFA samples directly into channels (no interpolation).")
        drizzle_layout.addWidget(self.main.cfa_drizzle_cb)

        layout.addLayout(drizzle_layout)

        # ─────────────────────────────────────────
        # 5) Reference Frame Selection
        # ─────────────────────────────────────────
        self.main.ref_frame_label = QLabel("Select Reference Frame:")
        self.main.ref_frame_path = QLabel("No file selected")
        self.main.ref_frame_path.setWordWrap(True)
        self.main.select_ref_frame_btn = QPushButton("Select Reference Frame")
        self.main.select_ref_frame_btn.clicked.connect(self.select_reference_frame)

        self.main.auto_accept_ref_cb = QCheckBox("Auto-accept measured reference")
        self.main.auto_accept_ref_cb.setChecked(self.main.settings.value("stacking/auto_accept_ref", False, type=bool))
        self.main.auto_accept_ref_cb.setToolTip("If checked, the best measured frame is accepted automatically.")
        self.main.auto_accept_ref_cb.toggled.connect(self.main._on_auto_accept_toggled)

        # If the setting is already on at startup but a user ref is locked, unlock it now.
        if self.main.auto_accept_ref_cb.isChecked() and getattr(self, "_user_ref_locked", False):
            self.main.reset_reference_to_auto()

        ref_layout = QHBoxLayout()
        ref_layout.addWidget(self.main.ref_frame_label)
        ref_layout.addWidget(self.main.ref_frame_path, 1)
        ref_layout.addWidget(self.main.select_ref_frame_btn)
        ref_layout.addWidget(self.main.auto_accept_ref_cb)
        ref_layout.addStretch()
        layout.addLayout(ref_layout)

        # Disable Select button when auto-accept is on
        self.main.auto_accept_ref_cb.toggled.connect(self.main.select_ref_frame_btn.setDisabled)
        self.main.select_ref_frame_btn.setDisabled(self.main.auto_accept_ref_cb.isChecked())

        # ─────────────────────────────────────────
        # 6) MFDeconv (title cleaned; no “beta”)
        # ─────────────────────────────────────────
        mf_box = QGroupBox("MFDeconv — Multi-Frame Deconvolution (ImageMM)")
        mf_v = QVBoxLayout(mf_box)

        def _get(key, default, t):
            return self.main.settings.value(key, default, type=t)

        # row 1: enable
        mf_row1 = QHBoxLayout()
        self.main.mf_enabled_cb = QCheckBox("Enable MFDeconv during integration")
        self.main.mf_enabled_cb.setChecked(_get("stacking/mfdeconv/enabled", False, bool))
        self.main.mf_enabled_cb.setToolTip("Runs multi-frame deconvolution during integration. Turn off if testing.")
        mf_row1.addWidget(self.main.mf_enabled_cb)

        # NEW: Super-Resolution checkbox goes here (between Enable and Save-Intermediate)
        self.main.mf_sr_cb = QCheckBox("Super Resolution")
        self.main.mf_sr_cb.setToolTip(
            "Reconstruct on an r× super-res grid using SR PSFs.\n"
            "Compute cost grows roughly ~ r^4. Drizzle usually provides better results."
        )
        self.main.mf_sr_cb.setChecked(self.main.settings.value("stacking/mfdeconv/sr_enabled", False, type=bool))
        mf_row1.addWidget(self.main.mf_sr_cb)

        # Integer box to the RIGHT of the checkbox (2..4× typical; tweak if you want bigger)
        mf_row1.addSpacing(6)
        self.main.mf_sr_factor_spin = QSpinBox()
        self.main.mf_sr_factor_spin.setRange(2, 4)         # set 2..8 if you dare; r^4 cost!
        self.main.mf_sr_factor_spin.setSingleStep(1)
        self.main.mf_sr_factor_spin.setSuffix("×")
        self.main.mf_sr_factor_spin.setToolTip("Super-resolution scale factor r (integer ≥2).")
        self.main.mf_sr_factor_spin.setValue(self.main.settings.value("stacking/mfdeconv/sr_factor", 2, type=int))
        self.main.mf_sr_factor_spin.setEnabled(self.main.mf_sr_cb.isChecked())
        mf_row1.addWidget(self.main.mf_sr_factor_spin)

        mf_row1.addSpacing(16)

        self.main.mf_save_intermediate_cb = QCheckBox("Save intermediate iterative images")
        self.main.mf_save_intermediate_cb.setToolTip("If enabled, saves the seed and every iteration image into a subfolder next to the final output.")
        self.main.mf_save_intermediate_cb.setChecked(self.main.settings.value("stacking/mfdeconv/save_intermediate", False, type=bool))
        mf_row1.addWidget(self.main.mf_save_intermediate_cb)

        mf_row1.addStretch(1)
        mf_v.addLayout(mf_row1)

        # row 2: iterations, min iters, kappa
        mf_row2 = QHBoxLayout()
        mf_row2.addWidget(QLabel("Iterations (max):"))
        self.main.mf_iters_spin = QSpinBox(); self.main.mf_iters_spin.setRange(1, 500)
        self.main.mf_iters_spin.setValue(_get("stacking/mfdeconv/iters", 20, int))
        mf_row2.addWidget(self.main.mf_iters_spin)

        mf_row2.addSpacing(12)
        mf_row2.addWidget(QLabel("Min iters:"))
        self.main.mf_min_iters_spin = QSpinBox(); self.main.mf_min_iters_spin.setRange(1, 500)
        self.main.mf_min_iters_spin.setValue(_get("stacking/mfdeconv/min_iters", 3, int))
        mf_row2.addWidget(self.main.mf_min_iters_spin)

        mf_row2.addSpacing(16)
        mf_row2.addWidget(QLabel("Update clip (κ):"))
        self.main.mf_kappa_spin = QDoubleSpinBox(); self.main.mf_kappa_spin.setRange(0.0, 10.0)
        self.main.mf_kappa_spin.setDecimals(3); self.main.mf_kappa_spin.setSingleStep(0.1)
        self.main.mf_kappa_spin.setValue(_get("stacking/mfdeconv/kappa", 2.0, float))
        # NEW: κ tooltip (layman’s terms)
        self.main.mf_kappa_spin.setToolTip(
            "κ (kappa) limits how big each multiplicative update can be per iteration.\n"
            "• κ = 1.0 → no change (1×); larger κ allows bigger step sizes.\n"
            "• Lower values = gentler, safer updates; higher values = faster but riskier.\n"
            "Typical: 1.05–1.5 for conservative, ~2 for punchier updates."
        )
        mf_row2.addWidget(self.main.mf_kappa_spin)
        mf_row2.addStretch(1)
        mf_v.addLayout(mf_row2)

        # row 3: color / rho / huber / toggles
        mf_row3 = QHBoxLayout()
        mf_row3.addWidget(QLabel("Color mode:"))
        self.main.mf_color_combo = QComboBox(); self.main.mf_color_combo.addItems(["PerChannel", "Luma"])
        _cm = _get("stacking/mfdeconv/color_mode", "PerChannel", str)
        if _cm not in ("PerChannel", "Luma"): _cm = "PerChannel"
        self.main.mf_color_combo.setCurrentText(_cm)
        self.main.mf_color_combo.setToolTip("‘Luma’ deconvolves luminance only; ‘PerChannel’ runs on RGB independently.")
        mf_row3.addWidget(self.main.mf_color_combo)

        mf_row3.addSpacing(16)
        mf_row3.addWidget(QLabel("ρ (loss):"))
        self.main.mf_rho_combo = QComboBox(); self.main.mf_rho_combo.addItems(["Huber", "L2"])
        self.main.mf_rho_combo.setCurrentText(self.main.settings.value("stacking/mfdeconv/rho", "Huber", type=str))
        self.main.mf_rho_combo.currentTextChanged.connect(lambda s: self.main.settings.setValue("stacking/mfdeconv/rho", s))
        mf_row3.addWidget(self.main.mf_rho_combo)

        mf_row3.addSpacing(16)
        mf_row3.addWidget(QLabel("Huber δ:"))
        self.main.mf_Huber_spin = QDoubleSpinBox()
        self.main.mf_Huber_spin.setRange(-1000.0, 1000.0); self.main.mf_Huber_spin.setDecimals(4); self.main.mf_Huber_spin.setSingleStep(0.1)
        self.main.mf_Huber_spin.setValue(_get("stacking/mfdeconv/Huber_delta", -2.0, float))
        # NEW: Huber tooltip (layman’s terms)
        self.main.mf_Huber_spin.setToolTip(
            "Huber δ sets the cutoff between ‘quadratic’ (treat as normal) and ‘linear’ (treat as outlier) behavior.\n"
            "• |residual| ≤ δ → quadratic (more aggressive corrections)\n"
            "• |residual| > δ → linear (gentler, more robust)\n"
            "Negative values mean ‘scale by RMS’: e.g., δ = -2 uses 2×RMS.\n"
            "Smaller |δ| (closer to 0) → more pixels counted as outliers → more conservative.\n"
            "Examples: κ=1.1 & δ=-0.7 = gentle; κ=2 & δ=-2 = more aggressive."
        )
        mf_row3.addWidget(self.main.mf_Huber_spin)

        self.main.mf_Huber_hint = QLabel("(<0 = scale×RMS, >0 = absolute Δ)")
        self.main.mf_Huber_hint.setStyleSheet("color:#888;")
        mf_row3.addWidget(self.main.mf_Huber_hint)

        mf_row3.addSpacing(16)
        self.main.mf_use_star_mask_cb = QCheckBox("Auto Star Mask")
        self.main.mf_use_noise_map_cb = QCheckBox("Auto Noise Map")
        self.main.mf_use_star_mask_cb.setChecked(self.main.settings.value("stacking/mfdeconv/use_star_masks", False, type=bool))
        self.main.mf_use_noise_map_cb.setChecked(self.main.settings.value("stacking/mfdeconv/use_noise_maps", False, type=bool))
        mf_row3.addWidget(self.main.mf_use_star_mask_cb)
        mf_row3.addWidget(self.main.mf_use_noise_map_cb)
        mf_row3.addStretch(1)
        mf_v.addLayout(mf_row3)

        # persist
        self.main.mf_enabled_cb.toggled.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/enabled", bool(v)))
        self.main.mf_iters_spin.valueChanged.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/iters", int(v)))
        self.main.mf_min_iters_spin.valueChanged.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/min_iters", int(v)))
        self.main.mf_kappa_spin.valueChanged.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/kappa", float(v)))
        self.main.mf_color_combo.currentTextChanged.connect(lambda s: self.main.settings.setValue("stacking/mfdeconv/color_mode", s))
        self.main.mf_Huber_spin.valueChanged.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/Huber_delta", float(v)))
        self.main.mf_use_star_mask_cb.toggled.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/use_star_masks", bool(v)))
        self.main.mf_use_noise_map_cb.toggled.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/use_noise_maps", bool(v)))
        self.main.mf_sr_cb.toggled.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/sr_enabled", bool(v)))
        self.main.mf_save_intermediate_cb.toggled.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/save_intermediate", bool(v)))

        layout.addWidget(mf_box)

        # ─────────────────────────────────────────
        # 7) Comet + Star-Trail checkboxes (same row) — now BELOW MFDeconv
        # ─────────────────────────────────────────
        comet_trail_row = QHBoxLayout()
        self.main.comet_cb = QCheckBox("🌠 Create comet stack (comet-aligned)")
        self.main.comet_cb.setChecked(self.main.settings.value("stacking/comet/enabled", False, type=bool))
        comet_trail_row.addWidget(self.main.comet_cb)

        comet_trail_row.addSpacing(12)
        self.main.trail_cb = QCheckBox("★★ Star-Trail Mode ★★ (Max-Value Stack)")
        self.main.trail_cb.setChecked(self.main.star_trail_mode)
        self.main.trail_cb.setToolTip("Skip registration/alignment and use Maximum-Intensity projection for star trails")
        self.main.trail_cb.stateChanged.connect(self.main._on_star_trail_toggled)
        comet_trail_row.addWidget(self.main.trail_cb)
        comet_trail_row.addStretch(1)
        layout.addLayout(comet_trail_row)

        # keep comet options in a compact row just beneath
        comet_opts = QHBoxLayout()
        self.main.comet_pick_btn = QPushButton("Pick comet center…")
        self.main.comet_pick_btn.setEnabled(self.main.comet_cb.isChecked())
        self.main.comet_pick_btn.clicked.connect(self.main._pick_comet_center)
        self.main.comet_cb.toggled.connect(self.main.comet_pick_btn.setEnabled)
        comet_opts.addWidget(self.main.comet_pick_btn)

        self.main.comet_blend_cb = QCheckBox("Also output Stars+Comet blend")
        self.main.comet_blend_cb.setChecked(self.main.settings.value("stacking/comet/blend", True, type=bool))
        comet_opts.addWidget(self.main.comet_blend_cb)

        comet_opts.addWidget(QLabel("Mix:"))
        self.main.comet_mix = QDoubleSpinBox(); self.main.comet_mix.setRange(0.0, 1.0); self.main.comet_mix.setSingleStep(0.05)
        self.main.comet_mix.setValue(self.main.settings.value("stacking/comet/mix", 1.0, type=float))
        comet_opts.addWidget(self.main.comet_mix)
        self.main.comet_save_starless_cb = QCheckBox("Save all starless comet-aligned frames")
        self.main.comet_save_starless_cb.setChecked(self.main.settings.value("stacking/comet/save_starless", False, type=bool))
        comet_opts.addWidget(self.main.comet_save_starless_cb)        
        comet_opts.addStretch(1)

        layout.addLayout(comet_opts)

        # persist comet settings
        self.main.settings.setValue("stacking/comet/enabled", self.main.comet_cb.isChecked())
        self.main.settings.setValue("stacking/comet/blend", self.main.comet_blend_cb.isChecked())
        self.main.settings.setValue("stacking/comet/mix", self.main.comet_mix.value())
        self.main.comet_save_starless_cb.toggled.connect(
            lambda v: self.main.settings.setValue("stacking/comet/save_starless", bool(v))
        )
        # ─────────────────────────────────────────
        # 8) Backend / Install GPU Acceleration — MOVED BELOW comet+trail row
        # ─────────────────────────────────────────
        accel_row = QHBoxLayout()
        self.main.backend_label = QLabel(f"Backend: {current_backend()}")
        accel_row.addWidget(self.main.backend_label)

        self.main.install_accel_btn = QPushButton("Install/Update GPU Acceleration…")
        self.main.install_accel_btn.setToolTip("Downloads PyTorch with the right backend (CUDA/MPS/CPU). One-time per machine.")
        accel_row.addWidget(self.main.install_accel_btn)

        gpu_help_btn = QToolButton()
        gpu_help_btn.setText("?")
        gpu_help_btn.setToolTip("If GPU still not being used — click for fix steps")
        gpu_help_btn.clicked.connect(self.main._show_gpu_accel_fix_help)
        accel_row.addWidget(gpu_help_btn)

        accel_row.addStretch(1)
        layout.addLayout(accel_row)

        # same installer wiring as before
        def _install_accel():
            v = sys.version_info
            if not (v.major == 3 and v.minor in (10, 11, 12)):
                why = (f"This app is running on Python {v.major}.{v.minor}. "
                    "GPU acceleration requires Python 3.10, 3.11, or 3.12.")
                tip = ""
                sysname = platform.system()
                if sysname == "Darwin":
                    tip = ("\n\nmacOS tip (Apple Silicon):\n"
                        " • Install Python 3.12:  brew install python@3.12\n"
                        " • Then relaunch the app so it can create its runtime with 3.12.")
                elif sysname == "Windows":
                    tip = ("\n\nWindows tip:\n"
                        " • Install Python 3.12/3.11/3.10 (x64) from python.org\n"
                        " • Then relaunch the app.")
                else:
                    tip = ("\n\nLinux tip:\n"
                        " • Install python3.12 or 3.11 via your package manager\n"
                        " • Then relaunch the app.")

                QMessageBox.warning(self.main, "Unsupported Python Version", why + tip)
                # reflect the abort in UI/status and leave button enabled
                try:
                    self.main.backend_label.setText("Backend: CPU (Python version not supported for GPU install)")
                    self.main.status_signal.emit("❌ GPU Acceleration install aborted: unsupported Python version.")
                except Exception:
                    pass
                return
            self.main.install_accel_btn.setEnabled(False)
            self.main.backend_label.setText("Backend: installing…")
            self.main._accel_pd = QProgressDialog("Preparing runtime…", "Cancel", 0, 0, self.main)
            self.main._accel_pd.setWindowTitle("Installing GPU Acceleration")
            self.main._accel_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
            self.main._accel_pd.setAutoClose(True)
            self.main._accel_pd.setMinimumDuration(0)
            self.main._accel_pd.show()

            self.main._accel_thread = QThread(self)
            self.main._accel_worker = AccelInstallWorker(prefer_gpu=True)
            self.main._accel_worker.moveToThread(self.main._accel_thread)
            self.main._accel_thread.started.connect(self.main._accel_worker.run, Qt.ConnectionType.QueuedConnection)
            self.main._accel_worker.progress.connect(self.main._accel_pd.setLabelText, Qt.ConnectionType.QueuedConnection)
            self.main._accel_worker.progress.connect(lambda s: self.main.status_signal.emit(s), Qt.ConnectionType.QueuedConnection)

            def _cancel():
                if self.main._accel_thread.isRunning():
                    self.main._accel_thread.requestInterruption()
            self.main._accel_pd.canceled.connect(_cancel, Qt.ConnectionType.QueuedConnection)

            def _done(ok: bool, msg: str):
                if getattr(self, "_accel_pd", None):
                    self.main._accel_pd.reset(); self.main._accel_pd.deleteLater(); self.main._accel_pd = None
                self.main._accel_thread.quit(); self.main._accel_thread.wait()
                self.main.install_accel_btn.setEnabled(True)
                from pro.accel_installer import current_backend
                self.main.backend_label.setText(f"Backend: {current_backend()}")
                self.main.status_signal.emit(("✅ " if ok else "❌ ") + msg)
                if ok: QMessageBox.information(self.main, "Acceleration", f"✅ {msg}")
                else:  QMessageBox.warning(self.main, "Acceleration", f"❌ {msg}")

            self.main._accel_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)
            self.main._accel_thread.finished.connect(self.main._accel_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
            self.main._accel_thread.finished.connect(self.main._accel_thread.deleteLater, Qt.ConnectionType.QueuedConnection)
            self.main._accel_thread.start()

        self.main.install_accel_btn.clicked.connect(_install_accel)

        # ─────────────────────────────────────────
        # 9) Action Buttons
        # ─────────────────────────────────────────
        self.main.register_images_btn = QPushButton("🔥🚀Register and Integrate Images🔥🚀")
        self.main.register_images_btn.clicked.connect(self.register_images)
        self.main.register_images_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #FF6347; }
        """)
        layout.addWidget(self.main.register_images_btn)

        self.main._registration_busy = False

        self.main.integrate_registered_btn = QPushButton("Integrate Previously Registered Images")
        self.main.integrate_registered_btn.clicked.connect(self.main.integration_ctrl.integrate_registered_images)
        self.main.integrate_registered_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;
            }
            QPushButton:hover  { border: 2px solid #FFD700; }
            QPushButton:pressed{ background-color: #222; border: 2px solid #FFA500; }
        """)
        layout.addWidget(self.main.integrate_registered_btn)

        # ─────────────────────────────────────────
        # 10) Init + persist bits
        # ─────────────────────────────────────────
        self.main.populate_calibrated_lights()
        self.main._refresh_reg_tree_summaries()
        tab.setLayout(layout)

        self.main.drizzle_checkbox.setChecked(self.main.settings.value("stacking/drizzle_enabled", False, type=bool))
        self.main.drizzle_scale_combo.setCurrentText(self.main.settings.value("stacking/drizzle_scale", "2x", type=str))
        self.main.drizzle_drop_shrink_spin.setValue(self.main.settings.value("stacking/drizzle_drop", 0.65, type=float))

        drizzle_on = self.main.settings.value("stacking/drizzle_enabled", False, type=bool)
        self.main.cfa_drizzle_cb.setEnabled(drizzle_on)
        if not drizzle_on and self.main.cfa_drizzle_cb.isChecked():
            self.main.cfa_drizzle_cb.blockSignals(True)
            self.main.cfa_drizzle_cb.setChecked(False)
            self.main.cfa_drizzle_cb.blockSignals(False)
            self.main.settings.setValue("stacking/cfa_drizzle", False)
        self.main._update_drizzle_summary_columns()

        # persist SR on/off and factor
        self.main.mf_sr_cb.toggled.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/sr_enabled", bool(v)))
        self.main.mf_sr_cb.toggled.connect(self.main.mf_sr_factor_spin.setEnabled)
        self.main.mf_sr_factor_spin.valueChanged.connect(lambda v: self.main.settings.setValue("stacking/mfdeconv/sr_factor", int(v)))

        # persist auto-crop settings
        self.main.autocrop_cb.toggled.connect(
            lambda v: (self.main.settings.setValue("stacking/autocrop_enabled", bool(v)),
                    self.main.settings.sync())
        )
        self.main.autocrop_pct.valueChanged.connect(
            lambda v: (self.main.settings.setValue("stacking/autocrop_pct", float(v)),
                    self.main.settings.sync())
        )


        # If comet star-removal is globally disabled, gray this out with a helpful tooltip
        csr_enabled_globally = self.main.settings.value("stacking/comet_starrem/enabled", False, type=bool)

        def _refresh_comet_starless_enable():
            csr_on   = bool(self.main.settings.value("stacking/comet_starrem/enabled", False, type=bool))
            comet_on = bool(self.main.comet_cb.isChecked())
            ok = comet_on and csr_on

            self.main.comet_save_starless_cb.setEnabled(ok)

            tip = ("Save all comet-aligned starless frames.\n"
                "Requires: ‘Create comet stack’ AND ‘Remove stars on comet-aligned frames’.")
            if not csr_on:
                tip += "\n\n(Comet Star Removal is currently OFF in Settings.)"
            self.main.comet_save_starless_cb.setToolTip(tip)

            if not ok and self.main.comet_save_starless_cb.isChecked():
                self.main.comet_save_starless_cb.blockSignals(True)
                self.main.comet_save_starless_cb.setChecked(False)
                self.main.comet_save_starless_cb.blockSignals(False)
                self.main.settings.setValue("stacking/comet/save_starless", False)

        self.main._refresh_comet_starless_enable = _refresh_comet_starless_enable  # expose to self

        # wire it
        self.main.comet_cb.toggled.connect(lambda _v: self.main._refresh_comet_starless_enable())
        self.main.comet_save_starless_cb.toggled.connect(lambda _v: self.main._refresh_comet_starless_enable())  # optional belt/suspenders

        # run once AFTER you restore all initial states from settings
        self.main._refresh_comet_starless_enable()


        return tab



    def select_reference_frame(self):
        """ Opens a file dialog to select the reference frame. """
        file_path, _ = QFileDialog.getOpenFileName(self.main, "Select Reference Frame", "", 
                                                "FITS Images (*.fits *.fit);;All Files (*)")
        if file_path:
            self.main._set_user_reference(file_path)


    def prompt_for_reference_frame(self):
        new_ref, _ = QFileDialog.getOpenFileName(
            self.main,
            "Select New Reference Frame",
            "",  # default directory
            "FITS Files (*.fit *.fits);;All Files (*)"
        )
        return new_ref if new_ref else None


    def register_images(self):

        # ---- local helper: force exact (H,W) via center-crop or reflect-pad ----
        def _force_shape_hw(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
            import numpy as np
            import cv2
            a = np.asarray(img)
            if a.ndim < 2:
                return a

            # detect layout
            if a.ndim == 2:
                H, W = a.shape; Caxis = None
            elif a.ndim == 3:
                if a.shape[-1] in (1, 3):      # HWC
                    H, W = a.shape[:2]; Caxis = -1
                elif a.shape[0] in (1, 3):     # CHW
                    H, W = a.shape[1:]; Caxis = 0
                else:
                    H, W = a.shape[:2]; Caxis = -1
            else:
                return a

            th, tw = int(target_h), int(target_w)
            if (H, W) == (th, tw):
                return a

            # center-crop if larger
            y0 = max(0, (H - th) // 2); x0 = max(0, (W - tw) // 2)
            y1 = min(H, y0 + th);       x1 = min(W, x0 + tw)

            if a.ndim == 2:
                cropped = a[y0:y1, x0:x1]
            elif Caxis == -1:
                cropped = a[y0:y1, x0:x1, ...]
            else:
                cropped = a[:, y0:y1, x0:x1]

            ch, cw = (cropped.shape[:2] if (cropped.ndim == 2 or Caxis == -1) else cropped.shape[1:3])

            # reflect-pad if smaller
            pad_t = max(0, (th - ch) // 2)
            pad_b = max(0, th - ch - pad_t)
            pad_l = max(0, (tw - cw) // 2)
            pad_r = max(0, tw - cw - pad_l)

            if pad_t or pad_b or pad_l or pad_r:
                if cropped.ndim == 2:
                    cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                                borderType=cv2.BORDER_REFLECT_101)
                elif Caxis == -1:
                    cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r,
                                                borderType=cv2.BORDER_REFLECT_101)
                else:
                    chans = []
                    for c in range(cropped.shape[0]):
                        chans.append(cv2.copyMakeBorder(cropped[c], pad_t, pad_b, pad_l, pad_r,
                                                        borderType=cv2.BORDER_REFLECT_101))
                    cropped = np.stack(chans, axis=0)

            return cropped

        if getattr(self, "_registration_busy", False):
            self.main.update_status("⏸ Registration already running; ignoring extra click.")
            return
        self.main.update_status("🧹 Doing a little tidying up...")
        user_ref_locked = bool(getattr(self, "_user_ref_locked", False))

        # Only clear derived geometry/maps when NOT locked
        if not user_ref_locked:
            self.main._norm_target_hw = None
            self.main._orig2norm = {}
            try:
                if hasattr(self, "ref_frame_path") and self.main.ref_frame_path:
                    self.main.ref_frame_path.setText("Auto (not set)")
            except Exception:
                pass
        else:
            # Keep the UI showing the user’s chosen ref (basename for display)
            try:
                if hasattr(self, "ref_frame_path") and self.main.ref_frame_path and self.main.reference_frame:
                    self.main.ref_frame_path.setText(os.path.basename(self.main.reference_frame))
            except Exception:
                pass

        # 🚫 Do NOT remove persisted user ref here; that defeats locking.
        # (No settings.remove() and no reference_frame = None if locked)

        self.main._set_registration_busy(True)

        try:
            if self.main.star_trail_mode:
                self.main.update_status("🌠 Star-Trail Mode enabled: skipping registration & using max-value stack")
                QApplication.processEvents()
                return self.main._make_star_trail()

            self.main.update_status("🔄 Image Registration Started...")
            self.main.extract_light_files_from_tree(debug=True)

            comet_mode = bool(getattr(self, "comet_cb", None) and self.main.comet_cb.isChecked())
            if comet_mode:
                self.main.update_status("🌠 Comet mode: please click the comet center to continue…")
                QApplication.processEvents()
                ok = self.main._ensure_comet_seed_now()
                if not ok:
                    QMessageBox.information(self.main, "Comet Mode",
                        "No comet center was selected. Registration has been cancelled so you can try again.")
                    self.main.update_status("❌ Registration cancelled (no comet seed).")
                    return
                else:
                    self.main._comet_ref_xy = None
                    self.main.update_status("✅ Comet seed set. Proceeding with registration…")
                    QApplication.processEvents()

            if not self.main.light_files:
                self.main.update_status("⚠️ No light files to register!")
                return

            # dual-band split unchanged...
            selected_groups = set()
            for it in self.main.reg_tree.selectedItems():
                top = it if it.parent() is None else it.parent()
                selected_groups.add(top.text(0))
            if self.main.split_dualband_cb.isChecked():
                self.main.update_status("🌈 Splitting dual-band OSC frames into Ha / SII / OIII...")
                self.main._split_dual_band_osc(selected_groups=selected_groups)
                self.main._refresh_reg_tree_from_light_files()

            self.main._maybe_warn_cfa_low_frames()

            all_files = [f for lst in self.main.light_files.values() for f in lst]
            self.main.update_status(f"📊 Found {len(all_files)} total frames. Now measuring in parallel batches...")

            # ── binning (FITS/XISF aware) ─────────────────────────────
            bin_map = {}
            min_xbin, min_ybin = None, None
            for fp in all_files:
                xb, yb = self.main._bin_from_header_fast_any(fp)
                bin_map[fp] = (xb, yb)
                if min_xbin is None or xb < min_xbin:
                    min_xbin = xb
                if min_ybin is None or yb < min_ybin:
                    min_ybin = yb

            target_xbin, target_ybin = (min_xbin or 1), (min_ybin or 1)
            self.main.update_status(
                f"🧮 Binning summary → target={target_xbin}×{target_ybin} "
                f"(range observed: x=[{min(b[0] for b in bin_map.values())}..{max(b[0] for b in bin_map.values())}], "
                f"y=[{min(b[1] for b in bin_map.values())}..{max(b[1] for b in bin_map.values())}])"
            )

            # ─────────────────────────────────────────────────────────────────────
            # Helpers
            # ─────────────────────────────────────────────────────────────────────
            def mono_preview_for_stats(img: np.ndarray, hdr, fp: str) -> np.ndarray:
                """
                Returns a 2D float32 preview that is:
                • debayer-aware (uses luma if RGB; superpixel if mono/CFA)
                • resampled to the target bin (so previews from mixed binning match scale)
                • superpixel-averaged 2×2 for speed
                """
                if img is None:
                    return None

                def _superpixel2x2(x: np.ndarray) -> np.ndarray:
                    h, w = x.shape[:2]
                    h2, w2 = h - (h % 2), w - (w % 2)
                    if h2 <= 0 or w2 <= 0:
                        return x.astype(np.float32, copy=False)
                    x = x[:h2, :w2].astype(np.float32, copy=False)
                    if x.ndim == 2:
                        return (x[0:h2:2, 0:w2:2] + x[0:h2:2, 1:w2:2] +
                                x[1:h2:2, 0:w2:2] + x[1:h2:2, 1:w2:2]) * 0.25
                    else:
                        r = x[..., 0]; g = x[..., 1]; b = x[..., 2]
                        Luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                        return (Luma[0:h2:2, 0:w2:2] + Luma[0:h2:2, 1:w2:2] +
                                Luma[1:h2:2, 0:w2:2] + Luma[1:h2:2, 1:w2:2]) * 0.25

                # 1) quick 2D preview
                if img.ndim == 3 and img.shape[-1] == 3:
                    prev2d = _superpixel2x2(img)
                elif hdr and hdr.get('BAYERPAT') and img.ndim == 2:
                    prev2d = _superpixel2x2(img)          # CFA → superpixel Luma-ish
                elif img.ndim == 2:
                    prev2d = _superpixel2x2(img)          # mono
                else:
                    prev2d = img.astype(np.float32, copy=False)
                    if prev2d.ndim == 3:
                        prev2d = _superpixel2x2(prev2d)

                # 2) resample preview to the target bin
                xb, yb = bin_map.get(fp, (1, 1))
                sx = float(xb) / float(target_xbin)
                sy = float(yb) / float(target_ybin)
                if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
                    prev2d = _resize_to_scale(prev2d, sx, sy)

                return np.ascontiguousarray(prev2d.astype(np.float32, copy=False))

            def chunk_list(lst, size):
                for i in range(0, len(lst), size):
                    yield lst[i:i+size]

            def _robust_scale_from_header(hdr):
                """
                Return (sx, sy) arcsec/px or (None, None).
                Priority:
                1) WCS CD / (CDELT + PC)
                2) XPIXSZ/YPIXSZ treated as *effective* (already includes binning!) if present
                3) PIXSIZE1/PIXSIZE2 treated as *base* pixel size → multiply by XBINNING/YBINNING
                """
                try:
                    # 1) WCS (preferred)
                    if all(k in hdr for k in ("CD1_1","CD1_2","CD2_1","CD2_2")):
                        cd11 = float(hdr["CD1_1"]); cd12 = float(hdr["CD1_2"])
                        cd21 = float(hdr["CD2_1"]); cd22 = float(hdr["CD2_2"])
                        sx_deg = (cd11**2 + cd12**2) ** 0.5
                        sy_deg = (cd21**2 + cd22**2) ** 0.5
                        return abs(sx_deg) * 3600.0, abs(sy_deg) * 3600.0

                    if ("CDELT1" in hdr) and ("CDELT2" in hdr):
                        cdelt1 = float(hdr["CDELT1"]); cdelt2 = float(hdr["CDELT2"])
                        pc11 = float(hdr.get("PC1_1", 1.0)); pc12 = float(hdr.get("PC1_2", 0.0))
                        pc21 = float(hdr.get("PC2_1", 0.0)); pc22 = float(hdr.get("PC2_2", 1.0))
                        m11 = cdelt1 * pc11; m12 = cdelt1 * pc12
                        m21 = cdelt2 * pc21; m22 = cdelt2 * pc22
                        sx_deg = (m11**2 + m12**2) ** 0.5
                        sy_deg = (m21**2 + m22**2) ** 0.5
                        return abs(sx_deg) * 3600.0, abs(sy_deg) * 3600.0

                    # 2) Instrumental: prefer XPIXSZ/YPIXSZ as *effective* pixel size (already includes binning)
                    fl_mm = float(hdr.get("FOCALLEN", 0.0) or 0.0)
                    if fl_mm > 0:
                        xpixsz  = hdr.get("XPIXSZ")
                        ypixsz  = hdr.get("YPIXSZ")
                        if (xpixsz is not None) or (ypixsz is not None):
                            px_um = float(xpixsz or ypixsz or 0.0)
                            py_um = float(ypixsz or xpixsz or px_um)
                            # 🚫 DO NOT multiply by XBINNING/YBINNING here (effective size already includes it)
                            sx = 206.265 * (px_um) / fl_mm
                            sy = 206.265 * (py_um) / fl_mm
                            return sx, sy

                        # 3) Otherwise, PIXSIZE1/PIXSIZE2 are *base* pixel sizes → multiply by bin
                        pxsize1 = hdr.get("PIXSIZE1"); pxsize2 = hdr.get("PIXSIZE2")
                        if (pxsize1 is not None) or (pxsize2 is not None):
                            px_um = float(pxsize1 or pxsize2 or 0.0)
                            py_um = float(pxsize2 or pxsize1 or px_um)
                            xb = int(hdr.get("XBINNING", hdr.get("BINX", 1)) or 1)
                            yb = int(hdr.get("YBINNING", hdr.get("BINY", 1)) or 1)
                            sx = 206.265 * (px_um * xb) / fl_mm
                            sy = 206.265 * (py_um * yb) / fl_mm
                            return sx, sy
                except Exception:
                    pass
                return (None, None)

            def _eff_scale_for_target_bin(raw_sx, raw_sy, xb, yb, target_xbin, target_ybin):
                if (raw_sx is None) or (raw_sy is None):
                    return (None, None)
                kx = float(xb) / float(target_xbin)
                ky = float(yb) / float(target_ybin)
                return float(raw_sx) / max(1e-12, kx), float(raw_sy) / max(1e-12, ky)

            def _rel_delta(a, b):
                a = float(a); b = float(b)
                if a == 0 or b == 0:
                    return abs(a - b)
                return abs(a - b) / max(abs(a), abs(b))


            # ─────────────────────────────────────────────────────────────────────
            # PHASE 1: measure (NO demosaic here)
            # ─────────────────────────────────────────────────────────────────────
            self.main.frame_weights = {}
            mean_values = {}
            star_counts = {}
            measured_frames = []
            preview_medians = {}

            max_workers = os.cpu_count() or 4
            chunk_size = max_workers
            chunks = list(chunk_list(all_files, chunk_size))
            total_chunks = len(chunks)

            from concurrent.futures import ThreadPoolExecutor, as_completed

            for idx, chunk in enumerate(chunks, 1):
                self.main.update_status(f"📦 Measuring chunk {idx}/{total_chunks} ({len(chunk)} frames)")
                chunk_images = []
                chunk_valid_files = []

                self.main.update_status(f"🌍 Loading {len(chunk)} previews in parallel (up to {max_workers} threads)...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futs = {executor.submit(self.main._quick_preview_any, fp, target_xbin, target_ybin): fp
                            for fp in chunk}
                    for fut in as_completed(futs):
                        fp = futs[fut]
                        try:
                            preview = fut.result()
                            if preview is None:
                                continue
                            chunk_images.append(preview)
                            chunk_valid_files.append(fp)
                        except Exception as e:
                            self.main.update_status(f"⚠️ Error previewing {fp}: {e}")
                        QApplication.processEvents()

                if not chunk_images:
                    self.main.update_status("⚠️ No valid previews in this chunk (couldn’t find image data in any HDU).")
                    continue

                # size align (crop) before stats
                min_h = min(im.shape[0] for im in chunk_images)
                min_w = min(im.shape[1] for im in chunk_images)
                if any((im.shape[0] != min_h or im.shape[1] != min_w) for im in chunk_images):
                    chunk_images = [_center_crop_2d(im, min_h, min_w) for im in chunk_images]

                self.main.update_status("🌍 Measuring global means in parallel...")
                QApplication.processEvents()
                means = np.array([float(np.mean(ci)) for ci in chunk_images], dtype=np.float32)
                mean_values.update({fp: float(means[i]) for i, fp in enumerate(chunk_valid_files)})

                def _star_job(i_fp):
                    i, fp = i_fp
                    p = chunk_images[i]
                    pmin = float(np.nanmin(p))
                    med = float(np.median(p - pmin))
                    c, ecc = compute_star_count_fast_preview(p)
                    return fp, med, c, ecc

                star_workers = min(max_workers, 8)
                with ThreadPoolExecutor(max_workers=star_workers) as ex:
                    for fp, med, c, ecc in ex.map(_star_job, enumerate(chunk_valid_files)):
                        preview_medians[fp] = med
                        star_counts[fp] = {"count": c, "eccentricity": ecc}
                        measured_frames.append(fp)

                del chunk_images
                gc.collect()  # Free memory after processing each chunk

            if not measured_frames:
                self.main.update_status("⚠️ No frames could be measured!")
                return

            self.main.update_status(f"✅ All chunks complete! Measured {len(measured_frames)} frames total.")
            # ─────────────────────────────────────────────────────────────────────
            # FAST reference selection: score = starcount / (median * ecc)
            # uses stats we ALREADY measured → good for 100s of frames
            # ─────────────────────────────────────────────────────────────────────
            self.main.update_status("🧠 Selecting reference optimized for AstroAlign (starcount/(median*ecc))…")
            QApplication.processEvents()

            def _dominant_pa_cluster_simple(fps, get_pa, tol=12.0):
                vals = []
                for fp in fps:
                    pa = get_pa(fp)
                    if pa is None:
                        continue
                    v = ((pa % 180.0) + 180.0) % 180.0  # normalize [0,180)
                    vals.append((fp, v))
                if not vals:
                    return fps
                vals.sort(key=lambda t: t[1])
                best_group = []; best_size = -1
                for i in range(len(vals)):
                    center = vals[i][1]
                    inliers = []
                    for fp, v in vals:
                        d = abs(v - center); d = min(d, 180.0 - d)
                        if d <= tol:
                            inliers.append(fp)
                    if len(inliers) > best_size:
                        best_group, best_size = inliers, len(inliers)
                return best_group if best_group else fps

            def _fast_ref_score(fp: str) -> float:
                info = star_counts.get(fp, {"count": 0, "eccentricity": 1.0})
                star_count = float(info.get("count", 0.0))
                ecc = float(info.get("eccentricity", 1.0))
                med = float(preview_medians.get(fp, 0.0))
                med = max(med, 1e-3)
                ecc = max(0.25, min(ecc, 3.0))
                return star_count / (med * ecc)

            user_ref_locked = bool(getattr(self, "_user_ref_locked", False))
            user_ref = getattr(self, "reference_frame", None)

            if user_ref_locked and user_ref:
                self.main.update_status(f"📌 Using user-specified reference (locked): {user_ref}")
                self.main.reference_frame = user_ref
                self.main.reference_frame = os.path.normpath(self.main.reference_frame)
            else:
                def _pa_of(fp):
                    try:
                        hdr0 = fits.getheader(fp, ext=0)
                        return self.main._extract_pa_deg(hdr0)
                    except Exception:
                        return None

                if len(measured_frames) > 60:
                    candidates = _dominant_pa_cluster_simple(measured_frames, _pa_of, tol=12.0) or measured_frames[:]
                else:
                    candidates = measured_frames[:]

                best_fp = None; best_score = -1.0
                for fp in candidates:
                    s = _fast_ref_score(fp)
                    if s > best_score:
                        best_fp, best_score = fp, s

                if best_fp is None and measured_frames:
                    best_fp = max(measured_frames, key=lambda f: self.main.frame_weights.get(f, 0.0))
                    best_score = float(self.main.frame_weights.get(best_fp, 0.0))

                self.main.reference_frame = best_fp or measured_frames[0]
                self.main.update_status(
                    f"📌 Auto-selected reference: {os.path.basename(self.main.reference_frame)} "
                    + (f"(score={best_score:.4f})" if best_fp else "(fallback)")
                )
                QApplication.processEvents()


            ref_stats_meas = star_counts.get(self.main.reference_frame, {"count": 0, "eccentricity": 0.0})
            ref_count = ref_stats_meas["count"]
            ref_ecc   = ref_stats_meas["eccentricity"]

            # ─────────────────────────────────────────────────────────────────────
            # Debayer the reference ONCE and compute ref_median from debayered ref
            # ─────────────────────────────────────────────────────────────────────
            ref_img_raw, ref_hdr = self.main._load_image_any(self.main.reference_frame)
            if ref_img_raw is None:
                self.main.update_status(f"🚨 Could not load reference {self.main.reference_frame}. Aborting.")
                return

            bayer = self.main._hdr_get(ref_hdr, 'BAYERPAT')
            splitdb = bool(self.main._hdr_get(ref_hdr, 'SPLITDB', False))
            if bayer and not splitdb and (ref_img_raw.ndim == 2 or (ref_img_raw.ndim == 3 and ref_img_raw.shape[-1] == 1)):
                self.main.update_status("📦 Debayering reference frame…")
                ref_img = self.main.debayer_image(ref_img_raw, self.main.reference_frame, ref_hdr)
            else:
                ref_img = ref_img_raw
                if ref_img.ndim == 3 and ref_img.shape[-1] == 1:
                    ref_img = np.squeeze(ref_img, axis=-1)

            # Use Luma median if color, else direct median
            if ref_img.ndim == 3 and ref_img.shape[-1] == 3:
                r, g, b = ref_img[..., 0], ref_img[..., 1], ref_img[..., 2]
                ref_Luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ref_median = float(np.median(ref_Luma))
            else:
                ref_median = float(np.median(ref_img))

            self.main.update_status(f"📊 Reference (debayered) median: {ref_median:.4f}")
            QApplication.processEvents()

            # Modeless ref review (unchanged)
            stats_payload = {"star_count": ref_count, "eccentricity": ref_ecc, "mean": ref_median}

            if user_ref_locked:
                self.main.update_status("✅ User reference is locked; skipping reference review dialog.")
                try:
                    self.main.ref_frame_path.setText(os.path.basename(self.main.reference_frame or "") or "No file selected")
                except Exception:
                    pass
            elif self.main.auto_accept_ref_cb.isChecked():
                self.main.update_status("✅ Auto-accept measured reference is enabled; using the measured best frame.")
                try:
                    self.main.ref_frame_path.setText(os.path.basename(self.main.reference_frame or "") or "No file selected")
                except Exception:
                    pass
            else:
                dialog = ReferenceFrameReviewDialog(self.main.reference_frame, stats_payload, parent=self)
                dialog.setModal(False)
                dialog.setWindowModality(Qt.WindowModality.NonModal)
                dialog.show(); dialog.raise_(); dialog.activateWindow()
                _loop = QEventLoop(self); dialog.finished.connect(_loop.quit); _loop.exec()
                result = dialog.result(); user_choice = dialog.getUserChoice()
                if result != QDialog.DialogCode.Accepted and user_choice == "select_other":
                    new_ref = self.prompt_for_reference_frame()
                    if new_ref:
                        self.main._set_user_reference(new_ref)  # sets lock + updates UI/settings
                        self.main.update_status(f"User selected a new reference frame: {new_ref}")
                        ref_img_raw, ref_hdr = self.main._load_image_any(self.main.reference_frame)
                        if ref_img_raw is None:
                            self.main.update_status(f"🚨 Could not load reference {self.main.reference_frame}. Aborting.")
                            return
                        if ref_hdr and ref_hdr.get('BAYERPAT') and not ref_hdr.get('SPLITDB', False) and (ref_img_raw.ndim == 2 or (ref_img_raw.ndim == 3 and ref_img_raw.shape[-1] == 1)):
                            self.main.update_status("📦 Debayering reference frame…")
                            ref_img = self.main.debayer_image(ref_img_raw, self.main.reference_frame, ref_hdr)
                        else:
                            ref_img = ref_img_raw
                            if ref_img.ndim == 3 and ref_img.shape[-1] == 1:
                                ref_img = np.squeeze(ref_img, axis=-1)
                        if ref_img.ndim == 3 and ref_img.shape[-1] == 3:
                            r, g, b = ref_img[..., 0], ref_img[..., 1], ref_img[..., 2]
                            ref_Luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            ref_median = float(np.median(ref_Luma))
                        else:
                            ref_median = float(np.median(ref_img))
                        self.main.update_status(f"📊 (New) reference median: {ref_median:.4f}")
                    else:
                        self.main.update_status("No new reference selected; using previous reference.")

            ref_L = _Luma(ref_img)
            ref_min = float(np.nanmin(ref_L))
            ref_target_median = float(np.nanmedian(ref_L - ref_min))
            self.main.update_status(f"📊 Reference min={ref_min:.6f}, normalized-median={ref_target_median:.6f}")
            QApplication.processEvents()

            # Initial per-file scale factors from preview medians
            eps = 1e-6
            scale_guess = {}
            missing = []
            for fp in measured_frames:
                m = preview_medians.get(fp, 0.0)
                if m <= eps:
                    missing.append(os.path.basename(fp))
                    m = 1.0
                scale_guess[fp] = ref_target_median / max(m, eps)
            if missing:
                self.main.update_status(f"ℹ️ {len(missing)} frame(s) had zero/NaN preview medians; using neutral scale for those.")

            # ─────────────────────────────────────────────────────────────────────
            # PHASE 1b: Meridian flips
            # ─────────────────────────────────────────────────────────────────────
            ref_pa = self.main._extract_pa_deg(ref_hdr)
            self.main.update_status(f"🧭 Reference PA: {ref_pa:.2f}°" if ref_pa is not None else "🧭 Reference PA: (unknown)")
            QApplication.processEvents()

            # ─────────────────────────────────────────────────────────────────────
            # NEW: build arcsec/px map & choose target = reference scale
            # ─────────────────────────────────────────────────────────────────────
            self.main.arcsec_per_px = {}  # fp -> (sx, sy)

            def _scale_from_header_raw(hdr):
                try:
                    if all(k in hdr for k in ("CD1_1","CD1_2","CD2_1","CD2_2")):
                        cd11 = float(hdr["CD1_1"]); cd12 = float(hdr["CD1_2"])
                        cd21 = float(hdr["CD2_1"]); cd22 = float(hdr["CD2_2"])
                        sx_deg = (cd11**2 + cd12**2) ** 0.5
                        sy_deg = (cd21**2 + cd22**2) ** 0.5
                        return abs(sx_deg) * 3600.0, abs(sy_deg) * 3600.0
                    if ("CDELT1" in hdr) and ("CDELT2" in hdr):
                        cdelt1 = float(hdr["CDELT1"]); cdelt2 = float(hdr["CDELT2"])
                        pc11 = float(hdr.get("PC1_1", 1.0)); pc12 = float(hdr.get("PC1_2", 0.0))
                        pc21 = float(hdr.get("PC2_1", 0.0)); pc22 = float(hdr.get("PC2_2", 1.0))
                        m11 = cdelt1 * pc11; m12 = cdelt1 * pc12
                        m21 = cdelt2 * pc21; m22 = cdelt2 * pc22
                        sx_deg = (m11**2 + m12**2) ** 0.5
                        sy_deg = (m21**2 + m22**2) ** 0.5
                        return abs(sx_deg) * 3600.0, abs(sy_deg) * 3600.0
                    fl_mm = float(hdr.get("FOCALLEN", 0.0) or 0.0)
                    if fl_mm <= 0:
                        return (None, None)
                    pxsize1 = hdr.get("PIXSIZE1"); pxsize2 = hdr.get("PIXSIZE2")
                    if (pxsize1 is not None) or (pxsize2 is not None):
                        px_um = float(pxsize1 or pxsize2 or 0.0)
                        py_um = float(pxsize2 or pxsize1 or px_um)
                        xb = int(hdr.get("XBINNING", hdr.get("BINX", 1)) or 1)
                        yb = int(hdr.get("YBINNING", hdr.get("BINY", 1)) or 1)
                        sx = 206.265 * (px_um * xb) / fl_mm
                        sy = 206.265 * (py_um * yb) / fl_mm
                        return sx, sy
                    xpixsz = hdr.get("XPIXSZ"); ypixsz = hdr.get("YPIXSZ")
                    if (xpixsz is not None) or (ypixsz is not None):
                        px_um = float(xpixsz or ypixsz or 0.0)
                        py_um = float(ypixsz or xpixsz or px_um)
                        sx = 206.265 * px_um / fl_mm
                        sy = 206.265 * py_um / fl_mm
                        return sx, sy
                except Exception:
                    pass
                return (None, None)

            for fp in measured_frames:
                try:
                    hdr0 = fits.getheader(fp, ext=0)
                except Exception:
                    hdr0 = {}
                self.main.arcsec_per_px[fp] = _scale_from_header_raw(hdr0)

            # --- Build per-frame raw header scale (arcsec/px) ---
            raw_scale = {}  # fp -> (sx, sy)
            for fp in all_files:
                try:
                    hdr0 = fits.getheader(fp, ext=0)
                except Exception:
                    hdr0 = {}
                raw_scale[fp] = _robust_scale_from_header(hdr0)

            # ---- Scale normalization policy ----
            groups_count = len(self.main.light_files or {})
            skip_single = bool(self.main.settings.value("stacking/skip_scale_if_single_group", True, type=bool))
            tol_pct = float(self.main.settings.value("stacking/scale_tol_pct", 5.0, type=float))  # you said 5%
            tol = tol_pct / 100.0

            # If only one group and policy says skip, we don't even *compute* scales.
            if groups_count == 1 and skip_single:
                target_sx = target_sy = None
                skip_scale_norm = True
                self.main.update_status("⏭️ Pixel-scale normalize skipped: single-group policy (no header reads).")
            else:
                # --- multi-group (or policy off): compute target from medians ---
                # Build per-file raw header scale (arcsec/px)
                raw_scale = {}
                for fp in all_files:
                    try:
                        hdr0 = fits.getheader(fp, ext=0)
                    except Exception:
                        hdr0 = {}
                    raw_scale[fp] = _robust_scale_from_header(hdr0)

                # Effective scales at the chosen target bin
                # --- Compute target pixel scale = highest resolution across frames (min arcsec/px) ---
                eff_sx_list, eff_sy_list = [], []
                for fp in all_files:
                    try:
                        hdr0 = fits.getheader(fp, ext=0)
                    except Exception:
                        hdr0 = {}
                    sx, sy = _robust_scale_from_header(hdr0)
                    if sx and sy:
                        eff_sx_list.append(float(sx))
                        eff_sy_list.append(float(sy))

                if eff_sx_list and eff_sy_list:
                    # pick the *smallest* arcsec/px (best resolution) so we never downsample the hi-res set
                    target_sx = float(min(eff_sx_list))
                    target_sy = float(min(eff_sy_list))
                    self.main.update_status(f"🎯 Target pixel scale (hi-res): {target_sx:.3f}\"/px × {target_sy:.3f}\"/px")
                else:
                    target_sx = target_sy = None
                    self.main.update_status("🎯 Target pixel scale unknown (no WCS/pixel size). Will skip scale normalization.")


                # Decide skip for single-group+uniform bin case (only reached if policy didn't auto-skip)
                uniform_binning = (min(b[0] for b in bin_map.values()) == max(b[0] for b in bin_map.values())
                                and min(b[1] for b in bin_map.values()) == max(b[1] for b in bin_map.values()))
                skip_scale_norm = False
                if groups_count == 1 and uniform_binning and eff_sx_list and eff_sy_list:
                    med_sx = float(np.median(eff_sx_list))
                    med_sy = float(np.median(eff_sy_list))
                    max_dev = max(max(_rel_delta(x, med_sx) for x in eff_sx_list),
                                max(_rel_delta(y, med_sy) for y in eff_sy_list))
                    if max_dev <= tol:
                        skip_scale_norm = True
                        self.main.update_status(f"⏭️ Pixel-scale normalize skipped: spread ≤ {tol_pct:.2f}% (single group).")
                    else:
                        self.main.update_status(f"ℹ️ Single group spread {max_dev*100:.3f}% > tol {tol_pct:.2f}% → will normalize.")

            do_scale_norm = (not skip_scale_norm) and (target_sx is not None) and (target_sy is not None)

            # ─────────────────────────────────────────────────────────────────────
            # PHASE 2: normalize (DEBAYER everything once here)
            # ─────────────────────────────────────────────────────────────────────
            norm_dir = os.path.join(self.main.stacking_directory, "Normalized_Images")
            os.makedirs(norm_dir, exist_ok=True)

            ocv_prev = None
            try:
                ocv_prev = cv2.getNumThreads()
                cv2.setNumThreads(max(2, min(8, (os.cpu_count() or 8)//2)))
            except Exception:
                pass

            normalized_files = []
            chunks = list(chunk_list(measured_frames, chunk_size))
            total_chunks = len(chunks)

            ncpu = os.cpu_count() or 8
            io_workers = min(8, max(2, ncpu // 2))

            ref_H, ref_W = ref_img.shape[:2] if ref_img.ndim == 2 else ref_img.shape[:2]
            self.main._norm_target_hw = (int(ref_H), int(ref_W))

            for idx, chunk in enumerate(chunks, 1):
                self.main.update_status(f"🌀 Normalizing chunk {idx}/{total_chunks} ({len(chunk)} frames)…")
                QApplication.processEvents()

                abe_enabled = bool(self.main.settings.value("stacking/grad_poly2/enabled", False, type=bool))
                if abe_enabled:
                    mode         = "divide" if self.main.settings.value("stacking/grad_poly2/mode", "subtract") == "divide" else "subtract"
                    samples      = int(self.main.settings.value("stacking/grad_poly2/samples", 120, type=int))
                    downsample   = int(self.main.settings.value("stacking/grad_poly2/downsample", 6, type=int))
                    patch_size   = int(self.main.settings.value("stacking/grad_poly2/patch_size", 15, type=int))
                    min_strength = float(self.main.settings.value("stacking/grad_poly2/min_strength", 0.01, type=float))
                    gain_lo      = float(self.main.settings.value("stacking/grad_poly2/gain_lo", 0.20, type=float))
                    gain_hi      = float(self.main.settings.value("stacking/grad_poly2/gain_hi", 5.0, type=float))

                scaled_images = []; scaled_paths = []; scaled_hdrs = []

                from concurrent.futures import ThreadPoolExecutor, as_completed
                self.main.update_status(f"🌍 Loading {len(chunk)} images in parallel for normalization (up to {io_workers} threads)…")
                with ThreadPoolExecutor(max_workers=io_workers) as ex:
                    futs = {ex.submit(self.main._load_image_any, fp): fp for fp in chunk}
                    for fut in as_completed(futs):
                        fp = futs[fut]
                        try:
                            img, hdr = fut.result()
                            if img is None:
                                self.main.update_status(f"⚠️ No data for {fp}")
                                continue

                            img = _to_writable_f32(img)

                            bayer = self.main._hdr_get(hdr, 'BAYERPAT')
                            splitdb = bool(self.main._hdr_get(hdr, 'SPLITDB', False))
                            if bayer and not splitdb and (img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)):
                                self.main.update_status(f"📦 Debayering {os.path.basename(fp)}…")
                                img = self.main.debayer_image(img, fp, hdr)  # HxWx3
                            else:
                                if img.ndim == 3 and img.shape[-1] == 1:
                                    img = np.squeeze(img, axis=-1)

                            # meridian flip assist
                            if self.main.auto_rot180 and ref_pa is not None:
                                pa = self.main._extract_pa_deg(hdr)
                                img, did = self.main._maybe_rot180(img, pa, ref_pa, self.main.auto_rot180_tol_deg)
                                if did:
                                    self.main.update_status(f"↻ 180° rotate (PA Δ≈180°): {os.path.basename(fp)}")
                                    try:
                                        if hasattr(hdr, "__setitem__"):
                                            hdr['ROT180'] = (True, 'Rotated 180° pre-align by SAS')
                                    except Exception:
                                        pass

                            # --- ONE geometry normalization path ---
                            if do_scale_norm:
                                # We normalize to physical pixel scale (arcsec/px) → this ALSO compensates binning,
                                # so we must NOT pre-resample to target bin.
                                raw_psx, raw_psy = _robust_scale_from_header(hdr)  # arcsec/px as shot
                                if raw_psx and raw_psy and target_sx and target_sy:
                                    gx = float(raw_psx) / float(target_sx)
                                    gy = float(raw_psy) / float(target_sy)

                                    # clamp tiny jitter
                                    if _rel_delta(gx, 1.0) <= tol:
                                        gx = 1.0
                                    if _rel_delta(gy, 1.0) <= tol:
                                        gy = 1.0

                                    if (gx != 1.0) or (gy != 1.0):
                                        before_hw = img.shape[:2]
                                        img = _resize_to_scale(img, gx, gy)
                                        after_hw = img.shape[:2]
                                        self.main.update_status(
                                            f"📏 Pixel-scale normalize {raw_psx:.3f}\"/{raw_psy:.3f}\" → "
                                            f"{target_sx:.3f}\"/{target_sy:.3f}\" | "
                                            f"size {before_hw[1]}×{before_hw[0]} → {after_hw[1]}×{after_hw[0]}"
                                        )
                            else:
                                # We are NOT doing physical/pixel-scale normalization (single group, within tol).
                                # In that case we ONLY need to unify binning → simple pixel resample.
                                xb, yb = bin_map.get(fp, (1, 1))
                                sx = float(xb) / float(target_xbin)
                                sy = float(yb) / float(target_ybin)
                                if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
                                    before = img.shape[:2]
                                    img = _resize_to_scale(img, sx, sy)
                                    after = img.shape[:2]
                                    self.main.update_status(
                                        f"🔧 Resampled for binning {xb}×{yb} → {target_xbin}×{target_ybin} "
                                        f"size {before[1]}×{before[0]} → {after[1]}×{after[0]}"
                                    )



                            # 3) Brightness normalization / scale refine
                            pm = float(preview_medians.get(fp, 0.0))
                            s = _compute_scale(ref_target_median, pm if pm > 0 else 1.0,
                                            img, refine_stride=8, refine_if_rel_err=0.10)
                            img = _apply_scale_inplace(img, s)

                            # 🔒 4) Enforce canonical geometry BEFORE ABE / writing
                            if hasattr(self, "_norm_target_hw") and self.main._norm_target_hw:
                                img = _force_shape_hw(img, *self.main._norm_target_hw)

                            if abe_enabled:
                                scaled_images.append(img.astype(np.float32, copy=False))
                                scaled_paths.append(fp)
                                scaled_hdrs.append(hdr)
                            else:
                                # write out normalized FITS
                                base = os.path.basename(fp)
                                if base.endswith("_n.fit"):
                                    base = base.replace("_n.fit", ".fit")
                                if base.lower().endswith(".fits"):
                                    out_name = base[:-5] + "_n.fit"
                                elif base.lower().endswith(".fit"):
                                    out_name = base[:-4] + "_n.fit"
                                else:
                                    out_name = base + "_n.fit"
                                out_path = os.path.join(norm_dir, out_name)

                                try:
                                    if os.path.splitext(fp)[1].lower() in (".fits", ".fit", ".fz"):
                                        orig_header = fits.getheader(fp, ext=0)
                                    else:
                                        orig_header = fits.Header()
                                except Exception:
                                    orig_header = fits.Header()

                                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                                    orig_header["DEBAYERED"] = (True, "Color debayered normalized")
                                else:
                                    orig_header["DEBAYERED"] = (False, "Mono normalized")

                                from os import path
                                _key = path.normcase(path.normpath(fp))
                                _val = path.normpath(out_path)
                                self.main._orig2norm[_key] = _val
                                fits.PrimaryHDU(data=img.astype(np.float32), header=orig_header).writeto(out_path, overwrite=True)
                                normalized_files.append(out_path)

                        except Exception as e:
                            self.main.update_status(f"⚠️ Error normalizing {fp}: {e}")
                        finally:
                            QApplication.processEvents()

                # 2) ABE with canonical size lock
                if abe_enabled and scaled_images:
                    self.main.update_status(
                        f"Gradient removal (ABE Poly²): mode={mode}, samples={samples}, "
                        f"downsample={downsample}, patch={patch_size}, min_strength={min_strength*100:.2f}%, "
                        f"gain_clip=[{gain_lo},{gain_hi}]"
                    )
                    QApplication.processEvents()

                    abe_stack = np.ascontiguousarray(np.stack(scaled_images, axis=0, dtype=np.float32))
                    abe_stack = remove_gradient_stack_abe(
                        abe_stack,
                        target_hw=getattr(self, "_norm_target_hw", None),  # 🔒 enforce (H,W) for every frame
                        mode=mode,
                        num_samples=samples,
                        downsample=downsample,
                        patch_size=patch_size,
                        min_strength=min_strength,
                        gain_clip=(gain_lo, gain_hi),
                        log_fn=(self.main._ui_log if hasattr(self, "_ui_log") else self.main.update_status),
                    )
                    QApplication.processEvents()

                    for i, fp in enumerate(scaled_paths):
                        img_out = abe_stack[i]; hdr = scaled_hdrs[i]
                        base = os.path.basename(fp)
                        if base.endswith("_n.fit"):
                            base = base.replace("_n.fit", ".fit")
                        if base.lower().endswith(".fits"):
                            out_name = base[:-5] + "_n.fit"
                        elif base.lower().endswith(".fit"):
                            out_name = base[:-4] + "_n.fit"
                        else:
                            out_name = base + "_n.fit"
                        out_path = os.path.join(norm_dir, out_name)

                        try:
                            orig_header = fits.getheader(fp, ext=0)
                        except Exception:
                            orig_header = fits.Header()

                        if isinstance(img_out, np.ndarray) and img_out.ndim == 3 and img_out.shape[-1] == 3:
                            orig_header["DEBAYERED"] = (True, "Color debayered normalized")
                        else:
                            orig_header["DEBAYERED"] = (False, "Mono normalized")

                        from os import path
                        _key = path.normcase(path.normpath(fp))
                        _val = path.normpath(out_path)
                        self.main._orig2norm[_key] = _val
                        fits.PrimaryHDU(data=img_out.astype(np.float32), header=orig_header).writeto(out_path, overwrite=True)
                        normalized_files.append(out_path)

            # restore OpenCV threads
            try:
                if ocv_prev is not None:
                    cv2.setNumThreads(ocv_prev)
            except Exception:
                pass

            # Update self.main.light_files to *_n.fit
            for group, file_list in self.main.light_files.items():
                new_list = []
                for old_path in file_list:
                    base = os.path.basename(old_path)
                    if base.endswith("_n.fit"):
                        new_list.append(os.path.join(norm_dir, base))
                    else:
                        if base.lower().endswith(".fits"):
                            n_name = base[:-5] + "_n.fit"
                        elif base.lower().endswith(".fit"):
                            n_name = base[:-4] + "_n.fit"
                        else:
                            n_name = base + "_n.fit"
                        new_list.append(os.path.join(norm_dir, n_name))
                self.main.light_files[group] = new_list

            self.main.update_status("✅ Updated self.main.light_files to use debayered, normalized *_n.fit frames.")

            from os import path
            ref_path = path.normpath(self.main.reference_frame)
            self.main.update_status(f"📌 Reference for alignment (verbatim): {ref_path}")
            if not path.exists(ref_path):
                self.main.update_status(f"🚨 Reference file does not exist: {ref_path}")
                return

            # ─────────────────────────────────────────────────────────────────────
            # Start alignment on the normalized files
            # ─────────────────────────────────────────────────────────────────────
            align_dir = os.path.join(self.main.stacking_directory, "Aligned_Images")
            os.makedirs(align_dir, exist_ok=True)

            passes = self.main.settings.value("stacking/refinement_passes", 3, type=int)
            shift_tol = self.main.settings.value("stacking/shift_tolerance", 0.2, type=float)

            normalized_files = [path.normpath(p) for p in normalized_files]

            self.main.alignment_thread = StarRegistrationThread(
                ref_path,  
                normalized_files,
                align_dir,
                max_refinement_passes=passes,
                shift_tolerance=shift_tol,
                parent_window=self.main
            )
            try:
                self.main.alignment_thread.progress_update.disconnect(self.main.update_status)
            except TypeError:
                pass
            self.main.alignment_thread.progress_update.connect(
                self.main.update_status, Qt.ConnectionType.QueuedConnection
            )

            self.main.alignment_thread.registration_complete.connect(self.on_registration_complete)

            self.main.align_progress = QProgressDialog("Aligning stars…", None, 0, 0, self.main)
            self.main.align_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.main.align_progress.setMinimumDuration(0)
            self.main.align_progress.setCancelButton(None)
            self.main.align_progress.setWindowTitle("Stellar Alignment")
            self.main.align_progress.setValue(0)
            self.main.align_progress.show()

            try:
                self.main.alignment_thread.progress_step.disconnect(self.main._on_align_progress)
            except TypeError:
                pass
            self.main.alignment_thread.progress_step.connect(
                self.main._on_align_progress, Qt.ConnectionType.QueuedConnection
            )
            try:
                self.main.alignment_thread.registration_complete.disconnect(self.main._on_align_done)
            except TypeError:
                pass
            self.main.alignment_thread.registration_complete.connect(
                self.main._on_align_done, Qt.ConnectionType.QueuedConnection
            )
            self.main.alignment_thread.start()

        except Exception as e:
            self.main._set_registration_busy(False)
            raise

        

    @pyqtSlot(bool, str)

    def on_registration_complete(self, success, msg):
       
        self.main.update_status(msg)
        if not success:
            self.main._set_registration_busy(False)
            return

        alignment_thread = self.main.alignment_thread
        if alignment_thread is None:
            self.main.update_status("⚠️ Error: No alignment data available.")
            self.main._set_registration_busy(False) 
            return

        # ----------------------------
        # Gather results from the thread
        # ----------------------------
        all_transforms = dict(alignment_thread.alignment_matrices)  # {orig_norm_path -> 2x3 or None}
        keys = list(all_transforms.keys())

        # Build a per-file shift map (last pass), defaulting to 0 when missing
        shift_map = {}
        if alignment_thread.transform_deltas and alignment_thread.transform_deltas[-1]:
            last = alignment_thread.transform_deltas[-1]
            for i, k in enumerate(keys):
                if i < len(last):
                    shift_map[k] = float(last[i])
                else:
                    shift_map[k] = 0.0
        else:
            shift_map = {k: 0.0 for k in keys}

        # Fast mode if only 1 pass was requested
        fast_mode = (getattr(alignment_thread, "max_refinement_passes", 3) <= 1)
        # Threshold is only used in normal mode
        accept_thresh = float(self.main.settings.value("stacking/accept_shift_px", 2.0, type=float))

        def _accept(k: str) -> bool:
            """Accept criteria for a frame."""
            if all_transforms.get(k) is None:
                # real failure (e.g., astroalign couldn't find a transform)
                return False
            if fast_mode:
                # In fast mode we keep everything that didn't fail
                return True
            # Normal (multi-pass) behavior: keep small last-pass shifts
            return shift_map.get(k, 0.0) <= accept_thresh

        accepted = [k for k in keys if _accept(k)]
        rejected = [k for k in keys if not _accept(k)]

        # ----------------------------
        # Persist numeric transforms we accepted (for drizzle, etc.)
        # ----------------------------
        valid_matrices = {k: all_transforms[k] for k in accepted}
        self.main.valid_matrices = {
            os.path.normpath(k): np.asarray(v, dtype=np.float32)
            for k, v in valid_matrices.items() if v is not None
        }

        # ✅ Write SASD v2 using model-aware transforms captured by the thread
        try:
            sasd_out = os.path.join(self.main.stacking_directory, "alignment_transforms.sasd")
            # pull over the per-file model-aware xforms and reference info
            self.main.drizzle_xforms = dict(getattr(alignment_thread, "drizzle_xforms", {}))
            self.main.ref_shape_for_drizzle = tuple(getattr(alignment_thread, "reference_image_2d", np.zeros((1,1), np.float32)).shape[:2])
            self.main.ref_path_for_drizzle  = alignment_thread.reference if isinstance(alignment_thread.reference, str) else "__ACTIVE_VIEW__"
            self.main._save_alignment_transforms_sasd_v2(
                out_path=sasd_out,
                ref_shape=self.main.ref_shape_for_drizzle,
                ref_path=self.main.ref_path_for_drizzle,
                drizzle_xforms=self.main.drizzle_xforms,
                fallback_affine=self.main.valid_matrices,  # in case a few files missed model-aware
            )
            self.main.update_status("✅ Transform file saved as alignment_transforms.sasd (v2)")
        except Exception as e:
            self.main.update_status(f"⚠️ Failed to write SASD v2 ({e}); writing affine-only fallback.")
            self.save_alignment_matrices_sasd(valid_matrices)  # old writer as last resort


        try:
            if self.main._comet_seed and self.main.reference_frame and getattr(self, "valid_matrices", None):
                seed_orig = os.path.normpath(self.main._comet_seed["path"])
                seed_xy   = self.main._comet_seed["xy"]

                # find the normalized counterpart of the original seed frame
                seed_norm = self.main._orig2norm.get(seed_orig)
                if not seed_norm:
                    # if the seed was picked on an already-normalized path, try that as-is
                    if seed_orig in self.main.valid_matrices:
                        seed_norm = seed_orig

                M = self.main.valid_matrices.get(os.path.normpath(seed_norm)) if seed_norm else None
                if M is not None:
                    x, y = seed_xy
                    X = float(M[0,0]*x + M[0,1]*y + M[0,2])
                    Y = float(M[1,0]*x + M[1,1]*y + M[1,2])
                    self.main._comet_ref_xy = (X, Y)
                    self.main.update_status(f"🌠 Comet anchor in reference frame: ({X:.1f}, {Y:.1f})")
                else:
                    self.main.update_status("ℹ️ Could not resolve comet seed to reference (no matrix for that frame).")
        except Exception as e:
            self.main.update_status(f"⚠️ Comet seed resolve failed: {e}")

        # ----------------------------
        # Build mapping from normalized -> aligned paths
        # Use the *actual* final paths produced by the thread.
        # ----------------------------
        final_map = alignment_thread.file_key_to_current_path  # {orig_norm_path -> final_aligned_path}
        self.main.valid_transforms = {
            os.path.normpath(k): os.path.normpath(final_map[k])
            for k in accepted
            if k in final_map and os.path.exists(final_map[k])
        }

        self.main.matrix_by_aligned = {}
        self.main.orig_by_aligned   = {}
        for norm_path, aligned_path in self.main.valid_transforms.items():
            M = self.main.valid_matrices.get(norm_path)
            if M is not None:
                self.main.matrix_by_aligned[os.path.normpath(aligned_path)] = M
            self.main.orig_by_aligned[os.path.normpath(aligned_path)] = os.path.normpath(norm_path)
        # For drizzle: keep model-aware xforms and reference geometry
        self.main.drizzle_xforms = dict(getattr(alignment_thread, "drizzle_xforms", {}))
        self.main.ref_shape_for_drizzle = tuple(getattr(alignment_thread, "reference_image_2d", np.zeros((1,1), np.float32)).shape[:2])
        self.main.ref_path_for_drizzle  = alignment_thread.reference if isinstance(alignment_thread.reference, str) else "__ACTIVE_VIEW__"

        # finalize alignment phase
        self.main.alignment_thread = None

        # Status
        prefix = "⚡ Fast mode: " if fast_mode else ""
        self.main.update_status(f"{prefix}Alignment summary: {len(accepted)} succeeded, {len(rejected)} rejected.")
        QApplication.processEvents()
        if (not fast_mode) and rejected:
            self.main.update_status(f"🚨 Rejected {len(rejected)} frame(s) due to shift > {accept_thresh}px.")
            for rf in rejected:
                self.main.update_status(f"  ❌ {os.path.basename(rf)}")

        if not self.main.valid_transforms:
            self.main.update_status("⚠️ No frames to stack; aborting.")
            self.main._set_registration_busy(False)
            return

        # ----------------------------
        # Build aligned file groups (unchanged)
        # ----------------------------
        filtered_light_files = {}
        for group, file_list in self.main.light_files.items():
            filtered = [f for f in file_list if os.path.normpath(f) in self.main.valid_transforms]
            filtered_light_files[group] = filtered
            self.main.update_status(f"Group '{group}' has {len(filtered)} file(s) after filtering.")
            QApplication.processEvents()

        aligned_light_files = {}
        for group, file_list in filtered_light_files.items():
            new_list = []
            for f in file_list:
                normed = os.path.normpath(f)
                aligned = self.main.valid_transforms.get(normed)
                if aligned and os.path.exists(aligned):
                    new_list.append(aligned)
                else:
                    self.main.update_status(f"DEBUG: File '{aligned}' does not exist on disk.")
            aligned_light_files[group] = new_list

        def _start_after_align_worker(aligned_light_files: dict[str, list[str]]):
            # ----------------------------
            # Snapshot UI-dependent settings (your existing code)
            # ----------------------------
            drizzle_dict = self.main.gather_drizzle_settings_from_tree()
            try:
                autocrop_enabled = self.main.autocrop_cb.isChecked()
                autocrop_pct = float(self.main.autocrop_pct.value())
            except Exception:
                autocrop_enabled = self.main.settings.value("stacking/autocrop_enabled", False, type=bool)
                autocrop_pct = float(self.main.settings.value("stacking/autocrop_pct", 95.0, type=float))

            cfa_effective = bool(
                self.main._cfa_for_this_run
                if getattr(self, "_cfa_for_this_run", None) is not None
                else (getattr(self, "cfa_drizzle_cb", None) and self.main.cfa_drizzle_cb.isChecked())
            )
            if cfa_effective and getattr(self, "valid_matrices", None):
                fill = self.main._dither_phase_fill(self.main.valid_matrices, bins=8)
                self.main.update_status(f"🔎 CFA drizzle sub-pixel phase fill (8×8): {fill*100:.1f}%")
                if fill < 0.65:
                    self.main.update_status("💡 For best results with CFA drizzle, aim for >65% fill.")
                    self.main.update_status("   With <~40–55% fill, expect visible patching even with many frames)")
            QApplication.processEvents()

            # ----------------------------
            # Kick off post-align worker (unchanged body)
            # ----------------------------
            self.main.post_thread = QThread(self)
            self.main.post_worker = AfterAlignWorker(
                self,
                light_files=aligned_light_files,
                frame_weights=dict(self.main.frame_weights),
                transforms_dict=dict(self.main.valid_transforms),
                drizzle_dict=drizzle_dict,
                autocrop_enabled=autocrop_enabled,
                autocrop_pct=autocrop_pct,
                ui_owner=self
            )
            self.main.post_worker.ui_owner = self
            self.main.post_worker.need_comet_review.connect(self.main.on_need_comet_review)
            self.main.post_worker.progress.connect(self.main._on_post_status)
            self.main.post_worker.finished.connect(self.main._on_post_pipeline_finished)
            self.main.post_worker.moveToThread(self.main.post_thread)
            self.main.post_thread.started.connect(self.main.post_worker.run)
            self.main.post_thread.start()

            self.main.post_progress = QProgressDialog("Stacking & drizzle (if enabled)…", None, 0, 0, self.main)
            self.main.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.main.post_progress.setCancelButton(None)
            self.main.post_progress.setMinimumDuration(0)
            self.main.post_progress.setWindowTitle("Post-Alignment")
            self.main.post_progress.show()

            self.main._set_registration_busy(False)

        try:
            autocrop_enabled_ui = self.main.autocrop_cb.isChecked()
            autocrop_pct_ui = float(self.main.autocrop_pct.value())
        except Exception:
            autocrop_enabled_ui = self.main.settings.value("stacking/autocrop_enabled", False, type=bool)
            autocrop_pct_ui = float(self.main.settings.value("stacking/autocrop_pct", 95.0, type=float))

        # Build a single global rect from all aligned frames (registered paths)
        _mf_global_rect = None
        if autocrop_enabled_ui:
            pd = QProgressDialog("Calculating autocrop bounding box…", None, 0, 0, self.main)
            pd.setWindowTitle("Auto Crop")
            pd.setWindowModality(Qt.WindowModality.WindowModal)
            pd.setCancelButton(None)
            pd.setMinimumDuration(0)
            pd.setRange(0, 0)
            pd.show()
            QApplication.processEvents()

            try:
                self.main.update_status("✂️ (MF) Auto Crop: using transform footprints…")
                QApplication.processEvents()

                # Prefer model-aware (drizzle) transforms, else affine fallback
                xforms = (dict(getattr(self, "drizzle_xforms", {}))
                        if getattr(self, "drizzle_xforms", None)
                        else dict(getattr(self, "valid_matrices", {})))
                ref_hw = tuple(getattr(self, "ref_shape_for_drizzle", ()))  # (H,W)

                if xforms and len(ref_hw) == 2:
                    _mf_global_rect = self.main._rect_from_transforms_fast(
                        xforms,
                        src_hw=ref_hw,
                        coverage_pct=float(autocrop_pct_ui),
                        allow_homography=True,
                        min_side=16
                    )
                    if _mf_global_rect:
                        x0,y0,x1,y1 = _mf_global_rect
                        self.main.update_status(
                            f"✂️ (MF) Transform crop → [{x0}:{x1}]×[{y0}:{y1}] "
                            f"({x1-x0}×{y1-y0})"
                        )
                    else:
                        self.main.update_status("✂️ (MF) Transform crop yielded no valid rect; falling back to mask-based method…")
                else:
                    self.main.update_status("✂️ (MF) No transforms/geometry available; falling back to mask-based method…")

                # Fallback to existing (mask-based) method if needed
                if _mf_global_rect is None:
                    _mf_global_rect = self.main._compute_common_autocrop_rect(
                        aligned_light_files,
                        autocrop_pct_ui,
                        status_cb=self.main.update_status
                    )
                    if _mf_global_rect:
                        x0,y0,x1,y1 = _mf_global_rect
                        self.main.update_status(
                            f"✂️ (MF) Mask-based crop → [{x0}:{x1}]×[{y0}:{y1}] "
                            f"({x1-x0}×{y1-y0})"
                        )
                    else:
                        self.main.update_status("✂️ (MF) Auto-crop disabled (no common region).")

                QApplication.processEvents()
            except Exception as e:
                self.main.update_status(f"⚠️ (MF) Global crop failed: {e}")
                _mf_global_rect = None
            finally:
                try:
                    pd.reset()
                    pd.deleteLater()
                except Exception:
                    pass

        self.main._mf_autocrop_rect   = _mf_global_rect
        self.main._mf_autocrop_enabled = bool(autocrop_enabled_ui)
        self.main._mf_autocrop_pct     = float(autocrop_pct_ui)


        mf_enabled = self.main.settings.value("stacking/mfdeconv/enabled", False, type=bool)

        if mf_enabled:
            self.main.update_status("🧪 Multi-frame PSF-aware deconvolution path enabled.")

            mf_groups = [(g, lst) for g, lst in aligned_light_files.items() if lst]
            if not mf_groups:
                self.main.update_status("⚠️ No aligned frames available for MF deconvolution.")
            else:
                self.main._mf_pd = QProgressDialog("Multi-frame deconvolving…", "Cancel", 0, len(mf_groups), self.main)
                # self.main._mf_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
                self.main._mf_pd.setMinimumDuration(0)
                self.main._mf_pd.setWindowTitle("MF Deconvolution")
                self.main._mf_pd.setValue(0)
                self.main._mf_pd.show()

                if getattr(self.main, "_mf_pd", None):
                    self.main._mf_pd.setLabelText("Preparing MF deconvolution…")
                    self.main._mf_pd.setMinimumWidth(520)

                self.main._mf_total_groups = len(mf_groups)
                self.main._mf_groups_done = 0
                # progress range = groups * 1000 (each group gets 0..1000)
                self.main._mf_pd.setRange(0, self.main._mf_total_groups * 1000)
                self.main._mf_pd.setValue(0)
                self.main._mf_queue = list(mf_groups)
                self.main._mf_results = {}
                self.main._mf_cancelled = False
                self.main._mf_thread = None
                self.main._mf_worker = None

                def _cancel_all():
                    self.main._mf_cancelled = True
                self.main._mf_pd.canceled.connect(_cancel_all, Qt.ConnectionType.QueuedConnection)

                def _finish_mf_phase_and_exit():
                    """Tear down MF UI/threads and either continue or exit."""
                    if getattr(self.main, "_mf_pd", None):
                        try:
                            self.main._mf_pd.reset()
                            self.main._mf_pd.deleteLater()
                        except Exception:
                            pass
                        self.main._mf_pd = None
                    try:
                        if self.main._mf_thread:
                            self.main._mf_thread.quit()
                            self.main._mf_thread.wait()
                    except Exception:
                        pass
                    self.main._mf_thread = None
                    self.main._mf_worker = None

                    run_after = self.main.settings.value("stacking/mfdeconv/after_mf_run_integration", False, type=bool)
                    if run_after:
                        _start_after_align_worker(aligned_light_files)
                    else:
                        self.main.update_status("✅ MFDeconv complete for all groups. Skipping normal integration as requested.")
                        self.main._set_registration_busy(False)

                def _start_next_mf_job():
                    # end of queue or canceled → finish
                    if self.main._mf_cancelled or not self.main._mf_queue:
                        _finish_mf_phase_and_exit()
                        return

                    group_key, frames = self.main._mf_queue.pop(0)

                    out_dir = os.path.join(self.main.stacking_directory, "Masters")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"MasterLight_{group_key}_MFDeconv.fit")

                    # ── read config ─────────────────────────────────────────
                    iters = self.main.settings.value("stacking/mfdeconv/iters", 20, type=int)
                    min_iters = self.main.settings.value("stacking/mfdeconv/min_iters", 3, type=int)
                    kappa = self.main.settings.value("stacking/mfdeconv/kappa", 2.0, type=float)
                    mode  = self.main.mf_color_combo.currentText()
                    Huber = self.main.settings.value("stacking/mfdeconv/Huber_delta", 0.0, type=float)
                    seed_mode_cfg = str(self.main.settings.value("stacking/mfdeconv/seed_mode", "robust"))
                    use_star_masks    = self.main.mf_use_star_mask_cb.isChecked()
                    use_variance_maps = self.main.mf_use_noise_map_cb.isChecked()
                    rho               = self.main.mf_rho_combo.currentText()
                    save_intermediate = self.main.mf_save_intermediate_cb.isChecked()

                    sr_enabled_ui = self.main.mf_sr_cb.isChecked()
                    sr_factor_ui  = getattr(self, "mf_sr_factor_spin", None)
                    sr_factor_val = sr_factor_ui.value() if sr_factor_ui is not None else self.main.settings.value("stacking/mfdeconv/sr_factor", 2, type=int)
                    super_res_factor = int(sr_factor_val) if sr_enabled_ui else 1

                    star_mask_cfg = {
                        "thresh_sigma":  self.main.settings.value("stacking/mfdeconv/star_mask/thresh_sigma",  _SM_DEF_THRESH, type=float),
                        "grow_px":       self.main.settings.value("stacking/mfdeconv/star_mask/grow_px",       _SM_DEF_GROW, type=int),
                        "soft_sigma":    self.main.settings.value("stacking/mfdeconv/star_mask/soft_sigma",    _SM_DEF_SOFT, type=float),
                        "max_radius_px": self.main.settings.value("stacking/mfdeconv/star_mask/max_radius_px", _SM_DEF_RMAX, type=int),
                        "max_objs":      self.main.settings.value("stacking/mfdeconv/star_mask/max_objs",      _SM_DEF_MAXOBJS, type=int),
                        "keep_floor":    self.main.settings.value("stacking/mfdeconv/star_mask/keep_floor",    _SM_DEF_KEEPF, type=float),
                        "ellipse_scale": self.main.settings.value("stacking/mfdeconv/star_mask/ellipse_scale", _SM_DEF_ES, type=float),
                    }
                    varmap_cfg = {
                        "sample_stride": self.main.settings.value("stacking/mfdeconv/varmap/sample_stride", _VM_DEF_STRIDE, type=int),
                        "smooth_sigma":  self.main.settings.value("stacking/mfdeconv/varmap/smooth_sigma", 1.0, type=float),
                        "floor":         self.main.settings.value("stacking/mfdeconv/varmap/floor",        1e-8, type=float),
                    }

                    self.main._mf_thread = QThread(self)
                    star_mask_ref = self.main.reference_frame if use_star_masks else None

                    # ── choose engine plainly (Normal / cuDNN-free / High Octane) ─────────────
                    # Expect a setting saved by your radio buttons: "normal" | "cudnn" | "sport"
                    engine = str(self.main.settings.value("stacking/mfdeconv/engine", "normal")).lower()

                    try:
                        if engine == "cudnn":
                            from pro.mfdeconvcudnn import MultiFrameDeconvWorkercuDNN as MFCls
                            eng_name = "Normal (cuDNN-free)"
                        elif engine == "sport":  # High Octane let 'er rip
                            from pro.mfdeconvsport import MultiFrameDeconvWorkerSport as MFCls
                            eng_name = "High Octane"
                        else:
                            from pro.mfdeconv import MultiFrameDeconvWorker as MFCls
                            eng_name = "Normal"
                    except Exception as e:
                        # if an import fails, fall back to the safe Normal path
                        self.main.update_status(f"⚠️ MFDeconv engine import failed ({e}); falling back to Normal.")
                        from pro.mfdeconv import MultiFrameDeconvWorker as MFCls
                        eng_name = "Normal (fallback)"

                    self.main.update_status(f"⚙️ MFDeconv engine: {eng_name}")

                    # ── build worker exactly the same in all modes ────────────────────────────
                    self.main._mf_worker = MFCls(
                        parent=None,
                        aligned_paths=frames,
                        output_path=out_path,
                        iters=iters,
                        kappa=kappa,
                        color_mode=mode,
                        huber_delta=Huber,
                        min_iters=min_iters,
                        use_star_masks=use_star_masks,
                        use_variance_maps=use_variance_maps,
                        rho=rho,
                        star_mask_cfg=star_mask_cfg,
                        varmap_cfg=varmap_cfg,
                        save_intermediate=save_intermediate,
                        super_res_factor=super_res_factor,
                        star_mask_ref_path=star_mask_ref,
                        seed_mode=seed_mode_cfg,
                    )

                    # ── standard Qt wiring (no gymnastics) ───────────────────────────────────
                    self.main._mf_worker.moveToThread(self.main._mf_thread)
                    self.main._mf_worker.progress.connect(self.main._on_mf_progress, Qt.ConnectionType.QueuedConnection)

                    # when the worker says finished, log & cleanup; the *thread* quitting will trigger starting the next job
                    def _job_finished(ok: bool, message: str, out: str):
                        if getattr(self.main, "_mf_pd", None):
                            self.main._mf_pd.setLabelText(f"{'✅' if ok else '❌'} {group_key}: {message}")
                        if ok and out:
                            self.main._mf_results[group_key] = out
                            if getattr(self, "_mf_autocrop_enabled", False) and getattr(self, "_mf_autocrop_rect", None):
                                try:
                                    from astropy.io import fits
                                    with fits.open(out, memmap=False) as hdul:
                                        img = hdul[0].data; hdr = hdul[0].header
                                    rect = tuple(map(int, self.main._mf_autocrop_rect))
                                    sr_enabled_ui = self.main.mf_sr_cb.isChecked()
                                    sr_factor_ui  = getattr(self, "mf_sr_factor_spin", None)
                                    sr_factor_val = sr_factor_ui.value() if sr_factor_ui is not None else self.main.settings.value("stacking/mfdeconv/sr_factor", 2, type=int)
                                    sr_factor = int(sr_factor_val) if sr_enabled_ui else 1
                                    if sr_factor > 1:
                                        x1,y1,x2,y2 = rect
                                        rect = (x1*sr_factor, y1*sr_factor, x2*sr_factor, y2*sr_factor)
                                    x1,y1,x2,y2 = rect
                                    if img.ndim == 2:
                                        crop = img[y1:y2, x1:x2]
                                    elif img.ndim == 3:
                                        crop = (img[:, y1:y2, x1:x2] if img.shape[0] in (1,3)
                                                else img[y1:y2, x1:x2, :])
                                    out_crop = out.replace(".fit", "_autocrop.fit").replace(".fits", "_autocrop.fits")
                                    fits.PrimaryHDU(data=crop.astype(np.float32, copy=False), header=hdr).writeto(out_crop, overwrite=True)
                                    self.main.update_status(f"✂️ (MF) Saved auto-cropped copy → {out_crop}")
                                except Exception as e:
                                    self.main.update_status(f"⚠️ (MF) Auto-crop of output failed: {e}")

                        # advance progress segment
                        if getattr(self.main, "_mf_pd", None):
                            self.main._mf_groups_done = min(self.main._mf_groups_done + 1, self.main._mf_total_groups)
                            self.main._mf_pd.setValue(self.main._mf_groups_done * 1000)

                    self.main._mf_worker.finished.connect(_job_finished, Qt.ConnectionType.QueuedConnection)

                    # thread start/stop
                    self.main._mf_thread.started.connect(self.main._mf_worker.run, Qt.ConnectionType.QueuedConnection)
                    self.main._mf_worker.finished.connect(self.main._mf_thread.quit, Qt.ConnectionType.QueuedConnection)
                    self.main._mf_thread.finished.connect(self.main._mf_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
                    self.main._mf_thread.finished.connect(self.main._mf_thread.deleteLater, Qt.ConnectionType.QueuedConnection)

                    # when the thread is fully down, kick the next job
                    def _next_after_thread():
                        try:
                            self.main._mf_thread.finished.disconnect(_next_after_thread)
                        except Exception:
                            pass
                        QTimer.singleShot(0, _start_next_mf_job)

                    self.main._mf_thread.finished.connect(_next_after_thread, Qt.ConnectionType.QueuedConnection)

                    # go
                    self.main._mf_thread.start()
                    if getattr(self.main, "_mf_pd", None):
                        self.main._mf_pd.setLabelText(f"Deconvolving '{group_key}' ({len(frames)} frames)…")


                # Kick off the first job (queue-driven)
                QTimer.singleShot(0, _start_next_mf_job)

                # Defer rest of pipeline; MF block will decide at MF completion.
                self.main._set_registration_busy(False)
                return



        # ----------------------------
        # Snapshot UI-dependent settings
        # ----------------------------
        drizzle_dict = self.main.gather_drizzle_settings_from_tree()
        try:
            autocrop_enabled = self.main.autocrop_cb.isChecked()
            autocrop_pct = float(self.main.autocrop_pct.value())
        except Exception:
            autocrop_enabled = self.main.settings.value("stacking/autocrop_enabled", False, type=bool)
            autocrop_pct = float(self.main.settings.value("stacking/autocrop_pct", 95.0, type=float))

        # Only report fill % if CFA mapping is actually in use for this run
        cfa_effective = bool(
            self.main._cfa_for_this_run
            if getattr(self, "_cfa_for_this_run", None) is not None
            else (getattr(self, "cfa_drizzle_cb", None) and self.main.cfa_drizzle_cb.isChecked())
        )
        print("CFA effective for this run:", cfa_effective)

        if cfa_effective and getattr(self, "valid_matrices", None):
            fill = self.main._dither_phase_fill(self.main.valid_matrices, bins=8)
            self.main.update_status(f"🔎 CFA drizzle sub-pixel phase fill (8×8): {fill*100:.1f}%")
            if fill < 0.65:
                self.main.update_status("💡 For best results with CFA drizzle, aim for >65% fill.")
                self.main.update_status("   With <~40–55% fill, expect visible patching even with many frames.")
        QApplication.processEvents()

        # ----------------------------
        # Kick off post-align worker (unchanged)
        # ----------------------------
        self.main.post_thread = QThread(self)
        self.main.post_worker = AfterAlignWorker(
            self,                                   # parent QObject
            light_files=aligned_light_files,
            frame_weights=dict(self.main.frame_weights),
            transforms_dict=dict(self.main.valid_transforms),
            drizzle_dict=drizzle_dict,
            autocrop_enabled=autocrop_enabled,
            autocrop_pct=autocrop_pct,
            ui_owner=self                           # 👈 PASS THE OWNER HERE
        )
        self.main.post_worker.ui_owner = self
        self.main.post_worker.need_comet_review.connect(self.main.on_need_comet_review) 

        self.main.post_worker.progress.connect(self.main._on_post_status)
        self.main.post_worker.finished.connect(self.main._on_post_pipeline_finished)

        self.main.post_worker.moveToThread(self.main.post_thread)
        self.main.post_thread.started.connect(self.main.post_worker.run)
        self.main.post_thread.start()

        self.main.post_progress = QProgressDialog("Stacking & drizzle (if enabled)…", None, 0, 0, self.main)
        self.main.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.main.post_progress.setCancelButton(None)
        self.main.post_progress.setMinimumDuration(0)
        self.main.post_progress.setWindowTitle("Post-Alignment")
        self.main.post_progress.show()

        self.main._set_registration_busy(False)


    def save_alignment_matrices_sasd(self, transforms_dict):
        out_path = os.path.join(self.main.stacking_directory, "alignment_transforms.sasd")
        try:
            with open(out_path, "w") as f:
                for norm_path, matrix in transforms_dict.items():
                    # Use the original normalized input path (e.g., *_n.fit)
                    orig_path = os.path.normpath(norm_path)

                    a, b, tx = matrix[0]
                    c, d, ty = matrix[1]

                    f.write(f"FILE: {orig_path}\n")
                    f.write("MATRIX:\n")
                    f.write(f"{a:.4f}, {b:.4f}, {tx:.4f}\n")
                    f.write(f"{c:.4f}, {d:.4f}, {ty:.4f}\n")
                    f.write("\n")  # blank line
            self.main.update_status(f"✅ Transform file saved as {os.path.basename(out_path)}")
        except Exception as e:
            self.main.update_status(f"⚠️ Failed to save transform file: {e}")




    def load_alignment_matrices_custom(self, file_path):
        transforms = {}
        with open(file_path, "r") as f:
            content = f.read()

        blocks = re.split(r"\n\s*\n", content.strip())
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            if not lines or not lines[0].startswith("FILE:"):
                continue
            curr_file = os.path.normpath(lines[0].replace("FILE:", "").strip())
            # find "MATRIX:" line
            try:
                m_idx = next(i for i,l in enumerate(lines) if l.startswith("MATRIX:"))
            except StopIteration:
                continue

            # Try parse 2×3 first, else 3×3
            try:
                r0 = [float(x) for x in lines[m_idx+1].split(",")]
                r1 = [float(x) for x in lines[m_idx+2].split(",")]
                if len(r0) == 3 and len(r1) == 3:
                    transforms[curr_file] = np.array([[r0[0], r0[1], r0[2]],
                                                    [r1[0], r1[1], r1[2]]], dtype=np.float32)
                    continue
                # 3×3
                r2 = [float(x) for x in lines[m_idx+3].split(",")]
                if len(r0) == len(r1) == len(r2) == 3:
                    H = np.array([r0, r1, r2], dtype=np.float32)
                    transforms[curr_file] = H
            except Exception:
                # skip malformed block
                continue
        return transforms



