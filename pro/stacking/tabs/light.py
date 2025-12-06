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


class LightTab(QObject):
    """Extracted Light tab functionality."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def create_light_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not hasattr(self, "manual_flat_overrides"):
            self.main.manual_flat_overrides = {}
        if not hasattr(self, "manual_dark_overrides"):
            self.main.manual_dark_overrides = {}

        # Tree widget for light frames
        self.main.light_tree = QTreeWidget()
        self.main.light_tree.setColumnCount(5)  # Added columns for Master Dark and Flat
        self.main.light_tree.setHeaderLabels(["Filter & Exposure", "Metadata", "Master Dark", "Master Flat", "Corrections"])
        self.main.light_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        layout.addWidget(QLabel("Light Frames"))
        layout.addWidget(self.main.light_tree)

        # Buttons for adding files and directories
        btn_layout = QHBoxLayout()
        self.main.add_light_files_btn = QPushButton("Add Light Files")
        self.main.add_light_files_btn.clicked.connect(self.add_light_files)
        self.main.add_light_dir_btn = QPushButton("Add Light Directory")
        self.main.add_light_dir_btn.clicked.connect(self.add_light_directory)
        btn_layout.addWidget(self.main.add_light_files_btn)
        btn_layout.addWidget(self.main.add_light_dir_btn)
        layout.addLayout(btn_layout)
        # under your existing buttons:
        opts_row = QHBoxLayout()

        self.main.light_recurse_cb = QCheckBox("Recurse subfolders")
        self.main._bind_shared_setting_checkbox("stacking/recurse_dirs", self.main.light_recurse_cb, default=True)

        self.main.light_auto_session_cb = QCheckBox("Auto-detect session")
        self.main._bind_shared_setting_checkbox("stacking/auto_session", self.main.light_auto_session_cb, default=True)

        # keep this one — it's independent of the shared pair
        self.main.auto_register_after_calibration_cb = QCheckBox("Auto-register & integrate after calibration")
        self.main.auto_register_after_calibration_cb.setToolTip(
            "When checked, once calibration finishes the app will switch to Image Registration and run "
            "‘Register and Integrate Images’ automatically."
        )
        self.main.auto_register_after_calibration_cb.setChecked(
            self.main.settings.value("stacking/auto_register_after_cal", False, type=bool)
        )
        self.main.auto_register_after_calibration_cb.toggled.connect(
            lambda v: self.main.settings.setValue("stacking/auto_register_after_cal", bool(v))
        )

        opts_row.addWidget(self.main.light_recurse_cb)
        opts_row.addWidget(self.main.light_auto_session_cb)
        opts_row.addWidget(self.main.auto_register_after_calibration_cb)
        layout.addLayout(opts_row)       
        session_hint_label = QLabel("Right Click to Assign Session Keys if desired")
        session_hint_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px; margin-left: 4px;")
        layout.addWidget(session_hint_label)

        clear_selection_btn = QPushButton("Remove Selected")
        clear_selection_btn.clicked.connect(lambda: self.main.clear_tree_selection_light(self.main.light_tree))
        layout.addWidget(clear_selection_btn)

        # Cosmetic Correction & Pedestal Controls
        correction_layout = QHBoxLayout()

        self.main.cosmetic_checkbox = QCheckBox("Enable Cosmetic Correction")
        # default = True, but keep it sticky via QSettings
        self.main.cosmetic_checkbox.setChecked(
            self.main.settings.value("stacking/cosmetic_enabled", True, type=bool)
        )
        # 🔧 NEW: persist initial value immediately so background code sees it
        self.main.settings.setValue("stacking/cosmetic_enabled", bool(self.main.cosmetic_checkbox.isChecked()))        
        self.main.cosmetic_checkbox.toggled.connect(
            lambda v: self.main.settings.setValue("stacking/cosmetic_enabled", bool(v))
        )

        self.main.pedestal_checkbox = QCheckBox("Apply Pedestal")
        self.main.pedestal_checkbox.setChecked(
            self.main.settings.value("stacking/pedestal_enabled", False, type=bool)
        )
        self.main.pedestal_checkbox.toggled.connect(
            lambda v: self.main.settings.setValue("stacking/pedestal_enabled", bool(v))
        )

        self.main.bias_checkbox = QCheckBox("Apply Bias Subtraction (For CCD Users)")
        self.main.bias_checkbox.setChecked(
            self.main.settings.value("stacking/bias_enabled", False, type=bool)
        )
        self.main.bias_checkbox.toggled.connect(
            lambda v: self.main.settings.setValue("stacking/bias_enabled", bool(v))
        )

        correction_layout.addWidget(self.main.cosmetic_checkbox)
        correction_layout.addWidget(self.main.pedestal_checkbox)
        correction_layout.addWidget(self.main.bias_checkbox)

        # Pedestal Value (0-1000, converted to 0-1)
        pedestal_layout = QHBoxLayout()
        self.main.pedestal_label = QLabel("Pedestal (0-1000):")
        self.main.pedestal_spinbox = QSpinBox()
        self.main.pedestal_spinbox.setRange(0, 1000)
        self.main.pedestal_spinbox.setValue(self.main.settings.value("stacking/pedestal_value", 50, type=int))
        self.main.pedestal_spinbox.valueChanged.connect(
            lambda v: self.main.settings.setValue("stacking/pedestal_value", int(v))
        )

        pedestal_layout.addWidget(self.main.pedestal_label)
        pedestal_layout.addWidget(self.main.pedestal_spinbox)
        pedestal_layout.addStretch(1)
        layout.addLayout(pedestal_layout)

        # 👇 tie enabled state to the checkbox (initial + live updates)
        def _sync_pedestal_enabled(checked: bool):
            self.main.pedestal_label.setEnabled(checked)
            self.main.pedestal_spinbox.setEnabled(checked)

        _sync_pedestal_enabled(self.main.pedestal_checkbox.isChecked())
        self.main.pedestal_checkbox.toggled.connect(_sync_pedestal_enabled)

        # Tooltips (unchanged)
        self.main.bias_checkbox.setToolTip(
            "CMOS users: Bias Subtraction is not needed.\n"
            "Modern CMOS cameras use Correlated Double Sampling (CDS),\n"
            "meaning bias is already subtracted at the sensor level."
        )

        # Connect to your existing correction updater
        self.main.cosmetic_checkbox.stateChanged.connect(self.main.update_light_corrections)
        self.main.pedestal_checkbox.stateChanged.connect(self.main.update_light_corrections)
        self.main.bias_checkbox.stateChanged.connect(self.main.update_light_corrections)

        # Add checkboxes to layout
        correction_layout.addWidget(self.main.cosmetic_checkbox)
        correction_layout.addWidget(self.main.pedestal_checkbox)
        correction_layout.addWidget(self.main.bias_checkbox)

        layout.addLayout(correction_layout)        

        # --- RIGHT SIDE CONTROLS: Override Dark & Flat ---
        override_layout = QHBoxLayout()

        self.main.override_dark_btn = QPushButton("Override Dark Frame")
        self.main.override_dark_btn.clicked.connect(self.main.override_selected_master_dark)
        override_layout.addWidget(self.main.override_dark_btn)

        self.main.override_flat_btn = QPushButton("Override Flat Frame")
        self.main.override_flat_btn.clicked.connect(self.main.override_selected_master_flat)
        override_layout.addWidget(self.main.override_flat_btn)

        layout.addLayout(override_layout)

        # Calibrate Lights Button
        self.main.calibrate_lights_btn = QPushButton("🚀 Calibrate Light Frames 🚀")
        self.main.calibrate_lights_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF4500;
                color: white;
                font-size: 16px;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF6347;
            }
        """)
        self.main.calibrate_lights_btn.clicked.connect(self.calibrate_lights)
        layout.addWidget(self.main.calibrate_lights_btn)

        # Enable Context Menu
        self.main.light_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.main.light_tree.customContextMenuRequested.connect(self.light_tree_context_menu)

        return tab




    def add_light_files(self):
        auto = self.main.settings.value("stacking/auto_session", True, type=bool)
        if auto:
            self.main.add_files(self.main.light_tree, "Select Light Files", "LIGHT")
            self.main.assign_best_master_files()
        else:
            self.main.prompt_session_before_adding("LIGHT", directory_mode=False)

    

    def add_light_directory(self):
        auto = self.main.settings.value("stacking/auto_session", True, type=bool)
        if auto:
            self.main.add_directory(self.main.light_tree, "Select Light Directory", "LIGHT")
            self.main.assign_best_master_files()
        else:
            self.main.prompt_session_before_adding("LIGHT", directory_mode=True)



    def calibrate_lights(self):
        """Performs calibration on selected light frames using Master Darks and Flats, considering overrides."""
        # Make sure columns 2/3 have something where possible
        self.main.assign_best_master_files(fill_only=True)

        if not self.main.stacking_directory:
            QMessageBox.warning(self.main, "Error", "Please set the stacking directory first.")
            return

        calibrated_dir = os.path.join(self.main.stacking_directory, "Calibrated")
        os.makedirs(calibrated_dir, exist_ok=True)

        leaf_paths = self.main._collect_leaf_paths_from_tree()
        total_files = len(leaf_paths)
        processed_files = 0

        # (optional) GC stale entries so future counts stay sane
        for key in list(self.main.light_files.keys()):
            self.main.light_files[key] = [p for p in self.main.light_files[key] if p in leaf_paths]
            if not self.main.light_files[key]:
                del self.main.light_files[key]
        processed_files = 0

        # ---------- LOAD MASTER BIAS ONCE (optional) ----------
        master_bias = None
        bias_path = self.main.master_files.get("Bias")
        if bias_path:
            try:
                master_bias, _, _, bias_is_mono = load_image(bias_path)
                if master_bias is not None:
                    # ensure H,W (mono) or CHW (color) to match your subtract
                    if (not bias_is_mono) and master_bias.ndim == 3 and master_bias.shape[-1] == 3:
                        master_bias = master_bias.transpose(2,0,1)  # HWC -> CHW
                    self.main.update_status(f"Using Master Bias: {os.path.basename(bias_path)}")
            except Exception as e:
                self.main.update_status(f"⚠️ Could not load Master Bias: {e}")
                master_bias = None

        for i in range(self.main.light_tree.topLevelItemCount()):
            filter_item = self.main.light_tree.topLevelItem(i)
            filter_name = filter_item.text(0)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)

                # Get default corrections
                apply_cosmetic, apply_pedestal = self.main._resolve_corrections_for_exposure(exposure_item)
                pedestal_value = (self.main.pedestal_spinbox.value() / 65535.0) if apply_pedestal else 0.0

                # (optional) keep the Corrections column in sync for visibility/debugging
                try:
                    exposure_item.setText(
                        4,
                        f"Cosmetic: {'True' if apply_cosmetic else 'False'}, "
                        f"Pedestal: {'True' if apply_pedestal else 'False'}"
                    )
                except Exception:
                    pass

                for k in range(exposure_item.childCount()):
                    leaf = exposure_item.child(k)
                    filename = leaf.text(0)
                    meta = leaf.text(1)

                    # Get session from metadata
                    session_name = "Default"
                    m = re.search(r"Session: ([^|]+)", meta)
                    if m:
                        session_name = m.group(1).strip()

                    # Look up the light file from session-specific group
                    composite_key = (f"{filter_name} - {exposure_text}", session_name)
                    light_file_list = self.main.light_files.get(composite_key, [])
                    light_file = next((f for f in light_file_list if os.path.basename(f) == filename), None)
                    if not light_file:
                        continue
                    if light_file not in leaf_paths:
                        continue
                    # Determine size from header
                    header, _ = get_valid_header(light_file)
                    width = int(header.get("NAXIS1", 0))
                    height = int(header.get("NAXIS2", 0))
                    image_size = f"{width}x{height}"

                    # ---------- RESOLVE MASTER DARK ----------
                    # 0) Per-leaf override takes precedence
                    master_dark_path = self.main._leaf_assigned_dark_path(leaf)

                    if master_dark_path is None:
                        # 1) Your existing manual map (full + short)
                        manual_dark_key_full  = f"{filter_name} - {exposure_text}"
                        manual_dark_key_short = exposure_text
                        master_dark_path = (
                            self.main.manual_dark_overrides.get(manual_dark_key_full)
                            or self.main.manual_dark_overrides.get(manual_dark_key_short)
                        )

                    if master_dark_path is None:
                        # 2) If the leaf shows a basename, map back to a stored path
                        name_in_leaf = (leaf.text(2) or "").strip()
                        if name_in_leaf:
                            for _, path in self.main.master_files.items():
                                if os.path.basename(path) == name_in_leaf:
                                    master_dark_path = path
                                    break

                    if master_dark_path is None:
                        # 3) Last resort: auto-pick by size+exposure
                        mm = re.match(r"([\d.]+)s", exposure_text or "")
                        exp_time = float(mm.group(1)) if mm else 0.0
                        master_dark_path = self.main._auto_pick_master_dark(image_size, exp_time)

                    print(f"master_dark_path is {master_dark_path}")

                    # ---------- RESOLVE MASTER FLAT ----------
                    # 0) Per-leaf override takes precedence
                    master_flat_path = self.main._leaf_assigned_flat_path(leaf)

                    if master_flat_path is None:
                        # 1) Your existing manual map (scoped to filter+exposure)
                        manual_flat_key_full = f"{filter_name} - {exposure_text}"
                        master_flat_path = self.main.manual_flat_overrides.get(manual_flat_key_full)

                    if master_flat_path is None:
                        # 2) If the leaf shows a basename, map back to a stored path
                        name_in_leaf = (leaf.text(3) or "").strip()
                        if name_in_leaf:
                            for _, path in self.main.master_files.items():
                                if os.path.basename(path) == name_in_leaf:
                                    master_flat_path = path
                                    break

                    if master_flat_path is None:
                        # 3) Prefer session-matched flat, else size+filter fallback
                        master_flat_path = self.main._auto_pick_master_flat(filter_name, image_size, session_name)

                    print(f"master_flat_path is {master_flat_path}")


                    # ---------- LOAD LIGHT ----------
                    light_data, hdr, bit_depth, is_mono = load_image(light_file)
                    if light_data is None or hdr is None:
                        self.main.update_status(f"❌ ERROR: Failed to load {os.path.basename(light_file)}")
                        continue

                    # Work in CHW for color; leave mono as H,W
                    if not is_mono and light_data.ndim == 3 and light_data.shape[-1] == 3:
                        light_data = light_data.transpose(2, 0, 1)  # HWC -> CHW

                    # ---------- APPLY BIAS (optional) ----------
                    if master_bias is not None:
                        if is_mono:
                            light_data -= master_bias
                        else:
                            light_data -= master_bias[np.newaxis, :, :]
                        self.main.update_status("Bias Subtracted")
                        QApplication.processEvents()

                    # ---------- APPLY DARK (if resolved) ----------
                    if master_dark_path:
                        dark_data, _, _, dark_is_mono = load_image(master_dark_path)
                        if dark_data is not None:
                            if not dark_is_mono and dark_data.ndim == 3 and dark_data.shape[-1] == 3:
                                dark_data = dark_data.transpose(2, 0, 1)  # HWC -> CHW
                            # shape-safe subtract with pedestal (expects stack,F dimension)
                            if light_data.ndim == 2:  # mono
                                tmp = subtract_dark_with_pedestal(light_data[np.newaxis, :, :], dark_data, pedestal_value)
                                light_data = tmp[0]
                            else:                      # CHW
                                tmp = subtract_dark_with_pedestal(light_data[np.newaxis, :, :], dark_data, pedestal_value)
                                light_data = tmp[0]
                            self.main.update_status(f"Dark Subtracted: {os.path.basename(master_dark_path)}")
                            QApplication.processEvents()

                    # ---------- APPLY FLAT (if resolved) ----------
                    if master_flat_path:
                        flat_data, _, _, flat_is_mono = load_image(master_flat_path)
                        if flat_data is not None:
                            if not flat_is_mono and flat_data.ndim == 3 and flat_data.shape[-1] == 3:
                                flat_data = flat_data.transpose(2, 0, 1)  # HWC -> CHW

                            # Ensure float32 and normalize flat to mean 1.0 to preserve flux scaling
                            flat_data = flat_data.astype(np.float32, copy=False)
                            # Avoid zero/NaN in flats
                            flat_data = np.nan_to_num(flat_data, nan=1.0, posinf=1.0, neginf=1.0)
                            if flat_data.ndim == 2:
                                denom = np.median(flat_data[flat_data > 0])
                                if not np.isfinite(denom) or denom <= 0: denom = 1.0
                                flat_data /= denom
                            else:  # CHW: normalize per-channel
                                for c in range(flat_data.shape[0]):
                                    band = flat_data[c]
                                    denom = np.median(band[band > 0])
                                    if not np.isfinite(denom) or denom <= 0: denom = 1.0
                                    flat_data[c] = band / denom

                            # Safety: forbid exact zeros in denominator
                            flat_data[flat_data == 0] = 1.0

                            light_data = apply_flat_division_numba(light_data, flat_data)
                            self.main.update_status(f"Flat Applied: {os.path.basename(master_flat_path)}")
                            QApplication.processEvents()

                    # ---------- COSMETIC (optional) ----------
                    if apply_cosmetic:
                        # Pull configured values (fallbacks preserve current behavior)
                        hot_sigma      = self.main.settings.value("stacking/cosmetic/hot_sigma", 5.0, type=float)
                        cold_sigma     = self.main.settings.value("stacking/cosmetic/cold_sigma", 5.0, type=float)
                        star_mean_ratio= self.main.settings.value("stacking/cosmetic/star_mean_ratio", 0.22, type=float)
                        star_max_ratio = self.main.settings.value("stacking/cosmetic/star_max_ratio", 0.55, type=float)
                        sat_quantile   = self.main.settings.value("stacking/cosmetic/sat_quantile", 0.9995, type=float)

                        # --- Decide Bayer vs debayered from DATA, not header ---
                        # After your HWC->CHW transpose, debayered color lights will be (3,H,W).
                        array_is_color = (
                            light_data.ndim == 3 and (
                                light_data.shape[0] == 3 or light_data.shape[-1] == 3
                            )
                        )
                        if array_is_color:
                            is_mono = False
                        bayerpat = hdr.get("BAYERPAT")
                        use_bayer_cosmetic = (not array_is_color) and bool(bayerpat)

                        if use_bayer_cosmetic:
                            pattern = str(bayerpat).strip().upper()
                            if pattern not in ("RGGB","BGGR","GRBG","GBRG"):
                                pattern = "RGGB"

                            # light_data is guaranteed 2D here
                            light_data = bulk_cosmetic_correction_bayer(
                                light_data,
                                hot_sigma=hot_sigma,
                                cold_sigma=cold_sigma
                            )
                            self.main.update_status(f"Cosmetic Correction Applied for Bayer Pattern ({pattern})")
                        else:
                            light_data = bulk_cosmetic_correction_numba(
                                light_data,
                                hot_sigma=hot_sigma,
                                cold_sigma=cold_sigma
                            )
                            self.main.update_status("Cosmetic Correction Applied (debayered/mono)")

                        QApplication.processEvents()

                    # Back to HWC for saving if color
                    # Back to HWC for saving if color
                    if not is_mono and light_data.ndim == 3 and light_data.shape[0] == 3:
                        light_data = light_data.transpose(1, 2, 0)  # CHW -> HWC

                    # Sanitize numerics but DO NOT rescale
                    light_data = np.nan_to_num(light_data.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

                    min_val = float(np.min(light_data))
                    max_val = float(np.max(light_data))
                    self.main.update_status(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    print(f"Before saving: min = {min_val:.4f}, max = {max_val:.4f}")
                    QApplication.processEvents()

                    # Annotate header
                    try:
                        hdr['HISTORY'] = 'Calibrated: bias/dark sub, flat division'
                        hdr['CALMIN']  = (min_val, 'Min pixel before save (float)')
                        hdr['CALMAX']  = (max_val, 'Max pixel before save (float)')
                    except Exception:
                        pass

                    base = os.path.basename(light_file)
                    root, ext = os.path.splitext(base)

                    # handle double extensions like .fits.gz
                    if root.lower().endswith(".fits") or root.lower().endswith(".fit"):
                        root2, ext2 = os.path.splitext(root)
                        root, ext = root2, ext2 + ext

                    calibrated_filename = os.path.join(calibrated_dir, f"{root}_c.fit")

                    # Force float32 FITS regardless of camera bit depth
                    save_image(
                        img_array=light_data,
                        filename=calibrated_filename,
                        original_format="fit",
                        bit_depth="32-bit floating point",          # << force float32 to avoid clipping negatives
                        original_header=hdr,
                        is_mono=is_mono
                    )

                    processed_files += 1
                    self.main.update_status(f"Saved: {os.path.basename(calibrated_filename)} ({processed_files}/{total_files})")
                    QApplication.processEvents()

        self.main.update_status("✅ Calibration Complete!")
        QApplication.processEvents()
        self.main.populate_calibrated_lights()


        # ── NEW: optionally roll straight into registration+integration ──
        try:
            if self.main.settings.value("stacking/auto_register_after_cal", False, type=bool):
                # Switch UI to the Image Registration tab, if we can find it
                if hasattr(self, "tabs") and self.main.tabs is not None:
                    try:
                        for i in range(self.main.tabs.count()):
                            # match by tab label if available
                            if self.main.tabs.tabText(i).lower().startswith("image registration"):
                                self.main.tabs.setCurrentIndex(i)
                                break
                    except Exception:
                        pass  # harmless if tab text lookup fails

                self.main.update_status("⚙️ Auto: starting registration & integration…")
                QApplication.processEvents()

                # Prefer button .click() (preserves any guard/flags)
                if hasattr(self, "register_images_btn") and self.main.register_images_btn is not None:
                    # Guard against re-entrancy if a run is already in progress
                    if not getattr(self, "_registration_busy", False):
                        self.main.register_images_btn.click()
                    else:
                        self.main.update_status("ℹ️ Registration already in progress; auto-run skipped.")
                # Fallback: call the method directly
                elif hasattr(self, "register_images"):
                    if not getattr(self, "_registration_busy", False):
                        self.main.register_images()
                    else:
                        self.main.update_status("ℹ️ Registration already in progress; auto-run skipped.")
        except Exception as e:
            self.main.update_status(f"⚠️ Auto register/integrate failed: {e}")



    def light_tree_context_menu(self, pos):
        item = self.main.light_tree.itemAt(pos)
        if not item:
            return

        menu = QMenu(self.main.light_tree)
        override_dark_action = menu.addAction("Override Dark Frame")
        override_flat_action = menu.addAction("Override Flat Frame")
        set_session_action = menu.addAction("Set Session Tag...")

        action = menu.exec(self.main.light_tree.viewport().mapToGlobal(pos))

        if action == override_dark_action:
            self.main.override_selected_master_dark()
        elif action == override_flat_action:
            self.main.override_selected_master_flat()
        elif action == set_session_action:
            self.main.prompt_set_session(item, "LIGHT")



    def _refresh_light_tree_summaries(self):
        """
        Fill the Metadata column (col=1) with:
        • for each exposure node:  "<N> files · <HHh MMm SSs>"
        • for each filter node:    "<N_total> files · <HHh MMm SSs> (sum of all children)"
        """
        tree = getattr(self, "light_tree", None)
        if tree is None:
            return

        total_filters = tree.topLevelItemCount()
        for i in range(total_filters):
            filt_item = tree.topLevelItem(i)
            if filt_item is None:
                continue

            filt_total_files = 0
            filt_total_secs  = 0.0

            # children are exposure-size groups like "20s (1080x1920)"
            for j in range(filt_item.childCount()):
                exp_item = filt_item.child(j)
                if exp_item is None:
                    continue

                exp_seconds = self.main._exposure_from_label(exp_item.text(0)) or 0.0
                n_files     = exp_item.childCount()
                n_secs      = exp_seconds * n_files

                # set exposure-row metadata
                if n_files == 1:
                    exp_item.setText(1, f"1 file · {self.main._fmt_hms(n_secs)}")
                else:
                    exp_item.setText(1, f"{n_files} files · {self.main._fmt_hms(n_secs)}")

                filt_total_files += n_files
                filt_total_secs  += n_secs

            # set filter-row metadata (sum of children)
            if filt_total_files == 1:
                filt_item.setText(1, f"1 file · {self.main._fmt_hms(filt_total_secs)}")
            else:
                filt_item.setText(1, f"{filt_total_files} files · {self.main._fmt_hms(filt_total_secs)}")



