# Tab module - imports from parent packages
from __future__ import annotations
import os
import sys
import platform
import math
import re
import gc
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
from pro.stacking.dialogs import _Progress, _count_tiles
from pro.stacking.functions import (
    _torch_ok, _gpu_algo_supported, _torch_reduce_tile,
    _tile_grid, _read_tile_stack, load_fits_tile, _free_torch_memory,
)


class FlatTab(QObject):
    """Extracted Flat tab functionality."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def create_flat_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)  # Main layout to organize sections

        # --- FLAT FRAMES TREEBOX (TOP) ---
        flats_layout = QHBoxLayout()  # Left = Flat Tree, Right = Controls

        # Left Side - Flat Frames
        flat_frames_layout = QVBoxLayout()
        flat_frames_layout.addWidget(QLabel("Flat Frames"))

        self.main.flat_tree = QTreeWidget()
        self.main.flat_tree.setColumnCount(3)  # Added 3rd column for Master Dark Used
        self.main.flat_tree.setHeaderLabels(["Filter & Exposure", "Metadata", "Master Dark Used"])
        self.main.flat_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.main.flat_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.main.flat_tree.customContextMenuRequested.connect(self.flat_tree_context_menu)
        flat_frames_layout.addWidget(self.main.flat_tree)

        # Buttons to Add Flat Files & Directories
        btn_layout = QHBoxLayout()
        self.main.add_flat_files_btn = QPushButton("Add Flat Files")
        self.main.add_flat_files_btn.clicked.connect(self.add_flat_files)
        self.main.add_flat_dir_btn = QPushButton("Add Flat Directory")
        self.main.add_flat_dir_btn.clicked.connect(self.add_flat_directory)
        btn_layout.addWidget(self.main.add_flat_files_btn)
        btn_layout.addWidget(self.main.add_flat_dir_btn)
        flat_frames_layout.addLayout(btn_layout)
        # under your existing buttons:
        opts_row = QHBoxLayout()

        self.main.flat_recurse_cb = QCheckBox("Recurse subfolders")
        self.main._bind_shared_setting_checkbox("stacking/recurse_dirs", self.main.flat_recurse_cb, default=True)

        self.main.flat_auto_session_cb = QCheckBox("Auto-detect session")
        self.main._bind_shared_setting_checkbox("stacking/auto_session", self.main.flat_auto_session_cb, default=True)

        opts_row.addWidget(self.main.flat_recurse_cb)
        opts_row.addWidget(self.main.flat_auto_session_cb)
        flat_frames_layout.addLayout(opts_row)      
        # 🔧 Session Tag Hint
        session_hint_label = QLabel("Right Click to Assign Session Keys if desired")
        session_hint_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px; margin-left: 4px;")
        flat_frames_layout.addWidget(session_hint_label)

        # Add "Clear Selection" button for Flat Frames
        self.main.clear_flat_selection_btn = QPushButton("Clear Selection")
        self.main.clear_flat_selection_btn.clicked.connect(lambda: self.main.clear_tree_selection_flat(self.main.flat_tree, self.main.flat_files))
        flat_frames_layout.addWidget(self.main.clear_flat_selection_btn)

        flats_layout.addLayout(flat_frames_layout, 2)  # Left side takes more space

        # --- RIGHT SIDE: Exposure Tolerance & Master Dark Selection ---
        right_controls_layout = QVBoxLayout()

        # Exposure Tolerance
        exposure_tolerance_layout = QHBoxLayout()
        exposure_tolerance_label = QLabel("Exposure Tolerance (seconds):")
        self.main.flat_exposure_tolerance_spinbox = QSpinBox()
        self.main.flat_exposure_tolerance_spinbox.setRange(0, 30)  # Allow ±0 to 30 seconds
        self.main.flat_exposure_tolerance_spinbox.setValue(5)  # Default: ±5 sec
        exposure_tolerance_layout.addWidget(exposure_tolerance_label)
        exposure_tolerance_layout.addWidget(self.main.flat_exposure_tolerance_spinbox)
        right_controls_layout.addLayout(exposure_tolerance_layout)
        self.main.flat_exposure_tolerance_spinbox.valueChanged.connect(self.rebuild_flat_tree)


        # Auto-Select Master Dark
        self.main.auto_select_dark_checkbox = QCheckBox("Auto-Select Closest Master Dark")
        self.main.auto_select_dark_checkbox.setChecked(True)  # Default enabled
        right_controls_layout.addWidget(self.main.auto_select_dark_checkbox)

        # Manual Override: Select a Master Dark
        self.main.override_dark_combo = QComboBox()
        self.main.override_dark_combo.addItem("None (Use Auto-Select)")
        self.main.override_dark_combo.currentIndexChanged.connect(self.main.override_selected_master_dark_for_flats)
        right_controls_layout.addWidget(QLabel("Override Master Dark Selection"))
        right_controls_layout.addWidget(self.main.override_dark_combo)

        self.main.create_master_flat_btn = QPushButton("Turn Those Flats Into Master Flats")
        self.main.create_master_flat_btn.clicked.connect(self.create_master_flat)

        # Apply a bold font, padding, and a glowing effect
        self.main.create_master_flat_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;  /* Dark gray */
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 5px;
                border: 2px solid yellow;  /* Subtle yellow border */
            }
            QPushButton:hover {
                border: 2px solid #FFD700;  /* Brighter yellow on hover */
            }
            QPushButton:pressed {
                background-color: #222;  /* Darker gray on press */
                border: 2px solid #FFA500;  /* Orange border when pressed */
            }
        """)


        right_controls_layout.addWidget(self.main.create_master_flat_btn)

        flats_layout.addLayout(right_controls_layout, 1)  # Right side takes less space

        main_layout.addLayout(flats_layout)

        # --- MASTER FLATS TREEBOX (BOTTOM) ---
        main_layout.addWidget(QLabel("Master Flats"))
        self.main.master_flat_tree = QTreeWidget()
        self.main.master_flat_tree.setColumnCount(2)
        self.main.master_flat_tree.setHeaderLabels(["Filter", "Master File"])
        self.main.master_flat_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        main_layout.addWidget(self.main.master_flat_tree)

        # Master Flat Selection Button
        self.main.master_flat_btn = QPushButton("Load Master Flat")
        self.main.master_flat_btn.clicked.connect(self.load_master_flat)
        main_layout.addWidget(self.main.master_flat_btn)

        self.main.clear_master_flat_selection_btn = QPushButton("Clear Selection")
        self.main.clear_master_flat_selection_btn.clicked.connect(
            lambda: (self.main.clear_tree_selection(self.main.master_flat_tree, self.main.master_files),
                    self.main.save_master_paths_to_settings())
        )
        main_layout.addWidget(self.main.clear_master_flat_selection_btn)
        self.main.override_dark_combo.currentIndexChanged[int].connect(self.main.override_selected_master_dark_for_flats)
        self.main.update_override_dark_combo()
        self.rebuild_flat_tree()
    
        return tab


    def flat_tree_context_menu(self, position):
        item = self.main.flat_tree.itemAt(position)
        if item:
            menu = QMenu()
            set_session_action = menu.addAction("Set Session Tag")
            action = menu.exec(self.main.flat_tree.viewport().mapToGlobal(position))
            if action == set_session_action:
                self.main.prompt_set_session(item, "flat")


    def add_flat_files(self):
        self.main.prompt_session_before_adding("FLAT")



    def add_flat_directory(self):
        auto = self.main.settings.value("stacking/auto_session", True, type=bool)
        if auto:
            # No prompt — auto session tagging happens inside add_directory()
            self.main.add_directory(self.main.flat_tree, "Select Flat Directory", "FLAT")
            self.main.assign_best_master_dark()
            self.rebuild_flat_tree()
        else:
            # Manual path (will prompt once for a session name)
            self.main.prompt_session_before_adding("FLAT", directory_mode=True)


    

    def load_master_flat(self):
        last_dir = self.main.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(self.main, "Select Master Flat", last_dir, "FITS Files (*.fits *.fit)")

        if files:
            self.main.settings.setValue("last_opened_folder", os.path.dirname(files[0]))
            self.main.add_master_files(self.main.master_flat_tree, "FLAT", files)
            self.main.save_master_paths_to_settings() 



    def create_master_flat(self):
        """Creates master flats using per-frame dark subtraction before stacking (GPU-accelerated if available),
        with adaptive reducers and fast per-frame normalization."""
        self.main.update_status("Starting Master Flat Creation...")
        if not self.main.stacking_directory:
            QMessageBox.warning(self.main, "Error", "Please set the stacking directory first using the wrench button.")
            return

        # Keep both paths available; we'll override algo selection per group.
        ui_algo = getattr(self, "calib_rejection_algorithm", "Windsorized Sigma Clipping")
        if ui_algo == "Weighted Windsorized Sigma Clipping":
            ui_algo = "Windsorized Sigma Clipping"

        exposure_tolerance = self.main.flat_exposure_tolerance_spinbox.value()
        flat_files_by_group = {}  # (Exposure, Size, Filter, Session) -> list

        # --- group flats exactly as before ---
        for (filter_exposure, session), file_list in self.main.flat_files.items():
            try:
                filter_name, exposure_size = filter_exposure.split(" - ")
                exposure_time_str, image_size = exposure_size.split(" (")
                image_size = image_size.rstrip(")")
            except ValueError:
                self.main.update_status(f"⚠️ ERROR: Could not parse {filter_exposure}")
                continue

            match = re.match(r"([\d.]+)s?", exposure_time_str)
            exposure_time = float(match.group(1)) if match else -10.0

            matched_group = None
            for key in flat_files_by_group:
                existing_exposure, existing_size, existing_filter, existing_session = key
                if (
                    abs(existing_exposure - exposure_time) <= exposure_tolerance
                    and existing_size == image_size
                    and existing_filter == filter_name
                    and existing_session == session
                ):
                    matched_group = key
                    break

            if matched_group is None:
                matched_group = (exposure_time, image_size, filter_name, session)
                flat_files_by_group[matched_group] = []
            flat_files_by_group[matched_group].extend(file_list)

        # Discovery summary
        try:
            n_groups = sum(1 for k, v in flat_files_by_group.items() if len(v) >= 2)
            total_files = sum(len(v) for v in flat_files_by_group.values())
            self.main.update_status(f"🔎 Discovered {len(flat_files_by_group)} flat groups ({n_groups} eligible to stack) — {total_files} files total.")
        except Exception:
            pass
        QApplication.processEvents()

        # Output folder
        master_dir = os.path.join(self.main.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # Pre-count tiles
        total_tiles = 0
        group_shapes = {}
        for (exposure_time, image_size, filter_name, session), file_list in flat_files_by_group.items():
            if len(file_list) < 2:
                continue
            ref_data, _, _, _ = load_image(file_list[0])
            if ref_data is None:
                continue
            H, W = ref_data.shape[:2]
            C = 1 if ref_data.ndim == 2 else 3
            group_shapes[(exposure_time, image_size, filter_name, session)] = (H, W, C)
            total_tiles += _count_tiles(H, W, self.main.chunk_height, self.main.chunk_width)

        if total_tiles == 0:
            self.main.update_status("⚠️ No eligible flat groups found to stack.")
            return

        self.main.update_status(f"🧭 Total tiles to process: {total_tiles} (chunk size {self.main.chunk_height}×{self.main.chunk_width})")
        QApplication.processEvents()

        # ------- helpers (local to this function) --------------------------------
        def _select_reducer(kind: str, N: int):
            """
            kind: 'flat' or 'dark'; return (algo_name, params_dict, cpu_label)
            algo_name is a GPU-string if GPU is used; CPU gets cpu_label switch.
            """
            if kind == "flat":
                # <16: robust median; 17–200: trimmed mean 5%; >200: trimmed mean 2% (nearly mean)
                if N < 16:
                    return ("Simple Median (No Rejection)", {}, "median")
                elif N <= 200:
                    return ("Trimmed Mean", {"trim_fraction": 0.05}, "trimmed")
                else:
                    return ("Trimmed Mean", {"trim_fraction": 0.02}, "trimmed")
            else:  # darks
                # <16: Kappa-Sigma 1-iter; 17–200: simple median; >200: trimmed mean 5% to reduce noise
                if N < 16:
                    return ("Kappa-Sigma Clipping", {"kappa": 3.0, "iterations": 1}, "kappa1")
                elif N <= 200:
                    return ("Simple Median (No Rejection)", {}, "median")
                else:
                    return ("Trimmed Mean", {"trim_fraction": 0.05}, "trimmed")

        def _cpu_tile_median(ts4: np.ndarray) -> np.ndarray:
            # ts4: (F, th, tw, C)
            return np.median(ts4, axis=0).astype(np.float32, copy=False)

        def _cpu_tile_trimmed_mean(ts4: np.ndarray, frac: float) -> np.ndarray:
            if frac <= 0.0:
                return ts4.mean(axis=0, dtype=np.float32)
            F = ts4.shape[0]
            k = int(max(1, round(frac * F)))
            if 2 * k >= F:  # if too aggressive, fall back to median
                return _cpu_tile_median(ts4)
            # sort along frame axis and average middle slice
            s = np.sort(ts4, axis=0)
            core = s[k:F - k]
            return core.mean(axis=0, dtype=np.float32)

        def _cpu_tile_kappa_sigma_1iter(ts4: np.ndarray, kappa: float = 3.0) -> np.ndarray:
            med = np.median(ts4, axis=0)
            std = ts4.std(axis=0, dtype=np.float32)
            lo = med - kappa * std
            hi = med + kappa * std
            mask = (ts4 >= lo) & (ts4 <= hi)
            # avoid division by zero
            num = (ts4 * mask).sum(axis=0, dtype=np.float32)
            cnt = mask.sum(axis=0).astype(np.float32)
            out = np.where(cnt > 0, num / np.maximum(cnt, 1.0), med)
            return out.astype(np.float32, copy=False)

        def _estimate_flat_scales(file_list: list[str], H: int, W: int, C: int, dark_data: np.ndarray | None):
            """
            Read one central patch (min(512, H/W)) from each frame, subtract dark (if present),
            compute per-frame median, and normalize scales to overall median.
            """
            # central patch
            th = min(512, H); tw = min(512, W)
            y0 = (H - th) // 2; y1 = y0 + th
            x0 = (W - tw) // 2; x1 = x0 + tw

            N = len(file_list)
            meds = np.empty((N,), dtype=np.float64)

            # small parallel read
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as exe:
                fut2i = {exe.submit(load_fits_tile, fp, y0, y1, x0, x1): i for i, fp in enumerate(file_list)}
                for fut in as_completed(fut2i):
                    i = fut2i[fut]
                    sub = fut.result()
                    if sub is None:
                        meds[i] = 1.0
                        continue
                    # to HWC f32
                    if sub.ndim == 2:
                        sub = sub[:, :, None]
                    elif sub.ndim == 3 and sub.shape[0] in (1, 3):
                        sub = sub.transpose(1, 2, 0)
                    sub = sub.astype(np.float32, copy=False)

                    if dark_data is not None:
                        dd = dark_data
                        if dd.ndim == 3 and dd.shape[0] in (1, 3):
                            dd = dd.transpose(1, 2, 0)
                        d_tile = dd[y0:y1, x0:x1].astype(np.float32, copy=False)
                        if d_tile.ndim == 2 and sub.shape[2] == 3:
                            d_tile = np.repeat(d_tile[..., None], 3, axis=2)
                        sub = sub - d_tile

                    meds[i] = np.median(sub, axis=(0, 1, 2))
            # normalize to global median
            gmed = np.median(meds) if np.all(np.isfinite(meds)) else 1.0
            gmed = 1.0 if gmed == 0.0 else gmed
            scales = meds / gmed
            # clamp to sane range
            scales = np.clip(scales, 1e-3, 1e3).astype(np.float32)
            return scales

        pd = _Progress(self.main, "Create Master Flats", total_tiles)
        self.main.update_status(f"Progress initialized: {total_tiles} tiles across groups.")
        QApplication.processEvents()
        try:
            for (exposure_time, image_size, filter_name, session), file_list in flat_files_by_group.items():
                if len(file_list) < 2:
                    self.main.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) [{filter_name}] [{session}] - Not enough frames to stack.")
                    continue
                if pd.cancelled:
                    self.main.update_status("⛔ Master Flat creation cancelled.")
                    break

                exp_label = f"{exposure_time}s" if exposure_time >= 0 else "Unknown"
                self.main.update_status(
                    f"🟢 Processing {len(file_list)} flats for {exp_label} ({image_size}) [{filter_name}] in session '{session}'…"
                )
                QApplication.processEvents()

                # Select matching master dark (optional)
                selected_master_dark = None
                override_val = None

                # --- resolve OVERRIDE from flat_dark_override robustly ---
                fdo = getattr(self, "flat_dark_override", {}) or {}
                if fdo:
                    exp_here = float(exposure_time)
                    tol = float(self.main.flat_exposure_tolerance_spinbox.value())
                    for k, v in fdo.items():
                        # keys are stored as "filter|size|exp_lo|exp_hi"
                        try:
                            f_name, sz, lo_s, hi_s = k.split("|")
                        except Exception:
                            continue

                        # filter must match
                        if f_name != filter_name:
                            continue

                        # size: treat "Unknown" as wildcard on either side
                        if sz != "Unknown" and image_size != "Unknown" and sz != image_size:
                            continue

                        # exposure: if we don't know (negative), accept any in range
                        if exp_here >= 0:
                            lo = float(lo_s); hi = float(hi_s)
                            if not (lo - 1e-6 <= exp_here <= hi + 1e-6):
                                continue

                        override_val = v
                        break

                if override_val == "__NO_DARK__":
                    self.main.update_status("ℹ️ This flat group: override = No Calibration.")
                elif isinstance(override_val, str):
                    if os.path.exists(override_val):
                        selected_master_dark = override_val
                        self.main.update_status(
                            f"🌓 This flat group: using OVERRIDE dark → {os.path.basename(selected_master_dark)}"
                        )
                    else:
                        self.main.update_status("⚠️ Override dark missing on disk; falling back to Auto/None.")
                        override_val = None  # fall through

                # --- AUTO-SELECT when no override ---
                if (override_val is None) and self.main.auto_select_dark_checkbox.isChecked():
                    best_diff = float("inf")
                    exp_here = float(exposure_time)

                    for key, path in (getattr(self, "master_files", {}) or {}).items():
                        # keep only MasterDark files
                        if "MasterDark" not in os.path.basename(path):
                            continue

                        dark_size = self.main.master_sizes.get(path, "Unknown")
                        m = re.match(r"([\d.]+)s", key or "")
                        dark_exp = float(m.group(1)) if m else float("inf")

                        # size: again, "Unknown" on either side is a wildcard
                        size_ok = (
                            image_size == "Unknown"
                            or dark_size == "Unknown"
                            or dark_size == image_size
                        )
                        if not size_ok:
                            continue

                        diff = abs(dark_exp - exp_here)
                        if diff < best_diff:
                            best_diff = diff
                            selected_master_dark = path

                    if selected_master_dark:
                        self.main.update_status(
                            f"🌓 This flat group: using AUTO dark → {os.path.basename(selected_master_dark)}"
                        )
                    else:
                        self.main.update_status(
                            "ℹ️ This flat group: no matching Master Dark (size) — proceeding without subtraction."
                        )

                elif (override_val is None) and (not self.main.auto_select_dark_checkbox.isChecked()):
                    # explicit: no auto and no override → no dark
                    self.main.update_status(
                        "ℹ️ This flat group: Auto-Select is OFF and no override set → No Calibration."
                    )

                # Load the chosen dark if any
                if selected_master_dark:
                    dark_data, _, _, _ = load_image(selected_master_dark)
                else:
                    dark_data = None

                # reference shape
                if (exposure_time, image_size, filter_name, session) in group_shapes:
                    height, width, channels = group_shapes[(exposure_time, image_size, filter_name, session)]
                else:
                    ref_data, _, _, _ = load_image(file_list[0])
                    if ref_data is None:
                        self.main.update_status(f"❌ Failed to load reference {os.path.basename(file_list[0])}")
                        continue
                    height, width = ref_data.shape[:2]
                    channels = 1 if ref_data.ndim == 2 else 3

                # --- choose reducer based on N (adaptive) ---
                N = len(file_list)
                algo_name, params, cpu_label = _select_reducer("flat", N)
                use_gpu = bool(self.main._hw_accel_enabled()) and _torch_ok() and _gpu_algo_supported(algo_name)
                self.main.update_status(f"⚙️ Normalizing {N} flats by per-frame medians (central patch).")
                QApplication.processEvents()
                # --- precompute normalization scales (fast, one central patch) ---
                scales = _estimate_flat_scales(file_list, height, width, channels, dark_data)

                self.main.update_status(f"⚙️ {'GPU' if use_gpu else 'CPU'} reducer for flats — {algo_name} ({'k=%.1f' % params.get('kappa', 0) if cpu_label=='kappa1' else 'trim=%.0f%%' % (params.get('trim_fraction', 0)*100) if cpu_label=='trimmed' else 'median'})")
                QApplication.processEvents()

                memmap_path = os.path.join(master_dir, f"temp_flat_{session}_{exposure_time}_{image_size}_{filter_name}.dat")
                self.main.update_status(f"🗂️ Creating temp memmap: {os.path.basename(memmap_path)} (shape={height}×{width}×{channels})")
                QApplication.processEvents()
                final_stacked = np.memmap(memmap_path, dtype=np.float32, mode="w+", shape=(height, width, channels))

                tiles = _tile_grid(height, width, self.main.chunk_height, self.main.chunk_width)
                total_tiles_group = len(tiles)
                self.main.update_status(f"📦 {total_tiles_group} tiles to process for this group.")
                QApplication.processEvents()

                # allocate max-chunk buffers (C-order, float32)
                buf0 = np.empty((N, min(self.main.chunk_height, height), min(self.main.chunk_width, width), channels),
                                dtype=np.float32, order="C")
                buf1 = np.empty_like(buf0)

                from concurrent.futures import ThreadPoolExecutor
                tp = ThreadPoolExecutor(max_workers=1)

                # prime first read
                (y0, y1, x0, x1) = tiles[0]
                fut = tp.submit(_read_tile_stack, file_list, y0, y1, x0, x1, channels, buf0)
                use0 = True

                self.main.update_status(f"▶️ Starting tile processing for group '{filter_name}' ({exposure_time}s, {image_size})")
                QApplication.processEvents()

                for t_idx, (y0, y1, x0, x1) in enumerate(tiles, start=1):
                    if pd.cancelled:
                        break

                    th, tw = fut.result()
                    ts_np = (buf0 if use0 else buf1)[:N, :th, :tw, :channels]  # (F, th, tw, C)

                    # prefetch next
                    if t_idx < total_tiles_group:
                        ny0, ny1, nx0, nx1 = tiles[t_idx]
                        fut = tp.submit(_read_tile_stack, file_list, ny0, ny1, nx0, nx1, channels,
                                        (buf1 if use0 else buf0))

                    pd.set_label(f"[{filter_name}] {session} — y:{y0}-{y1} x:{x0}-{x1}")

                    # ---- per-tile dark subtraction (HWC), BEFORE normalization ----
                    if dark_data is not None:
                        dsub = dark_data
                        if dsub.ndim == 3 and dsub.shape[0] in (1, 3):  # CHW → HWC
                            dsub = dsub.transpose(1, 2, 0)
                        d_tile = dsub[y0:y1, x0:x1].astype(np.float32, copy=False)
                        if d_tile.ndim == 2 and channels == 3:
                            d_tile = np.repeat(d_tile[..., None], 3, axis=2)
                        _subtract_dark_stack_inplace_hwc(ts_np, d_tile, pedestal=0.0)

                    # ---- fast per-frame normalization (divide by precomputed scale) ----
                    ts_np /= scales.reshape(N, 1, 1, 1)

                    # ---- reduction (GPU or CPU) ----
                    if use_gpu:
                        # pass parameters through the GPU reducer
                        tile_result, _ = _torch_reduce_tile(
                            ts_np,
                            np.ones((N,), dtype=np.float32),
                            algo_name=algo_name,
                            kappa=float(params.get("kappa", getattr(self, "kappa", 3.0))),
                            iterations=int(params.get("iterations", getattr(self, "iterations", 1))),
                            sigma_low=float(getattr(self, "sigma_low", 2.5)),
                            sigma_high=float(getattr(self, "sigma_high", 2.5)),
                            trim_fraction=float(params.get("trim_fraction", getattr(self, "trim_fraction", 0.05))),
                            esd_threshold=float(getattr(self, "esd_threshold", 3.0)),
                            biweight_constant=float(getattr(self, "biweight_constant", 6.0)),
                            modz_threshold=float(getattr(self, "modz_threshold", 3.5)),
                            comet_hclip_k=float(self.main.settings.value("stacking/comet_hclip_k", 1.30, type=float)),
                            comet_hclip_p=float(self.main.settings.value("stacking/comet_hclip_p", 25.0, type=float)),
                        )
                    else:
                        if cpu_label == "median":
                            tile_result = _cpu_tile_median(ts_np)
                        elif cpu_label == "trimmed":
                            tile_result = _cpu_tile_trimmed_mean(ts_np, float(params.get("trim_fraction", 0.05)))
                        else:  # 'kappa1'
                            tile_result = _cpu_tile_kappa_sigma_1iter(ts_np, float(params.get("kappa", 3.0)))

                    # commit + progress
                    final_stacked[y0:y1, x0:x1, :] = tile_result
                    pd.step()
                    use0 = not use0

                tp.shutdown(wait=True)

                if pd.cancelled:
                    try: del final_stacked
                    except Exception as e:
                        import logging
                        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                    try: os.remove(memmap_path)
                    except Exception as e:
                        import logging
                        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                    self.main.update_status("⛔ Master Flat creation cancelled; cleaning up temporary files.")
                    break

                master_flat_data = np.asarray(final_stacked, dtype=np.float32)
                del final_stacked
                gc.collect()  # Free memory after master flat creation

                master_flat_stem = f"MasterFlat_{session}_{int(exposure_time)}s_{image_size}_{filter_name}"
                master_flat_path = self.main._build_out(master_dir, master_flat_stem, "fit")

                header = fits.Header()
                header["IMAGETYP"] = "FLAT"
                header["EXPTIME"]  = (exposure_time, "grouped exposure")
                header["FILTER"]   = filter_name
                header["NAXIS"]    = 3 if channels == 3 else 2
                header["NAXIS1"]   = width
                header["NAXIS2"]   = height
                if channels == 3: header["NAXIS3"] = 3

                save_image(master_flat_data, master_flat_path, "fit", "32-bit floating point", header, is_mono=(channels == 1))
                key = f"{filter_name} ({image_size}) [{session}]"
                self.main.master_files[key] = master_flat_path
                self.main.master_sizes[master_flat_path] = image_size
                self.add_master_flat_to_tree(filter_name, master_flat_path)
                self.main.update_status(f"✅ Master Flat saved: {master_flat_path}")
                QApplication.processEvents()
                self.main.save_master_paths_to_settings()

            self.main.assign_best_master_dark()
            self.main.assign_best_master_files()
        finally:
            try: _free_torch_memory()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            pd.close()



    def add_master_flat_to_tree(self, filter_name, master_flat_path):
        """ Adds the newly created Master Flat to the Master Flat TreeBox and stores it. """

        key = f"{filter_name} ({self.main.master_sizes[master_flat_path]})"
        self.main.master_files[key] = master_flat_path  # ✅ Store the flat file for future use
        print(f"📝 DEBUG: Stored Master Flat -> {key}: {master_flat_path}")

        existing_items = self.main.master_flat_tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)

        if existing_items:
            filter_item = existing_items[0]
        else:
            filter_item = QTreeWidgetItem([filter_name])
            self.main.master_flat_tree.addTopLevelItem(filter_item)

        master_item = QTreeWidgetItem([os.path.basename(master_flat_path)])
        filter_item.addChild(master_item)


    def rebuild_flat_tree(self):
        """Regroup flat frames in the flat_tree based on the exposure tolerance."""
        self.main.flat_tree.clear()
        if not self.main.flat_files:
            return

        import numpy as np
        import re

        tol = float(self.main.flat_exposure_tolerance_spinbox.value())

        # Flatten from our canonical flat_files mapping:
        #   (flat_key, session_tag) -> [paths]
        # where flat_key looks like "FilterName - 1.00s (6264x4180)"
        all_flats = []
        for (filter_exp_size, session_tag), files in self.main.flat_files.items():
            for f in files:
                all_flats.append((filter_exp_size, session_tag, f))

        grouped = {}  # (filter_name, min_exp, max_exp, image_size) -> [(path, exposure, session)]

        for (filter_exp_size, session_tag, file_path) in all_flats:
            try:
                # Same parsing logic we use in create_master_flat()
                filter_name, exposure_size = filter_exp_size.split(" - ")
                exposure_str, image_size = exposure_size.split(" (")
                image_size = image_size.rstrip(")")

                m = re.match(r"([\d.]+)", exposure_str)
                exposure = float(m.group(1)) if m else 0.0
            except Exception as e:
                self.main.update_status(f"⚠️ rebuild_flat_tree: could not parse key '{filter_exp_size}': {e}")
                continue

            # find compatible bucket (by filter+size+exp within tol)
            found = None
            for (filt, mn, mx, sz) in grouped.keys():
                if filt == filter_name and sz == image_size and (mn - tol) <= exposure <= (mx + tol):
                    found = (filt, mn, mx, sz)
                    break

            if found:
                grouped[found].append((file_path, exposure, session_tag))
            else:
                key = (filter_name, exposure, exposure, image_size)
                grouped.setdefault(key, []).append((file_path, exposure, session_tag))

        # Build tree from grouped buckets
        for (filter_name, min_exp, max_exp, image_size), files in grouped.items():
            e_min = min(p[1] for p in files)
            e_max = max(p[1] for p in files)

            expmin = np.floor(e_min)
            # label matches what create_master_flat expects to re-parse
            exposure_str = f"{expmin:.1f}s–{(expmin + tol):.1f}s" if len(files) > 1 else f"{e_min:.1f}s"

            top_item = QTreeWidgetItem()
            top_item.setText(0, f"{filter_name} - {exposure_str} ({image_size})")
            top_item.setText(1, f"{len(files)} files")

            # canonical group_key (used by overrides & create_master_flat)
            group_key = f"{filter_name}|{image_size}|{expmin:.3f}|{(expmin+tol):.3f}"
            top_item.setData(0, Qt.ItemDataRole.UserRole, group_key)

            # column 2 shows what dark will be used
            ud = self.main.flat_dark_override.get(group_key, None)  # None→Auto, "__NO_DARK__", or path
            if ud is None:
                col2_txt = "Auto" if self.main.auto_select_dark_checkbox.isChecked() else "None"
            elif ud == "__NO_DARK__":
                col2_txt = "No Calibration"
            else:
                col2_txt = os.path.basename(ud)
            top_item.setText(2, col2_txt)
            top_item.setData(2, Qt.ItemDataRole.UserRole, ud)

            self.main.flat_tree.addTopLevelItem(top_item)

            # leaves: we don’t touch file paths or sessions
            for file_path, _, session_tag in files:
                leaf_item = QTreeWidgetItem([
                    os.path.basename(file_path),
                    f"Size: {image_size} | Session: {session_tag}"
                ])
                leaf_item.setData(0, Qt.ItemDataRole.UserRole, file_path)
                top_item.addChild(leaf_item)




