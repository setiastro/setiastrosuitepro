# Tab module - imports from parent packages
from __future__ import annotations
import os
import sys
import platform
import math
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


class DarkTab(QObject):
    """Extracted Dark tab functionality."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def create_dark_tab(self):
        tab = QWidget()
        main_layout = QVBoxLayout(tab)  # Vertical layout to separate sections

        # --- DARK FRAMES TREEBOX (TOP) ---
        darks_layout = QHBoxLayout()  # Left = Dark Tree, Right = Controls

        # Left Side - Dark Frames
        dark_frames_layout = QVBoxLayout()
        dark_frames_layout.addWidget(QLabel("Dark Frames"))
        # 1) Create the tree
        self.main.dark_tree = QTreeWidget()
        self.main.dark_tree.setColumnCount(2)
        self.main.dark_tree.setHeaderLabels(["Exposure Time", "Metadata"])
        self.main.dark_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # 2) Make columns user-resizable
        header = self.main.dark_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)

        # 3) After you fill the tree with items, auto-resize
        self.main.dark_tree.resizeColumnToContents(0)
        self.main.dark_tree.resizeColumnToContents(1)

        # Then add it to the layout
        dark_frames_layout.addWidget(self.main.dark_tree)

        # Buttons to Add Dark Files & Directories
        btn_layout = QHBoxLayout()
        self.main.add_dark_files_btn = QPushButton("Add Dark Files")
        self.main.add_dark_files_btn.clicked.connect(self.add_dark_files)
        self.main.add_dark_dir_btn = QPushButton("Add Dark Directory")
        self.main.add_dark_dir_btn.clicked.connect(self.add_dark_directory)
        btn_layout.addWidget(self.main.add_dark_files_btn)
        btn_layout.addWidget(self.main.add_dark_dir_btn)
        dark_frames_layout.addLayout(btn_layout)

        self.main.clear_dark_selection_btn = QPushButton("Clear Selection")
        self.main.clear_dark_selection_btn.clicked.connect(lambda: self.main.clear_tree_selection(self.main.dark_tree, self.main.dark_files))
        dark_frames_layout.addWidget(self.main.clear_dark_selection_btn)

        darks_layout.addLayout(dark_frames_layout, 2)  # Dark Frames Tree takes more space


        # --- RIGHT SIDE: Exposure Tolerance & Master Darks Button ---
        right_controls_layout = QVBoxLayout()

        # Exposure Tolerance
        exposure_tolerance_layout = QHBoxLayout()
        exposure_tolerance_label = QLabel("Exposure Tolerance (seconds):")
        self.main.exposure_tolerance_spinbox = QSpinBox()
        self.main.exposure_tolerance_spinbox.setRange(0, 30)  # Acceptable range
        self.main.exposure_tolerance_spinbox.setValue(5)  # Default: ±5 sec
        exposure_tolerance_layout.addWidget(exposure_tolerance_label)
        exposure_tolerance_layout.addWidget(self.main.exposure_tolerance_spinbox)
        right_controls_layout.addLayout(exposure_tolerance_layout)

        # --- "Turn Those Darks Into Master Darks" Button ---
        self.main.create_master_dark_btn = QPushButton("Turn Those Darks Into Master Darks")
        self.main.create_master_dark_btn.clicked.connect(self.create_master_dark)

        # Apply a bold font, padding, and a highlighted effect
        self.main.create_master_dark_btn.setStyleSheet("""
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

        right_controls_layout.addWidget(self.main.create_master_dark_btn)


        darks_layout.addLayout(right_controls_layout, 1)  # Right side takes less space

        main_layout.addLayout(darks_layout)

        # --- MASTER DARKS TREEBOX (BOTTOM) ---
        main_layout.addWidget(QLabel("Master Darks"))
        self.main.master_dark_tree = QTreeWidget()
        self.main.master_dark_tree.setColumnCount(2)
        self.main.master_dark_tree.setHeaderLabels(["Exposure Time", "Master File"])
        self.main.master_dark_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        main_layout.addWidget(self.main.master_dark_tree)

        # Master Dark Selection Button
        self.main.master_dark_btn = QPushButton("Load Master Dark")
        self.main.master_dark_btn.clicked.connect(self.load_master_dark)
        main_layout.addWidget(self.main.master_dark_btn)

        # Add "Clear Selection" button for Master Darks
        self.main.clear_master_dark_selection_btn = QPushButton("Clear Selection")
        self.main.clear_master_dark_selection_btn.clicked.connect(
            lambda: self.main.clear_tree_selection(self.main.master_dark_tree, self.main.master_files)
        )
        self.main.clear_master_dark_selection_btn.clicked.connect(
            lambda: (self.main.clear_tree_selection(self.main.master_dark_tree, self.main.master_files),
                    self.main.save_master_paths_to_settings())
        )        
        main_layout.addWidget(self.main.clear_master_dark_selection_btn)

        return tab


    def add_dark_files(self):
        self.main.add_files(self.main.dark_tree, "Select Dark Files", "DARK")
    

    def add_dark_directory(self):
        self.main.add_directory(self.main.dark_tree, "Select Dark Directory", "DARK")


    def load_master_dark(self):
        """ Loads a Master Dark and updates the UI. """
        last_dir = self.main.settings.value("last_opened_folder", "", type=str)  # Get last folder
        files, _ = QFileDialog.getOpenFileNames(self.main, "Select Master Dark", last_dir, "FITS Files (*.fits *.fit)")
        
        if files:
            self.main.settings.setValue("last_opened_folder", os.path.dirname(files[0]))  # Save last used folder
            self.main.add_master_files(self.main.master_dark_tree, "DARK", files)
            self.main.save_master_paths_to_settings() 

        self.main.update_override_dark_combo()
        self.main.assign_best_master_dark()
        self.main.assign_best_master_files()
        print("DEBUG: Loaded Master Darks and updated assignments.")



    def create_master_dark(self):
        """Creates master darks with minimal RAM usage by loading frames in small tiles (GPU-accelerated if available),
        with adaptive reducers."""
        self.main.update_status(f"Starting Master Dark Creation...")
        if not self.main.stacking_directory:
            self.main.select_stacking_directory()
            if not self.main.stacking_directory:
                QMessageBox.warning(self.main, "Error", "Output directory is not set.")
                return

        # Keep both paths available; we'll override algo selection per group.
        ui_algo = getattr(self, "calib_rejection_algorithm", "Windsorized Sigma Clipping")
        if ui_algo == "Weighted Windsorized Sigma Clipping":
            ui_algo = "Windsorized Sigma Clipping"

        exposure_tolerance = self.main.exposure_tolerance_spinbox.value()
        dark_files_by_group = {}

        # group darks by exposure + size
        for exposure_key, file_list in self.main.dark_files.items():
            exposure_time_str, image_size = exposure_key.split(" (")
            image_size = image_size.rstrip(")")
            exposure_time = float(exposure_time_str.replace("s", "")) if "Unknown" not in exposure_time_str else 0

            matched_group = None
            for (existing_exposure, existing_size) in dark_files_by_group.keys():
                if abs(existing_exposure - exposure_time) <= exposure_tolerance and existing_size == image_size:
                    matched_group = (existing_exposure, existing_size)
                    break
            if matched_group is None:
                matched_group = (exposure_time, image_size)
                dark_files_by_group[matched_group] = []
            dark_files_by_group[matched_group].extend(file_list)

        master_dir = os.path.join(self.main.stacking_directory, "Master_Calibration_Files")
        os.makedirs(master_dir, exist_ok=True)

        # Informative status about discovery
        try:
            n_groups = sum(1 for k, v in dark_files_by_group.items() if len(v) >= 2)
            total_files = sum(len(v) for v in dark_files_by_group.values())
            self.main.update_status(f"🔎 Discovered {len(dark_files_by_group)} grouped exposures ({n_groups} eligible to stack) — {total_files} files total.")
        except Exception:
            pass
        QApplication.processEvents()

        # Pre-count tiles
        chunk_height = self.main.chunk_height
        chunk_width  = self.main.chunk_width
        total_tiles = 0
        group_shapes = {}
        for (exposure_time, image_size), file_list in dark_files_by_group.items():
            if len(file_list) < 2:
                continue
            ref_data, _, _, _ = load_image(file_list[0])
            if ref_data is None:
                continue
            H, W = ref_data.shape[:2]
            C = 1 if (ref_data.ndim == 2) else 3
            group_shapes[(exposure_time, image_size)] = (H, W, C)
            total_tiles += _count_tiles(H, W, chunk_height, chunk_width)

        if total_tiles == 0:
            self.main.update_status("⚠️ No eligible dark groups found to stack.")
            return

        self.main.update_status(f"🧭 Total tiles to process: {total_tiles} (chunk size {chunk_height}×{chunk_width})")
        QApplication.processEvents()

        # ------- small helpers (local) -------------------------------------------
        def _select_reducer(kind: str, N: int):
            if kind == "dark":
                if N < 16:
                    return ("Kappa-Sigma Clipping", {"kappa": 3.0, "iterations": 1}, "kappa1")
                elif N <= 200:
                    return ("Simple Median (No Rejection)", {}, "median")
                else:
                    return ("Trimmed Mean", {"trim_fraction": 0.05}, "trimmed")
            else:
                raise ValueError("wrong kind")

        def _cpu_tile_median(ts4: np.ndarray) -> np.ndarray:
            return np.median(ts4, axis=0).astype(np.float32, copy=False)

        def _cpu_tile_trimmed_mean(ts4: np.ndarray, frac: float) -> np.ndarray:
            if frac <= 0.0:
                return ts4.mean(axis=0, dtype=np.float32)
            F = ts4.shape[0]
            k = int(max(1, round(frac * F)))
            if 2 * k >= F:
                return _cpu_tile_median(ts4)
            s = np.sort(ts4, axis=0)
            core = s[k:F - k]
            return core.mean(axis=0, dtype=np.float32)

        def _cpu_tile_kappa_sigma_1iter(ts4: np.ndarray, kappa: float = 3.0) -> np.ndarray:
            med = np.median(ts4, axis=0)
            std = ts4.std(axis=0, dtype=np.float32)
            lo = med - kappa * std
            hi = med + kappa * std
            mask = (ts4 >= lo) & (ts4 <= hi)
            num = (ts4 * mask).sum(axis=0, dtype=np.float32)
            cnt = mask.sum(axis=0).astype(np.float32)
            out = np.where(cnt > 0, num / np.maximum(cnt, 1.0), med)
            return out.astype(np.float32, copy=False)

        pd = _Progress(self.main, "Create Master Darks", total_tiles)
        try:
            for (exposure_time, image_size), file_list in dark_files_by_group.items():
                if len(file_list) < 2:
                    self.main.update_status(f"⚠️ Skipping {exposure_time}s ({image_size}) - Not enough frames to stack.")
                    QApplication.processEvents()
                    continue
                if pd.cancelled:
                    self.main.update_status("⛔ Master Dark creation cancelled.")
                    break

                self.main.update_status(f"🟢 Processing {len(file_list)} darks for {exposure_time}s ({image_size}) exposure…")
                QApplication.processEvents()

                # reference shape
                if (exposure_time, image_size) in group_shapes:
                    height, width, channels = group_shapes[(exposure_time, image_size)]
                else:
                    ref_data, _, _, _ = load_image(file_list[0])
                    if ref_data is None:
                        self.main.update_status(f"❌ Failed to load reference {os.path.basename(file_list[0])}")
                        continue
                    height, width = ref_data.shape[:2]
                    channels = 1 if (ref_data.ndim == 2) else 3

                # --- choose reducer adaptively ---
                N = len(file_list)
                algo_name, params, cpu_label = _select_reducer("dark", N)
                use_gpu = bool(self.main._hw_accel_enabled()) and _torch_ok() and _gpu_algo_supported(algo_name)
                algo_brief = ('GPU' if use_gpu else 'CPU') + " " + algo_name
                self.main.update_status(f"⚙️ {algo_brief} selected for {N} frames ({'channels='+str(channels)})")
                QApplication.processEvents()

                memmap_path = os.path.join(master_dir, f"temp_dark_{exposure_time}_{image_size}.dat")
                self.main.update_status(f"🗂️ Creating temp memmap: {os.path.basename(memmap_path)} (shape={height}×{width}×{channels})")
                QApplication.processEvents()
                final_stacked = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(height, width, channels))

                tiles = _tile_grid(height, width, chunk_height, chunk_width)
                total_tiles_group = len(tiles)
                self.main.update_status(f"📦 {total_tiles_group} tiles to process for this group.")
                QApplication.processEvents()

                # double-buffer
                buf0 = np.empty((N, min(chunk_height, height), min(chunk_width, width), channels),
                                dtype=np.float32, order="C")
                buf1 = np.empty_like(buf0)

                from concurrent.futures import ThreadPoolExecutor
                tp = ThreadPoolExecutor(max_workers=1)

                # prime first read
                (y0, y1, x0, x1) = tiles[0]
                fut = tp.submit(_read_tile_stack, file_list, y0, y1, x0, x1, channels, buf0)
                use0 = True

                for t_idx, (y0, y1, x0, x1) in enumerate(tiles, start=1):
                    if pd.cancelled:
                        self.main.update_status("⛔ Master Dark creation cancelled during tile processing.")
                        break

                    th, tw = fut.result()
                    ts_np = (buf0 if use0 else buf1)[:N, :th, :tw, :channels]

                    # prefetch next
                    if t_idx < total_tiles_group:
                        ny0, ny1, nx0, nx1 = tiles[t_idx]
                        fut = tp.submit(_read_tile_stack, file_list, ny0, ny1, nx0, nx1, channels,
                                        (buf1 if use0 else buf0))

                    pd.set_label(f"{int(exposure_time)}s ({image_size}) — y:{y0}-{y1} x:{x0}-{x1}")

                    # ---- reduction (GPU or CPU) ----
                    if use_gpu:
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

                    # commit
                    final_stacked[y0:y1, x0:x1, :] = tile_result
                    pd.step()
                    use0 = not use0

                tp.shutdown(wait=True)

                if pd.cancelled:
                    self.main.update_status("⛔ Master Dark creation cancelled; cleaning up temporary files.")
                    try: del final_stacked
                    except Exception as e:
                        import logging
                        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                    try: os.remove(memmap_path)
                    except Exception as e:
                        import logging
                        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                    break

                master_dark_data = np.asarray(final_stacked, dtype=np.float32)
                del final_stacked
                gc.collect()  # Free memory after master dark creation

                master_dark_stem = f"MasterDark_{int(exposure_time)}s_{image_size}"
                master_dark_path = self.main._build_out(master_dir, master_dark_stem, "fit")

                master_header = fits.Header()
                master_header["IMAGETYP"] = "DARK"
                master_header["EXPTIME"]  = (exposure_time, "User-specified or from grouping")
                master_header["NAXIS"]    = 3 if channels==3 else 2
                master_header["NAXIS1"]   = master_dark_data.shape[1]
                master_header["NAXIS2"]   = master_dark_data.shape[0]
                if channels==3: master_header["NAXIS3"] = 3

                save_image(master_dark_data, master_dark_path, "fit", "32-bit floating point", master_header, is_mono=(channels==1))
                self.add_master_dark_to_tree(f"{exposure_time}s ({image_size})", master_dark_path)
                self.main.update_status(f"✅ Master Dark saved: {master_dark_path}")
                QApplication.processEvents()
                self.main.assign_best_master_files()
                self.main.save_master_paths_to_settings()

            # wrap-up
            self.main.assign_best_master_dark()
            self.main.update_override_dark_combo()
            self.main.assign_best_master_files()
        finally:
            try: _free_torch_memory()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            pd.close()
            

    def add_master_dark_to_tree(self, exposure_label: str, master_dark_path: str):
        """
        Adds the newly created Master Dark to the Master Dark TreeBox and updates the dropdown.

        Parameters
        ----------
        exposure_label : str
            A human-friendly label like "30s (4128x2806)". This is used as the
            tree's top-level item text and as the dictionary key.
        master_dark_path : str
            Full path to the master dark FITS.
        """
        # Build/confirm size string from the file (for master_sizes)
        try:
            with fits.open(master_dark_path, memmap=False) as hdul:
                data = hdul[0].data
                if data is None:
                    size = "Unknown"
                else:
                    if data.ndim == 2:
                        h, w = data.shape
                    elif data.ndim == 3:
                        h, w = data.shape[:2]
                    else:
                        h = w = 0
                    size = f"{w}x{h}" if (h and w) else "Unknown"
        except Exception:
            size = "Unknown"

        exposure_key = str(exposure_label).strip()  # e.g. "30s (4128x2806)"

        # ✅ Record paths/sizes
        if not hasattr(self, "master_files"):
            self.main.master_files = {}
        if not hasattr(self, "master_sizes"):
            self.main.master_sizes = {}

        self.main.master_files[exposure_key] = master_dark_path
        self.main.master_sizes[master_dark_path] = size

        # ✅ Update UI Tree
        existing_items = self.main.master_dark_tree.findItems(
            exposure_key, Qt.MatchFlag.MatchExactly, 0
        )
        if existing_items:
            exposure_item = existing_items[0]
        else:
            exposure_item = QTreeWidgetItem([exposure_key])
            self.main.master_dark_tree.addTopLevelItem(exposure_item)

        master_item = QTreeWidgetItem([os.path.basename(master_dark_path)])
        exposure_item.addChild(master_item)

        # ✅ Refresh UI bits that depend on master darks
        self.main.update_override_dark_combo()
        self.main.assign_best_master_dark()  # auto-select
        self.main.update_status(f"✅ Master Dark saved and added to UI: {master_dark_path}")
        print(f"📝 DEBUG: Stored Master Dark -> {exposure_key}: {master_dark_path}")





