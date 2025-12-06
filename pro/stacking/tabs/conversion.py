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


class ConversionTab(QObject):
    """Extracted Conversion tab functionality."""
    
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

    def create_conversion_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("Batch Convert Files to Debayered FITS (.fit)"))

        # 1) Create the tree
        self.main.conversion_tree = QTreeWidget()
        self.main.conversion_tree.setColumnCount(2)
        self.main.conversion_tree.setHeaderLabels(["File", "Status"])

        # 2) Make columns user-resizable (Interactive)
        header = self.main.conversion_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)

        # 3) After populating the tree, do an initial auto-resize
        self.main.conversion_tree.resizeColumnToContents(0)
        self.main.conversion_tree.resizeColumnToContents(1)
        layout.addWidget(self.main.conversion_tree)

        # Buttons for adding files, adding a directory,
        # selecting an output directory, and clearing the list.
        btn_layout = QHBoxLayout()
        self.main.add_conversion_files_btn = QPushButton("Add Conversion Files")
        self.main.add_conversion_files_btn.clicked.connect(self.add_conversion_files)
        self.main.add_conversion_dir_btn = QPushButton("Add Conversion Directory")
        self.main.add_conversion_dir_btn.clicked.connect(self.add_conversion_directory)
        self.main.select_conversion_output_btn = QPushButton("Select Output Directory")
        self.main.select_conversion_output_btn.clicked.connect(self.select_conversion_output_dir)
        self.main.clear_conversion_btn = QPushButton("Clear List")
        self.main.clear_conversion_btn.clicked.connect(self.clear_conversion_list)
        btn_layout.addWidget(self.main.add_conversion_files_btn)
        btn_layout.addWidget(self.main.add_conversion_dir_btn)
        btn_layout.addWidget(self.main.select_conversion_output_btn)
        btn_layout.addWidget(self.main.clear_conversion_btn)
        layout.addLayout(btn_layout)

        # Convert All button (converts all files in the tree).
        self.main.convert_btn = QPushButton("Convert All Files to FITS")
        self.main.convert_btn.clicked.connect(self.convert_all_files)
        layout.addWidget(self.main.convert_btn)

        return tab


    def add_conversion_files(self):
        last_dir = self.main.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(self.main, "Select Files for Conversion", last_dir,
                                                "Supported Files (*.fits *.fit *.fz *.fz *.fits.gz *.fit.gz *.tiff *.tif *.png *.jpg *.jpeg *.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2 *.pef *.xisf)")
        if files:
            self.main.settings.setValue("last_opened_folder", os.path.dirname(files[0]))
            for file in files:
                item = QTreeWidgetItem([os.path.basename(file), "Pending"])
                item.setData(0, 1000, file)  # store full path in role 1000
                self.main.conversion_tree.addTopLevelItem(item)


    def add_conversion_directory(self):
        last_dir = self.main.settings.value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self.main, "Select Directory for Conversion", last_dir)
        if directory:
            self.main.settings.setValue("last_opened_folder", directory)
            for file in os.listdir(directory):
                if file.lower().endswith((".fits", ".fit", ".fz", ".fz", ".fit.gz", ".fits.gz", ".tiff", ".tif", ".png", ".jpg", ".jpeg", 
                                           ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".xisf")):
                    full_path = os.path.join(directory, file)
                    item = QTreeWidgetItem([file, "Pending"])
                    item.setData(0, 1000, full_path)
                    self.main.conversion_tree.addTopLevelItem(item)


    def select_conversion_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self.main, "Select Conversion Output Directory")
        if directory:
            self.main.conversion_output_directory = directory
            self.main.update_status(f"Conversion output directory set to: {directory}")


    def clear_conversion_list(self):
        self.main.conversion_tree.clear()
        self.main.update_status("Conversion list cleared.")


    def convert_all_files(self):
        # If no output directory is set, ask the user if they want to set it now.
        if not self.main.conversion_output_directory:
            reply = QMessageBox.question(
                self.main,
                "No Output Directory",
                "No output directory is set. Do you want to select one now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.select_conversion_output_dir()  # Let them pick a folder
            else:
                # They chose 'No' → just stop
                return

            # If it's still empty after that, bail out
            if not self.main.conversion_output_directory:
                QMessageBox.warning(self.main, "No Output Directory", "Please select a conversion output directory first.")
                return

        count = self.main.conversion_tree.topLevelItemCount()
        if count == 0:
            QMessageBox.information(self.main, "No Files", "There are no files to convert.")
            return

        # 1) Show the batch settings dialog
        dialog = BatchSettingsDialog(self.main)
        result = dialog.exec()
        if result == int(QDialog.DialogCode.Rejected):
            # user canceled
            return
        # user pressed OK => get the values
        imagetyp_user, exptime_user, filter_user = dialog.get_values()

        for i in range(count):
            item = self.main.conversion_tree.topLevelItem(i)
            file_path = item.data(0, 1000)
            result = load_image(file_path)
            if result[0] is None:
                item.setText(1, "Failed to load")
                self.main.update_status(f"Failed to load {os.path.basename(file_path)}")
                continue

            image, header, bit_depth, is_mono = result

            if image is None:
                item.setText(1, "Failed to load")
                self.main.update_status(f"Failed to load {os.path.basename(file_path)}")
                continue

            # 🔹 Always ensure we have a basic FITS header for non-FITS sources
            header = _ensure_minimal_header(header, file_path)

            # Debayer if needed:
            image = self.debayer_image(image, file_path, header)
            if image.ndim == 3:
                is_mono = False

            if image.ndim == 3:
                header['COLOR'] = True
                header['NAXIS'] = 3
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
                header['NAXIS3'] = image.shape[2]
            else:
                header['COLOR'] = False
                header['NAXIS'] = 2
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]

            # If it's a RAW format, definitely treat as color
            if _is_raw_file(file_path):
                is_mono = False
                header = _enrich_header_from_exif(header, file_path)

            header['IMAGETYP'] = imagetyp_user
            header['FILTER']   = filter_user

            # Normalize the batch EXPTIME input once, outside the loop
            user_exptime_str = (exptime_user or "").strip()

            # Treat certain values as "do not override"
            sentinel_values = {"", "unknown", "auto", "camera", "keep"}

            if user_exptime_str.lower() not in sentinel_values:
                # User is explicitly forcing an exposure value → override
                try:
                    if "/" in user_exptime_str:
                        top, bot = user_exptime_str.split("/", 1)
                        exptime_val = float(top) / float(bot)
                    else:
                        exptime_val = float(user_exptime_str)
                    header["EXPTIME"] = (exptime_val, "User-specified exposure (s)")
                except (ValueError, ZeroDivisionError):
                    # If they type something weird but non-sentinel, store it verbatim
                    header["EXPTIME"] = (user_exptime_str, "User-specified exposure")

            # -- Ensure EXPTIME is defined at all --
            if 'EXPTIME' not in header:
                header['EXPTIME'] = ('Unknown', "No exposure info available")


            # Remove any existing NAXIS keywords
            for key in ["NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"]:
                header.pop(key, None)

            if image.ndim == 2:
                header['NAXIS'] = 2
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
            elif image.ndim == 3:
                header['NAXIS'] = 3
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
                header['NAXIS3'] = image.shape[2]

            # -- Ensure EXPTIME is defined --
            if 'EXPTIME' not in header:
                # If the camera or exif didn't provide it, we set it to 'Unknown'
                header['EXPTIME'] = 'Unknown'

            # Build output filename and save
            base = os.path.basename(file_path)
            name, _ = os.path.splitext(base)
            output_filename = os.path.join(self.main.conversion_output_directory, f"{name}.fit")
            maxv = float(np.max(image))


            try:
                save_image(
                    img_array=image,
                    filename=output_filename,
                    original_format="fit",
                    bit_depth="16-bit",
                    original_header=header,
                    is_mono=is_mono
                )
                item.setText(1, "Converted")
                self.main.update_status(
                    f"Converted {os.path.basename(file_path)} to FITS with "
                    f"IMAGETYP={header['IMAGETYP']}, EXPTIME={header['EXPTIME']}."
                )
            except Exception as e:
                item.setText(1, f"Error: {e}")
                self.main.update_status(f"Error converting {os.path.basename(file_path)}: {e}")

            QApplication.processEvents()

        self.main.update_status("Conversion complete.")




    def debayer_image(self, image, file_path, header):
        """
        Returns RGB if debayered, otherwise returns input unchanged.
        Also stamps header with BAYERPAT or XTRANS where possible.
        """
        # per-run override if set, else honor the checkbox
        if self.main._cfa_for_this_run is None:
            cfa = bool(getattr(self, "cfa_drizzle_cb", None) and self.main.cfa_drizzle_cb.isChecked())
        else:
            cfa = bool(self.main._cfa_for_this_run)
        #print(f"[DEBUG] Debayering with CFA drizzle = {cfa}")

        ext = file_path.lower()
        is_raw = ext.endswith(('.cr2','.cr3','.nef','.arw','.dng','.raf','.orf','.rw2','.pef'))

        # --- RAW files ---
        if is_raw:
            try:
                import rawpy
                with rawpy.imread(file_path) as rp:
                    fam = _rawpy_is_xtrans_or_bayer(rp)
                    if fam == "BAYER":
                        # Try to get a concrete 2x2 token
                        token = _rawpy_pattern_to_token(rp)
                        if token:
                            try: header['BAYERPAT'] = token
                            except Exception as e:
                                import logging
                                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                            try:
                                from legacy.numba_utils import debayer_raw_fast
                                return debayer_raw_fast(image, bayer_pattern=token, cfa_drizzle=cfa, method="edge")
                            except Exception:
                                return debayer_fits_fast(image, token, cfa_drizzle=cfa)

                    # X-Trans or ambiguous → treat Fuji RAF/X-Series as X-Trans
                    if fam == "XTRANS" or _probably_fuji_xtrans(file_path, header):
                        print("[INFO] Demosaicing via rawpy (X-Trans path).")
                        rgb16 = rp.postprocess(
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # or DHT
                            no_auto_bright=True,
                            gamma=(1.0, 1.0),
                            output_bps=16,
                            use_camera_wb=True,
                            fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                            four_color_rgb=False,
                            half_size=False,
                            bright=1.0,
                            highlight_mode=rawpy.HighlightMode.Clip
                        )
                        rgb = (rgb16.astype(np.float32) / 65535.0)
                        try: header['XTRANS'] = True
                        except Exception as e:
                            import logging
                            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                        return rgb

                    # If still unknown, last attempt: if we do have a 2x2 pattern, use it
                    token = _rawpy_pattern_to_token(rp)
                    if token:
                        try: header['BAYERPAT'] = token
                        except Exception as e:
                            import logging
                            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                        try:
                            from legacy.numba_utils import debayer_raw_fast
                            return debayer_raw_fast(image, bayer_pattern=token, cfa_drizzle=cfa, method="edge")
                        except Exception:
                            return debayer_fits_fast(image, token, cfa_drizzle=cfa)

                    print("[WARN] RAW family unknown; leaving as-is.")
            except Exception as e:
                print(f"[WARN] rawpy probe failed for {file_path}: {e}")
                # fall through

        # --- FITS (already mosaic) ---
        if ext.endswith(('.fits','.fit','.fz')):
            bp = (str(header.get('BAYERPAT') or header.get('BAYERPATN') or header.get('CFA_PATTERN') or "")).upper()
            if bp in {"RGGB","BGGR","GRBG","GBRG"}:
                try:
                    from legacy.numba_utils import debayer_raw_fast
                    return debayer_raw_fast(image, bayer_pattern=bp, cfa_drizzle=cfa, method="edge")
                except Exception:
                    return debayer_fits_fast(image, bp, cfa_drizzle=cfa)

            # If FITS came from Fuji RAW (RAF) and no BAYERPAT, it was likely X-Trans;
            # without the RAW we cannot demosaic now, so keep mosaic but mark it.
            if _probably_fuji_xtrans(file_path, header):
                try: header['XTRANS'] = True
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                print("[INFO] FITS likely from X-Trans; cannot demosaic without RAW. Keeping mosaic.")
                return image

        return image




