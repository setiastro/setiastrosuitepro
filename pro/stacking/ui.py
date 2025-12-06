# Stacking module - imports from functions.py for shared utilities
from __future__ import annotations
import os
import sys
import math
import time
import numpy as np
import cv2
cv2.setNumThreads(0)

from PyQt6.QtCore import Qt, QTimer, QSettings, pyqtSignal, QObject, pyqtSlot, QThread, QEvent, QPoint, QSize, QEventLoop, QCoreApplication, QRectF, QPointF, QMetaObject
from PyQt6.QtGui import QIcon, QImage, QPixmap, QAction, QIntValidator, QDoubleValidator, QFontMetrics, QTextCursor, QPalette, QPainter, QPen, QTransform, QColor, QBrush, QCursor
from PyQt6.QtWidgets import (QDialog, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QTreeWidget, QHeaderView, QTreeWidgetItem, QProgressBar, QProgressDialog,
                             QFormLayout, QDialogButtonBox, QToolBar, QToolButton, QFileDialog, QTabWidget, QAbstractItemView, QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton,
                             QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QApplication, QScrollArea, QTextEdit, QMenu, QPlainTextEdit, QGraphicsEllipseItem,
                             QMessageBox, QSlider, QCheckBox, QInputDialog, QComboBox)

from astropy.io import fits
from datetime import datetime
import re
import shutil
import tempfile
import hashlib
import unicodedata
from pathlib import Path
from PyQt6 import sip
from numpy.lib.format import open_memmap

# Import shared utilities from functions.py
from .functions import (
    _asarray, _WINDOWS_RESERVED, _FITS_EXTS,
    get_valid_header, LRUDict, load_image, save_image,
    _torch_ok, _gpu_algo_supported, _torch_reduce_tile,
    windsorized_sigma_clip_weighted, kappa_sigma_clip_weighted,
    debayer_raw_fast, drizzle_deposit_numba_kernel_mono, drizzle_deposit_color_kernel,
    finalize_drizzle_2d, finalize_drizzle_3d,
    bulk_cosmetic_correction_numba, drizzle_deposit_numba_naive, drizzle_deposit_color_naive,
    bulk_cosmetic_correction_bayer,
    compute_star_count_fast_preview, siril_style_autostretch,
)
from .functions import *
from .functions import _get_log_dock
from .tabs import ConversionTab, DarkTab, FlatTab, LightTab, RegistrationTab, IntegrationTab


class StackingSuiteDialog(QDialog):
    requestRelaunch = pyqtSignal(str, str)  # old_dir, new_dir
    status_signal = pyqtSignal(str)

    def __init__(self, parent=None, wrench_path=None, spinner_path=None, **_ignored):
        super().__init__(parent)

        # Tab controller instances
        self.conversion_ctrl = ConversionTab(self)
        self.dark_ctrl = DarkTab(self)
        self.flat_ctrl = FlatTab(self)
        self.light_ctrl = LightTab(self)
        self.registration_ctrl = RegistrationTab(self)
        self.integration_ctrl = IntegrationTab(self)

        self.settings = QSettings()
        self._wrench_path = wrench_path
        self._spinner_path = spinner_path
        self._post_progress_label = None


        self.setWindowTitle("Stacking Suite")
        self.setGeometry(300, 200, 800, 600)

        self.per_group_drizzle = {}
        self.manual_dark_overrides = {}
        self.manual_flat_overrides = {}
        self.conversion_output_directory = None
        self.reg_files = {}
        self.session_tags = {}           # file_path => session_tag
        self.flat_dark_override = {}  # {(group_key:str): path | None | "__NO_DARK__"}
        self.deleted_calibrated_files = []
        self._norm_map = {}
        if not hasattr(self, "_mismatch_policy"):
            self._mismatch_policy = {}
        # Remember GUI thread for fast-path updates
        self._gui_thread = QThread.currentThread()

        # Status bus (singleton across app)
        app = QApplication.instance()
        if not hasattr(app, "_sasd_log_bus"):
            app._sasd_log_bus = LogBus()
        self._log_bus = app._sasd_log_bus

        # Connect status signal ONCE, queued to GUI thread
        try:
            self.status_signal.disconnect()
        except Exception:
            pass
        self.status_signal.connect(self._update_status_gui, Qt.ConnectionType.QueuedConnection)
        self.status_signal.connect(self._log_bus.posted.emit, Qt.ConnectionType.QueuedConnection)

        self._cfa_for_this_run = None  # None = follow checkbox; True/False = override for this run

        # Debounced progress (alignment)
        self._align_prog_timer = QTimer(self)
        self._align_prog_timer.setSingleShot(True)
        self._align_prog_timer.timeout.connect(self._flush_align_progress)
        self._align_prog_pending = None      # tuple[int, int] (done, total)
        self._align_prog_in_slot = False
        self._align_prog_last = None

        self.reference_frame = None
        self._comet_seed = None             # {'path': <original file>, 'xy': (x,y)}
        self._orig2norm = {}                # original path -> normalized *_n.fit
        self._comet_ref_xy = None           # comet coordinate in reference frame

        self.manual_light_files = []
        self._reg_excluded_files = set()

        # Show docking log window (if main window created it)
        self._ensure_log_visible_once()

        # Settings
        self.auto_rot180 = self.settings.value("stacking/auto_rot180", True, type=bool)
        self.auto_rot180_tol_deg = self.settings.value("stacking/auto_rot180_tol_deg", 89.0, type=float)

        dtype_str = self.settings.value("stacking/internal_dtype", "float64", type=str)
        self.internal_dtype = np.float64 if dtype_str == "float64" else np.float32
        self.star_trail_mode = self.settings.value("stacking/star_trail_mode", False, type=bool)

        self.align_refinement_passes = self.settings.value("stacking/refinement_passes", 3, type=int)
        self.align_shift_tolerance = self.settings.value("stacking/shift_tolerance_px", 0.2, type=float)

        # Load or default these
        self.stacking_directory = self.settings.value("stacking/dir", "", type=str)
        self.sigma_high = self.settings.value("stacking/sigma_high", 3.0, type=float)
        self.sigma_low = self.settings.value("stacking/sigma_low", 3.0, type=float)
        self.rejection_algorithm = self.settings.value(
            "stacking/rejection_algorithm", "Weighted Windsorized Sigma Clipping", type=str
        )
        self.kappa = self.settings.value("stacking/kappa", 2.5, type=float)
        self.iterations = self.settings.value("stacking/iterations", 3, type=int)
        self.esd_threshold = self.settings.value("stacking/esd_threshold", 3.0, type=float)
        self.biweight_constant = self.settings.value("stacking/biweight_constant", 6.0, type=float)
        self.trim_fraction = self.settings.value("stacking/trim_fraction", 0.1, type=float)
        self.modz_threshold = self.settings.value("stacking/modz_threshold", 3.5, type=float)
        self.chunk_height = self.settings.value("stacking/chunk_height", 2048, type=int)
        self.chunk_width = self.settings.value("stacking/chunk_width", 2048, type=int)

        # Dictionaries to store file paths
        self.conversion_files = {}
        self.dark_files = {}
        self.flat_files = {}
        self.light_files = {}
        self.master_files = {}
        self.master_sizes = {}

        # Layout & tabs
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.dir_path_edit = QLineEdit(self.stacking_directory)

        # Create tabs
        self.conversion_tab = self.conversion_ctrl.create_conversion_tab()
        self.dark_tab = self.dark_ctrl.create_dark_tab()
        self.flat_tab = self.flat_ctrl.create_flat_tab()
        self.light_tab = self.light_ctrl.create_light_tab()
        add_runtime_to_sys_path(status_cb=lambda *_: None)
        self.image_integration_tab = self.registration_ctrl.create_image_registration_tab()

        # Add tabs
        self.tabs.addTab(self.conversion_tab, "Convert Non-FITS Formats")
        self.tabs.addTab(self.dark_tab, "Darks")
        self.tabs.addTab(self.flat_tab, "Flats")
        self.tabs.addTab(self.light_tab, "Lights")
        self.tabs.addTab(self.image_integration_tab, "Image Integration")
        self.tabs.setCurrentIndex(1)

        # Header row
        self.wrench_button = QPushButton()
        self.wrench_button.setIcon(QIcon(self._wrench_path))
        self.wrench_button.setToolTip("Set Stacking Directory & Sigma Clipping")
        self.wrench_button.clicked.connect(self.open_stacking_settings)
        self.wrench_button.setStyleSheet("""
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

        header_row = QHBoxLayout()
        header_row.addWidget(self.wrench_button)

        self.stacking_path_display = QLineEdit(self.stacking_directory or "")
        self.stacking_path_display.setReadOnly(True)
        self.stacking_path_display.setPlaceholderText("No stacking folder selected")
        self.stacking_path_display.setFrame(False)
        self.stacking_path_display.setToolTip(self.stacking_directory or "No stacking folder selected")
        header_row.addWidget(self.stacking_path_display, 1)

        layout.addLayout(header_row)

        self.log_btn = QToolButton(self)
        self.log_btn.setText("Open Log")
        self.log_btn.setToolTip("Show the Stacking Suite log window")
        self.log_btn.clicked.connect(self._show_log_window)
        header_row.addWidget(self.log_btn)

        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.restore_saved_master_calibrations()
        self.update_override_dark_combo()  
        self._update_stacking_path_display()

        if self.settings.value("stacking/mfdeconv/after_mf_run_integration", None) is None:
            self.settings.setValue("stacking/mfdeconv/after_mf_run_integration", False)

        # Drizzle UI wiring
        self.drizzle_checkbox.toggled.connect(
            lambda v: (
                self.drizzle_scale_combo.setEnabled(v),
                self.drizzle_drop_shrink_spin.setEnabled(v),
                self.settings.setValue("stacking/drizzle_enabled", bool(v))
            )
        )
        drizzle_on = self.settings.value("stacking/drizzle_enabled", False, type=bool)
        self.drizzle_scale_combo.setEnabled(drizzle_on)
        self.drizzle_drop_shrink_spin.setEnabled(drizzle_on)

        # Comet wiring
        self.comet_cb.toggled.connect(self._on_comet_toggled_public)
        self._on_comet_toggled_public(self.comet_cb.isChecked())
        self.comet_blend_cb.toggled.connect(lambda v: self.settings.setValue("stacking/comet/blend", bool(v)))
        self.comet_mix.valueChanged.connect(lambda v: self.settings.setValue("stacking/comet/mix", float(v)))

        # Multi-frame wiring
        self.mf_enabled_cb.toggled.connect(self._on_mf_toggled_public)
        self._on_mf_toggled_public(self.mf_enabled_cb.isChecked())

        # Mutually exclusive modes
        self.drizzle_checkbox.toggled.connect(lambda v: v and self._apply_mode_enforcement(self.drizzle_checkbox))
        self.comet_cb.toggled.connect(        lambda v: v and self._apply_mode_enforcement(self.comet_cb))
        self.mf_enabled_cb.toggled.connect(   lambda v: v and self._apply_mode_enforcement(self.mf_enabled_cb))
        self.trail_cb.toggled.connect(        lambda v: v and self._apply_mode_enforcement(self.trail_cb))

        for cb in (self.trail_cb, self.comet_cb, self.mf_enabled_cb, self.drizzle_checkbox):
            if cb.isChecked():
                self._apply_mode_enforcement(cb)
                break

        self.use_gpu_integration = self.settings.value("stacking/use_hardware_accel", True, type=bool)
        self._migrate_drizzle_keys_once()

    def ingest_paths_from_blink(self, paths: list[str], target: str):
        """
        Called by the main window when Blink wants to send files over.
        target = "lights"       → Light tab
        target = "integration"  → Image Registration / Integration tab
        """
        if not paths:
            return

        if target == "lights":
            self._ingest_paths_with_progress(
                paths=paths,
                tree=self.light_tree,
                expected_type="LIGHT",
                title="Adding Blink files to Light tab…"
            )
            # same behavior as normal add
            self.assign_best_master_files()
            self.tabs.setCurrentWidget(self.light_tab)
            return

        if target == "integration":
            # treat them as calibrated lights you want to integrate
            self._ingest_paths_with_progress(
                paths=paths,
                tree=self.reg_tree,
                expected_type="LIGHT",
                title="Adding Blink files to Image Integration…"
            )
            self._refresh_reg_tree_summaries()
            self.tabs.setCurrentWidget(self.image_integration_tab)
            return


    def _migrate_drizzle_keys_once(self):
        s = self.settings
        if s.value("stacking/drizzle_pixfrac", None) is None:
            # take whichever existed, prefer Integration tab value
            v = s.value("stacking/drizzle_drop", None, type=float)
            if v is None:
                v = s.value("stacking/drop_shrink", 0.65, type=float)
            self._set_drizzle_pixfrac(v)


    def _on_align_progress(self, done: int, total: int):
        """
        Queued, debounced progress sink. DO NOT emit any signals from here.
        Only update local state and schedule a single UI update via QTimer.
        """
        # Optional: drop exact duplicates to reduce churn
        tup = (int(done), int(total))
        if tup == self._align_prog_last:
            return
        self._align_prog_last = tup

        self._align_prog_pending = tup

        # If you created a QProgressDialog earlier, make sure its range matches total
        if getattr(self, "align_progress", None) and total > 0:
            if self.align_progress.maximum() != total:
                self.align_progress.setRange(0, total)
                self.align_progress.setAutoClose(True)
                self.align_progress.setAutoReset(True)

        # Coalesce bursts to a single UI update on the event loop
        if not self._align_prog_timer.isActive():
            self._align_prog_timer.start(0)

    def _flush_align_progress(self):
        """
        Coalesced progress update for the align QProgressDialog.
        Called by self._align_prog_timer (singleShot).
        """
        if self._align_prog_in_slot or self._align_prog_pending is None:
            return
        self._align_prog_in_slot = True
        try:
            dlg = getattr(self, "align_progress", None)
            if dlg is not None:
                done, total = self._align_prog_pending
                if total > 0:
                    done = max(0, min(int(done), int(total)))
                    dlg.setRange(0, int(total))
                    dlg.setValue(int(done))
                    dlg.setLabelText(f"Aligning stars… ({int(done)}/{int(total)})")
                else:
                    # unknown total: keep it as a pulsing dialog
                    dlg.setRange(0, 0)
                    dlg.setLabelText("Aligning stars…")
        finally:
            self._align_prog_in_slot = False
            self._align_prog_pending = None



    def _hw_accel_enabled(self) -> bool:
        try:
            return bool(self.settings.value("stacking/use_hardware_accel", True, type=bool))
        except Exception:
            return bool(getattr(self, "use_gpu_integration", True))

    def _set_check_safely(self, cb, on: bool):
        """Set a checkbox without firing its signals (prevents recursion)."""
        old = cb.blockSignals(True)
        try:
            cb.setChecked(on)
        finally:
            cb.blockSignals(old)

    def _apply_mode_enforcement(self, who):
        """
        When one of the 'mode' boxes turns ON, uncheck the others,
        persist settings, and re-apply per-mode enable/disable.
        """
        # 1) Uncheck other modes
        mode_boxes = (self.drizzle_checkbox, self.comet_cb, self.mf_enabled_cb, self.trail_cb)
        for cb in mode_boxes:
            if cb is not who:
                self._set_check_safely(cb, False)

        # 2) Persist current state
        self.settings.setValue("stacking/drizzle_enabled",      self.drizzle_checkbox.isChecked())
        self.settings.setValue("stacking/comet/enabled",        self.comet_cb.isChecked())
        self.settings.setValue("stacking/mfdeconv/enabled",     self.mf_enabled_cb.isChecked())
        self.settings.setValue("stacking/star_trail_enabled",   self.trail_cb.isChecked())

        # 3) Re-apply UI gating so widgets match the new state
        self._on_drizzle_checkbox_toggled(self.drizzle_checkbox.isChecked())
        self._on_star_trail_toggled(self.trail_cb.isChecked())
        self._on_comet_toggled_public(self.comet_cb.isChecked())
        self._on_mf_toggled_public(self.mf_enabled_cb.isChecked())

    def _on_comet_toggled_public(self, v: bool):
        """
        Public comet toggle (replaces the inline closure). Also handles the
        'Stars+Comet blend' interlock + persists setting.
        """
        self.comet_pick_btn.setEnabled(v)
        self.comet_blend_cb.setEnabled(v)
        self.comet_mix.setEnabled(v)
        self.settings.setValue("stacking/comet/enabled", bool(v))

    def _on_mf_toggled_public(self, v: bool):
        widgets = [
            getattr(self, "mf_iters_spin", None),
            getattr(self, "mf_min_iters_spin", None),
            getattr(self, "mf_kappa_spin", None),
            getattr(self, "mf_color_combo", None),
            getattr(self, "mf_rho_combo", None),
            getattr(self, "mf_Huber_spin", None),
            getattr(self, "mf_Huber_hint", None),
            getattr(self, "mf_use_star_mask_cb", None),
            getattr(self, "mf_use_noise_map_cb", None),
            getattr(self, "mf_save_intermediate_cb", None),
            getattr(self, "mf_sr_cb", None),  # NEW
        ]
        for w in widgets:
            if w is not None:
                w.setEnabled(v)
        self.settings.setValue("stacking/mfdeconv/enabled", bool(v))

    def _elide_chars(self, s: str, max_chars: int) -> str:
        if not s:
            return ""
        s = s.replace("\n", " ")
        if len(s) <= max_chars:
            return s
        # reserve one char for the ellipsis
        return s[:max_chars - 1] + "…"

    def _dtype(self):
        return self.internal_dtype

    def _dtype_name(self):
        return np.dtype(self.internal_dtype).name

    def _set_last_status(self, message: str):
        disp = self._elide_chars(message, getattr(self, "_last_status_max_chars", 50))
        self._last_status_label.setText(disp)
        # keep full message available on hover
        self._last_status_label.setToolTip(message)


    def _ensure_log_visible_once(self):
        dock = _get_log_dock()
        if dock:
            dock.setVisible(True)
            dock.raise_()

    def _show_log_window(self):
        dock = _get_log_dock()
        if dock:
            dock.setVisible(True)
            dock.raise_()
        else:
            QMessageBox.information(
                self, "Stacking Log",
                "Open the main window to see the Stacking Log dock."
            )

    def _label_with_dims(self, label: str, width: int, height: int) -> str:
        """Replace or append (WxH) in a human label."""
        clean = _DIM_RE.sub("", label).rstrip()
        return f"{clean} ({width}x{height})"

    def _update_stacking_path_display(self):
        txt = self.stacking_directory or ""
        self.stacking_path_display.setText(txt)
        self.stacking_path_display.setToolTip(txt or "No stacking folder selected")

    def restore_saved_master_calibrations(self):
        saved_darks = self.settings.value("stacking/master_darks", [], type=list)
        saved_flats = self.settings.value("stacking/master_flats", [], type=list)

        if saved_darks:
            self.add_master_files(self.master_dark_tree, "DARK", saved_darks)

        if saved_flats:
            self.add_master_files(self.master_flat_tree, "FLAT", saved_flats)

    def setup_status_bar(self, layout):
        """ Sets up a scrollable status log at the bottom of the UI. """
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.status_text.setStyleSheet(
            "background-color: black; color: white; font-family: Monospace; padding: 4px;"
        )

        self.status_scroll = QScrollArea()
        self.status_scroll.setWidgetResizable(True)
        self.status_scroll.setWidget(self.status_text)
        # Make the scroll area respect a fixed height
        self.status_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.status_scroll)

        # show ~10 lines
        self.set_status_visible_lines(6)

    def set_status_visible_lines(self, n_lines: int):
        fm = QFontMetrics(self.status_text.font())
        line_h = fm.lineSpacing()

        # Add margins/frames (a small fudge keeps things from clipping)
        frame = self.status_text.frameWidth()
        docm  = int(self.status_text.document().documentMargin())
        extra = 2 * frame + 2 * docm + 8

        self.status_scroll.setFixedHeight(int(n_lines * line_h + extra))

    @pyqtSlot(str)
    def _update_status_gui(self, message: str):
        # Update in-dialog status label if you have one
        if hasattr(self, "_last_status_label") and self._last_status_label:
            self._last_status_label.setText(message)

        # Optional: your own helper for status breadcrumb/history
        if hasattr(self, "_set_last_status"):
            try:
                self._set_last_status(message)
            except Exception:
                pass


    def update_status(self, message: str):
        """
        Thread-safe status update:
        • If already on GUI thread -> update immediately + push to log bus directly
        • If from worker thread    -> emit once; slots are queued to GUI + log
        """
        if QThread.currentThread() is self._gui_thread:
            # Immediate UI update
            self._update_status_gui(message)
            # Send to log window directly (don’t signal back into ourselves)
            try:
                self._log_bus.posted.emit(message)
            except Exception:
                pass
        else:
            # One queued signal fan-out to GUI + log
            self.status_signal.emit(message)


    @pyqtSlot(str)
    def _on_post_status(self, msg: str):
        # 1) your central logger
        self.update_status(msg)
        # 2) also reflect in the progress dialog label if it exists
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.setLabelText(msg)
                QApplication.processEvents()
        except Exception:
            pass


    def _norm_dir(self, p: str) -> str:
        if not p:
            return ""
        p = os.path.expanduser(os.path.expandvars(p))
        p = os.path.abspath(p)
        p = os.path.normpath(p)
        if os.name == "nt":
            p = p.lower()
        return p

    def _choose_dir_into(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Stacking Directory",
                                            line_edit.text() or self.stacking_directory or "")
        if d:
            line_edit.setText(d)

    # --- Stacking profile helpers ---------------------------------------------

    def _stacking_profile_names(self) -> list[str]:
        """
        Return list of saved stacking profile names.
        Stored as a QStringList / Python list under 'stacking/profiles/names'.
        """
        val = self.settings.value("stacking/profiles/names", [], type=list)
        if isinstance(val, (tuple, set)):
            val = list(val)
        if isinstance(val, str):
            val = [val] if val else []
        return [str(x) for x in val]

    def _set_stacking_profile_names(self, names: list[str]) -> None:
        names = [str(n) for n in names]
        self.settings.setValue("stacking/profiles/names", names)
        self.settings.sync()

    def _stacking_snapshot_keys(self) -> list[str]:
        """
        All QSettings keys that define a stacking configuration.
        We copy these into / out of profiles.

        We explicitly ignore any existing profile storage to avoid recursion.
        If you *don’t* want the directory to be per-profile, skip 'stacking/dir'.
        """
        keys = []
        for k in self.settings.allKeys():
            if not k.startswith("stacking/"):
                continue
            if k.startswith("stacking/profiles/"):
                continue
            # If you want the stacking directory to be global, uncomment this:
            # if k == "stacking/dir":
            #     continue
            keys.append(k)
        return keys

    def _save_current_stacking_to_profile(self, profile_name: str) -> None:
        """
        Snapshot all current 'stacking/*' keys into:
        'stacking/profiles/<profile_name>/*'
        This assumes those keys already reflect the current config
        (i.e. user has hit OK at least once).
        """
        profile_name = str(profile_name).strip()
        if not profile_name:
            return

        base_prefix    = "stacking/"
        profile_prefix = f"stacking/profiles/{profile_name}/"

        for key in self._stacking_snapshot_keys():
            subkey = key[len(base_prefix):]  # e.g. align/model
            val    = self.settings.value(key)
            self.settings.setValue(profile_prefix + subkey, val)

        # Track profile list + active profile
        names = self._stacking_profile_names()
        if profile_name not in names:
            names.append(profile_name)
            self._set_stacking_profile_names(names)

        self.settings.setValue("stacking/active_profile", profile_name)
        self.settings.sync()

    def _load_profile_into_settings(self, profile_name: str) -> None:
        """
        Copy keys from 'stacking/profiles/<profile_name>/*' back into 'stacking/*'.
        Does NOT touch UI widgets directly; caller can restart / reopen dialog.
        """
        profile_name = str(profile_name).strip()
        if not profile_name:
            return

        base_prefix    = "stacking/"
        profile_prefix = f"stacking/profiles/{profile_name}/"

        for key in self.settings.allKeys():
            if not key.startswith(profile_prefix):
                continue
            subkey   = key[len(profile_prefix):]      # e.g. align/model
            base_key = base_prefix + subkey          # 'stacking/align/model'
            val      = self.settings.value(key)
            self.settings.setValue(base_key, val)

        self.settings.setValue("stacking/active_profile", profile_name)
        self.settings.sync()

    def _delete_stacking_profile(self, profile_name: str) -> None:
        """
        Remove a profile and its keys.
        """
        profile_name = str(profile_name).strip()
        if not profile_name:
            return

        profile_prefix = f"stacking/profiles/{profile_name}/"

        # Remove all keys under this prefix
        to_delete = [k for k in self.settings.allKeys() if k.startswith(profile_prefix)]
        for k in to_delete:
            self.settings.remove(k)

        # Update profile list
        names = self._stacking_profile_names()
        if profile_name in names:
            names.remove(profile_name)
            self._set_stacking_profile_names(names)

        # Clear active profile if we just deleted it
        active = self.settings.value("stacking/active_profile", "", type=str)
        if active == profile_name:
            self.settings.remove("stacking/active_profile")

        self.settings.sync()


    def open_stacking_settings(self):
        """Opens a 2-column Stacking Settings dialog."""
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QFormLayout,
            QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
            QCheckBox, QDialogButtonBox, QScrollArea, QWidget, QInputDialog
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Stacking Settings")

        # Top-level layout
        root = QVBoxLayout(dialog)

        # === Profiles row (at the very top) ===
        gb_profiles = QGroupBox("Profiles")
        prof_layout = QHBoxLayout(gb_profiles)

        prof_label = QLabel("Profile:")
        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(False)

        # Populate combo from QSettings
        profile_names = self._stacking_profile_names()
        for name in profile_names:
            self.profile_combo.addItem(name)

        # Optional: select last active profile, if any
        active_profile = self.settings.value("stacking/active_profile", "", type=str)
        if active_profile:
            idx = self.profile_combo.findText(active_profile)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)

        btn_new    = QPushButton("New…")
        btn_save   = QPushButton("Save")
        btn_load   = QPushButton("Load")
        btn_delete = QPushButton("Delete")

        prof_layout.addWidget(prof_label)
        prof_layout.addWidget(self.profile_combo, 1)
        prof_layout.addWidget(btn_new)
        prof_layout.addWidget(btn_save)
        prof_layout.addWidget(btn_load)
        prof_layout.addWidget(btn_delete)

        root.addWidget(gb_profiles)

        # Now your scroll area, as before
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)

        def _refresh_profile_combo():
            self.profile_combo.blockSignals(True)
            self.profile_combo.clear()
            for name in self._stacking_profile_names():
                self.profile_combo.addItem(name)
            active = self.settings.value("stacking/active_profile", "", type=str)
            if active:
                i = self.profile_combo.findText(active)
                if i >= 0:
                    self.profile_combo.setCurrentIndex(i)
            self.profile_combo.blockSignals(False)

        def _ask_profile_name(title: str, default: str = "") -> str | None:
            text, ok = QInputDialog.getText(dialog, title, "Profile name:", text=default)
            if not ok:
                return None
            name = text.strip()
            return name or None

        def _on_new_profile():
            name = _ask_profile_name("New stacking profile")
            if not name:
                return
            # For now: profile = snapshot of current applied settings.
            # (User can tweak, click OK to apply, reopen, then New to save.)
            self._save_current_stacking_to_profile(name)
            _refresh_profile_combo()

        def _on_save_profile():
            """Overwrite the selected profile with the CURRENT applied settings."""
            name = self.profile_combo.currentText().strip()
            if not name:
                # If none exists yet, ask for one.
                name = _ask_profile_name("Save profile as…")
                if not name:
                    return
            self._save_current_stacking_to_profile(name)
            _refresh_profile_combo()
            QMessageBox.information(dialog, "Profile saved",
                                    f"Current stacking settings saved to profile '{name}'.")

        def _on_load_profile():
            name = self.profile_combo.currentText().strip()
            if not name:
                QMessageBox.warning(dialog, "No profile selected",
                                    "Please choose a profile to load.")
                return

            # Confirm, since this will overwrite current stacking settings.
            msg = (f"Load profile '{name}'?\n\n"
                "This will overwrite the current stacking configuration "
                "and restart the Stacking Suite.")
            if QMessageBox.question(dialog, "Load profile", msg,
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                    ) != QMessageBox.StandardButton.Yes:
                return

            # Copy profile → stacking/* and restart
            self._load_profile_into_settings(name)
            dialog.accept()
            self.update_status(f"📂 Loaded stacking profile: {name}")
            self._restart_self()

        def _on_delete_profile():
            name = self.profile_combo.currentText().strip()
            if not name:
                return
            if QMessageBox.question(
                dialog,
                "Delete profile",
                f"Delete stacking profile '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return
            self._delete_stacking_profile(name)
            _refresh_profile_combo()

        btn_new.clicked.connect(_on_new_profile)
        btn_save.clicked.connect(_on_save_profile)
        btn_load.clicked.connect(_on_load_profile)
        btn_delete.clicked.connect(_on_delete_profile)


        body = QWidget()
        scroll.setWidget(body)

        cols = QHBoxLayout(body)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        cols.addLayout(left_col, 1)
        cols.addSpacing(12)
        cols.addLayout(right_col, 1)

        # ========== LEFT COLUMN ==========
        # --- General ---
        gb_general = QGroupBox("General")
        fl_general = QFormLayout(gb_general)

        # Stacking directory
        dir_row = QHBoxLayout()
        dir_edit = QLineEdit(self.stacking_directory or "")
        dialog._dir_edit = dir_edit
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(lambda: self._choose_dir_into(dir_edit))
        dir_row.addWidget(dir_edit, 1)
        dir_row.addWidget(btn_browse)
        fl_general.addRow(QLabel("Stacking Directory:"), QWidget())
        fl_general.addRow(dir_row)

        # Precision
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["32-bit float", "64-bit float"])

        # Restore from current internal dtype / QSettings instead of hard-coding 32-bit
        saved_dtype = (self.settings.value("stacking/internal_dtype", "float32") or "float32").lower()

        if "64" in saved_dtype:
            idx = 1
        elif getattr(self, "internal_dtype", None) is np.float64:
            # fallback if settings missing but runtime already using 64-bit
            idx = 1
        else:
            idx = 0

        self.precision_combo.setCurrentIndex(idx)
        self.precision_combo.setToolTip("64-bit uses ~2× RAM; 32-bit is faster/lighter.")
        fl_general.addRow("Internal Precision:", self.precision_combo)

        # Chunk sizes
        self.chunkHeightSpinBox = QSpinBox()
        self.chunkHeightSpinBox.setRange(128, 8192)
        self.chunkHeightSpinBox.setValue(self.settings.value("stacking/chunk_height", 512, type=int))
        self.chunkWidthSpinBox = QSpinBox()
        self.chunkWidthSpinBox.setRange(128, 8192)
        self.chunkWidthSpinBox.setValue(self.settings.value("stacking/chunk_width", 512, type=int))
        hw_row = QHBoxLayout()
        hw_row.addWidget(QLabel("H:")); hw_row.addWidget(self.chunkHeightSpinBox)
        hw_row.addSpacing(8)
        hw_row.addWidget(QLabel("W:")); hw_row.addWidget(self.chunkWidthSpinBox)
        w_hw = QWidget(); w_hw.setLayout(hw_row)
        fl_general.addRow("Chunk Size:", w_hw)

        left_col.addWidget(gb_general)

        # --- Distortion / Transform model ---
        # --- Distortion / Transform model ---
        disto_box  = QGroupBox("Distortion / Transform")
        disto_form = QFormLayout(disto_box)

        self.align_model_combo = QComboBox()
        # Order matters for index mapping below
        self.align_model_combo.addItems([
            "Affine (fast)",
            "No Distortion (rotate/translate/scale)",
            "Homography (projective)",
            "Polynomial 3rd-order",
            "Polynomial 4th-order",
        ])

        # Map saved string -> index
        _saved_model = (self.settings.value("stacking/align/model", "affine") or "affine").lower()
        _model_to_idx = {
            "affine": 0,
            "similarity": 1,
            "no_distortion": 1,
            "nodistortion": 1,
            "homography": 2,
            "poly3": 3,
            "poly4": 4,
        }
        self.align_model_combo.setCurrentIndex(_model_to_idx.get(_saved_model, 0))
        disto_form.addRow("Model:", self.align_model_combo)

        # Shared
        self.align_max_cp = QSpinBox()
        self.align_max_cp.setRange(20, 2000)
        self.align_max_cp.setValue(self.settings.value("stacking/align/max_cp", 250, type=int))
        disto_form.addRow("Max control points:", self.align_max_cp)

        self.align_downsample = QSpinBox()
        self.align_downsample.setRange(1, 8)
        self.align_downsample.setValue(self.settings.value("stacking/align/downsample", 2, type=int))
        disto_form.addRow("Solve downsample:", self.align_downsample)

        # Homography / Similarity-specific RANSAC reprojection threshold
        self.h_ransac_reproj = QDoubleSpinBox()
        self.h_ransac_reproj.setRange(0.1, 10.0)
        self.h_ransac_reproj.setDecimals(2)
        self.h_ransac_reproj.setSingleStep(0.1)
        self.h_ransac_reproj.setValue(self.settings.value("stacking/align/h_reproj", 3.0, type=float))
        self._h_label = QLabel("RANSAC reproj (px):")
        disto_form.addRow(self._h_label, self.h_ransac_reproj)

        def _toggle_disto_rows():
            m = self.align_model_combo.currentIndex()
            is_sim = (m == 1)   # similarity / no distortion
            is_h   = (m == 2)   # homography
            # enable RANSAC threshold for homography and similarity
            enable_ransac = is_h or is_sim
            self._h_label.setEnabled(enable_ransac)
            self.h_ransac_reproj.setEnabled(enable_ransac)

        _toggle_disto_rows()
        self.align_model_combo.currentIndexChanged.connect(lambda _: _toggle_disto_rows())

        left_col.addWidget(disto_box)


        # --- Alignment ---
        gb_align = QGroupBox("Alignment")
        fl_align = QFormLayout(gb_align)

        self.align_passes_combo = QComboBox()
        self.align_passes_combo.addItems(["Fast (1 pass)", "Accurate (3 passes)"])
        curr_passes = self.settings.value("stacking/refinement_passes", 1, type=int)
        self.align_passes_combo.setCurrentIndex(0 if curr_passes <= 1 else 1)
        self.align_passes_combo.setToolTip("Fast = single pass; Accurate = 3-pass refinement.")
        fl_align.addRow("Refinement:", self.align_passes_combo)

        self.shift_tol_spin = QDoubleSpinBox()
        self.shift_tol_spin.setRange(0.05, 5.0)
        self.shift_tol_spin.setDecimals(2)
        self.shift_tol_spin.setSingleStep(0.05)
        self.shift_tol_spin.setValue(self.settings.value("stacking/shift_tolerance", 0.2, type=float))
        fl_align.addRow("Accept tolerance (px):", self.shift_tol_spin)

        self.accept_shift_spin = QDoubleSpinBox()
        self.accept_shift_spin.setRange(0.0, 50.0)
        self.accept_shift_spin.setDecimals(2)
        self.accept_shift_spin.setSingleStep(0.1)
        self.accept_shift_spin.setToolTip("Reject a frame if its residual shift exceeds this many pixels after alignment.")
        self.accept_shift_spin.setValue(
            self.settings.value("stacking/accept_shift_px", 2.0, type=float)
        )
        fl_align.addRow("Accept max shift (px):", self.accept_shift_spin)

        # Star detection sigma (used by astroalign / your detector)
        self.align_det_sigma = QDoubleSpinBox()
        self.align_det_sigma.setRange(0.5, 50.0)
        self.align_det_sigma.setDecimals(1)
        self.align_det_sigma.setSingleStep(0.5)
        self.align_det_sigma.setValue(
            self.settings.value("stacking/align/det_sigma", 20.0, type=float)
        )
        self.align_det_sigma.setToolTip(
            "Star detection threshold in σ above background. "
            "Lower = more stars (faster saturation, more false positives); higher = fewer stars."
        )
        fl_align.addRow("Star detect σ:", self.align_det_sigma)

        # (Optional) Minimum star area in pixels
        self.align_minarea = QSpinBox()
        self.align_minarea.setRange(1, 200)
        self.align_minarea.setSingleStep(1)
        self.align_minarea.setValue(
            self.settings.value("stacking/align/minarea", 10, type=int)
        )
        self.align_minarea.setToolTip(
            "Minimum connected-pixel area to keep a detection as a star (px). Helps reject hot pixels/noise."
        )
        fl_align.addRow("Min star area (px):", self.align_minarea)

        # NEW: Max stars (Astroalign control points cap)
        self.align_limit_stars_spin = QSpinBox()
        self.align_limit_stars_spin.setRange(50, 5000)
        self.align_limit_stars_spin.setSingleStep(50)
        self.align_limit_stars_spin.setValue(
            self.settings.value("stacking/align/limit_stars", 100, type=int)
        )
        self.align_limit_stars_spin.setToolTip(
            "Caps Astroalign max_control_points (typical 500–1500). Lower = faster, higher = more robust."
        )
        fl_align.addRow("Max stars:", self.align_limit_stars_spin)

        # NEW: Timeout per frame (seconds)
        self.align_timeout_spin = QSpinBox()
        self.align_timeout_spin.setRange(10, 3600)
        self.align_timeout_spin.setSingleStep(10)
        self.align_timeout_spin.setValue(
            self.settings.value("stacking/align/timeout_per_job_sec", 300, type=int)
        )
        self.align_timeout_spin.setToolTip(
            "Per-frame alignment timeout for the parallel workers. Default 300s."
        )
        fl_align.addRow("Timeout per frame (s):", self.align_timeout_spin)

        left_col.addWidget(gb_align)

        # --- Performance ---
        gb_perf = QGroupBox("Performance")
        fl_perf = QFormLayout(gb_perf)

        self.hw_accel_cb = QCheckBox("Use hardware acceleration if available")
        self.hw_accel_cb.setToolTip("Enable GPU/MPS via PyTorch when supported; falls back to CPU automatically.")
        self.hw_accel_cb.setChecked(self.settings.value("stacking/use_hardware_accel", True, type=bool))
        fl_perf.addRow(self.hw_accel_cb)

        # NEW: MFDeconv engine choice (radio buttons)
        eng_box = QGroupBox("MFDeconv Engine")
        eng_row = QHBoxLayout(eng_box)
        self.mf_eng_normal_rb = QRadioButton("Normal")
        self.mf_eng_cudnn_rb  = QRadioButton("Normal (cuDNN-free)")
        self.mf_eng_sport_rb  = QRadioButton("High-Octane (Let ’er rip)")

        # restore from settings (default "normal")
        _saved_eng = (self.settings.value("stacking/mfdeconv/engine", "normal", type=str) or "normal").lower()
        if   _saved_eng == "cudnn":
            self.mf_eng_cudnn_rb.setChecked(True)
            # If user previously chose cuDNN-free, force HW accel off
            self.hw_accel_cb.setChecked(False)
        elif _saved_eng == "sport":
            self.mf_eng_sport_rb.setChecked(True)
        else:
            self.mf_eng_normal_rb.setChecked(True)

        eng_row.addWidget(self.mf_eng_normal_rb)
        eng_row.addWidget(self.mf_eng_cudnn_rb)
        eng_row.addWidget(self.mf_eng_sport_rb)
        fl_perf.addRow(eng_box)

        # When user selects "Normal (cuDNN-free)", automatically turn off HW accel
        def _on_mfdeconv_engine_changed(checked: bool):
            if checked and self.mf_eng_cudnn_rb.isChecked():
                self.hw_accel_cb.setChecked(False)

        self.mf_eng_cudnn_rb.toggled.connect(_on_mfdeconv_engine_changed)

        # (Optional) show detected backend for user feedback
        try:
            backend_str = current_backend() or "CPU only"
        except Exception:
            backend_str = "CPU only"
        fl_perf.addRow("Detected backend:", QLabel(backend_str))

        left_col.addWidget(gb_perf)


        # ========== RIGHT COLUMN ==========
        # --- Normalization & Gradient (ABE poly2) ---
        gb_normgrad = QGroupBox("Normalization & Gradient (ABE Poly²)")
        fl_ng = QFormLayout(gb_normgrad)

        # master enable
        self.chk_poly2 = QCheckBox("Remove background gradient (ABE Poly²)")
        self.chk_poly2.setChecked(self.settings.value("stacking/grad_poly2/enabled", False, type=bool))
        fl_ng.addRow(self.chk_poly2)

        # mode (subtract vs divide)
        self.grad_mode_combo = QComboBox()
        self.grad_mode_combo.addItems(["Subtract (additive)", "Divide (flat-like)"])
        _saved_mode = self.settings.value("stacking/grad_poly2/mode", "subtract")
        self.grad_mode_combo.setCurrentIndex(0 if _saved_mode.lower() != "divide" else 1)
        fl_ng.addRow("Mode:", self.grad_mode_combo)

        # ABE-style controls
        self.grad_samples_spin = QSpinBox()
        self.grad_samples_spin.setRange(20, 600)
        self.grad_samples_spin.setValue(self.settings.value("stacking/grad_poly2/samples", 120, type=int))
        fl_ng.addRow("Sample points:", self.grad_samples_spin)

        self.grad_downsample_spin = QSpinBox()
        self.grad_downsample_spin.setRange(1, 16)
        self.grad_downsample_spin.setValue(self.settings.value("stacking/grad_poly2/downsample", 6, type=int))
        fl_ng.addRow("Downsample (AREA):", self.grad_downsample_spin)

        self.grad_patch_spin = QSpinBox()
        self.grad_patch_spin.setRange(5, 51)
        self.grad_patch_spin.setSingleStep(2)
        self.grad_patch_spin.setValue(self.settings.value("stacking/grad_poly2/patch_size", 15, type=int))
        fl_ng.addRow("Patch size (small):", self.grad_patch_spin)

        self.grad_min_strength = QDoubleSpinBox()
        self.grad_min_strength.setRange(0.0, 0.20)
        self.grad_min_strength.setDecimals(3)
        self.grad_min_strength.setSingleStep(0.005)
        self.grad_min_strength.setValue(self.settings.value("stacking/grad_poly2/min_strength", 0.01, type=float))
        fl_ng.addRow("Skip if strength <", self.grad_min_strength)

        # division-only gain clip
        self.grad_gain_lo = QDoubleSpinBox()
        self.grad_gain_lo.setRange(0.01, 1.00); self.grad_gain_lo.setDecimals(2); self.grad_gain_lo.setSingleStep(0.01)
        self.grad_gain_lo.setValue(self.settings.value("stacking/grad_poly2/gain_lo", 0.20, type=float))
        self.grad_gain_hi = QDoubleSpinBox()
        self.grad_gain_hi.setRange(1.0, 25.0); self.grad_gain_hi.setDecimals(1); self.grad_gain_hi.setSingleStep(0.5)
        self.grad_gain_hi.setValue(self.settings.value("stacking/grad_poly2/gain_hi", 5.0, type=float))

        row_gain = QWidget()
        row_gain_h = QHBoxLayout(row_gain); row_gain_h.setContentsMargins(0,0,0,0)
        row_gain_h.addWidget(QLabel("Clip (lo/hi):"))
        row_gain_h.addWidget(self.grad_gain_lo)
        row_gain_h.addWidget(QLabel(" / "))
        row_gain_h.addWidget(self.grad_gain_hi)
        fl_ng.addRow("Divide gain limits:", row_gain)

        # enable/disable
        def _toggle_grad_enabled(on: bool):
            for w in (self.grad_mode_combo, self.grad_samples_spin, self.grad_downsample_spin,
                    self.grad_patch_spin, self.grad_min_strength, row_gain):
                w.setEnabled(on)

        def _toggle_gain_row():
            is_div = (self.grad_mode_combo.currentIndex() == 1)
            row_gain.setVisible(is_div)
            row_gain.setEnabled(self.chk_poly2.isChecked() and is_div)

        self.chk_poly2.toggled.connect(_toggle_grad_enabled)
        self.grad_mode_combo.currentIndexChanged.connect(lambda _: _toggle_gain_row())

        _toggle_grad_enabled(self.chk_poly2.isChecked())
        _toggle_gain_row()

        left_col.addWidget(gb_normgrad)

        gb_drizzle = QGroupBox("Drizzle")
        fl_dz = QFormLayout(gb_drizzle)

        self.drizzle_kernel_combo = QComboBox()
        self.drizzle_kernel_combo.addItems(["Square (pixfrac)", "Circular (disk)", "Gaussian"])
        # restore
        _saved_k = self.settings.value("stacking/drizzle_kernel", "square").lower()
        if _saved_k.startswith("gauss"): self.drizzle_kernel_combo.setCurrentIndex(2)
        elif _saved_k.startswith("circ"): self.drizzle_kernel_combo.setCurrentIndex(1)
        else: self.drizzle_kernel_combo.setCurrentIndex(0)
        fl_dz.addRow("Kernel:", self.drizzle_kernel_combo)

        self.drop_shrink_spin = QDoubleSpinBox()
        self.drop_shrink_spin.setRange(0.0, 1.0)  # make this the same concept: pixfrac
        self.drop_shrink_spin.setDecimals(3)
        self.drop_shrink_spin.setValue(self._get_drizzle_pixfrac())
        self.drop_shrink_spin.valueChanged.connect(lambda v: self._set_drizzle_pixfrac(v))
        fl_dz.addRow("Kernel width:", self.drop_shrink_spin)

        # Optional: a separate σ for Gaussian (if you want it distinct)
        self.gauss_sigma_spin = QDoubleSpinBox()
        self.gauss_sigma_spin.setRange(0.05, 3.0)
        self.gauss_sigma_spin.setDecimals(3)
        self.gauss_sigma_spin.setSingleStep(0.05)
        self.gauss_sigma_spin.setValue(self.settings.value("stacking/drizzle_gauss_sigma",
                                                        self.drop_shrink_spin.value()*0.5, type=float))
        fl_dz.addRow("Gaussian σ (px):", self.gauss_sigma_spin)

        def _toggle_gauss_sigma():
            self.gauss_sigma_spin.setEnabled(self.drizzle_kernel_combo.currentIndex()==2)
        _toggle_gauss_sigma()
        self.drizzle_kernel_combo.currentIndexChanged.connect(lambda _ : _toggle_gauss_sigma())

        right_col.addWidget(gb_drizzle)

        # --- MF Deconvolution  ---
        gb_mf = QGroupBox("Multi-frame Deconvolution")
        fl_mf = QFormLayout(gb_mf)
        def _row(lbl, w):
            c = QWidget(); h = QHBoxLayout(c); h.setContentsMargins(0,0,0,0); h.addWidget(w, 1); return (lbl, c)
        
        self.mf_seed_combo = QComboBox()
        self.mf_seed_combo.addItems([
            "Robust μ–σ (live stack)",
            "Median (Sukhdeep et al.)"
        ])
        # Persisted value → UI
        seed_mode_saved = str(self.settings.value("stacking/mfdeconv/seed_mode", "robust"))
        seed_idx = 0 if seed_mode_saved.lower() != "median" else 1
        self.mf_seed_combo.setCurrentIndex(seed_idx)
        self.mf_seed_combo.setToolTip(
            "Choose the initial seed image for MFDeconv:\n"
            "• Robust μ–σ: running mean with sigma clipping (RAM-friendly, default)\n"
            "• Median: tiled median stack (more outlier-resistant; heavier I/O, esp. for XISF)"
        )
        fl_mf.addRow(*_row("Seed image:", self.mf_seed_combo))

        self.sm_thresh = QDoubleSpinBox(); self.sm_thresh.setRange(0.1, 20.0); self.sm_thresh.setDecimals(2)
        self.sm_thresh.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/thresh_sigma", _SM_DEF_THRESH, type=float)
        )
        fl_mf.addRow(*_row("Star detect σ:", self.sm_thresh))

        self.sm_grow = QSpinBox(); self.sm_grow.setRange(0, 128)
        self.sm_grow.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/grow_px", _SM_DEF_GROW, type=int)
        )
        fl_mf.addRow(*_row("Dilate (+px):", self.sm_grow))

        self.sm_soft = QDoubleSpinBox(); self.sm_soft.setRange(0.0, 10.0); self.sm_soft.setDecimals(2)
        self.sm_soft.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/soft_sigma", _SM_DEF_SOFT, type=float)
        )
        fl_mf.addRow(*_row("Feather σ (px):", self.sm_soft))

        self.sm_rmax = QSpinBox(); self.sm_rmax.setRange(2, 256)
        self.sm_rmax.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/max_radius_px", _SM_DEF_RMAX, type=int)
        )
        fl_mf.addRow(*_row("Max star radius (px):", self.sm_rmax))

        self.sm_maxobjs = QSpinBox(); self.sm_maxobjs.setRange(10, 50000)
        self.sm_maxobjs.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/max_objs", _SM_DEF_MAXOBJS, type=int)
        )
        fl_mf.addRow(*_row("Max stars kept:", self.sm_maxobjs))

        self.sm_keepfloor = QDoubleSpinBox(); self.sm_keepfloor.setRange(0.0, 0.95); self.sm_keepfloor.setDecimals(3)
        self.sm_keepfloor.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/keep_floor", _SM_DEF_KEEPF, type=float)
        )
        self.sm_keepfloor.setToolTip("Lower = stronger masking near stars; 0 = hard mask, 0.2 = gentle.")
        fl_mf.addRow(*_row("Keep-floor:", self.sm_keepfloor))

        # (optional) expose ellipse scale if you like:
        self.sm_es = QDoubleSpinBox(); self.sm_es.setRange(0.5, 3.0); self.sm_es.setDecimals(2)
        self.sm_es.setValue(
            self.settings.value("stacking/mfdeconv/star_mask/ellipse_scale", _SM_DEF_ES, type=float)
        )
        fl_mf.addRow(*_row("Ellipse scale:", self.sm_es))

        # --- Variance map tuning ---
        self.vm_stride = QSpinBox(); self.vm_stride.setRange(1, 64)
        self.vm_stride.setValue(
            self.settings.value("stacking/mfdeconv/varmap/sample_stride", _VM_DEF_STRIDE, type=int)
        )
        fl_mf.addRow(*_row("VarMap sample stride:", self.vm_stride))

        self.vm_sigma = QDoubleSpinBox(); self.vm_sigma.setRange(0.0, 5.0); self.vm_sigma.setDecimals(2)
        self.vm_sigma.setValue(self.settings.value("stacking/mfdeconv/varmap/smooth_sigma", 1.0, type=float))
        fl_mf.addRow(*_row("VarMap smooth σ:", self.vm_sigma))

        self.vm_floor_log = QDoubleSpinBox()
        self.vm_floor_log.setRange(-12.0, -2.0)
        self.vm_floor_log.setDecimals(2)
        self.vm_floor_log.setSingleStep(0.5)
        self.vm_floor_log.setValue(math.log10(
            self.settings.value("stacking/mfdeconv/varmap/floor", 1e-8, type=float)
        ))
        self.vm_floor_log.setToolTip("log10 of variance floor (DN²). -8 ≡ 1e-8.")
        fl_mf.addRow(*_row("VarMap floor (log10):", self.vm_floor_log))

        btn_mf_reset = QPushButton("Reset MFDeconv to Recommended")
        btn_mf_reset.setToolTip(
            "Restore MFDeconv star mask + variance map tuning to the recommended defaults."
        )

        def _reset_mfdeconv_defaults():
            # Star mask tuning
            self.mf_seed_combo.setCurrentIndex(0)
            self.sm_thresh.setValue(4.5)     # Star detect σ
            self.sm_grow.setValue(6)         # Dilate (+px)
            self.sm_soft.setValue(3.0)       # Feather σ (px)
            self.sm_rmax.setValue(36)        # Max star radius (px)
            self.sm_maxobjs.setValue(5000)   # Max stars kept
            self.sm_keepfloor.setValue(0.015)# Keep-floor
            self.sm_es.setValue(1.12)        # Ellipse scale

            # Variance map tuning
            self.vm_stride.setValue(8)       # VarMap sample stride
            self.vm_sigma.setValue(3.0)      # VarMap smooth σ
            self.vm_floor_log.setValue(-10)  # VarMap floor (log10)

            # (Optional) preload QSettings so Cancel still reverts if user wants.
            # If you prefer to save only on OK, you can omit this block.
            s = self.settings
            s.setValue("stacking/mfdeconv/seed_mode", "robust")
            s.setValue("stacking/mfdeconv/star_mask/thresh_sigma", 4.5)
            s.setValue("stacking/mfdeconv/star_mask/grow_px", 6)
            s.setValue("stacking/mfdeconv/star_mask/soft_sigma", 3.0)
            s.setValue("stacking/mfdeconv/star_mask/max_radius_px", 36)
            s.setValue("stacking/mfdeconv/star_mask/max_objs", 5000)
            s.setValue("stacking/mfdeconv/star_mask/keep_floor", 0.015)
            s.setValue("stacking/mfdeconv/star_mask/ellipse_scale", 1.12)
            s.setValue("stacking/mfdeconv/varmap/sample_stride", 8)
            s.setValue("stacking/mfdeconv/varmap/smooth_sigma", 3.0)
            s.setValue("stacking/mfdeconv/varmap/floor", 10 ** (-10))  # store linear value if you persist floor linearly

        btn_mf_reset.clicked.connect(_reset_mfdeconv_defaults)
        fl_mf.addRow(btn_mf_reset)

        right_col.addWidget(gb_mf)



        # --- Rejection ---
        gb_rej = QGroupBox("Rejection")
        rej_layout = QVBoxLayout(gb_rej)

        # Algorithm choice
        algo_row = QHBoxLayout()
        algo_label = QLabel("Algorithm:")
        self.rejection_algo_combo = QComboBox()
        self.rejection_algo_combo.addItems([
            "Weighted Windsorized Sigma Clipping",
            "Kappa-Sigma Clipping",
            "Simple Average (No Rejection)",
            "Simple Median (No Rejection)",
            "Trimmed Mean",
            "Extreme Studentized Deviate (ESD)",
            "Biweight Estimator",
            "Modified Z-Score Clipping",
            "Max Value"
        ])
        saved_algo = self.settings.value("stacking/rejection_algorithm", "Weighted Windsorized Sigma Clipping")
        idx = self.rejection_algo_combo.findText(saved_algo)
        if idx >= 0:
            self.rejection_algo_combo.setCurrentIndex(idx)
        algo_row.addWidget(algo_label); algo_row.addWidget(self.rejection_algo_combo, 1)
        rej_layout.addLayout(algo_row)

        # Param rows as small containers we can show/hide
        def _mini_row(label_text, widget, help_text=None):
            row = QWidget()
            h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0)
            h.addWidget(QLabel(label_text))
            h.addWidget(widget, 1)
            if help_text:
                btn = QPushButton("?"); btn.setFixedSize(20,20)
                btn.clicked.connect(lambda: QMessageBox.information(self, label_text, help_text))
                h.addWidget(btn)
            return row

        # ── NEW: Sigma thresholds (moved here)
        self.sigma_high_spinbox = QDoubleSpinBox()
        self.sigma_high_spinbox.setRange(0.1, 10.0)
        self.sigma_high_spinbox.setDecimals(2)
        self.sigma_high_spinbox.setSingleStep(0.1)
        self.sigma_high_spinbox.setValue(
            getattr(self, "sigma_high", self.settings.value("stacking/sigma_high", 3.0, type=float))
        )

        self.sigma_low_spinbox = QDoubleSpinBox()
        self.sigma_low_spinbox.setRange(0.1, 10.0)
        self.sigma_low_spinbox.setDecimals(2)
        self.sigma_low_spinbox.setSingleStep(0.1)
        self.sigma_low_spinbox.setValue(
            getattr(self, "sigma_low", self.settings.value("stacking/sigma_low", 2.0, type=float))
        )

        _sigma_pair = QWidget()
        _sigma_h = QHBoxLayout(_sigma_pair); _sigma_h.setContentsMargins(0,0,0,0)
        _sigma_h.addWidget(QLabel("High:")); _sigma_h.addWidget(self.sigma_high_spinbox)
        _sigma_h.addSpacing(8)
        _sigma_h.addWidget(QLabel("Low:"));  _sigma_h.addWidget(self.sigma_low_spinbox)
        row_sigma = _mini_row("Sigma thresholds:", _sigma_pair,
            "High/Low σ used by sigma-based rejection.")

        # Existing param rows
        self.kappa_spinbox = QDoubleSpinBox()
        self.kappa_spinbox.setRange(0.1, 10.0); self.kappa_spinbox.setDecimals(2)
        self.kappa_spinbox.setValue(self.settings.value("stacking/kappa", 2.5, type=float))
        row_kappa = _mini_row("Kappa:", self.kappa_spinbox, "Std-devs from median; higher = more lenient.")

        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(1, 10)
        self.iterations_spinbox.setValue(self.settings.value("stacking/iterations", 3, type=int))
        row_iters = _mini_row("Iterations:", self.iterations_spinbox, "Number of kappa-sigma iterations.")

        self.esd_spinbox = QDoubleSpinBox()
        self.esd_spinbox.setRange(0.1, 10.0); self.esd_spinbox.setDecimals(2)
        self.esd_spinbox.setValue(self.settings.value("stacking/esd_threshold", 3.0, type=float))
        row_esd = _mini_row("ESD threshold:", self.esd_spinbox, "Lower = more aggressive outlier rejection.")

        self.biweight_spinbox = QDoubleSpinBox()
        self.biweight_spinbox.setRange(1.0, 10.0); self.biweight_spinbox.setDecimals(2)
        self.biweight_spinbox.setValue(self.settings.value("stacking/biweight_constant", 6.0, type=float))
        row_bi = _mini_row("Biweight constant:", self.biweight_spinbox, "Controls down-weighting strength.")

        self.trim_spinbox = QDoubleSpinBox()
        self.trim_spinbox.setRange(0.0, 0.5); self.trim_spinbox.setDecimals(2)
        self.trim_spinbox.setValue(self.settings.value("stacking/trim_fraction", 0.1, type=float))
        row_trim = _mini_row("Trim fraction:", self.trim_spinbox, "Fraction trimmed on each end before averaging.")

        self.modz_spinbox = QDoubleSpinBox()
        self.modz_spinbox.setRange(0.1, 10.0); self.modz_spinbox.setDecimals(2)
        self.modz_spinbox.setValue(self.settings.value("stacking/modz_threshold", 3.5, type=float))
        row_modz = _mini_row("Modified Z threshold:", self.modz_spinbox, "Lower = more aggressive (MAD-based).")

        # Add all; visibility managed below
        for w in (row_sigma, row_kappa, row_iters, row_esd, row_bi, row_trim, row_modz):
            rej_layout.addWidget(w)

        # show/hide param rows based on algorithm
        def _update_algo_params():
            algo = self.rejection_algo_combo.currentText()
            rows = {
                "sigma": row_sigma,
                "kappa": row_kappa,
                "iters": row_iters,
                "esd": row_esd,
                "bi": row_bi,
                "trim": row_trim,
                "modz": row_modz
            }
            for w in rows.values():
                w.hide()

            # Sigma-based algos
            if "Kappa-Sigma" in algo:
                row_sigma.show()
                row_kappa.show()
                row_iters.show()
            elif "Windsorized" in algo or "Winsorized" in algo:  # Weighted Winsorized Sigma Clipping
                row_sigma.show()

            # Others
            elif "ESD" in algo:
                row_esd.show()
            elif "Biweight" in algo:
                row_bi.show()
            elif "Trimmed Mean" in algo:
                row_trim.show()
            elif "Modified Z-Score" in algo:
                row_modz.show()
            # Simple Average/Median/Max Value → no params

        self.rejection_algo_combo.currentTextChanged.connect(_update_algo_params)
        _update_algo_params()

        right_col.addWidget(gb_rej)


        # --- Cosmetic Correction (Advanced) ---
        gb_cosm = QGroupBox("Cosmetic Correction (Advanced)")
        fl_cosm = QFormLayout(gb_cosm)

        # Enable/disable advanced controls (purely for UI clarity)
        self.cosm_enable_cb = QCheckBox("Enable advanced cosmetic tuning")
        self.cosm_enable_cb.setChecked(
            self.settings.value("stacking/cosmetic/custom_enable", False, type=bool)
        )
        fl_cosm.addRow(self.cosm_enable_cb)

        def _mk_fspin(minv, maxv, step, decimals, key, default):
            sb = QDoubleSpinBox()
            sb.setRange(minv, maxv)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setValue(self.settings.value(key, default, type=float))
            return sb

        # σ thresholds
        self.cosm_hot_sigma = _mk_fspin(0.1, 20.0, 0.1, 2,
            "stacking/cosmetic/hot_sigma", 5.0)
        self.cosm_cold_sigma = _mk_fspin(0.1, 20.0, 0.1, 2,
            "stacking/cosmetic/cold_sigma", 5.0)

        row_sig = QWidget(); row_sig_h = QHBoxLayout(row_sig); row_sig_h.setContentsMargins(0,0,0,0)
        row_sig_h.addWidget(QLabel("Hot σ:")); row_sig_h.addWidget(self.cosm_hot_sigma)
        row_sig_h.addSpacing(8)
        row_sig_h.addWidget(QLabel("Cold σ:")); row_sig_h.addWidget(self.cosm_cold_sigma)
        fl_cosm.addRow("Sigma thresholds:", row_sig)

        # Star guards (skip replacements if neighbors look like a PSF)
        self.cosm_star_mean_ratio = _mk_fspin(0.05, 0.60, 0.01, 3,
            "stacking/cosmetic/star_mean_ratio", 0.22)
        self.cosm_star_max_ratio  = _mk_fspin(0.10, 0.95, 0.01, 3,
            "stacking/cosmetic/star_max_ratio", 0.55)
        row_star = QWidget(); row_star_h = QHBoxLayout(row_star); row_star_h.setContentsMargins(0,0,0,0)
        row_star_h.addWidget(QLabel("Mean ratio:")); row_star_h.addWidget(self.cosm_star_mean_ratio)
        row_star_h.addSpacing(8)
        row_star_h.addWidget(QLabel("Max ratio:"));  row_star_h.addWidget(self.cosm_star_max_ratio)
        fl_cosm.addRow("Star guards:", row_star)

        # Saturation guard quantile
        self.cosm_sat_quantile = _mk_fspin(0.90, 0.9999, 0.0005, 4,
            "stacking/cosmetic/sat_quantile", 0.9995)
        self.cosm_sat_quantile.setToolTip("Pixels above this image quantile are treated as saturated and never replaced.")
        fl_cosm.addRow("Saturation quantile:", self.cosm_sat_quantile)

        # Small helper to enable/disable rows by master checkbox
        def _toggle_cosm_enabled(on: bool):
            for w in (row_sig, row_star, self.cosm_sat_quantile):
                w.setEnabled(on)

        # Defaults button
        btn_defaults = QPushButton("Restore Recommended")
        def _restore_defaults():
            self.cosm_hot_sigma.setValue(5.0)
            self.cosm_cold_sigma.setValue(5.0)
            self.cosm_star_mean_ratio.setValue(0.22)
            self.cosm_star_max_ratio.setValue(0.55)
            self.cosm_sat_quantile.setValue(0.9995)
        btn_defaults.clicked.connect(_restore_defaults)
        fl_cosm.addRow(btn_defaults)

        # wire
        self.cosm_enable_cb.toggled.connect(_toggle_cosm_enabled)
        _toggle_cosm_enabled(self.cosm_enable_cb.isChecked())

        right_col.addWidget(gb_cosm)

        # --- Comet (tuning only; not an algorithm picker) ---
        gb_comet = QGroupBox("Comet (High-Clip Percentile tuning)")
        fl_comet = QFormLayout(gb_comet)

        # load saved values (with defaults)
        def _getf(key, default):
            return self.settings.value(key, default, type=float)

        self.comet_hclip_k = QDoubleSpinBox()
        self.comet_hclip_k.setRange(0.1, 10.0)
        self.comet_hclip_k.setDecimals(2)
        self.comet_hclip_k.setSingleStep(0.05)
        self.comet_hclip_k.setValue(_getf("stacking/comet_hclip_k", 1.30))

        self.comet_hclip_p = QDoubleSpinBox()
        self.comet_hclip_p.setRange(1.0, 99.0)
        self.comet_hclip_p.setDecimals(1)
        self.comet_hclip_p.setSingleStep(1.0)
        self.comet_hclip_p.setValue(_getf("stacking/comet_hclip_p", 25.0))

        row_hclip = QWidget()
        row_hclip_h = QHBoxLayout(row_hclip); row_hclip_h.setContentsMargins(0,0,0,0)
        row_hclip_h.addWidget(QLabel("High-clip k / Percentile p:"))
        row_hclip_h.addWidget(self.comet_hclip_k)
        row_hclip_h.addWidget(QLabel(" / "))
        row_hclip_h.addWidget(self.comet_hclip_p)

        fl_comet.addRow(row_hclip)
        right_col.addWidget(gb_comet)

        # --- Comet Star Removal (Optional) ---
        gb_csr = QGroupBox("Comet Star Removal (Optional)")
        fl_csr = QFormLayout(gb_csr)

        self.csr_enable = QCheckBox("Remove stars on comet-aligned frames")
        self.csr_enable.setChecked(self.settings.value("stacking/comet_starrem/enabled", False, type=bool))
        fl_csr.addRow(self.csr_enable)
        self.csr_enable.toggled.connect(lambda v: self.settings.setValue("stacking/comet_starrem/enabled", bool(v)))

        self.csr_tool = QComboBox()
        self.csr_tool.addItems(["StarNet", "CosmicClarityDarkStar"])
        curr_tool = self.settings.value("stacking/comet_starrem/tool", "StarNet", type=str)
        self.csr_tool.setCurrentText(curr_tool if curr_tool in ("StarNet","CosmicClarityDarkStar") else "StarNet")
        fl_csr.addRow("Tool:", self.csr_tool)

        self.csr_core_r = QDoubleSpinBox(); self.csr_core_r.setRange(2.0, 200.0); self.csr_core_r.setDecimals(1)
        self.csr_core_r.setValue(self.settings.value("stacking/comet_starrem/core_r", 22.0, type=float))
        fl_csr.addRow("Protect core radius (px):", self.csr_core_r)

        self.csr_core_soft = QDoubleSpinBox(); self.csr_core_soft.setRange(0.0, 100.0); self.csr_core_soft.setDecimals(1)
        self.csr_core_soft.setValue(self.settings.value("stacking/comet_starrem/core_soft", 6.0, type=float))
        fl_csr.addRow("Core mask feather (px):", self.csr_core_soft)

        def _toggle_csr(on: bool):
            for w in (self.csr_tool, self.csr_core_r, self.csr_core_soft):
                w.setEnabled(on)
        _toggle_csr(self.csr_enable.isChecked())
        self.csr_enable.toggled.connect(_toggle_csr)

        right_col.addWidget(gb_csr)

        right_col.addStretch(1)

        # --- Buttons ---
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(lambda: self.save_stacking_settings(dialog))
        btns.rejected.connect(dialog.reject)
        root.addWidget(btns)

        dialog.resize(900, 640)
        dialog.exec()

        try:
            self._refresh_comet_starless_enable()
        except Exception:
            pass        


    def save_stacking_settings(self, dialog):
        """
        Save settings and restart the Stacking Suite if the directory OR internal dtype changed.
        Uses dialog-scoped dir_edit and normalized path comparison.
        """
        # --- capture previous state BEFORE we change anything ---
        prev_dir_raw   = self.stacking_directory or ""
        prev_dir       = self._norm_dir(prev_dir_raw)
        prev_dtype_str = "float64" if (getattr(self, "internal_dtype", np.float32) is np.float64) else "float32"


        # --- read dialog widgets ---
        dir_edit   = getattr(dialog, "_dir_edit", None)
        new_dir_raw = (dir_edit.text() if dir_edit else prev_dir_raw)
        new_dir     = self._norm_dir(new_dir_raw)

        # Persist the rest
        self.sigma_high       = self.sigma_high_spinbox.value()
        self.sigma_low        = self.sigma_low_spinbox.value()
        self.rejection_algorithm = self.rejection_algo_combo.currentText()
        self.kappa           = self.kappa_spinbox.value()
        self.iterations      = self.iterations_spinbox.value()
        self.esd_threshold   = self.esd_spinbox.value()
        self.biweight_constant = self.biweight_spinbox.value()
        self.trim_fraction   = self.trim_spinbox.value()
        self.modz_threshold  = self.modz_spinbox.value()
        self.chunk_height    = self.chunkHeightSpinBox.value()
        self.chunk_width     = self.chunkWidthSpinBox.value()

        # Update instance + QSettings (write RAW path; use normalized only for comparison)
        self.stacking_directory = new_dir_raw
        self.settings.setValue("stacking/dir", new_dir_raw)
        self.settings.setValue("stacking/sigma_high", self.sigma_high)
        self.settings.setValue("stacking/sigma_low", self.sigma_low)
        self.settings.setValue("stacking/rejection_algorithm", self.rejection_algorithm)
        self.settings.setValue("stacking/kappa", self.kappa)
        self.settings.setValue("stacking/iterations", self.iterations)
        self.settings.setValue("stacking/esd_threshold", self.esd_threshold)
        self.settings.setValue("stacking/biweight_constant", self.biweight_constant)
        self.settings.setValue("stacking/trim_fraction", self.trim_fraction)
        self.settings.setValue("stacking/modz_threshold", self.modz_threshold)
        self.settings.setValue("stacking/chunk_height", self.chunk_height)
        self.settings.setValue("stacking/chunk_width", self.chunk_width)
        self.settings.setValue("stacking/autocrop_enabled", self.autocrop_cb.isChecked())
        self.settings.setValue("stacking/autocrop_pct", float(self.autocrop_pct.value()))

        # ----- alignment model (affine | homography | poly3 | poly4) -----
        model_idx = self.align_model_combo.currentIndex()
        if   model_idx == 0: model_name = "affine"
        elif model_idx == 1: model_name = "similarity"   # No Distortion mode
        elif model_idx == 2: model_name = "homography"
        elif model_idx == 3: model_name = "poly3"
        else:                model_name = "poly4"

        self.settings.setValue("stacking/align/model",      model_name)
        self.settings.setValue("stacking/align/max_cp",     int(self.align_max_cp.value()))
        self.settings.setValue("stacking/align/downsample", int(self.align_downsample.value()))
        self.settings.setValue("stacking/align/h_reproj",   float(self.h_ransac_reproj.value()))

        # Seed mode (persist as stable tokens: 'robust' | 'median')
        seed_idx = int(self.mf_seed_combo.currentIndex())
        seed_mode_val = "median" if seed_idx == 1 else "robust"
        self.settings.setValue("stacking/mfdeconv/seed_mode", seed_mode_val)

        # Star mask params
        self.settings.setValue("stacking/mfdeconv/star_mask/thresh_sigma",  float(self.sm_thresh.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/grow_px",       int(self.sm_grow.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/soft_sigma",    float(self.sm_soft.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/max_radius_px", int(self.sm_rmax.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/max_objs",      int(self.sm_maxobjs.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/keep_floor",    float(self.sm_keepfloor.value()))
        self.settings.setValue("stacking/mfdeconv/star_mask/ellipse_scale", float(self.sm_es.value()))

        # Variance map params
        self.settings.setValue("stacking/mfdeconv/varmap/sample_stride", int(self.vm_stride.value()))
        self.settings.setValue("stacking/mfdeconv/varmap/smooth_sigma",  float(self.vm_sigma.value()))
        vm_floor = 10.0 ** float(self.vm_floor_log.value())
        self.settings.setValue("stacking/mfdeconv/varmap/floor", vm_floor)

        # MFDeconv engine selection
        if   self.mf_eng_cudnn_rb.isChecked(): mf_engine = "cudnn"
        elif self.mf_eng_sport_rb.isChecked(): mf_engine = "sport"
        else:                                   mf_engine = "normal"
        self.settings.setValue("stacking/mfdeconv/engine", mf_engine)

        # (compat: drop the legacy boolean; keep it synced for older configs if you like)
        self.settings.setValue("stacking/high_octane", mf_engine == "sport")
        # Gradient settings
        self.settings.setValue("stacking/grad_poly2/enabled",   self.chk_poly2.isChecked())
        self.settings.setValue("stacking/grad_poly2/mode",       "divide" if self.grad_mode_combo.currentIndex() == 1 else "subtract")
        self.settings.setValue("stacking/grad_poly2/samples",    self.grad_samples_spin.value())
        self.settings.setValue("stacking/grad_poly2/downsample", self.grad_downsample_spin.value())
        self.settings.setValue("stacking/grad_poly2/patch_size", self.grad_patch_spin.value())
        self.settings.setValue("stacking/grad_poly2/min_strength", float(self.grad_min_strength.value()))
        self.settings.setValue("stacking/grad_poly2/gain_lo",    float(self.grad_gain_lo.value()))
        self.settings.setValue("stacking/grad_poly2/gain_hi",    float(self.grad_gain_hi.value()))

        # Cosmetic (Advanced)
        self.settings.setValue("stacking/cosmetic/custom_enable", self.cosm_enable_cb.isChecked())
        self.settings.setValue("stacking/cosmetic/hot_sigma",      float(self.cosm_hot_sigma.value()))
        self.settings.setValue("stacking/cosmetic/cold_sigma",     float(self.cosm_cold_sigma.value()))
        self.settings.setValue("stacking/cosmetic/star_mean_ratio",float(self.cosm_star_mean_ratio.value()))
        self.settings.setValue("stacking/cosmetic/star_max_ratio", float(self.cosm_star_max_ratio.value()))
        self.settings.setValue("stacking/cosmetic/sat_quantile",   float(self.cosm_sat_quantile.value()))

        self.settings.setValue("stacking/use_hardware_accel", self.hw_accel_cb.isChecked())
        self.use_gpu_integration = bool(self.hw_accel_cb.isChecked())

        self.settings.setValue("stacking/comet_starrem/enabled", self.csr_enable.isChecked())
        self.settings.setValue("stacking/comet_starrem/tool", self.csr_tool.currentText())
        self.settings.setValue("stacking/comet_starrem/core_r", float(self.csr_core_r.value()))
        self.settings.setValue("stacking/comet_starrem/core_soft", float(self.csr_core_soft.value()))

        passes = 1 if self.align_passes_combo.currentIndex() == 0 else 3
        self.settings.setValue("stacking/refinement_passes", passes)
        self.settings.setValue("stacking/shift_tolerance", self.shift_tol_spin.value())
        self.settings.setValue("stacking/accept_shift_px", float(self.accept_shift_spin.value()))
        self.accept_thresh = float(self.accept_shift_spin.value())  # keep runtime attribute in sync        
        self.settings.setValue("stacking/align/det_sigma",   float(self.align_det_sigma.value()))
        self.settings.setValue("stacking/align/minarea",     int(self.align_minarea.value()))

        self.settings.setValue("stacking/align/limit_stars", int(self.align_limit_stars_spin.value()))
        self.settings.setValue("stacking/align/timeout_per_job_sec", int(self.align_timeout_spin.value()))

        self.settings.setValue("stacking/drop_shrink", float(self.drop_shrink_spin.value()))

        kidx = self.drizzle_kernel_combo.currentIndex()
        kname = "square" if kidx==0 else ("circular" if kidx==1 else "gaussian")
        self.settings.setValue("stacking/drizzle_kernel", kname)

        self.settings.setValue("stacking/drizzle_gauss_sigma", float(self.gauss_sigma_spin.value()))
        self.settings.setValue("stacking/comet_hclip_k", float(self.comet_hclip_k.value()))
        self.settings.setValue("stacking/comet_hclip_p", float(self.comet_hclip_p.value()))
        # --- precision (internal dtype) ---
        chosen = self.precision_combo.currentText()  # "32-bit float" or "64-bit float"
        new_dtype_str = "float64" if "64" in chosen else "float32"
        dtype_changed = (new_dtype_str != prev_dtype_str)

        self.internal_dtype = np.float64 if new_dtype_str == "float64" else np.float32
        self.settings.setValue("stacking/internal_dtype", new_dtype_str)

        # Make sure everything is flushed
        self.settings.sync()

        # Logging
        self.update_status("✅ Saved stacking settings.")
        self.update_status(f"• Internal precision: {new_dtype_str}")
        self.update_status(f"• Hardware acceleration: {'ON' if self.use_gpu_integration else 'OFF'}")
        self._update_stacking_path_display()

        # --- restart if needed ---
        dir_changed = (new_dir != prev_dir)
        if dir_changed or dtype_changed:
            reasons = []
            if dir_changed:
                reasons.append("folder change")
            if dtype_changed:
                reasons.append(f"precision → {new_dtype_str}")
            self.update_status(f"🔁 Restarting Stacking Suite to apply {', '.join(reasons)}…")
            dialog.accept()
            self._restart_self()
            return

        dialog.accept()



    # --- Drizzle config: single source of truth ---
    def _get_drizzle_pixfrac(self) -> float:
        s = self.settings
        # new canonical -> old aliases -> default
        v = s.value("stacking/drizzle_pixfrac", None, type=float)
        if v is None:
            v = s.value("stacking/drizzle_drop", None, type=float)
        if v is None:
            v = s.value("stacking/drop_shrink", 0.65, type=float)
        # clamp to [0, 1]
        return float(max(0.0, min(1.0, v)))

    def _set_drizzle_pixfrac(self, v: float) -> None:
        v = float(max(0.0, min(1.0, v)))
        s = self.settings
        # write to canonical + legacy keys (back-compat)
        s.setValue("stacking/drizzle_pixfrac", v)
        s.setValue("stacking/drizzle_drop", v)
        s.setValue("stacking/drop_shrink", v)

        # reflect in any live widgets without feedback loops
        for wname in ("drizzle_drop_shrink_spin", "drop_shrink_spin"):
            w = getattr(self, wname, None)
            if w is not None and abs(float(w.value()) - v) > 1e-9:
                w.blockSignals(True); w.setValue(v); w.blockSignals(False)

    def _get_drizzle_scale(self) -> float:
        # Accepts "1x/2x/3x" or numeric
        val = self.settings.value("stacking/drizzle_scale", "2x", type=str)
        if isinstance(val, str) and val.endswith("x"):
            try: return float(val[:-1])
            except: return 2.0
        return float(val)

    def _set_drizzle_scale(self, r: float | str) -> None:
        if isinstance(r, str):
            try: r = float(r.rstrip("xX"))
            except: r = 2.0
        r = float(max(1.0, min(3.0, r)))
        # store as “Nx” so the combo’s string stays in sync
        self.settings.setValue("stacking/drizzle_scale", f"{int(r)}x")
        if hasattr(self, "drizzle_scale_combo"):
            txt = f"{int(r)}x"
            if self.drizzle_scale_combo.currentText() != txt:
                self.drizzle_scale_combo.blockSignals(True)
                self.drizzle_scale_combo.setCurrentText(txt)
                self.drizzle_scale_combo.blockSignals(False)


    def closeEvent(self, e):
        # Graceful shutdown for any running workers
        try:
            if hasattr(self, "alignment_thread") and self.alignment_thread and self.alignment_thread.isRunning():
                self.alignment_thread.requestInterruption()
                self.alignment_thread.wait(1500)
        except Exception:
            pass
        super().closeEvent(e)

    def _mf_worker_class_from_settings(self):
        """Return (WorkerClass, engine_name) from settings."""
        # local import avoids import-time cost if user never runs MFDeconv
        from pro.mfdeconv import MultiFrameDeconvWorker
        from pro.mfdeconvcudnn import MultiFrameDeconvWorkercuDNN
        from pro.mfdeconvsport import MultiFrameDeconvWorkerSport

        eng = str(self.settings.value("stacking/mfdeconv/engine", "normal", type=str) or "normal").lower()
        if eng == "cudnn":
            return (MultiFrameDeconvWorkercuDNN, "Normal (cuDNN-free)")
        if eng == "sport":
            return (MultiFrameDeconvWorkerSport, "High-Octane")
        return (MultiFrameDeconvWorker, "Normal")


    def _restart_self(self):
        geom = self.saveGeometry()
        try:
            cur_tab = self.tabs.currentIndex()
        except Exception:
            cur_tab = None

        parent = self.parent()  # may be None

        app = QApplication.instance()
        # Keep a global strong ref so GC can't collect the new dialog
        if not hasattr(app, "_stacking_suite_ref"):
            app._stacking_suite_ref = None

        def spawn():
            new = StackingSuiteDialog(parent=parent)
            if geom:
                new.restoreGeometry(geom)
            if cur_tab is not None:
                try:
                    new.tabs.setCurrentIndex(cur_tab)
                except Exception:
                    pass
            new.show()
            app._stacking_suite_ref = new  # <<< strong ref lives for app lifetime

        QTimer.singleShot(0, spawn)
        self.close()

    def _on_stacking_directory_changed(self, old_dir: str, new_dir: str):
        # Stop any running worker safely
        if hasattr(self, "alignment_thread") and self.alignment_thread:
            try:
                if self.alignment_thread.isRunning():
                    self.alignment_thread.requestInterruption()
                    self.alignment_thread.wait(1500)
            except Exception:
                pass

        self._ensure_stacking_subdirs(new_dir)
        self._clear_integration_state()

        # 🔁 RESCAN + REPOPULATE (the key bit you’re missing)
        self._reload_lists_for_new_dir()

        # If your tabs populate on change, poke the active one:
        if hasattr(self, "on_tab_changed"):
            self.on_tab_changed(self.tabs.currentIndex())

        # Update any path labels
        self._update_stacking_path_display()

        # Reload any persisted master selections
        try:
            self.restore_saved_master_calibrations()
        except Exception:
            pass

        self.update_status(f"📂 Stacking directory changed:\n    {old_dir or '(none)'} → {new_dir}")

    def _reload_lists_for_new_dir(self):
        """
        Re-scan the new stacking directory and repopulate internal dicts AND UI.
        """
        base = self.stacking_directory or ""
        self.conversion_output_directory = os.path.join(base, "Converted_Images")

        # Rebuild dictionaries from disk
        self.dark_files  = self._discover_grouped(os.path.join(base, "Calibrated_Darks"))
        self.flat_files  = self._discover_grouped(os.path.join(base, "Calibrated_Flats"))
        self.light_files = self._discover_grouped(os.path.join(base, "Calibrated_Lights"))

        # If you store master lists/sizes by path, clear/reseed minimally
        self.master_files.clear()
        self.master_sizes.clear()

        # 🔄 Update the tab UIs if you have builders; try common method names safely
        # Darks
        if hasattr(self, "rebuild_dark_tree"):
            self.rebuild_dark_tree(self.dark_files)
        elif hasattr(self, "populate_dark_tab"):
            self.populate_dark_tab()

        # Flats
        if hasattr(self, "rebuild_flat_tree"):
            self.flat_tab.rebuild_flat_tree(self.flat_files)
        elif hasattr(self, "populate_flat_tab"):
            self.populate_flat_tab()

        # Lights
        if hasattr(self, "rebuild_light_tree"):
            self.rebuild_light_tree(self.light_files)
        elif hasattr(self, "populate_light_tab"):
            self.populate_light_tab()

        # Image Integration (registration) tab often shows counts/paths
        if hasattr(self, "refresh_integration_tab"):
            self.refresh_integration_tab()

        self.update_status(f"🔄 Re-scanned calibrated sets in: {base}")

    def _discover_grouped(self, root_dir: str) -> dict:
        """
        Walk 'root_dir' and return {group_name: [file_paths,...]}.
        Group = immediate subfolder name; if files are directly in root, group 'Ungrouped'.
        """
        groups = {}
        if not root_dir or not os.path.isdir(root_dir):
            return groups

        valid_ext = (".fit", ".fits", ".xisf", ".tif", ".tiff")
        root_dir = os.path.normpath(root_dir)

        for dirpath, _, files in os.walk(root_dir):
            for fn in files:
                if not fn.lower().endswith(valid_ext):
                    continue
                fpath = os.path.normpath(os.path.join(dirpath, fn))
                parent = os.path.basename(os.path.dirname(fpath))
                group  = parent if os.path.dirname(fpath) != root_dir else "Ungrouped"
                groups.setdefault(group, []).append(fpath)

        # Stable ordering helps
        for g in groups:
            groups[g].sort()
        return groups

    def _refresh_all_tabs_once(self):
        current = self.tabs.currentIndex()
        if hasattr(self, "on_tab_changed"):
            for idx in range(self.tabs.count()):
                self.on_tab_changed(idx)
        self.tabs.setCurrentIndex(current)

    def _ensure_stacking_subdirs(self, base_dir: str):
        try:
            os.makedirs(base_dir, exist_ok=True)
            for sub in (
                "Aligned_Images",
                "Normalized_Images",
                "Calibrated_Darks",
                "Calibrated_Flats",
                "Calibrated_Lights",
                "Converted_Images",
                "Masters",
            ):
                os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
        except Exception as e:
            self.update_status(f"⚠️ Could not ensure subfolders in '{base_dir}': {e}")

    def _clear_integration_state(self):
        # wipe per-run state so we don't “blend” two directories
        self.per_group_drizzle.clear()
        self.manual_dark_overrides.clear()
        self.manual_flat_overrides.clear()
        self.reg_files.clear()
        self.session_tags.clear()
        self.deleted_calibrated_files.clear()
        self._norm_map.clear()
        setattr(self, "valid_transforms", {})
        setattr(self, "frame_weights", {})
        setattr(self, "_global_autocrop_rect", None)

    def _rebuild_tabs_after_dir_change(self):
        # Rebuild the tab widgets so any path assumptions inside them reset to the new dir
        current = self.tabs.currentIndex()

        # Remove all tabs & delete widgets
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            try:
                w.deleteLater()
            except Exception:
                pass

        # Recreate against the new base path
        self.conversion_tab = self.conversion_tab.create_conversion_tab()
        self.dark_tab       = self.dark_tab.create_dark_tab()
        self.flat_tab       = self.flat_tab.create_flat_tab()
        self.light_tab      = self.light_tab.create_light_tab()
        self.image_integration_tab = self.registration_tab.create_image_registration_tab()

        self.tabs.addTab(self.conversion_tab, "Convert Non-FITS Formats")
        self.tabs.addTab(self.dark_tab,       "Darks")
        self.tabs.addTab(self.flat_tab,       "Flats")
        self.tabs.addTab(self.light_tab,      "Lights")
        self.tabs.addTab(self.image_integration_tab, "Image Integration")

        # Restore previously active tab if possible
        if 0 <= current < self.tabs.count():
            self.tabs.setCurrentIndex(current)
        else:
            self.tabs.setCurrentIndex(1)  # Darks by default

    def select_stacking_directory(self):
        """ Opens a dialog to choose a stacking directory. """
        directory = QFileDialog.getExistingDirectory(self, "Select Stacking Directory")
        if directory:
            self.stacking_directory = directory
            self.dir_path_edit.setText(directory)  # No more AttributeError
            self.settings.setValue("stacking/dir", directory)  # Save the new directory
            self._update_stacking_path_display()

    def _bind_shared_setting_checkbox(self, key: str, checkbox: QCheckBox, default: bool = True):
        """Bind a QCheckBox to a shared QSettings key and keep all bound boxes in sync."""
        # registry of sets of checkboxes per setting key
        if not hasattr(self, "_shared_checkboxes"):
            self._shared_checkboxes = {}

        # initialize from settings
        val = self.settings.value(key, default, type=bool)
        checkbox.blockSignals(True)
        checkbox.setChecked(bool(val))
        checkbox.blockSignals(False)

        # track this checkbox (weak so GC is fine when tabs close)
        boxset = self._shared_checkboxes.setdefault(key, weakref.WeakSet())
        boxset.add(checkbox)

        # when this one toggles, update settings and all siblings
        def _on_toggled(v: bool):
            self.settings.setValue(key, bool(v))
            # sync siblings without re-emitting signals
            for cb in list(self._shared_checkboxes.get(key, [])):
                if cb is checkbox:
                    continue
                try:
                    cb.blockSignals(True)
                    cb.setChecked(bool(v))
                    cb.blockSignals(False)
                except RuntimeError:
                    # widget was likely deleted; ignore
                    pass

        checkbox.toggled.connect(_on_toggled, Qt.ConnectionType.QueuedConnection)

    def _tree_for_type(self, t: str):
        t = (t or "").upper()
        if t == "LIGHT": return getattr(self, "light_tree", None)
        if t == "FLAT":  return getattr(self, "flat_tree", None)
        if t == "DARK":  return getattr(self, "dark_tree", None)
        return None

    def _guess_type_from_imagetyp(self, imagetyp: str) -> str | None:
        s = (imagetyp or "").strip().lower()
        # common patterns
        if "flat" in s: return "FLAT"
        if "dark" in s: return "DARK"
        if "bias" in s or "offset" in s: return "DARK"  # treat bias as darks bucket if you like
        if "light" in s or "object" in s: return "LIGHT"
        return None

    def _parse_exposure_from_groupkey(self, group_key: str) -> float | None:
        # expects "... - 300s (WxH)" → 300

        m = re.search(r"\s-\s([0-9.]+)s\s*\(", group_key)
        if not m: return None
        try:
            return float(m.group(1))
        except Exception:
            return None


    def _report_group_summary(self, expected_type: str):
        """
        Print #files and total integration per group (and session when applicable).
        For LIGHT/FLAT we key by (group_key, session_tag). For DARK we key by group_key.
        """
        t = (expected_type or "").upper()
        if t == "LIGHT":
            store = getattr(self, "light_files", {})
            # store keys: (group_key, session_tag)
            self.update_status("📈 Light groups summary:")
            seen = set()
            for (gkey, sess), paths in store.items():
                if not paths: continue
                exp = self._parse_exposure_from_groupkey(gkey) or 0.0
                tot = exp * len(paths)
                label = f"• {gkey}  |  session: {sess}  →  {len(paths)} files, {self._fmt_hms(tot)}"
                if (gkey, sess) not in seen:
                    self.update_status(label)
                    seen.add((gkey, sess))
        elif t == "FLAT":
            store = getattr(self, "flat_files", {})
            self.update_status("📈 Flat groups summary:")
            seen = set()
            for (gkey, sess), paths in store.items():
                if not paths: continue
                exp = self._parse_exposure_from_groupkey(gkey) or 0.0
                tot = exp * len(paths)
                label = f"• {gkey}  |  session: {sess}  →  {len(paths)} files, {self._fmt_hms(tot)}"
                if (gkey, sess) not in seen:
                    self.update_status(label)
                    seen.add((gkey, sess))
        elif t == "DARK":
            store = getattr(self, "dark_files", {})
            self.update_status("📈 Dark groups summary:")
            for gkey, paths in store.items():
                if not paths: continue
                exp = self._parse_exposure_from_groupkey(gkey) or 0.0
                tot = exp * len(paths)
                self.update_status(f"• {gkey}  →  {len(paths)} files, {self._fmt_hms(tot)}")


    def prompt_set_session(self, item, frame_type):
        text, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session name:")
        if not (ok and text.strip()):
            return

        session_name = text.strip()
        is_flat = frame_type.upper() == "FLAT"
        tree = self.flat_tree if is_flat else self.light_tree
        target_dict = self.flat_files if is_flat else self.light_files

        selected_items = tree.selectedItems()

        def update_file_session(filename, widget_item):
            for key in list(target_dict.keys()):
                if isinstance(key, tuple) and len(key) == 2:
                    group_key, old_session = key
                else:
                    continue  # Skip malformed keys

                files = target_dict.get(key, [])
                for f in list(files):
                    if os.path.basename(f) == filename:
                        if old_session != session_name:
                            new_key = (group_key, session_name)
                            if new_key not in target_dict:
                                target_dict[new_key] = []
                            target_dict[new_key].append(f)
                            target_dict[key].remove(f)
                            if not target_dict[key]:
                                del target_dict[key]

                        # Update internal session tag
                        self.session_tags[f] = session_name

                        # Update leaf's metadata column
                        old_meta = widget_item.text(1)
                        if "Session:" in old_meta:
                            new_meta = re.sub(r"Session: [^|]*", f"Session: {session_name}", old_meta)
                        else:
                            new_meta = f"{old_meta} | Session: {session_name}"
                        widget_item.setText(1, new_meta)
                        return

        def recurse_all_leaf_items(parent_item):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                if child.childCount() == 0:
                    update_file_session(child.text(0), child)
                else:
                    recurse_all_leaf_items(child)

        # Case 1: Multi-leaf selection (e.g. Shift/Ctrl-click)
        if selected_items and any(i.childCount() == 0 for i in selected_items):
            for leaf in selected_items:
                if leaf.childCount() == 0:
                    update_file_session(leaf.text(0), leaf)

        # Case 2: Right-clicked on a group (e.g. filter+exposure node)
        elif item and item.childCount() > 0:
            recurse_all_leaf_items(item)

        # ✅ Reassign matching master flats/darks per leaf
        self.assign_best_master_files()

    def _quad_coverage_add(self, cov: np.ndarray, quad: np.ndarray):
        """
        Rasterize a convex quad (4x2 float array of (x,y) in aligned coords) into 'cov' by +1 filling.
        Bounds/clipping are handled. Small, robust scanline fill.
        """
        H, W = cov.shape
        pts = quad.astype(np.float32)

        ymin = max(int(np.floor(np.min(pts[:,1]))), 0)
        ymax = min(int(np.ceil (np.max(pts[:,1]))), H-1)
        if ymin > ymax: return

        # Edges (x0,y0)->(x1,y1), 4 of them
        edges = []
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i+1) % 4]
            edges.append((x0, y0, x1, y1))

        for y in range(ymin, ymax+1):
            xs = []
            yf = float(y) + 0.5  # sample at pixel center
            for (x0, y0, x1, y1) in edges:
                # Skip horizontal edges
                if (y0 <= yf < y1) or (y1 <= yf < y0):
                    # Linear interpolate X at scanline yf
                    t = (yf - y0) / (y1 - y0)
                    xs.append(x0 + t * (x1 - x0))

            if len(xs) < 2:
                continue
            xs.sort()
            # Fill between pairs
            for i in range(0, len(xs), 2):
                xL = int(np.floor(min(xs[i], xs[i+1])))
                xR = int(np.ceil (max(xs[i], xs[i+1])))
                if xR < 0 or xL > W-1: 
                    continue
                xL = max(xL, 0); xR = min(xR, W)
                if xR > xL:
                    cov[y, xL:xR] += 1


    def _max_rectangle_in_binary(self, mask: np.ndarray):
        """
        Largest axis-aligned rectangle of 1s in a binary mask (H×W, dtype=bool).
        Returns (x0, y0, x1, y1) where x1,y1 are exclusive, or None if empty.
        O(H*W) using 'largest rectangle in histogram' per row.
        """
        H, W = mask.shape
        heights = np.zeros(W, dtype=np.int32)
        best = (0, 0, 0, 0, 0)  # (area, x0, y0, x1, y1)

        for y in range(H):
            row = mask[y]
            heights[row] += 1
            heights[~row] = 0

            # Largest rectangle in histogram 'heights'
            stack = []
            i = 0
            while i <= W:
                h = heights[i] if i < W else 0
                if not stack or h >= heights[stack[-1]]:
                    stack.append(i); i += 1
                else:
                    top = stack.pop()
                    height = heights[top]
                    left = stack[-1] + 1 if stack else 0
                    right = i
                    area = height * (right - left)
                    if area > best[0]:
                        # rectangle spans rows [y-height+1 .. y], columns [left .. right-1]
                        y0 = y - height + 1
                        y1 = y + 1
                        best = (area, left, y0, right, y1)

        if best[0] == 0:
            return None
        _, x0, y0, x1, y1 = best
        return (x0, y0, x1, y1)


    def _rect_from_transforms_fast(
        self,
        transforms: dict[str, object],
        src_hw: tuple[int, int],
        coverage_pct: float = 95.0,
        *,
        allow_homography: bool = True,
        min_side: int = 16,
    ) -> tuple[int, int, int, int] | None:
        """
        Robust fast autocrop from a bunch of transforms.

        Accepts:
        - plain 2x3 affine np.array / list-of-lists
        - plain 3x3 homography
        - tuples like (M, meta) or (M, bbox, ...)
        - dicts like {"M": M} or {"matrix": M} or {"H": M}

        Returns (x0, y0, x1, y1) in reference coords, or None.
        """
        H, W = map(int, src_hw)
        if not transforms:
            return None

        # corners in source-image coords
        corners = np.array(
            [
                [0, 0, 1],
                [W, 0, 1],
                [W, H, 1],
                [0, H, 1],
            ],
            dtype=np.float64,
        )

        def _extract_matrix(raw):
            """
            Try hard to get a 2x3 or 3x3 ndarray out of whatever `raw` is.
            Returns np.ndarray or None.
            """
            if raw is None:
                return None

            # common case: (M, extra...) → pick the first thing that looks like a matrix
            if isinstance(raw, (tuple, list)):
                for item in raw:
                    m = _extract_matrix(item)
                    if m is not None:
                        return m
                return None

            # dict style: {"M": ..., "bbox": ...} or {"matrix": ...}
            if isinstance(raw, dict):
                for key in ("M", "matrix", "affine", "H", "homography"):
                    if key in raw:
                        return _extract_matrix(raw[key])
                return None

            # already ndarray-ish
            try:
                arr = np.asarray(raw, dtype=np.float64)
            except Exception:
                return None

            if arr.shape == (2, 3) or arr.shape == (3, 3):
                return arr

            # anything else (like 1D or wrong shape) → ignore
            return None

        lefts, rights, tops, bottoms = [], [], [], []

        for path, raw_M in transforms.items():
            M = _extract_matrix(raw_M)
            if M is None:
                # e.g. this was (M, bbox) but we couldn't unwrap → skip
                continue

            # normalize to 3x3
            if M.shape == (2, 3):
                A = np.eye(3, dtype=np.float64)
                A[:2, :3] = M
                M = A
            elif M.shape == (3, 3):
                if not allow_homography:
                    continue
            else:
                # should not happen due to _extract_matrix, but be safe
                continue

            pts = M @ corners.T  # 3x4
            w = pts[2, :]
            # avoid div0
            w = np.where(np.abs(w) < 1e-12, 1.0, w)
            xs = pts[0, :] / w
            ys = pts[1, :] / w

            lefts.append(xs.min())
            rights.append(xs.max())
            tops.append(ys.min())
            bottoms.append(ys.max())

        if not lefts:
            # all entries were weird / non-matrix
            return None

        lefts = np.asarray(lefts, dtype=np.float64)
        rights = np.asarray(rights, dtype=np.float64)
        tops = np.asarray(tops, dtype=np.float64)
        bottoms = np.asarray(bottoms, dtype=np.float64)

        p = float(np.clip(coverage_pct, 0.0, 100.0)) / 100.0

        # quantile logic: keep the region covered by ≥ p of frames
        x0 = float(np.quantile(lefts, p))
        y0 = float(np.quantile(tops, p))
        x1 = float(np.quantile(rights, 1.0 - p))
        y1 = float(np.quantile(bottoms, 1.0 - p))

        if not np.isfinite((x0, y0, x1, y1)).all():
            return None

        # round inwards
        xi0 = int(np.ceil(x0))
        yi0 = int(np.ceil(y0))
        xi1 = int(np.floor(x1))
        yi1 = int(np.floor(y1))

        if xi1 - xi0 < min_side or yi1 - yi0 < min_side:
            return None

        return (xi0, yi0, xi1, yi1)


    def _compute_common_autocrop_rect(self, grouped_files: dict, coverage_pct: float, status_cb=None):
        log = status_cb or self.update_status
        transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
        common_mask = None
        for group_key, file_list in grouped_files.items():
            if not file_list:
                continue
            mask = self._compute_coverage_mask(file_list, transforms_path, coverage_pct)
            if mask is None:
                log(f"✂️ Global crop: no mask for '{group_key}' → disabling global crop.")
                return None
            if common_mask is None:
                common_mask = mask.astype(bool, copy=True)
            else:
                if mask.shape != common_mask.shape:
                    log("✂️ Global crop: mask shapes differ across groups.")
                    return None
                np.logical_and(common_mask, mask, out=common_mask)

        if common_mask is None or not common_mask.any():
            return None

        rect = self._max_rectangle_in_binary(common_mask)
        # Optional safety guard so we never get pencil-thin rectangles:
        if rect:
            x0, y0, x1, y1 = rect
            if (x1 - x0) < 16 or (y1 - y0) < 16:
                log("✂️ Global crop: rect too small; disabling global crop.")
                return None
            log(f"✂️ Global crop rect={rect} → size {x1-x0}×{y1-y0}")
        return rect

    def _first_non_none(self, *vals):
        for v in vals:
            if v is not None:
                return v
        return None

    def _compute_coverage_mask(self, file_list: List[str], transforms_path: str, coverage_pct: float):
        """
        Build a coverage-count image on the aligned canvas for 'file_list'.
        Threshold at coverage_pct, but use the number of frames we ACTUALLY rasterized (N_eff).
        Returns a bool mask (H×W) or None if nothing rasterized.
        """
        if not file_list:
            return None

        # Canvas from first aligned image
        ref_img, _, _, _ = load_image(file_list[0])
        if ref_img is None:
            self.update_status("✂️ Auto-crop: could not load first aligned ref.")
            return None
        H, W = (ref_img.shape if ref_img.ndim == 2 else ref_img.shape[:2])

        if not os.path.exists(transforms_path):
            self.update_status(f"✂️ Auto-crop: no transforms file at {transforms_path}")
            return None
        self.update_status(f"✂️ Auto-crop: Loading transforms...")
        QApplication.processEvents()
        transforms = self.registration_tab.load_alignment_matrices_custom(transforms_path)

        # --- Robust transform lookup: key by normalized full path AND by basename ---
        def _normcase(p):  # windows-insensitive
            p = os.path.normpath(os.path.abspath(p))
            return p.lower() if os.name == "nt" else p

        xforms_by_full = { _normcase(k): v for k, v in transforms.items() }
        xforms_by_name = {}
        for k, v in transforms.items():
            xforms_by_name.setdefault(os.path.basename(k), v)

        cov = np.zeros((H, W), dtype=np.uint16)
        used = 0

        for aligned_path in file_list:
            base = os.path.basename(aligned_path)
            if base.endswith("_n_r.fit"):
                raw_base = base.replace("_n_r.fit", "_n.fit")
            elif base.endswith("_r.fit"):
                raw_base = base.replace("_r.fit", ".fit")
            else:
                raw_base = base

            # try normalized-Images location first
            raw_path_guess = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)

            # find transform
            M = self._first_non_none(
                xforms_by_full.get(_normcase(raw_path_guess)),
                xforms_by_full.get(_normcase(aligned_path)),
                transforms.get(raw_path_guess),
                transforms.get(os.path.normpath(aligned_path)),
                xforms_by_name.get(raw_base),
            )

            if M is None:
                # Can't rasterize this frame
                continue

            # raw size
            h_raw = w_raw = None
            if os.path.exists(raw_path_guess):
                raw_img, _, _, _ = load_image(raw_path_guess)
                if raw_img is not None:
                    h_raw, w_raw = (raw_img.shape if raw_img.ndim == 2 else raw_img.shape[:2])

            if h_raw is None or w_raw is None:
                # fallback to aligned canvas size (still okay; affine provides placement)
                h_raw, w_raw = H, W

            corners = np.array([[0,0],[w_raw-1,0],[w_raw-1,h_raw-1],[0,h_raw-1]], dtype=np.float32)
            A = M[:, :2]; t = M[:, 2]
            quad = (corners @ A.T) + t

            self._quad_coverage_add(cov, quad)
            used += 1

        if used == 0:
            self.update_status("✂️ Auto-crop: 0/{} frames had usable transforms; skipping.".format(len(file_list)))
            return None

        need = int(np.ceil((coverage_pct / 100.0) * used))
        mask = (cov >= need)
        self.update_status(f"✂️ Auto-crop: rasterized {used}/{len(file_list)} frames; need {need} per-pixel.")
        QApplication.processEvents()
        if not mask.any():
            self.update_status("✂️ Auto-crop: threshold produced empty mask.")
            return None
        return mask



    def _compute_autocrop_rect(self, file_list: List[str], transforms_path: str, coverage_pct: float):
        """
        Build a coverage-count image (aligned canvas), threshold at pct, and extract largest rectangle.e
        Returns (x0, y0, x1, y1) or None.
        """
        if not file_list:
            return None

        # Load aligned reference to get canvas size
        ref_img, ref_hdr, _, _ = load_image(file_list[0])
        if ref_img is None:
            return None
        if ref_img.ndim == 2:
            H, W = ref_img.shape
        else:
            H, W = ref_img.shape[:2]

        # Load transforms (raw _n path -> 2x3 matrix mapping raw->aligned)
        if not os.path.exists(transforms_path):
            return None
        transforms = self.registration_tab.load_alignment_matrices_custom(transforms_path)

        # We need the raw (normalized) image size for each file to transform its corners
        # From aligned name "..._n_r.fit" get raw name "..._n.fit" (like in your drizzle code)
        cov = np.zeros((H, W), dtype=np.uint16)
        for aligned_path in file_list:
            base = os.path.basename(aligned_path)
            if base.endswith("_n_r.fit"):
                raw_base = base.replace("_n_r.fit", "_n.fit")
            elif base.endswith("_r.fit"):
                raw_base = base.replace("_r.fit", ".fit")  # fallback
            else:
                raw_base = base  # fallback

            raw_path = os.path.join(self.stacking_directory, "Normalized_Images", raw_base)
            # Fallback if normalized folder differs:
            raw_key = os.path.normpath(raw_path)
            M = transforms.get(raw_key, None)
            if M is None:
                # Try direct key (some pipelines use normalized path equal to aligned key)
                M = transforms.get(os.path.normpath(aligned_path), None)
            if M is None:
                continue

            # Determine raw size
            raw_img, _, _, _ = load_image(raw_key) if os.path.exists(raw_key) else (None, None, None, None)
            if raw_img is None:
                # last resort: assume same canvas; still yields a conservative crop
                h_raw, w_raw = H, W
            else:
                if raw_img.ndim == 2:
                    h_raw, w_raw = raw_img.shape
                else:
                    h_raw, w_raw = raw_img.shape[:2]

            # Transform raw rectangle corners into aligned coords
            corners = np.array([
                [0,       0      ],
                [w_raw-1, 0      ],
                [w_raw-1, h_raw-1],
                [0,       h_raw-1]
            ], dtype=np.float32)

            # Apply affine: [x' y']^T = A*[x y]^T + t
            A = M[:, :2]; t = M[:, 2]
            quad = (corners @ A.T) + t  # shape (4,2)

            # Rasterize into coverage
            self._quad_coverage_add(cov, quad)

        # Threshold at requested coverage
        N = len(file_list)
        need = int(np.ceil((coverage_pct / 100.0) * N))
        mask = (cov >= need)

        # Largest rectangle of 1s
        rect = self._max_rectangle_in_binary(mask)
        return rect

    def _refresh_reg_tree_summaries(self):
        """
        Image Integration (Registration) tree:
        • For a top-level group like "LP - 20s (1080x1920)":  "<N> files · <HHh MMm SSs>"
        • For a parent filter row (if present):               sum over its children.
        """
        tree = getattr(self, "reg_tree", None)
        if tree is None:
            return

        def _summarize_item(item) -> tuple[int, float]:
            """Return (n_files_total, seconds_total) for this node (recurses children)."""
            if item is None:
                return (0, 0.0)

            # If this node has leaf children (files), assume label carries exposure (e.g., "… 20s (…)").
            n_children = item.childCount()
            if n_children == 0:
                # leaf (file) → contribute 1 file, but exposure is taken from parent; return (1, 0) so parent can add its exp
                return (1, 0.0)

            # group / parent
            label_exp = self._exposure_from_label(item.text(0))  # may be None for pure filter rows
            total_files = 0
            total_secs  = 0.0

            for i in range(n_children):
                ch = item.child(i)
                ch_files, ch_secs = _summarize_item(ch)
                total_files += ch_files
                total_secs  += ch_secs

            # If this node itself encodes an exposure, multiply its exposure by its direct leaf count
            # (i.e., files directly under this group).
            if label_exp is not None:
                direct_files = sum(1 for i in range(n_children) if item.child(i).childCount() == 0)
                total_secs += (label_exp * direct_files)

                # Also set this row's Metadata to its own group’s numbers:
                if direct_files > 0:
                    item.setText(1, (f"{direct_files} file · {self._fmt_hms(label_exp * direct_files)}"
                                    if direct_files == 1 else
                                    f"{direct_files} files · {self._fmt_hms(label_exp * direct_files)}"))
                else:
                    # clear if empty
                    item.setText(1, "")

            # For a pure parent (filter) row with no exposure in its label, show the sum across children
            if label_exp is None:
                if total_files == 1:
                    item.setText(1, f"1 file · {self._fmt_hms(total_secs)}")
                else:
                    item.setText(1, f"{total_files} files · {self._fmt_hms(total_secs)}")

            return (total_files, total_secs)

        for i in range(tree.topLevelItemCount()):
            _summarize_item(tree.topLevelItem(i))


    def _show_gpu_accel_fix_help(self):
        from PyQt6.QtWidgets import QMessageBox, QApplication
        msg = QMessageBox(self)
        msg.setWindowTitle("GPU still not being used?")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(
            "Open Command Prompt and run the following.\n\n"
            "Step 1: uninstall PyTorch\n"
            "Step 2: install the correct build for your GPU"
        )

        # Exact commands (kept as Windows-friendly with %LOCALAPPDATA%)
        cmds = r'''
    "%LOCALAPPDATA%\SASpro\runtime\py312\venv\Scripts\python.exe" -m pip uninstall -y torch

    -> Then install ONE of the following:

    -> AMD / Intel GPUs:
    "%LOCALAPPDATA%\SASpro\runtime\py312\venv\Scripts\python.exe" -m pip install torch-directml

    -> NVIDIA GPUs (CUDA 12.9):
    "%LOCALAPPDATA%\SASpro\runtime\py312\venv\Scripts\python.exe" -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu129
    '''.strip()

        # Show commands in the expandable details area
        msg.setDetailedText(cmds)

        # Add a one-click copy button
        copy_btn = msg.addButton("Copy commands", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Close)

        msg.exec()
        if msg.clickedButton() is copy_btn:
            QApplication.clipboard().setText(cmds)


    def _on_drizzle_checkbox_toggled(self, checked: bool):
        """Drizzle master switch."""
        self.drizzle_scale_combo.setEnabled(checked)
        self.drizzle_drop_shrink_spin.setEnabled(checked)

        # NEW: make CFA Drizzle depend on Enable Drizzle
        self.cfa_drizzle_cb.setEnabled(checked)
        if not checked and self.cfa_drizzle_cb.isChecked():
            # force false without re-entering handlers
            self.cfa_drizzle_cb.blockSignals(True)
            self.cfa_drizzle_cb.setChecked(False)
            self.cfa_drizzle_cb.blockSignals(False)
            self.settings.setValue("stacking/cfa_drizzle", False)

        self.settings.setValue("stacking/drizzle_enabled", bool(checked))
        self._update_drizzle_summary_columns()

    def _on_drizzle_param_changed(self, *_):
        # Persist drizzle params whenever changed
        self.settings.setValue("stacking/drizzle_scale", self.drizzle_scale_combo.currentText())
        self.settings.setValue("stacking/drizzle_drop", float(self.drizzle_drop_shrink_spin.value()))
        # If you reflect params to tree rows, update here:
        # self._refresh_reg_tree_drizzle_column()

    def _on_star_trail_toggled(self, enabled: bool):
        """
        When Star-Trail mode is ON, we skip registration/alignment and use max-value stack.
        Disable other registration-dependent features (drizzle/comet/MFDeconv) to avoid confusion.
        """
        # Controls to gate
        drizzle_widgets = (self.drizzle_checkbox, self.drizzle_scale_combo, self.drizzle_drop_shrink_spin, self.cfa_drizzle_cb)
        comet_widgets = (self.comet_cb, self.comet_pick_btn, self.comet_blend_cb, self.comet_mix)
        mf_widgets = (self.mf_enabled_cb, self.mf_iters_spin, self.mf_kappa_spin, self.mf_color_combo, self.mf_Huber_spin)

        for w in drizzle_widgets + comet_widgets + mf_widgets:
            w.setEnabled(not enabled)

        if enabled:
            self.status_signal.emit("⭐ Star-Trail Mode enabled: Drizzle, Comet stack, and MFDeconv disabled.")
        else:
            self.status_signal.emit("⭐ Star-Trail Mode disabled: other options re-enabled.")


    def _pick_comet_center(self):
        """
        Let the user click a point on ANY light frame. We store (file_path, x, y)
        and defer mapping into the reference frame until after alignment.
        """
        # choose a source file
        src_path = None

        # 1) try current selection in reg_tree
        it = self._first_selected_leaf(self.reg_tree) if hasattr(self, "_first_selected_leaf") else None
        if it and it.parent() is not None:
            group = it.parent().text(0)
            fname = it.text(0)
            # reconstruct full path from our dicts
            lst = self.light_files.get(group) or []
            for p in lst:
                if os.path.basename(p) == fname or os.path.splitext(os.path.basename(p))[0] in fname:
                    src_path = p; break

        # 2) else, fall back to “first light”, or prompt
        if not src_path:
            all_files = [f for lst in self.light_files.values() for f in lst]
            if all_files:
                src_path = all_files[0]
            else:
                fp, _ = QFileDialog.getOpenFileName(
                    self, "Pick a frame to mark the comet center", self.stacking_directory or "",
                    "Images (*.fit *.fits *.tif *.tiff *.png *.jpg *.jpeg)"
                )
                if not fp:
                    QMessageBox.information(self, "Comet Center", "No file chosen.")
                    return
                src_path = fp

        # load and show a simple click-to-pick dialog
        try:
            img, hdr, _, _ = load_image(src_path)
            if img is None:
                raise RuntimeError("Failed to load image.")
        except Exception as e:
            QMessageBox.critical(self, "Comet Center", f"Could not load:\n{src_path}\n\n{e}")
            return

        dlg = _SimplePickDialog(img, parent=self)  # small helper below
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        x, y = dlg.point()

        # store the seed in ORIGINAL file space (or the path we used)
        self._comet_seed = {"path": os.path.normpath(src_path), "xy": (float(x), float(y))}
        self._comet_ref_xy = None  # will be resolved post-align
        self.update_status(f"🌠 Comet seed set on {os.path.basename(src_path)} at ({x:.1f}, {y:.1f}).")


    def _on_cfa_drizzle_toggled(self, checked: bool):
        # If Drizzle is OFF, CFA Drizzle must remain False
        if not self.drizzle_checkbox.isChecked():
            if checked:
                self.cfa_drizzle_cb.blockSignals(True)
                self.cfa_drizzle_cb.setChecked(False)
                self.cfa_drizzle_cb.blockSignals(False)
            checked = False

        self.settings.setValue("stacking/cfa_drizzle", bool(checked))
        self._update_drizzle_summary_columns()


    def _on_drizzle_param_changed(self, *_):
        # persist
        self.settings.setValue("stacking/drizzle_scale", self.drizzle_scale_combo.currentText())
        self.settings.setValue("stacking/drizzle_drop", float(self.drizzle_drop_shrink_spin.value()))
        self._update_drizzle_summary_columns()

    def _update_drizzle_summary_columns(self):
        desc = "OFF"
        if self.drizzle_checkbox.isChecked():
            scale = self.drizzle_scale_combo.currentText()
            drop  = self.drizzle_drop_shrink_spin.value()
            desc = f"ON, Scale {scale}, Drop {drop:.2f}"
            if self.cfa_drizzle_cb.isChecked():
                desc += " + CFA"

        root = self.reg_tree.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setText(2, f"Drizzle: {desc}")

    def _on_star_trail_toggled(self, state):
        self.star_trail_mode = bool(state)
        self.settings.setValue("stacking/star_trail_mode", self.star_trail_mode)
        # if they turn it on, immediately override the rejection combo:
        if self.star_trail_mode:
            self.rejection_algorithm = "Maximum Value"
        else:
            # reload whatever the user picked
            self.rejection_algorithm = self.settings.value("stacking/rejection_algorithm",
                                                          self.rejection_algorithm,
                                                          type=str)

    def save_master_paths_to_settings(self):
        """Save current master dark and flat paths to QSettings using their actual trees."""

        # Master Darks
        dark_paths = []
        for i in range(self.master_dark_tree.topLevelItemCount()):
            group = self.master_dark_tree.topLevelItem(i)
            for j in range(group.childCount()):
                fname = group.child(j).text(0)
                for path in self.master_files.values():
                    if os.path.basename(path) == fname:
                        dark_paths.append(path)

        # Master Flats
        flat_paths = []
        for i in range(self.master_flat_tree.topLevelItemCount()):
            group = self.master_flat_tree.topLevelItem(i)
            for j in range(group.childCount()):
                fname = group.child(j).text(0)
                for path in self.master_files.values():
                    if os.path.basename(path) == fname:
                        flat_paths.append(path)

        self.settings.setValue("stacking/master_darks", dark_paths)
        self.settings.setValue("stacking/master_flats", flat_paths)

    def clear_tree_selection(self, tree, file_dict):
        """Clears selected items from a simple (non-tuple-keyed) tree like Master Darks or Darks tab."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()
            if parent is None:
                # Top-level group item
                key = item.text(0)
                if key in file_dict:
                    del file_dict[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            else:
                # Child item
                key = parent.text(0)
                filename = item.text(0)
                if key in file_dict:
                    file_dict[key] = [f for f in file_dict[key] if os.path.basename(f) != filename]
                    if not file_dict[key]:
                        del file_dict[key]
                parent.removeChild(item)


    def clear_tree_selection_light(self, tree):
        """Clears the selection in the light tree and updates self.light_files accordingly."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()
            if parent is None:
                # Top-level filter node selected
                filter_name = item.text(0)
                # Remove all composite keys whose group_key starts with filter_name
                keys_to_remove = [key for key in list(self.light_files.keys())
                                if isinstance(key, tuple) and key[0].startswith(f"{filter_name} - ")]
                for key in keys_to_remove:
                    del self.light_files[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))
            else:
                if parent.parent() is None:
                    # Exposure node selected (child)
                    filter_name = parent.text(0)
                    exposure_text = item.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                    keys_to_remove = [key for key in list(self.light_files.keys())
                                    if isinstance(key, tuple) and key[0] == group_key]
                    for key in keys_to_remove:
                        del self.light_files[key]
                    parent.removeChild(item)
                else:
                    # Grandchild file node selected
                    filter_name = parent.parent().text(0)
                    exposure_text = parent.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                    filename = item.text(0)

                    keys_to_check = [key for key in list(self.light_files.keys())
                                    if isinstance(key, tuple) and key[0] == group_key]

                    for key in keys_to_check:
                        self.light_files[key] = [
                            f for f in self.light_files[key] if os.path.basename(f) != filename
                        ]
                        if not self.light_files[key]:
                            del self.light_files[key]
                    parent.removeChild(item)

        self.light_tab._refresh_light_tree_summaries()            

    def clear_tree_selection_flat(self, tree, file_dict):
        """Clears the selection in the given tree widget and removes items from the corresponding dictionary."""
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            parent = item.parent()

            if parent:
                # Grandchild level (actual file)
                if parent.parent() is not None:
                    filter_name = parent.parent().text(0)
                    exposure_text = parent.text(0)
                    group_key = f"{filter_name} - {exposure_text}"
                else:
                    # Exposure level
                    filter_name = parent.text(0)
                    exposure_text = item.text(0)
                    group_key = f"{filter_name} - {exposure_text}"

                filename = item.text(0)

                # Remove from all matching (group_key, session) tuples
                keys_to_check = [key for key in list(file_dict.keys())
                                if isinstance(key, tuple) and key[0] == group_key]

                for key in keys_to_check:
                    file_dict[key] = [f for f in file_dict[key] if os.path.basename(f) != filename]
                    if not file_dict[key]:
                        del file_dict[key]

                parent.removeChild(item)
            else:
                # Top-level (filter group) selected
                filter_name = item.text(0)
                keys_to_remove = [key for key in list(file_dict.keys())
                                if isinstance(key, tuple) and key[0].startswith(f"{filter_name} - ")]
                for key in keys_to_remove:
                    del file_dict[key]
                tree.takeTopLevelItem(tree.indexOfTopLevelItem(item))

    def _sync_group_userrole(self, top_item: QTreeWidgetItem):
        paths = []
        for i in range(top_item.childCount()):
            child = top_item.child(i)
            fp = child.data(0, Qt.ItemDataRole.UserRole)
            if fp:
                paths.append(fp)
        top_item.setData(0, Qt.ItemDataRole.UserRole, paths)

    def clear_tree_selection_registration(self, tree):
        """
        Remove selected rows from the Registration tree and *persist* those removals,
        so refreshes / 'Add Light Files' won't resurrect them.
        """
        selected_items = tree.selectedItems()
        if not selected_items:
            return

        # ensure attrs exist
        if not hasattr(self, "_reg_excluded_files"):
            self._reg_excluded_files = set()
        if not hasattr(self, "deleted_calibrated_files"):
            self.deleted_calibrated_files = []

        removed_paths = []

        for item in selected_items:
            parent = item.parent()

            if parent is None:
                # Top-level group
                group_key = item.text(0)

                # paths are stored on the group's UserRole
                full_paths = item.data(0, Qt.ItemDataRole.UserRole) or []
                for p in full_paths:
                    if isinstance(p, str):
                        removed_paths.append(p)

                # Keep internal dict in sync
                self.reg_files.pop(group_key, None)

                # Remove group row
                idx = tree.indexOfTopLevelItem(item)
                if idx >= 0:
                    tree.takeTopLevelItem(idx)

            else:
                # Leaf (single file)
                group_key = parent.text(0)
                fp = item.data(0, Qt.ItemDataRole.UserRole)

                # Track the absolute path if available, else fall back to name
                if isinstance(fp, str):
                    removed_paths.append(fp)
                else:
                    # fallback to name-based match (kept for backward compat)
                    filename = item.text(0)
                    removed_paths.append(filename)

                # Update reg_files
                if group_key in self.reg_files:
                    self.reg_files[group_key] = [
                        f for f in self.reg_files[group_key]
                        if f != fp and os.path.basename(f) != item.text(0)
                    ]
                    if not self.reg_files[group_key]:
                        del self.reg_files[group_key]

                # Remove leaf row
                parent.removeChild(item)

                # Keep parent's stored list in sync (your helper)
                self._sync_group_userrole(parent)

        # Persist the exclusions so they won't reappear on refresh
        self._reg_excluded_files.update(p for p in removed_paths if isinstance(p, str))

        # Maintain your legacy list too (if you still use it elsewhere)
        for p in removed_paths:
            if p not in self.deleted_calibrated_files:
                self.deleted_calibrated_files.append(p)

        # Also prune manual list so it doesn't re-inject removed files
        if hasattr(self, "manual_light_files") and self.manual_light_files:
            self.manual_light_files = [p for p in self.manual_light_files if p not in self._reg_excluded_files]

        # Optional but helpful: rebuild so empty groups disappear cleanly
        self.populate_calibrated_lights()
        self._refresh_reg_tree_summaries()

    def exposures_within_tolerance(self, exp1, exp2, tolerance):
        try:
            return abs(float(exp1) - float(exp2)) <= tolerance
            
        except Exception:
            return False

    def parse_group_key(self, group_key):
        """
        Parses a group key string like 'Luminance - 90s (3000x2000)'
        into filter_name, exposure (float), and image_size (str).
        """
        try:
            parts = group_key.split(' - ')
            filter_name = parts[0]
            exp_size_part = parts[1] if len(parts) > 1 else ""

            # Separate exposure and size correctly
            if '(' in exp_size_part and ')' in exp_size_part:
                exposure_str, size_part = exp_size_part.split('(', 1)
                exposure = exposure_str.replace('s', '').strip()
                size = size_part.strip(') ').strip()
            else:
                exposure = exp_size_part.replace('s', '').strip()
                size = "Unknown"

            
            return filter_name, float(exposure), size

        except Exception as e:
            
            return "Unknown", 0.0, "Unknown"

    def _get_image_size(self, fp):
        ext = os.path.splitext(fp)[1].lower()
        # first try FITS
        if ext in (".fits", ".fit"):
            hdr0 = fits.getheader(fp, ext=0)
            data0 = fits.getdata(fp, ext=0)
            h, w = data0.shape[-2:]
        else:
            # try Pillow
            try:
                with Image.open(fp) as img:
                    w, h = img.size
            except Exception:
                # Pillow failed on TIFF or exotic format → try tifffile
                try:
                    arr = tiff.imread(fp)
                    h, w = arr.shape[:2]
                except Exception:
                    # last resort: OpenCV
                    arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                    if arr is None:
                        raise IOError(f"Cannot read image size for {fp}")
                    h, w = arr.shape[:2]
        return w, h

    def _probe_fits_meta(self, fp: str):
        """
        Return (filter:str, exposure:float, size_str:'WxH') from FITS PRIMARY HDU.
        Robust to missing keywords.
        """
        try:
            hdr0 = fits.getheader(fp, ext=0)
            filt = self._sanitize_name(hdr0.get("FILTER", "Unknown"))
            # exposure can be EXPTIME or EXPOSURE, sometimes a string
            exp_raw = hdr0.get("EXPTIME", hdr0.get("EXPOSURE", 0.0))
            try:
                exp = float(exp_raw)
            except Exception:
                exp = 0.0
            # size
            data0 = fits.getdata(fp, ext=0)
            h, w = (int(data0.shape[-2]), int(data0.shape[-1])) if hasattr(data0, "shape") else (0, 0)
            size = f"{w}x{h}" if (w and h) else "Unknown"
            return filt or "Unknown", float(exp), size
        except Exception as e:
            print(f"⚠️ Could not read FITS {fp}: {e}; treating as generic image")
            return "Unknown", 0.0, "Unknown"


    def _probe_xisf_meta(self, fp: str):
        """
        Return (filter:str, exposure:float, size_str:'WxH') from XISF header.
        Uses first <Image> block; never loads full pixel data.
        """
        try:
            x = XISF(fp)
            ims = x.get_images_metadata()
            if not ims:
                return "Unknown", 0.0, "Unknown"
            m0 = ims[0]  # first image block
            # size from geometry tuple (w,h,channels)
            try:
                w, h, _ = m0.get("geometry", (0, 0, 0))
                size = f"{int(w)}x{int(h)}" if (w and h) else "Unknown"
            except Exception:
                size = "Unknown"

            # FITS-like keywords are stored in dict of lists: {"FILTER":[{"value":..., "comment":...}], ...}
            def _kw(name, default=None):
                lst = m0.get("FITSKeywords", {}).get(name)
                if lst and isinstance(lst, list) and lst[0] and "value" in lst[0]:
                    return lst[0]["value"]
                return default

            filt = self._sanitize_name(_kw("FILTER", "Unknown"))

            exp_raw = _kw("EXPTIME", None)
            if exp_raw is None:
                exp_raw = _kw("EXPOSURE", 0.0)
            try:
                exp = float(exp_raw)
            except Exception:
                exp = 0.0

            # bonus: sometimes exposure is in XISF properties (project dependent). Try a few common ids.
            if exp == 0.0:
                props = m0.get("XISFProperties", {}) or {}
                for pid in ("EXPTIME", "ExposureTime", "XISF:ExposureTime"):
                    v = props.get(pid, {}).get("value")
                    try:
                        exp = float(v)
                        break
                    except Exception:
                        pass

            return filt or "Unknown", float(exp), size
        except Exception as e:
            print(f"⚠️ Could not read XISF {fp}: {e}; treating as generic image")
            return "Unknown", 0.0, "Unknown"


    def populate_calibrated_lights(self):
        from PIL import Image

        def _fmt(enabled, scale, drop):
            return (f"Drizzle: True, Scale: {scale:g}x, Drop: {drop:.2f}" if enabled else "Drizzle: False")

        self.reg_tree.clear()
        self.reg_tree.setColumnCount(3)
        self.reg_tree.setHeaderLabels(["Filter - Exposure - Size", "Metadata", "Drizzle"])
        hdr = self.reg_tree.header()
        for col in (0, 1, 2):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)

        # gather files
        calibrated_folder = os.path.join(self.stacking_directory or "", "Calibrated")
        files = []
        if os.path.isdir(calibrated_folder):
            for fn in os.listdir(calibrated_folder):
                files.append(os.path.join(calibrated_folder, fn))

        # include manual files
        files += self.manual_light_files

        # filter exclusions + dedupe
        if self._reg_excluded_files:
            files = [f for f in files if f not in self._reg_excluded_files]
        files = list(dict.fromkeys(files))
        if not files:
            self.light_files = {}
            return

        # group by (filter, ~exposure, size) within tolerance
        grouped = {}  # key -> list of dicts: {"path", "exp", "size"}
        tol = self.exposure_tolerance_spin.value()

        for fp in files:
            ext = os.path.splitext(fp)[1].lower()
            filt = "Unknown"
            exp = 0.0
            size = "Unknown"

            if ext in (".fits", ".fit", ".fz"):
                filt, exp, size = self._probe_fits_meta(fp)
            elif ext == ".xisf":
                filt, exp, size = self._probe_xisf_meta(fp)
            else:
                # generic image (TIFF/PNG/JPEG, etc.)
                try:
                    w, h = self._get_image_size(fp)
                    size = f"{w}x{h}"
                except Exception as e:
                    print(f"⚠️ Cannot read image size for {fp}: {e}")
                    continue

            # find existing group with same filter+size and exposure within tolerance
            match_key = None
            for key in grouped:
                f2, e2, s2 = self.parse_group_key(key)
                if filt == f2 and s2 == size and abs(exp - e2) <= tol:
                    match_key = key
                    break

            key = match_key or f"{filt} - {exp:.1f}s ({size})"
            grouped.setdefault(key, []).append({"path": fp, "exp": exp, "size": size})

        # populate tree & self.light_files
        self.light_files = {}

        # current global drizzle defaults
        global_enabled = self.drizzle_checkbox.isChecked()
        try:
            global_scale = float(self.drizzle_scale_combo.currentText().replace("x", "", 1))
        except Exception:
            global_scale = 1.0
        global_drop = self.drizzle_drop_shrink_spin.value()

        for key, entries in grouped.items():
            paths = [d["path"] for d in entries]
            exps  = [d["exp"]  for d in entries]

            top = QTreeWidgetItem()
            top.setText(0, key)
            if len(exps) > 1:
                mn, mx = min(exps), max(exps)
                top.setText(1, f"{len(paths)} files, {mn:.0f}s–{mx:.0f}s")
            else:
                top.setText(1, f"{len(paths)} file")

            # per-group drizzle state (persisted), default to global
            state = self.per_group_drizzle.get(key)
            if state is None:
                state = {"enabled": bool(global_enabled), "scale": float(global_scale), "drop": float(global_drop)}
                self.per_group_drizzle[key] = state

            try:
                top.setText(2, self._format_drizzle_text(state["enabled"], state["scale"], state["drop"]))
            except AttributeError:
                top.setText(2, _fmt(state["enabled"], state["scale"], state["drop"]))

            top.setData(0, Qt.ItemDataRole.UserRole, paths)
            self.reg_tree.addTopLevelItem(top)

            # leaf rows: show basename + per-file size
            for d in entries:
                fp = d["path"]
                leaf = QTreeWidgetItem([os.path.basename(fp), f"Size: {d['size']}"])
                leaf.setData(0, Qt.ItemDataRole.UserRole, fp)
                top.addChild(leaf)

            top.setExpanded(True)
            self.light_files[key] = paths


    def _iter_group_items(self):
        for i in range(self.reg_tree.topLevelItemCount()):
            yield self.reg_tree.topLevelItem(i)

    def _format_drizzle_text(self, enabled: bool, scale: float, drop: float) -> str:
        return (f"Drizzle: True, Scale: {scale:g}x, Drop: {drop:.2f}"
                if enabled else "Drizzle: False")

    def _set_drizzle_on_items(self, items, enabled: bool, scale: float, drop: float):
        txt_on  = self._format_drizzle_text(True,  scale, drop)
        txt_off = self._format_drizzle_text(False, scale, drop)
        for it in items:
            # dedupe child selection → parent group
            if it.parent() is not None:
                it = it.parent()
            group_key = it.text(0)
            it.setText(2, txt_on if enabled else txt_off)
            self.per_group_drizzle[group_key] = {
                "enabled": bool(enabled),
                "scale": float(scale),
                "drop":  float(drop),
            }

    def update_drizzle_settings(self):
        """
        Called whenever the user toggles the 'Enable Drizzle' checkbox,
        changes the scale combo, or changes the drop shrink spinbox.
        Applies to all *selected* top-level items in the reg_tree.
        """
        # Current states from global controls
        drizzle_enabled = self.drizzle_checkbox.isChecked()
        scale_str = self.drizzle_scale_combo.currentText()  # e.g. "1x","2x","3x"
        drop_val = self.drizzle_drop_shrink_spin.value()    # e.g. 0.65

        # Gather selected items
        selected_items = self.reg_tree.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            # If the user selected a child row, go up to its parent group
            if item.parent() is not None:
                item = item.parent()

            group_key = item.text(0)

            if drizzle_enabled:
                # Show scale + drop shrink
                drizzle_text = (f"Drizzle: True, "
                                f"Scale: {scale_str}, "
                                f"Drop: {drop_val:.2f}")
            else:
                # Just show "Drizzle: False"
                drizzle_text = "Drizzle: False"

            # Update column 2 with the new text
            item.setText(2, drizzle_text)

            # If you also store it in a dictionary:
            self.per_group_drizzle[group_key] = {
                "enabled": drizzle_enabled,
                "scale": float(scale_str.replace("x","", 1)),
                "drop": drop_val
            }

    def _on_drizzle_param_changed(self, *_):
        enabled = self.drizzle_checkbox.isChecked()
        scale   = float(self.drizzle_scale_combo.currentText().replace("x","",1))
        drop    = self.drizzle_drop_shrink_spin.value()

        sel = self.reg_tree.selectedItems()
        if sel:
            # update selected groups
            seen, targets = set(), []
            for it in sel:
                top = it if it.parent() is None else it.parent()
                key = top.text(0)
                if key not in seen:
                    seen.add(key); targets.append(top)
        else:
            # no selection → update ALL groups (keeps UI intuitive)
            targets = list(self._iter_group_items())

        self._set_drizzle_on_items(targets, enabled, scale, drop)

    def gather_drizzle_settings_from_tree(self):
        """Return per-group drizzle settings based on the global controls."""
        enabled = bool(self.drizzle_checkbox.isChecked())
        scale_txt = self.drizzle_scale_combo.currentText()
        try:
            scale_factor = float(scale_txt.replace("x", "").strip())
        except Exception:
            scale_factor = 1.0
        drop_shrink = float(self.drizzle_drop_shrink_spin.value())

        out = {}
        root = self.reg_tree.invisibleRootItem()
        for i in range(root.childCount()):
            group_key = root.child(i).text(0)   # e.g. "L Ultimate - 300.0s (4144x2822)"
            out[group_key] = {
                "drizzle_enabled": enabled,
                "scale_factor": scale_factor,
                "drop_shrink": drop_shrink,
            }
        # Optional: debug once to verify
        self.update_status(f"🧪 drizzle_dict: {out}")
        return out



    def add_light_files_to_registration(self):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Frames", last_dir,
            "FITS Files (*.fits *.fit *.fz *.xisf *.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if not files:
            return

        self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))

        # Exclude files the user has removed previously
        new_files = [f for f in files if f not in self._reg_excluded_files]

        # Deduplicate while preserving order
        merged = list(dict.fromkeys(self.manual_light_files + new_files))
        self.manual_light_files = merged

        self.populate_calibrated_lights()
        self._refresh_reg_tree_summaries()


    def on_tab_changed(self, idx):
        try:
            if self.tabs.tabText(idx) == "Flats":
                self.update_override_dark_combo()
        except Exception:
            pass



    def prompt_session_before_adding(self, frame_type, directory_mode=False):
        # Respect auto-detect; do nothing here if auto is ON
        if self.settings.value("stacking/auto_session", True, type=bool):
            # Defer to the non-prompt paths
            if frame_type.upper() == "FLAT":
                if directory_mode:
                    self.add_directory(self.flat_tree, "Select Flat Directory", "FLAT")
                    self.assign_best_master_dark()
                    self.flat_tab.rebuild_flat_tree()
                else:
                    self.add_files(self.flat_tree, "Select Flat Files", "FLAT")
                    self.assign_best_master_dark()
                    self.flat_tab.rebuild_flat_tree()
            else:
                if directory_mode:
                    self.add_directory(self.light_tree, "Select Light Directory", "LIGHT")
                else:
                    self.add_files(self.light_tree, "Select Light Files", "LIGHT")
                self.assign_best_master_files()
            return

        # Manual session flow (auto OFF): ask once
        text, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session name:", text="Default")
        if not (ok and text.strip()):
            return
        self.current_session_tag = text.strip()

        if frame_type.upper() == "FLAT":
            if directory_mode:
                self.add_directory(self.flat_tree, "Select Flat Directory", "FLAT")
            else:
                self.add_files(self.flat_tree, "Select Flat Files", "FLAT")
            self.assign_best_master_dark()
            self.flat_tab.rebuild_flat_tree()
        else:
            if directory_mode:
                self.add_directory(self.light_tree, "Select Light Directory", "LIGHT")
            else:
                self.add_files(self.light_tree, "Select Light Files", "LIGHT")
            self.assign_best_master_files()


    def add_files(self, tree, title, expected_type):
        """ Adds FITS files and assigns best master files if needed. """
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(
            self, title, last_dir,
            "FITS Files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz)"
        )
        if not files:
            return

        self.settings.setValue("last_opened_folder", os.path.dirname(files[0]))

        # Show a standalone progress dialog while ingesting
        self._ingest_paths_with_progress(
            paths=files,
            tree=tree,
            expected_type=expected_type,
            title=f"Adding {expected_type.title()} Files…"
        )

        # Auto-assign after ingest (LIGHT only, same behavior you had)
        if expected_type.upper() == "LIGHT":
            busy = self._busy_progress("Assigning best Master Dark/Flat…")
            try:
                self.assign_best_master_files()
            finally:
                busy.close()

    def on_auto_session_toggled(self, checked: bool):
        self.settings.setValue("stacking/auto_session", bool(checked))
        self.sessionNameEdit.setEnabled(not checked)      # text box
        self.sessionNameButton.setEnabled(not checked)    # any “set session” button


    def add_directory(self, tree, title, expected_type):
        last_dir = self.settings.value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self, title, last_dir)
        if not directory:
            return
        self.settings.setValue("last_opened_folder", directory)

        recursive = self.settings.value("stacking/recurse_dirs", True, type=bool)
        paths = self._collect_fits_paths(directory, recursive=recursive)
        if not paths:
            return

        auto_session = self.settings.value("stacking/auto_session", True, type=bool)

        # ✅ Only build auto session map if auto is ON
        session_by_path = self._auto_tag_sessions_for_paths(paths) if auto_session else {}

        # ✅ Manual session name ONLY if auto is OFF
        manual_session_name = None
        if not auto_session:
            from PyQt6.QtWidgets import QInputDialog
            session_name, ok = QInputDialog.getText(self, "Session name",
                                                    "Enter a session name (e.g. 2024-10-09):")
            if ok and session_name.strip():
                manual_session_name = session_name.strip()

        # Ingest (tell it the manual_session_name – it may be None)
        self._ingest_paths_with_progress(
            paths=paths,
            tree=tree,
            expected_type=expected_type,
            title=f"Adding {expected_type.title()} from Directory…",
            manual_session_name=manual_session_name,   # NEW
        )

        # ✅ Apply auto tags to the UI only when auto is ON
        if auto_session:
            target_dict = self.flat_files if expected_type.upper() == "FLAT" else self.light_files
            self._apply_session_tags_to_tree(tree=tree, target_dict=target_dict, session_by_path=session_by_path)

        # As before…
        if expected_type.upper() == "LIGHT":
            busy = self._busy_progress("Assigning best Master Dark/Flat…")
            try:
                self.assign_best_master_files()
            finally:
                busy.close()
        elif expected_type.upper() == "FLAT":
            self.assign_best_master_dark()
            self.flat_tab.rebuild_flat_tree()



    # --- Directory walking ---------------------------------------------------------
    def _collect_fits_paths(self, root: str, recursive: bool = True) -> list[str]:
        exts = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fz")
        paths = []
        if recursive:
            for d, _subdirs, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(exts):
                        paths.append(os.path.join(d, f))
        else:
            for f in os.listdir(root):
                if f.lower().endswith(exts):
                    paths.append(os.path.join(root, f))
        # stable order (helps reproducibility + nice UX)
        paths.sort(key=lambda p: (os.path.dirname(p).lower(), os.path.basename(p).lower()))
        return paths


    # --- Session autodetect --------------------------------------------------------
    _SESSION_PATTERNS = [
        # "Night1", "night_02", "Session-3", "sess7"
        (re.compile(r"(session|sess|night|noche|nuit)[ _-]?(\d{1,2})", re.I),
        lambda m: f"Session-{int(m.group(2)):02d}"),

        # ISO-ish dates in folder names: 2024-10-09, 2024_10_09, 20241009
        (re.compile(r"\b(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)\b"),
        lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    ]

    def _auto_session_from_path(self, path: str, hdr=None) -> str:
        # 1) Prefer DATE-OBS → local night bucket
        if hdr:
            sess = self._session_from_dateobs_local_night(hdr)
            if sess:
                return sess

        # 2) Otherwise fall back to folder patterns (unchanged)
        parts = [p.lower() for p in Path(path).parts][:-1]
        for name in reversed(parts):
            for rx, fmt in _SESSION_PATTERNS:
                m = rx.search(name)
                if m:
                    return fmt(m)

        # 3) Parent folder name
        try:
            parent = Path(path).parent.name.strip()
            if parent:
                return re.sub(r"\s+", "_", parent)
        except Exception:
            pass

        return "Default"

    def _get_site_timezone(self, hdr=None):
        """
        Resolve a tz for 'local night' bucketing.

        Priority:
        1) QSettings key 'stacking/site_timezone' (e.g., 'America/Los_Angeles')
        2) FITS header TZ-like hints (TIMEZONE, TZ)
        3) System local timezone; finally, a best-effort local offset or UTC
        """
        # 1) App setting
        tz_name = self.settings.value("stacking/site_timezone", "", type=str) or ""
        try:
            from zoneinfo import ZoneInfo
        except Exception:
            ZoneInfo = None

        if tz_name and ZoneInfo:
            try:
                return ZoneInfo(tz_name)
            except Exception:
                pass

        # 2) From header
        if hdr:
            for key in ("TIMEZONE", "TZ"):
                if key in hdr and ZoneInfo:
                    try:
                        return ZoneInfo(str(hdr[key]).strip())
                    except Exception:
                        pass

        # 3) System local timezone
        if ZoneInfo:
            try:

                return ZoneInfo(str(tzlocal.get_localzone()))
            except Exception:
                pass

        # 3b) Fallback to the current local offset (aware tzinfo) or UTC
        try:
            now = dt_datetime.now().astimezone()
            return now.tzinfo or dt_timezone.utc
        except Exception:
            return dt_timezone.utc


    def _parse_date_obs(self, s: str):
        """
        Parse DATE-OBS robustly → aware UTC datetime if possible.
        Accepts ISO8601 with or without 'Z', fractional seconds, or date-only.
        """
        if not s:
            return None
        s = str(s).strip()

        # Prefer dateutil if present
        try:
            from dateutil import parser as date_parser
        except Exception:
            date_parser = None

        if date_parser:
            try:
                dt = date_parser.isoparse(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=dt_timezone.utc)
                return dt.astimezone(dt_timezone.utc)
            except Exception:
                pass

        # Manual fallbacks
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ",
                    "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d"):
            try:
                dt = dt_datetime.strptime(s, fmt)
                if fmt.endswith("Z") or "T" in fmt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=dt_timezone.utc)
                else:
                    # date-only; assume midday UTC to avoid accidental date flips
                    dt = dt.replace(tzinfo=dt_timezone.utc, hour=12)
                return dt.astimezone(dt_timezone.utc)
            except Exception:
                continue
        return None


    def _session_from_dateobs_local_night(self, hdr, cutoff_hour: int = None) -> str | None:
        """
        Return 'YYYY-MM-DD' session key by snapping DATE-OBS to local 'night'.

        Rule: if local_time < cutoff_hour → use previous calendar date.
        Default cutoff = self.settings('stacking/night_cutoff_hour', 12)  # noon
        """
        date_obs = hdr.get("DATE-OBS") if hdr else None
        dt_utc = self._parse_date_obs(date_obs) if date_obs else None
        if dt_utc is None:
            return None

        if cutoff_hour is None:
            cutoff_hour = int(self.settings.value("stacking/night_cutoff_hour", 12, type=int))

        tz = self._get_site_timezone(hdr)
        dt_local = dt_utc.astimezone(tz)

        # ✅ Use datetime.time alias, not the 'time' module
        cutoff = dt_time(hour=max(0, min(23, int(cutoff_hour))), minute=0)

        local_date = dt_local.date()
        if dt_local.time() < cutoff:
            local_date = (dt_local - dt_timedelta(days=1)).date()

        return local_date.isoformat()


    def _auto_tag_sessions_for_paths(self, paths: list[str]) -> dict[str, str]:
        session_map = {}
        for p in paths:
            try:
                hdr, _ = get_valid_header(p)
            except Exception:
                hdr = None
            session_map[p] = self._auto_session_from_path(p, hdr)           # ✅ self.
        return session_map


    # --- Apply session tags to your internal dicts and tree -----------------------
    def _apply_session_tags_to_tree(self, *, tree: QTreeWidget, target_dict: dict, session_by_path: dict[str,str]):
        """
        After ingest, set self.session_tags[path] and update the UI 'Metadata' column
        to include 'Session: <tag>'. Also move items across the (group_key, session) buckets.
        """
        # Build reverse index: basename -> full path we just tagged
        by_base = {os.path.basename(p): p for p in session_by_path.keys()}

        def _update_leaf(leaf: QTreeWidgetItem):
            base = leaf.text(0)
            full = by_base.get(base)
            if not full:
                return
            sess = session_by_path.get(full, "Default")
            self.session_tags[full] = sess

            # Update the metadata column
            old_meta = leaf.text(1) or ""
            if "Session:" in old_meta:
                new_meta = re.sub(r"Session:\s*[^|]*", f"Session: {sess}", old_meta)
            else:
                new_meta = (old_meta + (" | " if old_meta else "") + f"Session: {sess}")
            leaf.setText(1, new_meta)

            # Move file into (group_key, session) bucket in target_dict
            # group_key currently your "Filter & Exposure" string for that leaf's top-level parent
            # Ascend to find the top-level "Filter & Exposure" node:
            parent = leaf.parent()
            while parent and parent.parent():
                parent = parent.parent()
            group_key = parent.text(0) if parent else "Unknown"

            # locate and move inside target_dict
            for key in list(target_dict.keys()):
                files = target_dict.get(key, [])
                if full in files:
                    old_session = key[1] if (isinstance(key, tuple) and len(key) == 2) else "Default"
                    if old_session != sess:
                        # remove from old bucket
                        files.remove(full)
                        if not files:
                            target_dict.pop(key, None)
                        new_key = (key[0] if isinstance(key, tuple) else group_key, sess)
                        target_dict.setdefault(new_key, []).append(full)
                    break

        # Walk all leaves
        root_count = tree.topLevelItemCount()
        for i in range(root_count):
            g = tree.topLevelItem(i)
            for j in range(g.childCount()):
                e = g.child(j)
                for k in range(e.childCount()):
                    leaf = e.child(k)
                    if leaf.childCount() == 0:
                        _update_leaf(leaf)

    def _get_file_dt_cache(self):
        # lazy init
        cache = getattr(self, "_file_dt_cache", None)
        if cache is None:
            cache = {}
            self._file_dt_cache = cache
        return cache

    def _file_local_night_date(self, path: str, hdr=None):
        """
        Return a date() representing the local 'night' for this file,
        using DATE-OBS and the app/site timezone & cutoff logic you already have.
        """
        cache = self._get_file_dt_cache()
        if path in cache:
            return cache[path]

        try:
            if hdr is None:
                with fits.open(path, memmap=True) as hdul:
                    hdr = hdul[0].header
        except Exception:
            hdr = None

        sess = self._session_from_dateobs_local_night(hdr) if hdr else None
        # sess is 'YYYY-MM-DD' or None
        import datetime as _dt
        d = _dt.date.fromisoformat(sess) if sess else None
        cache[path] = d
        return d

    def _closest_flat_for(self, *, filter_name: str, image_size: str, light_path: str):
        """
        Choose a flat master path by:
        1) exact key 'Filter (WxH) [Session]' matching light's session (if present)
        2) else among 'Filter (WxH) ...', pick by nearest local-night date to light.
        Returns path or None.
        """
        # normalize filter token the same way you build master keys
        ftoken = self._sanitize_name(filter_name)
        # light's date
        light_d = self._file_local_night_date(light_path)

        # Collect candidate flats (same filter+size)
        candidates = []
        for key, path in self.master_files.items():
            # flats often keyed like "Filter (WxH) [Session]" or "Filter (WxH)"
            if (ftoken in key) and (f"({image_size})" in key):
                candidates.append((key, path))

        if not candidates:
            return None

        # 1) exact session match if we can infer the light's session name
        light_hdr = None
        try:
            with fits.open(light_path, memmap=True) as hdul:
                light_hdr = hdul[0].header
        except Exception:
            pass
        light_session = self._session_from_dateobs_local_night(light_hdr) if light_hdr else None

        if light_session:
            exact_key = f"{ftoken} ({image_size}) [{light_session}]"
            for key, path in candidates:
                if key == exact_key:
                    return path

        # 2) otherwise pick by nearest date (local night)
        if light_d is None:
            # No DATE-OBS for light → fall back to first candidate
            return candidates[0][1]

        best = (None, 10**9)  # (path, |Δdays|)
        for _key, fpath in candidates:
            fd = self._file_local_night_date(fpath)
            if fd is None:
                continue
            delta = abs((fd - light_d).days)
            if delta < best[1]:
                best = (fpath, delta)

        return best[0] if best[0] else candidates[0][1]


    def _ingest_paths_with_progress(self, paths, tree, expected_type, title, manual_session_name=None):
        """
        Show a small standalone progress dialog while ingesting headers,
        with cancel support. Keeps UI responsive via processEvents().
        """
        total = len(paths)
        dlg = QProgressDialog(title, "Cancel", 0, total, self)
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoReset(True)
        dlg.setAutoClose(True)
        dlg.setValue(0)

        added = 0
        for i, path in enumerate(paths, start=1):
            if dlg.wasCanceled():
                break

            try:
                base = os.path.basename(path)
                dlg.setLabelText(f"{base}  ({i}/{total})")
                # Process events so the dialog repaints & remains responsive
                QCoreApplication.processEvents()
                self.process_fits_header(path, tree, expected_type, manual_session_name=manual_session_name)
                added += 1
            except Exception as e:
                # Optional: log or show a brief error — keep going
                # print(f"Failed to add {path}: {e}")
                pass

            dlg.setValue(i)
            QCoreApplication.processEvents()

        # Make sure it closes
        dlg.setValue(total)
        QCoreApplication.processEvents()

        try:
            self._report_group_summary(expected_type)
        except Exception:
            pass

        try:
            if (expected_type or "").upper() == "LIGHT":
                self.light_tab._refresh_light_tree_summaries()
        except Exception:
            pass

        # Optional: brief status line (non-intrusive)
        try:
            if expected_type.upper() == "LIGHT":
                self.statusBar().showMessage(f"Added {added}/{total} Light frames", 3000)
        except Exception:
            pass

    def _fmt_hms(self, seconds: float) -> str:
        s = int(round(max(0.0, float(seconds))))
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        if h: return f"{h}h {m}m {s}s"
        if m: return f"{m}m {s}s"
        return f"{s}s"

    def _exposure_from_label(self, label: str) -> float | None:
        """
        Extract exposure in seconds from labels like:
        "300s (6264x4180)"
        "Luminance - 300s (6264x4180)"
        "Ha - 0.5s (3000x2000)"
        Returns float(seconds) or None if not found.
        """
        if not label:
            return None

        # Look for the first number followed by optional space and 's'
        m = re.search(r"([+-]?\d+(?:\.\d*)?)\s*s\b", label)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None



    def _busy_progress(self, text):
        """
        Returns a modal, indeterminate QProgressDialog you can open during
        short post-steps (e.g., assigning masters). Caller must .close().
        """
        dlg = QProgressDialog(text, None, 0, 0, self)
        dlg.setWindowTitle("Please wait")
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.show()
        QCoreApplication.processEvents()
        return dlg


    def _sanitize_name(self, name: str) -> str:
        """
        Replace any character that isn’t a letter, digit, space, dash or underscore
        with an underscore so it’s safe to use in filenames, dict-keys, tree labels, etc.
        """
        return re.sub(r"[^\w\s\-]", "_", name)
    
    def process_fits_header(self, path, tree, expected_type, manual_session_name=None):
        try:
            # --- Read header only (fast) ---
            header, _ = get_valid_header(path)   # FIX: use 'path', not 'file_path'

            # --- Basic image size ---
            try:
                width = int(header.get("NAXIS1"))
                height = int(header.get("NAXIS2"))
                image_size = f"{width}x{height}"
            except Exception as e:
                self.update_status(f"Warning: Could not read dimensions for {os.path.basename(path)}: {e}")
                width = height = None
                image_size = "Unknown"

            # --- Image type & exposure ---
            imagetyp = str(header.get("IMAGETYP", "UNKNOWN")).lower()

            exp_val = header.get("EXPOSURE")
            if exp_val is None:
                exp_val = header.get("EXPTIME")
            if exp_val is None:
                exposure_text = "Unknown"
            else:
                try:
                    # canonical like "300s" or "0.5s"
                    fexp = float(exp_val)
                    # use :g for tidy formatting (no trailing .0 unless needed)
                    exposure_text = f"{fexp:g}s"
                except Exception:
                    exposure_text = str(exp_val)

            # --- Mismatch prompt (respects "Yes to all / No to all") ---
            # --- Mismatch prompt (redirect/keep/skip with 'apply to all') ---
            if expected_type.upper() == "DARK":
                forbidden = ["light", "flat"]
            elif expected_type.upper() == "FLAT":
                forbidden = ["dark", "light"]
            elif expected_type.upper() == "LIGHT":
                forbidden = ["dark", "flat"]
            else:
                forbidden = []

            actual_type = self._guess_type_from_imagetyp(header.get("IMAGETYP"))
            decision_key = (expected_type.upper(), actual_type or "UNKNOWN")

            # Respect prior "Yes to all/No to all" style cache first
            # (compat with your existing decision_attr if you want)
            decision_attr = f"auto_confirm_{expected_type.lower()}"
            if hasattr(self, decision_attr):
                decision = getattr(self, decision_attr)
                if decision is False:
                    return
                # if True, keep going as-is (legacy behavior)

            # New logic: if actual looks forbidden, propose options
            if (actual_type is not None) and (actual_type.lower() in forbidden):
                # has the user already decided during this ingest burst?
                cached = self._mismatch_policy.get(decision_key)
                if cached == "skip":
                    return
                elif cached == "redirect":
                    # Re-route to the proper tab
                    dst_tree = self._tree_for_type(actual_type)
                    if dst_tree is not None:
                        # Recursively ingest into the correct type & tree
                        self.process_fits_header(path, dst_tree, actual_type, manual_session_name=manual_session_name)
                        return
                    # if no tree (shouldn't happen), fall through to prompt

                elif cached == "keep":
                    pass  # keep going in current tab

                else:
                    # Prompt the user
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Mismatched Image Type")
                    pretty_actual = actual_type or "Unknown"
                    msg.setText(
                        f"Found '{os.path.basename(path)}' with IMAGETYP = {header.get('IMAGETYP')}\n"
                        f"which looks like **{pretty_actual}**, not {expected_type}.\n\n"
                        f"What would you like to do?"
                    )
                    btn_redirect = msg.addButton(f"Send to {pretty_actual.title()} tab", QMessageBox.ButtonRole.YesRole)
                    btn_keep     = msg.addButton(f"Add to {expected_type.title()} tab", QMessageBox.ButtonRole.YesRole)
                    btn_cancel   = msg.addButton("Skip file", QMessageBox.ButtonRole.RejectRole)

                    # “Apply to all” for the rest of this add session
                    from PyQt6.QtWidgets import QCheckBox
                    apply_all_cb = QCheckBox("Apply this choice to all remaining mismatches of this kind")
                    msg.setCheckBox(apply_all_cb)

                    msg.exec()
                    clicked = msg.clickedButton()
                    apply_all = bool(apply_all_cb.isChecked())

                    if clicked is btn_cancel:
                        choice = "skip"
                    elif clicked is btn_redirect:
                        choice = "redirect"
                    else:
                        choice = "keep"

                    if apply_all:
                        self._mismatch_policy[decision_key] = choice

                    if choice == "skip":
                        return
                    if choice == "redirect":
                        dst_tree = self._tree_for_type(actual_type)
                        if dst_tree is not None:
                            self.process_fits_header(path, dst_tree, actual_type, manual_session_name=manual_session_name)
                            return

            # --- Resolve session tag (auto vs manual vs legacy current_session_tag) ---
            auto_session = self.settings.value("stacking/auto_session", True, type=bool)
            if manual_session_name:                           # only passed when auto is OFF and user typed one
                session_tag = manual_session_name.strip()
            elif auto_session:
                session_tag = self._auto_session_from_path(path, header) or "Default"
            else:
                session_tag = getattr(self, "current_session_tag", "Default")

            # --- Common helpers ---
            filter_name_raw = header.get("FILTER", "Unknown")
            filter_name     = self._sanitize_name(filter_name_raw)

            # === DARKs ===
            if expected_type.upper() == "DARK":
                key = f"{exposure_text} ({image_size})"
                self.dark_files.setdefault(key, []).append(path)

                # Tree: top-level = key; child = file
                items = tree.findItems(key, Qt.MatchFlag.MatchExactly, 0)
                exposure_item = items[0] if items else QTreeWidgetItem([key])
                if not items:
                    tree.addTopLevelItem(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                leaf = QTreeWidgetItem([os.path.basename(path), metadata])
                leaf.setData(0, Qt.ItemDataRole.UserRole, path)  # store full path
                exposure_item.addChild(leaf)

            # === FLATs ===
            elif expected_type.upper() == "FLAT":
                # internal dict keying (group by filter+exp+size, partition by session)
                flat_key = f"{filter_name} - {exposure_text} ({image_size})"
                composite_key = (flat_key, session_tag)
                self.flat_files.setdefault(composite_key, []).append(path)
                self.session_tags[path] = session_tag

                # Tree: top-level = filter; second = "exp (WxH)"; leaf = file
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                filter_item = filter_items[0] if filter_items else QTreeWidgetItem([filter_name])
                if not filter_items:
                    tree.addTopLevelItem(filter_item)

                want_label = f"{exposure_text} ({image_size})"
                exposure_item = None
                for i in range(filter_item.childCount()):
                    if filter_item.child(i).text(0) == want_label:
                        exposure_item = filter_item.child(i); break
                if exposure_item is None:
                    exposure_item = QTreeWidgetItem([want_label])
                    filter_item.addChild(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                leaf = QTreeWidgetItem([os.path.basename(path), metadata])
                leaf.setData(0, Qt.ItemDataRole.UserRole, path)  # store full path
                exposure_item.addChild(leaf)

            # === LIGHTs ===
            elif expected_type.upper() == "LIGHT":
                light_key = f"{filter_name} - {exposure_text} ({image_size})"
                composite_key = (light_key, session_tag)
                self.light_files.setdefault(composite_key, []).append(path)
                self.session_tags[path] = session_tag

                # Tree: top-level = filter; second = "exp (WxH)"; leaf = file
                filter_items = tree.findItems(filter_name, Qt.MatchFlag.MatchExactly, 0)
                filter_item = filter_items[0] if filter_items else QTreeWidgetItem([filter_name])
                if not filter_items:
                    tree.addTopLevelItem(filter_item)

                want_label = f"{exposure_text} ({image_size})"
                exposure_item = None
                for i in range(filter_item.childCount()):
                    if filter_item.child(i).text(0) == want_label:
                        exposure_item = filter_item.child(i); break
                if exposure_item is None:
                    exposure_item = QTreeWidgetItem([want_label])
                    filter_item.addChild(exposure_item)

                metadata = f"Size: {image_size} | Session: {session_tag}"
                leaf = QTreeWidgetItem([os.path.basename(path), metadata])
                leaf.setData(0, Qt.ItemDataRole.UserRole, path)  # ✅ needed for date-aware flat fallback
                exposure_item.addChild(leaf)

            # --- Done ---
            self.update_status(f"✅ Added {os.path.basename(path)} as {expected_type}")
            QApplication.processEvents()

        except Exception as e:
            self.update_status(f"❌ ERROR: Could not read FITS header for {os.path.basename(path)} - {e}")
            QApplication.processEvents()



    def add_master_files(self, tree, file_type, files):
        """ 
        Adds multiple master calibration files to the correct treebox with metadata including image dimensions.
        This version only reads the FITS header to extract image dimensions, making it much faster.
        """
        for file_path in files:
            try:
                # Read only the FITS header (fast)
                header = fits.getheader(file_path)
                
                # Check for both EXPOSURE and EXPTIME
                exposure = header.get("EXPOSURE", header.get("EXPTIME", "Unknown"))
                filter_name = header.get("FILTER", "Unknown")
                filter_name     = self._sanitize_name(filter_name)
                # Extract image dimensions from header keywords NAXIS1 and NAXIS2
                width = header.get("NAXIS1")
                height = header.get("NAXIS2")
                if width is not None and height is not None:
                    image_size = f"{width}x{height}"
                else:
                    image_size = "Unknown"
                
                # Construct key based on file type
                if file_type.upper() == "DARK":
                    key = f"{exposure}s ({image_size})"
                    self.master_files[key] = file_path  # Store master dark
                    self.master_sizes[file_path] = image_size  # Store size
                elif file_type.upper() == "FLAT":
                    # Attempt to extract session name from filename
                    session_name = "Default"
                    filename = os.path.basename(file_path)
                    if filename.lower().startswith("masterflat_"):
                        parts = filename.split("_")
                        if len(parts) > 1:
                            session_name = parts[1]

                    key = f"{filter_name} ({image_size}) [{session_name}]"
                    self.master_files[key] = file_path
                    self.master_sizes[file_path] = image_size

                # Extract additional metadata from header.
                sensor_temp = header.get("CCD-TEMP", "N/A")
                date_obs = header.get("DATE-OBS", "Unknown")
                metadata = f"Size: {image_size}, Temp: {sensor_temp}°C, Date: {date_obs}"

                # Check if category item already exists in the tree.
                items = tree.findItems(key, Qt.MatchFlag.MatchExactly, 0)
                if not items:
                    item = QTreeWidgetItem([key])
                    tree.addTopLevelItem(item)
                else:
                    item = items[0]

                # Add the master file as a child node with metadata.
                item.addChild(QTreeWidgetItem([os.path.basename(file_path), metadata]))

                print(f"✅ DEBUG: Added Master {file_type} -> {file_path} under {key} with metadata: {metadata}")
                self.update_status(f"✅ Added Master {file_type} -> {file_path} under {key} with metadata: {metadata}")
                print(f"📂 DEBUG: Master Files Stored: {self.master_files}")
                self.update_status(f"📂 DEBUG: Master Files Stored: {self.master_files}")
                QApplication.processEvents()
                self.assign_best_master_files()

            except Exception as e:
                print(f"❌ ERROR: Failed to load master file {file_path} - {e}")
                self.update_status(f"❌ ERROR: Failed to load master file {file_path} - {e}")
                QApplication.processEvents()

    def assign_best_master_dark(self):
        """ Assigns the closest matching master dark based on exposure & image size. """
        print("\n🔍 DEBUG: Assigning best master darks to flats...\n")

        if not self.master_files:
            print("⚠️ WARNING: No Master Darks available.")
            self.update_status("⚠️ WARNING: No Master Darks available.")
            return  # Exit early if there are no master darks

        print(f"📂 Loaded Master Darks ({len(self.master_files)} total):")
        for key, value in self.master_files.items():
            print(f"   📌 {key} -> {value}")

        # Iterate through all flat filters
        for i in range(self.flat_tree.topLevelItemCount()):
            filter_item = self.flat_tree.topLevelItem(i)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)  # Example: "0.0007s (8288x5644)"

                # Extract exposure time
                match = re.match(r"([\d.]+)s?", exposure_text)
                if not match:
                    print(f"⚠️ WARNING: Could not parse exposure time from {exposure_text}")
                    continue  # Skip if exposure is invalid

                exposure_time = float(match.group(1))  # Extracted number
                print(f"🟢 Checking Flat Group: {exposure_text} (Parsed: {exposure_time}s)")

                # Extract image size from metadata
                if exposure_item.childCount() > 0:
                    metadata_text = exposure_item.child(0).text(1)  # Metadata column
                    size_match = re.search(r"Size: (\d+x\d+)", metadata_text)
                    image_size = size_match.group(1) if size_match else "Unknown"
                else:
                    image_size = "Unknown"

                print(f"✅ Parsed Flat Size: {image_size}")

                # Find the best matching master dark
                best_match = None
                best_diff = float("inf")

                for master_dark_exposure, master_dark_path in self.master_files.items():
                    master_dark_exposure_match = re.match(r"([\d.]+)s?", master_dark_exposure)
                    if not master_dark_exposure_match:
                        continue  # Skip if master dark exposure is invalid

                    master_dark_exposure_time = float(master_dark_exposure_match.group(1))
                    master_dark_size = self.master_sizes.get(master_dark_path, "Unknown")
                    if master_dark_size == "Unknown":
                        with fits.open(master_dark_path) as hdul:
                            master_dark_size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                            self.master_sizes[master_dark_path] = master_dark_size  # ✅ Store it

                    print(f"🔎 Comparing with Master Dark: {master_dark_exposure_time}s ({master_dark_size})")

                    # Match both image size and exposure time
                    if image_size == master_dark_size:
                        diff = abs(master_dark_exposure_time - exposure_time)
                        if diff < best_diff:
                            best_match = master_dark_path
                            best_diff = diff

                # Assign best match in column 3
                if best_match:
                    exposure_item.setText(2, os.path.basename(best_match))
                    print(f"🔵 Assigned Master Dark: {os.path.basename(best_match)}")
                else:
                    exposure_item.setText(2, "None")
                    print(f"⚠️ No matching Master Dark found for {exposure_text}")

        # 🔥 Force UI update to reflect changes
        self.flat_tree.viewport().update()

        print("\n✅ DEBUG: Finished assigning best matching Master Darks to Flats.\n")



    def update_override_dark_combo(self):
        """Populate the dropdown with available Master Darks (from self.master_files)."""
        if not hasattr(self, "override_dark_combo"):
            return

        self.override_dark_combo.blockSignals(True)
        try:
            self.override_dark_combo.clear()
            self.override_dark_combo.addItem("None (Use Auto-Select)", userData=None)
            self.override_dark_combo.addItem("None (Use no Dark to Calibrate)", userData="__NO_DARK__")

            seen = set()
            for key, path in (self.master_files or {}).items():
                fn = os.path.basename(path or "")
                if not fn:
                    continue
                # Keep only dark-like files (bias-as-dark is fine)
                if ("masterdark" in fn.lower()) or fn.lower().startswith(("masterbias", "master_bias")):
                    if path and os.path.exists(path) and fn not in seen:
                        self.override_dark_combo.addItem(fn, userData=path)
                        seen.add(fn)
            print(f"✅ DEBUG: override_dark_combo items = {self.override_dark_combo.count()}")
        finally:
            self.override_dark_combo.blockSignals(False)



    def override_selected_master_dark_for_flats(self, idx: int):
        """Apply combo choice to selected flat groups; stores path/token in row + dict."""
        items = self.flat_tree.selectedItems()
        if not items:
            return

        # read combo selection (userData carries the path or sentinel)
        ud = self.override_dark_combo.itemData(idx)   # None | "__NO_DARK__" | "/path/to/MasterDark_..."
        txt = self.override_dark_combo.currentText()

        disp = ("Auto" if ud is None else
                "No Calibration" if ud == "__NO_DARK__" else
                os.path.basename(str(ud)))

        for it in items:
            # ensure we’re on a group row (no parent)
            if it.parent():
                continue
            gk = it.data(0, Qt.ItemDataRole.UserRole)
            if not gk:
                continue
            it.setText(2, disp)
            it.setData(2, Qt.ItemDataRole.UserRole, ud)
            self.flat_dark_override[gk] = ud

        print(f"✅ Override Master Dark applied → {disp}")



    def assign_best_master_files(self, fill_only: bool = True):
        """
        Assign best matching Master Dark and Flat to each Light leaf.
        - Honors manual overrides (never ignored).
        - If fill_only is True, do NOT overwrite non-empty cells.
        """
        print("\n🔍 DEBUG: Assigning best Master Darks & Flats to Lights...\n")

        if not getattr(self, "master_files", None):
            print("⚠️ WARNING: No Master Calibration Files available.")
            self.update_status("⚠️ WARNING: No Master Calibration Files available.")
            return

        # Ensure override dicts exist
        dark_over = getattr(self, "manual_dark_overrides", {}) or {}
        flat_over = getattr(self, "manual_flat_overrides", {}) or {}
        master_sizes = getattr(self, "master_sizes", {})
        self.master_sizes = master_sizes  # keep cache alive

        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            filter_name_raw = filter_item.text(0)
            filter_name = self._sanitize_name(filter_name_raw)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)  # e.g. "300.0s (4144x2822)"

                # Parse exposure seconds (for dark matching)
                mexp = re.match(r"([\d.]+)s?", exposure_text or "")
                exposure_time = float(mexp.group(1)) if mexp else 0.0

                for k in range(exposure_item.childCount()):
                    leaf_item = exposure_item.child(k)
                    meta_text = leaf_item.text(1) or ""

                    # Parse size & session from metadata column
                    msize = re.search(r"Size:\s*(\d+x\d+)", meta_text)
                    image_size = msize.group(1) if msize else "Unknown"
                    msess = re.search(r"Session:\s*([^|]+)", meta_text)
                    session_name = (msess.group(1).strip() if msess else "Default")

                    # Current cells (so we can skip if fill_only)
                    curr_dark = (leaf_item.text(2) or "").strip()
                    curr_flat = (leaf_item.text(3) or "").strip()

                    # ---------- DARK RESOLUTION ----------
                    # 1) Manual overrides: prefer "Filter - exposure" then bare exposure
                    dark_key_full  = f"{filter_name_raw} - {exposure_text}"
                    dark_key_short = exposure_text
                    dark_override  = dark_over.get(dark_key_full) or dark_over.get(dark_key_short)

                    if dark_override:
                        dark_choice = os.path.basename(dark_override)
                    else:
                        # 2) If fill_only and cell already nonempty & not "None", keep it
                        if fill_only and curr_dark and curr_dark.lower() != "none":
                            dark_choice = curr_dark
                        else:
                            # 3) Auto-pick by size+closest exposure
                            best_dark_match = None
                            best_dark_diff = float("inf")
                            for master_key, master_path in self.master_files.items():
                                dmatch = re.match(r"^([\d.]+)s\b", master_key)  # darks start with "<exp>s"
                                if not dmatch:
                                    continue
                                master_dark_exposure_time = float(dmatch.group(1))

                                # Ensure size known/cached
                                md_size = master_sizes.get(master_path)
                                if not md_size:
                                    try:
                                        with fits.open(master_path) as hdul:
                                            md_size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                                    except Exception:
                                        md_size = "Unknown"
                                    master_sizes[master_path] = md_size

                                if md_size == image_size:
                                    diff = abs(master_dark_exposure_time - exposure_time)
                                    if diff < best_dark_diff:
                                        best_dark_diff = diff
                                        best_dark_match = master_path

                            dark_choice = os.path.basename(best_dark_match) if best_dark_match else ("None" if not curr_dark else curr_dark)

                    # ---------- FLAT RESOLUTION ----------
                    flat_key_full  = f"{filter_name_raw} - {exposure_text}"
                    flat_key_short = exposure_text
                    flat_override  = flat_over.get(flat_key_full) or flat_over.get(flat_key_short)

                    if flat_override:
                        flat_choice = os.path.basename(flat_override)
                    else:
                        if fill_only and curr_flat and curr_flat.lower() != "none":
                            flat_choice = curr_flat
                        else:
                            # Get the full path of the light leaf (we stored it during ingest)
                            light_path = leaf_item.data(0, Qt.ItemDataRole.UserRole)
                            # Prefer exact session match; otherwise nearest-night fallback
                            best_flat_path = None

                            # Fast exact-key path
                            exact_key = f"{filter_name} ({image_size}) [{session_name}]"
                            if exact_key in self.master_files:
                                best_flat_path = self.master_files[exact_key]
                            else:
                                # Date-aware fallback across same filter+size
                                best_flat_path = self._closest_flat_for(
                                    filter_name=filter_name,
                                    image_size=image_size,
                                    light_path=light_path
                                )

                            flat_choice = os.path.basename(best_flat_path) if best_flat_path else ("None" if not curr_flat else curr_flat)

                    # ---------- WRITE CELLS ----------
                    leaf_item.setText(2, dark_choice)
                    leaf_item.setText(3, flat_choice)

                    print(f"📌 Assigned to {leaf_item.text(0)} -> Dark: {leaf_item.text(2)}, Flat: {leaf_item.text(3)}")

        self.light_tree.viewport().update()
        print("\n✅ DEBUG: Finished assigning Master Files per leaf.\n")


    def update_light_corrections(self):
        """ Updates the light frame corrections when checkboxes change. """
        corrections = []
        if self.cosmetic_checkbox.isChecked():
            corrections.append("Cosmetic: True")
        else:
            corrections.append("Cosmetic: False")

        if self.pedestal_checkbox.isChecked():
            corrections.append("Pedestal: True")
        else:
            corrections.append("Pedestal: False")

        if self.bias_checkbox.isChecked():
            # Show file dialog to select a Master Bias
            bias_file, _ = QFileDialog.getOpenFileName(self, "Select Master Bias Frame", "", "FITS Files (*.fits *.fit)")
            if bias_file:
                self.master_files["Bias"] = bias_file  # ✅ Store bias path
                corrections.append(f"Bias: {os.path.basename(bias_file)}")
            else:
                self.bias_checkbox.setChecked(False)  # If no file selected, uncheck
                return

        # Update all rows
        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_item.setText(4, ", ".join(corrections))

    def set_session_tag_for_group(self, item):
        """
        Prompt the user to assign a session tag to all frames in this group.
        """
        session_name, ok = QInputDialog.getText(self, "Set Session Tag", "Enter session label (e.g., Night1, RedFilterSet2):")
        if not ok or not session_name.strip():
            return

        session_name = session_name.strip()
        filter_name = item.text(0)

        for i in range(item.childCount()):
            exposure_item = item.child(i)
            exposure_label = exposure_item.text(0)

            # Update metadata text
            if exposure_item.childCount() > 0:
                metadata_item = exposure_item.child(0)
                metadata_text = metadata_item.text(1)
                metadata_text = re.sub(r"Session: [^|]+", f"Session: {session_name}", metadata_text)
                if "Session:" not in metadata_text:
                    metadata_text += f" | Session: {session_name}"
                metadata_item.setText(1, metadata_text)

            # Update internal session tag mapping
            composite_key = (f"{filter_name} - {exposure_label}", session_name)
            original_key = f"{filter_name} - {exposure_label}"

            if original_key in self.light_files:
                self.light_files[composite_key] = self.light_files.pop(original_key)

                for path in self.light_files[composite_key]:
                    self.session_tags[path] = session_name

        self.update_status(f"🟢 Assigned session '{session_name}' to group '{filter_name}'")


    def override_selected_master_dark(self):
        """ Override Dark for selected Light exposure group or individual files. """
        selected_items = self.light_tree.selectedItems()
        if not selected_items:
            print("⚠️ No light item selected for dark frame override.")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Master Dark", "", "FITS Files (*.fits *.fit)")
        if not file_path:
            return

        for item in selected_items:
            # If the user clicked a group (exposure row), push override to all leaves:
            if item.parent() and item.childCount() > 0:
                # exposure row under a filter
                filter_name = item.parent().text(0)
                exposure_text = item.text(0)
                # store override under BOTH keys
                self.manual_dark_overrides[f"{filter_name} - {exposure_text}"] = file_path
                self.manual_dark_overrides[exposure_text] = file_path

                for i in range(item.childCount()):
                    leaf = item.child(i)
                    leaf.setText(2, os.path.basename(file_path))
            # If the user clicked a leaf, just set that leaf and still store under both keys
            elif item.parent() and item.parent().parent():
                exposure_item = item.parent()
                filter_name = exposure_item.parent().text(0)
                exposure_text = exposure_item.text(0)
                self.manual_dark_overrides[f"{filter_name} - {exposure_text}"] = file_path
                self.manual_dark_overrides[exposure_text] = file_path
                item.setText(2, os.path.basename(file_path))

        print("✅ DEBUG: Light Dark override applied.")

    def _auto_pick_master_dark(self, image_size: str, exposure_time: float):
        best_path, best_diff = None, float("inf")
        for key, path in self.master_files.items():
            m = re.match(r"^\s*([\d.]+)s\b", str(key))
            if not m:
                continue
            try:
                dark_exp = float(m.group(1))
            except Exception:
                continue
            size = self.master_sizes.get(path)
            if size is None:
                try:
                    with fits.open(path) as hdul:
                        size = f"{hdul[0].data.shape[1]}x{hdul[0].data.shape[0]}"
                    self.master_sizes[path] = size
                except Exception:
                    continue
            if size == image_size:
                diff = abs(dark_exp - exposure_time)
                if diff < best_diff:
                    best_diff, best_path = diff, path
        return best_path

    def _auto_pick_master_flat(self, filter_name: str, image_size: str, session_name: str):
        # Prefer session-specific, then session-agnostic
        key_pref = f"{filter_name} ({image_size}) [{session_name}]"
        if key_pref in self.master_files:
            return self.master_files[key_pref]
        fallback_key = f"{filter_name} ({image_size})"
        return self.master_files.get(fallback_key)

    def _lookup_flat_override(self, filter_name: str, exposure_text: str) -> str | None:
        """Prefer 'Filter - exposure' override, else bare exposure."""
        if not hasattr(self, "manual_flat_overrides"):
            self.manual_flat_overrides = {}
        key_full  = f"{filter_name} - {exposure_text}"
        key_short = exposure_text
        return (self.manual_flat_overrides.get(key_full)
                or self.manual_flat_overrides.get(key_short))

    def _lookup_dark_override(self, filter_name: str, exposure_text: str) -> str | None:
        if not hasattr(self, "manual_dark_overrides"):
            self.manual_dark_overrides = {}
        key_full  = f"{filter_name} - {exposure_text}"
        key_short = exposure_text
        return (self.manual_dark_overrides.get(key_full)
                or self.manual_dark_overrides.get(key_short))

    # --- key helpers (scoped to filter+exposure) ---------------------------------
    def _light_key(self, filter_name: str, exposure_text: str) -> str:
        # Keep this EXACT format if other code relies on it
        return f"{filter_name} - {exposure_text}"

    def _item_scope(self, item):
        """
        Returns (filter_name, exposure_item) for any selected tree item:
        - leaf  -> (filter_name, its exposure parent)
        - exposure row -> (filter_name, exposure row)
        - filter row   -> (filter_name, None)    # caller can iterate children
        """
        if item.parent() and item.parent().parent():      # leaf
            exp_item = item.parent()
            return exp_item.parent().text(0), exp_item
        elif item.parent() and item.childCount() > 0:     # exposure row
            return item.parent().text(0), item
        elif item.parent() is None and item.childCount()>0:  # filter row
            return item.text(0), None
        return None, None

    def _iter_selected_light_leaves(self):
        """
        Yield only selected leaf rows (filename rows) from the Light tree.
        Leaf = has a parent (exposure) and grandparent (filter) and childCount()==0.
        """
        for it in self.light_tree.selectedItems():
            if it and it.childCount() == 0 and it.parent() and it.parent().parent():
                yield it


    def override_selected_master_flat(self):
        """
        Override Master Flat for ONLY the selected leaf (file) rows.
        Does not touch siblings, exposure groups, or filters unless those leaves are selected.
        """
        leaves = list(self._iter_selected_light_leaves())
        if not leaves:
            print("⚠️ Select individual Light files (leaf rows) to override their flat.")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Master Flat", "", "FITS Files (*.fits *.fit)"
        )
        if not file_path:
            return

        base = os.path.basename(file_path)
        for leaf in leaves:
            # Column 3 = "Master Flat" per your header
            leaf.setText(3, base)
            # stash the full path so calibration can use it directly
            leaf.setData(3, Qt.ItemDataRole.UserRole, file_path)

        print(f"✅ Flat override applied to {len(leaves)} selected file(s).")




    def toggle_group_correction(self, group_item, which):
        """
        group_item: a top-level item in the light_tree
        which: either "cosmetic" or "pedestal"
        """
        old_text = group_item.text(4)  # e.g. "Cosmetic: True, Pedestal: False"
        # If there's nothing, default them to False
        if not old_text:
            old_text = "Cosmetic: False, Pedestal: False"

        # Parse
        # old_text might be "Cosmetic: True, Pedestal: False"
        # split by comma
        # part[0] => "Cosmetic: True"
        # part[1] => " Pedestal: False"
        parts = old_text.split(",")
        cosmetic_str = "False"
        pedestal_str = "False"
        if len(parts) == 2:
            # parse cosmetic
            cos_part = parts[0].split(":")[-1].strip()  # "True" or "False"
            cosmetic_str = cos_part
            # parse pedestal
            ped_part = parts[1].split(":")[-1].strip()
            pedestal_str = ped_part

        # Convert to bool
        cosmetic_bool = (cosmetic_str.lower() == "true")
        pedestal_bool = (pedestal_str.lower() == "true")

        # Toggle whichever was requested
        if which == "cosmetic":
            cosmetic_bool = not cosmetic_bool
        elif which == "pedestal":
            pedestal_bool = not pedestal_bool

        # Rebuild the new text
        new_text = f"Cosmetic: {str(cosmetic_bool)}, Pedestal: {str(pedestal_bool)}"
        group_item.setText(4, new_text)

    def _resolve_corrections_for_exposure(self, exposure_item):
        """
        Decide whether to apply cosmetic correction & pedestal for a given exposure row.
        Priority:
        1) Live UI checkboxes (if present)
        2) Corrections column text on the row
        3) QSettings defaults
        """
        # 1) Live UI
        try:
            cosmetic_ui  = bool(self.cosmetic_checkbox.isChecked())
        except Exception:
            cosmetic_ui = None
        try:
            pedestal_ui  = bool(self.pedestal_checkbox.isChecked())
        except Exception:
            pedestal_ui = None

        # 2) Row text (Corrections column)
        apply_cosmetic_col = apply_pedestal_col = None
        try:
            correction_text = exposure_item.text(4) or ""
            if correction_text:
                parts = [p.strip().lower() for p in correction_text.split(",")]
                # Expect "Cosmetic: True, Pedestal: False"
                for p in parts:
                    if p.startswith("cosmetic:"):
                        apply_cosmetic_col = (p.split(":")[-1].strip() == "true")
                    elif p.startswith("pedestal:"):
                        apply_pedestal_col = (p.split(":")[-1].strip() == "true")
        except Exception:
            pass

        # 3) Settings default
        cosmetic_cfg = self.settings.value("stacking/cosmetic_enabled", True, type=bool)
        pedestal_cfg = self.settings.value("stacking/pedestal_enabled", False, type=bool)

        apply_cosmetic = (
            cosmetic_ui if cosmetic_ui is not None
            else (apply_cosmetic_col if apply_cosmetic_col is not None else cosmetic_cfg)
        )
        apply_pedestal = (
            pedestal_ui if pedestal_ui is not None
            else (apply_pedestal_col if apply_pedestal_col is not None else pedestal_cfg)
        )
        return bool(apply_cosmetic), bool(apply_pedestal)

    def _leaf_assigned_dark_path(self, leaf):
        # Column 2 = Master Dark (full path stashed in UserRole when overridden)
        try:
            p = leaf.data(2, Qt.ItemDataRole.UserRole)
            return str(p) if p else None
        except Exception:
            return None

    def _leaf_assigned_flat_path(self, leaf):
        # Column 3 = Master Flat (full path stashed in UserRole when overridden)
        try:
            p = leaf.data(3, Qt.ItemDataRole.UserRole)
            return str(p) if p else None
        except Exception:
            return None

    def _collect_leaf_paths_from_tree(self):
        """Return the exact set of light file paths represented by the current tree."""
        paths = []
        for i in range(self.light_tree.topLevelItemCount()):
            filter_item = self.light_tree.topLevelItem(i)
            filter_name = filter_item.text(0)

            for j in range(filter_item.childCount()):
                exposure_item = filter_item.child(j)
                exposure_text = exposure_item.text(0)

                for k in range(exposure_item.childCount()):
                    leaf = exposure_item.child(k)
                    filename = leaf.text(0)
                    meta = leaf.text(1) or ""

                    # Session from metadata (matches how you look it up later)
                    m = re.search(r"Session: ([^|]+)", meta)
                    session_name = m.group(1).strip() if m else "Default"

                    composite_key = (f"{filter_name} - {exposure_text}", session_name)
                    file_list = self.light_files.get(composite_key, [])
                    light_path = next((p for p in file_list if os.path.basename(p) == filename), None)
                    if light_path:
                        paths.append(light_path)
        # unique, preserves order
        return list(dict.fromkeys(paths))


    def select_reference_frame_robust(self, frame_weights, sigma_threshold=1.0):
        """
        Instead of sigma filtering, pick the frame at the 75th percentile of frame weights.
        This assumes that higher weights are better and that the 75th percentile represents
        a good-quality frame.
        
        Parameters
        ----------
        frame_weights : dict
            Mapping { file_path: weight_value } for each frame.
        
        Returns
        -------
        best_frame : str or None
            The file path of the chosen reference frame, or None if no frames are available.
        """
        items = list(frame_weights.items())  # List of (file_path, weight) pairs
        if not items:
            return None

        # Sort frames by weight in ascending order.
        items.sort(key=lambda x: x[1])
        n = len(items)
        # Get the index corresponding to the 75th percentile.
        index = int(0.75 * (n - 1))
        best_frame = items[index][0]
        return best_frame

    def extract_light_files_from_tree(self, *, debug: bool = False):
        """
        Rebuild self.light_files from what's *currently shown* in reg_tree.
        - Only uses leaf items (childCount()==0)
        - Repairs missing leaf UserRole by matching basename against parent's cached list
        - Filters non-existent paths
        """
        light_files: dict[str, list[str]] = {}
        total_leafs = 0
        total_paths = 0

        for i in range(self.reg_tree.topLevelItemCount()):
            top = self.reg_tree.topLevelItem(i)
            group_key = top.text(0)
            repaired_from_parent = 0

            # Parent's cached list (may be stale but useful for repairing)
            parent_cached = top.data(0, Qt.ItemDataRole.UserRole) or []

            paths: list[str] = []
            for j in range(top.childCount()):
                leaf = top.child(j)
                # Only accept real leaf rows (no grandchildren expected in this tree)
                if leaf.childCount() != 0:
                    continue

                total_leafs += 1

                fp = leaf.data(0, Qt.ItemDataRole.UserRole)
                if not fp:
                    # Try to repair by basename match against parent's cached list
                    name = leaf.text(0).lstrip("⚠️ ").strip()
                    match = next((p for p in parent_cached if os.path.basename(p) == name), None)
                    if match:
                        leaf.setData(0, Qt.ItemDataRole.UserRole, match)
                        fp = match
                        repaired_from_parent += 1

                if fp and isinstance(fp, str) and os.path.exists(fp):
                    paths.append(fp)

            if paths:
                light_files[group_key] = paths
                # keep the parent cache in sync for future repairs
                top.setData(0, Qt.ItemDataRole.UserRole, paths)
                total_paths += len(paths)

            if debug:
                self.update_status(
                    f"⤴ {group_key}: {len(paths)} files"
                    + (f" (repaired {repaired_from_parent})" if repaired_from_parent else "")
                )

        self.light_files = light_files
        if debug:
            self.update_status(f"🧭 Tree snapshot → groups: {len(light_files)}, leaves seen: {total_leafs}, paths kept: {total_paths}")
        return light_files

    def _norm_filter_key(self, s: str) -> str:
        s = (s or "").lower()
        # map greek letters to ascii
        s = s.replace("α", "a").replace("β", "b")
        return re.sub(r"[^a-z0-9]+", "", s)

    def _classify_filter(self, filt_str: str) -> str:
        """
        Return one of:
        'DUAL_HA_OIII', 'DUAL_SII_OIII', 'DUAL_SII_HB',
        'MONO_HA', 'MONO_SII', 'MONO_OIII', 'MONO_HB',
        'UNKNOWN'
        """
        k = self._norm_filter_key(filt_str)
        comps = set()

        # explicit component tokens
        if "ha"    in k or "halpha" in k: comps.add("ha")
        if "sii"   in k or "s2"     in k: comps.add("sii")
        if "oiii"  in k or "o3"     in k: comps.add("oiii")
        if "hb"    in k or "hbeta"  in k: comps.add("hb")

        # common vendor aliases → Ha/OIII
        vendor_aliases = (
            "lextreme", "lenhance", "lultimate",
            "nbz", "nbzu", "alpt", "alp",
            "duo-band", "duoband", "dual band", "dual-band", "dualband"
        )
        if any(alias in k for alias in vendor_aliases):
            comps.update({"ha", "oiii"})

        # generic dual/duo/bicolor markers → assume Ha/OIII (most OSC duals)
        dual_markers = (
            "dual", "duo", "2band", "2-band", "two band",
            "bicolor", "bi-color", "bicolour", "bi-colour",
            "dualnb", "dual-nb", "duo-nb", "duonb",
            "duo narrow", "dual narrow"
        )
        if any(m in k for m in dual_markers):
            comps.update({"ha", "oiii"})

        # decide
        if {"ha","oiii"}.issubset(comps):  return "DUAL_HA_OIII"
        if {"sii","oiii"}.issubset(comps): return "DUAL_SII_OIII"
        if {"sii","hb"}.issubset(comps):   return "DUAL_SII_HB"

        if comps == {"ha"}:   return "MONO_HA"
        if comps == {"sii"}:  return "MONO_SII"
        if comps == {"oiii"}: return "MONO_OIII"
        if comps == {"hb"}:   return "MONO_HB"

        # NEW: if user explicitly asked to split dual-band, default to Ha/OIII
        try:
            if hasattr(self, "split_dualband_cb") and self.split_dualband_cb.isChecked():
                return "DUAL_HA_OIII"
        except Exception:
            pass

        return "UNKNOWN"

    def _get_filter_name(self, path: str) -> str:
        # Prefer FITS header 'FILTER'; fall back to filename tokens
        try:
            hdr = fits.getheader(path, ext=0)
            for key in ("FILTER", "FILTER1", "HIERARCH INDI FILTER", "HIERARCH ESO INS FILT1 NAME"):
                if key in hdr and str(hdr[key]).strip():
                    return str(hdr[key]).strip()
        except Exception:
            pass
        return os.path.basename(path)

    def _current_global_drizzle(self):
        # read from the “global” controls (used as a template)
        return {
            "enabled": self.drizzle_checkbox.isChecked(),
            "scale": float(self.drizzle_scale_combo.currentText().replace("x","", 1)),
            "drop": float(self.drizzle_drop_shrink_spin.value())
        }

    def _split_dual_band_osc(self, selected_groups=None):
        """
        Create mono Ha/SII/OIII frames from dual-band OSC files and
        update self.light_files so integration sees separate channels.
        """
        selected_groups = selected_groups or set()
        out_dir = os.path.join(self.stacking_directory, "DualBand_Split")
        os.makedirs(out_dir, exist_ok=True)

        ha_files, sii_files, oiii_files, hb_files = [], [], [], []
        inherit_map = {}                      # gk -> set(parent_group names)   # <<< NEW
        parent_of = {}                        # path -> parent_group            # <<< NEW

        # Walk all groups/files you already collected
        old_groups = list(self.light_files.items())
        old_drizzle = dict(self.per_group_drizzle)
        for group, files in old_groups:
            for fp in files:
                try:
                    img, hdr, _, _ = load_image(fp)
                    if img is None:
                        self.update_status(f"⚠️ Cannot load {fp}; skipping.")
                        continue

                    if hdr and hdr.get("BAYERPAT"):
                        img = self.conversion_tab.debayer_image(img, fp, hdr)

                    # 3-channel split; otherwise treat mono via classifier
                    if img.ndim != 3 or img.shape[-1] < 2:
                        filt = self._get_filter_name(fp)
                        cls  = self._classify_filter(filt)
                        if cls == "MONO_HA":
                            ha_files.append(fp);   parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_SII":
                            sii_files.append(fp);  parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_OIII":
                            oiii_files.append(fp); parent_of[fp] = group        # <<< NEW
                        elif cls == "MONO_HB":   hb_files.append(fp);  parent_of[fp] = group        # <<< NEW
                        # else: leave in original groups
                        continue

                    filt = self._get_filter_name(fp)
                    cls  = self._classify_filter(filt)

                    R = img[..., 0]; G = img[..., 1]
                    base = os.path.splitext(os.path.basename(fp))[0]

                    if cls == "DUAL_HA_OIII":
                        ha_path   = os.path.join(out_dir, f"{base}_Ha.fit")
                        oiii_path = os.path.join(out_dir, f"{base}_OIII.fit")
                        self._write_band_fit(ha_path,  R, hdr, "Ha",  src_filter=filt)
                        self._write_band_fit(oiii_path, G, hdr, "OIII", src_filter=filt)
                        ha_files.append(ha_path);     parent_of[ha_path]   = group   # <<< NEW
                        oiii_files.append(oiii_path); parent_of[oiii_path] = group   # <<< NEW

                    elif cls == "DUAL_SII_OIII":
                        sii_path  = os.path.join(out_dir, f"{base}_SII.fit")
                        oiii_path = os.path.join(out_dir, f"{base}_OIII.fit")
                        self._write_band_fit(sii_path, R, hdr, "SII",  src_filter=filt)
                        self._write_band_fit(oiii_path, G, hdr, "OIII", src_filter=filt)
                        sii_files.append(sii_path);    parent_of[sii_path]  = group  # <<< NEW
                        oiii_files.append(oiii_path);  parent_of[oiii_path] = group  # <<< NEW

                    elif cls == "DUAL_SII_HB":  # NEW → R=SII, G=Hb  (G works well; we can add G+B later if you want)
                        sii_path = os.path.join(out_dir, f"{base}_SII.fit")
                        hb_path  = os.path.join(out_dir, f"{base}_Hb.fit")
                        self._write_band_fit(sii_path, R, hdr, "SII", src_filter=filt)
                        self._write_band_fit(hb_path,  G, hdr, "Hb",  src_filter=filt)
                        sii_files.append(sii_path); parent_of[sii_path] = group
                        hb_files.append(hb_path);   parent_of[hb_path]  = group

                    else:
                        pass

                except Exception as e:
                    self.update_status(f"⚠️ Split error on {os.path.basename(fp)}: {e}")

        # Group the new files
        def _group_key(band: str, path: str) -> str:
            try:
                h = fits.getheader(path, ext=0)
                exp = h.get("EXPTIME") or h.get("EXPOSURE") or ""
                w   = h.get("NAXIS1","?"); hgt = h.get("NAXIS2","?")
                exp_str = f"{float(exp):.1f}s" if isinstance(exp, (int,float)) else str(exp)
                return f"{band} - {exp_str} - {w}x{hgt}"
            except Exception:
                return f"{band} - ? - ?x?"

        new_groups = {}
        for band, flist in (("Ha", ha_files), ("SII", sii_files), ("OIII", oiii_files), ("Hb", hb_files)):  # NEW Hb
            for p in flist:
                gk = _group_key(band, p)
                new_groups.setdefault(gk, []).append(p)
                parent = parent_of.get(p)
                if parent:
                    inherit_map.setdefault(gk, set()).add(parent)

        if new_groups:
            self.light_files = new_groups

            # Seed drizzle for the new groups based on parents
            seeded = 0
            global_template = self._current_global_drizzle()   # make sure this helper exists
            self.per_group_drizzle = {}  # rebuild for the new groups

            for gk, parents in inherit_map.items():
                parent_cfgs = [old_drizzle.get(pg) for pg in parents if old_drizzle.get(pg)]
                chosen = None
                for cfg in parent_cfgs:
                    if cfg.get("enabled"):
                        chosen = cfg
                        break
                if not chosen and parent_cfgs:
                    chosen = parent_cfgs[0]

                if not chosen and (parents & selected_groups) and global_template.get("enabled"):
                    chosen = global_template

                if chosen:
                    self.per_group_drizzle[gk] = dict(chosen)
                    seeded += 1


            self.update_status(
                f"✅ Dual-band split complete: Ha={len(ha_files)}, SII={len(sii_files)}, "
                f"OIII={len(oiii_files)}, Hb={len(hb_files)} (drizzle seeded on {seeded} new group(s))"
            )
        else:
            self.update_status("ℹ️ No dual-band frames detected or split.")

    def _write_band_fit(self, out_path: str, data: np.ndarray, src_header: Optional[fits.Header],
                        band: str, src_filter: str):

        arr = np.ascontiguousarray(data.astype(np.float32))

        hdr = (src_header.copy() if isinstance(src_header, fits.Header) else fits.Header())

        # --- strip CFA/Bayer-related cards so we never try to debayer these ---
        cfa_like = (
            "BAYERPAT", "BAYER_PATTERN", "DEBAYER", "DEBAYERING", "DEMAT", "DEMOSAIC",
            "XBAYROFF", "YBAYROFF", "COLORTYP", "COLORSPACE", "HIERARCH CFA", "HIERARCH OSC",
            "HIERARCH ASI BAYERPATTERN", "HIERARCH DNG CFA", "HIERARCH ZWO CFA"
        )
        for k in list(hdr.keys()):
            kk = str(k).upper()
            if any(token in kk for token in ("BAYER", "CFA", "DEMOSA")) or kk in cfa_like:
                try:
                    del hdr[k]
                except Exception:
                    pass

        # Mark these as mono split files & set the band as the filter
        hdr["FILTER"] = (band, "Channel from dual-band split")
        hdr["SPLITDB"] = (True, "This frame was generated by dual-band splitting")
        hdr.add_history(f"Dual-band split: {band} from {src_filter}")

        fits.PrimaryHDU(data=arr, header=hdr).writeto(out_path, overwrite=True)

    def _drizzle_text_for_group(self, group_key: str) -> str:
        d = self.per_group_drizzle.get(group_key)
        if not d:
            return ""
        return f"Drizzle: {d.get('enabled', False)}, Scale: {d.get('scale','1x')}, Drop:{d.get('drop',0.65)}"

    def _refresh_reg_tree_from_light_files(self):
        self.reg_tree.clear()
        for group, files in self.light_files.items():
            top = QTreeWidgetItem([group, f"{len(files)} file(s)", self._drizzle_text_for_group(group)])
            self.reg_tree.addTopLevelItem(top)
            for fp in files:
                # Optional: show some header metadata
                meta = ""
                try:
                    hdr = fits.getheader(fp, ext=0)
                    filt = hdr.get("FILTER", "")
                    exp  = hdr.get("EXPTIME") or hdr.get("EXPOSURE") or ""
                    if isinstance(exp, (int, float)): exp = f"{exp:.1f}s"
                    meta = f"Filter={filt}  Exp={exp}"
                except Exception:
                    pass
                child = QTreeWidgetItem([os.path.basename(fp), meta, ""])
                top.addChild(child)
        self.reg_tree.expandAll()

    def _norm_ang(self, a):
        a = a % 360.0
        return a + 360.0 if a < 0 else a

    def _angdiff(self, a, b):
        # smallest absolute difference in degrees
        return abs((self._norm_ang(a) - self._norm_ang(b) + 180.0) % 360.0 - 180.0)

    def _extract_pa_deg(self, hdr):
        """
        Try common FITS/PI keys for camera/sky position angle (degrees).
        Works with both astropy FITS Header and XISF image-metadata dicts.

        Heuristics:
        1) Direct keywords (first match wins): POSANGLE, ANGLE, ROTANGLE, ROTSKYPA,
            ROTATOR, PA, ORIENTAT, CROTA2, CROTA1.
            • Values may be numbers or strings like '123.4 deg' — we parse the first float.
        2) WCS fallback:
            • CD matrix:  pa = atan2(-CD1_2, CD2_2)
            • PC matrix (+ optional CDELT scaling): pa = atan2(-PC1_2, PC2_2)
        Returns float in degrees, or None if unknown.
        """


        if hdr is None:
            return None

        # helper: unified header getter (FITS Header or XISF dict)
        def _get(k, default=None):
            try:
                return self._hdr_get(hdr, k, default)
            except Exception:
                # fall back to direct mapping-like access if needed
                try:
                    return hdr.get(k, default)
                except Exception:
                    return default

        # parse possibly-string value → float
        def _as_float_deg(v):
            if v is None:
                return None
            if isinstance(v, (int, float, np.integer, np.floating)):
                try:
                    return float(v)
                except Exception:
                    return None
            # strings like "123.4", "123.4 deg", "123,4", etc.
            try:
                s = str(v)
                m = re.search(r"[-+]?\d+(?:[.,]\d+)?", s)
                if not m:
                    return None
                val = float(m.group(0).replace(",", "."))
                return val
            except Exception:
                return None

        # 1) direct keyword attempts (first hit wins)
        direct_keys = (
            "POSANGLE", "ANGLE", "ROTANGLE", "ROTSKYPA", "ROTATOR",
            "PA", "ORIENTAT", "CROTA2", "CROTA1"
        )
        for k in direct_keys:
            v = _get(k, None)
            ang = _as_float_deg(v)
            if ang is not None:
                return float(ang)

        # 2) WCS fallback using CD or PC matrices
        # CD first
        cd11 = _get('CD1_1'); cd12 = _get('CD1_2'); cd22 = _get('CD2_2')
        if cd11 is not None and cd12 is not None and cd22 is not None:
            try:
                pa = np.degrees(np.arctan2(-float(cd12), float(cd22)))
                return float(pa)
            except Exception:
                pass

        # PC (optionally combined with CDELT)
        pc11 = _get('PC1_1'); pc12 = _get('PC1_2'); pc22 = _get('PC2_2')
        if pc11 is not None and pc12 is not None and pc22 is not None:
            try:
                # If CDELT present, rotation is still given by PC (scale cancels out for angle)
                pa = np.degrees(np.arctan2(-float(pc12), float(pc22)))
                return float(pa)
            except Exception:
                pass

        return None


    def _maybe_rot180(self, img, pa_cur, pa_ref, tol_deg):
        """
        If |(pa_cur - pa_ref)| ≈ 180° (within tol), rotate image 180°.
        Works for (H,W) or (H,W,3).
        Returns (img_out, rotated_bool).
        """
        
        if pa_cur is None or pa_ref is None:
            return img, False
        d = self._angdiff(pa_cur, pa_ref)
        if abs(d - 180.0) <= tol_deg:
            # 180° is just two 90° rotations; cheap & exact
            # np.rot90 returns a view, make contiguous for downstream processing
            self.update_status(f"Flipping Image")
            QApplication.processEvents()
            return np.ascontiguousarray(np.rot90(img, 2)), True
        return img, False

    def _ui_log(self, msg: str):
        self.update_status(msg)  # your existing status logger
        # let Qt process pending paint/input signals so the UI updates
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 25)

    def _maybe_warn_cfa_low_frames(self):
        if not (getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked()):
            self._cfa_for_this_run = None  # follow checkbox (OFF)
            return

        # Count frames per group (use the *current* reg tree groups)
        per_group_counts = {g: len(v) for g, v in (self.light_files or {}).items() if v}
        if not per_group_counts:
            self._cfa_for_this_run = None
            return

        worst = min(per_group_counts.values())

        # Scale-aware cutoff (you can expose this in QSettings if you like)
        try:
            scale_txt = self.drizzle_scale_combo.currentText()
            scale = float(scale_txt.replace("x", "").strip())
        except Exception:
            scale = 1.0
        cutoff = {1.0: 32, 2.0: 64, 3.0: 96}.get(scale, 64)

        if worst >= cutoff:
            self._cfa_for_this_run = True   # keep raw CFA mapping
            return

        # Ask the user
        msg = (f"CFA Drizzle is enabled but at least one group has only {worst} frames.\n\n"
            f"CFA Drizzle typically needs ≥{cutoff} frames (scale {scale:.0f}×) for good coverage.\n"
            "Switch to Edge-Aware Interpolation for this run?")
        ret = QMessageBox.question(
            self, "CFA Drizzle: Low Sample Count",
            msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if ret == QMessageBox.StandardButton.Yes:
            # Disable raw CFA just for this run
            self._cfa_for_this_run = False
            self.update_status("⚠️ CFA Drizzle: low-count fallback → using Edge-Aware Interpolation for this run.")
        else:
            self._cfa_for_this_run = True
            self.update_status("ℹ️ CFA Drizzle kept despite low frame count (you chose No).")

    def _ensure_comet_seed_now(self) -> bool:
        """If no comet seed exists, open the picker. Return True IFF we have a seed after."""
        if getattr(self, "_comet_seed", None):
            return True
        # Open the same picker you already have
        self._pick_comet_center()
        return bool(getattr(self, "_comet_seed", None))

    # small helper to toggle UI while registration is running
    def _set_registration_busy(self, busy: bool):
        self._registration_busy = bool(busy)
        self.register_images_btn.setEnabled(not busy)
        self.integrate_registered_btn.setEnabled(not busy)
        # optional visual hint
        if busy:
            self.register_images_btn.setText("⏳ Registering…")
            self.register_images_btn.setToolTip("Registration in progress…")
        else:
            self.register_images_btn.setText("🔥🚀Register and Integrate Images🔥🚀")
            self.register_images_btn.setToolTip("")

        # prevent accidental double-queue from keyboard/space
        self.register_images_btn.blockSignals(busy)

    def _cosmetic_enabled(self) -> bool:
        try:
            if hasattr(self, "cosmetic_checkbox") and self.cosmetic_checkbox is not None:
                return bool(self.cosmetic_checkbox.isChecked())
        except Exception:
            pass
        return bool(self.settings.value("stacking/cosmetic_enabled", True, type=bool))

    # ——— generic header access ———
    def _hdr_get(self, hdr, key, default=None):
        """Uniform header getter for FITS (astropy Header) or XISF (dict with FITSKeywords)."""
        if hdr is None:
            return default
        # FITS Header
        try:
            if hasattr(hdr, "get"):
                return hdr.get(key, default)
        except Exception:
            pass
        # XISF-style dict from XISF.get_images_metadata()[i]
        try:
            # FITSKeywords: { KEY: [ {value, comment}, ... ] }
            fkw = hdr.get("FITSKeywords", {})
            lst = fkw.get(key)
            if lst and isinstance(lst, list) and "value" in lst[0]:
                return lst[0]["value"]
        except Exception:
            pass
        # fall back to plain dict
        try:
            return hdr.get(key, default)
        except Exception:
            return default


    # ——— binning for either format ———
    def _bin_from_header_fast_any(self, fp: str) -> tuple[int,int]:
        ext = os.path.splitext(fp)[1].lower()
        # FITS quick path
        if ext in (".fits", ".fit", ".fz"):
            try:
                h = fits.getheader(fp, ext=0)
                # common variants
                xb = int(h.get("XBINNING", h.get("XBIN", 1)))
                yb = int(h.get("YBINNING", h.get("YBIN", 1)))
                return max(1, xb), max(1, yb)
            except Exception:
                return (1,1)
        # XISF path
        if ext == ".xisf":
            try:
                from legacy.xisf import XISF
                x = XISF(fp)
                ims = x.get_images_metadata()
                if not ims:
                    return (1,1)
                m0 = ims[0]
                fkw = m0.get("FITSKeywords", {})
                def _kw(name, default=None):
                    vals = fkw.get(name)
                    return vals[0]["value"] if vals and "value" in vals[0] else default
                xb = int(_kw("XBINNING", _kw("XBIN", 1)) or 1)
                yb = int(_kw("YBINNING", _kw("YBIN", 1)) or 1)
                return max(1, xb), max(1, yb)
            except Exception:
                return (1,1)
        return (1,1)


    # ——— fast preview for either format (2D float32 at target bin scale) ———
    def _quick_preview_any(self, fp: str, target_xbin: int, target_ybin: int) -> np.ndarray | None:
        ext = os.path.splitext(fp)[1].lower()
        # 1) Load a small-ish *image* block without full decode work
        img = None; hdr = None

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
                L = 0.2126*r + 0.7152*g + 0.0722*b
                return (L[0:h2:2, 0:w2:2] + L[0:h2:2, 1:w2:2] +
                        L[1:h2:2, 0:w2:2] + L[1:h2:2, 1:w2:2]) * 0.25

        try:
            if ext in (".fits", ".fit", ".fz"):
                # primary data only for speed
                data = fits.getdata(fp, ext=0)
                hdr  = fits.getheader(fp, ext=0)
                img = np.asanyarray(data)
            elif ext == ".xisf":
                from legacy.xisf import XISF
                x = XISF(fp)
                ims = x.get_images_metadata()
                if not ims:
                    return None
                hdr = ims[0]  # carry XISF image metadata dict as "header"
                img = x.read_image(0)  # channels-last
            else:
                # generic formats: use your existing thumb path
                w, h = self._get_image_size(fp)
                # cheap preview load (PIL) — grayscale
                from PIL import Image
                im = Image.open(fp).convert("L")
                img = np.asarray(im, dtype=np.float32) / 255.0
                hdr = {}

            a = np.asarray(img)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[...,0]
            # if it’s color, make a luma-like 2×2 superpixel preview; if mono/CFA, same superpixel trick
            if a.ndim == 3 and a.shape[-1] == 3:
                prev2d = _superpixel2x2(a)
            else:
                prev2d = _superpixel2x2(a)

            # resample preview to the target bin scale
            xb, yb = self._bin_from_header_fast_any(fp)
            sx = float(xb) / float(target_xbin)
            sy = float(yb) / float(target_ybin)
            if (abs(sx - 1.0) > 1e-6) or (abs(sy - 1.0) > 1e-6):
                prev2d = _resize_to_scale(prev2d, sx, sy)

            return np.ascontiguousarray(prev2d.astype(np.float32, copy=False))
        except Exception:
            return None


    # ——— on-demand full load (float32, header-like), for normalization stage ———
    def _load_image_any(self, fp: str):
        """
        Return (img, hdr_like) for FITS/XISF/other.
        img is float32, channels-last if color, shape (H,W) if mono.
        hdr_like: FITS Header for FITS; XISF image metadata dict for XISF; {} otherwise.
        """
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext in (".fits", ".fit", ".fz"):
                from legacy.image_manager import load_image as legacy_load_image
                img, hdr, _, _ = legacy_load_image(fp)
                return img, (hdr or fits.Header())
            if ext == ".xisf":
                from legacy.xisf import XISF
                x = XISF(fp)
                ims = x.get_images_metadata()
                if not ims:
                    return None, {}
                img = x.read_image(0)  # channels-last
                # normalize integer types to [0..1] just like FITS path
                a = np.asarray(img)
                if a.dtype.kind in "ui":
                    a = a.astype(np.float32) / np.float32(np.iinfo(a.dtype).max)
                elif a.dtype.kind == "f":
                    a = a.astype(np.float32, copy=False)
                else:
                    a = a.astype(np.float32, copy=False)
                hdr_like = ims[0]  # carry XISF image metadata dict
                return a, hdr_like
            # generic images (TIFF/PNG/JPEG)
            from legacy.image_manager import load_image as legacy_load_image
            img, hdr, _, _ = legacy_load_image(fp)
            return img, (hdr or {})
        except Exception:
            return None, {}

    def _zap_reg_caches(self):
        """Clear all registration-related caches so a run always starts fresh."""
        # Dict/array caches on self (clear if present)
        for name in (
            "_preview_cache", "_quick_preview_cache", "_load_cache",
            "_debayer_cache", "_abe_cache", "_poly_fit_cache",
            "_affine_cache", "_star_features_cache", "_orig2norm",
            "frame_weights", "arcsec_per_px", "_reg_debug",
        ):
            obj = getattr(self, name, None)
            try:
                if isinstance(obj, dict): obj.clear()
                elif isinstance(obj, (list, set)): obj.clear()
            except Exception:
                pass

        # Reset shapes/targets computed in previous runs
        for name in ("_norm_target_hw", "reference_frame", "_comet_ref_xy"):
            if hasattr(self, name):
                setattr(self, name, None)

        # Proactively clear any lru_cache’d helpers if available
        def _cc(f):
            try:
                if hasattr(f, "cache_clear"): f.cache_clear()
            except Exception:
                pass

        for f in (
            getattr(self, "_bin_from_header_fast_any", None),
            getattr(self, "_quick_preview_any", None),
            getattr(self, "debayer_image", None),
            getattr(self, "_extract_pa_deg", None),
            getattr(self, "_hdr_get", None),
        ):
            if callable(f): _cc(f)

    def _set_user_reference(self, path: str):
        import os
        norm = os.path.normpath(path)            # ← unify separators like the auto flow
        self.reference_frame = norm
        self._user_ref_locked = True
        try:
            self.settings.setValue("stacking/user_reference_frame", norm)  # store same format
        except Exception:
            pass
        try:
            if hasattr(self, "ref_frame_path") and self.ref_frame_path:
                self.ref_frame_path.setText(os.path.basename(norm))        # UI shows basename (like auto)
        except Exception:
            pass

    def reset_reference_to_auto(self):
        self._user_ref_locked = False
        # Keep self.reference_frame as-is; it will be ignored next run if not locked
        try:
            self.settings.remove("stacking/user_reference_frame")
        except Exception:
            pass
        try:
            if hasattr(self, "ref_frame_path") and self.ref_frame_path:
                self.ref_frame_path.setText("Auto (not set)")
        except Exception:
            pass
        self.update_status("🔓 Reference unlocked. Next run will auto-select the best reference.")

    def _on_auto_accept_toggled(self, v: bool):
        # Persist the checkbox state
        try:
            self.settings.setValue("stacking/auto_accept_ref", bool(v))
        except Exception:
            pass

        # When turning ON auto-accept, immediately revert to auto reference selection
        if v and getattr(self, "_user_ref_locked", False):
            self.reset_reference_to_auto()
        # When turning OFF, do nothing special—user can pick a ref again if they want.

    def _on_align_done(self, success: bool, message: str):
        # Stop any coalesced progress updates
        try:
            if hasattr(self, "_align_prog_timer") and self._align_prog_timer.isActive():
                self._align_prog_timer.stop()
            # Clear pending state (if you added the debounce members)
            if hasattr(self, "_align_prog_in_slot"):
                self._align_prog_in_slot = False
            if hasattr(self, "_align_prog_pending"):
                self._align_prog_pending = None
        except Exception:
            pass

        # Close/reset the progress dialog without triggering more signals
        dlg = getattr(self, "align_progress", None)
        if dlg is not None:
            try:
                # If total is known this forces the bar to 100% (purely cosmetic)
                if dlg.maximum() > 0:
                    dlg.setValue(dlg.maximum())
            except Exception:
                pass
            try:
                # If AutoClose/AutoReset is set, this both resets and closes
                dlg.reset()
            except Exception:
                try:
                    dlg.close()
                except Exception:
                    pass
            # Remove the attribute to avoid stale references
            try:
                del self.align_progress
            except Exception:
                try:
                    self.align_progress = None
                except Exception:
                    pass

        # Re-enable UI / busy state
        try:
            self._set_registration_busy(False)
        except Exception:
            pass

        # Update any local label directly (avoid update_status to prevent feedback)
        try:
            if hasattr(self, "progress_label"):
                color = "green" if success else "red"
                self.progress_label.setText(f"Status: {message}")
                self.progress_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception:
            pass

    def _make_star_trail(self):
        # 1) collect all your calibrated light frames
        all_files = [f for flist in self.light_files.values() for f in flist]
        n_frames = len(all_files)
        if not all_files:
            self.update_status("⚠️ No calibrated lights available for star trails.")
            return

        # 2) load every frame (once), compute its median, and remember its header
        frames: list[tuple[np.ndarray, fits.Header]] = []
        medians: list[float] = []

        for fn in all_files:
            img, hdr, _, _ = load_image(fn)
            if img is None:
                self.update_status(f"⚠️ Failed to load {os.path.basename(fn)}; skipping")
                QApplication.processEvents()
                continue

            arr = img.astype(np.float32)
            medians.append(float(np.median(arr)))
            frames.append((arr, hdr))

        if not frames:
            self.update_status("⚠️ No valid frames to compute reference median; aborting star-trail.")
            return

        # reference median is the median of per-frame medians
        ref_median = float(np.median(medians))

        # grab the header from the first valid frame, strip out extra NAXIS keywords
        first_hdr = frames[0][1]
        if first_hdr is not None:
            hdr_to_use = first_hdr.copy()
            for key in list(hdr_to_use):
                if key.startswith("NAXIS") and key not in ("NAXIS", "NAXIS1", "NAXIS2"):
                    hdr_to_use.pop(key, None)
        else:
            hdr_to_use = None

        # 3) normalize each frame and write to a temp dir
        with tempfile.TemporaryDirectory(prefix="startrail_norm_") as norm_dir:
            normalized_paths = []
            for idx, (arr, hdr) in enumerate(frames, start=1):
                self.update_status(f"🔄 Normalizing frame {idx}/{len(frames)}")
                QApplication.processEvents()

                # guard against divide-by-zero
                m = float(np.median(arr))
                scale = ref_median / (m + 1e-12)
                img_norm = arr * scale

                stem = Path(all_files[idx-1]).stem
                out_path = os.path.join(norm_dir, f"{stem}_st.fit")
                fits.PrimaryHDU(data=img_norm, header=hdr).writeto(out_path, overwrite=True)
                normalized_paths.append(out_path)

            # 4) stack and do max-value projection
            self.update_status(f"📊 Stacking {len(normalized_paths)} frames")
            QApplication.processEvents()
            stack = np.stack([fits.getdata(p).astype(np.float32) for p in normalized_paths], axis=0)
            trail_img, _ = max_value_stack(stack)

            # 5) stretch final image and prompt user for save location & format
            trail_img = trail_img.astype(np.float32)
            # normalize to [0–1] for our save helper
            trail_norm = trail_img / (trail_img.max() + 1e-12)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = self._safe_component(f"StarTrail_{n_frames:03d}frames_{ts}")
            filters = "TIFF (*.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;FITS (*.fits);;XISF (*.xisf)"
            path, chosen_filter = QFileDialog.getSaveFileName(
                self, "Save Star-Trail Image",
                os.path.join(self.stacking_directory, default_name),
                "TIFF (*.tif);;PNG (*.png);;JPEG (*.jpg *.jpeg);;FITS (*.fits);;XISF (*.xisf)"
            )
            if not path:
                self.update_status("✖ Star-trail save cancelled.")
                return

            # figure out extension
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            if not ext:
                ext = chosen_filter.split('(')[1].split(')')[0].lstrip('*.').lower()
                path += f".{ext}"

            # if user picked FITS, supply the first frame’s header; else None
            use_hdr = hdr_to_use if ext in ('fits', 'fit') else None

            # 16-bit everywhere
            save_image(
                img_array=trail_norm,
                filename=path,
                original_format=ext,
                bit_depth="16-bit",
                original_header=use_hdr,
                is_mono=False
            )

        # once we exit the with-block, all the _st.fit files are deleted
        self.update_status(f"✅ Star‐Trail image written to {path}")
        return


    def _apply_autocrop(self, arr, file_list, header, scale=1.0, rect_override=None):
        """
        If rect_override is provided, use it; else compute per-file_list.
        """
        try:
            enabled = self.autocrop_cb.isChecked()
            pct = float(self.autocrop_pct.value())
        except Exception:
            enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        if not enabled or not file_list:
            return arr, header

        rect = rect_override
        if rect is None:
            transforms_path = os.path.join(self.stacking_directory, "alignment_transforms.sasd")
            rect = self._compute_autocrop_rect(file_list, transforms_path, pct)

        if not rect:
            self.update_status("✂️ Auto-crop: no common area found; skipping.")
            return arr, header

        x0, y0, x1, y1 = rect
        if scale != 1.0:
            # scale rect to drizzle resolution
            x0 = int(math.floor(x0 * scale))
            y0 = int(math.floor(y0 * scale))
            x1 = int(math.ceil (x1 * scale))
            y1 = int(math.ceil (y1 * scale))

        # Clamp to image bounds
        H, W = arr.shape[:2]
        x0 = max(0, min(W, x0)); x1 = max(x0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(y0, min(H, y1))

        # --- Crop while preserving channels ---
        if arr.ndim == 2:
            arr = arr[y0:y1, x0:x1]
        else:
            arr = arr[y0:y1, x0:x1, :]
            # If this is actually mono stored as (H,W,1), squeeze back to (H,W)
            if arr.shape[-1] == 1:
                arr = arr[..., 0]

        # Update header dims (+ shift CRPIX if present)
        if header is None:
            header = fits.Header()

        # NAXIS / sizes consistent with the new array
        if arr.ndim == 2:
            header["NAXIS"]  = 2
            header["NAXIS1"] = arr.shape[1]
            header["NAXIS2"] = arr.shape[0]
            # Remove any stale NAXIS3
            if "NAXIS3" in header:
                del header["NAXIS3"]
        else:
            header["NAXIS"]  = 3
            header["NAXIS1"] = arr.shape[1]
            header["NAXIS2"] = arr.shape[0]
            header["NAXIS3"] = arr.shape[2]

        if "CRPIX1" in header:
            header["CRPIX1"] = float(header["CRPIX1"]) - x0
        if "CRPIX2" in header:
            header["CRPIX2"] = float(header["CRPIX2"]) - y0

        self.update_status(f"✂️ Auto-cropped to [{x0}:{x1}]×[{y0}:{y1}] (scale {scale}×)")
        return arr, header

    def _dither_phase_fill(self, matrices: dict[str, np.ndarray], bins=8) -> float:
        hist = np.zeros((bins, bins), dtype=np.int32)
        for M in matrices.values():
            M = np.asarray(M)
            if M.shape == (2,3):
                tx, ty = float(M[0,2]), float(M[1,2])
            elif M.shape == (3,3):
                # translation at origin for homography
                tx, ty = float(M[0,2]), float(M[1,2])
            else:
                continue
            fx = (tx - math.floor(tx)) % 1.0
            fy = (ty - math.floor(ty)) % 1.0
            ix = min(int(fx * bins), bins - 1)
            iy = min(int(fy * bins), bins - 1)
            hist[iy, ix] += 1
        return float(np.count_nonzero(hist)) / float(hist.size)


    def _on_mf_progress(self, s: str):
        # Mirror non-token messages
        if not s.startswith("__PROGRESS__"):
            self._on_post_status(s)
            if getattr(self, "_mf_pd", None):
                self._mf_pd.setLabelText(s)
            return

        # "__PROGRESS__ <float> [message]"
        parts = s.split(maxsplit=2)
        try:
            pct = float(parts[1])
        except Exception:
            return

        if len(parts) >= 3 and getattr(self, "_mf_pd", None):
            self._mf_pd.setLabelText(parts[2])

        if getattr(self, "_mf_pd", None):
            groups_done = getattr(self, "_mf_groups_done", 0)
            total_groups = max(1, getattr(self, "_mf_total_groups", 1))
            base = groups_done * 1000
            val = base + int(round(max(0.0, min(1.0, pct)) * 1000))
            self._mf_pd.setRange(0, total_groups * 1000)
            self._mf_pd.setValue(min(val, total_groups * 1000))

    @pyqtSlot(bool, str)
    def _on_post_pipeline_finished(self, ok: bool, message: str):
        try:
            if getattr(self, "post_progress", None):
                self.post_progress.close()
                self.post_progress = None
        except Exception:
            pass

        try:
            self.post_thread.quit()
            self.post_thread.wait()
        except Exception:
            pass
        try:
            self.post_worker.deleteLater()
            self.post_thread.deleteLater()
        except Exception:
            pass

        self.update_status(message)
        self._cfa_for_this_run = None
        QApplication.processEvents()


    def save_rejection_map_sasr(self, rejection_map, out_file):
        """
        Writes the per-file rejection map to a custom text file.
        Format:
            FILE: path/to/file1
            x1, y1
            x2, y2

            FILE: path/to/file2
            ...
        """
        with open(out_file, "w") as f:
            for fpath, coords_list in rejection_map.items():
                f.write(f"FILE: {fpath}\n")
                for (x, y) in coords_list:
                    # Convert to Python int in case they're NumPy int64
                    f.write(f"{int(x)}, {int(y)}\n")
                f.write("\n")  # blank line to separate blocks

    def load_rejection_map_sasr(self, in_file):
        """
        Reads a .sasr text file and rebuilds the rejection map dictionary.
        Returns a dict { fpath: [(x, y), (x, y), ...], ... }
        """
        rejections = {}
        with open(in_file, "r") as f:
            content = f.read().strip()

        # Split on blank lines
        blocks = re.split(r"\n\s*\n", content)
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue

            # First line should be 'FILE: <path>'
            if lines[0].startswith("FILE:"):
                raw_path = lines[0].replace("FILE:", "").strip()
                coords = []
                for line in lines[1:]:
                    # Each subsequent line is "x, y"
                    parts = line.split(",")
                    if len(parts) == 2:
                        x_str, y_str = parts
                        x = int(x_str.strip())
                        y = int(y_str.strip())
                        coords.append((x, y))
                rejections[raw_path] = coords
        return rejections


    @pyqtSlot(list, dict, result=object)   # (files: list[str], initial_xy: dict[str, (x,y)]) -> dict|None
    def show_comet_preview(self, files, initial_xy):
        dlg = CS.CometCentroidPreview(files, initial_xy=initial_xy, parent=self)
        if dlg.exec() == int(QDialog.DialogCode.Accepted):
            return dlg.get_seeds()
        return None

    @pyqtSlot(list, dict, object)
    def on_need_comet_review(self, files, initial_xy, responder):
        # This runs on the GUI thread.
        dlg = CS.CometCentroidPreview(files, initial_xy=initial_xy, parent=self)
        if dlg.exec() == int(QDialog.DialogCode.Accepted):
            result = dlg.get_seeds()
        else:
            result = None
        responder.finished.emit(result)

    def integrate_comet_aligned(
        self,
        group_key: str,
        file_list: list[str],
        comet_xy: dict[str, tuple[float,float]],
        frame_weights: dict[str, float],
        status_cb=None,
        *,
        algo_override: str | None = None
    ):
        
        debug_starrem = bool(self.settings.value("stacking/comet_starrem/debug_dump", False, type=bool))
        debug_dir = os.path.join(self.stacking_directory, "debug_comet_starrem")
        os.makedirs(debug_dir, exist_ok=True)        
        """
        Translate each frame so its comet centroid lands on a single reference pixel
        (from file_list[0]). Optional comet star-removal runs AFTER this alignment,
        with a single fixed core mask in comet space. No NaNs; reduction uses the
        selected rejection algorithm.
        """
        log = status_cb or (lambda *_: None)
        if not file_list:
            return None, {}, None

        # --- Reference frame / canvas shape ---
        ref_file = file_list[0]
        ref_img, ref_header, _, _ = load_image(ref_file)
        if ref_img is None:
            log(f"⚠️ Could not load reference '{ref_file}' for comet stack.")
            return None, {}, None

        is_color = (ref_img.ndim == 3 and ref_img.shape[2] == 3)
        H, W = ref_img.shape[:2]
        C = 3 if is_color else 1

        # The single pixel we align to (in ref frame):
        ref_xy = comet_xy[ref_file]
        log(f"📌 Comet reference pixel @ {ref_file} → ({ref_xy[0]:.2f},{ref_xy[1]:.2f})")

        # --- Open sources (mem-mapped readers) ---
        sources = []
        try:
            for p in file_list:
                sources.append(_MMImage(p))   # << was _MMFits
        except Exception as e:
            for s in sources:
                try: s.close()
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            log(f"⚠️ Failed to open images (memmap): {e}")
            return None, {}, None

        DTYPE = self._dtype()
        integrated_image = np.zeros((H, W, C), dtype=DTYPE)
        per_file_rejections = {p: [] for p in file_list}

        # --- Chunking (same policy as normal integration) ---
        pref_h, pref_w = self.chunk_height, self.chunk_width
        try:
            chunk_h, chunk_w = compute_safe_chunk(H, W, len(file_list), C, DTYPE, pref_h, pref_w)
            log(f"🔧 Comet stack chunk {chunk_h}×{chunk_w}")
        except MemoryError as e:
            for s in sources:
                try: s.close()
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
            log(f"⚠️ {e}")
            return None, {}, None

        # Reusable tile buffer
        ts_buf = np.empty((len(file_list), chunk_h, chunk_w, C), dtype=np.float32, order='F')
        weights_array = np.array([frame_weights.get(p, 1.0) for p in file_list], dtype=np.float32)

        # Rejection maps (for MEF layers)
        rej_any   = np.zeros((H, W), dtype=np.bool_)
        rej_count = np.zeros((H, W), dtype=np.uint16)

        # --- Per-frame pure-translation affines (into comet space) ---
        affines = {}
        for p in file_list:
            cx, cy = comet_xy[p]
            dx = ref_xy[0] - cx
            dy = ref_xy[1] - cy
            affines[p] = np.array([[1.0, 0.0, dx],
                                [0.0, 1.0, dy]], dtype=np.float32)

        # ---------- OPTIONAL comet star removal (pre-process per frame) ----------
        csr_enabled = self.settings.value("stacking/comet_starrem/enabled", False, type=bool)
        csr_tool    = self.settings.value("stacking/comet_starrem/tool", "StarNet", type=str)
        core_r      = float(self.settings.value("stacking/comet_starrem/core_r", 22.0, type=float))
        core_soft   = float(self.settings.value("stacking/comet_starrem/core_soft", 6.0, type=float))

        csr_outputs_are_aligned = False   # tells the tile loop whether to warp again
        tmp_root = None
        starless_temp_paths: list[str] | None = None

        if csr_enabled:
            log("✨ Comet star removal enabled — pre-processing frames…")

            # Build a single core-protection mask in comet-aligned coords (center = ref_xy)
            core_mask = CS._protect_core_mask(H, W, ref_xy[0], ref_xy[1], core_r, core_soft).astype(np.float32)

            starless_temp_paths = []
            starless_map = {}  # ← add this
            tmp_root = tempfile.mkdtemp(prefix="sas_comet_starless_")
            try:
                for i, p in enumerate(file_list, 1):
                    try:
                        src = sources[i-1].read_full()  # float32 (H,W) or (H,W,3)
                        # Ensure 3ch for the external tools
                        if src.ndim == 2:
                            src = src[..., None]
                        if src.shape[2] == 1:
                            src = np.repeat(src, 3, axis=2)

                        # Warp into comet space once (so the same mask applies to all frames)
                        M = affines[p]
                        warped = cv2.warpAffine(
                            src, M, (W, H),
                            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
                        ).astype(np.float32, copy=False)

                        # Run chosen remover in comet space
                        if csr_tool == "CosmicClarityDarkStar":
                            log("  ◦ DarkStar comet star removal…")
                            starless = CS.darkstar_starless_from_array(warped, self.settings)
                            orig_for_blend = warped

                            m3 = _expand_mask_for(warped, core_mask)
                            protected = np.clip(starless * (1.0 - m3) + orig_for_blend * m3, 0.0, 1.0).astype(np.float32)                            
                        else:
                            log("  ◦ StarNet comet star removal…")
                            # Frames are linear at this stage
                            protected, _ = CS.starnet_starless_pair_from_array(
                                warped, self.settings, is_linear=True,
                                debug_save_dir=debug_dir, debug_tag=f"{i:04d}_{os.path.splitext(os.path.basename(p))[0]}", core_mask=core_mask
                            )
                            protected = np.clip(protected, 0.0, 1.0).astype(np.float32)


                        # Persist as temp FITS (comet-aligned)
                        outp = os.path.join(tmp_root, f"starless_{i:04d}.fit")
                        save_image(
                            img_array=protected,
                            filename=outp,
                            original_format="fit",
                            bit_depth="32-bit floating point",
                            original_header=ref_header,  # simple header OK
                            is_mono=False
                        )
                        if self.settings.value("stacking/comet/save_starless", False, type=bool):
                            save_dir = os.path.join(self.stacking_directory, "starless_comet_aligned")
                            os.makedirs(save_dir, exist_ok=True)
                            base_name = os.path.splitext(os.path.basename(p))[0]
                            out_user = os.path.join(save_dir, f"{base_name}_starless.fit")
                            save_image(
                                img_array=protected,
                                filename=out_user,
                                original_format="fit",
                                bit_depth="32-bit floating point",
                                original_header=ref_header,
                                is_mono=False
                            )                        
                        starless_temp_paths.append(outp)
                        starless_map[p] = outp    
                        log(f"    ✓ [{i}/{len(file_list)}] starless saved")
                    except Exception as e:
                        log(f"  ⚠️ star removal failed on {os.path.basename(p)}: {e}")
                        # Fallback: use the warped original (still comet-aligned)
                        outp = os.path.join(tmp_root, f"starless_{i:04d}.fit")
                        save_image(
                            img_array=warped.astype(np.float32, copy=False),
                            filename=outp,
                            original_format="fit",
                            bit_depth="32-bit floating point",
                            original_header=ref_header,
                            is_mono=False
                        )
                        starless_temp_paths.append(outp)

                # Swap readers to the comet-aligned starless temp files
                for s in sources:
                    try: s.close()
                    except Exception as e:
                        import logging
                        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                sources = [_MMFits(p) for p in starless_temp_paths]
                starless_readers_paths = list(starless_temp_paths)  

                # These temp frames are already comet-aligned ⇒ no further warp in tile loop
                for p in file_list:
                    affines[p] = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]], dtype=np.float32)
                csr_outputs_are_aligned = True
                self._last_comet_used_starless = True                    # ← record for UI/summary
                log(f"✨ Using comet-aligned STARLESS frames for stack ({len(starless_temp_paths)} files).")
            except Exception as e:
                log(f"⚠️ Comet star removal pre-process aborted: {e}")
                csr_outputs_are_aligned = False
                self._last_comet_used_starless = False

        # --- Tile loop ---
        t_idx = 0
        for y0 in range(0, H, chunk_h):
            y1 = min(y0 + chunk_h, H); th = y1 - y0
            for x0 in range(0, W, chunk_w):
                x1 = min(x0 + chunk_w, W); tw = x1 - x0
                t_idx += 1
                log(f"Integrating comet tile {t_idx}…")
                if csr_outputs_are_aligned:
                    log("   • Tile source: STARLESS (pre-aligned)")

                ts = ts_buf[:, :th, :tw, :C]

                for i, src in enumerate(sources):
                    full = src.read_full()  # (H,W) or (H,W,3) float32

                    # --- sanity: ensure this reader corresponds to the original file index
                    if csr_outputs_are_aligned:
                        # Optional soft sanity check (index-based)
                        expected = os.path.normpath(starless_readers_paths[i])
                        actual   = os.path.normpath(getattr(src, "path", expected))
                        if actual != expected:
                            log(f"   ⚠️ Starless reader path mismatch at i={i}; "
                                f"got {os.path.basename(actual)}, expected {os.path.basename(expected)}. Using index order.")

                    if csr_outputs_are_aligned:
                        # Already comet-aligned; just slice the tile
                        if C == 1:
                            if full.ndim == 3:
                                full = full[..., 0]  # collapse RGB→mono (same as stars stack behavior)
                            tile = full[y0:y1, x0:x1]
                            ts[i, :, :, 0] = tile
                        else:
                            if full.ndim == 2:
                                full = full[..., None].repeat(3, axis=2)
                            ts[i, :, :, :] = full[y0:y1, x0:x1, :]
                    else:
                        # Warp into comet space on the fly
                        M = affines[file_list[i]]
                        if C == 1:
                            full2d = full[..., 0] if full.ndim == 3 else full
                            warped2d = cv2.warpAffine(full2d, M, (W, H),
                                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                            ts[i, :, :, 0] = warped2d[y0:y1, x0:x1]
                        else:
                            if full.ndim == 2:
                                full = full[..., None].repeat(3, axis=2)
                            warped = cv2.warpAffine(full, M, (W, H),
                                                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                            ts[i, :, :, :] = warped[y0:y1, x0:x1, :]

                # --- Apply selected rejection algorithm ---
                algo = (algo_override or self.rejection_algorithm)
                log(f"  ◦ applying rejection algorithm: {algo}")

                if algo in ("Comet Median", "Simple Median (No Rejection)"):
                    tile_result  = np.median(ts, axis=0)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Comet High-Clip Percentile":
                    k = self.settings.value("stacking/comet_hclip_k", 1.30, type=float)
                    p = self.settings.value("stacking/comet_hclip_p", 25.0, type=float)
                    # keep a small dict across tiles to reuse scratch buffers

                    tile_result = _high_clip_percentile(ts, k=float(k), p=float(p))
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Comet Lower-Trim (30%)":
                    tile_result  = _lower_trimmed_mean(ts, trim_hi_frac=0.30)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Comet Percentile (40th)":
                    tile_result  = _percentile40(ts)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Simple Average (No Rejection)":
                    tile_result  = np.average(ts, axis=0, weights=weights_array)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                elif algo == "Weighted Windsorized Sigma Clipping":
                    tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                        ts, weights_array, lower=self.sigma_low, upper=self.sigma_high
                    )

                elif algo == "Kappa-Sigma Clipping":
                    tile_result, tile_rej_map = kappa_sigma_clip_weighted(
                        ts, weights_array, kappa=self.kappa, iterations=self.iterations
                    )

                elif algo == "Trimmed Mean":
                    tile_result, tile_rej_map = trimmed_mean_weighted(
                        ts, weights_array, trim_fraction=self.trim_fraction
                    )

                elif algo == "Extreme Studentized Deviate (ESD)":
                    tile_result, tile_rej_map = esd_clip_weighted(
                        ts, weights_array, threshold=self.esd_threshold
                    )

                elif algo == "Biweight Estimator":
                    tile_result, tile_rej_map = biweight_location_weighted(
                        ts, weights_array, tuning_constant=self.biweight_constant
                    )

                elif algo == "Modified Z-Score Clipping":
                    tile_result, tile_rej_map = modified_zscore_clip_weighted(
                        ts, weights_array, threshold=self.modz_threshold
                    )

                elif algo == "Max Value":
                    tile_result, tile_rej_map = max_value_stack(ts, weights_array)

                else:
                    # default to comet-safe median
                    tile_result  = np.median(ts, axis=0)
                    tile_rej_map = np.zeros((len(file_list), th, tw), dtype=bool)

                integrated_image[y0:y1, x0:x1, :] = tile_result

                # Accumulate rejection bookkeeping
                trm = tile_rej_map
                if trm.ndim == 4:
                    trm = np.any(trm, axis=-1)  # (N, th, tw)
                rej_any[y0:y1, x0:x1]  |= np.any(trm, axis=0)
                rej_count[y0:y1, x0:x1] += trm.sum(axis=0).astype(np.uint16)

                for i, fpath in enumerate(file_list):
                    ys, xs = np.where(trm[i])
                    if ys.size:
                        per_file_rejections[fpath].extend(zip(x0 + xs, y0 + ys))

        # Close readers and clean temp
        for s in sources:
            try: s.close()
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        if tmp_root is not None:
            try: shutil.rmtree(tmp_root, ignore_errors=True)
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        if C == 1:
            integrated_image = integrated_image[..., 0]

        # Store MEF rejection maps for this comet stack
        if not hasattr(self, "_rej_maps"):
            self._rej_maps = {}
        rej_frac = (rej_count.astype(np.float32) / float(max(1, len(file_list))))
        self._rej_maps[group_key + " (COMET)"] = {
            "any":   rej_any,
            "frac":  rej_frac,
            "count": rej_count,
            "n":     len(file_list),
        }

        return integrated_image, per_file_rejections, ref_header


    def save_registered_images(self, success, msg, frame_weights):
        if not success:
            self.update_status(f"⚠️ Image registration failed: {msg}")
            return

        self.update_status("✅ All frames registered successfully!")
        QApplication.processEvents()
        
        # Use the grouped files already stored from the tree view.
        if not self.light_files:
            self.update_status("⚠️ No light frames available for stacking!")
            return
        
        self.update_status(f"📂 Preparing to stack {sum(len(v) for v in self.light_files.values())} frames in {len(self.light_files)} groups.")
        QApplication.processEvents()
        
        # Pass the dictionary (grouped by filter, exposure, dimensions) to the stacking function.
        self.stack_registered_images(self.light_files, frame_weights)

    def _mmcache_dir(self) -> str:
        d = os.path.join(self.stacking_directory, "_mmcache")
        os.makedirs(d, exist_ok=True)
        return d

    def _memmap_key(self, file_path: str) -> str:
        """Stable key bound to path + size + mtime (regen if source changes)."""
        st = os.stat(file_path)
        sig = f"{os.path.abspath(file_path)}|{st.st_size}|{st.st_mtime_ns}"
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]

    def _ensure_float32_memmap(self, file_path: str) -> tuple[str, tuple[int,int,int]]:
        """
        Ensure a (H,W,C float32) .npy exists for file_path. Returns (npy_path, shape).
        We keep C=1 for mono, C=3 for color. Values in [0..1].
        """
        key = self._memmap_key(file_path)
        npy_path = os.path.join(self._mmcache_dir(), f"{key}.npy")
        if os.path.exists(npy_path):
            # Shape header is embedded in the .npy; we’ll read when opening.
            return npy_path, None

        img, hdr, _, _ = load_image(file_path)
        if img is None:
            raise RuntimeError(f"Could not load {file_path} to create memmap cache.")

        # Normalize → float32 [0..1], ensure channels-last (H,W,C).
        if img.ndim == 2:
            arr = img.astype(np.float32, copy=False)
            if arr.dtype == np.uint16: arr = arr / 65535.0
            elif arr.dtype == np.uint8: arr = arr / 255.0
            else: arr = np.clip(arr, 0.0, 1.0)
            arr = arr[..., None]  # (H,W,1)
        elif img.ndim == 3:
            if img.shape[0] == 3 and img.shape[2] != 3:
                img = np.transpose(img, (1,2,0))  # (H,W,3)
            arr = img.astype(np.float32, copy=False)
            if arr.dtype == np.uint16: arr = arr / 65535.0
            elif arr.dtype == np.uint8: arr = arr / 255.0
            else: arr = np.clip(arr, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported image ndim={img.ndim} for {file_path}")

        H, W, C = arr.shape
        mm = open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(H, W, C))
        mm[:] = arr  # single write
        del mm
        return npy_path, (H, W, C)

    def _open_memmaps_readonly(self, paths: list[str]) -> dict[str, np.memmap]:
        """Open all cached arrays in read-only mmap mode."""
        views = {}
        for p in paths:
            npy, _ = self._ensure_float32_memmap(p)
            views[p] = np.load(npy, mmap_mode="r")  # returns numpy.memmap
        return views


    def stack_registered_images_chunked(
        self,
        grouped_files,
        frame_weights,
        chunk_height=2048,
        chunk_width=2048
    ):
        self.update_status(f"✅ Chunked stacking {len(grouped_files)} group(s)...")
        QApplication.processEvents()

        all_rejection_coords = []

        for group_key, file_list in grouped_files.items():
            num_files = len(file_list)
            self.update_status(f"📊 Group '{group_key}' has {num_files} aligned file(s).")
            QApplication.processEvents()
            if num_files < 2:
                self.update_status(f"⚠️ Group '{group_key}' does not have enough frames to stack.")
                continue

            # Reference shape/header (unchanged)
            ref_file = file_list[0]
            if not os.path.exists(ref_file):
                self.update_status(f"⚠️ Reference file '{ref_file}' not found, skipping group.")
                continue

            ref_data, ref_header, _, _ = load_image(ref_file)
            if ref_data is None:
                self.update_status(f"⚠️ Could not load reference '{ref_file}', skipping group.")
                continue

            is_color = (ref_data.ndim == 3 and ref_data.shape[2] == 3)
            height, width = ref_data.shape[:2]
            channels = 3 if is_color else 1

            # Final output memmap (unchanged)
            memmap_path = self._build_out(self.stacking_directory, f"chunked_{group_key}", "dat")
            final_stacked = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(height, width, channels))

            # Valid files + weights
            aligned_paths, weights_list = [], []
            for fpath in file_list:
                if os.path.exists(fpath):
                    aligned_paths.append(fpath)
                    weights_list.append(frame_weights.get(fpath, 1.0))
                else:
                    self.update_status(f"⚠️ File not found: {fpath}, skipping.")
            if len(aligned_paths) < 2:
                self.update_status(f"⚠️ Not enough valid frames in group '{group_key}' to stack.")
                continue

            weights_list = np.array(weights_list, dtype=np.float32)

            # ⬇️ NEW: open read-only memmaps for all aligned frames (float32 [0..1], HxWxC)
            mm_views = self._open_memmaps_readonly(aligned_paths)

            self.update_status(f"📊 Stacking group '{group_key}' with {self.rejection_algorithm}")
            QApplication.processEvents()

            rejection_coords = []
            N = len(aligned_paths)
            DTYPE  = self._dtype()
            pref_h = self.chunk_height
            pref_w = self.chunk_width

            try:
                chunk_h, chunk_w = compute_safe_chunk(height, width, N, channels, DTYPE, pref_h, pref_w)
                self.update_status(f"🔧 Using chunk size {chunk_h}×{chunk_w} for {self._dtype()}")
            except MemoryError as e:
                self.update_status(f"⚠️ {e}")
                return None, {}, None

            # Tile loop (same structure, but tile loading reads from memmaps)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            LOADER_WORKERS = min(max(2, (os.cpu_count() or 4) // 2), 8)  # tuned for memory bw

            for y_start in range(0, height, chunk_h):
                y_end  = min(y_start + chunk_h, height)
                tile_h = y_end - y_start

                for x_start in range(0, width, chunk_w):
                    x_end  = min(x_start + chunk_w, width)
                    tile_w = x_end - x_start

                    # Preallocate tile stack
                    tile_stack = np.empty((N, tile_h, tile_w, channels), dtype=np.float32)

                    # ⬇️ NEW: fill tile_stack from the memmaps (parallel copy)
                    def _copy_one(i, path):
                        v = mm_views[path][y_start:y_end, x_start:x_end]  # view on disk
                        if v.ndim == 2:
                            # mono memmap stored as (H,W,1); but if legacy mono npy exists as (H,W),
                            # make it (H,W,1) here:
                            vv = v[..., None]
                        else:
                            vv = v
                        if vv.shape[2] == 1 and channels == 3:
                            vv = np.repeat(vv, 3, axis=2)
                        tile_stack[i] = vv

                    with ThreadPoolExecutor(max_workers=LOADER_WORKERS) as exe:
                        futs = {exe.submit(_copy_one, i, p): i for i, p in enumerate(aligned_paths)}
                        for _ in as_completed(futs):
                            pass

                    # Rejection (unchanged – uses your Numba kernels)
                    algo = self.rejection_algorithm
                    if algo == "Simple Median (No Rejection)":
                        tile_result  = np.median(tile_stack, axis=0)
                        tile_rej_map = np.zeros(tile_stack.shape[1:3], dtype=np.bool_)
                    elif algo == "Simple Average (No Rejection)":
                        tile_result  = np.average(tile_stack, axis=0, weights=weights_list)
                        tile_rej_map = np.zeros(tile_stack.shape[1:3], dtype=np.bool_)
                    elif algo == "Weighted Windsorized Sigma Clipping":
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            tile_stack, weights_list, lower=self.sigma_low, upper=self.sigma_high
                        )
                    elif algo == "Kappa-Sigma Clipping":
                        tile_result, tile_rej_map = kappa_sigma_clip_weighted(
                            tile_stack, weights_list, kappa=self.kappa, iterations=self.iterations
                        )
                    elif algo == "Trimmed Mean":
                        tile_result, tile_rej_map = trimmed_mean_weighted(
                            tile_stack, weights_list, trim_fraction=self.trim_fraction
                        )
                    elif algo == "Extreme Studentized Deviate (ESD)":
                        tile_result, tile_rej_map = esd_clip_weighted(
                            tile_stack, weights_list, threshold=self.esd_threshold
                        )
                    elif algo == "Biweight Estimator":
                        tile_result, tile_rej_map = biweight_location_weighted(
                            tile_stack, weights_list, tuning_constant=self.biweight_constant
                        )
                    elif algo == "Modified Z-Score Clipping":
                        tile_result, tile_rej_map = modified_zscore_clip_weighted(
                            tile_stack, weights_list, threshold=self.modz_threshold
                        )
                    elif algo == "Max Value":
                        tile_result, tile_rej_map = max_value_stack(
                            tile_stack, weights_list
                        )
                    else:
                        tile_result, tile_rej_map = windsorized_sigma_clip_weighted(
                            tile_stack, weights_list, lower=self.sigma_low, upper=self.sigma_high
                        )

                    # Commit tile
                    final_stacked[y_start:y_end, x_start:x_end, :] = tile_result

                    # Collect per-tile rejection coords (unchanged logic)
                    if tile_rej_map.ndim == 3:          # (N, tile_h, tile_w)
                        combined_rej = np.any(tile_rej_map, axis=0)
                    elif tile_rej_map.ndim == 4:        # (N, tile_h, tile_w, C)
                        combined_rej = np.any(tile_rej_map, axis=0)
                        combined_rej = np.any(combined_rej, axis=-1)
                    else:
                        combined_rej = np.zeros((tile_h, tile_w), dtype=np.bool_)

                    ys_tile, xs_tile = np.where(combined_rej)
                    for dy, dx in zip(ys_tile, xs_tile):
                        rejection_coords.append((x_start + dx, y_start + dy))

            # Finish/save (unchanged from your version) …
            final_array = np.array(final_stacked)
            del final_stacked

            flat = final_array.ravel()
            nz = np.where(flat > 0)[0]
            if nz.size > 0:
                final_array -= flat[nz[0]]

            new_max = final_array.max()
            if new_max > 1.0:
                new_min = final_array.min()
                rng = new_max - new_min
                final_array = (final_array - new_min) / rng if rng != 0 else np.zeros_like(final_array, np.float32)

            if final_array.ndim == 3 and final_array.shape[-1] == 1:
                final_array = final_array[..., 0]
            is_mono = (final_array.ndim == 2)

            if ref_header is None:
                ref_header = fits.Header()
            ref_header["IMAGETYP"] = "MASTER STACK"
            ref_header["BITPIX"] = -32
            ref_header["STACKED"] = (True, "Stacked using chunked approach")
            ref_header["CREATOR"] = "SetiAstroSuite"
            ref_header["DATE-OBS"] = datetime.utcnow().isoformat()
            if is_mono:
                ref_header["NAXIS"]  = 2
                ref_header["NAXIS1"] = final_array.shape[1]
                ref_header["NAXIS2"] = final_array.shape[0]
                if "NAXIS3" in ref_header: del ref_header["NAXIS3"]
            else:
                ref_header["NAXIS"]  = 3
                ref_header["NAXIS1"] = final_array.shape[1]
                ref_header["NAXIS2"] = final_array.shape[0]
                ref_header["NAXIS3"] = 3

            output_stem = f"MasterLight_{group_key}_{len(aligned_paths)}stacked"
            output_path  = self._build_out(self.stacking_directory, output_stem, "fit")

            save_image(
                img_array=final_array,
                filename=output_path,
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=ref_header,
                is_mono=is_mono
            )

            self.update_status(f"✅ Group '{group_key}' stacked {len(aligned_paths)} frame(s)! Saved: {output_path}")

            print(f"✅ Master Light saved for group '{group_key}': {output_path}")

            # Optionally, you might want to store or log 'rejection_coords' (here appended to all_rejection_coords)
            all_rejection_coords.extend(rejection_coords)

            # Clean up memmap file
            try:
                os.remove(memmap_path)
            except OSError:
                pass

        QMessageBox.information(
            self,
            "Stacking Complete",
            f"All stacking finished successfully.\n"
            f"Frames per group:\n" +
            "\n".join([f"{group_key}: {len(files)} frame(s)" for group_key, files in grouped_files.items()])
        )

        # Optionally, you could return the global rejection coordinate list.
        return all_rejection_coords        

    def _start_after_align_worker(self, aligned_light_files: dict[str, list[str]]):
        # Snapshot UI settings
        if getattr(self, "_suppress_normal_integration_once", False):
            self._suppress_normal_integration_once = False
            self.update_status("⏭️ Normal integration suppressed (MFDeconv-only run).")
            self._set_registration_busy(False)
            return        
        drizzle_dict = self.gather_drizzle_settings_from_tree()
        try:
            autocrop_enabled = self.autocrop_cb.isChecked()
            autocrop_pct = float(self.autocrop_pct.value())
        except Exception:
            autocrop_enabled = self.settings.value("stacking/autocrop_enabled", False, type=bool)
            autocrop_pct = float(self.settings.value("stacking/autocrop_pct", 95.0, type=float))

        # CFA fill log (optional)
        if getattr(self, "valid_matrices", None):
            try:
                cfa_effective = bool(
                    self._cfa_for_this_run
                    if getattr(self, "_cfa_for_this_run", None) is not None
                    else (getattr(self, "cfa_drizzle_cb", None) and self.cfa_drizzle_cb.isChecked())
                )
                if cfa_effective:
                    fill = self._dither_phase_fill(self.valid_matrices, bins=8)
                    self.update_status(f"🔎 CFA drizzle sub-pixel phase fill (8×8): {fill*100:.1f}%")
            except Exception:
                pass

        # Launch the normal post-align worker
        self.post_thread = QThread(self)
        self.post_worker = AfterAlignWorker(
            self,
            light_files=aligned_light_files,
            frame_weights=dict(self.frame_weights),
            transforms_dict=dict(self.valid_transforms),
            drizzle_dict=drizzle_dict,
            autocrop_enabled=autocrop_enabled,
            autocrop_pct=autocrop_pct,
            ui_owner=self
        )
        self.post_worker.ui_owner = self
        self.post_worker.need_comet_review.connect(self.on_need_comet_review)
        self.post_worker.progress.connect(self._on_post_status)
        self.post_worker.finished.connect(self._on_post_pipeline_finished)
        self.post_worker.moveToThread(self.post_thread)
        self.post_thread.started.connect(self.post_worker.run)
        self.post_thread.start()

        self.post_progress = QProgressDialog("Stacking & drizzle (if enabled)…", None, 0, 0, self)
        self.post_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.post_progress.setCancelButton(None)
        self.post_progress.setMinimumDuration(0)
        self.post_progress.setWindowTitle("Post-Alignment")
        self.post_progress.show()

        # Important for button state
        self._set_registration_busy(False)


    def _pd_alive(self):
        pd = getattr(self, "_mf_pd", None)
        if pd is None:
            return None
        # If Qt already destroyed it, skip
        if sip.isdeleted(pd):
            return None
        return pd



    def _run_mfdeconv_then_continue(self, aligned_light_files: dict[str, list[str]]):
        """Queue MFDeconv per group if enabled, then continue into AfterAlignWorker for all groups."""
        mf_enabled = self.settings.value("stacking/mfdeconv/enabled", False, type=bool)
        if not mf_enabled:
            self._start_after_align_worker(aligned_light_files)
            return

        # Build list of non-empty groups
        mf_groups = [(g, lst) for g, lst in aligned_light_files.items() if lst]
        if not mf_groups:
            self.update_status("⚠️ No aligned frames available for MF deconvolution.")
            self._start_after_align_worker(aligned_light_files)
            return

        # Progress UI for the entire MF phase
        self._mf_total_groups = len(mf_groups)
        self._mf_groups_done = 0
        self._mf_pd = QProgressDialog("Multi-frame deconvolving…", "Cancel", 0, self._mf_total_groups * 1000, self)
        self._mf_pd.setValue(0)
        # self._mf_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._mf_pd.setMinimumDuration(0)
        self._mf_pd.setWindowTitle("MF Deconvolution")
        self._mf_pd.setRange(0, self._mf_total_groups * 1000)
        self._mf_pd.setValue(0)
        self._mf_pd.show()

        self._mf_queue = list(mf_groups)
        self._mf_results = {}
        self._mf_cancelled = False
        self._mf_thread = None
        self._mf_worker = None

        def _cancel_all():
            self._mf_cancelled = True
        self._mf_pd.canceled.connect(_cancel_all, Qt.ConnectionType.QueuedConnection)

        def _start_next():
            # End of queue or canceled → finish gracefully
            if self._mf_cancelled or not self._mf_queue:
                if getattr(self, "_mf_pd", None):
                    pd = self._pd_alive()
                    if pd:
                        pd.reset()
                        pd.deleteLater()
                    self._mf_pd = None
                try:
                    if self._mf_thread:
                        self._mf_thread.quit()
                        self._mf_thread.wait()
                except Exception:
                    pass
                self._mf_thread = None
                self._mf_worker = None

                # Continue the normal pipeline for ALL groups
                self._suppress_normal_integration_once = True
                self.update_status("✅ MFDeconv complete for all groups. Skipping normal integration.")
                self._set_registration_busy(False)
                return

            group_key, frames = self._mf_queue.pop(0)
            out_dir = os.path.join(self.stacking_directory, "Masters")
            os.makedirs(out_dir, exist_ok=True)

            # Settings snapshot
            iters = self.settings.value("stacking/mfdeconv/iters", 20, type=int)
            min_iters = self.settings.value("stacking/mfdeconv/min_iters", 3, type=int)
            kappa = self.settings.value("stacking/mfdeconv/kappa", 2.0, type=float)
            mode = self.mf_color_combo.currentText()
            Huber = self.settings.value("stacking/mfdeconv/Huber_delta", 0.0, type=float)
            save_intermediate = self.mf_save_intermediate_cb.isChecked()
            seed_mode_cfg = str(self.settings.value("stacking/mfdeconv/seed_mode", "robust"))
            use_star_masks = self.mf_use_star_mask_cb.isChecked()
            use_variance_maps = self.mf_use_noise_map_cb.isChecked()
            rho = self.mf_rho_combo.currentText()

            star_mask_cfg = {
                "thresh_sigma":  self.settings.value("stacking/mfdeconv/star_mask/thresh_sigma",  _SM_DEF_THRESH, type=float),
                "grow_px":       self.settings.value("stacking/mfdeconv/star_mask/grow_px",       _SM_DEF_GROW, type=int),
                "soft_sigma":    self.settings.value("stacking/mfdeconv/star_mask/soft_sigma",    _SM_DEF_SOFT, type=float),
                "max_radius_px": self.settings.value("stacking/mfdeconv/star_mask/max_radius_px", _SM_DEF_RMAX, type=int),
                "max_objs":      self.settings.value("stacking/mfdeconv/star_mask/max_objs",      _SM_DEF_MAXOBJS, type=int),
                "keep_floor":    self.settings.value("stacking/mfdeconv/star_mask/keep_floor",    _SM_DEF_KEEPF, type=float),
                "ellipse_scale": self.settings.value("stacking/mfdeconv/star_mask/ellipse_scale", _SM_DEF_ES, type=float),
            }
            varmap_cfg = {
                "sample_stride": self.settings.value("stacking/mfdeconv/varmap/sample_stride", _VM_DEF_STRIDE, type=int),
                "smooth_sigma":  self.settings.value("stacking/mfdeconv/varmap/smooth_sigma", 1.0, type=float),
                "floor":         self.settings.value("stacking/mfdeconv/varmap/floor",        1e-8, type=float),
            }

            sr_enabled_ui = self.mf_sr_cb.isChecked()
            sr_factor_ui = getattr(self, "mf_sr_factor_spin", None)
            sr_factor_val = sr_factor_ui.value() if sr_factor_ui is not None else self.settings.value("stacking/mfdeconv/sr_factor", 2, type=int)
            super_res_factor = int(sr_factor_val) if sr_enabled_ui else 1

            # Unique, safe filename
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(group_key)) or "Group"
            out_path = os.path.join(out_dir, f"MasterLight_{safe_name}_{len(frames)}f_MFDeconv_{mode}_{iters}it_k{int(round(kappa*100))}.fit")

            # Thread + worker
            self._mf_thread = QThread(self)
            star_mask_ref = self.reference_frame if use_star_masks else None

            # ── choose engine plainly (Normal / cuDNN-free / High Octane) ─────────────
            # Expect a setting saved by your radio buttons: "normal" | "cudnn" | "sport"
            engine = str(self.settings.value("stacking/mfdeconv/engine", "normal")).lower()

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
                self.update_status(f"⚠️ MFDeconv engine import failed ({e}); falling back to Normal.")
                from pro.mfdeconv import MultiFrameDeconvWorker as MFCls
                eng_name = "Normal (fallback)"

            self.update_status(f"⚙️ MFDeconv engine: {eng_name}")

            # ── build worker exactly the same in all modes ────────────────────────────
            self._mf_worker = MFCls(
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

            # Wiring
            self._mf_worker.moveToThread(self._mf_thread)
            self._mf_worker.progress.connect(self._on_mf_progress, Qt.ConnectionType.QueuedConnection)
            self._mf_thread.started.connect(self._mf_worker.run, Qt.ConnectionType.QueuedConnection)
            self._mf_worker.finished.connect(self._mf_thread.quit, Qt.ConnectionType.QueuedConnection)
            self._mf_thread.finished.connect(self._mf_worker.deleteLater)    # free worker on thread end
            self._mf_thread.finished.connect(self._mf_thread.deleteLater)    # free thread object

            def _done(ok: bool, message: str, out: str):
                pd = self._pd_alive()
                if pd:
                    try:
                        # if you keep the 0..groups*1000 range, snap to segment boundary:
                        val = min(pd.value() + 1000, pd.maximum())
                        pd.setValue(val)
                        pd.setLabelText(f"{'✅' if ok else '❌'} {group_key}: {message}")
                    except Exception:
                        pass

                if ok and out:
                    self._mf_results[group_key] = out
                else:
                    self.update_status(f"❌ MFDeconv failed for '{group_key}': {message}")

                try:
                    self._mf_thread.quit()
                    self._mf_thread.wait()
                except Exception:
                    pass
                self._mf_thread = None
                self._mf_worker = None

                QTimer.singleShot(0, _start_next)

            self._mf_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)

            self._mf_thread.start()

            if getattr(self, "_mf_pd", None):
                pd = self._pd_alive()
                if pd:
                    pd.setLabelText(f"Deconvolving '{group_key}' ({len(frames)} frames)…")

        QTimer.singleShot(0, _start_next)



    def invert_affine_transform(matrix):
        """
        Inverts a 2x3 affine transformation matrix.
        Given matrix = [[a, b, tx],
                        [c, d, ty]],
        returns the inverse matrix.
        """
        A = matrix[:, :2]
        t = matrix[:, 2]
        A_inv = np.linalg.inv(A)
        t_inv = -A_inv @ t
        inv = np.hstack([A_inv, t_inv.reshape(2, 1)])
        return inv

    @staticmethod
    def apply_affine_transform_point(matrix, x, y):
        """
        Applies a 2x3 affine transformation to a point (x, y).
        Returns the transformed (x, y) coordinates.
        """
        point = np.array([x, y])
        result = matrix[:, :2] @ point + matrix[:, 2]
        return result[0], result[1]

    def _save_alignment_transforms_sasd_v2(
        self,
        *,
        out_path: str,
        ref_shape: tuple[int, int],
        ref_path: str,
        drizzle_xforms: dict,
        fallback_affine: dict,
    ):
        """
        Write SASD v2 with per-file KIND and proper matrix sizes.

        drizzle_xforms: {orig_path: (kind, matrix_or_None)}
            kind ∈ {"affine","homography","poly3","poly4","tps","thin_plate_spline"} (case-insensitive)
            For poly*/tps, matrix must be None (we'll write a sentinel).

        fallback_affine: {orig_path: 2x3}
            Used only if we truly lack model info for a file, or the numeric
            matrix for an affine/homography entry is malformed.
        """

        # --- header dimensions ---
        try:
            Href, Wref = (int(ref_shape[0]), int(ref_shape[1]))
        except Exception:
            Href, Wref = 0, 0

        # --- normalize keys once ---
        def _normdict(d):
            return {os.path.normpath(k): v for k, v in (d or {}).items()}

        dx = _normdict(drizzle_xforms or {})
        fa = _normdict(fallback_affine or {})

        # Deterministic union of originals we know about
        originals = sorted(set(dx.keys()) | set(fa.keys()))

        # Helper to write non-matrix kinds (poly*/tps/unknown) with UNSUPPORTED sentinel
        def _write_poly_or_unknown(_f, _k, _kind_text: str):
            _f.write(f"FILE: {_k}\n")
            _f.write(f"KIND: {_kind_text}\n")
            _f.write("MATRIX:\n")
            _f.write("UNSUPPORTED\n\n")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"REF_SHAPE: {Href}, {Wref}\n")
            f.write(f"REF_PATH: {ref_path}\n")
            f.write("MODEL: mixed\n\n")

            for k in originals:
                kind, M = (None, None)
                if k in dx:
                    kind, M = dx[k]

                kind = (kind or "").strip().lower()

                # --- Decide whether to use fallback affine ---
                # We ONLY use fallback when:
                #   a) there is no kind at all, or
                #   b) kind is affine/homography but matrix is missing/malformed.
                use_fallback_affine = False
                if not kind:
                    use_fallback_affine = True
                elif kind in ("affine", "homography"):
                    # We'll validate below; mark for possible fallback
                    use_fallback_affine = (M is None)

                if use_fallback_affine and (k in fa):
                    kind = "affine"
                    M = np.asarray(fa[k], np.float32).reshape(2, 3)

                # --- Write by kind ---
                if kind == "homography":
                    # Expect 3x3; if bad, try fallback or write UNSUPPORTED homography
                    try:
                        H = np.asarray(M, np.float64).reshape(3, 3)
                    except Exception:
                        if k in fa:
                            # Fallback to affine numbers if available
                            A = np.asarray(fa[k], np.float64).reshape(2, 3)
                            f.write(f"FILE: {k}\n")
                            f.write("KIND: affine\n")
                            f.write("MATRIX:\n")
                            f.write(f"{A[0,0]:.9f}, {A[0,1]:.9f}, {A[0,2]:.9f}\n")
                            f.write(f"{A[1,0]:.9f}, {A[1,1]:.9f}, {A[1,2]:.9f}\n\n")
                        else:
                            _write_poly_or_unknown(f, k, "homography")
                        continue

                    f.write(f"FILE: {k}\n")
                    f.write("KIND: homography\n")
                    f.write("MATRIX:\n")
                    f.write(f"{H[0,0]:.9f}, {H[0,1]:.9f}, {H[0,2]:.9f}\n")
                    f.write(f"{H[1,0]:.9f}, {H[1,1]:.9f}, {H[1,2]:.9f}\n")
                    f.write(f"{H[2,0]:.9f}, {H[2,1]:.9f}, {H[2,2]:.9f}\n\n")
                    continue

                if kind == "affine":
                    # Expect 2x3; if bad, try fallback or write UNSUPPORTED affine
                    try:
                        A = np.asarray(M, np.float64).reshape(2, 3)
                    except Exception:
                        if k in fa:
                            A = np.asarray(fa[k], np.float64).reshape(2, 3)
                        else:
                            _write_poly_or_unknown(f, k, "affine")
                            continue

                    f.write(f"FILE: {k}\n")
                    f.write("KIND: affine\n")
                    f.write("MATRIX:\n")
                    f.write(f"{A[0,0]:.9f}, {A[0,1]:.9f}, {A[0,2]:.9f}\n")
                    f.write(f"{A[1,0]:.9f}, {A[1,1]:.9f}, {A[1,2]:.9f}\n\n")
                    continue

                # Non-matrix kinds we want to preserve (identity deposit on _n_r):
                if kind in ("poly3", "poly4", "tps", "thin_plate_spline"):
                    _write_poly_or_unknown(f, k, kind)
                    continue

                # Unknown but present -> keep KIND so loader can choose strategy; mark matrix unsupported
                if kind:
                    _write_poly_or_unknown(f, k, kind)
                    continue

                # Truly nothing for this file: skip writing a broken block
                # (drizzle will simply not see this frame)
                continue



    def _load_sasd_v2(self, path: str):
        """
        Returns (ref_H, ref_W, xforms) where xforms maps
        normalized original paths → (kind, matrix)
        Supports:
        • HEADER: REF_SHAPE, REF_PATH, MODEL
        • Per-file: FILE, KIND, MATRIX with either 2 rows (affine) or 3 rows (homography)
        """
        if not os.path.exists(path):
            return 0, 0, {}

        ref_H = ref_W = 0
        xforms = {}
        cur_file = None
        cur_kind = None
        reading_matrix = False
        mat_rows = []
        matrix_unsupported = False   # <— NEW

        def _commit_block():
            nonlocal cur_file, cur_kind, mat_rows, matrix_unsupported
            if cur_file and cur_kind:
                k = os.path.normpath(cur_file)
                ck = (cur_kind or "").lower()

                if ck == "homography" and (mat_rows and len(mat_rows) == 3 and len(mat_rows[0]) == 3):
                    M = np.array(mat_rows, dtype=np.float32).reshape(3, 3)
                    xforms[k] = ("homography", M)

                elif ck == "affine" and (mat_rows and len(mat_rows) == 2 and len(mat_rows[0]) == 3):
                    M = np.array(mat_rows, dtype=np.float32).reshape(2, 3)
                    xforms[k] = ("affine", M)

                elif ck.startswith("poly"):
                    # poly3 / poly4 are callable in-memory; SASD stores no numeric matrix
                    xforms[k] = (ck, None)

                elif matrix_unsupported:
                    # Future kinds that explicitly say UNSUPPORTED: keep kind, no matrix
                    xforms[k] = (ck, None)

            cur_file = None
            cur_kind = None
            mat_rows = []
            matrix_unsupported = False  # <— reset

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    if reading_matrix:
                        reading_matrix = False
                    _commit_block()
                    continue

                if reading_matrix:
                    # Handle explicit sentinel
                    if line.upper().startswith("UNSUPPORTED"):
                        matrix_unsupported = True          # <— flag it
                        reading_matrix = False
                        continue
                    # Try numeric row
                    try:
                        row = [float(x.strip()) for x in line.split(",")]
                        mat_rows.append(row)
                    except Exception:
                        # Non-numeric inside MATRIX: treat as unsupported for safety
                        matrix_unsupported = True
                        reading_matrix = False
                    continue

                if line.startswith("REF_SHAPE:"):
                    parts = line.split(":", 1)[1].split(",")
                    if len(parts) >= 2:
                        ref_H = int(float(parts[0].strip()))
                        ref_W = int(float(parts[1].strip()))
                    continue

                if line.startswith("FILE:"):
                    _commit_block()
                    cur_file = line.split(":", 1)[1].strip()
                    continue

                if line.startswith("KIND:"):
                    cur_kind = line.split(":", 1)[1].strip().lower()
                    continue

                if line.startswith("MATRIX:"):
                    reading_matrix = True
                    mat_rows = []
                    matrix_unsupported = False
                    continue

                if line.startswith("MODEL:"):
                    continue

        if reading_matrix or cur_file:
            _commit_block()

        return int(ref_H), int(ref_W), xforms



    def _safe_component(self, s: str, *, replacement:str="_", maxlen:int=180) -> str:
        """
        Sanitize a *single* path component for cross-platform safety.
        - normalizes unicode (NFKC)
        - replaces path separators and illegal chars
        - collapses whitespace to `_`
        - strips leading/trailing dots/spaces (Windows rule)
        - avoids reserved device names (Windows)
        - truncates to maxlen, keeping extension if present
        """
        s = unicodedata.normalize("NFKC", str(s))

        # nuke path separators
        other_sep = "/" if os.sep == "\\" else "\\"
        s = s.replace(os.sep, replacement).replace(other_sep, replacement)

        # replace illegal Windows chars + control chars
        s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, s)

        # collapse whitespace → _
        s = re.sub(r"\s+", replacement, s)

        # allow only [A-Za-z0-9._-()] plus 'x' (for 1234x5678) — tweak if you want
        s = re.sub(r"[^A-Za-z0-9._\-()x]", replacement, s)

        # collapse multiple replacements
        rep = re.escape(replacement)
        s = re.sub(rep + r"+", replacement, s)

        # trim leading/trailing spaces/dots/dashes/underscores
        s = s.strip(" .-_")

        if not s:
            s = "untitled"

        # avoid reserved basenames on Windows (compare stem only)
        stem, ext = os.path.splitext(s)
        if stem.upper() in _WINDOWS_RESERVED:
            s = "_" + s  # prefix underscore

        # enforce maxlen, preserving extension if present
        if len(s) > maxlen:
            stem, ext = os.path.splitext(s)
            keep = max(1, maxlen - len(ext))
            s = stem[:keep].rstrip(" .-_") + ext

        return s

    def _normalize_master_stem(self, stem: str) -> str:
        """
        Clean up common artifacts in master filenames:
        - collapse _-_ / -_ / _- into a single _
        - turn 40.0s → 40s (strip trailing .0…)
        - keep non-integer exposures filename-safe (e.g., 2.5s → 2p5s)
        """
        # 1) collapse weird joiners like "_-_" or "-_" or "_-"
        stem = re.sub(r'(?:_-+|-+_)+', '_', stem)

        # 2) normalize exposures: <number>s
        def _fix_exp(m):
            txt = m.group(1)  # the numeric part
            try:
                val = float(txt)
            except ValueError:
                return m.group(0)  # leave it as-is

            # If it's an integer (e.g., 40.0) → 40s
            if abs(val - round(val)) < 1e-6:
                return f"{int(round(val))}s"
            # Otherwise make it filename-friendly by replacing '.' with '_' → 2_5s
            return txt.replace('.', '_') + 's'

        stem = re.sub(r'(\d+(?:\.\d+)?)s', _fix_exp, stem)

        stem = re.sub(r'\((\d+)x(\d+)\)', r'\1x\2', stem)
        return stem

    def _build_out(self, directory: str, stem: str, ext: str) -> str:
        """
        Join directory + sanitized stem + sanitized extension.
        Ensures parent dir exists.
        """
        ext = (ext or "").lstrip(".").lower() or "fit"
        safe_stem = self._safe_component(stem)
        safe_dir  = os.path.abspath(directory or ".")
        os.makedirs(safe_dir, exist_ok=True)
        return os.path.join(safe_dir, f"{safe_stem}.{ext}")