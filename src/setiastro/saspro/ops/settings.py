# ops.settings.py
from PyQt6.QtWidgets import (
QLineEdit, QDialogButtonBox, QFileDialog, QDialog, QPushButton, QFormLayout,QApplication, QMenu, QScrollArea, QSizePolicy,
    QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QColorDialog, QFontDialog, QSlider)
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QAction, QGuiApplication
import pytz  # for timezone list
from setiastro.saspro.accel_installer import current_backend
import sys, platform
from PyQt6.QtWidgets import QToolButton, QProgressDialog
from PyQt6.QtCore import QThread
# i18n support
from setiastro.saspro.i18n import get_available_languages, get_saved_language, save_language
import importlib.util
import importlib.metadata
import webbrowser
import shutil
import subprocess
import os

class SettingsDialog(QDialog):
    """
    Simple settings UI for external executable paths + WIMS defaults.
    Values are persisted via the provided QSettings instance.
    """
    def __init__(self, parent, settings: QSettings):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Preferences"))
        self.settings = settings

        # Ensure we don't delete on close, so we can cache it
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        # ---- Existing fields (paths, checkboxes, etc.) ----
        self.le_graxpert = QLineEdit()

        self.le_starnet  = QLineEdit()
        self.le_astap    = QLineEdit()

        self.chk_updates_startup = QCheckBox(self.tr("Check for updates on startup"))

        self.le_updates_url = QLineEdit()
        self.le_updates_url.setPlaceholderText("Raw JSON URL (advanced)")
        self.le_updates_url.setMinimumWidth(240)

        btn_reset_updates_url = QPushButton(self.tr("Reset"))
        btn_reset_updates_url.setToolTip(self.tr("Restore default updates URL"))
        btn_reset_updates_url.clicked.connect(
            lambda: self.le_updates_url.setText(
                "https://raw.githubusercontent.com/setiastro/setiastrosuitepro/main/updates.json"
            )
        )

        # Optional: “Check Now…” button
        self.btn_check_now = QPushButton(self.tr("Check Now…"))
        self.btn_check_now.setToolTip(self.tr("Run an update check immediately"))
        self.btn_check_now.setVisible(hasattr(parent, "_check_for_updates_async"))
        self.btn_check_now.clicked.connect(self._check_updates_now_clicked)

        self.chk_save_shortcuts = QCheckBox(self.tr("Save desktop shortcuts on exit"))

        self.cb_theme = QComboBox()
        # Order: Dark, Gray, Light, System, Custom
        self.cb_theme.addItems(["Dark", "Gray", "Light", "System", "Custom"])

        # "Customize…" button for custom theme
        self.btn_theme_custom = QPushButton(self.tr("Customize…"))
        self.btn_theme_custom.setToolTip(self.tr("Edit custom colors and font"))
        self.btn_theme_custom.clicked.connect(self._open_theme_editor)

        # Keep button enabled state in sync with combo
        self.cb_theme.currentIndexChanged.connect(self._on_theme_changed)

        # ---- Language selector ----
        self.cb_language = QComboBox()
        self._lang_codes = list(get_available_languages().keys())   # ["en", "it", "fr", "es"]
        self._lang_names = list(get_available_languages().values()) # ["English", "Italiano", ...]
        self.cb_language.addItems(self._lang_names)
        self._initial_language = "en"  # placeholder, set in refresh_ui

        btn_grax  = QPushButton(self.tr("Browse…")); btn_grax.clicked.connect(lambda: self._browse_into(self.le_graxpert))
        btn_star  = QPushButton(self.tr("Browse…")); btn_star.clicked.connect(lambda: self._browse_into(self.le_starnet))
        btn_astap = QPushButton(self.tr("Browse…")); btn_astap.clicked.connect(lambda: self._browse_into(self.le_astap))

        # Path rows
        row_grax = QHBoxLayout()
        row_grax.setContentsMargins(0, 0, 0, 0)
        row_grax.addWidget(self.le_graxpert, 1)
        row_grax.addWidget(btn_grax)

        row_star = QHBoxLayout()
        row_star.setContentsMargins(0, 0, 0, 0)
        row_star.addWidget(self.le_starnet, 1)
        row_star.addWidget(btn_star)

        row_astap = QHBoxLayout()
        row_astap.setContentsMargins(0, 0, 0, 0)
        row_astap.addWidget(self.le_astap, 1)
        row_astap.addWidget(btn_astap)

        self.le_astrometry = QLineEdit()
        self.le_astrometry.setEchoMode(QLineEdit.EchoMode.Password)

        # ---- WIMS defaults ----
        self.sp_lat = QDoubleSpinBox();  self.sp_lat.setRange(-90.0, 90.0);       self.sp_lat.setDecimals(6)
        self.sp_lon = QDoubleSpinBox();  self.sp_lon.setRange(-180.0, 180.0);     self.sp_lon.setDecimals(6)
        self.le_date = QLineEdit()   # YYYY-MM-DD
        self.le_time = QLineEdit()   # HH:MM
        self.cb_tz   = QComboBox();  self.cb_tz.addItems(pytz.all_timezones)
        self.cb_tz.setEditable(True)
        self.cb_tz.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)        
        self.sp_min_alt = QDoubleSpinBox(); self.sp_min_alt.setRange(0.0, 90.0);  self.sp_min_alt.setDecimals(1)
        self.sp_obj_limit = QSpinBox(); self.sp_obj_limit.setRange(1, 1000)

        self.chk_autostretch_24bit = QCheckBox(self.tr("High-quality autostretch (24-bit; slower)"))
        self.chk_autostretch_24bit.setToolTip(self.tr("Compute autostretch on a 24-bit histogram (smoother gradients)."))

        self.chk_smooth_zoom_settle = QCheckBox(self.tr("Smooth zoom final redraw (higher quality when zoom stops)"))
        self.chk_smooth_zoom_settle.setToolTip(self.tr(
            "When enabled, zooming is fast while scrolling, then a single high-quality redraw occurs after you stop zooming.\n"
            "Disable if you prefer maximum responsiveness or older GPUs/CPUs."
        ))

        self.slider_bg_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg_opacity.setRange(0, 100)
        self._initial_bg_opacity = 50

        self.lbl_bg_opacity_val = QLabel("50%")
        self.lbl_bg_opacity_val.setFixedWidth(48)

        def _on_opacity_changed(val):
            self.lbl_bg_opacity_val.setText(f"{val}%")
            # Update in real time
            self.settings.setValue("display/bg_opacity", val)
            self.settings.sync()
            p = self.parent()
            if p and hasattr(p, "mdi") and hasattr(p.mdi, "viewport"):
                p.mdi.viewport().update()

        self.slider_bg_opacity.valueChanged.connect(_on_opacity_changed)

        row_bg_opacity = QHBoxLayout()
        row_bg_opacity.setContentsMargins(0, 0, 0, 0)
        row_bg_opacity.addWidget(self.slider_bg_opacity, 1)
        row_bg_opacity.addWidget(self.lbl_bg_opacity_val)
        w_bg_opacity = QWidget()
        w_bg_opacity.setLayout(row_bg_opacity)

        # ---- Custom background: choose/clear preview ----
        self.le_bg_path = QLineEdit()
        self.le_bg_path.setReadOnly(True)
        self.le_bg_path.setMinimumWidth(240)
        self._initial_bg_path = ""

        btn_choose_bg = QPushButton(self.tr("Choose Background…"))
        btn_choose_bg.setToolTip(self.tr("Pick a PNG or JPG to use as the application background"))
        btn_choose_bg.clicked.connect(self._choose_background_clicked)

        btn_clear_bg = QPushButton(self.tr("Clear"))
        btn_clear_bg.setToolTip(self.tr("Remove custom background and restore default"))
        btn_clear_bg.clicked.connect(self._clear_background_clicked)

        row_bg_image = QHBoxLayout()
        row_bg_image.setContentsMargins(0, 0, 0, 0)
        row_bg_image.addWidget(self.le_bg_path, 1)
        row_bg_image.addWidget(btn_choose_bg)
        row_bg_image.addWidget(btn_clear_bg)
        w_bg_image = QWidget()
        w_bg_image.setLayout(row_bg_image)

        # ─────────────────────────────────────────────────────────────────────
        # MAIN LAYOUT (SCROLLABLE + RESPONSIVE)
        # ─────────────────────────────────────────────────────────────────────
        root = QVBoxLayout(self)

        # Scroll area prevents clipping on smaller displays
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root.addWidget(self._scroll, 1)

        # Content widget inside scroll area
        self._scroll_content = QWidget()
        self._scroll.setWidget(self._scroll_content)

        self._scroll_root = QVBoxLayout(self._scroll_content)
        self._scroll_root.setContentsMargins(0, 0, 0, 0)
        self._scroll_root.setSpacing(0)

        # Row container that will hold either:
        # - two widgets side-by-side
        # - or a single stacked layout on narrow width
        self._cols_layout = QHBoxLayout()
        self._cols_layout.setContentsMargins(0, 0, 0, 0)
        self._cols_layout.setSpacing(0)
        self._scroll_root.addLayout(self._cols_layout)

        # Column wrapper widgets (so we can reflow them responsively)
        self._left_col_widget = QWidget()
        self._right_col_widget = QWidget()

        left_col = QFormLayout(self._left_col_widget)
        right_col = QFormLayout(self._right_col_widget)

        self.left_col = left_col
        self.right_col = right_col

        for f in (left_col, right_col):
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            # Wrap long rows instead of clipping
            f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            f.setFormAlignment(Qt.AlignmentFlag.AlignTop)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            f.setHorizontalSpacing(10)
            f.setVerticalSpacing(8)

        # ---- Left column: Paths & Integrations ----
        left_col.addRow(QLabel(self.tr("<b>Paths & Integrations</b>")))

        w = QWidget(); w.setLayout(row_grax)
        left_col.addRow(self.tr("GraXpert executable:"), w)

        w = QWidget(); w.setLayout(row_star)
        left_col.addRow(self.tr("StarNet executable:"), w)

        w = QWidget(); w.setLayout(row_astap)
        left_col.addRow(self.tr("ASTAP executable:"), w)

        left_col.addRow(self.tr("Astrometry.net API key:"), self.le_astrometry)
        left_col.addRow(self.chk_save_shortcuts)

        row_theme = QHBoxLayout()
        row_theme.setContentsMargins(0, 0, 0, 0)
        row_theme.addWidget(self.cb_theme, 1)
        row_theme.addWidget(self.btn_theme_custom)
        w_theme = QWidget()
        w_theme.setLayout(row_theme)
        left_col.addRow(self.tr("Theme:"), w_theme)

        left_col.addRow(self.tr("Language:"), self.cb_language)

        # ---- Display ----
        left_col.addRow(QLabel(self.tr("<b>Display</b>")))
        left_col.addRow(self.chk_autostretch_24bit)
        left_col.addRow(
            "",
            QLabel(self.tr("• ON  = 24-bit (best gradient smoothness, slower)\n"
                        "• OFF = 12-bit (faster, still high quality)"))
        )        
        left_col.addRow(self.chk_smooth_zoom_settle)
        left_col.addRow(self.tr("Background Opacity:"), w_bg_opacity)
        left_col.addRow(self.tr("Background Image:"), w_bg_image)

        # ---- Acceleration ----
        left_col.addRow(QLabel(self.tr("<b>Acceleration</b>")))

        # Backend/deps/pref/install split into multi-line compact box (much better on small widths)
        self.backend_label = QLabel(self.tr("Backend: {0}").format(current_backend()))

        self.accel_deps_label = QLabel()
        self.accel_deps_label.setTextFormat(Qt.TextFormat.RichText)
        self.accel_deps_label.setStyleSheet("color:#888;")
        self.accel_deps_label.setToolTip(self.tr("Installed acceleration-related Python packages"))
        self.accel_deps_label.setWordWrap(True)
        self.accel_deps_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # preference combo
        self.cb_accel_pref = QComboBox()
        self._accel_items = [
            (self.tr("Auto (recommended)"), "auto"),
            (self.tr("CUDA (NVIDIA)"), "cuda"),
        ]

        # Linux AMD ROCm option (now supported)
        if platform.system() == "Linux":
            self._accel_items.append((self.tr("ROCm (AMD on Linux)"), "rocm"))

        # Intel XPU is Windows/Linux
        if platform.system() in ("Windows", "Linux"):
            self._accel_items.append((self.tr("Intel XPU (Arc/Xe)"), "xpu"))

        # DirectML is Windows-only fallback
        if platform.system() == "Windows":
            self._accel_items.append((self.tr("DirectML (Windows AMD/Intel)"), "directml"))

        self._accel_items.append((self.tr("CPU only"), "cpu"))


        self.cb_accel_pref.clear()
        for label, _key in self._accel_items:
            self.cb_accel_pref.addItem(label)
        self.cb_accel_pref.currentIndexChanged.connect(self._accel_pref_changed)

        self.install_accel_btn = QPushButton(self.tr("Install/Update GPU Acceleration…"))

        gpu_help_btn = QToolButton()
        gpu_help_btn.setText("?")
        gpu_help_btn.setToolTip(self.tr("If GPU still not being used — click for fix steps"))
        gpu_help_btn.clicked.connect(self._show_gpu_accel_fix_help)

        accel_box = QVBoxLayout()
        accel_box.setContentsMargins(0, 0, 0, 0)
        accel_box.setSpacing(6)

        accel_row_top = QHBoxLayout()
        accel_row_top.setContentsMargins(0, 0, 0, 0)
        accel_row_top.addWidget(self.backend_label)
        accel_row_top.addStretch(1)
        accel_row_top.addWidget(gpu_help_btn)

        accel_row_pref = QHBoxLayout()
        accel_row_pref.setContentsMargins(0, 0, 0, 0)
        accel_row_pref.addWidget(QLabel(self.tr("Preference:")))
        accel_row_pref.addWidget(self.cb_accel_pref, 1)
        accel_row_pref.addWidget(self.install_accel_btn)

        accel_box.addLayout(accel_row_top)
        accel_box.addWidget(self.accel_deps_label)
        accel_box.addLayout(accel_row_pref)

        w_accel = QWidget()
        w_accel.setLayout(accel_box)
        left_col.addRow(w_accel)

        # ---- Right column: AI Models ----
        right_col.addRow(QLabel(self.tr("<b>AI Models</b>")))

        self.lbl_models_status = QLabel(self.tr("Status: (unknown)"))
        self.lbl_models_status.setStyleSheet("color:#888;")
        self.lbl_models_status.setWordWrap(True)
        self.lbl_models_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right_col.addRow(self.lbl_models_status)

        self.btn_models_update = QPushButton(self.tr("Download/Update Models…"))
        self.btn_models_update.clicked.connect(self._models_update_clicked)

        self.btn_models_install_zip = QPushButton(self.tr("Install from ZIP…"))
        self.btn_models_install_zip.setToolTip(self.tr("Use a manually downloaded models .zip file"))
        self.btn_models_install_zip.clicked.connect(self._models_install_from_zip_clicked)

        self.btn_models_open_drive = QPushButton(self.tr("Open Drive…"))
        self.btn_models_open_drive.setToolTip(self.tr("Download models (Primary/Backup/GitHub mirror)"))
        self.btn_models_open_drive.clicked.connect(self._models_open_drive_clicked)

        # Break wide model-buttons row into 2 rows
        models_box = QVBoxLayout()
        models_box.setContentsMargins(0, 0, 0, 0)
        models_box.setSpacing(6)

        # Top row: one big primary action
        row_models_top = QHBoxLayout()
        row_models_top.setContentsMargins(0, 0, 0, 0)
        row_models_top.addWidget(self.btn_models_update, 1)

        # Bottom row: secondary actions side-by-side
        row_models_bottom = QHBoxLayout()
        row_models_bottom.setContentsMargins(0, 0, 0, 0)
        row_models_bottom.addWidget(self.btn_models_open_drive)
        row_models_bottom.addWidget(self.btn_models_install_zip)
        row_models_bottom.addStretch(1)

        models_box.addLayout(row_models_top)
        models_box.addLayout(row_models_bottom)

        w_models = QWidget()
        w_models.setLayout(models_box)
        right_col.addRow(w_models)

        # ---- WIMS + RA/Dec + Updates ----
        right_col.addRow(QLabel(self.tr("<b>What's In My Sky — Defaults</b>")))
        right_col.addRow(self.tr("Latitude (°):"), self.sp_lat)
        right_col.addRow(self.tr("Longitude (°):"), self.sp_lon)
        right_col.addRow(self.tr("Date (YYYY-MM-DD):"), self.le_date)
        right_col.addRow(self.tr("Time (HH:MM):"), self.le_time)
        right_col.addRow(self.tr("Time Zone:"), self.cb_tz)
        right_col.addRow(self.tr("Min Altitude (°):"), self.sp_min_alt)
        right_col.addRow(self.tr("Object Limit:"), self.sp_obj_limit)

        # ---- RA/Dec Overlay ----
        right_col.addRow(QLabel(self.tr("<b>RA/Dec Overlay</b>")))
        self.chk_wcs_enabled = QCheckBox(self.tr("Show RA/Dec grid"))
        right_col.addRow(self.chk_wcs_enabled)

        self.cb_wcs_mode = QComboBox(); self.cb_wcs_mode.addItems(["Auto", "Fixed spacing"])
        self.cb_wcs_unit = QComboBox(); self.cb_wcs_unit.addItems(["deg", "arcmin"])

        self.sp_wcs_step = QDoubleSpinBox()
        self.sp_wcs_step.setDecimals(3)
        self.sp_wcs_step.setRange(0.001, 90.0)

        def _sync_suffix():
            self.sp_wcs_step.setSuffix(" °" if self.cb_wcs_unit.currentIndex() == 0 else " arcmin")
        _sync_suffix()

        self.cb_wcs_unit.currentIndexChanged.connect(_sync_suffix)
        self.cb_wcs_mode.currentIndexChanged.connect(lambda i: self.sp_wcs_step.setEnabled(i == 1))

        row_wcs = QHBoxLayout()
        row_wcs.setContentsMargins(0, 0, 0, 0)
        row_wcs.addWidget(QLabel(self.tr("Mode:")))
        row_wcs.addWidget(self.cb_wcs_mode)
        row_wcs.addSpacing(8)
        row_wcs.addWidget(QLabel(self.tr("Step:")))
        row_wcs.addWidget(self.sp_wcs_step, 1)
        row_wcs.addWidget(self.cb_wcs_unit)

        _w = QWidget()
        _w.setLayout(row_wcs)
        right_col.addRow(_w)

        # ---- Updates ----
        right_col.addRow(QLabel(self.tr("<b>Updates</b>")))
        right_col.addRow(self.chk_updates_startup)

        # Split updates URL row into field + buttons rows (avoids giant horizontal line)
        updates_box = QVBoxLayout()
        updates_box.setContentsMargins(0, 0, 0, 0)
        updates_box.setSpacing(6)
        updates_box.addWidget(self.le_updates_url)

        row_updates_btns = QHBoxLayout()
        row_updates_btns.setContentsMargins(0, 0, 0, 0)
        row_updates_btns.addWidget(btn_reset_updates_url)
        row_updates_btns.addWidget(self.btn_check_now)
        row_updates_btns.addStretch(1)
        updates_box.addLayout(row_updates_btns)

        w_updates = QWidget()
        w_updates.setLayout(updates_box)
        right_col.addRow(self.tr("Updates JSON URL:"), w_updates)

        # ---- Put columns into responsive container (initially)
        self._cols_layout.addWidget(self._left_col_widget, 1)
        self._cols_layout.addSpacing(16)
        self._cols_layout.addWidget(self._right_col_widget, 1)

        # ---- Buttons (fixed at bottom, not inside scroll) ----
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # Connect accel install button after widget exists
        self.install_accel_btn.clicked.connect(self._install_or_update_accel)

        # Reasonable initial size, clamped to available screen
        self.setSizeGripEnabled(True)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)

        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1200, max(900, int(avail.width() * 0.92)))
            h = min(680,  max(680, int(avail.height() * 0.88)))
            self.resize(w, h)
        else:
            self.resize(1200, 680)

        # Initial load
        self.refresh_ui()

        # Apply responsive layout once after widgets are built
        self._layout_mode = None
        self._update_responsive_layout()


    def showEvent(self, event):
        super().showEvent(event)
        self._update_responsive_layout()


    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()


    def _update_responsive_layout(self):
        """
        Reflow columns based on current dialog width:
        - wide: left + right columns side-by-side
        - narrow: stack right column under left column
        """
        # Tune threshold as needed; ~1180 works well for your control density
        narrow = self.width() < 1180
        mode = "stacked" if narrow else "two_col"

        if getattr(self, "_layout_mode", None) == mode:
            return
        self._layout_mode = mode

        # Clear current contents of _cols_layout without destroying widgets
        while self._cols_layout.count():
            item = self._cols_layout.takeAt(0)
            # intentionally no deleteLater; widgets/layouts are reused

        if narrow:
            stack = QVBoxLayout()
            stack.setContentsMargins(0, 0, 0, 0)
            stack.setSpacing(12)
            stack.addWidget(self._left_col_widget)
            stack.addWidget(self._right_col_widget)
            stack.addStretch(1)
            self._cols_layout.addLayout(stack, 1)
        else:
            self._cols_layout.addWidget(self._left_col_widget, 1)
            self._cols_layout.addSpacing(16)
            self._cols_layout.addWidget(self._right_col_widget, 1)

    def _models_open_drive_clicked(self):
        PRIMARY_FOLDER = "https://drive.google.com/drive/folders/1-fktZb3I9l-mQimJX2fZAmJCBj_t0yAF?usp=drive_link"
        BACKUP_FOLDER  = "https://drive.google.com/drive/folders/1j46RV6touQtOmtxkhdFWGm_LQKwEpTl9?usp=drive_link"
        GITHUB_ZIP     = "https://github.com/setiastro/setiastrosuitepro/releases/download/benchmarkFIT/SASPro_Models_AI4.zip"

        menu = QMenu(self)
        act_primary = menu.addAction(self.tr("Primary (Google Drive)"))
        act_backup  = menu.addAction(self.tr("Backup (Google Drive)"))
        menu.addSeparator()
        act_gh      = menu.addAction(self.tr("GitHub (no quota limit)"))

        chosen = menu.exec(self.btn_models_open_drive.mapToGlobal(self.btn_models_open_drive.rect().bottomLeft()))
        if chosen == act_primary:
            webbrowser.open(PRIMARY_FOLDER)
        elif chosen == act_backup:
            webbrowser.open(BACKUP_FOLDER)
        elif chosen == act_gh:
            webbrowser.open(GITHUB_ZIP)

    def start_models_update(self):
        self._models_update_clicked()

    def _models_install_from_zip_clicked(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt, QThread
        import os

        zip_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select models ZIP"),
            "",
            self.tr("ZIP files (*.zip);;All files (*)")
        )
        if not zip_path:
            return

        if not os.path.exists(zip_path):
            QMessageBox.warning(self, self.tr("Models"), self.tr("File not found."))
            return

        self.btn_models_update.setEnabled(False)
        self.btn_models_install_zip.setEnabled(False)

        pd = QProgressDialog(self.tr("Preparing…"), self.tr("Cancel"), 0, 0, self)
        pd.setWindowTitle(self.tr("Installing Models"))
        pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        pd.setAutoClose(True)
        pd.setMinimumDuration(0)
        pd.show()

        from setiastro.saspro.model_workers import ModelsInstallZipWorker

        self._models_thread = QThread(self)
        self._models_worker = ModelsInstallZipWorker(zip_path)
        self._models_worker.moveToThread(self._models_thread)

        self._models_thread.started.connect(self._models_worker.run, Qt.ConnectionType.QueuedConnection)
        self._models_worker.progress.connect(pd.setLabelText, Qt.ConnectionType.QueuedConnection)

        def _cancel():
            if self._models_thread.isRunning():
                self._models_thread.requestInterruption()
        pd.canceled.connect(_cancel, Qt.ConnectionType.QueuedConnection)

        def _done(ok: bool, msg: str):
            pd.reset()
            pd.deleteLater()

            self._models_thread.quit()
            self._models_thread.wait()

            self.btn_models_update.setEnabled(True)
            self.btn_models_install_zip.setEnabled(True)
            self._refresh_models_status()

            if ok:
                QMessageBox.information(self, self.tr("Models"), self.tr("✅ {0}").format(msg))
            else:
                QMessageBox.warning(self, self.tr("Models"), self.tr("❌ {0}").format(msg))

        self._models_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)
        self._models_thread.finished.connect(self._models_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
        self._models_thread.finished.connect(self._models_thread.deleteLater, Qt.ConnectionType.QueuedConnection)

        self._models_thread.start()

    def _models_update_clicked(self):
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt, QThread

        # NOTE: these must be FILE links or file IDs, not folder links.
        PRIMARY  = "https://drive.google.com/file/d/1d0wQr8Oau9UH3IalMW5anC0_oddxBjh3/view?usp=drive_link"
        BACKUP   = "https://drive.google.com/file/d/1XgqKNd8iBgV3LW8CfzGyS4jigxsxIf86/view?usp=drive_link"
        TERTIARY = "https://github.com/setiastro/setiastrosuitepro/releases/download/benchmarkFIT/SASPro_Models_AI4.zip"

        self.btn_models_update.setEnabled(False)
        self.btn_models_install_zip.setEnabled(False)  # optional but nice consistency

        pd = QProgressDialog(self.tr("Preparing…"), self.tr("Cancel"), 0, 0, self)
        pd.setWindowTitle(self.tr("Updating Models"))
        pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        pd.setAutoClose(True)
        pd.setMinimumDuration(0)
        pd.show()

        from setiastro.saspro.model_workers import ModelsDownloadWorker

        self._models_thread = QThread(self)

        # Define callbacks BEFORE passing them into the worker
        def should_cancel():
            return self._models_thread.isInterruptionRequested()

        def _cancel():
            if self._models_thread.isRunning():
                self._models_thread.requestInterruption()

        self._models_worker = ModelsDownloadWorker(
            PRIMARY, BACKUP, TERTIARY,
            expected_sha256=None,
            should_cancel=should_cancel
        )
        self._models_worker.moveToThread(self._models_thread)

        self._models_thread.started.connect(self._models_worker.run, Qt.ConnectionType.QueuedConnection)
        self._models_worker.progress.connect(pd.setLabelText, Qt.ConnectionType.QueuedConnection)
        pd.canceled.connect(_cancel, Qt.ConnectionType.QueuedConnection)

        def _done(ok: bool, msg: str):
            pd.reset()
            pd.deleteLater()

            self._models_thread.quit()
            self._models_thread.wait()

            self.btn_models_update.setEnabled(True)
            self.btn_models_install_zip.setEnabled(True)
            self._refresh_models_status()

            if ok:
                QMessageBox.information(self, self.tr("Models"), self.tr("✅ {0}").format(msg))
            else:
                QMessageBox.warning(self, self.tr("Models"), self.tr("❌ {0}").format(msg))

        self._models_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)
        self._models_thread.finished.connect(self._models_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
        self._models_thread.finished.connect(self._models_thread.deleteLater, Qt.ConnectionType.QueuedConnection)

        self._models_thread.start()

    def _refresh_models_status(self):
        from setiastro.saspro.model_manager import read_installed_manifest, models_root
        m = read_installed_manifest()
        if not m:
            self.lbl_models_status.setText(self.tr("Status: not installed"))
            return

        # New fields (preferred)
        src = (m.get("source") or "").strip()
        ref = (m.get("source_ref") or "").strip()
        sha = (m.get("sha256") or "").strip()

        # Back-compat with older manifests
        if not src:
            # old versions used google_drive + file_id
            src = "google_drive" if m.get("file_id") else (m.get("source") or "unknown")
        if not ref:
            ref = (m.get("file_id") or m.get("file") or "").strip()

        # Keep it short + readable
        src_label = {
            "google_drive": "Google Drive",
            "http": "GitHub/HTTP",
            "manual_zip": "Manual ZIP",
        }.get(src, src or "Unknown")

        lines = [self.tr("Status: installed"),
                self.tr("Location: {0}").format(models_root())]

        if ref:
            # show just a compact hint (id or url or filename)
            lines.append(self.tr("Source: {0}").format(src_label))
            lines.append(self.tr("Ref: {0}").format(ref))

        if sha:
            lines.append(self.tr("SHA256: {0}").format(sha[:12] + "…"))

        self.lbl_models_status.setText("\n".join(lines))
        self.lbl_models_status.setStyleSheet("color:#888;")

    def _accel_pref_changed(self, idx: int):
        key = "auto"
        if 0 <= idx < len(getattr(self, "_accel_items", [])):
            key = (self._accel_items[idx][1] or "auto").lower()
        self.settings.setValue("accel/preferred_backend", key)
        self.settings.sync()


    def _show_gpu_accel_fix_help(self):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, self.tr("GPU Acceleration Help"),
            self.tr(
                "If GPU is not being used:\n"
                " • Click Install/Update GPU Acceleration…\n"
                " • Restart SAS Pro\n"
                " • On NVIDIA systems, verify drivers and that 'nvidia-smi' works.\n"
                " • On Linux AMD systems, select ROCm and ensure ROCm-compatible drivers/runtime are installed.\n"
                " • On Intel Arc/Xe systems, select Intel XPU.\n"
                " • On Windows non-NVIDIA, DirectML may be used.\n"
            )
        )

    def _pkg_status(self, dist_name: str, import_name: str | None = None) -> tuple[bool, str]:
        """
        Returns (installed?, display_text). Does NOT import the module.
        dist_name  = pip distribution name used for version lookup (e.g., 'torchvision')
        import_name = python import name used for find_spec (e.g., 'torch_directml')
        """
        mod = import_name or dist_name.replace("-", "_")
        present = importlib.util.find_spec(mod) is not None

        ver = ""
        if present:
            try:
                ver = importlib.metadata.version(dist_name)
            except importlib.metadata.PackageNotFoundError:
                # Sometimes dist name differs; fall back to "installed" without version
                ver = ""
            except Exception:
                ver = ""

        if present:
            return True, (f"✅ {ver}" if ver else "✅ installed")
        return False, "— not installed"


    def _format_accel_deps_text(self) -> str:
        try:
            from setiastro.saspro.runtime_torch import add_runtime_to_sys_path
            add_runtime_to_sys_path(status_cb=lambda *_: None)
        except Exception:
            pass

        torch_ok, torch_txt = self._pkg_status("torch", "torch")
        dml_ok, dml_txt     = self._pkg_status("torch-directml", "torch_directml")
        ta_ok, ta_txt       = self._pkg_status("torchaudio", "torchaudio")
        tv_ok, tv_txt       = self._pkg_status("torchvision", "torchvision")

        return (
            f"Torch: <b>{torch_txt}</b><br>"
            f"Torch-DirectML: <b>{dml_txt}</b><br>"
            f"TorchAudio: <b>{ta_txt}</b><br>"
            f"TorchVision: <b>{tv_txt}</b>"
        )


    def _run_capture(self, cmd: list[str]) -> tuple[int, str]:
        """Run a command and return (returncode, combined_output)."""
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return r.returncode, (r.stdout or "")
        except Exception as e:
            return 999, str(e)

    def _probe_python_version(self, cmd: list[str]) -> tuple[bool, tuple[int, int] | None, str]:
        """
        Try to execute cmd and read sys.version_info.major/minor.
        Returns (ok, (maj,min) or None, display_string).
        """
        code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        rc, out = self._run_capture(cmd + ["-c", code])
        if rc != 0:
            return False, None, (out.strip() or "failed")

        s = (out.strip().splitlines()[-1] if out.strip() else "")
        try:
            maj_s, min_s = s.split(".", 1)
            ver = (int(maj_s), int(min_s))
            return True, ver, s
        except Exception:
            return False, None, s or "unknown"

    def _find_python312_cmd(self) -> tuple[list[str] | None, str]:
        """
        Find a runnable Python 3.12 command on this system.
        Returns (cmd or None, info_text).
        """
        sysname = platform.system()

        # Windows: prefer py launcher
        if sysname == "Windows":
            candidates = [
                ["py", "-3.12"],
                ["python3.12"],
                ["python", "-3.12"],  # uncommon but harmless to try
            ]
            for cmd in candidates:
                ok, ver, disp = self._probe_python_version(cmd)
                if ok and ver == (3, 12):
                    return cmd, f"{' '.join(cmd)} -> {disp}"
            return None, "No runnable Python 3.12 found via 'py -3.12' or python3.12."

        # macOS: common Homebrew locations + PATH
        if sysname == "Darwin":
            candidates = [
                ["/opt/homebrew/bin/python3.12"],
                ["/usr/local/bin/python3.12"],
                ["/usr/bin/python3.12"],
                ["python3.12"],
            ]
            for cmd in candidates:
                exe = cmd[0]
                if exe.startswith("/") and not os.path.exists(exe):
                    continue
                ok, ver, disp = self._probe_python_version(cmd)
                if ok and ver == (3, 12):
                    return cmd, f"{' '.join(cmd)} -> {disp}"
            return None, "No runnable Python 3.12 found (Homebrew/PATH)."

        # Linux: PATH
        candidates = [
            ["python3.12"],
        ]
        for cmd in candidates:
            ok, ver, disp = self._probe_python_version(cmd)
            if ok and ver == (3, 12):
                return cmd, f"{' '.join(cmd)} -> {disp}"

        return None, "No runnable python3.12 found in PATH."

    def _gate_python312_for_accel_install(self) -> bool:
        """
        HARD STOP unless a runnable Python 3.12 is available on the system.
        (We use this because runtime GPU wheels are only supported/validated for 3.12.)
        """
        from PyQt6.QtWidgets import QMessageBox

        cmd, info = self._find_python312_cmd()
        if cmd is not None:
            return True  # Python 3.12 is available

        v = sys.version_info
        running = f"{v.major}.{v.minor}"

        # Stronger wording for 3.13/3.14 (your “full stop” policy)
        is_future = (v.major, v.minor) >= (3, 13)

        if is_future:
            title = self.tr("Unsupported Python Version")
            headline = self.tr("GPU acceleration cannot be installed with Python {0}.").format(running)
            details = self.tr(
                "SAS Pro hardware acceleration requires Python 3.12.\n\n"
                "Please install Python 3.12 and re-launch SAS Pro.\n\n"
                "Probe details: {0}"
            ).format(info)
        else:
            title = self.tr("Python 3.12 Required")
            headline = self.tr("Python 3.12 was not found on this system.")
            details = self.tr(
                "GPU acceleration setup requires a runnable Python 3.12.\n\n"
                "You are currently running Python {0}.\n\n"
                "Install Python 3.12 and re-launch SAS Pro.\n\n"
                "Probe details: {1}"
            ).format(running, info)

        QMessageBox.warning(self, title, headline + "\n\n" + details)
        return False

    def _install_or_update_accel(self):
        # Single hard-stop gate
        if not self._gate_python312_for_accel_install():
            self.backend_label.setText(self.tr("Backend: CPU (Python 3.12 required)"))
            return

        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt, QThread
        from setiastro.saspro.accel_installer import current_backend
        from setiastro.saspro.accel_workers import AccelInstallWorker

        self.install_accel_btn.setEnabled(False)
        self.backend_label.setText(self.tr("Backend: installing…"))

        # Read preference using the dynamic accel list
        pref_key = "auto"
        try:
            idx = int(self.cb_accel_pref.currentIndex())
            if 0 <= idx < len(self._accel_items):
                pref_key = (self._accel_items[idx][1] or "auto").lower()
        except Exception:
            pref_key = (self.settings.value("accel/preferred_backend", "auto", type=str) or "auto").lower()

        self._accel_pd = QProgressDialog(self.tr("Preparing runtime…"), self.tr("Cancel"), 0, 0, self)
        self._accel_pd.setWindowTitle(self.tr("Installing GPU Acceleration"))
        self._accel_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._accel_pd.setAutoClose(True)
        self._accel_pd.setMinimumDuration(0)
        self._accel_pd.show()

        self._accel_thread = QThread(self)
        self._accel_worker = AccelInstallWorker(prefer_gpu=True, preferred_backend=pref_key)
        self._accel_worker.moveToThread(self._accel_thread)

        self._accel_thread.started.connect(self._accel_worker.run, Qt.ConnectionType.QueuedConnection)
        self._accel_worker.progress.connect(self._accel_pd.setLabelText, Qt.ConnectionType.QueuedConnection)

        def _cancel():
            if self._accel_thread.isRunning():
                self._accel_thread.requestInterruption()
        self._accel_pd.canceled.connect(_cancel, Qt.ConnectionType.QueuedConnection)

        def _done(ok: bool, msg: str):
            if getattr(self, "_accel_pd", None):
                self._accel_pd.reset()
                self._accel_pd.deleteLater()
                self._accel_pd = None

            self._accel_thread.quit()
            self._accel_thread.wait()

            self.install_accel_btn.setEnabled(True)
            self.backend_label.setText(self.tr("Backend: {0}").format(current_backend()))
            try:
                self.accel_deps_label.setText(self._format_accel_deps_text())
            except Exception:
                pass

            if ok:
                QMessageBox.information(self, self.tr("Acceleration"), self.tr("✅ {0}").format(msg))
            else:
                QMessageBox.warning(self, self.tr("Acceleration"), self.tr("❌ {0}").format(msg))

        self._accel_worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)
        self._accel_thread.finished.connect(self._accel_worker.deleteLater, Qt.ConnectionType.QueuedConnection)
        self._accel_thread.finished.connect(self._accel_thread.deleteLater, Qt.ConnectionType.QueuedConnection)

        self._accel_thread.start()


    def refresh_ui(self):
        """
        Reloads all settings from self.settings and updates the UI widgets.
        Call this before showing the cached dialog to ensure it matches current state.
        """
        # Updates
        self.chk_updates_startup.setChecked(
            self.settings.value("updates/check_on_startup", True, type=bool)
        )
        self.le_updates_url.setText(
            self.settings.value(
                "updates/url",
                "https://raw.githubusercontent.com/setiastro/setiastrosuitepro/main/updates.json",
                type=str
            )
        )
        
        # Shortcuts
        self.chk_save_shortcuts.setChecked(
            self.settings.value("shortcuts/save_on_exit", True, type=bool)
        )
        
        # Theme
        theme_val = (self.settings.value("ui/theme", "system", type=str) or "system").lower()
        index_map = {"dark": 0, "gray": 1, "light": 2, "system": 3, "custom": 4}
        self.cb_theme.setCurrentIndex(index_map.get(theme_val, 2))
        self.btn_theme_custom.setEnabled(theme_val == "custom")
        
        # Language
        current_lang = get_saved_language()
        try:
            lang_idx = self._lang_codes.index(current_lang)
        except ValueError:
            lang_idx = 0
            # fallback to whatever is first or default
            if "en" in self._lang_codes:
                lang_idx = self._lang_codes.index("en")
        
        self.cb_language.blockSignals(True)
        self.cb_language.setCurrentIndex(lang_idx)
        self.cb_language.blockSignals(False)
        self._initial_language = current_lang  # Track for restart notification
        
        # Path fields
        self.le_graxpert.setText(self.settings.value("paths/graxpert", "", type=str))

        self.le_starnet.setText(self.settings.value("paths/starnet", "", type=str))
        self.le_astap.setText(self.settings.value("paths/astap", "", type=str))
        self.le_astrometry.setText(self.settings.value("api/astrometry_key", "", type=str))

        # WIMS
        self.sp_lat.setValue(self.settings.value("latitude", 0.0, type=float))
        self.sp_lon.setValue(self.settings.value("longitude", 0.0, type=float))
        self.le_date.setText(self.settings.value("date", "", type=str) or "")
        self.le_time.setText(self.settings.value("time", "", type=str) or "")
        tz_val = self.settings.value("timezone", "UTC", type=str) or "UTC"
        idx = max(0, self.cb_tz.findText(tz_val))
        self.cb_tz.setCurrentIndex(idx)
        self.sp_min_alt.setValue(self.settings.value("min_altitude", 0.0, type=float))
        self.sp_obj_limit.setValue(self.settings.value("object_limit", 100, type=int))
        
        # Display
        self.chk_autostretch_24bit.setChecked(
            self.settings.value("display/autostretch_24bit", True, type=bool)
        )
        self.chk_smooth_zoom_settle.setChecked(
            self.settings.value("display/smooth_zoom_settle", True, type=bool)
        )
        current_opacity = self.settings.value("display/bg_opacity", 50, type=int)
        self.slider_bg_opacity.blockSignals(True)
        self.slider_bg_opacity.setValue(current_opacity)
        self.slider_bg_opacity.blockSignals(False)
        self.lbl_bg_opacity_val.setText(f"{current_opacity}%")
        self._initial_bg_opacity = int(current_opacity) # For cancel/revert
        
        # Custom background
        self._initial_bg_path = self.settings.value("ui/custom_background", "", type=str) or ""
        self.le_bg_path.setText(self._initial_bg_path)
        
        # RA/Dec Overlay
        self.chk_wcs_enabled.setChecked(self.settings.value("wcs_grid/enabled", True, type=bool))
        
        self.cb_wcs_mode.blockSignals(True)
        self.cb_wcs_mode.setCurrentIndex(
            0 if (self.settings.value("wcs_grid/mode", "auto", type=str) == "auto") else 1
        )
        self.cb_wcs_mode.blockSignals(False)
        
        self.cb_wcs_unit.blockSignals(True)
        self.cb_wcs_unit.setCurrentIndex(
            0 if (self.settings.value("wcs_grid/step_unit", "deg", type=str) == "deg") else 1
        )
        self.cb_wcs_unit.blockSignals(False)
        
        self.sp_wcs_step.setValue(self.settings.value("wcs_grid/step_value", 1.0, type=float))
        self.sp_wcs_step.setEnabled(self.cb_wcs_mode.currentIndex() == 1)
        self.sp_wcs_step.setSuffix(" °" if self.cb_wcs_unit.currentIndex() == 0 else " arcmin")

        pref = (self.settings.value("accel/preferred_backend", "auto", type=str) or "auto").lower()
        idx = next((i for i, (_lbl, key) in enumerate(self._accel_items) if key == pref), 0)
        self.cb_accel_pref.setCurrentIndex(idx)

        from setiastro.saspro.accel_installer import current_backend
        self.backend_label.setText(self.tr("Backend: {0}").format(current_backend()))
        try:
            self.accel_deps_label.setText(self._format_accel_deps_text())
        except Exception:
            self.accel_deps_label.setText(self.tr("Torch: — unknown"))
        try:
            self._refresh_models_status()
        except Exception:
            pass

    def reject(self):
        """User cancelled: restore the original background opacity (revert live changes)."""
        try:
            # Restore saved original value
            self.settings.setValue("display/bg_opacity", int(self._initial_bg_opacity))
            self.settings.sync()
            # Ask parent to redraw with restored value
            parent = self.parent()
            if parent:
                # restore original custom background (may be empty)
                try:
                    # If there was an initial custom background, restore it; otherwise clear.
                    if self._initial_bg_path:
                        if hasattr(parent, "_apply_custom_background"):
                            parent._apply_custom_background(self._initial_bg_path)
                    else:
                        # Avoid calling _apply_custom_background("") which shows a warning
                        if hasattr(parent, "_clear_custom_background"):
                            parent._clear_custom_background()
                        elif hasattr(parent, "_apply_custom_background"):
                            parent._apply_custom_background("")
                except Exception:
                    pass
                # update MDI viewport/redraw
                try:
                    if hasattr(parent, "mdi") and hasattr(parent.mdi, "viewport"):
                        parent.mdi.viewport().update()
                except Exception:
                    pass
        except Exception:
            pass
        super().reject()


    # ----------------- helpers -----------------
    def _browse_into(self, lineedit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select Executable", "", "Executables (*)")
        if path:
            lineedit.setText(path)

    def _browse_dir(self, lineedit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if path:
            lineedit.setText(path)

    def _check_updates_now_clicked(self):
        """Persist update settings, then ask the main window to run an interactive check (if available)."""
        self.settings.setValue("updates/check_on_startup", self.chk_updates_startup.isChecked())
        self.settings.setValue("updates/url", self.le_updates_url.text().strip())
        self.settings.sync()

        parent = self.parent()
        if parent and hasattr(parent, "_check_for_updates_async"):
            try:
                parent._check_for_updates_async(interactive=True)
            except Exception:
                pass

    def _choose_background_clicked(self):
        """Open a file picker and apply a custom background image for the app."""
        path, _ = QFileDialog.getOpenFileName(self, "Select background image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            # Do NOT persist yet — just update UI and preview via parent.
            self.le_bg_path.setText(path)
            parent = self.parent()
            if parent and hasattr(parent, "_apply_custom_background"):
                try:
                    parent._apply_custom_background(path)
                except Exception:
                    pass
        except Exception:
            pass

    def _clear_background_clicked(self):
        """Clear persisted custom background and ask main window to restore defaults."""
        try:
            # Do NOT modify settings yet — clear preview and let Save apply
            self.le_bg_path.setText("")
            parent = self.parent()
            if parent:
                # request parent to clear preview/background for now
                try:
                    if hasattr(parent, "_clear_custom_background"):
                        parent._clear_custom_background()
                    elif hasattr(parent, "_apply_custom_background"):
                        parent._apply_custom_background("")
                except Exception:
                    pass
        except Exception:
            pass

    def _on_theme_changed(self, idx: int):
        # Enable the "Customize…" button only when Custom is selected
        text = self.cb_theme.currentText().lower()
        self.btn_theme_custom.setEnabled(text == "custom")

    def _open_theme_editor(self):
        from PyQt6.QtWidgets import QDialog
        dlg = ThemeEditorDialog(self, self.settings)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # If user saved a custom theme, make sure "Custom" is selected
            self.cb_theme.setCurrentIndex(4)  # Custom


    def _save_and_accept(self):
        # Paths / Integrations
        self.settings.setValue("paths/graxpert", self.le_graxpert.text().strip())

        self.settings.setValue("paths/starnet", self.le_starnet.text().strip())
        self.settings.setValue("paths/astap", self.le_astap.text().strip())
        self.settings.setValue("shortcuts/save_on_exit", self.chk_save_shortcuts.isChecked())
        self.settings.setValue("api/astrometry_key", self.le_astrometry.text().strip())

        # WIMS defaults
        self.settings.setValue("latitude", float(self.sp_lat.value()))
        self.settings.setValue("longitude", float(self.sp_lon.value()))
        self.settings.setValue("date", self.le_date.text().strip())
        self.settings.setValue("time", self.le_time.text().strip())
        self.settings.setValue("timezone", self.cb_tz.currentText())
        self.settings.setValue("min_altitude", float(self.sp_min_alt.value()))
        self.settings.setValue("object_limit", int(self.sp_obj_limit.value()))

        # RA/Dec Overlay
        self.settings.setValue("wcs_grid/enabled", self.chk_wcs_enabled.isChecked())
        self.settings.setValue("wcs_grid/mode", "auto" if self.cb_wcs_mode.currentIndex() == 0 else "fixed")
        self.settings.setValue("wcs_grid/step_unit", "deg" if self.cb_wcs_unit.currentIndex() == 0 else "arcmin")
        self.settings.setValue("wcs_grid/step_value", float(self.sp_wcs_step.value()))


        # Updates + Display
        self.settings.setValue("updates/check_on_startup", self.chk_updates_startup.isChecked())
        self.settings.setValue("updates/url", self.le_updates_url.text().strip())
        self.settings.setValue("display/autostretch_24bit", self.chk_autostretch_24bit.isChecked())
        self.settings.setValue("display/smooth_zoom_settle", self.chk_smooth_zoom_settle.isChecked())

        # accel preference (dynamic)
        pref_idx = int(self.cb_accel_pref.currentIndex())
        pref_key = "auto"
        if 0 <= pref_idx < len(getattr(self, "_accel_items", [])):
            pref_key = (self._accel_items[pref_idx][1] or "auto").lower()
        self.settings.setValue("accel/preferred_backend", pref_key)

        # Custom background: persist the chosen path (empty -> remove)
        bg_path = (self.le_bg_path.text() or "").strip()
        if bg_path:
            self.settings.setValue("ui/custom_background", bg_path)
        else:
            try:
                self.settings.remove("ui/custom_background")
            except Exception:
                self.settings.setValue("ui/custom_background", "")

        # bg_opacity is already saved in real-time by _on_opacity_changed()

        # Theme
        idx = max(0, self.cb_theme.currentIndex())
        if idx == 0:
            theme_val = "dark"
        elif idx == 1:
            theme_val = "gray"
        elif idx == 2:
            theme_val = "light"
        elif idx == 3:
            theme_val = "system"
        else:
            theme_val = "custom"
        self.settings.setValue("ui/theme", theme_val)

        # Language
        lang_idx = self.cb_language.currentIndex()
        new_lang = self._lang_codes[lang_idx] if 0 <= lang_idx < len(self._lang_codes) else "en"
        save_language(new_lang)
        
        # Apply language change immediately if changed
        if new_lang != self._initial_language:
            from PyQt6.QtWidgets import QMessageBox
            
            QMessageBox.information(
                self,
                self.tr("Restart required"),
                self.tr("Language changed. Please manually restart the application to apply the new language.")
            )

        self.settings.sync()

        # Apply now if the parent knows how
        p = self.parent()
        if p and hasattr(p, "apply_theme_from_settings"):
            try:
                p.apply_theme_from_settings()
            except Exception:
                pass
        
        if hasattr(p, "mdi") and hasattr(p.mdi, "viewport"):
                p.mdi.viewport().update()
        if p and hasattr(p, "apply_display_settings_to_open_views"):
            try:
                p.apply_display_settings_to_open_views()
            except Exception:
                pass
        try:
            self.settings.remove("paths/cosmic_clarity")
        except Exception:
            pass

        self.accept()

from PyQt6.QtGui import QColor, QFont


class ThemeEditorDialog(QDialog):
    """
    Simple "Custom Theme" editor: lets the user pick main colors and a UI font.
    Colors are stored in QSettings as hex strings (e.g. '#404040').
    """
    def __init__(self, parent, settings: QSettings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Custom Theme")
        self.colors: dict[str, QColor] = {}
        self.font_str: str = self.settings.value("ui/custom/font", "", type=str) or ""

        form = QFormLayout(self)

        # Helper: add color pickers for key roles
        self._add_color_picker(form, "Window / Panels",   "ui/custom/window",   QColor(40, 40, 40))
        self._add_color_picker(form, "Base (Editors)",    "ui/custom/base",     QColor(24, 24, 24))
        self._add_color_picker(form, "Alternate Base",    "ui/custom/altbase",  QColor(32, 32, 32))
        self._add_color_picker(form, "Text",              "ui/custom/text",     QColor(230, 230, 230))
        self._add_color_picker(form, "Buttons",           "ui/custom/button",   QColor(40, 40, 40))
        self._add_color_picker(form, "Highlight / Accent","ui/custom/highlight",QColor(30, 144, 255))
        self._add_color_picker(form, "Link",              "ui/custom/link",     QColor(120, 170, 255))
        self._add_color_picker(form, "Visited Link",      "ui/custom/link_visited", QColor(180, 150, 255))

        # Font picker
        self.btn_font = QPushButton("Choose…")
        self.btn_font.clicked.connect(self._pick_font)
        form.addRow("UI Font:", self.btn_font)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    # ---------- helpers ----------

    def _add_color_picker(self, form: QFormLayout, label_text: str,
                          key: str, default: QColor):
        # Load from settings or default
        stored = self.settings.value(key, default.name(), type=str)
        color = QColor(stored) if stored else default
        self.colors[key] = color

        btn = QPushButton(color.name())
        btn.setMinimumWidth(90)
        btn.setStyleSheet(f"background-color: {color.name()}; color: #ffffff;")
        btn.clicked.connect(lambda _=False, k=key, b=btn: self._pick_color(k, b))

        form.addRow(label_text + ":", btn)

    def _pick_color(self, key: str, button: QPushButton):
        initial = self.colors.get(key, QColor("#404040"))
        col = QColorDialog.getColor(initial, self, "Select Color")
        if col.isValid():
            self.colors[key] = col
            button.setText(col.name())
            button.setStyleSheet(f"background-color: {col.name()}; color: #ffffff;")

    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import QFontDialog

    def _pick_font(self):
        # Load previous font if we have one
        base_str = self.settings.value("ui/custom_font", "", type=str)
        base_font = QFont()
        if base_str:
            try:
                base_font.fromString(base_str)
            except Exception:
                pass

        # ✅ NOTE: (font, ok) — NOT (ok, font)
        font, ok = QFontDialog.getFont(base_font, self, "Select UI Font")
        if not ok:
            return  # user cancelled

        # Store and update preview
        self.font_str = font.toString()
        self.settings.setValue("ui/custom_font", self.font_str)
        self.settings.sync()

        # If you have a label/button to show the chosen font:
        try:
            self.font_button.setText(f"{font.family()}, {font.pointSize()} pt")
        except Exception:
            pass

        # Re-apply theme so the new font takes effect
        parent = self.parent()
        if parent and hasattr(parent, "apply_theme_from_settings"):
            try:
                parent.apply_theme_from_settings()
            except Exception:
                pass

    def _save_and_accept(self):
        # Persist colors
        for key, col in self.colors.items():
            self.settings.setValue(key, col.name())

        # Persist font if chosen
        if self.font_str:
            self.settings.setValue("ui/custom/font", self.font_str)

        self.settings.sync()
        self.accept()
