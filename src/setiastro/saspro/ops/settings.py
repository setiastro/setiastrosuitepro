# ops.settings.py
from PyQt6.QtWidgets import (
    QLineEdit, QDialogButtonBox, QFileDialog, QDialog, QPushButton, QFormLayout,QApplication,
    QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLabel, QColorDialog, QFontDialog, QSlider)
from PyQt6.QtCore import QSettings, Qt
import pytz  # for timezone list
from setiastro.saspro.accel_installer import current_backend
import sys, platform
from PyQt6.QtWidgets import QToolButton, QProgressDialog
from PyQt6.QtCore import QThread
# i18n support
from setiastro.saspro.i18n import get_available_languages, get_saved_language, save_language
import importlib.util
import importlib.metadata

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

        # Build the updates URL row ONCE (we'll insert it later on the right column)
        row_updates_url = QHBoxLayout()
        row_updates_url.addWidget(self.le_updates_url, 1)
        row_updates_url.addWidget(btn_reset_updates_url)
        row_updates_url.addWidget(self.btn_check_now)

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
        self._lang_codes = list(get_available_languages().keys())  # ["en", "it", "fr", "es"]
        self._lang_names = list(get_available_languages().values())  # ["English", "Italiano", ...]
        self.cb_language.addItems(self._lang_names)
        self._initial_language = "en" # placeholder, set in refresh_ui

        btn_grax  = QPushButton(self.tr("Browse…")); btn_grax.clicked.connect(lambda: self._browse_into(self.le_graxpert))

        btn_star  = QPushButton(self.tr("Browse…")); btn_star.clicked.connect(lambda: self._browse_into(self.le_starnet))
        btn_astap = QPushButton(self.tr("Browse…")); btn_astap.clicked.connect(lambda: self._browse_into(self.le_astap))

        row_grax  = QHBoxLayout(); row_grax.addWidget(self.le_graxpert); row_grax.addWidget(btn_grax)

        row_star  = QHBoxLayout(); row_star.addWidget(self.le_starnet); row_star.addWidget(btn_star)
        row_astap = QHBoxLayout(); row_astap.addWidget(self.le_astap);  row_astap.addWidget(btn_astap)

        self.le_astrometry = QLineEdit()
        self.le_astrometry.setEchoMode(QLineEdit.EchoMode.Password)

        # ---- WIMS defaults ----
        self.sp_lat = QDoubleSpinBox();  self.sp_lat.setRange(-90.0, 90.0);       self.sp_lat.setDecimals(6)
        self.sp_lon = QDoubleSpinBox();  self.sp_lon.setRange(-180.0, 180.0);     self.sp_lon.setDecimals(6)
        self.le_date = QLineEdit()   # YYYY-MM-DD
        self.le_time = QLineEdit()   # HH:MM
        self.cb_tz   = QComboBox();  self.cb_tz.addItems(pytz.all_timezones)
        self.sp_min_alt = QDoubleSpinBox(); self.sp_min_alt.setRange(0.0, 90.0);  self.sp_min_alt.setDecimals(1)
        self.sp_obj_limit = QSpinBox(); self.sp_obj_limit.setRange(1, 1000)

        self.chk_autostretch_24bit = QCheckBox(self.tr("High-quality autostretch (24-bit; better gradients)"))
        self.chk_autostretch_24bit.setToolTip(self.tr("Compute autostretch on a 24-bit histogram (smoother gradients)."))

        self.slider_bg_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_bg_opacity.setRange(0, 100)
        self._initial_bg_opacity = 50

        self.lbl_bg_opacity_val = QLabel("50%")
        self.lbl_bg_opacity_val.setFixedWidth(40)

        def _on_opacity_changed(val):
            self.lbl_bg_opacity_val.setText(f"{val}%")
            # Aggiorna in tempo reale il valore nei settings
            self.settings.setValue("display/bg_opacity", val)
            self.settings.sync()
            # Richiedi al parent (main window) di aggiornare il rendering della MDI
            parent = self.parent()
            if parent and hasattr(parent, "mdi") and hasattr(parent.mdi, "viewport"):
                parent.mdi.viewport().update()

        self.slider_bg_opacity.valueChanged.connect(_on_opacity_changed)

        row_bg_opacity = QHBoxLayout()
        row_bg_opacity.addWidget(self.slider_bg_opacity)
        row_bg_opacity.addWidget(self.lbl_bg_opacity_val)
        w_bg_opacity = QWidget()
        w_bg_opacity.setLayout(row_bg_opacity)

        # ---- Custom background: choose/clear preview ----
        self.le_bg_path = QLineEdit()
        self.le_bg_path.setReadOnly(True)
        self._initial_bg_path = ""
        
        btn_choose_bg = QPushButton(self.tr("Choose Background…"))
        btn_choose_bg.setToolTip(self.tr("Pick a PNG or JPG to use as the application background"))
        btn_choose_bg.clicked.connect(self._choose_background_clicked)
        btn_clear_bg = QPushButton(self.tr("Clear"))
        btn_clear_bg.setToolTip(self.tr("Remove custom background and restore default"))
        btn_clear_bg.clicked.connect(self._clear_background_clicked)

        row_bg_image = QHBoxLayout()
        row_bg_image.addWidget(self.le_bg_path, 1)
        row_bg_image.addWidget(btn_choose_bg)
        row_bg_image.addWidget(btn_clear_bg)
        w_bg_image = QWidget()
        w_bg_image.setLayout(row_bg_image)

        # ─────────────────────────────────────────────────────────────────────
        # LAYOUT MUST EXIST BEFORE ANY addRow(...) — build it here
        # ─────────────────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        cols = QHBoxLayout(); root.addLayout(cols)

        left_col  = QFormLayout()
        right_col = QFormLayout()
        for f in (left_col, right_col):
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            f.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        cols.addLayout(left_col, 1)
        cols.addSpacing(16)
        cols.addLayout(right_col, 1)

        # ---- Left column: Paths & Integrations ----
        left_col.addRow(QLabel(self.tr("<b>Paths & Integrations</b>")))
        w = QWidget(); w.setLayout(row_grax);  left_col.addRow(self.tr("GraXpert executable:"), w)

        w = QWidget(); w.setLayout(row_star);  left_col.addRow(self.tr("StarNet executable:"), w)
        w = QWidget(); w.setLayout(row_astap); left_col.addRow(self.tr("ASTAP executable:"), w)
        left_col.addRow(self.tr("Astrometry.net API key:"), self.le_astrometry)
        left_col.addRow(self.chk_save_shortcuts)
        row_theme = QHBoxLayout()
        row_theme.addWidget(self.cb_theme, 1)
        row_theme.addWidget(self.btn_theme_custom)
        w_theme = QWidget()
        w_theme.setLayout(row_theme)
        left_col.addRow(self.tr("Theme:"), w_theme)
        left_col.addRow(self.tr("Language:"), self.cb_language)

        # ---- Display (moved under Theme) ----
        left_col.addRow(QLabel(self.tr("<b>Display</b>")))
        left_col.addRow(self.chk_autostretch_24bit)
        left_col.addRow(self.tr("Background Opacity:"), w_bg_opacity)
        left_col.addRow(self.tr("Background Image:"), w_bg_image)

        left_col.addRow(QLabel(self.tr("<b>Acceleration</b>")))

        accel_row = QHBoxLayout()
        self.backend_label = QLabel(self.tr("Backend: {0}").format(current_backend()))
        accel_row.addWidget(self.backend_label)
        # NEW: dependency status label (rich text)
        self.accel_deps_label = QLabel()
        self.accel_deps_label.setTextFormat(Qt.TextFormat.RichText)
        self.accel_deps_label.setStyleSheet("color:#888;")  # subtle
        self.accel_deps_label.setToolTip(self.tr("Installed acceleration-related Python packages"))
        accel_row.addSpacing(12)
        accel_row.addWidget(self.accel_deps_label)

        # NEW: preference combo
        self.cb_accel_pref = QComboBox()
        self.cb_accel_pref.addItems([
            "Auto (recommended)",
            "CUDA (NVIDIA)",
            "Intel XPU (Arc/Xe)",
            "DirectML (Windows AMD/Intel)",
            "CPU only",
        ])
        # hide DirectML option on non-Windows if you want:
        if platform.system() != "Windows":
            # remove the DirectML entry (index 3)
            self.cb_accel_pref.removeItem(3)

        accel_row.addSpacing(8)
        accel_row.addWidget(QLabel(self.tr("Preference:")))
        accel_row.addWidget(self.cb_accel_pref)

        self.install_accel_btn = QPushButton(self.tr("Install/Update GPU Acceleration…"))
        accel_row.addWidget(self.install_accel_btn)

        gpu_help_btn = QToolButton()
        gpu_help_btn.setText("?")
        gpu_help_btn.setToolTip(self.tr("If GPU still not being used — click for fix steps"))
        gpu_help_btn.clicked.connect(self._show_gpu_accel_fix_help)
        accel_row.addWidget(gpu_help_btn)

        accel_row.addStretch(1)
        w_accel = QWidget()
        w_accel.setLayout(accel_row)
        left_col.addRow(w_accel)

        # ---- Models ----
        right_col.addRow(QLabel(self.tr("<b>AI Models</b>")))

        self.lbl_models_status = QLabel(self.tr("Status: (unknown)"))
        self.lbl_models_status.setStyleSheet("color:#888;")
        right_col.addRow(self.lbl_models_status)

        self.btn_models_update = QPushButton(self.tr("Download/Update Models…"))
        self.btn_models_update.clicked.connect(self._models_update_clicked)
        right_col.addRow(self.btn_models_update)

        # ---- Right column: WIMS + RA/Dec + Updates + Display ----
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
        self.sp_wcs_step.setDecimals(3); self.sp_wcs_step.setRange(0.001, 90.0)
        
        def _sync_suffix():
            self.sp_wcs_step.setSuffix(" °" if self.cb_wcs_unit.currentIndex() == 0 else " arcmin")
        _sync_suffix()
        self.cb_wcs_unit.currentIndexChanged.connect(_sync_suffix)
        self.cb_wcs_mode.currentIndexChanged.connect(lambda i: self.sp_wcs_step.setEnabled(i == 1))

        row_wcs = QHBoxLayout()
        row_wcs.addWidget(QLabel(self.tr("Mode:"))); row_wcs.addWidget(self.cb_wcs_mode)
        row_wcs.addSpacing(8)
        row_wcs.addWidget(QLabel(self.tr("Step:"))); row_wcs.addWidget(self.sp_wcs_step, 1); row_wcs.addWidget(self.cb_wcs_unit)
        _w = QWidget(); _w.setLayout(row_wcs)
        right_col.addRow(_w)

        # ---- Updates ----
        right_col.addRow(QLabel(self.tr("<b>Updates</b>")))
        right_col.addRow(self.chk_updates_startup)
        w = QWidget(); w.setLayout(row_updates_url)
        right_col.addRow(self.tr("Updates JSON URL:"), w)

        # ---- Buttons ----
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self
        )
        self.cb_accel_pref.currentIndexChanged.connect(self._accel_pref_changed)



        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)
        self.install_accel_btn.clicked.connect(self._install_or_update_accel)
        # Initial Load:
        self.refresh_ui()

    def _models_update_clicked(self):
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt, QThread

        # NOTE: these must be FILE links or file IDs, not folder links.
        # Put your actual *zip file* share links here once you create them.
        PRIMARY = "https://drive.google.com/file/d/1n4p0grtNpfllalMqtgaEmsTYaFhT5u7Y/view?usp=drive_link"
        BACKUP  = "https://drive.google.com/file/d/1uRGJCITlfMMN89ZkOO5ICWEKMH24KGit/view?usp=drive_link"


        self.btn_models_update.setEnabled(False)

        pd = QProgressDialog(self.tr("Preparing…"), self.tr("Cancel"), 0, 0, self)
        pd.setWindowTitle(self.tr("Updating Models"))
        pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        pd.setAutoClose(True)
        pd.setMinimumDuration(0)
        pd.show()

        from setiastro.saspro.model_workers import ModelsDownloadWorker

        self._models_thread = QThread(self)
        self._models_worker = ModelsDownloadWorker(PRIMARY, BACKUP, expected_sha256=None)
        self._models_worker.moveToThread(self._models_thread)

        self._models_thread.started.connect(self._models_worker.run, Qt.ConnectionType.QueuedConnection)
        self._models_worker.progress.connect(pd.setLabelText, Qt.ConnectionType.QueuedConnection)
        def should_cancel():
            return self._models_thread.isInterruptionRequested()
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
        fid = m.get("file_id", "")
        self.lbl_models_status.setText(self.tr("Status: installed (Drive id: {0})\nLocation: {1}").format(fid, models_root()))


    def _accel_pref_changed(self, idx: int):
        inv = {0:"auto", 1:"cuda", 2:"xpu", 3:"directml", 4:"cpu"}
        val = inv.get(idx, "auto")
        self.settings.setValue("accel/preferred_backend", val)
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
        # Torch + friends
        torch_ok, torch_txt = self._pkg_status("torch", "torch")
        dml_ok, dml_txt     = self._pkg_status("torch-directml", "torch_directml")
        ta_ok, ta_txt       = self._pkg_status("torchaudio", "torchaudio")
        tv_ok, tv_txt       = self._pkg_status("torchvision", "torchvision")

        # Pretty, compact, and stable in a QLabel
        return (
            f"Torch: <b>{torch_txt}</b><br>"
            f"Torch-DirectML: <b>{dml_txt}</b><br>"
            f"TorchAudio: <b>{ta_txt}</b><br>"
            f"TorchVision: <b>{tv_txt}</b>"
        )


    def _install_or_update_accel(self):
        import sys, platform
        from PyQt6.QtWidgets import QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt, QThread
        from setiastro.saspro.accel_installer import current_backend
        from setiastro.saspro.accel_workers import AccelInstallWorker  # wherever yours lives

        v = sys.version_info
        if not (v.major == 3 and v.minor in (10, 11, 12)):
            why = self.tr("This app is running on Python {0}.{1}. GPU acceleration requires Python 3.10, 3.11, or 3.12.").format(v.major, v.minor)
            tip = ""
            sysname = platform.system()
            if sysname == "Darwin":
                tip = self.tr("\n\nmacOS tip (Apple Silicon):\n • Install Python 3.12:  brew install python@3.12\n • Relaunch the app.")
            elif sysname == "Windows":
                tip = self.tr("\n\nWindows tip:\n • Install Python 3.12/3.11/3.10 (x64) from python.org\n • Relaunch the app.")
            else:
                tip = self.tr("\n\nLinux tip:\n • Install python3.12 or 3.11 via your package manager\n • Relaunch the app.")

            QMessageBox.warning(self, self.tr("Unsupported Python Version"), why + tip)
            self.backend_label.setText(self.tr("Backend: CPU (Python version not supported for GPU install)"))
            return

        self.install_accel_btn.setEnabled(False)
        self.backend_label.setText(self.tr("Backend: installing…"))

        pref = (self.settings.value("accel/preferred_backend", "auto", type=str) or "auto").lower()

        self._accel_pd = QProgressDialog(self.tr("Preparing runtime…"), self.tr("Cancel"), 0, 0, self)
        self._accel_pd.setWindowTitle(self.tr("Installing GPU Acceleration"))
        self._accel_pd.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._accel_pd.setAutoClose(True)
        self._accel_pd.setMinimumDuration(0)
        self._accel_pd.show()

        self._accel_thread = QThread(self)
        self._accel_worker = AccelInstallWorker(prefer_gpu=True, preferred_backend=pref)
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

        # map stored -> combobox index (adjust if you removed DirectML on non-Windows)
        idx_map = {"auto": 0, "cuda": 1, "xpu": 2, "directml": 3, "cpu": 4}

        idx = idx_map.get(pref, 0)
        # if non-Windows and directml was saved earlier, clamp to Auto
        if platform.system() != "Windows" and pref == "directml":
            idx = 0

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

        # accel preference
        pref_idx = self.cb_accel_pref.currentIndex()
        # map index -> stored string (again, adjust if DirectML removed on non-Windows)
        inv = {0:"auto", 1:"cuda", 2:"xpu", 3:"directml", 4:"cpu"}
        self.settings.setValue("accel/preferred_backend", inv.get(pref_idx, "auto"))

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
