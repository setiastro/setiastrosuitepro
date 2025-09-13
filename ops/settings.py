# ops.settings.py
from PyQt6.QtWidgets import (
    QLineEdit, QDialogButtonBox, QFileDialog, QDialog, QPushButton, QFormLayout,
    QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLabel
)
from PyQt6.QtCore import QSettings
import pytz  # for timezone list

class SettingsDialog(QDialog):
    """
    Simple settings UI for external executable paths + WIMS defaults.
    Values are persisted via the provided QSettings instance.
    """
    def __init__(self, parent, settings: QSettings):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.settings = settings

        # ---- Existing fields (paths, checkboxes, etc.) ----
        self.le_graxpert = QLineEdit()
        self.le_cosmic   = QLineEdit()
        self.le_starnet  = QLineEdit()
        self.le_astap    = QLineEdit()

        # ---- Updates (SASpro) ----
        self.chk_updates_startup = QCheckBox("Check for updates on startup")
        self.chk_updates_startup.setChecked(
            self.settings.value("updates/check_on_startup", True, type=bool)
        )

        self.le_updates_url = QLineEdit()
        self.le_updates_url.setPlaceholderText("Raw JSON URL (advanced)")
        self.le_updates_url.setText(
            self.settings.value(
                "updates/url",
                "https://raw.githubusercontent.com/setiastro/setiastrosuitepro/main/updates.json",
                type=str
            )
        )

        btn_reset_updates_url = QPushButton("Reset")
        btn_reset_updates_url.setToolTip("Restore default updates URL")
        btn_reset_updates_url.clicked.connect(
            lambda: self.le_updates_url.setText("https://raw.githubusercontent.com/setiastro/setiastrosuitepro/main/updates.json")
        )

        # Optional: “Check Now…” button (only shown if the parent has the method)
        self.btn_check_now = QPushButton("Check Now…")
        self.btn_check_now.setToolTip("Run an update check immediately")
        self.btn_check_now.setVisible(hasattr(parent, "_check_for_updates_async"))
        self.btn_check_now.clicked.connect(self._check_updates_now_clicked)

        self.chk_save_shortcuts = QCheckBox("Save desktop shortcuts on exit")
        self.chk_save_shortcuts.setChecked(
            self.settings.value("shortcuts/save_on_exit", True, type=bool)
        )

        self.cb_theme = QComboBox()
        self.cb_theme.clear()
        self.cb_theme.addItems(["Dark", "Light"])
        theme_val = (self.settings.value("ui/theme", "system", type=str) or "system").lower()
        self.cb_theme.setCurrentIndex({"dark":0, "light":1, "system":2}.get(theme_val, 2))

        self.le_graxpert.setText(self.settings.value("paths/graxpert", "", type=str))
        self.le_cosmic.setText(self.settings.value("paths/cosmic_clarity", "", type=str))
        self.le_starnet.setText(self.settings.value("paths/starnet", "", type=str))
        self.le_astap.setText(self.settings.value("paths/astap", "", type=str))

        btn_grax  = QPushButton("Browse…"); btn_grax.clicked.connect(lambda: self._browse_into(self.le_graxpert))
        btn_ccl   = QPushButton("Browse…"); btn_ccl.clicked.connect(lambda: self._browse_dir(self.le_cosmic))
        btn_star  = QPushButton("Browse…"); btn_star.clicked.connect(lambda: self._browse_into(self.le_starnet))
        btn_astap = QPushButton("Browse…"); btn_astap.clicked.connect(lambda: self._browse_into(self.le_astap))

        row_grax  = QHBoxLayout(); row_grax.addWidget(self.le_graxpert); row_grax.addWidget(btn_grax)
        row_ccl   = QHBoxLayout(); row_ccl.addWidget(self.le_cosmic);   row_ccl.addWidget(btn_ccl)
        row_star  = QHBoxLayout(); row_star.addWidget(self.le_starnet); row_star.addWidget(btn_star)
        row_astap = QHBoxLayout(); row_astap.addWidget(self.le_astap);  row_astap.addWidget(btn_astap)

        self.le_astrometry = QLineEdit()
        self.le_astrometry.setEchoMode(QLineEdit.EchoMode.Password)
        self.le_astrometry.setText(self.settings.value("api/astrometry_key", "", type=str))

        # ---- NEW: What's In My Sky (WIMS) defaults ----
        # Same keys your WIMS tool already reads/writes: latitude, longitude, date, time, timezone, min_altitude, object_limit
        self.sp_lat = QDoubleSpinBox();  self.sp_lat.setRange(-90.0, 90.0);       self.sp_lat.setDecimals(6)
        self.sp_lon = QDoubleSpinBox();  self.sp_lon.setRange(-180.0, 180.0);     self.sp_lon.setDecimals(6)
        self.le_date = QLineEdit()   # YYYY-MM-DD
        self.le_time = QLineEdit()   # HH:MM
        self.cb_tz   = QComboBox();  self.cb_tz.addItems(pytz.all_timezones)
        self.sp_min_alt = QDoubleSpinBox(); self.sp_min_alt.setRange(0.0, 90.0);  self.sp_min_alt.setDecimals(1)
        self.sp_obj_limit = QSpinBox(); self.sp_obj_limit.setRange(1, 1000)

        # Load existing values (defaults match your WIMS load_settings)
        self.sp_lat.setValue(self.settings.value("latitude", 0.0, type=float))
        self.sp_lon.setValue(self.settings.value("longitude", 0.0, type=float))
        self.le_date.setText(self.settings.value("date", "", type=str) or "")
        self.le_time.setText(self.settings.value("time", "", type=str) or "")
        tz_val = self.settings.value("timezone", "UTC", type=str) or "UTC"
        idx = max(0, self.cb_tz.findText(tz_val))
        self.cb_tz.setCurrentIndex(idx)
        self.sp_min_alt.setValue(self.settings.value("min_altitude", 0.0, type=float))
        self.sp_obj_limit.setValue(self.settings.value("object_limit", 100, type=int))

        self.chk_autostretch_16bit = QCheckBox("High-quality autostretch (16-bit; better gradients)")
        self.chk_autostretch_16bit.setToolTip(
            "When enabled, autostretch computes on a 16-bit histogram for smooth tones (slightly slower)."
        )
        self.chk_autostretch_16bit.setChecked(
            self.settings.value("display/autostretch_16bit", True, type=bool)
        )

        # ---- Layout ----
        form = QFormLayout()

        # Paths / API (existing)
        form.addRow("GraXpert executable:", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(row_grax)
        form.addRow("Cosmic Clarity folder:", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(row_ccl)
        form.addRow("StarNet executable:", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(row_star)
        form.addRow("ASTAP executable:", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(row_astap)
        form.addRow("Astrometry.net API key:", self.le_astrometry)
        form.addRow(self.chk_save_shortcuts)
        form.addRow("Theme:", self.cb_theme)
        # Separator / header for WIMS section
        hdr = QLabel("<b>What's In My Sky — Defaults</b>")
        form.addRow(hdr)

        form.addRow("Latitude (°):", self.sp_lat)
        form.addRow("Longitude (°):", self.sp_lon)
        form.addRow("Date (YYYY-MM-DD):", self.le_date)
        form.addRow("Time (HH:MM):", self.le_time)
        form.addRow("Time Zone:", self.cb_tz)
        form.addRow("Min Altitude (°):", self.sp_min_alt)
        form.addRow("Object Limit:", self.sp_obj_limit)

        # Separator / header for Updates section
        hdr_updates = QLabel("<b>Updates</b>")
        form.addRow(hdr_updates)
        form.addRow(self.chk_updates_startup)

        # URL row with Reset + Check Now
        row_updates_url = QHBoxLayout()
        row_updates_url.addWidget(self.le_updates_url, 1)
        row_updates_url.addWidget(btn_reset_updates_url)
        row_updates_url.addWidget(self.btn_check_now)

        # Use your existing pattern for embedding layouts in QFormLayout
        form.addRow("Updates JSON URL:", QWidget())
        form.itemAt(form.rowCount()-1, QFormLayout.ItemRole.FieldRole).widget().setLayout(row_updates_url)

        hdr_display = QLabel("<b>Display</b>")
        form.addRow(hdr_display)
        form.addRow(self.chk_autostretch_16bit)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self
        )
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(btns)

    def _browse_into(self, lineedit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select Executable", "", "Executables (*)")
        if path:
            lineedit.setText(path)

    def _browse_dir(self, lineedit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder", "")
        if path:
            lineedit.setText(path)

    def _check_updates_now_clicked(self):
        """
        Persist the current update settings, then ask the main window to run
        an interactive check. Safe no-op if method is missing.
        """
        # Save current fields first so the update checker uses them
        self.settings.setValue("updates/check_on_startup", self.chk_updates_startup.isChecked())
        self.settings.setValue("updates/url", self.le_updates_url.text().strip())
        self.settings.sync()

        # Call the main window's async checker if available
        parent = self.parent()
        if parent and hasattr(parent, "_check_for_updates_async"):
            try:
                parent._check_for_updates_async(interactive=True)
            except Exception:
                pass

    def _save_and_accept(self):
        # Existing
        self.settings.setValue("paths/graxpert", self.le_graxpert.text().strip())
        self.settings.setValue("paths/cosmic_clarity", self.le_cosmic.text().strip())
        self.settings.setValue("paths/starnet", self.le_starnet.text().strip())
        self.settings.setValue("paths/astap", self.le_astap.text().strip())
        self.settings.setValue("shortcuts/save_on_exit", self.chk_save_shortcuts.isChecked())
        self.settings.setValue("api/astrometry_key", self.le_astrometry.text().strip())

        # NEW: WIMS defaults (keys match WhatsInMySky.load_settings / save_settings)
        self.settings.setValue("latitude", float(self.sp_lat.value()))
        self.settings.setValue("longitude", float(self.sp_lon.value()))
        self.settings.setValue("date", self.le_date.text().strip())
        self.settings.setValue("time", self.le_time.text().strip())
        self.settings.setValue("timezone", self.cb_tz.currentText())
        self.settings.setValue("min_altitude", float(self.sp_min_alt.value()))
        self.settings.setValue("object_limit", int(self.sp_obj_limit.value()))

        self.settings.setValue("updates/check_on_startup", self.chk_updates_startup.isChecked())
        self.settings.setValue("updates/url", self.le_updates_url.text().strip())

        self.settings.setValue("display/autostretch_16bit", self.chk_autostretch_16bit.isChecked())

        # Theme
        idx = self.cb_theme.currentIndex()
        if idx < 0:
            idx = 0
        theme_val = "dark" if idx == 0 else "light"
        self.settings.setValue("ui/theme", theme_val)

        self.settings.sync()
        # Apply now if the parent knows how
        p = self.parent()
        if p and hasattr(p, "apply_theme_from_settings"):
            try:
                p.apply_theme_from_settings()
            except Exception:
                pass        
        self.accept()
