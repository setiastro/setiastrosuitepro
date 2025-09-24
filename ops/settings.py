# ops.settings.py
from PyQt6.QtWidgets import (
    QLineEdit, QDialogButtonBox, QFileDialog, QDialog, QPushButton, QFormLayout,
    QHBoxLayout, QVBoxLayout, QWidget, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLabel)
from PyQt6.QtCore import QSettings, Qt
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
            lambda: self.le_updates_url.setText(
                "https://raw.githubusercontent.com/setiastro/setiastrosuitepro/main/updates.json"
            )
        )

        # Optional: “Check Now…” button (only shown if the parent has the method)
        self.btn_check_now = QPushButton("Check Now…")
        self.btn_check_now.setToolTip("Run an update check immediately")
        self.btn_check_now.setVisible(hasattr(parent, "_check_for_updates_async"))
        self.btn_check_now.clicked.connect(self._check_updates_now_clicked)

        # Build the updates URL row (used later in the right column)
        row_updates_url = QHBoxLayout()
        row_updates_url.addWidget(self.le_updates_url, 1)
        row_updates_url.addWidget(btn_reset_updates_url)
        row_updates_url.addWidget(self.btn_check_now)

        self.chk_save_shortcuts = QCheckBox("Save desktop shortcuts on exit")
        self.chk_save_shortcuts.setChecked(
            self.settings.value("shortcuts/save_on_exit", True, type=bool)
        )

        self.cb_theme = QComboBox()
        self.cb_theme.clear()
        self.cb_theme.addItems(["Dark", "Light", "System"])
        theme_val = (self.settings.value("ui/theme", "system", type=str) or "system").lower()
        self.cb_theme.setCurrentIndex({"dark": 0, "light": 1, "system": 2}.get(theme_val, 2))

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

        # ---- Layout (two columns, no scroll) ----
        root = QVBoxLayout(self)

        cols = QHBoxLayout()
        root.addLayout(cols)

        left_col  = QFormLayout()
        right_col = QFormLayout()

        # keep forms tidy and top-aligned
        for f in (left_col, right_col):
            f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            f.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        cols.addLayout(left_col, 1)
        cols.addSpacing(16)
        cols.addLayout(right_col, 1)

        # ---- Left column: Paths & Integrations ----
        left_col.addRow(QLabel("<b>Paths & Integrations</b>"))

        w = QWidget(); w.setLayout(row_grax)
        left_col.addRow("GraXpert executable:", w)

        w = QWidget(); w.setLayout(row_ccl)
        left_col.addRow("Cosmic Clarity folder:", w)

        w = QWidget(); w.setLayout(row_star)
        left_col.addRow("StarNet executable:", w)

        w = QWidget(); w.setLayout(row_astap)
        left_col.addRow("ASTAP executable:", w)

        left_col.addRow("Astrometry.net API key:", self.le_astrometry)
        left_col.addRow(self.chk_save_shortcuts)
        left_col.addRow("Theme:", self.cb_theme)

        # ---- Right column: WIMS + Updates + Display ----
        right_col.addRow(QLabel("<b>What's In My Sky — Defaults</b>"))
        right_col.addRow("Latitude (°):", self.sp_lat)
        right_col.addRow("Longitude (°):", self.sp_lon)
        right_col.addRow("Date (YYYY-MM-DD):", self.le_date)
        right_col.addRow("Time (HH:MM):", self.le_time)
        right_col.addRow("Time Zone:", self.cb_tz)
        right_col.addRow("Min Altitude (°):", self.sp_min_alt)
        right_col.addRow("Object Limit:", self.sp_obj_limit)

        right_col.addRow(QLabel("<b>Updates</b>"))
        right_col.addRow(self.chk_updates_startup)

        row_updates_url = QHBoxLayout()
        row_updates_url.addWidget(self.le_updates_url, 1)
        row_updates_url.addWidget(btn_reset_updates_url)
        row_updates_url.addWidget(self.btn_check_now)
        w = QWidget(); w.setLayout(row_updates_url)
        right_col.addRow("Updates JSON URL:", w)

        right_col.addRow(QLabel("<b>Display</b>"))
        right_col.addRow(self.chk_autostretch_16bit)

        # ---- Buttons ----
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self
        )
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

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

    def _save_and_accept(self):
        # Paths / Integrations
        self.settings.setValue("paths/graxpert", self.le_graxpert.text().strip())
        self.settings.setValue("paths/cosmic_clarity", self.le_cosmic.text().strip())
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

        # Updates + Display
        self.settings.setValue("updates/check_on_startup", self.chk_updates_startup.isChecked())
        self.settings.setValue("updates/url", self.le_updates_url.text().strip())
        self.settings.setValue("display/autostretch_16bit", self.chk_autostretch_16bit.isChecked())

        # Theme
        idx = max(0, self.cb_theme.currentIndex())
        theme_val = "dark" if idx == 0 else ("light" if idx == 1 else "system")
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
