# whatsinmysky.py
from __future__ import annotations

# --- stdlib ---
import os
import sys
import shutil
import warnings
import webbrowser
from datetime import datetime
from decimal import getcontext
from typing import Optional

# --- third-party ---
import numpy as np
import pandas as pd
import pytz
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body
from astropy.time import Time

# --- Qt / PyQt6 ---
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QComboBox, QCheckBox, QRadioButton, QButtonGroup,
    QPushButton, QGridLayout, QTreeWidget, QTreeWidgetItem, QHeaderView, QFileDialog,
    QScrollArea, QInputDialog, QMessageBox, QWidget, QHBoxLayout
)

# ---------------------------------------------------
#  paths / globals
# ---------------------------------------------------
def _app_root() -> str:
    # this file sits next to setiastrosuitepro.py and imgs/
    return getattr(sys, "_MEIPASS", os.path.dirname(__file__))

def imgs_path(*parts) -> str:
    return os.path.join(_app_root(), "imgs", *parts)

getcontext().prec = 24
warnings.filterwarnings("ignore")


# ---------------------------------------------------
#  Worker thread
# ---------------------------------------------------
class CalculationThread(QThread):
    calculation_complete = pyqtSignal(pd.DataFrame, str)
    lunar_phase_calculated = pyqtSignal(int, str)  # phase_percentage, phase_image_name
    lst_calculated = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(
        self,
        latitude: float,
        longitude: float,
        date: str,
        time: str,
        timezone: str,
        min_altitude: float,
        catalog_filters: list[str],
        object_limit: int,
    ):
        super().__init__()
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.date = date
        self.time = time
        self.timezone = timezone
        self.min_altitude = float(min_altitude)
        self.catalog_filters = list(catalog_filters or [])
        self.object_limit = int(object_limit)

        self.catalog_file = self.get_catalog_file_path()

    def get_catalog_file_path(self) -> str:
        user_catalog_path = os.path.join(os.path.expanduser("~"), "celestial_catalog.csv")
        if not os.path.exists(user_catalog_path):
            bundled = os.path.join(_app_root(), "data", "catalogs", "celestial_catalog.csv")
            if os.path.exists(bundled):
                try: shutil.copyfile(bundled, user_catalog_path)
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        return user_catalog_path

    def run(self):
        try:
            # local date/time → astropy Time
            local_tz = pytz.timezone(self.timezone)
            naive = datetime.strptime(f"{self.date} {self.time}", "%Y-%m-%d %H:%M")
            local_dt = local_tz.localize(naive)
            t = Time(local_dt)

            # observer + LST
            loc = EarthLocation(lat=self.latitude * u.deg, lon=self.longitude * u.deg, height=0 * u.m)
            lst = t.sidereal_time("apparent", self.longitude * u.deg)
            self.lst_calculated.emit(f"Local Sidereal Time: {lst.to_string(unit=u.hour, precision=3)}")

            # moon phase + icon
            phase_pct, phase_icon = self.calculate_lunar_phase(t, loc)
            self.lunar_phase_calculated.emit(phase_pct, phase_icon)

            # load catalog
            catalog_file = self.catalog_file
            if not os.path.exists(catalog_file):
                self.calculation_complete.emit(pd.DataFrame(), "Catalog file not found.")
                return
            df = pd.read_csv(catalog_file, encoding="ISO-8859-1")

            if self.catalog_filters:
                df = df[df["Catalog"].isin(self.catalog_filters)]
            df.dropna(subset=["RA", "Dec"], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # coordinates → AltAz
            sky = SkyCoord(ra=df["RA"].to_numpy() * u.deg, dec=df["Dec"].to_numpy() * u.deg, frame="icrs")
            altaz_frame = AltAz(obstime=t, location=loc)
            altaz = sky.transform_to(altaz_frame)
            df["Altitude"] = np.round(altaz.alt.deg, 1)
            df["Azimuth"]  = np.round(altaz.az.deg, 1)

            # separation from Moon
            moon_altaz = get_body("moon", t, loc).transform_to(altaz_frame)
            df["Degrees from Moon"] = np.round(altaz.separation(moon_altaz).deg, 2)

            # altitude gate
            df = df[df["Altitude"] >= self.min_altitude]

            # minutes to transit
            ra_hours = df["RA"].to_numpy() * (24.0 / 360.0)
            minutes = ((ra_hours - lst.hour) * u.hour) % (24 * u.hour)
            mins = minutes.to_value(u.hour) * 60.0
            df["Minutes to Transit"] = np.round(mins, 1)
            df["Before/After Transit"] = np.where(df["Minutes to Transit"] > 720, "After", "Before")
            df["Minutes to Transit"]   = np.where(df["Minutes to Transit"] > 720,
                                                  1440 - df["Minutes to Transit"],
                                                  df["Minutes to Transit"])

            # pick N nearest
            df = df.nsmallest(self.object_limit, "Minutes to Transit")
            self.calculation_complete.emit(df, "Calculation complete.")
        except Exception as e:
            self.calculation_complete.emit(pd.DataFrame(), f"Error: {e!s}")

    def calculate_lunar_phase(self, t: Time, loc: EarthLocation):
        moon = get_body("moon", t, loc)
        sun  = get_sun(t)
        elong = moon.separation(sun).deg

        phase_pct = int(round((1 - np.cos(np.radians(elong))) / 2 * 100))

        future = t + (6 * u.hour)
        is_waxing = get_body("moon", future, loc).separation(get_sun(future)).deg > elong

        name = "new_moon.png"
        if   0   <= elong < 9:   name = "new_moon.png"
        elif 9   <= elong < 18:  name = "waxing_crescent_1.png" if is_waxing else "waning_crescent_5.png"
        elif 18  <= elong < 27:  name = "waxing_crescent_2.png" if is_waxing else "waning_crescent_4.png"
        elif 27  <= elong < 36:  name = "waxing_crescent_3.png" if is_waxing else "waning_crescent_3.png"
        elif 36  <= elong < 45:  name = "waxing_crescent_4.png" if is_waxing else "waning_crescent_2.png"
        elif 45  <= elong < 54:  name = "waxing_crescent_5.png" if is_waxing else "waning_crescent_1.png"
        elif 54  <= elong < 90:  name = "first_quarter.png"
        elif 90  <= elong < 108: name = "waxing_gibbous_1.png" if is_waxing else "waning_gibbous_4.png"
        elif 108 <= elong < 126: name = "waxing_gibbous_2.png" if is_waxing else "waning_gibbous_3.png"
        elif 126 <= elong < 144: name = "waxing_gibbous_3.png" if is_waxing else "waning_gibbous_2.png"
        elif 144 <= elong < 162: name = "waxing_gibbous_4.png" if is_waxing else "waning_gibbous_1.png"
        elif 162 <= elong <= 180: name = "full_moon.png"

        return phase_pct, name


# ---------------------------------------------------
#  UI dialog
# ---------------------------------------------------
class SortableTreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        numeric_cols = [3, 4, 5, 7, 10]  # Alt, Az, Minutes, Sep, Mag
        if col in numeric_cols:
            try:
                return float(self.text(col)) < float(other.text(col))
            except ValueError:
                return self.text(col) < other.text(col)
        return self.text(col) < other.text(col)

# ---------- coordinate parsing / formatting ----------
def _parse_deg_with_suffix(txt: str, kind: str) -> float:
    """
    Parse latitude/longitude accepting:
      30.1, -111, "30.1N", "111W", " -30.0 s ", etc.
    kind: "lat" or "lon" (for range checks and suffix semantics)
    Returns signed decimal degrees (E+, W-, N+, S-).
    Raises ValueError on bad input.
    """
    if txt is None:
        raise ValueError("empty")
    t = str(txt).strip().replace("°", "")
    if not t:
        raise ValueError("empty")

    # extract trailing letter (N/S/E/W), case-insensitive
    suffix = ""
    if t and t[-1].upper() in ("N", "S", "E", "W"):
        suffix = t[-1].upper()
        t = t[:-1].strip()

    val = float(t)  # may be signed already

    # apply suffix to sign if present
    if suffix:
        if kind == "lat":
            if suffix == "N":
                val = abs(val)
            elif suffix == "S":
                val = -abs(val)
            else:
                raise ValueError("Latitude suffix must be N or S")
        elif kind == "lon":
            if suffix == "E":
                val = abs(val)   # E is positive
            elif suffix == "W":
                val = -abs(val)  # W is negative
            else:
                raise ValueError("Longitude suffix must be E or W")

    # clamp / validate ranges
    if kind == "lat":
        if not (-90.0 <= val <= 90.0):
            raise ValueError("Latitude must be in [-90, 90]")
    else:
        if not (-180.0 <= val <= 180.0):
            raise ValueError("Longitude must be in [-180, 180]")

    return val


def _format_with_suffix(val: float, kind: str) -> str:
    """
    Render signed degrees with hemisphere suffix.
    e.g. lat  -33.5 -> '33.5S'
         lon  -111  -> '111W'
    """
    v = float(val)
    if kind == "lat":
        hemi = "N" if v >= 0 else "S"
    else:
        hemi = "E" if v >= 0 else "W"
    return f"{abs(v):g}{hemi}"

def _tz_vs_longitude_hint(tz_name: str, date_str: str, time_str: str, lon_deg: float):
    """
    Compare timezone UTC offset to longitude.
    Heuristic:
      • sign check: West longitudes (~W) usually have negative UTC offsets; East longitudes (~E) positive
      • central meridian check: |lon| should be near |offset_hours*15|; flag if > 45°
    Returns (should_warn: bool, human_msg: str, utc_str: str, central_meridian: float)
    """
    try:
        local_tz = pytz.timezone(tz_name)
        naive = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        local_dt = local_tz.localize(naive)
        off_hours = (local_dt.utcoffset() or pd.Timedelta(0)).total_seconds() / 3600.0
    except Exception:
        return (False, "", "", 0.0)

    # UTC string like UTC−7 or UTC+5:30
    hours = int(off_hours)
    mins  = int(round(abs(off_hours - hours) * 60))
    sign  = "−" if off_hours < 0 else "+"
    if mins:
        utc_str = f"UTC{sign}{abs(hours)}:{mins:02d}"
    else:
        utc_str = f"UTC{sign}{abs(hours)}"

    central = off_hours * 15.0  # “central meridian” for that offset
    sign_ok = (abs(off_hours) < 1e-9) or (lon_deg == 0) or ((lon_deg > 0) == (off_hours > 0))
    far     = abs(abs(lon_deg) - abs(central)) > 45.0

    if (not sign_ok) or far:
        msg = (f"Timezone {tz_name} ({utc_str}) looks inconsistent with longitude "
               f"{abs(lon_deg):g}{'E' if lon_deg>0 else 'W'} "
               f"(central meridian ≈ {abs(central):.0f}°{'E' if central>0 else 'W'}).")
        return (True, msg, utc_str, central)
    return (False, "", utc_str, central)


class WhatsInMySkyDialog(QDialog):
    def __init__(self, parent=None, wims_path: Optional[str] = None, wrench_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("What's In My Sky")
        if wims_path:
            self.setWindowIcon(QIcon(wims_path))

        self.settings = QSettings()
        self.object_limit = int(self.settings.value("object_limit", 100, int))

        self._build_ui(wrench_path)
        self._load_settings_into_ui()

        self.calc_thread: Optional[CalculationThread] = None
        self.catalog_file: Optional[str] = None

    # ---------- UI ----------
    def _build_ui(self, wrench_path: Optional[str]):
        layout = QGridLayout(self)
        fixed_w = 150

        self.latitude_entry  = QLineEdit(); self.latitude_entry.setFixedWidth(fixed_w)
        self.longitude_entry = QLineEdit(); self.longitude_entry.setFixedWidth(fixed_w)
        self.date_entry      = QLineEdit(); self.date_entry.setFixedWidth(fixed_w)
        self.time_entry      = QLineEdit(); self.time_entry.setFixedWidth(fixed_w)

        self.timezone_combo  = QComboBox(); self.timezone_combo.addItems(pytz.all_timezones)
        self.timezone_combo.setFixedWidth(fixed_w)

        r = 0
        layout.addWidget(QLabel("Latitude:"), r, 0); layout.addWidget(self.latitude_entry, r, 1); r += 1
        layout.addWidget(QLabel("Longitude (E+, W−):"), r, 0); layout.addWidget(self.longitude_entry, r, 1); r += 1
        layout.addWidget(QLabel("Date (YYYY-MM-DD):"), r, 0); layout.addWidget(self.date_entry, r, 1); r += 1
        layout.addWidget(QLabel("Time (HH:MM):"), r, 0); layout.addWidget(self.time_entry, r, 1); r += 1
        layout.addWidget(QLabel("Time Zone:"), r, 0); layout.addWidget(self.timezone_combo, r, 1); r += 1

        self.min_altitude_entry = QLineEdit(); self.min_altitude_entry.setFixedWidth(fixed_w)
        layout.addWidget(QLabel("Min Altitude (0–90°):"), r, 0); layout.addWidget(self.min_altitude_entry, r, 1); r += 1

        # catalogs
        catalog_frame = QScrollArea()
        cat_widget = QWidget(); cat_layout = QGridLayout(cat_widget)
        self.catalog_vars: dict[str, QCheckBox] = {}
        for i, name in enumerate(["Messier","NGC","IC","Caldwell","Abell","Sharpless","LBN","LDN","PNG","User"]):
            cb = QCheckBox(name); cb.setChecked(False)
            cat_layout.addWidget(cb, i // 5, i % 5)
            self.catalog_vars[name] = cb
        catalog_frame.setWidget(cat_widget); catalog_frame.setFixedWidth(fixed_w + 250)
        layout.addWidget(QLabel("Catalog Filters:"), r, 0); layout.addWidget(catalog_frame, r, 1); r += 1

        # RA/Dec format
        self.ra_dec_degrees = QRadioButton("Degrees")
        self.ra_dec_hms     = QRadioButton("H:M:S / D:M:S")
        self.ra_dec_degrees.setChecked(True)
        g = QButtonGroup(self); g.addButton(self.ra_dec_degrees); g.addButton(self.ra_dec_hms)
        ra_row = QHBoxLayout(); ra_row.addWidget(self.ra_dec_degrees); ra_row.addWidget(self.ra_dec_hms)
        layout.addWidget(QLabel("RA/Dec Format:"), r, 0); layout.addLayout(ra_row, r, 1); r += 1
        self.ra_dec_degrees.toggled.connect(self.update_ra_dec_format)
        self.ra_dec_hms.toggled.connect(self.update_ra_dec_format)

        # action buttons / status
        calc_btn = QPushButton("Calculate"); calc_btn.setFixedWidth(fixed_w); calc_btn.clicked.connect(self.start_calculation)
        layout.addWidget(calc_btn, r, 0); r += 1

        self.status_label = QLabel("Status: Idle"); layout.addWidget(self.status_label, r, 0, 1, 2); r += 1
        self.lst_label    = QLabel("Local Sidereal Time: 0.000"); layout.addWidget(self.lst_label, r, 0, 1, 2); r += 1

        # moon phase preview
        self.lunar_phase_image_label = QLabel()
        layout.addWidget(self.lunar_phase_image_label, 0, 2, 4, 1)
        self.lunar_phase_label = QLabel("Lunar Phase: N/A")
        layout.addWidget(self.lunar_phase_label, 4, 2)

        # results tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([
            "Name","RA","Dec","Altitude","Azimuth","Minutes to Transit","Before/After Transit",
            "Degrees from Moon","Alt Name","Type","Magnitude","Size (arcmin)"
        ])
        self.tree.setSortingEnabled(True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        self.tree.sortByColumn(5, Qt.SortOrder.AscendingOrder)
        self.tree.itemDoubleClicked.connect(self.on_row_double_click)
        layout.addWidget(self.tree, r, 0, 1, 3); r += 1

        # bottom row
        add_btn = QPushButton("Add Custom Object"); add_btn.setFixedWidth(fixed_w); add_btn.clicked.connect(self.add_custom_object)
        layout.addWidget(add_btn, r, 0)

        save_btn = QPushButton("Save to CSV"); save_btn.setFixedWidth(fixed_w); save_btn.clicked.connect(self.save_to_csv)
        layout.addWidget(save_btn, r, 1)

        settings_btn = QPushButton(); settings_btn.setFixedWidth(fixed_w)
        if wrench_path and os.path.exists(wrench_path):
            settings_btn.setIcon(QIcon(wrench_path))
        settings_btn.clicked.connect(self.open_settings)
        layout.addWidget(settings_btn, r, 2)

        layout.setColumnStretch(2, 1)

    # ---------- settings ----------
    def _load_settings_into_ui(self):
        def cast(v, typ, default):
            try: return typ(v)
            except Exception: return default
        lat = cast(self.settings.value("latitude", 0.0), float, 0.0)
        lon = cast(self.settings.value("longitude", 0.0), float, 0.0)
        date = self.settings.value("date", datetime.now().strftime("%Y-%m-%d"))
        time = self.settings.value("time", "00:00")
        tz   = self.settings.value("timezone", "UTC")
        min_alt = cast(self.settings.value("min_altitude", 0.0), float, 0.0)
        self.object_limit = cast(self.settings.value("object_limit", 100), int, 100)

        self.latitude_entry.setText(str(lat))
        self.longitude_entry.setText(str(lon))
        self.date_entry.setText(date)
        self.time_entry.setText(time)
        self.timezone_combo.setCurrentText(tz)
        self.min_altitude_entry.setText(str(min_alt))

    def _save_settings(self, latitude, longitude, date, time, timezone, min_altitude):
        self.settings.setValue("latitude", latitude)
        self.settings.setValue("longitude", longitude)
        self.settings.setValue("date", date)
        self.settings.setValue("time", time)
        self.settings.setValue("timezone", timezone)
        self.settings.setValue("min_altitude", min_altitude)

    # ---------- actions ----------
    def start_calculation(self):
        try:
            orig_lat_txt = self.latitude_entry.text()
            orig_lon_txt = self.longitude_entry.text()

            latitude  = _parse_deg_with_suffix(orig_lat_txt,  "lat")
            longitude = _parse_deg_with_suffix(orig_lon_txt, "lon")

            # Pretty-print back with suffixes
            self.latitude_entry.setText(_format_with_suffix(latitude,  "lat"))
            self.longitude_entry.setText(_format_with_suffix(longitude, "lon"))

            date_str  = self.date_entry.text().strip()
            time_str  = self.time_entry.text().strip()
            tz_str    = self.timezone_combo.currentText()
            min_alt   = float(self.min_altitude_entry.text())
        except ValueError as e:
            self.update_status(f"Invalid input: {e}")
            return

        # Heuristic warning (and gentle auto-fix if user probably forgot the suffix)
        warn, msg, utc_str, central = _tz_vs_longitude_hint(tz_str, date_str, time_str, longitude)
        if warn:
            # If the user typed a bare number (no N/S/E/W) and sign mismatches TZ, suggest flip
            bare_lon = (orig_lon_txt.strip() and orig_lon_txt.strip()[-1].upper() not in ("E","W"))
            sign_mismatch = not ((longitude > 0) == (central > 0) or abs(central) < 1e-6 or longitude == 0)

            if bare_lon and sign_mismatch:
                # Flip once, write back, and tell the user.
                longitude = -longitude
                self.longitude_entry.setText(_format_with_suffix(longitude, "lon"))
                self.update_status(f"{msg} → Assuming you meant {_format_with_suffix(longitude, 'lon')} (auto-corrected).")
            else:
                self.update_status(msg + " Please verify your longitude/timezone.")
        else:
            self.update_status("Inputs look consistent.")

        # Persist settings (numeric)
        self._save_settings(latitude, longitude, date_str, time_str, tz_str, min_alt)

        catalogs = [name for name, cb in self.catalog_vars.items() if cb.isChecked()]
        self.calc_thread = CalculationThread(latitude, longitude, date_str, time_str, tz_str,
                                             min_alt, catalogs, self.object_limit)
        self.catalog_file = self.calc_thread.catalog_file

        self.calc_thread.calculation_complete.connect(self.on_calculation_complete)
        self.calc_thread.lunar_phase_calculated.connect(self.update_lunar_phase)
        self.calc_thread.lst_calculated.connect(self.update_lst)
        self.calc_thread.status_update.connect(self.update_status)

        self.update_status("Calculating…")
        self.calc_thread.start()

    def update_lunar_phase(self, phase_percentage: int, phase_image_name: str):
        self.lunar_phase_label.setText(f"Lunar Phase: {phase_percentage}% illuminated")
        pth = imgs_path(phase_image_name)
        if os.path.exists(pth):
            pm = QPixmap(pth).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
            self.lunar_phase_image_label.setPixmap(pm)

    def on_calculation_complete(self, df: pd.DataFrame, message: str):
        self.update_status(message)
        self.tree.clear()
        if df.empty:
            return
        for _, row in df.iterrows():
            ra_disp, dec_disp = row["RA"], row["Dec"]
            if self.ra_dec_hms.isChecked():
                sc = SkyCoord(ra=row["RA"] * u.deg, dec=row["Dec"] * u.deg)
                ra_disp  = sc.ra.to_string(unit=u.hour, sep=":")
                dec_disp = sc.dec.to_string(unit=u.deg,  sep=":")
            size_arcmin = row.get("Info", "")
            if pd.notna(size_arcmin):
                size_arcmin = str(size_arcmin)
            vals = [
                str(row.get("Name","") or ""),
                str(ra_disp),
                str(dec_disp),
                str(row.get("Altitude","")),
                str(row.get("Azimuth","")),
                str(int(row.get("Minutes to Transit",0))) if pd.notna(row.get("Minutes to Transit", np.nan)) else "",
                str(row.get("Before/After Transit","")),
                str(round(row.get("Degrees from Moon", 0.0), 2)) if pd.notna(row.get("Degrees from Moon", np.nan)) else "",
                row.get("Alt Name","") if pd.notna(row.get("Alt Name","")) else "",
                row.get("Type","") if pd.notna(row.get("Type","")) else "",
                str(row.get("Magnitude","")) if pd.notna(row.get("Magnitude","")) else "",
                str(size_arcmin) if pd.notna(size_arcmin) else "",
            ]
            self.tree.addTopLevelItem(SortableTreeWidgetItem(vals))

    def update_status(self, msg: str):
        self.status_label.setText(f"Status: {msg}")

    def update_lst(self, msg: str):
        self.lst_label.setText(msg)

    def open_settings(self):
        n, ok = QInputDialog.getInt(self, "Settings", "Enter number of objects to display:",
                                    value=int(self.object_limit), min=1, max=1000)
        if ok:
            self.object_limit = int(n)
            self.settings.setValue("object_limit", int(n))

    def on_row_double_click(self, item: QTreeWidgetItem, column: int):
        name = item.text(0).replace(" ", "")
        webbrowser.open(f"https://www.astrobin.com/search/?q={name}")

    def add_custom_object(self):
        name, ok = QInputDialog.getText(self, "Add Custom Object", "Enter object name:")
        if not ok or not name:
            return
        ra, ok = QInputDialog.getDouble(self, "Add Custom Object", "Enter RA (deg):", decimals=3)
        if not ok: return
        dec, ok = QInputDialog.getDouble(self, "Add Custom Object", "Enter Dec (deg):", decimals=3)
        if not ok: return

        entry = {"Name": name, "RA": ra, "Dec": dec, "Catalog": "User",
                 "Alt Name": "User Defined", "Type": "Custom", "Magnitude": "", "Info": ""}

        catalog_csv = self.catalog_file or os.path.join(os.path.expanduser("~"), "celestial_catalog.csv")
        try:
            df = pd.read_csv(catalog_csv, encoding="ISO-8859-1") if os.path.exists(catalog_csv) else pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(catalog_csv, index=False, encoding="ISO-8859-1")
            self.update_status(f"Added custom object: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Add Custom Object", f"Could not update catalog:\n{e}")

    def update_ra_dec_format(self):
        use_deg = self.ra_dec_degrees.isChecked()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            ra_txt, dec_txt = it.text(1), it.text(2)
            try:
                if use_deg:
                    if ":" in ra_txt:
                        sc = SkyCoord(ra=ra_txt, dec=dec_txt, unit=(u.hourangle, u.deg))
                        it.setText(1, f"{sc.ra.deg:.3f}")
                        it.setText(2, f"{sc.dec.deg:.3f}")
                else:
                    if ":" not in ra_txt:
                        sc = SkyCoord(ra=float(ra_txt) * u.deg, dec=float(dec_txt) * u.deg)
                        it.setText(1, sc.ra.to_string(unit=u.hour, sep=":"))
                        it.setText(2, sc.dec.to_string(unit=u.deg,  sep=":"))
            except Exception:
                pass

    def save_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV files (*.csv);;All Files (*)")
        if not path:
            return
        cols = [self.tree.headerItem().text(i) for i in range(self.tree.columnCount())]
        rows = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            rows.append([it.text(j) for j in range(self.tree.columnCount())])
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
        self.update_status(f"Data saved to {path}")
