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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QAction
from PyQt6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QComboBox, QCheckBox, QRadioButton, QButtonGroup,
    QPushButton, QGridLayout, QTreeWidget, QTreeWidgetItem, QHeaderView, QFileDialog,
    QScrollArea, QInputDialog, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFrame, QTabWidget, QTextEdit, QMenu
)

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

# ---------------------------------------------------
#  paths / globals
# ---------------------------------------------------
def _app_root() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(__file__))

def imgs_path(*parts) -> str:
    return os.path.join(_app_root(), "imgs", *parts)

from setiastro.saspro.resources import get_icon_path

getcontext().prec = 24
warnings.filterwarnings("ignore")


# ---------------------------------------------------
#  Worker thread
# ---------------------------------------------------
class CalculationThread(QThread):
    calculation_complete   = pyqtSignal(pd.DataFrame, str)
    lunar_phase_calculated = pyqtSignal(int, str, str)
    lst_calculated         = pyqtSignal(str)
    status_update          = pyqtSignal(str)

    def __init__(self, latitude, longitude, date, time, timezone,
                 min_altitude, catalog_filters, object_limit):
        super().__init__()
        self.latitude        = float(latitude)
        self.longitude       = float(longitude)
        self.date            = date
        self.time            = time
        self.timezone        = timezone
        self.min_altitude    = float(min_altitude)
        self.catalog_filters = list(catalog_filters or [])
        self.object_limit    = int(object_limit)
        self.catalog_file    = self.get_catalog_file_path()

    def get_catalog_file_path(self) -> str:
        user_catalog_path = os.path.join(os.path.expanduser("~"), "celestial_catalog.csv")
        if not os.path.exists(user_catalog_path):
            bundled = os.path.join(_app_root(), "data", "catalogs", "celestial_catalog.csv")
            if os.path.exists(bundled):
                try:
                    shutil.copyfile(bundled, user_catalog_path)
                except Exception:
                    pass
        return user_catalog_path

    def run(self):
        try:
            local_tz = pytz.timezone(self.timezone)
            naive    = datetime.strptime(f"{self.date} {self.time}", "%Y-%m-%d %H:%M")
            local_dt = local_tz.localize(naive)
            t        = Time(local_dt)

            loc = EarthLocation(lat=self.latitude * u.deg,
                                lon=self.longitude * u.deg, height=0 * u.m)
            lst = t.sidereal_time("apparent", self.longitude * u.deg)
            self.lst_calculated.emit(
                f"Local Sidereal Time: {lst.to_string(unit=u.hour, precision=3)}")

            phase_pct, phase_icon, rts = self.calculate_lunar_phase(t, loc, self.timezone)
            self.lunar_phase_calculated.emit(phase_pct, phase_icon, rts)

            catalog_file = self.catalog_file
            if not os.path.exists(catalog_file):
                self.calculation_complete.emit(pd.DataFrame(), "Catalog file not found.")
                return
            df = pd.read_csv(catalog_file, encoding="ISO-8859-1")

            if self.catalog_filters:
                df = df[df["Catalog"].isin(self.catalog_filters)]
            df.dropna(subset=["RA", "Dec"], inplace=True)
            df.reset_index(drop=True, inplace=True)

            sky         = SkyCoord(ra=df["RA"].to_numpy() * u.deg,
                                   dec=df["Dec"].to_numpy() * u.deg, frame="icrs")
            altaz_frame = AltAz(obstime=t, location=loc)
            altaz       = sky.transform_to(altaz_frame)
            df["Altitude"] = np.round(altaz.alt.deg, 1)
            df["Azimuth"]  = np.round(altaz.az.deg,  1)

            moon_altaz = get_body("moon", t, loc).transform_to(altaz_frame)
            df["Degrees from Moon"] = np.round(altaz.separation(moon_altaz).deg, 2)

            df = df[df["Altitude"] >= self.min_altitude]

            ra_hours = df["RA"].to_numpy() * (24.0 / 360.0)
            minutes  = ((ra_hours - lst.hour) * u.hour) % (24 * u.hour)
            mins     = minutes.to_value(u.hour) * 60.0
            df["Minutes to Transit"]   = np.round(mins, 1)
            df["Before/After Transit"] = np.where(df["Minutes to Transit"] > 720,
                                                   "After", "Before")
            df["Minutes to Transit"]   = np.where(df["Minutes to Transit"] > 720,
                                                   1440 - df["Minutes to Transit"],
                                                   df["Minutes to Transit"])

            df = df.nsmallest(self.object_limit, "Minutes to Transit")
            self.calculation_complete.emit(df, "Calculation complete.")
        except Exception as e:
            self.calculation_complete.emit(pd.DataFrame(), f"Error: {e!s}")

    def calculate_lunar_phase(self, t, loc, tz_name):
        moon  = get_body("moon", t, loc)
        sun   = get_sun(t)
        elong = moon.separation(sun).deg
        phase_pct = int(round((1 - np.cos(np.radians(elong))) / 2 * 100))
        future    = t + (6 * u.hour)
        is_waxing = (get_body("moon", future, loc)
                     .separation(get_sun(future)).deg > elong)

        name = "new_moon.png"
        if   0   <= elong < 9:    name = "new_moon.png"
        elif 9   <= elong < 18:   name = "waxing_crescent_1.png" if is_waxing else "waning_crescent_5.png"
        elif 18  <= elong < 27:   name = "waxing_crescent_2.png" if is_waxing else "waning_crescent_4.png"
        elif 27  <= elong < 36:   name = "waxing_crescent_3.png" if is_waxing else "waning_crescent_3.png"
        elif 36  <= elong < 45:   name = "waxing_crescent_4.png" if is_waxing else "waning_crescent_2.png"
        elif 45  <= elong < 54:   name = "waxing_crescent_5.png" if is_waxing else "waning_crescent_1.png"
        elif 54  <= elong < 90:   name = "first_quarter.png"
        elif 90  <= elong < 108:  name = "waxing_gibbous_1.png"  if is_waxing else "waning_gibbous_4.png"
        elif 108 <= elong < 126:  name = "waxing_gibbous_2.png"  if is_waxing else "waning_gibbous_3.png"
        elif 126 <= elong < 144:  name = "waxing_gibbous_3.png"  if is_waxing else "waning_gibbous_2.png"
        elif 144 <= elong < 162:  name = "waxing_gibbous_4.png"  if is_waxing else "waning_gibbous_1.png"
        elif 162 <= elong <= 180: name = "full_moon.png"

        if   elong < 9:    phase_emoji = "🌑"
        elif elong < 45:   phase_emoji = "🌒" if is_waxing else "🌘"
        elif elong < 90:   phase_emoji = "🌓" if is_waxing else "🌗"
        elif elong < 135:  phase_emoji = "🌔" if is_waxing else "🌖"
        elif elong <= 180: phase_emoji = "🌕"
        else:              phase_emoji = "🌕"

        rts = self._moon_rise_transit_set(t, loc, tz_name, phase_emoji)
        return phase_pct, name, rts

    def _moon_rise_transit_set(self, t, loc, tz_name, phase_emoji="🌕"):
        try:
            local_tz = pytz.timezone(tz_name)
            t_start  = t - 12 * u.hour
            times    = t_start + np.linspace(0, 24, 1440) * u.hour
            alts     = (get_body("moon", times, loc)
                        .transform_to(AltAz(obstime=times, location=loc)).alt.deg)

            def _to_local(at):
                return (at.to_datetime(timezone=pytz.utc)
                        .astimezone(local_tz).strftime("%H:%M"))

            transit_str = _to_local(times[int(np.argmax(alts))])
            crossings   = np.where(np.diff(np.sign(alts)))[0]
            t_jd        = t.jd
            times_jd    = np.array([tt.jd for tt in times])
            rise_cands, set_cands = [], []

            for ci in crossings:
                rising   = alts[ci + 1] > alts[ci]
                cjd      = 0.5 * (times_jd[ci] + times_jd[ci + 1])
                cands    = rise_cands if rising else set_cands
                cands.append((abs(cjd - t_jd), times[ci]))

            rise_str = _to_local(min(rise_cands)[1]) if rise_cands else "—"
            set_str  = _to_local(min(set_cands)[1])  if set_cands  else "—"
            return f"{phase_emoji} Rise: {rise_str}   Transit: {transit_str}   Set: {set_str}"
        except Exception as e:
            return f"Moon times unavailable ({e})"


# ---------------------------------------------------
#  Object detail worker (runs SIMBAD query off-thread)
# ---------------------------------------------------
class ObjectDetailThread(QThread):
    result_ready = pyqtSignal(dict)

    def __init__(self, name, ra_deg, dec_deg):
        super().__init__()
        self.name   = name
        self.ra_deg = float(ra_deg)
        self.dec_deg= float(dec_deg)

    def run(self):
        info = {"name": self.name, "ra": self.ra_deg, "dec": self.dec_deg,
                "simbad": None, "error": None}
        try:
            from astroquery.simbad import Simbad
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            s = Simbad()
            s.reset_votable_fields()
            s.add_votable_fields("otype", "flux(V)", "flux(B)", "rvz_redshift",
                                 "rvz_radvel", "sptype", "dim")

            clean_name = " ".join(self.name.split())

            name_variants = [clean_name, clean_name.replace(" ", "")]
            name_lower = clean_name.lower()
            if name_lower.startswith("abell"):
                num = clean_name.split(None, 1)[1].strip() if " " in clean_name else clean_name[5:].strip()
                for prefix in ("ACO", "aco", "Abell"):
                    name_variants.append(f"{prefix} {num}")
                    name_variants.append(f"{prefix}{num}")

            seen = set()
            name_variants = [v for v in name_variants if not (v in seen or seen.add(v))]

            tbl = None
            for variant in name_variants:
                try:
                    tbl = s.query_object(variant)
                    if tbl is not None and len(tbl) > 0:
                        break
                    tbl = None
                except Exception:
                    tbl = None

            # TAP ADQL fallback — handles ACO/Abell and other catalogs query_object misses
            if tbl is None or len(tbl) == 0:
                try:
                    from astroquery.utils.tap.core import TapPlus
                    tap = TapPlus(url="https://simbad.cds.unistra.fr/simbad/sim-tap")
                    for variant in name_variants:
                        adql = f"""
                            SELECT b.main_id, b.ra, b.dec, b.otype, b.rvz_radvel,
                                   b.rvz_redshift, b.sp_type,
                                   f1.flux as flux_v, f2.flux as flux_b
                            FROM basic b
                            LEFT JOIN flux f1 ON f1.oidref = b.oid AND f1.filter = 'V'
                            LEFT JOIN flux f2 ON f2.oidref = b.oid AND f2.filter = 'B'
                            JOIN ident i ON i.oidref = b.oid
                            WHERE i.id = '{variant}'
                        """
                        result = tap.launch_job(adql).get_results()
                        if result is not None and len(result) > 0:
                            tbl = result
                            break
                except Exception:
                    pass

            if tbl is not None and len(tbl) > 0:
                row = tbl[0]
                def _val(*cols):
                    for col in cols:
                        if col in tbl.colnames:
                            try:
                                v = row[col]
                                s2 = v.decode() if isinstance(v, bytes) else str(v)
                                if s2 not in ("--", "nan", "", "None", "masked"):
                                    return s2
                            except Exception:
                                pass
                    return "—"
                info["simbad"] = {
                    "main_id":  _val("main_id", "MAIN_ID"),
                    "otype":    _val("otype", "OTYPE"),
                    "vmag":     _val("flux_v", "V", "FLUX_V"),
                    "bmag":     _val("flux_b", "B", "FLUX_B"),
                    "rv":       _val("rvz_radvel", "RVZ_RADVEL"),
                    "redshift": _val("rvz_redshift", "RVZ_REDSHIFT"),
                    "sp_type":  _val("sp_type", "SP_TYPE"),
                    "maj_axis": _val("galdim_majaxis"),
                    "min_axis": _val("galdim_minaxis"),
                }
            else:
                info["simbad"] = {}
        except Exception as e:
            info["error"] = str(e)
        self.result_ready.emit(info)

# ---------------------------------------------------
#  Altitude / visibility dialog
# ---------------------------------------------------
def _compute_alt_curve(ra_deg, dec_deg, lat, lon, date_str, tz_name):
    local_tz = pytz.timezone(tz_name)
    naive    = datetime.strptime(date_str, "%Y-%m-%d")
    # Start at noon on the observing date
    t_start  = Time(local_tz.localize(
        datetime(naive.year, naive.month, naive.day, 12, 0)))
    loc      = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=0 * u.m)

    n     = 288  # 5-minute resolution
    times = t_start + np.linspace(0, 24, n) * u.hour
    frame = AltAz(obstime=times, location=loc)

    obj_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    obj_alts  = obj_coord.transform_to(frame).alt.deg
    sun_alts  = get_sun(times).transform_to(frame).alt.deg
    moon_alts = get_body("moon", times, loc).transform_to(frame).alt.deg

    # Use cumulative hours from noon (12, 13, ... 24, 25 ... 36)
    # so the plot runs noon→midnight→noon without wrapping
    local_hours = 12.0 + np.linspace(0, 24, n)

    return times, local_hours, obj_alts, sun_alts, moon_alts, loc

class ObjectVisibilityDialog(QDialog):
    def __init__(self, item_data: dict, observer: dict, parent=None):
        super().__init__(parent)
        self.item_data = item_data
        self.observer  = observer
        name = item_data.get("name", "Object")
        self.setWindowTitle(f"Visibility — {name}")
        self.resize(980, 640)
        self._detail_thread = None
        self._build_ui()
        self._compute_and_plot()
        self._fetch_simbad()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        # ── Tab 1: Altitude / Visibility plot ─────────────────────────────
        self.plot_tab = QWidget()
        plot_lay = QVBoxLayout(self.plot_tab)
        plot_lay.setContentsMargins(4, 4, 4, 4)

        if _HAS_PG:
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setLabel("bottom", "Local Time (h)")
            self.plot_widget.setLabel("left", "Altitude (°)")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setYRange(-10, 90)
            self.plot_widget.addLine(y=0,  pen=pg.mkPen("w", width=1, style=Qt.PenStyle.DashLine))
            self.plot_widget.addLine(y=30, pen=pg.mkPen("g", width=1, style=Qt.PenStyle.DotLine))
            plot_lay.addWidget(self.plot_widget)
        else:
            plot_lay.addWidget(QLabel("pyqtgraph not installed — plot unavailable."))

        self.summary_lbl = QLabel("")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setStyleSheet("font-size: 12px; padding: 4px;")
        plot_lay.addWidget(self.summary_lbl, 0)

        self.tabs.addTab(self.plot_tab, "Altitude & Visibility")

        # ── Tab 2: Rise / Transit / Set ───────────────────────────────────
        self.rts_tab = QWidget()
        rts_lay = QVBoxLayout(self.rts_tab)
        self.rts_text = QTextEdit()
        self.rts_text.setReadOnly(True)
        self.rts_text.setStyleSheet("font-size: 13px;")
        rts_lay.addWidget(self.rts_text)
        self.tabs.addTab(self.rts_tab, "Rise / Transit / Set")

        # ── Tab 3: SIMBAD Details ─────────────────────────────────────────
        self.simbad_tab = QWidget()
        simbad_lay = QVBoxLayout(self.simbad_tab)
        self.simbad_text = QTextEdit()
        self.simbad_text.setReadOnly(True)
        self.simbad_text.setStyleSheet("font-size: 13px;")
        self.simbad_text.setPlainText("Querying SIMBAD…")
        simbad_lay.addWidget(self.simbad_text)
        self.tabs.addTab(self.simbad_tab, "SIMBAD Details")

        # ── Tab 4: Planet Separations ─────────────────────────────────────
        self.planet_tab = QWidget()
        planet_lay = QVBoxLayout(self.planet_tab)
        self.planet_text = QTextEdit()
        self.planet_text.setReadOnly(True)
        self.planet_text.setStyleSheet("font-size: 13px;")
        planet_lay.addWidget(self.planet_text)
        self.tabs.addTab(self.planet_tab, "Planet Separations")

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        open_aladin = QPushButton("Open in Aladin")
        open_aladin.clicked.connect(self._open_aladin)
        open_astrobin = QPushButton("Search AstroBin")
        open_astrobin.clicked.connect(self._open_astrobin)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        for b in (open_aladin, open_astrobin, close_btn):
            btn_row.addWidget(b)
        outer.addLayout(btn_row, 0)

    def _compute_and_plot(self):
        ra  = self.item_data["ra"]
        dec = self.item_data["dec"]
        lat = self.observer["lat"]
        lon = self.observer["lon"]
        tz  = self.observer["tz"]
        date_str = self.observer["date"]
        name = self.item_data.get("name", "Object")

        try:
            times, hrs, obj_alts, sun_alts, moon_alts, loc = \
                _compute_alt_curve(ra, dec, lat, lon, date_str, tz)
        except Exception as e:
            self.summary_lbl.setText(f"Could not compute visibility: {e}")
            return

        # ── Plot ──────────────────────────────────────────────────────────
        if _HAS_PG:
            pw = self.plot_widget
            pw.setXRange(12, 36, padding=0)

            # Custom tick labels: 12→23 then 0→12
            ticks = [(h, str(h if h <= 23 else h - 24))
                     for h in range(12, 37, 2)]
            pw.getAxis("bottom").setTicks([ticks])

            pw.setBackground(pg.mkColor(40, 55, 120, 255))

            # Find sun altitude crossing points and draw clean bands
            def _sun_x_at_threshold(threshold):
                """Find all x values where sun_alts crosses threshold."""
                crossings = []
                for i in range(len(sun_alts) - 1):
                    if (sun_alts[i] - threshold) * (sun_alts[i+1] - threshold) < 0:
                        # Linear interpolation
                        t = (threshold - sun_alts[i]) / (sun_alts[i+1] - sun_alts[i])
                        crossings.append(hrs[i] + t * (hrs[i+1] - hrs[i]))
                return crossings

            def _fill_region(x0, x1, color, alpha):
                region = pg.LinearRegionItem(
                    values=(x0, x1),
                    orientation='vertical',
                    brush=pg.mkBrush(*color, alpha),
                    pen=pg.mkPen(None),
                    movable=False,
                )
                pw.addItem(region)

            # Get crossing x positions for each twilight boundary
            civil_x      = _sun_x_at_threshold(-6)
            nautical_x   = _sun_x_at_threshold(-12)
            astro_x      = _sun_x_at_threshold(-18)

            x_min, x_max = float(hrs[0]), float(hrs[-1])

            # Night zone: between astro crossings (middle of plot)
            if len(astro_x) >= 2:
                _fill_region(astro_x[0],    astro_x[1],    (0,   0,  15), 200)
            elif len(astro_x) == 1:
                _fill_region(astro_x[0],    x_max,         (0,   0,  15), 200)

            # Nautical twilight bands
            if len(nautical_x) >= 2 and len(astro_x) >= 2:
                _fill_region(nautical_x[0], astro_x[0],    (10, 20,  60), 160)
                _fill_region(astro_x[1],    nautical_x[1], (10, 20,  60), 160)

            # Civil twilight bands
            if len(civil_x) >= 2 and len(nautical_x) >= 2:
                _fill_region(civil_x[0],    nautical_x[0], (20, 35,  90), 120)
                _fill_region(nautical_x[1], civil_x[1],    (20, 35,  90), 120)
            # Object altitude
            pw.plot(hrs, obj_alts,
                    pen=pg.mkPen("y", width=2.5),
                    name=name)

            # Moon altitude (dashed grey)
            pw.plot(hrs, moon_alts,
                    pen=pg.mkPen((180, 180, 180), width=1.2,
                                 style=Qt.PenStyle.DashLine),
                    name="Moon")

            # Horizon and 30° guide lines already added in _build_ui

            # Mark current observing time
            local_tz  = pytz.timezone(tz)
            naive     = datetime.strptime(
                f"{date_str} {self.observer.get('time','12:00')}",
                "%Y-%m-%d %H:%M")
            obs_local = local_tz.localize(naive)
            obs_hr = obs_local.hour + obs_local.minute / 60.0
            # Shift into noon-to-noon space
            if obs_hr < 12:
                obs_hr += 24
            pw.addLine(x=obs_hr, pen=pg.mkPen("r", width=1.5,
                                               style=Qt.PenStyle.DashLine))

            pw.addLegend()

        # ── Rise / Transit / Set tab ──────────────────────────────────────
        local_tz = pytz.timezone(tz)

        def _to_local(t):
            return (t.to_datetime(timezone=pytz.utc)
                    .astimezone(local_tz).strftime("%H:%M"))

        # Object crossings
        obj_cross = np.where(np.diff(np.sign(obj_alts)))[0]
        obj_rise = obj_set = "—"
        for ci in obj_cross:
            rising = obj_alts[ci + 1] > obj_alts[ci]
            ts = _to_local(times[ci])
            if rising and obj_rise == "—":
                obj_rise = ts
            elif not rising and obj_set == "—":
                obj_set = ts

        # Handle circumpolar / never-rises cases
        if obj_rise == "—" and obj_set == "—":
            if obj_alts.min() > 0:
                obj_rise = obj_set = "Circumpolar"
            else:
                obj_rise = obj_set = "Below horizon"
        peak_idx    = int(np.argmax(obj_alts))
        obj_transit = _to_local(times[peak_idx])
        peak_alt    = float(obj_alts[peak_idx])

        # Imaging window: above 20° during astronomical night
        imaging_mask = (obj_alts >= 20) & (sun_alts < -18)
        imaging_hrs  = float(np.sum(imaging_mask)) * (24.0 / len(times))

        # Moon interference: moon above horizon during imaging window
        moon_up_during_imaging = (imaging_mask & (moon_alts > 0))
        moon_interference_hrs  = float(np.sum(moon_up_during_imaging)) * (24.0 / len(times))

        rts_lines = [
            f"Object:  {name}",
            f"Date:    {date_str}",
            f"",
            f"Rise:          {obj_rise}",
            f"Transit:       {obj_transit}  (peak alt {peak_alt:.1f}°)",
            f"Set:           {obj_set}",
            f"",
            f"Imaging window (>20°, astro night):  {imaging_hrs:.1f} h",
            f"Moon above horizon during window:    {moon_interference_hrs:.1f} h",
            f"",
        ]

        # Twilight start/end
        astro_night = np.where(sun_alts < -18)[0]
        if len(astro_night):
            rts_lines.append(
                f"Astronomical night: "
                f"{_to_local(times[astro_night[0]])} → "
                f"{_to_local(times[astro_night[-1]])}")
        nautical = np.where(sun_alts < -12)[0]
        if len(nautical):
            rts_lines.append(
                f"Nautical twilight:  "
                f"{_to_local(times[nautical[0]])} → "
                f"{_to_local(times[nautical[-1]])}")

        self.rts_text.setPlainText("\n".join(rts_lines))

        # ── Summary label under plot ──────────────────────────────────────
        self.summary_lbl.setText(
            f"Rise {obj_rise}  •  Transit {obj_transit} ({peak_alt:.0f}°)  •  "
            f"Set {obj_set}  •  Imaging window: {imaging_hrs:.1f} h  •  "
            f"Moon interference: {moon_interference_hrs:.1f} h"
        )

        # ── Planet separations ────────────────────────────────────────────
        self._compute_planet_separations(loc, times, ra, dec)

    def _compute_planet_separations(self, loc, times, ra_deg, dec_deg):
        planets = ["mars", "jupiter", "saturn", "venus", "mercury", "uranus", "neptune"]
        obj_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        t_mid = times[len(times) // 2]
        lines = [f"Separations at midnight ({t_mid.iso[:10]}):\n"]
        for planet in planets:
            try:
                pb = get_body(planet, t_mid, loc)
                sep = float(obj_coord.separation(
                    SkyCoord(ra=pb.ra.deg * u.deg, dec=pb.dec.deg * u.deg)).deg)
                warn = "  ⚠ close!" if sep < 5 else ""
                lines.append(f"  {planet.capitalize():10s}  {sep:6.2f}°{warn}")
            except Exception:
                pass
        self.planet_text.setPlainText("\n".join(lines))

    def _fetch_simbad(self):
        ra  = self.item_data["ra"]
        dec = self.item_data["dec"]
        name = self.item_data.get("name", "")
        self._detail_thread = ObjectDetailThread(name, ra, dec)
        self._detail_thread.result_ready.connect(self._on_simbad_ready)
        self._detail_thread.start()

    def _on_simbad_ready(self, info: dict):
        if info.get("error"):
            self.simbad_text.setPlainText(f"SIMBAD error: {info['error']}")
            return
        s = info.get("simbad", {})
        tried = info.get("_debug_tried", [])
        clean = info.get("_debug_clean", "")
        if not s:
            orig = info.get('name', '')
            tried_str = "\n".join(f"  - '{v}'" for v in tried)
            self.simbad_text.setPlainText(
                f"No SIMBAD match found.\n\n"
                f"Original name: '{orig}'\n"
                f"Cleaned name:  '{clean}'\n"
                f"Variants tried:\n{tried_str}"
            )
            return

        # Build angular size string if available
        maj = s.get("maj_axis", "—")
        mn  = s.get("min_axis", "—")
        size_str = f"{maj}′ × {mn}′" if maj != "—" else "—"

        lines = [
            f"Main ID:       {s.get('main_id','—')}",
            f"Object type:   {s.get('otype','—')}",
            f"",
            f"V magnitude:   {s.get('vmag','—')}",
            f"B magnitude:   {s.get('bmag','—')}",
            f"Spectral type: {s.get('sp_type','—')}",
            f"Angular size:  {size_str}",
            f"",
            f"Radial vel.:   {s.get('rv','—')}  km/s",
            f"Redshift z:    {s.get('redshift','—')}",
            f"",
            f"RA:            {info.get('ra', 0):.5f}°",
            f"Dec:           {info.get('dec', 0):.5f}°",
        ]
        self.simbad_text.setPlainText("\n".join(lines))
        self.tabs.setTabText(2, f"SIMBAD — {s.get('main_id','')}")

    def _open_aladin(self):
        ra  = self.item_data["ra"]
        dec = self.item_data["dec"]
        webbrowser.open(
            f"https://aladin.cds.unistra.fr/AladinLite/"
            f"?target={ra:.5f}+{dec:+.5f}&fov=0.5&survey=P/DSS2/color")

    def _open_astrobin(self):
        name = self.item_data.get("name", "").replace(" ", "")
        webbrowser.open(f"https://www.astrobin.com/search/?q={name}")


# ---------------------------------------------------
#  Sortable tree item
# ---------------------------------------------------
class SortableTreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        if col in [3, 4, 5, 7, 10]:
            try:
                return float(self.text(col)) < float(other.text(col))
            except ValueError:
                pass
        return self.text(col) < other.text(col)


# ---------------------------------------------------
#  Coordinate helpers
# ---------------------------------------------------
def _parse_deg_with_suffix(txt: str, kind: str) -> float:
    if txt is None:
        raise ValueError("empty")
    t = str(txt).strip().replace("°", "")
    if not t:
        raise ValueError("empty")
    suffix = ""
    if t[-1].upper() in ("N", "S", "E", "W"):
        suffix = t[-1].upper()
        t = t[:-1].strip()
    val = float(t)
    if suffix:
        if kind == "lat":
            val = abs(val) if suffix == "N" else -abs(val)
        elif kind == "lon":
            val = abs(val) if suffix == "E" else -abs(val)
    if kind == "lat" and not (-90 <= val <= 90):
        raise ValueError("Latitude must be in [-90, 90]")
    if kind == "lon" and not (-180 <= val <= 180):
        raise ValueError("Longitude must be in [-180, 180]")
    return val


def _format_with_suffix(val: float, kind: str) -> str:
    v = float(val)
    hemi = ("N" if v >= 0 else "S") if kind == "lat" else ("E" if v >= 0 else "W")
    return f"{abs(v):g}{hemi}"


def _tz_vs_longitude_hint(tz_name, date_str, time_str, lon_deg):
    try:
        local_tz = pytz.timezone(tz_name)
        naive    = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        local_dt = local_tz.localize(naive)
        off_hours = (local_dt.utcoffset() or pd.Timedelta(0)).total_seconds() / 3600.0
    except Exception:
        return (False, "", "", 0.0)
    hours = int(off_hours)
    mins  = int(round(abs(off_hours - hours) * 60))
    sign  = "−" if off_hours < 0 else "+"
    utc_str = f"UTC{sign}{abs(hours)}:{mins:02d}" if mins else f"UTC{sign}{abs(hours)}"
    central = off_hours * 15.0
    sign_ok = (abs(off_hours) < 1e-9) or (lon_deg == 0) or ((lon_deg > 0) == (off_hours > 0))
    far     = abs(abs(lon_deg) - abs(central)) > 45.0
    if (not sign_ok) or far:
        msg = (f"Timezone {tz_name} ({utc_str}) looks inconsistent with longitude "
               f"{abs(lon_deg):g}{'E' if lon_deg>0 else 'W'} "
               f"(central meridian ≈ {abs(central):.0f}°{'E' if central>0 else 'W'}).")
        return (True, msg, utc_str, central)
    return (False, "", utc_str, central)


# ---------------------------------------------------
#  Main dialog
# ---------------------------------------------------
class WhatsInMySkyDialog(QDialog):
    def __init__(self, parent=None, wims_path=None, wrench_path=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("What's In My Sky"))
        if wims_path:
            self.setWindowIcon(QIcon(wims_path))

        self.settings     = QSettings()
        self.object_limit = int(self.settings.value("object_limit", 100, int))
        self._observer    = {}   # populated in start_calculation

        self._build_ui(wrench_path)
        self._load_settings_into_ui()

        self.calc_thread: Optional[CalculationThread] = None
        self.catalog_file: Optional[str] = None

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self, wrench_path):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left panel ────────────────────────────────────────────────────
        left_outer = QFrame()
        left_outer.setFixedWidth(280)
        left_outer.setFrameShape(QFrame.Shape.NoFrame)
        left_vbox = QVBoxLayout(left_outer)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        form_widget = QWidget()
        form = QVBoxLayout(form_widget)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(0)

        def _section(title):
            lbl = QLabel(title.upper())
            lbl.setStyleSheet(
                "font-size: 10px; font-weight: 500; color: palette(window-text);"
                "letter-spacing: 0.05em; margin-top: 10px; margin-bottom: 4px;")
            return lbl

        def _field(label_text, widget):
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setContentsMargins(0, 0, 0, 6)
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 11px; color: palette(window-text);")
            col.addWidget(lbl)
            col.addWidget(widget)
            return col

        form.addWidget(_section("Location & Time"))
        fixed_w = 240
        self.latitude_entry     = QLineEdit(); self.latitude_entry.setFixedWidth(fixed_w)
        self.longitude_entry    = QLineEdit(); self.longitude_entry.setFixedWidth(fixed_w)
        self.date_entry         = QLineEdit(); self.date_entry.setFixedWidth(fixed_w)
        self.time_entry         = QLineEdit(); self.time_entry.setFixedWidth(fixed_w)
        self.timezone_combo     = QComboBox(); self.timezone_combo.addItems(pytz.all_timezones)
        self.timezone_combo.setFixedWidth(fixed_w)
        self.min_altitude_entry = QLineEdit(); self.min_altitude_entry.setFixedWidth(fixed_w)

        for lbl, w in [("Latitude", self.latitude_entry),
                        ("Longitude (E+, W−)", self.longitude_entry),
                        ("Date (YYYY-MM-DD)", self.date_entry),
                        ("Time (HH:MM)", self.time_entry),
                        ("Time Zone", self.timezone_combo),
                        ("Min Altitude (°)", self.min_altitude_entry)]:
            form.addLayout(_field(lbl, w))

        form.addWidget(_section("Catalogs"))
        cat_grid = QWidget()
        cat_lay  = QGridLayout(cat_grid)
        cat_lay.setContentsMargins(0, 0, 0, 0)
        cat_lay.setHorizontalSpacing(4)
        cat_lay.setVerticalSpacing(2)
        self.catalog_vars: dict[str, QCheckBox] = {}
        for i, name in enumerate(["Messier","NGC","IC","Caldwell","Abell",
                                   "Sharpless","LBN","LDN","PNG","User"]):
            cb = QCheckBox(name)
            cat_lay.addWidget(cb, i // 2, i % 2)
            self.catalog_vars[name] = cb
        form.addWidget(cat_grid)

        form.addWidget(_section("RA/Dec Format"))
        self.ra_dec_degrees = QRadioButton("Degrees")
        self.ra_dec_hms     = QRadioButton("H:M:S / D:M:S")
        self.ra_dec_degrees.setChecked(True)
        g = QButtonGroup(self)
        g.addButton(self.ra_dec_degrees); g.addButton(self.ra_dec_hms)
        ra_row = QHBoxLayout()
        ra_row.setContentsMargins(0, 2, 0, 6)
        ra_row.addWidget(self.ra_dec_degrees)
        ra_row.addWidget(self.ra_dec_hms)
        ra_row.addStretch()
        form.addLayout(ra_row)
        self.ra_dec_degrees.toggled.connect(self.update_ra_dec_format)
        self.ra_dec_hms.toggled.connect(self.update_ra_dec_format)

        form.addWidget(_section("Sidereal Time"))
        self.lst_label = QLabel("Local Sidereal Time: —")
        self.lst_label.setStyleSheet(
            "font-size: 12px; color: palette(window-text); margin-bottom: 6px;")
        self.lst_label.setWordWrap(True)
        form.addWidget(self.lst_label)
        form.addStretch(1)

        scroll.setWidget(form_widget)
        left_vbox.addWidget(scroll, 1)

        calc_btn = QPushButton("Calculate")
        calc_btn.setFixedHeight(36)
        calc_btn.clicked.connect(self.start_calculation)
        calc_btn.setStyleSheet("margin: 6px 12px;")
        left_vbox.addWidget(calc_btn)

        moon_frame = QFrame()
        moon_frame.setFrameShape(QFrame.Shape.NoFrame)
        moon_frame.setStyleSheet("border-top: 1px solid palette(shadow);")
        moon_hbox = QHBoxLayout(moon_frame)
        moon_hbox.setContentsMargins(12, 8, 12, 10)
        moon_hbox.setSpacing(10)
        self.lunar_phase_image_label = QLabel()
        self.lunar_phase_image_label.setFixedSize(100, 100)
        self.lunar_phase_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        moon_hbox.addWidget(self.lunar_phase_image_label, 0)
        moon_text = QVBoxLayout()
        moon_text.setSpacing(2)
        self.lunar_phase_label = QLabel("Lunar Phase: N/A")
        self.lunar_phase_label.setStyleSheet("font-size: 12px; font-weight: 500;")
        self.lunar_rts_label = QLabel("")
        self.lunar_rts_label.setWordWrap(True)
        self.lunar_rts_label.setStyleSheet(
            "font-size: 11px; color: palette(window-text);")
        moon_text.addWidget(self.lunar_phase_label)
        moon_text.addWidget(self.lunar_rts_label)
        moon_hbox.addLayout(moon_text, 1)
        left_vbox.addWidget(moon_frame, 0)

        # ── Right panel ───────────────────────────────────────────────────
        right_outer = QWidget()
        right_vbox  = QVBoxLayout(right_outer)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(0)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(10, 6, 10, 6)
        toolbar.setSpacing(8)
        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet(
            "font-size: 11px; color: palette(window-text);")
        toolbar.addWidget(self.status_label, 1)
        add_btn      = QPushButton("Add Object…"); add_btn.clicked.connect(self.add_custom_object)
        save_btn     = QPushButton("Save CSV…");   save_btn.clicked.connect(self.save_to_csv)
        settings_btn = QPushButton()
        if wrench_path and os.path.exists(wrench_path):
            settings_btn.setIcon(QIcon(wrench_path))
        else:
            settings_btn.setText("⚙")
        settings_btn.clicked.connect(self.open_settings)
        for b in (add_btn, save_btn, settings_btn):
            b.setFixedHeight(28); toolbar.addWidget(b)
        toolbar_frame = QFrame()
        toolbar_frame.setFrameShape(QFrame.Shape.NoFrame)
        toolbar_frame.setStyleSheet("border-bottom: 1px solid palette(shadow);")
        toolbar_frame.setLayout(toolbar)
        right_vbox.addWidget(toolbar_frame, 0)
        hint_lbl = QLabel("Tip: right-click any object for visibility plot, SIMBAD details, planet separations and more.")
        hint_lbl.setStyleSheet("font-size: 10px; color: palette(window-text); padding: 2px 10px;")
        hint_lbl.setWordWrap(True)
        right_vbox.addWidget(hint_lbl, 0)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([
            "Name","RA","Dec","Altitude","Azimuth",
            "Min→Transit","B/A Transit","°from Moon",
            "Alt Name","Type","Magnitude","Size (arcmin)"])
        self.tree.setSortingEnabled(True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(True)
        hdr.setMinimumSectionSize(50)
        self.tree.sortByColumn(5, Qt.SortOrder.AscendingOrder)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setFrameShape(QFrame.Shape.NoFrame)
        for col, w in enumerate([90,65,65,55,55,70,65,70,90,100,50,80]):
            self.tree.setColumnWidth(col, w)

        # Double-click → AstroBin, Right-click → context menu
        self.tree.itemDoubleClicked.connect(self.on_row_double_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)

        right_vbox.addWidget(self.tree, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_outer)
        splitter.addWidget(right_outer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        root.addWidget(splitter)
        self.resize(1100, 620)

    # ── Context menu ──────────────────────────────────────────────────────
    def _show_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)

        act_visibility = QAction("📈  Altitude / Visibility Plot…", self)
        act_simbad     = QAction("🔭  SIMBAD Details…", self)
        act_planets    = QAction("🪐  Planet Separations…", self)
        act_aladin     = QAction("🗺   Open in Aladin…", self)
        act_astrobin   = QAction("🖼   Search AstroBin…", self)

        menu.addAction(act_visibility)
        menu.addAction(act_simbad)
        menu.addAction(act_planets)
        menu.addSeparator()
        menu.addAction(act_aladin)
        menu.addAction(act_astrobin)

        act_visibility.triggered.connect(lambda: self._open_visibility_dialog(item))
        act_simbad.triggered.connect(lambda: self._open_visibility_dialog(item, tab=2))
        act_planets.triggered.connect(lambda: self._open_visibility_dialog(item, tab=3))
        act_aladin.triggered.connect(lambda: self._open_aladin_for(item))
        act_astrobin.triggered.connect(lambda: self.on_row_double_click(item, 0))

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _item_data(self, item):
        """Extract name, ra, dec from a tree item."""
        name = item.text(0)
        try:
            ra_txt  = item.text(1)
            dec_txt = item.text(2)
            if ":" in ra_txt:
                sc = SkyCoord(ra=ra_txt, dec=dec_txt, unit=(u.hourangle, u.deg))
            else:
                sc = SkyCoord(ra=float(ra_txt) * u.deg, dec=float(dec_txt) * u.deg)
            return {"name": name, "ra": float(sc.ra.deg), "dec": float(sc.dec.deg)}
        except Exception:
            return None

    def _open_visibility_dialog(self, item, tab=0):
        if not self._observer:
            QMessageBox.information(self, "WIMS",
                "Run Calculate first so the observer location is known.")
            return
        data = self._item_data(item)
        if data is None:
            QMessageBox.warning(self, "WIMS", "Could not parse RA/Dec for this object.")
            return
        dlg = ObjectVisibilityDialog(data, self._observer, parent=self)
        dlg.tabs.setCurrentIndex(tab)
        dlg.show()

    def _open_aladin_for(self, item):
        data = self._item_data(item)
        if data:
            webbrowser.open(
                f"https://aladin.cds.unistra.fr/AladinLite/"
                f"?target={data['ra']:.5f}+{data['dec']:+.5f}&fov=0.5&survey=P/DSS2/color")

    # ── Settings ──────────────────────────────────────────────────────────
    def _load_settings_into_ui(self):
        def cast(v, typ, default):
            try: return typ(v)
            except Exception: return default
        self.latitude_entry.setText(str(cast(self.settings.value("latitude",  0.0), float, 0.0)))
        self.longitude_entry.setText(str(cast(self.settings.value("longitude", 0.0), float, 0.0)))
        self.date_entry.setText(
            self.settings.value("date", datetime.now().strftime("%Y-%m-%d")))
        self.time_entry.setText(self.settings.value("time", "00:00"))
        self.timezone_combo.setCurrentText(self.settings.value("timezone", "UTC"))
        self.min_altitude_entry.setText(
            str(cast(self.settings.value("min_altitude", 0.0), float, 0.0)))
        self.object_limit = cast(self.settings.value("object_limit", 100), int, 100)

    def _save_settings(self, lat, lon, date, time, tz, min_alt):
        for k, v in [("latitude", lat), ("longitude", lon), ("date", date),
                     ("time", time), ("timezone", tz), ("min_altitude", min_alt)]:
            self.settings.setValue(k, v)

    # ── Calculation ───────────────────────────────────────────────────────
    def start_calculation(self):
        try:
            orig_lat = self.latitude_entry.text()
            orig_lon = self.longitude_entry.text()
            latitude  = _parse_deg_with_suffix(orig_lat, "lat")
            longitude = _parse_deg_with_suffix(orig_lon, "lon")
            self.latitude_entry.setText(_format_with_suffix(latitude,  "lat"))
            self.longitude_entry.setText(_format_with_suffix(longitude, "lon"))
            date_str = self.date_entry.text().strip()
            time_str = self.time_entry.text().strip()
            tz_str   = self.timezone_combo.currentText()
            min_alt  = float(self.min_altitude_entry.text())
        except ValueError as e:
            self.update_status(f"Invalid input: {e}"); return

        warn, msg, _, central = _tz_vs_longitude_hint(tz_str, date_str, time_str, longitude)
        if warn:
            bare = orig_lon.strip() and orig_lon.strip()[-1].upper() not in ("E","W")
            sign_mismatch = not ((longitude > 0) == (central > 0)
                                 or abs(central) < 1e-6 or longitude == 0)
            if bare and sign_mismatch:
                longitude = -longitude
                self.longitude_entry.setText(_format_with_suffix(longitude, "lon"))
                self.update_status(f"{msg} → Auto-corrected to {_format_with_suffix(longitude,'lon')}.")
            else:
                self.update_status(msg + " Please verify your longitude/timezone.")
        else:
            self.update_status("Inputs look consistent.")

        self._save_settings(latitude, longitude, date_str, time_str, tz_str, min_alt)

        # Store observer context for visibility dialogs
        self._observer = {
            "lat": latitude, "lon": longitude,
            "date": date_str, "time": time_str, "tz": tz_str,
            "min_alt": min_alt,
        }

        catalogs = [n for n, cb in self.catalog_vars.items() if cb.isChecked()]
        self.calc_thread = CalculationThread(
            latitude, longitude, date_str, time_str, tz_str,
            min_alt, catalogs, self.object_limit)
        self.catalog_file = self.calc_thread.catalog_file
        self.calc_thread.calculation_complete.connect(self.on_calculation_complete)
        self.calc_thread.lunar_phase_calculated.connect(self.update_lunar_phase)
        self.calc_thread.lst_calculated.connect(self.update_lst)
        self.calc_thread.status_update.connect(self.update_status)
        self.update_status("Calculating…")
        self.calc_thread.start()

    # ── Slots ─────────────────────────────────────────────────────────────
    def update_lunar_phase(self, phase_percentage, phase_image_name, rts):
        self.lunar_phase_label.setText(f"Lunar Phase: {phase_percentage}% illuminated")
        self.lunar_rts_label.setText(rts)
        pth = get_icon_path(phase_image_name)
        if os.path.exists(pth):
            pm = QPixmap(pth).scaled(100, 100,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.lunar_phase_image_label.setPixmap(pm)
        else:
            self.lunar_phase_image_label.clear()

    def on_calculation_complete(self, df, message):
        self.update_status(message)
        self.tree.clear()
        if df.empty:
            return
        for _, row in df.iterrows():
            ra_disp, dec_disp = row["RA"], row["Dec"]
            if self.ra_dec_hms.isChecked():
                sc = SkyCoord(ra=row["RA"] * u.deg, dec=row["Dec"] * u.deg)
                ra_disp  = sc.ra.to_string(unit=u.hour, sep=":")
                dec_disp = sc.dec.to_string(unit=u.deg, sep=":")
            size_arcmin = str(row.get("Info","")) if pd.notna(row.get("Info","")) else ""
            vals = [
                str(row.get("Name","") or ""),
                str(ra_disp), str(dec_disp),
                str(row.get("Altitude","")), str(row.get("Azimuth","")),
                str(int(row.get("Minutes to Transit",0)))
                    if pd.notna(row.get("Minutes to Transit", np.nan)) else "",
                str(row.get("Before/After Transit","")),
                str(round(row.get("Degrees from Moon",0.0),2))
                    if pd.notna(row.get("Degrees from Moon", np.nan)) else "",
                row.get("Alt Name","") if pd.notna(row.get("Alt Name","")) else "",
                row.get("Type","")     if pd.notna(row.get("Type",""))     else "",
                str(row.get("Magnitude","")) if pd.notna(row.get("Magnitude","")) else "",
                size_arcmin,
            ]
            self.tree.addTopLevelItem(SortableTreeWidgetItem(vals))

    def update_status(self, msg):
        self.status_label.setText(f"Status: {msg}")

    def update_lst(self, msg):
        self.lst_label.setText(msg)

    def open_settings(self):
        n, ok = QInputDialog.getInt(self, "Settings",
            "Enter number of objects to display:",
            value=self.object_limit, min=1, max=1000)
        if ok:
            self.object_limit = n
            self.settings.setValue("object_limit", n)

    def on_row_double_click(self, item, column):
        name = item.text(0).replace(" ", "")
        webbrowser.open(f"https://www.astrobin.com/search/?q={name}")

    def add_custom_object(self):
        name, ok = QInputDialog.getText(self, "Add Custom Object", "Enter object name:")
        if not ok or not name: return
        ra,  ok = QInputDialog.getDouble(self, "Add Custom Object", "Enter RA (deg):",  decimals=3)
        if not ok: return
        dec, ok = QInputDialog.getDouble(self, "Add Custom Object", "Enter Dec (deg):", decimals=3)
        if not ok: return
        entry = {"Name": name, "RA": ra, "Dec": dec, "Catalog": "User",
                 "Alt Name": "User Defined", "Type": "Custom",
                 "Magnitude": "", "Info": ""}
        csv = self.catalog_file or os.path.join(os.path.expanduser("~"),
                                                  "celestial_catalog.csv")
        try:
            df = pd.read_csv(csv, encoding="ISO-8859-1") if os.path.exists(csv) \
                 else pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
            df.to_csv(csv, index=False, encoding="ISO-8859-1")
            self.update_status(f"Added custom object: {name}")
        except Exception as e:
            QMessageBox.warning(self, "Add Custom Object",
                                f"Could not update catalog:\n{e}")

    def update_ra_dec_format(self):
        use_deg = self.ra_dec_degrees.isChecked()
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            ra_txt, dec_txt = it.text(1), it.text(2)
            try:
                if use_deg and ":" in ra_txt:
                    sc = SkyCoord(ra=ra_txt, dec=dec_txt, unit=(u.hourangle, u.deg))
                    it.setText(1, f"{sc.ra.deg:.3f}")
                    it.setText(2, f"{sc.dec.deg:.3f}")
                elif not use_deg and ":" not in ra_txt:
                    sc = SkyCoord(ra=float(ra_txt) * u.deg, dec=float(dec_txt) * u.deg)
                    it.setText(1, sc.ra.to_string(unit=u.hour, sep=":"))
                    it.setText(2, sc.dec.to_string(unit=u.deg,  sep=":"))
            except Exception:
                pass

    def save_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "",
                                               "CSV files (*.csv);;All Files (*)")
        if not path: return
        cols = [self.tree.headerItem().text(i) for i in range(self.tree.columnCount())]
        rows = [[self.tree.topLevelItem(i).text(j)
                 for j in range(self.tree.columnCount())]
                for i in range(self.tree.topLevelItemCount())]
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
        self.update_status(f"Data saved to {path}")