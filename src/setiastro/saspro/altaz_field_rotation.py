# ============================================================
#  SASpro — Alt/Az Field Rotation Calculator
#  Copyright (c) 2025 Franklin Marek / Seti Astro
#  www.setiastro.com
#
#    ___      _   _   _       _
#   / __| ___| |_(_) /_\  ___| |_ _ _ ___
#   \__ \/ -_)  _| |/ _ \(_-<|  _| '_/ _ \
#   |___/\___|\__|_/_/ \_/__/ \__|_| \___/
#
#  Alt/Az Field Rotation Calculator — standalone dialog
#  for use inside the What's In My Sky (WIMS) module.
#
#  Drop-in usage:
#    from setiastro.saspro.altaz_field_rotation import AltAzFieldRotationDialog
#    dlg = AltAzFieldRotationDialog(lat, lon, alt_deg, az_deg,
#                                   ra_deg, dec_deg,
#                                   date_str, tz_name,
#                                   settings, parent=self)
#    dlg.show()
# ============================================================

from __future__ import annotations

import math
import numpy as np

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QFrame, QWidget, QTabWidget,
)
from PyQt6.QtGui import QFont

try:
    import pyqtgraph as pg
    _HAS_PG = True
except ImportError:
    _HAS_PG = False


# ── Math core ────────────────────────────────────────────────────────────────

def field_rotation_rate_arcsec_per_sec(lat_deg: float, az_deg: float,
                                        alt_deg: float) -> float:
    """
    Field rotation rate in arcsec/sec for an Alt/Az mount.
    Formula: R = 15.04 * cos(lat) * cos(az) / cos(alt)
    Returns the signed rate. Returns inf near zenith (alt >= 89.99).
    """
    cos_alt = math.cos(math.radians(alt_deg))
    if abs(cos_alt) < 1e-9:
        return float("inf")
    return (15.04
            * math.cos(math.radians(lat_deg))
            * math.cos(math.radians(az_deg))
            / cos_alt)


def max_exposure_seconds(lat_deg: float, az_deg: float, alt_deg: float,
                          sensor_w_mm: float, sensor_h_mm: float,
                          pixel_pitch_um: float,
                          max_corner_pixels: float = 1.0) -> dict:
    """
    Maximum exposure before corner pixel motion exceeds max_corner_pixels.
    Returns dict with all intermediate values for display.
    """
    half_diag_mm = math.sqrt((sensor_w_mm / 2) ** 2 + (sensor_h_mm / 2) ** 2)
    half_diag_um = half_diag_mm * 1000.0

    rate_asec_sec = field_rotation_rate_arcsec_per_sec(lat_deg, az_deg, alt_deg)
    rate_mag      = abs(rate_asec_sec)
    rate_deg_sec  = rate_mag / 3600.0

    # t = max_pixels / ( (2pi/360) * (d/s) * rate_deg_sec )
    two_pi_over_360 = 2.0 * math.pi / 360.0
    d_over_s        = half_diag_um / pixel_pitch_um

    if rate_deg_sec < 1e-12:
        t_max = float("inf")
    else:
        t_max = max_corner_pixels / (two_pi_over_360 * d_over_s * rate_deg_sec)

    return {
        "rate_arcsec_per_sec": rate_asec_sec,
        "rate_arcsec_per_min": rate_asec_sec * 60.0,
        "rate_mag_asec_sec":   rate_mag,
        "half_diag_mm":        half_diag_mm,
        "half_diag_um":        half_diag_um,
        "pixel_pitch_um":      pixel_pitch_um,
        "d_over_s":            d_over_s,
        "max_corner_pixels":   max_corner_pixels,
        "t_max_sec":           t_max,
    }


def severity_badge(t_sec: float) -> str:
    if t_sec == float("inf"):
        return "⭐ No limit (E/W)"
    if t_sec >= 120:
        return "⭐ Excellent  (≥ 120 s)"
    if t_sec >= 60:
        return "🟢 Good  (60–120 s)"
    if t_sec >= 30:
        return "🟡 Marginal  (30–60 s)"
    if t_sec >= 10:
        return "🔴 Poor  (10–30 s)"
    return "❌ Very limited  (< 10 s)"


# ── Dialog ────────────────────────────────────────────────────────────────────

class AltAzFieldRotationDialog(QDialog):
    """
    Alt/Az Field Rotation Calculator.

    Parameters
    ----------
    lat         : float        Observer latitude  (degrees, north positive)
    lon         : float        Observer longitude (degrees, east positive)
    alt         : float|None   Target altitude at current time — pre-fill only
    az          : float|None   Target azimuth  at current time — pre-fill only
    ra_deg      : float|None   Target RA  (degrees) — drives the night curve
    dec_deg     : float|None   Target Dec (degrees) — drives the night curve
    date_str    : str|None     Observation date "YYYY-MM-DD" — drives the night curve
    tz_name     : str|None     Timezone name e.g. "US/Central"
    target_name : str|None     Display name shown in header and plot title
    settings    : QSettings
    parent      : QWidget
    """

    _SETTINGS_PREFIX = "wims/field_rotation/"

    def __init__(self, lat: float, lon: float,
                 alt:         float | None = None,
                 az:          float | None = None,
                 ra_deg:      float | None = None,
                 dec_deg:     float | None = None,
                 date_str:    str   | None = None,
                 tz_name:     str   | None = None,
                 target_name: str   | None = None,
                 settings:    QSettings | None = None,
                 parent=None):
        super().__init__(parent)
        name_part = f" — {target_name}" if target_name else ""
        self.setWindowTitle(f"Alt/Az Field Rotation Calculator{name_part}")
        self.resize(720, 600)

        self._lat         = lat
        self._lon         = lon
        self._ra_deg      = ra_deg
        self._dec_deg     = dec_deg
        self._date_str    = date_str
        self._tz_name     = tz_name
        self._target_name = target_name or ""
        self._settings    = settings or QSettings()

        self._build_ui()
        self._load_persisted()

        if alt is not None:
            self._alt_edit.setText(f"{alt:.1f}")
        if az is not None:
            self._az_edit.setText(f"{az:.1f}")

        if alt is not None and az is not None:
            self._calculate()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 10)
        outer.setSpacing(8)

        name_part = f"  —  <b>{self._target_name}</b>" if self._target_name else ""
        hdr = QLabel(
            f"<b>Alt/Az Field Rotation Calculator</b>{name_part}"
            f"<br><small>Observer latitude: <b>{self._lat:+.4f}°</b>  "
            f"longitude: <b>{self._lon:+.4f}°</b></small>"
        )
        hdr.setWordWrap(True)
        outer.addWidget(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

        input_box = QGroupBox("Target & Equipment")
        grid = QGridLayout(input_box)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        def _le(placeholder="", width=90):
            e = QLineEdit()
            e.setPlaceholderText(placeholder)
            e.setFixedWidth(width)
            e.returnPressed.connect(self._calculate)
            return e

        grid.addWidget(QLabel("Altitude (°):"),  0, 0, Qt.AlignmentFlag.AlignRight)
        self._alt_edit = _le("e.g. 45.0")
        grid.addWidget(self._alt_edit, 0, 1)

        grid.addWidget(QLabel("Azimuth (°):"),   0, 2, Qt.AlignmentFlag.AlignRight)
        self._az_edit = _le("0–360")
        grid.addWidget(self._az_edit, 0, 3)

        az_hint = QLabel("<small>0=N  90=E  180=S  270=W</small>")
        az_hint.setStyleSheet("color: palette(placeholderText);")
        grid.addWidget(az_hint, 0, 4)

        grid.addWidget(QLabel("Focal length (mm):"), 1, 0, Qt.AlignmentFlag.AlignRight)
        self._fl_edit = _le("e.g. 600")
        grid.addWidget(self._fl_edit, 1, 1)

        grid.addWidget(QLabel("Pixel pitch (µm):"),  1, 2, Qt.AlignmentFlag.AlignRight)
        self._px_edit = _le("e.g. 3.76")
        grid.addWidget(self._px_edit, 1, 3)

        grid.addWidget(QLabel("Sensor width (mm):"),  2, 0, Qt.AlignmentFlag.AlignRight)
        self._sw_edit = _le("e.g. 23.5")
        grid.addWidget(self._sw_edit, 2, 1)

        grid.addWidget(QLabel("Sensor height (mm):"), 2, 2, Qt.AlignmentFlag.AlignRight)
        self._sh_edit = _le("e.g. 15.6")
        grid.addWidget(self._sh_edit, 2, 3)

        grid.addWidget(QLabel("Max corner pixels:"),  3, 0, Qt.AlignmentFlag.AlignRight)
        self._tol_edit = _le("default: 1")
        self._tol_edit.setText("1")
        grid.addWidget(self._tol_edit, 3, 1)

        tol_hint = QLabel(
            "<small>Corner pixel movement before trailing is objectionable. "
            "1 = tightest, 5–10 = relaxed.</small>")
        tol_hint.setWordWrap(True)
        tol_hint.setStyleSheet("color: palette(placeholderText);")
        grid.addWidget(tol_hint, 3, 2, 1, 3)

        outer.addWidget(input_box)

        btn_row = QHBoxLayout()
        calc_btn = QPushButton("Calculate")
        calc_btn.setFixedHeight(32)
        calc_btn.setDefault(True)
        calc_btn.clicked.connect(self._calculate)
        btn_row.addStretch()
        btn_row.addWidget(calc_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        # ── Tab 1: Results ────────────────────────────────────────────────
        summary_tab = QWidget()
        st_lay = QVBoxLayout(summary_tab)
        st_lay.setContentsMargins(8, 8, 8, 8)
        st_lay.setSpacing(6)

        self._result_badge = QLabel("")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self._result_badge.setFont(font)
        self._result_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        st_lay.addWidget(self._result_badge)

        self._result_grid = QGridLayout()
        self._result_grid.setHorizontalSpacing(16)
        self._result_grid.setVerticalSpacing(4)
        st_lay.addLayout(self._result_grid)
        st_lay.addStretch(1)

        note = QLabel(
            "<small><b>How it works:</b> Stars rotate around the sensor centre on an Alt/Az "
            "mount. The corner sweeps the largest arc. "
            "Rate = 15.04 × cos(lat) × cos(az) / cos(alt)  arcsec/sec. "
            "Max exposure = (tolerance × pixel pitch) / (2π/360 × corner half-diagonal × rate). "
            "Field rotation is zero due East/West and maximum due North/South.</small>"
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "color: palette(placeholderText); font-size: 10px; padding-top: 6px;")
        st_lay.addWidget(note)

        self._tabs.addTab(summary_tab, "Results")

        # ── Tab 2: Exposure Over Night ────────────────────────────────────
        if _HAS_PG:
            plot_tab = QWidget()
            pt_lay = QVBoxLayout(plot_tab)
            pt_lay.setContentsMargins(4, 4, 4, 4)

            self._night_plot = pg.PlotWidget()
            self._night_plot.setBackground(pg.mkColor(40, 55, 120, 255))
            self._night_plot.showGrid(x=True, y=True, alpha=0.3)
            self._night_plot.setXRange(12, 36, padding=0)
            ticks = [(h, str(h if h <= 23 else h - 24)) for h in range(12, 37, 2)]
            self._night_plot.getAxis("bottom").setTicks([ticks])
            self._night_plot.setLabel("bottom", "Local Time (h)")
            self._night_plot.setLabel("left",   "Max Exposure (s)")
            pt_lay.addWidget(self._night_plot, 1)

            self._night_plot_hint = QLabel(
                "Run Calculate to generate the exposure curve over the night.")
            self._night_plot_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._night_plot_hint.setWordWrap(True)
            self._night_plot_hint.setStyleSheet(
                "color: palette(placeholderText); font-size: 11px; padding: 4px;")
            pt_lay.addWidget(self._night_plot_hint, 0)

            self._tabs.addTab(plot_tab, "Exposure Over Night")
        else:
            self._night_plot = None

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        outer.addLayout(close_row)

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_persisted(self):
        s, p = self._settings, self._SETTINGS_PREFIX
        for edit, key, default in [
            (self._fl_edit,  "focal_length", ""),
            (self._px_edit,  "pixel_pitch",  ""),
            (self._sw_edit,  "sensor_w",     ""),
            (self._sh_edit,  "sensor_h",     ""),
            (self._tol_edit, "tolerance",    "1"),
        ]:
            val = s.value(p + key, default, str)
            if val:
                edit.setText(val)

    def _save_persisted(self):
        s, p = self._settings, self._SETTINGS_PREFIX
        for edit, key in [
            (self._fl_edit,  "focal_length"),
            (self._px_edit,  "pixel_pitch"),
            (self._sw_edit,  "sensor_w"),
            (self._sh_edit,  "sensor_h"),
            (self._tol_edit, "tolerance"),
        ]:
            s.setValue(p + key, edit.text().strip())

    # ── Calculate ─────────────────────────────────────────────────────────

    def _parse_float(self, edit: QLineEdit, name: str) -> float:
        txt = edit.text().strip()
        if not txt:
            raise ValueError(f"{name} is empty")
        try:
            return float(txt)
        except ValueError:
            raise ValueError(f"'{txt}' is not a valid number for {name}")

    def _calculate(self):
        while self._result_grid.count():
            item = self._result_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            alt = self._parse_float(self._alt_edit, "Altitude")
            az  = self._parse_float(self._az_edit,  "Azimuth")
            fl  = self._parse_float(self._fl_edit,  "Focal length")   # noqa: F841
            px  = self._parse_float(self._px_edit,  "Pixel pitch")
            sw  = self._parse_float(self._sw_edit,  "Sensor width")
            sh  = self._parse_float(self._sh_edit,  "Sensor height")
            tol = self._parse_float(self._tol_edit, "Max corner pixels")

            if not (0.0 <= alt < 90.0):
                raise ValueError("Altitude must be 0°–89.9° (zenith singularity at 90°)")
            if not (0.0 <= az <= 360.0):
                raise ValueError("Azimuth must be 0°–360°")
            if px <= 0 or sw <= 0 or sh <= 0 or fl <= 0:
                raise ValueError("Focal length, sensor dimensions and pixel pitch must be > 0")
            if tol <= 0:
                raise ValueError("Tolerance must be > 0")

        except ValueError as e:
            self._result_badge.setText(f"⚠  {e}")
            self._result_badge.setStyleSheet("color: orange;")
            return

        self._save_persisted()

        res   = max_exposure_seconds(self._lat, az, alt, sw, sh, px,
                                      max_corner_pixels=tol)
        t     = res["t_max_sec"]
        badge = severity_badge(t)

        self._result_badge.setStyleSheet("")
        t_str = "∞  (no field rotation)" if t == float("inf") else f"{t:.1f} s"
        self._result_badge.setText(f"{t_str}   {badge}")

        def _row(label, value, row):
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 11px; color: palette(placeholderText);")
            val = QLabel(str(value))
            val.setStyleSheet("font-size: 12px; font-weight: 500;")
            self._result_grid.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignRight)
            self._result_grid.addWidget(val, row, 1)

        rate_sign = "+" if res["rate_arcsec_per_sec"] >= 0 else "−"
        _row("Field rotation rate:",
             f"{res['rate_mag_asec_sec']:.4f} arcsec/sec  "
             f"({res['rate_mag_asec_sec']*60:.2f} arcsec/min, {rate_sign}ve)", 0)
        _row("Corner half-diagonal:",
             f"{res['half_diag_mm']:.2f} mm  ({res['half_diag_um']:.0f} µm)", 1)
        _row("Diagonal / pixel ratio:",
             f"{res['d_over_s']:.1f} pixels", 2)
        _row("Tolerance:",
             f"{tol:.1f} corner pixel{'s' if tol != 1 else ''}", 3)
        _row("Max exposure at current Alt/Az:",
             t_str, 4)

        for ref_tol, row_offset in [(1, 5), (3, 6), (5, 7), (10, 8)]:
            res2  = max_exposure_seconds(self._lat, az, alt, sw, sh, px,
                                          max_corner_pixels=float(ref_tol))
            t2    = res2["t_max_sec"]
            t2_str = "∞" if t2 == float("inf") else f"{t2:.1f} s"
            _row(f"  @ {ref_tol} px tolerance:", t2_str, row_offset)

        self._tabs.setCurrentIndex(0)

        if _HAS_PG and self._night_plot is not None:
            self._build_night_curve(sw, sh, px, tol)

    # ── Night exposure curve ───────────────────────────────────────────────

    def _build_night_curve(self, sw: float, sh: float,
                            px: float, tol: float):
        """
        Plot max exposure vs. local time across a full noon-to-noon window.

        Uses the same _compute_alt_curve approach as ObjectVisibilityDialog:
          - n=288 samples (5-minute resolution)
          - x-axis: cumulative hours from noon (12 to 36)
          - twilight band shading: civil / nautical / astronomical
          - same dark blue background as all other WIMS plots
        """
        pw = self._night_plot
        pw.clear()

        ra_deg   = self._ra_deg
        dec_deg  = self._dec_deg
        date_str = self._date_str
        tz_name  = self._tz_name

        if None in (ra_deg, dec_deg, date_str, tz_name):
            self._night_plot_hint.setText(
                "No target RA/Dec or date available — "
                "open via the WIMS object list for a full night curve.")
            return

        # ── Build ephemeris — lifted directly from _compute_alt_curve ─────
        try:
            import pytz
            from datetime import datetime
            from astropy import units as u
            from astropy.coordinates import (SkyCoord, EarthLocation,
                                              AltAz, get_sun)
            from astropy.time import Time

            local_tz = pytz.timezone(tz_name)
            naive    = datetime.strptime(date_str, "%Y-%m-%d")
            # Start at noon on the observing date — same as _compute_alt_curve
            t_start  = Time(local_tz.localize(
                datetime(naive.year, naive.month, naive.day, 12, 0)))
            loc      = EarthLocation(lat=self._lat * u.deg,
                                     lon=self._lon * u.deg,
                                     height=0 * u.m)

            n     = 288   # 5-minute resolution, same as _compute_alt_curve
            times = t_start + np.linspace(0, 24, n) * u.hour
            frame = AltAz(obstime=times, location=loc)

            obj_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            altaz_arr = obj_coord.transform_to(frame)
            obj_alts  = altaz_arr.alt.deg
            obj_azs   = altaz_arr.az.deg
            sun_alts  = get_sun(times).transform_to(frame).alt.deg

            # Cumulative hours from noon: 12, 12.08, ..., 36
            hrs = 12.0 + np.linspace(0, 24, n)

        except Exception as e:
            self._night_plot_hint.setText(f"Could not compute night curve: {e}")
            return

        # ── Compute max exposure at each timestep — hard cap 300 s ───────────
        HARD_CAP  = 300.0   # meaningful ceiling: anything above ~5 min is "go for it"
        exp_curve = np.zeros(n, dtype=float)
        above     = obj_alts >= 0.0

        for i in range(n):
            if not above[i]:
                continue
            alt_i = float(obj_alts[i])
            az_i  = float(obj_azs[i])
            if alt_i >= 89.9:
                alt_i = 89.9
            r = max_exposure_seconds(self._lat, az_i, alt_i, sw, sh, px,
                                      max_corner_pixels=tol)
            exp_curve[i] = min(r["t_max_sec"], HARD_CAP)

        # ── Dynamic Y ceiling: night peak rounded up to a clean number ───────
        # Always <= 300 s.  Using the astronomical-night peak means a target
        # that peaks at 4 s gets a 5 s axis, not a 300 s axis with a flat line.
        astro_mask  = sun_alts < -18
        night_valid = astro_mask & above
        if np.any(night_valid):
            night_peak = float(np.max(exp_curve[night_valid]))
        elif np.any(above):
            night_peak = float(np.max(exp_curve[above]))
        else:
            night_peak = 10.0   # fallback — target never rises

        # Round up to the nearest clean number with ~25% headroom, max 300
        raw_ceil = night_peak * 1.25
        nice_levels = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300]
        CAP = next((v for v in nice_levels if v >= raw_ceil), 300)

        # ── Twilight shading — same logic as ObjectVisibilityDialog ───────
        def _sun_x_at_threshold(threshold):
            crossings = []
            for i in range(len(sun_alts) - 1):
                if (sun_alts[i] - threshold) * (sun_alts[i+1] - threshold) < 0:
                    frac = ((threshold - sun_alts[i])
                            / (sun_alts[i+1] - sun_alts[i]))
                    crossings.append(float(hrs[i] + frac * (hrs[i+1] - hrs[i])))
            return crossings

        def _fill_region(x0, x1, color, alpha):
            pw.addItem(pg.LinearRegionItem(
                values=(x0, x1),
                orientation="vertical",
                brush=pg.mkBrush(*color, alpha),
                pen=pg.mkPen(None),
                movable=False,
            ))

        civil_x    = _sun_x_at_threshold(-6)
        nautical_x = _sun_x_at_threshold(-12)
        astro_x    = _sun_x_at_threshold(-18)

        # Astronomical night (darkest)
        if len(astro_x) >= 2:
            _fill_region(astro_x[0],    astro_x[1],    (0,   0,  15), 200)
        elif len(astro_x) == 1:
            _fill_region(astro_x[0],    float(hrs[-1]), (0,   0,  15), 200)

        # Nautical twilight bands
        if len(nautical_x) >= 2 and len(astro_x) >= 2:
            _fill_region(nautical_x[0], astro_x[0],    (10, 20,  60), 160)
            _fill_region(astro_x[1],    nautical_x[1], (10, 20,  60), 160)

        # Civil twilight bands
        if len(civil_x) >= 2 and len(nautical_x) >= 2:
            _fill_region(civil_x[0],    nautical_x[0], (20, 35,  90), 120)
            _fill_region(nautical_x[1], civil_x[1],    (20, 35,  90), 120)

        # ── Severity threshold lines — only draw ones inside visible range ──
        for y_val, color, label in [
            (120, (80,  200,  80), "120 s"),
            ( 60, (180, 180,  40),  "60 s"),
            ( 30, (200, 120,  40),  "30 s"),
            ( 10, (200,  40,  40),  "10 s"),
        ]:
            if y_val > CAP:
                continue
            pw.addLine(y=y_val,
                       pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine))
            lbl = pg.TextItem(label, anchor=(0, 0.5), color=color)
            lbl.setPos(12.1, y_val)
            pw.addItem(lbl)

        # ── Main exposure curve ───────────────────────────────────────────
        masked = np.where(above, exp_curve, np.nan)
        pw.plot(hrs, masked,
                pen=pg.mkPen((100, 200, 255), width=2.5),
                name=f"Max exp ({tol:.0f} px tol)")

        # ── Object altitude overlay — thin dashed yellow, scaled to CAP ──
        alt_scaled = np.where(above, obj_alts * (CAP / 90.0), np.nan)
        pw.plot(hrs, alt_scaled,
                pen=pg.mkPen((255, 220, 80, 120), width=1.2,
                             style=Qt.PenStyle.DashLine),
                name="Altitude (scaled)")

        # Horizon line and 30°-equivalent guide line
        pw.addLine(y=0,                  pen=pg.mkPen("w", width=1,
                                         style=Qt.PenStyle.DashLine))
        pw.addLine(y=30.0 * CAP / 90.0, pen=pg.mkPen("g", width=1,
                                         style=Qt.PenStyle.DotLine))

        # ── Axis ranges and tick labels ───────────────────────────────────
        pw.setXRange(12, 36, padding=0)
        pw.setYRange(0, CAP * 1.08, padding=0)   # small breathing room above cap
        ticks = [(h, str(h if h <= 23 else h - 24)) for h in range(12, 37, 2)]
        pw.getAxis("bottom").setTicks([ticks])
        pw.setLabel("bottom", "Local Time (h)")

        cap_str = f"{int(CAP)} s" if CAP < 300 else "300 s (max)"
        pw.setLabel("left",
                    f"Max Exposure (s)  [{tol:.0f} px tolerance, cap {cap_str}]")

        pw.addLegend()

        # ── Best-night annotation — peak exposure during astro night ──────
        night_exp  = np.where(night_valid, exp_curve, 0.0)
        if np.any(night_exp > 0):
            best_idx = int(np.argmax(night_exp))
            best_val = float(night_exp[best_idx])
            best_hr  = float(hrs[best_idx])
            pw.addLine(x=best_hr,
                       pen=pg.mkPen("r", width=1.5, style=Qt.PenStyle.DashLine))
            ann = pg.TextItem(f"Best {best_val:.1f} s", color="r", anchor=(0, 1))
            ann.setPos(best_hr, best_val + CAP * 0.03)   # offset scales with axis
            pw.addItem(ann)

        name_part = f"{self._target_name}  —  " if self._target_name else ""
        self._night_plot_hint.setText(
            f"{name_part}{date_str}  —  "
            f"Blue = max exposure vs. time  •  "
            f"Yellow dashed = target altitude (scaled)  •  "
            f"Red dashed = best exposure during astronomical night  •  "
            f"Curve is zero while target is below horizon."
        )


# ── Convenience launcher ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = AltAzFieldRotationDialog(
        lat=30.4, lon=-90.2,
        alt=45.0, az=180.0,
        ra_deg=83.82, dec_deg=-5.39,   # Orion Nebula
        date_str="2025-12-15",
        tz_name="US/Central",
    )
    dlg.show()
    sys.exit(app.exec())