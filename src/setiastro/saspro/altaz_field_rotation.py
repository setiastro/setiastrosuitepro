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
#    dlg = AltAzFieldRotationDialog(lat, lon, alt_deg, az_deg, settings, parent=self)
#    dlg.show()
# ============================================================

from __future__ import annotations

import math
import os

import numpy as np

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QFrame, QSizePolicy, QWidget,
    QTabWidget,
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

    Formula (RASC Calgary / CaliforniaSkys):
        R = 15.04 × cos(lat) × cos(az) / cos(alt)

    Returns the *signed* rate; use abs() for magnitude.
    Raises ZeroDivisionError-equivalent if alt ≈ 90° (zenith singularity).
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
    Calculate maximum exposure time before corner pixel motion exceeds
    max_corner_pixels pixels.

    Returns a dict with all intermediate values for display.
    """
    # Corner half-diagonal in microns
    half_diag_mm  = math.sqrt((sensor_w_mm / 2) ** 2 + (sensor_h_mm / 2) ** 2)
    half_diag_um  = half_diag_mm * 1000.0

    # Field rotation rate (arcsec/sec, signed)
    rate_asec_sec = field_rotation_rate_arcsec_per_sec(lat_deg, az_deg, alt_deg)
    rate_mag      = abs(rate_asec_sec)          # magnitude only
    rate_deg_sec  = rate_mag / 3600.0

    # Path length formula:
    #   PL (µm) = 0.0000729 × d (µm) × cos(lat) × |cos(az)| × t / cos(alt)
    # Rearranged for t:
    #   t = (max_pixels × pixel_pitch) / (0.0000729 × d × rate_deg_sec × (180/π) ... )
    #
    # Cleaner: t = (max_pixels × pixel_pitch_um) / (2π/360 × half_diag_um × rate_deg_sec)
    #             where rate_deg_sec already encodes the full trig.
    #
    # From the reference derivation:
    #   Pixels = (2π/360) × (d/s) × R_deg_hr × t / 3600   [d in µm, s in µm, R in deg/hr]
    # Inverting:
    #   t = Pixels × s × 3600 / ((2π/360) × d × R_deg_hr)

    rate_deg_hr  = rate_mag / 240.0   # 15.04 deg/hr base, scaled
    # Actually compute directly from the page formula:
    #   PL / s = 0.0000729 × (d/s) × cos(lat) × cos(az) × t / cos(alt)
    # The 0.0000729 constant = 2π/360 × (15.04/3600)
    # We already have rate_deg_sec = 15.04 × cos(lat) × |cos(az)| / cos(alt) / 3600
    # So:  Pixels = (2π/360) × (d/s) × rate_deg_sec × t
    # Thus: t = Pixels / ((2π/360) × (d/s) × rate_deg_sec)

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
    lat : float   Observer latitude (degrees, north positive)
    lon : float   Observer longitude (degrees, east positive) — for future use
    alt : float   Target altitude (degrees)  — can be None for manual entry
    az  : float   Target azimuth  (degrees)  — can be None for manual entry
    settings : QSettings
    parent   : QWidget
    """

    _SETTINGS_PREFIX = "wims/field_rotation/"

    def __init__(self, lat: float, lon: float,
                 alt: float | None = None,
                 az:  float | None = None,
                 settings: QSettings | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alt/Az Field Rotation Calculator")
        self.resize(680, 580)

        self._lat = lat
        self._lon = lon
        self._settings = settings or QSettings()

        self._build_ui()
        self._load_persisted()

        # Pre-fill target position if supplied
        if alt is not None:
            self._alt_edit.setText(f"{alt:.1f}")
        if az is not None:
            self._az_edit.setText(f"{az:.1f}")

        # Run initial calculation if we have a full set of inputs
        if alt is not None and az is not None:
            self._calculate()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 10)
        outer.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────
        hdr = QLabel(
            f"<b>Alt/Az Field Rotation Calculator</b>"
            f"<br><small>Observer latitude: <b>{self._lat:+.4f}°</b>  "
            f"longitude: <b>{self._lon:+.4f}°</b></small>"
        )
        hdr.setWordWrap(True)
        outer.addWidget(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

        # ── Input grid ────────────────────────────────────────────────────
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

        # Row 0 — target position
        grid.addWidget(QLabel("Altitude (°):"),  0, 0, Qt.AlignmentFlag.AlignRight)
        self._alt_edit = _le("e.g. 45.0")
        grid.addWidget(self._alt_edit, 0, 1)

        grid.addWidget(QLabel("Azimuth (°):"),   0, 2, Qt.AlignmentFlag.AlignRight)
        self._az_edit = _le("0–360")
        grid.addWidget(self._az_edit, 0, 3)

        az_hint = QLabel("<small>0=N  90=E  180=S  270=W</small>")
        az_hint.setStyleSheet("color: palette(placeholderText);")
        grid.addWidget(az_hint, 0, 4)

        # Row 1 — optics
        grid.addWidget(QLabel("Focal length (mm):"), 1, 0, Qt.AlignmentFlag.AlignRight)
        self._fl_edit = _le("e.g. 600")
        grid.addWidget(self._fl_edit, 1, 1)

        grid.addWidget(QLabel("Pixel pitch (µm):"),  1, 2, Qt.AlignmentFlag.AlignRight)
        self._px_edit = _le("e.g. 3.76")
        grid.addWidget(self._px_edit, 1, 3)

        # Row 2 — sensor
        grid.addWidget(QLabel("Sensor width (mm):"),  2, 0, Qt.AlignmentFlag.AlignRight)
        self._sw_edit = _le("e.g. 23.5")
        grid.addWidget(self._sw_edit, 2, 1)

        grid.addWidget(QLabel("Sensor height (mm):"), 2, 2, Qt.AlignmentFlag.AlignRight)
        self._sh_edit = _le("e.g. 15.6")
        grid.addWidget(self._sh_edit, 2, 3)

        # Row 3 — tolerance
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

        # ── Calculate button ──────────────────────────────────────────────
        btn_row = QHBoxLayout()
        calc_btn = QPushButton("Calculate")
        calc_btn.setFixedHeight(32)
        calc_btn.setDefault(True)
        calc_btn.clicked.connect(self._calculate)
        btn_row.addStretch()
        btn_row.addWidget(calc_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        # ── Results area ──────────────────────────────────────────────────
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        # Tab 1: summary
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

        # Explanation note
        note = QLabel(
            "<small><b>How it works:</b> Stars rotate around the sensor centre on an Alt/Az "
            "mount. The corner of the sensor sweeps the largest arc. "
            "Rate = 15.04 × cos(lat) × cos(az) / cos(alt)  arcsec/sec. "
            "Max exposure = (tolerance × pixel pitch) / (2π/360 × corner half-diagonal × rate)."
            "<br>Field rotation is zero due East/West and maximum due North/South.</small>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(placeholderText); font-size: 10px; padding-top: 6px;")
        st_lay.addWidget(note)

        self._tabs.addTab(summary_tab, "Results")

        # Tab 2: night exposure plot
        if _HAS_PG:
            plot_tab = QWidget()
            pt_lay = QVBoxLayout(plot_tab)
            pt_lay.setContentsMargins(4, 4, 4, 4)
            self._night_plot = pg.PlotWidget()
            self._night_plot.setLabel("bottom", "Local Time (h from noon)")
            self._night_plot.setLabel("left", "Max Exposure (s)")
            self._night_plot.showGrid(x=True, y=True, alpha=0.3)
            self._night_plot.setBackground(pg.mkColor(30, 40, 80, 255))
            pt_lay.addWidget(self._night_plot)
            self._night_plot_hint = QLabel(
                "Run Calculate to generate the exposure curve over the night.")
            self._night_plot_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._night_plot_hint.setStyleSheet("color: palette(placeholderText); font-size: 11px;")
            pt_lay.addWidget(self._night_plot_hint)
            self._tabs.addTab(plot_tab, "Exposure Over Night")
        else:
            self._night_plot = None

        # ── Close button ──────────────────────────────────────────────────
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        outer.addLayout(close_row)

    # ── Persistence ───────────────────────────────────────────────────────

    def _load_persisted(self):
        s = self._settings
        p = self._SETTINGS_PREFIX
        for edit, key, default in [
            (self._fl_edit, "focal_length", ""),
            (self._px_edit, "pixel_pitch",  ""),
            (self._sw_edit, "sensor_w",     ""),
            (self._sh_edit, "sensor_h",     ""),
            (self._tol_edit,"tolerance",    "1"),
        ]:
            val = s.value(p + key, default, str)
            if val:
                edit.setText(val)

    def _save_persisted(self):
        s = self._settings
        p = self._SETTINGS_PREFIX
        for edit, key in [
            (self._fl_edit, "focal_length"),
            (self._px_edit, "pixel_pitch"),
            (self._sw_edit, "sensor_w"),
            (self._sh_edit, "sensor_h"),
            (self._tol_edit,"tolerance"),
        ]:
            s.setValue(p + key, edit.text().strip())

    # ── Calculation ───────────────────────────────────────────────────────

    def _parse_float(self, edit: QLineEdit, name: str) -> float:
        txt = edit.text().strip()
        if not txt:
            raise ValueError(f"{name} is empty")
        try:
            return float(txt)
        except ValueError:
            raise ValueError(f"'{txt}' is not a valid number for {name}")

    def _calculate(self):
        # Clear previous result rows
        while self._result_grid.count():
            item = self._result_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            alt  = self._parse_float(self._alt_edit, "Altitude")
            az   = self._parse_float(self._az_edit,  "Azimuth")
            fl   = self._parse_float(self._fl_edit,  "Focal length")   # noqa: F841
            px   = self._parse_float(self._px_edit,  "Pixel pitch")
            sw   = self._parse_float(self._sw_edit,  "Sensor width")
            sh   = self._parse_float(self._sh_edit,  "Sensor height")
            tol  = self._parse_float(self._tol_edit, "Max corner pixels")

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

        res = max_exposure_seconds(
            self._lat, az, alt, sw, sh, px, max_corner_pixels=tol)

        t = res["t_max_sec"]
        badge = severity_badge(t)

        self._result_badge.setStyleSheet("")
        if t == float("inf"):
            t_str = "∞  (no field rotation)"
        else:
            t_str = f"{t:.1f} s"

        self._result_badge.setText(f"{t_str}   {badge}")

        # Populate results grid
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
             f"({res['rate_mag_asec_sec']*60:.2f} arcsec/min, {rate_sign}ve)",  0)
        _row("Corner half-diagonal:",
             f"{res['half_diag_mm']:.2f} mm  ({res['half_diag_um']:.0f} µm)",   1)
        _row("Diagonal / pixel ratio:",
             f"{res['d_over_s']:.1f} pixels",                                    2)
        _row("Tolerance:",
             f"{tol:.1f} corner pixel{'s' if tol != 1 else ''}",                 3)
        _row("Max exposure:",
             t_str,                                                               4)

        # Nice context comparisons
        for ref_tol, row_offset in [(1, 5), (3, 6), (5, 7), (10, 8)]:
            res2 = max_exposure_seconds(self._lat, az, alt, sw, sh, px,
                                         max_corner_pixels=float(ref_tol))
            t2 = res2["t_max_sec"]
            t2_str = "∞" if t2 == float("inf") else f"{t2:.1f} s"
            _row(f"  @ {ref_tol} px tolerance:", t2_str, row_offset)

        self._tabs.setCurrentIndex(0)

        # ── Night curve ───────────────────────────────────────────────────
        if _HAS_PG and self._night_plot is not None:
            self._build_night_curve(alt, az, sw, sh, px, tol)

    def _build_night_curve(self, target_alt_now: float, target_az_now: float,
                            sw: float, sh: float, px: float, tol: float):
        """
        Plot max exposure as a function of time across a single night by
        re-computing alt/az at each hour using the observer's lat/lon and the
        target coordinates inferred from the current alt/az + sidereal time.

        Since we don't have the target's RA/Dec in this dialog, we use a
        pragmatic approach: sweep azimuth through a range anchored around the
        current az and compute max exposure vs. azimuth, annotated with compass
        directions and noting that real altitude also changes. This gives a
        useful sensitivity plot even without full ephemeris.

        For a richer curve the caller can pass RA/Dec in a future version.
        """
        pw = self._night_plot
        pw.clear()

        # Sweep azimuth 0–360 to show how exposure varies across the sky
        # at the *current* altitude — a "sky map slice" at fixed altitude.
        azs = np.linspace(0, 360, 721)
        ts  = []
        for az in azs:
            r = max_exposure_seconds(self._lat, float(az), target_alt_now,
                                      sw, sh, px, max_corner_pixels=tol)
            t = r["t_max_sec"]
            ts.append(min(t, 300.0))   # cap at 300 s for display sanity

        ts = np.array(ts, dtype=float)

        # Shade regions by severity
        def _fill_band(y_min, y_max, color, alpha):
            region = pg.LinearRegionItem(
                values=(y_min, y_max), orientation="horizontal",
                brush=pg.mkBrush(*color, alpha), pen=pg.mkPen(None), movable=False)
            pw.addItem(region)

        _fill_band(0,   10,  (200,  40,  40), 40)
        _fill_band(10,  30,  (200, 120,  40), 40)
        _fill_band(30,  60,  (180, 180,  40), 40)
        _fill_band(60,  120, ( 80, 180,  80), 40)
        _fill_band(120, 310, ( 80, 180, 200), 30)

        pw.plot(azs, ts,
                pen=pg.mkPen((100, 200, 255), width=2.5),
                name=f"Max exp ({tol:.0f} px tol)")

        # Mark current target azimuth
        pw.addLine(x=target_az_now,
                   pen=pg.mkPen("r", width=1.5, style=Qt.PenStyle.DashLine))
        txt = pg.TextItem(f"Target Az {target_az_now:.0f}°", color="r", anchor=(0, 1))
        txt.setPos(target_az_now, float(min(ts[int(target_az_now * 2)], 280)))
        pw.addItem(txt)

        # Compass ticks
        ticks = [(0,"N"),(45,"NE"),(90,"E"),(135,"SE"),
                 (180,"S"),(225,"SW"),(270,"W"),(315,"NW"),(360,"N")]
        pw.getAxis("bottom").setTicks([ticks])
        pw.setLabel("bottom", f"Azimuth (°)  [at fixed alt = {target_alt_now:.1f}°]")
        pw.setLabel("left",   f"Max Exposure (s, capped 300)  [{tol:.0f} px tol]")
        pw.setXRange(0, 360, padding=0.01)
        pw.setYRange(0, 310, padding=0.02)

        self._night_plot_hint.setText(
            "Shows maximum exposure vs. azimuth at the target's current altitude. "
            "Red dashed line = current target azimuth. "
            "Shaded bands: red <10 s · orange 10–30 s · yellow 30–60 s · "
            "green 60–120 s · teal ≥120 s.")


# ── Convenience launcher (for quick testing outside SASpro) ─────────────────

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = AltAzFieldRotationDialog(lat=30.4, lon=-90.2, alt=45.0, az=180.0)
    dlg.show()
    sys.exit(app.exec())