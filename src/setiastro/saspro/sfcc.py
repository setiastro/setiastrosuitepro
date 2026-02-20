# sfcc.py
# SASpro Spectral Flux Color Calibration (SFCC) — "current view" integration
# - Expects a "view adapter" you provide that exposes:
#       get_rgb_image() -> np.ndarray (H,W,3), uint8 or float32 in [0,1]
#       get_metadata() -> dict  (optional; may return {})
#       get_header() -> astropy.io.fits.Header or dict (optional but needed for WCS features)
#       set_rgb_image(img: np.ndarray, metadata: dict | None = None, step_name: str | None = None) -> None
#   If your adapter names differ, tweak _get_img_meta/_get_header/_push_image below (they already try a few fallbacks).
#
# - Call open_sfcc(view_adapter, sasp_data_path) to show the dialog.

from __future__ import annotations

from setiastro.saspro.main_helpers import non_blocking_sleep

import os
import re
import cv2
import math
import time
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

import numpy.ma as ma
import pandas as pd

# ── SciPy bits
from scipy.interpolate import RBFInterpolator, interp1d
from scipy.signal import medfilt

# ── Astropy / Astroquery
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy.wcs.wcs import NoConvergence

# ── SEP (Source Extractor)
import sep

# ── Matplotlib backend for Qt
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import (Qt, QPoint, QRect, QMimeData, QSettings, QByteArray,
                          QDataStream, QIODevice, QEvent, QStandardPaths)
from PyQt6.QtGui import (QAction, QDrag, QIcon, QMouseEvent, QPixmap, QKeyEvent)
from PyQt6.QtWidgets import (QToolBar, QWidget, QToolButton, QMenu, QApplication, QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox,
                             QInputDialog, QMessageBox, QDialog, QFileDialog,
    QFormLayout, QDialogButtonBox, QDoubleSpinBox, QCheckBox, QLabel, QRubberBand, QRadioButton, QMainWindow, QPushButton)

from setiastro.saspro.backgroundneutral import run_background_neutral_via_preset
from setiastro.saspro.backgroundneutral import background_neutralize_rgb, auto_rect_50x50
# --- Gaia XP fallback (optional) ---
from astroquery.gaia import Gaia
# you will add gaia_downloader.py to src/setiastro/saspro/gaia_downloader.py
from setiastro.saspro.gaia_downloader import GaiaDownloader, HAS_GAIAXPY

import warnings
from typing import Callable, Dict, Any, Iterable

ProgressMsgCB = Callable[[str], None]  # status_cb("text...")

try:
    import pandas as pd  # you already have
except Exception:
    pd = None


import gaiaxpy
from gaiaxpy import generate
from gaiaxpy.generator.photometric_system import PhotometricSystem

import io
import re
import contextlib

_GAIA_PW_BANNER_RE = re.compile(
    r"passwords of all user accounts have been inactivated", re.IGNORECASE
)

@contextlib.contextmanager
def suppress_gaia_archive_banner():
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield
    # If you ever want to keep *other* messages, you can inspect buf_out/buf_err here.
    # For now, we intentionally drop everything printed during generate().

def _pick_mag_column(df, band: str) -> str | None:
    """
    GaiaXPy returns a DataFrame with columns per band.
    Column names vary by system/version; pick something that looks like "<band>...mag".
    """
    b = band.strip().lower()
    cols = list(df.columns)
    # prefer exact-ish matches first
    preferred = [
        f"{b}_mag", f"{b}mag", f"mag_{b}", f"{b}_magnitude", f"{b}magnitude",
    ]
    lower_map = {c.lower(): c for c in cols}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]

    # fallback: anything containing band and "mag" but not "err"
    for c in cols:
        cl = c.lower()
        if ("mag" in cl) and (b in cl) and ("err" not in cl) and ("error" not in cl):
            return c

    return None


def gaiaxp_synth_bvr_from_source_ids(
    source_ids: Iterable[int],
    *,
    status_cb: ProgressMsgCB | None = None,
    cosmos_username: str | None = None,
    cosmos_password: str | None = None,
    cache_dir: str | None = None,
) -> Dict[int, Dict[str, float]]:
    """
    Returns:
      { source_id: {"B": Bmag, "V": Vmag, "R": Rmag} }

    Uses GaiaXPy `generate()` on the JKC system. `generate()` accepts a list of source ids. :contentReference[oaicite:1]{index=1}
    """
    out: Dict[int, Dict[str, float]] = {}
    ids = [int(x) for x in source_ids if x is not None]
    ids = list(dict.fromkeys(ids))  # de-dupe preserving order

    if not ids or generate is None or PhotometricSystem is None:
        return out

    # Reduce "it froze" anxiety + stop scary ResourceWarning spam
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Keep GaiaXPy cache somewhere sane (optional)
    if cache_dir:
        os.environ.setdefault("GAIAXPY_CACHE_DIR", cache_dir)

    # Also avoid any TeX weirdness if any plotting gets triggered somewhere
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.rcParams["text.usetex"] = False
    except Exception:
        pass

    if status_cb:
        status_cb(f"[SPCC] Gaia XP fallback: fetching synthetic photometry for {len(ids)} Gaia sources… (can take a bit)")

    # GaiaXPy: list-of-source-ids -> DataFrame synthetic photometry :contentReference[oaicite:2]{index=2}
    df = generate(
        ids,
        photometric_system=PhotometricSystem.JKC,
        save_file=False,
        username=cosmos_username,
        password=cosmos_password,
    )

    if df is None or getattr(df, "empty", False):
        return out

    # Identify the source id column
    sid_col = None
    for c in df.columns:
        if c.lower() in ("source_id", "sourceid", "gaia_source_id"):
            sid_col = c
            break
    if sid_col is None:
        # last-ditch: first column that contains "source" and "id"
        for c in df.columns:
            cl = c.lower()
            if "source" in cl and "id" in cl:
                sid_col = c
                break
    if sid_col is None:
        return out

    b_col = _pick_mag_column(df, "B")
    v_col = _pick_mag_column(df, "V")
    r_col = _pick_mag_column(df, "R")
    if not (b_col and v_col and r_col):
        return out

    for _, row in df.iterrows():
        try:
            sid = int(row[sid_col])
            B = float(row[b_col])
            V = float(row[v_col])
            R = float(row[r_col])
        except Exception:
            continue
        out[sid] = {"B": B, "V": V, "R": R}

    if status_cb:
        status_cb(f"[SPCC] Gaia XP fallback: synthetic B/V/R available for {len(out)} sources")

    return out



# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

# --- Debug/guards -----------------------------------------------------
def _debug_probe_channels(img: np.ndarray, label="input"):
    assert img.ndim == 3 and img.shape[2] == 3, f"[SPCC] {label}: not RGB"
    f = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    means = [float(f[...,i].mean()) for i in range(3)]
    stds  = [float(f[...,i].std())  for i in range(3)]
    rg = float(np.corrcoef(f[...,0].ravel(), f[...,1].ravel())[0,1])
    rb = float(np.corrcoef(f[...,0].ravel(), f[...,2].ravel())[0,1])
    gb = float(np.corrcoef(f[...,1].ravel(), f[...,2].ravel())[0,1])
    print(f"[SPCC] {label}: mean={means}, std={stds}, corr(R,G)={rg:.5f}, corr(R,B)={rb:.5f}, corr(G,B)={gb:.5f}")
    return rg, rb, gb

def _maybe_bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    # Heuristic: if channel-2 is consistently brightest in highlights and ch-0 the dimmest → likely BGR.
    f = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    lum = np.mean(f, axis=2)
    thr = np.quantile(lum, 0.95)
    m0 = f[...,0][lum >= thr].mean() if np.any(lum >= thr) else f[...,0].mean()
    m1 = f[...,1][lum >= thr].mean() if np.any(lum >= thr) else f[...,1].mean()
    m2 = f[...,2][lum >= thr].mean() if np.any(lum >= thr) else f[...,2].mean()
    if (m2 > m1 >= m0) and (m2 - m0 > 0.02):
        print("[SPCC] Heuristic suggests BGR input → converting to RGB")
        return img[..., ::-1]
    return img

def _ensure_angstrom(wl: np.ndarray) -> np.ndarray:
    """If wavelengths look like nm (≈300–1100), convert to Å."""
    med = float(np.median(wl))
    return wl * 10.0 if 250.0 <= med <= 2000.0 else wl


def pickles_match_for_simbad(simbad_sp: Optional[str], available_extnames: List[str]) -> List[str]:
    if not simbad_sp:   # handles None, "", etc.
        return []
    sp = str(simbad_sp).strip().upper()
    if not sp:
        return []
    m = re.match(r"^([OBAFGKMLT])(\d?)(I{1,3}|IV|V)?", sp)
    if not m:
        return []
    letter_class = m.group(1)
    digit_part   = m.group(2)
    lum_part     = m.group(3)
    subclass = int(digit_part) if digit_part != "" else None

    def parse_pickles_extname(ext: str):
        ext = ext.strip().upper()
        m2 = re.match(r"^([OBAFGKMLT])(\d+)(I{1,3}|IV|V)$", ext)
        if not m2:
            return None, None, None
        return m2.group(1), int(m2.group(2)), m2.group(3)

    parsed_templates = []
    for ext in available_extnames:
        l2, d2, L2 = parse_pickles_extname(ext)
        if l2 is not None:
            parsed_templates.append((ext, l2, d2, L2))

    # Exact
    if subclass is not None and lum_part is not None:
        target = f"{letter_class}{subclass}{lum_part}"
        if target in available_extnames:
            return [target]

    # Same letter (+same lum if we have it)
    same_letter_and_lum = []
    same_letter_any_lum = []
    for (ext, l2, d2, L2) in parsed_templates:
        if l2 != letter_class:
            continue
        if lum_part is not None and L2 == lum_part:
            same_letter_and_lum.append((ext, d2))
        else:
            same_letter_any_lum.append((ext, d2))

    def pick_nearest(candidates: List[Tuple[str, int]], target: int) -> List[str]:
        if not candidates or target is None:
            return []
        arr = np.abs(np.array([d for _, d in candidates]) - target)
        mind = np.min(arr)
        return [candidates[i][0] for i in np.where(arr == mind)[0]]

    if subclass is not None and lum_part is not None:
        if same_letter_and_lum:
            return pick_nearest(same_letter_and_lum, subclass)
        if same_letter_any_lum:
            return pick_nearest(same_letter_any_lum, subclass)

    if subclass is not None and lum_part is None:
        if same_letter_any_lum:
            return pick_nearest(same_letter_any_lum, subclass)

    if subclass is None and lum_part is None:
        return sorted([ext for (ext, l2, _, _) in parsed_templates if l2 == letter_class])

    if subclass is None and lum_part is not None:
        cands = [ (ext, d2) for (ext, l2, d2, L2) in parsed_templates if l2 == letter_class and L2 == lum_part ]
        if cands:
            return sorted([ext for (ext, _) in cands])
        return sorted([ext for (ext, l2, _, _) in parsed_templates if l2 == letter_class])

    return []


def compute_gradient_map(sources, delta_flux, shape, method="poly2"):
    H, W = shape
    xs, ys = sources[:, 0], sources[:, 1]

    if method == "poly2":
        A = np.vstack([np.ones_like(xs), xs, ys, xs**2, xs*ys, ys**2]).T
        coeffs, *_ = np.linalg.lstsq(A, delta_flux, rcond=None)
        YY, XX = np.mgrid[0:H, 0:W]
        return (coeffs[0] + coeffs[1]*XX + coeffs[2]*YY
                + coeffs[3]*XX**2 + coeffs[4]*XX*YY + coeffs[5]*YY**2)

    elif method == "poly3":
        A = np.vstack([
            np.ones_like(xs), xs, ys,
            xs**2, xs*ys, ys**2,
            xs**3, xs**2*ys, xs*ys**2, ys**3
        ]).T
        coeffs, *_ = np.linalg.lstsq(A, delta_flux, rcond=None)
        YY, XX = np.mgrid[0:H, 0:W]
        return (coeffs[0] + coeffs[1]*XX + coeffs[2]*YY
                + coeffs[3]*XX**2 + coeffs[4]*XX*YY + coeffs[5]*YY**2
                + coeffs[6]*XX**3 + coeffs[7]*XX**2*YY + coeffs[8]*XX*YY**2 + coeffs[9]*YY**3)

    elif method == "rbf":
        pts = np.vstack([xs, ys]).T
        rbfi = RBFInterpolator(pts, delta_flux, kernel="thin_plate_spline", smoothing=1.0)
        YY, XX = np.mgrid[0:H, 0:W]
        grid_pts = np.vstack([XX.ravel(), YY.ravel()]).T
        return rbfi(grid_pts).reshape(H, W)

    else:
        raise ValueError("method must be one of 'poly2','poly3','rbf'")

def _pivot_scale_channel(ch: np.ndarray, gain: np.ndarray | float, pivot: float) -> np.ndarray:
    """
    Apply gain around a pivot: pivot + (x - pivot)*gain.
    gain can be scalar or per-pixel array.
    """
    return pivot + (ch - pivot) * gain

# ──────────────────────────────────────────────────────────────────────────────
# Simple responses viewer (unchanged core logic; useful for diagnostics)
# ──────────────────────────────────────────────────────────────────────────────
class SaspViewer(QMainWindow):
    def __init__(self, sasp_data_path: str, user_custom_path: str):
        super().__init__()
        self.setWindowTitle(self.tr("SASP Viewer (Pickles + RGB Responses)"))

        self.base_hdul   = fits.open(sasp_data_path,   mode="readonly", memmap=False)
        self.custom_hdul = fits.open(user_custom_path, mode="readonly", memmap=False)

        self.pickles_templates = []
        self.filter_list       = []
        self.sensor_list       = []
        for hdul in (self.custom_hdul, self.base_hdul):
            for hdu in hdul:
                if not isinstance(hdu, fits.BinTableHDU): continue
                c = hdu.header.get("CTYPE","").upper()
                e = hdu.header.get("EXTNAME","")
                if c == "SED":    self.pickles_templates.append(e)
                elif c == "FILTER": self.filter_list.append(e)
                elif c == "SENSOR": self.sensor_list.append(e)

        for lst in (self.pickles_templates, self.filter_list, self.sensor_list):
            lst.sort()
        self.rgb_filter_choices = ["(None)"] + self.filter_list

        central = QWidget(); self.setCentralWidget(central)
        vbox = QVBoxLayout(); central.setLayout(vbox)

        row = QHBoxLayout(); vbox.addLayout(row)
        row.addWidget(QLabel(self.tr("Star Template:")))
        self.star_combo = QComboBox(); self.star_combo.addItems(self.pickles_templates); row.addWidget(self.star_combo)
        row.addWidget(QLabel(self.tr("R-Filter:")))
        self.r_filter_combo = QComboBox(); self.r_filter_combo.addItems(self.rgb_filter_choices); row.addWidget(self.r_filter_combo)
        row.addWidget(QLabel(self.tr("G-Filter:")))
        self.g_filter_combo = QComboBox(); self.g_filter_combo.addItems(self.rgb_filter_choices); row.addWidget(self.g_filter_combo)
        row.addWidget(QLabel(self.tr("B-Filter:")))
        self.b_filter_combo = QComboBox(); self.b_filter_combo.addItems(self.rgb_filter_choices); row.addWidget(self.b_filter_combo)

        row2 = QHBoxLayout(); vbox.addLayout(row2)
        row2.addWidget(QLabel(self.tr("LP/Cut Filter1:")))
        self.lp_filter_combo = QComboBox(); self.lp_filter_combo.addItems(self.rgb_filter_choices); row2.addWidget(self.lp_filter_combo)
        row2.addWidget(QLabel(self.tr("LP/Cut Filter2:")))
        self.lp_filter_combo2 = QComboBox(); self.lp_filter_combo2.addItems(self.rgb_filter_choices); row2.addWidget(self.lp_filter_combo2)
        row2.addSpacing(20); row2.addWidget(QLabel(self.tr("Sensor (QE):")))
        self.sens_combo = QComboBox(); self.sens_combo.addItems(self.sensor_list); row2.addWidget(self.sens_combo)

        self.plot_btn = QPushButton(self.tr("Plot")); self.plot_btn.clicked.connect(self.update_plot); row.addWidget(self.plot_btn)

        self.figure = Figure(figsize=(9, 6)); self.canvas = FigureCanvas(self.figure); vbox.addWidget(self.canvas)
        self.update_plot()

    def closeEvent(self, event):
        self.base_hdul.close(); self.custom_hdul.close()
        super().closeEvent(event)

    def load_any(self, extname, field):
        for hdul in (self.custom_hdul, self.base_hdul):
            if extname in hdul:
                return hdul[extname].data[field].astype(float)
        raise KeyError(f"Extension '{extname}' not found")

    def update_plot(self):
        star_ext = self.star_combo.currentText()
        r_filt   = self.r_filter_combo.currentText()
        g_filt   = self.g_filter_combo.currentText()
        b_filt   = self.b_filter_combo.currentText()
        sens_ext = self.sens_combo.currentText()
        lp_ext1  = self.lp_filter_combo.currentText()
        lp_ext2  = self.lp_filter_combo2.currentText()

        wl_star = self.load_any(star_ext, "WAVELENGTH")
        fl_star = self.load_any(star_ext, "FLUX")
        wl_sens = self.load_any(sens_ext, "WAVELENGTH")
        qe_sens = self.load_any(sens_ext, "THROUGHPUT")

        wl_min, wl_max = 1150.0, 10620.0
        common_wl = np.arange(wl_min, wl_max + 1.0, 1.0)

        sed_interp  = interp1d(wl_star, fl_star, kind="linear", bounds_error=False, fill_value=0.0)
        sens_interp = interp1d(wl_sens, qe_sens, kind="linear", bounds_error=False, fill_value=0.0)
        fl_common   = sed_interp(common_wl)
        sens_common = sens_interp(common_wl)

        rgb_data = {}
        for color, filt_name in (("red", r_filt), ("green", g_filt), ("blue", b_filt)):
            if filt_name == "(None)":
                rgb_data[color] = None; continue

            wl_filt = self.load_any(filt_name, "WAVELENGTH")
            tr_filt = self.load_any(filt_name, "THROUGHPUT")
            filt_common = interp1d(wl_filt, tr_filt, bounds_error=False, fill_value=0.0)(common_wl)

            def lp_curve(ext):
                if ext == "(None)": return np.ones_like(common_wl)
                wl_lp = self.load_any(ext, "WAVELENGTH"); tr_lp = self.load_any(ext, "THROUGHPUT")
                return interp1d(wl_lp, tr_lp, bounds_error=False, fill_value=0.0)(common_wl)

            T_LP = lp_curve(lp_ext1) * lp_curve(lp_ext2)
            T_sys = filt_common * sens_common * T_LP
            resp  = fl_common * T_sys

            rgb_data[color] = {"filter_name": filt_name, "T_sys": T_sys, "response": resp}

        mag_texts = []
        if "A0V" in self.pickles_templates:
            wl_veg = self.load_any("A0V", "WAVELENGTH")
            fl_veg = self.load_any("A0V", "FLUX")
            fl_veg_c = interp1d(wl_veg, fl_veg, kind="linear", bounds_error=False, fill_value=0.0)(common_wl)
            for color in ("red","green","blue"):
                data = rgb_data[color]
                if data is not None:
                    S_star = _trapz(data["response"], x=common_wl)
                    S_veg  = _trapz(fl_veg_c * data["T_sys"], x=common_wl)
                    if S_veg>0 and S_star>0:
                        mag = -2.5 * np.log10(S_star / S_veg)
                        mag_texts.append(f"{color[0].upper()}→{data['filter_name']}: {mag:.2f}")
                    else:
                        mag_texts.append(f"{color[0].upper()}→{data['filter_name']}: N/A")
        title_text = " | ".join(mag_texts) if mag_texts else self.tr("No channels selected")

        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        ax1.plot(common_wl, fl_common, color="black", linewidth=1, label=f"{star_ext} SED")
        for color, data in rgb_data.items():
            if data is not None:
                ax1.plot(common_wl, data["response"], color="gold", linewidth=1.5, label=self.tr("{0} Response").format(color.upper()))
        ax1.set_xlim(wl_min, wl_max); ax1.set_xlabel(self.tr("Wavelength (Å)"))
        ax1.set_ylabel(self.tr("Flux (erg s⁻¹ cm⁻² Å⁻¹)"), color="black"); ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        ax2.set_ylabel(self.tr("Relative Throughput"), color="red"); ax2.tick_params(axis="y", labelcolor="red"); ax2.set_ylim(0.0, 1.0)
        if rgb_data["red"] is not None:   ax2.plot(common_wl, rgb_data["red"]["T_sys"],   color="red",   linestyle="--", linewidth=1, label=self.tr("R filter×QE"))
        if rgb_data["green"] is not None: ax2.plot(common_wl, rgb_data["green"]["T_sys"], color="green", linestyle="--", linewidth=1, label=self.tr("G filter×QE"))
        if rgb_data["blue"] is not None:  ax2.plot(common_wl, rgb_data["blue"]["T_sys"],  color="blue",  linestyle="--", linewidth=1, label=self.tr("B filter×QE"))

        ax1.grid(True, which="both", linestyle="--", alpha=0.3); self.figure.suptitle(title_text, fontsize=10)
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        self.canvas.draw()

def _sfcc_status(self, msg: str):
    # Prefer label if present, else print
    try:
        if getattr(self, "count_label", None) is not None:
            self.count_label.setText(msg)
    except Exception:
        pass
    print(msg)
    try:
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
    except Exception:
        pass


def _sfcc_busy(self, on: bool, msg: str | None = None):
    """Indeterminate progress if you have a QProgressBar named progress_bar (or similar)."""
    if msg:
        _sfcc_status(self, msg)

    pb = getattr(self, "progress_bar", None) or getattr(self, "progress", None)
    if pb is None:
        return

    try:
        pb.setVisible(on)
        if on:
            pb.setRange(0, 0)   # indeterminate
        else:
            pb.setRange(0, 100)
            pb.setValue(0)
    except Exception:
        pass


def _gaiaxp_synth_bvr(self, source_ids, *, status_cb=None, username=None, password=None, cache_dir=None):
    """
    Returns {source_id: {"B":..., "V":..., "R":...}} using GaiaXPy generate() with JKC system.
    """
    if status_cb is None:
        status_cb = lambda s: None

    # Local-only suppression: kill socket/file ResourceWarnings spam in this worker path
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Prevent TeX related noise if any plot code runs somewhere
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.rcParams["text.usetex"] = False
    except Exception:
        pass

    try:
        from gaiaxpy import generate
        from gaiaxpy.generator.photometric_system import PhotometricSystem
    except Exception:
        return {}

    ids = [int(x) for x in source_ids if x is not None]
    ids = list(dict.fromkeys(ids))
    if not ids:
        return {}

    if cache_dir:
        os.environ.setdefault("GAIAXPY_CACHE_DIR", str(cache_dir))

    status_cb(f"[SPCC] Gaia XP: generating synthetic JKC photometry for {len(ids)} Gaia sources…")

    df_parts = []
    total = len(ids)
    chunk = 10  # or 50

    for i in range(0, total, chunk):
        sub = ids[i:i+chunk]
        status_cb(f"[SPCC] Gaia XP: synthetic photometry {i+1}-{min(i+len(sub), total)} of {total}…")
        try:
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
        except Exception:
            pass

        with suppress_gaia_archive_banner():
            df_i = generate(
                sub,
                photometric_system=PhotometricSystem.JKC,
                save_file=False,
                username=username,
                password=password,
            )
        if df_i is not None and not getattr(df_i, "empty", False):
            df_parts.append(df_i)

    df = pd.concat(df_parts, ignore_index=True) if df_parts else None

    if df is None or getattr(df, "empty", False):
        return {}

    # Figure out source-id column
    sid_col = None
    for c in df.columns:
        if c.lower() in ("source_id", "sourceid", "gaia_source_id"):
            sid_col = c
            break
    if sid_col is None:
        for c in df.columns:
            cl = c.lower()
            if "source" in cl and "id" in cl:
                sid_col = c
                break
    if sid_col is None:
        return {}

    def pick_col(band: str):
        b = band.lower()
        # prefer common patterns
        for key in (f"{b}_mag", f"mag_{b}", f"{b}mag"):
            for c in df.columns:
                if c.lower() == key:
                    return c
        # fallback: anything containing band and 'mag' but not err
        for c in df.columns:
            cl = c.lower()
            if b in cl and "mag" in cl and "err" not in cl and "error" not in cl:
                return c
        return None

    b_col = pick_col("B")
    v_col = pick_col("V")
    r_col = pick_col("R")
    if not (b_col and v_col and r_col):
        return {}

    out = {}
    for _, row in df.iterrows():
        try:
            sid = int(row[sid_col])
            out[sid] = {"B": float(row[b_col]), "V": float(row[v_col]), "R": float(row[r_col])}
        except Exception:
            continue

    status_cb(f"[SPCC] Gaia XP: synthetic B/V/R available for {len(out)} sources")
    return out



def _infer_letter_from_bv(bv):
    if bv is None:
        return None
    try:
        bv = float(bv)
    except Exception:
        return None
    if not np.isfinite(bv):
        return None
    if bv < 0.00:
        return "B"
    elif bv < 0.30:
        return "A"
    elif bv < 0.58:
        return "F"
    elif bv < 0.81:
        return "G"
    elif bv < 1.40:
        return "K"
    else:
        return "M"


# ──────────────────────────────────────────────────────────────────────────────
# SFCC Dialog (rewired for "current view")
# ──────────────────────────────────────────────────────────────────────────────
class SFCCDialog(QDialog):
    """
    Spectral Flux Color Calibration dialog, adapted for SASpro's current view.
    Pass a 'view' adapter providing:
       - get_rgb_image(), set_rgb_image(...)
       - get_metadata()  [optional]
       - get_header()    [preferred for WCS; else we look in metadata]
    """
    def __init__(self, doc_manager, sasp_data_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Spectral Photometric Flux Color Calibration"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions        
        self.setMinimumSize(800, 600)

        self.doc_manager = doc_manager
        self.sasp_data_path = sasp_data_path
        self.user_custom_path = self._ensure_user_custom_fits()
        self.current_image = None
        self.current_header = None
        self.orientation_label = QLabel(self.tr("Orientation: N/A"))
        self.sasp_viewer_window = None
        self.main_win = parent

        # user custom file init … (unchanged)
        # ...
        self._reload_hdu_lists()
        self.star_list = []
        self._build_ui()
        self.load_settings()

        # persist combobox choices
        self.r_filter_combo.currentIndexChanged.connect(self.save_r_filter_setting)
        self.g_filter_combo.currentIndexChanged.connect(self.save_g_filter_setting)
        self.b_filter_combo.currentIndexChanged.connect(self.save_b_filter_setting)
        self.lp_filter_combo.currentIndexChanged.connect(self.save_lp_setting)
        self.lp_filter_combo2.currentIndexChanged.connect(self.save_lp2_setting)
        self.sens_combo.currentIndexChanged.connect(self.save_sensor_setting)
        self.star_combo.currentIndexChanged.connect(self.save_star_setting)
        self.finished.connect(lambda *_: self._cleanup())

        self.grad_method = "poly3"
        self.grad_method_combo.currentTextChanged.connect(lambda m: setattr(self, "grad_method", m))

    # ── View plumbing ───────────────────────────────────────────────────
    def _get_active_image_and_header(self):
        doc = self.doc_manager.get_active_document()
        if doc is None:
            return None, None, None

        img = doc.image
        meta = doc.metadata or {}

        # Prefer the normalized WCS header if present, then fall back
        hdr = (
            meta.get("wcs_header") or
            meta.get("original_header") or
            meta.get("header")
        )

        return img, hdr, meta

    
    def _get_img_meta(self) -> Tuple[Optional[np.ndarray], dict]:
        """Try a few common shapes to obtain image + metadata from the view."""
        meta = {}
        img = None
        if hasattr(self.view, "get_image_and_metadata"):
            try:
                img, meta = self.view.get_image_and_metadata()
            except Exception:
                pass
        if img is None and hasattr(self.view, "get_rgb_image"):
            img = self.view.get_rgb_image()
        if not meta and hasattr(self.view, "get_metadata"):
            try:
                meta = self.view.get_metadata() or {}
            except Exception:
                meta = {}
        return img, (meta or {})

    def _get_header(self):
        header = None
        if hasattr(self.view, "get_header"):
            try:
                header = self.view.get_header()
            except Exception:
                header = None
        if header is None:
            # fall back to metadata
            _, meta = self._get_img_meta()
            header = meta.get("original_header") or meta.get("header")
        return header

    def _push_image(self, img: np.ndarray, meta: Optional[dict], step_name: str):
        """Send image back to the same current view."""
        if hasattr(self.view, "set_rgb_image"):
            self.view.set_rgb_image(img, meta or {}, step_name)
        elif hasattr(self.view, "set_image"):
            self.view.set_image(img, meta or {}, step_name=step_name)
        elif hasattr(self.view, "update_image"):
            self.view.update_image(img, meta or {}, step_name=step_name)
        else:
            # As a last resort, try attribute assignment (for custom apps)
            if hasattr(self.view, "image"):
                self.view.image = img
            if hasattr(self.view, "metadata"):
                self.view.metadata = meta or {}

    # ── File prep ───────────────────────────────────────────────────────

    def _ensure_user_custom_fits(self) -> str:
        app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        os.makedirs(app_data, exist_ok=True)
        path = os.path.join(app_data, "usercustomcurves.fits")
        if not os.path.exists(path):
            fits.HDUList([fits.PrimaryHDU()]).writeto(path)
        return path

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout(); layout.addLayout(row1)
        self.fetch_stars_btn = QPushButton(self.tr("Step 1: Fetch Stars from Current View"))
        f = self.fetch_stars_btn.font(); f.setBold(True); self.fetch_stars_btn.setFont(f)
        self.fetch_stars_btn.clicked.connect(self.fetch_stars)
        row1.addWidget(self.fetch_stars_btn)

        self.open_sasp_btn = QPushButton(self.tr("Open SASP Viewer"))
        self.open_sasp_btn.clicked.connect(self.open_sasp_viewer)
        row1.addWidget(self.open_sasp_btn)

        row1.addSpacing(20)
        row1.addWidget(QLabel(self.tr("Select White Reference:")))
        self.star_combo = QComboBox()
        self.star_combo.addItem(self.tr("Vega (A0V)"), userData="A0V")
        for sed in getattr(self, "sed_list", []):
            if sed.upper() == "A0V": continue
            self.star_combo.addItem(sed, userData=sed)
        row1.addWidget(self.star_combo)
        idx_g2v = self.star_combo.findData("G2V")
        if idx_g2v >= 0: self.star_combo.setCurrentIndex(idx_g2v)

        row2 = QHBoxLayout(); layout.addLayout(row2)
        row2.addWidget(QLabel(self.tr("R Filter:")))
        self.r_filter_combo = QComboBox(); self.r_filter_combo.addItem("(None)"); self.r_filter_combo.addItems(self.filter_list); row2.addWidget(self.r_filter_combo)
        row2.addSpacing(20); row2.addWidget(QLabel(self.tr("G Filter:")))
        self.g_filter_combo = QComboBox(); self.g_filter_combo.addItem("(None)"); self.g_filter_combo.addItems(self.filter_list); row2.addWidget(self.g_filter_combo)
        row2.addSpacing(20); row2.addWidget(QLabel(self.tr("B Filter:")))
        self.b_filter_combo = QComboBox(); self.b_filter_combo.addItem("(None)"); self.b_filter_combo.addItems(self.filter_list); row2.addWidget(self.b_filter_combo)

        row3 = QHBoxLayout(); layout.addLayout(row3)
        row3.addStretch()
        row3.addWidget(QLabel(self.tr("Sensor (QE):")))
        self.sens_combo = QComboBox(); self.sens_combo.addItem("(None)"); self.sens_combo.addItems(self.sensor_list); row3.addWidget(self.sens_combo)
        row3.addSpacing(20); row3.addWidget(QLabel(self.tr("LP/Cut Filter1:")))
        self.lp_filter_combo = QComboBox(); self.lp_filter_combo.addItem("(None)"); self.lp_filter_combo.addItems(self.filter_list); row3.addWidget(self.lp_filter_combo)
        row3.addSpacing(20); row3.addWidget(QLabel(self.tr("LP/Cut Filter2:")))
        self.lp_filter_combo2 = QComboBox(); self.lp_filter_combo2.addItem("(None)"); self.lp_filter_combo2.addItems(self.filter_list); row3.addWidget(self.lp_filter_combo2)
        row3.addStretch()

        row4 = QHBoxLayout(); layout.addLayout(row4)
        self.run_spcc_btn = QPushButton(self.tr("Step 2: Run Color Calibration"))
        f2 = self.run_spcc_btn.font(); f2.setBold(True); self.run_spcc_btn.setFont(f2)
        self.run_spcc_btn.clicked.connect(self.run_spcc)
        row4.addWidget(self.run_spcc_btn)

        self.neutralize_chk = QCheckBox(self.tr("Background Neutralization")); self.neutralize_chk.setChecked(False); row4.addWidget(self.neutralize_chk)

        self.run_grad_btn = QPushButton(self.tr("Run Gradient Extraction (Beta)"))
        f3 = self.run_grad_btn.font(); f3.setBold(True); self.run_grad_btn.setFont(f3)
        self.run_grad_btn.clicked.connect(self.run_gradient_extraction)
        row4.addWidget(self.run_grad_btn)

        self.grad_method_combo = QComboBox(); self.grad_method_combo.addItems(["poly2","poly3","rbf"]); self.grad_method_combo.setCurrentText("poly3")
        row4.addWidget(self.grad_method_combo)

        row4.addSpacing(15)
        row4.addWidget(QLabel(self.tr("Star detect σ:")))
        self.sep_thr_spin = QSpinBox()
        self.sep_thr_spin.setRange(2, 50)        # should be enough
        self.sep_thr_spin.setValue(5)            # our current hardcoded value
        self.sep_thr_spin.valueChanged.connect(self.save_sep_threshold_setting)
        row4.addWidget(self.sep_thr_spin)
        self.gaia_fallback_chk = QCheckBox(self.tr("Gaia XP fallback (slower, fills missing data)"))
        self.gaia_fallback_chk.setChecked(True)  # default ON
        self.gaia_fallback_chk.toggled.connect(self.save_gaia_fallback_setting)
        row4.addWidget(self.gaia_fallback_chk)
        row4.addStretch()
        self.add_curve_btn = QPushButton(self.tr("Add Custom Filter/Sensor Curve…"))
        self.add_curve_btn.clicked.connect(self.add_custom_curve); row4.addWidget(self.add_curve_btn)
        self.remove_curve_btn = QPushButton(self.tr("Remove Filter/Sensor Curve…"))
        self.remove_curve_btn.clicked.connect(self.remove_custom_curve); row4.addWidget(self.remove_curve_btn)
        row4.addStretch()
        self.close_btn = QPushButton(self.tr("Close")); self.close_btn.clicked.connect(self.reject); row4.addWidget(self.close_btn)

        self.count_label = QLabel(""); layout.addWidget(self.count_label)

        self.figure = Figure(figsize=(6, 4)); self.canvas = FigureCanvas(self.figure); self.canvas.setVisible(False); layout.addWidget(self.canvas, stretch=1)
        self.reset_btn = QPushButton(self.tr("Reset View/Close")); self.reset_btn.clicked.connect(self.reject); layout.addWidget(self.reset_btn)

        # hide gradient controls by default (enable if you like)
        self.run_grad_btn.hide(); self.grad_method_combo.hide()
        layout.addWidget(self.orientation_label)

    # ── Settings helpers ────────────────────────────────────────────────

    def _reload_hdu_lists(self):
        self.sed_list = []
        with fits.open(self.sasp_data_path, mode="readonly", memmap=False) as base:
            for hdu in base:
                if isinstance(hdu, fits.BinTableHDU) and hdu.header.get("CTYPE","").upper()=="SED":
                    self.sed_list.append(hdu.header["EXTNAME"])

        self.filter_list = []; self.sensor_list = []
        for path in (self.sasp_data_path, self.user_custom_path):
            with fits.open(path, mode="readonly", memmap=False) as hdul:
                for hdu in hdul:
                    if not isinstance(hdu, fits.BinTableHDU): continue
                    c = hdu.header.get("CTYPE","").upper(); e = hdu.header.get("EXTNAME","")
                    if c=="FILTER": self.filter_list.append(e)
                    elif c=="SENSOR": self.sensor_list.append(e)
        self.sed_list.sort(); self.filter_list.sort(); self.sensor_list.sort()

    def load_settings(self):
        s = QSettings()
        def apply(cb, key):
            val = s.value(key, "")
            if val:
                idx = cb.findText(val)
                if idx != -1:
                    cb.setCurrentIndex(idx)

        # existing stuff...
        saved_star = QSettings().value("SFCC/WhiteReference", "")
        if saved_star:
            idx = self.star_combo.findText(saved_star)
            if idx != -1:
                self.star_combo.setCurrentIndex(idx)
        use_gaia = s.value("SFCC/UseGaiaFallback", True, type=bool)
        if hasattr(self, "gaia_fallback_chk"):
            self.gaia_fallback_chk.setChecked(bool(use_gaia))
        apply(self.r_filter_combo, "SFCC/RFilter")
        apply(self.g_filter_combo, "SFCC/GFilter")
        apply(self.b_filter_combo, "SFCC/BFilter")
        apply(self.sens_combo,     "SFCC/Sensor")
        apply(self.lp_filter_combo,  "SFCC/LPFilter")
        apply(self.lp_filter_combo2, "SFCC/LPFilter2")

        # 👇 NEW: load SEP/star-detect threshold
        sep_thr = int(s.value("SFCC/SEPThreshold", 5))
        if hasattr(self, "sep_thr_spin"):
            self.sep_thr_spin.setValue(sep_thr)

    def save_sep_threshold_setting(self, v: int):
        QSettings().setValue("SFCC/SEPThreshold", int(v))

    def save_lp_setting(self, _):  QSettings().setValue("SFCC/LPFilter", self.lp_filter_combo.currentText())
    def save_lp2_setting(self, _): QSettings().setValue("SFCC/LPFilter2", self.lp_filter_combo2.currentText())
    def save_star_setting(self, _): QSettings().setValue("SFCC/WhiteReference", self.star_combo.currentText())
    def save_r_filter_setting(self, _): QSettings().setValue("SFCC/RFilter", self.r_filter_combo.currentText())
    def save_g_filter_setting(self, _): QSettings().setValue("SFCC/GFilter", self.g_filter_combo.currentText())
    def save_b_filter_setting(self, _): QSettings().setValue("SFCC/BFilter", self.b_filter_combo.currentText())
    def save_sensor_setting(self, _):   QSettings().setValue("SFCC/Sensor", self.sens_combo.currentText())
    def save_gaia_fallback_setting(self, v: bool):
        QSettings().setValue("SFCC/UseGaiaFallback", bool(v))

    # ── Curve utilities ─────────────────────────────────────────────────
    def _use_gaia_fallback(self) -> bool:
        try:
            if hasattr(self, "gaia_fallback_chk") and not self.gaia_fallback_chk.isChecked():
                return False
        except Exception:
            return False
        return bool(self._gaia_enabled())


    def interpolate_bad_points(self, wl, tr):
        tr = tr.copy()
        bad = (tr < 0.0) | (tr > 1.0)
        good = ~bad
        if not np.any(bad): return tr, np.array([], dtype=int)
        if np.sum(good) < 2: raise RuntimeError("Not enough valid points to interpolate anomalies.")
        tr_corr = tr.copy()
        tr_corr[bad] = np.interp(wl[bad], wl[good], tr[good])
        return tr_corr, np.where(bad)[0]

    def smooth_curve(self, tr, window_size=5):
        return medfilt(tr, kernel_size=window_size)

    def get_calibration_points(self, rgb_img: np.ndarray):
        print("\nClick three calibration points: BL (λmin,0), BR (λmax,0), TL (λmin,1)")
        fig, ax = plt.subplots(figsize=(8, 5)); ax.imshow(rgb_img); ax.set_title(self.tr("Click 3 points, then close"))
        pts = plt.ginput(3, timeout=-1); plt.close(fig)
        if len(pts) != 3: raise RuntimeError(self.tr("Need exactly three clicks for calibration."))
        return pts[0], pts[1], pts[2]

    def build_transforms(self, px_bl, py_bl, px_br, py_br, px_tl, py_tl, λ_min, λ_max, resp_min, resp_max):
        nm_per_px = (λ_max - λ_min) / (px_br - px_bl)
        resp_per_px = (resp_max - resp_min) / (py_bl - py_tl)
        def px_to_λ(px):  return λ_min + (px - px_bl) * nm_per_px
        def py_to_resp(py): return resp_max - (py - py_tl) * resp_per_px
        return px_to_λ, py_to_resp

    def extract_curve(self, gray_img, λ_mapper, resp_mapper, λ_min, λ_max, threshold=50):
        H, W = gray_img.shape
        data = []
        for px in range(W):
            col = gray_img[:, px]
            py_min = int(np.argmin(col)); val_min = int(col[py_min])
            if val_min < threshold:
                lam = λ_mapper(px)
                if λ_min <= lam <= λ_max:
                    data.append((lam, resp_mapper(py_min)))
        if not data:
            raise RuntimeError("No dark pixels found; raise threshold or adjust clicks.")
        df = (pd.DataFrame(data, columns=["wavelength_nm", "response"])
                .sort_values("wavelength_nm").reset_index(drop=True))
        df = df[(df["wavelength_nm"] >= λ_min) & (df["wavelength_nm"] <= λ_max)].copy()
        if df["wavelength_nm"].iloc[0] > λ_min:
            df = pd.concat([pd.DataFrame([[λ_min, 0.0]], columns=["wavelength_nm", "response"]), df], ignore_index=True)
        if df["wavelength_nm"].iloc[-1] < λ_max:
            df = pd.concat([df, pd.DataFrame([[λ_max, 0.0]], columns=["wavelength_nm", "response"])], ignore_index=True)
        return df.sort_values("wavelength_nm").reset_index(drop=True)

    def _query_name_channel(self):
        name_str, ok1 = QInputDialog.getText(self, self.tr("Curve Name"), self.tr("Enter curve name (EXTNAME):"))
        if not (ok1 and name_str.strip()): return False, None, None
        extname = name_str.strip().upper().replace(" ", "_")
        ch_str, ok2 = QInputDialog.getText(self, self.tr("Channel"), self.tr("Enter channel (R,G,B or Q for sensor):"))
        if not (ok2 and ch_str.strip()): return False, None, None
        return True, extname, ch_str.strip().upper()

    def _append_curve_hdu(self, wl_ang, tr_final, extname, ctype, origin):
        col_wl = fits.Column(name="WAVELENGTH", format="E", unit="Angstrom", array=wl_ang.astype(np.float32))
        col_tr = fits.Column(name="THROUGHPUT", format="E", unit="REL",      array=tr_final.astype(np.float32))
        new_hdu = fits.BinTableHDU.from_columns([col_wl, col_tr])
        new_hdu.header["EXTNAME"] = extname
        new_hdu.header["CTYPE"]   = ctype
        new_hdu.header["ORIGIN"]  = origin
        with fits.open(self.user_custom_path, mode="update", memmap=False) as hdul:
            hdul.append(new_hdu); hdul.flush()

    def add_custom_curve(self):
        msg = QMessageBox(self); msg.setWindowTitle(self.tr("Add Custom Curve")); msg.setText(self.tr("Choose how to add the curve:"))
        csv_btn = msg.addButton(self.tr("Import CSV"), QMessageBox.ButtonRole.AcceptRole)
        img_btn = msg.addButton(self.tr("Digitize Image"), QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        if msg.clickedButton() == csv_btn: self._import_curve_from_csv()
        elif msg.clickedButton() == img_btn: self._digitize_curve_from_image()

    def _import_curve_from_csv(self):
        csv_path, _ = QFileDialog.getOpenFileName(self, self.tr("Select 2-column CSV (λ_nm, response)"), "", "CSV Files (*.csv);;All Files (*)")
        if not csv_path: return
        try:
            df = (pd.read_csv(csv_path, comment="#", header=None).iloc[:, :2].dropna())
            df.columns = ["wavelength_nm","response"]
            wl_nm = df["wavelength_nm"].astype(float).to_numpy(); tp = df["response"].astype(float).to_numpy()
        except ValueError:
            try:
                df = (pd.read_csv(csv_path, comment="#", header=0).iloc[:, :2].dropna())
                df.columns = ["wavelength_nm","response"]
                wl_nm = df["wavelength_nm"].astype(float).to_numpy(); tp = df["response"].astype(float).to_numpy()
            except Exception as e2:
                QMessageBox.critical(self, self.tr("CSV Error"), self.tr("Could not read CSV:\n{0}").format(e2)); return
        except Exception as e:
            QMessageBox.critical(self, self.tr("CSV Error"), self.tr("Could not read CSV:\n{0}").format(e)); return

        ok, extname_base, channel_val = self._query_name_channel()
        if not ok: return
        wl_ang = (wl_nm * 10.0).astype(np.float32); tr_final = tp.astype(np.float32)
        self._append_curve_hdu(wl_ang, tr_final, extname_base, "SENSOR" if channel_val=="Q" else "FILTER", f"CSV:{os.path.basename(csv_path)}")
        self._reload_hdu_lists(); self.refresh_filter_sensor_lists()
        QMessageBox.information(self, self.tr("Done"), self.tr("CSV curve '{0}' added.").format(extname_base))

    def _digitize_curve_from_image(self):
        img_path_str, _ = QFileDialog.getOpenFileName(self, self.tr("Select Curve Image to Digitize"), "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if not img_path_str: return
        img_filename = os.path.basename(img_path_str)
        try:
            bgr = cv2.imread(img_path_str)
            if bgr is None: raise RuntimeError(f"cv2.imread returned None for '{img_path_str}'")
            rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Could not load image:\n{0}").format(e)); return

        try:
            (px_bl, py_bl), (px_br, py_br), (px_tl, py_tl) = self.get_calibration_points(rgb_img)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Digitization Error"), str(e)); return

        λ_min_str, ok1 = QInputDialog.getText(self, self.tr("λ_min"), self.tr("Enter λ_min (in nm):"))
        λ_max_str, ok2 = QInputDialog.getText(self, self.tr("λ_max"), self.tr("Enter λ_max (in nm):"))
        if not (ok1 and ok2 and λ_min_str.strip() and λ_max_str.strip()): return
        try:
            λ_min = float(λ_min_str); λ_max = float(λ_max_str)
        except ValueError:
            QMessageBox.critical(self, self.tr("Input Error"), self.tr("λ_min and λ_max must be numbers.")); return

        ok, extname_base, channel_val = self._query_name_channel()
        if not ok: return

        px_to_λ, py_to_resp = self.build_transforms(px_bl, py_bl, px_br, py_br, px_tl, py_tl, λ_min, λ_max, 0.0, 1.0)
        try:
            df_curve = self.extract_curve(gray_img, px_to_λ, py_to_resp, λ_min, λ_max, threshold=50)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Extraction Error"), str(e)); return

        df_curve["wl_int"] = df_curve["wavelength_nm"].round().astype(int)
        grp = (df_curve.groupby("wl_int")["response"].median().reset_index().sort_values("wl_int"))
        wl = grp["wl_int"].to_numpy(dtype=int); tr = grp["response"].to_numpy(dtype=float)

        try:
            tr_corr, _ = self.interpolate_bad_points(wl, tr)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Interpolation Error"), str(e)); return

        tr_smoothed = self.smooth_curve(tr_corr, window_size=5)
        wl_ang = (wl.astype(float) * 10.0).astype(np.float32); tr_final = tr_smoothed.astype(np.float32)
        self._append_curve_hdu(wl_ang, tr_final, extname_base, "SENSOR" if channel_val=="Q" else "FILTER", f"UserDefined:{img_filename}")
        self._reload_hdu_lists(); self.refresh_filter_sensor_lists()
        QMessageBox.information(self, self.tr("Done"), self.tr("Added curve '{0}'.").format(extname_base))

    def remove_custom_curve(self):
        all_curves = self.filter_list + self.sensor_list
        if not all_curves:
            QMessageBox.information(self, self.tr("Remove Curve"), self.tr("No custom curves to remove.")); return
        curve, ok = QInputDialog.getItem(self, self.tr("Remove Curve"), self.tr("Select a FILTER or SENSOR curve to delete:"), all_curves, 0, False)
        if not ok or not curve: return
        reply = QMessageBox.question(self, self.tr("Confirm Deletion"), self.tr("Delete '{0}'?").format(curve), QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes: return

        temp_path = self.user_custom_path + ".tmp"
        try:
            with fits.open(self.user_custom_path, memmap=False) as old_hdul:
                new_hdus = []
                for hdu in old_hdul:
                    if hdu is old_hdul[0]:
                        new_hdus.append(hdu.copy())
                    else:
                        if hdu.header.get("EXTNAME") != curve:
                            new_hdus.append(hdu.copy())
            fits.HDUList(new_hdus).writeto(temp_path, overwrite=True)
            os.replace(temp_path, self.user_custom_path)
        except Exception as e:
            if os.path.exists(temp_path): os.remove(temp_path)
            QMessageBox.critical(self, "Write Error", f"Could not remove curve:\n{e}"); return

        self._reload_hdu_lists(); self.refresh_filter_sensor_lists()
        QMessageBox.information(self, self.tr("Removed"), self.tr("Deleted curve '{0}'.").format(curve))

    def refresh_filter_sensor_lists(self):
        self._reload_hdu_lists()
        current_r = self.r_filter_combo.currentText()
        current_g = self.g_filter_combo.currentText()
        current_b = self.b_filter_combo.currentText()
        current_s = self.sens_combo.currentText()
        current_lp  = self.lp_filter_combo.currentText()
        current_lp2 = self.lp_filter_combo2.currentText()

        for cb, lst, prev in [
            (self.r_filter_combo, self.filter_list, current_r),
            (self.g_filter_combo, self.filter_list, current_g),
            (self.b_filter_combo, self.filter_list, current_b),
        ]:
            cb.clear(); cb.addItem("(None)"); cb.addItems(lst)
            idx = cb.findText(prev);  cb.setCurrentIndex(idx if idx != -1 else 0)

        for cb, prev in [(self.lp_filter_combo, current_lp), (self.lp_filter_combo2, current_lp2)]:
            cb.clear(); cb.addItem("(None)"); cb.addItems(self.filter_list)
            idx = cb.findText(prev);  cb.setCurrentIndex(idx if idx != -1 else 0)

        self.sens_combo.clear(); self.sens_combo.addItem("(None)"); self.sens_combo.addItems(self.sensor_list)
        idx = self.sens_combo.findText(current_s); self.sens_combo.setCurrentIndex(idx if idx != -1 else 0)

    # ── WCS utilities ──────────────────────────────────────────────────
    def calculate_orientation(self, header):
        try:
            cd1_1 = float(header.get("CD1_1", 0.0))
            cd1_2 = float(header.get("CD1_2", 0.0))
            return math.degrees(math.atan2(cd1_2, cd1_1))
        except Exception:
            return None

    def calculate_ra_dec_from_pixel(self, x, y):
        if not hasattr(self, "wcs"): return None, None
        return self.wcs.all_pix2world(x, y, 0)

    # ── Background neutralization ───────────────────────────────────────
    def _neutralize_background(self, rgb_f: np.ndarray, *, remove_pedestal: bool = False) -> np.ndarray:
        img = np.asarray(rgb_f, dtype=np.float32)

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Expected RGB image (H,W,3)")

        img = np.clip(img, 0.0, 1.0)

        try:
            rect = auto_rect_50x50(img)  # same SASv2-ish auto finder
            out = background_neutralize_rgb(
                img,
                rect,
                mode="pivot1",                # or "offset" if you prefer
                remove_pedestal=remove_pedestal,
            )
            return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

        except Exception as e:
            print(f"[SPCC] BN preset failed, falling back to simple neutralization: {e}")
            return self._neutralize_background_simple(img, patch_size=10)

    def _neutralize_background_simple(self, rgb_f: np.ndarray, patch_size: int = 50) -> np.ndarray:
        """
        Simple neutralization: find darkest patch by summed medians,
        then equalize channel medians around the mean.
        Assumes rgb_f is float in [0,1] with no negatives.
        """
        img = np.asarray(rgb_f, dtype=np.float32).copy()
        h, w = img.shape[:2]
        ph, pw = max(1, h // patch_size), max(1, w // patch_size)

        min_sum, best_med = np.inf, None
        for i in range(patch_size):
            for j in range(patch_size):
                y0, x0 = i * ph, j * pw
                patch = img[y0:min(y0 + ph, h), x0:min(x0 + pw, w), :]
                if patch.size == 0:
                    continue
                med = np.median(patch, axis=(0, 1))
                s = float(med.sum())
                if s < min_sum:
                    min_sum, best_med = s, med

        if best_med is None:
            return np.clip(img, 0.0, 1.0)

        target = float(best_med.mean())
        eps = 1e-8
        for c in range(3):
            diff = float(best_med[c] - target)
            if abs(diff) < eps:
                continue
            # Preserve [0,1] scale; keep the same form you were using.
            img[..., c] = np.clip((img[..., c] - diff) / (1.0 - diff), 0.0, 1.0)

        return np.clip(img, 0.0, 1.0)

    def _make_working_base_for_sep(self, img_float: np.ndarray) -> np.ndarray:
        """
        Build a working copy for SEP.

        IMPORTANT:
        - Do NOT do per-channel pedestal removal if you use annulus background subtraction.
        - Keep values stable for SEP by clamping to [0,1] (optional).
        """
        base = np.asarray(img_float, dtype=np.float32, order="C").copy()

        if base.ndim != 3 or base.shape[2] != 3:
            raise ValueError("Expected RGB image (H,W,3)")

        # Optional: clamp only (no shifting)
        base = np.clip(base, 0.0, 1.0)

        return base



    # ── SIMBAD/Star fetch ──────────────────────────────────────────────
    def initialize_wcs_from_header(self, header):
        """
        Build a robust 2D celestial WCS from the provided header.

        - Normalizes deprecated RADECSYS/EPOCH keywords.
        - Uses relax=True.
        - Stores:
            self.wcs (WCS)
            self.wcs_header (fits.Header)
            self.pixscale (arcsec/pixel approx)
            self.center_ra, self.center_dec (deg)
            self.orientation (deg, if derivable)
        """
        if header is None:
            print("[SPCC] No FITS header available; cannot build WCS.")
            self.wcs = None
            return

        try:
            hdr = header.copy()

            # --- normalize deprecated keywords ---
            if "RADECSYS" in hdr and "RADESYS" not in hdr:
                radesys_val = str(hdr["RADECSYS"]).strip()
                hdr["RADESYS"] = radesys_val
                try:
                    del hdr["RADECSYS"]
                except Exception:
                    pass

                # Carry to alternate WCS letters if present (CTYPE1A, CTYPE2A, etc.)
                alt_letters = {
                    k[-1]
                    for k in hdr.keys()
                    if re.match(r"^CTYPE[12][A-Z]$", k)
                }
                for a in alt_letters:
                    key = f"RADESYS{a}"
                    if key not in hdr:
                        hdr[key] = radesys_val

            if "EPOCH" in hdr and "EQUINOX" not in hdr:
                hdr["EQUINOX"] = hdr["EPOCH"]
                try:
                    del hdr["EPOCH"]
                except Exception:
                    pass

            # Build WCS
            self.wcs = WCS(hdr, naxis=2, relax=True)

            # Pixel scale estimate (arcsec/px) from pixel_scale_matrix if available
            try:
                psm = self.wcs.pixel_scale_matrix
                self.pixscale = float(np.hypot(psm[0, 0], psm[1, 0]) * 3600.0)
            except Exception:
                self.pixscale = None

            # CRVAL center
            try:
                self.center_ra, self.center_dec = [float(x) for x in self.wcs.wcs.crval]
            except Exception:
                self.center_ra, self.center_dec = None, None

            # Save normalized header form
            try:
                self.wcs_header = self.wcs.to_header(relax=True)
            except Exception:
                self.wcs_header = None

            # Orientation (optional)
            if "CROTA2" in hdr:
                try:
                    self.orientation = float(hdr["CROTA2"])
                except Exception:
                    self.orientation = None
            else:
                self.orientation = self.calculate_orientation(hdr)

            if getattr(self, "orientation_label", None) is not None:
                if self.orientation is not None:
                    self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
                else:
                    self.orientation_label.setText("Orientation: N/A")

        except Exception as e:
            print("[SPCC] WCS initialization error:\n", e)
            self.wcs = None

    # ── Gaia XP fallback helpers ─────────────────────────────────────────

    def _gaia_db_path(self) -> str:
        """
        Local cache DB for Gaia XP spectra (sqlite).
        Stored in AppDataLocation so it persists.
        """
        app_data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        os.makedirs(app_data, exist_ok=True)
        return os.path.join(app_data, "gaia_xp_cache.sqlite")

    def _gaia_enabled(self) -> bool:
        """
        Gaia fallback requires:
          - astroquery.gaia (for the region query)
          - gaia_downloader available (for caching)
          - gaiaxpy installed (to calibrate XP spectra)  [strictly required to actually download spectra]
        """
        return (Gaia is not None) and (GaiaDownloader is not None) and bool(HAS_GAIAXPY)

    def _get_gaia_downloader(self):
        """
        Lazy init: downloader holds sqlite DB connection.
        """
        if not hasattr(self, "_gaia_dl") or self._gaia_dl is None:
            self._gaia_dl = GaiaDownloader(self._gaia_db_path())
        return self._gaia_dl

    def _query_gaia_sources_in_field(self, center_ra_deg: float, center_dec_deg: float, radius_deg: float,
                                     mag_limit: float = 15.5, max_rows: int = 200000) -> list[dict]:
        """
        Query Gaia DR3 sources in field that have XP continuous spectra.
        Returns dicts with: source_id, ra, dec, gmag, bp_rp
        """
        if Gaia is None:
            return []
        # Keep this safe: astroquery Gaia can be finicky if ROW_LIMIT not set
        Gaia.ROW_LIMIT = max_rows

        adql = f"""
        SELECT
            source_id, ra, dec, phot_g_mean_mag, bp_rp
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {center_ra_deg}, {center_dec_deg}, {radius_deg})
        )
        AND has_xp_continuous = 'True'
        AND phot_g_mean_mag <= {float(mag_limit)}
        """
        try:
            job = Gaia.launch_job_async(adql, dump_to_file=False)
            tab = job.get_results()
        except Exception as e:
            print(f"[SPCC] Gaia query failed: {e}")
            return []

        out = []
        for row in tab:
            try:
                out.append({
                    "source_id": int(row["source_id"]),
                    "ra": float(row["ra"]),
                    "dec": float(row["dec"]),
                    "gmag": float(row["phot_g_mean_mag"]) if row["phot_g_mean_mag"] is not None else None,
                    "bp_rp": float(row["bp_rp"]) if row["bp_rp"] is not None else None,
                })
            except Exception:
                continue
        return out

    @staticmethod
    def _angsep2_arcsec(ra1, dec1, ra2, dec2) -> float:
        """
        Small-angle approx angular separation in arcsec (good enough for <= few arcsec matching).
        """
        dra = (ra1 - ra2) * math.cos(math.radians((dec1 + dec2) * 0.5))
        ddec = (dec1 - dec2)
        return 3600.0 * math.hypot(dra, ddec)

    def _attach_gaia_ids_to_star_list(self, *, radius_deg: float, mag_limit: float = 15.5,
                                      max_match_arcsec: float = 1.0):
        """
        Enrich self.star_list entries with Gaia XP source_id when:
          - SIMBAD spectral type is missing OR
          - pickles_match is None (no Pickles template)
        """
        if not self._gaia_enabled():
            print("[SPCC] Gaia fallback disabled (missing astroquery.gaia and/or gaiaxpy and/or gaia_downloader).")
            return

        if not getattr(self, "center_ra", None) or not getattr(self, "center_dec", None):
            # initialize_wcs_from_header sets these; but be defensive
            return

        gaia_sources = self._query_gaia_sources_in_field(
            self.center_ra, self.center_dec, radius_deg, mag_limit=mag_limit
        )
        if not gaia_sources:
            return

        # Optional: store in the cache DB so spectra downloads can be linked to source metadata
        try:
            dl = self._get_gaia_downloader()
            dl.db.add_sources([dl.GaiaSource(
                s["source_id"], s["ra"], s["dec"], s["gmag"], s["bp_rp"], None, None, None
            ) for s in gaia_sources])
        except Exception:
            # ok if this fails; spectra caching still works
            pass

        # For matching: brute-force is fine at these sizes if SIMBAD stars small,
        # but we’ll still keep it simple and safe.
        for st in self.star_list:
            needs_gaia = (not st.get("sp_clean")) or (st.get("pickles_match") is None)
            if not needs_gaia:
                continue

            ra = float(st["ra"]); dec = float(st["dec"])
            best = None
            best_sep = 1e9

            for gs in gaia_sources:
                sep_as = self._angsep2_arcsec(ra, dec, gs["ra"], gs["dec"])
                if sep_as < best_sep:
                    best_sep = sep_as
                    best = gs

            if best is not None and best_sep <= float(max_match_arcsec):
                st["gaia_source_id"] = int(best["source_id"])
                st["gaia_bp_rp"] = best.get("bp_rp")
                st["gaia_gmag"] = best.get("gmag")
                st["gaia_sep_arcsec"] = float(best_sep)

    def _gaia_integrals_for_source_ids(
        self,
        source_ids: list[int],
        wl_grid_ang: np.ndarray,
        T_sys_R: np.ndarray,
        T_sys_G: np.ndarray,
        T_sys_B: np.ndarray,
        *,
        batch_size: int = 128,
    ) -> dict[int, tuple[float, float, float]]:
        """
        Ensure Gaia XP spectra exist in the local cache (download missing ones),
        then integrate Gaia flux * throughput over wl_grid_ang (Å).

        Also caches the synthetic photometry integrals (Sr,Sg,Sb) per (source_id, system_key)
        so subsequent runs are fast.

        Returns dict: source_id -> (S_R, S_G, S_B)
        """
        if not self._gaia_enabled():
            return {}

        dl = self._get_gaia_downloader()

        # ---- sanity on arrays (avoid subtle shape bugs) ----
        wl_grid_ang = np.asarray(wl_grid_ang, dtype=np.float64).reshape(-1)
        T_sys_R = np.asarray(T_sys_R, dtype=np.float64).reshape(-1)
        T_sys_G = np.asarray(T_sys_G, dtype=np.float64).reshape(-1)
        T_sys_B = np.asarray(T_sys_B, dtype=np.float64).reshape(-1)

        n = wl_grid_ang.size
        if not (T_sys_R.size == T_sys_G.size == T_sys_B.size == n):
            raise ValueError(
                f"Throughput arrays must match wl_grid_ang length. "
                f"len(wl)={n}, len(R)={T_sys_R.size}, len(G)={T_sys_G.size}, len(B)={T_sys_B.size}"
            )

        # Normalize + uniquify IDs
        source_ids_i = [int(s) for s in source_ids if s is not None]
        if not source_ids_i:
            return {}

        # Keep order stable but avoid redoing work for duplicates
        seen = set()
        ordered_unique: list[int] = []
        for sid in source_ids_i:
            if sid not in seen:
                seen.add(sid)
                ordered_unique.append(sid)

        # ---- synthetic photometry cache key (depends on wl grid + throughputs) ----
        system_key = dl.db.make_system_key(
            wl_grid_ang, T_sys_R, T_sys_G, T_sys_B, tag="spcc_v1"
        )

        # ---- pull cached integrals first ----
        out: dict[int, tuple[float, float, float]] = {}
        try:
            cached = dl.db.get_synth_phot(ordered_unique, system_key)
            out.update(cached)
        except Exception as e:
            print(f"[SPCC] Gaia synth cache read failed (continuing): {e}")
            cached = {}

        missing_integrals = [sid for sid in ordered_unique if sid not in out]
        if not missing_integrals:
            if getattr(self, "count_label", None) is not None:
                self.count_label.setText(f"Gaia XP: synthetic photometry cache hit ({len(out)} sources).")
            return out
        else:
            if getattr(self, "count_label", None) is not None:
                self.count_label.setText(
                    f"Gaia XP: cache miss for {len(missing_integrals)} sources — preparing spectra…"
                )


        # ---- ensure spectra exist for those missing integrals ----
        missing_spectra: list[int] = []
        for sid in missing_integrals:
            try:
                if dl.db.get_spectrum(int(sid)) is None:
                    missing_spectra.append(int(sid))
            except Exception:
                missing_spectra.append(int(sid))

        # Download missing spectra in batches (GaiaXPy calibrate)
        if missing_spectra:
            try:
                total = len(missing_spectra)
                for i in range(0, total, int(batch_size)):
                    batch = missing_spectra[i:i + int(batch_size)]

                    # progress to label
                    if getattr(self, "count_label", None) is not None:
                        self.count_label.setText(
                            f"Gaia XP: downloading/calibrating spectra {i+1}-{min(i+len(batch), total)} of {total}…"
                        )
                    try:
                        from PyQt6.QtWidgets import QApplication
                        QApplication.processEvents()
                    except Exception:
                        pass

                    spectra = dl.download_xp_spectra(batch, batch_size=len(batch))

                    # Store into sqlite cache
                    if hasattr(dl.db, "add_spectra"):
                        dl.db.add_spectra(spectra)
                    elif hasattr(dl.db, "upsert_spectra"):
                        dl.db.upsert_spectra(spectra)
                    elif hasattr(dl.db, "insert_spectra"):
                        dl.db.insert_spectra(spectra)
                    elif hasattr(dl.db, "save_spectra"):
                        dl.db.save_spectra(spectra)
                    else:
                        raise AttributeError(
                            "GaiaSpectraDB has no spectra insert method. "
                            "Expected one of add_spectra/upsert_spectra/insert_spectra/save_spectra. "
                            f"Available: {', '.join([m for m in dir(dl.db) if 'spect' in m.lower()])}"
                        )

                    # optional sanity print if calibrate returned nothing useful
                    if not spectra:
                        print(f"[SPCC] Gaia XP: calibrate returned 0 spectra for batch size {len(batch)}")

            except Exception as e:
                print(f"[SPCC] Gaia XP download failed: {e}")

        # ---- integrate anything we can (and cache the results) ----
        eps = 1e-30
        new_rows: list[tuple[int, float, float, float]] = []

        for sid in missing_integrals:
            try:
                spec = dl.db.get_spectrum(int(sid))
                if spec is None or spec.flux is None or spec.wavelengths is None:
                    continue

                # spec.wavelengths are nm in gaia_downloader; convert to Å
                wl_spec_ang = np.asarray(spec.wavelengths, dtype=np.float64).reshape(-1) * 10.0
                fl_spec = np.asarray(spec.flux, dtype=np.float64).reshape(-1)

                if wl_spec_ang.size < 2 or fl_spec.size != wl_spec_ang.size:
                    continue

                # Ensure increasing wavelength for interp
                if wl_spec_ang[0] > wl_spec_ang[-1]:
                    wl_spec_ang = wl_spec_ang[::-1]
                    fl_spec = fl_spec[::-1]

                # Interpolate Gaia flux onto wl_grid_ang (your 3000–11000Å grid)
                fl_i = np.interp(wl_grid_ang, wl_spec_ang, fl_spec, left=0.0, right=0.0)

                Sr = float(_trapz(fl_i * T_sys_R, x=wl_grid_ang))
                Sg = float(_trapz(fl_i * T_sys_G, x=wl_grid_ang))
                Sb = float(_trapz(fl_i * T_sys_B, x=wl_grid_ang))

                if abs(Sg) > eps and np.isfinite(Sr) and np.isfinite(Sg) and np.isfinite(Sb):
                    out[int(sid)] = (Sr, Sg, Sb)
                    new_rows.append((int(sid), Sr, Sg, Sb))

            except Exception:
                continue

        # ---- write synthetic photometry cache for next time ----
        if new_rows:
            try:
                dl.db.upsert_synth_phot(new_rows, system_key)
            except Exception as e:
                print(f"[SPCC] Gaia synth cache write failed (continuing): {e}")

        return out



    def fetch_stars(self):
        import time
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.wcs.wcs import NoConvergence
        from astroquery.simbad import Simbad
        from astropy.io import fits
        from PyQt6.QtWidgets import QMessageBox, QApplication

        # 0) Grab current image + header from the active document
        img, hdr, _meta = self._get_active_image_and_header()
        self.current_image = img
        self.current_header = hdr

        if self.current_header is None or self.current_image is None:
            QMessageBox.warning(self, "No Plate Solution",
                                "Please plate-solve the active document first.")
            return

        # Pickles templates list (once)
        if not hasattr(self, "pickles_templates"):
            self.pickles_templates = []
            for p in (self.user_custom_path, self.sasp_data_path):
                try:
                    with fits.open(p, memmap=False) as hd:
                        for hdu in hd:
                            if (isinstance(hdu, fits.BinTableHDU)
                                    and hdu.header.get("CTYPE", "").upper() == "SED"):
                                extname = hdu.header.get("EXTNAME", None)
                                if extname and extname not in self.pickles_templates:
                                    self.pickles_templates.append(extname)
                except Exception as e:
                    print(f"[SPCC] [fetch_stars] Could not load Pickles templates from {p}: {e}")
            self.pickles_templates.sort()

        # Build WCS
        try:
            self.initialize_wcs_from_header(self.current_header)
        except Exception:
            QMessageBox.critical(self, "WCS Error", "Could not build a 2D WCS from header.")
            return

        if not getattr(self, "wcs", None):
            QMessageBox.critical(self, "WCS Error", "Could not build a 2D WCS from header.")
            return

        # Use celestial WCS if possible (safe when WCS has extra axes)
        wcs2 = self.wcs.celestial if hasattr(self.wcs, "celestial") else self.wcs

        H, W = self.current_image.shape[:2]

        # --- original radius method (center + 4 corners) ---
        pix = np.array([[W / 2, H / 2], [0, 0], [W, 0], [0, H], [W, H]], dtype=float)
        try:
            sky = wcs2.all_pix2world(pix, 0)
        except Exception as e:
            QMessageBox.critical(self, "WCS Conversion Error", str(e))
            return

        center_sky = SkyCoord(ra=float(sky[0, 0]) * u.deg, dec=float(sky[0, 1]) * u.deg, frame="icrs")
        corners_sky = SkyCoord(ra=sky[1:, 0] * u.deg, dec=sky[1:, 1] * u.deg, frame="icrs")
        radius = center_sky.separation(corners_sky).max() * 1.05  # small margin

        # --- SIMBAD fields (NEW first, fallback to legacy) ---
        Simbad.reset_votable_fields()

        def _try_new_fields():
            # new names: B,V,R + ra,dec
            Simbad.add_votable_fields("sp", "B", "V", "R", "ra", "dec")

        def _try_legacy_fields():
            # legacy names
            Simbad.add_votable_fields("sp", "flux(B)", "flux(V)", "flux(R)", "ra(d)", "dec(d)")

        ok = False
        for _ in range(5):
            try:
                _try_new_fields()
                ok = True
                break
            except Exception:
                QApplication.processEvents()
                non_blocking_sleep(0.8)

        if not ok:
            for _ in range(5):
                try:
                    _try_legacy_fields()
                    ok = True
                    break
                except Exception:
                    QApplication.processEvents()
                    non_blocking_sleep(0.8)

        if not ok:
            QMessageBox.critical(self, "SIMBAD Error", "Could not configure SIMBAD votable fields.")
            return

        Simbad.ROW_LIMIT = 10000

        # --- Query SIMBAD ---
        result = None
        for attempt in range(1, 6):
            try:
                if getattr(self, "count_label", None) is not None:
                    self.count_label.setText(f"Attempt {attempt}/5 to query SIMBAD…")
                QApplication.processEvents()
                result = Simbad.query_region(center_sky, radius=radius)
                break
            except Exception:
                QApplication.processEvents()
                non_blocking_sleep(1.2)
                result = None

        if result is None or len(result) == 0:
            QMessageBox.information(self, "No Stars", "SIMBAD returned zero objects in that region.")
            self.star_list = []
            if getattr(self, "star_combo", None) is not None:
                self.star_combo.clear()
                self.star_combo.addItem("Vega (A0V)", userData="A0V")
            return

        # --- helpers ---
        def _unmask_num(x):
            try:
                if x is None:
                    return None
                if ma.isMaskedArray(x) and ma.is_masked(x):
                    return None
                return float(x)
            except Exception:
                return None

        def infer_letter(bv):
            if bv is None or (isinstance(bv, float) and np.isnan(bv)):
                return None
            if bv < 0.00:
                return "B"
            elif bv < 0.30:
                return "A"
            elif bv < 0.58:
                return "F"
            elif bv < 0.81:
                return "G"
            elif bv < 1.40:
                return "K"
            elif bv > 1.40:
                return "M"
            return None

        def safe_world2pix(ra_deg, dec_deg):
            try:
                xpix, ypix = wcs2.all_world2pix(ra_deg, dec_deg, 0)
                xpix, ypix = float(xpix), float(ypix)
                if np.isfinite(xpix) and np.isfinite(ypix):
                    return xpix, ypix
                return None
            except NoConvergence as e:
                try:
                    xpix, ypix = e.best_solution
                    xpix, ypix = float(xpix), float(ypix)
                    if np.isfinite(xpix) and np.isfinite(ypix):
                        return xpix, ypix
                except Exception:
                    pass
                return None
            except Exception:
                return None

        # Column names (astroquery changed these)
        cols_lower = {c.lower(): c for c in result.colnames}

        # RA/Dec in degrees:
        ra_col = cols_lower.get("ra", None) or cols_lower.get("ra(d)", None) or cols_lower.get("ra_d", None)
        dec_col = cols_lower.get("dec", None) or cols_lower.get("dec(d)", None) or cols_lower.get("dec_d", None)

        # Mag columns:
        b_col = cols_lower.get("b", None) or cols_lower.get("flux_b", None)
        v_col = cols_lower.get("v", None) or cols_lower.get("flux_v", None)
        r_col = cols_lower.get("r", None) or cols_lower.get("flux_r", None)

        if ra_col is None or dec_col is None:
            QMessageBox.critical(
                self,
                "SIMBAD Columns",
                "SIMBAD result did not include degree RA/Dec columns (ra/dec).\n"
                "Print result.colnames to see what's returned."
            )
            return

        # --- main loop ---
        self.star_list = []
        templates_for_hist = []

        for row in result:
            # spectral type column name in table
            raw_sp = None
            if "SP_TYPE" in result.colnames:
                raw_sp = row["SP_TYPE"]
            elif "sp_type" in result.colnames:
                raw_sp = row["sp_type"]

            bmag = _unmask_num(row[b_col]) if b_col is not None else None
            vmag = _unmask_num(row[v_col]) if v_col is not None else None
            rmag = _unmask_num(row[r_col]) if r_col is not None else None

            # ra/dec degrees
            ra_deg = _unmask_num(row[ra_col])
            dec_deg = _unmask_num(row[dec_col])
            if ra_deg is None or dec_deg is None:
                continue

            try:
                sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            except Exception:
                continue

            # Clean / infer spectral class
            sp_clean = None
            if raw_sp and str(raw_sp).strip():
                sp = str(raw_sp).strip().upper()
                if not (sp.startswith("SN") or sp.startswith("KA")):
                    sp_clean = sp
            elif (bmag is not None) and (vmag is not None):
                sp_clean = infer_letter(bmag - vmag)

            match_list = pickles_match_for_simbad(sp_clean, self.pickles_templates) if sp_clean else []
            best_template = match_list[0] if match_list else None

            # Convert to pixel coords (must happen before using xpix/ypix)
            xy = safe_world2pix(sc.ra.deg, sc.dec.deg)
            if xy is None:
                continue
            xpix, ypix = xy

            # Keep only in-frame stars
            if not (0 <= xpix < W and 0 <= ypix < H):
                continue

            self.star_list.append({
                "ra": float(sc.ra.deg),
                "dec": float(sc.dec.deg),
                "sp_clean": sp_clean,            # may be None
                "pickles_match": best_template,  # may be None
                "x": float(xpix),
                "y": float(ypix),
                "Bmag": float(bmag) if bmag is not None else None,
                "Vmag": float(vmag) if vmag is not None else None,
                "Rmag": float(rmag) if rmag is not None else None,
            })

            if best_template is not None:
                templates_for_hist.append(best_template)

        # --- Gaia XP fallback enrichment (only for missing/failed Pickles cases) ---
        # --- Gaia XP fallback enrichment ---
        try:
            radius_deg = float(radius.to(u.deg).value)
        except Exception:
            radius_deg = None

        if radius_deg is not None and self._use_gaia_fallback():
            try:
                _sfcc_busy(self, True, "[SPCC] Matching SIMBAD stars to Gaia source_ids…")
                self._attach_gaia_ids_to_star_list(
                    radius_deg=radius_deg,
                    mag_limit=15.5,
                    max_match_arcsec=1.0
                )
                n_gaia = sum(1 for s in self.star_list if s.get("gaia_source_id") is not None)
                _sfcc_status(self, f"[SPCC] Gaia ID match: {n_gaia} of {len(self.star_list)} stars")
            except Exception as e:
                _sfcc_status(self, f"[SPCC] Gaia ID match failed: {e}")
            finally:
                _sfcc_busy(self, False)

            # Now actually compute Gaia XP synthetic B/V/R for matched stars and use it
            gaia_ids = [s.get("gaia_source_id") for s in self.star_list if s.get("gaia_source_id") is not None]
            if gaia_ids:
                try:
                    _sfcc_busy(self, True, "[SPCC] Gaia XP: downloading/deriving synthetic photometry (this may take a minute)…")

                    # If you have COSMOS creds stored somewhere, pass them; otherwise None.
                    user = getattr(self, "cosmos_username", None)
                    pw   = getattr(self, "cosmos_password", None)

                    cache_dir = os.path.join(
                        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation),
                        "gaiaxpy_cache"
                    )
                    os.makedirs(cache_dir, exist_ok=True)
                    gaia_ids = [s.get("gaia_source_id") for s in self.star_list if s.get("gaia_source_id") is not None]

                    # only compute for those missing cached gaia mags in star_list
                    needs_bvr = []
                    for s in self.star_list:
                        sid = s.get("gaia_source_id")
                        if sid is None:
                            continue
                        if (s.get("gaia_B") is None) or (s.get("gaia_V") is None) or (s.get("gaia_R") is None):
                            needs_bvr.append(int(sid))

                    needs_bvr = sorted(set(needs_bvr))
                    if needs_bvr:
                        bvr_map = _gaiaxp_synth_bvr(
                            self,
                            gaia_ids,
                            status_cb=lambda m: _sfcc_status(self, m),
                            username=user,
                            password=pw,
                            cache_dir=cache_dir,
                        )

                        # Attach Gaia B/V/R to star_list
                        got = 0
                        for s in self.star_list:
                            sid = s.get("gaia_source_id")
                            if sid in bvr_map:
                                s["gaia_B"] = bvr_map[sid]["B"]
                                s["gaia_V"] = bvr_map[sid]["V"]
                                s["gaia_R"] = bvr_map[sid]["R"]
                                got += 1

                                # If SIMBAD mags are missing, fill them from Gaia synthetic
                                if s.get("Bmag") is None:
                                    s["Bmag"] = s["gaia_B"]
                                if s.get("Vmag") is None:
                                    s["Vmag"] = s["gaia_V"]
                                if s.get("Rmag") is None:
                                    s["Rmag"] = s["gaia_R"]

                                # If SIMBAD sp_clean is missing/garbage, infer from Gaia B-V
                                if not s.get("sp_clean"):
                                    bv = (s["gaia_B"] - s["gaia_V"]) if (s.get("gaia_B") is not None and s.get("gaia_V") is not None) else None
                                    letter = _infer_letter_from_bv(bv)
                                    if letter:
                                        s["sp_clean"] = letter

                        _sfcc_status(self, f"[SPCC] Gaia XP: attached synthetic B/V/R to {got} stars")

                        # Recompute Pickles match if we upgraded sp_clean
                        for s in self.star_list:
                            if s.get("pickles_match") is None and s.get("sp_clean"):
                                match_list = pickles_match_for_simbad(s["sp_clean"], self.pickles_templates)
                                s["pickles_match"] = match_list[0] if match_list else None
                    else:
                        _sfcc_status(self, "[SPCC] Gaia XP: B/V/R already cached in star_list — skipping.")
                except Exception as e:
                    _sfcc_status(self, f"[SPCC] Gaia XP fallback failed: {e}")
                finally:
                    _sfcc_busy(self, False)



        # --- plot / UI feedback (unchanged) ---
        if getattr(self, "figure", None) is not None:
            self.figure.clf()
        doc = self.doc_manager.get_active_document()
        if doc is not None:
            meta = dict(doc.metadata or {})
            meta["SFCC_star_list"] = list(self.star_list)  # keep it JSON-ish
            self.doc_manager.update_active_document(doc.image, metadata=meta, step_name="SFCC Stars Cached", doc=doc)
        if templates_for_hist:
            uniq, cnt = np.unique(templates_for_hist, return_counts=True)
            types_str = ", ".join([str(u) for u in uniq])
            if getattr(self, "count_label", None) is not None:
                self.count_label.setText(f"Found {len(self.star_list)} stars; templates: {types_str}")

            if getattr(self, "figure", None) is not None and getattr(self, "canvas", None) is not None:
                ax = self.figure.add_subplot(111)
                ax.bar(uniq, cnt, edgecolor="black")
                ax.set_xlabel("Spectral Type")
                ax.set_ylabel("Count")
                ax.set_title("Spectral Distribution")
                ax.tick_params(axis='x', rotation=90)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
                self.canvas.setVisible(True)
                self.canvas.draw()
        else:
            if getattr(self, "count_label", None) is not None:
                self.count_label.setText(f"Found {len(self.star_list)} in-frame SIMBAD stars (0 with Pickles matches).")
            if getattr(self, "canvas", None) is not None:
                self.canvas.setVisible(False)
                self.canvas.draw()

    # ── Core SFCC ───────────────────────────────────────────────────────
    def run_spcc(self):
        ref_sed_name = self.star_combo.currentData()
        r_filt = self.r_filter_combo.currentText()
        g_filt = self.g_filter_combo.currentText()
        b_filt = self.b_filter_combo.currentText()
        sens_name = self.sens_combo.currentText()
        lp_filt  = self.lp_filter_combo.currentText()
        lp_filt2 = self.lp_filter_combo2.currentText()

        if not ref_sed_name:
            QMessageBox.warning(self, "Error", "Select a reference spectral type (e.g. A0V).")
            return
        if r_filt == "(None)" and g_filt == "(None)" and b_filt == "(None)":
            QMessageBox.warning(self, "Error", "Pick at least one of R, G or B filters.")
            return
        if sens_name == "(None)":
            QMessageBox.warning(self, "Error", "Select a sensor QE curve.")
            return

        doc = self.doc_manager.get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.critical(self, "Error", "No active document.")
            return

        img = doc.image
        H, W = img.shape[:2]
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.critical(self, "Error", "Active document must be RGB (3 channels).")
            return

        # ---- Convert to float working space ----
        if img.dtype == np.uint8:
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = img.astype(np.float32, copy=False)

        # ---- Build SEP working copy (ONE pedestal handling only) ----
        base = self._make_working_base_for_sep(img_float)

        # Optional BN after calibration:
        # IMPORTANT: do NOT remove pedestal here either (avoid double pedestal removal).
        if self.neutralize_chk.isChecked():
            base = self._neutralize_background(base, remove_pedestal=False)

        # SEP on grayscale
        gray = np.mean(base, axis=2).astype(np.float32)

        bkg = sep.Background(gray)
        data_sub = gray - bkg.back()
        err = float(bkg.globalrms)

        # User threshold
        sep_sigma = float(self.sep_thr_spin.value()) if hasattr(self, "sep_thr_spin") else 5.0
        self.count_label.setText(f"Detecting stars (SEP σ={sep_sigma:.1f})…")
        QApplication.processEvents()

        sources = sep.extract(data_sub, sep_sigma, err=err)

        MAX_SOURCES = 300_000
        if sources.size > MAX_SOURCES:
            QMessageBox.warning(
                self,
                "Too many detections",
                f"SEP found {sources.size:,} sources with σ={sep_sigma:.1f}.\n"
                f"Increase the threshold and rerun SFCC."
            )
            return

        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error", "SEP found no sources.")
            return

        # Radius filtering (unchanged)
        r_fluxrad, _ = sep.flux_radius(
            gray, sources["x"], sources["y"],
            2.0 * sources["a"], 0.5,
            normflux=sources["flux"], subpix=5
        )
        mask = (r_fluxrad > 0.2) & (r_fluxrad <= 10)
        sources = sources[mask]
        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error", "All SEP detections rejected by radius filter.")
            return

        if not getattr(self, "star_list", None):
            QMessageBox.warning(self, "Error", "Fetch Stars (with WCS) before running SFCC.")
            return

        # ---- Match SIMBAD stars to SEP detections ----
        raw_matches = []
        for i, star in enumerate(self.star_list):
            dx = sources["x"] - star["x"]
            dy = sources["y"] - star["y"]
            j = int(np.argmin(dx * dx + dy * dy))
            if (dx[j] * dx[j] + dy[j] * dy[j]) < (3.0 ** 2):
                x_c = float(sources["x"][j])
                y_c = float(sources["y"][j])
                tmpl = star.get("pickles_match")
                spc  = star.get("sp_clean")
                raw_matches.append({
                    "sim_index": i,
                    "template": tmpl or spc or "",   # never None
                    "src_index": j,

                    # New canonical centroid keys
                    "x": x_c,
                    "y": y_c,

                    # Back-compat keys used elsewhere (gradient step, older code paths)
                    "x_pix": x_c,
                    "y_pix": y_c,

                    "a": float(sources["a"][j]),
                })

        if not raw_matches:
            QMessageBox.warning(self, "No Matches", "No SIMBAD star matched to SEP detections.")
            return

        wl_min, wl_max = 3000, 11000
        wl_grid = np.arange(wl_min, wl_max + 1)

        def load_curve(ext):
            for p in (self.user_custom_path, self.sasp_data_path):
                with fits.open(p, memmap=False) as hd:
                    if ext in hd:
                        d = hd[ext].data
                        wl = _ensure_angstrom(d["WAVELENGTH"].astype(float))
                        tp = d["THROUGHPUT"].astype(float)
                        return wl, tp
            raise KeyError(f"Curve '{ext}' not found")

        def load_sed(ext):
            for p in (self.user_custom_path, self.sasp_data_path):
                with fits.open(p, memmap=False) as hd:
                    if ext in hd:
                        d = hd[ext].data
                        wl = _ensure_angstrom(d["WAVELENGTH"].astype(float))
                        fl = d["FLUX"].astype(float)
                        return wl, fl
            raise KeyError(f"SED '{ext}' not found")

        interp = lambda wl_o, tp_o: np.interp(wl_grid, wl_o, tp_o, left=0.0, right=0.0)

        T_R = interp(*load_curve(r_filt)) if r_filt != "(None)" else np.ones_like(wl_grid)
        T_G = interp(*load_curve(g_filt)) if g_filt != "(None)" else np.ones_like(wl_grid)
        T_B = interp(*load_curve(b_filt)) if b_filt != "(None)" else np.ones_like(wl_grid)
        QE  = interp(*load_curve(sens_name)) if sens_name != "(None)" else np.ones_like(wl_grid)
        LP1 = interp(*load_curve(lp_filt))   if lp_filt  != "(None)" else np.ones_like(wl_grid)
        LP2 = interp(*load_curve(lp_filt2))  if lp_filt2 != "(None)" else np.ones_like(wl_grid)
        LP  = LP1 * LP2

        T_sys_R, T_sys_G, T_sys_B = T_R * QE * LP, T_G * QE * LP, T_B * QE * LP

        wl_ref, fl_ref = load_sed(ref_sed_name)
        fr_i = np.interp(wl_grid, wl_ref, fl_ref, left=0.0, right=0.0)

        S_ref_R = _trapz(fr_i * T_sys_R, x=wl_grid)
        S_ref_G = _trapz(fr_i * T_sys_G, x=wl_grid)
        S_ref_B = _trapz(fr_i * T_sys_B, x=wl_grid)

        diag_meas_RG, diag_exp_RG = [], []
        diag_meas_BG, diag_exp_BG = [], []
        enriched = []

        # ---- Pre-calc integrals for unique templates ----
        unique_simbad_types = set(m["template"] for m in raw_matches)

        simbad_to_pickles = {}
        pickles_templates_needed = set()
        for sp in unique_simbad_types:
            cands = pickles_match_for_simbad(sp, getattr(self, "pickles_templates", []))
            if cands:
                pname = cands[0]
                simbad_to_pickles[sp] = pname
                pickles_templates_needed.add(pname)

        template_integrals = {}
        for pname in pickles_templates_needed:
            try:
                wl_s, fl_s = load_sed(pname)
                fs_i = np.interp(wl_grid, wl_s, fl_s, left=0.0, right=0.0)
                S_sr = _trapz(fs_i * T_sys_R, x=wl_grid)
                S_sg = _trapz(fs_i * T_sys_G, x=wl_grid)
                S_sb = _trapz(fs_i * T_sys_B, x=wl_grid)
                template_integrals[pname] = (S_sr, S_sg, S_sb)
            except Exception as e:
                print(f"[SPCC] Warning: failed to load/integrate template {pname}: {e}")

        # ---- Gaia XP fallback: precompute integrals for stars lacking Pickles ----
        gaia_integrals = {}
        if self._use_gaia_fallback():
            try:
                # Collect Gaia IDs only for stars that will need fallback
                gaia_ids_needed = []
                for m in raw_matches:
                    sp = m["template"]
                    pname = simbad_to_pickles.get(sp)
                    if pname:
                        continue
                    # find the SIMBAD star record so we can read gaia_source_id
                    si = int(m["sim_index"])
                    if 0 <= si < len(self.star_list):
                        sid = self.star_list[si].get("gaia_source_id")
                        if sid is not None:
                            gaia_ids_needed.append(int(sid))

                gaia_ids_needed = sorted(set(gaia_ids_needed))
                if gaia_ids_needed:
                    gaia_integrals = self._gaia_integrals_for_source_ids(
                        gaia_ids_needed,
                        wl_grid_ang=wl_grid.astype(np.float64),
                        T_sys_R=T_sys_R.astype(np.float64),
                        T_sys_G=T_sys_G.astype(np.float64),
                        T_sys_B=T_sys_B.astype(np.float64),
                        batch_size=128
                    )

                    print(f"[SPCC] Gaia integrals available for {len(gaia_integrals)} source_ids")
            except Exception as e:
                print(f"[SPCC] Gaia XP fallback failed: {e}")
                gaia_integrals = {}
        else:
            print("[SPCC] Gaia XP fallback skipped (disabled by user or unavailable).")

        def measure_star_rgb_aperture(img_rgb_f32: np.ndarray, x: float, y: float, r: float,
                                    rin: float, rout: float) -> tuple[float, float, float]:
            # SEP expects float32, C-contiguous, and (x,y) in pixel coords
            R = np.ascontiguousarray(img_rgb_f32[..., 0], dtype=np.float32)
            G = np.ascontiguousarray(img_rgb_f32[..., 1], dtype=np.float32)
            B = np.ascontiguousarray(img_rgb_f32[..., 2], dtype=np.float32)

            # sum_circle returns (flux, fluxerr, flag) when err not provided; handle either form
            def _sum(ch):
                out = sep.sum_circle(ch, np.array([x]), np.array([y]), r,
                                    subpix=5, bkgann=(rin, rout))
                # Depending on sep version, out can be (flux, fluxerr, flag) or (flux, flag)
                if len(out) == 3:
                    flux, _fluxerr, flag = out
                else:
                    flux, flag = out
                return float(flux[0]), int(flag[0])

            fR, flR = _sum(R)
            fG, flG = _sum(G)
            fB, flB = _sum(B)

            # If any flags set, you can reject (edge, etc.)
            if (flR | flG | flB) != 0:
                return None, None, None

            return fR, fG, fB


        # ---- Main match loop (measure from 'base' only) ----
        for m in raw_matches:
            xi = float(m.get("x_pix", m["x"]))
            yi = float(m.get("y_pix", m["y"]))
            sp = m["template"]

            # measure on the SEP working copy (already BN’d, only one pedestal handling)
            x = float(m["x"])
            y = float(m["y"])

            # Aperture radius choice (simple + robust)
            # sources["a"] is roughly semi-major sigma-ish from SEP; a common quick rule:
            # r ~ 2.5 * a, with sane clamps.
            a = float(m.get("a", 1.5))
            r = float(np.clip(2.5 * a, 2.0, 12.0))

            # Annulus (your “kicker”): inner/outer in pixels
            rin  = float(np.clip(3.0 * r, 6.0, 40.0))
            rout = float(np.clip(5.0 * r, rin + 2.0, 60.0))

            meas = measure_star_rgb_aperture(base, x, y, r, rin, rout)
            if meas[0] is None:
                continue
            Rm, Gm, Bm = meas

            if Gm <= 0:
                continue
            meas_RG = Rm / Gm
            meas_BG = Bm / Gm

            if Gm <= 0:
                continue

            pname = simbad_to_pickles.get(sp)

            # --- primary: Pickles template integrals ---
            integrals = template_integrals.get(pname) if pname else None

            # --- fallback: Gaia XP spectrum integrals (if available) ---
            if integrals is None:
                si = int(m["sim_index"])
                sid = None
                if 0 <= si < len(self.star_list):
                    sid = self.star_list[si].get("gaia_source_id")

                if sid is not None:
                    integrals = gaia_integrals.get(int(sid))

            if integrals is None:
                continue

            S_sr, S_sg, S_sb = integrals

            if S_sg <= 0:
                continue

            exp_RG = S_sr / S_sg
            exp_BG = S_sb / S_sg
            meas_RG = Rm / Gm
            meas_BG = Bm / Gm

            diag_meas_RG.append(meas_RG); diag_exp_RG.append(exp_RG)
            diag_meas_BG.append(meas_BG); diag_exp_BG.append(exp_BG)

            enriched.append({
                **m,
                "R_meas": Rm, "G_meas": Gm, "B_meas": Bm,
                "S_star_R": S_sr, "S_star_G": S_sg, "S_star_B": S_sb,
                "exp_RG": exp_RG, "exp_BG": exp_BG
            })

        self._last_matched = enriched
        diag_meas_RG = np.asarray(diag_meas_RG, dtype=np.float64)
        diag_exp_RG  = np.asarray(diag_exp_RG,  dtype=np.float64)
        diag_meas_BG = np.asarray(diag_meas_BG, dtype=np.float64)
        diag_exp_BG  = np.asarray(diag_exp_BG,  dtype=np.float64)

        if diag_meas_RG.size == 0 or diag_meas_BG.size == 0:
            QMessageBox.information(self, "No Valid Stars", "No stars with valid measured vs expected ratios.")
            return

        n_stars = int(diag_meas_RG.size)

        def rms_frac(pred, exp):
            return float(np.sqrt(np.mean(((pred / exp) - 1.0) ** 2)))

        slope_only = lambda x, m: m * x
        affine     = lambda x, m, b: m * x + b
        quad       = lambda x, a, b, c: a * x**2 + b * x + c

        denR = float(np.sum(diag_meas_RG**2))
        denB = float(np.sum(diag_meas_BG**2))
        mR_s = (float(np.sum(diag_meas_RG * diag_exp_RG)) / denR) if denR > 0 else 1.0
        mB_s = (float(np.sum(diag_meas_BG * diag_exp_BG)) / denB) if denB > 0 else 1.0
        rms_s = rms_frac(slope_only(diag_meas_RG, mR_s), diag_exp_RG) + rms_frac(slope_only(diag_meas_BG, mB_s), diag_exp_BG)

        mR_a, bR_a = np.linalg.lstsq(
            np.vstack([diag_meas_RG, np.ones_like(diag_meas_RG)]).T, diag_exp_RG, rcond=None
        )[0]
        mB_a, bB_a = np.linalg.lstsq(
            np.vstack([diag_meas_BG, np.ones_like(diag_meas_BG)]).T, diag_exp_BG, rcond=None
        )[0]
        rms_a = rms_frac(affine(diag_meas_RG, mR_a, bR_a), diag_exp_RG) + rms_frac(affine(diag_meas_BG, mB_a, bB_a), diag_exp_BG)

        aR_q, bR_q, cR_q = np.polyfit(diag_meas_RG, diag_exp_RG, 2)
        aB_q, bB_q, cB_q = np.polyfit(diag_meas_BG, diag_exp_BG, 2)
        rms_q = rms_frac(quad(diag_meas_RG, aR_q, bR_q, cR_q), diag_exp_RG) + rms_frac(quad(diag_meas_BG, aB_q, bB_q, cB_q), diag_exp_BG)

        idx = int(np.argmin([rms_s, rms_a, rms_q]))
        if idx == 0:
            coeff_R, coeff_B, model_choice = (0.0, float(mR_s), 0.0), (0.0, float(mB_s), 0.0), "slope-only"
        elif idx == 1:
            coeff_R, coeff_B, model_choice = (0.0, float(mR_a), float(bR_a)), (0.0, float(mB_a), float(bB_a)), "affine"
        else:
            coeff_R, coeff_B, model_choice = (float(aR_q), float(bR_q), float(cR_q)), (float(aB_q), float(bB_q), float(cB_q)), "quadratic"

        poly = lambda c, x: c[0] * x**2 + c[1] * x + c[2]

        # ---- Diagnostics plot (unchanged) ----
        self.figure.clf()
        res0_RG = (diag_meas_RG / diag_exp_RG) - 1.0
        res0_BG = (diag_meas_BG / diag_exp_BG) - 1.0
        res1_RG = (poly(coeff_R, diag_meas_RG) / diag_exp_RG) - 1.0
        res1_BG = (poly(coeff_B, diag_meas_BG) / diag_exp_BG) - 1.0

        ymin = float(np.min(np.concatenate([res0_RG, res0_BG])))
        ymax = float(np.max(np.concatenate([res0_RG, res0_BG])))
        pad  = 0.05 * (ymax - ymin) if ymax > ymin else 0.02
        y_lim = (ymin - pad, ymax + pad)

        def shade(ax, yvals, color):
            q1, q3 = np.percentile(yvals, [25, 75])
            ax.axhspan(q1, q3, color=color, alpha=0.10, zorder=0)

        ax2 = self.figure.add_subplot(1, 2, 1)
        ax2.axhline(0, color="0.65", ls="--", lw=1)
        shade(ax2, res0_RG, "firebrick"); shade(ax2, res0_BG, "royalblue")
        ax2.scatter(diag_exp_RG, res0_RG, c="firebrick", marker="o", alpha=0.7, label="R/G residual")
        ax2.scatter(diag_exp_BG, res0_BG, c="royalblue", marker="s", alpha=0.7, label="B/G residual")
        ax2.set_ylim(*y_lim)
        ax2.set_xlabel("Expected (band/G)")
        ax2.set_ylabel("Frac residual (meas/exp − 1)")
        ax2.set_title("Residuals • BEFORE")
        ax2.legend(frameon=False, fontsize=7, loc="lower right")

        ax3 = self.figure.add_subplot(1, 2, 2)
        ax3.axhline(0, color="0.65", ls="--", lw=1)
        shade(ax3, res1_RG, "firebrick"); shade(ax3, res1_BG, "royalblue")
        ax3.scatter(diag_exp_RG, res1_RG, c="firebrick", marker="o", alpha=0.7)
        ax3.scatter(diag_exp_BG, res1_BG, c="royalblue", marker="s", alpha=0.7)
        ax3.set_ylim(*y_lim)
        ax3.set_xlabel("Expected (band/G)")
        ax3.set_ylabel("Frac residual (corrected/exp − 1)")
        ax3.set_title("Residuals • AFTER")

        self.canvas.setVisible(True)
        self.figure.tight_layout(w_pad=2.0)
        self.canvas.draw()

        # ---- Apply SFCC correction to ORIGINAL floats (not the SEP base) ----
        self.count_label.setText("Applying SFCC color scales to image…")
        QApplication.processEvents()

        eps = 1e-8
        #calibrated = base.copy()
        calibrated = img_float.copy() 

        R = calibrated[..., 0]
        G = calibrated[..., 1]
        B = calibrated[..., 2]

        RG = R / np.maximum(G, eps)
        BG = B / np.maximum(G, eps)

        aR, bR, cR = coeff_R
        aB, bB, cB = coeff_B

        mR = aR * RG**2 + bR * RG + cR
        mB = aB * BG**2 + bB * BG + cB

        mR = np.clip(mR, 0.25, 4.0)
        mB = np.clip(mB, 0.25, 4.0)

        pR = float(np.median(R))
        pB = float(np.median(B))

        calibrated[..., 0] = _pivot_scale_channel(R, mR, pR)
        calibrated[..., 2] = _pivot_scale_channel(B, mB, pB)

        calibrated = np.clip(calibrated, 0.0, 1.0)

        # --- OPTIONAL: apply BN/pedestal to the FINAL calibrated image, not just SEP base ---
        if self.neutralize_chk.isChecked():
            try:
                print("[SPCC] Applying background neutralization to final calibrated image...")
                _debug_probe_channels(calibrated, "final_before_BN")

                # If you want pedestal removal as part of BN, set remove_pedestal=True here
                # (and/or make this a checkbox)
                calibrated = self._neutralize_background(calibrated, remove_pedestal=True)

                _debug_probe_channels(calibrated, "final_after_BN")
            except Exception as e:
                print(f"[SPCC] Final BN failed: {e}")


        # Convert back to original dtype
        if img.dtype == np.uint8:
            out_img = (np.clip(calibrated, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            out_img = np.clip(calibrated, 0.0, 1.0).astype(np.float32)

        new_meta = dict(doc.metadata or {})
        new_meta.update({
            "SFCC_applied": True,
            "SFCC_timestamp": datetime.now().isoformat(),
            "SFCC_model": model_choice,
            "SFCC_coeff_R": [float(v) for v in coeff_R],
            "SFCC_coeff_B": [float(v) for v in coeff_B],
        })

        self.doc_manager.update_active_document(
            out_img,
            metadata=new_meta,
            step_name="SFCC Calibrated",
            doc=doc,
        )

        self.count_label.setText(f"Applied SFCC color calibration using {n_stars} stars")
        QApplication.processEvents()

        def pretty(coeff):
            # coefficient sum gives you f(1) for quadratic form a*x^2+b*x+c at x=1
            return float(coeff[0] + coeff[1] + coeff[2])

        QMessageBox.information(
            self,
            "SFCC Complete",
            f"Applied SFCC using {n_stars} stars\n"
            f"Model: {model_choice}\n"
            f"R ratio @ x=1: {pretty(coeff_R):.4f}\n"
            f"B ratio @ x=1: {pretty(coeff_B):.4f}\n"
            f"Background neutralisation: {'ON' if self.neutralize_chk.isChecked() else 'OFF'}"
        )

        self.current_image = out_img  # keep for gradient step


    # ── Chromatic gradient (optional) ──────────────────────────────────

    def run_gradient_extraction(self):
        if not getattr(self, "_last_matched", None):
            QMessageBox.warning(self, "No Star Matches", "Run colour calibration first.")
            return

        doc = self.doc_manager.get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.critical(self, "Error", "No active document.")
            return

        img = doc.image
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.critical(self, "Error", "Active document must be RGB.")
            return

        is_u8 = (img.dtype == np.uint8)
        img_f = img.astype(np.float32) / (255.0 if is_u8 else 1.0)
        H, W = img_f.shape[:2]

        # Need star diagnostics from SPCC
        if not hasattr(self, "_last_matched") or not self._last_matched:
            QMessageBox.warning(self, "No Star Matches", "Run color calibration first."); return

        down_fact = 4
        Hs, Ws = H // down_fact, W // down_fact
        small  = cv2.resize(img_f, (Ws, Hs), interpolation=cv2.INTER_AREA)

        pts, dRG, dBG = [], [], []
        eps, box = 1e-8, 3
        for st in self._last_matched:
            xs_full, ys_full = st["x_pix"], st["y_pix"]
            xs, ys = xs_full / down_fact, ys_full / down_fact
            xs_c, ys_c = int(round(xs)), int(round(ys))
            if not (0 <= xs_c < Ws and 0 <= ys_c < Hs): continue
            xsl = slice(max(0, xs_c-box), min(Ws, xs_c+box+1))
            ysl = slice(max(0, ys_c-box), min(Hs, ys_c+box+1))
            Rm = np.median(small[ysl, xsl, 0]); Gm = np.median(small[ysl, xsl, 1]); Bm = np.median(small[ysl, xsl, 2])
            if Gm <= 0: continue
            meas_RG = Rm / Gm; meas_BG = Bm / Gm
            exp_RG, exp_BG = st["exp_RG"], st["exp_BG"]
            if exp_RG is None or exp_BG is None: continue
            dm_RG = -2.5 * np.log10((meas_RG+eps)/(exp_RG+eps))
            dm_BG = -2.5 * np.log10((meas_BG+eps)/(exp_BG+eps))
            pts.append([xs, ys]); dRG.append(dm_RG); dBG.append(dm_BG)

        pts  = np.asarray(pts); dRG = np.asarray(dRG); dBG = np.asarray(dBG)
        if pts.shape[0] < 5:
            QMessageBox.warning(self, "Too Few Stars", "Need ≥5 stars after clipping."); return

        def sclip(arr, p, s=2.5):
            m, sd = np.median(arr), np.std(arr); keep = np.abs(arr-m) < s*sd
            return p[keep], arr[keep]

        ptsRG, dRG = sclip(dRG, pts); ptsBG, dBG = sclip(dBG, pts)

        mode = getattr(self, "grad_method", "poly2")
        bgRG_s = compute_gradient_map(ptsRG, dRG, (Hs, Ws), method=mode)
        bgBG_s = compute_gradient_map(ptsBG, dBG, (Hs, Ws), method=mode)

        for bg in (bgRG_s, bgBG_s):
            bg -= np.median(bg)
            peak = np.max(np.abs(bg))
            if peak > 0.2: bg *= 0.2/peak

        bgRG = cv2.resize(bgRG_s, (W, H), interpolation=cv2.INTER_CUBIC)
        bgBG = cv2.resize(bgBG_s, (W, H), interpolation=cv2.INTER_CUBIC)

        scale_R = 10**(-0.4*bgRG); scale_B = 10**(-0.4*bgBG)

        self.figure.clf()
        for i,(surf,lbl) in enumerate(((bgRG,"Δm R/G"),(bgBG,"Δm B/G"))):
            ax  = self.figure.add_subplot(1,2,i+1)
            im  = ax.imshow(surf, origin="lower", cmap="RdBu")
            ax.set_title(lbl); self.figure.colorbar(im, ax=ax)
        self.canvas.setVisible(True); self.figure.tight_layout(); self.canvas.draw()

        corrected = img_f.copy()
        corrected[...,0] = np.clip(corrected[...,0] / scale_R, 0, 1.0)
        corrected[...,2] = np.clip(corrected[...,2] / scale_B, 0, 1.0)
        corrected = np.clip(corrected, 0, 1)
        if is_u8:
            corrected = (corrected * 255.0).astype(np.uint8)
        else:
            corrected = corrected.astype(np.float32)

        new_meta = dict(doc.metadata or {})
        new_meta["ColourGradRemoved"] = True

        self.doc_manager.update_active_document(
            corrected,
            metadata=new_meta,
            step_name="Colour-Gradient (star spectra, ¼-res fit)",
            doc=doc,   # 👈 same idea
        )
        self.count_label.setText("Chromatic gradient removed ✓")
        QApplication.processEvents()

    # ── Viewer, close ──────────────────────────────────────────────────

    def open_sasp_viewer(self):
        if self.sasp_viewer_window is not None:
            if self.sasp_viewer_window.isVisible():
                self.sasp_viewer_window.raise_()
            else:
                self.sasp_viewer_window.show()
            return

        self.sasp_viewer_window = SaspViewer(
            sasp_data_path=self.sasp_data_path,
            user_custom_path=self.user_custom_path
        )
        self.sasp_viewer_window.show()
        self.sasp_viewer_window.destroyed.connect(self._on_sasp_closed)

    def _cleanup(self):
        # 1) Close/cleanup child window (SaspViewer)
        try:
            if getattr(self, "sasp_viewer_window", None) is not None:
                try:
                    self.sasp_viewer_window.destroyed.disconnect(self._on_sasp_closed)
                except Exception:
                    pass
                try:
                    self.sasp_viewer_window.close()
                except Exception:
                    pass
                self.sasp_viewer_window = None
        except Exception:
            pass

        # 2) Disconnect any long-lived external signals (add these if/when used)
        # Example patterns:
        try:
            self.doc_manager.activeDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        try:
            self.main_win.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass

        # 3) Release large caches/refs (important since dialog may not be deleted)
        try:
            self.current_image = None
            self.current_header = None
            self.star_list = []
            self._last_matched = []
            if hasattr(self, "wcs"):
                self.wcs = None
            if hasattr(self, "wcs_header"):
                self.wcs_header = None
        except Exception:
            pass

        # 4) Matplotlib cleanup
        try:
            if getattr(self, "figure", None) is not None:
                self.figure.clf()
            if getattr(self, "canvas", None) is not None:
                self.canvas.setVisible(False)
                self.canvas.draw_idle()
        except Exception:
            pass


    def _on_sasp_closed(self, _=None):
        # Called when the SaspViewer window is destroyed
        self.sasp_viewer_window = None
        self._cleanup()

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Helper to open the dialog from your app
# ──────────────────────────────────────────────────────────────────────────────

def open_sfcc(doc_manager, sasp_data_path: str, parent=None) -> SFCCDialog:
    dlg = SFCCDialog(doc_manager=doc_manager, sasp_data_path=sasp_data_path, parent=parent)
    dlg.show()
    return dlg
