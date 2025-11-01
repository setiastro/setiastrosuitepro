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

import os, re, cv2, math, time
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
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

# ── SEP (Source Extractor)
import sep

# ── Matplotlib backend for Qt
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtCore import (Qt, QPoint, QRect, QMimeData, QSettings, QByteArray,
                          QDataStream, QIODevice, QEvent, QStandardPaths)
from PyQt6.QtGui import (QAction, QDrag, QIcon, QMouseEvent, QPixmap, QKeyEvent)
from PyQt6.QtWidgets import (QToolBar, QWidget, QToolButton, QMenu, QApplication, QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox,
                             QInputDialog, QMessageBox, QDialog, QFileDialog,
    QFormLayout, QDialogButtonBox, QDoubleSpinBox, QCheckBox, QLabel, QRubberBand, QRadioButton, QMainWindow, QPushButton)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

# --- Debug/guards -----------------------------------------------------
def _debug_probe_channels(img: np.ndarray, label="input"):
    assert img.ndim == 3 and img.shape[2] == 3, f"[SFCC] {label}: not RGB"
    f = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
    means = [float(f[...,i].mean()) for i in range(3)]
    stds  = [float(f[...,i].std())  for i in range(3)]
    rg = float(np.corrcoef(f[...,0].ravel(), f[...,1].ravel())[0,1])
    rb = float(np.corrcoef(f[...,0].ravel(), f[...,2].ravel())[0,1])
    gb = float(np.corrcoef(f[...,1].ravel(), f[...,2].ravel())[0,1])
    print(f"[SFCC] {label}: mean={means}, std={stds}, corr(R,G)={rg:.5f}, corr(R,B)={rb:.5f}, corr(G,B)={gb:.5f}")
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
        print("[SFCC] Heuristic suggests BGR input → converting to RGB")
        return img[..., ::-1]
    return img

def _ensure_angstrom(wl: np.ndarray) -> np.ndarray:
    """If wavelengths look like nm (≈300–1100), convert to Å."""
    med = float(np.median(wl))
    return wl * 10.0 if 250.0 <= med <= 2000.0 else wl


def pickles_match_for_simbad(simbad_sp: str, available_extnames: List[str]) -> List[str]:
    sp = simbad_sp.strip().upper()
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


# ──────────────────────────────────────────────────────────────────────────────
# Simple responses viewer (unchanged core logic; useful for diagnostics)
# ──────────────────────────────────────────────────────────────────────────────
class SaspViewer(QMainWindow):
    def __init__(self, sasp_data_path: str, user_custom_path: str):
        super().__init__()
        self.setWindowTitle("SASP Viewer (Pickles + RGB Responses)")

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
        row.addWidget(QLabel("Star Template:"))
        self.star_combo = QComboBox(); self.star_combo.addItems(self.pickles_templates); row.addWidget(self.star_combo)
        row.addWidget(QLabel("R-Filter:"))
        self.r_filter_combo = QComboBox(); self.r_filter_combo.addItems(self.rgb_filter_choices); row.addWidget(self.r_filter_combo)
        row.addWidget(QLabel("G-Filter:"))
        self.g_filter_combo = QComboBox(); self.g_filter_combo.addItems(self.rgb_filter_choices); row.addWidget(self.g_filter_combo)
        row.addWidget(QLabel("B-Filter:"))
        self.b_filter_combo = QComboBox(); self.b_filter_combo.addItems(self.rgb_filter_choices); row.addWidget(self.b_filter_combo)

        row2 = QHBoxLayout(); vbox.addLayout(row2)
        row2.addWidget(QLabel("LP/Cut Filter1:"))
        self.lp_filter_combo = QComboBox(); self.lp_filter_combo.addItems(self.rgb_filter_choices); row2.addWidget(self.lp_filter_combo)
        row2.addWidget(QLabel("LP/Cut Filter2:"))
        self.lp_filter_combo2 = QComboBox(); self.lp_filter_combo2.addItems(self.rgb_filter_choices); row2.addWidget(self.lp_filter_combo2)
        row2.addSpacing(20); row2.addWidget(QLabel("Sensor (QE):"))
        self.sens_combo = QComboBox(); self.sens_combo.addItems(self.sensor_list); row2.addWidget(self.sens_combo)

        self.plot_btn = QPushButton("Plot"); self.plot_btn.clicked.connect(self.update_plot); row.addWidget(self.plot_btn)

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
                    S_star = np.trapezoid(data["response"], x=common_wl)
                    S_veg  = np.trapezoid(fl_veg_c * data["T_sys"], x=common_wl)
                    if S_veg>0 and S_star>0:
                        mag = -2.5 * np.log10(S_star / S_veg)
                        mag_texts.append(f"{color[0].upper()}→{data['filter_name']}: {mag:.2f}")
                    else:
                        mag_texts.append(f"{color[0].upper()}→{data['filter_name']}: N/A")
        title_text = " | ".join(mag_texts) if mag_texts else "No channels selected"

        self.figure.clf()
        ax1 = self.figure.add_subplot(111)
        ax1.plot(common_wl, fl_common, color="black", linewidth=1, label=f"{star_ext} SED")
        for color, data in rgb_data.items():
            if data is not None:
                ax1.plot(common_wl, data["response"], color="gold", linewidth=1.5, label=f"{color.upper()} Response")
        ax1.set_xlim(wl_min, wl_max); ax1.set_xlabel("Wavelength (Å)")
        ax1.set_ylabel("Flux (erg s⁻¹ cm⁻² Å⁻¹)", color="black"); ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Relative Throughput", color="red"); ax2.tick_params(axis="y", labelcolor="red"); ax2.set_ylim(0.0, 1.0)
        if rgb_data["red"] is not None:   ax2.plot(common_wl, rgb_data["red"]["T_sys"],   color="red",   linestyle="--", linewidth=1, label="R filter×QE")
        if rgb_data["green"] is not None: ax2.plot(common_wl, rgb_data["green"]["T_sys"], color="green", linestyle="--", linewidth=1, label="G filter×QE")
        if rgb_data["blue"] is not None:  ax2.plot(common_wl, rgb_data["blue"]["T_sys"],  color="blue",  linestyle="--", linewidth=1, label="B filter×QE")

        ax1.grid(True, which="both", linestyle="--", alpha=0.3); self.figure.suptitle(title_text, fontsize=10)
        lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        self.canvas.draw()


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
        self.setWindowTitle("Spectral Flux Color Calibration")
        self.setMinimumSize(800, 600)

        self.doc_manager = doc_manager
        self.sasp_data_path = sasp_data_path
        self.user_custom_path = self._ensure_user_custom_fits()
        self.current_image = None
        self.current_header = None
        self.orientation_label = QLabel("Orientation: N/A")
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

        self.grad_method = "poly3"
        self.grad_method_combo.currentTextChanged.connect(lambda m: setattr(self, "grad_method", m))

    # ── View plumbing ───────────────────────────────────────────────────
    def _get_active_image_and_header(self):
        doc = self.doc_manager.get_active_document()
        if doc is None:
            return None, None, None
        img = doc.image
        meta = doc.metadata or {}
        hdr = meta.get("original_header")
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
        self.fetch_stars_btn = QPushButton("Step 1: Fetch Stars from Current View")
        f = self.fetch_stars_btn.font(); f.setBold(True); self.fetch_stars_btn.setFont(f)
        self.fetch_stars_btn.clicked.connect(self.fetch_stars)
        row1.addWidget(self.fetch_stars_btn)

        self.open_sasp_btn = QPushButton("Open SASP Viewer")
        self.open_sasp_btn.clicked.connect(self.open_sasp_viewer)
        row1.addWidget(self.open_sasp_btn)

        row1.addSpacing(20)
        row1.addWidget(QLabel("Select White Reference:"))
        self.star_combo = QComboBox()
        self.star_combo.addItem("Vega (A0V)", userData="A0V")
        for sed in getattr(self, "sed_list", []):
            if sed.upper() == "A0V": continue
            self.star_combo.addItem(sed, userData=sed)
        row1.addWidget(self.star_combo)
        idx_g2v = self.star_combo.findData("G2V")
        if idx_g2v >= 0: self.star_combo.setCurrentIndex(idx_g2v)

        row2 = QHBoxLayout(); layout.addLayout(row2)
        row2.addWidget(QLabel("R Filter:"))
        self.r_filter_combo = QComboBox(); self.r_filter_combo.addItem("(None)"); self.r_filter_combo.addItems(self.filter_list); row2.addWidget(self.r_filter_combo)
        row2.addSpacing(20); row2.addWidget(QLabel("G Filter:"))
        self.g_filter_combo = QComboBox(); self.g_filter_combo.addItem("(None)"); self.g_filter_combo.addItems(self.filter_list); row2.addWidget(self.g_filter_combo)
        row2.addSpacing(20); row2.addWidget(QLabel("B Filter:"))
        self.b_filter_combo = QComboBox(); self.b_filter_combo.addItem("(None)"); self.b_filter_combo.addItems(self.filter_list); row2.addWidget(self.b_filter_combo)

        row3 = QHBoxLayout(); layout.addLayout(row3)
        row3.addStretch()
        row3.addWidget(QLabel("Sensor (QE):"))
        self.sens_combo = QComboBox(); self.sens_combo.addItem("(None)"); self.sens_combo.addItems(self.sensor_list); row3.addWidget(self.sens_combo)
        row3.addSpacing(20); row3.addWidget(QLabel("LP/Cut Filter1:"))
        self.lp_filter_combo = QComboBox(); self.lp_filter_combo.addItem("(None)"); self.lp_filter_combo.addItems(self.filter_list); row3.addWidget(self.lp_filter_combo)
        row3.addSpacing(20); row3.addWidget(QLabel("LP/Cut Filter2:"))
        self.lp_filter_combo2 = QComboBox(); self.lp_filter_combo2.addItem("(None)"); self.lp_filter_combo2.addItems(self.filter_list); row3.addWidget(self.lp_filter_combo2)
        row3.addStretch()

        row4 = QHBoxLayout(); layout.addLayout(row4)
        self.run_spcc_btn = QPushButton("Step 2: Run Color Calibration")
        f2 = self.run_spcc_btn.font(); f2.setBold(True); self.run_spcc_btn.setFont(f2)
        self.run_spcc_btn.clicked.connect(self.run_spcc)
        row4.addWidget(self.run_spcc_btn)

        self.neutralize_chk = QCheckBox("Background Neutralization"); self.neutralize_chk.setChecked(True); row4.addWidget(self.neutralize_chk)

        self.run_grad_btn = QPushButton("Run Gradient Extraction (Beta)")
        f3 = self.run_grad_btn.font(); f3.setBold(True); self.run_grad_btn.setFont(f3)
        self.run_grad_btn.clicked.connect(self.run_gradient_extraction)
        row4.addWidget(self.run_grad_btn)

        self.grad_method_combo = QComboBox(); self.grad_method_combo.addItems(["poly2","poly3","rbf"]); self.grad_method_combo.setCurrentText("poly3")
        row4.addWidget(self.grad_method_combo)

        row4.addSpacing(15)
        row4.addWidget(QLabel("Star detect σ:"))
        self.sep_thr_spin = QSpinBox()
        self.sep_thr_spin.setRange(2, 50)        # should be enough
        self.sep_thr_spin.setValue(5)            # our current hardcoded value
        self.sep_thr_spin.valueChanged.connect(self.save_sep_threshold_setting)
        row4.addWidget(self.sep_thr_spin)

        row4.addStretch()
        self.add_curve_btn = QPushButton("Add Custom Filter/Sensor Curve…")
        self.add_curve_btn.clicked.connect(self.add_custom_curve); row4.addWidget(self.add_curve_btn)
        self.remove_curve_btn = QPushButton("Remove Filter/Sensor Curve…")
        self.remove_curve_btn.clicked.connect(self.remove_custom_curve); row4.addWidget(self.remove_curve_btn)
        row4.addStretch()
        self.close_btn = QPushButton("Close"); self.close_btn.clicked.connect(self.close); row4.addWidget(self.close_btn)

        self.count_label = QLabel(""); layout.addWidget(self.count_label)

        self.figure = Figure(figsize=(6, 4)); self.canvas = FigureCanvas(self.figure); self.canvas.setVisible(False); layout.addWidget(self.canvas, stretch=1)
        self.reset_btn = QPushButton("Reset View/Close"); self.reset_btn.clicked.connect(self.close); layout.addWidget(self.reset_btn)

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

    # ── Curve utilities ─────────────────────────────────────────────────

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
        fig, ax = plt.subplots(figsize=(8, 5)); ax.imshow(rgb_img); ax.set_title("Click 3 points, then close")
        pts = plt.ginput(3, timeout=-1); plt.close(fig)
        if len(pts) != 3: raise RuntimeError("Need exactly three clicks for calibration.")
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
        name_str, ok1 = QInputDialog.getText(self, "Curve Name", "Enter curve name (EXTNAME):")
        if not (ok1 and name_str.strip()): return False, None, None
        extname = name_str.strip().upper().replace(" ", "_")
        ch_str, ok2 = QInputDialog.getText(self, "Channel", "Enter channel (R,G,B or Q for sensor):")
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
        msg = QMessageBox(self); msg.setWindowTitle("Add Custom Curve"); msg.setText("Choose how to add the curve:")
        csv_btn = msg.addButton("Import CSV", QMessageBox.ButtonRole.AcceptRole)
        img_btn = msg.addButton("Digitize Image", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        if msg.clickedButton() == csv_btn: self._import_curve_from_csv()
        elif msg.clickedButton() == img_btn: self._digitize_curve_from_image()

    def _import_curve_from_csv(self):
        csv_path, _ = QFileDialog.getOpenFileName(self, "Select 2-column CSV (λ_nm, response)", "", "CSV Files (*.csv);;All Files (*)")
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
                QMessageBox.critical(self, "CSV Error", f"Could not read CSV:\n{e2}"); return
        except Exception as e:
            QMessageBox.critical(self, "CSV Error", f"Could not read CSV:\n{e}"); return

        ok, extname_base, channel_val = self._query_name_channel()
        if not ok: return
        wl_ang = (wl_nm * 10.0).astype(np.float32); tr_final = tp.astype(np.float32)
        self._append_curve_hdu(wl_ang, tr_final, extname_base, "SENSOR" if channel_val=="Q" else "FILTER", f"CSV:{os.path.basename(csv_path)}")
        self._reload_hdu_lists(); self.refresh_filter_sensor_lists()
        QMessageBox.information(self, "Done", f"CSV curve '{extname_base}' added.")

    def _digitize_curve_from_image(self):
        img_path_str, _ = QFileDialog.getOpenFileName(self, "Select Curve Image to Digitize", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if not img_path_str: return
        img_filename = os.path.basename(img_path_str)
        try:
            bgr = cv2.imread(img_path_str)
            if bgr is None: raise RuntimeError(f"cv2.imread returned None for '{img_path_str}'")
            rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image:\n{e}"); return

        try:
            (px_bl, py_bl), (px_br, py_br), (px_tl, py_tl) = self.get_calibration_points(rgb_img)
        except Exception as e:
            QMessageBox.critical(self, "Digitization Error", str(e)); return

        λ_min_str, ok1 = QInputDialog.getText(self, "λ_min", "Enter λ_min (in nm):")
        λ_max_str, ok2 = QInputDialog.getText(self, "λ_max", "Enter λ_max (in nm):")
        if not (ok1 and ok2 and λ_min_str.strip() and λ_max_str.strip()): return
        try:
            λ_min = float(λ_min_str); λ_max = float(λ_max_str)
        except ValueError:
            QMessageBox.critical(self, "Input Error", "λ_min and λ_max must be numbers."); return

        ok, extname_base, channel_val = self._query_name_channel()
        if not ok: return

        px_to_λ, py_to_resp = self.build_transforms(px_bl, py_bl, px_br, py_br, px_tl, py_tl, λ_min, λ_max, 0.0, 1.0)
        try:
            df_curve = self.extract_curve(gray_img, px_to_λ, py_to_resp, λ_min, λ_max, threshold=50)
        except Exception as e:
            QMessageBox.critical(self, "Extraction Error", str(e)); return

        df_curve["wl_int"] = df_curve["wavelength_nm"].round().astype(int)
        grp = (df_curve.groupby("wl_int")["response"].median().reset_index().sort_values("wl_int"))
        wl = grp["wl_int"].to_numpy(dtype=int); tr = grp["response"].to_numpy(dtype=float)

        try:
            tr_corr, _ = self.interpolate_bad_points(wl, tr)
        except Exception as e:
            QMessageBox.critical(self, "Interpolation Error", str(e)); return

        tr_smoothed = self.smooth_curve(tr_corr, window_size=5)
        wl_ang = (wl.astype(float) * 10.0).astype(np.float32); tr_final = tr_smoothed.astype(np.float32)
        self._append_curve_hdu(wl_ang, tr_final, extname_base, "SENSOR" if channel_val=="Q" else "FILTER", f"UserDefined:{img_filename}")
        self._reload_hdu_lists(); self.refresh_filter_sensor_lists()
        QMessageBox.information(self, "Done", f"Added curve '{extname_base}'.")

    def remove_custom_curve(self):
        all_curves = self.filter_list + self.sensor_list
        if not all_curves:
            QMessageBox.information(self, "Remove Curve", "No custom curves to remove."); return
        curve, ok = QInputDialog.getItem(self, "Remove Curve", "Select a FILTER or SENSOR curve to delete:", all_curves, 0, False)
        if not ok or not curve: return
        reply = QMessageBox.question(self, "Confirm Deletion", f"Delete '{curve}'?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
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
        QMessageBox.information(self, "Removed", f"Deleted curve '{curve}'.")

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

    def initialize_wcs_from_header(self, header):
        if header is None:
            print("No FITS header available; cannot build WCS."); return
        try:
            hdr = header.copy()

            # --- normalize deprecated keywords ---
            # RADECSYS → RADESYS (and remove the deprecated key)
            if "RADECSYS" in hdr and "RADESYS" not in hdr:
                radesys_val = str(hdr["RADECSYS"]).strip()
                hdr["RADESYS"] = radesys_val
                try:
                    del hdr["RADECSYS"]
                except Exception:
                    pass

                # If there are alternate WCS solutions (e.g., CTYPE1A/CTYPE2A),
                # also provide RADESYSA / RADESYSB, etc., when missing.
                alt_letters = {k[-1]
                            for k in hdr.keys()
                            if re.match(r"^CTYPE[12][A-Z]$", k)}
                for a in alt_letters:
                    key = f"RADESYS{a}"
                    if key not in hdr:
                        hdr[key] = radesys_val

            # EPOCH → EQUINOX (and remove EPOCH to avoid warnings)
            if "EPOCH" in hdr and "EQUINOX" not in hdr:
                hdr["EQUINOX"] = hdr["EPOCH"]
                try:
                    del hdr["EPOCH"]
                except Exception:
                    pass
            # -------------------------------------  
            self.wcs = WCS(header, naxis=2, relax=True)
            psm = self.wcs.pixel_scale_matrix
            self.pixscale = (np.hypot(psm[0,0], psm[1,0]) * 3600.0)
            self.center_ra, self.center_dec = self.wcs.wcs.crval
            self.wcs_header = self.wcs.to_header(relax=True)
            if 'CROTA2' in header:
                try: self.orientation = float(header['CROTA2'])
                except Exception: self.orientation = None
            else:
                self.orientation = self.calculate_orientation(header)
            if self.orientation is not None:
                self.orientation_label.setText(f"Orientation: {self.orientation:.2f}°")
            else:
                self.orientation_label.setText("Orientation: N/A")
        except Exception as e:
            print("WCS initialization error:\n", e)

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

    def _neutralize_background(self, rgb_img: np.ndarray, patch_size: int = 50) -> np.ndarray:
        img = rgb_img.copy()
        h, w = img.shape[:2]
        ph, pw = h // patch_size, w // patch_size
        min_sum, best_med = np.inf, None
        for i in range(patch_size):
            for j in range(patch_size):
                y0, x0 = i * ph, j * pw
                patch = img[y0:min(y0+ph, h), x0:min(x0+pw, w), :]
                med   = np.median(patch, axis=(0, 1))
                s     = med.sum()
                if s < min_sum:
                    min_sum, best_med = s, med
        if best_med is None:
            return img
        target = float(best_med.mean()); eps = 1e-8
        for c in range(3):
            diff = float(best_med[c] - target)
            if abs(diff) < eps: continue
            img[..., c] = np.clip((img[..., c] - diff) / (1.0 - diff), 0.0, 1.0)
        return img

    # ── SIMBAD/Star fetch ──────────────────────────────────────────────

    def fetch_stars(self):
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
                    with fits.open(p) as hd:
                        for hdu in hd:
                            if (isinstance(hdu, fits.BinTableHDU)
                                    and hdu.header.get("CTYPE", "").upper() == "SED"):
                                extname = hdu.header.get("EXTNAME", None)
                                if extname and extname not in self.pickles_templates:
                                    self.pickles_templates.append(extname)
                except Exception as e:
                    print(f"[fetch_stars] Could not load Pickles templates from {p}: {e}")

        # Build WCS
        try:
            self.initialize_wcs_from_header(self.current_header)
        except Exception:
            QMessageBox.critical(self, "WCS Error", "Could not build a 2D WCS from header."); return

        H, W = self.current_image.shape[:2]
        pix = np.array([[W/2, H/2], [0,0], [W,0], [0,H], [W,H]])
        try:
            sky = self.wcs.all_pix2world(pix, 0)
        except Exception as e:
            QMessageBox.critical(self, "WCS Conversion Error", str(e)); return
        center_sky  = SkyCoord(ra=sky[0,0]*u.deg, dec=sky[0,1]*u.deg, frame="icrs")
        corners_sky = SkyCoord(ra=sky[1:,0]*u.deg, dec=sky[1:,1]*u.deg, frame="icrs")
        radius_deg  = center_sky.separation(corners_sky).max().deg

        # Simbad fields
        Simbad.reset_votable_fields()
        for attempt in range(1, 6):
            try:
                Simbad.add_votable_fields('sp', 'flux(B)', 'flux(V)', 'flux(R)')
                break
            except Exception:
                QApplication.processEvents()
                time.sleep(1.2)
        Simbad.ROW_LIMIT = 10000

        for attempt in range(1, 6):
            try:
                result = Simbad.query_region(center_sky, radius=radius_deg * u.deg)
                break
            except Exception as e:
                self.count_label.setText(f"Attempt {attempt}/5 to query SIMBAD…")
                QApplication.processEvents(); time.sleep(1.2)
                result = None
        if result is None or len(result) == 0:
            QMessageBox.information(self, "No Stars", "SIMBAD returned zero objects in that region.")
            self.star_list = []; self.star_combo.clear(); self.star_combo.addItem("Vega (A0V)", userData="A0V"); return

        def infer_letter(bv):
            if bv is None or (isinstance(bv, float) and np.isnan(bv)): return None
            if   bv < 0.00: return "B"
            elif bv < 0.30: return "A"
            elif bv < 0.58: return "F"
            elif bv < 0.81: return "G"
            elif bv < 1.40: return "K"
            elif bv > 1.40: return "M"
            else: return "U"

        self.star_list = []; templates_for_hist = []
        for row in result:
            raw_sp = row['sp_type']
            bmag, vmag, rmag = row['B'], row['V'], row['R']
            ra_deg, dec_deg  = float(row['ra']), float(row['dec'])
            try:
                sc = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
            except Exception:
                continue

            def _unmask_num(x):
                try:
                    if x is None or np.ma.isMaskedArray(x) and np.ma.is_masked(x):
                        return None
                    return float(x)
                except Exception:
                    return None

            # inside your SIMBAD row loop:
            bmag = _unmask_num(row['B'])
            vmag = _unmask_num(row['V'])

            sp_clean = None
            if raw_sp and str(raw_sp).strip():
                sp = str(raw_sp).strip().upper()
                if not (sp.startswith("SN") or sp.startswith("KA")):
                    sp_clean = sp
            elif bmag is not None and vmag is not None:
                bv = bmag - vmag
                sp_clean = infer_letter(bv)
            if not sp_clean: continue

            match_list = pickles_match_for_simbad(sp_clean, self.pickles_templates)
            best_template = match_list[0] if match_list else None
            xpix, ypix = self.wcs.all_world2pix(sc.ra.deg, sc.dec.deg, 0)
            if 0 <= xpix < W and 0 <= ypix < H:
                self.star_list.append({
                    "ra": sc.ra.deg, "dec": sc.dec.deg, "sp_clean": sp_clean,
                    "pickles_match": best_template, "x": xpix, "y": ypix,
                    "Bmag": float(bmag) if bmag else None,
                    "Vmag": float(vmag) if vmag else None,
                    "Rmag": float(rmag) if rmag else None,
                })
                if best_template is not None: templates_for_hist.append(best_template)

        self.figure.clf()
        if templates_for_hist:
            uniq, cnt = np.unique(templates_for_hist, return_counts=True)
            types_str = ", ".join(uniq)
            self.count_label.setText(f"Found {len(templates_for_hist)} stars; templates: {types_str}")
            ax = self.figure.add_subplot(111)
            ax.bar(uniq, cnt, edgecolor="black")
            ax.set_xlabel("Spectral Type"); ax.set_ylabel("Count"); ax.set_title("Spectral Distribution")
            ax.tick_params(axis='x', rotation=90); ax.grid(axis="y", linestyle="--", alpha=0.3)
            self.canvas.setVisible(True); self.canvas.draw()
        else:
            self.count_label.setText("Found 0 stars with Pickles matches.")
            self.canvas.setVisible(False); self.canvas.draw()

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
            QMessageBox.warning(self, "Error", "Select a reference spectral type (e.g. A0V)."); return
        if r_filt == "(None)" and g_filt == "(None)" and b_filt == "(None)":
            QMessageBox.warning(self, "Error", "Pick at least one of R, G or B filters."); return
        if sens_name == "(None)":
            QMessageBox.warning(self, "Error", "Select a sensor QE curve."); return

        # -- Step 1A: get active image as float32 in [0..1]
        doc = self.doc_manager.get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.critical(self, "Error", "No active document.")
            return

        img = doc.image
        H, W = img.shape[:2]
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.critical(self, "Error", "Active document must be RGB (3 channels).")
            return

        if img.dtype == np.uint8:
            base = img.astype(np.float32) / 255.0
        else:
            base = img.astype(np.float32, copy=True)

        # pedestal removal
        base = np.clip(base - np.min(base, axis=(0,1)), 0.0, None)
        # light neutralization
        base = self._neutralize_background(base, patch_size=10)

        # SEP on grayscale
        gray = np.mean(base, axis=2)
        
        bkg = sep.Background(gray)
        data_sub = gray - bkg.back()
        err = bkg.globalrms

        # 👇 get user threshold (default 5.0)
        if hasattr(self, "sep_thr_spin"):
            sep_sigma = float(self.sep_thr_spin.value())
        else:
            sep_sigma = 5.0
        self.count_label.setText(f"Detecting stars (SEP σ={sep_sigma:.1f})…"); QApplication.processEvents()
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
            QMessageBox.critical(self, "SEP Error", "SEP found no sources."); return
        r_fluxrad, _ = sep.flux_radius(gray, sources["x"], sources["y"], 2.0*sources["a"], 0.5, normflux=sources["flux"], subpix=5)
        mask = (r_fluxrad > .2) & (r_fluxrad <= 10); sources = sources[mask]
        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error", "All SEP detections rejected by radius filter."); return

        if not getattr(self, "star_list", None):
            QMessageBox.warning(self, "Error", "Fetch Stars (with WCS) before running SFCC."); return

        raw_matches = []
        for i, star in enumerate(self.star_list):
            dx = sources["x"] - star["x"]; dy = sources["y"] - star["y"]
            j = np.argmin(dx*dx + dy*dy)
            if (dx[j]**2 + dy[j]**2) < 3.0**2:
                xi, yi = int(round(sources["x"][j])), int(round(sources["y"][j]))
                if 0 <= xi < W and 0 <= yi < H:
                    raw_matches.append({"sim_index": i, "template": star.get("pickles_match") or star["sp_clean"], "x_pix": xi, "y_pix": yi})
        if not raw_matches:
            QMessageBox.warning(self, "No Matches", "No SIMBAD star matched to SEP detections."); return

        wl_min, wl_max = 3000, 11000
        wl_grid = np.arange(wl_min, wl_max+1)

        def load_curve(ext):
            for p in (self.user_custom_path, self.sasp_data_path):
                with fits.open(p) as hd:
                    if ext in hd:
                        d = hd[ext].data
                        wl = _ensure_angstrom(d["WAVELENGTH"].astype(float))
                        tp = d["THROUGHPUT"].astype(float)
                        return wl, tp
            raise KeyError(f"Curve '{ext}' not found")

        def load_sed(ext):
            for p in (self.user_custom_path, self.sasp_data_path):
                with fits.open(p) as hd:
                    if ext in hd:
                        d = hd[ext].data
                        wl = _ensure_angstrom(d["WAVELENGTH"].astype(float))
                        fl = d["FLUX"].astype(float)
                        return wl, fl
            raise KeyError(f"SED '{ext}' not found")

        interp = lambda wl_o, tp_o: np.interp(wl_grid, wl_o, tp_o, left=0., right=0.)
        T_R = interp(*load_curve(r_filt)) if r_filt!="(None)" else np.ones_like(wl_grid)
        T_G = interp(*load_curve(g_filt)) if g_filt!="(None)" else np.ones_like(wl_grid)
        T_B = interp(*load_curve(b_filt)) if b_filt!="(None)" else np.ones_like(wl_grid)
        QE  = interp(*load_curve(sens_name)) if sens_name!="(None)" else np.ones_like(wl_grid)
        LP1 = interp(*load_curve(lp_filt))   if lp_filt != "(None)"  else np.ones_like(wl_grid)
        LP2 = interp(*load_curve(lp_filt2))  if lp_filt2!= "(None)"  else np.ones_like(wl_grid)
        LP  = LP1 * LP2
        T_sys_R, T_sys_G, T_sys_B = T_R*QE*LP, T_G*QE*LP, T_B*QE*LP

        wl_ref, fl_ref = load_sed(ref_sed_name)
        fr_i = np.interp(wl_grid, wl_ref, fl_ref, left=0., right=0.)
        S_ref_R = np.trapezoid(fr_i * T_sys_R, x=wl_grid)
        S_ref_G = np.trapezoid(fr_i * T_sys_G, x=wl_grid)
        S_ref_B = np.trapezoid(fr_i * T_sys_B, x=wl_grid)

        diag_meas_RG, diag_exp_RG = [], []
        diag_meas_BG, diag_exp_BG = [], []
        enriched = []

        for m in raw_matches:
            xi, yi, sp = m["x_pix"], m["y_pix"], m["template"]
            Rm = float(base[yi, xi, 0]); Gm = float(base[yi, xi, 1]); Bm = float(base[yi, xi, 2])
            if Gm <= 0: continue

            cands = pickles_match_for_simbad(sp, getattr(self, "pickles_templates", []))
            if not cands: continue
            wl_s, fl_s = load_sed(cands[0])
            fs_i = np.interp(wl_grid, wl_s, fl_s, left=0., right=0.)
            S_sr = np.trapezoid(fs_i * T_sys_R, x=wl_grid)
            S_sg = np.trapezoid(fs_i * T_sys_G, x=wl_grid)
            S_sb = np.trapezoid(fs_i * T_sys_B, x=wl_grid)
            if S_sg <= 0: continue

            exp_RG = S_sr / S_sg; exp_BG = S_sb / S_sg
            meas_RG = Rm / Gm;    meas_BG = Bm / Gm

            diag_meas_RG.append(meas_RG); diag_exp_RG.append(exp_RG)
            diag_meas_BG.append(meas_BG); diag_exp_BG.append(exp_BG)

            enriched.append({
                **m, "R_meas": Rm, "G_meas": Gm, "B_meas": Bm,
                "S_star_R": S_sr, "S_star_G": S_sg, "S_star_B": S_sb,
                "exp_RG": exp_RG, "exp_BG": exp_BG
            })
        self._last_matched = enriched  # <-- missing in SASpro
        diag_meas_RG = np.array(diag_meas_RG); diag_exp_RG = np.array(diag_exp_RG)
        diag_meas_BG = np.array(diag_meas_BG); diag_exp_BG = np.array(diag_exp_BG)
        if diag_meas_RG.size == 0 or diag_meas_BG.size == 0:
            QMessageBox.information(self, "No Valid Stars", "No stars with valid measured vs expected ratios."); return
        n_stars = diag_meas_RG.size

        def rms_frac(pred, exp): return np.sqrt(np.mean(((pred/exp) - 1.0) ** 2))
        slope_only = lambda x, m: m*x
        affine     = lambda x, m, b: m*x + b
        quad       = lambda x, a, b, c: a*x**2 + b*x + c

        denR = np.sum(diag_meas_RG**2); denB = np.sum(diag_meas_BG**2)
        mR_s = (np.sum(diag_meas_RG * diag_exp_RG) / denR) if denR > 0 else 1.0
        mB_s = (np.sum(diag_meas_BG * diag_exp_BG) / denB) if denB > 0 else 1.0
        rms_s = rms_frac(slope_only(diag_meas_RG, mR_s), diag_exp_RG) + rms_frac(slope_only(diag_meas_BG, mB_s), diag_exp_BG)

        mR_a, bR_a = np.linalg.lstsq(np.vstack([diag_meas_RG, np.ones_like(diag_meas_RG)]).T, diag_exp_RG, rcond=None)[0]
        mB_a, bB_a = np.linalg.lstsq(np.vstack([diag_meas_BG, np.ones_like(diag_meas_BG)]).T, diag_exp_BG, rcond=None)[0]
        rms_a = rms_frac(affine(diag_meas_RG, mR_a, bR_a), diag_exp_RG) + rms_frac(affine(diag_meas_BG, mB_a, bB_a), diag_exp_BG)

        aR_q, bR_q, cR_q = np.polyfit(diag_meas_RG, diag_exp_RG, 2)
        aB_q, bB_q, cB_q = np.polyfit(diag_meas_BG, diag_exp_BG, 2)
        rms_q = rms_frac(quad(diag_meas_RG, aR_q, bR_q, cR_q), diag_exp_RG) + rms_frac(quad(diag_meas_BG, aB_q, bB_q, cB_q), diag_exp_BG)

        idx = np.argmin([rms_s, rms_a, rms_q])
        if idx == 0: coeff_R, coeff_B, model_choice = (0, mR_s, 0), (0, mB_s, 0), "slope-only"
        elif idx == 1: coeff_R, coeff_B, model_choice = (0, mR_a, bR_a), (0, mB_a, bB_a), "affine"
        else: coeff_R, coeff_B, model_choice = (aR_q, bR_q, cR_q), (aB_q, bB_q, cB_q), "quadratic"

        poly = lambda c, x: c[0]*x**2 + c[1]*x + c[2]
        self.figure.clf()
        #ax1 = self.figure.add_subplot(1, 3, 1); bins=20
        #ax1.hist(diag_meas_RG, bins=bins, alpha=.65, label="meas R/G", color="firebrick", edgecolor="black")
        #ax1.hist(diag_exp_RG,  bins=bins, alpha=.55, label="exp R/G",  color="salmon",   edgecolor="black")
        #ax1.hist(diag_meas_BG, bins=bins, alpha=.65, label="meas B/G", color="royalblue", edgecolor="black")
        #ax1.hist(diag_exp_BG,  bins=bins, alpha=.55, label="exp B/G",  color="lightskyblue", edgecolor="black")
        #ax1.set_xlabel("Ratio (band / G)"); ax1.set_ylabel("Count"); ax1.set_title("Measured vs expected"); ax1.legend(fontsize=7, frameon=False)

        res0_RG = (diag_meas_RG / diag_exp_RG) - 1.0
        res0_BG = (diag_meas_BG / diag_exp_BG) - 1.0
        res1_RG = (poly(coeff_R, diag_meas_RG) / diag_exp_RG) - 1.0
        res1_BG = (poly(coeff_B, diag_meas_BG) / diag_exp_BG) - 1.0

        ymin = np.min(np.concatenate([res0_RG, res0_BG])); ymax = np.max(np.concatenate([res0_RG, res0_BG]))
        pad  = 0.05 * (ymax - ymin) if ymax > ymin else 0.02; y_lim = (ymin - pad, ymax + pad)
        def shade(ax, yvals, color):
            q1, q3 = np.percentile(yvals, [25,75]); ax.axhspan(q1, q3, color=color, alpha=.10, zorder=0)

        ax2 = self.figure.add_subplot(1, 2, 1)
        ax2.axhline(0, color="0.65", ls="--", lw=1); shade(ax2, res0_RG, "firebrick"); shade(ax2, res0_BG, "royalblue")
        ax2.scatter(diag_exp_RG, res0_RG, c="firebrick",  marker="o", alpha=.7, label="R/G residual")
        ax2.scatter(diag_exp_BG, res0_BG, c="royalblue", marker="s", alpha=.7, label="B/G residual")
        ax2.set_ylim(*y_lim); ax2.set_xlabel("Expected (band/G)"); ax2.set_ylabel("Frac residual (meas/exp − 1)")
        ax2.set_title("Residuals • BEFORE"); ax2.legend(frameon=False, fontsize=7, loc="lower right")

        ax3 = self.figure.add_subplot(1, 2, 2)
        ax3.axhline(0, color="0.65", ls="--", lw=1); shade(ax3, res1_RG, "firebrick"); shade(ax3, res1_BG, "royalblue")
        ax3.scatter(diag_exp_RG, res1_RG, c="firebrick",  marker="o", alpha=.7)
        ax3.scatter(diag_exp_BG, res1_BG, c="royalblue", marker="s", alpha=.7)
        ax3.set_ylim(*y_lim); ax3.set_xlabel("Expected (band/G)"); ax3.set_ylabel("Frac residual (corrected/exp − 1)")
        ax3.set_title("Residuals • AFTER")
        self.canvas.setVisible(True); self.figure.tight_layout(w_pad=2.); self.canvas.draw()

        self.count_label.setText("Applying SFCC color scales to image…"); QApplication.processEvents()
        if img.dtype == np.uint8: img_float = img.astype(np.float32) / 255.0
        else:                     img_float = img.astype(np.float32)

        RG = img_float[..., 0] / np.maximum(img_float[..., 1], 1e-8)
        BG = img_float[..., 2] / np.maximum(img_float[..., 1], 1e-8)
        aR, bR, cR = coeff_R; aB, bB, cB = coeff_B
        RG_corr = aR*RG**2 + bR*RG + cR
        BG_corr = aB*BG**2 + bB*BG + cB
        calibrated = img_float.copy()
        calibrated[..., 0] = RG_corr * img_float[..., 1]
        calibrated[..., 2] = BG_corr * img_float[..., 1]
        calibrated = np.clip(calibrated, 0, 1)

        if self.neutralize_chk.isChecked():
            calibrated = self._neutralize_background(calibrated, patch_size=10)

        if img.dtype == np.uint8:
            calibrated = (np.clip(calibrated, 0, 1) * 255.0).astype(np.uint8)
        else:
            calibrated = np.clip(calibrated, 0, 1).astype(np.float32)

        new_meta = dict(doc.metadata or {})
        new_meta.update({
            "SFCC_applied": True,
            "SFCC_timestamp": datetime.now().isoformat(),
            "SFCC_model": model_choice,
            "SFCC_coeff_R": [float(v) for v in coeff_R],
            "SFCC_coeff_B": [float(v) for v in coeff_B],
        })

        self.doc_manager.update_active_document(
            calibrated, metadata=new_meta, step_name="SFCC Calibrated"
        )

        self.count_label.setText(f"Applied SFCC color calibration using {n_stars} stars")
        QApplication.processEvents()

        def pretty(coeff): return coeff[0] + coeff[1] + coeff[2]
        ratio_R, ratio_B = pretty(coeff_R), pretty(coeff_B)
        QMessageBox.information(self, "SFCC Complete",
                                f"Applied SFCC using {n_stars} stars\n"
                                f"Model: {model_choice}\n"
                                f"R ratio @ x=1: {ratio_R:.4f}\n"
                                f"B ratio @ x=1: {ratio_B:.4f}\n"
                                f"Background neutralisation: {'ON' if self.neutralize_chk.isChecked() else 'OFF'}")

        self.current_image = calibrated  # keep for gradient step

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
            corrected, metadata=new_meta,
            step_name="Colour-Gradient (star spectra, ¼-res fit)"
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

    def _on_sasp_closed(self, _=None):
        # Called when the SaspViewer window is destroyed
        self.sasp_viewer_window = None

    def closeEvent(self, event):
        super().closeEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Helper to open the dialog from your app
# ──────────────────────────────────────────────────────────────────────────────

def open_sfcc(doc_manager, sasp_data_path: str, parent=None) -> SFCCDialog:
    dlg = SFCCDialog(doc_manager=doc_manager, sasp_data_path=sasp_data_path, parent=parent)
    dlg.show()
    return dlg
