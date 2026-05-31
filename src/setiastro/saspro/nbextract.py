# src/setiastro/saspro/nbextract.py
# SetiAstro Suite Pro — Narrowband Channel Extractor
#
#   ███╗   ██╗██████╗ ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗
#   ████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
#   ██╔██╗ ██║██████╔╝█████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║
#   ██║╚██╗██║██╔══██╗██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║
#   ██║ ╚████║██████╔╝███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║
#   ╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
#
# Copyright (c) 2026 Franklin Marek / SetiAstro
#
# Empirically calibrated dual-band narrowband channel extraction.
#
# Philosophy
# ----------
# Rather than relying on published sensor QE curves (which are often promotional
# approximations), we use stars with known Pickles spectral templates to fit the
# mixing matrix A empirically from the actual image data.  The filter bandwidths
# tell us WHERE to integrate on the SED; the stars tell us HOW MUCH each channel
# responds.  QE uncertainty cancels out because we're fitting ratios against the
# same instrument that took the image.
#
# For a dual-band Ha+OIII filter the system is:
#
#   R_meas = a·Ha_signal  +  b·OIII_signal  +  noise
#   G_meas = c·Ha_signal  +  d·OIII_signal  +  noise
#   B_meas = e·Ha_signal  +  f·OIII_signal  +  noise
#
# With 3 equations and 2 unknowns this is a well-conditioned overdetermined
# system solved per-pixel via non-negative least squares.
#
# Supported filter presets
# ------------------------
#   Ha / OIII   — most common dual-band (L-eXtreme, Antlia ALP-T, ...)
#   SII / OIII  — tri-band variant where SII replaces Ha in the red window
#   SII / Hβ    — less common, but correctly handled
#
# Triband (Ha + OIII + SII in one filter) is explicitly NOT supported for
# 3-line extraction; you still only get 2 reliably separated channels.  The
# tool warns the user accordingly.
#
# Imports shared with sfcc.py
# ---------------------------
# fetch_stars infrastructure, Pickles SED loading, SEP photometry helpers,
# Gaia XP fallback, and the WCS / SIMBAD plumbing all come straight from
# sfcc.py via explicit imports below.  Nothing is duplicated.

from __future__ import annotations

import os
import math
from typing import Optional, Dict, Tuple, List

import numpy as np
from scipy.optimize import nnls

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from astropy.io import fits
from PyQt6.QtCore import Qt, QSettings, QStandardPaths
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QDoubleSpinBox, QPushButton,
    QCheckBox, QMessageBox, QApplication, QGroupBox,
    QSizePolicy,
)
from PyQt6.QtGui import QIcon

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ── shared helpers from sfcc ──────────────────────────────────────────────────
from setiastro.saspro.sfcc import (
    pickles_match_for_simbad,
    measure_star_rgb_photometry,
    _ensure_angstrom,
    _sfcc_status,
    _sfcc_busy,
    _force_mpl_no_tex,
    SFCCDialog,          # we subclass for fetch_stars / Gaia machinery
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Emission line rest wavelengths in nm (air)
LINE_CENTERS_NM: Dict[str, float] = {
    "Ha":   656.28,
    "OIII": 500.70,
    "SII":  671.64,
    "Hb":   486.13,   # Hβ
}

#: Dual-band filter presets  →  (line1_key, line2_key)
FILTER_PRESETS: Dict[str, Tuple[str, str]] = {
    "Ha / OIII":  ("Ha",  "OIII"),
    "SII / OIII": ("SII", "OIII"),
    "SII / Hβ":   ("SII", "Hb"),
}

#: Default bandwidths (FWHM, nm) for each line — user can override
DEFAULT_BW_NM: Dict[str, float] = {
    "Ha":   7.0,
    "OIII": 6.5,
    "SII":  7.0,
    "Hb":   6.5,
}

#: Settings keys
_SK_PRESET  = "NBExtract/FilterPreset"
_SK_BW1     = "NBExtract/BW_Line1"
_SK_BW2     = "NBExtract/BW_Line2"
_SK_CENTER1 = "NBExtract/Center_Line1"
_SK_CENTER2 = "NBExtract/Center_Line2"


# ─────────────────────────────────────────────────────────────────────────────
# Core maths
# ─────────────────────────────────────────────────────────────────────────────

def integrate_sed_over_window(
    sed_wl_nm: np.ndarray,
    sed_fl: np.ndarray,
    center_nm: float,
    bw_nm: float,
) -> float:
    """
    Integrate a Pickles SED over a top-hat passband window.

    Parameters
    ----------
    sed_wl_nm : wavelength array in nm
    sed_fl    : flux array (arbitrary units; only ratios matter)
    center_nm : line centre in nm
    bw_nm     : FWHM bandwidth in nm

    Returns
    -------
    Integrated flux (same units as sed_fl × nm).  Returns 0.0 if the
    window falls entirely outside the SED coverage.
    """
    lo = center_nm - bw_nm * 0.5
    hi = center_nm + bw_nm * 0.5

    wl = np.asarray(sed_wl_nm, dtype=np.float64)
    fl = np.asarray(sed_fl,    dtype=np.float64)

    mask = (wl >= lo) & (wl <= hi)
    if not np.any(mask):
        # Fallback: point estimate via linear interpolation
        return float(np.interp(center_nm, wl, fl, left=0.0, right=0.0) * bw_nm)

    wl_w = wl[mask].copy()
    fl_w = fl[mask].copy()

    # Add boundary points for accurate trapezoidal integration
    if wl_w[0] > lo:
        fl_lo = float(np.interp(lo, wl, fl))
        wl_w = np.concatenate([[lo], wl_w])
        fl_w = np.concatenate([[fl_lo], fl_w])
    if wl_w[-1] < hi:
        fl_hi = float(np.interp(hi, wl, fl))
        wl_w = np.concatenate([wl_w, [hi]])
        fl_w = np.concatenate([fl_w, [fl_hi]])

    return float(_trapz(fl_w, x=wl_w))


def fit_mixing_matrix(
    star_records: List[Dict],
) -> Optional[np.ndarray]:
    """
    Fit the 3x2 empirical mixing matrix A from stellar photometry.

    Each record in star_records must contain:
        "R_meas", "G_meas", "B_meas"   — background-subtracted aperture fluxes
        "S_line1", "S_line2"           — Pickles SED integrals over the two
                                          filter windows (same arbitrary units)

    The forward model is:
        [R]   [a  b]   [k1]
        [G] = [c  d] * [k2]
        [B]   [e  f]

    where k1, k2 are the effective continuum flux densities in each passband.
    For a pure continuum star (no emission lines) k1 and k2 equal the Pickles
    SED integrals S_line1 and S_line2, so we fit:

        measured_channel ≈ gain_channel_line1 * S_line1
                         + gain_channel_line2 * S_line2

    We solve this as a non-negative least squares problem per channel so that
    all mixing coefficients stay physically meaningful (>= 0).

    Returns
    -------
    A : ndarray shape (3, 2)  or None if fitting fails
        A[channel, line] = sensitivity of channel to that line
        Channels: 0=R, 1=G, 2=B
        Lines:    0=line1 (e.g. Ha), 1=line2 (e.g. OIII)
    """
    if not star_records:
        return None

    S1 = np.array([s["S_line1"] for s in star_records], dtype=np.float64)
    S2 = np.array([s["S_line2"] for s in star_records], dtype=np.float64)
    Rm = np.array([s["R_meas"]  for s in star_records], dtype=np.float64)
    Gm = np.array([s["G_meas"]  for s in star_records], dtype=np.float64)
    Bm = np.array([s["B_meas"]  for s in star_records], dtype=np.float64)

    # Design matrix  (n_stars x 2)
    X = np.column_stack([S1, S2])

    A = np.zeros((3, 2), dtype=np.float64)

    # Fit each channel independently via NNLS
    for ch_idx, y in enumerate((Rm, Gm, Bm)):
        try:
            coeffs, _residual = nnls(X, y)
            A[ch_idx] = coeffs
        except Exception as e:
            print(f"[NBExtract] NNLS failed for channel {ch_idx}: {e}")
            return None

    # Sanity checks
    if np.any(~np.isfinite(A)):
        print("[NBExtract] Mixing matrix contains non-finite values.")
        return None
    if np.linalg.matrix_rank(A) < 2:
        print("[NBExtract] Mixing matrix is rank-deficient.")
        return None

    return A


def extract_channels_nnls(
    img_rgb: np.ndarray,
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the per-pixel 3x2 system for line1 and line2 channel images.

    Uses the Moore-Penrose pseudo-inverse of A followed by non-negativity
    clipping.  This is equivalent to per-pixel NNLS but vectorised over
    the entire image at once, making it fast enough for 50+ MP images.

    Parameters
    ----------
    img_rgb : float32 array (H, W, 3), values in [0, 1]
    A       : (3, 2) mixing matrix from fit_mixing_matrix()

    Returns
    -------
    line1_img, line2_img : float32 arrays (H, W)
        Values normalised to [0, 1] by 99.9th percentile.
    """
    H, W = img_rgb.shape[:2]
    # Reshape to (N, 3) where N = H*W
    pixels = img_rgb.reshape(-1, 3).astype(np.float64)

    # Pseudo-inverse: (2, 3)
    A_pinv = np.linalg.pinv(A)

    # Solve: (N, 2) = (N, 3) @ (3, 2)
    out = (A_pinv @ pixels.T).T      # (N, 2)
    out = np.clip(out, 0.0, None)    # enforce non-negativity

    line1 = out[:, 0].reshape(H, W).astype(np.float32)
    line2 = out[:, 1].reshape(H, W).astype(np.float32)

    def _norm_to_01(ch: np.ndarray) -> np.ndarray:
        p = float(np.percentile(ch, 99.9))
        if p > 0:
            return np.clip(ch / p, 0.0, 1.0)
        return ch

    return _norm_to_01(line1), _norm_to_01(line2)


def condition_number_warning(A: np.ndarray) -> Optional[str]:
    """
    Return a human-readable warning string if the mixing matrix is
    ill-conditioned, or None if it looks acceptable.
    """
    try:
        k = float(np.linalg.cond(A))
        if k > 50:
            return (
                f"Condition number is {k:.1f} — extraction may amplify noise.\n"
                "Consider using more calibration stars or adjusting bandwidths."
            )
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Dialog
# ─────────────────────────────────────────────────────────────────────────────

class NBExtractDialog(SFCCDialog):
    """
    Narrowband Channel Extractor dialog.

    Subclasses SFCCDialog to inherit:
        fetch_stars()                   — SIMBAD + Gaia XP star catalogue
        _make_working_base_for_sep()    — SEP-ready image preparation
        _load_nb_settings / save        — QSettings persistence
        Pickles FITS loading machinery
        WCS initialisation

    Adds:
        Filter type + bandwidth UI
        Step 2: empirical mixing matrix calibration
        Step 3: per-pixel NNLS channel extraction
        Two new output documents (line1, line2)
    """

    def __init__(self, doc_manager, sasp_data_path: str, parent=None):
        super().__init__(doc_manager, sasp_data_path, parent)

        _force_mpl_no_tex()
        self.setWindowTitle("Narrowband Channel Extractor (NBExtract)")
        self.setMinimumSize(740, 560)

        # Internal state
        self._A_matrix: Optional[np.ndarray] = None
        self._nb_star_records: List[Dict] = []

        self._inject_nb_ui()
        self._load_nb_settings()

    # ── UI injection ──────────────────────────────────────────────────────────

    def _inject_nb_ui(self):
        """
        Hide SFCC-specific controls that don't apply here and inject the
        NBExtract filter configuration group above the status label.
        """
        # Widgets from SFCCDialog that we don't need
        _sfcc_only = [
            "r_filter_combo", "g_filter_combo", "b_filter_combo",
            "sens_combo", "lp_filter_combo", "lp_filter_combo2",
            "run_spcc_btn", "run_grad_btn", "grad_method_combo",
            "neutralize_chk", "open_sasp_btn",
            "add_curve_btn", "remove_curve_btn",
        ]
        for attr in _sfcc_only:
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.hide()
                except Exception:
                    pass

        self._hide_sfcc_filter_labels()

        # ── Group box: filter configuration ──────────────────────────────
        grp = QGroupBox("Narrowband Filter Configuration")
        grp_lay = QVBoxLayout(grp)

        # Preset selector row
        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("Filter type:"))
        self.nb_preset_combo = QComboBox()
        for name in FILTER_PRESETS:
            self.nb_preset_combo.addItem(name)
        self.nb_preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        row_preset.addWidget(self.nb_preset_combo)
        row_preset.addStretch()
        grp_lay.addLayout(row_preset)

        # Line 1 row
        row_l1 = QHBoxLayout()
        self.nb_line1_label = QLabel("Ha centre (nm):")
        row_l1.addWidget(self.nb_line1_label)
        self.nb_center1_spin = QDoubleSpinBox()
        self.nb_center1_spin.setRange(400.0, 800.0)
        self.nb_center1_spin.setDecimals(2)
        self.nb_center1_spin.setSingleStep(0.1)
        self.nb_center1_spin.setSuffix(" nm")
        row_l1.addWidget(self.nb_center1_spin)
        row_l1.addSpacing(16)
        row_l1.addWidget(QLabel("Bandwidth (FWHM):"))
        self.nb_bw1_spin = QDoubleSpinBox()
        self.nb_bw1_spin.setRange(1.0, 30.0)
        self.nb_bw1_spin.setDecimals(1)
        self.nb_bw1_spin.setSingleStep(0.5)
        self.nb_bw1_spin.setSuffix(" nm")
        row_l1.addWidget(self.nb_bw1_spin)
        row_l1.addStretch()
        grp_lay.addLayout(row_l1)

        # Line 2 row
        row_l2 = QHBoxLayout()
        self.nb_line2_label = QLabel("OIII centre (nm):")
        row_l2.addWidget(self.nb_line2_label)
        self.nb_center2_spin = QDoubleSpinBox()
        self.nb_center2_spin.setRange(400.0, 800.0)
        self.nb_center2_spin.setDecimals(2)
        self.nb_center2_spin.setSingleStep(0.1)
        self.nb_center2_spin.setSuffix(" nm")
        row_l2.addWidget(self.nb_center2_spin)
        row_l2.addSpacing(16)
        row_l2.addWidget(QLabel("Bandwidth (FWHM):"))
        self.nb_bw2_spin = QDoubleSpinBox()
        self.nb_bw2_spin.setRange(1.0, 30.0)
        self.nb_bw2_spin.setDecimals(1)
        self.nb_bw2_spin.setSingleStep(0.5)
        self.nb_bw2_spin.setSuffix(" nm")
        row_l2.addWidget(self.nb_bw2_spin)
        row_l2.addStretch()
        grp_lay.addLayout(row_l2)

        # Action buttons row
        row_act = QHBoxLayout()

        self.nb_calibrate_btn = QPushButton("Step 2: Calibrate Mixing Matrix")
        f = self.nb_calibrate_btn.font()
        f.setBold(True)
        self.nb_calibrate_btn.setFont(f)
        self.nb_calibrate_btn.clicked.connect(self._calibrate_mixing_matrix)
        row_act.addWidget(self.nb_calibrate_btn)

        self.nb_extract_btn = QPushButton("Step 3: Extract Channels")
        f2 = self.nb_extract_btn.font()
        f2.setBold(True)
        self.nb_extract_btn.setFont(f2)
        self.nb_extract_btn.setEnabled(False)   # enabled after calibration
        self.nb_extract_btn.clicked.connect(self._extract_channels)
        row_act.addWidget(self.nb_extract_btn)

        self.nb_normalize_chk = QCheckBox("Normalize outputs to [0,1]")
        self.nb_normalize_chk.setChecked(True)
        row_act.addWidget(self.nb_normalize_chk)
        row_act.addStretch()
        grp_lay.addLayout(row_act)

        # Matrix readout
        self.nb_matrix_label = QLabel("Mixing matrix: (not yet calibrated)")
        self.nb_matrix_label.setWordWrap(True)
        grp_lay.addWidget(self.nb_matrix_label)

        # Insert group above the status label
        layout = self.layout()
        idx = self._find_widget_index_in_layout(layout, self.count_label)
        if idx >= 0:
            layout.insertWidget(idx, grp)
        else:
            layout.addWidget(grp)

        # Kick the labels into their initial state
        self._on_preset_changed()

    def _hide_sfcc_filter_labels(self):
        """Hide QLabel widgets that belong to SFCC filter/sensor rows."""
        layout = self.layout()
        if layout is None:
            return
        _sfcc_label_fragments = (
            "R Filter", "G Filter", "B Filter",
            "Sensor", "LP/Cut", "Select White",
        )
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            sub = item.layout()
            if sub is None:
                continue
            for j in range(sub.count()):
                sub_item = sub.itemAt(j)
                if sub_item is None:
                    continue
                w = sub_item.widget()
                if isinstance(w, QLabel):
                    if any(frag in w.text() for frag in _sfcc_label_fragments):
                        w.hide()

    @staticmethod
    def _find_widget_index_in_layout(layout, target) -> int:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget() is target:
                return i
        return -1

    # ── Preset / settings ─────────────────────────────────────────────────────

    def _on_preset_changed(self, _=None):
        preset_name = self.nb_preset_combo.currentText()
        if preset_name not in FILTER_PRESETS:
            return
        l1_key, l2_key = FILTER_PRESETS[preset_name]

        self.nb_line1_label.setText(f"{l1_key} centre (nm):")
        self.nb_line2_label.setText(f"{l2_key} centre (nm):")

        for sp in (self.nb_center1_spin, self.nb_center2_spin,
                   self.nb_bw1_spin, self.nb_bw2_spin):
            sp.blockSignals(True)

        self.nb_center1_spin.setValue(LINE_CENTERS_NM[l1_key])
        self.nb_center2_spin.setValue(LINE_CENTERS_NM[l2_key])
        self.nb_bw1_spin.setValue(DEFAULT_BW_NM[l1_key])
        self.nb_bw2_spin.setValue(DEFAULT_BW_NM[l2_key])

        for sp in (self.nb_center1_spin, self.nb_center2_spin,
                   self.nb_bw1_spin, self.nb_bw2_spin):
            sp.blockSignals(False)

        # Invalidate any existing calibration when filter type changes
        self._A_matrix = None
        self.nb_extract_btn.setEnabled(False)
        self.nb_matrix_label.setText("Mixing matrix: (not yet calibrated)")

    def _load_nb_settings(self):
        s = QSettings()
        preset = s.value(_SK_PRESET, "Ha / OIII")
        idx = self.nb_preset_combo.findText(preset)
        if idx >= 0:
            self.nb_preset_combo.setCurrentIndex(idx)
        # Per-axis overrides (applied after preset sets defaults)
        for key, spin in (
            (_SK_BW1,     self.nb_bw1_spin),
            (_SK_BW2,     self.nb_bw2_spin),
            (_SK_CENTER1, self.nb_center1_spin),
            (_SK_CENTER2, self.nb_center2_spin),
        ):
            val = s.value(key, None)
            if val is not None:
                try:
                    spin.setValue(float(val))
                except Exception:
                    pass

    def _save_nb_settings(self):
        s = QSettings()
        s.setValue(_SK_PRESET,  self.nb_preset_combo.currentText())
        s.setValue(_SK_BW1,     self.nb_bw1_spin.value())
        s.setValue(_SK_BW2,     self.nb_bw2_spin.value())
        s.setValue(_SK_CENTER1, self.nb_center1_spin.value())
        s.setValue(_SK_CENTER2, self.nb_center2_spin.value())

    # ── SED loading — nm units ────────────────────────────────────────────────

    def _load_sed_nm(self, extname: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a Pickles SED from the SASpro FITS data files.

        Returns (wavelength_nm, flux) where wavelength is guaranteed to be
        in nm regardless of how it was stored (Angstrom or nm).
        """
        for path in (self.user_custom_path, self.sasp_data_path):
            try:
                with fits.open(path, memmap=False) as hd:
                    if extname in hd:
                        d      = hd[extname].data
                        wl_raw = d["WAVELENGTH"].astype(np.float64)
                        fl     = d["FLUX"].astype(np.float64)
                        # _ensure_angstrom converts nm→Å if needed; we then
                        # divide by 10 to normalise back to nm for our maths.
                        wl_ang = _ensure_angstrom(wl_raw)
                        wl_nm  = wl_ang / 10.0
                        return wl_nm, fl
            except Exception:
                continue
        raise KeyError(f"SED extension '{extname}' not found in FITS data files.")

    # ── Step 2: calibrate mixing matrix ──────────────────────────────────────

    def _calibrate_mixing_matrix(self):
        """
        For each matched star:
            1. Run SEP aperture photometry → measured R, G, B
            2. Integrate Pickles SED over each filter window → S_line1, S_line2
            3. Collect all records and fit the 3x2 A matrix via NNLS
        """
        if not getattr(self, "star_list", None):
            QMessageBox.warning(
                self, "No Stars",
                "Please run Step 1: Fetch Stars first.\n"
                "The image must be plate-solved."
            )
            return

        doc = self.doc_manager.get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.critical(self, "Error", "No active document.")
            return

        img = doc.image
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.critical(self, "Error", "Active document must be RGB (3 channels).")
            return

        center1_nm  = float(self.nb_center1_spin.value())
        center2_nm  = float(self.nb_center2_spin.value())
        bw1_nm      = float(self.nb_bw1_spin.value())
        bw2_nm      = float(self.nb_bw2_spin.value())
        preset_name = self.nb_preset_combo.currentText()
        l1_key, l2_key = FILTER_PRESETS.get(preset_name, ("Ha", "OIII"))

        _sfcc_status(self, "NBExtract: preparing image for star photometry…")
        QApplication.processEvents()

        img_float = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)
        base = self._make_working_base_for_sep(img_float)

        import sep
        gray     = np.mean(base, axis=2).astype(np.float32)
        bkg      = sep.Background(gray)
        data_sub = gray - bkg.back()
        err      = float(bkg.globalrms)

        sep_sigma = float(self.sep_thr_spin.value()) if hasattr(self, "sep_thr_spin") else 5.0
        _sfcc_status(self, f"NBExtract: detecting stars (SEP σ={sep_sigma:.1f})…")
        QApplication.processEvents()

        sources = sep.extract(data_sub, sep_sigma, err=err)
        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error", "SEP found no sources.")
            return

        r_fluxrad, _ = sep.flux_radius(
            gray, sources["x"], sources["y"],
            2.0 * sources["a"], 0.5,
            normflux=sources["flux"], subpix=5,
        )
        mask    = (r_fluxrad > 0.2) & (r_fluxrad <= 10)
        sources = sources[mask]
        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error",
                                 "All SEP detections rejected by radius filter.")
            return

        # ── Match SIMBAD stars → SEP detections → Pickles photometry ─────
        star_records = []
        n_no_template = 0

        for star in self.star_list:
            tmpl = star.get("pickles_match")
            if tmpl is None:
                n_no_template += 1
                continue

            dx = sources["x"] - star["x"]
            dy = sources["y"] - star["y"]
            j  = int(np.argmin(dx * dx + dy * dy))
            if (dx[j] ** 2 + dy[j] ** 2) >= 9.0:   # > 3 px tolerance
                continue

            x    = float(sources["x"][j])
            y    = float(sources["y"][j])
            a    = float(sources["a"][j])
            r    = float(np.clip(2.5 * a, 2.0, 12.0))
            rin  = float(np.clip(3.0 * r, 6.0, 40.0))
            rout = float(np.clip(5.0 * r, rin + 2.0, 60.0))

            phot = measure_star_rgb_photometry(base, x, y, r, rin, rout)
            if phot is None:
                continue

            Rm = float(phot["R"]["star_sum"])
            Gm = float(phot["G"]["star_sum"])
            Bm = float(phot["B"]["star_sum"])

            if not (np.isfinite(Rm) and np.isfinite(Gm) and np.isfinite(Bm)):
                continue
            if Rm <= 0 or Gm <= 0 or Bm <= 0:
                continue

            # Load Pickles SED and integrate over each narrow window
            try:
                wl_nm, fl = self._load_sed_nm(tmpl)
            except Exception as e:
                print(f"[NBExtract] Cannot load SED '{tmpl}': {e}")
                continue

            S1 = integrate_sed_over_window(wl_nm, fl, center1_nm, bw1_nm)
            S2 = integrate_sed_over_window(wl_nm, fl, center2_nm, bw2_nm)

            if S1 <= 0.0 or S2 <= 0.0:
                continue

            star_records.append({
                "template": tmpl,
                "x": x,
                "y": y,
                "R_meas":  Rm,
                "G_meas":  Gm,
                "B_meas":  Bm,
                "S_line1": S1,
                "S_line2": S2,
            })

        n_used = len(star_records)
        _sfcc_status(
            self,
            f"NBExtract: {n_used} stars usable for calibration "
            f"({n_no_template} had no Pickles template).",
        )
        QApplication.processEvents()

        if n_used < 6:
            QMessageBox.warning(
                self, "Too Few Stars",
                f"Only {n_used} usable calibration stars (need ≥ 6).\n\n"
                "Try:\n"
                "  • Lowering the SEP detection threshold\n"
                "  • Re-plate-solving the image\n"
                "  • Enabling the Gaia XP fallback"
            )
            return

        # ── Fit ───────────────────────────────────────────────────────────
        A = fit_mixing_matrix(star_records)
        if A is None:
            QMessageBox.critical(
                self, "Calibration Failed",
                "Could not fit a valid mixing matrix.\n"
                "Check that the star sample is diverse enough in colour."
            )
            return

        self._A_matrix        = A
        self._nb_star_records = star_records
        self._save_nb_settings()

        # ── Diagnostics ───────────────────────────────────────────────────
        self._plot_calibration_diagnostics(star_records, A, l1_key, l2_key)

        # ── Matrix readout ────────────────────────────────────────────────
        k    = float(np.linalg.cond(A))
        warn = condition_number_warning(A)
        lines = [
            f"Mixing matrix A  (rows = R, G, B  |  cols = {l1_key}, {l2_key}):",
            f"  R : {A[0,0]:+.5f}   {A[0,1]:+.5f}",
            f"  G : {A[1,0]:+.5f}   {A[1,1]:+.5f}",
            f"  B : {A[2,0]:+.5f}   {A[2,1]:+.5f}",
            f"  Condition number : {k:.2f}",
        ]
        if warn:
            lines.append(f"  ⚠  {warn}")
        self.nb_matrix_label.setText("\n".join(lines))

        self.nb_extract_btn.setEnabled(True)

        summary = (
            f"Mixing matrix fitted from {n_used} stars.\n"
            f"Condition number: {k:.2f}\n\n"
        )
        summary += warn if warn else "Matrix looks well-conditioned ✓"
        QMessageBox.information(self, "Calibration Complete", summary)

    def _plot_calibration_diagnostics(
        self,
        star_records: List[Dict],
        A: np.ndarray,
        l1_key: str,
        l2_key: str,
    ):
        """
        Three-panel scatter: measured vs model-predicted flux per channel.
        A tight 1:1 line indicates the mixing matrix is well-calibrated.
        """
        _force_mpl_no_tex()
        self.figure.clf()

        S1  = np.array([s["S_line1"] for s in star_records])
        S2  = np.array([s["S_line2"] for s in star_records])
        X   = np.column_stack([S1, S2])          # (N, 2)
        pred = X @ A.T                            # (N, 3)

        meas = np.column_stack([
            [s["R_meas"] for s in star_records],
            [s["G_meas"] for s in star_records],
            [s["B_meas"] for s in star_records],
        ])                                         # (N, 3)

        palette   = ("firebrick", "seagreen", "royalblue")
        ch_labels = ("R channel", "G channel", "B channel")

        for i, (color, label) in enumerate(zip(palette, ch_labels)):
            ax = self.figure.add_subplot(1, 3, i + 1)
            ax.scatter(
                pred[:, i], meas[:, i],
                c=color, alpha=0.7, s=18, edgecolors="none",
            )
            lo = min(float(pred[:, i].min()), float(meas[:, i].min()))
            hi = max(float(pred[:, i].max()), float(meas[:, i].max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="1:1")
            ax.set_xlabel("Model predicted flux")
            ax.set_ylabel("Measured flux")
            ax.set_title(label)
            ax.grid(True, linestyle="--", alpha=0.3)

        self.figure.suptitle(
            f"NBExtract calibration: predicted vs measured  ({l1_key} / {l2_key})",
            fontsize=10,
        )
        self.figure.tight_layout()
        self.canvas.setVisible(True)
        self.canvas.draw()

    # ── Step 3: extract channels ──────────────────────────────────────────────

    def _extract_channels(self):
        """
        Apply the calibrated A matrix to every pixel in the active image,
        producing two new mono documents: one per emission line.
        """
        if self._A_matrix is None:
            QMessageBox.warning(self, "Not Calibrated",
                                "Please run Step 2: Calibrate Mixing Matrix first.")
            return

        doc = self.doc_manager.get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.critical(self, "Error", "No active document.")
            return

        img = doc.image
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.critical(self, "Error",
                                 "Active document must be RGB (3 channels).")
            return

        preset_name = self.nb_preset_combo.currentText()
        l1_key, l2_key = FILTER_PRESETS.get(preset_name, ("Ha", "OIII"))

        img_float = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)

        _sfcc_status(self, "NBExtract: solving per-pixel mixing system…")
        QApplication.processEvents()

        normalize = self.nb_normalize_chk.isChecked()

        if normalize:
            line1_img, line2_img = extract_channels_nnls(img_float, self._A_matrix)
        else:
            # Raw pseudo-flux without percentile normalisation
            H, W = img_float.shape[:2]
            pixels  = img_float.reshape(-1, 3).astype(np.float64)
            A_pinv  = np.linalg.pinv(self._A_matrix)
            out     = np.clip((A_pinv @ pixels.T).T, 0.0, None)
            line1_img = out[:, 0].reshape(H, W).astype(np.float32)
            line2_img = out[:, 1].reshape(H, W).astype(np.float32)

        meta_base = dict(doc.metadata or {})
        A_list    = self._A_matrix.tolist()
        n_cal     = len(self._nb_star_records)

        meta1 = {
            **meta_base,
            "NBExtract_line":       l1_key,
            "NBExtract_preset":     preset_name,
            "NBExtract_center_nm":  float(self.nb_center1_spin.value()),
            "NBExtract_bw_nm":      float(self.nb_bw1_spin.value()),
            "NBExtract_n_cal_stars": n_cal,
            "NBExtract_A_matrix":   A_list,
        }
        meta2 = {
            **meta_base,
            "NBExtract_line":       l2_key,
            "NBExtract_preset":     preset_name,
            "NBExtract_center_nm":  float(self.nb_center2_spin.value()),
            "NBExtract_bw_nm":      float(self.nb_bw2_spin.value()),
            "NBExtract_n_cal_stars": n_cal,
            "NBExtract_A_matrix":   A_list,
        }

        self._push_new_document(line1_img, meta1, f"{l1_key} (NBExtract)")
        self._push_new_document(line2_img, meta2, f"{l2_key} (NBExtract)")

        _sfcc_status(
            self,
            f"NBExtract complete — '{l1_key}' and '{l2_key}' channels created "
            f"from {n_cal} calibration stars.",
        )

        QMessageBox.information(
            self, "Extraction Complete",
            f"Two new documents created:\n"
            f"  •  {l1_key}  (NBExtract)\n"
            f"  •  {l2_key}  (NBExtract)\n\n"
            f"Calibrated from {n_cal} stars.\n"
            f"Condition number: {np.linalg.cond(self._A_matrix):.2f}\n\n"
            f"Tip: for best results, apply a gentle denoise pass to each\n"
            f"extracted channel before combining into a palette."
        )

    def _push_new_document(
        self,
        img_mono: np.ndarray,
        metadata: dict,
        title: str,
    ):
        """
        Open a new SASpro document from a mono float32 array.
        Tries the standard doc manager API variants in order.
        """
        dm = self.doc_manager
        for method_name in ("open_new_document", "create_document", "add_document"):
            method = getattr(dm, method_name, None)
            if method is not None:
                try:
                    method(img_mono, metadata=metadata, title=title)
                    return
                except Exception as e:
                    print(f"[NBExtract] {method_name} failed: {e}")

        # Last resort — update the active document (avoids a crash but is
        # not ideal; the caller will see one channel overwrite the other)
        print(f"[NBExtract] WARNING: no new-document API found; "
              f"falling back to update_active_document for '{title}'")
        dm.update_active_document(
            img_mono, metadata=metadata, step_name=title
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def open_nbextract(
    doc_manager,
    sasp_data_path: str,
    parent=None,
) -> NBExtractDialog:
    """
    Instantiate and show the Narrowband Channel Extractor dialog.

    Typical call from a toolbar action or menu item::

        from setiastro.saspro.nbextract import open_nbextract
        self._nb_dlg = open_nbextract(
            self.doc_manager, self._sasp_data_path, parent=self
        )
    """
    dlg = NBExtractDialog(
        doc_manager=doc_manager,
        sasp_data_path=sasp_data_path,
        parent=parent,
    )
    dlg.show()
    return dlg