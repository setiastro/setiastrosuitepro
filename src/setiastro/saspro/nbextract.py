# nbextract.py
# SetiAstro Suite Pro — Narrowband Channel Extractor
#
#   ███╗   ██╗██████╗ ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗
#   ████╗  ██║██╔══██╗██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
#   ██╔██╗ ██║██████╔╝█████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║
#   ██║╚██╗██║██╔══██╗██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║
#   ██║ ╚████║██████╔╝███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║
#   ╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝
#
# Copyright (c) 2024 Franklin Marek / SetiAstro
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
    QSizePolicy, QWidget,
)
from PyQt6.QtGui import QIcon

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ── shared helpers from sfcc ──────────────────────────────────────────────────
from setiastro.saspro.sfcc import (
    pickles_match_for_simbad,
    measure_star_rgb_photometry,
    measure_star_rgb_raw_aperture,
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
        return float(np.interp(center_nm, wl, fl, left=0.0, right=0.0) * bw_nm)

    wl_w = wl[mask].copy()
    fl_w = fl[mask].copy()

    if wl_w[0] > lo:
        fl_lo = float(np.interp(lo, wl, fl))
        wl_w = np.concatenate([[lo], wl_w])
        fl_w = np.concatenate([[fl_lo], fl_w])
    if wl_w[-1] < hi:
        fl_hi = float(np.interp(hi, wl, fl))
        wl_w = np.concatenate([wl_w, [hi]])
        fl_w = np.concatenate([fl_w, [fl_hi]])

    return float(_trapz(fl_w, x=wl_w))


def auto_q_from_condition_number(k: float) -> float:
    """
    Global Q from condition number — used as a baseline / fallback.
    Formula: Q = clip(1 / (1 + 0.1 * k), 0.2, 0.9)
    """
    return float(np.clip(1.0 / (1.0 + 0.1 * k), 0.2, 0.9))


def auto_q_per_channel(A: np.ndarray) -> Tuple[float, float]:
    """
    Derive independent Q values for line1 and line2 by examining how
    well each line direction is resolved in the mixing matrix.

    Uses the SVD of A to measure the relative strength of each singular
    direction.  The first singular vector captures whichever line dominates
    (usually Ha/SII in R), the second captures the cross-channel separation
    (usually OIII/Hβ in G/B).

    The second singular value is almost always smaller — meaning OIII is
    harder to separate.  We penalise Q₂ more aggressively to prevent the
    over-subtraction that shows up as dark holes in the OIII channel.

    Additionally we look directly at the G and B rows of the OIII column:
    if they are nearly identical the pseudo-inverse has no independent
    information to separate OIII from Ha using those channels, so we
    drive Q₂ toward zero regardless of the global condition number.

    Returns
    -------
    (q1, q2) : floats in [0.2, 0.9]
        q1 — mixing strength for line1 (Ha/SII dominant)
        q2 — mixing strength for line2 (OIII/Hβ dominant)
    """
    try:
        # ── Global condition number → baseline Q ─────────────────────────
        k = float(np.linalg.cond(A))
        q_global = float(np.clip(1.0 / (1.0 + 0.1 * k), 0.2, 0.9))

        # ── SVD: how independently are the two lines resolved? ────────────
        _, sv, _ = np.linalg.svd(A, full_matrices=False)   # sv shape (2,)
        sv1, sv2 = float(sv[0]), float(sv[1])

        # Ratio of singular values: how much weaker is the second direction?
        # sv_ratio = 1.0 means both lines equally well-resolved (ideal)
        # sv_ratio → 0 means second line barely separable
        sv_ratio = sv2 / sv1 if sv1 > 0 else 0.0

        # Q₁ (line1 / Ha): mainly depends on global conditioning
        # Cap at 0.9 even for best case — small regulariser always helps
        q1 = float(np.clip(q_global, 0.2, 0.9))

        # Q₂ (line2 / OIII): penalised by how weak the second singular value is
        # sv_ratio=1.0 → no extra penalty; sv_ratio=0.1 → heavy penalty
        # Multiply q_global by sv_ratio so poor second-direction resolution
        # drives Q₂ toward the conservative end independently of Q₁
        q2_svd = float(np.clip(q_global * sv_ratio, 0.2, 0.9))

        # ── G/B degeneracy check ──────────────────────────────────────────
        # If G-row and B-row OIII entries are nearly identical the pseudo-
        # inverse is extrapolating, not measuring.  Scale Q₂ down further.
        g_oiii = float(A[1, 1])   # G channel, line2 (OIII)
        b_oiii = float(A[2, 1])   # B channel, line2 (OIII)

        oiii_sum = abs(g_oiii) + abs(b_oiii)
        if oiii_sum > 1e-10:
            # similarity = 1 when G and B OIII entries are identical
            # similarity = 0 when they differ maximally
            similarity = 1.0 - abs(g_oiii - b_oiii) / oiii_sum
        else:
            similarity = 1.0   # both zero — completely degenerate

        # Penalise Q₂ by how similar G and B are in the OIII column.
        # Squared similarity mirrors the noise amplification physics — matrix
        # inversion noise gain scales as condition^2, so the penalty should too.
        # similarity=0.0 (very different) → no penalty,   q2 = q2_svd
        # similarity=0.7 (moderate)       → factor 0.608, q2 = 0.608 * q2_svd
        # similarity=0.9 (nearly same)    → factor 0.352, q2 = 0.352 * q2_svd
        # similarity=1.0 (identical)      → factor 0.2,   q2 = floor 0.2
        q2 = float(np.clip(q2_svd * (1.0 - similarity**2 * 0.8), 0.2, 0.9))


        print(
            f"[NBExtract] Auto-Q: k={k:.2f}  sv_ratio={sv_ratio:.3f}  "
            f"GB_similarity={similarity:.3f}  →  Q₁={q1:.2f}  Q₂={q2:.2f}"
        )

        return q1, q2

    except Exception as e:
        print(f"[NBExtract] auto_q_per_channel failed ({e}), using global fallback")
        k = float(np.linalg.cond(A)) if A is not None else 10.0
        q = auto_q_from_condition_number(k)
        return q, q


def fit_mixing_matrix(
    star_records: List[Dict],
    *,
    sigma_clip: float = 3.0,
    max_iter: int = 3,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Fit the 3x2 empirical mixing matrix A from stellar photometry,
    with iterative sigma clipping to reject outliers.

    Returns
    -------
    (A, n_used) where:
        A      : ndarray shape (3, 2) or None if fitting fails
        n_used : int, number of stars surviving sigma clipping
    """
    if not star_records:
        return None, 0

    S1 = np.array([s["S_line1"] for s in star_records], dtype=np.float64)
    S2 = np.array([s["S_line2"] for s in star_records], dtype=np.float64)
    Rm = np.array([s["R_meas"]  for s in star_records], dtype=np.float64)
    Gm = np.array([s["G_meas"]  for s in star_records], dtype=np.float64)
    Bm = np.array([s["B_meas"]  for s in star_records], dtype=np.float64)

    # Normalise design matrix columns so NNLS isn't fighting scale differences.
    s1_scale = float(np.median(S1)) if np.any(S1 > 0) else 1.0
    s2_scale = float(np.median(S2)) if np.any(S2 > 0) else 1.0
    if s1_scale <= 0: s1_scale = 1.0
    if s2_scale <= 0: s2_scale = 1.0

    S1_n = S1 / s1_scale
    S2_n = S2 / s2_scale

    r_scale = float(np.median(Rm)) if np.any(Rm > 0) else 1.0
    g_scale = float(np.median(Gm)) if np.any(Gm > 0) else 1.0
    b_scale = float(np.median(Bm)) if np.any(Bm > 0) else 1.0
    if r_scale <= 0: r_scale = 1.0
    if g_scale <= 0: g_scale = 1.0
    if b_scale <= 0: b_scale = 1.0

    Rm_n = Rm / r_scale
    Gm_n = Gm / g_scale
    Bm_n = Bm / b_scale

    X_full = np.column_stack([S1_n, S2_n])
    keep   = np.ones(len(star_records), dtype=bool)
    A_n    = np.zeros((3, 2), dtype=np.float64)

    for iteration in range(max_iter):
        X        = X_full[keep]
        channels = [Rm_n[keep], Gm_n[keep], Bm_n[keep]]

        A_iter = np.zeros((3, 2), dtype=np.float64)
        ok = True
        for ch_idx, y in enumerate(channels):
            try:
                coeffs, _ = nnls(X, y)
                A_iter[ch_idx] = coeffs
            except Exception as e:
                print(f"[NBExtract] NNLS failed for channel {ch_idx} iter {iteration}: {e}")
                ok = False
                break
        if not ok:
            break

        A_n = A_iter

        predicted = X_full @ A_n.T
        measured  = np.column_stack([Rm_n, Gm_n, Bm_n])
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_res = np.where(predicted > 0,
                                np.abs(measured - predicted) / predicted,
                                0.0)
        star_res = np.mean(frac_res, axis=1)

        med = np.median(star_res[keep])
        mad = np.median(np.abs(star_res[keep] - med)) * 1.4826
        if mad <= 0:
            break

        new_keep  = keep & (star_res < med + sigma_clip * mad)
        n_clipped = int(np.sum(keep) - np.sum(new_keep))

        if n_clipped == 0:
            break

        print(f"[NBExtract] Sigma clip iter {iteration+1}: "
              f"clipped {n_clipped} stars, {int(np.sum(new_keep))} remaining")

        if np.sum(new_keep) < 6:
            print("[NBExtract] Too few stars after clipping; keeping previous mask.")
            break

        keep = new_keep

    n_used = int(np.sum(keep))

    flux_scales = [r_scale, g_scale, b_scale]
    A = np.zeros((3, 2), dtype=np.float64)
    for ch in range(3):
        A[ch, 0] = A_n[ch, 0] * flux_scales[ch] / s1_scale
        A[ch, 1] = A_n[ch, 1] * flux_scales[ch] / s2_scale

    if np.any(~np.isfinite(A)):
        print("[NBExtract] Mixing matrix contains non-finite values.")
        return None, 0

    try:
        k = float(np.linalg.cond(A))
        if k > 1e8:
            print(
                f"[NBExtract] Mixing matrix is numerically singular (cond={k:.2e}).\n"
                f"  Likely cause: G and B channels are identical (e.g. a synthetic HOO image\n"
                f"  where G=B=OIII). NBExtract requires a real dual-band OSC image where\n"
                f"  the green and blue Bayer pixels have genuinely different responses."
            )
            return None, 0
    except Exception:
        pass

    return A, n_used


def extract_channels_nnls(
    img_rgb: np.ndarray,
    A: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the per-pixel 3x2 system for line1 and line2 channel images.

    Uses the Moore-Penrose pseudo-inverse of A followed by non-negativity
    clipping.  Returns raw linear pseudo-flux arrays — no normalisation or
    scaling is applied here.  The caller is responsible for any stretch.
    """
    H, W = img_rgb.shape[:2]
    pixels = img_rgb.reshape(-1, 3).astype(np.float64)

    A_pinv = np.linalg.pinv(A)
    out = (A_pinv @ pixels.T).T
    out = np.clip(out, 0.0, None)

    line1 = out[:, 0].reshape(H, W).astype(np.float32)
    line2 = out[:, 1].reshape(H, W).astype(np.float32)

    return line1, line2


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
        Pickles FITS loading machinery
        WCS initialisation

    Adds:
        Filter type + bandwidth UI
        Step 2: empirical mixing matrix calibration (auto-Q from condition number)
        Step 3: per-pixel NNLS channel extraction with Q-blended regularisation
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
        self._fallback_mode: bool = False

        self._inject_nb_ui()
        self._load_nb_settings()

    # ── UI injection ──────────────────────────────────────────────────────────

    def _inject_nb_ui(self):
        _sfcc_only = [
            "r_filter_combo", "g_filter_combo", "b_filter_combo",
            "sens_combo", "lp_filter_combo", "lp_filter_combo2",
            "run_spcc_btn", "run_grad_btn", "grad_method_combo",
            "neutralize_chk", "open_sasp_btn",
            "add_curve_btn", "remove_curve_btn", "star_combo",
        ]
        for attr in _sfcc_only:
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.hide()
                except Exception:
                    pass

        self._hide_sfcc_filter_labels()

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
        self.nb_extract_btn.setEnabled(False)
        self.nb_extract_btn.clicked.connect(self._extract_channels)
        row_act.addWidget(self.nb_extract_btn)

        self.nb_stretch_chk = QCheckBox("Match output median to source")
        self.nb_stretch_chk.setChecked(True)
        self.nb_stretch_chk.setToolTip(
            "Applies a statistical stretch to each extracted channel so its\n"
            "median matches the corresponding source channel median.\n"
            "Leave checked for a linear output that sits at the same\n"
            "intensity level as the input image."
        )
        row_act.addWidget(self.nb_stretch_chk)
        row_act.addStretch()
        grp_lay.addLayout(row_act)

        # Matrix readout
        self.nb_matrix_label = QLabel("Mixing matrix: (not yet calibrated)")
        self.nb_matrix_label.setWordWrap(True)
        grp_lay.addWidget(self.nb_matrix_label)

        # ── Advanced: per-channel Q blending ─────────────────────────────
        adv_hdr = QHBoxLayout()
        self.nb_adv_btn = QPushButton("Advanced ▸")
        self.nb_adv_btn.setFlat(True)
        self.nb_adv_btn.setStyleSheet("font-size:11px; text-align:left;")
        self.nb_adv_btn.clicked.connect(self._toggle_nb_advanced)

        self.nb_adv_summary = QLabel("Q auto")
        self.nb_adv_summary.setStyleSheet("font-size:10px; color: palette(placeholderText);")
        adv_hdr.addWidget(self.nb_adv_btn)
        adv_hdr.addStretch(1)
        adv_hdr.addWidget(self.nb_adv_summary)
        grp_lay.addLayout(adv_hdr)

        self.nb_adv_panel = QWidget()
        adv_l = QVBoxLayout(self.nb_adv_panel)
        adv_l.setContentsMargins(4, 0, 0, 0)
        adv_l.setSpacing(6)

        # Auto-Q explanation label
        self.nb_q_auto_label = QLabel(
            "Q values are set automatically from the matrix condition number.\n"
            "Lower Q = more conservative (less NNLS correction, closer to raw channel).\n"
            "Override below if needed."
        )
        self.nb_q_auto_label.setStyleSheet("font-size:10px; color: palette(placeholderText);")
        self.nb_q_auto_label.setWordWrap(True)
        adv_l.addWidget(self.nb_q_auto_label)

        # Q1 row (line1)
        q1_row = QHBoxLayout()
        self.nb_q1_label = QLabel("Line 1 (Ha) mixing strength Q:")
        self.nb_q1_label.setStyleSheet("font-size:11px;")
        self.nb_q1_spin = QDoubleSpinBox()
        self.nb_q1_spin.setRange(0.0, 1.0)
        self.nb_q1_spin.setDecimals(2)
        self.nb_q1_spin.setSingleStep(0.05)
        self.nb_q1_spin.setValue(1.0)
        self.nb_q1_spin.setToolTip(
            "Blend between raw R channel (0.0) and full NNLS extraction (1.0).\n"
            "Auto-set from condition number after calibration.\n"
            "If the Ha channel shows dark halos or over-subtraction, reduce toward 0.5-0.7."
        )
        q1_row.addWidget(self.nb_q1_label)
        q1_row.addWidget(self.nb_q1_spin)
        q1_row.addStretch(1)
        adv_l.addLayout(q1_row)

        # Q2 row (line2)
        q2_row = QHBoxLayout()
        self.nb_q2_label = QLabel("Line 2 (OIII) mixing strength Q:")
        self.nb_q2_label.setStyleSheet("font-size:11px;")
        self.nb_q2_spin = QDoubleSpinBox()
        self.nb_q2_spin.setRange(0.0, 1.0)
        self.nb_q2_spin.setDecimals(2)
        self.nb_q2_spin.setSingleStep(0.05)
        self.nb_q2_spin.setValue(1.0)
        self.nb_q2_spin.setToolTip(
            "Blend between raw G channel (0.0) and full NNLS extraction (1.0).\n"
            "Auto-set from condition number after calibration.\n"
            "If the OIII channel shows dark holes or over-subtraction, reduce toward 0.5-0.7."
        )
        q2_row.addWidget(self.nb_q2_label)
        q2_row.addWidget(self.nb_q2_spin)
        q2_row.addStretch(1)
        adv_l.addLayout(q2_row)

        self.nb_q1_spin.valueChanged.connect(self._update_nb_adv_summary)
        self.nb_q2_spin.valueChanged.connect(self._update_nb_adv_summary)

        self.nb_adv_panel.setVisible(False)
        grp_lay.addWidget(self.nb_adv_panel)

        # Insert group above the status label
        layout = self.layout()
        idx = self._find_widget_index_in_layout(layout, self.count_label)
        if idx >= 0:
            layout.insertWidget(idx, grp)
        else:
            layout.addWidget(grp)

        self._on_preset_changed()

    def _hide_sfcc_filter_labels(self):
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

        if hasattr(self, "nb_q1_label"):
            self.nb_q1_label.setText(f"Line 1 ({l1_key}) mixing strength Q:")
        if hasattr(self, "nb_q2_label"):
            self.nb_q2_label.setText(f"Line 2 ({l2_key}) mixing strength Q:")

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

        self._A_matrix = None
        self.nb_extract_btn.setEnabled(False)
        self.nb_matrix_label.setText("Mixing matrix: (not yet calibrated)")

    def _toggle_nb_advanced(self):
        show = not self.nb_adv_panel.isVisible()
        self.nb_adv_panel.setVisible(show)
        self.nb_adv_btn.setText("Advanced ▾" if show else "Advanced ▸")
        self.nb_adv_summary.setVisible(not show)

    def _update_nb_adv_summary(self):
        q1 = float(self.nb_q1_spin.value())
        q2 = float(self.nb_q2_spin.value())
        self.nb_adv_summary.setText(f"Q\u2081 {q1:.2f}   Q\u2082 {q2:.2f}")

    def _set_auto_q(self, A: np.ndarray):
        """
        Set Q spinners from the mixing matrix using per-channel SVD analysis.
        Q₁ (line1/Ha) depends on global conditioning.
        Q₂ (line2/OIII) is additionally penalised by G/B row degeneracy —
        the most common cause of OIII over-subtraction.
        """
        k = float(np.linalg.cond(A))
        q1, q2 = auto_q_per_channel(A)

        self.nb_q1_spin.blockSignals(True)
        self.nb_q2_spin.blockSignals(True)
        self.nb_q1_spin.setValue(q1)
        self.nb_q2_spin.setValue(q2)
        self.nb_q1_spin.blockSignals(False)
        self.nb_q2_spin.blockSignals(False)

        self._update_nb_adv_summary()

        if k < 5:
            quality = "well-conditioned — channels separate cleanly"
        elif k < 15:
            quality = "moderate — some channel overlap"
        elif k < 50:
            quality = "poorly conditioned — significant channel overlap"
        else:
            quality = "very poorly conditioned — channels barely separable"

        self.nb_q_auto_label.setText(
            f"Q auto-derived from matrix structure (condition number {k:.2f}, {quality}).\n"
            f"Q₁ = {q1:.2f}\n"
            f"Q₂ = {q2:.2f}— "
            f"penalised by G/B channel similarity in line2 column.\n"
            f"Override below if needed."
        )

    # ── fetch_stars override ──────────────────────────────────────────────────

    def fetch_stars(self):
        """
        Override to redraw histogram after parent runs, showing only
        calibration-eligible stars (not B-V inferred).
        """
        super().fetch_stars()

        if not getattr(self, "star_list", None):
            return

        templates_real = []
        for s in self.star_list:
            tmpl      = s.get("pickles_match")
            sp_source = s.get("sp_source", "")
            sp_clean  = s.get("sp_clean", "") or ""
            if tmpl is None:
                continue
            is_bv_inferred = (
                sp_source == "bv_inferred"
                or (
                    sp_source not in ("simbad", "gaia_xp")
                    and len(sp_clean.strip()) == 1
                )
            )
            if not is_bv_inferred:
                templates_real.append(tmpl)

        if not templates_real:
            return

        try:
            _force_mpl_no_tex()
            self.figure.clf()
            uniq, cnt = np.unique(templates_real, return_counts=True)
            ax = self.figure.add_subplot(111)
            ax.bar(uniq, cnt, edgecolor="black")
            ax.set_xlabel("Spectral Type")
            ax.set_ylabel("Count")
            ax.set_title(
                f"Spectral Distribution (calibration-eligible stars only, n={len(templates_real)})"
            )
            ax.tick_params(axis="x", rotation=90)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            self.canvas.setVisible(True)
            self.canvas.draw()

            types_str = ", ".join([str(u) for u in uniq])
            if getattr(self, "count_label", None) is not None:
                self.count_label.setText(
                    f"Found {len(self.star_list)} stars; "
                    f"{len(templates_real)} eligible for NBExtract calibration; "
                    f"templates: {types_str}"
                )
        except Exception as e:
            print(f"[NBExtract] Histogram redraw failed: {e}")

    # ── Settings ──────────────────────────────────────────────────────────────

    def _load_nb_settings(self):
        s = QSettings()
        preset = s.value(_SK_PRESET, "Ha / OIII")
        idx = self.nb_preset_combo.findText(preset)
        if idx >= 0:
            self.nb_preset_combo.setCurrentIndex(idx)
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

    # ── SED loading ───────────────────────────────────────────────────────────

    def _load_sed_nm(self, extname: str) -> Tuple[np.ndarray, np.ndarray]:
        for path in (self.user_custom_path, self.sasp_data_path):
            try:
                with fits.open(path, memmap=False) as hd:
                    if extname in hd:
                        d      = hd[extname].data
                        wl_raw = d["WAVELENGTH"].astype(np.float64)
                        fl     = d["FLUX"].astype(np.float64)
                        wl_ang = _ensure_angstrom(wl_raw)
                        wl_nm  = wl_ang / 10.0
                        return wl_nm, fl
            except Exception:
                continue
        raise KeyError(f"SED extension '{extname}' not found in FITS data files.")

    # ── Step 2: calibrate mixing matrix ──────────────────────────────────────

    def _calibrate_mixing_matrix(self):
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

        star_records = []
        n_no_template  = 0
        n_bv_skipped   = 0

        for star in self.star_list:
            tmpl = star.get("pickles_match")
            if tmpl is None:
                n_no_template += 1
                continue

            sp_source = star.get("sp_source", "")
            sp_clean  = star.get("sp_clean", "") or ""

            is_bv_inferred = (
                sp_source == "bv_inferred"
                or (
                    sp_source not in ("simbad", "gaia_xp")
                    and len(sp_clean.strip()) == 1
                )
            )
            if is_bv_inferred:
                n_bv_skipped += 1
                continue

            dx = sources["x"] - star["x"]
            dy = sources["y"] - star["y"]
            j  = int(np.argmin(dx * dx + dy * dy))
            if (dx[j] ** 2 + dy[j] ** 2) >= 9.0:
                continue

            x    = float(sources["x"][j])
            y    = float(sources["y"][j])
            a    = float(sources["a"][j])
            r    = float(np.clip(2.0 * a, 2.0, 10.0))

            phot = measure_star_rgb_raw_aperture(base, x, y, r)
            if phot is None:
                continue

            Rm = float(phot["R_raw"])
            Gm = float(phot["G_raw"])
            Bm = float(phot["B_raw"])

            if not (np.isfinite(Rm) and np.isfinite(Gm) and np.isfinite(Bm)):
                continue
            if Rm <= 0 or Gm <= 0 or Bm <= 0:
                continue

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
            f"({n_no_template} no Pickles template, {n_bv_skipped} B-V-only rejected).",
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

        A, n_clipped_used = fit_mixing_matrix(star_records)
        if A is None:
            reply = QMessageBox.question(
                self, "Matrix Fit Failed — Use Fallback?",
                "Could not fit the full NNLS mixing matrix.\n\n"
                "Common causes:\n"
                "  • Image is a synthetic palette (e.g. HOO where G=B=OIII exactly)\n"
                "  • Too few spectrally diverse calibration stars survived filtering.\n"
                "  • All remaining stars are the same spectral type.\n\n"
                "Fall back to color channel mixing extraction?\n"
                "This applies a star-flux color channel mixing correction, then\n"
                f"returns the corrected R mixed channel as {l1_key} and G mixed channel as {l2_key}.\n"
                "Less precise than NNLS but works on any image.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._run_wb_fallback(doc, img_float, l1_key, l2_key)
            return

        self._A_matrix        = A
        self._fallback_mode   = False
        self._nb_star_records = star_records
        self._save_nb_settings()

        # ── Auto-Q from condition number ──────────────────────────────────
        k = float(np.linalg.cond(A))
        self._set_auto_q(A)

        # ── Diagnostics ───────────────────────────────────────────────────
        self._plot_calibration_diagnostics(star_records, A, l1_key, l2_key)

        # ── Matrix readout ────────────────────────────────────────────────
        warn = condition_number_warning(A)
        def _pct(row, col):
            total = float(A[row, 0] + A[row, 1])
            if total <= 0:
                return 0.0
            return 100.0 * float(A[row, col]) / total

        lines = [
            f"Channel sensitivity  ({l1_key} %  |  {l2_key} %):",
            f"  R :  {_pct(0,0):.1f}%  {l1_key}   {_pct(0,1):.1f}%  {l2_key}",
            f"  G :  {_pct(1,0):.1f}%  {l1_key}   {_pct(1,1):.1f}%  {l2_key}",
            f"  B :  {_pct(2,0):.1f}%  {l1_key}   {_pct(2,1):.1f}%  {l2_key}",
            f"  Condition number : {k:.2f}",
            f"  Stars used : {n_clipped_used} of {n_used}  ({n_used - n_clipped_used} outliers clipped)",
        ]
        if warn:
            lines.append(f"  ⚠  {warn}")
        self.nb_matrix_label.setText("\n".join(lines))

        self.nb_extract_btn.setEnabled(True)

        summary = (
            f"Mixing matrix fitted from {n_clipped_used} stars "
            f"({n_used - n_clipped_used} clipped as outliers).\n"
            f"Condition number: {k:.2f}\n"
            f"Auto Q₁ = {float(self.nb_q1_spin.value()):.2f}  Q₂ = {float(self.nb_q2_spin.value()):.2f}  (derived from matrix structure)\n\n"
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
        _force_mpl_no_tex()
        self.figure.clf()

        S1   = np.array([s["S_line1"] for s in star_records])
        S2   = np.array([s["S_line2"] for s in star_records])
        X    = np.column_stack([S1, S2])
        pred = X @ A.T

        meas = np.column_stack([
            [s["R_meas"] for s in star_records],
            [s["G_meas"] for s in star_records],
            [s["B_meas"] for s in star_records],
        ])

        palette   = ("firebrick", "seagreen", "royalblue")
        ch_labels = ("R channel", "G channel", "B channel")

        for i, (color, label) in enumerate(zip(palette, ch_labels)):
            ax = self.figure.add_subplot(1, 3, i + 1)
            ax.scatter(pred[:, i], meas[:, i], c=color, alpha=0.7, s=18, edgecolors="none")
            x_lo, x_hi = float(pred[:, i].min()), float(pred[:, i].max())
            y_lo, y_hi = float(meas[:, i].min()), float(meas[:, i].max())
            ref_lo = min(x_lo, y_lo)
            ref_hi = max(x_hi, y_hi)
            ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], "k--", lw=0.8, alpha=0.5)
            pad_x = max((x_hi - x_lo) * 0.05, 1e-6)
            pad_y = max((y_hi - y_lo) * 0.05, 1e-6)
            ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
            ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
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

    # ── WB fallback ───────────────────────────────────────────────────────────

    def _run_wb_fallback(self, doc, img_float: np.ndarray, l1_key: str, l2_key: str):
        from setiastro.saspro.imageops.starbasedwhitebalance import (
            apply_star_based_white_balance,
        )

        _sfcc_status(self, "NBExtract fallback: applying stellar flux color mixing…")
        QApplication.processEvents()

        sep_sigma = float(self.sep_thr_spin.value()) if hasattr(self, "sep_thr_spin") else 5.0

        try:
            balanced, star_count, _overlay = apply_star_based_white_balance(
                img_float,
                threshold=sep_sigma,
                autostretch=False,
                reuse_cached_sources=False,
                return_star_colors=False,
                use_color_matrix=True,
            )
        except Exception as e:
            print(f"[NBExtract] WB fallback failed ({e}), using raw channel split.")
            balanced = img_float
            star_count = 0

        line1_img = balanced[..., 0].copy()
        line2_img = balanced[..., 1].copy()

        if self.nb_stretch_chk.isChecked():
            try:
                from setiastro.saspro.imageops.stretch import stretch_mono_image
                median_r = float(np.median(img_float[..., 0]))
                median_g = float(np.median(img_float[..., 1]))
                if median_r > 1e-6:
                    line1_img = stretch_mono_image(
                        line1_img, target_median=median_r,
                        normalize=False, no_black_clip=True,
                    )
                if median_g > 1e-6:
                    line2_img = stretch_mono_image(
                        line2_img, target_median=median_g,
                        normalize=False, no_black_clip=True,
                    )
            except Exception as e:
                print(f"[NBExtract] WB fallback median match failed: {e}")

        def _linear_scale_to_unity(ch: np.ndarray) -> np.ndarray:
            ch = np.maximum(np.asarray(ch, dtype=np.float32), 0.0)
            mx = float(ch.max())
            return ch / mx if mx > 1.0 else ch

        line1_img = _linear_scale_to_unity(line1_img)
        line2_img = _linear_scale_to_unity(line2_img)

        preset_name = self.nb_preset_combo.currentText()
        meta_base   = dict(doc.metadata or {})

        meta1 = {**meta_base, "NBExtract_line": l1_key, "NBExtract_preset": preset_name,
                 "NBExtract_center_nm": float(self.nb_center1_spin.value()),
                 "NBExtract_bw_nm": float(self.nb_bw1_spin.value()),
                 "NBExtract_method": "wb_fallback", "NBExtract_wb_stars": int(star_count)}
        meta2 = {**meta_base, "NBExtract_line": l2_key, "NBExtract_preset": preset_name,
                 "NBExtract_center_nm": float(self.nb_center2_spin.value()),
                 "NBExtract_bw_nm": float(self.nb_bw2_spin.value()),
                 "NBExtract_method": "wb_fallback", "NBExtract_wb_stars": int(star_count)}

        self._push_new_document(line1_img, meta1, f"{l1_key} (NBExtract)")
        self._push_new_document(line2_img, meta2, f"{l2_key} (NBExtract)")

        self.nb_matrix_label.setText(
            f"Method: Stellar flux color mixing fallback\n"
            f"  {l1_key} = color-corrected R channel\n"
            f"  {l2_key} = color-corrected G channel\n"
            f"  Stars used: {star_count}"
        )
        self.nb_extract_btn.setEnabled(False)

        _sfcc_status(self,
            f"NBExtract color mixing fallback complete — '{l1_key}' and '{l2_key}' "
            f"created from {star_count} stars.")

        QMessageBox.information(
            self, "Color Mixing Fallback Complete",
            f"Created two new documents using stellar flux color mixing:\n"
            f"  •  {l1_key} (NBExtract) — color-corrected R channel\n"
            f"  •  {l2_key} (NBExtract) — color-corrected G channel\n\n"
            f"3×3 color mixing matrix fitted from {star_count} stellar flux measurements.\n"
        )

    # ── Step 3: extract channels ──────────────────────────────────────────────

    def _extract_channels(self):
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
            QMessageBox.critical(self, "Error", "Active document must be RGB (3 channels).")
            return

        preset_name = self.nb_preset_combo.currentText()
        l1_key, l2_key = FILTER_PRESETS.get(preset_name, ("Ha", "OIII"))

        img_float = img.astype(np.float32) / (255.0 if img.dtype == np.uint8 else 1.0)

        _sfcc_status(self, "NBExtract: solving per-pixel mixing system…")
        QApplication.processEvents()

        line1_nnls, line2_nnls = extract_channels_nnls(img_float, self._A_matrix)

        raw1 = img_float[..., 0].copy()
        raw2 = img_float[..., 1].copy()

        def _linear_scale_to_unity(ch: np.ndarray) -> np.ndarray:
            ch = np.asarray(ch, dtype=np.float32)
            ch = np.maximum(ch, 0.0)
            mx = float(ch.max())
            return ch / mx if mx > 1.0 else ch

        # Bring NNLS outputs and raw priors to the same intensity level
        # BEFORE blending so Q operates in a consistent scale.
        if self.nb_stretch_chk.isChecked():
            try:
                from setiastro.saspro.imageops.stretch import stretch_mono_image

                median_r = float(np.median(img_float[..., 0]))
                median_g = float(np.median(img_float[..., 1]))

                if median_r > 1e-6:
                    line1_nnls = stretch_mono_image(
                        line1_nnls, target_median=median_r,
                        normalize=False, no_black_clip=True,
                    )
                    raw1 = stretch_mono_image(
                        raw1, target_median=median_r,
                        normalize=False, no_black_clip=True,
                    )
                if median_g > 1e-6:
                    line2_nnls = stretch_mono_image(
                        line2_nnls, target_median=median_g,
                        normalize=False, no_black_clip=True,
                    )
                    raw2 = stretch_mono_image(
                        raw2, target_median=median_g,
                        normalize=False, no_black_clip=True,
                    )
            except Exception as e:
                print(f"[NBExtract] Median match stretch failed (continuing): {e}")

        # Q was auto-set from condition number at calibration time;
        # user may have overridden in Advanced panel.
        q1 = float(self.nb_q1_spin.value()) if hasattr(self, "nb_q1_spin") else 1.0
        q2 = float(self.nb_q2_spin.value()) if hasattr(self, "nb_q2_spin") else 1.0

        line1_img = _linear_scale_to_unity(
            (q1 * line1_nnls + (1.0 - q1) * raw1).astype(np.float32)
        )
        line2_img = _linear_scale_to_unity(
            (q2 * line2_nnls + (1.0 - q2) * raw2).astype(np.float32)
        )

        meta_base = dict(doc.metadata or {})
        A_list    = self._A_matrix.tolist()
        n_cal     = len(self._nb_star_records)

        meta1 = {**meta_base, "NBExtract_line": l1_key, "NBExtract_preset": preset_name,
                 "NBExtract_center_nm": float(self.nb_center1_spin.value()),
                 "NBExtract_bw_nm": float(self.nb_bw1_spin.value()),
                 "NBExtract_n_cal_stars": n_cal, "NBExtract_A_matrix": A_list,
                 "NBExtract_Q1": q1}
        meta2 = {**meta_base, "NBExtract_line": l2_key, "NBExtract_preset": preset_name,
                 "NBExtract_center_nm": float(self.nb_center2_spin.value()),
                 "NBExtract_bw_nm": float(self.nb_bw2_spin.value()),
                 "NBExtract_n_cal_stars": n_cal, "NBExtract_A_matrix": A_list,
                 "NBExtract_Q2": q2}

        self._push_new_document(line1_img, meta1, f"{l1_key} (NBExtract)")
        self._push_new_document(line2_img, meta2, f"{l2_key} (NBExtract)")

        k = float(np.linalg.cond(self._A_matrix))
        _sfcc_status(self,
            f"NBExtract complete — '{l1_key}' and '{l2_key}' channels created "
            f"from {n_cal} stars  (Q₁={q1:.2f}  Q₂={q2:.2f}).")

        QMessageBox.information(
            self, "Extraction Complete",
            f"Two new documents created:\n"
            f"  •  {l1_key}  (NBExtract)\n"
            f"  •  {l2_key}  (NBExtract)\n\n"
            f"Calibrated from {n_cal} stars.\n"
            f"Condition number: {k:.2f}\n"
            f"Q₁ = {q1:.2f}   Q₂ = {q2:.2f}  (auto-derived from condition number)\n\n"
            f"Tip: if you see over-subtraction (dark halos/holes), open Advanced\n"
            f"and reduce Q. If under-subtraction (residual bleed), increase Q."
        )

    # ── Document / window helpers ─────────────────────────────────────────────

    def _main_window(self):
        p = self.parent()
        while p is not None:
            if hasattr(p, "mdi"):
                return p
            p = p.parent()
        from PyQt6.QtWidgets import QApplication
        for w in QApplication.topLevelWidgets():
            if hasattr(w, "mdi"):
                return w
        return None

    def _push_new_document(self, img_mono: np.ndarray, metadata: dict, title: str):
        dm = self.doc_manager
        mw = self._main_window()

        if dm is None or mw is None or not hasattr(mw, "_spawn_subwindow_for"):
            QMessageBox.critical(self, "NBExtract",
                f"Cannot create document '{title}':\nDocManager or MainWindow not available.")
            return

        meta = {**metadata, "display_name": title, "file_path": title,
                "bit_depth": "32-bit floating point", "is_mono": True}
        try:
            doc = dm.create_document(img_mono, metadata=meta, name=title)
            mw._spawn_subwindow_for(doc)
        except Exception as e:
            QMessageBox.critical(self, "NBExtract", f"Failed to create document '{title}':\n{e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def open_nbextract(doc_manager, sasp_data_path: str, parent=None) -> NBExtractDialog:
    """
    Instantiate and show the Narrowband Channel Extractor dialog.

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