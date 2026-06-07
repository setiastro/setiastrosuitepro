# ══════════════════════════════════════════════════════════════════════════════
# sssc.py  —  Spectrophotometric Standard Star Calibration
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SSSC: THE NEXT GENERATION OF SPCC                                      │
# │                                                                         │
# │  SFCC/SPCC-style calibration presumes it knows the sensor QE curve.     │
# │  Manufacturer QE curves are marketing material — measured at room temp   │
# │  on a bare die, ignorant of your AR coating, your electronics chain,     │
# │  your optics, your atmosphere, and your specific imaging train.          │
# │                                                                         │
# │  SSSC abandons the QE curve entirely. Instead, it does exactly what     │
# │  Gaia's own internal photometric calibration pipeline does:              │
# │                                                                         │
# │    measured_c(i) = k_c × ∫ flux_star_i(λ) × T_filter_c(λ) × R(λ) dλ  │
# │                                                                         │
# │  where R(λ) — the true effective system response — is SOLVED FROM THE   │
# │  DATA using Gaia XP spectra as calibrators. No QE presumption at all.   │
# │                                                                         │
# │  Filter transmission curves remain as inputs (Antlia, Chroma, Baader    │
# │  etc. publish reliable interferometric measurements). Everything else    │
# │  — sensor QE, optics throughput, atmosphere, AR coating, microlenses —  │
# │  is absorbed into R(λ) and solved empirically.                          │
# └─────────────────────────────────────────────────────────────────────────┘
#
# BOOTSTRAP STAGES (data-driven progression):
# ────────────────────────────────────────────
#  Stage 1  (<50 stars)      : scalar per-channel gains only (k_R, k_G, k_B)
#                              Equivalent to current SFCC scalar/quadratic model.
#
#  Stage 2  (50–200 stars)   : color-dependent gain within each band.
#                              First moment of R(λ) per filter resolved.
#                              Quadratic model per channel becomes well-posed.
#
#  Stage 3  (200–1000 stars) : full R(λ) as smooth polynomial/spline.
#                              Requires spectral diversity spanning B-V [-0.3, 1.8].
#                              Hot blue O/B stars probe blue end of each passband;
#                              cool M dwarfs probe the red end. Full population
#                              triangulates R(λ) shape within each filter.
#                              TARGET FOR GAIA DR4.
#
#  Stage 4  (1000+ stars,    : atmosphere decorrelation. With enough sessions at
#            multi-session)    different airmasses, R(λ) separates into:
#                                R(λ) = QE_true(λ) × optics(λ) × atmosphere(λ)
#                              The hardware component stabilizes; atmosphere varies.
#                              Builds a persistent calibrated system model over time.
#
# REFERENCES:
#   Riello et al. 2021  — "Gaia EDR3: Photometric content and validation"
#                          A&A 649, A3  (the pipeline we are replicating)
#   Carrasco et al. 2021 — "Gaia photometric science alerts programme"
#   Fabricius et al. 2021 — "Gaia EDR3: Catalogue validation"
#
# DR4 MIGRATION CHECKLIST (update sfcc.py too):
#   □ _query_gaia_sources_in_field(): gaiadr3 → gaiadr4, mag_limit 15.5 → 16.5
#   □ Verify has_xp_continuous column name unchanged in DR4 schema
#   □ Rebuild split library files with DR4 source list
#   □ _solve_system_response(): fill in the NNLS solver (stubbed below)
#   □ Enable Stage 3 path once field star counts confirm >200 XP sources
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import os
import re
import cv2
import math
import time
import hashlib
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

import numpy.ma as ma

# ── SciPy
from scipy.interpolate import interp1d
from scipy.optimize import nnls

# ── Astropy / Astroquery
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy.wcs.wcs import NoConvergence

# ── SEP
import sep

# ── Matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ── PyQt6
from PyQt6.QtCore import (Qt, QSettings, QStandardPaths, QThread,
                           pyqtSignal as _Signal)
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLabel, QMainWindow, QPushButton,
    QDialog, QFileDialog, QInputDialog, QMessageBox, QWidget,
    QProgressDialog,
)
from PyQt6.QtCore import QEventLoop

# ── SASpro internals
from setiastro.saspro.main_helpers import non_blocking_sleep
from setiastro.saspro.backgroundneutral import background_neutralize_rgb, auto_rect_50x50
from astroquery.gaia import Gaia
from setiastro.saspro.gaia_downloader import GaiaDownloader, HAS_GAIAXPY
from setiastro.saspro.gaia_downloader import GaiaSpectraDB
from setiastro.saspro.gaia_database import get_library

# ── Re-use shared utilities from sfcc.py
from setiastro.saspro.sfcc import (
    _force_mpl_no_tex,
    _ensure_angstrom,
    _debug_probe_channels,
    _pivot_scale_channel,
    pickles_match_for_simbad,
    measure_star_rgb_photometry,
    measure_star_rgb_raw_aperture,
    _sfcc_status,
    _sfcc_busy,
    _GaiaSpectraWorker,
    _infer_letter_from_bv,
)

import warnings
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message=r"unclosed <ssl\.SSLSocket.*"
)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Wavelength grid shared across all integrations (Angstrom)
_WL_MIN_ANG = 3000
_WL_MAX_ANG = 11000
_WL_GRID    = np.arange(_WL_MIN_ANG, _WL_MAX_ANG + 1, dtype=np.float64)

# Bootstrap stage thresholds
_STAGE1_MIN =   5   # scalar gains only
_STAGE2_MIN =  50   # color-dependent gain within band
_STAGE3_MIN = 200   # full R(λ) spline — DR4 target
_STAGE4_MIN = 500   # multi-session atmosphere decorrelation

# R(λ) polynomial degree for Stage 3 solver
_RESPONSE_POLY_DEGREE = 6

# Session cache table name in gaia_xp_cache.sqlite
_SESSION_TABLE = "sssc_system_response"

# QSettings keys
_SK_RFILTER  = "SSSC/RFilter"
_SK_GFILTER  = "SSSC/GFilter"
_SK_BFILTER  = "SSSC/BFilter"
_SK_LP1      = "SSSC/LPFilter"
_SK_LP2      = "SSSC/LPFilter2"
_SK_SENSOR   = "SSSC/WhiteReference"   # white reference SED only — no QE
_SK_SEP_THR  = "SSSC/SEPThreshold"
_SK_BN       = "SSSC/BackgroundNeutralization"


# ══════════════════════════════════════════════════════════════════════════════
# System response model
# ══════════════════════════════════════════════════════════════════════════════

class SystemResponse:
    """
    Represents the effective system throughput R(λ) solved from stellar data.

    R(λ) absorbs everything the manufacturer QE curve cannot tell you:
      - True sensor QE at operating temperature
      - AR coating and cover glass transmission
      - Telescope mirror/lens throughput
      - Field flattener / focal reducer transmission
      - Atmospheric extinction at the time of observation

    Attributes
    ----------
    wl_ang : np.ndarray
        Wavelength grid in Angstrom.
    response : np.ndarray
        Solved R(λ) values, normalized so max = 1.0.
    stage : int
        Bootstrap stage (1–4) indicating solution quality.
    n_stars : int
        Number of calibrator stars used.
    bv_range : tuple[float, float]
        (min, max) B-V coverage of calibrator population.
    session_id : str
        Hash identifying this imaging session's configuration.
    timestamp : str
        ISO timestamp of when this solution was computed.
    poly_coeffs : np.ndarray | None
        Polynomial coefficients if Stage 3+ solution, else None.
    gains : np.ndarray
        Per-channel scalar gains [k_R, k_G, k_B] (always solved).
    residual_rms : float
        RMS fractional residual of the fit across all calibrators.
    """

    def __init__(
        self,
        wl_ang: np.ndarray,
        response: np.ndarray,
        *,
        stage: int,
        n_stars: int,
        bv_range: tuple[float, float],
        session_id: str,
        gains: np.ndarray,
        residual_rms: float = 0.0,
        poly_coeffs: np.ndarray | None = None,
        timestamp: str | None = None,
    ):
        self.wl_ang       = np.asarray(wl_ang,    dtype=np.float64)
        self.response     = np.asarray(response,  dtype=np.float64)
        self.stage        = int(stage)
        self.n_stars      = int(n_stars)
        self.bv_range     = tuple(bv_range)
        self.session_id   = str(session_id)
        self.gains        = np.asarray(gains,     dtype=np.float64)
        self.residual_rms = float(residual_rms)
        self.poly_coeffs  = poly_coeffs
        self.timestamp    = timestamp or datetime.now().isoformat()

    def evaluate(self, wl_ang: np.ndarray) -> np.ndarray:
        """Interpolate R(λ) onto an arbitrary wavelength grid."""
        return np.interp(wl_ang, self.wl_ang, self.response,
                         left=0.0, right=0.0)

    def to_dict(self) -> dict:
        return {
            "wl_ang":       self.wl_ang.tolist(),
            "response":     self.response.tolist(),
            "stage":        self.stage,
            "n_stars":      self.n_stars,
            "bv_range":     list(self.bv_range),
            "session_id":   self.session_id,
            "gains":        self.gains.tolist(),
            "residual_rms": self.residual_rms,
            "poly_coeffs":  self.poly_coeffs.tolist() if self.poly_coeffs is not None else None,
            "timestamp":    self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemResponse":
        pc = d.get("poly_coeffs")
        return cls(
            wl_ang       = np.array(d["wl_ang"]),
            response     = np.array(d["response"]),
            stage        = d["stage"],
            n_stars      = d["n_stars"],
            bv_range     = tuple(d["bv_range"]),
            session_id   = d["session_id"],
            gains        = np.array(d["gains"]),
            residual_rms = d.get("residual_rms", 0.0),
            poly_coeffs  = np.array(pc) if pc is not None else None,
            timestamp    = d.get("timestamp"),
        )

    @property
    def stage_label(self) -> str:
        return {
            1: "Stage 1 — scalar gains",
            2: "Stage 2 — band-level response",
            3: "Stage 3 — full R(λ) curve",
            4: "Stage 4 — hardware+atmosphere separated",
        }.get(self.stage, f"Stage {self.stage}")


# ══════════════════════════════════════════════════════════════════════════════
# Session identity
# ══════════════════════════════════════════════════════════════════════════════

def make_session_id(
    r_filter: str,
    g_filter: str,
    b_filter: str,
    lp1: str,
    lp2: str,
) -> str:
    """
    Stable hash identifying an imaging session's filter configuration.

    The session ID is deliberately independent of the sensor — since the
    sensor response is what we are *solving for*, it cannot be part of the
    key. Two sessions with the same filter set on different sensors will
    build separate response models automatically because their photometric
    measurements will differ.

    Parameters
    ----------
    r_filter, g_filter, b_filter : str
        Filter curve EXTNAME values.
    lp1, lp2 : str
        LP/cut filter EXTNAME values (or "(None)").

    Returns
    -------
    str
        8-character hex hash.
    """
    key = json.dumps(
        [r_filter, g_filter, b_filter, lp1, lp2],
        sort_keys=True
    ).encode()
    return hashlib.sha256(key).hexdigest()[:8]


# ══════════════════════════════════════════════════════════════════════════════
# System response solver  (THE CORE NEW METHOD)
# ══════════════════════════════════════════════════════════════════════════════

def _solve_system_response(
    enriched: list[dict],
    wl_grid: np.ndarray,
    T_sys_R: np.ndarray,
    T_sys_G: np.ndarray,
    T_sys_B: np.ndarray,
    session_id: str,
    *,
    status_cb=None,
) -> SystemResponse:
    """
    Solve for the effective system throughput R(λ) from stellar photometry.

    This is the heart of SSSC. Given measured aperture fluxes and known
    Gaia XP spectra convolved with filter curves, we solve for R(λ) —
    the wavelength-dependent system response that maps expected flux to
    measured flux.

    The formulation follows Riello et al. 2021 (Gaia EDR3 photometric
    calibration). For each star i and channel c:

        measured_c(i) = k_c × ∫ flux_i(λ) × T_filter_c(λ) × R(λ) dλ

    We parameterize R(λ) as a degree-6 polynomial constrained to [0, 1]
    and solve via non-negative least squares across all stars and channels
    simultaneously.

    Bootstrap stages are selected automatically based on star count and
    spectral diversity. Stage 1 always runs; Stage 3 requires ≥ 200 stars
    spanning B-V ≥ 1.5 range.

    Parameters
    ----------
    enriched : list[dict]
        Output of the parallel photometry step. Each entry must contain:
          R_meas, G_meas, B_meas   — background-subtracted star fluxes
          S_star_R, S_star_G, S_star_B — ∫ flux×T_filter dλ  (no R(λ))
          used_gaia                — True if Gaia XP spectrum used
          gaia_B, gaia_V           — for B-V color (spectral diversity check)
    wl_grid : np.ndarray
        Wavelength grid in Angstrom (shared _WL_GRID).
    T_sys_R, T_sys_G, T_sys_B : np.ndarray
        Filter × LP throughput arrays on wl_grid. No QE term included.
    session_id : str
        From make_session_id().
    status_cb : callable | None
        Progress callback f(str).

    Returns
    -------
    SystemResponse
        Solved system response with stage, gains, R(λ), and diagnostics.

    Notes
    -----
    STUB — Stage 1 (scalar gains) is fully implemented.
    Stage 3 (full R(λ) polynomial) is stubbed pending DR4 data density.

    Stage 3 implementation sketch (fill in when DR4 available):
    ─────────────────────────────────────────────────────────────
    1. For each star i, we need the XP spectrum flux sampled on wl_grid.
       Currently enriched[] only stores the integrated S_star_R/G/B values.
       We need to store the full flux array in enriched[] for Stage 3.
       Add "xp_flux" : np.ndarray to the _measure_one() return dict.

    2. Parameterize R(λ) as a polynomial basis:
           R(λ) = Σ_k  c_k × B_k(λ)
       where B_k are Legendre polynomials on [-1, 1] mapped to wl_grid.
       Degree 6 gives 7 free parameters — well constrained by 200+ stars.

    3. For each star i and channel c, the model integral becomes:
           I_c(i) = ∫ flux_i(λ) × T_c(λ) × R(λ) dλ
                  = Σ_k  c_k × ∫ flux_i(λ) × T_c(λ) × B_k(λ) dλ
                  = Σ_k  c_k × A_ck(i)
       where A_ck(i) is precomputed for each (star, channel, basis_func).

    4. Build the full linear system:
           [A_R(0)  A_R(1) ... A_R(N-1)]   [c_0]   [measured_R / k_R]
           [A_G(0)  A_G(1) ... A_G(N-1)] × [c_1] = [measured_G / k_G]
           [A_B(0)  A_B(1) ... A_B(N-1)]   [...]   [measured_B / k_B]
       Dimensions: (3 × n_stars) rows, (n_poly_coeffs) columns.

    5. Solve with scipy.optimize.nnls (enforces R(λ) ≥ 0):
           c, residual = nnls(A_matrix, b_vector)

    6. Sigma clip: compute per-star residuals, remove 3σ outliers, re-solve.

    7. Normalize so max(R(λ)) = 1.0. Store poly_coeffs in SystemResponse.

    8. Iterate k_c gains: with R(λ) fixed, re-solve for k_R, k_G, k_B as
       simple linear scalars. Repeat steps 5-8 until convergence (<0.1%).
    """
    if status_cb is None:
        status_cb = lambda m: None

    eps = 1e-12

    # ── Build per-star measurement arrays ────────────────────────────────────
    Rm_arr = np.array([float(e["R_meas"]) for e in enriched], dtype=np.float64)
    Gm_arr = np.array([float(e["G_meas"]) for e in enriched], dtype=np.float64)
    Bm_arr = np.array([float(e["B_meas"]) for e in enriched], dtype=np.float64)
    Sr_arr = np.array([float(e["S_star_R"]) for e in enriched], dtype=np.float64)
    Sg_arr = np.array([float(e["S_star_G"]) for e in enriched], dtype=np.float64)
    Sb_arr = np.array([float(e["S_star_B"]) for e in enriched], dtype=np.float64)

    # Guard against zeros
    valid = (
        (Gm_arr > eps) & (Rm_arr > eps) & (Bm_arr > eps) &
        (Sg_arr > eps) & (Sr_arr > eps) & (Sb_arr > eps) &
        np.isfinite(Rm_arr) & np.isfinite(Gm_arr) & np.isfinite(Bm_arr) &
        np.isfinite(Sr_arr) & np.isfinite(Sg_arr) & np.isfinite(Sb_arr)
    )
    Rm_arr = Rm_arr[valid]
    Gm_arr = Gm_arr[valid]
    Bm_arr = Bm_arr[valid]
    Sr_arr = Sr_arr[valid]
    Sg_arr = Sg_arr[valid]
    Sb_arr = Sb_arr[valid]
    enriched_valid = [e for e, v in zip(enriched, valid) if v]
    n_stars = len(Rm_arr)

    if n_stars < _STAGE1_MIN:
        raise ValueError(
            f"Too few valid calibrator stars ({n_stars}). "
            f"Need at least {_STAGE1_MIN}."
        )

    # ── Determine bootstrap stage ─────────────────────────────────────────────
    bv_vals = []
    for e in enriched_valid:
        bv = None
        b = e.get("gaia_B")
        v = e.get("gaia_V")
        if b is not None and v is not None:
            try:
                bv = float(b) - float(v)
            except Exception:
                pass
        if bv is not None and np.isfinite(bv):
            bv_vals.append(bv)

    bv_range = (
        (float(np.min(bv_vals)), float(np.max(bv_vals)))
        if bv_vals else (0.0, 0.0)
    )
    bv_span = bv_range[1] - bv_range[0]

    if n_stars >= _STAGE3_MIN and bv_span >= 1.5:
        stage = 3
    elif n_stars >= _STAGE2_MIN:
        stage = 2
    else:
        stage = 1

    status_cb(
        f"[SSSC] {n_stars} calibrator stars · B-V span={bv_span:.2f} · "
        f"{['', 'Stage 1 (scalar)', 'Stage 2 (band-level)', 'Stage 3 (full R(λ))'][stage]}"
    )

    # ── Stage 1: solve scalar per-channel gains k_R, k_G, k_B ───────────────
    # In ratio space: measured_R/measured_G = (k_R/k_G) × (Sr/Sg)
    # So:  meas_RG / exp_RG  =  k_R / k_G  (a single scalar per ratio)
    #
    # We solve for k_R/k_G and k_B/k_G via weighted least squares,
    # then set k_G = 1.0 (G is the reference channel).

    meas_RG = Rm_arr / Gm_arr
    meas_BG = Bm_arr / Gm_arr
    exp_RG  = Sr_arr / Sg_arr
    exp_BG  = Sb_arr / Sg_arr

    # Simple robust median ratio (insensitive to outliers)
    ratio_RG = np.median(meas_RG / np.where(exp_RG > eps, exp_RG, eps))
    ratio_BG = np.median(meas_BG / np.where(exp_BG > eps, exp_BG, eps))

    # k_R/k_G = ratio_RG  →  k_G=1, k_R=ratio_RG, k_B=ratio_BG
    k_G = 1.0
    k_R = float(np.clip(ratio_RG, 0.1, 10.0))
    k_B = float(np.clip(ratio_BG, 0.1, 10.0))
    gains = np.array([k_R, k_G, k_B], dtype=np.float64)

    status_cb(f"[SSSC] Stage 1 gains: k_R={k_R:.4f}  k_G={k_G:.4f}  k_B={k_B:.4f}")

    # Residual RMS for Stage 1
    pred_RG  = k_R * exp_RG
    pred_BG  = k_B * exp_BG
    resid_RG = (meas_RG / np.where(pred_RG > eps, pred_RG, eps)) - 1.0
    resid_BG = (meas_BG / np.where(pred_BG > eps, pred_BG, eps)) - 1.0
    residual_rms = float(np.sqrt(np.mean(resid_RG**2 + resid_BG**2) / 2.0))

    # ── Stage 1 R(λ): flat response scaled by gains ──────────────────────────
    # A flat R(λ) = 1.0 everywhere is the Stage 1 assumption — we know nothing
    # about the shape yet, only the integrated ratio per channel pair.
    response = np.ones_like(_WL_GRID, dtype=np.float64)

    # ── Stage 2: color-dependent gain within each band ───────────────────────
    # Fit a quadratic model: exp_RG_corrected = a_R × meas_RG² + b_R × meas_RG + c_R
    # This captures the first-order non-linearity of R(λ) within each band.
    # Directly mirrors the SFCC quadratic model — but now framed as a step
    # toward the full R(λ) solution rather than an end in itself.
    if stage >= 2:
        status_cb("[SSSC] Stage 2: fitting color-dependent band response…")
        # (implementation matches SFCC run_spcc Stage D — reuse that logic here
        #  when wiring up the full pipeline; for now gains capture the bulk)
        pass  # TODO: wire in quadratic fit from sfcc.run_spcc Stage D

    # ── Stage 3: full R(λ) polynomial ────────────────────────────────────────
    # STUB — see docstring above for the full implementation sketch.
    # Requires "xp_flux" arrays stored in enriched[] (not yet added to
    # _measure_one() in sfcc.py). Enable when DR4 provides sufficient density.
    if stage >= 3:
        status_cb("[SSSC] Stage 3: solving full R(λ) — STUB (pending DR4)…")
        # TODO: implement per docstring above
        # When implemented, overwrite `response` with the polynomial solution
        # and set poly_coeffs.
        stage = 2  # fall back until implemented
        pass

    poly_coeffs = None  # set by Stage 3 when implemented

    return SystemResponse(
        wl_ang       = _WL_GRID.copy(),
        response     = response,
        stage        = stage,
        n_stars      = n_stars,
        bv_range     = bv_range,
        session_id   = session_id,
        gains        = gains,
        residual_rms = residual_rms,
        poly_coeffs  = poly_coeffs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Session response cache  (persistent across runs)
# ══════════════════════════════════════════════════════════════════════════════

class SessionResponseCache:
    """
    Persists solved SystemResponse objects in the Gaia XP sqlite database.

    Each session is identified by its filter configuration hash. Multiple
    solutions accumulate over time, enabling Stage 4 atmosphere decorrelation
    when enough multi-session data is available.

    Table schema (added to gaia_xp_cache.sqlite):
        sssc_system_response (
            session_id   TEXT NOT NULL,
            timestamp    TEXT NOT NULL,
            n_stars      INTEGER,
            stage        INTEGER,
            bv_min       REAL,
            bv_max       REAL,
            residual_rms REAL,
            payload      TEXT   -- JSON-serialized SystemResponse.to_dict()
        )
    """

    def __init__(self, db_path: str):
        import sqlite3
        self._path = db_path
        self._conn = sqlite3.connect(db_path)
        self._ensure_table()

    def _ensure_table(self):
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_SESSION_TABLE} (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                n_stars      INTEGER,
                stage        INTEGER,
                bv_min       REAL,
                bv_max       REAL,
                residual_rms REAL,
                payload      TEXT
            )
        """)
        self._conn.commit()

    def save(self, sr: SystemResponse):
        self._conn.execute(
            f"""INSERT INTO {_SESSION_TABLE}
                (session_id, timestamp, n_stars, stage, bv_min, bv_max,
                 residual_rms, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sr.session_id,
                sr.timestamp,
                sr.n_stars,
                sr.stage,
                float(sr.bv_range[0]),
                float(sr.bv_range[1]),
                sr.residual_rms,
                json.dumps(sr.to_dict()),
            )
        )
        self._conn.commit()

    def load_latest(self, session_id: str) -> SystemResponse | None:
        """Load the most recent solution for this session configuration."""
        cur = self._conn.execute(
            f"""SELECT payload FROM {_SESSION_TABLE}
                WHERE session_id = ?
                ORDER BY timestamp DESC LIMIT 1""",
            (session_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return SystemResponse.from_dict(json.loads(row[0]))

    def load_all(self, session_id: str) -> list[SystemResponse]:
        """Load all historical solutions for Stage 4 averaging."""
        cur = self._conn.execute(
            f"""SELECT payload FROM {_SESSION_TABLE}
                WHERE session_id = ?
                ORDER BY timestamp ASC""",
            (session_id,)
        )
        results = []
        for (payload,) in cur.fetchall():
            try:
                results.append(SystemResponse.from_dict(json.loads(payload)))
            except Exception:
                pass
        return results

    def session_count(self, session_id: str) -> int:
        cur = self._conn.execute(
            f"SELECT COUNT(*) FROM {_SESSION_TABLE} WHERE session_id = ?",
            (session_id,)
        )
        return int(cur.fetchone()[0])

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Apply correction
# ══════════════════════════════════════════════════════════════════════════════

def apply_sssc_correction(
    img_float: np.ndarray,
    sr: SystemResponse,
    enriched: list[dict],
    wl_grid: np.ndarray,
    T_sys_R: np.ndarray,
    T_sys_G: np.ndarray,
    T_sys_B: np.ndarray,
    *,
    status_cb=None,
) -> np.ndarray:
    """
    Apply the solved system response correction to the image.

    For Stage 1/2 this is equivalent to the SFCC per-channel correction
    using the solved gains k_R, k_G, k_B. For Stage 3+ it uses the full
    R(λ)-corrected integrals to compute per-pixel corrections.

    Parameters
    ----------
    img_float : np.ndarray
        Float32 RGB image in [0, 1].
    sr : SystemResponse
        Solved system response from _solve_system_response().
    enriched : list[dict]
        Per-star photometry (for residual-based correction in Stage 3).
    wl_grid : np.ndarray
        Wavelength grid in Angstrom.
    T_sys_R, T_sys_G, T_sys_B : np.ndarray
        Filter throughput arrays (no QE) on wl_grid.
    status_cb : callable | None

    Returns
    -------
    np.ndarray
        Corrected float32 RGB image in [0, 1].
    """
    if status_cb is None:
        status_cb = lambda m: None

    eps = 1e-8
    k_R, k_G, k_B = sr.gains

    if sr.stage <= 2:
        # ── Stage 1/2: scalar gain correction ────────────────────────────────
        # k_R = median(measured_RG / expected_RG)
        # So measured_RG = k_R × expected_RG
        # To correct:  multiply R channel by (1/k_R), leave G unchanged.
        # k_G is always 1.0 (G is the reference), so only R and B need scaling.
        status_cb(f"[SSSC] Applying Stage {sr.stage} gain correction…")

        calibrated = img_float.copy()
        R = calibrated[..., 0]
        G = calibrated[..., 1]
        B = calibrated[..., 2]

        scale_R = float(np.clip(1.0 / max(k_R, eps), 0.25, 4.0))
        scale_B = float(np.clip(1.0 / max(k_B, eps), 0.25, 4.0))

        calibrated[..., 0] = _pivot_scale_channel(R, scale_R, float(np.median(R)))
        calibrated[..., 2] = _pivot_scale_channel(B, scale_B, float(np.median(B)))
        return np.clip(calibrated, 0.0, 1.0).astype(np.float32)

    else:
        # ── Stage 3+: full R(λ)-corrected integrals ──────────────────────────
        # TODO: implement when Stage 3 solver is complete.
        # For now fall back to Stage 1/2 correction.
        status_cb("[SSSC] Stage 3 correction — STUB, falling back to Stage 2…")
        sr_fallback = SystemResponse(
            wl_ang=sr.wl_ang, response=sr.response,
            stage=2, n_stars=sr.n_stars, bv_range=sr.bv_range,
            session_id=sr.session_id, gains=sr.gains,
            residual_rms=sr.residual_rms,
        )
        return apply_sssc_correction(
            img_float, sr_fallback, enriched,
            wl_grid, T_sys_R, T_sys_G, T_sys_B,
            status_cb=status_cb,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def build_sssc_diagnostics_figure(
    figure: Figure,
    sr: SystemResponse,
    enriched: list[dict],
    T_sys_R: np.ndarray,
    T_sys_G: np.ndarray,
    T_sys_B: np.ndarray,
    wl_grid: np.ndarray,
    *,
    manufacturer_qe: np.ndarray | None = None,
    manufacturer_qe_label: str = "Manufacturer QE",
) -> Figure:
    """
    Build the SSSC diagnostic figure into an existing matplotlib Figure.

    Four panels:
      1. Solved R(λ) vs manufacturer QE (the "gotcha" plot — shows how
         wrong the datasheet was for your actual setup)
      2. Before/After residual scatter in (R/G, B/G) ratio space
      3. Bootstrap stage indicator and star count history
      4. B-V color distribution of calibrator population

    Parameters
    ----------
    figure : Figure
        Existing matplotlib Figure to draw into (cleared first).
    sr : SystemResponse
        Solved system response.
    enriched : list[dict]
        Per-star photometry results.
    T_sys_R, T_sys_G, T_sys_B : np.ndarray
        Filter throughput arrays on wl_grid.
    wl_grid : np.ndarray
        Wavelength grid in Angstrom.
    manufacturer_qe : np.ndarray | None
        If provided, overlay manufacturer QE curve for comparison.
        Must be on the same wl_grid.
    manufacturer_qe_label : str
        Legend label for manufacturer curve.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure passed in, now populated.
    """
    _force_mpl_no_tex()
    fig = figure
    fig.clf()

    eps = 1e-12

    # ── Panel 1: R(λ) vs manufacturer QE ────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    wl_nm = wl_grid / 10.0  # Å → nm for readability

    ax1.plot(wl_nm, sr.response, color="#44cc88", linewidth=2.0,
             label=f"Solved R(λ)  [{sr.stage_label}]")

    if manufacturer_qe is not None:
        qe_norm = manufacturer_qe / max(float(np.max(manufacturer_qe)), eps)
        ax1.plot(wl_nm, qe_norm, color="#cc4444", linewidth=1.5,
                 linestyle="--", label=manufacturer_qe_label, alpha=0.8)

    # Overlay filter bands
    for T, color, label in [
        (T_sys_R, "red",   "R filter"),
        (T_sys_G, "green", "G filter"),
        (T_sys_B, "blue",  "B filter"),
    ]:
        T_norm = T / max(float(np.max(T)), eps)
        ax1.fill_between(wl_nm, T_norm * 0.3, alpha=0.12, color=color)
        ax1.plot(wl_nm, T_norm * 0.3, color=color, linewidth=0.8,
                 linestyle=":", alpha=0.6, label=label)

    ax1.set_xlim(300, 1100)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Normalized throughput")
    ax1.set_title("Solved System Response R(λ)")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    if manufacturer_qe is not None:
        ax1.set_title("Solved R(λ)  vs  Manufacturer QE")

    # ── Panel 2: Residuals before/after ──────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)

    k_R, k_G, k_B = sr.gains

    meas_RG, exp_RG, meas_BG, exp_BG = [], [], [], []
    for e in enriched:
        Rm = float(e.get("R_meas", 0))
        Gm = float(e.get("G_meas", 0))
        Bm = float(e.get("B_meas", 0))
        Sr = float(e.get("S_star_R", 0))
        Sg = float(e.get("S_star_G", 0))
        Sb = float(e.get("S_star_B", 0))
        if Gm <= eps or Sg <= eps:
            continue
        meas_RG.append(Rm / Gm)
        exp_RG.append(Sr / Sg)
        meas_BG.append(Bm / Gm)
        exp_BG.append(Sb / Sg)

    if meas_RG:
        meas_RG = np.array(meas_RG)
        exp_RG  = np.array(exp_RG)
        meas_BG = np.array(meas_BG)
        exp_BG  = np.array(exp_BG)

        res_RG_before = (meas_RG / np.where(exp_RG > eps, exp_RG, eps)) - 1.0
        res_BG_before = (meas_BG / np.where(exp_BG > eps, exp_BG, eps)) - 1.0
        res_RG_after  = (meas_RG / (k_R / k_G) / np.where(exp_RG > eps, exp_RG, eps)) - 1.0
        res_BG_after  = (meas_BG / (k_B / k_G) / np.where(exp_BG > eps, exp_BG, eps)) - 1.0

        ax2.scatter(exp_RG, res_RG_before, c="firebrick",  s=8,
                    alpha=0.4, label="R/G before")
        ax2.scatter(exp_BG, res_BG_before, c="royalblue",  s=8,
                    alpha=0.4, label="B/G before")
        ax2.scatter(exp_RG, res_RG_after,  c="salmon",     s=8,
                    alpha=0.6, marker="^", label="R/G after")
        ax2.scatter(exp_BG, res_BG_after,  c="lightblue",  s=8,
                    alpha=0.6, marker="^", label="B/G after")
        ax2.axhline(0, color="0.5", ls="--", lw=1)
        ax2.set_xlabel("Expected ratio (band/G)")
        ax2.set_ylabel("Fractional residual")
        ax2.set_title("Residuals Before / After")
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)

    # ── Panel 3: Bootstrap stage indicator ───────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    stages     = [_STAGE1_MIN, _STAGE2_MIN, _STAGE3_MIN, _STAGE4_MIN]
    stage_lbls = ["Stage 1\nScalar", "Stage 2\nBand", "Stage 3\nR(λ)", "Stage 4\nHardware"]
    colors     = ["#888888", "#5588cc", "#44cc88", "#cc8844"]

    for i, (thresh, lbl, col) in enumerate(zip(stages, stage_lbls, colors)):
        reached = sr.n_stars >= thresh
        ax3.barh(i, sr.n_stars if reached else thresh,
                 color=col if reached else "#333333",
                 alpha=0.8 if reached else 0.3, height=0.6)
        ax3.axvline(thresh, color=col, lw=1.5, ls="--", alpha=0.7)
        ax3.text(thresh + 5, i, f"{thresh}", va="center", fontsize=8, color=col)

    ax3.axvline(sr.n_stars, color="white", lw=2.0, label=f"This run: {sr.n_stars} stars")
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(stage_lbls, fontsize=8)
    ax3.set_xlabel("Calibrator star count")
    ax3.set_title("Bootstrap Stage Progress")
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="x", alpha=0.3)

    # ── Panel 4: B-V distribution ─────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    bv_vals = []
    for e in enriched:
        b = e.get("gaia_B")
        v = e.get("gaia_V")
        if b is not None and v is not None:
            try:
                bv_vals.append(float(b) - float(v))
            except Exception:
                pass

    if bv_vals:
        bv_arr = np.array(bv_vals)
        bv_arr = bv_arr[np.isfinite(bv_arr)]
        ax4.hist(bv_arr, bins=30, color="#5588cc", edgecolor="black",
                 linewidth=0.5, alpha=0.8)
        ax4.axvline(float(sr.bv_range[0]), color="orange", ls="--", lw=1.5,
                    label=f"Range: {sr.bv_range[0]:.2f}–{sr.bv_range[1]:.2f}")
        ax4.axvline(float(sr.bv_range[1]), color="orange", ls="--", lw=1.5)

    ax4.set_xlabel("B−V color index")
    ax4.set_ylabel("Count")
    ax4.set_title("Calibrator Spectral Coverage")
    ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", alpha=0.3)

    # Stage 3 note
    if sr.stage < 3:
        needed = _STAGE3_MIN - sr.n_stars
        ax4.set_title(
            f"Spectral Coverage  (need {needed} more stars for Stage 3)"
            if needed > 0 else "Spectral Coverage  ✓ Stage 3 ready"
        )

    fig.suptitle(
        f"SSSC Calibration Report  ·  {sr.n_stars} stars  ·  "
        f"{sr.stage_label}  ·  RMS={sr.residual_rms:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SSSCDialog
# ══════════════════════════════════════════════════════════════════════════════

class SSSCDialog(QDialog):
    """
    Spectrophotometric Standard Star Calibration dialog.

    Derives the effective system throughput R(λ) from Gaia XP stellar
    spectra and filter transmission curves — no sensor QE curve required.

    Architecture overview:
      Step 1 — Fetch Stars & Spectra  (reuses SFCC fetch_stars logic)
      Step 2 — Run SSSC Calibration   (calls _solve_system_response)
        Step A: integrate XP spectra × T_filter (no QE term)
        Step B: parallel aperture photometry on spectrum-bearing stars
        Step C: solve R(λ) at appropriate bootstrap stage
        Step D: apply correction via apply_sssc_correction()
        Step E: diagnostics plot
        Step F: persist solution to session cache

    See module docstring for full architecture and DR4 migration plan.
    """

    def __init__(self, doc_manager, sasp_data_path: str, parent=None):
        super().__init__(parent)
        _force_mpl_no_tex()
        self.setWindowTitle("Spectrophotometric Standard Star Calibration (SSSC)")
        self.setWindowFlag(Qt.WindowType.Window, True)

        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)

        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        self.setMinimumSize(900, 650)

        self.doc_manager    = doc_manager
        self.sasp_data_path = sasp_data_path
        self.user_custom_path = self._ensure_user_custom_fits()
        self.main_win       = parent

        self.star_list:    list[dict] = []
        self._last_matched: list[dict] = []
        self._session_cache: SessionResponseCache | None = None
        self._last_sr:      SystemResponse | None = None
        self.current_image  = None
        self.current_header = None
        self.wcs            = None

        self._reload_hdu_lists()

        # ── Attrs that SFCC fetch_stars expects on self ───────────────────────
        # fetch_stars is delegated to SFCCDialog.fetch_stars via method binding.
        # All attrs it touches must be pre-initialised here.
        self.pickles_templates: list[str] = []
        for p in (self.user_custom_path, self.sasp_data_path):
            try:
                with fits.open(p, memmap=False) as hd:
                    for hdu in hd:
                        if (isinstance(hdu, fits.BinTableHDU)
                                and hdu.header.get("CTYPE", "").upper() == "SED"):
                            extname = hdu.header.get("EXTNAME", None)
                            if extname and extname not in self.pickles_templates:
                                self.pickles_templates.append(extname)
            except Exception:
                pass
        self.pickles_templates.sort()

        self.sasp_viewer_window = None   # SaspViewer window ref (unused in SSSC but
        self._gaia_dl           = None   # expected by SFCC helpers we delegate to)
        self.center_ra          = None
        self.center_dec         = None
        self.wcs_header         = None
        self.pixscale           = None
        self.orientation        = None

        self._build_ui()
        self.load_settings()
        self.finished.connect(lambda *_: self._cleanup())

    # ── File prep ─────────────────────────────────────────────────────────────

    def _ensure_user_custom_fits(self) -> str:
        app_data = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation)
        os.makedirs(app_data, exist_ok=True)
        path = os.path.join(app_data, "usercustomcurves.fits")
        if not os.path.exists(path):
            fits.HDUList([fits.PrimaryHDU()]).writeto(path)
        return path

    def _gaia_db_path(self) -> str:
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation)
        d = os.path.join(base, "gaia")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "gaia_xp_cache.sqlite")

    def _get_session_cache(self) -> SessionResponseCache:
        if self._session_cache is None:
            self._session_cache = SessionResponseCache(self._gaia_db_path())
        return self._session_cache

    def _reload_hdu_lists(self):
        self.sed_list    = []
        self.filter_list = []
        self.sensor_list = []

        with fits.open(self.sasp_data_path, mode="readonly", memmap=False) as base:
            for hdu in base:
                if not isinstance(hdu, fits.BinTableHDU):
                    continue
                c = hdu.header.get("CTYPE", "").upper()
                e = hdu.header.get("EXTNAME", "")
                if c == "SED":
                    self.sed_list.append(e)
                elif c == "FILTER":
                    self.filter_list.append(e)
                elif c == "SENSOR":
                    self.sensor_list.append(e)

        for path in (self.user_custom_path,):
            try:
                with fits.open(path, mode="readonly", memmap=False) as hdul:
                    for hdu in hdul:
                        if not isinstance(hdu, fits.BinTableHDU):
                            continue
                        c = hdu.header.get("CTYPE", "").upper()
                        e = hdu.header.get("EXTNAME", "")
                        if c == "FILTER":
                            self.filter_list.append(e)
                        elif c == "SENSOR":
                            self.sensor_list.append(e)
            except Exception:
                pass

        self.sed_list.sort()
        self.filter_list.sort()
        self.sensor_list.sort()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── Row 1: Step 1 button + white reference ────────────────────────────
        row1 = QHBoxLayout()
        self.fetch_btn = QPushButton("Step 1: Fetch Stars & Spectra from Current View")
        f = self.fetch_btn.font()
        f.setBold(True)
        self.fetch_btn.setFont(f)
        self.fetch_btn.clicked.connect(self.fetch_stars)
        row1.addWidget(self.fetch_btn)

        row1.addSpacing(20)
        row1.addWidget(QLabel("White Reference:"))
        self.star_combo = QComboBox()
        self.star_combo.addItem("G2V (Solar)", userData="G2V")
        self.star_combo.addItem("Vega (A0V)",  userData="A0V")
        for sed in self.sed_list:
            if sed.upper() in ("A0V", "G2V"):
                continue
            self.star_combo.addItem(sed, userData=sed)
        row1.addWidget(self.star_combo)
        row1.addStretch()
        layout.addLayout(row1)

        # ── Row 2: Filter selectors (NO sensor QE) ───────────────────────────
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("R Filter:"))
        self.r_filter_combo = QComboBox()
        self.r_filter_combo.addItem("(None)")
        self.r_filter_combo.addItems(self.filter_list)
        row2.addWidget(self.r_filter_combo)

        row2.addSpacing(16)
        row2.addWidget(QLabel("G Filter:"))
        self.g_filter_combo = QComboBox()
        self.g_filter_combo.addItem("(None)")
        self.g_filter_combo.addItems(self.filter_list)
        row2.addWidget(self.g_filter_combo)

        row2.addSpacing(16)
        row2.addWidget(QLabel("B Filter:"))
        self.b_filter_combo = QComboBox()
        self.b_filter_combo.addItem("(None)")
        self.b_filter_combo.addItems(self.filter_list)
        row2.addWidget(self.b_filter_combo)
        row2.addStretch()

        # Note: deliberately no Sensor (QE) selector — it is solved from data
        no_qe_lbl = QLabel("⚠ No QE curve — sensor response solved from stars")
        no_qe_lbl.setStyleSheet("color: #44cc88; font-style: italic;")
        row2.addWidget(no_qe_lbl)
        layout.addLayout(row2)

        # ── Row 3: LP/cut filters ─────────────────────────────────────────────
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("LP/Cut Filter 1:"))
        self.lp_filter_combo = QComboBox()
        self.lp_filter_combo.addItem("(None)")
        self.lp_filter_combo.addItems(self.filter_list)
        row3.addWidget(self.lp_filter_combo)

        row3.addSpacing(16)
        row3.addWidget(QLabel("LP/Cut Filter 2:"))
        self.lp_filter_combo2 = QComboBox()
        self.lp_filter_combo2.addItem("(None)")
        self.lp_filter_combo2.addItems(self.filter_list)
        row3.addWidget(self.lp_filter_combo2)
        row3.addStretch()
        layout.addLayout(row3)

        # ── Row 4: Step 2 button + controls ──────────────────────────────────
        row4 = QHBoxLayout()
        self.run_btn = QPushButton("Step 2: Run SSSC Calibration")
        f2 = self.run_btn.font()
        f2.setBold(True)
        self.run_btn.setFont(f2)
        self.run_btn.clicked.connect(self.run_sssc)
        row4.addWidget(self.run_btn)

        self.neutralize_chk = QCheckBox("Background Neutralization")
        self.neutralize_chk.setChecked(False)
        row4.addWidget(self.neutralize_chk)

        row4.addSpacing(16)
        row4.addWidget(QLabel("Star detect σ:"))
        self.sep_thr_spin = QSpinBox()
        self.sep_thr_spin.setRange(2, 100)
        self.sep_thr_spin.setValue(15)
        self.sep_thr_spin.valueChanged.connect(
            lambda v: QSettings().setValue(_SK_SEP_THR, int(v)))
        row4.addWidget(self.sep_thr_spin)

        row4.addStretch()
        self.session_info_lbl = QLabel("")
        self.session_info_lbl.setStyleSheet("color: #888888; font-size: 10px;")
        row4.addWidget(self.session_info_lbl)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        row4.addWidget(self.close_btn)
        layout.addLayout(row4)

        # ── Status label ──────────────────────────────────────────────────────
        self.count_label = QLabel("")
        layout.addWidget(self.count_label)

        # ── Matplotlib canvas ─────────────────────────────────────────────────
        self.figure = Figure(figsize=(14, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setVisible(False)
        layout.addWidget(self.canvas, stretch=1)

        self.reset_btn = QPushButton("Reset View / Close")
        self.reset_btn.clicked.connect(self.reject)
        layout.addWidget(self.reset_btn)

        # ── Persist filter combos ─────────────────────────────────────────────
        self.r_filter_combo.currentIndexChanged.connect(
            lambda _: QSettings().setValue(_SK_RFILTER, self.r_filter_combo.currentText()))
        self.g_filter_combo.currentIndexChanged.connect(
            lambda _: QSettings().setValue(_SK_GFILTER, self.g_filter_combo.currentText()))
        self.b_filter_combo.currentIndexChanged.connect(
            lambda _: QSettings().setValue(_SK_BFILTER, self.b_filter_combo.currentText()))
        self.lp_filter_combo.currentIndexChanged.connect(
            lambda _: QSettings().setValue(_SK_LP1, self.lp_filter_combo.currentText()))
        self.lp_filter_combo2.currentIndexChanged.connect(
            lambda _: QSettings().setValue(_SK_LP2, self.lp_filter_combo2.currentText()))
        self.star_combo.currentIndexChanged.connect(
            lambda _: QSettings().setValue(_SK_SENSOR, self.star_combo.currentText()))

    def load_settings(self):
        s = QSettings()

        def apply(cb, key):
            val = s.value(key, "")
            if val:
                idx = cb.findText(val)
                if idx != -1:
                    cb.setCurrentIndex(idx)

        apply(self.r_filter_combo,  _SK_RFILTER)
        apply(self.g_filter_combo,  _SK_GFILTER)
        apply(self.b_filter_combo,  _SK_BFILTER)
        apply(self.lp_filter_combo,  _SK_LP1)
        apply(self.lp_filter_combo2, _SK_LP2)
        apply(self.star_combo,       _SK_SENSOR)

        sep_thr = int(s.value(_SK_SEP_THR, 15))
        self.sep_thr_spin.setValue(sep_thr)

    # ── View plumbing ─────────────────────────────────────────────────────────

    def _get_active_image_and_header(self):
        doc = self.doc_manager.get_active_document()
        if doc is None:
            return None, None, None
        img  = doc.image
        meta = doc.metadata or {}
        hdr  = (meta.get("wcs_header") or
                meta.get("original_header") or
                meta.get("header"))
        return img, hdr, meta

    # ── Gaia helpers (delegated to sfcc infrastructure) ───────────────────────

    def _gaia_enabled(self) -> bool:
        return (Gaia is not None) and (GaiaDownloader is not None) and bool(HAS_GAIAXPY)

    def _use_gaia_fallback(self) -> bool:
        return self._gaia_enabled()

    def _get_gaia_downloader(self) -> GaiaDownloader:
        if not hasattr(self, "_gaia_dl") or self._gaia_dl is None:
            self._gaia_dl = GaiaDownloader(self._gaia_db_path())
        return self._gaia_dl

    def _gaia_integrals_for_source_ids(
        self,
        source_ids: list[int],
        wl_grid_ang: np.ndarray,
        T_sys_R: np.ndarray,
        T_sys_G: np.ndarray,
        T_sys_B: np.ndarray,
        *,
        batch_size: int = 25,
    ) -> dict[int, tuple[float, float, float]]:
        """
        Delegate to the SFCC implementation — identical logic, no QE term
        because T_sys_R/G/B here are filter-only (QE not included).

        See sfcc.SFCCDialog._gaia_integrals_for_source_ids for full docs.
        """
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        # Borrow the method by binding self — all required attrs are present
        return _SFCC._gaia_integrals_for_source_ids(
            self, source_ids,
            wl_grid_ang, T_sys_R, T_sys_G, T_sys_B,
            batch_size=batch_size,
        )

    def _download_gaia_spectra_with_progress(self, missing, dl, batch_size=25):
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        return _SFCC._download_gaia_spectra_with_progress(
            self, missing, dl, batch_size=batch_size)

    def initialize_wcs_from_header(self, header):
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        return _SFCC.initialize_wcs_from_header(self, header)

    def _make_working_base_for_sep(self, img_float):
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        return _SFCC._make_working_base_for_sep(self, img_float)

    def _neutralize_background(self, rgb_f, *, remove_pedestal=False):
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        return _SFCC._neutralize_background(self, rgb_f,
                                            remove_pedestal=remove_pedestal)

    # ── Step 1: Fetch Stars ───────────────────────────────────────────────────

    def fetch_stars(self):
        """
        Fetch stars, WCS-convert positions, match Gaia XP library, query SIMBAD.

        Delegates entirely to the SFCC fetch_stars implementation — the star
        catalog logic is identical. The only difference is what we do with
        those stars in Step 2.
        """
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        _SFCC.fetch_stars(self)

        # Update session info label after fetch
        self._update_session_info_label()

    def _update_session_info_label(self):
        """Show session ID and historical run count in UI."""
        try:
            sid = make_session_id(
                self.r_filter_combo.currentText(),
                self.g_filter_combo.currentText(),
                self.b_filter_combo.currentText(),
                self.lp_filter_combo.currentText(),
                self.lp_filter_combo2.currentText(),
            )
            cache = self._get_session_cache()
            n_sessions = cache.session_count(sid)
            prev = cache.load_latest(sid)
            if prev:
                self.session_info_lbl.setText(
                    f"Session {sid}  ·  {n_sessions} prior run(s)  ·  "
                    f"last: {prev.stage_label}  ·  {prev.n_stars} stars"
                )
            else:
                self.session_info_lbl.setText(
                    f"Session {sid}  ·  first run for this filter set"
                )
        except Exception:
            pass

    # ── Step 2: Run SSSC ──────────────────────────────────────────────────────

    def run_sssc(self):
        """
        Main calibration pipeline.

        Step A — Integrate XP spectra × T_filter (no QE anywhere)
        Step B — Filter to spectrum-bearing stars, parallel photometry
        Step C — Solve R(λ) at appropriate bootstrap stage
        Step D — Apply correction
        Step E — Diagnostics plot
        Step F — Persist solution to session cache
        """
        import concurrent.futures

        r_filt  = self.r_filter_combo.currentText()
        g_filt  = self.g_filter_combo.currentText()
        b_filt  = self.b_filter_combo.currentText()
        lp_filt  = self.lp_filter_combo.currentText()
        lp_filt2 = self.lp_filter_combo2.currentText()
        ref_sed  = self.star_combo.currentData()

        if r_filt == "(None)" and g_filt == "(None)" and b_filt == "(None)":
            QMessageBox.warning(self, "Error", "Select at least one R/G/B filter.")
            return
        if not ref_sed:
            QMessageBox.warning(self, "Error", "Select a white reference SED.")
            return

        doc = self.doc_manager.get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.critical(self, "Error", "No active document.")
            return

        img = doc.image
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.critical(self, "Error", "Active document must be RGB.")
            return

        if not getattr(self, "star_list", None):
            QMessageBox.warning(self, "Error",
                "Run Step 1: Fetch Stars before calibration.")
            return

        img_float = (img.astype(np.float32) / 255.0
                     if img.dtype == np.uint8
                     else img.astype(np.float32, copy=False))

        base = self._make_working_base_for_sep(img_float)
        if self.neutralize_chk.isChecked():
            base = self._neutralize_background(base, remove_pedestal=False)

        # ── SEP re-detection ─────────────────────────────────────────────────
        gray     = np.mean(base, axis=2).astype(np.float32)
        bkg      = sep.Background(gray)
        data_sub = gray - bkg.back()
        err      = float(bkg.globalrms)

        sep_sigma = float(self.sep_thr_spin.value())
        _sfcc_status(self, f"Re-detecting stars (SEP σ={sep_sigma:.1f})…")
        QApplication.processEvents()

        sources = sep.extract(data_sub, sep_sigma, err=err)
        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error", "No sources detected.")
            return

        r_fluxrad, _ = sep.flux_radius(
            gray, sources["x"], sources["y"],
            2.0 * sources["a"], 0.5,
            normflux=sources["flux"], subpix=5)
        mask    = (r_fluxrad > 0.2) & (r_fluxrad <= 10)
        sources = sources[mask]
        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error",
                "All detections rejected by radius filter.")
            return

        # ── Match star_list → SEP detections ─────────────────────────────────
        _sfcc_status(self, f"Matching {sources.size:,} SEP sources to star catalog…")
        QApplication.processEvents()

        raw_matches = []
        for i, star in enumerate(self.star_list):
            dx = sources["x"] - star["x"]
            dy = sources["y"] - star["y"]
            j  = int(np.argmin(dx * dx + dy * dy))
            if (dx[j]**2 + dy[j]**2) < (3.0**2):
                raw_matches.append({
                    "sim_index": i,
                    "template":  star.get("pickles_match") or star.get("sp_clean") or "",
                    "src_index": j,
                    "x":         float(sources["x"][j]),
                    "y":         float(sources["y"][j]),
                    "a":         float(sources["a"][j]),
                    "sep_flux":  float(sources["flux"][j]),
                })

        if not raw_matches:
            QMessageBox.warning(self, "No Matches",
                "No catalog stars matched to SEP detections.")
            return

        MAX_PHOT_STARS = 500
        if len(raw_matches) > MAX_PHOT_STARS:
            raw_matches.sort(key=lambda m: m["sep_flux"], reverse=True)
            raw_matches = raw_matches[:MAX_PHOT_STARS]

        # ── Load filter curves (NO QE) ───────────────────────────────────────
        wl_grid = _WL_GRID.copy()

        def load_curve(ext):
            for p in (self.user_custom_path, self.sasp_data_path):
                with fits.open(p, memmap=False) as hd:
                    if ext in hd:
                        d  = hd[ext].data
                        wl = _ensure_angstrom(d["WAVELENGTH"].astype(float))
                        tp = d["THROUGHPUT"].astype(float)
                        return wl, tp
            raise KeyError(f"Curve '{ext}' not found")

        def load_sed(ext):
            for p in (self.user_custom_path, self.sasp_data_path):
                with fits.open(p, memmap=False) as hd:
                    if ext in hd:
                        d  = hd[ext].data
                        wl = _ensure_angstrom(d["WAVELENGTH"].astype(float))
                        fl = d["FLUX"].astype(float)
                        return wl, fl
            raise KeyError(f"SED '{ext}' not found")

        interp_tp = lambda wl_o, tp_o: np.interp(
            wl_grid, wl_o, tp_o, left=0.0, right=0.0)

        # Filter-only throughput — deliberately omitting QE
        T_R = interp_tp(*load_curve(r_filt)) if r_filt != "(None)" else np.ones_like(wl_grid)
        T_G = interp_tp(*load_curve(g_filt)) if g_filt != "(None)" else np.ones_like(wl_grid)
        T_B = interp_tp(*load_curve(b_filt)) if b_filt != "(None)" else np.ones_like(wl_grid)
        LP1 = interp_tp(*load_curve(lp_filt))  if lp_filt  != "(None)" else np.ones_like(wl_grid)
        LP2 = interp_tp(*load_curve(lp_filt2)) if lp_filt2 != "(None)" else np.ones_like(wl_grid)
        LP  = LP1 * LP2

        # T_sys = filter × LP only — no QE multiplied in
        T_sys_R = T_R * LP
        T_sys_G = T_G * LP
        T_sys_B = T_B * LP

        # ── Step A: Gaia XP integrals (filter-only, no QE) ──────────────────
        _sfcc_status(self, "Computing Gaia XP integrals (filter curves only, no QE)…")
        QApplication.processEvents()

        gaia_integrals: dict[int, tuple[float, float, float]] = {}
        if self._use_gaia_fallback():
            try:
                gaia_ids = sorted(set(
                    int(self.star_list[int(m["sim_index"])]["gaia_source_id"])
                    for m in raw_matches
                    if 0 <= int(m["sim_index"]) < len(self.star_list)
                    and self.star_list[int(m["sim_index"])].get("gaia_source_id") is not None
                ))
                if gaia_ids:
                    gaia_integrals = self._gaia_integrals_for_source_ids(
                        gaia_ids,
                        wl_grid_ang=wl_grid.astype(np.float64),
                        T_sys_R=T_sys_R.astype(np.float64),
                        T_sys_G=T_sys_G.astype(np.float64),
                        T_sys_B=T_sys_B.astype(np.float64),
                        batch_size=25,
                    )
                    _sfcc_status(self,
                        f"Gaia XP: {len(gaia_integrals):,} integrals computed…")
                    QApplication.processEvents()
            except Exception as e:
                print(f"[SSSC] Gaia XP integrals failed: {e}")

        # Pickles fallback for unmatched stars
        types_needed:     set[str]  = set()
        simbad_to_pickles: dict[str, str] = {}
        if not hasattr(self, "pickles_templates"):
            self.pickles_templates = []

        for m in raw_matches:
            si  = int(m["sim_index"])
            sid = self.star_list[si].get("gaia_source_id") \
                  if 0 <= si < len(self.star_list) else None
            if sid is not None and int(sid) in gaia_integrals:
                continue
            sp = m["template"]
            if not sp:
                continue
            cands = pickles_match_for_simbad(sp, self.pickles_templates)
            if cands:
                simbad_to_pickles[sp] = cands[0]
                types_needed.add(cands[0])

        template_integrals: dict[str, tuple[float, float, float]] = {}
        for pname in types_needed:
            try:
                wl_s, fl_s = load_sed(pname)
                fs_i = np.interp(wl_grid, wl_s, fl_s, left=0.0, right=0.0)
                template_integrals[pname] = (
                    float(_trapz(fs_i * T_sys_R, x=wl_grid)),
                    float(_trapz(fs_i * T_sys_G, x=wl_grid)),
                    float(_trapz(fs_i * T_sys_B, x=wl_grid)),
                )
            except Exception as e:
                print(f"[SSSC] Pickles {pname} failed: {e}")

        # Filter to stars with spectra
        raw_matches_with_spectrum = [
            m for m in raw_matches
            if (
                self.star_list[int(m["sim_index"])].get("gaia_source_id") is not None
                and int(self.star_list[int(m["sim_index"])]["gaia_source_id"]) in gaia_integrals
            ) or simbad_to_pickles.get(m["template"]) is not None
        ]

        n_with_spectrum = len(raw_matches_with_spectrum)
        _sfcc_status(self,
            f"{len(gaia_integrals):,} Gaia XP + {len(template_integrals):,} Pickles  ·  "
            f"{n_with_spectrum:,} of {len(raw_matches):,} stars have a spectrum")
        QApplication.processEvents()

        if n_with_spectrum == 0:
            QMessageBox.warning(self, "No Spectra",
                "No stars with Gaia XP spectrum or Pickles template found.")
            return

        # ── Step B: Parallel aperture photometry ─────────────────────────────
        _sfcc_status(self, f"Measuring flux for {n_with_spectrum} stars (parallel)…")
        QApplication.processEvents()

        base_ro  = np.ascontiguousarray(base, dtype=np.float32)
        enriched = []
        eps      = 1e-12

        def _measure_one(m):
            si   = int(m["sim_index"])
            x, y = float(m["x"]), float(m["y"])
            a    = float(m.get("a", 1.5))
            r    = float(np.clip(2.5 * a, 2.0, 12.0))
            rin  = float(np.clip(3.0 * r, 6.0, 40.0))
            rout = float(np.clip(5.0 * r, rin + 2.0, 60.0))

            phot = measure_star_rgb_photometry(base_ro, x, y, r, rin, rout)
            if phot is None:
                return None

            Rm = float(phot["R"]["star_sum"])
            Gm = float(phot["G"]["star_sum"])
            Bm = float(phot["B"]["star_sum"])
            if not (np.isfinite(Rm) and np.isfinite(Gm) and np.isfinite(Bm)):
                return None
            if Rm <= 0 or Gm <= 0 or Bm <= 0:
                return None

            sid = self.star_list[si].get("gaia_source_id") \
                  if 0 <= si < len(self.star_list) else None
            integrals = gaia_integrals.get(int(sid)) if sid is not None else None
            if integrals is None:
                sp    = m["template"]
                pname = simbad_to_pickles.get(sp)
                integrals = template_integrals.get(pname) if pname else None
            if integrals is None:
                return None

            S_sr, S_sg, S_sb = integrals
            if not (np.isfinite(S_sr) and np.isfinite(S_sg) and np.isfinite(S_sb)):
                return None
            if S_sr <= 0 or S_sg <= 0 or S_sb <= 0:
                return None

            st = self.star_list[si]
            return {
                **m,
                "R_meas":    Rm,
                "G_meas":    Gm,
                "B_meas":    Bm,
                "S_star_R":  S_sr,
                "S_star_G":  S_sg,
                "S_star_B":  S_sb,
                "used_gaia": sid is not None and int(sid) in gaia_integrals,
                "gaia_B":    st.get("gaia_B"),
                "gaia_V":    st.get("gaia_V"),
                # xp_flux stored here in Stage 3 — add when DR4 enabled:
                # "xp_flux": <np.ndarray on wl_grid>,
            }

        n_workers  = min(8, max(1, os.cpu_count() or 4))
        done_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_measure_one, m): m
                       for m in raw_matches_with_spectrum}
            for fut in concurrent.futures.as_completed(futures):
                done_count += 1
                if done_count % 25 == 0 or done_count == n_with_spectrum:
                    _sfcc_status(self,
                        f"Photometry: {done_count}/{n_with_spectrum} stars  ·  "
                        f"{len(enriched)} valid…")
                    QApplication.processEvents()
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"[SSSC] Photometry worker: {e}")
                    continue
                if result is not None:
                    enriched.append(result)

        self._last_matched = enriched
        _sfcc_status(self,
            f"Photometry complete — {len(enriched)} valid stars")
        QApplication.processEvents()

        if len(enriched) < _STAGE1_MIN:
            QMessageBox.warning(self, "Too Few Stars",
                f"Only {len(enriched)} valid calibrator stars — need ≥ {_STAGE1_MIN}.")
            return

        # ── Step C: Solve R(λ) ───────────────────────────────────────────────
        _sfcc_status(self, "Solving system response R(λ)…")
        QApplication.processEvents()

        session_id = make_session_id(r_filt, g_filt, b_filt, lp_filt, lp_filt2)

        try:
            sr = _solve_system_response(
                enriched, wl_grid,
                T_sys_R, T_sys_G, T_sys_B,
                session_id,
                status_cb=lambda m: _sfcc_status(self, m),
            )
        except Exception as e:
            QMessageBox.critical(self, "Solver Error",
                f"Failed to solve system response:\n{e}")
            return

        self._last_sr = sr

        # ── Step D: Apply correction ─────────────────────────────────────────
        _sfcc_status(self, "Applying system response correction…")
        QApplication.processEvents()

        calibrated = apply_sssc_correction(
            img_float, sr, enriched,
            wl_grid, T_sys_R, T_sys_G, T_sys_B,
            status_cb=lambda m: _sfcc_status(self, m),
        )

        if self.neutralize_chk.isChecked():
            try:
                calibrated = self._neutralize_background(
                    calibrated, remove_pedestal=True)
            except Exception as e:
                print(f"[SSSC] Final BN failed: {e}")

        out_img = (
            (np.clip(calibrated, 0.0, 1.0) * 255.0).astype(np.uint8)
            if img.dtype == np.uint8
            else np.clip(calibrated, 0.0, 1.0).astype(np.float32)
        )

        # ── Step E: Diagnostics ──────────────────────────────────────────────
        _sfcc_status(self, "Building diagnostics…")
        QApplication.processEvents()

        try:
            build_sssc_diagnostics_figure(
                self.figure,
                sr, enriched,
                T_sys_R, T_sys_G, T_sys_B, wl_grid,
            )
            self.canvas.setVisible(True)
            _force_mpl_no_tex()
            self.canvas.draw()
        except Exception as e:
            print(f"[SSSC] Diagnostics failed: {e}")

        # ── Step F: Save to document + session cache ─────────────────────────
        new_meta = dict(doc.metadata or {})
        new_meta.update({
            "SSSC_applied":      True,
            "SSSC_timestamp":    sr.timestamp,
            "SSSC_stage":        sr.stage,
            "SSSC_stage_label":  sr.stage_label,
            "SSSC_n_stars":      sr.n_stars,
            "SSSC_bv_range":     list(sr.bv_range),
            "SSSC_gains":        sr.gains.tolist(),
            "SSSC_residual_rms": sr.residual_rms,
            "SSSC_session_id":   sr.session_id,
        })

        self.doc_manager.update_active_document(
            out_img, metadata=new_meta,
            step_name="SSSC Calibrated", doc=doc)

        try:
            cache = self._get_session_cache()
            cache.save(sr)
            n_sessions = cache.session_count(session_id)
        except Exception as e:
            print(f"[SSSC] Cache save failed: {e}")
            n_sessions = 1

        self._update_session_info_label()

        n_xp  = sum(1 for e in enriched if e.get("used_gaia"))
        n_pkl = len(enriched) - n_xp

        _sfcc_status(self,
            f"SSSC complete — {len(enriched)} stars  ·  "
            f"{n_xp} Gaia XP  ·  {n_pkl} Pickles  ·  "
            f"{sr.stage_label}  ·  RMS={sr.residual_rms:.4f}")
        QApplication.processEvents()

        QMessageBox.information(self, "SSSC Complete",
            f"Applied SSSC using {len(enriched)} calibrator stars\n\n"
            f"  Gaia XP spectra:   {n_xp}\n"
            f"  Pickles templates: {n_pkl}\n"
            f"  Bootstrap stage:   {sr.stage_label}\n"
            f"  B-V range:         {sr.bv_range[0]:.2f} – {sr.bv_range[1]:.2f}\n"
            f"  Residual RMS:      {sr.residual_rms:.4f}\n"
            f"  Session runs:      {n_sessions}\n"
            f"  Gains:  k_R={sr.gains[0]:.4f}  k_G={sr.gains[1]:.4f}  k_B={sr.gains[2]:.4f}\n\n"
            f"  No sensor QE curve was used.\n"
            f"  Filter curves only — sensor response solved from stars.\n\n"
            f"  {_STAGE3_MIN - sr.n_stars} more stars needed for Stage 3 (full R(λ))"
            if sr.stage < 3 else
            f"Applied SSSC using {len(enriched)} calibrator stars\n\n"
            f"  Bootstrap stage:   {sr.stage_label}  ✓\n"
            f"  Full R(λ) curve solved from stellar data."
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup(self):
        try:
            if self._session_cache is not None:
                self._session_cache.close()
                self._session_cache = None
        except Exception:
            pass

        try:
            if getattr(self, "_gaia_dl", None) is not None:
                try:
                    self._gaia_dl.close()
                except Exception:
                    pass
                self._gaia_dl = None
        except Exception:
            pass

        try:
            self.current_image  = None
            self.current_header = None
            self.star_list      = []
            self._last_matched  = []
            self._last_sr       = None
            if hasattr(self, "wcs"):
                self.wcs = None
        except Exception:
            pass

        try:
            if getattr(self, "figure", None) is not None:
                self.figure.clf()
            if getattr(self, "canvas", None) is not None:
                self.canvas.setVisible(False)
                self.canvas.draw_idle()
        except Exception:
            pass

    def closeEvent(self, event):
        self._cleanup()
        event.accept()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def open_sssc(doc_manager, sasp_data_path: str, parent=None) -> SSSCDialog:
    """Open the SSSC calibration dialog."""
    dlg = SSSCDialog(
        doc_manager=doc_manager,
        sasp_data_path=sasp_data_path,
        parent=parent,
    )
    dlg.show()
    return dlg