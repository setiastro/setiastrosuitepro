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
#  Stage 1  (<50 stars)      : Scalar per-channel gains only (k_R, k_G, k_B).
#                              Equivalent to SFCC/SPCC scalar model. Corrects
#                              the dominant white-balance error but cannot
#                              capture any wavelength-dependent variation.
#
#  Stage 2  (50–200 stars)   : Color-dependent gain within each band.
#                              Fits a quadratic model in ratio space (same as
#                              SFCC Step D) but using filter-only integrals —
#                              no QE presumption. Captures the first moment of
#                              R(λ) per filter. Dominant improvement over Stage 1.
#
#  Stage 3  (200+ stars,     : Full R(λ) shape solved per channel via coordinate
#            B-V span ≥ 1.5)   descent on piecewise-linear control points.
#                              Hot O/B stars constrain the blue edge of each
#                              passband; cool K/M stars constrain the red edge.
#                              The full B-V population triangulates R(λ) shape
#                              within each filter. Each run refines the solution;
#                              prior sessions seed the next via ctrl_points cache.
#                              Control point count (default 8) is user-adjustable:
#                                8  pts → 200+ stars  (default, conservative)
#                               12  pts → 600+ stars  (finer resolution)
#                               16  pts → 1000+ stars (high resolution)
#                              Pixel correction still uses Stage 2 quadratic gains
#                              (well-conditioned); Stage 3 R(λ) drives diagnostics
#                              and seeds future sessions.
#
#  Stage 4  (future —        : Atmosphere/hardware decorrelation.
#            DR4 + multi-      Requires Gaia DR4 star density AND multi-session
#            night airmass)    data spanning a range of airmasses. The solver
#                              would decompose R(λ) as:
#                                R(λ) = R_hardware(λ) × R_atmosphere(λ, X)
#                              where X is airmass. R_hardware(λ) is stable
#                              night to night (sensor QE, mirror coatings,
#                              optics, AR coating). R_atmosphere(λ, X) varies
#                              with airmass and atmospheric conditions.
#                              By imaging the same filter+camera combination
#                              across many nights at different airmasses, the
#                              hardware component can be isolated and locked in
#                              as a persistent, empirically-measured system
#                              response curve — your actual QE curve, not the
#                              manufacturer's. Once stable, calibration collapses
#                              back to Stage 2 speed but with a physically correct
#                              R_hardware(λ) as input instead of a datasheet guess.
#                              The curve will also naturally track aging: mirror
#                              oxidation, sensor temperature variation, filter
#                              aging, or optical train changes all appear as
#                              drift in R_hardware(λ) over time.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE ENDGAME
# ─────────────────────────────────────────────────────────────────────────────
#  Once Stage 4 has converged on a stable R_hardware(λ) for your rig:
#
#    measured_c(i) = k_c × ∫ flux_i(λ) × T_filter(λ) × R_hardware(λ) dλ
#
#  R_hardware(λ) is now a KNOWN INPUT, not something being solved for.
#  Every calibration run becomes Stage 2 speed with Stage 3+ accuracy —
#  SFCC/SPCC but with an empirically-measured, rig-specific QE curve that
#  no manufacturer datasheet could ever give you.
#
#  The bootstrap sequence is a one-time investment per imaging rig. After
#  that, the system self-maintains: each new session either confirms the
#  hardware curve is stable or detects drift and flags an update.
#
# ─────────────────────────────────────────────────────────────────────────────
# REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
#  Riello et al. 2021     — "Gaia EDR3: Photometric content and validation"
#                            A&A 649, A3  (the calibration pipeline we replicate)
#  Carrasco et al. 2021   — "Gaia photometric science alerts programme"
#  Fabricius et al. 2021  — "Gaia EDR3: Catalogue validation"
#
# ─────────────────────────────────────────────────────────────────────────────
# DR4 MIGRATION CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────
#  When Gaia DR4 releases, update the following (sfcc.py too):
#   □ _query_gaia_sources_in_field(): gaiadr3 → gaiadr4, mag_limit 15.5 → 16.5
#   □ Verify has_xp_continuous column name unchanged in DR4 schema
#   □ Rebuild split library files with DR4 source list
#   □ Lower _STAGE3_MIN threshold — DR4 density means more fields qualify
#   □ Enable Stage 4 path: implement airmass decorrelation solver
#   □ Add AIRMASS / ALTITUDE FITS header ingestion to session metadata
#   □ Add R_hardware(λ) persistence and drift-detection to SessionResponseCache
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

# Patch np.trapz for any dependency code that calls it directly on NumPy 2.x
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

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
    QDoubleSpinBox, QCheckBox, QLabel, QLineEdit, QMainWindow, QPushButton,
    QDialog, QFileDialog, QInputDialog, QMessageBox, QWidget,
    QProgressDialog, QGroupBox,
)
from PyQt6.QtCore import QEventLoop

# ── SASpro internals
from setiastro.saspro.main_helpers import non_blocking_sleep
from setiastro.saspro.backgroundneutral import background_neutralize_rgb, auto_rect_50x50
# astroquery.gaia is imported lazily — see _get_sssc_gaia_tap() below.
# Importing it at module level triggers a network connection to the ESA
# Gaia TAP service, which hangs the splash screen during DR4 maintenance.
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
_SK_N_CTRL   = "SSSC/NCtrlPoints"
_SK_BN       = "SSSC/BackgroundNeutralization"

def _get_sssc_gaia_tap():
    """
    Lazily import and return the astroquery Gaia TAP object.
    Deferred so the archive connection is not attempted at module load time.
    During DR4 migration periods the Gaia archive can be slow or unresponsive;
    importing astroquery.gaia at module level would cause the splash screen
    to hang indefinitely.
    """
    try:
        from astroquery.gaia import Gaia
        return Gaia
    except ImportError:
        return None

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
    ctrl_points : dict | None
        Per-channel control point values from Stage 3 solver.
        Keys: "R", "G", "B" — each a dict with "wl" and "vals" arrays.
        Used to seed the next session's x0 directly in control-point space,
        avoiding the stitched-response misinterpretation bug.
    stage_rms : dict
        Per-stage residual RMS waterfall — {1: rms1, 2: rms2, 3: rms3} for
        each stage that ran.  Used by the diagnostics panel.  Older cached
        solutions without this field default to {solved_stage: residual_rms}.
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
        ctrl_points: dict | None = None,
        stage_rms: dict | None = None,
        channel_response: dict | None = None,
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
        self.ctrl_points     = ctrl_points
        self.stage_rms       = stage_rms if stage_rms is not None else {stage: residual_rms}
        # Per-channel normalized R(lambda) on _WL_GRID — populated by Stage 3 solver.
        # Keys: "R", "G", "B". Each array normalized to its own peak=1.
        # None for Stage 1/2 solutions.
        self.channel_response: dict | None = channel_response

    def evaluate(self, wl_ang: np.ndarray) -> np.ndarray:
        """Interpolate R(λ) onto an arbitrary wavelength grid."""
        return np.interp(wl_ang, self.wl_ang, self.response,
                         left=0.0, right=0.0)

    def to_dict(self) -> dict:
        d = {
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
            "stage_rms":    {str(k): v for k, v in self.stage_rms.items()},
        }
        if self.ctrl_points is not None:
            d["ctrl_points"] = {
                ch: {"wl": v["wl"].tolist(), "vals": v["vals"].tolist()}
                for ch, v in self.ctrl_points.items()
            }
        if self.channel_response is not None:
            d["channel_response"] = {
                ch: v.tolist() for ch, v in self.channel_response.items()
            }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SystemResponse":
        pc = d.get("poly_coeffs")
        cp_raw = d.get("ctrl_points")
        ctrl_points = None
        if cp_raw is not None:
            try:
                ctrl_points = {
                    ch: {
                        "wl":   np.array(v["wl"]),
                        "vals": np.array(v["vals"]),
                    }
                    for ch, v in cp_raw.items()
                }
            except Exception:
                ctrl_points = None
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
            ctrl_points  = ctrl_points,
            stage_rms    = {int(k): float(v) for k, v in d["stage_rms"].items()}
                           if "stage_rms" in d else None,
            channel_response = {ch: np.array(v) for ch, v in d["channel_response"].items()}
                               if "channel_response" in d else None,
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
    camera_label: str = "",
) -> str:
    """
    Stable hash identifying an imaging session's filter + camera configuration.

    The camera_label is a user-supplied free-text string (e.g. "IMX492 EdgeHD
    f/7") that distinguishes different sensors or optical trains using the same
    filter set. Leaving it blank gives the old filter-only behaviour so existing
    sessions are not broken.

    The session ID is deliberately independent of any QE curve name — since the
    sensor response is what we are *solving for*, it cannot be part of the key.

    Parameters
    ----------
    r_filter, g_filter, b_filter : str
        Filter curve EXTNAME values.
    lp1, lp2 : str
        LP/cut filter EXTNAME values (or "(None)").
    camera_label : str
        Optional free-text camera / rig identifier.

    Returns
    -------
    str
        8-character hex hash.
    """
    key = json.dumps(
        [r_filter, g_filter, b_filter, lp1, lp2, camera_label.strip()],
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
    prior_response: "SystemResponse | None" = None,
    n_ctrl: int = 8,
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

    Bootstrap stages are selected automatically based on star count and
    spectral diversity. Stage 1 always runs; Stage 3 requires ≥ 200 stars
    spanning B-V ≥ 1.5 range.

    Stage 3 prior seeding
    ─────────────────────
    When a prior Stage 3 solution exists, we seed x0 from the prior session's
    ctrl_points (per-channel control point values stored in the same coordinate
    space as x_cur).  This is safe because ctrl_points["R"]["vals"] etc. are
    in units of W-matrix scale — exactly what the coordinate descent operates
    on.  We do NOT use the stitched sr.response array for seeding because that
    is a filter-transmission-weighted composite across all channels and maps
    very poorly back to individual channel control points.
    """
    if status_cb is None:
        status_cb = lambda m: None

    eps = 1e-30   # guard only against literal zeros/underflow

    # ── Build per-star measurement arrays ────────────────────────────────────
    Rm_arr = np.array([float(e["R_meas"]) for e in enriched], dtype=np.float64)
    Gm_arr = np.array([float(e["G_meas"]) for e in enriched], dtype=np.float64)
    Bm_arr = np.array([float(e["B_meas"]) for e in enriched], dtype=np.float64)
    Sr_arr = np.array([float(e["S_star_R"]) for e in enriched], dtype=np.float64)
    Sg_arr = np.array([float(e["S_star_G"]) for e in enriched], dtype=np.float64)
    Sb_arr = np.array([float(e["S_star_B"]) for e in enriched], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_RG = np.where(Sg_arr > eps, Sr_arr / Sg_arr, np.nan)
        ratio_BG = np.where(Sg_arr > eps, Sb_arr / Sg_arr, np.nan)
        meas_RG  = np.where(Gm_arr > eps, Rm_arr / Gm_arr, np.nan)
        meas_BG  = np.where(Gm_arr > eps, Bm_arr / Gm_arr, np.nan)

    valid = (
        np.isfinite(Rm_arr) & np.isfinite(Gm_arr) & np.isfinite(Bm_arr) &
        np.isfinite(Sr_arr) & np.isfinite(Sg_arr) & np.isfinite(Sb_arr) &
        (Rm_arr > 0) & (Gm_arr > 0) & (Bm_arr > 0) &
        (Sg_arr > eps) &
        np.isfinite(ratio_RG) & np.isfinite(ratio_BG) &
        np.isfinite(meas_RG)  & np.isfinite(meas_BG)  &
        (ratio_RG > 0) & (ratio_BG > 0) &
        (meas_RG  > 0) & (meas_BG  > 0)
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
    meas_RG = Rm_arr / Gm_arr
    meas_BG = Bm_arr / Gm_arr
    exp_RG  = Sr_arr / Sg_arr
    exp_BG  = Sb_arr / Sg_arr

    ratio_RG = np.median(meas_RG / np.where(exp_RG > eps, exp_RG, eps))
    ratio_BG = np.median(meas_BG / np.where(exp_BG > eps, exp_BG, eps))

    k_G = 1.0
    k_R = float(np.clip(ratio_RG, 0.1, 10.0))
    k_B = float(np.clip(ratio_BG, 0.1, 10.0))
    gains = np.array([k_R, k_G, k_B], dtype=np.float64)

    status_cb(f"[SSSC] Stage 1 gains: k_R={k_R:.4f}  k_G={k_G:.4f}  k_B={k_B:.4f}")

    pred_RG  = k_R * exp_RG
    pred_BG  = k_B * exp_BG
    resid_RG = (meas_RG / np.where(pred_RG > eps, pred_RG, eps)) - 1.0
    resid_BG = (meas_BG / np.where(pred_BG > eps, pred_BG, eps)) - 1.0
    # Stage 1 RMS is computed inside the Stage 2 block using the same sigma-clipped
    # population and _rms_frac formula so all stages are directly comparable.
    # Fallback for Stage 1 only (< _STAGE2_MIN stars):
    rms_stage1_raw = float(np.sqrt(np.mean(resid_RG**2 + resid_BG**2) / 2.0))
    residual_rms = rms_stage1_raw   # fallback; overwritten by Stage 2/3 if they run
    rms_stage1   = None   # will be overwritten with comparable value in Stage 2 block
    stage_rms    = {}

    response         = np.ones_like(_WL_GRID, dtype=np.float64)
    poly_coeffs      = None
    ctrl_points      = None   # populated by Stage 3
    ch_response_norm = None   # populated by Stage 3 — per-channel normalized R(λ)

    # ── Stage 2: color-dependent gain — quadratic model per channel ──────────
    if stage >= 2:
        status_cb("[SSSC] Stage 2: fitting color-dependent band response…")

        def _rms_frac(pred, exp_v):
            return float(np.sqrt(np.mean(((pred / np.where(exp_v > eps, exp_v, eps)) - 1.0) ** 2)))

        raw_resid = np.abs(resid_RG) + np.abs(resid_BG)
        med_r = float(np.median(raw_resid))
        mad_r = float(np.median(np.abs(raw_resid - med_r))) * 1.4826
        if mad_r > 0:
            keep2 = raw_resid < med_r + 3.0 * mad_r
        else:
            keep2 = np.ones(len(meas_RG), dtype=bool)

        mrg = meas_RG[keep2];  erg = exp_RG[keep2]
        mbg = meas_BG[keep2];  ebg = exp_BG[keep2]

        # Slope-only
        denR = float(np.sum(mrg ** 2))
        denB = float(np.sum(mbg ** 2))
        mR_s = float(np.sum(mrg * erg)) / denR if denR > 0 else 1.0
        mB_s = float(np.sum(mbg * ebg)) / denB if denB > 0 else 1.0
        rms_s = (_rms_frac(mR_s * mrg, erg) + _rms_frac(mB_s * mbg, ebg))

        # Stage 1 RMS using the SAME metric, population, and _rms_frac as Stage 2.
        # k_R/k_B are the median-ratio scalar gains from Stage 1. Using these
        # (not the lstsq slope mR_s) gives the true Stage 1 prediction quality.
        _r1_R = _rms_frac(k_R * mrg, erg)
        _r1_B = _rms_frac(k_B * mbg, ebg)
        rms_stage1 = float(np.sqrt((_r1_R**2 + _r1_B**2) / 2.0))
        stage_rms[1] = rms_stage1
        status_cb(f"[SSSC] Stage 1 RMS={rms_stage1:.4f}  (scalar k_R={k_R:.4f} k_B={k_B:.4f})")

        # Affine
        mR_a, bR_a = np.linalg.lstsq(
            np.vstack([mrg, np.ones_like(mrg)]).T, erg, rcond=None)[0]
        mB_a, bB_a = np.linalg.lstsq(
            np.vstack([mbg, np.ones_like(mbg)]).T, ebg, rcond=None)[0]
        rms_a = (_rms_frac(mR_a * mrg + bR_a, erg) +
                 _rms_frac(mB_a * mbg + bB_a, ebg))

        # Quadratic — fit on clipped population, then validate before accepting.
        # With wide B-V fields (e.g. B-V span > 3) contaminated photometry or
        # mis-matched stars produce extreme ratio values. np.polyfit extrapolates
        # wildly on those, yielding near-zero or negative predictions and
        # astronomically large RMS when evaluated on the full population.
        # Guard: only accept quadratic if all predictions on the clipped
        # population are positive and the coefficients are physically reasonable
        # (leading term small enough that it doesn't flip sign within the data range).
        if len(mrg) >= 6:
            try:
                aR_q, bR_q, cR_q = np.polyfit(mrg, erg, 2)
                aB_q, bB_q, cB_q = np.polyfit(mbg, ebg, 2)
                pred_q_R = aR_q * mrg**2 + bR_q * mrg + cR_q
                pred_q_B = aB_q * mbg**2 + bB_q * mbg + cB_q
                # Also evaluate on the FULL population — the quadratic must stay
                # positive there too or the final residual_rms will explode on
                # outliers (carbon stars etc. with extreme ratio values).
                pred_q_R_full = aR_q * meas_RG**2 + bR_q * meas_RG + cR_q
                pred_q_B_full = aB_q * meas_BG**2 + bB_q * meas_BG + cB_q
                quad_ok = (
                    np.all(pred_q_R > 0) and np.all(pred_q_B > 0)
                    and np.all(pred_q_R_full > 0) and np.all(pred_q_B_full > 0)
                    and np.isfinite(aR_q) and np.isfinite(aB_q)
                    and (abs(aR_q) < 1e6 and abs(aB_q) < 1e6)
                )
                if quad_ok:
                    rms_q = (_rms_frac(pred_q_R, erg) + _rms_frac(pred_q_B, ebg))
                    # Reject if the computed rms_q is not finite (overflow possible)
                    if not np.isfinite(rms_q):
                        rms_q = np.inf
                else:
                    rms_q = np.inf
            except Exception:
                aR_q = bR_q = aB_q = bB_q = 0.0
                cR_q = cB_q = 1.0
                rms_q = np.inf
        else:
            aR_q = bR_q = aB_q = bB_q = 0.0
            cR_q = cB_q = 1.0
            rms_q = np.inf

        idx2 = int(np.argmin([rms_s, rms_a, rms_q]))
        if idx2 == 0:
            coeff_R2 = (0.0, float(mR_s), 0.0)
            coeff_B2 = (0.0, float(mB_s), 0.0)
            model2   = "slope-only"
        elif idx2 == 1:
            coeff_R2 = (0.0, float(mR_a), float(bR_a))
            coeff_B2 = (0.0, float(mB_a), float(bB_a))
            model2   = "affine"
        else:
            coeff_R2 = (float(aR_q), float(bR_q), float(cR_q))
            coeff_B2 = (float(aB_q), float(bB_q), float(cB_q))
            model2   = "quadratic"

        def _poly2(c, x):
            return c[0] * x**2 + c[1] * x + c[2]

        # Compute final residual RMS on the full population (original formula).
        # If the model is degenerate (non-finite or astronomically large),
        # fall back to the Stage 1 RMS so the display is sensible.
        pred2_RG = _poly2(coeff_R2, meas_RG)
        pred2_BG = _poly2(coeff_B2, meas_BG)
        r2_RG = (meas_RG / np.where(pred2_RG > eps, pred2_RG, eps)) - 1.0
        r2_BG = (meas_BG / np.where(pred2_BG > eps, pred2_BG, eps)) - 1.0
        residual_rms_raw = float(np.sqrt(np.mean(r2_RG**2 + r2_BG**2) / 2.0))

        # Guard: if the quadratic blew up (carbon stars, extreme B-V, etc.),
        # the reported Stage 2 RMS would be meaningless. In that case fall back
        # to Stage 1 RMS — it's the honest "what we actually applied" baseline,
        # and Stage 3 will improve from there regardless.
        if not np.isfinite(residual_rms_raw) or residual_rms_raw > 1e6:
            residual_rms = rms_stage1
            model2 = model2 + " [degenerate→S1 fallback]"
        else:
            residual_rms = residual_rms_raw

        gains = np.array([
            k_R, k_G, k_B,
            coeff_R2[0], coeff_R2[1], coeff_R2[2],
            coeff_B2[0], coeff_B2[1], coeff_B2[2],
        ], dtype=np.float64)

        stage_rms[2] = residual_rms

        status_cb(
            f"[SSSC] Stage 2 model={model2}  "
            f"RMS={residual_rms:.4f}  "
            f"({int(np.sum(keep2))} stars after sigma clip)"
        )

    # ── Stage 3: full R(λ) via coordinate descent ─────────────────────────────
    if stage >= 3:
        xp_fluxes = [e.get("xp_flux") for e in enriched_valid]
        have_flux = [f is not None and len(f) == len(_WL_GRID) for f in xp_fluxes]

        xp_indices  = np.array([i for i, h in enumerate(have_flux) if h], dtype=np.intp)
        n_xp_stage3 = len(xp_indices)

        if n_xp_stage3 < _STAGE3_MIN:
            status_cb(
                f"[SSSC] Stage 3: only {n_xp_stage3} XP stars — "
                f"need {_STAGE3_MIN}, keeping Stage 2"
            )
            stage = 2

        else:
            T_G_f64 = T_sys_G.astype(np.float64)
            T_R_f64 = T_sys_R.astype(np.float64)
            T_B_f64 = T_sys_B.astype(np.float64)

            # Per-spectrum G-band normalisation
            fl_xp_norm = []
            g_integrals = []
            for flux_raw in [xp_fluxes[i] for i in xp_indices]:
                fl = np.asarray(flux_raw, dtype=np.float64)
                fl = np.where(fl > 0, fl, 0.0)
                g_int = float(_trapz(fl * T_G_f64, x=_WL_GRID))
                if g_int < eps:
                    g_int = float(np.max(fl)) if fl.max() > 0 else 1.0
                fl_xp_norm.append(fl / g_int)
                g_integrals.append(g_int)
            g_integrals = np.array(g_integrals, dtype=np.float64)

            Rm_xp_n = Rm_arr[xp_indices] / g_integrals
            Gm_xp_n = Gm_arr[xp_indices] / g_integrals
            Bm_xp_n = Bm_arr[xp_indices] / g_integrals
            n_xp_s3 = n_xp_stage3


            # N_CTRL: control points per channel. More = finer R(λ) resolution
            # but requires more stars to avoid fitting noise. Rule of thumb:
            #   8  pts -> 200+ stars (default)
            #   12 pts -> 600+ stars recommended
            #   16 pts -> 1000+ stars recommended
            N_CTRL = max(4, int(n_ctrl))

            channels_cfg = [
                ("R", T_R_f64, Rm_xp_n, k_R),
                ("G", T_G_f64, Gm_xp_n, 1.0),
                ("B", T_B_f64, Bm_xp_n, k_B),
            ]

            # Per-channel passband thresholds.
            # Green uses 0.707 (half-power) to cut the broad secondary red lobe
            # present in OSC Bayer green filters. R and B use 0.30 which is tight
            # enough to exclude cross-channel leakage while still covering the
            # full passband of steep astro filters.
            PASSBAND_THRESH_PER_CH = {"R": 0.30, "G": 0.707, "B": 0.30}

            ch_data = {}
            for ch_name, T_c, meas_n, k_c_init in channels_cfg:
                T_peak = float(np.max(T_c))
                if T_peak < eps:
                    ch_data[ch_name] = None
                    continue
                pb = T_c >= PASSBAND_THRESH_PER_CH[ch_name] * T_peak
                pb_wl = _WL_GRID[pb]
                T_pb  = T_c[pb]

                ctrl_wl = np.linspace(pb_wl[0], pb_wl[-1], N_CTRL)

                W = np.zeros((n_xp_s3, N_CTRL), dtype=np.float64)
                for j in range(N_CTRL):
                    hat = np.zeros_like(pb_wl)
                    if j == 0:
                        mask_j = pb_wl <= ctrl_wl[1]
                        hat[mask_j] = (ctrl_wl[1] - pb_wl[mask_j]) / (ctrl_wl[1] - ctrl_wl[0])
                    elif j == N_CTRL - 1:
                        mask_j = pb_wl >= ctrl_wl[-2]
                        hat[mask_j] = (pb_wl[mask_j] - ctrl_wl[-2]) / (ctrl_wl[-1] - ctrl_wl[-2])
                    else:
                        l_mask = (pb_wl >= ctrl_wl[j-1]) & (pb_wl <= ctrl_wl[j])
                        r_mask = (pb_wl >= ctrl_wl[j])   & (pb_wl <= ctrl_wl[j+1])
                        hat[l_mask] = (pb_wl[l_mask] - ctrl_wl[j-1]) / (ctrl_wl[j] - ctrl_wl[j-1])
                        hat[r_mask] = (ctrl_wl[j+1] - pb_wl[r_mask]) / (ctrl_wl[j+1] - ctrl_wl[j])

                    for i in range(n_xp_s3):
                        fl_pb = fl_xp_norm[i][pb]
                        W[i, j] = float(_trapz(fl_pb * T_pb * hat, x=pb_wl))

                ch_data[ch_name] = {
                    "T_c":     T_c,
                    "pb":      pb,
                    "pb_wl":   pb_wl,
                    "T_pb":    T_pb,
                    "ctrl_wl": ctrl_wl,
                    "W":       W,
                    "meas_n":  meas_n,
                    "k_c":     k_c_init,
                }

            active_channels = [
                (ch_name, ch_data[ch_name])
                for ch_name in ("R", "G", "B")
                if ch_data[ch_name] is not None
            ]

            # Stage 2 gains used throughout — we minimize the SAME residual
            # that gets reported as RMS, not an internal surrogate.
            s3_gains = {"R": k_R, "G": 1.0, "B": k_B}

            # ── x0: Stage 2 operating point (flat per-channel scale) ──────────
            # Compute the uniform scale for each channel so W @ x0_flat ≈ meas_n/k_c.
            x0 = np.ones(len(active_channels) * N_CTRL, dtype=np.float64)
            ch_scales = {}
            for ci, (ch_name, cd) in enumerate(active_channels):
                W_flat = cd["W"].sum(axis=1)
                W_med  = float(np.median(W_flat[W_flat > eps]))
                k_c    = s3_gains[ch_name]
                if W_med > eps and k_c > eps:
                    scale = float(np.median(cd["meas_n"])) / (k_c * W_med)
                    scale = max(scale, eps)
                else:
                    scale = 1.0
                ch_scales[ch_name] = scale
                x0[ci * N_CTRL : (ci + 1) * N_CTRL] = scale

            # ── Prior session seeding (ctrl_points space only) ────────────────
            # We seed from prior ctrl_points if available — these are stored in
            # the same W-matrix scale as x_cur, so interpolation is safe.
            # We do NOT use the stitched sr.response array: that is a filter-
            # transmission-weighted composite across channels and maps very
            # poorly back to individual channel control points (causes 10^13 RMS).
            seeded_from_prior = False

            # Prior seeding is only valid when:
            #   1. Prior has ctrl_points (not just stitched response)
            #   2. Prior N_CTRL matches current N_CTRL — W-matrix scale is
            #      proportional to hat function width, which depends on N_CTRL.
            #      An 8-point prior seeded into a 16-point solver has ~2x scale
            #      mismatch, sending sweep 1 to RMS >> Stage 2 RMS.
            #   3. Sanity check: seeded x0 loss is not dramatically worse than
            #      the flat Stage 2 x0 (catches any other scale mismatches).
            prior_n_ctrl = None
            if prior_response is not None and prior_response.ctrl_points is not None:
                # Read stored n_ctrl if available, else infer from wl array length
                for _cp in prior_response.ctrl_points.values():
                    prior_n_ctrl = int(_cp.get("n_ctrl", len(_cp["wl"])))
                    break

            can_seed = (
                prior_response is not None
                and prior_response.stage >= 3
                and prior_response.residual_rms < 2.0
                and prior_response.ctrl_points is not None
                and prior_n_ctrl == N_CTRL   # N_CTRL must match exactly
            )

            if can_seed:
                # Build candidate seeded x0
                x0_seeded = x0.copy()
                for ci, (ch_name, cd) in enumerate(active_channels):
                    prior_cp = prior_response.ctrl_points.get(ch_name)
                    if prior_cp is None:
                        continue
                    prior_wl   = np.asarray(prior_cp["wl"],   dtype=np.float64)
                    prior_vals = np.asarray(prior_cp["vals"], dtype=np.float64)
                    if len(prior_wl) < 2 or len(prior_vals) < 2:
                        continue
                    seeded_vals = np.interp(
                        cd["ctrl_wl"], prior_wl, prior_vals,
                        left=prior_vals[0], right=prior_vals[-1],
                    )
                    seeded_vals = np.clip(seeded_vals, eps, None)
                    x0_seeded[ci * N_CTRL : (ci + 1) * N_CTRL] = seeded_vals

                # Sanity check: seeded loss must be <= 5x the flat Stage 2 loss.
                # If it's worse, the prior scale is incompatible (e.g. different
                # N_CTRL, different filter set, or corrupted cache entry).
                def _quick_loss(x_test):
                    total = 0.0
                    for ci2, (ch2, cd2) in enumerate(active_channels):
                        r2 = np.maximum(x_test[ci2 * N_CTRL : (ci2+1) * N_CTRL], 0.0)
                        I2 = cd2["W"] @ r2
                        I2s = np.where(I2 > eps, I2, eps)
                        total += float(np.sum((cd2["meas_n"] / (s3_gains[ch2] * I2s) - 1.0)**2))
                    return total

                loss_flat   = _quick_loss(x0)
                loss_seeded = _quick_loss(x0_seeded)

                if loss_seeded <= 5.0 * loss_flat:
                    x0 = x0_seeded
                    seeded_from_prior = True
                    status_cb(
                        f"[SSSC] Stage 3: seeding from prior ctrl_points "
                        f"({prior_response.n_stars} stars, "
                        f"RMS={prior_response.residual_rms:.4f}, "
                        f"N_CTRL={N_CTRL})"
                    )
                else:
                    status_cb(
                        f"[SSSC] Stage 3: prior seed rejected — "
                        f"seeded loss ({loss_seeded:.1f}) > 5x flat loss ({loss_flat:.1f}), "
                        f"starting from Stage 2 operating point"
                    )
            else:
                if prior_response is not None and prior_n_ctrl != N_CTRL:
                    status_cb(
                        f"[SSSC] Stage 3: prior N_CTRL={prior_n_ctrl} != current {N_CTRL} "
                        f"— W-matrix scale mismatch, starting fresh"
                    )
                elif prior_response is not None and prior_response.ctrl_points is None:
                    status_cb(
                        "[SSSC] Stage 3: prior session has no ctrl_points — "
                        "starting from Stage 2 operating point"
                    )
                elif prior_response is not None and prior_response.residual_rms >= 2.0:
                    status_cb(
                        f"[SSSC] Stage 3: prior RMS too high "
                        f"({prior_response.residual_rms:.4f}) — starting fresh"
                    )
                else:
                    status_cb(
                        "[SSSC] Stage 3: no prior session — "
                        "starting from Stage 2 operating point"
                    )

            def _rms_loss(x):
                """
                Sum of squared fractional residuals using Stage 2 gains.
                This is exactly what gets reported as RMS.
                """
                total = 0.0
                for ci, (ch_name, cd) in enumerate(active_channels):
                    r_vals = np.maximum(x[ci * N_CTRL : (ci + 1) * N_CTRL], 0.0)
                    I = cd["W"] @ r_vals
                    I_safe = np.where(I > eps, I, eps)
                    k_c = s3_gains[ch_name]
                    resid = cd["meas_n"] / (k_c * I_safe) - 1.0
                    total += float(np.sum(resid ** 2))
                return total

            def _rms_loss_1d(ci, ch_name, cd, j, I_rest, r_j):
                """1D loss: only this channel, only this control point varies."""
                r_j = max(r_j, 0.0)
                I = r_j * cd["W"][:, j] + I_rest
                I_safe = np.where(I > eps, I, eps)
                k_c = s3_gains[ch_name]
                resid = cd["meas_n"] / (k_c * I_safe) - 1.0
                return float(np.sum(resid ** 2))

            # ── Coordinate descent ────────────────────────────────────────────
            # Coordinate descent — directional search per control point.
            #
            # Strategy: for each point, try UP first (×1.1, ×1.2). If either
            # improves, take the best and move on. If not, try DOWN (×0.9, ×0.8).
            # If neither direction improves, stay put.
            #
            # This is much more efficient in the refinement phase (seeded from
            # prior) where most points are already near their optimum — we find
            # the right direction on the first probe and skip the other side.
            # On a fresh start all points need to move so both directions get
            # tried anyway, same as before.
            #
            # Step sizes ±10%/±20% per sweep. Large enough to make progress
            # on a fresh start (reaching 0.1× from 1.0 takes ~22 sweeps at
            # 10% steps); small enough that no single point runs away from
            # its neighbors before they've had a chance to move.
            STEPS_UP   = [1.1, 1.2]   # try larger first
            STEPS_DOWN = [0.9, 0.8]   # try smaller if up didn't help

            x_cur = x0.copy()
            loss_prev = _rms_loss(x_cur)
            status_cb(
                f"[SSSC] Stage 3: coordinate descent "
                f"({len(active_channels)} channels × {N_CTRL} ctrl pts, "
                f"{n_xp_s3} stars)  "
                f"Stage 2 RMS={residual_rms:.4f}  "
                f"{'[seeded from prior]' if seeded_from_prior else '[fresh start]'}…"
            )
            try:
                QApplication.processEvents()
            except Exception:
                pass

            for sweep in range(60):
                any_improved = False
                for ci, (ch_name, cd) in enumerate(active_channels):
                    r_vals = np.maximum(x_cur[ci * N_CTRL : (ci + 1) * N_CTRL], 0.0)
                    for j in range(N_CTRL):
                        r_j       = r_vals[j]
                        I_rest    = cd["W"] @ r_vals - r_j * cd["W"][:, j]
                        cur_loss  = _rms_loss_1d(ci, ch_name, cd, j, I_rest, r_j)
                        best_r    = r_j
                        best_loss = cur_loss

                        # Try UP first
                        for s in STEPS_UP:
                            l = _rms_loss_1d(ci, ch_name, cd, j, I_rest, r_j * s)
                            if l < best_loss:
                                best_loss = l
                                best_r    = r_j * s

                        # Only try DOWN if UP didn't improve
                        if best_r == r_j:
                            for s in STEPS_DOWN:
                                l = _rms_loss_1d(ci, ch_name, cd, j, I_rest, r_j * s)
                                if l < best_loss:
                                    best_loss = l
                                    best_r    = r_j * s

                        if best_r != r_j:
                            r_vals[j] = best_r
                            any_improved = True
                    x_cur[ci * N_CTRL : (ci + 1) * N_CTRL] = r_vals

                loss_new = _rms_loss(x_cur)
                rms_new  = float(np.sqrt(loss_new / (n_xp_s3 * len(active_channels))))
                delta    = loss_prev - loss_new
                status_cb(
                    f"[SSSC] Stage 3 sweep {sweep + 1}: "
                    f"RMS={rms_new:.4f}  Δloss={delta:.4f}"
                )
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
                # True convergence: no point moved at all, or loss change negligible
                if not any_improved or delta < 1e-4:
                    break
                loss_prev = loss_new

            x_opt = x_cur

            # Reconstruct per-channel R(λ) on full grid
            R_channels = {}
            for ci, (ch_name, cd) in enumerate(active_channels):
                r_vals = np.maximum(x_opt[ci * N_CTRL : (ci + 1) * N_CTRL], 0.0)
                R_full = np.zeros(len(_WL_GRID), dtype=np.float64)
                R_full[cd["pb"]] = np.interp(
                    cd["pb_wl"], cd["ctrl_wl"], r_vals)
                R_channels[ch_name] = R_full

            # Stitch into single R(λ) weighted by filter transmission
            T_sum = T_R_f64 + T_G_f64 + T_B_f64
            T_safe = np.where(T_sum > eps, T_sum, eps)
            response_iter = (
                R_channels.get("R", np.ones(len(_WL_GRID))) * T_R_f64 +
                R_channels.get("G", np.ones(len(_WL_GRID))) * T_G_f64 +
                R_channels.get("B", np.ones(len(_WL_GRID))) * T_B_f64
            ) / T_safe
            response_iter = np.maximum(response_iter, 0.0)
            r_max = float(np.max(response_iter))
            if r_max > eps:
                response_iter /= r_max

            # Preserve Stage 2 gains for pixel correction
            gains       = gains   # already set by Stage 2 (9-element array)
            response    = response_iter
            poly_coeffs = None

            # Store per-channel ctrl_points for next-session seeding.
            # These are in W-matrix scale — safe to interpolate directly.
            ctrl_points = {}
            for ci, (ch_name, cd) in enumerate(active_channels):
                ctrl_points[ch_name] = {
                    "wl":     cd["ctrl_wl"].copy(),
                    "vals":   np.maximum(
                                  x_opt[ci * N_CTRL : (ci + 1) * N_CTRL], 0.0).copy(),
                    "n_ctrl": int(N_CTRL),   # stored so seeding can verify scale compatibility
                }

            # Store per-channel R(λ) for diagnostics panel, normalized TOGETHER
            # so relative heights are preserved. Divide all channels by the single
            # global max across R, G, B — the channel with highest response = 1.0,
            # others scale proportionally. This is what the plot needs to show
            # the true relative throughput of each channel.
            ch_response_norm = {}
            global_ch_max = max(
                float(np.max(v)) for v in R_channels.values()
                if np.max(v) > eps
            ) if R_channels else 1.0
            if global_ch_max < eps:
                global_ch_max = 1.0
            for ch_name, R_full in R_channels.items():
                ch_response_norm[ch_name] = R_full / global_ch_max

            # Final RMS
            rms_terms = []
            for ci, (ch_name, cd) in enumerate(active_channels):
                r_vals = np.maximum(x_opt[ci * N_CTRL : (ci + 1) * N_CTRL], 0.0)
                I_c    = cd["W"] @ r_vals
                I_safe = np.where(I_c > eps, I_c, eps)
                k_use  = s3_gains[ch_name]
                resid_c = cd["meas_n"] / (k_use * I_safe) - 1.0
                rms_terms.append(float(np.mean(resid_c ** 2)))
            residual_rms = float(np.sqrt(np.mean(rms_terms)))

            stage_rms[3] = residual_rms

            status_cb(
                f"[SSSC] Stage 3 complete — "
                f"k_R={k_R:.4f}  k_G=1.0000  k_B={k_B:.4f}  "
                f"RMS={residual_rms:.4f}"
            )

    # If Stage 2 never ran (< _STAGE2_MIN stars), fill stage_rms[1] with the
    # raw fallback value computed from the unclipped population.
    if 1 not in stage_rms:
        stage_rms[1] = rms_stage1_raw

    return SystemResponse(
        wl_ang          = _WL_GRID.copy(),
        response        = response,
        stage           = stage,
        n_stars         = n_stars,
        bv_range        = bv_range,
        session_id      = session_id,
        gains           = gains,
        residual_rms    = residual_rms,
        poly_coeffs     = poly_coeffs,
        ctrl_points     = ctrl_points,
        stage_rms       = stage_rms,
        channel_response = ch_response_norm if stage >= 3 else None,
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
    """
    if status_cb is None:
        status_cb = lambda m: None

    eps = 1e-8
    k_R = float(sr.gains[0])
    k_G = float(sr.gains[1])
    k_B = float(sr.gains[2])

    if sr.stage <= 2:
        status_cb(f"[SSSC] Applying Stage {sr.stage} correction…")

        calibrated = img_float.copy()
        R = calibrated[..., 0]
        G = calibrated[..., 1]
        B = calibrated[..., 2]

        if sr.stage == 2 and len(sr.gains) == 9:
            aR, bR, cR = sr.gains[3], sr.gains[4], sr.gains[5]
            aB, bB, cB = sr.gains[6], sr.gains[7], sr.gains[8]

            RG = R / np.maximum(G, eps)
            BG = B / np.maximum(G, eps)

            mR = np.clip(aR * RG**2 + bR * RG + cR, 0.25, 4.0)
            mB = np.clip(aB * BG**2 + bB * BG + cB, 0.25, 4.0)

            calibrated[..., 0] = _pivot_scale_channel(
                R, mR / np.maximum(RG, eps), float(np.median(R)))
            calibrated[..., 2] = _pivot_scale_channel(
                B, mB / np.maximum(BG, eps), float(np.median(B)))
        else:
            scale_R = float(np.clip(1.0 / max(k_R, eps), 0.25, 4.0))
            scale_B = float(np.clip(1.0 / max(k_B, eps), 0.25, 4.0))
            calibrated[..., 0] = _pivot_scale_channel(R, scale_R, float(np.median(R)))
            calibrated[..., 2] = _pivot_scale_channel(B, scale_B, float(np.median(B)))

        return np.clip(calibrated, 0.0, 1.0).astype(np.float32)

    else:
        # Stage 3: apply via Stage 2 quadratic gains (well-conditioned)
        status_cb("[SSSC] Applying Stage 3 correction (via Stage 2 gains)…")

        calibrated = img_float.copy()
        R = calibrated[..., 0]
        G = calibrated[..., 1]
        B = calibrated[..., 2]

        if len(sr.gains) == 9:
            aR, bR, cR = sr.gains[3], sr.gains[4], sr.gains[5]
            aB, bB, cB = sr.gains[6], sr.gains[7], sr.gains[8]
            RG = R / np.maximum(G, eps)
            BG = B / np.maximum(G, eps)
            mR = np.clip(aR * RG**2 + bR * RG + cR, 0.25, 4.0)
            mB = np.clip(aB * BG**2 + bB * BG + cB, 0.25, 4.0)
            calibrated[..., 0] = _pivot_scale_channel(
                R, mR / np.maximum(RG, eps), float(np.median(R)))
            calibrated[..., 2] = _pivot_scale_channel(
                B, mB / np.maximum(BG, eps), float(np.median(B)))
        else:
            scale_R = float(np.clip(1.0 / max(k_R, eps), 0.25, 4.0))
            scale_B = float(np.clip(1.0 / max(k_B, eps), 0.25, 4.0))
            calibrated[..., 0] = _pivot_scale_channel(R, scale_R, float(np.median(R)))
            calibrated[..., 2] = _pivot_scale_channel(B, scale_B, float(np.median(B)))

        return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


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
      1. Solved R(λ) vs manufacturer QE
      2. Before/After residual scatter in (R/G, B/G) ratio space
      3. Bootstrap stage indicator and star count history
      4. B-V color distribution of calibrator population
    """
    _force_mpl_no_tex()
    fig = figure
    fig.clf()

    eps = 1e-12

    # ── Panel 1: Per-channel solved R(λ) ─────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    wl_nm = wl_grid / 10.0  # Å → nm

    # Per-channel R(λ) curves — each solved independently within its passband.
    # For Stage 3 we have channel_response with per-channel shapes.
    # For Stage 1/2 we fall back to the stitched composite.
    ch_resp = getattr(sr, "channel_response", None)
    ch_cfg = [
        ("R", T_sys_R, "#ee4444", "R channel R(λ)"),
        ("G", T_sys_G, "#44cc44", "G channel R(λ)"),
        ("B", T_sys_B, "#4488ee", "B channel R(λ)"),
    ]

    if ch_resp is not None:
        # Detect stale per-channel-normalized cache entries: if all three channels
        # peak at ~1.0, they were normalized individually (old format) and relative
        # heights are lost. Fall back to composite for that session.
        ch_peaks = [float(np.max(ch_resp[ch])) for ch in ("R", "G", "B") if ch in ch_resp]
        all_near_one = all(abs(p - 1.0) < 0.05 for p in ch_peaks) and len(ch_peaks) == 3
        if all_near_one:
            ch_resp = None   # treat as Stage 1/2 and fall through to composite

    if ch_resp is not None:
        # channel_response values are already normalized together globally
        # (divided by the single max across all channels in the solver).
        # Just plot them directly with zero-anchor points at passband edges.
        for ch_name, T_c, color, label in ch_cfg:
            R_ch = ch_resp.get(ch_name)
            if R_ch is None:
                continue
            T_peak = float(np.max(T_c))
            if T_peak < eps:
                continue
            pb_mask = T_c >= 0.05 * T_peak
            pb_indices = np.where(pb_mask)[0]
            if len(pb_indices) == 0:
                continue

            i_lo = pb_indices[0]
            i_hi = pb_indices[-1]
            lo_wl = wl_nm[max(0, i_lo - 1)]
            hi_wl = wl_nm[min(len(wl_nm) - 1, i_hi + 1)]

            wl_plot = np.concatenate([[lo_wl], wl_nm[pb_mask], [hi_wl]])
            R_plot  = np.concatenate([[0.0],   R_ch[pb_mask],  [0.0]])

            ax1.plot(wl_plot, R_plot, color=color, linewidth=2.5,
                     label=label, zorder=4)
    else:
        # Stage 1/2: single composite (stitched) — grey to distinguish from Stage 3
        ax1.plot(wl_nm, sr.response, color="#888888", linewidth=2.0,
                 linestyle="--", label=f"R(λ) composite  [{sr.stage_label}]",
                 zorder=4)

    if manufacturer_qe is not None:
        qe_norm = manufacturer_qe / max(float(np.max(manufacturer_qe)), eps)
        ax1.plot(wl_nm, qe_norm, color="#ffaa44", linewidth=1.5,
                 linestyle="--", label=manufacturer_qe_label, alpha=0.85, zorder=3)

    # Filter passband fills — light shading only
    for T_c, color in [(T_sys_R, "red"), (T_sys_G, "green"), (T_sys_B, "blue")]:
        T_peak = float(np.max(T_c))
        if T_peak < eps:
            continue
        T_norm = T_c / T_peak
        ax1.fill_between(wl_nm, T_norm, alpha=0.07, color=color, zorder=1)
        ax1.plot(wl_nm, T_norm, color=color, linewidth=0.7,
                 linestyle=":", alpha=0.5, zorder=2)

    ax1.set_xlim(300, 1100)
    ax1.set_ylim(0, 1.15)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Normalized throughput")
    if ch_resp is not None:
        ax1.set_title("Solved R(λ) per Channel  [Stage 3]")
    elif manufacturer_qe is not None:
        ax1.set_title("Solved R(λ)  vs  Manufacturer QE")
    else:
        ax1.set_title(f"System Response  [{sr.stage_label}]")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Residuals before/after ──────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)

    k_R = float(sr.gains[0])
    k_G = float(sr.gains[1])
    k_B = float(sr.gains[2])

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

    # ── Panel 3: Per-stage RMS waterfall ─────────────────────────────────────
    # Shows how much each bootstrap stage contributed to calibration quality.
    # Bar length = RMS (longer = worse), so improvement reads left-to-right.
    ax3 = fig.add_subplot(2, 2, 3)

    # Collect per-stage RMS from sr.stage_rms (may be missing for old cached sessions)
    s_rms = getattr(sr, "stage_rms", {sr.stage: sr.residual_rms})

    wf_rows = [
        (1, "Stage 1\nScalar gains",   "#888888", _STAGE1_MIN),
        (2, "Stage 2\nColor model",    "#5588cc", _STAGE2_MIN),
        (3, "Stage 3\nFull R(\u03bb)",      "#44cc88", _STAGE3_MIN),
        (4, "Stage 4\nAtmosphere",     "#cc8844", _STAGE4_MIN),
    ]

    # RMS axis: scale to the worst RMS that ran, with a little headroom
    ran_rms = [v for k, v in s_rms.items() if v > 0]
    rms_max = max(ran_rms) * 1.15 if ran_rms else 1.0

    for i, (stg, lbl, col, thresh) in enumerate(wf_rows):
        rms_val = s_rms.get(stg)
        did_run = rms_val is not None
        bar_color = col if did_run else "#2a2a2a"
        bar_alpha = 0.85 if did_run else 0.25

        if did_run:
            ax3.barh(i, rms_val, color=bar_color, alpha=bar_alpha,
                     height=0.55, zorder=3)
            # RMS value label inside or outside bar
            label_x = rms_val + rms_max * 0.01
            ax3.text(label_x, i, f"RMS={rms_val:.4f}",
                     va="center", ha="left", fontsize=8.5,
                     color=col, fontweight="bold", zorder=4)

            # Improvement delta vs previous stage
            prev_stg = stg - 1
            prev_rms = s_rms.get(prev_stg)
            if prev_rms is not None and prev_rms > rms_val:
                delta = prev_rms - rms_val
                ax3.text(rms_max * 0.99, i,
                         f"▼ {delta:.4f}",
                         va="center", ha="right", fontsize=7.5,
                         color="#aaffaa", alpha=0.9, zorder=4)
        else:
            # Stage not yet reached — show as a faint placeholder
            ax3.barh(i, rms_max * 0.15, color=bar_color, alpha=bar_alpha,
                     height=0.55, zorder=2)
            stars_needed = max(0, thresh - sr.n_stars)
            if stg == 4:
                note = "Requires DR4 + multi-night airmass data"
            elif stars_needed > 0:
                note = f"need {stars_needed:,} more stars"
            else:
                note = "criteria met — runs next"
            ax3.text(rms_max * 0.16, i, note,
                     va="center", ha="left", fontsize=7.5,
                     color="#666666", style="italic", zorder=3)

    ax3.set_yticks(range(4))
    ax3.set_yticklabels([r[1] for r in wf_rows], fontsize=8)
    ax3.set_xlim(0, rms_max)
    ax3.set_xlabel("Fractional residual RMS  (lower = better)")
    ax3.set_title("Calibration Quality by Stage")
    ax3.grid(True, axis="x", alpha=0.25, zorder=1)
    ax3.set_axisbelow(True)

    # Session context annotation bottom-right
    bv_span = sr.bv_range[1] - sr.bv_range[0]
    k_R_disp = float(sr.gains[0])
    k_B_disp = float(sr.gains[2])
    ctx = (
        f"{sr.n_stars:,} stars  ·  B-V span={bv_span:.2f}\nk_R={k_R_disp:.4f}  k_G=1.0000  k_B={k_B_disp:.4f}"
    )
    ax3.text(0.99, 0.03, ctx, transform=ax3.transAxes,
             ha="right", va="bottom", fontsize=7.5,
             color="#aaaaaa", family="monospace")

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
    ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", alpha=0.3)

    if sr.stage < 3:
        needed = _STAGE3_MIN - sr.n_stars
        ax4.set_title(
            f"Spectral Coverage  (need {needed} more stars for Stage 3)"
            if needed > 0 else "Spectral Coverage  ✓ Stage 3 ready"
        )
    else:
        ax4.set_title("Calibrator Spectral Coverage")

    # Build suptitle with per-stage RMS waterfall summary if available
    _sr = getattr(sr, "stage_rms", {sr.stage: sr.residual_rms})
    _rms_parts = []
    for _stg, _lbl in [(1, "S1"), (2, "S2"), (3, "S3")]:
        if _stg in _sr:
            _rms_parts.append(f"{_lbl}={_sr[_stg]:.3f}")
    _rms_str = "  →  ".join(_rms_parts) if len(_rms_parts) > 1 else f"RMS={sr.residual_rms:.3f}"
    fig.suptitle(
        f"SSSC Calibration Report  ·  {sr.n_stars} stars  ·  "
        f"{sr.stage_label}  ·  {_rms_str}",
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
 
        self.sasp_viewer_window = None
        self._gaia_dl           = None
        self.center_ra          = None
        self.center_dec         = None
        self.wcs_header         = None
        self.pixscale           = None
        self.orientation        = None
 
        self._build_ui()
        self.load_settings()
        self._restore_geometry()
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
        # ── Shared style helpers ──────────────────────────────────────────────
        # Step action buttons: large, bold, colored so they read as primary actions
        STEP1_STYLE = """
            QPushButton {
                background: #1a6b5a;
                color: #e8f8f5;
                border: 1px solid #25a080;
                border-radius: 4px;
                padding: 5px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover  { background: #1f8570; border-color: #2ebf9a; }
            QPushButton:pressed { background: #154f42; }
        """
        STEP2_STYLE = """
            QPushButton {
                background: #1a4f2a;
                color: #e8f8ee;
                border: 1px solid #2a8040;
                border-radius: 4px;
                padding: 5px 14px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover  { background: #1f6535; border-color: #35a050; }
            QPushButton:pressed { background: #123520; }
        """
        # Utility buttons: subtle, clearly secondary
        UTIL_STYLE = """
            QPushButton {
                background: #2d2d2d;
                color: #aaaaaa;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover  { background: #383838; color: #cccccc; border-color: #555; }
            QPushButton:pressed { background: #222222; }
        """
        # Danger button (clear history)
        DANGER_STYLE = """
            QPushButton {
                background: #3a1f1f;
                color: #cc8888;
                border: 1px solid #6b3333;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover  { background: #4a2525; color: #ffaaaa; border-color: #8b4444; }
            QPushButton:pressed { background: #2a1515; }
        """
        # Label styling
        LBL_STYLE   = "color: #aaaaaa; font-size: 11px;"
        LBL_R_STYLE = "color: #cc6666; font-size: 11px; font-weight: bold;"
        LBL_G_STYLE = "color: #66aa66; font-size: 11px; font-weight: bold;"
        LBL_B_STYLE = "color: #6688cc; font-size: 11px; font-weight: bold;"
        COMBO_STYLE = """
            QComboBox {
                background: #2a2a2a;
                color: #cccccc;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 11px;
            }
            QComboBox:hover { border-color: #666; }
            QComboBox::drop-down { border: none; }
        """
        SPIN_STYLE = """
            QSpinBox {
                background: #2a2a2a;
                color: #cccccc;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 11px;
            }
            QSpinBox:hover { border-color: #666; }
        """
        EDIT_STYLE = """
            QLineEdit {
                background: #2a2a2a;
                color: #aaaaaa;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QLineEdit:focus { border-color: #25a080; color: #cccccc; }
        """
        CHK_STYLE = "color: #aaaaaa; font-size: 11px;"
        SEP_STYLE = "color: #444444;"
 
        def lbl(text, style=LBL_STYLE):
            w = QLabel(text)
            w.setStyleSheet(style)
            return w
 
        def vsep():
            """Thin vertical separator line."""
            s = QLabel("│")
            s.setStyleSheet(SEP_STYLE)
            return s
 
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 6, 8, 6)
 
        # ── Row 1: Step 1 + White Reference ──────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(8)
 
        self.fetch_btn = QPushButton("⭐  Step 1: Fetch Stars && Spectra")
        self.fetch_btn.setStyleSheet(STEP1_STYLE)
        self.fetch_btn.setMinimumHeight(30)
        self.fetch_btn.clicked.connect(self.fetch_stars)
        row1.addWidget(self.fetch_btn)
 
        row1.addWidget(vsep())
        row1.addWidget(lbl("White Reference:"))
        self.star_combo = QComboBox()
        self.star_combo.addItem("G2V (Solar)", userData="G2V")
        self.star_combo.addItem("Vega (A0V)",  userData="A0V")
        for sed in self.sed_list:
            if sed.upper() in ("A0V", "G2V"):
                continue
            self.star_combo.addItem(sed, userData=sed)
        self.star_combo.setStyleSheet(COMBO_STYLE)
        self.star_combo.setFixedWidth(140)
        row1.addWidget(self.star_combo)
        row1.addStretch()
        self.about_btn = QPushButton("About")
        self.about_btn.setStyleSheet(
            "QPushButton { background: #1a2f4a; color: #88bbee; border: 1px solid #2a5080;"
            " border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            " QPushButton:hover { background: #1f3a5f; color: #aaccff; border-color: #3a70aa; }"
            " QPushButton:pressed { background: #111f33; }"
        )
        self.about_btn.setToolTip("About SSSC — what it is and how it works")
        self.about_btn.clicked.connect(self._show_about)
        row1.addWidget(self.about_btn)        
        layout.addLayout(row1)
 
        # ── Row 2: RGB Filters ────────────────────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(8)
 
        row2.addWidget(lbl("R Filter:", LBL_R_STYLE))
        self.r_filter_combo = QComboBox()
        self.r_filter_combo.addItem("(None)")
        self.r_filter_combo.addItems(self.filter_list)
        self.r_filter_combo.setStyleSheet(COMBO_STYLE)
        row2.addWidget(self.r_filter_combo)
 
        row2.addWidget(vsep())
        row2.addWidget(lbl("G Filter:", LBL_G_STYLE))
        self.g_filter_combo = QComboBox()
        self.g_filter_combo.addItem("(None)")
        self.g_filter_combo.addItems(self.filter_list)
        self.g_filter_combo.setStyleSheet(COMBO_STYLE)
        row2.addWidget(self.g_filter_combo)
 
        row2.addWidget(vsep())
        row2.addWidget(lbl("B Filter:", LBL_B_STYLE))
        self.b_filter_combo = QComboBox()
        self.b_filter_combo.addItem("(None)")
        self.b_filter_combo.addItems(self.filter_list)
        self.b_filter_combo.setStyleSheet(COMBO_STYLE)
        row2.addWidget(self.b_filter_combo)
        row2.addStretch()
        layout.addLayout(row2)
 
        # ── Row 3: LP/cut filters + Camera label ─────────────────────────────
        row3 = QHBoxLayout()
        row3.setSpacing(8)
 
        row3.addWidget(lbl("LP/Cut 1:"))
        self.lp_filter_combo = QComboBox()
        self.lp_filter_combo.addItem("(None)")
        self.lp_filter_combo.addItems(self.filter_list)
        self.lp_filter_combo.setStyleSheet(COMBO_STYLE)
        row3.addWidget(self.lp_filter_combo)
 
        row3.addWidget(vsep())
        row3.addWidget(lbl("LP/Cut 2:"))
        self.lp_filter_combo2 = QComboBox()
        self.lp_filter_combo2.addItem("(None)")
        self.lp_filter_combo2.addItems(self.filter_list)
        self.lp_filter_combo2.setStyleSheet(COMBO_STYLE)
        row3.addWidget(self.lp_filter_combo2)
 
        row3.addWidget(vsep())
        row3.addWidget(lbl("Camera / Rig:"))
        self.camera_label_edit = QLineEdit()
        self.camera_label_edit.setPlaceholderText(
            "e.g. IMX492 · EdgeHD f/7  (separates session history per sensor)")
        self.camera_label_edit.setMaxLength(80)
        self.camera_label_edit.setStyleSheet(EDIT_STYLE)
        self.camera_label_edit.textChanged.connect(
            lambda v: QSettings().setValue("SSSC/CameraLabel", v))
        row3.addWidget(self.camera_label_edit, stretch=1)
        layout.addLayout(row3)
 
        # ── Row 4: Step 2 + options + session controls ────────────────────────
        row4 = QHBoxLayout()
        row4.setSpacing(8)
 
        self.run_btn = QPushButton("🔬  Step 2: Run SSSC Calibration")
        self.run_btn.setStyleSheet(STEP2_STYLE)
        self.run_btn.setMinimumHeight(30)
        self.run_btn.clicked.connect(self.run_sssc)
        row4.addWidget(self.run_btn)
 
        row4.addWidget(vsep())
 
        self.neutralize_chk = QCheckBox("BG Neutralize")
        self.neutralize_chk.setChecked(False)
        self.neutralize_chk.setStyleSheet(CHK_STYLE)
        self.neutralize_chk.setToolTip("Background Neutralization")
        row4.addWidget(self.neutralize_chk)
 
        row4.addWidget(vsep())
        row4.addWidget(lbl("Star σ:"))
        self.sep_thr_spin = QSpinBox()
        self.sep_thr_spin.setRange(2, 100)
        self.sep_thr_spin.setValue(15)
        self.sep_thr_spin.setFixedWidth(52)
        self.sep_thr_spin.setStyleSheet(SPIN_STYLE)
        self.sep_thr_spin.setToolTip("Star detection threshold (SEP σ)")
        self.sep_thr_spin.valueChanged.connect(
            lambda v: QSettings().setValue(_SK_SEP_THR, int(v)))
        row4.addWidget(self.sep_thr_spin)
 

        adv_box = QGroupBox("Advanced")
        adv_box.setStyleSheet("""
            QGroupBox {
                color: #555555;
                border: 1px solid #333333;
                border-radius: 4px;
                margin-top: 6px;
                font-size: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 3px;
            }
        """)
        adv_layout = QHBoxLayout(adv_box)
        adv_layout.setSpacing(6)
        adv_layout.setContentsMargins(6, 4, 6, 4)

        adv_layout.addWidget(lbl("R(λ) ctrl pts:"))
        self.nctrl_spin = QSpinBox()
        self.nctrl_spin.setRange(4, 24)
        self.nctrl_spin.setValue(8)
        self.nctrl_spin.setSingleStep(2)
        self.nctrl_spin.setFixedWidth(52)
        self.nctrl_spin.setStyleSheet(SPIN_STYLE)
        self.nctrl_spin.setToolTip(
            "Control points per channel for Stage 3 R(λ) solver.\n"
            "8 = default (200–600 stars)\n"
            "12 = finer (~600+ stars)\n"
            "16 = high resolution (~1000+ stars)")
        self.nctrl_spin.valueChanged.connect(
            lambda v: QSettings().setValue(_SK_N_CTRL, int(v)))
        adv_layout.addWidget(self.nctrl_spin)

        adv_layout.addWidget(lbl("M(Σ) ctrl pts:"))
        self.max_stars_spin = QSpinBox()
        self.max_stars_spin.setRange(100, 2000)
        self.max_stars_spin.setValue(500)
        self.max_stars_spin.setSingleStep(100)
        self.max_stars_spin.setFixedWidth(58)
        self.max_stars_spin.setStyleSheet(SPIN_STYLE)
        self.max_stars_spin.setToolTip(
            "Maximum calibrator stars for photometry.\n"
            "Stars are selected brightest-first, XP stars prioritized.\n"
            "500 is recommended — dim stars add noise, not accuracy.")
        self.max_stars_spin.valueChanged.connect(
            lambda v: QSettings().setValue("SSSC/MaxStars", int(v)))
        adv_layout.addWidget(self.max_stars_spin)

        row4.addWidget(vsep())
        row4.addWidget(adv_box)
        row4.addStretch()
 
        self.session_info_lbl = QLabel("")
        self.session_info_lbl.setStyleSheet("color: #606060; font-size: 10px;")
        row4.addWidget(self.session_info_lbl)
 
        row4.addWidget(vsep())
 
        self.clear_session_btn = QPushButton("Clear History")
        self.clear_session_btn.setStyleSheet(DANGER_STYLE)
        self.clear_session_btn.setToolTip(
            "Delete all saved R(λ) solutions for the current filter+camera combination")
        self.clear_session_btn.clicked.connect(self._clear_session_history)
        row4.addWidget(self.clear_session_btn)
 

 
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet(UTIL_STYLE)
        self.close_btn.clicked.connect(self.reject)
        row4.addWidget(self.close_btn)
        layout.addLayout(row4)
 
        # ── Status label ──────────────────────────────────────────────────────
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color: #888888; font-size: 11px; padding: 1px 2px;")
        layout.addWidget(self.count_label)
 
        # ── Matplotlib canvas ─────────────────────────────────────────────────
        self.figure = Figure(figsize=(14, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setVisible(False)
        layout.addWidget(self.canvas, stretch=1)
 
        self.reset_btn = QPushButton("Reset View / Close")
        self.reset_btn.setStyleSheet(UTIL_STYLE)
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

        self.max_stars_spin.setValue(int(s.value("SSSC/MaxStars", 500)))

        nctrl = int(s.value(_SK_N_CTRL, 8))
        if hasattr(self, "nctrl_spin"):
            self.nctrl_spin.setValue(nctrl)

        cam_label = s.value("SSSC/CameraLabel", "")
        if hasattr(self, "camera_label_edit"):
            self.camera_label_edit.setText(str(cam_label))

    def _camera_label(self) -> str:
        try:
            return self.camera_label_edit.text().strip()
        except Exception:
            return ""

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

    # ── Gaia helpers ──────────────────────────────────────────────────────────

    def _gaia_enabled(self) -> bool:
        return (GaiaDownloader is not None) and bool(HAS_GAIAXPY)

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
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
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
        if header is None:
            self.wcs = None
            return
        try:
            hdr = header.copy()
            if "RADECSYS" in hdr and "RADESYS" not in hdr:
                radesys_val = str(hdr["RADECSYS"]).strip()
                hdr["RADESYS"] = radesys_val
                try:
                    del hdr["RADECSYS"]
                except Exception:
                    pass
            if "EPOCH" in hdr and "EQUINOX" not in hdr:
                hdr["EQUINOX"] = hdr["EPOCH"]
                try:
                    del hdr["EPOCH"]
                except Exception:
                    pass
            self.wcs = WCS(hdr, naxis=2, relax=True)

            # Guard against a WCS with no celestial axes (naxis=0). This happens
            # when the header lacks valid CTYPE1/CTYPE2 RA/DEC keywords — e.g. a
            # purely linear WCS, or a header with stray WCS-like keywords but no
            # real plate solution. Calling .all_pix2world() on such a WCS later
            # raises the cryptic:
            #   "When providing two arguments, the array must be of shape (N, 0)"
            # Catch it here and fall through to the existing "no WCS" path instead.
            _wcs_check = self.wcs.celestial if hasattr(self.wcs, "celestial") else self.wcs
            if _wcs_check.naxis < 2:
                print("[SSSC] WCS has no celestial axes (naxis < 2) — "
                      "header is missing valid CTYPE1/CTYPE2 RA/DEC. "
                      "Treating as no WCS.")
                self.wcs = None
                return

            try:
                psm = self.wcs.pixel_scale_matrix
                self.pixscale = float(np.hypot(psm[0, 0], psm[1, 0]) * 3600.0)
            except Exception:
                self.pixscale = None
            try:
                self.center_ra, self.center_dec = [
                    float(x) for x in self.wcs.wcs.crval]
            except Exception:
                self.center_ra, self.center_dec = None, None
            try:
                self.wcs_header = self.wcs.to_header(relax=True)
            except Exception:
                self.wcs_header = None
            if "CROTA2" in hdr:
                try:
                    self.orientation = float(hdr["CROTA2"])
                except Exception:
                    self.orientation = None
            else:
                try:
                    cd1_1 = float(hdr.get("CD1_1", 0.0))
                    cd1_2 = float(hdr.get("CD1_2", 0.0))
                    self.orientation = math.degrees(math.atan2(cd1_2, cd1_1))
                except Exception:
                    self.orientation = None
        except Exception as e:
            print(f"[SSSC] WCS initialization error: {e}")
            self.wcs = None

    def _make_working_base_for_sep(self, img_float):
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        return _SFCC._make_working_base_for_sep(self, img_float)

    def _neutralize_background(self, rgb_f, *, remove_pedestal=False):
        from setiastro.saspro.sfcc import SFCCDialog as _SFCC
        return _SFCC._neutralize_background(self, rgb_f,
                                            remove_pedestal=remove_pedestal)

    # ── Step 1: Fetch Stars ───────────────────────────────────────────────────

    def fetch_stars(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astroquery.simbad import Simbad
        from astropy.io import fits as _fits

        img, hdr, _meta = self._get_active_image_and_header()
        self.current_image  = img
        self.current_header = hdr

        if self.current_header is None or self.current_image is None:
            QMessageBox.warning(self, "No Plate Solution",
                "Please plate-solve the active document first.")
            return

        try:
            self.initialize_wcs_from_header(self.current_header)
        except Exception:
            QMessageBox.critical(self, "WCS Error",
                "Could not build a 2D WCS from header.")
            return

        if not getattr(self, "wcs", None):
            QMessageBox.critical(self, "WCS Error",
                "Could not build a 2D WCS from header.")
            return

        wcs2 = self.wcs.celestial if hasattr(self.wcs, "celestial") else self.wcs
        H, W = self.current_image.shape[:2]

        _sfcc_status(self, "Detecting stars with SEP…")
        QApplication.processEvents()

        if self.current_image.dtype == np.uint8:
            img_float = self.current_image.astype(np.float32) / 255.0
        else:
            img_float = self.current_image.astype(np.float32, copy=False)

        base     = self._make_working_base_for_sep(img_float)
        gray     = np.mean(base, axis=2).astype(np.float32)
        bkg      = sep.Background(gray)
        data_sub = gray - bkg.back()
        err      = float(bkg.globalrms)

        sep_sigma = float(self.sep_thr_spin.value()) if hasattr(self, "sep_thr_spin") else 5.0
        sources   = sep.extract(data_sub, sep_sigma, err=err)

        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error", "SEP found no sources.")
            return

        r_fluxrad, _ = sep.flux_radius(
            gray, sources["x"], sources["y"],
            2.0 * sources["a"], 0.5,
            normflux=sources["flux"], subpix=5)
        mask    = (r_fluxrad > 0.2) & (r_fluxrad <= 10)
        sources = sources[mask]
        if sources.size == 0:
            QMessageBox.critical(self, "SEP Error",
                "All SEP detections rejected by radius filter.")
            return

        _sfcc_status(self,
            f"SEP detected {sources.size:,} stars — converting to sky coords…")
        QApplication.processEvents()

        xs = sources["x"].astype(np.float64)
        ys = sources["y"].astype(np.float64)

        try:
            sky_coords = wcs2.all_pix2world(np.column_stack([xs, ys]), 0)
        except Exception as e:
            QMessageBox.critical(self, "WCS Error", str(e))
            return

        valid_mask = (
            np.isfinite(sky_coords[:, 0]) & np.isfinite(sky_coords[:, 1]) &
            (sky_coords[:, 1] >= -90) & (sky_coords[:, 1] <= 90)
        )
        sources    = sources[valid_mask]
        sky_coords = sky_coords[valid_mask]

        if sources.size == 0:
            QMessageBox.critical(self, "WCS Error",
                "No valid sky coordinates after WCS conversion.")
            return

        self.star_list      = []
        gaia_source_map: dict[int, int] = {}

        if self._use_gaia_fallback():
            try:
                lib = get_library()
                if lib.installed_bands():
                    _sfcc_status(self,
                        f"Matching {sources.size:,} SEP stars against local Gaia library…")
                    QApplication.processEvents()

                    coords_list  = [(float(sky_coords[i, 0]), float(sky_coords[i, 1]))
                                    for i in range(len(sources))]
                    batch_results = lib.find_nearest_batch(
                        coords_list, radius_arcsec=3.0)

                    for sep_idx, (sid, sep_arcsec) in batch_results.items():
                        gaia_source_map[sep_idx] = sid

                    _sfcc_status(self,
                        f"Gaia library matched {len(gaia_source_map):,} of "
                        f"{sources.size:,} SEP stars")
                    QApplication.processEvents()
            except Exception as e:
                print(f"[SSSC] Gaia bulk match failed: {e}")

        for i in range(len(sources)):
            sid  = gaia_source_map.get(i)
            info = None
            if sid is not None:
                try:
                    info = get_library().get_source_info(sid)
                except Exception:
                    pass

            self.star_list.append({
                "ra":              float(sky_coords[i, 0]),
                "dec":             float(sky_coords[i, 1]),
                "x":               float(sources["x"][i]),
                "y":               float(sources["y"][i]),
                "a":               float(sources["a"][i]),
                "main_id":         None,
                "sp_clean":        None,
                "pickles_match":   None,
                "sp_source":       "gaia_xp" if sid is not None else None,
                "Bmag":            None,
                "Vmag":            None,
                "Rmag":            None,
                "gaia_source_id":  sid,
                "gaia_gmag":       info["gmag"] if info else None,
                "gaia_sep_arcsec": batch_results.get(i, (None, None))[1]
                                   if i in gaia_source_map else None,
            })

        pix_corners = np.array([[W/2, H/2], [0,0], [W,0], [0,H], [W,H]], dtype=float)
        try:
            sky_corners = wcs2.all_pix2world(pix_corners, 0)
        except Exception as e:
            QMessageBox.critical(self, "WCS Error", str(e))
            return

        center_sky  = SkyCoord(ra=float(sky_corners[0,0])*u.deg,
                               dec=float(sky_corners[0,1])*u.deg, frame="icrs")
        corners_sky = SkyCoord(ra=sky_corners[1:,0]*u.deg,
                               dec=sky_corners[1:,1]*u.deg, frame="icrs")
        radius = center_sky.separation(corners_sky).max() * 1.05

        Simbad.reset_votable_fields()
        ok = False
        for _ in range(5):
            try:
                Simbad.add_votable_fields("sp", "B", "V", "R", "ra", "dec", "main_id")
                ok = True; break
            except Exception:
                QApplication.processEvents(); non_blocking_sleep(0.8)
        if not ok:
            for _ in range(5):
                try:
                    Simbad.add_votable_fields("sp", "flux(B)", "flux(V)", "flux(R)",
                                              "ra(d)", "dec(d)", "main_id")
                    ok = True; break
                except Exception:
                    QApplication.processEvents(); non_blocking_sleep(0.8)

        simbad_result = None
        if ok:
            Simbad.ROW_LIMIT = 10000
            for attempt in range(1, 6):
                try:
                    _sfcc_status(self,
                        f"Querying SIMBAD for spectral types (attempt {attempt}/5)…")
                    QApplication.processEvents()
                    simbad_result = Simbad.query_region(center_sky, radius=radius)
                    break
                except Exception:
                    QApplication.processEvents(); non_blocking_sleep(1.2)

        templates_for_hist = []

        if simbad_result is not None and len(simbad_result) > 0:
            cols_lower  = {c.lower(): c for c in simbad_result.colnames}
            ra_col      = (cols_lower.get("ra") or cols_lower.get("ra(d)")
                           or cols_lower.get("ra_d"))
            dec_col     = (cols_lower.get("dec") or cols_lower.get("dec(d)")
                           or cols_lower.get("dec_d"))
            b_col       = cols_lower.get("b") or cols_lower.get("flux_b")
            v_col       = cols_lower.get("v") or cols_lower.get("flux_v")
            r_col       = cols_lower.get("r") or cols_lower.get("flux_r")
            main_id_col = cols_lower.get("main_id")

            def _unmask_num(x):
                try:
                    if x is None: return None
                    if ma.isMaskedArray(x) and ma.is_masked(x): return None
                    return float(x)
                except Exception:
                    return None

            def _infer(bv):
                if bv is None or (isinstance(bv, float) and np.isnan(bv)):
                    return None
                if bv < 0.00: return "B"
                elif bv < 0.30: return "A"
                elif bv < 0.58: return "F"
                elif bv < 0.81: return "G"
                elif bv < 1.40: return "K"
                else: return "M"

            sl_ras  = np.array([s["ra"]  for s in self.star_list], dtype=np.float64)
            sl_decs = np.array([s["dec"] for s in self.star_list], dtype=np.float64)
            matched_simbad = 0

            for row in simbad_result:
                if ra_col is None or dec_col is None:
                    continue
                sra  = _unmask_num(row[ra_col])
                sdec = _unmask_num(row[dec_col])
                if sra is None or sdec is None:
                    continue

                cosd  = max(1e-6, abs(math.cos(math.radians(sdec))))
                seps  = np.hypot((sl_ras - sra) * cosd,
                                 sl_decs - sdec) * 3600.0
                j = int(np.argmin(seps))
                if seps[j] > 3.0:
                    continue

                st   = self.star_list[j]
                raw_sp = None
                if "SP_TYPE" in simbad_result.colnames:
                    raw_sp = row["SP_TYPE"]
                elif "sp_type" in simbad_result.colnames:
                    raw_sp = row["sp_type"]

                bmag = _unmask_num(row[b_col]) if b_col else None
                vmag = _unmask_num(row[v_col]) if v_col else None
                rmag = _unmask_num(row[r_col]) if r_col else None

                sp_clean = sp_source = None
                if raw_sp and str(raw_sp).strip():
                    sp = str(raw_sp).strip().upper()
                    if not (sp.startswith("SN") or sp.startswith("KA")):
                        sp_clean  = sp
                        sp_source = "simbad"
                elif bmag is not None and vmag is not None:
                    sp_clean  = _infer(bmag - vmag)
                    sp_source = "bv_inferred"

                match_list    = pickles_match_for_simbad(
                    sp_clean, self.pickles_templates) if sp_clean else []
                best_template = match_list[0] if match_list else None

                st["main_id"]       = (str(row[main_id_col]).strip()
                                       if main_id_col and row[main_id_col] else None)
                st["sp_clean"]      = sp_clean
                st["sp_source"]     = (sp_source
                                       if st["sp_source"] != "gaia_xp" else "gaia_xp")
                st["pickles_match"] = best_template
                st["Bmag"]          = float(bmag) if bmag is not None else None
                st["Vmag"]          = float(vmag) if vmag is not None else None
                st["Rmag"]          = float(rmag) if rmag is not None else None

                if best_template:
                    templates_for_hist.append(best_template)
                matched_simbad += 1

            _sfcc_status(self,
                f"SIMBAD matched {matched_simbad} stars — "
                f"{len(gaia_source_map):,} have Gaia XP spectra")
            QApplication.processEvents()

        if self._use_gaia_fallback():
            gaia_ids_all = [st["gaia_source_id"] for st in self.star_list
                            if st.get("gaia_source_id") is not None]
            if gaia_ids_all:
                try:
                    _sfcc_busy(self, True,
                        f"Inferring spectral types from Gaia XP for "
                        f"{len(gaia_ids_all):,} stars…")
                    needs_bvr = sorted(set(
                        int(st["gaia_source_id"]) for st in self.star_list
                        if st.get("gaia_source_id") is not None
                        and (st.get("gaia_B") is None or st.get("gaia_V") is None)
                    ))
                    if needs_bvr:
                        cache_dir = os.path.join(
                            QStandardPaths.writableLocation(
                                QStandardPaths.StandardLocation.AppDataLocation),
                            "gaiaxpy_cache")
                        os.makedirs(cache_dir, exist_ok=True)
                        from setiastro.saspro.sfcc import _gaiaxp_synth_bvr_cached
                        bvr_map = _gaiaxp_synth_bvr_cached(
                            self, needs_bvr,
                            db_path=self._gaia_db_path(),
                            status_cb=lambda m: _sfcc_status(self, m),
                            cache_dir=cache_dir,
                        )
                        for st in self.star_list:
                            sid_i = st.get("gaia_source_id")
                            if sid_i is None or int(sid_i) not in bvr_map:
                                continue
                            bvr = bvr_map[int(sid_i)]
                            st["gaia_B"] = bvr["B"]
                            st["gaia_V"] = bvr["V"]
                            st["gaia_R"] = bvr["R"]
                            if st.get("Bmag") is None: st["Bmag"] = bvr["B"]
                            if st.get("Vmag") is None: st["Vmag"] = bvr["V"]
                            if st.get("Rmag") is None: st["Rmag"] = bvr["R"]
                            if not st.get("sp_clean"):
                                letter = _infer_letter_from_bv(bvr["B"] - bvr["V"])
                                if letter:
                                    st["sp_clean"]  = letter
                                    st["sp_source"] = "bv_inferred"
                            if st.get("sp_clean") and not st.get("pickles_match"):
                                ml = pickles_match_for_simbad(
                                    st["sp_clean"], self.pickles_templates)
                                st["pickles_match"] = ml[0] if ml else None
                                if st["pickles_match"]:
                                    templates_for_hist.append(st["pickles_match"])
                except Exception as e:
                    _sfcc_status(self, f"[SSSC] Gaia XP BVR failed: {e}")
                finally:
                    _sfcc_busy(self, False)

        n_gaia_xp        = sum(1 for s in self.star_list
                               if s.get("gaia_source_id") is not None)
        n_simbad_pickles = sum(1 for s in self.star_list
                               if s.get("gaia_source_id") is None
                               and s.get("sp_source") == "simbad"
                               and s.get("pickles_match") is not None)
        n_bv_pickles     = sum(1 for s in self.star_list
                               if s.get("gaia_source_id") is None
                               and s.get("sp_source") == "bv_inferred")
        n_none           = (len(self.star_list)
                            - n_gaia_xp - n_simbad_pickles - n_bv_pickles)

        if getattr(self, "figure", None) is not None:
            self.figure.clf()

        if (getattr(self, "figure", None) is not None
                and getattr(self, "canvas", None) is not None):
            fig = self.figure

            ax1 = fig.add_subplot(1, 2, 1)
            if templates_for_hist:
                uniq, cnt = np.unique(templates_for_hist, return_counts=True)
                ax1.bar(uniq, cnt, edgecolor="black", color="#5588cc", label="Pickles")
            bv_types = [s["sp_clean"] for s in self.star_list
                        if s.get("sp_source") == "bv_inferred" and s.get("sp_clean")]
            if bv_types:
                bv_uniq, bv_cnt = np.unique(bv_types, return_counts=True)
                ax1.bar(bv_uniq, bv_cnt, edgecolor="black", color="#cc8844",
                        alpha=0.7, label="B-V inferred")
            ax1.set_xlabel("Spectral Type")
            ax1.set_ylabel("Count")
            ax1.set_title("Spectral Type Distribution")
            ax1.tick_params(axis="x", rotation=90)
            ax1.grid(axis="y", linestyle="--", alpha=0.3)
            ax1.legend(fontsize=8)

            ax2 = fig.add_subplot(1, 2, 2)
            labels, sizes, colors = [], [], []
            for label, size, color in [
                (f"Gaia XP ({n_gaia_xp})",               n_gaia_xp,        "#44cc88"),
                (f"Pickles/SIMBAD ({n_simbad_pickles})",  n_simbad_pickles, "#5588cc"),
                (f"Pickles/B-V ({n_bv_pickles})",         n_bv_pickles,     "#cc8844"),
                (f"No spectrum ({n_none})",                n_none,           "#555555"),
            ]:
                if size > 0:
                    labels.append(label); sizes.append(size); colors.append(color)
            if sizes:
                ax2.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
                        textprops={"fontsize": 8})
            ax2.set_title("Calibration Source Breakdown")

            fig.tight_layout()
            self.canvas.setVisible(True)
            _force_mpl_no_tex()
            self.canvas.draw()

        _sfcc_status(self,
            f"Step 1 complete — {len(self.star_list):,} stars  ·  "
            f"{n_gaia_xp:,} Gaia XP  ·  {n_simbad_pickles:,} Pickles/SIMBAD  ·  "
            f"{n_bv_pickles:,} Pickles/B-V  ·  {n_none:,} unclassified")

        self._update_session_info_label()

    def _show_about(self):
        """Display the SSSC About dialog."""
        from PyQt6.QtWidgets import QScrollArea, QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle("About SSSC — Spectrophotometric Standard Star Calibration")
        dlg.setMinimumSize(720, 580)
        dlg.resize(760, 640)
        layout = QVBoxLayout(dlg)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("font-size: 13px; background: #1a1a1a; color: #dddddd;")
        browser.setHtml("""
<style>
  body  { font-family: sans-serif; font-size: 13px; color: #dddddd;
          background: #1a1a1a; margin: 16px; line-height: 1.55; }
  h2    { color: #44cc88; margin-top: 18px; margin-bottom: 4px; }
  h3    { color: #88bbff; margin-top: 14px; margin-bottom: 3px; }
  p     { margin: 6px 0; }
  ul    { margin: 4px 0 8px 20px; }
  li    { margin-bottom: 3px; }
  .stage  { color: #ffcc66; font-weight: bold; }
  .note   { color: #aaaaaa; font-style: italic; }
  .future { color: #cc88ff; }
  code  { background: #2a2a2a; padding: 1px 4px; border-radius: 3px;
          color: #88ffcc; font-size: 12px; }
  hr    { border: none; border-top: 1px solid #333; margin: 14px 0; }
</style>
 
<h2>What is SSSC?</h2>
<p>
SSSC (Spectrophotometric Standard Star Calibration) is SetiAstro&rsquo;s next-generation
empirical color calibration, derived directly from the photometric calibration pipeline
used internally by the Gaia space mission (Riello et al. 2021). The same mathematical
framework that Gaia uses to self-calibrate a billion stars is applied here to calibrate
your image &mdash; using Gaia&rsquo;s own XP stellar spectra as the reference standard.
</p>
<p>
Traditional calibration tools require a <b>sensor QE curve</b> as input. The problem is
that most published QE curves are promotional material &mdash; measured at room temperature
on a bare die, often representing a best-case sample rather than an independently-verified
response for your specific sensor batch. They have no knowledge of your AR coating, your
optical train, your atmosphere, or your electronics chain. The curve you apply is rarely
the curve your camera actually has in practice.
</p>
<p>
<b>SSSC abandons the QE curve entirely.</b> Instead, it solves for the true effective
system throughput R(&lambda;) directly from your image data, using Gaia XP stellar
spectra as calibration references. No assumptions. No datasheets. Just your actual system,
measured empirically from the stars in your field.
</p>
 
<hr>
 
<h2>How does it work?</h2>
<p>
For each calibrator star, Gaia has measured a continuous flux spectrum from ~330&ndash;1050&nbsp;nm
at ~2.5&nbsp;nm resolution. SSSC integrates those spectra through your filter transmission
curves and solves for the R(&lambda;) that best explains the difference between what Gaia
predicts and what your camera actually measured. The core equation is:
</p>
<p style="margin-left:20px; font-family:monospace; color:#88ffcc;">
measured<sub>c</sub>(i) = k<sub>c</sub> &times; &int; flux<sub>i</sub>(&lambda;)
&times; T<sub>filter</sub>(&lambda;) &times; R(&lambda;) d&lambda;
</p>
<p>
R(&lambda;) absorbs everything the datasheet cannot tell you: true sensor QE at
operating temperature, mirror and lens throughput, AR coating, field flattener,
and atmospheric extinction. Filter transmission curves remain as inputs &mdash;
manufacturers publish reliable interferometric measurements for these.
</p>
 
<hr>
 
<h2>Bootstrap Stages</h2>
<p>
The solution quality improves automatically as more calibrator stars are available.
SSSC selects the highest stage your field supports:
</p>
<ul>
  <li><span class="stage">Stage 1 (&lt;50 stars):</span>
      Scalar per-channel gains k<sub>R</sub>, k<sub>G</sub>, k<sub>B</sub>.
      This is equivalent to traditional scalar SPCC &mdash; a single white-balance
      correction per channel. Always runs as the baseline.</li>
  <li><span class="stage">Stage 2 (50&ndash;200 stars):</span>
      Color-dependent gain within each band, fit as slope, affine, or quadratic
      depending on which best describes your data. Captures the dominant wavelength
      dependence within each filter passband. Comparable to SASpro&rsquo;s polynomial
      SPCC fit, which already exceeds traditional scalar SPCC quality on most normal
      star fields &mdash; and SSSC achieves this without any QE curve assumption at all.</li>
  <li><span class="stage">Stage 3 (200+ stars, B&minus;V span &ge; 1.5):</span>
      Full R(&lambda;) shape solved per channel using piecewise-linear control points
      and coordinate descent. Hot blue stars constrain the blue edge of each filter;
      cool red stars constrain the red edge. The solution refines with each run &mdash;
      prior sessions seed the next, so calibration accuracy accumulates over time.</li>
  <li><span class="stage future">Stage 4 (future &mdash; DR4 + multi-night):</span>
      Atmosphere/hardware decorrelation. By imaging across multiple nights at different
      airmasses, R(&lambda;) separates into a stable hardware component and a variable
      atmospheric component. Once converged, your empirically-measured QE curve is locked
      in permanently &mdash; calibration collapses back to Stage 2 speed with Stage 3+
      accuracy, using your actual curve instead of a manufacturer guess.</li>
</ul>
 
<hr>
 
<h2>What can SSSC do today with Gaia DR3?</h2>
<p>
Gaia DR3 provides continuous XP spectra for stars to about G&nbsp;&le;&nbsp;15.5.
On a typical wide-field image (0.5&ndash;2&deg; FOV) this yields 200&ndash;600
calibrator stars, comfortably in Stage 3 territory for most fields.
</p>
<p>
In testing, SSSC Stage 3 already outperforms traditional SPCC on the same data,
precisely because it does not inherit the QE curve assumption. The residual RMS
improvement from Stage 1 &rarr; Stage 2 &rarr; Stage 3 is visible in the
Calibration Quality waterfall panel on every run.
</p>
<p class="note">
Tip: run SSSC on multiple images from the same rig. Each run refines the R(&lambda;)
solution and seeds the next. The Camera&nbsp;/&nbsp;Rig label field separates history
per sensor so solutions don&rsquo;t mix between different imaging trains.
</p>
 
<hr>
 
<h2 class="future">What will SSSC do with Gaia DR4?</h2>
<p>
Gaia DR4 is expected to extend XP spectra to G&nbsp;&le;&nbsp;16.5, roughly
<b>tripling the calibrator star count</b> in most fields. This means:
</p>
<ul>
  <li>Fields that currently reach Stage 2 will reliably reach Stage 3</li>
  <li>Stage 3 with 16+ control points becomes well-constrained (&gt;1000 stars)</li>
  <li>Stage 4 atmosphere decorrelation becomes feasible &mdash; enough stars per
      session to resolve the airmass-dependent component of R(&lambda;)</li>
</ul>
<p>
The Stage 4 endgame: once your hardware R(&lambda;) has converged from multi-night
data, it is saved permanently to your session cache. Every subsequent calibration
run loads that curve as a known input &mdash; no solver needed, Stage 2 speed,
empirically-correct accuracy. The curve also self-maintains: mirror oxidation,
sensor aging, filter changes all appear as drift in R(&lambda;) and trigger an update.
</p>
 
<hr>
 
<h2>References</h2>
<ul>
  <li>Riello et al. 2021 &mdash; <i>Gaia EDR3: Photometric content and validation</i>,
      A&amp;A 649, A3 &mdash; the calibration pipeline SSSC replicates</li>
  <li>Carrasco et al. 2021 &mdash; <i>Gaia photometric science alerts programme</i></li>
  <li>Fabricius et al. 2021 &mdash; <i>Gaia EDR3: Catalogue validation</i></li>
</ul>
<p class="note" style="margin-top:16px;">
SSSC is part of SetiAstro Suite Pro &mdash; www.setiastro.com
</p>
""")
        layout.addWidget(browser)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec()

    def _clear_session_history(self):
        """Delete all saved R(λ) solutions for the current session ID, with confirmation."""
        try:
            sid = make_session_id(
                self.r_filter_combo.currentText(),
                self.g_filter_combo.currentText(),
                self.b_filter_combo.currentText(),
                self.lp_filter_combo.currentText(),
                self.lp_filter_combo2.currentText(),
                self._camera_label(),
            )
            cache = self._get_session_cache()
            n = cache.session_count(sid)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read session cache:\n{e}")
            return

        if n == 0:
            QMessageBox.information(self, "Clear Session History",
                "No saved solutions found for this filter+camera combination.")
            return

        reply = QMessageBox.question(
            self,
            "Clear Session History",
            f"Delete all {n} saved R(λ) solution(s) for session {sid}?\n\n"
            f"This will remove the accumulated calibration history for this\n"
            f"filter+camera combination. The next run will start fresh.\n\n"
            f"This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            import sqlite3
            cache._conn.execute(
                f"DELETE FROM {_SESSION_TABLE} WHERE session_id = ?", (sid,))
            cache._conn.commit()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clear session:\n{e}")
            return

        self._update_session_info_label()
        _sfcc_status(self, f"[SSSC] Session history cleared — {n} run(s) deleted for {sid}")
        QMessageBox.information(self, "Clear Session History",
            f"Cleared {n} saved solution(s) for session {sid}.\n"
            f"The next calibration run will start fresh.")

    def _update_session_info_label(self):
        try:
            sid = make_session_id(
                self.r_filter_combo.currentText(),
                self.g_filter_combo.currentText(),
                self.b_filter_combo.currentText(),
                self.lp_filter_combo.currentText(),
                self.lp_filter_combo2.currentText(),
                self._camera_label(),
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

        MAX_PHOT_STARS = int(self.max_stars_spin.value()) if hasattr(self, "max_stars_spin") else 500
        if len(raw_matches) > MAX_PHOT_STARS:
            def _match_priority(m):
                si = int(m["sim_index"])
                has_xp = (0 <= si < len(self.star_list)
                        and self.star_list[si].get("gaia_source_id") is not None)
                return (0 if has_xp else 1, -m["sep_flux"])
            raw_matches.sort(key=_match_priority)
            raw_matches = raw_matches[:MAX_PHOT_STARS]

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

        T_R = interp_tp(*load_curve(r_filt)) if r_filt != "(None)" else np.ones_like(wl_grid)
        T_G = interp_tp(*load_curve(g_filt)) if g_filt != "(None)" else np.ones_like(wl_grid)
        T_B = interp_tp(*load_curve(b_filt)) if b_filt != "(None)" else np.ones_like(wl_grid)
        LP1 = interp_tp(*load_curve(lp_filt))  if lp_filt  != "(None)" else np.ones_like(wl_grid)
        LP2 = interp_tp(*load_curve(lp_filt2)) if lp_filt2 != "(None)" else np.ones_like(wl_grid)
        LP  = LP1 * LP2

        T_sys_R = T_R * LP
        T_sys_G = T_G * LP
        T_sys_B = T_B * LP

        # Half-power clip on green for XP integral computation only.
        # This suppresses OSC Bayer red-tail bleed in S_star_G values used
        # by Stage 1/2 ratio math. We keep the unclipped T_sys_G for passing
        # to _solve_system_response where Stage 3 applies its own per-channel
        # passband threshold (PASSBAND_THRESH_PER_CH G=0.707).
        T_sys_G_integrals = T_sys_G.copy()
        T_peak_G = float(np.max(T_sys_G_integrals))
        if T_peak_G > 0:
            T_sys_G_integrals = np.where(
                T_sys_G_integrals >= 0.707 * T_peak_G, T_sys_G_integrals, 0.0)

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
                        T_sys_G=T_sys_G_integrals.astype(np.float64),   # ← clipped
                        T_sys_B=T_sys_B.astype(np.float64),
                        batch_size=25,
                    )
                    _sfcc_status(self,
                        f"Gaia XP: {len(gaia_integrals):,} integrals computed…")
                    QApplication.processEvents()
            except Exception as e:
                print(f"[SSSC] Gaia XP integrals failed: {e}")

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
                    float(_trapz(fs_i * T_sys_G_integrals, x=wl_grid)),  # ← clipped
                    float(_trapz(fs_i * T_sys_B, x=wl_grid)),
                )
            except Exception as e:
                print(f"[SSSC] Pickles {pname} failed: {e}")

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

        _sfcc_status(self, f"Pre-loading XP spectra for {n_with_spectrum} stars…")
        QApplication.processEvents()

        xp_flux_cache: dict[int, np.ndarray | None] = {}
        if self._use_gaia_fallback():
            dl = self._get_gaia_downloader()
            for m in raw_matches_with_spectrum:
                si  = int(m["sim_index"])
                sid = self.star_list[si].get("gaia_source_id") \
                      if 0 <= si < len(self.star_list) else None
                if sid is None or int(sid) in xp_flux_cache:
                    continue
                try:
                    spec = dl.db.get_spectrum(int(sid))
                    if spec is not None and spec.flux is not None:
                        wl_spec = np.asarray(
                            spec.wavelengths, dtype=np.float64) * 10.0
                        fl_spec = np.asarray(spec.flux, dtype=np.float64)
                        if wl_spec[0] > wl_spec[-1]:
                            wl_spec = wl_spec[::-1]
                            fl_spec = fl_spec[::-1]
                        arr = np.interp(
                            _WL_GRID, wl_spec, fl_spec, left=0.0, right=0.0)
                        xp_flux_cache[int(sid)] = np.where(arr > 0, arr, 0.0)
                    else:
                        xp_flux_cache[int(sid)] = None
                except Exception:
                    xp_flux_cache[int(sid)] = None

        _sfcc_status(self,
            f"Measuring flux for {n_with_spectrum} stars (parallel)…")
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

            xp_flux_arr = xp_flux_cache.get(int(sid)) if sid is not None else None

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
                "xp_flux":   xp_flux_arr,
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

        _sfcc_status(self, "Solving system response R(λ)…")
        QApplication.processEvents()

        session_id = make_session_id(r_filt, g_filt, b_filt, lp_filt, lp_filt2,
                                     self._camera_label())

        try:
            prior_sr = self._get_session_cache().load_latest(session_id)
            if prior_sr is not None and prior_sr.stage >= 3:
                has_cp = prior_sr.ctrl_points is not None
                _sfcc_status(self,
                    f"Found prior Stage 3 solution ({prior_sr.n_stars} stars, "
                    f"RMS={prior_sr.residual_rms:.4f}, "
                    f"ctrl_points={'yes' if has_cp else 'no'}) — will seed optimizer")
                QApplication.processEvents()
        except Exception:
            prior_sr = None

        n_ctrl_pts = int(self.nctrl_spin.value()) if hasattr(self, "nctrl_spin") else 8

        try:
            sr = _solve_system_response(
                enriched, wl_grid,
                T_sys_R, T_sys_G_integrals, T_sys_B,
                session_id,
                prior_response=prior_sr,
                n_ctrl=n_ctrl_pts,
                status_cb=lambda m: _sfcc_status(self, m),
            )
        except Exception as e:
            QMessageBox.critical(self, "Solver Error",
                f"Failed to solve system response:\n{e}")
            return

        self._last_sr = sr

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

        stars_needed = max(0, _STAGE3_MIN - sr.n_stars)
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
            + (
                f"  ✓ Stage 3 (full R(λ)) solved from {sr.n_stars} stars.\n"
                f"  Each run refines R(λ) further — run again on the same\n"
                f"  filter+camera combination to improve the solution."
                if sr.stage >= 3 else
                f"  {stars_needed} more stars needed for Stage 3 (full R(λ))."
                if stars_needed > 0 else
                f"  Stage 3 criteria met — will activate on next run."
            )
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _restore_geometry(self):
        s = QSettings()
        geom = s.value("SSSC/WindowGeometry")
        if geom:
            self.restoreGeometry(geom)

    def _save_geometry(self):
        QSettings().setValue("SSSC/WindowGeometry", self.saveGeometry())

    def _cleanup(self):
        self._save_geometry()
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