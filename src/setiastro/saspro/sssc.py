# ══════════════════════════════════════════════════════════════════════════════
# src/setiastro/saspro/sssc.py  —  Spectrophotometric Standard Star Calibration
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
sep.set_extract_pixstack(20000000)

# ── Matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ── PyQt6
from PyQt6.QtCore import (Qt, QSettings, QStandardPaths, QThread, QTimer,
                           pyqtSignal as _Signal)
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLabel, QLineEdit, QMainWindow, QPushButton,
    QDialog, QFileDialog, QInputDialog, QMessageBox, QWidget,
    QProgressDialog, QGroupBox, QFormLayout, QToolButton, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
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
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.widgets.factory_reset import (
    make_reset_button, snapshot_controls, restore_controls)

# ── Re-use shared utilities from sfcc.py
from setiastro.saspro.sfcc import (
    _force_mpl_no_tex,
    _debug_probe_channels,
    _pivot_scale_channel,
    pickles_match_for_simbad,
    measure_star_rgb_raw_aperture,
    _sfcc_status,
    _sfcc_busy,
    _GaiaSpectraWorker,
    _infer_letter_from_bv,
    _gray_for_sep,
    sep_detect_auto_sigma,
    measure_stars_rgb_photometry,
    _read_sasp_tables,
    _SimbadPrefetch,
    _primary_button_sheet,
    _draw_star_summary,
    _star_source_counts,
    _relayout_on_draw,
    _grow_to_fit,
    _seed_sep_rng,
    _RunButtonState,
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
# Photon-counting weight. A sensor counts photons (energy hc/λ each), so the
# counts a spectrum F_λ gives through T are ∫F_λ·λ·T dλ, not ∫F_λ·T dλ.
# Every expected integral (XP, Pickles, the white reference, Stage 3's W)
# carries it; the λ/5500 Å scale is a constant and cancels in all ratios.
_PHOTON_W   = _WL_GRID / 5500.0
# A Stage 1 (scalar) residual RMS above this means the calibrators do not
# follow the chosen curves: ~0.02 with the right ones on real data, 0.16 for
# an OSC with an IR cut described by UV-cut-only curves (synthetic).
SSSC_CURVE_MISMATCH_RMS = 0.08

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
_SK_AUTO_SIG = "SSSC/AutoSigma"
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
        Per-channel control point values from the Stage 3 solve.
        Keys: "R", "G", "B" — each a dict with "wl" and "vals" arrays and,
        since 2026-11-28, "cov" (their covariance) and "solver". Kept in the
        session history for the multi-session Stage 4.
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
        meta: dict | None = None,
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
        # What the run was on (since 2026-11-28): filters, camera label, white
        # reference, image name, and the header's DATE-OBS / AIRMASS /
        # altitude, the observations the multi-night Stage 4 will need.
        self.meta: dict = dict(meta or {})

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
            # Everything the solve stored, not just wl/vals: the covariance is
            # what lets Stage 4 weight one session against another (it was
            # dropped here until 2026-11-28).
            d["ctrl_points"] = {}
            for ch, v in self.ctrl_points.items():
                e = {"wl": np.asarray(v["wl"]).tolist(), "vals": np.asarray(v["vals"]).tolist()}
                for k in ("cov", "n_ctrl", "solver"):
                    if v.get(k) is not None:
                        e[k] = np.asarray(v[k]).tolist() if k == "cov" else v[k]
                d["ctrl_points"][ch] = e
        if self.meta:
            d["meta"] = self.meta
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
                ctrl_points = {}
                for ch, v in cp_raw.items():
                    e = {"wl": np.array(v["wl"]), "vals": np.array(v["vals"])}
                    if v.get("cov") is not None:
                        e["cov"] = np.array(v["cov"], dtype=np.float64)
                    for k in ("n_ctrl", "solver"):
                        if v.get(k) is not None:
                            e[k] = v[k]
                    ctrl_points[ch] = e
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
            meta         = d.get("meta") or {},
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

    Stage 3 is a bounded linear least-squares solve (_solve_stage3_response),
    exact on this run's stars. `prior_response` is accepted for the session
    history but no longer seeds anything: a convex solve has one answer
    whatever it starts from. Each session's control points are stored with
    their covariance so the multi-session Stage 4 can combine sessions.
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
        # Stage 1 says measured = k x expected, so its prediction of the
        # expected ratio is m / k. (It was k x m until 2026-11-28, a residual
        # of about k^2 - 1: the "Stage 1 RMS 0.31" on ngc1333 was mostly k.)
        _r1_R = _rms_frac(mrg / k_R, erg)
        _r1_B = _rms_frac(mbg / k_B, ebg)
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

        # Final Stage 2 RMS on the sigma-clipped population, the same stars and
        # formula as Stage 1's. Over every star a handful of blends, variables
        # or saturated cores set the number: 0.18 on ngc1333 while half the
        # stars sat within ±2% (2026-11-28). Clipped, it measures the fit.
        # If the model is degenerate (non-finite or astronomically large),
        # fall back to the Stage 1 RMS so the display is sensible.
        # The residual is the model's predicted expected ratio against the
        # expected ratio, p(m)/e - 1, as in the model selection above. (It was
        # m/p(m) - 1, measured against expected, about k - 1: 0.17 on ngc1333.)
        pred2_RG = _poly2(coeff_R2, meas_RG)
        pred2_BG = _poly2(coeff_B2, meas_BG)
        r2_RG = (pred2_RG / np.where(exp_RG > eps, exp_RG, eps)) - 1.0
        r2_BG = (pred2_BG / np.where(exp_BG > eps, exp_BG, eps)) - 1.0
        residual_rms_raw = float(np.sqrt(np.mean(r2_RG[keep2]**2 + r2_BG[keep2]**2) / 2.0))

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

        # The stars Stage 2's 3σ clip kept: an outlier (blend, variable,
        # saturated core) would otherwise bend R(λ) to fit itself.
        xp_indices  = np.array([i for i, h in enumerate(have_flux) if h and keep2[i]],
                               dtype=np.intp)
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
                fl = np.where(fl > 0, fl, 0.0) * _PHOTON_W      # photon counts
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
            # Measured color ratios (XP subset) — per-star flux/zero-point cancels,
            # exactly as Stage 1/2. This is the space we optimise AND report in, so
            # Stage 3 RMS becomes directly comparable to Stage 2 RMS.
            _Gm_safe_s3 = np.where(Gm_xp_n > eps, Gm_xp_n, eps)
            meas_RG_s3  = Rm_xp_n / _Gm_safe_s3
            meas_BG_s3  = Bm_xp_n / _Gm_safe_s3

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

            # Control points sit on each channel's main lobe: the contiguous
            # run around the peak where T >= threshold x peak. Green's 0.707
            # keeps them off an OSC Bayer green's secondary red lobe; R and B
            # use 0.30, which covers the passband of steep astro filters.
            PASSBAND_THRESH_PER_CH = {"R": 0.30, "G": 0.707, "B": 0.30}

            # The integral itself runs over the WHOLE curve, as the camera's
            # does: beyond the outer control points R(λ) holds their values
            # (np.interp clamps) rather than dropping to zero. Integrating
            # only inside the threshold left out the flux the camera records
            # in the filter's shoulders, which the solver then pushed into
            # R(λ) (2026-11-28).
            _wq = np.full(len(_WL_GRID), float(_WL_GRID[1] - _WL_GRID[0]))
            _wq[0] *= 0.5
            _wq[-1] *= 0.5                       # trapezoid weights, uniform grid
            F_xp = np.vstack(fl_xp_norm)          # (stars, wavelengths)

            ch_data = {}
            for ch_name, T_c, meas_n, k_c_init in channels_cfg:
                T_peak = float(np.max(T_c))
                if T_peak < eps:
                    ch_data[ch_name] = None
                    continue
                thr = PASSBAND_THRESH_PER_CH[ch_name] * T_peak
                i_lo = i_hi = int(np.argmax(T_c))
                while i_lo > 0 and T_c[i_lo - 1] >= thr:
                    i_lo -= 1
                while i_hi < len(T_c) - 1 and T_c[i_hi + 1] >= thr:
                    i_hi += 1
                ctrl_wl = np.linspace(_WL_GRID[i_lo], _WL_GRID[i_hi], N_CTRL)
                pb = T_c > 1e-3 * T_peak          # the curve's support
                pb_wl = _WL_GRID[pb]
                T_pb  = T_c[pb]

                # Basis: R(λ) = Σ r_j φ_j(λ), φ_j = np.interp of a unit vector
                # (hats inside, flat beyond the ends).
                eye = np.eye(N_CTRL)
                Phi = np.stack([np.interp(_WL_GRID, ctrl_wl, eye[j]) for j in range(N_CTRL)], axis=1)
                W = (F_xp * (T_c * _wq)) @ Phi    # W_ij = ∫ f_i T_c φ_j dλ

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

        if stage >= 3 and len(active_channels) < 3:
            status_cb("[SSSC] Stage 3 needs all three channels — keeping Stage 2")
            stage = 2

        if stage >= 3:
            if prior_response is not None and prior_response.stage >= 3:
                status_cb("[SSSC] Stage 3 is solved exactly on this run's stars; earlier "
                          "sessions stay in the history (they no longer seed it).")
            status_cb(f"[SSSC] Stage 3: solving R(λ) — {N_CTRL} control points per "
                      f"channel, {n_xp_s3} stars…")
            W_ch = {cn: cd["W"] for cn, cd in active_channels}
            m_ch = {cn: cd["meas_n"] for cn, cd in active_channels}
            a_ch, _grey, keep3, cov3 = _solve_stage3_response(W_ch, m_ch)
            # One overall level is free: mean G control value = 1. The channels'
            # relative levels are kept (they are the camera's channel balance).
            lev = float(np.mean(a_ch["G"]))
            lev = lev if lev > eps else 1.0
            a_ch = {c: v / lev for c, v in a_ch.items()}
            cov3 = {c: v / (lev * lev) for c, v in cov3.items()}

            # Reconstruct per-channel R(λ) on full grid
            R_channels = {}
            for ch_name, cd in active_channels:
                R_full = np.zeros(len(_WL_GRID), dtype=np.float64)
                R_full[cd["pb"]] = np.interp(
                    cd["pb_wl"], cd["ctrl_wl"], a_ch[ch_name])   # flat beyond the ends, as in W
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
            response    = response_iter
            poly_coeffs = None

            # Per-channel control points and their covariance, for the session
            # history (and the multi-session Stage 4, which combines sessions by
            # weighting each one's shape with its covariance).
            ctrl_points = {}
            for ch_name, cd in active_channels:
                ctrl_points[ch_name] = {
                    "wl":     cd["ctrl_wl"].copy(),
                    "vals":   np.asarray(a_ch[ch_name], dtype=np.float64).copy(),
                    "cov":    np.asarray(cov3[ch_name], dtype=np.float64).tolist(),
                    "n_ctrl": int(N_CTRL),
                    "solver": "linear-v2",
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

            # ── Final RMS — colour-ratio space, the stars the solve kept ──────
            I_f = {c: W_ch[c] @ a_ch[c] for c in ("R", "G", "B")}
            IG_f = np.where(I_f["G"] > eps, I_f["G"], eps)
            rRG_f = meas_RG_s3 / np.maximum(I_f["R"] / IG_f, eps) - 1.0
            rBG_f = meas_BG_s3 / np.maximum(I_f["B"] / IG_f, eps) - 1.0
            residual_rms = float(np.sqrt(np.mean(
                np.concatenate([rRG_f[keep3], rBG_f[keep3]]) ** 2)))
            stage_rms[3] = residual_rms

            # ── Guard ─────────────────────────────────────────────────────────
            # Stage 3 is a physical model (one R(λ) per channel for every star),
            # Stage 2 a free quadratic fitted straight to the ratios, so Stage 3
            # can sit a little above Stage 2's ratio RMS and still be right
            # (synthetic: 0.0211 against 0.0209 with R(λ) recovered to ~1.5%).
            # Far above it means the solve is wrong (bad curves, a pathological
            # field): keep Stage 2 then.
            _s2 = stage_rms.get(2)
            if _s2 is not None and residual_rms > 1.25 * _s2 + 1e-6:
                status_cb(
                    f"[SSSC] Stage 3 RMS={residual_rms:.4f} > 1.25 x "
                    f"Stage 2 RMS={_s2:.4f} — reverting to Stage 2"
                )
                stage            = 2
                residual_rms     = _s2
                response         = np.ones_like(_WL_GRID, dtype=np.float64)
                ctrl_points      = None
                ch_response_norm = None

            if stage >= 3:
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

    def runs(self, session_id: str) -> list[tuple[int, SystemResponse]]:
        """Every saved run of a session, oldest first, with its row id."""
        cur = self._conn.execute(
            f"""SELECT id, payload FROM {_SESSION_TABLE}
                WHERE session_id = ? ORDER BY timestamp ASC""",
            (session_id,)
        )
        out = []
        for rid, payload in cur.fetchall():
            try:
                out.append((int(rid), SystemResponse.from_dict(json.loads(payload))))
            except Exception:
                pass
        return out

    def sessions(self) -> list[dict]:
        """Every session in the cache: id, run count, first/last run time and
        the latest run's metadata (filters, camera label), newest first."""
        cur = self._conn.execute(
            f"""SELECT session_id, COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM {_SESSION_TABLE} GROUP BY session_id ORDER BY MAX(timestamp) DESC"""
        )
        out = []
        for sid, n, first, last in cur.fetchall():
            meta = {}
            try:
                row = self._conn.execute(
                    f"""SELECT payload FROM {_SESSION_TABLE} WHERE session_id = ?
                        ORDER BY timestamp DESC LIMIT 1""", (sid,)).fetchone()
                meta = (json.loads(row[0]).get("meta") or {}) if row else {}
            except Exception:
                meta = {}
            out.append({"session_id": sid, "n": int(n), "first": first, "last": last, "meta": meta})
        return out

    def delete_run(self, row_id: int):
        self._conn.execute(f"DELETE FROM {_SESSION_TABLE} WHERE id = ?", (int(row_id),))
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Apply correction
# ══════════════════════════════════════════════════════════════════════════════

def _diff2_matrix(n_ctrl: int) -> np.ndarray:
    D = np.zeros((max(n_ctrl - 2, 0), n_ctrl), dtype=np.float64)
    for j in range(n_ctrl - 2):
        D[j, j:j + 3] = (1.0, -2.0, 1.0)
    return D


def _solve_stage3_response(W: dict, m: dict, *, smooth: float = 0.05,
                           iters: int = 6, clip: float = 3.0):
    """Stage 3: each channel's response control values, solved exactly.

    Model, per star i and channel c (spectra and measurements both divided by
    the star's G integral, so they are dimensionless):

        m_c,i  =  s_i · (W_c a_c)_i ,     a_c = k_c · r_c  (gain folded in)

    s_i is a per-star grey factor (aperture loss, the XP flux zero point,
    passing cloud: anything the same in all three channels). Given s, each
    channel is a bounded linear least-squares problem whose coefficients are
    exact (spectrum x filter x basis) and whose noise sits only on the
    right-hand side; given a, s_i is the mean over channels of m/(W a).
    Rows are weighted by the model (iteratively), so a residual is a
    fractional error; a mild second-difference penalty keeps R(λ) smooth;
    stars beyond `clip` sigma (MAD) are dropped each pass.

    This replaced (2026-11-28) a coordinate descent in colour-ratio space,
    which (1) moved control values in ±10-20% steps, too coarse to follow
    the ~1% colour signal, so R(λ) stayed at its flat start, and (2) put the
    noisy measured ratios into the coefficients, which biases the shape
    (regression dilution) by an amount more stars do not reduce. On real
    Gaia XP spectra with a known synthetic response (1.5% noise, 2% grey
    per star) this solve recovers R(λ) to about 1-3% from 87 stars and
    ~1% from 2000; the old one stayed at the flat curve's 3-10% error.

    Returns (a, s, keep, cov): a and cov per channel, s and keep per star.
    """
    from scipy.optimize import lsq_linear
    chans = ("R", "G", "B")
    n, N = W["G"].shape
    D = _diff2_matrix(N)
    s = np.ones(n, dtype=np.float64)
    a = {}
    for c in chans:
        tot = W[c].sum(axis=1)
        good = tot > 0
        a[c] = np.full(N, float(np.median(m[c][good] / tot[good])) if good.any() else 1.0)
    keep = np.ones(n, dtype=bool)
    cov = {c: np.zeros((N, N)) for c in chans}
    for it in range(iters):
        for c in chans:
            pred = np.maximum(s * (W[c] @ a[c]), 1e-30)
            A = (s / pred)[:, None] * W[c]
            b = m[c] / pred
            A, b = A[keep], b[keep]
            lev = max(float(np.mean(a[c])), 1e-30)
            reg = D * (smooth * np.sqrt(max(int(keep.sum()), 1)) / lev)
            Ar = np.vstack([A, reg])
            br = np.concatenate([b, np.zeros(D.shape[0])])
            a[c] = lsq_linear(Ar, br, bounds=(0.0, np.inf), method="bvls").x
            if it == iters - 1:
                res = A @ a[c] - b
                dof = max(len(b) - N, 1)
                try:
                    cov[c] = float(res @ res / dof) * np.linalg.pinv(Ar.T @ Ar)
                except Exception:
                    cov[c] = np.zeros((N, N))
        q = np.stack([m[c] / np.maximum(W[c] @ a[c], 1e-30) for c in chans], axis=1)
        s = np.median(q, axis=1) if it == 0 else np.mean(q, axis=1)
        s_med = float(np.median(s[keep])) if keep.any() else 1.0
        s = s / (s_med if s_med > 0 else 1.0)
        r = np.stack([m[c] / np.maximum(s * (W[c] @ a[c]), 1e-30) - 1.0 for c in chans], axis=1)
        rr = np.sqrt(np.mean(r ** 2, axis=1))
        med = float(np.median(rr))
        mad = 1.4826 * float(np.median(np.abs(rr - med)))
        if mad > 0:
            keep = rr < med + clip * mad
    return a, s, keep, cov


def _apply_quadratic_gains(calibrated: np.ndarray, gains, eps: float,
                           white_ref=(1.0, 1.0)) -> None:
    """The Stage 2 colour-dependent gains, in place on R and B.

    The quadratic maps a measured band/G ratio to the star's expected ratio;
    dividing by the white reference's expected ratio makes that star come out
    neutral (see apply_sssc_correction). R and B are independent and numpy
    releases the GIL, so they run side by side.
    """
    G_safe = np.maximum(calibrated[..., 1], eps)

    def _scaled(c, a, b, cc, ref):
        X = calibrated[..., c]
        ratio = X / G_safe
        m = np.clip(a * ratio**2 + b * ratio + cc, 0.25, 4.0)
        return _pivot_scale_channel(X, m / np.maximum(ratio, eps) / ref, float(np.median(X)))

    ref_R, ref_B = float(white_ref[0]), float(white_ref[1])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_R = pool.submit(_scaled, 0, gains[3], gains[4], gains[5], ref_R)
        fut_B = pool.submit(_scaled, 2, gains[6], gains[7], gains[8], ref_B)
        new_R, new_B = fut_R.result(), fut_B.result()
    calibrated[..., 0] = new_R
    calibrated[..., 2] = new_B


def apply_sssc_correction(
    img_float: np.ndarray,
    sr: SystemResponse,
    enriched: list[dict],
    wl_grid: np.ndarray,
    T_sys_R: np.ndarray,
    T_sys_G: np.ndarray,
    T_sys_B: np.ndarray,
    *,
    white_ref=(1.0, 1.0),
    status_cb=None,
) -> np.ndarray:
    """
    Apply the solved system response correction to the image.

    The solver maps a star's measured band/G ratio to its expected ratio
    through the filters. Expected ratios are not colours: through the Antlia
    R/G/B set a G2V star's expected (R/G, B/G) is about (0.85, 1.12). The
    corrected ratio is therefore the expected ratio divided by the white
    reference's expected ratio, `white_ref` = (R/G, B/G) of the white
    reference SED through the same curves, so that the white reference comes
    out R = G = B. Without that division (before 2026-11-28) a G2V star came
    out R/G 0.91, B/G 1.13 on ngc1333: purple. Per-band scale factors the
    solver cannot know (QE, exposure, any curve scale) cancel in the
    division.

    Stage 1 applies the scalar gains, Stage 2 and 3 the Stage 2 quadratic
    (Stage 3's R(λ) is the measurement: the diagnostics and the session
    history).
    """
    if status_cb is None:
        status_cb = lambda m: None

    eps = 1e-8
    k_R = float(sr.gains[0])
    k_G = float(sr.gains[1])
    k_B = float(sr.gains[2])
    ref_R, ref_B = float(white_ref[0]), float(white_ref[1])
    if not (np.isfinite(ref_R) and ref_R > 0 and np.isfinite(ref_B) and ref_B > 0):
        ref_R = ref_B = 1.0

    if sr.stage <= 2:
        status_cb(f"[SSSC] Applying Stage {sr.stage} correction…")

        calibrated = img_float.copy()
        R = calibrated[..., 0]
        B = calibrated[..., 2]

        if sr.stage == 2 and len(sr.gains) == 9:
            _apply_quadratic_gains(calibrated, sr.gains, eps, (ref_R, ref_B))
        else:
            scale_R = float(np.clip(1.0 / max(k_R * ref_R, eps), 0.25, 4.0))
            scale_B = float(np.clip(1.0 / max(k_B * ref_B, eps), 0.25, 4.0))
            calibrated[..., 0] = _pivot_scale_channel(R, scale_R, float(np.median(R)))
            calibrated[..., 2] = _pivot_scale_channel(B, scale_B, float(np.median(B)))

        return np.clip(calibrated, 0.0, 1.0).astype(np.float32)

    else:
        # Stage 3: apply via Stage 2 quadratic gains (well-conditioned)
        status_cb("[SSSC] Applying Stage 3 correction (via Stage 2 gains)…")

        calibrated = img_float.copy()
        R = calibrated[..., 0]
        B = calibrated[..., 2]

        if len(sr.gains) == 9:
            _apply_quadratic_gains(calibrated, sr.gains, eps, (ref_R, ref_B))
        else:
            scale_R = float(np.clip(1.0 / max(k_R * ref_R, eps), 0.25, 4.0))
            scale_B = float(np.clip(1.0 / max(k_B * ref_B, eps), 0.25, 4.0))
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
    star_summary: tuple | None = None,
) -> Figure:
    """
    Build the SSSC diagnostic figure into an existing matplotlib Figure.

    Four panels:
      1. Solved R(λ) vs manufacturer QE
      2. Before/After residual scatter in (R/G, B/G) ratio space
      3. Bootstrap stage indicator and star count history
      4. B-V color distribution of calibrator population

    star_summary = (star_list, templates_for_hist) adds a third column: the
    spectral-type distribution and the calibration source breakdown, so a
    one-click run shows every chart at once.
    """
    _force_mpl_no_tex()
    fig = figure
    fig.clf()

    eps = 1e-12
    ncols = 3 if star_summary is not None else 2

    def _cell(k: int) -> int:
        """Panel k (1-4, row by row in 2x2) as a subplot index in the grid."""
        return k if ncols == 2 else (k + (1 if k > 2 else 0))

    # ── Panel 1: Per-channel solved R(λ) ─────────────────────────────────────
    ax1 = fig.add_subplot(2, ncols, _cell(1))
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
    ax2 = fig.add_subplot(2, ncols, _cell(2))

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
        # Relative guards: Gaia XP fluxes are W/m^2/nm, so a star's expected
        # integral is ~1e-15. The absolute 1e-12 used here hid all but the
        # brightest few stars (2026-11-28).
        if not (Gm > 0 and Sg > 0 and np.isfinite(Rm / Gm) and np.isfinite(Sr / Sg)):
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

        res_RG_before = (meas_RG / np.where(exp_RG > 0, exp_RG, np.inf)) - 1.0
        res_BG_before = (meas_BG / np.where(exp_BG > 0, exp_BG, np.inf)) - 1.0
        # "After" is what the applied correction predicts: the Stage 2
        # quadratic when it ran (it drives the pixels for Stage 2 and 3),
        # else the Stage 1 scalar m/k.
        g = np.asarray(sr.gains, dtype=np.float64)
        if len(g) == 9:
            pred_RG = g[3] * meas_RG**2 + g[4] * meas_RG + g[5]
            pred_BG = g[6] * meas_BG**2 + g[7] * meas_BG + g[8]
        else:
            pred_RG = meas_RG / (k_R / k_G)
            pred_BG = meas_BG / (k_B / k_G)
        res_RG_after  = (pred_RG / np.where(exp_RG > 0, exp_RG, np.inf)) - 1.0
        res_BG_after  = (pred_BG / np.where(exp_BG > 0, exp_BG, np.inf)) - 1.0

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
    ax3 = fig.add_subplot(2, ncols, _cell(3))

    # Collect per-stage RMS from sr.stage_rms (may be missing for old cached sessions)
    s_rms = getattr(sr, "stage_rms", {sr.stage: sr.residual_rms})

    wf_rows = [
        (1, "Stage 1\nScalar gains",   "#888888", _STAGE1_MIN),
        (2, "Stage 2\nColor model",    "#5588cc", _STAGE2_MIN),
        (3, "Stage 3\nFull R(\u03bb)",      "#44cc88", _STAGE3_MIN),
        (4, "Stage 4\nAtmosphere",     "#cc8844", _STAGE4_MIN),
    ]

    # RMS axis: scale to the worst RMS that ran, with room for the labels
    ran_rms = [v for k, v in s_rms.items() if v > 0]
    rms_max = max(ran_rms) * 1.9 if ran_rms else 1.0

    for i, (stg, lbl, col, thresh) in enumerate(wf_rows):
        rms_val = s_rms.get(stg)
        did_run = rms_val is not None
        bar_color = col if did_run else "#2a2a2a"
        bar_alpha = 0.85 if did_run else 0.25

        if did_run:
            ax3.barh(i, rms_val, color=bar_color, alpha=bar_alpha,
                     height=0.55, zorder=3)
            # RMS value after the bar (the suptitle carries the S1 -> S3 steps)
            ax3.text(rms_val + rms_max * 0.01, i, f"{rms_val:.4f}",
                     va="center", ha="left", fontsize=8,
                     color=col, fontweight="bold", zorder=4)
        else:
            # Stage not yet reached — show as a faint placeholder
            ax3.barh(i, rms_max * 0.15, color=bar_color, alpha=bar_alpha,
                     height=0.55, zorder=2)
            stars_needed = max(0, thresh - sr.n_stars)
            if stg == 4:
                note = "DR4 + multi-night"
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

    # ── Panel 4: B-V distribution ─────────────────────────────────────────────
    ax4 = fig.add_subplot(2, ncols, _cell(4))
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
    if star_summary is not None:
        _draw_star_summary(fig.add_subplot(2, 3, 3), fig.add_subplot(2, 3, 6),
                           star_summary[0], star_summary[1])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Session history: what each filter + camera combination has built up
# ══════════════════════════════════════════════════════════════════════════════

def _filter_set_text(names) -> str:
    """Three channel curve names, short: "ANTLIA_V_PRO R/G/B" when they share
    a prefix, else the names joined with " / "."""
    names = [str(n) for n in (names or []) if n and n != "(None)"]
    if not names:
        return "—"
    if len(names) == 3:
        pre = os.path.commonprefix(names)
        cut = max(pre.rfind("_"), pre.rfind(" "), pre.rfind("-"))
        if cut > 0:
            pre = pre[:cut + 1]
            tails = [n[len(pre):] for n in names]
            if all(tails) and all(len(t) <= 12 for t in tails):
                return pre.rstrip("_- ") + " " + "/".join(tails)
    return " / ".join(names)


def _parse_when(s) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "").split("+")[0][:26])
    except ValueError:
        return None


def _run_when(sr) -> datetime | None:
    """A run's observation time from its header (DATE-AVG/DATE-OBS), else
    when it was calibrated."""
    return _parse_when((sr.meta or {}).get("date_obs")) or _parse_when(sr.timestamp)


def _count_nights(runs) -> int:
    """Distinct observing nights among runs whose header carried a date
    (a night runs noon to noon)."""
    from datetime import timedelta
    nights = set()
    for sr in runs:
        t = _parse_when((sr.meta or {}).get("date_obs"))
        if t is not None:
            nights.add((t - timedelta(hours=12)).date())
    return len(nights)


def _run_rms(sr) -> float:
    try:
        return float((sr.stage_rms or {}).get(sr.stage, sr.residual_rms))
    except Exception:
        return float(sr.residual_rms)


def _runs_text(tr, n: int) -> str:
    return tr("1 run") if n == 1 else tr("{0} runs").format(n)


def _nights_text(tr, n: int) -> str:
    if n <= 0:
        return tr("nights unknown (no DATE-OBS)")
    return tr("1 night") if n == 1 else tr("{0} nights").format(n)


def _history_line(tr, runs) -> str:
    best = max(runs, key=lambda r: (r.stage, r.n_stars))
    return tr("{0}, {1}  ·  best: Stage {2} from {3} stars").format(
        _runs_text(tr, len(runs)), _nights_text(tr, _count_nights(runs)),
        best.stage, best.n_stars)


class SSSCHistoryDialog(QDialog):
    """Everything a filter + camera combination has built up: every run, the
    R(λ) control points Stage 3 measured (each with its uncertainty, and
    their weighted mean), and the channel gains run by run. One bad run can
    be deleted without clearing the rest."""

    def __init__(self, parent, cache: "SessionResponseCache", session_id: str):
        super().__init__(parent)
        _force_mpl_no_tex()
        self.setWindowTitle(self.tr("SSSC Session History"))
        self.resize(1180, 780)
        self._cache = cache
        self._runs: list = []
        self.changed = False
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(self.tr("Session:")))
        self.session_combo = QComboBox()
        top.addWidget(self.session_combo, 1)
        lay.addLayout(top)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.info)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            self.tr("Date"), self.tr("Image"), self.tr("Stage"), self.tr("Stars"),
            self.tr("B−V range"), self.tr("RMS"), "k_R", "k_B", self.tr("White ref."),
            self.tr("Airmass")])
        hdr_item = self.table.horizontalHeaderItem(0)
        if hdr_item is not None:
            hdr_item.setToolTip(self.tr("DATE-AVG / DATE-OBS from the image header, "
                                        "else when the run was calibrated"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(170)
        lay.addWidget(self.table, 2)

        self.figure = Figure(figsize=(11, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(320)
        lay.addWidget(self.canvas, 3)

        btns = QHBoxLayout()
        self.delete_btn = QPushButton(self.tr("Delete Run"))
        self.delete_btn.setToolTip(self.tr("Remove the selected run(s) from this session's history"))
        self.delete_btn.clicked.connect(self._delete_selected)
        btns.addWidget(self.delete_btn)
        btns.addStretch(1)
        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._current = session_id
        self._fill_sessions()
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)
        self._load()

    def _fill_sessions(self, select=None):
        try:
            sessions = self._cache.sessions()
        except Exception as e:
            print(f"[SSSC] History: could not list sessions: {e}")
            sessions = []
        if not any(s["session_id"] == self._current for s in sessions):
            sessions.insert(0, {"session_id": self._current, "n": 0, "meta": {}})
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        for s in sessions:
            m = s.get("meta") or {}
            parts = [s["session_id"]]
            if m.get("filters"):
                parts.append(_filter_set_text(m["filters"]))
            if m.get("camera"):
                parts.append(f"“{m['camera']}”")
            parts.append(_runs_text(self.tr, s["n"]))
            if s["session_id"] == self._current:
                parts.append(self.tr("(current settings)"))
            self.session_combo.addItem("  ·  ".join(parts), s["session_id"])
        i = self.session_combo.findData(select) if select else -1
        self.session_combo.setCurrentIndex(i if i >= 0 else
                                            max(0, self.session_combo.findData(self._current)))
        self.session_combo.blockSignals(False)

    def _on_session_changed(self, _i):
        self._load()

    def _load(self):
        import html as _html
        sid = self.session_combo.currentData()
        try:
            self._runs = self._cache.runs(sid) if sid else []
        except Exception as e:
            print(f"[SSSC] History: could not read session {sid}: {e}")
            self._runs = []
        self._runs.sort(key=lambda t: _run_when(t[1]) or datetime.min)
        runs = [sr for _rid, sr in self._runs]
        if runs:
            m = runs[-1].meta or {}
            head = []
            if m.get("filters"):
                head.append(_html.escape(_filter_set_text(m["filters"])))
            lp = [x for x in (m.get("lp") or []) if x and x != "(None)"]
            if lp:
                head.append(self.tr("LP: {0}").format(_html.escape(" + ".join(lp))))
            if m.get("camera"):
                head.append(f"“{_html.escape(m['camera'])}”")
            line = _html.escape(_history_line(self.tr, runs))
            self.info.setText(("<b>" + "  ·  ".join(head) + "</b><br>" if head else "") + line)
        else:
            self.info.setText(self.tr("No runs saved for this session yet."))
        self.table.setRowCount(len(runs))
        for row, sr in enumerate(runs):
            m = sr.meta or {}
            t = _run_when(sr)
            am = m.get("airmass")
            cells = [
                t.strftime("%Y-%m-%d %H:%M") if t else "—",
                m.get("image") or "—",
                str(sr.stage),
                str(sr.n_stars),
                f"{sr.bv_range[0]:.2f} – {sr.bv_range[1]:.2f}",
                f"{_run_rms(sr):.4f}",
                f"{float(sr.gains[0]):.4f}" if len(sr.gains) > 0 else "—",
                f"{float(sr.gains[2]):.4f}" if len(sr.gains) > 2 else "—",
                m.get("white_reference") or "—",
                f"{am:.2f}" if isinstance(am, (int, float)) else "—",
            ]
            for c, v in enumerate(cells):
                it = QTableWidgetItem(v)
                if c != 1:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, c, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.delete_btn.setEnabled(bool(runs))
        self._plot(runs)

    def _plot(self, runs):
        fig = self.figure
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        colors = {"R": "#d33", "G": "#2a2", "B": "#36d"}

        def _is_v2(sr):
            cp = sr.ctrl_points or {}
            return all((cp.get(c) or {}).get("solver") == "linear-v2" for c in "RGB")

        s3 = [sr for sr in runs if sr.stage >= 3 and _is_v2(sr)]
        n_old = sum(1 for sr in runs if sr.stage >= 3 and not _is_v2(sr))
        if s3:
            for c in "RGB":
                ref_wl = np.asarray(s3[-1].ctrl_points[c]["wl"], dtype=np.float64)
                stack, wts = [], []
                for sr in s3:
                    cp = sr.ctrl_points[c]
                    wl_a = np.asarray(cp["wl"], dtype=np.float64)
                    v = np.asarray(cp["vals"], dtype=np.float64)
                    sig = None
                    if cp.get("cov") is not None:
                        cov = np.asarray(cp["cov"], dtype=np.float64)
                        if cov.ndim == 2 and cov.shape[0] == len(v):
                            sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                    if sig is not None:
                        ax1.errorbar(wl_a / 10.0, v, yerr=sig, color=colors[c], alpha=0.35,
                                     lw=1, marker="o", ms=3, capsize=2)
                    else:
                        ax1.plot(wl_a / 10.0, v, color=colors[c], alpha=0.35, lw=1,
                                 marker="o", ms=3)
                    if len(wl_a) == len(ref_wl) and np.allclose(wl_a, ref_wl, atol=1.0):
                        stack.append(v)
                        wts.append(1.0 / np.maximum(sig, 1e-6) ** 2 if sig is not None
                                   else np.ones_like(v))
                if len(stack) >= 2:
                    S, Wt = np.vstack(stack), np.vstack(wts)
                    mean = (S * Wt).sum(0) / Wt.sum(0)
                    ax1.plot(ref_wl / 10.0, mean, color=colors[c], lw=2.5,
                             label=f"{c}: weighted mean of {len(stack)} runs")
            if len(s3) >= 2:
                ax1.legend(fontsize=7)
            ax1.set_title(f"Stage 3 R(λ) control points, {len(s3)} run(s), ±1σ",
                          fontsize=10)
        else:
            ax1.text(0.5, 0.5, "No Stage 3 runs yet.\nR(λ) appears here once a field reaches\n"
                     "Stage 3 (200+ Gaia XP stars).", ha="center", va="center",
                     transform=ax1.transAxes, fontsize=9, color="0.4")
            ax1.set_title("Stage 3 R(λ)", fontsize=10)
        if n_old:
            ax1.annotate(f"{n_old} Stage 3 run(s) from before 2026-11-28 not shown "
                         "(no uncertainties stored)",
                         (0.01, 0.01), xycoords="axes fraction", fontsize=7, color="0.45")
        ax1.set_xlabel("Wavelength (nm)", fontsize=8)
        ax1.set_ylabel("Relative response (mean G = 1)", fontsize=8)
        ax1.tick_params(labelsize=7)
        ax1.grid(alpha=0.3)
        if runs:
            x = np.arange(1, len(runs) + 1)
            ax2.plot(x, [float(r.gains[0]) for r in runs], "o-", color=colors["R"], label="k_R")
            ax2.plot(x, [float(r.gains[2]) for r in runs], "s-", color=colors["B"], label="k_B")
            ax2.set_xticks(x)
            ax2.legend(fontsize=7)
        ax2.set_title("Channel gains by run (relative to G)", fontsize=10)
        ax2.set_xlabel("Run (by date, as in the table)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(alpha=0.3)
        _relayout_on_draw(fig)
        self.canvas.draw_idle()

    def _delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        ids = [self._runs[r][0] for r in rows if 0 <= r < len(self._runs)]
        if not ids:
            QMessageBox.information(self, self.tr("Delete Run"),
                                    self.tr("Select a run in the table first."))
            return
        reply = QMessageBox.question(
            self, self.tr("Delete Run"),
            self.tr("Delete {0} run(s) from this session's history? "
                    "This cannot be undone.").format(len(ids)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            for rid in ids:
                self._cache.delete_run(rid)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"), str(e))
        self.changed = True
        self._fill_sessions(select=self.session_combo.currentData())
        self._load()


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
        self.setWindowTitle(self.tr("Spectrophotometric Standard Star Calibration (SSSC)"))
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
        # Factory state, before saved settings load. The Camera / Rig label
        # names the session history (a text field: not captured), and the
        # curve library is user data.
        self._factory = snapshot_controls(self)
        self.load_settings()
        self._update_session_info_label()
        self._restore_geometry()
        self.finished.connect(lambda *_: self._cleanup())

        # Curve Manager writes -> refresh the filter combos live.
        self._curve_lib_conn = None
        try:
            from setiastro.saspro.spectral_curves import curve_library_signals
            curve_library_signals().changed.connect(self._on_curve_library_changed)
            self._curve_lib_conn = True
        except Exception as e:
            print(f"[SSSC] curve library hook unavailable: {e}")
 
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
        """One click: the settings on the left, Calibrate under them, and the
        results (hidden until the first run) on the right. Plain widgets, so
        the window follows the theme."""
        outer = QHBoxLayout(self)

        # ── Settings column ───────────────────────────────────────────────────
        self.settings_panel = QWidget()
        col = QVBoxLayout(self.settings_panel)
        col.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.settings_panel, 0)

        chain = QGroupBox(self.tr("Optical Chain"))
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.star_combo = QComboBox()
        self.star_combo.addItem("G2V (Solar)", userData="G2V")
        self.star_combo.addItem("Vega (A0V)",  userData="A0V")
        for sed in self.sed_list:
            if sed.upper() in ("A0V", "G2V"):
                continue
            self.star_combo.addItem(sed, userData=sed)
        self.star_combo.setToolTip(self.tr(
            "The star that should come out white. G2V (the Sun) is the usual choice."))

        def _combo():
            cb = QComboBox()
            cb.addItem("(None)")
            cb.addItems(self.filter_list)
            cb.setMinimumContentsLength(18)
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            return cb

        self.r_filter_combo   = _combo()
        self.g_filter_combo   = _combo()
        self.b_filter_combo   = _combo()
        self.lp_filter_combo  = _combo()
        self.lp_filter_combo2 = _combo()
        for cb in (self.lp_filter_combo, self.lp_filter_combo2):
            cb.setToolTip(self.tr("Light-pollution or UV/IR-cut filters in the light path, if any."))

        form.addRow(self.tr("White Reference:"), self.star_combo)
        form.addRow(self.tr("R Filter:"), self.r_filter_combo)
        form.addRow(self.tr("G Filter:"), self.g_filter_combo)
        form.addRow(self.tr("B Filter:"), self.b_filter_combo)
        form.addRow(self.tr("LP/Cut 1:"), self.lp_filter_combo)
        form.addRow(self.tr("LP/Cut 2:"), self.lp_filter_combo2)
        note = QLabel(self.tr("No sensor curve: SSSC solves the sensor's response from the stars."))
        note.setWordWrap(True)
        note.setEnabled(False)            # the theme's quieter text colour

        tools = QHBoxLayout()
        self.curve_mgr_btn = QPushButton(self.tr("Filter / Sensor Curves…"))
        self.curve_mgr_btn.setToolTip(
            self.tr("Open the Curve Manager: add, label, edit, share or delete filter and sensor curves\n"
            "(paste data, import a CSV, digitize a chart image, or generate a bandpass)."))
        self.curve_mgr_btn.clicked.connect(self.open_curve_manager)
        self.sasp_btn = QPushButton(self.tr("SASP Viewer"))
        self.sasp_btn.setToolTip(
            self.tr("Open the SASP curve viewer (Pickles SEDs + filter/sensor responses)"))
        self.sasp_btn.clicked.connect(self.open_sasp_viewer)
        tools.addWidget(self.curve_mgr_btn)
        tools.addWidget(self.sasp_btn)
        chain_l = QVBoxLayout(chain)
        chain_l.addLayout(form)
        chain_l.addWidget(note)
        chain_l.addLayout(tools)
        col.addWidget(chain)

        # ── Session history: which session these settings select, and what it
        # has built up. Live: follows every filter or camera-name change.
        self._session_info_closed = False
        self._session_info_timer = QTimer(self)
        self._session_info_timer.setSingleShot(True)
        self._session_info_timer.setInterval(200)
        self._session_info_timer.timeout.connect(self._update_session_info_label)
        hist = QGroupBox(self.tr("Session History"))
        hist.setToolTip(self.tr(
            "Every calibration is saved under a session: the R/G/B filters, LP filters and\n"
            "camera / rig name together. Each run is solved on its own stars; the saved\n"
            "R(λ) curves and their uncertainties are the record the multi-night Stage 4\n"
            "will combine."))
        hl = QVBoxLayout(hist)
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel(self.tr("Camera / Rig:")))
        self.camera_label_edit = QLineEdit()
        self.camera_label_edit.setPlaceholderText(
            self.tr("e.g. IMX492 · EdgeHD f/7  (separates session history per sensor)"))
        self.camera_label_edit.setMaxLength(80)
        self.camera_label_edit.textChanged.connect(
            lambda v: QSettings().setValue("SSSC/CameraLabel", v))
        cam_row.addWidget(self.camera_label_edit, 1)
        hl.addLayout(cam_row)
        self.session_info_lbl = QLabel("")
        self.session_info_lbl.setWordWrap(True)
        self.session_info_lbl.setTextFormat(Qt.TextFormat.RichText)
        hl.addWidget(self.session_info_lbl)
        hist_btns = QHBoxLayout()
        self.view_history_btn = QPushButton(self.tr("View History…"))
        self.view_history_btn.setToolTip(self.tr(
            "Every saved run of this session and the others: the R(λ) curves Stage 3\n"
            "measured, the channel gains run by run, and deleting a single bad run"))
        self.view_history_btn.clicked.connect(self._open_session_history)
        hist_btns.addWidget(self.view_history_btn)
        self.clear_session_btn = QPushButton(self.tr("Clear History"))
        self.clear_session_btn.setToolTip(
            self.tr("Delete all saved R(λ) solutions for the current filter+camera combination"))
        self.clear_session_btn.clicked.connect(self._clear_session_history)
        hist_btns.addWidget(self.clear_session_btn)
        hist_btns.addStretch(1)
        hl.addLayout(hist_btns)
        col.addWidget(hist)

        # ── Options, with the rarely touched ones folded away ─────────────────
        opts = QGroupBox(self.tr("Options"))
        ol = QVBoxLayout(opts)
        self.neutralize_chk = QCheckBox(self.tr("Background Neutralization"))
        self.neutralize_chk.setChecked(False)
        self.neutralize_chk.setToolTip(self.tr(
            "Also make the sky background neutral grey after calibrating."))
        ol.addWidget(self.neutralize_chk)

        self.adv_toggle = QToolButton()
        self.adv_toggle.setText(self.tr("Advanced"))
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setChecked(False)
        self.adv_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.adv_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.adv_toggle.setAutoRaise(True)
        ol.addWidget(self.adv_toggle)

        self.adv_panel = QWidget()
        adv = QFormLayout(self.adv_panel)
        adv.setContentsMargins(16, 0, 0, 0)
        sig_row = QHBoxLayout()
        self.sep_thr_spin = QSpinBox()
        self.sep_thr_spin.setRange(2, 100)
        self.sep_thr_spin.setValue(15)
        self.sep_thr_spin.setToolTip(self.tr("Star detection threshold (SEP σ)"))
        self.sep_thr_spin.valueChanged.connect(
            lambda v: QSettings().setValue(_SK_SEP_THR, int(v)))
        sig_row.addWidget(self.sep_thr_spin)
        self.auto_sigma_chk = QCheckBox(self.tr("Auto"))
        self.auto_sigma_chk.setChecked(True)
        self.auto_sigma_chk.setToolTip(
            self.tr("Auto σ waterfall:\n"
            "  Tries σ = 100, 50, 25, 12, 8, 5, 3 in order and stops at the\n"
            "  first value that yields ≥ 1200 detections. Removes the need to\n"
            "  guess a threshold for different fields.\n\n"
            "  Uncheck to use the manual σ value in the spin box.")
        )
        self.auto_sigma_chk.toggled.connect(self._on_auto_sigma_toggled)
        sig_row.addWidget(self.auto_sigma_chk)
        sig_row.addStretch(1)
        self.sep_thr_spin.setEnabled(not self.auto_sigma_chk.isChecked())
        adv.addRow(self.tr("Star σ:"), sig_row)

        self.nctrl_spin = QSpinBox()
        self.nctrl_spin.setRange(4, 24)
        self.nctrl_spin.setValue(8)
        self.nctrl_spin.setSingleStep(2)
        self.nctrl_spin.setToolTip(
            self.tr("Control points per channel for Stage 3 R(λ) solver.\n"
            "8 = default (200–600 stars)\n"
            "12 = finer (~600+ stars)\n"
            "16 = high resolution (~1000+ stars)"))
        self.nctrl_spin.valueChanged.connect(
            lambda v: QSettings().setValue(_SK_N_CTRL, int(v)))
        adv.addRow(self.tr("R(λ) ctrl pts:"), self.nctrl_spin)

        self.max_stars_spin = QSpinBox()
        self.max_stars_spin.setRange(100, 2000)
        self.max_stars_spin.setValue(500)
        self.max_stars_spin.setSingleStep(100)
        self.max_stars_spin.setToolTip(
            self.tr("Maximum calibrator stars for photometry.\n"
            "Stars are selected brightest-first, XP stars prioritized.\n"
            "500 is recommended — dim stars add noise, not accuracy."))
        self.max_stars_spin.valueChanged.connect(
            lambda v: QSettings().setValue("SSSC/MaxStars", int(v)))
        adv.addRow(self.tr("Max stars:"), self.max_stars_spin)
        self.adv_panel.setVisible(False)
        ol.addWidget(self.adv_panel)

        def _toggle_adv(on):
            self.adv_panel.setVisible(bool(on))
            self.adv_toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.adv_toggle.toggled.connect(_toggle_adv)
        col.addWidget(opts)
        col.addStretch(1)

        # ── Status, run, close ────────────────────────────────────────────────
        self.count_label = QLabel(self.tr(
            "Calibrates the colour of the active view from Gaia XP star spectra, "
            "solving the sensor's response instead of assuming it. "
            "It needs a plate solution and offers to make one."))
        self.count_label.setWordWrap(True)
        self.count_label.setMinimumWidth(320)
        col.addWidget(self.count_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setVisible(False)
        col.addWidget(self.progress_bar)

        actions = QHBoxLayout()
        self.factory_reset_btn = make_reset_button(self)
        self.factory_reset_btn.clicked.connect(self._reset_to_factory)
        actions.addWidget(self.factory_reset_btn)
        self.about_btn = QPushButton(self.tr("About"))
        self.about_btn.setToolTip(self.tr("About SSSC — what it is and how it works"))
        self.about_btn.clicked.connect(self._show_about)
        actions.addWidget(self.about_btn)
        actions.addStretch(1)
        self.close_btn = QPushButton(self.tr("Close"))
        self.close_btn.clicked.connect(self.reject)
        actions.addWidget(self.close_btn)
        self.run_btn = QPushButton(self.tr("Calibrate"))
        self.run_btn.setDefault(True)
        self.run_btn.setAutoDefault(True)
        run_tip = self.tr(
            "Find the stars, look up their spectra, measure them, solve the system\n"
            "response and correct the colour of the active view, all in one go.")
        self.run_btn.setToolTip(run_tip)
        self.run_btn.setStyleSheet(_primary_button_sheet())
        self.run_btn.clicked.connect(self.calibrate)
        actions.addWidget(self.run_btn)
        col.addLayout(actions)
        # After a run: a quiet "Calibrated ✓" until something changes.
        self._run_state = _RunButtonState(
            self, self.run_btn, self.tr("Calibrate"), self.tr("Calibrated ✓"), run_tip,
            self.tr("Applied to this image. It turns back into Calibrate when you change a setting,\n"
            "switch to another image or undo."))
        self._run_state.watch(self.star_combo, self.r_filter_combo, self.g_filter_combo,
                              self.b_filter_combo, self.lp_filter_combo, self.lp_filter_combo2,
                              self.camera_label_edit, self.neutralize_chk, self.sep_thr_spin,
                              self.auto_sigma_chk, self.nctrl_spin, self.max_stars_spin)
        self.finished.connect(self._run_state.shutdown)

        # Grip + help: their own row, last in the controls column.
        from setiastro.saspro.shortcuts import ShortcutGrip
        self.btn_help = make_help_button("sssc", self)
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addWidget(ShortcutGrip("sssc", self))
        grip_row.addWidget(self.btn_help, 0, Qt.AlignmentFlag.AlignRight)
        col.addLayout(grip_row)

        # ── Results: summary + every chart, after a run ───────────────────────
        self.results_panel = QWidget()
        rl = QVBoxLayout(self.results_panel)
        rl.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        rl.addWidget(self.summary_label)
        self.figure = Figure(figsize=(14, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(860, 560)
        rl.addWidget(self.canvas, 1)
        self.results_panel.setVisible(False)
        outer.addWidget(self.results_panel, 1)

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
        # The session follows the filters and the camera name.
        for cb in (self.r_filter_combo, self.g_filter_combo, self.b_filter_combo,
                   self.lp_filter_combo, self.lp_filter_combo2):
            cb.currentIndexChanged.connect(self._schedule_session_info)
        self.camera_label_edit.textChanged.connect(self._schedule_session_info)
        self.setMinimumSize(0, 0)     # compact until the first run's results

    def changeEvent(self, ev):
        from setiastro.saspro.widgets.theme_colors import PALETTE_EVENTS, restyle
        if (ev.type() in PALETTE_EVENTS and getattr(self, "run_btn", None) is not None
                and not getattr(getattr(self, "_run_state", None), "done", False)):
            restyle(self.run_btn, _primary_button_sheet())
        if ev.type() in PALETTE_EVENTS:
            self._schedule_session_info()       # its hint lines use theme colours
        super().changeEvent(ev)

    def reject(self):
        if getattr(self, "_running", False):
            return            # a run is pumping events; closing now would pull the dialog out from under it
        super().reject()

    def _set_running(self, on: bool):
        self._running = bool(on)
        for w in (self.run_btn, self.close_btn, self.factory_reset_btn, self.settings_panel):
            w.setEnabled(not on)
        self.progress_bar.setVisible(bool(on))

    def _ensure_plate_solution(self) -> bool:
        """The active view has a celestial WCS, or the user had it solved now."""
        doc = self.doc_manager.get_active_document()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, self.tr("No Image"),
                self.tr("Open an image and make it the active view first."))
            return False
        _img, hdr, _meta = self._get_active_image_and_header()
        try:
            if hdr is not None:
                self.initialize_wcs_from_header(hdr)
                # An unsolved header still makes a WCS, just not a celestial one.
                if getattr(getattr(self, "wcs", None), "has_celestial", False):
                    return True
        except Exception:
            pass
        resp = QMessageBox.question(
            self, self.tr("No Plate Solution"),
            self.tr("This image isn't plate-solved yet. Run the plate solver now?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return False
        _sfcc_status(self, "Plate solving…")
        return self.run_plate_solver(announce=False)

    def calibrate(self):
        """Calibrate: find and look up the stars, measure them, solve the
        response, correct the colour and show every chart. Was Steps 0-2."""
        if getattr(self, "_running", False):
            return
        if all(cb.currentText() == "(None)"
               for cb in (self.r_filter_combo, self.g_filter_combo, self.b_filter_combo)):
            QMessageBox.warning(self, self.tr("Error"), self.tr("Select at least one R/G/B filter."))
            return
        if not self.star_combo.currentData():
            QMessageBox.warning(self, self.tr("Error"), self.tr("Select a white reference SED."))
            return
        doc = self.doc_manager.get_active_document()
        img = getattr(doc, "image", None) if doc is not None else None
        if img is not None and (img.ndim != 3 or img.shape[2] != 3):
            QMessageBox.critical(self, self.tr("Error"), self.tr("Active document must be RGB."))
            return
        if not self._ensure_plate_solution():
            return
        t0 = time.perf_counter()
        self._one_click = True
        self._last_result = None
        self._set_running(True)
        ok = False
        try:
            self.star_list = []
            self.fetch_stars()
            if self.star_list:
                ok = bool(self.run_sssc())
        finally:
            self._one_click = False
            self._set_running(False)
        if ok:
            self._show_results(time.perf_counter() - t0)
            self._run_state.mark_done(self.doc_manager.get_active_document())

    def _show_results(self, seconds: float):
        r = self._last_result or {}
        g = r.get("gains") or [float("nan")] * 3
        self.summary_label.setText(self.tr(
            "<b>Calibrated</b> with {0} stars ({1} Gaia XP spectra, {2} Pickles templates) "
            "in {3:.1f} s.<br>{4} · B-V {5:.2f} – {6:.2f} · residual RMS {7:.4f} · "
            "session runs {8}<br>Gains: k_R={9:.4f}  k_G={10:.4f}  k_B={11:.4f} · {12}"
        ).format(r.get("n_stars", 0), r.get("n_xp", 0), r.get("n_pickles", 0), seconds,
                 r.get("stage_label", ""), r.get("bv_lo", float("nan")), r.get("bv_hi", float("nan")),
                 r.get("rms", float("nan")), r.get("sessions", 0), g[0], g[1], g[2],
                 r.get("stage_note", "")))
        warn = r.get("curve_warning") or ""
        if warn:
            from html import escape
            self.summary_label.setText(self.summary_label.text()
                                       + f'<br><span style="color:#e0a030;">⚠ {escape(warn)}</span>')
        first = not self.results_panel.isVisible()
        self.results_panel.setVisible(True)
        if first:
            _grow_to_fit(self, results_w=1180, min_h=860)
        _force_mpl_no_tex()
        self.canvas.draw_idle()

    def _refresh_filter_combos(self):
        """Repopulate R/G/B/LP filter combos from self.filter_list,
        preserving the current selection and without firing the
        persistence signals."""
        combos = (self.r_filter_combo, self.g_filter_combo, self.b_filter_combo,
                  self.lp_filter_combo, self.lp_filter_combo2)
        for cb in combos:
            prev = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("(None)")
            cb.addItems(self.filter_list)
            idx = cb.findText(prev)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.blockSignals(False)

    def open_sasp_viewer(self):
        from setiastro.saspro.sfcc import SaspViewer
        try:
            if self.sasp_viewer_window is not None:
                try:
                    self.sasp_viewer_window.close()
                except Exception:
                    pass
            self.sasp_viewer_window = SaspViewer(
                self.sasp_data_path, self.user_custom_path)
            self.sasp_viewer_window.show()
            self.sasp_viewer_window.raise_()
        except Exception as e:
            QMessageBox.critical(self, self.tr("SASP Viewer"),
                self.tr("Could not open SASP viewer:\n{0}").format(e))

    def open_curve_manager(self):
        """Filter / Sensor Curve Manager (spectral_curves.py), opened with the
        current filter chain selected so it is plotted straight away."""
        from setiastro.saspro.spectral_curves import open_curve_manager
        chain = [cb.currentText() for cb in (
            self.r_filter_combo, self.g_filter_combo, self.b_filter_combo,
            self.lp_filter_combo, self.lp_filter_combo2)]
        open_curve_manager(parent=self.main_win or self,
                           sasp_data_path=self.sasp_data_path, select=chain)

    def _on_curve_library_changed(self):
        self._reload_hdu_lists()
        self._refresh_filter_combos()

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

        # Auto-σ waterfall preference (default: ON — removes guess-and-check)
        auto_sig = s.value(_SK_AUTO_SIG, True, type=bool)
        if hasattr(self, "auto_sigma_chk"):
            self.auto_sigma_chk.setChecked(bool(auto_sig))
            self.sep_thr_spin.setEnabled(not bool(auto_sig))

        self.max_stars_spin.setValue(int(s.value("SSSC/MaxStars", 500)))

        nctrl = int(s.value(_SK_N_CTRL, 8))
        if hasattr(self, "nctrl_spin"):
            self.nctrl_spin.setValue(nctrl)

        cam_label = s.value("SSSC/CameraLabel", "")
        if hasattr(self, "camera_label_edit"):
            self.camera_label_edit.setText(str(cam_label))

    def _reset_to_factory(self):
        # Every control saves itself on change, so the restore is also saved.
        restore_controls(self._factory)

    def _camera_label(self) -> str:
        try:
            return self.camera_label_edit.text().strip()
        except Exception:
            return ""

    # ── Auto-σ waterfall config / helpers ─────────────────────────────
    # Ladder of σ values tried from highest (strictest) to lowest.
    # Highest σ that still yields ≥ SSSC_AUTO_SIGMA_MIN_STARS wins; if
    # none clear the bar, the last (most sensitive) rung is used so
    # genuinely sparse fields still get something to work with.
    SSSC_AUTO_SIGMA_LADDER = (100.0, 50.0, 25.0, 12.0, 8.0, 5.0, 3.0)
    SSSC_AUTO_SIGMA_MIN_STARS = 1200

    def _on_auto_sigma_toggled(self, on: bool):
        try:
            self.sep_thr_spin.setEnabled(not bool(on))
        except Exception:
            pass
        QSettings().setValue(_SK_AUTO_SIG, bool(on))

    def _auto_sigma_enabled(self) -> bool:
        try:
            return bool(self.auto_sigma_chk.isChecked())
        except Exception:
            return False

    def _sep_detect_waterfall(self, data_sub, err, *, status_prefix: str = ""):
        """
        Run sep.extract with the σ waterfall when Auto is enabled, or with
        the manual spin value when Auto is off.

        Returns (sources, used_sigma). `sources` may be empty if nothing was
        detected at any σ in the ladder — callers should handle that case
        exactly as they handled an empty result before.
        """
        if self._auto_sigma_enabled():
            # Same σ as stepping down the whole ladder, in 2-3 passes.
            sources, sig = sep_detect_auto_sigma(
                data_sub, err, self.SSSC_AUTO_SIGMA_LADDER, self.SSSC_AUTO_SIGMA_MIN_STARS,
                status_cb=(lambda m: _sfcc_status(self, f"{status_prefix} ({m})…"))
                if status_prefix else None)
            n = 0 if sources is None else int(sources.size)
            if status_prefix:
                if n >= self.SSSC_AUTO_SIGMA_MIN_STARS:
                    _sfcc_status(self, f"{status_prefix} — auto σ={sig:.1f} → {n:,} sources.")
                else:
                    _sfcc_status(self, f"{status_prefix} — auto σ ladder exhausted, "
                                       f"using σ={sig:.1f} ({n:,} sources).")
            return sources, float(sig)

        # Manual mode — single shot at the spin value
        sep_sigma = float(self.sep_thr_spin.value()) if hasattr(self, "sep_thr_spin") else 5.0
        if status_prefix:
            _sfcc_status(self, f"{status_prefix} (manual σ={sep_sigma:.1f})…")
            QApplication.processEvents()
        _seed_sep_rng()          # same stars every run (see sfcc._seed_sep_rng)
        return sep.extract(data_sub, sep_sigma, err=err), sep_sigma

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
        self._wcs_fail_reason = None
        if header is None:
            self._wcs_fail_reason = "No FITS header available for the active image."
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
                ct1 = str(hdr.get("CTYPE1", "")).strip() or "(missing)"
                ct2 = str(hdr.get("CTYPE2", "")).strip() or "(missing)"
                self._wcs_fail_reason = (
                    "Header has no celestial axes — it doesn't contain a valid "
                    "RA/DEC plate solution.\n\n"
                    f"CTYPE1 = {ct1}\nCTYPE2 = {ct2}\n\n"
                    "The image is most likely not plate-solved yet."
                )
                print("[SSSC] " + self._wcs_fail_reason.replace("\n", " "))
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
            self._wcs_fail_reason = (
                "astropy could not parse the WCS keywords in this header "
                f"(likely malformed CD matrix, SIP, or CRVAL/CRPIX):\n\n{e}"
            )
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
    def run_plate_solver(self, announce: bool = True) -> bool:
        """Step 0: plate-solve the active document in place so Step 1 has a WCS."""
        from setiastro.saspro.plate_solver import plate_solve_doc_inplace

        doc = self.doc_manager.get_active_document()
        if doc is None or getattr(doc, "image", None) is None:
            QMessageBox.warning(self, self.tr("No Image"),
                self.tr("Open an image and make it the active view first."))
            return False

        try:
            ok, res = plate_solve_doc_inplace(self.main_win or self, doc, QSettings())
        except Exception as e:
            QMessageBox.critical(self, self.tr("Plate Solve Error"), str(e))
            return False

        if not ok:
            QMessageBox.critical(self, self.tr("Plate Solve Failed"), str(res))
            return False

        # Refresh SSSC's cached image/header/WCS from the freshly-solved doc.
        img, hdr, _meta = self._get_active_image_and_header()
        self.current_image  = img
        self.current_header = hdr
        self.initialize_wcs_from_header(hdr)

        if announce:
            QMessageBox.information(self, self.tr("Plate Solve"),
                self.tr("Solved. You can now run Step 1: Fetch Stars & Spectra."))
        return True


    def fetch_stars(self):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astroquery.simbad import Simbad
        from astropy.io import fits as _fits

        img, hdr, _meta = self._get_active_image_and_header()
        self.current_image  = img
        self.current_header = hdr

        if self.current_image is None:
            QMessageBox.warning(self, self.tr("No Image"),
                self.tr("Open an image and make it the active view first."))
            return

        if self.current_header is None:
            resp = QMessageBox.question(
                self, self.tr("No Plate Solution"),
                self.tr("This image isn't plate-solved yet. Run the plate solver now?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
            if not self.run_plate_solver(announce=False):
                return
            # current_image / current_header refreshed inside run_plate_solver

        try:
            self.initialize_wcs_from_header(self.current_header)
        except Exception as e:
            QMessageBox.critical(self, self.tr("WCS Error"),
                self.tr("Could not build a 2D WCS from header.\n\n{0}").format(
                    getattr(self, '_wcs_fail_reason', None) or e))
            return

        if not getattr(self, "wcs", None):
            QMessageBox.critical(self, self.tr("WCS Error"),
                self.tr("Could not build a 2D WCS from header.\n\n{0}").format(
                    getattr(self, "_wcs_fail_reason", None)
                    or self.tr("The header did not yield a usable 2D celestial WCS.")))
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
        gray     = _gray_for_sep(base)
        bkg      = sep.Background(gray)
        data_sub = gray - bkg.back()
        err      = float(bkg.globalrms)
        self._det_sources = None           # set below once this image's stars are found
        self._det_image = None

        sources, sep_sigma = self._sep_detect_waterfall(
            data_sub, err, status_prefix="Detecting stars with SEP"
        )
        del base, data_sub

        if sources is None or sources.size == 0:
            QMessageBox.critical(self, self.tr("SEP Error"), self.tr("SEP found no sources."))
            return

        r_fluxrad, _ = sep.flux_radius(
            gray, sources["x"], sources["y"],
            2.0 * sources["a"], 0.5,
            normflux=sources["flux"], subpix=5)
        del gray
        mask    = (r_fluxrad > 0.2) & (r_fluxrad <= 10)
        sources = sources[mask]
        if sources.size == 0:
            QMessageBox.critical(self, self.tr("SEP Error"),
                self.tr("All SEP detections rejected by radius filter."))
            return

        # SIMBAD is optional enrichment (Gaia XP is the primary source). It
        # runs on its own thread from here, while the Gaia matching below
        # goes on; started after detection because sep.extract holds the GIL.
        pix_corners = np.array([[W/2, H/2], [0,0], [W,0], [0,H], [W,H]], dtype=float)
        try:
            sky_corners = wcs2.all_pix2world(pix_corners, 0)
        except Exception as e:
            QMessageBox.critical(self, self.tr("WCS Error"), str(e))
            return
        center_sky  = SkyCoord(ra=float(sky_corners[0,0])*u.deg,
                               dec=float(sky_corners[0,1])*u.deg, frame="icrs")
        corners_sky = SkyCoord(ra=sky_corners[1:,0]*u.deg,
                               dec=sky_corners[1:,1]*u.deg, frame="icrs")
        radius = center_sky.separation(corners_sky).max() * 1.05
        simbad_job = _SimbadPrefetch(center_sky, radius)

        _sfcc_status(self,
            f"SEP detected {sources.size:,} stars (σ={sep_sigma:.1f}) — converting to sky coords…")
        QApplication.processEvents()

        xs = sources["x"].astype(np.float64)
        ys = sources["y"].astype(np.float64)

        try:
            sky_coords = wcs2.all_pix2world(np.column_stack([xs, ys]), 0)
        except Exception as e:
            QMessageBox.critical(self, self.tr("WCS Error"), str(e))
            return

        valid_mask = (
            np.isfinite(sky_coords[:, 0]) & np.isfinite(sky_coords[:, 1]) &
            (sky_coords[:, 1] >= -90) & (sky_coords[:, 1] <= 90)
        )
        sources    = sources[valid_mask]
        sky_coords = sky_coords[valid_mask]

        if sources.size == 0:
            QMessageBox.critical(self, self.tr("WCS Error"),
                self.tr("No valid sky coordinates after WCS conversion."))
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

        # Does the plate solution fit? A wrong one matches Gaia at chance level.
        _sfcc_status(self, "Checking the plate solution against Gaia…")
        QApplication.processEvents()
        from setiastro.saspro.plate_solution_check import check_dialog_plate_solution
        _ps = check_dialog_plate_solution(self, sky_coords, tool="SSSC", log_tag="SSSC")
        if _ps == "resolved":                  # re-solved: Step 1 again on the new WCS
            return self.fetch_stars()
        if _ps == "stop":
            _sfcc_status(self, "Stopped: the plate solution looks wrong.")
            return

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

        # The detections, aligned with star_list, so the calibration measures
        # these same stars instead of detecting them all over again.
        self._det_sources = sources
        self._det_image = self.current_image
        self._det_sigma = float(sep_sigma)

        # SIMBAD (started after detection, above). prepare_simbad() probes
        # with a hard 4 s deadline, fails over to the Harvard / IDIA mirrors
        # and bounds the timeout, so a CDS outage costs seconds.
        if not simbad_job.done():
            _sfcc_status(self, "Waiting for SIMBAD spectral types…")
        simbad_result = simbad_job.wait()
        for m in simbad_job.messages:
            _sfcc_status(self, m)

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

        n_gaia_xp, n_simbad_pickles, n_bv_pickles, n_none = _star_source_counts(self.star_list)
        self._templates_for_hist = list(templates_for_hist)
        # One-click runs show the calibration's own four-panel figure at the
        # end; these star charts were only ever a between-steps view.
        if (not getattr(self, "_one_click", False)
                and getattr(self, "figure", None) is not None
                and getattr(self, "canvas", None) is not None):
            fig = self.figure
            fig.clf()
            _draw_star_summary(fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2),
                               self.star_list, templates_for_hist)
            fig.tight_layout()
            self.canvas.setVisible(True)
            _force_mpl_no_tex()
            self.canvas.draw()

        _sfcc_status(self,
            f"Stars found — {len(self.star_list):,} stars  ·  "
            f"{n_gaia_xp:,} Gaia XP  ·  {n_simbad_pickles:,} Pickles/SIMBAD  ·  "
            f"{n_bv_pickles:,} Pickles/B-V  ·  {n_none:,} unclassified")

        self._update_session_info_label()

    def _show_about(self):
        """Display the SSSC About dialog."""
        from PyQt6.QtWidgets import QScrollArea, QTextBrowser
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("About SSSC — Spectrophotometric Standard Star Calibration"))
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
</style>""" + self.tr("""
 
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
      Full R(&lambda;) shape solved per channel on piecewise-linear control points
      by an exact least-squares fit. Hot blue stars constrain the blue edge of each filter;
      cool red stars constrain the red edge. Each run&rsquo;s solution is saved to the
      session history, which the multi-night Stage 4 will combine.</li>
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
Gaia DR3 provides flux-calibrated XP spectra for about 34 million stars to G&nbsp;15.
On a typical wide-field image (0.5&ndash;2&deg; FOV) this yields 200&ndash;600
calibrator stars, comfortably in Stage 3 territory for most fields.
</p>
<p>
In testing on real data, SSSC&rsquo;s star colours match a full spectrophotometric
standard to within 1&ndash;2&nbsp;% without any QE curve. The residual RMS of every
stage that ran is shown in the Calibration Quality waterfall panel on every run.
</p>
<p class="note">
Tip: run SSSC on images from the same rig across several nights. Each run&rsquo;s
R(&lambda;) is saved to the session history, ready for the multi-night Stage 4.
The Camera&nbsp;/&nbsp;Rig label field separates history per sensor so solutions
don&rsquo;t mix between different imaging trains.
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
"""))
        layout.addWidget(browser)

        close_btn = QPushButton(self.tr("Close"))
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
            QMessageBox.critical(self, self.tr("Error"), self.tr("Could not read session cache:\n{0}").format(e))
            return

        if n == 0:
            QMessageBox.information(self, self.tr("Clear Session History"),
                self.tr("No saved solutions found for this filter+camera combination."))
            return

        reply = QMessageBox.question(
            self,
            self.tr("Clear Session History"),
            self.tr("Delete all {0} saved R(λ) solution(s) for session {1}?\n\n"
            "This will remove the accumulated calibration history for this\n"
            "filter+camera combination. The next run will start fresh.\n\n"
            "This cannot be undone.").format(n, sid),
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
            QMessageBox.critical(self, self.tr("Error"), self.tr("Failed to clear session:\n{0}").format(e))
            return

        self._update_session_info_label()
        _sfcc_status(self, f"[SSSC] Session history cleared — {n} run(s) deleted for {sid}")
        QMessageBox.information(self, self.tr("Clear Session History"),
            self.tr("Cleared {0} saved solution(s) for session {1}.\n"
            "The next calibration run will start fresh.").format(n, sid))

    def _run_meta(self, doc, ref_sed, enriched) -> dict:
        """What this run was on, saved with it in the session history: the
        filter set and camera label (the session's identity, readable), the
        white reference, the image, and the header's time and airmass, the
        observations the multi-night Stage 4 will need."""
        meta = {
            "filters": [self.r_filter_combo.currentText(), self.g_filter_combo.currentText(),
                        self.b_filter_combo.currentText()],
            "lp": [self.lp_filter_combo.currentText(), self.lp_filter_combo2.currentText()],
            "camera": self._camera_label(),
            "white_reference": str(ref_sed or ""),
            "n_xp": int(sum(1 for e in enriched if e.get("used_gaia"))),
        }
        name = ""
        try:
            name = str(doc.display_name()) if hasattr(doc, "display_name") else ""
        except Exception:
            name = ""
        md = getattr(doc, "metadata", None) or {}
        if not name:
            fp = md.get("file_path") or ""
            name = os.path.basename(str(fp)) if fp else ""
        meta["image"] = name
        hdr = md.get("original_header") or md.get("header") or md.get("wcs_header")
        if hdr is not None:
            def _get(*keys):
                for k in keys:
                    try:
                        v = hdr.get(k) if hasattr(hdr, "get") else None
                    except Exception:
                        v = None
                    if v not in (None, ""):
                        return v
                return None
            obs = _get("DATE-AVG", "DATE-OBS")
            if obs is not None:
                meta["date_obs"] = str(obs)
            for key, names in (("airmass", ("AIRMASS",)),
                               ("altitude", ("OBJCTALT", "CENTALT", "ALTITUDE", "ALT-OBJ"))):
                v = _get(*names)
                try:
                    if v is not None and np.isfinite(float(v)):
                        meta[key] = float(v)
                except (TypeError, ValueError):
                    pass
        return meta

    def _current_session_id(self) -> str:
        return make_session_id(
            self.r_filter_combo.currentText(),
            self.g_filter_combo.currentText(),
            self.b_filter_combo.currentText(),
            self.lp_filter_combo.currentText(),
            self.lp_filter_combo2.currentText(),
            self._camera_label(),
        )

    def _schedule_session_info(self, *_):
        t = getattr(self, "_session_info_timer", None)
        if t is not None and not getattr(self, "_session_info_closed", False):
            t.start()

    def _update_session_info_label(self):
        """Which session the current filters and camera name select, and what
        it has built up: runs, nights, the best run and the last one."""
        if getattr(self, "_session_info_closed", False) or not hasattr(self, "session_info_lbl"):
            return
        import html as _html
        from setiastro.saspro.widgets.theme_colors import theme_colors
        tr, esc = self.tr, _html.escape
        soft = theme_colors().soft.name()
        hint = lambda s: f'<span style="color:{soft}">{esc(s)}</span>'
        filters = [self.r_filter_combo.currentText(), self.g_filter_combo.currentText(),
                   self.b_filter_combo.currentText()]
        lp_now = [self.lp_filter_combo.currentText(), self.lp_filter_combo2.currentText()]
        cam = self._camera_label()
        try:
            sid = self._current_session_id()
            cache = self._get_session_cache()
            runs = [sr for _rid, sr in cache.runs(sid)]
        except Exception as e:
            self.session_info_lbl.setText(
                hint(tr("Session history unavailable: {0}").format(e)))
            self.clear_session_btn.setEnabled(False)
            return
        head = [esc(tr("Session {0}").format(sid)), esc(_filter_set_text(filters))]
        lp = [x for x in lp_now if x and x != "(None)"]
        if lp:
            head.append(esc(tr("LP: {0}").format(" + ".join(lp))))
        if cam:
            head.append(f"“{esc(cam)}”")
        lines = ["<b>" + "  ·  ".join(head) + "</b>"]
        if runs:
            lines.append(esc(_history_line(tr, runs)))
            last = runs[-1]                     # the cache is in calibration order
            t = _run_when(last)
            what = (last.meta or {}).get("image") or (t.strftime("%Y-%m-%d %H:%M") if t else "—")
            lines.append(esc(tr("Last: {0}  ·  Stage {1}  ·  {2} stars  ·  RMS {3}").format(
                what, last.stage, last.n_stars, f"{_run_rms(last):.4f}")))
        else:
            lines.append(esc(tr("No runs yet: the first calibration starts this setup's history.")))
            try:
                others = [s for s in cache.sessions() if s["session_id"] != sid
                          and list((s.get("meta") or {}).get("filters") or []) == filters
                          and list((s.get("meta") or {}).get("lp") or []) == lp_now]
            except Exception:
                others = []
            if others:
                lines.append(hint(tr("{0} other session(s) use these filters under a different "
                                     "camera name: see View History.").format(len(others))))
        if not cam:
            lines.append(hint(tr("Name the camera / rig to keep each setup's history apart.")))
        self.session_info_lbl.setText("<br>".join(lines))
        self.clear_session_btn.setEnabled(bool(runs))

    def _open_session_history(self):
        try:
            cache = self._get_session_cache()
            sid = self._current_session_id()
        except Exception as e:
            QMessageBox.critical(self, self.tr("Error"),
                                 self.tr("Could not read session cache:\n{0}").format(e))
            return
        dlg = SSSCHistoryDialog(self, cache, sid)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.exec()
        self._update_session_info_label()

    # ── Step 2: Run SSSC ──────────────────────────────────────────────────────

    def run_sssc(self):
        r_filt  = self.r_filter_combo.currentText()
        g_filt  = self.g_filter_combo.currentText()
        b_filt  = self.b_filter_combo.currentText()
        lp_filt  = self.lp_filter_combo.currentText()
        lp_filt2 = self.lp_filter_combo2.currentText()
        ref_sed  = self.star_combo.currentData()

        if r_filt == "(None)" and g_filt == "(None)" and b_filt == "(None)":
            QMessageBox.warning(self, self.tr("Error"), self.tr("Select at least one R/G/B filter."))
            return
        if not ref_sed:
            QMessageBox.warning(self, self.tr("Error"), self.tr("Select a white reference SED."))
            return

        doc = self.doc_manager.get_active_document()
        if doc is None or doc.image is None:
            QMessageBox.critical(self, self.tr("Error"), self.tr("No active document."))
            return

        img = doc.image
        if img.ndim != 3 or img.shape[2] != 3:
            QMessageBox.critical(self, self.tr("Error"), self.tr("Active document must be RGB."))
            return

        if not getattr(self, "star_list", None):
            QMessageBox.warning(self, self.tr("Error"),
                self.tr("Run Step 1: Fetch Stars before calibration."))
            return

        img_float = (img.astype(np.float32) / 255.0
                     if img.dtype == np.uint8
                     else img.astype(np.float32, copy=False))

        base = self._make_working_base_for_sep(img_float)
        if self.neutralize_chk.isChecked():
            base = self._neutralize_background(base, remove_pedestal=False)

        # Stars: the ones Step 1 found in this very image. Re-detecting them
        # found the same sources again at the cost of a second σ waterfall;
        # only an image that changed since Step 1 is detected again.
        det = getattr(self, "_det_sources", None)
        if (det is not None and getattr(self, "_det_image", None) is img
                and len(det) == len(self.star_list)):
            sources = det
            reuse = True
            _sfcc_status(self, f"Measuring the {sources.size:,} stars found in this image…")
        else:
            reuse = False
            gray     = _gray_for_sep(base)
            bkg      = sep.Background(gray)
            data_sub = gray - bkg.back()
            err      = float(bkg.globalrms)

            sources, sep_sigma = self._sep_detect_waterfall(
                data_sub, err, status_prefix="Re-detecting stars"
            )
            if sources is None or sources.size == 0:
                QMessageBox.critical(self, self.tr("SEP Error"), self.tr("No sources detected."))
                return

            r_fluxrad, _ = sep.flux_radius(
                gray, sources["x"], sources["y"],
                2.0 * sources["a"], 0.5,
                normflux=sources["flux"], subpix=5)
            mask    = (r_fluxrad > 0.2) & (r_fluxrad <= 10)
            sources = sources[mask]
            del gray, data_sub
            if sources.size == 0:
                QMessageBox.critical(self, self.tr("SEP Error"),
                    self.tr("All detections rejected by radius filter."))
                return

            _sfcc_status(self, f"Matching {sources.size:,} SEP sources (σ={sep_sigma:.1f}) to star catalog…")
            QApplication.processEvents()

        raw_matches = []
        for i, star in enumerate(self.star_list):
            if reuse:
                j = i                          # star i is source i
            else:
                dx = sources["x"] - star["x"]
                dy = sources["y"] - star["y"]
                j  = int(np.argmin(dx * dx + dy * dy))
                if (dx[j]**2 + dy[j]**2) >= (3.0**2):
                    continue
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
            QMessageBox.warning(self, self.tr("No Matches"),
                self.tr("No catalog stars matched to SEP detections."))
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

        # Curves and SEDs come from a per-session cache of the SASP files
        # (user curves first, as before); reopening per curve cost ~0.3 s each.
        curve_paths = (self.user_custom_path, self.sasp_data_path)

        def load_curve(ext):
            got = _read_sasp_tables(curve_paths, (ext,), "THROUGHPUT")
            if ext not in got:
                raise KeyError(f"Curve '{ext}' not found")
            return got[ext]

        def load_sed(ext):
            got = _read_sasp_tables(curve_paths, (ext,), "FLUX")
            if ext not in got:
                raise KeyError(f"SED '{ext}' not found")
            return got[ext]

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

        # The WHOLE green curve, as the green pixels see it. Until 2026-11-28
        # the expected green integrals used only the half-power core
        # (T >= 0.707 peak), meant to keep an OSC Bayer green's secondary red
        # lobe out. But that lobe's light reaches the green pixels, and it is
        # a larger share for redder stars, so no white reference cancels it:
        # with the SONY_COLOR_SENSOR_*-UVCUT curves it put star colours off by
        # 1.84 (R/G) and 1.67 (B/G); with the full curve, 1.002 and 0.977
        # (synthetic camera, real XP spectra). On mono filters it changed
        # nothing. A camera whose IR cut removes the lobes is described by
        # choosing the UV/IR-cut curves (or adding the cut filter as LP).
        # 0.707 survives only as where Stage 3 puts its green control points.
        T_sys_G_integrals = T_sys_G

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
                        T_sys_R=(T_sys_R * _PHOTON_W).astype(np.float64),
                        T_sys_G=(T_sys_G_integrals * _PHOTON_W).astype(np.float64),   # full curve
                        T_sys_B=(T_sys_B * _PHOTON_W).astype(np.float64),
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
                fs_i = np.interp(wl_grid, wl_s, fl_s, left=0.0, right=0.0) * _PHOTON_W
                template_integrals[pname] = (
                    float(_trapz(fs_i * T_sys_R, x=wl_grid)),
                    float(_trapz(fs_i * T_sys_G_integrals, x=wl_grid)),  # full curve
                    float(_trapz(fs_i * T_sys_B, x=wl_grid)),
                )
            except Exception as e:
                print(f"[SSSC] Pickles {pname} failed: {e}")

        def _has_xp(m):
            st = self.star_list[int(m["sim_index"])]
            return (st.get("gaia_source_id") is not None
                    and int(st["gaia_source_id"]) in gaia_integrals)

        raw_matches_with_spectrum = [
            m for m in raw_matches
            if _has_xp(m) or simbad_to_pickles.get(m["template"]) is not None
        ]
        # Gaia XP spectra are observed spectra, dust included. A Pickles
        # template picked from a SIMBAD type is a dust-free spectrum, so in any
        # reddened field it predicts the star too blue (ngc1333: measured /
        # expected R/G 1.30, IQR 1.21-1.79, against 1.03, IQR 1.02-1.05, for XP).
        # So templates only fill in when there are too few XP stars for Stage 2.
        n_xp_only = sum(1 for m in raw_matches_with_spectrum if _has_xp(m))
        if n_xp_only >= _STAGE2_MIN:
            n_dropped = len(raw_matches_with_spectrum) - n_xp_only
            raw_matches_with_spectrum = [m for m in raw_matches_with_spectrum if _has_xp(m)]
            if n_dropped:
                _sfcc_status(self, f"{n_xp_only} Gaia XP stars: calibrating on XP spectra only "
                                   f"({n_dropped} Pickles-template stars left out).")

        n_with_spectrum = len(raw_matches_with_spectrum)
        _sfcc_status(self,
            f"{len(gaia_integrals):,} Gaia XP + {len(template_integrals):,} Pickles  ·  "
            f"{n_with_spectrum:,} of {len(raw_matches):,} stars have a spectrum")
        QApplication.processEvents()

        if n_with_spectrum == 0:
            QMessageBox.warning(self, self.tr("No Spectra"),
                self.tr("No stars with Gaia XP spectrum or Pickles template found."))
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
            f"Measuring flux for {n_with_spectrum} stars…")
        QApplication.processEvents()

        base_ro  = np.ascontiguousarray(base, dtype=np.float32)
        enriched = []
        eps      = 1e-12

        # All stars in one pass (the same SEP sums the per-star version made).
        def _apertures(m):
            a    = float(m.get("a", 1.5))
            r    = float(np.clip(2.5 * a, 2.0, 12.0))
            rin  = float(np.clip(3.0 * r, 6.0, 40.0))
            rout = float(np.clip(5.0 * r, rin + 2.0, 60.0))
            return r, rin, rout

        apertures = [_apertures(m) for m in raw_matches_with_spectrum]
        phots = measure_stars_rgb_photometry(
            base_ro,
            [float(m["x"]) for m in raw_matches_with_spectrum],
            [float(m["y"]) for m in raw_matches_with_spectrum],
            [ap[0] for ap in apertures], [ap[1] for ap in apertures],
            [ap[2] for ap in apertures])

        def _measure_one(m, phot):
            si   = int(m["sim_index"])
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

        for m, phot in zip(raw_matches_with_spectrum, phots):
            try:
                result = _measure_one(m, phot)
            except Exception as e:
                print(f"[SSSC] Photometry failed for a star: {e}")
                continue
            if result is not None:
                enriched.append(result)

        self._last_matched = enriched
        _sfcc_status(self,
            f"Photometry complete — {len(enriched)} valid stars")
        QApplication.processEvents()

        if len(enriched) < _STAGE1_MIN:
            QMessageBox.warning(self, self.tr("Too Few Stars"),
                self.tr("Only {0} valid calibrator stars — need ≥ {1}.").format(len(enriched), _STAGE1_MIN))
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
                    f"ctrl_points={'yes' if has_cp else 'no'}) in this session's history")
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
            QMessageBox.critical(self, self.tr("Solver Error"),
                self.tr("Failed to solve system response:\n{0}").format(e))
            return

        self._last_sr = sr

        _sfcc_status(self, "Applying system response correction…")
        QApplication.processEvents()

        # The white reference through the SAME curves as the stars (photon-
        # counting, full curves): the correction divides by its ratios so it
        # comes out neutral.
        white_ref = (1.0, 1.0)
        try:
            wl_ref, fl_ref = load_sed(ref_sed)
            f_ref = np.interp(wl_grid, wl_ref, fl_ref, left=0.0, right=0.0) * _PHOTON_W
            S_ref_R = float(_trapz(f_ref * T_sys_R, x=wl_grid))
            S_ref_G = float(_trapz(f_ref * T_sys_G_integrals, x=wl_grid))
            S_ref_B = float(_trapz(f_ref * T_sys_B, x=wl_grid))
            if S_ref_G > 0 and S_ref_R > 0 and S_ref_B > 0:
                white_ref = (S_ref_R / S_ref_G, S_ref_B / S_ref_G)
            else:
                print(f"[SSSC] White reference {ref_sed} has no flux in a band; not applied.")
        except Exception as e:
            print(f"[SSSC] White reference {ref_sed} unavailable ({e}); not applied.")
        print(f"[SSSC] White reference {ref_sed}: expected R/G {white_ref[0]:.4f}, B/G {white_ref[1]:.4f}")

        calibrated = apply_sssc_correction(
            img_float, sr, enriched,
            wl_grid, T_sys_R, T_sys_G, T_sys_B,
            white_ref=white_ref,
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

        one_click = bool(getattr(self, "_one_click", False))
        try:
            # One-click runs add the star charts as a third column, so every
            # chart is on screen at the end.
            build_sssc_diagnostics_figure(
                self.figure,
                sr, enriched,
                T_sys_R, T_sys_G, T_sys_B, wl_grid,
                star_summary=((self.star_list, getattr(self, "_templates_for_hist", []))
                              if one_click else None),
            )
            if one_click:
                # drawn once the panel is shown; room kept for the report title
                _relayout_on_draw(self.figure, rect=(0, 0, 1, 0.95))
            else:
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
            "SSSC_white_reference": str(ref_sed),
            "SSSC_white_ref_ratios": [float(white_ref[0]), float(white_ref[1])],
        })

        self.doc_manager.update_active_document(
            out_img, metadata=new_meta,
            step_name="SSSC Calibrated", doc=doc)

        try:
            sr.meta = self._run_meta(doc, ref_sed, enriched)
        except Exception as e:
            print(f"[SSSC] Run metadata incomplete: {e}")
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
        if sr.stage >= 3:
            _stage_note = self.tr(
                "  ✓ Stage 3 (full R(λ)) solved from {0} stars.\n"
                "  Saved to the session history of this filter+camera combination.").format(sr.n_stars)
        elif stars_needed > 0:
            _stage_note = self.tr(
                "  {0} more stars needed for Stage 3 (full R(λ)).").format(stars_needed)
        else:
            _stage_note = self.tr("  Stage 3 criteria met — will activate on next run.")
        # Do the stars follow the chosen curves at all? With the right curves
        # the scalar (Stage 1) fit leaves ~2% (ngc1333: 1.8%); an OSC camera
        # behind an IR cut described by UV-cut-only curves left 16% (synthetic).
        _s1 = (sr.stage_rms or {}).get(1)
        _curve_warning = ""
        if _s1 is not None and np.isfinite(_s1) and _s1 > SSSC_CURVE_MISMATCH_RMS:
            _curve_warning = self.tr(
                "The stars do not follow these filter curves well (scalar fit residual {0:.0%}). "
                "Check the filters chosen: an OSC camera behind a UV/IR cut needs the UV/IR-cut "
                "curves, or the cut filter as LP/Cut.").format(_s1)
            print(f"[SSSC] Stage 1 residual {_s1:.3f} > {SSSC_CURVE_MISMATCH_RMS}: curves may not match the camera")
        self._last_result = {
            "n_stars": len(enriched), "n_xp": n_xp, "n_pickles": n_pkl,
            "stage_label": sr.stage_label, "bv_lo": float(sr.bv_range[0]),
            "bv_hi": float(sr.bv_range[1]), "rms": float(sr.residual_rms),
            "sessions": int(n_sessions), "gains": [float(g) for g in sr.gains],
            "stage_note": _stage_note.strip(),
            "curve_warning": _curve_warning,
        }
        if one_click:
            return True      # calibrate() shows the summary beside the charts
        QMessageBox.information(self, self.tr("SSSC Complete"),
            self.tr("Applied SSSC using {0} calibrator stars\n\n"
            "  Gaia XP spectra:   {1}\n"
            "  Pickles templates: {2}\n"
            "  Bootstrap stage:   {3}\n"
            "  B-V range:         {4:.2f} – {5:.2f}\n"
            "  Residual RMS:      {6:.4f}\n"
            "  Session runs:      {7}\n"
            "  Gains:  k_R={8:.4f}  k_G={9:.4f}  k_B={10:.4f}\n\n"
            "  No sensor QE curve was used.\n"
            "  Filter curves only — sensor response solved from stars.\n\n").format(
                len(enriched), n_xp, n_pkl, sr.stage_label,
                sr.bv_range[0], sr.bv_range[1], sr.residual_rms, n_sessions,
                sr.gains[0], sr.gains[1], sr.gains[2])
            + _stage_note
            + ("\n\n" + _curve_warning if _curve_warning else "")
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _restore_geometry(self):
        # Where the window was, not how big: it opens compact and grows when
        # results appear (a size saved by the old layout would leave it huge).
        s = QSettings()
        geom = s.value("SSSC/WindowGeometry")
        if geom:
            self.restoreGeometry(geom)
            self.resize(self.sizeHint())

    def _save_geometry(self):
        QSettings().setValue("SSSC/WindowGeometry", self.saveGeometry())

    def _cleanup(self):
        # No session-label refresh after close: it would reopen the cache.
        self._session_info_closed = True
        t = getattr(self, "_session_info_timer", None)
        if t is not None:
            t.stop()
        self._save_geometry()
        if getattr(self, "_curve_lib_conn", None):
            try:
                from setiastro.saspro.spectral_curves import curve_library_signals
                curve_library_signals().changed.disconnect(self._on_curve_library_changed)
            except Exception:
                pass
            self._curve_lib_conn = None
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
        if getattr(self, "_run_state", None) is not None:
            self._run_state.shutdown()
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