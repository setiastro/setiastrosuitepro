#src/setiastro/saspro/gaia_database.py
#!/usr/bin/env python3
# ======================================================
#   _____      __  _ ___         __           
#  / ___/___  / /_(_)   |  _____/ /__________ 
#  \__ \/ _ \/ __/ / /| | / ___/ __/ ___/ __ \
# ___/ /  __/ /_/ / ___ |(__  ) /_/ /  / /_/ /
#/____/\___/\__/_/_/  |_/____/\__/_/   \____/ 
#  SASpro Gaia DR3 Library
#
#  gaia_database.py  —  SASpro Gaia DR3 Spectral + Astrometric Libraries
#  Manages bulk-downloaded Gaia DR3 SQLite files, three flavors:
#    - XP spectra   (magnitude-banded, for color calibration)   34,414,104 rows
#    - Astrometry   (magnitude-banded, for plate solving)    1,585,128,120 rows
#    - Neighborhood (<=300 pc extract, for the 3D explorer)
#  Each provides a unified lookup layer:
#    1. Local bulk library (distributed SQLite files)
#    2. Live cache DB   (gaia_xp_cache.sqlite, spectral only)
#    3. Live archive    (fallback, handled by GaiaDownloader, spectral only)
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │  DR4 REBUILD WISHLIST                                                │
# │                                                                      │
# │  Gaia DR4 lands Dec 2025. Both catalogs get rebuilt from scratch at  │
# │  that point, so this is the list of everything we currently work     │
# │  around, in priority order. None of these are fixable without a      │
# │  re-fetch — they're all "the column isn't in the file".              │
# └──────────────────────────────────────────────────────────────────────┘
#
#  === ASTROMETRIC BUILDER (gaia_astrometric_builder.py) ===
#
#  1. CAPTURE `parallax_error`.  [biggest one]
#     Currently absent, so we cannot compute parallax_over_error, which is the
#     standard distance-reliability cut. We substitute (a) parallax > 0, (b) a
#     RUWE < 1.4 cut, and (c) a hard 300 pc radius on the neighborhood extract.
#
#     Be clear about what RUWE does and doesn't do: it is a dimensionless
#     goodness-of-fit for the 5-parameter astrometric solution. It flags BAD
#     FITS — unresolved binaries, blends, marginally resolved sources. It does
#     NOT flag a perfectly clean fit whose parallax is simply too small to
#     measure. A single star at 5 kpc can have RUWE = 1.0 and a parallax
#     consistent with zero. That second failure mode is why the radius is
#     capped at 300 pc: at close range the fractional parallax error is small
#     almost regardless, so a tight radius does the work parallax_over_error
#     would otherwise do.
#
#     With parallax_error in hand we can drop the arbitrary radius cap and let
#     each star qualify on its own merit (parallax_over_error > 5, say), which
#     means the Neighborhood Explorer reaches farther *honestly* rather than
#     farther *optimistically*. DR4's longer baseline also tightens parallaxes
#     across the board, so the reachable volume grows twice over.
#
#  2. CAPTURE `bp_rp` (and ideally `phot_bp_mean_mag` / `phot_rp_mean_mag`).
#     Gaia's color index. One REAL per source. Gives real stellar color for
#     every astrometric source — including the ~80% of the solar neighborhood
#     that has no XP spectrum at all. Would let the 3D explorer color every
#     point by temperature instead of coloring 19% and greying out the rest.
#
#  3. Consider `pmra_error` / `pmdec_error` and `phot_g_mean_flux_over_error`.
#     Cheap to carry, useful for weighting the plate solve.
#
#  === XP SPECTRAL BUILDER (gaia_bulk_builder.py) ===
#
#  4. POPULATE `sources.bp_rp`.
#     The column exists in the schema and is NULL in every file. Verified with
#     inspect_xp_colour_sources.py across lt8 / 8_10 / 100_105 / 140_141.
#
#  5. POPULATE `synth_phot(source_id, system_key, Sr, Sg, Sb)`.
#     Table exists, is empty everywhere. If we precompute synthetic RGB at
#     build time, star coloring becomes three REAL reads instead of a
#     zlib.decompress + three integrals per star. Today the Neighborhood
#     Explorer integrates the raw 343-point spectrum against the CIE 1931
#     color matching functions at runtime (see spectrum_to_srgb below), which
#     is correct and pretty but caps us at ~20k stars.
#
#     If we populate synth_phot we can color millions of points instantly, and
#     spectrum_to_srgb becomes the *builder's* job rather than the UI's.
#
#  6. Note DR4's XP magnitude limit. DR3 XP spectra stop at G ~= 15 and
#     saturate below G ~= 2.2 (so Sirius, Vega, Alpha Cen, Procyon are all
#     simply absent — a recurring "is this a bug?" question). If DR4 extends
#     either end, the tier boundaries in GROUP_DEFS need revisiting.
#
#  === GENERAL ===
#
#  7. Re-derive every n_items in GROUP_DEFS / ASTRO_GROUP_DEFS after the
#     rebuild. They are exact row counts, hardcoded so the UI never has to run
#     COUNT(*) over multi-GB WITHOUT ROWID tables. count_xp_library.py emits a
#     ready-to-paste block; the astrometric splitter prints its own totals.
#
#  8. Watch out for np.trapz — REMOVED in NumPy 2.0. We bind np.trapezoid with
#     a fallback. Any new numeric code should do the same.
#
# ======================================================

from __future__ import annotations

import math
import os
import shutil
import sqlite3
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np

from PyQt6.QtCore import (
    Qt, QThread, QStandardPaths, QSettings, QTimer,
    pyqtSignal as _Signal,
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QWidget, QMessageBox, QFileDialog,
    QScrollArea, QFrame, QTabWidget, QTextEdit, QLineEdit, QComboBox,
    QSizePolicy, QRadioButton, QButtonGroup, QSplitter,
    QDoubleSpinBox, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QMenu,
)
from PyQt6.QtGui import QColor
from PyQt6 import sip
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib
    matplotlib.rcParams["text.usetex"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# 3D Stellar Neighborhood Explorer needs pyqtgraph's OpenGL module, which in
# turn needs PyOpenGL. Both are optional — the tab degrades to a message if
# either is missing (e.g. a headless box, or a frozen build missing the
# OpenGL.platform.* hidden imports).
try:
    import pyqtgraph.opengl as gl
    HAS_GL = True
    _GL_IMPORT_ERROR = ""
except Exception as _e:            # ImportError, or OpenGL platform failures
    HAS_GL = False
    _GL_IMPORT_ERROR = str(_e)


# ── Backblaze public URLs ──────────────────────────────────────────────────────
LIBRARY_DOWNLOAD_BASE       = "https://f005.backblazeb2.com/file/setiastro-gaia/"
ASTRO_LIBRARY_DOWNLOAD_BASE = "https://f005.backblazeb2.com/file/setiastro-astrometry/"

# Precomputed 3D neighborhood extract (built by gaia_neighborhood_builder.py).
# Lives in the astrometric library dir alongside the magnitude tiers, but has
# its own schema (`neighborhood` table with galactic XYZ baked in).
NEIGHBORHOOD_FILENAME = "gaia_neighborhood.sqlite"
NEIGHBORHOOD_MAX_PC   = 300.0     # radius the extract was built with

# Gaia XP wavelength grid (nm) — 343 points, 336–1020 nm, 2 nm step
WL_GRID = np.arange(336, 1022, 2, dtype=np.float32)

# QSettings keys for user-chosen library directories
_SETTINGS_LIBRARY_DIR       = "gaia_library/custom_dir"
_SETTINGS_ASTRO_LIBRARY_DIR = "gaia_astro_library/custom_dir"


# ══════════════════════════════════════════════════════════════════════════════
#  Group definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GroupDef:
    key:         str
    label:       str
    mag_lo:      Optional[float]
    mag_hi:      Optional[float]
    filenames:   List[str]
    est_size:    str
    est_stars:   str
    description: str
    recommended: bool = False
    warning:     str  = ""
    # Astrometric bands only — unused (None) for spectral groups.
    dec_lo:      Optional[float] = None
    dec_hi:      Optional[float] = None
    # Word used in status text ("spectra" for XP groups, "stars" for astrometric).
    unit_label:  str = "spectra"
    # Overrides the tab's default table name when counting rows. The
    # neighborhood extract uses a `neighborhood` table, not `stars`.
    table_name:  Optional[str] = None
    # Exact row count for this group, known from the build. When set, the UI
    # displays it directly instead of running COUNT(*) over multi-GB
    # WITHOUT ROWID tables — which is minutes of pointless disk churn to
    # rediscover a number we already have.
    n_items:     Optional[int] = None


GROUP_DEFS: List[GroupDef] = [
    GroupDef(
        key="ultra_bright",
        label="Ultra-Bright  (G < 8)",
        mag_lo=None, mag_hi=8.0,
        filenames=["gaia_xp_lt8.sqlite"],
        est_size="~220 MB",
        est_stars="54,735 spectra",
        n_items=54735,
        description="Bright stars from G≈2.2 to G<8 — Arneb, Muphrid, Tania Australis, and tens of "
                    "thousands more. Note: Gaia's detector saturates below G≈2.2, so stars "
                    "like Vega, Sirius, Arcturus and Rigel are not in the dataset.",
        recommended=True,
    ),
    GroupDef(
        key="bright",
        label="Bright  (G 8–10)",
        mag_lo=8.0, mag_hi=10.0,
        filenames=["gaia_xp_8_10.sqlite"],
        est_size="~1.5 GB",
        est_stars="385,269 spectra",
        n_items=385269,
        description="Covers most calibration stars reachable from backyard setups.",
        recommended=True,
    ),
    GroupDef(
        key="medium",
        label="Medium  (G 10–12)",
        mag_lo=10.0, mag_hi=12.0,
        filenames=[
            "gaia_xp_100_105.sqlite",
            "gaia_xp_105_110.sqlite",
            "gaia_xp_110_115.sqlite",
            "gaia_xp_115_120.sqlite",
        ],
        est_size="~9.5 GB  (4 files)",
        est_stars="2,426,887 spectra",
        n_items=2426887,
        description="Dense coverage — recommended for wide-field and narrowband imaging.",
        recommended=True,
    ),
    GroupDef(
        key="faint",
        label="Faint  (G 12–14)",
        mag_lo=12.0, mag_hi=14.0,
        filenames=[
            "gaia_xp_120_122.sqlite", "gaia_xp_122_124.sqlite",
            "gaia_xp_124_126.sqlite", "gaia_xp_126_128.sqlite",
            "gaia_xp_128_130.sqlite", "gaia_xp_130_132.sqlite",
            "gaia_xp_132_134.sqlite", "gaia_xp_134_136.sqlite",
            "gaia_xp_136_138.sqlite", "gaia_xp_138_140.sqlite",
        ],
        est_size="~50 GB  (10 files)",
        est_stars="12,852,953 spectra",
        n_items=12852953,
        description="Deep coverage for long-exposure narrowband work. "
                    "Note: file sizes range from 2–9 GB each.",
        warning="Large download — ~50 GB total. Allow many hours on a typical connection.",
    ),
    GroupDef(
        key="very_faint",
        label="Very Faint  (G 14–15)",
        mag_lo=14.0, mag_hi=15.0,
        filenames=[
            "gaia_xp_140_141.sqlite", "gaia_xp_141_142.sqlite",
            "gaia_xp_142_143.sqlite", "gaia_xp_143_144.sqlite",
            "gaia_xp_144_145.sqlite", "gaia_xp_145_146.sqlite",
            "gaia_xp_146_147.sqlite", "gaia_xp_147_148.sqlite",
            "gaia_xp_148_149.sqlite", "gaia_xp_149_150.sqlite",
        ],
        est_size="~73 GB  (10 files)",
        est_stars="18,694,260 spectra",
        n_items=18694260,
        description="Maximum depth. For the deepest narrowband fields. "
                    "Gaia DR3 XP spectra top out at G≈15.",
        warning="Very large download — ~73 GB total. Files range 5–10 GB each. "
                "Only needed for extreme deep-field work.",
    ),
]

_FILENAME_TO_GROUP: Dict[str, str] = {
    fn: g.key for g in GROUP_DEFS for fn in g.filenames
}


# ── Astrometric magnitude tiers ──────────────────────────────────────────────
# Matches gaia_astro_split_bands.py's SPLIT_PLAN exactly: 41 physical files,
# grouped here into 5 user-facing tiers at clean magnitude boundaries so the
# UI reuses the same "one tier = many files, one Install downloads them all"
# pattern the Spectrum Library tab already uses for its faint tiers.
#
# Row counts and sizes below are the REAL measured values from the completed
# split (1,585,128,120 rows / 183.9 GB total), not estimates.

def _astro_filenames(lo: float, hi: float, step: float) -> List[str]:
    """
    Generate gaia_astro_<lo*10>_<hi*10>.sqlite filenames for a magnitude
    range, matching gaia_astro_split_bands.py's _make_splits/_filename
    naming exactly — never hand-type these, a mismatch breaks file lookup.
    """
    edges = [round(lo + i * step, 4) for i in range(int(round((hi - lo) / step)) + 1)]
    edges[0]  = lo
    edges[-1] = hi
    return [
        f"gaia_astro_{int(round(a * 10))}_{int(round(b * 10))}.sqlite"
        for a, b in zip(edges[:-1], edges[1:])
    ]


ASTRO_GROUP_DEFS: List[GroupDef] = [
    GroupDef(
        key="astro_neighborhood",
        label="Stellar Neighborhood  (within 300 pc)",
        mag_lo=None, mag_hi=None,
        filenames=[NEIGHBORHOOD_FILENAME],
        est_size="~2.5 GB  (1 file)",
        est_stars="14,708,327 stars",
        description="Precomputed 3D positions (galactic XYZ, Sun at origin) for every "
                    "Gaia DR3 source with a usable parallax within 300 parsecs. Powers "
                    "the Stellar Neighborhood Explorer tab. Complete and self-contained — "
                    "no magnitude tiers needed.",
        recommended=True,
        unit_label="stars",
        table_name="neighborhood",
    ),
    GroupDef(
        key="astro_bright",
        label="Bright  (G < 17)",
        mag_lo=None, mag_hi=17.0,
        filenames=(
            ["gaia_astro_lt15.sqlite", "gaia_astro_150_160.sqlite"]
            + _astro_filenames(16.0, 17.0, 0.5)
        ),
        est_size="16.9 GB  (4 files)",
        est_stars="139,482,635 stars",
        description="Bright astrometric anchors — enough for most wide-field plate solves.",
        recommended=True,
        unit_label="stars",
        n_items=139482635,
    ),
    GroupDef(
        key="astro_faint",
        label="Faint  (G 17–18)",
        mag_lo=17.0, mag_hi=18.0,
        filenames=_astro_filenames(17.0, 18.0, 0.2),
        est_size="15.7 GB  (5 files)",
        est_stars="129,386,971 stars",
        description="Dense coverage for typical narrow-field solves and fainter guide stars.",
        recommended=True,
        unit_label="stars",
        n_items=129386971,
    ),
    GroupDef(
        key="astro_very_faint",
        label="Very Faint  (G 18–19)",
        mag_lo=18.0, mag_hi=19.0,
        filenames=_astro_filenames(18.0, 19.0, 0.1),
        est_size="28.9 GB  (10 files)",
        est_stars="239,207,768 stars",
        description="For deep, narrow-field solves needing very dense star fields.",
        warning="Large download — ~29 GB total across 10 files.",
        unit_label="stars",
        n_items=239207768,
    ),
    GroupDef(
        key="astro_extreme",
        label="Extreme  (G 19–20)",
        mag_lo=19.0, mag_hi=20.0,
        filenames=_astro_filenames(19.0, 20.0, 0.1),
        est_size="50.4 GB  (10 files)",
        est_stars="420,229,783 stars",
        description="Rarely needed — extremely deep solving fields only.",
        warning="Very large download — ~50 GB total across 10 files.",
        unit_label="stars",
        n_items=420229783,
    ),
    GroupDef(
        key="astro_max_depth",
        label="Maximum Depth  (G ≥ 20, full DR3)",
        mag_lo=20.0, mag_hi=None,
        filenames=_astro_filenames(20.0, 21.0, 0.1) + [
            "gaia_astro_210_213.sqlite", "gaia_astro_gt213.sqlite",
        ],
        est_size="72.0 GB  (12 files)",
        est_stars="656,820,963 stars",
        description="Full DR3 depth — the practical faint limit of the catalog. "
                    "Includes sources with no measured G magnitude.",
        warning="Enormous download — ~72 GB total across 12 files.",
        unit_label="stars",
        n_items=656820963,
    ),
]

_ASTRO_FILENAME_TO_GROUP: Dict[str, str] = {
    fn: g.key for g in ASTRO_GROUP_DEFS for fn in g.filenames
}


# ══════════════════════════════════════════════════════════════════════════════
#  Library directories  (each respects its own user override in QSettings)
# ══════════════════════════════════════════════════════════════════════════════

def _default_library_dir() -> Path:
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    d = base / "gaia_library"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_library_dir() -> Path:
    """
    Returns the active spectral-library directory.
    Checks QSettings for a user override; falls back to the default
    AppData location if none is set or the stored path no longer exists.
    """
    s = QSettings()
    custom = s.value(_SETTINGS_LIBRARY_DIR, "", type=str)
    if custom:
        p = Path(custom)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass   # fall through to default
    return _default_library_dir()


def set_library_dir(path: Path):
    """Persist a user-chosen spectral-library directory to QSettings."""
    QSettings().setValue(_SETTINGS_LIBRARY_DIR, str(path))


def clear_library_dir_override():
    """Reset the spectral library to the default AppData location."""
    QSettings().remove(_SETTINGS_LIBRARY_DIR)


def _default_astro_library_dir() -> Path:
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    d = base / "gaia_astro_library"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_astro_library_dir() -> Path:
    """Returns the active astrometric-library directory (mirrors get_library_dir())."""
    s = QSettings()
    custom = s.value(_SETTINGS_ASTRO_LIBRARY_DIR, "", type=str)
    if custom:
        p = Path(custom)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    return _default_astro_library_dir()


def set_astro_library_dir(path: Path):
    """Persist a user-chosen astrometric-library directory to QSettings."""
    QSettings().setValue(_SETTINGS_ASTRO_LIBRARY_DIR, str(path))


def clear_astro_library_dir_override():
    """Reset the astrometric library to the default AppData location."""
    QSettings().remove(_SETTINGS_ASTRO_LIBRARY_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  Spectral data layer  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CalibratedSpectrum:
    source_id:   int
    wavelengths: np.ndarray
    flux:        np.ndarray
    flux_error:  Optional[np.ndarray] = None

    def get_flux_at(self, wavelength_nm: float) -> float:
        return float(np.interp(wavelength_nm, self.wavelengths, self.flux))


def _decompress(data: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(data), dtype=np.float32)


@dataclass
class GroupStatus:
    group:        GroupDef
    installed:    List[str]
    missing:      List[str]
    total_mb:     float
    # Row count for whatever table this group's kind uses ("spectra" or
    # "stars") — field name kept for backward compatibility, just holds
    # whichever count applies.
    total_spectra: int

    @property
    def fully_installed(self) -> bool:
        return len(self.missing) == 0

    @property
    def partially_installed(self) -> bool:
        return bool(self.installed) and bool(self.missing)

    @property
    def not_installed(self) -> bool:
        return not self.installed


class GaiaBulkLibrary:
    def __init__(self, library_dir: Optional[Path] = None):
        self._dir = library_dir or get_library_dir()
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._open_installed()

    def _open_installed(self):
        for path in sorted(self._dir.glob("gaia_xp_*.sqlite")):
            fname = path.name
            if fname not in self._connections:
                try:
                    conn = sqlite3.connect(str(path), check_same_thread=False)
                    conn.execute("PRAGMA query_only = ON;")
                    self._connections[fname] = conn
                except Exception as e:
                    print(f"[GaiaBulkLibrary] Could not open {fname}: {e}")

    def refresh(self):
        self._dir = get_library_dir()
        self._open_installed()

    def close(self):
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

    def get_spectrum(self, source_id: int) -> Optional[CalibratedSpectrum]:
        sid = int(source_id)
        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT flux_compressed, flux_error_compressed "
                    "FROM spectra WHERE source_id=?", (sid,))
                row = cur.fetchone()
                if row is not None:
                    return CalibratedSpectrum(
                        source_id=sid,
                        wavelengths=WL_GRID.copy(),
                        flux=_decompress(row[0]),
                        flux_error=_decompress(row[1]) if row[1] else None,
                    )
            except Exception:
                continue
        return None

    def get_source_info(self, source_id: int) -> Optional[Dict]:
        sid = int(source_id)
        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT ra, dec, phot_g_mean_mag FROM sources WHERE source_id=?",
                    (sid,))
                row = cur.fetchone()
                if row:
                    return {"ra": row[0], "dec": row[1], "gmag": row[2]}
            except Exception:
                continue
        return None

    def has_spectrum(self, source_id: int) -> bool:
        sid = int(source_id)
        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1 FROM spectra WHERE source_id=? LIMIT 1", (sid,))
                if cur.fetchone():
                    return True
            except Exception:
                continue
        return False

    def find_nearest_batch(
        self,
        coords: list[tuple[float, float]],
        radius_arcsec: float = 10.0,
    ) -> dict[int, tuple[int, float]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not coords or not self._connections:
            return {}

        radius_deg = radius_arcsec / 3600.0
        ras  = [c[0] for c in coords]
        decs = [c[1] for c in coords]
        dec_min = min(decs) - radius_deg
        dec_max = max(decs) + radius_deg
        mid_dec = (min(decs) + max(decs)) / 2.0
        cosd = max(1e-6, abs(math.cos(math.radians(mid_dec))))
        ra_min  = min(ras) - radius_deg / cosd
        ra_max  = max(ras) + radius_deg / cosd

        def _query_file(conn):
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT source_id, ra, dec FROM sources
                    WHERE dec BETWEEN ? AND ?
                      AND ra  BETWEEN ? AND ?
                """, (dec_min, dec_max, ra_min, ra_max))
                return cur.fetchall()
            except Exception:
                return []

        all_sources: list[tuple[int, float, float]] = []
        conns = list(self._connections.values())
        with ThreadPoolExecutor(max_workers=min(8, len(conns))) as ex:
            futures = [ex.submit(_query_file, conn) for conn in conns]
            for fut in as_completed(futures):
                try:
                    all_sources.extend(fut.result())
                except Exception:
                    pass

        if not all_sources:
            return {}

        src_ids  = np.array([s[0] for s in all_sources], dtype=np.int64)
        src_ras  = np.array([s[1] for s in all_sources], dtype=np.float64)
        src_decs = np.array([s[2] for s in all_sources], dtype=np.float64)

        results: dict[int, tuple[int, float]] = {}
        for i, (ra, dec) in enumerate(coords):
            cosd_i = max(1e-6, abs(math.cos(math.radians(dec))))
            dra    = (src_ras  - ra)  * cosd_i
            ddec   = (src_decs - dec)
            seps   = np.hypot(dra, ddec) * 3600.0
            j = int(np.argmin(seps))
            if seps[j] <= radius_arcsec:
                results[i] = (int(src_ids[j]), float(seps[j]))

        return results

    def find_nearest(self, ra: float, dec: float,
                     radius_arcsec: float = 3.0) -> Optional[Tuple[int, float]]:
        radius_deg = radius_arcsec / 3600.0
        cosd = max(1e-6, abs(math.cos(math.radians(dec))))
        best_sid, best_sep = None, 1e9

        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT source_id, ra, dec FROM sources
                    WHERE dec BETWEEN ? AND ?
                      AND ra  BETWEEN ? AND ?
                """, (
                    dec - radius_deg, dec + radius_deg,
                    ra  - radius_deg / cosd, ra + radius_deg / cosd,
                ))
                for sid, sra, sdec in cur.fetchall():
                    dra  = (sra - ra) * cosd
                    ddec = sdec - dec
                    sep  = math.hypot(dra, ddec) * 3600.0
                    if sep < best_sep:
                        best_sep, best_sid = sep, sid
            except Exception:
                continue

        if best_sid is not None and best_sep <= radius_arcsec:
            return (int(best_sid), float(best_sep))
        return None

    def get_group_status(self) -> List[GroupStatus]:
        """
        WARNING: runs COUNT(*) on every installed file. The UI no longer calls
        this — GroupDef.n_items carries exact build-time counts. Kept only for
        external callers; expect it to be slow on the large tiers.
        """
        statuses = []
        for g in GROUP_DEFS:
            installed, missing = [], []
            total_mb = 0.0
            total_spectra = 0
            for fname in g.filenames:
                path = self._dir / fname
                if path.exists():
                    installed.append(fname)
                    total_mb += path.stat().st_size / (1024 * 1024)
                    if fname in self._connections:
                        try:
                            n = self._connections[fname].execute(
                                "SELECT COUNT(*) FROM spectra").fetchone()[0]
                            total_spectra += n
                        except Exception:
                            pass
                else:
                    missing.append(fname)
            statuses.append(GroupStatus(
                group=g,
                installed=installed,
                missing=missing,
                total_mb=total_mb,
                total_spectra=total_spectra,
            ))
        return statuses

    def total_spectra(self) -> int:
        total = 0
        for conn in self._connections.values():
            try:
                total += conn.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
            except Exception:
                pass
        return total

    def installed_bands(self) -> List[str]:
        return list(self._connections.keys())

    def close_file(self, filename: str):
        conn = self._connections.pop(filename, None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ── Spectral singleton ─────────────────────────────────────────────────────────

_library_instance: Optional[GaiaBulkLibrary] = None


def get_library() -> GaiaBulkLibrary:
    global _library_instance
    if _library_instance is None:
        _library_instance = GaiaBulkLibrary()
    return _library_instance


def refresh_library():
    global _library_instance
    if _library_instance is not None:
        _library_instance.close()
    _library_instance = GaiaBulkLibrary()


# ══════════════════════════════════════════════════════════════════════════════
#  Astrometric data layer  (new)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StarRecord:
    source_id: int
    ra: float
    dec: float
    pmra: Optional[float]
    pmdec: Optional[float]
    parallax: Optional[float]
    ref_epoch: float
    gmag: Optional[float]
    ruwe: Optional[float]
    astrometric_params_solved: Optional[int]


class GaiaAstrometricLibrary:
    """
    Unified lookup layer over the installed Gaia DR3 astrometric SQLite
    bands (built by gaia_astrometric_builder.py). Uses the same dec_zone +
    ra composite-index "zone file" technique the builder documents, so
    bounding-box/radius queries stay fast without needing SQLite's R-Tree
    extension (which isn't guaranteed to be compiled into every platform's
    SQLite under a frozen build).
    """

    def __init__(self, library_dir: Optional[Path] = None):
        self._dir = library_dir or get_astro_library_dir()
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._zone_height_deg = 1.0
        self._open_installed()

    def _open_installed(self):
        # The distributed tier files are named gaia_astro_lt15.sqlite,
        # gaia_astro_150_160.sqlite, ... gaia_astro_gt213.sqlite. (The
        # gaia_astro_dec_*.sqlite files are the *build-time* declination bands,
        # never shipped.) Exclude the neighborhood extract, which has its own
        # schema and its own library class.
        for path in sorted(self._dir.glob("gaia_astro_*.sqlite")):
            fname = path.name
            if fname == NEIGHBORHOOD_FILENAME:
                continue
            if fname not in self._connections:
                try:
                    conn = sqlite3.connect(str(path), check_same_thread=False)
                    conn.execute("PRAGMA query_only = ON;")
                    self._connections[fname] = conn
                    try:
                        row = conn.execute(
                            "SELECT value FROM metadata WHERE key='zone_height_deg'"
                        ).fetchone()
                        if row:
                            self._zone_height_deg = float(row[0])
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[GaiaAstrometricLibrary] Could not open {fname}: {e}")

    def refresh(self):
        self._dir = get_astro_library_dir()
        self._open_installed()

    def close(self):
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

    def close_file(self, filename: str):
        conn = self._connections.pop(filename, None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    def installed_bands(self) -> List[str]:
        return list(self._connections.keys())

    def _dec_zone_of(self, dec: float) -> int:
        return int(math.floor((dec + 90.0) / self._zone_height_deg))

    def find_stars_in_box(
        self,
        ra_lo: float, ra_hi: float,
        dec_lo: float, dec_hi: float,
        max_mag: Optional[float] = None,
    ) -> List[StarRecord]:
        """
        Bounding-box query using the dec_zone + ra composite index.
        Handles RA wraparound at 0/360 by splitting into two ranges
        when ra_lo > ra_hi (i.e. the box crosses the meridian).
        """
        if not self._connections:
            return []

        zone_lo = self._dec_zone_of(dec_lo)
        zone_hi = self._dec_zone_of(dec_hi)

        ra_ranges = ([(ra_lo, 360.0), (0.0, ra_hi)] if ra_lo > ra_hi
                    else [(ra_lo, ra_hi)])

        mag_clause = " AND phot_g_mean_mag < ?" if max_mag is not None else ""
        results: List[StarRecord] = []

        for conn in self._connections.values():
            for r_lo, r_hi in ra_ranges:
                try:
                    sql = (
                        "SELECT source_id, ra, dec, pmra, pmdec, parallax, "
                        "ref_epoch, phot_g_mean_mag, ruwe, astrometric_params_solved "
                        "FROM stars WHERE dec_zone BETWEEN ? AND ? "
                        "AND ra BETWEEN ? AND ?" + mag_clause
                    )
                    params = [zone_lo, zone_hi, r_lo, r_hi]
                    if max_mag is not None:
                        params.append(max_mag)
                    cur = conn.execute(sql, params)
                    for row in cur.fetchall():
                        results.append(StarRecord(
                            source_id=row[0], ra=row[1], dec=row[2],
                            pmra=row[3], pmdec=row[4], parallax=row[5],
                            ref_epoch=row[6], gmag=row[7], ruwe=row[8],
                            astrometric_params_solved=row[9],
                        ))
                except Exception:
                    continue

        # Zone bounds are coarser than the exact box — final precise dec filter.
        return [s for s in results if dec_lo <= s.dec <= dec_hi]

    def find_stars_near(
        self, ra: float, dec: float,
        radius_arcsec: float = 60.0,
        max_mag: Optional[float] = None,
    ) -> List[Tuple[StarRecord, float]]:
        """Returns (StarRecord, separation_arcsec) pairs within radius, nearest first."""
        radius_deg = radius_arcsec / 3600.0
        cosd = max(1e-6, abs(math.cos(math.radians(dec))))
        box = self.find_stars_in_box(
            ra - radius_deg / cosd, ra + radius_deg / cosd,
            dec - radius_deg, dec + radius_deg,
            max_mag=max_mag,
        )
        out = []
        for s in box:
            dra  = (s.ra - ra) * cosd
            ddec = s.dec - dec
            sep  = math.hypot(dra, ddec) * 3600.0
            if sep <= radius_arcsec:
                out.append((s, sep))
        out.sort(key=lambda t: t[1])
        return out

    def get_group_status(self) -> List[GroupStatus]:
        """
        WARNING: runs COUNT(*) on every installed file — on the deep tiers that
        is a full scan of a multi-GB WITHOUT ROWID table. The UI uses
        GroupDef.n_items instead. Kept only for external callers.
        """
        statuses = []
        for g in ASTRO_GROUP_DEFS:
            installed, missing = [], []
            total_mb = 0.0
            total_stars = 0
            for fname in g.filenames:
                path = self._dir / fname
                if path.exists():
                    installed.append(fname)
                    total_mb += path.stat().st_size / (1024 * 1024)
                    if fname in self._connections:
                        try:
                            n = self._connections[fname].execute(
                                "SELECT COUNT(*) FROM stars").fetchone()[0]
                            total_stars += n
                        except Exception:
                            pass
                else:
                    missing.append(fname)
            statuses.append(GroupStatus(
                group=g, installed=installed, missing=missing,
                total_mb=total_mb, total_spectra=total_stars,
            ))
        return statuses

    def total_stars(self) -> int:
        total = 0
        for conn in self._connections.values():
            try:
                total += conn.execute("SELECT COUNT(*) FROM stars").fetchone()[0]
            except Exception:
                pass
        return total


# ── Astrometric singleton ──────────────────────────────────────────────────────

_astro_library_instance: Optional[GaiaAstrometricLibrary] = None


def get_astro_library() -> GaiaAstrometricLibrary:
    global _astro_library_instance
    if _astro_library_instance is None:
        _astro_library_instance = GaiaAstrometricLibrary()
    return _astro_library_instance


def refresh_astro_library():
    global _astro_library_instance
    if _astro_library_instance is not None:
        _astro_library_instance.close()
    _astro_library_instance = GaiaAstrometricLibrary()


# ══════════════════════════════════════════════════════════════════════════════
#  Neighborhood data layer  (3D explorer)
# ══════════════════════════════════════════════════════════════════════════════

class GaiaNeighborhoodLibrary:
    """
    Read-only access to gaia_neighborhood.sqlite — the precomputed extract of
    every Gaia DR3 source with a usable parallax inside 300 pc, built by
    gaia_neighborhood_builder.py.

    Galactic Cartesian XYZ (Sun at origin, +Z toward the north galactic pole,
    disk in the XY plane) are baked in at build time, so the explorer can
    stream them straight into a GL vertex buffer with no per-point math.

    Quality note: DR3's astrometric builder did not capture parallax_error, so
    distance reliability rests on (a) parallax > 0 (enforced at build time),
    (b) a RUWE cut applied here at query time, and (c) a conservative radius.
    RUWE catches *bad astrometric fits* (binaries, blends); it does NOT catch a
    clean fit whose parallax is simply too small to be meaningful. That second
    failure mode is what the radius limit guards against — which is why the
    extract stops at 300 pc. DR4 will add parallax_error and let this reach
    farther honestly.
    """

    def __init__(self, library_dir: Optional[Path] = None):
        self._dir = library_dir or get_astro_library_dir()
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    def _path(self) -> Path:
        return self._dir / NEIGHBORHOOD_FILENAME

    def _open(self):
        p = self._path()
        if not p.exists():
            return
        try:
            self._conn = sqlite3.connect(str(p), check_same_thread=False)
            self._conn.execute("PRAGMA query_only = ON;")
        except Exception as e:
            print(f"[GaiaNeighborhoodLibrary] Could not open {p.name}: {e}")
            self._conn = None

    def refresh(self):
        self.close()
        self._dir = get_astro_library_dir()
        self._open()

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def is_installed(self) -> bool:
        return self._conn is not None

    def max_pc(self) -> float:
        """Radius the extract was built with, read from its metadata table."""
        if self._conn is None:
            return NEIGHBORHOOD_MAX_PC
        try:
            row = self._conn.execute(
                "SELECT value FROM metadata WHERE key='max_pc'").fetchone()
            return float(row[0]) if row else NEIGHBORHOOD_MAX_PC
        except Exception:
            return NEIGHBORHOOD_MAX_PC

    def total_stars(self) -> int:
        if self._conn is None:
            return 0
        try:
            return self._conn.execute("SELECT COUNT(*) FROM neighborhood").fetchone()[0]
        except Exception:
            return 0

    def load_sphere(
        self,
        max_pc: float,
        max_ruwe: Optional[float] = 1.4,
        max_gmag: Optional[float] = None,
        limit: int = 2_000_000,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Fetch every star inside `max_pc`, optionally RUWE- and magnitude-cut,
        nearest first. Returns numpy arrays ready for GLScatterPlotItem, or
        None if the library isn't installed.

        `limit` is a hard cap so an accidental 300 pc + no-cuts query can't try
        to push ~14M points into a vertex buffer and wedge the UI.
        """
        if self._conn is None:
            return None

        where = ["dist_pc <= ?"]
        params: list = [float(max_pc)]
        if max_ruwe is not None:
            # NULL ruwe passes: any row here has a parallax, so it came from a
            # 5- or 6-parameter solution and should carry a RUWE. NULLs are rare
            # and dropping them silently would be a quiet data loss.
            where.append("(ruwe IS NULL OR ruwe < ?)")
            params.append(float(max_ruwe))
        if max_gmag is not None:
            where.append("(phot_g_mean_mag IS NOT NULL AND phot_g_mean_mag <= ?)")
            params.append(float(max_gmag))
        params.append(int(limit))

        sql = (
            "SELECT source_id, x_pc, y_pc, z_pc, dist_pc, phot_g_mean_mag, ruwe, ra, dec "
            "FROM neighborhood WHERE " + " AND ".join(where) +
            " ORDER BY dist_pc LIMIT ?"
        )

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except Exception as e:
            print(f"[GaiaNeighborhoodLibrary] query failed: {e}")
            return None

        n = len(rows)
        if n == 0:
            return {k: np.empty(0, dtype=t) for k, t in (
                ("source_id", np.int64), ("dist_pc", np.float32),
                ("gmag", np.float32), ("ruwe", np.float32),
                ("ra", np.float64), ("dec", np.float64),
            )} | {"xyz": np.empty((0, 3), dtype=np.float32)}

        src  = np.empty(n, dtype=np.int64)
        xyz  = np.empty((n, 3), dtype=np.float32)
        dist = np.empty(n, dtype=np.float32)
        gmag = np.empty(n, dtype=np.float32)
        ruwe = np.empty(n, dtype=np.float32)
        ra   = np.empty(n, dtype=np.float64)
        dec  = np.empty(n, dtype=np.float64)

        for i, (sid, x, y, z, d, g, r, a, dd) in enumerate(rows):
            src[i] = sid
            xyz[i, 0] = x; xyz[i, 1] = y; xyz[i, 2] = z
            dist[i] = d
            gmag[i] = np.nan if g is None else g
            ruwe[i] = np.nan if r is None else r
            ra[i]   = a
            dec[i]  = dd

        return {"source_id": src, "xyz": xyz, "dist_pc": dist,
                "gmag": gmag, "ruwe": ruwe, "ra": ra, "dec": dec}

    def get_star(self, source_id: int) -> Optional[Dict]:
        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT source_id, ra, dec, parallax, dist_pc, x_pc, y_pc, z_pc, "
                "pmra, pmdec, phot_g_mean_mag, ruwe FROM neighborhood WHERE source_id=?",
                (int(source_id),)).fetchone()
        except Exception:
            return None
        if not row:
            return None
        keys = ("source_id", "ra", "dec", "parallax", "dist_pc", "x_pc", "y_pc",
                "z_pc", "pmra", "pmdec", "gmag", "ruwe")
        return dict(zip(keys, row))


_neighborhood_instance: Optional[GaiaNeighborhoodLibrary] = None


def get_neighborhood_library() -> GaiaNeighborhoodLibrary:
    global _neighborhood_instance
    if _neighborhood_instance is None:
        _neighborhood_instance = GaiaNeighborhoodLibrary()
    return _neighborhood_instance


def refresh_neighborhood_library():
    global _neighborhood_instance
    if _neighborhood_instance is not None:
        _neighborhood_instance.close()
    _neighborhood_instance = GaiaNeighborhoodLibrary()


# ══════════════════════════════════════════════════════════════════════════════
#  Migration worker  (generic — used for both spectral and astrometric moves)
# ══════════════════════════════════════════════════════════════════════════════

class _MigrateWorker(QThread):
    """
    Moves or copies installed sqlite files from src_dir to dest_dir,
    one file at a time.  Uses shutil.move/copy2 so cross-device transfers
    work correctly.
    """
    file_progress  = _Signal(int, int, str)   # files_done, files_total, current_filename
    file_done      = _Signal(str, bool)        # filename, success
    finished       = _Signal(bool, str)        # success, message
    cancelled      = _Signal()

    def __init__(self, src_dir: Path, dest_dir: Path,
                 filenames: List[str], mode: str, parent=None):
        """
        mode: 'move' | 'copy'
        filenames: list of sqlite filenames that exist in src_dir
        """
        super().__init__(parent)
        self._src      = src_dir
        self._dest     = dest_dir
        self._files    = filenames
        self._mode     = mode
        self._cancel   = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total = len(self._files)
        done  = 0

        for fname in self._files:
            if self._cancel:
                self.cancelled.emit()
                return

            src  = self._src  / fname
            dest = self._dest / fname

            if not src.exists():
                done += 1
                self.file_done.emit(fname, True)
                self.file_progress.emit(done, total, fname)
                continue

            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                # Already there (same size) — skip copy, remove source if moving
                if self._mode == "move":
                    try:
                        src.unlink()
                    except Exception:
                        pass
                done += 1
                self.file_done.emit(fname, True)
                self.file_progress.emit(done, total, fname)
                continue

            self.file_progress.emit(done, total, fname)
            try:
                if self._mode == "move":
                    shutil.move(str(src), str(dest))
                else:
                    shutil.copy2(str(src), str(dest))
                done += 1
                self.file_done.emit(fname, True)
                self.file_progress.emit(done, total, fname)
            except Exception as e:
                self.file_done.emit(fname, False)
                self.finished.emit(
                    False,
                    f"Failed on {fname}:\n{e}\n\n"
                    f"{done}/{total} files completed before the error.\n"
                    f"The new location has been saved — files already transferred "
                    f"are accessible, and the remaining originals are still in the "
                    f"old location."
                )
                return

        verb = "moved" if self._mode == "move" else "copied"
        self.finished.emit(True, f"{done} file(s) {verb} successfully.")


# ══════════════════════════════════════════════════════════════════════════════
#  Change-location dialog  (generic — kind-aware)
# ══════════════════════════════════════════════════════════════════════════════

class _ChangeLocationDialog(QDialog):
    """
    Asks the user where they want the library, and what to do with
    any files already installed in the current location. `kind` selects
    which singleton/settings-key pair to update: "spectral" or "astro".
    """
    def __init__(self, current_dir: Path, installed_files: List[str],
                 kind: str = "spectral", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Library Location")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(True)
        self.setMinimumWidth(520)

        self._kind         = kind
        self._current_dir  = current_dir
        self._installed    = installed_files   # filenames present on disk
        self._chosen_dir   = current_dir
        self._chosen_mode  = "move"            # 'move' | 'copy' | 'switch'

        v = QVBoxLayout(self)
        v.setSpacing(12)

        # Current path
        v.addWidget(QLabel("<b>Current location:</b>"))
        cur_lbl = QLabel(str(current_dir))
        cur_lbl.setStyleSheet("color:#6688aa; font-family:monospace; font-size:11px;")
        cur_lbl.setWordWrap(True)
        v.addWidget(cur_lbl)

        # New path picker
        v.addWidget(QLabel("<b>New location:</b>"))
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit(str(current_dir))
        self._path_edit.setReadOnly(True)
        self._path_edit.setStyleSheet(
            "background:#12121f; color:#ddd; border:1px solid #334; "
            "border-radius:3px; padding:3px 6px; font-family:monospace;")
        path_row.addWidget(self._path_edit, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(btn_browse)
        v.addLayout(path_row)

        # Migration options (only shown when there are installed files)
        if installed_files:
            size_mb = sum(
                (current_dir / f).stat().st_size / (1024 * 1024)
                for f in installed_files
                if (current_dir / f).exists()
            )
            size_str = (f"{size_mb/1024:.1f} GB" if size_mb >= 1024
                        else f"{size_mb:.0f} MB")

            v.addWidget(QLabel(
                f"<b>What to do with {len(installed_files)} installed file(s) "
                f"({size_str})?</b>"))

            self._rdo_move   = QRadioButton(
                "Move files to new location  (recommended — frees space on old drive)")
            self._rdo_copy   = QRadioButton(
                "Copy files  (keep originals — useful as backup)")
            self._rdo_switch = QRadioButton(
                "Just switch pointer  (leave files where they are)")
            self._rdo_move.setChecked(True)

            grp = QButtonGroup(self)
            for rdo in (self._rdo_move, self._rdo_copy, self._rdo_switch):
                grp.addButton(rdo)
                v.addWidget(rdo)

            self._rdo_move.toggled.connect(self._update_mode)
            self._rdo_copy.toggled.connect(self._update_mode)
            self._rdo_switch.toggled.connect(self._update_mode)
        else:
            # Nothing installed — no files to move
            self._rdo_move   = None
            self._rdo_copy   = None
            self._rdo_switch = None
            self._chosen_mode = "switch"
            v.addWidget(QLabel(
                "No files installed yet — new downloads will go to the chosen folder."))

        # Progress bar (hidden until migration starts)
        self._prog_label = QLabel("")
        self._prog_label.setStyleSheet("color:#aaa; font-size:11px;")
        self._prog_label.setVisible(False)
        v.addWidget(self._prog_label)

        self._prog_bar = QProgressBar()
        self._prog_bar.setTextVisible(True)
        self._prog_bar.setVisible(False)
        self._prog_bar.setStyleSheet("""
            QProgressBar {
                border:1px solid #333; border-radius:3px;
                background:#0a0a18; color:#aaa;
                text-align:center; font-size:10px;
            }
            QProgressBar::chunk { background:#4466cc; border-radius:2px; }
        """)
        v.addWidget(self._prog_bar)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_ok     = QPushButton("Apply")
        self._btn_ok.setFixedWidth(90)
        self._btn_ok.clicked.connect(self._apply)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedWidth(90)
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(self._btn_cancel)
        v.addLayout(btn_row)

        self._worker: Optional[_MigrateWorker] = None

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose Library Folder", str(self._chosen_dir))
        if chosen:
            self._chosen_dir = Path(chosen)
            self._path_edit.setText(chosen)

    def _update_mode(self):
        if self._rdo_move and self._rdo_move.isChecked():
            self._chosen_mode = "move"
        elif self._rdo_copy and self._rdo_copy.isChecked():
            self._chosen_mode = "copy"
        else:
            self._chosen_mode = "switch"

    def _apply(self):
        dest = self._chosen_dir

        if dest == self._current_dir:
            self.accept()
            return

        try:
            dest.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                f"Could not create directory:\n{e}")
            return

        # ── Close the relevant open SQLite connections before touching files ──
        # Windows locks open sqlite files; close the right singleton first.
        if self._kind == "spectral":
            global _library_instance
            if _library_instance is not None:
                try:
                    _library_instance.close()
                except Exception:
                    pass
                _library_instance = None
            set_library_dir(dest)
        else:
            global _astro_library_instance
            if _astro_library_instance is not None:
                try:
                    _astro_library_instance.close()
                except Exception:
                    pass
                _astro_library_instance = None
            set_astro_library_dir(dest)
        # ─────────────────────────────────────────────────────────────────────

        if self._chosen_mode == "switch" or not self._installed:
            self.accept()
            return

        # Start migration worker
        self._btn_ok.setEnabled(False)
        self._btn_cancel.setText("Cancel Migration")
        self._prog_label.setVisible(True)
        self._prog_bar.setVisible(True)
        self._prog_bar.setMaximum(len(self._installed))
        self._prog_bar.setValue(0)

        self._worker = _MigrateWorker(
            self._current_dir, dest,
            self._installed, self._chosen_mode,
            parent=None,
        )
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._btn_cancel.clicked.disconnect()
        self._btn_cancel.clicked.connect(self._cancel_migration)
        self._worker.start()

    def _cancel_migration(self):
        if self._worker:
            self._worker.cancel()

    def _on_file_progress(self, done: int, total: int, fname: str):
        self._prog_bar.setValue(done)
        self._prog_bar.setFormat(f"{done}/{total} files")
        self._prog_label.setText(f"{'Moving' if self._chosen_mode == 'move' else 'Copying'}: {fname}")

    def _on_finished(self, ok: bool, msg: str):
        if self._worker:
            self._worker.wait()
        self._worker = None

        if ok:
            self.accept()
        else:
            QMessageBox.warning(self, "Migration Incomplete", msg)
            # Still accept — the new path is already saved, partial migration is valid
            self.accept()

    def _on_cancelled(self):
        if self._worker:
            self._worker.wait()
        self._worker = None
        # New path is already saved; files transferred so far are usable
        QMessageBox.information(
            self, "Migration Cancelled",
            "Migration was cancelled. The library location has been updated — "
            "files that were already transferred are accessible at the new location. "
            "Remaining files are still in the old location and will be skipped by SASpro."
        )
        self.accept()

    def chosen_dir(self) -> Path:
        return self._chosen_dir


# ══════════════════════════════════════════════════════════════════════════════
#  Download workers  (generic — kind selects the download base URL)
# ══════════════════════════════════════════════════════════════════════════════

class _FileDownloadWorker(QThread):
    progress  = _Signal(int, int, str)
    finished  = _Signal(bool, str)
    cancelled = _Signal()

    def __init__(self, url: str, dest: Path, parent=None):
        super().__init__(parent)
        self._url    = url
        self._dest   = dest
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        tmp = self._dest.with_suffix(".tmp")
        try:
            req = Request(self._url, headers={"User-Agent": "SASpro-GaiaLibrary/1.0"})
            with urlopen(req, timeout=7200) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done  = 0
                chunk = 1024 * 1024

                with open(tmp, "wb") as f:
                    while True:
                        if self._cancel:
                            f.close()
                            tmp.unlink(missing_ok=True)
                            self.cancelled.emit()
                            return
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        pct      = int(done / total * 100) if total else 0
                        done_mb  = done  / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        self.progress.emit(
                            done, total,
                            f"{self._dest.name}  {done_mb:.0f}/{total_mb:.0f} MB  ({pct}%)"
                        )

            tmp.rename(self._dest)
            self.finished.emit(True, self._dest.name)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            self.finished.emit(False, str(e))


class _GroupDownloadWorker(QThread):
    file_progress  = _Signal(float, float, str)
    file_done      = _Signal(str, bool)
    group_progress = _Signal(int, int)
    group_finished = _Signal(bool, str)
    cancelled      = _Signal()

    def __init__(self, group: GroupDef, missing: List[str],
                 dest_dir: Path, download_base: str = LIBRARY_DOWNLOAD_BASE,
                 parent=None):
        super().__init__(parent)
        self._group         = group
        self._missing       = missing
        self._dir           = dest_dir
        self._download_base = download_base
        self._cancel        = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total_files = len(self._missing)
        done_files  = 0

        for fname in self._missing:
            if self._cancel:
                self.cancelled.emit()
                return

            url  = self._download_base + fname
            dest = self._dir / fname
            tmp  = dest.with_suffix(".tmp")

            try:
                req = Request(url, headers={"User-Agent": "SASpro-GaiaLibrary/1.0"})
                with urlopen(req, timeout=7200) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                    done  = 0
                    chunk = 1024 * 1024

                    with open(tmp, "wb") as f:
                        while True:
                            if self._cancel:
                                f.close()
                                tmp.unlink(missing_ok=True)
                                self.cancelled.emit()
                                return
                            buf = resp.read(chunk)
                            if not buf:
                                break
                            f.write(buf)
                            done += len(buf)
                            pct      = int(done / total * 100) if total else 0
                            done_mb  = done  / (1024 * 1024)
                            total_mb = total / (1024 * 1024)
                            self.file_progress.emit(
                                done_mb, total_mb,
                                f"[{done_files+1}/{total_files}]  {fname}\n"
                                f"{done_mb:.0f} / {total_mb:.0f} MB  ({pct}%)"
                            )

                tmp.rename(dest)
                done_files += 1
                self.file_done.emit(fname, True)
                self.group_progress.emit(done_files, total_files)

            except Exception as e:
                tmp.unlink(missing_ok=True)
                self.file_done.emit(fname, False)
                self.group_finished.emit(
                    False,
                    f"Failed to download {fname}:\n{e}\n\n"
                    f"{done_files}/{total_files} files completed before failure.\n"
                    f"You can retry — already-downloaded files will be skipped."
                )
                return

        self.group_finished.emit(
            True,
            f"{self._group.label} installed successfully "
            f"({total_files} file{'s' if total_files != 1 else ''})."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Spectrum viewer widget  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class SpectrumViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MPL:
            self._fig    = Figure(figsize=(6, 3), facecolor="#1a1a2e")
            self._ax     = self._fig.add_subplot(111)
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setMinimumHeight(220)
            layout.addWidget(self._canvas)
            self._draw_empty()
        else:
            lbl = QLabel("matplotlib required for spectrum viewer")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)

        self._info_lbl = QLabel("")
        self._info_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._info_lbl)

    def _draw_empty(self):
        if not HAS_MPL:
            return
        self._ax.clear()
        self._ax.set_facecolor("#1a1a2e")
        self._ax.text(0.5, 0.5, "No spectrum loaded",
                      transform=self._ax.transAxes,
                      ha="center", va="center", color="#555", fontsize=12,
                      usetex=False)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for sp in self._ax.spines.values():
            sp.set_color("#333")
        try:
            self._fig.tight_layout(pad=0.5)
        except Exception:
            pass
        self._canvas.draw()

    def show_spectrum(self, spectrum: CalibratedSpectrum, title: str = ""):
        if not HAS_MPL:
            return
        wl   = spectrum.wavelengths
        flux = spectrum.flux

        self._ax.clear()
        self._ax.set_facecolor("#1a1a2e")
        self._fig.patch.set_facecolor("#1a1a2e")

        bp = wl <= 680
        rp = wl >= 640

        if np.any(bp):
            self._ax.fill_between(wl[bp], flux[bp], alpha=0.25, color="#5599ff")
            self._ax.plot(wl[bp], flux[bp], color="#88bbff", lw=1.2)
        if np.any(rp):
            self._ax.fill_between(wl[rp], flux[rp], alpha=0.25, color="#ff6644")
            self._ax.plot(wl[rp], flux[rp], color="#ffaa88", lw=1.2)

        if spectrum.flux_error is not None:
            self._ax.fill_between(wl,
                                  flux - spectrum.flux_error,
                                  flux + spectrum.flux_error,
                                  alpha=0.12, color="#ffffff")

        self._ax.set_xlabel("Wavelength (nm)", color="#aaa", fontsize=9)
        self._ax.set_ylabel("Flux (W nm⁻¹ m⁻²)", color="#aaa", fontsize=9)
        self._ax.tick_params(colors="#888", labelsize=8)
        for sp in self._ax.spines.values():
            sp.set_color("#333")
        self._ax.set_xlim(330, 1030)
        if title:
            self._ax.set_title(title, color="#ccc", fontsize=9, pad=4)

        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()

        peak_wl = float(wl[np.argmax(flux)])
        self._info_lbl.setText(
            f"source_id: {spectrum.source_id}   peak: {peak_wl:.0f} nm   "
            f"flux: {float(np.min(flux)):.2e} – {float(np.max(flux)):.2e}"
        )

    def clear(self):
        self._draw_empty()
        self._info_lbl.setText("")


# ══════════════════════════════════════════════════════════════════════════════
#  Group row widget  (generic — unit_label swaps "spectra" / "stars" in text)
# ══════════════════════════════════════════════════════════════════════════════

class _GroupRowWidget(QFrame):
    install_requested = _Signal(str)
    remove_requested  = _Signal(str)
    browse_requested  = _Signal(str)

    def __init__(self, status: GroupStatus, parent=None):
        super().__init__(parent)
        self._key = status.group.key
        self._build_ui()
        self.update_status(status)

    def _build_ui(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                border: 1px solid #2a2a3e;
                border-radius: 6px;
                background: #12121f;
            }
        """)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(12)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._dot)

        info = QVBoxLayout()
        info.setSpacing(3)

        title_row = QHBoxLayout()
        self._lbl_title = QLabel()
        self._lbl_title.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #e0e0ff; border: none;")
        title_row.addWidget(self._lbl_title)

        self._lbl_badge = QLabel()
        self._lbl_badge.setStyleSheet(
            "font-size: 10px; color: #44cc88; font-weight: bold; "
            "border: 1px solid #44cc88; border-radius: 3px; padding: 1px 5px;")
        self._lbl_badge.setVisible(False)
        title_row.addWidget(self._lbl_badge)
        title_row.addStretch()
        info.addLayout(title_row)

        self._lbl_desc = QLabel()
        self._lbl_desc.setStyleSheet("font-size: 11px; color: #888; border: none;")
        self._lbl_desc.setWordWrap(True)
        info.addWidget(self._lbl_desc)

        self._lbl_warning = QLabel()
        self._lbl_warning.setStyleSheet(
            "font-size: 11px; color: #ddaa44; border: none;")
        self._lbl_warning.setWordWrap(True)
        self._lbl_warning.setVisible(False)
        info.addWidget(self._lbl_warning)

        self._lbl_stats = QLabel()
        self._lbl_stats.setStyleSheet(
            "font-size: 11px; color: #6688aa; border: none;")
        info.addWidget(self._lbl_stats)

        outer.addLayout(info, stretch=1)

        self._progress = QProgressBar()
        self._progress.setFixedWidth(200)
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        self._progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333; border-radius: 3px;
                background: #0a0a18; color: #aaa;
                text-align: center; font-size: 9px;
            }
            QProgressBar::chunk { background: #4466cc; border-radius: 2px; }
        """)
        outer.addWidget(self._progress)

        self._lbl_file_count = QLabel()
        self._lbl_file_count.setStyleSheet(
            "font-size: 10px; color: #6688aa; border: none;")
        self._lbl_file_count.setVisible(False)
        outer.addWidget(self._lbl_file_count)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self._btn_install = QPushButton("Install")
        self._btn_install.setFixedWidth(80)
        self._btn_install.clicked.connect(
            lambda: self.install_requested.emit(self._key))
        self._btn_install.setStyleSheet(self._btn_style("#224488", "#3366cc"))

        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.setFixedWidth(80)
        self._btn_browse.clicked.connect(
            lambda: self.browse_requested.emit(self._key))
        self._btn_browse.setStyleSheet(self._btn_style("#223322", "#336633"))

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setFixedWidth(80)
        self._btn_cancel.setVisible(False)
        self._btn_cancel.setStyleSheet(self._btn_style("#443322", "#885533"))

        self._btn_remove = QPushButton("Remove")
        self._btn_remove.setFixedWidth(80)
        self._btn_remove.clicked.connect(
            lambda: self.remove_requested.emit(self._key))
        self._btn_remove.setStyleSheet(self._btn_style("#442222", "#883333"))

        btn_layout.addWidget(self._btn_install)
        btn_layout.addWidget(self._btn_browse)
        btn_layout.addWidget(self._btn_cancel)
        btn_layout.addWidget(self._btn_remove)
        outer.addLayout(btn_layout)

    @staticmethod
    def _btn_style(bg, hover):
        return (f"QPushButton {{ background:{bg}; color:#ddd; border:1px solid #444; "
                f"border-radius:4px; padding:4px 8px; font-size:11px; }}"
                f"QPushButton:hover {{ background:{hover}; }}"
                f"QPushButton:disabled {{ color:#555; background:#1a1a2e; }}")

    def update_status(self, status: GroupStatus):
        g = status.group
        unit = getattr(g, "unit_label", "spectra")

        self._lbl_title.setText(g.label)
        self._lbl_desc.setText(g.description)

        if g.recommended:
            self._lbl_badge.setText("Recommended")
            self._lbl_badge.setVisible(True)

        if g.warning:
            self._lbl_warning.setText(f"⚠  {g.warning}")
            self._lbl_warning.setVisible(True)

        if status.fully_installed:
            self._dot.setStyleSheet("color: #44cc66; font-size: 14px;")
            # Prefer the exact build-time count when we have it; only fall back
            # to a live COUNT(*) result (spectral tab) when we don't.
            n = g.n_items if g.n_items is not None else status.total_spectra
            n_str = f"{n:,} {unit}" if n else f"— {unit}"
            self._lbl_stats.setText(
                f"{n_str}  ·  "
                f"{status.total_mb / 1024:.1f} GB on disk  ·  "
                f"{len(status.installed)}/{len(g.filenames)} files"
            )
            self._btn_install.setVisible(False)
            self._btn_browse.setVisible(False)
            self._btn_remove.setVisible(True)

        elif status.partially_installed:
            self._dot.setStyleSheet("color: #ddaa44; font-size: 14px;")
            # Partial: a build-time total would be misleading (not all files are
            # here), so report files + disk only.
            counted = (f"{status.total_spectra:,} {unit}  ·  "
                       if status.total_spectra else "")
            self._lbl_stats.setText(
                f"Partial: {len(status.installed)}/{len(g.filenames)} files  ·  "
                f"{counted}"
                f"{status.total_mb / 1024:.1f} GB on disk  —  "
                f"click Install to resume"
            )
            self._btn_install.setText("Resume")
            self._btn_install.setVisible(True)
            self._btn_browse.setVisible(True)
            self._btn_remove.setVisible(True)

        else:
            self._dot.setStyleSheet("color: #555; font-size: 14px;")
            self._lbl_stats.setText(
                f"Not installed  ·  {g.est_stars}  ·  {g.est_size}"
            )
            self._btn_install.setText("Install")
            self._btn_install.setVisible(True)
            self._btn_browse.setVisible(True)
            self._btn_remove.setVisible(False)

    def set_downloading(self, downloading: bool, cancel_cb=None):
        self._btn_install.setEnabled(not downloading)
        self._btn_browse.setEnabled(not downloading)
        self._btn_remove.setEnabled(not downloading)
        self._progress.setVisible(downloading)
        self._lbl_file_count.setVisible(downloading)
        self._btn_cancel.setVisible(downloading)
        if cancel_cb:
            try:
                self._btn_cancel.clicked.disconnect()
            except Exception:
                pass
            self._btn_cancel.clicked.connect(cancel_cb)
        if not downloading:
            self._progress.setValue(0)
            self._lbl_file_count.setText("")

    def update_file_progress(self, done_mb: float, total_mb: float, msg: str):
        try:
            from PyQt6 import sip
            if sip.isdeleted(self) or sip.isdeleted(self._progress):
                return
        except Exception:
            return
        self._progress.setVisible(True)
        if total_mb > 0:
            total_kb = int(total_mb * 1024)
            done_kb  = int(done_mb  * 1024)
            self._progress.setMaximum(total_kb)
            self._progress.setValue(done_kb)
            self._progress.setFormat(f"{done_mb:.0f}/{total_mb:.0f} MB  ({int(done_mb/total_mb*100)}%)")
        else:
            self._progress.setMaximum(0)
            self._progress.setValue(0)
            self._progress.setFormat("")
            self._lbl_file_count.setVisible(True)
            self._lbl_file_count.setText(f"{done_mb:,.0f} MB")

    def update_group_progress(self, done: int, total: int):
        try:
            from PyQt6 import sip
            if sip.isdeleted(self) or sip.isdeleted(self._progress):
                return
        except Exception:
            return
        if self._progress.maximum() > 0:
            self._lbl_file_count.setText(f"{done}/{total} files")
        else:
            current = self._lbl_file_count.text()
            mb_part = current.split("·")[0].strip() if "·" in current else current
            self._lbl_file_count.setText(f"{mb_part}  ·  {done}/{total} files")


# ══════════════════════════════════════════════════════════════════════════════
#  True star colour from XP spectra
# ══════════════════════════════════════════════════════════════════════════════
#
# The XP library's `sources.bp_rp` column is NULL and its `synth_phot` table is
# empty (never populated by the bulk builder), so there is no shortcut: to get
# a real colour we integrate the actual 343-point spectrum against the CIE 1931
# colour matching functions. That costs a zlib.decompress + a few dot products
# per star, which is fine for thousands and hopeless for millions — hence the
# star cap and the worker thread.

# np.trapz was REMOVED in NumPy 2.0. SASpro pins numpy = "*", so bind whichever
# name exists rather than assuming.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _cie_cmf(wl: np.ndarray):
    """
    CIE 1931 2° colour matching functions via Wyman, Sloan & Shirley (2013),
    JCGT — multi-lobe piecewise-Gaussian fits. ~1% accurate, no lookup table.
    """
    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    xb = (1.056 * g(wl, 599.8, 37.9, 31.0)
          + 0.362 * g(wl, 442.0, 16.0, 26.7)
          - 0.065 * g(wl, 501.1, 20.4, 26.2))
    yb = (0.821 * g(wl, 568.8, 46.9, 40.5)
          + 0.286 * g(wl, 530.9, 16.3, 31.1))
    zb = (1.217 * g(wl, 437.0, 11.8, 36.0)
          + 0.681 * g(wl, 459.0, 26.0, 13.8))
    return xb, yb, zb


_CMF_X, _CMF_Y, _CMF_Z = _cie_cmf(WL_GRID.astype(np.float64))

# The Sun's north rotational pole (IAU: RA 286.13°, Dec +63.87°, ICRS),
# expressed as a unit vector in the same galactic frame the explorer plots in.
# Verified: the angle between this and the ecliptic pole comes out to 7.25°,
# the Sun's known obliquity — which cross-checks both the pole coordinates and
# the ICRS->galactic rotation.
#
# Note this is a *tilt*, not a direction of motion. It says how the solar
# system is oriented relative to the galactic disk. Nothing travels along it.
SOLAR_NORTH_POLE_GAL = np.array([-0.0716, +0.9193, +0.3870], dtype=np.float32)

# XYZ -> linear sRGB (D65)
_XYZ_TO_RGB = np.array([[ 3.2406, -1.5372, -0.4986],
                        [-0.9689,  1.8758,  0.0415],
                        [ 0.0557, -0.2040,  1.0570]])


def spectrum_to_srgb(flux: np.ndarray) -> np.ndarray:
    """
    One XP spectrum (343 samples, 336–1020 nm) -> sRGB in 0..1.

    Luminance is normalised away: we keep only chromaticity, because apparent
    brightness is already encoded in the point size. Without this, every star
    would render as some shade of "very dim" and the colour would be invisible.

    Verified against blackbodies: 2500 K -> deep orange, 5772 K -> near-white,
    10000 K -> blue, with B/R strictly increasing in temperature.
    """
    X = _trapz(flux * _CMF_X, WL_GRID)
    Y = _trapz(flux * _CMF_Y, WL_GRID)
    Z = _trapz(flux * _CMF_Z, WL_GRID)
    if not np.isfinite(Y) or Y <= 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    xyz = np.array([X, Y, Z], dtype=np.float64) / Y
    lin = _XYZ_TO_RGB @ xyz
    lin = np.clip(lin, 0.0, None)          # clip out-of-gamut negatives
    peak = lin.max()
    if peak > 0:
        lin = lin / peak

    srgb = np.where(lin <= 0.0031308,
                    12.92 * lin,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0).astype(np.float32)


class _SpectrumColourWorker(QThread):
    """
    Resolves true colours for a set of Gaia source_ids off the UI thread.

    get_spectrum(sid) is a primary-key hit, but it may probe each installed XP
    file in turn, and every hit costs a zlib.decompress. So we cap the work and
    spend it on the brightest stars, where colour is most visible anyway.
    """
    finished_colours = _Signal(object, object)   # rgb (N,3) float32, found mask (N,) bool
    failed           = _Signal(str)

    def __init__(self, source_ids: np.ndarray, order: np.ndarray, parent=None):
        super().__init__(parent)
        self._sids  = source_ids
        self._order = order          # indices to attempt, brightest first
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            lib = get_library()
            if not lib.installed_bands():
                self.failed.emit("No XP spectral tiers installed.")
                return

            n = len(self._sids)
            rgb   = np.zeros((n, 3), dtype=np.float32)
            found = np.zeros(n, dtype=bool)

            for i in self._order:
                if self._cancel:
                    return
                sid = int(self._sids[i])
                try:
                    spec = lib.get_spectrum(sid)
                except Exception:
                    spec = None
                if spec is None:
                    continue
                rgb[i]   = spectrum_to_srgb(spec.flux.astype(np.float64))
                found[i] = True

            self.finished_colours.emit(rgb, found)
        except Exception as e:
            self.failed.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  Stellar Neighborhood Explorer  (3D)
# ══════════════════════════════════════════════════════════════════════════════

def _star_colors(gmag: np.ndarray, dist_pc: np.ndarray, mode: str) -> np.ndarray:
    """
    Map a per-star scalar to RGBA. No matplotlib dependency — a hand-rolled
    blue-white-orange ramp that reads roughly like stellar luminosity.

    mode:
      "absmag"  — absolute G, M_G = G - 5*log10(d) + 5. A luminosity proxy:
                  low (bright/luminous) -> blue-white, high (dim dwarfs) -> orange.
                  NOTE: without BP-RP this is NOT temperature. A luminous red
                  giant and a hot blue dwarf can share an absolute magnitude.
      "appmag"  — apparent G, i.e. how bright it looks from here.
      "dist"    — distance from the Sun.
    """
    n = gmag.shape[0]
    if n == 0:
        return np.empty((0, 4), dtype=np.float32)

    if mode == "dist":
        v = dist_pc.astype(np.float64)
    elif mode == "appmag":
        v = gmag.astype(np.float64)
    else:  # absmag
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.maximum(dist_pc.astype(np.float64), 1e-6)
            v = gmag.astype(np.float64) - 5.0 * np.log10(d) + 5.0

    finite = np.isfinite(v)
    if not finite.any():
        out = np.tile(np.array([1, 1, 1, 0.9], dtype=np.float32), (n, 1))
        return out

    # Robust normalization: 5th-95th percentile so a couple of outliers don't
    # flatten the whole ramp.
    lo, hi = np.percentile(v[finite], [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(v[finite]), np.nanmax(v[finite])
        if hi <= lo:
            hi = lo + 1.0
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    t[~finite] = 0.5   # unknown magnitude -> mid ramp

    # ramp: 0 -> blue-white (luminous), 0.5 -> white/cream, 1 -> orange-red (dim)
    c0 = np.array([0.68, 0.80, 1.00])
    c1 = np.array([1.00, 0.99, 0.92])
    c2 = np.array([1.00, 0.52, 0.28])
    tt = t[:, None]
    lower = c0 + (c1 - c0) * (tt / 0.5)
    upper = c1 + (c2 - c1) * ((tt - 0.5) / 0.5)
    rgb = np.where(tt < 0.5, lower, upper)

    rgba = np.empty((n, 4), dtype=np.float32)
    rgba[:, :3] = rgb
    rgba[:, 3] = 0.85
    return rgba


def _star_sizes(gmag: np.ndarray, base: float = 4.0) -> np.ndarray:
    """Apparent brightness -> point size in pixels. Brighter stars render bigger."""
    n = gmag.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)
    g = gmag.astype(np.float64)
    finite = np.isfinite(g)
    if not finite.any():
        return np.full(n, base, dtype=np.float32)
    lo, hi = np.percentile(g[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((g - lo) / (hi - lo), 0.0, 1.0)   # 0 = bright, 1 = faint
    t[~finite] = 0.8
    sizes = base * (2.6 - 2.1 * t)                # bright ~2.6x base, faint ~0.5x
    return np.maximum(sizes, 1.2).astype(np.float32)


_bright_cache: Optional[Dict[str, np.ndarray]] = None
_BRIGHT_LOAD_ERROR = ""


def load_bright_stars() -> Optional[Dict[str, np.ndarray]]:
    """
    The hardcoded bright stars Gaia can't see (it saturates below G≈3), built by
    build_bright_stars_3d.py. Returns arrays shaped like a neighborhood load, or
    None if the module isn't present.

    Their `source_id` is NEGATIVE — a sentinel meaning "not a Gaia source". The
    explorer keys off `is_bright` rather than the sign, but a negative id makes
    an accidental Gaia lookup fail loudly instead of silently hitting the wrong
    star.
    """
    global _bright_cache, _BRIGHT_LOAD_ERROR
    if _bright_cache is not None:
        return _bright_cache
    try:
        from setiastro.saspro.bright_stars_3d import BRIGHT_STARS_3D
    except Exception as e:
        _BRIGHT_LOAD_ERROR = str(e)
        return None

    n = len(BRIGHT_STARS_3D)
    if n == 0:
        return None

    d = {
        "source_id": np.arange(-1, -(n + 1), -1, dtype=np.int64),
        "xyz":       np.empty((n, 3), dtype=np.float32),
        "dist_pc":   np.empty(n, dtype=np.float32),
        "gmag":      np.empty(n, dtype=np.float32),   # V mag; used for sizing
        "ruwe":      np.full(n, np.nan, dtype=np.float32),
        "ra":        np.empty(n, dtype=np.float64),
        "dec":       np.empty(n, dtype=np.float64),
        "rgb":       np.empty((n, 3), dtype=np.float32),
        "is_bright": np.ones(n, dtype=bool),
        "name":      np.empty(n, dtype=object),
        "simbad_id": np.empty(n, dtype=object),
        "sp_type":   np.empty(n, dtype=object),
        "teff":      np.empty(n, dtype=np.float32),
        "plx_ok":    np.empty(n, dtype=bool),
    }
    for i, row in enumerate(BRIGHT_STARS_3D):
        (name, sid, ra, dec, vmag, dist, plx, plx_err, plx_ok,
         x, y, z, r, g, b, teff, sp) = row
        d["xyz"][i]     = (x, y, z)
        d["dist_pc"][i] = dist
        d["gmag"][i]    = vmag
        d["ra"][i]      = ra
        d["dec"][i]     = dec
        d["rgb"][i]     = (r, g, b)
        d["name"][i]      = name
        d["simbad_id"][i] = sid
        d["sp_type"][i]   = sp
        d["teff"][i]      = teff
        d["plx_ok"][i]    = bool(plx_ok)

    _bright_cache = d
    return d


def _merge_bright(gaia: Dict[str, np.ndarray], max_pc: float) -> Dict[str, np.ndarray]:
    """
    Append the hardcoded bright stars (within max_pc) to a neighborhood load.

    No dedup pass: by construction these stars are absent from gaia_source
    (Gaia saturates on them), and the one nominal overlap — Arneb, brightest
    source in the XP library — sits at 680 pc, far outside any radius the
    extract covers. If a future DR does start resolving them, this is where a
    positional dedup would go.
    """
    bright = load_bright_stars()
    n_g = len(gaia["source_id"])

    # Give the Gaia rows the extra columns the bright stars carry.
    out = dict(gaia)
    out["is_bright"] = np.zeros(n_g, dtype=bool)
    out["rgb"]       = np.full((n_g, 3), np.nan, dtype=np.float32)
    out["name"]      = np.full(n_g, None, dtype=object)
    out["simbad_id"] = np.full(n_g, None, dtype=object)
    out["sp_type"]   = np.full(n_g, None, dtype=object)
    out["teff"]      = np.zeros(n_g, dtype=np.float32)
    out["plx_ok"]    = np.ones(n_g, dtype=bool)

    if bright is None:
        return out

    keep = bright["dist_pc"] <= max_pc
    if not keep.any():
        return out

    for k in out:
        if k == "xyz":
            out[k] = np.vstack([out[k], bright[k][keep]])
        else:
            out[k] = np.concatenate([out[k], bright[k][keep]])

    # Nearest first, so Alpha Cen / Sirius head the table as they should.
    order = np.argsort(out["dist_pc"], kind="stable")
    for k in out:
        out[k] = out[k][order]
    return out


class _NeighborhoodLoadWorker(QThread):
    """Runs the (potentially large) sphere query off the UI thread."""
    loaded = _Signal(object)
    failed = _Signal(str)

    def __init__(self, max_pc: float, max_ruwe: Optional[float],
                 max_gmag: Optional[float], limit: int, parent=None):
        super().__init__(parent)
        self._max_pc   = max_pc
        self._max_ruwe = max_ruwe
        self._max_gmag = max_gmag
        self._limit    = limit

    def run(self):
        try:
            lib = get_neighborhood_library()
            data = lib.load_sphere(self._max_pc, self._max_ruwe,
                                   self._max_gmag, self._limit)
            if data is None:
                self.failed.emit("Neighborhood library is not installed.")
                return
            self.loaded.emit(data)
        except Exception as e:
            self.failed.emit(str(e))


if HAS_GL:

    class _PickableGLView(gl.GLViewWidget):
        """
        GLViewWidget that reports which scatter point was clicked.

        pyqtgraph has no built-in 3D picking, so we project every displayed
        point through the current MVP matrix and take the nearest hit within a
        pixel radius. Cheap enough at the point counts this tab renders, and it
        avoids a colour-picking render pass.

        Click vs. drag is distinguished by cursor travel: an orbit drag moves
        several pixels, a selection click does not.
        """
        pointPicked      = _Signal(int)          # index into the current point array
        pointRightClicked = _Signal(int, object)  # index, global QPoint

        def __init__(self, parent=None):
            super().__init__(parent)
            self._pick_pts: Optional[np.ndarray] = None    # (N,3) float32
            self._press_pos = None

        def set_pick_points(self, pts: Optional[np.ndarray]):
            self._pick_pts = pts

        def mousePressEvent(self, ev):
            self._press_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
            super().mousePressEvent(ev)

        def mouseReleaseEvent(self, ev):
            # base class mouseReleaseEvent is a no-op; safe to not call through
            pos = ev.position() if hasattr(ev, "position") else ev.localPos()
            if self._press_pos is not None:
                dx = pos.x() - self._press_pos.x()
                dy = pos.y() - self._press_pos.y()
                if (dx * dx + dy * dy) <= 16.0:      # < 4 px travel => a click
                    idx = self._pick_index(pos)
                    if idx is not None:
                        btn = ev.button()
                        if btn == Qt.MouseButton.LeftButton:
                            self.pointPicked.emit(idx)
                        elif btn == Qt.MouseButton.RightButton:
                            if hasattr(ev, "globalPosition"):
                                gp = ev.globalPosition().toPoint()
                            else:
                                gp = ev.globalPos()
                            self.pointRightClicked.emit(idx, gp)
            self._press_pos = None

        def _mvp(self) -> Optional[np.ndarray]:
            """
            Model-view-projection matrix, across pyqtgraph versions.

            0.13.x:  projectionMatrix(region=None)      -- derives the viewport
            0.14.0:  projectionMatrix(region, viewport) -- both REQUIRED

            Calling the 0.13 form on 0.14 raises TypeError. That used to be
            swallowed by a bare except, so _pick_index returned None and every
            click silently did nothing. Try the new signature first, fall back
            to the old, and report anything else rather than hiding it.
            """
            try:
                viewport = self.getViewport()
            except Exception:
                viewport = (0, 0, self.width(), self.height())

            proj_q = None
            try:
                # pyqtgraph >= 0.14: region and viewport are required.
                proj_q = self.projectionMatrix(viewport, viewport)
            except TypeError:
                try:
                    proj_q = self.projectionMatrix()      # pyqtgraph <= 0.13
                except Exception as e:
                    print(f"[Neighborhood] projectionMatrix failed: {e}")
                    return None
            except Exception as e:
                print(f"[Neighborhood] projectionMatrix failed: {e}")
                return None

            try:
                view_q = self.viewMatrix()
                # QMatrix4x4.data() is COLUMN-major; reshape gives the transpose.
                proj = np.array(proj_q.data(), dtype=np.float64).reshape(4, 4).T
                view = np.array(view_q.data(), dtype=np.float64).reshape(4, 4).T
                return proj @ view
            except Exception as e:
                print(f"[Neighborhood] viewMatrix failed: {e}")
                return None

        def _pick_index(self, pos) -> Optional[int]:
            """Nearest displayed point to `pos`, or None if nothing within 12 px."""
            pts = self._pick_pts
            if pts is None or len(pts) == 0:
                return None
            mvp = self._mvp()
            if mvp is None:
                return None

            h = np.empty((pts.shape[0], 4), dtype=np.float64)
            h[:, :3] = pts
            h[:, 3] = 1.0
            clip = h @ mvp.T

            w = clip[:, 3]
            in_front = w > 1e-9
            if not in_front.any():
                return None

            ndc = np.empty((pts.shape[0], 2), dtype=np.float64)
            ndc[:, 0] = clip[:, 0] / np.where(in_front, w, 1.0)
            ndc[:, 1] = clip[:, 1] / np.where(in_front, w, 1.0)

            sx = (ndc[:, 0] * 0.5 + 0.5) * self.width()
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * self.height()

            d2 = (sx - pos.x()) ** 2 + (sy - pos.y()) ** 2
            d2[~in_front] = np.inf

            i = int(np.argmin(d2))
            if np.isfinite(d2[i]) and d2[i] <= (12.0 ** 2):   # 12 px tolerance
                return i
            return None


class StellarNeighborhoodTab(QWidget):
    """
    Interactive 3D view of every Gaia DR3 star with a usable parallax inside a
    user-chosen radius. Sun at the origin, galactic plane in XY, +Z toward the
    north galactic pole.
    """

    # Emitted when the user asks to see a star's XP spectrum. The dialog owns
    # the Spectrum Viewer tab, so it handles the actual switch + lookup.
    spectrumRequested = _Signal(int)   # gaia source_id

    # Opens small on purpose: within a few parsecs there are only a handful of
    # stars, so the first render is instant and obviously correct (you should
    # recognise the names). Expand from there.
    DEFAULT_RADIUS_PC = 3.0
    MAX_POINTS        = 2_000_000
    TABLE_ROWS        = 300
    # Ceiling on stars we'll fetch+decompress spectra for. Each costs a PK
    # lookup and a zlib.decompress; 20k is a couple of seconds, millions is not
    # happening. Spent on the brightest stars, where colour actually reads.
    MAX_SPECTRUM_STARS = 20_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._worker: Optional[_NeighborhoodLoadWorker] = None
        self._selected_idx: Optional[int] = None

        # True-colour state. _spec_rgb/_spec_found are aligned with self._data.
        self._colour_worker: Optional[_SpectrumColourWorker] = None
        self._spec_rgb:   Optional[np.ndarray] = None    # (N,3) float32
        self._spec_found: Optional[np.ndarray] = None    # (N,) bool
        # source_id -> rgb, so re-entering the mode (or reloading the same
        # radius) doesn't re-decompress spectra we've already integrated.
        self._colour_cache: Dict[int, np.ndarray] = {}
        # True when the Gaia neighborhood extract isn't installed and we're
        # rendering the hardcoded naked-eye stars alone.
        self._gaia_missing = False

        # The GL widget is created lazily, on first show. Constructing a
        # QOpenGLWidget is what can force Qt to recreate the top-level native
        # window (destroying dock layouts), so we don't do it just because the
        # Gaia dialog was opened — only when this tab is actually visited.
        self._view = None
        self._grid = None
        self._scatter = None
        self._sun = None
        self._marker = None
        self._droplines = None
        self._axes = None
        self._axis_labels = []
        self._gl_ready = False
        self._view_slot = None
        self._view_container = None
        self._view_placeholder = None
        self._gl_build_count = 0

        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(300)
        self._reload_timer.timeout.connect(self._reload)

        self._build_ui()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not HAS_GL:
            return
        # First visit to this tab builds the GL view. That is when Qt rebuilds
        # the dialog's native window — a one-off flicker of *this dialog*,
        # which is the price of not flickering the SASpro main window.
        # _ensure_gl() is idempotent; later visits do nothing.
        self._ensure_gl()
        busy = self._worker is not None and self._worker.isRunning()
        if self._data is None and not busy:
            self._reload()

    def _ensure_gl(self):
        """
        Build the OpenGL view on first display of this tab.

        _gl_ready is set FIRST, not last: addWidget() and the pyqtgraph item
        constructors can pump the event loop, which could re-deliver showEvent
        and re-enter this method before it finished. Setting the flag up front
        makes that a no-op instead of a second stacked viewport.
        """
        if self._gl_ready or not HAS_GL:
            return
        self._gl_ready = True

        self._gl_build_count = getattr(self, "_gl_build_count", 0) + 1
        if self._gl_build_count > 1:
            print(f"[Neighborhood] WARNING: _ensure_gl ran "
                  f"{self._gl_build_count}x — this should never happen")

        # Nuke whatever is in the container (placeholder, or a stale view).
        while self._view_slot.count():
            item = self._view_slot.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._view_placeholder = None

        self._view = _PickableGLView()
        self._view.setMinimumSize(280, 260)
        self._view.setCameraPosition(distance=self.DEFAULT_RADIUS_PC * 3.0)
        self._view.pointPicked.connect(self._on_point_picked)
        self._view.pointRightClicked.connect(self._on_point_right_clicked)

        self._grid = gl.GLGridItem()
        self._view.addItem(self._grid)

        # Drop lines: a vertical segment from each star down to the galactic
        # plane (z=0). Without them a 3D scatter is genuinely ambiguous — a
        # star high on screen could be far above the plane or merely distant.
        # mode='lines' means each consecutive vertex PAIR is one segment.
        self._droplines = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode="lines",
            width=1.0, antialias=True)
        self._view.addItem(self._droplines)

        # Galactic axes: +X to the galactic centre, +Y along galactic rotation
        # (l = 90°), +Z to the north galactic pole. The grid is the galactic
        # midplane — NOT the ecliptic, which sits ~60° to it.
        self._axes = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), mode="lines",
            width=2.0, antialias=True)
        self._view.addItem(self._axes)

        self._axis_labels = []
        for _ in range(4):
            t = gl.GLTextItem(pos=np.zeros(3), text="", color=(150, 170, 200, 255))
            self._view.addItem(t)
            self._axis_labels.append(t)

        self._scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), pxMode=True)
        self._view.addItem(self._scatter)

        # Sun marker at the origin
        self._sun = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=np.array([[1.0, 0.92, 0.35, 1.0]], dtype=np.float32),
            size=np.array([16.0], dtype=np.float32), pxMode=True)
        self._view.addItem(self._sun)

        # Highlight for the selected star
        self._marker = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=np.array([[0.35, 1.0, 1.0, 1.0]], dtype=np.float32),
            size=np.array([18.0], dtype=np.float32), pxMode=True)
        self._view.addItem(self._marker)

        self._view_slot.addWidget(self._view)

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        if not HAS_GL:
            msg = QLabel(
                "<b>3D view unavailable.</b><br><br>"
                "The Stellar Neighborhood Explorer needs <code>pyqtgraph.opengl</code> "
                "and <code>PyOpenGL</code>.<br><br>"
                f"<span style='color:#a66;'>Import error: {_GL_IMPORT_ERROR}</span><br><br>"
                "Install with:<br><code>pip install PyOpenGL</code>"
            )
            msg.setWordWrap(True)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet("color:#aab; font-size:12px;")
            root.addWidget(msg, stretch=1)
            return

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(6)
        body.setStyleSheet(
            "QSplitter::handle { background:#1a1a2e; }"
            "QSplitter::handle:hover { background:#2a2a4e; }")
        self._splitter = body
        root.addWidget(body, stretch=1)

        # ── 3D view container (filled lazily on first show) ──
        # A real QWidget, not a bare layout: we clear its layout before
        # inserting the GL view, so a second _ensure_gl() can never stack a
        # second viewport on top of the first.
        self._view_container = QWidget()
        self._view_container.setMinimumSize(280, 260)
        self._view_slot = QVBoxLayout(self._view_container)
        self._view_slot.setContentsMargins(0, 0, 0, 0)

        self._view_placeholder = QLabel("Initialising 3D view…")
        self._view_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view_placeholder.setStyleSheet(
            "color:#556; font-size:12px; background:#0a0a12; "
            "border:1px solid #2a2a3e; border-radius:4px;")
        self._view_slot.addWidget(self._view_placeholder)
        body.addWidget(self._view_container)

        # ── side panel ──
        # QSplitter takes widgets, not layouts — so the side column gets a host
        # widget. Everything below still does side.addWidget(...) unchanged.
        self._side_container = QWidget()
        side = QVBoxLayout(self._side_container)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)

        # The side stack (controls + table + info + buttons) has a ~500px natural
        # minimum that cannot compress. Scroll it rather than letting it dictate
        # the dialog's height.
        self._side_scroll = QScrollArea()
        self._side_scroll.setWidgetResizable(True)
        self._side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._side_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        self._side_scroll.setWidget(self._side_container)
        self._side_scroll.setMinimumWidth(300)
        body.addWidget(self._side_scroll)

        # 3D view gets the growth; the side panel keeps its width on resize.
        body.setStretchFactor(0, 3)
        body.setStretchFactor(1, 1)

        ctrl = QFrame()
        ctrl.setStyleSheet("QFrame { border:1px solid #2a2a3e; border-radius:6px; "
                           "background:#12121f; }")
        cl = QVBoxLayout(ctrl)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(6)

        # radius
        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Radius (pc):"))
        self._spin_radius = QDoubleSpinBox()
        # The extract stops at NEIGHBORHOOD_MAX_PC, but the hardcoded bright
        # stars run out to ~680 pc (Arneb). Allow the radius past the extract's
        # limit; the status line says what you are and aren't seeing out there.
        self._spin_radius.setRange(0.5, 750.0)
        self._spin_radius.setDecimals(1)
        self._spin_radius.setSingleStep(1.0)
        self._spin_radius.setValue(self.DEFAULT_RADIUS_PC)
        self._spin_radius.valueChanged.connect(self._schedule_reload)
        r_row.addWidget(self._spin_radius)
        r_row.addStretch()
        cl.addLayout(r_row)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for pc in (3, 10, 25, 50, 100, 300, 750):
            b = QPushButton(str(pc))
            b.setFixedWidth(38)
            b.setStyleSheet("QPushButton{background:#1a1a2e;color:#9ab;border:1px solid "
                            "#334;border-radius:3px;padding:2px;font-size:10px;}"
                            "QPushButton:hover{background:#222240;color:#cde;}")
            b.clicked.connect(lambda _=False, v=pc: self._spin_radius.setValue(float(v)))
            preset_row.addWidget(b)
        preset_row.addStretch()
        cl.addLayout(preset_row)

        # ruwe
        q_row = QHBoxLayout()
        self._chk_ruwe = QCheckBox("Max RUWE:")
        self._chk_ruwe.setChecked(True)
        self._chk_ruwe.toggled.connect(self._schedule_reload)
        q_row.addWidget(self._chk_ruwe)
        self._spin_ruwe = QDoubleSpinBox()
        self._spin_ruwe.setRange(1.0, 10.0)
        self._spin_ruwe.setDecimals(2)
        self._spin_ruwe.setSingleStep(0.1)
        self._spin_ruwe.setValue(1.4)
        self._spin_ruwe.valueChanged.connect(self._schedule_reload)
        q_row.addWidget(self._spin_ruwe)
        q_row.addStretch()
        cl.addLayout(q_row)

        # magnitude cut
        m_row = QHBoxLayout()
        self._chk_mag = QCheckBox("Max G mag:")
        self._chk_mag.setChecked(False)
        self._chk_mag.toggled.connect(self._schedule_reload)
        m_row.addWidget(self._chk_mag)
        self._spin_mag = QDoubleSpinBox()
        self._spin_mag.setRange(-2.0, 22.0)
        self._spin_mag.setDecimals(1)
        self._spin_mag.setSingleStep(0.5)
        self._spin_mag.setValue(12.0)
        self._spin_mag.valueChanged.connect(self._schedule_reload)
        m_row.addWidget(self._spin_mag)
        m_row.addStretch()
        cl.addLayout(m_row)

        # colour mode
        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("Colour by:"))
        self._combo_color = QComboBox()
        self._combo_color.addItems(["Absolute G (luminosity)",
                                    "Apparent G", "Distance",
                                    "XP Spectrum (true colour)"])
        self._combo_color.currentIndexChanged.connect(self._on_color_mode_changed)
        c_row.addWidget(self._combo_color, stretch=1)
        cl.addLayout(c_row)

        # point size
        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("Point size:"))
        self._slider_size = QSlider(Qt.Orientation.Horizontal)
        self._slider_size.setRange(10, 120)
        self._slider_size.setValue(40)
        self._slider_size.valueChanged.connect(self._recolor)
        s_row.addWidget(self._slider_size, stretch=1)
        cl.addLayout(s_row)

        # Depth cues
        v_row = QHBoxLayout()
        self._chk_droplines = QCheckBox("Drop lines")
        self._chk_droplines.setChecked(True)
        self._chk_droplines.setToolTip(
            "Vertical line from each star down to the galactic plane.\n"
            "Without it, height above the plane and distance from you\n"
            "are visually indistinguishable.")
        self._chk_droplines.toggled.connect(lambda _=False: self._recolor())
        v_row.addWidget(self._chk_droplines)

        self._chk_axes = QCheckBox("Galactic axes")
        self._chk_axes.setChecked(True)
        self._chk_axes.setToolTip(
            "Red   — toward the galactic centre (l=0°)\n"
            "Green — the direction the Sun is moving in its galactic orbit (l=90°)\n"
            "Blue  — galactic north\n"
            "Yellow— the Sun's own rotation axis (a tilt, not a motion)\n\n"
            "The grid is the galactic midplane, inclined ~60° to the ecliptic —\n"
            "it is not the plane the planets orbit in.")
        self._chk_axes.toggled.connect(lambda _=False: self._update_axes())
        v_row.addWidget(self._chk_axes)
        v_row.addStretch()
        cl.addLayout(v_row)

        btn_row = QHBoxLayout()
        btn_fit = QPushButton("Fit View")
        btn_fit.clicked.connect(self._fit_view)
        btn_row.addWidget(btn_fit)
        btn_reload = QPushButton("Reload")
        btn_reload.clicked.connect(self._reload)
        btn_row.addWidget(btn_reload)
        cl.addLayout(btn_row)

        side.addWidget(ctrl)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color:#668; font-size:11px;")
        side.addWidget(self._status)

        # nearest-star table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["star", "dist (pc)", "mag", "RUWE"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setStyleSheet(
            "QTableWidget{background:#0a0a18;color:#9ab;border:1px solid #2a2a3e;"
            "font-size:11px;gridline-color:#1a1a2e;}"
            "QHeaderView::section{background:#12121f;color:#778;border:none;padding:3px;}")
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_table_select)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)
        side.addWidget(self._table, stretch=1)

        # selected-star info
        self._info = QTextEdit()
        self._info.setReadOnly(True)
        self._info.setMinimumHeight(72)
        self._info.setMaximumHeight(96)
        self._info.setStyleSheet(
            "background:#0a0a18; color:#99aacc; border:1px solid #2a2a3e; "
            "font-size:11px; font-family:monospace;")
        side.addWidget(self._info)

        sb_row = QHBoxLayout()
        self._btn_simbad = QPushButton("Resolve in SIMBAD")
        self._btn_simbad.setEnabled(False)
        self._btn_simbad.clicked.connect(self._resolve_simbad)
        sb_row.addWidget(self._btn_simbad)
        self._btn_simbad_web = QPushButton("Open in browser")
        self._btn_simbad_web.setEnabled(False)
        self._btn_simbad_web.clicked.connect(self._open_simbad_web)
        sb_row.addWidget(self._btn_simbad_web)
        side.addLayout(sb_row)

        note = QLabel(
            "Drag to orbit · scroll to zoom · Ctrl+drag to pan · click a star to select.<br>"
            "<span style='color:#886;'>Gaia saturates below G≈3, so the naked-eye stars "
            "(Sirius, Alpha Cen, Vega, Procyon…) carry distances and colours derived from "
            "other catalogues — Hipparcos parallaxes and B–V photometry via SIMBAD — rather "
            "than from Gaia astrometry.</span>")
        note.setWordWrap(True)
        note.setStyleSheet("color:#667; font-size:10px;")
        note.setMaximumHeight(44)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._restore_splitter()
        body.splitterMoved.connect(lambda *_: self._save_splitter())        
        root.addWidget(note)

    _SPLIT_KEY = "gaia_neighborhood/splitter_state"

    def _save_splitter(self):
        try:
            QSettings("SetiAstro", "SASpro").setValue(
                self._SPLIT_KEY, self._splitter.saveState())
        except Exception:
            pass

    def _restore_splitter(self):
        try:
            st = QSettings("SetiAstro", "SASpro").value(self._SPLIT_KEY)
            if st:
                self._splitter.restoreState(st)
                return
        except Exception:
            pass
        # First run: give the 3D view ~70% of the width.
        self._splitter.setSizes([700, 300])

    # ── loading ───────────────────────────────────────────────────────────

    def _schedule_reload(self):
        self._reload_timer.start()

    def _empty_gaia(self) -> Dict[str, np.ndarray]:
        """Zero Gaia rows, correct dtypes — so _merge_bright has something to append to."""
        return {
            "source_id": np.empty(0, dtype=np.int64),
            "xyz":       np.empty((0, 3), dtype=np.float32),
            "dist_pc":   np.empty(0, dtype=np.float32),
            "gmag":      np.empty(0, dtype=np.float32),
            "ruwe":      np.empty(0, dtype=np.float32),
            "ra":        np.empty(0, dtype=np.float64),
            "dec":       np.empty(0, dtype=np.float64),
        }

    def _reload(self):
        # Never load before the GL view exists — showEvent triggers the first
        # load once _ensure_gl() has run.
        if not HAS_GL or not self._gl_ready:
            return
        lib = get_neighborhood_library()

        if not lib.is_installed():
            # No Gaia extract — but the hardcoded bright stars stand alone.
            # Render them rather than showing an empty box: at 10 pc that is
            # still Alpha Cen, Sirius, Procyon, Vega, Altair, Fomalhaut.
            self._gaia_missing = True
            self._on_loaded(self._empty_gaia())
            return

        self._gaia_missing = False

        if self._worker is not None and self._worker.isRunning():
            return

        self._status.setText("Loading…")
        QApplication.processEvents()

        max_ruwe = self._spin_ruwe.value() if self._chk_ruwe.isChecked() else None
        max_mag  = self._spin_mag.value()  if self._chk_mag.isChecked()  else None

        self._worker = _NeighborhoodLoadWorker(
            self._spin_radius.value(), max_ruwe, max_mag, self.MAX_POINTS, parent=None)
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_failed(self, msg: str):
        self._status.setText(f"Load failed: {msg}")
        self._worker = None

    def _on_loaded(self, data: dict):
        self._worker = None

        # The old rgb/found arrays are indexed by position in the *previous*
        # star list — meaningless now. The source_id cache survives, so nothing
        # already integrated has to be recomputed.
        if self._colour_worker is not None and self._colour_worker.isRunning():
            self._colour_worker.cancel()
            self._colour_worker.wait(1500)
        self._colour_worker = None
        self._spec_rgb = None
        self._spec_found = None

        # Fold in the bright stars Gaia can't see. Done here rather than in the
        # worker so the merge is cheap and always reflects the current radius.
        data = _merge_bright(data, self._spin_radius.value())

        self._data = data
        n = len(data["source_id"])
        n_bright = int(data["is_bright"].sum())

        truncated = (n >= self.MAX_POINTS)
        cuts = []
        if self._chk_ruwe.isChecked():
            cuts.append(f"RUWE<{self._spin_ruwe.value():.2f}")
        if self._chk_mag.isChecked():
            cuts.append(f"G≤{self._spin_mag.value():.1f}")
        cut_str = ("  ·  " + ", ".join(cuts)) if cuts else ""

        r = self._spin_radius.value()

        if n == 0:
            if self._gaia_missing and load_bright_stars() is None:
                self._status.setText(
                    "Nothing to show: the neighborhood library isn't installed "
                    "(Astrometry tab) and bright_stars_3d.py is missing "
                    f"({_BRIGHT_LOAD_ERROR}).")
            elif self._gaia_missing:
                self._status.setText(
                    f"No stars within {r:.1f} pc. Install “Stellar Neighborhood "
                    f"(within 300 pc)” on the Astrometry tab for the ~10M faint ones.")
            else:
                self._status.setText(f"No stars within {r:.1f} pc matching the cuts.")
            self._set_points(data)
            self._fill_table(data)
            self._fit_view()
            return

        extra = ""
        if self._gaia_missing:
            extra = ("  ·  Gaia neighborhood library not installed — showing only "
                     "naked-eye stars. Install it on the Astrometry tab.")
        elif n_bright:
            extra = f"  ·  incl. {n_bright} naked-eye stars (distances from Hipparcos)"
        if r > NEIGHBORHOOD_MAX_PC and not self._gaia_missing:
            extra += (f"  ·  ⚠ beyond {NEIGHBORHOOD_MAX_PC:.0f} pc only the "
                      f"naked-eye stars are plotted")
        self._status.setText(
            f"{n:,} stars within {r:.1f} pc{cut_str}{extra}"
            + ("  ·  ⚠ truncated at the render cap" if truncated else ""))

        self._set_points(data)
        self._fill_table(data)
        self._fit_view()
        if self._combo_color.currentIndex() == 3:
            self._start_colour_resolve()

    def _set_points(self, data: Optional[dict]):
        if not HAS_GL or not self._gl_ready:
            return
        if data is None or len(data["source_id"]) == 0:
            self._scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._marker.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._droplines.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._view.set_pick_points(None)
            self._table.setRowCount(0)
            self._clear_info()
            return
        self._view.set_pick_points(data["xyz"])
        self._recolor()

    def _on_color_mode_changed(self, index: int):
        if index == 3:                     # XP Spectrum (true colour)
            self._start_colour_resolve()
        else:
            self._recolor()

    def _start_colour_resolve(self):
        """Kick off spectrum -> colour resolution for the loaded stars."""
        if self._data is None or len(self._data["source_id"]) == 0:
            return
        if self._colour_worker is not None and self._colour_worker.isRunning():
            return

        sids = self._data["source_id"]
        n = len(sids)

        # Serve whatever we've already integrated, then only work on the rest.
        rgb   = np.zeros((n, 3), dtype=np.float32)
        found = np.zeros(n, dtype=bool)
        is_bright = self._data.get("is_bright")
        todo  = []
        for i in range(n):
            # Bright stars already carry a colour, derived at build time from
            # B-V through the same CIE integrator. Never ask Gaia for them —
            # it has no spectrum, that's the whole reason they're hardcoded.
            if is_bright is not None and is_bright[i]:
                rgb[i] = self._data["rgb"][i]
                found[i] = True
                continue
            c = self._colour_cache.get(int(sids[i]))
            if c is not None:
                rgb[i] = c
                found[i] = True
            else:
                todo.append(i)

        if not todo:
            # Everything already coloured (all bright, and/or all cached).
            # No XP tiers needed — don't nag about installing them.
            self._spec_rgb, self._spec_found = rgb, found
            self._recolor()
            return

        # Only now do we actually need spectra. If no XP tier is installed we
        # can still colour the bright stars; the Gaia ones stay grey.
        try:
            installed = bool(get_library().installed_bands())
        except Exception:
            installed = False
        if not installed:
            self._spec_rgb, self._spec_found = rgb, found
            n_bright = int(found.sum())
            self._status.setText(
                f"True colour: {n_bright} naked-eye stars coloured from B–V. "
                f"Gaia stars need XP spectra — install a tier on the Spectrum "
                f"Library tab (Ultra-Bright is only ~220 MB)."
            )
            self._recolor()
            return

        # Spend the budget on the brightest stars — colour reads best there,
        # and faint M dwarfs mostly have no XP spectrum anyway.
        todo = np.array(todo, dtype=np.int64)
        g = self._data["gmag"][todo]
        g = np.where(np.isfinite(g), g, 99.0)
        order = todo[np.argsort(g)][: self.MAX_SPECTRUM_STARS]

        self._spec_rgb, self._spec_found = rgb, found
        self._status.setText(
            f"Integrating XP spectra for {len(order):,} stars…"
            + (f"  (capped from {len(todo):,})" if len(todo) > len(order) else ""))

        self._colour_worker = _SpectrumColourWorker(sids, order, parent=None)
        self._colour_worker.finished_colours.connect(self._on_colours_ready)
        self._colour_worker.failed.connect(self._on_colours_failed)
        self._colour_worker.start()

    def _on_colours_failed(self, msg: str):
        self._colour_worker = None
        self._status.setText(f"Colour resolve failed: {msg}")

    def _on_colours_ready(self, rgb: np.ndarray, found: np.ndarray):
        self._colour_worker = None
        if self._data is None or len(rgb) != len(self._data["source_id"]):
            return   # data changed under us

        # Merge with anything already cached, then update the cache.
        if self._spec_rgb is not None and self._spec_found is not None:
            merged_found = self._spec_found | found
            merged_rgb = np.where(found[:, None], rgb, self._spec_rgb)
        else:
            merged_found, merged_rgb = found, rgb

        sids = self._data["source_id"]
        for i in np.nonzero(found)[0]:
            sid = int(sids[i])
            if sid > 0:                     # negative ids are hardcoded stars
                self._colour_cache[sid] = merged_rgb[i].copy()

        self._spec_rgb, self._spec_found = merged_rgb, merged_found

        # Report the XP-spectrum fraction over GAIA stars only. Bright stars
        # are coloured from B-V, not from a spectrum; counting them would
        # inflate the number and misrepresent Gaia's spectroscopic coverage.
        is_bright = self._data.get("is_bright")
        if is_bright is None:
            is_bright = np.zeros(len(sids), dtype=bool)
        gaia_mask = ~is_bright
        n_gaia = int(gaia_mask.sum())
        n_found = int((merged_found & gaia_mask).sum())
        n_bright = int(is_bright.sum())
        if n_gaia == 0:
            self._status.setText(
                f"True colour: {n_bright} naked-eye stars coloured from B–V "
                f"(no Gaia stars loaded).")
        else:
            bright_note = (f" + {n_bright} naked-eye stars coloured from B–V."
                           if n_bright else "")
            self._status.setText(
                f"True colour: {n_found:,}/{n_gaia:,} Gaia stars have XP spectra "
                f"({n_found/n_gaia:.0%}). Grey = no spectrum "
                f"(Gaia XP covers G≈2.2–15).{bright_note}"
            )
        self._recolor()

    MAX_DROPLINE_STARS = 20_000

    def _update_droplines(self, colors: Optional[np.ndarray] = None):
        """
        One segment per star, from (x,y,z) to (x,y,0). Faded toward the plane so
        it reads as a shadow-line rather than a stalk the star is standing on.

        Capped: 2 vertices per star, so 20k stars = 40k vertices. Beyond that
        the lines stop clarifying and start being a grey fog anyway.
        """
        if not self._gl_ready:
            return
        if (self._data is None or len(self._data["source_id"]) == 0
                or not self._chk_droplines.isChecked()):
            self._droplines.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        xyz = self._data["xyz"]
        n = len(xyz)
        if n > self.MAX_DROPLINE_STARS:
            # Show them for the nearest stars only — that's where the depth
            # ambiguity actually matters.
            keep = np.argsort(self._data["dist_pc"])[: self.MAX_DROPLINE_STARS]
            xyz = xyz[keep]
            colors = colors[keep] if colors is not None else None
            n = len(xyz)

        pts = np.empty((2 * n, 3), dtype=np.float32)
        pts[0::2] = xyz                       # at the star
        pts[1::2] = xyz
        pts[1::2, 2] = 0.0                    # dropped to the galactic plane

        cols = np.empty((2 * n, 4), dtype=np.float32)
        if colors is not None and len(colors) == n:
            cols[0::2, :3] = colors[:, :3]
            cols[1::2, :3] = colors[:, :3]
        else:
            cols[:, :3] = 0.55
        cols[0::2, 3] = 0.33                  # opaque-ish at the star
        cols[1::2, 3] = 0.0                   # invisible at the plane

        self._droplines.setData(pos=pts, color=cols, width=1.0, antialias=True)

    def _update_axes(self):
        """
        Three galactic reference directions plus the Sun's own tilt.

        The first three are large-scale galactic geometry. The fourth — the
        Sun's rotation axis — is a different kind of thing: it's the tilt of
        our own star, not a direction anything travels. Drawn short, from the
        Sun, and labelled so nobody mistakes it for a motion vector.
        """
        if not self._gl_ready:
            return
        if not self._chk_axes.isChecked():
            self._axes.setData(pos=np.zeros((0, 3), dtype=np.float32))
            for t in self._axis_labels:
                t.setData(text="")
            return

        r = float(self._spin_radius.value())
        L = r * 1.15                       # push just past the star field
        S = r * 0.30                       # solar pole: short, it's a local cue

        sp = SOLAR_NORTH_POLE_GAL * S

        pts = np.array([
            [0, 0, 0], [L, 0, 0],         # +X -> galactic centre
            [0, 0, 0], [0, L, 0],         # +Y -> where the Sun is headed
            [0, 0, 0], [0, 0, L * 0.6],   # +Z -> north galactic pole
            [0, 0, 0], sp,                # Sun's rotation axis
        ], dtype=np.float32)

        cols = np.array([
            [1.00, 0.45, 0.40, 0.15], [1.00, 0.45, 0.40, 0.85],
            [0.45, 0.95, 0.55, 0.15], [0.45, 0.95, 0.55, 0.75],
            [0.50, 0.70, 1.00, 0.15], [0.50, 0.70, 1.00, 0.75],
            [1.00, 0.92, 0.35, 0.25], [1.00, 0.92, 0.35, 0.95],   # solar yellow
        ], dtype=np.float32)

        self._axes.setData(pos=pts, color=cols, width=2.0, antialias=True)

        labels = [
            (np.array([L * 1.02, 0, 0]), "toward Galactic Centre", (255, 130, 115, 255)),
            (np.array([0, L * 1.02, 0]), "Sun's orbital direction", (120, 240, 145, 255)),
            (np.array([0, 0, L * 0.62]), "Galactic North", (130, 180, 255, 255)),
            (sp * 1.06,                  "Sun's N pole",  (255, 230, 110, 255)),
        ]
        for item, (pos, text, col) in zip(self._axis_labels, labels):
            item.setData(pos=pos, text=text, color=col)

    def _recolor(self):
        if (not HAS_GL or not self._gl_ready or self._data is None
                or len(self._data["source_id"]) == 0):
            return

        idx = self._combo_color.currentIndex()
        sizes = _star_sizes(self._data["gmag"], base=self._slider_size.value() / 10.0)

        if idx == 3:
            n = len(self._data["source_id"])
            colors = np.empty((n, 4), dtype=np.float32)
            if self._spec_rgb is None or self._spec_found is None:
                colors[:] = np.array([0.45, 0.45, 0.5, 0.5], dtype=np.float32)
            else:
                # Stars with a spectrum: their true colour, opaque.
                colors[:, :3] = self._spec_rgb
                colors[:, 3] = 0.95
                # Stars without: dim neutral grey. Not a guess dressed up as
                # data — you can see exactly which stars Gaia characterised.
                miss = ~self._spec_found
                colors[miss] = np.array([0.42, 0.42, 0.48, 0.35], dtype=np.float32)
        else:
            mode = {0: "absmag", 1: "appmag", 2: "dist"}[idx]
            colors = _star_colors(self._data["gmag"], self._data["dist_pc"], mode)

        self._scatter.setData(pos=self._data["xyz"], color=colors, size=sizes,
                              pxMode=True)
        self._update_droplines(colors)

    def _fit_view(self):
        if not HAS_GL or not self._gl_ready:
            return
        r = float(self._spin_radius.value())
        self._view.setCameraPosition(distance=max(r * 3.0, 2.0))
        try:
            self._grid.setSize(x=2 * r, y=2 * r, z=1)
            step = max(r / 5.0, 0.2)
            self._grid.setSpacing(x=step, y=step, z=step)
        except Exception:
            pass
        self._update_axes()

    # ── table + selection ─────────────────────────────────────────────────

    def _fill_table(self, data: dict):
        n = min(len(data["source_id"]), self.TABLE_ROWS)
        self._table.setRowCount(n)
        ib = data.get("is_bright")
        for i in range(n):
            g = data["gmag"][i]
            r = data["ruwe"][i]
            bright = ib is not None and ib[i]
            label = str(data["name"][i]) if bright else f"{i + 1}"
            vals = [
                label,
                f"{data['dist_pc'][i]:.3f}",
                "—" if not np.isfinite(g) else f"{g:.2f}",
                "—" if not np.isfinite(r) else f"{r:.2f}",
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if bright:
                    item.setForeground(QColor("#ffd966"))
                    item.setToolTip("Naked-eye star — too bright for Gaia (saturates below G≈3)")
                self._table.setItem(i, c, item)
        self._table.resizeColumnsToContents()

    def _on_table_select(self):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        self._select_index(rows[0].row(), scroll_table=False)

    def _on_point_picked(self, idx: int):
        self._select_index(idx, scroll_table=True)

    def _on_point_right_clicked(self, idx: int, global_pos):
        """Right-click a star in the 3D view: select it, then offer actions."""
        self._select_index(idx, scroll_table=True)
        self._show_star_menu(global_pos)

    def _spectrum_availability(self, sid: int) -> Tuple[bool, str]:
        """
        (available, reason). Cheap: get_spectrum/has_spectrum are primary-key
        lookups on source_id — no spatial search, no scan.
        """
        if self._selected_is_bright():
            return False, ("Too bright for Gaia — saturates below G≈3, which is "
                           "exactly why this star is hardcoded")
        try:
            lib = get_library()
        except Exception as e:
            return False, f"Spectral library unavailable ({e})"

        if not lib.installed_bands():
            return False, "No XP spectral tiers installed (Spectrum Library tab)"
        try:
            if lib.has_spectrum(sid):
                return True, ""
        except Exception as e:
            return False, f"Lookup failed ({e})"

        # Gaia XP only covers roughly G 2.2–15. Most stars in a 300 pc
        # neighbourhood are faint M dwarfs well past that limit, so "no
        # spectrum" is the common, expected case — say so rather than
        # implying something is broken.
        g = float("nan")
        if self._data is not None and self._selected_idx is not None:
            g = float(self._data["gmag"][self._selected_idx])
        if np.isfinite(g) and g > 15.0:
            return False, f"No XP spectrum — G={g:.2f} is past Gaia XP's ~15 mag limit"
        if np.isfinite(g) and g < 2.2:
            return False, f"No XP spectrum — G={g:.2f} saturates Gaia's detectors"
        return False, "Not present in the installed XP tiers"

    def _show_star_menu(self, global_pos):
        sid = self._selected_sid()
        if sid is None:
            return

        menu = QMenu(self)

        available, reason = self._spectrum_availability(sid)
        act_spec = menu.addAction("View XP Spectrum")
        act_spec.setEnabled(available)
        if not available:
            act_spec.setToolTip(reason)

        menu.addSeparator()
        act_simbad = menu.addAction("Resolve in SIMBAD")
        act_web    = menu.addAction("Open in SIMBAD (browser)")
        menu.addSeparator()
        act_copy = menu.addAction(
            "Copy name" if self._selected_is_bright() else "Copy source_id")

        # Menus don't show disabled-item tooltips by default.
        menu.setToolTipsVisible(True)

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is act_spec:
            self.spectrumRequested.emit(sid)
        elif chosen is act_simbad:
            self._resolve_simbad()
        elif chosen is act_web:
            self._open_simbad_web()
        elif chosen is act_copy:
            txt = (str(self._data["name"][self._selected_idx])
                   if self._selected_is_bright() else str(sid))
            QApplication.clipboard().setText(txt)
            self._status.setText(f"Copied “{txt}” to clipboard.")

    def _on_table_context_menu(self, pos):
        """Right-click a row in the nearest-star table -> same menu."""
        item = self._table.itemAt(pos)
        if item is None:
            return
        self._select_index(item.row(), scroll_table=False)
        self._show_star_menu(self._table.viewport().mapToGlobal(pos))

    def _select_index(self, idx: int, scroll_table: bool):
        if self._data is None or idx < 0 or idx >= len(self._data["source_id"]):
            return
        self._selected_idx = idx

        if HAS_GL and self._gl_ready:
            self._marker.setData(pos=self._data["xyz"][idx:idx + 1],
                                 color=np.array([[0.35, 1.0, 1.0, 1.0]], dtype=np.float32),
                                 size=np.array([18.0], dtype=np.float32), pxMode=True)

        if scroll_table:
            if idx < self._table.rowCount():
                self._table.blockSignals(True)
                self._table.selectRow(idx)
                self._table.scrollToItem(self._table.item(idx, 0),
                                         QTableWidget.ScrollHint.PositionAtCenter)
                self._table.blockSignals(False)
            else:
                # The table only lists the nearest TABLE_ROWS stars. A star
                # picked from the 3D view can legitimately lie beyond that, so
                # clear the stale highlight rather than leaving a wrong row lit.
                self._table.blockSignals(True)
                self._table.clearSelection()
                self._table.blockSignals(False)

        sid  = int(self._data["source_id"][idx])
        d    = float(self._data["dist_pc"][idx])
        g    = float(self._data["gmag"][idx])
        ruwe = float(self._data["ruwe"][idx])
        x, y, z = (float(v) for v in self._data["xyz"][idx])
        ra   = float(self._data["ra"][idx])
        dec  = float(self._data["dec"][idx])
        ly = d * 3.26156

        ib = self._data.get("is_bright")
        if ib is not None and ib[idx]:
            name = self._data["name"][idx]
            simbad_id = self._data["simbad_id"][idx] or "—"
            sp = self._data["sp_type"][idx] or "—"
            teff = float(self._data["teff"][idx])
            plx_ok = bool(self._data["plx_ok"][idx])

            warn = ("" if plx_ok else
                    "\n⚠ weak parallax (SNR<5) — distance is indicative only")
            self._info.setPlainText(
                f"{name}    [not in Gaia — saturates below G≈3]\n"
                f"SIMBAD  {simbad_id}    type {sp}\n"
                f"dist  {d:.4f} pc   ({ly:.3f} ly)\n"
                f"V {g:.2f}"
                + (f"   T_eff≈{teff:.0f} K (from B–V)" if teff else "")
                + f"\nRA {ra:.6f}°   Dec {dec:.6f}°\n"
                f"galactic XYZ  ({x:.3f}, {y:.3f}, {z:.3f}) pc"
                + warn
            )
        else:
            absg = (g - 5.0 * math.log10(max(d, 1e-6)) + 5.0) if np.isfinite(g) else float("nan")
            self._info.setPlainText(
                f"Gaia DR3 {sid}\n"
                f"dist  {d:.4f} pc   ({ly:.3f} ly)\n"
                f"G {'—' if not np.isfinite(g) else f'{g:.3f}'}"
                f"   M_G {'—' if not np.isfinite(absg) else f'{absg:.3f}'}"
                f"   RUWE {'—' if not np.isfinite(ruwe) else f'{ruwe:.3f}'}\n"
                f"RA {ra:.6f}°   Dec {dec:.6f}°\n"
                f"galactic XYZ  ({x:.3f}, {y:.3f}, {z:.3f}) pc"
            )
        self._btn_simbad.setEnabled(True)
        self._btn_simbad_web.setEnabled(True)

    def _clear_info(self):
        self._info.clear()
        self._selected_idx = None
        self._btn_simbad.setEnabled(False)
        self._btn_simbad_web.setEnabled(False)

    # ── SIMBAD ────────────────────────────────────────────────────────────

    def _selected_sid(self) -> Optional[int]:
        if self._data is None or self._selected_idx is None:
            return None
        return int(self._data["source_id"][self._selected_idx])

    def _selected_is_bright(self) -> bool:
        if self._data is None or self._selected_idx is None:
            return False
        ib = self._data.get("is_bright")
        return bool(ib is not None and ib[self._selected_idx])

    def _selected_identifier(self) -> Optional[str]:
        """SIMBAD identifier: the star's name if hardcoded, else its Gaia DR3 id."""
        if self._data is None or self._selected_idx is None:
            return None
        if self._selected_is_bright():
            sid = self._data["simbad_id"][self._selected_idx]
            nm  = self._data["name"][self._selected_idx]
            return str(sid) if sid else str(nm)
        return f"Gaia DR3 {self._selected_sid()}"

    def _resolve_simbad(self):
        ident = self._selected_identifier()
        if ident is None:
            return
        self._status.setText(f"Resolving {ident} via SIMBAD…")
        QApplication.processEvents()
        try:
            from astroquery.simbad import Simbad
            s = Simbad()
            try:
                s.add_votable_fields("otype")
            except Exception:
                pass
            res = s.query_object(ident)
            if res is None or len(res) == 0:
                self._status.setText(f"SIMBAD has no entry for {ident}.")
                return
            cols = {c.upper(): c for c in res.colnames}
            main = res[cols["MAIN_ID"]][0] if "MAIN_ID" in cols else "?"
            otype = res[cols["OTYPE"]][0] if "OTYPE" in cols else ""
            extra = f"\nSIMBAD: {main}" + (f"   ({otype})" if otype else "")
            self._info.setPlainText(self._info.toPlainText() + extra)
            self._status.setText(f"SIMBAD: {main}")
        except Exception as e:
            self._status.setText(f"SIMBAD lookup failed: {e}")

    def _open_simbad_web(self):
        ident = self._selected_identifier()
        if ident is None:
            return
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        from urllib.parse import quote_plus
        url = ("https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
               + quote_plus(ident))
        QDesktopServices.openUrl(QUrl(url))

    # ── lifecycle ─────────────────────────────────────────────────────────

    def library_changed(self):
        """
        Called when the astrometric library dir / installed files change.
        If the GL view hasn't been built yet (tab never visited), this is a
        no-op — showEvent will do the first load when the tab is opened.
        """
        refresh_neighborhood_library()
        if HAS_GL and self._gl_ready:
            # Force a reload: installing/removing the extract changes what
            # _reload() will produce, even at an unchanged radius.
            self._data = None
            self._reload()

    def shutdown(self):
        for w in (self._worker, self._colour_worker):
            if w is not None and w.isRunning():
                try:
                    w.cancel()
                except Exception:
                    pass
                w.wait(2000)
        self._worker = None
        self._colour_worker = None


# ══════════════════════════════════════════════════════════════════════════════
#  Main dialog
# ══════════════════════════════════════════════════════════════════════════════

class GaiaDatabaseDialog(QDialog):
    library_changed = _Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaia DR3 Library")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setMinimumSize(800, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._library       = get_library()
        self._astro_library = get_astro_library()

        self._workers:       Dict[str, _GroupDownloadWorker] = {}
        self._rows:          Dict[str, _GroupRowWidget]      = {}
        self._astro_workers: Dict[str, _GroupDownloadWorker] = {}
        self._astro_rows:    Dict[str, _GroupRowWidget]      = {}
        self._neighborhood_tab = None


        self._build_ui()
        self._refresh_groups("spectral")
        self._refresh_groups("astro")

        # Size AFTER the layout exists — resize() before _build_ui() is discarded
        # when the layout is installed and Qt re-derives from sizeHint().
        try:
            avail = self.screen().availableGeometry()
            self.resize(min(1180, int(avail.width()  * 0.88)),
                        min(840,  int(avail.height() * 0.84)))
        except Exception:
            self.resize(1000, 680)

        # NOTE: the GL view is deliberately NOT built here. Creating a
        # QOpenGLWidget inside a visible window makes Qt destroy and recreate
        # that window's native handle. We let that happen to *this dialog*, on
        # first visit to the Neighborhood Explorer tab (see its showEvent), and
        # not to the SASpro main window. Building it here instead — while the
        # dialog is still hidden — hides the flicker but pushes the rebuild up
        # to the main window, which is much worse. A blinking dialog beats a
        # blinking main window.

    # ── per-kind context helpers ───────────────────────────────────────────

    def _group_defs(self, kind: str) -> List[GroupDef]:
        return GROUP_DEFS if kind == "spectral" else ASTRO_GROUP_DEFS

    def _download_base(self, kind: str) -> str:
        return LIBRARY_DOWNLOAD_BASE if kind == "spectral" else ASTRO_LIBRARY_DOWNLOAD_BASE

    def _lib_dir(self, kind: str) -> Path:
        return get_library_dir() if kind == "spectral" else get_astro_library_dir()

    def _set_lib_dir(self, kind: str, path: Path):
        if kind == "spectral":
            set_library_dir(path)
        else:
            set_astro_library_dir(path)

    def _library_obj(self, kind: str):
        return self._library if kind == "spectral" else self._astro_library

    def _refresh_library_obj(self, kind: str):
        if kind == "spectral":
            refresh_library()
            self._library = get_library()
        else:
            refresh_astro_library()
            self._astro_library = get_astro_library()
            # The neighborhood extract lives in the same directory, so any
            # install/remove/relocate on the Astrometry tab can invalidate it.
            refresh_neighborhood_library()
            tab = getattr(self, "_neighborhood_tab", None)
            if tab is not None:
                try:
                    tab.library_changed()
                except Exception:
                    pass

    def _workers_dict(self, kind: str) -> Dict[str, _GroupDownloadWorker]:
        return self._workers if kind == "spectral" else self._astro_workers

    def _rows_dict(self, kind: str) -> Dict[str, _GroupRowWidget]:
        return self._rows if kind == "spectral" else self._astro_rows

    def _group_layout_for(self, kind: str):
        return self._group_layout if kind == "spectral" else self._astro_group_layout

    def _total_lbl_for(self, kind: str):
        return self._total_lbl if kind == "spectral" else self._astro_total_lbl

    def _dir_lbl_for(self, kind: str):
        return self._dir_lbl if kind == "spectral" else self._astro_dir_lbl

    def _build_ui(self):
        self.setStyleSheet("""
            QDialog { background: #0d0d1a; color: #ddd; }
            QTabWidget::pane { border: 1px solid #2a2a3e; background: #0d0d1a; }
            QTabBar::tab {
                background: #12121f; color: #888;
                padding: 6px 18px; border: 1px solid #2a2a3e;
                border-bottom: none; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: #1a1a2e; color: #ccd; }
            QLabel  { color: #ccc; }
            QLineEdit {
                background: #12121f; color: #ddd;
                border: 1px solid #334; border-radius: 3px; padding: 3px 6px;
            }
            QPushButton {
                background: #1a1a2e; color: #ccc;
                border: 1px solid #334; border-radius: 4px; padding: 5px 12px;
            }
            QPushButton:hover { background: #222240; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Header ────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet("background: #0a0a18; border-bottom: 1px solid #1a1a3e;")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(20, 14, 20, 14)
        title_lbl = QLabel("Gaia DR3 Library")
        title_lbl.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #8899ee; "
            "letter-spacing: 1px; border: none;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch()
        root.addWidget(hdr)

        # ── Tabs ──────────────────────────────────────────────────────────
        tabs = QTabWidget()
        self._tabs = tabs
        root.addWidget(tabs, stretch=1)

        tabs.addTab(self._build_group_tab("spectral"), "Spectrum Library")
        tabs.addTab(self._build_group_tab("astro"), "Astrometry")

        # Tab: Stellar Neighborhood Explorer (3D)
        self._neighborhood_tab = StellarNeighborhoodTab()
        self._neighborhood_tab.spectrumRequested.connect(self._show_spectrum_for_source)
        tabs.addTab(self._neighborhood_tab, "Neighborhood Explorer")

        # Tab: Spectrum Viewer
        viewer_tab    = QWidget()
        viewer_layout = QVBoxLayout(viewer_tab)
        viewer_layout.setContentsMargins(16, 16, 16, 16)
        viewer_layout.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Search by:"))
        self._search_mode = QComboBox()
        self._search_mode.addItems(["Star Name (SIMBAD)", "RA / Dec", "Gaia source_id"])
        self._search_mode.setFixedWidth(200)
        self._search_mode.currentIndexChanged.connect(self._on_search_mode_changed)
        mode_row.addWidget(self._search_mode)
        mode_row.addStretch()
        viewer_layout.addLayout(mode_row)

        self._name_row   = QWidget()
        name_layout = QHBoxLayout(self._name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        self._name_edit  = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Arneb, Muphrid, Tania Australis, HD 12345…")
        self._name_edit.returnPressed.connect(self._lookup_spectrum)
        name_layout.addWidget(self._name_edit, stretch=1)
        btn_name = QPushButton("Look Up")
        btn_name.clicked.connect(self._lookup_spectrum)
        name_layout.addWidget(btn_name)
        viewer_layout.addWidget(self._name_row)

        self._radec_row  = QWidget()
        radec_layout = QHBoxLayout(self._radec_row)
        radec_layout.setContentsMargins(0, 0, 0, 0)
        radec_layout.addWidget(QLabel("RA (deg):"))
        self._ra_edit = QLineEdit()
        self._ra_edit.setPlaceholderText("e.g. 279.234")
        self._ra_edit.setFixedWidth(110)
        self._ra_edit.returnPressed.connect(self._lookup_spectrum)
        radec_layout.addWidget(self._ra_edit)
        radec_layout.addWidget(QLabel("Dec (deg):"))
        self._dec_edit = QLineEdit()
        self._dec_edit.setPlaceholderText("e.g. 38.783")
        self._dec_edit.setFixedWidth(110)
        self._dec_edit.returnPressed.connect(self._lookup_spectrum)
        radec_layout.addWidget(self._dec_edit)
        radec_layout.addWidget(QLabel("Radius (arcsec):"))
        self._radius_edit = QLineEdit("5.0")
        self._radius_edit.setFixedWidth(60)
        radec_layout.addWidget(self._radius_edit)
        btn_radec = QPushButton("Look Up")
        btn_radec.clicked.connect(self._lookup_spectrum)
        radec_layout.addWidget(btn_radec)
        radec_layout.addStretch()
        self._radec_row.setVisible(False)
        viewer_layout.addWidget(self._radec_row)

        self._sid_row = QWidget()
        sid_layout = QHBoxLayout(self._sid_row)
        sid_layout.setContentsMargins(0, 0, 0, 0)
        self._sid_edit = QLineEdit()
        self._sid_edit.setPlaceholderText("Enter Gaia DR3 source_id…")
        self._sid_edit.returnPressed.connect(self._lookup_spectrum)
        sid_layout.addWidget(self._sid_edit, stretch=1)
        btn_sid = QPushButton("Look Up")
        btn_sid.clicked.connect(self._lookup_spectrum)
        sid_layout.addWidget(btn_sid)
        self._sid_row.setVisible(False)
        viewer_layout.addWidget(self._sid_row)

        self._viewer_status = QLabel(
            "Search for a star by name, coordinates, or Gaia source_id.")
        self._viewer_status.setStyleSheet("color: #668; font-size: 11px;")
        viewer_layout.addWidget(self._viewer_status)

        self._spectrum_viewer = SpectrumViewerWidget()
        viewer_layout.addWidget(self._spectrum_viewer, stretch=1)

        self._source_info = QTextEdit()
        self._source_info.setReadOnly(True)
        self._source_info.setMaximumHeight(72)
        self._source_info.setStyleSheet(
            "background: #0a0a18; color: #99aacc; border: 1px solid #2a2a3e; "
            "font-size: 11px; font-family: monospace;")
        viewer_layout.addWidget(self._source_info)
        self._spectrum_tab_index = tabs.addTab(viewer_tab, "Spectrum Viewer")

        # Tab: About
        about_tab    = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_layout.setContentsMargins(20, 20, 20, 20)
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setStyleSheet(
            "background: #0a0a18; color: #aab; border: none; font-size: 12px;")
        about_text.setHtml("""
        <h3 style="color:#8899ee;">Gaia DR3 XP Spectral Library</h3>
        <p>Pre-calibrated Gaia BP/RP spectra (336–1020 nm, 343 points, 2 nm step)
        from Gaia Data Release 3, stored in compact compressed SQLite databases.</p>
        <p>Gaia DR3 XP spectra cover <b>G ≈ 2.2 to G ≲ 15.0</b>. Stars brighter than G≈2.2 
        (Vega, Sirius, Arcturus, Rigel, Betelgeuse, etc.) saturate Gaia's detectors and 
        have no XP spectra. The brightest sources in this library are stars like Arneb 
        (Alpha Leporis, G=2.55) and Muphrid (Eta Bootis, G=2.56). The Very Faint group 
        represents the practical depth limit of the DR3 dataset at G≈15.</p>
        <p>Spectra were extracted from the <b>xp_sampled_mean_spectrum</b> table
        of the Gaia DR3 bulk CDN. No GaiaXPy required — calibrated by DPAC.</p>
        <h4 style="color:#8899ee;">Lookup priority in SASpro</h4>
        <ol>
        <li>Local bulk library (these files) — instant, no network</li>
        <li>Live download cache (gaia_xp_cache.sqlite)</li>
        <li>Live Gaia archive — fallback when neither has the source</li>
        </ol>
        <h4 style="color:#8899ee;">Astrometric library</h4>
        <p>The Astrometry tab installs full-depth Gaia DR3 <b>gaia_source</b> astrometry
        (RA, Dec, proper motion, parallax) split by magnitude across 41 files —
        1,585,128,120 sources, 183.9 GB at full depth. It is a far denser catalog than
        the XP spectral library, which only covers sources bright enough for usable
        BP/RP spectra. Start with the Bright tier; add depth only if your fields need it.</p>
        <h4 style="color:#8899ee;">Stellar Neighborhood Explorer</h4>
        <p>A 3D view of every DR3 source with a usable parallax within 300 parsecs,
        Sun at the origin, galactic plane in XY. Distances come from
        <i>d</i> = 1000 / parallax(mas).</p>
        <p style="color:#a98;">Caveat: DR3 astrometry here does not carry
        <code>parallax_error</code>, so distance quality is guarded by a RUWE cut plus a
        conservative 300 pc limit. RUWE flags <i>bad astrometric fits</i> (unresolved
        binaries, blends) — it does <b>not</b> flag a clean fit whose parallax is simply
        too small to measure. The radius limit is what guards against that. Gaia DR4
        will add <code>parallax_error</code> and let this reach farther honestly.
        Also note Gaia saturates below G≈3, so the very nearest bright stars
        (Sirius, Alpha Centauri, Procyon) are absent from the catalog entirely.</p>
        <h4 style="color:#8899ee;">Citation</h4>
        <p>Gaia Collaboration et al. (2022), Gaia DR3.
        <i>A&amp;A</i> 649, A1.<br>
        <a href="https://gea.esac.esa.int/archive/" style="color:#6688cc;">
        https://gea.esac.esa.int/archive/</a></p>
        <p style="color:#556; font-size:10px;">
        Written by Franklin Marek / www.setiastro.com</p>
        """)
        about_layout.addWidget(about_text)
        tabs.addTab(about_tab, "About")

        # Footer
        footer = QWidget()
        footer.setStyleSheet("background: #0a0a18; border-top: 1px solid #1a1a3e;")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 8, 16, 8)
        footer_layout.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        footer_layout.addWidget(btn_close)
        root.addWidget(footer)

    def _build_group_tab(self, kind: str) -> QWidget:
        """Builds one group-list tab (Spectrum Library or Astrometry) and stashes
        its per-kind widget references (dir label, total label, group layout)."""
        tab    = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        if kind == "spectral":
            info_text = (
                "Install Gaia DR3 XP spectral library files for offline color calibration. "
                "SASpro checks these before hitting the live Gaia archive. "
                "Interrupted downloads resume automatically — just click Install again."
            )
        else:
            info_text = (
                "Install Gaia DR3 astrometric library files (RA/Dec/proper motion/parallax) "
                "for plate solving and star identification, split by magnitude — start with "
                "the brightest tier and add fainter ones only as needed, same approach as "
                "the Spectrum Library tab. The Stellar Neighborhood extract additionally "
                "powers the 3D Neighborhood Explorer tab."
            )

        info_lbl = QLabel(info_text)
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #778; font-size: 11px; margin-bottom: 4px;")
        layout.addWidget(info_lbl)

        if kind == "astro":
            banner = QLabel(
                "ℹ️ <b>Upload pending.</b> All 41 files are built and verified "
                "(1,585,128,120 sources, 183.9 GB) but aren't on the CDN yet, so "
                "<b>Install</b> will fail until they are. Use <b>Browse…</b> to point at "
                "your local copies for testing."
            )
            banner.setWordWrap(True)
            banner.setStyleSheet(
                "background: #0d1a22; color: #6aa9c4; border: 1px solid #1e3a47; "
                "border-radius: 4px; padding: 6px 10px; font-size: 11px;")
            layout.addWidget(banner)

        # Directory row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Library folder:"))
        dir_lbl = QLabel(str(self._lib_dir(kind)))
        dir_lbl.setStyleSheet("color: #6688aa; font-size: 11px; font-family: monospace;")
        dir_lbl.setWordWrap(True)
        dir_row.addWidget(dir_lbl, stretch=1)

        btn_change_loc = QPushButton("Change Location…")
        btn_change_loc.setFixedWidth(130)
        btn_change_loc.setToolTip(
            "Move the library to a different drive or folder.\n"
            "Useful for keeping large files off your system drive.")
        btn_change_loc.clicked.connect(
            self._change_location_spectral if kind == "spectral"
            else self._change_location_astro)
        dir_row.addWidget(btn_change_loc)

        btn_open_dir = QPushButton("Open Folder")
        btn_open_dir.setFixedWidth(100)
        btn_open_dir.clicked.connect(
            self._open_library_dir_spectral if kind == "spectral"
            else self._open_library_dir_astro)
        dir_row.addWidget(btn_open_dir)
        layout.addLayout(dir_row)

        # Totals row. No Count button on either tab: every group carries an
        # exact build-time n_items, so COUNT(*) over multi-GB tables would be
        # disk churn to recompute numbers we already have.
        total_row = QHBoxLayout()
        total_lbl = QLabel("")
        total_lbl.setStyleSheet("font-size: 11px; color: #556688;")
        total_row.addWidget(total_lbl, stretch=1)
        layout.addLayout(total_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        group_layout = QVBoxLayout(container)
        group_layout.setSpacing(8)
        group_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, stretch=1)

        if kind == "spectral":
            self._dir_lbl      = dir_lbl
            self._total_lbl    = total_lbl
            self._group_layout = group_layout
        else:
            self._astro_dir_lbl      = dir_lbl
            self._astro_total_lbl    = total_lbl
            self._astro_group_layout = group_layout

        return tab

    # ── Change location ───────────────────────────────────────────────────

    def _change_location_spectral(self):
        """Button slot — no positional args, so Qt's clicked(bool) can't leak in."""
        self._change_location("spectral")

    def _change_location_astro(self):
        self._change_location("astro")

    def _open_library_dir_spectral(self):
        self._open_library_dir("spectral")

    def _open_library_dir_astro(self):
        self._open_library_dir("astro")

    def _change_location(self, kind: str = "spectral"):
        # Belt-and-suspenders: if a bare clicked.connect ever hands us
        # Qt's checked=bool, fall back to spectral rather than crashing.
        if not isinstance(kind, str):
            kind = "spectral"
        if self._workers_dict(kind):
            QMessageBox.warning(self, "Downloads Active",
                                "Please wait for active downloads to finish "
                                "before changing the library location.")
            return

        current_dir = self._lib_dir(kind)

        installed_files = [
            f for g in self._group_defs(kind)
            for f in g.filenames
            if (current_dir / f).exists()
        ]

        dlg = _ChangeLocationDialog(current_dir, installed_files, kind=kind, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # ── Reload library from new location and refresh UI ──────────────────
        self._refresh_library_obj(kind)
        self._dir_lbl_for(kind).setText(str(self._lib_dir(kind)))
        self._refresh_groups(kind)
        self.library_changed.emit()

    # ── Group management ──────────────────────────────────────────────────

    def _refresh_groups(self, kind: str = "spectral"):
        rows = self._rows_dict(kind)
        group_layout = self._group_layout_for(kind)
        for row in rows.values():
            group_layout.removeWidget(row)
            row.deleteLater()
        rows.clear()

        lib_dir = self._lib_dir(kind)
        for g in self._group_defs(kind):
            installed = [f for f in g.filenames if (lib_dir / f).exists()]
            missing   = [f for f in g.filenames if not (lib_dir / f).exists()]
            total_mb  = sum(
                (lib_dir / f).stat().st_size / (1024 * 1024)
                for f in installed
            )
            status = GroupStatus(
                group=g,
                installed=installed,
                missing=missing,
                total_mb=total_mb,
                total_spectra=0,
            )
            row = _GroupRowWidget(status)
            row.install_requested.connect(lambda key, k=kind: self._on_install(key, k))
            row.remove_requested.connect(lambda key, k=kind: self._on_remove(key, k))
            row.browse_requested.connect(lambda key, k=kind: self._on_browse(key, k))
            group_layout.insertWidget(group_layout.count() - 1, row)
            rows[status.group.key] = row

        self._update_total_label(kind)

    def _update_total_label(self, kind: str):
        """
        Sum the exact build-time counts of fully-installed groups. No COUNT(*)
        anywhere: every group knows its own row count from when it was built.
        """
        defs     = self._group_defs(kind)
        lib_dir  = self._lib_dir(kind)
        n_groups = self._n_installed_groups(kind)
        unit     = "spectra" if kind == "spectral" else "stars"

        installed_total = sum(
            (g.n_items or 0) for g in defs
            if g.n_items is not None
            and all((lib_dir / f).exists() for f in g.filenames)
        )
        catalog_total = sum(g.n_items or 0 for g in defs)

        if installed_total:
            txt = (f"{n_groups}/{len(defs)} groups installed  ·  "
                   f"{installed_total:,} {unit} installed "
                   f"of {catalog_total:,} in the catalog")
        else:
            txt = (f"{n_groups}/{len(defs)} groups installed  ·  "
                   f"{catalog_total:,} {unit} available")
        self._total_lbl_for(kind).setText(txt)

    def _n_installed_groups(self, kind: str = "spectral") -> int:
        lib_dir = self._lib_dir(kind)
        return sum(1 for g in self._group_defs(kind)
                   if all((lib_dir / f).exists() for f in g.filenames))

    def _on_install(self, key: str, kind: str = "spectral"):
        group_def = next(g for g in self._group_defs(kind) if g.key == key)
        lib_dir   = self._lib_dir(kind)
        missing   = [f for f in group_def.filenames if not (lib_dir / f).exists()]

        if not missing:
            return

        rows = self._rows_dict(kind)
        row = rows.get(key)
        if row is None:
            return

        worker = _GroupDownloadWorker(
            group_def, missing, lib_dir,
            download_base=self._download_base(kind), parent=self)
        self._workers_dict(kind)[key] = worker

        row.set_downloading(True, cancel_cb=worker.cancel)

        worker.file_progress.connect(
            lambda d, t, m, r=row: r.update_file_progress(d, t, m) if not sip.isdeleted(r) else None)
        worker.group_progress.connect(
            lambda d, t, r=row: r.update_group_progress(d, t) if not sip.isdeleted(r) else None)
        worker.file_done.connect(
            lambda fname, ok, k=kind: self._on_file_done(fname, ok, k))
        worker.group_finished.connect(
            lambda ok, msg, key=key, k=kind: self._on_group_finished(key, ok, msg, k))
        worker.cancelled.connect(
            lambda key=key, k=kind: self._on_cancelled(key, k))
        worker.start()

    def _on_file_done(self, fname: str, ok: bool, kind: str = "spectral"):
        if ok:
            self._refresh_library_obj(kind)

    def _on_group_finished(self, key: str, ok: bool, msg: str, kind: str = "spectral"):
        rows = self._rows_dict(kind)
        row = rows.get(key)
        if row:
            row.set_downloading(False)

        worker = self._workers_dict(kind).pop(key, None)
        if worker:
            worker.wait()

        self._refresh_library_obj(kind)
        self._refresh_groups(kind)
        self.library_changed.emit()

        if ok:
            QMessageBox.information(self, "Install Complete", msg)
        else:
            QMessageBox.critical(self, "Download Failed", msg)

    def _on_cancelled(self, key: str, kind: str = "spectral"):
        rows = self._rows_dict(kind)
        row = rows.get(key)
        if row:
            row.set_downloading(False)
        self._workers_dict(kind).pop(key, None)
        self._refresh_library_obj(kind)
        self._refresh_groups(kind)

    def _on_remove(self, key: str, kind: str = "spectral"):
        group_def = next(g for g in self._group_defs(kind) if g.key == key)
        lib_dir   = self._lib_dir(kind)
        installed = [f for f in group_def.filenames if (lib_dir / f).exists()]
        total_gb  = sum(
            (lib_dir / f).stat().st_size / (1024 * 1024 * 1024)
            for f in installed
        )

        fallback_note = (
            "SASpro will fall back to live Gaia archive downloads for "
            "sources in this magnitude range."
            if kind == "spectral" else
            "SASpro will no longer have offline astrometry for this "
            "declination band."
        )

        reply = QMessageBox.question(
            self, "Remove Group",
            f"Delete all installed files for '{group_def.label}'?\n\n"
            f"{len(installed)} file(s) · {total_gb:.1f} GB will be freed.\n\n"
            f"{fallback_note}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        lib = self._library_obj(kind)
        for fname in installed:
            lib.close_file(fname)
            path = lib_dir / fname
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete {fname}:\n{e}")
                break

        self._refresh_library_obj(kind)
        self._refresh_groups(kind)
        self.library_changed.emit()

    def _on_browse(self, key: str, kind: str = "spectral"):
        group_def = next(g for g in self._group_defs(kind) if g.key == key)
        lib_dir   = self._lib_dir(kind)
        missing   = [f for f in group_def.filenames if not (lib_dir / f).exists()]

        if not missing:
            QMessageBox.information(self, "Already Installed",
                                    f"{group_def.label} is already fully installed.")
            return

        start_dir = getattr(self, "_last_browse_dir", str(lib_dir))
        path, _ = QFileDialog.getOpenFileName(
            self, f"Locate a file for '{group_def.label}'", start_dir,
            "SQLite files (*.sqlite);;All files (*)")
        if not path:
            return

        selected_path = Path(path)
        self._last_browse_dir = str(selected_path.parent)

        found_others = []
        for fname in missing:
            candidate = selected_path.parent / fname
            if candidate.exists() and candidate != selected_path:
                found_others.append((fname, candidate))

        to_install = [(selected_path.name, selected_path)]
        if found_others:
            names_list = "\n".join(f"  {f}" for f, _ in found_others)
            reply = QMessageBox.question(
                self, "Additional Files Found",
                f"Found {len(found_others)} other file(s) for '{group_def.label}' "
                f"in the same folder:\n\n{names_list}\n\n"
                f"Install all of them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                to_install.extend(found_others)

        copied = 0
        for fname, src in to_install:
            dest = lib_dir / fname
            if dest.exists():
                continue
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not copy {fname}:\n{e}")

        if copied > 0:
            self._refresh_library_obj(kind)
            self._refresh_groups(kind)
            self.library_changed.emit()

    def _open_library_dir(self, kind: str = "spectral"):
        import subprocess, sys
        d = str(self._lib_dir(kind))
        if sys.platform == "win32":
            os.startfile(d)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])

    # ── Cross-tab: Neighborhood Explorer -> Spectrum Viewer ───────────────

    def _show_spectrum_for_source(self, source_id: int):
        """
        Right-click a star in the 3D explorer -> "View XP Spectrum". Switches to
        the Spectrum Viewer tab and runs the existing source_id lookup path, so
        there's exactly one implementation of "load and draw a spectrum".

        This is a primary-key hit on source_id across the open XP connections —
        no coordinate search, no epoch matching. It's about as fast as a lookup
        gets, which is the whole point of routing through the Gaia ID.
        """
        try:
            self._tabs.setCurrentIndex(self._spectrum_tab_index)
            self._search_mode.setCurrentIndex(2)          # "Gaia source_id"
            self._sid_edit.setText(str(int(source_id)))
            self._lookup_spectrum()
        except Exception as e:
            QMessageBox.warning(self, "Spectrum",
                                f"Could not open spectrum for {source_id}:\n{e}")

    # ── Spectrum viewer  (unchanged — spectral only) ───────────────────────

    def _on_search_mode_changed(self, index: int):
        self._name_row.setVisible(index == 0)
        self._radec_row.setVisible(index == 1)
        self._sid_row.setVisible(index == 2)

    def _lookup_spectrum(self):
        mode = self._search_mode.currentIndex()
        ra = dec = None
        sid   = None
        label = ""

        if mode == 0:
            name = self._name_edit.text().strip()
            if not name:
                return
            self._viewer_status.setText(f"Resolving '{name}' via SIMBAD…")
            QApplication.processEvents()
            try:
                from astroquery.simbad import Simbad
                from astropy.coordinates import SkyCoord

                try:
                    s = Simbad()
                    s.add_votable_fields("ids", "ra", "dec")
                    result = s.query_object(name)
                except Exception:
                    result = None

                if result is not None and len(result) > 0:
                    ids_col = None
                    for col in result.colnames:
                        if col.upper() == "IDS":
                            ids_col = col
                            break

                    if ids_col is not None:
                        ids_str = str(result[ids_col][0])
                        for part in ids_str.split("|"):
                            part = part.strip()
                            if part.startswith("Gaia DR3 "):
                                try:
                                    sid = int(part.replace("Gaia DR3 ", "").strip())
                                    label = name
                                    self._viewer_status.setText(
                                        f"Resolved '{name}' → Gaia DR3 {sid} — searching library…")
                                    QApplication.processEvents()

                                    spec = self._library.get_spectrum(sid)
                                    if spec is not None:
                                        info = self._library.get_source_info(sid)
                                        if info:
                                            label = (f"{name}  →  source {sid}  "
                                                    f"(G={info['gmag']:.2f})")
                                            self._source_info.setPlainText(
                                                f"source_id: {sid}\n"
                                                f"RA: {info['ra']:.6f}°    Dec: {info['dec']:.6f}°    "
                                                f"G mag: {info['gmag']:.3f}")
                                        else:
                                            self._source_info.setPlainText(f"source_id: {sid}")
                                        self._spectrum_viewer.show_spectrum(spec, title=label)
                                        self._viewer_status.setText("Spectrum loaded — local library")
                                        return
                                    else:
                                        self._viewer_status.setText(
                                            f"Gaia DR3 {sid} found in SIMBAD but not in installed "
                                            f"library groups — trying positional fallback…")
                                        QApplication.processEvents()
                                except ValueError:
                                    pass
                                break

                    try:
                        ra_val  = float(result["RA"][0])
                        dec_val = float(result["DEC"][0])
                    except Exception:
                        coord = SkyCoord.from_name(name)
                        ra_val, dec_val = coord.ra.deg, coord.dec.deg

                    ra, dec = ra_val, dec_val
                    label = name
                    self._viewer_status.setText(
                        f"Resolved {name} → RA={ra:.5f}°  Dec={dec:.5f}°  — searching…")
                    QApplication.processEvents()
                else:
                    coord = SkyCoord.from_name(name)
                    ra, dec = coord.ra.deg, coord.dec.deg
                    label = name
                    self._viewer_status.setText(
                        f"Resolved {name} → RA={ra:.5f}°  Dec={dec:.5f}°  — searching…")
                    QApplication.processEvents()

            except Exception as e:
                self._viewer_status.setText(
                    f"Could not resolve '{name}' via SIMBAD: {e}")
                return

        elif mode == 1:
            try:
                ra  = float(self._ra_edit.text().strip())
                dec = float(self._dec_edit.text().strip())
                label = f"RA={ra:.5f}° Dec={dec:.5f}°"
            except ValueError:
                self._viewer_status.setText("Invalid RA or Dec — enter decimal degrees.")
                return

        elif mode == 2:
            try:
                sid = int(self._sid_edit.text().strip())
            except ValueError:
                self._viewer_status.setText("Invalid source_id — must be an integer.")
                return

        spec = None

        if sid is not None:
            spec = self._library.get_spectrum(sid)
            info = self._library.get_source_info(sid)
            if info:
                ra, dec = info["ra"], info["dec"]
                label = f"Gaia source {sid}  (G={info['gmag']:.2f})"
            else:
                label = f"Gaia source {sid}"

        elif ra is not None and dec is not None:
            try:
                radius = float(self._radius_edit.text().strip()) if mode == 1 else 5.0
            except ValueError:
                radius = 5.0

            search_radii = [radius, 30.0, 120.0, 600.0] if mode == 0 else [radius]

            result = None
            for r in search_radii:
                result = self._library.find_nearest(ra, dec, radius_arcsec=r)
                if result is not None:
                    if r > radius and mode == 0:
                        self._viewer_status.setText(
                            f"Found at {r:.0f}\" (SIMBAD/Gaia epoch offset) — loading…")
                        QApplication.processEvents()
                    break

            if result is None:
                self._viewer_status.setText(
                    f"No spectrum found within {search_radii[-1]:.0f}\" of {label}. "
                    f"Try RA/Dec mode or source_id directly. "
                    f"Note: very bright stars (G<2.2) are not in Gaia XP.")
                self._spectrum_viewer.clear()
                self._source_info.clear()
                return

            sid, sep = result
            spec = self._library.get_spectrum(sid)
            info = self._library.get_source_info(sid)
            if info:
                label = (f"{label}  →  source {sid}  "
                         f"(G={info['gmag']:.2f}, sep={sep:.2f}\")")
            else:
                label = f"{label}  →  source {sid}  (sep={sep:.2f}\")"

        if spec is None and sid is not None:
            try:
                from setiastro.saspro.gaia_downloader import GaiaSpectraDB
                base     = QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppDataLocation)
                db_path  = os.path.join(base, "gaia", "gaia_xp_cache.sqlite")
                if os.path.exists(db_path):
                    db = GaiaSpectraDB(db_path)
                    cached = db.get_spectrum(sid)
                    db.close()
                    if cached is not None:
                        spec = CalibratedSpectrum(
                            source_id=sid,
                            wavelengths=cached.wavelengths,
                            flux=cached.flux,
                            flux_error=cached.flux_error,
                        )
            except Exception:
                pass

        if spec is None:
            self._viewer_status.setText(
                f"No spectrum found for {label}. "
                f"The relevant magnitude group may not be installed.")
            self._spectrum_viewer.clear()
            self._source_info.clear()
            return

        if sid is not None:
            info = self._library.get_source_info(sid)
            if info:
                self._source_info.setPlainText(
                    f"source_id: {sid}\n"
                    f"RA: {info['ra']:.6f}°    Dec: {info['dec']:.6f}°    "
                    f"G mag: {info['gmag']:.3f}")
            else:
                self._source_info.setPlainText(f"source_id: {sid}")
        else:
            self._source_info.clear()

        self._spectrum_viewer.show_spectrum(spec, title=label)
        source_str = "local library" if self._library.has_spectrum(sid) else "live cache"
        self._viewer_status.setText(f"Spectrum loaded — {source_str}")

    def closeEvent(self, event):
        tab = getattr(self, "_neighborhood_tab", None)
        if tab is not None:
            try:
                tab.shutdown()
            except Exception:
                pass

        for workers_dict in (self._workers, self._astro_workers):
            for worker in list(workers_dict.values()):
                worker.cancel()
                worker.wait()
            workers_dict.clear()
        super().closeEvent(event)


# ── Public entry point ────────────────────────────────────────────────────────

def open_gaia_database_dialog(parent=None) -> GaiaDatabaseDialog:
    dlg = GaiaDatabaseDialog(parent=parent)
    dlg.show()
    return dlg