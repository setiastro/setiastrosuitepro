#src/setiastro/saspro/gaia_database.py
#!/usr/bin/env python3
# ======================================================
#   _____      __  _ ___         __           
#  / ___/___  / /_(_)   |  _____/ /__________ 
#  \__ \/ _ \/ __/ / /| | / ___/ __/ ___/ __ \
# ___/ /  __/ /_/ / ___ |(__  ) /_/ /  / /_/ /
#/____/\___/\__/_/_/  |_/____/\__/_/   \____/ 
#  SASpro Gaia XP Spectral Library                                             
#
#  gaia_database.py  —  SASpro Gaia XP Spectral Library + Astrometric Library
#  Manages bulk-downloaded Gaia DR3 SQLite files, both flavors:
#    - XP spectra (mag-banded, for color calibration)
#    - Astrometry (declination-banded, for plate solving / star ID)
#  Each provides a unified lookup layer:
#    1. Local bulk library (distributed SQLite files)
#    2. Live cache DB   (gaia_xp_cache.sqlite, spectral only)
#    3. Live archive    (fallback, handled by GaiaDownloader, spectral only)
#
#  Written by Franklin Marek
#  www.setiastro.com
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
    Qt, QThread, QStandardPaths, QSettings,
    pyqtSignal as _Signal,
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QWidget, QMessageBox, QFileDialog,
    QScrollArea, QFrame, QTabWidget, QTextEdit, QLineEdit, QComboBox,
    QSizePolicy, QRadioButton, QButtonGroup,
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


# ── Backblaze public URLs ──────────────────────────────────────────────────────
LIBRARY_DOWNLOAD_BASE       = "https://f005.backblazeb2.com/file/setiastro-gaia/"
# NOTE: confirm the exact host/bucket-name Backblaze assigns once the
# "setiastro-astrometry" bucket is created — this is a placeholder mirroring
# the existing spectral bucket's URL pattern.
ASTRO_LIBRARY_DOWNLOAD_BASE = "https://f005.backblazeb2.com/file/setiastro-astrometry/"

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
    # True while this group's files aren't hosted anywhere yet — shows a
    # "Coming Soon" state instead of live Install/Browse/Remove buttons.
    coming_soon: bool = False


GROUP_DEFS: List[GroupDef] = [
    GroupDef(
        key="ultra_bright",
        label="Ultra-Bright  (G < 8)",
        mag_lo=None, mag_hi=8.0,
        filenames=["gaia_xp_lt8.sqlite"],
        est_size="~220 MB",
        est_stars="~55k stars",
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
        est_stars="~385k stars",
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
        est_stars="~2.4M stars",
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
        est_stars="~12.9M stars",
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
        est_stars="~18.7M stars",
        description="Maximum depth. For the deepest narrowband fields. "
                    "Gaia DR3 XP spectra top out at G≈15.",
        warning="Very large download — ~73 GB total. Files range 5–10 GB each. "
                "Only needed for extreme deep-field work.",
    ),
]

_FILENAME_TO_GROUP: Dict[str, str] = {
    fn: g.key for g in GROUP_DEFS for fn in g.filenames
}


# ── Astrometric magnitude tiers (real split plan — pending upload) ───────────
# Matches gaia_astro_split_bands.py's SPLIT_PLAN exactly: 41 physical files,
# grouped here into 5 user-facing tiers at clean magnitude boundaries so the
# UI reuses the same "one tier = many files, one Install downloads them all"
# pattern already used by the Spectrum Library tab's faint/very_faint groups.
# Sizes below are computed from real dry-run row counts (~130.9 bytes/row),
# not guesses — still marked coming_soon until the files are actually
# uploaded to the setiastro-astrometry bucket.

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
        key="astro_bright",
        label="Bright  (G < 17)",
        mag_lo=None, mag_hi=17.0,
        filenames=(
            ["gaia_astro_lt15.sqlite", "gaia_astro_150_160.sqlite"]
            + _astro_filenames(16.0, 17.0, 0.5)
        ),
        est_size="~17.0 GB  (4 files)",
        est_stars="~139.5M stars",
        description="Bright astrometric anchors — enough for most wide-field plate solves.",
        recommended=True,
        unit_label="stars",
        coming_soon=True,
    ),
    GroupDef(
        key="astro_faint",
        label="Faint  (G 17–18)",
        mag_lo=17.0, mag_hi=18.0,
        filenames=_astro_filenames(17.0, 18.0, 0.2),
        est_size="~15.8 GB  (5 files)",
        est_stars="~129.4M stars",
        description="Dense coverage for typical narrow-field solves and fainter guide stars.",
        recommended=True,
        unit_label="stars",
        coming_soon=True,
    ),
    GroupDef(
        key="astro_very_faint",
        label="Very Faint  (G 18–19)",
        mag_lo=18.0, mag_hi=19.0,
        filenames=_astro_filenames(18.0, 19.0, 0.1),
        est_size="~29.2 GB  (10 files)",
        est_stars="~239.2M stars",
        description="For deep, narrow-field solves needing very dense star fields.",
        unit_label="stars",
        coming_soon=True,
    ),
    GroupDef(
        key="astro_extreme",
        label="Extreme  (G 19–20)",
        mag_lo=19.0, mag_hi=20.0,
        filenames=_astro_filenames(19.0, 20.0, 0.1),
        est_size="~51.2 GB  (10 files)",
        est_stars="~420.2M stars",
        description="Rarely needed — extremely deep solving fields only.",
        warning="Large download — ~51 GB total across 10 files.",
        unit_label="stars",
        coming_soon=True,
    ),
    GroupDef(
        key="astro_max_depth",
        label="Maximum Depth  (G ≥ 20, full DR3)",
        mag_lo=20.0, mag_hi=None,
        filenames=_astro_filenames(20.0, 21.0, 0.1) + [
            "gaia_astro_210_213.sqlite", "gaia_astro_gt213.sqlite",
        ],
        est_size="~80.1 GB  (12 files)",
        est_stars="~656.8M stars",
        description="Full DR3 depth — the practical faint limit of the catalog. "
                    "Only needed for extreme deep-field work.",
        warning="Very large download — ~80 GB total across 12 files.",
        unit_label="stars",
        coming_soon=True,
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
        for path in sorted(self._dir.glob("gaia_astro_dec_*.sqlite")):
            fname = path.name
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

        if getattr(g, "coming_soon", False):
            self._lbl_badge.setText("Coming Soon")
            self._lbl_badge.setStyleSheet(
                "font-size: 10px; color: #8899cc; font-weight: bold; "
                "border: 1px solid #8899cc; border-radius: 3px; padding: 1px 5px;")
            self._lbl_badge.setVisible(True)
            self._dot.setStyleSheet("color: #444; font-size: 14px;")
            self._lbl_stats.setText(
                f"Not yet available  ·  {g.est_stars}  ·  {g.est_size}"
            )
            if g.warning:
                self._lbl_warning.setText(f"⚠  {g.warning}")
                self._lbl_warning.setVisible(True)
            self._btn_install.setVisible(False)
            self._btn_browse.setVisible(False)
            self._btn_remove.setVisible(False)
            self._btn_cancel.setVisible(False)
            return

        if g.recommended:
            self._lbl_badge.setText("Recommended")
            self._lbl_badge.setVisible(True)

        if g.warning:
            self._lbl_warning.setText(f"⚠  {g.warning}")
            self._lbl_warning.setVisible(True)

        if status.fully_installed:
            self._dot.setStyleSheet("color: #44cc66; font-size: 14px;")
            self._lbl_stats.setText(
                f"{status.total_spectra:,} {unit}  ·  "
                f"{status.total_mb / 1024:.1f} GB on disk  ·  "
                f"{len(status.installed)}/{len(g.filenames)} files"
            )
            self._btn_install.setVisible(False)
            self._btn_browse.setVisible(False)
            self._btn_remove.setVisible(True)

        elif status.partially_installed:
            self._dot.setStyleSheet("color: #ddaa44; font-size: 14px;")
            self._lbl_stats.setText(
                f"Partial: {len(status.installed)}/{len(g.filenames)} files  ·  "
                f"{status.total_spectra:,} {unit}  ·  "
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


class _ItemsCountWorker(QThread):
    """
    Counts rows across installed group files. Generalized over the previous
    _SpectraCountWorker: takes the group-def list and table name to count
    ("spectra" for XP groups, "stars" for astrometric groups) as params so
    one worker class serves both tabs.
    """
    progress = _Signal(int)
    finished = _Signal(dict, int)

    def __init__(self, library_dir: Path, group_defs: List[GroupDef],
                 table_name: str, parent=None):
        super().__init__(parent)
        self._dir        = library_dir
        self._group_defs = group_defs
        self._table      = table_name
        self._cancel     = False

    def cancel(self):
        self._cancel = True

    def run(self):
        group_counts: dict[str, int] = {g.key: 0 for g in self._group_defs}
        running_total = 0

        for g in self._group_defs:
            for fname in g.filenames:
                if self._cancel:
                    self.finished.emit(group_counts, running_total)
                    return
                path = self._dir / fname
                if not path.exists():
                    continue
                try:
                    conn = sqlite3.connect(str(path), check_same_thread=True)
                    conn.execute("PRAGMA query_only = ON;")
                    n = conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]
                    conn.close()
                    group_counts[g.key] += n
                    running_total += n
                    self.progress.emit(running_total)
                except Exception:
                    pass

        self.finished.emit(group_counts, running_total)


# ══════════════════════════════════════════════════════════════════════════════
#  Main dialog
# ══════════════════════════════════════════════════════════════════════════════

class GaiaDatabaseDialog(QDialog):
    library_changed = _Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaia DR3 Library")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setMinimumSize(800, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._library       = get_library()
        self._astro_library = get_astro_library()

        self._workers:       Dict[str, _GroupDownloadWorker] = {}
        self._rows:          Dict[str, _GroupRowWidget]      = {}
        self._astro_workers: Dict[str, _GroupDownloadWorker] = {}
        self._astro_rows:    Dict[str, _GroupRowWidget]      = {}

        self._count_worker       = None
        self._astro_count_worker = None

        self._build_ui()
        self._refresh_groups("spectral")
        self._refresh_groups("astro")

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

    def _table_name(self, kind: str) -> str:
        return "spectra" if kind == "spectral" else "stars"

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
        root.addWidget(tabs, stretch=1)

        tabs.addTab(self._build_group_tab("spectral"), "Spectrum Library")
        tabs.addTab(self._build_group_tab("astro"), "Astrometry")

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
        tabs.addTab(viewer_tab, "Spectrum Viewer")

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
        <p>The Astrometry tab (under construction) will install full-depth Gaia DR3
        <b>gaia_source</b> astrometry (RA, Dec, proper motion, parallax) split by
        magnitude, mirroring the Spectrum Library tab above — a much denser catalog
        than the XP spectral library, since it covers the full DR3 source list rather
        than only sources bright/well-measured enough to have usable BP/RP spectra.</p>
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
                "the Spectrum Library tab."
            )

        info_lbl = QLabel(info_text)
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #778; font-size: 11px; margin-bottom: 4px;")
        layout.addWidget(info_lbl)

        if kind == "astro":
            banner = QLabel(
                "🚧 <b>Under construction.</b> This tab previews the planned structure "
                "for the astrometric library — magnitude tiers, same layout as the "
                "Spectrum Library tab. Sizes and star counts shown are rough estimates; "
                "the actual files aren't hosted yet. Check back soon!"
            )
            banner.setWordWrap(True)
            banner.setStyleSheet(
                "background: #221a08; color: #ddaa55; border: 1px solid #443311; "
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

        # Totals + count row
        total_row = QHBoxLayout()
        total_lbl = QLabel("")
        total_lbl.setStyleSheet("font-size: 11px; color: #556688;")
        total_row.addWidget(total_lbl, stretch=1)
        btn_count = QPushButton("Count")
        btn_count.setFixedWidth(60)
        btn_count.setStyleSheet(
            "QPushButton { background: #1a1a2e; color: #556688; border: 1px solid #2a2a3e; "
            "border-radius: 3px; padding: 3px 8px; font-size: 10px; }"
            "QPushButton:hover { background: #222240; color: #8899ee; }")
        btn_count.clicked.connect(
            self._run_count_spectral if kind == "spectral"
            else self._run_count_astro)
        total_row.addWidget(btn_count)
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

    def _run_count_spectral(self):
        self._run_count("spectral")

    def _run_count_astro(self):
        self._run_count("astro")

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

        self._total_lbl_for(kind).setText(
            f"{self._n_installed_groups(kind)}/{len(self._group_defs(kind))} groups installed  ·  "
            f"click Count to tally {self._table_name(kind)}")

    def _n_installed_groups(self, kind: str = "spectral") -> int:
        lib_dir = self._lib_dir(kind)
        return sum(1 for g in self._group_defs(kind)
                   if all((lib_dir / f).exists() for f in g.filenames))

    def _run_count(self, kind: str = "spectral"):
        attr = "_count_worker" if kind == "spectral" else "_astro_count_worker"
        existing = getattr(self, attr, None)
        if existing is not None and existing.isRunning():
            return

        self._total_lbl_for(kind).setText(
            f"{self._n_installed_groups(kind)}/{len(self._group_defs(kind))} groups installed  ·  "
            f"counting… 0 {self._table_name(kind)}")

        worker = _ItemsCountWorker(
            self._lib_dir(kind), self._group_defs(kind), self._table_name(kind), parent=None)
        worker.progress.connect(lambda total, k=kind: self._on_count_progress(k, total))
        worker.finished.connect(lambda counts, total, k=kind: self._on_items_counted(k, counts, total))
        worker.start()
        setattr(self, attr, worker)

    def _on_count_progress(self, kind: str, running_total: int):
        self._total_lbl_for(kind).setText(
            f"{self._n_installed_groups(kind)}/{len(self._group_defs(kind))} groups installed  ·  "
            f"counting… {running_total:,} {self._table_name(kind)}")

    def _on_items_counted(self, kind: str, group_counts: dict, total: int):
        lib_dir = self._lib_dir(kind)
        rows = self._rows_dict(kind)
        for key, row in rows.items():
            g = next((g for g in self._group_defs(kind) if g.key == key), None)
            if g is None:
                continue
            installed = [f for f in g.filenames if (lib_dir / f).exists()]
            missing   = [f for f in g.filenames if not (lib_dir / f).exists()]
            total_mb  = sum(
                (lib_dir / f).stat().st_size / (1024 * 1024)
                for f in installed
            )
            status = GroupStatus(
                group=g, installed=installed, missing=missing,
                total_mb=total_mb, total_spectra=group_counts.get(key, 0),
            )
            row.update_status(status)

        self._total_lbl_for(kind).setText(
            f"{self._n_installed_groups(kind)}/{len(self._group_defs(kind))} groups installed  ·  "
            f"{total:,} {self._table_name(kind)}")

        attr = "_count_worker" if kind == "spectral" else "_astro_count_worker"
        setattr(self, attr, None)

    def _on_install(self, key: str, kind: str = "spectral"):
        group_def = next(g for g in self._group_defs(kind) if g.key == key)
        if getattr(group_def, "coming_soon", False):
            QMessageBox.information(self, "Coming Soon",
                                    f"{group_def.label} isn't available yet — check back soon!")
            return
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
        if getattr(group_def, "coming_soon", False):
            QMessageBox.information(self, "Coming Soon",
                                    f"{group_def.label} isn't available yet — check back soon!")
            return
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
        for attr in ("_count_worker", "_astro_count_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(2000)   # brief wait; it checks _cancel between files
            setattr(self, attr, None)

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