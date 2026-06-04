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
#  gaia_database.py  —  SASpro Gaia XP Spectral Library
#  Manages bulk-downloaded Gaia DR3 XP spectrum SQLite files.
#  Provides a unified lookup layer:
#    1. Local bulk library (distributed SQLite files)
#    2. Live cache DB   (gaia_xp_cache.sqlite)
#    3. Live archive    (fallback, handled by GaiaDownloader)
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================

from __future__ import annotations

import math
import os
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
    Qt, QThread, QStandardPaths,
    pyqtSignal as _Signal,
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QWidget, QMessageBox, QFileDialog,
    QScrollArea, QFrame, QTabWidget, QTextEdit, QLineEdit, QComboBox,
    QSizePolicy,
)
from PyQt6.QtGui import QColor

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Backblaze public URL ───────────────────────────────────────────────────────
LIBRARY_DOWNLOAD_BASE = "https://f005.backblazeb2.com/file/setiastro-gaia/"

# Gaia XP wavelength grid (nm) — 343 points, 336–1020 nm, 2 nm step
WL_GRID = np.arange(336, 1022, 2, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Group definitions
#  Each group has a list of sub-filenames that together cover its mag range.
#  GaiaBulkLibrary opens whatever sub-files are present — no hardcoded list.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GroupDef:
    key:         str            # internal key e.g. "medium"
    label:       str            # display label
    mag_lo:      Optional[float]
    mag_hi:      float
    filenames:   List[str]      # ordered list of sub-files
    est_size:    str            # total estimated download size
    est_stars:   str            # approximate star count
    description: str
    recommended: bool = False
    warning:     str  = ""      # shown in orange if non-empty


GROUP_DEFS: List[GroupDef] = [
    GroupDef(
        key="ultra_bright",
        label="Ultra-Bright  (G < 8)",
        mag_lo=None, mag_hi=8.0,
        filenames=["gaia_xp_lt8.sqlite"],
        est_size="~220 MB",
        est_stars="~55k stars",
        description="Bright stars from G≈2.2 to G<8 — Arneb, Muphrid, Algieba and tens of "
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
# Fast lookup: filename → group key
_FILENAME_TO_GROUP: Dict[str, str] = {
    fn: g.key for g in GROUP_DEFS for fn in g.filenames
}


# ══════════════════════════════════════════════════════════════════════════════
#  Library directory
# ══════════════════════════════════════════════════════════════════════════════

def get_library_dir() -> Path:
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    d = base / "gaia_library"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  Data layer
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
    """Runtime status of one group."""
    group:        GroupDef
    installed:    List[str]     # sub-files present on disk
    missing:      List[str]     # sub-files not yet downloaded
    total_mb:     float         # disk usage of installed files
    total_spectra: int          # spectra across installed files

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
    """
    Manages the collection of bulk Gaia XP SQLite files.
    Dynamically discovers installed files — no hardcoded filename list required.
    """

    def __init__(self, library_dir: Optional[Path] = None):
        self._dir = library_dir or get_library_dir()
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._open_installed()

    def _open_installed(self):
        """Open all gaia_xp_*.sqlite files found in library_dir."""
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
        self._open_installed()

    def close(self):
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

    # ── Lookups ───────────────────────────────────────────────────────────

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

    def find_nearest(self, ra: float, dec: float,
                     radius_arcsec: float = 3.0) -> Optional[Tuple[int, float]]:
        """Return (source_id, sep_arcsec) of the closest source, or None."""
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

    # ── Status ────────────────────────────────────────────────────────────

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
        """Close and remove one file's connection (before delete)."""
        conn = self._connections.pop(filename, None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ── Singleton ─────────────────────────────────────────────────────────────────

_library_instance: Optional[GaiaBulkLibrary] = None


def get_library() -> GaiaBulkLibrary:
    global _library_instance
    if _library_instance is None:
        _library_instance = GaiaBulkLibrary()
    return _library_instance


def refresh_library():
    global _library_instance
    if _library_instance is not None:
        _library_instance.refresh()
    else:
        _library_instance = GaiaBulkLibrary()


# ══════════════════════════════════════════════════════════════════════════════
#  Download worker  — downloads one sub-file at a time
# ══════════════════════════════════════════════════════════════════════════════

class _FileDownloadWorker(QThread):
    progress  = _Signal(int, int, str)   # bytes_done, bytes_total, msg
    finished  = _Signal(bool, str)       # success, message
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
    """
    Downloads all missing sub-files for one group sequentially.
    Emits file-level and overall progress.
    """
    file_progress    = _Signal(int, int, str)   # bytes, total, msg
    file_done        = _Signal(str, bool)        # filename, success
    group_progress   = _Signal(int, int)         # files_done, files_total
    group_finished   = _Signal(bool, str)        # success, message
    cancelled        = _Signal()

    def __init__(self, group: GroupDef, missing: List[str],
                 dest_dir: Path, parent=None):
        super().__init__(parent)
        self._group   = group
        self._missing = missing
        self._dir     = dest_dir
        self._cancel  = False

    def cancel(self):
        self._cancel = True

    def run(self):
        total_files = len(self._missing)
        done_files  = 0

        for fname in self._missing:
            if self._cancel:
                self.cancelled.emit()
                return

            url  = LIBRARY_DOWNLOAD_BASE + fname
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
                                done, total,
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
#  Spectrum viewer widget
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
                      ha="center", va="center", color="#555", fontsize=12)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for sp in self._ax.spines.values():
            sp.set_color("#333")
        self._fig.tight_layout(pad=0.5)
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
#  Group row widget
# ══════════════════════════════════════════════════════════════════════════════

class _GroupRowWidget(QFrame):
    install_requested = _Signal(str)   # group key
    remove_requested  = _Signal(str)   # group key
    browse_requested  = _Signal(str)   # group key

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

        # Status dot
        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._dot)

        # Info block
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

        # Progress bar
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

        # File counter label (shows "3/10 files" during download)
        self._lbl_file_count = QLabel()
        self._lbl_file_count.setStyleSheet(
            "font-size: 10px; color: #6688aa; border: none;")
        self._lbl_file_count.setVisible(False)
        outer.addWidget(self._lbl_file_count)

        # Buttons
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
            self._lbl_stats.setText(
                f"{status.total_spectra:,} spectra  ·  "
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
                f"{status.total_spectra:,} spectra  ·  "
                f"{status.total_mb / 1024:.1f} GB on disk  —  "
                f"click Install to resume"
            )
            self._btn_install.setText("Resume")
            self._btn_install.setVisible(True)
            self._btn_browse.setVisible(False)
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

    def update_file_progress(self, done: int, total: int, msg: str):
        self._progress.setVisible(True)
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(done)
        else:
            self._progress.setMaximum(0)
        # Show filename on the bar (first line of msg)
        short = msg.split("\n")[0] if "\n" in msg else msg
        self._progress.setFormat(short[-40:])

    def update_group_progress(self, done: int, total: int):
        self._lbl_file_count.setText(f"{done}/{total} files")


# ══════════════════════════════════════════════════════════════════════════════
#  Main dialog
# ══════════════════════════════════════════════════════════════════════════════

class GaiaDatabaseDialog(QDialog):
    library_changed = _Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaia XP Spectral Library")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setMinimumSize(800, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._library  = get_library()
        self._workers: Dict[str, _GroupDownloadWorker] = {}
        self._rows:    Dict[str, _GroupRowWidget]      = {}

        self._build_ui()
        self._refresh_groups()

    # ── UI ────────────────────────────────────────────────────────────────

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
        title_lbl = QLabel("Gaia XP Spectral Library")
        title_lbl.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #8899ee; "
            "letter-spacing: 1px; border: none;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch()
        self._total_lbl = QLabel("")
        self._total_lbl.setStyleSheet(
            "font-size: 11px; color: #556688; border: none;")
        hdr_layout.addWidget(self._total_lbl)
        root.addWidget(hdr)

        # ── Tabs ──────────────────────────────────────────────────────────
        tabs = QTabWidget()
        root.addWidget(tabs, stretch=1)

        # Tab 1: Library
        lib_tab    = QWidget()
        lib_layout = QVBoxLayout(lib_tab)
        lib_layout.setContentsMargins(16, 16, 16, 16)
        lib_layout.setSpacing(8)

        info_lbl = QLabel(
            "Install Gaia DR3 XP spectral library files for offline color calibration. "
            "SASpro checks these before hitting the live Gaia archive. "
            "Interrupted downloads resume automatically — just click Install again."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #778; font-size: 11px; margin-bottom: 4px;")
        lib_layout.addWidget(info_lbl)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Library folder:"))
        self._dir_lbl = QLabel(str(get_library_dir()))
        self._dir_lbl.setStyleSheet(
            "color: #6688aa; font-size: 11px; font-family: monospace;")
        self._dir_lbl.setWordWrap(True)
        dir_row.addWidget(self._dir_lbl, stretch=1)
        btn_open_dir = QPushButton("Open Folder")
        btn_open_dir.setFixedWidth(100)
        btn_open_dir.clicked.connect(self._open_library_dir)
        dir_row.addWidget(btn_open_dir)
        lib_layout.addLayout(dir_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        band_container = QWidget()
        band_container.setStyleSheet("background: transparent;")
        self._group_layout = QVBoxLayout(band_container)
        self._group_layout.setSpacing(8)
        self._group_layout.addStretch()
        scroll.setWidget(band_container)
        lib_layout.addWidget(scroll, stretch=1)
        tabs.addTab(lib_tab, "Library")

        # Tab 2: Spectrum Viewer
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

        # Name row
        self._name_row   = QWidget()
        name_layout = QHBoxLayout(self._name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        self._name_edit  = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Arneb, Muphrid, Algieba, HD 12345…")
        self._name_edit.returnPressed.connect(self._lookup_spectrum)
        name_layout.addWidget(self._name_edit, stretch=1)
        btn_name = QPushButton("Look Up")
        btn_name.clicked.connect(self._lookup_spectrum)
        name_layout.addWidget(btn_name)
        viewer_layout.addWidget(self._name_row)

        # RA/Dec row
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

        # Source ID row
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

        # Tab 3: About
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

    # ── Group management ──────────────────────────────────────────────────

    def _refresh_groups(self):
        for row in self._rows.values():
            self._group_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()

        for status in self._library.get_group_status():
            row = _GroupRowWidget(status)
            row.install_requested.connect(self._on_install)
            row.remove_requested.connect(self._on_remove)
            row.browse_requested.connect(self._on_browse)
            self._group_layout.insertWidget(self._group_layout.count() - 1, row)
            self._rows[status.group.key] = row

        total = self._library.total_spectra()
        n_groups = sum(1 for s in self._library.get_group_status() if s.fully_installed)
        self._total_lbl.setText(
            f"{n_groups}/{len(GROUP_DEFS)} groups installed  ·  {total:,} spectra")

    def _on_install(self, key: str):
        group_def = next(g for g in GROUP_DEFS if g.key == key)
        statuses  = {s.group.key: s for s in self._library.get_group_status()}
        status    = statuses[key]
        missing   = status.missing

        if not missing:
            return  # already fully installed

        row = self._rows.get(key)
        if row is None:
            return

        worker = _GroupDownloadWorker(group_def, missing, get_library_dir(), parent=self)
        self._workers[key] = worker

        row.set_downloading(True, cancel_cb=worker.cancel)

        worker.file_progress.connect(
            lambda d, t, m, r=row: r.update_file_progress(d, t, m))
        worker.group_progress.connect(
            lambda d, t, r=row: r.update_group_progress(d, t))
        worker.file_done.connect(
            lambda fname, ok: self._on_file_done(fname, ok))
        worker.group_finished.connect(
            lambda ok, msg, k=key: self._on_group_finished(k, ok, msg))
        worker.cancelled.connect(
            lambda k=key: self._on_cancelled(k))
        worker.start()

    def _on_file_done(self, fname: str, ok: bool):
        """A single sub-file finished — refresh library so it's queryable."""
        if ok:
            refresh_library()
            self._library = get_library()

    def _on_group_finished(self, key: str, ok: bool, msg: str):
        row = self._rows.get(key)
        if row:
            row.set_downloading(False)

        worker = self._workers.pop(key, None)
        if worker:
            worker.wait()

        refresh_library()
        self._library = get_library()
        self._refresh_groups()
        self.library_changed.emit()

        if ok:
            QMessageBox.information(self, "Install Complete", msg)
        else:
            QMessageBox.critical(self, "Download Failed", msg)

    def _on_cancelled(self, key: str):
        row = self._rows.get(key)
        if row:
            row.set_downloading(False)
        self._workers.pop(key, None)
        refresh_library()
        self._library = get_library()
        self._refresh_groups()

    def _on_remove(self, key: str):
        group_def = next(g for g in GROUP_DEFS if g.key == key)
        statuses  = {s.group.key: s for s in self._library.get_group_status()}
        installed = statuses[key].installed

        total_gb = statuses[key].total_mb / 1024
        reply = QMessageBox.question(
            self, "Remove Group",
            f"Delete all installed files for '{group_def.label}'?\n\n"
            f"{len(installed)} file(s) · {total_gb:.1f} GB will be freed.\n\n"
            f"SASpro will fall back to live Gaia archive downloads for "
            f"sources in this magnitude range.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for fname in installed:
            self._library.close_file(fname)
            path = get_library_dir() / fname
            try:
                path.unlink(missing_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete {fname}:\n{e}")
                break

        refresh_library()
        self._library = get_library()
        self._refresh_groups()
        self.library_changed.emit()

    def _on_browse(self, key: str):
        """Let user point to a locally-built SQLite file for one group."""
        group_def = next(g for g in GROUP_DEFS if g.key == key)
        statuses  = {s.group.key: s for s in self._library.get_group_status()}
        missing   = statuses[key].missing

        if not missing:
            QMessageBox.information(self, "Already Installed",
                                    f"{group_def.label} is already fully installed.")
            return

        QMessageBox.information(
            self, "Browse for Files",
            f"Select the sub-files for '{group_def.label}' one at a time.\n\n"
            f"Missing files:\n" + "\n".join(f"  {f}" for f in missing)
        )

        for fname in missing:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Locate {fname}", str(Path.home()),
                f"{fname};;SQLite files (*.sqlite);;All files (*)")
            if not path:
                continue
            import shutil
            dest = get_library_dir() / fname
            try:
                shutil.copy2(path, dest)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not copy {fname}:\n{e}")
                continue

        refresh_library()
        self._library = get_library()
        self._refresh_groups()
        self.library_changed.emit()

    def _open_library_dir(self):
        import subprocess, sys
        d = str(get_library_dir())
        if sys.platform == "win32":
            os.startfile(d)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", d])
        else:
            subprocess.Popen(["xdg-open", d])

    # ── Spectrum viewer ───────────────────────────────────────────────────

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
                from astropy.coordinates import SkyCoord
                coord = SkyCoord.from_name(name)
                ra, dec = coord.ra.deg, coord.dec.deg
                label   = name
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

        # Spectrum lookup
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

            # For name searches, try progressively larger radii since SIMBAD
            # and Gaia coordinate epochs can differ by arcseconds for bright stars
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

        # Fall back to live cache
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
        for worker in list(self._workers.values()):
            worker.cancel()
            worker.wait()
        self._workers.clear()
        super().closeEvent(event)


# ── Public entry point ────────────────────────────────────────────────────────

def open_gaia_database_dialog(parent=None) -> GaiaDatabaseDialog:
    dlg = GaiaDatabaseDialog(parent=parent)
    dlg.show()
    return dlg