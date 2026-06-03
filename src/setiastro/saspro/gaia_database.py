#src/setiastro/saspro/gaia_database.py
#!/usr/bin/env python3
# ======================================================
#   _____ __  __ ___   ___      _        _
#  / ____|  \/  |__ \ / _ \    | |      | |
# | |  __| \  / |  ) | | | |   | |   ___| |__
# | | |_ | |\/| | / /| | | |   | |  / _ \ '_ \
# | |__| | |  | |/ /_| |_| |   | |_|  __/ |_) |
#  \_____|_|  |_|____|\___/    |_____\___|_.__/
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

import os
import re
import sqlite3
import time
import zlib
from dataclasses import dataclass
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
    QPushButton, QProgressBar, QWidget, QGroupBox, QGridLayout,
    QMessageBox, QFileDialog, QSizePolicy, QScrollArea, QFrame,
    QTabWidget, QTextEdit, QSpacerItem,
)
from PyQt6.QtGui import QFont, QColor, QPalette

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Band definitions (must match gaia_bulk_builder.py) ────────────────────────

BAND_DEFS: List[Dict] = [
    {
        "filename": "gaia_xp_lt8.sqlite",
        "label":    "Ultra-Bright  (G < 8)",
        "mag_lo":   None,
        "mag_hi":   8.0,
        "est_stars": "~60k stars",
        "est_size":  "~35 MB",
        "description": "Brightest stars — Vega, Sirius, and their ilk. "
                        "Essential for any colour calibration.",
    },
    {
        "filename": "gaia_xp_8_10.sqlite",
        "label":    "Bright  (G 8–10)",
        "mag_lo":   8.0,
        "mag_hi":   10.0,
        "est_stars": "~415k stars",
        "est_size":  "~200 MB",
        "description": "Covers most calibration stars reachable from backyard setups.",
    },
    {
        "filename": "gaia_xp_10_12.sqlite",
        "label":    "Medium  (G 10–12)",
        "mag_lo":   10.0,
        "mag_hi":   12.0,
        "est_stars": "~2.5M stars",
        "est_size":  "~1.2 GB",
        "description": "Dense coverage — recommended for wide-field imaging.",
    },
    {
        "filename": "gaia_xp_12_14.sqlite",
        "label":    "Faint  (G 12–14)",
        "mag_lo":   12.0,
        "mag_hi":   14.0,
        "est_stars": "~13M stars",
        "est_size":  "~6.5 GB",
        "description": "Deep coverage for long-exposure narrowband work.",
    },
    {
        "filename": "gaia_xp_14_155.sqlite",
        "label":    "Very Faint  (G 14–15.5)",
        "mag_lo":   14.0,
        "mag_hi":   15.5,
        "est_stars": "~19M stars",
        "est_size":  "~9.5 GB",
        "description": "Maximum depth. Large download — optional for most users.",
    },
]

# Gaia XP wavelength grid (nm)
WL_GRID = np.arange(336, 1022, 2, dtype=np.float32)   # 343 points

# ── Download base URL (update when you host the files) ────────────────────────
LIBRARY_DOWNLOAD_BASE = "https://setiastro.com/gaia_library/"   # TODO: update when hosted


# ══════════════════════════════════════════════════════════════════════════════
#  Data layer
# ══════════════════════════════════════════════════════════════════════════════

def get_library_dir() -> Path:
    """Per-user directory where bulk SQLite files are stored."""
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    d = base / "gaia_library"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class BandInfo:
    """Runtime status of one magnitude band file."""
    filename:    str
    label:       str
    mag_lo:      Optional[float]
    mag_hi:      float
    est_stars:   str
    est_size:    str
    description: str
    path:        Path
    installed:   bool
    spectra:     int    = 0     # actual row count if installed
    size_mb:     float  = 0.0   # actual file size if installed


class GaiaBulkLibrary:
    """
    Manages the collection of bulk Gaia XP SQLite files.
    Provides O(1) spectrum lookup across all installed bands.

    Usage:
        lib = GaiaBulkLibrary()
        spec = lib.get_spectrum(source_id)   # CalibratedSpectrum or None
    """

    def __init__(self, library_dir: Optional[Path] = None):
        self._dir = library_dir or get_library_dir()
        self._connections: Dict[str, sqlite3.Connection] = {}
        self._open_installed()

    # ── Connection management ─────────────────────────────────────────────

    def _open_installed(self):
        """Open SQLite connections for all installed band files."""
        for band in BAND_DEFS:
            path = self._dir / band["filename"]
            if path.exists() and band["filename"] not in self._connections:
                try:
                    conn = sqlite3.connect(str(path), check_same_thread=False)
                    conn.execute("PRAGMA query_only = ON;")
                    self._connections[band["filename"]] = conn
                except Exception as e:
                    print(f"[GaiaBulkLibrary] Could not open {band['filename']}: {e}")

    def refresh(self):
        """Re-scan for newly installed files."""
        self._open_installed()

    def close(self):
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:
                pass
        self._connections.clear()

    def find_nearest(self, ra: float, dec: float, radius_arcsec: float = 3.0
                    ) -> Optional[Tuple[int, float]]:
        """
        Return (source_id, sep_arcsec) of the closest source within radius,
        or None if nothing found.
        """
        radius_deg = radius_arcsec / 3600.0
        cosd = max(1e-6, abs(math.cos(math.radians(dec))))
        
        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT source_id, ra, dec FROM sources
                    WHERE dec BETWEEN ? AND ?
                    AND ra BETWEEN ? AND ?
                """, (
                    dec - radius_deg, dec + radius_deg,
                    ra - radius_deg / cosd, ra + radius_deg / cosd,
                ))
                best_sid, best_sep = None, 1e9
                for sid, sra, sdec in cur.fetchall():
                    dra  = (sra - ra) * cosd
                    ddec = sdec - dec
                    sep  = math.hypot(dra, ddec) * 3600.0
                    if sep < best_sep:
                        best_sep, best_sid = sep, sid
                if best_sid is not None and best_sep <= radius_arcsec:
                    return (int(best_sid), float(best_sep))
            except Exception:
                continue
        return None

    # ── Spectrum lookup ───────────────────────────────────────────────────

    def get_spectrum(self, source_id: int) -> Optional["CalibratedSpectrum"]:
        """
        Look up a spectrum by Gaia source_id across all installed band files.
        Returns CalibratedSpectrum or None if not found.
        """
        sid = int(source_id)
        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT flux_compressed, flux_error_compressed "
                    "FROM spectra WHERE source_id=?", (sid,)
                )
                row = cur.fetchone()
                if row is not None:
                    flux = _decompress(row[0])
                    flux_err = _decompress(row[1]) if row[1] else None
                    return CalibratedSpectrum(
                        source_id=sid,
                        wavelengths=WL_GRID.copy(),
                        flux=flux,
                        flux_error=flux_err,
                    )
            except Exception:
                continue
        return None

    def get_source_info(self, source_id: int) -> Optional[Dict]:
        """Return ra, dec, phot_g_mean_mag for a source_id."""
        sid = int(source_id)
        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT ra, dec, phot_g_mean_mag FROM sources WHERE source_id=?",
                    (sid,)
                )
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
                cur.execute(
                    "SELECT 1 FROM spectra WHERE source_id=? LIMIT 1", (sid,)
                )
                if cur.fetchone():
                    return True
            except Exception:
                continue
        return False

    # ── Band status ───────────────────────────────────────────────────────

    def get_band_info(self) -> List[BandInfo]:
        """Return status of all bands (installed or not)."""
        result = []
        for band in BAND_DEFS:
            path = self._dir / band["filename"]
            installed = path.exists()
            spectra = 0
            size_mb = 0.0
            if installed:
                size_mb = path.stat().st_size / (1024 * 1024)
                if band["filename"] in self._connections:
                    try:
                        cur = self._connections[band["filename"]].cursor()
                        cur.execute("SELECT COUNT(*) FROM spectra")
                        spectra = cur.fetchone()[0]
                    except Exception:
                        pass
            result.append(BandInfo(
                filename=band["filename"],
                label=band["label"],
                mag_lo=band["mag_lo"],
                mag_hi=band["mag_hi"],
                est_stars=band["est_stars"],
                est_size=band["est_size"],
                description=band["description"],
                path=path,
                installed=installed,
                spectra=spectra,
                size_mb=size_mb,
            ))
        return result

    def total_spectra(self) -> int:
        total = 0
        for conn in self._connections.values():
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM spectra")
                total += cur.fetchone()[0]
            except Exception:
                pass
        return total

    def installed_bands(self) -> List[str]:
        return list(self._connections.keys())


# ── Compression helpers ────────────────────────────────────────────────────────

def _decompress(data: bytes) -> np.ndarray:
    return np.frombuffer(zlib.decompress(data), dtype=np.float32)


@dataclass
class CalibratedSpectrum:
    source_id:  int
    wavelengths: np.ndarray
    flux:        np.ndarray
    flux_error:  Optional[np.ndarray] = None

    def get_flux_at(self, wavelength_nm: float) -> float:
        return float(np.interp(wavelength_nm, self.wavelengths, self.flux))


# ── Singleton library instance ────────────────────────────────────────────────

_library_instance: Optional[GaiaBulkLibrary] = None


def get_library() -> GaiaBulkLibrary:
    """Return the process-wide GaiaBulkLibrary singleton."""
    global _library_instance
    if _library_instance is None:
        _library_instance = GaiaBulkLibrary()
    return _library_instance


def refresh_library():
    """Refresh after a new band is installed."""
    global _library_instance
    if _library_instance is not None:
        _library_instance.refresh()
    else:
        _library_instance = GaiaBulkLibrary()


# ══════════════════════════════════════════════════════════════════════════════
#  Download worker
# ══════════════════════════════════════════════════════════════════════════════

class _BandDownloadWorker(QThread):
    """Downloads one band SQLite file with progress reporting."""

    progress  = _Signal(int, int, str)   # bytes_done, bytes_total, msg
    finished  = _Signal(bool, str)       # success, message
    cancelled = _Signal()

    def __init__(self, url: str, dest_path: Path, parent=None):
        super().__init__(parent)
        self._url       = url
        self._dest      = dest_path
        self._cancel    = False

    def cancel(self):
        self._cancel = True

    def run(self):
        tmp = self._dest.with_suffix(".tmp")
        try:
            req = Request(self._url, headers={"User-Agent": "SASpro-GaiaLibrary/1.0"})
            with urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done  = 0
                chunk = 512 * 1024   # 512 KB chunks

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

                        pct  = int(done / total * 100) if total else 0
                        done_mb  = done  / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        self.progress.emit(
                            done, total,
                            f"Downloading… {done_mb:.1f} / {total_mb:.1f} MB  ({pct}%)"
                        )

            tmp.rename(self._dest)
            self.finished.emit(True, f"Download complete: {self._dest.name}")

        except Exception as e:
            tmp.unlink(missing_ok=True)
            self.finished.emit(False, f"Download failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Spectrum viewer widget
# ══════════════════════════════════════════════════════════════════════════════

class SpectrumViewerWidget(QWidget):
    """
    Embeddable widget that plots a Gaia XP spectrum.
    Pass a CalibratedSpectrum to show().
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectrum: Optional[CalibratedSpectrum] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_MPL:
            self._fig = Figure(figsize=(6, 3), facecolor="#1a1a2e")
            self._ax  = self._fig.add_subplot(111)
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
        self._ax.text(
            0.5, 0.5, "No spectrum loaded",
            transform=self._ax.transAxes,
            ha="center", va="center",
            color="#555", fontsize=12,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_color("#333")
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()

    def show_spectrum(self, spectrum: CalibratedSpectrum, title: str = ""):
        if not HAS_MPL:
            return
        self._spectrum = spectrum
        wl   = spectrum.wavelengths
        flux = spectrum.flux

        self._ax.clear()
        self._ax.set_facecolor("#1a1a2e")
        self._fig.patch.set_facecolor("#1a1a2e")

        # Colour the spectrum by wavelength band
        # BP: 330–680 nm  RP: 640–1050 nm  overlap blended
        bp_mask = wl <= 680
        rp_mask = wl >= 640

        if np.any(bp_mask):
            self._ax.fill_between(wl[bp_mask], flux[bp_mask], alpha=0.25, color="#5599ff")
            self._ax.plot(wl[bp_mask], flux[bp_mask], color="#88bbff", lw=1.2)
        if np.any(rp_mask):
            self._ax.fill_between(wl[rp_mask], flux[rp_mask], alpha=0.25, color="#ff6644")
            self._ax.plot(wl[rp_mask], flux[rp_mask], color="#ffaa88", lw=1.2)

        # Flux error shading
        if spectrum.flux_error is not None:
            self._ax.fill_between(
                wl,
                flux - spectrum.flux_error,
                flux + spectrum.flux_error,
                alpha=0.12, color="#ffffff"
            )

        self._ax.set_xlabel("Wavelength (nm)", color="#aaa", fontsize=9)
        self._ax.set_ylabel("Flux (W nm⁻¹ m⁻²)", color="#aaa", fontsize=9)
        self._ax.tick_params(colors="#888", labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color("#333")
        self._ax.set_xlim(330, 1030)

        if title:
            self._ax.set_title(title, color="#ccc", fontsize=9, pad=4)

        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()

        peak_wl = float(wl[np.argmax(flux)])
        self._info_lbl.setText(
            f"source_id: {spectrum.source_id}   "
            f"peak: {peak_wl:.0f} nm   "
            f"flux range: {float(np.min(flux)):.2e} – {float(np.max(flux)):.2e}"
        )

    def clear(self):
        self._draw_empty()
        self._info_lbl.setText("")


# ══════════════════════════════════════════════════════════════════════════════
#  Band row widget
# ══════════════════════════════════════════════════════════════════════════════

class _BandRowWidget(QFrame):
    """One row in the library manager showing a magnitude band."""

    install_requested  = _Signal(str)   # filename
    remove_requested   = _Signal(str)   # filename
    browse_requested   = _Signal(str)   # filename

    def __init__(self, band: BandInfo, parent=None):
        super().__init__(parent)
        self._band = band
        self._worker: Optional[_BandDownloadWorker] = None
        self._build_ui()
        self.update_band(band)

    def _build_ui(self):
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            _BandRowWidget {
                border: 1px solid #2a2a3e;
                border-radius: 6px;
                background: #12121f;
            }
        """)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(12)

        # Status indicator dot
        self._dot = QLabel("●")
        self._dot.setFixedWidth(16)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._dot)

        # Info block
        info = QVBoxLayout()
        info.setSpacing(2)
        self._lbl_title = QLabel()
        self._lbl_title.setStyleSheet("font-weight: bold; font-size: 13px; color: #e0e0ff;")
        self._lbl_desc  = QLabel()
        self._lbl_desc.setStyleSheet("font-size: 11px; color: #888;")
        self._lbl_stats = QLabel()
        self._lbl_stats.setStyleSheet("font-size: 11px; color: #6688aa;")
        info.addWidget(self._lbl_title)
        info.addWidget(self._lbl_desc)
        info.addWidget(self._lbl_stats)
        outer.addLayout(info, stretch=1)

        # Progress bar (hidden until download)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333;
                border-radius: 3px;
                background: #0a0a18;
                color: #aaa;
                text-align: center;
                font-size: 10px;
            }
            QProgressBar::chunk { background: #4466cc; border-radius: 2px; }
        """)
        outer.addWidget(self._progress)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self._btn_install = QPushButton("Install")
        self._btn_install.setFixedWidth(80)
        self._btn_install.clicked.connect(
            lambda: self.install_requested.emit(self._band.filename))
        self._btn_install.setStyleSheet(self._btn_style("#224488", "#3366cc"))

        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.setFixedWidth(80)
        self._btn_browse.clicked.connect(
            lambda: self.browse_requested.emit(self._band.filename))
        self._btn_browse.setStyleSheet(self._btn_style("#223322", "#336633"))

        self._btn_remove = QPushButton("Remove")
        self._btn_remove.setFixedWidth(80)
        self._btn_remove.clicked.connect(
            lambda: self.remove_requested.emit(self._band.filename))
        self._btn_remove.setStyleSheet(self._btn_style("#442222", "#883333"))

        btn_layout.addWidget(self._btn_install)
        btn_layout.addWidget(self._btn_browse)
        btn_layout.addWidget(self._btn_remove)
        outer.addLayout(btn_layout)

    @staticmethod
    def _btn_style(bg, hover):
        return f"""
            QPushButton {{
                background: {bg};
                color: #ddd;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:disabled {{ color: #555; background: #1a1a2e; }}
        """

    def update_band(self, band: BandInfo):
        self._band = band
        self._lbl_title.setText(band.label)
        self._lbl_desc.setText(band.description)

        if band.installed:
            self._dot.setStyleSheet("color: #44cc66; font-size: 14px;")
            self._lbl_stats.setText(
                f"{band.spectra:,} spectra installed  ·  "
                f"{band.size_mb:.0f} MB on disk"
            )
            self._btn_install.setVisible(False)
            self._btn_browse.setVisible(False)
            self._btn_remove.setVisible(True)
        else:
            self._dot.setStyleSheet("color: #555; font-size: 14px;")
            self._lbl_stats.setText(
                f"Not installed  ·  est. {band.est_stars}  ·  {band.est_size}"
            )
            self._btn_install.setVisible(True)
            self._btn_browse.setVisible(True)
            self._btn_remove.setVisible(False)

    def set_downloading(self, downloading: bool):
        self._btn_install.setEnabled(not downloading)
        self._btn_browse.setEnabled(not downloading)
        self._progress.setVisible(downloading)
        if not downloading:
            self._progress.setValue(0)

    def update_progress(self, done: int, total: int, msg: str):
        self._progress.setVisible(True)
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(done)
        else:
            self._progress.setMaximum(0)
        self._progress.setFormat(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  Main dialog
# ══════════════════════════════════════════════════════════════════════════════

class GaiaDatabaseDialog(QDialog):
    """
    Gaia XP Spectral Library manager.

    Shows installed bands, allows install/remove, and has a
    spectrum viewer for exploring individual stars.
    """

    library_changed = _Signal()   # emitted when a band is installed/removed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaia XP Spectral Library")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setMinimumSize(760, 580)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._library = get_library()
        self._workers: Dict[str, _BandDownloadWorker] = {}
        self._band_rows: Dict[str, _BandRowWidget] = {}

        self._build_ui()
        self._refresh_bands()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet("""
            QDialog {
                background: #0d0d1a;
                color: #ddd;
            }
            QTabWidget::pane {
                border: 1px solid #2a2a3e;
                background: #0d0d1a;
            }
            QTabBar::tab {
                background: #12121f;
                color: #888;
                padding: 6px 18px;
                border: 1px solid #2a2a3e;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected {
                background: #1a1a2e;
                color: #ccd;
            }
            QGroupBox {
                color: #99aacc;
                border: 1px solid #2a2a3e;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLabel { color: #ccc; }
            QLineEdit {
                background: #12121f;
                color: #ddd;
                border: 1px solid #334;
                border-radius: 3px;
                padding: 3px 6px;
            }
            QPushButton {
                background: #1a1a2e;
                color: #ccc;
                border: 1px solid #334;
                border-radius: 4px;
                padding: 5px 12px;
            }
            QPushButton:hover { background: #222240; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Header
        hdr = QWidget()
        hdr.setStyleSheet("background: #0a0a18; border-bottom: 1px solid #1a1a3e;")
        hdr_layout = QHBoxLayout(hdr)
        hdr_layout.setContentsMargins(20, 14, 20, 14)

        title_lbl = QLabel("Gaia XP Spectral Library")
        title_lbl.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #8899ee; letter-spacing: 1px;")
        hdr_layout.addWidget(title_lbl)
        hdr_layout.addStretch()

        self._total_lbl = QLabel("")
        self._total_lbl.setStyleSheet("font-size: 11px; color: #556688;")
        hdr_layout.addWidget(self._total_lbl)

        root.addWidget(hdr)

        # Tabs
        tabs = QTabWidget()
        root.addWidget(tabs, stretch=1)

        # ── Tab 1: Library ────────────────────────────────────────────────
        lib_tab = QWidget()
        lib_layout = QVBoxLayout(lib_tab)
        lib_layout.setContentsMargins(16, 16, 16, 16)
        lib_layout.setSpacing(8)

        info_lbl = QLabel(
            "Install Gaia DR3 XP spectral library files for offline colour calibration. "
            "SASpro checks these before hitting the live Gaia archive."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #778; font-size: 11px; margin-bottom: 6px;")
        lib_layout.addWidget(info_lbl)

        # Library dir row
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

        # Band rows in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        band_container = QWidget()
        band_container.setStyleSheet("background: transparent;")
        self._band_layout = QVBoxLayout(band_container)
        self._band_layout.setSpacing(6)
        self._band_layout.addStretch()
        scroll.setWidget(band_container)
        lib_layout.addWidget(scroll, stretch=1)

        tabs.addTab(lib_tab, "Library")

        # ── Tab 2: Spectrum Viewer ────────────────────────────────────────
        viewer_tab = QWidget()
        viewer_layout = QVBoxLayout(viewer_tab)
        viewer_layout.setContentsMargins(16, 16, 16, 16)

        search_row = QHBoxLayout()
        from PyQt6.QtWidgets import QLineEdit
        self._sid_edit = QLineEdit()
        self._sid_edit.setPlaceholderText("Enter Gaia source_id…")
        self._sid_edit.returnPressed.connect(self._lookup_spectrum)
        search_row.addWidget(self._sid_edit, stretch=1)
        btn_lookup = QPushButton("Look Up")
        btn_lookup.clicked.connect(self._lookup_spectrum)
        search_row.addWidget(btn_lookup)
        viewer_layout.addLayout(search_row)

        self._viewer_status = QLabel("Enter a Gaia DR3 source_id to view its XP spectrum.")
        self._viewer_status.setStyleSheet("color: #668; font-size: 11px;")
        viewer_layout.addWidget(self._viewer_status)

        self._spectrum_viewer = SpectrumViewerWidget()
        viewer_layout.addWidget(self._spectrum_viewer, stretch=1)

        # Source info box
        self._source_info = QTextEdit()
        self._source_info.setReadOnly(True)
        self._source_info.setMaximumHeight(80)
        self._source_info.setStyleSheet(
            "background: #0a0a18; color: #99aacc; "
            "border: 1px solid #2a2a3e; font-size: 11px; font-family: monospace;"
        )
        viewer_layout.addWidget(self._source_info)

        tabs.addTab(viewer_tab, "Spectrum Viewer")

        # ── Tab 3: About ──────────────────────────────────────────────────
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_layout.setContentsMargins(20, 20, 20, 20)

        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setStyleSheet(
            "background: #0a0a18; color: #aab; border: none; font-size: 12px;")
        about_text.setHtml("""
        <h3 style="color:#8899ee;">Gaia DR3 XP Spectral Library</h3>
        <p>These SQLite databases contain pre-calibrated Gaia BP/RP spectra
        (336–1020 nm, 343 wavelength points, 2 nm step) from Gaia Data Release 3.</p>
        <p>Spectra were extracted from the
        <b>xp_sampled_mean_spectrum</b> table of the Gaia DR3 bulk download
        repository and stored in a compact, compressed format compatible with
        SASpro's spectral photometric colour calibration (SFCC) pipeline.</p>
        <p><b>No GaiaXPy required</b> — the spectra are already calibrated by DPAC.</p>
        <h4 style="color:#8899ee;">Lookup priority in SASpro</h4>
        <ol>
        <li>Local bulk library (these files) — instant, no network</li>
        <li>Live download cache (gaia_xp_cache.sqlite)</li>
        <li>Live Gaia archive — fallback when neither has the source</li>
        </ol>
        <h4 style="color:#8899ee;">Citation</h4>
        <p>Gaia Collaboration et al. (2022), Gaia DR3.
        <i>Astronomy &amp; Astrophysics</i>, 649, A1.<br>
        <a href="https://gea.esac.esa.int/archive/" style="color:#6688cc;">
        https://gea.esac.esa.int/archive/</a></p>
        <p style="color:#556; font-size:10px;">
        Written by Franklin Marek / www.setiastro.com</p>
        """)
        about_layout.addWidget(about_text)
        tabs.addTab(about_tab, "About")


    # ── Band management ───────────────────────────────────────────────────

    def _refresh_bands(self):
        """Rebuild band rows from current library state."""
        # Clear existing rows
        for row in self._band_rows.values():
            self._band_layout.removeWidget(row)
            row.deleteLater()
        self._band_rows.clear()

        bands = self._library.get_band_info()
        for band in bands:
            row = _BandRowWidget(band)
            row.install_requested.connect(self._on_install)
            row.remove_requested.connect(self._on_remove)
            row.browse_requested.connect(self._on_browse)
            # Insert before the stretch
            self._band_layout.insertWidget(self._band_layout.count() - 1, row)
            self._band_rows[band.filename] = row

        total = self._library.total_spectra()
        n_installed = sum(1 for b in bands if b.installed)
        self._total_lbl.setText(
            f"{n_installed}/{len(bands)} bands installed  ·  "
            f"{total:,} spectra"
        )

    def _on_install(self, filename: str):
        url = LIBRARY_DOWNLOAD_BASE + filename
        dest = get_library_dir() / filename

        row = self._band_rows.get(filename)
        if row is None:
            return

        # Check if URL is placeholder
        if "setiastro.com" in url and LIBRARY_DOWNLOAD_BASE == "https://setiastro.com/gaia_library/":
            QMessageBox.information(
                self,
                "Coming Soon",
                "Hosted library files are not yet available for download.\n\n"
                "You can build the files yourself using gaia_bulk_builder.py,\n"
                "then use the 'Browse…' button to point SASpro to your local files.",
            )
            return

        worker = _BandDownloadWorker(url, dest, parent=self)
        self._workers[filename] = worker

        row.set_downloading(True)

        worker.progress.connect(
            lambda done, total, msg, r=row: r.update_progress(done, total, msg))
        worker.finished.connect(
            lambda ok, msg, fn=filename: self._on_download_finished(fn, ok, msg))
        worker.cancelled.connect(
            lambda fn=filename: self._on_download_cancelled(fn))
        worker.start()

    def _on_download_finished(self, filename: str, ok: bool, msg: str):
        row = self._band_rows.get(filename)
        if row:
            row.set_downloading(False)

        worker = self._workers.pop(filename, None)
        if worker:
            worker.wait()

        if ok:
            refresh_library()
            self._library = get_library()
            self._refresh_bands()
            self.library_changed.emit()
            QMessageBox.information(self, "Install Complete", msg)
        else:
            QMessageBox.critical(self, "Download Failed", msg)

    def _on_download_cancelled(self, filename: str):
        row = self._band_rows.get(filename)
        if row:
            row.set_downloading(False)
        self._workers.pop(filename, None)

    def _on_remove(self, filename: str):
        path = get_library_dir() / filename
        reply = QMessageBox.question(
            self, "Remove Band",
            f"Delete {filename} from disk?\n\n"
            f"This will free up space but SASpro will fall back to\n"
            f"live Gaia archive downloads for sources in this band.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Close the connection first
        conn = self._library._connections.pop(filename, None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass

        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not delete file:\n{e}")
            return

        refresh_library()
        self._library = get_library()
        self._refresh_bands()
        self.library_changed.emit()

    def _on_browse(self, filename: str):
        """Let user point to a locally-built SQLite file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Locate {filename}",
            str(Path.home()),
            "SQLite files (*.sqlite);;All files (*)",
        )
        if not path:
            return

        dest = get_library_dir() / filename
        import shutil
        reply = QMessageBox.question(
            self, "Copy File",
            f"Copy selected file to the library folder as:\n{dest}\n\n"
            f"(Original file will not be moved or deleted)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            shutil.copy2(path, dest)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not copy file:\n{e}")
            return

        refresh_library()
        self._library = get_library()
        self._refresh_bands()
        self.library_changed.emit()
        QMessageBox.information(self, "Done", f"Band installed from local file.")

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

    def _lookup_spectrum(self):
        from PyQt6.QtWidgets import QLineEdit
        raw = self._sid_edit.text().strip()
        if not raw:
            return
        try:
            sid = int(raw)
        except ValueError:
            self._viewer_status.setText("Invalid source_id — must be an integer.")
            return

        spec = self._library.get_spectrum(sid)
        if spec is None:
            # Try the live cache db as well
            from setiastro.saspro.gaia_downloader import GaiaSpectraDB
            try:
                from PyQt6.QtCore import QStandardPaths
                base = QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppDataLocation)
                db_path = os.path.join(base, "gaia", "gaia_xp_cache.sqlite")
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
                f"source_id {sid} not found in local library or cache.\n"
                f"Star may not have an XP spectrum, or the relevant band is not installed."
            )
            self._spectrum_viewer.clear()
            self._source_info.clear()
            return

        # Get source info
        info = self._library.get_source_info(sid)
        if info:
            self._source_info.setPlainText(
                f"source_id: {sid}\n"
                f"RA: {info['ra']:.6f}°    Dec: {info['dec']:.6f}°    "
                f"G mag: {info['gmag']:.3f}"
            )
            title = f"Gaia source {sid}  (G={info['gmag']:.2f})"
        else:
            self._source_info.setPlainText(f"source_id: {sid}")
            title = f"Gaia source {sid}"

        self._spectrum_viewer.show_spectrum(spec, title=title)
        self._viewer_status.setText(
            f"Spectrum found — "
            f"{'local library' if self._library.has_spectrum(sid) else 'live cache'}"
        )

    def closeEvent(self, event):
        # Cancel any running downloads
        for worker in list(self._workers.values()):
            worker.cancel()
            worker.wait()
        self._workers.clear()
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def open_gaia_database_dialog(parent=None) -> GaiaDatabaseDialog:
    dlg = GaiaDatabaseDialog(parent=parent)
    dlg.show()
    return dlg