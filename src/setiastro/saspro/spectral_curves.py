# src/setiastro/saspro/spectral_curves.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
#
# Filter / Sensor Curve Manager
# =============================
# One place to add, label, edit, share and delete the spectral response curves
# that SPCC (sfcc.py), SSSC (sssc.py) and the SASP Viewer read.
#
# Ways to get a curve in:
#   * paste two (or more) columns straight from Excel / a web page / a PDF table
#   * import a CSV / TXT / TSV / DAT file (any delimiter, decimal commas, % values)
#   * digitize a chart image (file, clipboard or drag-and-drop): calibrate the
#     axes with two ticks per axis, then either trace a curve by its color or
#     click key points by hand; several curves can be taken from one chart
#   * generate a bandpass from published specs (CWL / FWHM / peak) for
#     narrowband, dual-band and broadband filters
#   * import a curve pack (.fits) someone shared
#
# Storage contract — UNCHANGED, so every existing reader keeps working:
#   * built-ins : data/SASP_data.fits                (read-only, shipped)
#   * user      : <AppData>/usercustomcurves.fits    (this module writes it)
#   * one BinTableHDU per curve: WAVELENGTH (E, Angstrom) + THROUGHPUT (E, REL),
#     EXTNAME = curve name, CTYPE = FILTER | SENSOR, ORIGIN = provenance.
# New, optional label cards (readers that don't know them ignore them):
#   CURVCAT  category code    CHANNEL  R/G/B/L/HA/OIII/...
#   MAKER    manufacturer     MODEL    product / model
#   REFURL   datasheet link   NOTES    free text
#   CMDATE   date saved       CMVER    curve-manager schema version
# Built-in curves carry none of these; their labels are inferred from the name
# for display and grouping only.

from __future__ import annotations

import os
import re
import sys
import shutil
import time
import unicodedata
from dataclasses import dataclass
from datetime import date

import numpy as np
from astropy.io import fits

from PyQt6.QtCore import (
    Qt, QObject, QPointF, QRect, QRectF, QSettings, QStandardPaths,
    QStringListModel, QTimer, pyqtSignal,
)
from PyQt6.QtGui import (
    QAction, QBrush, QColor, QIcon, QImage, QKeySequence, QPainter,
    QPainterPath, QPen, QPixmap, QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QCompleter, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGraphicsItem, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)


def _pg():
    """pyqtgraph, imported on first use (SPCC/SSSC import this module just to
    hook the library-changed signal; they shouldn't pay for pyqtgraph)."""
    import pyqtgraph as pg
    return pg


# ─────────────────────────────────────────────────────────────────────────────
# Label vocabulary
# ─────────────────────────────────────────────────────────────────────────────

CM_VERSION = 1
KINDS = ("FILTER", "SENSOR")

CATEGORY_CHOICES: dict[str, list[tuple[str, str]]] = {
    "FILTER": [
        ("RGB",   "RGB channel filter"),
        ("LUM",   "Luminance / UV-IR cut"),
        ("LPS",   "Light-pollution / broadband"),
        ("NB",    "Narrowband (single line)"),
        ("MULTI", "Dual / multi-band"),
        ("OSC",   "OSC color channel (sensor × CFA)"),
        ("PHOT",  "Photometric band"),
        ("OTHER", "Other"),
    ],
    "SENSOR": [
        ("MONOQE", "Sensor QE"),
        ("REF",    "Ideal / reference"),
        ("OTHER",  "Other"),
    ],
}

CHANNEL_CHOICES: list[tuple[str, str]] = [
    ("",      "—"),
    ("R",     "R"),
    ("G",     "G"),
    ("B",     "B"),
    ("L",     "L / clear"),
    ("HA",    "Hα"),
    ("HB",    "Hβ"),
    ("OIII",  "O III"),
    ("SII",   "S II"),
    ("NII",   "N II"),
    ("MULTI", "Multi-line"),
    ("IR",    "IR"),
    ("UV",    "UV"),
]

# (label, code, center nm) for the bandpass generator's quick-add buttons
EMISSION_LINES: list[tuple[str, str, float]] = [
    ("Hα",    "HA",   656.28),
    ("Hβ",    "HB",   486.13),
    ("O III", "OIII", 500.68),
    ("S II",  "SII",  672.0),
    ("N II",  "NII",  658.35),
]

_K_CAT, _K_CHAN, _K_MAKER, _K_MODEL = "CURVCAT", "CHANNEL", "MAKER", "MODEL"
_K_URL, _K_NOTES, _K_DATE, _K_VER = "REFURL", "NOTES", "CMDATE", "CMVER"

_CHANNEL_COLORS = {
    "R": (235, 80, 80), "G": (80, 200, 100), "B": (90, 140, 255),
    "L": (225, 225, 225), "HA": (255, 70, 110), "HB": (90, 200, 255),
    "OIII": (60, 220, 210), "SII": (200, 50, 50), "NII": (255, 130, 60),
    "IR": (170, 90, 90), "UV": (170, 110, 255),
}
_PALETTE = [
    (255, 170, 60), (120, 200, 255), (200, 130, 255), (120, 255, 180),
    (255, 120, 170), (210, 210, 90), (150, 160, 255), (255, 210, 150),
    (90, 220, 120), (255, 150, 110), (160, 230, 230), (230, 160, 230),
]
_WORKING_COLOR = (255, 215, 0)


def category_label(kind: str, code: str) -> str:
    for c, lbl in CATEGORY_CHOICES.get(kind, []):
        if c == code:
            return lbl
    return code or ""


def channel_label(code: str) -> str:
    for c, lbl in CHANNEL_CHOICES:
        if c == code:
            return lbl if c else ""
    return code or ""


# ─────────────────────────────────────────────────────────────────────────────
# Library records + label inference for built-ins
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CurveInfo:
    name: str
    kind: str                 # FILTER | SENSOR
    builtin: bool
    path: str
    category: str = ""        # code from CATEGORY_CHOICES
    channel: str = ""         # code from CHANNEL_CHOICES
    maker: str = ""
    model: str = ""
    origin: str = ""
    url: str = ""
    notes: str = ""
    date: str = ""
    n_points: int = 0
    labelled: bool = False    # True: cards read from the file; False: inferred

    def meta(self) -> dict:
        return {
            "category": self.category, "channel": self.channel,
            "maker": self.maker, "model": self.model, "url": self.url,
            "notes": self.notes, "origin": self.origin,
        }

    def search_blob(self) -> str:
        parts = [
            self.name, self.kind, self.maker, self.model, self.channel,
            channel_label(self.channel), category_label(self.kind, self.category),
            self.notes, "built-in builtin" if self.builtin else "mine my custom",
        ]
        return " ".join(p for p in parts if p).lower()


_OSC_MAKERS = {"CANON", "NIKON", "PENTAX"}


def infer_labels(name: str, kind: str) -> dict:
    """Best-effort labels for a curve that has no label cards (all built-ins).
    Display/grouping only — never written back."""
    up = str(name or "").upper()
    toks = [t for t in re.split(r"[_\s]+", up) if t]
    maker = toks[0] if toks else ""
    base = up.split("_/_")[0]
    m = re.search(r"(?:^|_)([RGB])(?:-[A-Z0-9]+)?$", base)
    channel = m.group(1) if m else ""
    if kind == "SENSOR":
        category = "REF" if ("100%" in up or "IDEAL" in up) else "MONOQE"
    elif maker in ("JOHNSON", "SDSS"):
        category = "PHOT"
    elif "COLOR_SENSOR" in up or "SONY_CMOS" in up or maker in _OSC_MAKERS:
        category = "OSC"
    elif "LIGHT_POLLUTION" in up or "_LPS" in up:
        category = "LPS"
    elif "UV-IR" in up or "UV/IR" in up or "IR_CUT" in up or "IR-CUT" in up:
        category = "LUM"
    elif channel:
        category = "RGB"
    else:
        category = "OTHER"
    return {"maker": maker, "channel": channel, "category": category}


# ─────────────────────────────────────────────────────────────────────────────
# Paths, signals
# ─────────────────────────────────────────────────────────────────────────────

def user_curves_path() -> str:
    """Same file SFCC/SSSC's _ensure_user_custom_fits() create."""
    app_data = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation)
    os.makedirs(app_data, exist_ok=True)
    path = os.path.join(app_data, "usercustomcurves.fits")
    if not os.path.exists(path):
        fits.HDUList([fits.PrimaryHDU()]).writeto(path)
    return path


def builtin_curves_path() -> str:
    from setiastro.saspro.resources import sasp_data_path
    return sasp_data_path


class _CurveLibrarySignals(QObject):
    changed = pyqtSignal()


_SIGNALS: _CurveLibrarySignals | None = None


def curve_library_signals() -> _CurveLibrarySignals:
    """App-wide hub: `changed` fires after any write to the user curve file,
    so open SPCC / SSSC dialogs can refresh their filter/sensor combos."""
    global _SIGNALS
    if _SIGNALS is None:
        _SIGNALS = _CurveLibrarySignals()
    return _SIGNALS


def notify_library_changed() -> None:
    try:
        curve_library_signals().changed.emit()
    except Exception as e:
        print(f"[CurveManager] change notification failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FITS I/O
# ─────────────────────────────────────────────────────────────────────────────

_ASCII_MAP = {
    "α": "a", "β": "b", "γ": "g", "λ": "lambda", "µ": "u", "μ": "u",
    "Å": "A", "°": "deg", "–": "-", "—": "-", "‐": "-", "’": "'",
    "‘": "'", "“": '"', "”": '"', "×": "x", "·": ".", "…": "...",
}


def _ascii(s, limit: int = 400) -> str:
    """FITS header values must be printable ASCII."""
    s = "".join(_ASCII_MAP.get(ch, ch) for ch in str(s or ""))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = "".join(ch for ch in s if 32 <= ord(ch) < 127)
    return s.strip()[:limit]


def sanitize_curve_name(name: str) -> str:
    """Library convention: UPPER_CASE_WITH_UNDERSCORES (see SASP_data.fits)."""
    s = _ascii(name, 200).upper()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Z0-9_\-./%+()]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:64]


def suggest_curve_name(maker: str, model: str, channel: str) -> str:
    parts = [p.strip() for p in (maker, model, channel) if p and p.strip()]
    return sanitize_curve_name("_".join(parts))


def _to_angstrom(wl: np.ndarray) -> np.ndarray:
    """Mirror of sfcc._ensure_angstrom: nm-looking wavelengths become Å."""
    if wl.size == 0:
        return wl
    med = float(np.median(wl))
    return wl * 10.0 if 250.0 <= med <= 2000.0 else wl


def _info_from_header(h, path: str, builtin: bool) -> CurveInfo | None:
    kind = str(h.get("CTYPE", "")).strip().upper()
    if kind not in KINDS:
        return None
    name = str(h.get("EXTNAME", "")).strip()
    if not name:
        return None
    info = CurveInfo(
        name=name, kind=kind, builtin=builtin, path=path,
        origin=str(h.get("ORIGIN", "") or ""),
        n_points=int(h.get("NAXIS2", 0) or 0),
    )
    if _K_VER in h or _K_CAT in h or _K_MAKER in h:
        info.category = str(h.get(_K_CAT, "") or "")
        info.channel = str(h.get(_K_CHAN, "") or "")
        info.maker = str(h.get(_K_MAKER, "") or "")
        info.model = str(h.get(_K_MODEL, "") or "")
        info.url = str(h.get(_K_URL, "") or "")
        info.notes = str(h.get(_K_NOTES, "") or "")
        info.date = str(h.get(_K_DATE, "") or "")
        info.labelled = True
    else:
        inf = infer_labels(name, kind)
        info.category, info.channel, info.maker = inf["category"], inf["channel"], inf["maker"]
    return info


_BUILTIN_LIST_CACHE: dict[tuple[str, float], list[CurveInfo]] = {}


def _scan_curve_file(path: str, builtin: bool) -> list[CurveInfo]:
    out: list[CurveInfo] = []
    with fits.open(path, mode="readonly", memmap=False) as hd:
        for hdu in hd:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            info = _info_from_header(hdu.header, path, builtin)
            if info is not None:
                out.append(info)
    return out


def list_curves(builtin_path: str | None, user_path: str | None) -> list[CurveInfo]:
    """FILTER and SENSOR curves from the user file first, then the built-ins
    (same precedence SFCC's load_curve uses). SEDs are not listed. The
    built-in scan (350 HDUs) is cached until that file changes."""
    out: list[CurveInfo] = []
    for path, builtin in ((user_path, False), (builtin_path, True)):
        if not path or not os.path.exists(path):
            continue
        try:
            if builtin:
                key = (os.path.abspath(path), os.path.getmtime(path))
                if key not in _BUILTIN_LIST_CACHE:
                    _BUILTIN_LIST_CACHE.clear()
                    _BUILTIN_LIST_CACHE[key] = _scan_curve_file(path, True)
                out.extend(_BUILTIN_LIST_CACHE[key])
            else:
                out.extend(_scan_curve_file(path, False))
        except Exception as e:
            print(f"[CurveManager] could not read {path}: {e}")
    return out


def read_curve(path: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(wavelength_nm, response) as float64."""
    with fits.open(path, mode="readonly", memmap=False) as hd:
        d = hd[name].data
        wl = _to_angstrom(np.asarray(d["WAVELENGTH"], dtype=np.float64)) / 10.0
        v = np.asarray(d["THROUGHPUT"], dtype=np.float64)
    return wl, v


def make_curve_hdu(name: str, kind: str, wl_nm, values, meta: dict | None = None) -> fits.BinTableHDU:
    meta = meta or {}
    wl_ang = (np.asarray(wl_nm, dtype=np.float64) * 10.0).astype(np.float32)
    tp = np.asarray(values, dtype=np.float64).astype(np.float32)
    cols = [
        fits.Column(name="WAVELENGTH", format="E", unit="Angstrom", array=wl_ang),
        fits.Column(name="THROUGHPUT", format="E", unit="REL", array=tp),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    h = hdu.header
    h["EXTNAME"] = name
    h["CTYPE"] = kind
    h["ORIGIN"] = _ascii(meta.get("origin") or "SASpro Curve Manager") or "SASpro Curve Manager"
    for key, mkey in ((_K_CAT, "category"), (_K_CHAN, "channel"),
                      (_K_MAKER, "maker"), (_K_MODEL, "model"),
                      (_K_URL, "url"), (_K_NOTES, "notes")):
        val = _ascii(meta.get(mkey, ""))
        if val:
            h[key] = val
    h[_K_DATE] = date.today().isoformat()
    h[_K_VER] = CM_VERSION
    return hdu


def _load_user_hdus(user_path: str):
    with fits.open(user_path, mode="readonly", memmap=False) as hd:
        primary = hd[0].copy()
        rest = [h.copy() for h in hd[1:]]
    return primary, rest


def _write_user_hdus(user_path: str, primary, tables) -> None:
    """Write to a temp file, keep a .bak of the previous file, then swap in.
    A crash mid-write can't leave the user's curve file half-written."""
    tmp = user_path + ".tmp"
    fits.HDUList([primary] + list(tables)).writeto(tmp, overwrite=True, output_verify="silentfix")
    try:
        if os.path.exists(user_path):
            shutil.copy2(user_path, user_path + ".bak")
    except Exception as e:
        print(f"[CurveManager] backup failed (continuing): {e}")
    last = None
    for _ in range(10):
        try:
            os.replace(tmp, user_path)
            return
        except PermissionError as e:   # Windows: something holds the file open
            last = e
            time.sleep(0.15)
    try:
        os.remove(tmp)
    except Exception:
        pass
    raise OSError(f"Could not update {user_path} — is it open in another program?\n{last}")


def save_curves(user_path: str, items: list[dict], *, replace: dict | None = None) -> None:
    """items: [{name, kind, wl_nm, values, meta}]. Existing curves with the same
    name (case-insensitive) are replaced. replace maps new name -> old name for
    renames (the old one is removed too). One file rewrite for the batch."""
    primary, tables = _load_user_hdus(user_path)
    drop = set()
    for it in items:
        drop.add(it["name"].upper())
        old = (replace or {}).get(it["name"])
        if old:
            drop.add(old.upper())
    tables = [t for t in tables if str(t.header.get("EXTNAME", "")).upper() not in drop]
    for it in items:
        tables.append(make_curve_hdu(it["name"], it["kind"], it["wl_nm"], it["values"], it.get("meta")))
    _write_user_hdus(user_path, primary, tables)


def save_curve(user_path: str, name: str, kind: str, wl_nm, values, meta: dict | None = None,
               *, replace: str | None = None) -> None:
    save_curves(user_path,
                [{"name": name, "kind": kind, "wl_nm": wl_nm, "values": values, "meta": meta}],
                replace={name: replace} if replace else None)


def delete_curves(user_path: str, names) -> int:
    kill = {str(n).upper() for n in names}
    primary, tables = _load_user_hdus(user_path)
    kept = [t for t in tables if str(t.header.get("EXTNAME", "")).upper() not in kill]
    removed = len(tables) - len(kept)
    if removed:
        _write_user_hdus(user_path, primary, kept)
    return removed


def write_curve_csv(path: str, name: str, kind: str, wl_nm, values, meta: dict | None = None) -> None:
    meta = meta or {}
    lines = ["# SASpro spectral curve", f"# name: {name}", f"# type: {kind}"]
    for k in ("category", "channel", "maker", "model", "url", "notes", "origin"):
        if meta.get(k):
            lines.append(f"# {k}: {meta[k]}")
    lines.append("wavelength_nm,response")
    lines += [f"{float(w):.4f},{float(v):.6f}" for w, v in zip(wl_nm, values)]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_curve_pack(path: str, items: list[dict]) -> None:
    """A shareable FITS with the same per-curve HDU layout as the library."""
    prim = fits.PrimaryHDU()
    prim.header["SASPPACK"] = (1, "SASpro filter/sensor curve pack")
    prim.header[_K_VER] = CM_VERSION
    prim.header["DATE"] = date.today().isoformat()
    hdus = [prim] + [make_curve_hdu(it["name"], it["kind"], it["wl_nm"], it["values"], it.get("meta"))
                     for it in items]
    fits.HDUList(hdus).writeto(path, overwrite=True, output_verify="silentfix")


def read_curve_pack(path: str) -> list[dict]:
    """Every FILTER/SENSOR table in any FITS laid out like the library
    (a curve pack, another user's usercustomcurves.fits, SASP_data.fits)."""
    out = []
    with fits.open(path, mode="readonly", memmap=False) as hd:
        for hdu in hd:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            info = _info_from_header(hdu.header, path, builtin=False)
            if info is None:
                continue
            names = [n.upper() for n in hdu.columns.names]
            if "WAVELENGTH" not in names or "THROUGHPUT" not in names:
                continue
            d = hdu.data
            wl = _to_angstrom(np.asarray(d["WAVELENGTH"], dtype=np.float64)) / 10.0
            v = np.asarray(d["THROUGHPUT"], dtype=np.float64)
            out.append({"name": sanitize_curve_name(info.name) or info.name, "kind": info.kind,
                        "wl_nm": wl, "values": v, "meta": info.meta()})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Text parsing (paste / CSV / TXT)
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*"
    r"(%|nm|um|µm|μm|a|å|angstroms?)?$", re.IGNORECASE)


_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?\s*%?$")


def _parse_number(tok: str, decimal_comma: bool = False):
    """-> (value or None, had_percent_sign)"""
    t = str(tok).strip().strip('"').strip("'").strip()
    if decimal_comma:
        t = t.replace(",", ".")
    elif _THOUSANDS_RE.match(t):
        t = t.replace(",", "")
    m = _TOKEN_RE.match(t)
    if not m:
        return None, False
    try:
        return float(m.group(1)), (m.group(2) == "%")
    except ValueError:
        return None, False


@dataclass
class ParsedTable:
    names: list[str]
    data: np.ndarray            # (N, M) float, NaN where a cell didn't parse
    meta: dict
    percent_cols: set


def parse_curve_table(text: str) -> ParsedTable:
    """Numeric columns out of pasted text or a CSV/TXT file. Handles tab, ';',
    ',', '|' or whitespace delimiters, decimal commas, '%' and unit suffixes,
    header rows, and '# key: value' comment metadata (our own CSV export)."""
    meta: dict = {}
    raw: list[str] = []
    for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip().lstrip("﻿")
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            m = re.match(r"^(?:#|//)\s*([A-Za-z][\w ]{0,30}?)\s*:\s*(.+)$", s)
            if m:
                meta[m.group(1).strip().lower()] = m.group(2).strip()
            continue
        raw.append(s)
    if not raw:
        raise ValueError("No data found.")

    def split(line, delim):
        if delim is None:
            return re.split(r"\s+", line.strip())
        return [t.strip() for t in line.split(delim)]

    best = None
    for delim in ("\t", ";", ",", "|", None):
        if delim is not None and not any(delim in l for l in raw):
            continue
        rows = [split(l, delim) for l in raw]
        if delim == ",":
            dec_comma = False
        else:
            toks = [t for r in rows for t in r]
            dec_comma = any("," in t for t in toks) and not any("." in t for t in toks)
        parsed = [[_parse_number(t, dec_comma) for t in r] for r in rows]
        numeric = [i for i, p in enumerate(parsed) if sum(v is not None for v, _ in p) >= 2]
        if not numeric:
            continue
        score = len(numeric)
        if best is None or score > best[0]:
            best = (score, delim, rows, parsed, numeric)
    if best is None:
        raise ValueError("Could not find at least two numeric columns.")
    _, delim, rows, parsed, numeric = best

    widths = [len(parsed[i]) for i in numeric]
    M = max(set(widths), key=widths.count)
    data = np.full((len(numeric), M), np.nan)
    pct = np.zeros(M, dtype=bool)
    for k, i in enumerate(numeric):
        for j, (v, is_pct) in enumerate(parsed[i][:M]):
            if v is not None:
                data[k, j] = v
                pct[j] |= is_pct

    header = None
    for i in range(numeric[0] - 1, -1, -1):
        if sum(v is not None for v, _ in parsed[i]) <= 1:
            header = rows[i]
            break

    keep = [j for j in range(M) if np.isfinite(data[:, j]).sum() >= 2]
    if len(keep) < 2:
        raise ValueError("Could not find at least two numeric columns.")
    names = []
    for n, j in enumerate(keep):
        nm = ""
        if header is not None and j < len(header):
            nm = header[j].strip().strip('"').strip("'")
        names.append(nm or f"Column {n + 1}")
    return ParsedTable(names=names, data=data[:, keep],
                       meta=meta, percent_cols={k for k, j in enumerate(keep) if pct[j]})


def guess_wavelength_column(names: list[str]) -> int:
    for i, n in enumerate(names):
        if re.search(r"wave|lambda|λ|\bwl\b|\bnm\b|angstrom|micron", n, re.IGNORECASE):
            return i
    return 0


def detect_wavelength_unit(wl) -> str:
    """'nm', 'A' or 'um' from the median wavelength."""
    wl = np.asarray(wl, dtype=np.float64)
    wl = wl[np.isfinite(wl)]
    if wl.size == 0:
        return "nm"
    med = float(np.median(np.abs(wl)))
    if med < 5.0:
        return "um"
    if med > 2000.0:
        return "A"
    return "nm"


def wavelength_to_nm(wl, unit: str) -> np.ndarray:
    wl = np.asarray(wl, dtype=np.float64)
    return wl / 10.0 if unit == "A" else wl * 1000.0 if unit == "um" else wl


def detect_percent(values) -> bool:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return bool(v.size and float(v.max()) > 1.5)


def auto_units(wl, values, percent_hint: bool = False):
    """-> (wl_nm, fraction, [notes])"""
    notes = []
    unit = detect_wavelength_unit(wl)
    wl = wavelength_to_nm(wl, unit)
    if unit == "A":
        notes.append("wavelengths looked like Å — converted to nm")
    elif unit == "um":
        notes.append("wavelengths looked like µm — converted to nm")
    v = np.asarray(values, dtype=np.float64)
    if percent_hint or detect_percent(v):
        v = v / 100.0
        notes.append("response looked like percent — divided by 100")
    return wl, v, notes


def guess_channel(label: str) -> str:
    s = str(label or "").strip().lower()
    toks = set(t for t in re.split(r"[^a-z0-9α-ω\[\]]+", s) if t)
    toks |= {t.strip("[]") for t in toks}
    table = [
        ("HA", {"ha", "halpha", "hα", "h-alpha", "h_alpha"}),
        ("HB", {"hb", "hbeta", "hβ"}),
        ("OIII", {"oiii", "o3"}),
        ("SII", {"sii", "s2"}),
        ("NII", {"nii", "n2"}),
        ("R", {"r", "red"}),
        ("G", {"g", "green"}),
        ("B", {"b", "blue"}),
        ("L", {"l", "lum", "luminance", "clear"}),
    ]
    compact = re.sub(r"[^a-z0-9α-ω]", "", s)
    for code, keys in table:
        if toks & keys or compact in keys:
            return code
    return ""


def guess_channel_from_rgb(rgb) -> str:
    """Hue of a digitized curve's color -> R/G/B suggestion."""
    try:
        c = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        h, s, v = c.hsvHueF(), c.hsvSaturationF(), c.valueF()
    except Exception:
        return ""
    if s < 0.25 or v < 0.2 or h < 0:
        return ""
    deg = h * 360.0
    if deg < 30 or deg >= 330:
        return "R"
    if 75 <= deg < 165:
        return "G"
    if 195 <= deg < 270:
        return "B"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Curve tools
# ─────────────────────────────────────────────────────────────────────────────

def clean_curve(wl, v):
    """Finite only, sorted by wavelength, duplicate wavelengths averaged."""
    wl = np.asarray(wl, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    n = min(wl.size, v.size)
    wl, v = wl[:n], v[:n]
    m = np.isfinite(wl) & np.isfinite(v)
    wl, v = wl[m], v[m]
    if wl.size == 0:
        return wl, v
    order = np.argsort(wl, kind="mergesort")
    wl, v = wl[order], v[order]
    uniq, inv = np.unique(wl, return_inverse=True)
    if uniq.size != wl.size:
        v = np.bincount(inv, weights=v) / np.bincount(inv)
        wl = uniq
    return wl, v


def resample_curve(wl, v, step: float, smooth_join: bool = False):
    wl, v = clean_curve(wl, v)
    if wl.size < 2 or step <= 0:
        return wl, v
    lo = np.ceil(wl[0] / step - 1e-9) * step
    hi = np.floor(wl[-1] / step + 1e-9) * step
    if hi < lo:
        return wl, v
    grid = np.arange(lo, hi + step * 0.5, step)
    if smooth_join and wl.size >= 3:
        from scipy.interpolate import PchipInterpolator
        out = PchipInterpolator(wl, v)(grid)
    else:
        out = np.interp(grid, wl, v)
    return grid, out


def smooth_curve(wl, v, window_nm: float):
    """Savitzky–Golay (order 2) — keeps peaks and edges better than a median."""
    wl, v = clean_curve(wl, v)
    if wl.size < 7:
        return wl, v
    d = np.diff(wl)
    step = float(np.median(d))
    if step <= 0:
        return wl, v
    if not np.allclose(d, step, rtol=1e-3, atol=1e-6):
        wl, v = resample_curve(wl, v, step)
    from scipy.signal import savgol_filter
    win = max(5, int(round(window_nm / step)) | 1)
    if win > v.size:
        win = v.size if v.size % 2 else v.size - 1
    if win < 5:
        return wl, v
    return wl, savgol_filter(v, win, polyorder=2, mode="interp")


def bin_curve(wl, v, step: float):
    """Median of all samples per step-wide wavelength bin, placed at the
    samples' median wavelength (not the bin center — on a steep filter edge
    that half-bin shift alone would be several percent of response)."""
    wl = np.asarray(wl, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    m = np.isfinite(wl) & np.isfinite(v)
    wl, v = wl[m], v[m]
    if wl.size == 0:
        return wl, v
    idx = np.round(wl / step).astype(np.int64)
    order = np.argsort(idx, kind="mergesort")
    idx, wl, v = idx[order], wl[order], v[order]
    _, starts = np.unique(idx, return_index=True)
    w_med = np.array([np.median(g) for g in np.split(wl, starts[1:])])
    v_med = np.array([np.median(g) for g in np.split(v, starts[1:])])
    return w_med, v_med


def validate_curve(wl, v, kind: str) -> list[tuple[str, str]]:
    """[(level, message)] with level in ok / info / warn / error.
    Only 'error' blocks saving."""
    wl = np.asarray(wl, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if wl.size < 2:
        return [("error", "Need at least 2 points.")]
    wl_c, v_c = clean_curve(wl, v)
    if wl_c.size < 2:
        return [("error", "Need at least 2 points with distinct, numeric wavelengths.")]
    msgs: list[tuple[str, str]] = []
    if wl_c.size != wl.size or np.any(np.diff(wl) <= 0):
        msgs.append(("info", "Points are not strictly increasing in wavelength — "
                             "they are sorted (and duplicates averaged) on save."))
    lo, hi = float(wl_c[0]), float(wl_c[-1])
    unit = detect_wavelength_unit(wl_c)
    if unit == "A":
        msgs.append(("error", f"Wavelengths run {lo:g}–{hi:g} — that looks like Ångström. "
                              "Use Tools → Wavelength Å → nm."))
    elif unit == "um":
        msgs.append(("error", f"Wavelengths run {lo:g}–{hi:g} — that looks like µm. "
                              "Use Tools → Wavelength µm → nm."))
    elif hi < 300 or lo > 1100:
        msgs.append(("warn", f"The curve ({lo:.0f}–{hi:.0f} nm) lies outside 300–1100 nm, "
                             "the range SPCC integrates over."))
    vmax, vmin = float(v_c.max()), float(v_c.min())
    if vmax <= 0:
        msgs.append(("error", "All response values are zero or negative."))
    elif vmax > 1.5:
        msgs.append(("warn", f"Response peaks at {vmax:.3g} — looks like percent. "
                             "Use Tools → Percent → fraction (÷100)."))
    elif vmax > 1.0001:
        msgs.append(("info", f"Response peaks at {vmax:.3f}, slightly above 1.0."))
    if vmin < 0:
        msgs.append(("warn", "Some values are negative — Tools → Clip to 0–1 fixes that."))
    if vmax > 0 and unit == "nm":
        if kind == "FILTER":
            ends = []
            if v_c[0] > 0.05 * vmax:
                ends.append("blue")
            if v_c[-1] > 0.05 * vmax:
                ends.append("red")
            if ends:
                which = "either end" if len(ends) == 2 else f"the {ends[0]} end"
                msgs.append(("info", f"The curve doesn't fall to zero at {which}; outside "
                                     f"{lo:.0f}–{hi:.0f} nm it is treated as fully blocking (0)."))
        elif lo > 400 or hi < 900:
            msgs.append(("info", f"QE only covers {lo:.0f}–{hi:.0f} nm; outside that range "
                                 "it is treated as 0."))
    if not any(lvl in ("warn", "error") for lvl, _ in msgs):
        msgs.insert(0, ("ok", f"{wl_c.size} points · {lo:.0f}–{hi:.0f} nm · peak {vmax:.3f}"))
    return msgs


def bandpass_curve(bands, floor: float = 0.0):
    """bands: [(center_nm, fwhm_nm, peak_0to1, sharpness)].
    Super-Gaussian T = peak·exp(-ln2·|2(λ-c)/FWHM|^(2n)): the FWHM is exact for
    any sharpness n (1 = Gaussian, 3–6 ≈ interference filter, 15+ ≈ box).
    Multiple bands combine by maximum."""
    bands = [(float(c), float(w), float(p), max(1.0, float(n)))
             for c, w, p, n in bands if float(w) > 0]
    if not bands:
        raise ValueError("Add at least one band with a positive FWHM.")
    lo = min(c - 1.6 * w for c, w, _, _ in bands)
    hi = max(c + 1.6 * w for c, w, _, _ in bands)
    if floor > 0:
        lo, hi = min(lo, 300.0), max(hi, 1100.0)
    lo = max(lo, 1.0)
    min_w = min(w for _, w, _, _ in bands)
    step = 1.0
    for s in (1.0, 0.5, 0.25, 0.2, 0.1, 0.05):
        step = s
        if s <= min_w / 30.0:
            break
    wl = np.arange(np.floor(lo / step) * step, hi + step * 0.5, step)
    t = np.zeros_like(wl)
    for c, w, p, n in bands:
        x = np.abs(2.0 * (wl - c) / w)
        t = np.maximum(t, p * np.exp(-np.log(2.0) * x ** (2.0 * n)))
    if floor > 0:
        t = np.maximum(t, floor)
    return wl, t


# ─────────────────────────────────────────────────────────────────────────────
# Digitizer core (pure numpy — the widget below is only the front end)
# ─────────────────────────────────────────────────────────────────────────────

class AxisCalibration:
    """Two reference points per axis, WebPlotDigitizer style: X1/X2 are two
    labelled ticks on the wavelength axis (same height), Y1/Y2 two ticks on
    the response axis. Solving in the (X1→X2, Y1→Y2) basis handles charts that
    are scanned slightly rotated or skewed, and charts whose axes don't start
    at the image edge."""

    KEYS = ("X1", "X2", "Y1", "Y2")

    def __init__(self):
        self.px: dict[str, tuple[float, float] | None] = {k: None for k in self.KEYS}
        self.val: dict[str, float] = {"X1": 400.0, "X2": 700.0, "Y1": 0.0, "Y2": 100.0}

    def complete(self) -> bool:
        return all(self.px[k] is not None for k in self.KEYS)

    def _basis(self):
        if not self.complete():
            raise ValueError("Place all four axis reference points first.")
        X1, X2, Y1, Y2 = (np.asarray(self.px[k], dtype=np.float64) for k in self.KEYS)
        if self.val["X1"] == self.val["X2"]:
            raise ValueError("The two wavelength reference values must differ.")
        if self.val["Y1"] == self.val["Y2"]:
            raise ValueError("The two response reference values must differ.")
        ux, uy = X2 - X1, Y2 - Y1
        nx, ny = float(np.hypot(*ux)), float(np.hypot(*uy))
        if nx < 2 or ny < 2:
            raise ValueError("Reference points on the same axis are too close together.")
        M = np.array([[ux[0], uy[0]], [ux[1], uy[1]]], dtype=np.float64)
        if abs(np.linalg.det(M)) < 0.05 * nx * ny:
            raise ValueError("The X and Y reference pairs run in the same direction — "
                             "X1/X2 go on the wavelength axis, Y1/Y2 on the response axis.")
        Minv = np.linalg.inv(M)
        tY1 = float((Minv @ (Y1 - X1))[1])
        tY2 = float((Minv @ (Y2 - X1))[1])
        if abs(tY2 - tY1) < 1e-9:
            raise ValueError("Y reference points are degenerate.")
        return X1, M, Minv, tY1, tY2

    def to_data(self, px, py):
        X1, _, Minv, tY1, tY2 = self._basis()
        d = np.vstack([np.asarray(px, dtype=np.float64).ravel() - X1[0],
                       np.asarray(py, dtype=np.float64).ravel() - X1[1]])
        s, t = Minv @ d
        x1, x2, y1, y2 = (self.val[k] for k in self.KEYS)
        return x1 + s * (x2 - x1), y1 + (t - tY1) / (tY2 - tY1) * (y2 - y1)

    def to_pixel(self, x, y):
        X1, M, _, tY1, tY2 = self._basis()
        x1, x2, y1, y2 = (self.val[k] for k in self.KEYS)
        s = (np.asarray(x, dtype=np.float64).ravel() - x1) / (x2 - x1)
        t = tY1 + (np.asarray(y, dtype=np.float64).ravel() - y1) / (y2 - y1) * (tY2 - tY1)
        P = X1[:, None] + M @ np.vstack([s, t])
        return P[0], P[1]


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8),
                        cv2.COLOR_RGB2LAB).astype(np.float32)


def color_mask(lab: np.ndarray, target_rgb, tol: float, bg_lab=None) -> np.ndarray:
    """Pixels that belong to a line of the target color.

    Anti-aliasing, thin strokes and JPEG chroma blur all blend a line toward
    the paper: such a pixel lies ON the paper→line ray in Lab, just short of
    the line color, however faint it is. So match on (a) distance off that
    ray, in absolute Lab units, and (b) enough ink along it — not on raw color
    distance (which drops every fringe pixel) and not on angle (a gray
    gridline is only ~25° from a desaturated blue, but ~26 units off its ray).
    `tol` (≈4–120) widens both. Without a paper color, or for a line barely
    distinguishable from the paper, fall back to plain Lab distance."""
    t = rgb_to_lab(np.uint8([[list(target_rgb)]]))[0, 0]
    tol = float(tol)
    if bg_lab is not None:
        v_t = t - np.asarray(bg_lab, dtype=np.float32)
        n_t = float(np.linalg.norm(v_t))
        if n_t >= 25.0:
            u = v_t / n_t
            v = lab - np.asarray(bg_lab, dtype=np.float32)
            proj = v @ u
            perp2 = (v * v).sum(axis=2) - proj * proj
            perp_tol = 3.0 + 0.3 * tol
            frac = max(0.08, 0.6 - 0.005 * tol)
            return ((proj >= frac * n_t) & (proj <= 1.35 * n_t + tol)
                    & (perp2 <= perp_tol * perp_tol))
    d2 = ((lab - t) ** 2).sum(axis=2)
    return d2 <= tol ** 2


def ink_strength(lab, target_rgb, bg_lab):
    """How far each pixel lies along the paper→line-color ray, as a fraction
    of the picked line color (1.0 = as inked as the line). None without a
    usable paper color."""
    if bg_lab is None:
        return None
    t = rgb_to_lab(np.uint8([[list(target_rgb)]]))[0, 0]
    v_t = t - np.asarray(bg_lab, dtype=np.float32)
    n_t = float(np.linalg.norm(v_t))
    if n_t < 25.0:
        return None
    return (((lab - np.asarray(bg_lab, dtype=np.float32)) @ (v_t / n_t)) / n_t).astype(np.float32)


def _suppress_line_bands(det, ref, strength, horizontal):
    """Remove gridlines / frame lines running along the rows of `det`.

    det: the color mask (finds the lines); ref: the mask being cleaned (it is
    what gets removed from, and what tells whether the curve touches a line).
    A band of ≤6 rows holding a long straight line goes when
      • it is paler than the curve (a lighter shade of the same ink — gray
        gridlines under a black curve) and spans most of the plot: only its
        pale pixels go, so the curve's own dark pixels survive; or
      • (rows only) it runs edge to edge — a gridline or frame as dark as the
        curve. A dark full-height COLUMN is left alone: it may be a steep
        filter edge; the tracer steps over vertical crossings itself.
    Columns where ink continues ≥2 px beyond the band — the curve crossing it
    — are kept, so the stroke stays whole."""
    import cv2
    H, W = det.shape
    out = ref.copy()
    if H < 8 or W < 30:
        return out
    joined = cv2.morphologyEx(det.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)))   # dotted lines
    L = max(12, int(0.2 * W))
    long_ = cv2.morphologyEx(joined, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))) > 0
    line_rows = long_.sum(axis=1) >= 0.2 * W
    edge = 0.03 * W + 2
    nothing = np.zeros(W, dtype=bool)
    r = 0
    while r < H:
        if not line_rows[r]:
            r += 1
            continue
        r0 = r
        while r + 1 < H and line_rows[r + 1]:
            r += 1
        r1 = r
        r += 1
        if r1 - r0 + 1 > 6:          # a thick bar, not a line
            continue
        on = long_[r0:r1 + 1].any(axis=0)
        idx = np.flatnonzero(on)
        first, last = int(idx[0]), int(idx[-1])
        cover = idx.size / W
        span = (last - first + 1) / W
        light = False
        if strength is not None:
            meds = [float(np.median(strength[rr, det[rr] & on])) for rr in range(r0, r1 + 1)
                    if (det[rr] & on).any()]
            light = bool(meds) and max(meds) < 0.7
        a, b = max(0, r0 - 1), min(H, r1 + 2)
        above = (ref[r0 - 1] & ref[r0 - 2]) if r0 >= 2 else nothing
        below = (ref[r1 + 1] & ref[r1 + 2]) if r1 + 2 < H else nothing
        touched = above | below
        touched = touched | np.r_[touched[1:], False] | np.r_[False, touched[:-1]]
        if light and span >= (0.6 if horizontal else 0.85) and cover >= 0.4:
            kill = ref[a:b] & (strength[a:b] < 0.7) & ~touched[None, :]
        elif horizontal and first <= edge and last >= W - 1 - edge and cover >= 0.5:
            kill = ref[a:b] & ~touched[None, :]
        else:
            continue
        out[a:b] &= ~kill
    return out


def suppress_grid(mask, strength=None):
    """`mask` (the plot area's bool color mask) without its gridlines."""
    h = _suppress_line_bands(mask, mask, strength, True)
    st = None if strength is None else np.ascontiguousarray(strength.T)
    v = _suppress_line_bands(np.ascontiguousarray(mask.T), np.ascontiguousarray(h.T), st, False)
    return np.ascontiguousarray(v.T)


def detect_plot_frame(lab: np.ndarray, bg_lab, min_frac: float = 0.45, ink_thresh: float = 45.0):
    """(x0, y0, x1, y1) of the axes box from long straight lines, or None.
    The bottom/left axes are the outermost long lines; top and right are only
    taken when they join up with the others (a real frame), so a long flat
    curve segment can't masquerade as the top edge. `ink_thresh` (Lab units)
    is low enough to see the light-gray axes many datasheets use."""
    import cv2
    d = np.sqrt(((lab - np.asarray(bg_lab, dtype=np.float32)) ** 2).sum(axis=2))
    ink = (d > ink_thresh).astype(np.uint8)
    H, W = ink.shape
    if H < 40 or W < 40:
        return None
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(min_frac * W)), 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(min_frac * H))))
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kh)
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kv)
    rows = np.flatnonzero(hor.any(axis=1))
    cols = np.flatnonzero(ver.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    yb, xl = int(rows.max()), int(cols.min())
    if not ver[max(0, yb - 3):yb + 4, xl].any():
        return None
    yt, xr = 0, W
    top = int(rows.min())
    if top < yb - 0.3 * H and hor[top, max(0, xl - 3):xl + 4].any():
        yt = top
    right = int(cols.max())
    if right > xl + 0.3 * W and ver[max(0, yb - 3):yb + 4, right].any():
        xr = right
    return xl, yt, xr, yb


def background_lab(lab: np.ndarray) -> np.ndarray:
    """Most common (quantized) color — the chart's paper."""
    q = (lab // 8).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    vals, counts = np.unique(key.ravel(), return_counts=True)
    k = vals[int(np.argmax(counts))]
    sel = key == k
    return lab[sel].mean(axis=0)


def pick_curve_color(rgb: np.ndarray, lab: np.ndarray, x: float, y: float,
                     bg_lab: np.ndarray, radius: int = 4):
    """The core color of the line under (or next to) the click.

    An anti-aliased fringe pixel is a blend of the line color and the paper,
    so it sits on the paper→line-color ray in Lab, short of the line itself.
    Walk that ray: among nearby pixels in the same color direction, take the
    one farthest from the paper. A click on bare paper takes the most
    'inked' pixel within the radius instead."""
    H, W = lab.shape[:2]
    xi = int(np.clip(np.floor(x), 0, W - 1))
    yi = int(np.clip(np.floor(y), 0, H - 1))
    y0, y1 = max(0, yi - radius), min(H, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(W, xi + radius + 1)
    vec = lab[y0:y1, x0:x1] - bg_lab
    mag = np.linalg.norm(vec, axis=2)
    v_click = lab[yi, xi] - bg_lab
    n_click = float(np.linalg.norm(v_click))
    if n_click >= 15.0:
        cos = (vec @ v_click) / (mag * n_click + 1e-9)
        cand = (cos > 0.93) & (mag >= n_click)
        if not cand.any():
            return tuple(int(c) for c in rgb[yi, xi])
        mag = np.where(cand, mag, -1.0)
    j = np.unravel_index(int(np.argmax(mag)), mag.shape)
    return tuple(int(c) for c in rgb[y0 + j[0], x0 + j[1]])


def trace_mask(mask, roi=None, min_feature=5, seed=None, strength=None):
    """Pixel path of one curve from a boolean 'ink' mask.

    1. inside the plot area, drop gridlines and frame lines that matched the
       curve color (see _suppress_line_bands), close 1-px holes, and drop
       specks smaller than min_feature px;
    2. per column, split the ink into vertical runs;
    3. start where the user clicked (`seed`, image px) — or, without one, in
       the longest stretch of columns holding a single run — and walk outward.
       A run touching the previous column's run is the same stroke; when a
       crossing line splits it, the pieces are joined only if together they
       are no taller than the stroke should be (else the piece that continues
       the curve's slope wins). A detached run must sit near where the curve
       was heading. A run far taller than the stroke is a vertical line
       crossing (the curve carries on where it was) or a steep edge (it
       carries on at the run's other end). Where the curve runs along the
       top or bottom frame — a filter's blocked region on the 0 % axis —
       and nothing else continues it, it follows the frame line.
    Returns (xs, ys) in image pixel coordinates (pixel centers)."""
    import cv2
    H, W = mask.shape
    if roi is None:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        x0, y0, x1, y1 = (int(v) for v in roi)
        x0, x1 = max(0, min(x0, W)), max(0, min(x1, W))
        y0, y1 = max(0, min(y0, H)), max(0, min(y1, H))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.empty(0), np.empty(0)
    raw = np.ascontiguousarray(mask[y0:y1, x0:x1]).astype(bool)
    st = None if strength is None else np.ascontiguousarray(strength[y0:y1, x0:x1])
    sub = suppress_grid(raw, st)
    sub = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    Hr, Wr = sub.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if (h <= 4 and w >= 0.5 * Wr) or (w <= 4 and h >= 0.5 * Hr):
            continue
        if max(w, h) >= min_feature:
            keep[i] = True
    clean = keep[labels]

    cands = []
    for x in range(Wr):
        rows = np.flatnonzero(clean[:, x])
        if rows.size == 0:
            cands.append(None)
            continue
        br = np.flatnonzero(np.diff(rows) > 1)
        starts = np.r_[rows[0], rows[br + 1]].astype(np.float64)
        ends = np.r_[rows[br], rows[-1]].astype(np.float64)
        cands.append(((starts + ends) / 2.0, starts, ends))

    # Frame lines along the top / bottom of the plot area (the strongest row
    # within a few px of the edge, if it spans most of the width).
    zone = max(4, int(0.015 * Hr) + 3)
    frames = {}
    for side, rows_ in (("top", range(0, min(zone, Hr))), ("bottom", range(max(0, Hr - zone), Hr))):
        cov = [(raw[rr].sum(), rr) for rr in rows_]
        if cov:
            c_, rr = max(cov)
            if c_ >= 0.5 * Wr:
                frames[side] = rr

    def frame_run(x, near_y):
        for side, rr in frames.items():
            if abs(near_y - rr) <= zone + 3:
                lo, hi = max(0, rr - zone), min(Hr, rr + zone + 1)
                if raw[lo:hi, x].any():
                    return float(rr), float(rr) - 1.0, float(rr) + 1.0
        return None

    tall = max(15.0, 0.25 * Hr)      # a run this tall is a vertical line or a steep edge

    # ── anchor ──
    start = None
    if seed is not None:
        sx, sy = float(seed[0]) - x0, float(seed[1]) - y0
        best = None
        for dx in range(0, 7):
            for xx in {int(np.floor(sx)) - dx, int(np.floor(sx)) + dx}:
                if 0 <= xx < Wr and cands[xx] is not None:
                    _c, s_, e_ = cands[xx]
                    d = np.maximum(0.0, np.maximum(s_ - sy, sy - e_))
                    # a vertical gridline under the click is not the curve
                    d = np.where(e_ - s_ + 1.0 > tall, np.inf, d)
                    k = int(np.argmin(d))
                    if d[k] <= 8 and (best is None or d[k] + dx < best[0]):
                        best = (d[k] + dx, xx, float(s_[k]), float(e_[k]))
            if best is not None and best[0] <= dx:
                break
        if best is not None:
            start = (best[1], (best[2] + best[3]) / 2.0, best[2], best[3], False)
        else:
            xi = int(np.clip(np.floor(sx), 0, Wr - 1))
            fr = frame_run(xi, sy)
            if fr is not None:
                start = (xi,) + fr + (True,)
    if start is None:
        have = [i for i, c in enumerate(cands) if c is not None]
        if not have:
            return np.empty(0), np.empty(0)
        single = np.array([c is not None and c[0].size == 1 for c in cands])
        best_len, best_start, cur_start = 0, -1, None
        for i, s in enumerate(np.r_[single, False]):
            if s and cur_start is None:
                cur_start = i
            elif not s and cur_start is not None:
                if i - cur_start > best_len:
                    best_len, best_start = i - cur_start, cur_start
                cur_start = None
        if best_len > 0:
            ax_ = best_start + best_len // 2
            aj = 0
        else:
            allc = np.concatenate([cands[i][0] for i in have])
            ax_ = have[len(have) // 2]
            aj = int(np.argmin(np.abs(cands[ax_][0] - float(np.median(allc)))))
        c_, s_, e_ = cands[ax_]
        start = (ax_, float(c_[aj]), float(s_[aj]), float(e_[aj]), False)

    base_jump = max(4.0, 0.04 * Hr)
    short_gap = max(8, int(0.02 * Wr))
    ys = np.full(Wr, np.nan)
    anchor = start[0]
    ys[anchor] = start[1]
    for step in (1, -1):
        _, pc, ps, pe, on_frame = start
        last_x = anchor
        hist = [(float(anchor), pc)]
        heights = [pe - ps + 1.0]

        def accept(xx, s_, e_, frame=False):
            nonlocal pc, ps, pe, last_x, on_frame
            on_frame = frame
            ys[xx] = (s_ + e_) / 2.0
            pc, ps, pe, last_x = ys[xx], s_, e_, xx
            hist.append((float(xx), pc))
            if len(hist) > 6:
                hist.pop(0)
            heights.append(e_ - s_ + 1.0)
            if len(heights) > 9:
                heights.pop(0)

        x = anchor + step
        while 0 <= x < Wr:
            c = cands[x]
            gap = abs(x - last_x)
            hx, hy = hist[0]
            slope = (pc - hy) / (last_x - hx) if last_x != hx else 0.0
            pred = float(np.clip(pc + slope * (x - last_x), 0.0, Hr - 1.0))
            expect = float(np.median(heights)) + abs(slope) + 3.0
            pick = None
            if c is not None:
                centers, starts, ends = c
                if gap <= 2:
                    # leaving the frame line: the curve's stroke starts just
                    # past where the frame was cleaned away
                    tol_c = zone + 2.0 if on_frame else 2.0
                    conn = np.flatnonzero((starts <= pe + tol_c) & (ends >= ps - tol_c))
                    if conn.size:
                        s_, e_ = float(starts[conn].min()), float(ends[conn].max())
                        if conn.size == 1 or e_ - s_ + 1.0 <= expect:
                            pick = (s_, e_)
                        else:
                            # the piece that continues the curve's slope; a
                            # sliver thinner than the stroke has ever been is
                            # a leftover of a crossing line, not the curve
                            hts = ends[conn] - starts[conn] + 1.0
                            cost = np.abs(centers[conn] - pred) + 2.0 * np.maximum(0.0, min(heights) - hts)
                            k = conn[int(np.argmin(cost))]
                            pick = (float(starts[k]), float(ends[k]))
                if pick is None:
                    d = np.minimum(np.abs(centers - pred), np.abs(centers - pc))
                    k = int(np.argmin(d))
                    if d[k] <= base_jump + (0.5 * gap if gap <= short_gap else 0.0):
                        pick = (float(starts[k]), float(ends[k]))
            if pick is None:
                fr = frame_run(x, pc)
                if fr is not None:
                    accept(x, fr[1], fr[2], frame=True)
                x += step
                continue
            s_, e_ = pick
            if e_ - s_ + 1.0 > tall and e_ - s_ + 1.0 > 3.0 * expect:
                # follow the chain of tall runs (a steep edge spans a few columns)
                chain = [(x, s_, e_)]
                while len(chain) < 12:
                    xn = chain[-1][0] + step
                    if not (0 <= xn < Wr) or cands[xn] is None:
                        break
                    _c, sC, eC = cands[xn]
                    t = np.flatnonzero((sC <= chain[-1][2] + 2.0) & (eC >= chain[-1][1] - 2.0)
                                       & (eC - sC + 1.0 > tall))
                    if not t.size:
                        break
                    kk = t[int(np.argmax(eC[t] - sC[t]))]
                    chain.append((xn, float(sC[kk]), float(eC[kk])))
                nx = chain[-1][0] + step
                if not (0 <= nx < Wr):
                    break                       # a frame line at the side of the plot
                # does the curve carry on where it was, just past the line?
                crossing = False
                for xx in range(nx, nx + 3 * step, step):
                    if 0 <= xx < Wr and cands[xx] is not None:
                        _c, sN, eN = cands[xx]
                        slack = 2.0 + abs(slope) * abs(xx - last_x)
                        if ((sN <= pe + slack) & (eN >= ps - slack)
                                & (eN - sN + 1.0 <= tall)).any():
                            crossing = True
                            break
                if crossing:
                    x = nx
                    continue
                # steep edge: the tall runs are the stroke itself
                first_s, first_e = chain[0][1], chain[0][2]
                going_down = abs(pc - first_s) < abs(pc - first_e)
                for xx, cs, ce in chain:
                    ys[xx] = (cs + ce) / 2.0
                ls, le = chain[-1][1], chain[-1][2]
                far = le if going_down else ls
                pc, ps, pe, last_x = far, far - 1.0, far + 1.0, chain[-1][0]
                on_frame = False
                hist[:] = [(float(last_x), pc)]
                heights[:] = [3.0]
                x = nx
                continue
            accept(x, s_, e_)
            x += step
    ok = np.isfinite(ys)
    xs = np.arange(Wr, dtype=np.float64)[ok] + x0 + 0.5
    return xs, ys[ok] + y0 + 0.5


def combine_digitized(traced, manual, step: float, smooth_join: bool):
    """traced/manual: (wl_nm, fraction) arrays or None. Traced samples are
    binned to `step`; hand-placed points win over traced bins within one step;
    the result is resampled onto a uniform grid (gaps get interpolated)."""
    parts_wl, parts_v = [], []
    if traced is not None and len(traced[0]):
        wl_t, v_t = bin_curve(traced[0], traced[1], step)
        if manual is not None and len(manual[0]):
            keep = np.ones(wl_t.size, dtype=bool)
            for w in np.asarray(manual[0], dtype=np.float64):
                keep &= np.abs(wl_t - w) > step
            wl_t, v_t = wl_t[keep], v_t[keep]
        parts_wl.append(wl_t)
        parts_v.append(v_t)
    if manual is not None and len(manual[0]):
        parts_wl.append(np.asarray(manual[0], dtype=np.float64))
        parts_v.append(np.asarray(manual[1], dtype=np.float64))
    if not parts_wl:
        return np.empty(0), np.empty(0)
    wl, v = clean_curve(np.concatenate(parts_wl), np.concatenate(parts_v))
    if wl.size >= 2:
        wl, v = resample_curve(wl, v, step, smooth_join=smooth_join)
    return wl, np.clip(v, 0.0, None)


def _nice_ticks(lo: float, hi: float, n: int = 8) -> np.ndarray:
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.empty(0)
    raw = span / max(1, n)
    mag = 10 ** np.floor(np.log10(raw))
    step = mag * min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * mag - raw))
    start = np.ceil(lo / step) * step
    return np.arange(start, hi + step * 0.001, step)


def qimage_to_rgb(img: QImage) -> tuple[np.ndarray, QImage]:
    """-> (H×W×3 uint8 RGB, flattened RGB32 QImage). Transparent charts are
    composited onto white, which is what they look like on a datasheet."""
    flat = QImage(img.size(), QImage.Format.Format_RGB32)
    flat.fill(QColor(255, 255, 255))
    p = QPainter(flat)
    p.drawImage(0, 0, img)
    p.end()
    rgb888 = flat.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = rgb888.width(), rgb888.height(), rgb888.bytesPerLine()
    ptr = rgb888.constBits()
    ptr.setsize(h * bpl)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl)[:, :w * 3]
    return arr.reshape(h, w, 3).copy(), flat


# ─────────────────────────────────────────────────────────────────────────────
# Small shared widgets
# ─────────────────────────────────────────────────────────────────────────────

def _make_plot(height: int | None = None):
    pg = _pg()
    pw = pg.PlotWidget()
    pi = pw.getPlotItem()
    pi.getAxis("bottom").enableAutoSIPrefix(False)
    pi.getAxis("left").enableAutoSIPrefix(False)
    pw.setLabel("bottom", "Wavelength (nm)")
    pw.setLabel("left", "Response")
    pw.showGrid(x=True, y=True, alpha=0.2)
    pw.setMenuEnabled(False)
    if height:
        pw.setMinimumHeight(height)
    return pw


def _plot_curve(pw, wl, v, color, width=1.6, name=None, symbols=False):
    pg = _pg()
    kw = dict(pen=pg.mkPen(color=color, width=width), name=name, antialias=True)
    if symbols:
        kw.update(symbol="o", symbolSize=5, symbolBrush=color, symbolPen=None)
    return pw.plot(np.asarray(wl), np.asarray(v), **kw)


def _fit_y(pw, *arrays):
    ymin, ymax = 0.0, 1.0
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.size and np.isfinite(a).any():
            ymax = max(ymax, float(np.nanmax(a)))
            ymin = min(ymin, float(np.nanmin(a)))
    pw.setYRange(ymin - 0.02 * (ymax - ymin) if ymin < 0 else 0.0, ymax * 1.05, padding=0)


def _fmt(v: float, decimals: int) -> str:
    s = f"{float(v):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


class _CurveTable(QTableWidget):
    """Two-column point table: Ctrl+V pastes (handled by the manager), Delete
    removes the selected rows, Ctrl+C copies rows as tab-separated text."""
    pasteRequested = pyqtSignal()
    deleteRequested = pyqtSignal()

    def keyPressEvent(self, e):
        editing = self.state() == QAbstractItemView.State.EditingState
        if not editing and e.matches(QKeySequence.StandardKey.Paste):
            self.pasteRequested.emit()
            return
        if not editing and e.matches(QKeySequence.StandardKey.Copy):
            rows = sorted({i.row() for i in self.selectedIndexes()})
            lines = []
            for r in rows:
                cells = [self.item(r, c).text() if self.item(r, c) else "" for c in range(self.columnCount())]
                lines.append("\t".join(cells))
            QApplication.clipboard().setText("\n".join(lines))
            return
        if not editing and e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selectedIndexes():
                self.deleteRequested.emit()
                return
        super().keyPressEvent(e)


# ─────────────────────────────────────────────────────────────────────────────
# Column picker (multi-column paste / file)
# ─────────────────────────────────────────────────────────────────────────────

class _ColumnPickerDialog(QDialog):
    """Map pasted/imported columns: one wavelength column, one or more
    response columns (each becomes its own curve)."""

    def __init__(self, parsed: ParsedTable, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Choose Columns"))
        self.parsed = parsed
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        n, m = parsed.data.shape
        lay.addWidget(QLabel(self.tr(
            "Found {0} rows × {1} numeric columns. Pick the wavelength column and the "
            "response column(s) — each checked column becomes its own curve.").format(n, m)))

        top = QHBoxLayout()
        form = QFormLayout()
        self.wl_combo = QComboBox()
        self.wl_combo.addItems(parsed.names)
        self.wl_combo.setCurrentIndex(guess_wavelength_column(parsed.names))
        form.addRow(self.tr("Wavelength column:"), self.wl_combo)
        self.unit_combo = QComboBox()
        for lbl, code in ((self.tr("Auto-detect"), "auto"), ("nm", "nm"), ("Å", "A"), ("µm", "um")):
            self.unit_combo.addItem(lbl, code)
        form.addRow(self.tr("Wavelength unit:"), self.unit_combo)
        self.scale_combo = QComboBox()
        for lbl, code in ((self.tr("Auto-detect"), "auto"), (self.tr("Fraction (0–1)"), "frac"),
                          (self.tr("Percent (0–100)"), "pct")):
            self.scale_combo.addItem(lbl, code)
        form.addRow(self.tr("Response scale:"), self.scale_combo)
        self.val_list = QListWidget()
        form.addRow(self.tr("Response columns:"), self.val_list)
        top.addLayout(form, 1)
        self.plot = _make_plot(240)
        self.plot.addLegend(offset=(-10, 10))
        top.addWidget(self.plot, 2)
        lay.addLayout(top, 1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.wl_combo.currentIndexChanged.connect(self._rebuild_list)
        self.unit_combo.currentIndexChanged.connect(self._preview)
        self.scale_combo.currentIndexChanged.connect(self._preview)
        self.val_list.itemChanged.connect(lambda *_: self._preview())
        self._rebuild_list()

    def _rebuild_list(self):
        wl_i = self.wl_combo.currentIndex()
        self.val_list.blockSignals(True)
        self.val_list.clear()
        for j, nm in enumerate(self.parsed.names):
            if j == wl_i:
                continue
            it = QListWidgetItem(nm)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, j)
            self.val_list.addItem(it)
        self.val_list.blockSignals(False)
        self._preview()

    def results(self) -> list[dict]:
        wl_i = self.wl_combo.currentIndex()
        wl_raw = self.parsed.data[:, wl_i]
        unit = self.unit_combo.currentData()
        if unit == "auto":
            unit = detect_wavelength_unit(wl_raw)
        wl = wavelength_to_nm(wl_raw, unit)
        out = []
        for k in range(self.val_list.count()):
            it = self.val_list.item(k)
            if it.checkState() != Qt.CheckState.Checked:
                continue
            j = int(it.data(Qt.ItemDataRole.UserRole))
            v = self.parsed.data[:, j].astype(np.float64)
            scale = self.scale_combo.currentData()
            if scale == "pct" or (scale == "auto" and (j in self.parsed.percent_cols or detect_percent(v))):
                v = v / 100.0
            w2, v2 = clean_curve(wl, v)
            if w2.size >= 2:
                out.append({"label": it.text(), "wl": w2, "v": v2, "channel": guess_channel(it.text())})
        return out

    def _preview(self):
        self.plot.clear()
        for k, r in enumerate(self.results()):
            col = _CHANNEL_COLORS.get(r["channel"], _PALETTE[k % len(_PALETTE)])
            _plot_curve(self.plot, r["wl"], r["v"], col, name=r["label"])
        self.plot.enableAutoRange()


# ─────────────────────────────────────────────────────────────────────────────
# Bandpass generator
# ─────────────────────────────────────────────────────────────────────────────

class BandpassDialog(QDialog):
    """Filter curve from published specs: CWL, FWHM, peak, edge sharpness."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Generate Bandpass Curve"))
        self.resize(820, 560)
        self.result_wl = None
        self.result_val = None
        self.origin = ""
        self.suggested_category = "NB"
        self.suggested_channel = ""
        self._codes: list[str] = []

        lay = QVBoxLayout(self)
        intro = QLabel(self.tr(
            "Build a filter curve from its published specs: center wavelength (CWL), "
            "bandwidth (FWHM) and peak transmission. Add one row per passband for "
            "dual- and tri-band filters. Sharpness shapes the edges: 1 = Gaussian, "
            "3–6 ≈ typical interference filter, 15+ ≈ rectangular (broadband / UV-IR cut)."))
        intro.setWordWrap(True)
        lay.addWidget(intro)

        body = QHBoxLayout()
        left = QVBoxLayout()
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([self.tr("CWL (nm)"), self.tr("FWHM (nm)"),
                                              self.tr("Peak (%)"), self.tr("Sharpness")])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(lambda *_: self._preview())
        left.addWidget(self.table, 1)

        defaults = QHBoxLayout()
        defaults.addWidget(QLabel(self.tr("New lines:")))
        self.def_fwhm = QDoubleSpinBox()
        self.def_fwhm.setRange(0.5, 200.0)
        self.def_fwhm.setValue(7.0)
        self.def_fwhm.setSuffix(" nm")
        self.def_fwhm.setDecimals(1)
        defaults.addWidget(self.def_fwhm)
        self.def_peak = QDoubleSpinBox()
        self.def_peak.setRange(1.0, 100.0)
        self.def_peak.setValue(90.0)
        self.def_peak.setSuffix(" %")
        self.def_peak.setDecimals(0)
        defaults.addWidget(self.def_peak)
        defaults.addStretch(1)
        left.addLayout(defaults)

        lines = QHBoxLayout()
        for lbl, code, cwl in EMISSION_LINES:
            b = QPushButton(f"+ {lbl}")
            b.clicked.connect(lambda _=False, c=cwl, k=code: self._add_row(
                c, self.def_fwhm.value(), self.def_peak.value(), 4.0, k))
            lines.addWidget(b)
        left.addLayout(lines)
        lines2 = QHBoxLayout()
        b_bb = QPushButton(self.tr("+ Broadband 400–700"))
        b_bb.clicked.connect(lambda: self._add_row(550.0, 300.0, 95.0, 20.0, "L"))
        lines2.addWidget(b_bb)
        b_empty = QPushButton(self.tr("+ Row"))
        b_empty.clicked.connect(lambda: self._add_row(550.0, self.def_fwhm.value(), self.def_peak.value(), 4.0, ""))
        lines2.addWidget(b_empty)
        b_rm = QPushButton(self.tr("− Remove Row"))
        b_rm.clicked.connect(self._remove_rows)
        lines2.addWidget(b_rm)
        lines2.addStretch(1)
        left.addLayout(lines2)

        floor_row = QHBoxLayout()
        floor_row.addWidget(QLabel(self.tr("Out-of-band floor:")))
        self.floor_spin = QDoubleSpinBox()
        self.floor_spin.setRange(0.0, 20.0)
        self.floor_spin.setDecimals(2)
        self.floor_spin.setSuffix(" %")
        self.floor_spin.setToolTip(self.tr("Leakage outside the passbands (0 for a well-blocked filter)."))
        self.floor_spin.valueChanged.connect(lambda *_: self._preview())
        floor_row.addWidget(self.floor_spin)
        floor_row.addStretch(1)
        left.addLayout(floor_row)
        body.addLayout(left, 1)

        self.plot = _make_plot(260)
        body.addWidget(self.plot, 1)
        lay.addLayout(body, 1)

        self.err_lbl = QLabel("")
        self.err_lbl.setStyleSheet("color:#e0a040;")
        lay.addWidget(self.err_lbl)
        bb = QDialogButtonBox()
        self.ok_btn = bb.addButton(self.tr("Use Curve"), QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self._add_row(656.28, 7.0, 90.0, 4.0, "HA")

    def _add_row(self, cwl, fwhm, peak, sharp, code=""):
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate((cwl, fwhm, peak, sharp)):
            self.table.setItem(r, c, QTableWidgetItem(_fmt(v, 2)))
        self._codes.append(code)
        self.table.blockSignals(False)
        self._preview()

    def _remove_rows(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows and self.table.rowCount():
            rows = {self.table.rowCount() - 1}
        for r in sorted(rows, reverse=True):
            self.table.removeRow(r)
            if r < len(self._codes):
                del self._codes[r]
        self._preview()

    def _bands(self):
        bands = []
        for r in range(self.table.rowCount()):
            vals = []
            for c in range(4):
                it = self.table.item(r, c)
                v, _ = _parse_number(it.text() if it else "", decimal_comma=True)
                vals.append(v)
            if None in vals:
                raise ValueError(self.tr("Row {0} has a value that isn't a number.").format(r + 1))
            bands.append((vals[0], vals[1], vals[2] / 100.0, vals[3]))
        return bands

    def _preview(self):
        self.plot.clear()
        try:
            wl, t = bandpass_curve(self._bands(), self.floor_spin.value() / 100.0)
        except Exception as e:
            self.err_lbl.setText(str(e))
            self.ok_btn.setEnabled(False)
            return
        self.err_lbl.setText("")
        self.ok_btn.setEnabled(True)
        _plot_curve(self.plot, wl, t, _WORKING_COLOR, width=2.0)
        self.plot.enableAutoRange()
        _fit_y(self.plot, t)

    def _accept(self):
        try:
            bands = self._bands()
            wl, t = bandpass_curve(bands, self.floor_spin.value() / 100.0)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Bandpass"), str(e))
            return
        self.result_wl, self.result_val = wl, t
        parts = [f"{_fmt(c, 2)}/{_fmt(w, 2)} nm @ {_fmt(p * 100, 1)}% n={_fmt(n, 1)}"
                 for c, w, p, n in bands]
        self.origin = "Generated bandpass: " + "; ".join(parts)
        codes = list(self._codes)
        if len(bands) > 1:
            self.suggested_category, self.suggested_channel = "MULTI", "MULTI"
        elif codes and codes[0] == "L":
            self.suggested_category, self.suggested_channel = "LUM", "L"
        else:
            self.suggested_category, self.suggested_channel = "NB", (codes[0] if codes else "")
        self.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Digitizer widgets
# ─────────────────────────────────────────────────────────────────────────────

_CAL_COLORS = {"X1": QColor(255, 90, 90), "X2": QColor(255, 160, 60),
               "Y1": QColor(80, 200, 255), "Y2": QColor(120, 140, 255)}


class _CalibMarker(QGraphicsItem):
    """Draggable crosshair for an axis reference point (constant screen size)."""

    def __init__(self, key: str, on_moved):
        super().__init__()
        self.key = key
        self.color = _CAL_COLORS[key]
        self.on_moved = on_moved
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
                      | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(60)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def boundingRect(self):
        return QRectF(-14, -26, 50, 40)

    def shape(self):
        p = QPainterPath()
        p.addEllipse(QPointF(0, 0), 9, 9)
        return p

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        dark = QPen(QColor(0, 0, 0, 160), 3.5)
        lite = QPen(self.color, 1.6)
        for pen in (dark, lite):
            p.setPen(pen)
            p.drawLine(QPointF(-12, 0), QPointF(-4, 0))
            p.drawLine(QPointF(4, 0), QPointF(12, 0))
            p.drawLine(QPointF(0, -12), QPointF(0, -4))
            p.drawLine(QPointF(0, 4), QPointF(0, 12))
            p.drawEllipse(QPointF(0, 0), 7, 7)
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.setBrush(QBrush(self.color))
        p.drawRect(QRectF(8, -24, 26, 14))
        p.setPen(QPen(QColor(0, 0, 0)))
        f = p.font()
        f.setPointSizeF(8.0)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(8, -24, 26, 14), Qt.AlignmentFlag.AlignCenter, self.key)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.on_moved:
            self.on_moved(self.key, self.pos())
        return super().itemChange(change, value)


class _PointHandle(QGraphicsItem):
    """Draggable hand-placed curve point."""

    def __init__(self, on_moved):
        super().__init__()
        self.on_moved = on_moved
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
                      | QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(55)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def boundingRect(self):
        return QRectF(-6, -6, 12, 12)

    def paint(self, p, opt, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(0, 0, 0), 1.5))
        p.setBrush(QBrush(QColor(*_WORKING_COLOR)))
        p.drawEllipse(QPointF(0, 0), 4.5, 4.5)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and self.on_moved:
            self.on_moved()
        return super().itemChange(change, value)


class _ImageView(QGraphicsView):
    sceneClicked = pyqtSignal(QPointF, object)     # scene pos, Qt.MouseButton
    sceneRectDragged = pyqtSignal(QRectF)
    sceneMouseMoved = pyqtSignal(QPointF)
    handleRemoveRequested = pyqtSignal(object)
    fileDropped = pyqtSignal(str)
    imageDropped = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setBackgroundBrush(QColor(45, 45, 45))
        self.mode = "calib"
        self._rubber_origin = None
        self._rubber_item = None
        self._panning = False
        self._pan_last = None

    # zoom / pan
    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        e.accept()

    def fit(self):
        r = self.scene().itemsBoundingRect()
        if r.width() > 0 and r.height() > 0:
            self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton or self.mode == "pan":
            self._panning = True
            self._pan_last = e.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            e.accept()
            return
        pos = self.mapToScene(e.position().toPoint())
        item = self.itemAt(e.position().toPoint())
        if self.mode in ("roi", "erase") and e.button() == Qt.MouseButton.LeftButton:
            self._rubber_origin = pos
            pen = QPen(QColor(30, 144, 255) if self.mode == "roi" else QColor(255, 60, 60), 1.5,
                       Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber_item = self.scene().addRect(QRectF(pos, pos), pen)
            self._rubber_item.setZValue(100)
            e.accept()
            return
        if self.mode == "points" and isinstance(item, _PointHandle):
            if e.button() == Qt.MouseButton.RightButton:
                self.handleRemoveRequested.emit(item)
                e.accept()
                return
            super().mousePressEvent(e)
            return
        if isinstance(item, _CalibMarker) and e.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(e)
            return
        self.sceneClicked.emit(pos, e.button())
        e.accept()

    def mouseMoveEvent(self, e):
        if self._panning and self._pan_last is not None:
            d = e.position() - self._pan_last
            self._pan_last = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            e.accept()
            return
        pos = self.mapToScene(e.position().toPoint())
        self.sceneMouseMoved.emit(pos)
        if self._rubber_item is not None:
            self._rubber_item.setRect(QRectF(self._rubber_origin, pos).normalized())
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._panning:
            self._panning = False
            self._pan_last = None
            self.viewport().unsetCursor()
            e.accept()
            return
        if self._rubber_item is not None:
            r = self._rubber_item.rect()
            self.scene().removeItem(self._rubber_item)
            self._rubber_item = None
            if r.width() > 2 and r.height() > 2:
                self.sceneRectDragged.emit(r)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # drag & drop
    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() or md.hasImage():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasImage():
            img = md.imageData()
            if isinstance(img, QImage) and not img.isNull():
                self.imageDropped.emit(img)
                e.acceptProposedAction()
                return
        for u in md.urls():
            if u.isLocalFile():
                self.fileDropped.emit(u.toLocalFile())
                e.acceptProposedAction()
                return
        e.ignore()


class CurveDigitizerDialog(QDialog):
    """Turn a chart image (datasheet screenshot, scan, web graphic) into curve data.

    1. Calibrate: place X1/X2 on two labelled wavelength ticks and Y1/Y2 on two
       response ticks, and type their values.
    2. Trace: click the curve to pick its color — every matching pixel is
       traced column by column. Draw a plot-area box to keep the legend out,
       erase stray bits with a box, or …
    3. … click points by hand (drag to move, right-click to delete). Hand
       points can also patch gaps in a traced curve.
    Several curves can be taken from one chart (Keep && Trace Another)."""

    def __init__(self, parent=None, image: QImage | None = None, image_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Digitize Curve from Image"))
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.captured: list[dict] = []
        self.source_label = ""
        self._qimg: QImage | None = None
        self._rgb = None
        self._lab = None
        self._bg_lab = None
        self._target = None
        self._seed: tuple[float, float] | None = None     # where the curve was clicked
        self._roi: QRectF | None = None
        self._erase_rects: list[QRectF] = []
        self._traced_px = None
        self._handles: list[_PointHandle] = []
        self._markers: dict[str, _CalibMarker] = {}
        self._output = (np.empty(0), np.empty(0))
        self.cal = AxisCalibration()

        self._retrace_timer = QTimer(self)
        self._retrace_timer.setSingleShot(True)
        self._retrace_timer.setInterval(120)
        self._retrace_timer.timeout.connect(self._retrace)

        self._build_ui()
        self._restore()
        if image is not None and not image.isNull():
            self._set_image(image, self.tr("clipboard image"))
        elif image_path:
            self._load_image_file(image_path)
        else:
            self._set_status(self.tr("Open, paste (Ctrl+V) or drop a chart image to begin."))

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        b_open = QPushButton(self.tr("Open Image…"))
        b_open.clicked.connect(self._open_image)
        bar.addWidget(b_open)
        b_paste = QPushButton(self.tr("Paste Image"))
        b_paste.clicked.connect(self._paste_image)
        bar.addWidget(b_paste)
        b_fit = QPushButton(self.tr("Fit"))
        b_fit.clicked.connect(lambda: self.view.fit())
        bar.addWidget(b_fit)
        hint = QLabel(self.tr("Wheel: zoom · Middle-drag: pan"))
        hint.setStyleSheet("color:#888;")
        bar.addWidget(hint)
        bar.addStretch(1)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        self._split = split
        self.view = _ImageView()
        self.view.sceneClicked.connect(self._on_scene_click)
        self.view.sceneRectDragged.connect(self._on_rect_dragged)
        self.view.sceneMouseMoved.connect(self._on_mouse_moved)
        self.view.handleRemoveRequested.connect(self._remove_handle)
        self.view.fileDropped.connect(self._load_image_file)
        self.view.imageDropped.connect(lambda img: self._set_image(img, self.tr("dropped image")))
        split.addWidget(self.view)

        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(4, 0, 4, 0)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        # ① calibration
        g1 = QGroupBox(self.tr("1 · Calibrate axes"))
        gl = QGridLayout(g1)
        self.cal_btns: dict[str, QPushButton] = {}
        self.cal_spins: dict[str, QDoubleSpinBox] = {}
        self.cal_state: dict[str, QLabel] = {}
        tips = {
            "X1": self.tr("a labelled tick on the wavelength axis"),
            "X2": self.tr("another wavelength tick, far from X1"),
            "Y1": self.tr("a labelled tick on the response axis (e.g. 0)"),
            "Y2": self.tr("another response tick (e.g. 100 or 1.0)"),
        }
        for r, key in enumerate(AxisCalibration.KEYS):
            b = QPushButton(key)
            b.setCheckable(True)
            c = _CAL_COLORS[key]
            b.setStyleSheet(f"QPushButton {{ color: rgb({c.red()},{c.green()},{c.blue()}); font-weight:bold; }}"
                            "QPushButton:checked { background:#3a5a3a; }")
            b.setToolTip(self.tr("Then click {0}.").format(tips[key]))
            b.setFixedWidth(44)
            self.mode_group.addButton(b)
            b.toggled.connect(lambda on, k=key: on and self._set_mode(f"calib:{k}"))
            sp = QDoubleSpinBox()
            sp.setRange(-1e6, 1e6)
            sp.setDecimals(3)
            sp.setValue(self.cal.val[key])
            sp.valueChanged.connect(lambda v, k=key: self._on_cal_value(k, v))
            st = QLabel("○")
            gl.addWidget(b, r, 0)
            gl.addWidget(sp, r, 1)
            gl.addWidget(st, r, 2)
            self.cal_btns[key], self.cal_spins[key], self.cal_state[key] = b, sp, st
        self.xunit_combo = QComboBox()
        for lbl, code in (("nm", "nm"), ("Å", "A"), ("µm", "um")):
            self.xunit_combo.addItem(lbl, code)
        self.xunit_combo.currentIndexChanged.connect(lambda *_: self._recompute())
        gl.addWidget(QLabel(self.tr("Wavelength axis unit:")), 4, 0, 1, 2)
        gl.addWidget(self.xunit_combo, 4, 2)
        self.yscale_combo = QComboBox()
        self.yscale_combo.addItem(self.tr("Percent (0–100)"), "pct")
        self.yscale_combo.addItem(self.tr("Fraction (0–1)"), "frac")
        self.yscale_combo.currentIndexChanged.connect(lambda *_: self._recompute())
        gl.addWidget(QLabel(self.tr("Response axis is:")), 5, 0, 1, 2)
        gl.addWidget(self.yscale_combo, 5, 2)
        self.grid_chk = QCheckBox(self.tr("Show calibration grid (should sit on the chart's grid)"))
        self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(lambda *_: self._update_grid_overlay())
        gl.addWidget(self.grid_chk, 6, 0, 1, 3)
        sv.addWidget(g1)

        # ② trace by color
        g2 = QGroupBox(self.tr("2 · Trace a curve by its color"))
        g2l = QGridLayout(g2)
        self.pick_btn = QPushButton(self.tr("Pick Curve Color"))
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(self.tr("Click on the curve line in the image."))
        self.mode_group.addButton(self.pick_btn)
        self.pick_btn.toggled.connect(lambda on: on and self._set_mode("pick"))
        self.swatch = QLabel()
        self.swatch.setFixedSize(36, 18)
        self.swatch.setStyleSheet("border:1px solid #666; background:transparent;")
        g2l.addWidget(self.pick_btn, 0, 0)
        g2l.addWidget(self.swatch, 0, 1)
        g2l.addWidget(QLabel(self.tr("Color tolerance:")), 1, 0)
        self.tol_slider = QSlider(Qt.Orientation.Horizontal)
        self.tol_slider.setRange(4, 120)
        self.tol_slider.setValue(40)
        self.tol_slider.valueChanged.connect(lambda *_: self._retrace_timer.start())
        g2l.addWidget(self.tol_slider, 1, 1)
        g2l.addWidget(QLabel(self.tr("Ignore specks under:")), 2, 0)
        self.speck_spin = QSpinBox()
        self.speck_spin.setRange(1, 200)
        self.speck_spin.setValue(6)
        self.speck_spin.setSuffix(" px")
        self.speck_spin.setToolTip(self.tr("Ink blobs narrower than this (text, tick labels, markers) are ignored."))
        self.speck_spin.valueChanged.connect(lambda *_: self._retrace_timer.start())
        g2l.addWidget(self.speck_spin, 2, 1)
        self.roi_btn = QPushButton(self.tr("Draw Plot-Area Box"))
        self.roi_btn.setCheckable(True)
        self.roi_btn.setToolTip(self.tr("Drag a box around the plotting area to keep the legend and labels out of the trace."))
        self.mode_group.addButton(self.roi_btn)
        self.roi_btn.toggled.connect(lambda on: on and self._set_mode("roi"))
        b_roi_clear = QPushButton(self.tr("Clear Box"))
        b_roi_clear.clicked.connect(self._clear_roi)
        g2l.addWidget(self.roi_btn, 3, 0)
        g2l.addWidget(b_roi_clear, 3, 1)
        self.erase_btn = QPushButton(self.tr("Erase Traced Points"))
        self.erase_btn.setCheckable(True)
        self.erase_btn.setToolTip(self.tr("Drag a box over mis-traced bits to drop them."))
        self.mode_group.addButton(self.erase_btn)
        self.erase_btn.toggled.connect(lambda on: on and self._set_mode("erase"))
        b_unerase = QPushButton(self.tr("Undo Erasing"))
        b_unerase.clicked.connect(self._clear_erase)
        g2l.addWidget(self.erase_btn, 4, 0)
        g2l.addWidget(b_unerase, 4, 1)
        sv.addWidget(g2)

        # ③ manual points
        g3 = QGroupBox(self.tr("3 · …or click points by hand"))
        g3l = QGridLayout(g3)
        self.points_btn = QPushButton(self.tr("Add / Move Points"))
        self.points_btn.setCheckable(True)
        self.points_btn.setToolTip(self.tr("Click to add a point, drag to move it, right-click to delete it."))
        self.mode_group.addButton(self.points_btn)
        self.points_btn.toggled.connect(lambda on: on and self._set_mode("points"))
        b_clear_pts = QPushButton(self.tr("Clear Points"))
        b_clear_pts.clicked.connect(self._clear_points)
        g3l.addWidget(self.points_btn, 0, 0)
        g3l.addWidget(b_clear_pts, 0, 1)
        g3l.addWidget(QLabel(self.tr("Join points with:")), 1, 0)
        self.join_combo = QComboBox()
        self.join_combo.addItem(self.tr("Smooth curve"), True)
        self.join_combo.addItem(self.tr("Straight lines"), False)
        self.join_combo.currentIndexChanged.connect(lambda *_: self._recompute())
        g3l.addWidget(self.join_combo, 1, 1)
        sv.addWidget(g3)

        # ④ output
        g4 = QGroupBox(self.tr("Result"))
        g4l = QVBoxLayout(g4)
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel(self.tr("Sample every:")))
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.05, 20.0)
        self.step_spin.setDecimals(2)
        self.step_spin.setValue(1.0)
        self.step_spin.setSuffix(" nm")
        self.step_spin.valueChanged.connect(lambda *_: self._recompute())
        step_row.addWidget(self.step_spin)
        step_row.addStretch(1)
        g4l.addLayout(step_row)
        self.preview = _make_plot(150)
        g4l.addWidget(self.preview)
        loupe_row = QHBoxLayout()
        self.loupe = QLabel()
        self.loupe.setFixedSize(120, 120)
        self.loupe.setStyleSheet("border:1px solid #555; background:#222;")
        loupe_row.addWidget(self.loupe)
        self.coord_lbl = QLabel("")
        self.coord_lbl.setWordWrap(True)
        self.coord_lbl.setStyleSheet("color:#aaa;")
        loupe_row.addWidget(self.coord_lbl, 1)
        g4l.addLayout(loupe_row)
        self.captured_lbl = QLabel("")
        self.captured_lbl.setStyleSheet("color:#8fd7c6;")
        g4l.addWidget(self.captured_lbl)
        sv.addWidget(g4)
        sv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(520)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        split.addWidget(scroll)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([900, 380])

        hint = self.view.scene().addSimpleText(self.tr("Open, paste (Ctrl+V) or drop a chart image here"))
        hint.setBrush(QBrush(QColor(170, 170, 170)))
        f = hint.font()
        f.setPointSize(13)
        hint.setFont(f)
        root.addWidget(split, 1)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#cfe8e0; padding:4px; background:#20302c; border-radius:3px;")
        root.addWidget(self.status_lbl)

        bb = QHBoxLayout()
        bb.addStretch(1)
        self.keep_btn = QPushButton(self.tr("Keep && Trace Another"))
        self.keep_btn.setToolTip(self.tr("Store this curve and trace another one from the same chart "
                                         "(e.g. the R, G and B curves of a color sensor)."))
        self.keep_btn.clicked.connect(self._keep_and_continue)
        bb.addWidget(self.keep_btn)
        self.use_btn = QPushButton(self.tr("Use Curve"))
        self.use_btn.setDefault(True)
        self.use_btn.clicked.connect(self._finish)
        bb.addWidget(self.use_btn)
        b_cancel = QPushButton(self.tr("Cancel"))
        b_cancel.clicked.connect(self.reject)
        bb.addWidget(b_cancel)
        root.addLayout(bb)

        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self._paste_image)

        self._trace_item = QGraphicsPathItem()
        pen = QPen(QColor(255, 0, 200), 2)
        pen.setCosmetic(True)
        self._trace_item.setPen(pen)
        self._trace_item.setZValue(40)
        self._manual_item = QGraphicsPathItem()
        pen2 = QPen(QColor(*_WORKING_COLOR), 1.5, Qt.PenStyle.DashLine)
        pen2.setCosmetic(True)
        self._manual_item.setPen(pen2)
        self._manual_item.setZValue(45)
        self._grid_item = QGraphicsPathItem()
        pen3 = QPen(QColor(60, 255, 120, 150), 1)
        pen3.setCosmetic(True)
        self._grid_item.setPen(pen3)
        self._grid_item.setZValue(30)
        self._roi_item = None
        self._grid_labels: list[QGraphicsSimpleTextItem] = []
        self._pix_item: QGraphicsPixmapItem | None = None
        self._update_buttons()

    # ── state persistence ────────────────────────────────────────────────
    def _restore(self):
        s = QSettings()
        g = s.value("CurveManager/digitizer_geometry")
        if g is not None:
            try:
                self.restoreGeometry(g)
            except Exception:
                pass
        else:
            self.resize(1280, 820)
        st = s.value("CurveManager/digitizer_split")
        if st is not None:
            try:
                self._split.restoreState(st)
            except Exception:
                pass
        self.step_spin.setValue(float(s.value("CurveManager/digitizer_step", 1.0)))
        idx = self.yscale_combo.findData(s.value("CurveManager/digitizer_yscale", "pct"))
        if idx >= 0:
            self.yscale_combo.setCurrentIndex(idx)

    def done(self, r):
        s = QSettings()
        s.setValue("CurveManager/digitizer_geometry", self.saveGeometry())
        s.setValue("CurveManager/digitizer_split", self._split.saveState())
        s.setValue("CurveManager/digitizer_step", self.step_spin.value())
        s.setValue("CurveManager/digitizer_yscale", self.yscale_combo.currentData())
        super().done(r)

    # ── image loading ────────────────────────────────────────────────────
    def _open_image(self):
        start = QSettings().value("CurveManager/last_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Open Chart Image"), start,
            self.tr("Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff);;All Files (*)"))
        if path:
            QSettings().setValue("CurveManager/last_dir", os.path.dirname(path))
            self._load_image_file(path)

    def _paste_image(self):
        md = QApplication.clipboard().mimeData()
        if md is not None and md.hasImage():
            img = QApplication.clipboard().image()
            if not img.isNull():
                self._set_image(img, self.tr("clipboard image"))
                return
        if md is not None and md.hasUrls():
            for u in md.urls():
                if u.isLocalFile():
                    self._load_image_file(u.toLocalFile())
                    return
        QMessageBox.information(self, self.tr("Paste Image"),
                                self.tr("The clipboard doesn't hold an image. Take a screenshot of the chart "
                                        "(Win+Shift+S on Windows, Cmd+Ctrl+Shift+4 on macOS) and paste again."))

    def _load_image_file(self, path: str):
        img = QImage(path)
        if img.isNull():
            QMessageBox.warning(self, self.tr("Open Image"), self.tr("Could not read an image from:\n{0}").format(path))
            return
        self._set_image(img, os.path.basename(path))

    def _set_image(self, img: QImage, label: str):
        try:
            self._rgb, self._qimg = qimage_to_rgb(img)
            self._lab = rgb_to_lab(self._rgb)
            self._bg_lab = background_lab(self._lab)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Image"), self.tr("Could not use this image:\n{0}").format(e))
            return
        self.source_label = label
        sc = self.view.scene()
        for it in list(sc.items()):
            sc.removeItem(it)
        self._markers.clear()
        self._handles.clear()
        self._grid_labels.clear()
        self._roi_item = None
        self._roi = None
        self._erase_rects = []
        self._traced_px = None
        self._target = None
        self._seed = None
        self.cal = AxisCalibration()
        for k, sp in self.cal_spins.items():
            self.cal.val[k] = sp.value()
        self._pix_item = sc.addPixmap(QPixmap.fromImage(self._qimg))
        self._pix_item.setZValue(0)
        for it in (self._trace_item, self._manual_item, self._grid_item):
            sc.addItem(it)
            it.setPath(QPainterPath())
        sc.setSceneRect(QRectF(self._qimg.rect()).adjusted(-40, -40, 40, 40))
        self.swatch.setStyleSheet("border:1px solid #666; background:transparent;")
        self.view.resetTransform()
        QTimer.singleShot(0, self.view.fit)
        self._update_cal_state()
        frame = None
        try:
            frame = detect_plot_frame(self._lab, self._bg_lab)
        except Exception as e:
            print(f"[CurveManager] plot-frame detection failed: {e}")
        if frame is not None:
            # The box sits ON the axes: a filter's blocked region lies along
            # the 0 % axis, and the tracer strips the frame lines itself.
            fx0, fy0, fx1, fy1 = frame
            W = self._qimg.width()
            self._set_roi(QRectF(QPointF(fx0, fy0), QPointF(fx1 + 1 if fx1 < W else W, fy1 + 1)),
                          retrace=False)
        self.cal_btns["X1"].setChecked(True)
        self._set_mode("calib:X1")
        if frame is not None:
            self._set_status(self.status_lbl.text() + "  " + self.tr(
                "(The plot area was found automatically — dashed box. Redraw or clear it if it's wrong.)"))
        self._recompute()

    # ── modes ────────────────────────────────────────────────────────────
    def _set_mode(self, mode: str):
        self.view.mode = mode.split(":")[0]
        self._mode = mode
        msgs = {
            "calib:X1": self.tr("Click a labelled tick on the WAVELENGTH axis (e.g. 400), then type its value next to X1."),
            "calib:X2": self.tr("Click another wavelength tick far from X1 (e.g. 700) and type its value next to X2."),
            "calib:Y1": self.tr("Click a labelled tick on the RESPONSE axis (e.g. 0) and type its value next to Y1."),
            "calib:Y2": self.tr("Click another response tick (e.g. 100 % or 1.0) and type its value next to Y2."),
            "pick": self.tr("Click on the curve line you want. Raise the tolerance if parts are missing, "
                            "lower it if a neighbouring curve gets picked up. Gridlines are ignored."),
            "roi": self.tr("Drag a box around the plotting area (on the axes) to keep legends and labels out."),
            "erase": self.tr("Drag a box over traced points that went astray to remove them."),
            "points": self.tr("Click along the curve to add points — the important corners and peaks are enough. "
                              "Drag to move, right-click to delete."),
        }
        if self._qimg is not None:
            self._set_status(msgs.get(mode, ""))

    def _set_status(self, text: str):
        self.status_lbl.setText(text)

    # ── scene interaction ────────────────────────────────────────────────
    def _on_scene_click(self, pos: QPointF, button):
        if self._qimg is None or button != Qt.MouseButton.LeftButton:
            return
        mode = getattr(self, "_mode", "")
        if mode.startswith("calib:"):
            key = mode.split(":")[1]
            self._place_marker(key, pos)
            nxt = [k for k in AxisCalibration.KEYS if self.cal.px[k] is None]
            if nxt:
                self.cal_btns[nxt[0]].setChecked(True)
            else:
                if self._target is None:
                    self.pick_btn.setChecked(True)
                    self._set_status(self.tr("Axes calibrated. Check the green grid sits on the chart's grid, then "
                                             "click the curve to trace it — or use Add / Move Points."))
            self.cal_spins[key].setFocus()
            self.cal_spins[key].selectAll()
        elif mode == "pick":
            if not QRectF(self._qimg.rect()).contains(pos):
                return
            target = pick_curve_color(self._rgb, self._lab, pos.x(), pos.y(), self._bg_lab)
            t_lab = rgb_to_lab(np.uint8([[list(target)]]))[0, 0]
            if float(np.linalg.norm(t_lab - self._bg_lab)) < 12.0:
                self._set_status(self.tr("That's the chart background — click right on the curve line "
                                         "(zoom in with the wheel if it's thin)."))
                return
            self._target = target
            self._seed = (pos.x(), pos.y())
            r, g, b = self._target
            self.swatch.setStyleSheet(f"border:1px solid #666; background: rgb({r},{g},{b});")
            self._erase_rects = []
            self._retrace()
            if self._traced_px is not None and self._traced_px[0].size == 0:
                self._set_status(self.tr("Nothing traced from there. Click closer to the line, raise the color "
                                         "tolerance, or check the plot-area box contains the curve."))
            elif self._output[0].size >= 2:
                self._set_status(self.tr("Traced {0}–{1} nm (magenta). Click elsewhere on the same curve to restart "
                                         "the trace from there; erase stray bits or add hand points for gaps.").format(
                    _fmt(float(self._output[0][0]), 1), _fmt(float(self._output[0][-1]), 1)))
        elif mode == "points":
            h = _PointHandle(self._on_handle_moved)
            h.setPos(pos)
            self.view.scene().addItem(h)
            self._handles.append(h)
            self._recompute()

    def _place_marker(self, key: str, pos: QPointF):
        m = self._markers.get(key)
        if m is None:
            m = _CalibMarker(key, self._on_marker_moved)
            self.view.scene().addItem(m)
            self._markers[key] = m
        m.setPos(pos)
        self.cal.px[key] = (pos.x(), pos.y())
        self._update_cal_state()
        self._recompute()

    def _on_marker_moved(self, key: str, pos: QPointF):
        self.cal.px[key] = (pos.x(), pos.y())
        self._update_cal_state()
        self._recompute()

    def _on_cal_value(self, key: str, v: float):
        self.cal.val[key] = float(v)
        self._recompute()

    def _update_cal_state(self):
        for k in AxisCalibration.KEYS:
            self.cal_state[k].setText("●" if self.cal.px[k] is not None else "○")

    def _set_roi(self, r: QRectF, retrace: bool = True):
        self._roi = r
        if self._roi_item is None:
            pen = QPen(QColor(30, 144, 255, 230), 1.5, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._roi_item = self.view.scene().addRect(r, pen)
            self._roi_item.setZValue(35)
        else:
            self._roi_item.setRect(r)
        if retrace:
            self._retrace()

    def _on_rect_dragged(self, r: QRectF):
        if self.view.mode == "roi":
            self._set_roi(r)
            if self._target is None:
                self.pick_btn.setChecked(True)
        elif self.view.mode == "erase":
            self._erase_rects.append(r)
            self._apply_erase_and_draw()
            self._recompute()

    def _clear_roi(self):
        self._roi = None
        if self._roi_item is not None:
            self.view.scene().removeItem(self._roi_item)
            self._roi_item = None
        self._retrace()

    def _clear_erase(self):
        self._erase_rects = []
        self._retrace()

    def _remove_handle(self, h):
        if h in self._handles:
            self._handles.remove(h)
            self.view.scene().removeItem(h)
            self._recompute()

    def _on_handle_moved(self):
        self._recompute()

    def _clear_points(self):
        for h in self._handles:
            self.view.scene().removeItem(h)
        self._handles.clear()
        self._recompute()

    def _on_mouse_moved(self, pos: QPointF):
        if self._qimg is None:
            return
        x, y = int(np.floor(pos.x())), int(np.floor(pos.y()))
        r = 15
        crop = self._qimg.copy(QRect(x - r, y - r, 2 * r + 1, 2 * r + 1))
        pm = QPixmap.fromImage(crop).scaled(self.loupe.width(), self.loupe.height(),
                                            Qt.AspectRatioMode.IgnoreAspectRatio,
                                            Qt.TransformationMode.FastTransformation)
        p = QPainter(pm)
        p.setPen(QPen(QColor(255, 0, 0, 200), 1))
        cx, cy = pm.width() / 2, pm.height() / 2
        p.drawLine(QPointF(cx, 0), QPointF(cx, pm.height()))
        p.drawLine(QPointF(0, cy), QPointF(pm.width(), cy))
        p.end()
        self.loupe.setPixmap(pm)
        txt = f"px ({pos.x():.1f}, {pos.y():.1f})"
        if self.cal.complete():
            try:
                wx, wy = self.cal.to_data([pos.x()], [pos.y()])
                txt += f"\n→ x {wx[0]:.2f}, y {wy[0]:.3f}"
            except ValueError:
                pass
        self.coord_lbl.setText(txt)

    # ── tracing / output ─────────────────────────────────────────────────
    def _retrace(self):
        if self._lab is None or self._target is None:
            self._traced_px = None
            self._apply_erase_and_draw()
            self._recompute()
            return
        mask = color_mask(self._lab, self._target, self.tol_slider.value(), self._bg_lab)
        roi = None
        if self._roi is not None:
            r = self._roi
            roi = (int(np.floor(r.left())), int(np.floor(r.top())),
                   int(np.ceil(r.right())), int(np.ceil(r.bottom())))
        xs, ys = trace_mask(mask, roi=roi, min_feature=self.speck_spin.value(), seed=self._seed,
                            strength=ink_strength(self._lab, self._target, self._bg_lab))
        self._traced_px = (xs, ys)
        self._apply_erase_and_draw()
        self._recompute()

    def _traced_after_erase(self):
        if self._traced_px is None:
            return None
        xs, ys = self._traced_px
        keep = np.ones(xs.size, dtype=bool)
        for r in self._erase_rects:
            keep &= ~((xs >= r.left()) & (xs <= r.right()) & (ys >= r.top()) & (ys <= r.bottom()))
        return xs[keep], ys[keep]

    def _apply_erase_and_draw(self):
        path = QPainterPath()
        t = self._traced_after_erase()
        if t is not None and t[0].size:
            xs, ys = t
            path.moveTo(xs[0], ys[0])
            for i in range(1, xs.size):
                if xs[i] - xs[i - 1] > 3.0:
                    path.moveTo(xs[i], ys[i])
                else:
                    path.lineTo(xs[i], ys[i])
        self._trace_item.setPath(path)

    def _recompute(self):
        # manual polyline
        path = QPainterPath()
        pts = sorted((h.pos().x(), h.pos().y()) for h in self._handles)
        if pts:
            path.moveTo(*pts[0])
            for p in pts[1:]:
                path.lineTo(*p)
        self._manual_item.setPath(path)
        self._update_grid_overlay()

        self._output = (np.empty(0), np.empty(0))
        self.preview.clear()
        if not self.cal.complete():
            self._update_buttons()
            return
        try:
            xunit = self.xunit_combo.currentData()
            yscale = 0.01 if self.yscale_combo.currentData() == "pct" else 1.0
            traced = None
            t = self._traced_after_erase()
            if t is not None and t[0].size:
                wx, wy = self.cal.to_data(*t)
                traced = (wavelength_to_nm(wx, xunit), wy * yscale)
            manual = None
            if pts:
                arr = np.asarray(pts, dtype=np.float64)
                mx, my = self.cal.to_data(arr[:, 0], arr[:, 1])
                manual = (wavelength_to_nm(mx, xunit), my * yscale)
            wl, v = combine_digitized(traced, manual, self.step_spin.value(),
                                      bool(self.join_combo.currentData()))
        except ValueError as e:
            self._set_status(str(e))
            self._update_buttons()
            return
        self._output = (wl, v)
        for k, c in enumerate(self.captured):
            _plot_curve(self.preview, c["wl"], c["v"], _PALETTE[k % len(_PALETTE)], width=1.0)
        if wl.size:
            _plot_curve(self.preview, wl, v, _WORKING_COLOR, width=2.0)
            self.preview.enableAutoRange()
            _fit_y(self.preview, v)
        self._update_buttons()

    def _update_grid_overlay(self):
        for it in self._grid_labels:
            self.view.scene().removeItem(it)
        self._grid_labels.clear()
        path = QPainterPath()
        if self._qimg is not None and self.grid_chk.isChecked() and self.cal.complete():
            try:
                W, H = self._qimg.width(), self._qimg.height()
                cx, cy = self.cal.to_data([0, W, 0, W], [0, 0, H, H])
                xr = (float(cx.min()), float(cx.max()))
                yr = (float(cy.min()), float(cy.max()))
                for xv in _nice_ticks(*xr, n=8):
                    px, py = self.cal.to_pixel([xv, xv], [yr[0], yr[1]])
                    path.moveTo(px[0], py[0])
                    path.lineTo(px[1], py[1])
                    bx, by = self.cal.to_pixel([xv], [self.cal.val["Y1"]])
                    lbl = QGraphicsSimpleTextItem(_fmt(xv, 2))
                    lbl.setBrush(QBrush(QColor(60, 255, 120)))
                    lbl.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                    lbl.setPos(bx[0] + 2, by[0] + 2)
                    lbl.setZValue(31)
                    self.view.scene().addItem(lbl)
                    self._grid_labels.append(lbl)
                for yv in _nice_ticks(*yr, n=6):
                    px, py = self.cal.to_pixel([xr[0], xr[1]], [yv, yv])
                    path.moveTo(px[0], py[0])
                    path.lineTo(px[1], py[1])
            except ValueError:
                pass
        self._grid_item.setPath(path)

    def _update_buttons(self):
        n = int(self._output[0].size)
        self.use_btn.setEnabled(n >= 2 or bool(self.captured))
        self.keep_btn.setEnabled(n >= 2)
        if self.captured:
            self.captured_lbl.setText(self.tr("{0} curve(s) kept from this chart.").format(len(self.captured)))
        else:
            self.captured_lbl.setText("")

    def _current_capture(self) -> dict | None:
        wl, v = self._output
        if wl.size < 2:
            return None
        n = len(self.captured) + 1
        return {"wl": wl.copy(), "v": v.copy(),
                "label": self.tr("curve {0}").format(n),
                "channel": guess_channel_from_rgb(self._target) if self._target is not None else ""}

    def _keep_and_continue(self):
        c = self._current_capture()
        if c is None:
            return
        self.captured.append(c)
        self._target = None
        self._seed = None
        self._traced_px = None
        self._erase_rects = []
        self.swatch.setStyleSheet("border:1px solid #666; background:transparent;")
        self._clear_points()
        self._apply_erase_and_draw()
        self.pick_btn.setChecked(True)
        self._recompute()
        self._set_status(self.tr("Kept. Click the next curve's line to trace it — the axis calibration stays."))

    def _finish(self):
        c = self._current_capture()
        if c is not None:
            self.captured.append(c)
        if not self.captured:
            return
        self.accept()

    def origin(self) -> str:
        return f"Digitized: {self.source_label}" if self.source_label else "Digitized"


# ─────────────────────────────────────────────────────────────────────────────
# The manager window
# ─────────────────────────────────────────────────────────────────────────────

class _RangeDialog(QDialog):
    def __init__(self, lo, hi, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Trim to Range"))
        f = QFormLayout(self)
        self.lo = QDoubleSpinBox()
        self.hi = QDoubleSpinBox()
        for sp, v in ((self.lo, lo), (self.hi, hi)):
            sp.setRange(0.0, 100000.0)
            sp.setDecimals(2)
            sp.setSuffix(" nm")
            sp.setValue(v)
        f.addRow(self.tr("Keep from:"), self.lo)
        f.addRow(self.tr("to:"), self.hi)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)


class SpectralCurveManager(QDialog):
    """Library browser + curve editor for FILTER and SENSOR response curves."""

    def __init__(self, parent=None, sasp_data_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Filter / Sensor Curve Manager"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        if sys.platform == "darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.setMinimumSize(960, 640)

        self.builtin_path = sasp_data_path or builtin_curves_path()
        self.user_path = user_curves_path()
        self._infos: list[CurveInfo] = []
        self._cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self._wl = np.empty(0)
        self._val = np.empty(0)
        self._undo: list[tuple[np.ndarray, np.ndarray]] = []
        self._editing: str | None = None
        self._origin = ""
        self._dirty = False
        self._pending: list[dict] = []
        self._filling = False

        self._build_ui()
        self.reload_library()
        self._restore_state()
        self._validate()
        self._update_buttons()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 6)
        self._split_main = QSplitter(Qt.Orientation.Horizontal)

        # Library ─────────────────────────────────────────────────────────
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        head = QLabel(self.tr("<b>Curve library</b>"))
        lv.addWidget(head)
        srow = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(self.tr("Search name, maker, channel…"))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(lambda *_: self._apply_filter())
        srow.addWidget(self.search_edit, 1)
        self.show_combo = QComboBox()
        for lbl, code in ((self.tr("All"), "all"), (self.tr("Filters"), "FILTER"),
                          (self.tr("Sensors"), "SENSOR"), (self.tr("My curves"), "mine")):
            self.show_combo.addItem(lbl, code)
        self.show_combo.currentIndexChanged.connect(lambda *_: self._apply_filter())
        srow.addWidget(self.show_combo)
        lv.addLayout(srow)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels([self.tr("Name"), self.tr("Ch"), self.tr("Category")])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        hdr = self.tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(False)
        self.tree.itemSelectionChanged.connect(self._on_selection)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        lv.addWidget(self.tree, 1)
        tip = QLabel(self.tr("Select several to compare · double-click to edit"))
        tip.setStyleSheet("color:#888; font-size:11px;")
        lv.addWidget(tip)

        lb1 = QHBoxLayout()
        self.btn_open = QPushButton(self.tr("Open in Editor"))
        self.btn_open.setToolTip(self.tr("Edit one of your curves, or start a custom copy of a built-in one."))
        self.btn_open.clicked.connect(self._open_selected_in_editor)
        lb1.addWidget(self.btn_open)
        self.btn_delete = QPushButton(self.tr("Delete"))
        self.btn_delete.setToolTip(self.tr("Delete the selected curves from My Curves (built-ins can't be deleted)."))
        self.btn_delete.clicked.connect(self._delete_selected)
        lb1.addWidget(self.btn_delete)
        lv.addLayout(lb1)
        lb2 = QHBoxLayout()
        self.btn_export = QToolButton()
        self.btn_export.setText(self.tr("Export"))
        self.btn_export.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_export.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_export.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        em = QMenu(self.btn_export)
        em.addAction(self.tr("Selected as CSV…"), self._export_csv)
        em.addAction(self.tr("Selected as curve pack (.fits)…"), self._export_pack)
        self.btn_export.setMenu(em)
        lb2.addWidget(self.btn_export)
        self.btn_import_pack = QPushButton(self.tr("Import Pack…"))
        self.btn_import_pack.setToolTip(self.tr("Add the curves from a curve pack (.fits) someone shared."))
        self.btn_import_pack.clicked.connect(self._import_pack)
        lb2.addWidget(self.btn_import_pack)
        lv.addLayout(lb2)
        self._split_main.addWidget(left)

        # Plot + editor ───────────────────────────────────────────────────
        self._split_right = QSplitter(Qt.Orientation.Vertical)
        plot_box = QWidget()
        pv = QVBoxLayout(plot_box)
        pv.setContentsMargins(0, 0, 0, 0)
        self.plot = _make_plot(220)
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setXRange(350, 1050, padding=0)
        self.plot.setYRange(0, 1.05, padding=0)
        self.plot.scene().sigMouseMoved.connect(self._on_plot_hover)
        pv.addWidget(self.plot, 1)
        self.hover_lbl = QLabel(" ")
        self.hover_lbl.setStyleSheet("color:#999; font-size:11px;")
        pv.addWidget(self.hover_lbl)
        self._split_right.addWidget(plot_box)
        self._split_right.addWidget(self._build_editor())
        self._split_main.addWidget(self._split_right)
        self._split_main.setStretchFactor(0, 1)
        self._split_main.setStretchFactor(1, 3)
        root.addWidget(self._split_main, 1)

        bottom = QHBoxLayout()
        try:
            from setiastro.saspro.help_support import make_help_button
            bottom.addWidget(make_help_button("curve_manager", self))
        except Exception:
            pass
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color:#3dbf9f;")   # reads on dark and light themes
        bottom.addWidget(self.status_lbl, 1)
        b_close = QPushButton(self.tr("Close"))
        b_close.clicked.connect(self.reject)
        bottom.addWidget(b_close)
        root.addLayout(bottom)

        QShortcut(QKeySequence.StandardKey.Save, self, activated=self._save_working)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._undo_last)

    def _build_editor(self) -> QWidget:
        box = QGroupBox(self.tr("Curve editor"))
        v = QVBoxLayout(box)

        self.queue_frame = QFrame()
        self.queue_frame.setStyleSheet("QFrame { background:#20302c; border-left:3px solid #3dbf9f; }"
                                       "QLabel { color:#d8f0e8; background:transparent; border:none; }")
        qf = QHBoxLayout(self.queue_frame)
        qf.setContentsMargins(8, 4, 8, 4)
        self.queue_lbl = QLabel("")
        self.queue_lbl.setWordWrap(True)
        qf.addWidget(self.queue_lbl, 1)
        self.btn_skip = QPushButton(self.tr("Skip This One"))
        self.btn_skip.clicked.connect(self._advance_queue)
        qf.addWidget(self.btn_skip)
        self.queue_frame.setVisible(False)
        v.addWidget(self.queue_frame)

        src = QHBoxLayout()
        self.btn_new = QPushButton(self.tr("New"))
        self.btn_new.setToolTip(self.tr("Start a new, empty curve."))
        self.btn_new.clicked.connect(self._new_curve)
        src.addWidget(self.btn_new)
        self.btn_paste = QPushButton(self.tr("Paste Data"))
        self.btn_paste.setToolTip(self.tr(
            "Paste columns copied from Excel, a web page, a PDF table or a text file:\n"
            "wavelength + one or more response columns. Ctrl+V in the table works too.\n"
            "Units (nm / Å / µm, fraction / percent) are detected automatically."))
        self.btn_paste.clicked.connect(self._paste_data)
        src.addWidget(self.btn_paste)
        self.btn_import = QPushButton(self.tr("Import File…"))
        self.btn_import.setToolTip(self.tr("CSV, TXT, TSV or DAT with a wavelength column and response column(s). "
                                           "A .fits curve pack is imported into the library."))
        self.btn_import.clicked.connect(self._import_file)
        src.addWidget(self.btn_import)
        self.btn_digitize = QPushButton(self.tr("Digitize Image…"))
        self.btn_digitize.setToolTip(self.tr("Trace a curve from a chart image — a datasheet screenshot, scan or "
                                             "web graphic. Paste, open or drop the image."))
        self.btn_digitize.clicked.connect(lambda: self._open_digitizer())
        src.addWidget(self.btn_digitize)
        self.btn_generate = QPushButton(self.tr("Generate Bandpass…"))
        self.btn_generate.setToolTip(self.tr("Build a narrowband, multi-band or broadband filter curve from its "
                                             "published CWL / FWHM / peak transmission."))
        self.btn_generate.clicked.connect(self._open_generator)
        src.addWidget(self.btn_generate)
        src.addStretch(1)
        self.btn_tools = QToolButton()
        self.btn_tools.setText(self.tr("Tools"))
        self.btn_tools.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tm = QMenu(self.btn_tools)
        tm.addAction(self.tr("Sort && merge duplicate wavelengths"), self._tool_sort)
        tm.addAction(self.tr("Resample to uniform step…"), self._tool_resample)
        tm.addAction(self.tr("Smooth…"), self._tool_smooth)
        tm.addAction(self.tr("Trim to range…"), self._tool_trim)
        tm.addSeparator()
        tm.addAction(self.tr("Clip to 0–1"), self._tool_clip)
        tm.addAction(self.tr("Normalize peak to 1.0"), self._tool_normalize)
        tm.addAction(self.tr("Percent → fraction (÷100)"), lambda: self._tool_scale_values(0.01))
        tm.addSeparator()
        tm.addAction(self.tr("Wavelength Å → nm (÷10)"), lambda: self._tool_scale_wl(0.1))
        tm.addAction(self.tr("Wavelength µm → nm (×1000)"), lambda: self._tool_scale_wl(1000.0))
        self.btn_tools.setMenu(tm)
        src.addWidget(self.btn_tools)
        self.btn_undo = QPushButton(self.tr("Undo"))
        self.btn_undo.clicked.connect(self._undo_last)
        src.addWidget(self.btn_undo)
        v.addLayout(src)

        body = QSplitter(Qt.Orientation.Horizontal)
        self._split_editor = body

        tbl_box = QWidget()
        tv = QVBoxLayout(tbl_box)
        tv.setContentsMargins(0, 0, 0, 0)
        self.table = _CurveTable(0, 2)
        self.table.setHorizontalHeaderLabels([self.tr("λ (nm)"), self.tr("Response (0–1)")])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.pasteRequested.connect(self._paste_data)
        self.table.deleteRequested.connect(self._delete_rows)
        tv.addWidget(self.table, 1)
        trow = QHBoxLayout()
        b_add = QPushButton(self.tr("+ Row"))
        b_add.clicked.connect(self._add_row)
        trow.addWidget(b_add)
        b_del = QPushButton(self.tr("− Rows"))
        b_del.clicked.connect(self._delete_rows)
        trow.addWidget(b_del)
        self.points_lbl = QLabel("")
        self.points_lbl.setStyleSheet("color:#888;")
        trow.addWidget(self.points_lbl, 1)
        tv.addLayout(trow)
        body.addWidget(tbl_box)

        form_box = QWidget()
        fv = QVBoxLayout(form_box)
        fv.setContentsMargins(6, 0, 0, 0)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.kind_combo = QComboBox()
        self.kind_combo.addItem(self.tr("Filter"), "FILTER")
        self.kind_combo.addItem(self.tr("Sensor (QE)"), "SENSOR")
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        form.addRow(self.tr("Type:"), self.kind_combo)
        self.cat_combo = QComboBox()
        self.cat_combo.currentIndexChanged.connect(lambda *_: self._mark_dirty())
        form.addRow(self.tr("Category:"), self.cat_combo)
        self.chan_combo = QComboBox()
        for code, lbl in CHANNEL_CHOICES:
            self.chan_combo.addItem(lbl, code)
        self.chan_combo.currentIndexChanged.connect(lambda *_: self._on_label_changed())
        form.addRow(self.tr("Channel:"), self.chan_combo)
        self.maker_edit = QLineEdit()
        self.maker_edit.setPlaceholderText(self.tr("e.g. Antlia, Baader, ZWO, Sony"))
        self.maker_edit.textEdited.connect(lambda *_: self._on_label_changed())
        self._maker_completer = QCompleter([])
        self._maker_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.maker_edit.setCompleter(self._maker_completer)
        form.addRow(self.tr("Maker:"), self.maker_edit)
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(self.tr("e.g. V-Pro Series, IMX585, L-eXtreme"))
        self.model_edit.textEdited.connect(lambda *_: self._on_label_changed())
        form.addRow(self.tr("Model:"), self.model_edit)
        nrow = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(self.tr("MAKER_MODEL_CHANNEL"))
        self.name_edit.setToolTip(self.tr("The name SPCC and SSSC show in their filter / sensor lists."))
        self.name_edit.textEdited.connect(self._on_name_edited)
        nrow.addWidget(self.name_edit, 1)
        self.name_auto_chk = QCheckBox(self.tr("Auto"))
        self.name_auto_chk.setToolTip(self.tr("Build the name from Maker, Model and Channel."))
        self.name_auto_chk.setChecked(True)
        self.name_auto_chk.toggled.connect(lambda on: on and self._apply_auto_name())
        nrow.addWidget(self.name_auto_chk)
        form.addRow(self.tr("Name:"), nrow)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(self.tr("Datasheet / source link (optional)"))
        self.url_edit.textEdited.connect(lambda *_: self._mark_dirty())
        form.addRow(self.tr("Source link:"), self.url_edit)
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText(self.tr("Notes (optional)"))
        self.notes_edit.textEdited.connect(lambda *_: self._mark_dirty())
        form.addRow(self.tr("Notes:"), self.notes_edit)
        fv.addLayout(form)
        self.origin_lbl = QLabel("")
        self.origin_lbl.setWordWrap(True)
        self.origin_lbl.setStyleSheet("color:#888; font-size:11px;")
        fv.addWidget(self.origin_lbl)
        self.valid_lbl = QLabel("")
        self.valid_lbl.setWordWrap(True)
        self.valid_lbl.setTextFormat(Qt.TextFormat.RichText)
        fv.addWidget(self.valid_lbl)
        fv.addStretch(1)
        srow = QHBoxLayout()
        srow.addStretch(1)
        self.btn_save = QPushButton(self.tr("Save to My Curves"))
        self.btn_save.setToolTip(self.tr("Ctrl+S. Saved curves appear in the SPCC and SSSC filter / sensor lists."))
        self.btn_save.setStyleSheet(
            "QPushButton { background:#1a4f2a; color:#e8f8ee; border:1px solid #2a8040; border-radius:4px;"
            " padding:5px 14px; font-weight:bold; }"
            "QPushButton:hover { background:#1f6535; }"
            "QPushButton:disabled { background:#2d2d2d; color:#777; border-color:#444; }")
        self.btn_save.clicked.connect(self._save_working)
        srow.addWidget(self.btn_save)
        fv.addLayout(srow)
        body.addWidget(form_box)
        body.setStretchFactor(0, 2)
        body.setStretchFactor(1, 3)
        v.addWidget(body, 1)
        self._on_kind_changed()
        return box

    # ── state persistence ────────────────────────────────────────────────
    def _restore_state(self):
        s = QSettings()
        g = s.value("CurveManager/geometry")
        if g is not None:
            try:
                self.restoreGeometry(g)
            except Exception:
                pass
        else:
            self.resize(1360, 880)
        for key, sp in (("split_main", self._split_main), ("split_right", self._split_right),
                        ("split_editor", self._split_editor)):
            st = s.value(f"CurveManager/{key}")
            if st is not None:
                try:
                    sp.restoreState(st)
                except Exception:
                    pass
        if s.value("CurveManager/split_main") is None:
            self._split_main.setSizes([380, 980])
            self._split_right.setSizes([380, 500])

    def _save_state(self):
        s = QSettings()
        s.setValue("CurveManager/geometry", self.saveGeometry())
        s.setValue("CurveManager/split_main", self._split_main.saveState())
        s.setValue("CurveManager/split_right", self._split_right.saveState())
        s.setValue("CurveManager/split_editor", self._split_editor.saveState())

    def done(self, r):
        if self._dirty and self._wl.size >= 2:
            ans = QMessageBox.question(
                self, self.tr("Unsaved Curve"),
                self.tr("The curve in the editor hasn't been saved. Close anyway?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        self._save_state()
        super().done(r)

    # ── library ──────────────────────────────────────────────────────────
    def reload_library(self, select: list[str] | None = None):
        prev = select if select is not None else [i.name for i in self._selected_infos()]
        self._infos = list_curves(self.builtin_path, self.user_path)
        self._cache = {k: v for k, v in self._cache.items() if k[0] != self.user_path}
        makers = sorted({i.maker for i in self._infos if i.maker and (i.labelled or i.builtin)},
                        key=str.lower)
        self._maker_completer.setModel(QStringListModel(makers, self._maker_completer))
        self._populate_tree()
        if prev:
            self.select_curves(prev)

    def _populate_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        bold = self.tree.font()
        bold.setBold(True)

        def group(text):
            it = QTreeWidgetItem([text, "", ""])
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            it.setFont(0, bold)
            return it

        def leaf(parent, idx, info):
            it = QTreeWidgetItem(parent, [info.name, channel_label(info.channel),
                                          category_label(info.kind, info.category)])
            it.setData(0, Qt.ItemDataRole.UserRole, idx)
            tip = [f"<b>{info.name}</b>", f"{'Filter' if info.kind == 'FILTER' else 'Sensor QE'}"
                   f" · {'built-in' if info.builtin else 'my curve'} · {info.n_points} points"]
            if info.maker or info.model:
                tip.append(" ".join(x for x in (info.maker, info.model) if x))
            if info.origin:
                tip.append(f"Source: {info.origin}")
            if info.url:
                tip.append(info.url)
            if info.notes:
                tip.append(info.notes)
            it.setToolTip(0, "<br>".join(tip))
            return it

        self._g_mine = group(self.tr("My curves"))
        self._g_filters = group(self.tr("Built-in filters"))
        self._g_sensors = group(self.tr("Built-in sensors"))
        self._maker_groups: list[QTreeWidgetItem] = []
        by_maker: dict[str, QTreeWidgetItem] = {}
        order = sorted(range(len(self._infos)), key=lambda i: self._infos[i].name.lower())
        for idx in order:
            info = self._infos[idx]
            if not info.builtin:
                leaf(self._g_mine, idx, info)
            elif info.kind == "SENSOR":
                leaf(self._g_sensors, idx, info)
            else:
                mk = info.maker or "?"
                g = by_maker.get(mk)
                if g is None:
                    g = QTreeWidgetItem(self._g_filters, [mk, "", ""])
                    g.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    by_maker[mk] = g
                    self._maker_groups.append(g)
                leaf(g, idx, info)
        self.tree.addTopLevelItems([self._g_mine, self._g_filters, self._g_sensors])
        self._g_mine.setExpanded(True)
        self._g_filters.setExpanded(True)
        self.tree.blockSignals(False)
        self._apply_filter()

    def _iter_leaves(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            it = stack.pop()
            if it.data(0, Qt.ItemDataRole.UserRole) is not None:
                yield it
            for k in range(it.childCount()):
                stack.append(it.child(k))

    def _apply_filter(self):
        toks = self.search_edit.text().lower().split()
        mode = self.show_combo.currentData()
        for it in self._iter_leaves():
            info = self._infos[int(it.data(0, Qt.ItemDataRole.UserRole))]
            ok = (mode == "all" or (mode == "mine" and not info.builtin) or info.kind == mode)
            if ok and toks:
                blob = info.search_blob()
                ok = all(t in blob for t in toks)
            it.setHidden(not ok)

        def visible_children(g):
            n = 0
            for k in range(g.childCount()):
                c = g.child(k)
                if c.data(0, Qt.ItemDataRole.UserRole) is not None:
                    n += 0 if c.isHidden() else 1
                else:
                    sub = visible_children(c)
                    c.setHidden(sub == 0)
                    if toks:
                        c.setExpanded(sub > 0)
                    n += sub
            return n

        for g, label in ((self._g_mine, self.tr("My curves")),
                         (self._g_filters, self.tr("Built-in filters")),
                         (self._g_sensors, self.tr("Built-in sensors"))):
            total = sum(1 for _ in self._leaves_under(g))
            vis = visible_children(g)
            g.setHidden(vis == 0 and g is not self._g_mine)
            g.setText(0, f"{label} ({vis})" if vis == total else f"{label} ({vis} of {total})")
            if toks:
                g.setExpanded(vis > 0)
        if self._g_mine.childCount() == 0:
            self._g_mine.setText(0, self.tr("My curves (none yet — use the editor →)"))
        if not toks and getattr(self, "_was_searching", False):
            # back to browsing: fold the maker groups a search opened, except
            # the ones holding a selected curve
            for g in self._maker_groups:
                g.setExpanded(any(g.child(k).isSelected() for k in range(g.childCount())))
            self._g_mine.setExpanded(True)
            self._g_filters.setExpanded(True)
        self._was_searching = bool(toks)

    def _leaves_under(self, g):
        stack = [g]
        while stack:
            it = stack.pop()
            for k in range(it.childCount()):
                c = it.child(k)
                if c.data(0, Qt.ItemDataRole.UserRole) is not None:
                    yield c
                else:
                    stack.append(c)

    def select_curves(self, names):
        want = {str(n).upper() for n in names if n and str(n) != "(None)"}
        if not want:
            return
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        first = None
        for it in self._iter_leaves():
            info = self._infos[int(it.data(0, Qt.ItemDataRole.UserRole))]
            if info.name.upper() in want:
                it.setSelected(True)
                p = it.parent()
                while p is not None:
                    p.setExpanded(True)
                    p = p.parent()
                first = first or it
        self.tree.blockSignals(False)
        if first is not None:
            self.tree.scrollToItem(first)
        self._on_selection()

    def _selected_infos(self) -> list[CurveInfo]:
        out = []
        for it in self.tree.selectedItems():
            d = it.data(0, Qt.ItemDataRole.UserRole)
            if d is not None:
                out.append(self._infos[int(d)])
        return out

    def _curve_data(self, info: CurveInfo):
        key = (info.path, info.name.upper())
        if key not in self._cache:
            self._cache[key] = read_curve(info.path, info.name)
        return self._cache[key]

    def _on_selection(self):
        self._refresh_plot()
        self._update_buttons()

    def _on_double_click(self, item, _col):
        d = item.data(0, Qt.ItemDataRole.UserRole)
        if d is not None:
            self._open_in_editor(self._infos[int(d)])

    def _update_buttons(self):
        sel = self._selected_infos()
        self.btn_open.setEnabled(len(sel) == 1)
        self.btn_delete.setEnabled(bool(sel) and all(not i.builtin for i in sel))
        self.btn_export.setEnabled(bool(sel))
        self.btn_undo.setEnabled(bool(self._undo))
        self.btn_save.setEnabled(self._wl.size >= 2)

    # ── plot ─────────────────────────────────────────────────────────────
    def _refresh_plot(self):
        self.plot.clear()
        sel = self._selected_infos()[:16]
        codes = [i.channel for i in sel]
        use_channel_colors = len(set(codes)) == len(codes) and all(c in _CHANNEL_COLORS for c in codes)
        ys = []
        editing = (self._editing or "").upper()
        for k, info in enumerate(sel):
            if self._wl.size and not info.builtin and info.name.upper() == editing:
                continue        # the working curve (drawn below) is this one
            try:
                wl, v = self._curve_data(info)
            except Exception as e:
                self._status(self.tr("Could not read {0}: {1}").format(info.name, e))
                continue
            col = _CHANNEL_COLORS[info.channel] if use_channel_colors else _PALETTE[k % len(_PALETTE)]
            _plot_curve(self.plot, wl, v, col, width=1.6, name=info.name)
            ys.append(v)
        if self._wl.size:
            order = np.argsort(self._wl, kind="mergesort")
            label = self.name_edit.text().strip() or self.tr("working curve")
            _plot_curve(self.plot, self._wl[order], self._val[order], _WORKING_COLOR, width=2.6,
                        name=f"✎ {label}", symbols=self._wl.size <= 400)
            ys.append(self._val)
        if ys:
            self.plot.enableAutoRange(axis="x")
            _fit_y(self.plot, *ys)

    def _on_plot_hover(self, pos):
        pi = self.plot.getPlotItem()
        if not pi.sceneBoundingRect().contains(pos):
            return
        p = pi.vb.mapSceneToView(pos)
        self.hover_lbl.setText(f"λ {p.x():.1f} nm · {p.y():.3f}")

    # ── editor: working data ─────────────────────────────────────────────
    def _push_undo(self):
        self._undo.append((self._wl.copy(), self._val.copy()))
        if len(self._undo) > 60:
            self._undo.pop(0)
        self.btn_undo.setEnabled(True)

    def _undo_last(self):
        if not self._undo:
            return
        wl, v = self._undo.pop()
        self._set_working(wl, v, push_undo=False)
        self._status(self.tr("Undone."))

    def _set_working(self, wl, v, *, push_undo=True, dirty=True):
        if push_undo:
            self._push_undo()
        self._wl = np.asarray(wl, dtype=np.float64).copy()
        self._val = np.asarray(v, dtype=np.float64).copy()
        self._fill_table()
        self._dirty = bool(dirty)
        self._refresh_plot()
        self._validate()
        self._update_buttons()

    def _fill_table(self):
        self._filling = True
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(int(self._wl.size))
            for i, (w, v) in enumerate(zip(self._wl, self._val)):
                self.table.setItem(i, 0, QTableWidgetItem(_fmt(w, 3)))
                self.table.setItem(i, 1, QTableWidgetItem(_fmt(v, 5)))
        finally:
            self.table.blockSignals(False)
            self._filling = False
        self.points_lbl.setText(self.tr("{0} points").format(self._wl.size))

    def _on_table_item_changed(self, item):
        if self._filling:
            return
        r, c = item.row(), item.column()
        if r >= self._wl.size:
            return
        txt = item.text()
        val, pct = _parse_number(txt, decimal_comma=("," in txt and "." not in txt))
        if val is None:
            self.table.blockSignals(True)
            item.setText(_fmt(self._wl[r] if c == 0 else self._val[r], 3 if c == 0 else 5))
            self.table.blockSignals(False)
            self._status(self.tr("“{0}” isn't a number.").format(txt))
            return
        if c == 1 and pct:
            val /= 100.0
        self._push_undo()
        if c == 0:
            self._wl[r] = val
        else:
            self._val[r] = val
        self._dirty = True
        self._refresh_plot()
        self._validate()

    def _add_row(self):
        if self._wl.size:
            step = float(np.median(np.diff(np.sort(self._wl)))) if self._wl.size > 1 else 10.0
            w = float(np.max(self._wl)) + (step if step > 0 else 10.0)
        else:
            w = 400.0
        self._set_working(np.r_[self._wl, w], np.r_[self._val, 0.0])
        r = self._wl.size - 1
        self.table.scrollToItem(self.table.item(r, 0))
        self.table.setCurrentCell(r, 0)
        self.table.editItem(self.table.item(r, 0))

    def _delete_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        keep = np.ones(self._wl.size, dtype=bool)
        keep[[r for r in rows if r < keep.size]] = False
        self._set_working(self._wl[keep], self._val[keep])
        self._status(self.tr("Removed {0} point(s).").format(len(rows)))

    def _validate(self):
        kind = self.kind_combo.currentData()
        if self._wl.size == 0:
            self.valid_lbl.setText(
                "<span style='color:#888'>" + self.tr(
                    "No data yet. Paste columns, import a file, digitize a chart image or "
                    "generate a bandpass — or pick a curve in the library and Open in Editor.") + "</span>")
            return []
        msgs = validate_curve(self._wl, self._val, kind)
        icons = {"ok": ("✓", "#6fd08c"), "info": ("ⓘ", "#9ab"), "warn": ("⚠", "#e0b040"),
                 "error": ("✖", "#ff6b6b")}
        html = []
        for lvl, m in msgs:
            ic, col = icons[lvl]
            html.append(f"<div style='color:{col}'>{ic} {m}</div>")
        self.valid_lbl.setText("".join(html))
        return msgs

    def _mark_dirty(self):
        if self._wl.size:
            self._dirty = True

    def _on_kind_changed(self, *_):
        kind = self.kind_combo.currentData()
        prev = self.cat_combo.currentData()
        self.cat_combo.blockSignals(True)
        self.cat_combo.clear()
        for code, lbl in CATEGORY_CHOICES.get(kind, []):
            self.cat_combo.addItem(lbl, code)
        i = self.cat_combo.findData(prev)
        self.cat_combo.setCurrentIndex(i if i >= 0 else 0)
        self.cat_combo.blockSignals(False)
        self._mark_dirty()
        self._validate()

    def _on_label_changed(self):
        self._mark_dirty()
        if self.name_auto_chk.isChecked():
            self._apply_auto_name()

    def _apply_auto_name(self):
        name = suggest_curve_name(self.maker_edit.text(), self.model_edit.text(),
                                  self.chan_combo.currentData() or "")
        self.name_edit.setText(name)

    def _on_name_edited(self, _text):
        self.name_auto_chk.blockSignals(True)
        self.name_auto_chk.setChecked(False)
        self.name_auto_chk.blockSignals(False)
        self._mark_dirty()

    def _set_form(self, *, kind="FILTER", category="", channel="", maker="", model="",
                  url="", notes="", name="", auto=True):
        for w in (self.kind_combo, self.cat_combo, self.chan_combo):
            w.blockSignals(True)
        i = self.kind_combo.findData(kind)
        self.kind_combo.setCurrentIndex(i if i >= 0 else 0)
        for w in (self.kind_combo,):
            w.blockSignals(False)
        self._on_kind_changed()
        self.cat_combo.blockSignals(True)
        i = self.cat_combo.findData(category)
        if i >= 0:
            self.cat_combo.setCurrentIndex(i)
        self.cat_combo.blockSignals(False)
        i = self.chan_combo.findData(channel)
        self.chan_combo.setCurrentIndex(i if i >= 0 else 0)
        self.chan_combo.blockSignals(False)
        self.maker_edit.setText(maker)
        self.model_edit.setText(model)
        self.url_edit.setText(url)
        self.notes_edit.setText(notes)
        self.name_auto_chk.blockSignals(True)
        self.name_auto_chk.setChecked(auto)
        self.name_auto_chk.blockSignals(False)
        if auto:
            self._apply_auto_name()
        else:
            self.name_edit.setText(name)

    def _meta_from_form(self) -> dict:
        return {
            "category": self.cat_combo.currentData() or "",
            "channel": self.chan_combo.currentData() or "",
            "maker": self.maker_edit.text().strip(),
            "model": self.model_edit.text().strip(),
            "url": self.url_edit.text().strip(),
            "notes": self.notes_edit.text().strip(),
            "origin": self._origin or "SASpro Curve Manager",
        }

    def _set_origin(self, origin: str):
        self._origin = origin or ""
        self.origin_lbl.setText(self.tr("Source: {0}").format(self._origin) if self._origin else "")

    def _status(self, text: str):
        self.status_lbl.setText(text)

    def _confirm_discard(self) -> bool:
        if not (self._dirty and self._wl.size >= 2):
            return True
        ans = QMessageBox.question(
            self, self.tr("Unsaved Curve"),
            self.tr("The curve in the editor hasn't been saved. Discard it?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return ans == QMessageBox.StandardButton.Yes

    def _new_curve(self):
        if not self._confirm_discard():
            return
        self._pending = []
        self._update_queue_banner()
        self._editing = None
        self._undo.clear()
        self._set_form()
        self._set_origin("")
        self._set_working(np.empty(0), np.empty(0), push_undo=False, dirty=False)
        self._dirty = False
        self.name_edit.setFocus()
        self._status(self.tr("New curve — paste, import, digitize or generate its data."))

    def _open_selected_in_editor(self):
        sel = self._selected_infos()
        if len(sel) == 1:
            self._open_in_editor(sel[0])

    def _open_in_editor(self, info: CurveInfo):
        if not self._confirm_discard():
            return
        try:
            wl, v = self._curve_data(info)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Open Curve"), str(e))
            return
        self._pending = []
        self._update_queue_banner()
        self._undo.clear()
        if info.builtin:
            self._editing = None
            self._set_form(kind=info.kind, category=info.category, channel=info.channel,
                           maker=info.maker if info.labelled else "", model=info.model,
                           url=info.url, notes=info.notes, name=f"{info.name}_MOD", auto=False)
            self._set_origin(f"Copy of built-in {info.name}")
            self._set_working(wl, v, push_undo=False, dirty=True)
            self._status(self.tr("Editing a copy of built-in {0}. Built-ins stay unchanged; give the copy "
                                 "its own name and save it to My Curves.").format(info.name))
        else:
            self._editing = info.name
            self._set_form(kind=info.kind, category=info.category, channel=info.channel,
                           maker=info.maker, model=info.model, url=info.url, notes=info.notes,
                           name=info.name, auto=False)
            self._set_origin(info.origin)
            self._set_working(wl, v, push_undo=False, dirty=False)
            self._status(self.tr("Editing {0}.").format(info.name))

    # ── getting data in ──────────────────────────────────────────────────
    def _incoming_mode(self) -> str | None:
        """'replace', 'append' or None (cancel) when data arrives while the
        editor already has points."""
        if self._wl.size == 0 or not self._dirty:
            return "replace"
        box = QMessageBox(self)
        box.setWindowTitle(self.tr("Editor Has Data"))
        box.setText(self.tr("The editor already holds {0} points. Replace them, or append the new "
                            "points to them?").format(self._wl.size))
        b_rep = box.addButton(self.tr("Replace"), QMessageBox.ButtonRole.AcceptRole)
        b_app = box.addButton(self.tr("Append"), QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is b_rep:
            return "replace"
        if box.clickedButton() is b_app:
            return "append"
        return None

    def _ingest(self, curves: list[dict], origin: str, notes: list[str] | None = None,
                meta: dict | None = None):
        """curves: [{label, wl, v, channel}] — the first goes into the editor,
        the rest queue up behind it."""
        if not curves:
            return
        mode = self._incoming_mode()
        if mode is None:
            return
        first = curves[0]
        if mode == "append":
            wl, v = clean_curve(np.r_[self._wl, first["wl"]], np.r_[self._val, first["v"]])
            self._set_working(wl, v)
        else:
            # New data is a new curve. Saving it under an existing name asks
            # first, so a library curve can't be overwritten by accident.
            fresh = self._editing is None and not self._wl.size
            self._editing = None
            self._undo.clear()
            if fresh:
                self._apply_meta_hint(meta or {})
            if first.get("channel"):
                i = self.chan_combo.findData(first["channel"])
                if i >= 0:
                    self.chan_combo.setCurrentIndex(i)
            self._set_origin(origin)
            self._set_working(first["wl"], first["v"])
        self._pending = [dict(c, origin=origin) for c in curves[1:]]
        self._update_queue_banner(first.get("label"))
        msg = self.tr("Loaded {0} points").format(len(first["wl"]))
        if notes:
            msg += " — " + "; ".join(notes)
        self._status(msg + ".")

    def _apply_meta_hint(self, meta: dict):
        """Labels carried in '# key: value' comments (our CSV export)."""
        if not meta:
            return
        kind = str(meta.get("type", "")).upper()
        if kind in KINDS:
            i = self.kind_combo.findData(kind)
            self.kind_combo.setCurrentIndex(i)
        if meta.get("category"):
            i = self.cat_combo.findData(meta["category"])
            if i >= 0:
                self.cat_combo.setCurrentIndex(i)
        if meta.get("channel"):
            i = self.chan_combo.findData(str(meta["channel"]).upper())
            if i >= 0:
                self.chan_combo.setCurrentIndex(i)
        for key, w in (("maker", self.maker_edit), ("model", self.model_edit),
                       ("url", self.url_edit), ("notes", self.notes_edit)):
            if meta.get(key):
                w.setText(str(meta[key]))
        if meta.get("name"):
            self.name_auto_chk.setChecked(False)
            self.name_edit.setText(sanitize_curve_name(meta["name"]))
        elif self.name_auto_chk.isChecked():
            self._apply_auto_name()

    def _update_queue_banner(self, current_label: str | None = None):
        if not self._pending:
            self.queue_frame.setVisible(False)
            return
        nxt = ", ".join(c.get("label") or "?" for c in self._pending[:4])
        more = "" if len(self._pending) <= 4 else self.tr(" and {0} more").format(len(self._pending) - 4)
        cur = self.tr("Editing “{0}”. ").format(current_label) if current_label else ""
        self.queue_lbl.setText(cur + self.tr("Waiting: {0}{1} — each loads after you save (or skip) "
                                             "this one. Maker / model carry over.").format(nxt, more))
        self.queue_frame.setVisible(True)

    def _advance_queue(self):
        if not self._pending:
            self.queue_frame.setVisible(False)
            return
        c = self._pending.pop(0)
        self._editing = None
        self._undo.clear()
        i = self.chan_combo.findData(c.get("channel") or "")
        self.chan_combo.setCurrentIndex(i if i >= 0 else 0)
        self.name_auto_chk.setChecked(True)
        self._apply_auto_name()
        self._set_origin(c.get("origin", ""))
        self._set_working(c["wl"], c["v"], push_undo=False)
        self._update_queue_banner(c.get("label"))
        self._status(self.tr("Loaded “{0}”. Check the channel and name, then save.").format(c.get("label", "")))

    def _paste_data(self):
        cb = QApplication.clipboard()
        md = cb.mimeData()
        text = cb.text() or ""
        if not text.strip() and md is not None and md.hasImage():
            img = cb.image()
            if not img.isNull():
                if QMessageBox.question(
                        self, self.tr("Paste"),
                        self.tr("The clipboard holds an image, not numbers. Open it in the digitizer?")
                ) == QMessageBox.StandardButton.Yes:
                    self._open_digitizer(image=img)
                return
        if not text.strip():
            QMessageBox.information(self, self.tr("Paste Data"),
                                    self.tr("The clipboard is empty. Copy two or more columns (wavelength, "
                                            "response…) from a spreadsheet, web page or text file first."))
            return
        self._ingest_text(text, self.tr("Pasted data"))

    def _ingest_text(self, text: str, origin: str):
        try:
            parsed = parse_curve_table(text)
        except ValueError as e:
            QMessageBox.warning(self, self.tr("Read Data"), str(e))
            return
        if parsed.data.shape[1] == 2:
            wl, v, notes = auto_units(parsed.data[:, 0], parsed.data[:, 1],
                                      percent_hint=1 in parsed.percent_cols)
            wl, v = clean_curve(wl, v)
            curves = [{"label": parsed.names[1], "wl": wl, "v": v,
                       "channel": guess_channel(parsed.names[1]) or
                       str(parsed.meta.get("channel", "")).upper()}]
        else:
            dlg = _ColumnPickerDialog(parsed, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            curves = dlg.results()
            notes = []
            if not curves:
                return
        self._ingest(curves, origin, notes, parsed.meta)

    def _import_file(self):
        start = QSettings().value("CurveManager/last_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Import Curve Data"), start,
            self.tr("Curve data (*.csv *.txt *.tsv *.dat *.tab *.fits *.fit);;All Files (*)"))
        if not path:
            return
        QSettings().setValue("CurveManager/last_dir", os.path.dirname(path))
        if path.lower().endswith((".fits", ".fit")):
            self._import_pack(path)
            return
        text = None
        for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                if "\x00" in text:
                    continue
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except OSError as e:
                QMessageBox.warning(self, self.tr("Import"), str(e))
                return
        if text is None:
            QMessageBox.warning(self, self.tr("Import"), self.tr("Could not read the file as text."))
            return
        self._ingest_text(text, f"File: {os.path.basename(path)}")

    def _open_digitizer(self, image: QImage | None = None):
        dlg = CurveDigitizerDialog(self, image=image)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.captured:
            return
        curves = [{"label": c["label"], "wl": c["wl"], "v": c["v"], "channel": c["channel"]}
                  for c in dlg.captured]
        self._ingest(curves, dlg.origin())

    def _open_generator(self):
        dlg = BandpassDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_wl is None:
            return
        if self._wl.size == 0 and self._editing is None:
            i = self.kind_combo.findData("FILTER")
            self.kind_combo.setCurrentIndex(i)
            i = self.cat_combo.findData(dlg.suggested_category)
            if i >= 0:
                self.cat_combo.setCurrentIndex(i)
        self._ingest([{"label": self.tr("bandpass"), "wl": dlg.result_wl, "v": dlg.result_val,
                       "channel": dlg.suggested_channel}], dlg.origin)

    # ── tools ────────────────────────────────────────────────────────────
    def _need_data(self) -> bool:
        if self._wl.size < 2:
            self._status(self.tr("The editor needs at least two points first."))
            return False
        return True

    def _tool_sort(self):
        if self._need_data():
            wl, v = clean_curve(self._wl, self._val)
            self._set_working(wl, v)
            self._status(self.tr("Sorted — {0} points.").format(wl.size))

    def _tool_resample(self):
        if not self._need_data():
            return
        step, ok = QInputDialog.getDouble(self, self.tr("Resample"), self.tr("Uniform step (nm):"),
                                          1.0, 0.01, 50.0, 2)
        if ok:
            wl, v = resample_curve(self._wl, self._val, step)
            self._set_working(wl, v)
            self._status(self.tr("Resampled to {0} nm — {1} points.").format(_fmt(step, 2), wl.size))

    def _tool_smooth(self):
        if not self._need_data():
            return
        win, ok = QInputDialog.getDouble(self, self.tr("Smooth"),
                                         self.tr("Smoothing window (nm). Keep it well below the narrowest "
                                                 "feature — steep filter edges soften otherwise:"),
                                         5.0, 0.1, 200.0, 1)
        if ok:
            wl, v = smooth_curve(self._wl, self._val, win)
            self._set_working(wl, v)
            self._status(self.tr("Smoothed over {0} nm.").format(_fmt(win, 1)))

    def _tool_trim(self):
        if not self._need_data():
            return
        dlg = _RangeDialog(float(self._wl.min()), float(self._wl.max()), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            lo, hi = sorted((dlg.lo.value(), dlg.hi.value()))
            keep = (self._wl >= lo) & (self._wl <= hi)
            self._set_working(self._wl[keep], self._val[keep])

    def _tool_clip(self):
        if self._need_data():
            self._set_working(self._wl, np.clip(self._val, 0.0, 1.0))

    def _tool_normalize(self):
        if self._need_data():
            peak = float(np.nanmax(self._val))
            if peak > 0:
                self._set_working(self._wl, self._val / peak)

    def _tool_scale_values(self, f: float):
        if self._need_data():
            self._set_working(self._wl, self._val * f)

    def _tool_scale_wl(self, f: float):
        if self._need_data():
            self._set_working(self._wl * f, self._val)

    # ── saving / deleting ────────────────────────────────────────────────
    def _save_working(self):
        if self._wl.size < 2:
            self._status(self.tr("Nothing to save yet."))
            return False
        name = sanitize_curve_name(self.name_edit.text())
        if not name:
            QMessageBox.warning(self, self.tr("Save Curve"),
                                self.tr("Give the curve a name (or fill Maker / Model / Channel with Auto on)."))
            self.name_edit.setFocus()
            return False
        kind = self.kind_combo.currentData()
        wl, v = clean_curve(self._wl, self._val)
        errors = [m for lvl, m in validate_curve(wl, v, kind) if lvl == "error"]
        if errors:
            QMessageBox.warning(self, self.tr("Save Curve"),
                                self.tr("This curve can't be saved yet:\n\n") + "\n".join(errors))
            return False
        builtin = {i.name.upper() for i in self._infos if i.builtin}
        if name.upper() in builtin:
            QMessageBox.warning(self, self.tr("Save Curve"),
                                self.tr("A built-in curve is already called {0}. Pick a different name — "
                                        "e.g. add a suffix or your rig.").format(name))
            return False
        replace = None
        editing = self._editing
        if editing and editing.upper() != name.upper():
            box = QMessageBox(self)
            box.setWindowTitle(self.tr("Name Changed"))
            box.setText(self.tr("Rename {0} to {1}, or keep {0} and save {1} as a new curve?")
                        .format(editing, name))
            b_ren = box.addButton(self.tr("Rename"), QMessageBox.ButtonRole.AcceptRole)
            b_new = box.addButton(self.tr("Save as New"), QMessageBox.ButtonRole.ActionRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is b_ren:
                replace = editing
            elif box.clickedButton() is not b_new:
                return False
        mine = {i.name.upper() for i in self._infos if not i.builtin}
        same_as_editing = editing is not None and editing.upper() == name.upper()
        if name.upper() in mine and not same_as_editing:
            if QMessageBox.question(
                    self, self.tr("Overwrite?"),
                    self.tr("You already have a curve called {0}. Overwrite it?").format(name),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return False
        try:
            save_curve(self.user_path, name, kind, wl, v, self._meta_from_form(), replace=replace)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Save Curve"), self.tr("Could not save:\n{0}").format(e))
            return False
        self._editing = name
        self.name_edit.setText(name)
        self._set_working(wl, v, push_undo=False, dirty=False)
        self._dirty = False
        self.reload_library(select=[name])
        notify_library_changed()
        self._status(self.tr("Saved {0} to My Curves — it's now in the SPCC and SSSC lists.").format(name))
        if self._pending:
            self._advance_queue()
        return True

    def _delete_selected(self):
        sel = [i for i in self._selected_infos() if not i.builtin]
        if not sel:
            return
        names = [i.name for i in sel]
        listing = "\n".join(names[:12]) + ("\n…" if len(names) > 12 else "")
        if QMessageBox.question(
                self, self.tr("Delete Curves"),
                self.tr("Delete {0} curve(s) from My Curves?\n\n{1}\n\nThe previous file is kept as "
                        "usercustomcurves.fits.bak.").format(len(names), listing),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            n = delete_curves(self.user_path, names)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Delete Curves"), self.tr("Could not delete:\n{0}").format(e))
            return
        if self._editing and self._editing.upper() in {x.upper() for x in names}:
            self._editing = None
            if self._wl.size:
                self._dirty = True
        self.reload_library(select=[])
        notify_library_changed()
        self._status(self.tr("Deleted {0} curve(s).").format(n))

    # ── export / import packs ────────────────────────────────────────────
    def _export_items(self, infos) -> list[dict]:
        items = []
        for info in infos:
            wl, v = self._curve_data(info)
            meta = info.meta()
            if info.builtin and not meta.get("origin"):
                meta["origin"] = "SASpro built-in"
            items.append({"name": info.name, "kind": info.kind, "wl_nm": wl, "values": v, "meta": meta})
        return items

    def _export_csv(self):
        sel = self._selected_infos()
        if not sel:
            return
        start = QSettings().value("CurveManager/last_dir", "")
        try:
            items = self._export_items(sel)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Export"), str(e))
            return
        safe = lambda n: re.sub(r'[\\/:*?"<>|]+', "-", n)
        if len(items) == 1:
            it = items[0]
            path, _ = QFileDialog.getSaveFileName(self, self.tr("Export Curve as CSV"),
                                                  os.path.join(start, safe(it["name"]) + ".csv"),
                                                  self.tr("CSV (*.csv)"))
            if not path:
                return
            write_curve_csv(path, it["name"], it["kind"], it["wl_nm"], it["values"], it["meta"])
            QSettings().setValue("CurveManager/last_dir", os.path.dirname(path))
            self._status(self.tr("Exported {0}.").format(os.path.basename(path)))
            return
        folder = QFileDialog.getExistingDirectory(self, self.tr("Export Curves to Folder"), start)
        if not folder:
            return
        for it in items:
            write_curve_csv(os.path.join(folder, safe(it["name"]) + ".csv"),
                            it["name"], it["kind"], it["wl_nm"], it["values"], it["meta"])
        QSettings().setValue("CurveManager/last_dir", folder)
        self._status(self.tr("Exported {0} CSV files to {1}.").format(len(items), folder))

    def _export_pack(self):
        sel = self._selected_infos()
        if not sel:
            return
        start = QSettings().value("CurveManager/last_dir", "")
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Export Curve Pack"),
                                              os.path.join(start, "saspro_curves.fits"),
                                              self.tr("FITS curve pack (*.fits)"))
        if not path:
            return
        try:
            write_curve_pack(path, self._export_items(sel))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Export"), self.tr("Could not write the pack:\n{0}").format(e))
            return
        QSettings().setValue("CurveManager/last_dir", os.path.dirname(path))
        self._status(self.tr("Exported {0} curve(s) to {1}.").format(len(sel), os.path.basename(path)))

    def _import_pack(self, path: str | None = None):
        if not path:
            start = QSettings().value("CurveManager/last_dir", "")
            path, _ = QFileDialog.getOpenFileName(self, self.tr("Import Curve Pack"), start,
                                                  self.tr("FITS curve pack (*.fits *.fit);;All Files (*)"))
            if not path:
                return
            QSettings().setValue("CurveManager/last_dir", os.path.dirname(path))
        try:
            items = read_curve_pack(path)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Import Pack"), self.tr("Could not read:\n{0}").format(e))
            return
        if not items:
            QMessageBox.information(self, self.tr("Import Pack"),
                                    self.tr("No filter or sensor curves found in that file."))
            return
        builtin = {i.name.upper() for i in self._infos if i.builtin}
        mine = {i.name.upper() for i in self._infos if not i.builtin}
        skipped = [it["name"] for it in items if it["name"].upper() in builtin]
        items = [it for it in items if it["name"].upper() not in builtin]
        clash = [it["name"] for it in items if it["name"].upper() in mine]
        if clash:
            ans = QMessageBox.question(
                self, self.tr("Import Pack"),
                self.tr("{0} curve(s) in the pack have the same name as one of yours:\n\n{1}\n\n"
                        "Overwrite yours? (No keeps yours and skips those.)")
                .format(len(clash), "\n".join(clash[:12]) + ("\n…" if len(clash) > 12 else "")),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.No)
            if ans == QMessageBox.StandardButton.Cancel:
                return
            if ans != QMessageBox.StandardButton.Yes:
                skipped += clash
                items = [it for it in items if it["name"].upper() not in mine]
        if not items:
            QMessageBox.information(self, self.tr("Import Pack"), self.tr("Nothing new to import."))
            return
        try:
            save_curves(self.user_path, items)
        except Exception as e:
            QMessageBox.critical(self, self.tr("Import Pack"), self.tr("Could not save:\n{0}").format(e))
            return
        self.reload_library(select=[it["name"] for it in items])
        notify_library_changed()
        msg = self.tr("Imported {0} curve(s).").format(len(items))
        if skipped:
            msg += " " + self.tr("Skipped {0} (name already used by a built-in or one of yours).").format(len(skipped))
        self._status(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (one window per app)
# ─────────────────────────────────────────────────────────────────────────────

_INSTANCE: SpectralCurveManager | None = None


def _forget_instance(*_):
    global _INSTANCE
    _INSTANCE = None


def open_curve_manager(parent=None, sasp_data_path: str | None = None,
                       select=None) -> SpectralCurveManager:
    """Show the manager (reusing an open one). Pass the main window as parent
    so the manager outlives the SPCC / SSSC dialog it was opened from.
    `select` pre-selects curves by name (e.g. the current filter chain)."""
    global _INSTANCE
    inst = _INSTANCE
    if inst is not None:
        try:
            from PyQt6 import sip
            if sip.isdeleted(inst):
                inst = None
        except Exception:
            inst = None
    if inst is None:
        inst = SpectralCurveManager(parent=parent, sasp_data_path=sasp_data_path)
        inst.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        inst.destroyed.connect(_forget_instance)
        try:
            from setiastro.saspro.resources import spcc_icon_path
            inst.setWindowIcon(QIcon(spcc_icon_path))
        except Exception:
            pass
        _INSTANCE = inst
    inst.show()
    inst.raise_()
    inst.activateWindow()
    if select:
        inst.select_curves(select)
    return inst
