#saspro/skychart_engine.py
# Copyright (C) Franklin Marek / SetiAstro. Part of Seti Astro Suite Pro (GPLv3).
# =============================================================================
#  Sky chart engine
#
#  Annotated sky charts as vector graphics (SVG / PDF) or raster images: stars
#  with halos, deep-sky objects drawn to scale, constellation lines, borders and
#  names, the Milky Way, emission and dark nebulae from all-sky maps, the
#  coordinate grid and a legend. Used by Finder Chart for its "Drawn chart"
#  style and for the sky-chart overlay on survey images.
#
#  Based on the Sky Chart Generator by Jean d'Hooghvorst, contributed to SASpro.
#
#  Data (bundled in setiastro/data/skychart, built by skychart_data_build.py):
#    - HYG star database v4.1 (D. Nash, CC BY-SA 4.0)
#    - OpenNGC and its nebula outlines (M. Verga, CC BY-SA 4.0)
#    - d3-celestial constellations and Milky Way (O. Frohn, BSD-3)
#    - Stellarium "modern_st" constellation figures (GPL-2.0-or-later)
#    - Washington Double Star catalog (USNO)
#    - Finkbeiner 2003 H-alpha and SFD 1998 dust maps (NASA LAMBDA)
#    - Milky Way starlight integrated from Gaia DR3 (ESA/Gaia/DPAC)
#  At run time: SASpro's celestial catalog, and the local Gaia DR3 libraries.
# =============================================================================

SKYCHART_VERSION = "2.8.2"
__version__ = SKYCHART_VERSION

import csv
import json
import math
import os
import re
import sqlite3
import sys
import time
import zlib
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPointF, QSize, QSizeF, QMarginsF, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPainterPath, QPen, QBrush, QColor, QFont, QFontMetricsF,
    QTransform, QPolygonF, QImage, QPicture, QPageSize, QPdfWriter,
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QDoubleSpinBox, QComboBox, QCheckBox,
    QTabWidget, QWidget, QTreeWidget, QTreeWidgetItem, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QFileDialog, QMessageBox, QGroupBox,
    QProgressDialog, QSplitter, QFontComboBox, QSpinBox, QHeaderView,
    QListWidget, QListWidgetItem, QDialogButtonBox,
)

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# -----------------------------------------------------------------------------
# Data files
# -----------------------------------------------------------------------------
# All-sky maps (HEALPix NESTED, nside 512, galactic), log10-quantised to uint8.
SKY_MAPS = {"halpha": "halpha_map.npz", "dust": "dust_map.npz"}
# PixInsight's SkyChartGenerator figures, measured pixel by pixel on its own charts:
# Sky & Telescope (IAU) figures, except these constellations (Hipparcos numbers),
# which PixInsight draws differently (Cepheus and Draco as Stellarium's 'modern'
# culture; the Cygnus neck through eta, chi and phi; 100 Her; the 51-49-xi And arm).
CON_FIGURES_PI = {
    "And": [[677, 3092, 5447, 9640], [113726, 116631, 116805, 116584], [116631, 2912, 3092],
            [2912, 5447, 4436, 3881, 6411, 6999, 7607], [3092, 3031, 3693, 4463]],
    "Cep": [[109492, 112724, 106032, 105199, 109492], [112724, 116727, 106032]],
    "Dra": [[87585, 87833, 85670, 85829, 87585, 94376, 97433, 94648, 89937, 83895, 80331, 78527,
             75458, 68756, 61281, 56211]],
    "Cyg": [[102098, 100453, 102488], [100453, 98110, 97629, 96683, 95947],
            [100453, 97165, 95853, 94779], [95853, 99848, 102098, 103413, 104732, 102488]],
    "Her": [[84379, 84345, 80816, 80170, 80816, 81693, 83207, 81693, 81833, 81126, 79992, 81126,
             81833, 84380, 85112, 87808, 86414, 87808, 85112, 84380, 83207, 84379, 85693, 86974,
             87933, 88794, 88817], [77760, 79101, 79992], [80170, 80463, 81008]],
}
GAIA_CACHE_MAG = 11.5      # all-sky Gaia cache (for wide fields) goes this deep
GAIA_MIN_MAG = 7.0         # Gaia completes the bright catalog from this magnitude


def data_dir() -> Path:
    """Bundled, read-only chart tables (setiastro/data/skychart)."""
    from setiastro.saspro.resources import get_data_path
    return Path(get_data_path("data/skychart"))


def cache_dir() -> Path:
    """Writable folder for caches built on this machine (Gaia bright-star cache,
    online colours). Never the bundled data folder: that is read-only once
    installed."""
    base = None
    try:
        from PyQt6.QtCore import QStandardPaths
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    except Exception:
        base = None
    p = Path(base) if base else Path.home() / ".saspro"
    p = p / "skychart_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _gaia_tier_floor(stem: str) -> float:
    """Faintest-first lower bound of a Gaia astrometric tier from its file name:
    gaia_astro_lt15 -> -inf, gaia_astro_150_160 -> 15.0, gaia_astro_gt213 -> 21.3."""
    m = re.match(r"gaia_astro_(\d+)_(\d+)$", stem)
    if m:
        return float(m.group(1)) / 10.0
    m = re.match(r"gaia_astro_gt(\d+)$", stem)
    if m:
        return float(m.group(1)) / 10.0
    return float("-inf")


def gaia_box(ra_lo, ra_hi, dec_lo, dec_hi, gmin, gmax):
    """(ra, dec, G, source_id) arrays from SASpro's local Gaia DR3 library.

    One query per declination zone with dec_zone pinned by equality, so the
    (dec_zone, ra) index bounds both columns -- the access pattern
    GaiaAstrometricLibrary.find_stars_in_box documents (a dec_zone BETWEEN walks
    every row of the zone span, measured 500x slower). Tiers that cannot hold a
    star brighter than gmax are skipped: on a G <= 11 chart the deep tiers would
    otherwise be walked for nothing."""
    empty = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64))
    try:
        from setiastro.saspro.gaia_database import get_astro_library
        lib = get_astro_library()
    except Exception:
        return empty
    conns = dict(getattr(lib, "_connections", {}) or {})
    if not conns:
        return empty
    zh = float(getattr(lib, "_zone_height_deg", 1.0) or 1.0)
    z_lo = int(math.floor((max(dec_lo, -90.0) + 90.0) / zh))
    z_hi = int(math.floor((min(dec_hi, 90.0) + 90.0) / zh))
    if (ra_hi - ra_lo) >= 359.9:
        ranges = [(0.0, 360.0)]
    else:
        lo, hi = ra_lo % 360.0, ra_hi % 360.0
        ranges = [(lo, hi)] if lo <= hi else [(lo, 360.0), (0.0, hi)]
    ra, dec, mg, sid = [], [], [], []
    sql = ("SELECT ra, dec, phot_g_mean_mag, source_id FROM stars WHERE dec_zone = ? "
           "AND ra BETWEEN ? AND ? AND phot_g_mean_mag >= ? AND phot_g_mean_mag <= ?")
    for fname, conn in conns.items():
        if _gaia_tier_floor(Path(fname).stem) >= gmax:
            continue
        try:
            for lo, hi in ranges:
                for zone in range(z_lo, z_hi + 1):
                    rows = conn.execute(sql, (zone, lo, hi, gmin, gmax)).fetchall()
                    if rows:
                        a = np.array([r[:3] for r in rows], dtype=np.float64)
                        ra.append(a[:, 0]); dec.append(a[:, 1]); mg.append(a[:, 2])
                        sid.append(np.array([r[3] for r in rows], dtype=np.int64))
        except Exception:
            continue
    if not ra:
        return empty
    ra, dec, mg, sid = np.concatenate(ra), np.concatenate(dec), np.concatenate(mg), np.concatenate(sid)
    sel = (dec >= dec_lo) & (dec <= dec_hi)
    return ra[sel], dec[sel], mg[sel], sid[sel]


def gaia_library_dir():
    """Folder of the local Gaia DR3 astrometric library, or None if none installed."""
    try:
        from setiastro.saspro.gaia_database import get_astro_library_dir
        p = Path(get_astro_library_dir())
        if p.is_dir() and any(f for f in p.glob("gaia_astro_*.sqlite") if "neighborhood" not in f.stem):
            return p
    except Exception:
        pass
    return None


# -----------------------------------------------------------------------------
# Catalog loading
# -----------------------------------------------------------------------------
GREEK = {
    "alp": "α", "bet": "β", "gam": "γ", "del": "δ", "eps": "ε",
    "zet": "ζ", "eta": "η", "the": "θ", "iot": "ι", "kap": "κ",
    "lam": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omi": "ο",
    "pi": "π", "rho": "ρ", "sig": "σ", "tau": "τ", "ups": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "ome": "ω",
}

# DSO kinds
K_GALAXY, K_GROUP, K_OC, K_GC, K_PN, K_NEB, K_CLNEB, K_RFN, K_DARK, K_ASSOC = range(10)
KIND_NAMES = {
    K_GALAXY: "Galaxy", K_GROUP: "Group of galaxies", K_OC: "Open cluster",
    K_GC: "Globular cluster", K_PN: "Planetary nebula", K_NEB: "Bright nebula",
    K_CLNEB: "Cluster with nebula", K_RFN: "Reflection nebula",
    K_DARK: "Dark nebula", K_ASSOC: "Stellar association",
}
_ONGC_KIND = {
    "G": K_GALAXY, "GPair": K_GALAXY, "GTrpl": K_GALAXY, "GGroup": K_GROUP,
    "OCl": K_OC, "GCl": K_GC, "PN": K_PN, "Cl+N": K_CLNEB, "Neb": K_NEB,
    "HII": K_NEB, "EmN": K_NEB, "SNR": K_NEB, "RfN": K_RFN, "DrkN": K_DARK,
    "*Ass": K_ASSOC,
}


def _f(s, default=float("nan")):
    try:
        s = s.strip()
        return float(s) if s else default
    except Exception:
        return default


def _lead_mag(s) -> float:
    """'17.1 - 10th bri glxy' -> 17.1; 'Bright' -> NaN."""
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", s or "")
    return float(m.group(1)) if m else float("nan")


def _sex2deg(s: str) -> float:
    s = s.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    parts = [float(x) for x in re.split(r"[:\s]+", s.lstrip("+-")) if x]
    while len(parts) < 3:
        parts.append(0.0)
    return sign * (parts[0] + parts[1] / 60.0 + parts[2] / 3600.0)


class ChartData:
    """All catalogs, loaded once and cached in memory."""

    def __init__(self, log=print):
        self.log = log
        d = data_dir()
        self._load_stars(d)
        self._load_dsos(d)
        self._load_constellations(d)
        self._load_milkyway(d)
        self._load_outlines(d)
        self._load_saspro_catalog()
        self._load_wds(d)
        self._gaia = None
        self._xp_ci = {}          # Gaia source_id -> B-V derived from its XP spectrum
        self._xp_note = False

    # ---- double stars (WDS) ----------------------------------------------
    def _load_wds(self, d: Path):
        """Primaries of real double stars from the Washington Double Star catalog: a
        companion 0.5-45" away and at most 2 mag fainter (or brighter than mag 7).
        Faint, distant optical companions (Vega, Deneb...) do not count."""
        self.w_ra = self.w_dec = self.w_mag = None
        try:
            z = np.load(d / "wds_doubles.npz", allow_pickle=False)
            self.w_ra, self.w_dec, self.w_mag = z["ra"], z["dec"], z["mag"]
        except Exception as e:
            self.log(f"Double star catalog not used: {e}")

    # ---- stars ---------------------------------------------------------
    def _load_stars(self, d: Path):
        z = np.load(d / "stars_hyg.npz", allow_pickle=False)
        self.s_ra, self.s_dec, self.s_mag, self.s_ci = z["ra"], z["dec"], z["mag"], z["ci"]
        self.s_name, self.s_bayer, self.s_flam = z["name"], z["bayer"], z["flam"]
        self.s_con, self.s_var, self.s_dbl = z["con"], z["var"], z["dbl"]
        self.s_hip, self.s_hd, self.s_varname = z["hip"], z["hd"], z["varname"]

    # ---- deep sky --------------------------------------------------------
    def _load_dsos(self, d: Path):
        rows = []
        for fn in ("NGC.csv", "addendum.csv"):
            p = d / fn
            if not p.exists():
                continue
            with open(p, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter=";"):
                    kind = _ONGC_KIND.get(r["Type"])
                    if kind is None or not r["RA"] or not r["Dec"]:
                        continue
                    nm = r["Name"].strip()
                    mnum = r.get("M", "").strip()
                    m = re.match(r"^([A-Za-z]+)\s*0*(\d+)\s*(.*)$", nm)
                    pre, num, suf = (m.group(1), m.group(2), m.group(3).strip()) if m else (nm, "", "")
                    if suf and not re.fullmatch(r"[A-Za-z]", suf):
                        continue                      # "NGC1234 NED02" sub-components
                    if mnum:
                        layer, label, key = "Messier", f"M{int(mnum)}", f"M{int(mnum)}"
                    elif pre == "NGC":
                        layer, label, key = "NGC", f"{num}{suf}", f"NGC{num}{suf}"
                    elif pre == "IC":
                        layer, label, key = "IC", f"IC {num}{suf}", f"IC{num}{suf}"
                    else:
                        layer, label, key = "Other", f"{pre} {num}".strip(), f"{pre}{num}"
                    mag = _f(r["V-Mag"])
                    if mag != mag:
                        mag = _f(r["B-Mag"])
                    maj, mnr = _f(r["MajAx"]), _f(r["MinAx"])
                    if mnr != mnr:
                        mnr = maj
                    alias = [key.upper()]
                    if pre in ("NGC", "IC") and mnum:
                        alias.append(f"{pre}{num}{suf}".upper())
                    common = r.get("Common names", "").strip()
                    rows.append((layer, label, _sex2deg(r["RA"]) * 15.0, _sex2deg(r["Dec"]), mag,
                                 maj, mnr, _f(r["PosAng"]), kind, common, alias))
        self.dsos = rows
        self.dso_index = {}
        for i, r in enumerate(rows):
            for a in r[10]:
                self.dso_index.setdefault(a, i)
            for c in r[9].split(","):
                c = c.strip().upper()
                if c:
                    self.dso_index.setdefault(c, i)

    def _load_saspro_catalog(self):
        """Optional extra layers from SASpro's bundled celestial_catalog.csv."""
        self.extra = {}
        path = None
        try:
            from setiastro.saspro.resources import get_data_path
            path = Path(get_data_path("data/catalogs/celestial_catalog.csv"))
        except Exception:
            path = None
        if path is None or not path.exists():
            return
        want = {"Sharpless": "Sharpless", "LBN": "LBN", "LDN": "LDN", "Abell": "Abell", "PNG": "PNG"}
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for r in csv.DictReader(f):
                    cat = (r.get("Catalog") or "").strip()
                    if cat not in want:
                        continue
                    t = (r.get("Type") or "").lower()
                    if cat == "Abell" or "galaxy cluster" in t:
                        kind = K_GROUP
                    elif cat == "PNG" or "planetary" in t or " pn" in " " + t:
                        kind = K_PN
                    elif cat == "LDN" or "dark" in t:
                        kind = K_DARK
                    elif "reflection" in t:
                        kind = K_RFN
                    else:
                        kind = K_NEB
                    if cat == "PNG" and (r.get("Type") or "").strip() not in (
                            "Planetary Nebula", "True PN", "Likely PN"):
                        continue                          # candidates and unknown objects
                    maj = mnr = float("nan")
                    info = (r.get("Info") or "").strip()
                    # the Info column is a size only for nebulae (Abell gives a galaxy count)
                    if cat != "Abell" and re.fullmatch(r"[\d.]+(\s*x\s*[\d.]+)?(\s*arc\s*min)?", info):
                        m = re.findall(r"\d+(?:\.\d+)?", info)
                        maj = float(m[0])
                        mnr = float(m[1]) if len(m) > 1 else maj
                    label = re.sub(r"\s+", " ", (r.get("Name") or "").strip())
                    self.extra.setdefault(cat, []).append(
                        (cat, label, _f(r.get("RA", "")), _f(r.get("Dec", "")), _lead_mag(r.get("Magnitude", "")),
                         maj, mnr, float("nan"), kind, "", []))
        except Exception as e:
            self.log(f"SASpro catalog not loaded: {e}")

    # ---- constellations --------------------------------------------------
    def _load_constellations(self, d: Path):
        self.con_names, self.con_lines, self.con_bounds = {}, [], []
        j = json.load(open(d / "constellations.json", encoding="utf-8"))
        for ft in j["features"]:
            c = ft["geometry"]["coordinates"]
            pr = ft["properties"]
            cid = ft["id"] if ft["id"] not in self.con_names else ft["id"] + "2"   # Serpens Caput / Cauda
            self.con_names[cid] = (pr.get("name", ft["id"]).replace(" ", " "), float(c[0]), float(c[1]),
                                        int(pr.get("rank", 2) or 2))
        if not self._load_stellarium_lines(d):
            j = json.load(open(d / "constellations.lines.json", encoding="utf-8"))
            for ft in j["features"]:
                for ln in ft["geometry"]["coordinates"]:
                    self.con_lines.append(np.array(ln, dtype=np.float64))
        j = json.load(open(d / "constellations.bounds.json", encoding="utf-8"))
        seen = set()
        self.con_polys = {}
        for ft in j["features"]:
            g = ft["geometry"]
            polys = g["coordinates"] if g["type"] == "Polygon" else [r for p in g["coordinates"] for r in p]
            self.con_polys.setdefault(ft["id"], []).extend(np.array(r, dtype=np.float64) for r in polys)
            for ring in polys:
                a = np.array(ring, dtype=np.float64)
                # split into edges, dropping the ones already drawn by a neighbour
                run = [a[0]]
                for k in range(len(a) - 1):
                    p, q = a[k], a[k + 1]
                    key = tuple(sorted(((round(p[0] % 360, 2), round(p[1], 2)),
                                        (round(q[0] % 360, 2), round(q[1], 2)))))
                    if key in seen:
                        if len(run) > 1:
                            self.con_bounds.append(np.array(run))
                        run = [q]
                    else:
                        seen.add(key)
                        run.append(q)
                if len(run) > 1:
                    self.con_bounds.append(np.array(run))

    def _load_stellarium_lines(self, d: Path) -> bool:
        """Constellation figures of Sky & Telescope (Alan MacRobert), as adopted by the
        IAU charts (Stellarium's 'modern_st' sky culture), given as Hipparcos numbers:
        every line ends exactly on its star."""
        f = d / "stellarium_modern_st.json"
        if not f.exists() or not hasattr(self, "s_hip"):
            return False
        try:
            j = json.load(open(f, encoding="utf-8"))
            pos = {}
            for i in np.nonzero(self.s_hip > 0)[0]:
                h = int(self.s_hip[i])
                if h not in pos or self.s_mag[i] < pos[h][2]:
                    pos[h] = (float(self.s_ra[i]), float(self.s_dec[i]), float(self.s_mag[i]))
            lines, missing = [], 0
            for c in j.get("constellations", []):
                cid = str(c.get("id", "")).split()[-1]
                for poly in CON_FIGURES_PI.get(cid, c.get("lines", [])):
                    pts = []
                    for h in poly:
                        p = pos.get(int(h)) if isinstance(h, (int, float)) else None
                        if p is None:
                            missing += 1
                            if len(pts) > 1:
                                lines.append(np.array(pts))
                            pts = []
                            continue
                        pts.append((p[0], p[1]))
                    if len(pts) > 1:
                        lines.append(np.array(pts))
            if not lines:
                return False
            self.con_lines = lines
            if missing:
                self.log(f"Constellation figures: {missing} stars not found in the catalog")
            return True
        except Exception as e:
            self.log(f"Stellarium constellation figures not used: {e}")
            return False

    def _load_milkyway(self, d: Path):
        self.mw = []
        j = json.load(open(d / "mw.json", encoding="utf-8"))
        feats = sorted(j["features"], key=lambda f: f.get("id", ""))
        for ft in feats:
            rings = []
            g = ft["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for poly in polys:
                for ring in poly:
                    a = np.array(ring, dtype=np.float64)
                    if len(a) >= 4:
                        rings.append(a)
            self.mw.append(rings)

    # ---- nebula outlines (OpenNGC) ---------------------------------------
    def _load_outlines(self, d: Path):
        """self.outlines: list of dicts(name, key, level, kind, ring=Nx2 ra/dec)."""
        self.outlines = []
        try:
            z = np.load(d / "ngc_outlines.npz", allow_pickle=False)
        except Exception:
            return
        names, levels, starts = z["names"], z["levels"], z["starts"]
        pts = z["pts"].astype(np.float64)
        for i in range(len(names)):
            nm = str(names[i])
            m = re.match(r"^(NGC|IC|M)0*(\d+)([A-Za-z]?)$", nm)
            key = f"{m.group(1)}{m.group(2)}{m.group(3)}".upper() if m else None
            kind = K_NEB
            if key in self.dso_index:
                kind = self.dsos[self.dso_index[key]][8]
            if key == "M45":
                kind = K_RFN
            if kind not in (K_NEB, K_RFN, K_DARK):
                kind = K_NEB
            self.outlines.append(dict(name=nm, key=key, level=int(levels[i]), kind=kind,
                                      ring=pts[starts[i]:starts[i + 1]]))

    # ---- Milky Way: integrated starlight of Gaia DR3 -------------------------
    MW_NSIDE = 128

    def milkyway_map(self, gaia_dir=None, build=False):
        """Integrated G-band starlight of Gaia DR3 (HEALPix NESTED, nside 128, ICRS).

        Bundled, built by J. d'Hooghvorst from SASpro's local Gaia library: the
        Milky Way contour levels are tuned to this file's scale, which depends on
        the tiers it was summed over, so it is carried as-is rather than rebuilt
        from whatever tiers this machine happens to have installed."""
        if getattr(self, "_mw_map", None) is not None:
            return self._mw_map
        try:
            self._mw_map = np.load(data_dir() / "milkyway_gaia.npy").astype(np.float64)
        except Exception as e:
            self.log(f"Milky Way map not loaded: {e}")
            self._mw_map = None
        return self._mw_map

    # ---- all-sky H-alpha and dust maps --------------------------------------
    def sky_map(self, key):
        """HEALPix NESTED (nside 512, galactic) float32 array, or None.

        Stored log10-quantised to uint8 (worst case 2.5%, ~0.01 dex): the chart
        only contours these maps, at 80 R and 0.32 dex."""
        cache = getattr(self, "_maps", None)
        if cache is None:
            cache = self._maps = {}
        if key in cache:
            return cache[key]
        arr = None
        try:
            z = np.load(data_dir() / SKY_MAPS[key], allow_pickle=False)
            lo, hi = float(z["log_lo"]), float(z["log_hi"])
            q = z["q"].astype(np.float32)
            arr = np.power(np.float32(10.0), np.float32(lo) + q * np.float32((hi - lo) / 255.0))
        except Exception as e:
            self.log(f"Sky map {key} not loaded: {e}")
            arr = None
        cache[key] = arr
        return arr

    # ---- Gaia DR3 all-sky bright cache ------------------------------------
    def gaia_bright(self, gaia_dir: Path):
        """(ra, dec, gmag, source_id) of every Gaia star brighter than
        GAIA_CACHE_MAG, read once from SASpro's local astrometric library and
        cached on disk."""
        if self._gaia is not None:
            return self._gaia
        cache = cache_dir() / "gaia_bright_cache.npz"
        if cache.exists():
            try:
                z = np.load(cache, allow_pickle=False)
                if float(z["limit"]) >= GAIA_CACHE_MAG and "sid" in z.files:
                    self._gaia = (z["ra"], z["dec"], z["mag"], z["sid"])
                    return self._gaia
            except Exception:
                pass
        src = gaia_dir / "gaia_astro_lt15.sqlite"
        if not src.exists():
            return None
        dlg = None
        if QApplication.instance() is not None:
            dlg = QProgressDialog("Reading bright stars from the local Gaia DR3 library\n"
                                  "(one time only, a few minutes)...", "Cancel", 0, 100)
            dlg.setWindowTitle("Sky Chart")
            dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
            dlg.setMinimumDuration(0)
            dlg.setValue(0)
        t0 = time.time()
        chunks, ids = [], []
        try:
            con = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
            # no INDEXED BY: if the index were ever renamed the hint would make
            # the query fail outright, while SQLite picks it up on its own
            total = con.execute("SELECT count(*) FROM stars "
                                "WHERE phot_g_mean_mag < ?", (GAIA_CACHE_MAG,)).fetchone()[0]
            cur = con.execute("SELECT ra, dec, phot_g_mean_mag, source_id FROM stars "
                              "WHERE phot_g_mean_mag < ?", (GAIA_CACHE_MAG,))
            done = 0
            while True:
                rows = cur.fetchmany(5000)
                if not rows:
                    break
                chunks.append(np.array([r[:3] for r in rows], dtype=np.float64))
                ids.append(np.array([r[3] for r in rows], dtype=np.int64))
                done += len(rows)
                if dlg is not None:
                    dlg.setValue(int(99 * done / max(total, 1)))
                    QApplication.processEvents()
                    if dlg.wasCanceled():
                        con.close()
                        return None
            con.close()
        except Exception as e:
            self.log(f"Gaia bright-star cache failed: {e}")
            return None
        finally:
            if dlg is not None:
                dlg.close()
        if not chunks:
            return None
        a = np.concatenate(chunks)
        ra, dec, mag = a[:, 0], a[:, 1], a[:, 2].astype(np.float32)
        sid = np.concatenate(ids)
        try:
            np.savez_compressed(cache, ra=ra, dec=dec, mag=mag, sid=sid,
                                limit=np.float64(GAIA_CACHE_MAG))
        except Exception:
            pass
        self.log(f"Gaia DR3: {len(ra)} stars to G = {GAIA_CACHE_MAG} cached in {time.time() - t0:.0f} s")
        self._gaia = (ra, dec, mag, sid)
        return self._gaia

    def catalog_box(self, ra_lo, ra_hi, dec_lo, dec_hi, max_mag):
        """(ra, dec, mag) of HYG + local Gaia stars inside an RA/Dec box."""
        def in_ra(r):
            return (r >= ra_lo) & (r <= ra_hi) if ra_lo <= ra_hi else (r >= ra_lo) | (r <= ra_hi)
        m = in_ra(self.s_ra) & (self.s_dec >= dec_lo) & (self.s_dec <= dec_hi) & (self.s_mag <= max_mag)
        ra, dec, mag = [self.s_ra[m]], [self.s_dec[m]], [self.s_mag[m].astype(np.float64)]
        if (dec_hi - dec_lo) < 15:
            g = gaia_box(ra_lo, ra_hi, dec_lo, dec_hi, 7.0, max_mag)
            ra.append(g[0]); dec.append(g[1]); mag.append(g[2])
        return np.concatenate(ra), np.concatenate(dec), np.concatenate(mag)

    # ---- star colours from the Gaia XP spectral library --------------------
    @staticmethod
    def xp_dir():
        """Folder of SASpro's Gaia XP spectral library, or None if not installed."""
        try:
            from setiastro.saspro.gaia_database import get_library_dir
            c = Path(get_library_dir())
            if c.is_dir() and any(c.glob("gaia_xp_*.sqlite")):
                return c
        except Exception:
            pass
        return None

    @staticmethod
    def _xp_mag_range(stem: str):
        """Magnitude tier of an XP file from its name, or None if unknown.
        gaia_xp_lt8 -> (-9, 8); gaia_xp_8_10 -> (8, 10); gaia_xp_100_105 -> (10, 10.5)."""
        def val(t):
            v = float(t)
            return v / 10.0 if v >= 30 else v
        m = re.match(r"gaia_xp_lt(\d+)$", stem)
        if m:
            return -9.0, val(m.group(1))
        m = re.match(r"gaia_xp_(\d+)_(\d+)$", stem)
        if m:
            return val(m.group(1)), val(m.group(2))
        return None

    def xp_colors(self, sid: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """B-V colour index for Gaia sources, synthesised from their XP spectra
        (343 samples, 336-1020 nm, 2 nm step). NaN where no spectrum exists."""
        out = np.full(len(sid), np.nan)
        xdir = self.xp_dir()
        if xdir is None:
            if not self._xp_note:
                self._xp_note = True
                self.log("Gaia XP spectral library not installed: stars without a B-V "
                         "color index are drawn in neutral grey.")
            return out
        known = self._xp_ci
        todo = []
        for k, v in enumerate(sid.tolist()):
            c = known.get(v)
            if c is None:
                todo.append(k)
            else:
                out[k] = c
        if not todo:
            return out
        todo = np.array(todo, dtype=np.int64)
        t0 = time.time()
        found = 0
        for path in sorted(xdir.glob("gaia_xp_*.sqlite")):
            if not len(todo):
                break
            rng = self._xp_mag_range(path.stem)
            sub = todo
            if rng is not None:        # Gaia G vs. the tier limits: allow some slack
                mm = mag[todo]
                sub = todo[(mm >= rng[0] - 0.3) & (mm <= rng[1] + 0.3)]
            if not len(sub):
                continue
            try:
                con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
            except Exception:
                continue
            got_ids, fluxes = [], []
            try:
                # SASpro's astrometric library stores source_id rounded to float64
                # (multiples of 256-1024), while the XP library keeps the exact ids:
                # look within the rounding interval and confirm with the magnitude.
                for k in sub.tolist():
                    v = int(sid[k])
                    m_ = float(mag[k])
                    half = int(np.spacing(float(v))) // 2
                    rows = con.execute("SELECT source_id, phot_g_mean_mag FROM sources "
                                       "WHERE source_id BETWEEN ? AND ?", (v - half, v + half)).fetchall()
                    if not rows:
                        continue
                    best = min(rows, key=lambda r_: abs((r_[1] if r_[1] is not None else m_) - m_))
                    if best[1] is not None and abs(float(best[1]) - m_) > 0.3:
                        continue
                    row = con.execute("SELECT flux_compressed FROM spectra WHERE source_id = ?",
                                      (best[0],)).fetchone()
                    if row is None:
                        continue
                    try:
                        f = np.frombuffer(zlib.decompress(row[0]), dtype=np.float32)
                    except Exception:
                        continue
                    if len(f) == 343:
                        got_ids.append(v)              # keyed by the id the chart uses
                        fluxes.append(f)
            except Exception as e:
                self.log(f"XP colour query failed on {path.name}: {e}")
            finally:
                con.close()
            if not got_ids:
                continue
            F = np.vstack(fluxes).astype(np.float64)
            # mean flux density in B (400-480 nm) and V (510-590 nm); zero point
            # set so that an A0V (Vega-like) spectrum gives B-V = 0
            fb = F[:, 32:73].mean(axis=1)
            fv = F[:, 87:128].mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ci = np.where((fb > 0) & (fv > 0), -2.5 * np.log10(fb / fv) + 0.60, np.nan)
            for s_id, c in zip(got_ids, ci.tolist()):
                known[s_id] = c
            found += len(got_ids)
            gs = set(got_ids)
            todo = np.array([k for k in todo.tolist() if int(sid[k]) not in gs], dtype=np.int64)
        for k, v in enumerate(sid.tolist()):
            c = known.get(v)
            if c is not None:
                out[k] = c
        for k in todo.tolist():            # remember the misses too
            known.setdefault(int(sid[k]), float("nan"))
        self.log(f"Gaia XP: colours for {found} stars in {time.time() - t0:.1f} s")
        return out

    # ---- online Gaia colours (BP-RP), cached on disk -----------------------
    # Stars without a local XP spectrum (fainter than the installed XP groups, or no
    # spectrum in Gaia) get their colour from the Gaia archive, fetched per HEALPix
    # level-5 tile (source_id >> 49, 3.4 square degrees) and kept forever in a local
    # cache, so every piece of sky is downloaded once.
    BPRP_SHIFT = 49
    BPRP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

    def _bprp_con(self):
        import threading
        if not hasattr(self, "_bprp_lock"):
            self._bprp_lock = threading.Lock()
            self._bprp = {}
            self._bprp_tiles = None
            self._bprp_loaded = set()
        con = sqlite3.connect(str(cache_dir() / "gaia_bprp_cache.sqlite"), timeout=30)
        con.execute("CREATE TABLE IF NOT EXISTS tiles (tile INTEGER PRIMARY KEY, maglim REAL)")
        con.execute("CREATE TABLE IF NOT EXISTS col (sid INTEGER PRIMARY KEY, bprp REAL)")
        return con

    @staticmethod
    def bprp_to_bv(x):
        """Gaia BP-RP -> B-V, fitted on the XP-derived B-V of 2535 stars (rms 0.13)."""
        x = np.clip(np.asarray(x, dtype=np.float64), -0.5, 4.0)
        return -0.1275 * x * x + 1.1821 * x - 0.2624

    def _bprp_tile_table(self):
        if getattr(self, "_bprp_tiles", None) is None:
            con = self._bprp_con()
            try:
                self._bprp_tiles = {int(t): float(m) for t, m in con.execute("SELECT tile, maglim FROM tiles")}
            finally:
                con.close()
        return self._bprp_tiles

    def bprp_colors(self, sid) -> np.ndarray:
        """B-V of Gaia stars from the cached online BP-RP (NaN where not cached).
        Ids are compared after float64 rounding, as SASpro's library stores them."""
        sid = np.asarray(sid, dtype=np.int64)
        out = np.full(len(sid), np.nan)
        if not len(sid):
            return out
        self._bprp_tile_table()
        tiles = np.unique(sid >> self.BPRP_SHIFT)
        need = [int(t) for t in tiles if int(t) in self._bprp_tiles and int(t) not in self._bprp_loaded]
        if need:
            con = self._bprp_con()
            try:
                for t in need:
                    a, b = t << self.BPRP_SHIFT, ((t + 1) << self.BPRP_SHIFT) - 1
                    for k, v in con.execute("SELECT sid, bprp FROM col WHERE sid BETWEEN ? AND ?", (a, b)):
                        self._bprp[int(k)] = v
                    self._bprp_loaded.add(t)
            finally:
                con.close()
        get = self._bprp.get
        vals = np.array([get(int(float(int(v))), np.nan) for v in sid.tolist()], dtype=np.float64)
        ok = ~np.isnan(vals)
        out[ok] = self.bprp_to_bv(vals[ok])
        return out

    def bprp_missing(self, sid, mag, min_stars=10):
        """{tile: magnitude needed} for tiles not cached deep enough."""
        sid = np.asarray(sid, dtype=np.int64)
        if not len(sid):
            return {}
        tab = self._bprp_tile_table()
        tiles = sid >> self.BPRP_SHIFT
        out = {}
        for t in np.unique(tiles):
            sel = tiles == t
            if int(sel.sum()) < min_stars:
                continue
            m = float(np.max(np.asarray(mag)[sel])) + 0.15
            if tab.get(int(t), -99.0) < m:
                out[int(t)] = min(m, 17.0)
        return out

    def fetch_bprp(self, want: dict, timeout=30.0) -> int:
        """Download BP-RP for the tiles in `want` ({tile: maglim}). Returns rows stored."""
        import urllib.request
        import urllib.parse
        import csv as _csv
        import io as _io
        if not want:
            return 0
        self._bprp_con().close()
        with self._bprp_lock:
            tab = self._bprp_tile_table()
            want = {t: m for t, m in want.items() if tab.get(t, -99.0) < m}
            if not want:
                return 0
            items = sorted(want.items())
            total = 0
            for a in range(0, len(items), 12):
                part = items[a:a + 12]
                maglim = max(m for _t, m in part)
                cond = " OR ".join(f"source_id BETWEEN {t << self.BPRP_SHIFT} AND "
                                   f"{((t + 1) << self.BPRP_SHIFT) - 1}" for t, _m in part)
                q = (f"SELECT source_id, bp_rp FROM gaiadr3.gaia_source WHERE ({cond}) "
                     f"AND phot_g_mean_mag < {maglim:.2f} AND bp_rp IS NOT NULL")
                body = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL",
                                               "FORMAT": "csv", "QUERY": q}).encode()
                req = urllib.request.Request(self.BPRP_URL, data=body,
                                             headers={"User-Agent": "SASpro-SkyChart"})
                txt = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
                rows = []
                for r_ in list(_csv.reader(_io.StringIO(txt)))[1:]:
                    try:
                        rows.append((int(float(int(r_[0]))), float(r_[1])))
                    except (ValueError, IndexError):
                        continue
                con = self._bprp_con()
                try:
                    con.executemany("INSERT OR REPLACE INTO col VALUES (?, ?)", rows)
                    con.executemany("INSERT OR REPLACE INTO tiles VALUES (?, ?)", part)
                    con.commit()
                finally:
                    con.close()
                for k, v in rows:
                    self._bprp[k] = v
                for t, m in part:
                    self._bprp_tiles[t] = m
                    self._bprp_loaded.add(t)
                total += len(rows)
            return total

    # ---- name resolution -------------------------------------------------
    def _build_name_index(self):
        """self.names: normalized key -> (ra, dec, title). Built on first search."""
        idx = {}

        def put(key, ra, dec, title, force=False):
            if not key or ra != ra or dec != dec:
                return
            if force or key not in idx:
                idx[key] = (float(ra) % 360.0, float(dec), title)

        def put_name(name, ra, dec, title):
            n = norm_name(name)
            put(n, ra, dec, title)
            s = strip_kind(n)
            if s != n and len(s) >= 3:
                put(s, ra, dec, title)

        # OpenNGC: designations and common names
        for r in self.dsos:
            common = [c.strip() for c in r[9].split(",") if c.strip()]
            title = common[0] if common else r[1]
            for a in r[10]:
                put(canon_id(a), r[2], r[3], title)
            for c in common:
                put_name(c, r[2], r[3], c)
        # SASpro's bundled catalog (Sh2, LBN, LDN, Abell, PN G, Caldwell, Arp...)
        try:
            from setiastro.saspro.resources import get_data_path
            path = Path(get_data_path("data/catalogs/celestial_catalog.csv"))
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                for r in csv.DictReader(f):
                    ra, dec = _f(r.get("RA", "")), _f(r.get("Dec", ""))
                    nm = re.sub(r"\s+", " ", (r.get("Name") or "").strip())
                    alt = (r.get("Alt Name") or "").strip()
                    title = nm
                    m = re.match(r"^(?:(?:NGC|IC)\s*\d+\s*)?([A-Za-z].*)$", alt)
                    if m and re.search(r"[a-z]{3}", m.group(1)):
                        title = m.group(1).split(",")[0].strip()
                    put(canon_id(nm), ra, dec, title)
                    for part in re.split(r",", alt):
                        part = part.strip()
                        if not part or part.lower() == "none":
                            continue
                        put(canon_id(part), ra, dec, title)
                        mm = re.match(r"^(?:(?:NGC|IC)\s*\d+\s*)?([A-Za-z][A-Za-z' ?/\-()]+)$", part)
                        if mm and re.search(r"[a-z]{3}", mm.group(1)):
                            for piece in re.split(r"[/(]", mm.group(1)):
                                piece = piece.strip(" )?")
                                if len(piece) >= 4:
                                    put_name(piece, ra, dec, piece)
        except Exception as e:
            self.log(f"SASpro catalog not indexed for search: {e}")
        # constellations and named stars
        for cid, (nm, ra, dec, _rk) in self.con_names.items():
            put(norm_name(nm), ra, dec, nm, force=True)
            put(norm_name(cid), ra, dec, nm)
        for i in np.nonzero(self.s_name != "")[0]:
            nm = str(self.s_name[i])
            put(norm_name(nm), self.s_ra[i], self.s_dec[i], nm)
        # popular nicknames that the catalogs above do not carry
        for nick, target in NICKNAMES.items():
            if isinstance(target, tuple):
                ra, dec = target
                put(norm_name(nick), ra, dec, nick, force=True)
            else:
                hit = idx.get(canon_id(target))
                if hit is not None:
                    put(norm_name(nick), hit[0], hit[1], nick, force=True)
                    s = strip_kind(norm_name(nick))
                    if s and len(s) >= 3:
                        put(s, hit[0], hit[1], nick)
        self.names = idx
        self._name_keys = sorted(idx.keys())

    def resolve(self, text: str):
        """Return (ra_deg, dec_deg, title) for an object name, or None."""
        t = text.strip()
        if not t:
            return None
        if getattr(self, "names", None) is None:
            self._build_name_index()
        idx = self.names
        for key in (canon_id(t), norm_name(t), strip_kind(norm_name(t))):
            if key and key in idx:
                return idx[key]
        # partial match on names ("Horsehead" -> "Horsehead Nebula")
        q = strip_kind(norm_name(t))
        if len(q) >= 4:
            starts = [k for k in self._name_keys if k.startswith(q) and not k[:1].isdigit()]
            if starts:
                return idx[min(starts, key=len)]
            inside = [k for k in self._name_keys if q in k and not k[:1].isdigit()]
            if inside:
                return idx[min(inside, key=len)]
        return None

    def search_names(self):
        """Every name the search understands, for suggestions while typing."""
        out = []
        seen = set()

        def add(t):
            if t and t not in seen:
                seen.add(t)
                out.append(t)

        for cid, (nm, _ra, _dec, _rk) in sorted(self.con_names.items(), key=lambda kv: kv[1][0]):
            add(f"{nm} \u2014 constellation")
        for r in self.dsos:
            layer, label = r[0], r[1]
            desig = {"Messier": label.replace("M", "M ", 1), "NGC": f"NGC {label}"}.get(layer, label)
            commons = [c.strip() for c in r[9].split(",") if c.strip()]
            add(f"{desig} \u2014 {commons[0]}" if commons else desig)
            for c in commons:
                add(f"{c} \u2014 {desig}")
        known = {norm_name(t.split(" \u2014 ")[0]) for t in out}
        for nick, target in NICKNAMES.items():
            if norm_name(nick) in known:
                continue
            add(f"{nick.strip()} \u2014 {target}" if isinstance(target, str) else nick.strip())
        names = sorted({str(n) for n in self.s_name if str(n)})
        for nm in names:
            add(f"{nm} \u2014 star")
        for cat, rows in self.extra.items():
            for r in rows:
                lab = re.sub(r"^SH2-", "Sh2-", r[1])
                lab = re.sub(r"^(LBN|LDN)\s*", r"\1 ", lab)
                add(lab)
        return out

    def _constellation_id(self, text):
        t = text.strip().lower()
        for cid, (nm, _ra, _dec, _rk) in self.con_names.items():
            if t in (nm.lower(), cid.lower()):
                return cid[:3]
        for cid, (nm, _ra, _dec, _rk) in self.con_names.items():
            if nm.lower().startswith(t + " "):                   # "Serpens" -> both parts
                return cid[:3]
        return None

    def constellation_view(self, cid, aspect, projection):
        """Centre and field that frame the whole constellation (its borders)."""
        rings = self.con_polys.get(cid) or []
        if not rings:
            return None
        ra, dec = [], []
        for ring in rings:
            r, d = ChartRenderer._densify(ring, 0.5)
            ra.append(r)
            dec.append(d)
        ra, dec = np.concatenate(ra), np.concatenate(dec)
        v = np.vstack([np.cos(dec * D2R) * np.cos(ra * D2R), np.cos(dec * D2R) * np.sin(ra * D2R),
                       np.sin(dec * D2R)]).mean(axis=1)
        ra0 = math.degrees(math.atan2(v[1], v[0])) % 360.0
        dec0 = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        kind = projection if projection in _PROJ_TMAX else "Stereographic"
        for _ in range(4):                      # centre on the bounding box of the projection
            pr = Projection(ra0, dec0, kind)
            pr.scale, pr.cx, pr.cy = 1.0, 0.0, 0.0
            x, y, th = pr.project(ra, dec)
            ok = th < pr.tmax
            xc, yc = (x[ok].max() + x[ok].min()) / 2, (y[ok].max() + y[ok].min()) / 2
            r1, d1 = pr.inverse(np.array([xc]), np.array([yc]))
            if r1[0] != r1[0]:
                break
            ra0, dec0 = float(r1[0]), float(d1[0])
        pr = Projection(ra0, dec0, kind)
        pr.scale, pr.cx, pr.cy = 1.0, 0.0, 0.0
        x, y, th = pr.project(ra, dec)
        ex, ey = (x.max() - x.min()) / 2, (y.max() - y.min()) / 2
        hw = max(ex, ey * aspect) * 1.08
        t = float(pr.Rinv(np.array([hw]))[0])
        fov = 2.0 * math.degrees(t) if t == t else _PROJ_FOVMAX.get(kind, 240.0)
        return ra0, dec0, min(max(fov, 1.0), _PROJ_FOVMAX.get(kind, 240.0))

    def view_for(self, text, aspect, projection):
        """(ra, dec, title, fov or None) for a search text, or None."""
        cid = self._constellation_id(text)
        if cid is not None:
            v = self.constellation_view(cid, aspect, projection)
            if v is not None:
                nm = self.con_names[cid][0]
                if (cid + "2") in self.con_names:
                    nm = nm.split(" ")[0]
                return v[0], v[1], nm, v[2]
        r = self.resolve(text)
        if r is None:
            return None
        return r[0], r[1], r[2], None

    def object_size(self, text):
        """Largest dimension of a catalog object in degrees (0 when unknown)."""
        u = canon_id(text)
        i = self.dso_index.get(u) if u else None
        if i is None:
            i = self.dso_index.get(text.strip().upper()) or self.dso_index.get(re.sub(r"\s+", "", text).upper())
        if i is None:                                   # popular names
            nt = norm_name(text)
            for nick, target in NICKNAMES.items():
                if isinstance(target, str) and nt in (norm_name(nick), strip_kind(norm_name(nick))):
                    i = self.dso_index.get(canon_id(target)) or self.dso_index.get(target.upper())
                    break
        if i is None:
            return 0.0
        maj = self.dsos[i][5]
        return float(maj) / 60.0 if maj == maj else 0.0

    def suggestions(self, text: str, n=6):
        import difflib
        if getattr(self, "names", None) is None:
            self._build_name_index()
        q = norm_name(text)
        keys = difflib.get_close_matches(q, self._name_keys, n=n * 3, cutoff=0.72)
        out = []
        for k in keys:
            title = self.names[k][2]
            if title not in out:
                out.append(title)
        return out[:n]

    def constellation_at(self, ra, dec) -> str:
        """Approximate constellation: that of the nearest naked-eye star."""
        m = self.s_mag < 6.5
        ra_s, dec_s = self.s_ra[m] * D2R, self.s_dec[m] * D2R
        c = (math.sin(dec * D2R) * np.sin(dec_s) +
             math.cos(dec * D2R) * np.cos(dec_s) * np.cos(ra_s - ra * D2R))
        cid = str(self.s_con[m][int(np.argmax(c))])
        return self.con_names.get(cid, (cid, 0, 0, 0))[0]


def norm_name(s: str) -> str:
    """'Thor's Helmet' -> 'thorshelmet'; 'Eagle Nebula' -> 'eaglenebula'."""
    s = s.lower().replace("&", "and")
    s = re.sub(r"^the\s+", "", s.strip())
    return re.sub(r"[^a-z0-9+\-]", "", s).replace("-", "")


def strip_kind(n: str) -> str:
    """Remove generic words: 'horseheadnebula' -> 'horsehead'."""
    return re.sub(r"(nebula|nebulae|galaxy|cluster|starcluster|complex|region|nebulosity)$", "", n)


_ID_PATTERNS = [
    (r"(?:messier|m)\s*0*(\d+)", "M{}"),
    (r"ngc\s*0*(\d+)\s*([a-z]?)", "NGC{}{}"),
    (r"ic\s*0*(\d+)\s*([a-z]?)", "IC{}{}"),
    (r"(?:sh\s*2|sharpless\s*2?|sh)\s*0*(\d+)", "SH2-{}"),
    (r"lbn\s*0*(\d+)", "LBN{}"),
    (r"ldn\s*0*(\d+)", "LDN{}"),
    (r"(?:barnard|b)\s*0*(\d+)", "B{}"),
    (r"(?:caldwell|c)\s*0*(\d+)", "C{}"),
    (r"(?:melotte|mel)\s*0*(\d+)", "MEL{}"),
    (r"(?:collinder|cr|cl)\s*0*(\d+)", "CR{}"),
    (r"abell\s*0*(\d+)", "ABELL{}"),
    (r"arp\s*0*(\d+)\s*([a-z]?)", "ARP{}{}"),
    (r"(?:vdb|van\s*den\s*bergh)\s*0*(\d+)", "VDB{}"),
    (r"(?:ugc)\s*0*(\d+)", "UGC{}"),
    (r"(?:pgc)\s*0*(\d+)", "PGC{}"),
    (r"(?:hcg)\s*0*(\d+)", "HCG{}"),
]


def canon_id(s: str) -> str:
    """Catalog designation in canonical form ('Sh 2-155' -> 'SH2-155'), or ''."""
    t = re.sub(r"[\s\-_.]+", " ", s.strip().lower())
    for pat, fmt in _ID_PATTERNS:
        m = re.fullmatch(pat, t) or re.fullmatch(pat, t.replace(" ", ""))
        if m:
            return fmt.format(*[g.upper() if g else "" for g in m.groups()])
    return ""


# Popular names that OpenNGC / SASpro catalogs do not list (target id or (ra, dec))
NICKNAMES = {
    "Heart Nebula": "IC1805", "Soul Nebula": "IC1848", "Embryo Nebula": "IC1848",
    "Fish Head Nebula": "IC1795", "Rosette Nebula": "NGC2237", "Pacman Nebula": "NGC281",
    "Elephant's Trunk Nebula": "IC1396", "Elephant Trunk Nebula": "IC1396", "Trifid Nebula": "M20", "Dumbbell Nebula": "M27",
    "Little Dumbbell Nebula": "M76", "Cat's Eye Nebula": "NGC6543", "Wizard Nebula": "NGC7380",
    "Tadpoles Nebula": "IC410", "Flaming Star Nebula": "IC405", "Jellyfish Nebula": "IC443",
    "Medusa Nebula": (112.2613, 13.2467), "Thor's Helmet": "NGC2359", "Seagull Nebula": "IC2177",
    "Running Chicken Nebula": "IC2944", "Carina Nebula": "NGC3372", "Eta Carinae Nebula": "NGC3372",
    "Cone Nebula": "NGC2264", "Christmas Tree Cluster": "NGC2264", "Fox Fur Nebula": "SH2-273",
    "Whirlpool Galaxy": "M51", "Simeis 147": "SH2-240", "Spaghetti Nebula": "SH2-240",
    "Barnard's Loop": "SH2-276", "Bow-Tie Nebula": "NGC40", "Ghost of Cassiopeia": "IC63",
    "Witch Head Nebula": "IC2118", "Horsehead Nebula": "B33", "Cave Nebula": "SH2-155",
    "Bubble Nebula": "NGC7635", "California Nebula": "NGC1499", "North America Nebula": "NGC7000",
    "Pelican Nebula": "IC5070", "Veil Nebula": "NGC6992", "Eastern Veil Nebula": "NGC6992",
    "Western Veil Nebula": "NGC6960", "Witch's Broom Nebula": "NGC6960",
    "Pickering's Triangle": "NGC6979", "Crescent Nebula": "NGC6888", "Iris Nebula": "NGC7023",
    "Cocoon Nebula": "IC5146", "Lagoon Nebula": "M8", "Eagle Nebula": "M16",
    "Pillars of Creation": "M16", "Omega Nebula": "M17", "Swan Nebula": "M17",
    "Orion Nebula": "M42", "Great Orion Nebula": "M42", "Running Man Nebula": "NGC1977",
    "Flame Nebula": "NGC2024", "Monkey Head Nebula": "NGC2174", "Crab Nebula": "M1",
    "Helix Nebula": "NGC7293", "Ring Nebula": "M57", "Owl Nebula": "M97",
    "Eskimo Nebula": "NGC2392", "Clown Face Nebula": "NGC2392", "Ghost of Jupiter": "NGC3242",
    "Saturn Nebula": "NGC7009", "Blue Snowball Nebula": "NGC7662", "Double Cluster": "NGC869",
    "Pleiades": "M45", "Seven Sisters": "M45", "Hyades": "C41", "Beehive Cluster": "M44",
    "Andromeda Galaxy": "M31", "Triangulum Galaxy": "M33", "Bode's Galaxy": "M81",
    "Cigar Galaxy": "M82", "Sombrero Galaxy": "M104", "Pinwheel Galaxy": "M101",
    "Sunflower Galaxy": "M63", "Black Eye Galaxy": "M64", "Needle Galaxy": "NGC4565",
    "Sculptor Galaxy": "NGC253", "Silver Dollar Galaxy": "NGC253", "Centaurus A": "NGC5128",
    "Statue of Liberty Nebula": "NGC3576", "Prawn Nebula": "IC4628", "Cat's Paw Nebula": "NGC6334",
    "War and Peace Nebula": "NGC6357", "Lobster Nebula": "NGC6357", "Rho Ophiuchi": "IC4604",
    "Blue Horsehead Nebula": "IC4592", "Squid Nebula": (317.95, 59.98), "Lion Nebula": "SH2-132",
    "Spider Nebula": "IC417", "Fly Nebula": "NGC1931", "Dolphin Nebula": "SH2-308",
    "Angelfish Nebula": "SH2-264", "Lambda Orionis Ring": "SH2-264", "Gamma Cygni Nebula": "IC1318",
    "Butterfly Nebula": "IC1318", "Sadr Region": "IC1318", "Wolf's Cave": "LDN1622",
    "Boogeyman Nebula": "LDN1622", "Dark Shark Nebula": "LDN1235",
    "Snake Nebula": (260.875, -23.633), "Pipe Nebula ": (263.25, -25.97), "Coalsack": "C99", "Leo Triplet": "M66", "Markarian's Chain": "M86",
    "Wild Duck Cluster": "M11", "Hercules Cluster": "M13", "Omega Centauri": "NGC5139",
    "47 Tucanae": "NGC104", "Large Magellanic Cloud": (80.894, -69.756),
    "Small Magellanic Cloud": (13.187, -72.829), "Tarantula Nebula": "NGC2070",
    "Shrimp Nebula": "IC4628", "Heart and Soul": "IC1805", "Bat Nebula": "NGC6995",
    "Fireworks Galaxy": "NGC6946", "Stephan's Quintet": "NGC7317", "Sharpless 2-155": "SH2-155",
}


# cds.unistra.fr currently redirects Sesame to a mirror that does not answer,
# so the Harvard mirrors go first; CDS is kept as the last resort.
_DATA = None


def get_data(log=print) -> ChartData:
    global _DATA
    if _DATA is None:
        _DATA = ChartData(log)
    return _DATA


# -----------------------------------------------------------------------------
# Projection (all azimuthal: only the radial function differs)
# -----------------------------------------------------------------------------
PROJECTIONS = ["Stereographic", "Gnomonic", "Orthographic", "Azimuthal equidistant", "Lambert equal-area"]
_PROJ_TMAX = {"Stereographic": 150.0, "Gnomonic": 75.0, "Orthographic": 85.0,
              "Azimuthal equidistant": 172.0, "Lambert equal-area": 168.0}
_PROJ_FOVMAX = {"Stereographic": 240.0, "Gnomonic": 120.0, "Orthographic": 160.0,
                "Azimuthal equidistant": 300.0, "Lambert equal-area": 300.0}


class Projection:
    def __init__(self, ra0, dec0, kind="Stereographic", rotation=0.0, mirrored=False):
        self.ra0, self.dec0 = ra0 * D2R, dec0 * D2R
        self.kind = kind if kind in _PROJ_TMAX else "Stereographic"
        self.tmax = _PROJ_TMAX[self.kind] * D2R
        self.rot = rotation * D2R
        self.mir = bool(mirrored)
        self.scale = 1.0
        self.cx = self.cy = 0.0
        self._rmax = float(self._R0(np.array([self.tmax]))[0])
        e = 1e-5
        self._slope = float((self._R0(np.array([self.tmax]))[0] - self._R0(np.array([self.tmax - e]))[0]) / e)

    def _R0(self, t):
        k = self.kind
        if k == "Gnomonic":
            return np.tan(t)
        if k == "Orthographic":
            return np.sin(t)
        if k == "Azimuthal equidistant":
            return t
        if k == "Lambert equal-area":
            return 2.0 * np.sin(t / 2.0)
        return 2.0 * np.tan(t / 2.0)

    def R(self, t):
        """Radial function, continued linearly beyond tmax so that the mapping
        stays monotonic (keeps polygon topology for far-away geometry)."""
        t = np.asarray(t, dtype=np.float64)
        tc = np.minimum(t, self.tmax)
        return self._R0(tc) + self._slope * (t - tc)

    def Rinv(self, r):
        r = np.asarray(r, dtype=np.float64)
        k = self.kind
        with np.errstate(invalid="ignore"):
            if k == "Gnomonic":
                t = np.arctan(r)
            elif k == "Orthographic":
                t = np.arcsin(np.clip(r, -1, 1))
            elif k == "Azimuthal equidistant":
                t = r
            elif k == "Lambert equal-area":
                t = 2.0 * np.arcsin(np.clip(r / 2.0, -1, 1))
            else:
                t = 2.0 * np.arctan(r / 2.0)
        return np.where(r > self._rmax, np.nan, t)

    def fit(self, fov_deg, width_mm, cx, cy):
        self.scale = (width_mm / 2.0) / float(self._R0(np.array([fov_deg / 2.0 * D2R]))[0])
        self.cx, self.cy = cx, cy

    @property
    def mm_per_deg(self):
        return self.scale * D2R

    def project(self, ra_deg, dec_deg):
        """-> x_mm, y_mm, theta(rad)"""
        a = np.asarray(ra_deg, dtype=np.float64) * D2R
        d = np.asarray(dec_deg, dtype=np.float64) * D2R
        da = a - self.ra0
        sd, cd = np.sin(d), np.cos(d)
        s0, c0 = math.sin(self.dec0), math.cos(self.dec0)
        cda = np.cos(da)
        cost = s0 * sd + c0 * cd * cda
        xe = cd * np.sin(da)
        yn = c0 * sd - s0 * cd * cda
        h = np.hypot(xe, yn)
        theta = np.arctan2(h, cost)
        hs = np.where(h > 1e-15, h, 1.0)
        r = self.R(theta)
        X = (r * xe / hs) * (1.0 if self.mir else -1.0)
        Y = r * yn / hs
        cr, sr = math.cos(self.rot), math.sin(self.rot)
        Xr = X * cr - Y * sr
        Yr = X * sr + Y * cr
        return self.cx + self.scale * Xr, self.cy - self.scale * Yr, theta

    def inverse(self, x_mm, y_mm):
        """-> ra_deg, dec_deg (nan outside the projection domain)"""
        Xr = (np.asarray(x_mm, dtype=np.float64) - self.cx) / self.scale
        Yr = -(np.asarray(y_mm, dtype=np.float64) - self.cy) / self.scale
        cr, sr = math.cos(self.rot), math.sin(self.rot)
        X = Xr * cr + Yr * sr
        Y = -Xr * sr + Yr * cr
        r = np.hypot(X, Y)
        t = self.Rinv(r)
        rs = np.where(r > 1e-15, r, 1.0)
        ue = (X / rs) * (1.0 if self.mir else -1.0)
        un = Y / rs
        s0, c0 = math.sin(self.dec0), math.cos(self.dec0)
        st, ct = np.sin(t), np.cos(t)
        dec = np.arcsin(np.clip(ct * s0 + st * c0 * un, -1, 1))
        da = np.arctan2(st * ue, ct * c0 - st * s0 * un)
        return np.mod((self.ra0 + da) * R2D, 360.0), dec * R2D


def sky_offset(ra, dec, pa_deg, dist_deg):
    """Destination point from (ra,dec) at position angle pa (N through E)."""
    a, d = np.asarray(ra) * D2R, np.asarray(dec) * D2R
    pa, s = np.asarray(pa_deg) * D2R, np.asarray(dist_deg) * D2R
    d1 = np.arcsin(np.clip(np.sin(d) * np.cos(s) + np.cos(d) * np.sin(s) * np.cos(pa), -1, 1))
    a1 = a + np.arctan2(np.sin(pa) * np.sin(s) * np.cos(d), np.cos(s) - np.sin(d) * np.sin(d1))
    return a1 * R2D, d1 * R2D


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def clip_ring_rect(pts: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    """Sutherland-Hodgman clipping of a closed ring (Nx2) against a rectangle."""
    for axis, val, keep_ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        n = len(pts)
        if n < 3:
            return pts[:0]
        c = pts[:, axis]
        inside = (c >= val) if keep_ge else (c <= val)
        if inside.all():
            continue
        if not inside.any():
            return pts[:0]
        nxt = np.roll(pts, -1, axis=0)
        in_n = np.roll(inside, -1)
        cross = inside != in_n
        idx = np.arange(n)
        denom = (nxt[:, axis] - c)
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        t = (val - c) / denom
        inter = pts + (nxt - pts) * t[:, None]
        inter[:, axis] = val
        keys = np.concatenate([2 * idx[inside], 2 * idx[cross] + 1])
        allp = np.concatenate([pts[inside], inter[cross]])
        pts = allp[np.argsort(keys, kind="stable")]
    return pts


def _spread_bits(v):
    v = v.astype(np.int64)
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def healpix_nest(nside, lon_rad, lat_rad):
    """HEALPix NESTED pixel index of (lon, lat) in radians (no healpy needed)."""
    z = np.sin(lat_rad)
    za = np.abs(z)
    tt = np.mod(lon_rad, 2.0 * np.pi) / (0.5 * np.pi)          # [0, 4)
    face = np.zeros(z.shape, dtype=np.int64)
    ix = np.zeros(z.shape, dtype=np.int64)
    iy = np.zeros(z.shape, dtype=np.int64)
    eq = za <= 2.0 / 3.0
    # equatorial belt
    t1 = nside * (0.5 + tt[eq])
    t2 = nside * z[eq] * 0.75
    jp = (t1 - t2).astype(np.int64)
    jm = (t1 + t2).astype(np.int64)
    ifp, ifm = jp // nside, jm // nside
    f = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    face[eq] = f
    ix[eq] = jm & (nside - 1)
    iy[eq] = nside - (jp & (nside - 1)) - 1
    # polar caps
    pc = ~eq
    ntt = np.minimum(3, tt[pc].astype(np.int64))
    tp = tt[pc] - ntt
    tmp = nside * np.sqrt(3.0 * (1.0 - za[pc]))
    jp = np.minimum((tp * tmp).astype(np.int64), nside - 1)
    jm = np.minimum(((1.0 - tp) * tmp).astype(np.int64), nside - 1)
    north = z[pc] >= 0
    face[pc] = np.where(north, ntt, ntt + 8)
    ix[pc] = np.where(north, nside - jm - 1, jp)
    iy[pc] = np.where(north, nside - jp - 1, jm)
    return face * nside * nside + _spread_bits(ix) + (_spread_bits(iy) << 1)


# equatorial (ICRS) -> galactic rotation (transpose of galactic -> equatorial)
_EQ2GAL = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                    [-0.8734370902, -0.4448296300, -0.1980763734],
                    [-0.4838350155, +0.7469822445, +0.4559837762]]).T


def equ_to_gal(ra_deg, dec_deg):
    a, d = np.asarray(ra_deg) * D2R, np.asarray(dec_deg) * D2R
    v = np.vstack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    g = _EQ2GAL @ v
    return np.arctan2(g[1], g[0]), np.arcsin(np.clip(g[2], -1, 1))


def winding_number(px, py, rings) -> int:
    """Winding number of a set of rings (list of Nx2 arrays) around a point."""
    wn = 0
    for r in rings:
        x, y = r[:, 0], r[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        up = (y <= py) & (yn > py)
        dn = (y > py) & (yn <= py)
        if not (up.any() or dn.any()):
            continue
        cross = (xn - x) * (py - y) - (px - x) * (yn - y)
        wn += int(np.count_nonzero(up & (cross > 0))) - int(np.count_nonzero(dn & (cross < 0)))
    return wn


def _clip_seg(xa, ya, xb, yb, x0, y0, x1, y1):
    """Liang-Barsky. Returns (xa,ya,xb,yb, ta, tb) clipped or None."""
    dx, dy = xb - xa, yb - ya
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, xa - x0), (dx, x1 - xa), (-dy, ya - y0), (dy, y1 - ya)):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return None
                if t < t1:
                    t1 = t
    return xa + t0 * dx, ya + t0 * dy, xa + t1 * dx, ya + t1 * dy, t0, t1


def polyline_path(x, y, ok, rect, crossings=None, tag=None):
    """Build a QPainterPath of a polyline clipped to rect=(x0,y0,x1,y1).
    `ok` flags usable vertices. Edge crossings are appended to `crossings`
    as (tag, x, y)."""
    x0, y0, x1, y1 = rect
    path = QPainterPath()
    n = len(x)
    if n < 2:
        return path
    a, b = slice(0, n - 1), slice(1, n)
    cand = ok[a] & ok[b] & ~(((x[a] < x0) & (x[b] < x0)) | ((x[a] > x1) & (x[b] > x1)) |
                             ((y[a] < y0) & (y[b] < y0)) | ((y[a] > y1) & (y[b] > y1)))
    last = None
    for i in np.nonzero(cand)[0]:
        c = _clip_seg(float(x[i]), float(y[i]), float(x[i + 1]), float(y[i + 1]), x0, y0, x1, y1)
        if c is None:
            continue
        xa, ya, xb, yb, ta, tb = c
        if last is None or abs(last[0] - xa) > 1e-6 or abs(last[1] - ya) > 1e-6:
            path.moveTo(xa, ya)
        path.lineTo(xb, yb)
        last = (xb, yb)
        if crossings is not None:
            if ta > 0.0:
                crossings.append((tag, xa, ya))
            if tb < 1.0:
                crossings.append((tag, xb, yb))
    return path


# -----------------------------------------------------------------------------
# Text engine: text as scalable outlines (or real text when requested)
# -----------------------------------------------------------------------------
class TextEngine:
    EM = 100.0

    def __init__(self, family: str, as_paths: bool = True):
        self.family = family
        self.as_paths = as_paths
        self._fonts, self._cache = {}, {}

    def _font(self, bold, italic):
        k = (bold, italic)
        f = self._fonts.get(k)
        if f is None:
            f = QFont(self.family)
            f.setPixelSize(int(self.EM))
            f.setBold(bold)
            f.setItalic(italic)
            f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
            self._fonts[k] = (f, QFontMetricsF(f))
            f = self._fonts[k]
        return f

    def _piece(self, text, bold, italic):
        k = (text, bold, italic)
        v = self._cache.get(k)
        if v is None:
            f, fm = self._font(bold, italic)
            p = QPainterPath()
            p.addText(0.0, 0.0, f, text)
            v = (p, fm.horizontalAdvance(text))
            self._cache[k] = v
        return v

    @staticmethod
    def _runs(text):
        return text if isinstance(text, (list, tuple)) else [(text, False)]

    def width(self, text, size, bold=False, italic=False):
        w = 0.0
        for s, sup in self._runs(text):
            w += self._piece(s, bold, italic)[1] * (0.62 if sup else 1.0)
        return w * size / self.EM

    def draw(self, p: QPainter, x, y, text, size, color, bold=False, italic=False,
             halo=None, halo_w=0.5, align="l"):
        """Draw text with baseline-left at (x, y). `text` may be a list of
        (string, superscript) runs. `size` is the em height in mm."""
        runs = self._runs(text)
        if align != "l":
            w = self.width(runs, size, bold, italic)
            x -= w / 2.0 if align == "c" else w
        k = size / self.EM
        full = QPainterPath()
        segs = []
        cx = x
        for s, sup in runs:
            pp, adv = self._piece(s, bold, italic)
            kk = k * (0.62 if sup else 1.0)
            yy = y - (0.36 * size if sup else 0.0)
            tr = QTransform().translate(cx, yy).scale(kk, kk)
            segs.append((s, cx, yy, kk))
            full.addPath(tr.map(pp))
            cx += adv * kk
        if halo is not None:
            pen = QPen(QColor(halo), halo_w)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.strokePath(full, pen)
        if self.as_paths:
            p.fillPath(full, QBrush(QColor(color)))
        else:
            f, _ = self._font(bold, italic)
            p.setPen(QPen(QColor(color)))
            p.setFont(f)
            for s, sx, sy, kk in segs:
                p.save()
                p.translate(sx, sy)
                p.scale(kk, kk)
                p.drawText(QPointF(0.0, 0.0), s)
                p.restore()


# -----------------------------------------------------------------------------
# Label placement
# -----------------------------------------------------------------------------
class Placer:
    def __init__(self, rect, cell=6.0):
        self.rect = rect
        self.cell = cell
        self.grid = {}
        self.marks = None        # (cKDTree, radii, owners) of every star and symbol

    def set_marks(self, xs, ys, rs, owners):
        try:
            from scipy.spatial import cKDTree
            if len(xs):
                self.marks = (cKDTree(np.column_stack([xs, ys])), np.asarray(rs, float), owners)
        except Exception:
            self.marks = None

    def seg_near_mark(self, ax, ay, bx, by, owner, clear=0.25):
        """Does the segment pass over (or right next to) a star or symbol other than
        the owner's? A leader line must never seem to point at a neighbour."""
        if self.marks is None:
            return False
        tree, rs, owners = self.marks
        L = math.hypot(bx - ax, by - ay)
        n = max(2, int(L / 0.15) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = np.column_stack([ax + (bx - ax) * t, ay + (by - ay) * t])
        rmax = float(rs.max()) + clear if len(rs) else clear
        for js in tree.query_ball_point(pts, rmax):
            for j in js:
                if owners[j] == owner:
                    continue
                dd = np.hypot(pts[:, 0] - tree.data[j, 0], pts[:, 1] - tree.data[j, 1])
                if rs[j] < 0.3:
                    # faint stars only matter near the object, where they could be
                    # taken for the target; further out the line may cross them
                    dd = dd[t * L < 1.5]
                if len(dd) and np.min(dd) < rs[j] + clear:
                    return True
        return False

    def _cells(self, r):
        c = self.cell
        for i in range(int(r[0] // c), int(r[2] // c) + 1):
            for j in range(int(r[1] // c), int(r[3] // c) + 1):
                yield (i, j)

    def add(self, r, owner=None, hard=True):
        item = (r, owner, hard)
        for k in self._cells(r):
            self.grid.setdefault(k, []).append(item)

    def hits(self, r, owner):
        hard = soft = False
        for k in self._cells(r):
            for (o, own, hd) in self.grid.get(k, ()):
                if own is not None and own == owner:
                    continue
                if o[0] < r[2] and o[2] > r[0] and o[1] < r[3] and o[3] > r[1]:
                    if hd:
                        return True, soft
                    soft = True
        return hard, soft

    def place(self, x, y, rad, w, h, owner, must=False, gap=0.35):
        """Find a free spot for a w*h label around a symbol of radius `rad`.
        Returns baseline-left (lx, ly) or None."""
        d = 0.72 * rad + gap
        cands = (
            (x + rad + gap, y + h / 2),
            (x + d, y - d),
            (x + d, y + d + h),
            (x - rad - gap - w, y + h / 2),
            (x - d - w, y - d),
            (x - d - w, y + d + h),
            (x - w / 2, y - rad - gap),
            (x - w / 2, y + rad + gap + h),
        )
        x0, y0, x1, y1 = self.rect
        fallback = None
        for lx, ly in cands:
            r = (lx - 0.2, ly - h - 0.2, lx + w + 0.2, ly + 0.25)
            if r[0] < x0 or r[1] < y0 or r[2] > x1 or r[3] > y1:
                continue
            hard, soft = self.hits(r, owner)
            if hard:
                continue
            if not soft:
                self.add(r, None, True)
                return lx, ly
            if fallback is None:
                fallback = (lx, ly, r)
        if must and fallback is not None:
            self.add(fallback[2], None, True)
            return fallback[0], fallback[1]
        return None

    def seg_hits(self, ax, ay, bx, by, owner):
        """Does the segment a-b cross any obstacle (other than the owner's)?"""
        c = self.cell
        seen = set()
        for k in self._cells((min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))):
            for item in self.grid.get(k, ()):
                if id(item) in seen:
                    continue
                seen.add(id(item))
                o, own, _hd = item
                if own is not None and own == owner:
                    continue
                # Liang-Barsky clip of the segment against the box
                t0, t1 = 0.0, 1.0
                dx, dy = bx - ax, by - ay
                ok = True
                for pp, qq in ((-dx, ax - o[0]), (dx, o[2] - ax), (-dy, ay - o[1]), (dy, o[3] - ay)):
                    if abs(pp) < 1e-12:
                        if qq < 0:
                            ok = False
                            break
                    else:
                        t = qq / pp
                        if pp < 0:
                            t0 = max(t0, t)
                        else:
                            t1 = min(t1, t)
                        if t0 > t1:
                            ok = False
                            break
                if ok:
                    return True
        return False

    def add_segment(self, ax, ay, bx, by, owner=None, width=0.3):
        """Register a leader line so that later labels and lines keep off it."""
        n = max(1, int(math.hypot(bx - ax, by - ay) / 1.0))
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            xa, ya = ax + (bx - ax) * t0, ay + (by - ay) * t0
            xb, yb = ax + (bx - ax) * t1, ay + (by - ay) * t1
            self.add((min(xa, xb) - width, min(ya, yb) - width,
                      max(xa, xb) + width, max(ya, yb) + width), owner, True)

    def place_far(self, x, y, rad, w, h, owner, dists=(2.2, 3.5, 5.0, 7.0, 9.5), edge=0.15):
        """Label away from a crowded object, joined to it by a leader line.
        Returns (lx, ly, ax, ay, sx, sy): label baseline-left, the end of the line on
        the label and its start on the object, or None. The line takes the shortest
        free path: it never crosses another label, star or object, so that it is
        always clear which object it points to."""
        x0, y0, x1, y1 = self.rect
        gap = 0.35
        for dd in dists:
            best = None
            for k in range(16):
                a = math.radians(k * 22.5)
                ca, sa = math.cos(a), math.sin(a)
                L = rad + dd
                ax, ay = x + L * ca, y - L * sa          # end of the line, on the label
                # the label touches that point with the middle of its nearest side
                if ca > 0.38:
                    lx, ly = ax + gap, ay + h / 2
                elif ca < -0.38:
                    lx, ly = ax - gap - w, ay + h / 2
                elif sa > 0:
                    lx, ly = ax - w / 2, ay - gap
                else:
                    lx, ly = ax - w / 2, ay + gap + h
                r = (lx - 0.2, ly - h - 0.2, lx + w + 0.2, ly + 0.25)
                if r[0] < x0 or r[1] < y0 or r[2] > x1 or r[3] > y1:
                    continue
                hard, soft = self.hits(r, owner)
                if hard or soft:
                    continue
                sx, sy = x + (rad + edge) * ca, y - (rad + edge) * sa
                if self.seg_hits(sx, sy, ax, ay, owner) or self.seg_near_mark(sx, sy, ax, ay, owner):
                    continue
                # prefer the right, then the diagonals: the usual chart reading order
                score = (0 if ca > 0.38 else 1) + 0.1 * (k % 2)
                if best is None or score < best[0]:
                    best = (score, lx, ly, ax, ay, sx, sy, r)
            if best is not None:
                _, lx, ly, ax, ay, sx, sy, r = best
                self.add(r, None, True)
                self.add_segment(sx, sy, ax, ay)
                return lx, ly, ax, ay, sx, sy
        return None


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
PAGE_FORMATS = {
    "A5 (148 x 210 mm)": (148.0, 210.0), "A4 (210 x 297 mm)": (210.0, 297.0),
    "A3 (297 x 420 mm)": (297.0, 420.0), "A2 (420 x 594 mm)": (420.0, 594.0),
    "A1 (594 x 841 mm)": (594.0, 841.0), "A0 (841 x 1189 mm)": (841.0, 1189.0),
    "Letter (8.5 x 11 in)": (215.9, 279.4), "Legal (8.5 x 14 in)": (215.9, 355.6),
    "Tabloid (11 x 17 in)": (279.4, 431.8), "Custom": (0.0, 0.0),
}

LAYERS = [
    ("milkyway", "Milky Way", "The Milky Way filled with its isophotes", True),
    ("halpha", "Emission Nebulae", "Real shapes of emission nebulae from the all-sky H-alpha map "
                                   "(Finkbeiner 2003, 6' resolution)", True),
    ("dust", "Dark Nebulae", "Dark nebulae from the all-sky dust map (Schlegel, Finkbeiner & Davis 1998)", True),
    ("grid", "Grid", "Grid in ICRS equatorial coordinates", True),
    ("ecliptic", "Ecliptic", "The ecliptic great circle", False),
    ("galactic", "Galactic Equator", "The ICRS galactic equator great circle", False),
    ("borders", "Constellation Borders", "Borders of the constellations", True),
    ("lines", "Constellation Lines", "Asterisms of the constellations", True),
    ("connames", "Constellation Names", "Names of the constellations", True),
    ("messier", "Messier", "Messier catalog (110 objects)", True),
    ("ngc", "NGC", "New General Catalogue (OpenNGC)", True),
    ("ic", "IC", "Index Catalogue (OpenNGC)", True),
    ("other", "Other DSO", "OpenNGC addendum (Barnard, Melotte, Collinder...)", False),
    ("sharpless", "Sharpless", "Sh2 HII regions (SASpro catalog)", False),
    ("lbn", "LBN", "Lynds bright nebulae (SASpro catalog)", False),
    ("ldn", "LDN", "Lynds dark nebulae (SASpro catalog)", False),
    ("abell", "Abell", "Abell galaxy clusters (SASpro catalog)", False),
    ("png", "PNG", "Galactic planetary nebulae (SASpro catalog)", False),
    ("outlines", "Nebula Outlines", "Real outlines of nebulae at three brightness levels (OpenNGC)", True),
    ("stars", "Bright Stars", "HYG 4.1: bright stars with names, Bayer letters and B-V colors", True),
    ("gaia", "Gaia DR3 (local)", "All other stars, from SASpro's local Gaia DR3 library "
                                 "(colors from the XP spectra when installed)", True),
    ("starnames", "Star Names", "Proper names of bright stars", True),
    ("bayer", "Bayer / Flamsteed", "Bayer letters and Flamsteed numbers", True),
    ("variable", "Variable Stars", "Mark catalogued variable stars (GCVS designation)", True),
    ("double", "Double Stars", "Mark double stars (Washington Double Star catalog)", True),
]


@dataclass
class ChartParams:
    ra: float = 283.0            # degrees
    dec: float = 36.9
    fov: float = 40.0            # horizontal field, degrees
    projection: str = "Stereographic"
    rotation: float = 0.0
    mirrored: bool = False
    page_format: str = "A4 (210 x 297 mm)"
    landscape: bool = True
    page_w: float = 297.0
    page_h: float = 210.0
    margin: float = 10.0
    title: str = ""
    layers: dict = field(default_factory=lambda: {k: on for k, _n, _d, on in LAYERS})
    # contents
    star_mag_auto: bool = True
    star_mag: float = 8.8
    star_min: float = 0.30       # smallest star diameter, mm
    star_max: float = 3.60       # largest star diameter, mm
    halos: bool = True
    colored_stars: bool = True
    online_colors: bool = True   # Gaia BP-RP from the archive for stars beyond the local XP library
    dso_mag_auto: bool = True
    dso_mag: float = 12.3
    dso_min: float = 1.6         # smallest DSO symbol, mm
    # rendering
    text_scale: float = 1.0
    graphics_scale: float = 1.0
    black_white: bool = False
    legend: bool = True
    font: str = ""
    text_as_paths: bool = True
    # image outline (list of [ra, dec])
    outline: list = field(default_factory=list)
    outline_name: str = ""
    image_mag: float = 0.0       # limiting magnitude measured on the outlined image (0 = none)
    image_stars: int = 0
    ref_grid: dict = field(default_factory=dict)   # linear WCS of the reference image
    # ---- contents (stars) ----
    star_density: float = 20.0   # automatic limit: at most this many stars per cm2 of map
    star_names: list = field(default_factory=list)   # chosen proper names ([] = automatic)
    halo_width: float = 0.20     # mm
    halo_map_bg: bool = True     # halos take the map background colour
    halo_color: str = "#ffffff"
    halo_alpha: int = 255
    outlined_stars: bool = False
    star_outline_color: str = "#000000"
    star_outline_width: float = 0.10
    # ---- rendering ----
    bg_color: str = "#ffffff"
    map_color: str = "#ffffff"
    frame_color: str = "#000000"
    frame_alpha: int = 255
    remove_duplicates: bool = True
    label_opt: bool = True
    overlap_stars: bool = False
    outlined_labels: bool = True
    leader_lines: bool = True
    leader_color: str = "#303030"
    leader_width: float = 0.08
    reduction: float = 0.5
    dso_styles: dict = field(default_factory=dict)
    legend_pos: str = "Bottom"
    title_font: str = ""
    title_size: float = 12.0     # points
    title_bold: bool = True
    title_italic: bool = False
    text_font: str = ""
    text_size: float = 6.0       # points
    text_bold: bool = False
    text_italic: bool = False
    layer_marks: dict = field(default_factory=dict)
    layer_labels: dict = field(default_factory=dict)
    # ---- output ----
    out_svg: bool = False
    out_pdf: bool = False
    out_png: bool = False
    out_image: bool = True
    svg_path: str = ""
    pdf_path: str = ""
    png_path: str = ""
    resolution: float = 300.0
    res_unit: str = "pixels per inch"
    image_id: str = ""
    match_image: bool = False    # image output on the reference image's own pixel grid
    # Drawn over a survey photograph (Finder Chart): dark-sky palette, transparent
    # page, no fills, frame, edge labels or legend. Set by render_overlay().
    overlay: bool = False
    # Unnamed main stars fall back to their HD/HIP number on narrow charts. Right
    # for a printed chart; noise on an image view, which turns it off.
    catalog_ids: bool = True

    def dpi(self):
        return self.resolution * (2.54 if self.res_unit.startswith("pixels per cm") else 1.0)

    def marks(self, key):
        return bool(self.layers.get(key)) and self.layer_marks.get(key, True)

    def labels(self, key):
        return bool(self.layers.get(key)) and self.layer_labels.get(key, True)

    def page_size(self):
        if self.page_format in PAGE_FORMATS and self.page_format != "Custom":
            w, h = PAGE_FORMATS[self.page_format]
            return (max(w, h), min(w, h)) if self.landscape else (min(w, h), max(w, h))
        return self.page_w, self.page_h


LINE_TYPES = ["Solid", "Dash", "Dot", "Dash-dot"]
DSO_STYLE_DEFAULTS = {
    "Galaxy": ("Solid", "#c8323c", 255, "#c8323c", 56),
    "Group of galaxies": ("Dash", "#c8323c", 255, "#c8323c", 0),
    "Open cluster": ("Dash", "#c08a00", 255, "#c08a00", 56),
    "Globular cluster": ("Solid", "#c08a00", 255, "#c08a00", 56),
    "Planetary nebula": ("Solid", "#22885a", 255, "#22885a", 56),
    "Bright nebula": ("Solid", "#22885a", 255, "#22885a", 56),
    "Reflection nebula": ("Dash-dot", "#3a72c0", 255, "#3a72c0", 56),
    "Cluster with nebula": ("Solid", "#22885a", 255, "#22885a", 56),
    "Dark nebula": ("Dot", "#7a5a3a", 255, "#7a5a3a", 32),
    "Stellar association": ("Dot", "#c08a00", 255, "#c08a00", 0),
}


def dso_style(P, kind):
    """(line type, line QColor, fill QColor or None) for a deep-sky object kind."""
    name = KIND_NAMES[kind]
    st = (P.dso_styles or {}).get(name) or DSO_STYLE_DEFAULTS[name]
    lt, lc, la, fc, fa = st
    if P.black_white:
        lc, fc = "#000000", "#000000"
        fa = min(int(fa), 40)
    line = QColor(lc)
    line.setAlpha(int(la))
    fill = None
    if int(fa) > 0:
        fill = QColor(fc)
        fill.setAlpha(int(fa))
    return lt, line, fill


def pen_dash(pen, line_type, w):
    if line_type == "Dash":
        pen.setDashPattern([3.0, 2.0])
    elif line_type == "Dot":
        pen.setDashPattern([0.6, 1.6])
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    elif line_type == "Dash-dot":
        pen.setDashPattern([3.0, 1.5, 0.6, 1.5])


# nebula boundaries on the all-sky maps (H-alpha excess in rayleigh, dust contrast in dex)
NEB_HA_LEVEL = 80.0
NEB_DUST_LEVEL = 0.32


def fov_star_mag(fov_deg: float) -> float:
    """Automatic star limit from the field of view (PixInsight-like: 40 deg -> 8.8,
    70 deg -> 7.7, 10 deg -> 11.5, 2 deg -> 14.7)."""
    return float(np.clip(16.06 - 4.53 * math.log10(max(fov_deg, 0.05)), 4.0, 17.0))


def auto_star_mag(mm_per_deg: float) -> float:
    return float(np.clip(8.8 + 5.0 * math.log10(max(mm_per_deg, 1e-6) / 6.7), 4.5, 16.0))


def fmt_ra(ra_deg, prec=0):
    h = (ra_deg % 360.0) / 15.0
    tot = round(h * 3600.0 * 10 ** prec) / 10 ** prec
    hh = int(tot // 3600) % 24
    mm = int((tot % 3600) // 60)
    ss = tot % 60
    return f"{hh:02d}h{mm:02d}m{ss:0{2 + (prec + 1 if prec else 0)}.{prec}f}s"


def fmt_dec(dec_deg, prec=0):
    sg = "-" if dec_deg < 0 else "+"
    tot = round(abs(dec_deg) * 3600.0 * 10 ** prec) / 10 ** prec
    dd = int(tot // 3600)
    mm = int((tot % 3600) // 60)
    ss = tot % 60
    return f"{sg}{dd:02d}°{mm:02d}′{ss:0{2 + (prec + 1 if prec else 0)}.{prec}f}″"


def parse_ra(text: str) -> float:
    """Hours (sexagesimal or decimal) -> degrees. A trailing 'd' or '°' means degrees."""
    t = text.strip().lower()
    if not t:
        raise ValueError("empty RA")
    is_deg = t.endswith("d") or t.endswith("°") or "deg" in t
    parts = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", t)]
    if not parts:
        raise ValueError("bad RA")
    if is_deg:
        return parts[0] % 360.0
    while len(parts) < 3:
        parts.append(0.0)
    return ((parts[0] + parts[1] / 60.0 + parts[2] / 3600.0) * 15.0) % 360.0


def parse_dec(text: str) -> float:
    t = text.strip()
    parts = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    if not parts:
        raise ValueError("bad Dec")
    while len(parts) < 3:
        parts.append(0.0)
    v = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    if t.startswith("-") or t.startswith("−"):
        v = -v
    return max(-90.0, min(90.0, v))


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------
COLORS = {
    "page": "#ffffff", "map": "#ffffff", "frame": "#000000", "text": "#111111",
    "mw": ["#e7eef8", "#d4dff1", "#c2d2ec", "#b3c7e8", "#a6bee4", "#9ab4e0", "#8eaadb"],
    "grid": "#c9c9c9", "ecliptic": "#d19a2a", "galactic": "#7f8fd6",
    "border": "#cfa9a9", "line": "#5f84c9", "conname": "#93a9d3",
    "label": "#1a1a1a", "halo": "#ffffff", "outline": "#e0452a",
    "star_none": "#30303a",
    "halpha": ("#2c9a68", "#3aa877"),
    "dust": ("#5e5648", "#6d6455"),
    "reflection": ("#4a6cc9", "#7f9ae0"),
    "outline_fill": {
        K_NEB: ("#2c9a68", ["#dcf1e6", "#c3e8d5", "#a8dcc1"]),
        K_RFN: ("#5878d2", ["#dde5f9", "#cad6f5", "#b6c7f0"]),
        K_DARK: ("#7a7a7a", ["#ededed", "#e2e2e2", "#d6d6d6"]),
    },
    "dso": {
        K_GALAXY: ("#cf4f5b", "#f7cdd0"), K_GROUP: ("#cf4f5b", None),
        K_OC: ("#a47d1e", "#f4e3a6"), K_GC: ("#a47d1e", "#f4e3a6"),
        K_PN: ("#2c9a68", "#c4ead7"), K_NEB: ("#2c9a68", "#aadfc3"),
        K_CLNEB: ("#2c9a68", "#aadfc3"), K_RFN: ("#5878d2", "#cdd9f6"),
        K_DARK: ("#7a7a7a", None), K_ASSOC: ("#8a7a5a", None),
    },
}
COLORS_BW = {
    "page": "#ffffff", "map": "#ffffff", "frame": "#000000", "text": "#000000",
    "mw": ["#f4f4f4", "#ececec", "#e3e3e3", "#dadada", "#d0d0d0", "#c6c6c6", "#bcbcbc"],
    "grid": "#bdbdbd", "ecliptic": "#555555", "galactic": "#555555",
    "border": "#9a9a9a", "line": "#444444", "conname": "#9a9a9a",
    "label": "#000000", "halo": "#ffffff", "outline": "#000000",
    "star_none": "#000000",
    "halpha": ("#555555", "#777777"),
    "dust": ("#333333", "#444444"),
    "reflection": ("#555555", "#888888"),
    "outline_fill": {k: ("#000000", ["#f0f0f0", "#e2e2e2", "#d2d2d2"]) for k in (K_NEB, K_RFN, K_DARK)},
    "dso": {k: ("#000000", ("#e6e6e6" if k in (K_GALAXY, K_OC, K_GC, K_PN, K_NEB, K_CLNEB, K_RFN) else None))
            for k in range(10)},
}


# Over a survey photograph: light linework and text on dark halos, so the chart
# reads against black sky without hiding the image under it.
COLORS_OVERLAY = dict(COLORS)
COLORS_OVERLAY.update({
    "page": "#000000", "map": "#000000", "frame": "#f0f0f0", "text": "#f0f0f0",
    "grid": "#8a8a8a", "ecliptic": "#e6b55a", "galactic": "#a6b4ff",
    "border": "#d99a9a", "line": "#86adff", "conname": "#b4c6ea",
    "label": "#f4f4f4", "halo": "#000000", "outline": "#ff7a5c",
    "star_none": "#d0d0d0",
})
# Deep-sky symbols over a photograph: brighter lines, no fills.
OVERLAY_DSO_STYLES = {
    "Galaxy": ("Solid", "#ff7d86", 255, "#ff7d86", 0),
    "Group of galaxies": ("Dash", "#ff7d86", 255, "#ff7d86", 0),
    "Open cluster": ("Dash", "#ffd25a", 255, "#ffd25a", 0),
    "Globular cluster": ("Solid", "#ffd25a", 255, "#ffd25a", 0),
    "Planetary nebula": ("Solid", "#6be0a6", 255, "#6be0a6", 0),
    "Bright nebula": ("Solid", "#6be0a6", 255, "#6be0a6", 0),
    "Reflection nebula": ("Dash-dot", "#86adff", 255, "#86adff", 0),
    "Cluster with nebula": ("Solid", "#6be0a6", 255, "#6be0a6", 0),
    "Dark nebula": ("Dot", "#c8a882", 255, "#c8a882", 0),
    "Stellar association": ("Dot", "#ffd25a", 255, "#ffd25a", 0),
}


def star_color(ci: float) -> str:
    if ci != ci:
        return "#8c8c96"             # no colour data: neutral grey, unlike the dark F/G stars
    if ci < 0.0:
        return "#1b3bd8"
    if ci < 0.30:
        return "#3458c4"
    if ci < 0.58:
        return "#26262e"
    if ci < 0.90:
        return "#b1832c"
    if ci < 1.30:
        return "#c96d16"
    return "#c9471c"


# FITS codes of the zenithal projections (all of them exactly the ones drawn here)
_FITS_PROJ = {"Stereographic": "STG", "Gnomonic": "TAN", "Orthographic": "SIN",
              "Azimuthal equidistant": "ARC", "Lambert equal-area": "ZEA"}
_PROJ_OF_FITS = {v: k for k, v in _FITS_PROJ.items()}


def ref_grid_from_wcs(wcs, w, h) -> dict:
    """Linear part of an image's celestial WCS (SIP distortions are ignored)."""
    ww = wcs.wcs
    code = ""
    try:
        toks = [t for t in str(ww.ctype[0]).split("-") if t]
        code = toks[1].upper() if len(toks) > 1 else ""
    except Exception:
        pass
    if not str(ww.ctype[0]).upper().startswith("RA"):
        return {}
    cd = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
    return {"w": int(w), "h": int(h), "crval": [float(ww.crval[0]), float(ww.crval[1])],
            "crpix0": [float(ww.crpix[0]) - 1.0, float(ww.crpix[1]) - 1.0],
            "cd": cd.tolist(), "proj": _PROJ_OF_FITS.get(code, "Gnomonic")}


def ref_grid_geometry(g: dict, k: float):
    """Chart geometry reproducing a reference image's pixel grid at k px/mm:
    (ra0, dec0, projection, rotation_deg, mirrored, scale_mm_per_rad, cx_mm, cy_mm, w_mm, h_mm).
    Pixel offset = k * scale * (pi/180) * M(rot, mirror) @ (x, y) intermediate world
    coordinates in degrees (x east, y north), with M = [[c*m, -s], [-s*m, -c]] and
    m = -1 for the normal (east-left) orientation, +1 mirrored."""
    B = np.linalg.inv(np.asarray(g["cd"], dtype=np.float64))
    det = float(np.linalg.det(B))
    m = -1.0 if det > 0 else 1.0
    c = math.sqrt(abs(det))                       # pixels per degree
    M = B / c
    cr, sr = float(M[0, 0]) * m, -float(M[0, 1])
    rot = math.degrees(math.atan2(sr, cr))
    scale = c / (k * D2R)
    cx, cy = (g["crpix0"][0] + 0.5) / k, (g["crpix0"][1] + 0.5) / k
    return (g["crval"][0], g["crval"][1], g.get("proj", "Gnomonic"), rot, m > 0, scale,
            cx, cy, g["w"] / k, g["h"] / k)


def chart_wcs_header(r, k: float, w: int, h: int):
    """FITS header with the exact WCS of a chart rendered at k px/mm (pixel (0,0) is the
    top-left pixel, as SASpro uses image rows)."""
    from astropy.io import fits
    pr = r.proj
    m = 1.0 if pr.mir else -1.0
    cr, sr = math.cos(pr.rot), math.sin(pr.rot)
    c = k * pr.scale * D2R
    A = c * np.array([[cr * m, -sr], [-sr * m, -cr]])
    CD = np.linalg.inv(A)
    code = _FITS_PROJ.get(pr.kind, "STG")
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = int(w)
    hdr["NAXIS2"] = int(h)
    hdr["WCSAXES"] = 2
    hdr["CTYPE1"] = "RA---" + code
    hdr["CTYPE2"] = "DEC--" + code
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CRVAL1"] = math.degrees(pr.ra0) % 360.0
    hdr["CRVAL2"] = math.degrees(pr.dec0)
    hdr["CRPIX1"] = k * pr.cx + 0.5
    hdr["CRPIX2"] = k * pr.cy + 0.5
    hdr["CD1_1"], hdr["CD1_2"] = float(CD[0, 0]), float(CD[0, 1])
    hdr["CD2_1"], hdr["CD2_2"] = float(CD[1, 0]), float(CD[1, 1])
    hdr["LONPOLE"] = 180.0
    hdr["RADESYS"] = "ICRS"
    hdr["EQUINOX"] = 2000.0
    hdr["CREATOR"] = "Seti Astro Suite Pro sky chart"
    return hdr


class ChartRenderer:
    """Computes and paints one chart. Painter units are millimetres."""

    def __init__(self, params: ChartParams, data: ChartData, log=print, wait_online=False,
                 projection=None):
        self.wait_online = wait_online       # outputs wait for the online colours
        self._match = None
        # A ready-made projection (WcsProjection) supplies the whole geometry:
        # the page is its image, edge to edge, with no frame, margin or legend.
        self._custom = projection
        if params.match_image and params.ref_grid:
            # the chart takes the reference image's own geometry and pixel grid
            ra0, dec0, kind, rot, mir, sc, cx, cy, pw, ph = ref_grid_geometry(
                params.ref_grid, params.dpi() / 25.4)
            pr = Projection(ra0, dec0, kind, rot, mir)
            half = float(pr.Rinv(np.array([pw / 2.0 / sc]))[0])
            fov = 2.0 * half * R2D if half == half else params.fov
            params = replace(params, ra=ra0, dec=dec0, projection=kind, rotation=rot,
                             mirrored=mir, fov=fov, legend=False)
            self._match = (sc, cx, cy, pw, ph)
        self.p = params
        self.d = data
        self.log = log
        if params.overlay:
            C = dict(COLORS_OVERLAY)
        else:
            C = dict(COLORS_BW if params.black_white else COLORS)
            if not params.black_white:
                C["page"], C["map"] = params.bg_color, params.map_color
            fc = QColor(params.frame_color)
            fc.setAlpha(int(params.frame_alpha))
            C["frame"] = C["text"] = fc
            C["halo"] = params.map_color if params.halo_map_bg else params.halo_color
        # nebula shapes follow the deep-sky object styles
        for key, kind in (("halpha", K_NEB), ("dust", K_DARK), ("reflection", K_RFN)):
            _lt, lc, fl = dso_style(params, kind)
            C[key] = (lc.name(), (fl or lc).name())
        self.C = C
        self.info = {}
        self._layout()

    # ---- layout ----------------------------------------------------------
    def _layout(self):
        P = self.p
        if self._custom is not None:
            pr = self._custom
            self.pw, self.ph = pr.page_w, pr.page_h
            self.legend_h = 0.0
            self.frame = (0.0, 0.0, self.pw, self.ph)
            self.proj = pr
            self.fov_w, self.fov_h = pr.fov_w, pr.fov_h
            self.mm_per_deg = pr.mm_per_deg
            self.star_mag = P.star_mag if not P.star_mag_auto else fov_star_mag(self.fov_w)
            self.dso_mag = min(self.star_mag + 3.5, 20.0) if P.dso_mag_auto else P.dso_mag
            self._visible_range()
            return
        self.pw, self.ph = P.page_size()
        ts = P.text_scale
        pad = 5.0 * ts if P.layers.get("grid", True) else 0.0
        m = P.margin
        self.legend_h = (25.0 * ts * max(1.0, P.text_size / 6.0, P.title_size / 12.0)) if P.legend else 0.0
        top = P.legend and P.legend_pos == "Top"
        x0, y0 = m + pad, m + pad + (self.legend_h if top else 0.0)
        x1 = self.pw - m - pad
        y1 = self.ph - m - pad - (0.0 if top else self.legend_h)
        if x1 - x0 < 20 or y1 - y0 < 20:
            raise ValueError("The page is too small for the chosen margin and legend.")
        if self._match is not None:
            # the whole page is the map, pixel for pixel the reference image
            self.pw, self.ph = self._match[3], self._match[4]
            self.legend_h = 0.0
            x0, y0, x1, y1 = 0.0, 0.0, self.pw, self.ph
        self.frame = (x0, y0, x1, y1)
        fov = min(P.fov, _PROJ_FOVMAX.get(P.projection, 240.0))
        self.proj = Projection(P.ra, P.dec, P.projection, P.rotation, P.mirrored)
        self.proj.fit(fov, x1 - x0, (x0 + x1) / 2.0, (y0 + y1) / 2.0)
        if self._match is not None:
            self.proj.scale, self.proj.cx, self.proj.cy = self._match[0], self._match[1], self._match[2]
        self.fov_w = fov
        _, dtop = self.proj.inverse(np.array([(x0 + x1) / 2.0] * 2), np.array([y0, y1]))
        # vertical field measured through the centre
        r = (y1 - y0) / 2.0 / self.proj.scale
        t = self.proj.Rinv(np.array([r]))[0]
        self.fov_h = float(2.0 * t * R2D) if t == t else 180.0
        self.mm_per_deg = self.proj.mm_per_deg
        if not P.star_mag_auto:
            self.star_mag = P.star_mag
        else:
            # balanced for the field of view; a reference image shallower than that
            # lowers the limit to what the image actually shows
            self.star_mag = fov_star_mag(self.fov_w)
            if P.image_mag and P.outline:
                self.star_mag = min(self.star_mag, P.image_mag)
        self.dso_mag = min(self.star_mag + 3.5, 20.0) if P.dso_mag_auto else P.dso_mag
        self._visible_range()

    def _visible_range(self):
        x0, y0, x1, y1 = self.frame
        gx, gy = np.meshgrid(np.linspace(x0, x1, 25), np.linspace(y0, y1, 25))
        ra, dec = self.proj.inverse(gx.ravel(), gy.ravel())
        self.all_sky = False
        if np.isnan(dec).any():
            self.all_sky = True
            self.dec_lo, self.dec_hi, self.ra_lo, self.ra_hi = -90.0, 90.0, 0.0, 360.0
            self.pole_in = True
            return
        self.dec_lo, self.dec_hi = float(dec.min()), float(dec.max())
        self.pole_in = False
        for pd in (90.0, -90.0):
            px, py, th = self.proj.project(0.0, pd)
            if x0 <= px <= x1 and y0 <= py <= y1 and th < self.proj.tmax:
                self.pole_in = True
                if pd > 0:
                    self.dec_hi = 90.0
                else:
                    self.dec_lo = -90.0
        if self.pole_in:
            self.ra_lo, self.ra_hi = 0.0, 360.0
        else:
            dra = (ra - self.p.ra + 180.0) % 360.0 - 180.0
            self.ra_lo, self.ra_hi = self.p.ra + float(dra.min()), self.p.ra + float(dra.max())
            if self.ra_hi - self.ra_lo > 300:
                self.ra_lo, self.ra_hi = 0.0, 360.0

    def inside(self, x, y, pad=0.0):
        x0, y0, x1, y1 = self.frame
        return (x >= x0 + pad) & (x <= x1 - pad) & (y >= y0 + pad) & (y <= y1 - pad)

    # ---- main entry --------------------------------------------------------
    def render(self, painter: QPainter):
        P, C = self.p, self.C
        # layers that only draw marks are off when their markers are off
        L = dict(P.layers)
        for k in ("milkyway", "halpha", "dust", "ecliptic", "galactic", "borders", "lines", "outlines",
                  "variable", "double"):
            L[k] = P.marks(k)
        self.L = L
        gs, ts = P.graphics_scale, P.text_scale
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fam = P.font or QApplication.font().family()
        self.T = TextEngine(fam, P.text_as_paths)
        x0, y0, x1, y1 = self.frame
        self.placer = Placer(self.frame)
        self.edge_labels = []
        self.kinds_drawn = set()
        self.has_var = self.has_dbl = False

        if not P.overlay:          # an overlay leaves the photograph showing through
            painter.fillRect(QRectF(0, 0, self.pw, self.ph), QColor(C["page"]))
            painter.fillRect(QRectF(x0, y0, x1 - x0, y1 - y0), QColor(C["map"]))
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if P.marks("milkyway"):
            self._draw_milkyway(painter)
        self.ha_lookup = self.dust_lookup = None
        self.outlined = set()
        self.info["outlines"] = 0
        if L.get("dust") or L.get("halpha") or L.get("outlines"):
            self._draw_nebulae(painter, gs)
        if L.get("grid"):
            self._draw_grid(painter, gs)
        if not P.labels("grid"):
            self.edge_labels = []
        if L.get("ecliptic"):
            lam = np.linspace(0, 360, 721) * D2R
            eps = 23.4392911 * D2R
            ra = np.arctan2(np.sin(lam) * math.cos(eps), np.cos(lam)) * R2D
            dec = np.arcsin(math.sin(eps) * np.sin(lam)) * R2D
            self._stroke_sky(painter, ra, dec, C["ecliptic"], 0.22 * gs, dash=(4, 2.5))
        if L.get("galactic"):
            l = np.linspace(0, 360, 721) * D2R
            A = np.array([[-0.0548755604, +0.4941094279, -0.8676661490],
                          [-0.8734370902, -0.4448296300, -0.1980763734],
                          [-0.4838350155, +0.7469822445, +0.4559837762]])
            v = A @ np.vstack([np.cos(l), np.sin(l), np.zeros_like(l)])
            self._stroke_sky(painter, np.arctan2(v[1], v[0]) * R2D, np.arcsin(v[2]) * R2D,
                             C["galactic"], 0.22 * gs, dash=(4, 2.5))
        if P.marks("borders"):
            for seg in self.d.con_bounds:
                ra, dec = self._densify(seg)
                self._stroke_sky(painter, ra, dec, C["border"], 0.16 * gs)
        if P.labels("connames"):
            self._draw_connames(painter, ts)

        stars = self._select_stars()
        if P.marks("lines"):
            self._draw_conlines(painter, stars, gs)
        dsos = self._select_dsos()
        self._last_dsos = dsos
        self._draw_dsos(painter, dsos, gs)
        if P.outline:
            o = np.array(P.outline, dtype=np.float64)
            self._stroke_sky(painter, o[:, 0], o[:, 1], C["outline"], 0.30 * gs)
        if P.marks("stars") or P.marks("gaia"):
            self._draw_stars(painter, self._marked_stars(stars), gs)
        self._draw_labels(painter, stars, dsos, ts)

        # frame + coordinate labels (an overlay has the host chart's own axes)
        if not P.overlay:
            pen = QPen(QColor(C["frame"]), 0.25 * gs)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            # (frame colour may be translucent)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            self._draw_edge_labels(painter, ts, gs)
            if P.legend:
                self._draw_legend(painter, ts, gs)
        self.info["faintest"] = float(stars["mag"].max()) if len(stars["mag"]) else None
        self.info.update(stars=len(stars["x"]), dsos=len(dsos), star_mag=self.star_mag,
                         dso_mag=self.dso_mag, fov_w=self.fov_w, fov_h=self.fov_h,
                         mm_per_deg=self.mm_per_deg)

    # ---- generic sky polyline ---------------------------------------------
    @staticmethod
    def _densify(seg, step=1.0):
        ra, dec = seg[:, 0], seg[:, 1]
        out_ra, out_dec = [], []
        for k in range(len(ra) - 1):
            d = (ra[k + 1] - ra[k] + 180.0) % 360.0 - 180.0
            n = max(1, int(max(abs(d), abs(dec[k + 1] - dec[k])) / step))
            t = np.arange(n) / n
            out_ra.append(ra[k] + d * t)
            out_dec.append(dec[k] + (dec[k + 1] - dec[k]) * t)
        out_ra.append(ra[-1:])
        out_dec.append(dec[-1:])
        return np.concatenate(out_ra), np.concatenate(out_dec)

    def _stroke_sky(self, painter, ra, dec, color, width, dash=None, crossings=None, tag=None):
        x, y, th = self.proj.project(ra, dec)
        path = polyline_path(x, y, th < self.proj.tmax, self.frame, crossings, tag)
        if path.isEmpty():
            return
        pen = QPen(QColor(color), width)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if dash:
            pen.setDashPattern([dash[0], dash[1]])
        painter.strokePath(path, pen)

    # ---- Milky Way ---------------------------------------------------------
    def _gaia_library_dir(self):
        return gaia_library_dir()

    def _draw_milkyway(self, painter):
        F = self.d.milkyway_map(self._gaia_library_dir())
        if F is not None and self._draw_milkyway_gaia(painter, F):
            return
        self._draw_milkyway_polygons(painter)

    def _draw_milkyway_gaia(self, painter, F):
        """Isophotes of the Gaia integrated starlight, filled with nested shades."""
        P = self.p
        x0, y0, x1, y1 = self.frame
        key = ("mw", round(P.ra, 6), round(P.dec, 6), P.fov, P.projection, P.rotation, P.mirrored,
               tuple(round(v, 3) for v in self.frame))
        cache = self.d.__dict__.setdefault("_map_cache", {})
        if key in cache:
            xs, ys, Z, step = cache[key]
        else:
            nside = ChartData.MW_NSIDE
            pix_deg = math.degrees(math.sqrt(4 * math.pi / (12 * nside * nside)))
            sig_deg = max(0.9, 0.6 * pix_deg, self.fov_w / 60.0)      # smooth, map-like isophotes
            sig_mm = sig_deg * self.mm_per_deg
            W, H = x1 - x0, y1 - y0
            cell = max(sig_mm / 4.0, 0.2, math.sqrt((W + 6 * sig_mm) * (H + 6 * sig_mm) / 4.0e5))
            xs, ys = self._grid(min(3.0 * sig_mm, max(W, H)), cell)
            X, Y = np.meshgrid(xs, ys)
            ra, dec = self.proj.inverse(X.ravel(), Y.ravel())
            ok = ~(np.isnan(ra) | np.isnan(dec))
            vals = np.zeros(len(ra))
            if ok.any():
                vals[ok] = F[healpix_nest(nside, ra[ok] * D2R, dec[ok] * D2R)]
            vals = vals.reshape(X.shape)
            good = ok.reshape(X.shape) & (vals > 0)              # empty pixels: no data
            Z = self._blur(np.log10(np.where(good, vals, 1.0)), sig_mm / cell, good)
            step = max(0.15, 0.5 * cell)
            if len(cache) > 8:
                cache.clear()
            cache[key] = (xs, ys, Z, step)
        # levels: log10 flux, from the faint galactic outskirts to the bright star clouds.
        # The first five are calibrated pixel by pixel on PixInsight's own charts
        # (Lyra and Cepheus, 92-99 % agreement); the last two continue the series.
        levels = [-2.76, -2.59, -2.43, -2.32, -2.145, -1.98, -1.815]
        paths = self._contour_paths(xs, ys, Z, levels, step)
        if paths is None:
            return False
        painter.save()
        painter.setClipRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        for k, path in enumerate(paths):
            painter.fillPath(path, QBrush(QColor(self.C["mw"][min(k, len(self.C["mw"]) - 1)])))
        painter.restore()
        return True

    def _draw_milkyway_polygons(self, painter):
        x0, y0, x1, y1 = self.frame
        s0 = math.sin(self.proj.dec0)
        # a galactic pole (never inside the Milky Way) as the parity reference
        refs = [(192.8595, 27.1283), (12.8595, -27.1283)]
        best = None
        for rr, rd in refs:
            px, py, th = self.proj.project(rr, rd)
            if best is None or th < best[2]:
                best = (float(px), float(py), float(th))
        for lvl, rings in enumerate(self.d.mw):
            proj_rings = []
            for ring in rings:
                x, y, _th = self.proj.project(ring[:, 0], ring[:, 1])
                proj_rings.append(np.column_stack([x, y]))
            # Winding numbers on the plane equal the true (spherical) coverage
            # minus the coverage of the projection's far point: compensate it.
            wref = winding_number(best[0], best[1], proj_rings)
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.WindingFill)
            if wref:
                rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
                if winding_number((x0 + x1) / 2, (y0 + y1) / 2, [rect]) * wref > 0:
                    rect = rect[::-1]
                for _ in range(abs(wref)):
                    path.addPolygon(QPolygonF([QPointF(float(a), float(b)) for a, b in rect]))
                    path.closeSubpath()
            for pr in proj_rings:
                if pr[:, 0].max() < x0 or pr[:, 0].min() > x1 or pr[:, 1].max() < y0 or pr[:, 1].min() > y1:
                    continue
                c = clip_ring_rect(pr, x0, y0, x1, y1)
                if len(c) >= 3:
                    path.addPolygon(QPolygonF([QPointF(float(a), float(b)) for a, b in c]))
                    path.closeSubpath()
            if not path.isEmpty():
                painter.fillPath(path, QBrush(QColor(self.C["mw"][min(lvl, len(self.C["mw"]) - 1)])))

    # ---- emission and dark nebulae from all-sky maps -------------------------
    def _grid(self, margin_mm, cell):
        x0, y0, x1, y1 = self.frame
        xs = np.arange(x0 - margin_mm, x1 + margin_mm + cell, cell)
        ys = np.arange(y0 - margin_mm, y1 + margin_mm + cell, cell)
        return xs, ys

    def _sample(self, arr, xs, ys):
        """Map values on a grid of chart positions (NaN outside the sky domain)."""
        X, Y = np.meshgrid(xs, ys)
        ra, dec = self.proj.inverse(X.ravel(), Y.ravel())
        ok = ~(np.isnan(ra) | np.isnan(dec))
        out = np.full(len(ra), np.nan)
        if ok.any():
            l, b = equ_to_gal(ra[ok], dec[ok])
            out[ok] = np.asarray(arr)[healpix_nest(512, l, b)]
        return out.reshape(X.shape)

    @staticmethod
    def _blur(Z, sigma, weights=None):
        """Gaussian blur (normalized when weights are given); identity without scipy."""
        if sigma < 0.3:
            return Z
        try:
            from scipy.ndimage import gaussian_filter
        except Exception:
            return Z
        if weights is None:
            return gaussian_filter(Z, sigma, mode="nearest")
        num = gaussian_filter(np.where(weights, Z, 0.0), sigma, mode="nearest")
        den = gaussian_filter(weights.astype(np.float64), sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 1e-6, num / den, 0.0)

    @staticmethod
    def _resample(Zc, xc, yc, xs, ys):
        """Bilinear resampling of a coarse grid (xc, yc) onto a fine one (xs, ys)."""
        fi = np.clip((ys - yc[0]) / (yc[1] - yc[0]), 0, len(yc) - 1)
        fj = np.clip((xs - xc[0]) / (xc[1] - xc[0]), 0, len(xc) - 1)
        i0 = np.minimum(fi.astype(int), len(yc) - 2)
        j0 = np.minimum(fj.astype(int), len(xc) - 2)
        ti, tj = (fi - i0)[:, None], (fj - j0)[None, :]
        a = Zc[np.ix_(i0, j0)]
        b = Zc[np.ix_(i0, j0 + 1)]
        c = Zc[np.ix_(i0 + 1, j0)]
        d = Zc[np.ix_(i0 + 1, j0 + 1)]
        return (a * (1 - tj) + b * tj) * (1 - ti) + (c * (1 - tj) + d * tj) * ti

    @staticmethod
    def _contour_paths(xs, ys, Z, levels, min_step):
        """Filled contour regions Z >= level as QPainterPaths (holes kept)."""
        try:
            import contourpy
        except Exception:
            return None
        gen = contourpy.contour_generator(xs, ys, np.ascontiguousarray(Z, dtype=np.float64),
                                          fill_type="OuterCode")
        out = []
        for lev in levels:
            pts_list, codes_list = gen.filled(lev, np.inf)
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for pts, codes in zip(pts_list, codes_list):
                starts = np.nonzero(codes == 1)[0].tolist() + [len(codes)]
                for a, b in zip(starts[:-1], starts[1:]):
                    ring = pts[a:b]
                    if len(ring) < 3:
                        continue
                    if len(ring) > 8:     # drop vertices closer than min_step to the previous one
                        seg = np.hypot(np.diff(ring[:, 0]), np.diff(ring[:, 1]))
                        q = np.floor(np.concatenate([[0.0], np.cumsum(seg)]) / min_step)
                        keep = np.concatenate([[True], np.diff(q) > 0])
                        keep[-1] = True
                        ring = ring[keep]
                        if len(ring) < 3:
                            continue
                    poly = QPolygonF()
                    for u, v in ring.tolist():
                        poly.append(QPointF(u, v))
                    path.addPolygon(poly)
                    path.closeSubpath()
            out.append(path)
        return out

    def _make_lookup(self, xs, ys, Z):
        x00, y00, c = xs[0], ys[0], xs[1] - xs[0]

        def look(x, y):
            i = int(round((y - y00) / c))
            j = int(round((x - x00) / c))
            if 0 <= i < Z.shape[0] and 0 <= j < Z.shape[1]:
                return float(Z[i, j])
            return 0.0
        return look

    def _map_fields(self, want_dust):
        """(xs, ys, cell, H-alpha excess, dust contrast) on the chart, cached per geometry."""
        P = self.p
        key = (round(P.ra, 6), round(P.dec, 6), P.fov, P.projection, P.rotation, P.mirrored,
               tuple(round(v, 3) for v in self.frame), bool(want_dust))
        cache = self.d.__dict__.setdefault("_map_cache", {})
        if key in cache:
            return cache[key]
        ha = self.d.sky_map("halpha")
        dust = self.d.sky_map("dust") if want_dust else None
        x0, y0, x1, y1 = self.frame
        pix_mm = 6.87 / 60.0 * self.mm_per_deg            # one HEALPix pixel on the chart
        sig_bg_mm = 2.0 * self.mm_per_deg                 # local background over ~2 deg
        # fine grid (frame + a little), at a third of a map pixel
        W, H = x1 - x0, y1 - y0
        cell = max(pix_mm / 3.0, 0.15, math.sqrt(W * H / 7.0e5))
        xs, ys = self._grid(2.0 * pix_mm, cell)
        # coarse grid for the backgrounds (wide margin, a quarter of the blur scale)
        mrg = min(3.0 * sig_bg_mm, 2.0 * max(W, H))
        cc = max(sig_bg_mm / 4.0, cell, math.sqrt((W + 2 * mrg) * (H + 2 * mrg) / 2.5e5))
        xc, yc = self._grid(mrg, cc)
        Hx = Cz = Hz = None
        if ha is not None:
            Hz = self._sample(ha, xs, ys)
            okf = np.isfinite(Hz)
            Hz = self._blur(np.where(okf, Hz, 0.0), 0.5 * pix_mm / cell)
            Hc = self._sample(ha, xc, yc)
            okc = np.isfinite(Hc)
            Hb = self._blur(np.minimum(np.where(okc, Hc, 0.0), 400.0), sig_bg_mm / cc, okc)
            Hx = Hz - 0.8 * self._resample(Hb, xc, yc, xs, ys)
        if dust is not None:
            E = self._sample(dust, xs, ys)
            valid = np.isfinite(E) & (E > 0)
            Zl = self._blur(np.log10(np.where(valid, E, 1e-3)), 0.9 * pix_mm / cell, valid)
            Ec = self._sample(dust, xc, yc)
            vc = np.isfinite(Ec) & (Ec > 0)
            bg = self._blur(np.log10(np.where(vc, Ec, 1e-3)), sig_bg_mm / cc, vc)
            Cz = np.where(valid, Zl - self._resample(bg, xc, yc, xs, ys), 0.0)
            Cz = np.where(np.where(valid, E, 0) > 0.25, Cz, 0.0)        # skip faint cirrus
            if Hz is not None:                                        # warm dust of HII regions
                Cz = np.where(Hz > 40.0, np.minimum(Cz, 0.0), Cz)
        res = (xs, ys, cell, Hx, Cz)
        if len(cache) > 6:
            cache.clear()
        cache[key] = res
        return res

    # ---- nebulae: one clean profile per nebula ---------------------------------
    @staticmethod
    def _rings_path(rings):
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        for ring in rings:
            poly = QPolygonF()
            for u, v in ring.tolist():
                poly.append(QPointF(u, v))
            path.addPolygon(poly)
            path.closeSubpath()
        return path

    def _outline_rings(self, gs):
        """OpenNGC outlines on the chart: {kind: {level: [Nx2 rings]}}."""
        x0, y0, x1, y1 = self.frame
        out = {}
        n = 0
        for o in self.d.outlines:
            ring = o["ring"]
            x, y, th = self.proj.project(ring[:, 0], ring[:, 1])
            if th.max() >= self.proj.tmax:
                continue
            bx0, bx1, by0, by1 = x.min(), x.max(), y.min(), y.max()
            if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                continue
            if max(bx1 - bx0, by1 - by0) < 1.2 * gs:
                continue
            seg = np.hypot(np.diff(x), np.diff(y))
            q = np.floor(np.concatenate([[0.0], np.cumsum(seg)]) / 0.12)
            keep = np.concatenate([[True], np.diff(q) > 0])
            keep[-1] = True
            pts = np.column_stack([x[keep], y[keep]])
            if len(pts) < 4:
                continue
            kind = o["kind"] if o["kind"] in (K_RFN, K_DARK) else K_NEB
            out.setdefault(kind, {}).setdefault(min(max(o["level"], 1), 3), []).append(pts)
            n += 1
            if o["key"]:
                self.outlined.add(o["key"])
        self.info["outlines"] = n
        return out

    def _draw_nebulae(self, painter, gs):
        L_eff = self.L
        """Emission, reflection and dark nebulae as filled shapes with a single
        outline each. Sources: OpenNGC outlines (finest), the H-alpha map and the
        dust map. Overlapping shapes of the same kind are merged, so every nebula
        shows one clean profile; inner brightness levels are shaded, not stroked."""
        L, C = L_eff, self.C
        x0, y0, x1, y1 = self.frame
        frame = QPainterPath()
        frame.addRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        rings = self._outline_rings(gs) if L.get("outlines") else {}
        use_maps = L.get("halpha") or L.get("dust")
        have_maps = use_maps and (self.d.sky_map("halpha") is not None or self.d.sky_map("dust") is not None)
        xs = ys = Hx = Cz = None
        step = 0.15
        if have_maps:
            xs, ys, cell, Hx, Cz = self._map_fields(bool(L.get("dust")))
            step = max(0.12, 0.5 * cell)

        def level_paths(kind):
            """One region per kind: the standard-brightness outline (level 2) of each
            object, or its only one. No inner isophotes."""
            lv = rings.get(kind, {})
            if not lv:
                return []
            k = max(lv)                         # the brightest (tightest) outline
            return [self._rings_path(lv[k])]

        e_out = level_paths(K_NEB)
        r_out = level_paths(K_RFN)
        d_out = level_paths(K_DARK)
        e_out_union = e_out[0].simplified() if e_out else QPainterPath()
        r_out_union = r_out[0].simplified() if r_out else QPainterPath()

        ha_paths = []
        if Hx is not None and L.get("halpha"):
            ha_paths = self._contour_paths(xs, ys, Hx, [NEB_HA_LEVEL], step) or []
            if ha_paths:
                self.ha_lookup = self._make_lookup(xs, ys, Hx)
        dust_paths = []
        if Cz is not None and L.get("dust"):
            dust_paths = self._contour_paths(xs, ys, Cz, [NEB_DUST_LEVEL], step) or []
            if dust_paths:
                self.dust_lookup = self._make_lookup(xs, ys, Cz)

        # full extent of each family (one outline per family region)
        e_total = e_out_union
        if ha_paths:
            e_total = ha_paths[0].united(e_out_union) if not e_out_union.isEmpty() else ha_paths[0]
        e_total = e_total.intersected(frame)
        d_total = QPainterPath()
        if dust_paths:
            d_total = dust_paths[0]
        if d_out:
            d_total = d_total.united(d_out[0].simplified()) if not d_total.isEmpty() else d_out[0].simplified()
        if not d_total.isEmpty():
            d_total = d_total.intersected(frame).subtracted(e_total)
        r_total = r_out_union.intersected(frame).subtracted(e_total) if not r_out_union.isEmpty() else r_out_union

        def shade(paths, color, alpha, clip):
            if not paths:
                return
            painter.save()
            painter.setClipPath(clip)
            fc = QColor(color)
            fc.setAlpha(alpha)
            for pth in paths:
                painter.fillPath(pth, QBrush(fc))
            painter.restore()

        def outline(path, color, alpha, width, dashed=False):
            if path.isEmpty():
                return
            sc = QColor(color)
            sc.setAlpha(alpha)
            pen = QPen(sc, width)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            if dashed:
                pen.setDashPattern([3.0, 2.0])
            painter.save()
            painter.setClipRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            painter.strokePath(path, pen)
            painter.restore()

        e_stroke, e_fill = C["halpha"]
        d_stroke, d_fill = C["dust"]
        r_stroke, r_fill = C["reflection"]
        # dark nebulae: below everything, never over an emission region
        if not d_total.isEmpty():
            self.kinds_drawn.add(K_DARK)
            shade([d_total], d_fill, 40, frame)
            outline(d_total, d_stroke, 110, 0.10 * gs, dashed=True)
        # emission: map levels outside the outlined objects, outline levels inside
        if not e_total.isEmpty():
            self.kinds_drawn.add(K_NEB)
            clip_out = frame.subtracted(e_out_union) if not e_out_union.isEmpty() else frame
            shade([e_total], e_fill, 45, frame)
            outline(e_total, e_stroke, 150, 0.12 * gs)
        if not r_total.isEmpty():
            self.kinds_drawn.add(K_RFN)
            shade([r_total], r_fill, 45, frame)
            outline(r_total, r_stroke, 150, 0.12 * gs)

    # ---- grid ----------------------------------------------------------------
    def _grid_steps(self):
        span = max(min(self.fov_w, self.fov_h), 1e-3)
        dec_c = [30, 20, 10, 5, 2, 1, 0.5, 1 / 3, 1 / 6, 1 / 12, 1 / 30, 1 / 60, 1 / 120, 1 / 360]
        dstep = next((s for s in dec_c if s <= span / 4.0), dec_c[-1])
        ra_c = [30, 15, 10, 5, 2.5, 1.25, 0.5, 0.25, 0.125, 1 / 12, 1 / 24, 1 / 48, 1 / 240]
        target = dstep / max(math.cos(self.p.dec * D2R), 0.05)
        rstep = min(ra_c, key=lambda s: abs(math.log(s / target)))
        return dstep, rstep

    @staticmethod
    def _ra_label(ra, step):
        sec = int(round((ra % 360.0) * 240.0)) % 86400
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        runs = [(f"{h}", False), ("h", True)]
        if step < 15 or m or s:
            runs += [(f"{m:02d}", False), ("m", True)]
        if step < 0.25 or s:
            runs += [(f"{s:02d}", False), ("s", True)]
        return runs

    @staticmethod
    def _dec_label(dec, step):
        asec = int(round(abs(dec) * 3600.0))
        d, m, s = asec // 3600, (asec % 3600) // 60, asec % 60
        sg = "+" if dec > 0 else ("−" if dec < 0 else "")
        t = f"{sg}{d}°"
        if step < 1 or m or s:
            t += f"{m:02d}′"
        if step < 1 / 60 or s:
            t += f"{s:02d}″"
        return [(t, False)]

    def _draw_grid(self, painter, gs):
        dstep, rstep = self._grid_steps()
        col, w = self.C["grid"], 0.13 * gs
        if not self.p.marks("grid"):
            col = QColor(0, 0, 0, 0)          # invisible lines, labels still computed
        pad = dstep
        dlo = max(-90.0, self.dec_lo - pad)
        dhi = min(90.0, self.dec_hi + pad)
        full = (self.ra_hi - self.ra_lo) >= 359.9
        # parallels
        k0, k1 = math.ceil(dlo / dstep - 1e-9), math.floor(dhi / dstep + 1e-9)
        for k in range(k0, k1 + 1):
            dec = k * dstep
            if abs(dec) >= 90.0 - 1e-9:
                continue
            if full:
                ra = np.linspace(0, 360, 721)
            else:
                ra = np.linspace(self.ra_lo - rstep, self.ra_hi + rstep, 241)
            cr = []
            self._stroke_sky(painter, ra, np.full_like(ra, dec), col, w, crossings=cr, tag=None)
            lab = self._dec_label(dec, dstep)
            for _t, cx, cy in cr:
                self.edge_labels.append((cx, cy, lab))
        # meridians
        if full:
            ras = np.arange(0.0, 360.0 - 1e-9, rstep)
        else:
            k0 = math.ceil((self.ra_lo - rstep) / rstep)
            k1 = math.floor((self.ra_hi + rstep) / rstep)
            ras = np.arange(k0, k1 + 1) * rstep
        cap = 90.0 - min(10.0, 4.0 * dstep)
        for ra in ras:
            major = abs((ra / 30.0) - round(ra / 30.0)) < 1e-9
            lo, hi = dlo, dhi
            if not major and rstep < 30:
                lo, hi = max(lo, -cap), min(hi, cap)
            if hi <= lo:
                continue
            dec = np.linspace(lo, hi, 181)
            cr = []
            self._stroke_sky(painter, np.full_like(dec, ra), dec, col, w, crossings=cr, tag=None)
            lab = self._ra_label(ra, rstep)
            for _t, cx, cy in cr:
                self.edge_labels.append((cx, cy, lab))

    def _draw_edge_labels(self, painter, ts, gs):
        if self._match is not None:
            return                              # no margin around a matched image
        x0, y0, x1, y1 = self.frame
        size = 2.1 * ts
        tick = 0.9 * gs
        placed = []
        pen = QPen(QColor(self.C["frame"]), 0.18 * gs)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen)
        eps = 1e-4
        for cx, cy, lab in self.edge_labels:
            w = self.T.width(lab, size)
            h = size * 0.74
            if abs(cy - y0) < eps:
                bx, by = cx - w / 2, y0 - tick - 0.5
                t = (cx, y0, cx, y0 - tick)
            elif abs(cy - y1) < eps:
                bx, by = cx - w / 2, y1 + tick + 0.5 + h
                t = (cx, y1, cx, y1 + tick)
            elif abs(cx - x0) < eps:
                bx, by = x0 - tick - 0.5 - w, cy + h / 2
                t = (x0, cy, x0 - tick, cy)
            elif abs(cx - x1) < eps:
                bx, by = x1 + tick + 0.5, cy + h / 2
                t = (x1, cy, x1 + tick, cy)
            else:
                continue
            r = (bx - 0.6, by - h - 0.4, bx + w + 0.6, by + 0.4)
            if r[0] < 0.5 or r[2] > self.pw - 0.5 or r[1] < 0.5 or r[3] > self.ph - 0.5:
                continue
            if any(o[0] < r[2] and o[2] > r[0] and o[1] < r[3] and o[3] > r[1] for o in placed):
                continue
            placed.append(r)
            painter.setPen(pen)
            painter.drawLine(QPointF(t[0], t[1]), QPointF(t[2], t[3]))
            self.T.draw(painter, bx, by, lab, size, self.C["text"])

    # ---- constellations ------------------------------------------------------
    def _draw_connames(self, painter, ts):
        x0, y0, x1, y1 = self.frame
        size = float(np.clip(self.mm_per_deg * 0.9, 3.0, 6.5)) * ts
        placed = []
        drawn = set()
        for cid, (name, ra, dec, _rk) in sorted(self.d.con_names.items(), key=lambda kv: kv[1][3]):
            px, py, th = self.proj.project(ra, dec)
            px, py = float(px), float(py)
            if th >= self.proj.tmax or not self.inside(px, py):
                continue
            txt = name.upper()
            w = self.T.width(txt, size)
            if w > (x1 - x0) * 0.8:
                continue
            bx = min(max(px - w / 2, x0 + 1.0), x1 - 1.0 - w)
            by = min(max(py + size * 0.36, y0 + size), y1 - 1.0)
            r = (bx, by - size * 0.8, bx + w, by + size * 0.1)
            if any(o[0] < r[2] and o[2] > r[0] and o[1] < r[3] and o[3] > r[1] for o in placed):
                continue
            placed.append(r)
            drawn.add(cid[:3])
            self.T.draw(painter, bx, by, txt, size, self.C["conname"])
        # constellations that fill part of the chart but whose usual name position is off
        # it (narrow fields, or a chart across a border): name them inside their own area
        try:
            self._draw_area_connames(painter, size, placed, drawn)
        except Exception as e:
            self.log(f"Constellation names by area not drawn: {e}")

    @staticmethod
    def _pip(px, py, qx, qy):
        """Even-odd point-in-polygon test of many points against one closed ring."""
        inside = np.zeros(len(px), dtype=bool)
        x1, y1, x2, y2 = qx[:-1], qy[:-1], qx[1:], qy[1:]
        for a in range(0, len(x1), 400):
            s = slice(a, a + 400)
            ya, yb, xa, xb = y1[s][None, :], y2[s][None, :], x1[s][None, :], x2[s][None, :]
            Py = py[:, None]
            cond = (ya > Py) != (yb > Py)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = xa + (Py - ya) * (xb - xa) / (yb - ya)
            inside ^= (np.count_nonzero(cond & (px[:, None] < xi), axis=1) % 2).astype(bool)
        return inside

    def con_membership(self, nx=72, ny=48):
        """Constellation of each point of a grid over the chart frame.
        Returns (xs, ys, labels (index into ids or -1), ids)."""
        x0, y0, x1, y1 = self.frame
        gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
        gx, gy = gx.ravel(), gy.ravel()
        ra, dec = self.proj.inverse(gx, gy)
        ok = np.isfinite(ra) & np.isfinite(dec)
        lab = np.full(len(gx), -1, dtype=np.int32)
        if not ok.any():
            return gx, gy, lab, []
        ra0, dec0 = float(self.proj.ra0) * R2D, float(self.proj.dec0) * R2D
        s0, c0 = math.sin(dec0 * D2R), math.cos(dec0 * D2R)
        ids = []
        n_ok = int(np.count_nonzero(ok))
        sub = np.full(n_ok, -1, dtype=np.int32)
        dense = getattr(self.d, "_con_dense", None)
        if dense is None:
            # each border ring in the stereographic plane around its own mean direction,
            # where the constellation (smaller than a hemisphere) is the ring's inside
            dense = []
            for cid, rr in self.d.con_polys.items():
                for ring in rr:
                    ring = np.asarray(ring, dtype=np.float64)
                    if len(ring) < 3:
                        continue
                    if np.any(ring[0] != ring[-1]):
                        ring = np.vstack([ring, ring[:1]])
                    r1, d1 = self._densify(ring, 0.5)
                    v = np.vstack([np.cos(d1 * D2R) * np.cos(r1 * D2R), np.cos(d1 * D2R) * np.sin(r1 * D2R),
                                   np.sin(d1 * D2R)])
                    m = v.mean(axis=1)
                    m /= np.linalg.norm(m)
                    e = np.cross([0.0, 0.0, 1.0], m)
                    if np.linalg.norm(e) < 1e-9:
                        e = np.array([0.0, 1.0, 0.0])
                    e /= np.linalg.norm(e)
                    nn = np.cross(m, e)
                    k = 2.0 / (1.0 + m @ v)
                    rad = float(np.arccos(np.clip((m @ v).min(), -1.0, 1.0)))
                    dense.append((cid, v, m, e, nn, k * (e @ v), k * (nn @ v), rad))
            self.d._con_dense = dense
        cv = np.array([c0 * math.cos(ra0 * D2R), c0 * math.sin(ra0 * D2R), s0])
        pv = np.vstack([np.cos(dec[ok] * D2R) * np.cos(ra[ok] * D2R),
                        np.cos(dec[ok] * D2R) * np.sin(ra[ok] * D2R), np.sin(dec[ok] * D2R)])
        fr = float(np.arccos(np.clip((cv @ pv).min(), -1.0, 1.0)))      # chart radius on the sky

        def plane(u, m, e, nn):
            k = 2.0 / np.maximum(1.0 + m @ u, 1e-6)
            return k * (e @ u), k * (nn @ u)

        for cid, v, m, e, nn, qx, qy, rad in dense:
            dm = float(np.arccos(np.clip(m @ cv, -1.0, 1.0)))
            if dm > rad + fr + 0.01:
                continue                                   # too far to reach the chart
            near = float(np.arccos(np.clip((cv @ v).max(), -1.0, 1.0)))
            if near > fr + 0.01:
                # its border stays off the chart: all of the chart is in it, or none
                cx_, cy_ = plane(cv[:, None], m, e, nn)
                if not self._pip(cx_, cy_, qx, qy)[0]:
                    continue
                ins = sub < 0
            else:
                ux, uy = plane(pv, m, e, nn)
                ins = self._pip(ux, uy, qx, qy) & (sub < 0)
            if ins.any():
                if cid not in ids:
                    ids.append(cid)
                sub[ins] = ids.index(cid)
        lab[np.nonzero(ok)[0]] = sub
        return gx, gy, lab, ids

    def _draw_area_connames(self, painter, size, placed, drawn):
        x0, y0, x1, y1 = self.frame
        nx, ny = 72, 48
        gx, gy, lab, ids = self.con_membership(nx, ny)
        if not ids:
            return
        self.info["constellations"] = [self.d.con_names.get(c, (c,))[0] for c in ids]
        total = max(int(np.count_nonzero(lab >= 0)), 1)
        W, H = x1 - x0, y1 - y0

        def at(xs, ys):                    # constellation index at chart points
            i = np.clip(np.round((np.asarray(xs) - x0) / W * (nx - 1)), 0, nx - 1).astype(int)
            j = np.clip(np.round((np.asarray(ys) - y0) / H * (ny - 1)), 0, ny - 1).astype(int)
            return lab[j * nx + i]

        order = sorted(range(len(ids)), key=lambda n: -np.count_nonzero(lab == n))
        for n in order:
            cid = ids[n]
            if cid[:3] in drawn:
                continue
            sel = lab == n
            if np.count_nonzero(sel) / total < 0.06:
                continue                   # only a sliver of it on the chart
            name = self.d.con_names.get(cid, (cid,))[0]
            if (cid[:3] + "2") in self.d.con_names:
                name = name.split(" ")[0]            # Serpens
            txt = name.upper()
            sz = size
            w = self.T.width(txt, sz)
            if w > W * 0.8:
                sz *= W * 0.8 / w
                w = self.T.width(txt, sz)
            h = sz * 0.8
            cxr = float(gx[sel].mean())
            best = None
            # along the top and bottom edges first (the centre holds the target), then inside
            rows = [(y0 + 2.2 + h, 0), (y1 - 2.0, 0), (y0 + H * 0.3 + h / 2, 1),
                    (y0 + H * 0.7 + h / 2, 1), (y0 + H * 0.5 + h / 2, 2)]
            xs_c = np.linspace(x0 + 1.2, x1 - 1.2 - w, 48)
            for by, pen in rows:
                for bx in xs_c:
                    r = (bx, by - h, bx + w, by + sz * 0.1)
                    if any(o[0] < r[2] and o[2] > r[0] and o[1] < r[3] and o[3] > r[1] for o in placed):
                        continue
                    sx, sy = np.meshgrid(np.linspace(r[0], r[2], 6), np.linspace(r[1], r[3], 3))
                    frac = float(np.mean(at(sx.ravel(), sy.ravel()) == n))
                    if frac < 0.6:
                        continue
                    key = (frac < 0.999, pen, abs(bx + w / 2 - cxr) / W)
                    if best is None or key < best[0]:
                        best = (key, bx, by, r)
            if best is None:
                continue
            _k, bx, by, r = best
            placed.append(r)
            drawn.add(cid[:3])
            self.placer.add(r, ("c", cid), hard=False)
            # a little darker than the usual names: these often sit over the Milky Way
            self.T.draw(painter, bx, by, txt, sz, QColor(self.C["conname"]).darker(145).name())

    def _draw_conlines(self, painter, stars, gs):
        # star radii lookup so that lines stop short of the star discs
        cell = 2.0
        grid = {}
        big = np.nonzero(stars["r"] > 0.3)[0]
        for i in big:
            grid.setdefault((int(stars["x"][i] // cell), int(stars["y"][i] // cell)), []).append(i)

        def radius_at(px, py):
            best, bd = 0.0, 0.9
            ci, cj = int(px // cell), int(py // cell)
            for a in (ci - 1, ci, ci + 1):
                for b in (cj - 1, cj, cj + 1):
                    for i in grid.get((a, b), ()):
                        dd = math.hypot(stars["x"][i] - px, stars["y"][i] - py)
                        if dd < bd:
                            bd, best = dd, float(stars["r"][i])
            return best

        path = QPainterPath()
        x0, y0, x1, y1 = self.frame
        for ln in self.d.con_lines:
            x, y, th = self.proj.project(ln[:, 0], ln[:, 1])
            for k in range(len(x) - 1):
                if th[k] >= self.proj.tmax or th[k + 1] >= self.proj.tmax:
                    continue
                xa, ya, xb, yb = float(x[k]), float(y[k]), float(x[k + 1]), float(y[k + 1])
                ln_len = math.hypot(xb - xa, yb - ya)
                if ln_len < 1e-6 or ln_len > 4 * max(self.pw, self.ph):
                    continue
                ga = (radius_at(xa, ya) + 0.55 * gs) if self.inside(xa, ya) else 0.0
                gb = (radius_at(xb, yb) + 0.55 * gs) if self.inside(xb, yb) else 0.0
                if ga + gb >= ln_len:
                    continue
                ux, uy = (xb - xa) / ln_len, (yb - ya) / ln_len
                c = _clip_seg(xa + ux * ga, ya + uy * ga, xb - ux * gb, yb - uy * gb, x0, y0, x1, y1)
                if c is None:
                    continue
                path.moveTo(c[0], c[1])
                path.lineTo(c[2], c[3])
        pen = QPen(QColor(self.C["line"]), 0.26 * gs)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.strokePath(path, pen)

    # ---- stars ---------------------------------------------------------------
    def _star_radius(self, mag):
        P = self.p
        rmin, rmax = P.star_min / 2.0 * P.graphics_scale, P.star_max / 2.0 * P.graphics_scale
        mref = max(-1.5, self.star_mag - 9.5)
        t = np.clip((self.star_mag - np.asarray(mag, dtype=np.float64)) / max(self.star_mag - mref, 1e-6), 0, 1)
        return rmin + (rmax - rmin) * t ** 1.35

    def _select_stars(self):
        d, L = self.d, self.p.layers
        empty = dict(x=np.zeros(0), y=np.zeros(0), r=np.zeros(0), mag=np.zeros(0), ci=np.zeros(0),
                     idx=np.zeros(0, dtype=int))
        if not L.get("stars") and not L.get("gaia"):
            return empty
        P = self.p
        density_auto = P.star_mag_auto
        final_mag = self.star_mag
        m = d.s_mag <= self.star_mag
        if not L.get("stars"):
            # Gaia only: keep no bright-catalog star, but let Gaia fill the whole range
            m = np.zeros(len(d.s_mag), dtype=bool)
        idx = np.nonzero(m)[0]
        x, y, th = self.proj.project(d.s_ra[idx], d.s_dec[idx])
        keep = self.inside(x, y) & (th < self.proj.tmax)
        idx, x, y = idx[keep], x[keep], y[keep]
        mag = d.s_mag[idx].astype(np.float64)
        ci = d.s_ci[idx].astype(np.float64)
        if L.get("gaia") and self.star_mag > GAIA_MIN_MAG:
            g = self._query_gaia(x, y)
            if g is not None and len(g[0]):
                gx, gy, gm, gsid = g
                gci = np.full(len(gx), np.nan)
                if self.p.colored_stars and not self.p.black_white:
                    gci = self.d.xp_colors(gsid, gm)
                    n_xp = int(np.count_nonzero(~np.isnan(gci)))
                    miss = np.isnan(gci)
                    if self.p.online_colors and miss.any():
                        try:
                            gci[miss] = self.d.bprp_colors(gsid[miss])
                            miss = np.isnan(gci)
                            want = self.d.bprp_missing(gsid[miss], gm[miss]) if miss.any() else {}
                            if want and len(want) <= 48:
                                if self.wait_online:
                                    n = self.d.fetch_bprp(want)
                                    self.log(f"Gaia archive: colours for {n} stars downloaded")
                                    gci[miss] = self.d.bprp_colors(gsid[miss])
                                else:
                                    self.info["bprp_pending"] = want
                        except Exception as e:
                            self.log(f"Online Gaia colours not available: {e}")
                    self.info["gaia_xp"] = n_xp
                self.info["gaia"] = len(gx)
                self.info["gaia_colored"] = int(np.count_nonzero(~np.isnan(gci)))
                x = np.concatenate([x, gx])
                y = np.concatenate([y, gy])
                mag = np.concatenate([mag, gm])
                ci = np.concatenate([ci, gci])
                idx = np.concatenate([idx, np.full(len(gx), -1, dtype=idx.dtype)])
        if density_auto:
            x0, y0, x1, y1 = self.frame
            # the limit comes from the field of view; the density is only a ceiling for
            # very rich fields (Milky Way star clouds), so the chart never gets crowded
            n_target = max(30, int(P.star_density * (x1 - x0) * (y1 - y0) / 100.0))
            srt = np.sort(mag)
            final_mag = float(srt[n_target - 1]) if len(srt) > n_target else self.star_mag
            final_mag = float(np.clip(final_mag, 3.0, 17.0))
            sel = mag <= final_mag
            x, y, mag, ci, idx = x[sel], y[sel], mag[sel], ci[sel], idx[sel]
            self.star_mag = final_mag
            if P.dso_mag_auto:
                self.dso_mag = min(self.star_mag + 3.5, 20.0)
            if "gaia" in self.info:
                self.info["gaia"] = int(np.count_nonzero(idx < 0))
        rad = self._star_radius(mag)
        hyg = idx >= 0
        if hyg.any() and (~hyg).any():
            try:
                from scipy.spatial import cKDTree
                big = hyg & (rad > 0.25)
                if big.any():
                    tree = cKDTree(np.column_stack([x[big], y[big]]))
                    dist, j = tree.query(np.column_stack([x[~hyg], y[~hyg]]))
                    covered = dist < rad[big][j] * 1.15 + 0.08
                    drop = np.zeros(len(x), dtype=bool)
                    drop[np.nonzero(~hyg)[0][covered]] = True
                    x, y, mag, ci, idx, rad = x[~drop], y[~drop], mag[~drop], ci[~drop], idx[~drop], rad[~drop]
            except Exception:
                pass
        order = np.argsort(-mag, kind="stable")       # faint first, bright on top
        x, y, mag, ci, idx, rad = x[order], y[order], mag[order], ci[order], idx[order], rad[order]
        return dict(x=x, y=y, r=rad, mag=mag, ci=ci, idx=idx)

    def _marked_stars(self, s):
        P = self.p
        keep = np.ones(len(s["x"]), dtype=bool)
        if not P.marks("stars"):
            keep &= s["idx"] < 0
        if not P.marks("gaia"):
            keep &= s["idx"] >= 0
        return {k: v[keep] for k, v in s.items()}

    def _gaia_dir(self):
        return gaia_library_dir()

    def _query_gaia(self, hx, hy):
        """Stars fainter than GAIA_MIN_MAG from the local Gaia DR3 library, minus
        the ones already supplied by the bright-star catalog. Narrow fields are
        read straight from the library; wide fields use the all-sky cache."""
        gdir = self._gaia_dir()
        if gdir is None:
            return None
        glo, ghi = (GAIA_MIN_MAG if self.p.layers.get("stars") else -5.0), self.star_mag
        area = (self.dec_hi - self.dec_lo) * min(360.0, self.ra_hi - self.ra_lo) * max(
            math.cos(self.p.dec * D2R), 0.05)
        t0 = time.time()
        if self.all_sky or area > 250.0:
            g = self.d.gaia_bright(gdir)
            if g is None:
                return None
            if ghi > GAIA_CACHE_MAG:
                self.info["gaia_capped"] = True
                self.log(f"Wide field: Gaia stars limited to G = {GAIA_CACHE_MAG}.")
            ra, dec, mg, sid = g
            sel = (mg >= glo) & (mg <= ghi)
            if not self.all_sky:
                sel &= (dec >= self.dec_lo - 0.5) & (dec <= self.dec_hi + 0.5)
            ra, dec, mg, sid = ra[sel], dec[sel], mg[sel].astype(np.float64), sid[sel]
        else:
            ra_lo, ra_hi = (0.0, 360.0) if (self.ra_hi - self.ra_lo) >= 359.9 else (self.ra_lo, self.ra_hi)
            ra, dec, mg, sid = gaia_box(ra_lo, ra_hi, self.dec_lo, self.dec_hi, glo, ghi)
            if not len(ra):
                return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64)
        x, y, th = self.proj.project(ra, dec)
        keep = self.inside(x, y) & (th < self.proj.tmax)
        x, y, mg, sid = x[keep], y[keep], mg[keep], sid[keep]
        # drop Gaia sources that coincide with a bright-catalog star on the chart
        if len(hx) and len(x):
            tol = max(0.10, 5.0 / 3600.0 * self.mm_per_deg)
            cell = tol * 2.0
            big = 1 << 20
            hi_ = np.floor(hx / cell).astype(np.int64)
            hj_ = np.floor(hy / cell).astype(np.int64)
            occ = np.unique(np.concatenate([(hi_ + di) * big + (hj_ + dj)
                                            for di in (-1, 0, 1) for dj in (-1, 0, 1)]))
            gk = np.floor(x / cell).astype(np.int64) * big + np.floor(y / cell).astype(np.int64)
            sel = ~np.isin(gk, occ)
            x, y, mg, sid = x[sel], y[sel], mg[sel], sid[sel]
        if len(x) > 80000:
            o = np.argsort(mg)[:80000]
            x, y, mg, sid = x[o], y[o], mg[o], sid[o]
            self.log("Gaia layer limited to the 80000 brightest stars in the field.")
        self.log(f"Gaia DR3 (local): {len(x)} stars in {time.time() - t0:.1f} s")
        return x, y, mg, sid

    def _double_flags(self, s):
        """Which of the drawn stars are the primary of a double star (WDS): a pair
        within 12" whose primary magnitude matches the star's."""
        d = self.d
        n = len(s["x"])
        out = np.zeros(n, dtype=bool)
        if getattr(d, "w_ra", None) is None:
            idx = s["idx"]
            ok = idx >= 0
            out[ok] = d.s_dbl[idx[ok]]          # fallback: HYG multiple systems
            return out
        if not n:
            return out
        x, y, th = self.proj.project(d.w_ra, d.w_dec)
        ok = self.inside(x, y) & (th < self.proj.tmax)
        if not ok.any():
            return out
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return out
        wx, wy, wm = x[ok], y[ok], d.w_mag[ok]
        tol = 12.0 / 3600.0 * self.mm_per_deg
        tree = cKDTree(np.column_stack([wx, wy]))
        for i, js in enumerate(tree.query_ball_point(np.column_stack([s["x"], s["y"]]), tol)):
            if js and np.any(np.abs(wm[js] - s["mag"][i]) < 1.5):
                out[i] = True
        return out

    def _draw_stars(self, painter, s, gs):
        P, C, d = self.p, self.C, self.d
        L = P.layers
        n = len(s["x"])
        if n == 0:
            return
        halo_w = P.halo_width * gs
        hc = QColor(C["halo"])
        if not P.halo_map_bg:
            hc.setAlpha(int(P.halo_alpha))
        halo = QBrush(hc)
        outline_pen = None
        if P.outlined_stars:
            outline_pen = QPen(QColor(P.star_outline_color), P.star_outline_width * gs)
        mark_var, mark_dbl = P.marks("variable"), P.marks("double")
        dbl_flags = self._double_flags(s) if mark_dbl else None
        self._gs_last = gs
        brushes = {}
        var_rings = []
        painter.setPen(Qt.PenStyle.NoPen)
        X, Y, Rr, CI, IDX = s["x"], s["y"], s["r"], s["ci"], s["idx"]
        for i in range(n):
            x, y, r = float(X[i]), float(Y[i]), float(Rr[i])
            col = star_color(float(CI[i])) if (P.colored_stars and not P.black_white) else C["star_none"]
            b = brushes.get(col)
            if b is None:
                b = brushes[col] = QBrush(QColor(col))
            hi = int(IDX[i])
            is_var = mark_var and hi >= 0 and bool(d.s_var[hi])
            is_dbl = dbl_flags is not None and bool(dbl_flags[i])
            if is_dbl:
                self.has_dbl = True
                ext = r + 0.75 * gs
                if P.halos:
                    painter.setPen(QPen(hc, 0.18 * gs + 2 * halo_w))
                    painter.drawLine(QPointF(x - ext, y), QPointF(x + ext, y))
                painter.setPen(QPen(QColor(col), 0.18 * gs))
                painter.drawLine(QPointF(x - ext, y), QPointF(x + ext, y))
                painter.setPen(Qt.PenStyle.NoPen)
            if is_var:
                self.has_var = True
                var_rings.append((x, y, r, col))
            if P.halos:
                painter.setBrush(halo)
                painter.drawEllipse(QPointF(x, y), r + halo_w, r + halo_w)
            painter.setBrush(b)
            if outline_pen is not None:
                painter.setPen(outline_pen)
            painter.drawEllipse(QPointF(x, y), r, r)
            if outline_pen is not None:
                painter.setPen(Qt.PenStyle.NoPen)
        # variable stars: a ring around the star, on top of the neighbouring stars
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for x, y, r, col in var_rings:
            rr = max(r + max(halo_w, 0.08) + 0.30 * gs, 0.55 * gs)
            w = 0.13 * gs
            painter.setPen(QPen(QColor(col), w))
            painter.drawEllipse(QPointF(x, y), rr, rr)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    # ---- deep-sky objects ------------------------------------------------------
    def _select_dsos(self):
        L, d = self.p.layers, self.d
        rows = []
        want = {"Messier": L.get("messier"), "NGC": L.get("ngc"), "IC": L.get("ic"), "Other": L.get("other")}
        if any(want.values()):
            rows += [r for r in d.dsos if want.get(r[0])]
        for key, cat in (("sharpless", "Sharpless"), ("lbn", "LBN"), ("ldn", "LDN"),
                         ("abell", "Abell"), ("png", "PNG")):
            if L.get(key):
                rows += d.extra.get(cat, [])
        if not rows:
            return []
        ra = np.array([r[2] for r in rows])
        dec = np.array([r[3] for r in rows])
        x, y, th = self.proj.project(ra, dec)
        keep = np.nonzero(self.inside(x, y) & (th < self.proj.tmax))[0]
        out = []
        rmin = self.p.dso_min / 2.0 * self.p.graphics_scale
        for i in keep:
            layer, label, ra_i, dec_i, mag, maj, mnr, pa, kind, common, _al = rows[i]
            a_mm = (maj / 120.0 * self.mm_per_deg) if maj == maj else 0.0      # semi-major, mm
            if mag == mag:
                if mag > self.dso_mag and layer != "Messier":
                    continue
            elif (layer != "Messier" and a_mm < 0.9 * rmin and not common
                  and not any(a in self.outlined for a in _al)):
                continue
            has_outline = any(a in self.outlined for a in _al)
            out.append(dict(layer=layer, label=label, ra=ra_i, dec=dec_i, x=float(x[i]), y=float(y[i]),
                            mag=mag, maj=maj, mnr=mnr, pa=pa, kind=kind, a=a_mm, outlined=has_outline,
                            common=common))
        for o in out:
            if o["layer"] == "Sharpless":
                o["label"] = re.sub(r"^SH2-", "Sh2-", o["label"])
            elif o["layer"] in ("LBN", "LDN"):
                o["label"] = re.sub(r"^(LBN|LDN)\s*", r"\1 ", o["label"])
        if self.p.remove_duplicates:
            out = self._dedupe_dsos(out)
        # Nebulae are drawn as shapes by the nebula layers, never as boxes:
        #  - with a real outline (Nebula Outlines layer): label only;
        #  - emission / dark nebulae resolved by the H-alpha / dust maps: label only;
        #  - nebulae too small for the maps (< ~2 map pixels): ellipse at true size;
        #  - otherwise (its layer is off, or the map shows nothing): not shown.
        L = self.p.layers
        pix_mm = 6.87 / 60.0 * self.mm_per_deg
        for o in out:
            o["mapped"] = False
            o["hide"] = False
            if o["outlined"]:
                continue
            k = o["kind"]
            if k in (K_NEB, K_CLNEB, K_DARK):
                is_dark = k == K_DARK
                layer_on = L.get("dust") if is_dark else L.get("halpha")
                look = self.dust_lookup if is_dark else self.ha_lookup
                if not layer_on or look is None:
                    o["hide"] = True                     # its nebula layer is off
                    continue
                visible = look(o["x"], o["y"]) >= (NEB_DUST_LEVEL if is_dark else NEB_HA_LEVEL)
                small = 2.0 * o["a"] < 2.0 * pix_mm
                if visible:
                    o["mapped"] = True
                elif small and o["layer"] in ("Messier", "NGC", "IC", "Other"):
                    pass                                 # compact: ellipse symbol
                else:
                    o["hide"] = True
            elif k == K_RFN and not L.get("outlines"):
                o["hide"] = True
        out = [o for o in out if not o["hide"]]
        for o in out:
            # Lynds / Sharpless entries crowd wide fields: label only those that are
            # not tiny on the chart
            if o["mapped"] and o["layer"] in ("LBN", "LDN", "Sharpless"):
                if 2.0 * o["a"] < (1.5 if o["layer"] == "Sharpless" else 4.0):
                    o["skip_label"] = True
        out = [o for o in out if not o.get("skip_label")]
        # Messier first, then brighter / larger
        out.sort(key=lambda o: (0 if o["layer"] == "Messier" else 1,
                                o["mag"] if o["mag"] == o["mag"] else 99.0, -o["a"]))
        if len(out) > 2500:
            self.log(f"DSO layer limited to 2500 of {len(out)} objects.")
            out = out[:2500]
        # on-chart geometry
        for o in out:
            a = max(o["a"], rmin)
            ratio = (o["mnr"] / o["maj"]) if (o["maj"] == o["maj"] and o["maj"] > 0 and o["mnr"] == o["mnr"]) else 1.0
            ratio = float(np.clip(ratio, 0.12, 1.0))
            ang = 0.0
            if o["kind"] in (K_GALAXY, K_NEB, K_RFN, K_DARK, K_CLNEB) and o["pa"] == o["pa"] and ratio < 0.98:
                r1, d1 = sky_offset(o["ra"], o["dec"], o["pa"], 0.02)
                px, py, _ = self.proj.project(r1, d1)
                ang = math.degrees(math.atan2(float(py) - o["y"], float(px) - o["x"]))
                b = max(a * ratio, min(rmin * 0.55, a))
            elif o["kind"] in (K_GALAXY,):
                a = max(a * math.sqrt(ratio), rmin)
                b = max(a * 0.999, rmin * 0.55) if ratio > 0.98 else a
                a, b = a, b
            elif o["kind"] in (K_NEB, K_RFN, K_DARK, K_CLNEB):
                b = max(a * ratio, min(rmin, a))
            else:
                b = a
            o["sa"], o["sb"], o["ang"] = a, b, ang
        return out

    _LAYER_PRIO = {"Messier": 0, "NGC": 1, "IC": 2, "Other": 3, "Sharpless": 4, "LBN": 5,
                   "LDN": 6, "PNG": 7, "Abell": 8}
    _FAMILY = {K_NEB: "n", K_CLNEB: "n", K_RFN: "n", K_DARK: "d", K_PN: "p", K_OC: "c",
               K_GC: "c", K_ASSOC: "c", K_GALAXY: "g", K_GROUP: "g"}

    def _dedupe_dsos(self, objs):
        """The same nebula is listed by several catalogs (NGC 281 = Sh2-184 =
        LBN 616...). Keep the best-known designation and drop the copies."""
        if len(objs) < 2:
            return objs
        objs = sorted(objs, key=lambda o: (self._LAYER_PRIO.get(o["layer"], 9), -o["a"]))
        kept = []
        kx, ky, kr, kf, kl = [], [], [], [], []
        for o in objs:
            fam = self._FAMILY.get(o["kind"], "x")
            if fam == "c" and o["kind"] == K_CLNEB:
                fam = "n"
            if kx:
                dx = np.array(kx) - o["x"]
                dy = np.array(ky) - o["y"]
                dist = np.hypot(dx, dy)
                rr = np.array(kr)
                # same family, other catalog, centres closer than ~40% of the larger size
                tol = np.maximum(np.maximum(0.4 * np.maximum(rr, o["a"]), 4.0 / 60.0 * self.mm_per_deg), 0.6)
                same = (np.array(kf) == fam) & (np.array(kl) != o["layer"]) & (dist < tol)
                if fam == "n":       # an emission region inside a big one of another catalog
                    same |= (np.array(kf) == "n") & (np.array(kl) != o["layer"]) & (dist < 0.5 * rr) & \
                            (o["a"] < rr * 1.2) & (o["a"] > 0.25 * rr)
                if same.any():
                    continue
            kept.append(o)
            kx.append(o["x"]); ky.append(o["y"]); kr.append(max(o["a"], 0.1)); kf.append(fam)
            kl.append(o["layer"])
        return kept

    def _symbol(self, painter, kind, x, y, a, b, ang, gs, clip=True):
        """Draw one DSO symbol with semi-axes a, b (mm) rotated by ang (deg)."""
        lt, stroke, fill = dso_style(self.p, kind)
        w = 0.20 * gs
        pen = QPen(stroke, w)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        pen_dash(pen, lt, w)
        shape = QPainterPath()
        shape.addEllipse(QPointF(0, 0), a, b)
        tr = QTransform().translate(x, y).rotate(ang)
        shape = tr.map(shape)
        x0, y0, x1, y1 = self.frame
        ext = max(a, b) + w
        clipped = clip and not (x - ext >= x0 and x + ext <= x1 and y - ext >= y0 and y + ext <= y1)
        if clipped:
            fr = QPainterPath()
            fr.addRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            shape = shape.intersected(fr)
        big = max(a, b) > 5.0 and kind in (K_NEB, K_RFN, K_CLNEB)
        if fill is not None:
            fc = QColor(fill)
            if big:                      # large nebulae must not hide the field
                fc.setAlpha(min(fc.alpha(), 40))
            painter.fillPath(shape, QBrush(fc))
        painter.strokePath(shape, pen)
        if clipped:
            return
        solid = QPen(stroke, w)
        solid.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(solid)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if kind == K_GC:
            painter.drawLine(QPointF(x - a, y), QPointF(x + a, y))
            painter.drawLine(QPointF(x, y - a), QPointF(x, y + a))
        elif kind == K_PN:
            t = max(0.7 * gs, 0.45 * a)
            for ux, uy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                painter.drawLine(QPointF(x + ux * a, y + uy * a), QPointF(x + ux * (a + t), y + uy * (a + t)))
        elif kind == K_CLNEB:
            r = 0.62 * min(a, b)
            if not big:
                painter.setBrush(QBrush(QColor(self.C["map"])))
            painter.drawEllipse(QPointF(x, y), r, r)
            painter.setBrush(Qt.BrushStyle.NoBrush)

    _CAT_LAYER = {"Messier": "messier", "NGC": "ngc", "IC": "ic", "Other": "other",
                  "Sharpless": "sharpless", "LBN": "lbn", "LDN": "ldn", "Abell": "abell", "PNG": "png"}

    def _draw_dsos(self, painter, dsos, gs):
        # large objects first so that small ones remain visible on top
        for o in sorted(dsos, key=lambda o: -o["sa"]):
            if o["outlined"] or o.get("mapped"):
                self.kinds_drawn.add(K_NEB if o["kind"] == K_CLNEB else o["kind"])
                continue                 # its real shape is already on the chart
            if not self.p.marks(self._CAT_LAYER.get(o["layer"], "ngc")):
                continue
            self._symbol(painter, o["kind"], o["x"], o["y"], o["sa"], o["sb"], o["ang"], gs)
            self.kinds_drawn.add(o["kind"])

    # ---- labels ---------------------------------------------------------------
    def _bayer_runs(self, i):
        d = self.d
        b = str(d.s_bayer[i])
        if b:
            m = re.match(r"^([A-Za-z]+)[-\s]?(\d*)$", b)
            if m:
                g = GREEK.get(m.group(1).lower())
                if g:
                    return [(g, False)] + ([(m.group(2), True)] if m.group(2) else [])
            return [(b, False)]
        f = str(d.s_flam[i])
        return [(f, False)] if f else None

    def _catalog_designation(self, i):
        """Label for a star without Bayer letter or Flamsteed number (narrow fields):
        its variable-star name ("P Cyg", "V1770 Cyg"), else HD, else HIP."""
        d = self.d
        v = str(d.s_varname[i]) if hasattr(d, "s_varname") else ""
        con = str(d.s_con[i])
        if v and con and v.lower()[:3] not in GREEK:
            return [(f"{v} {con}", False)]
        hd = int(d.s_hd[i]) if hasattr(d, "s_hd") else 0
        if hd > 0:
            return [(f"HD {hd}", False)]
        hip = int(d.s_hip[i])
        return [(f"HIP {hip}", False)] if hip > 0 else None

    def _draw_labels(self, painter, stars, dsos, ts):
        P, C, d, L = self.p, self.C, self.d, self.p.layers
        pl = self.placer
        # double stars: their bar is part of the symbol, labels keep off it
        bar = np.zeros(len(stars["x"]))
        if P.marks("double") and len(stars["x"]):
            gs = getattr(self, "_gs_last", P.graphics_scale)
            bar = np.where(self._double_flags(stars), stars["r"] + 0.75 * gs, 0.0)
        # obstacles: star discs and DSO symbols
        if not P.overlap_stars:
            for k in np.nonzero((stars["r"] > 0.45) | (bar > 0))[0]:
                x, y, r = float(stars["x"][k]), float(stars["y"][k]), float(stars["r"][k])
                e = max(r, float(bar[k]))
                pl.add((x - e, y - r, x + e, y + r), ("s", int(k)), hard=False)
        for n, o in enumerate(dsos):
            if o["outlined"] or o.get("mapped"):
                continue
            e = min(max(o["sa"], o["sb"]), 6.0)
            pl.add((o["x"] - e, o["y"] - e, o["x"] + e, o["y"] + e), ("d", n), hard=False)
        # every star and symbol, for leader lines (which must not cross any of them)
        mx = [float(v) for v in stars["x"]]
        my = [float(v) for v in stars["y"]]
        mr = [float(v) + 0.12 for v in stars["r"]]
        mo = [("s", k) for k in range(len(stars["x"]))]
        for n, o in enumerate(dsos):
            if o["outlined"] or o.get("mapped"):
                continue
            mx.append(float(o["x"])); my.append(float(o["y"]))
            mr.append(float(min(max(o["sa"], o["sb"]), 6.0)))
            mo.append(("d", n))
        pl.set_marks(mx, my, mr, mo)
        todo = []     # (priority, mag, x, y, rad, text, size, bold, owner, must, alternatives)
        halo = C["halo"]
        # how much a label says depends on the field: wide fields keep the short forms
        # ("6888", "γ", "34"); narrow ones spell them out ("NGC 6888", "γ Cyg", "34 Cyg")
        # and add the popular name of the object when there is room
        fov = max(self.fov_w, 1e-6)
        narrow = fov <= 10.0
        with_common = fov <= 6.0
        # deep sky
        s_m, s_n = 2.5 * ts, 2.15 * ts
        for n, o in enumerate(dsos):
            if not P.labels(self._CAT_LAYER.get(o["layer"], "ngc")):
                continue
            mes = o["layer"] == "Messier"
            size = s_m if mes else s_n
            rad = min(max(o["sa"], o["sb"]) * (0.75 if o["ang"] else 1.0), 5.0)
            if o["outlined"] or o.get("mapped"):
                rad = 0.6
            pr = 0 if mes else (2 if o["layer"] in ("NGC", "IC", "Other", "Sharpless") else 4)
            # always label Messier objects, and NGC/IC objects that are big on the chart
            # or drawn with their own outline (in rich star fields they may cover stars)
            must = mes or (o["layer"] in ("NGC", "IC") and
                           (o["outlined"] or o.get("mapped") or max(o["sa"], o["sb"]) >= 2.0))
            text = str(o["label"])
            if narrow and o["layer"] == "NGC" and text[:1].isdigit():
                text = f"NGC {text}"
            elif narrow and o["layer"] == "IC" and re.match(r"^I(C)?\s*\d", text):
                text = "IC " + re.sub(r"^I(C)?\s*", "", text)
            alts = [text]
            common = str(o.get("common") or "").split(",")[0].strip()
            if with_common and common and common.lower() not in text.lower() and \
                    o["layer"] in ("Messier", "NGC", "IC", "Other"):
                alts = [f"{text}  {common}", text]
            todo.append((pr, o["mag"] if o["mag"] == o["mag"] else 30.0, o["x"], o["y"], rad,
                         text, size, False, ("d", n), must, alts))
        # stars
        chosen = set(P.star_names or [])
        if len(stars["x"]) and P.labels("stars"):
            # limits follow the field, not only the star limit chosen by the user, so that
            # a deep star limit on a wide field does not bury the chart in labels
            m_ref = min(self.star_mag, fov_star_mag(fov) + 0.5)
            name_lim = min(m_ref - 4.6, 4.2)
            bay_lim = min(m_ref - 2.4, 6.6)
            flam_lim = min(m_ref - 3.4, 5.6)
            lim = max(bay_lim, name_lim, 99.0 if chosen else -99.0)
            # the main stars of the field are always labelled, whatever their magnitude:
            # their number grows with the map area (a few on a small map, about a dozen on A4)
            x0, y0, x1, y1 = self.frame
            n_main = int(np.clip((x1 - x0) * (y1 - y0) / 100.0 / 40.0, 4, 12))
            hy = np.nonzero((stars["idx"] >= 0) & self.inside(stars["x"], stars["y"], 1.0))[0]
            main = set()
            if len(hy):
                srt = hy[np.argsort(stars["mag"][hy], kind="stable")]
                main = {int(k) for k in srt[:n_main] if stars["mag"][k] <= max(m_ref - 3.0, 4.0)}
            cand = set(np.nonzero(stars["mag"] <= lim)[0].tolist()) | main
            for k in sorted(cand, key=lambda k: -float(stars["mag"][k])):
                hi = int(stars["idx"][k])
                if hi < 0:
                    continue
                mag = float(stars["mag"][k])
                is_main = k in main
                x, y, r = float(stars["x"][k]), float(stars["y"][k]), float(stars["r"][k])
                if bar[k] > 0:                         # beyond the double-star bar
                    r = max(r, float(bar[k]) + 0.4 * ts)
                nm = str(d.s_name[hi])
                if P.labels("starnames") and nm and ((nm in chosen) if chosen else
                                                     (mag <= name_lim or (is_main and narrow))):
                    todo.append((1, mag, x, y, r, nm, 2.35 * ts, False, ("s", int(k)),
                                 mag < 2.0 or is_main, None))
                if P.labels("bayer") and (mag <= max(bay_lim, flam_lim) or is_main):
                    runs = self._bayer_runs(hi)
                    if runs and not ((str(d.s_bayer[hi]) and mag <= bay_lim) or mag <= flam_lim or is_main):
                        runs = None
                    con = str(d.s_con[hi])
                    if runs and narrow and con:
                        runs = list(runs) + [(" " + con, False)]
                    if not runs and narrow and is_main and P.catalog_ids:
                        runs = self._catalog_designation(hi)
                    if runs:
                        todo.append((3, mag, x, y, r, runs, 2.25 * ts, False, ("s", int(k)), is_main, None))
        todo.sort(key=lambda t: (t[0], t[1]))
        lead_pen = QPen(QColor(P.leader_color), P.leader_width * P.graphics_scale)
        lead_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        hal = halo if P.outlined_labels else None
        lead_halo = QPen(QColor(halo), P.leader_width * P.graphics_scale + 0.22 * ts)
        lead_halo.setCapStyle(Qt.PenCapStyle.FlatCap)
        for pr, _mag, x, y, rad, text, size, bold, owner, must, alts in todo:
            placed = None
            sizes = (size, size * P.reduction) if 0.2 < P.reduction < 0.999 else (size,)
            alts = alts or [text]
            # the long form (with the popular name) first, where there is room for it
            tries = [(t, sz, False) for t in alts[:-1] for sz in sizes]
            if must:
                tries += [(t, sz, True) for t in alts[:-1] for sz in sizes]
            tries += [(alts[-1], sz, must) for sz in sizes]
            for txt, sz, mu in tries:
                w = self.T.width(txt, sz, bold)
                h = sz * 0.74
                pos = pl.place(x, y, rad, w, h, owner,
                               must=True if (not P.label_opt and txt is alts[-1]) else mu)
                if pos is not None:
                    placed = (pos, sz, None)
                    break
                if P.leader_lines and txt is alts[-1]:
                    far = pl.place_far(x, y, rad, w, h, owner)
                    if far is not None:
                        placed = (far[:2], sz, far[2:])
                        break
            if placed is None:
                continue
            text = txt
            (lx, ly), sz, lead = placed
            if lead is not None:
                # from just outside the object to the middle of the label's near side,
                # on a thin halo so that it reads over the Milky Way and the nebulae
                ax, ay, sx, sy = lead
                if hal is not None:
                    painter.setPen(lead_halo)
                    painter.drawLine(QPointF(sx, sy), QPointF(ax, ay))
                painter.setPen(lead_pen)
                painter.drawLine(QPointF(sx, sy), QPointF(ax, ay))
            self.T.draw(painter, lx, ly, text, sz, C["label"], bold=bold, halo=hal, halo_w=0.55 * ts)

    def _symbol_box(self, painter, kind, cx, cy, r, gs):
        """Legend key for nebulae: a small filled square in the style colours."""
        lt, line, fill = dso_style(self.p, kind)
        pen = QPen(line, 0.18 * gs)
        pen_dash(pen, lt, 0.18 * gs)
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        if fill is not None:
            painter.fillRect(rect, QBrush(fill))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        if kind == K_CLNEB:
            painter.drawEllipse(QPointF(cx, cy), 0.6 * r, 0.6 * r)

    # ---- legend ---------------------------------------------------------------
    def _draw_legend(self, painter, ts, gs):
        P, C = self.p, self.C
        fam = P.font or QApplication.font().family()
        TT = TextEngine(P.title_font or fam, P.text_as_paths)
        T = TextEngine(P.text_font or fam, P.text_as_paths)
        x0, y0f, x1, y1 = self.frame
        pad = 5.0 * ts if P.layers.get("grid", True) else 0.0
        if P.legend_pos == "Top":
            top = P.margin
        else:
            top = y1 + pad + 1.5 * ts
        painter.setPen(QPen(QColor(C["frame"]), 0.12 * gs))
        ly = top if P.legend_pos != "Top" else top + self.legend_h - 1.5 * ts
        painter.drawLine(QPointF(x0, ly), QPointF(x1, ly))
        if P.legend_pos == "Top":
            top = top - 1.5 * ts
        col = C["text"]
        tsz = P.title_size * 0.3528 * ts           # points -> mm
        s = P.text_size * 0.3528 * ts
        tb, ti = P.title_bold, P.title_italic
        bb, bi = P.text_bold, P.text_italic
        # --- left block: title and chart data
        title = P.title.strip() or self.d.constellation_at(P.ra, P.dec)
        y = top + 1.5 * ts + tsz
        TT.draw(painter, x0, y, title, tsz, col, bold=tb, italic=ti)
        lines = [
            f"Center α {fmt_ra(P.ra)}  δ {fmt_dec(P.dec)}",
            f"Field {self.fov_w:.4g}° × {self.fov_h:.4g}°, 1° = {self.mm_per_deg:.3g} mm",
            f"{self.proj.kind} projection" + (", mirrored" if P.mirrored else ""),
            f"Epoch J2000.0, stars to magnitude {self.star_mag:.1f}",
        ]
        y += 0.6 * ts
        for ln in lines:
            y += s * 1.48
            T.draw(painter, x0, y, ln, s, col, bold=bb, italic=bi)
        W = x1 - x0
        # --- magnitude scale
        m0 = 0
        m1 = int(math.floor(self.star_mag))
        if m1 - m0 > 10:
            m0 = m1 - 10
        mags = list(range(m0, m1 + 1))
        mx = x0 + W * 0.33
        step = min(5.2 * ts, (W * 0.25) / max(len(mags), 1))
        cy = top + 7.5 * ts
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(C["star_none"] if P.black_white else "#111111")))
        for k, m in enumerate(mags):
            r = float(self._star_radius(m))
            painter.drawEllipse(QPointF(mx + k * step, cy), r, r)
            T.draw(painter, mx + k * step, cy + 3.0 * ts + s, str(m), s, col, align="c", bold=bb, italic=bi)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        T.draw(painter, mx + (len(mags) - 1) * step / 2.0, cy + 4.6 * ts + 2 * s, "Magnitudes", s, col,
               align="c", bold=bb, italic=bi)
        # --- symbol key: only what appears on the chart
        kinds = [K_GALAXY, K_GROUP, K_OC, K_GC, K_PN, K_NEB, K_RFN, K_CLNEB, K_DARK, K_ASSOC]
        items = [("dso", k, KIND_NAMES[k]) for k in kinds if k in self.kinds_drawn]
        if self.has_var:
            items.append(("var", None, "Variable star"))
        if self.has_dbl:
            items.append(("dbl", None, "Double star"))
        if not items:
            items = []
        rows = 3
        colw = (max(T.width(n, s) for _a, _b, n in items) + 7.0 * ts) if items else 0.0
        ncol = int(math.ceil(len(items) / rows))
        kx = x1 - ncol * colw
        kx = max(kx, mx + len(mags) * step + 4.0)
        r = 1.15 * ts
        for n, (what, k, name) in enumerate(items):
            cx = kx + (n // rows) * colw + r
            yy = top + 4.4 * ts + (n % rows) * max(3.4 * ts, 1.6 * s)
            if what == "dso":
                b = r * (0.6 if k == K_GALAXY else 1.0)
                if k in (K_NEB, K_RFN, K_DARK, K_CLNEB):
                    self._symbol_box(painter, k, cx, yy, r, gs * min(ts, 1.0))
                else:
                    self._symbol(painter, k, cx, yy, r, b, 0.0, gs * min(ts, 1.0), clip=False)
            else:
                dot = 0.55 * ts
                colr = "#111111"
                if what == "dbl":
                    painter.setPen(QPen(QColor(colr), 0.18 * gs))
                    painter.drawLine(QPointF(cx - r, yy), QPointF(cx + r, yy))
                else:
                    painter.setPen(QPen(QColor(colr), 0.14 * gs))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(QPointF(cx, yy), r * 0.95, r * 0.95)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(colr)))
                painter.drawEllipse(QPointF(cx, yy), dot, dot)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            T.draw(painter, cx + r + 1.4 * ts, yy + s * 0.36, name, s, col, bold=bb, italic=bi)
        # --- credits
        cs = max(1.5 * ts, 0.83 * s)
        yb = top + self.legend_h - 3.2 * ts
        T.draw(painter, x1, yb - 2.6 * ts, "Seti Astro Suite Pro — sky chart", cs, col, align="r")
        T.draw(painter, x1, yb, "Data: Gaia DR3, HYG 4.1, OpenNGC, d3-celestial", cs, col, align="r")


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def render_svg(params: ChartParams, data: ChartData, path: str, log=print) -> ChartRenderer:
    from PyQt6.QtSvg import QSvgGenerator
    r = ChartRenderer(params, data, log)
    k = 10.0                               # user units per mm
    gen = QSvgGenerator()
    gen.setFileName(path)
    gen.setSize(QSize(int(round(r.pw * k)), int(round(r.ph * k))))
    gen.setViewBox(QRectF(0, 0, r.pw * k, r.ph * k))
    gen.setResolution(254)
    gen.setTitle(params.title or "Sky chart")
    gen.setDescription("Generated by Seti Astro Suite Pro")
    p = QPainter(gen)
    try:
        p.scale(k, k)
        r.render(p)
    finally:
        p.end()
    return r


def render_pdf(params: ChartParams, data: ChartData, path: str, log=print) -> ChartRenderer:
    r = ChartRenderer(params, data, log)
    w = QPdfWriter(path)
    w.setPageSize(QPageSize(QSizeF(r.pw, r.ph), QPageSize.Unit.Millimeter))
    w.setPageMargins(QMarginsF(0, 0, 0, 0))
    w.setResolution(1200)
    w.setTitle(params.title or "Sky chart")
    w.setCreator("Seti Astro Suite Pro")
    p = QPainter(w)
    try:
        k = 1200.0 / 25.4
        p.scale(k, k)
        r.render(p)
    finally:
        p.end()
    return r


def render_image(params: ChartParams, data: ChartData, dpi: float = 300.0, log=print):
    """-> (QImage, renderer)"""
    r = ChartRenderer(params, data, log)
    k = dpi / 25.4
    W, H = int(round(r.pw * k)), int(round(r.ph * k))
    if W * H > 400e6:
        raise ValueError("The image would be too large; lower the resolution.")
    img = QImage(W, H, QImage.Format.Format_RGB32)
    img.fill(QColor("#ffffff"))
    p = QPainter(img)
    try:
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.scale(k, k)
        r.render(p)
    finally:
        p.end()
    return img, r


# Layers that make sense over a survey photograph. The photograph already shows
# the Milky Way, the nebulae and the stars themselves, so those are labelled,
# never painted; the grid is the host chart's own.
OVERLAY_LAYERS = ("ecliptic", "galactic", "borders", "lines", "connames", "messier", "ngc", "ic",
                  "other", "sharpless", "lbn", "ldn", "abell", "png", "starnames", "bayer",
                  "variable", "double")
OVERLAY_DEFAULTS = {"borders": True, "lines": True, "connames": True, "messier": True,
                    "ngc": True, "ic": True, "starnames": True, "bayer": True}


def overlay_params(layers: dict, *, text_scale: float = 1.0,
                   graphics_scale: float = 1.0) -> ChartParams:
    """ChartParams for an overlay, from {layer key: on} (keys in OVERLAY_LAYERS)."""
    lay = {k: False for k, _n, _d, _on in LAYERS}
    for k in OVERLAY_LAYERS:
        lay[k] = bool(layers.get(k, OVERLAY_DEFAULTS.get(k, False)))
    # star names and Bayer letters are labels of the bright-star layer, whose
    # discs stay off: the photograph has the stars
    lay["stars"] = bool(lay["starnames"] or lay["bayer"] or lay["variable"] or lay["double"])
    return ChartParams(
        layers=lay, layer_marks={"stars": False, "gaia": False}, overlay=True, legend=False,
        dso_styles=dict(OVERLAY_DSO_STYLES), black_white=False, text_scale=float(text_scale),
        graphics_scale=float(graphics_scale), online_colors=False, halo_map_bg=False,
        halo_color="#000000")


def render_overlay(params: ChartParams, data: "ChartData", wcs, w: int, h: int,
                   px_per_mm: float | None = None, log=print) -> np.ndarray:
    """
    The chart's labels and linework drawn on a w x h image whose celestial WCS
    is `wcs`, as float32 RGBA in [0, 1], rows in FITS order (row 0 = pixel y 0,
    the bottom row), ready for imshow(..., origin="lower") over that image.

    It reuses the reference-grid path (ref_grid_geometry), which reproduces an
    image's pixel grid exactly. That path counts rows top-down, so the WCS is
    handed over with its second pixel axis reversed, and the result is flipped
    back. SIP terms are not carried: a survey cutout has none.
    """
    M = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64).copy()   # deg/px, FITS rows
    cr = wcs.wcs.crpix
    c0x, c0y = float(cr[0]) - 1.0, float(cr[1]) - 1.0
    M[:, 1] *= -1.0                                   # y' = (h - 1) - y
    code = ""
    try:
        toks = [t for t in str(wcs.wcs.ctype[0]).split("-") if t]
        code = toks[1].upper() if len(toks) > 1 else ""
    except Exception:
        pass
    ref = {"w": int(w), "h": int(h),
           "crval": [float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])],
           "crpix0": [c0x, (h - 1.0) - c0y], "cd": M.tolist(),
           "proj": _PROJ_OF_FITS.get(code, "Gnomonic")}
    # Symbol and text sizes are in millimetres; pick the scale a ~22 cm wide
    # chart would have, so labels read the same at any output size.
    k = float(px_per_mm) if px_per_mm else max(w, h) / 220.0
    P = replace(params, match_image=True, ref_grid=ref, resolution=k * 25.4,
                res_unit="pixels per inch", overlay=True, legend=False, outline=[])
    r = ChartRenderer(P, data, log)
    img = QImage(int(w), int(h), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    try:
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.scale(k, k)
        r.render(p)
    finally:
        p.end()
    im = img.convertToFormat(QImage.Format.Format_RGBA8888)
    bpl = im.bytesPerLine()
    ptr = im.constBits()
    ptr.setsize(int(h) * bpl)
    a = np.frombuffer(ptr, dtype=np.uint8).reshape(int(h), bpl)[:, : int(w) * 4].reshape(int(h), int(w), 4)
    return np.ascontiguousarray(a[::-1].astype(np.float32) / 255.0)


class WcsProjection:
    """
    Drop-in for Projection, backed by an image's own astropy WCS -- distortion
    (SIP / lookup tables) included -- so chart layers land exactly where the
    image's stars are. Projection's ref-grid path keeps only the linear part of a
    WCS, which on a wide-field or lens solution puts labels pixels to tens of
    pixels off their objects toward the corners.

    Chart millimetres are image pixels times `mm_per_px`: pixel (x, y), 0-based,
    rows top-down as SASpro stores images, sits at (x * mm_per_px, y * mm_per_px)
    -- the same x * s mapping the view's RA/Dec grid uses, so the two agree.

    Sky -> pixel goes through the linear WCS for everything, and through the full
    (iterative) inverse only for points near the image: a distortion polynomial
    inverted far outside the frame does not converge, and those points only
    matter for clipping lines that leave the picture anyway.
    """

    kind = "WCS"

    def __init__(self, wcs, w_px: int, h_px: int, mm_per_px: float, tmax_deg: float = 80.0):
        self.w = wcs
        self.W, self.H = int(w_px), int(h_px)
        self.f = float(mm_per_px)
        self.page_w, self.page_h = self.W * self.f, self.H * self.f
        self.tmax = tmax_deg * D2R
        self.rot, self.mir = 0.0, False
        self._dist = bool(getattr(wcs, "has_distortion", False))
        cx, cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0
        ra0, dec0 = wcs.all_pix2world(np.array([cx]), np.array([cy]), 0)
        self.ra0, self.dec0 = float(ra0[0]) * D2R, float(dec0[0]) * D2R
        self.cx, self.cy = cx * self.f, cy * self.f
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            deg_px = float(np.mean(np.abs(proj_plane_pixel_scales(wcs.celestial if hasattr(wcs, "celestial") else wcs))))
        except Exception:
            deg_px = 1.0 / 3600.0
        self.deg_px = deg_px
        self.mm_per_deg = self.f / max(deg_px, 1e-12)
        self.scale = self.mm_per_deg / D2R           # mm per radian, near the centre
        self.fov_w, self.fov_h = self.W * deg_px, self.H * deg_px

    def fit(self, *_a, **_k):
        pass                                         # the WCS is the geometry

    def R(self, t):
        return np.asarray(t, dtype=np.float64)

    def Rinv(self, r):
        return np.asarray(r, dtype=np.float64)

    def _theta(self, a, d):
        s0, c0 = math.sin(self.dec0), math.cos(self.dec0)
        cost = s0 * np.sin(d) + c0 * np.cos(d) * np.cos(a - self.ra0)
        return np.arccos(np.clip(cost, -1.0, 1.0))

    def project(self, ra_deg, dec_deg):
        """-> x_mm, y_mm, theta(rad); NaN (and theta beyond tmax) where it fails."""
        ra = np.atleast_1d(np.asarray(ra_deg, dtype=np.float64))
        dec = np.atleast_1d(np.asarray(dec_deg, dtype=np.float64))
        theta = self._theta(ra * D2R, dec * D2R)
        px = np.full(ra.shape, np.nan)
        py = np.full(ra.shape, np.nan)
        ok = np.isfinite(ra) & np.isfinite(dec) & (theta < self.tmax)
        if ok.any():
            try:
                lx, ly = self.w.wcs_world2pix(ra[ok], dec[ok], 0)
                px[ok], py[ok] = lx, ly
            except Exception:
                pass
            if self._dist:
                m = self.W * 0.3, self.H * 0.3
                near = ok & np.isfinite(px) & (px > -m[0]) & (px < self.W + m[0]) & \
                    (py > -m[1]) & (py < self.H + m[1])
                if near.any():
                    try:
                        fx, fy = self.w.all_world2pix(ra[near], dec[near], 0, quiet=True,
                                                      maxiter=30, tolerance=1e-4)
                        good = np.isfinite(fx) & np.isfinite(fy)
                        idx = np.nonzero(near)[0]
                        px[idx[good]], py[idx[good]] = fx[good], fy[good]
                    except Exception:
                        pass
        bad = ~(np.isfinite(px) & np.isfinite(py))
        theta = np.where(bad, self.tmax + 1.0, theta)
        x, y = px * self.f, py * self.f
        if np.ndim(ra_deg) == 0:
            return x[0], y[0], theta[0]
        return x, y, theta

    def inverse(self, x_mm, y_mm):
        """-> ra_deg, dec_deg (NaN where the WCS cannot answer)"""
        x = np.atleast_1d(np.asarray(x_mm, dtype=np.float64)) / self.f
        y = np.atleast_1d(np.asarray(y_mm, dtype=np.float64)) / self.f
        try:
            ra, dec = self.w.all_pix2world(x, y, 0)
            ra = np.mod(np.asarray(ra, dtype=np.float64), 360.0)
            dec = np.asarray(dec, dtype=np.float64)
        except Exception:
            ra = np.full(x.shape, np.nan)
            dec = np.full(x.shape, np.nan)
        return ra, dec


# Layers the image-view overlay offers (the RA/Dec grid is the view's own).
WCS_OVERLAY_GROUPS = {
    "constellations": ("borders", "lines", "connames"),
    "dso": ("messier", "ngc", "ic", "other", "sharpless", "lbn", "ldn", "abell", "png"),
    "stars": ("starnames", "bayer"),
}


# Image views' own layer choice (Settings > WCS + Chart Overlay > Layers…),
# kept apart from Finder Chart's. Until it is saved, the three switches it
# replaced seed it (each turning its whole group above on or off), so what a
# view draws doesn't change until the layers are actually picked.
WCS_OVERLAY_LAYERS_KEY = "wcs_chart/layers"
WCS_OVERLAY_DEFAULTS = {k: (k in {m for g in WCS_OVERLAY_GROUPS.values() for m in g})
                        for k in OVERLAY_LAYERS}
WCS_OVERLAY_OPTION_DEFAULTS = {"dso_mag": 0.0, "text_scale": 1.6, "graphics_scale": 1.3}
_WCS_OVERLAY_LEGACY = {"constellations": "wcs_chart/constellations", "dso": "wcs_chart/dso",
                       "stars": "wcs_chart/star_names"}
_WCS_OVERLAY_GROUP_NAMES = {"constellations": "Constellations",
                            "dso": "all deep-sky catalogs", "stars": "star labels"}


def wcs_overlay_config(settings) -> dict:
    """The image views' chart overlay: {"layers": {key: on} for every
    OVERLAY_LAYERS key, "dso_mag" (0 = automatic), "text_scale",
    "graphics_scale"}."""
    import json as _json
    lay = None
    raw = settings.value(WCS_OVERLAY_LAYERS_KEY, "", type=str) or ""
    if raw:
        try:
            d = _json.loads(raw)
            if isinstance(d, dict):
                lay = {k: bool(d.get(k, False)) for k in OVERLAY_LAYERS}
        except Exception:
            lay = None
    if lay is None:
        lay = {k: False for k in OVERLAY_LAYERS}
        for g, key in _WCS_OVERLAY_LEGACY.items():
            on = bool(settings.value(key, True, type=bool))
            for k in WCS_OVERLAY_GROUPS[g]:
                lay[k] = on
    out = {"layers": lay}
    for k, d in WCS_OVERLAY_OPTION_DEFAULTS.items():
        try:
            out[k] = float(settings.value(f"wcs_chart/{k}", d, type=float))
        except Exception:
            out[k] = float(d)
    return out


def save_wcs_overlay_config(settings, cfg: dict) -> None:
    import json as _json
    lay = cfg.get("layers") or {}
    settings.setValue(WCS_OVERLAY_LAYERS_KEY,
                      _json.dumps({k: bool(lay.get(k, False)) for k in OVERLAY_LAYERS}))
    for k, d in WCS_OVERLAY_OPTION_DEFAULTS.items():
        settings.setValue(f"wcs_chart/{k}", float(cfg.get(k, d)))


def wcs_overlay_summary(cfg: dict) -> str:
    """Short description of the chosen layers, for the Settings page."""
    lay = cfg.get("layers") or {}
    names = {k: n for k, n, _d, _on in LAYERS}
    parts, done = [], set()
    for g, keys in WCS_OVERLAY_GROUPS.items():
        if all(lay.get(k) for k in keys):
            parts.append(_WCS_OVERLAY_GROUP_NAMES[g])
            done.update(keys)
    for k in OVERLAY_LAYERS:
        if lay.get(k) and k not in done:
            parts.append(names.get(k, k))
    return ", ".join(parts) if parts else "None"


def paint_wcs_overlay(painter: QPainter, wcs, w_full: int, h_full: int, zoom: float,
                      data: "ChartData", *, constellations=True, dso=True, star_names=True,
                      layers: dict | None = None,
                      dso_mag=None, px_per_mm: float = 3.8, text_scale: float = 1.6,
                      graphics_scale: float = 1.3, log=print) -> dict:
    """
    Paint chart layers onto an image view's pixmap, the image drawn `zoom` times
    its full size (pixel x lands at x * zoom, the view grid's own mapping).

    Positions follow the zoom; text and symbols do not: px_per_mm fixes how many
    screen pixels a chart millimetre is, so labels stay the same size on screen
    and the label placer declutters for each zoom level -- zooming in shows more.
    Annotation only: no star discs, nebula fills or Milky Way; the image has them.
    Returns the renderer's info dict.

    `layers` ({key: on}, keys in OVERLAY_LAYERS) picks layers one by one and
    overrides the three group switches. Either way every layer is given an
    explicit on/off: overlay_params() fills a missing key from
    OVERLAY_DEFAULTS, so a group switched off used to come back on.
    """
    if layers is not None:
        lay = {k: bool(layers.get(k, False)) for k in OVERLAY_LAYERS}
    else:
        lay = {k: False for k in OVERLAY_LAYERS}
        for g, on in (("constellations", constellations), ("dso", dso),
                      ("stars", star_names)):
            if on:
                lay.update({k: True for k in WCS_OVERLAY_GROUPS[g]})
    P = overlay_params(lay, text_scale=text_scale, graphics_scale=graphics_scale)
    P = replace(P, catalog_ids=False)
    if dso_mag is not None:
        P = replace(P, dso_mag_auto=False, dso_mag=float(dso_mag))
    k = float(px_per_mm)
    proj = WcsProjection(wcs, w_full, h_full, mm_per_px=float(zoom) / k)
    r = ChartRenderer(P, data, log, projection=proj)
    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.scale(k, k)
        r.render(painter)
    finally:
        painter.restore()
    return r.info


def qimage_to_array(img: QImage) -> np.ndarray:
    im = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = im.width(), im.height(), im.bytesPerLine()
    ptr = im.constBits()
    ptr.setsize(h * bpl)
    a = np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl)[:, : w * 3].reshape(h, w, 3)
    return (a.astype(np.float32) / 255.0).copy()
