from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING
import io
import numpy as np
import csv
import re
from pathlib import Path
from typing import List, Dict, Any
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QFileDialog, QMessageBox, QSpinBox, QSlider, QApplication, QDoubleSpinBox
)

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib import patheffects as pe
from pathlib import Path
from setiastro.saspro.resources import get_data_path
from setiastro.saspro.bright_stars import BRIGHT_STARS

if TYPE_CHECKING:
    from astropy.wcs import WCS as AstropyWCS
    from astropy.coordinates import SkyCoord as AstropySkyCoord
else:
    AstropyWCS = object
    AstropySkyCoord = object


@dataclass
class FinderChartRequest:
    survey: str
    scale_mult: int
    show_grid: bool

    # star overlay
    show_star_names: bool = False
    star_mag_limit: float = 2.0
    star_max_labels: int = 30

    # deep-sky overlay
    show_dso: bool = False
    dso_catalog: str = "Messier"
    dso_mag_limit: float = 10.0
    dso_max_labels: int = 30

    # chart aids
    show_compass: bool = True
    show_scale_bar: bool = True

    out_px: int = 900
    overlay_opacity: float = 0.35

# ---------------- Catalog loading (cached) ----------------

_CATALOG_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def _catalog_path(name: str) -> Path:
    # everything now comes from celestial_catalog.csv
    return Path(get_data_path("data/catalogs/celestial_catalog.csv"))



def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


_size_re = re.compile(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)")

def _parse_size_arcmin(info: str) -> Optional[tuple]:
    """
    Parse sizes like " 6.0x4.0" or "90x40" from Info field.
    Returns (w_arcmin, h_arcmin) or None.
    """
    if not info:
        return None
    m = _size_re.search(str(info))
    if not m:
        return None
    w = _safe_float(m.group(1), None)
    h = _safe_float(m.group(2), None)
    if w is None or h is None:
        return None
    # Assume arcminutes (matches your Messier examples)
    return (float(w), float(h))

def _open_catalog_csv(path: Path):
    """
    Catalogs may be saved as UTF-8, UTF-8 with BOM, or Windows-1252 / latin-1.
    IMPORTANT: we must *force a decode* here (open() alone doesn't decode until read).
    Returns a *text* file-like object suitable for csv.DictReader.
    """
    data = path.read_bytes()

    last = None
    for enc in ("latin-1", "utf-8-sig", "utf-8", "cp1252"):
        try:
            text = data.decode(enc)  # <-- force decode NOW
            # newline="" behavior like open(..., newline="") for csv module
            return io.StringIO(text, newline="")
        except UnicodeDecodeError as e:
            last = e

    # If we get here, decoding failed for all options
    raise last or UnicodeDecodeError("utf-8", b"", 0, 1, "Unknown decode error")

def _canon_catalog_code(name: str, catalog: str) -> str:
    n = (name or "").strip().upper()
    c = (catalog or "").strip().upper()

    # --- Sharpless (SH2) ---
    if c in {"SHARPLESS", "SH2", "SH-2", "SH 2", "SH_2"}:
        return "SH2"
    if n.startswith(("SH2-", "SH2 ", "SH-2", "SH 2", "SH_2")):
        return "SH2"

    # --- Planetary Nebula Galactic (PN-G / PNG) ---
    if c in {"PNG", "PN-G", "PN G", "PN_G"}:
        return "PN-G"
    if n.startswith(("PN-G", "PN G", "PN_G", "PNG ")):
        return "PN-G"

    # Keep the catalog column as-is for everything else (NGC/IC/Abell/etc.)
    return c


def _load_catalog_rows(kind: str) -> List[Dict[str, Any]]:
    if kind in _CATALOG_CACHE:
        return _CATALOG_CACHE[kind]

    path = _catalog_path(kind)
    rows: List[Dict[str, Any]] = []

    if not path.exists():
        print(f"[DSO] catalog path missing: {path}")
        _CATALOG_CACHE[kind] = rows
        return rows

    try:
        with _open_catalog_csv(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                def getk(*keys):
                    for k in keys:
                        if k in r and r[k] is not None:
                            return r[k]
                        kb = "\ufeff" + k
                        if kb in r and r[kb] is not None:
                            return r[kb]
                    return None

                name = str(getk("Name", "NAME") or "").strip()
                ra   = _safe_float(getk("RA", "Ra", "ra"), None)
                dec  = _safe_float(getk("Dec", "DEC", "dec"), None)
                if not name or ra is None or dec is None:
                    continue

                mag  = _safe_float(getk("Magnitude", "MAG", "mag"), None)
                info = str(getk("Info", "Diameter") or "").strip()
                typ  = str(getk("Type", "LongType") or "").strip()
                cat  = str(getk("Catalog") or "").strip()
                name = str(getk("Name", "NAME") or "").strip()
                canon = _canon_catalog_code(name, cat)

                rows.append({
                    "name": name,
                    "ra": float(ra),
                    "dec": float(dec),
                    "mag": mag,
                    "info": info,
                    "catalog": cat,          # raw
                    "catalog_code": canon,   # canonical (used for filtering)
                    "type": typ,
                })

    except Exception as e:
        print(f"[DSO] catalog read failed: {path} ({type(e).__name__}: {e})")
        _CATALOG_CACHE[kind] = []
        return []

    # per-kind filtering (based on Catalog column in celestial_catalog.csv)
    k = (kind or "").strip().upper()

    # Map UI label -> allowed catalog codes (also uppercase)
    allowed_map = {
        "M": {"M", "MESSIER"},
        "NGC": {"NGC"},
        "IC": {"IC"},
        "ABELL": {"ABELL"},
        "SH2": {"SH2"},
        "LBN": {"LBN"},
        "LDN": {"LDN"},
        "PN-G": {"PN-G"},
        "ALL (DSO)": None,
    }

    allowed = allowed_map.get(k, None)

    def _catcode(row) -> str:
        return (row.get("catalog_code") or row.get("catalog") or "").strip().upper()


    if allowed is not None:
        rows = [r for r in rows if _catcode(r) in allowed]

    _CATALOG_CACHE[kind] = rows
    return rows

def _pixel_scale_arcsec(bg_wcs: "WCS") -> Optional[float]:
    """
    Approx arcsec/pixel using astropy helper. Works for TAN-ish WCS.
    """
    if bg_wcs is None:
        return None
    try:
        # returns degrees/pixel per axis
        sc = proj_plane_pixel_scales(bg_wcs)  # deg/pix
        deg_per_pix = float(np.nanmedian(sc))
        if not np.isfinite(deg_per_pix) or deg_per_pix <= 0:
            return None
        return deg_per_pix * 3600.0
    except Exception:
        return None

def _draw_dso_overlay(ax, bg_wcs: "WCS", center: "SkyCoord", fov_deg: float, req: FinderChartRequest, renderer):
    """
    Plot catalog objects within ~0.75*fov radius; declutter via coarse grid.
    """
    if bg_wcs is None or center is None:
        return

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    rows = _load_catalog_rows(req.dso_catalog)
 
    if not rows:
        return
    out_px = int(getattr(ax.figure, "_sas_out_px", 0) or 0)
    if out_px <= 0:
        out_px = int(ax.figure.get_figwidth() * ax.figure.dpi)
    ra0 = float(center.ra.deg)
    dec0 = float(center.dec.deg)
    radius = float(fov_deg) * 0.75

    c0 = SkyCoord(ra0*u.deg, dec0*u.deg, frame="icrs")

    # filter by radius + mag
    cand = []
    for r in rows:
        if abs(r["dec"] - dec0) > radius + 2.0:
            continue
        c1 = SkyCoord(r["ra"]*u.deg, r["dec"]*u.deg, frame="icrs")
        if c0.separation(c1).deg > radius:
            continue

        mag = r.get("mag", None)
        if mag is not None and mag > float(req.dso_mag_limit):
            continue

        # score: brighter first; unknown mag goes later
        score = (mag if mag is not None else 99.0)
        cand.append((score, r))

    if not cand:
        return

    cand.sort(key=lambda t: t[0])
    cand = cand[:max(1, int(req.dso_max_labels) * 4)]  # keep a bit extra before declutter

    # project
    coords = SkyCoord([t[1]["ra"] for t in cand]*u.deg, [t[1]["dec"] for t in cand]*u.deg, frame="icrs")
    xs, ys = bg_wcs.world_to_pixel(coords)
    out_px = int(getattr(ax.figure, "_sas_out_px", 0) or 0)
    if out_px <= 0:
        out_px = int(ax.figure.get_figwidth() * ax.figure.dpi)
    # declutter: one label per coarse cell
    kept = []
    cell = 34  # px
    used = set()

    for (t, x, y) in zip(cand, xs, ys):
        x = float(x); y = float(y)
        if not _inside_px(x, y, out_px, pad=0):
            continue        
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        gx = int(x // cell)
        gy = int(y // cell)
        key = (gx, gy)
        if key in used:
            continue
        used.add(key)
        kept.append((t[1], float(x), float(y)))
        if len(kept) >= int(req.dso_max_labels):
            break

    if not kept:
        return

    # optional size ellipse (Info field, arcmin)
    arcsec_per_pix = _pixel_scale_arcsec(bg_wcs)

    for (r, x, y) in kept:
        name = r["name"]

        # marker
        ax.plot(
            [x], [y],
            marker="s",
            markersize=3.5,
            alpha=0.9,
            color="#66ccff",  # light cyan (optional)
            transform=ax.get_transform("pixel"),
        )

        # size circle if we can parse it (Info field, arcmin)
        if arcsec_per_pix is not None:
            sz = _parse_size_arcmin(r.get("info", ""))
            if sz is not None:
                w_arcmin, h_arcmin = sz

                # Use the MAJOR axis only (honest; no PA info)
                major_arcmin = float(max(w_arcmin, h_arcmin))

                # convert major-axis diameter arcmin -> pixel radius
                diam_px = (major_arcmin * 60.0) / arcsec_per_pix
                rad_px = 0.5 * diam_px

                if np.isfinite(rad_px) and rad_px > 2:
                    try:
                        from matplotlib.patches import Circle
                        c = Circle(
                            (x, y),
                            radius=rad_px,
                            fill=False,
                            linewidth=1,
                            alpha=0.75,
                            edgecolor="#5145ff",   # cyan outline (NOT black)
                            transform=ax.get_transform("pixel"),
                        )
                        ax.add_patch(c)
                    except Exception:
                        pass

        # label
        _place_label_inside(
            ax, x, y, name,
            dx=7, dy=5, fontsize=9,
            color="white", alpha=0.95, outline=True,
            out_px=out_px, pad=4, renderer=renderer
        )

def _draw_compass_NE(ax, bg_wcs: "WCS", out_px: int, fov_deg: float):
    """
    Draw N/E arrows in the bottom-right using image pixel/data coordinates.
    Works reliably with WCSAxes.
    """
    if bg_wcs is None:
        return

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    # place near bottom-right in IMAGE pixel coords
    cx = out_px * 0.86
    cy = out_px * 0.12
    base_len = out_px * 0.09  # pixels

    # choose a small sky step relative to FOV (avoid huge jumps on big FOV)
    step_deg = max(0.01, min(0.15, float(fov_deg) * 0.08)) * u.deg

    try:
        csky = bg_wcs.pixel_to_world(out_px * 0.5, out_px * 0.5)
        ra = csky.ra.to(u.deg)
        dec = csky.dec.to(u.deg)

        # North: +Dec
        cn = SkyCoord(ra, dec + step_deg, frame="icrs")

        # East: +RA (compensate a bit for cos(dec))
        cosd = max(0.15, float(np.cos(dec.to_value(u.rad))))
        ce = SkyCoord(ra + (step_deg / cosd), dec, frame="icrs")

        x0, y0 = bg_wcs.world_to_pixel(SkyCoord(ra, dec, frame="icrs"))
        xn, yn = bg_wcs.world_to_pixel(cn)
        xe, ye = bg_wcs.world_to_pixel(ce)

        vn = np.array([float(xn - x0), float(yn - y0)], dtype=np.float64)
        ve = np.array([float(xe - x0), float(ye - y0)], dtype=np.float64)

        def _n(v):
            n = float(np.hypot(v[0], v[1])) or 1.0
            return v / n

        vn = _n(vn) * base_len
        ve = _n(ve) * base_len

    except Exception:
        # fallback: at least show something
        vn = np.array([0.0, base_len])
        ve = np.array([base_len, 0.0])

    # Draw in DATA coords (image pixels) — this is the big fix
    tx = ax.transData

    ax.annotate(
        "", xy=(cx + ve[0], cy + ve[1]), xytext=(cx, cy),
        xycoords=tx, textcoords=tx,
        arrowprops=dict(arrowstyle="->", linewidth=2.2, color="white", alpha=0.95)
    )
    ax.annotate(
        "", xy=(cx + vn[0], cy + vn[1]), xytext=(cx, cy),
        xycoords=tx, textcoords=tx,
        arrowprops=dict(arrowstyle="->", linewidth=2.2, color="white", alpha=0.95)
    )

    bbox = dict(boxstyle="round,pad=0.15", facecolor=(0, 0, 0, 0.55), edgecolor=(1, 1, 1, 0.15))

    ax.text(cx + ve[0] + 6, cy + ve[1] + 2, "E",
            transform=tx, fontsize=10, color="white", alpha=0.98, bbox=bbox)
    ax.text(cx + vn[0] + 6, cy + vn[1] + 2, "N",
            transform=tx, fontsize=10, color="white", alpha=0.98, bbox=bbox)


def _draw_scale_bar(ax, bg_wcs: "WCS", out_px: int, fov_deg: float):
    """
    Draw a scale bar bottom-left with units + arcsec/pixel.
    """
    if bg_wcs is None:
        return

    arcsec_per_pix = _pixel_scale_arcsec(bg_wcs)
    if arcsec_per_pix is None:
        return

    fov_arcmin = float(fov_deg) * 60.0
    if fov_arcmin >= 240:
        bar_arcmin = 30
    elif fov_arcmin >= 120:
        bar_arcmin = 20
    elif fov_arcmin >= 60:
        bar_arcmin = 10
    else:
        bar_arcmin = 5

    bar_px = (bar_arcmin * 60.0) / arcsec_per_pix
    if not np.isfinite(bar_px) or bar_px < 30:
        return

    x0 = out_px * 0.08
    y0 = out_px * 0.10
    x1 = x0 + bar_px

    tx = ax.transData  # BIG FIX

    ax.plot([x0, x1], [y0, y0], linewidth=3, color="white", alpha=0.95, transform=tx)
    ax.plot([x0, x0], [y0 - 5, y0 + 5], linewidth=2, color="white", alpha=0.95, transform=tx)
    ax.plot([x1, x1], [y0 - 5, y0 + 5], linewidth=2, color="white", alpha=0.95, transform=tx)

    bbox = dict(boxstyle="round,pad=0.2", facecolor=(0, 0, 0, 0.55), edgecolor=(1, 1, 1, 0.12))
    ax.text(
        x0, y0 + 14,
        f"{bar_arcmin}′  ({arcsec_per_pix:.2f}″/px)",
        transform=tx,
        fontsize=10,
        color="white",
        alpha=0.98,
        bbox=bbox
    )


def get_doc_wcs(meta: dict) -> Optional["AstropyWCS"]:
    """Prefer prebuilt WCS object; else build from header. Always return *celestial* (2D) WCS."""
    if WCS is None:
        return None

    w = meta.get("wcs")
    if w is None:
        hdr = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
        if hdr is None:
            return None

        # normalize to fits.Header if needed
        if fits is not None and not isinstance(hdr, fits.Header):
            try:
                h2 = fits.Header()
                for k, v in dict(hdr).items():
                    try:
                        h2[k] = v
                    except Exception:
                        pass
                hdr = h2
            except Exception:
                return None

        try:
            w = WCS(hdr, relax=True)
        except Exception:
            return None

    # --- CRITICAL FIX: if WCS has extra axes, use celestial slice ---
    try:
        if hasattr(w, "celestial"):
            wc = w.celestial
            if wc is not None:
                return wc
    except Exception:
        pass

    return w


def image_footprint_sky(wcs: "WCS", w: int, h: int):
    """Return (corners SkyCoord[4], center SkyCoord)."""
    wc = wcs
    try:
        if hasattr(wcs, "celestial") and wcs.celestial is not None:
            wc = wcs.celestial
    except Exception:
        pass

    xs = np.array([0.5, w - 0.5, w - 0.5, 0.5], dtype=np.float64)
    ys = np.array([0.5, 0.5, h - 0.5, h - 0.5], dtype=np.float64)

    corners = wc.pixel_to_world(xs, ys)
    center = wc.pixel_to_world(np.array([w / 2.0]), np.array([h / 2.0]))
    return corners, center


def _ang_sep_deg(a1, d1, a2, d2) -> float:
    """Small helper; inputs degrees."""
    ra1 = math.radians(a1); dec1 = math.radians(d1)
    ra2 = math.radians(a2); dec2 = math.radians(d2)
    return math.degrees(math.acos(
        max(-1.0, min(1.0, math.sin(dec1)*math.sin(dec2)
                      + math.cos(dec1)*math.cos(dec2)*math.cos(ra1-ra2)))
    ))


def estimate_fov_deg(corners: "AstropySkyCoord") -> Tuple[float, float]:
    """Approx FOV width/height in degrees using corner separations."""
    ra = corners.ra.deg
    dec = corners.dec.deg

    w1 = _ang_sep_deg(ra[0], dec[0], ra[1], dec[1])
    w2 = _ang_sep_deg(ra[3], dec[3], ra[2], dec[2])
    width = 0.5 * (w1 + w2)

    h1 = _ang_sep_deg(ra[0], dec[0], ra[3], dec[3])
    h2 = _ang_sep_deg(ra[1], dec[1], ra[2], dec[2])
    height = 0.5 * (h1 + h2)

    return float(width), float(height)

def _wcs_shift_for_crop(w: WCS, x0: int, y0: int) -> WCS:
    """
    Return a copy of WCS adjusted for cropping (x0,y0) pixels off left/bottom.
    Array pixels are 0-based; FITS WCS CRPIX is 1-based, but the shift is the same in pixels.
    """
    if w is None:
        return None
    try:
        wc = w.deepcopy()
        # CRPIX is in pixel units; subtract crop origin
        wc.wcs.crpix = np.array(wc.wcs.crpix, dtype=float) - np.array([float(x0), float(y0)], dtype=float)
        return wc
    except Exception:
        return w


def _crop_center(bg_rgb01: np.ndarray, bg_wcs: Optional[WCS], out_px: int):
    """
    Center-crop bg to (out_px,out_px). Returns (cropped_bg, cropped_wcs, (x0,y0)).
    """
    H, W = bg_rgb01.shape[:2]
    out_px = int(out_px)
    if out_px <= 0 or out_px > min(H, W):
        return bg_rgb01, bg_wcs, (0, 0)

    x0 = int((W - out_px) // 2)
    y0 = int((H - out_px) // 2)

    crop = bg_rgb01[y0:y0 + out_px, x0:x0 + out_px, :]
    wc = _wcs_shift_for_crop(bg_wcs, x0, y0) if bg_wcs is not None else None
    return crop, wc, (x0, y0)

def _inside_px(x: float, y: float, out_px: int, pad: int = 0) -> bool:
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    return (-pad) <= x <= (out_px - 1 + pad) and (-pad) <= y <= (out_px - 1 + pad)

def _place_label_inside(ax, x, y, text, *, dx=6, dy=4, fontsize=9,
                        color="white", alpha=0.95, outline=True,
                        out_px=900, pad=3, renderer=None):
    """
    Place a label near (x,y) but ensure its bounding box stays inside [0,out_px).
    Requires a renderer (create once per chart render).
    """
    if renderer is None:
        # best-effort fallback (still avoids draw); may be None on some backends until first draw
        renderer = getattr(ax.figure.canvas, "get_renderer", lambda: None)()
        if renderer is None:
            # can't measure; just place it and clip
            t = ax.text(x + dx, y + dy, text, fontsize=fontsize, color=color, alpha=alpha,
                        transform=ax.get_transform("pixel"), clip_on=True)
            if outline:
                t.set_path_effects([pe.Stroke(linewidth=2.0, foreground=(0, 0, 0, 0.85)), pe.Normal()])
            return t

    t = ax.text(x + dx, y + dy, text, fontsize=fontsize, color=color, alpha=alpha,
                transform=ax.get_transform("pixel"), clip_on=True)
    if outline:
        t.set_path_effects([pe.Stroke(linewidth=2.0, foreground=(0, 0, 0, 0.85)), pe.Normal()])

    bb = t.get_window_extent(renderer=renderer)
    inv = ax.get_transform("pixel").inverted()

    (x0, y0) = inv.transform((bb.x0, bb.y0))
    (x1, y1) = inv.transform((bb.x1, bb.y1))

    shift_x = 0.0
    shift_y = 0.0

    if x0 < pad:
        shift_x += (pad - x0)
    if x1 > (out_px - pad):
        shift_x -= (x1 - (out_px - pad))
    if y0 < pad:
        shift_y += (pad - y0)
    if y1 > (out_px - pad):
        shift_y -= (y1 - (out_px - pad))

    if shift_x or shift_y:
        t.set_position((x + dx + shift_x, y + dy + shift_y))

        # Re-check once (no draw needed)
        bb2 = t.get_window_extent(renderer=renderer)
        (xx0, yy0) = inv.transform((bb2.x0, bb2.y0))
        (xx1, yy1) = inv.transform((bb2.x1, bb2.y1))
        if (xx0 < 0) or (yy0 < 0) or (xx1 > out_px) or (yy1 > out_px):
            t.remove()
            return None

    return t


def _survey_to_hips_id(label: str) -> str:
    key = (label or "").strip().lower()
    if "dss" in key:
        return "CDS/P/DSS2/color"
    if "pan" in key:
        return "CDS/P/PanSTARRS/DR1/color"
    if "gaia" in key:
        return "CDS/P/Gaia/DR3/flux-color"
    return "CDS/P/DSS2/color"

def try_fetch_hips_cutout(center: "SkyCoord", fov_deg: float, out_px: int, survey_label: str):
    """
    Returns (rgb_float01, bg_wcs_celestial, err)
      - rgb_float01: (H,W,3) float32 [0..1] or None
      - bg_wcs_celestial: WCS(2D) or None
      - err: str or None
    """
    hips_id = _survey_to_hips_id(survey_label)

    try:
        from astroquery.hips2fits import hips2fits
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

    def _decode(hdul):
        data = np.array(hdul[0].data, dtype=np.float32)

        bg_wcs = None
        try:
            bg_wcs = WCS(hdul[0].header, relax=True)
            if hasattr(bg_wcs, "celestial") and bg_wcs.celestial is not None:
                bg_wcs = bg_wcs.celestial
        except Exception:
            bg_wcs = None

        # Normalize to RGB
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        elif data.ndim == 3 and data.shape[0] in (3, 4):
            data = np.transpose(data[:3, ...], (1, 2, 0))
        elif data.ndim == 3 and data.shape[2] >= 3:
            data = data[..., :3]

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        lo, hi = np.percentile(data, [1.0, 99.5])
        if hi > lo:
            data = (data - lo) / (hi - lo)

        return np.clip(data, 0.0, 1.0), bg_wcs

    out_px = int(out_px)
    ra_deg = float(center.ra.deg)
    dec_deg = float(center.dec.deg)

    # Prefer quantity for fov, but fall back if this build wants float
    try:
        import astropy.units as u
        fov = float(fov_deg) * u.deg
    except Exception:
        fov = float(fov_deg)

    last_err = None

    # ---- Correct signature for YOUR astroquery 0.4.11 ----
    # query(hips, width, height, projection, ra, dec, fov, *, coordsys='icrs', ...)
    import astropy.units as u
    from astropy.coordinates import Angle

    out_px = int(out_px)

    # IMPORTANT: pass Angle/Quantity, not floats
    ra  = center.ra.to(u.deg)          # Angle
    dec = center.dec.to(u.deg)         # Angle
    fov = Angle(float(fov_deg), unit=u.deg)

    try:
        hdul = hips2fits.query(
            hips_id,
            out_px, out_px,
            "TAN",
            ra, dec,
            fov,
            coordsys="icrs",
            format="fits",
        )
        rgb01, bg_wcs = _decode(hdul)
        return rgb01, bg_wcs, None

    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

    return None, None, last_err


def render_finder_chart(doc_image: np.ndarray, meta: dict, req: FinderChartRequest) -> Optional[np.ndarray]:
    if WCS is None:
        return None

    doc_wcs = get_doc_wcs(meta)
    if doc_wcs is None:
        return None

    H, Wimg = doc_image.shape[:2]
    corners, center = image_footprint_sky(doc_wcs, Wimg, H)
    fov_w, fov_h = estimate_fov_deg(corners)
    fov = max(fov_w, fov_h) * float(req.scale_mult)

    # Fetch background (+ WCS + error string)
    bg, bg_wcs, err = try_fetch_hips_cutout(center[0], fov_deg=fov, out_px=req.out_px, survey_label=req.survey)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(req.out_px / 100.0, req.out_px / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    # --- Draw background OR error message ---
    if bg is None:
        ax.set_facecolor((0, 0, 0))
        msg = "No HiPS background.\n" + (err or "Unknown error")
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    else:
        # If you want doc overlay, do it here
        if bg_wcs is not None and doc_wcs is not None:
            try:
                overlay_u8 = _overlay_doc_on_bg(bg, bg_wcs, doc_image, doc_wcs, alpha=req.overlay_opacity)
                ax.imshow(overlay_u8, origin="lower")
            except Exception:
                # fallback to plain bg if overlay fails
                ax.imshow(bg, origin="lower")
        else:
            ax.imshow(bg, origin="lower")

        # Optional: draw WCS-correct footprint polygon
        if bg_wcs is not None:
            try:
                xs, ys = bg_wcs.world_to_pixel(corners)
                ax.plot(
                    [xs[0], xs[1], xs[2], xs[3], xs[0]],
                    [ys[0], ys[1], ys[2], ys[3], ys[0]],
                    linewidth=2
                )
            except Exception:
                pass

    # center crosshair (axes coords)
    ax.plot([0.5], [0.5], marker="+", markersize=20, transform=ax.transAxes)

    # labels
    ra = float(center[0].ra.deg)
    dec = float(center[0].dec.deg)
    ax.text(
        0.02, 0.98,
        f"{req.survey}  |  {req.scale_mult}×FOV\nRA {ra:.6f}°  Dec {dec:.6f}°\nFOV ~ {fov*60:.1f}′",
        transform=ax.transAxes, va="top",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.45), edgecolor=(1, 1, 1, 0.12))
    )

    if req.show_grid:
        # keep axis ON so grid can render
        ax.set_axis_on()
        ax.set_xticks(np.linspace(0, req.out_px, 7))
        ax.set_yticks(np.linspace(0, req.out_px, 7))
        ax.grid(True, alpha=0.35)
        # optionally hide tick labels but keep grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()

    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()

    plt.close(fig)
    return rgb

def _to_u8_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[2] > 3:
        a = a[..., :3]
    a = a.astype(np.float32)
    # simple robust normalize for display
    lo, hi = np.percentile(a, [1.0, 99.5])
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)

def _overlay_doc_on_bg(bg_rgb01: np.ndarray, bg_wcs: "WCS", doc_img: np.ndarray, doc_wcs: "WCS", alpha=0.35) -> np.ndarray:
    import cv2

    Hbg, Wbg = bg_rgb01.shape[:2]
    bg_u8 = (np.clip(bg_rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    doc_u8 = _to_u8_rgb(doc_img)
    H, W = doc_u8.shape[:2]

    # doc pixel corners -> sky -> bg pixels
    src = np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]], dtype=np.float32)
    sky = doc_wcs.pixel_to_world(src[:,0], src[:,1])   # SkyCoord
    xbg, ybg = bg_wcs.world_to_pixel(sky)
    dst = np.stack([xbg, ybg], axis=1).astype(np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc_u8, M, (Wbg, Hbg), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # alpha blend where warped has content
    mask = (warped.sum(axis=2) > 0).astype(np.float32)[..., None]
    out = bg_u8.astype(np.float32) * (1 - alpha*mask) + warped.astype(np.float32) * (alpha*mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def _rgb_u8_to_qimage(rgb_u8: np.ndarray) -> QImage:
    rgb_u8 = np.ascontiguousarray(rgb_u8)
    h, w, _ = rgb_u8.shape
    bpl = rgb_u8.strides[0]
    # QImage uses the buffer; to be safe, copy via .copy() when making pixmap
    return QImage(rgb_u8.data, w, h, bpl, QImage.Format.Format_RGB888)

def _draw_star_names(ax, bg_wcs: "WCS", center: "SkyCoord", fov_deg: float, *,
                     mag_limit: float = 2.0, max_labels: int = 30, renderer=None):
    if bg_wcs is None:
        return

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ra0 = float(center.ra.deg)
    dec0 = float(center.dec.deg)
    radius = float(fov_deg) * 0.75

    c0 = SkyCoord(ra0*u.deg, dec0*u.deg, frame="icrs")  # <-- MOVE OUTSIDE LOOP

    rows = []
    for (name, ra, dec, vmag) in BRIGHT_STARS:
        if float(vmag) > float(mag_limit):
            continue
        if abs(dec - dec0) > radius + 2.0:
            continue
        c1 = SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame="icrs")
        if c0.separation(c1).deg <= radius:
            rows.append((name, float(ra), float(dec), float(vmag)))

    if not rows:
        return

    rows.sort(key=lambda r: r[3])
    rows = rows[:max_labels]

    coords = SkyCoord([r[1] for r in rows]*u.deg, [r[2] for r in rows]*u.deg, frame="icrs")
    xs, ys = bg_wcs.world_to_pixel(coords)

    out_px = int(getattr(ax.figure, "_sas_out_px", 0) or 0)
    if out_px <= 0:
        out_px = int(ax.figure.get_figwidth() * ax.figure.dpi)

    kept = []
    cell = 28
    used = set()

    for (row, x, y) in zip(rows, xs, ys):
        x = float(x); y = float(y)
        if not _inside_px(x, y, out_px, pad=0):
            continue
        gx = int(x // cell)
        gy = int(y // cell)
        key = (gx, gy)
        if key in used:
            continue
        used.add(key)
        kept.append((row[0], x, y))
        if len(kept) >= int(max_labels):
            break

    for (name, x, y) in kept:
        ax.plot([x], [y],
                marker="o", markersize=2.5, alpha=0.85,
                color="#ffb000",
                transform=ax.get_transform("pixel"),
                clip_on=True)

        _place_label_inside(
            ax, x, y, name,
            dx=6, dy=4, fontsize=9,
            color="#ffb000", alpha=0.95, outline=True,
            out_px=out_px, pad=4,
            renderer=renderer,          # <-- PASS IT
        )


def render_finder_chart_cached(
    *,
    doc_image: np.ndarray,
    doc_wcs: WCS,
    corners: SkyCoord,
    center: SkyCoord,
    fov_deg: float,
    req: FinderChartRequest,
    bg: Optional[np.ndarray],
    bg_wcs: Optional[WCS],
    err: Optional[str],
    base_u8: Optional[np.ndarray] = None,   # <-- NEW
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(req.out_px / 100.0, req.out_px / 100.0), dpi=100)
    fig._sas_out_px = int(req.out_px)

    # Use WCSAxes when we have bg_wcs so we can draw RA/Dec labels & grid properly
    if bg_wcs is not None:
        ax = fig.add_subplot(111, projection=bg_wcs)
    else:
        ax = fig.add_axes([0, 0, 1, 1])

    # ---- background (or error) ----
    if bg is None:
        ax.set_facecolor((0, 0, 0))
        msg = "No HiPS background.\n" + (err or "Unknown error")
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    else:
        # If provided, base_u8 already includes hips + optional warped doc overlay.
        if base_u8 is not None:
            ax.imshow(base_u8, origin="lower")
        else:
            # fallback (old path)
            if bg_wcs is not None and doc_wcs is not None and req.overlay_opacity > 0.0:
                try:
                    overlay_u8 = _overlay_doc_on_bg(bg, bg_wcs, doc_image, doc_wcs, alpha=req.overlay_opacity)
                    ax.imshow(overlay_u8, origin="lower")
                except Exception:
                    ax.imshow(bg, origin="lower")
            else:
                ax.imshow(bg, origin="lower")


        # footprint polygon in pixel coords
        if bg_wcs is not None:
            try:
                xs, ys = bg_wcs.world_to_pixel(corners)
                ax.plot(
                    [xs[0], xs[1], xs[2], xs[3], xs[0]],
                    [ys[0], ys[1], ys[2], ys[3], ys[0]],
                    linewidth=2,
                    transform=ax.get_transform("pixel"),
                )
            except Exception:
                pass

    # center crosshair
    ax.plot([0.5], [0.5], marker="+", markersize=20, transform=ax.transAxes)

    # top-left info text
    ra = float(center[0].ra.deg)
    dec = float(center[0].dec.deg)
    ax.text(
        0.02, 0.98,
        f"{req.survey}  |  {req.scale_mult}×FOV\nRA {ra:.6f}°  Dec {dec:.6f}°\nFOV ~ {fov_deg*60:.1f}′",
        transform=ax.transAxes, va="top"
    )

    # ---- grid + RA/Dec labels ----
    if bg_wcs is not None:
        # RA/Dec edge labels
        try:
            ax.coords[0].set_axislabel("RA")
            ax.coords[1].set_axislabel("Dec")
        except Exception:
            pass

        # toggle grid lines
        try:
            ax.coords.grid(bool(req.show_grid), alpha=0.35)
        except Exception:
            pass

        # If grid is off, you may still want edge tick labels; keep axis visible.
        # WCSAxes handles ticks/labels automatically.
    else:
        # fallback: pixel grid only
        if req.show_grid:
            ax.set_axis_on()
            ax.set_xticks(np.linspace(0, req.out_px, 7))
            ax.set_yticks(np.linspace(0, req.out_px, 7))
            ax.grid(True, alpha=0.35)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()

    # After background + grid setup (before overlays that need bbox measurements)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ---- star names overlay ----
    if getattr(req, "show_star_names", False) and (bg_wcs is not None):
        try:
            _draw_star_names(
                ax, bg_wcs, center[0], fov_deg,
                mag_limit=float(getattr(req, "star_mag_limit", 2.0)),
                max_labels=int(getattr(req, "star_max_labels", 30)), renderer=renderer
            )
        except Exception:
            pass

    # ---- deep-sky overlay ----
    if getattr(req, "show_dso", False):

        if bg_wcs is not None:
            try:
                _draw_dso_overlay(ax, bg_wcs, center[0], fov_deg, req, renderer=renderer)
            except Exception as e:
                print(f"[DSO] overlay error: {type(e).__name__}: {e}")

    # ---- compass + scale bar ----
    if bg_wcs is not None and getattr(req, "show_compass", True):
        try:
            _draw_compass_NE(ax, bg_wcs, int(req.out_px), float(fov_deg))
        except Exception:
            pass

    if bg_wcs is not None and getattr(req, "show_scale_bar", True):
        try:
            _draw_scale_bar(ax, bg_wcs, int(req.out_px), float(fov_deg))
        except Exception:
            pass

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


class FinderChartDialog(QDialog):
    """
    Minimal v1 Finder Chart dialog:
    - Survey dropdown
    - Size multiplier dropdown
    - Show grid checkbox
    - Render preview
    - Save PNG
    - Send to New Document (push into SASpro)
    """
    def __init__(self, doc, settings, parent=None):
        super().__init__(parent)
        self._doc = doc
        self._settings = settings
        self._last_rgb_u8: Optional[np.ndarray] = None
        # ---- HiPS cache (avoid refetching for UI-only changes) ----
        self._hips_cache_key = None
        self._hips_bg = None          # float01 RGB background
        self._hips_wcs = None         # celestial WCS for background
        self._hips_err = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_debounced_fire)
        self._pending_force_refetch = False
        self._base_cache_key = None
        self._base_u8 = None        
        # Cached geometry derived from the current doc WCS (used for overlays/labels)
        self._doc_wcs_cached = None
        self._corners_cached = None
        self._center_cached = None
        self._fov_deg_cached = None
        self.setWindowTitle("Finder Chart…")
        self.setModal(False)
        self.resize(920, 980)

        root = QVBoxLayout(self)

        # controls row 1 (primary)
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Survey:"))
        self.cmb_survey = QComboBox()
        self.cmb_survey.addItems(["DSS2", "Pan-STARRS", "Gaia"])
        row1.addWidget(self.cmb_survey)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Size:"))
        self.cmb_size = QComboBox()
        self.cmb_size.addItems(["2× FOV", "4× FOV", "8× FOV"])
        row1.addWidget(self.cmb_size)

        row1.addSpacing(12)
        self.chk_grid = QCheckBox("Show grid")
        row1.addWidget(self.chk_grid)

        row1.addSpacing(12)
        row1.addWidget(QLabel("Output px:"))
        self.sb_px = QSpinBox()
        self.sb_px.setRange(300, 2400)
        self.sb_px.setSingleStep(100)
        self.sb_px.setValue(900)
        row1.addWidget(self.sb_px)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Image opacity:"))
        self.sld_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sld_opacity.setRange(0, 100)
        self.sld_opacity.setValue(35)
        self.sld_opacity.setFixedWidth(140)
        row1.addWidget(self.sld_opacity)

        self.lbl_opacity = QLabel("35%")
        self.lbl_opacity.setFixedWidth(40)
        row1.addWidget(self.lbl_opacity)
        row1.addStretch(1)
        self.btn_render = QPushButton("Render")
        row1.addWidget(self.btn_render)


        root.addLayout(row1)

        # controls row 2 (overlays)
        row2 = QHBoxLayout()

        self.chk_stars = QCheckBox("Star names")
        row2.addWidget(self.chk_stars)

        row2.addSpacing(8)
        row2.addWidget(QLabel("Star ≤"))
        self.sb_star_mag = QDoubleSpinBox()
        self.sb_star_mag.setRange(-2.0, 8.0)
        self.sb_star_mag.setSingleStep(0.5)
        self.sb_star_mag.setValue(5.0)
        self.sb_star_mag.setFixedWidth(70)
        row2.addWidget(self.sb_star_mag)

        row2.addWidget(QLabel("Max"))
        self.sb_star_max = QSpinBox()
        self.sb_star_max.setRange(5, 200)
        self.sb_star_max.setValue(30)
        self.sb_star_max.setFixedWidth(60)
        row2.addWidget(self.sb_star_max)

        row2.addSpacing(12)
        self.chk_dso = QCheckBox("Deep-sky")
        row2.addWidget(self.chk_dso)

        self.cmb_dso = QComboBox()
        self.cmb_dso.addItems(["All (DSO)", "M", "NGC", "IC", "Abell", "SH2", "LBN", "LDN", "PN-G"])
        self.cmb_dso.setFixedWidth(170)
        row2.addWidget(self.cmb_dso)

        row2.addWidget(QLabel("Mag ≤"))
        self.sb_dso_mag = QDoubleSpinBox()
        self.sb_dso_mag.setRange(-2.0, 30.0)
        self.sb_dso_mag.setSingleStep(0.5)
        self.sb_dso_mag.setValue(12.0)
        self.sb_dso_mag.setFixedWidth(70)
        row2.addWidget(self.sb_dso_mag)

        row2.addWidget(QLabel("Max"))
        self.sb_dso_max = QSpinBox()
        self.sb_dso_max.setRange(5, 300)
        self.sb_dso_max.setValue(30)
        self.sb_dso_max.setFixedWidth(60)
        row2.addWidget(self.sb_dso_max)

        row2.addSpacing(12)
        self.chk_compass = QCheckBox("Compass")
        self.chk_compass.setChecked(True)
        row2.addWidget(self.chk_compass)

        self.chk_scale = QCheckBox("Scale bar")
        self.chk_scale.setChecked(True)
        row2.addWidget(self.chk_scale)



        row2.addStretch(1)
        root.addLayout(row2)

        # preview
        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl.setMinimumHeight(700)
        self.lbl.setStyleSheet("QLabel { background:#111; border:1px solid #333; }")
        root.addWidget(self.lbl, 1)

        # buttons
        brow = QHBoxLayout()

        self.lbl_status = QLabel("")                 # <-- NEW
        self.lbl_status.setStyleSheet("color:#bbb;") # subtle
        self.lbl_status.setMinimumWidth(160)
        brow.addWidget(self.lbl_status)              # <-- NEW

        brow.addStretch(1)

        self.btn_send = QPushButton("Send to New Document")
        self.btn_save = QPushButton("Save PNG…")
        self.btn_close = QPushButton("Close")
        brow.addWidget(self.btn_send)
        brow.addWidget(self.btn_save)
        brow.addWidget(self.btn_close)
        root.addLayout(brow)

        self.btn_render.clicked.connect(lambda: self._render_now(force_refetch=True))
        self.btn_save.clicked.connect(self._save_png)
        self.btn_send.clicked.connect(self._send_to_new_doc)
        self.btn_close.clicked.connect(self.close)

        # grid: debounced (no refetch)
        self.chk_grid.toggled.connect(lambda _=False: self._schedule_render(force_refetch=False, delay_ms=150))

        # survey/size/px: immediate refetch (or debounce if you want, but refetch is required)
        self.cmb_survey.currentIndexChanged.connect(lambda _=0: self._render_now(force_refetch=True))
        self.cmb_size.currentIndexChanged.connect(lambda _=0: self._render_now(force_refetch=True))
        self.sb_px.valueChanged.connect(lambda _=0: self._render_now(force_refetch=True))

        # opacity: update label + debounce render (no refetch)
        self.sld_opacity.valueChanged.connect(self._on_opacity_changed)
        self.sld_opacity.sliderReleased.connect(lambda: self._render_now(force_refetch=False))
        # auto render once
        # placeholder so the user sees *something* immediately
        self.lbl.setText("Fetching survey background…")
        self.lbl.setStyleSheet("QLabel { background:#111; border:1px solid #333; color:#ccc; }")
        # --- overlay enable/disable: update enabled state + one refresh ---
        self.chk_stars.toggled.connect(lambda _=False: (self._set_overlay_controls_enabled(),
                                                       self._schedule_render(force_refetch=False, delay_ms=150)))

        self.chk_dso.toggled.connect(lambda _=False: (self._set_overlay_controls_enabled(),
                                                     self._schedule_render(force_refetch=False, delay_ms=150)))

        # --- star params: ONLY refresh if Star names is ON ---
        self.sb_star_mag.valueChanged.connect(lambda _=0: self._maybe_schedule_stars(150))
        self.sb_star_max.valueChanged.connect(lambda _=0: self._maybe_schedule_stars(150))

        # --- dso params: ONLY refresh if Deep-sky is ON ---
        self.cmb_dso.currentIndexChanged.connect(lambda _=0: self._maybe_schedule_dso(150))
        self.sb_dso_mag.valueChanged.connect(lambda _=0: self._maybe_schedule_dso(150))
        self.sb_dso_max.valueChanged.connect(lambda _=0: self._maybe_schedule_dso(150))

        self.chk_compass.toggled.connect(lambda _=False: self._schedule_render(force_refetch=False, delay_ms=150))
        self.chk_scale.toggled.connect(lambda _=False: self._schedule_render(force_refetch=False, delay_ms=150))

        self._set_overlay_controls_enabled()
        # kick initial render AFTER the dialog has had a chance to show/paint
        QTimer.singleShot(0, self._initial_render)

    def _maybe_schedule_stars(self, delay_ms: int = 150):
        # Only auto-refresh if the overlay is enabled
        if not self.chk_stars.isChecked():
            return
        self._schedule_render(force_refetch=False, delay_ms=delay_ms)

    def _maybe_schedule_dso(self, delay_ms: int = 150):
        # Only auto-refresh if the overlay is enabled
        if not self.chk_dso.isChecked():
            return
        self._schedule_render(force_refetch=False, delay_ms=delay_ms)

    def _set_overlay_controls_enabled(self):
        stars_on = self.chk_stars.isChecked()
        self.sb_star_mag.setEnabled(stars_on)
        self.sb_star_max.setEnabled(stars_on)

        dso_on = self.chk_dso.isChecked()
        self.cmb_dso.setEnabled(dso_on)
        self.sb_dso_mag.setEnabled(dso_on)
        self.sb_dso_max.setEnabled(dso_on)


    def _set_busy(self, busy: bool, msg: str = "Rendering…"):
        self.btn_render.setEnabled(not busy)
        self.btn_send.setEnabled((not busy) and (self._last_rgb_u8 is not None))
        self.btn_save.setEnabled((not busy) and (self._last_rgb_u8 is not None))

        if hasattr(self, "lbl_status") and self.lbl_status is not None:
            self.lbl_status.setText(msg if busy else "")
            self.lbl_status.setVisible(True)

        # ensures the label paints immediately before heavy work
        QApplication.processEvents()


    def _initial_render(self):
        self._set_busy(True, "Fetching survey background…")
        # schedule again so the UI paints the busy message + cursor first
        QTimer.singleShot(0, lambda: self._render_now(force_refetch=True))


    def _schedule_render(self, *, force_refetch: bool = False, delay_ms: int = 200):
        # show immediate feedback during debounce
        if hasattr(self, "lbl_status") and self.lbl_status is not None:
            self.lbl_status.setText("Rendering…")
            QApplication.processEvents()

        self._pending_force_refetch = self._pending_force_refetch or bool(force_refetch)
        self._render_timer.stop()
        self._render_timer.start(int(delay_ms))

    def _render_debounced_fire(self):
        force = bool(self._pending_force_refetch)
        self._pending_force_refetch = False
        self._render(force_refetch=force)

    def _render_now(self, *, force_refetch: bool = False):
        # Cancel any pending debounced render and render immediately
        self._render_timer.stop()
        self._pending_force_refetch = False
        self._render(force_refetch=force_refetch)


    def _compute_doc_geometry(self, img: np.ndarray, meta: dict, req: FinderChartRequest):
        doc_wcs = get_doc_wcs(meta)
        if doc_wcs is None:
            return None, None, None, None

        H, Wimg = img.shape[:2]
        corners, center = image_footprint_sky(doc_wcs, Wimg, H)
        fov_w, fov_h = estimate_fov_deg(corners)
        fov = max(fov_w, fov_h) * float(req.scale_mult)
        return doc_wcs, corners, center, float(fov)

    def _ensure_hips_background(self, req: FinderChartRequest, center: SkyCoord, fov_deg: float, *, force: bool = False):
        # Overscan factor: enough to cover the half-diagonal circle of the final square
        # sqrt(2) covers exactly; add a hair for safety near edges.
        s = float(math.sqrt(2.0) * 1.02)

        out_px = int(req.out_px)
        fetch_px = int(math.ceil(out_px * s))

        # IMPORTANT: scale FOV by the same factor so arcsec/px stays the same
        fetch_fov = float(fov_deg) * s

        # Cache key must reflect the *fetch* parameters, not just final output
        key = (
            str(req.survey),
            int(fetch_px),
            round(float(center.ra.deg), 8),
            round(float(center.dec.deg), 8),
            round(float(fetch_fov), 8),
        )

        if (not force) and (self._hips_cache_key == key) and (self._hips_bg is not None):
            return

        bg_big, wcs_big, err = try_fetch_hips_cutout(
            center,
            fov_deg=fetch_fov,
            out_px=fetch_px,
            survey_label=req.survey,
        )

        if bg_big is not None:
            # Crop back down to the user-requested size, and shift WCS accordingly
            bg, wcs_cropped, _ = _crop_center(bg_big, wcs_big, out_px)
        else:
            bg, wcs_cropped = None, None

        self._hips_cache_key = key
        self._hips_bg = bg
        self._hips_wcs = wcs_cropped
        self._hips_err = err


    def _doc_key(self, img: np.ndarray, meta: dict) -> tuple:
        # cheap-ish: shape + metadata wcs fingerprint (or header checksum if you have one)
        w = meta.get("wcs")
        w_id = id(w) if w is not None else id(meta.get("original_header") or meta.get("fits_header") or meta.get("header"))
        return (img.shape, w_id, int(self.cmb_size.currentIndex()))  # size affects scale_mult

    def _compute_doc_geometry_cached(self, img, meta, req):
        key = self._doc_key(img, meta)
        if getattr(self, "_geom_cache_key", None) == key and self._doc_wcs_cached is not None:
            return self._doc_wcs_cached, self._corners_cached, self._center_cached, self._fov_deg_cached

        doc_wcs, corners, center, fov_deg = self._compute_doc_geometry(img, meta, req)
        self._geom_cache_key = key
        self._doc_wcs_cached, self._corners_cached, self._center_cached, self._fov_deg_cached = doc_wcs, corners, center, fov_deg
        return doc_wcs, corners, center, fov_deg


    def _req(self) -> FinderChartRequest:
        survey = str(self.cmb_survey.currentText())
        mult = {0: 2, 1: 4, 2: 8}.get(int(self.cmb_size.currentIndex()), 2)
        show_grid = bool(self.chk_grid.isChecked())
        out_px = int(self.sb_px.value())
        overlay_opacity = float(self.sld_opacity.value()) / 100.0

        return FinderChartRequest(
            survey=survey,
            scale_mult=mult,
            show_grid=show_grid,

            show_star_names=bool(self.chk_stars.isChecked()),
            star_mag_limit=float(self.sb_star_mag.value()),
            star_max_labels=int(self.sb_star_max.value()),

            show_dso=bool(self.chk_dso.isChecked()),
            dso_catalog=str(self.cmb_dso.currentText()),
            dso_mag_limit=float(self.sb_dso_mag.value()),
            dso_max_labels=int(self.sb_dso_max.value()),

            show_compass=bool(self.chk_compass.isChecked()),
            show_scale_bar=bool(self.chk_scale.isChecked()),

            out_px=out_px,
            overlay_opacity=overlay_opacity,
        )


    def _render(self, *, force_refetch: bool = False):
        self._set_busy(True, "Rendering finder chart…")
        try:
            img = np.asarray(self._doc.image)
            meta = dict(getattr(self._doc, "metadata", None) or {})
            req = self._req()

            # 1) compute geometry from doc WCS
            doc_wcs, corners, center, fov_deg = self._compute_doc_geometry_cached(img, meta, req)
            if doc_wcs is None:
                QMessageBox.warning(self, "Finder Chart", "Could not render finder chart (missing WCS).")
                return

            # cache these for reuse (overlay / footprint / labels)
            self._doc_wcs_cached = doc_wcs
            self._corners_cached = corners
            self._center_cached = center
            self._fov_deg_cached = fov_deg

            # 2) fetch background only if needed
            self._ensure_hips_background(req, center[0], fov_deg, force=force_refetch)
            # 2.5) build/cache base raster (NO re-warp on overlay toggles)
            self._ensure_base_raster(req, img, doc_wcs, corners, center, fov_deg)
            # 3) render using cached background (NO network)


            rgb = render_finder_chart_cached(
                doc_image=img,
                doc_wcs=doc_wcs,
                corners=corners,
                center=center,
                fov_deg=fov_deg,
                req=req,
                bg=self._hips_bg,
                bg_wcs=self._hips_wcs,
                err=self._hips_err,
                base_u8=self._base_u8,
            )

            self._last_rgb_u8 = rgb
            qimg = _rgb_u8_to_qimage(rgb).copy()
            self.lbl.setPixmap(QPixmap.fromImage(qimg))

        except Exception as e:
            QMessageBox.critical(self, "Finder Chart", str(e))
        finally:
            self._set_busy(False)

    def _ensure_base_raster(self, req, img, doc_wcs, corners, center, fov_deg):
        doc_sig = (img.shape, str(type(self._doc)), id(self._doc))
        base_key = (
            self._hips_cache_key,                 # ties to survey/out_px/center/fov
            round(req.overlay_opacity, 4),
            id(doc_wcs),
            doc_sig,
        )
        if getattr(self, "_base_cache_key", None) == base_key:
            return

        if self._hips_bg is None:
            self._base_u8 = None
            self._base_cache_key = base_key
            return

        if self._hips_wcs is not None and req.overlay_opacity > 0:
            self._base_u8 = _overlay_doc_on_bg(self._hips_bg, self._hips_wcs, img, doc_wcs, alpha=req.overlay_opacity)
        else:
            self._base_u8 = (np.clip(self._hips_bg, 0, 1) * 255.0 + 0.5).astype(np.uint8)

        self._base_cache_key = base_key


    def _on_opacity_changed(self, v: int):
        self.lbl_opacity.setText(f"{int(v)}%")
        self._schedule_render(delay_ms=200)  # no force refetch


    def _save_png(self):
        if self._last_rgb_u8 is None:
            QMessageBox.information(self, "Finder Chart", "Nothing rendered yet.")
            return

        start_dir = ""
        try:
            start_dir = self._settings.value("finder_chart/last_dir", "", type=str) or ""
        except Exception:
            start_dir = ""

        fn, _ = QFileDialog.getSaveFileName(self, "Save Finder Chart", start_dir, "PNG Image (*.png)")
        if not fn:
            return

        try:
            if not fn.lower().endswith(".png"):
                fn += ".png"
            qimg = _rgb_u8_to_qimage(self._last_rgb_u8).copy()
            ok = qimg.save(fn, "PNG")
            if not ok:
                raise RuntimeError("QImage.save() failed.")
            try:
                self._settings.setValue("finder_chart/last_dir", fn)
                self._settings.sync()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Finder Chart", str(e))

    def _send_to_new_doc(self):
        if self._last_rgb_u8 is None:
            QMessageBox.information(self, "Finder Chart", "Nothing rendered yet.")
            return

        img01 = self._last_rgb_u8.astype(np.float32) / 255.0

        req = self._req()
        meta = {
            "step_name": "Finder Chart",
            "finder_chart": {
                "survey": req.survey,
                "scale_mult": req.scale_mult,
                "show_grid": req.show_grid,
                "out_px": req.out_px,
                "overlay_opacity": req.overlay_opacity,
            },
        }

        dm = self._get_doc_manager()
        if dm is None:
            QMessageBox.warning(self, "Finder Chart", "DocManager not found.")
            return

        title = f"Finder Chart ({req.survey})"

        try:
            if hasattr(dm, "open_array"):
                # matches PerfectPalettePicker
                dm.open_array(img01, metadata=meta, title=title)
                return

            if hasattr(dm, "create_document"):
                # PPP fallback
                doc = dm.create_document(image=img01, metadata=meta, name=title)
                if hasattr(dm, "add_document"):
                    dm.add_document(doc)
                return

            raise RuntimeError("DocManager lacks open_array/create_document")

        except Exception as e:
            QMessageBox.critical(self, "Finder Chart", f"Failed to open new view:\n{e}")


    def _get_doc_manager(self):
        mw = self.parent()
        if mw is None:
            return None
        return getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)

    