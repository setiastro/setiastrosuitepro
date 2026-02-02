# src/setiastro/saspro/magnitude_tool.py
# SASpro Magnitude / Surface Brightness (mag/arcsec^2) — v0
#
# Assumptions:
#   - Image is LINEAR (like SFCC).
#   - Image is RGB float in [0,1] OR uint8 (we normalize). If you have other dtypes,
#     you can extend _to_float_rgb() the same way you do elsewhere.
#   - WCS header is available in doc.metadata["original_header"] (or similar) like SFCC.
#
# Strategy:
#   1) Reuse SFCC’s SIMBAD star_list pipeline (Fetch Stars) for catalog B/V/R mags + pixel coords.
#   2) Use SEP to detect stars (for centroid-ish matching and optional radius sanity).
#   3) Do aperture photometry (per channel) on the ORIGINAL linear image (not clamped/pedestal-removed).
#   4) Compute per-channel photometric zero point:
#          ZP = m_cat + 2.5 log10( flux / exptime )
#      (no gain required; ZP is in “mag per ADU/sec (or per normalized unit/sec)”)
#   5) For a user region (object rect + background rect), compute:
#        - integrated magnitude per channel
#        - surface brightness mag/arcsec^2 per channel (if pixscale known)
#
# Notes:
#   - This is intentionally “initial version”: no color-term modeling. You can add a color term later.
#   - Robustness: sigma-clip the ZP list per channel.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import numpy.ma as ma

import sep
import time
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.vizier import Vizier

from PyQt6.QtCore import Qt, QRect, QEvent, QPointF, QRectF, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget, QWidget,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,QComboBox, QGraphicsPathItem, QGraphicsEllipseItem,
    QMessageBox, QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem
)
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QPainterPath
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.wcs.wcs import NoConvergence
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# IMPORTANT: use the centralized one (adjust import path to where you moved it)
from setiastro.saspro.sfcc import pickles_match_for_simbad

# Reuse useful SFCC pieces
from setiastro.saspro.sfcc import non_blocking_sleep  # already used in SFCC; optional
from setiastro.saspro.backgroundneutral import auto_rect_box, auto_rect_50x50
from setiastro.saspro.imageops.stretch import stretch_color_image
# We *intentionally* do NOT reuse SFCC pedestal-removal/clamp for photometry.

import socket

import multiprocessing as mp
import queue
import traceback

def _run_in_subprocess(timeout_s: float, target, *args, **kwargs):
    """
    Run `target(*args, **kwargs)` in a fresh subprocess.
    If it hangs (TLS/SSL wedged), terminate the process.
    Returns target result or raises.
    """
    ctx = mp.get_context("spawn")  # safest on Windows/macOS
    q = ctx.Queue()

    def _worker(q_, args_, kwargs_):
        try:
            out = target(*args_, **kwargs_)
            q_.put(("ok", out))
        except Exception as e:
            q_.put(("err", (repr(e), traceback.format_exc())))

    p = ctx.Process(target=_worker, args=(q, args, kwargs))
    p.daemon = True
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(2.0)
        raise TimeoutError(f"Catalog query timed out after {timeout_s:.1f}s (killed subprocess).")

    try:
        kind, payload = q.get_nowait()
    except queue.Empty:
        raise RuntimeError("Catalog subprocess ended without returning a result.")

    if kind == "ok":
        return payload

    err_repr, tb = payload
    raise RuntimeError(f"Catalog query failed: {err_repr}\n{tb}")

def _row_get(tab_row, colname: str):
    try:
        return tab_row[colname]
    except Exception:
        return None

def _simbad_query_worker(center_ra_deg: float, center_dec_deg: float, radius_deg: float,
                        servers: list[str], hard_timeout_s: float, row_limit: int):
    # Import inside subprocess
    from astroquery import conf as aq_conf
    from astroquery.simbad import Simbad, conf as simbad_conf
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    aq_conf.timeout = float(hard_timeout_s)
    try:
        simbad_conf.timeout = float(hard_timeout_s)
    except Exception:
        pass

    Simbad.reset_votable_fields()
    # try “new” fields first, then fallback
    try:
        Simbad.add_votable_fields("sp", "B", "V", "R", "ra", "dec")
    except Exception:
        Simbad.add_votable_fields("sp", "flux(B)", "flux(V)", "flux(R)", "ra(d)", "dec(d)")

    Simbad.ROW_LIMIT = int(row_limit)

    center = SkyCoord(center_ra_deg * u.deg, center_dec_deg * u.deg, frame="icrs")

    last_err = None
    for server in (servers or []):
        try:
            try:
                simbad_conf.server = server
            except Exception:
                pass
            tab = Simbad.query_region(center, radius=radius_deg * u.deg)
            if tab is None:
                continue
            # Return as “simple” python structures (picklable)
            return {
                "colnames": list(tab.colnames),
                "rows": [ {c: tab[i][c] for c in tab.colnames} for i in range(len(tab)) ]
            }
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"SIMBAD failed on all servers. Last error: {last_err!r}")


def _tcp_reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """
    Fast reachability check that catches DNS failures + connect stalls.
    This is *not* a full HTTP check; it's enough to avoid UI hangs.
    """
    try:
        # create_connection does DNS + connect and obeys timeout_s
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


# ---------------------------- helpers ----------------------------

def _unmask_num(x):
    try:
        if x is None:
            return None
        if ma.isMaskedArray(x) and ma.is_masked(x):
            return None
        return float(x)
    except Exception:
        return None


def _to_float_image(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        # mono
        if a.dtype == np.uint8:
            return (a.astype(np.float32) / 255.0).astype(np.float32, copy=False)
        return a.astype(np.float32, copy=False)

    if a.ndim == 3:
        if a.shape[2] == 1:
            return _to_float_image(a[..., 0])
        if a.shape[2] >= 3:
            a = a[..., :3]
            if a.dtype == np.uint8:
                return (a.astype(np.float32) / 255.0).astype(np.float32, copy=False)
            return a.astype(np.float32, copy=False)

    raise ValueError("Unsupported image shape for magnitude tool.")

def _mask_sum(img_f: np.ndarray, mask: np.ndarray):
    m = mask.astype(bool)
    if img_f.ndim == 2:
        return float(np.sum(img_f[m], dtype=np.float64))
    else:
        v = img_f[..., :3][m]
        return v.reshape(-1, 3).sum(axis=0).astype(np.float64)

def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))

def _build_wcs_and_pixscale(header) -> Tuple[Optional[WCS], Optional[float]]:
    if header is None:
        return None, None
    try:
        hdr = header.copy()
    except Exception:
        try:
            hdr = fits.Header(header)
        except Exception:
            return None, None

    try:
        wcs = WCS(hdr, naxis=2, relax=True)
    except Exception:
        return None, None

    pixscale = None
    try:
        wcs2 = wcs.celestial if hasattr(wcs, "celestial") else wcs
        psm = wcs2.pixel_scale_matrix
        pixscale = float(np.hypot(psm[0, 0], psm[1, 0]) * 3600.0)  # arcsec/px
    except Exception:
        pixscale = None

    return wcs, pixscale

def _rect_pixels(img_f: np.ndarray, rect: QRect) -> np.ndarray:
    H, W = img_f.shape[:2]
    x0 = max(0, rect.left()); y0 = max(0, rect.top())
    x1 = min(W, rect.right() + 1); y1 = min(H, rect.bottom() + 1)
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=np.float32)
    patch = img_f[y0:y1, x0:x1]
    return patch.reshape(-1, 3) if (img_f.ndim == 3) else patch.reshape(-1)

def _bg_stats(img_f: np.ndarray, bg_rect: QRect):
    p = _rect_pixels(img_f, bg_rect)
    if p.size == 0:
        return None
    # robust: median + MAD->sigma
    if img_f.ndim == 2:
        med = float(np.median(p))
        mad = float(np.median(np.abs(p - med)))
        sigma = 1.4826 * mad
        mean = float(np.mean(p))
        return {"mean": mean, "median": med, "sigma": sigma}
    else:
        med = np.median(p, axis=0)
        mad = np.median(np.abs(p - med), axis=0)
        sigma = 1.4826 * mad
        mean = np.mean(p, axis=0)
        return {"mean": mean.astype(float), "median": med.astype(float), "sigma": sigma.astype(float)}

_LOGC = 2.5 / math.log(10.0)  # 1.085736...

def _mag_err_from_flux(flux: float, flux_err: float, zp_err: float) -> Optional[float]:
    if not (flux > 0) or flux_err is None or zp_err is None:
        return None
    # clamp
    flux_err = max(0.0, float(flux_err))
    return float(math.sqrt(zp_err*zp_err + (_LOGC * (flux_err / float(flux)))**2))

def _mu_err_from_flux(flux: float, flux_err: float, zp_err: float) -> Optional[float]:
    # same functional form as magnitude; dividing by area doesn't change relative error in flux
    return _mag_err_from_flux(flux, flux_err, zp_err)


def _sigma_clip(vals: np.ndarray, sigma: float = 2.5, iters: int = 3) -> np.ndarray:
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return v
    for _ in range(max(1, int(iters))):
        med = np.median(v)
        sd = np.std(v)
        if not np.isfinite(sd) or sd <= 0:
            break
        keep = np.abs(v - med) <= sigma * sd
        if keep.sum() == v.size:
            break
        v = v[keep]
        if v.size < 3:
            break
    return v


def _detect_sources(gray: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    bkg = sep.Background(gray)
    data_sub = gray - bkg.back()
    err = float(bkg.globalrms)
    sources = sep.extract(data_sub, float(sigma), err=err)
    return sources


def _match_starlist_to_sources(star_list: List[dict], sources: np.ndarray, max_dist_px: float = 3.0) -> List[dict]:
    if sources is None or sources.size == 0:
        return []
    sx = sources["x"].astype(np.float64)
    sy = sources["y"].astype(np.float64)

    out = []
    r2max = float(max_dist_px) ** 2

    for st in star_list:
        x0 = float(st.get("x", np.nan))
        y0 = float(st.get("y", np.nan))
        if not (np.isfinite(x0) and np.isfinite(y0)):
            continue
        dx = sx - x0
        dy = sy - y0
        j = int(np.argmin(dx * dx + dy * dy))
        if float(dx[j] * dx[j] + dy[j] * dy[j]) <= r2max:
            out.append({
                "star": st,
                "src": sources[j],
                "x": float(sx[j]),
                "y": float(sy[j]),
            })
    return out

def _aperture_photometry_mono(img_f, xs, ys, r_ap, r_in, r_out):
    x = np.ascontiguousarray(np.asarray(xs, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(ys, dtype=np.float64))
    plane = np.ascontiguousarray(np.asarray(img_f, dtype=np.float32))

    f_ap, _, _ = sep.sum_circle(plane, x, y, r=float(r_ap), subpix=5)
    bkg, _, _ = sep.sum_circann(plane, x, y, float(r_in), float(r_out), subpix=5)

    ap_area = np.pi * float(r_ap) * float(r_ap)
    ann_area = np.pi * (float(r_out) * float(r_out) - float(r_in) * float(r_in))
    bkg_per_pix = bkg / max(ann_area, 1e-8)

    flux_net = (f_ap - bkg_per_pix * ap_area).astype(np.float32)
    return flux_net


def _aperture_photometry_rgb(img_f, xs, ys, r_ap, r_in, r_out):
    import numpy as np
    import sep

    # SEP wants float arrays; also make sure x/y are contiguous float64
    x = np.ascontiguousarray(np.asarray(xs, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(ys, dtype=np.float64))

    # ensure the image itself is contiguous float32
    img_f = np.ascontiguousarray(np.asarray(img_f, dtype=np.float32))

    # split planes and make each plane contiguous too
    planes = [
        np.ascontiguousarray(img_f[..., 0]),
        np.ascontiguousarray(img_f[..., 1]),
        np.ascontiguousarray(img_f[..., 2]),
    ]

    flux_net = np.zeros((len(x), 3), dtype=np.float32)

    for c, plane in enumerate(planes):
        # aperture sum
        f_ap, f_aperr, _ = sep.sum_circle(plane, x, y, r=float(r_ap), subpix=5)

        # annulus background (per-star)
        bkg, bkgerr, _ = sep.sum_circann(plane, x, y, float(r_in), float(r_out), subpix=5)

        # convert annulus sum to per-pixel background then to aperture background
        ap_area = np.pi * float(r_ap) * float(r_ap)
        ann_area = np.pi * (float(r_out) * float(r_out) - float(r_in) * float(r_in))
        bkg_per_pix = bkg / max(ann_area, 1e-8)

        flux_net[:, c] = (f_ap - bkg_per_pix * ap_area).astype(np.float32)

    return flux_net, None

def _compute_zero_points_mono(matches, img_f, r_ap, r_in, r_out, band="L", clip_sigma=2.5):
    xs = np.array([m["x"] for m in matches], dtype=np.float64)
    ys = np.array([m["y"] for m in matches], dtype=np.float64)

    flux_net = _aperture_photometry_mono(img_f, xs, ys, r_ap, r_in, r_out)

    magkey = _magkey_for_band(band)

    zps = []
    used = 0
    x, y = [], []
    for i, m in enumerate(matches):
        st = m["star"]
        f = float(flux_net[i])
        if not (f > 0):
            continue

        mag = _unmask_num(st.get(magkey))
        if mag is None:
            continue

        zps.append(float(mag) + 2.5 * math.log10(f))
        x.append(float(-2.5 * math.log10(f)))
        y.append(float(mag))
        used += 1

    zps = _sigma_clip(np.asarray(zps, dtype=np.float64), sigma=clip_sigma)
    out = {
        "ZP": float(np.median(zps)) if zps.size else None,
        "n": int(zps.size),
        "std": float(np.std(zps)) if zps.size else None,
        "sem": (float(np.std(zps))/math.sqrt(int(zps.size))) if zps.size > 1 else None,
        "band": band,
        "magkey": magkey,
        "used_matches": used,
        "plot": {
            "Mono": {
                "x": x, "y": y,
                "zp": float(np.median(zps)) if zps.size else None,
                "title": f"Mono: {magkey} vs m_inst (y = x + ZP)",
            }
        }
    }
    return out

def _compute_zero_points(
    matches: List[dict],
    img_f: np.ndarray,
    r_ap: float,
    r_in: float,
    r_out: float,
    clip_sigma: float = 2.5,
) -> Dict[str, Any]:
    """
    Build per-channel ZP from catalog mags:
      Blue  -> Bmag
      Green -> Vmag
      Red   -> Rmag
    """

    xs = np.array([m["x"] for m in matches], dtype=np.float64)
    ys = np.array([m["y"] for m in matches], dtype=np.float64)

    flux_net, _flux_ap = _aperture_photometry_rgb(img_f, xs, ys, r_ap, r_in, r_out)

    # Collect ZP candidates
    zp_R, zp_G, zp_B = [], [], []
    xR, yR = [], []
    xG, yG = [], []
    xB, yB = [], []

    used = 0

    for i, m in enumerate(matches):
        st = m["star"]
        fR, fG, fB = float(flux_net[i, 0]), float(flux_net[i, 1]), float(flux_net[i, 2])
        # reject non-positive net flux
        if not (fR > 0 or fG > 0 or fB > 0):
            continue
        def minst(f):
            return float(-2.5 * math.log10(f)) if f > 0 else None
        # catalog mags
        Rmag = _unmask_num(st.get("Rmag"))
        Vmag = _unmask_num(st.get("Vmag"))
        Bmag = _unmask_num(st.get("Bmag"))

        # NO flux/sec — use flux directly
        if fR > 0 and Rmag is not None:
            zp_R.append(float(Rmag) + 2.5 * math.log10(fR))
            xR.append(minst(fR)); yR.append(float(Rmag))

        if fG > 0 and Vmag is not None:
            zp_G.append(float(Vmag) + 2.5 * math.log10(fG))
            xG.append(minst(fG)); yG.append(float(Vmag))

        if fB > 0 and Bmag is not None:
            zp_B.append(float(Bmag) + 2.5 * math.log10(fB))
            xB.append(minst(fB)); yB.append(float(Bmag))

        used += 1

    zp_R = _sigma_clip(np.asarray(zp_R, dtype=np.float64), sigma=clip_sigma)
    zp_G = _sigma_clip(np.asarray(zp_G, dtype=np.float64), sigma=clip_sigma)
    zp_B = _sigma_clip(np.asarray(zp_B, dtype=np.float64), sigma=clip_sigma)

    def summarize(arr):
        if arr.size == 0:
            return None, 0, None, None
        med = float(np.median(arr))
        sd  = float(np.std(arr))
        n   = int(arr.size)
        sem = float(sd / math.sqrt(n)) if (n > 1 and np.isfinite(sd)) else None
        return med, n, sd, sem

    ZP_R, nR, sR, semR = summarize(zp_R)
    ZP_G, nG, sG, semG = summarize(zp_G)
    ZP_B, nB, sB, semB = summarize(zp_B)

    return {
        "ZP_R": ZP_R, "ZP_G": ZP_G, "ZP_B": ZP_B,
        "n_R": nR, "n_G": nG, "n_B": nB,
        "std_R": sR, "std_G": sG, "std_B": sB,
        "sem_R": semR, "sem_G": semG, "sem_B": semB,
        "used_matches": used,
        "plot": {
            "R": {"x": xR, "y": yR, "zp": ZP_R, "title": "Red channel: Rmag vs m_inst (y = x + ZP_R)"},
            "G": {"x": xG, "y": yG, "zp": ZP_G, "title": "Green channel: Vmag vs m_inst (y = x + ZP_G)"},
            "B": {"x": xB, "y": yB, "zp": ZP_B, "title": "Blue channel: Bmag vs m_inst (y = x + ZP_B)"},
        },
    }

BAND_TO_MAGKEY = {
    "L": "Vmag",
    "R": "Rmag",
    "G": "Vmag",
    "B": "Bmag",
}
def _magkey_for_band(band: str) -> str:
    b = (band or "L").strip().upper()
    return BAND_TO_MAGKEY.get(b, "Vmag")

def _rect_sum(img_f: np.ndarray, rect: QRect) -> np.ndarray:
    H, W = img_f.shape[:2]
    x0 = max(0, rect.left()); y0 = max(0, rect.top())
    x1 = min(W, rect.right() + 1); y1 = min(H, rect.bottom() + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0 if img_f.ndim == 2 else np.zeros((3,), dtype=np.float64)

    patch = img_f[y0:y1, x0:x1]
    if img_f.ndim == 2:
        return float(np.sum(patch, dtype=np.float64))
    else:
        patch = patch[..., :3]
        return patch.reshape(-1, 3).sum(axis=0).astype(np.float64)


def _rect_area_px(img_f: np.ndarray, rect: QRect) -> int:
    H, W = img_f.shape[:2]
    x0 = max(0, rect.left())
    y0 = max(0, rect.top())
    x1 = min(W, rect.right() + 1)
    y1 = min(H, rect.bottom() + 1)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _mag_from_flux(flux: float, zp: Optional[float]) -> Optional[float]:
    if zp is None or not (flux > 0):
        return None
    return float(-2.5 * math.log10(flux) + float(zp))


def _mu_from_flux(flux: float, area_asec2: float, zp: Optional[float]) -> Optional[float]:
    if zp is None or not (flux > 0) or not (area_asec2 > 0):
        return None
    return float(-2.5 * math.log10(flux / area_asec2) + float(zp))

# ---------------------------- SQM/Bortle helpers ----------------------------

_BORTLE_BINS = [
    (1, 21.99, 99.0,  "Excellent (B1)"),
    (2, 21.89, 21.99, "Typical Dark (B2)"),
    (3, 21.69, 21.89, "Rural (B3)"),
    (4, 20.49, 21.69, "Rural/Suburban (B4)"),
    (5, 19.50, 20.49, "Suburban (B5)"),
    (6, 18.94, 19.50, "Bright Suburban (B6)"),
    (7, 18.38, 18.94, "Suburban/Urban (B7)"),
    (8, 17.80, 18.38, "City (B8)"),
    (9, -99.0, 17.80, "Inner City (B9)"),
]

def _bortle_from_sqm(mu_mag_arcsec2: Optional[float]) -> Optional[dict]:
    """
    Map sky surface brightness (mag/arcsec^2) to a Bortle estimate.
    Returns dict with {bortle, label, range, mu}.
    """
    if mu_mag_arcsec2 is None:
        return None
    try:
        mu = float(mu_mag_arcsec2)
    except Exception:
        return None
    if not np.isfinite(mu):
        return None

    for bortle, lo, hi, label in _BORTLE_BINS:
        if (mu >= lo) and (mu < hi):
            return {
                "bortle": int(bortle),
                "label": str(label),
                "range": f"{lo:.2f}–{hi:.2f}" if (lo > -90 and hi < 90) else (f"≥{lo:.2f}" if hi >= 90 else f"<{hi:.2f}"),
                "mu": mu,
            }
    return None

def _header_str(header, key: str) -> str:
    try:
        if header is None:
            return ""
        v = header.get(key, "")
        return str(v).strip()
    except Exception:
        return ""

def _is_narrowband_like(header) -> bool:
    """
    Best-effort heuristic. If we can see a filter name like Ha/OIII/SII, or NB,
    we should NOT present a Bortle estimate.
    """
    if header is None:
        return False

    # Common FITS keys people have
    candidates = [
        _header_str(header, "FILTER"),
        _header_str(header, "FILTER1"),
        _header_str(header, "FILTER2"),
        _header_str(header, "FILTNAME"),
        _header_str(header, "OBJECT"),     # sometimes includes "Ha" etc (not ideal but helps)
        _header_str(header, "IMAGETYP"),
    ]
    s = " ".join([c for c in candidates if c]).upper()

    # Narrowband tokens
    nb_tokens = [
        "HA", "H-A", "Hα", "HALPHA", "H_ALPHA", "H-ALPHA",
        "OIII", "O3", "[OIII]", "O-III", "O_III",
        "SII", "S2", "[SII]", "S-II", "S_II",
        "HBETA", "H-BETA", "H_BETA",
        "NARROW", "NARROWBAND", "NB",
        "DUOBAND", "TRIBAND", "QUADBAND",
        "L-ENHANCE", "L-EXTREME", "L-ULTIMATE",
        "ALP-T", "ALPT", "ALP", "ANTLIA",  # some common LP/NB product names
    ]
    return any(tok in s for tok in nb_tokens)


class ZeroPointPlotsDialog(QDialog):
    """
    Shows ZP scatter plot(s):
      x = instrumental magnitude = -2.5 log10(flux)
      y = catalog magnitude (SIMBAD)
      overlay line y = x + ZP (slope fixed to 1)
    """
    def __init__(self, parent, plots: dict):
        super().__init__(parent)
        self.setWindowTitle("Zero Point Graphs")
        self.resize(900, 700)

        root = QVBoxLayout(self)
        tabs = QTabWidget(self)
        root.addWidget(tabs)

        # plots: { "Mono": {...} } or { "R": {...}, "G": {...}, "B": {...} }
        for name, payload in plots.items():
            w = QWidget()
            lay = QVBoxLayout(w)

            fig = Figure(figsize=(6, 4), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)

            x = payload.get("x", [])
            y = payload.get("y", [])
            zp = payload.get("zp", None)
            title = payload.get("title", name)

            ax.scatter(x, y, s=16, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("Instrumental magnitude  m_inst = -2.5 log10(flux)")
            ax.set_ylabel("Catalog magnitude  m_cat (SIMBAD)")

            # reference line for your model
            if zp is not None and len(x) > 0:
                xmin = float(min(x)); xmax = float(max(x))
                ax.plot([xmin, xmax], [xmin + float(zp), xmax + float(zp)], linewidth=2)

            ax.grid(True, alpha=0.25)

            lay.addWidget(canvas)
            tabs.addTab(w, name)

# ---------------------------- UI dialog (initial) ----------------------------
class MagnitudeRegionDialog(QDialog):
    """
    Pick ONE target rectangle (object). Background is auto-selected via auto_rect_50x50().
    Preview can be toggled to ABE hard_autostretch to see dim regions on linear data.
    """
    def __init__(self, parent, doc_manager, icon=None):
        super().__init__(parent)
        self._main = parent
        self.doc_manager = doc_manager
        self.doc = self.doc_manager.get_active_document()
        self._path = QPainterPath()
        self._path_item: QGraphicsPathItem | None = None
        self._ellipse_item: QGraphicsEllipseItem | None = None
        self._pen_live = QPen(QColor(0, 255, 0), 3, Qt.PenStyle.DashLine)
        self._pen_live.setCosmetic(True)
        self._pen_final = QPen(QColor(255, 0, 0), 3)
        self._pen_final.setCosmetic(True)
        if icon:
            try: self.setWindowIcon(icon)
            except Exception: pass

        self.setWindowTitle("Magnitude Tool — Select Target Region")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.resize(900, 600)

        self.auto_stretch = True
        self.zoom_factor = 1.0
        self._user_zoomed = False

        self.target_rect_scene = QRectF()
        self.target_item: QGraphicsRectItem | None = None
        self.bg_item: QGraphicsRectItem | None = None
        self._drawing = False
        self._origin_scene = QPointF()

        # --- scene/view ---
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._zoom_debounce_ms = 70
        self._interactive_timer = QTimer(self)
        self._interactive_timer.setSingleShot(True)
        self._interactive_timer.timeout.connect(self._end_interactive_present)

        self._interactive_active = False

        # Make interactive updates cheaper (optional but helps a lot)
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontSavePainterState, True)
        self.view.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)
        # --- layout ---
        v = QVBoxLayout(self)
        self.lbl = QLabel("Draw a Target region (Box/Ellipse/Freehand). Background will be auto-selected (gold).")

        self.lbl.setWordWrap(True)
        v.addWidget(self.lbl)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Box", "Ellipse", "Freehand"])
        v.insertWidget(1, self.mode_combo)  # under label, above view

        v.addWidget(self.view, 1)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Use Target")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_find_bg = QPushButton("Find Background")
        self.btn_toggle = QPushButton("Disable Auto-Stretch" if self.auto_stretch else "Enable Auto-Stretch")

        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_toggle)
        btn_row.addWidget(self.btn_find_bg)
        v.addLayout(btn_row)

        from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

        zoom_row = QHBoxLayout()
        self.btn_zoom_out = themed_toolbtn("zoom-out", "Zoom Out")
        self.btn_zoom_in  = themed_toolbtn("zoom-in", "Zoom In")
        self.btn_zoom_100 = themed_toolbtn("zoom-original", "1:1")
        self.btn_zoom_fit = themed_toolbtn("zoom-fit-best", "Fit")

        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.btn_zoom_in)
        zoom_row.addWidget(self.btn_zoom_100)
        zoom_row.addWidget(self.btn_zoom_fit)
        zoom_row.addStretch(1)
        v.addLayout(zoom_row)


        # wiring
        self.btn_cancel.clicked.connect(self.close)
        self.btn_toggle.clicked.connect(self._toggle_autostretch)
        self.btn_find_bg.clicked.connect(self._on_find_background)
        self.btn_zoom_in.clicked.connect(lambda: self._zoom(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(0.8))
        self.btn_zoom_100.clicked.connect(self.zoom_100)
        self.btn_zoom_fit.clicked.connect(self.fit_to_view)
        self.btn_apply.clicked.connect(self._on_use_target)

        # mouse events
        self.view.viewport().installEventFilter(self)

        # active doc tracking (optional, matches your style)
        self._connected_current_doc_changed = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_current_doc_changed = True
            except Exception:
                self._connected_current_doc_changed = False
        self.finished.connect(self._cleanup_connections)

        self._pixmap_item = None
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._load_image()
        QTimer.singleShot(0, self.fit_to_view)

    # ---------- doc helpers ----------
    def _begin_interactive_present(self):
        """
        Switch to FAST transform while user is actively zooming/panning.
        """
        if self._interactive_active:
            # restart debounce
            self._interactive_timer.start(self._zoom_debounce_ms)
            return

        self._interactive_active = True

        # FAST path: disable smooth pixmap transform
        try:
            self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        except Exception:
            pass

        # keep debounce running
        self._interactive_timer.start(self._zoom_debounce_ms)

    def _end_interactive_present(self):
        """
        Restore SMOOTH transform after interaction stops, then repaint once.
        """
        self._interactive_active = False
        try:
            self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        except Exception:
            pass

        # Force a final high-quality redraw
        try:
            self.view.viewport().update()
        except Exception:
            pass


    def _active_doc(self):
        d = self.doc_manager.get_active_document()
        return d if d is not None else self.doc

    def _doc_image_float01(self) -> np.ndarray:
        doc = self._active_doc()
        img = getattr(doc, "image", None)
        if img is None:
            raise ValueError("No active image.")

        img = np.asarray(img)
        if img.size == 0:
            raise ValueError("No active image.")

        # normalize integers → [0,1]
        if img.dtype.kind in "ui":
            mx = float(np.iinfo(img.dtype).max)
            img = img.astype(np.float32) / (mx if mx > 0 else 1.0)
        else:
            img = img.astype(np.float32, copy=False)

        # ---- allow mono for ROI picking (convert to RGB for DISPLAY only) ----
        if img.ndim == 2:
            return np.dstack([img, img, img]).astype(np.float32, copy=False)

        if img.ndim == 3:
            if img.shape[2] == 1:
                m = img[..., 0]
                return np.dstack([m, m, m]).astype(np.float32, copy=False)
            if img.shape[2] >= 3:
                return img[..., :3].astype(np.float32, copy=False)

        raise ValueError(f"Unsupported image shape for ROI picker: {img.shape}")

    def _capture_view_state(self):
        """
        Capture current view transform + center point in scene coords.
        """
        try:
            t = self.view.transform()
            center_scene = self.view.mapToScene(self.view.viewport().rect().center())
            return (t, center_scene)
        except Exception:
            return None

    def _restore_view_state(self, state):
        """
        Restore transform + center point.
        """
        if not state:
            return
        try:
            t, center_scene = state
            self.view.setTransform(t)
            self.view.centerOn(center_scene)
        except Exception:
            pass

    def _on_mode_changed(self, _idx):
        # If the user changes drawing mode, clear any existing selection
        # so it’s always “exactly one region”.
        self._clear_target_items()

    def _display_rgb01(self, img_rgb01: np.ndarray) -> np.ndarray:
        # preview-only stretch
        if not self.auto_stretch:
            return np.clip(img_rgb01, 0.0, 1.0).astype(np.float32, copy=False)

        # Use SASpro canonical stretch for preview (non-destructive; does NOT affect photometry)
        try:
            disp = stretch_color_image(
                img_rgb01,
                target_median=0.35,
                linked=True,          # better visibility for NB/odd color balance data
                normalize=False,       # keep the look stable; you can flip to True if you want punchier preview
                apply_curves=False,    # keep it “honest” and fast
                curves_boost=0.0,
                blackpoint_sigma=3.5,  # roughly similar vibe to your old "sigma=2"
                no_black_clip=False,
                hdr_compress=False,
                hdr_amount=0.0,
                hdr_knee=0.75,
                luma_only=False,
                high_range=False,
            )
            return np.clip(disp, 0.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            # worst case: just show clipped linear
            return np.clip(img_rgb01, 0.0, 1.0).astype(np.float32, copy=False)


    # ---------- render ----------
    def _load_image(self, preserve_view_state=None, keep_overlays=True):
        try:
            img = self._doc_image_float01()
        except Exception as e:
            QMessageBox.warning(self, "No Image", str(e))
            self.close()
            return

        disp = self._display_rgb01(img)
        h, w, _ = disp.shape

        self._disp_buf8 = np.ascontiguousarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))
        qimg = QImage(self._disp_buf8.data, w, h, self._disp_buf8.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        if self._pixmap_item is None or (not keep_overlays):
            self.scene.clear()
            self._pixmap_item = self.scene.addPixmap(pix)
            self._pixmap_item.setPos(0, 0)
        else:
            self._pixmap_item.setPixmap(pix)
            self._pixmap_item.setPos(0, 0)

        self.scene.setSceneRect(0, 0, pix.width(), pix.height())

        if preserve_view_state is not None:
            self._restore_view_state(preserve_view_state)
        else:
            self.view.resetTransform()
            self.view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._user_zoomed = False

        # If you're keeping overlays, DON'T blow away bg/target here.
        # Only auto-find bg if bg doesn't exist.
        if self.bg_item is None:
            self._on_find_background()


    def _toggle_autostretch(self):
        view_state = self._capture_view_state() if self._user_zoomed else None

        self.auto_stretch = not self.auto_stretch
        self.btn_toggle.setText("Disable Auto-Stretch" if self.auto_stretch else "Enable Auto-Stretch")

        # reload image but keep zoom/pan if user zoomed
        self._load_image(preserve_view_state=view_state, keep_overlays=True)

    def _zoom(self, factor: float):
        self._user_zoomed = True

        # start FAST interactive mode
        self._begin_interactive_present()

        cur = self.view.transform().m11()
        new_scale = cur * factor
        if new_scale < 0.01 or new_scale > 100.0:
            return

        self.view.scale(factor, factor)

    def zoom_100(self):
        self._user_zoomed = True
        self.view.resetTransform()
        self.view.scale(1.0, 1.0)


    def fit_to_view(self):
        self._user_zoomed = False
        self.view.resetTransform()
        if self._pixmap_item is not None:
            self.view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, e):
        super().showEvent(e)
        QTimer.singleShot(0, self.fit_to_view)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._user_zoomed:
            self.fit_to_view()

    # ---------- background auto-box ----------
    def _on_find_background(self):
        try:
            img = self._doc_image_float01()

            # Pull desired box size from the parent tool if available
            box = 50
            p = self.parent()
            if p is not None and hasattr(p, "bg_box_size"):
                try:
                    box = int(p.bg_box_size.value())
                except Exception:
                    box = 50

            x, y, w, h = auto_rect_box(img, box=box, margin=100)
            # (or auto_rect_50x50(img) if you want fixed behavior)
        except Exception as e:
            QMessageBox.warning(self, "Background", str(e))
            return

        if self.bg_item:
            try:
                self.scene.removeItem(self.bg_item)
            except Exception:
                pass

        pen = QPen(QColor(255, 215, 0), 3)  # gold
        pen.setCosmetic(True)
        rect_scene = QRectF(float(x), float(y), float(w), float(h))
        self.bg_item = self.scene.addRect(rect_scene, pen)


    def _target_mask(self) -> Optional[np.ndarray]:
        if self._pixmap_item is None:
            return None
        bounds = self._pixmap_item.boundingRect()
        W = int(bounds.width())
        H = int(bounds.height())
        if W <= 0 or H <= 0:
            return None

        mode = self.mode_combo.currentText()

        # start with empty mask
        mask_img = QImage(W, H, QImage.Format.Format_Grayscale8)
        mask_img.fill(0)

        p = QPainter(mask_img)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255))

        if mode == "Box" and not self.target_rect_scene.isNull():
            p.drawRect(self.target_rect_scene.toRect())
        elif mode == "Ellipse" and not self.target_rect_scene.isNull():
            p.drawEllipse(self.target_rect_scene)
        elif mode == "Freehand" and not self._path.isEmpty():
            p.drawPath(self._path)

        p.end()

        ptr = mask_img.bits()
        ptr.setsize(mask_img.bytesPerLine() * H)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(H, mask_img.bytesPerLine())
        arr = arr[:, :W]
        return (arr > 0)


    # ---------- target drawing ----------
    def _clear_target_items(self):
        for it in (self.target_item, self._ellipse_item, self._path_item):
            if it is not None:
                try: self.scene.removeItem(it)
                except Exception: pass
        self.target_item = None
        self._ellipse_item = None
        self._path_item = None
        self.target_rect_scene = QRectF()
        self._path = QPainterPath()

    def eventFilter(self, source, event):
        if source is self.view.viewport():
            et = event.type()

            if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drawing = True
                self._origin_scene = self.view.mapToScene(event.pos())
                mode = self.mode_combo.currentText()

                self._clear_target_items()

                if mode == "Freehand":
                    self._path = QPainterPath(self._origin_scene)
                    self._path_item = self.scene.addPath(self._path, self._pen_live)

                return True

            elif et == QEvent.Type.MouseMove and self._drawing:
                cur = self.view.mapToScene(event.pos())
                mode = self.mode_combo.currentText()

                if mode == "Box":
                    self.target_rect_scene = QRectF(self._origin_scene, cur).normalized()
                    if self.target_item is None:
                        self.target_item = self.scene.addRect(self.target_rect_scene, self._pen_live)
                    else:
                        self.target_item.setRect(self.target_rect_scene)

                elif mode == "Ellipse":
                    self.target_rect_scene = QRectF(self._origin_scene, cur).normalized()
                    if self._ellipse_item is None:
                        self._ellipse_item = self.scene.addEllipse(self.target_rect_scene, self._pen_live)
                    else:
                        self._ellipse_item.setRect(self.target_rect_scene)

                else:  # Freehand
                    self._path.lineTo(cur)
                    if self._path_item is not None:
                        self._path_item.setPath(self._path)

                return True

            elif et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton and self._drawing:
                self._drawing = False
                cur = self.view.mapToScene(event.pos())
                mode = self.mode_combo.currentText()

                if mode in ("Box", "Ellipse"):
                    self.target_rect_scene = QRectF(self._origin_scene, cur).normalized()
                    if self.target_rect_scene.width() < 10 or self.target_rect_scene.height() < 10:
                        QMessageBox.warning(self, "Selection Too Small", "Please draw a larger target selection.")
                        self._clear_target_items()
                        return True

                    # finalize pen
                    if mode == "Box" and self.target_item is not None:
                        self.target_item.setPen(self._pen_final)
                    if mode == "Ellipse" and self._ellipse_item is not None:
                        self._ellipse_item.setPen(self._pen_final)

                else:  # Freehand
                    # close path and finalize
                    if self._path.elementCount() < 10:
                        QMessageBox.warning(self, "Selection Too Small", "Freehand selection too small.")
                        self._clear_target_items()
                        return True
                    self._path.closeSubpath()
                    if self._path_item is not None:
                        self._path_item.setPen(self._pen_final)
                        self._path_item.setPath(self._path)

                return True

            elif et == QEvent.Type.Wheel:
                self._begin_interactive_present()
                angle = event.angleDelta().y()
                if angle == 0:
                    return True
                self._zoom(1.25 if angle > 0 else 0.8)
                return True

        return super().eventFilter(source, event)


    def _scene_rect_to_qrect(self, r: QRectF) -> QRect:
        if r is None or r.isNull():
            return QRect()
        bounds = self._pixmap_item.boundingRect() if self._pixmap_item else QRectF()
        W = int(bounds.width()); H = int(bounds.height())
        x = int(max(0.0, min(bounds.width(),  r.left())))
        y = int(max(0.0, min(bounds.height(), r.top())))
        w = int(max(1.0, min(bounds.width()  - x, r.width())))
        h = int(max(1.0, min(bounds.height() - y, r.height())))
        return QRect(x, y, w, h)

    def _on_use_target(self):
        mask = self._target_mask()
        if mask is None or int(mask.sum()) < 25:
            QMessageBox.warning(self, "No Target", "Draw a target selection first.")
            return

        # background is whatever is drawn (gold); if missing, recompute
        if self.bg_item is None:
            self._on_find_background()

        bgq = QRect()
        if self.bg_item is not None:
            bgq = self._scene_rect_to_qrect(self.bg_item.rect())

        # compute a bbox for info text (optional, but useful)
        try:
            ys, xs = np.nonzero(mask)
            if xs.size > 0 and ys.size > 0:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                bbox = QRect(x0, y0, (x1 - x0 + 1), (y1 - y0 + 1))
            else:
                bbox = QRect()
        except Exception:
            bbox = QRect()

        # push to parent MagnitudeToolDialog if it has setters
        parent = self.parent()
        if parent is not None:
            if hasattr(parent, "set_object_mask"):
                parent.set_object_mask(mask)

            # (optional but recommended) also pass bg as mask if you add the setter
            if hasattr(parent, "set_background_rect"):
                parent.set_background_rect(bgq)

            # convenience: update label if present
            if hasattr(parent, "lbl_info"):
                parent.lbl_info.setText(
                    f"Target set: {int(mask.sum())} px"
                    + (f"  (bbox x={bbox.x()}, y={bbox.y()}, w={bbox.width()}, h={bbox.height()})" if not bbox.isNull() else "")
                    + "\n"
                    f"Background(auto): x={bgq.x()}, y={bgq.y()}, w={bgq.width()}, h={bgq.height()}"
                )

        self.close()


    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return
        self.doc = doc
        self._load_image()

    def _cleanup_connections(self):
        try:
            if self._connected_current_doc_changed and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._connected_current_doc_changed = False

def _combine_err(stat: Optional[float], sys: Optional[float]) -> Optional[float]:
    if stat is None and sys is None:
        return None
    a = 0.0 if stat is None else float(stat)
    b = 0.0 if sys is None else float(sys)
    return float(math.sqrt(a*a + b*b))


class MagnitudeToolDialog(QDialog):
    """
    Initial photometry/magnitude tool.

    UX intent:
      - Step 1: Fetch stars (reuses SFCC’s fetch_stars logic pattern: SIMBAD B/V/R + pixel coords)
      - Step 2: Compute ZP from aperture photometry
      - Step 3: Report magnitude / mag/arcsec^2 for a user-defined object/background rectangle

    NOTE:
      This v0 assumes you already have a way to define two rectangles in your main UI.
      Wire `set_object_rect()` and `set_background_rect()` from your ROI selection tool.
    """
    def __init__(self, doc_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Magnitude / Surface Brightness")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumSize(520, 320)
        self.sys_floor_mag = 0.10  # mag (typical 0.05–0.15)
        self.object_mask = None
        self.background_mask = None
        self.doc_manager = doc_manager

        self.star_list: List[dict] = []
        self.wcs = None
        self.pixscale = None

        self.object_rect = QRect()
        self.background_rect = QRect()

        self.last_zp: Dict[str, Any] = {}

        self._build_ui()
        self._update_band_controls()

    def _build_ui(self):
        v = QVBoxLayout(self)

        row = QHBoxLayout()
        self.btn_fetch = QPushButton("Step 1: Fetch Catalog Stars (needs WCS)")
        self.btn_fetch.clicked.connect(self.fetch_stars_from_active_doc)
        row.addWidget(self.btn_fetch)

        self.btn_zp = QPushButton("Step 2: Compute Zero Points")
        self.btn_zp.clicked.connect(self.compute_zero_points)
        row.addWidget(self.btn_zp)
        v.addLayout(row)

        self.btn_pick = QPushButton("Step 3: Pick Target Region…")
        self.btn_pick.clicked.connect(self.open_region_picker)
        v.addWidget(self.btn_pick)
        self.btn_zp_plot = QPushButton("Show ZP Graphs…")
        self.btn_zp_plot.clicked.connect(self.show_zp_graphs)
        self.btn_zp_plot.setEnabled(False)
        v.addWidget(self.btn_zp_plot)
        box = QGroupBox("Photometry settings")
        form = QFormLayout(box)

        # --- Band mapping (mono only) ---
        self.band_combo = QComboBox()
        self.band_combo.addItems(["L", "R", "G", "B"])
        self.band_combo.setCurrentText("L")
        form.addRow("Mono/L band", self.band_combo)

        self.band_hint = QLabel("Mapping: L→V, R→R, G→V, B→B (SIMBAD provides B/V/R).")
        self.band_hint.setWordWrap(True)
        form.addRow("", self.band_hint)

        self.sep_sigma = QSpinBox()
        self.sep_sigma.setRange(2, 50)
        self.sep_sigma.setValue(5)
        form.addRow("SEP detect σ", self.sep_sigma)

        self.ap_r = QDoubleSpinBox()
        self.ap_r.setRange(1.0, 25.0)
        self.ap_r.setSingleStep(0.5)
        self.ap_r.setValue(6.0)
        form.addRow("Aperture radius (px)", self.ap_r)

        self.ann_in = QDoubleSpinBox()
        self.ann_in.setRange(2.0, 60.0)
        self.ann_in.setSingleStep(0.5)
        self.ann_in.setValue(12.0)
        form.addRow("Annulus r_in (px)", self.ann_in)

        self.ann_out = QDoubleSpinBox()
        self.ann_out.setRange(3.0, 80.0)
        self.ann_out.setSingleStep(0.5)
        self.ann_out.setValue(18.0)
        form.addRow("Annulus r_out (px)", self.ann_out)

        self.clip_sigma = QDoubleSpinBox()
        self.clip_sigma.setRange(1.0, 10.0)
        self.clip_sigma.setSingleStep(0.5)
        self.clip_sigma.setValue(2.5)
        form.addRow("ZP sigma-clip", self.clip_sigma)

        # --- Systematic uncertainty floor (rolled into total; popup reports totals only) ---
        self.sys_floor_spin = QDoubleSpinBox()
        self.sys_floor_spin.setRange(0.0, 1.0)
        self.sys_floor_spin.setDecimals(3)
        self.sys_floor_spin.setSingleStep(0.01)
        self.sys_floor_spin.setValue(float(getattr(self, "sys_floor_mag", 0.10) or 0.0))
        form.addRow("Systematic floor (mag)", self.sys_floor_spin)

        self.bg_box_size = QSpinBox()
        self.bg_box_size.setRange(10, 300)
        self.bg_box_size.setSingleStep(5)
        self.bg_box_size.setValue(50)
        form.addRow("Auto background box (px)", self.bg_box_size)

        hint = QLabel(
            "Popup reports total 3σ only: sqrt(stat² + sys_floor²). "
            "sys_floor is a conservative calibration mismatch term."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)

        v.addWidget(box)

        self.lbl_info = QLabel("No stars fetched yet.")
        self.lbl_info.setWordWrap(True)
        v.addWidget(self.lbl_info)

        row2 = QHBoxLayout()
        self.btn_measure = QPushButton("Step 4: Measure Object Region")
        self.btn_measure.clicked.connect(self.measure_object_region)
        row2.addWidget(self.btn_measure)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        row2.addWidget(self.btn_close)
        v.addLayout(row2)

    # --- external wiring (from your ROI tool) ---
    def show_zp_graphs(self):
        p = self.last_zp.get("plot") if isinstance(self.last_zp, dict) else None
        if not p:
            QMessageBox.information(self, "ZP Graphs", "Compute zero points first.")
            return
        dlg = ZeroPointPlotsDialog(self, p)
        dlg.exec()

    def set_object_mask(self, mask: np.ndarray):
        self.object_mask = mask

    def set_background_mask(self, mask: np.ndarray):
        self.background_mask = mask

    def _active_is_rgb(self) -> bool:
        img, _hdr, _doc = self._get_active_image_and_header()
        if img is None:
            return False
        a = np.asarray(img)
        return (a.ndim == 3 and a.shape[2] >= 3)

    def _update_band_controls(self):
        is_rgb = self._active_is_rgb()

        # band selector only matters for mono
        self.band_combo.setEnabled(not is_rgb)

        if is_rgb:
            self.band_hint.setText(
                "RGB image detected: fixed mapping is used: "
                "R→Rmag, G→Vmag, B→Bmag. (Mono/L band selector disabled.)"
            )
        else:
            self.band_hint.setText("Mapping: L→V, R→R, G→V, B→B (SIMBAD provides B/V/R).")


    def open_region_picker(self):
        dlg = MagnitudeRegionDialog(parent=self, doc_manager=self.doc_manager)
        dlg.show()

        # optional: keep bg box updated when size changes
        try:
            self.bg_box_size.valueChanged.connect(lambda _v: dlg._on_find_background())
        except Exception:
            pass


    def set_object_rect(self, rect: QRect):
        self.object_rect = QRect(rect)

    def set_background_rect(self, rect: QRect):
        self.background_rect = QRect(rect)

    # --- internals ---
    def _get_active_image_and_header(self):
        doc = self.doc_manager.get_active_document()
        if doc is None:
            return None, None, None
        img = getattr(doc, "image", None)
        meta = getattr(doc, "metadata", {}) or {}
        hdr = meta.get("wcs_header") or meta.get("original_header") or meta.get("header")
        return img, hdr, doc

    def fetch_stars_from_active_doc(self):
        self._update_band_controls()
        img, hdr, doc = self._get_active_image_and_header()
        if img is None or hdr is None:
            QMessageBox.warning(self, "No Data", "Active document must have an image and WCS header.")
            return

        meta = getattr(doc, "metadata", {}) or {}
        cached = meta.get("SFCC_star_list")
        cached_catalog = str(meta.get("SFCC_catalog") or "").strip()

        # 0) If we already have APASS cached, use it (fast path, no network)
        if isinstance(cached, list) and len(cached) > 0 and cached_catalog.upper().startswith("APASS"):
            self.star_list = cached
        else:
            # 1) APASS first
            apass_ok = False
            try:
                self.lbl_info.setText("Querying APASS (VizieR)…")
                QApplication.processEvents()

                apass = self._fetch_apass_stars_and_cache(img, hdr, doc)
                if isinstance(apass, list) and len(apass) > 0:
                    self.star_list = apass
                    apass_ok = True
            except Exception:
                apass_ok = False

            # 2) Fall back to cached stars (any catalog) if APASS failed/empty
            if not apass_ok:
                if isinstance(cached, list) and len(cached) > 0:
                    self.star_list = cached
                else:
                    # 3) Finally SIMBAD
                    try:
                        self.lbl_info.setText("Querying SIMBAD (subprocess)…")
                        QApplication.processEvents()
                        self.star_list = self._fetch_simbad_stars_and_cache(img, hdr, doc)
                    except Exception:
                        QMessageBox.information(
                            self, "Catalog Unavailable",
                            "APASS query failed (or returned no stars), no cached stars available, "
                            "and SIMBAD is currently unavailable.\n\n"
                            "Please try again later."
                        )
                        return

        # WCS / pixscale
        self.wcs, self.pixscale = _build_wcs_and_pixscale(hdr)

        n = len(self.star_list or [])
        ps = f"{self.pixscale:.3f} arcsec/px" if self.pixscale else "N/A"
        cat = str((getattr(doc, "metadata", {}) or {}).get("SFCC_catalog") or cached_catalog or "Unknown")

        self.lbl_info.setText(f"Loaded {n} catalog stars ({cat}). Pixscale: {ps}")



    def compute_zero_points(self):
        self._update_band_controls()
        img, hdr, doc = self._get_active_image_and_header()
        if img is None or hdr is None:
            QMessageBox.warning(self, "No Data", "Active document must have an image and WCS header.")
            return
        if not self.star_list:
            QMessageBox.warning(self, "No Stars", "Fetch stars first.")
            return

        img_f = _to_float_image(img)
        if img_f.ndim == 2:
            gray = img_f.astype(np.float32)
        else:
            gray = np.mean(img_f, axis=2).astype(np.float32)
        self.lbl_info.setText("Detecting sources with SEP…")
        QApplication.processEvents()
        sources = _detect_sources(gray, sigma=float(self.sep_sigma.value()))
        if sources.size == 0:
            QMessageBox.warning(self, "SEP", "SEP found no sources.")
            return

        matches = _match_starlist_to_sources(self.star_list, sources, max_dist_px=3.0)
        if not matches:
            QMessageBox.warning(self, "No Matches", "No SIMBAD stars matched SEP detections.")
            return

        self.lbl_info.setText(f"Matched {len(matches)} stars. Measuring apertures…")
        QApplication.processEvents()

        band = self.band_combo.currentText().strip().upper()

        if img_f.ndim == 2:
            zp = _compute_zero_points_mono(
                matches=matches,
                img_f=img_f,
                r_ap=float(self.ap_r.value()),
                r_in=float(self.ann_in.value()),
                r_out=float(self.ann_out.value()),
                band=band,
                clip_sigma=float(self.clip_sigma.value()),
            )
            self.last_zp = {"mode": "mono", **zp}
            self.btn_zp_plot.setEnabled(bool(self.last_zp.get("plot")))

            self.lbl_info.setText(
                "Zero point (mono):\n"
                f"  Band={zp.get('band')}  (catalog {zp.get('magkey')})\n"
                f"  ZP={zp.get('ZP')} (n={zp.get('n')}, σ={zp.get('std')})"
            )
        else:
            zp = _compute_zero_points(
                matches=matches,
                img_f=img_f,
                r_ap=float(self.ap_r.value()),
                r_in=float(self.ann_in.value()),
                r_out=float(self.ann_out.value()),
                clip_sigma=float(self.clip_sigma.value()),
            )
            self.last_zp = {"mode": "rgb", **zp}
            self.btn_zp_plot.setEnabled(bool(self.last_zp.get("plot")))

            self.lbl_info.setText(
                "Zero points (median ± SEM):\n"
                f"  ZP_R={zp['ZP_R']} (n={zp['n_R']}, scatter={zp['std_R']}, sem={zp['sem_R']})\n"
                f"  ZP_G={zp['ZP_G']} (n={zp['n_G']}, scatter={zp['std_G']}, sem={zp['sem_G']})\n"
                f"  ZP_B={zp['ZP_B']} (n={zp['n_B']}, scatter={zp['std_B']}, sem={zp['sem_B']})"
            )

    def measure_object_region(self):
        img, hdr, doc = self._get_active_image_and_header()
        if img is None:
            QMessageBox.warning(self, "No Data", "No active image.")
            return
        if not self.last_zp:
            QMessageBox.warning(self, "No ZP", "Compute zero points first.")
            return

        img_f = _to_float_image(img)
        H, W = img_f.shape[:2]

        def rect_to_mask(r: QRect) -> Optional[np.ndarray]:
            if r is None or r.isNull():
                return None
            x0 = max(0, int(r.left()))
            y0 = max(0, int(r.top()))
            x1 = min(W, int(r.right()) + 1)
            y1 = min(H, int(r.bottom()) + 1)
            if x1 <= x0 or y1 <= y0:
                return None
            m = np.zeros((H, W), dtype=bool)
            m[y0:y1, x0:x1] = True
            return m

        def sigma_from_mask(img_f: np.ndarray, m: np.ndarray):
            # robust sigma via MAD on masked pixels
            m = np.asarray(m, dtype=bool)
            if m.shape != (H, W):
                return None
            if np.count_nonzero(m) < 25:
                return None

            if img_f.ndim == 2:
                v = img_f[m].astype(np.float64, copy=False)
                med = float(np.median(v))
                mad = float(np.median(np.abs(v - med)))
                sig = 1.4826 * mad
                return sig
            else:
                v = img_f[..., :3][m].reshape(-1, 3).astype(np.float64, copy=False)
                med = np.median(v, axis=0)
                mad = np.median(np.abs(v - med), axis=0)
                sig = 1.4826 * mad
                return sig.astype(float)

        # ---------------- choose object mask ----------------
        obj_mask = self.object_mask
        if obj_mask is None:
            # fallback to rect mode if someone wired it
            obj_mask = rect_to_mask(self.object_rect)

        if obj_mask is None or np.count_nonzero(obj_mask) < 25:
            QMessageBox.information(self, "No Regions", "Pick a target region first.")
            return
        if obj_mask.shape != (H, W):
            QMessageBox.warning(self, "Mask Mismatch", "Object mask size does not match the active image.")
            return

        # ---------------- choose background mask ----------------
        bg_mask = self.background_mask

        if bg_mask is None:
            # if a rect exists, use it
            bg_mask = rect_to_mask(self.background_rect)

        if bg_mask is None:
            # auto-pick a background rect, then convert to mask
            try:
                img_f0 = img_f
                if img_f0.ndim == 2:
                    img_f0 = np.dstack([img_f0] * 3)
                box = int(self.bg_box_size.value()) if hasattr(self, "bg_box_size") else 50

                bx, by, bw, bh = auto_rect_box(img_f0, box=box, margin=100)
                self.background_rect = QRect(int(bx), int(by), int(bw), int(bh))
                bg_mask = rect_to_mask(self.background_rect)
            except Exception:
                bg_mask = None

        if bg_mask is None or np.count_nonzero(bg_mask) < 25:
            QMessageBox.warning(self, "Background", "Background region is missing or too small.")
            return
        if bg_mask.shape != (H, W):
            QMessageBox.warning(self, "Mask Mismatch", "Background mask size does not match the active image.")
            return

        # ---------------- compute sums/areas via masks ----------------
        obj_sum = _mask_sum(img_f, obj_mask)
        bkg_sum = _mask_sum(img_f, bg_mask)

        obj_area = _mask_area(obj_mask)
        bkg_area = _mask_area(bg_mask)
        if obj_area <= 0 or bkg_area <= 0:
            QMessageBox.warning(self, "Bad Regions", "Object or background region has zero area.")
            return

        # background scaled to object area
        scale = float(obj_area) / max(1.0, float(bkg_area))
        net = obj_sum - (bkg_sum * scale)

        # ---------------- flux uncertainty from background sigma ----------------
        sigma_bg = sigma_from_mask(img_f, bg_mask)
        if sigma_bg is None:
            QMessageBox.warning(self, "Background", "Background stats failed.")
            return

        if img_f.ndim == 2:
            flux_err = float(sigma_bg) * math.sqrt(float(obj_area))
        else:
            sigma_bg = np.asarray(sigma_bg, dtype=float)          # (3,)
            flux_err = sigma_bg * math.sqrt(float(obj_area))      # (3,)

        mode = (self.last_zp.get("mode") or ("mono" if img_f.ndim == 2 else "rgb"))

        # pixscale / area for surface brightness
        _, pixscale = _build_wcs_and_pixscale(hdr)
        area_asec2 = (float(obj_area) * float(pixscale) * float(pixscale)) if (pixscale and pixscale > 0) else None
        pix_area_asec2 = (float(pixscale) * float(pixscale)) if (pixscale and pixscale > 0) else None

        # background mean flux per pixel (mono: float, rgb: (3,))
        bkg_mean = (bkg_sum / float(bkg_area)) if bkg_area > 0 else None
        # systematic floor (mag) rolled into TOTAL only
        try:
            sys_floor = float(self.sys_floor_spin.value()) if hasattr(self, "sys_floor_spin") else float(getattr(self, "sys_floor_mag", 0.0) or 0.0)
        except Exception:
            sys_floor = float(getattr(self, "sys_floor_mag", 0.0) or 0.0)

        def fmt(x):
            return "N/A" if x is None else f"{x:.3f}"

        def fmt_sci(x):
            try:
                return "N/A" if x is None else f"{float(x):.6g}"
            except Exception:
                return "N/A"

        def total_sigma(stat_mag_err: Optional[float]) -> Optional[float]:
            return _combine_err(stat_mag_err, sys_floor)

        def total_3sigma(stat_mag_err: Optional[float]) -> Optional[float]:
            t = total_sigma(stat_mag_err)
            return None if t is None else (3.0 * float(t))

        # -------- MONO --------
        if img_f.ndim == 2 or mode == "mono":
            ZP = self.last_zp.get("ZP")
            band = self.last_zp.get("band", self.band_combo.currentText().strip().upper())
            magkey = self.last_zp.get("magkey", _magkey_for_band(band))

            # ZP uncertainty: prefer SEM; fallback to std/sqrt(n)
            zp_sem = self.last_zp.get("sem")
            if zp_sem is None:
                sd = self.last_zp.get("std")
                n = int(self.last_zp.get("n") or 0)
                zp_sem = (float(sd) / math.sqrt(n)) if (sd is not None and n > 1) else None

            net_f = float(net)
            m = _mag_from_flux(net_f, ZP)
            mu = _mu_from_flux(net_f, float(area_asec2), ZP) if area_asec2 is not None else None
            mu_bg = None
            if pix_area_asec2 is not None and bkg_mean is not None:
                mu_bg = _mu_from_flux(float(bkg_mean), float(pix_area_asec2), ZP)
            # --- Bortle estimate from background μ (mono uses its own channel; still okay) ---
            bortle_info = None
            if not _is_narrowband_like(hdr):
                bortle_info = _bortle_from_sqm(mu_bg)                
            m_stat = _mag_err_from_flux(net_f, float(flux_err), float(zp_sem)) if (zp_sem is not None) else None
            mu_stat = _mu_err_from_flux(net_f, float(flux_err), float(zp_sem)) if (zp_sem is not None and area_asec2 is not None) else None

            m_3 = total_3sigma(m_stat)
            mu_3 = total_3sigma(mu_stat)

            msg = (
                "Object region photometry (background-subtracted):\n"
                f"  Net flux: {fmt_sci(net_f)}   (flux σ: {fmt_sci(flux_err)})\n"
                f"  Object area: {obj_area} px   Background area: {bkg_area} px\n"
                f"  Systematic floor (included): ±{sys_floor:.3f} mag\n\n"
                f"Integrated magnitude ({band}, catalog {magkey}):\n"
                f"  m = {fmt(m)} ± {fmt(m_3)}  (total 3σ)\n\n"
            )
            if pix_area_asec2 is not None:
                msg += f"  Background μ (mag/arcsec²): {fmt(mu_bg)}\n"
            if pix_area_asec2 is not None:
                msg += f"  Background μ (mag/arcsec²): {fmt(mu_bg)}\n"
                if _is_narrowband_like(hdr):
                    msg += "  Estimated Bortle: N/A (narrowband-like filter detected)\n"
                else:
                    if bortle_info is not None:
                        msg += f"  Estimated Bortle: {bortle_info['label']} (μ≈{bortle_info['mu']:.2f})\n"
                    else:
                        msg += "  Estimated Bortle: N/A\n"
            msg += "\n"
            if area_asec2 is not None:
                msg += (
                    f"Surface brightness (mag/arcsec²)  [area={area_asec2:.3f} arcsec²]:\n"
                    f"  μ = {fmt(mu)} ± {fmt(mu_3)}  (total 3σ)\n"
                )
            else:
                msg += "Surface brightness: N/A (no pixscale from WCS)\n"

            QMessageBox.information(self, "Magnitude Results", msg)
            return

        # -------- RGB --------
        ZP_R = self.last_zp.get("ZP_R")
        ZP_G = self.last_zp.get("ZP_G")
        ZP_B = self.last_zp.get("ZP_B")

        sem_R = self.last_zp.get("sem_R")
        sem_G = self.last_zp.get("sem_G")
        sem_B = self.last_zp.get("sem_B")

        netR, netG, netB = float(net[0]), float(net[1]), float(net[2])
        errR, errG, errB = float(flux_err[0]), float(flux_err[1]), float(flux_err[2])

        mR = _mag_from_flux(netR, ZP_R)
        mG = _mag_from_flux(netG, ZP_G)
        mB = _mag_from_flux(netB, ZP_B)

        mR_stat = _mag_err_from_flux(netR, errR, float(sem_R)) if (sem_R is not None) else None
        mG_stat = _mag_err_from_flux(netG, errG, float(sem_G)) if (sem_G is not None) else None
        mB_stat = _mag_err_from_flux(netB, errB, float(sem_B)) if (sem_B is not None) else None

        mR_3 = total_3sigma(mR_stat)
        mG_3 = total_3sigma(mG_stat)
        mB_3 = total_3sigma(mB_stat)

        mu_bg_R = mu_bg_G = mu_bg_B = None
        if pix_area_asec2 is not None and bkg_mean is not None:
            # bkg_mean is (3,)
            bR, bG, bB = float(bkg_mean[0]), float(bkg_mean[1]), float(bkg_mean[2])
            mu_bg_R = _mu_from_flux(bR, float(pix_area_asec2), ZP_R) if (ZP_R is not None and bR > 0) else None
            mu_bg_G = _mu_from_flux(bG, float(pix_area_asec2), ZP_G) if (ZP_G is not None and bG > 0) else None
            mu_bg_B = _mu_from_flux(bB, float(pix_area_asec2), ZP_B) if (ZP_B is not None and bB > 0) else None


        muR = muG = muB = None
        muR_3 = muG_3 = muB_3 = None

        if area_asec2 is not None:
            A = float(area_asec2)
            muR = _mu_from_flux(netR, A, ZP_R)
            muG = _mu_from_flux(netG, A, ZP_G)
            muB = _mu_from_flux(netB, A, ZP_B)

            muR_stat = _mu_err_from_flux(netR, errR, float(sem_R)) if (sem_R is not None) else None
            muG_stat = _mu_err_from_flux(netG, errG, float(sem_G)) if (sem_G is not None) else None
            muB_stat = _mu_err_from_flux(netB, errB, float(sem_B)) if (sem_B is not None) else None

            muR_3 = total_3sigma(muR_stat)
            muG_3 = total_3sigma(muG_stat)
            muB_3 = total_3sigma(muB_stat)

        # --- Bortle estimate from GREEN channel background μ ---
        bortle_info = None
        if not _is_narrowband_like(hdr):
            bortle_info = _bortle_from_sqm(mu_bg_G)

        msg = (
            "Object region photometry (background-subtracted):\n"
            f"  Net flux (R,G,B): {fmt_sci(netR)}, {fmt_sci(netG)}, {fmt_sci(netB)}\n"
            f"  Flux σ   (R,G,B): {fmt_sci(errR)}, {fmt_sci(errG)}, {fmt_sci(errB)}\n"
            f"  Object area: {obj_area} px   Background area: {bkg_area} px\n"
            f"  Systematic floor (included): ±{sys_floor:.3f} mag\n\n"
            "Integrated magnitude (total 3σ):\n"
            f"  m_R = {fmt(mR)} ± {fmt(mR_3)}\n"
            f"  m_G = {fmt(mG)} ± {fmt(mG_3)}\n"
            f"  m_B = {fmt(mB)} ± {fmt(mB_3)}\n\n"
        )

        if pix_area_asec2 is not None:
            msg += (
                f"Background surface brightness (mag/arcsec²):\n"
                f"  μ_bg_R = {fmt(mu_bg_R)}   μ_bg_G = {fmt(mu_bg_G)}   μ_bg_B = {fmt(mu_bg_B)}\n"
            )
            if _is_narrowband_like(hdr):
                msg += "  Estimated Bortle: N/A (narrowband-like filter detected)\n\n"
            else:
                if bortle_info is not None:
                    msg += f"  Estimated Bortle (from GREEN): {bortle_info['label']} (μ≈{bortle_info['mu']:.2f})\n\n"
                else:
                    msg += "  Estimated Bortle (from GREEN): N/A\n\n"


        if area_asec2 is not None:
            msg += (
                f"Surface brightness (mag/arcsec²)  [area={area_asec2:.3f} arcsec²] (total 3σ):\n"
                f"  μ_R = {fmt(muR)} ± {fmt(muR_3)}\n"
                f"  μ_G = {fmt(muG)} ± {fmt(muG_3)}\n"
                f"  μ_B = {fmt(muB)} ± {fmt(muB_3)}\n"
            )
        else:
            msg += "Surface brightness: N/A (no pixscale from WCS)\n"

        QMessageBox.information(self, "Magnitude Results", msg)

    def _fetch_apass_stars_and_cache(self, img, hdr, doc) -> List[dict]:
        wcs, _ = _build_wcs_and_pixscale(hdr)
        if wcs is None:
            raise RuntimeError("Could not build WCS for APASS query.")

        wcs2 = wcs.celestial if hasattr(wcs, "celestial") else wcs
        H, W = img.shape[:2]

        # center + radius (same logic as SIMBAD)
        pix = np.array([[W/2, H/2], [0,0], [W,0], [0,H], [W,H]], dtype=float)
        sky = wcs2.all_pix2world(pix, 0)
        center = SkyCoord(sky[0,0]*u.deg, sky[0,1]*u.deg)
        radius = center.separation(SkyCoord(sky[1:,0]*u.deg, sky[1:,1]*u.deg)).max() * 1.05

        Vizier.ROW_LIMIT = 20000

        # Ask for several likely “red” columns; VizieR will ignore unknown ones
        Vizier.columns = ["RAJ2000", "DEJ2000", "Bmag", "Vmag", "r'mag", "rmag", "Rmag"]

        self.lbl_info.setText("Querying APASS (VizieR)…")
        QApplication.processEvents()

        result = Vizier.query_region(center, radius=radius, catalog="II/336/apass9")
        if not result:
            return []

        tab = result[0]
        cols_lower = {c.strip().lower(): c for c in tab.colnames}

        def pick_col(*cands: str) -> Optional[str]:
            for c in cands:
                k = c.strip().lower()
                if k in cols_lower:
                    return cols_lower[k]
            return None

        ra_col  = pick_col("RAJ2000")
        dec_col = pick_col("DEJ2000")
        b_col   = pick_col("Bmag")
        v_col   = pick_col("Vmag")

        # “Red” in APASS is usually Sloan r′ => r'mag
        r_col   = pick_col("r'mag", "rmag", "Rmag")

        if ra_col is None or dec_col is None:
            raise RuntimeError(f"APASS table missing RA/Dec columns. colnames={tab.colnames}")

        stars = []
        for row in tab:
            ra  = _unmask_num(_row_get(row, ra_col))
            dec = _unmask_num(_row_get(row, dec_col))
            if ra is None or dec is None:
                continue

            try:
                x, y = wcs2.all_world2pix(ra, dec, 0)
            except Exception:
                continue

            if not (0 <= x < W and 0 <= y < H):
                continue

            bmag = _unmask_num(_row_get(row, b_col)) if b_col else None
            vmag = _unmask_num(_row_get(row, v_col)) if v_col else None
            rmag = _unmask_num(_row_get(row, r_col)) if r_col else None  # this is r′ most of the time

            stars.append({
                "ra": float(ra),
                "dec": float(dec),
                "x": float(x),
                "y": float(y),
                "Bmag": bmag,
                "Vmag": vmag,
                # Store APASS r′ as “Rmag” for your tool’s red channel mapping:
                "Rmag": rmag,
                "sp_clean": None,
                "pickles_match": None,
            })

        meta = dict(getattr(doc, "metadata", {}) or {})
        meta["SFCC_star_list"] = stars
        meta["SFCC_catalog"] = "APASS_DR9"
        self.doc_manager.update_active_document(
            doc.image, metadata=meta,
            step_name="Magnitude Stars Cached (APASS)", doc=doc
        )

        # optional: quick debug to confirm which column you used
        # print("APASS colnames:", tab.colnames, "picked red:", r_col)

        return stars


    def _fetch_simbad_stars_and_cache(self, img, hdr, doc) -> List[dict]:
        """
        Fetch SIMBAD stars using a subprocess worker (prevents GUI hangs when TLS/SSL wedges).
        Produces list of dicts with keys:
        ra, dec, sp_clean, pickles_match, x, y, Bmag, Vmag, Rmag
        Caches into doc.metadata['SFCC_star_list'].
        """
        # ---- build WCS ----
        wcs, _pixscale = _build_wcs_and_pixscale(hdr)
        if wcs is None:
            raise RuntimeError("Could not build 2D WCS from header.")
        wcs2 = wcs.celestial if hasattr(wcs, "celestial") else wcs

        H, W = img.shape[:2]

        # ---- center + corner radius ----
        pix = np.array([[W / 2, H / 2], [0, 0], [W, 0], [0, H], [W, H]], dtype=float)
        try:
            sky = wcs2.all_pix2world(pix, 0)
        except Exception as e:
            raise RuntimeError(f"WCS Conversion Error: {e}")

        center_ra = float(sky[0, 0])
        center_dec = float(sky[0, 1])

        center_sky = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame="icrs")
        corners_sky = SkyCoord(ra=sky[1:, 0] * u.deg, dec=sky[1:, 1] * u.deg, frame="icrs")
        radius = center_sky.separation(corners_sky).max() * 1.05
        radius_deg = float(radius.to_value(u.deg))

        # ---- mirror list (use yours or keep static) ----
        servers = ["simbad.u-strasbg.fr", "simbad.harvard.edu"]
        HARD_TIMEOUT_S = 20.0
        ROW_LIMIT = 10000

        # Total wall time allowed for the whole subprocess run.
        # Should be > HARD_TIMEOUT_S, because worker may try multiple mirrors.
        SUBPROC_TIMEOUT_S = 60.0

        # status text (safe; no astroquery here)
        try:
            self.lbl_info.setText("Querying SIMBAD (subprocess)…")
            QApplication.processEvents()
        except Exception:
            pass

        # ---- run astroquery in subprocess ----
        payload = _run_in_subprocess(
            SUBPROC_TIMEOUT_S,
            _simbad_query_worker,
            center_ra, center_dec, radius_deg,
            servers,
            HARD_TIMEOUT_S,
            ROW_LIMIT,
        )

        # payload: {"colnames":[...], "rows":[{col: value, ...}, ...]}
        colnames = list(payload.get("colnames", []) or [])
        rows = list(payload.get("rows", []) or [])

        if not rows:
            # cache empty list to avoid re-query spam
            meta = dict(getattr(doc, "metadata", {}) or {})
            meta["SFCC_star_list"] = []
            meta["SFCC_catalog"] = "SIMBAD"
            self.doc_manager.update_active_document(doc.image, metadata=meta, step_name="Magnitude Stars Cached (SIMBAD)", doc=doc)
            return []

        # ---- helpers for parsing returned dict rows ----
        cols_lower = {str(c).strip().lower(): str(c) for c in colnames}

        def _pick_col(*cands: str) -> Optional[str]:
            for c in cands:
                key = str(c).strip().lower()
                if key in cols_lower:
                    return cols_lower[key]
            return None

        # SIMBAD columns vary depending on field names used/returned
        ra_col  = _pick_col("ra", "ra(d)", "ra_d", "ra_deg")
        dec_col = _pick_col("dec", "dec(d)", "dec_d", "dec_deg")

        # mags
        b_col = _pick_col("b", "flux_b", "flux(b)", "bmag")
        v_col = _pick_col("v", "flux_v", "flux(v)", "vmag")
        r_col = _pick_col("r", "flux_r", "flux(r)", "rmag")

        # spectral type
        sp_col = _pick_col("sp", "sp_type", "sptype", "spectral_type")

        if ra_col is None or dec_col is None:
            raise RuntimeError(f"SIMBAD result missing ra/dec columns. colnames={colnames}")

        def infer_letter(bv: Optional[float]) -> Optional[str]:
            if bv is None or (isinstance(bv, float) and np.isnan(bv)):
                return None
            if bv < 0.00:   return "B"
            if bv < 0.30:   return "A"
            if bv < 0.58:   return "F"
            if bv < 0.81:   return "G"
            if bv < 1.40:   return "K"
            if bv > 1.40:   return "M"
            return None

        def safe_world2pix(ra_deg: float, dec_deg: float):
            try:
                xpix, ypix = wcs2.all_world2pix(ra_deg, dec_deg, 0)
                xpix, ypix = float(xpix), float(ypix)
                if np.isfinite(xpix) and np.isfinite(ypix):
                    return xpix, ypix
                return None
            except NoConvergence as e:
                try:
                    xpix, ypix = e.best_solution
                    xpix, ypix = float(xpix), float(ypix)
                    if np.isfinite(xpix) and np.isfinite(ypix):
                        return xpix, ypix
                except Exception:
                    pass
                return None
            except Exception:
                return None

        # pickles templates (optional)
        pickles_templates = getattr(self, "pickles_templates", None)
        if pickles_templates is None:
            pickles_templates = []
            setattr(self, "pickles_templates", pickles_templates)

        star_list: List[dict] = []

        for row in rows:
            # row is plain dict {colname: value}
            ra_deg  = _unmask_num(row.get(ra_col))
            dec_deg = _unmask_num(row.get(dec_col))
            if ra_deg is None or dec_deg is None:
                continue

            bmag = _unmask_num(row.get(b_col)) if b_col else None
            vmag = _unmask_num(row.get(v_col)) if v_col else None
            rmag = _unmask_num(row.get(r_col)) if r_col else None

            raw_sp = row.get(sp_col) if sp_col else None

            sp_clean = None
            if raw_sp is not None and str(raw_sp).strip():
                sp = str(raw_sp).strip().upper()
                # keep your original filtering
                if not (sp.startswith("SN") or sp.startswith("KA")):
                    sp_clean = sp
            elif (bmag is not None) and (vmag is not None):
                sp_clean = infer_letter(bmag - vmag)

            if not sp_clean:
                continue

            xy = safe_world2pix(float(ra_deg), float(dec_deg))
            if xy is None:
                continue
            xpix, ypix = xy

            if not (0 <= xpix < W and 0 <= ypix < H):
                continue

            best_template = None
            try:
                matches = pickles_match_for_simbad(sp_clean, pickles_templates) if pickles_templates is not None else []
                best_template = matches[0] if matches else None
            except Exception:
                best_template = None

            star_list.append({
                "ra": float(ra_deg),
                "dec": float(dec_deg),
                "sp_clean": sp_clean,
                "pickles_match": best_template,
                "x": float(xpix),
                "y": float(ypix),
                "Bmag": float(bmag) if bmag is not None else None,
                "Vmag": float(vmag) if vmag is not None else None,
                "Rmag": float(rmag) if rmag is not None else None,
            })

        # ---- cache into metadata ----
        meta = dict(getattr(doc, "metadata", {}) or {})
        meta["SFCC_star_list"] = list(star_list)  # JSON-ish
        meta["SFCC_catalog"] = "SIMBAD"
        self.doc_manager.update_active_document(doc.image, metadata=meta, step_name="Magnitude Stars Cached (SIMBAD)", doc=doc)

        return star_list

def open_magnitude_tool(doc_manager, parent=None) -> MagnitudeToolDialog:
    dlg = MagnitudeToolDialog(doc_manager=doc_manager, parent=parent)
    dlg.show()
    return dlg
