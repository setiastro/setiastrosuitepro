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

from PyQt6.QtCore import Qt, QRect, QEvent, QPointF, QRectF, QTimer, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTabWidget, QWidget,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout,QComboBox, QGraphicsPathItem, QGraphicsEllipseItem,
    QMessageBox, QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem, QListWidget, QListWidgetItem, QCheckBox, QInputDialog,
    QTableWidget, QTableWidgetItem, QLineEdit, QInputDialog,
    QFileDialog, QAbstractItemView, QHeaderView,    
)
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor, QPainter, QPainterPath, QPolygonF
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.wcs.wcs import NoConvergence
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# IMPORTANT: use the centralized one (adjust import path to where you moved it)
from setiastro.saspro.sfcc import pickles_match_for_simbad
from setiastro.saspro import magnitude_regions as mreg

# Reuse useful SFCC pieces
from setiastro.saspro.sfcc import non_blocking_sleep  # already used in SFCC; optional
from setiastro.saspro.backgroundneutral import auto_rect_box, auto_rect_50x50
from setiastro.saspro.imageops.stretch import stretch_color_image
# We *intentionally* do NOT reuse SFCC pedestal-removal/clamp for photometry.

import socket

import multiprocessing as mp
import queue
import traceback

from setiastro.saspro import photometry_aavso as aav

import webbrowser


def _run_in_subprocess(timeout_s: float, target, *args, **kwargs):
    """
    Run `target(*args, **kwargs)` in a fresh subprocess (spawn-safe).
    If it hangs, terminate it. Returns target result or raises.
    """
    ctx = mp.get_context("spawn")  # safest on Windows/macOS
    q = ctx.Queue()

    p = ctx.Process(target=_subproc_entry, args=(q, target, args, kwargs))
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

    err_msg, tb = payload
    # Full traceback goes to the console for debugging; the UI gets one line.
    try:
        import sys
        print(f"[Catalog subprocess error]\n{tb}", file=sys.stderr)
    except Exception:
        pass
    raise RuntimeError(err_msg)


def _row_get(tab_row, colname: str):
    try:
        return tab_row[colname]
    except Exception:
        return None

def _subproc_entry(q, target, args, kwargs):
    """
    Top-level worker entry for multiprocessing spawn (must be picklable).
    """
    try:
        out = target(*args, **kwargs)
        q.put(("ok", out))
    except Exception as e:
        q.put(("err", (str(e), traceback.format_exc())))


def _simbad_query_worker(center_ra_deg: float, center_dec_deg: float, radius_deg: float,
                        servers: list[str], hard_timeout_s: float, row_limit: int):
    # Import inside subprocess
    from astroquery.simbad import Simbad
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    # astroquery/astropy config validators require an INT timeout; a float
    # like 20.0 raises VdtTypeError. Coerce once, reuse everywhere.
    _timeout_i = int(round(float(hard_timeout_s)))

    try:
        from astroquery import conf as aq_conf
        aq_conf.timeout = _timeout_i
    except Exception:
        pass

    try:
        from astroquery.simbad import conf as simbad_conf
    except Exception:
        simbad_conf = None

    try:
        Simbad.TIMEOUT = _timeout_i
    except Exception:
        pass

    center = SkyCoord(center_ra_deg * u.deg, center_dec_deg * u.deg, frame="icrs")

    def _host_of(server: str) -> str:
        s = str(server).replace("https://", "").replace("http://", "")
        return s.split("/")[0]

    def _configure_fields():
        # This makes a TAP capabilities network call in modern astroquery,
        # which is the exact call that fails when the endpoint is down. Doing
        # it per-mirror means a dead endpoint on one mirror is recoverable.
        Simbad.reset_votable_fields()
        try:
            Simbad.add_votable_fields("sp", "B", "V", "R", "ra", "dec")
        except Exception:
            # legacy field names (older astroquery)
            Simbad.add_votable_fields("sp", "flux(B)", "flux(V)", "flux(R)", "ra(d)", "dec(d)")

    last_err = None
    for server in (servers or []):
        host = _host_of(server)

        # Fast reachability probe: skip a dead/DNS-broken mirror in ~3s instead
        # of letting a stalled TLS handshake ride the subprocess out to its kill.
        if not _tcp_reachable(host, 443, timeout_s=3.0):
            last_err = RuntimeError(f"{host}: not reachable (DNS/connect failed)")
            continue

        try:
            if simbad_conf is not None:
                try:
                    simbad_conf.server = server
                except Exception:
                    pass

            Simbad.ROW_LIMIT = int(row_limit)
            _configure_fields()

            tab = Simbad.query_region(center, radius=radius_deg * u.deg)
            if tab is None:
                last_err = RuntimeError(f"{host}: query returned no table")
                continue

            return {
                "colnames": list(tab.colnames),
                "rows": [{c: tab[i][c] for c in tab.colnames} for i in range(len(tab))],
            }
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(_classify_simbad_error(last_err))

def _classify_vizier_error(err) -> str:
    """Turn a raw VizieR/APASS exception into a short, user-facing reason."""
    if err is None:
        return "VizieR query failed (unknown error)."
    s = str(err)
    low = s.lower()
    if "timeout" in low or "timed out" in low:
        return ("VizieR timed out — the APASS server is slow or down "
                "(it has had recent outages). Try again later.")
    if any(t in low for t in ("connection", "refused", "reset", "broken pipe")):
        return "VizieR connection failed (server down or network issue). Try again later."
    if any(t in low for t in ("name or service", "getaddrinfo", "dns", "resolve")):
        return "VizieR connection failed (DNS or network issue). Try again later."
    if "500" in low or "502" in low or "503" in low or "504" in low:
        return "VizieR returned a server error (temporary outage). Try again later."
    if "empty" in low or "index" in low:
        return "VizieR returned no rows for this field."
    return f"VizieR query failed: {s.splitlines()[0]}"

def _classify_simbad_error(err) -> str:
    """Turn a raw SIMBAD/pyvo exception into a short, user-facing reason."""
    if err is None:
        return "SIMBAD query failed (unknown error)."
    s = f"{type(err).__name__}: {err}"
    low = s.lower()
    if "dalservice" in low or "capabilities" in low:
        return ("SIMBAD is unreachable (the service capabilities endpoint failed). "
                "The server may be down or blocked by your network — try again later.")
    if "timeout" in low or "timed out" in low:
        return "SIMBAD timed out (server slow or unreachable). Try again later."
    if any(k in low for k in ("connection", "dns", "getaddrinfo", "name or service", "failed to resolve")):
        return "SIMBAD connection failed (DNS or network issue). Try again later."
    if "not reachable" in low:
        return f"SIMBAD mirrors were not reachable ({err})."
    return f"SIMBAD query failed: {s.splitlines()[0]}"

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


def _sigma_clip(vals: np.ndarray, sigma: float = 2.5, iters: int = 3,
                return_mask: bool = False):
    v_all = np.asarray(vals, dtype=np.float64)
    finite = np.isfinite(v_all)
    v = v_all[finite]
    if v.size < 3:
        if return_mask:
            # everything finite is "kept" when we can't meaningfully clip
            return v, finite.copy()
        return v

    # Track which of the finite entries survive, in the finite-subset frame.
    keep_finite = np.ones(v.size, dtype=bool)
    idx = np.arange(v.size)      # indices into the finite subset that remain
    cur = v
    for _ in range(max(1, int(iters))):
        med = np.median(cur)
        sd = np.std(cur)
        if not np.isfinite(sd) or sd <= 0:
            break
        keep = np.abs(cur - med) <= sigma * sd
        if keep.sum() == cur.size:
            break
        # map this round's keep back onto the finite-subset mask
        keep_finite[idx[~keep]] = False
        idx = idx[keep]
        cur = cur[keep]
        if cur.size < 3:
            break

    if return_mask:
        # lift the finite-subset mask back into the full input frame
        full_mask = np.zeros(v_all.shape, dtype=bool)
        full_mask[np.where(finite)[0][keep_finite]] = True
        return cur, full_mask
    return cur

def _estimate_fwhm_from_sources(sources: np.ndarray, clip_sigma: float = 2.5) -> Optional[float]:
    """
    Estimate scene FWHM in pixels from SEP source second-moment semi-axes.
    SEP 'a' is the RMS radius along the major axis (Gaussian sigma equivalent).
    FWHM ≈ 2.35 * 2 * a  (factor of 2: SEP a is semi-axis, not full-width half).
    Returns None if sources is empty or result is implausible.
    """
    if sources is None or len(sources) == 0:
        return None
    try:
        a = np.asarray(sources["a"], dtype=np.float64)
        a = a[np.isfinite(a) & (a > 0.3) & (a < 50.0)]
        if a.size < 3:
            return None
        # sigma-clip to reject saturated/cosmic-ray outliers
        a = _sigma_clip(a, sigma=clip_sigma, iters=3)
        if a.size == 0:
            return None
        fwhm = float(2.35 * 2.0 * np.median(a))
        # sanity clamp: 1.5–30 px is a reasonable range for astrophotos
        return float(np.clip(fwhm, 1.5, 30.0))
    except Exception:
        return None

def _detect_sources(gray: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    bkg = sep.Background(gray)
    data_sub = gray - bkg.back()
    err = float(bkg.globalrms)
    sources = sep.extract(data_sub, float(sigma), err=err)
    return sources


def _match_starlist_to_sources(star_list: List[dict], sources: np.ndarray, max_dist_px: float = 3.0) -> List[dict]:
    if sources is None or len(sources) == 0:
        return []
    sx = sources["x"].astype(np.float64)
    sy = sources["y"].astype(np.float64)

    # precompute nearest catalog index for each source (mutual requirement)
    cat_xy = []
    for st in star_list:
        x0 = float(st.get("x", np.nan))
        y0 = float(st.get("y", np.nan))
        if np.isfinite(x0) and np.isfinite(y0):
            cat_xy.append((x0, y0))
        else:
            cat_xy.append((np.nan, np.nan))
    cat_xy = np.asarray(cat_xy, dtype=np.float64)

    # source -> nearest catalog
    src_to_cat = np.full(len(sx), -1, dtype=int)
    src_to_d2  = np.full(len(sx), np.inf, dtype=np.float64)

    for j in range(len(sx)):
        dx = cat_xy[:, 0] - sx[j]
        dy = cat_xy[:, 1] - sy[j]
        d2 = dx*dx + dy*dy
        i = int(np.nanargmin(d2))
        src_to_cat[j] = i
        src_to_d2[j] = float(d2[i])

    out = []
    r2max = float(max_dist_px)**2

    # catalog -> nearest source, but require mutual
    for i, st in enumerate(star_list):
        x0, y0 = cat_xy[i]
        if not (np.isfinite(x0) and np.isfinite(y0)):
            continue
        dx = sx - x0
        dy = sy - y0
        j = int(np.argmin(dx*dx + dy*dy))
        d2 = float(dx[j]*dx[j] + dy[j]*dy[j])
        if d2 <= r2max and src_to_cat[j] == i and src_to_d2[j] <= r2max:
            out.append({"star": st, "src": sources[j], "x": float(sx[j]), "y": float(sy[j])})
    return out

def _aperture_photometry_mono(img_f, xs, ys, r_ap, r_in, r_out):
    """
    Per-star aperture photometry for mono images.
    r_ap, r_in, r_out may be scalars OR arrays (one value per star).
    """
    x = np.ascontiguousarray(np.asarray(xs, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(ys, dtype=np.float64))
    plane = np.ascontiguousarray(np.asarray(img_f, dtype=np.float32))
    N = len(x)

    r_ap_arr  = np.broadcast_to(np.asarray(r_ap,  dtype=np.float64), (N,)).copy()
    r_in_arr  = np.broadcast_to(np.asarray(r_in,  dtype=np.float64), (N,)).copy()
    r_out_arr = np.broadcast_to(np.asarray(r_out, dtype=np.float64), (N,)).copy()

    flux_net = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ap_area  = math.pi * float(r_ap_arr[i]) ** 2
        ann_area = math.pi * (float(r_out_arr[i]) ** 2 - float(r_in_arr[i]) ** 2)

        f_ap, _, _ = sep.sum_circle(plane, x[i:i+1], y[i:i+1],
                                    r=float(r_ap_arr[i]), subpix=5)
        bkg, _, _  = sep.sum_circann(plane, x[i:i+1], y[i:i+1],
                                     float(r_in_arr[i]), float(r_out_arr[i]), subpix=5)

        bkg_per_pix = float(bkg[0]) / max(ann_area, 1e-8)
        flux_net[i] = float(f_ap[0]) - bkg_per_pix * ap_area

    return flux_net


def _aperture_photometry_rgb(img_f, xs, ys, r_ap, r_in, r_out):
    """
    Per-star aperture photometry for RGB images.
    r_ap, r_in, r_out may be scalars OR arrays (one value per star).
    """
    x = np.ascontiguousarray(np.asarray(xs, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(ys, dtype=np.float64))
    img_f = np.ascontiguousarray(np.asarray(img_f, dtype=np.float32))
    N = len(x)

    r_ap_arr  = np.broadcast_to(np.asarray(r_ap,  dtype=np.float64), (N,)).copy()
    r_in_arr  = np.broadcast_to(np.asarray(r_in,  dtype=np.float64), (N,)).copy()
    r_out_arr = np.broadcast_to(np.asarray(r_out, dtype=np.float64), (N,)).copy()

    planes = [
        np.ascontiguousarray(img_f[..., 0]),
        np.ascontiguousarray(img_f[..., 1]),
        np.ascontiguousarray(img_f[..., 2]),
    ]

    flux_net = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        ap_area  = math.pi * float(r_ap_arr[i]) ** 2
        ann_area = math.pi * (float(r_out_arr[i]) ** 2 - float(r_in_arr[i]) ** 2)

        for c, plane in enumerate(planes):
            f_ap, _, _ = sep.sum_circle(plane, x[i:i+1], y[i:i+1],
                                        r=float(r_ap_arr[i]), subpix=5)
            bkg, _, _  = sep.sum_circann(plane, x[i:i+1], y[i:i+1],
                                         float(r_in_arr[i]), float(r_out_arr[i]), subpix=5)

            bkg_per_pix = float(bkg[0]) / max(ann_area, 1e-8)
            flux_net[i, c] = float(f_ap[0]) - bkg_per_pix * ap_area

    return flux_net, None

def _crowding_fraction(matches: List[dict]) -> Tuple[float, int, int]:
    """
    Fraction of matched stars whose background annulus is contaminated by another
    catalog star. For each star we use the SAME per-star r_out the photometry uses
    as the annulus radius, and count a star as "crowded" if any OTHER matched star
    falls within that radius (i.e. sits inside its background annulus).

    Returns (fraction, n_crowded, n_total). In a globular core this runs high,
    warning the user that annulus background subtraction — and therefore the ZP —
    may be degraded by neighbor light. On a sparse field it is ~0.
    """
    n = len(matches)
    if n < 2:
        return 0.0, 0, n

    xs = np.array([float(m["src"]["x"]) for m in matches], dtype=np.float64)
    ys = np.array([float(m["src"]["y"]) for m in matches], dtype=np.float64)
    _r_ap, _r_in, r_out = _per_star_radii(matches)

    n_crowded = 0
    for i in range(n):
        dx = xs - xs[i]
        dy = ys - ys[i]
        d2 = dx * dx + dy * dy
        d2[i] = np.inf                      # exclude self
        # crowded if the nearest OTHER star sits inside this star's annulus radius
        if np.sqrt(np.min(d2)) <= r_out[i]:
            n_crowded += 1

    return (n_crowded / n), n_crowded, n

def _per_star_radii(
    matches: List[dict],
    r_ap_min: float = 2.0,
    r_ap_max: float = 15.0,
    ap_factor: float = 1.5,
    in_factor: float = 3.0,
    out_factor: float = 4.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-star aperture/annulus radii from SEP second-moment axes.
    fwhm_i = 2.35 * 2 * sqrt(a_i * b_i)  (geometric mean handles elliptical PSFs)
    r_ap  = ap_factor  * fwhm_i   (clamped to [r_ap_min, r_ap_max])
    r_in  = in_factor  * fwhm_i   (always > r_ap)
    r_out = out_factor * fwhm_i   (always > r_in)
    """
    N = len(matches)
    r_ap_arr  = np.empty(N, dtype=np.float64)
    r_in_arr  = np.empty(N, dtype=np.float64)
    r_out_arr = np.empty(N, dtype=np.float64)

    for i, m in enumerate(matches):
        src = m["src"]
        a = float(src["a"]) if np.isfinite(src["a"]) and src["a"] > 0 else None
        b = float(src["b"]) if np.isfinite(src["b"]) and src["b"] > 0 else None

        if a is not None and b is not None:
            fwhm = 2.35 * 2.0 * math.sqrt(a * b)
        elif a is not None:
            fwhm = 2.35 * 2.0 * a
        else:
            # fallback: use scene median from whatever the spinbox says
            fwhm = r_ap_max / ap_factor

        r_ap  = float(np.clip(ap_factor  * fwhm, r_ap_min, r_ap_max))
        r_in  = max(r_ap  + 1.0, in_factor  * fwhm)
        r_out = max(r_in  + 2.0, out_factor * fwhm)

        r_ap_arr[i]  = r_ap
        r_in_arr[i]  = r_in
        r_out_arr[i] = r_out

    return r_ap_arr, r_in_arr, r_out_arr

def _compute_zero_points_mono(matches, img_f, band="L", clip_sigma=2.5):
    xs = np.array([m["x"] for m in matches], dtype=np.float64)
    ys = np.array([m["y"] for m in matches], dtype=np.float64)

    # per-star radii (r_ap/r_in/r_out args kept as fallback scale reference only)
    r_ap_arr, r_in_arr, r_out_arr = _per_star_radii(matches)
    flux_net = _aperture_photometry_mono(img_f, xs, ys, r_ap_arr, r_in_arr, r_out_arr)

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

    zps_arr = np.asarray(zps, dtype=np.float64)
    zps, keep_mono = _sigma_clip(zps_arr, sigma=clip_sigma, return_mask=True)
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
                "x": x, "y": y, "kept": keep_mono.tolist(),
                "zp": float(np.median(zps)) if zps.size else None,
                "title": f"Mono: {magkey} vs m_inst (y = x + ZP)",
            }
        }
    }
    return out

def _compute_zero_points(matches, img_f, clip_sigma=2.5):
    xs = np.array([m["x"] for m in matches], dtype=np.float64)
    ys = np.array([m["y"] for m in matches], dtype=np.float64)

    # per-star radii
    r_ap_arr, r_in_arr, r_out_arr = _per_star_radii(matches)
    flux_net, _flux_ap = _aperture_photometry_rgb(img_f, xs, ys, r_ap_arr, r_in_arr, r_out_arr)

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

    zp_R_arr = np.asarray(zp_R, dtype=np.float64)
    zp_G_arr = np.asarray(zp_G, dtype=np.float64)
    zp_B_arr = np.asarray(zp_B, dtype=np.float64)
    zp_R, keep_R = _sigma_clip(zp_R_arr, sigma=clip_sigma, return_mask=True)
    zp_G, keep_G = _sigma_clip(zp_G_arr, sigma=clip_sigma, return_mask=True)
    zp_B, keep_B = _sigma_clip(zp_B_arr, sigma=clip_sigma, return_mask=True)

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
            "R": {"x": xR, "y": yR, "kept": keep_R.tolist(), "zp": ZP_R,
                  "title": "Red channel: Rmag vs m_inst (y = x + ZP_R)"},
            "G": {"x": xG, "y": yG, "kept": keep_G.tolist(), "zp": ZP_G,
                  "title": "Green channel: Vmag vs m_inst (y = x + ZP_G)"},
            "B": {"x": xB, "y": yB, "kept": keep_B.tolist(), "zp": ZP_B,
                  "title": "Blue channel: Bmag vs m_inst (y = x + ZP_B)"},
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

class _RegionManagerDialog(QDialog):
    def __init__(self, parent, lib):
        super().__init__(parent)
        self.setWindowTitle("Saved Regions")
        self.setModal(True)
        self._lib = lib
        self.selected_name = None

        v = QVBoxLayout(self)
        self._list = QListWidget(self)
        for n in lib.names():
            r = lib.get(n)
            tag = f"   [{r.kind}]" + (f"  {r.object_name}" if r.object_name else "")
            it = QListWidgetItem(f"{n}{tag}", self._list)
            it.setData(Qt.ItemDataRole.UserRole, n)
        self._list.itemDoubleClicked.connect(lambda *_: self._accept())
        v.addWidget(self._list)

        row = QHBoxLayout()
        b_load = QPushButton("Load"); b_del = QPushButton("Delete"); b_close = QPushButton("Close")
        b_load.clicked.connect(self._accept)
        b_del.clicked.connect(self._delete)
        b_close.clicked.connect(self.reject)
        row.addWidget(b_load); row.addWidget(b_del); row.addStretch(1); row.addWidget(b_close)
        v.addLayout(row)
        self.resize(440, 340)

    def _current_name(self):
        it = self._list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _accept(self):
        n = self._current_name()
        if n:
            self.selected_name = n
            self.accept()

    def _delete(self):
        n = self._current_name()
        if not n:
            return
        if QMessageBox.question(self, "Delete", f"Delete region '{n}'?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                                ) == QMessageBox.StandardButton.Yes:
            self._lib.remove(n)
            self._list.takeItem(self._list.currentRow())

class _ResultsDialog(QDialog):
    """
    Non-blocking results window. Stays open alongside ZP graphs or other dialogs.
    """
    def __init__(self, parent, title: str, text: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMinimumWidth(520)

        lay = QVBoxLayout(self)

        self.lbl = QLabel(text, self)
        self.lbl.setWordWrap(True)
        self.lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        from PyQt6.QtGui import QFontDatabase
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        self.lbl.setFont(mono)
        lay.addWidget(self.lbl)

        # Copy to clipboard button — useful for observatory logs
        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Copy to Clipboard", self)
        btn_close = QPushButton("Close", self)
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(text))
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)

        self.adjustSize()

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
            kept = payload.get("kept", None)
            zp = payload.get("zp", None)
            title = payload.get("title", name)

            xa = np.asarray(x, dtype=float)
            ya = np.asarray(y, dtype=float)
            if kept is not None and len(kept) == len(xa):
                km = np.asarray(kept, dtype=bool)
                # rejected first so kept points draw on top
                ax.scatter(xa[~km], ya[~km], s=16, alpha=0.5, color="red",
                           label=f"clipped ({int((~km).sum())})", zorder=2)
                ax.scatter(xa[km], ya[km], s=16, alpha=0.8, color="C0",
                           label=f"fit ({int(km.sum())})", zorder=3)
                ax.legend(fontsize=8, loc="upper left")
            else:
                # older payloads without a mask — behave as before
                ax.scatter(xa, ya, s=16, alpha=0.8, color="C0")

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

        btn_close = QPushButton("Close", self)
        btn_close.clicked.connect(self.close)
        root.addWidget(btn_close)
        
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)



class _AavsoBatchDialog(QDialog):
    """Review accumulated AAVSO rows, delete unwanted ones, and export a valid
    AAVSO Extended file (observer code persisted; offer to open WebObs)."""
 
    def __init__(self, parent, rows):
        super().__init__(parent)
        self.setWindowTitle("AAVSO Submission Batch")
        self.resize(900, 420)
        self._rows = rows           # shared list[(AavsoRow, ra, dec)] (mutated in place)
 
        v = QVBoxLayout(self)
        cols = ["NAME", "DATE (JD)", "MAG", "MERR", "FILT", "KNAME", "KMAG", "AMASS"]
        self.table = QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table, 1)
        self._reload()
 
        btns = QHBoxLayout()
        self.btn_del = QPushButton("Delete Selected")
        self.btn_del.clicked.connect(self._delete_selected)
        self.btn_clear = QPushButton("Clear Batch")
        self.btn_clear.clicked.connect(self._clear)
        self.btn_export = QPushButton("Export AAVSO File…")
        self.btn_export.clicked.connect(self._export)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_clear)
        btns.addStretch(1)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_close)
        v.addLayout(btns)
 
    def _reload(self):
        self.table.setRowCount(0)
        for (r, _ra, _dec) in self._rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            vals = [r.name, f"{r.date_jd:.5f}", f"{r.mag:.3f}",
                    ("na" if r.merr is None else f"{r.merr:.3f}"), r.filt,
                    str(r.kname), (f"{float(r.kmag):.3f}" if _num(r.kmag) else str(r.kmag)),
                    ("na" if r.amass is None else f"{r.amass:.3f}")]
            for j, val in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(val))
 
    def _delete_selected(self):
        idx = sorted({m.row() for m in self.table.selectedIndexes()}, reverse=True)
        for i in idx:
            if 0 <= i < len(self._rows):
                del self._rows[i]
        self._reload()
 
    def _clear(self):
        self._rows.clear()
        self._reload()
 
    def _export(self):
        if not self._rows:
            QMessageBox.information(self, "Export", "Batch is empty.")
            return
        names = {r.name for (r, _ra, _dec) in self._rows}
        if len(names) > 1:
            QMessageBox.warning(
                self, "Multiple Targets",
                "This batch mixes STARIDs: " + ", ".join(sorted(names)) +
                ".\nAAVSO Extended files are per-target. Export one target at a time "
                "(delete the others first).")
            return
 
        settings = QSettings()
        prev = settings.value("AAVSO/observer_code", "", type=str)
        code, ok = QInputDialog.getText(self, "Observer Code",
                                        "AAVSO observer code:", text=prev)
        if not ok or not code.strip():
            return
        code = code.strip().upper()
        settings.setValue("AAVSO/observer_code", code)
 
        _r0, ra, dec = self._rows[0]
        rows_only = [r for (r, _ra, _dec) in self._rows]
        try:
            text = aav.format_aavso_extended(rows_only, obscode=code, ra_deg=ra, dec_deg=dec)
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Could not format file:\n{e}")
            return
 
        path, _ = QFileDialog.getSaveFileName(self, "Save AAVSO File", "",
                                              "Text files (*.txt *.dat *.csv)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                f.write(text)
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Failed to write file:\n{e}")
            return
 
        msg = QMessageBox(self)
        msg.setWindowTitle("Export AAVSO")
        msg.setText(f"Wrote {len(rows_only)} row(s) →\n{path}\n\nOpen the AAVSO WebObs upload page now?")
        yes = msg.addButton("Open WebObs", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == yes:
            webbrowser.open("https://apps.aavso.org/v2/data/submit/photometry/")
 
 
def _num(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


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
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.resize(900, 600)

        self.auto_stretch = True
        self.zoom_factor = 1.0
        self._user_zoomed = False

        self.target_rect_scene = QRectF()
        self.target_item: QGraphicsRectItem | None = None
        self.bg_item: QGraphicsRectItem | None = None
        self._auto_ellipse_item = None
        self._auto_mask = None
        self._auto_verts = None
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
        self.mode_combo.addItems(["Box", "Ellipse", "Freehand", "Star (auto)"])
        v.insertWidget(1, self.mode_combo)  # under label, above view

        # Auto-star controls (only meaningful in 'Star (auto)' mode)
        self._auto_ctl = QWidget()
        _ar = QHBoxLayout(self._auto_ctl); _ar.setContentsMargins(0, 0, 0, 0)
        _ar.addWidget(QLabel("Capture:"))
        self.spin_capture = QDoubleSpinBox()
        self.spin_capture.setRange(90.0, 99.9)
        self.spin_capture.setSingleStep(0.5)
        self.spin_capture.setDecimals(1)
        self.spin_capture.setValue(99.0)
        self.spin_capture.setSuffix(" %")
        self.spin_capture.setToolTip(
            "Fraction of the star's total flux the aperture should enclose.\n"
            "The half-flux radius (HFR) alone captures only ~50%; this scales the\n"
            "ellipse out along the measured curve-of-growth so the magnitude\n"
            "reflects (nearly) the full integrated flux.")
        _ar.addWidget(self.spin_capture)
        self.chk_circular = QCheckBox("Circular")
        self.chk_circular.setToolTip("Ignore the measured eccentricity and use a circular aperture.")
        _ar.addWidget(self.chk_circular)
        _ar.addStretch(1)
        v.insertWidget(2, self._auto_ctl)
        self._auto_ctl.setVisible(False)

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
        # Changing mode clears any existing selection so it's always exactly
        # one region.
        self._clear_target_items()
        mode = self.mode_combo.currentText()
        is_auto = (mode == "Star (auto)")
        if hasattr(self, "_auto_ctl"):
            self._auto_ctl.setVisible(is_auto)
        if is_auto:
            self.lbl.setText(
                "Click a star's core to auto-fit an aperture ellipse "
                "(centroid + HFR + eccentricity). Set Capture% for the flux "
                "fraction to enclose, then Use Target.")
        else:
            self.lbl.setText(
                "Draw a Target region (Box/Ellipse/Freehand). Background will "
                "be auto-selected (gold).")

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
        seed = getattr(self.parent(), "_pending_seed_xy", None)
        if seed is not None:
            try:
                self.parent()._pending_seed_xy = None
                self.mode_combo.setCurrentText("Star (auto)")
                self._auto_pick_star(QPointF(float(seed[0]), float(seed[1])))
                # center the view on the star
                self.view.centerOn(float(seed[0]), float(seed[1]))
            except Exception:
                pass

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

        if mode == "Star (auto)":
            return self._auto_mask

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
        for it in (self.target_item, self._ellipse_item, self._path_item,
                   getattr(self, "_auto_ellipse_item", None)):
            if it is not None:
                try: self.scene.removeItem(it)
                except Exception: pass
        self.target_item = None
        self._ellipse_item = None
        self._path_item = None
        self._auto_ellipse_item = None
        self._auto_mask = None
        self._auto_verts = None
        self.target_rect_scene = QRectF()
        self._path = QPainterPath()

    def eventFilter(self, source, event):
        if source is self.view.viewport():
            et = event.type()

            if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                mode = self.mode_combo.currentText()
                if mode == "Star (auto)":
                    self._auto_pick_star(self.view.mapToScene(event.pos()))
                    return True
                self._drawing = True
                self._origin_scene = self.view.mapToScene(event.pos())

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

    def _current_geometry_px(self):
        mode = self.mode_combo.currentText()
        if mode == "Star (auto)":
            if self._auto_verts is not None and len(self._auto_verts) >= 3:
                return {"kind": "ellipse", "verts_px": np.asarray(self._auto_verts, dtype=np.float64)}
            return None
        if mode == "Box" and not self.target_rect_scene.isNull():
            r = self.target_rect_scene
            corners = [(r.left(), r.top()), (r.right(), r.top()),
                       (r.right(), r.bottom()), (r.left(), r.bottom())]
            return {"kind": "box", "verts_px": mreg.densify_polygon(corners, per_edge=16)}
        if mode == "Ellipse" and not self.target_rect_scene.isNull():
            r = self.target_rect_scene
            cx, cy = r.center().x(), r.center().y()
            ax, ay = r.width() / 2.0, r.height() / 2.0
            th = np.linspace(0.0, 2.0 * np.pi, 160, endpoint=False)
            verts = np.column_stack([cx + ax * np.cos(th), cy + ay * np.sin(th)])
            return {"kind": "ellipse", "verts_px": verts}
        if mode == "Freehand" and not self._path.isEmpty():
            verts = []
            for poly in self._path.toSubpathPolygons():
                for i in range(poly.count()):
                    p = poly.at(i)
                    verts.append((p.x(), p.y()))
            if len(verts) >= 3:
                return {"kind": "freehand", "verts_px": np.asarray(verts, dtype=np.float64)}
        return None

    # ---------- auto star detection ----------
    def _auto_gray_linear(self) -> np.ndarray:
        """Single linear grayscale plane for SEP (mean of channels)."""
        img = self._doc_image_float01()
        gray = img.mean(axis=2) if img.ndim == 3 else img
        return np.ascontiguousarray(gray, dtype=np.float32)

    def _flux_radius_one(self, data_sub, x, y, rmax, frac):
        """Radius enclosing `frac` of the flux within rmax, via SEP curve-of-growth."""
        try:
            r, _flag = sep.flux_radius(
                np.ascontiguousarray(data_sub, dtype=np.float32),
                np.array([float(x)]), np.array([float(y)]),
                np.array([float(rmax)]), [float(frac)], subpix=5)
            rv = float(np.atleast_1d(r).ravel()[0])
            if np.isfinite(rv) and rv > 0:
                return rv
        except Exception:
            pass
        return None

    def _rotated_ellipse_mask(self, H, W, cx, cy, A, B, theta) -> np.ndarray:
        A = max(float(A), 0.5); B = max(float(B), 0.5)
        R = int(math.ceil(max(A, B))) + 2
        x0 = max(0, int(math.floor(cx - R))); x1 = min(W, int(math.ceil(cx + R)) + 1)
        y0 = max(0, int(math.floor(cy - R))); y1 = min(H, int(math.ceil(cy + R)) + 1)
        mask = np.zeros((H, W), dtype=bool)
        if x1 <= x0 or y1 <= y0:
            return mask
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs.astype(np.float64) - cx
        dy = ys.astype(np.float64) - cy
        ct = math.cos(theta); st = math.sin(theta)
        xr =  dx * ct + dy * st
        yr = -dx * st + dy * ct
        mask[y0:y1, x0:x1] = (xr / A) ** 2 + (yr / B) ** 2 <= 1.0
        return mask

    def _rotated_ellipse_verts(self, cx, cy, A, B, theta, n=160) -> np.ndarray:
        t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        ex = A * np.cos(t); ey = B * np.sin(t)
        ct = math.cos(theta); st = math.sin(theta)
        vx = cx + ex * ct - ey * st
        vy = cy + ex * st + ey * ct
        return np.column_stack([vx, vy]).astype(np.float64)

    def _auto_detect_star_at(self, gray, cx, cy, capfrac, circular):
        """Detect the star nearest (cx, cy) and return an aperture ellipse enclosing
        ~capfrac of its flux: (Cx, Cy, A, B, theta, hfr, ecc, r_cap) in full-image
        px, or None."""
        H, W = gray.shape[:2]
        win = 64
        x0 = int(max(0, math.floor(cx) - win)); x1 = int(min(W, math.floor(cx) + win + 1))
        y0 = int(max(0, math.floor(cy) - win)); y1 = int(min(H, math.floor(cy) + win + 1))
        cut = np.ascontiguousarray(gray[y0:y1, x0:x1], dtype=np.float32)
        if cut.size < 25:
            return None

        try:
            bkg = sep.Background(cut)
            data_sub = cut - bkg.back()
            err = float(bkg.globalrms)
        except Exception:
            med = float(np.median(cut))
            data_sub = cut - med
            err = float(np.std(cut))
        if not (err > 0):
            err = float(np.std(data_sub)) or 1e-6

        sources = None
        for thr in (5.0, 3.0, 2.0):
            try:
                s = sep.extract(data_sub, float(thr), err=err)
            except Exception:
                s = None
            if s is not None and len(s) > 0:
                sources = s
                break
        if sources is None or len(sources) == 0:
            return None

        lx = cx - x0; ly = cy - y0
        sxx = np.asarray(sources["x"], dtype=np.float64)
        syy = np.asarray(sources["y"], dtype=np.float64)
        j = int(np.argmin((sxx - lx) ** 2 + (syy - ly) ** 2))
        if math.hypot(sxx[j] - lx, syy[j] - ly) > win:
            return None

        sx = float(sxx[j]); sy = float(syy[j])
        a = max(float(sources["a"][j]), 0.5)
        b = max(min(float(sources["b"][j]), a), 0.3)
        theta = float(sources["theta"][j])
        ecc = float(math.sqrt(max(0.0, 1.0 - (b * b) / (a * a))))

        rmax_hfr = float(min(win - 1, max(6.0 * a, 12.0)))
        hfr = self._flux_radius_one(data_sub, sx, sy, rmax_hfr, 0.5)
        if hfr is None or not (hfr > 0):
            hfr = 1.1774 * (2.0 * a)   # HWHM of a Gaussian with sigma~=a

        rmax_cap = float(min(win - 1, max(4.0 * hfr, 8.0)))
        r_cap = self._flux_radius_one(data_sub, sx, sy, rmax_cap, capfrac)
        if r_cap is None or not (r_cap > hfr):
            # Gaussian-equivalent scaling from HFR (fallback when wings are missing)
            f = min(max(capfrac, 0.5), 0.9999)
            r_cap = hfr * math.sqrt(math.log(1.0 - f) / math.log(0.5))
        r_cap = float(min(r_cap, win - 1))

        Cx = x0 + sx; Cy = y0 + sy
        if circular:
            A = B = r_cap
        else:
            q = max(1e-3, b / a)          # minor/major axis ratio
            A = r_cap / math.sqrt(q)      # area-preserving: A*B = r_cap^2, A/B = a/b
            B = r_cap * math.sqrt(q)
        return (Cx, Cy, A, B, theta, float(hfr), ecc, float(r_cap))

    def _auto_pick_star(self, scene_pt):
        try:
            gray = self._auto_gray_linear()
        except Exception as e:
            QMessageBox.warning(self, "Auto Star", str(e))
            return
        H, W = gray.shape[:2]
        cx = float(np.clip(scene_pt.x(), 0, W - 1))
        cy = float(np.clip(scene_pt.y(), 0, H - 1))
        capfrac = float(self.spin_capture.value()) / 100.0
        circular = self.chk_circular.isChecked()

        try:
            res = self._auto_detect_star_at(gray, cx, cy, capfrac, circular)
        except Exception as e:
            QMessageBox.warning(self, "Auto Star", f"Detection failed:\n{e}")
            return
        if res is None:
            QMessageBox.information(
                self, "Auto Star",
                "No star found near the click. Click closer to the star's core, "
                "or adjust Capture% and try again.")
            return

        Cx, Cy, A, B, theta, hfr, ecc, r_cap = res
        mask = self._rotated_ellipse_mask(H, W, Cx, Cy, A, B, theta)
        if int(mask.sum()) < 9:
            QMessageBox.warning(self, "Auto Star", "Detected aperture is too small.")
            return
        verts = self._rotated_ellipse_verts(Cx, Cy, A, B, theta, n=160)

        self._clear_target_items()
        self._auto_mask = mask
        self._auto_verts = verts
        poly = QPolygonF([QPointF(float(px), float(py)) for px, py in verts])
        path = QPainterPath(); path.addPolygon(poly); path.closeSubpath()
        self._auto_ellipse_item = self.scene.addPath(path, self._pen_final)

        self.lbl.setText(
            f"Auto star @ ({Cx:.1f}, {Cy:.1f}) - HFR {hfr:.2f}px, ecc {ecc:.2f}, "
            f"aperture r~={r_cap:.1f}px (~{capfrac*100:.1f}% flux, {int(mask.sum())} px). "
            f"Re-click or adjust Capture% if needed, then Use Target.")

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
                parent.set_object_mask(mask, geometry=self._current_geometry_px())

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

def _nearest_dist_stats(star_list, sources):
    if sources is None or len(sources) == 0:
        return None
    sx = sources["x"].astype(np.float64)
    sy = sources["y"].astype(np.float64)

    d = []
    for st in star_list:
        x0 = float(st.get("x", np.nan))
        y0 = float(st.get("y", np.nan))
        if not (np.isfinite(x0) and np.isfinite(y0)):
            continue
        dx = sx - x0
        dy = sy - y0
        j = int(np.argmin(dx*dx + dy*dy))
        d.append(float(np.hypot(dx[j], dy[j])))

    if not d:
        return None
    d = np.asarray(d)
    return {
        "n": int(d.size),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "min": float(d.min()),
        "max": float(d.max()),
    }

class FullImageAnnotatedDialog(QDialog):
    """
    Shows the full image with target mask (red overlay) and background rect (gold)
    annotated, suitable for export to PNG for white papers etc.
    """
    def __init__(self, parent, img_rgb8: np.ndarray, obj_mask: np.ndarray,
                 bg_rect: QRect, target_opacity: float = 0.35,
                 qualifying_bgs=None):
        super().__init__(parent)
        self.setWindowTitle("Full Image — Target & Background Annotated")
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(900, 700)

        self._img_rgb8 = img_rgb8
        self._obj_mask = obj_mask
        self._bg_rect = bg_rect
        self._opacity = target_opacity
        self._qualifying_bgs = list(qualifying_bgs or [])   # [(label, QRect), ...]
        self._show_qual_bg = False
        # build the annotated pixmap once
        self._annotated_pix = self._build_pixmap()

        lay = QVBoxLayout(self)

        # controls row
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Target opacity:"))
        self._spn_opacity = QDoubleSpinBox()
        self._spn_opacity.setRange(0.0, 1.0)
        self._spn_opacity.setSingleStep(0.05)
        self._spn_opacity.setDecimals(2)
        self._spn_opacity.setValue(target_opacity)
        self._spn_opacity.valueChanged.connect(self._on_opacity_changed)
        ctrl.addWidget(self._spn_opacity)
        self._btn_qual = QPushButton("Show Qualifying Backgrounds")
        self._btn_qual.setCheckable(True)
        self._btn_qual.setEnabled(bool(self._qualifying_bgs))
        self._btn_qual.toggled.connect(self._on_toggle_qual)
        ctrl.addWidget(self._btn_qual)        
        ctrl.addStretch(1)

        self._btn_save = QPushButton("Save Image…")
        self._btn_save.clicked.connect(self._on_save)
        ctrl.addWidget(self._btn_save)

        self._btn_copy = QPushButton("Copy to Clipboard")
        self._btn_copy.clicked.connect(self._on_copy)
        ctrl.addWidget(self._btn_copy)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        ctrl.addWidget(btn_close)
        lay.addLayout(ctrl)

        # scrollable image view
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene, self)
        self._view.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._view.viewport().installEventFilter(self)        
        self._pix_item = self._scene.addPixmap(self._annotated_pix)
        self._scene.setSceneRect(self._pix_item.boundingRect())
        lay.addWidget(self._view, 1)

        # zoom buttons
        zoom_row = QHBoxLayout()
        for label, factor in [("Zoom In", 1.25), ("Zoom Out", 0.8), ("Fit", None), ("1:1", 0.0)]:
            b = QPushButton(label)
            if factor is None:
                b.clicked.connect(self._fit)
            elif factor == 0.0:
                b.clicked.connect(self._zoom_100)
            else:
                b.clicked.connect(lambda _=False, f=factor: self._view.scale(f, f))
            zoom_row.addWidget(b)
        zoom_row.addStretch(1)
        lay.addLayout(zoom_row)

        QTimer.singleShot(0, self._fit)

    def _on_copy(self):
        QApplication.clipboard().setPixmap(self._annotated_pix)
        # brief visual feedback
        self._btn_copy.setText("Copied!")
        QTimer.singleShot(1500, self._reset_copy_btn)

    def _reset_copy_btn(self):
        try:
            self._btn_copy.setText("Copy to Clipboard")
        except RuntimeError:
            pass        

    def _build_pixmap(self) -> QPixmap:
        img = self._img_rgb8.copy()
        H, W = img.shape[:2]

        # red overlay for target mask
        if self._obj_mask is not None:
            mask = np.asarray(self._obj_mask, dtype=bool)
            if mask.shape == (H, W):
                alpha = float(self._opacity)
                base = img.astype(np.float32)
                red = np.array([255.0, 60.0, 60.0], dtype=np.float32)
                base[mask] = (1.0 - alpha) * base[mask] + alpha * red
                img = np.ascontiguousarray(base.clip(0, 255).astype(np.uint8))

        qimg = QImage(img.data, W, H, img.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())

        # draw background rect in gold on top
        if self._bg_rect is not None and not self._bg_rect.isNull():
            painter = QPainter(pix)
            pen = QPen(QColor(255, 215, 0), 3)
            pen.setCosmetic(False)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 215, 0, int(0.35 * 255)))  # translucent gold fill
            painter.drawRect(self._bg_rect)

            # label
            painter.setFont(painter.font())
            painter.setPen(QColor(255, 215, 0))
            painter.drawText(
                self._bg_rect.left() + 4,
                self._bg_rect.top() - 6,
                "Background"
            )

            # draw target bounding box outline in red
            if self._obj_mask is not None:
                mask = np.asarray(self._obj_mask, dtype=bool)
                if mask.any():
                    ys, xs = np.nonzero(mask)
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    pen2 = QPen(QColor(255, 80, 80), 2)
                    pen2.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(pen2)
                    painter.drawRect(QRect(x0, y0, x1 - x0, y1 - y0))
                    painter.setPen(QColor(255, 80, 80))
                    painter.drawText(x0 + 4, y0 - 6, "Target")

            painter.end()

        # qualifying per-quadrant backgrounds (cyan) — robustness-check placements
        if self._show_qual_bg and self._qualifying_bgs:
            painter = QPainter(pix)
            pen = QPen(QColor(0, 200, 255), 3)   # cyan, distinct from gold primary
            pen.setCosmetic(False)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 200, 255, int(0.35 * 255)))  # translucent cyan fill
            for _label, r in self._qualifying_bgs:
                if r is not None and not r.isNull():
                    painter.drawRect(r)
            painter.setPen(QColor(0, 200, 255))
            for label, r in self._qualifying_bgs:
                if r is not None and not r.isNull():
                    painter.drawText(r.left() + 4, r.top() - 6, f"BG {label}")
            painter.end()

        return pix

    def _on_opacity_changed(self, val):
        self._opacity = float(val)
        self._annotated_pix = self._build_pixmap()
        self._pix_item.setPixmap(self._annotated_pix)
        self._scene.setSceneRect(self._pix_item.boundingRect())

    def _fit(self):
        self._view.resetTransform()
        self._view.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _zoom_100(self):
        self._view.resetTransform()

    def _on_toggle_qual(self, checked: bool):
        self._show_qual_bg = bool(checked)
        self._btn_qual.setText(
            "Hide Qualifying Backgrounds" if self._show_qual_bg
            else "Show Qualifying Backgrounds"
        )
        self._annotated_pix = self._build_pixmap()
        self._pix_item.setPixmap(self._annotated_pix)
        self._scene.setSceneRect(self._pix_item.boundingRect())

    def eventFilter(self, source, event):
        if source is self._view.viewport() and event.type() == QEvent.Type.Wheel:
            angle = event.angleDelta().y()
            if angle != 0:
                factor = 1.25 if angle > 0 else 0.8
                new_scale = self._view.transform().m11() * factor
                if 0.02 <= new_scale <= 200.0:
                    self._view.scale(factor, factor)
            return True
        return super().eventFilter(source, event)

    def _on_save(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Image", "magnitude_annotated.png",
            "PNG Image (*.png);;TIFF Image (*.tiff *.tif);;All Files (*)"
        )
        if not path:
            return
        if self._annotated_pix.save(path):
            QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
        else:
            QMessageBox.warning(self, "Save Failed", f"Could not save to:\n{path}")

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
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setMinimumSize(780, 580)
        self.resize(880, 760)
 
        self.sys_floor_mag = 0.10  # mag (typical 0.05–0.15)
        self.object_mask = None
        self.object_geometry = None   # {"kind":..., "verts_px": Nx2} from picker/recall
        self.background_mask = None
        self.doc_manager = doc_manager
 
        self.star_list: List[dict] = []
        self.wcs = None
        self.pixscale = None
 
        self.object_rect = QRect()
        self.background_rect = QRect()
 
        self.last_zp: Dict[str, Any] = {}
        self._last_measurement = None      # dict from _record_measurement()
        self._aavso_rows = []              # list[(aav.AavsoRow, ra_deg, dec_deg)]
        self._target_name = None           # resolved STARID for the current target
        self._check_star = None            # (kname, kmag_V, star_dict) once chosen for this batch
 
        self._build_ui()
        self._update_band_controls()
 
    def _build_ui(self):
        root = QVBoxLayout(self)
 
        # ── two columns ────────────────────────────────────────────────────
        cols = QHBoxLayout()
        cols.setSpacing(12)
        left = QVBoxLayout()
        right = QVBoxLayout()
        cols.addLayout(left, 3)
        cols.addLayout(right, 2)
        root.addLayout(cols, 1)
 
        # ========================= LEFT COLUMN =============================
 
        # --- Workflow (Steps 1–4, in order, with their inline helpers) ---
        wf_group = QGroupBox("Workflow")
        wf = QVBoxLayout(wf_group)
 
        self.btn_fetch = QPushButton("Step 1:  Fetch Catalog Stars  (needs WCS)")
        self.btn_fetch.clicked.connect(self.fetch_stars_from_active_doc)
        wf.addWidget(self.btn_fetch)
 
        self.btn_zp = QPushButton("Step 2:  Compute Zero Points")
        self.btn_zp.clicked.connect(self.compute_zero_points)
        wf.addWidget(self.btn_zp)
 
        self.btn_zp_plot = QPushButton("Show ZP Graphs…")
        self.btn_zp_plot.clicked.connect(self.show_zp_graphs)
        self.btn_zp_plot.setEnabled(False)
        wf.addWidget(self.btn_zp_plot)
 
        wf.addSpacing(8)
 
        self.btn_pick = QPushButton("Step 3:  Pick Target Region…")
        self.btn_pick.clicked.connect(self.open_region_picker)
        wf.addWidget(self.btn_pick)
 
        find_row = QHBoxLayout()
        self.txt_find_name = QLineEdit()
        self.txt_find_name.setPlaceholderText("…or find a star by name (e.g. IRAS 17382-5308)")
        self.txt_find_name.returnPressed.connect(self.find_star_by_name)
        self.btn_find_name = QPushButton("Go to Star")
        self.btn_find_name.clicked.connect(self.find_star_by_name)
        find_row.addWidget(self.txt_find_name, 1)
        find_row.addWidget(self.btn_find_name)
        wf.addLayout(find_row)
 
        region_row = QHBoxLayout()
        self.btn_save_region = QPushButton("Save Region…")
        self.btn_save_region.clicked.connect(self.save_region_to_library)
        self.btn_load_region = QPushButton("Load Region…")
        self.btn_load_region.clicked.connect(self.load_region_from_library)
        region_row.addWidget(self.btn_save_region)
        region_row.addWidget(self.btn_load_region)
        wf.addLayout(region_row)
 
        wf.addSpacing(8)
 
        self.btn_measure = QPushButton("Step 4:  Measure Object Region")
        self.btn_measure.clicked.connect(self.measure_object_region)
        wf.addWidget(self.btn_measure)
 
        left.addWidget(wf_group)
 
        # --- Photometry settings ---
        box = QGroupBox("Photometry Settings")
        form = QFormLayout(box)
 
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
        self.sep_sigma.setToolTip(
            "Detection threshold for SEP source extraction, in multiples of the "
            "background noise (σ above background).\n\n"
            "Only pixels brighter than this many σ are treated as sources. This sets "
            "which stars are found for matching to the catalog and for the zero-point "
            "fit — it does NOT affect the object/background measurement itself.\n\n"
            "Lower = detects fainter stars (more matches, but more noise and spurious "
            "detections); higher = only bright, clean stars. Raise toward 8–10 for "
            "noisy data or crowded fields with many false detections; 5 suits most stacks."
        )
        form.addRow("SEP detect σ", self.sep_sigma)
 
        self.clip_sigma = QDoubleSpinBox()
        self.clip_sigma.setRange(1.0, 10.0)
        self.clip_sigma.setSingleStep(0.5)
        self.clip_sigma.setValue(2.5)
        self.clip_sigma.setToolTip(
            "Sigma-clip threshold for rejecting outlier stars when deriving the "
            "photometric zero point.\n\n"
            "Each matched star gives one ZP estimate (m_cat − m_inst); stars whose "
            "ZP deviates by more than this many standard deviations from the median "
            "are discarded, over 3 iterations, before the final ZP is taken.\n\n"
            "Lower = more aggressive rejection (cleaner fit, fewer stars); "
            "higher = keeps more stars. 2.5 is a good default; drop toward 2.0 for "
            "crowded or blended fields, raise toward 3.0 if too few stars survive."
        )
        form.addRow("ZP sigma-clip", self.clip_sigma)
 
        self.sys_floor_spin = QDoubleSpinBox()
        self.sys_floor_spin.setRange(0.0, 1.0)
        self.sys_floor_spin.setDecimals(3)
        self.sys_floor_spin.setSingleStep(0.01)
        self.sys_floor_spin.setValue(float(getattr(self, "sys_floor_mag", 0.10) or 0.0))
        self.sys_floor_spin.setToolTip(
            "Systematic uncertainty floor, in magnitudes, added in quadrature to the "
            "statistical error: total σ = √(σ_stat² + σ_sys²).\n\n"
            "This is a conservative catch-all for effects not captured by photon/"
            "background statistics — zero-point color term, narrowband-to-broadband "
            "mapping, flat-field residuals, and background-placement sensitivity.\n\n"
            "The reported ± value is 3× the combined σ (total 3σ). 0.10 mag is a "
            "defensible default for calibrated narrowband surface photometry; raise it "
            "if you know your calibration is less certain, lower only with justification."
        )
        form.addRow("Systematic floor (mag)", self.sys_floor_spin)
 
        self.bg_box_size = QSpinBox()
        self.bg_box_size.setRange(10, 300)
        self.bg_box_size.setSingleStep(5)
        self.bg_box_size.setValue(50)
        self.bg_box_size.setToolTip(
            "Side length, in pixels, of the automatically-placed background sampling "
            "box (the box is square).\n\n"
            "The tool searches for the darkest, most source-free region of this size "
            "to estimate the sky background that's subtracted from the object. The same "
            "size is used for the four per-quadrant boxes in the robustness check.\n\n"
            "Larger = more pixels, lower background noise (σ ∝ 1/√area), but harder to "
            "fit into a clean gap in a crowded or nebulosity-filled field; smaller = "
            "fits tight gaps but noisier. 50 px is a good balance for most fields."
        )
        form.addRow("Auto background box (px)", self.bg_box_size)
 
        hint = QLabel(
            "Popup reports total 3σ: sqrt(stat² + sys_floor²). "
            "sys_floor is a conservative calibration-mismatch term (surface photometry). "
            "AAVSO rows export the 1σ statistical error only."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
 
        self.chk_verify_bg = QCheckBox("Verify against 4-quadrant backgrounds")
        self.chk_verify_bg.setChecked(False)
        form.addRow("", self.chk_verify_bg)
 
        left.addWidget(box)
        left.addStretch(1)
 
        # ========================= RIGHT COLUMN ============================
 
        # --- ROI preview (fills available vertical space) ---
        roi_group = QGroupBox("ROI Preview")
        roi_v = QVBoxLayout(roi_group)
 
        self.roi_preview = QLabel("No target selected.")
        self.roi_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.roi_preview.setMinimumSize(240, 240)
        self.roi_preview.setStyleSheet("border: 1px solid #444;")
        roi_v.addWidget(self.roi_preview, 1)
 
        roi_form = QFormLayout()
        self.roi_opacity = QDoubleSpinBox()
        self.roi_opacity.setRange(0.0, 1.0)
        self.roi_opacity.setSingleStep(0.05)
        self.roi_opacity.setDecimals(2)
        self.roi_opacity.setValue(0.25)
        self.roi_opacity.valueChanged.connect(self._update_roi_preview)
        roi_form.addRow("Overlay opacity", self.roi_opacity)
 
        self.roi_pad = QSpinBox()
        self.roi_pad.setRange(0, 200)
        self.roi_pad.setSingleStep(5)
        self.roi_pad.setValue(20)
        self.roi_pad.valueChanged.connect(self._update_roi_preview)
        roi_form.addRow("ROI padding (px)", self.roi_pad)
        roi_v.addLayout(roi_form)
 
        self.btn_full_view = QPushButton("Full Image View…")
        self.btn_full_view.clicked.connect(self._open_full_image_view)
        roi_v.addWidget(self.btn_full_view)
 
        right.addWidget(roi_group, 1)
 
        # --- AAVSO submission ---
        aavso_group = QGroupBox("AAVSO Submission")
        av = QVBoxLayout(aavso_group)
 
        self.btn_identify = QPushButton("Identify Target…")
        self.btn_identify.clicked.connect(self.identify_target)
        av.addWidget(self.btn_identify)
 
        self.btn_add_batch = QPushButton("Add Measurement → AAVSO Batch")
        self.btn_add_batch.clicked.connect(self.add_last_measurement_to_batch)
        av.addWidget(self.btn_add_batch)
 
        self.btn_aavso = QPushButton("AAVSO Batch…")
        self.btn_aavso.clicked.connect(self.open_aavso_batch)
        av.addWidget(self.btn_aavso)
 
        right.addWidget(aavso_group)
 
        # ========================= BOTTOM (full width) =====================
        self.lbl_info = QLabel("No stars fetched yet.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #9aa0a6;")
        root.addWidget(self.lbl_info)
 
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all)
        bottom.addWidget(self.btn_clear_all)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        bottom.addWidget(self.btn_close)
        root.addLayout(bottom)
 
        self._refresh_aavso_button()


    def _open_full_image_view(self):
        if self.object_mask is None:
            QMessageBox.information(self, "No Target", "Pick a target region first.")
            return

        try:
            disp = self._doc_image_rgb01_for_display()
        except Exception as e:
            QMessageBox.warning(self, "No Image", str(e))
            return

        img_rgb8 = np.ascontiguousarray(
            (np.clip(disp, 0, 1) * 255).astype(np.uint8)
        )

        opacity = float(self.roi_opacity.value()) if hasattr(self, "roi_opacity") else 0.35

        # Compute qualifying per-quadrant backgrounds on the LINEAR image, so the
        # boxes drawn here are exactly the placements the robustness check measures.
        qual_bgs = []
        try:
            img, _hdr, _doc = self._get_active_image_and_header()
            if img is not None:
                img_f = _to_float_image(img)
                box = int(self.bg_box_size.value()) if hasattr(self, "bg_box_size") else 50
                px, py, pw, ph = (self.background_rect.x(), self.background_rect.y(),
                                  self.background_rect.width(), self.background_rect.height())
                quads = mreg.find_quadrant_backgrounds(
                    img_f, box=box, margin=100, auto_rect_box=auto_rect_box,
                    exclude_rect=(px, py, pw, ph) if not self.background_rect.isNull() else None)
                qual_bgs = [(lbl, QRect(int(x), int(y), int(w), int(h)))
                            for (lbl, x, y, w, h) in quads]
        except Exception:
            qual_bgs = []

        dlg = FullImageAnnotatedDialog(
            parent=self,
            img_rgb8=img_rgb8,
            obj_mask=self.object_mask,
            bg_rect=self.background_rect,
            target_opacity=opacity,
            qualifying_bgs=qual_bgs,
        )
        dlg.show()
        dlg.raise_()

    def save_region_to_library(self):
        if self.object_mask is None or np.count_nonzero(self.object_mask) < 25:
            QMessageBox.information(self, "No Region", "Draw or load a target region first.")
            return
        _img, hdr, _doc = self._get_active_image_and_header()
        wcs, pixscale = _build_wcs_and_pixscale(hdr)
        if wcs is None:
            QMessageBox.warning(self, "No WCS", "Storing a region needs a plate-solved (WCS) image.")
            return

        geom = getattr(self, "object_geometry", None)
        if geom and geom.get("verts_px") is not None:
            kind = geom.get("kind", "freehand")
            verts_px = np.asarray(geom["verts_px"], dtype=np.float64)
        else:
            # fallback: rebuild a box polygon from the mask bbox
            ys, xs = np.nonzero(self.object_mask)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            kind = "box"
            verts_px = mreg.densify_polygon(
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], per_edge=16)

        obj_default = _header_str(hdr, "OBJECT")
        name, ok = QInputDialog.getText(self, "Save Region", "Region name:",
                                        text=(obj_default or ""))
        if not ok or not name.strip():
            return
        name = name.strip()

        region = mreg.build_sky_region(name, kind, verts_px, self.object_mask, wcs,
                                       object_name=obj_default, pixscale=pixscale)
        lib = mreg.RegionLibrary()
        if lib.get(name) is not None:
            if QMessageBox.question(
                self, "Overwrite", f"Region '{name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                return
        lib.add(region, overwrite=True)

        ctr = (f"RA={region.ra_center:.5f}, Dec={region.dec_center:.5f}"
               if region.ra_center is not None else "center N/A")
        QMessageBox.information(self, "Saved", f"Saved region '{name}'.\n{ctr}\n\n{lib.path}")

    def load_region_from_library(self):
        img, hdr, _doc = self._get_active_image_and_header()
        if img is None:
            QMessageBox.warning(self, "No Image", "Open an image first.")
            return
        wcs, _pixscale = _build_wcs_and_pixscale(hdr)
        if wcs is None:
            QMessageBox.warning(self, "No WCS", "Recalling a region needs a plate-solved (WCS) image.")
            return
        lib = mreg.RegionLibrary()
        if not lib.names():
            QMessageBox.information(self, "Library Empty", "No saved regions yet.")
            return

        dlg = _RegionManagerDialog(self, lib)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.selected_name:
            return
        region = lib.get(dlg.selected_name)
        if region is None:
            return

        H, W = np.asarray(img).shape[:2]
        mask, px = mreg.region_to_mask(region, H, W, wcs)
        if np.count_nonzero(mask) < 25:
            QMessageBox.warning(
                self, "Off-Field",
                "The recalled region projects to <25 px on this image "
                "(different field, or region off-frame).")
            return

        self.set_object_mask(mask, geometry={"kind": region.kind, "verts_px": px})
        ctr = (f"RA={region.ra_center:.5f}, Dec={region.dec_center:.5f}"
               if region.ra_center is not None else "")
        self.lbl_info.setText(
            f"Loaded region '{region.name}' ({int(np.count_nonzero(mask))} px). {ctr}")

    # --- external wiring (from your ROI tool) ---
    def clear_all(self):
        """
        Clear ALL cached state:
          - in-memory stars + zero points + masks/rects
          - any cached catalog stars stored in the active document metadata
        """
        if QMessageBox.question(
            self, "Clear All",
            "Clear all cached stars, zero points, and regions for this image?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) != QMessageBox.StandardButton.Yes:
            return        
        # ---- clear in-memory state ----
        self.star_list = []
        self.last_zp = {}

        self.wcs = None
        self.pixscale = None

        self.object_mask = None
        self.background_mask = None

        self.object_rect = QRect()
        self.background_rect = QRect()

        # UI state
        try:
            self.btn_zp_plot.setEnabled(False)
        except Exception:
            pass

        self.lbl_info.setText("Cleared. No stars fetched yet.")

        # ---- clear cached metadata on the active document ----
        img, hdr, doc = self._get_active_image_and_header()
        if doc is None:
            return

        meta = dict(getattr(doc, "metadata", {}) or {})
        # Remove / invalidate the shared SFCC cache keys (used by this tool too)
        # IMPORTANT: set to None so metadata-merge update overwrites old values.
        for k in ("SFCC_star_list", "SFCC_catalog",
                  "MAG_star_list", "MAG_catalog", "MAG_last_zp"):
            meta[k] = None
        # Remove the shared SFCC cache keys (used by this tool too)
        meta.pop("SFCC_star_list", None)
        meta.pop("SFCC_catalog", None)

        # Optional: if you add magnitude-specific caches later, clear them here too:
        meta.pop("MAG_star_list", None)
        meta.pop("MAG_catalog", None)
        meta.pop("MAG_last_zp", None)
        try:
            doc.metadata = meta
        except Exception:
            pass
        # Write back
        try:
            self.doc_manager.update_active_document(
                doc.image,
                metadata=meta,
                step_name="Magnitude Tool: Clear All",
                doc=doc,
            )
        except Exception as e:
            print("MagnitudeTool clear_all: metadata update failed:", e)


    def show_zp_graphs(self):
            p = self.last_zp.get("plot") if isinstance(self.last_zp, dict) else None
            if not p:
                QMessageBox.information(self, "ZP Graphs", "Compute zero points first.")
                return
            dlg = ZeroPointPlotsDialog(self, p)
            dlg.setWindowFlag(Qt.WindowType.Window, True)
            dlg.setWindowModality(Qt.WindowModality.NonModal)
            dlg.setModal(False)
            dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            dlg.show()
            dlg.raise_()

    def set_object_mask(self, mask: np.ndarray, geometry=None):
        self.object_mask = mask
        self.object_geometry = geometry
        self._update_roi_preview()

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

    def _doc_image_rgb01_for_display(self) -> np.ndarray:
        """
        Display-only RGB float in [0,1]. Safe for mono too (replicates to RGB).
        DOES NOT affect photometry.
        """
        img, _hdr, _doc = self._get_active_image_and_header()
        if img is None:
            raise ValueError("No active image.")
        a = np.asarray(img)

        # normalize ints
        if a.dtype.kind in "ui":
            mx = float(np.iinfo(a.dtype).max)
            a = a.astype(np.float32) / (mx if mx > 0 else 1.0)
        else:
            a = a.astype(np.float32, copy=False)

        if a.ndim == 2:
            a = np.dstack([a, a, a])
        elif a.ndim == 3:
            if a.shape[2] == 1:
                m = a[..., 0]
                a = np.dstack([m, m, m])
            else:
                a = a[..., :3]
        else:
            raise ValueError(f"Unsupported image shape: {a.shape}")

        # stretch for preview (fast + visible)
        try:
            disp = stretch_color_image(
                a,
                target_median=0.35,
                linked=True,
                normalize=False,
                apply_curves=False,
                curves_boost=0.0,
                blackpoint_sigma=3.5,
                no_black_clip=False,
                hdr_compress=False,
                hdr_amount=0.0,
                hdr_knee=0.75,
                luma_only=False,
                high_range=False,
            )
            return np.clip(disp, 0.0, 1.0).astype(np.float32, copy=False)
        except Exception:
            return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)

    def _background_robustness_text(self, img_f, obj_mask, hdr, primary_bg_mask):
        if not (getattr(self, "chk_verify_bg", None) and self.chk_verify_bg.isChecked()):
            return ""
        _, pixscale = _build_wcs_and_pixscale(hdr)
        if not (pixscale and pixscale > 0):
            return "\nBackground robustness: N/A (no pixscale from WCS).\n"

        mode = (self.last_zp.get("mode") or ("mono" if img_f.ndim == 2 else "rgb"))
        try:
            sys_floor = float(self.sys_floor_spin.value())
        except Exception:
            sys_floor = float(getattr(self, "sys_floor_mag", 0.0) or 0.0)

        if mode == "mono":
            zp_sem = self.last_zp.get("sem")
            if zp_sem is None:
                sd = self.last_zp.get("std"); n = int(self.last_zp.get("n") or 0)
                zp_sem = (float(sd) / math.sqrt(n)) if (sd is not None and n > 1) else None
            zp_state = {"ZP": self.last_zp.get("ZP"), "zp_sem": zp_sem}
        else:
            zp_state = {k: self.last_zp.get(k)
                        for k in ("ZP_R", "ZP_G", "ZP_B", "sem_R", "sem_G", "sem_B")}

        H, W = img_f.shape[:2]

        def rect_mask(x, y, w, h):
            m = np.zeros((H, W), dtype=bool)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = True
            return m

        px, py, pw, ph = (self.background_rect.x(), self.background_rect.y(),
                          self.background_rect.width(), self.background_rect.height())
        quads = mreg.find_quadrant_backgrounds(
            img_f, box=int(self.bg_box_size.value()), margin=100,
            auto_rect_box=auto_rect_box,
            exclude_rect=(px, py, pw, ph) if not self.background_rect.isNull() else None)

        rows = [("Primary", primary_bg_mask)]
        rows += [(f"Quad {lbl}", rect_mask(x, y, w, h)) for (lbl, x, y, w, h) in quads]
        results = [(lbl, mreg.measure_mu(img_f, obj_mask, bm, pixscale, zp_state, sys_floor, mode))
                   for (lbl, bm) in rows]

        lines = ["", "Background robustness (object µ vs background placement):"]
        if mode == "mono":
            prim = next((r["mu"] for l, r in results if l == "Primary" and r), None)
            prim3 = next((r["mu_3"] for l, r in results if l == "Primary" and r), None)
            vals = []
            for lbl, r in results:
                mu = r.get("mu") if r else None
                if mu is None:
                    lines.append(f"  {lbl:<12} µ = N/A"); continue
                vals.append(mu)
                if prim is not None and lbl != "Primary":
                    lines.append(f"  {lbl:<12} µ = {mu:.3f}   (Δ {mu - prim:+.3f})")
                else:
                    lines.append(f"  {lbl:<12} µ = {mu:.3f}")
            if len(vals) >= 2 and prim is not None:
                maxdev = max(abs(v - prim) for v in vals)
                spread = max(vals) - min(vals)
                tail = f"   |   Primary total 3σ = ±{prim3:.3f} mag" if prim3 else ""
                lines.append(f"  Spread (max−min) = {spread:.3f} mag{tail}")
                if prim3:
                    verdict = "ROBUST" if maxdev <= prim3 else "SENSITIVE"
                    lines.append(f"  → {verdict}: max |Δµ| = {maxdev:.3f} mag "
                                 f"= {maxdev / prim3:.2f}× the 3σ envelope.")
        else:
            chan = ["R", "G", "B"]
            for lbl, r in results:
                if not r or r.get("mu") is None:
                    lines.append(f"  {lbl:<12} µ(R,G,B) = N/A"); continue
                s = ", ".join((f"{m:.3f}" if m is not None else "N/A") for m in r["mu"])
                lines.append(f"  {lbl:<12} µ(R,G,B) = {s}")
            prim = next((r for l, r in results if l == "Primary" and r), None)
            if prim and prim.get("mu") is not None:
                pm, p3 = prim["mu"], prim.get("mu_3")
                for c in range(3):
                    cvals = [r["mu"][c] for _, r in results
                             if r and r.get("mu") and r["mu"][c] is not None]
                    if len(cvals) >= 2 and pm[c] is not None:
                        maxdev = max(abs(v - pm[c]) for v in cvals)
                        spread = max(cvals) - min(cvals)
                        env = p3[c] if p3 else None
                        extra = f"  (3σ ±{env:.3f}, {maxdev / env:.2f}×)" if env else ""
                        lines.append(f"  {chan[c]}: spread {spread:.3f}, max|Δ| {maxdev:.3f}{extra}")
        return "\n".join(lines) + "\n"

    def _update_roi_preview(self):
        """
        Show a cropped preview around the current object_mask with a red alpha overlay.
        """
        if self.roi_preview is None:
            return

        img, _hdr, _doc = self._get_active_image_and_header()
        if img is None or self.object_mask is None:
            self.roi_preview.setText("No target selected.")
            self.roi_preview.setPixmap(QPixmap())
            return

        try:
            disp = self._doc_image_rgb01_for_display()
        except Exception:
            self.roi_preview.setText("No image.")
            self.roi_preview.setPixmap(QPixmap())
            return

        mask = np.asarray(self.object_mask, dtype=bool)
        H, W = disp.shape[:2]
        if mask.shape != (H, W):
            self.roi_preview.setText("ROI mask size mismatch.")
            self.roi_preview.setPixmap(QPixmap())
            return

        if np.count_nonzero(mask) < 10:
            self.roi_preview.setText("ROI too small.")
            self.roi_preview.setPixmap(QPixmap())
            return

        # bbox + padding
        ys, xs = np.nonzero(mask)
        pad = int(self.roi_pad.value()) if hasattr(self, "roi_pad") else 20
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(W, int(xs.max()) + pad + 1)
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(H, int(ys.max()) + pad + 1)

        crop = disp[y0:y1, x0:x1, :]
        crop_mask = mask[y0:y1, x0:x1]

        # to 8-bit RGB for QImage
        rgb8 = np.ascontiguousarray((crop * 255.0).clip(0, 255).astype(np.uint8))

        # alpha blend overlay (red)
        alpha = float(self.roi_opacity.value()) if hasattr(self, "roi_opacity") else 0.35
        alpha = max(0.0, min(1.0, alpha))
        if alpha > 0.0:
            base = rgb8.astype(np.float32)
            red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
            m = crop_mask
            base[m] = (1.0 - alpha) * base[m] + alpha * red
            rgb8 = np.ascontiguousarray(base.clip(0, 255).astype(np.uint8))

        h, w = rgb8.shape[:2]
        qimg = QImage(rgb8.data, w, h, rgb8.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())  # copy protects against buffer lifetime issues

        # scale to label
        target = self.roi_preview.size()
        pix = pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.roi_preview.setText("")
        self.roi_preview.setPixmap(pix)


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
        cached_catalog = str(meta.get("SFCC_catalog") or "").strip().upper()

        cached_ok = isinstance(cached, list) and len(cached) > 0

        # 0) Fast path: any cached stars are always better than re-querying
        #    (and avoids your “APASS empty” and “SIMBAD wedged” scenarios entirely)
        if cached_ok:
            self.star_list = cached
        else:
            # 1) Try APASS (may legitimately return empty, or fail if VizieR is down)
            apass = []
            apass_reason = ""
            try:
                self.lbl_info.setText("Querying APASS (VizieR)…")
                QApplication.processEvents()
                apass = self._fetch_apass_stars_and_cache(img, hdr, doc) or []
                if not apass:
                    apass_reason = ("VizieR returned an empty table for this field "
                                    "(no APASS coverage here, or the field is too small).")
            except Exception as e:
                apass = []
                apass_reason = _classify_vizier_error(e)

            # Reasons are defined up front so the popup can never NameError,
            # regardless of which tier we bailed out on.
            local_reason = ""
            simbad_reason = ""

            if isinstance(apass, list) and len(apass) > 0:
                self.star_list = apass
            else:
                # 2) SIMBAD (network) — real catalog magnitudes
                simbad_reason = ""
                try:
                    self.lbl_info.setText("Querying SIMBAD (subprocess)…")
                    QApplication.processEvents()
                    self.star_list = self._fetch_simbad_stars_and_cache(img, hdr, doc) or []
                    if not self.star_list:
                        simbad_reason = "SIMBAD returned no stars for this field."
                except Exception as e:
                    self.star_list = []
                    simbad_reason = str(e)

                # 3) Local Gaia XP — offline fallback (now G-anchored, calibrated)
                if not self.star_list:
                    try:
                        self.lbl_info.setText("Falling back to local Gaia XP library…")
                        QApplication.processEvents()
                        self.star_list = self._fetch_local_gaia_stars_and_cache(img, hdr, doc) or []
                        if not self.star_list:
                            local_reason = "No local Gaia matches for this field."
                    except Exception as e:
                        self.star_list = []
                        local_reason = str(e)

                if not self.star_list:
                    self.lbl_info.setText("Catalog unavailable — no stars fetched.")
                    lines = [
                        "No catalog stars could be retrieved for this field.",
                        "",
                        f"• APASS (VizieR):  {apass_reason or 'failed'}",
                        f"• Local Gaia XP:  {local_reason or 'not available'}",
                        f"• SIMBAD:  {simbad_reason or 'failed'}",
                        "• No cached stars were available for this image.",
                        "",
                    ]
                    all_reasons = (apass_reason + local_reason + simbad_reason).lower()
                    if any(t in all_reasons for t in ("down", "unreachable", "timed out")):
                        lines.append(
                            "The online catalogs (VizieR/APASS, SIMBAD) look like "
                            "transient outages rather than a problem with your image — "
                            "VizieR has had recent downtime. Try again in a few minutes.")
                    else:
                        lines.append(
                            "Try a wider field or verify the WCS/plate solution.")
                    # The durable fix: local Gaia removes the network entirely.
                    if "not installed" in local_reason.lower() or "no local gaia" in local_reason.lower():
                        lines.append("")
                        lines.append(
                            "Tip: install the Gaia XP library (used by SPCC) to make "
                            "star fetching work fully offline — it isn't affected by "
                            "VizieR or SIMBAD outages.")
                    QMessageBox.information(self, "Catalog Unavailable", "\n".join(lines))
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

        # --- Auto-size aperture/annulus from scene FWHM ---
        fwhm = _estimate_fwhm_from_sources(sources)
        if fwhm is not None:
            self.lbl_info.setText(
                f"Scene FWHM ≈ {fwhm:.2f} px (median) — per-star apertures will be used for ZP."
            )
        else:
            self.lbl_info.setText("FWHM estimation failed; per-star radii from SEP axes will still be used.")
        QApplication.processEvents()

        for r in (3.0, 6.0, 12.0, 24.0):
            matches = _match_starlist_to_sources(self.star_list, sources, max_dist_px=r)
            if len(matches) >= 16:
                break

        if not matches:
            QMessageBox.warning(self, "No Matches",
                "No catalog stars matched SEP detections.\n\n"
                "Likely causes:\n"
                "• Cached star list from different crop/scale\n"
                "• WCS header doesn't match this pixel grid\n"
                "• Solve is offset/rotated beyond match tolerance\n"
            )
            return

        # --- warn if match quality is suspiciously low ---
        n_cat = len(self.star_list or [])
        n_m   = len(matches)

        frac_cat = (n_m / max(1, n_cat))

        # Optional: keep SEP count ONLY for informational text (not gating)
        try:
            n_src = int(len(sources))
        except Exception:
            n_src = int(getattr(sources, "size", 0) or 0)
        frac_src = (n_m / max(1, n_src)) if n_src else 0.0

        # thresholds:
        # - if you have a real catalog (>= 20 stars), warn if you match < 15%
        # - always warn if absolute matches are very low (WCS/crop mismatch often gives < ~10-12)
        warn = ((n_cat >= 20) and (frac_cat < 0.15)) or (n_m < 12)

        if warn:
            msg = (
                "Low catalog↔source match rate.\n\n"
                f"Catalog stars: {n_cat}\n"
                f"Matched:       {n_m}\n"
                f"Matched/Catalog: {frac_cat*100:.1f}%\n\n"
                # informational only:
                f"(SEP sources: {n_src}, Matched/Sources: {frac_src*100:.1f}% — informational)\n\n"
                "This can mean the WCS header does not match the current pixel grid "
                "(crop/resize/CRPIX shift), or cached stars were generated from a different image.\n\n"
                "Recommended:\n"
                "• Re-solve (or blind-solve) THIS exact image/crop.\n"
                "• Then re-run Step 1 and Step 2.\n\n"
                "If the image is noisy, consider raising SEP detect σ to reduce non-stellar detections."
            )
            QMessageBox.warning(self, "WCS / Match Warning", msg)
        self.lbl_info.setText(f"Matched {len(matches)} stars. Measuring apertures…")
        QApplication.processEvents()
        stats = _nearest_dist_stats(self.star_list, sources)
        print("nearest dist stats:", stats)

        # Crowding heuristic: how many stars have a neighbor inside their
        # background annulus? High fractions (dense fields, globular cores) mean
        # annulus background subtraction is contaminated by neighbor light, which
        # biases faint-end photometry and can bend the ZP relation.
        crowd_frac, n_crowded, n_crowd_total = _crowding_fraction(matches)
        self._last_crowd_note = ""
        if crowd_frac >= 0.25 and n_crowd_total >= 10:
            self._last_crowd_note = (
                f"Dense field: {crowd_frac*100:.0f}% of stars "
                f"({n_crowded}/{n_crowd_total}) have a neighbor inside their "
                f"background annulus — ZP and faint-end photometry may be "
                f"affected by crowding."
            )

        band = self.band_combo.currentText().strip().upper()

        if img_f.ndim == 2:
            zp = _compute_zero_points_mono(
                matches=matches,
                img_f=img_f,
                band=band,
                clip_sigma=float(self.clip_sigma.value()),
            )
            self.last_zp = {"mode": "mono", **zp}
            self.btn_zp_plot.setEnabled(bool(self.last_zp.get("plot")))

            _crowd = getattr(self, "_last_crowd_note", "")
            self.lbl_info.setText(
                "Zero point (mono):\n"
                f"  Band={zp.get('band')}  (catalog {zp.get('magkey')})\n"
                f"  ZP={zp.get('ZP')} (n={zp.get('n')}, σ={zp.get('std')})"
                + (f"\n\n⚠ {_crowd}" if _crowd else "")
            )
        else:
            zp = _compute_zero_points(
                matches=matches,
                img_f=img_f,
                clip_sigma=float(self.clip_sigma.value()),
            )
            self.last_zp = {"mode": "rgb", **zp}
            self.btn_zp_plot.setEnabled(bool(self.last_zp.get("plot")))

            _crowd = getattr(self, "_last_crowd_note", "")
            self.lbl_info.setText(
                "Zero points (median ± SEM):\n"
                f"  ZP_R={zp['ZP_R']} (n={zp['n_R']}, scatter={zp['std_R']}, sem={zp['sem_R']})\n"
                f"  ZP_G={zp['ZP_G']} (n={zp['n_G']}, scatter={zp['std_G']}, sem={zp['sem_G']})\n"
                f"  ZP_B={zp['ZP_B']} (n={zp['n_B']}, scatter={zp['std_B']}, sem={zp['sem_B']})"
                + (f"\n\n⚠ {_crowd}" if _crowd else "")
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
            msg += self._background_robustness_text(img_f, obj_mask, hdr, bg_mask)
            self._record_measurement(
                mode="mono",
                mags={"L": m}, merrs={"L": total_sigma(m_stat)},
                merrs_stat={"L": m_stat},
                mono_band=band, hdr=hdr,
            )

            dlg = _ResultsDialog(self, "Magnitude Results", msg)
            dlg.show()
            dlg.raise_()
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
        msg += self._background_robustness_text(img_f, obj_mask, hdr, bg_mask)
        self._record_measurement(
            mode="rgb",
            mags={"R": mR, "G": mG, "B": mB},
            merrs={"R": total_sigma(mR_stat),
                   "G": total_sigma(mG_stat),
                   "B": total_sigma(mB_stat)},
            merrs_stat={"R": mR_stat, "G": mG_stat, "B": mB_stat},
            hdr=hdr,
        )

        dlg = _ResultsDialog(self, "Magnitude Results", msg)
        dlg.show()
        dlg.raise_()

    @staticmethod
    def _gaia_g_passband_on_grid(wl_nm: np.ndarray) -> np.ndarray:
        """Approximate Gaia G passband on the given nm grid, normalized to peak 1.
        Used only to synthesize a G magnitude for per-star anchoring to the real
        catalog G — its exact shape is not critical (offset absorbs the rest)."""
        wl = np.asarray(wl_nm, dtype=np.float64)
        # Gaia G is very broad (~330–1050 nm), peak near ~600 nm.
        center, fwhm = 600.0, 440.0
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        T = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
        T *= (wl >= 330) & (wl <= 1050)
        m = float(np.max(T)) if T.size else 0.0
        return (T / m) if m > 0 else T

    def _fetch_local_gaia_stars_and_cache(self, img, hdr, doc) -> List[dict]:
        """
        Network-free FALLBACK catalog. Matches field positions to the local Gaia
        XP library and synthesizes Johnson B/V/R by integrating each star's stored
        XP spectrum through Johnson passbands. Read-only, fully offline.

        CRITICAL: the raw XP integrals are on an arbitrary instrumental scale.
        To be usable as catalog reference magnitudes (for the ZP fit), each star's
        synthetic mags are anchored to the star's REAL Gaia G magnitude
        (phot_g_mean_mag from the library) via a per-star offset. This puts B/V/R
        on a real magnitude scale and prevents the nonsensical 30–44 mag y-axis.
        """
        from setiastro.saspro.gaia_database import get_library
        from setiastro.saspro.sfcc import (
            _johnson_bvr_passbands_on_gaia_grid, _integrate_flux_times_T)

        wcs, _ = _build_wcs_and_pixscale(hdr)
        if wcs is None:
            raise RuntimeError("Could not build WCS for local Gaia query.")
        wcs2 = wcs.celestial if hasattr(wcs, "celestial") else wcs
        H, W = img.shape[:2]

        try:
            lib = get_library()
        except Exception as e:
            raise RuntimeError(f"Local Gaia library unavailable: {e}")
        if not lib or not lib.installed_bands():
            raise RuntimeError("No local Gaia XP library is installed.")

        self.lbl_info.setText("Matching field against local Gaia XP library…")
        QApplication.processEvents()

        step = max(1, int(min(H, W) / 200))
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        pix = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        try:
            sky = wcs2.all_pix2world(pix, 0)
        except Exception as e:
            raise RuntimeError(f"WCS conversion failed for local Gaia query: {e}")

        coords = [(float(sky[i, 0]), float(sky[i, 1]))
                  for i in range(len(sky))
                  if np.isfinite(sky[i, 0]) and np.isfinite(sky[i, 1])]
        if not coords:
            return []

        matched = lib.find_nearest_batch(coords, radius_arcsec=5.0)
        source_ids = sorted({sid for (sid, _sep) in matched.values()})
        if not source_ids:
            return []

        # Passbands on the library grid (nm).
        probe = None
        for sid in source_ids:
            probe = lib.get_spectrum(int(sid))
            if probe is not None and probe.flux is not None:
                break
        if probe is None:
            return []
        wl_nm = np.asarray(probe.wavelengths, dtype=np.float64)
        T_B, T_V, T_R = _johnson_bvr_passbands_on_gaia_grid(wl_nm)
        T_G = self._gaia_g_passband_on_grid(wl_nm)   # for anchoring only

        self.lbl_info.setText(
            f"Synthesizing calibrated B/V/R from {len(source_ids):,} "
            f"local Gaia XP spectra…")
        QApplication.processEvents()

        stars = []
        for sid in source_ids:
            spec = lib.get_spectrum(int(sid))
            if spec is None or spec.flux is None:
                continue
            flux = np.asarray(spec.flux, dtype=np.float64)

            S_B = _integrate_flux_times_T(flux, wl_nm, T_B)
            S_V = _integrate_flux_times_T(flux, wl_nm, T_V)
            S_R = _integrate_flux_times_T(flux, wl_nm, T_R)
            S_G = _integrate_flux_times_T(flux, wl_nm, T_G)
            if min(S_B, S_V, S_R, S_G) <= 0:
                continue

            # Real Gaia G for this star — the calibration anchor.
            info = lib.get_source_info(int(sid))
            if not info:
                continue
            G_real = info.get("gmag")
            ra, dec = info.get("ra"), info.get("dec")
            if G_real is None or not np.isfinite(G_real) or ra is None or dec is None:
                continue

            # Per-star offset: synthetic instrumental G → real Gaia G.
            # Applying the SAME offset to B/V/R puts them on a real mag scale.
            g_synth = -2.5 * math.log10(S_G)
            C = float(G_real) - g_synth

            Bmag = -2.5 * math.log10(S_B) + C
            Vmag = -2.5 * math.log10(S_V) + C
            Rmag = -2.5 * math.log10(S_R) + C

            try:
                x, y = wcs2.all_world2pix(float(ra), float(dec), 0)
            except Exception:
                continue
            if not (0 <= x < W and 0 <= y < H):
                continue

            stars.append({
                "ra": float(ra), "dec": float(dec),
                "x": float(x), "y": float(y),
                "Bmag": float(Bmag), "Vmag": float(Vmag), "Rmag": float(Rmag),
                "gaia_gmag": float(G_real),
                "gaia_source_id": int(sid),
                "sp_clean": None, "pickles_match": None,
            })

        if not stars:
            return []

        meta = dict(getattr(doc, "metadata", {}) or {})
        meta["SFCC_star_list"] = stars
        meta["SFCC_catalog"] = "GAIA_XP_LOCAL"
        self.doc_manager.update_active_document(
            doc.image, metadata=meta,
            step_name="Magnitude Stars Cached (Local Gaia XP)", doc=doc
        )
        return stars

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
        # unistra is the current primary; u-strasbg redirects to it; harvard is
        # the independent US mirror. Order = try-first order.
        servers = ["simbad.cds.unistra.fr", "simbad.u-strasbg.fr", "simbad.harvard.edu"]
        HARD_TIMEOUT_S = 15.0
        ROW_LIMIT = 10000

        # Whole-run cap. Worst case ≈ n_mirrors × (reachability 3s + query 15s).
        # This is the hard backstop for a mirror that accepts the connection but
        # never responds — the case the reachability probe can't catch.
        SUBPROC_TIMEOUT_S = 55.0

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

    def _wcs_or_build(self):
        """Return a usable celestial WCS, building one from the header if fetch
        hasn't run yet."""
        w = getattr(self, "wcs", None)
        if w is not None:
            return w
        _img, hdr, _doc = self._get_active_image_and_header()
        w, _ps = _build_wcs_and_pixscale(hdr)   # module-level helper
        return w
    
    
    def _target_centroid_radec(self):
        """Flux-weighted centroid of the current object mask → (ra_deg, dec_deg),
        or None. Falls back to a stored measurement centroid."""
        img, _hdr, _doc = self._get_active_image_and_header()
        mask = getattr(self, "object_mask", None)
        w = self._wcs_or_build()
        if img is None or mask is None or w is None:
            # fall back to whatever the last measurement stored
            lm = getattr(self, "_last_measurement", None)
            if lm and lm.get("radec"):
                return lm["radec"]
            return None
        cen = aav.flux_weighted_centroid(_to_float_image(img), mask)
        if cen is None:
            return None
        cx, cy = cen
        try:
            world = w.pixel_to_world(cx, cy)
            return float(world.ra.deg), float(world.dec.deg)
        except Exception:
            try:
                ra, dec = w.all_pix2world(cx, cy, 0)
                return float(ra), float(dec)
            except Exception:
                return None
    
    
    def _record_measurement(self, *, mode, mags, merrs, hdr, mono_band=None, merrs_stat=None):
        """Persist the just-computed photometry so it can be identified/exported.
        Called at the tail of measure_object_region."""
        # object coordinates + observation time from the ORIGINAL header (the WCS
        # header from a plate solve often lacks DATE-OBS/EXPOSURE)
        _img, wcs_hdr, doc = self._get_active_image_and_header()
        meta = getattr(doc, "metadata", {}) or {}
        time_hdr = meta.get("original_header") or hdr
    
        radec = self._target_centroid_radec()
        exptime = aav.exptime_from_header(time_hdr)
        jd = aav.obs_jd_from_header(time_hdr, exptime_s=exptime, to_midpoint=True)
        amass = None
        if radec is not None:
            amass = aav.airmass_from_header(time_hdr, radec[0], radec[1])
    
        # AAVSO filter code from the FITS FILTER card. RGB frames map channels to
        # TR/TG/TB directly, so this only drives the mono path — but we capture it
        # for both so a mono sub with a FILTER card needs no prompt.
        filt_raw = None
        for k in ("FILTER", "FILTER1", "FILT", "FILTNAME", "INSFLNAM"):
            val = aav._hget(time_hdr, k)
            if val not in (None, ""):
                filt_raw = str(val)
                break
        f_code, f_conf, _ = aav.filter_to_aavso(filt_raw)
    
        self._last_measurement = {
            "mode": mode,
            "mags": dict(mags),
            "merrs": dict(merrs),
            "merrs_stat": (dict(merrs_stat) if merrs_stat is not None else dict(merrs)),
            "mono_band": mono_band,
            "radec": radec,
            "jd": jd,
            "amass": amass,
            "filter_raw": filt_raw,
            "filter_aavso": f_code,
            "filter_conf": f_conf,
        }
    
    
    def find_star_by_name(self):
        """Resolve a typed name → sky position → pixel, then open the region picker
        positioned on it (auto-fitting if the picker supports a seed)."""
        name = self.txt_find_name.text().strip()
        if not name:
            return
        w = self._wcs_or_build()
        if w is None:
            QMessageBox.warning(self, "Find Star",
                                "Need a WCS. Fetch catalog stars (Step 1) first, or plate-solve the image.")
            return
        self.lbl_info.setText(f"Resolving '{name}' via SIMBAD…")
        QApplication.processEvents()
        try:
            radec = aav.resolve_name_to_radec(name)
        except Exception as e:
            QMessageBox.critical(self, "Find Star", f"Name resolver failed:\n{e}")
            return
        if radec is None:
            QMessageBox.information(self, "Find Star", f"Could not resolve '{name}'.")
            return
    
        ra, dec = radec
        img, _hdr, _doc = self._get_active_image_and_header()
        H, W = np.asarray(img).shape[:2]
        try:
            x, y = w.all_world2pix(ra, dec, 0)
            x, y = float(x), float(y)
        except Exception as e:
            QMessageBox.warning(self, "Find Star", f"Could not project to pixels:\n{e}")
            return
        if not (0 <= x < W and 0 <= y < H):
            QMessageBox.information(
                self, "Find Star",
                f"'{name}' resolves to RA={ra:.5f}, Dec={dec:.5f}, which is OUTSIDE this image.")
            return
    
        self._target_name = name          # remember the user's chosen designation
        # Open the region picker seeded at (x, y). See block 5 for the tiny hook in
        # MagnitudeRegionDialog; if you skip that hook, this still opens the picker
        # and reports the pixel for a manual click.
        self._pending_seed_xy = (x, y)
        self.open_region_picker()
        self.lbl_info.setText(f"'{name}' → pixel ({x:.0f}, {y:.0f}). Auto-fitting star…")
    
    
    def identify_target(self):
        """Identify the current target via SIMBAD cone search and resolve a VSX
        designation. Sets self._target_name for AAVSO export."""
        radec = self._target_centroid_radec()
        if radec is None:
            QMessageBox.warning(self, "Identify Target",
                                "Pick/measure a target region first (need a target and a WCS).")
            return
        ra, dec = radec
        self.lbl_info.setText("Querying SIMBAD…")
        QApplication.processEvents()
        try:
            info = aav.identify_object(ra, dec, radius_arcsec=5.0)
        except Exception as e:
            QMessageBox.critical(self, "SIMBAD Error", f"Could not reach SIMBAD:\n{e}")
            return
        if not info or not info.get("main_id"):
            QMessageBox.information(self, "Identify Target",
                                f"No SIMBAD object within 5″ of RA={ra:.5f}, Dec={dec:.5f}.")
            return
    
        main_id = info["main_id"]
        vsx = None
        try:
            vsx = aav.resolve_vsx_name(main_id=main_id, ra_deg=ra, dec_deg=dec)
        except Exception:
            vsx = None
    
        star_id = vsx or main_id
        self._target_name = star_id
    
        off = info.get("offset_arcsec")
        vmag = info.get("vmag")
        lines = [
            f"SIMBAD ID:   {main_id}",
            f"Object type: {info.get('otype') or 'n/a'}",
            f"V mag:       {vmag:.3f}" if vmag is not None else "V mag:       n/a",
            f"Offset:      {off:.2f}″" if off is not None else "Offset:      n/a",
        ]
        if vsx:
            lines.append(f"VSX / AAVSO: {vsx}   ← will be used as STARID")
        else:
            lines.append("VSX / AAVSO: not found (SIMBAD ID will be used as STARID)")
        QMessageBox.information(self, "Identify Target", "\n".join(lines))
    
    
    def _suggest_check_star(self):
        """Suggest a check star: a catalog star of *similar brightness* to the
        target (not the target itself), preferring the brightest within a tight
        magnitude window for good SNR. Returns (kname, kmag) or None."""
        star_list = getattr(self, "star_list", None) or []
        tgt = self._target_centroid_radec()

        # Target brightness reference: the green (≈V) channel, else mono L.
        tgt_mag = None
        lm = getattr(self, "_last_measurement", None)
        if lm:
            mags = lm.get("mags") or {}
            tgt_mag = mags.get("G", mags.get("L"))
            if tgt_mag is not None and not np.isfinite(tgt_mag):
                tgt_mag = None

        # Collect valid, non-target candidates with a known V.
        candidates = []  # (vmag, star)
        for s in star_list:
            v = s.get("Vmag")
            if v is None or not np.isfinite(v):
                continue
            if tgt is not None:
                d = SkyCoord(s["ra"] * u.deg, s["dec"] * u.deg).separation(
                    SkyCoord(tgt[0] * u.deg, tgt[1] * u.deg)).arcsec
                if d < 10.0:      # that's the target itself
                    continue
            candidates.append((float(v), s))
        if not candidates:
            return None

        if tgt_mag is not None:
            # Similar brightness to target; widen the window until something fits,
            # and within the window favor the brightest (best SNR check).
            chosen = None
            for window in (1.0, 2.0, 3.0, None):
                pool = (candidates if window is None
                        else [c for c in candidates if abs(c[0] - tgt_mag) <= window])
                if pool:
                    chosen = min(pool, key=lambda c: c[0])
                    break
            v, s = chosen
        else:
            # No target magnitude yet → fall back to the brightest available.
            v, s = min(candidates, key=lambda c: c[0])

        try:
            kname, _kv = aav.identify_name_and_vmag(s["ra"], s["dec"])
        except Exception:
            kname = None
        return (kname or f"RA{s['ra']:.4f}", float(v), s)
    
    def add_last_measurement_to_batch(self):
        """Turn the last measurement into AAVSO rows and append to the batch."""
        lm = getattr(self, "_last_measurement", None)
        if not lm:
            QMessageBox.information(self, "AAVSO", "Measure a region first (Step 4).")
            return
        if lm.get("radec") is None:
            QMessageBox.warning(self, "AAVSO", "No target coordinates (need a WCS). Fetch stars / plate-solve first.")
            return
        if lm.get("jd") is None:
            jd, ok = QInputDialog.getDouble(self, "Observation Date",
                                            "No DATE-OBS in header. Enter Julian Date (mid-exposure):",
                                            decimals=5)
            if not ok:
                return
            lm["jd"] = float(jd)
    
        # STARID: identify if we haven't yet, then confirm — prepopulated with the
        # identified name so it's visible and editable (this is the field the user
        # expects to see filled after Identify Target).
        if not self._target_name:
            self.identify_target()          # sets self._target_name (VSX / SIMBAD id)
        star_id, ok = QInputDialog.getText(
            self, "Target Name (STARID)",
            "AAVSO STARID for this target:",
            text=(self._target_name or ""),
        )
        if not ok or not star_id.strip():
            return
        star_id = star_id.strip()
        self._target_name = star_id
    
        # Check star (once per batch)
        if self._check_star is None:
            sug = self._suggest_check_star()
            kname = kmag = None
            kstar = None
            if sug is not None:
                kname, kmag, kstar = sug
            kname_in, ok = QInputDialog.getText(
                self, "Check Star", "Check-star designation (KNAME):",
                text=(kname or ""))
            if not ok or not kname_in.strip():
                return
            kmag_in, ok = QInputDialog.getDouble(
                self, "Check Star Magnitude", f"Catalog magnitude for {kname_in.strip()}:",
                value=float(kmag) if kmag is not None else 0.0, decimals=3)
            if not ok:
                return
            self._check_star = (kname_in.strip(), float(kmag_in), kstar)
        kname, kmag, kstar = self._check_star
    
        amass = lm.get("amass")
        if amass is None:
            amass_in, ok = QInputDialog.getDouble(self, "Airmass",
                                                "No airmass derivable from header. Enter airmass (or Cancel for na):",
                                                value=1.0, min=1.0, max=40.0, decimals=3)
            amass = float(amass_in) if ok else None
    
        ra, dec = lm["radec"]
        if lm["mode"] == "rgb":
            kmags = aav.check_star_kmags(kstar) if kstar else None
            rows = aav.build_rgb_rows(
                name=star_id, date_jd=lm["jd"],
                mags=lm["mags"], merrs=lm["merrs_stat"], amass=amass,
                kname=kname, kmag=kmag, kmags=kmags,
            )
        else:
            filt = self._resolve_mono_filter(lm)
            if filt is None:
                return
            rows = aav.build_mono_row(
                name=star_id, date_jd=lm["jd"],
                mag=lm["mags"]["L"], merr=lm["merrs_stat"]["L"], filt=filt,
                amass=amass, kname=kname, kmag=kmag,
            )
    
        for r in rows:
            self._aavso_rows.append((r, ra, dec))
        self._refresh_aavso_button()
        QMessageBox.information(self, "AAVSO",
                            f"Added {len(rows)} row(s) for {star_id}. Batch now has "
                            f"{len(self._aavso_rows)} row(s).")
    
    
    def _resolve_mono_filter(self, lm):
        """AAVSO filter code for a mono measurement. Auto-fills from the FITS FILTER
        card when the mapping is confident; otherwise prompts with a best guess
        pre-selected. Returns the code, or None if cancelled."""
        code = lm.get("filter_aavso")
        conf = lm.get("filter_conf")
        raw = lm.get("filter_raw")
    
        # Confident header mapping (e.g. FILTER='V', 'Astrodon B', "g'") → no prompt.
        if code and conf == "high":
            return code
    
        choices = ["U", "B", "V", "R", "I", "SG", "SR", "SI", "SZ",
                "CV", "CR", "TG", "TB", "TR", "J", "H", "K"]
        default = code if code in choices else "V"
        if raw:
            prefix = f"FITS FILTER = {raw!r}\n"
        else:
            prefix = "No FILTER card found in the header.\n"
        if conf == "none" and raw:
            prefix += ("This looks narrowband or non-standard for AAVSO photometry — "
                    "choose the closest valid band, or Cancel.\n")
        filt, ok = QInputDialog.getItem(
            self, "AAVSO Filter", prefix + "AAVSO filter code:",
            choices, choices.index(default), True)   # editable
        if not ok or not filt.strip():
            return None
        return filt.strip()
    
    
    def _refresh_aavso_button(self):
        n = len(getattr(self, "_aavso_rows", []))
        if hasattr(self, "btn_aavso"):
            self.btn_aavso.setText(f"AAVSO Batch… ({n})")
            self.btn_aavso.setEnabled(n > 0)
    
    
    def open_aavso_batch(self):
        if not self._aavso_rows:
            QMessageBox.information(self, "AAVSO Batch", "No measurements in the batch yet.")
            return
        dlg = _AavsoBatchDialog(self, self._aavso_rows)
        dlg.exec()
        self._refresh_aavso_button()


def open_magnitude_tool(doc_manager, parent=None) -> MagnitudeToolDialog:
    dlg = MagnitudeToolDialog(doc_manager=doc_manager, parent=parent)
    dlg.show()
    return dlg