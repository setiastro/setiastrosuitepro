"""
SASpro — Supernova / Asteroid Hunter (overhauled)

This is a from-scratch overhaul of the original SASv1-era hunter. Changes vs.
the legacy module:

  1.  Parallelism everywhere. Load, cosmetic-correct, align, ADBE and
      difference-detect all run through ProcessPoolExecutor (or a thread
      fallback when process pools don't work — same `_make_executor`
      pattern star_alignment.py uses, with the frozen-build carve-out).

  2.  Modern ADBE from abe.py — `abe_run()` with polynomial degree, optional
      RBF refinement, exclusion-aware sampling and anchored median rescaling.
      Replaces the ancient `PolyGradientRemoval` the old hunter was still
      pulling out of star_alignment.py.

  3.  Optional pre-registration. A checkbox lets the user star-align every
      search frame to the reference before differencing. Uses the same
      astroalign + anti-ring warp stack from star_alignment.py
      (`aa_find_transform_with_backoff`, `_warp_like_ref`). This fixes the
      biggest correctness bug in the old flow: pixel-wise subtraction on
      unregistered frames produced false anomalies around every star.

  4.  Noise-aware thresholding. The diff mask is built from
      median + k*MAD of the diff image, not a flat 0.10 — survives
      per-frame background-level differences and photometric mismatch.
      User still gets a sigma slider.

  5.  Star-mask subtraction. A dilated star mask built from the REFERENCE
      is subtracted from the diff before thresholding. Kills the vast
      majority of alignment-residual false positives at bright stars.

  6.  Photometric normalization. Each search frame's median/MAD is
      matched to the reference's before subtraction so a thin-cloud frame
      doesn't light up the whole diff.

  7.  Blink mode. Results dialog has a Blink tab per flagged frame that
      cycles ref → search → diff at a user-chosen rate. Way easier than
      clicking each anomaly individually.

  8.  Non-blocking. All heavy work happens in a QThread that drives the
      process pool and emits progress signals. No more processEvents()
      spam, and the dialog stays responsive with a Cancel button.

  9.  Push-to-new-view flow preserved (the user kept this).

The minor-body / catalog cross-match path is kept intact — it was already
threaded via MinorBodyWorker in the original, and that part still works.
"""

from __future__ import annotations

import os
import sys
import platform
import tempfile
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtCore import (
    Qt, QObject, QThread, QTimer, QPointF, QRectF,
    pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import (
    QAction, QCursor, QIcon, QImage, QPainter, QPixmap, QTransform,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QTextEdit,
    QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from astropy.wcs import WCS
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from astropy.io.fits import Header

from setiastro.saspro.legacy.image_manager import load_image, save_image
from setiastro.saspro.legacy.numba_utils import bulk_cosmetic_correction_numba
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from setiastro.saspro import minorbodycatalog as mbc
from setiastro.saspro.plate_solver import PlateSolverDialog as PlateSolver
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.help_support import make_help_button

# Interactive 3D orbit viewer for catalogued minor bodies. Guarded so a
# missing pyqtgraph.opengl / PyOpenGL (or any import-time hiccup) degrades
# to "no elements attached + Explore button reports why" rather than
# taking the whole hunter down.
try:
    from setiastro.saspro.orbit_explorer import (
        open_orbit_explorer, extract_elements_from_row,
    )
    _HAS_ORBIT_EXPLORER = True
    _ORBIT_EXPLORER_ERR = ""
except Exception as _oe_err:                     # pragma: no cover
    _HAS_ORBIT_EXPLORER = False
    _ORBIT_EXPLORER_ERR = str(_oe_err)

    def extract_elements_from_row(_row):         # no-op fallback
        return None

    def open_orbit_explorer(*_a, **_k):
        return None

from setiastro.saspro.plate_solver import (
    _solve_numpy_with_fallback,
    _as_header,
    _strip_wcs_keys,
    _merge_wcs_into_base_header,
)

# Modern ADBE core — replaces the ancient PolyGradientRemoval the legacy
# hunter was still using.
from setiastro.saspro.abe import abe_run

# Star alignment helpers — reused unchanged from the alignment module so
# the hunter stays consistent with whatever ring-artifact fixes land there.
from setiastro.saspro.star_alignment import (
    aa_find_transform_with_backoff,
    _warp_like_ref,
    _make_executor,
)


# ─────────────────────────────────────────────────────────────────────
# Metadata helpers (kept from the original — WCS / JD derivation)
# ─────────────────────────────────────────────────────────────────────

def _xisf_kw_value(xisf_meta: dict, key: str, default=None):
    if not xisf_meta:
        return default
    fk = xisf_meta.get("FITSKeywords", {})
    if key not in fk:
        return default
    entry = fk[key]
    if isinstance(entry, list) and entry:
        v = entry[0].get("value", default)
    elif isinstance(entry, dict):
        v = entry.get("value", default)
    else:
        v = entry
    return v


def ensure_jd_from_xisf_meta(meta: dict) -> None:
    if "jd" in meta and np.isfinite(meta["jd"]):
        return
    xisf_meta = meta.get("xisf_meta")
    if not isinstance(xisf_meta, dict):
        return
    date_obs = _xisf_kw_value(xisf_meta, "DATE-OBS")
    if not date_obs:
        return
    exp_str = (_xisf_kw_value(xisf_meta, "EXPOSURE")
               or _xisf_kw_value(xisf_meta, "EXPTIME"))
    exposure = None
    if exp_str is not None:
        try:
            exposure = float(exp_str)
        except Exception:
            exposure = None
    try:
        t = Time(date_obs, format="isot", scale="utc")
    except Exception:
        try:
            t = Time(date_obs, scale="utc")
        except Exception:
            return
    if exposure and exposure > 0:
        t = t + 0.5 * exposure * u.s
    meta["jd"] = float(t.jd)
    meta["mjd"] = float(t.mjd)
    meta.setdefault("date_obs", t.isot)


def _rescale_wcs_for_bin(header: Header, factor: int) -> Header:
    """
    Return a copy of `header` whose WCS cards are rescaled to describe the
    image after NxN binning by `factor`. Covers both the CDELT + CROTA
    style and the CD matrix style; SIP terms (A_*, B_*, AP_*, BP_*) are
    adjusted by the same factor since they're polynomial fits in pixel
    coordinates.

    Formulas (standard FITS binning convention — pixel (0,0) stays pixel
    (0,0), so CRPIX scales as (crpix - 0.5) / factor + 0.5):

        CRPIX*         →  (CRPIX* - 0.5) / f + 0.5
        CDELT*         →  CDELT* * f
        CDi_j          →  CDi_j * f
        PCi_j          →  unchanged (orientation only; CDELT carries scale)
        SIP A_m_n      →  A_m_n * f ** (1 - m - n)
        NAXIS1 / NAXIS2 → // f  (so viewers show the right size)
    """
    f = int(max(1, factor))
    if f == 1:
        return header
    h = header.copy()

    def _scale_crpix(key: str) -> None:
        if key in h:
            try:
                h[key] = (float(h[key]) - 0.5) / f + 0.5
            except Exception:
                pass

    _scale_crpix("CRPIX1")
    _scale_crpix("CRPIX2")

    for key in ("CDELT1", "CDELT2"):
        if key in h:
            try:
                h[key] = float(h[key]) * f
            except Exception:
                pass

    for i in (1, 2):
        for j in (1, 2):
            k = f"CD{i}_{j}"
            if k in h:
                try:
                    h[k] = float(h[k]) * f
                except Exception:
                    pass

    # SIP distortion coefficients — polynomial in pixel coords, so each
    # A_m_n multiplies pixel^(m+n). Rescale by f**(1-m-n) so the mapping
    # is invariant under the pixel-coordinate rescale above.
    import re
    sip_re = re.compile(r"^(A|B|AP|BP)_(\d+)_(\d+)$")
    for k in list(h.keys()):
        m = sip_re.match(str(k))
        if not m:
            continue
        try:
            exp = int(m.group(2)) + int(m.group(3))
            h[k] = float(h[k]) * (f ** (1 - exp))
        except Exception:
            pass

    for key in ("NAXIS1", "NAXIS2"):
        if key in h:
            try:
                h[key] = int(h[key]) // f
            except Exception:
                pass

    return h


def _derive_jd_from_header(header) -> float | None:
    """FITS-header JD fallback. Returns mid-exposure JD or None."""
    if not isinstance(header, (dict, Header)):
        return None
    try:
        date_obs = header.get("DATE-OBS")
        if not date_obs:
            return None
        exptime = float(header.get("EXPTIME", header.get("EXPOSURE", 0.0)))
        t = Time(str(date_obs), scale="utc")
        t_mid = t + (exptime / 2.0) * u.s
        return float(t_mid.tt.jd)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Qt helpers
# ─────────────────────────────────────────────────────────────────────

def _wcs_is_usable(wcs_obj, header=None):
    """
    Decide whether a WCS object is a REAL astrometric solution.

    This matters far more than it looks. astropy's WCS(header) does NOT
    raise on a header with no WCS keywords — it returns a perfectly
    valid default object with CRVAL=(0,0) and a 1 deg/px scale. Any code
    that only checks `wcs is not None` therefore sails straight past a
    missing solution and then projects the entire sky onto the frame.

    That is exactly how a minor-body search can grind through the whole
    catalog and then report "nothing in field": it was looking at
    RA=0, Dec=0 at one degree per pixel, so nothing ever landed inside
    the image bounds.

    Returns (ok: bool, reason: str).
    """
    if wcs_obj is None:
        return False, "no WCS object"
    try:
        if not wcs_obj.has_celestial:
            return False, "no celestial axes"
        cel = wcs_obj.celestial
        ctype = [str(c).strip() for c in cel.wcs.ctype]
        if not any(ctype):
            return False, "CTYPE empty (header carries no WCS)"
        if not any(("RA" in c.upper() or "GLON" in c.upper()) for c in ctype):
            return False, f"CTYPE not celestial: {ctype}"
        crval = np.asarray(cel.wcs.crval, dtype=float)
        if not np.all(np.isfinite(crval)):
            return False, "CRVAL not finite"
        try:
            scale = float(np.sqrt(np.abs(np.linalg.det(cel.pixel_scale_matrix))) * 3600.0)
        except Exception:
            return False, "cannot derive pixel scale"
        if not np.isfinite(scale) or scale <= 0:
            return False, "pixel scale not finite"
        # Sanity window for a deep-sky frame. astropy's unsolved default
        # is exactly 1 deg/px = 3600"/px, which this rejects outright.
        if scale > 120.0:
            return False, f'implausible scale {scale:.2f}"/px — unsolved header'
        if scale < 0.01:
            return False, f'implausible scale {scale:.4f}"/px'
        if header is not None:
            try:
                if not any(k in header for k in
                           ("CRVAL1", "CD1_1", "CDELT1", "PC1_1")):
                    return False, "header carries no CRVAL/CD/CDELT"
            except Exception:
                pass
        return True, f'{scale:.3f}"/px'
    except Exception as e:
        return False, f"WCS check failed: {e}"


def _local_utc_offset_hours(t_like) -> float:
    """
    UTC offset of THIS computer's timezone at the given wall-clock time,
    DST-aware.

    The DST part is the whole point. Asking for "my timezone offset" in
    the abstract gives whatever is true today; asking for it *at the
    moment of the exposure* gives CDT (−5) in October and CST (−6) in
    January. Getting that wrong is a silent one-hour error that moves a
    main-belt asteroid by a few arcminutes — far enough to miss, close
    enough to look like the ephemeris is merely imprecise.
    """
    import datetime as _dt
    try:
        naive = t_like.to_datetime() if hasattr(t_like, "to_datetime") else t_like
        if getattr(naive, "tzinfo", None) is not None:
            naive = naive.replace(tzinfo=None)
        # .astimezone() on a naive datetime interprets it as local time and
        # attaches the tz rules in force on THAT date.
        return float(naive.astimezone().utcoffset().total_seconds()) / 3600.0
    except Exception:
        return 0.0


def _named_tz_offset_hours(t_like, zone_name: str) -> float:
    """
    UTC offset of a NAMED IANA timezone at the given wall-clock time.

    This is the robust option for data from a site in a different zone
    than the machine doing the analysis. Pinning a fixed numeric offset
    instead works until that site crosses its own DST boundary, at which
    point every prediction quietly shifts by an hour — which on a
    main-belt asteroid is a couple of arcminutes, far enough to miss.
    """
    if not zone_name:
        return 0.0
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return 0.0
    try:
        naive = t_like.to_datetime() if hasattr(t_like, "to_datetime") else t_like
        if getattr(naive, "tzinfo", None) is not None:
            naive = naive.replace(tzinfo=None)
        return float(naive.replace(
            tzinfo=ZoneInfo(str(zone_name))).utcoffset().total_seconds()) / 3600.0
    except Exception:
        return 0.0


def _parse_sexagesimal(val):
    """
    Parse 'DD MM SS.SS', '-DD:MM:SS', '+DD MM SS' or a plain decimal into
    degrees.

    SGP writes SITELAT/SITELONG as space-separated sexagesimal, and the
    sign belongs to the WHOLE value rather than just the degrees field —
    treating '-90 11 44.16' as -90 + 11/60 + 44/3600 puts the observer
    in the wrong place by over 20 arcminutes of longitude.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return float(s)                      # already decimal
    except ValueError:
        pass
    neg = s.startswith("-")
    parts = [p for p in s.replace(":", " ").replace(",", " ").split() if p]
    try:
        nums = [abs(float(p)) for p in parts]
    except ValueError:
        return None
    if not nums:
        return None
    deg = nums[0]
    if len(nums) > 1:
        deg += nums[1] / 60.0
    if len(nums) > 2:
        deg += nums[2] / 3600.0
    return -deg if neg else deg


def _site_from_header(header):
    """
    Observer location from the acquisition header: (lat, lon, elev) or None.

    Worth reading automatically — the camera already recorded where it
    was standing, and the topocentric correction only applies when a site
    is known. Making the user retype it is a good way to have it stay
    switched off forever.
    """
    if not isinstance(header, (dict, Header)):
        return None

    def g(*keys):
        for k in keys:
            try:
                if k in header and header[k] not in (None, ""):
                    return header[k]
            except Exception:
                continue
        return None

    lat = _parse_sexagesimal(g("SITELAT", "LAT-OBS", "OBSGEO-B", "LATITUDE"))
    lon = _parse_sexagesimal(g("SITELONG", "LONG-OBS", "OBSGEO-L", "LONGITUD"))
    if lat is None or lon is None:
        return None
    try:
        elev = float(g("SITEELEV", "ALT-OBS", "OBSGEO-H", "ELEVATIO") or 0.0)
    except Exception:
        elev = 0.0
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (float(lat), float(lon), float(elev))


def _extract_obs_time(header, meta=None):
    """
    Pull the exposure START time and exposure length from whatever
    metadata is available.

    Returns (t_start | None, exptime_seconds, source_description).

    The source string is deliberately reported so the UI can show the
    user where the timestamp came from. A wrong or locally-timezoned
    DATE-OBS is one of the most common reasons an ephemeris lands
    nowhere near the object, and it is completely invisible unless the
    value is put on screen.
    """
    # 1) XISF FITSKeywords
    if isinstance(meta, dict):
        xm = meta.get("xisf_meta")
        if isinstance(xm, dict):
            d = _xisf_kw_value(xm, "DATE-OBS")
            if d:
                e = (_xisf_kw_value(xm, "EXPOSURE")
                     or _xisf_kw_value(xm, "EXPTIME") or 0.0)
                try:
                    e = float(e)
                except Exception:
                    e = 0.0
                for fmt in ("isot", None):
                    try:
                        t = (Time(str(d), format=fmt, scale="utc") if fmt
                             else Time(str(d), scale="utc"))
                        return t, e, "XISF DATE-OBS", False
                    except Exception:
                        continue

    if not isinstance(header, (dict, Header)):
        return None, 0.0, "no header available", False

    def _hget(*keys):
        for k in keys:
            try:
                if k in header:
                    v = header[k]
                    if v not in (None, ""):
                        return v
            except Exception:
                continue
        return None

    exp = _hget("EXPTIME", "EXPOSURE", "EXPOSURE_TIME")
    try:
        exptime = float(exp) if exp is not None else 0.0
    except Exception:
        exptime = 0.0

    d = _hget("DATE-OBS", "DATE_OBS")
    if d:
        s = str(d).strip()
        if "T" not in s:
            tv = _hget("TIME-OBS", "TIME_OBS")
            if tv:
                s = f"{s}T{str(tv).strip()}"
        for fmt in ("isot", None):
            try:
                t = (Time(s, format=fmt, scale="utc") if fmt
                     else Time(s, scale="utc"))
                # DATE-LOC alongside DATE-OBS means the writer keeps
                # local and UTC in separate cards, so DATE-OBS is UTC by
                # construction. SGP, NINA and ACP all do this. When we can
                # see that, there is nothing for the user to guess at.
                dloc = _hget("DATE-LOC", "DATE_LOC", "LOCALTIM")
                confirmed = bool(dloc) and str(dloc).strip() != s
                src = ("FITS DATE-OBS (UTC — DATE-LOC present separately)"
                       if confirmed else "FITS DATE-OBS")
                return t, exptime, src, confirmed
            except Exception:
                continue

    mjd = _hget("MJD-OBS", "MJD_OBS")
    if mjd is not None:
        try:
            return (Time(float(mjd), format="mjd", scale="utc"), exptime,
                    "FITS MJD-OBS (UTC by definition)", True)
        except Exception:
            pass

    jd = _hget("JD", "JD-OBS")
    if jd is not None:
        try:
            return (Time(float(jd), format="jd", scale="utc"), exptime,
                    "FITS JD (UTC by definition)", True)
        except Exception:
            pass

    return None, exptime, "no DATE-OBS / MJD-OBS / JD found", False


def _numpy_to_qimage(img: np.ndarray) -> QImage:
    if img is None:
        return QImage()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, img.strides[0],
                      QImage.Format.Format_Grayscale8)
    if img.ndim == 3:
        h, w, c = img.shape
        if c == 3:
            return QImage(img.data, w, h, img.strides[0],
                          QImage.Format.Format_RGB888)
        if c == 4:
            return QImage(img.data, w, h, img.strides[0],
                          QImage.Format.Format_RGBA8888)
        if c == 1:
            img = np.repeat(img, 3, axis=2)
            h, w, _ = img.shape
            return QImage(img.data, w, h, img.strides[0],
                          QImage.Format.Format_RGB888)
    return QImage()


def _rects_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _draw_label(img, text, x, y, color, scale=0.6, thickness=1,
                occupied=None, anchor=None):
    """
    Draw text with a dark outline so it stays legible over both empty sky
    and a bright star. Anchored at (x, y) baseline-left, clamped to stay
    inside the frame — a designation running off the edge is worse than
    no designation.

    `occupied` is a running list of rectangles already claimed by other
    labels. When supplied, a label that would collide is pushed down
    until it finds clear space, and a thin leader line is drawn back to
    `anchor` so it stays obvious which marker it belongs to. Without
    that, two labels for the same object land on top of each other and
    both become unreadable.
    """
    if not text:
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = str(text)
    (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness + 1)
    x = int(max(2, min(int(x), w - tw - 2)))
    y = int(max(th + 2, min(int(y), h - 2)))

    if occupied is not None:
        step = th + 6
        y_want = y
        for _ in range(16):
            rect = (x, y - th - 2, x + tw, y + 3)
            if not any(_rects_overlap(rect, r) for r in occupied):
                break
            y += step
            if y > h - 2:                       # ran out below — go up instead
                y = max(th + 2, y_want - step)
        y = int(max(th + 2, min(y, h - 2)))
        occupied.append((x, y - th - 2, x + tw, y + 3))
        if anchor is not None and abs(y - y_want) > th:
            cv2.line(img, (int(anchor[0]), int(anchor[1])),
                     (x + 4, y - th // 2), color, 1, cv2.LINE_AA)

    cv2.putText(img, txt, (x, y), font, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), font, scale, color,
                thickness, cv2.LINE_AA)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return np.mean(image, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    raise ValueError(f"Unsupported image shape for grayscale: {image.shape}")


# ─────────────────────────────────────────────────────────────────────
# Pure helpers for the pool — must be top-level so they pickle.
# These are the per-frame functions the pool workers call.
# ─────────────────────────────────────────────────────────────────────

def _bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Integer-factor downsample by block averaging (true bin, not nearest).
    factor<=1 is a passthrough. Trims the frame to a multiple of `factor`
    on each axis so the reshape doesn't fail on odd-sized sensors.

    Done inside the worker so the big full-size buffer is dropped before
    the frame travels back to the main process — IPC payload drops by
    factor^2 along with the in-memory footprint.
    """
    f = int(max(1, factor))
    if f == 1 or img is None:
        return img
    a = np.asarray(img)
    if a.ndim == 2:
        h, w = a.shape
        h2 = (h // f) * f
        w2 = (w // f) * f
        a = a[:h2, :w2]
        return a.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3)).astype(np.float32, copy=False)
    if a.ndim == 3:
        h, w, c = a.shape
        h2 = (h // f) * f
        w2 = (w // f) * f
        a = a[:h2, :w2, :]
        return a.reshape(h2 // f, f, w2 // f, f, c).mean(axis=(1, 3)).astype(np.float32, copy=False)
    return a.astype(np.float32, copy=False)


def _worker_load_and_cosmetic(args):
    """
    Worker: load one image and optionally bin it. Returns
    (path, img_or_None, obs_tuple, err_or_None).

    Cosmetic correction is deliberately NOT done here. It is internally
    parallel (GPU, then all-core Numba), so running it inside each of N
    pool workers oversubscribes the whole machine and chokes the system.
    It is applied once, serially, AFTER this load+bin pass completes (see
    the pipeline's cosmetic stage) — which also means it runs on the
    BINNED frame. Binning already averages away most hot/cold pixels, so
    the serial pass only has to mop up the few that survive, on a buffer
    factor^2 smaller and fast to process one at a time.

    Takes a single tuple arg so the pool can pickle the callable directly
    (a lambda wrapper around a multi-arg function pickles as a
    <locals>.<lambda> reference, which the spawn-based pool can't resolve
    inside the worker).
    """
    try:
        path, bin_factor = args

        # Read this frame's OWN observation time while the file is open.
        # Every search frame was taken at a different moment and a minor
        # body has moved between them, so each frame needs its own epoch.
        # Costs nothing here — the file is already being decoded.
        meta = None
        header = None
        try:
            res = load_image(path, return_metadata=True)
            if res and len(res) >= 5:
                img, header, bit_depth, is_mono, meta = res
            elif res:
                img, header, bit_depth, is_mono = res[:4]
            else:
                img = None
        except TypeError:
            img, header, bit_depth, is_mono = load_image(path)

        if img is None:
            return (path, None, (None, 0.0, "load failed", False),
                    f"load_image returned None for {os.path.basename(path)}")

        hdr_for_time = (meta.get("fits_header", header)
                        if isinstance(meta, dict) else header)
        try:
            t0, exptime, tsrc, utc_ok = _extract_obs_time(
                hdr_for_time, meta if isinstance(meta, dict) else None)
            obs = (str(t0.utc.isot) if t0 is not None else None,
                   float(exptime), str(tsrc), bool(utc_ok))
        except Exception as e:
            obs = (None, 0.0, f"time parse failed: {e}", False)

        img = np.asarray(img, dtype=np.float32)
        if bin_factor and int(bin_factor) > 1:
            img = _bin_image(img, int(bin_factor))
        return (path, img, obs, None)
    except Exception as e:
        try:
            base = os.path.basename(args[0]) if args else "?"
        except Exception:
            base = "?"
        return (args[0] if args else "?", None, (None, 0.0, "error", False),
                f"{base}: {e}")


def _worker_adbe(args):
    """
    Worker: run the modern abe_run() on an image. args tuple keeps the
    pool call signature narrow.

    `ignore_nodata=True` tells abe_run to auto-detect the zero-valued or
    non-finite border wedge that a registered / rotated frame leaves at
    the edges. Without that, the polynomial fit happily samples pixels
    inside the wedge (they ARE the dimmest spots in the frame) and tries
    to interpolate the whole image down to 0.0 along the wedge, which
    yanks the real sky toward 0 at the frame edge nearest the wedge.
    Essential whenever ADBE runs on aligned frames.
    """
    try:
        tag, img, degree, use_rbf, downsample = args
        if img is None:
            return (tag, None, "no image")
        a = np.asarray(img, dtype=np.float32)
        corrected = abe_run(
            a,
            degree=int(degree),
            num_samples=120,
            downsample=int(max(2, downsample)),
            patch_size=15,
            use_rbf=bool(use_rbf),
            rbf_smooth=1.0,
            exclusion_mask=None,
            return_background=False,
            legacy_prestretch=True,
            correction_mode="subtract",
            ignore_nodata=True,
            nodata_pad_px=4,
        )
        return (tag, corrected.astype(np.float32, copy=False), None)
    except Exception as e:
        return (tag, None, f"ABE failed for {tag}: {e}")


def _worker_align(args):
    """
    Worker: star-align one search frame to the reference using astroalign,
    then warp with the shared anti-ring warp. Returns
    (path, aligned_or_None, err_or_None).
    """
    try:
        path, search_img, ref_gray = args
        if search_img is None or ref_gray is None:
            return (path, None, "missing input")
        src_gray = _to_grayscale(search_img).astype(np.float32)
        src_gray = np.ascontiguousarray(src_gray)
        ref = np.ascontiguousarray(ref_gray.astype(np.float32))
        try:
            transform_obj, _pts = aa_find_transform_with_backoff(src_gray, ref)
        except Exception as e:
            return (path, None, f"astroalign failed: {e}")
        M = np.array(transform_obj.params[0:2, :], dtype=np.float64)
        H, W = ref.shape[:2]
        aligned = _warp_like_ref(search_img, M, (H, W))
        return (path, aligned.astype(np.float32, copy=False), None)
    except Exception as e:
        return (args[0] if args else "?", None, f"align: {e}")


# ─────────────────────────────────────────────────────────────────────
# Photometric matching — modelled on the continuum-subtraction path
# ─────────────────────────────────────────────────────────────────────
#
# WHY THIS WAS REWRITTEN (two separate bugs, both fatal for faint work):
#
#   v1 matched frames with
#       scaled = (search - med_s) * (mad_r / mad_s) + med_r
#   and ran that on PRE-STRETCHED data.
#
#   Bug 1 — MAD is a NOISE statistic, not a FLUX statistic.
#     mad_r/mad_s is the ratio of the two frames' sky+read noise, which
#     has nothing to do with their relative throughput. If transparency
#     dropped 10% between frames, stellar flux scales by 0.90 while the
#     noise ratio might be 1.03. Every star then subtracts with a ~13%
#     residual. A 16-mag object is about that size, so it drowns in the
#     residuals of its own neighbours.
#
#   Bug 2 — the stretch, and this is the dominant one.
#     stretch_mono_image() is a midtones-transfer function whose
#     parameters come from each frame's OWN median. Two frames whose sky
#     differs by 1% get DIFFERENT nonlinear curves. Subtracting them
#     leaves a curvature residual that is largest near the background —
#     exactly where a faint asteroid lives. So lowering the sigma slider
#     never reached the noise floor; it reached the stretch-mismatch
#     floor, which sits far above it. That is precisely the symptom of
#     "by the time I lower sigma enough, random noise starts triggering".
#
#   The fix mirrors what continuum_subtract.py does for NB/continuum:
#     (a) background-neutralise both frames (additive pedestal),
#     (b) take the multiplicative gain from STARS — real flux sources
#         measured in both frames — not from noise statistics,
#     (c) do all of it on LINEAR data; stretch only for display.


def _sigma_clipped_stats(a, sigma: float = 3.0, iters: int = 5,
                         max_samples: int = 400_000):
    """Robust (median, sigma) via iterative MAD clipping, subsampled."""
    d = np.asarray(a, dtype=np.float32).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0, 1e-6
    if d.size > max_samples:
        d = d[::int(np.ceil(d.size / max_samples))]
    med, sig = float(np.median(d)), 1e-6
    for _ in range(iters):
        med = float(np.median(d))
        sig = float(np.median(np.abs(d - med)) * 1.4826)
        if sig <= 0:
            sig = 1e-6
            break
        keep = np.abs(d - med) <= sigma * sig
        if keep.all() or int(keep.sum()) < 32:
            break
        d = d[keep]
    return med, max(sig, 1e-6)


def _find_photometry_sources(ref, bg, noise, *, n_sigma=15.0,
                             max_sources=400, min_area=4, max_area=2000,
                             edge=12):
    """
    Locate bright, compact, unsaturated sources in the REFERENCE frame.
    These become the photometric tie points. Deliberately conservative:
    a handful of clean stars gives a better gain than hundreds of
    marginal ones.
    """
    m = (ref > (bg + n_sigma * noise)).astype(np.uint8)
    if int(m.sum()) == 0:
        return []
    num, _lbl, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = ref.shape[:2]
    out = []
    for i in range(1, num):
        x, y, w_, h_, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if w_ > 40 or h_ > 40:          # galaxies / big blobs aren't tie points
            continue
        cx, cy = cents[i]
        if not (edge <= cx < W - edge and edge <= cy < H - edge):
            continue
        out.append((float(cx), float(cy), float(area)))
    out.sort(key=lambda t: -t[2])       # area ∝ brightness at fixed PSF
    return out[:max_sources]


def _aperture_flux(img, cx, cy, bg, r=5):
    """Background-subtracted square-aperture sum."""
    H, W = img.shape[:2]
    x0 = int(max(0, round(cx) - r)); x1 = int(min(W, round(cx) + r + 1))
    y0 = int(max(0, round(cy) - r)); y1 = int(min(H, round(cy) + r + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.sum(img[y0:y1, x0:x1] - bg))


def _photometric_match(search, ref, *, aperture=5, min_stars=5):
    """
    Put `search` onto `ref`'s photometric system.

        matched = (search - bg_search) * gain + bg_ref

    `gain` is the sigma-clipped median of per-star flux ratios
    (f_ref / f_search). That is a true throughput ratio — it tracks
    transparency, extinction and exposure differences, and is blind to
    the frames' noise levels, unlike the MAD ratio v1 used.

    Returns (matched, info). `info` carries the gain, both backgrounds,
    the star count actually used and which method fired, so a bad match
    shows up in the log instead of silently poisoning the detections.
    """
    bg_r, nz_r = _sigma_clipped_stats(ref)
    bg_s, nz_s = _sigma_clipped_stats(search)

    info = {"bg_ref": bg_r, "bg_search": bg_s,
            "noise_ref": nz_r, "noise_search": nz_s,
            "gain": 1.0, "n_stars": 0, "method": "none"}

    srcs = _find_photometry_sources(ref, bg_r, nz_r)
    npix = float((2 * aperture + 1) ** 2)
    ratios = []
    for cx, cy, _a in srcs:
        f_r = _aperture_flux(ref, cx, cy, bg_r, r=aperture)
        f_s = _aperture_flux(search, cx, cy, bg_s, r=aperture)
        # Require real signal in BOTH frames, so a star that landed in
        # the alignment wedge (f_s ≈ 0) can't produce a wild ratio.
        if f_r <= 0.0 or f_s <= 0.0:
            continue
        if f_r < 10.0 * nz_r * np.sqrt(npix):
            continue
        if f_s < 10.0 * nz_s * np.sqrt(npix):
            continue
        ratios.append(f_r / f_s)

    if len(ratios) >= min_stars:
        r = np.asarray(ratios, dtype=np.float64)
        for _ in range(2):              # clip blends / cosmic-ray hits
            m = float(np.median(r))
            s = float(np.median(np.abs(r - m)) * 1.4826)
            if s <= 0:
                break
            keep = np.abs(r - m) <= 3.0 * s
            if keep.all() or int(keep.sum()) < min_stars:
                break
            r = r[keep]
        gain = float(np.median(r))
        if not np.isfinite(gain) or gain <= 0:
            gain, info["method"] = 1.0, "fallback-unity"
        else:
            info["method"] = "stars"
        info["n_stars"] = int(r.size)
    else:
        # Sparse field, heavy cloud or a bad alignment. Fall back to the
        # old noise-ratio gain — better than nothing, and the log says so.
        gain = float(nz_r / max(nz_s, 1e-9))
        info["method"] = "noise-ratio-fallback"
        info["n_stars"] = len(ratios)

    info["gain"] = float(gain)
    matched = (search - bg_s) * gain + bg_r
    return matched.astype(np.float32, copy=False), info


def _worker_make_diff(args):
    """
    STAGE A (expensive, cached): photometric-match one search frame to the
    reference and return the LINEAR difference plus a validity mask and
    the measured difference noise.

    Detection deliberately does NOT happen here — see _detect_on_diff.
    That split is what lets the sigma slider re-run in well under a
    second instead of re-driving the whole pipeline.

    Returns (path, diff, valid_mask, sigma, info, err)
    """
    try:
        path, search_img, ref_gray, aperture = args
        if search_img is None or ref_gray is None:
            return (path, None, None, 0.0, {}, "missing input")

        search = _to_grayscale(search_img).astype(np.float32)
        ref = np.asarray(ref_gray, dtype=np.float32)

        if search.shape != ref.shape:
            return (path, None, None, 0.0, {},
                    f"shape mismatch (search={search.shape} vs ref={ref.shape}) — "
                    "enable 'Align search images to reference'")

        matched, info = _photometric_match(search, ref, aperture=int(aperture))
        diff = matched - ref

        # Validity. Where either frame has no data — the zero wedge a
        # rotated registration leaves, or a NaN — the difference is
        # meaningless. Erode so the resampling fringe at the wedge edge
        # goes too. Excluded from BOTH the noise estimate and detection:
        # leaving the wedge in was inflating sigma and desensitising the
        # entire frame.
        valid = (np.isfinite(diff) & np.isfinite(search) & np.isfinite(ref)
                 & (np.abs(search) > 1e-7) & (np.abs(ref) > 1e-7))
        if not valid.all():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            valid = cv2.erode(valid.astype(np.uint8), k).astype(bool)

        diff = np.where(valid, diff, 0.0).astype(np.float32, copy=False)

        # Noise of the DIFFERENCE, on valid pixels only, sigma-clipped so
        # stars and genuine transients don't inflate the threshold.
        _, sigma = _sigma_clipped_stats(diff[valid] if valid.any() else diff)
        info["sigma_diff"] = float(sigma)

        return (path, diff, valid, float(sigma), info, None)
    except Exception as e:
        return (args[0] if args else "?", None, None, 0.0, {}, f"diff: {e}")


# Connectivity dilation is a fixed internal detail, not a user knob. A
# source that threshold-fragments into a couple of islands should be
# labelled once; 1px handles that without inflating anything. (The
# user-facing "Grow" control was repurposed to dilate the STAR MASK —
# see star_grow_px below — which is what people actually reach for when
# shifting star halos throw false positives.)
_CONN_DILATE_PX = 1


def _detect_on_diff(diff, valid, sigma, star_mask, *, k_sigma, min_area,
                    max_dim, border_frac, star_grow_px=4, psf_sigma=1.5):
    """
    STAGE B (cheap, re-runnable): find point-source detections in a
    cached difference image. A pure function of the cached arrays plus
    the detection knobs, so re-tuning costs tens of milliseconds per
    frame instead of a full pipeline re-run.

    `star_grow_px` dilates the reference star mask before it is
    subtracted from the threshold mask. This is the knob to turn up when
    a star's halo or diffraction spikes differ slightly frame-to-frame
    and light up the difference in a ring around every bright star: a
    wider mask swallows that ring. The star mask handed in here is the
    bare threshold (undilated), so this value IS the full margin the
    mask extends past each detected star — 0 masks only the star cores,
    larger values reach further into the halo. Trade-off: too wide and
    the mask starts covering real objects that pass near a field star,
    so grow only as far as the halo residuals actually reach.

    MATCHED FILTERING — why this isn't a plain threshold any more:

      A faint point source is NOT best found by thresholding individual
      pixels. A 16-mag asteroid whose PSF peak sits at ~3σ clears a 3σ
      cut on only one or two pixels; those then die to the min-area
      filter that exists to kill noise spikes. So the object is
      simultaneously too faint to pass on area and too bright to ignore.

      Convolving the difference with a Gaussian matched to the PSF is
      the optimal linear detector for a point source. It attenuates the
      source peak by ~2x but suppresses pixel-to-pixel noise by
      ~2·σ_psf·√π, a net SNR gain around 3x for typical seeing — which
      is the difference between "invisible" and "obvious". It also
      suppresses single-pixel cosmic rays for free, since they don't
      match the PSF.

      Set psf_sigma=0 to disable and threshold the raw difference.
    """
    if diff is None:
        return []
    h, w = diff.shape[:2]

    if psf_sigma and psf_sigma > 0:
        ks = int(2 * max(1, int(round(3.0 * psf_sigma))) + 1)
        g = cv2.getGaussianKernel(ks, float(psf_sigma))
        det = cv2.sepFilter2D(np.asarray(diff, dtype=np.float32), cv2.CV_32F,
                              g, g, borderType=cv2.BORDER_REPLICATE)
    else:
        det = np.asarray(diff, dtype=np.float32)

    # The noise MUST be re-measured on the detection image. Filtering
    # changes it by sqrt(sum(kernel^2)) — roughly 6x smaller for a
    # σ=1.5 kernel — so reusing the raw-difference sigma would set the
    # threshold about six times too high and find nothing.
    has_valid = (valid is not None
                 and getattr(valid, "shape", None) == det.shape
                 and bool(valid.any()))
    med_d, sig_d = _sigma_clipped_stats(det[valid] if has_valid else det)

    raw = det > (med_d + float(k_sigma) * sig_d)

    if has_valid:
        raw &= valid
    if star_mask is not None and getattr(star_mask, "shape", None) == raw.shape:
        sm = star_mask
        if star_grow_px and star_grow_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * int(star_grow_px) + 1, 2 * int(star_grow_px) + 1))
            sm = cv2.dilate(star_mask.astype(np.uint8), k).astype(bool)
        raw &= ~sm

    bx, by = int(border_frac * w), int(border_frac * h)
    if bx > 0:
        raw[:, :bx] = False
        raw[:, w - bx:] = False
    if by > 0:
        raw[:by, :] = False
        raw[h - by:, :] = False

    if not raw.any():
        return []

    # Dilate for CONNECTIVITY only — so a source that fragments into a
    # couple of islands is labelled once. Fixed at _CONN_DILATE_PX; this
    # is no longer the user-facing "Grow" (that now dilates the star
    # mask above). Area is still counted on the undilated mask below, so
    # `min_area` keeps meaning "significant pixels" rather than "pixels
    # after I inflated them".
    raw_u8 = raw.astype(np.uint8)
    if _CONN_DILATE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * _CONN_DILATE_PX + 1, 2 * _CONN_DILATE_PX + 1))
        conn = cv2.dilate(raw_u8, k)
    else:
        conn = raw_u8

    num, lbl, stats, cents = cv2.connectedComponentsWithStats(conn, connectivity=8)
    if num <= 1:
        return []

    # True significant-pixel count per label, measured pre-dilation.
    raw_counts = np.bincount(lbl[raw].ravel(), minlength=num)

    out = []
    for i in range(1, num):
        x, y, w_, h_, _dil_area = stats[i]
        area = int(raw_counts[i])
        if area < min_area or w_ > max_dim or h_ > max_dim:
            continue
        y0, y1 = max(0, y), min(h, y + h_)
        x0, x1 = max(0, x), min(w, x + w_)
        fpatch = det[y0:y1, x0:x1]
        rpatch = diff[y0:y1, x0:x1]
        peak_f = float(np.max(fpatch)) if fpatch.size else 0.0
        peak_r = float(np.max(rpatch)) if rpatch.size else 0.0
        cx, cy = cents[i]
        out.append({
            "minX": int(x), "minY": int(y),
            "maxX": int(x + w_ - 1), "maxY": int(y + h_ - 1),
            "area": area,
            "cx": float(cx), "cy": float(cy),
            # Significance in the matched-filtered image — this is the
            # number that actually ranks detections.
            "peak_sigma": float(peak_f / max(sig_d, 1e-12)),
            # Raw-difference peak, in units of the raw difference noise.
            "raw_sigma": float(peak_r / max(float(sigma), 1e-12)),
        })
    # Strongest first: the real object usually sits at or near the top
    # even when the frame also picked up a pile of marginal noise.
    out.sort(key=lambda a: -a["peak_sigma"])
    return out


# ─────────────────────────────────────────────────────────────────────
# Star mask (reference-derived) — used to kill alignment residuals
# ─────────────────────────────────────────────────────────────────────

def _build_star_mask(ref_gray: np.ndarray,
                     k_sigma: float = 8.0,
                     dilate_px: int = 6) -> np.ndarray:
    """
    Threshold the LINEAR reference at median + k·sigma (sigma-clipped),
    optionally dilating by `dilate_px`.

    The pipeline now builds this with dilate_px=0 — the bare threshold —
    and lets the detection stage own the dilation via its `star_grow_px`
    knob (the UI "Star mask grow"). That way the margin around each star
    is re-tunable per re-detect without rebuilding the mask: turn it up
    when shifting star halos throw a ring of false positives, down when
    the mask is eating objects near field stars. `dilate_px` is kept for
    any caller that wants a pre-dilated mask.
    """
    g = np.asarray(ref_gray, dtype=np.float32)
    med, sig = _sigma_clipped_stats(g)
    hot = (g > med + k_sigma * sig).astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        hot = cv2.dilate(hot, k)
    return hot.astype(bool)


# ─────────────────────────────────────────────────────────────────────
# Display frames — 8-bit, for the results viewer ONLY
# ─────────────────────────────────────────────────────────────────────

def _display_u8(image: np.ndarray) -> np.ndarray:
    """
    Build a compact 8-bit display frame.

    DISPLAY ONLY — detection never touches this. Keeping the stretched
    copy strictly out of the maths is the core correctness fix in this
    revision; keeping it as uint8 mono (1 byte/px) instead of float32
    RGB (12 bytes/px) is what holds a 120-frame results set to a few
    hundred MB instead of tens of GB.
    """
    a = np.asarray(image)
    if a.ndim == 3 and a.shape[2] == 3:
        a = np.mean(a, axis=2)
    a = np.asarray(a, dtype=np.float32).squeeze()
    try:
        if stretch_mono_image is not None:
            a = stretch_mono_image(a, target_median=0.25, normalize=False,
                                   apply_curves=False, curves_boost=0.0)
    except Exception:
        pass
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = 0.0, 1.0
    return (np.clip((a - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def _worker_display(args):
    """Worker: build one uint8 display frame (parallel, cheap)."""
    try:
        path, img = args
        if img is None:
            return (path, None, "no image")
        return (path, _display_u8(img), None)
    except Exception as e:
        return (args[0] if args else "?", None, str(e))


# ─────────────────────────────────────────────────────────────────────
# Pipeline thread — orchestrates the pool-driven passes
# ─────────────────────────────────────────────────────────────────────

class HunterPipelineThread(QThread):
    """
    Runs the full pipeline off the GUI thread. Signals:

      progress(int done, int total, str msg)  — progress bar + status
      log(str)                                — append to log view
      finished_ok(dict payload)               — success
      finished_err(str message)               — fatal
    """

    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(object)
    finished_err = pyqtSignal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params
        self._cancel = threading.Event()

    def request_cancel(self):
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # ---- stage helpers ----------------------------------------------------

    def _emit(self, done, total, msg):
        self.progress.emit(int(done), int(total), str(msg))
        self.log.emit(msg)

    def _run_pool(self, worker_fn, items, label, workers, n_fields=3):
        """
        Drive one pass across a list of items through a process pool. Yields
        results as they come in. Honours cancellation between results.

        Every result is a tuple of width `n_fields` whose FIRST element is
        the item's path/tag and whose LAST element is an error string (or
        None) — the shape each caller unpacks. `n_fields` must match that
        caller (4 for load, 6 for the diff pass, 3 otherwise) so the
        crash-fallback below can synthesise a correctly shaped result.

        Why this matters: when a worker process dies — a segfault or an
        OOM kill in any one task marks the whole ProcessPoolExecutor
        broken, so EVERY pending fut.result() then raises
        BrokenProcessPool — the fallback must still yield something the
        caller can unpack. The old fallback used the whole input ITEM as
        the first element; for the align/adbe/diff passes that item is a
        tuple containing numpy arrays, so using it as a dict key blew up
        with "unhashable type: 'numpy.ndarray'" and masked the real
        failure. Use the item's path (its first element) instead.
        """
        def _err_result(it, msg):
            tag = it[0] if isinstance(it, (tuple, list)) and len(it) else it
            return (tag,) + (None,) * (n_fields - 2) + (msg,)

        total = len(items)
        done = 0
        self._emit(done, total, f"{label}: starting ({total} tasks, {workers} workers)")
        with _make_executor(workers) as ex:
            futs = {ex.submit(worker_fn, it): it for it in items}
            for fut in as_completed(futs):
                if self.cancelled:
                    for f in futs:
                        f.cancel()
                    self._emit(done, total, f"{label}: cancelled")
                    return
                try:
                    res = fut.result()
                except Exception as e:
                    res = _err_result(futs[fut], f"worker crashed: {e}")
                done += 1
                self._emit(done, total, f"{label}: {done}/{total}")
                yield res

    # ---- the pipeline -----------------------------------------------------

    def run(self):
        try:
            p = self._params
            ref_path = p["ref_path"]
            search_paths = list(p["search_paths"])
            do_cosmetic = bool(p["do_cosmetic"])
            do_align = bool(p["do_align"])
            do_adbe = bool(p["do_adbe"])
            adbe_degree = int(p["adbe_degree"])
            adbe_rbf = bool(p["adbe_rbf"])
            k_sigma = float(p["k_sigma"])
            min_area = int(p["min_area"])
            max_dim = int(p["max_dim"])
            border_frac = float(p["border_frac"])
            workers = int(p["workers"])
            bin_factor = max(1, int(p.get("bin_factor", 1)))
            # Photometry aperture radius, in BINNED pixels. A typical
            # star is ~4-6px FWHM at native scale, so scale the aperture
            # with the bin factor rather than leaving it fixed — an
            # oversized aperture at bin 4 would swallow neighbours and
            # bias the flux ratio.
            aperture = max(3, int(round(6.0 / bin_factor)))
            psf_sigma = float(p.get("psf_sigma", 1.5))

            if not ref_path or not search_paths:
                self.finished_err.emit("No reference or search images provided.")
                return

            all_paths = [ref_path] + search_paths
            n_all = len(all_paths)

            # ── Stage 1: load + optional bin (parallel across the pool)
            # Pack (path, bin_factor) into a tuple so the worker is a plain
            # top-level function and the process pool can pickle it.
            # Wrapping in a lambda here fails under spawn-start pools with
            # "Can't get local object '…<lambda>'".
            #
            # Binning is applied INSIDE the worker so the full-res buffer
            # never travels back across the pool boundary — IPC payload
            # drops by factor^2 along with the in-memory footprint. For a
            # 45MP frame at bin=2 that is ~45MB → ~11MB per frame; over
            # 120 frames that is a ~4GB saving in flight on top of the 4×
            # drop in peak RAM once the pipeline is holding them all.
            #
            # Cosmetic correction is NOT done in this pool — it runs once,
            # serially, after this pass (see below).
            if bin_factor > 1:
                self._emit(0, len(all_paths),
                           f"Load: binning {bin_factor}× on load "
                           f"(~{bin_factor * bin_factor}× less memory)")
            loaded = {}
            frame_obs = {}
            load_items = [(p, bin_factor) for p in all_paths]
            for path, img, obs, err in self._run_pool(
                _worker_load_and_cosmetic, load_items, "Load", workers,
                n_fields=4,
            ):
                if err:
                    self.log.emit(f"⚠ {err}")
                loaded[path] = img
                frame_obs[path] = obs or (None, 0.0, "unknown", False)
                if self.cancelled:
                    return

            _n_timed = sum(1 for v in frame_obs.values() if v and v[0])
            self.log.emit(f"   observation times read for "
                          f"{_n_timed}/{len(all_paths)} frames")

            if loaded.get(ref_path) is None:
                self.finished_err.emit("Failed to load reference image.")
                return

            # ── Stage 1b: cosmetic correction — SERIAL, post-bin ──────────
            # Run AFTER binning and one frame at a time, on purpose:
            #   • bulk_cosmetic_correction_numba is already heavily parallel
            #     internally (GPU, then all-core Numba). Fanning it out
            #     across N pool workers oversubscribes every core/the GPU
            #     and chokes the whole machine, so it runs serially here.
            #   • binning already block-averages away most hot/cold pixels;
            #     this pass just mops up the few survivors, on a buffer
            #     factor^2 smaller — cheap even one frame at a time.
            # A failure on any frame is non-fatal: log it and keep that
            # frame's (binned, uncorrected) data so the run still proceeds.
            if do_cosmetic:
                paths_to_fix = [pth for pth in all_paths
                                if loaded.get(pth) is not None]
                total_cc = len(paths_to_fix)
                for i_cc, pth in enumerate(paths_to_fix, start=1):
                    if self.cancelled:
                        return
                    self._emit(i_cc, total_cc,
                               f"Cosmetic correction (post-bin): {i_cc}/{total_cc}")
                    try:
                        loaded[pth] = np.asarray(
                            bulk_cosmetic_correction_numba(
                                loaded[pth], hot_sigma=5.0, cold_sigma=5.0),
                            dtype=np.float32)
                    except Exception as ce:
                        self.log.emit(
                            f"⚠ {os.path.basename(pth)}: cosmetic skipped: {ce}")

            # Reference metadata on the GUI thread would be nicer, but we
            # need JD / WCS back on the dialog for minor-body work. Do it
            # here on the pipeline thread — fast, no GUI touches.
            #
            # When binning is active, the WCS CRPIX / CDELT / CD cards have
            # to be rescaled so pixel (x, y) in the BINNED frame still maps
            # to the same (RA, Dec). Otherwise the minor-body cross-match
            # (which uses ref_wcs.world_to_pixel_values on the preview
            # image's shape) lands on the wrong spot.
            ref_header = None
            ref_wcs = None
            ref_jd = None
            ref_obs_iso = None
            ref_obs_exptime = 0.0
            ref_obs_source = "not read"
            try:
                ref_res = load_image(ref_path, return_metadata=True)
                if ref_res and len(ref_res) >= 5:
                    _, header, _, _, meta = ref_res
                    ref_header = (meta.get("fits_header", header)
                                  if isinstance(meta, dict) else header)

                    # Observation time first — it is independent of binning.
                    (t0, ref_obs_exptime, ref_obs_source,
                     _ref_utc_ok) = _extract_obs_time(
                        ref_header, meta if isinstance(meta, dict) else None)
                    if t0 is not None:
                        ref_obs_iso = str(t0.utc.isot)
                        ref_jd = float((t0 + (ref_obs_exptime / 2.0) * u.s).utc.jd)
                    self.log.emit(
                        f"   observation time: {ref_obs_iso or 'UNKNOWN'}"
                        f"  exp={ref_obs_exptime:g}s  source={ref_obs_source}")

                    # The WCS must describe the BINNED grid, because that is
                    # the grid the anomaly pixel coordinates live in.
                    if bin_factor > 1 and isinstance(ref_header, Header):
                        try:
                            ref_header = _rescale_wcs_for_bin(ref_header, bin_factor)
                        except Exception as e:
                            self.log.emit(f"⚠ WCS bin rescale failed: {e}")
                    try:
                        ref_wcs = WCS(ref_header)
                    except Exception:
                        ref_wcs = None

                    ok_wcs, why = _wcs_is_usable(ref_wcs, ref_header)
                    self.log.emit(f"   reference WCS: "
                                  f"{'usable, ' + why if ok_wcs else 'NOT usable — ' + why}")
                    if not ok_wcs:
                        # Drop it. A default/unsolved WCS is worse than none:
                        # it looks valid and silently points at RA=0 Dec=0.
                        ref_wcs = None
            except Exception as e:
                self.log.emit(f"⚠ metadata read failed: {e}")

            # ── Stage 2: optional star alignment of search frames ─────────
            if do_align:
                if self.cancelled:
                    return
                ref_gray = _to_grayscale(loaded[ref_path])
                align_items = [
                    (p, loaded.get(p), ref_gray) for p in search_paths
                ]
                aligned = {}
                for path, img, err in self._run_pool(
                    _worker_align, align_items, "Star alignment", workers,
                ):
                    if err:
                        self.log.emit(f"⚠ align: {err}")
                    aligned[path] = img
                    if self.cancelled:
                        return
                # Replace with aligned versions (keep ref as-is).
                for sp in search_paths:
                    if aligned.get(sp) is not None:
                        loaded[sp] = aligned[sp]

            # ── Stage 3: optional ADBE (parallel across all N+1 frames) ───
            if do_adbe:
                if self.cancelled:
                    return
                adbe_items = [
                    (path, loaded.get(path), adbe_degree, adbe_rbf, 4)
                    for path in all_paths
                ]
                for path, img, err in self._run_pool(
                    _worker_adbe, adbe_items, "ADBE background removal", workers,
                ):
                    if err:
                        self.log.emit(f"⚠ ADBE: {err}")
                    if img is not None:
                        loaded[path] = img
                    if self.cancelled:
                        return

            # ── Display frames (uint8, parallel) ─────────────────────────
            # These feed the results viewer ONLY. Detection runs on the
            # linear data below. Keeping the two apart is the central fix
            # of this revision — a per-frame nonlinear stretch is derived
            # from that frame's own statistics, so differencing two
            # independently-stretched frames leaves a curvature residual
            # that buries anything faint.
            if self.cancelled:
                return
            display = {}
            for path, u8, err in self._run_pool(
                _worker_display, [(p, loaded.get(p)) for p in all_paths],
                "Building display frames", workers,
            ):
                if err:
                    self.log.emit(f"⚠ display: {err}")
                display[path] = u8
                if self.cancelled:
                    return

            ref_lin = loaded.get(ref_path)
            if ref_lin is None:
                self.finished_err.emit("Reference frame unavailable.")
                return
            ref_gray_lin = _to_grayscale(ref_lin).astype(np.float32)

            # Star mask from the LINEAR reference. Stored UNDILATED — the
            # bare threshold. The detection stage dilates it by the
            # "Star mask grow" knob (star_grow_px), so the margin around
            # each star is re-tunable per re-detect without rebuilding.
            star_mask = _build_star_mask(ref_gray_lin, k_sigma=10.0, dilate_px=0)

            # ── Stage 4a: photometric match + linear difference ──────────
            if self.cancelled:
                return
            self._emit(0, len(search_paths),
                       f"Photometric match: star-flux gain, aperture r={aperture}px")
            diff_items = [(sp, loaded.get(sp), ref_gray_lin, aperture)
                          for sp in search_paths]

            diffs, valids, sigmas, infos = {}, {}, {}, {}
            for path, diff, valid, sigma, info, err in self._run_pool(
                _worker_make_diff, diff_items,
                "Photometric match + difference", workers, n_fields=6,
            ):
                if err:
                    self.log.emit(f"⚠ {os.path.basename(str(path))}: {err}")
                diffs[path] = diff
                valids[path] = valid
                sigmas[path] = sigma
                infos[path] = info or {}
                if diff is not None and info:
                    self.log.emit(
                        f"   {os.path.basename(path)}: gain={info.get('gain', 1.0):.4f} "
                        f"({info.get('method', '?')}, {info.get('n_stars', 0)} stars) "
                        f"σ_diff={info.get('sigma_diff', 0.0):.3e}"
                    )
                if self.cancelled:
                    return

            # The linear frames have done their job — the differences
            # carry everything detection needs. Drop them before the
            # results dialog opens so peak RAM settles at roughly one
            # diff plus one uint8 display per frame.
            loaded.clear()

            # ── Stage 4b: detection (cheap, re-runnable) ─────────────────
            # star_grow_px defaults to 4, which reproduces the mask
            # coverage this stage used to hardcode (threshold dilated by
            # 4). The ResultsDialog exposes it as "Star mask grow".
            star_grow_px = int(p.get("star_grow_px", 4))
            det_params = {
                "k_sigma": k_sigma,
                "min_area": min_area,
                "max_dim": max_dim,
                "border_frac": border_frac,
                "use_star_mask": True,
                "star_grow_px": star_grow_px,
                "psf_sigma": psf_sigma,
            }
            anomaly_data = []
            for sp in search_paths:
                anoms = _detect_on_diff(
                    diffs.get(sp), valids.get(sp), sigmas.get(sp, 1.0),
                    star_mask,
                    k_sigma=k_sigma, min_area=min_area, max_dim=max_dim,
                    border_frac=border_frac, star_grow_px=star_grow_px,
                    psf_sigma=psf_sigma,
                )
                _obs = frame_obs.get(sp) or (None, 0.0, "unknown", False)
                anomaly_data.append({
                    "path": sp,
                    "imageName": os.path.basename(sp),
                    "obs_iso": _obs[0],
                    "obs_exptime": float(_obs[1] or 0.0),
                    "obs_source": _obs[2],
                    "obs_utc_confirmed": bool(_obs[3]),
                    "anomalyCount": len(anoms),
                    "anomalies": anoms,
                    "search_image": display.get(sp),
                    "diff_image": diffs.get(sp),
                    "valid_mask": valids.get(sp),
                    "sigma": sigmas.get(sp, 1.0),
                    "match_info": infos.get(sp, {}),
                })

            payload = {
                "ref_path": ref_path,
                "ref_image": display.get(ref_path),
                # The linear, binned reference is kept so a plate solve can
                # run on EXACTLY the pixel grid the anomaly coordinates are
                # measured in. Solving the full-res file from disk instead
                # would hand back a WCS one bin factor off, and every
                # predicted position would land at the wrong pixel.
                "ref_linear": ref_gray_lin,
                "ref_header": ref_header,
                "ref_wcs": ref_wcs,
                "ref_jd": ref_jd,
                "ref_obs_iso": ref_obs_iso,
                "ref_obs_exptime": ref_obs_exptime,
                "ref_obs_source": ref_obs_source,
                "bin_factor": bin_factor,
                "site_from_header": _site_from_header(ref_header),
                "star_mask": star_mask,
                "det_params": det_params,
                "anomaly_data": anomaly_data,
            }
            self.finished_ok.emit(payload)

        except Exception as e:
            tb = traceback.format_exc()
            self.finished_err.emit(f"Pipeline crashed: {e}\n{tb}")


# ─────────────────────────────────────────────────────────────────────
# Re-detection thread — Stage B only, against cached differences
# ─────────────────────────────────────────────────────────────────────

class RedetectThread(QThread):
    """
    Re-runs ONLY the detection stage against the cached difference
    images already sitting in the payload.

    No reload, no re-align, no ADBE, no re-match. For a 120-frame set
    this is typically under a second, versus minutes for a full run —
    which means the sigma knob becomes something you can actually sweep
    while watching the result, instead of a parameter you commit to and
    wait on.
    """

    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(object)

    def __init__(self, payload: dict, det_params: dict, parent=None):
        super().__init__(parent)
        self._payload = payload
        self._p = dict(det_params)

    def run(self):
        p = self._p
        star_mask = (self._payload.get("star_mask")
                     if p.get("use_star_mask", True) else None)
        data = self._payload.get("anomaly_data", [])
        total = len(data)
        for i, entry in enumerate(data):
            anoms = _detect_on_diff(
                entry.get("diff_image"), entry.get("valid_mask"),
                entry.get("sigma", 1.0), star_mask,
                k_sigma=p["k_sigma"], min_area=p["min_area"],
                max_dim=p["max_dim"], border_frac=p["border_frac"],
                star_grow_px=int(p.get("star_grow_px", p.get("grow_px", 4))),
                psf_sigma=float(p.get("psf_sigma", 1.5)),
            )
            entry["anomalies"] = anoms
            entry["anomalyCount"] = len(anoms)
            self.progress.emit(i + 1, total, f"Re-detect {i + 1}/{total}")
        self._payload["det_params"] = p
        self.done.emit(self._payload)


# ─────────────────────────────────────────────────────────────────────
# Minor-body worker (kept from the original — already threaded)
# ─────────────────────────────────────────────────────────────────────

class MinorBodyWorker(QObject):
    # object rather than list: the result carries both the bodies found at
    # the selected frame's epoch and a per-frame position table.
    finished = pyqtSignal(object, str)
    progress = pyqtSignal(int, str)

    def __init__(self, owner, jd_for_calc: float, frame_jds=None):
        super().__init__()
        self._owner = owner
        self._jd = jd_for_calc
        self._frame_jds = list(frame_jds or [])

    def run(self):
        try:
            self.progress.emit(0, "Minor-body search: preparing catalog query...")
            result = self._owner._get_predicted_minor_bodies_for_field(
                H_ast_max=self._owner.minor_H_ast_max,
                H_com_max=self._owner.minor_H_com_max,
                jd=self._jd,
                progress_cb=self.progress.emit,
                frame_jds=self._frame_jds,
            )
            if result is None:
                result = {"ref": [], "per_frame": {}}
            self.finished.emit(result, "")
        except Exception as e:
            self.finished.emit({"ref": [], "per_frame": {}}, str(e))


# ─────────────────────────────────────────────────────────────────────
# Zoomable preview widget (same shape as original)
# ─────────────────────────────────────────────────────────────────────

class ZoomableImageView(QGraphicsView):
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pix = QGraphicsPixmapItem()
        self.scene().addItem(self._pix)
        self.setRenderHints(QPainter.RenderHint.Antialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._fit_mode = False
        self._scale = 1.0

    def set_image(self, np_img, *, preserve_view: bool = False):
        """
        Load a new image into the view.

        `preserve_view=True` keeps the current transform (zoom) AND the
        viewport scroll position across the swap, provided the new
        pixmap matches the old one in size. This is what the results-
        dialog blink/play needs: zoom in on a corner, start play, and
        see the asteroid actually drift past that zoomed-in corner
        instead of the viewport snapping back to Fit on every frame.

        Falls back to a full reset when the sizes don't match (nothing
        else is safe — the previous scroll offset would be meaningless
        on a different-sized frame) or when the view is still untouched
        (initial load — Fit will run from the caller).
        """
        qimg = _numpy_to_qimage(np_img)
        pix = QPixmap.fromImage(qimg)
        old_pix = self._pix.pixmap()
        same_size = (not old_pix.isNull()
                     and old_pix.size() == pix.size())
        if preserve_view and same_size:
            # Capture scroll BEFORE swapping the pixmap — the swap can
            # nudge the scrollbars if the scene rect updates, and we
            # want the user's exact position back.
            hsb = self.horizontalScrollBar()
            vsb = self.verticalScrollBar()
            hx, vx = hsb.value(), vsb.value()
            tf = self.transform()
            self._pix.setPixmap(pix)
            self.scene().setSceneRect(QRectF(pix.rect()))
            # Restore transform + scroll in that order so Qt doesn't
            # recompute the scroll range against the identity transform.
            self.setTransform(tf)
            hsb.setValue(hx)
            vsb.setValue(vx)
            # Don't emit zoomChanged — the zoom didn't actually change,
            # and emitting here would make any listener (toolbars with
            # a zoom %) flash on every blink tick.
            return
        self._pix.setPixmap(pix)
        self.scene().setSceneRect(QRectF(pix.rect()))
        self.reset_view()

    def set_pixmap(self, pix: QPixmap):
        self._pix.setPixmap(pix)
        self.scene().setSceneRect(QRectF(pix.rect()))

    def reset_view(self):
        self._fit_mode = False
        self._scale = 1.0
        self.setTransform(QTransform())
        self.centerOn(self._pix)
        self.zoomChanged.emit(self._scale)

    def fit_to_view(self):
        if self._pix.pixmap().isNull():
            return
        self._fit_mode = True
        self.setTransform(QTransform())
        self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)
        self._scale = self.transform().m11()
        self.zoomChanged.emit(self._scale)

    def set_1to1(self):
        self._fit_mode = False
        self.setTransform(QTransform())
        self._scale = 1.0
        self.zoomChanged.emit(self._scale)

    def zoom(self, factor: float, anchor_pos: QPointF | None = None):
        if self._pix.pixmap().isNull():
            return
        self._fit_mode = False
        new_scale = max(0.05, min(32.0, self._scale * factor))
        factor = new_scale / self._scale
        if abs(factor - 1.0) < 1e-6:
            return
        if anchor_pos is not None:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scale(factor, factor)
        self._scale = new_scale
        self.zoomChanged.emit(self._scale)

    def wheelEvent(self, event):
        # Bare wheel zooms, no modifier required. The previous Ctrl+wheel
        # requirement comes from browser habit, but a results viewer is
        # closer to an image viewer: both hands can be elsewhere, and
        # scrolling a static picture has no useful meaning. Shift+wheel
        # still falls through to the default (horizontal scroll).
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        step = 1.25 if delta > 0 else 0.8
        self.zoom(step, anchor_pos=event.position())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().unsetCursor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode and not self._pix.pixmap().isNull():
            self.fit_to_view()


# ─────────────────────────────────────────────────────────────────────
# Results dialog — single viewer, tree-driven, time-sequence blink
# ─────────────────────────────────────────────────────────────────────
#
# Design rationale (v2 — tabs removed):
#
#   The previous version put one tab per flagged frame at the top of the
#   dialog. For a 120-frame run that meant 120 widgets instantiated at
#   dialog-open time, each carrying three overlaid uint8 images plus a
#   QGraphicsScene — tens of GB of RAM and a multi-second freeze while
#   the dialog built.
#
#   The new design is a single ZoomableImageView fed on demand:
#
#     • Tree at the bottom lists every search frame + its anomaly count.
#       Double-click → load that frame into the viewer.
#     • Blink/play plays the SEARCH FRAMES in sequence so an actual
#       asteroid drifts visibly across the field — that's the whole
#       reason an astronomer blinks in the first place.
#     • Ref / Diff radios let you flip the single viewer to the shared
#       reference or the current-frame difference instead, for post-hoc
#       inspection of a specific detection.
#     • Zoom In / Out / Fit buttons on the toolbar (missing from the
#       old UI, which forced Ctrl+wheel only).
#
#   Overlays (anomaly boxes + minor-body circles) are drawn lazily into
#   a per-frame uint8 cache the first time each frame is shown, then
#   kept in memory. On a 100-frame play loop that caps at ~100 × framesize
#   bytes once every frame has been visited at least once; worst-case
#   on a 45MP mono-stretched set is ~1-2 GB, not the ~30 GB the tabbed
#   version held permanently.

class ResultsDialog(QDialog):
    """
    Single-viewer results UI with time-sequence blink.

    pushed(image, title, source_path) — emitted when the user clicks
      "Push current frame to new view". Carries the overlay-stamped uint8
      image and the source FITS path for metadata recovery.
    """

    pushed = pyqtSignal(object, str, str)
    redetectRequested = pyqtSignal(object)   # det_params dict
    exploreRequested = pyqtSignal()          # open the 3D orbit explorer

    MODE_SEARCH = "search"
    MODE_REF = "reference"
    MODE_DIFF = "difference"

    def __init__(self, payload: dict, icon: QIcon | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Anomaly Results")
        if icon:
            self.setWindowIcon(icon)
        self.resize(1100, 820)
        self._payload = payload
        self._anoms = payload.get("anomaly_data", [])
        self._minor_bodies = payload.get("minor_bodies") or []
        self._bodies_per_frame = payload.get("minor_bodies_per_frame") or {}

        # Caches of overlay-stamped uint8 images, keyed by (mode, idx).
        # idx is the index into self._anoms; for the Reference mode the
        # idx is ignored because the ref image is shared across frames
        # (overlays reflect whichever search frame is currently selected).
        self._cache: dict[tuple[str, int], np.ndarray] = {}

        # Which search-frame index is currently shown + what mode.
        self._current_idx = 0
        self._mode = self.MODE_SEARCH
        self._show_labels = True

        # Playback.
        self._playing = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_search_frame)

        self._init_ui()
        if self._anoms:
            self._show(0, self.MODE_SEARCH, fit=True)

    # ---- UI construction -----------------------------------------------

    def _init_ui(self):
        lay = QVBoxLayout(self)

        # Summary line.
        total_anom = sum(d["anomalyCount"] for d in self._anoms)
        n_frames = len(self._anoms)
        self._summary_lbl = QLabel(
            f"Processed {n_frames} search frame(s). "
            f"Total candidate anomalies: {total_anom}"
        )
        self._summary_lbl.setStyleSheet("font-weight: bold; padding: 4px;")
        lay.addWidget(self._summary_lbl)

        # ── Viewer toolbar ───────────────────────────────────────────────
        tb = QHBoxLayout()

        # Zoom controls — the big ask from the previous rev.
        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_in.setToolTip("Zoom in on the viewer (Ctrl+wheel also works).")
        self.btn_zoom_in.clicked.connect(lambda: self.view.zoom(1.25))
        tb.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("Zoom −")
        self.btn_zoom_out.setToolTip("Zoom out on the viewer.")
        self.btn_zoom_out.clicked.connect(lambda: self.view.zoom(0.8))
        tb.addWidget(self.btn_zoom_out)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setToolTip("Fit the current frame to the viewport.")
        self.btn_fit.clicked.connect(lambda: self.view.fit_to_view())
        tb.addWidget(self.btn_fit)

        self.btn_1to1 = QPushButton("1:1")
        self.btn_1to1.setToolTip("Reset to 100% zoom.")
        self.btn_1to1.clicked.connect(lambda: self.view.set_1to1())
        tb.addWidget(self.btn_1to1)

        tb.addSpacing(16)

        # Mode toggles — tiny radio-like QPushButtons so you can inspect
        # a flagged frame as Search / Reference / Difference without the
        # blink timer flipping it away. Default is Search because that's
        # the frame the drift animation plays over.
        self.btn_mode_search = QPushButton("Search")
        self.btn_mode_search.setCheckable(True)
        self.btn_mode_search.setChecked(True)
        self.btn_mode_search.clicked.connect(
            lambda: self._set_mode(self.MODE_SEARCH))
        self.btn_mode_ref = QPushButton("Reference")
        self.btn_mode_ref.setCheckable(True)
        self.btn_mode_ref.clicked.connect(
            lambda: self._set_mode(self.MODE_REF))
        self.btn_mode_diff = QPushButton("Difference")
        self.btn_mode_diff.setCheckable(True)
        self.btn_mode_diff.clicked.connect(
            lambda: self._set_mode(self.MODE_DIFF))
        for b in (self.btn_mode_search, self.btn_mode_ref, self.btn_mode_diff):
            tb.addWidget(b)

        tb.addSpacing(12)
        self.chk_labels = QCheckBox("Labels")
        self.chk_labels.setChecked(True)
        self.chk_labels.setToolTip(
            "Print the catalog designation next to each identified object,\n"
            "and the peak significance next to unidentified candidates.\n"
            "Turn off in a crowded field where the text obscures the stars."
        )
        self.chk_labels.toggled.connect(self._on_labels_toggled)
        tb.addWidget(self.chk_labels)

        tb.addSpacing(16)

        # Playback. "Blink" here plays the SEARCH FRAMES in sequence,
        # which is what makes drifting objects (asteroids, satellites)
        # visibly move. The old ref/diff/search cycle is covered by the
        # three mode buttons to its left — orthogonal controls.
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setToolTip(
            "Play through the search frames in order. An asteroid will\n"
            "drift visibly across the field; a static anomaly won't."
        )
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._on_play_toggled)
        tb.addWidget(self.btn_play)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setToolTip("Previous search frame")
        self.btn_prev.clicked.connect(self._prev_frame)
        tb.addWidget(self.btn_prev)

        self.btn_next = QPushButton("▶|")
        self.btn_next.setToolTip("Next search frame")
        self.btn_next.clicked.connect(self._next_frame)
        tb.addWidget(self.btn_next)

        tb.addWidget(QLabel("Rate (ms):"))
        self.sp_rate = QSpinBox()
        self.sp_rate.setRange(50, 5000)
        self.sp_rate.setSingleStep(50)
        self.sp_rate.setValue(400)
        self.sp_rate.valueChanged.connect(
            lambda v: self._timer.setInterval(int(v)))
        tb.addWidget(self.sp_rate)

        tb.addStretch(1)

        self.btn_push = QPushButton("Push to New View")
        self.btn_push.setToolTip(
            "Create a new SASpro view from the frame currently showing in\n"
            "the viewer. The overlays (anomaly boxes, minor-body circles)\n"
            "come along with it."
        )
        self.btn_push.clicked.connect(self._on_push)
        tb.addWidget(self.btn_push)

        lay.addLayout(tb)

        # Status line directly under the toolbar — tells you which frame
        # you're on and what mode. Updates as play advances.
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color:#2c7;font-weight:bold;padding-left:4px;")
        lay.addWidget(self._status_lbl)

        # ── Detection controls (Stage B — re-runnable in place) ─────────
        # These live here, next to the results, because tuning the
        # threshold is something you do WHILE looking at what it found.
        # Re-detect re-thresholds the cached difference images; it never
        # reloads, re-aligns or re-runs ADBE.
        dp = self._payload.get("det_params", {}) or {}
        dr = QHBoxLayout()

        dr.addWidget(QLabel("Detection σ:"))
        self.sp_sigma = QDoubleSpinBox()
        self.sp_sigma.setRange(0.5, 30.0)
        self.sp_sigma.setSingleStep(0.25)
        self.sp_sigma.setDecimals(2)
        self.sp_sigma.setValue(float(dp.get("k_sigma", 5.0)))
        self.sp_sigma.setToolTip(
            "Threshold = σ × the measured noise of the DIFFERENCE image\n"
            "(sigma-clipped, computed per frame on valid pixels only).\n"
            "With star-flux photometric matching the noise floor is real,\n"
            "so 3-4σ is usable for faint objects."
        )
        dr.addWidget(self.sp_sigma)

        dr.addWidget(QLabel("Min area:"))
        self.sp_minarea = QSpinBox()
        self.sp_minarea.setRange(1, 2000)
        self.sp_minarea.setValue(int(dp.get("min_area", 5)))
        self.sp_minarea.setToolTip(
            "Minimum number of significant pixels, counted on the\n"
            "matched-filtered image BEFORE the connectivity dilation.\n"
            "5 is a good faint-object value; raise it to suppress\n"
            "marginal noise clumps."
        )
        dr.addWidget(self.sp_minarea)

        dr.addWidget(QLabel("PSF σ:"))
        self.sp_psf = QDoubleSpinBox()
        self.sp_psf.setRange(0.0, 6.0)
        self.sp_psf.setSingleStep(0.25)
        self.sp_psf.setDecimals(2)
        self.sp_psf.setValue(float(dp.get("psf_sigma", 1.5)))
        self.sp_psf.setToolTip(
            "Matched-filter width, in binned pixels. Set it near your\n"
            "actual star PSF sigma (≈ FWHM / 2.35).\n\n"
            "This is the single most important control for faint\n"
            "objects: filtering at the PSF scale buys roughly 3x in\n"
            "SNR over raw pixel thresholding. 0 disables the filter."
        )
        dr.addWidget(self.sp_psf)

        dr.addWidget(QLabel("Max dim:"))
        self.sp_maxdim = QSpinBox()
        self.sp_maxdim.setRange(10, 10000)
        self.sp_maxdim.setValue(int(dp.get("max_dim", 200)))
        dr.addWidget(self.sp_maxdim)

        dr.addWidget(QLabel("Star mask grow:"))
        self.sp_grow = QSpinBox()
        self.sp_grow.setRange(0, 50)
        self.sp_grow.setValue(int(dp.get("star_grow_px", dp.get("grow_px", 4))))
        self.sp_grow.setToolTip(
            "Grow the reference star mask by N px before it rejects\n"
            "detections. Turn this UP when a star's halo or diffraction\n"
            "spikes shift slightly between frames and light up a ring of\n"
            "false positives around every bright star — a wider mask\n"
            "swallows that ring.\n\n"
            "This is the full margin the mask extends past each star:\n"
            "0 masks only the star cores, 4 is the default, larger\n"
            "values reach further into the halo. Too wide and the mask\n"
            "starts covering real objects that pass near a field star,\n"
            "so grow only as far as the residuals actually reach.\n\n"
            "Only active while 'Reject at ref stars' is on."
        )
        dr.addWidget(self.sp_grow)

        self.chk_starmask = QCheckBox("Reject at ref stars")
        self.chk_starmask.setChecked(bool(dp.get("use_star_mask", True)))
        self.chk_starmask.setToolTip(
            "Suppress detections that land on a reference-frame star.\n"
            "Turn OFF when hunting a supernova inside a galaxy or an\n"
            "object passing close to a field star."
        )
        dr.addWidget(self.chk_starmask)

        self.btn_redetect = QPushButton("Re-detect")
        self.btn_redetect.setStyleSheet("font-weight: bold;")
        self.btn_redetect.setToolTip(
            "Re-threshold the cached difference images with these\n"
            "settings. Sub-second — no reload, no re-align, no ADBE."
        )
        self.btn_redetect.clicked.connect(self._on_redetect)
        dr.addWidget(self.btn_redetect)

        dr.addStretch(1)
        lay.addLayout(dr)

        # ── Viewer ───────────────────────────────────────────────────────
        self.view = ZoomableImageView(self)
        lay.addWidget(self.view, 1)

        # ── Summary tree ─────────────────────────────────────────────────
        # Double-click a top-level row to show that frame. Child rows are
        # per-anomaly bounding boxes; double-clicking a child shows the
        # parent frame too (and could be extended later to pan+zoom to
        # the box — out of scope for this rev).
        self._tree = QTreeWidget(self)
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(
            ["Image (double-click to show)", "Anomalies", "Peak σ", "Identified"])
        self._tree.setMaximumHeight(220)
        self._tree.itemDoubleClicked.connect(self._on_tree_double)
        self._rebuild_tree()
        lay.addWidget(self._tree)

        # ── Minor-body progress strip ───────────────────────────────────
        # Mirrors the status/progress on the parent, which is normally
        # hidden behind this window during a search. Without it a solve +
        # catalog query looks like a frozen UI, and the user ends up
        # clicking Re-detect or Search again. Hidden until a search runs.
        self._mb_status = QLabel("")
        self._mb_status.setWordWrap(True)
        self._mb_status.setStyleSheet(
            "color:#2c7; padding: 3px 4px 0px 4px; font-size: 11px;")
        self._mb_status.setVisible(False)
        lay.addWidget(self._mb_status)

        self._mb_progress = QProgressBar()
        self._mb_progress.setRange(0, 100)
        self._mb_progress.setValue(0)
        self._mb_progress.setTextVisible(True)
        self._mb_progress.setVisible(False)
        lay.addWidget(self._mb_progress)

        # ── Bottom button bar ───────────────────────────────────────────
        bb = QHBoxLayout()
        bb.addStretch(1)
        self.btn_minor = QPushButton("Catalogue Minor Bodies in Field")
        self.btn_minor.setToolTip(
            "Query the minor-body catalog using the frame's own WCS/JD and\n"
            "cross-match against detected anomalies."
        )
        bb.addWidget(self.btn_minor)
        self.btn_explore = QPushButton("Explore Orbit (3D)…")
        self.btn_explore.setToolTip(
            "Open an interactive 3D heliocentric view of a catalogued\n"
            "body's orbit — Sun, planets, and the asteroid placed at this\n"
            "frame's exact observation time. Run the catalog search first."
        )
        self.btn_explore.setEnabled(bool(self._minor_bodies))
        self.btn_explore.clicked.connect(lambda: self.exploreRequested.emit())
        bb.addWidget(self.btn_explore)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        bb.addWidget(self.btn_close)
        lay.addLayout(bb)

    # ---- Minor-body progress mirroring ---------------------------------

    def minor_body_begin(self, msg: str = "Minor-body search starting…"):
        """
        Reveal the mirrored progress strip at the bottom of the results
        window and reset it to zero. Called by the main dialog when it
        kicks off the catalog query, so the user sees activity even when
        the parent window is covered.
        """
        self._mb_status.setText(msg)
        self._mb_status.setVisible(True)
        self._mb_progress.setValue(0)
        self._mb_progress.setVisible(True)

    def minor_body_progress(self, pct: int, msg: str):
        self._mb_status.setText(str(msg or ""))
        self._mb_status.setVisible(True)
        try:
            v = int(max(0, min(100, int(pct))))
        except Exception:
            v = 0
        self._mb_progress.setValue(v)
        self._mb_progress.setVisible(True)

    def minor_body_end(self, final_msg: str = "", keep_status: bool = True):
        self._mb_progress.setValue(100)
        self._mb_progress.setVisible(False)
        if final_msg and keep_status:
            self._mb_status.setText(final_msg)
            self._mb_status.setVisible(True)
        else:
            self._mb_status.setVisible(False)

    # ---- Overlay helpers -----------------------------------------------

    def _bodies_for(self, idx: int, mode: str):
        """
        Predicted positions to draw on a given frame.

        Search and difference views get that frame's OWN epoch, so a
        moving object's circle tracks it as you step or play through the
        set. The reference view keeps the anchor epoch — drawing a search
        frame's positions over the reference would mark where the object
        was not.
        """
        if mode == self.MODE_REF:
            return self._minor_bodies
        if not (0 <= idx < len(self._anoms)):
            return self._minor_bodies
        return (self._bodies_per_frame.get(self._anoms[idx].get("path"))
                or self._minor_bodies)

    def _overlay_boxes_on(self, base: np.ndarray, anomalies: list,
                          bodies: list | None = None) -> np.ndarray:
        """Draw anomaly bboxes + minor-body circles onto a copy of `base`.
        Returns a uint8 RGB image sized to match."""
        if bodies is None:
            bodies = self._minor_bodies
        if base is None:
            return None
        img = base
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=2)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = img.copy()
        margin = 10
        h, w = img.shape[:2]
        labels_on = bool(getattr(self, "_show_labels", True))
        fs = float(np.clip(w / 2400.0, 0.38, 0.75))
        lt = int(max(1, round(fs * 1.6)))
        occupied = []

        # A body that has already been named on a detection box must not be
        # named again beside its catalog circle — those two markers sit on
        # top of each other by construction (that is what "matched" means),
        # so printing both guarantees overlapping text.
        named_on_box = set()

        for a in anomalies or []:
            x1 = max(0, a["minX"] - margin)
            y1 = max(0, a["minY"] - margin)
            x2 = min(w - 1, a["maxX"] + margin)
            y2 = min(h - 1, a["maxY"] + margin)
            mbs = a.get("matched_bodies") or []
            color = (0, 255, 0) if mbs else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=5)
            if labels_on:
                if mbs:
                    # A detection that matches the catalog gets ONE combined
                    # label on the box — designation and magnitude together.
                    # The identification is the whole answer; making the
                    # user hunt for it in a tree defeats the box.
                    b0 = mbs[0]
                    name = str(b0.get("designation", "?"))
                    named_on_box.add(name)
                    try:
                        H = float(b0.get("H"))
                        if np.isfinite(H):
                            name = f"{name}  H={H:.1f}"
                    except Exception:
                        pass
                    if len(mbs) > 1:
                        name += f"  (+{len(mbs) - 1})"
                        for extra in mbs[1:]:
                            named_on_box.add(str(extra.get("designation", "?")))
                    _draw_label(img, name, x2 + 8, y1 + 4, (0, 255, 0), fs, lt,
                                occupied=occupied, anchor=(x2, y1))
                else:
                    ps = a.get("peak_sigma")
                    if ps:
                        _draw_label(img, f"{ps:.1f}σ", x1, y1 - 6,
                                    (255, 90, 90), fs * 0.85, lt,
                                    occupied=occupied, anchor=(x1, y1))

        for body in bodies:
            try:
                x = int(round(float(body["x"])))
                y = int(round(float(body["y"])))
            except Exception:
                continue
            if not (0 <= x < w and 0 <= y < h):
                continue
            cv2.circle(img, (x, y), 8, (255, 255, 0), thickness=2)
            desig = str(body.get("designation", "") or "?")
            if labels_on and desig not in named_on_box:
                H = body.get("H")
                try:
                    if H is not None and np.isfinite(float(H)):
                        desig = f"{desig}  H={float(H):.1f}"
                except Exception:
                    pass
                # Offset down-right of the circle so the marker itself
                # stays unobstructed.
                _draw_label(img, desig, x + 12, y + 16, (255, 255, 0), fs, lt,
                            occupied=occupied, anchor=(x, y))
        return img

    def _diff_display(self, diff: np.ndarray | None) -> np.ndarray | None:
        """Normalize a signed diff into [0,1] for display — MAD-based so
        a bright transient doesn't blow out the display."""
        if diff is None:
            return None
        d = diff.astype(np.float32)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)) * 1.4826)
        mad = max(mad, 1e-6)
        lo = med - 2.0 * mad
        hi = med + 8.0 * mad
        return np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    def _frame_image(self, idx: int, mode: str) -> np.ndarray | None:
        """
        Fetch (and cache) the overlay-stamped uint8 image for a given
        (idx, mode). Lazy — not built until first requested. Ref overlays
        are built per-index because the anomaly boxes depend on which
        search frame is currently selected.
        """
        key = (mode, idx)
        if key in self._cache:
            return self._cache[key]
        if not (0 <= idx < len(self._anoms)):
            return None
        data = self._anoms[idx]
        anomalies = data.get("anomalies", [])
        if mode == self.MODE_SEARCH:
            base = data.get("search_image")
        elif mode == self.MODE_REF:
            base = self._payload.get("ref_image")
        elif mode == self.MODE_DIFF:
            base = self._diff_display(data.get("diff_image"))
        else:
            return None
        out = self._overlay_boxes_on(base, anomalies,
                                     self._bodies_for(idx, mode))
        self._cache[key] = out
        return out

    # ---- Viewer updates ------------------------------------------------

    def _show(self, idx: int, mode: str, *, fit: bool = False):
        """
        Swap the viewer to frame `idx` in `mode`. By default we PRESERVE
        the user's current zoom + pan — the whole point of play-through
        is to zoom in on one spot and see motion past it. Only the very
        first show (`fit=True`) resets the view.
        """
        if not (0 <= idx < len(self._anoms)):
            return
        img = self._frame_image(idx, mode)
        if img is None:
            return
        self._current_idx = idx
        self._mode = mode

        # Keep mode button check-state consistent regardless of entry path.
        self.btn_mode_search.setChecked(mode == self.MODE_SEARCH)
        self.btn_mode_ref.setChecked(mode == self.MODE_REF)
        self.btn_mode_diff.setChecked(mode == self.MODE_DIFF)

        # preserve_view keeps the current transform + scroll offset
        # across the pixmap swap. All frames in a run share the reference
        # grid (same H×W), so the preserve path takes effect — the user
        # stays zoomed in on whatever region they were inspecting.
        self.view.set_image(img, preserve_view=not fit)
        if fit:
            QTimer.singleShot(0, self.view.fit_to_view)

        name = self._anoms[idx]["imageName"]
        n = self._anoms[idx]["anomalyCount"]
        ids = [self._match_text(a)
               for a in (self._anoms[idx].get("anomalies") or [])]
        ids = [i for i in ids if i]
        extra = f" · ID: {', '.join(ids)}" if ids else ""
        self._status_lbl.setText(
            f"Frame {idx + 1}/{len(self._anoms)} · {name} · {n} anomalies · "
            f"[{mode}]{extra}"
        )

        # Sync tree selection without re-triggering double-click.
        try:
            self._tree.blockSignals(True)
            top = self._tree.topLevelItem(idx)
            if top is not None:
                self._tree.setCurrentItem(top)
        finally:
            self._tree.blockSignals(False)

    def _set_mode(self, mode: str):
        # Mode buttons are "radio-like" — manually sync the uncheck state
        # so clicking the already-checked one doesn't turn it off.
        if self._mode == mode:
            # Re-check so it doesn't look toggled-off.
            if mode == self.MODE_SEARCH: self.btn_mode_search.setChecked(True)
            elif mode == self.MODE_REF: self.btn_mode_ref.setChecked(True)
            elif mode == self.MODE_DIFF: self.btn_mode_diff.setChecked(True)
            return
        self._show(self._current_idx, mode)

    # ---- Playback ------------------------------------------------------

    def _on_play_toggled(self, playing: bool):
        self._playing = playing
        if playing:
            self.btn_play.setText("⏸ Pause")
            # Flip to Search mode when play starts — that's the sequence
            # that makes drift visible. User can still manually mode-switch.
            if self._mode != self.MODE_SEARCH:
                self._set_mode(self.MODE_SEARCH)
            self._timer.start(int(self.sp_rate.value()))
        else:
            self.btn_play.setText("▶ Play")
            self._timer.stop()

    def _advance_search_frame(self):
        if not self._anoms:
            return
        nxt = (self._current_idx + 1) % len(self._anoms)
        self._show(nxt, self.MODE_SEARCH)

    def _prev_frame(self):
        if not self._anoms:
            return
        prv = (self._current_idx - 1) % len(self._anoms)
        self._show(prv, self._mode)

    def _next_frame(self):
        if not self._anoms:
            return
        nxt = (self._current_idx + 1) % len(self._anoms)
        self._show(nxt, self._mode)

    # ---- Tree ----------------------------------------------------------

    def _on_tree_double(self, item, column):
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            parent = item.parent()
            if parent is None:
                return
            idx = parent.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        # Pause play when the user picks a specific frame — they want to
        # inspect this one, not have it flip away a half-second later.
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)
        self._show(int(idx), self._mode)

    def _on_labels_toggled(self, on: bool):
        self._show_labels = bool(on)
        self._cache.clear()          # overlays are baked in, so rebuild
        if self._anoms:
            self._show(self._current_idx, self._mode)

    @staticmethod
    def _match_text(anomaly) -> str:
        """Designation(s) this detection cross-matched to, for the tree."""
        mbs = anomaly.get("matched_bodies") or []
        if not mbs:
            return ""
        names = [str(b.get("designation", "?")) for b in mbs[:2]]
        txt = ", ".join(names)
        if len(mbs) > 2:
            txt += f" (+{len(mbs) - 2})"
        return txt

    def current_frame_index(self) -> int:
        """Index of the frame currently in the viewer."""
        return int(self._current_idx)

    def current_frame_path(self) -> str | None:
        if 0 <= self._current_idx < len(self._anoms):
            return self._anoms[self._current_idx].get("path")
        return None

    def _rebuild_tree(self):
        """
        Repopulate the summary tree from the current anomaly data. Split
        out of _init_ui so a re-detect can refresh it without rebuilding
        the whole dialog.
        """
        self._tree.blockSignals(True)
        try:
            self._tree.clear()
            for i, data in enumerate(self._anoms):
                anoms = data.get("anomalies", []) or []
                top_peak = max((a.get("peak_sigma", 0.0) for a in anoms),
                               default=0.0)
                named = [self._match_text(a) for a in anoms]
                named = [n for n in named if n]
                item = QTreeWidgetItem([
                    data["imageName"],
                    str(data["anomalyCount"]),
                    f"{top_peak:.1f}" if top_peak else "",
                    named[0] if len(named) == 1 else
                    (f"{len(named)} identified" if named else ""),
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, i)
                for a in anoms:
                    child = QTreeWidgetItem([
                        f"  ({a['minX']},{a['minY']}) → ({a['maxX']},{a['maxY']})",
                        f"area={a['area']}",
                        f"{a.get('peak_sigma', 0.0):.1f}",
                        self._match_text(a),
                    ])
                    item.addChild(child)
                self._tree.addTopLevelItem(item)
        finally:
            self._tree.blockSignals(False)

    # ---- Re-detect -----------------------------------------------------

    def current_det_params(self) -> dict:
        return {
            "k_sigma": float(self.sp_sigma.value()),
            "min_area": int(self.sp_minarea.value()),
            "max_dim": int(self.sp_maxdim.value()),
            "border_frac": float(
                (self._payload.get("det_params", {}) or {}).get("border_frac", 0.05)),
            "use_star_mask": bool(self.chk_starmask.isChecked()),
            "star_grow_px": int(self.sp_grow.value()),
            "psf_sigma": float(self.sp_psf.value()),
        }

    def _on_redetect(self):
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)
        self.btn_redetect.setEnabled(False)
        self.btn_redetect.setText("Re-detecting…")
        self.redetectRequested.emit(self.current_det_params())

    def refresh(self, payload: dict | None = None):
        """
        Re-render everything from (possibly updated) payload data. Used
        after a re-detect. Preserves which frame you were on and — via
        _show's preserve_view — your zoom and pan.
        """
        if payload is not None:
            self._payload = payload
            self._anoms = payload.get("anomaly_data", [])
        self._cache.clear()
        self._rebuild_tree()
        total = sum(d["anomalyCount"] for d in self._anoms)
        self._summary_lbl.setText(
            f"Processed {len(self._anoms)} search frame(s). "
            f"Total candidate anomalies: {total}"
        )
        self.btn_redetect.setEnabled(True)
        self.btn_redetect.setText("Re-detect")
        if self._anoms:
            self._show(min(self._current_idx, len(self._anoms) - 1), self._mode)

    # ---- Push ----------------------------------------------------------

    def _on_push(self):
        img = self._frame_image(self._current_idx, self._mode)
        if img is None:
            QMessageBox.information(self, "Push", "No frame to push.")
            return
        data = self._anoms[self._current_idx]
        title = f"Anomalies in {data['imageName']} [{self._mode}]"
        self.pushed.emit(img, title, data["path"])

    # ---- Minor-body refresh --------------------------------------------

    def update_minor_bodies(self, bodies: list, per_frame: dict | None = None):
        """
        Called after the catalog query returns. Blow the overlay cache so
        the next render stamps the circles and names on; the next _show()
        call rebuilds whatever frame is visible.
        """
        self._minor_bodies = bodies or []
        if per_frame is not None:
            self._bodies_per_frame = per_frame or {}
        try:
            self.btn_explore.setEnabled(bool(self._minor_bodies))
        except Exception:
            pass
        self._cache.clear()
        self._rebuild_tree()
        total = sum(d["anomalyCount"] for d in self._anoms)
        named = sum(1 for d in self._anoms
                    for a in (d.get("anomalies") or [])
                    if a.get("matched_bodies"))
        self._summary_lbl.setText(
            f"Processed {len(self._anoms)} search frame(s). "
            f"Total candidate anomalies: {total}.  "
            f"{len(self._minor_bodies)} catalogued minor bod"
            f"{'y' if len(self._minor_bodies) == 1 else 'ies'} in frame, "
            f"{named} detection(s) identified."
        )
        if self._anoms:
            self._show(self._current_idx, self._mode)

    # ---- Cleanup -------------------------------------------------------

    def closeEvent(self, e):
        try:
            self._timer.stop()
        except Exception:
            pass
        super().closeEvent(e)


# ─────────────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────────────

class SupernovaAsteroidHunterDialog(QDialog):
    def __init__(self, parent=None, settings=None,
                 image_manager=None, doc_manager=None,
                 supernova_path=None, wrench_path=None, spinner_path=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Supernova / Asteroid Hunter"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        if supernova_path:
            self.setWindowIcon(QIcon(supernova_path))
        self.supernova_path = supernova_path

        self.settings = settings
        self.image_manager = image_manager
        self.doc_manager = doc_manager

        # Pipeline + minor-body state.
        self._pipeline_thread: HunterPipelineThread | None = None
        self._redetect_thread: RedetectThread | None = None
        self._redetect_reopen: bool = False
        self._last_payload: dict | None = None
        self._results_dialog: ResultsDialog | None = None
        self._mb_thread: QThread | None = None
        self._mb_worker: MinorBodyWorker | None = None

        # WCS / JD state (populated after a successful run).
        self.ref_header = None
        self.ref_wcs = None
        self.ref_jd = None
        self.ref_site = None
        self.predicted_minor_bodies: list | None = None
        self.minor_bodies_per_frame: dict = {}
        self._minor_frame_index: int = -1

        # Minor-body knobs (identical defaults to original).
        self.minor_H_ast_max = 20.0
        self.minor_H_com_max = 15.0
        self.minor_ast_max_count = 50000
        self.minor_com_max_count = 5000
        self.minor_time_offset_hours = 0.0
        self.minor_use_midexposure = True
        self._minor_jd: float | None = None
        self._minor_force_solve: bool = False

        # Simple parameters dict used by the UI.
        self.parameters = {
            "referenceImagePath": "",
            "searchImagePaths": [],
        }

        self.setLayout(QVBoxLayout())
        self.initUI()
        self.resize(960, 760)

    # ---- UI construction -------------------------------------------------

    def initUI(self):
        layout = self.layout()

        intro = QLabel(self.tr(
            "Pick a reference and one or more search images. The pipeline "
            "will (optionally) register and background-correct the frames "
            "in parallel, then flag per-pixel differences as candidate "
            "anomalies. Review each flagged frame with the Blink view."
        ))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- Reference ---
        ref_box = QGroupBox(self.tr("Reference Image"))
        ref_lay = QHBoxLayout(ref_box)
        self.ref_line_edit = QLineEdit(self)
        self.ref_line_edit.setPlaceholderText(self.tr("No reference image selected"))
        self.ref_line_edit.setReadOnly(True)
        self.ref_button = QPushButton(self.tr("Select…"), self)
        self.ref_button.clicked.connect(self.selectReferenceImage)
        ref_lay.addWidget(self.ref_line_edit, 1)
        ref_lay.addWidget(self.ref_button)
        layout.addWidget(ref_box)

        # --- Search images ---
        search_box = QGroupBox(self.tr("Search Images"))
        search_lay = QHBoxLayout(search_box)
        self.search_list = QListWidget(self)
        self.search_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        search_btns = QVBoxLayout()
        self.search_button = QPushButton(self.tr("Add Files…"))
        self.search_button.clicked.connect(self.selectSearchImages)
        self.search_remove_btn = QPushButton(self.tr("Remove Selected"))
        self.search_remove_btn.clicked.connect(self._remove_selected_search)
        self.search_clear_btn = QPushButton(self.tr("Clear"))
        self.search_clear_btn.clicked.connect(self._clear_search)
        search_btns.addWidget(self.search_button)
        search_btns.addWidget(self.search_remove_btn)
        search_btns.addWidget(self.search_clear_btn)
        search_btns.addStretch(1)
        search_lay.addWidget(self.search_list, 1)
        search_lay.addLayout(search_btns)
        layout.addWidget(search_box)

        # --- Pipeline options ---
        opts_box = QGroupBox(self.tr("Pipeline Options"))
        opts_lay = QGridLayout(opts_box)

        # row 0: cosmetic correction
        self.cosmetic_checkbox = QCheckBox(
            self.tr("Cosmetic correction (hot/cold pixel suppression)"))
        self.cosmetic_checkbox.setChecked(False)
        self.cosmetic_checkbox.setToolTip(self.tr(
            "Suppress hot/cold pixels. Runs AFTER binning (binning already\n"
            "averages away most of them; this mops up the rest) and one\n"
            "frame at a time — the correction is already parallel inside,\n"
            "so fanning it across workers would choke the machine.\n\n"
            "Leave off for already-stacked integrations; turn on for raw\n"
            "single subs, where a hot pixel absent from the reference would\n"
            "otherwise read as a transient."))
        opts_lay.addWidget(self.cosmetic_checkbox, 0, 0, 1, 3)

        # row 1: alignment
        self.align_checkbox = QCheckBox(
            self.tr("Align search images to reference (astroalign)"))
        self.align_checkbox.setChecked(True)
        self.align_checkbox.setToolTip(
            "Register every search frame to the reference before differencing.\n"
            "Enable this unless the frames are already pixel-registered —\n"
            "pixel-wise subtraction on unregistered frames produces false\n"
            "anomalies around every star."
        )
        opts_lay.addWidget(self.align_checkbox, 1, 0, 1, 3)

        # row 2: ADBE on/off + degree + rbf
        self.adbe_checkbox = QCheckBox(self.tr("ADBE background removal"))
        self.adbe_checkbox.setChecked(True)
        self.adbe_checkbox.setToolTip(
            "Runs the modern Automatic Dynamic Background Extraction on\n"
            "every frame (reference + search) in parallel. Flattens gradients\n"
            "so photometric differences aren't swamped by sky slope."
        )
        opts_lay.addWidget(self.adbe_checkbox, 2, 0)

        opts_lay.addWidget(QLabel(self.tr("Degree:")), 2, 1)
        self.adbe_degree = QSpinBox()
        self.adbe_degree.setRange(0, 6)
        self.adbe_degree.setValue(2)
        self.adbe_degree.setToolTip("Polynomial degree. 0 = RBF only.")
        opts_lay.addWidget(self.adbe_degree, 2, 2)

        self.adbe_rbf = QCheckBox(self.tr("RBF refinement"))
        self.adbe_rbf.setChecked(True)
        opts_lay.addWidget(self.adbe_rbf, 2, 3)

        self.adbe_checkbox.toggled.connect(self._on_adbe_toggled)
        self._on_adbe_toggled(self.adbe_checkbox.isChecked())

        # row 3: detection threshold
        opts_lay.addWidget(QLabel(self.tr("Detection σ (above MAD):")), 3, 0)
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 20.0)
        self.sigma_spin.setSingleStep(0.5)
        self.sigma_spin.setValue(5.0)
        self.sigma_spin.setDecimals(1)
        self.sigma_spin.setToolTip(
            "Threshold = σ × noise of the matched-filtered difference,\n"
            "measured per frame on valid pixels only.\n\n"
            "All of these can be re-tuned after a run without\n"
            "reprocessing — see 'Re-detect (cached)'."
        )
        opts_lay.addWidget(self.sigma_spin, 3, 1)

        opts_lay.addWidget(QLabel(self.tr("Min area (px):")), 3, 2)
        self.minarea_spin = QSpinBox()
        self.minarea_spin.setRange(1, 2000)
        self.minarea_spin.setValue(5)
        self.minarea_spin.setToolTip(
            "Minimum significant-pixel count, measured on the matched-\n"
            "filtered image before the connectivity dilation."
        )
        opts_lay.addWidget(self.minarea_spin, 3, 3)

        # row 4: max dim, border
        opts_lay.addWidget(QLabel(self.tr("Max bbox dim (px):")), 4, 0)
        self.maxdim_spin = QSpinBox()
        self.maxdim_spin.setRange(10, 10000)
        self.maxdim_spin.setValue(200)
        opts_lay.addWidget(self.maxdim_spin, 4, 1)

        opts_lay.addWidget(QLabel(self.tr("Border exclude (%):")), 4, 2)
        self.border_spin = QDoubleSpinBox()
        self.border_spin.setRange(0.0, 20.0)
        self.border_spin.setValue(5.0)
        self.border_spin.setSingleStep(1.0)
        self.border_spin.setDecimals(1)
        opts_lay.addWidget(self.border_spin, 4, 3)

        # row 5: workers + bin factor (both resource knobs — group them)
        opts_lay.addWidget(QLabel(self.tr("Parallel workers:")), 5, 0)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 64)
        default_workers = max(2, min((os.cpu_count() or 4), 16))
        self.workers_spin.setValue(default_workers)
        self.workers_spin.setToolTip(
            "Number of parallel processes used for each pipeline stage.\n"
            "Defaults to roughly half your CPU count to leave headroom for\n"
            "the GUI and the OS. Raise it on headless/benchmark runs."
        )
        opts_lay.addWidget(self.workers_spin, 5, 1)

        opts_lay.addWidget(QLabel(self.tr("Bin factor:")), 5, 2)
        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(1, 8)
        # Default 2: a 45MP frame drops to ~11MP, memory footprint
        # drops ~4x, SNR for a transient DETECTION rises by a factor of
        # 2 (block-average), and bounding boxes stay well above the
        # min-area filter. Transients wider than 1 pixel are unaffected.
        self.bin_spin.setValue(2)
        self.bin_spin.setToolTip(
            "Downsample every frame on load by NxN block averaging.\n"
            "1 = no binning (full resolution — expect ~1.2 GB per 45MP\n"
            "    frame once ADBE, align and diff buffers are live).\n"
            "2 = 4x less memory + IPC, slightly BETTER transient SNR\n"
            "    (block averaging compresses read noise). Recommended\n"
            "    for 24MP+ sensors unless you need sub-pixel bounding\n"
            "    boxes.\n"
            "3-4 = aggressive — for huge mosaics or low-RAM machines.\n"
            "The WCS is rescaled along with the pixels so minor-body\n"
            "cross-match still lands correctly."
        )
        opts_lay.addWidget(self.bin_spin, 5, 3)

        # row 6: matched-filter width — the key faint-object control
        opts_lay.addWidget(QLabel(self.tr("PSF σ (matched filter):")), 6, 0)
        self.psf_spin = QDoubleSpinBox()
        self.psf_spin.setRange(0.0, 6.0)
        self.psf_spin.setSingleStep(0.25)
        self.psf_spin.setDecimals(2)
        self.psf_spin.setValue(1.5)
        self.psf_spin.setToolTip(
            "Width of the Gaussian the difference image is convolved with\n"
            "before thresholding, in BINNED pixels (≈ star FWHM / 2.35).\n\n"
            "Matched filtering is the optimal detector for a point source\n"
            "and is worth roughly 3x in SNR over thresholding raw pixels —\n"
            "it is what makes a faint asteroid detectable at all rather\n"
            "than lost among single-pixel noise spikes. It also rejects\n"
            "single-pixel cosmic rays, which don't match the PSF.\n\n"
            "Set to 0 to threshold the raw difference instead."
        )
        opts_lay.addWidget(self.psf_spin, 6, 1)

        layout.addWidget(opts_box)

        # --- Run row ---
        run_row = QHBoxLayout()
        self.process_button = QPushButton(self.tr("Run Hunter"))
        self.process_button.setStyleSheet("font-weight: bold;")
        self.process_button.clicked.connect(self.process)
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_pipeline)
        run_row.addWidget(self.process_button, 1)
        run_row.addWidget(self.cancel_button)
        # Re-detect against the cached differences from the last run.
        # Changing only a detection knob never needs the pipeline again.
        self.redetect_button = QPushButton(self.tr("Re-detect (cached)"))
        self.redetect_button.setEnabled(False)
        self.redetect_button.setToolTip(
            "Re-run only the detection stage against the difference images\n"
            "cached from the last run, using the detection settings above,\n"
            "and reopen the results. No reload, no re-align, no ADBE."
        )
        self.redetect_button.clicked.connect(self._on_redetect_from_main)
        run_row.addWidget(self.redetect_button)
        run_row.addWidget(make_help_button("supernova_asteroid_hunter", self))
        layout.addLayout(run_row)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel(self.tr("Status: Idle"))
        layout.addWidget(self.status_label)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_view)

        # Minor-body progress bar (hidden; reused from original flow).
        self.minor_progress = QProgressBar()
        self.minor_progress.setRange(0, 100)
        self.minor_progress.setValue(0)
        self.minor_progress.setVisible(False)
        layout.addWidget(self.minor_progress)

        # --- Reset / New Instance ---
        bottom_row = QHBoxLayout()
        bottom_row.addStretch(1)
        self.new_instance_button = QPushButton(self.tr("New Instance"))
        self.new_instance_button.clicked.connect(self.newInstance)
        bottom_row.addWidget(self.new_instance_button)
        layout.addLayout(bottom_row)

    def _on_adbe_toggled(self, on: bool):
        self.adbe_degree.setEnabled(on)
        self.adbe_rbf.setEnabled(on)

    # ---- File pickers ----------------------------------------------------

    # ---- Remembered folder helpers --------------------------------------
    #
    # Separate keys for the reference and the search set, because in
    # practice they usually live in different folders (reference in an
    # archive directory, search frames in the current night's capture).
    # Falls back to the generic "working_directory" setting other SASpro
    # tools use when a dedicated hunter key hasn't been written yet.

    _REF_DIR_KEY = "supernova_hunter/last_ref_dir"
    _SEARCH_DIR_KEY = "supernova_hunter/last_search_dir"

    def _remembered_dir(self, key: str) -> str:
        if self.settings is None:
            return ""
        try:
            v = self.settings.value(key, "", type=str) or ""
            if v and os.path.isdir(v):
                return v
            # One-time back-compat: fall through to the generic working dir
            # other SASpro tools write.
            v2 = self.settings.value("working_directory", "", type=str) or ""
            return v2 if v2 and os.path.isdir(v2) else ""
        except Exception:
            return ""

    def _remember_dir(self, key: str, path: str) -> None:
        if self.settings is None or not path:
            return
        try:
            d = path if os.path.isdir(path) else os.path.dirname(path)
            if d:
                self.settings.setValue(key, d)
        except Exception:
            pass

    def selectReferenceImage(self):
        start_dir = self._remembered_dir(self._REF_DIR_KEY)
        file_path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Select Reference Image"), start_dir,
            self.tr("Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"))
        if file_path:
            self.parameters["referenceImagePath"] = file_path
            self.ref_line_edit.setText(os.path.basename(file_path))
            self._remember_dir(self._REF_DIR_KEY, file_path)

    def selectSearchImages(self):
        start_dir = self._remembered_dir(self._SEARCH_DIR_KEY)
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Select Search Images"), start_dir,
            self.tr("Images (*.png *.tif *.tiff *.fits *.fit *.xisf)"))
        if file_paths:
            existing = set(self.parameters["searchImagePaths"])
            for p in file_paths:
                if p not in existing:
                    self.parameters["searchImagePaths"].append(p)
                    self.search_list.addItem(os.path.basename(p))
            # Anchor on the last pick so a batch spanning subfolders settles
            # on where the user actually ended up.
            self._remember_dir(self._SEARCH_DIR_KEY, file_paths[-1])

    def _remove_selected_search(self):
        rows = sorted({self.search_list.row(it)
                       for it in self.search_list.selectedItems()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.parameters["searchImagePaths"]):
                del self.parameters["searchImagePaths"][r]
            self.search_list.takeItem(r)

    def _clear_search(self):
        self.parameters["searchImagePaths"].clear()
        self.search_list.clear()

    # ---- Pipeline control -----------------------------------------------

    def _append_log(self, msg: str):
        self.log_view.append(msg)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_progress(self, done: int, total: int, msg: str):
        if total > 0:
            pct = int(round(100.0 * done / float(total)))
            self.progress_bar.setValue(max(0, min(100, pct)))
        self.status_label.setText(f"Status: {msg}")

    def process(self):
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            QMessageBox.information(
                self, "Hunter",
                "A pipeline run is already in progress — wait for it to\n"
                "finish or press Cancel first.")
            return

        ref_path = self.parameters["referenceImagePath"]
        if not ref_path:
            QMessageBox.warning(self, "Error", "Please select a reference image.")
            return
        search_paths = self.parameters["searchImagePaths"]
        if not search_paths:
            QMessageBox.warning(self, "Error", "Please add at least one search image.")
            return

        params = {
            "ref_path": ref_path,
            "search_paths": list(search_paths),
            "do_cosmetic": self.cosmetic_checkbox.isChecked(),
            "do_align": self.align_checkbox.isChecked(),
            "do_adbe": self.adbe_checkbox.isChecked(),
            "adbe_degree": self.adbe_degree.value(),
            "adbe_rbf": self.adbe_rbf.isChecked(),
            "k_sigma": self.sigma_spin.value(),
            "min_area": self.minarea_spin.value(),
            "max_dim": self.maxdim_spin.value(),
            "border_frac": max(0.0, self.border_spin.value() / 100.0),
            "workers": self.workers_spin.value(),
            "bin_factor": self.bin_spin.value(),
            "psf_sigma": self.psf_spin.value(),
        }

        self.log_view.clear()
        self._append_log(f"Starting hunter: {len(search_paths)} search frame(s)")
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Running…")

        self._pipeline_thread = HunterPipelineThread(params, parent=self)
        self._pipeline_thread.progress.connect(self._on_progress)
        self._pipeline_thread.log.connect(self._append_log)
        self._pipeline_thread.finished_ok.connect(self._on_pipeline_ok)
        self._pipeline_thread.finished_err.connect(self._on_pipeline_err)

        self.process_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._pipeline_thread.start()

    def _cancel_pipeline(self):
        if self._pipeline_thread is None:
            return
        self._pipeline_thread.request_cancel()
        self.status_label.setText("Status: cancelling…")
        self.cancel_button.setEnabled(False)

    def _on_pipeline_ok(self, payload):
        self._reset_pipeline_buttons()
        self._last_payload = payload
        self.ref_header = payload.get("ref_header")
        self.ref_wcs = payload.get("ref_wcs")
        self.ref_jd = payload.get("ref_jd")
        self.predicted_minor_bodies = None

        self.progress_bar.setValue(100)
        self.status_label.setText("Status: Done.")
        self._append_log("✅ Pipeline complete.")

        # Show the results dialog.
        self.redetect_button.setEnabled(True)
        self._open_results_dialog(payload)

    def _open_results_dialog(self, payload):
        """Create (or recreate) the results window for a payload."""
        if self._results_dialog is not None:
            try:
                self._results_dialog.close()
            except Exception:
                pass
            self._results_dialog = None
        icon = QIcon(self.supernova_path) if self.supernova_path else None
        self._results_dialog = ResultsDialog(payload, icon=icon, parent=self)
        self._results_dialog.pushed.connect(self._handle_push_to_new_view)
        self._results_dialog.redetectRequested.connect(self._on_redetect_requested)
        # Fire the mirrored progress UI immediately so the user gets a
        # response the instant they click, before the dialog's own setup
        # work (prompt, plate solve) has a chance to make them wonder if
        # the click registered.
        self._results_dialog.btn_minor.clicked.connect(
            lambda: self._results_dialog.minor_body_begin(
                "Minor-body: preparing search…"))
        self._results_dialog.btn_minor.clicked.connect(self.runMinorBodySearch)
        self._results_dialog.exploreRequested.connect(self.exploreOrbit)
        self._results_dialog.show()

    # ---- 3D orbit explorer ----------------------------------------------

    def exploreOrbit(self):
        """
        Open the interactive 3D orbit viewer for a catalogued body at the
        exact JD this search was evaluated at. If several bodies are in
        the field, let the user pick which one; bodies that matched a
        detection are flagged and listed first.
        """
        from PyQt6.QtWidgets import QInputDialog

        if not _HAS_ORBIT_EXPLORER:
            QMessageBox.information(
                self, "Orbit Explorer",
                "The 3D orbit explorer could not be loaded.\n\n"
                f"{_ORBIT_EXPLORER_ERR}\n\n"
                "It needs pyqtgraph.opengl and PyOpenGL "
                "(pip install PyOpenGL).")
            return

        bodies = list(self.predicted_minor_bodies or [])
        if not bodies:
            QMessageBox.information(
                self, "Orbit Explorer",
                "No catalogued minor bodies yet — run "
                "'Catalogue Minor Bodies in Field' first.")
            return

        jd = getattr(self, "_minor_jd", None) or self.ref_jd
        if jd is None:
            QMessageBox.warning(
                self, "Orbit Explorer",
                "No observation time is available to place the orbit at.")
            return

        # Which designations actually matched a detection — flag and float
        # them to the top of the chooser, since those are the ones the user
        # most likely wants to inspect.
        matched = set()
        payload = self._last_payload or {}
        for entry in payload.get("anomaly_data", []) or []:
            for a in entry.get("anomalies") or []:
                for b in a.get("matched_bodies") or []:
                    d = b.get("designation")
                    if d:
                        matched.add(d)

        def _sort_key(b):
            d = b.get("designation", "")
            return (0 if d in matched else 1, d)

        bodies.sort(key=_sort_key)

        if len(bodies) == 1:
            chosen = bodies[0]
        else:
            labels = []
            for b in bodies:
                d = str(b.get("designation", "?"))
                H = b.get("H")
                try:
                    hs = f"  H={float(H):.1f}" if H is not None and np.isfinite(float(H)) else ""
                except Exception:
                    hs = ""
                flag = "  ✓ detected" if d in matched else ""
                noel = "" if b.get("elements") else "  (no elements)"
                labels.append(f"{d}{hs}{flag}{noel}")
            label, ok = QInputDialog.getItem(
                self, "Explore Orbit",
                "Choose a body to view in 3D:", labels, 0, False)
            if not ok:
                return
            chosen = bodies[labels.index(label)]

        if chosen.get("elements") is None:
            QMessageBox.information(
                self, "Orbit Explorer",
                f"{chosen.get('designation', 'This body')} has no orbital "
                "elements in the catalog row, so its orbit can't be drawn.\n\n"
                "Other bodies in the field may still have them.")
            return

        icon = QIcon(self.supernova_path) if self.supernova_path else None
        dlg = open_orbit_explorer(self, chosen, float(jd),
                                  settings=self.settings, icon=icon)
        if dlg is not None:
            # Keep a reference so the non-modal window isn't garbage
            # collected the moment this method returns.
            if not hasattr(self, "_orbit_dialogs"):
                self._orbit_dialogs = []
            self._orbit_dialogs.append(dlg)

    # ---- Re-detect (Stage B only) ---------------------------------------

    def _on_redetect_from_main(self):
        """
        Re-detect using the MAIN dialog's detection widgets. This is the
        path for "I closed the results, nudged sigma, want to look
        again" — it reopens the results window when it finishes.
        """
        if self._last_payload is None:
            QMessageBox.information(
                self, "Re-detect",
                "Nothing cached yet — run the hunter once first.")
            return
        # The main dialog has no "Star mask grow" widget of its own —
        # that control lives in the results window. Carry forward whatever
        # grow was last applied (defaulting to 4) instead of resetting it.
        _prev = (self._last_payload or {}).get("det_params", {}) or {}
        _star_grow = int(_prev.get("star_grow_px", _prev.get("grow_px", 4)))
        self._redetect_reopen = True
        self._on_redetect_requested({
            "k_sigma": float(self.sigma_spin.value()),
            "min_area": int(self.minarea_spin.value()),
            "max_dim": int(self.maxdim_spin.value()),
            "border_frac": max(0.0, self.border_spin.value() / 100.0),
            "use_star_mask": True,
            "star_grow_px": _star_grow,
            "psf_sigma": float(self.psf_spin.value()),
        })

    def _on_redetect_requested(self, det_params):
        if self._last_payload is None:
            return
        if self._redetect_thread is not None and self._redetect_thread.isRunning():
            return
        self.status_label.setText("Status: re-detecting (cached diffs)…")
        self._redetect_thread = RedetectThread(
            self._last_payload, det_params, parent=self)
        self._redetect_thread.progress.connect(self._on_progress)
        self._redetect_thread.done.connect(self._on_redetect_done)
        self._redetect_thread.start()

    def _on_redetect_done(self, payload):
        self._last_payload = payload
        total = sum(d["anomalyCount"] for d in payload.get("anomaly_data", []))
        dp = payload.get("det_params", {})
        self.status_label.setText(
            f"Status: re-detect complete — {total} candidates at "
            f"σ={dp.get('k_sigma', '?')}"
        )
        self._append_log(
            f"Re-detect: σ={dp.get('k_sigma')} min_area={dp.get('min_area')} "
            f"star_grow={dp.get('star_grow_px')} star_mask={dp.get('use_star_mask')} "
            f"→ {total} candidates"
        )
        if getattr(self, "_redetect_reopen", False) or self._results_dialog is None:
            self._redetect_reopen = False
            self._open_results_dialog(payload)
        else:
            self._results_dialog.refresh(payload)
        try:
            self._redetect_thread = None
        except Exception:
            pass

    def _on_pipeline_err(self, message):
        self._reset_pipeline_buttons()
        self.status_label.setText("Status: Error.")
        self._append_log(f"⚠ {message}")
        QMessageBox.critical(self, "Hunter", message)

    def _reset_pipeline_buttons(self):
        self.process_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self._pipeline_thread is not None:
            try:
                self._pipeline_thread.quit()
                self._pipeline_thread.wait(1000)
            except Exception:
                pass
            self._pipeline_thread = None

    # ---- Push-to-new-view (preserved behavior) --------------------------

    def _handle_push_to_new_view(self, np_img, title: str, source_path: str):
        if not self.doc_manager:
            QMessageBox.warning(self, "No DocManager",
                                "No document manager is available to push the preview.")
            return
        if not source_path:
            QMessageBox.warning(self, "No Source File",
                                "Could not determine the original file for this preview.")
            return
        try:
            res = load_image(source_path, return_metadata=True)
        except Exception as e:
            QMessageBox.critical(self, "Load Error",
                                 f"Failed to load original image:\n{e}")
            return
        if not res or res[0] is None:
            QMessageBox.critical(self, "Load Error",
                                 "Could not read original image data from disk.")
            return
        orig_img, original_header, bit_depth, is_mono, meta = res
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("fits_header", original_header)
        meta.setdefault("original_header", original_header)
        meta.setdefault("bit_depth", bit_depth)
        meta.setdefault("is_mono", is_mono)
        meta.setdefault("source_path", source_path)
        meta["display_name"] = title

        img = np.asarray(np_img, copy=False)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        if img.max() > 1.01 or img.min() < -0.01:
            img = np.clip(img, 0, 255) / 255.0

        self.doc_manager.create_document(
            image=img, metadata=meta, name=title,
        )

    # ---- Minor body search (kept from original — already threaded) ------

    def _ensure_wcs(self, force: bool = False):
        """
        Guarantee a REAL astrometric solution for the reference frame.

        The solve runs on the binned linear reference kept in the payload
        — the same pixel grid the anomalies were measured on — so no bin
        rescaling is needed afterwards. Solving the full-resolution file
        from disk instead would return a WCS one bin factor off, and
        every predicted position would land on the wrong pixel.

        The solve itself is plate_solver._solve_numpy_with_fallback,
        which already runs the whole in-house chain:

            in-house Gaia DR3  →  ASTAP seeded  →  ASTAP blind
                               →  Astrometry.net (local solve-field,
                                  then the web API)

        Nothing here needed building — the chain was always there. It
        simply never ran, because the old guard was
        `if self.ref_wcs is not None`, and an unsolved header still
        yields a non-None (default, useless) WCS object.

        Returns (ok: bool, message: str).
        """
        if not force:
            ok, why = _wcs_is_usable(self.ref_wcs, self.ref_header)
            if ok:
                return True, f"using existing solution ({why})"

        payload = self._last_payload or {}
        img = payload.get("ref_linear")
        if img is None:
            path = (payload.get("ref_path")
                    or self.parameters.get("referenceImagePath") or "")
            if not path:
                return False, "no reference image available to solve"
            try:
                img, _hdr0, _bd, _mono = load_image(path)
            except Exception as e:
                return False, f"could not load reference for solving: {e}"
            if img is None:
                return False, "reference image unreadable"

        seed_h = (_as_header(self.ref_header)
                  if isinstance(self.ref_header, (dict, Header)) else None)
        acq_base = _strip_wcs_keys(seed_h) if isinstance(seed_h, Header) else None

        _solving_msg = ("Plate solving the reference frame "
                        "(Gaia DR3 → ASTAP → Astrometry.net)…")
        self.status_label.setText(f"Status: {_solving_msg}")
        if self._results_dialog is not None:
            try:
                # Mirror the solve stage too — this is the step that
                # takes several seconds, so without the mirrored strip
                # the results window just sits there looking frozen while
                # the main dialog shows "plate solving" behind it.
                self._results_dialog.minor_body_progress(5, _solving_msg)
            except Exception:
                pass
        QApplication.processEvents()

        try:
            ok, res = _solve_numpy_with_fallback(self, self.settings, img, seed_h)
        except Exception as e:
            return False, f"plate solve raised: {e}"
        if not ok:
            return False, f"plate solve failed: {res}"

        solver_hdr = res if isinstance(res, Header) else Header()
        hdr_final = (_merge_wcs_into_base_header(acq_base, solver_hdr)
                     if isinstance(acq_base, Header) else solver_hdr)
        try:
            new_wcs = WCS(hdr_final)
        except Exception as e:
            return False, f"WCS build failed after solve: {e}"

        good, why = _wcs_is_usable(new_wcs, hdr_final)
        if not good:
            return False, f"solver returned an unusable WCS ({why})"

        self.ref_header = hdr_final
        self.ref_wcs = new_wcs
        if self._last_payload is not None:
            self._last_payload["ref_header"] = hdr_final
            self._last_payload["ref_wcs"] = new_wcs
        self._append_log(f"Plate solve OK — {why}")
        return True, f"solved ({why})"

    def _prompt_minor_body_limits(self) -> bool:
        """
        Minor-body search setup.

        Beyond the magnitude limits, this shows the observation timestamp
        the ephemeris will actually be evaluated at, broken into its
        parts, recomputed live as the offset changes.

        That display is the whole point. An ephemeris is only as good as
        its timestamp, and a silently wrong DATE-OBS — a camera writing
        local time, a DST slip, a mid-exposure convention mismatch —
        looks EXACTLY like "no objects in this field". Putting the
        effective UTC and JD on screen turns an invisible failure into
        an obvious one.
        """
        payload = self._last_payload or {}
        anoms = payload.get("anomaly_data", []) or []

        # ------------------------------------------------------------------
        # Anchor on the frame the user is LOOKING AT.
        #
        # This used to take the reference frame's timestamp. With a
        # reference that is frame 1 of the night and the object being
        # inspected on frame 108 of 120-second subs, that is a 3.6-hour
        # epoch error — several hundred pixels for a main-belt asteroid.
        # No amount of timezone fiddling can fix that, though it can
        # accidentally mask part of it, which is exactly what a manual
        # "+1 hour" offset was doing.
        # ------------------------------------------------------------------
        idx = -1
        if anoms:
            idx = 0
            if self._results_dialog is not None:
                try:
                    idx = int(self._results_dialog.current_frame_index())
                except Exception:
                    idx = 0
            idx = max(0, min(idx, len(anoms) - 1))
        self._minor_frame_index = idx

        if idx >= 0:
            _entry = anoms[idx]
            frame_label = (f"frame {idx + 1}/{len(anoms)} — "
                           f"{_entry.get('imageName', '?')}")
            obs_iso = _entry.get("obs_iso")
            exptime = float(_entry.get("obs_exptime") or 0.0)
            obs_src = _entry.get("obs_source") or "unknown"
            utc_ok = bool(_entry.get("obs_utc_confirmed"))
        else:
            frame_label = "reference frame"
            obs_iso = payload.get("ref_obs_iso")
            exptime = float(payload.get("ref_obs_exptime") or 0.0)
            obs_src = payload.get("ref_obs_source") or "unknown"
            utc_ok = False

        t0 = None
        if obs_iso:
            try:
                t0 = Time(str(obs_iso), format="isot", scale="utc")
            except Exception:
                t0 = None

        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("Minor-body Search"))
        layout = QVBoxLayout(dlg)

        # ── Observation time ────────────────────────────────────────────
        tbox = QGroupBox(self.tr("Observation time — {0}").format(frame_label))
        tg = QGridLayout(tbox)
        rr = 0
        tg.addWidget(QLabel(self.tr("Header value:")), rr, 0)
        lbl_src = QLabel(f"{obs_iso or '— none found —'}")
        lbl_src.setStyleSheet("font-family: monospace;")
        tg.addWidget(lbl_src, rr, 1)
        tg.addWidget(QLabel(f"[{obs_src}]"), rr, 2)
        rr += 1

        # How to INTERPRET that timestamp. The FITS standard says DATE-OBS
        # is UTC, but plenty of capture software writes local wall-clock
        # time instead, and a DST boundary turns that into a silent
        # one-hour error. Nothing downstream can detect this, so the user
        # declares it here and watches the effective UTC update.
        tg.addWidget(QLabel(self.tr("Header time is:")), rr, 0)
        cmb_tz = QComboBox(dlg)
        cmb_tz.addItems([
            self.tr("UTC (as written)"),
            self.tr("Local time — this computer's timezone"),
            self.tr("Local time — named timezone (observing site)"),
            self.tr("Local time — fixed UTC offset"),
        ])
        _mode_default = 0
        if self.settings is not None:
            try:
                _mode_default = int(self.settings.value(
                    "supernova_hunter/header_time_mode", 0, type=int))
            except Exception:
                _mode_default = 0
        if utc_ok:
            # The header carries DATE-LOC separately, so DATE-OBS is UTC by
            # construction. A saved "local time" preference from before we
            # knew that would silently add hours, so proof wins over memory.
            _mode_default = 0
        cmb_tz.setCurrentIndex(max(0, min(3, _mode_default)))
        cmb_tz.setToolTip(
            "UTC (as written): trust DATE-OBS as UTC — the FITS standard.\n\n"
            "Local, this computer: convert using your system timezone, with\n"
            "    the DST rule that applied on the night of the exposure\n"
            "    (CDT vs CST, etc.) — not today's rule.\n\n"
            "Local, named timezone: for data from a site in a different\n"
            "    zone than this computer. Resolves that zone's own DST\n"
            "    rule at the date of the exposure, so it keeps working\n"
            "    across the next DST change.\n\n"
            "Local, fixed offset: a hard number, for a site that does not\n"
            "    observe DST."
        )
        tg.addWidget(cmb_tz, rr, 1, 1, 2)
        rr += 1

        tg.addWidget(QLabel(self.tr("    Site timezone:")), rr, 0)
        cmb_zone = QComboBox(dlg)
        cmb_zone.setEditable(True)
        try:
            from zoneinfo import available_timezones
            _zones = sorted(available_timezones())
        except Exception:
            _zones = []
        _common = ["America/Chicago", "America/New_York", "America/Denver",
                   "America/Phoenix", "America/Los_Angeles", "America/Santiago",
                   "Atlantic/Canary", "Europe/London", "Australia/Sydney", "UTC"]
        _ordered = ([z for z in _common if z in _zones]
                    + [z for z in _zones if z not in _common])
        cmb_zone.addItems(_ordered or _common)
        _zone_default = "America/Chicago"
        if self.settings is not None:
            try:
                _zone_default = str(self.settings.value(
                    "supernova_hunter/header_timezone",
                    _zone_default, type=str)) or _zone_default
            except Exception:
                pass
        _zi = cmb_zone.findText(_zone_default)
        cmb_zone.setCurrentIndex(_zi if _zi >= 0 else 0)
        cmb_zone.setToolTip(
            "The timezone the CAMERA was writing DATE-OBS in — the\n"
            "observing site's zone, which is not necessarily this\n"
            "computer's. DST is resolved for the night of the exposure."
        )
        tg.addWidget(cmb_zone, rr, 1, 1, 2)
        rr += 1

        tg.addWidget(QLabel(self.tr("    UTC offset of header:")), rr, 0)
        tz_spin = QDoubleSpinBox(dlg)
        tz_spin.setDecimals(2)
        tz_spin.setRange(-14.0, 14.0)
        tz_spin.setSingleStep(0.5)
        _tz_default = 0.0
        if self.settings is not None:
            try:
                _tz_default = float(self.settings.value(
                    "supernova_hunter/header_utc_offset_hours", 0.0, type=float))
            except Exception:
                _tz_default = 0.0
        tz_spin.setValue(_tz_default)
        tg.addWidget(tz_spin, rr, 1)
        lbl_tz = QLabel("")
        lbl_tz.setStyleSheet("font-family: monospace;")
        tg.addWidget(lbl_tz, rr, 2)
        rr += 1

        tg.addWidget(QLabel(self.tr("Exposure:")), rr, 0)
        tg.addWidget(QLabel(f"{exptime:g} s"), rr, 1)
        rr += 1

        chk_mid = QCheckBox(self.tr("Use mid-exposure"))
        chk_mid.setChecked(bool(getattr(self, "minor_use_midexposure", True)))
        chk_mid.setToolTip(
            "Evaluate the ephemeris at the middle of the exposure instead\n"
            f"of its start — worth {exptime / 2.0:.0f}s here.\n"
            "Untick to match tools that use DATE-OBS as written."
        )
        tg.addWidget(chk_mid, rr, 0, 1, 2)
        rr += 1

        tg.addWidget(QLabel(self.tr("Additional offset (hours):")), rr, 0)
        dt_spin = QDoubleSpinBox(dlg)
        dt_spin.setDecimals(2)
        dt_spin.setRange(-72.0, 72.0)
        dt_spin.setSingleStep(0.25)
        _off_default = float(getattr(self, "minor_time_offset_hours", 0.0))
        if self.settings is not None:
            try:
                _off_default = float(self.settings.value(
                    "wimi/minor/time_offset_hours", _off_default, type=float))
            except Exception:
                pass
        dt_spin.setValue(_off_default)
        dt_spin.setToolTip(
            "Added to the header time. Use it when the camera wrote local\n"
            "time instead of UTC, or when a DST slip puts you an hour out.\n\n"
            "This shares the same saved setting as WIMI, so an offset you\n"
            "established there carries over here and vice versa."
        )
        tg.addWidget(dt_spin, rr, 1)
        rr += 1

        tg.addWidget(QLabel(self.tr("→ Effective UTC:")), rr, 0)
        lbl_eff = QLabel("")
        lbl_eff.setStyleSheet("font-family: monospace; font-weight: bold; color:#2c7;")
        tg.addWidget(lbl_eff, rr, 1, 1, 2)
        rr += 1

        tg.addWidget(QLabel(self.tr("→ Julian Date:")), rr, 0)
        lbl_jd = QLabel("")
        lbl_jd.setStyleSheet("font-family: monospace; font-weight: bold; color:#2c7;")
        tg.addWidget(lbl_jd, rr, 1, 1, 2)
        rr += 1

        layout.addWidget(tbox)

        state = {"t": None, "tz": 0.0}

        def _tz_offset_hours() -> float:
            """The UTC offset the header timestamp is expressed in."""
            idx = cmb_tz.currentIndex()
            if idx == 0 or t0 is None:
                return 0.0
            if idx == 1:
                return _local_utc_offset_hours(t0)
            if idx == 2:
                return _named_tz_offset_hours(t0, cmb_zone.currentText())
            return float(tz_spin.value())

        def _recompute():
            idx = cmb_tz.currentIndex()
            cmb_zone.setEnabled(idx == 2)
            tz_spin.setEnabled(idx == 3)
            off_tz = _tz_offset_hours()
            state["tz"] = off_tz
            if idx == 0:
                lbl_tz.setText("UTC")
            else:
                sign = "+" if off_tz >= 0 else "−"
                note = {1: "   (this computer, DST-aware)",
                        2: f"   ({cmb_zone.currentText()}, DST-aware)",
                        3: ""}.get(idx, "")
                lbl_tz.setText(f"UTC{sign}{abs(off_tz):05.2f}{note}")
            if t0 is None:
                lbl_eff.setText("— unavailable —")
                lbl_jd.setText("—")
                state["t"] = None
                return
            t = t0
            if chk_mid.isChecked() and exptime > 0:
                t = t + (exptime / 2.0) * u.s
            # Local → UTC: subtract the zone the header was written in.
            t = t - off_tz * u.hour
            t = t + float(dt_spin.value()) * u.hour
            lbl_eff.setText(str(t.utc.isot))
            lbl_jd.setText(f"{float(t.utc.jd):.6f}")
            state["t"] = t

        chk_mid.toggled.connect(lambda _c: _recompute())
        dt_spin.valueChanged.connect(lambda _v: _recompute())
        cmb_tz.currentIndexChanged.connect(lambda _i: _recompute())
        tz_spin.valueChanged.connect(lambda _v: _recompute())
        cmb_zone.currentTextChanged.connect(lambda _t: _recompute())
        _recompute()

        # ── Astrometry ──────────────────────────────────────────────────
        abox = QGroupBox(self.tr("Astrometry"))
        ag = QVBoxLayout(abox)
        ok_wcs, why = _wcs_is_usable(self.ref_wcs, self.ref_header)
        lbl_wcs = QLabel(("Reference WCS: usable — " + why) if ok_wcs
                         else ("Reference WCS: NOT usable — " + why))
        lbl_wcs.setStyleSheet("color:#2c7;" if ok_wcs
                              else "color:#d66; font-weight: bold;")
        lbl_wcs.setWordWrap(True)
        ag.addWidget(lbl_wcs)
        chk_solve = QCheckBox(self.tr(
            "Plate solve the reference now  (in-house Gaia DR3 → ASTAP → Astrometry.net)"))
        chk_solve.setChecked(not ok_wcs)
        chk_solve.setToolTip(
            "Runs SASpro's own solver chain on the binned reference frame.\n"
            "Forced on when the header has no usable solution, because\n"
            "nothing downstream can work without one."
        )
        if not ok_wcs:
            chk_solve.setEnabled(False)
        ag.addWidget(chk_solve)
        layout.addWidget(abox)

        # ── Observer site ───────────────────────────────────────────────
        # Without this the ephemeris is computed from the centre of the
        # Earth. One Earth radius subtends ~4" on a main-belt asteroid at
        # 2 AU and ~18" at 0.5 AU, so for confirming a specific object
        # against a plate solution it is not a negligible term.
        sbox = QGroupBox(self.tr("Observer site  (topocentric parallax)"))
        sg = QGridLayout(sbox)
        _lat = _lon = _elev = 0.0
        _hdr_site = payload.get("site_from_header")
        if _hdr_site:
            _lat, _lon, _elev = _hdr_site
        if not _hdr_site and self.settings is not None:
            try:
                _lat = float(self.settings.value("site/latitude_deg", 0.0, type=float) or 0.0)
                _lon = float(self.settings.value("site/longitude_deg", 0.0, type=float) or 0.0)
                _elev = float(self.settings.value("site/elevation_m", 0.0, type=float) or 0.0)
            except Exception:
                pass

        sg.addWidget(QLabel(self.tr("Latitude °")), 0, 0)
        lat_spin = QDoubleSpinBox(dlg)
        lat_spin.setDecimals(5); lat_spin.setRange(-90.0, 90.0)
        lat_spin.setSingleStep(0.001); lat_spin.setValue(_lat)
        sg.addWidget(lat_spin, 0, 1)

        sg.addWidget(QLabel(self.tr("Longitude °")), 0, 2)
        lon_spin = QDoubleSpinBox(dlg)
        lon_spin.setDecimals(5); lon_spin.setRange(-180.0, 180.0)
        lon_spin.setSingleStep(0.001); lon_spin.setValue(_lon)
        lon_spin.setToolTip("East positive, west negative.")
        sg.addWidget(lon_spin, 0, 3)

        sg.addWidget(QLabel(self.tr("Elevation m")), 1, 0)
        elev_spin = QDoubleSpinBox(dlg)
        elev_spin.setDecimals(0); elev_spin.setRange(-500.0, 6000.0)
        elev_spin.setValue(_elev)
        sg.addWidget(elev_spin, 1, 1)

        chk_topo = QCheckBox(self.tr("Use topocentric positions"))
        chk_topo.setChecked(abs(_lat) > 1e-9 or abs(_lon) > 1e-9)
        chk_topo.setToolTip(
            "Compute positions from your actual location on Earth rather\n"
            "than from the geocentre.\n\n"
            "Worth ~4\" on a main-belt asteroid at 2 AU and ~18\" at\n"
            "0.5 AU. Leave unticked for geocentric positions."
        )
        sg.addWidget(chk_topo, 1, 2, 1, 2)
        layout.addWidget(sbox)

        # ── Catalog limits ──────────────────────────────────────────────
        lbox = QGroupBox(self.tr("Catalog limits"))
        row_layout = QGridLayout(lbox)
        layout.addWidget(lbox)

        ast_H_default = getattr(self, "minor_H_ast_max", 9.0)
        com_H_default = getattr(self, "minor_H_com_max", 10.0)
        ast_max_default = getattr(self, "minor_ast_max_count", 5000)
        com_max_default = getattr(self, "minor_com_max_count", 1000)

        row_layout.addWidget(QLabel("Asteroid H ≤"), 0, 0)
        ast_H_spin = QDoubleSpinBox(dlg)
        ast_H_spin.setDecimals(1); ast_H_spin.setRange(-5.0, 40.0)
        ast_H_spin.setSingleStep(0.1); ast_H_spin.setValue(ast_H_default)
        row_layout.addWidget(ast_H_spin, 0, 1)
        row_layout.addWidget(QLabel("Max asteroid"), 0, 2)
        ast_max_spin = QSpinBox(dlg)
        ast_max_spin.setRange(1, 2000000); ast_max_spin.setValue(ast_max_default)
        row_layout.addWidget(ast_max_spin, 0, 3)

        row_layout.addWidget(QLabel("Comet H ≤"), 1, 0)
        com_H_spin = QDoubleSpinBox(dlg)
        com_H_spin.setDecimals(1); com_H_spin.setRange(-5.0, 40.0)
        com_H_spin.setSingleStep(0.1); com_H_spin.setValue(com_H_default)
        row_layout.addWidget(com_H_spin, 1, 1)
        row_layout.addWidget(QLabel("Max comet"), 1, 2)
        com_max_spin = QSpinBox(dlg)
        com_max_spin.setRange(1, 200000); com_max_spin.setValue(com_max_default)
        row_layout.addWidget(com_max_spin, 1, 3)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        btn_row.addStretch(1)
        ok_btn = QPushButton("Search", dlg)
        ok_btn.setStyleSheet("font-weight: bold;")
        cancel_btn = QPushButton("Cancel", dlg)
        btn_row.addWidget(ok_btn); btn_row.addWidget(cancel_btn)

        def on_ok():
            self.minor_H_ast_max = float(ast_H_spin.value())
            self.minor_H_com_max = float(com_H_spin.value())
            self.minor_ast_max_count = int(ast_max_spin.value())
            self.minor_com_max_count = int(com_max_spin.value())
            hours = float(dt_spin.value())
            self.minor_time_offset_hours = hours
            self.minor_time_offset_days = hours / 24.0
            self.minor_use_midexposure = bool(chk_mid.isChecked())
            self.minor_header_time_mode = int(cmb_tz.currentIndex())
            self.minor_header_utc_offset = float(tz_spin.value())
            self.minor_header_timezone = str(cmb_zone.currentText())
            if chk_topo.isChecked():
                self.ref_site = (float(lat_spin.value()),
                                 float(lon_spin.value()),
                                 float(elev_spin.value()))
            else:
                self.ref_site = None
            self._minor_force_solve = bool(chk_solve.isChecked())
            # The JD shown in the dialog is the one we search at — no
            # recomputation downstream, so what the user verified on
            # screen is exactly what the ephemeris uses.
            t = state.get("t")
            self._minor_jd = float(t.utc.jd) if t is not None else None
            # Share the offset with WIMI through the same settings key.
            if self.settings is not None:
                try:
                    self.settings.setValue("wimi/minor/time_offset_hours", hours)
                    self.settings.setValue(
                        "supernova_hunter/header_time_mode",
                        int(cmb_tz.currentIndex()))
                    self.settings.setValue(
                        "supernova_hunter/header_utc_offset_hours",
                        float(tz_spin.value()))
                    self.settings.setValue(
                        "supernova_hunter/header_timezone",
                        str(cmb_zone.currentText()))
                    self.settings.setValue("site/latitude_deg", float(lat_spin.value()))
                    self.settings.setValue("site/longitude_deg", float(lon_spin.value()))
                    self.settings.setValue("site/elevation_m", float(elev_spin.value()))
                except Exception:
                    pass
            dlg.accept()

        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dlg.reject)
        return dlg.exec() == QDialog.DialogCode.Accepted

    def _on_minor_body_progress(self, pct: int, msg: str):
        self.status_label.setText(msg)
        self.minor_progress.setVisible(True)
        self.minor_progress.setValue(int(pct))
        # Mirror to the results window if it exists — the parent dialog
        # is normally covered by it during a search, so the bar on this
        # side is invisible to the user.
        if self._results_dialog is not None:
            try:
                self._results_dialog.minor_body_progress(int(pct), str(msg))
            except Exception:
                pass

    def _on_minor_body_finished(self, result, error: str):
        self.minor_progress.setValue(100 if not error else 0)
        self.minor_progress.setVisible(False)
        if self._results_dialog is not None:
            try:
                self._results_dialog.minor_body_end(
                    final_msg="Minor-body search failed." if error else "",
                    keep_status=bool(error),
                )
            except Exception:
                pass
        if error:
            QMessageBox.critical(self, "Minor-body Search",
                                 f"Minor-body prediction failed:\n{error}")
            self.status_label.setText("Minor-body search failed.")
            return
        if isinstance(result, dict):
            bodies = result.get("ref") or []
            per_frame = result.get("per_frame") or {}
        else:                                   # defensive: older shape
            bodies, per_frame = (result or []), {}

        self.predicted_minor_bodies = bodies
        self.minor_bodies_per_frame = per_frame
        if not self.predicted_minor_bodies:
            self.status_label.setText(
                "Minor-body search complete: no catalogued objects in field.")
            QMessageBox.information(
                self, "Minor-body Search",
                "No catalogued minor bodies were found in this field.")
            return
        n_tracked = sum(1 for v in per_frame.values() if v)
        self.status_label.setText(
            f"Minor-body search complete: {len(bodies)} object(s), "
            f"tracked across {n_tracked} frame(s).")
        self._append_log(
            f"Minor-body: {len(bodies)} in field; per-frame positions for "
            f"{n_tracked} frame(s)")

        if self._last_payload and self._last_payload.get("anomaly_data"):
            self._match_anomalies_to_minor_bodies(
                bodies, search_radius_arcsec=60.0, per_frame=per_frame,
            )
        if self._results_dialog is not None:
            try:
                self._last_payload["minor_bodies"] = bodies
                self._last_payload["minor_bodies_per_frame"] = per_frame
                self._results_dialog.update_minor_bodies(bodies, per_frame)
            except Exception as e:
                print("[Hunter] results dialog update failed:", e)

    def runMinorBodySearch(self):
        if self._last_payload is None:
            QMessageBox.warning(
                self, "Minor-body Search",
                "Run the hunter first — minor-body cross-match needs the\n"
                "reference image's WCS/JD.")
            return
        ref_path = self.parameters.get("referenceImagePath") or ""
        if not ref_path:
            QMessageBox.warning(self, "Minor-body Search",
                                "No reference image selected.")
            return
        if self.settings is None:
            QMessageBox.warning(
                self, "Minor-body Search",
                "Settings object is not available; cannot locate the "
                "minor-body database path.")
            return
        if not self._prompt_minor_body_limits():
            return

        # ---- Astrometry: make sure we have a REAL solution ----
        ok_solve, msg = self._ensure_wcs(
            force=bool(getattr(self, "_minor_force_solve", False)))
        self._append_log(f"Minor-body: WCS — {msg}")
        if not ok_solve:
            self.status_label.setText("Minor-body search aborted: no WCS.")
            QMessageBox.warning(
                self, "Minor-body Search",
                "Could not obtain an astrometric solution for the reference "
                f"frame.\n\n{msg}\n\n"
                "Check that ASTAP and/or an Astrometry.net API key are set up "
                "in SASpro's settings, or that the local Gaia DR3 index is "
                "installed.")
            return

        # ---- Time: use exactly the JD the user verified in the dialog ----
        jd_for_calc = getattr(self, "_minor_jd", None)
        if jd_for_calc is None:
            self.status_label.setText("Minor-body search aborted: no observation time.")
            QMessageBox.warning(
                self, "Minor-body Search",
                "No observation time is available for the reference frame.\n\n"
                "An ephemeris cannot be evaluated without DATE-OBS, MJD-OBS "
                "or JD in the header.")
            return
        _tz_mode = {0: "header=UTC", 1: "header=local (system tz)",
                    2: f"header=local ({getattr(self, 'minor_header_timezone', '?')})",
                    3: "header=local (fixed offset)"}.get(
                        int(getattr(self, "minor_header_time_mode", 0)), "?")
        self._append_log(
            f"Minor-body: evaluating at JD {jd_for_calc:.6f}  [{_tz_mode}, "
            f"extra offset {getattr(self, 'minor_time_offset_hours', 0.0):+.2f} h, "
            f"mid-exposure={'on' if getattr(self, 'minor_use_midexposure', True) else 'off'}]"
        )

        # self.ref_site was set by the dialog above — do not re-read it from
        # settings here, or an unticked "topocentric" box would be silently
        # overridden by whatever happens to be stored.
        if self.ref_site:
            self._append_log(
                f"Minor-body: topocentric from lat={self.ref_site[0]:.4f} "
                f"lon={self.ref_site[1]:.4f} elev={self.ref_site[2]:.0f}m")
        else:
            self._append_log("Minor-body: geocentric positions (no site set)")

        self.status_label.setText("Minor-body: starting background query…")
        self.minor_progress.setVisible(True)
        self.minor_progress.setValue(0)
        # Light up the mirrored strip in the results window so a user
        # looking at it (which they almost always are) sees activity.
        if self._results_dialog is not None:
            try:
                self._results_dialog.minor_body_begin(
                    "Minor-body: starting background query…")
            except Exception:
                pass

        _ad = self._last_payload.get("anomaly_data", []) or []
        _fi = int(getattr(self, "_minor_frame_index", -1))
        if 0 <= _fi < len(_ad):
            self._append_log(
                f"Minor-body: anchored on frame {_fi + 1}/{len(_ad)} "
                f"({_ad[_fi].get('imageName')}), not the reference")

        # Per-frame epochs, via the same conversion the dialog showed.
        frame_jds = []
        _missing = 0
        for entry in _ad:
            fj = self._frame_jd(entry.get("obs_iso"),
                                float(entry.get("obs_exptime") or 0.0))
            if fj is None:
                _missing += 1
            frame_jds.append((entry["path"], fj))
        if _missing:
            self._append_log(
                f"Minor-body: {_missing} frame(s) have no usable DATE-OBS — "
                "those keep the anchor frame's positions")

        self._mb_thread = QThread(self)
        self._mb_worker = MinorBodyWorker(self, jd_for_calc, frame_jds)
        self._mb_worker.moveToThread(self._mb_thread)
        self._mb_thread.started.connect(self._mb_worker.run)
        self._mb_worker.progress.connect(self._on_minor_body_progress)
        self._mb_worker.finished.connect(self._on_minor_body_finished)
        self._mb_worker.finished.connect(self._mb_thread.quit)
        self._mb_worker.finished.connect(self._mb_worker.deleteLater)
        self._mb_thread.finished.connect(self._mb_thread.deleteLater)
        self._mb_thread.start()

    def _frame_jd(self, obs_iso, exptime: float):
        """
        One frame's header timestamp -> JD, using exactly the rules the
        dialog displayed for the selected frame: mid-exposure, header
        timezone interpretation, and the extra manual offset.

        Deliberately a single code path. If per-frame epochs were derived
        any differently from the one the user verified on screen, every
        propagated position would carry a constant bias against it.
        """
        if not obs_iso:
            return None
        try:
            t = Time(str(obs_iso), format="isot", scale="utc")
        except Exception:
            return None
        try:
            if getattr(self, "minor_use_midexposure", True) and exptime > 0:
                t = t + (float(exptime) / 2.0) * u.s
            mode = int(getattr(self, "minor_header_time_mode", 0))
            if mode == 1:
                off = _local_utc_offset_hours(t)
            elif mode == 2:
                off = _named_tz_offset_hours(
                    t, getattr(self, "minor_header_timezone", ""))
            elif mode == 3:
                off = float(getattr(self, "minor_header_utc_offset", 0.0))
            else:
                off = 0.0
            t = t - off * u.hour
            t = t + float(getattr(self, "minor_time_offset_hours", 0.0)) * u.hour
            return float(t.utc.jd)
        except Exception:
            return None

    def _get_predicted_minor_bodies_for_field(
        self, H_ast_max, H_com_max, jd=None, progress_cb=None, frame_jds=None,
    ):
        """Needs ref_wcs and the reference image's shape — pulled from payload."""
        _EMPTY = {"ref": [], "per_frame": {}}
        if self.ref_wcs is None or self._last_payload is None:
            return _EMPTY
        ref_img = self._last_payload.get("ref_image")
        if ref_img is None:
            return _EMPTY

        def emit(pct, msg):
            if progress_cb is not None:
                try:
                    progress_cb(int(pct), msg)
                except TypeError:
                    progress_cb(msg)

        if jd is None:
            jd = self.ref_jd
        if jd is None or self.settings is None:
            return _EMPTY
        ast_limit = getattr(self, "minor_ast_max_count", 50000)
        com_limit = getattr(self, "minor_com_max_count", 5000)

        emit(5, "Minor-body search: opening minor-body database...")
        try:
            data_dir = Path(
                self.settings.value("wimi/minorbody_data_dir", "", type=str)
                or os.path.join(os.path.expanduser("~"), ".saspro_minor_bodies")
            )
            db_path, manifest = mbc.ensure_minor_body_db(data_dir)
            catalog = mbc.MinorBodyCatalog(db_path)
        except Exception as e:
            print("[MinorBodies] could not open DB:", e)
            emit(100, "Minor-body search: failed to open database.")
            return _EMPTY
        try:
            import pandas as pd

            emit(10, "Minor-body search: selecting bright asteroids/comets…")
            try:
                ast_df = catalog.get_bright_asteroids(H_max=H_ast_max, limit=ast_limit)
            except Exception as e:
                print("[MinorBodies] get_bright_asteroids failed:", e)
                ast_df = None
            try:
                com_df = catalog.get_bright_comets(H_max=H_com_max, limit=com_limit)
            except Exception as e:
                print("[MinorBodies] get_bright_comets failed:", e)
                com_df = None

            n_ast = 0 if ast_df is None else len(ast_df)
            n_com = 0 if com_df is None else len(com_df)
            print(f"[MinorBodies] candidate rows: asteroids={n_ast} comets={n_com}")
            if n_ast == 0 and n_com == 0:
                emit(100, "Minor-body search: no catalog rows matched those limits.")
                return _EMPTY

            # Designation → kind / magnitude maps, built before the concat
            # so each computed position can be labelled afterwards.
            ast_names, com_names = set(), set()
            mag_by_name = {}
            for df, names, mag_keys in (
                (ast_df, ast_names, ("magnitude_H", "H", "absolute_magnitude")),
                (com_df, com_names, ("absolute_magnitude", "magnitude_H", "H",
                                     "magnitude_g", "magnitude_k")),
            ):
                if df is None or not len(df):
                    continue
                for _, row in df.iterrows():
                    desig = row.get("designation", "") or row.get("name", "")
                    if not desig:
                        continue
                    names.add(desig)
                    for mk in mag_keys:
                        if mk in row and row[mk] is not None:
                            try:
                                mag_by_name[desig] = float(row[mk])
                            except Exception:
                                pass
                            break

            frames = [d for d in (ast_df, com_df) if d is not None and len(d)]
            rows_df = pd.concat(frames, ignore_index=True)
            total_rows = len(rows_df)

            # NOTE: this is compute_positions_ASTROPY, matching what WIMI
            # calls. The previous code called compute_positions_skyfield,
            # which does not exist on MinorBodyCatalog at all — the only
            # position method is the astropy/Kepler one. Every search
            # therefore died with AttributeError the moment it got here,
            # after sitting through the slow catalog query first.
            #
            # The propagation is a per-row Python loop, so with a generous
            # H limit it is by far the slowest step. Report real progress
            # through it rather than leaving the bar frozen.
            def _pos_progress(done, total):
                if total:
                    emit(15 + int(70.0 * done / float(total)),
                         f"Minor-body search: propagating orbits {done}/{total}…")
                return True

            emit(15, f"Minor-body search: propagating {total_rows} orbits…")
            positions = catalog.compute_positions_astropy(
                asteroid_rows=rows_df,
                jd=jd,
                ephemeris_path=str(data_dir),
                topocentric=self.ref_site,
                progress_cb=_pos_progress,
                debug=False,
            )
            print(f"[MinorBodies] positions computed: {len(positions)}")

            emit(92, "Minor-body search: projecting onto image pixels…")
            h, w = ref_img.shape[:2]
            bodies = []
            for pos in positions:
                try:
                    ra = float(pos["ra_deg"]); dec = float(pos["dec_deg"])
                    x, y = self.ref_wcs.world_to_pixel_values(ra, dec)
                    x = float(x); y = float(y)
                except Exception:
                    continue
                if not (0 <= x < w and 0 <= y < h):
                    continue
                desig = pos.get("designation", "") or "Unknown"
                kind = ("asteroid" if desig in ast_names
                        else "comet" if desig in com_names else "minor body")
                bodies.append({
                    "designation": desig,
                    "kind": kind,
                    "ra_deg": ra, "dec_deg": dec,
                    "x": x, "y": y,
                    "H": float(mag_by_name.get(desig, np.nan)),
                    "distance_au": pos.get("distance_au", np.nan),
                    "elements": None,     # filled below for in-FOV bodies only
                })
            print(f"[MinorBodies] inside frame: {len(bodies)} of {len(positions)}")

            # Attach Keplerian elements for the 3D Orbit Explorer — only for
            # the handful of bodies that actually landed in the frame, so we
            # never materialise the whole (tens-of-thousands-row) catalog
            # into dicts. Elements live on rows_df under the same column
            # names compute_positions_astropy reads (semimajor_axis_au,
            # mean_anomaly_degrees, …); orbit_explorer.extract_elements_from_row
            # knows them.
            try:
                in_fov = {b["designation"] for b in bodies}
                if in_fov and "designation" in rows_df.columns:
                    sub = rows_df[rows_df["designation"].isin(in_fov)]
                    row_by_desig = {r.get("designation"): r
                                    for r in sub.to_dict("records")}
                    n_with = 0
                    for b in bodies:
                        el = extract_elements_from_row(
                            row_by_desig.get(b["designation"]))
                        b["elements"] = el
                        n_with += 1 if el else 0
                    print(f"[MinorBodies] orbital elements attached to "
                          f"{n_with}/{len(bodies)} in-frame bodies")
                    if n_with < len(bodies):
                        # Surface the real column names once so the aliases in
                        # orbit_explorer can be reconciled if a schema changes.
                        try:
                            from setiastro.saspro import orbit_explorer as _oemod
                            print("[MinorBodies] (no elements for some) catalog "
                                  f"columns: {getattr(_oemod, '_last_available_keys', [])}")
                        except Exception:
                            pass
            except Exception as _e:
                print("[MinorBodies] element attachment failed:", _e)

            # ---------------------------------------------------------------
            # Propagate the FOUND bodies to every other frame's epoch.
            #
            # Discovery above is unchanged — it answers "what is in this
            # field" at the selected frame's time. But each frame was taken
            # at a different moment and the object has moved between them;
            # that drift is the entire signal being hunted. Re-propagating
            # just this handful of bodies is cheap next to the full-catalog
            # pass that produced them.
            # ---------------------------------------------------------------
            per_frame = {}
            in_fov = {b["designation"] for b in bodies}
            if frame_jds and in_fov and "designation" in rows_df.columns:
                sub = rows_df[rows_df["designation"].isin(in_fov)]
                n_f = len(frame_jds)
                print(f"[MinorBodies] propagating {len(sub)} body(s) to {n_f} frames")
                for k, (fpath, fjd) in enumerate(frame_jds, 1):
                    if fjd is None:
                        continue
                    emit(92 + int(7.0 * k / max(1, n_f)),
                         f"Minor-body search: positions for frame {k}/{n_f}…")
                    try:
                        fpos = catalog.compute_positions_astropy(
                            asteroid_rows=sub, jd=float(fjd),
                            ephemeris_path=str(data_dir),
                            topocentric=self.ref_site, debug=False,
                        )
                    except Exception as exc:
                        print(f"[MinorBodies] frame {k} failed: {exc!r}")
                        continue
                    lst = []
                    for pos in fpos:
                        try:
                            ra = float(pos["ra_deg"]); dec = float(pos["dec_deg"])
                            x, y = self.ref_wcs.world_to_pixel_values(ra, dec)
                        except Exception:
                            continue
                        desig = pos.get("designation", "") or "Unknown"
                        lst.append({
                            "designation": desig,
                            "kind": ("asteroid" if desig in ast_names
                                     else "comet" if desig in com_names
                                     else "minor body"),
                            "ra_deg": ra, "dec_deg": dec,
                            "x": float(x), "y": float(y),
                            "H": float(mag_by_name.get(desig, np.nan)),
                            "distance_au": pos.get("distance_au", np.nan),
                        })
                    per_frame[fpath] = lst

            emit(100, f"Minor-body search: {len(bodies)} object(s) inside the frame.")
            return {"ref": bodies, "per_frame": per_frame}
        finally:
            try:
                catalog.close()
            except Exception:
                pass

    def _match_anomalies_to_minor_bodies(self, bodies, search_radius_arcsec=20.0,
                                         per_frame=None):
        """
        Cross-match detections against predicted positions, each frame
        against ITS OWN epoch where available. Over a night of subs a
        main-belt object drifts far beyond any sane match radius, so
        matching every frame to one epoch identifies it near that epoch
        and quietly fails everywhere else.
        """
        if self.ref_wcs is None or not bodies or self._last_payload is None:
            return
        per_frame = per_frame or {}
        try:
            cd = self.ref_wcs.pixel_scale_matrix
            from numpy.linalg import det
            deg_per_pix = np.sqrt(abs(det(cd)))
            arcsec_per_pix = deg_per_pix * 3600.0
        except Exception:
            arcsec_per_pix = 1.0
        pix_radius = search_radius_arcsec / arcsec_per_pix
        default_bodies = bodies
        for entry in self._last_payload["anomaly_data"]:
            # New name on purpose: rebinding `bodies` here would leak one
            # frame's positions into the next frame that lacks its own.
            frame_bodies = per_frame.get(entry.get("path")) or default_bodies
            for anomaly in entry["anomalies"]:
                cx = 0.5 * (anomaly["minX"] + anomaly["maxX"])
                cy = 0.5 * (anomaly["minY"] + anomaly["maxY"])
                matches = []
                for body in frame_bodies:
                    dx = body["x"] - cx
                    dy = body["y"] - cy
                    r_pix = np.hypot(dx, dy)
                    if r_pix <= pix_radius:
                        matches.append((r_pix, body))
                if matches:
                    matches.sort(key=lambda t: t[0])
                    anomaly["matched_body"] = matches[0][1]
                    anomaly["matched_bodies"] = [b for _, b in matches]
                else:
                    anomaly["matched_body"] = None
                    anomaly["matched_bodies"] = []

    # ---- New Instance ---------------------------------------------------

    def newInstance(self):
        self.parameters = {
            "referenceImagePath": "",
            "searchImagePaths": [],
        }
        self.ref_line_edit.clear()
        self.search_list.clear()
        self.cosmetic_checkbox.setChecked(False)
        self.align_checkbox.setChecked(True)
        self.adbe_checkbox.setChecked(True)
        self.adbe_degree.setValue(2)
        self.adbe_rbf.setChecked(True)
        self.sigma_spin.setValue(5.0)
        self.minarea_spin.setValue(5)
        self.maxdim_spin.setValue(200)
        self.psf_spin.setValue(1.5)
        self.border_spin.setValue(5.0)
        self.bin_spin.setValue(2)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Idle")
        self.log_view.clear()

        self._last_payload = None
        self._redetect_reopen = False
        self.redetect_button.setEnabled(False)
        if self._results_dialog is not None:
            try:
                self._results_dialog.close()
            except Exception:
                pass
            self._results_dialog = None

        self.ref_header = None
        self.ref_wcs = None
        self.ref_jd = None
        self.ref_site = None
        self.predicted_minor_bodies = None
        self.minor_bodies_per_frame = {}
        self._minor_frame_index = -1
        self._minor_jd = None
        self._minor_force_solve = False

        QMessageBox.information(self, "New Instance", "Reset for a new instance.")

    # ---- Cleanup --------------------------------------------------------

    def closeEvent(self, ev):
        if self._pipeline_thread is not None and self._pipeline_thread.isRunning():
            self._pipeline_thread.request_cancel()
            self._pipeline_thread.wait(1500)
        super().closeEvent(ev)