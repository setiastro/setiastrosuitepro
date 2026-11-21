# ExoPlanet Detector (SASpro) — standalone plate solving, no WIMI

from __future__ import annotations

from setiastro.saspro.main_helpers import non_blocking_sleep

import os
import shutil
import tempfile
import time
from typing import List, Tuple, Set
import webbrowser
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote
from types import SimpleNamespace
import math
import numpy as np
import pandas as pd
import sep
import pyqtgraph as pg
import matplotlib.pyplot as plt
from types import SimpleNamespace
from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.timeseries import LombScargle, BoxLeastSquares
import re

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from astroquery.mast import Tesscut

from lightkurve import TessTargetPixelFile

import lightkurve as lk

# ---- project-local imports (adjust paths if needed) --------------------
from setiastro.saspro.legacy.numba_utils import bin2x2_numba, apply_flat_division_numba
from setiastro.saspro.imageops.stretch import stretch_mono_image, stretch_color_image
from setiastro.saspro.help_support import make_help_button
from setiastro.saspro.plate_solver import plate_solve_doc_inplace
from setiastro.saspro.star_alignment import (
    StarRegistrationWorker,
    StarRegistrationThread,
    IDENTITY_2x3,
)
from setiastro.saspro.legacy.image_manager import load_image, save_image, get_valid_header  # adjust if different
from setiastro.saspro import photometry_aavso as aav
from setiastro.saspro import cds_health
from setiastro.saspro import exoplanet_photometry as xphot
from setiastro.saspro import exoplanet_transit as xfit
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn

# ------------------------------------------------------------------------
from setiastro.saspro.legacy.xisf import XISF
from PyQt6.QtCore import Qt, QTimer, QSettings, QRectF, QPoint, QPointF, QByteArray
from PyQt6.QtGui import QIcon, QColor, QBrush, QPen, QPainter, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QApplication, QGraphicsView, QGraphicsPixmapItem,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QListWidgetItem, QMessageBox, QPushButton, QProgressBar, QRadioButton, QSpinBox, QDoubleSpinBox,
    QSlider, QToolButton, QVBoxLayout, QInputDialog, QLineEdit, QCheckBox,
    QStackedWidget, QWidget, QSplitter, QProgressDialog
)
import pyqtgraph as pg

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings("ignore", category=AstropyWarning, message=".*more axes.*")


# ───────────────────────── CDS / SIMBAD hardening ─────────────────────────
#
# Everything in this module that touches SIMBAD or VizieR runs on the GUI
# thread, and until now none of it bounded its timeouts. astroquery with no
# timeout set falls back to the OS TCP connect timeout -- roughly 130 s on
# Linux -- and the lookup helpers below each retried five times. A CDS
# outage therefore froze the window for ~11 minutes per lookup with no
# cancel and no progress.
#
# Three things fix that:
#   1. A bounded timeout, set on BOTH spellings (`TIMEOUT` was renamed to
#      `timeout` in astroquery 0.4.8; setting only the old name on a new
#      astroquery silently does nothing, which is how an unbounded call
#      sneaks through).
#   2. A cds_health preflight, so we learn CDS is down in ~4 s instead of
#      ~130 s, and the session-wide breaker makes every later lookup return
#      instantly.
#   3. No retrying of connection-class failures. If the host is not
#      answering, a second attempt just buys the same timeout again.
#
# NOTE ON VizieR / VSX: the VSX lookups in this module are not an
# independent fallback for a SIMBAD outage. B/vsx is a VizieR catalog, and
# VizieR is CDS. It runs on separate hosts from the SIMBAD TAP endpoint, so
# it survives a SIMBAD-only fault and is worth trying -- but a Strasbourg-
# wide outage takes both down, so these calls need the same bounding.

_SIMBAD_TIMEOUT_S = 20      # per-call ceiling for interactive lookups
_VIZIER_TIMEOUT_S = 20


def _configure_simbad_endpoint():
    """
    Point astroquery at whichever mirror cds_health last found healthy and
    bound its timeout. Returns the hostname in use.

    Safe to call repeatedly; every assignment is individually guarded so a
    version that lacks one of these attributes doesn't break the others.
    """
    host = cds_health.current_server()
    applied = cds_health.apply_server(host)
    for setter in (
        lambda: setattr(Simbad, "timeout", int(_SIMBAD_TIMEOUT_S)),
        lambda: setattr(Simbad, "TIMEOUT", int(_SIMBAD_TIMEOUT_S)),
    ):
        try:
            setter()
        except Exception:
            pass
    if not applied:
        print(f"[DEBUG] astroquery refused SIMBAD server {host!r} — "
              f"queries would go to the default host instead.")
    return host, applied


def _new_simbad():
    """
    Build a fresh, bounded Simbad object for a one-off lookup.

    Why this exists rather than just setting Simbad.TIMEOUT once:
    astroquery exports `Simbad` as a module-level *instance* of
    SimbadClass, and the lookups in this module each build their own with
    `Simbad()`. An attribute set on the module-level object does not
    propagate to a newly constructed one -- the new object reads its
    default from `conf`. So a bound set only on the module-level object
    silently fails to apply to exactly the calls we are trying to bound.

    Belt and braces: `_configure_simbad_endpoint()` sets `conf.timeout`
    (which new instances inherit) AND this sets the attribute directly on
    the instance we are about to use, under both spellings.
    """
    host, applied = _configure_simbad_endpoint()
    s = Simbad()
    if applied:
        try:
            s.server = host
        except Exception:
            pass
    for setter in (
        lambda: setattr(s, "timeout", int(_SIMBAD_TIMEOUT_S)),
        lambda: setattr(s, "TIMEOUT", int(_SIMBAD_TIMEOUT_S)),
    ):
        try:
            setter()
        except Exception:
            pass
    try:
        s.reset_votable_fields()
    except Exception:
        pass
    return s


def _vizier_effective_url(v) -> str:
    """The URL this Vizier client will actually hit.

    Ground truth rather than attribute-guessing: astroquery builds the
    request from its own server field, so reading back the attribute we
    just set would confirm nothing when that attribute is the dead one.
    Returns "" if the private helper is not available.
    """
    fn = getattr(v, "_server_to_url", None)
    if callable(fn):
        for args in ((), ("votable",)):
            try:
                return str(fn(*args))
            except Exception:
                continue
    return ""


def _apply_vizier_server(v, host: str) -> bool:
    """Point a Vizier client at `host` under every spelling astroquery has
    used, and report whether it actually took.

    astroquery renamed this exactly the way it renamed TIMEOUT -> timeout:
    the legacy class attribute `VIZIER_SERVER` became a `server` property
    backed by `conf.server`. Setting only the old name creates a dead
    attribute while queries keep going to the default host -- silently,
    which is how a mirror selection can look applied and do nothing. That
    is why the VizieR mirror dropdown appeared to have no effect during
    the CDS outage.

    conf.server is set too, not just the instance: new Vizier() objects
    inherit their server from conf, and the module-level `Vizier` singleton
    (which magnitude_tool's APASS query uses) reads the same place.
    """
    if not host:
        return False
    # Never for an http-only mirror (NAOJ): astroquery accepts the name, and
    # every fresh Vizier() would then try it over https and time out.
    if host not in cds_health.VIZIER_HTTP_ONLY:
        try:
            import astroquery.vizier as _aqv
            try:
                _aqv.conf.server = host
            except Exception:
                pass
        except Exception:
            pass
    # Sets server / _server / VIZIER_SERVER, plus the http:// scheme for the
    # mirrors that serve nothing else, and checks the URL that results.
    applied = cds_health.point_vizier(v, host)
    url = _vizier_effective_url(v)
    if url and not applied:
        print(f"[DEBUG] VizieR mirror {host!r} did NOT take -- client will "
              f"query {url!r}")
    return applied


def _new_vizier(**kwargs):
    """
    Build a bounded Vizier client pointed at the chosen VizieR mirror.

    VizieR runs on different hosts from the SIMBAD TAP endpoint, so it
    survives a SIMBAD-only fault -- but vizier.cds.unistra.fr is still a
    Strasbourg host, so a CDS-wide outage takes it down too. Hence a
    mirror here as well, and the same bounded timeout either way.
    """
    kwargs.setdefault("timeout", int(_VIZIER_TIMEOUT_S))
    v = Vizier(**kwargs)
    # The PROBED mirror, not the preference. vizier_server() is "preference,
    # else Strasbourg" and never checks health, so with no mirror pinned in
    # Settings every lookup went to vizier.cds.unistra.fr -- which has been
    # answering 404 for weeks while vizier.idia.ac.za served the same
    # queries. prepare_vizier races the mirrors (cached for minutes).
    try:
        _host = (cds_health.prepare_vizier(timeout=_VIZIER_TIMEOUT_S)
                 or cds_health.vizier_server())
    except Exception:
        _host = ""
    # Assigning VIZIER_SERVER alone is a no-op on current astroquery; see
    # _apply_vizier_server for why, and for what it sets instead.
    _apply_vizier_server(v, _host)
    return v


def _vizier_query_region(coord, radius, catalog, *, columns, row_limit=50,
                         context="", require_rows=False, status_cb=None):
    """
    query_region on the first VizieR mirror that serves it, failing over
    across cds_health's mirror list (probe -> query -> on failure re-race to
    another mirror and repeat).  Returns (TableList, host).  Raises when no
    mirror answered.  An empty TableList is returned as-is -- mirrors are
    partial, so callers that expect rows should name the mirror in their
    failure note.
    """
    def _run(host):
        v = cds_health.vizier_client(host, columns=list(columns),
                                     row_limit=int(row_limit),
                                     timeout=int(_VIZIER_TIMEOUT_S))
        return v.query_region(coord, radius=radius, catalog=catalog)
    return cds_health.query_vizier(_run, context=context or f"exoplanet/{catalog}",
                                   timeout=_VIZIER_TIMEOUT_S, require_rows=require_rows,
                                   status_cb=status_cb)


def _simbad_region_direct(ra_deg, dec_deg, radius_arcsec, timeout=12.0):
    """
    SIMBAD cone search over plain HTTP against each mirror's TAP endpoint --
    WIMI's last resort.  astroquery wraps pyvo, which has its own session and
    job handling, so "astroquery raised" and "SIMBAD is down" are different
    statements; this path sometimes answers when that one will not.
    Returns an astropy Table with main_id, ra, dec, otype, V -- or None.
    """
    r = float(radius_arcsec) / 3600.0
    adql = ("SELECT TOP 50 b.main_id, b.ra, b.dec, b.otype, f.V "
            "FROM basic AS b LEFT JOIN allfluxes AS f ON f.oidref = b.oid "
            f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', "
            f"{float(ra_deg):.7f}, {float(dec_deg):.7f}, {r:.7f})) = 1")
    for host in dict.fromkeys((cds_health.current_server(), *cds_health.MIRRORS)):
        try:
            t = cds_health.query_tap_direct(adql, server=host, timeout=timeout)
        except Exception:
            t = None
        if t is not None:
            print(f"[DEBUG] SIMBAD answered directly via {host}")
            return t
    return None


def _configure_vizier_timeout():
    """Bound the module-level Vizier object and point it at the mirror."""
    host = cds_health.prepare_vizier(timeout=_VIZIER_TIMEOUT_S) or cds_health.vizier_server()
    try:
        import astroquery.vizier as _aqv
        try:
            _aqv.conf.server = host
        except Exception:
            pass
    except Exception:
        pass
    for setter in (
        lambda: setattr(Vizier, "TIMEOUT", int(_VIZIER_TIMEOUT_S)),
        lambda: setattr(Vizier, "timeout", int(_VIZIER_TIMEOUT_S)),
    ):
        try:
            setter()
        except Exception:
            pass
    # VIZIER_SERVER on its own is a no-op on current astroquery.
    _apply_vizier_server(Vizier, host)


def _extract_ra_dec_from_header(h: fits.Header):
    """Return (ra_deg, dec_deg) if found, else (None, None)."""
    if not isinstance(h, fits.Header):
        return None, None

    # 1) If WCS is already present, use its center
    try:
        w = WCS(h)
        if w.has_celestial:
            # center of the current pixel grid
            nx = h.get("NAXIS1"); ny = h.get("NAXIS2")
            if nx and ny:
                sky = w.pixel_to_world(nx/2, ny/2)
                return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        pass

    # 2) Common RA/DEC keyword pairs
    pairs = [
        ("OBJCTRA", "OBJCTDEC"),  # PixInsight/ASCOM style (strings)
        ("RA",      "DEC"),
        ("TELRA",   "TELDEC"),
        ("RA_OBJ",  "DEC_OBJ"),
        ("CAT-RA",  "CAT-DEC"),
        ("RA_DEG",  "DEC_DEG"),   # degrees already
    ]

    for rak, deck in pairs:
        if rak in h and deck in h:
            ra_raw, dec_raw = h[rak], h[deck]

            # Try a few parse paths
            for parser in (
                lambda r,d: SkyCoord(r, d, unit=(u.hourangle, u.deg)),
                lambda r,d: SkyCoord(float(r)*u.deg, float(d)*u.deg),
                lambda r,d: SkyCoord(r, d, unit=(u.deg, u.deg)),
            ):
                try:
                    c = parser(ra_raw, dec_raw)
                    return float(c.ra.deg), float(c.dec.deg)
                except Exception:
                    pass

    return None, None


def _estimate_scale_arcsec_per_pix(h: fits.Header):
    """Return pixel scale (arcsec/pix) if derivable, else None."""
    if not isinstance(h, fits.Header):
        return None

    # Direct scale keywords (various conventions)
    for k in ("PIXSCALE", "PIXSCL", "SECPIX", "SECPIX1"):
        if k in h:
            try:
                val = float(h[k])
                if val > 0:
                    return val
            except Exception:
                pass

    # Derive from pixel size & focal length:
    # scale["/pix] ≈ 206.265 * pixel_size_μm / focal_length_mm
    px_um = None
    for k in ("XPIXSZ", "PIXSIZE1", "PIXSIZE"):  # μm
        if k in h:
            try:
                px_um = float(h[k])
                break
            except Exception:
                pass

    foc_mm = None
    for k in ("FOCALLEN", "FOCLEN", "FOCALLENGTH"):
        if k in h:
            try:
                foc_mm = float(h[k])
                break
            except Exception:
                pass

    if px_um and foc_mm and foc_mm > 0:
        return 206.265 * px_um / foc_mm

    return None

_TZ_RE = re.compile(r'([+-])(\d{2})(\d{2})$')  # -0700 -> -07:00

def _fix_iso_tz(s: str) -> str:
    s = s.strip()
    m = _TZ_RE.search(s)
    if m:
        s = s[:m.start()] + f"{m.group(1)}{m.group(2)}:{m.group(3)}"
    return s

def _parse_obs_time_from_header(hdr) -> Time | None:
    # hdr can be fits.Header or your dict-ish header
    def _get(key):
        try:
            return hdr.get(key)
        except Exception:
            return None

    # 1) Prefer UT-OBS if present (already “UTC-ish”)
    for key in ("UT-OBS", "DATE-OBS", "DATE-END"):
        v = _get(key)
        if isinstance(v, str) and v.strip():
            try:
                return Time(_fix_iso_tz(v), format="isot", scale="utc")
            except Exception:
                pass

    # 2) MJD-OBS is super reliable
    v = _get("MJD-OBS")
    if v is not None:
        try:
            return Time(float(v), format="mjd", scale="utc")
        except Exception:
            pass

    return None

def _estimate_fov_deg(img_shape, scale_arcsec):
    """Rough FOV (deg) from image size and scale (max of X/Y)."""
    try:
        h, w = img_shape[:2]
        if scale_arcsec and h and w:
            fov_x = (w * scale_arcsec) / 3600.0
            fov_y = (h * scale_arcsec) / 3600.0
            return float(max(fov_x, fov_y))
    except Exception:
        pass
    return None


def _build_astrometry_hints(hdr: fits.Header, plane: np.ndarray):
    """Compose a hints dict for the solver."""
    ra_deg, dec_deg = _extract_ra_dec_from_header(hdr)
    scale = _estimate_scale_arcsec_per_pix(hdr)
    fov   = _estimate_fov_deg(plane.shape, scale)

    # A generous search radius: ≥1°, or 3×FOV if we have it
    radius = None
    if fov is not None:
        radius = max(1.0, 3.0 * fov)

    hints = {}
    if ra_deg is not None and dec_deg is not None:
        hints["ra_deg"]  = ra_deg
        hints["dec_deg"] = dec_deg
    if scale is not None:
        hints["pixel_scale_arcsec"] = scale
    if fov is not None:
        hints["fov_deg"] = fov
    if radius is not None:
        hints["search_radius_deg"] = radius

    # Optional: parity if you know you’re mirrored or not (None = let solver decide)
    # hints["parity"] = +1  # or -1

    return hints

# ── Photometric gain ─────────────────────────────────────────────────────
#
# sep's `gain` converts DATA UNITS to Poisson counts: it adds flux/gain to
# the variance. load_image divides by the source bit depth, so one data
# unit is 65535 ADU for 16-bit input, not one electron. The old gain=1.0
# therefore claimed a star with ~0.1 units of total aperture flux carried
# ~0.1 electrons -- sqrt(0.1)/0.1 = 3.2, i.e. a 3.5 mag error on every
# measurement. That is what pushed MERR past AAVSO's 1.0 ceiling and wrote
# "na" on every row while the magnitudes themselves stayed correct.
#
#     gain_eff = EGAIN[e-/ADU] * ADU_per_unit * b^2
#
# The b^2 is because the b x b software bin AVERAGES: a binned pixel of
# value v holds the electrons of b^2 unbinned pixels averaging v, so b^2
# times as many electrons sit behind each unit of the binned array.
#
# Note the noise model this completes: err=rmsmap already carries sky and
# read noise (sep's background rms measures exactly that), and because `ds`
# is background-subtracted the gain term adds only the star's own shot
# noise. The two are complementary, which is why the annulus call needs the
# same gain rather than none.

_ADU_PER_UNIT = {
    "8-bit":           255.0,
    "16-bit":          65535.0,
    "16-bit signed":   65535.0,
    "32-bit unsigned": 4294967295.0,
    "32-bit signed":   4294967295.0,
}

# ASCOM/INDI report true e-/ADU as EGAIN. A bare GAIN keyword is usually
# the CMOS gain *setting* (0-500 on ZWO/QHY), not e-/ADU, so it is left out
# on purpose -- using it here would be wrong by roughly 100x.
_EGAIN_KEYS = ("EGAIN", "GAINE", "E_GAIN", "ELECGAIN")


def _header_scalar(hdr, keys):
    """First of `keys` present in a fits.Header or XISF-style dict, as float."""
    if hdr is None:
        return None
    if isinstance(hdr, fits.Header):
        for k in keys:
            if k in hdr:
                try:
                    return float(hdr[k])
                except (TypeError, ValueError):
                    pass
        return None
    if isinstance(hdr, dict):
        kw = None
        im = hdr.get("image_meta")
        if isinstance(im, dict):
            kw = im.get("FITSKeywords")
        if kw is None:
            kw = hdr.get("FITSKeywords")
        for k in keys:
            v = None
            if isinstance(kw, dict) and k in kw:
                ent = kw[k]
                v = ent[0].get("value") if isinstance(ent, list) and ent else ent
            elif k in hdr:
                v = hdr[k]
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        for sub in ("fits_header", "original_header", "header"):
            h = hdr.get(sub)
            if isinstance(h, fits.Header):
                r = _header_scalar(h, keys)
                if r is not None:
                    return r
    return None


def _rescale_header_for_binning(hdr, bin_factor: int = 2):
    """
    Return a copy of `hdr` describing the same sky on a `bin_factor`-binned
    pixel grid.

    _solve_reference hands the plate solver a 2x2 binned plane, but a header
    straight off the file describes the unbinned frame. plate_solver reads
    it for the pixel-scale hint it cannot solve without
    (_estimate_scale_arcsec_from_header: WCS/CD, then CDELT, then
    PIXSCALE-style keys, then XPIXSZ * XBINNING / FOCALLEN), so uncorrected
    it is told the scale is half what it really is.

    What changes, and why:

      CRPIX        (p - 0.5)/b + 0.5
                   FITS numbers pixel CENTRES from 1, so the outer edge of
                   pixel 1 sits at 0.5. Binning maps edges, not centres;
                   dividing the centre directly is off by half a binned
                   pixel.
      CD, CDELT    x b   one binned pixel spans b unbinned ones.
      PIXSCALE,
      SECPIX, ...  x b   the same statement, made directly in arcsec/px.
      XBINNING,
      YBINNING     x b   this is where the solver expects a binning factor
                         to live: it computes XPIXSZ * XBINNING / FOCALLEN.
                         Scaling XPIXSZ as well would square the correction.

    SIP is dropped rather than rescaled. Those coefficients carry units of
    unbinned pixels to the (p+q)th power, so a correct rescale is
    A_pq * b**(p + q - 1) -- easy to get subtly wrong, and the solver refits
    SIP from scratch regardless. A seed with no distortion model is fine; a
    seed with a wrong one is not.
    """
    if not isinstance(hdr, fits.Header):
        return hdr
    b = float(max(1, int(bin_factor)))
    out = hdr.copy()
    if b == 1.0:
        return out

    for k in ("CRPIX1", "CRPIX2"):
        if k in out:
            try:
                out[k] = (float(out[k]) - 0.5) / b + 0.5
            except (TypeError, ValueError):
                pass

    for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2",
              "PIXSCALE", "SECPIX", "SECPIX1", "SECPIX2",
              "SCALE", "PLATESCL", "PLTSCALE"):
        if k in out:
            try:
                out[k] = float(out[k]) * b
            except (TypeError, ValueError):
                pass

    _had_bin = False
    for k in ("XBINNING", "YBINNING", "XBIN", "YBIN"):
        if k in out:
            try:
                out[k] = int(round(float(out[k]) * b))
                _had_bin = True
            except (TypeError, ValueError):
                pass
    if not _had_bin and any(k in out for k in ("XPIXSZ", "YPIXSZ",
                                               "PIXSIZE1", "PIXSIZE2")):
        out["XBINNING"] = int(b)
        out["YBINNING"] = int(b)

    for k in [k for k in list(out.keys())
              if str(k).upper().startswith(("A_", "B_", "AP_", "BP_"))]:
        try:
            del out[k]
        except Exception:
            pass
    for k in ("CTYPE1", "CTYPE2"):
        if k in out:
            try:
                out[k] = str(out[k]).replace("-SIP", "")
            except Exception:
                pass

    try:
        out.add_history(f"Rescaled for {int(b)}x{int(b)} software binning "
                        f"(SASpro exoplanet tool)")
    except Exception:
        pass
    return out


# ============================================================
# Transit search helpers — ported from exoplanet_ground_telescope_script - Kyle Lynch
# ============================================================

def _robust_sigma(x: np.ndarray) -> float:
    """Median-absolute-deviation based sigma (1.4826 * MAD)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad) if np.isfinite(mad) else np.nan


def _sort_lightcurve(time_jd: np.ndarray, flux: np.ndarray,
                     positive_only: bool = True):
    t = np.asarray(time_jd, float)
    y = np.asarray(flux, float)
    good = np.isfinite(t) & np.isfinite(y)
    if positive_only:
        good &= (y > 0)
    if not np.any(good):
        return np.array([], float), np.array([], float)
    s = np.argsort(t[good])
    return t[good][s], y[good][s]


def _segment_slices(time: np.ndarray, gap_days: float):
    t = np.asarray(time, float)
    if t.size == 0:
        return []
    cuts = np.where(np.diff(t) > gap_days)[0]
    starts = np.r_[0, cuts + 1]
    ends   = np.r_[cuts + 1, t.size]
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


def _upper_outlier_mask(y: np.ndarray, sigma_mult: float = 6.0) -> np.ndarray:
    arr = np.asarray(y, float)
    mask = np.zeros(arr.shape, dtype=bool)
    good = np.isfinite(arr) & (arr > 0)
    if np.sum(good) < 6:
        return mask
    med = float(np.nanmedian(arr[good]))
    sig = _robust_sigma(arr[good] - med)
    if not (np.isfinite(sig) and sig > 0):
        return mask
    return np.isfinite(arr) & (arr > med + sigma_mult * sig)


def _prepare_search_lightcurve(time_jd: np.ndarray, flux: np.ndarray):
    """Clean, sort, median-normalise, and clip upper outliers."""
    t, y = _sort_lightcurve(time_jd, flux, positive_only=True)
    if t.size == 0:
        return t, y
    med = np.nanmedian(y[np.isfinite(y) & (y > 0)])
    if np.isfinite(med) and med != 0:
        y = y / med
    bad = _upper_outlier_mask(y, sigma_mult=6.0)
    if np.any(bad):
        keep = np.isfinite(y) & (~bad)
        t, y = t[keep], y[keep]
        if y.size > 0:
            med2 = np.nanmedian(y)
            if np.isfinite(med2) and med2 != 0:
                y = y / med2
    return t, y


def _phase_centered(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Fold to [-0.5, 0.5]."""
    ph = ((np.asarray(time, float) - t0) / period) % 1.0
    return ((ph + 0.5) % 1.0) - 0.5


def _whiten_periodogram(power: np.ndarray):
    """Subtract running-median background from BLS power spectrum."""
    from scipy.ndimage import median_filter as _mf
    power = np.asarray(power, float)
    if power.size == 0:
        return np.array([], float), np.array([], float)
    finite = np.isfinite(power)
    fill = float(np.nanmedian(power[finite])) if np.any(finite) else 0.0
    arr = np.where(finite, power, fill)
    max_window = power.size if power.size % 2 == 1 else power.size - 1
    if max_window < 3:
        bg = np.full_like(arr, fill, dtype=float)
    else:
        window = max(31, int(power.size // 150))
        if window % 2 == 0:
            window += 1
        window = max(3, min(window, max_window))
        bg = _mf(arr, size=window, mode="nearest")
    whitened = arr - bg
    whitened -= np.nanmedian(whitened[finite])
    whitened[~finite] = np.nan
    bg[~finite] = np.nan
    return bg, whitened


def _top_peak_indices(periods: np.ndarray, power: np.ndarray,
                      max_peaks: int = 12, merge_frac: float = 0.01):
    periods = np.asarray(periods, float)
    power   = np.asarray(power, float)
    if periods.size == 0:
        return []
    cand = []
    for i in range(power.size):
        if not np.isfinite(power[i]):
            continue
        left  = power[i-1] if i > 0 else -np.inf
        right = power[i+1] if i+1 < power.size else -np.inf
        if power[i] >= left and power[i] >= right:
            cand.append(i)
    if not cand:
        cand = list(np.where(np.isfinite(power))[0])
    cand = np.asarray(cand, int)
    order = cand[np.argsort(power[cand])[::-1]]
    keep = []
    for idx in order:
        p = periods[idx]
        if not (np.isfinite(p) and p > 0):
            continue
        if any(abs(p / periods[j] - 1.0) <= merge_frac for j in keep
               if np.isfinite(periods[j]) and periods[j] > 0):
            continue
        keep.append(int(idx))
        if len(keep) >= max_peaks:
            break
    return keep


def _make_binned_curve(x: np.ndarray, y: np.ndarray, bin_width: float):
    """Robust-sigma-clipped binning; returns (bx, by, be)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 10:
        return np.array([], float), np.array([], float), np.array([], float)
    edges = np.arange(np.nanmin(x), np.nanmax(x) + bin_width, bin_width)
    bx, by, be = [], [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.sum(m) < 4:
            continue
        yy = y[m]; xx = x[m]
        med = np.nanmedian(yy)
        sig = _robust_sigma(yy)
        if np.isfinite(sig) and sig > 0:
            keep = np.abs(yy - med) < 4.0 * sig
            xx, yy = xx[keep], yy[keep]
        if yy.size < 4:
            continue
        sig2 = _robust_sigma(yy)
        bx.append(float(np.nanmedian(xx)))
        by.append(float(np.nanmedian(yy)))
        be.append(float(sig2 / np.sqrt(yy.size)) if np.isfinite(sig2) and sig2 > 0
                  else float(np.nanstd(yy) / np.sqrt(max(1, yy.size))))
    return np.asarray(bx, float), np.asarray(by, float), np.asarray(be, float)


def _estimate_night_dip_events(time_jd: np.ndarray, flux: np.ndarray,
                                gap_days: float = 0.20):
    """
    Per-night dip detection: returns list of dicts with
    t_center, depth_frac, snr, duration_d.
    """
    from scipy.ndimage import median_filter as _mf
    t, y = _sort_lightcurve(time_jd, flux, positive_only=True)
    if t.size < 12:
        return []
    events = []
    for sl in _segment_slices(t, gap_days):
        tn, yn = t[sl], y[sl]
        good = np.isfinite(tn) & np.isfinite(yn) & (yn > 0)
        tn, yn = tn[good], yn[good]
        if tn.size < 8:
            continue
        med = float(np.nanmedian(yn))
        if med <= 0 or not np.isfinite(med):
            continue
        seg = yn / med
        scatter = _robust_sigma(seg)
        if not np.isfinite(scatter) or scatter <= 0:
            scatter = float(np.nanstd(seg))
        scatter = max(scatter, 1e-5)

        filled = seg.copy()
        filled[~np.isfinite(filled)] = 1.0
        w = min(11, max(5, (tn.size // 12) * 2 + 1))
        if w % 2 == 0:
            w += 1
        if w >= tn.size:
            w = tn.size if tn.size % 2 == 1 else max(5, tn.size - 1)
        smooth = _mf(filled, size=max(5, w), mode="nearest").astype(float)

        oot = float(np.nanmedian(smooth))
        min_idx = int(np.nanargmin(smooth))
        depth = max(0.0, oot - float(smooth[min_idx]))
        if not np.isfinite(depth) or depth < max(2.0 * scatter, 8e-4):
            continue

        thresh = oot - max(1.2 * scatter, 0.45 * depth)
        low = smooth <= thresh
        left = right = min_idx
        while left > 0 and low[left-1]:
            left -= 1
        while right + 1 < low.size and low[right+1]:
            right += 1

        # cadence estimate
        dts = np.diff(np.sort(tn))
        dts = dts[dts > 1e-6]
        cad = float(np.nanmedian(dts)) if dts.size > 0 else 5.0/1440.0
        dur = float((tn[right] - tn[left]) + max(cad, 5.0/1440.0))
        events.append({
            "t_center": float(np.nanmedian(tn[left:right+1])),
            "depth_frac": float(depth),
            "snr": float(depth / scatter),
            "duration_d": dur,
        })
    return events


def _alias_penalty(period_d: float, tol: float = 0.02) -> float:
    if not (np.isfinite(period_d) and period_d > 0):
        return 0.0
    penalty = 0.0
    for alias in (0.5, 1.0, 2.0, 3.0):
        frac = abs(period_d / alias - 1.0)
        if frac <= tol:
            penalty = max(penalty, 1.0 - frac / tol)
    return float(penalty)


class OverlayView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        # disable built-in hand drag
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # always arrow cursor
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        self._panning = False
        self._last_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pt = self.mapToScene(event.pos())
            # if we clicked an ellipse, let it handle the event
            for it in self.scene().items(scene_pt):
                if isinstance(it, ClickableEllipseItem):
                    super().mousePressEvent(event)
                    return
            # else: start panning
            self._panning = True
            self._last_pos = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

class ClickableEllipseItem(QGraphicsEllipseItem):
    def __init__(self, rect: QRectF, index: int, callback):
        super().__init__(rect)
        self.index = index
        self.callback = callback
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.callback(self.index, shift)
        super().mousePressEvent(ev)


class ReferenceOverlayDialog(QDialog):
    def __init__(self, plane: np.ndarray, positions: List[Tuple], target_median: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Reference Frame: Stars Overlay"))
        self.plane = np.asarray(plane, dtype=np.float32)
        self.positions = positions
        self.target_median = target_median
        self.autostretch = True

        self._normal_pen   = QPen(QColor('lightblue'), 3)
        self._dip_pen      = QPen(QColor('yellow'), 3)
        self._selected_pen = QPen(QColor('red'), 4)

        self.ellipse_items: dict[int, ClickableEllipseItem] = {}
        self.flagged_stars: Set[int] = set()

        self._build_ui()
        self._init_graphics()

        if parent and hasattr(parent, 'star_list'):
            parent.star_list.itemSelectionChanged.connect(self._update_highlights)

        QTimer.singleShot(0, self._fit_to_100pct)

    def _build_ui(self):
        self.view = OverlayView(self)
        self.view.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        btns = QHBoxLayout()

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(lambda: self.view.scale(1.2, 1.2))
        btns.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(lambda: self.view.scale(1/1.2, 1/1.2))
        btns.addWidget(self.zoom_out_btn)

        self.reset_btn = QPushButton("Reset Zoom")
        self.reset_btn.clicked.connect(self._fit_to_100pct)
        btns.addWidget(self.reset_btn)

        self.fit_btn = QPushButton("Fit to Window")
        self.fit_btn.clicked.connect(self._fit_to_window)
        btns.addWidget(self.fit_btn)

        self.stretch_btn = QPushButton()
        self.stretch_btn.clicked.connect(self._toggle_autostretch)
        btns.addWidget(self.stretch_btn)

        self._update_stretch_button_text()
        btns.addStretch()

        lay = QVBoxLayout(self)
        lay.addWidget(self.view)
        lay.addLayout(btns)
        self.resize(800, 600)

    def _update_stretch_button_text(self):
        self.stretch_btn.setText(f"Stretch: {'ON' if self.autostretch else 'OFF'}")

    def _normalize_to_u8_linear(self, img: np.ndarray) -> np.ndarray:
        """
        Linear preview:
        - if already in [0,1], keep it linear
        - otherwise percentile-compress to 8-bit
        """
        arr = np.asarray(img, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=np.uint8)

        amin = float(np.min(arr[finite]))
        amax = float(np.max(arr[finite]))

        # Common SASpro case: float image already normalized to [0,1]
        if amin >= 0.0 and amax <= 1.0:
            out = np.clip(arr, 0.0, 1.0)
            return (out * 255.0).astype(np.uint8)

        # Otherwise make a robust linear preview
        lo, hi = np.percentile(arr[finite], [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = amin
            hi = amax if amax > amin else (amin + 1.0)

        out = (arr - lo) / max(hi - lo, 1e-8)
        out = np.clip(out, 0.0, 1.0)
        return (out * 255.0).astype(np.uint8)

    def _to_preview_u8(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)


    def _normalize_preview_source(self, img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)

        # keep original values, just move into a sane preview/stretch domain
        if arr.dtype.kind in "iu":
            arr = arr.astype(np.float32)

        else:
            arr = np.asarray(arr, dtype=np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=np.float32)

        amin = float(np.min(arr[finite]))
        amax = float(np.max(arr[finite]))

        # already normalized
        if amin >= 0.0 and amax <= 1.0:
            return np.clip(arr, 0.0, 1.0)

        # signed / weird-range data -> normalize to 0..1 first
        if amax <= amin:
            return np.zeros(arr.shape, dtype=np.float32)

        arr01 = (arr - amin) / (amax - amin)
        return np.clip(arr01, 0.0, 1.0).astype(np.float32)

    def _init_graphics(self):
        base = self._normalize_preview_source(self.plane)
        img = base

        if self.autostretch:
            try:
                stretched = stretch_mono_image(base, target_median=0.3)
                stretched = np.asarray(stretched, dtype=np.float32)

                if np.isfinite(stretched).any() and np.nanstd(stretched) > 1e-4:
                    img = np.clip(stretched, 0.0, 1.0)
            except Exception as e:
                print(f"[Exoplanet preview] stretch_mono_image failed: {e}")
                img = base

        arr8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        h, w = arr8.shape
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        pix = QPixmap.fromImage(qimg)

        self.scene.clear()
        self.ellipse_items.clear()
        self.scene.addItem(QGraphicsPixmapItem(pix))

        radius = max(2, int(math.ceil(1.2 * self.target_median)))
        for idx, (x, y) in enumerate(self.positions):
            r = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
            ell = ClickableEllipseItem(r, idx, self._on_star_clicked)
            ell.setPen(self._normal_pen)
            ell.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            self.scene.addItem(ell)
            self.ellipse_items[idx] = ell


    def _fit_to_100pct(self):
        self.view.resetTransform()
        rect = self.scene.itemsBoundingRect()
        self.view.setSceneRect(rect)
        self.view.centerOn(rect.center())

    def _fit_to_window(self):
        rect = self.scene.itemsBoundingRect()
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _toggle_autostretch(self):
        self.autostretch = not self.autostretch
        self._update_stretch_button_text()
        self._init_graphics()

    def _on_star_clicked(self, index: int, shift: bool):
        parent = self.parent()
        if not parent or not hasattr(parent, 'star_list'):
            return

        lst = parent.star_list
        item = lst.item(index)
        if not item:
            return

        if shift:
            item.setSelected(not item.isSelected())
        else:
            lst.clearSelection()
            item.setSelected(True)

        lst.scrollToItem(item)
        self._update_highlights()

    def _update_highlights(self):
        parent = self.parent()
        if not parent or not hasattr(parent, 'star_list'):
            return

        sel = {item.data(Qt.ItemDataRole.UserRole)
               for item in parent.star_list.selectedItems()}

        for idx, ell in self.ellipse_items.items():
            if idx in sel:
                ell.setPen(self._selected_pen)
            elif idx in self.flagged_stars:
                ell.setPen(self._dip_pen)
            else:
                ell.setPen(self._normal_pen)

    def update_dip_flags(self, flagged_indices: Set[int]):
        self.flagged_stars = flagged_indices
        self._update_highlights()


# ---------------------------------------------------------------------------
# SIMBAD/VSX name-resolution helpers.  Used by ExoPlanetWindow to pick the
# best display / export name near a (RA, Dec) — biasing toward variable-star
# designations and named catalogs over incidental Gaia DR3 entries.
# ---------------------------------------------------------------------------

# SIMBAD otype prefixes considered "variable star".
_VARIABLE_OTYPE_PREFIXES = (
    "V*",   "Mi*",  "EB*", "EA*", "EW*", "RR*", "Ce*", "dS*", "SX*",
    "Sr*",  "SR*",  "LP*", "LPV", "Ir*", "IR*", "Pu*", "Ro*", "RS*",
    "BY*",  "Er*",  "FU*", "GC*", "Ta*", "TT*", "Y*O",
)

# Catalog prefixes we prefer as an exported #NAME over a generic Gaia id.
_PREFERRED_NAME_PREFIXES = (
    "V*",   "V ",   "NSV",
    "HD ",  "HR ",  "HIP ", "TYC ", "BD",  "CD",
    "ASAS ", "ASASSN-", "ASASSN ", "ZTF ", "OGLE",
    "GSC ", "PPM ",  "SAO ", "WDS ", "WISE ",
)

# Deprioritise plain Gaia identifiers.
_DEPRIORITIZED_NAME_PREFIXES = (
    "Gaia DR3 ", "Gaia DR2 ", "Gaia DR1 ", "Gaia EDR3 ",
)


def _is_variable_otype(otype: str) -> bool:
    if not otype:
        return False
    ot = otype.strip()
    return any(ot.startswith(p) for p in _VARIABLE_OTYPE_PREFIXES)


def _name_score(name: str) -> int:
    n = (name or "").strip()
    for p in _DEPRIORITIZED_NAME_PREFIXES:
        if n.startswith(p):
            return -30
    for p in _PREFERRED_NAME_PREFIXES:
        if n.startswith(p):
            return 30
    return 0


# AAVSO's STARID and KNAME fields want the designation alone -- "V336 Aps",
# not SIMBAD's "V* V336 Aps". That leading token is SIMBAD's object-type
# tag (V*, **, Cl*, EM*, SV*, ...) or the literal "NAME" marker, and WebObs
# rejects a submission that still carries it.
#
# The same tag is why the old VSX fallback never fired: query_object() on a
# tagged SIMBAD id resolves nothing, so the name it was meant to replace
# was handed straight through to the file. Resolve VSX by coordinate and
# sanitise whatever comes back.
def _strip_simbad_type_prefix(name: str) -> str:
    n = " ".join(str(name or "").split())
    if not n:
        return ""
    if n.upper().startswith("NAME "):
        n = n[5:].strip()
    head, _sep, rest = n.partition(" ")
    if rest and head.endswith("*"):
        n = rest.strip()
    return " ".join(n.split())


def _aavso_starid(name: str) -> str:
    """SIMBAD/VSX identifier -> a form AAVSO WebObs will accept."""
    return _strip_simbad_type_prefix(name)


class ExoPlanetWindow(QDialog):
    def __init__(self, parent=None, wrench_path=None, settings=None):
        super().__init__(parent)
        self.settings = settings
        self.main_win = parent
        self.setWindowTitle(self.tr("Exoplanet Transit Detector"))

        self.resize(900, 600)
        self.wrench_path = wrench_path
        # State
        self.image_paths      = []
        self._cached_images   = []
        self._cached_headers  = []   # parallel to _cached_images
        self.times            = None  # astropy Time array
        self.star_positions   = []
        self.fluxes           = None  # stars × frames
        self.flags            = None
        self.raw_flux_rgb     = None  # (3, n_stars, n_frames) for color inputs; None for mono
        self.raw_flux_err_rgb = None
        self.median_fwhm      = None
        self.master_dark      = None
        self.master_flat      = None
        self.exposure_time    = None
        self._last_ensemble   = []
        self.ensemble_map     = {}
        self.flagged_stars    = set()   # read by the AAVSO export (comp exclusion)
        self._cached_satcoords = []     # per frame: (N,2) saturated pixels, binned grid
        self._frame_jd_utc    = None    # per frame: mid-exposure JD (UTC), NaN if unknown
        self._frame_exptime   = None    # per frame: exposure seconds
        self._time_how        = ""
        self.phot_cube        = None    # exoplanet_photometry.PhotometryCube (kept stars)
        self.diff_result      = None    # exoplanet_photometry.DiffResult (kept stars)
        self.rel_flux_raw     = None    # differential flux before detrending
        self.screen           = {}      # star idx -> screening metrics
        self.transit_fits     = {}      # star idx -> last transit-fit output
        self.star_names       = {}      # star idx -> name from Identify / Find Star
        self.session_idx      = None    # per frame: observing-night index
        self.sessions         = []      # per night: dict(label, date, start_jd, n, ...)
        self.screen_by_session = {}     # night -> {star idx -> metrics}, built on demand
        self.frame_noise      = None    # per frame: excess fractional noise measured from comps
        self.bin_factor       = 2       # software bin of the cached frames (Auto picks per set)
        self._frame_store     = None    # exoplanet_frame_worker.FrameStore of the binned frames
        self._frame_slots     = []      # per frame: FrameStore slot, or None (kept in RAM)
        self._cached_catalogs = None    # per frame: star catalogue from loading, if any
        self._pool            = None    # process pool, alive from load through measure
        self.side_idx         = None    # per frame: pier side vs the reference (0/1)
        self.side_how         = ""
        self.sci_plot         = None    # matplotlib "Scientific" plot, built on first use
        self._sci_time_cache  = None
        self.detrend_mode     = "airmass"
        self.detrend_degree   = None

        # --- new settings ---
        self.sep_threshold     = 5.0   # SEP σ
        self.border_fraction   = 0.10  # ignore border fraction
        self.ensemble_k        = 10    # max comparison stars in the weighted ensemble
        self.recenter_apertures = True
        self.aperture_mult     = 0.0   # 0 = auto (minimum ensemble scatter), else × FWHM
        # e-/ADU override for the photometric gain. 0 = take EGAIN from the
        # header, and fall back to no Poisson term if it isn't there.
        self.camera_egain      = 0.0

        # Analysis
        self.ls_min_frequency     = 0.01
        self.ls_max_frequency     = 10.0
        self.ls_samples_per_peak  = 10
        self.bls_min_period       = 0.05
        self.bls_max_period       = 2.0
        self.bls_n_periods        = 1000
        self.bls_duration_min_frac= 0.01
        self.bls_duration_max_frac= 0.5
        self.bls_n_durations      = 20

        # WCS (standalone; no WIMI)
        self._wcs   = None
        self.wcs_ra = None
        self.wcs_dec= None

        # — Mode selector —
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel(self.tr("Mode:")))
        self.aligned_mode_rb = QRadioButton(self.tr("Aligned Subs"))
        self.raw_mode_rb     = QRadioButton(self.tr("Raw Subs"))
        self.aligned_mode_rb.setChecked(True)
        mg = QButtonGroup(self)
        mg.addButton(self.aligned_mode_rb); mg.addButton(self.raw_mode_rb)
        mg.buttonToggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.aligned_mode_rb)
        mode_layout.addWidget(self.raw_mode_rb)
        mode_layout.addStretch()
        self.force_solve_cb = QCheckBox(self.tr("Force blind plate solve"))
        self.force_solve_cb.setChecked(False)
        self.force_solve_cb.setToolTip(
            "Always run the plate solver, even if a valid WCS already exists in the file header.\n"
            "Use this when the embedded WCS is inaccurate or from an unreliable source."
        )
        mode_layout.addWidget(self.force_solve_cb)
        mode_layout.addStretch()   # move this addStretch() to after the checkbox        
        self.wrench_button = QToolButton()
        self.wrench_button.setIcon(QIcon(self.wrench_path))
        self.wrench_button.setToolTip("Settings…")
        self.wrench_button.setStyleSheet("""
            QToolButton {
                background-color: #FF4500;
                color: white;
                padding: 4px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background-color: #FF6347;
            }
        """)
        self.wrench_button.clicked.connect(self.open_settings)
        mode_layout.addWidget(self.wrench_button)
        mode_layout.addWidget(make_help_button("exoplanet", self))

        # — Calibration controls (hidden in Aligned) —
        cal_layout = QHBoxLayout()
        self.load_darks_btn = QPushButton(self.tr("Load Master Dark…"))
        self.load_flats_btn = QPushButton(self.tr("Load Master Flat…"))
        for w in (self.load_darks_btn, self.load_flats_btn):
            w.clicked.connect(self.load_masters)
            w.hide()
            cal_layout.addWidget(w)
        self.dark_status_label = QLabel("Dark: ❌");  self.dark_status_label.hide()
        self.flat_status_label = QLabel("Flat: ❌");  self.flat_status_label.hide()
        cal_layout.addWidget(self.dark_status_label)
        cal_layout.addWidget(self.flat_status_label)
        cal_layout.addStretch()

        # — Status & Progress —
        # Word-wrapped: the screening summary runs to ~1700 px on one line,
        # and an unwrapped QLabel makes that the window's minimum width.
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # — Top controls —
        top_layout = QHBoxLayout()
        self.load_raw_btn     = QPushButton(self.tr("1: Load Raw Subs…"))
        self.load_aligned_btn = QPushButton(self.tr("Load, Measure && Photometry…"))
        self.calibrate_btn    = QPushButton(self.tr("1a: Calibrate && Align Subs"))
        self.measure_btn      = QPushButton(self.tr("2: Measure && Photometry"))
        self.load_raw_btn.    clicked.connect(self.load_raw_subs)
        self.load_aligned_btn.clicked.connect(self.load_and_measure_subs)
        self.calibrate_btn.clicked.connect(self.calibrate_and_align)
        self.measure_btn.     clicked.connect(self.detect_stars)
        self.detrend_combo = QComboBox()
        self.detrend_combo.addItems(["No Detrend", "Linear", "Quadratic", "Airmass"])
        self.detrend_combo.setToolTip(
            "Baseline removed from every differential light curve, fitted in\n"
            "TIME with 3σ clipping.  The dip search fits the same baseline\n"
            "jointly with the dip, so detection never relies on the divided\n"
            "curve.  Quadratic can absorb part of a transit that sits mid-\n"
            "night; Airmass (EXOTIC's systematics term) is the physical choice\n"
            "and falls back to Linear when there is no airmass information.")
        self.save_aligned_btn = QPushButton("Save Aligned Frames…")
        self.save_aligned_btn.clicked.connect(self.save_aligned_frames)

        top_layout.addWidget(self.load_raw_btn)
        top_layout.addWidget(self.load_aligned_btn)
        top_layout.addWidget(self.calibrate_btn)
        top_layout.addWidget(self.measure_btn)
        top_layout.addStretch()
        self.night_label = QLabel("Night:")
        self.night_combo = QComboBox()
        self.night_combo.addItem("All nights", None)
        self.night_combo.setToolTip(
            "Frames are grouped into observing nights (local noon to noon, per\n"
            "telescope). Pick one night to plot and screen it on its own; the\n"
            "list's σ and flags, the overlay and Fit Transit follow the choice.\n"
            "All nights plots every night with the daytime gaps collapsed.")
        self.night_combo.currentIndexChanged.connect(self.on_night_changed)
        self.night_label.setVisible(False)
        self.night_combo.setVisible(False)
        top_layout.addWidget(self.night_label)
        top_layout.addWidget(self.night_combo)
        top_layout.addWidget(QLabel("Detrend:"))
        top_layout.addWidget(self.detrend_combo)
        top_layout.addWidget(self.save_aligned_btn)

        # — Star list & Plot —
        # A splitter, so the list (whose flag tags get long) and the plot can
        # trade width on a small screen.
        middle = QSplitter(Qt.Orientation.Horizontal)
        middle.setChildrenCollapsible(False)
        self.star_list  = QListWidget()
        self.star_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.star_list.itemSelectionChanged.connect(self.update_plot_for_selection)
        self.star_list.setStyleSheet("""
            QListWidget::item:selected {
                background: #3399ff; 
                color: white;
            }
        """)
        middle.addWidget(self.star_list)
        self.plot_widget = pg.PlotWidget(title="Light Curves")
        self.plot_widget.addLegend()
        # Graphical (pyqtgraph, interactive) or Scientific (matplotlib,
        # publication style); the matplotlib canvas is built on first use.
        ps = self._plot_settings()
        prow = QHBoxLayout()
        prow.setContentsMargins(0, 0, 0, 0)
        prow.addWidget(QLabel("Plot:"))
        self.plot_style_combo = QComboBox()
        self.plot_style_combo.addItem("Graphical", "graphical")
        self.plot_style_combo.addItem("Scientific", "scientific")
        self.plot_style_combo.setToolTip(
            "Graphical: the interactive dark plot.\n"
            "Scientific: publication-style figure -- every point with its\n"
            "error bar, binned points with propagated errors, BJD_TDB time\n"
            "axis, one panel per night.  Save Plot writes it as a 300 dpi\n"
            "PNG, PDF or SVG.")
        self.plot_units_label = QLabel("Y:")
        self.plot_units_combo = QComboBox()
        self.plot_units_combo.addItem("Relative flux", "flux")
        self.plot_units_combo.addItem("Δ magnitude", "mag")
        self.plot_units_combo.setToolTip(
            "Scientific plot y axis: relative flux, or differential magnitude\n"
            "(-2.5 log10 of the relative flux; brighter is up).")
        self.plot_bin_combo = QComboBox()
        for lab, mins in (("No bins", 0), ("5-min bins", 5), ("10-min bins", 10),
                          ("15-min bins", 15), ("30-min bins", 30)):
            self.plot_bin_combo.addItem(lab, mins)
        self.plot_bin_combo.setToolTip(
            "Binned points: inverse-variance weighted mean per bin.  The error\n"
            "bar is the propagated error, scaled up by the bin's own scatter\n"
            "when the points disagree more than their error bars allow.")
        self.save_plot_btn = QPushButton("Save Plot…")
        self.save_plot_btn.clicked.connect(self.save_plot)
        for cb, key, default in ((self.plot_style_combo, "Exoplanet/plot_style", "graphical"),
                                 (self.plot_units_combo, "Exoplanet/plot_units", "flux"),
                                 (self.plot_bin_combo, "Exoplanet/plot_bin_min", 10)):
            want = ps.value(key, default) if ps is not None else default
            for j in range(cb.count()):
                if str(cb.itemData(j)) == str(want):
                    cb.setCurrentIndex(j)
                    break
        prow.addWidget(self.plot_style_combo)
        prow.addSpacing(8)
        prow.addWidget(self.plot_units_label)
        prow.addWidget(self.plot_units_combo)
        prow.addSpacing(8)
        prow.addWidget(self.plot_bin_combo)
        prow.addStretch(1)
        prow.addWidget(self.save_plot_btn)
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.plot_widget)
        plot_box = QWidget()
        pv = QVBoxLayout(plot_box)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(2)
        pv.addLayout(prow)
        pv.addWidget(self.plot_stack, 1)
        middle.addWidget(plot_box)
        middle.setStretchFactor(0, 2)
        middle.setStretchFactor(1, 5)
        middle.setSizes([200, 500])        # the old 2:5 split; the user's drag is remembered
        st = ps.value("Exoplanet/list_plot_split") if ps is not None else None
        if isinstance(st, QByteArray) and not st.isEmpty():
            middle.restoreState(st)
        if ps is not None:
            middle.splitterMoved.connect(
                lambda *_: ps.setValue("Exoplanet/list_plot_split", middle.saveState()))
        self.plot_style_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        self.plot_units_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        self.plot_bin_combo.currentIndexChanged.connect(self._on_plot_options_changed)
        self._sync_plot_style_widgets()

        # — Bottom rows —
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Min depth / amplitude (ppt):"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(5)
        self.threshold_slider.setToolTip(
            "A star is flagged when its best single dip is at least this deep\n"
            "(DIP), or its smoothed light curve spans at least this much (VAR),\n"
            "AND the signal clears the significance below.  Hot Jupiters are\n"
            "typically 5-25 ppt deep.")
        row1.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel(f"{self.threshold_slider.value()} ppt")
        row1.addWidget(self.threshold_value_label)

        row1.addSpacing(16)
        row1.addWidget(QLabel("Min significance (σ):"))
        self.temporal_snr_spin = QDoubleSpinBox()
        self.temporal_snr_spin.setRange(0.0, 100.0)
        self.temporal_snr_spin.setSingleStep(0.5)
        self.temporal_snr_spin.setValue(7.0)
        self.temporal_snr_spin.setDecimals(1)
        self.temporal_snr_spin.setToolTip(
            "DIP: signal-to-noise of the best box-shaped dip, fitted jointly\n"
            "with the detrend baseline, against white noise PLUS the red noise\n"
            "measured at the dip's own duration (Pont et al. 2006).  Correlated\n"
            "wander therefore can't masquerade as a transit.\n\n"
            "VAR: significance of smooth variability from the von Neumann ratio\n"
            "(how much more alike neighbouring points are than white noise\n"
            "allows).\n\n"
            "~7σ is the usual single-transit detection threshold."
        )
        self.temporal_snr_spin.valueChanged.connect(
            lambda _: self.apply_threshold(self.threshold_slider.value())
        )
        row1.addWidget(self.temporal_snr_spin)

        row1.addStretch()
        self.identify_btn = QPushButton("Identify Star…")
        self.identify_btn.clicked.connect(self.on_identify_star)
        row1.addWidget(self.identify_btn)
        self.find_star_btn = QPushButton("Find Star…")
        self.find_star_btn.setToolTip(
            "Look up a star by name (SIMBAD-resolvable — e.g. V336 Aps,\n"
            "HD 12345, TYC 9256-617-1, Gaia DR3 …). Selects the closest\n"
            "detected star in the list."
        )
        self.find_star_btn.clicked.connect(self.on_find_star_by_name)
        row1.addWidget(self.find_star_btn)
        self.show_ensemble_btn = QPushButton("Show Ensemble Members")
        self.show_ensemble_btn.clicked.connect(self.show_ensemble_members)
        self.analyze_btn = QPushButton("Analyze Star…")
        self.analyze_btn.clicked.connect(self.on_analyze)
        self.fit_transit_btn = QPushButton("Fit Transit…")
        self.fit_transit_btn.setToolTip(
            "EXOTIC-style transit fit of the selected star: archive priors,\n"
            "aperture and comparison-set optimisation, transit x airmass\n"
            "model, MCMC uncertainties, BJD_TDB, AAVSO Exoplanet report.")
        self.fit_transit_btn.clicked.connect(self.on_fit_transit)
        self.fit_transit_btn.setEnabled(False)

        # Two rows of about equal width: thresholds and lookup on the first,
        # per-star analysis and export on the second.  All of it on one row
        # set a ~900 px minimum window width.
        row2 = QHBoxLayout()
        row2.addWidget(self.show_ensemble_btn)
        row2.addWidget(self.analyze_btn)
        row2.addWidget(self.fit_transit_btn)
        self.fetch_tesscut_btn = QPushButton("Query TESScut Light Curve")
        self.fetch_tesscut_btn.setEnabled(False)
        self.fetch_tesscut_btn.clicked.connect(self.query_tesscut)
        row2.addWidget(self.fetch_tesscut_btn)
        self.export_btn = QPushButton("Export CSV/FITS")
        self.export_btn.clicked.connect(self.export_data)
        row2.addWidget(self.export_btn)
        self.export_aavso_btn = QPushButton("Export → AAVSO")
        self.export_aavso_btn.clicked.connect(self.export_to_aavso)
        row2.addWidget(self.export_aavso_btn)

        # — Assemble —
        main = QVBoxLayout(self)
        main.addLayout(mode_layout)
        main.addLayout(cal_layout)
        main.addLayout(top_layout)
        main.addWidget(middle, 1)
        main.addLayout(row1)
        main.addLayout(row2)
        statlay = QHBoxLayout()
        statlay.addWidget(self.status_label, 3)
        statlay.addWidget(self.progress_bar, 1)
        main.addLayout(statlay)

        # init
        self.on_mode_changed(self.aligned_mode_rb, True)
        self.detrend_combo.setCurrentIndex(3)
        self.on_detrend_changed(3)
        self.detrend_combo.currentIndexChanged.connect(self.on_detrend_changed)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.analyze_btn.setEnabled(False)
        self.calibrate_btn.hide()

    # ---------------- UI wiring ----------------

    def _on_threshold_changed(self, v: int):
        self.threshold_value_label.setText(f"{v} ppt")
        self.apply_threshold(v)
        if hasattr(self, '_ref_overlay'):
            self._ref_overlay._update_highlights()

    def open_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Photometry & Analysis Settings")
        layout = QVBoxLayout(dlg)

        photo_box = QGroupBox("Photometry")
        fb = QFormLayout(photo_box)
        self.sep_spin = QDoubleSpinBox(); self.sep_spin.setRange(1.0, 20.0); self.sep_spin.setSingleStep(0.5); self.sep_spin.setValue(self.sep_threshold)
        fb.addRow("SEP detection σ:", self.sep_spin)
        self.border_spin = QDoubleSpinBox(); self.border_spin.setRange(0.0, 0.5); self.border_spin.setSingleStep(0.01); self.border_spin.setValue(self.border_fraction)
        fb.addRow("Border fraction:", self.border_spin)
        self.aperture_combo = QComboBox()
        self.aperture_combo.addItem("Auto (lowest ensemble scatter)", 0.0)
        for _m in xphot.APERTURE_MULTS:
            self.aperture_combo.addItem(f"{_m:g} × FWHM", float(_m))
        _ai = self.aperture_combo.findData(float(getattr(self, "aperture_mult", 0.0) or 0.0))
        self.aperture_combo.setCurrentIndex(_ai if _ai >= 0 else 0)
        self.aperture_combo.setToolTip(
            "Every radius in the grid is measured; this picks the one used for\n"
            "the star list, plots and AAVSO export.  Auto chooses the radius\n"
            "that minimises the comparison stars' mutual scatter.  The transit\n"
            "fit re-optimises the radius for its own target.")
        fb.addRow("Aperture radius:", self.aperture_combo)
        self.recenter_cb = QCheckBox("Recentre apertures on each frame")
        self.recenter_cb.setChecked(bool(getattr(self, "recenter_apertures", True)))
        self.recenter_cb.setToolTip(
            "Apertures always follow the field (per-frame affine from the star\n"
            "catalogues).  With this on, stars bright enough for a reliable\n"
            "centroid are also individually recentred.")
        fb.addRow("", self.recenter_cb)
        self.egain_spin = QDoubleSpinBox()
        self.egain_spin.setRange(0.0, 100.0)
        self.egain_spin.setSingleStep(0.05)
        self.egain_spin.setDecimals(3)
        self.egain_spin.setValue(self._camera_gain_setting())
        self.egain_spin.setSpecialValueText("Auto (from EGAIN header)")
        self.egain_spin.setToolTip(
            "Electrons per ADU at the gain setting you imaged with, read off "
            "the camera maker's gain chart (ZWO unity gain = 1.0). Used for "
            "the stars' own shot noise in the error bars and MERR.\n\n"
            "Auto = EGAIN from the frame headers. If neither is available "
            "the shot-noise term is dropped, which understates the errors "
            "of bright stars.\n\n"
            "This is NOT the gain setting itself (0-500 on ZWO/QHY) — it is "
            "e-/ADU, typically 0.1-10. Set the ADC depth below too."
        )
        fb.addRow("Camera gain (e-/ADU):", self.egain_spin)
        self.adc_combo = QComboBox()
        self.adc_combo.addItem("Auto (from the raw frames)", "auto")
        for _bits in (16, 14, 12, 10):
            self.adc_combo.addItem(f"{_bits}-bit", str(_bits))
        _ai = self.adc_combo.findData(str(self._adc_bits_setting()))
        self.adc_combo.setCurrentIndex(max(0, _ai))
        self.adc_combo.setToolTip(
            "Bit depth of the camera's ADC in the mode you imaged in.\n\n"
            "Gain charts and EGAIN count the ADC's own steps. A 12-bit ADC "
            "is written into a 16-bit file ×16, so the gain is divided by "
            "16 (14-bit: by 4).\n\n"
            "Auto reads it from raw 16-bit frames (a 12-bit camera writes "
            "only multiples of 16). Calibrated or registered float frames "
            "can't show it — set it here. ASI294MM: 12-bit unbinned "
            "(unlocked), 14-bit at bin 2."
        )
        fb.addRow("Camera ADC:", self.adc_combo)
        self.bin_combo = QComboBox()
        self.bin_combo.addItem("Auto (from the star size)", "auto")
        for _b in (1, 2, 3, 4):
            self.bin_combo.addItem(f"{_b}×{_b}", str(_b))
        _bi = self.bin_combo.findData(str(self._bin_mode()))
        self.bin_combo.setCurrentIndex(_bi if _bi >= 0 else 0)
        self.bin_combo.setToolTip(
            "Frames are averaged in b×b blocks as they load.  This does not add\n"
            "photometric signal-to-noise -- an aperture sums the same photons\n"
            "and read noise either way -- but it divides memory and time by b²,\n"
            "and costs nothing while stars stay a few pixels wide.\n\n"
            "Auto measures the first frame: 1×1 when stars are under 3 px FWHM,\n"
            "otherwise the largest of 2-4 that keeps them at least 2.5 px.\n"
            "Raw colour mosaics always bin by an even factor.  Applies when\n"
            "frames are next loaded.")
        fb.addRow("Software binning:", self.bin_combo)
        self.gpu_detect_cb = QCheckBox("Use the GPU for star detection")
        try:
            self.gpu_detect_cb.setChecked(bool(self._plot_settings().value(
                "Exoplanet/gpu_detect", True, type=bool)))
        except Exception:
            self.gpu_detect_cb.setChecked(True)
        self.gpu_detect_cb.setToolTip(
            "Per-frame star catalogues (field tracking, frame quality) from the\n"
            "GPU detector when a CUDA / DirectML / MPS device is available.  The\n"
            "stars to measure and the FWHM still come from SEP on the reference\n"
            "frame.  Without a GPU, SEP runs in one worker process per core.")
        fb.addRow("", self.gpu_detect_cb)
        layout.addWidget(photo_box)

        ens_box = QGroupBox("Ensemble Normalization")
        ef = QFormLayout(ens_box)
        self.ensemble_spin = QSpinBox(); self.ensemble_spin.setRange(1, 50); self.ensemble_spin.setValue(self.ensemble_k)
        self.ensemble_spin.setToolTip(
            "Maximum number of comparison stars in the weighted ensemble.\n"
            "Candidates are the brightest unsaturated stars; each is weighted by\n"
            "1/σ² of its scatter against the others, and variables are rejected\n"
            "iteratively before the best k are kept.")
        ef.addRow("Max comparison stars:", self.ensemble_spin)
        layout.addWidget(ens_box)

        scr_box = QGroupBox("Screening")
        sf = QFormLayout(scr_box)
        self.field_rel_cb = QCheckBox("Flag only stars that stand out from the field")
        self.field_rel_cb.setChecked(bool(getattr(self, "field_relative", True)))
        self.field_rel_cb.setToolTip(
            "Each star's dip significance, von Neumann statistic and amplitude are\n"
            "compared with stars of similar brightness (median and robust spread\n"
            "within ±0.5 mag).  Systematics shared by the whole field -- seeing-\n"
            "dependent aperture loss, flat-field error, a meridian flip -- raise\n"
            "every star together; with this off they can flag the whole field.\n"
            "The ppt and σ thresholds still apply on top.")
        sf.addRow("", self.field_rel_cb)
        self.field_sigma_spin = QDoubleSpinBox()
        self.field_sigma_spin.setRange(2.0, 30.0)
        self.field_sigma_spin.setSingleStep(0.5)
        self.field_sigma_spin.setValue(float(getattr(self, "field_sigma", 5.0)))
        self.field_sigma_spin.setToolTip(
            "How many robust σ (1.4826 × MAD) above stars of similar brightness a\n"
            "star's statistic must be.  Lower finds fainter signals and more false\n"
            "alarms; VaST's study found the best cut between ~4 and ~16 depending\n"
            "on the statistic and the data.")
        sf.addRow("Field outlier threshold (σ):", self.field_sigma_spin)
        self.night_var_cb = QCheckBox("Flag night-to-night variability (long-period variables)")
        self.night_var_cb.setChecked(self._night_check_setting())
        self.night_var_cb.setToolTip(
            "With two or more nights loaded, also flag stars whose nightly mean\n"
            "level changes well beyond the comparison stars' own night-to-night\n"
            "scatter -- Miras, semiregulars and other long-period variables.\n\n"
            "Off by default: a transit happens within one night, and a flat-\n"
            "field residual that changes from night to night trips this test\n"
            "too.  On the NGC660 data 6 of the 18 stars along one edge of the\n"
            "frame were flagged this way, against 7 of the other 297.\n\n"
            "Stars it flags are also kept out of the AAVSO comparison stars.")
        sf.addRow("", self.night_var_cb)
        layout.addWidget(scr_box)

        cds_box = QGroupBox("CDS / SIMBAD")
        cds_form = QFormLayout(cds_box)

        _pref, _pinned = cds_health.server_preference()
        self.cds_server_combo = QComboBox()
        self.cds_server_combo.addItem(
            "Automatic (Strasbourg, then Harvard, then IDIA)", "")
        for _host in cds_health.MIRRORS:
            self.cds_server_combo.addItem(_host, _host)
        _i = self.cds_server_combo.findData(_pref)
        self.cds_server_combo.setCurrentIndex(_i if _i >= 0 else 0)
        self.cds_server_combo.setToolTip(
            "Which SIMBAD mirror to try first.\n\n"
            "Harvard (simbad.harvard.edu) is the US mirror \u2014 the same "
            "machine the web UI calls simbad.cfa.harvard.edu. IDIA "
            "(simbad.idia.ac.za, South Africa) is an older copy: in November "
            "2026 its newest record was from April 2026, against October at "
            "Harvard, so it is tried last."
        )
        cds_form.addRow("SIMBAD mirror:", self.cds_server_combo)

        self.cds_pin_cb = QCheckBox("Don't fall back to the other mirrors")
        self.cds_pin_cb.setChecked(_pinned)
        self.cds_pin_cb.setToolTip(
            "If your network reaches one mirror and not the other, failing "
            "over just costs a timeout on every lookup."
        )
        cds_form.addRow("", self.cds_pin_cb)

        self.cds_vizier_combo = QComboBox()
        # Name the mirror actually in use when a probe has run, not the
        # preference (vizier_server() is "preference, else Strasbourg").
        try:
            _vz_txt = str(cds_health.vizier_status_text() or "")
        except Exception:
            _vz_txt = ""
        _vz_now = _vz_txt.split("available via ", 1)[1] if "available via " in _vz_txt else ""
        self.cds_vizier_combo.addItem(
            f"Automatic (now {_vz_now})" if _vz_now else "Automatic (fastest working mirror)", "")
        for _host in cds_health.VIZIER_MIRRORS:
            self.cds_vizier_combo.addItem(_host, _host)
        # Select from the STORED PREFERENCE, never from the resolved host.
        # vizier_server() resolves "Automatic" to a concrete hostname, so
        # findData() on it landed on a real mirror row -- and because OK
        # writes the combo's data back unconditionally, merely opening this
        # dialog and pressing OK silently converted Automatic into a pinned
        # mirror and killed the failover. An unknown preference falls back
        # to Automatic, which is the safe direction: the worst case is that
        # someone who deliberately pinned a mirror picks it again.
        _vz_pref = ""
        for _getter in ("vizier_preference", "vizier_server_preference"):
            _fn = getattr(cds_health, _getter, None)
            if callable(_fn):
                try:
                    _r = _fn()
                    _vz_pref = str((_r[0] if isinstance(_r, tuple) else _r) or "")
                    break
                except Exception:
                    pass
        _vi = self.cds_vizier_combo.findData(_vz_pref) if _vz_pref else 0
        self.cds_vizier_combo.setCurrentIndex(_vi if _vi > 0 else 0)
        self.cds_vizier_combo.setToolTip(
            "Which VizieR mirror serves APASS, UCAC4 and VSX -- the catalogs "
            "that supply comparison-star magnitudes for the ensemble zero "
            "point.\n\n"
            "Automatic races the mirrors and uses the first that serves a real "
            "query. A mirror picked here is tried FIRST, and every lookup "
            "still fails over to the others when it stops answering.\n\n"
            "SIMBAD lookups fall back to VSX, UCAC4 (VizieR) and Sesame name "
            "resolution, which all run on these mirrors.")
        cds_form.addRow("VizieR mirror (VSX, APASS):", self.cds_vizier_combo)

        self.cds_status_label = QLabel(
            cds_health.status_text() + "\n" + cds_health.vizier_status_text())
        self.cds_status_label.setWordWrap(True)
        cds_form.addRow("Status:", self.cds_status_label)

        _test_btn = QPushButton("Test now")

        def _test_cds():
            cds_health.set_server_preference(
                self.cds_server_combo.currentData() or "",
                pin=self.cds_pin_cb.isChecked(),
                vizier=self.cds_vizier_combo.currentData() or "",
            )
            self.cds_status_label.setText("Probing\u2026")
            QApplication.processEvents()
            cds_health.probe(force=True)
            cds_health.vizier_probe(force=True)
            self.cds_status_label.setText(
                cds_health.status_text() + "\n" + cds_health.status_detail()
                + "\n" + cds_health.vizier_status_text() + "\n"
                + cds_health.vizier_status_detail())

        _test_btn.clicked.connect(_test_cds)
        cds_form.addRow("", _test_btn)

        layout.addWidget(cds_box)

        ana_box = QGroupBox("Analysis (period search)")
        form = QFormLayout(ana_box)
        self.ls_samp_spin = QSpinBox(); self.ls_samp_spin.setRange(1, 100); self.ls_samp_spin.setValue(self.ls_samples_per_peak)
        form.addRow("LS samples / peak:", self.ls_samp_spin)
        self.bls_min_spin = QDoubleSpinBox(); self.bls_min_spin.setRange(0.01, 10.0); self.bls_min_spin.setValue(self.bls_min_period)
        form.addRow("BLS min period [d]:", self.bls_min_spin)
        self.bls_max_spin = QDoubleSpinBox(); self.bls_max_spin.setRange(0.01, 10.0); self.bls_max_spin.setValue(self.bls_max_period)
        form.addRow("BLS max period [d]:", self.bls_max_spin)
        self.bls_nper_spin = QSpinBox(); self.bls_nper_spin.setRange(10, 20000); self.bls_nper_spin.setValue(self.bls_n_periods)
        form.addRow("BLS # periods:", self.bls_nper_spin)
        self.bls_min_frac_spin = QDoubleSpinBox(); self.bls_min_frac_spin.setRange(0.0001, 1.0); self.bls_min_frac_spin.setSingleStep(0.001); self.bls_min_frac_spin.setValue(self.bls_duration_min_frac)
        form.addRow("BLS min dur frac:", self.bls_min_frac_spin)
        self.bls_max_frac_spin = QDoubleSpinBox(); self.bls_max_frac_spin.setRange(0.01, 1.0); self.bls_max_frac_spin.setSingleStep(0.01); self.bls_max_frac_spin.setValue(self.bls_duration_max_frac)
        form.addRow("BLS max dur frac:", self.bls_max_frac_spin)
        self.bls_ndur_spin = QSpinBox(); self.bls_ndur_spin.setRange(1, 200); self.bls_ndur_spin.setValue(self.bls_n_durations)
        form.addRow("BLS # durations:", self.bls_ndur_spin)
        layout.addWidget(ana_box)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.sep_threshold   = self.sep_spin.value()
            self.border_fraction = self.border_spin.value()
            self.camera_egain    = self.egain_spin.value()
            self.ensemble_k      = self.ensemble_spin.value()
            self.aperture_mult   = float(self.aperture_combo.currentData() or 0.0)
            self.recenter_apertures = bool(self.recenter_cb.isChecked())
            self.field_relative  = bool(self.field_rel_cb.isChecked())
            self.field_sigma     = float(self.field_sigma_spin.value())
            try:
                _ps = self._plot_settings()
                _ps.setValue("Exoplanet/bin_mode", str(self.bin_combo.currentData()))
                _ps.setValue("Exoplanet/gpu_detect", bool(self.gpu_detect_cb.isChecked()))
                _ps.setValue("Exoplanet/egain", float(self.egain_spin.value()))
                _ps.setValue("Exoplanet/adc_bits", str(self.adc_combo.currentData()))
                _ps.setValue("Exoplanet/night_to_night", bool(self.night_var_cb.isChecked()))
                self._gpu_ok = None          # re-check the device next time
            except Exception:
                pass
            # Re-flag after the settings are stored: the night-to-night
            # switch is read from them.
            if getattr(self, "fluxes", None) is not None:
                self._on_threshold_changed(self.threshold_slider.value())
            cds_health.set_server_preference(
                self.cds_server_combo.currentData() or "",
                pin=self.cds_pin_cb.isChecked(),
                vizier=self.cds_vizier_combo.currentData() or "",
            )
            self.ls_samples_per_peak = self.ls_samp_spin.value()
            self.bls_min_period        = self.bls_min_spin.value()
            self.bls_max_period        = self.bls_max_spin.value()
            self.bls_n_periods         = self.bls_nper_spin.value()
            self.bls_duration_min_frac = self.bls_min_frac_spin.value()
            self.bls_duration_max_frac = self.bls_max_frac_spin.value()
            self.bls_n_durations       = self.bls_ndur_spin.value()

    def on_mode_changed(self, button, checked):
        is_raw = checked and (button is self.raw_mode_rb)
        for w in (
            self.load_raw_btn,
            self.load_darks_btn,
            self.load_flats_btn,
            self.dark_status_label,
            self.flat_status_label,
            self.calibrate_btn,
        ):
            w.setVisible(is_raw)
        self.load_aligned_btn.setVisible(not is_raw)
        self.measure_btn.setVisible(is_raw)

    # ---- parallel frame loading ----------------------------------------------
    #
    # Frames are loaded, binned, saturation-mapped and (SEP) catalogued in
    # worker PROCESSES -- sep.extract holds the GIL, so the old thread pool
    # ran it one frame at a time.  Binned frames go to a FrameStore (a temp
    # file every process maps) instead of being pickled back, so the set no
    # longer has to fit in RAM.  With a GPU, star catalogues come from
    # gpu_star_detect on the GUI process as each frame lands (~10 ms each);
    # SEP still measures the reference frame for the star list and FWHM.

    def _camera_gain_setting(self) -> float:
        """Settings -> Camera gain: e-/ADU off the maker's chart, 0 = EGAIN."""
        try:
            v = float(self._plot_settings().value("Exoplanet/egain", 0.0))
        except Exception:
            v = float(getattr(self, "camera_egain", 0.0) or 0.0)
        return v if np.isfinite(v) and v > 0.0 else 0.0

    def _night_check_setting(self) -> bool:
        """Settings -> Screening -> night-to-night (long-period) variability."""
        try:
            return bool(self._plot_settings().value("Exoplanet/night_to_night", False, type=bool))
        except Exception:
            return False

    def _adc_bits_setting(self):
        """Settings -> Camera ADC: 10/12/14/16, or "auto"."""
        try:
            v = str(self._plot_settings().value("Exoplanet/adc_bits", "auto")).strip().lower()
        except Exception:
            v = "auto"
        return int(v) if v in ("10", "12", "14", "16") else "auto"

    def _bin_mode(self):
        try:
            v = str(self._plot_settings().value("Exoplanet/bin_mode", "auto")).strip().lower()
        except Exception:
            v = "auto"
        return int(v) if v in ("1", "2", "3", "4") else "auto"

    @staticmethod
    def _is_cfa(img, hdr) -> bool:
        """A raw Bayer mosaic: a bin must hold whole 2x2 colour cells."""
        if img is None or getattr(img, "ndim", 0) != 2 or hdr is None:
            return False
        pat = xfit.header_str(hdr, ("BAYERPAT", "COLORTYP", "CFA-PAT", "BAYER"))
        return bool(pat) and str(pat).strip().upper() not in ("NONE", "MONO", "L", "FALSE")

    def _choose_bin_factor(self, img, hdr) -> int:
        """
        Software bin factor.  Auto measures the stars on the first frame and
        bins only while they stay sampled: 1x1 when the FWHM is under 3 px,
        otherwise the largest of 2-4 that keeps the binned FWHM >= 2.5 px.
        Binning doesn't add photometric SNR (an aperture sums the same
        photons and the same read noise either way); it cuts memory and
        time, and is free as long as stars stay a few pixels wide.
        """
        from setiastro.saspro.exoplanet_frame_worker import bin_frame
        mode = self._bin_mode()
        cfa = self._is_cfa(img, hdr)
        self._native_fwhm = float("nan")
        if mode != "auto":
            b = int(mode)
        else:
            plane = img.mean(axis=2) if img.ndim == 3 else img
            H, W = plane.shape[:2]
            c = int(min(2048, H, W))
            y0, x0 = (H - c) // 2, (W - c) // 2
            crop = np.ascontiguousarray(plane[y0:y0 + c, x0:x0 + c], dtype=np.float32)
            scale = 1.0
            if cfa:            # the colour pattern would read as structure
                crop, scale = bin_frame(crop, 2), 2.0
            try:
                cat = xphot.extract_catalog(crop, max(float(self.sep_threshold), 5.0))
                fw = float(cat["fwhm"]) * scale if cat.get("n", 0) >= 5 else float("nan")
            except Exception:
                fw = float("nan")
            self._native_fwhm = fw
            if not np.isfinite(fw):
                b = 2
            elif fw < 3.0:
                b = 1
            else:
                b = int(max(2, min(4, fw // 2.5)))
        if cfa and b % 2:
            b = max(2, b - 1)
        return int(b)

    def _gpu_detect_ok(self) -> bool:
        try:
            if not bool(self._plot_settings().value("Exoplanet/gpu_detect", True, type=bool)):
                return False
        except Exception:
            pass
        if getattr(self, "_gpu_ok", None) is None:
            try:
                from setiastro.saspro import gpu_star_detect as G
                self._gpu_ok = bool(G.gpu_available())
                print("[Exoplanet]", G.backend_note())
            except Exception as e:
                print(f"[Exoplanet] GPU star detection unavailable: {e}")
                self._gpu_ok = False
        return bool(self._gpu_ok)

    def _start_warmup(self):
        """
        The one-time startup costs of a load, on a background thread: the
        process-pool preflight (spawns a worker that imports the scientific
        stack, ~11 s), the torch / GPU probe (~11 s) and the stale frame-store
        sweep.  They used to run on the GUI thread behind "Reading the first
        frame…", freezing the window.  Started before the file dialog opens,
        they run while the user picks files.  Both probes cache for the
        session, so later loads find them done.
        """
        th = getattr(self, "_warm_thread", None)
        if th is not None and th.is_alive():
            return
        try:
            want_gpu = bool(self._plot_settings().value("Exoplanet/gpu_detect", True, type=bool))
        except Exception:
            want_gpu = True
        need_gpu = want_gpu and getattr(self, "_gpu_ok", None) is None

        def work():
            from setiastro.saspro import exoplanet_frame_worker as fw
            try:
                fw.sweep_stale()
            except Exception:
                pass
            try:
                from setiastro.saspro.worker_env import check_process_pools
                check_process_pools(log_fn=lambda m: print("[Exoplanet]", m))
            except Exception as e:
                print(f"[Exoplanet] process-pool preflight failed: {e}")
            if need_gpu:
                try:
                    from setiastro.saspro import gpu_star_detect as G
                    ok = bool(G.gpu_available())
                    print("[Exoplanet]", G.backend_note())
                except Exception as e:
                    print(f"[Exoplanet] GPU star detection unavailable: {e}")
                    ok = False
                self._gpu_ok = ok

        import threading
        th = threading.Thread(target=work, name="exoplanet-warmup", daemon=True)
        self._warm_thread = th
        th.start()

    def _await_warmup(self):
        """Wait for _start_warmup's thread with the GUI responsive."""
        th = getattr(self, "_warm_thread", None)
        if th is None:
            return
        if th.is_alive():
            self.status_label.setText("Starting worker processes and the GPU…")
            while th.is_alive():
                QApplication.processEvents()
                th.join(0.05)
        self._warm_thread = None

    def _gpu_catalog(self, plane):
        """Star catalogue from gpu_star_detect, in extract_catalog's shape; None = use SEP."""
        try:
            from setiastro.saspro import gpu_star_detect as G
            a = np.asarray(plane, dtype=np.float32)
            nd = a == 0.0
            fw = float(getattr(self, "_fwhm_hint", float("nan")))
            fw = fw if np.isfinite(fw) and fw > 0 else 3.0
            res = G.detect_stars_full(
                a, detection_sigma=float(self.sep_threshold), min_area=5,
                mask=(nd if nd.any() else None),
                min_separation=max(2, int(round(0.8 * fw))),
                stamp=int(max(11, 2 * int(math.ceil(2.5 * fw)) + 1)))
        except Exception as e:
            print(f"[Exoplanet] GPU detect failed: {e}")
            return None
        if res is None:
            return None
        fwv = np.asarray(res["fwhm"], dtype=np.float64)
        good = np.isfinite(fwv) & (fwv > 0)
        top = fwv[good][: max(5, int(0.5 * np.count_nonzero(good)))]
        return {"x": res["x"], "y": res["y"], "flux": res["flux"],
                "fwhm": float(np.median(top)) if top.size else float("nan"),
                "n": int(np.asarray(res["x"]).size), "bkg_rms": float(res["globalrms"]),
                "src": "gpu"}

    @staticmethod
    def _worker_count(n_tasks: int, per_task_bytes: float) -> int:
        """Every core, capped so the workers' peak memory fits in RAM."""
        cpu = os.cpu_count() or 4
        try:
            import psutil
            avail = float(psutil.virtual_memory().available)
        except Exception:
            avail = 8.0 * 1024 ** 3
        ram_cap = int(max(1, (0.6 * avail) // max(float(per_task_bytes), 1.0)))
        return int(max(1, min(cpu, ram_cap, max(1, int(n_tasks)))))

    def _get_pool(self, n_workers: int):
        """A spawn-context process pool (kept for the load -> measure run), or None."""
        pool = getattr(self, "_pool", None)
        if pool is not None:
            return pool
        try:
            from setiastro.saspro.worker_env import check_process_pools, process_pools_ok
            check_process_pools(log_fn=lambda m: print("[Exoplanet]", m))
            if not process_pools_ok():
                return None
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(max_workers=int(max(1, n_workers)),
                                             mp_context=_mp.get_context("spawn"))
            self._pool_n = int(max(1, n_workers))
            return self._pool
        except Exception as e:
            print(f"[Exoplanet] process pool unavailable ({type(e).__name__}: {e}); using threads")
            self._pool = None
            return None

    def _shutdown_pool(self):
        pool, self._pool = getattr(self, "_pool", None), None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def _release_frames(self):
        """Drop the cached frames and delete their FrameStore file."""
        self._cached_images = []
        self._frame_slots = []
        self._cached_catalogs = None
        st, self._frame_store = getattr(self, "_frame_store", None), None
        if st is not None:
            import gc
            gc.collect()
            st.close()

    def _put_frame(self, k: int, arr):
        """Replace frame k, in its FrameStore slot when it has one."""
        slots = getattr(self, "_frame_slots", None) or []
        st = getattr(self, "_frame_store", None)
        if (st is not None and k < len(slots) and slots[k] is not None
                and tuple(np.shape(arr)) == tuple(st.shape[1:])):
            st.mm[slots[k]] = arr
            self._cached_images[k] = np.asarray(st.mm[slots[k]])
        else:
            self._cached_images[k] = arr
            if k < len(slots):
                slots[k] = None
        self._cached_catalogs = None

    def _load_frames(self, paths, *, want_catalogs: bool) -> int:
        """
        Load, bin, saturation-map and (optionally) catalogue every frame in
        parallel, then put everything in time order.  Returns frames loaded.
        """
        from concurrent.futures import wait, FIRST_COMPLETED
        from setiastro.saspro import exoplanet_frame_worker as fw
        paths = list(paths)
        n = len(paths)
        self._release_frames()
        self._start_warmup()                 # no-op if the file dialog started it
        self.status_label.setText("Reading the first frame…")
        QApplication.processEvents()
        img0 = hdr0 = None
        for p in paths[:5]:
            try:
                img0, hdr0, _bd, _m = load_image(p)
            except Exception:
                img0 = None
            if img0 is not None:
                break
        if img0 is None:
            QMessageBox.warning(self, "Load Error", "None of the first frames could be read.")
            return 0
        img0 = np.asarray(img0, dtype=np.float32)
        b = self._choose_bin_factor(img0, hdr0)
        self.bin_factor = b
        nf = float(getattr(self, "_native_fwhm", float("nan")))
        self._fwhm_hint = nf / b if np.isfinite(nf) else float("nan")
        H, W = img0.shape[:2]
        fshape = (H // b, W // b) + tuple(img0.shape[2:])
        full_bytes = float(img0.nbytes)
        del img0

        store = None
        try:
            if fw.FrameStore.fits(n, fshape):
                store = fw.FrameStore(n, fshape)
        except Exception as e:
            print(f"[Exoplanet] frame store unavailable ({e}); frames kept in RAM")
            store = None
        self._frame_store = store
        sd = store.desc() if store is not None else None
        self._await_warmup()
        use_gpu = bool(want_catalogs) and self._gpu_detect_ok()
        cat_sigma = float(self.sep_threshold) if (want_catalogs and not use_gpu) else None
        n_work = self._worker_count(n, 3.5 * full_bytes + 3e8)
        if use_gpu:
            # The GPU detector runs here as frames land; kernel launches are
            # CPU-bound, so keep two logical cores out of the pool for it.
            n_work = max(1, min(n_work, (os.cpu_count() or 4) - 2))
        pool = self._get_pool(n_work)

        results = [None] * n
        cats = [None] * n
        loaded = [0]

        def _run(executor, idxs):
            futs = {executor.submit(fw.load_task, i, paths[i], b, sd, cat_sigma): i for i in idxs}
            pending, broken = set(futs), []
            while pending:
                fin, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for f in fin:
                    i = futs[f]
                    try:
                        r = f.result()
                    except Exception:           # a worker died (BrokenProcessPool, OOM)
                        broken.append(i)
                        continue
                    results[i] = r
                    loaded[0] += 1
                    if r.get("ok"):
                        if use_gpu:
                            v = r.get("arr")
                            v = np.asarray(store.mm[i]) if v is None else v
                            cats[i] = self._gpu_catalog(v.mean(axis=2) if v.ndim == 3 else v)
                        else:
                            cats[i] = r.get("cat")
                    self.progress_bar.setValue(loaded[0])
                    self.status_label.setText(f"Loaded {loaded[0]}/{n} frames…")
                QApplication.processEvents()
            return broken

        how = f"{getattr(self, '_pool_n', n_work)} processes" if pool is not None else "threads"
        self.status_label.setText(
            f"Loading {n} frames — {b}×{b} software bin, {how}"
            + (", GPU star detection" if use_gpu else "") + "…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(n)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        broken = _run(pool, range(n)) if pool is not None else list(range(n))
        if broken:
            if pool is not None:
                print(f"[Exoplanet] {len(broken)} frame(s) lost to a worker failure; "
                      "loading them in-process")
                self._shutdown_pool()
            with ThreadPoolExecutor(max_workers=max(2, min(8, os.cpu_count() or 4))) as tp:
                _run(tp, broken)

        def _key(i):
            r = results[i]
            t = r.get("time") if (r and r.get("ok")) else None
            return (t is None, t[0] if t else 0.0, paths[i])

        order = sorted(range(n), key=_key)
        imgs, hdrs, bds, sats, cl, slots, means, datelist, failed = ([] for _ in range(9))
        for i in order:
            r = results[i]
            if not r or not r.get("ok"):
                imgs.append(None); hdrs.append(None); bds.append(None); cl.append(None)
                sats.append(np.zeros((0, 2), np.float32)); slots.append(None)
                means.append(np.nan); datelist.append((paths[i], None))
                failed.append(os.path.basename(paths[i])
                              + (f" ({r['err']})" if r and r.get("err") else ""))
                continue
            if r.get("arr") is not None:
                imgs.append(r["arr"]); slots.append(None)
            else:
                imgs.append(np.asarray(store.mm[i])); slots.append(i)
            hdrs.append(r.get("hdr")); bds.append(r.get("bit_depth"))
            sats.append(r.get("sat") if r.get("sat") is not None else np.zeros((0, 2), np.float32))
            cl.append(cats[i]); means.append(float(r.get("mean", np.nan)))
            datelist.append((paths[i], r.get("time")))

        self.image_paths = [paths[i] for i in order]
        self._cached_images, self._cached_headers = imgs, hdrs
        self._cached_bitdepths, self._cached_satcoords = bds, sats
        self._frame_slots = slots
        self._frame_means = np.asarray(means, dtype=np.float64)
        # Low bits shared by every pixel of the 16-bit frames: 4 = a 12-bit
        # ADC shifted into 16 bits.  The smallest over the set, so one frame
        # with a true 16th bit settles it.
        _sh = [r.get("adc_shift") for r in results
               if r and r.get("ok") and r.get("adc_shift") is not None]
        self._adc_shift_detected = int(min(_sh)) if _sh else None
        self._cached_catalogs = cl if want_catalogs else None
        self._cat_sigma = float(self.sep_threshold) if want_catalogs else None
        self.exposure_time = next((v for v in (xfit.header_float(h, ("EXPOSURE", "EXPTIME"))
                                               for h in hdrs if h is not None) if v), None)
        self.airmasses = [self._header_airmass(h) for h in hdrs]
        self._set_frame_times(datelist)
        self.progress_bar.setVisible(False)
        if failed:
            QMessageBox.warning(self, "Load Error",
                                f"{len(failed)} of {n} frame(s) could not be loaded:\n"
                                + "\n".join(failed[:15]) + ("\n…" if len(failed) > 15 else ""))
        return sum(im is not None for im in imgs)

    def _frame_catalogs(self, images):
        """
        Star catalogue and mean level per frame.  Catalogues made while
        loading are reused when still valid (same frames, same σ); missing
        ones come from the GPU in this process, then SEP in worker processes.
        """
        from concurrent.futures import wait, FIRST_COMPLETED
        from setiastro.saspro.exoplanet_frame_worker import catalog_task
        n = len(images)
        cached = getattr(self, "_cached_catalogs", None)
        valid = (cached is not None and len(cached) == n
                 and getattr(self, "_cat_sigma", None) == float(self.sep_threshold))
        cats = list(cached) if valid else [None] * n
        fm = getattr(self, "_frame_means", None)
        means = (np.asarray(fm, dtype=np.float64).copy() if fm is not None and len(fm) == n
                 else np.full(n, np.nan))

        def _plane(i):
            im = images[i]
            return im.mean(axis=2) if im.ndim == 3 else im

        for i in range(n):
            if images[i] is not None and not np.isfinite(means[i]):
                means[i] = float(np.mean(_plane(i)))
        todo = [i for i in range(n) if images[i] is not None and cats[i] is None]
        if todo and self._gpu_detect_ok():
            self.status_label.setText(f"Detecting stars on the GPU ({len(todo)} frames)…")
            for k, i in enumerate(todo, 1):
                cats[i] = self._gpu_catalog(_plane(i))
                if k % 4 == 0 or k == len(todo):
                    self.progress_bar.setValue(k)
                    QApplication.processEvents()
            todo = [i for i in todo if cats[i] is None]
        if todo:
            pool = self._get_pool(self._worker_count(len(todo), 3e8))
            self.status_label.setText(
                f"Detecting stars (SEP) in {len(todo)} frames"
                + (f", {getattr(self, '_pool_n', 0)} processes…" if pool is not None else "…"))
            QApplication.processEvents()
            st = getattr(self, "_frame_store", None)
            sd = st.desc() if st is not None else None
            slots = getattr(self, "_frame_slots", None) or []
            ex = pool if pool is not None else ThreadPoolExecutor(
                max_workers=max(1, int((os.cpu_count() or 4) * 0.8)))
            try:
                futs = {}
                for i in todo:
                    s = slots[i] if (sd is not None and i < len(slots)) else None
                    futs[ex.submit(catalog_task, (s if s is not None else i),
                                   (sd if s is not None else None),
                                   (None if s is not None else np.asarray(images[i])),
                                   float(self.sep_threshold))] = i
                pending, done = set(futs), 0
                while pending:
                    fin, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    for f in fin:
                        try:
                            cats[futs[f]] = f.result()[1]
                        except Exception as e:
                            print(f"[Exoplanet] star detection failed on frame {futs[f]}: {e}")
                        done += 1
                    self.progress_bar.setValue(done)
                    QApplication.processEvents()
            finally:
                if pool is None:
                    ex.shutdown(wait=True)
        self._cached_catalogs = cats
        self._cat_sigma = float(self.sep_threshold)
        return cats, means

    def _sep_catalog(self, i: int):
        """SEP catalogue of frame i, in a worker process when one is running."""
        from setiastro.saspro.exoplanet_frame_worker import catalog_task
        slots = getattr(self, "_frame_slots", None) or []
        st = getattr(self, "_frame_store", None)
        s = slots[i] if (st is not None and i < len(slots)) else None
        args = ((s if s is not None else i), (st.desc() if s is not None else None),
                (None if s is not None else np.asarray(self._cached_images[i])),
                float(self.sep_threshold))
        pool = getattr(self, "_pool", None)
        try:
            if pool is not None:
                fut = pool.submit(catalog_task, *args)
                while not fut.done():
                    QApplication.processEvents()
                    time.sleep(0.02)
                return fut.result()[1]
        except Exception as e:
            print(f"[Exoplanet] reference SEP in a worker failed ({e}); running it here")
        try:
            return catalog_task(*args)[1]
        except Exception:
            return None

    def _input_frames_note(self) -> str:
        """
        One-line advice on the kind of frames loaded.  Measured on 485 real
        NGC660 frames: calibrated and calibrated+normalised+registered gave
        the same comparison scatter (5.3 / 5.4 ppt) once saturation was
        handled; raw frames without a flat gave 14 ppt and 80-190 ppt steps
        across a meridian flip.  Calibrated, unregistered frames are best --
        the tool tracks the field itself, so nothing needs resampling.
        """
        hdrs = [h for h in (getattr(self, "_cached_headers", None) or []) if h is not None]
        if not hdrs:
            return ""
        h = hdrs[0]
        paths = getattr(self, "image_paths", None) or []
        name = os.path.basename(paths[0]).lower() if paths else ""
        try:
            if isinstance(h, fits.Header) and "HISTORY" in h:
                hist = " ".join(str(v) for v in h["HISTORY"])
            else:
                hist = str(xfit.header_value(h, "HISTORY") or "")
        except Exception:
            hist = ""
        registered = (xfit.header_value(h, "SASDZ00") is not None or "regist" in hist.lower()
                      or re.search(r"_r(\.|_)", name) is not None)
        calibrated = (xfit.header_value(h, "CALMIN") is not None or "calibrat" in hist.lower()
                      or "flat" in hist.lower() or re.search(r"_c(\.|_)", name) is not None
                      or xfit.header_value(h, "PEDESTAL") is not None)
        bds = [str(b or "") for b in (getattr(self, "_cached_bitdepths", None) or [])]
        is_int = any(b.endswith("-bit") or "unsigned" in b for b in bds if b)
        if is_int and not calibrated:
            return ("raw frames: without a flat, dithers and meridian flips move stars across "
                    "vignetting — calibrated (dark + flat) frames are preferred")
        if registered:
            return ("registered (resampled) frames work, but calibrated unregistered frames are "
                    "preferred — the field is tracked, nothing needs resampling")
        return ""

    @staticmethod
    def _header_wcs_is_approximate(hdr) -> bool:
        """
        True when the header WCS is not a plate solution: SASpro's
        synthesised-WCS marker (SASWCSSY), or a reference point that is just
        the mount's RA/DEC with no solver keywords.
        """
        if hdr is None:
            return False
        v = xfit.header_value(hdr, "SASWCSSY")
        if v is not None and str(v).strip().upper() in ("TRUE", "T", "1", "YES"):
            return True
        for k in ("PLTSOLVD", "SOLVSRC", "A_ORDER", "ASTRSOLV"):
            if xfit.header_value(hdr, k) is not None:
                return False
        c1, c2 = xfit.header_float(hdr, ("CRVAL1",)), xfit.header_float(hdr, ("CRVAL2",))
        r, d = xfit.header_float(hdr, ("RA", "OBJCTRA")), xfit.header_float(hdr, ("DEC", "OBJCTDEC"))
        return (c1 is not None and c2 is not None and r is not None and d is not None
                and abs(c1 - r) < 1e-7 and abs(c2 - d) < 1e-7)

    def _pier_sides(self, transforms=None) -> np.ndarray:
        """
        Pier side of each frame relative to the reference frame (0 = same side,
        1 = across a meridian flip), from the first source that covers the set:
        a PIERSIDE keyword, SGP's FLIPPED keyword, the registration matrix
        SASpro writes into frames it registered (SASDZ00/04), or the tracked
        180-degree rotation.  Registered frames all face the same way, so
        tracking alone can't see a flip in them.
        """
        hdrs = getattr(self, "_cached_headers", None) or []
        n = len(hdrs)
        side = np.zeros(n, dtype=int)
        self.side_how = ""
        if n == 0:
            return side
        ref = int(getattr(self, "ref_idx", 0) or 0)
        ps = [(str(xfit.header_str(h, ("PIERSIDE", "PIER_SIDE")) or "").strip().upper()[:1]
               if h is not None else "") for h in hdrs]
        known = [p for p in ps if p in ("E", "W")]
        if len(known) >= 0.8 * n:
            r = ps[ref] if ps[ref] in ("E", "W") else known[0]
            side = np.array([0 if p in ("", r) else 1 for p in ps], dtype=int)
            self.side_how = "PIERSIDE keyword"
            return side
        # Sequence Generator Pro: FLIPPED = T after a meridian flip.  Only
        # trusted when it changes within the set (constant = no flip, or a
        # program using the word for a mirrored image).
        fv = []
        for h in hdrs:
            v = xfit.header_value(h, "FLIPPED") if h is not None else None
            fv.append(None if v is None else str(v).strip().upper() in ("TRUE", "T", "1", "YES"))
        known_f = [v for v in fv if v is not None]
        if len(known_f) >= 0.8 * n and len(set(known_f)) == 2:
            r = fv[ref] if fv[ref] is not None else known_f[0]
            side = np.array([0 if (v is None or v == r) else 1 for v in fv], dtype=int)
            self.side_how = "FLIPPED keyword"
            return side
        regs = []
        for h in hdrs:
            a = xfit.header_float(h, ("SASDZ00",)) if h is not None else None
            d = xfit.header_float(h, ("SASDZ04",)) if h is not None else None
            regs.append(None if a is None or d is None else bool(a + d < 0))
        if sum(r is not None for r in regs) >= 0.8 * n:
            rr = bool(regs[ref]) if regs[ref] is not None else False
            side = np.array([0 if (r is None or r == rr) else 1 for r in regs], dtype=int)
            if np.any(side):
                self.side_how = "registration header"
            return side
        for i, M in enumerate(transforms or []):
            if M is not None and i < n:
                A = np.asarray(M, dtype=np.float64).reshape(2, 3)
                side[i] = 1 if (A[0, 0] + A[1, 1]) < 0 else 0
        if np.any(side):
            self.side_how = "tracking (field rotated 180°)"
        return side

    def _blended_stars(self) -> set:
        """
        Stars with a neighbour at least a quarter as bright within two
        aperture radii.  Their apertures share light, and the share changes
        with seeing -- a classic source of false variability -- so a flag on
        one is tagged "blend?" for the user to judge.
        """
        pos = np.asarray(getattr(self, "star_positions", []) or [], dtype=np.float64)
        if pos.ndim != 2 or len(pos) < 2:
            return set()
        mags = self._star_mags()
        if mags.size != len(pos):
            return set()
        r = 2.0 * float(getattr(self, "aperture_radius", 0.0) or 0.0)
        if not r > 0:
            return set()
        try:
            from scipy.spatial import cKDTree
            pairs = cKDTree(pos).query_pairs(r, output_type="ndarray")
        except Exception:
            return set()
        out = set()
        for a, b in pairs:
            # flux ratio from the instrumental magnitudes
            if np.isfinite(mags[a]) and np.isfinite(mags[b]):
                if mags[b] - mags[a] < 1.5:      # b at least 1/4 of a
                    out.add(int(a))
                if mags[a] - mags[b] < 1.5:
                    out.add(int(b))
        return out

    def _has_dip_flag(self, i: int) -> bool:
        return any(str(s).startswith("DIP")
                   for s in (getattr(self, "flag_kinds", None) or {}).get(i, []))

    def _dip_aperture_test(self, i: int, m: dict):
        """
        Depth of star i's best dip at the photometry aperture and at the
        largest one (3 × FWHM), against the same comparison ensemble.

        A real eclipse dims the whole star, so its depth is the same in every
        aperture: the NGC660 EW binary measured 189 / 188 / 185 ppt at r =
        3.7 / 7.4 / 14.7 px.  Light smeared out of the aperture by a PSF
        change -- the scope settling after a meridian flip, a focus shift,
        coma toward a corner -- makes a "dip" that fades as the aperture
        grows: NGC660 #20, right after the flip, measured 417 / 76 / 7 ppt.
        A brighter neighbour inside the big aperture also dilutes a dip.

        Returns (fades, d_ap_ppt, d_big_ppt, r_ap, r_big) or None.
        """
        cube = getattr(self, "phot_cube", None)
        if cube is None or not np.isfinite(m.get("dip_t", np.nan)) \
                or not np.isfinite(m.get("dip_dur_h", np.nan)):
            return None
        F = np.asarray(cube.flux)
        k, kb = int(self.ap_index), F.shape[0] - 1
        comps = [c for c in getattr(self, "comp_stars", []) if c != i]
        wts = np.asarray([w for c, w in zip(getattr(self, "comp_stars", []),
                                            getattr(self, "comp_weights", [])) if c != i],
                         dtype=np.float64)
        if kb <= k or len(comps) < 2 or wts.size != len(comps):
            return None
        t = self._frame_time_days()
        if t.size != F.shape[2]:
            return None
        S = (np.asarray(self.session_idx) if self.session_idx is not None
             and len(self.session_idx) == t.size else np.zeros(t.size, dtype=int))
        night = S == S[int(np.nanargmin(np.abs(t - m["dip_t"])))]
        tn = t[night]
        box = np.abs(tn - m["dip_t"]) <= 0.5 * m["dip_dur_h"] / 24.0
        okn = (self.flags[i] == 0)[night]
        Xa = np.asarray(getattr(self, "airmasses", []) or [], dtype=np.float64)
        Xn = Xa[night] if Xa.size == t.size else None
        P = self._side_array(t.size)
        Pn = P[night] if P is not None else None
        mode = getattr(self, "detrend_mode", "airmass") or "none"

        def depth(r):
            # The same baseline + box model the search fitted, so the depth
            # here is the depth it found (a median-based depth disagrees with
            # it whenever the baseline has a slope).
            fc = F[r][comps]
            good = np.isfinite(fc) & (fc > 0)
            ens = (np.where(good, fc, 0.0) * wts[:, None]).sum(0) / \
                np.maximum((good * wts[:, None]).sum(0), 1e-30)
            with np.errstate(invalid="ignore", divide="ignore"):
                y = (F[r, i] / ens)[night]
            fin = okn & np.isfinite(y) & (y > 0)
            if fin.sum() < 12:
                return np.nan, np.nan
            y = y[fin] / float(np.median(y[fin]))
            return xphot.box_depth(tn[fin], y, box[fin], mode,
                                   Xn[fin] if Xn is not None else None,
                                   Pn[fin] if Pn is not None else None)

        d0, e0 = depth(k)
        d1, e1 = depth(kb)
        if not (np.isfinite(d0) and np.isfinite(d1)):
            return None
        fades = d0 > 0 and d1 < 0.5 * d0 and (d0 - d1) > 3.0 * math.hypot(e0, e1)
        radii = np.asarray(cube.radii, dtype=np.float64)
        return bool(fades), d0 * 1e3, d1 * 1e3, float(radii[k]), float(radii[kb])

    def _star_mags(self) -> np.ndarray:
        """Instrumental magnitude per star: median flux at the chosen aperture."""
        raw = getattr(self, "raw_flux", None)
        if raw is None:
            return np.zeros(0)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(np.asarray(raw) > 0, raw, np.nan), axis=1)
            return -2.5 * np.log10(med)

    def _segments(self) -> np.ndarray:
        """Per-frame (night, pier side) label: where instrumental levels may step."""
        n = 0 if self.fluxes is None and self.rel_flux_raw is None else (
            (self.rel_flux_raw if self.rel_flux_raw is not None else self.fluxes).shape[1])
        S = (np.asarray(self.session_idx) if self.session_idx is not None
             and len(self.session_idx) == n else None)
        P = getattr(self, "side_idx", None)
        P = np.asarray(P) if P is not None and len(P) == n else None
        return xphot.segment_labels(S, P, n)

    def _side_array(self, n: int):
        P = getattr(self, "side_idx", None)
        return np.asarray(P) if P is not None and len(P) == n and np.any(P) else None

    def _master_binned(self, kind: str):
        """Master dark/flat binned to the frames' software bin (cached)."""
        full = getattr(self, "_master_full", {}).get(kind)
        if full is None:
            return getattr(self, f"master_{kind}", None)
        from setiastro.saspro.exoplanet_frame_worker import bin_frame
        cache = self.__dict__.setdefault("_master_cache", {})
        key = (kind, int(getattr(self, "bin_factor", 2)))
        if key not in cache:
            cache[key] = bin_frame(full, key[1])
        return cache[key]

    def _header_airmass(self, hdr) -> float:
        """Header airmass, else from the altitude keywords, else 1 (refined later)."""
        if hdr is None:
            return 1.0
        a = xfit.header_float(hdr, ("AIRMASS",))
        if a is not None and np.isfinite(a) and a >= 1.0:
            return float(a)
        alt = xfit.header_float(hdr, ("OBJCTALT", "ALT", "ALTITUDE", "EL"))
        if alt is not None and np.isfinite(alt) and alt > 0:
            return float(self.estimate_airmass_from_altitude(alt))
        return 1.0

    def closeEvent(self, ev):
        # Workers and the frame store die with the window; a closed window
        # used to keep every loaded frame in memory.
        try:
            self._shutdown_pool()
            self._release_frames()
        except Exception:
            pass
        super().closeEvent(ev)

    def load_and_measure_subs(self):
        before = len(getattr(self, "image_paths", []))
        self.load_aligned_subs()
        after = len(getattr(self, "image_paths", []))
        if after == 0 or after == before and not self._cached_images:
            return
        self.detect_stars()
    # --------------- I/O + Calibration ----------------

    def load_raw_subs(self):
        self._start_warmup()                 # overlaps the file dialog
        settings = QSettings()
        start_dir = settings.value("ExoPlanet/lastRawFolder", os.path.expanduser("~"), type=str)
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Raw Frames", start_dir, "FITS, TIFF or XISF (*.fit *.fits *.fts *.fz *.tif *.tiff *.xisf)")
        if not paths: return
        settings.setValue("ExoPlanet/lastRawFolder", os.path.dirname(paths[0]))

        self.star_list.clear()
        self.plot_widget.clear()
        loaded = self._load_frames(paths, want_catalogs=False)
        # Raw frames are calibrated next; workers return for Measure.
        self._shutdown_pool()
        self.status_label.setText(f"Loaded {loaded}/{len(paths)} raw frames "
                                  f"({self.bin_factor}×{self.bin_factor} software bin)")

    def load_aligned_subs(self) -> bool:
        self._start_warmup()                 # overlaps the file dialog
        settings = QSettings()
        start_dir = settings.value("ExoPlanet/lastAlignedFolder", os.path.expanduser("~"), type=str)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Aligned Frames", start_dir,
            "FITS, TIFF or XISF (*.fit *.fits *.fts *.fz *.tif *.tiff *.xisf)"
        )
        if not paths:
            self.status_label.setText("Load canceled.")
            return False

        settings.setValue("ExoPlanet/lastAlignedFolder", os.path.dirname(paths[0]))

        loaded = self._load_frames(paths, want_catalogs=True)
        self.status_label.setText(f"Loaded {loaded}/{len(paths)} frames "
                                  f"({self.bin_factor}×{self.bin_factor} software bin)")
        return loaded > 0

    def load_masters(self):
        settings = QSettings()
        last_master_dir = settings.value("ExoPlanet/lastMasterFolder", os.path.expanduser("~"), type=str)
        sender = self.sender()
        dlg = QFileDialog(self, "Select Master File", last_master_dir, "FITS, TIFF or XISF (*.fit *.fits *.tif *.tiff *.xisf)")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        if not dlg.exec(): return
        path = dlg.selectedFiles()[0]
        settings.setValue("ExoPlanet/lastMasterFolder", os.path.dirname(path))

        img, hdr, bit_depth, is_mono = load_image(path)
        if img is None:
            QMessageBox.warning(self, "Load Error", f"Failed to load master file:\n{path}")
            return

        img = img.astype(np.float32)
        # Kept unbinned too: the software bin is chosen when the frames load
        # (Auto), so the master is binned to match at calibration time.
        kind = "dark" if "Dark" in sender.text() else "flat"
        self.__dict__.setdefault("_master_full", {})[kind] = img
        self.__dict__.setdefault("_master_cache", {}).clear()
        binned = self._master_binned(kind)

        if "Dark" in sender.text():
            if self.master_flat is not None and not self._shapes_compatible(binned, self.master_flat):
                QMessageBox.warning(self, "Shape Mismatch", "This master dark (binned) doesn’t match your existing flat.")
                return
            self.master_dark = binned
            self.dark_status_label.setText("Dark: ✅")
            self.dark_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")
        else:
            if self.master_dark is not None and not self._shapes_compatible(self.master_dark, binned):
                QMessageBox.warning(self, "Shape Mismatch", "This master flat (binned) doesn’t match your existing dark.")
                return
            self.master_flat = binned
            self.flat_status_label.setText("Flat: ✅")
            self.flat_status_label.setStyleSheet("color: #00cc66; font-weight: bold;")

    def _try_wcs_from_header(self, hdr, plane):
        H = self._get_best_header_for_wcs(hdr)
        if H is None:
            return None

        try:
            w = WCS(H, naxis=2, relax=True)
            if not w.has_celestial:
                return None

            # sanity check by transforming image center
            h, wimg = plane.shape[:2]
            sky = w.pixel_to_world(wimg / 2, h / 2)
            _ = float(sky.ra.deg), float(sky.dec.deg)
            return w
        except Exception:
            return None

    def _shapes_compatible(self, master: np.ndarray, other: np.ndarray) -> bool:
        if master.shape == other.shape:
            return True
        if master.ndim == 2 and other.ndim == 3 and other.shape[:2] == master.shape:
            return True
        if other.ndim == 2 and master.ndim == 3 and master.shape[:2] == other.shape:
            return True
        return False

    def calibrate_and_align(self):
        if not self._cached_images:
            QMessageBox.warning(self, "Calibrate", "Load raw subs first.")
            return
        self.status_label.setText("Calibrating & aligning frames…")
        self.progress_bar.setVisible(True)
        n = len(self._cached_images)
        self.progress_bar.setMaximum(n)

        reference_image_2d = None
        for i, (img, hdr) in enumerate(zip(self._cached_images, self._cached_headers), start=1):
            if img is None:
                self.progress_bar.setValue(i)
                continue
            if self.master_dark is not None:
                img = img.astype(np.float32) - self._master_binned("dark")
            if self.master_flat is not None:
                img = apply_flat_division_numba(img, self._master_binned("flat"))
            # Mono stays mono.  This used to replicate mono frames into three
            # identical channels, which tripled memory and made the per-channel
            # (TB/TG/TR) pass treat mono data as colour.

            plane = img if img.ndim == 2 else img.mean(axis=2)

            if reference_image_2d is None:
                reference_image_2d = plane.copy()

            delta = StarRegistrationWorker.compute_affine_transform_astroalign(plane, reference_image_2d)
            if delta is None:
                delta = IDENTITY_2x3

            img_aligned = StarRegistrationThread.apply_affine_transform_static(img, delta)
            self._put_frame(i - 1, img_aligned)
            # Saturated pixels were located on the raw frame; carry them through
            # the same warp (delta maps source -> reference, cv2 convention).
            try:
                if i - 1 < len(self._cached_satcoords):
                    self._cached_satcoords[i-1] = xphot.transform_coords(
                        self._cached_satcoords[i-1], delta)
            except Exception:
                pass
            self.progress_bar.setValue(i)
            QApplication.processEvents()

        self.progress_bar.setVisible(False)
        self.status_label.setText("Calibration & alignment complete")

    def save_aligned_frames(self):
        if not self._cached_images:
            QMessageBox.warning(self, "Save Aligned Frames", "No images to save. Run Calibrate & Align first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose Output Folder")
        if not out_dir:
            return
        for i, orig_path in enumerate(self.image_paths):
            img = self._cached_images[i]
            ext = os.path.splitext(orig_path)[1].lstrip(".").lower()
            fmt = ext if ext in ("fits","fit","tiff","tif","xisf","png","jpg","jpeg") else "fits"
            hdr = self._cached_headers[i] if hasattr(self, "_cached_headers") and i < len(self._cached_headers) else None
            base = os.path.splitext(os.path.basename(orig_path))[0]
            out_name = f"{base}_aligned.{fmt}"
            out_path = os.path.join(out_dir, out_name)
            save_image(
                img_array=img, filename=out_path, original_format=fmt, bit_depth=None,
                original_header=hdr, is_mono=(img.ndim==2), image_meta=None, file_meta=None
            )
        QMessageBox.information(self, "Save Complete", f"Saved {len(self._cached_images)} aligned frames to:\n{out_dir}")

    # --------------- Detection + Photometry ----------------
    def _seed_header_for_astap(self, ref_idx: int) -> fits.Header | None:
        """
        Build a *real* FITS header to seed ASTAP, harvested from the reference
        frame's original header. We preserve RA/Dec (OBJCTRA/OBJCTDEC or CRVAL1/2),
        size (NAXIS*), basic camera hints (PIXSZ, BINNING, FOCALLEN) if present.
        """
        if not (0 <= ref_idx < len(self._cached_headers)):
            return None

        src = self._cached_headers[ref_idx]
        H = fits.Header()

        # try to copy directly if it's already a FITS Header
        if isinstance(src, fits.Header):
            H = src.copy()
        elif isinstance(src, dict):
            # Could be a nested XISF-like dict. Look for FITSKeywords first.
            kw = None
            if "image_meta" in src and isinstance(src["image_meta"], dict):
                kw = src["image_meta"].get("FITSKeywords", None)
            if kw is None:
                kw = src.get("FITSKeywords", None)
            if isinstance(kw, dict):
                for k, v in kw.items():
                    try:
                        # XISF stores [{'value':...}], FITS-like dicts store raw
                        if isinstance(v, list):
                            vv = v[0].get("value", None)
                        else:
                            vv = v
                        if vv is not None:
                            H[k] = vv
                    except Exception:
                        pass
            else:
                # best-effort: flat dict of scalars
                for k, v in src.items():
                    try:
                        H[k] = v
                    except Exception:
                        pass

        # be sure image size is present (ASTAP likes NAXIS1/2)
        if self._cached_images and self._cached_images[ref_idx] is not None:
            img = self._cached_images[ref_idx]
            h, w = (img.shape if img.ndim == 2 else img.shape[:2])
            H.setdefault("NAXIS", 2)
            H["NAXIS1"] = w
            H["NAXIS2"] = h

        # If we only have OBJCTRA/OBJCTDEC strings, just leave them — your
        # _build_astap_seed() handles these. If we *also* have CRVAL*, keep them.
        # Do NOT inject WCS—we want ASTAP to solve, not be constrained by stale WCS.
        # (_solve_numpy_with_astap will strip WCS keys before writing temp FITS)

        # optional: exposure time
        if self.exposure_time is not None:
            H.setdefault("EXPTIME", float(self.exposure_time))

        return H if len(H) else None

    def _coerce_seed_header(self, hdr_in, plane) -> fits.Header:
        """
        Make a real fits.Header usable by _build_astap_seed():
        - copy fields from FITS, XISF-like dicts, or flat dicts
        - ensure NAXIS/NAXIS1/NAXIS2
        - try to expose RA/Dec in a form the seeder can use
            (OBJCTRA/OBJCTDEC strings and/or CRVAL1/CRVAL2 degrees).
        """
        H = fits.Header()

        # 1) copy what we can
        if isinstance(hdr_in, fits.Header):
            H = hdr_in.copy()
        elif isinstance(hdr_in, dict):
            # XISF-style nested?
            kw = None
            if "image_meta" in hdr_in and isinstance(hdr_in["image_meta"], dict):
                kw = hdr_in["image_meta"].get("FITSKeywords")
            if kw is None:
                kw = hdr_in.get("FITSKeywords")
            if isinstance(kw, dict):
                for k, v in kw.items():
                    try:
                        vv = v[0]["value"] if isinstance(v, list) else v
                        if vv is not None:
                            H[k] = vv
                    except Exception:
                        pass
            else:
                # flat dict of scalars
                for k, v in hdr_in.items():
                    try:
                        H[k] = v
                    except Exception:
                        pass

        # 2) ensure image size
        h, w = (plane.shape if plane.ndim == 2 else plane.shape[:2])
        H.setdefault("NAXIS", 2)
        H["NAXIS1"] = int(w)
        H["NAXIS2"] = int(h)

        # 3) try to normalize RA/Dec
        # If OBJCTRA/OBJCTDEC are present as strings, keep them.
        # Otherwise, if we find numeric RA/DEC/CRVAL*, ensure CRVAL1/2 are set.
        def _try_deg(val):
            try:
                return float(val)
            except Exception:
                return None

        ra_deg = None
        dec_deg = None

        # prefer CRVAL1/2 if they seem finite
        ra_deg = _try_deg(H.get("CRVAL1"))
        dec_deg = _try_deg(H.get("CRVAL2"))

        if ra_deg is None or dec_deg is None:
            # common alternates
            for rakey in ("RA", "OBJCTRA", "OBJRA"):
                if rakey in H:
                    try:
                        # could be sexagesimal string
                        ra_deg = SkyCoord(H[rakey], H.get("OBJCTDEC", None) or H.get("DEC", None), unit=(u.hourangle, u.deg)).ra.deg
                        dec_deg = SkyCoord(H[rakey], H.get("OBJCTDEC", None) or H.get("DEC", None), unit=(u.hourangle, u.deg)).dec.deg
                        break
                    except Exception:
                        pass
            if ra_deg is None and "RA" in H and "DEC" in H:
                ra_deg = _try_deg(H["RA"])
                dec_deg = _try_deg(H["DEC"])

        # Set CRVAL* if we have clean degrees
        if ra_deg is not None and dec_deg is not None:
            H["CRVAL1"] = float(ra_deg)
            H["CRVAL2"] = float(dec_deg)

            # Also supply OBJCTRA/OBJCTDEC sexagesimal (helps some ASTAP setups)
            try:
                c = SkyCoord(ra_deg*u.deg, dec_deg*u.deg, frame="icrs")
                H.setdefault("OBJCTRA",  c.ra.to_string(unit=u.hour, sep=":", precision=2, pad=True))
                H.setdefault("OBJCTDEC", c.dec.to_string(unit=u.deg, sep=":", precision=1, pad=True, alwayssign=True))
            except Exception:
                pass

        # Optional: if you have pixel size / focal length / binning in the original
        # header, leaving them in place is good; _build_astap_seed will use them.

        return H

    def _get_best_header_for_wcs(self, hdr):
        if isinstance(hdr, fits.Header):
            return hdr

        if isinstance(hdr, dict):
            # first try canonical SASpro metadata keys
            for k in ("wcs_header", "original_header", "header", "fits_header"):
                h = hdr.get(k)
                if isinstance(h, fits.Header):
                    return h

            # XISF-style FITSKeywords fallback
            kw = None
            if "image_meta" in hdr and isinstance(hdr["image_meta"], dict):
                kw = hdr["image_meta"].get("FITSKeywords")
            if kw is None:
                kw = hdr.get("FITSKeywords")

            if isinstance(kw, dict):
                H = fits.Header()
                for k, v in kw.items():
                    try:
                        vv = v[0]["value"] if isinstance(v, list) else v
                        if vv is not None:
                            H[k] = vv
                    except Exception:
                        pass
                return H if len(H) else None

        return None

    def _try_existing_wcs(self, hdr):
        H = self._get_best_header_for_wcs(hdr)
        if H is None:
            return None
        try:
            w = WCS(H, naxis=2, relax=True)
            if not w.has_celestial:
                return None
            return w
        except Exception:
            return None


    def _photometry_gain(self):
        """
        Electrons per data unit, for sep's `gain`, or None when unknown.

        Order: explicit user setting, then EGAIN from the frame headers.
        Both are in the camera ADC's own units (gain charts, ZWO/QHY
        drivers), so they are divided by 2^(16 - ADC bits) for 16-bit data:
        a 12-bit ADC is written into the file x16.  Float frames whose values
        sit around 0..1 are taken at load_image's 0..1 = 0..65535 ADU scale
        (SASpro's, PixInsight's and most calibrated output); float frames in
        ADU (Siril) at one ADU per unit.  Returns None when no gain is known
        -- guessing one is worse than dropping the Poisson term.
        Sets self._photometry_gain_note to say where the number came from.
        """
        self._photometry_gain_note = ""
        b2 = float(int(getattr(self, "bin_factor", 2)) ** 2)
        bds = [str(b).strip() for b in (getattr(self, "_cached_bitdepths", None) or []) if b]
        is_float = any(b.startswith(("32-bit floating", "64-bit floating")) for b in bds)
        if is_float:
            # Pipelines such as LCO's BANZAI deliver float frames already in
            # electrons (BUNIT='electron', GAIN rewritten to 1).  Float data
            # pass through load_image untouched, so one unit is one electron.
            for hdr in (getattr(self, "_cached_headers", None) or []):
                if hdr is None:
                    continue
                bunit = str(xfit.header_value(hdr, "BUNIT") or "").strip().lower()
                if bunit.startswith("electron") or bunit in ("e-", "e", "e-/pix"):
                    self._photometry_gain_note = "frames in electrons (BUNIT)"
                    # 1 e-/unit x b^2 for the averaging b x b software bin
                    return b2
                break

        egain, src = self._camera_gain_setting(), "Settings"
        if egain <= 0.0:
            egain, src = 0.0, "EGAIN"
            for hdr in (getattr(self, "_cached_headers", None) or []):
                v = _header_scalar(hdr, _EGAIN_KEYS)
                # Sanity band: real e-/ADU sits around 0.1-10. A value of
                # 100+ is almost certainly a CMOS gain setting written to
                # the wrong keyword, and using it would under-weight the
                # shot noise by the same factor.
                if v is not None and 0.01 <= v <= 100.0:
                    egain = float(v)
                    break
        if egain <= 0.0:
            return None

        scale = _ADU_PER_UNIT.get(bds[0]) if bds else None
        if scale is None and is_float:
            fm = np.asarray(getattr(self, "_frame_means", []), dtype=np.float64)
            fm = fm[np.isfinite(fm)]
            scale = 65535.0 if (fm.size == 0 or float(np.median(fm)) < 2.0) else 1.0
        if not scale:
            return None

        shift, how = 0, ""
        if scale == 65535.0 or is_float:          # data in 16-bit file ADU
            bits = self._adc_bits_setting()
            if bits == "auto":
                det = getattr(self, "_adc_shift_detected", None)
                if det is not None:
                    shift, how = int(det), "from the raw frames"
                elif is_float:
                    how = "float frames hide it; set Settings > Camera ADC"
            else:
                shift, how = max(0, 16 - int(bits)), "Settings"
        note = f"{egain:.3g} e-/ADU ({src})"
        if shift:
            note += f" ÷ {2 ** shift} for a {16 - shift}-bit ADC ({how})"
        elif how:
            note += f", ADC taken as 16-bit ({how})"
        self._photometry_gain_note = note

        # The software bin averages, so each binned unit stands for b^2
        # pixels' worth of electrons.
        return float(egain) / float(2 ** shift) * float(scale) * b2

    def detect_stars(self):
        """
        Measure every frame.

          1. SEP catalogue per frame (parallel) -> FWHM, star count.
          2. Best frame becomes the reference; plate-solve it.
          3. Track the field: reference -> frame affine per frame from the
             catalogues.  Apertures follow the stars; nothing is resampled.
             (The old code measured fixed reference positions on every frame,
             so any drift walked the stars out of their apertures -- target
             and comparisons each going to ~0 at different times is exactly
             the "flux 0 to 3" light curve.)
          4. Aperture photometry at a grid of radii with a sigma-clipped
             annulus median sky, saturation / edge / no-data flags.
          5. Weighted ensemble with variable-comp rejection, frame rejection,
             isolated-outlier flags, then detrending and screening.
        """
        n_frames = len(self.image_paths)
        if n_frames == 0:
            QMessageBox.warning(self, "Photometry", "Load frames first.")
            return

        # 0) ensure frames are cached (binned, with headers and saturation maps)
        if not getattr(self, "_cached_images", None) or len(self._cached_images) != n_frames:
            if self._load_frames(list(self.image_paths), want_catalogs=True) == 0:
                QMessageBox.warning(self, "Photometry", "None of the frames could be loaded.")
                return
            n_frames = len(self.image_paths)
        if len(getattr(self, "_cached_satcoords", []) or []) != n_frames:
            self._cached_satcoords = [np.zeros((0, 2), np.float32)] * n_frames
        if len(getattr(self, "_cached_headers", []) or []) != n_frames:
            self._cached_headers = [None] * n_frames

        images = self._cached_images
        if not any(im is not None for im in images):
            QMessageBox.warning(self, "Photometry", "None of the frames could be loaded.")
            return

        cpu_cnt = multiprocessing.cpu_count()
        n_workers = max(1, int(cpu_cnt * 0.8))

        # --- PASS 1: per-frame catalogues ----------------------------------
        # Made while loading when possible (GPU, or SEP in the load workers);
        # anything missing is detected here.
        self.status_label.setText("Detecting stars…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(n_frames)
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        cats, means = self._frame_catalogs(images)

        def quality(i):
            c = cats[i]
            if not c or c["n"] < 3 or not np.isfinite(c["fwhm"]):
                return -1.0
            return c["n"] / (c["fwhm"] * abs(means[i]) + 1e-8)

        ref_idx = max(range(n_frames), key=quality)
        if quality(ref_idx) < 0:
            QMessageBox.warning(self, "No Stars", "No stars found in any frame.")
            self.progress_bar.setVisible(False)
            self._shutdown_pool()
            return

        # --- WCS on the reference ------------------------------------------
        self.ref_idx = ref_idx
        ref_img = images[ref_idx]
        # A copy: the solver and the overlay keep it, and a live view would
        # pin the frame store's file open.
        plane = np.array(ref_img.mean(axis=2) if ref_img.ndim == 3 else ref_img,
                         dtype=np.float32)
        hdr = self._cached_headers[ref_idx]
        self._solve_reference(plane, hdr)

        # --- reference star list ---------------------------------------------
        # GPU catalogues track the field; the stars to MEASURE, and the FWHM
        # the apertures scale with, come from SEP (deblended, the established
        # FWHM convention) on the reference frame.  GPU FWHMs (moments above
        # threshold) run smaller, so they are rescaled by the reference's ratio.
        rc = cats[ref_idx]
        rc_phot, fw_scale = rc, 1.0
        if rc.get("src") == "gpu":
            self.status_label.setText("Measuring the reference frame (SEP)…")
            QApplication.processEvents()
            sep_rc = self._sep_catalog(ref_idx)
            if sep_rc and sep_rc.get("n", 0) >= 3:
                rc_phot = sep_rc
                if (np.isfinite(sep_rc["fwhm"]) and np.isfinite(rc["fwhm"]) and rc["fwhm"] > 0):
                    fw_scale = float(sep_rc["fwhm"]) / float(rc["fwhm"])
        self._gpu_fwhm_scale = fw_scale
        h, w = plane.shape[:2]
        bf = self.border_fraction
        xs_all, ys_all = rc_phot["x"], rc_phot["y"]
        keep_border = ((xs_all > w * bf) & (xs_all < w * (1 - bf)) &
                       (ys_all > h * bf) & (ys_all < h * (1 - bf)))
        ref_xy = np.column_stack([xs_all[keep_border], ys_all[keep_border]])
        if len(ref_xy) == 0:
            QMessageBox.warning(self, "No Stars", "No stars found inside the border margin.")
            self.progress_bar.setVisible(False)
            self._shutdown_pool()
            return
        fwhms = [c["fwhm"] * (fw_scale if c.get("src") == "gpu" else 1.0)
                 for c in cats if c and c["n"] >= 3 and np.isfinite(c["fwhm"])]
        fwhm = float(np.median(fwhms)) if fwhms else float(rc_phot["fwhm"])
        self.median_fwhm = fwhm

        # --- tracking ---------------------------------------------------------
        self.status_label.setText("Tracking the field across frames…")
        QApplication.processEvents()
        Ms, infos = xphot.compute_transforms(cats, ref_idx, tol=max(2.0, 1.5 * fwhm))
        self.frame_transforms, self.frame_track_info = Ms, infos
        self.side_idx = self._pier_sides(Ms)
        if np.any(self.side_idx):
            print(f"[Exoplanet] meridian flip: {int(np.count_nonzero(self.side_idx))}/{n_frames} "
                  f"frames on the other pier side ({self.side_how}); levels fitted per side")
        n_fail = int(sum(M is None for M in Ms))
        if n_fail:
            print(f"[Exoplanet] tracking failed on {n_fail}/{n_frames} frames: "
                  + "; ".join(f"{i}: {s}" for i, (M, s) in enumerate(zip(Ms, infos)) if M is None)[:400])

        # --- PASS 2: multi-aperture photometry --------------------------------
        ap = xphot.aperture_setup(fwhm, recenter=bool(getattr(self, "recenter_apertures", True)))
        _gain = self._photometry_gain()
        self._photometry_gain_used = _gain
        if _gain:
            print(f"[DEBUG] photometric gain: {_gain:.4g} e-/data-unit — "
                  f"{getattr(self, '_photometry_gain_note', '')}")
        else:
            print("[DEBUG] photometric gain unknown — Poisson term omitted; "
                  "errors reflect sky/read noise only (the ensemble weights and "
                  "the transit fit rescale them empirically). Set the camera "
                  "gain in Settings for a complete error budget.")
        pool = self._get_pool(self._worker_count(n_frames, 3e8))
        self.status_label.setText(
            f"Aperture photometry: {len(ref_xy)} stars × {n_frames} frames × {len(ap.radii)} radii"
            + (f" ({getattr(self, '_pool_n', 0)} processes)…" if pool is not None else "…"))
        self.progress_bar.setMaximum(n_frames)
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        def _prog(done, total):
            self.progress_bar.setValue(done)
            QApplication.processEvents()

        st = getattr(self, "_frame_store", None)
        slots = self._frame_slots if len(getattr(self, "_frame_slots", []) or []) == n_frames else None
        try:
            cube = xphot.run_photometry(images, Ms, ref_xy, ap, gain=_gain,
                                        sat_coords=self._cached_satcoords,
                                        n_workers=n_workers, progress=_prog,
                                        executor=pool, store=(st.desc() if st is not None else None),
                                        slots=slots)
        except Exception as e:
            if pool is None:
                raise
            print(f"[Exoplanet] process photometry failed ({type(e).__name__}: {e}); using threads")
            self._shutdown_pool()
            cube = xphot.run_photometry(images, Ms, ref_xy, ap, gain=_gain,
                                        sat_coords=self._cached_satcoords,
                                        n_workers=n_workers, progress=_prog)
        # Idle workers hold ~150 MB each; the next load starts fresh ones.
        self._shutdown_pool()

        # --- differential photometry -------------------------------------------
        self.status_label.setText("Building the comparison ensemble…")
        QApplication.processEvents()
        t_days = self._frame_time_days()
        self._set_sessions()
        ap_idx = None
        mult = float(getattr(self, "aperture_mult", 0.0) or 0.0)
        if mult > 0:
            ap_idx = int(np.argmin(np.abs(cube.radii - max(1.5, mult * fwhm))))
        dr = xphot.differential_photometry(cube, max_comps=int(self.ensemble_k),
                                           ap_index=ap_idx, times=t_days,
                                           sessions=xphot.segment_labels(
                                               self.session_idx, self.side_idx, n_frames))

        # Drop stars that are unusable most of the time (off the edge, in
        # no-data borders, no sky).  Saturated stars stay in the list, tagged,
        # so a saturated target is reported rather than silently missing.
        unusable = (dr.flags & (xphot.FLAG_EDGE | xphot.FLAG_SKY | xphot.FLAG_INVALID)) != 0
        live = ~np.all((dr.flags & xphot.FLAG_BADFRAME) != 0, axis=0)
        if np.any(live):
            frac_ok = (~unusable[:, live]).mean(axis=1)
        else:
            frac_ok = np.zeros(unusable.shape[0])
        keep = frac_ok >= 0.5
        if not np.any(keep):
            QMessageBox.warning(self, "Photometry",
                                "No star was measurable in at least half of the frames.")
            self.progress_bar.setVisible(False)
            return
        self._apply_star_subset(cube, dr, ref_xy, keep)

        # Airmass now that the field centre is known (header first, then
        # computed from site + time when the header has none).
        try:
            radec_c = (self.wcs_ra, self.wcs_dec) if self.wcs_ra is not None else None
            self._resolve_airmass(radec_c)
        except Exception as e:
            print(f"[Exoplanet] airmass resolution failed: {e}")

        # Supplementary per-channel R/G/B pass for color inputs, at the
        # tracked per-frame positions.
        self._run_multiband_photometry(self.phot_cube.x, self.phot_cube.y, keep_mask=None)

        self.status_label.setText("Screening light curves…")
        QApplication.processEvents()
        self._refresh_detrend_and_screen()
        self._populate_star_list()

        # overlay & finish: use the real reference plane
        self._show_reference_with_circles(plane, self.star_positions)
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.fit_transit_btn.setEnabled(True)
        self._on_threshold_changed(self.threshold_slider.value())
        n_bad = int(np.count_nonzero(np.all((self.flags & xphot.FLAG_BADFRAME) != 0, axis=0)))
        msg = (f"{len(self.star_positions)} stars · ensemble of {len(self.comp_stars)} comps"
               f" ({len(self.rejected_comps)} rejected as variable/noisy) · aperture "
               f"r={self.aperture_radius:.2f}px · FWHM {fwhm:.2f}px")
        if n_fail:
            msg += f" · tracking failed on {n_fail} frame(s)"
        if n_bad:
            msg += f" · {n_bad} frame(s) rejected"
        if self.times is None:
            msg += " · no timestamps: x-axis is frame number"
        if np.any(getattr(self, "side_idx", None) if getattr(self, "side_idx", None) is not None else []):
            msg += f" · meridian flip handled ({self.side_how})"
        note = self._input_frames_note()
        if note:
            msg += " · " + note
        self._photometry_status = msg
        self.status_label.setText(msg + " — " + self.status_label.text())

    # ---- photometry state helpers ---------------------------------------------

    def _set_frame_times(self, datelist):
        jd = np.array([d[1][0] if d[1] else np.nan for d in datelist], dtype=np.float64)
        exp = np.array([d[1][1] if (d[1] and d[1][1] is not None) else np.nan
                        for d in datelist], dtype=np.float64)
        self._frame_jd_utc = jd
        self._frame_exptime = exp
        hows = [d[1][2] for d in datelist if d[1]]
        self._time_how = hows[0] if hows else ""
        if jd.size and np.all(np.isfinite(jd)):
            self.times = Time(jd, format="jd", scale="utc")
        else:
            self.times = None
            if jd.size:
                print(f"[Exoplanet] {int(np.count_nonzero(~np.isfinite(jd)))}/{jd.size} frames "
                      "have no usable timestamp — time axis falls back to frame number")

    def _frame_time_days(self) -> np.ndarray:
        """Days since the first frame (mid-exposure), or a nominal cadence."""
        n = len(getattr(self, "_cached_images", []) or [])
        jd = self._frame_jd_utc
        if jd is not None and len(jd) == n and n and np.all(np.isfinite(jd)):
            return np.asarray(jd, dtype=np.float64) - float(jd[0])
        exp = self._frame_exptime
        cad = float(np.nanmedian(exp)) if exp is not None and np.any(np.isfinite(exp)) else 60.0
        return np.arange(n, dtype=np.float64) * max(cad, 1.0) / 86400.0

    # ---- observing nights ------------------------------------------------------

    def _set_sessions(self):
        """Group frames into observing nights and fill the Night selector."""
        n = len(getattr(self, "_cached_images", []) or [])
        jd = self._frame_jd_utc
        if jd is not None and len(jd) == n and n and np.all(np.isfinite(jd)):
            self.session_idx, self.sessions = xfit.assign_sessions(
                jd, getattr(self, "_cached_headers", None))
        else:
            self.session_idx = np.zeros(n, dtype=int)
            self.sessions = [{"index": 0, "label": f"All frames — {n} frames",
                              "date": "", "instrument": "", "start_jd": float("nan"),
                              "end_jd": float("nan"), "n": n}]
        self.screen_by_session = {}
        self.night_combo.blockSignals(True)
        self.night_combo.clear()
        self.night_combo.addItem(f"All nights ({len(self.sessions)})", None)
        for s in self.sessions:
            self.night_combo.addItem(s["label"], int(s["index"]))
        self.night_combo.setCurrentIndex(0)
        self.night_combo.blockSignals(False)
        multi = len(self.sessions) > 1
        self.night_label.setVisible(multi)
        self.night_combo.setVisible(multi)

    def _active_session(self):
        """Selected night index, or None for All nights."""
        d = self.night_combo.currentData() if hasattr(self, "night_combo") else None
        return None if d is None else int(d)

    def _active_frame_mask(self) -> np.ndarray:
        n = 0 if self.fluxes is None else self.fluxes.shape[1]
        s = self._active_session()
        if s is None or self.session_idx is None or len(self.session_idx) != n:
            return np.ones(n, dtype=bool)
        return np.asarray(self.session_idx) == s

    def _active_screen(self) -> dict:
        """Screening metrics for the selected night (computed once, cached)."""
        s = self._active_session()
        if s is None:
            return self.screen
        return self._session_screen(s)

    def _session_screen(self, s: int) -> dict:
        """Screening metrics of night s alone (computed once, cached)."""
        if s in self.screen_by_session:
            return self.screen_by_session[s]
        rel = getattr(self, "rel_flux_raw", None)
        if rel is None or self.session_idx is None or len(self.session_idx) != rel.shape[1]:
            return {}
        m = np.asarray(self.session_idx) == s
        t = self._frame_time_days()[m]
        X = (np.asarray(self.airmasses, dtype=np.float64)[m]
             if len(self.airmasses) == rel.shape[1] else None)
        mode = getattr(self, "detrend_mode", "airmass") or "none"
        P = self._side_array(rel.shape[1])
        P = P[m] if P is not None else None
        out = {}
        for i in range(rel.shape[0]):
            ok = (self.flags[i][m] == 0) & np.isfinite(rel[i][m])
            out[i] = xphot.screen_star(t, rel[i][m], self.rel_err_raw[i][m], ok,
                                       baseline=mode, airmass=X, side=P)
            if i % 100 == 0:
                QApplication.processEvents()
        self.screen_by_session[s] = out
        return out

    def on_night_changed(self, _idx: int = 0):
        if getattr(self, "rel_flux_raw", None) is None:
            return
        self._on_threshold_changed(self.threshold_slider.value())
        self.update_plot_for_selection()

    def _session_axis(self):
        """
        Plot x (hours) for every frame plus night dividers.

        One night: hours since that night's first frame.  All nights: each
        night's own hours laid end to end with the daytime gaps collapsed to
        half an hour, so several nights stay readable on one axis.
        Returns (x, dividers[(x, label)], to_x(t_days) -> x).
        """
        t_h = self._frame_time_days() * 24.0
        n = t_h.size
        if self.times is None:
            x = np.arange(n, dtype=np.float64)
            return x, [], (lambda td: float("nan"))
        s_idx = (np.asarray(self.session_idx) if self.session_idx is not None
                 and len(self.session_idx) == n else np.zeros(n, dtype=int))
        active = self._active_session()
        x = np.full(n, np.nan)
        starts, offsets, dividers = {}, {}, []
        off = 0.0
        for s in self.sessions:
            k = int(s["index"])
            m = s_idx == k
            if not np.any(m) or (active is not None and k != active):
                continue
            t0 = float(t_h[m].min())
            if active is None and dividers is not None and off > 0:
                dividers.append((off - 0.25, s.get("date") or s["label"]))
            starts[k], offsets[k] = t0, off
            x[m] = t_h[m] - t0 + off
            off += float(t_h[m].max() - t0) + 0.5
        if active is None and self.sessions and len(self.sessions) > 1:
            first = self.sessions[0]
            dividers.insert(0, (None, first.get("date") or first["label"]))

        def to_x(td):
            th = float(td) * 24.0
            j = int(np.nanargmin(np.abs(t_h - th)))
            k = int(s_idx[j])
            if k not in starts:
                return float("nan")
            return th - starts[k] + offsets[k]
        return x, dividers, to_x

    def _apply_star_subset(self, cube, dr, ref_xy, keep):
        idx = np.where(keep)[0]
        old_to_new = -np.ones(len(keep), dtype=int)
        old_to_new[idx] = np.arange(idx.size)
        self.phot_cube = xphot.PhotometryCube(
            radii=cube.radii, flux=cube.flux[:, idx], err=cube.err[:, idx],
            nonpos=cube.nonpos[:, idx], flags=cube.flags[idx], x=cube.x[idx],
            y=cube.y[idx], sky=cube.sky[idx], peak=cube.peak[idx],
            frame_ok=cube.frame_ok, frame_info=list(getattr(self, "frame_track_info", [])),
            setup=cube.setup)
        ens = dr.ensemble
        self.comp_stars = [int(old_to_new[c]) for c in ens.comps if old_to_new[c] >= 0]
        self.comp_weights = [float(wt) for c, wt in zip(ens.comps, ens.weights) if old_to_new[c] >= 0]
        self.rejected_comps = [int(old_to_new[c]) for c in ens.rejected if old_to_new[c] >= 0]
        self.transparency = dr.transparency
        self.ap_index = int(dr.ap_index)
        self.aperture_radius = float(cube.radii[dr.ap_index])
        self.aperture_scores = list(dr.aperture_scores)
        flags = dr.flags[idx].astype(np.int16)
        self.flags = flags
        self.rel_flux_raw = dr.rel[idx].astype(np.float64)
        self.rel_err_raw = dr.rel_err[idx].astype(np.float64)
        # Per-frame noise beyond the CCD equation, measured from the comps
        # (already inside rel_err; kept for the magnitude exports and plots).
        self.frame_noise = (np.asarray(dr.frame_noise, dtype=np.float64)
                            if dr.frame_noise is not None else None)
        k = self.ap_index
        raw = cube.flux[k][idx].astype(np.float32).copy()
        rawe = cube.err[k][idx].astype(np.float32).copy()
        # Instrumental flux for the AAVSO magnitude export: blank every
        # measurement the photometry itself rejected, so a saturated comp or a
        # clouded frame can't reach the zero point.
        bad = (flags & (xphot.FLAG_EDGE | xphot.FLAG_SAT | xphot.FLAG_BADFRAME |
                        xphot.FLAG_SKY | xphot.FLAG_INVALID | xphot.FLAG_OUTLIER)) != 0
        raw[bad] = np.nan
        rawe[bad] = np.nan
        self.raw_flux = raw
        self.raw_flux_err = rawe
        self.star_positions = [(float(x), float(y)) for x, y in np.asarray(ref_xy)[idx]]
        comps = list(self.comp_stars)
        self.ensemble_map = {i: [c for c in comps if c != i] for i in range(idx.size)}
        self.transit_fits = {}
        self.star_names = {}            # new star numbering
        self._list_tags = {}
        self.screen = {}

    def _resolve_airmass(self, radec=None):
        """Per-frame airmass: header AIRMASS, else altitude keys, else computed."""
        hdrs = getattr(self, "_cached_headers", []) or []
        n = len(hdrs)
        X = np.full(n, np.nan)
        for i, hh in enumerate(hdrs):
            a = xfit.header_float(hh, ("AIRMASS", "AIRMAS", "SECZ"))
            if a is not None and 1.0 <= a < 40.0:
                X[i] = a
                continue
            alt = xfit.header_float(hh, ("OBJCTALT", "ALTITUDE", "TELALT", "CENTALT", "ALT"))
            if alt is not None and 0.0 < alt <= 90.0:
                X[i] = float(xfit.kasten_young_airmass(alt))
        self._site = next((s for s in (xfit.observatory_site(hh) for hh in hdrs) if s), None)
        jd = self._frame_jd_utc
        if (np.any(~np.isfinite(X)) and self._site is not None and radec is not None
                and jd is not None and len(jd) == n and np.all(np.isfinite(jd))):
            Xc = xfit.airmass_from_time(jd, radec[0], radec[1], self._site[0], self._site[1])
            X = np.where(np.isfinite(X), X, Xc)
        X = np.where(np.isfinite(X) & (X >= 1.0), X, 1.0)
        self.airmasses = [float(v) for v in X]
        return X

    def _refresh_detrend_and_screen(self):
        rel = getattr(self, "rel_flux_raw", None)
        if rel is None:
            return
        n_s, n_f = rel.shape
        t = self._frame_time_days()
        X = np.asarray(self.airmasses, dtype=np.float64) if len(self.airmasses) == n_f else None
        mode = getattr(self, "detrend_mode", "airmass") or "none"
        S = (np.asarray(self.session_idx) if self.session_idx is not None
             and len(self.session_idx) == n_f else None)
        P = self._side_array(n_f)          # pier side: a level step per side
        flux = rel.copy()
        err = self.rel_err_raw.copy()
        self.screen = {}
        self.screen_by_session = {}
        # A transit happens within one night, so with several nights the dip
        # search runs on each night alone -- its own baseline, its own red
        # noise -- and the all-nights entry keeps the best night's dip.  One
        # search across every night let the noisiest night's red noise set
        # the bar for all of them: NGC660 #20 scored 21.8 ppt at 6.6σ in a
        # diluted 2 h box that way, 40.3 ppt at 17.8σ on its own night, so the
        # All nights and single-night views showed different boxes.
        multi = S is not None and np.unique(S).size > 1
        for i in range(n_s):
            ok = (self.flags[i] == 0) & np.isfinite(flux[i])
            # Fitted jointly over every night (shared airmass term, per-night
            # time terms, per-night levels as nuisance); only the within-night
            # shape is divided out, so night-to-night levels survive.
            tr = xphot.robust_trend(t, flux[i], ok, mode, X, S, P)
            flux[i] = flux[i] / tr
            err[i] = err[i] / tr
            if np.any(ok):
                m = float(np.median(flux[i][ok]))
                if np.isfinite(m) and m > 0:
                    flux[i] /= m
                    err[i] /= m
            self.screen[i] = xphot.screen_star(t, rel[i], self.rel_err_raw[i], ok,
                                               baseline=mode, airmass=X, session=S,
                                               side=P, search_dip=not multi)
            if i % 50 == 0:
                QApplication.processEvents()
        if multi:
            keys = ("dip_depth_ppt", "dip_snr", "dip_t", "dip_dur_h", "dip_n")
            for s in np.unique(S):
                for i, mn in self._session_screen(int(s)).items():
                    ma = self.screen.get(i)
                    if ma is not None and mn.get("dip_snr", 0.0) > ma.get("dip_snr", 0.0):
                        ma.update({k: mn.get(k) for k in keys})
                        ma["dip_night"] = int(s)
        self.fluxes = flux.astype(np.float32)
        self.flux_errors = err.astype(np.float32)

    def _star_item_text(self, i: int, tags=None) -> str:
        x, y = self.star_positions[i]
        scr = self._active_screen()
        m = scr.get(i, {}) if isinstance(scr, dict) else {}
        rms = m.get("rms_ppt", np.nan)
        name = self.star_names.get(i)
        parts = [f"#{i}" + (f" {name}" if name else "") + f": x={x:.1f}, y={y:.1f}",
                 f"σ={rms:.1f} ppt" if np.isfinite(rms) else "σ=na"]
        if i in getattr(self, "comp_stars", []):
            parts.append("[COMP]")
        if self.flags is not None and i < self.flags.shape[0]:
            fm = self._active_frame_mask()
            fl_i = self.flags[i][fm] if fm.shape[0] == self.flags.shape[1] else self.flags[i]
            sat = float(np.mean((fl_i & xphot.FLAG_SAT) != 0)) if fl_i.size else 0.0
            if sat > 0:
                parts.append(f"[SAT {sat * 100:.0f}%]")
        for tg in (tags or []):
            parts.append(f"[{tg}]")
        return "   ".join(parts)

    def _populate_star_list(self):
        self.star_list.blockSignals(True)
        self.star_list.clear()
        for i in range(len(self.star_positions)):
            item = QListWidgetItem(self._star_item_text(i))
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.star_list.addItem(item)
        self.star_list.blockSignals(False)

    def _star_label(self, idx: int) -> str:
        """Plot / dialog name for a star: its identified name once Identify
        or Find Star has named it, keeping the list number for reference."""
        name = self.star_names.get(int(idx))
        return f"{name} (#{idx})" if name else f"Star #{idx}"

    def _set_star_name(self, idx: int, name: str):
        """Remember a star's name and show it in the list and the plots."""
        name = str(name or "").strip()
        if not name or self.star_names.get(int(idx)) == name:
            return
        self.star_names[int(idx)] = name
        for row in range(self.star_list.count()):
            it = self.star_list.item(row)
            if it.data(Qt.ItemDataRole.UserRole) == idx:
                it.setText(self._star_item_text(
                    idx, (getattr(self, "_list_tags", None) or {}).get(idx)))
                break
        if idx in [it.data(Qt.ItemDataRole.UserRole) for it in self.star_list.selectedItems()]:
            self.update_plot_for_selection()

    def _set_graph_title(self, title: str, detail: str = ""):
        """
        Title of the interactive plot.  pyqtgraph gives the title one 30 px
        row and makes its text width the plot's minimum width, so a long
        title pushed the plot past the right edge of a narrow window.  The
        detail goes on a smaller second line, and the title no longer sets
        a minimum width (a line too long for the plot is clipped instead).
        """
        from html import escape
        pi = self.plot_widget.getPlotItem()
        lab = pi.titleLabel
        if not getattr(lab, "_exo_no_min_width", False):
            orig = lab.updateMin

            def _update_min():
                orig()
                lab.setMinimumWidth(0)
            lab.updateMin = _update_min
            lab._exo_no_min_width = True
        text = escape(title)
        if detail:
            text += f"<br><span style='font-size:9pt'>{escape(detail)}</span>"
        pi.setTitle(text)
        h = 48 if detail else 30
        pi.layout.setRowFixedHeight(0, h)
        lab.setMaximumHeight(h)
        lab.setMinimumWidth(0)

    def _pix_to_radec(self, x, y):
        if self._wcs is None:
            return None
        bf = getattr(self, "_wcs_bin_factor", 1)
        sky = self._wcs.pixel_to_world(float(x) * bf, float(y) * bf)
        return float(sky.ra.deg), float(sky.dec.deg)

    def _remember_transit_fit(self, idx, out):
        self.transit_fits[int(idx)] = out

    def on_fit_transit(self):
        sels = self.star_list.selectedItems()
        if len(sels) != 1:
            QMessageBox.information(self, "Fit Transit", "Select exactly one star (the transit host).")
            return
        idx = sels[0].data(Qt.ItemDataRole.UserRole)
        ctx = self._transit_context(idx)
        if ctx is None:
            return
        from setiastro.saspro.exoplanet_fit_dialog import TransitFitDialog
        dlg = TransitFitDialog(self, ctx)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.setWindowFlag(Qt.WindowType.Window, True)
        dlg.show()

    def _transit_context(self, idx: int):
        cube = self.phot_cube
        if cube is None:
            QMessageBox.warning(self, "Fit Transit", "Run photometry first.")
            return None
        jd = self._frame_jd_utc
        if jd is None or len(jd) != cube.flux.shape[2] or not np.all(np.isfinite(jd)):
            QMessageBox.warning(self, "Fit Transit",
                                "Some frames have no usable timestamp (DATE-OBS / MJD-OBS), "
                                "so a transit time can't be measured.")
            return None
        radec = None
        try:
            x, y = self.star_positions[idx]
            radec = self._pix_to_radec(x, y)
        except Exception:
            radec = None
        X = self._resolve_airmass(radec if radec else
                                  ((self.wcs_ra, self.wcs_dec) if self.wcs_ra is not None else None))
        n_ap = cube.flux.shape[0]
        bad_base = (cube.flags & (xphot.FLAG_EDGE | xphot.FLAG_SAT | xphot.FLAG_SKY |
                                  xphot.FLAG_BADFRAME)) != 0
        bad_base |= (self.flags & xphot.FLAG_BADFRAME) != 0
        valid = np.stack([~bad_base & ~cube.nonpos[k] for k in range(n_ap)])
        eligible = ~np.any((cube.flags & xphot.FLAG_SAT) != 0, axis=1)
        exclude = sorted(set(int(i) for i in getattr(self, "flagged_stars", set())) - {int(idx)})
        hdr = self._cached_headers[self.ref_idx] if self._cached_headers else None
        xb = xfit.header_float(hdr, ("XBINNING", "XBIN"))
        yb = xfit.header_float(hdr, ("YBINNING", "YBIN"))
        binning = f"{int(xb)}x{int(yb)}" if xb and yb else "1x1"
        inst = (xfit.header_str(hdr, ("INSTRUME", "CAMERA")) or "").lower()
        cam = "CMOS" if any(k in inst for k in ("asi", "zwo", "qhy", "cmos", "imx", "poa",
                                                    "touptek", "moravian c", "atik horizon",
                                                    "player one")) else "CCD"
        exp = self._frame_exptime
        return {
            "target": int(idx), "label": self._star_label(idx),
            "ra": radec[0] if radec else None, "dec": radec[1] if radec else None,
            "jd_utc": np.asarray(jd, dtype=np.float64),
            "exptime_s": float(np.nanmedian(exp)) if exp is not None and np.any(np.isfinite(exp))
                         else float(self.exposure_time or 0.0),
            "time_how": self._time_how, "site": getattr(self, "_site", None),
            "airmass": np.asarray(X, dtype=np.float64),
            "filter_kw": xfit.header_str(hdr, ("FILTER", "FILTER1", "FILTNAME", "FILT")),
            "object_name": xfit.header_str(hdr, ("OBJECT", "TARGNAME", "TARGET")),
            "binning": binning, "camera_hint": cam,
            "soft_bin": int(getattr(self, "bin_factor", 2)),
            "side_idx": (np.asarray(self.side_idx) if getattr(self, "side_idx", None) is not None
                         and np.any(self.side_idx) else None),
            "flux": cube.flux, "err": cube.err, "valid": valid, "radii": cube.radii,
            "best_ap": self.ap_index, "eligible": eligible, "exclude": exclude,
            "max_comps": int(self.ensemble_k),
            "star_xy": np.asarray(self.star_positions, dtype=np.float64),
            "wcs_to_radec": self._pix_to_radec,
            # dip_t in the screening metrics is days since the FIRST frame of
            # the whole set, for a single night as well as for all nights.
            "screen": self._active_screen().get(idx),
            "rel": self.rel_flux_raw[idx], "rel_ok": self.flags[idx] == 0,
            "rel_err": self.rel_err_raw[idx],
            "session_idx": (np.asarray(self.session_idx) if self.session_idx is not None
                            else np.zeros(cube.flux.shape[2], dtype=int)),
            "sessions": list(self.sessions),
            "active_session": self._active_session(),
        }

    def _solve_reference(self, plane, hdr):
        plane2d = plane.mean(axis=2) if getattr(plane, "ndim", 2) == 3 else plane

        # Images were 2x2 binned during load — track this so pixel→sky
        # conversions apply the correct scale factor.
        self._wcs_bin_factor = int(getattr(self, "bin_factor", 2))  # always set; plate solver works on binned plane
                                   # so solved WCS is already in binned coords.
                                   # Header WCS from original file is NOT — see below.

        def _is_xisf_source(h) -> bool:
            if h is None:
                return False
            # XISF loaded via legacy loader stores format in metadata dict
            if isinstance(h, dict):
                fmt = str(h.get("original_format", "")).lower()
                if fmt == "xisf":
                    return True
                # Any key starting with XISF: is a tell-tale from xisf_fits_header_from_meta
                for k in h.keys():
                    if isinstance(k, str) and k.upper().startswith("XISF:"):
                        return True
                # Check nested fits_header / original_header
                for sub_key in ("fits_header", "original_header", "header"):
                    sub = h.get(sub_key)
                    if isinstance(sub, fits.Header):
                        for k in sub.keys():
                            if isinstance(k, str) and k.upper().startswith("XISF:"):
                                return True
            if isinstance(h, fits.Header):
                for k in h.keys():
                    if isinstance(k, str) and k.upper().startswith("XISF:"):
                        return True
            return False

        source_is_xisf = _is_xisf_source(hdr)

        # Also check the file extension of the first loaded path as a fallback
        if not source_is_xisf and self.image_paths:
            ref_path = self.image_paths[self.ref_idx] if hasattr(self, "ref_idx") else self.image_paths[0]
            if str(ref_path).lower().endswith(".xisf"):
                source_is_xisf = True

        # ── Step 1: trust existing WCS only if NOT from XISF and not forcing ─
        force_solve = self.force_solve_cb.isChecked()
        approx_wcs = None

        if not source_is_xisf and not force_solve:
            existing_wcs = self._try_wcs_from_header(hdr, plane2d)
            if existing_wcs is not None and self._header_wcs_is_approximate(hdr):
                # Synthesised from the mount's pointing (SGP / SASpro write one
                # when nothing was solved): arcminutes off -- measured ~190"
                # on NGC660 -- which mis-identifies every star in VSX/SIMBAD
                # and pulls the wrong catalogue magnitudes.  Solve instead.
                print("[Exoplanet] header WCS is synthesised from the mount pointing; plate-solving")
                approx_wcs, existing_wcs = existing_wcs, None
            if existing_wcs is not None:
                # Ask the user whether to reuse the existing WCS or re-solve
                ans = QMessageBox.question(
                    self,
                    "WCS Found",
                    "A valid WCS solution was found in the image header.\n\n"
                    "Use the existing WCS?\n\n"
                    "• Yes — use the header WCS as-is (fast)\n"
                    "• No  — run the plate solver to get a fresh solution",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if ans == QMessageBox.StandardButton.Yes:
                    self._wcs = existing_wcs
                    self._wcs_bin_factor = int(getattr(self, "bin_factor", 2))  # header WCS is full-res; star coords are binned
                    H, W = plane2d.shape[:2]
                    try:
                        # This WCS is in FULL-RES pixels while plane2d is
                        # binned, so the frame centre sits at
                        # (W, H) * bin_factor / 2 in its coordinates. At
                        # bin_factor 2 that is exactly (W, H) -- which is
                        # what the old line passed, correct by coincidence
                        # rather than by construction. Spelled out so it
                        # survives the factor ever changing.
                        _bf = float(getattr(self, "_wcs_bin_factor", 2) or 2)
                        center = self._wcs.pixel_to_world(W * _bf / 2.0,
                                                          H * _bf / 2.0)
                        self.wcs_ra  = float(center.ra.deg)
                        self.wcs_dec = float(center.dec.deg)
                    except Exception:
                        self.wcs_ra = self.wcs_dec = None
 
                    ra_str  = "nan" if self.wcs_ra  is None else f"{self.wcs_ra:.5f}"
                    dec_str = "nan" if self.wcs_dec is None else f"{self.wcs_dec:.5f}"
                    self.status_label.setText(f"WCS from header: RA={ra_str}, Dec={dec_str}")
                    self.fetch_tesscut_btn.setEnabled(True)
                    return
                # else: fall through to plate solve below
        else:
            if force_solve:
                self.status_label.setText("Force blind solve enabled — ignoring any existing WCS…")
            else:
                self.status_label.setText(
                    "XISF source detected — skipping approximate WCS, running plate solve…")
            QApplication.processEvents()



        # ── Step 2: plate-solve ───────────────────────────────────────────
        #
        # plane2d is 2x2 binned; the header is not. plate_solver takes its
        # pixel-scale hint straight out of this header, so handing it an
        # unbinned one tells it the scale is half what it is.
        #
        # The blind path does not merely degrade on that: solve_blind_core
        # discards any candidate outside +/-30% of the hint, and the truth
        # is 2.0x it, so every candidate is thrown away and the solve
        # reports "no match" on a field it had already located. The same
        # number also sets the |AB| baseline window and picks the index
        # layer, so it is wrong three ways before the filter even runs.
        _BIN = int(getattr(self, "bin_factor", 2))
        seed_hdr = _rescale_header_for_binning(
            self._coerce_seed_header(hdr if hdr is not None else {}, plane2d),
            _BIN)
        meta = {}
        if isinstance(hdr, fits.Header):
            meta["original_header"] = _rescale_header_for_binning(hdr, _BIN)
        elif isinstance(hdr, dict):
            meta.update(hdr)
            # _seed_header_from_meta lets meta["wcs"] / meta["wcs_header"]
            # OVERRIDE original_header. Both describe the unbinned grid, so
            # leaving them in means the corrected header is quietly
            # overwritten by the very WCS it was correcting.
            for _k in ("wcs", "wcs_header", "fits_header"):
                meta.pop(_k, None)
            H = self._get_best_header_for_wcs(hdr)
            meta["original_header"] = (_rescale_header_for_binning(H, _BIN)
                                       if H is not None else seed_hdr)
        else:
            meta["original_header"] = seed_hdr

        # For XISF, the seed header still has useful RA/Dec/scale hints
        # even though we don't trust its distortion model — keep them.
        if source_is_xisf and "original_header" in meta:
            existing_h = meta["original_header"]
            if isinstance(existing_h, fits.Header):
                # Strip SIP and CD/CDELT so the solver starts clean,
                # but keep CRVAL*/OBJCTRA/OBJCTDEC/FOCALLEN/XPIXSZ for hints
                strip_keys = [k for k in existing_h.keys()
                              if k.startswith(("A_", "B_", "AP_", "BP_",
                                               "CD1_", "CD2_", "CDELT",
                                               "CTYPE", "CRPIX", "NAXIS"))]
                clean_hdr = existing_h.copy()
                for k in strip_keys:
                    try:
                        del clean_hdr[k]
                    except Exception:
                        pass
                meta["original_header"] = clean_hdr

        doc = SimpleNamespace(
            image=plane2d,
            metadata=meta,
        )

        settings = getattr(self, "settings", None)
        if settings is None and hasattr(self, "main_win"):
            settings = getattr(self.main_win, "settings", None)
        if settings is None and self.parent() is not None:
            settings = getattr(self.parent(), "settings", None)

        ok, res = plate_solve_doc_inplace(self, doc, settings)
        if not ok:
            if source_is_xisf:
                # For XISF, offer to fall back to the approximate WCS rather
                # than leaving the user with nothing at all
                ans = QMessageBox.question(
                    self,
                    "Plate Solve Failed",
                    f"Plate solving failed:\n{res}\n\n"
                    "The loaded files are XISF — the embedded WCS is an approximation "
                    "of PixInsight's TPS solution and may be off by 10–30 arcsec in the "
                    "frame corners.\n\n"
                    "Use the approximate XISF WCS anyway? Star matching may fail for "
                    "some targets near the edges.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if ans == QMessageBox.StandardButton.Yes:
                    fallback_wcs = self._try_wcs_from_header(hdr, plane2d)
                    if fallback_wcs is not None:
                        self._wcs = fallback_wcs
                        H2, W2 = plane2d.shape[:2]
                        try:
                            # Same correction as the reuse path above, and
                            # here it was genuinely wrong: this WCS is
                            # full-res, so (W2/2, H2/2) is a QUARTER of the
                            # way into the frame, not the centre.
                            _bf2 = float(getattr(self, "_wcs_bin_factor", 2) or 2)
                            center = self._wcs.pixel_to_world(W2 * _bf2 / 2.0,
                                                              H2 * _bf2 / 2.0)
                            self.wcs_ra  = float(center.ra.deg)
                            self.wcs_dec = float(center.dec.deg)
                        except Exception:
                            self.wcs_ra = self.wcs_dec = None
                        self.status_label.setText(
                            "⚠️ Using approximate XISF WCS (star matching may be imprecise)"
                        )
                        self.fetch_tesscut_btn.setEnabled(True)
                        return
            if approx_wcs is not None:
                # Good to arcminutes: enough for BJD and airmass, not for
                # naming stars or pulling catalogue magnitudes.
                self._wcs = approx_wcs
                self._wcs_bin_factor = int(getattr(self, "bin_factor", 2))
                H2, W2 = plane2d.shape[:2]
                try:
                    _bf2 = float(self._wcs_bin_factor)
                    center = self._wcs.pixel_to_world(W2 * _bf2 / 2.0, H2 * _bf2 / 2.0)
                    self.wcs_ra, self.wcs_dec = float(center.ra.deg), float(center.dec.deg)
                except Exception:
                    self.wcs_ra = self.wcs_dec = None
                self.status_label.setText(
                    "⚠️ Plate solve failed — using the mount-pointing WCS from the header "
                    "(arcminutes off: fine for BJD/airmass, unreliable for star identification "
                    "and catalogue magnitudes)")
                print(f"[Exoplanet] plate solve failed ({res}); using the approximate header WCS")
                self.fetch_tesscut_btn.setEnabled(True)
                return
            QMessageBox.critical(self, "Plate Solve", f"Plate solving failed:\n{res}")
            self._wcs = None
            self.fetch_tesscut_btn.setEnabled(False)
            return

        # ── Step 3: grab solved WCS ───────────────────────────────────────
        self._wcs = doc.metadata.get("wcs", None)
        self._wcs_bin_factor = 1  # solver worked on binned plane → WCS is in binned coords        
        if self._wcs is None or not getattr(self._wcs, "has_celestial", False):
            QMessageBox.warning(self, "Plate Solve", "Solver finished but no usable WCS was found.")
            self.fetch_tesscut_btn.setEnabled(False)
            self._wcs = None
            return

        H, W = plane2d.shape[:2]
        try:
            center = self._wcs.pixel_to_world(W / 2, H / 2)
            self.wcs_ra  = float(center.ra.deg)
            self.wcs_dec = float(center.dec.deg)
        except Exception:
            self.wcs_ra = self.wcs_dec = None

        ra_str  = "nan" if self.wcs_ra  is None else f"{self.wcs_ra:.5f}"
        dec_str = "nan" if self.wcs_dec is None else f"{self.wcs_dec:.5f}"
        self.status_label.setText(f"WCS solved: RA={ra_str}, Dec={dec_str}")
        self.fetch_tesscut_btn.setEnabled(True)

    # ---------------- Plotting & helpers ----------------

    def show_ensemble_members(self):
        sels = self.star_list.selectedItems()
        if len(sels) != 1: return
        target = sels[0].data(Qt.ItemDataRole.UserRole)
        members = self.ensemble_map.get(target, [])
        for idx in self._last_ensemble:
            item = self.star_list.item(idx)
            if item:
                color = item.background().color()
                if color == QColor('lightblue'):
                    item.setBackground(QBrush())
        for idx in members:
            item = self.star_list.item(idx)
            if item and item.background().color() != QColor('yellow'):
                item.setBackground(QBrush(QColor('lightblue')))
        self._last_ensemble = members

    def on_detrend_changed(self, idx: int):
        # idx: 0 None, 1 Linear, 2 Quadratic, 3 Airmass (falls back to linear)
        self.detrend_degree = {0: None, 1: 1, 2: 2, 3: None}.get(idx)
        self.detrend_mode = {0: "none", 1: "linear", 2: "quadratic", 3: "airmass"}.get(idx, "linear")
        if getattr(self, "rel_flux_raw", None) is not None:
            self._refresh_detrend_and_screen()
            self._on_threshold_changed(self.threshold_slider.value())
            self.update_plot_for_selection()

    @staticmethod
    def _detrend_curve(curve: np.ndarray, deg: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.arange(curve.size)
        if mask is None:
            mask = np.isfinite(curve) & (curve > 0)
        n_good = int(mask.sum())
        if n_good < 2:
            return curve
        fit_deg = min(deg, n_good - 1)
        if fit_deg < 1:
            return curve
        try:
            coeffs = np.polyfit(x[mask], curve[mask], fit_deg)
        except Exception:
            return curve
        trend = np.polyval(coeffs, x)
        trend[trend == 0] = 1.0
        return curve / trend

    def _show_reference_with_circles(self, plane, positions):
        old = getattr(self, "_ref_overlay", None)
        if old is not None:
            try:
                old.close()
            except RuntimeError:
                pass
        dlg = ReferenceOverlayDialog(
            plane=plane,
            positions=positions,
            target_median=self.median_fwhm,
            parent=self
        )
        self._ref_overlay = dlg
        dlg.show()

    # ---- plot style: Graphical (pyqtgraph) / Scientific (matplotlib) ---------

    def _plot_settings(self):
        s = getattr(self, "settings", None)
        return s if s is not None else QSettings()

    def _plot_style(self) -> str:
        cb = getattr(self, "plot_style_combo", None)
        return str(cb.currentData()) if cb is not None else "graphical"

    def _bin_minutes(self) -> int:
        cb = getattr(self, "plot_bin_combo", None)
        try:
            return int(cb.currentData()) if cb is not None else 10
        except (TypeError, ValueError):
            return 10

    def _sync_plot_style_widgets(self):
        sci = self._plot_style() == "scientific"
        self.plot_units_label.setVisible(sci)
        self.plot_units_combo.setVisible(sci)
        if sci:
            w = self._ensure_sci_plot()
            if w is None:          # matplotlib Qt backend unavailable
                self.plot_style_combo.blockSignals(True)
                self.plot_style_combo.setCurrentIndex(0)
                self.plot_style_combo.blockSignals(False)
                self.plot_units_label.setVisible(False)
                self.plot_units_combo.setVisible(False)
                self.plot_stack.setCurrentWidget(self.plot_widget)
                return
            self.plot_stack.setCurrentWidget(w)
        else:
            self.plot_stack.setCurrentWidget(self.plot_widget)

    def _on_plot_options_changed(self, _i: int = 0):
        ps = self._plot_settings()
        try:
            ps.setValue("Exoplanet/plot_style", self._plot_style())
            ps.setValue("Exoplanet/plot_units", str(self.plot_units_combo.currentData()))
            ps.setValue("Exoplanet/plot_bin_min", self._bin_minutes())
        except Exception:
            pass
        self._sync_plot_style_widgets()
        self.update_plot_for_selection()

    def _ensure_sci_plot(self):
        if self.sci_plot is None:
            try:
                from setiastro.saspro.exoplanet_sciplot import SciPlotWidget
                self.sci_plot = SciPlotWidget(self)
            except Exception as e:
                print(f"[Exoplanet] scientific plot unavailable: {e}")
                return None
            self.plot_stack.addWidget(self.sci_plot)
        return self.sci_plot

    def _frame_noise_mag(self, n_frames: int) -> np.ndarray:
        """Comp-measured per-frame excess noise in magnitudes (0 where unknown)."""
        s = getattr(self, "frame_noise", None)
        if s is None or len(s) != n_frames:
            return np.zeros(n_frames)
        return (2.5 / np.log(10.0)) * np.nan_to_num(np.asarray(s, dtype=np.float64))

    def _sci_time_axis(self):
        """
        Times for the Scientific plot: BJD_TDB at the field centre when the
        field is solved, else JD_UTC (mid-exposure), minus a whole day.
        Returns (x_days | None, xlabel, kind).
        """
        n = 0 if self.fluxes is None else self.fluxes.shape[1]
        jd = self._frame_jd_utc
        if jd is None or len(jd) != n or n == 0 or not np.all(np.isfinite(jd)):
            return None, "Frame number", "frame"
        ra, dec = getattr(self, "wcs_ra", None), getattr(self, "wcs_dec", None)
        key = (n, float(jd[0]), float(jd[-1]), ra, dec)
        if self._sci_time_cache is not None and self._sci_time_cache[0] == key:
            return self._sci_time_cache[1]
        tt, kind = np.asarray(jd, dtype=np.float64), "JD_UTC"
        if ra is not None and dec is not None:
            try:
                tt = np.asarray(xfit.bjd_tdb(jd, float(ra), float(dec),
                                             getattr(self, "_site", None)), dtype=np.float64)
                kind = "BJD_TDB"
            except Exception:
                tt, kind = np.asarray(jd, dtype=np.float64), "JD_UTC"
        off = float(np.floor(np.nanmin(tt)))
        label = (r"BJD$_\mathrm{TDB}$" if kind == "BJD_TDB" else r"JD$_\mathrm{UTC}$") \
            + f" − {off:.0f}  (mid-exposure)"
        out = (tt - off, label, kind)
        self._sci_time_cache = (key, out)
        return out

    def _star_radec_text(self, idx: int) -> str:
        try:
            x, y = self.star_positions[idx]
            rd = self._pix_to_radec(x, y)
            if not rd:
                return ""
            c = SkyCoord(ra=float(rd[0]) * u.deg, dec=float(rd[1]) * u.deg)
            return c.to_string("hmsdms", sep=":", precision=1)
        except Exception:
            return ""

    def _draw_scientific(self):
        w = self._ensure_sci_plot()
        if w is None:
            return
        from setiastro.saspro import exoplanet_sciplot as sciplot
        fig = w.figure
        inds = [it.data(Qt.ItemDataRole.UserRole) for it in self.star_list.selectedItems()]
        if self.fluxes is None:
            sciplot.draw_message(fig, "Load frames and run photometry")
            w.refresh()
            return
        if not inds:
            sciplot.draw_message(fig, "Select a star in the list")
            w.refresh()
            return
        n_frames = self.fluxes.shape[1]
        fmask = self._active_frame_mask()
        active = self._active_session()
        S = (np.asarray(self.session_idx) if self.session_idx is not None
             and len(self.session_idx) == n_frames else np.zeros(n_frames, dtype=int))
        tx, xlabel, kind = self._sci_time_axis()
        has_t = tx is not None
        if not has_t:
            tx = np.arange(n_frames, dtype=np.float64)
        bmin = self._bin_minutes()
        bin_w = (bmin / 1440.0) if has_t else (5.0 if bmin else 0.0)
        bin_label = (f"{bmin}-min bins" if has_t else "5-frame bins") if bmin else ""

        panels = None
        if has_t and active is None and 1 < len(self.sessions) <= 6:
            panels = []
            for s in self.sessions:
                m = (S == int(s["index"])) & fmask
                if np.any(m):
                    panels.append((float(tx[m].min()), float(tx[m].max()),
                                   s.get("date") or s["label"]))

        errs = getattr(self, "flux_errors", None)
        series = []
        for idx in inds[:sciplot.MAX_SERIES]:
            f = np.asarray(self.fluxes[idx], dtype=np.float64)
            e = (np.asarray(errs[idx], dtype=np.float64)
                 if errs is not None and idx < errs.shape[0] else None)
            fl = self.flags[idx] if (self.flags is not None and idx < self.flags.shape[0]) \
                else np.zeros(n_frames, dtype=np.int16)
            good = np.isfinite(f) & (fl == 0) & fmask
            rej = np.isfinite(f) & (fl != 0) & fmask
            tag = " (comp)" if idx in getattr(self, "comp_stars", []) else ""
            series.append({"label": (self.star_names.get(idx) or f"#{idx}") + tag,
                           "x": tx[good], "y": f[good],
                           "err": (e[good] if e is not None else None), "group": S[good],
                           "rej_x": tx[rej], "rej_y": f[rej]})

        spans = []
        title = ""
        if len(inds) == 1:
            idx = inds[0]
            scr = self._active_screen()
            m = scr.get(idx, {}) if isinstance(scr, dict) else {}
            parts = [self._star_label(idx)]
            pos = self._star_radec_text(idx)
            if pos:
                parts.append(pos)
            if active is not None:
                parts.append(self.sessions[active].get("date") or self.sessions[active]["label"])
            if m and np.isfinite(m.get("rms_ppt", np.nan)):
                parts.append(f"σ = {m['rms_ppt']:.2f} ppt")
            vt = (getattr(self, "dip_vetoes", None) or {}).get(idx)
            if vt is not None:
                parts.append(f"dip withdrawn: {vt[1]:.0f} ppt at r = {vt[3]:.1f} px, "
                             f"{vt[2]:.0f} ppt at r = {vt[4]:.1f} px (PSF change)")
            title = "  ·  ".join(parts)
            if (has_t and m and self._has_dip_flag(idx)
                    and np.isfinite(m.get("dip_t", np.nan)) and m.get("dip_snr", 0) > 0):
                td = self._frame_time_days()
                o = np.argsort(td)
                c = float(np.interp(m["dip_t"], td[o], tx[o]))
                hw = 0.5 * float(m["dip_dur_h"]) / 24.0
                kk = S[int(np.nanargmin(np.abs(td - m["dip_t"])))]
                xs = tx[(S == kk) & fmask]
                lo, hi = c - hw, c + hw
                if xs.size:
                    lo, hi = max(lo, float(xs.min())), min(hi, float(xs.max()))
                if hi > lo:
                    spans.append((lo, hi, f"candidate dip ({m['dip_depth_ppt']:.1f} ppt, "
                                          f"{m['dip_snr']:.1f}σ)"))
        elif len(inds) > sciplot.MAX_SERIES:
            title = f"First {sciplot.MAX_SERIES} of {len(inds)} selected stars"

        sciplot.draw_light_curves(fig, series, panels=panels,
                                  units=str(self.plot_units_combo.currentData()),
                                  bin_w=bin_w, bin_label=bin_label, xlabel=xlabel,
                                  title=title, spans=spans)
        w.refresh()

    def save_plot(self):
        """Save the plot as shown: Scientific as 300 dpi PNG / PDF / SVG."""
        sci = self._plot_style() == "scientific" and self.sci_plot is not None
        if sci:
            path, filt = QFileDialog.getSaveFileName(
                self, "Save Plot", "light_curve.png",
                "PNG image, 300 dpi (*.png);;PDF (*.pdf);;SVG (*.svg)")
            if not path:
                return
            if not os.path.splitext(path)[1]:
                path += ".pdf" if "PDF" in filt else ".svg" if "SVG" in filt else ".png"
            try:
                self.sci_plot.save(path)
            except Exception as e:
                QMessageBox.warning(self, "Save Plot", f"Could not save the plot:\n{e}")
                return
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "light_curve.png",
                                                  "PNG (*.png)")
            if not path:
                return
            if not path.lower().endswith(".png"):
                path += ".png"
            self.plot_widget.grab().save(path)
        self.status_label.setText(f"Saved plot → {path}")

    def update_plot_for_selection(self):
        """Redraw light curves for the selected stars (selected night, or all)."""
        if self._plot_style() == "scientific":
            self._draw_scientific()
            return
        if self.fluxes is None:
            return
        n_stars, n_frames = self.fluxes.shape
        has_t = self.times is not None
        x_all, dividers, to_x = self._session_axis()
        fmask = self._active_frame_mask()
        active = self._active_session()
        S = (np.asarray(self.session_idx) if self.session_idx is not None
             and len(self.session_idx) == n_frames else np.zeros(n_frames, dtype=int))
        if not has_t:
            bottom_label = "Frame"
        elif active is not None:
            bottom_label = ("Hours since the night's first frame (mid-exposure UTC) — "
                            f"{self.sessions[active]['label']}")
        elif len(self.sessions) > 1:
            bottom_label = "Hours within each night (daytime gaps collapsed; dashed line = next night)"
        else:
            bottom_label = "Hours since first frame (mid-exposure UTC)"

        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('bottom', bottom_label)
        self.plot_widget.setLabel('left', 'Relative Flux')
        self._set_graph_title("Light Curves")

        inds = [it.data(Qt.ItemDataRole.UserRole) for it in self.star_list.selectedItems()]
        if not inds:
            return
        single = len(inds) == 1
        bmin = self._bin_minutes()
        bin_w = (bmin / 60.0) if has_t else (5.0 if bmin else 0.0)

        for idx in inds:
            f = self.fluxes[idx]
            fl = self.flags[idx] if (self.flags is not None and idx < self.flags.shape[0]) \
                else np.zeros_like(f, dtype=np.int16)
            good = np.isfinite(f) & (fl == 0) & fmask & np.isfinite(x_all)
            color = pg.intColor(idx, hues=max(n_stars, 9))
            pt = QColor(color)
            pt.setAlpha(110 if single else 70)
            self.plot_widget.plot(x_all[good], f[good], pen=None, symbol='o', symbolSize=4,
                                  symbolBrush=pg.mkBrush(pt), symbolPen=None,
                                  name=self._star_label(idx))
            # Bins per night, and one line per night: a line drawn across the
            # daytime gap would invent a light curve where there is no data.
            first_bins = True
            fe = getattr(self, "flux_errors", None)
            for k in (np.unique(S[good]) if bin_w > 0 else ()):
                g = good & (S == k)
                tb, yb, eb = xphot.bin_curve(x_all[g], f[g], bin_w,
                                             e=(fe[idx][g] if fe is not None else None))
                if not tb.size:
                    continue
                if single:
                    self.plot_widget.addItem(pg.ErrorBarItem(x=tb, y=yb, height=2 * eb,
                                                             pen=pg.mkPen(color, width=1)))
                self.plot_widget.plot(tb, yb, pen=pg.mkPen(color, width=2),
                                      symbol='o', symbolSize=6, symbolBrush=pg.mkBrush(color),
                                      name=((f"{bmin}-min bins" if has_t else "5-frame bins")
                                            if first_bins else None))
                first_bins = False
            if not single:
                continue
            rejected = np.isfinite(f) & (fl != 0) & fmask & np.isfinite(x_all)
            if np.any(rejected):
                self.plot_widget.plot(x_all[rejected], f[rejected], pen=None, symbol='x',
                                      symbolSize=7, symbolPen=pg.mkPen((255, 80, 80)),
                                      name="rejected")
            scr = self._active_screen()
            m = scr.get(idx, {}) if isinstance(scr, dict) else {}
            title = self._star_label(idx)
            detail = ""
            if m:
                title += f" — σ {m.get('rms_ppt', np.nan):.2f} ppt"
                if np.isfinite(m.get("dip_t", np.nan)) and m.get("dip_snr", 0) > 0:
                    detail = (f"best dip {m['dip_depth_ppt']:.1f} ppt at "
                              f"{m['dip_snr']:.1f}σ ({m['dip_dur_h']:.2f} h)")
                    if idx in (getattr(self, "dip_vetoes", None) or {}):
                        detail += " — withdrawn: fades in a bigger aperture (PSF change)"
                    if self._has_dip_flag(idx) and has_t:
                        c = to_x(m["dip_t"])
                        d = m["dip_dur_h"]
                        if np.isfinite(c):
                            # keep the box inside the night it was found in
                            td = self._frame_time_days()
                            kk = S[int(np.nanargmin(np.abs(td - m["dip_t"])))]
                            xs = x_all[(S == kk) & np.isfinite(x_all)]
                            lo, hi = c - d / 2, c + d / 2
                            if xs.size:
                                lo, hi = max(lo, float(xs.min())), min(hi, float(xs.max()))
                            reg = pg.LinearRegionItem(values=(lo, hi),
                                                      movable=False,
                                                      brush=pg.mkBrush(255, 210, 80, 40))
                            self.plot_widget.addItem(reg)
            self._set_graph_title(title, detail)

        # Night dividers and date labels (All nights view)
        if has_t and active is None and len(self.sessions) > 1:
            vb = self.plot_widget.getPlotItem().getViewBox()
            ymax = vb.viewRange()[1][1]
            for xd, label in dividers:
                xpos = 0.0 if xd is None else xd
                if xd is not None:
                    self.plot_widget.addLine(x=xd, pen=pg.mkPen((170, 170, 170),
                                                                style=Qt.PenStyle.DashLine))
                txt = pg.TextItem(str(label), color=(200, 200, 200), anchor=(0, 0))
                txt.setPos(xpos + (0.05 if xd is not None else 0.0), ymax)
                self.plot_widget.addItem(txt)

    def apply_threshold(self, ppt_threshold: int, sigma_upper: float = 3.0):
        """
        Flag stars from the screening statistics.

          DIP  best box dip >= threshold (ppt) at >= significance (box SNR
               with red noise at the dip's own duration).
          VAR  binned variability amplitude >= threshold at >= significance
               (von Neumann eta).

        The list, overlay and status follow the selected night.  The set
        other code uses to keep variables out of comparison and check stars
        (self.flagged_stars) is the union with the all-nights flags, so a star
        that varied on another night is still excluded.

        Replaces the moving-average peak test, which fired on any star whose
        5-point MA wandered by the threshold -- i.e. on noisy stars and on
        broken photometry as readily as on real signals.
        """
        if getattr(self, "fluxes", None) is None or not self.screen:
            return
        sig = float(self.temporal_snr_spin.value())

        rel_on = bool(getattr(self, "field_relative", True))
        a_f = float(getattr(self, "field_sigma", 5.0))
        nv_on = self._night_check_setting()
        mags = self._star_mags()
        blends = self._blended_stars()

        def _classify(screen):
            # Night-to-night variability, judged against the comparison stars'
            # own night-to-night scatter (only present for multi-night sets).
            # Opt-in: it is a long-period-variable test, and night-to-night
            # flat residuals trip it too.
            floor = (xphot.night_level_floor(screen, getattr(self, "comp_stars", []))
                     if nv_on else None)
            ids = sorted(screen)
            zs = {}
            if rel_on and len(ids) >= 20:
                # Field-relative: the same statistic for stars of similar
                # brightness sets the bar.  Systematics shared by the whole
                # field (seeing-driven aperture loss, flat error, a flip) lift
                # every star together and no longer flag them all.
                mm = np.array([mags[i] if i < len(mags) else np.nan for i in ids])

                def _z(vals):
                    return dict(zip(ids, xphot.field_outlier_z(np.asarray(vals, dtype=np.float64), mm)))
                zs["dip"] = _z([screen[i].get("dip_snr", np.nan) for i in ids])
                zs["eta"] = _z([screen[i].get("eta_z", np.nan) for i in ids])
                zs["amp"] = _z([screen[i].get("amp_ppt", np.nan) for i in ids])
                if nv_on:
                    zs["nv"] = _z([xphot.night_variability_z(screen[i], floor) for i in ids])

            def _out(key, i):
                z = zs.get(key, {}).get(i, np.nan)
                return (not np.isfinite(z)) or z >= a_f, z

            # A star usable in under half the frames (saturated, off the edge
            # most of the time) has statistics made of leftovers: not judged.
            n_max = max([int(screen[j].get("n", 0)) for j in ids] + [0])
            fl, kd, vt = set(), {}, {}
            for i, m in screen.items():
                if int(m.get("n", 0)) < 0.5 * n_max:
                    continue
                k = []
                ok_d, zd = _out("dip", i)
                if (m.get("dip_depth_ppt", 0.0) >= ppt_threshold and m.get("dip_snr", 0.0) >= sig
                        and ok_d):
                    # A real eclipse has the same depth in every aperture; a
                    # PSF change (post-flip settling, focus) does not.
                    ap = self._dip_aperture_test(i, m)
                    if ap is not None and ap[0]:
                        vt[i] = ap
                    else:
                        k.append(f"DIP {m['dip_depth_ppt']:.1f} ppt {m['dip_snr']:.1f}σ"
                                 + (f", field +{zd:.0f}σ" if np.isfinite(zd) else ""))
                ok_e, _ = _out("eta", i)
                ok_a, _ = _out("amp", i)
                ok_n, _ = _out("nv", i)
                if (m.get("amp_ppt", 0.0) >= ppt_threshold and m.get("eta_z", 0.0) >= sig
                        and ok_e and ok_a):
                    k.append("VAR")
                elif (nv_on and m.get("night_amp_ppt", 0.0) >= ppt_threshold
                      and xphot.night_variability_z(m, floor) >= sig and ok_n):
                    k.append("VAR night-to-night")
                if k:
                    if i in blends:
                        k.append("blend?")
                    fl.add(i)
                    kd[i] = k
            return fl, kd, vt

        flagged, kinds, vetoes = _classify(self._active_screen())
        flagged_all = flagged if self._active_session() is None else _classify(self.screen)[0]
        self.flagged_stars = set(flagged) | set(flagged_all)
        self.flag_kinds = kinds
        self.dip_vetoes = vetoes

        for dlg in self.findChildren(ReferenceOverlayDialog):
            dlg.update_dip_flags(set(flagged))

        self.star_list.blockSignals(True)
        self._list_tags = {}
        for row in range(self.star_list.count()):
            item = self.star_list.item(row)
            i = item.data(Qt.ItemDataRole.UserRole)
            tags = list(kinds.get(i) or [])
            tip = ""
            if i in vetoes:
                _f, d0, d1, r0, r1 = vetoes[i]
                why = "blend/PSF" if i in blends else "PSF"
                tags.append(f"dip ✗ {why} {d0:.0f}→{d1:.0f} ppt")
                tip = (f"Dip withdrawn: {d0:.0f} ppt deep in the r = {r0:.1f} px photometry "
                       f"aperture, {d1:.0f} ppt at r = {r1:.1f} px.\nA real eclipse dims the "
                       "whole star, so its depth is the same in any aperture; this is a PSF "
                       "change\n(focus, post-flip settling, corner coma)"
                       + (" or the brighter neighbour." if i in blends else "."))
            self._list_tags[i] = tags
            item.setToolTip(tip)
            item.setText(self._star_item_text(i, tags))
            item.setBackground(QBrush(QColor('yellow')) if i in flagged else QBrush())
        self.star_list.blockSignals(False)

        n_dip = sum(1 for v in kinds.values() if any(s.startswith("DIP") for s in v))
        n_var = sum(1 for v in kinds.values() if "VAR" in v)
        n_nvar = sum(1 for v in kinds.values() if "VAR night-to-night" in v)
        act = self._active_session()
        where = "" if act is None else f" on {self.sessions[act].get('date') or 'this night'}"
        self.status_label.setText(
            f"{len(flagged)} star(s) flagged{where} (≥ {ppt_threshold} ppt, ≥ {sig:.1f}σ"
            + (f", ≥ {a_f:g}σ above the field" if rel_on else "") + "): "
            f"{n_dip} dip, {n_var} variable" + (f", {n_nvar} night-to-night" if n_nvar else "")
            + (f"; {len(vetoes)} dip(s) withdrawn — depth fades in a bigger aperture "
               "(PSF change, not an eclipse)" if vetoes else "")
            + (f" — {self._photometry_status}" if getattr(self, "_photometry_status", "") else ""))

    def moving_average(self, curve, window=5):
        pad = window//2
        ext = np.pad(curve, pad, mode="edge")
        kernel = np.ones(window)/window
        ma = np.convolve(ext, kernel, mode="valid")
        return ma

    # ---------------- Analysis + Identify ----------------

    # Trial periods in Analyze's full BLS scan.  astropy's own grid steps by
    # min(duration) / baseline^2: ~3,500 periods for one night, but 28
    # million for seven nights spread over 27 days (NGC660) -- about 30 hours
    # of single-threaded C with the GUI frozen and Ctrl+C ignored.
    BLS_MAX_PERIODS = 100_000

    def _bls_scan(self, model, periods_d, dur_d, oversample, what="trial periods"):
        """
        BoxLeastSquares.power over `periods_d`, in chunks on a thread pool.
        astropy's BLS releases the GIL, so the chunks run in parallel and the
        GUI stays responsive; a long scan shows a progress dialog with Cancel.
        Returns a namespace of plain float arrays (period, power, duration,
        transit_time, depth), or None when cancelled.
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        periods_d = np.asarray(periods_d, dtype=np.float64)
        n = int(periods_d.size)
        n_chunks = int(min(400, max(1, n // 250)))
        chunks = np.array_split(periods_d, n_chunks)
        workers = max(1, min(16, (os.cpu_count() or 2) - 1, n_chunks))
        dlg = None
        if n > 2000:
            dlg = QProgressDialog(f"Box least squares: {n:,} {what}…", "Cancel", 0, n_chunks, self)
            dlg.setWindowTitle("Analyze Star")
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setMinimumDuration(300)
            dlg.setValue(0)
        out = [None] * n_chunks
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(model.power, c * u.day, dur_d * u.day,
                              objective="snr", oversample=oversample): j
                    for j, c in enumerate(chunks)}
            pending, n_done = set(futs), 0
            while pending:
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for fu in done:
                    out[futs[fu]] = fu.result()
                    n_done += 1
                if dlg is not None:
                    dlg.setValue(n_done)
                    QApplication.processEvents()
                    if dlg.wasCanceled():
                        return None
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            if dlg is not None:
                dlg.close()

        def _cat(name):
            return np.concatenate([np.asarray(getattr(getattr(r, name), "value", getattr(r, name)),
                                              dtype=np.float64).ravel() for r in out])
        return SimpleNamespace(**{k: _cat(k) for k in
                                  ("period", "power", "duration", "transit_time", "depth")})

    def _fold_transit_model(self, idx, period, t0, duration, depth):
        """
        The transit model for Analyze's phase fold -- the shape EXOTIC and Fit
        Transit fit -- in place of the BLS box.  The box (instant ingress and
        egress, flat floor) is how BLS finds the period; it isn't a shape.
        At the BLS period:

          1. The systematics (shared airmass term, per-night levels and time
             terms, pier-side step) are refitted from the raw ratio with the
             in-transit frames masked, so the baseline can't eat part of the
             dip.  EXOTIC fits them jointly with the transit on one night;
             masking is the multi-night equivalent.  See prepare() for a
             night or pier side with no out-of-transit frames.
          2. Every night is normalised to its out-of-transit median.
          3. A quadratic limb-darkened transit, averaged over the exposure,
             is fitted by least squares: Tc, Rp/R*, a/R*, inc and the level
             free; period fixed; limb darkening at the Fit Transit defaults;
             4-sigma neighbour clip and refit.
          Then 1-3 again with the mask set from the fitted duration.

        Returns a dict for the plot and the summary, or None when there is
        nothing to fit or the fit found no transit.
        """
        rel = getattr(self, "rel_flux_raw", None)
        times = getattr(self, "times", None)
        if rel is None or times is None or not (0 <= int(idx) < rel.shape[0]):
            return None
        P, dur = float(period), float(duration)
        if not (np.isfinite(P) and P > 0 and np.isfinite(dur) and dur > 0 and np.isfinite(t0)):
            return None
        jd = np.asarray(times.utc.jd, dtype=np.float64)
        n_f = rel.shape[1]
        if jd.size != n_f:
            return None
        y_raw = np.asarray(rel[idx], dtype=np.float64)
        e_src = getattr(self, "rel_err_raw", None)
        e_raw = (np.asarray(e_src[idx], dtype=np.float64)
                 if e_src is not None and np.shape(e_src) == np.shape(rel) else np.full(n_f, np.nan))
        ok = ((np.asarray(self.flags[idx]) == 0) & np.isfinite(y_raw) & (y_raw > 0)
              & np.isfinite(jd))
        if np.count_nonzero(ok) < 20:
            return None

        def fold_d(t, tc):          # days from the nearest mid-transit
            return ((np.asarray(t, dtype=np.float64) - tc) / P + 0.5) % 1.0 * P - 0.5 * P

        # reference epoch in the middle of the data
        t_mid = 0.5 * (jd[ok].min() + jd[ok].max())
        tc_ref = float(t0) + round((t_mid - float(t0)) / P) * P

        X = (np.asarray(self.airmasses, dtype=np.float64)
             if len(getattr(self, "airmasses", None) or []) == n_f else None)
        S = getattr(self, "session_idx", None)
        S = np.asarray(S) if S is not None and len(S) == n_f else None
        side = self._side_array(n_f)
        side_b = (np.asarray(side) != 0).astype(np.int64) if side is not None else None
        nights = S if S is not None else np.zeros(n_f, dtype=np.int64)
        mode = getattr(self, "detrend_mode", "airmass") or "none"
        t_rel = jd - jd[ok].min()

        def prepare(in_tr):
            """
            Steps 1-2 for one transit mask: (use, y, e, notes).  A baseline
            level is only ever fitted from out-of-transit frames.  A night
            without any gets left out -- fitting its level to transit frames
            erased a whole 48 ppt eclipse in testing.  A pier-side segment
            without any (the flip inside the transit) takes the night's other
            side and the flip step measured on the nights that have both.
            """
            oot = ok & ~in_tr
            use = ok.copy()
            notes = []
            for k in np.unique(nights[ok]):
                m = ok & (nights == k)
                if np.count_nonzero(oot & m) < 5:
                    use &= ~m
                    notes.append(("night", int(k), int(np.count_nonzero(m))))
            y_adj, e_adj, side_fit = y_raw.copy(), e_raw.copy(), side
            if side is not None and np.any(use):
                seg = xphot.segment_labels(S, side, n_f)
                orphan = np.zeros(n_f, bool)
                for lab in np.unique(seg[use]):
                    k = use & (seg == lab)
                    if np.count_nonzero(oot & k) < 5:
                        orphan |= k
                collapse = [int(k) for k in np.unique(nights[orphan])
                            if np.unique(side_b[use & (nights == k)]).size > 1]
                if np.any(orphan):
                    step = float("nan")
                    if collapse:
                        y0 = y_raw / xphot.robust_trend(t_rel, y_raw, oot & use, mode, X, S, None)
                        ratios = []
                        for k in np.unique(nights[use]):
                            if k in collapse:
                                continue
                            m = oot & use & (nights == k)
                            a = y0[m & (side_b == 0)]
                            b_ = y0[m & (side_b == 1)]
                            if a.size >= 5 and b_.size >= 5:
                                ratios.append(float(np.median(b_) / np.median(a)))
                        if ratios:
                            step = float(np.median(ratios))
                    if np.isfinite(step) and step > 0:
                        side_fit = side_b.copy()
                        for k in collapse:
                            m = nights == k
                            s1 = m & (side_b == 1)
                            y_adj[s1] /= step
                            e_adj[s1] /= step
                            side_fit[m] = 0
                        notes.append(("step", len(collapse), step))
                    else:
                        use &= ~orphan
                        notes.append(("orphan", 0, int(np.count_nonzero(orphan))))
            trend = xphot.robust_trend(t_rel, y_adj, oot & use, mode, X, S, side_fit)
            y = y_adj / trend
            e = e_adj / trend
            for k in np.unique(nights[use]):
                m = use & (nights == k)
                base = m & oot
                if np.count_nonzero(base) < 5:
                    use &= ~m
                    continue
                lev = float(np.median(y[base]))
                if np.isfinite(lev) and lev > 0:
                    y[m] /= lev
                    e[m] /= lev
            use &= np.isfinite(y) & (y > 0)
            return use, y, e, notes

        exp = getattr(self, "_frame_exptime", None)
        exp_s = (float(np.nanmedian(exp)) if exp is not None and np.any(np.isfinite(exp))
                 else float(getattr(self, "exposure_time", 0.0) or 0.0))

        def fit(t, y, e, tc_c, half, seed):
            """Step 3: least squares, 4-sigma clip, refit.  (f, th, cov, resid, keep)"""
            if seed is None:
                dep = float(depth) if np.isfinite(depth) and depth > 0 else 0.01
                rp0 = float(np.clip(math.sqrt(dep), 0.02, 0.6))
                ars0 = xfit.ars_from_duration(P, min(dur, 0.45 * P), rp0, b=0.3)
                if not np.isfinite(ars0):
                    ars0 = 3.0
                ars0 = float(np.clip(ars0, max(1.6, 1.05 + rp0), 150.0))
                inc0 = 89.0
            else:
                rp0, ars0, inc0 = seed
            pr = xfit.TransitPriors(name=self._star_label(idx), per=P, tc_ref=tc_c,
                                    rp=rp0, ars=ars0, inc=inc0)
            opt = xfit.FitOptions(systematics="none", free_inc=True, free_ars=True,
                                  exptime_s=exp_s, use_beta=False,
                                  tc_window=(tc_c - half, tc_c + half))
            f = xfit.TransitFitter(t, y, e, None, pr, opt, nq=24)
            f.lo[f.ix["inc"]] = 60.0     # unknown candidate: let the impact parameter roam
            th0 = f.initial_theta(f.tc_grid_start(n=160) if seed is None else tc_c)
            th, cov = f.fit_lsq(th0)
            resid = f.y - f.model(th)
            keep = xfit._clip_mask(f.t, resid, 4.0)
            if np.count_nonzero(keep) < 0.8 * keep.size:
                keep = np.isfinite(resid)
            elif np.any(~keep):
                th, cov = f.fit_lsq(th, mask=keep)
                resid = f.y - f.model(th)
            return f, th, cov, resid, keep

        # Two passes: the transit mask from the BLS box, then from the fitted
        # duration (a box shorter than the transit leaves its wings in the
        # "baseline").
        in_tr = np.abs(fold_d(jd, tc_ref)) < 0.75 * dur
        tc_c, half, seed = tc_ref, float(min(0.25 * P, max(dur, 0.02 * P))), None
        for _pass in range(2):
            use, y_all, e_all, notes = prepare(in_tr)
            if np.count_nonzero(use) < 20:
                return None
            order = np.argsort(jd[use])
            t, y, e = jd[use][order], y_all[use][order], e_all[use][order]
            bad_e = ~(np.isfinite(e) & (e > 0))
            if np.all(bad_e):
                e = np.full_like(y, max(xfit.robust_std(np.diff(y)) / math.sqrt(2.0), 1e-5))
            elif np.any(bad_e):
                e[bad_e] = float(np.median(e[~bad_e]))
            f, th, cov, resid, keep = fit(t, y, e, tc_c, half, seed)
            g = lambda n: float(th[f.ix[n]]) if n in f.ix else float(f.fixed[n])
            tc, rp, ars, inc, c0 = g("tc"), g("rp"), g("ars"), g("inc"), g("c0")
            b = float(xfit.impact_parameter(ars, inc))
            if not (np.isfinite(rp) and rp > 0.003 and np.isfinite(b) and b < 1.0 + rp
                    and c0 > 0):
                print(f"[Analyze] transit model found no transit at P = {P:.5f} d "
                      f"(Rp/R* {rp:.4f}, b {b:.2f}); the fold shows the box only")
                return None
            t14 = float(xfit.transit_duration(P, rp, ars, inc))
            if _pass or not (np.isfinite(t14) and t14 > 0):
                break
            new_tr = np.abs(fold_d(jd, tc)) < 0.5 * t14 + max(0.25 * t14, 10.0 / 1440.0)
            if np.array_equal(new_tr, in_tr):
                break
            in_tr = new_tr
            tc_c, half, seed = tc, float(min(0.25 * P, max(0.5 * t14, 0.01 * P))), (rp, ars, inc)
        u1, u2 = float(f.fixed["u1"]), float(f.fixed["u2"])
        n_nights = int(np.unique(nights[use]).size)
        n_left_out = int(np.count_nonzero(ok & ~use))
        step_note = next((nt for nt in notes if nt[0] == "step"), None)
        depth_model = 1.0 - float(xfit.transit_model(np.array([tc]), tc, P, rp, ars, inc,
                                                     u1, u2)[0])
        n_k = int(np.count_nonzero(keep))
        chi2 = float(np.sum((resid[keep] / e[keep]) ** 2))
        chi2_red = chi2 / max(1, n_k - len(f.names))
        w = 1.0 / e[keep] ** 2
        ybar = float(np.sum(w * y[keep]) / np.sum(w))
        chi2_flat = float(np.sum(((y[keep] - ybar) / e[keep]) ** 2))
        # model vs a flat line, with the errors scaled up to the residual scatter
        snr = math.sqrt(max(0.0, chi2_flat - chi2) / max(1.0, chi2_red))
        sig = lambda n: (math.sqrt(max(0.0, float(cov[f.ix[n], f.ix[n]])) * max(1.0, chi2_red))
                         if cov is not None and n in f.ix else float("nan"))
        rms10 = xfit.binned_rms(t[keep], resid[keep] / c0, 10.0 / 1440.0)[0]
        steps = np.diff(t)
        steps = steps[(steps > 0) & (steps < 0.1)]
        cad = float(np.median(steps)) * 1440.0 if steps.size else float("nan")

        # model across the whole orbit, dense through the transit
        wide = max(t14 if np.isfinite(t14) else 0.0, dur)
        xd = np.unique(np.concatenate([np.linspace(-0.5 * P, 0.5 * P, 1500),
                                       np.linspace(-min(1.5 * wide, 0.5 * P),
                                                   min(1.5 * wide, 0.5 * P), 1500)]))
        mf = xfit.transit_model(tc + xd, tc, P, rp, ars, inc, u1, u2,
                                exptime_d=f.exp_d, n_sub=f.n_sub)

        # The separate transits the fold is built from.  The smooth model makes
        # a single event look as convincing as ten, but one event says nothing
        # about the period.
        h14 = 0.5 * (t14 if np.isfinite(t14) else dur)
        intr = keep & (np.abs(fold_d(t, tc)) <= h14)
        ep = np.round((t - tc) / P).astype(np.int64)
        events = []
        for e_ in np.unique(ep[intr]):
            n_e = int(np.count_nonzero(intr & (ep == e_)))
            if n_e >= 3:
                events.append((int(e_), float(tc + e_ * P), n_e))

        # Odd/even depths (the Kepler vetting test).  An eclipsing binary
        # found at half its period puts its primary and secondary eclipses on
        # alternate epochs, and they rarely have the same depth.  The fitted
        # shape is kept and only its depth is refitted, transit by transit --
        # linear, so partial transits count.  Errors: sqrt(chi2_red) x the
        # red-noise beta on the transit's own timescale (a depth averages over
        # the whole transit, so 10-30 min bins understate it), then x the
        # Birge ratio of the transits about their own parity's mean, so the
        # errors are as large as transits of the SAME parity actually
        # scatter.  Without that, one equal-depth binary in a synthetic test
        # came out 3.7 sigma "unequal".
        odd = np.mod(ep, 2) != 0
        odd_even = None
        n_odd = sum(1 for e_, _tc, _n in events if e_ % 2 != 0)
        n_even = len(events) - n_odd
        if n_odd and n_even:
            dip = 1.0 - xfit.transit_model(t, tc, P, rp, ars, inc, u1, u2,
                                           exptime_d=f.exp_d, n_sub=f.n_sub)
            yy, ee = y / c0, e / c0
            in_dip = keep & (dip > 0.05 * float(np.max(dip)))

            def wscale(sel):
                m = sel & in_dip
                wv = 1.0 / ee[m] ** 2
                return float(np.sum(wv * (1.0 - yy[m]) * dip[m]) / max(np.sum(wv * dip[m] ** 2),
                                                                       1e-30))

            # Noise measured with each parity at its own depth, so the
            # difference being tested doesn't count as noise.
            r2 = yy - (1.0 - np.where(odd, wscale(odd), wscale(~odd)) * dip)
            chi2_red2 = float(np.sum((r2[keep] / ee[keep]) ** 2)) / max(1, n_k - len(f.names) - 1)
            t14_min = t14 * 1440.0 if np.isfinite(t14) else 60.0
            beta, _ = xfit.red_noise_beta(
                t[keep], r2[keep],
                bin_minutes=sorted({max(5.0, round(t14_min * q)) for q in (0.25, 0.5, 0.75, 1.0)}))
            e_scale = math.sqrt(max(1.0, chi2_red2)) * max(1.0, beta)
            per_ev = []
            for e_, _tc, _n in events:
                m = in_dip & (ep == e_)
                if np.count_nonzero(m) < 3:
                    continue
                wv = 1.0 / ee[m] ** 2
                swss = float(np.sum(wv * dip[m] ** 2))
                per_ev.append((e_ % 2 != 0, float(np.sum(wv * (1.0 - yy[m]) * dip[m]) / swss),
                               e_scale / math.sqrt(swss)))
            par = {}
            for p_odd in (True, False):
                a = np.array([v for o_, v, _s in per_ev if o_ == p_odd])
                s = np.array([s_ for o_, _v, s_ in per_ev if o_ == p_odd])
                if a.size:
                    wv = 1.0 / s ** 2
                    par[p_odd] = (float(np.sum(wv * a) / np.sum(wv)),
                                  float(1.0 / math.sqrt(np.sum(wv))), a, s)
            if len(par) == 2:
                chi_w = sum(float(np.sum(((a - m_) / s) ** 2)) for m_, _s, a, s in par.values())
                dof = len(per_ev) - 2
                birge = math.sqrt(max(1.0, chi_w / dof)) if dof >= 1 else 1.0
                (a_o, s_o, _, _), (a_e, s_e, _, _) = par[True], par[False]
                s_o, s_e = s_o * birge, s_e * birge
                odd_even = {"a_odd": a_o, "a_even": a_e,
                            "odd_ppt": a_o * depth_model * 1e3, "odd_err": s_o * depth_model * 1e3,
                            "even_ppt": a_e * depth_model * 1e3, "even_err": s_e * depth_model * 1e3,
                            "n_odd": int(par[True][2].size), "n_even": int(par[False][2].size),
                            "beta": float(beta), "birge": float(birge),
                            "sigma": abs(a_o - a_e) / max(math.hypot(s_o, s_e), 1e-12)}
        return {
            "t": t, "events": events, "n_in": int(np.count_nonzero(intr)),
            "odd": odd, "odd_even": odd_even, "notes": notes, "n_left_out": n_left_out,
            "x_h": fold_d(t, tc) * 24.0, "y": y / c0, "e": e / c0, "used": keep,
            "resid_ppt": resid / c0 * 1e3, "model_x_h": xd * 24.0, "model_y": mf,
            "tc": tc, "tc_err": sig("tc"), "rp": rp, "rp_err": sig("rp"), "ars": ars,
            "inc": inc, "b": b, "u1": u1, "u2": u2, "t14_h": t14 * 24.0,
            "depth_ppt": depth_model * 1e3, "rms_ppt": float(np.std(resid[keep] / c0)) * 1e3,
            "rms10_ppt": rms10 * 1e3, "snr": snr, "chi2_red": chi2_red, "n_sub": int(f.n_sub),
            "exp_s": exp_s, "cadence_min": cad, "n_nights": n_nights,
            "n_clipped": int(keep.size - n_k),
        }

    def on_analyze(self):
        sels = self.star_list.selectedItems()
        if len(sels) != 1:
            QMessageBox.information(self, "Analyze", "Please select exactly one star.")
            return
        idx = sels[0].data(Qt.ItemDataRole.UserRole)

        # ── 1. Build raw time/flux arrays ────────────────────────────────────
        t_jd  = self.times.utc.jd                  # full JD
        f_all = self.fluxes[idx]
        good  = np.isfinite(f_all) & (self.flags[idx] == 0)
        t_jd_good = t_jd[good]
        f_good    = f_all[good]

        if len(t_jd_good) < 10:
            QMessageBox.warning(self, "Analyze", "Not enough good points to analyze.")
            return

        # ── 2. Lomb-Scargle on relative MJD (unit-median flux) ───────────────
        t_rel = t_jd_good - t_jd_good[0]          # days from first obs
        f_norm = f_good / np.nanmedian(f_good[np.isfinite(f_good) & (f_good > 0)])

        ls = LombScargle(t_rel, f_norm)
        Tspan = float(np.ptp(t_rel))
        steps = np.diff(np.sort(t_rel))
        steps = steps[steps > 0]          # duplicate timestamps would make dt 0
        dt    = float(np.median(steps)) if steps.size else Tspan
        min_f = max(1.0 / Tspan, 1e-6)
        max_f = 0.5 / max(dt, 1e-9)
        # The grid has samples_per_peak x span x (max_f - min_f) frequencies;
        # keep it under ~1M whatever the span and cadence.
        spp = float(getattr(self, "ls_samples_per_peak", 10))
        n_ls = spp * Tspan * (max_f - min_f)
        if n_ls > 1_000_000:
            spp = max(1.0, spp * 1_000_000 / n_ls)
            max_f = min(max_f, min_f + 1_000_000 / (spp * Tspan))
        freq, power_ls = ls.autopower(
            minimum_frequency=min_f,
            maximum_frequency=max_f,
            samples_per_peak=spp,
        )
        mask = (freq > 0) & np.isfinite(power_ls)
        freq, power_ls = freq[mask], power_ls[mask]
        periods_ls = 1.0 / freq
        order = np.argsort(periods_ls)
        periods_ls, power_ls = periods_ls[order], power_ls[order]
        best_ls_period = float(periods_ls[np.argmax(power_ls)])

        # ── 3. Prepare cleaned light curve for BLS ───────────────────────────
        # BLS input: each night normalised to its own median.  The levels
        # between nights are kept everywhere else (they are real for a
        # long-period variable, and Lomb-Scargle above uses them), but BLS
        # can only read a step between two nights as a box.
        f_bls = np.asarray(f_good, dtype=np.float64).copy()
        S_all = getattr(self, "session_idx", None)
        if S_all is not None and len(S_all) == len(f_all):
            S_good = np.asarray(S_all)[good]
            for k in np.unique(S_good):
                mk = (S_good == k) & np.isfinite(f_bls) & (f_bls > 0)
                if np.count_nonzero(mk) >= 3:
                    f_bls[S_good == k] /= float(np.median(f_bls[mk]))
        t_clean, y_clean = _prepare_search_lightcurve(t_jd_good, f_bls)
        if t_clean.size < 20:
            QMessageBox.warning(self, "Analyze", "Too few clean points for BLS.")
            return

        baseline_days = float(np.ptp(t_clean))
        bls_min_p = float(getattr(self, "bls_min_period", 0.5))
        bls_max_p = min(
            float(getattr(self, "bls_max_period", 30.0)),
            0.95 * baseline_days,
        )
        if bls_max_p <= bls_min_p:
            bls_max_p = max(bls_min_p * 2.0, 0.95 * baseline_days)

        # Duration grid
        min_p_for_dur = bls_min_p
        dur_min_frac  = float(getattr(self, "bls_duration_min_frac", 0.01))
        dur_max_frac  = float(getattr(self, "bls_duration_max_frac", 0.10))
        n_durations   = int(getattr(self, "bls_n_durations", 8))
        dur_grid = np.linspace(
            dur_min_frac * min_p_for_dur,
            dur_max_frac * min_p_for_dur,
            n_durations,
        )  # days

        # ── 4. BLS with whitened power + multi-peak scanning ─────────────────
        dy_scalar = _robust_sigma(y_clean)
        if not (np.isfinite(dy_scalar) and dy_scalar > 0):
            dy_scalar = float(np.nanstd(y_clean - np.nanmedian(y_clean)))
        if not (np.isfinite(dy_scalar) and dy_scalar > 0):
            dy_scalar = 1e-4

        bls_model = BoxLeastSquares(
            t_clean * u.day,
            y_clean,
            dy=np.full_like(y_clean, dy_scalar),
        )
        # astropy's autoperiod grid, uniform in frequency with step
        # min(duration) / baseline^2 -- capped.  When the cap applies (several
        # nights), each peak of the coarse scan is refined below at the full
        # resolution, so only the coarse pass gives up resolution.
        f_lo, f_hi = 1.0 / bls_max_p, 1.0 / bls_min_p
        df_full = float(np.min(dur_grid)) / max(baseline_days, 1e-9) ** 2
        nf_full = int(np.ceil((f_hi - f_lo) / df_full)) + 1
        nf = int(min(nf_full, self.BLS_MAX_PERIODS))
        coarse = nf < nf_full
        df_used = (f_hi - f_lo) / max(nf - 1, 1)
        if coarse:
            print(f"[Analyze] BLS: {nf_full:,} periods at full resolution over a "
                  f"{baseline_days:.1f} d baseline; scanning {nf:,}, then refining the peaks")
        prev_status = self.status_label.text()
        self.status_label.setText(f"Analyzing star #{idx}: box least squares…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self._bls_scan(bls_model, 1.0 / np.linspace(f_hi, f_lo, nf), dur_grid,
                                    oversample=10)
        finally:
            QApplication.restoreOverrideCursor()
        if result is None:
            self.status_label.setText("Analysis cancelled.")
            return

        power_raw = np.asarray(result.power, float).ravel()
        periods_bls = np.asarray(result.period.value
                                if hasattr(result.period, "value")
                                else result.period, float).ravel()

        if power_raw.size == 0 or not np.any(np.isfinite(power_raw)):
            QMessageBox.warning(self, "Analyze", "BLS returned no usable power spectrum.")
            return

        bg_raw, power_white = _whiten_periodogram(power_raw)
        use_white = bool(np.any(np.isfinite(power_white)))
        power_metric = power_white if use_white else power_raw
        white_off = float(np.nanmedian(power_raw - bg_raw)) if use_white else 0.0

        # Compute SDE from whitened power
        fin_m = np.isfinite(power_metric)
        power_med = float(np.nanmedian(power_metric[fin_m]))
        power_std = float(max(
            _robust_sigma(power_metric[fin_m]),
            float(np.nanstd(power_metric[fin_m])),
            1e-12,
        ))

        # Scan top peaks, pick best by SNR after local refinement
        peak_idxs = _top_peak_indices(periods_bls, power_metric, max_peaks=12, merge_frac=0.01)
        if not peak_idxs:
            peak_idxs = [int(np.nanargmax(power_metric))]

        best_period_d  = np.nan
        best_duration_d = np.nan
        best_t0_jd     = np.nan
        best_depth_frac = np.nan
        best_sde       = -np.inf
        best_metric    = -np.inf

        # Local refinement windows, scanned together in one parallel pass:
        # ±20 % around each peak in 101 steps, or -- after a capped coarse
        # scan -- ±1.5 coarse steps around it at the full frequency resolution.
        windows = []
        for pi in peak_idxs:
            cand_period = float(periods_bls[pi])
            if not (np.isfinite(cand_period) and bls_min_p <= cand_period <= bls_max_p):
                continue
            if coarse:
                fc = 1.0 / cand_period
                n_fine = int(min(4000, max(101, np.ceil(3.0 * df_used / df_full) + 1)))
                p_grid = 1.0 / np.linspace(min(f_hi, fc + 1.5 * df_used),
                                           max(f_lo, fc - 1.5 * df_used), n_fine)
            else:
                half_w = max(0.01, 0.20 * cand_period)
                p_grid = np.linspace(max(bls_min_p, cand_period - half_w),
                                     min(bls_max_p, cand_period + half_w), 101)
            windows.append((pi, cand_period, p_grid))
        refined = None
        if windows:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                refined = self._bls_scan(bls_model, np.concatenate([w[2] for w in windows]),
                                         dur_grid, oversample=5,
                                         what="periods around the strongest peaks")
                if refined is None:
                    self.status_label.setText("Analysis cancelled.")
                    return
            except Exception as e:
                print(f"[Analyze] BLS refinement failed: {e}")
                refined = None
            finally:
                QApplication.restoreOverrideCursor()
        offs = np.cumsum([0] + [w[2].size for w in windows])
        fin_bg = np.isfinite(bg_raw)

        for j, (pi, cand_period, _pg) in enumerate(windows):
            try:
                if refined is None:
                    raise RuntimeError("no refinement")
                sl = slice(int(offs[j]), int(offs[j + 1]))
                ref_power = refined.power[sl]
                ref_periods = refined.period[sl]
                if coarse and use_white and np.any(fin_bg):
                    # the coarse spectrum's own background, so the SDE below
                    # compares like with like (a ±1.5-step window is too
                    # narrow to whiten on its own)
                    ref_metric = (ref_power - white_off
                                  - np.interp(ref_periods, periods_bls[fin_bg], bg_raw[fin_bg]))
                else:
                    _, ref_white = _whiten_periodogram(ref_power)
                    ref_metric = ref_white if np.any(np.isfinite(ref_white)) else ref_power
                k = int(np.nanargmax(ref_metric))

                cand_p  = float(ref_periods[k])
                cand_d  = float(refined.duration[sl][k])
                cand_t0 = float(refined.transit_time[sl][k])
                cand_depth = abs(float(refined.depth[sl][k]))
                cand_metric = float(ref_metric[k]) if np.isfinite(ref_metric[k]) else float(ref_power[k])
            except Exception:
                cand_p  = cand_period
                cand_d  = float(result.duration[pi].value
                                if hasattr(result.duration[pi], "value")
                                else result.duration[pi])
                cand_t0 = float(result.transit_time[pi].value
                                if hasattr(result.transit_time[pi], "value")
                                else result.transit_time[pi])
                cand_depth = abs(float(result.depth[pi].value
                                    if hasattr(result.depth[pi], "value")
                                    else result.depth[pi])) if hasattr(result, "depth") else np.nan
                cand_metric = float(power_metric[pi]) if np.isfinite(power_metric[pi]) else 0.0

            sde = (cand_metric - power_med) / power_std if power_std > 0 else 0.0

            # Apply alias penalty
            alias_pen = _alias_penalty(cand_p)
            sde_adj = sde - 1.5 * alias_pen

            if sde_adj > best_sde:
                best_sde        = sde_adj
                best_period_d   = cand_p
                best_duration_d = cand_d
                best_t0_jd      = cand_t0
                best_depth_frac = cand_depth
                best_metric     = cand_metric

        if not np.isfinite(best_period_d):
            # Fall back to raw argmax
            k = int(np.nanargmax(power_metric))
            best_period_d   = float(periods_bls[k])
            best_duration_d = float(result.duration[k].value
                                    if hasattr(result.duration[k], "value")
                                    else result.duration[k])
            best_t0_jd      = float(result.transit_time[k].value
                                    if hasattr(result.transit_time[k], "value")
                                    else result.transit_time[k])
            best_depth_frac = abs(float(result.depth[k].value
                                        if hasattr(result.depth[k], "value")
                                        else result.depth[k])) if hasattr(result, "depth") else np.nan
            best_sde        = (float(power_metric[k]) - power_med) / power_std if power_std > 0 else 0.0

        self.status_label.setText(prev_status)

        # ── 5. Phase fold ────────────────────────────────────────────────────
        # Shown in hours from mid-transit with the limb-darkened transit model
        # and its residuals (_fold_transit_model); the BLS box stays as a
        # dashed reference.  Without a model (no fit, or the fit found no
        # transit) the fold shows the BLS input and the box, as before.
        fold = None
        if np.isfinite(best_period_d) and np.isfinite(best_t0_jd):
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                fold = self._fold_transit_model(idx, best_period_d, best_t0_jd,
                                                best_duration_d, best_depth_frac)
            except Exception as e:
                print(f"[Analyze] transit model on the fold failed: {e}")
                fold = None
            finally:
                QApplication.restoreOverrideCursor()
        depth_plot = float(best_depth_frac) if np.isfinite(best_depth_frac) else 0.0
        P_h = best_period_d * 24.0
        tc_plot = fold["tc"] if fold else best_t0_jd

        def fold_h(t):
            return ((np.asarray(t, dtype=np.float64) - tc_plot) / best_period_d + 0.5) % 1.0 \
                * P_h - 0.5 * P_h

        if fold:
            fx, fy, fe, fused = fold["x_h"], fold["y"], fold["e"], fold["used"]
        else:
            fx, fy, fe = fold_h(t_clean), y_clean, None
            fused = np.ones(fx.size, bool)
        # BLS box, centred where BLS put it
        box_c = float(fold_h(best_t0_jd)) if np.isfinite(best_t0_jd) else 0.0
        box_hw = min(0.45 * best_period_d, 0.5 * best_duration_d) * 24.0
        box_x = np.array([-0.5 * P_h, box_c - box_hw, box_c - box_hw,
                          box_c + box_hw, box_c + box_hw, 0.5 * P_h])
        box_y = np.array([1.0, 1.0, 1.0 - depth_plot, 1.0 - depth_plot, 1.0, 1.0])

        # Bins: 10 minutes, finer for a short transit (5 bins across it, but
        # never under two exposures)
        t14_min = (fold["t14_h"] if fold else best_duration_d * 24.0) * 60.0
        steps = np.diff(np.sort(t_clean))
        steps = steps[(steps > 0) & (steps < 0.1)]
        cad_min = float(np.median(steps)) * 1440.0 if steps.size else 2.0
        bin_min = 10.0
        if np.isfinite(t14_min) and t14_min < 50.0:
            bin_min = max(t14_min / 5.0, 2.0 * cad_min)
        bx, by, be = xphot.bin_curve(fx[fused], fy[fused], bin_min / 60.0,
                                     e=(fe[fused] if fe is not None else None))

        # Dip event detection for the summary
        dip_events = _estimate_night_dip_events(t_jd_good, f_good)

        # ── 6. Build dialog ───────────────────────────────────────────────────
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Analysis: {self._star_label(idx)}  |  P = {best_period_d:.4f} d  |  SDE = {best_sde:.1f}")
        layout = QVBoxLayout(dlg)

        # Summary label
        dur_hr   = best_duration_d * 24.0
        depth_ppt = depth_plot * 1000.0
        n_dips   = len(dip_events)
        alias_pen = _alias_penalty(best_period_d)
        summary = (
            f"BLS — Period: {best_period_d:.5f} d   "
            f"Box duration: {dur_hr:.2f} h   "
            f"Box depth: {depth_ppt:.2f} ppt   "
            f"SDE: {best_sde:.1f}   "
            f"Alias penalty: {alias_pen:.2f}   "
            f"Night dips detected: {n_dips}"
        )
        warn = []
        if fold:
            def pm(v, s, fmt):
                if not np.isfinite(s):
                    return f"{v:{fmt}}"
                return f"{v:{fmt}}" + (f" ± {s:{fmt}}" if s <= 3.0 * abs(v) else " (unconstrained)")

            # Which nights the transits come from, and the aperture test on each:
            # a dip that fades in a bigger aperture is a PSF change, and a
            # transit model fitted to it looks every bit as convincing.
            S_all = getattr(self, "session_idx", None)
            jd_frames = np.asarray(self.times.utc.jd, dtype=np.float64)
            jd0 = getattr(self, "_frame_jd_utc", None)
            jd0 = float(jd0[0]) if jd0 is not None and len(jd0) else None
            ev_dates = []
            for _e, tc_e, _n in fold["events"]:
                date = ""
                if S_all is not None and len(S_all) == jd_frames.size and self.sessions:
                    s = int(np.asarray(S_all)[int(np.argmin(np.abs(jd_frames - tc_e)))])
                    if 0 <= s < len(self.sessions):
                        date = self.sessions[s].get("date") or ""
                ev_dates.append(date or f"JD {tc_e:.2f}")
                if jd0 is not None:
                    try:
                        ap = self._dip_aperture_test(idx, {"dip_t": tc_e - jd0,
                                                           "dip_dur_h": fold["t14_h"]})
                    except Exception:
                        ap = None
                    if ap is not None and ap[0]:
                        warn.append(f"The {ev_dates[-1]} transit fades in a bigger aperture "
                                    f"({ap[1]:.0f} → {ap[2]:.0f} ppt at r = {ap[3]:.1f} → "
                                    f"{ap[4]:.1f} px): a PSF change, not an eclipse.")
            n_ev = len(fold["events"])
            if n_ev == 0:
                warn.insert(0, "No night covers the transit itself.")
            oe = fold.get("odd_even")
            if oe is not None and oe["sigma"] > 3.0:
                warn.append(f"Odd and even transits differ by {oe['sigma']:.1f}σ "
                            f"({oe['odd_ppt']:.0f} vs {oe['even_ppt']:.0f} ppt).  Either they "
                            "are an eclipsing binary's primary and secondary eclipses (true "
                            f"period 2 × P = {2.0 * best_period_d:.5f} d), or the period is "
                            "wrong and one set isn't a transit at all.")
            summary += (
                f"\nTransit model (limb-darkened, u1 {fold['u1']:.3f}, u2 {fold['u2']:.3f} fixed) — "
                f"Rp/R* {pm(fold['rp'], fold['rp_err'], '.4f')}   "
                f"depth {fold['depth_ppt']:.2f} ppt   T14 {fold['t14_h']:.2f} h   "
                f"b {fold['b']:.2f}   a/R* {fold['ars']:.1f}   "
                f"Tc {pm(fold['tc'], fold['tc_err'], '.5f')} JD_UTC   "
                f"model vs flat line {fold['snr']:.1f}σ   "
                f"residual RMS {fold['rms_ppt']:.2f} ppt/point"
                + (f", {fold['rms10_ppt']:.2f} ppt per 10 min"
                   if np.isfinite(fold['rms10_ppt']) else ""))
            if n_ev == 1:
                # The usual case: one dip, then the hunt for the next one.
                summary += (f"\nSingle transit, {ev_dates[0]} ({fold['n_in']} points in transit): "
                            "Tc, depth and T14 above are its measurement.  One transit doesn't "
                            f"fix the period: {best_period_d:.5f} d is just a period the other "
                            "nights don't rule out, and the next transit may be weeks or months "
                            "away.")
            else:
                summary += (f"\nIn transit: {fold['n_in']} points from {n_ev} transits"
                            + (f" ({', '.join(ev_dates)})" if ev_dates else ""))
            for nt in fold.get("notes", []):
                if nt[0] == "night":
                    s = nt[1]
                    date = (self.sessions[s].get("date") if 0 <= s < len(self.sessions or [])
                            else "") or f"night {s + 1}"
                    summary += (f"\n{date} left out ({nt[2]} frames): every frame is inside the "
                                "transit, so the night has no baseline.")
                elif nt[0] == "orphan":
                    summary += (f"\n{nt[2]} frames left out: the meridian flip falls inside the "
                                "transit, and no other night has both pier sides to measure the "
                                "flip step from.")
                elif nt[0] == "step":
                    summary += (f"\nFlip inside the transit on {nt[1]} night"
                                + ("s" if nt[1] != 1 else "")
                                + f": the far pier side is set by the flip step from the other "
                                  f"nights ({(nt[2] - 1.0) * 1e3:+.1f} ppt).")
            if oe is not None:
                summary += (f"\nOdd/even transits: {oe['odd_ppt']:.1f} ± {oe['odd_err']:.1f} vs "
                            f"{oe['even_ppt']:.1f} ± {oe['even_err']:.1f} ppt "
                            f"({oe['n_odd']} odd, {oe['n_even']} even, red-noise β "
                            f"{oe['beta']:.2f}) — {oe['sigma']:.1f}σ apart")
            if fold["rp"] >= 0.3 or fold["b"] >= 0.9:
                warn.append("Deep or V-shaped: typical of an eclipsing binary.  An EW's two "
                            "eclipses fold onto one at half the period — check the light curve "
                            "at 2 × P.")
        else:
            summary += "\nNo transit model: the fold shows the BLS box only."
        lbl = QLabel(summary)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setStyleSheet("font-size: 11px; color: #aaddff; padding: 4px;")
        layout.addWidget(lbl)
        if warn:
            wl = QLabel("\n".join(warn))
            wl.setWordWrap(True)
            wl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            wl.setStyleSheet("font-size: 11px; color: #ffb070; padding: 0px 4px 4px 4px;")
            layout.addWidget(wl)

        tabs = __import__("PyQt6.QtWidgets", fromlist=["QTabWidget"]).QTabWidget()
        layout.addWidget(tabs)

        # ── Tab 1: Lomb-Scargle ───────────────────────────────────────────────
        pg_ls = pg.PlotWidget(title="Lomb–Scargle Periodogram")
        pg_ls.plot(periods_ls, power_ls, pen=pg.mkPen("w", width=1))
        pg_ls.addLine(x=best_ls_period,
                    pen=pg.mkPen("y", style=Qt.PenStyle.DashLine, width=2))
        pg_ls.setLabel("bottom", "Period [d]")
        pg_ls.setLabel("left", "LS Power")
        pg_ls.showGrid(True, True, alpha=0.3)
        tabs.addTab(pg_ls, "Lomb-Scargle")

        # ── Tab 2: BLS Periodogram (whitened) ────────────────────────────────
        pg_bls = pg.PlotWidget(title="BLS Periodogram (whitened SNR)")
        pg_bls.plot(periods_bls, power_metric, pen=pg.mkPen("c", width=1))
        pg_bls.addLine(x=best_period_d,
                    pen=pg.mkPen("r", style=Qt.PenStyle.DashLine, width=2))
        pg_bls.setLabel("bottom", "Period [d]")
        pg_bls.setLabel("left", "Whitened Power")
        pg_bls.showGrid(True, True, alpha=0.3)
        tabs.addTab(pg_bls, "BLS Periodogram")

        # ── Tab 3: Phase fold ─────────────────────────────────────────────────
        # Same look as Fit Transit: detrended points, binned means with their
        # errors, the transit model, residuals underneath on a linked axis.
        n_nt = fold["n_nights"] if fold else 0
        pg_fold = pg.PlotWidget(
            title=f"{self._star_label(idx)} — folded at P = {best_period_d:.5f} d"
                  + (f" ({n_nt} nights)" if n_nt > 1 else "")
                  + (" — detrended, transit masked" if fold else ""))
        pg_fold.addLegend(offset=(10, 10))
        pg_fold.plot(fx[fused], fy[fused], pen=None, symbol="o", symbolSize=4,
                     symbolBrush=pg.mkBrush(150, 190, 255, 130), symbolPen=None,
                     name="detrended" if fold else "BLS input")
        if np.any(~fused):
            pg_fold.plot(fx[~fused], fy[~fused], pen=None, symbol="x", symbolSize=8,
                         symbolPen=pg.mkPen((255, 80, 80)), name="clipped")
        if bx.size > 0:
            pg_fold.addItem(pg.ErrorBarItem(x=bx, y=by, height=2 * be,
                                            pen=pg.mkPen((255, 210, 80), width=1)))
            pg_fold.plot(bx, by, pen=None, symbol="o", symbolSize=7,
                         symbolBrush=pg.mkBrush(255, 210, 80), symbolPen=None,
                         name=f"{bin_min:.0f}-min bins" if bin_min >= 10 else f"{bin_min:.1f}-min bins")
        pg_fold.plot(box_x, box_y,
                     pen=pg.mkPen((200, 200, 200), width=1, style=Qt.PenStyle.DashLine),
                     name="BLS box")
        if fold:
            pg_fold.plot(fold["model_x_h"], fold["model_y"],
                         pen=pg.mkPen((255, 90, 90), width=2), name="transit model")
        pg_fold.setLabel("bottom", "Hours from mid-transit")
        pg_fold.setLabel("left", "Relative flux")
        pg_fold.showGrid(True, True, alpha=0.3)
        # Open on the transit; the whole orbit is one zoom-out (or the A button) away.
        span_h = max((fold["t14_h"] if fold else 0.0), best_duration_d * 24.0, 0.5)
        x_half = min(0.5 * P_h, 2.5 * span_h)
        pg_fold.setXRange(-x_half, x_half, padding=0.02)
        vis = fused & (np.abs(fx) <= x_half) & np.isfinite(fy)
        if np.count_nonzero(vis) >= 5:
            lo, hi = np.percentile(fy[vis], [0.5, 99.5])
            if fold:
                lo = min(lo, float(np.min(fold["model_y"])))
            pad = 0.08 * max(hi - lo, 1e-4)
            pg_fold.setYRange(lo - pad, hi + pad, padding=0)

        fold_page = QSplitter(Qt.Orientation.Vertical)
        fold_page.setChildrenCollapsible(False)
        fold_page.addWidget(pg_fold)
        if fold:
            pg_res = pg.PlotWidget(title="Residuals")
            pg_res.showGrid(True, True, alpha=0.3)
            pg_res.setLabel("left", "ppt")
            pg_res.setLabel("bottom", "Hours from mid-transit")
            pg_res.setXLink(pg_fold)
            r = fold["resid_ppt"]
            pg_res.plot(fx[fused], r[fused], pen=None, symbol="o", symbolSize=4,
                        symbolBrush=pg.mkBrush(150, 190, 255, 130), symbolPen=None)
            rb_x, rb_y, _rb_e = xphot.bin_curve(fx[fused], r[fused], bin_min / 60.0,
                                                e=fe[fused] * 1e3)
            if rb_x.size:
                pg_res.plot(rb_x, rb_y, pen=None, symbol="o", symbolSize=6,
                            symbolBrush=pg.mkBrush(255, 210, 80), symbolPen=None)
            pg_res.addLine(y=0.0, pen=pg.mkPen((200, 200, 200), style=Qt.PenStyle.DashLine))
            rv = fused & (np.abs(fx) <= x_half) & np.isfinite(r)
            if np.count_nonzero(rv) >= 5:
                rlo, rhi = np.percentile(r[rv], [0.5, 99.5])
                rpad = 0.1 * max(rhi - rlo, 1e-3)
                pg_res.setYRange(rlo - rpad, rhi + rpad, padding=0)
            fold_page.addWidget(pg_res)
            fold_page.setStretchFactor(0, 3)
            fold_page.setStretchFactor(1, 1)
            fold_page.setSizes([450, 170])
        tabs.addTab(fold_page, "Phase Fold")

        # ── Tab 3b: Odd / Even ────────────────────────────────────────────────
        # The two halves side by side on shared axes: each with the transit
        # model scaled to its own depth (red) and the joint model (dashed).
        oe = fold.get("odd_even") if fold else None
        if oe is not None:
            oe_page = QWidget()
            oe_lay = QHBoxLayout(oe_page)
            oe_lay.setContentsMargins(0, 0, 0, 0)
            y_view = pg_fold.viewRange()[1]
            panels = []
            for name, sel, a, d, s, n in (
                    ("Odd", fold["odd"], oe["a_odd"], oe["odd_ppt"], oe["odd_err"], oe["n_odd"]),
                    ("Even", ~fold["odd"], oe["a_even"], oe["even_ppt"], oe["even_err"],
                     oe["n_even"])):
                pw = pg.PlotWidget(title=f"{name} transits ({n}): {d:.1f} ± {s:.1f} ppt")
                m = sel & fused
                pw.plot(fx[m], fy[m], pen=None, symbol="o", symbolSize=4,
                        symbolBrush=pg.mkBrush(150, 190, 255, 130), symbolPen=None)
                tb, yb, eb = xphot.bin_curve(fx[m], fy[m], bin_min / 60.0, e=fe[m])
                if tb.size:
                    pw.addItem(pg.ErrorBarItem(x=tb, y=yb, height=2 * eb,
                                               pen=pg.mkPen((255, 210, 80), width=1)))
                    pw.plot(tb, yb, pen=None, symbol="o", symbolSize=7,
                            symbolBrush=pg.mkBrush(255, 210, 80), symbolPen=None)
                pw.plot(fold["model_x_h"], fold["model_y"],
                        pen=pg.mkPen((200, 200, 200), width=1, style=Qt.PenStyle.DashLine))
                pw.plot(fold["model_x_h"], 1.0 - a * (1.0 - fold["model_y"]),
                        pen=pg.mkPen((255, 90, 90), width=2))
                pw.setLabel("bottom", "Hours from mid-transit")
                pw.setLabel("left", "Relative flux")
                pw.showGrid(True, True, alpha=0.3)
                pw.setXRange(-x_half, x_half, padding=0.02)
                pw.setYRange(y_view[0], y_view[1], padding=0)
                oe_lay.addWidget(pw)
                panels.append(pw)
            panels[1].setXLink(panels[0])
            panels[1].setYLink(panels[0])
            tabs.addTab(oe_page, "Odd / Even")

        # ── Tab 4: Raw light curve ────────────────────────────────────────────
        pg_lc = pg.PlotWidget(title="Raw Light Curve")
        # t_jd_good and f_good are already filtered (finite + flag==0)
        # so t_plot and f_norm are the same size — no secondary mask needed
        t_plot = t_jd_good - t_jd_good[0]
        f_norm_plot = f_good / np.nanmedian(f_good[np.isfinite(f_good) & (f_good > 0)])
        pg_lc.plot(
            t_plot, f_norm_plot,
            pen=None, symbol="o",
            symbolBrush=pg.mkBrush(180, 200, 255, 160),
            symbolSize=4,
        )
        # Mark predicted transit centers
        if np.isfinite(best_t0_jd) and np.isfinite(best_period_d):
            t0_rel = best_t0_jd - t_jd_good[0]
            span   = float(np.ptp(t_plot))
            epoch  = math.floor((0.0 - t0_rel) / best_period_d)
            while True:
                tc = t0_rel + epoch * best_period_d
                if tc > span + best_period_d:
                    break
                if tc >= -best_period_d:
                    pg_lc.addLine(
                        x=tc,
                        pen=pg.mkPen("r", style=Qt.PenStyle.DashLine, width=1),
                    )
                epoch += 1
        pg_lc.setLabel("bottom", "Days from first obs")
        pg_lc.setLabel("left", "Relative Flux")
        pg_lc.showGrid(True, True, alpha=0.3)
        tabs.addTab(pg_lc, "Light Curve")

        dlg.resize(1000, 700)
        dlg.exec()

    # ---- shared SIMBAD/VSX name resolver ------------------------------
    #
    # Query SIMBAD around (ra, dec) at a moderately wide radius, then score
    # every returned row so that variable-star otypes and named catalog IDs
    # rank above generic Gaia DR3 entries.  Ties break on angular offset.
    # If the winning row still isn't obviously a variable, ask VSX (by
    # coordinate, NOT by the main_id we just got — VSX has no reverse
    # index on Gaia identifiers) whether anything in the region carries a
    # variable-star designation, and upgrade the name if so.
    #
    # This fixes the exported #NAME landing on "Gaia DR3 5790…" when the
    # user clicked V336 Aps: with a 5″ SIMBAD radius and a distance-only
    # sort, a Gaia catalog entry sitting 2–3″ from the click could beat the
    # actual Mira in row 0, and the old VSX fallback (query_object on that
    # Gaia string) then returned nothing.

    def _resolve_target_name_row(self, ra_deg, dec_deg, radius_arcsec=15.0):
        """Return dict {main_id, otype, vmag, offset_arcsec, is_variable,
        used_vsx} for the best-scoring match near (ra_deg, dec_deg), or
        None on failure / no match.  Never raises."""
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
        table = None

        # Preflight: ~4 s to learn CDS is down, versus 5 unbounded attempts
        # at the OS connect timeout. Once the breaker is tripped this costs
        # nothing on later lookups in the same session.
        if not cds_health.simbad_available():
            print(f"[DEBUG] SIMBAD skipped — {cds_health.status_text()}")
            return self._vsx_only_lookup(coord, radius_arcsec)

        # Two attempts, not five, and a connection failure breaks out
        # immediately instead of buying the same timeout four more times.
        # note_failure() moves astroquery to the other SIMBAD mirror when one
        # answers, so the second attempt is on a different host.
        for attempt in range(1, 3):
            try:
                custom = _new_simbad()
                custom.add_votable_fields("otype")
                custom.add_votable_fields("flux(V)")
                table = custom.query_region(coord, radius=radius_arcsec*u.arcsec)
                cds_health.note_success()
                break
            except Exception as e:
                print(f"[DEBUG] SIMBAD attempt {attempt} failed: {e}")
                gave_up = cds_health.note_failure(e, context="exoplanet/resolve_target")
                if gave_up or attempt == 2:
                    # Every mirror refused the astroquery path: try the TAP
                    # endpoints directly (different client stack), then VSX.
                    table = _simbad_region_direct(ra_deg, dec_deg, radius_arcsec)
                    if table is None:
                        print("[DEBUG] SIMBAD unreachable — falling back to VSX.")
                        return self._vsx_only_lookup(coord, radius_arcsec)
                    break
                non_blocking_sleep(1)

        if table is None or len(table) == 0:
            # SIMBAD blank — try VSX directly as a last resort.
            return self._vsx_only_lookup(coord, radius_arcsec)

        try:
            id_col    = next(c for c in table.colnames if c.lower() == "main_id")
            ra_col    = next(c for c in table.colnames if c.lower() == "ra")
            dec_col   = next(c for c in table.colnames if c.lower() == "dec")
        except StopIteration:
            return None
        otype_col = next((c for c in table.colnames if c.lower() == "otype"), None)
        flux_col  = next((c for c in table.colnames if c.upper() in ("FLUX_V", "V")), None)

        best = None
        best_score = None
        for row in table:
            name = row[id_col]
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            name = str(name).strip()

            try:
                rra  = float(row[ra_col])
                rdec = float(row[dec_col])
            except Exception:
                continue
            match  = SkyCoord(ra=rra*u.deg, dec=rdec*u.deg, frame="icrs")
            offset = float(coord.separation(match).arcsec)

            otype = ""
            if otype_col is not None:
                ov = row[otype_col]
                if isinstance(ov, bytes):
                    ov = ov.decode("utf-8")
                otype = str(ov).strip()

            vmag = None
            if flux_col is not None:
                try:
                    fv = float(row[flux_col])
                    if np.isfinite(fv):
                        vmag = fv
                except Exception:
                    vmag = None

            score = 0.0
            if _is_variable_otype(otype):
                score += 100.0
            score += _name_score(name)
            score -= offset  # closer wins ties

            if (best_score is None) or (score > best_score):
                best_score = score
                best = {
                    "main_id":       name,
                    "otype":         otype,
                    "vmag":          vmag,
                    "offset_arcsec": offset,
                    "is_variable":   _is_variable_otype(otype),
                    "used_vsx":      False,
                }

        if best is None:
            return None

        # If the winner still isn't a variable, see if VSX has something
        # nearby that we should upgrade to.
        if not best["is_variable"]:
            vsx_name = self._vsx_name_near(coord, radius_arcsec)
            if vsx_name:
                best["main_id"]     = vsx_name
                best["used_vsx"]    = True
                best["is_variable"] = True
                best["offset_arcsec"] = self._vsx_last_sep   # the VSX star's, not SIMBAD's

        return best

    def _vsx_name_near(self, coord, radius_arcsec):
        """Return the closest VSX 'Name' within `radius_arcsec`, or None.
        Its separation in arcsec is left in self._vsx_last_sep."""
        self._vsx_last_sep = float("nan")
        try:
            # Deliberately NOT gated on SIMBAD health: VizieR runs on separate
            # hosts from the SIMBAD TAP endpoint, so this is a genuine
            # fallback when SIMBAD is broken -- and it fails over across the
            # VizieR mirrors, so a Strasbourg-wide outage is covered too.
            tbls, _vz_host = _vizier_query_region(
                coord, radius_arcsec * u.arcsec, "B/vsx",
                columns=["+_r", "Name", "RAJ2000", "DEJ2000"], row_limit=25,
                context="exoplanet/vsx")
        except Exception as e:
            print(f"[DEBUG] VSX region lookup failed on every VizieR mirror: {e}")
            return None
        if not tbls or len(tbls) == 0 or len(tbls[0]) == 0:
            return None
        t = tbls[0]
        try:
            ras  = np.asarray(t["RAJ2000"], dtype=float)
            decs = np.asarray(t["DEJ2000"], dtype=float)
        except Exception:
            return None
        cands = SkyCoord(ra=ras*u.deg, dec=decs*u.deg, frame="icrs")
        seps  = coord.separation(cands).arcsec
        i     = int(np.argmin(seps))
        name  = t["Name"][i]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        self._vsx_last_sep = float(seps[i])
        return str(name).strip() or None

    def _vsx_only_lookup(self, coord, radius_arcsec):
        """SIMBAD returned nothing — try VSX alone.  Same dict shape."""
        name = self._vsx_name_near(coord, radius_arcsec)
        if not name:
            return None
        return {
            "main_id":       name,
            "otype":         "V*",
            "vmag":          None,
            "offset_arcsec": self._vsx_last_sep,
            "is_variable":   True,
            "used_vsx":      True,
        }

    def _gaia_near(self, ra_deg, dec_deg, radius_arcsec=15.0):
        """
        Gaia DR3 sources within `radius_arcsec`, nearest first, as
        [(source_id, gmag, sep_arcsec)], plus where they came from.  The
        local astrometric library first (offline, instant -- the plate
        solver's own catalogue), else Gaia DR3 on the VizieR mirrors.
        """
        try:
            from setiastro.saspro import gaia_database as GD
            lib = GD.get_astro_library()
            if lib.installed_bands():
                hits = lib.find_stars_near(float(ra_deg), float(dec_deg),
                                           radius_arcsec=float(radius_arcsec))
                return ([(int(h.source_id), h.gmag, float(s)) for h, s in hits],
                        "local Gaia DR3 library")
        except Exception as e:
            print(f"[DEBUG] local Gaia lookup failed: {e}")
        coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
        try:
            tbls, host = _vizier_query_region(
                coord, float(radius_arcsec) * u.arcsec, "I/355/gaiadr3",
                columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag"], row_limit=25,
                context="exoplanet/gaia")
        except Exception as e:
            print(f"[DEBUG] Gaia DR3 lookup failed on every VizieR mirror: {e}")
            return [], "Gaia DR3 unreachable (no local library, VizieR mirrors down)"
        if not tbls or len(tbls) == 0 or len(tbls[0]) == 0:
            return [], f"Gaia DR3 via {host}"
        t = tbls[0]
        sc = SkyCoord(ra=np.asarray(t["RA_ICRS"], float) * u.deg,
                      dec=np.asarray(t["DE_ICRS"], float) * u.deg, frame="icrs")
        sep = coord.separation(sc).arcsec
        rows = []
        for j in np.argsort(sep):
            try:
                g = float(t["Gmag"][j])
                g = g if np.isfinite(g) else None
            except (TypeError, ValueError):
                g = None                       # masked
            rows.append((int(t["Source"][j]), g, float(sep[j])))
        return rows, f"Gaia DR3 via {host}"

    def _identify_from_gaia(self, ra, dec, idx=None):
        """
        SIMBAD and VSX gave nothing -- name the star from Gaia DR3 instead.
        Most field stars aren't in SIMBAD at all, and when SIMBAD is down
        (both mirrors, as on 2026-11-19) this still identifies the star and
        shows whether the plate solution put it where Gaia says it is.
        """
        st = cds_health.status_text()
        down = "unavailable" in st.lower()
        head = ("SIMBAD is unreachable right now (every mirror), and VSX has no "
                "variable within 15″." if down else
                "No SIMBAD object or VSX variable within 15″.")
        rows, where = self._gaia_near(ra, dec, 15.0)
        pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).to_string("hmsdms", sep=":", precision=1)
        if rows:
            sid, g, sep = rows[0]
            # Name the star only when the Gaia source is the star itself, not
            # a neighbour up to 15″ away (the plate solution is good to < 1″).
            if idx is not None and sep <= 3.0:
                self._set_star_name(idx, f"Gaia DR3 {sid}")
            lines = [head, "", f"Nearest Gaia DR3 source ({where}):",
                     f"  ID:     Gaia DR3 {sid}"]
            if g is not None:
                lines.append(f"  G mag:  {g:.2f}")
            lines.append(f"  Offset: {sep:.1f}″")
            if len(rows) > 1:
                lines.append(f"  ({len(rows) - 1} more Gaia source(s) within 15″)")
            lines += ["", f"Star position: {pos}"]
            body = "\n".join(lines)
        else:
            body = f"{head}\n\nNo Gaia DR3 source within 15″ either ({where}).\n\nStar position: {pos}"
        if down:
            body += f"\n\n{st}"
        msg = QMessageBox(self)
        msg.setWindowTitle("Identify Star")
        msg.setText(body)
        sim_btn = msg.addButton("Open in SIMBAD", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Ok)
        msg.exec()
        if msg.clickedButton() == sim_btn:
            webbrowser.open(cds_health.simbad_web_url(
                "sim-coo?"
                f"Coord={ra:+.6f}{dec:+.6f}&CooFrame=ICRS&CooEpoch=2000"
                "&CooEqui=2000&Radius=15&Radius.unit=arcsec&submit=submit+query"))

    # ---- UI: Identify Star (uses the shared resolver) -----------------
    def on_identify_star(self):
        radec = self.get_selected_star_radec()
        if radec is None:
            QMessageBox.warning(self, "Identify Star", "Please select exactly one star first.")
            return
        ra, dec = radec
        sel = self.star_list.selectedItems()
        idx = sel[0].data(Qt.ItemDataRole.UserRole) if len(sel) == 1 else None

        hit = self._resolve_target_name_row(ra, dec, radius_arcsec=15.0)
        if hit is None:
            self._identify_from_gaia(ra, dec, idx)
            return

        main_id  = hit["main_id"]
        obj_type = hit["otype"] or "n/a"
        vmag     = hit["vmag"]
        offset   = hit["offset_arcsec"]
        # The plots use the name from now on -- unless the match is a
        # neighbour (the lookup prefers a variable anywhere within 15″).
        if idx is not None and np.isfinite(offset) and offset <= 5.0:
            self._set_star_name(idx, main_id)
        vmag_str = f"{vmag:.3f}" if vmag is not None else "n/a"
        off_str  = f"{offset:.2f}″" if np.isfinite(offset) else "n/a"
        tag      = " (VSX)" if hit["used_vsx"] else ""

        simbad_url = cds_health.simbad_web_url(f"sim-id?Ident={quote(main_id)}")
        msg = QMessageBox(self)
        msg.setWindowTitle("SIMBAD/VSX Lookup")
        msg.setText(
            f"Best match{tag}:\n"
            f"  ID:     {main_id}\n"
            f"  Type:   {obj_type}\n"
            f"  V mag:  {vmag_str}\n"
            f"  Offset: {off_str}"
        )
        open_btn = msg.addButton("Open in SIMBAD", QMessageBox.ButtonRole.ActionRole)
        ok_btn   = msg.addButton(QMessageBox.StandardButton.Ok)
        msg.exec()
        if msg.clickedButton() == open_btn:
            webbrowser.open(simbad_url)

    # ---- UI: Find Star by name ----------------------------------------
    #
    # David's request #1: instead of star-hopping from WIMI to locate a
    # named target, let the user type the name.  We resolve via SIMBAD,
    #sasproject through the WCS, and single-select the closest detected
    # star in the list (which triggers the existing overlay highlight and
    # light-curve redraw).  Tolerance is generous — max(2·median FWHM,
    # 10″) — to cover slight WCS residuals; if nothing detected is close
    # enough we report the resolved pixel so the user can see whether SEP
    # dropped it (faint, on the border, saturated).

    def on_find_star_by_name(self):
        if not self.star_positions:
            QMessageBox.warning(
                self, "Find Star",
                "Run photometry first so there are detected stars to search among."
            )
            return
        if self._wcs is None:
            QMessageBox.warning(
                self, "Find Star",
                "No WCS available — plate-solve during photometry first."
            )
            return

        name, ok = QInputDialog.getText(
            self, "Find Star by Name",
            "Enter a SIMBAD-resolvable name\n"
            "(e.g. V336 Aps, HD 12345, TYC 9256-617-1, Gaia DR3 …):",
            QLineEdit.EchoMode.Normal, ""
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # 1) resolve name -> RA/Dec: SIMBAD (with mirror failover), then
        #    Sesame across its mirrors.
        #
        # Name resolution doesn't need the SIMBAD TAP service: Sesame is a
        # separate CDS service on cds.unistra.fr and the VizieR mirrors, and
        # it kept resolving names through the SIMBAD outage that has had both
        # SIMBAD TAP mirrors down.
        ra_deg = dec_deg = None
        simbad_said_unknown = False
        if cds_health.simbad_available():
            tbl = None
            for attempt in range(1, 3):
                try:
                    custom = _new_simbad()
                    tbl = custom.query_object(name)
                    cds_health.note_success()
                    break
                except Exception as e:
                    print(f"[DEBUG] SIMBAD name resolve attempt {attempt} failed: {e}")
                    if cds_health.note_failure(e, context="exoplanet/query_object") \
                            or attempt == 2:
                        tbl = None
                        break
                    non_blocking_sleep(1)
            if tbl is not None and len(tbl) == 0:
                simbad_said_unknown = True
            elif tbl is not None:
                try:
                    ra_col  = next(c for c in tbl.colnames if c.lower() == "ra")
                    dec_col = next(c for c in tbl.colnames if c.lower() == "dec")
                    try:
                        ra_deg  = float(tbl[0][ra_col])
                        dec_deg = float(tbl[0][dec_col])
                    except (TypeError, ValueError):
                        # Older astroquery returns sexagesimal strings
                        sc = SkyCoord(str(tbl[0][ra_col]), str(tbl[0][dec_col]),
                                      unit=(u.hourangle, u.deg), frame="icrs")
                        ra_deg  = float(sc.ra.deg)
                        dec_deg = float(sc.dec.deg)
                except Exception as e:
                    print(f"[DEBUG] SIMBAD response unusable ({e}) — trying Sesame")
                    ra_deg = dec_deg = None

        if ra_deg is None:
            self.status_label.setText(f"Resolving “{name}” via Sesame mirrors…")
            QApplication.processEvents()
            hit = cds_health.sesame_resolve(
                name, status_cb=lambda m: (self.status_label.setText(f"Sesame: {m}"),
                                           QApplication.processEvents()))
            if hit is not None:
                ra_deg, dec_deg, _url = hit
                self.status_label.setText(f"“{name}” resolved via {_url.split('/')[2]}")
            elif simbad_said_unknown or cds_health.sesame_resolve.last == "not found":
                QMessageBox.information(
                    self, "Not Found",
                    f"Neither SIMBAD nor Sesame resolved “{name}”.\n"
                    "Try a fuller catalog form (e.g. “V* V336 Aps”, “TYC 9256-617-1”)."
                )
                return
            else:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Warning)
                box.setWindowTitle("Name Resolution Unavailable")
                box.setText("No SIMBAD or Sesame mirror answered.")
                box.setInformativeText(
                    "Every SIMBAD mirror and every Sesame mirror was tried. You can "
                    "still select the star directly in the image.\n\n"
                    "CDS status: https://cds.unistra.fr"
                )
                box.setDetailedText(
                    f"{cds_health.status_text()}\n\nSIMBAD mirrors:\n"
                    f"{cds_health.status_detail()}\n\nSesame: "
                    f"{cds_health.sesame_resolve.last}"
                )
                retry = box.addButton("Re-test mirrors",
                                      QMessageBox.ButtonRole.ActionRole)
                box.addButton(QMessageBox.StandardButton.Ok)
                box.exec()
                if box.clickedButton() is retry:
                    cds_health.reset()
                    ok_probe, srv, _d = cds_health.probe(force=True)
                    if ok_probe:
                        QMessageBox.information(
                            self, "SIMBAD",
                            f"SIMBAD is reachable via {srv}. Try the lookup again.")
                return

        # 2) sky -> detector pixel coordinates
        sky = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
        try:
            xw, yw = self._wcs.world_to_pixel(sky)
            xw = float(xw); yw = float(yw)
        except Exception as e:
            QMessageBox.warning(self, "Find Star", f"WCS reverse projection failed:\n{e}")
            return

        bin_factor = getattr(self, "_wcs_bin_factor", 1) or 1
        xd = xw / bin_factor
        yd = yw / bin_factor

        # 3) find nearest detected star
        xs = np.array([p[0] for p in self.star_positions], dtype=float)
        ys = np.array([p[1] for p in self.star_positions], dtype=float)
        d = np.hypot(xs - xd, ys - yd)
        i_near = int(np.argmin(d))
        d_near = float(d[i_near])

        # tolerance: max(2·median FWHM in detector px, 10″/px converted, 6px floor)
        tol_px = max(2.0 * float(self.median_fwhm or 3.0), 6.0)
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales
            scales = proj_plane_pixel_scales(self._wcs) * 3600.0  # deg -> arcsec
            px_scale = float(np.mean(scales)) / float(bin_factor)
            if px_scale > 0:
                tol_px = max(tol_px, 10.0 / px_scale)
        except Exception:
            pass

        if d_near > tol_px:
            QMessageBox.information(
                self, "Find Star",
                f"“{name}” resolves to RA/Dec ({ra_deg:.6f}, {dec_deg:.6f})\n"
                f"= detector pixel ({xd:.1f}, {yd:.1f}), but the closest\n"
                f"detected star (#{i_near}) is {d_near:.1f} px away — beyond\n"
                f"the {tol_px:.1f} px tolerance.\n\n"
                "The star may have been rejected by SEP (too faint / near\n"
                "the border) or the frame doesn't cover it."
            )
            return

        # 4) single-select the matching list item and scroll into view; the
        #    name the user typed labels the star in the plots from now on
        self._set_star_name(i_near, name)
        for row in range(self.star_list.count()):
            it = self.star_list.item(row)
            if int(it.data(Qt.ItemDataRole.UserRole)) == i_near:
                self.star_list.clearSelection()
                self.star_list.setCurrentItem(it)
                self.star_list.scrollToItem(
                    it, QAbstractItemView.ScrollHint.PositionAtCenter
                )
                break

    def _query_simbad_main_id(self):
        """Kept for API compatibility (called by export_to_aavso).  Now
        thin — the real work is in _resolve_target_name_row."""
        radec = self.get_selected_star_radec()
        if radec is None:
            return None
        hit = self._resolve_target_name_row(radec[0], radec[1], radius_arcsec=15.0)
        if hit is None:
            return None
        return hit["main_id"]

    def _query_simbad_name_and_vmag(self, ra_deg, dec_deg, radius=5*u.arcsec):
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame="icrs")
        table = None

        # Called from the AAVSO export path, where SIMBAD only supplies a
        # suggested STARID and V magnitude -- both of which the user can
        # type. Returning (None, None) quickly leaves the export usable;
        # freezing for minutes does not. So no modal error dialog here on
        # an outage: the caller already prompts for the name.
        if not cds_health.simbad_available():
            print(f"[DEBUG] SIMBAD skipped for name/vmag — {cds_health.status_text()}"
                  " — trying UCAC4 on the VizieR mirrors")
            return self._ucac4_name_and_vmag(coord, radius)

        for attempt in range(1, 3):
            try:
                custom = _new_simbad()
                custom.add_votable_fields("otype","flux(V)")
                table = custom.query_region(coord, radius=radius)
                cds_health.note_success()
                break
            except Exception as e:
                gave_up = cds_health.note_failure(e, context="exoplanet/name_and_vmag")
                if gave_up or attempt == 2:
                    print(f"[DEBUG] SIMBAD name/vmag failed ({e}) — trying TAP directly")
                    table = _simbad_region_direct(
                        ra_deg, dec_deg, float(u.Quantity(radius).to_value(u.arcsec)))
                    break
                non_blocking_sleep(1)

        if table is None or len(table)==0:
            return self._ucac4_name_and_vmag(coord, radius)
        try:
            id_col = next(c for c in table.colnames if c.lower()=="main_id")
        except StopIteration:
            return None, None
        raw_id = table[0][id_col]
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode()
        v_col = next((c for c in table.colnames if c.upper() in ("FLUX_V","V")), None)
        vmag = None
        if v_col:
            try:
                v = float(table[0][v_col])
                if np.isfinite(v): vmag = v
            except Exception:
                vmag = None
        return raw_id, vmag

    def _check_star_candidates(self, idx: int, max_n: int = 10) -> list:
        """
        Stars to try, best first, as the AAVSO check star for target `idx`.

        AAVSO wants the check star constant, close to the target in
        brightness, near it in the field, and not one of the comparison
        stars (the export's comparison pool already excludes whichever star
        is picked here).  So: drop the target, flagged variables, anything
        saturated in any frame or unusable in more than 10% of frames, and
        stars whose own light curve shows smooth variability; rank the rest
        by instrumental-magnitude difference from the target, with distance
        as a lighter second term.

        This list used to come from ensemble_map[idx]. That held the k stars
        nearest the target in FLUX, so its members were brightness-matched by
        construction. Since the weighted ensemble it holds the most STABLE
        comparison stars -- usually the brightest in the field -- which is a
        poor check on a faint variable's photometry.
        """
        n = len(self.star_positions)
        fallback = [m for m in self.ensemble_map.get(idx, []) if 0 <= m < n][:max_n]
        F = getattr(self, "raw_flux", None)
        if F is None or not (0 <= idx < n) or F.shape[0] != n:
            return fallback
        med = np.full(n, np.nan)
        for i in range(n):
            row = F[i]
            ok = np.isfinite(row) & (row > 0)
            if np.count_nonzero(ok) >= 3:
                med[i] = float(np.median(row[ok]))
        if not (np.isfinite(med[idx]) and med[idx] > 0):
            return fallback
        flags = getattr(self, "flags", None)
        flagged = set(getattr(self, "flagged_stars", set()) or set())
        xy = np.asarray(self.star_positions, dtype=np.float64)
        diag = float(np.hypot(*(xy.max(axis=0) - xy.min(axis=0)))) or 1.0
        scored = []
        for i in range(n):
            if i == idx or i in flagged or not (np.isfinite(med[i]) and med[i] > 0):
                continue
            if flags is not None and i < flags.shape[0]:
                if np.any((flags[i] & xphot.FLAG_SAT) != 0):
                    continue
                if np.mean(flags[i] == 0) < 0.9:
                    continue
            m = self.screen.get(i, {}) if isinstance(self.screen, dict) else {}
            if float(m.get("eta_z", 0.0) or 0.0) >= 5.0:
                continue
            dmag = abs(-2.5 * math.log10(med[i] / med[idx]))
            dist = float(np.hypot(*(xy[i] - xy[idx]))) / diag
            scored.append((dmag + 0.5 * dist, i))
        scored.sort()
        out = [i for _s, i in scored[:max_n]]
        return out or fallback

    def _ucac4_name_and_vmag(self, coord, radius):
        """
        (ID, V) of the nearest UCAC4 star -- the fallback when SIMBAD can't
        name a comparison/check star.  UCAC4 lives on VizieR, which fails
        over across mirrors, and its V is the APASS DR9 value, so the AAVSO
        check-star fields stay filled during a SIMBAD outage instead of
        dropping the user to manual entry.
        """
        try:
            tbls, host = _vizier_query_region(
                coord, radius, "I/322A/out",
                columns=["+_r", "UCAC4", "RAJ2000", "DEJ2000", "Vmag"], row_limit=20,
                context="exoplanet/ucac4-id", require_rows=True)
        except Exception as e:
            print(f"[DEBUG] UCAC4 name/vmag fallback failed on every VizieR mirror: {e}")
            return None, None
        if not tbls or len(tbls[0]) == 0:
            return None, None
        t = tbls[0]
        try:
            cands = SkyCoord(ra=np.asarray(t["RAJ2000"], float) * u.deg,
                             dec=np.asarray(t["DEJ2000"], float) * u.deg, frame="icrs")
            i = int(np.argmin(coord.separation(cands).arcsec))
            uid = t["UCAC4"][i]
            if isinstance(uid, bytes):
                uid = uid.decode("utf-8")
            vmag = None
            try:
                v = float(t["Vmag"][i])
                vmag = v if np.isfinite(v) else None
            except Exception:
                vmag = None
            print(f"[DEBUG] check star named from UCAC4 via {host}")
            return f"UCAC4 {str(uid).strip()}", vmag
        except Exception as e:
            print(f"[DEBUG] UCAC4 name/vmag parse failed: {e}")
            return None, None

    # ---------------- Export ----------------

    def export_data(self):
        if self.fluxes is None or self.times is None:
            QMessageBox.warning(self, "Export", "No photometry to export. Run Measure & Photometry first.")
            return
        wcs = self._wcs
        if wcs is None:
            QMessageBox.warning(self, "Export", "No WCS available. Run plate solve during photometry first.")
            return

        dlg = QFileDialog(self, "Export Light Curves")
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dlg.setNameFilters(["CSV files (*.csv)", "FITS files (*.fits)"])
        if not dlg.exec():
            return
        path = dlg.selectedFiles()[0]
        fmt  = dlg.selectedNameFilter()

        times_mjd = self.times.mjd
        n_stars   = self.fluxes.shape[0]

        xs = np.array([xy[0] for xy in self.star_positions])
        ys = np.array([xy[1] for xy in self.star_positions])
        bin_factor = getattr(self, "_wcs_bin_factor", 1)
        sky = wcs.pixel_to_world(xs * bin_factor, ys * bin_factor)

        ras = sky.ra.deg
        decs = sky.dec.deg

        errs = getattr(self, "flux_errors", None)
        n_f = self.fluxes.shape[1]
        am = np.asarray(self.airmasses, dtype=np.float64) if len(self.airmasses) == n_f \
            else np.full(n_f, np.nan)
        trans = np.asarray(getattr(self, "transparency", None), dtype=np.float64) \
            if getattr(self, "transparency", None) is not None else np.full(n_f, np.nan)

        if fmt.startswith("CSV") or path.lower().endswith(".csv"):
            # MJD is mid-exposure UTC.  Columns built in one go: inserting
            # hundreds of columns one at a time fragments the DataFrame.
            cols = {"MJD": times_mjd, "AIRMASS": am, "TRANSPARENCY": trans}
            for i in range(n_stars):
                cols[f"STAR_{i}"]     = self.fluxes[i]
                if errs is not None:
                    cols[f"STAR_{i}_ERR"] = errs[i]
                cols[f"FLAG_{i}"]     = self.flags[i]
                cols[f"STAR_{i}_RA"]  = np.full(n_f, ras[i])
                cols[f"STAR_{i}_DEC"] = np.full(n_f, decs[i])
            df = pd.DataFrame(cols)
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Export CSV", f"Wrote CSV →\n{path}")
            return

        hdr_out = fits.Header()
        orig_hdr = None
        if hasattr(self, "_cached_headers") and 0 <= self.ref_idx < len(self._cached_headers):
            orig_hdr = self._cached_headers[self.ref_idx]
        if isinstance(orig_hdr, fits.Header):
            for key in ("OBJECT","TELESCOP","INSTRUME","OBSERVER",
                        "DATE-OBS","EXPTIME","FILTER",
                        "CRVAL1","CRVAL2","CRPIX1","CRPIX2",
                        "CDELT1","CDELT2","CTYPE1","CTYPE2"):
                if key in orig_hdr:
                    hdr_out[key] = orig_hdr[key]
        elif isinstance(orig_hdr, dict):
            for key in ("OBJECT","TELESCOP","INSTRUME","DATE-OBS","EXPTIME","FILTER"):
                val = orig_hdr.get(key, [{}])[0].get("value")
                if val is not None:
                    hdr_out[key] = val

        hdr_out["SEPTHR"]  = (self.sep_threshold,   "SEP detection threshold (sigma)")
        hdr_out["BFRAC"]   = (self.border_fraction, "Border ignore fraction")
        hdr_out["REFIDX"]  = (self.ref_idx,         "Reference frame index")
        hdr_out["MEDFWHM"] = (self.median_fwhm,     "Median FWHM over frames (binned px)")
        if getattr(self, "aperture_radius", None):
            _b = int(getattr(self, "bin_factor", 2))
            hdr_out["APRAD"] = (float(self.aperture_radius), f"Aperture radius ({_b}x{_b}-binned px)")
            hdr_out["SOFTBIN"] = (_b, "Software bin of the measured frames")
        hdr_out["TIMESYS"] = ("UTC", "MJD column is mid-exposure UTC")
        hdr_out.add_history("Exported by Seti Astro Suite")

        cols = [fits.Column(name="MJD", format="D", array=times_mjd),
                fits.Column(name="AIRMASS", format="D", array=am),
                fits.Column(name="TRANSPARENCY", format="D", array=trans)]
        for i in range(n_stars):
            cols.append(fits.Column(name=f"STAR_{i}", format="E", array=self.fluxes[i]))
            cols.append(fits.Column(name=f"FLAG_{i}", format="I", array=self.flags[i]))
        lc_hdu = fits.BinTableHDU.from_columns(cols, header=hdr_out, name="LIGHTCURVE")

        star_idx = np.arange(n_stars, dtype=int)
        cols2 = [
            fits.Column(name="INDEX", format="I", array=star_idx),
            fits.Column(name="X",     format="E", array=xs),
            fits.Column(name="Y",     format="E", array=ys),
            fits.Column(name="RA",    format="D", array=ras),
            fits.Column(name="DEC",   format="D", array=decs),
        ]
        stars_hdu = fits.BinTableHDU.from_columns(cols2, name="STARS")

        primary = fits.PrimaryHDU(header=hdr_out)
        hdul    = fits.HDUList([primary, lc_hdu, stars_hdu])
        hdul.writeto(path, overwrite=True)
        QMessageBox.information(self, "Export FITS", f"Wrote FITS →\n{path}")

    def estimate_airmass_from_altitude(self, alt_deg):
        alt_rad = np.deg2rad(np.clip(alt_deg, 0.1, 90.0))
        return 1.0 / np.sin(alt_rad)

    # ------------------------------------------------------------------
    # Per-channel (R/G/B) supplementary aperture photometry for OSC inputs
    # ------------------------------------------------------------------
    def _run_multiband_photometry(self, xs, ys, keep_mask=None):
        """Second-pass aperture photometry, done per channel on color inputs.

        Uses the per-frame TRACKED positions from the main pass and its
        chosen aperture radius, with the same sigma-clipped annulus sky and
        edge/saturation flags (flagged measurements are NaN).  Sets
        self.raw_flux_rgb / self.raw_flux_err_rgb
        shaped (3, n_stars, n_frames).  For mono inputs, or if anything
        fails, both attributes are set to None and the exporter falls
        through to the existing single-band path.
        """
        import sep
        from astropy.stats import sigma_clipped_stats

        self.raw_flux_rgb     = None
        self.raw_flux_err_rgb = None

        images = getattr(self, "_cached_images", None) or []
        if not images:
            return
        # Any frame with three planes counts as color for this dataset.
        is_color = any(getattr(im, "ndim", 2) == 3 and im.shape[2] >= 3 for im in images)
        if not is_color:
            return

        n_frames = len(images)
        # Positions: (n_stars,) reference positions, or (n_stars, n_frames)
        # per-frame tracked positions from the main photometry pass.
        try:
            X = np.asarray(xs, dtype=np.float64)
            Y = np.asarray(ys, dtype=np.float64)
        except Exception:
            return
        if X.ndim == 1:
            X = np.repeat(X[:, None], n_frames, axis=1)
            Y = np.repeat(Y[:, None], n_frames, axis=1)
        n_stars = int(X.shape[0])
        if n_stars <= 0 or X.shape[1] != n_frames:
            return
        # A "colour" set whose three channels are identical is mono stored as
        # RGB; per-channel magnitudes from it would be three copies of one.
        first = next((im for im in images if im is not None and getattr(im, "ndim", 2) == 3), None)
        if (first is not None
                and np.array_equal(first[..., 0], first[..., 1])
                and np.array_equal(first[..., 1], first[..., 2])):
            return

        fwhm = float(self.median_fwhm) if self.median_fwhm else 3.0
        fwhm = max(1.5, fwhm)
        r_ap = float(getattr(self, "aperture_radius", 0.0) or max(2.5, 1.5 * fwhm))
        r_in = r_ap + max(2.0, 0.5 * fwhm)
        ap = xphot.ApertureSetup(radii=np.array([r_ap]), r_in=r_in,
                                 r_out=r_in + max(5.0, 2.5 * fwhm), fwhm=fwhm,
                                 recenter=False)

        raw_flux_rgb     = np.full((3, n_stars, n_frames), np.nan, dtype=np.float32)
        raw_flux_err_rgb = np.full((3, n_stars, n_frames), np.nan, dtype=np.float32)

        prev_status = self.status_label.text() if hasattr(self, "status_label") else ""
        try:
            if hasattr(self, "status_label"):
                self.status_label.setText("Per-channel photometry (R/G/B)…")
            if hasattr(self, "progress_bar"):
                self.progress_bar.setMaximum(n_frames)
                self.progress_bar.setValue(0)
            _mb_gain = self._photometry_gain()
            sats = getattr(self, "_cached_satcoords", None) or []

            for t, img in enumerate(images):
                if img is None or getattr(img, "ndim", 2) != 3 or img.shape[2] < 3:
                    continue  # frame is mono in an otherwise-color set; leave NaN
                sel = np.isfinite(X[:, t]) & np.isfinite(Y[:, t])
                if not np.any(sel):
                    continue
                for c in range(3):
                    plane = np.ascontiguousarray(img[..., c], dtype=np.float32)
                    try:
                        m = xphot.measure_frame(plane, X[sel, t], Y[sel, t], ap,
                                                gain=_mb_gain,
                                                sat_xy=sats[t] if t < len(sats) else None)
                    except Exception:
                        continue
                    f = np.asarray(m["flux"][0], dtype=np.float64)
                    e = np.asarray(m["err"][0], dtype=np.float64)
                    bad = (m["flags"] != 0) | ~np.isfinite(f) | (f <= 0)
                    f[bad] = np.nan
                    e[bad] = np.nan
                    raw_flux_rgb[c, sel, t]     = f.astype(np.float32)
                    raw_flux_err_rgb[c, sel, t] = e.astype(np.float32)

                if hasattr(self, "progress_bar") and (t % 4 == 0 or t == n_frames - 1):
                    self.progress_bar.setValue(t + 1)
                    QApplication.processEvents()
        finally:
            if hasattr(self, "status_label"):
                self.status_label.setText(prev_status or "Ready")

        # Only publish if at least ONE channel got a finite measurement
        # somewhere (guards against a broken color pass yielding all-NaN).
        if np.any(np.isfinite(raw_flux_rgb)):
            self.raw_flux_rgb     = raw_flux_rgb
            self.raw_flux_err_rgb = raw_flux_err_rgb

    # ------------------------------------------------------------------
    # Catalog-lookup failure reporting.
    #
    # The APASS/UCAC4 tiers below used to swallow every exception with a
    # bare `except: pass`. A total VizieR outage was therefore
    # indistinguishable from a genuinely sparse field: the ensemble just
    # quietly became a single-comparison measurement with nothing anywhere
    # saying why -- which is exactly how the CDS outage presented. These
    # record the reason so the export can put it in front of the user AND
    # in the AAVSO NOTES field.
    # ------------------------------------------------------------------
    def _note_catalog_failure(self, source, err):
        """Record (and log) one failed catalog tier. Never raises."""
        try:
            _first = str(err).splitlines()[0] if str(err) else ""
            msg = f"{source}: {type(err).__name__}: {_first}".strip().rstrip(":")
        except Exception:
            msg = f"{source}: lookup failed"
        try:
            if getattr(self, "_catalog_errors", None) is None:
                self._catalog_errors = []
            if msg not in self._catalog_errors:
                self._catalog_errors.append(msg)
            print(f"[DEBUG] catalog tier failed -- {msg}")
        except Exception:
            pass

    def _catalog_failure_note(self, prefix="; catalog lookup failed: "):
        """One-line summary for the AAVSO NOTES field, or "" if all tiers
        answered. Commas are stripped -- NOTES is comma-delimited."""
        errs = getattr(self, "_catalog_errors", None) or []
        if not errs:
            return ""
        return (prefix + "; ".join(errs[:3])).replace(",", ";")

    def _warn_catalog_unavailable(self, n_resolved, n_tried):
        """Say that the ensemble is being skipped because the catalog lookup
        FAILED, not because the field is thin. Fires at most once per export,
        and only when a tier actually errored."""
        errs = getattr(self, "_catalog_errors", None) or []
        if not errs or getattr(self, "_catalog_warned", False):
            return
        self._catalog_warned = True
        try:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Comparison Catalog Unavailable")
            box.setText(
                f"Only {n_resolved} of {n_tried} field stars got a catalog "
                f"magnitude, so the ensemble zero point (needs 3) will be "
                f"skipped and this export falls back to a single comparison "
                f"star."
            )
            box.setInformativeText(
                "This is a catalog lookup failure, not a thin field: APASS "
                "and/or UCAC4 could not be reached.\n\n"
                "Try another mirror under the wrench \u2192 CDS / SIMBAD "
                "\u2192 VizieR mirror. During a Strasbourg outage the "
                "mirrors still serve the viz-bin interface this tool uses; "
                "only VizieR's TAP service is unmirrored.\n\n"
                "Offline alternative: open this frame as a document and run "
                "the Magnitude tool's Step 1 (Fetch Catalog Stars). That "
                "caches comparison magnitudes from the local Gaia XP "
                "library and needs no network at all."
            )
            box.setDetailedText("\n".join(errs))
            box.exec()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-channel comparison-star catalog mags (extends the V-only helper).
    # Returns {member_index: {"V": float|None, "B": float|None, "R": float|None,
    #                         "R_is_sloan": bool, "source": str}}
    # ------------------------------------------------------------------
    def _catalog_bvr_mags_for_members(self, members, wcs):
        """Fetch Johnson B, V and an R-band anchor (Cousins Rc if available,
        else Sloan r') for as many ensemble members as possible.  Same
        3-tier resolution as _catalog_vmags_for_members: MagTool cache →
        APASS DR9 → UCAC4.  R_is_sloan flags when the R value is really
        Sloan r' (for AAVSO NOTES honesty)."""
        import numpy as _np
        out = {}
        self._catalog_errors = []
        self._catalog_warned = False
        members = [int(m) for m in members if 0 <= int(m) < len(self.star_positions)]
        if not members:
            return out
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
        except Exception:
            return out

        bf = getattr(self, "_wcs_bin_factor", 1)
        mras, mdecs, mkeys = [], [], []
        for m in members:
            try:
                x, y = self.star_positions[m]
                sky = wcs.pixel_to_world(x * bf, y * bf)
                mras.append(float(sky.ra.deg))
                mdecs.append(float(sky.dec.deg))
                mkeys.append(m)
            except Exception:
                continue
        if not mkeys:
            return out
        member_sc = SkyCoord(_np.asarray(mras) * u.deg, _np.asarray(mdecs) * u.deg)
        TOL = 3.0 * u.arcsec

        def _to_float_nan(arr):
            try:
                mm = _np.ma.masked_invalid(_np.ma.asarray(arr, dtype=float))
                return _np.ma.filled(mm, _np.nan)
            except Exception:
                return _np.array([_np.nan if v is None else float(v) for v in arr],
                                 dtype=float)

        def _apply_catalog(cat_ra, cat_dec, cat_v, cat_b, cat_r, r_is_sloan, source):
            cra = _to_float_nan(cat_ra); cde = _to_float_nan(cat_dec)
            cv  = _to_float_nan(cat_v);  cb  = _to_float_nan(cat_b); cr = _to_float_nan(cat_r)
            good = _np.isfinite(cra) & _np.isfinite(cde) & (
                _np.isfinite(cv) | _np.isfinite(cb) | _np.isfinite(cr)
            )
            if not _np.any(good):
                return
            cat_sc = SkyCoord(cra[good] * u.deg, cde[good] * u.deg)
            cv, cb, cr = cv[good], cb[good], cr[good]
            todo = [i for i, m in enumerate(mkeys) if m not in out]
            if not todo:
                return
            try:
                idx_cat, sep2d, _ = member_sc[todo].match_to_catalog_sky(cat_sc)
            except Exception:
                return
            for t, ic, sp in zip(todo, _np.atleast_1d(idx_cat), _np.atleast_1d(sep2d)):
                if sp <= TOL:
                    i = int(ic)
                    entry = {
                        "V": (float(cv[i]) if _np.isfinite(cv[i]) else None),
                        "B": (float(cb[i]) if _np.isfinite(cb[i]) else None),
                        "R": (float(cr[i]) if _np.isfinite(cr[i]) else None),
                        "R_is_sloan": bool(r_is_sloan),
                        "source": source,
                    }
                    # Only accept if at least one of V/B/R survived
                    if entry["V"] is not None or entry["B"] is not None or entry["R"] is not None:
                        out[mkeys[t]] = entry

        # 1) Magnitude tool's cached catalog (no network) — expect V (± B/R)
        try:
            from setiastro.saspro.plate_solver import _active_doc_from_parent
            doc = _active_doc_from_parent(self.parent())
            meta = getattr(doc, "metadata", {}) or {}
            cat = meta.get("SFCC_star_list") or []
            if cat:
                _apply_catalog(
                    [c.get("ra") for c in cat],
                    [c.get("dec") for c in cat],
                    [c.get("Vmag") for c in cat],
                    [c.get("Bmag") for c in cat],
                    [c.get("Rmag") for c in cat],
                    r_is_sloan=False, source="MagTool",
                )
        except Exception:
            pass

        def _cone():
            c0 = SkyCoord(_np.mean(mras) * u.deg, _np.mean(mdecs) * u.deg)
            rad = c0.separation(member_sc).max() * 1.15 + 30 * u.arcsec
            return c0, rad

        # 2) APASS DR9 — Vmag, Bmag, r'mag (Sloan) → R proxy
        if any(m not in out for m in mkeys):
            try:
                c0, rad = _cone()
                res, _vzh = _vizier_query_region(
                    c0, rad, "II/336/apass9",
                    columns=["RAJ2000", "DEJ2000", "Vmag", "Bmag", "r_mag", "r\'mag"],
                    row_limit=50000, context="exoplanet/apass9", require_rows=True)
                if res:
                    tbl = res[0]
                    # APASS r'-band column is spelled differently across astroquery
                    # versions ("r_mag", "r'mag", or "rmag") — try each.
                    r_col = None
                    for cand in ("r_mag", "r'mag", "rmag", "rpmag"):
                        if cand in tbl.colnames:
                            r_col = cand; break
                    rvals = tbl[r_col] if r_col else [None] * len(tbl)
                    _apply_catalog(
                        tbl["RAJ2000"], tbl["DEJ2000"],
                        tbl["Vmag"], tbl["Bmag"], rvals,
                        r_is_sloan=True, source="APASS9",
                    )
                else:
                    # VizieR mirrors are partial: say which one came back empty.
                    self._note_catalog_failure(
                        "APASS9", RuntimeError(f"no rows from VizieR mirror {_vzh}"))
            except Exception as _e:
                self._note_catalog_failure("APASS9", _e)

        # 3) UCAC4 — Vmag (APASS-derived), Bmag, rmag (Sloan) → R proxy
        if any(m not in out for m in mkeys):
            try:
                c0, rad = _cone()
                res, _vzh = _vizier_query_region(
                    c0, rad, "I/322A/out",
                    columns=["RAJ2000", "DEJ2000", "Vmag", "Bmag", "rmag"],
                    row_limit=50000, context="exoplanet/ucac4", require_rows=True)
                if res:
                    tbl = res[0]
                    r_col = "rmag" if "rmag" in tbl.colnames else None
                    rvals = tbl[r_col] if r_col else [None] * len(tbl)
                    _apply_catalog(
                        tbl["RAJ2000"], tbl["DEJ2000"],
                        tbl["Vmag"], tbl["Bmag"], rvals,
                        r_is_sloan=True, source="UCAC4",
                    )
                else:
                    self._note_catalog_failure(
                        "UCAC4", RuntimeError(f"no rows from VizieR mirror {_vzh}"))
            except Exception as _e:
                self._note_catalog_failure("UCAC4", _e)

        return out

    # ------------------------------------------------------------------
    # Multi-band AAVSO export path (TB + TG + TR interleaved in one file).
    # ------------------------------------------------------------------
    def _export_aavso_multiband(self, idx):
        """AAVSO Extended export using per-channel raw fluxes.  Reuses
        photometry_aavso.build_rgb_rows + format_aavso_extended for the
        row/file plumbing so both tools stay in sync.  Falls back to
        single-comparison calibration on a per-channel basis when the
        catalog can't cover that band."""
        wcs = self._wcs
        if wcs is None:
            QMessageBox.warning(self, "Export AAVSO", "No WCS available.")
            return
        if self.raw_flux_rgb is None or self.raw_flux_err_rgb is None:
            QMessageBox.warning(self, "Export AAVSO",
                                "No per-channel photometry available.")
            return

        # -- STARID ---------------------------------------------------------
        star_id = self._query_simbad_main_id()
        # Resolve the VSX designation by COORDINATE, not by name. VSX has
        # no reverse index on a tagged SIMBAD id ("V* V336 Aps") or on a
        # Gaia DR3 number, which between them cover most targets -- so the
        # old query_object() call returned nothing and the unusable SIMBAD
        # spelling went into the file. Whatever we settle on is stripped of
        # its object-type tag, because AAVSO rejects a STARID carrying one.
        _radec_for_id = self.get_selected_star_radec()
        _vsx_id = None
        if _radec_for_id is not None:
            try:
                _c_for_id = SkyCoord(ra=_radec_for_id[0] * u.deg,
                                     dec=_radec_for_id[1] * u.deg,
                                     frame="icrs")
                _vsx_id = self._vsx_name_near(_c_for_id, 15.0)
            except Exception as e:
                print(f"[DEBUG] VSX coordinate lookup failed: {e}")
        star_id = _aavso_starid(_vsx_id or star_id or "")
        star_id, ok = QInputDialog.getText(
            self, "Target Star Name",
            "AAVSO STARID for this target:",
            QLineEdit.EchoMode.Normal, star_id or ""
        )
        if not ok or not star_id.strip():
            return
        star_id = star_id.strip()

        # -- Observer code + systematic floor -------------------------------
        settings = QSettings()
        prev_code = settings.value("AAVSO/observer_code", "", type=str)
        code, ok = QInputDialog.getText(
            self, "Observer Code", "Enter your AAVSO observer code:",
            QLineEdit.EchoMode.Normal, prev_code
        )
        if not ok:
            return
        code = code.strip().upper()
        settings.setValue("AAVSO/observer_code", code)

        prev_floor = float(settings.value("AAVSO/sys_floor_mag", 0.10, type=float))
        sys_floor, ok = QInputDialog.getDouble(
            self, "Systematic Floor",
            "Systematic error floor (mag) added in quadrature to MERR.\n"
            "  0.10 = defensive default;  0.05 ≈ PixInsight;  0.00 disables.",
            value=prev_floor, min=0.00, max=0.30, decimals=3
        )
        if not ok:
            return
        settings.setValue("AAVSO/sys_floor_mag", sys_floor)

        # -- Check star ------------------------------------------------------
        # Brightness-matched, constant, nearby -- see _check_star_candidates.
        members = self._check_star_candidates(idx)

        # Check star: first candidate SIMBAD (or UCAC4) can name with a V mag.
        kname = None; k_idx = None; kstar_catalog = None
        for m in members:
            x, y = self.star_positions[m]
            bf = getattr(self, "_wcs_bin_factor", 1)
            sky = wcs.pixel_to_world(x * bf, y * bf)
            name, v = self._query_simbad_name_and_vmag(sky.ra.deg, sky.dec.deg)
            if name and (v is not None) and np.isfinite(v):
                kname, k_idx = _aavso_starid(name), m
                kstar_catalog = {"Vmag": float(v), "Bmag": None, "Rmag": None}
                break
        if kname is None:
            kname, ok = QInputDialog.getText(
                self, "Check Star Name",
                "Could not auto-identify a check star. Enter check-star ID:"
            )
            if not ok or not kname.strip():
                return
            kname = kname.strip()
            k_idx, ok = QInputDialog.getInt(
                self, "Check Star",
                "Enter the list index (#) of the check star used for calibration:",
                0, 0, max(0, len(self.star_positions) - 1)
            )
            if not ok:
                return
            kstar_catalog = {"Vmag": None, "Bmag": None, "Rmag": None}

        # Per-channel KMAGs: use catalog for check star, then fill missing.
        kmags = aav.check_star_kmags(kstar_catalog)   # {"R":..,"G":..,"B":..}
        #sasprompt default = V-mag when we have it. Better than 0 (which
        # gives MAG=0-ish outputs if the user hits OK without editing —
        # David saw exactly this on TB/TR for V336 Aps). For most stars
        # V is within a magnitude or so of B and R, so if the user
        # doesn't know the exact value, V is a defensible fallback.
        for ch, prompt_label in (("B", "B"), ("G", "V (green anchor)"), ("R", "R")):
            if kmags.get(ch) is None:
                _v_default = float(kmags.get("G") or 0.0)
                _has_v_hint = kmags.get("G") is not None
                _hint = (
                    f"\n\nDefaulted to V-mag ({_v_default:.3f}) as a plausible"
                    f"\nfallback. Edit for accuracy or Cancel to skip this band."
                    if _has_v_hint else
                    "\n\nCancel to skip this band."
                )
                val, ok = QInputDialog.getDouble(
                    self, f"Check-Star {prompt_label} Magnitude",
                    f"Catalog {prompt_label} magnitude for {kname}\n"
                    f"(needed to calibrate the {aav.RGB_FILTER_MAP[ch][0]} channel)."
                    f"{_hint}",
                    value=(kmags.get("G") if kmags.get("G") is not None else 0.0),
                    min=-5.0, max=30.0, decimals=3
                )
                if ok:
                    kmags[ch] = float(val)
                else:
                    kmags[ch] = None

        # -- File path -----------------------------------------------------
        path, _ = QFileDialog.getSaveFileName(
            self, "Save AAVSO File (multi-band)", "",
            "Text files (*.txt *.dat *.csv)"
        )
        if not path:
            return

        # -- Per-channel ensemble ZP or single-comparison fallback ---------
        # Broadened comp pool: every detected star EXCEPT the target,
        # the check star, and anything flagged as variable / dip. This
        # is David0944's msg-6 suggestion — the tight ensemble_map[idx]
        # (nearest-flux neighbors, K≈8) is too small to reliably clear
        # the ≥3-catalog-mag bar for the ensemble tier. A bigger pool
        # gets us more comps with catalog mags; MAD clipping downstream
        # still removes outliers.
        _all_indices = list(range(len(self.star_positions)))
        _flagged     = getattr(self, "flagged_stars", None) or set()
        comp_members = [m for m in _all_indices
                        if m != idx and m != k_idx and m not in _flagged]
        cmap = self._catalog_bvr_mags_for_members(comp_members, wcs)
        if len(cmap) < 3:
            self._warn_catalog_unavailable(len(cmap), len(comp_members))

        LOGC = 2.5 / np.log(10.0)
        floor = float(sys_floor or 0.0)
        n_frames = int(self.raw_flux_rgb.shape[2])
        fn = self._frame_noise_mag(n_frames)     # per-frame excess, from the comps
        jd = self.times.utc.jd

        # For each channel produce (mags[n_frames], merr[n_frames], notes).
        # RGB_FILTER_MAP: R→("TR","R"), G→("TG","V"), B→("TB","B")
        per_ch = {}
        used_comps = set()   # comp indices that actually formed the ensemble/comparison
        r_note_extra = ""
        # Detect if any comp's R is Sloan-derived; a single flag on the note
        # is honest enough for AAVSO's NOTES field.
        if any(v.get("R") is not None and v.get("R_is_sloan") for v in cmap.values()):
            r_note_extra = "; R anchor = Sloan r' (APASS/UCAC)"

        for ch in ("B", "G", "R"):
            filt, band_key = aav.RGB_FILTER_MAP[ch]
            ci = {"R": 0, "G": 1, "B": 2}[ch]     # numpy channel index
            F_T    = self.raw_flux_rgb[ci, idx, :].astype(np.float64)
            Ferr_T = self.raw_flux_err_rgb[ci, idx, :].astype(np.float64)

            comp_idx_all = np.array(
                [m for m in comp_members if m in cmap and cmap[m].get(band_key) is not None],
                dtype=int
            )
            comp_mag = np.array(
                [cmap[m][band_key] for m in comp_idx_all], dtype=float
            )

            if comp_idx_all.size >= 3:
                Fi  = self.raw_flux_rgb[ci, comp_idx_all, :].astype(np.float64)
                okF = np.isfinite(Fi) & (Fi > 0)
                zi  = np.where(okF,
                               comp_mag[:, None] + 2.5 * np.log10(np.where(okF, Fi, 1.0)),
                               np.nan)
                # robust MAD-based sigma-clip stars on run-median ZP.
                # std is inflated by the outliers we're trying to catch;
                # MAD isn't, so a comp star with a wrong catalog mag now
                # actually gets clipped instead of masquerading as normal
                # scatter. Floor sigma at 0.05 mag so tight ensembles
                # don't over-clip themselves down to one star.
                z_star = np.nanmedian(zi, axis=1)
                keepC  = np.isfinite(z_star)
                for _ in range(3):
                    if not np.any(keepC):
                        break
                    med  = np.nanmedian(z_star[keepC])
                    mad  = np.nanmedian(np.abs(z_star[keepC] - med))
                    sd_r = max(1.4826 * float(mad), 0.05)
                    if not np.isfinite(sd_r) or sd_r == 0:
                        break
                    prev_keep = keepC.copy()
                    keepC &= np.abs(z_star - med) <= 3.0 * sd_r
                    if np.array_equal(keepC, prev_keep):
                        break  # converged; no point re-computing MAD
                zi = zi[keepC, :]
                ZP_t  = np.nanmedian(zi, axis=0)
                n_t   = np.sum(np.isfinite(zi), axis=0)
                madz  = np.nanmedian(np.abs(zi - ZP_t[None, :]), axis=0)
                SEM_t = np.where(n_t > 1, 1.4826 * madz / np.sqrt(np.maximum(n_t, 1)), np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mags = -2.5 * np.log10(F_T) + ZP_t
                    merr = np.sqrt((LOGC * Ferr_T / F_T) ** 2
                                   + np.nan_to_num(SEM_t) ** 2 + fn ** 2 + floor ** 2)
                bad = ~np.isfinite(mags) | (F_T <= 0) | (n_t < 1)
                n_ens = int(np.count_nonzero(keepC))
                note = (f"{filt}: ensemble ZP N={n_ens} vs catalog {band_key}"
                        f"; err incl SEM+{floor:.2f}floor")
                if ch == "R" and r_note_extra:
                    note += r_note_extra
                # AAVSO ENSEMBLE requires a check star, and for ensemble the
                # KMAG field is that star's magnitude standardised through the
                # SAME ensemble ZP (not its catalog value, not a raw
                # instrumental mag). Compute K exactly like the target so a data
                # user can diff it against the catalog value as a quality check.
                used_comps.update(int(m) for m in comp_idx_all[keepC])
                F_K = self.raw_flux_rgb[ci, k_idx, :].astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    _K = -2.5 * np.log10(F_K) + ZP_t
                _K = _K[np.isfinite(_K)]
                cname_ch = "ENSEMBLE"
                cmag_ch  = "na"
                kname_ch = kname
                kmag_ch  = float(np.median(_K)) if _K.size else "na"
            elif comp_idx_all.size >= 1:
                # tier 2: single-comparison vs a comp-member with catalog mag
                # in this band. Prefer the brightest (highest median flux) —
                # best SNR, most stable ZP anchor. This tier exists because
                # the previous code fell straight through to the check star,
                # which is the wrong role for photometric anchoring AND is
                # empty in the ensemble=1 case where the sole ensemble
                # member is also the check star.
                _c_meds = np.array(
                    [np.nanmedian(self.raw_flux_rgb[ci, int(mm), :])
                     for mm in comp_idx_all], dtype=float
                )
                _c_local = int(np.argmax(np.nan_to_num(_c_meds, nan=-np.inf)))
                c_idx = int(comp_idx_all[_c_local])
                c_mag = float(comp_mag[_c_local])
                F_C    = self.raw_flux_rgb[ci, c_idx, :].astype(np.float64)
                Ferr_C = self.raw_flux_err_rgb[ci, c_idx, :].astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mags  = c_mag - 2.5 * np.log10(F_T / F_C)
                    frac2 = (Ferr_T / F_T) ** 2 + (Ferr_C / F_C) ** 2
                    merr  = np.sqrt((LOGC ** 2) * frac2 + 2.0 * fn ** 2 + floor ** 2)
                bad = ~np.isfinite(mags) | (F_T <= 0) | (F_C <= 0)
                note = (f"{filt}: single-comparison vs member #{c_idx} "
                        f"({band_key}={c_mag:.3f}); err incl {floor:.2f}floor"
                        f"; ensemble not used: {comp_idx_all.size} of "
                        f"{len(comp_members)} field stars had a catalog "
                        f"{band_key} (need 3)"
                        + self._catalog_failure_note())
                if ch == "R" and r_note_extra:
                    note += r_note_extra
                # NOT an ensemble: classic differential photometry, so CNAME is
                # the comparison and CMAG/KMAG are raw INSTRUMENTAL mags (target
                # standardised to the comp's catalog mag). Keep the independent
                # check star in KNAME/KMAG.
                used_comps.add(int(c_idx))
                F_K = self.raw_flux_rgb[ci, k_idx, :].astype(np.float64)
                _fc = float(np.nanmedian(F_C[F_C > 0])) if np.any(F_C > 0) else np.nan
                _fk = float(np.nanmedian(F_K[F_K > 0])) if np.any(F_K > 0) else np.nan
                cname_ch = f"C{int(c_idx)}"
                cmag_ch  = float(-2.5 * np.log10(_fc)) if np.isfinite(_fc) else "na"
                kname_ch = kname
                kmag_ch  = float(-2.5 * np.log10(_fk)) if np.isfinite(_fk) else "na"
            elif kmags.get(ch) is not None:
                # tier 3 (last resort): calibrate against the CHECK STAR
                # itself. Only lands here when zero comp-members have a
                # catalog mag in this band. Not ideal (the check star is
                # meant for verification, not anchoring) but better than
                # skipping the band entirely.
                F_C    = self.raw_flux_rgb[ci, k_idx, :].astype(np.float64)
                Ferr_C = self.raw_flux_err_rgb[ci, k_idx, :].astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mags  = kmags[ch] - 2.5 * np.log10(F_T / F_C)
                    frac2 = (Ferr_T / F_T) ** 2 + (Ferr_C / F_C) ** 2
                    merr  = np.sqrt((LOGC ** 2) * frac2 + 2.0 * fn ** 2 + floor ** 2)
                bad = ~np.isfinite(mags) | (F_T <= 0) | (F_C <= 0)
                note = (f"{filt}: check-star fallback vs {kname}"
                        f"; err incl {floor:.2f}floor"
                        f"; no field star had a catalog {band_key} "
                        f"({len(comp_members)} tried)"
                        + self._catalog_failure_note())
                # The check star is doing double duty as the comparison
                # here, so it belongs in the comps[RA/Dec] provenance --
                # otherwise that line comes out empty and the submission
                # records nothing about what set the zero point.
                used_comps.add(int(k_idx))
                # Degenerate: the check star is doubling as the comparison, so
                # it can't also be an independent check — CNAME=check with its
                # instrumental CMAG, and KNAME/KMAG left as na.
                _fc = float(np.nanmedian(F_C[F_C > 0])) if np.any(F_C > 0) else np.nan
                cname_ch = str(kname)
                cmag_ch  = float(-2.5 * np.log10(_fc)) if np.isfinite(_fc) else "na"
                kname_ch = "na"
                kmag_ch  = "na"
            else:
                # Nothing to calibrate this channel — skip it.
                per_ch[ch] = None
                continue

            mags = np.where(bad, np.nan, mags)
            merr = np.where(bad | ~np.isfinite(merr), np.nan, merr)
            per_ch[ch] = (mags, merr, note, cname_ch, cmag_ch, kname_ch, kmag_ch)

        if not any(v is not None for v in per_ch.values()):
            QMessageBox.warning(
                self, "Export AAVSO",
                "None of the R/G/B channels could be calibrated — no catalog\n"
                "comparison stars and no check-star magnitude for any band."
            )
            return

        # AAVSO asks that the comps forming an ensemble be identified in NOTES.
        # We have no AUIDs for arbitrary field stars, so list their RA/Dec (ICRS
        # deg) — enough for a data user to recover which stars set the zero
        # point. Capped so a large ensemble doesn't blow past NOTES length.
        def _radec_label(mi):
            try:
                x, y = self.star_positions[int(mi)]
                bf = getattr(self, "_wcs_bin_factor", 1)
                sky = wcs.pixel_to_world(x * bf, y * bf)
                return f"{sky.ra.deg:.4f}{sky.dec.deg:+.4f}"
            except Exception:
                return None
        _ids = [s for s in (_radec_label(m) for m in sorted(used_comps)) if s]
        _MAX_IDS = 10
        if not _ids:
            comps_note = ""
        elif len(_ids) > _MAX_IDS:
            comps_note = ("comps[RA/Dec]: " + ", ".join(_ids[:_MAX_IDS])
                          + f", +{len(_ids) - _MAX_IDS} more")
        else:
            comps_note = "comps[RA/Dec]: " + ", ".join(_ids)

        # -- Assemble AavsoRow list frame-by-frame -------------------------
        # Track per-band skip reasons so the summary popup can tell the
        # user *why* certain bands didn't make it into the file. AAVSO
        # Extended Format requires MERR ≤ 1.0 — a single over-limit row
        # gets the whole file rejected at upload, so we drop them here.
        rows = []
        _skipped_by_filt = {}
        _na_by_filt      = {}
        _skip_reasons = set()
        _MERR_MAX = 1.0
        for j, t in enumerate(jd):
            frame_mags = {}; frame_merrs = {}
            frame_cnames = {}; frame_cmags = {}
            frame_knames = {}; frame_kmags = {}
            for ch in ("B", "G", "R"):
                pc = per_ch.get(ch)
                if pc is None:
                    continue
                filt_lbl = aav.RGB_FILTER_MAP[ch][0]
                m = float(pc[0][j]); e = float(pc[1][j])
                if not np.isfinite(m):
                    # No MAG to report — drop the row entirely.
                    _skipped_by_filt[filt_lbl] = _skipped_by_filt.get(filt_lbl, 0) + 1
                    _skip_reasons.add("non-finite MAG")
                    continue
                # Per-band comparison identity (tier-dependent: ENSEMBLE vs single).
                frame_cnames[ch] = pc[3]
                frame_cmags[ch]  = pc[4]
                frame_knames[ch] = pc[5]
                frame_kmags[ch]  = pc[6]
                if not np.isfinite(e) or e > _MERR_MAX:
                    # MAG is real but MERR is out of AAVSO range; emit
                    # the row with MERR=na (David msg-6 request) rather
                    # than dropping the observation.
                    _na_by_filt[filt_lbl] = _na_by_filt.get(filt_lbl, 0) + 1
                    frame_mags[ch]  = m
                    frame_merrs[ch] = None
                    continue
                frame_mags[ch]  = m
                frame_merrs[ch] = e
            if not frame_mags:
                continue
            am = float(np.clip(self.airmasses[j] if j < len(self.airmasses) else 1.0, 1.0, 40.0))
            # Per-frame notes: combine per-channel notes for the bands that
            # actually landed this frame, then append the comp identifiers once.
            note_bits = [per_ch[ch][2] for ch in ("B", "G", "R")
                         if per_ch.get(ch) is not None and ch in frame_mags]
            frame_note = " | ".join(note_bits)
            if comps_note:
                frame_note = f"{frame_note} || {comps_note}" if frame_note else comps_note
            new_rows = aav.build_rgb_rows(
                name=star_id, date_jd=float(t),
                mags=frame_mags, merrs=frame_merrs, amass=am,
                cnames=frame_cnames, cmags=frame_cmags,
                knames=frame_knames, kmags=frame_kmags,
                notes=frame_note,
            )
            rows.extend(new_rows)

        if not rows:
            QMessageBox.warning(
                self, "Export AAVSO",
                "No frames produced a finite calibrated magnitude in any band."
            )
            return

        # -- #RA / #DEC come from the selected star's WCS position ---------
        radec = self.get_selected_star_radec()
        if radec is None:
            QMessageBox.warning(self, "Export AAVSO",
                                "Could not determine RA/Dec of selected star.")
            return

        try:
            body = aav.format_aavso_extended(rows, obscode=code,
                                             ra_deg=radec[0], dec_deg=radec[1])
            with open(path, "w") as f:
                f.write(body)
        except Exception as e:
            QMessageBox.critical(self, "Export AAVSO", f"Failed to write file:\n{e}")
            return

        n_by_filt = {}
        for r in rows:
            n_by_filt[r.filt] = n_by_filt.get(r.filt, 0) + 1
        breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(n_by_filt.items()))

        # summary of skipped-per-band (row dropped entirely — non-finite MAG)
        # and emitted-with-MERR=na (per David msg 6 — MAG kept, error unknown)
        _skip_lines = []
        if _skipped_by_filt:
            _skip_breakdown = ", ".join(
                f"{k}:{v}" for k, v in sorted(_skipped_by_filt.items())
            )
            _reason_txt = "; ".join(sorted(_skip_reasons)) if _skip_reasons else "invalid"
            _skip_lines.append(f"Skipped: {_skip_breakdown}  ({_reason_txt})")

        # summary of emitted-with-MERR=na (per David msg 6)
        if _na_by_filt:
            _na_breakdown = ", ".join(
                f"{k}:{v}" for k, v in sorted(_na_by_filt.items())
            )
            _skip_lines.append(
                f"MERR=na: {_na_breakdown}  (row kept but MERR exceeded {_MERR_MAX:.1f})"
            )

        _body = f"Wrote {len(rows)} row(s) → {path}\n\nBy filter: {breakdown}"
        for _ln in _skip_lines:
            _body += f"\n{_ln}"
        _body += "\n\nOpen AAVSO WebObs upload page now?"

        msg = QMessageBox(self)
        msg.setWindowTitle("Export AAVSO")
        msg.setText(_body)
        yes = msg.addButton("Yes", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("No", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == yes:
            webbrowser.open("https://www.aavso.org/webobs/file")

    def _catalog_vmags_for_members(self, members, wcs):
        """Return {member_index: (Vmag, source)} for as many ensemble members as
        we can attach a Johnson V catalog magnitude to. Tries, in order:
          1) the Magnitude tool's cached catalog on the active doc
             (metadata["SFCC_star_list"]) -- real/anchored Johnson V
             (APASS / SIMBAD / Gaia-XP), so no network call when it's present;
          2) APASS DR9 (VizieR II/336/apass9) -- Johnson V directly;
          3) UCAC4 (VizieR I/322A/out) -- APASS-derived Johnson V.
        Matching is done in SKY coordinates, so it is valid even though the two
        tools may sit on different pixel grids. Never raises: on any failure a
        member is simply left unresolved and the caller falls back to the
        single check-star calibration.
        """
        import numpy as _np
        out = {}
        self._catalog_errors = []
        self._catalog_warned = False
        members = [int(m) for m in members if 0 <= int(m) < len(self.star_positions)]
        if not members:
            return out

        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
        except Exception:
            return out

        # member sky coords (respect the detection bin factor, as elsewhere)
        bf = getattr(self, "_wcs_bin_factor", 1)
        mras, mdecs, mkeys = [], [], []
        for m in members:
            try:
                x, y = self.star_positions[m]
                sky = wcs.pixel_to_world(x * bf, y * bf)
                mras.append(float(sky.ra.deg)); mdecs.append(float(sky.dec.deg)); mkeys.append(m)
            except Exception:
                continue
        if not mkeys:
            return out
        member_sc = SkyCoord(_np.asarray(mras) * u.deg, _np.asarray(mdecs) * u.deg)
        TOL = 3.0 * u.arcsec

        def _to_float_nan(arr):
            try:
                mm = _np.ma.masked_invalid(_np.ma.asarray(arr, dtype=float))
                return _np.ma.filled(mm, _np.nan)
            except Exception:
                return _np.array([_np.nan if v is None else float(v) for v in arr], dtype=float)

        def _apply_catalog(cat_ra, cat_dec, cat_v, source):
            cra = _to_float_nan(cat_ra); cde = _to_float_nan(cat_dec); cvm = _to_float_nan(cat_v)
            good = _np.isfinite(cra) & _np.isfinite(cde) & _np.isfinite(cvm)
            if not _np.any(good):
                return
            cat_sc = SkyCoord(cra[good] * u.deg, cde[good] * u.deg)
            cvm = cvm[good]
            todo = [i for i, m in enumerate(mkeys) if m not in out]
            if not todo:
                return
            try:
                idx_cat, sep2d, _ = member_sc[todo].match_to_catalog_sky(cat_sc)
            except Exception:
                return
            for t, ic, sp in zip(todo, _np.atleast_1d(idx_cat), _np.atleast_1d(sep2d)):
                if sp <= TOL:
                    out[mkeys[t]] = (float(cvm[int(ic)]), source)

        # 1) Magnitude tool's cached catalog (no network)
        try:
            from setiastro.saspro.plate_solver import _active_doc_from_parent
            doc = _active_doc_from_parent(self.parent())
            meta = getattr(doc, "metadata", {}) or {}
            cat = meta.get("SFCC_star_list") or []
            if cat:
                _apply_catalog([c.get("ra") for c in cat],
                               [c.get("dec") for c in cat],
                               [c.get("Vmag") for c in cat], "MagTool")
        except Exception:
            pass

        def _cone():
            c0 = SkyCoord(_np.mean(mras) * u.deg, _np.mean(mdecs) * u.deg)
            rad = c0.separation(member_sc).max() * 1.15 + 30 * u.arcsec
            return c0, rad

        # 2) APASS DR9  then  3) UCAC4  -- only for members still unresolved
        for cat_id, source in (("II/336/apass9", "APASS9"), ("I/322A/out", "UCAC4")):
            if not any(m not in out for m in mkeys):
                break
            try:
                c0, rad = _cone()
                res, _vzh = _vizier_query_region(
                    c0, rad, cat_id, columns=["RAJ2000", "DEJ2000", "Vmag"],
                    row_limit=50000, context=f"exoplanet/{source}", require_rows=True)
                if res:
                    tbl = res[0]
                    _apply_catalog(tbl["RAJ2000"], tbl["DEJ2000"], tbl["Vmag"], source)
                else:
                    self._note_catalog_failure(
                        source, RuntimeError(f"no rows from VizieR mirror {_vzh}"))
            except Exception as _e:
                self._note_catalog_failure(source, _e)

        return out

    def export_to_aavso(self):
        if getattr(self, "fluxes", None) is None or getattr(self, "times", None) is None:
            QMessageBox.warning(self, "Export AAVSO", "No photometry available. Run Measure & Photometry first.")
            return
        wcs = self._wcs
        if wcs is None:
            QMessageBox.warning(self, "Export AAVSO", "No WCS available. Plate-solve first.")
            return

        sels = self.star_list.selectedItems()
        if len(sels) != 1:
            QMessageBox.warning(self, "Export AAVSO", "Please select exactly one star before exporting.")
            return
        idx = sels[0].data(Qt.ItemDataRole.UserRole)

        # Offer a multi-band (TB+TG+TR) export when color photometry is on hand.
        if getattr(self, "raw_flux_rgb", None) is not None:
            reply = QMessageBox.question(
                self, "Multi-Band Export",
                "Color (R/G/B) photometry is available for this dataset.\n\n"
                "Export as multi-band AAVSO (TB + TG + TR)?\n"
                "  Yes  → one file with interleaved TB/TG/TR rows\n"
                "  No   → single-band export (choose one filter code)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                return self._export_aavso_multiband(idx)

        star_id = self._query_simbad_main_id()
        # Resolve the VSX designation by COORDINATE, not by name. VSX has
        # no reverse index on a tagged SIMBAD id ("V* V336 Aps") or on a
        # Gaia DR3 number, which between them cover most targets -- so the
        # old query_object() call returned nothing and the unusable SIMBAD
        # spelling went into the file. Whatever we settle on is stripped of
        # its object-type tag, because AAVSO rejects a STARID carrying one.
        _radec_for_id = self.get_selected_star_radec()
        _vsx_id = None
        if _radec_for_id is not None:
            try:
                _c_for_id = SkyCoord(ra=_radec_for_id[0] * u.deg,
                                     dec=_radec_for_id[1] * u.deg,
                                     frame="icrs")
                _vsx_id = self._vsx_name_near(_c_for_id, 15.0)
            except Exception as e:
                print(f"[DEBUG] VSX coordinate lookup failed: {e}")
        star_id = _aavso_starid(_vsx_id or star_id or "")
        if not star_id:
            star_id, ok = QInputDialog.getText(self, "Target Star Name", "Could not auto-identify.  Enter target star name for STARID:", QLineEdit.EchoMode.Normal, "")
            if not ok or not star_id.strip():
                return
            star_id = star_id.strip()

        if not hasattr(self, "exposure_time") or self.exposure_time is None:
            exp, ok = QInputDialog.getDouble(self, "Exposure Time", "No EXPOSURE found in headers. Please enter exposure time (s):", decimals=1)
            if not ok: return
            self.exposure_time = exp

        settings = QSettings()
        prev_code = settings.value("AAVSO/observer_code", "", type=str)
        code, ok = QInputDialog.getText(self, "Observer Code", "Enter your AAVSO observer code:", QLineEdit.EchoMode.Normal, prev_code)
        if not ok: return
        code = code.strip().upper()
        settings.setValue("AAVSO/observer_code", code)

        fmt, ok = QInputDialog.getItem(self, "AAVSO Format", "Choose submission format:", ["Variable-Star Photometry", "Exoplanet Report"], 0, False)
        if not ok: return
        if fmt == "Exoplanet Report":
            # This choice used to fall through and write the variable-star
            # format.  An exoplanet report is the product of a transit fit
            # (Tc, Rp/R*, airmass terms, BJD_TDB), so it is written there.
            QMessageBox.information(
                self, "AAVSO Exoplanet Report",
                "The exoplanet report is written from a transit fit.\n\n"
                "The Transit Fit window will open for this star: run the fit, "
                "then use its “AAVSO Report…” button.")
            self.on_fit_transit()
            return

        # Check star: brightness-matched, constant, nearby -- see
        # _check_star_candidates. Not from ensemble_map any more.
        members = self._check_star_candidates(idx)
        kname = None; kmag = None; k_idx = None
        for m in members:
            x, y = self.star_positions[m]
            bin_factor = getattr(self, "_wcs_bin_factor", 1)
            sky = wcs.pixel_to_world(x * bin_factor, y * bin_factor)

            name, v = self._query_simbad_name_and_vmag(sky.ra.deg, sky.dec.deg)
            if name and (v is not None) and np.isfinite(v):
                kname, kmag, k_idx = _aavso_starid(name), v, m
                break
        if kname is None:
            kname, ok = QInputDialog.getText(self, "Check Star Name", "Could not auto-identify a check star. Enter check-star ID:")
            if not ok or not kname.strip(): return
            kname = kname.strip()
            kmag, ok = QInputDialog.getDouble(self, "Check Star Magnitude", f"Enter catalog magnitude for {kname}:", decimals=3)
            if not ok: return

        # Calibration needs the check star's MEASURED flux row. If it wasn't
        # auto-identified above, ask which list entry it is.
        if k_idx is None:
            k_idx, ok = QInputDialog.getInt(
                self, "Check Star",
                "Enter the list index (#) of the check star used for calibration:",
                0, 0, max(0, len(self.star_positions) - 1))
            if not ok:
                return

        # Full AAVSO Extended Format band table, grouped for readability.
        # Persist the last-used code so common workflows don't have to
        # re-scroll every export.
        filt_choices = [
            # Johnson–Cousins
            "V", "B", "U", "R", "I", "RC", "IC",
            # DSLR / OSC tri-color
            "TG", "TB", "TR", "TI",
            # Unfiltered CCD (reduced to a Johnson band)
            "CV", "CR", "CBB", "CM",
            # Sloan
            "SU", "SG", "SR", "SI", "SZ",
            # Near-IR / other
            "J", "H", "K", "Y", "ZS",
            # Narrowband / misc
            "HA", "HAC", "MA", "MI", "MB", "O",
        ]
        prev_filt   = settings.value("AAVSO/last_filter", "V", type=str)
        default_idx = filt_choices.index(prev_filt) if prev_filt in filt_choices else 0
        filt, ok = QInputDialog.getItem(
            self, "Filter",
            "Select AAVSO filter code for this dataset:\n"
            "  V/B/U/R/I — Johnson–Cousins\n"
            "  TG/TB/TR/TI — DSLR / OSC tri-color channels\n"
            "  CV/CR/CBB/CM — unfiltered CCD reduced to V/R\n"
            "  SU..SZ — Sloan; HA/HAC — H-alpha; J/H/K — near-IR",
            filt_choices, default_idx, False
        )
        if not ok: return
        settings.setValue("AAVSO/last_filter", filt)

        path, _ = QFileDialog.getSaveFileName(self, "Save AAVSO File", "", "Text files (*.txt *.dat *.csv)")
        if not path: return

        header_lines = [
            "#TYPE=EXTENDED",
            f"#OBSCODE={code}",
            f"#SOFTWARE=Seti Astro Suite Pro",
            "#DELIM=,",
            "#DATE=JD",
            "#OBSTYPE=CCD",
        ]
        radec = self.get_selected_star_radec()
        if radec is None:
            QMessageBox.warning(self, "Export AAVSO", "Could not determine RA/Dec of selected star.")
            return
        c = SkyCoord(ra=radec[0]*u.deg, dec=radec[1]*u.deg, frame="icrs")
        header_lines += [
            "#RA="  + c.ra.to_string(unit=u.hour, sep=":", pad=True, precision=2),
            "#DEC=" + c.dec.to_string(unit=u.degree, sep=":", pad=True, alwayssign=True, precision=1),
        ]
        header_lines.append("#NAME,DATE,MAG,MERR,FILT,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES")

        jd = self.times.utc.jd

        # -- Calibrated magnitudes --------------------------------------------
        # Build a real PER-FRAME photometric zero point from the target's
        # ensemble comparison stars that carry catalog V magnitudes, exactly
        # like the Magnitude/Surface-Brightness tool:
        #     ZP(t)  = median_i( Vcat_i + 2.5*log10 F_i(t) )
        #     m_T(t) = -2.5*log10 F_T(t) + ZP(t)
        # Computing ZP per frame keeps it differential (transparency/airmass
        # cancel frame-to-frame), anchors it to Johnson V, and yields an honest
        # per-frame SEM. If too few comparison stars have catalog mags, fall
        # back to a single-comparison calibration against the check star.
        #
        # NOTE: untransformed V (no color term) -- each comp's ZP absorbs its
        # own color, so the star-to-star scatter (hence SEM) reflects real
        # calibration spread, which is why we sigma-clip the comp set and keep a
        # systematic floor in quadrature rather than trusting SEM alone.
        # Broadened comp pool (single-band): every detected star EXCEPT
        # the target, the check star, and anything flagged as variable.
        # See exo_v336_features.py for rationale — bigger pool → more
        # catalog-mag comps → higher chance of clearing the ≥3 bar for
        # ensemble ZP.
        _all_indices = list(range(len(self.star_positions)))
        _flagged     = getattr(self, "flagged_stars", None) or set()
        comp_members = [m for m in _all_indices
                        if m != idx and m != k_idx and m not in _flagged]
        vmap = self._catalog_vmags_for_members(comp_members, wcs)   # {idx: (V, source)}
        comp_idx = np.array([m for m in comp_members if m in vmap], dtype=int)
        comp_mag = np.array([vmap[m][0] for m in comp_idx], dtype=float)
        if comp_idx.size < 3:
            self._warn_catalog_unavailable(int(comp_idx.size), len(comp_members))

        LOGC   = 2.5 / np.log(10.0)
        floor  = float(getattr(self, "sys_floor_mag", 0.10) or 0.0)
        fn     = self._frame_noise_mag(int(self.raw_flux.shape[1]))   # per-frame excess
        F_T    = np.asarray(self.raw_flux[idx, :],     dtype=np.float64)
        Ferr_T = np.asarray(self.raw_flux_err[idx, :], dtype=np.float64)

        def _comp_radec_labels(members, maxn=10):
            """RA/Dec (ICRS deg) labels for the comps that set the ZP, for NOTES."""
            w = self._wcs
            bf = getattr(self, "_wcs_bin_factor", 1)
            out = []
            for mi in members:
                try:
                    x, y = self.star_positions[int(mi)]
                    sky = w.pixel_to_world(x * bf, y * bf)
                    out.append(f"{sky.ra.deg:.4f}{sky.dec.deg:+.4f}")
                except Exception:
                    pass
            if not out:
                return ""
            if len(out) > maxn:
                return ", ".join(out[:maxn]) + f", +{len(out) - maxn} more"
            return ", ".join(out)

        if comp_idx.size >= 3:
            Fi  = np.asarray(self.raw_flux[comp_idx, :], dtype=np.float64)   # (Ncomp, Nframe)
            okF = np.isfinite(Fi) & (Fi > 0)
            zi  = np.where(okF, comp_mag[:, None] + 2.5 * np.log10(np.where(okF, Fi, 1.0)), np.nan)

            # choose a stable comparison set ONCE — robust MAD-based
            # sigma-clip on each star's run-median ZP. std is inflated by
            # the outliers we're trying to catch; MAD isn't, so a comp
            # star with a wrong catalog mag now actually gets clipped
            # instead of masquerading as normal scatter. Floor sigma at
            # 0.05 mag so tight ensembles don't over-clip themselves down
            # to one star.
            z_star = np.nanmedian(zi, axis=1)
            keepC  = np.isfinite(z_star)
            for _ in range(3):
                if not np.any(keepC):
                    break
                med  = np.nanmedian(z_star[keepC])
                mad  = np.nanmedian(np.abs(z_star[keepC] - med))
                sd_r = max(1.4826 * float(mad), 0.05)
                if not np.isfinite(sd_r) or sd_r == 0:
                    break
                prev_keep = keepC.copy()
                keepC &= np.abs(z_star - med) <= 3.0 * sd_r
                if np.array_equal(keepC, prev_keep):
                    break  # converged; no point re-computing MAD
            zi = zi[keepC, :]

            ZP_t  = np.nanmedian(zi, axis=0)
            n_t   = np.sum(np.isfinite(zi), axis=0)
            madz  = np.nanmedian(np.abs(zi - ZP_t[None, :]), axis=0)
            SEM_t = np.where(n_t > 1, 1.4826 * madz / np.sqrt(np.maximum(n_t, 1)), np.nan)

            with np.errstate(divide="ignore", invalid="ignore"):
                mags = -2.5 * np.log10(F_T) + ZP_t
                merr = np.sqrt((LOGC * Ferr_T / F_T) ** 2
                               + np.nan_to_num(SEM_t) ** 2 + fn ** 2 + floor ** 2)
            bad = ~np.isfinite(mags) | (F_T <= 0) | (n_t < 1)

            n_ens       = int(np.count_nonzero(keepC))
            # AAVSO ENSEMBLE: KMAG is the check star standardised through the
            # SAME ensemble ZP (not its catalog mag), so (K - catalog) is a real
            # quality check. Compute K exactly like the target.
            F_K = np.asarray(self.raw_flux[k_idx, :], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                _K = -2.5 * np.log10(F_K) + ZP_t
            _K = _K[np.isfinite(_K)]
            _used = [int(m) for m in comp_idx[keepC]]
            cname_field = "ENSEMBLE"
            cmag_field  = "na"
            kname_field = kname
            kmag_field  = f"{float(np.median(_K)):.3f}" if _K.size else "na"
            note_field  = (f"ensemble ZP=median(Vcat+2.5log10 F) N={n_ens}; "
                           f"err incl SEM+{floor:.2f}floor")
            _cn = _comp_radec_labels(_used)
            if _cn:
                note_field += f"; comps[RA/Dec]: {_cn}"
        else:
            # Fallback: single-comparison vs the check star, raw-flux ratio.
            F_C    = np.asarray(self.raw_flux[k_idx, :],     dtype=np.float64)
            Ferr_C = np.asarray(self.raw_flux_err[k_idx, :], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                mags  = kmag - 2.5 * np.log10(F_T / F_C)
                frac2 = (Ferr_T / F_T) ** 2 + (Ferr_C / F_C) ** 2
                merr  = np.sqrt((LOGC ** 2) * frac2 + 2.0 * fn ** 2 + floor ** 2)
            bad = ~np.isfinite(mags) | (F_T <= 0) | (F_C <= 0)

            # Check star doubling as the comparison → CMAG is its INSTRUMENTAL
            # mag (classic differential); no independent check remains.
            with np.errstate(divide="ignore", invalid="ignore"):
                _fc = float(np.nanmedian(F_C[F_C > 0])) if np.any(F_C > 0) else np.nan
            cname_field = str(kname)
            cmag_field  = f"{-2.5 * np.log10(_fc):.3f}" if np.isfinite(_fc) else "na"
            kname_field = "na"
            kmag_field  = "na"
            # Say WHY there is no ensemble, and keep the comparison's
            # position. Previously the comps[RA/Dec] provenance existed
            # only in the ensemble branch, so a run that fell back here
            # lost it with no explanation -- the file then looks as though
            # an ensemble was never attempted, when in fact the catalog
            # lookup came up short (commonly because VizieR was
            # unreachable, so APASS DR9 and UCAC4 returned nothing).
            note_field  = (f"single-comparison vs {kname}; "
                           f"err incl {floor:.2f} floor; "
                           f"ensemble not used: {comp_idx.size} of "
                           f"{len(comp_members)} field stars had a catalog V "
                           f"(need 3)"
                           + self._catalog_failure_note())
            _cn_fb = _comp_radec_labels([k_idx])
            if _cn_fb:
                note_field += f"; comp[RA/Dec]: {_cn_fb}"

        mags = np.where(bad, np.nan, mags)
        merr = np.where(bad | ~np.isfinite(merr), np.nan, merr)

        _MERR_MAX = 1.0
        _rows_written = 0
        _na_count     = 0
        try:
            with open(path, "w") as f:
                for L in header_lines:
                    f.write(L + "\n")
                for j, t in enumerate(jd):
                    m = mags[j]; me = merr[j]
                    if not np.isfinite(m):
                        continue  # no valid calibrated magnitude for this frame
                    # AAVSO Extended requires MERR ≤ 1.0 or the literal
                    # "na". Downgrade out-of-range values to "na" (David
                    # msg-6 request) rather than let WebObs reject the
                    # whole file at upload.
                    if np.isfinite(me) and me <= _MERR_MAX:
                        me_str = f"{me:.3f}"
                    else:
                        me_str = "na"
                        _na_count += 1  # single-band MERR-out-of-range → emit na
                    am = float(np.clip(self.airmasses[j] if j < len(self.airmasses) else 1.0, 1.0, 40.0))
                    fields = [
                        star_id,
                        f"{t:.5f}",
                        f"{m:.3f}",
                        me_str,
                        filt,
                        "NO",
                        "STD",
                        cname_field,
                        cmag_field,
                        kname_field,
                        kmag_field,
                        f"{am:.1f}",
                        "na",
                        "na",
                        note_field.replace(",", ";"),
                    ]
                    f.write(",".join(fields) + "\n")
                    _rows_written += 1
        except Exception as e:
            QMessageBox.critical(self, "Export AAVSO", f"Failed to write file:\n{e}")
            return

        _body = f"Wrote {fmt} ({_rows_written} row(s)) →\n{path}"
        if _na_count > 0:
            _body += (f"\n\nMERR=na for {_na_count} row(s) "
                      f"(computed error > {_MERR_MAX:.1f}).")
        _body += "\n\nOpen AAVSO WebObs upload page now?"

        msg = QMessageBox(self)
        msg.setWindowTitle("Export AAVSO")
        msg.setText(_body)
        yes = msg.addButton("Yes", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("No", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() == yes:
            webbrowser.open("https://www.aavso.org/webobs/file")

    # ---------------- TESScut ----------------

    def query_tesscut(self):
        radec = self.get_selected_star_radec()
        if radec is None:
            QMessageBox.warning(self, "No Star Selected", "Please select a star from the list to fetch TESScut data.")
            return
        ra, dec = radec
        print(f"[DEBUG] TESScut Query Requested for RA={ra:.6f}, Dec={dec:.6f}")
        coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        size = 10
        MAX_RETRIES = 5

        manifest = None
        for mtry in range(1, MAX_RETRIES+1):
            try:
                print(f"[DEBUG] Manifest attempt {mtry}/{MAX_RETRIES}…")
                manifest = Tesscut.get_cutouts(coordinates=coord, size=size)
                if manifest:
                    print(f"[DEBUG] Manifest OK: {len(manifest)} sector(s).")
                    break
                else:
                    raise RuntimeError("Empty manifest")
            except Exception as me:
                print(f"[DEBUG] Manifest attempt {mtry} failed: {me}")
                if mtry == MAX_RETRIES:
                    QMessageBox.information(self, "No TESS Data", "There are no TESS cutouts available at that position.")
                    self.status_label.setText("No TESScut data found.")
                    return
                non_blocking_sleep(2)

        self.status_label.setText("Querying TESScut…")
        QApplication.processEvents()

        cache_dir = os.path.join(os.path.expanduser("~"), ".setiastro", "tesscut_cache")
        os.makedirs(cache_dir, exist_ok=True)

        for dtry in range(1, MAX_RETRIES+1):
            try:
                print(f"[DEBUG] Download attempt {dtry}/{MAX_RETRIES}…")
                cutouts = Tesscut.download_cutouts(coordinates=coord, size=size, path=cache_dir)
                if not cutouts:
                    raise RuntimeError("No cutouts downloaded")
                print(f"[DEBUG] Downloaded {len(cutouts)} cutout(s).")

                for cutout in cutouts:
                    original_path = cutout['Local Path']
                    print(f"[DEBUG] Processing: {original_path}")
                    with fits.open(original_path, mode='readonly') as hdul:
                        sector = hdul[1].header.get('SECTOR', 'unknown')
                    ext = os.path.splitext(original_path)[1]
                    cache_key = f"tess_sector{sector}_ra{int(round(ra*10000))}_dec{int(round(dec*10000))}{ext}"
                    cached_path = os.path.join(cache_dir, cache_key)

                    if not os.path.exists(cached_path):
                        print(f"[DEBUG] Caching as: {cached_path}")
                        shutil.move(original_path, cached_path)
                    else:
                        print(f"[DEBUG] Already cached: {cached_path}")
                        os.remove(original_path)

                    tpf = TessTargetPixelFile(cached_path)
                    xpix, ypix = tpf.wcs.world_to_pixel(coord)
                    ny, nx = tpf.flux.shape[1], tpf.flux.shape[2]
                    Y, X = np.mgrid[:ny, :nx]
                    r_pix = 2.5
                    aper_mask = ((X - xpix)**2 + (Y - ypix)**2) <= r_pix**2

                    lc = (tpf.to_lightcurve(aperture_mask=aper_mask).remove_nans().normalize())
                    upper, lower = 5.0, -1.0
                    mask = (lc.flux < upper) & (lc.flux > lower)
                    n_clipped = np.sum(~mask)
                    print(f"[DEBUG] Clipping {n_clipped} points outside [{lower}, {upper}]×")
                    lc = lc[mask]

                    lc.plot(label=f"Sector {tpf.sector} (clipped)")
                    plt.title(f"TESS Light Curve - Sector {tpf.sector}")
                    plt.tight_layout()
                    plt.show()

                self.status_label.setText("TESScut fetch complete.")
                return

            except Exception as de:
                print(f"[ERROR] Download attempt {dtry} failed: {de}")
                self.status_label.setText(f"TESScut attempt {dtry}/{MAX_RETRIES} failed.")
                QApplication.processEvents()
                if dtry == MAX_RETRIES:
                    QMessageBox.critical(self, "TESScut Error", f"TESScut failed after {MAX_RETRIES} attempts.\n\n{de}")
                    self.status_label.setText("TESScut fetch failed.")
                else:
                    non_blocking_sleep(2)

    # ---------------- Pixel → Sky helper ----------------
    def get_selected_star_radec(self):
        selected_items = self.star_list.selectedItems()
        if not selected_items:
            return None

        selected_index = selected_items[0].data(Qt.ItemDataRole.UserRole)
        x, y = self.star_positions[selected_index]

        if self._wcs is None:
            return None

        bin_factor = getattr(self, "_wcs_bin_factor", 1)
        xw = x * bin_factor
        yw = y * bin_factor

        sky = self._wcs.pixel_to_world(xw, yw)
        return sky.ra.degree, sky.dec.degree