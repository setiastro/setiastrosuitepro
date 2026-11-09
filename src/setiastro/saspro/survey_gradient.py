# saspro/survey_gradient.py — Survey Reference Gradient Correction (SRGC)
# -----------------------------------------------------------------------------
# Additive gradient correction against a free, publicly available all-sky
# reference survey.
#
# THE IDEA
# --------
# ABE/ADBE models the background of your image and removes it. The problem it
# cannot solve is attribution: your background is
#
#       B_t(x,y) = s·S(x,y) + G(x,y) + c
#
# where S is REAL diffuse sky structure (nebulosity, IFN, ISM), G is YOUR
# additive gradient (light pollution, airglow, vignetting residual), and s, c
# are an unknown scale and pedestal. A self-contained model has no way to tell
# S from G, so it eats nebulosity along with the gradient. That is the whole
# reason PixInsight funded the MARS survey.
#
# This module takes the same structural approach but with free catalog data.
# We fetch a reference image of the SAME patch of sky from a published survey,
# model its background the same way, and fit
#
#       B_t  ≈  a + b·B_r
#
# robustly over the frame. Whatever is left over,
#
#       G  =  B_t − (a + b·B_r)
#
# is the part of your background that the real sky does NOT explain. That is
# your gradient, and that is what we subtract. Real nebulosity survives because
# it is present in the reference too and is absorbed by the b·B_r term.
#
# WHY hips2fits
# -------------
# Every survey we care about is published as a HiPS (IVOA standard). The CDS
# hips2fits service takes a HiPS id plus an astropy WCS and returns a FITS
# cutout already reprojected onto exactly that WCS. So there is no reprojection,
# no mosaicking and no tile handling to write: hand it the plate-solved WCS of
# the open document and the reference comes back pixel-aligned.
#
# SURVEYS
# -------
#   NSNS   Northern Sky Narrowband Survey (Ziegenbalg). Ha / [OIII] / [SII],
#          whole northern hemisphere down to dec -16, ~10 arcsec resolution.
#          The right reference for narrowband work in the north.
#   DSS2   All-sky, both hemispheres, red/blue. Lower quality but it is the
#          fallback that always has coverage, which matters in the south.
#
# SCALE NOTE
# ----------
# The reference is fetched and both backgrounds are modelled at REDUCED
# resolution (default long side 1024 px). This is not a shortcut — a gradient
# is by definition a low-spatial-frequency field, and the degeneracy this
# module exists to break only lives at large scales. NSNS at ~10 arcsec cannot
# tell you anything at your native sampling anyway. Working small makes the
# fetch fast and keeps the fit well conditioned.
# -----------------------------------------------------------------------------
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QSize, QSettings
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox, QMessageBox,
    QApplication, QScrollArea, QPlainTextEdit,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from astropy.wcs import WCS

from setiastro.saspro.abe import abe_run, _downsample_area, _upscale_bg
from setiastro.saspro.autostretch import autostretch as hard_autostretch


# =============================================================================
#                              Survey registry
# =============================================================================
# `hips` values are HiPS ids resolvable by the CDS hips2fits service.
# `mono` marks a single-plane reference (everything here is mono — we solve a
# separate a/b per target channel against the same structure, which is what
# lets a colour-selective gradient like red light pollution come out).

#
# LICENSING — READ BEFORE SHIPPING
# --------------------------------
# NSNS is published by Stefan Ziegenbalg under CC-BY-NC-SA 4.0. The NC clause
# is NON-COMMERCIAL. SASpro is a paid product, so routing NSNS through it needs
# a decision you have to make deliberately — fetching at runtime on the user's
# behalf is arguably not redistribution, but "arguably" is not a licence. Get
# written permission from Ziegenbalg, or gate the NSNS entries behind a notice,
# before this ships in a build you sell.
#
# DSS2 carries no such restriction and is the safe default for a shipping
# product, which is why it is listed and why nothing here hard-defaults to NSNS
# without the user choosing it.
#
# `hips` ids verified against the CDS MOCServer and simg.de/nebulae3/dr0_2.
# These are the LINEAR-INTENSITY FITS layers, not the 8-bit PNG display
# variants (halpha8, tc8, ohs8 …) — those are quantised and star-subtracted
# for visualisation and are useless for photometric background work.
# NSNS intensity layers are in Rayleighs, i.e. genuinely calibrated surface
# brightness, which is better than this method strictly needs.

# ---------------------------------------------------------------------------
# hips2fits service endpoints.
#
# CDS publishes two INDEPENDENT endpoints for this service. They go down
# separately, so never pin to one — an outage on the primary is a routine
# event, not an error worth showing the user. The last endpoint that actually
# worked is remembered in QSettings and tried first next time.
# ---------------------------------------------------------------------------
HIPS2FITS_ENDPOINTS: list[str] = [
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits",
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
]

# Per-request timeout, seconds. Short on purpose: when a host is down we want
# to fall through to the next one quickly rather than make the user wait out
# a 120 s connect timeout on each.
HIPS_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# Tracing. ON by default while this is beta — a reference fetch crosses a WCS
# rescale, a URL serialisation, somebody else's server and a reprojection, and
# when it comes back empty you cannot tell which of those lied without seeing
# all four. Set to False once it is trustworthy.
# ---------------------------------------------------------------------------
DEBUG = True

_LOG_SINK = None   # set_log_sink() points this at the main window's _log


def set_log_sink(fn) -> None:
    """Route trace lines to the host app's log in addition to stdout."""
    global _LOG_SINK
    _LOG_SINK = fn


def _log(msg: str) -> None:
    if not DEBUG:
        return
    line = f"[SRGC] {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    if _LOG_SINK is not None:
        try:
            _LOG_SINK(line)
        except Exception:
            pass


def _wcs_summary(w: "WCS", label: str = "") -> str:
    """One-line-per-field dump of the parts of a WCS that actually travel."""
    try:
        import astropy.wcs.utils as wu
        scales = np.abs(wu.proj_plane_pixel_scales(w)) * 3600.0
        sc = f"{scales[0]:.3f}/{scales[1]:.3f} arcsec/px"
    except Exception:
        sc = "scale=?"
    try:
        shape = w.pixel_shape
    except Exception:
        shape = None
    bits = [
        f"ctype={list(w.wcs.ctype)}",
        f"crval={[round(float(v), 6) for v in w.wcs.crval]}",
        f"crpix={[round(float(v), 3) for v in w.wcs.crpix]}",
        f"pixel_shape={shape}",
        sc,
        f"sip={'yes' if getattr(w, 'sip', None) is not None else 'no'}",
    ]
    try:
        if w.wcs.has_cd():
            bits.append(f"cd={np.array(w.wcs.cd).ravel().round(10).tolist()}")
        else:
            bits.append(f"cdelt={np.array(w.wcs.cdelt).round(10).tolist()}")
            if w.wcs.has_pc():
                bits.append(f"pc={np.array(w.wcs.pc).ravel().round(10).tolist()}")
    except Exception:
        pass
    return f"WCS[{label}] " + "  ".join(bits)


def _array_stats(a: np.ndarray, label: str = "") -> str:
    """What actually came back. Finite-fraction is the one that matters."""
    a = np.asarray(a)
    finite = np.isfinite(a)
    n = a.size or 1
    ff = float(finite.sum()) / n
    if finite.any():
        v = a[finite]
        rng = (f"min={float(v.min()):.6g} med={float(np.median(v)):.6g} "
               f"max={float(v.max()):.6g}")
        if float(v.min()) == float(v.max()):
            rng += "  <-- CONSTANT, no structure"
    else:
        rng = "ALL NON-FINITE"
    return (f"array[{label}] shape={a.shape} dtype={a.dtype} "
            f"finite={ff*100:.1f}%  {rng}")


def _strip_sip(w: "WCS") -> "WCS":
    """
    Drop SIP coefficients AND the '-SIP' CTYPE suffix.

    Nulling w.sip alone leaves CTYPE reading 'RA---TAN-SIP' with no A_/B_
    coefficients present, which is malformed FITS. astropy tolerates it, but
    the WCS gets serialised into a URL and parsed by somebody else's server,
    and there is no reason to hand a remote parser a header that declares a
    distortion it does not carry.
    """
    try:
        w.sip = None
    except Exception:
        pass
    try:
        w.wcs.ctype = [str(c).replace("-SIP", "") for c in w.wcs.ctype]
    except Exception:
        pass
    return w

#
# LICENSING — READ BEFORE SHIPPING
# --------------------------------
# NSNS is published by Stefan Ziegenbalg under CC-BY-NC-SA 4.0. The NC clause
# is NON-COMMERCIAL. SASpro is a paid product, so routing NSNS through it needs
# a decision you have to make deliberately — fetching at runtime on the user's
# behalf is arguably not redistribution, but "arguably" is not a licence. Get
# written permission from Ziegenbalg, or gate the NSNS entries behind a notice,
# before this ships in a build you sell.
#
# DSS2 carries no such restriction (ODbL) and is the safe default for a
# shipping product, which is why it is listed.
#
# `hips` ids verified against the CDS MOCServer and simg.de/nebulae3/dr0_2.
# These are the LINEAR-INTENSITY FITS layers, not the 8-bit PNG display
# variants (halpha8, tc8, ohs8 …) — those are quantised and star-subtracted
# for visualisation and are useless for photometric background work.
# NSNS intensity layers are in Rayleighs, i.e. genuinely calibrated surface
# brightness, which is better than this method strictly needs.
#
# `mirrors` are DIRECT HiPS base URLs (static tile trees, no service layer).
# They are the fallback for when hips2fits itself is unavailable, and they
# deliberately lead with a NON-CDS host where one exists — simg.de serves NSNS
# itself, and IRSA at Caltech mirrors DSS2 — so a total CDS outage is survivable.

SURVEYS: dict[str, dict] = {
    "NSNS Ha (north, dec > -16)": {
        "hips": "simg.de/P/NSNS/DR0_2/halpha",
        "mirrors": [
            "http://www.simg.de/nebulae3/dr0_2/halpha",
            "https://alaskybis.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_halpha",
            "https://alasky.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_halpha",
        ],
        "note": "Northern Sky Narrowband Survey, H-alpha, linear intensity in "
                "Rayleighs. Best match for Ha/SHO data. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS [OIII] (north, dec > -16)": {
        "hips": "simg.de/P/NSNS/DR0_2/oiii",
        "mirrors": [
            "http://www.simg.de/nebulae3/dr0_2/oiii",
            "https://alaskybis.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_oiii",
            "https://alasky.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_oiii",
        ],
        "note": "Northern Sky Narrowband Survey, [OIII] 500.7 nm. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS [SII] (north, dec > -16)": {
        "hips": "simg.de/P/NSNS/DR0_2/sii",
        "mirrors": [
            "http://www.simg.de/nebulae3/dr0_2/sii",
            "https://alaskybis.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_sii",
            "https://alasky.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_sii",
        ],
        "note": "Northern Sky Narrowband Survey, [SII] 671.7/673.0 nm. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS visual continuum (north)": {
        "hips": "simg.de/P/NSNS/DR0_2/vc",
        "mirrors": [
            "http://www.simg.de/nebulae3/dr0_2/vc",
            "https://alaskybis.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_vc",
            "https://alasky.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_vc",
        ],
        "note": "NSNS star-subtracted visible continuum. Use for RGB/LRGB in "
                "the north. Star subtraction is a bonus here. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "DSS2 red (all sky)": {
        "hips": "CDS/P/DSS2/red",
        "mirrors": [
            "https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/red",
            "https://alaskybis.cds.unistra.fr/DSS/DSS2Merged",
            "https://alasky.cds.unistra.fr/DSS/DSS2Merged",
            # NOT skies.esac.esa.int/DSSColor — it serves jpeg tiles only.
            # An 8-bit JPEG of a colour composite would reproject happily and
            # produce a confident, wrong background. Absence is safer.
        ],
        "note": "All-sky fallback, both hemispheres. Shallower and plate-based, "
                "but always covered and ODbL — no licence restriction.",
        "dec_min": -90.0,
        "noncommercial": False,
    },
    "DSS2 blue (all sky)": {
        "hips": "CDS/P/DSS2/blue",
        "mirrors": [
            "https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/blue",
            "https://alaskybis.cds.unistra.fr/DSS/DSS2-blue-XJ-S",
            "https://alasky.cds.unistra.fr/DSS/DSS2-blue-XJ-S",
        ],
        "note": "All-sky fallback, blue plates. ODbL, no licence restriction.",
        "dec_min": -90.0,
        "noncommercial": False,
    },
}


def mirrors_for(hips_id: str) -> list[str]:
    """Direct HiPS base URLs registered for a hips id (may be empty)."""
    for info in SURVEYS.values():
        if info.get("hips") == hips_id:
            return list(info.get("mirrors") or [])
    return []

DEFAULT_SURVEY = "NSNS Ha (north, dec > -16)"


# =============================================================================
#                                WCS helpers
# =============================================================================

def _celestial_wcs_from_header(header) -> WCS | None:
    """
    Build a clean 2D celestial WCS from a document header.

    Mirrors the normalisation SFCC does (deprecated RADECSYS/EPOCH keywords)
    because the same headers flow through both tools. SIP distortion is
    deliberately dropped: we rescale the WCS below, and SIP coefficients are
    expressed in ORIGINAL pixel units, so carrying them onto a rescaled grid
    would be wrong. At background scale the distortion term is irrelevant.
    """
    if header is None:
        return None
    try:
        hdr = header.copy() if hasattr(header, "copy") else dict(header)
        if "RADECSYS" in hdr and "RADESYS" not in hdr:
            hdr["RADESYS"] = str(hdr["RADECSYS"]).strip()
            try:
                del hdr["RADECSYS"]
            except Exception:
                pass
        if "EPOCH" in hdr and "EQUINOX" not in hdr:
            hdr["EQUINOX"] = hdr["EPOCH"]
            try:
                del hdr["EPOCH"]
            except Exception:
                pass
        w = _strip_sip(WCS(hdr, naxis=2, relax=True).celestial)
        if not w.has_celestial:
            _log("header parsed but has no celestial axes")
            return None
        _log(_wcs_summary(w, "from header"))
        return w
    except Exception as e:
        _log(f"WCS parse FAILED: {type(e).__name__}: {e}")
        return None


def _rescaled_wcs(wcs: WCS, src_hw: tuple[int, int], out_hw: tuple[int, int]) -> WCS:
    """
    Return a copy of `wcs` describing the SAME sky footprint on a coarser grid.

    Pixel scale is multiplied by the decimation factor and CRPIX is remapped
    with the half-pixel convention (FITS pixel centres are 1-based at the
    centre of the pixel, so the edge sits at 0.5).

    Handles both CD-matrix and PC+CDELT headers, which is the usual thing that
    bites when rescaling a WCS by hand.
    """
    H, W = int(src_hw[0]), int(src_hw[1])
    h, w = int(out_hw[0]), int(out_hw[1])
    fx = W / float(max(w, 1))
    fy = H / float(max(h, 1))

    w2 = _strip_sip(wcs.deepcopy())

    scaled = False
    try:
        if wcs.wcs.has_cd():
            cd = np.array(wcs.wcs.cd, dtype=float).copy()
            cd[:, 0] *= fx
            cd[:, 1] *= fy
            w2.wcs.cd = cd
            scaled = True
    except Exception:
        scaled = False

    if not scaled:
        cdelt = np.array(wcs.wcs.cdelt, dtype=float).copy()
        cdelt[0] *= fx
        cdelt[1] *= fy
        w2.wcs.cdelt = cdelt

    crpix = np.array(wcs.wcs.crpix, dtype=float).copy()
    crpix[0] = (crpix[0] - 0.5) / fx + 0.5
    crpix[1] = (crpix[1] - 0.5) / fy + 0.5
    w2.wcs.crpix = crpix

    try:
        w2.pixel_shape = (w, h)
    except Exception:
        pass
    try:
        w2.array_shape = (h, w)
    except Exception:
        pass
    return w2


def _work_shape(H: int, W: int, long_side: int) -> tuple[int, int]:
    """Reduced grid shape preserving aspect ratio, never upscaling."""
    long_side = int(max(64, long_side))
    if max(H, W) <= long_side:
        return H, W
    f = max(H, W) / float(long_side)
    return max(8, int(round(H / f))), max(8, int(round(W / f)))


# =============================================================================
#                              Reference fetch
# =============================================================================

class SurveyFetchError(RuntimeError):
    """Raised when the reference cutout cannot be obtained or is unusable."""


def _flatten_to_2d(data, h: int, w: int) -> np.ndarray:
    """Coerce whatever a HiPS gave us into a single (h, w) float32 plane."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        # Colour HiPS arrive as (bands, h, w) or (h, w, bands); collapse to
        # luminance either way — only background structure matters here.
        if data.shape[0] in (3, 4) and data.shape[0] < data.shape[-1]:
            data = np.nanmean(data[:3], axis=0)
        else:
            data = np.nanmean(data[..., :3], axis=-1)
    if data.shape[:2] != (h, w):
        raise SurveyFetchError(
            f"cutout shape {data.shape[:2]} != requested grid {(h, w)}"
        )
    return data.astype(np.float32, copy=False)


def _last_good_endpoint() -> str | None:
    try:
        v = QSettings().value("srgc/last_endpoint", None)
        return str(v) if v else None
    except Exception:
        return None


def _remember_endpoint(url: str) -> None:
    try:
        QSettings().setValue("srgc/last_endpoint", url)
    except Exception:
        pass


def _ordered_endpoints() -> list[str]:
    """Known-good endpoint first, then the rest in declared order."""
    eps = list(HIPS2FITS_ENDPOINTS)
    last = _last_good_endpoint()
    if last and last in eps:
        eps.remove(last)
        eps.insert(0, last)
    elif last:
        eps.insert(0, last)
    return eps


def _fetch_via_hips2fits(qw: WCS, h: int, w: int, hips_id: str,
                         endpoint: str, timeout: float) -> np.ndarray:
    from astroquery.hips2fits import hips2fits
    prev = getattr(hips2fits, "server", None)
    try:
        # astroquery reads url=self.server at request time (hips2fits.py), so
        # assigning the instance attribute really does redirect the call. This
        # is asserted rather than assumed because the whole failover depends
        # on it and a silent revert would look exactly like "the mirror is
        # also down".
        hips2fits.server = endpoint
        assert hips2fits.server == endpoint, "server override did not stick"
        try:
            hips2fits.timeout = float(timeout)
        except Exception:
            pass

        if DEBUG:
            try:
                payload = hips2fits.query_with_wcs(
                    hips=hips_id, wcs=qw, format="fits", get_query_payload=True)
                _log(f"  endpoint   : {hips2fits.server}")
                _log(f"  hips id    : {hips_id}")
                if isinstance(payload, dict):
                    wcs_part = payload.get("wcs") or payload.get("WCS")
                    for k, v in payload.items():
                        if k.lower() != "wcs":
                            _log(f"  param      : {k}={v}")
                    if wcs_part:
                        _log(f"  wcs param  : {wcs_part}")
                else:
                    _log(f"  payload    : {payload}")
            except Exception as e:
                _log(f"  (could not dump query payload: {_brief(e)})")

        result = hips2fits.query_with_wcs(hips=hips_id, wcs=qw, format="fits")
    finally:
        if prev is not None:
            try:
                hips2fits.server = prev
            except Exception:
                pass
    try:
        data = result[0].data
        _log(f"  response   : HDU0 header NAXIS={result[0].header.get('NAXIS')} "
             f"BITPIX={result[0].header.get('BITPIX')}")
    except Exception:
        data = result
    out = _flatten_to_2d(data, h, w)
    _log("  " + _array_stats(out, "hips2fits"))
    return out


def _hips_level_for(qw: WCS, h: int, w: int) -> int:
    """
    HEALPix order whose pixel size is closest to the work grid's.

    A HEALPix pixel at order n subtends about 58.63/2**n degrees. Matching the
    order to the grid avoids pulling far more tile data than the output can
    show, which on an all-sky survey is the difference between a few MB and a
    few GB.
    """
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scale = float(np.mean(np.abs(proj_plane_pixel_scales(qw))))
    except Exception:
        scale = 1.0 / 3600.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0 / 3600.0
    order = int(round(np.log2(58.63 / scale)))
    return int(np.clip(order, 3, 11))


# ---------------------------------------------------------------------------
# Direct HiPS tile reader.
#
# reproject's HiPS reader was tried first and discarded: given a live dataset
# it fetched `properties` successfully, then silently NaN-filled every tile and
# reported success. A reader that cannot distinguish "no data here" from "I
# failed to fetch" is worse than no reader for a tool whose entire job is
# deciding what the sky really looks like.
#
# So this goes sky -> tile -> pixel directly. The only non-obvious part is the
# layout of pixels inside a tile, and that was not guessed: it was solved
# against NSNS tiles, which carry their own WCS, and the winning convention
# reproduced 1058/1058 sampled pixels across two tiles at two different orders
# with a max positional error of 0.0000 arcsec.
#
#     j = interleave(a = tile_width-1-row,  b = col)
#
# with a's bits in the even positions and b's in the odd. HiPS is a standard,
# so the same layout holds for every dataset — which is what lets IRSA's bare
# DSS2 tiles (no WCS at all in the header) be read exactly like NSNS's.
# ---------------------------------------------------------------------------

_PROPS_CACHE: dict[str, dict] = {}
_TILE_CACHE: dict[tuple, np.ndarray | None] = {}
_TILE_CACHE_MAX = 256


def _http_get(url: str, timeout: float = HIPS_TIMEOUT) -> bytes:
    import urllib.request
    req = urllib.request.Request(
        url, headers={"User-Agent": "SASpro-SurveyGradient"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def hips_properties(base_url: str, *, timeout: float = HIPS_TIMEOUT) -> dict:
    """Fetch and parse a HiPS properties file (cached per session)."""
    base = base_url.rstrip("/")
    if base in _PROPS_CACHE:
        return _PROPS_CACHE[base]
    raw = _http_get(base + "/properties", timeout=timeout)
    props: dict[str, str] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        props[k.strip()] = v.strip()
    _PROPS_CACHE[base] = props
    _log(f"  properties : order={props.get('hips_order')} "
         f"width={props.get('hips_tile_width')} "
         f"fmt={props.get('hips_tile_format')} "
         f"frame={props.get('hips_frame')}")
    return props


def _deinterleave(j: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of interleave(a, b): a from even bits, b from odd bits."""
    j = np.asarray(j, dtype=np.int64)
    a = np.zeros_like(j)
    b = np.zeros_like(j)
    for bit in range(m):
        a |= ((j >> (2 * bit)) & 1) << bit
        b |= ((j >> (2 * bit + 1)) & 1) << bit
    return a, b


def _fetch_tile(base: str, order: int, npix: int, ext: str,
                timeout: float) -> np.ndarray | None:
    """One tile as float32, or None if it is genuinely absent."""
    key = (base, order, int(npix), ext)
    if key in _TILE_CACHE:
        return _TILE_CACHE[key]

    url = f"{base}/Norder{order}/Dir{(int(npix) // 10000) * 10000}/Npix{int(npix)}.{ext}"
    arr: np.ndarray | None = None
    try:
        raw = _http_get(url, timeout=timeout)
        import io
        import warnings as _w
        from astropy.io import fits
        with _w.catch_warnings():
            # IRSA serves tiles without the trailing FITS padding block, which
            # astropy flags as truncated. The pixel data is all there; only the
            # pad is missing, so the warning is noise.
            _w.simplefilter("ignore")
            with fits.open(io.BytesIO(raw), ignore_missing_end=True) as hdul:
                hdu = next((x for x in hdul if getattr(x, "data", None) is not None),
                           None)
                if hdu is None:
                    raise ValueError("no image HDU in tile")
                data = np.asarray(hdu.data)
                blank = hdu.header.get("BLANK", None)
        arr = data.astype(np.float32)
        if blank is not None:
            try:
                arr[data == int(blank)] = np.nan
            except Exception:
                pass
    except Exception as e:
        # A 404 is ordinary: HiPS is sparse, and an absent tile means no survey
        # coverage there. Anything else is logged so a dead host or a bad path
        # can never masquerade as "the sky is empty here".
        code = getattr(e, "code", None)
        if code == 404:
            _log(f"  tile 404   : Norder{order}/Npix{npix} (no coverage)")
        else:
            _log(f"  tile ERROR : Norder{order}/Npix{npix}: {_brief(e)}")
        arr = None

    if len(_TILE_CACHE) > _TILE_CACHE_MAX:
        _TILE_CACHE.clear()
    _TILE_CACHE[key] = arr
    return arr


def _fetch_via_tiles(qw: WCS, h: int, w: int, base_url: str,
                     *, timeout: float = HIPS_TIMEOUT) -> np.ndarray:
    """
    Sample a HiPS tile tree onto the grid described by `qw`.

    Works off static tile files only — no hips2fits, no service layer, so it
    keeps working through a CDS outage and can read a host that has nothing to
    do with CDS.
    """
    from concurrent.futures import ThreadPoolExecutor
    from astropy_healpix import HEALPix
    from astropy.coordinates import ICRS, Galactic

    base = base_url.rstrip("/")
    props = hips_properties(base, timeout=timeout)

    max_order = int(props.get("hips_order", 5))
    tile_w = int(props.get("hips_tile_width", 512))
    fmts = str(props.get("hips_tile_format", "fits")).lower().split()
    if "fits" not in fmts:
        raise SurveyFetchError(
            f"{base} serves only {fmts or ['?']} tiles. This needs FITS — an "
            "8-bit JPEG/PNG display product cannot carry the surface "
            "brightness this method measures."
        )
    m = int(round(np.log2(tile_w)))
    if 2 ** m != tile_w:
        raise SurveyFetchError(f"tile width {tile_w} is not a power of two")

    frame = Galactic() if str(props.get("hips_frame", "equatorial")).lower() \
        .startswith("gal") else ICRS()

    # Pick the tile order whose pixels match the output grid: no point pulling
    # order-9 tiles to fill a 3-arcsec grid, and no point pulling order-3 when
    # the output can resolve more.
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scale_as = float(np.mean(np.abs(proj_plane_pixel_scales(qw)))) * 3600.0
    except Exception:
        scale_as = 3.0
    d_want = int(round(np.log2(58.6323 * 3600.0 / max(scale_as, 1e-6))))
    order = int(np.clip(d_want - m, 0, max_order))
    deep = order + m
    _log(f"  grid {scale_as:.2f}\"/px -> deep order {d_want}, "
         f"using tile order {order}/{max_order} (deep {deep})")

    hp = HEALPix(nside=2 ** deep, order="nested", frame=frame)

    yy, xx = np.mgrid[0:h, 0:w]
    sky = qw.pixel_to_world(xx.ravel(), yy.ravel())
    try:
        sky = sky.transform_to(frame)
    except Exception:
        pass
    idx = np.asarray(hp.skycoord_to_healpix(sky), dtype=np.int64)

    npix_tile = idx >> (2 * m)
    j = idx & (4 ** m - 1)
    a, b = _deinterleave(j, m)
    row = (tile_w - 1) - a
    col = b

    out = np.full(h * w, np.nan, dtype=np.float32)
    uniq = np.unique(npix_tile)
    _log(f"  tiles needed: {len(uniq)} at order {order}")

    ext = "fits"
    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = dict(zip(
            uniq.tolist(),
            pool.map(lambda n: _fetch_tile(base, order, int(n), ext, timeout),
                     uniq.tolist())))

    got = 0
    for n in uniq.tolist():
        t = tiles.get(n)
        if t is None:
            continue
        got += 1
        sel = npix_tile == n
        out[sel] = t[row[sel], col[sel]]

    _log(f"  tiles ok   : {got}/{len(uniq)}")
    res = out.reshape(h, w)
    _log("  " + _array_stats(res, f"tiles order={order}"))
    if got == 0:
        raise SurveyFetchError(
            f"none of the {len(uniq)} tiles covering this field could be "
            f"fetched from {base}"
        )
    return res


def fetch_reference(wcs: WCS, out_hw: tuple[int, int], hips_id: str,
                    *, progress_cb=None, timeout: float = HIPS_TIMEOUT,
                    allow_tiles: bool = True) -> np.ndarray:
    """
    Pull a reference cutout for `wcs`, trying every route we know about.

    Order of attempts:
      1. hips2fits on each CDS endpoint (known-good one first). Fast, because
         the reprojection happens server-side.
      2. the direct HiPS tile tree on each registered mirror, reprojected
         locally. Slower, but survives a total hips2fits outage and can use
         hosts that have nothing to do with CDS.

    Returns a float32 (h, w) array. Pixels outside the survey footprint come
    back as NaN and are carried through — a partially covered field is still
    usable, and a mostly uncovered one is caught by the coverage check in
    survey_gradient_model().
    """
    h, w = int(out_hw[0]), int(out_hw[1])
    qw = _rescaled_wcs(wcs, (h, w), (h, w))
    try:
        qw.pixel_shape = (w, h)
    except Exception:
        pass

    import time

    attempts: list[str] = []
    _log("=" * 62)
    _log(f"fetch_reference  hips={hips_id}  grid={w}x{h}  timeout={timeout}s")
    _log(_wcs_summary(qw, "request"))
    try:
        c = qw.pixel_to_world(w / 2.0, h / 2.0)
        _log(f"field centre : RA={c.ra.deg:.5f}  Dec={c.dec.deg:+.5f}")
    except Exception:
        pass

    try:
        import astroquery.hips2fits  # noqa: F401
        have_aq = True
    except Exception as e:
        have_aq = False
        attempts.append(f"astroquery.hips2fits import failed: {e}")
        _log(f"astroquery.hips2fits UNAVAILABLE: {_brief(e)}")

    # Direct tiles go FIRST. They need nothing but static files, they are the
    # only route that survived the CDS outage, and the reader below is ours —
    # a missing tile is a logged 404, not a silent hole in the data.
    if allow_tiles:
        mirrors = mirrors_for(hips_id)
        if not mirrors:
            _log(f"no direct mirrors registered for {hips_id}")
        for base in mirrors:
            host = base.split("//", 1)[-1].split("/", 1)[0]
            if progress_cb:
                progress_cb(f"Reading reference tiles from {host}…")
            _log(f"-- route: direct tiles @ {host}")
            t0 = time.monotonic()
            try:
                data = _fetch_via_tiles(qw, h, w, base, timeout=timeout)
                dt = (time.monotonic() - t0) * 1000.0
                _log(f"   OK in {dt:.0f} ms")
                if progress_cb:
                    progress_cb(f"Reference built from {host} tiles.")
                return data
            except Exception as e:
                dt = (time.monotonic() - t0) * 1000.0
                _log(f"   FAILED after {dt:.0f} ms: {_brief(e, 300)}")
                attempts.append(f"tiles {host} ({dt:.0f} ms): {_brief(e)}")

    # hips2fits is now the fallback: it is server-side and convenient, but it
    # is a single vendor's service layer and it was the thing that went down.
    if have_aq:
        for ep in _ordered_endpoints():
            host = ep.split("//", 1)[-1].split("/", 1)[0]
            if progress_cb:
                progress_cb(f"Trying hips2fits ({host})…")
            _log(f"-- route: hips2fits @ {host}")
            t0 = time.monotonic()
            try:
                data = _fetch_via_hips2fits(qw, h, w, hips_id, ep, timeout)
                dt = (time.monotonic() - t0) * 1000.0
                _remember_endpoint(ep)
                _log(f"   OK in {dt:.0f} ms")
                if progress_cb:
                    progress_cb(f"Reference received from {host}.")
                return data
            except Exception as e:
                dt = (time.monotonic() - t0) * 1000.0
                _log(f"   FAILED after {dt:.0f} ms: {_brief(e, 300)}")
                attempts.append(f"hips2fits {host} ({dt:.0f} ms): {_brief(e)}")

    _log("ALL ROUTES FAILED")
    _log("=" * 62)
    raise SurveyFetchError(
        "Could not fetch a reference cutout. Every route failed:\n\n"
        + "\n".join(f"  • {a}" for a in attempts)
        + "\n\nIf this is a CDS outage, the DSS2 options also mirror from IRSA "
          "(Caltech), which is independent of CDS."
    )


def _brief(e: Exception, n: int = 140) -> str:
    s = f"{type(e).__name__}: {e}".replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# =============================================================================
#                                 Robust fit
# =============================================================================

def _robust_linear_fit(ref: np.ndarray, tgt: np.ndarray,
                       *, n_iter: int = 5, sigma: float = 2.5
                       ) -> tuple[float, float, np.ndarray]:
    """
    Sigma-clipped least squares for  tgt ≈ a + b·ref.

    Returns (a, b, keep_mask).

    Clipping matters more than usual here. Any pixel where the target has
    structure the reference does not (a satellite trail, a reflection, a
    processing artifact, a galaxy the survey is too shallow to show) would
    otherwise tilt the fit, and a tilted fit becomes a gradient that gets
    subtracted. Clipping pulls those out before they can do that.
    """
    r = np.asarray(ref, dtype=np.float64).ravel()
    t = np.asarray(tgt, dtype=np.float64).ravel()

    keep = np.isfinite(r) & np.isfinite(t)
    if keep.sum() < 32:
        raise SurveyFetchError(
            "Not enough overlapping valid pixels between image and reference "
            "to fit a relation (is the field inside the survey footprint?)."
        )

    a, b = 0.0, 1.0
    for _ in range(int(n_iter)):
        rr, tt = r[keep], t[keep]
        if rr.size < 32:
            break
        # Degenerate reference (flat cutout) -> no structure to match; fall
        # back to an offset-only model so the caller still gets something
        # sane rather than a divide-by-zero slope.
        if float(np.std(rr)) < 1e-12:
            a, b = float(np.median(tt)), 0.0
            break
        A = np.vstack([np.ones_like(rr), rr]).T
        coef, *_ = np.linalg.lstsq(A, tt, rcond=None)
        a, b = float(coef[0]), float(coef[1])

        resid = t - (a + b * r)
        rs = resid[keep]
        med = float(np.median(rs))
        mad = float(np.median(np.abs(rs - med))) * 1.4826
        if mad <= 0:
            break
        new_keep = np.isfinite(r) & np.isfinite(t) & (np.abs(resid - med) < sigma * mad)
        if new_keep.sum() < 32 or new_keep.sum() == keep.sum():
            keep = new_keep if new_keep.sum() >= 32 else keep
            break
        keep = new_keep

    return a, b, keep.reshape(np.asarray(ref).shape)


# =============================================================================
#                              Headless core
# =============================================================================

def survey_gradient_model(
    image: np.ndarray,
    header,
    *,
    hips_id: str = SURVEYS[DEFAULT_SURVEY]["hips"],
    work_long_side: int = 1024,
    degree: int = 1,
    use_rbf: bool = True,
    rbf_smooth: float = 1.0,
    min_coverage: float = 0.25,
    progress_cb=None,
    reference: np.ndarray | None = None,
) -> dict:
    """
    Compute the additive gradient of `image` relative to a survey reference.

    Returns a dict with:
        gradient       (H, W) or (H, W, 3) float32 — YOUR gradient, full res
        reference      (h, w) float32 — the fetched cutout (work scale)
        bg_target      work-scale background model of the image
        bg_ref         work-scale background model of the reference
        coeffs         per-channel (a, b)
        coverage       fraction of the frame covered by the reference
        work_shape     (h, w)

    The caller subtracts `gradient` (optionally scaled by a strength factor).
    Nothing is applied here, so the dialog can preview before committing.
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    mono = (img.ndim == 2)
    H, W = img.shape[:2]

    wcs = _celestial_wcs_from_header(header)
    if wcs is None:
        raise SurveyFetchError(
            "No usable WCS in this document. Plate-solve the image first — "
            "the reference is fetched by sky coordinates."
        )

    h, w = _work_shape(H, W, work_long_side)
    work_wcs = _rescaled_wcs(wcs, (H, W), (h, w))

    # ---- reference ------------------------------------------------------
    if reference is None:
        ref = fetch_reference(work_wcs, (h, w), hips_id, progress_cb=progress_cb)
    else:
        ref = np.asarray(reference, dtype=np.float32)
        if ref.shape[:2] != (h, w):
            raise SurveyFetchError(
                f"Supplied reference shape {ref.shape[:2]} != work grid {(h, w)}."
            )

    valid = np.isfinite(ref)
    coverage = float(valid.mean()) if valid.size else 0.0
    _log(_array_stats(ref, "reference as modelled"))
    _log(f"coverage = {coverage*100:.2f}%  (minimum {min_coverage*100:.0f}%)")

    # Distinguish the failure modes, because "no data covering that area" is a
    # misleading thing to say when the real problem is that the request was
    # malformed and the server handed back an empty frame. Those look identical
    # from the array alone, so say which one it is by HOW empty it came back.
    if coverage < float(min_coverage):
        try:
            c = work_wcs.pixel_to_world(w / 2.0, h / 2.0)
            where = f"Field centre: RA={c.ra.deg:.5f}  Dec={c.dec.deg:+.5f}"
        except Exception:
            where = ""
        if coverage == 0.0:
            raise SurveyFetchError(
                "The reference came back with NO usable pixels at all "
                "(100% non-finite).\n\n"
                f"{where}\n\n"
                "A total blank usually means the request was rejected or "
                "misparsed rather than that the field is uncovered — a "
                "genuine footprint miss on a partially covered survey "
                "normally returns a mix. Check the [SRGC] trace in the log "
                "for the exact URL and WCS that were sent, and compare the "
                "centre above against the survey's footprint."
            )
        raise SurveyFetchError(
            f"The reference covers only {coverage*100:.1f}% of this field "
            f"(minimum {min_coverage*100:.0f}%).\n\n"
            f"{where}\n\n"
            "Partial coverage like this is consistent with the field running "
            "off the edge of the survey. NSNS stops at dec -16; the DSS2 "
            "options are all-sky."
        )

    if np.any(valid):
        v = ref[valid]
        if float(v.min()) == float(v.max()):
            raise SurveyFetchError(
                f"The reference is a constant value ({float(v.min()):.6g}) "
                "across the whole field, so it carries no sky structure to "
                "fit against.\n\nThat is a server or request problem, not a "
                "coverage problem. Check the [SRGC] trace for the URL sent."
            )

    # ---- background models ----------------------------------------------
    # legacy_prestretch is OFF on purpose, for both. The relation we fit below
    # is linear, so both backgrounds have to be modelled in a linear domain.
    # ABE's legacy prestretch is a nonlinear MTF-style mapping; running it here
    # would put a nonlinearity between the two sides of a linear fit.
    if progress_cb:
        progress_cb("Modelling image background…")

    img_small = _downsample_area(img, max(1, int(round(max(H, W) / max(h, w)))))
    if img_small.shape[:2] != (h, w):
        img_small = _resize_to(img_small, (h, w))

    _, bg_t = abe_run(
        img_small, degree=int(degree), num_samples=200, downsample=1,
        patch_size=15, use_rbf=bool(use_rbf), rbf_smooth=float(rbf_smooth),
        return_background=True, legacy_prestretch=False,
        progress_cb=None,
    )

    if progress_cb:
        progress_cb("Modelling reference background…")

    # NaN outside the footprint would poison the fit; fill with the median of
    # the covered part so the model stays smooth, then re-mask afterwards.
    ref_filled = ref.copy()
    if not valid.all():
        fill = float(np.nanmedian(ref)) if np.any(valid) else 0.0
        ref_filled[~valid] = fill

    _, bg_r = abe_run(
        ref_filled, degree=int(degree), num_samples=200, downsample=1,
        patch_size=15, use_rbf=bool(use_rbf), rbf_smooth=float(rbf_smooth),
        return_background=True, legacy_prestretch=False,
        progress_cb=None,
    )
    bg_r = np.asarray(bg_r, dtype=np.float32)
    if bg_r.ndim == 3:
        bg_r = bg_r[..., 0]

    # ---- per-channel fit -------------------------------------------------
    if progress_cb:
        progress_cb("Fitting image background against reference…")

    bg_t = np.asarray(bg_t, dtype=np.float32)
    if mono and bg_t.ndim == 3:
        bg_t = bg_t[..., 0]

    bg_r_masked = np.where(valid, bg_r, np.nan).astype(np.float32)

    coeffs: list[tuple[float, float]] = []
    if mono:
        a, b, _ = _robust_linear_fit(bg_r_masked, bg_t)
        grad_small = (bg_t - (a + b * bg_r)).astype(np.float32)
        grad_small -= float(np.nanmedian(grad_small))
        coeffs.append((a, b))
    else:
        planes = []
        for c in range(bg_t.shape[2]):
            a, b, _ = _robust_linear_fit(bg_r_masked, bg_t[..., c])
            g = (bg_t[..., c] - (a + b * bg_r)).astype(np.float32)
            g -= float(np.nanmedian(g))
            planes.append(g)
            coeffs.append((a, b))
        grad_small = np.stack(planes, axis=-1).astype(np.float32)

    # ---- back to full resolution ----------------------------------------
    if progress_cb:
        progress_cb("Upscaling gradient to full resolution…")
    gradient = _upscale_bg(grad_small, (H, W)).astype(np.float32)
    if mono and gradient.ndim == 3:
        gradient = gradient[..., 0]

    return {
        "gradient": gradient,
        "gradient_small": grad_small,
        "reference": ref,
        "bg_target": bg_t,
        "bg_ref": bg_r,
        "coeffs": coeffs,
        "coverage": coverage,
        "work_shape": (h, w),
    }


def _resize_to(a: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize to an exact (h, w), area-averaged. Falls back to index sampling."""
    h, w = int(hw[0]), int(hw[1])
    if a.shape[:2] == (h, w):
        return a
    if cv2 is not None:
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    ys = np.linspace(0, a.shape[0] - 1, h).astype(int)
    xs = np.linspace(0, a.shape[1] - 1, w).astype(int)
    if a.ndim == 2:
        return a[ys][:, xs].astype(np.float32)
    return np.stack([a[..., c][ys][:, xs] for c in range(a.shape[2])], axis=-1).astype(np.float32)


def apply_survey_gradient(image: np.ndarray, gradient: np.ndarray,
                          strength: float = 1.0) -> np.ndarray:
    """
    Subtract the gradient and restore the original median.

    Additive, always. The whole premise is that what the reference fails to
    explain is additive sky, so dividing it out would be the wrong operator
    (and is the mistake the old SFCC gradient beta made).
    """
    img = np.asarray(image, dtype=np.float32)
    g = np.asarray(gradient, dtype=np.float32)
    if img.ndim == 3 and g.ndim == 2:
        g = g[..., None]
    if img.ndim == 2 and g.ndim == 3:
        g = g[..., 0]

    med_before = float(np.nanmedian(img))
    out = img - float(strength) * g
    med_after = float(np.nanmedian(out))
    out = out + (med_before - med_after)

    mn = float(np.nanmin(out))
    if mn < 0.0:
        out = out - mn
    mx = float(np.nanmax(out))
    if mx > 1.0:
        p = med_before - min(0.0, mn)
        denom = max(mx - p, 1e-8)
        out = p + (out - p) * ((1.0 - p) / denom)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


# =============================================================================
#                                   Dialog
# =============================================================================

class SurveyGradientDialog(QDialog):
    """
    Fetch a survey reference for the current field, model both backgrounds,
    and remove the difference.

    Deliberately NOT folded into ABE: this is reference-based correction and
    it has a hard prerequisite ABE does not (a plate solution) plus a network
    dependency. Keeping it separate means ABE stays offline and unconditional.
    """

    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Survey Reference Gradient Correction (beta)"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass

        self._main = parent
        self.doc = document
        self._result: dict | None = None
        self._preview_buf = None

        # Mirror the [SRGC] trace into the main window log so the user can
        # copy it out of the app rather than hunting for a console.
        _hostlog = getattr(parent, "_log", None)
        if callable(_hostlog):
            set_log_sink(_hostlog)

        self.cmb_survey = QComboBox()
        for name in SURVEYS:
            self.cmb_survey.addItem(name, userData=SURVEYS[name]["hips"])
        self.cmb_survey.currentTextChanged.connect(self._survey_changed)

        self.sp_work = QSpinBox()
        self.sp_work.setRange(256, 4096)
        self.sp_work.setSingleStep(256)
        self.sp_work.setValue(1024)
        self.sp_work.setToolTip(
            "Long side of the grid the reference is fetched on and both\n"
            "backgrounds are modelled at.\n\n"
            "A gradient is low-frequency by definition and NSNS is ~10 arcsec,\n"
            "so there is nothing above this scale to recover. Larger is slower\n"
            "and makes the fetch heavier without improving the correction."
        )

        self.sp_degree = QSpinBox()
        self.sp_degree.setRange(0, 4)
        self.sp_degree.setValue(1)
        self.sp_degree.setToolTip(
            "Polynomial degree for BOTH background models.\n\n"
            "Keep this low. A high-degree model on the target can absorb real\n"
            "structure before the reference ever gets a chance to explain it,\n"
            "which defeats the point of this tool. Degree 1-2 is right."
        )

        self.chk_rbf = QCheckBox(self.tr("RBF refinement on both backgrounds"))
        self.chk_rbf.setChecked(True)

        self.sp_strength = QSpinBox()
        self.sp_strength.setRange(0, 100)
        self.sp_strength.setValue(100)
        self.sp_strength.setSuffix(" %")

        form = QFormLayout()
        form.addRow(self.tr("Reference survey:"), self.cmb_survey)
        form.addRow(self.tr("Work resolution:"), self.sp_work)
        form.addRow(self.tr("Background degree:"), self.sp_degree)
        form.addRow(self.chk_rbf)
        form.addRow(self.tr("Strength:"), self.sp_strength)

        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet(
            "font-size: 10px; color: palette(window-text); "
            "background: palette(base); border: 1px solid palette(mid); "
            "border-radius: 3px; padding: 4px;"
        )

        gb = QGroupBox(self.tr("Reference"))
        gb.setLayout(form)

        self.lbl_preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setMinimumSize(QSize(420, 300))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.lbl_preview)

        self.cmb_view = QComboBox()
        self.cmb_view.addItems([
            "Corrected", "Gradient removed", "Reference", "Reference background",
        ])
        self.cmb_view.currentTextChanged.connect(self._redraw)

        self.btn_run = QPushButton(self.tr("Fetch && Model"))
        self.btn_apply = QPushButton(self.tr("Apply"))
        self.btn_diag = QPushButton(self.tr("Run diagnostics"))
        self.btn_diag.setToolTip(
            "Probe every route for the selected survey with a tiny cutout and\n"
            "report what each one actually returned.\n\n"
            "Use this when a fetch fails or comes back empty — it separates a\n"
            "dead host from a rejected request from a genuine footprint miss,\n"
            "which all look the same from the main error."
        )
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_apply.setEnabled(False)
        self.btn_run.clicked.connect(self._do_run)
        self.btn_apply.clicked.connect(self._do_apply)
        self.btn_diag.clicked.connect(self._do_diagnostics)
        self.btn_close.clicked.connect(self.close)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)

        rowv = QHBoxLayout()
        rowv.addWidget(QLabel(self.tr("View:")))
        rowv.addWidget(self.cmb_view, 1)

        rowb = QHBoxLayout()
        rowb.addWidget(self.btn_run)
        rowb.addWidget(self.btn_apply)
        rowb.addStretch(1)
        rowb.addWidget(self.btn_close)

        rowd = QHBoxLayout()
        rowd.addWidget(self.btn_diag)
        rowd.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(gb)
        left.addWidget(self.lbl_note)
        left.addLayout(rowv)
        left.addLayout(rowb)
        left.addLayout(rowd)
        left.addWidget(self.status)
        left.addStretch(1)

        main = QHBoxLayout(self)
        lw = QVBoxLayout()
        lw.addLayout(left)
        main.addLayout(lw, 0)
        main.addWidget(self.scroll, 1)

        self.resize(1000, 620)
        self._load_settings()
        self._survey_changed(self.cmb_survey.currentText())

    # ---- settings -------------------------------------------------------
    def _load_settings(self):
        s = QSettings()
        name = str(s.value("srgc/survey", DEFAULT_SURVEY))
        if self.cmb_survey.findText(name) >= 0:
            self.cmb_survey.setCurrentText(name)
        self.sp_work.setValue(int(s.value("srgc/work", 1024)))
        self.sp_degree.setValue(int(s.value("srgc/degree", 1)))
        self.chk_rbf.setChecked(bool(s.value("srgc/rbf", True, type=bool)))
        self.sp_strength.setValue(int(s.value("srgc/strength", 100)))

    def _save_settings(self):
        s = QSettings()
        s.setValue("srgc/survey", self.cmb_survey.currentText())
        s.setValue("srgc/work", self.sp_work.value())
        s.setValue("srgc/degree", self.sp_degree.value())
        s.setValue("srgc/rbf", self.chk_rbf.isChecked())
        s.setValue("srgc/strength", self.sp_strength.value())

    def _survey_changed(self, name: str):
        info = SURVEYS.get(name)
        if info:
            self.lbl_note.setText(info["note"])
        self._save_settings()

    def _set_status(self, text: str):
        try:
            self.status.setText(text)
            QApplication.processEvents()
        except Exception:
            pass

    # ---- data -----------------------------------------------------------
    def _source_float(self) -> np.ndarray | None:
        src = np.asarray(self.doc.image)
        if src is None or src.size == 0:
            return None
        if np.issubdtype(src.dtype, np.integer):
            return src.astype(np.float32) / float(np.iinfo(src.dtype).max)
        return src.astype(np.float32, copy=False)

    def _header(self):
        meta = getattr(self.doc, "metadata", None) or {}
        return (meta.get("wcs_header") or meta.get("original_header")
                or meta.get("header"))

    # ---- run ------------------------------------------------------------
    def _do_run(self):
        img = self._source_float()
        if img is None:
            QMessageBox.information(self, "No image", "No image in the active document.")
            return

        hdr = self._header()
        if hdr is None:
            QMessageBox.warning(
                self, "No plate solution",
                "This document has no WCS. Plate-solve it first — the reference "
                "is fetched by sky coordinates."
            )
            return

        self._save_settings()
        self.btn_run.setEnabled(False)
        self.btn_apply.setEnabled(False)
        try:
            self._result = survey_gradient_model(
                img, hdr,
                hips_id=self.cmb_survey.currentData(),
                work_long_side=int(self.sp_work.value()),
                degree=int(self.sp_degree.value()),
                use_rbf=bool(self.chk_rbf.isChecked()),
                progress_cb=self._set_status,
            )
        except SurveyFetchError as e:
            self._set_status("Failed")
            QMessageBox.warning(self, "Survey reference", str(e))
            self.btn_run.setEnabled(True)
            return
        except Exception as e:
            self._set_status("Error")
            QMessageBox.critical(self, "Survey gradient failed", repr(e))
            self.btn_run.setEnabled(True)
            return

        self.btn_run.setEnabled(True)
        self.btn_apply.setEnabled(True)

        cov = self._result["coverage"] * 100.0
        cs = ", ".join(f"b={b:.3g}" for _, b in self._result["coeffs"])
        g = self._result["gradient"]
        self._set_status(
            f"Coverage {cov:.1f}%  ·  work {self._result['work_shape'][1]}×"
            f"{self._result['work_shape'][0]}  ·  {cs}  ·  "
            f"gradient peak-to-peak {float(np.nanmax(g) - np.nanmin(g)):.4f}"
        )
        self._redraw()

    # ---- diagnostics ----------------------------------------------------
    def _do_diagnostics(self):
        """
        Probe every route for the selected survey and report what each one
        actually did.

        The point is to separate failure modes that the normal error cannot:
        a host that never answered, a host that answered and rejected the
        request, a host that answered with an empty frame, and a field that is
        genuinely off the edge of the survey. Those produce four very different
        fixes and one identical-looking error.
        """
        import time
        global _LOG_SINK

        lines: list[str] = []
        prev_sink = _LOG_SINK

        def tee(msg: str):
            lines.append(msg)
            if prev_sink is not None:
                try:
                    prev_sink(msg)
                except Exception:
                    pass

        set_log_sink(tee)
        self._set_status("Running diagnostics…")
        try:
            lines.append("=" * 64)
            lines.append("SRGC DIAGNOSTICS")
            lines.append("=" * 64)

            # -- environment
            for mod in ("astropy", "astroquery", "reproject"):
                try:
                    m = __import__(mod)
                    lines.append(f"{mod:<12} {getattr(m, '__version__', '?')}")
                except Exception as e:
                    lines.append(f"{mod:<12} MISSING ({_brief(e)})")
            try:
                from reproject.hips import hips_as_dask_array  # noqa: F401
                lines.append("reproject.hips available   : yes "
                             "(direct-tile fallback usable)")
            except Exception as e:
                lines.append(f"reproject.hips available   : NO ({_brief(e)}) "
                             "— direct-tile fallback disabled, bump reproject")

            # -- document / WCS
            img = self._source_float()
            hdr = self._header()
            lines.append("")
            if img is None:
                lines.append("image: NONE")
                raise RuntimeError("no image")
            lines.append(f"image shape                : {img.shape} {img.dtype}")
            if hdr is None:
                lines.append("header                     : NONE — not plate-solved")
                raise RuntimeError("no header")

            wcs = _celestial_wcs_from_header(hdr)
            if wcs is None:
                lines.append("WCS                        : FAILED TO PARSE")
                raise RuntimeError("no wcs")

            H, W = img.shape[:2]
            h, w = _work_shape(H, W, int(self.sp_work.value()))
            work = _rescaled_wcs(wcs, (H, W), (h, w))
            lines.append(_wcs_summary(wcs, "document"))
            lines.append(_wcs_summary(work, "work grid"))
            try:
                c = work.pixel_to_world(w / 2.0, h / 2.0)
                lines.append(f"field centre               : "
                             f"RA={c.ra.deg:.5f}  Dec={c.dec.deg:+.5f}")
                info = SURVEYS.get(self.cmb_survey.currentText(), {})
                dmin = info.get("dec_min")
                if dmin is not None:
                    inside = c.dec.deg >= float(dmin)
                    lines.append(f"survey dec limit           : >= {dmin:+.0f}  "
                                 f"-> field is {'INSIDE' if inside else 'OUTSIDE'}")
            except Exception as e:
                lines.append(f"field centre               : ? ({_brief(e)})")

            # -- probe every route with a small cutout
            hips_id = self.cmb_survey.currentData()
            ph, pw = 64, 64
            probe = _rescaled_wcs(wcs, (H, W), (ph, pw))
            try:
                probe.pixel_shape = (pw, ph)
            except Exception:
                pass

            lines.append("")
            lines.append(f"probing routes for: {hips_id}   ({pw}x{ph} cutout)")
            lines.append("-" * 64)

            any_ok = False
            for ep in _ordered_endpoints():
                host = ep.split("//", 1)[-1].split("/", 1)[0]
                t0 = time.monotonic()
                try:
                    a = _fetch_via_hips2fits(probe, ph, pw, hips_id, ep,
                                             HIPS_TIMEOUT)
                    dt = (time.monotonic() - t0) * 1000
                    lines.append(f"hips2fits {host}")
                    lines.append(f"   OK {dt:.0f} ms  {_array_stats(a)}")
                    any_ok = True
                except Exception as e:
                    dt = (time.monotonic() - t0) * 1000
                    lines.append(f"hips2fits {host}")
                    lines.append(f"   FAIL {dt:.0f} ms  {_brief(e, 300)}")
                QApplication.processEvents()

            for base in mirrors_for(hips_id):
                host = base.split("//", 1)[-1].split("/", 1)[0]
                t0 = time.monotonic()
                try:
                    a = _fetch_via_tiles(probe, ph, pw, base)
                    dt = (time.monotonic() - t0) * 1000
                    lines.append(f"tiles {host}")
                    lines.append(f"   OK {dt:.0f} ms  {_array_stats(a)}")
                    any_ok = True
                except Exception as e:
                    dt = (time.monotonic() - t0) * 1000
                    lines.append(f"tiles {host}")
                    lines.append(f"   FAIL {dt:.0f} ms  {_brief(e, 300)}")
                QApplication.processEvents()

            lines.append("-" * 64)
            lines.append("at least one route returned data: "
                         f"{'YES' if any_ok else 'NO'}")
        except Exception as e:
            lines.append(f"\ndiagnostics stopped early: {_brief(e)}")
        finally:
            set_log_sink(prev_sink)
            self._set_status("Diagnostics complete.")

        self._show_report("\n".join(lines))

    def _show_report(self, text: str):
        dlg = QDialog(self)
        dlg.setWindowTitle(self.tr("SRGC diagnostics"))
        dlg.resize(900, 600)
        v = QVBoxLayout(dlg)
        box = QPlainTextEdit()
        box.setPlainText(text)
        box.setReadOnly(True)
        try:
            from PyQt6.QtGui import QFont
            box.setFont(QFont("Courier New", 9))
        except Exception:
            pass
        v.addWidget(box, 1)
        row = QHBoxLayout()
        btn_copy = QPushButton(self.tr("Copy to clipboard"))

        def _copy():
            try:
                QApplication.clipboard().setText(text)
                btn_copy.setText(self.tr("Copied"))
            except Exception:
                pass

        btn_copy.clicked.connect(_copy)
        btn_ok = QPushButton(self.tr("Close"))
        btn_ok.clicked.connect(dlg.accept)
        row.addWidget(btn_copy)
        row.addStretch(1)
        row.addWidget(btn_ok)
        v.addLayout(row)
        dlg.exec()

    # ---- preview --------------------------------------------------------
    def _redraw(self, *_):
        if self._result is None:
            return
        mode = self.cmb_view.currentText()
        img = self._source_float()
        if img is None:
            return

        if mode == "Corrected":
            arr = apply_survey_gradient(img, self._result["gradient"],
                                        self.sp_strength.value() / 100.0)
        elif mode == "Gradient removed":
            g = np.asarray(self._result["gradient_small"], dtype=np.float32)
            arr = _norm01(g)
        elif mode == "Reference":
            arr = _norm01(np.nan_to_num(self._result["reference"],
                                        nan=float(np.nanmedian(self._result["reference"]))))
        else:
            arr = _norm01(np.asarray(self._result["bg_ref"], dtype=np.float32))

        try:
            disp = hard_autostretch(arr, target_median=0.4, sigma=2,
                                    linked=False, use_24bit=True)
            disp = np.asarray(disp, dtype=np.float32)
        except Exception:
            disp = np.clip(arr, 0.0, 1.0)

        if disp.ndim == 2:
            disp = np.repeat(disp[..., None], 3, axis=2)
        buf = np.ascontiguousarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))
        self._preview_buf = buf
        h, w, _ = buf.shape
        qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaled(
            self.lbl_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.lbl_preview.setPixmap(pm)

    # ---- apply ----------------------------------------------------------
    def _do_apply(self):
        if self._result is None:
            return
        img = self._source_float()
        if img is None:
            return
        strength = self.sp_strength.value() / 100.0
        out = apply_survey_gradient(img, self._result["gradient"], strength)

        name = self.cmb_survey.currentText()
        step = f"Survey Gradient ({name}, {int(strength*100)}%)"
        meta = {
            "step_name": "SurveyGradient",
            "survey_gradient": {
                "survey": name,
                "hips": self.cmb_survey.currentData(),
                "work_long_side": int(self.sp_work.value()),
                "degree": int(self.sp_degree.value()),
                "rbf": bool(self.chk_rbf.isChecked()),
                "strength": strength,
                "coverage": float(self._result["coverage"]),
                "coeffs": [[float(a), float(b)] for a, b in self._result["coeffs"]],
            },
        }
        try:
            self.doc.apply_edit(out.astype(np.float32, copy=False),
                                step_name=step, metadata=meta)
        except Exception as e:
            QMessageBox.critical(self, "Apply failed", str(e))
            return

        mw = self._main
        if hasattr(mw, "_log"):
            try:
                mw._log(f"[SRGC] {step} coverage={self._result['coverage']*100:.1f}%")
            except Exception:
                pass
        self._set_status("Applied.")


def _norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return ((a - mn) / (mx - mn)).astype(np.float32)


def open_survey_gradient(parent, document) -> SurveyGradientDialog:
    dlg = SurveyGradientDialog(parent, document)
    dlg.show()
    return dlg