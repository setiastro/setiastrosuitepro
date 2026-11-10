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

from PyQt6.QtCore import (Qt, QSize, QSettings, QThread, pyqtSignal,
                          QPoint, QPointF, QRectF)
from PyQt6.QtGui import (QImage, QPixmap, QIcon, QPainter, QPen, QBrush,
                         QColor, QPolygonF)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QGroupBox, QMessageBox,
    QApplication, QScrollArea, QPlainTextEdit, QFileDialog, QWidget,
    QTabWidget, QToolButton, QSizePolicy,
)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from astropy.wcs import WCS

from setiastro.saspro.abe import (
    abe_run, _downsample_area, _upscale_bg,
    _generate_sample_points, _to_luminance, _exclude_bright_regions,
)

try:
    # median + k*MAD -> dilate -> blur. Same protection ADBE's multiscale
    # stage uses, so "protect k" means the same thing in both tools.
    from setiastro.saspro.abe_multiscale_gradient import build_signal_mask
except Exception:  # pragma: no cover
    build_signal_mask = None
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

# Connect pre-flight budget, seconds. SEPARATE from HIPS_TIMEOUT, and it has
# to be: HIPS_TIMEOUT is handed to urllib, which applies it as a socket
# timeout only once a connection is being established. A host that is simply
# not answering is capped by the OS instead -- ~21 s on Windows (WinError
# 10060) -- so a 30 s read timeout never gets the chance to fire, and three
# dead CDS hosts cost ~63 s before the first error reaches the user.
#
# One TCP connect with an explicit short timeout settles the question in
# under three seconds, and the answer is cached per session.
CONNECT_TIMEOUT = 2.5

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

# How each tree encodes surface brightness. SRGC fits  tgt = a + b*ref,
# which absorbs any LINEAR scale and offset exactly (measured: a target or
# reference rescaled by 1e4 changes the recovered gradient by ~1e-8, i.e.
# float32 round-off). What it cannot absorb is curvature -- though over a
# background's narrow range even a log is close to its own tangent, so the
# practical cost is small. This table exists so the tool can SAY what it
# fetched instead of leaving the user to wonder, which is the actual
# complaint it answers.
#
#   linear   - counts proportional to surface brightness. Nothing to say.
#   log      - a monotone compressive stretch applied by the tree builder.
#              Mostly absorbed into b; amplitude slightly soft.
#   highpass - large scales REMOVED. This is the one real hazard, because
#              the removed frequencies are exactly the ones being fitted.
#              reference_share measures the damage directly.
_TREE_ENCODING: dict[str, tuple[str, str]] = {
    "CDS/P/DSS2/color": (
        "linear",
        "CDS built this from the DSS2 FITS planes with linear cuts "
        "(recorded in its properties), so it is the plate, scaled and "
        "quantised to 8 bits."),
    "CDS/P/SDSS9/color": (
        "log",
        "CDS applied a Log stretch building this composite. A monotone "
        "curve over a background's narrow range is close to linear and is "
        "largely absorbed into the fitted scale factor."),
}


def _tree_encoding(hips_id: str) -> tuple[str, str]:
    """(kind, why) for a tree. NSNS is matched by prefix, not by layer."""
    hid = str(hips_id or "")
    hit = _TREE_ENCODING.get(hid)
    if hit is not None:
        return hit
    if "NSNS" in hid:
        return ("highpass",
                "NSNS states these mosaics are dynamic-range compressed "
                "(non-linearly high-pass filtered) and tonal-curve "
                "corrected. A high-pass removes the large scales this "
                "tool fits, so check the survey/yours split on the "
                "run summary.")
    return ("linear", "")


def _nsns(layer: str) -> dict:
    return {
        "hips": f"simg.de/P/NSNS/DR0_2/{layer}",
        "mirrors": [
            f"http://www.simg.de/nebulae3/dr0_2/{layer}",
            f"https://alaskybis.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_{layer}",
            f"https://alasky.cds.unistra.fr/simg.de/simg.de_P_NSNS_DR0_2_{layer}",
        ],
    }


# NOT skies.esac.esa.int/DSSColor — jpeg tiles only. An 8-bit JPEG of a colour
# composite would reproject happily and produce a confident, wrong background.
# Mirror hosts. These are MEASURED, not guessed — probe_mirrors.py walked the
# candidate matrix from a real network on 2026-11-09 and the results below are
# what actually answered:
#
#   UP   : irsa, casda, esa, simg, camras, irap, svo-cab
#   DOWN : alasky, alaskybis (CDS outage), healpix.ias.u-psud.fr AND
#          wwwfirback.ias.u-psud.fr (both IAS hostnames timed out, so IAS is
#          not the CDS-independent fallback it looked like on paper and has
#          been removed rather than left here to cost a timeout)
#
# Coverage is per-survey, not per-node, and the gaps are real:
#   DSS2 red   : IRSA, FITS, order 9, tile fetch verified
#   DSS2 blue  : NOT MIRRORED ANYWHERE REACHABLE. IRSA, CASDA and ESA all 404.
#                Only the jpeg `color` composite is mirrored, which is useless
#                here. So while CDS is down there is no blue plate, which is
#                what the per-channel degradation in fetch_reference_channels
#                exists to survive.
#
# That asymmetry is exactly why a 404 retires one URL and never the host.
#
# The alasky static trees use plate names rather than band names (DSS2Merged
# for red, DSS2-blue-XJ-S for blue), so they cannot be generated from the band.
def _dss2(band: str, alasky_name: str) -> dict:
    return {
        "hips": f"CDS/P/DSS2/{band}",
        "mirrors": [
            f"https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/{band}",
            f"https://casda.csiro.au/hips/DSS2/{band}",
            f"https://alaskybis.cds.unistra.fr/DSS/{alasky_name}",
            f"https://alasky.cds.unistra.fr/DSS/{alasky_name}",
        ],
    }


DSS2_RED = _dss2("red", "DSS2Merged")
DSS2_BLUE = _dss2("blue", "DSS2-blue-XJ-S")


# The DSS2 colour tree, as 8-bit tiles. CDS builds it from the two FITS plate
# surveys with LINEAR cuts, recorded in its own properties:
#     hips_rgb_red  = DSS2Merged      [1488.0  8488.8125 14666.0 Linear]
#     hips_rgb_blue = DSS2-blue-XJ-S  [4286.0 12122.5   19959.0 Linear]
# so band 0 is the red plate and band 2 is the blue plate, each linearly
# scaled into 0-255. While the blue FITS tree is mirrored nowhere reachable,
# this is the only real blue available anywhere, and for fitting a
# low-frequency background it is entirely adequate.
# Band 1 is CDS's synthetic green, which is a blend of the other two rather
# than an independent measurement.
def _dss2_colour(band: int) -> dict:
    return {
        "hips": "CDS/P/DSS2/color",
        "channel": band,
        "mirrors": [
            "https://irsa.ipac.caltech.edu/data/hips/CDS/DSS2/color",
            "https://casda.csiro.au/hips/DSS2/color",
            "https://skies.esac.esa.int/DSSColor",
            "https://alaskybis.cds.unistra.fr/DSS/DSSColor",
            "https://alasky.cds.unistra.fr/DSS/DSSColor",
        ],
    }


# SDSS9 colour, 8-bit, order 10. Built by CDS from the g/r/i FITS with a Log
# stretch (hips_rgb_red = SDSS9 i [... Log], green = r, blue = g). A monotonic
# curve over the narrow range a sky background occupies is close to linear and
# is largely absorbed into b, so this is usable — softer than the real bands
# would be, but it is band-matched to NSNS and it is reachable today.
def _sdss9_colour(band: int) -> dict:
    return {
        "hips": "CDS/P/SDSS9/color",
        "channel": band,
        "mirrors": [
            "https://irsa.ipac.caltech.edu/data/hips/CDS/SDSS/DR9/color",
            "https://alaskybis.cds.unistra.fr/SDSS/DR9/color",
            "https://alasky.cds.unistra.fr/SDSS/DR9/color",
        ],
    }


# SDSS9 per-band HiPS, FITS tiles, BITPIX 16, order 10 (0.40"/px).
#
# This is the band-matched partner to NSNS: the NSNS continuum cameras were
# calibrated in the SDSS g, r and i bands (Northern Sky Narrowband Survey
# DR0.2; see the North Polar Spur paper using DR0.2, which states the visual
# continuum was obtained in SDSS g/r/i). So pairing NSNS luminance with SDSS
# chrominance keeps both legs in the same photometric system, which DSS2
# photographic plates cannot do.
#
# Mapping follows CDS's own SDSS9 colour composite — i to red, r to green,
# g to blue (its properties file reads
#   hips_rgb_red = SDSS9 i / hips_rgb_green = SDSS9 r / hips_rgb_blue = SDSS9 g
# with hipsgen inBlue=band-g inGreen=band-r inRed=band-i).
#
# COVERAGE, and it matters: SDSS imaging is 36.2% of sky (14,555 deg^2),
# concentrated in the north galactic cap plus the equatorial stripes. It is
# NOT a southern survey. Out of footprint the fetch returns non-finite and
# the caller falls back to DSS2, which is why DSS2 stays in the registry.
#
# IRSA namespaces SDSS differently from DSS2. Its own hipslist gives
#   ivo://CDS/P/SDSS9/color -> /data/hips/CDS/SDSS/DR9/color
# i.e. "CDS/" prepended to the alasky static path (SDSS/DR9/...), whereas for
# DSS2 it uses a tidied CDS/DSS2/<band>. Probing CDS/SDSS9/<band> and
# SDSS/DR9/band-<band> both 404; the combination below was never tried and is
# the one the node's own list implies. Still UNVERIFIED for the per-band trees
# — only `color` is confirmed present, and IRSA's whole hipslist is 14 KB, so
# it may genuinely hold no SDSS bands at all. A wrong guess costs one 404 at
# ~220 ms, and probe_mirrors.py --only sdss settles it.
def _sdss9(band: str) -> dict:
    return {
        "hips": f"CDS/P/SDSS9/{band}",
        "mirrors": [
            f"https://irsa.ipac.caltech.edu/data/hips/CDS/SDSS/DR9/band-{band}",
            f"https://irsa.ipac.caltech.edu/data/hips/CDS/SDSS/DR9/{band}",
            f"https://casda.csiro.au/hips/SDSS/DR9/band-{band}",
            f"https://alaskybis.cds.unistra.fr/SDSS/DR9/band-{band}",
            f"https://alasky.cds.unistra.fr/SDSS/DR9/band-{band}",
        ],
    }


SDSS9_G = _sdss9("g")
SDSS9_R = _sdss9("r")
SDSS9_I = _sdss9("i")

# `channels` is a list of 1 (mono) or 3 (R,G,B) entries. "synth" means the
# plane is the mean of the fetched neighbours — DSS2 has red and blue plates
# but no green, and a gradient correction on colour data needs all three.
SURVEYS: dict[str, dict] = {
    "DSS2 colour (red / synth / blue) — all sky": {
        "channels": [DSS2_RED, "synth", DSS2_BLUE],
        "note": "All-sky, both hemispheres, FITS. Green is synthesised as the "
                "mean of red and blue — crude, but it gives each of your "
                "channels its own reference. ODbL, no licence restriction.",
        "dec_min": -90.0,
        "noncommercial": False,
    },
    "DSS2 red + NSNS continuum (north) — colour": {
        "channels": [DSS2_RED, _nsns("vc"), _nsns("vc")],
        "note": "Best colour reference reachable during the CDS outage. The "
                "red plate drives red; NSNS star-subtracted continuum drives "
                "green and blue. Mixing surveys is fine here because each "
                "channel solves its own offset and scale — only the SHAPE of "
                "the structure has to be right per channel. North of -16 "
                "only, since NSNS stops there. NSNS is CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "DSS2 colour from 8-bit tiles (all sky) — colour": {
        "channels": [_dss2_colour(0), _dss2_colour(1), _dss2_colour(2)],
        "note": "Real red AND real blue plates, linearly scaled into 8-bit by "
                "CDS. 256 levels is ample for a low-frequency background, and "
                "this is the only route to the blue plate while CDS is down. "
                "Green is CDS's synthetic blend, not a measurement. All sky, "
                "both hemispheres. ODbL.",
        "dec_min": -90.0,
        "noncommercial": False,
    },
    "SDSS colour from 8-bit tiles (north) — colour": {
        "channels": [_sdss9_colour(0), _sdss9_colour(1), _sdss9_colour(2)],
        "note": "i/r/g mapped to R/G/B, band-matched to the NSNS continuum, "
                "from CDS's 8-bit composite. Log-stretched rather than "
                "linear, which a background's narrow range mostly absorbs "
                "into the scale factor. 36% of sky, north galactic cap. "
                "Check the survey/yours structure split on this one.",
        "dec_min": -25.0,
        "noncommercial": False,
    },
    "SDSS g/r/i colour (north, 36% of sky) — colour": {
        "channels": [SDSS9_I, SDSS9_R, SDSS9_G],
        "note": "Band-matched broadband colour: real photometry in the same "
                "g/r/i system the NSNS continuum was calibrated in, mapped "
                "i/r/g to R/G/B as CDS does. Only 36% of sky — north galactic "
                "cap plus the equatorial stripes — so outside the footprint "
                "fall back to DSS2 colour. No licence restriction.",
        "dec_min": -25.0,
        "noncommercial": False,
    },
    "NSNS visual continuum (north) — mono": {
        "channels": [_nsns("vc")],
        "note": "NSNS star-subtracted visible continuum, mono (it is already "
                "an R+G+B combination). Each of your channels still gets its "
                "own scale factor. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS SHO (Ha / OIII / SII) — colour": {
        "channels": [_nsns("halpha"), _nsns("oiii"), _nsns("sii")],
        "note": "Narrowband colour reference: Ha, [OIII], [SII] as R, G, B. "
                "For SHO-mapped data, so each channel is matched to the line "
                "it was actually imaged in. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    # One-shot-colour cameras behind a dual narrowband filter. The Bayer
    # matrix sorts the two passbands for you: Ha (656 nm) lands almost
    # entirely on the red pixels, [OIII] (501 nm) on green and blue. So the
    # frame IS an HOO composite, and its reference has to be one too —
    # correcting an HOO frame against a continuum plane would fit the gradient
    # in each channel against structure that is not in that channel.
    #
    # SOO is the same arrangement for a filter passing [SII] rather than Ha.
    #
    # Both reference the same [OIII] tree twice; fetch_reference_channels
    # dedupes by hips id so it is downloaded once.
    #
    # CAVEAT, and read this before trusting the result: the NSNS line maps are
    # continuum-SUBTRACTED, and your dual-band frame is not. Whatever continuum
    # leaks through the filter passband (wider filter, more leak) is real sky
    # structure in your frame with no counterpart in the reference, so a
    # single-regressor fit can attribute it to the gradient and subtract real
    # signal — exactly the failure this tool exists to avoid. See
    # CONTINUUM_CO_REGRESSOR below.
    "NSNS HOO (Ha / OIII / OIII) — OSC dual-band, north": {
        "channels": [_nsns("halpha"), _nsns("oiii"), _nsns("oiii")],
        "note": "For one-shot-colour data behind a dual narrowband filter "
                "passing Ha and [OIII] (L-eXtreme, L-Ultimate, ALP-T, NBZ…). "
                "Ha to red, [OIII] to green and blue, matching what the Bayer "
                "matrix did to your frame. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS SOO (SII / OIII / OIII) — OSC dual-band, north": {
        "channels": [_nsns("sii"), _nsns("oiii"), _nsns("oiii")],
        "note": "As HOO, for a dual-band filter passing [SII] rather than Ha. "
                "CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS Ha (north) — mono": {
        "channels": [_nsns("halpha")],
        "note": "NSNS H-alpha, linear intensity in Rayleighs. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS [OIII] (north) — mono": {
        "channels": [_nsns("oiii")],
        "note": "NSNS [OIII] 500.7 nm. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "NSNS [SII] (north) — mono": {
        "channels": [_nsns("sii")],
        "note": "NSNS [SII] 671.7/673.0 nm. CC-BY-NC-SA.",
        "dec_min": -16.0,
        "noncommercial": True,
    },
    "DSS2 red (all sky) — mono": {
        "channels": [DSS2_RED],
        "note": "All-sky red plates, FITS. ODbL.",
        "dec_min": -90.0,
        "noncommercial": False,
    },
    "DSS2 blue (all sky) — mono": {
        "channels": [DSS2_BLUE],
        "note": "All-sky blue plates, FITS. ODbL.",
        "dec_min": -90.0,
        "noncommercial": False,
    },
}

DEFAULT_SURVEY = "NSNS visual continuum (north) — mono"


# Measured availability. probe_mirrors.py walked every candidate path from an
# ordinary consumer connection on 2026-11-09 and then read each node's own
# hipslist, so this is an inventory rather than an inference.
#
# IRSA's HiPS node holds 27 surveys in total. Its complete optical holdings:
#     CDS/P/DSS2/red     -> /data/hips/CDS/DSS2/red      jpeg fits   <-- ONLY
#     CDS/P/DSS2/color   -> /data/hips/CDS/DSS2/color    jpeg
#     CDS/P/SDSS9/color  -> /data/hips/CDS/SDSS/DR9/color jpeg, order 10
# plus 2MASS, DENIS, GALEX, IRIS, MSX, Planck, Spitzer, AllWISE, Herschel.
# There are NO SDSS band trees and NO DSS2 blue. CASDA holds 9 surveys
# (2MASS, DSS2 color, HI4PI, NVSS, Spitzer, SUMSS, RACS) with no optical
# FITS at all. ESA, CAMRAS, IRAP and SVO-CAB have nothing optical either,
# and both IAS hostnames are unreachable.
#
# So while CDS is down, the entire set of usable optical FITS references on
# the open internet is five trees: NSNS vc/halpha/oiii/sii, and DSS2 red.
# Everything else here is waiting on Strasbourg to come back.
#
# This is surfaced in the dialog rather than left for the user to discover by
# watching a fetch fail.
_AVAILABILITY = {
    "verified": "",
    "degraded": (
        "⚠ DEGRADED while CDS is down: the blue plate is mirrored nowhere "
        "reachable, so this runs on the red plate alone and the other two "
        "channels are synthesised from it. Each channel still solves its own "
        "scale factor, but there is no real colour information in the "
        "reference. "
    ),
    "unavailable": (
        "⚠ UNAVAILABLE while CDS is down: no mirror carries this as FITS "
        "(IRSA has only a jpeg composite, CASDA and ESA have nothing). "
        "Left here because it becomes the right choice when Strasbourg "
        "returns. "
    ),
}

_SURVEY_AVAILABILITY = {
    "DSS2 colour (red / synth / blue) — all sky":        "degraded",
    "SDSS g/r/i colour (north, 36% of sky) — colour":    "unavailable",
    "DSS2 blue (all sky) — mono":                        "unavailable",
    "SDSS colour from 8-bit tiles (north) — colour":     "stretched",
}

# The state a liveness ping cannot see, and the reason this module checks
# /properties instead of pinging. A node that answers again after an outage
# has NOT necessarily restored the same tile trees at the same paths, and
# silently re-enabling a survey on that evidence would hand the user a
# confident option that fails in a new way.
_AVAILABILITY["moved"] = (
    "⚠ The mirror host is answering again but its tile tree is no longer at "
    "the registered path. The survey is probably back under a different URL "
    "— re-run probe_mirrors.py to find it. "
)

_AVAILABILITY["stretched"] = (
    "ℹ Log-stretched 8-bit source. Quantisation is irrelevant at these "
    "frequencies, but the stretch is only approximately linear over the "
    "background's range, so check the survey/yours structure split on "
    "the run summary. "
)

# Prefer a reference that actually fetches today. NSNS continuum stays the
# default (verified, linear FITS, best data in the north); the red+vc hybrid
# is the one to reach for on colour data while CDS is down.
DEFAULT_COLOUR_SURVEY = "DSS2 red + NSNS continuum (north) — colour"

for _name, _info in SURVEYS.items():
    _info["availability"] = _SURVEY_AVAILABILITY.get(_name, "verified")


def mirrors_for(hips_id: str) -> list[str]:
    """Direct HiPS base URLs registered for a hips id (may be empty)."""
    for info in SURVEYS.values():
        for ch in info.get("channels") or []:
            if isinstance(ch, dict) and ch.get("hips") == hips_id:
                return list(ch.get("mirrors") or [])
    return []



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

# Fetched references kept per dialog, so re-running with a different degree
# or protect k does not re-download. A 1024x1024x3 float32 reference is
# ~12 MB, so this is a handful of entries, not a general-purpose cache.
_REF_CACHE_MAX = 3

# Long side the PREVIEW is rendered at. The label is a few hundred pixels
# and the pixmap is smooth-scaled down to it anyway, so correcting and
# tone-mapping a 24 MP frame to fill it is pure waste: measured 8.0 s per
# redraw at 6248x4168 against ~0.1 s here. Nothing is DECIDED from the
# preview -- Apply always runs on the full-resolution snapshot.
_PREVIEW_LONG_SIDE = 1400

# Rendered 8-bit preview buffers kept per dialog. One is ~5.9 MB at the
# long side above, and the useful set is small: Before, After, and a couple
# of strength values the user is flicking between.
_DISP_CACHE_MAX = 8

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


def _decode_image_tile(raw: bytes, ext: str) -> np.ndarray:
    """
    Decode a PNG/JPEG HiPS tile to float32, (H, W) grey or (H, W, 3) RGB.

    8-bit tiles are perfectly usable for this job and were wrongly excluded
    at first. A gradient is low-frequency by definition, and a background
    model averages thousands of pixels per sample patch, so 256 levels carry
    far more precision than the fit needs. What actually matters is whether
    the tree is LINEAR in the large scale:

      DSS2 colour  - CDS builds it from the very FITS planes this module
                     wants, with linear cuts recorded in its properties
                     (hips_rgb_red = DSS2Merged [... Linear], hips_rgb_blue =
                     DSS2-blue-XJ-S [... Linear]). So its blue channel IS the
                     blue plate, scaled and quantised. Fine.
      SDSS9 colour - same construction with a Log stretch. A monotonic curve
                     over a background's narrow range is close to linear, so
                     it is largely absorbed into b. Usable, slightly soft.
      NSNS 8-bit   - the survey states these are "dynamic range compressed
                     (non-linearly high-pass filtered) and tonal curve
                     corrected". A high-pass genuinely removes the frequencies
                     being fitted, which quantisation does not. That is the
                     one real hazard, and reference_share measures it
                     directly: if the structure is gone, b*span collapses and
                     the diagnostic says so. Offered, with that caveat.

    NOTE the two-way alpha handling: some HiPS PNG tiles carry transparency to
    mark absent coverage, and treating alpha=0 as black would invent real
    "sky is dark here" structure where there is simply no data.
    """
    if cv2 is None:
        raise SurveyFetchError(
            "cv2 is needed to read PNG/JPEG HiPS tiles and is not available.")
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode a {ext} tile ({len(raw)} bytes)")

    img = np.asarray(img)
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[..., 3]
        img = img[..., :3]
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[..., ::-1]          # cv2 gives BGR; the tree is RGB
    out = img.astype(np.float32)
    if alpha is not None:
        # Fully transparent means "no coverage", which must read as NaN and
        # not as zero, or the fit sees a hard dark edge that is not sky.
        mask = alpha == 0
        if mask.any():
            out[mask] = np.nan
    elif out.ndim == 3 and out.shape[2] == 3:
        # A JPEG has no alpha, so a tree that is only published as jpeg has
        # no way to say "nothing here" -- out-of-footprint comes back as
        # exact (0,0,0). np.isfinite cannot tell that from data, so the
        # coverage check passed at 100% on a field that was 40% EMPTY, and
        # the fit then treated that void as real sky structure.
        #
        # Exact zero in all THREE channels is the giveaway. A stretched
        # 8-bit sky composite is noise around some low value, never exactly
        # 0,0,0 across a region: measured 0.0% inside the SDSS footprint
        # against 40.4% outside it. Only applied to 3-channel tiles -- a
        # single 0 in a mono tile is genuinely ambiguous.
        void = np.all(out == 0.0, axis=2)
        if void.any():
            out[void] = np.nan
    return out


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
        if ext in ("png", "jpg", "jpeg"):
            arr = _decode_image_tile(raw, ext)
        else:
            import io
            import warnings as _w
            from astropy.io import fits
            with _w.catch_warnings():
                # IRSA serves tiles without the trailing FITS padding block,
                # which astropy flags as truncated. The pixel data is all
                # there; only the pad is missing, so the warning is noise.
                _w.simplefilter("ignore")
                with fits.open(io.BytesIO(raw), ignore_missing_end=True) as hdul:
                    hdu = next(
                        (x for x in hdul if getattr(x, "data", None) is not None),
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
                     *, timeout: float = HIPS_TIMEOUT,
                     info: dict | None = None) -> np.ndarray:
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
    # FITS first when offered — linear and unquantised. Otherwise fall back to
    # the 8-bit trees, which carry a gradient perfectly well (see
    # _decode_image_tile). PNG before JPEG: PNG is lossless, and JPEG's block
    # artifacts, while far below the scale of anything being fitted, are free
    # to avoid when there is a choice.
    ext = next((f for f in ("fits", "png", "jpeg", "jpg") if f in fmts), None)
    if ext is None:
        raise SurveyFetchError(
            f"{base} advertises tile formats {fmts or ['?']}, none of which "
            "can be read (expected one of fits, png, jpeg)."
        )
    if ext == "jpeg":
        ext = "jpg"          # the tile files are .jpg even when advertised as jpeg
    is_image = ext != "fits"
    if is_image:
        _log(f"  using 8-bit {ext} tiles (no FITS at this tree)")
    # An out-parameter rather than a changed return type: every existing
    # caller wants the array and nothing else, and the dict is owned by
    # whoever asked, so two concurrent fetches cannot scribble on each
    # other the way a module-level "last fetch" global would.
    if info is not None:
        info.update({
            "route": "tiles",
            "format": ext,
            "bits": (8 if is_image else None),
            "offered": sorted(fmts),
        })
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
    if info is not None:
        info.update({"order": order, "max_order": max_order, "deep": deep})
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
    # ORIENTATION: the HiPS standard stores FITS tiles with the origin at the
    # BOTTOM-left (FITS convention) and PNG/JPEG tiles at the TOP-left (image
    # convention), so the vertical flip is opposite between them. Getting this
    # backwards does not error — it mirrors every tile about its own centre
    # and quietly returns scrambled sky. test_tile_orientation.py checks it
    # against DSS2 red, which IRSA serves as both fits and jpeg.
    row = a if is_image else (tile_w - 1) - a
    col = b

    uniq = np.unique(npix_tile)
    _log(f"  tiles needed: {len(uniq)} at order {order}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = dict(zip(
            uniq.tolist(),
            pool.map(lambda n: _fetch_tile(base, order, int(n), ext, timeout),
                     uniq.tolist())))

    # Colour trees give three planes; carry them all so a caller can ask for
    # one. DSS2 colour's blue channel is the blue plate that is mirrored
    # nowhere as FITS, so throwing it away here would discard the only real
    # blue available during the outage.
    nband = 1
    for t in tiles.values():
        if t is not None and t.ndim == 3:
            nband = max(nband, t.shape[2])

    out = np.full((h * w, nband) if nband > 1 else (h * w,), np.nan,
                  dtype=np.float32)

    got = 0
    for n in uniq.tolist():
        t = tiles.get(n)
        if t is None:
            continue
        got += 1
        sel = npix_tile == n
        vals = t[row[sel], col[sel]]
        if nband > 1:
            if vals.ndim == 1:                 # grey tile in a colour tree
                vals = np.repeat(vals[:, None], nband, axis=1)
            out[sel, :] = vals[:, :nband]
        else:
            out[sel] = vals

    _log(f"  tiles ok   : {got}/{len(uniq)}")
    res = out.reshape((h, w, nband) if nband > 1 else (h, w))
    _log("  " + _array_stats(res, f"tiles order={order}"
                                  f"{' x%d' % nband if nband > 1 else ''}"))
    if got == 0:
        raise SurveyFetchError(
            f"none of the {len(uniq)} tiles covering this field could be "
            f"fetched from {base}"
        )
    return res


# Hosts that did not answer at all this session. During the 2026 CDS outage a
# colour fetch walks four CDS routes in turn and each one burns its full
# connect timeout, so a single failed channel cost ~84 s of staring at a
# progress bar before the error appeared. Once a host has proved unreachable
# there is nothing to learn by waiting for it again.
_DEAD_HOSTS: set[str] = set()


# Result of the TCP pre-flight, per host, for this session. Separate from
# _DEAD_HOSTS because that set also collects hosts that failed mid-transfer;
# this one is purely 'did the socket open'.
_HOST_ALIVE: dict[str, bool] = {}


def reset_dead_hosts() -> None:
    """Forget the unreachable-host list (call when the tool is reopened)."""
    _DEAD_HOSTS.clear()
    _HOST_ALIVE.clear()


def _host_of(url: str) -> str:
    """Hostname (no port) from a base URL."""
    return url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]


def _host_reachable(url: str, *, timeout: float = CONNECT_TIMEOUT) -> bool:
    """
    Can we open a TCP connection to this URL's host at all?

    This only ever lets us SKIP work. A refused connect is proof that every
    further route on that host will fail; a successful one proves nothing
    beyond 'something is listening', which is exactly why it is never used
    to decide that a survey is available (see recheck_availability).

    Cached per host for the session, and a failure also feeds _DEAD_HOSTS so
    the existing skip paths and their user-facing message stay consistent.
    """
    import socket
    import time as _time

    host = _host_of(url)
    cached = _HOST_ALIVE.get(host)
    if cached is not None:
        return cached

    port = 80 if url.lower().startswith("http://") else 443
    t0 = _time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=float(timeout)):
            ok = True
    except Exception as e:
        ok = False
        _log(f"  pre-flight : {host} did not answer in "
             f"{(_time.monotonic() - t0) * 1000:.0f} ms ({_brief(e)})")
    _HOST_ALIVE[host] = ok
    if not ok:
        _DEAD_HOSTS.add(host)
    return ok


def _is_connection_failure(e: BaseException) -> bool:
    """
    True when the host never answered — timeout, refused, DNS, TLS — as
    opposed to answering with an HTTP status.

    The distinction decides whether to blacklist the HOST or only this path.
    A 404 means the survey is not at that URL and says nothing about the
    server's health: IRSA serves DSS2 red and 404s on blue, and blacklisting
    IRSA over that would throw away a working mirror. A connect timeout means
    every further route on that host will burn its timeout too.
    """
    from urllib.error import HTTPError as _UHTTPError, URLError as _UURLError

    # Must precede every check below: urllib's HTTPError subclasses URLError,
    # which subclasses OSError, so it would match the broad cases otherwise.
    if isinstance(e, _UHTTPError) or type(e).__name__ == "HTTPError":
        return False
    if isinstance(e, _UURLError):
        return True
    if isinstance(e, (TimeoutError, ConnectionError, OSError)):
        return True
    # requests/urllib3 raise their own hierarchy that shares no base with the
    # stdlib one, so fall back to matching the names.
    return type(e).__name__ in {
        "ConnectTimeout", "ReadTimeout", "Timeout", "ConnectionError",
        "NewConnectionError", "MaxRetryError", "SSLError", "ProxyError",
    }


def fetch_reference(wcs: WCS, out_hw: tuple[int, int], hips_id: str,
                    *, mirrors: list[str] | None = None,
                    progress_cb=None, timeout: float = HIPS_TIMEOUT,
                    allow_tiles: bool = True,
                    info: dict | None = None) -> np.ndarray:
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
        mirrors = list(mirrors) if mirrors else mirrors_for(hips_id)
        if not mirrors:
            _log(f"no direct mirrors registered for {hips_id}")
        for base in mirrors:
            host = base.split("//", 1)[-1].split("/", 1)[0]
            if host in _DEAD_HOSTS:
                _log(f"-- route: direct tiles @ {host} — SKIPPED, unreachable "
                     f"earlier this session")
                attempts.append(f"tiles {host}: skipped (unreachable earlier)")
                continue
            # Cheap TCP pre-flight before committing to a 30 s read timeout
            # the OS will pre-empt with its own ~21 s connect timeout anyway.
            if not _host_reachable(base):
                _log(f"-- route: direct tiles @ {host} — SKIPPED, no answer "
                     f"within the {CONNECT_TIMEOUT:.1f}s pre-flight")
                attempts.append(
                    f"tiles {host}: no TCP answer within {CONNECT_TIMEOUT:.1f}s")
                continue
            if progress_cb:
                progress_cb(f"Reading reference tiles from {host}…")
            _log(f"-- route: direct tiles @ {host}")
            t0 = time.monotonic()
            try:
                _inf: dict = {}
                data = _fetch_via_tiles(qw, h, w, base, timeout=timeout,
                                        info=_inf)
                dt = (time.monotonic() - t0) * 1000.0
                _log(f"   OK in {dt:.0f} ms")
                if info is not None:
                    info.update(_inf)
                    info.update({"host": host, "hips": hips_id})
                if progress_cb:
                    progress_cb(f"Reference built from {host} tiles.")
                return data
            except Exception as e:
                dt = (time.monotonic() - t0) * 1000.0
                _log(f"   FAILED after {dt:.0f} ms: {_brief(e, 300)}")
                attempts.append(f"tiles {host} ({dt:.0f} ms): {_brief(e)}")
                if _is_connection_failure(e):
                    _DEAD_HOSTS.add(host)
                    _log(f"   {host} marked unreachable for this session")

    # hips2fits is now the fallback: it is server-side and convenient, but it
    # is a single vendor's service layer and it was the thing that went down.
    if have_aq:
        for ep in _ordered_endpoints():
            host = ep.split("//", 1)[-1].split("/", 1)[0]
            if host in _DEAD_HOSTS:
                _log(f"-- route: hips2fits @ {host} — SKIPPED, unreachable "
                     f"earlier this session")
                attempts.append(
                    f"hips2fits {host}: skipped (unreachable earlier)")
                continue
            if not _host_reachable(ep):
                _log(f"-- route: hips2fits @ {host} — SKIPPED, no answer "
                     f"within the {CONNECT_TIMEOUT:.1f}s pre-flight")
                attempts.append(
                    f"hips2fits {host}: no TCP answer within "
                    f"{CONNECT_TIMEOUT:.1f}s")
                continue
            if progress_cb:
                progress_cb(f"Trying hips2fits ({host})…")
            _log(f"-- route: hips2fits @ {host}")
            t0 = time.monotonic()
            try:
                data = _fetch_via_hips2fits(qw, h, w, hips_id, ep, timeout)
                dt = (time.monotonic() - t0) * 1000.0
                _remember_endpoint(ep)
                _log(f"   OK in {dt:.0f} ms")
                if info is not None:
                    # hips2fits always hands back FITS, whatever the
                    # underlying tree is published as.
                    info.update({"route": "hips2fits", "format": "fits",
                                 "bits": None, "host": host,
                                 "hips": hips_id})
                if progress_cb:
                    progress_cb(f"Reference received from {host}.")
                return data
            except Exception as e:
                dt = (time.monotonic() - t0) * 1000.0
                _log(f"   FAILED after {dt:.0f} ms: {_brief(e, 300)}")
                attempts.append(f"hips2fits {host} ({dt:.0f} ms): {_brief(e)}")
                if _is_connection_failure(e):
                    _DEAD_HOSTS.add(host)
                    _log(f"   {host} marked unreachable for this session")

    _log("ALL ROUTES FAILED")
    _log("=" * 62)
    unreachable = sorted(_DEAD_HOSTS)
    tail = ""
    if unreachable:
        tail = ("\n\nUnreachable this session: " + ", ".join(unreachable) +
                ". These are skipped from now on rather than waiting out "
                "each timeout again; reopen the tool to retry them.")
    raise SurveyFetchError(
        "Could not fetch a reference cutout. Every route failed:\n\n"
        + "\n".join(f"  • {a}" for a in attempts)
        + tail
        + "\n\nDuring a CDS outage the DSS2 and SDSS options also mirror from "
          "IRSA (Caltech), CASDA (CSIRO, Australia) and IAS (Orsay) — but not "
          "every band is mirrored at every site, so a 404 on one host is "
          "normal and the next route is tried automatically."
    )


# The hosts whose outage is the REASON the availability tags above exist.
# Only these can produce a "moved" verdict. A 404 from a host that has been
# up all along is not news -- it is the inventory we already measured and
# recorded (IRSA has never carried DSS2 blue), and reporting it as "the
# trees have moved" would fire on every single dialog open.
_OUTAGE_HOSTS: set = {
    _host_of(_ep) for _ep in HIPS2FITS_ENDPOINTS
} | {"alasky.cds.unistra.fr", "alaskybis.cds.unistra.fr",
     "cdsarc.cds.unistra.fr", "healpix.ias.u-psud.fr",
     "wwwfirback.ias.u-psud.fr"}

# Tags that mean "we could not REACH this", and so are worth re-testing.
# "stretched" and any future quality caveat are deliberately absent: they
# describe what the data IS, not whether it can be fetched, and clearing
# one on a successful fetch would delete a warning that is still true.
_RECHECKABLE_TAGS: set = {"unavailable", "degraded"}


def _tree_state(base: str, *, need_fits: bool,
                timeout: float = CONNECT_TIMEOUT) -> str:
    """
    One mirror base URL, classified: "live" | "moved" | "down".

      live  the host answered, /properties exists, and it advertises a tile
            format this reader can use for that channel.
      moved the host answered but /properties did not resolve. The node is
            back and the tree is somewhere else. This is the state a
            liveness ping reports as a successful restore, and it is why
            this function exists.
      down  no TCP answer.
      absent  the host answered but does not carry this tree, and never
            did -- the measured inventory, not a change worth reporting.
    """
    # "moved" is only meaningful for a host that was part of the outage.
    # Everywhere else a missing path is the known inventory, not a change.
    missing = "moved" if _host_of(base) in _OUTAGE_HOSTS else "absent"
    if not _host_reachable(base, timeout=timeout):
        return "down"
    try:
        props = hips_properties(base, timeout=max(timeout, 5.0))
    except Exception as e:
        # The host answered the socket, so this is about the PATH, not the
        # node -- unless it turns out to be a connection failure after all,
        # in which case say so rather than claiming the tree moved.
        return "down" if _is_connection_failure(e) else missing
    fmts = str(props.get("hips_tile_format", "")).lower().split()
    if need_fits:
        return "live" if "fits" in fmts else missing
    usable = {"fits", "png", "jpeg", "jpg"}
    return "live" if (set(fmts) & usable) else missing


def recheck_availability(*, timeout: float = CONNECT_TIMEOUT) -> dict:
    """
    Re-verify the surveys that are NOT currently tagged "verified" and
    return {survey name: new tag} for those whose tag changed.

    Only ever UPGRADES, and only tags in _RECHECKABLE_TAGS are considered.
    A verified survey is left alone: a transient network blip must not be
    able to grey out a working option, and the fetch path already reports a
    real failure properly when one happens. A quality tag like "stretched"
    is left alone too -- it is not a statement about reachability, and
    clearing it on a successful fetch would delete a warning still true.

    The check is /properties, not a ping. A host that answers tells you
    nothing about whether its static tile trees came back at the same paths
    after an outage, and re-enabling a survey on that evidence produces a
    confident option that fails in a new way. /properties is one small GET
    that proves the tree exists AND advertises a format we can read, and it
    shares _PROPS_CACHE with the real fetch path so it costs nothing twice.

    A channel carrying a "channel" key is a band selected out of an 8-bit
    colour composite, so any readable format counts. A channel without one
    wants its own FITS tree, and a jpeg-only mirror does not restore it.
    """
    changes: dict = {}
    for name, info in SURVEYS.items():
        current = info.get("availability", "verified")
        if current not in _RECHECKABLE_TAGS:
            continue

        states: list[str] = []
        for ch in info.get("channels") or []:
            if not isinstance(ch, dict):
                continue          # "synth" has nothing to fetch
            need_fits = "channel" not in ch
            per_mirror = [
                _tree_state(m, need_fits=need_fits, timeout=timeout)
                for m in (ch.get("mirrors") or [])
            ]
            if "live" in per_mirror:
                states.append("live")
            elif "moved" in per_mirror:
                states.append("moved")
            else:
                states.append("down")

        if not states:
            continue
        if all(s == "live" for s in states):
            new = "verified"
        elif "moved" in states:
            new = "moved"
        else:
            new = current

        if new != current:
            changes[name] = new
            _log(f"availability: {name!r}  {current} -> {new}")
    if not changes:
        _log("availability re-check: nothing changed")
    return changes


# QThread objects are kept here for the duration of a run. A QThread that is
# garbage collected while still running takes the process with it, and the
# dialog that started this one may well be closed and deleted first.
_LIVE_PROBES: set = set()


class _AvailabilityProbe(QThread):
    """Runs recheck_availability() off the UI thread."""

    finishedWith = pyqtSignal(dict)

    def __init__(self):
        # parent=None per SASpro's QThread rule; lifetime is held by
        # _LIVE_PROBES, not by the dialog.
        super().__init__(parent=None)

    def run(self):
        try:
            self.finishedWith.emit(recheck_availability())
        except Exception as e:
            _log(f"availability re-check failed: {_brief(e)}")
            self.finishedWith.emit({})


def _brief(e: Exception, n: int = 140) -> str:
    s = f"{type(e).__name__}: {e}".replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _select_band(arr: np.ndarray, k, hips_id: str, i: int) -> np.ndarray:
    """
    Pick one plane out of a fetched reference.

    A colour tile tree returns three planes. `channel` in a registry entry
    says which one this target channel is fitted against — that is how DSS2
    colour supplies a real red AND a real blue from one download, each being
    the original plate linearly scaled.

    k=None on a colour tree means "use it as mono", and the mean is taken
    rather than a weighted luminance: the planes here are separate filters,
    not a perceptual colour image, so weighting them by human sensitivity
    would be meaningless.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        if k is not None:
            _log(f"  channel {i} asked for band {k} of {hips_id}, which is "
                 f"mono — using it as-is")
        return a
    nb = a.shape[2]
    if k is None:
        _log(f"  channel {i} averages the {nb} bands of {hips_id}")
        return np.nanmean(a, axis=2).astype(np.float32)
    kk = int(np.clip(int(k), 0, nb - 1))
    if kk != k:
        _log(f"  channel {i} asked for band {k} of {hips_id} which has {nb}; "
             f"using band {kk}")
    _log(f"  channel {i} takes band {kk} of {hips_id}")
    return a[..., kk].astype(np.float32)


def fetch_reference_channels(wcs: WCS, out_hw: tuple[int, int],
                             channels: list, *, progress_cb=None,
                             timeout: float = HIPS_TIMEOUT,
                             info: dict | None = None) -> np.ndarray:
    """
    Fetch a 1- or 3-channel reference.

    Gradient correction on colour data is a per-channel problem: light
    pollution is strongly colour-selective, so each of the target's channels
    needs its own reference plane and its own scale factor. A single mono
    reference can only say "the structure here is X times brighter"; it
    cannot say that the structure is redder than your sky but bluer than
    your gradient.

    `channels` holds 1 or 3 entries. Each is a dict with hips/mirrors, or the
    string "synth" for a plane synthesised as the mean of the fetched ones —
    DSS2 has red and blue plates and no green, and an absent green is worse
    than an interpolated one when the alternative is correcting two of your
    three channels against a plane that does not exist.

    Returns (h, w) for mono or (h, w, 3) for colour.
    """
    h, w = int(out_hw[0]), int(out_hw[1])

    def _rollup():
        """
        Summarise the channels into the top-level fields the UI reads.

        A colour reference can mix trees -- DSS2 red for R and NSNS for
        G/B is a shipped option -- so a single "format" is only honest
        when every channel agrees. When they do not, say "mixed" rather
        than picking one channel's answer and implying it covers all.
        """
        if info is None:
            return
        chans = info.get("channels") or []
        if not chans:
            return
        for field in ("route", "format", "bits", "host"):
            vals = {c.get(field) for c in chans}
            if len(vals) == 1:
                info[field] = next(iter(vals))
            else:
                info[field] = "mixed"
        info["n_channels"] = len(chans)
    planes: list[np.ndarray | None] = []
    synth: list[int] = []

    # Dedupe by hips id. The HOO and SOO composites both reference the same
    # [OIII] tree for green and blue, so a naive loop would fetch and reproject
    # it twice for an identical result. The plane is immutable downstream (the
    # fit reads it, nothing writes to it), so aliasing the same array into two
    # channel slots is safe and halves the download.
    seen: dict[str, np.ndarray] = {}
    failures: list[str] = []

    for i, ch in enumerate(channels):
        if isinstance(ch, str) and ch.lower() == "synth":
            planes.append(None)
            synth.append(i)
            continue
        hips_id = ch["hips"]
        cached = seen.get(hips_id)
        if cached is not None:
            _log(f"  channel {i} reuses the already-fetched {hips_id}")
            planes.append(_select_band(cached, ch.get("channel"), hips_id, i))
            continue
        if progress_cb and len(channels) > 1:
            progress_cb(f"Reference channel {i + 1}/{len(channels)}…")
        try:
            _ci: dict = {}
            plane = fetch_reference(
                wcs, (h, w), hips_id, mirrors=ch.get("mirrors"),
                progress_cb=progress_cb, timeout=timeout, info=_ci)
            if info is not None:
                _ci["channel"] = i
                _enc, _why = _tree_encoding(hips_id)
                _ci["encoding"] = _enc
                _ci["encoding_why"] = _why
                info.setdefault("channels", []).append(_ci)
        except Exception as e:
            # One unavailable channel must not throw away the ones that
            # worked. While CDS is down, DSS2 blue is mirrored nowhere
            # reachable (measured: IRSA, CASDA and ESA all 404) while DSS2
            # red fetches fine, so aborting here turned a usable red-only
            # correction into a total failure. Degrade instead: the missing
            # plane is synthesised from whatever did arrive, which is the
            # same handling "synth" already gets, and every channel still
            # solves its own (a, b).
            _log(f"  channel {i} ({hips_id}) FAILED: {_brief(e)}")
            _log(f"  -> degrading: this channel will be synthesised from the "
                 f"channels that did fetch")
            failures.append(f"channel {i} ({hips_id}): {_brief(e)}")
            planes.append(None)
            synth.append(i)
            continue
        # Cache the WHOLE fetch, unsliced, so a second channel spec pointing
        # at the same colour tree reuses it instead of downloading it again.
        seen[hips_id] = plane
        planes.append(_select_band(plane, ch.get("channel"), hips_id, i))

    real = [p for p in planes if p is not None]
    if not real:
        if failures:
            raise SurveyFetchError(
                "No reference channel could be fetched:\n\n"
                + "\n".join(f"  • {f}" for f in failures)
            )
        raise SurveyFetchError("no real reference channels were requested")

    if failures:
        _log(f"DEGRADED: {len(failures)} of {len(channels)} reference "
             f"channel(s) unavailable; the correction is running on "
             f"{len(real)} real plane(s). Per-channel colour information in "
             f"the gradient is reduced accordingly.")

    for i in synth:
        planes[i] = np.nanmean(np.stack(real, axis=0), axis=0).astype(np.float32)
        _log(f"  channel {i} synthesised as the mean of "
             f"{len(real)} fetched plane(s)")

    _rollup()
    if len(planes) == 1:
        return np.asarray(planes[0], dtype=np.float32)
    return np.stack([np.asarray(p, dtype=np.float32) for p in planes],
                    axis=-1)


# =============================================================================
#                                 Robust fit
# =============================================================================

def _robust_norm01(a: np.ndarray) -> np.ndarray:
    """
    Percentile-normalise to [0,1] so two images with wildly different units
    (Rayleighs vs ADU vs 16-bit plate counts) can be compared for brightness.
    Clipped at the 1st/99.5th percentile so a handful of saturated stars
    cannot compress everything else into the bottom of the range.
    """
    a = np.asarray(a, dtype=np.float32)
    f = np.isfinite(a)
    if not f.any():
        return np.zeros_like(a)
    lo, hi = np.percentile(a[f], [1.0, 99.5])
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _selection_map(tgt_small: np.ndarray, ref_small: np.ndarray,
                   patch_size: int) -> np.ndarray:
    """
    The brightness map that sample points are chosen on.

    THE SURVEY IS THE GROUND TRUTH for where real diffuse emission is. The
    target's own brightness must NOT disqualify a location, because the target
    is the frame with the gradient in it — that is the entire premise. Treating
    a bright patch of light pollution as "not background" would starve the fit
    on the bright side of the gradient and bias the model toward the dim
    corner, which is the same failure ADBE's cell-bounded descent exists to
    prevent. It is also self-defeating: the gradient is the quantity being
    measured, so declining to sample where it is strong declines to measure it.

    So the reference supplies the diffuse-structure veto, and the target
    contributes only its HIGH-PASS — stars, satellite trails, sharp artifacts.
    A high-pass is blind to a smooth gradient by construction, so a point sat
    in the brightest corner of your light pollution is still a perfectly good
    background sample, while a point on a star is not.

        map = max( norm(reference),  norm(positive high-pass of target) )
    """
    r = _robust_norm01(ref_small)

    t = _to_luminance(tgt_small) if tgt_small.ndim == 3 else tgt_small
    t = np.asarray(t, dtype=np.float32)
    hp = None
    if cv2 is not None:
        try:
            sigma = max(2.0, float(patch_size))
            k = int(2 * round(3 * sigma) + 1)
            low = cv2.GaussianBlur(t, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
            hp = np.clip(t - low, 0.0, None)      # positive excursions only
        except Exception:
            hp = None
    if hp is None:
        _log("  (no cv2: star rejection via high-pass disabled)")
        return r
    return np.maximum(r, _robust_norm01(hp)).astype(np.float32)


def _sampling_mask(ref_small: np.ndarray, valid: np.ndarray, *,
                   exclusion_fraction: float = 0.5,
                   protect_k: float = 3.0,
                   protect_grow: int = 3,
                   patch_size: int = 15,
                   min_sky_frac: float = 0.12) -> tuple[np.ndarray, dict]:
    """
    Where sampling is ALLOWED (True). The hard veto the sampler was missing.

    Built entirely from the SURVEY, because the survey is ground truth for
    where real emission is — the frame being corrected has a gradient in it
    and gets no vote (see _selection_map).

    Two vetoes, both already standard in this codebase rather than invented
    here:

      _exclude_bright_regions(..., 0.5)  drops the brightest half outright.
          Already used this way in gradient_abe.py and stacking_suite.py.

      build_signal_mask(k, grow)         median + k*MAD, dilated and blurred.
          The same protection ADBE's multiscale stage uses, so "protect k"
          means the same thing in both tools.

    Two additions on top, for reasons the M31 case made obvious:

    SMOOTH FIRST. A dust lane inside M31 is dimmer than the sky outside the
    galaxy, so on a raw map it passes a brightness test AND acts as a local
    minimum that the cell-bounded descent walks straight into. Smoothing at
    the patch scale means "dim" is judged over a neighbourhood.

    FILL HOLES. Whatever the lanes still win, binary_fill_holes closes: a
    lane is a hole inside the galaxy's mask, and a hole inside structure is
    structure. This is what stops a point landing between M31's dust lanes.

    If structure dominates the frame the thresholds are relaxed until at
    least `min_sky_frac` is samplable, and the relaxation is reported — a
    field with no sky in it cannot be corrected against a reference, and the
    user should be told rather than handed a confident fit through a galaxy.
    """
    r = np.nan_to_num(np.asarray(ref_small, dtype=np.float32), nan=np.nan)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(r)
    info: dict = {"exclusion_fraction": float(exclusion_fraction),
                  "protect_k": float(protect_k), "relaxed": False}
    if not ok.any():
        return ok, info

    fill = float(np.median(r[ok]))
    rs = np.where(np.isfinite(r), r, fill).astype(np.float32)
    if cv2 is not None:
        try:
            sigma = max(1.5, float(patch_size) * 0.75)
            ks = int(2 * round(3 * sigma) + 1)
            rs = cv2.GaussianBlur(rs, (ks, ks), sigma,
                                  borderType=cv2.BORDER_REPLICATE)
        except Exception:
            pass

    try:
        from scipy.ndimage import binary_fill_holes
    except Exception:
        binary_fill_holes = None

    frac, k = float(exclusion_fraction), float(protect_k)
    allowed = ok.copy()
    for _ in range(14):
        # veto 1 — the brightest `frac` of the covered sky, house pattern
        dim_enough = np.zeros_like(ok)
        sub = rs[ok]
        if sub.size:
            thresh = np.percentile(sub, 100.0 * (1.0 - max(0.0, min(frac, 0.95))))
            dim_enough = (rs < thresh)
        struct = ~dim_enough

        # veto 2 — median + k*MAD, dilated and blurred
        if build_signal_mask is not None and np.isfinite(k):
            try:
                sig = build_signal_mask(rs, k=k, grow=int(protect_grow),
                                        blur_px=max(1.0, patch_size * 0.5))
                struct |= (np.asarray(sig, dtype=np.float32) > 0.35)
            except Exception as e:
                _log(f"  build_signal_mask unavailable: {_brief(e)}")

        if binary_fill_holes is not None and struct.any():
            try:
                struct = binary_fill_holes(struct)
            except Exception:
                pass

        allowed = ok & ~struct
        if float(allowed.mean()) >= min_sky_frac:
            break
        info["relaxed"] = True
        frac = max(0.05, frac - 0.08)
        k = k + 1.0

    info.update(exclusion_fraction=frac, protect_k=k,
                allowed_frac=float(allowed.mean()),
                vetoed_frac=float((ok & ~allowed).mean()))
    _log(f"  sampling mask: {info['allowed_frac']*100:.1f}% allowed, "
         f"{info['vetoed_frac']*100:.1f}% vetoed "
         f"(exclude_frac={frac:.2f}, protect_k={k:.1f}"
         + (", RELAXED — structure dominates this field)" if info["relaxed"]
            else ")"))
    return allowed, info


def _shared_sample_points(tgt_small: np.ndarray, ref_small: np.ndarray,
                          *, valid: np.ndarray, num_samples: int,
                          patch_size: int, exclusion_fraction: float = 0.5,
                          protect_k: float = 3.0, protect_grow: int = 3,
                          user_exclusion: np.ndarray | None = None,
                          seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    One set of background sample points, used for BOTH background models.

    This is the difference between a reference-based correction that works and
    one that quietly eats nebulosity.

    Modelling the two backgrounds independently lets each one chase structure
    in its own way — different noise, different resolution (NSNS is ~10", your
    data is sub-arcsec), different star content. The fit downstream is
    B_t ~ a + b*B_r, so any structure one model follows and the other does not
    lands in the residual and gets subtracted as "gradient". Sampling both at
    the SAME sky positions makes those errors common-mode, and common-mode
    error is exactly what the b*B_r term is there to absorb.

    See _selection_map for what counts as a valid background location — in
    short, the survey decides, not the frame being corrected.

    `valid` (True = usable) keeps points out of the region where the reference
    had no coverage and was median-filled — sampling there would be fitting an
    invented flat value and would drag the model toward it.
    """
    combined = _selection_map(tgt_small, ref_small, patch_size)

    allowed, info = _sampling_mask(
        ref_small, valid,
        exclusion_fraction=float(exclusion_fraction),
        protect_k=float(protect_k), protect_grow=int(protect_grow),
        patch_size=int(patch_size),
    )

    # Applied AFTER the relaxation loop inside _sampling_mask, on purpose:
    # relaxation exists to find enough sky when the SURVEY vetoes too much,
    # and it must not be able to claw back a region the user explicitly put
    # off limits.
    if user_exclusion is not None:
        ue = np.asarray(user_exclusion, dtype=bool)
        if ue.shape[:2] == allowed.shape[:2]:
            before = float(allowed.mean())
            allowed = allowed & ~ue
            info["user_excluded_frac"] = float(ue.mean())
            info["allowed_frac"] = float(allowed.mean())
            _log(f"  user exclusions: samplable {before*100:.1f}% -> "
                 f"{info['allowed_frac']*100:.1f}%")

    pts = _generate_sample_points(
        combined,
        num_points=int(num_samples),
        exclusion_mask=allowed,          # True = allowed, matching ADBE
        patch_size=int(patch_size),
        rng=np.random.default_rng(int(seed)),
    )

    # Belt and braces. _generate_sample_points falls back to a coarse grid if
    # exclusion is so aggressive that nothing survives, and that fallback
    # ignores the mask — which is correct for ADBE (a background model with no
    # points is useless) but wrong here, where a point inside a galaxy is
    # worse than a sparser fit. Drop any straggler that ended up vetoed.
    if len(pts):
        p = np.asarray(pts, dtype=int)
        H, W = allowed.shape[:2]
        keep = allowed[np.clip(p[:, 1], 0, H - 1), np.clip(p[:, 0], 0, W - 1)]
        dropped = int((~keep).sum())
        if dropped and int(keep.sum()) >= 12:
            pts = p[keep]
            _log(f"  dropped {dropped} point(s) that landed in vetoed regions")
        elif dropped:
            _log(f"  WARNING: {dropped} of {len(p)} points are in vetoed "
                 f"regions but too few remain to drop them — this field is "
                 f"mostly structure and the correction will be weak")

    _log(f"shared sample points: {len(pts)} (requested {num_samples}, "
         f"patch {patch_size}px, {info['allowed_frac']*100:.0f}% of frame "
         f"samplable)")
    return np.asarray(pts, dtype=np.int32), allowed, info


# A background spanning fewer levels than this cannot describe a gradient
# with any shape to it. Measured on real fetches of M81 at 1.5 deg: a FITS
# tree gave a background spanning 3445 quantisation steps, CDS's 8-bit
# composites 62 and 78. Those look comfortable, but that is a whole-frame
# figure -- the SKY-ONLY background the fit actually uses was 40x smaller on
# the same field, which puts the 8-bit trees at a couple of levels. Hence
# the check.
_REF_MIN_LEVELS = 8.0


def describe_reference_provenance(info: dict | None,
                                  diag: dict | None = None) -> str:
    """
    What was fetched, how it encodes brightness, and whether it can
    resolve a gradient -- as one line for the UI.

    This is the answer to "are we comparing apples to apples?". The fit
    absorbs any linear scale exactly, so the honest reportable facts are
    the ENCODING (linear / log / high-passed) and the RESOLUTION (how many
    quantisation steps the background spans). Both were previously in the
    log only, which meant the one number that decides whether a reference
    is usable was invisible unless you went looking.
    """
    bits = []
    info = info or {}
    fmt = info.get("format")
    route = info.get("route")
    nbits = info.get("bits")
    if route == "supplied":
        bits.append("your own reference (no survey provenance)")
    elif fmt == "fits":
        src = ("hips2fits" if route == "hips2fits" else "FITS tiles")
        bits.append(f"{src}, linear")
    elif fmt:
        bits.append(f"{nbits or 8}-bit {fmt} tiles")

    # Encoding: unanimous across channels, or named per channel.
    chans = info.get("channels") or []
    encs = {c.get("encoding", "linear") for c in chans}
    if encs and encs != {"linear"}:
        bits.append("encoding: " + "/".join(sorted(encs)))
    elif encs == {"linear"} and fmt and fmt != "fits":
        bits.append("linearly scaled")

    d = diag or {}
    res = d.get("ref_resolution")
    lv = d.get("ref_levels")
    if res == "ample" and lv:
        bits.append(f"background spans {lv:.0f} levels")
    elif res == "coarse" and lv:
        bits.append(f"background spans only {lv:.0f} quantisation steps, "
                    f"undithered — shape fine, amplitude stepped")
    elif res == "coarse-dithered" and lv:
        bits.append(f"only {lv:.1f} levels, but dithered by noise "
                    f"(shape usable, amplitude soft)")
    elif res == "pinned" and lv is not None:
        bits.append(f"⚠ background pinned to {max(lv, 1.0):.0f} "
                    f"level(s) and undithered — it cannot resolve "
                    f"your gradient; prefer a FITS tree")
    return ("  " + chr(0xB7) + "  ").join(b for b in bits if b)


def _quantisation(ref: np.ndarray, valid: np.ndarray | None = None,
                  *, max_px: int = 400_000) -> dict:
    """
    Measure a reference's quantisation step and its noise, from the DATA.

    Derived rather than inferred from the tile format on purpose: a
    user-supplied reference can be an 8-bit PNG off disk, with no HiPS
    provenance to consult, and that is exactly the case where the user
    has no idea what they handed over.

    Returns {step, lattice, distinct, hf}:
      step     the smallest positive gap between distinct values, or None
               if the values do not sit on a regular lattice (i.e. the
               data is effectively continuous and quantisation is moot)
      lattice  share of gaps that are near-integer multiples of step --
               the test for "is this really a lattice"
      distinct how many distinct values the cutout carries
      hf       robust per-pixel scatter with the smooth part removed.
               This decides whether the quantisation is DITHERED: if the
               noise is wider than a step, sky pixels straddle several
               levels and averaging a patch recovers sub-level structure.
               If it is not, every sky pixel reads the same integer and
               the gradient is simply gone.
    """
    a = np.asarray(ref, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    fin = np.isfinite(a)
    if valid is not None:
        v2 = np.asarray(valid, dtype=bool)
        if v2.shape[:2] == a.shape[:2]:
            fin = fin & v2
    out = {"step": None, "lattice": 0.0, "distinct": 0, "hf": 0.0}
    if int(fin.sum()) < 256:
        return out

    # high-frequency scatter, before any subsampling breaks adjacency
    try:
        if cv2 is not None:
            filled = np.where(fin, a, float(np.nanmedian(a[fin])))
            sm = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 4.0,
                                  borderType=cv2.BORDER_REFLECT)
            r = (a - sm)[fin]
            out["hf"] = float(np.median(np.abs(r - np.median(r)))) * 1.4826
    except Exception as e:
        _log(f"quantisation: could not measure the noise: {_brief(e)}")

    v = a[fin].astype(np.float64)
    if v.size > max_px:
        v = v[:: int(np.ceil(v.size / max_px))]
    u = np.unique(v)
    out["distinct"] = int(u.size)
    if u.size < 4:
        # Degenerate: fewer than 4 values in the whole cutout. Report the
        # gap anyway, since that IS the finding.
        if u.size >= 2:
            out["step"] = float(np.min(np.diff(u)))
            out["lattice"] = 1.0
        return out
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return out
    step = float(np.min(d))
    if step <= 0:
        return out
    k = d / step
    out["lattice"] = float(np.mean(np.abs(k - np.round(k)) < 0.02))

    # Three tests, and all three are needed.
    #
    # The lattice test alone is NOT enough, and the way it fails is
    # instructive: float32 is itself a lattice. Continuous data near 0.1
    # has a representable spacing of 2**-27, so every gap is an exact
    # integer multiple of the smallest one and the lattice test returns a
    # confident 100% on a step of 7.45e-09. Measured, not hypothesised.
    #
    #   lattice  - the gaps are multiples of one step
    #   ulp      - that step is well clear of the float grid at this
    #              magnitude, so it is a property of the DATA rather than
    #              of its storage
    #   sparsity - the values are few relative to the pixels. 8-bit data
    #              gives at most 256 values over tens of thousands of
    #              pixels; continuous data gives nearly one per pixel.
    mag = float(max(abs(float(u[0])), abs(float(u[-1])))) or 1.0
    ulp = float(np.spacing(np.float32(mag)))
    sparse = out["distinct"] <= max(4, int(0.2 * v.size))
    if out["lattice"] >= 0.90 and step > 8.0 * ulp and sparse:
        out["step"] = step
    return out


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
    channels: list | None = None,
    work_long_side: int = 1024,
    degree: int = 1,
    use_rbf: bool = True,
    rbf_smooth: float = 1.0,
    num_samples: int = 200,
    patch_size: int = 15,
    exclusion_fraction: float = 0.5,
    protect_k: float = 3.0,
    protect_grow: int = 3,
    seed: int = 42,
    min_coverage: float = 0.25,
    progress_cb=None,
    reference: np.ndarray | None = None,
    user_exclusion: np.ndarray | None = None,
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
    ref_info: dict = {}
    if reference is None:
        if not channels:
            channels = SURVEYS[DEFAULT_SURVEY]["channels"]
        ref = fetch_reference_channels(work_wcs, (h, w), list(channels),
                                       progress_cb=progress_cb,
                                       info=ref_info)
    else:
        # A caller-supplied reference has no HiPS provenance to report.
        # Say that, rather than leaving the field blank and letting the
        # UI imply a tree it never fetched.
        ref_info = {"route": "supplied", "format": None, "bits": None}
        ref = np.asarray(reference, dtype=np.float32)
        if ref.shape[:2] != (h, w):
            raise SurveyFetchError(
                f"Supplied reference shape {ref.shape[:2]} != work grid {(h, w)}."
            )
    if ref.ndim == 3 and ref.shape[2] == 1:
        ref = ref[..., 0]
    ref_colour = (ref.ndim == 3)

    # A pixel is usable only where EVERY reference channel has data, otherwise
    # one channel's footprint edge would silently fit against a hole in another.
    valid = np.isfinite(ref)
    if ref_colour:
        valid = valid.all(axis=2)
    # Luminance drives point selection and the structure veto: the veto is a
    # statement about where real sky structure is, which is one fact about the
    # sky, not three.
    ref_luma = _to_luminance(ref) if ref_colour else ref
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
    img_small = _downsample_area(img, max(1, int(round(max(H, W) / max(h, w)))))
    if img_small.shape[:2] != (h, w):
        img_small = _resize_to(img_small, (h, w))

    # NaN outside the footprint would poison the fit; fill with the median of
    # the covered part so the model stays smooth. The fill is NOT sampled —
    # `valid` is passed through to the point generator below.
    ref_filled = ref.copy()
    if not valid.all():
        fill = float(np.nanmedian(ref)) if np.any(valid) else 0.0
        ref_filled[~valid] = fill
    ref_luma_filled = (_to_luminance(ref_filled) if ref_colour else ref_filled)

    if progress_cb:
        progress_cb("Choosing background sample points…")
    # Hand-drawn exclusions are a HARD veto, applied on top of the
    # survey-derived one. The survey decides what is real structure; this is
    # the user overriding that for something the survey cannot know about --
    # a dark nebula the survey is too shallow to show, a reflection, a
    # satellite trail. It only ever REMOVES candidate pixels.
    excl_small = None
    if user_exclusion is not None:
        e = np.asarray(user_exclusion)
        if e.shape[:2] != (h, w):
            e = _resize_to(e.astype(np.float32), (h, w)) > 0.5
        excl_small = np.asarray(e, dtype=bool)
        _log(f"user exclusion zones veto "
             f"{100.0 * float(excl_small.mean()):.1f}% of the work grid")

    pts, allowed, mask_info = _shared_sample_points(
        img_small, ref_luma_filled, valid=valid,
        user_exclusion=excl_small,
        num_samples=int(num_samples), patch_size=int(patch_size),
        exclusion_fraction=float(exclusion_fraction),
        protect_k=float(protect_k), protect_grow=int(protect_grow),
        seed=int(seed),
    )

    # Both models get the SAME points. abe_run uses manual_points for the
    # polynomial AND the RBF stage, so this pins the sampling for both stages
    # of both fits — see _shared_sample_points for why that matters.
    common = dict(
        degree=int(degree), num_samples=int(num_samples), downsample=1,
        patch_size=int(patch_size), use_rbf=bool(use_rbf),
        rbf_smooth=float(rbf_smooth), return_background=True,
        legacy_prestretch=False, manual_points=pts, progress_cb=None,
    )

    if progress_cb:
        progress_cb("Modelling image background…")
    _, bg_t = abe_run(img_small, **common)

    if progress_cb:
        progress_cb("Modelling reference background…")
    _, bg_r = abe_run(ref_filled, **common)
    bg_r = np.asarray(bg_r, dtype=np.float32)
    if bg_r.ndim == 3 and bg_r.shape[2] == 1:
        bg_r = bg_r[..., 0]

    # ---- per-channel fit -------------------------------------------------
    if progress_cb:
        progress_cb("Fitting image background against reference…")

    bg_t = np.asarray(bg_t, dtype=np.float32)
    if mono and bg_t.ndim == 3:
        bg_t = bg_t[..., 0]

    # Fit over the whole covered frame, INCLUDING the vetoed region.
    #
    # Restricting this to sample-constrained pixels looks principled and is
    # measurably worse: tried it, and b fell from 0.122 to 0.081 against a
    # known truth of 0.129, with recovery correlation 0.983 -> 0.936. Two
    # reasons. The vetoed region is not invented data — both backgrounds
    # there are smooth predictions from the SAME points through the SAME
    # kernel, so their extrapolation errors are correlated and cancel in a
    # linear fit. And excluding it throws away the high-B_r end of the
    # range, which is the leverage that determines b at all. Restricted-range
    # regression, straightforwardly.
    #
    # `fit_region` is kept for reporting only — see the diagnostics below.
    fit_region = np.asarray(allowed, dtype=bool) & valid

    def _ref_plane(c: int) -> np.ndarray:
        """
        Which reference plane target channel `c` is fitted against.

        Colour reference  -> channel c to channel c. Light pollution is
            strongly colour-selective, so red sky and blue sky need to be
            explained by red and blue structure, not by one grey average.
        Mono reference    -> every channel against the same plane, but each
            still solves its own (a, b). The structure is assumed to have the
            same SHAPE in all channels and a different amplitude, which is a
            real approximation but a far better one than no reference.
        """
        if bg_r.ndim == 2:
            return bg_r
        return bg_r[..., min(c, bg_r.shape[2] - 1)]

    nch = 1 if bg_r.ndim == 2 else bg_r.shape[2]
    _log(f"fitting {'mono' if mono else '%d' % bg_t.shape[2]} target "
         f"channel(s) against a {nch}-channel reference "
         f"({'matched per channel' if (not mono and nch > 1) else 'shared plane'})")

    coeffs: list[tuple[float, float]] = []
    if mono:
        rp = bg_r if bg_r.ndim == 2 else _to_luminance(bg_r)
        rm = np.where(valid, rp, np.nan).astype(np.float32)
        a, b, _ = _robust_linear_fit(rm, bg_t)
        grad_small = (bg_t - (a + b * rp)).astype(np.float32)
        grad_small -= float(np.nanmedian(grad_small))
        coeffs.append((a, b))
    else:
        planes = []
        for c in range(bg_t.shape[2]):
            rp = _ref_plane(c)
            rm = np.where(valid, rp, np.nan).astype(np.float32)
            a, b, _ = _robust_linear_fit(rm, bg_t[..., c])
            g = (bg_t[..., c] - (a + b * rp)).astype(np.float32)
            g -= float(np.nanmedian(g))
            planes.append(g)
            coeffs.append((a, b))
        grad_small = np.stack(planes, axis=-1).astype(np.float32)

    # ---- is the reference actually doing anything? -----------------------
    # Honest self-check. With a strict veto the reference background is
    # nearly flat (a gradient-corrected survey, sampled only on sky), so
    # b*B_r collapses toward a constant and the method quietly degenerates
    # into ADBE with a survey-derived exclusion mask. That is still a useful
    # tool, but it is NOT reference-based correction, and the difference
    # should be visible rather than assumed.
    def _span(a2d, m=None):
        v = np.asarray(a2d, dtype=np.float32)
        v = v[m] if m is not None else v.ravel()
        v = v[np.isfinite(v)]
        if v.size < 8:
            return 0.0
        return float(np.percentile(v, 99.5) - np.percentile(v, 0.5))

    ref_span = _span(bg_r, valid)
    grad_span = _span(grad_small)
    explained = [abs(b) * ref_span for _, b in coeffs]

    # Two amplitudes, split into shares that sum to 1:
    #
    #   E = |b| * span(B_r)   what the reference accounted for
    #   R = span(G)           what is left over, i.e. YOUR gradient
    #   share = E / (E + R)
    #
    # The denominator used to be R alone, which could and did exceed 100%
    # -- a real fetch reported "explains 164%". That was not a display
    # bug: SPANS DO NOT ADD. span(X + Y) equals span(X) + span(Y) only
    # when X and Y are perfectly correlated, and the fitted component and
    # its residual are generally not, so E can be larger than R while
    # both are correct measurements of their own component. Dividing one
    # by the other therefore has no upper bound and is not a fraction of
    # anything.
    #
    # E / (E + R) is bounded in [0, 1] by construction and reads as what
    # the number was always meant to mean: of the total structure the two
    # components describe between them, this much came from the survey and
    # the rest is yours. For a least-squares fit the exact orthogonal
    # decomposition is in QUADRATURE rather than linear -- E^2/(E^2+R^2)
    # -- but linear is the right choice here for two reasons: the spans
    # are robust percentile measures rather than variances, so exact
    # additivity is not available anyway; and in the small-E regime where
    # the warning below fires, E/(E+R) tends to E/R, which keeps the 5%
    # threshold calibrated exactly as it was. Quadrature would move that
    # trip point by a factor of about four.
    _E = max(explained) if explained else 0.0
    _R = grad_span
    _total = _E + _R
    diag = {
        "ref_bg_span": ref_span,
        "gradient_span": grad_span,
        "explained_amplitude": _E,
        "explained_by_reference": explained,
        "reference_share": (_E / _total) if _total > 0 else 0.0,
        "residual_share": (_R / _total) if _total > 0 else 0.0,
        "ref_info": ref_info,
    }

    # ---- can the reference RESOLVE a gradient at all? -------------------
    # Separate question from reference_share, and not covered by it. A
    # reference quantised so coarsely that the sky sits on one or two
    # levels can still produce a healthy b -- fitted to a staircase. This
    # measures the resolution directly: how many quantisation steps does
    # the modelled background actually span, and is the quantisation
    # dithered by noise (in which case patch averaging recovers far more
    # than the step suggests) or not (in which case the structure is gone).
    qz = _quantisation(ref, valid)
    diag.update({
        "ref_step": qz["step"],
        "ref_distinct": qz["distinct"],
        "ref_hf": qz["hf"],
        "ref_levels": None,
        "ref_dither": None,
        "ref_resolution": "continuous",
    })
    _step = qz["step"]
    if _step and _step > 0:
        _levels = ref_span / _step
        _dither = qz["hf"] / _step
        diag["ref_levels"] = float(_levels)
        diag["ref_dither"] = float(_dither)
        # Four cases, because levels and dithering fail independently.
        # Measured on a real DSS2 8-bit colour fetch: 8.5 levels with
        # noise/step 0.88 -- just over the level bar and just under the
        # dither bar. Calling that "ample" would be the tool flattering
        # itself, and calling it "pinned" would be alarmism; 8 levels of
        # ramp describe a shape perfectly well, they just describe its
        # amplitude in 8 steps.
        if _levels >= 32.0 or (_levels >= _REF_MIN_LEVELS
                               and _dither >= 1.0):
            diag["ref_resolution"] = "ample"
        elif _levels >= _REF_MIN_LEVELS:
            diag["ref_resolution"] = "coarse"
        elif _dither >= 1.0:
            diag["ref_resolution"] = "coarse-dithered"
        else:
            diag["ref_resolution"] = "pinned"
        _log(f"reference quantisation: step={_step:.6g}  "
             f"{qz['distinct']} distinct values  "
             f"background spans {_levels:.1f} steps  "
             f"noise/step={_dither:.2f}  -> {diag['ref_resolution']}")
        if diag["ref_resolution"] == "pinned":
            _log("  WARNING: the reference background is pinned to about "
                 f"{max(1.0, _levels):.0f} quantisation level(s) and the "
                 "noise is narrower than one step, so there is nothing "
                 "for patch averaging to recover. Prefer a survey that "
                 "publishes FITS tiles.")
        elif diag["ref_resolution"] == "coarse-dithered":
            _log("  NOTE: coarse quantisation, but dithered by noise -- "
                 "each sample patch averages thousands of pixels, which "
                 "recovers sub-step structure. The SHAPE is usable; treat "
                 "the amplitude with mild suspicion.")
        elif diag["ref_resolution"] == "coarse":
            _log(f"  NOTE: the background spans only {_levels:.1f} "
                 f"quantisation steps and the noise is narrower than one "
                 f"step, so there is no dither to average away. The "
                 f"gradient SHAPE is described, its amplitude in "
                 f"{_levels:.0f} discrete steps. A FITS tree would be "
                 f"better if one is offered for this field.")
    else:
        _log(f"reference quantisation: continuous "
             f"({qz['distinct']} distinct values, lattice "
             f"{100.0 * qz['lattice']:.0f}%) -- no step to report")
    _log(f"reference bg span={ref_span:.6g}  residual gradient span="
         f"{grad_span:.6g}  b*span={['%.4g' % e for e in explained]}  -> "
         f"reference accounts for {diag['reference_share']*100:.1f}% of the "
         f"structure, your own gradient for "
         f"{diag['residual_share']*100:.1f}%")
    if diag["reference_share"] < 0.05:
        _log("  NOTE: the reference background is essentially flat, so this "
             "run is ADBE with a survey-derived exclusion mask rather than a "
             "reference-based correction. Expected on a field whose only "
             "structure is a bright object that got vetoed.")

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
        "target_small": img_small,
        "bg_target": bg_t,
        "bg_ref": bg_r,
        "points": pts,
        "allowed": allowed,
        "user_exclusion": excl_small,
        "mask_info": mask_info,
        "diag": diag,
        "fit_region": fit_region,
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
#                       User-supplied reference (dual scope)
# =============================================================================
# A second scope on the same mount, shooting the same field at a shorter focal
# length, is a BETTER gradient reference than any survey: it saw the same sky,
# the same night, through the same light pollution and the same airglow. A
# survey tells you where real nebulosity is; a simultaneous wide frame tells
# you that AND what the sky was doing at the time.
#
# The requirement is only that it covers the target's field and is itself
# reasonably flat -- the fit solves a + b*B_r, so a reference carrying the
# SAME gradient would absorb it into b and leave nothing to subtract. That is
# the one failure mode worth warning about, and reference_share already
# measures it: a wide frame with the same gradient explains ~100% of it, which
# reads as "nothing to correct" rather than as a wrong answer.


def _reproject_plane(plane: np.ndarray, src_wcs: WCS, dst_wcs: WCS,
                     out_hw: tuple[int, int]) -> np.ndarray:
    """One plane onto the work grid. Outside the input footprint -> NaN."""
    from reproject import reproject_interp
    out, _ = reproject_interp(
        (np.asarray(plane, dtype=np.float32), src_wcs),
        dst_wcs, shape_out=(int(out_hw[0]), int(out_hw[1])),
    )
    return np.asarray(out, dtype=np.float32)


def reference_from_image(img: np.ndarray, header, work_wcs: WCS,
                         out_hw: tuple[int, int], *,
                         progress_cb=None) -> np.ndarray:
    """
    Reproject the user's own frame onto the target's work grid.

    Returns (h, w) for a mono input or (h, w, 3) for colour, matching what
    fetch_reference_channels produces, so everything downstream is identical
    whether the reference came off the internet or off a second scope.

    NaN outside the input footprint is deliberate and is the same signal a
    partially-covered survey gives: survey_gradient_model's `valid` mask and
    its min_coverage check handle it already, so a reference frame that is
    narrower than the target fails with the usual coverage message instead of
    silently fitting against invented pixels.
    """
    src = _celestial_wcs_from_header(header)
    if src is None:
        raise SurveyFetchError(
            "The reference image has no usable WCS, so it cannot be aligned "
            "to your target. Plate-solve it first."
        )

    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    h, w = int(out_hw[0]), int(out_hw[1])

    _log("=" * 62)
    _log(f"reference_from_image  src={a.shape}  -> grid={w}x{h}")
    _log(_wcs_summary(src, "user reference"))
    _log(_wcs_summary(work_wcs, "work grid"))

    if progress_cb:
        progress_cb("Reprojecting your reference onto the target grid…")

    if a.ndim == 2:
        out = _reproject_plane(a, src, work_wcs, (h, w))
    else:
        out = np.stack(
            [_reproject_plane(a[..., c], src, work_wcs, (h, w))
             for c in range(min(3, a.shape[2]))],
            axis=-1,
        )
    _log("  " + _array_stats(out, "user reference reprojected"))
    finite = float(np.isfinite(out).mean())
    if finite <= 0.0:
        raise SurveyFetchError(
            "Your reference image does not overlap this field at all after "
            "reprojection.\n\nBoth images are plate-solved, so this means "
            "they are pointing at different parts of the sky. Check that the "
            "reference really is the same target."
        )
    _log(f"  overlap    : {finite * 100:.1f}% of the work grid")
    return out


def ensure_solved_header(parent, img: np.ndarray, header, *, label: str,
                         settings=None, ask: bool = True):
    """
    Return a header carrying a celestial WCS, plate-solving if need be.

    Returns (header, solved_now). Raises SurveyFetchError when there is still
    no WCS afterwards -- SRGC cannot do anything without one, in either image.

    The solve is CONFIRMED, not automatic. A blind solve walks a quad index
    and can run for minutes, and it needs the Gaia databases downloaded; doing
    that silently because somebody opened a dialog would be a long unexplained
    freeze. plate_solve_doc_inplace already falls back to the blind path on its
    own when the header carries no pointing, so this does not reimplement any
    of that -- it only decides whether to start it.
    """
    if _celestial_wcs_from_header(header) is not None:
        return header, False

    _log(f"{label}: no celestial WCS in the header")
    if not ask:
        raise SurveyFetchError(
            f"{label} is not plate-solved and this run cannot ask. "
            f"Solve it first, or run with ask=True."
        )

    if parent is not None:
        reply = QMessageBox.question(
            parent, "Plate solve needed",
            f"{label} has no WCS, so the reference cannot be matched to it.\n\n"
            f"Run the Gaia plate solver on it now?\n\n"
            f"With no pointing in the header this falls back to a BLIND solve, "
            f"which searches the quad index and can take a few minutes. It "
            f"needs the Gaia databases to be downloaded.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            raise SurveyFetchError(
                f"{label} is not plate-solved. SRGC fetches and aligns the "
                f"reference by sky coordinates, so a solution is required."
            )

    # Lazy: plate_solver pulls in star_alignment and a good deal more, and a
    # module-level import here would make SRGC pay for it on every app start.
    from setiastro.saspro.plate_solver import plate_solve_doc_inplace
    if settings is None:
        settings = QSettings()

    class _SolveDoc:
        """The shape plate_solve_doc_inplace expects, nothing more."""
        pass

    d = _SolveDoc()
    d.image = np.asarray(img, dtype=np.float32)
    d.metadata = {"original_header": header} if header is not None else {}

    _log(f"{label}: starting plate solve (blind if no pointing)…")
    ok, result = plate_solve_doc_inplace(parent=parent, doc=d, settings=settings)
    if not ok:
        raise SurveyFetchError(
            f"Plate solving {label.lower()} failed:\n\n{result}"
        )

    new_hdr = result
    if _celestial_wcs_from_header(new_hdr) is None:
        # Fall back to whatever the solver wrote into the doc metadata.
        meta = getattr(d, "metadata", {}) or {}
        for k in ("wcs_header", "original_header", "header"):
            cand = meta.get(k)
            if _celestial_wcs_from_header(cand) is not None:
                new_hdr = cand
                break
    if _celestial_wcs_from_header(new_hdr) is None:
        raise SurveyFetchError(
            f"The plate solve of {label.lower()} reported success but produced "
            f"no usable WCS."
        )
    _log(f"{label}: solved")
    return new_hdr, True

# =============================================================================
#                                   Dialog
# =============================================================================

# =============================================================================
#                            Interactive preview pane
# =============================================================================


class _PreviewLabel(QLabel):
    """The canvas: draws the pixmap, the polygons, and the one being drawn."""

    def __init__(self, pane):
        super().__init__(alignment=Qt.AlignmentFlag.AlignCenter)
        self._pane = pane
        # Deliberately small: this feeds the scroll area's size hint, and a
        # large value here is the other half of the growth ratchet above.
        self.setMinimumSize(QSize(80, 60))
        self.setScaledContents(False)
        self.setMouseTracking(True)
        # Needs focus to see the spacebar at all.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            pane = getattr(self, "_pane", None)
            if pane is not None:
                pane._space_down = True
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            pane = getattr(self, "_pane", None)
            if pane is not None:
                pane._space_down = False
                self.unsetCursor()
            ev.accept()
            return
        super().keyReleaseEvent(ev)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        # getattr throughout: a paint can arrive while the pane is still
        # being built, and raising in here aborts the process.
        pane = getattr(self, "_pane", None)
        if pane is None or not getattr(pane, "draw_enabled", False):
            return
        polys = list(getattr(pane, "polygons", None) or [])
        pending = getattr(pane, "pending", None)
        if pending is not None and len(pending) > 1:
            polys = polys + [pending]
        if not polys:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 90, 90), 2)
        p.setPen(pen)
        p.setBrush(QBrush(QColor(255, 90, 90, 60)))
        for poly in polys:
            qp = QPolygonF([QPointF(*pane.image_to_widget(pt))
                            for pt in poly])
            if len(qp) >= 2:
                p.drawPolygon(qp)
        p.end()


class _PreviewPane(QScrollArea):
    """
    One preview tab: a zoomable, pannable image with optional polygon
    drawing.

    Written here rather than lifted out of ABEDialog because ABE's version is
    woven through that dialog's own event filters and state; extracting it
    would mean refactoring a tool that works, to no benefit for either.
    Polygons are kept in ORIGINAL IMAGE coordinates, the same convention
    ABE uses, so the mask built from them is independent of zoom and of the
    preview's reduced render scale.
    """

    def __init__(self, parent=None, *, drawable: bool = False):
        super().__init__(parent)

        # EVERY attribute the overridden virtuals touch is set before the
        # first Qt call that can fire one. setWidget() alone triggers a
        # resize and a paint, and an AttributeError raised inside a Qt
        # virtual does not surface as a Python traceback -- PyQt aborts the
        # process, which looks exactly like a segfault with no output.
        self._buf = None          # rendered RGB888 uint8 (preview scale)
        self._pm = None           # QPixmap at 1:1 of _buf
        self._img_hw = (1, 1)     # ORIGINAL image shape the polygons live in
        self.scale = 1.0
        self.fit_mode = True
        self.draw_enabled = bool(drawable)
        self.polygons: list = []   # list of [(x, y), ...] in image coords
        self.pending = None
        self._panning = False
        self._pan_from = None
        self.on_polygons_changed = None
        # Re-entrancy guard. _apply_scale resizes the label, which can add
        # or remove a scrollbar, which resizes the viewport, which fires
        # resizeEvent, which calls _apply_scale again. In fit mode the two
        # sizes chase each other and the recursion never bottoms out -- it
        # overflows the stack and takes the process down with no traceback.
        self._scaling = False
        self._space_down = False

        self.setWidgetResizable(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.label = _PreviewLabel(self)
        self.setWidget(self.label)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.installEventFilter(self)
        self.viewport().installEventFilter(self)

    # ---- content ---------------------------------------------------------
    def set_image(self, buf, img_hw=None):
        """`buf` is an (h, w, 3) uint8 render; `img_hw` the ORIGINAL shape."""
        self._buf = np.ascontiguousarray(buf)
        h, w, _ = self._buf.shape
        qi = QImage(self._buf.data, w, h, self._buf.strides[0],
                    QImage.Format.Format_RGB888)
        self._pm = QPixmap.fromImage(qi.copy())
        if img_hw is not None:
            self._img_hw = (int(img_hw[0]), int(img_hw[1]))
        self._apply_scale()

    def has_image(self) -> bool:
        return self._pm is not None

    # ---- coordinate mapping ----------------------------------------------
    def _render_to_image(self) -> float:
        """Original-image pixels per rendered-preview pixel."""
        if self._buf is None:
            return 1.0
        return float(self._img_hw[1]) / max(1.0, float(self._buf.shape[1]))

    def widget_to_image(self, x: float, y: float) -> tuple:
        """Label coords -> ORIGINAL image coords."""
        if self._pm is None:
            return (0.0, 0.0)
        pw = self._pm.width() * self.scale
        ph = self._pm.height() * self.scale
        ox = max(0.0, (self.label.width() - pw) / 2.0)
        oy = max(0.0, (self.label.height() - ph) / 2.0)
        rx = (x - ox) / max(self.scale, 1e-9)
        ry = (y - oy) / max(self.scale, 1e-9)
        f = self._render_to_image()
        return (rx * f, ry * f)

    def image_to_widget(self, pt) -> tuple:
        """ORIGINAL image coords -> label coords."""
        if self._pm is None:
            return (0.0, 0.0)
        f = self._render_to_image()
        pw = self._pm.width() * self.scale
        ph = self._pm.height() * self.scale
        ox = max(0.0, (self.label.width() - pw) / 2.0)
        oy = max(0.0, (self.label.height() - ph) / 2.0)
        return (ox + (pt[0] / f) * self.scale, oy + (pt[1] / f) * self.scale)

    # ---- zoom ------------------------------------------------------------
    def _apply_scale(self):
        if getattr(self, "_pm", None) is None or getattr(self, "_scaling", True):
            return
        self._scaling = True
        try:
            self._apply_scale_inner()
        finally:
            self._scaling = False

    def _apply_scale_inner(self):
        if self.fit_mode:
            vp = self.viewport().size()
            sx = vp.width() / max(1, self._pm.width())
            sy = vp.height() / max(1, self._pm.height())
            self.scale = max(0.02, min(sx, sy))
        w = max(1, int(self._pm.width() * self.scale))
        h = max(1, int(self._pm.height() * self.scale))
        self.label.setPixmap(self._pm.scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.label.resize(max(w, self.viewport().width()),
                          max(h, self.viewport().height()))
        self.label.update()

    def zoom_in(self):
        self.fit_mode = False
        self.scale = min(8.0, self.scale * 1.25)
        self._apply_scale()

    def zoom_out(self):
        self.fit_mode = False
        self.scale = max(0.02, self.scale / 1.25)
        self._apply_scale()

    def zoom_fit(self):
        self.fit_mode = True
        self._apply_scale()

    def zoom_1_1(self):
        """1:1 against the ORIGINAL image, not the reduced render."""
        self.fit_mode = False
        self.scale = min(8.0, max(0.02, self._render_to_image()))
        self._apply_scale()

    # A QScrollArea reports its child widget's sizeHint as its own, and a
    # QLabel's sizeHint follows its pixmap. That closes a loop: a bigger
    # pixmap -> bigger sizeHint -> the layout grants more room -> bigger
    # viewport -> fit mode picks a bigger scale -> bigger pixmap. Each pass
    # ratchets the dialog up a little, which is what made the window creep
    # outward every time a polygon was released (the release repaints and
    # pumps events, giving the layout a chance to run). Pinning the hint to
    # a constant cuts the loop; the pane still expands to whatever the
    # layout gives it, it just stops ASKING for more.
    def sizeHint(self):
        return QSize(560, 420)

    def minimumSizeHint(self):
        return QSize(240, 180)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, "fit_mode", False):
            self._apply_scale()

    # ---- polygons ---------------------------------------------------------
    def clear_polygons(self):
        self.polygons = []
        self.pending = None
        self.label.update()
        self._notify()

    def undo_polygon(self):
        if self.pending is not None:
            self.pending = None
        elif self.polygons:
            self.polygons.pop()
        self.label.update()
        self._notify()

    def _notify(self):
        if callable(self.on_polygons_changed):
            try:
                self.on_polygons_changed()
            except Exception:
                pass

    def exclusion_mask(self, hw=None):
        """
        Bool mask, True INSIDE an exclusion zone, at `hw` (default: the
        original image shape). None when nothing has been drawn.
        """
        polys = [p for p in self.polygons if p is not None and len(p) >= 3]
        if not polys:
            return None
        H, W = (self._img_hw if hw is None else (int(hw[0]), int(hw[1])))
        m = np.zeros((H, W), dtype=np.uint8)
        sy = H / float(max(1, self._img_hw[0]))
        sx = W / float(max(1, self._img_hw[1]))
        if cv2 is None:
            _log("exclusion zones need cv2 to rasterise; ignoring them")
            return None
        for poly in polys:
            pts = np.array([[int(round(x * sx)), int(round(y * sy))]
                            for (x, y) in poly], dtype=np.int32)
            if np.unique(pts, axis=0).shape[0] < 3:
                continue
            cv2.fillPoly(m, [pts], 1)
        return m.astype(bool) if m.any() else None

    # ---- interaction ------------------------------------------------------
    def eventFilter(self, obj, ev):
        if getattr(self, "label", None) is None:
            return False
        et = ev.type()
        if et == ev.Type.Wheel:
            # Plain wheel zooms, as in ADBE -- the preview is the thing you
            # want to scale, not scroll.
            if self._pm is not None:
                (self.zoom_in if ev.angleDelta().y() > 0 else self.zoom_out)()
                ev.accept()
                return True
            return False

        if obj is self.label and self._pm is not None:
            if et == ev.Type.MouseButtonPress:
                # Same convention ADBE uses, so the two tools feel alike:
                #   left-drag                      -> draw an exclusion zone
                #   middle-drag / Ctrl+left-drag   -> pan
                # plus space+left-drag, which is the reflex a lot of
                # image editors train.
                pan = (ev.button() == Qt.MouseButton.MiddleButton
                       or (ev.button() == Qt.MouseButton.LeftButton
                           and ((ev.modifiers()
                                 & Qt.KeyboardModifier.ControlModifier)
                                or self._space_down)))
                if (not pan) and self.draw_enabled \
                        and ev.button() == Qt.MouseButton.LeftButton:
                    pos = ev.position()
                    self.pending = [self.widget_to_image(pos.x(), pos.y())]
                    self.label.setFocus()
                    return True
                self._panning = True
                self._pan_from = ev.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True

            if et == ev.Type.MouseMove:
                if self.pending is not None:
                    pos = ev.position()
                    self.pending.append(self.widget_to_image(pos.x(), pos.y()))
                    self.label.update()
                    return True
                if self._panning and self._pan_from is not None:
                    d = ev.position() - self._pan_from
                    self._pan_from = ev.position()
                    hb = self.horizontalScrollBar()
                    vb = self.verticalScrollBar()
                    hb.setValue(hb.value() - int(d.x()))
                    vb.setValue(vb.value() - int(d.y()))
                    return True

            if et == ev.Type.MouseButtonRelease:
                if self.pending is not None:
                    # Keep it only if it encloses something. A stray click
                    # should not leave an invisible degenerate zone behind.
                    if len(self.pending) >= 3:
                        self.polygons.append(self.pending)
                        self._notify()
                    self.pending = None
                    self.label.update()
                    return True
                self._panning = False
                self._pan_from = None
                self.unsetCursor()
                return True
        return super().eventFilter(obj, ev)

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
        # A host that was down last time the tool was open may be back, and a
        # transient network drop should not permanently retire a good mirror.
        # Reopening the dialog is the natural "try everything again" gesture;
        # repeated runs within one session keep skipping the dead hosts.
        reset_dead_hosts()
        self.setWindowTitle(self.tr("Survey Reference Gradient Correction"))
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
        self._avail_probe = None
        self._avail_conn = None

        # Mirror the [SRGC] trace into the main window log so the user can
        # copy it out of the app rather than hunting for a console.
        _hostlog = getattr(parent, "_log", None)
        if callable(_hostlog):
            set_log_sink(_hostlog)

        # Where the reference comes from. A second scope on the same mount
        # beats any survey for this job -- same sky, same night, same light
        # pollution -- so it is a first-class source, not a fallback.
        self.cmb_source = QComboBox()
        self.cmb_source.addItem(self.tr("Survey (fetched online)"), "survey")
        self.cmb_source.addItem(
            self.tr("My own image (dual scope / wider frame)"), "user")
        self.cmb_source.setToolTip(
            "Survey: a published all-sky dataset of the same field.\n\n"
            "My own image: a frame YOU took of the same field, typically the\n"
            "wide scope on a dual rig. It must cover the whole target field\n"
            "and be plate-solved (SRGC will offer to solve it if not).\n\n"
            "A simultaneous wide frame is the better reference when you have\n"
            "one: it saw the same sky on the same night. The one thing to\n"
            "watch is that it must not carry the SAME gradient -- if it does,\n"
            "b absorbs it and there is nothing left to subtract. Check the\n"
            "reference share in diagnostics."
        )
        self.cmb_source.currentIndexChanged.connect(self._source_changed)

        self.cmb_survey = QComboBox()
        for name in SURVEYS:
            self.cmb_survey.addItem(name, userData=name)
        self.cmb_survey.currentTextChanged.connect(self._survey_changed)

        # User reference: an open document, or a file on disk.
        self._user_ref = None        # (image, header, label) once loaded
        self._ref_cache = {}         # cache_key -> fetched reference array
        self._ref_info = {}          # provenance of the reference on screen
        self._ref_info_cache = {}    # cache_key -> that provenance
        self._disp_cache = {}        # (mode, before, strength, patch) -> (buf, status)
        self.cmb_refdoc = QComboBox()
        self.cmb_refdoc.setToolTip(
            "Pick one of the images you already have open.\n"
            "Refreshed each time this dialog is shown.")
        self.cmb_refdoc.currentIndexChanged.connect(self._refdoc_changed)
        self.btn_reffile = QPushButton(self.tr("From file…"))
        self.btn_reffile.setToolTip(
            "Load a reference frame straight off disk, without opening it\n"
            "as a view first.")
        self.btn_reffile.clicked.connect(self._pick_reference_file)
        self.lbl_refinfo = QLabel("")
        self.lbl_refinfo.setWordWrap(True)
        self.lbl_refinfo.setStyleSheet("font-size: 10px;")

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

        self.sp_exclude = QSpinBox()
        self.sp_exclude.setRange(0, 95)
        self.sp_exclude.setSingleStep(5)
        self.sp_exclude.setValue(50)
        self.sp_exclude.setSuffix(" %")
        self.sp_exclude.setToolTip(
            "Fraction of the SURVEY's brightest pixels that are off-limits\n"
            "to sampling (_exclude_bright_regions, same as gradient_abe and\n"
            "the stacking suite use).\n\n"
            "The survey decides, not your frame — your gradient must not\n"
            "disqualify a location or the fit starves on the bright side.\n\n"
            "Raise it on a field dominated by a big object (M31, M42).\n"
            "Lower it on sparse fields where 50% is needlessly strict."
        )

        self.sp_protect = QDoubleSpinBox()
        self.sp_protect.setRange(0.5, 20.0)
        self.sp_protect.setSingleStep(0.5)
        self.sp_protect.setValue(3.0)
        self.sp_protect.setDecimals(1)
        self.sp_protect.setToolTip(
            "Signal protection on the reference, in MAD-sigma units — the\n"
            "same median + k*MAD threshold ADBE's multiscale stage uses, so\n"
            "k means the same thing in both tools.\n\n"
            "Thresholded, then dilated and blurred so a structure's faint\n"
            "wings are covered too, then interior holes are filled — which\n"
            "is what stops a point landing in M31's dust lanes.\n\n"
            "LOWER protects more. 3 is a reasonable default; drop to 1.5-2\n"
            "on a field with a large bright object."
        )

        self.chk_rbf = QCheckBox(self.tr("RBF refinement on both backgrounds"))
        self.chk_rbf.setChecked(True)

        # ---- multiplicative refinement -----------------------------------
        # Dust motes, internal reflections and filter-edge shadows are
        # MULTIPLICATIVE -- a shadow scales throughput -- and they have hard
        # edges. The additive fit above cannot touch them at any degree, and
        # neither can any smooth background model. ADBE already solves this
        # with a log-space multiscale band division; the same core is reused
        # here rather than reinvented.
        self.chk_ms = QCheckBox(self.tr("Remove dust motes / reflections "
                                        "(multiscale, multiplicative)"))
        self.chk_ms.setChecked(False)
        self.chk_ms.setToolTip(
            "A second, MULTIPLICATIVE pass for defects the additive fit\n"
            "structurally cannot reach: dust motes, internal reflections,\n"
            "filter-edge shadows.\n\n"
            "These attenuate throughput rather than adding light, and they\n"
            "have hard edges, so no background surface of any degree models\n"
            "them. This divides out a mid-scale band instead.\n\n"
            "Runs AFTER the gradient correction, on the corrected image.")
        self.chk_ms.toggled.connect(self._ms_toggled)

        self.sp_ms_lo = QSpinBox(); self.sp_ms_lo.setRange(1, 12)
        self.sp_ms_lo.setValue(6)
        self.sp_ms_hi = QSpinBox(); self.sp_ms_hi.setRange(1, 12)
        self.sp_ms_hi.setValue(8)
        for _sp in (self.sp_ms_lo, self.sp_ms_hi):
            _sp.setToolTip(
                "Wavelet scales the defect is estimated from. 6-8 is the\n"
                "band dust motes and reflections live in: coarse enough to\n"
                "miss stars and noise, fine enough to keep their edges.\n\n"
                "Lower catches smaller/sharper defects; higher drifts toward\n"
                "the smooth gradient the additive stage already removed.")

        self.sp_ms_strength = QDoubleSpinBox()
        self.sp_ms_strength.setRange(0.0, 2.0)
        self.sp_ms_strength.setSingleStep(0.1)
        self.sp_ms_strength.setDecimals(2)
        self.sp_ms_strength.setValue(1.0)

        self.sp_ms_protect = QDoubleSpinBox()
        self.sp_ms_protect.setRange(0.5, 20.0)
        self.sp_ms_protect.setSingleStep(0.5)
        self.sp_ms_protect.setDecimals(1)
        self.sp_ms_protect.setValue(3.0)
        self.sp_ms_protect.setToolTip(
            "Signal protection for the refinement, in MAD-sigma units.\n"
            "LOWER protects more.")

        self.chk_ms_residual = QCheckBox(
            self.tr("Include coarsest residual"))
        self.chk_ms_residual.setChecked(False)
        self.chk_ms_residual.setToolTip(
            "Fold everything COARSER than the band above into the\n"
            "multiplicative correction as well, instead of stopping at\n"
            "the top of the band.\n\n"
            "Leave OFF unless a broad multiplicative falloff survives\n"
            "the additive stage — vignetting the flat missed, say. The\n"
            "reason to be more careful with it here than in ADBE: the\n"
            "smooth background has ALREADY been handled additively, and\n"
            "handled against a reference that says which large-scale\n"
            "structure is real sky. The coarsest residual is exactly\n"
            "where that structure lives, so folding it in can divide out\n"
            "nebulosity the survey just finished vouching for, on top of\n"
            "double-correcting the gradient.\n\n"
            "Costs time too: the full pyramid has to be built rather\n"
            "than stopping at the top of the band."),

        self.chk_ms_stars = QCheckBox(self.tr("Remove stars first (DarkStar)"))
        self.chk_ms_stars.setChecked(True)
        self.chk_ms_stars.setToolTip(
            "Estimate the defect band on a STARLESS copy so stars do not\n"
            "pollute it, while the protection mask is still built on the\n"
            "star-inclusive image. This pairing is what makes the band\n"
            "clean; leave it on unless DarkStar is unavailable.")

        self.chk_ms_survey_protect = QCheckBox(
            self.tr("Protect real structure using the survey"))
        self.chk_ms_survey_protect.setChecked(True)
        self.chk_ms_survey_protect.setToolTip(
            "Use the SURVEY's own veto mask to decide what must not be\n"
            "touched by the refinement.\n\n"
            "This is the one thing SRGC knows that ADBE cannot: ADBE builds\n"
            "its protection from the image itself, so a bright extended blob\n"
            "looks the same whether it is a real nebula or an internal\n"
            "reflection. Here the survey says which it is — real structure is\n"
            "kept from the untouched source, and anything sharp sitting on\n"
            "sky the survey says is empty is treated as a defect.")

        _msf = QFormLayout()
        _msf.setContentsMargins(0, 0, 0, 0)
        _band = QHBoxLayout(); _band.setContentsMargins(0, 0, 0, 0)
        _band.addWidget(self.sp_ms_lo); _band.addWidget(QLabel("to"))
        _band.addWidget(self.sp_ms_hi); _band.addStretch(1)
        _bandw = QWidget(); _bandw.setLayout(_band)
        _msf.addRow(self.tr("Scales:"), _bandw)
        _msf.addRow(self.chk_ms_residual)
        _msf.addRow(self.tr("Refine strength:"), self.sp_ms_strength)
        _msf.addRow(self.tr("Protect k (MAD):"), self.sp_ms_protect)
        _msf.addRow(self.chk_ms_stars)
        _msf.addRow(self.chk_ms_survey_protect)
        self.w_ms = QWidget(); self.w_ms.setLayout(_msf)
        self.w_ms.setVisible(False)

        self.sp_samples = QSpinBox()
        self.sp_samples.setRange(20, 4000)
        self.sp_samples.setSingleStep(20)
        self.sp_samples.setValue(200)
        self.sp_samples.setToolTip(
            "Background sample points, placed by the same star-avoiding\n"
            "gradient descent ADBE uses.\n\n"
            "One set of points is generated and used for BOTH background\n"
            "models, on the per-pixel maximum of the two images — so a point\n"
            "only counts as background where it is dim in your frame AND in\n"
            "the reference.\n\n"
            "Too many points on a nebula-filled field starts sampling the\n"
            "nebula itself; too few and the fit cannot follow a real gradient."
        )

        self.sp_patch = QSpinBox()
        self.sp_patch.setRange(5, 151)
        self.sp_patch.setSingleStep(2)
        self.sp_patch.setValue(15)
        self.sp_patch.setToolTip(
            "Patch size in work-grid pixels for each sample point.\n\n"
            "The descent scores a patch by its 25th percentile, so a larger\n"
            "patch is more robust against stars but blurs the distinction\n"
            "between sky and faint extended signal."
        )

        self.sp_strength = QSpinBox()
        # Past 100% is deliberate over-correction. No physical reason to
        # want it, but the gradient model is an estimate, and pushing past
        # it is a quick way to check whether the residual you are still
        # looking at is actually the gradient or something else.
        self.sp_strength.setRange(0, 150)
        self.sp_strength.setValue(100)
        self.sp_strength.setSuffix(" %")
        self.sp_strength.setToolTip(
            "Percent of the modelled gradient to subtract.\n\n"
            "100% is the fit. Below that leaves some in; above it\n"
            "over-corrects, tipping the sky the other way. There is no\n"
            "physical reason to want that, but the model is an estimate\n"
            "and pushing past it is a quick way to tell a real gradient\n"
            "residual from something the reference never explained.")
        # Strength changes what Corrected looks like, so it has to redraw.
        # It never did, which is only survivable because the redraw used to
        # be too slow to drag anyway.
        self.sp_strength.valueChanged.connect(self._redraw)

        form = QFormLayout()
        form.addRow(self.tr("Reference source:"), self.cmb_source)

        _refrow = QHBoxLayout()
        _refrow.setContentsMargins(0, 0, 0, 0)
        _refrow.addWidget(self.cmb_refdoc, 1)
        _refrow.addWidget(self.btn_reffile, 0)
        self.w_userref = QWidget()
        self.w_userref.setLayout(_refrow)

        self.row_survey_label = QLabel(self.tr("Reference survey:"))
        self.row_userref_label = QLabel(self.tr("Reference image:"))
        form.addRow(self.row_survey_label, self.cmb_survey)
        form.addRow(self.row_userref_label, self.w_userref)
        form.addRow("", self.lbl_refinfo)
        form.addRow(self.tr("Work resolution:"), self.sp_work)
        form.addRow(self.tr("Background degree:"), self.sp_degree)
        form.addRow(self.tr("Sample points:"), self.sp_samples)
        form.addRow(self.tr("Patch size (px):"), self.sp_patch)
        form.addRow(self.tr("Exclude brightest:"), self.sp_exclude)
        form.addRow(self.tr("Protect k (MAD):"), self.sp_protect)
        form.addRow(self.chk_rbf)
        form.addRow(self.chk_ms)
        form.addRow("", self.w_ms)
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

        # One tab per thing worth looking at, instead of a dropdown over a
        # single canvas. Tab 1 is the one you interact with -- zoom, before/
        # after, and drawing exclusion zones -- so it carries the full
        # toolbar; the rest are read-only diagnostics. The two background
        # tabs carry one control of their own, the display mode for the
        # view they show, because that is the only place it means anything.
        self.tabs = QTabWidget()
        self.pane_target = _PreviewPane(drawable=True)
        self.pane_ref = _PreviewPane()
        self.pane_grad = _PreviewPane()
        self.pane_bgt = _PreviewPane()
        self.pane_bgr = _PreviewPane()
        self.pane_pts = _PreviewPane()
        self.pane_target.on_polygons_changed = self._exclusions_changed

        def _tb(txt, tip, slot):
            b = QToolButton()
            b.setText(txt)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            return b

        self.lbl_draw_hint = QLabel(self.tr("\u270e drag to exclude  ·  ctrl/space/middle-drag pans  ·  wheel zooms"))
        self.lbl_draw_hint.setStyleSheet(
            "font-size:10px; color: palette(placeholderText);")
        # Static text, but long: let it be squeezed instead of setting the
        # toolbar's minimum width.
        self.lbl_draw_hint.setSizePolicy(QSizePolicy.Policy.Ignored,
                                         QSizePolicy.Policy.Preferred)
        self.lbl_draw_hint.setToolTip(
            "Drag on the image to enclose a region that must be left alone.\n"
            "No mode to turn on \u2014 drawing is always live, as in ADBE.\n\n"
            "Exclusion zones do two things:\n"
            "  \u2022 no background sample point is placed inside one, so a dark\n"
            "    nebula or a big galaxy cannot be mistaken for sky;\n"
            "  \u2022 the dust-mote refinement composites those pixels back from\n"
            "    the untouched source, so nothing inside is altered.")

        # Display stretch. The old behaviour (hard autostretch to a 0.4
        # median) is punchy but unkind to faint data -- it exaggerates noise
        # and can make a perfectly good correction look blotchy. The normal
        # statistical stretch is gentler and is now the default; the hard
        # one stays for when you want to see every last thing.
        self.cmb_stretch = QComboBox()
        self.cmb_stretch.addItem(self.tr("Normal stretch"), "normal")
        self.cmb_stretch.addItem(self.tr("Aggressive stretch"), "aggressive")
        self.cmb_stretch.setToolTip(
            "How the PREVIEW is displayed. Affects nothing that is applied.\n\n"
            "Normal: statistical stretch, unlinked, target median 0.25, no\n"
            "black clip. What the rest of SASpro shows you.\n\n"
            "Aggressive: hard autostretch to a 0.4 median. Shows the faintest\n"
            "structure, at the cost of making noise look like signal.")
        self.cmb_stretch.currentIndexChanged.connect(self._stretch_changed)

        # Background views have their own problem, opposite to the one
        # above: min-max normalisation gives a field that varies by 0.1%
        # the whole tonal range, so a background that is genuinely almost
        # flat looks violent. Min-max stays the default because it is the
        # only way to SEE the SHAPE; the linear mode shows the true
        # AMPLITUDE, which is the other half of the judgement.
        # It belongs ON the background tabs, not in tab 0's toolbar:
        # that toolbar is a child of the Target image page, so the
        # control was only ever on screen in the one view where it has
        # nothing to act on -- and the enable rule (background views
        # only) therefore held it permanently disabled. One combo per
        # background tab, built by a factory, with the MODE as the
        # single source of truth and the combos kept in step.
        self._bg_stretch_mode = "minmax"
        self._bg_combos = []

        def _mk_bgstretch():
            c = QComboBox()
            c.addItem(self.tr("Scale: min/max"), "minmax")
            c.addItem(self.tr("Scale: linear"), "linear")
            c.setToolTip(
                "How the SURFACE views are displayed — Gradient removed,\n"
                "Target background, Survey background. One setting for\n"
                "all three. Affects nothing that is applied, and nothing\n"
                "in the other tabs.\n\n"
                "min/max: stretch the background's own range to fill 0-1,\n"
                "then tone-map it. Shows the SHAPE of the gradient at any\n"
                "amplitude, and makes a nearly-flat background look dramatic.\n\n"
                "linear: 0.5 + (bg - median(bg)) / 2, no tone map. The median\n"
                "sits at mid-grey and the gradient keeps its real size, so a\n"
                "0.002 background is a barely-visible grey wash — which is\n"
                "what it actually is. Full black to full white here would mean\n"
                "the background spans 1.0, i.e. something has gone wrong.")
            _i = c.findData(self._bg_stretch_mode)
            if _i >= 0:
                c.blockSignals(True)
                c.setCurrentIndex(_i)
                c.blockSignals(False)
            c.currentIndexChanged.connect(
                lambda *_a, _c=c: self._bgstretch_changed(_c))
            self._bg_combos.append(c)
            return c

        # ADBE can write its background out as its own document, and
        # the same question has more answers here: the gradient, both
        # modelled backgrounds, the reference, the point map. They are
        # all computed already, so the button is cheap -- the work is
        # saying honestly what each one IS, which _push_view_document
        # does per view.
        #
        # ONE button, pushing whichever view is on screen, and it lives
        # in the action column with the other verbs. Two other homes
        # were built and measured first:
        #
        #   per-page buttons -- a 200 px button in tab 0's toolbar, and
        #     that toolbar is what sets this dialog's minimum width:
        #     1704 -> 1910 px, which stops it fitting a 1920 screen.
        #   a tab-bar corner widget -- free in minimum width, but it
        #     spends TAB-BAR width, which is already tight: the six
        #     tabs want 1236 px and have 1212 at a 1700-wide dialog,
        #     so the corner put Sample points behind a scroll arrow.
        #
        # The column has the room, and its label says what it acts on
        # since it is no longer sitting on the thing it acts on.
        self.btn_push = QPushButton(
            self.tr("Current view → new document"))
        self.btn_push.clicked.connect(self._push_current_view)

        self.chk_before = QCheckBox(self.tr("Before"))
        self.chk_before.setToolTip(
            "Show the image as it was when this correction was modelled,\n"
            "so you can flick between before and after.\n\n"
            "This is the snapshot taken at model time, not the document as\n"
            "it stands now \u2014 so it stays a true 'before' even after Apply.")
        self.chk_before.toggled.connect(self._redraw)

        _bar = QHBoxLayout()
        _bar.setContentsMargins(2, 2, 2, 2)
        _bar.addWidget(_tb("\u2212", "Zoom out", self.pane_target.zoom_out))
        _bar.addWidget(_tb("+", "Zoom in", self.pane_target.zoom_in))
        _bar.addWidget(_tb("1:1", "Zoom to the original image scale",
                           self.pane_target.zoom_1_1))
        _bar.addWidget(_tb("Fit", "Fit the whole frame",
                           self.pane_target.zoom_fit))
        _bar.addSpacing(12)
        _bar.addWidget(self.cmb_stretch)
        _bar.addSpacing(12)
        _bar.addWidget(self.chk_before)
        _bar.addSpacing(12)
        _bar.addWidget(_tb("Undo zone", "Remove the last exclusion zone",
                           self.pane_target.undo_polygon))
        _bar.addWidget(_tb("Clear zones", "Remove every exclusion zone",
                           self.pane_target.clear_polygons))
        _bar.addSpacing(12)
        _bar.addWidget(self.lbl_draw_hint)
        _bar.addStretch(1)
        self.lbl_zones = QLabel("")
        self.lbl_zones.setStyleSheet("font-size:10px; color: palette(placeholderText);")
        # This label's TEXT changes as zones are added, and a QLabel reports
        # its text width as its sizeHint. Left free, going from "" to
        # "1 exclusion zone" to "2 exclusion zones" widened the toolbar, the
        # tab, and the whole dialog -- the window visibly grew every time a
        # polygon was released. Pinned to the width of the longest string it
        # will ever hold, so the layout never moves.
        self.lbl_zones.setFixedWidth(
            self.lbl_zones.fontMetrics().horizontalAdvance("88 exclusion zones") + 8)
        self.lbl_zones.setAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
        _bar.addWidget(self.lbl_zones)

        _tgt = QVBoxLayout()
        _tgt.setContentsMargins(0, 0, 0, 0)
        _tgt.addLayout(_bar)
        _tgt.addWidget(self.pane_target, 1)
        _tgtw = QWidget(); _tgtw.setLayout(_tgt)

        def _scale_page(pane):
            """A pane plus its own min/max-or-linear scale control.

            Used by every view that draws a SURFACE rather than a
            picture: the two backgrounds and the gradient. All three
            face the same problem -- min/max makes a nearly-flat
            surface look violent -- and all three share one setting.
            """
            _bb = QHBoxLayout()
            _bb.setContentsMargins(2, 2, 2, 2)
            _bb.addWidget(_mk_bgstretch())
            _bb.addStretch(1)
            _bl = QVBoxLayout()
            _bl.setContentsMargins(0, 0, 0, 0)
            _bl.addLayout(_bb)
            _bl.addWidget(pane, 1)
            _bw = QWidget()
            _bw.setLayout(_bl)
            return _bw

        self.tabs.addTab(_tgtw, self.tr("Target image"))
        self.tabs.addTab(self.pane_ref, self.tr("Survey reference"))
        self.tabs.addTab(_scale_page(self.pane_grad),
                         self.tr("Gradient removed"))
        self.tabs.addTab(_scale_page(self.pane_bgt),
                         self.tr("Target background"))
        self.tabs.addTab(_scale_page(self.pane_bgr),
                         self.tr("Survey background"))
        # _active_pane() maps a tab index onto the PANE, not the page,
        # so wrapping the panes changes nothing downstream.
        # Kept as an alias: the mode lives in _bg_stretch_mode now, but
        # this name is the one the rest of the file and the tests reach
        # for, and driving it still propagates through the handler.
        self.cmb_bgstretch = self._bg_combos[0]
        self.tabs.addTab(self.pane_pts, self.tr("Sample points"))
        self.tabs.setTabToolTip(0, "The result, and where you draw exclusions.")
        self.tabs.setTabToolTip(1, "The fetched reference. Check it is the "
                                   "right sky and the right way up.")
        self.tabs.setTabToolTip(2, "What is being subtracted. If you can see "
                                   "your target in here, the model is eating "
                                   "signal.")
        self.tabs.setTabToolTip(3, "Your modelled background.")
        self.tabs.setTabToolTip(4, "The survey's modelled background.")
        self.tabs.setTabToolTip(5, "Green = points used, red tint = vetoed.")
        self.tabs.currentChanged.connect(self._redraw)

        # The old dropdown is gone; _redraw still speaks in these names, so
        # the tab index maps back onto them.
        self._TAB_MODES = ["Corrected", "Reference", "Gradient removed",
                           "Target background", "Reference background",
                           "Sample points"]

        # The tooltip is written per view, so it can only be built once
        # _TAB_MODES exists.
        self.tabs.currentChanged.connect(self._sync_push_button)
        self._sync_push_button()

        self.btn_fetch = QPushButton(self.tr("1. Fetch Reference"))
        self.btn_fetch.setToolTip(
            "Pull the reference for this field and show it, WITHOUT doing\n"
            "the correction.\n\n"
            "Worth doing first on an unfamiliar field: survey coverage is\n"
            "uneven, and a reference that is empty, plate-damaged or simply\n"
            "has no structure where your data does is better found out now\n"
            "than after a full model run. The fetch is cached, so carrying\n"
            "on costs no second download.")
        self.btn_run = QPushButton(self.tr("2. Model && Correct"))
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
        self.btn_fetch.clicked.connect(self._do_fetch_reference)
        self.btn_run.clicked.connect(self._do_run)
        self.btn_apply.clicked.connect(self._do_apply)
        self.btn_diag.clicked.connect(self._do_diagnostics)
        self.btn_close.clicked.connect(self.close)

        self.status = QLabel("Ready")
        self.status.setWordWrap(True)


        rowb = QHBoxLayout()
        rowb.addWidget(self.btn_fetch)
        rowb.addWidget(self.btn_run)
        rowb.addWidget(self.btn_apply)
        rowb.addStretch(1)
        rowb.addWidget(self.btn_close)

        # PI-style grip: drag onto the canvas for a shortcut carrying these
        # exact settings, or straight onto an image to run them headlessly.
        # Deferred import -- shortcuts.py imports a good deal of the GUI.
        try:
            from setiastro.saspro.shortcuts import PresetDragHandle
            from setiastro.saspro.resources import srgc_path
            self.preset_drag_handle = PresetDragHandle(
                "srgc", self._srgc_params, icon=QIcon(srgc_path),
                tooltip=self.tr(
                    "Drag to the canvas to create an SRGC shortcut\n"
                    "with these exact settings.\n"
                    "Drop directly on an image to apply them headlessly."),
                parent=self)
        except Exception as _e:
            _log(f"preset drag handle unavailable: {_brief(_e)}")
            self.preset_drag_handle = None

        rowd = QHBoxLayout()
        rowd.addWidget(self.btn_diag)
        rowd.addStretch(1)
        rowd.addWidget(self.btn_push)

        left = QVBoxLayout()
        left.addWidget(gb)
        left.addWidget(self.lbl_note)
        left.addLayout(rowb)
        left.addLayout(rowd)
        left.addWidget(self.status)
        left.addStretch(1)

        # Bottom row, AFTER the stretch so it pins to the lower corners
        # rather than floating wherever the controls above happen to end:
        # grip at the far left, help at the far right. Same arrangement
        # Statistical Stretch uses.
        self.btn_help = None
        try:
            from setiastro.saspro.help_support import make_help_button
            self.btn_help = make_help_button("srgc", self)
        except Exception as _e:
            _log(f"help button unavailable: {_brief(_e)}")

        if self.preset_drag_handle is not None or self.btn_help is not None:
            grip_row = QHBoxLayout()
            grip_row.setContentsMargins(0, 0, 0, 0)
            if self.preset_drag_handle is not None:
                grip_row.addWidget(self.preset_drag_handle)
            grip_row.addStretch(1)
            if self.btn_help is not None:
                grip_row.addWidget(self.btn_help)
            left.addLayout(grip_row)

        # The status label wraps, and a wrapping QLabel still reports the
        # full unwrapped text as its hint unless the column is bounded.
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Preferred)
        self.lbl_note.setSizePolicy(QSizePolicy.Policy.Ignored,
                                    QSizePolicy.Policy.Preferred)

        main = QHBoxLayout(self)
        _lwrap = QWidget()
        _lwrap.setMaximumWidth(460)
        lw = QVBoxLayout(_lwrap)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.addLayout(left)
        main.addWidget(_lwrap, 0)
        main.addWidget(self.tabs, 1)

        self.resize(1000, 620)
        self._load_settings()
        self._survey_changed(self.cmb_survey.currentText())
        self._refresh_reference_docs()
        self._source_changed()
        self._show_source_now()
        self._start_availability_probe()

    # ---- availability ---------------------------------------------------
    def _start_availability_probe(self):
        """
        Ask, once per dialog open and off the UI thread, whether the surveys
        currently tagged unavailable have come back.

        Backgrounded because the answer is worth a few seconds but the dialog
        is not worth blocking for them: with CDS down the pre-flight settles
        every dead host in ~2.5 s, and with CDS up the /properties checks are
        a few hundred ms each.
        """
        try:
            self._avail_probe = _AvailabilityProbe()
            _LIVE_PROBES.add(self._avail_probe)
            self._avail_probe.finished.connect(
                lambda p=self._avail_probe: _LIVE_PROBES.discard(p))
            self._avail_probe.finishedWith.connect(self._on_availability)
            self._avail_conn = True
            self._avail_probe.start()
        except Exception as e:
            _log(f"could not start the availability probe: {_brief(e)}")
            self._avail_probe = None
            self._avail_conn = None

    def _on_availability(self, changes: dict):
        """Adopt whatever the probe found and tell the user if it matters."""
        if not changes:
            return
        for name, tag in changes.items():
            info = SURVEYS.get(name)
            if info is not None:
                info["availability"] = tag

        try:
            cur = self.cmb_survey.currentText()
            if cur in changes:
                self._survey_changed(cur)
            back = sorted(n for n, t in changes.items() if t == "verified")
            moved = sorted(n for n, t in changes.items() if t == "moved")
            if back:
                # Built outside the f-string: nested same-quotes inside
                # one is PEP 701, i.e. Python 3.12+, and this file has to
                # parse on the older interpreter the Intel-mac build uses.
                _names = "; ".join(back)
                self._set_status(
                    f"Reachable again: {_names}. "
                    f"Re-run probe_mirrors.py to see what else has returned.")
            elif moved:
                self._set_status(
                    f"{len(moved)} survey(s): the mirror host is answering "
                    f"again but its tile trees have moved. Re-run "
                    f"probe_mirrors.py to find the new paths.")
        except Exception:
            pass

    def closeEvent(self, ev):
        """
        Detach the background probe before this dialog is destroyed.

        WA_DeleteOnClose is set above, so a signal arriving from the probe
        after close would be delivered to a deleted C++ object. The thread
        itself is NOT waited on -- _LIVE_PROBES keeps it alive until it
        finishes on its own, so closing the dialog never blocks on a socket.
        """
        p = getattr(self, "_avail_probe", None)
        if p is not None and getattr(self, "_avail_conn", None):
            try:
                p.finishedWith.disconnect(self._on_availability)
            except Exception:
                pass
        self._avail_conn = None
        self._avail_probe = None
        try:
            super().closeEvent(ev)
        except Exception:
            ev.accept()

    # ---- settings -------------------------------------------------------
    def _load_settings(self):
        # Guard the save handlers: setValue below emits, and without this
        # the act of loading would write every default straight back out.
        self._loading_settings = True
        try:
            self._load_settings_inner()
        finally:
            self._loading_settings = False

    def _load_settings_inner(self):
        s = QSettings()
        name = str(s.value("srgc/survey", DEFAULT_SURVEY))
        if self.cmb_survey.findText(name) >= 0:
            self.cmb_survey.setCurrentText(name)
        _src = str(s.value("srgc/source", "survey"))
        _i = self.cmb_source.findData(_src)
        if _i >= 0:
            self.cmb_source.blockSignals(True)
            self.cmb_source.setCurrentIndex(_i)
            self.cmb_source.blockSignals(False)
        self.sp_work.setValue(int(s.value("srgc/work", 1024)))
        self.sp_degree.setValue(int(s.value("srgc/degree", 1)))
        self.sp_samples.setValue(int(s.value("srgc/samples", 200)))
        self.sp_patch.setValue(int(s.value("srgc/patch", 15)))
        self.sp_exclude.setValue(int(s.value("srgc/exclude", 50)))
        self.sp_protect.setValue(float(s.value("srgc/protect", 3.0)))
        self.chk_rbf.setChecked(bool(s.value("srgc/rbf", True, type=bool)))
        self.sp_strength.setValue(int(s.value("srgc/strength", 100)))
        self.chk_ms.setChecked(bool(s.value("srgc/ms", False, type=bool)))
        self.sp_ms_lo.setValue(int(s.value("srgc/ms_lo", 6)))
        self.sp_ms_hi.setValue(int(s.value("srgc/ms_hi", 8)))
        self.sp_ms_strength.setValue(float(s.value("srgc/ms_strength", 1.0)))
        self.sp_ms_protect.setValue(float(s.value("srgc/ms_protect", 3.0)))
        self.chk_ms_stars.setChecked(
            bool(s.value("srgc/ms_stars", True, type=bool)))
        self.chk_ms_survey_protect.setChecked(
            bool(s.value("srgc/ms_survey_protect", True, type=bool)))
        self.chk_ms_residual.setChecked(
            bool(s.value("srgc/ms_residual", False, type=bool)))
        self._ms_toggled(self.chk_ms.isChecked())
        _sm = str(s.value("srgc/stretch_mode", "normal"))
        _si = self.cmb_stretch.findData(_sm)
        if _si >= 0:
            self.cmb_stretch.blockSignals(True)
            self.cmb_stretch.setCurrentIndex(_si)
            self.cmb_stretch.blockSignals(False)
        self._set_bg_stretch_mode(
            str(s.value("srgc/bg_stretch_mode", "minmax")))

        # Save on CHANGE, not only when something is run. Settings that
        # only persist via _do_run silently lose anything a user tweaks and
        # then closes without running -- which is most of a tuning session.
        for _w in (self.sp_work, self.sp_degree, self.sp_samples,
                   self.sp_patch, self.sp_exclude, self.sp_protect,
                   self.sp_strength, self.sp_ms_lo, self.sp_ms_hi,
                   self.sp_ms_strength, self.sp_ms_protect):
            _w.valueChanged.connect(self._save_settings)
        for _c in (self.chk_rbf, self.chk_ms, self.chk_ms_stars,
                   self.chk_ms_survey_protect, self.chk_ms_residual):
            _c.toggled.connect(self._save_settings)

    def _save_settings(self, *_args):
        # *_args: wired to valueChanged/toggled, which pass a value.
        if getattr(self, "_loading_settings", False):
            return
        s = QSettings()
        s.setValue("srgc/survey", self.cmb_survey.currentText())
        s.setValue("srgc/source", self._source())
        s.setValue("srgc/ms", self.chk_ms.isChecked())
        s.setValue("srgc/ms_lo", self.sp_ms_lo.value())
        s.setValue("srgc/ms_hi", self.sp_ms_hi.value())
        s.setValue("srgc/ms_strength", float(self.sp_ms_strength.value()))
        s.setValue("srgc/ms_protect", float(self.sp_ms_protect.value()))
        s.setValue("srgc/ms_stars", self.chk_ms_stars.isChecked())
        s.setValue("srgc/ms_survey_protect",
                   self.chk_ms_survey_protect.isChecked())
        s.setValue("srgc/ms_residual", self.chk_ms_residual.isChecked())
        s.setValue("srgc/stretch_mode",
                   str(self.cmb_stretch.currentData() or "normal"))
        s.setValue("srgc/bg_stretch_mode",
                   str(getattr(self, "_bg_stretch_mode", "minmax")))
        s.setValue("srgc/work", self.sp_work.value())
        s.setValue("srgc/degree", self.sp_degree.value())
        s.setValue("srgc/samples", self.sp_samples.value())
        s.setValue("srgc/patch", self.sp_patch.value())
        s.setValue("srgc/exclude", self.sp_exclude.value())
        s.setValue("srgc/protect", float(self.sp_protect.value()))
        s.setValue("srgc/rbf", self.chk_rbf.isChecked())
        s.setValue("srgc/strength", self.sp_strength.value())

    def _channels(self) -> list:
        """Channel spec for the selected survey (1 entry = mono, 3 = colour)."""
        name = self.cmb_survey.currentData() or self.cmb_survey.currentText()
        info = SURVEYS.get(str(name)) or SURVEYS[DEFAULT_SURVEY]
        return list(info.get("channels") or [])

    # ---- reference source ------------------------------------------------
    def _source(self) -> str:
        return str(self.cmb_source.currentData() or "survey")

    def _source_changed(self, *_):
        """Show only the controls that belong to the selected source."""
        user = (self._source() == "user")
        for wdg in (self.row_survey_label, self.cmb_survey):
            wdg.setVisible(not user)
        for wdg in (self.row_userref_label, self.w_userref, self.lbl_refinfo):
            wdg.setVisible(user)
        if user:
            self.lbl_note.setText(
                "Your own frame of this field, reprojected onto the work "
                "grid. It must COVER the whole target field — a narrower "
                "reference fails the coverage check — and it must not carry "
                "the same gradient, or b will absorb it and leave nothing "
                "to subtract. Check the reference share in diagnostics.")
        else:
            self._survey_changed(self.cmb_survey.currentText())
        self._save_settings()

    def _open_documents(self) -> list:
        """(name, doc) for every open document except the one being corrected."""
        mw = self._main
        dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
        out = []
        if dm is None:
            return out
        try:
            docs = list(dm.all_documents())
        except Exception:
            docs = []
            for attr in ("documents", "_documents", "docs"):
                try:
                    docs = list(getattr(dm, attr))
                    break
                except Exception:
                    continue
        for d in docs:
            if d is self.doc or getattr(d, "image", None) is None:
                continue
            try:
                nm = d.display_name()
            except Exception:
                nm = getattr(d, "name", None) or repr(d)
            out.append((str(nm), d))
        return out

    def _refresh_reference_docs(self):
        """Repopulate the open-document list, keeping any current pick."""
        prev = self.cmb_refdoc.currentData()
        self.cmb_refdoc.blockSignals(True)
        self.cmb_refdoc.clear()
        self.cmb_refdoc.addItem(self.tr("— choose an open image —"), None)
        for nm, d in self._open_documents():
            self.cmb_refdoc.addItem(nm, d)
        if prev is not None:
            i = self.cmb_refdoc.findData(prev)
            if i >= 0:
                self.cmb_refdoc.setCurrentIndex(i)
        self.cmb_refdoc.blockSignals(False)

    def _refdoc_changed(self, *_):
        d = self.cmb_refdoc.currentData()
        if d is None:
            return
        meta = getattr(d, "metadata", None) or {}
        hdr = (meta.get("wcs_header") or meta.get("original_header")
               or meta.get("header"))
        self._set_user_reference(getattr(d, "image", None), hdr,
                                 self.cmb_refdoc.currentText())

    def _pick_reference_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Choose a reference image"), "",
            self.tr("Images (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png "
                    "*.jpg *.jpeg *.cr2 *.cr3 *.nef *.arw *.dng);;"
                    "All files (*)"))
        if not path:
            return
        try:
            from setiastro.saspro.legacy.image_manager import load_image
            img, hdr, _bd, _mono = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Reference image"),
                                f"Could not load that file:\n\n{e}")
            return
        if img is None:
            QMessageBox.warning(self, self.tr("Reference image"),
                                "That file produced no image data.")
            return
        import os
        self.cmb_refdoc.blockSignals(True)
        self.cmb_refdoc.setCurrentIndex(0)
        self.cmb_refdoc.blockSignals(False)
        self._set_user_reference(img, hdr, os.path.basename(path))

    def _set_user_reference(self, img, hdr, label: str):
        """Adopt a reference and say plainly what is and is not known about it."""
        if img is None:
            self._user_ref = None
            self.lbl_refinfo.setText("")
            return
        a = np.asarray(img)
        self._user_ref = (a, hdr, str(label))
        solved = _celestial_wcs_from_header(hdr) is not None
        kind = "colour" if (a.ndim == 3 and a.shape[2] >= 3) else "mono"
        self.lbl_refinfo.setText(
            f"{label} — {a.shape[1]}×{a.shape[0]} {kind}, "
            + ("plate-solved ✓" if solved
               else "NOT plate-solved — SRGC will offer to solve it"))

    def showEvent(self, ev):
        # The set of open documents changes behind this dialog, and it is
        # non-modal, so refresh on every show rather than only at build time.
        try:
            self._refresh_reference_docs()
        except Exception:
            pass
        super().showEvent(ev)

    def _survey_changed(self, name: str):
        info = SURVEYS.get(name)
        if info:
            # Lead with availability when there is something to warn about, so
            # a survey that cannot currently fetch says so before the user
            # spends a minute watching it fail.
            prefix = _AVAILABILITY.get(info.get("availability", "verified"), "")
            self.lbl_note.setText(prefix + info["note"])
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
    def _do_fetch_reference(self):
        """
        Step 1: get the reference and show it. No modelling, no correction.

        Survey coverage is uneven and the reference is the one input the
        user cannot inspect in advance. Fetching it on its own turns "the
        correction did something strange" into "that reference is no good
        for this field", which is a much easier thing to act on. The result
        goes into the same session cache the model run reads, so saying yes
        afterwards costs no second download.
        """
        img = self._source_float()
        if img is None:
            QMessageBox.information(self, "No image",
                                    "No image in the active document.")
            return
        self._save_settings()
        self.btn_fetch.setEnabled(False)
        self.btn_run.setEnabled(False)
        try:
            hdr = self._header()
            hdr, _solved = ensure_solved_header(
                self, img, hdr, label="This image", settings=QSettings())
            if _solved:
                self._adopt_solved_header(hdr)

            H, W = np.asarray(img).shape[:2]
            h, w = _work_shape(H, W, int(self.sp_work.value()))
            work_wcs = _rescaled_wcs(
                _celestial_wcs_from_header(hdr), (H, W), (h, w))

            if self._source() == "user":
                ref = self._prepare_user_reference(img, hdr)
                where = "your own image"
                self._ref_info = {"route": "supplied", "format": None,
                                  "bits": None}
            else:
                key = self._ref_cache_key(img, hdr)
                ref = self._ref_cache.get(key)
                if ref is None:
                    _info: dict = {}
                    ref = fetch_reference_channels(
                        work_wcs, (h, w), list(self._channels()),
                        progress_cb=self._set_status, info=_info)
                    self._ref_cache[key] = ref
                    self._ref_info_cache[key] = _info
                    self._ref_info = _info
                    while len(self._ref_cache) > _REF_CACHE_MAX:
                        _k = next(iter(self._ref_cache))
                        self._ref_cache.pop(_k)
                        self._ref_info_cache.pop(_k, None)
                else:
                    _log("reference served from the session cache")
                    self._ref_info = self._ref_info_cache.get(key, {})
                where = self.cmb_survey.currentText()
        except SurveyFetchError as e:
            self._set_status("Fetch failed")
            QMessageBox.warning(self, "Survey reference", str(e))
            self.btn_fetch.setEnabled(True); self.btn_run.setEnabled(True)
            return
        except Exception as e:
            self._set_status("Error")
            QMessageBox.critical(self, "Survey reference", repr(e))
            self.btn_fetch.setEnabled(True); self.btn_run.setEnabled(True)
            return

        self.btn_fetch.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_apply.setEnabled(False)

        # A preview-only result. Deliberately carries no gradient, so Apply
        # stays disabled and no other view can be opened on half a run.
        self._result = {"reference": np.asarray(ref, dtype=np.float32),
                        "work_shape": (h, w), "preview_only": True}
        self._disp_cache.clear()
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(1)          # Survey reference
        self.tabs.blockSignals(False)
        self._redraw()
        self._set_status(self._describe_reference(ref, where,
                                                  self._ref_info))

    def _ms_toggled(self, on: bool):
        self.w_ms.setVisible(bool(on))

    def _run_multiscale(self, img):
        """
        Divide out sharp multiplicative defects, after the additive fit.

        Reuses ADBE's multiscale core verbatim -- it is tested, it is
        already the right operator, and a second implementation of a
        log-space band division would only be a second thing to get wrong.

        What SRGC adds is the PROTECTION. ADBE builds its mask from the
        image alone, so a bright extended blob looks identical whether it is
        a real nebula or an internal reflection. SRGC already has a veto
        mask derived from the actual sky, so it can say which: real
        structure is composited back from the untouched source, while
        anything sharp sitting where the survey says there is only sky is
        treated as a defect.
        """
        from setiastro.saspro.abe_multiscale_gradient import (
            multiscale_gradient_correct)

        strength = self.sp_strength.value() / 100.0
        base = apply_survey_gradient(img, self._result["gradient"], strength)

        lo = int(self.sp_ms_lo.value())
        hi = int(self.sp_ms_hi.value())
        if hi < lo:
            lo, hi = hi, lo

        # Protection mask is built on the STAR-INCLUSIVE image, the band is
        # estimated on a starless copy. Same pairing ADBE uses, and the
        # reason the band comes out clean.
        mask_from = base
        estimate_from = base
        if self.chk_ms_stars.isChecked():
            self._set_status("Refinement: removing stars (DarkStar)…")
            try:
                from setiastro.saspro.remove_stars import (
                    darkstar_starless_from_array)
                # Returns (starless, stars_or_None, was_mono) -- unpack it,
                # and tolerate a None starless rather than feeding a tuple
                # into the band estimator.
                _starless, _stars, _mono = darkstar_starless_from_array(
                    base, use_gpu=True,
                    status_cb=lambda m: self._set_status(
                        f"Refinement: DarkStar {m}"))
                if _starless is None:
                    raise RuntimeError("DarkStar returned no starless image")
                estimate_from = np.asarray(_starless, dtype=np.float32)
                _log("multiscale: estimating on a DarkStar starless copy")
            except Exception as e:
                _log(f"multiscale: DarkStar unavailable ({_brief(e)}); "
                     f"estimating on the star-inclusive image")
                self._set_status(
                    "Refinement: DarkStar unavailable — estimating with "
                    "stars in. Expect a softer result.")

        # The survey-derived veto, upscaled to full resolution. True there
        # means "real structure", which is exactly what protect_blend_mask
        # wants: keep the ORIGINAL pixels.
        H, W = np.asarray(base).shape[:2]
        protect = None
        if self.chk_ms_survey_protect.isChecked():
            allowed = self._result.get("allowed")
            if allowed is not None:
                a = (~np.asarray(allowed, dtype=bool)).astype(np.float32)
                protect = _resize_to(a, (H, W)) > 0.5
                _log(f"multiscale: survey-derived protection covers "
                     f"{100.0 * float(protect.mean()):.1f}% of the frame")

        # Hand-drawn zones protect UNCONDITIONALLY, whether or not the
        # survey-derived veto is in use: the user drew them precisely
        # because they know something the survey does not.
        drawn = self._exclusion_mask_for(base)
        if drawn is not None:
            protect = drawn if protect is None else (protect | drawn)
            _log(f"multiscale: {len(self.pane_target.polygons)} drawn "
                 f"zone(s) added to the protection")

        # Off by default, and worth leaving off: the additive stage has
        # already removed the smooth background, using a reference that
        # says which large-scale structure is real. On it folds the
        # coarsest residual into the multiplicative correction as well.
        residual = bool(self.chk_ms_residual.isChecked())
        if residual:
            _log("multiscale: coarsest residual INCLUDED — the smooth "
                 "background is being corrected a second time, "
                 "multiplicatively")
        self._set_status("Refinement: dividing out the defect band…")
        out = multiscale_gradient_correct(
            base,
            estimate_from=estimate_from,
            mask_from=mask_from,
            band_lo=lo, band_hi=hi,
            include_residual=residual,
            strength=float(self.sp_ms_strength.value()),
            protect_k=float(self.sp_ms_protect.value()),
            protect_blend_mask=protect,
            estimate_downsample=1,
            progress_cb=lambda m: self._set_status(f"Refinement: {m}"),
        )
        if isinstance(out, tuple):
            out = out[0]
        out = np.asarray(out, dtype=np.float32)

        # Stored finished, with the strength it was computed at: the band is
        # estimated FROM the corrected image, so it is not valid for a
        # different additive strength. _redraw says so rather than showing a
        # stale frame as if it were current.
        self._result["corrected_full"] = out
        self._result["ms_strength"] = int(self.sp_strength.value())
        _ph, _pw = _work_shape(out.shape[0], out.shape[1], _PREVIEW_LONG_SIDE)
        self._result["preview_corrected"] = (
            out if (_ph, _pw) == out.shape[:2] else _resize_to(out, (_ph, _pw)))
        _log(f"multiscale refinement applied (scales {lo}-{hi}, "
             f"residual={residual}, strength="
             f"{float(self.sp_ms_strength.value()):.2f})")

    @staticmethod
    def _describe_reference(ref: np.ndarray, where: str,
                            info: dict | None = None) -> str:
        """
        A one-line verdict on a fetched reference, before any modelling.

        Coverage and CONTRAST are the two things that decide whether this
        reference can do anything. A plane that is covered but flat carries
        no structure to regress against, and the run would degenerate into
        ADBE with a survey-derived mask -- which is worth knowing before
        spending a model run rather than after.
        """
        a = np.asarray(ref, dtype=np.float32)
        fin = np.isfinite(a)
        cov = float(fin.mean()) * 100.0 if a.size else 0.0
        if not fin.any():
            return (f"Reference from {where}: NO usable pixels. This field "
                    f"is outside the survey, or the request was rejected — "
                    f"Run diagnostics to tell which.")
        v = a[fin]
        lo, hi = np.percentile(v, [0.5, 99.5])
        span = float(hi - lo)
        med = float(np.median(v))
        rel = (span / abs(med)) if abs(med) > 1e-12 else 0.0
        # Coverage first: it is the one that decides whether the run is
        # worth starting, and it only became trustworthy for 8-bit trees
        # once exact-black was read as no-data (see _decode_image_tile).
        if cov < 99.0:
            gap = (f"  ⚠ {100.0 - cov:.0f}% of this reference has NO DATA"
                   f" — the survey does not cover all of this field")
        else:
            gap = ""
        verdict = ("flat — little structure to fit against; expect this run "
                   "to behave like ADBE with a survey-derived mask"
                   if rel < 0.05 else
                   "has structure to fit against")
        if rel > 5.0:
            # Sky structure spans a factor of a few. Ten times its own
            # level usually means something instrumental is in frame --
            # mosaic seams, a plate edge, a scan stripe. Said, not acted
            # on: the picture in this tab is the real check.
            verdict += ("; that is a very wide range for sky — check this "
                        "tab for mosaic seams or plate edges before "
                        "trusting it")
        planes = 1 if a.ndim == 2 else a.shape[2]
        mono = ""
        if a.ndim == 3 and planes >= 2:
            p0 = np.nan_to_num(a[..., 0])
            if all(np.allclose(p0, np.nan_to_num(a[..., c]), rtol=1e-5)
                   for c in range(1, planes)):
                mono = "  ·  MONO (all channels are the same plane)"
        prov = describe_reference_provenance(info)
        if prov:
            prov = "  ·  " + prov
        return (f"Reference from {where}  ·  coverage {cov:.1f}%  ·  "
                f"contrast {rel * 100:.1f}% of its own level — {verdict}"
                f"{prov}{mono}{gap}")

    def _do_run(self):
        img = self._source_float()
        if img is None:
            QMessageBox.information(self, "No image", "No image in the active document.")
            return

        hdr = self._header()

        self._save_settings()
        self.btn_run.setEnabled(False)
        self.btn_apply.setEnabled(False)
        user_ref = None
        cached_ref = None
        cache_key = None
        try:
            # The target must be solved whichever source is used: a survey is
            # fetched by sky coordinates, and a user frame is reprojected onto
            # the target WCS. Offer the solver instead of just refusing.
            hdr, _solved = ensure_solved_header(
                self, img, hdr, label="This image", settings=QSettings())
            if _solved:
                self._adopt_solved_header(hdr)

            if self._source() == "user":
                user_ref = self._prepare_user_reference(img, hdr)
                self._ref_info = {"route": "supplied", "format": None,
                                  "bits": None}
            else:
                cache_key = self._ref_cache_key(img, hdr)
                cached_ref = self._ref_cache.get(cache_key)
                if cached_ref is not None:
                    _log("reference served from the session cache "
                         "(no fetch)")
                    self._ref_info = self._ref_info_cache.get(
                        cache_key, self._ref_info)
                    self._set_status(
                        "Reusing the reference already fetched for these "
                        "settings — no download.")

            self._result = survey_gradient_model(
                img, hdr,
                channels=(None if user_ref is not None else self._channels()),
                reference=(user_ref if user_ref is not None else cached_ref),
                user_exclusion=self._exclusion_mask_for(img),
                work_long_side=int(self.sp_work.value()),
                degree=int(self.sp_degree.value()),
                use_rbf=bool(self.chk_rbf.isChecked()),
                num_samples=int(self.sp_samples.value()),
                patch_size=int(self.sp_patch.value()),
                exclusion_fraction=self.sp_exclude.value() / 100.0,
                protect_k=float(self.sp_protect.value()),
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

        # Snapshot of exactly what was corrected. Every preview below works
        # off this, NOT off self.doc.image: once Apply commits, the document
        # IS the corrected image, and re-deriving "Corrected" from it would
        # subtract the same gradient a second time. It is also what makes
        # the Before toggle honest after an Apply.
        self._result["source"] = np.asarray(img, dtype=np.float32)

        # Preview-scale copies of the two things the Corrected/Before views
        # need, built once per run instead of per redraw.
        _src = self._result["source"]
        _H, _W = _src.shape[:2]
        _ph, _pw = _work_shape(_H, _W, _PREVIEW_LONG_SIDE)
        if (_ph, _pw) != (_H, _W):
            self._result["preview_source"] = _resize_to(_src, (_ph, _pw))
            self._result["preview_gradient"] = _resize_to(
                np.asarray(self._result["gradient"], np.float32), (_ph, _pw))
        else:
            self._result["preview_source"] = _src
            self._result["preview_gradient"] = np.asarray(
                self._result["gradient"], np.float32)
        self._disp_cache.clear()

        # Keep the fetched reference so tweaking degree / protect k / samples
        # and re-running does not pull it down the wire again. Keyed on the
        # things that actually change WHAT was fetched — survey, work grid,
        # image geometry and pointing — so any of those invalidates it.
        try:
            self._ref_cache[cache_key] = self._result["reference"]
            _ri = (self._result.get("diag") or {}).get("ref_info")
            # Only adopt provenance from a run that actually FETCHED. On a
            # cache hit the array is handed to the model, which honestly
            # reports "supplied" -- and letting that overwrite the real
            # record would relabel a fetched FITS tree as "your own
            # reference" on the second run.
            if _ri and _ri.get("format"):
                self._ref_info_cache[cache_key] = _ri
                self._ref_info = _ri
            while len(self._ref_cache) > _REF_CACHE_MAX:
                _k = next(iter(self._ref_cache))
                self._ref_cache.pop(_k)
                self._ref_info_cache.pop(_k, None)
        except Exception:
            pass

        # Second, MULTIPLICATIVE stage. Runs on the already-corrected image
        # because it is a different operator on a different defect class:
        # the additive fit removes what the sky does not explain, this
        # divides out what the optics attenuated.
        if self.chk_ms.isChecked():
            try:
                self._run_multiscale(img)
            except Exception as e:
                _log(f"multiscale refinement failed: {_brief(e)}")
                QMessageBox.warning(
                    self, "Dust mote refinement",
                    f"The gradient correction succeeded, but the "
                    f"multiplicative refinement failed:\n\n{e}\n\n"
                    f"The additive result is still available.")

        self.chk_before.setEnabled(True)
        cov = self._result["coverage"] * 100.0
        cs = ", ".join(f"b={b:.3g}" for _, b in self._result["coeffs"])
        g = self._result["gradient"]
        # reference_share is the number that says whether this was a
        # REFERENCE-based correction at all, so it belongs on screen rather
        # than buried in the log. It is a partition: the survey's share and
        # your own gradient's share sum to 100%. Near zero means the
        # reference explained nothing and the run degenerated into ADBE
        # with a survey mask.
        _d = self._result.get("diag") or {}
        _share = _d.get("reference_share", 0.0)
        _warn = ("  ·  ⚠ the reference explains almost none of this — "
                 "treat it as ADBE with a survey-derived mask"
                 if _share < 0.05 else "")
        # A reference can explain a healthy share and still be fitted to a
        # staircase, so the resolution verdict is its own warning rather
        # than something reference_share can stand in for.
        if _d.get("ref_resolution") == "pinned":
            _warn += (f"  ·  ⚠ the reference background is pinned to "
                      f"{max(_d.get('ref_levels') or 1.0, 1.0):.0f} "
                      f"quantisation level(s) — prefer a survey with "
                      f"FITS tiles")
        self._set_status(
            f"Coverage {cov:.1f}%  ·  work {self._result['work_shape'][1]}×"
            f"{self._result['work_shape'][0]}  ·  {cs}  ·  "
            f"structure: survey {_share * 100:.0f}% / yours "
            f"{_d.get('residual_share', 0.0) * 100:.0f}%  ·  "
            f"gradient peak-to-peak {float(np.nanmax(g) - np.nanmin(g)):.4f}"
            f"{_warn}"
        )

        # Step 1 flips to the reference so you can see what was fetched;
        # step 2 flips back, because the thing you asked for is the
        # corrected image and that is where the run should land you.
        # Without this the view stayed on whatever step 1 left behind,
        # and _redraw below would then overwrite the run summary with
        # that tab's own note -- so the result line above was never
        # even read on the usual 1-then-2 path.
        # Signals blocked for the same reason as the step 1 flip: the
        # currentChanged -> _redraw hook would render the new tab from
        # a result that is complete but not yet drawn, and the explicit
        # _redraw() on the next line does that job once.
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(0)          # Target image
        self.tabs.blockSignals(False)
        self._redraw()

    def _ref_cache_key(self, img, hdr):
        """
        What uniquely determines the fetched reference.

        Survey, work grid, image shape and the pointing/scale of the WCS.
        Anything the user can change that does NOT appear here — degree,
        sample count, patch size, exclusion, protect k, RBF, strength — is
        applied to the SAME reference, which is the whole point: those are
        the knobs people re-run, and re-downloading for them is pure waste.
        """
        shape = tuple(np.asarray(img).shape[:2])
        wcs_fp = None
        try:
            w = _celestial_wcs_from_header(hdr)
            if w is not None:
                wcs_fp = (
                    tuple(np.round(np.asarray(w.wcs.crval, float), 9)),
                    tuple(np.round(np.asarray(w.wcs.crpix, float), 6)),
                    tuple(str(c) for c in w.wcs.ctype),
                    tuple(np.round(
                        np.asarray(w.wcs.cd if w.wcs.has_cd()
                                   else w.wcs.cdelt, float).ravel(), 12)),
                )
        except Exception:
            wcs_fp = None
        return (self.cmb_survey.currentText(),
                int(self.sp_work.value()), shape, wcs_fp)

    def _adopt_solved_header(self, hdr):
        """Keep a freshly-solved header on the document, not just in memory."""
        try:
            meta = getattr(self.doc, "metadata", None)
            if isinstance(meta, dict):
                meta["wcs_header"] = hdr
                meta.setdefault("original_header", hdr)
        except Exception:
            pass

    def _prepare_user_reference(self, img, target_hdr):
        """
        Turn the chosen frame into a reference plane on the work grid.

        Solved first (offering the solver if it is not), then reprojected.
        Returns the array survey_gradient_model takes as `reference=`.
        """
        if not self._user_ref:
            raise SurveyFetchError(
                "No reference image chosen. Pick one of your open images or "
                "load a file.")
        ref_img, ref_hdr, label = self._user_ref

        ref_hdr, solved = ensure_solved_header(
            self, ref_img, ref_hdr, label=f"The reference image ({label})",
            settings=QSettings())
        if solved:
            self._user_ref = (ref_img, ref_hdr, label)
            self._set_user_reference(ref_img, ref_hdr, label)

        H, W = np.asarray(img).shape[:2]
        h, w = _work_shape(H, W, int(self.sp_work.value()))
        work_wcs = _rescaled_wcs(
            _celestial_wcs_from_header(target_hdr), (H, W), (h, w))

        a = np.asarray(ref_img, dtype=np.float32)
        if np.issubdtype(np.asarray(ref_img).dtype, np.integer):
            a = a / float(np.iinfo(np.asarray(ref_img).dtype).max)
        return reference_from_image(a, ref_hdr, work_wcs, (h, w),
                                    progress_cb=self._set_status)

    # ---- preset ---------------------------------------------------------
    def _srgc_params(self) -> dict:
        """
        The current UI as a preset, for the drag grip and headless replay.

        A user-supplied reference is deliberately NOT serialisable: it is a
        specific array in this session, not a name a later run could resolve.
        A preset dragged while "my own image" is selected therefore records
        the survey settings, and says so, rather than silently producing a
        shortcut that would quietly do something different.
        """
        return {
            "survey": self.cmb_survey.currentText(),
            "work_long_side": int(self.sp_work.value()),
            "degree": int(self.sp_degree.value()),
            "rbf": bool(self.chk_rbf.isChecked()),
            "samples": int(self.sp_samples.value()),
            "patch": int(self.sp_patch.value()),
            "exclude": int(self.sp_exclude.value()),
            "protect_k": float(self.sp_protect.value()),
            "strength": int(self.sp_strength.value()),
        }

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
            chans = self._channels()
            ph, pw = 64, 64
            probe = _rescaled_wcs(wcs, (H, W), (ph, pw))
            try:
                probe.pixel_shape = (pw, ph)
            except Exception:
                pass

            any_ok = False
            for ci, ch in enumerate(chans):
                lines.append("")
                if not isinstance(ch, dict):
                    lines.append(f"channel {ci}: SYNTHESISED (mean of the "
                                 f"fetched planes) — nothing to probe")
                    continue
                hips_id = ch["hips"]
                lines.append(f"channel {ci}: {hips_id}   ({pw}x{ph} cutout)")
                lines.append("-" * 64)

                for ep in _ordered_endpoints():
                    host = ep.split("//", 1)[-1].split("/", 1)[0]
                    t0 = time.monotonic()
                    try:
                        a = _fetch_via_hips2fits(probe, ph, pw, hips_id, ep,
                                                 HIPS_TIMEOUT)
                        dt = (time.monotonic() - t0) * 1000
                        lines.append(f"  hips2fits {host}")
                        lines.append(f"     OK {dt:.0f} ms  {_array_stats(a)}")
                        any_ok = True
                    except Exception as e:
                        dt = (time.monotonic() - t0) * 1000
                        lines.append(f"  hips2fits {host}")
                        lines.append(f"     FAIL {dt:.0f} ms  {_brief(e, 300)}")
                    QApplication.processEvents()

                for base in (ch.get("mirrors") or []):
                    host = base.split("//", 1)[-1].split("/", 1)[0]
                    t0 = time.monotonic()
                    try:
                        a = _fetch_via_tiles(probe, ph, pw, base)
                        dt = (time.monotonic() - t0) * 1000
                        lines.append(f"  tiles {host}")
                        lines.append(f"     OK {dt:.0f} ms  {_array_stats(a)}")
                        any_ok = True
                    except Exception as e:
                        dt = (time.monotonic() - t0) * 1000
                        lines.append(f"  tiles {host}")
                        lines.append(f"     FAIL {dt:.0f} ms  {_brief(e, 300)}")
                    QApplication.processEvents()

            lines.append("")
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
            # Nothing modelled yet: tab 1 still shows the source so zones
            # can be drawn, and the other tabs have nothing to say.
            if int(self.tabs.currentIndex()) == 0 \
                    and not self.pane_target.has_image():
                self._show_source_now()
            return
        idx = int(self.tabs.currentIndex())
        mode = self._TAB_MODES[idx] if 0 <= idx < len(self._TAB_MODES) \
            else "Corrected"
        if self._result.get("preview_only") and mode != "Reference":
            # Step 1 fetched a reference but nothing has been modelled, so
            # there is no gradient, no background and no sample points to
            # show. Say so instead of raising a KeyError.
            self._set_status(
                "Only the reference has been fetched so far — "
                "run '2. Model && Correct' to see this view.")
            return
        overlay_pts = False
        # Set by the linear background path: that mapping is already a
        # display mapping, and tone-mapping it would undo the one thing it
        # exists to do -- keep the amplitude true.
        raw_display = False
        # The snapshot taken when the model was built, never the live
        # document. After Apply the document already HAS the correction in
        # it, and correcting it again here would show a doubly-corrected
        # frame that no file will ever contain.
        img = self._result.get("source")
        if img is None:
            img = self._source_float()
        if img is None:
            return

        self.chk_before.setEnabled(mode == "Corrected")

        def _sp(a2d, m=None):
            """Robust span, the same measure the model reports."""
            v = np.asarray(a2d, np.float32)
            v = v[m] if (m is not None and
                         np.asarray(m).shape[:2] == v.shape[:2]) else v.ravel()
            v = v[np.isfinite(v)]
            return 0.0 if v.size < 8 else float(
                np.percentile(v, 99.5) - np.percentile(v, 0.5))

        # Rendered buffers are cached per view, so flicking Before/After or
        # returning to a view you have already seen is a dictionary lookup
        # rather than a correct-and-tone-map pass.
        # Everything the render depends on goes in the key -- including both
        # display modes. That is what lets a stretch change KEEP the cache:
        # the bg mode only affects two of the six views, and the corrected
        # view is the expensive one to rebuild.
        _bgmode = str(getattr(self, "_bg_stretch_mode", "minmax"))
        _key = (mode, bool(self.chk_before.isChecked()),
                int(self.sp_strength.value()), int(self.sp_patch.value()),
                str(self.cmb_stretch.currentData() or "normal"), _bgmode)
        _hit = self._disp_cache.get(_key)
        if _hit is not None:
            _buf, _st = _hit
            if _st:
                self._set_status(_st)
            self._show_buffer(_buf)
            return

        # Corrected and Before work at PREVIEW scale. Apply still uses the
        # full-resolution snapshot, so nothing committed depends on this.
        pimg = self._result.get("preview_source")
        if pimg is None:
            pimg = img
        # .get, not [] : a fetch-only run has a reference and no gradient,
        # and the Reference view is legitimately reachable in that state.
        pgrad = self._result.get("preview_gradient")
        if pgrad is None:
            pgrad = self._result.get("gradient")

        if mode == "Corrected" and self.chk_before.isChecked():
            arr = np.asarray(pimg, dtype=np.float32)
            self._set_status(
                "BEFORE — the image as it was when this correction was "
                "modelled. Untick to see the result.")
        elif mode == "Corrected":
            _pc = self._result.get("preview_corrected")
            _stale = (self._result.get("ms_strength") is not None
                      and int(self._result["ms_strength"])
                      != int(self.sp_strength.value()))
            if _pc is not None and not _stale:
                arr = np.asarray(_pc, dtype=np.float32)
            else:
                arr = apply_survey_gradient(pimg, pgrad,
                                            self.sp_strength.value() / 100.0)
                if _stale:
                    self._set_status(
                        "Strength changed — this preview is the additive "
                        "correction only. Re-run '2. Model && Correct' to "
                        "redo the dust-mote refinement at the new strength.")
        elif mode == "Gradient removed":
            g = np.asarray(self._result["gradient_small"], dtype=np.float32)
            _gs = _sp(g)
            if _bgmode == "linear":
                # The gradient is already median-subtracted, so mid-grey
                # here is literally ZERO: grey means nothing was removed
                # at that pixel, brighter means subtracted, darker means
                # added back. Under min/max that reading is lost, because
                # the darkest pixel becomes black whatever its value.
                arr = _linear_mid(g, gain=_BG_LINEAR_GAIN)
                raw_display = True
                self._set_status(
                    f"Gradient span = {_gs:.5g} in your image's units. "
                    f"Display is LINEAR at true scale (zero at mid-grey, "
                    f"gain {_BG_LINEAR_GAIN:g}), so it fills "
                    f"{100.0 * _gs * _BG_LINEAR_GAIN:.1f}% of the tonal "
                    f"range. Grey = nothing removed there; brighter = "
                    f"subtracted, darker = added back. Switch to min/max "
                    f"to see its shape.")
            else:
                arr = _norm01(g)
                self._set_status(
                    f"Gradient span = {_gs:.5g} in your image's units. "
                    f"Display is min-max stretched, so a gradient of any "
                    f"size fills the screen — trust the number, not the "
                    f"picture. If your target is visible in here, the "
                    f"model is eating signal. Switch to linear to see "
                    f"the true amplitude.")
        elif mode == "Reference":
            _ref = np.asarray(self._result["reference"], dtype=np.float32)
            arr = _norm01(np.nan_to_num(_ref, nan=float(np.nanmedian(_ref))))
            # What this actually IS: the format it arrived in, how it
            # encodes brightness, and whether it can resolve a gradient.
            # All three used to be log-only, which left "is this
            # comparable to my data?" unanswerable from the UI.
            _prov = describe_reference_provenance(
                getattr(self, "_ref_info", None) or
                (self._result.get("diag") or {}).get("ref_info"),
                self._result.get("diag"))
            if _prov:
                self._set_status(
                    _prov +
                    "   ·  the fit absorbs any linear scale difference "
                    "exactly, so units never need matching.")
            # Say WHY it is grey, when it is. A colour survey whose
            # channels all failed but one degrades to that survivor
            # replicated across all three slots -- an honest picture of a
            # mono reference, and a baffling one to look at after picking
            # something with "colour" in its name. The degradation is
            # logged by fetch_reference_channels but never reaches here,
            # so detect it from the planes themselves.
            if _ref.ndim == 3 and _ref.shape[2] >= 2:
                _p0 = np.nan_to_num(_ref[..., 0])
                _same = sum(
                    1 for _c in range(1, _ref.shape[2])
                    if np.allclose(_p0, np.nan_to_num(_ref[..., _c]),
                                   rtol=1e-5, atol=0.0))
                if _same == _ref.shape[2] - 1:
                    self._set_status(
                        f"Reference is MONO: all {_ref.shape[2]} channels "
                        f"resolved to the same plane, so this preview is "
                        f"grey because the DATA is grey — not because the "
                        f"view is. A colour survey degrades to this when "
                        f"every channel but one fails to fetch and the "
                        f"rest are synthesised from the survivor; the "
                        f"[SRGC] log names which one failed. For a "
                        f"reference with real colour, pick a survey whose "
                        f"note above carries no ⚠.")
        elif mode == "Sample points":
            # Exactly the map the points were chosen on: the survey's diffuse
            # structure, plus only the SHARP part of your frame. Your gradient
            # is deliberately invisible here — a point in the brightest corner
            # of your light pollution is a good sample. A point on anything
            # bright in THIS view is sampling structure, which is the catch.
            _rf = np.asarray(self._result["reference"], dtype=np.float32)
            _rf = np.nan_to_num(_rf, nan=float(np.nanmedian(_rf)))
            if _rf.ndim == 3:
                _rf = _to_luminance(_rf)
            arr = _selection_map(self._result.get("target_small"), _rf,
                                 int(self.sp_patch.value()))
            overlay_pts = True
        else:
            # Under min/max the background is normalised AND autostretched,
            # so a field varying by 0.1% fills the full tonal range and looks
            # like structure; the numbers below are then the only honest read.
            # Under linear the picture itself is honest and the numbers just
            # confirm it. Either way, report them.
            is_ref = mode.startswith("Reference")
            raw = np.asarray(self._result["bg_ref" if is_ref else "bg_target"],
                             dtype=np.float32)
            src = np.asarray(self._result["reference" if is_ref
                                          else "target_small"], dtype=np.float32)
            fr = self._result.get("fit_region")
            bg_span = _sp(raw, fr)
            src_span = _sp(src, fr)
            pct = (100.0 * bg_span / src_span) if src_span > 0 else 0.0
            d = self._result.get("diag") or {}
            note = ""
            if is_ref:
                share = d.get("reference_share", 0.0) * 100.0
                note = (f"  Accounts for {share:.0f}% of the background "
                        f"structure; the remaining "
                        f"{d.get('residual_share', 0.0) * 100.0:.0f}% is "
                        f"your own gradient."
                        + ("  Essentially flat — this run is ADBE with a "
                           "survey-derived mask." if share < 5 else ""))
            _who = "Reference" if is_ref else "Target"
            _lead = (f"{_who} background span = {bg_span:.5g} "
                     f"({pct:.2f}% of that image's own range). ")
            if _bgmode == "linear":
                # Gain is fixed at 0.5, so the span occupies half as much of
                # the display range as it does of the data range -- PROVIDED
                # the data is in the target's 0-1 units. The reference is
                # not: it is in survey counts, where a true-scale mapping
                # railed the whole view (a 2409-count span "filling 120480%
                # of the tonal range"). So the reference is shown as the
                # contribution it makes in your units instead.
                _lin = raw
                _units = ""
                if is_ref:
                    _lin, _fb = self._ref_bg_in_target_units(raw)
                    _units = (f"Shown as b·B_r{_fb} — what it "
                              f"contributes in YOUR units, since survey "
                              f"counts mean nothing against a 0-1 frame. ")
                _lin_span = _sp(_lin, fr)
                arr = _linear_mid(_lin, mask=fr, gain=_BG_LINEAR_GAIN)
                raw_display = True
                self._set_status(
                    _lead + _units +
                    f"Display is LINEAR at true scale (median at mid-grey, "
                    f"gain {_BG_LINEAR_GAIN:g}), so its {_lin_span:.5g} "
                    f"span fills "
                    f"{100.0 * _lin_span * _BG_LINEAR_GAIN:.1f}% of the tonal "
                    f"range — a flat background genuinely looks flat "
                    f"here. Switch to min/max to see its shape.{note}")
            else:
                arr = _norm01(raw)
                self._set_status(
                    _lead +
                    f"Display is min-max stretched, so flatness is invisible "
                    f"— trust the number, not the picture. Switch to "
                    f"linear to see the true amplitude.{note}")

        if raw_display:
            disp = np.clip(np.nan_to_num(np.asarray(arr, dtype=np.float32),
                                         nan=0.5, posinf=1.0, neginf=0.0),
                           0.0, 1.0)
        else:
            try:
                # Note for the aggressive path: use_24bit builds a
                # 2**24-entry LUT, ~1.6 s of FIXED cost independent of
                # image size, for a buffer written out as 8-bit and then
                # scaled down to a few hundred pixels. Off. Not
                # bit-identical (max 6/255 on a synthetic worst case) but
                # 32x faster, and the difference does not survive the
                # scale-down.
                disp = np.asarray(self._display_stretch(arr),
                                  dtype=np.float32)
            except Exception as e:
                _log(f"display stretch failed: {_brief(e)}")
                disp = np.clip(arr, 0.0, 1.0)

        if disp.ndim == 2:
            disp = np.repeat(disp[..., None], 3, axis=2)
        buf = np.ascontiguousarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))

        if overlay_pts:
            Hs, Ws = buf.shape[:2]
            # Red tint = vetoed by the survey, matching ADBE's protection
            # preview convention. Green dots = points that survived.
            allowed = self._result.get("allowed")
            vetoed_pct = 0.0
            if allowed is not None and np.asarray(allowed).shape[:2] == (Hs, Ws):
                veto = ~np.asarray(allowed, dtype=bool)
                vetoed_pct = float(veto.mean()) * 100.0
                tint = buf.astype(np.float32)
                tint[veto, 0] = np.minimum(255.0, tint[veto, 0] * 0.55 + 150.0)
                tint[veto, 1] *= 0.45
                tint[veto, 2] *= 0.45
                buf = np.ascontiguousarray(tint.astype(np.uint8))

            pts = self._result.get("points")
            n = 0
            if pts is not None and len(pts):
                p = np.asarray(pts, dtype=int)
                n = len(p)
                for px, py in p:
                    y0, y1 = max(0, py - 2), min(Hs, py + 3)
                    x0, x1 = max(0, px - 2), min(Ws, px + 3)
                    if y0 < y1 and x0 < x1:
                        buf[y0:y1, x0:x1] = (60, 255, 90)

            info = self._result.get("mask_info") or {}
            extra = ""
            if info.get("relaxed"):
                extra = ("  RELAXED — structure dominates this field, so the "
                         "veto was loosened to keep enough sky; treat the "
                         "result with suspicion.")
            self._set_status(
                f"{n} points (green). {vetoed_pct:.0f}% vetoed by the survey "
                f"(red): brightest "
                f"{info.get('exclusion_fraction', 0.5)*100:.0f}% plus "
                f"median+{info.get('protect_k', 3.0):.1f}·MAD, holes filled."
                + extra
            )
        # Cache what was rendered AND the status that went with it, so a
        # replay restores the same reading, not a stale one from whichever
        # view was drawn last.
        try:
            self._disp_cache[_key] = (buf, self.status.text())
            while len(self._disp_cache) > _DISP_CACHE_MAX:
                self._disp_cache.pop(next(iter(self._disp_cache)))
        except Exception:
            pass
        self._show_buffer(buf)

    def _ref_bg_in_target_units(self, bg_r):
        """
        The reference background as what it CONTRIBUTES, not as it is.

        bg_ref is in the survey's own units -- a DSS2 plate arrives as
        1130..20528 counts -- so a true-scale display of it against a
        0-1 frame has no meaning and simply rails to black and white.
        What DOES have meaning is the term the fit subtracts,
        b * B_r: that is in the target's units by construction, so it
        renders on exactly the same mapping as the Target background
        view and the two tabs can be read against each other.

        The constant a is dropped because the display subtracts the
        median anyway, and b keeps its SIGN: an anti-correlated
        reference contributes an inverted surface, and showing it
        upright would misrepresent what is being subtracted.
        """
        a = np.asarray(bg_r, dtype=np.float32)
        coeffs = (self._result or {}).get("coeffs") or []
        if not coeffs:
            # Nothing fitted to borrow a scale from. Fall back to the
            # reference's own range so the view is at least not clipped,
            # and say so rather than implying target units.
            v = a[np.isfinite(a)]
            rng = (float(np.percentile(v, 99.5) - np.percentile(v, 0.5))
                   if v.size >= 8 else 0.0)
            return ((a / rng) if rng > 0 else np.zeros_like(a),
                    " (no fit yet, so scaled by its own range)")
        nch_r = 1 if a.ndim == 2 else a.shape[2]
        planes = []
        for c, (_a, b) in enumerate(coeffs):
            if a.ndim == 2:
                p = a
            elif len(coeffs) == 1:
                # Mono target against a colour reference: the fit used
                # the luminance plane, so the display must too.
                p = _to_luminance(a)
            else:
                p = a[..., min(c, nch_r - 1)]
            planes.append((float(b) * np.asarray(p, np.float32)))
        out = (planes[0] if len(planes) == 1
               else np.stack(planes, axis=-1))
        return np.asarray(out, dtype=np.float32), ""

    def _bgstretch_changed(self, combo):
        """
        One of the background-view combos changed.

        There is one per background tab and they show the same setting,
        so the others are brought into step before the redraw. Signals
        are blocked on the sync so this does not recurse.
        """
        _m = str(combo.currentData() or "minmax")
        if _m == getattr(self, "_bg_stretch_mode", "minmax"):
            return
        self._bg_stretch_mode = _m
        for _c in getattr(self, "_bg_combos", ()):
            if _c is combo:
                continue
            _i = _c.findData(_m)
            if _i >= 0 and _i != _c.currentIndex():
                _c.blockSignals(True)
                _c.setCurrentIndex(_i)
                _c.blockSignals(False)
        self._stretch_changed()

    def _set_bg_stretch_mode(self, mode: str):
        """Set the mode and every combo, without triggering a redraw."""
        _m = str(mode or "minmax")
        if _m not in ("minmax", "linear"):
            _m = "minmax"
        self._bg_stretch_mode = _m
        for _c in getattr(self, "_bg_combos", ()):
            _i = _c.findData(_m)
            if _i >= 0:
                _c.blockSignals(True)
                _c.setCurrentIndex(_i)
                _c.blockSignals(False)

    def _stretch_changed(self, *_):
        """
        Either display-mode combo changed.

        No cache flush: both modes are part of the cache key, so the
        existing entries stay valid and flicking back to a mode you have
        already looked at is a dictionary lookup. The source preview is not
        cached, so that one does have to be rebuilt by hand.
        """
        self._save_settings()
        if self._result is None:
            self.pane_target._pm = None
            self._show_source_now()
        else:
            self._redraw()

    def _display_stretch(self, arr):
        """
        Tone-map a [0,1] array for display, per the chosen mode.

        Display only. Nothing that gets applied to the document passes
        through here.
        """
        a = np.asarray(arr, dtype=np.float32)
        mode = str(self.cmb_stretch.currentData() or "normal")
        if mode == "aggressive":
            return hard_autostretch(a, target_median=0.4, sigma=2,
                                    linked=False, use_24bit=False)
        # Normal: the ordinary statistical stretch, unlinked, 0.25, and no
        # black clip -- clipping the black point on a preview hides exactly
        # the low-level structure you are trying to judge.
        from setiastro.saspro.imageops.stretch import (
            stretch_mono_image, stretch_color_image)
        if a.ndim == 3 and a.shape[2] >= 3:
            return stretch_color_image(a, target_median=0.25, linked=False,
                                       normalize=False, no_black_clip=True)
        if a.ndim == 3:
            a = a[..., 0]
        return stretch_mono_image(a, target_median=0.25, normalize=False,
                                  no_black_clip=True)

    def _show_source_now(self):
        """
        Put the target image in tab 1 before anything has been modelled.

        Exclusion zones are an INPUT to the model -- they change where
        sampling is allowed and what the refinement may touch -- so they
        have to be drawable before the first run, which means there has to
        be something on screen to draw on.
        """
        try:
            img = self._source_float()
            if img is None:
                return
            a = np.asarray(img, dtype=np.float32)
            H, W = a.shape[:2]
            ph, pw = _work_shape(H, W, _PREVIEW_LONG_SIDE)
            small = a if (ph, pw) == (H, W) else _resize_to(a, (ph, pw))
            buf = self._render_buffer(small)
            self.pane_target.set_image(buf, img_hw=(H, W))
            self._set_status(
                "Draw exclusion zones now if you need them, then "
                "'1. Fetch Reference' or '2. Model && Correct'.")
        except Exception as e:
            _log(f"could not show the source image: {_brief(e)}")

    def _render_buffer(self, arr) -> np.ndarray:
        """Stretch a float image into an RGB888 buffer for display."""
        a = np.asarray(arr, dtype=np.float32)
        try:
            d = np.asarray(self._display_stretch(_norm01(a)), dtype=np.float32)
        except Exception as e:
            _log(f"display stretch failed: {_brief(e)}")
            d = np.clip(a, 0.0, 1.0)
        if d.ndim == 2:
            d = np.repeat(d[..., None], 3, axis=2)
        return np.ascontiguousarray((np.clip(d, 0, 1) * 255).astype(np.uint8))

    def _active_pane(self):
        return {0: self.pane_target, 1: self.pane_ref, 2: self.pane_grad,
                3: self.pane_bgt, 4: self.pane_bgr,
                5: self.pane_pts}.get(int(self.tabs.currentIndex()),
                                      self.pane_target)

    def _show_buffer(self, buf: np.ndarray):
        """Hand an already-rendered RGB888 buffer to the active tab."""
        self._preview_buf = buf
        src = self._result.get("source") if self._result else None
        img_hw = (np.asarray(src).shape[:2] if src is not None
                  else buf.shape[:2])
        self._active_pane().set_image(buf, img_hw=img_hw)

    def _exclusion_mask_for(self, img):
        """Hand-drawn zones as a bool mask at the image's own resolution."""
        try:
            hw = np.asarray(img).shape[:2]
            return self.pane_target.exclusion_mask(hw=hw)
        except Exception as e:
            _log(f"could not rasterise the exclusion zones: {_brief(e)}")
            return None

    def _exclusions_changed(self):
        """Exclusions change the model, so the current result is stale."""
        n = len(self.pane_target.polygons)
        self.lbl_zones.setText("" if not n else
                               f"{n} exclusion zone{'s' if n != 1 else ''}")
        if self._result is not None and not self._result.get("preview_only"):
            self._set_status(
                f"{n} exclusion zone(s) drawn — re-run '2. Model && "
                f"Correct' so they take effect on the sampling and the "
                f"dust-mote refinement.")

    # ---- apply ----------------------------------------------------------
    def _sync_push_button(self, *_):
        """Point the corner button's tooltip at the view on screen."""
        btn = getattr(self, "btn_push", None)
        if btn is None:
            return
        idx = int(self.tabs.currentIndex())
        modes = getattr(self, "_TAB_MODES", ())
        mode = modes[idx] if 0 <= idx < len(modes) else ""
        name = self.tabs.tabText(idx)
        tip = _PUSH_TIPS.get(mode, "")
        btn.setToolTip(
            f"Acts on the tab you are on, which is '{name}'.\n\n{tip}"
            if tip else f"Push '{name}' out as its own document.")

    def _push_current_view(self):
        """The corner button: push whichever view is on screen."""
        idx = int(self.tabs.currentIndex())
        modes = getattr(self, "_TAB_MODES", ())
        if not (0 <= idx < len(modes)):
            return None
        return self._push_view_document(modes[idx])

    def _push_view_document(self, mode: str):
        """
        Hand one view to the main window as its own document.

        Each view is a different KIND of thing -- full-resolution
        pixels, a work-grid surface in the target's units, a survey
        frame in survey counts, a display render with dots drawn on
        it -- so each one says what it is in its own status line and
        in the document metadata. Three rules hold across all of
        them:

          * Surfaces go out on the TARGET's pixel grid, upscaled by
            the same _upscale_bg the model used, so they line up with
            your image and can honestly carry its WCS.
          * Anything median-subtracted gets your median added back,
            so it is positive and lands where your data lives.
            Autostretch then shows it, and nothing that clips at zero
            can eat half of it.
          * Only a full-resolution push inherits the header. A
            work-grid push is a different size, and a header whose
            NAXIS disagrees with the array is a trap for whatever
            saves it later.
        """
        import os
        if self._result is None:
            QMessageBox.information(
                self, "Nothing to push",
                "Run '1. Fetch Reference' or '2. Model && Correct' "
                "first.")
            return
        if self._result.get("preview_only") and mode != "Reference":
            QMessageBox.information(
                self, "Nothing to push",
                "Only the reference has been fetched so far, so there "
                "is no gradient and no modelled background yet. Run "
                "'2. Model && Correct' first.")
            return

        mw = self._main
        dm = getattr(mw, "docman", None) or getattr(mw, "doc_manager", None)
        if dm is None or not hasattr(dm, "open_array"):
            QMessageBox.warning(
                self, "No document manager",
                "Could not reach the document manager from here, so "
                "there is nowhere to put the new document.")
            return

        # The snapshot, never the live document -- same reason Apply
        # uses it: after Apply the document already carries the
        # correction, and pushing a view built from it would be a
        # second subtraction.
        img = self._result.get("source")
        if img is None:
            img = self._source_float()
        if img is None:
            return
        img = np.asarray(img, dtype=np.float32)
        H, W = img.shape[:2]
        mono = (img.ndim == 2)
        med = float(np.nanmedian(img))

        def _full(a):
            """A work-grid surface, put on the target's pixel grid."""
            a = np.asarray(a, dtype=np.float32)
            if a.shape[:2] != (H, W):
                a = np.asarray(_upscale_bg(a, (H, W)), dtype=np.float32)
            if mono and a.ndim == 3:
                a = a[..., 0]
            return np.ascontiguousarray(a, dtype=np.float32)

        arr = None
        tag = ""
        what = ""
        full_grid = True

        if mode == "Corrected":
            if self.chk_before.isChecked():
                arr = img.copy()
                tag = "Before"
                what = ("the uncorrected snapshot this model was built "
                        "from, at full resolution")
            else:
                # Exactly what Apply would commit, refinement included
                # when it is current for this strength.
                _cf = self._result.get("corrected_full")
                if (_cf is not None
                        and self._result.get("ms_strength") is not None
                        and int(self._result["ms_strength"])
                        == int(self.sp_strength.value())):
                    arr = np.asarray(_cf, dtype=np.float32)
                else:
                    arr = apply_survey_gradient(
                        img, self._result["gradient"],
                        self.sp_strength.value() / 100.0)
                tag = "Corrected"
                what = (f"the corrected image at "
                        f"{int(self.sp_strength.value())}% strength "
                        f"— what Apply would commit, as a new document "
                        f"instead of an edit")

        elif mode == "Gradient removed":
            arr = _full(np.asarray(self._result["gradient"],
                                   dtype=np.float32) + med)
            tag = "Gradient"
            what = (f"the surface being subtracted, in your image's "
                    f"units, with your median ({med:.5g}) added back "
                    f"— subtract it and restore the median to redo "
                    f"the correction by hand")

        elif mode == "Target background":
            arr = _full(self._result["bg_target"])
            tag = "Background"
            what = ("your modelled background, already in your image's "
                    "units and at its own level, so it is left exactly "
                    "as the model made it")

        elif mode == "Reference background":
            # b*B_r, for the same reason the linear view shows it:
            # bg_ref is in survey counts, which mean nothing next to a
            # 0-1 frame, while the term the fit subtracts is in your
            # units by construction.
            _br, _fb = self._ref_bg_in_target_units(
                np.asarray(self._result["bg_ref"], dtype=np.float32))
            _br = np.asarray(_br, dtype=np.float32)
            _bm = float(np.nanmedian(_br))
            arr = _full(_br - _bm + med)
            tag = "RefBackground"
            what = (f"b·B_r{_fb} — what the reference contributes "
                    f"in YOUR units, median moved to yours ({med:.5g}) "
                    f"so it blinks against the target background")

        elif mode == "Reference":
            _ref = np.asarray(self._result["reference"], dtype=np.float32)
            _ref = np.nan_to_num(_ref, nan=float(np.nanmedian(_ref)))
            _lo = float(np.nanmin(_ref))
            _hi = float(np.nanmax(_ref))
            if _lo < 0.0 or _hi > 1.0:
                # A DSS2 plate arrives as 1130..20528 counts. Pushed
                # raw, it would be the one document in SASpro that
                # does not live in 0-1, and the first tool that clips
                # would flatten it to white. The fit absorbs any
                # linear scale anyway, so the range is a label, not
                # data -- rescale and record it.
                _ref = ((_ref - _lo)
                        / max(_hi - _lo, 1e-12)).astype(np.float32)
                _sc = f", rescaled from its own {_lo:.6g}..{_hi:.6g}"
            else:
                _sc = ", in the units it arrived in (already 0-1)"
            arr = _ref
            full_grid = False
            tag = "Reference"
            what = (f"the fetched reference on the work grid "
                    f"({arr.shape[1]}×{arr.shape[0]}, not your "
                    f"resolution){_sc}")

        elif mode == "Sample points":
            # The only view whose content exists solely as a render:
            # the dots and the veto tint are painted into the 8-bit
            # buffer, so that buffer IS the thing being asked for.
            # Make sure it is the one on screen first.
            try:
                _i = self._TAB_MODES.index(mode)
            except ValueError:
                _i = -1
            if _i >= 0 and int(self.tabs.currentIndex()) != _i:
                self.tabs.setCurrentIndex(_i)
                self._redraw()
            _buf = getattr(self, "_preview_buf", None)
            if _buf is None:
                QMessageBox.information(
                    self, "Nothing to push",
                    "The point map has not been drawn yet.")
                return
            arr = (np.asarray(_buf, dtype=np.float32) / 255.0)
            full_grid = False
            tag = "Points"
            what = ("the point map as drawn — green = used, red tint "
                    "= vetoed by the survey. A DISPLAY render promoted "
                    "from 8-bit, so it is a picture for the record, "
                    "not data to measure")

        if arr is None:
            return
        arr = np.nan_to_num(np.asarray(arr, dtype=np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.ascontiguousarray(arr, dtype=np.float32)

        try:
            base = os.path.splitext(str(self.doc.display_name()))[0]
        except Exception:
            base = "image"
        title = f"{base}_SRGC_{tag}"
        meta = {
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2),
            "source": f"SRGC {tag} — {what}",
            "survey_gradient": {
                "view": mode,
                "survey": self.cmb_survey.currentText(),
                "strength": self.sp_strength.value() / 100.0,
                "coverage": float(self._result.get("coverage") or 0.0),
                "coeffs": [[float(_a), float(_b)] for _a, _b
                           in (self._result.get("coeffs") or [])],
                "grid": "full resolution" if full_grid else "work grid",
            },
        }
        if full_grid:
            # Shares the target's pixel grid, so it can carry its WCS.
            # Copied, not referenced: open_array keeps the object it is
            # given, and a plate solve on the pushed document would
            # otherwise write its cards straight into the header of the
            # document it came from.
            _h = (getattr(self.doc, "metadata", None) or {}).get(
                "original_header")
            if _h is not None:
                try:
                    _h = _h.copy()
                except Exception:
                    pass
                meta["original_header"] = _h

        try:
            _newdoc = dm.open_array(arr, metadata=meta, title=title)
        except Exception as e:
            QMessageBox.critical(self, "Push failed", str(e))
            return
        # open_array registers the document, which the MDI answers by
        # spawning a subwindow; this raises and focuses the one that
        # already exists rather than making a second.
        if hasattr(mw, "_spawn_subwindow_for"):
            try:
                mw._spawn_subwindow_for(_newdoc)
            except Exception:
                pass
        if hasattr(mw, "_log"):
            try:
                mw._log(f"[SRGC] pushed '{title}' {arr.shape} — {what}")
            except Exception:
                pass
        _log(f"pushed '{title}' shape={arr.shape} — {what}")
        self._set_status(f"Pushed '{title}' — {what}.")
        return _newdoc

    def _do_apply(self):
        if self._result is None:
            return
        # Same snapshot the preview used. Applying twice from the live
        # document would subtract the gradient twice; this way a second
        # Apply is idempotent — it recommits the same corrected pixels.
        img = self._result.get("source")
        if img is None:
            img = self._source_float()
        if img is None:
            return
        strength = self.sp_strength.value() / 100.0
        # Commit the refined full-resolution result when the refinement ran
        # at this strength; otherwise the additive correction alone.
        _cf = self._result.get("corrected_full")
        if (_cf is not None and self._result.get("ms_strength") is not None
                and int(self._result["ms_strength"]) == int(self.sp_strength.value())):
            out = np.asarray(_cf, dtype=np.float32)
        else:
            out = apply_survey_gradient(img, self._result["gradient"], strength)

        name = self.cmb_survey.currentText()
        step = f"Survey Gradient ({name}, {int(strength*100)}%)"
        meta = {
            "step_name": "SurveyGradient",
            "survey_gradient": {
                "survey": name,
                "channels": [c["hips"] if isinstance(c, dict) else c
                             for c in self._channels()],
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


# Each push is a different kind of object and the tooltip is where that
# gets said before the click, not after. Keyed by _TAB_MODES.
_PUSH_TIPS = {
    "Corrected": (
        "Push the corrected image out as a NEW document, at full\n"
        "resolution and the current strength — the same pixels Apply\n"
        "would commit, but leaving this document untouched.\n\n"
        "With Before ticked you get the uncorrected snapshot instead,\n"
        "so you can keep both and blink them."),
    "Reference": (
        "Push the fetched reference out as a new document.\n\n"
        "It is on the WORK grid, not your resolution, and it arrives\n"
        "in the survey's own units — so if those fall outside 0-1 it\n"
        "is rescaled by its own range and the range is recorded in\n"
        "the metadata. The fit absorbs any linear scale, so nothing\n"
        "the correction used is lost."),
    "Gradient removed": (
        "Push the subtracted surface out as a new document, at full\n"
        "resolution and in your image's units.\n\n"
        "Your median is added back, because the gradient is\n"
        "median-subtracted and half of a zero-centred surface would\n"
        "sit below black. Subtract this document and restore the\n"
        "median and you have redone the correction by hand."),
    "Target background": (
        "Push your modelled background out as a new document, at full\n"
        "resolution.\n\n"
        "This is the one surface that needs no adjusting: it is\n"
        "already in your units and at your sky level, so it goes out\n"
        "exactly as the model made it. ADBE's background document is\n"
        "the same idea."),
    "Reference background": (
        "Push the reference's contribution out as a new document, at\n"
        "full resolution.\n\n"
        "What you get is b·B_r — the term the fit subtracts — not\n"
        "the raw survey background, because that is in survey counts\n"
        "and means nothing beside a 0-1 frame. Its median is moved to\n"
        "yours so it can be blinked against the target background."),
    "Sample points": (
        "Push the point map out as a new document.\n\n"
        "This one is a PICTURE, not data: the green dots and the red\n"
        "veto tint are painted into the display render, so what you\n"
        "get is that render promoted to float at work-grid size. Good\n"
        "for a record of what was sampled; not something to measure."),
}

_BG_LINEAR_GAIN = 0.5


def _linear_mid(a: np.ndarray, mask: np.ndarray | None = None,
                gain: float = _BG_LINEAR_GAIN) -> np.ndarray:
    """
    0.5 + (a - median(a)) * gain -- a background at its TRUE amplitude.

    Purely linear and purely additive, so what you see is the gradient's
    real size rather than its shape blown up to fill the screen. The
    median lands exactly on mid-grey; a 0.002 background is then a 0.001
    excursion either side of grey, i.e. almost nothing, which is the
    honest picture of an almost-flat field.

    Per channel, because an OSC background carries a different offset in
    each -- a single global median would push R and B off mid-grey and
    read as a colour cast that is not in the gradient.

    `mask` (the fit region) keeps off-footprint pixels out of the median.
    Non-finite pixels are set to the median, so no-data reads as grey
    rather than as real dark structure.
    """
    a = np.asarray(a, dtype=np.float32)
    g = float(gain)
    m = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape[:2] != a.shape[:2] or not m.any():
            m = None

    def _one(p: np.ndarray) -> np.ndarray:
        v = p[m] if m is not None else p
        v = v[np.isfinite(v)]
        med = float(np.median(v)) if v.size else 0.0
        q = np.where(np.isfinite(p), p, med).astype(np.float32)
        return (0.5 + (q - med) * g).astype(np.float32)

    if a.ndim == 2:
        out = _one(a)
    else:
        out = np.empty_like(a, dtype=np.float32)
        for c in range(a.shape[2]):
            out[..., c] = _one(a[..., c])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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