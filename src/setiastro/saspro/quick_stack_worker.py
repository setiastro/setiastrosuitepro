# src/setiastro/saspro/quick_stack_worker.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Quick Stack engine (GUI-free).

    load (Live Stacking's load_calibrated, bilinear debayer; a master dark and
    flat, if given, go on the mosaic first, made to fit the frames once per run)
      -> similarity delta to the reference (the alignment pass's own helpers,
         with the meridian-flip check)
      -> bilinear warp straight into shared accumulators
      -> trimmed mean

No weighting, rejection maps, distortion or normalization. Beyond the masters,
the one thing applied per frame is a single additive sky offset per channel (a
strided median), which makes the trim compare frames at the same sky level.
Calibrated and aligned frames live in memory only.

Trimming is streamed: each output pixel keeps its sum, its count, and its two
highest and two lowest samples, so memory does not depend on the number of
frames and the result does not depend on the order they finish in. The trim
is symmetric (the same number off each end): an uneven one biases the mean,
and where the count per pixel changes (dithered edges) the bias would step.
Accumulators live in shared memory and workers add to them in bands under a
lock per band, so there is one set for the whole run, not one per worker.

This module is the CPU engine, and the tasks both engines share. With an
NVIDIA GPU the pixels go through quick_stack_gpu instead; loading frames and
matching star lists stay on these CPU workers either way.

0.0 (and non-finite) means no data, as everywhere else in SASpro: such
samples are never added and never count.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import numba as nb

N_BANDS = 32
MIN_TRIM_FRAMES = 6        # below this a plain mean; 6-14 drop 1 high and 1 low; >= 15 drop 2 and 2
FULL_TRIM_FRAMES = 15
MIN_INLIERS = 10           # a match needs at least this many stars, and this share of the stars
MIN_INLIER_FRACTION = 0.25 # available: unrelated fields reach ~9 by chance, a true match far more
SOLVE_GRID_MP = 3.0        # the solve grid is about this many megapixels, so star detection costs
MAX_SOLVE_DOWNSAMPLE = 3   # about the same on any sensor; small frames are solved at full resolution

RAW_EXTS = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef')
FRAME_EXTS = ('.fits', '.fit', '.fts', '.fits.gz', '.fit.gz', '.fz', '.xisf',
              '.tif', '.tiff') + RAW_EXTS


# ─────────────────────────────────────────────────────────────────────────────
# Frame list and grouping
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FrameInfo:
    path: str
    filter: str = "Unknown"
    exposure: float = 0.0
    width: int = 0
    height: int = 0
    bayer: str = ""
    obj: str = ""
    binning: str = ""
    gain: float = 0.0
    roworder: str = ""
    imagetyp: str = ""          # what the file says it is (route_file sorts on it)
    bitpix: int = 0             # as stored: negative for floating-point pixels
    ncombine: int = 0


@dataclass
class FrameGroup:
    key: tuple
    frames: list = field(default_factory=list)

    @property
    def label(self) -> str:
        f = self.frames[0]
        parts = [f.filter or "Unknown"]
        if f.exposure:
            parts.append(f"{f.exposure:g} s")
        if f.width and f.height:
            parts.append(f"{f.width} x {f.height}")
        if f.bayer:
            parts.append(f.bayer)
        return "  ·  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Master dark and flat
# ─────────────────────────────────────────────────────────────────────────────
MASTER_FITS = ('.fits', '.fit', '.fts', '.fits.gz', '.fit.gz', '.fz')
MASTER_EXTS = MASTER_FITS + ('.xisf', '.tif', '.tiff')


@dataclass
class MasterChoice:
    """The masters one group is stacked with (file paths, or None)."""
    dark: str | None = None
    flat: str | None = None
    bayer: str = ""
    how: dict = field(default_factory=dict)     # "dark"/"flat" -> what was done to make it fit

    def __bool__(self) -> bool:
        return bool(self.dark or self.flat)


class MasterMismatch(ValueError):
    """A master that does not fit a frame. That frame is skipped; the run goes on."""


def _same_size(m: FrameInfo, f: FrameInfo) -> bool:
    return not (m.width and f.width) or (m.width, m.height) == (f.width, f.height)


def _size_why(mw, mh, fw, fh) -> str:
    """What two sizes suggest, when they suggest something."""
    if (mw, mh) == (fh, fw):
        return ", turned 90°"
    if (mw, mh) == (2 * fw, 2 * fh) or (2 * mw, 2 * mh) == (fw, fh):
        return ", another binning?"
    return ""


def _sizes_of(masters: list, frame: FrameInfo) -> str:
    """ (a.fit is 1080 x 1920, turned 90°; …) for a note, or ""."""
    known = [m for m in masters if m.width and m.height]
    if not known or not frame.width:
        return ""
    parts = [f"{os.path.basename(m.path)} is {m.width} x {m.height}"
             f"{_size_why(m.width, m.height, frame.width, frame.height)}" for m in known[:2]]
    return " (" + "; ".join(parts) + ("; …" if len(known) > 2 else "") + ")"


same_size = _same_size


# Filter names as capture programs and filter makers write them. A one-letter
# name counts only when it is the whole name ("H", "R"), never as a word of a
# longer one: the L of "L-eXtreme" is not luminance.
_FILTER_NAMES = {
    "ha": ("ha", "halpha", "hα", "hydrogenalpha", "h"),
    "hb": ("hb", "hbeta", "hβ"),
    "oiii": ("oiii", "o3", "oxygen", "oxygeniii", "o"),
    "sii": ("sii", "s2", "sulfur", "sulphur", "sulfurii", "sulphurii", "s"),
    "nii": ("nii", "n2"),
    "l": ("l", "lum", "luminance"),
    "r": ("r", "red"),
    "g": ("g", "green"),
    "b": ("b", "blue"),
}
_FILTER_ALIAS = {a: k for k, names in _FILTER_NAMES.items() for a in names}
FILTER_LABEL = {"ha": "Ha", "hb": "Hβ", "oiii": "OIII", "sii": "SII", "nii": "NII",
                "l": "L", "r": "R", "g": "G", "b": "B"}
_BANDWIDTH = re.compile(r"\d+(?:[.,]\d+)?\s*nm")
_WORD = re.compile(r"[0-9a-zα-ω]+")


def _filter_words(text) -> list:
    return _WORD.findall(_BANDWIDTH.sub(" ", str(text or "").lower()))


def filter_key(name) -> str:
    """A filter name reduced for matching: Ha, H-alpha, "Ha 7nm" and "Antlia
    3nm Ha Pro" all give "ha", O3 and OIII "oiii". A name it does not know
    comes back without spaces and punctuation ("L-eNhance" -> "lenhance"); no
    name, or "Unknown", gives ""."""
    words = _filter_words(name)
    compact = "".join(words)
    if not compact or compact == "unknown":
        return ""
    if compact in _FILTER_ALIAS:
        return _FILTER_ALIAS[compact]
    pairs = [a + b for a, b in zip(words, words[1:])]
    hits = {_FILTER_ALIAS[w] for w in words + pairs if len(w) > 1 and w in _FILTER_ALIAS}
    return hits.pop() if len(hits) == 1 else compact


def filter_from_name(path: str) -> str:
    """The filter a master's file name names, else its folder's
    ("MasterFlat_Ha.fit", "masterFlat_FILTER-R.xisf", "Flats/OIII/x.fit"), as
    filter_key gives it; "" if neither names one. In a file name a one-letter
    filter counts when nothing longer is there."""
    for part in (os.path.basename(path), os.path.basename(os.path.dirname(path))):
        words = _filter_words(part)
        pairs = [a + b for a, b in zip(words, words[1:])]
        hits = {_FILTER_ALIAS[w] for w in words + pairs if len(w) > 1 and w in _FILTER_ALIAS}
        if not hits:
            hits = {_FILTER_ALIAS[w] for w in words if len(w) == 1 and w in _FILTER_ALIAS}
        if len(hits) == 1:
            return hits.pop()
    return ""


@dataclass
class Pick:
    """The master of one kind for one group, and why. state:
        ""           no master of this kind is loaded
        "auto"       matched: size, then filter (flat) or gain and exposure (dark)
        "name"       a flat whose filter is written another way (O3 for OIII)
        "file"       a flat with no filter card, matched on its file or folder name
        "only"       the one flat loaded, used whatever its filter
        "none"       none fits; why: "size", "filter" or "exposure"
        "hand"       chosen in the window (why "size" if it is not the frames' size)
        "hand-none"  no master, chosen in the window"""
    master: FrameInfo | None = None
    state: str = ""
    why: str = ""
    notes: list = field(default_factory=list)


def _size_text(frame: FrameInfo) -> str:
    return f"{frame.width} x {frame.height}" if frame.width else "these frames' size"


def _dark_notes(dark: FrameInfo, frame: FrameInfo) -> list:
    notes = []
    if dark.exposure and frame.exposure and abs(dark.exposure - frame.exposure) > 0.05 * frame.exposure:
        notes.append(f"⚠ The master dark is {dark.exposure:g} s, the frames are {frame.exposure:g} s.")
    elif frame.exposure and not dark.exposure:
        notes.append(f"⚠ The master dark's exposure is 0 s or not recorded; the frames are "
                     f"{frame.exposure:g} s.")
    if dark.gain and frame.gain and dark.gain != frame.gain:
        notes.append(f"⚠ The master dark was taken at gain {dark.gain:g}, the frames at {frame.gain:g}.")
    return notes


def flat_filter_key(m: FrameInfo) -> str:
    """A flat's filter: its card, else what its file or folder name says."""
    return filter_key(m.filter) or filter_from_name(m.path)


def _flat_notes(flat: FrameInfo, frame: FrameInfo) -> list:
    fk, mk = filter_key(frame.filter), flat_filter_key(flat)
    if fk and mk and fk != mk:
        said = flat.filter if filter_key(flat.filter) else FILTER_LABEL.get(mk, mk)
        return [f"⚠ The master flat is for {said}, the frames are {frame.filter}."]
    return []


def _pick_dark(frame: FrameInfo, darks: list) -> Pick:
    """Same size, then the same gain, then the nearest exposure; with no
    exposure in the frames, only if there is no choice to make."""
    if not darks:
        return Pick()
    fit = [m for m in darks if _same_size(m, frame)]
    if not fit:
        return Pick(None, "none", "size", [f"⚠ No loaded master dark is {_size_text(frame)}"
                                           f"{_sizes_of(darks, frame)}: none was applied."])
    if not frame.exposure and len({m.exposure for m in fit}) > 1:
        return Pick(None, "none", "exposure", [
            "⚠ The frames do not say their exposure and master darks of several exposures are loaded: "
            "none was applied. Choose one for this group in the Quick Stack window."])
    dark = min(fit, key=lambda m: (bool(frame.gain and m.gain and m.gain != frame.gain),
                                   abs(m.exposure - frame.exposure)
                                   if m.exposure and frame.exposure else 1e9,
                                   m.path))
    return Pick(dark, "auto", "", _dark_notes(dark, frame))


def _pick_flat(frame: FrameInfo, flats: list) -> Pick:
    """Same size, then the same filter: as written, then written another way
    (O3 for OIII), then named in the file or folder name of a flat with no
    filter card. A single loaded flat is used whatever its filter."""
    if not flats:
        return Pick()
    fit = [m for m in flats if _same_size(m, frame)]
    if not fit:
        return Pick(None, "none", "size", [f"⚠ No loaded master flat is {_size_text(frame)}"
                                           f"{_sizes_of(flats, frame)}: none was applied."])
    exact = [m for m in fit if m.filter.upper() == frame.filter.upper()]
    if exact:
        return Pick(exact[0], "auto")
    key = filter_key(frame.filter)
    if key:
        named = [m for m in fit if filter_key(m.filter) == key]
        if named:
            m = named[0]
            return Pick(m, "name", "", [f"The master flat {os.path.basename(m.path)} is for {m.filter}, "
                                        f"taken as the frames' {frame.filter}."])
        by_file = [m for m in fit if not filter_key(m.filter) and filter_from_name(m.path) == key]
        if by_file:
            m = by_file[0]
            return Pick(m, "file", "", [f"The master flat {os.path.basename(m.path)} has no filter card; "
                                        f"its name says {frame.filter}."])
    if len(flats) == 1:
        return Pick(fit[0], "only", "", _flat_notes(fit[0], frame))
    return Pick(None, "none", "filter", [f"⚠ No loaded master flat is for {frame.filter}: none was applied."])


def _hand_pick(kind: str, path, pool: list, frame: FrameInfo) -> Pick | None:
    """A master chosen in the window (None: no master). None if it is no
    longer loaded, so the group falls back to its automatic choice."""
    if path is None:
        return Pick(None, "hand-none", "", [f"No master {kind}: chosen in the Quick Stack window."])
    m = next((m for m in pool if os.path.normcase(m.path) == os.path.normcase(path)), None)
    if m is None:
        return None
    notes = [f"Master {kind} {os.path.basename(m.path)}: chosen in the Quick Stack window."]
    notes += _dark_notes(m, frame) if kind == "dark" else _flat_notes(m, frame)
    # one of another size is said by MasterSet.load, which leaves it out
    return Pick(m, "hand", "" if _same_size(m, frame) else "size", notes)


def plan_masters(frame: FrameInfo, darks: list, flats: list, chosen=None) -> dict:
    """{"dark": Pick, "flat": Pick} for a group (its first frame). `chosen`
    is what was picked in the window: {"dark"/"flat": a master's path, or
    None for no master}; a kind it does not hold is picked automatically."""
    chosen = chosen or {}
    out = {}
    for kind, pool, auto in (("dark", darks, _pick_dark), ("flat", flats, _pick_flat)):
        p = _hand_pick(kind, chosen[kind], pool, frame) if kind in chosen else None
        out[kind] = p or auto(frame, pool)
    return out


def pick_masters(frame: FrameInfo, darks: list, flats: list, chosen=None):
    """The dark and the flat for a group, and lines saying what was chosen
    (plan_masters says how). Returns (MasterChoice, [notes])."""
    notes: list = []
    bayer = frame.bayer
    if frame.path.lower().endswith(RAW_EXTS) and (darks or flats):
        from setiastro.saspro.live_stack_worker import raw_cfa
        family, token = raw_cfa(frame.path)
        if family == "XTRANS":
            return MasterChoice(), ["⚠ These are X-Trans RAW frames: a master dark or flat cannot be "
                                    "applied to them."]
        bayer = bayer or token or ""
    plan = plan_masters(frame, darks, flats, chosen)
    for p in plan.values():
        notes.extend(p.notes)
    dark, flat = plan["dark"].master, plan["flat"].master
    used = [(w, m) for w, m in (("dark", dark), ("flat", flat)) if m is not None]
    for w, m in used:
        # load_image turns a BOTTOM-UP frame over, and can only do the same to a master that says so.
        if (frame.roworder == "BOTTOM-UP" and not m.roworder
                and m.path.lower().endswith(MASTER_FITS)):
            notes.append(f"⚠ The frames are stored bottom-up and the master {w} does not say how it is "
                         f"stored (no ROWORDER card): it may be upside down against them.")
    # "Calibrating with …" comes from MasterSet.load, once the masters are known to fit.
    return MasterChoice(dark.path if dark else None, flat.path if flat else None, bayer), notes


def _quiet():
    """The loaders narrate every file. Silenced in worker processes only:
    redirect_stdout swaps sys.stdout for the whole process."""
    return (contextlib.redirect_stdout(io.StringIO()) if _in_child_process()
            else contextlib.nullcontext())


def _bayer_of(hdr) -> str:
    try:
        bp = hdr.get("BAYERPAT") if hdr is not None else None
    except Exception:
        return ""
    bp = bp[0] if isinstance(bp, tuple) else bp
    s = str(bp or "").strip().upper()
    return s if s in BAYER else ""


def load_master(path: str):
    """(pixels, header) of a master as it is stored: (H,W) or (H,W,C) float32
    in the frames' [0,1] units. Raises MasterMismatch if it cannot be read."""
    with _quiet():
        from setiastro.saspro.legacy.image_manager import load_image
        img, hdr = load_image(path)[:2]
    if img is None:
        raise MasterMismatch("could not be read")
    img = np.asarray(img, np.float32)
    fin = img[np.isfinite(img)]
    if fin.size and float(fin.max()) > 10.0:          # ADU in a float file (stacking_suite's units guard)
        img = img / np.float32(65535.0)
    return np.ascontiguousarray(img, np.float32), hdr


@dataclass
class FrameLayout:
    """A group's frames as the calibration step gets them, before the debayer."""
    shape: tuple            # (H, W)
    channels: int           # 1: a mono plane or a raw Bayer mosaic; 3: debayered already
    bayer: str              # the mosaic's pattern, "" if none or unknown (only a hint if channels > 1)
    median: float           # of their samples, no-data zeros left out (nan if none)


def frame_layout(paths) -> FrameLayout | None:
    """The layout of the first of a group's frames that loads (of the first three)."""
    from setiastro.saspro.legacy.image_manager import load_image
    for p in list(paths)[:3]:
        try:
            with _quiet():
                img, hdr = load_image(p)[:2]
            if img is None:
                continue
            img = np.asarray(img, np.float32)
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[..., 0]
            bayer = _bayer_of(hdr)
            if img.ndim == 2 and str(p).lower().endswith(RAW_EXTS):
                from setiastro.saspro.live_stack_worker import raw_cfa
                bayer = raw_cfa(p)[1] or ""
            s = img[::4, ::4]
            s = s[np.isfinite(s) & (s != 0)]
            return FrameLayout(tuple(img.shape[:2]), 1 if img.ndim == 2 else int(img.shape[2]),
                               bayer, float(np.median(s)) if s.size else float("nan"))
        except Exception:
            continue
    return None


_RGB = {"R": 0, "G": 1, "B": 2}
NATIVE_MARGIN = 0.5


def remosaic(rgb: np.ndarray, pattern: str) -> np.ndarray:
    """The Bayer mosaic inside a debayered (H,W,3) image: at each site of the
    2x2 cell, the channel `pattern` says the camera measured there. For a
    demosaic that keeps the measured samples (SASpro's all do, as do most)
    this is the raw mosaic again, exactly."""
    out = np.empty(rgb.shape[:2], np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            out[dy::2, dx::2] = rgb[dy::2, dx::2, _RGB[pattern[2 * dy + dx]]]
    return out


def native_phase(rgb: np.ndarray):
    """(pattern, margin): the Bayer pattern a debayered image was made with,
    read from its pixels, and how far it is ahead of the next best.

    A bilinear-type demosaic fills a channel's missing sites with averages of
    measured ones, so pixel to pixel each channel is noisiest where the camera
    measured it. A pattern scores, over the four sites of the cell, the noise
    of that site's channel against the channel's noisiest site: the measured
    layout scores 4. Real Dwarf darks and flats through bilinear-type demosaics
    (SASpro's bilinear and edge, OpenCV's bilinear and edge-aware): margin 0.9
    or more, always the right pattern. RCD, VNG and AHD rebuild the missing
    colours from the measured channel's detail and leave nothing to see:
    margin 0.34 or less, as for real colour stacks, which hold no mosaic."""
    H, W = rgb.shape[:2]
    s = 1024
    y0, x0 = max(0, (H - s) // 2) & ~1, max(0, (W - s) // 2) & ~1
    c = rgb[y0:y0 + s, x0:x0 + s]
    noise = np.zeros((2, 2, 3))
    for dy in (0, 1):
        for dx in (0, 1):
            for ch in range(3):
                p = c[dy::2, dx::2, ch]
                d = np.concatenate([np.abs(np.diff(p, axis=1)).ravel(),
                                    np.abs(np.diff(p, axis=0)).ravel()])
                d = d[np.isfinite(d)]
                noise[dy, dx, ch] = float(np.median(d)) if d.size else 0.0
    top = noise.max(axis=(0, 1))
    if not np.all(top > 0):
        return None, 0.0
    rel = noise / top
    score = {p: float(sum(rel[dy, dx, _RGB[p[2 * dy + dx]]] for dy in (0, 1) for dx in (0, 1)))
             for p in BAYER}
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    return ranked[0][0], ranked[0][1] - ranked[1][1]


def _looks_like_mosaic(plane: np.ndarray) -> bool:
    """Do the four sites of the 2x2 cell sit at different levels, as a raw
    colour flat's do (red and blue well below green)? A mono plane's do not."""
    c = plane[:min(plane.shape[0], 2048) & ~1, :min(plane.shape[1], 2048) & ~1]
    meds = []
    for dy in (0, 1):
        for dx in (0, 1):
            v = c[dy::2, dx::2]
            v = v[np.isfinite(v) & (v != 0)]
            if not v.size:
                return False
            meds.append(float(np.median(v)))
    mid = float(np.median(meds))
    return mid > 0 and (max(meds) - min(meds)) / mid > 0.05


def fit_master(img, hdr, what: str, lay: FrameLayout, name: str):
    """A master made to fit a group's frames, or the reason it cannot be.

    Returns (pixels, sites, how, notes). pixels: (H,W), or (H,W,3) for frames
    that are debayered already; None if it cannot be used. sites: the Bayer
    pattern when pixels is a mosaic (the flat is then normalized per site).
    how: a few words on what was done to it. notes: lines for the log.

    A raw mosaic master on debayered frames is debayered as Quick Stack
    debayers; a debayered master on raw frames gives its mosaic back; a size,
    rotation or binning that differs is not guessed at. A master that cannot
    be made to fit is left out with a line saying why, never an error."""
    a = np.asarray(img, np.float32)
    notes: list = []
    sites = how = ""

    def no(why):
        return None, "", "", [f"⚠ The master {what} {name} {why}: not applied."]

    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    elif a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
        a = np.moveaxis(a, 0, -1)                      # channels first
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]                                 # alpha
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] != 3):
        return no(f"has a shape Quick Stack cannot use ({' x '.join(map(str, np.shape(img)))})")
    if a.ndim == 3 and all(np.array_equal(a[..., 0], a[..., c], equal_nan=True) for c in (1, 2)):
        a = a[..., 0]                                  # a mono plane saved as RGB
    H, W = lay.shape
    if a.shape[:2] != (H, W):
        mh, mw = a.shape[:2]
        return no(f"is {mw} x {mh}, the frames are {W} x {H}{_size_why(mw, mh, W, H)}")
    s = a[::4, ::4]
    s = s[np.isfinite(s) & (s != 0)]
    if not s.size:
        return no("holds no data")

    if lay.channels > 1:                               # the frames are debayered already
        if a.ndim == 3:
            if lay.channels != 3:
                return no(f"has 3 colour channels, the frames have {lay.channels}")
        else:
            pat = _bayer_of(hdr) or lay.bayer
            if pat:
                from setiastro.saspro.legacy.numba_utils import debayer_fits_fast
                a = np.asarray(debayer_fits_fast(a, pat, method="bilinear"), np.float32)
                how = f"debayered as {pat} to match frames that are already colour"
            elif _looks_like_mosaic(a):
                return no("is a raw Bayer mosaic with no BAYERPAT card and the frames are "
                          "debayered already, so it cannot be debayered to match them")
            else:
                how = "one plane, applied to each colour channel"
    elif a.ndim == 3:                                  # a debayered master, raw frames
        pat, margin = native_phase(a)
        if pat and margin >= NATIVE_MARGIN:
            if lay.bayer and pat != lay.bayer:
                notes.append(f"⚠ The master {what} {name} was debayered as {pat}, the frames say "
                             f"{lay.bayer}: its own {pat} samples were used. If it was made with these "
                             f"frames' camera, it may be upside down or mirrored against them.")
        elif lay.bayer:
            pat = lay.bayer
        else:
            return no("is a colour image and the frames are single-plane with no Bayer pattern, "
                      "so its mosaic cannot be taken back out")
        a = remosaic(a, pat)
        sites = pat
        how = f"its {pat} mosaic, taken back out of a debayered master"
    else:
        sites = lay.bayer

    if what == "dark" and np.isfinite(lay.median):
        d = a[::4, ::4]
        d = d[np.isfinite(d) & (d != 0)]
        md = float(np.median(d)) if d.size else 0.0
        if md > 2.0 * lay.median + 0.002:
            return no(f"is far brighter than the frames (median {md:.4f}, theirs {lay.median:.4f}): "
                      f"it is not a dark for them, or is in other units")
    if what == "flat":
        f = a[::4, ::4]
        if float(np.mean(~np.isfinite(f) | (f <= 0))) > 0.5:
            return no("is mostly empty (zero, negative or not a number)")
    return np.ascontiguousarray(a, np.float32), sites, how, notes


def normalize_flat(flat: np.ndarray, bayer: str = "") -> np.ndarray:
    """A flat divided by its own median (by its colour plane's median for a
    Bayer mosaic, as the Stacking Suite does, and by each channel's for an
    (H,W,C) flat). Nothing to divide by where it is not finite or not
    positive: those pixels become 1."""
    out = np.array(flat, np.float32, copy=True)
    bad = ~np.isfinite(out) | (out <= 0)

    def plane(view, mask):
        v = view[~mask]
        med = float(np.median(v)) if v.size else 1.0
        return med if np.isfinite(med) and med > 0 else 1.0

    if out.ndim == 3:
        for c in range(out.shape[2]):
            out[..., c] /= np.float32(plane(out[..., c], bad[..., c]))
    elif bayer in ("RGGB", "BGGR", "GRBG", "GBRG"):
        for y in (0, 1):
            for x in (0, 1):
                out[y::2, x::2] /= np.float32(plane(out[y::2, x::2], bad[y::2, x::2]))
    else:
        out /= np.float32(plane(out, bad))
    out[bad] = 1.0
    return out


@nb.njit(nogil=True, cache=True, parallel=True)
def _calibrate_plane(img, dark, flat, has_dark, has_flat, out):
    """(img - dark) / flat. A sample of exactly 0 is no data and stays 0; a
    result that lands on 0 is nudged off it, so only real holes are holes."""
    H, W = img.shape
    for y in nb.prange(H):
        for x in range(W):
            v = img[y, x]
            if v == 0.0:
                out[y, x] = np.float32(0.0)
                continue
            if has_dark:
                v = np.float32(v - dark[y, x])
            if has_flat:
                v = np.float32(v / flat[y, x])
            if v == 0.0:
                v = np.float32(1e-9)
            out[y, x] = v


_EMPTY_PLANE = np.zeros((1, 1), np.float32)


def _fits(m: np.ndarray, img: np.ndarray) -> bool:
    if m.ndim == 2:
        return m.shape == img.shape[:2]
    return img.ndim == 3 and m.shape == (img.shape[2],) + img.shape[:2]


def calibrate_image(img: np.ndarray, dark, flat, colour=None) -> np.ndarray:
    """Subtract the dark and divide by the normalized flat. img: (H,W) or (H,W,C).

    A master is (H,W), one plane for every channel, or (C,H,W), one per
    channel. colour: whether the masters were fitted to frames that are colour
    before the debayer (None: not known). A master that does not fit raises
    MasterMismatch, which skips this frame only."""
    if dark is None and flat is None:
        return img
    if colour is not None and (img.ndim == 3) != bool(colour):
        raise MasterMismatch("the masters were fitted to " + ("colour" if colour else "raw single-plane")
                             + " frames and this one is " + ("colour" if img.ndim == 3 else "single-plane"))
    for m, what in ((dark, "dark"), (flat, "flat")):
        if m is not None and not _fits(m, img):
            raise MasterMismatch(f"the master {what} ({' x '.join(map(str, m.shape[::-1]))}) does not fit "
                                 f"this frame ({' x '.join(map(str, img.shape[1::-1] + img.shape[2:]))})")
    img = np.asarray(img, np.float32)
    d = _EMPTY_PLANE if dark is None else dark
    f = _EMPTY_PLANE if flat is None else flat
    planes = img[..., None] if img.ndim == 2 else img
    out = np.empty_like(planes)
    from setiastro.saspro.legacy.numba_utils import NUMBA_PARALLEL_LOCK
    with NUMBA_PARALLEL_LOCK:
        for c in range(planes.shape[2]):
            p = np.ascontiguousarray(planes[..., c])
            r = np.empty_like(p)
            _calibrate_plane(p, d if d.ndim == 2 else d[c], f if f.ndim == 2 else f[c],
                             dark is not None, flat is not None, r)
            out[..., c] = r
    return out[..., 0] if img.ndim == 2 else out


class MasterSet:
    """One run's dark and normalized flat, made to fit its frames and shared
    with the workers (never written to disk unless the system has no shared
    memory to spare). A colour master is kept channels first, (C,H,W)."""

    def __init__(self):
        self.blocks: dict = {}
        self.shapes: dict = {}
        self.colour = False                 # fitted to frames that are colour before the debayer
        self.choice = MasterChoice()        # what is really applied

    @classmethod
    def load(cls, choice: MasterChoice, paths=(), say=None) -> "MasterSet":
        """Load the chosen masters and fit them to the frames (`paths`, the
        group). One that cannot be made to fit is left out with a line to
        `say` saying why; that is never an error, the frames are then stacked
        without it. self.choice says what is applied."""
        say = say or (lambda s: None)
        self = cls()
        wanted = [(w, p) for w, p in (("dark", choice.dark), ("flat", choice.flat)) if p]
        if not wanted:
            return self
        lay = frame_layout(paths)
        if lay is None:
            say("⚠ None of the first frames could be read to check the masters against: "
                "no master was applied.")
            return self
        self.colour = lay.channels > 1
        used, how = {}, {}
        try:
            for what, path in wanted:
                name = os.path.basename(path)
                b = None
                try:
                    img, hdr = load_master(path)
                    plane, sites, how_w, notes = fit_master(img, hdr, what, lay, name)
                    del img
                    for n in notes:
                        say(n)
                    if plane is None:
                        continue
                    if what == "flat":
                        plane = normalize_flat(plane, sites)
                    if plane.ndim == 3:
                        plane = np.ascontiguousarray(np.moveaxis(plane, -1, 0))
                    b = SharedBlock.create(plane.nbytes)
                    b.view(plane.shape, np.float32)[...] = plane
                except Exception as e:
                    if b is not None:
                        b.close(unlink=True)
                    why = str(e) if isinstance(e, MasterMismatch) else \
                        f"could not be used ({type(e).__name__}: {e})"
                    say(f"⚠ The master {what} {name} {why}: not applied.")
                    continue
                self.blocks[what], self.shapes[what] = b, plane.shape
                used[what] = path
                if how_w:
                    how[what] = how_w
        except BaseException:
            self.close()
            raise
        self.choice = MasterChoice(used.get("dark"), used.get("flat"), choice.bayer, how)
        if used:
            say("Calibrating with " + " and ".join(
                f"master {w} {os.path.basename(p)}" + (f" ({how[w]})" if w in how else "")
                for w, p in used.items()) + ".")
        else:
            say("No master could be applied: these frames are stacked uncalibrated.")
        return self

    def descriptor(self) -> dict:
        return {w: {**b.descriptor(), "shape": self.shapes[w], "colour": self.colour}
                for w, b in self.blocks.items()}

    def close(self) -> None:
        for b in self.blocks.values():
            b.close(unlink=True)
        self.blocks = {}


_MASTER_CACHE: dict = {}
_MASTER_LOCK = threading.Lock()


def open_masters(desc: dict | None):
    """(dark, flat) planes of a MasterSet descriptor; attached once per process."""
    if not desc:
        return None, None
    key = tuple(sorted((w, d["name"]) for w, d in desc.items()))
    with _MASTER_LOCK:
        hit = _MASTER_CACHE.get(key)
        if hit is None:
            _drop_masters()
            arrs: dict = {}
            blocks = []
            for w, d in desc.items():
                b = SharedBlock.attach(d)
                blocks.append(b)
                arrs[w] = b.view(tuple(d["shape"]), np.float32)
            hit = _MASTER_CACHE[key] = (arrs, blocks)
        return hit[0].get("dark"), hit[0].get("flat")


def _drop_masters() -> None:
    for arrs, blocks in _MASTER_CACHE.values():
        arrs.clear()                    # a live view would keep the memory mapped
        for b in blocks:
            b.close()
    _MASTER_CACHE.clear()


def forget_masters() -> None:
    with _MASTER_LOCK:
        _drop_masters()


def is_frame_path(path: str) -> bool:
    return str(path).lower().endswith(FRAME_EXTS)


def probe_frame(path: str) -> FrameInfo:
    """Filter / exposure / size from the header, without reading pixels."""
    info = FrameInfo(path=path)
    try:
        from setiastro.saspro.blink_fast_read import probe_header
        hdr, size = probe_header(path)
    except Exception:
        return info
    if size:
        info.width, info.height = int(size[0]), int(size[1])
    elif path.lower().endswith((".tif", ".tiff")):
        with contextlib.suppress(Exception):      # header only: Pillow reads no pixels here
            from PIL import Image
            with Image.open(path) as im:
                info.width, info.height = int(im.size[0]), int(im.size[1])
    if hdr is None:
        return info

    def _get(*names, default=None):
        for n in names:
            try:
                v = hdr.get(n)
            except Exception:
                v = None
            if v not in (None, ""):
                return v
        return default

    info.filter = str(_get("FILTER", "FILTERNAME", default="Unknown")).strip() or "Unknown"
    try:
        info.exposure = float(_get("EXPTIME", "EXPOSURE", default=0.0))
    except (TypeError, ValueError):
        info.exposure = 0.0
    info.bayer = str(_get("BAYERPAT", default="") or "").strip().upper()
    info.obj = str(_get("OBJECT", default="") or "").strip()
    info.binning = str(_get("XBINNING", default="") or "").strip()
    info.roworder = str(_get("ROWORDER", default="") or "").strip().upper()
    info.imagetyp = str(_get("IMAGETYP", "FRAMETYP", default="") or "").strip()
    try:
        info.gain = float(_get("GAIN", default=0.0))
    except (TypeError, ValueError):
        info.gain = 0.0
    for attr, card in (("bitpix", "BITPIX"), ("ncombine", "NCOMBINE")):
        with contextlib.suppress(TypeError, ValueError):
            setattr(info, attr, int(float(_get(card, default=0) or 0)))
    if not (info.width and info.height):
        try:
            info.width, info.height = int(hdr.get("NAXIS1") or 0), int(hdr.get("NAXIS2") or 0)
        except Exception:
            pass
    with contextlib.suppress(Exception):
        # A colour FITS stored pixel by pixel (the pipeline's own intermediates
        # are): NAXIS1 is the channel count and the image is NAXIS2 x NAXIS3.
        if (int(hdr.get("NAXIS") or 0) == 3 and int(hdr.get("NAXIS1") or 0) in (3, 4)
                and int(hdr.get("NAXIS3") or 0) > 4):
            info.width, info.height = int(hdr["NAXIS2"]), int(hdr["NAXIS3"])
    return info


# What an added file is: Quick Stack takes lights and master darks and flats
# from the same drop, and leaves out what it cannot use instead of stacking
# it as a group of "lights".
_KIND_WORDS = {
    "flat": ("flat", "flats", "masterflat", "masterflats", "flatfield", "flatfields", "skyflat", "skyflats"),
    "dark": ("dark", "darks", "masterdark", "masterdarks"),
    "bias": ("bias", "biases", "offset", "offsets", "masterbias", "masteroffset"),
    "darkflat": ("darkflat", "darkflats", "flatdark", "flatdarks", "masterdarkflat", "masterflatdark"),
    "light": ("light", "lights", "lightframe", "lightframes"),
}
_KIND_OF_WORD = {w: k for k, ws in _KIND_WORDS.items() for w in ws}
_ONE_OF = frozenset({"flat", "dark", "bias", "offset", "light"})


def _kind_of_name(text: str, folder: bool) -> str:
    """The kind a file or folder name gives. A folder counts when it is named
    for its frames ("Flats", "Darks 300s", "Master Flats", or just "Dark");
    "Dark Nebula" is not a folder of darks. A file name counts when it starts
    with the kind, as capture programs write them ("FLAT_Ha_0001"), or says
    master ("MasterFlat_Ha")."""
    words = re.findall(r"[a-z]+", str(text).lower())
    if not words:
        return ""
    pairs = [a + b for a, b in zip(words, words[1:])]
    kinds = set()
    for w in pairs + words:
        k = _KIND_OF_WORD.get(w)
        if k is None:
            continue
        if w in _ONE_OF and not (len(words) == 1 or (not folder and words[0] == w)):
            continue
        kinds.add(k)
    if "darkflat" in kinds or {"dark", "flat"} <= kinds:
        return "darkflat"
    return kinds.pop() if len(kinds) == 1 else ""


def frame_kind(info: FrameInfo) -> str:
    """"light", "dark", "flat", "bias" or "darkflat": what the IMAGETYP card
    says, else the file's name, its folder's or the one above ("Flats/Ha/");
    "" if nothing says."""
    t = info.imagetyp.upper()
    if t:
        if "FLAT" in t and "DARK" in t:
            return "darkflat"
        for word, kind in (("FLAT", "flat"), ("DARK", "dark"), ("BIAS", "bias"), ("OFFSET", "bias"),
                           ("ZERO", "bias"), ("LIGHT", "light"), ("SCIENCE", "light"),
                           ("OBJECT", "light"), ("STACK", "light")):
            if word in t:
                return kind
    k = _kind_of_name(os.path.splitext(os.path.basename(info.path))[0], folder=False)
    d = os.path.dirname(info.path)
    for _ in range(2):
        if k:
            return k
        k = _kind_of_name(os.path.basename(d), folder=True)
        d = os.path.dirname(d)
    return k


def is_master(info: FrameInfo) -> bool:
    """Does a dark or flat say it is a master, not a single frame? MASTER on
    its IMAGETYP card, "master" in its file or folder name, NCOMBINE above 1,
    or floating-point pixels (capture programs save integers)."""
    if info.path.lower().endswith(RAW_EXTS):
        return False
    if "MASTER" in info.imagetyp.upper() or info.ncombine > 1 or info.bitpix < 0:
        return True
    names = (os.path.basename(info.path), os.path.basename(os.path.dirname(info.path)))
    return any("master" in n.lower() for n in names)


def route_file(info: FrameInfo, as_master: bool = False) -> str:
    """Where an added file goes: "light", or "dark" / "flat" (to the masters);
    else why it is left out: "dark-frames" / "flat-frames" (single frames,
    not masters), "bias" or "darkflat". as_master (Add Masters): the file was
    given as a master, so a dark or flat needs no proof of it, and one that
    says neither gives "unknown" for the window to ask about."""
    kind = frame_kind(info)
    if as_master:
        if kind in ("dark", "flat", "bias", "darkflat"):
            return kind
        return "unknown"
    if kind in ("", "light"):
        return "light"
    if kind in ("dark", "flat"):
        return kind if is_master(info) else f"{kind}-frames"
    return kind


def group_frames(infos: list) -> list:
    """Groups that can be stacked together: same filter, exposure, size, binning, CFA."""
    groups: dict = {}
    for f in infos:
        key = (f.filter.upper(), round(f.exposure, 3), f.width, f.height, f.binning, f.bayer)
        groups.setdefault(key, FrameGroup(key)).frames.append(f)
    return sorted(groups.values(), key=lambda g: g.key)


# ─────────────────────────────────────────────────────────────────────────────
# Kernels
# ─────────────────────────────────────────────────────────────────────────────
@nb.njit(nogil=True, cache=True)
def _accumulate_band(src, inv, off, y0, y1, sumv, cnt, hi0, hi1, lo, lo1):
    """Bilinear inverse-warp `src` (H,W,C) into the rows [y0, y1) of the
    accumulators. `inv` maps output pixel -> source pixel. Samples whose 2x2
    neighbourhood holds a 0.0 (no data) or anything non-finite are skipped."""
    H, W, C = src.shape
    w = sumv.shape[1]
    dx = inv[0, 0]
    dy = inv[1, 0]
    for y in range(y0, y1):
        sx = inv[0, 1] * y + inv[0, 2]
        sy = inv[1, 1] * y + inv[1, 2]
        for x in range(w):
            xs = sx + dx * x
            ys = sy + dy * x
            if xs < 0.0 or ys < 0.0:
                continue
            ix = int(xs)
            iy = int(ys)
            if ix >= W - 1 or iy >= H - 1:
                continue
            fx = np.float32(xs - ix)
            fy = np.float32(ys - iy)
            for c in range(C):
                a = src[iy, ix, c]
                b = src[iy, ix + 1, c]
                d = src[iy + 1, ix, c]
                e = src[iy + 1, ix + 1, c]
                if a == 0.0 or b == 0.0 or d == 0.0 or e == 0.0:
                    continue
                t = a + (b - a) * fx
                u = d + (e - d) * fx
                v = t + (u - t) * fy
                if not (abs(v) < 1e30):
                    continue
                r = v - off[c]
                sumv[y, x, c] += r
                cnt[y, x, c] += 1
                if r > hi0[y, x, c]:
                    hi1[y, x, c] = hi0[y, x, c]
                    hi0[y, x, c] = r
                elif r > hi1[y, x, c]:
                    hi1[y, x, c] = r
                if r < lo[y, x, c]:
                    lo1[y, x, c] = lo[y, x, c]
                    lo[y, x, c] = r
                elif r < lo1[y, x, c]:
                    lo1[y, x, c] = r


@nb.njit(nogil=True, cache=True, parallel=True)
def _finish(sumv, cnt, hi0, hi1, lo, lo1, out, min_trim, full_trim):
    H, W, C = sumv.shape
    for y in nb.prange(H):
        for x in range(W):
            for c in range(C):
                n = int(cnt[y, x, c])
                if n == 0:
                    out[y, x, c] = 0.0
                    continue
                s = np.float64(sumv[y, x, c])      # double: the trim subtracts close values
                drop = 0
                if n >= min_trim:
                    s -= np.float64(hi0[y, x, c])
                    s -= np.float64(lo[y, x, c])
                    drop = 2
                    if n >= full_trim:
                        s -= np.float64(hi1[y, x, c])
                        s -= np.float64(lo1[y, x, c])
                        drop = 4
                out[y, x, c] = s / (n - drop)


# ─────────────────────────────────────────────────────────────────────────────
# Shared accumulators
# ─────────────────────────────────────────────────────────────────────────────
def _shm_has_room(nbytes: int) -> bool:
    """Linux creates a shared block sparsely and only fails (SIGBUS) when it is
    written, so ask /dev/shm first. Elsewhere creation itself reports failure."""
    import sys
    if not sys.platform.startswith("linux"):
        return True
    try:
        v = os.statvfs("/dev/shm")
        return v.f_bavail * v.f_frsize > nbytes * 1.1
    except OSError:
        return False


class SharedArrays:
    """The accumulator set. `create()` in the parent; workers `attach()` by name.

    Shared memory when the platform has room for it, else memory-mapped temp
    files (Linux caps /dev/shm; some containers give it 64 MB)."""

    SPEC = (("sumv", np.float32), ("cnt", np.uint16), ("hi0", np.float32),
            ("hi1", np.float32), ("lo", np.float32), ("lo1", np.float32))

    def __init__(self, shape, kind="shm", names=None, tmpdir=None):
        self.shape = tuple(int(v) for v in shape)
        self.kind = kind
        self.names = names
        self.tmpdir = tmpdir
        self.arrays: dict = {}
        self._handles: list = []

    # -- parent ------------------------------------------------------------
    @classmethod
    def create(cls, shape) -> "SharedArrays":
        self = cls(shape)
        n = int(np.prod(self.shape))
        try:
            if not _shm_has_room(n * sum(np.dtype(dt).itemsize for _, dt in cls.SPEC)):
                raise OSError("not enough shared memory")
            from multiprocessing import shared_memory
            names = {}
            for name, dt in cls.SPEC:
                shm = shared_memory.SharedMemory(create=True, size=n * np.dtype(dt).itemsize)
                self._handles.append(shm)
                names[name] = shm.name
                self.arrays[name] = np.ndarray(self.shape, dtype=dt, buffer=shm.buf)
            self.names = names
        except OSError:
            self.arrays = {}
            self._release_handles(unlink=True)
            self.kind = "mm"
            self.tmpdir = tempfile.mkdtemp(prefix="sas_quickstack_")
            names = {}
            for name, dt in cls.SPEC:
                path = os.path.join(self.tmpdir, f"{name}.dat")
                self.arrays[name] = np.memmap(path, dtype=dt, mode="w+", shape=self.shape)
                names[name] = path
            self.names = names
        self.arrays["sumv"][...] = 0.0
        self.arrays["cnt"][...] = 0
        self.arrays["hi0"][...] = -np.inf
        self.arrays["hi1"][...] = -np.inf
        self.arrays["lo"][...] = np.inf
        self.arrays["lo1"][...] = np.inf
        return self

    def descriptor(self) -> dict:
        return {"shape": self.shape, "kind": self.kind, "names": self.names}

    # -- worker ------------------------------------------------------------
    @classmethod
    def attach(cls, desc: dict) -> "SharedArrays":
        self = cls(desc["shape"], desc["kind"], desc["names"])
        for name, dt in cls.SPEC:
            if self.kind == "shm":
                from multiprocessing import shared_memory
                shm = shared_memory.SharedMemory(name=self.names[name])
                self._handles.append(shm)
                self.arrays[name] = np.ndarray(self.shape, dtype=dt, buffer=shm.buf)
            else:
                self.arrays[name] = np.memmap(self.names[name], dtype=dt, mode="r+",
                                              shape=self.shape)
        return self

    def close(self, unlink: bool = False) -> None:
        self.arrays = {}
        self._release_handles(unlink)
        if unlink and self.tmpdir:
            import shutil
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def _release_handles(self, unlink: bool) -> None:
        for h in self._handles:
            with contextlib.suppress(Exception):
                h.close()
            if unlink:
                with contextlib.suppress(Exception):
                    h.unlink()
        self._handles = []


# ─────────────────────────────────────────────────────────────────────────────
# Alignment: the registration delta pass's own helpers, on an in-memory frame
# ─────────────────────────────────────────────────────────────────────────────
def align_settings() -> dict:
    """The user's alignment settings (Stellar Registration / Stacking Suite)."""
    from setiastro.saspro.star_alignment import _align_prefs
    p = _align_prefs()
    return {"det_sigma": float(p["det_sigma"]), "minarea": int(p["minarea"]),
            "limit_stars": int(p["limit_stars"]),
            "downsample": 1,           # set from the frame size by solve_downsample()
            "h_reproj": float(p["h_reproj"]), "min_fwhm": float(p["min_fwhm"]),
            "max_ellipticity": float(p["max_ellipticity"])}


def solve_downsample(h: int, w: int) -> int:
    """Binning for the solve grid. Not the user's alignment downsample: on a
    small frame (1920 x 1080, undersampled stars) that would leave a handful
    of stars, and on a large one detection is the whole per-frame cost."""
    ds = int(round((h * w / (SOLVE_GRID_MP * 1e6)) ** 0.5))
    return max(1, min(MAX_SOLVE_DOWNSAMPLE, ds))


def _sigma_ladder(det_sigma: float) -> list:
    out = []
    for f in (1.0, 0.5, 0.25):
        s = max(3.0, float(det_sigma) * f)
        if not out or s < out[-1]:
            out.append(s)
    return out


def downsample_gray(gray: np.ndarray, ds: int) -> np.ndarray:
    import cv2
    if ds <= 1:
        return np.ascontiguousarray(gray, dtype=np.float32)
    h, w = gray.shape[:2]
    return cv2.resize(np.ascontiguousarray(gray, dtype=np.float32),
                      (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else (img[..., 0] if img.shape[2] == 1 else img.mean(axis=2))


def _seed_detection() -> None:
    """SEP's deblender draws from the C runtime's rand(). Reseeding before each
    detection makes a frame's star list (and so the stack) the same on every
    run, whatever else the process detected before it."""
    with contextlib.suppress(Exception):
        import ctypes
        (ctypes.cdll.ucrtbase if os.name == "nt" else ctypes.CDLL(None)).srand(1)


def _detect(gray_ds, sigma, cfg):
    from setiastro.saspro import star_alignment as sa
    _seed_detection()
    return sa._detect_stars_uniform(gray_ds, sigma, cfg["minarea"], grid=(4, 4), max_per_cell=25,
                                    max_total=cfg["limit_stars"], min_fwhm=cfg["min_fwhm"],
                                    max_ellipticity=cfg["max_ellipticity"])


def _similarity_once(src_pts, ref_pts, cfg, last):
    """astroalign on star lists, then a RANSAC similarity (no shear). 2x3 or None."""
    import cv2
    from setiastro.saspro import astroalign
    if src_pts.shape[0] < 8 or ref_pts.shape[0] < 8:
        last.update(why=f"too few stars ({src_pts.shape[0]} in the frame, "
                        f"{ref_pts.shape[0]} in the reference)")
        return None
    try:
        _tform, (s, t) = astroalign.find_transform(
            src_pts, ref_pts, max_control_points=cfg["limit_stars"])
    except Exception as e:
        last.update(why=f"no star match ({type(e).__name__})")
        return None
    s = np.asarray(s, np.float32)
    t = np.asarray(t, np.float32)
    A, inl = cv2.estimateAffinePartial2D(s, t, method=cv2.RANSAC,
                                         ransacReprojThreshold=cfg["h_reproj"])
    if A is None:
        last.update(why="no consistent star match")
        return None
    inl = inl.ravel().astype(bool)
    avail = min(len(src_pts), len(ref_pts))
    need = max(MIN_INLIERS, int(MIN_INLIER_FRACTION * avail))
    if int(inl.sum()) < need:
        last.update(why=f"only {int(inl.sum())} of {avail} stars matched (needs {need})")
        return None
    r = (s[inl] @ A[:, :2].T + A[:, 2]) - t[inl]
    last.update(inliers=int(inl.sum()), rms=float(np.sqrt(np.mean(np.sum(r * r, axis=1)))))
    return np.asarray(A, np.float64).reshape(2, 3)


def solve_points(src_pts, ref_pts, shape, cfg):
    """Similarity taking one star list to another, on the solve grid.

    star_alignment._solve_with_meridian_check on point lists: a result near a
    quarter turn is solved again on the list turned by exactly that (the
    points np.rot90 would give), so only a small residual is left to fit.
    Returns (T_2x3, {"inliers", "rms" (solve-grid px)}) or (None, reason)."""
    from setiastro.saspro import star_alignment as sa
    last = {"why": "no stars found", "inliers": 0, "rms": float("nan")}
    src_pts = np.asarray(src_pts, np.float32)
    H, W = int(shape[0]), int(shape[1])
    T = _similarity_once(src_pts, ref_pts, cfg, last)
    if T is None:
        return None, last["why"]
    k = sa._nearest_quarter_turn(T, src_shape=(H, W))
    if k:
        M = sa._rotk_affine_2x3(k, H, W)
        turned = (src_pts.astype(np.float64) @ M[:, :2].T + M[:, 2]).astype(np.float32)
        T_resid = _similarity_once(turned, ref_pts, cfg, last)
        if T_resid is not None:
            T = sa._compose_affine_2x3(T_resid, M)
    return T, {"inliers": last["inliers"], "rms": last["rms"]}


def lift_transform(T_ds, ds, S):
    """Solve-grid similarity -> full-resolution 2x3, or (None, reason)."""
    from setiastro.saspro import star_alignment as sa
    T = sa.lift_affine_2x3_from_ds(np.asarray(T_ds, np.float64).reshape(2, 3), ds, S)
    scale = float(np.hypot(T[0, 0], T[1, 0]))
    if not 0.9 <= scale <= 1.1:
        return None, f"implausible scale ({scale:.3f})"
    return T, None


def rotation_deg(T) -> float:
    return float(np.degrees(np.arctan2(T[1][0], T[0][0])))


def solve_delta(gray_ds, ref_ds, ref_wh, cfg, ref_key=None):
    """Similarity (no shear) mapping a frame to the reference, as full-res 2x3.

    Star detection on the solve grid (SEP), then solve_points. Detection sigma
    drops when the first try cannot match (narrowband).
    Returns (T_full_2x3, info) or (None, reason)."""
    from setiastro.saspro import star_alignment as sa
    ds = int(cfg["downsample"])
    S_ref = sa._S(ds, ref_wh, (ref_ds.shape[1], ref_ds.shape[0]))
    box = (0, 0, ref_ds.shape[0], ref_ds.shape[1])
    base = np.ascontiguousarray(gray_ds, dtype=np.float32)
    why = "no stars found"
    T = info = None
    for sigma in _sigma_ladder(cfg["det_sigma"]):
        src_pts = _detect(base, sigma, cfg)
        _seed_detection()
        ref_pts = sa._ref_pts_cached(ref_ds, ref_key, box, sigma, cfg["minarea"],
                                     cfg["limit_stars"], cfg["min_fwhm"],
                                     cfg["max_ellipticity"])
        T, info = solve_points(src_pts, ref_pts, base.shape, cfg)
        if T is not None:
            break
        why = info
    if T is None:
        return None, why
    T_full, bad = lift_transform(T, ds, S_ref)
    if T_full is None:
        return None, bad
    return T_full, {"inliers": info["inliers"], "rms": info["rms"] * ds,
                    "angle": rotation_deg(T_full)}


def channel_sky(img: np.ndarray) -> np.ndarray:
    """Per-channel median of a strided sample, ignoring no-data."""
    a = img[..., None] if img.ndim == 2 else img
    sky = np.zeros(a.shape[2], np.float32)
    for c in range(a.shape[2]):
        v = a[::8, ::8, c]
        v = v[np.isfinite(v) & (v != 0.0)]
        sky[c] = float(np.median(v)) if v.size else 0.0
    return sky


# ─────────────────────────────────────────────────────────────────────────────
# Tasks for the GPU engine's CPU workers
# ─────────────────────────────────────────────────────────────────────────────
def _in_child_process() -> bool:
    import multiprocessing as mp
    return mp.parent_process() is not None


_WATCHING = threading.Event()


def die_with_parent() -> None:
    """Leave when the app goes. A pool worker otherwise waits on its queue for
    good when the app is killed: the workers hold each other's end of the pipe
    open, so it never reports end of file."""
    import multiprocessing as mp
    parent = mp.parent_process()
    if parent is None or _WATCHING.is_set():
        return
    _WATCHING.set()

    def watch():
        parent.join()
        os._exit(1)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def _limit_child_threads() -> None:
    if _in_child_process():
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        with contextlib.suppress(Exception):
            import cv2
            cv2.setNumThreads(1)
            cv2.ocl.setUseOpenCL(False)
        with contextlib.suppress(Exception):
            nb.set_num_threads(1)


class SharedBlock:
    """One shared buffer: shared memory, else a memory-mapped temp file."""

    def __init__(self, kind, name, nbytes):
        self.kind, self.name, self.nbytes = kind, name, int(nbytes)
        self._shm = None
        self._mm = None
        self.tmpdir = None

    @classmethod
    def create(cls, nbytes: int) -> "SharedBlock":
        nbytes = int(nbytes)
        try:
            if not _shm_has_room(nbytes):
                raise OSError("not enough shared memory")
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self = cls("shm", shm.name, nbytes)
            self._shm = shm
        except OSError:
            d = tempfile.mkdtemp(prefix="sas_quickstack_")
            path = os.path.join(d, "block.dat")
            self = cls("mm", path, nbytes)
            self.tmpdir = d
            self._mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(nbytes,))
        return self

    def descriptor(self) -> dict:
        return {"kind": self.kind, "name": self.name, "nbytes": self.nbytes}

    @classmethod
    def attach(cls, desc: dict) -> "SharedBlock":
        self = cls(desc["kind"], desc["name"], desc["nbytes"])
        if self.kind == "shm":
            from multiprocessing import shared_memory
            self._shm = shared_memory.SharedMemory(name=self.name)
        else:
            self._mm = np.memmap(self.name, dtype=np.uint8, mode="r+", shape=(self.nbytes,))
        return self

    def view(self, shape, dtype, offset: int = 0) -> np.ndarray:
        buf = self._shm.buf if self._shm is not None else self._mm
        return np.ndarray(shape, dtype=dtype, buffer=buf, offset=int(offset))

    def close(self, unlink: bool = False) -> None:
        if self._shm is not None:
            with contextlib.suppress(Exception):
                self._shm.close()
            if unlink:
                with contextlib.suppress(Exception):
                    self._shm.unlink()
        self._shm = self._mm = None
        if unlink and self.tmpdir:
            import shutil
            shutil.rmtree(self.tmpdir, ignore_errors=True)


BAYER = ("RGGB", "BGGR", "GRBG", "GBRG")


def quick_u16(path: str) -> bool:
    """Does this look like a plain 16-bit camera FITS? Header cards only (well
    under a millisecond), so it can be asked of every frame before a run. The
    loader asks u16_plan, which is the real test."""
    if not path.lower().endswith((".fits", ".fit", ".fts")):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(2880 * 4)
            while b"END     " not in head and len(head) < 2880 * 200:
                more = f.read(2880 * 8)
                if not more:
                    return False
                head += more
        cards = {}
        for i in range(0, len(head), 80):
            c = head[i:i + 80].decode("ascii", "replace")
            if c.startswith("END"):
                break
            if c[8:10] == "= ":
                cards[c[:8].strip()] = c[10:].split("/")[0].strip()
        return (cards.get("SIMPLE") == "T" and cards.get("BITPIX") == "16"
                and cards.get("NAXIS") == "2" and float(cards.get("BZERO", 0)) == 32768.0
                and float(cards.get("BSCALE", 1)) == 1.0 and "BLANK" not in cards)
    except Exception:
        return False


def u16_plan(path: str):
    """How to read a plain 16-bit camera FITS straight into memory, or None.

    Blink's probe decides, so the rules are the loader's: unsigned 16-bit
    (BZERO 32768), uncompressed, one plane, and the same row-order flip. What
    it declines goes through load_frame."""
    if not path.lower().endswith((".fits", ".fit", ".fts")):
        return None
    try:
        from setiastro.saspro import blink_fast_read as bf
        src = bf._probe_fits(path)
        if (src is None or src.bit_depth != "16-bit" or len(src.offsets) != 1
                or src.dtype != np.dtype(">i2") or "BLANK" in src.header):
            return None
        bayer = src.header.get("BAYERPAT")
        bayer = bayer[0] if isinstance(bayer, tuple) else bayer
        bayer = str(bayer or "").strip().upper()
        if bayer and bayer not in BAYER:
            return None
        return {"offset": int(src.offsets[0]), "h": int(src.height), "w": int(src.width),
                "bayer": bayer, "flip": bool(bf._needs_vertical_flip(src.header))}
    except Exception:
        return None


_BLOCKS: dict = {}


def _block(desc: dict) -> SharedBlock:
    """Attach once per worker; a run's block is forgotten when the next one starts."""
    b = _BLOCKS.get(desc["name"])
    if b is None:
        for old in list(_BLOCKS.values()):
            old.close()
        _BLOCKS.clear()
        b = _BLOCKS[desc["name"]] = SharedBlock.attach(desc)
    return b


def forget_blocks() -> None:
    """Unmap a finished run's slots and masters in this worker (they can be gigabytes)."""
    for b in list(_BLOCKS.values()):
        b.close()
    _BLOCKS.clear()
    forget_masters()
    time.sleep(0.05)                # hold this worker so the other calls go elsewhere


def warm_worker() -> bool:
    """Import what loading and matching need, ahead of the first frame."""
    _limit_child_threads()
    from setiastro.saspro import astroalign, star_alignment                # noqa: F401
    from setiastro.saspro import blink_fast_read, live_stack_worker        # noqa: F401
    import cv2                                                             # noqa: F401
    return True


def load_slot_job(idx, path, as_u16, desc, slot, cap, shape, nch, masters=None) -> tuple:
    """Read one frame into its slot. ("ok", idx, {"kind", "bayer"}) or ("skip", idx, why).

    as_u16: quick_u16 thought it a plain 16-bit frame. kind "u16": the raw
    mosaic or mono plane, samples stored as FITS does (value - 32768, native
    int16), which the GPU calibrates. kind "f32": (H,W,C) linear float32,
    calibrated here (`masters`)."""
    _limit_child_threads()
    try:
        H, W = int(shape[0]), int(shape[1])
        blk = _block(desc)
        base = slot * cap
        plan = u16_plan(path) if as_u16 else None
        if plan is not None:
            if (plan["h"], plan["w"]) != (H, W):
                return "skip", idx, "size differs from the reference"
            if (3 if plan["bayer"] else 1) != nch:
                return "skip", idx, "colour channels differ from the reference"
            a = blk.view((H, W), np.int16, base)
            need = H * W * 2
            with open(path, "rb") as f:
                f.seek(plan["offset"])
                if f.readinto(memoryview(a).cast("B")) != need:
                    return "skip", idx, "the pixel data is cut short"
            a.byteswap(inplace=True)
            if plan["flip"]:
                a[:] = a[::-1].copy()
            return "ok", idx, {"kind": "u16", "bayer": plan["bayer"]}
        img, why = load_frame(path, masters)
        if img is None:
            return "skip", idx, why
        if img.shape[:2] != (H, W):
            return "skip", idx, "size differs from the reference"
        if (1 if img.ndim == 2 else img.shape[2]) != nch:
            return "skip", idx, "colour channels differ from the reference"
        if cap < H * W * nch * 4:
            return "skip", idx, "could not be read as a 16-bit frame"
        blk.view((H, W, nch), np.float32, base)[...] = img[..., None] if img.ndim == 2 else img
        return "ok", idx, {"kind": "f32", "bayer": ""}
    except Exception as e:
        return "skip", idx, f"{type(e).__name__}: {e}"


def match_job(src_pts, ref_pts, shape, cfg) -> tuple:
    """solve_points in a worker: (T_2x3, info) or (None, reason)."""
    _limit_child_threads()
    try:
        return solve_points(src_pts, ref_pts, shape, cfg)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
_G: dict = {}
_THREAD_MODE_LOCK = threading.Lock()     # thread fallback keeps its state in _G


@contextlib.contextmanager
def _cpu_detection():
    """Star detection through SEP only, for code running in the app's process.
    Puts back whatever the process's GPU detection state was."""
    from setiastro.saspro import star_alignment as sa
    saved = list(sa._GPU_DETECT_SLOT)
    sa._GPU_DETECT_SLOT[:] = [None]
    try:
        yield
    finally:
        sa._GPU_DETECT_SLOT[:] = saved


def init_session_worker() -> None:
    """Pool initializer for a GpuSession's workers."""
    die_with_parent()
    _limit_child_threads()


def _worker_init(cfg: dict) -> None:
    die_with_parent()
    # Process-wide limits belong in a worker process only: the thread fallback
    # runs inside the app and must not throttle the rest of it.
    _limit_child_threads()
    if _in_child_process():
        # Star detection stays on the CPU (SEP) in these workers: no torch
        # import, and the same result on every machine.
        with contextlib.suppress(Exception):
            from setiastro.saspro import star_alignment as sa
            sa._GPU_DETECT_SLOT[:] = [None]
    _G.clear()
    _G.update(cfg)
    if cfg.get("arrays") is None:
        _G["shared"] = SharedArrays.attach(cfg["desc"])
        _G["arrays"] = _G["shared"].arrays
    # A private, writable copy: SEP rejects read-only buffers.
    _G["ref_ds"] = np.array(np.load(cfg["ref_ds_npy"]), dtype=np.float32, order="C")
    from setiastro.saspro import star_alignment as sa
    _G["ref_key"] = sa._ref_file_key(cfg["ref_ds_npy"])


def load_frame(path: str, masters: dict | None = None):
    """((H,W) or (H,W,3) float32 linear, header), or (None, reason).

    masters: a MasterSet descriptor. The frame is calibrated on its mosaic,
    before it is debayered."""
    from setiastro.saspro.live_stack_worker import load_calibrated
    dark, flat = open_masters(masters)
    colour = next((d.get("colour") for d in (masters or {}).values()), None)
    hook = (None if dark is None and flat is None
            else (lambda m: calibrate_image(m, dark, flat, colour)))
    try:
        with _quiet():
            img, header, _key, why, _notes = load_calibrated(path, debayer_method="bilinear",
                                                             mosaic_fn=hook)
    except MasterMismatch as e:
        return None, str(e)
    if img is None:
        return None, why or "could not be loaded"
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    return img, header


def _frame_job(task) -> tuple:
    """One frame, start to finish. Returns ("ok" | "skip" | "stop", path, detail)."""
    path, is_ref = task
    if _G["stop"].is_set():
        return "stop", path, None
    try:
        img, hdr = load_frame(path, _G["masters"])
        if img is None:
            return "skip", path, hdr
        if img.ndim == 3 and img.shape[:2] != _G["shape"][:2] or \
                img.ndim == 2 and img.shape != _G["shape"][:2]:
            return "skip", path, "size differs from the reference"
        chans = 1 if img.ndim == 2 else img.shape[2]
        if chans != _G["shape"][2]:
            return "skip", path, "colour channels differ from the reference"
        sky = channel_sky(img)
        angle = 0.0
        inliers, rms = 0, 0.0
        if is_ref:
            T = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        else:
            gray_ds = downsample_gray(to_gray(img), _G["cfg"]["downsample"])
            T, info = solve_delta(gray_ds, _G["ref_ds"], _G["ref_wh"], _G["cfg"],
                                  _G["ref_key"])
            if T is None:
                return "skip", path, info
            angle, inliers, rms = info["angle"], info["inliers"], info["rms"]
        if _G["stop"].is_set():
            return "stop", path, None
        off = (sky - _G["ref_sky"]).astype(np.float32)
        inv = np.linalg.inv(np.vstack([T, [0, 0, 1]]))[:2].astype(np.float64)
        src = img[..., None] if img.ndim == 2 else img
        src = np.ascontiguousarray(src)
        A = _G["arrays"]
        H = _G["shape"][0]
        order = range(N_BANDS)
        start = (os.getpid() * 7919 + threading.get_ident()) % N_BANDS   # spread contention
        for k in order:
            b = (k + start) % N_BANDS
            y0 = H * b // N_BANDS
            y1 = H * (b + 1) // N_BANDS
            with _G["locks"][b]:
                _accumulate_band(src, inv, off, y0, y1, A["sumv"], A["cnt"],
                                 A["hi0"], A["hi1"], A["lo"], A["lo1"])
        return "ok", path, {"angle": angle, "inliers": inliers, "rms": rms,
                            "offset": float(off.mean())}
    except Exception as e:
        return "skip", path, f"{type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StackResult:
    image: np.ndarray
    header: object
    used: int
    total: int
    flipped: int
    skipped: list            # (path, reason)
    rms: float
    seconds: float
    coverage_max: int
    frames: list = field(default_factory=list)   # (path, {"angle", "inliers", "rms", "offset"})
    engine: str = "CPU"


def _worker_count(shape, requested: int = 0) -> int:
    """Processes to use, limited by memory: each holds a frame plus work space."""
    n = requested or max(1, min(12, (os.cpu_count() or 4) // 2))
    px = int(np.prod(shape))
    per_worker = px * 4 * 2.5 + 600e6           # frame, gray, resample buffers, libraries
    try:
        import psutil
        avail = float(psutil.virtual_memory().available) - px * 22.0     # the accumulators
        if avail > 0:
            n = min(n, max(1, int(avail * 0.75 // per_worker)))
    except Exception:
        n = min(n, 4)
    return max(1, n)


def _executor_kind() -> str:
    """"process" when spawn pools work here, else "thread"."""
    try:
        from setiastro.saspro.worker_env import process_pools_ok
        return "process" if process_pools_ok() else "thread"
    except Exception:
        return "process"


def choose_reference(paths, ref_path, cfg, say, masters=None):
    """(image, header, path) of the frame everything is aligned to: the one
    given, else the first that loads and shows stars. Sets cfg["downsample"]."""
    # Counted with SEP, not the GPU detector: this runs in the app's process.
    for cand in ([ref_path] if ref_path else paths[:5]):
        img, hdr = load_frame(cand, masters)
        if img is None:
            say(f"Reference {os.path.basename(cand)}: {hdr}")
            continue
        cfg["downsample"] = solve_downsample(img.shape[0], img.shape[1])
        gds = downsample_gray(to_gray(img), cfg["downsample"])
        npts = 0
        with _cpu_detection():
            for sigma in _sigma_ladder(cfg["det_sigma"]):
                npts = max(npts, len(_detect(gds, sigma, cfg)))
                if npts >= 8:
                    break
        if npts >= 8:
            return img, hdr, os.path.normpath(cand)
        say(f"Reference {os.path.basename(cand)}: only {npts} stars, trying the next frame")
    raise RuntimeError("No usable reference frame: none of the first frames "
                       "could be loaded with enough stars.")


def index_of(paths, path) -> int:
    """Where `path` is in the list, or -1."""
    want = os.path.normcase(path)
    return next((i for i, p in enumerate(paths) if os.path.normcase(p) == want), -1)


def require_stack(oks, skipped, paths) -> None:
    """A stack of the reference alone is not a result."""
    if not oks or (len(oks) < 2 and len(paths) > 1):
        why = "; ".join(f"{os.path.basename(p)}: {r}" for p, r in skipped[:3])
        raise RuntimeError(f"No frame could be aligned to the reference and stacked "
                           f"({len(skipped)} of {len(paths)} skipped). {why}")


def make_result(image, ref_hdr, nch, oks, skipped, paths, cov, t0, say, engine="CPU",
                masters=None, ref_path=None) -> StackResult:
    """The result both engines return, and the closing log lines."""
    flipped = sum(1 for _, d in oks if abs(d["angle"]) > 90.0)
    rms = [d["rms"] for _, d in oks if d["rms"] > 0]
    used = len(oks)
    secs = time.perf_counter() - t0
    say(f"Stacked {used} of {len(paths)} frames on the {engine} ({len(skipped)} skipped, "
        f"{flipped} rotated about 180°) in {secs:.0f} s")
    if flipped and flipped < used:
        say("Both pier sides are in this stack. They are aligned by position, rotation and "
            "scale only, so stars can look slightly soft, most near the corners.")
    like = paths[0] if ref_path and index_of(paths, ref_path) < 0 else None
    return StackResult(image, result_header(ref_hdr, used, nch, masters, like), used, len(paths), flipped,
                       skipped, float(np.median(rms)) if rms else 0.0, secs, cov,
                       sorted(oks, key=lambda t: t[0]), engine)


def reference_job(paths, ref_path, cfg, masters=None) -> tuple:
    """choose_reference in a worker: SEP holds the GIL, so in the app's process
    it would freeze the window. Returns (shape, channels, header, path,
    solve-grid binning, log lines)."""
    _limit_child_threads()
    lines: list = []
    img, hdr, path = choose_reference(paths, ref_path, cfg, lines.append, masters)
    return (img.shape[:2], 1 if img.ndim == 2 else img.shape[2], hdr, path,
            cfg["downsample"], lines)


def stack_frames(paths, *, ref_path=None, workers: int = 0, log=None, progress=None,
                 should_stop=None, kind=None, session=None,
                 masters: MasterChoice | None = None) -> StackResult:
    """Align every frame to the reference and return the trimmed mean.

    ref_path None: the first frame. A reference that is not one of `paths` is
    aligned to but not stacked. Frames that cannot be loaded, differ in size,
    or cannot be matched are skipped and listed in the result.

    masters: what pick_masters chose. Nothing is written to disk for them.

    session: a quick_stack_gpu.GpuSession. The stack then runs on the GPU, and
    on the CPU here if the GPU turns out not to be usable."""
    say = log or (lambda s: None)
    step = progress or (lambda done, total, msg: None)
    stopped = should_stop or (lambda: False)
    paths = [os.path.normpath(p) for p in paths]
    if not paths:
        raise ValueError("No frames to stack.")
    # Loaded and fitted once, for whichever engine runs: a master that does
    # not fit is left out here, with a line in the log, and is never an error.
    mset = None
    if masters:
        step(0, 0, "Loading the masters…")
        mset = MasterSet.load(masters, paths, say)
    used = mset.choice if mset is not None and mset.blocks else None
    mdesc = mset.descriptor() if used else None
    try:
        if session is not None and kind is None:
            from setiastro.saspro import quick_stack_gpu as gpu
            try:
                return gpu.stack_frames_gpu(paths, session, ref_path=ref_path, log=say,
                                            progress=step, should_stop=stopped, masters=used,
                                            mdesc=mdesc)
            except gpu.GpuUnavailable as e:
                if stopped():                   # the window closed under it
                    raise InterruptedError("Quick Stack was cancelled.") from e
                say(f"GPU: {e}. Stacking on the CPU instead.")
        return _stack_cpu(paths, ref_path, workers, kind, say, step, stopped, used, mdesc)
    finally:
        if mset is not None:
            forget_masters()
            mset.close()


def _stack_cpu(paths, ref_path, workers, kind, say, step, stopped, masters, mdesc) -> StackResult:
    import multiprocessing as mp
    from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                    ThreadPoolExecutor, wait)
    t0 = time.perf_counter()
    cfg = align_settings()

    # Reference: what was given, else the first frame that loads and shows stars.
    ref_img, ref_hdr, ref_path = choose_reference(paths, ref_path, cfg, say, mdesc)
    ref_i = index_of(paths, ref_path)           # -1: only aligned to, not stacked
    shape = (ref_img.shape[0], ref_img.shape[1], 1 if ref_img.ndim == 2 else ref_img.shape[2])
    ref_sky = channel_sky(ref_img)
    tmpdir = tempfile.mkdtemp(prefix="sas_quickstack_ref_")
    ref_ds_npy = os.path.join(tmpdir, "ref_ds.npy")
    np.save(ref_ds_npy, downsample_gray(to_gray(ref_img), cfg["downsample"]))
    ref_wh = (shape[1], shape[0])
    del ref_img

    kind = kind or _executor_kind()
    n_workers = _worker_count(shape, workers)
    say(f"{len(paths)} frames, {shape[1]} x {shape[0]}"
        f"{' x ' + str(shape[2]) if shape[2] > 1 else ''}, reference "
        f"{os.path.basename(ref_path)}{'' if ref_i >= 0 else ' (aligned to, not stacked)'}, "
        f"{n_workers} worker {'processes' if kind == 'process' else 'threads'}")

    shared = None
    skipped, oks = [], []
    if kind == "thread":
        _THREAD_MODE_LOCK.acquire()
    try:
        shared = SharedArrays.create(shape) if kind == "process" else SharedArrays(shape)
        if kind == "thread":
            shared.arrays = {n: np.zeros(shape, dt) for n, dt in SharedArrays.SPEC}
            shared.arrays["hi0"][...] = -np.inf
            shared.arrays["hi1"][...] = -np.inf
            shared.arrays["lo"][...] = np.inf
            shared.arrays["lo1"][...] = np.inf
        if kind == "process":
            ctx = mp.get_context("spawn")
            locks = [ctx.Lock() for _ in range(N_BANDS)]
            stop_ev = ctx.Event()
        else:
            locks = [threading.Lock() for _ in range(N_BANDS)]
            stop_ev = threading.Event()
        cfgd = {"cfg": cfg, "desc": None if kind == "thread" else shared.descriptor(),
                "arrays": shared.arrays if kind == "thread" else None,
                "locks": locks, "stop": stop_ev, "ref_ds_npy": ref_ds_npy,
                "ref_wh": ref_wh, "ref_sky": ref_sky, "shape": shape, "masters": mdesc}
        if kind == "process":
            pool = ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_worker_init,
                                       initargs=(cfgd,))
        else:
            # Once, up front: run per pool thread it would empty _G under a
            # thread that is already working.
            _worker_init(cfgd)
            pool = ThreadPoolExecutor(n_workers)
        total = len(paths)
        done = 0
        step(0, total, "Starting workers…")
        try:
            pending = {pool.submit(_frame_job, (p, i == ref_i)): p for i, p in enumerate(paths)}
            while pending:
                if stopped():
                    stop_ev.set()
                    for f in pending:
                        f.cancel()
                    break
                finished, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for f in finished:
                    pending.pop(f)
                    if f.cancelled():
                        continue
                    status, path, detail = f.result()
                    done += 1
                    if status == "ok":
                        oks.append((path, detail))
                    elif status == "skip":
                        skipped.append((path, detail))
                        say(f"Skipped {os.path.basename(path)}: {detail}")
                    step(done, total, f"Stacking frame {done} of {total}…")
        finally:
            _shutdown(pool, kind, hard=stopped())
        if stopped():
            raise InterruptedError("Quick Stack was cancelled.")
        require_stack(oks, skipped, paths)
        from setiastro.saspro.legacy.numba_utils import NUMBA_PARALLEL_LOCK
        out = np.empty(shape, np.float32)
        A = shared.arrays
        with NUMBA_PARALLEL_LOCK:
            _finish(A["sumv"], A["cnt"], A["hi0"], A["hi1"], A["lo"], A["lo1"], out,
                    MIN_TRIM_FRAMES, FULL_TRIM_FRAMES)
        cov = int(A["cnt"].max())
        image = out[..., 0] if shape[2] == 1 else out
    finally:
        A = None                      # a live view would block closing the memory
        if kind == "thread":
            _G.clear()
            _THREAD_MODE_LOCK.release()
        if shared is not None:
            shared.close(unlink=(kind == "process"))
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return make_result(image, ref_hdr, shape[2], oks, skipped, paths, cov, t0, say,
                       masters=masters, ref_path=ref_path)


def _shutdown(pool, kind, hard: bool) -> None:
    if kind == "process" and hard:
        procs = list(getattr(pool, "_processes", {}).values() or [])
        pool.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            with contextlib.suppress(Exception):
                p.kill()
    else:
        pool.shutdown(wait=True, cancel_futures=True)


# What a stack's header takes from its own frames, not from a reference of another group.
_PER_GROUP_CARDS = ("FILTER", "EXPTIME", "EXPOSURE", "GAIN", "OBJECT", "DATE-OBS", "DATE-LOC",
                    "XBINNING", "YBINNING")


def result_header(ref_header, used: int, channels: int, masters=None, like: str | None = None):
    """Reference header turned into a stack header. Astrometry stays valid: the
    stack is on the reference's pixel grid.

    like: one of the stacked frames, when the reference is not one of them."""
    from astropy.io import fits
    h = ref_header.copy() if ref_header is not None else fits.Header()
    if like:
        own = None
        with contextlib.suppress(Exception):
            from setiastro.saspro.blink_fast_read import probe_header
            own = probe_header(like)[0]
        for k in _PER_GROUP_CARDS:
            with contextlib.suppress(Exception):
                if own is not None and k in own:
                    h[k] = own[k]
                elif k in h:
                    del h[k]
    for k in ("BSCALE", "BZERO", "BLANK", "DATAMIN", "DATAMAX"):
        with contextlib.suppress(Exception):
            del h[k]
    if channels > 1:
        for k in ("BAYERPAT", "XBAYROFF", "YBAYROFF", "COLORTYP"):
            with contextlib.suppress(Exception):
                del h[k]
    try:
        exp = float(h.get("EXPTIME") or h.get("EXPOSURE") or 0.0)
    except (TypeError, ValueError):
        exp = 0.0
    h["IMAGETYP"] = ("Master Light", "SASpro Quick Stack")
    h["NCOMBINE"] = (int(used), "frames stacked")
    if exp > 0:
        h["TOTALEXP"] = (float(exp * used), "total integration (s), approximate")
    h["HISTORY"] = ("SASpro Quick Stack: similarity alignment, trimmed mean, "
                    "one sky offset per frame")
    how = getattr(masters, "how", None) or {}
    cal = [f"master {w} {os.path.basename(p)}" + (f" ({how[w]})" if how.get(w) else "") for w, p in
           (("dark", masters.dark if masters else None), ("flat", masters.flat if masters else None)) if p]
    text = "Quick Stack calibration: " + (", ".join(cal) or "none")
    h["HISTORY"] = text.encode("ascii", "replace").decode("ascii")       # FITS cards are ASCII
    return h
