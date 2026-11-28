# src/setiastro/saspro/planetary_metadata.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Observation metadata for Planetary Stacker results.

The stacker used to hand back bare pixels: no header, and a name of
``<source>_stack``. This module collects what the capture recorded and turns
it into a FITS header plus a WinJUPOS-style name for the stacked image.

Sources, best first
-------------------
- SER: the Observer / Instrument / Telescope strings, the per-frame
  timestamp trailer (UTC) and the header's DateTime_UTC (capture start).
- FITS sequences: the first frame's observational cards, and DATE-OBS of the
  first and last frames.
- AVI: the RIFF header's IDIT (capture start, capture PC clock), ISFT
  (capture software) and IART chunks.
- The file name: a WinJUPOS stamp, the target, and SharpCap's
  ``<date>/<target>/HH_MM_SS`` folders (via lunar_solver.capture_time).

Times
-----
DATE-OBS / DATE-END span the whole capture. DATE-AVG is the stack's epoch: the
quality-weighted mean time of the frames that went into it, or the derotation
reference frame's time when planetary derotation ran (every frame is warped to
that instant). DATE is when the stack was made, never the capture, as in the
Stacking Suite. When the file only has the capture PC's local clock (AVI IDIT,
a SER without DateTime_UTC) it goes in DATE-LOC, as N.I.N.A. writes it: the
UTC offset is not in the file, so it is never passed off as DATE-OBS.

WinJUPOS names
--------------
WinJUPOS reads the time from a ``YYYY-MM-DD-HHMM_T`` stamp in the file name:
UTC, T = tenths of a minute (``2105_9`` = 21:05:54). The stack's stamp comes
from its epoch. The rest of the source name (observer, filter, target as the
capture program wrote them) is kept as is: nothing here invents field order.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
import unicodedata
import warnings
from typing import Any, Optional

import numpy as np

_UTC = _dt.timezone.utc
_EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_UTC)

# SER dates are .NET ticks: 100 ns intervals since 0001-01-01. DateTime_UTC and
# the frame-timestamp trailer are UTC; DateTime is the capture PC's local time.
_TICKS_PER_SEC = 10_000_000
_UNIX_EPOCH_TICKS = 621_355_968_000_000_000
# Outside this window a date is an unset or garbage field, not a capture.
_MIN_UNIX = (_dt.datetime(1990, 1, 1, tzinfo=_UTC) - _EPOCH).total_seconds()
_MAX_UNIX = (_dt.datetime(2100, 1, 1, tzinfo=_UTC) - _EPOCH).total_seconds()

# WinJUPOS stamp: YYYY-MM-DD-HHMM_T (UTC, T = tenths of a minute).
WINJUPOS_STAMP_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})_(\d)(?!\d)")

# Observational cards carried from a FITS sequence's first frame. Geometry,
# Bayer and WCS cards are deliberately absent: the stack is debayered,
# cropped, maybe drizzled, so they would describe a different image.
_CARRY_KEYS = (
    "OBJECT", "OBSERVER", "TELESCOP", "INSTRUME", "FILTER",
    "EXPTIME", "EXPOSURE", "GAIN", "OFFSET", "CCD-TEMP",
    "FOCALLEN", "APTDIA", "SITELAT", "SITELONG", "SITEELEV", "SWCREATE",
)

_FITS_EXTS = (".fit", ".fits", ".fts", ".fz", ".fits.gz", ".fit.gz")

# Whole-word target names in file and folder names.
_TARGET_TOKENS = {
    "jupiter": "Jupiter", "jup": "Jupiter",
    "saturn": "Saturn", "sat": "Saturn",
    "mars": "Mars", "venus": "Venus", "mercury": "Mercury",
    "uranus": "Uranus", "neptune": "Neptune",
    "moon": "Moon", "luna": "Moon", "sun": "Sun",
}


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------

def ser_ticks_to_unix(ticks) -> np.ndarray:
    """SER .NET ticks -> Unix seconds (float64); NaN where unset or implausible."""
    t = np.atleast_1d(np.asarray(ticks, dtype=np.int64))
    with np.errstate(over="ignore"):
        out = (t - _UNIX_EPOCH_TICKS).astype(np.float64) / _TICKS_PER_SEC
    out[(t <= 0) | ~(out >= _MIN_UNIX) | ~(out <= _MAX_UNIX)] = np.nan
    return out


def _from_unix(s: float) -> _dt.datetime:
    return _EPOCH + _dt.timedelta(seconds=float(s))


def _to_unix(when: _dt.datetime) -> float:
    """Aware datetime, or naive datetime taken as UTC -> Unix seconds."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=_UTC)
    return (when - _EPOCH).total_seconds()


def fits_time(when: _dt.datetime) -> str:
    """Datetime (naive = UTC) -> FITS 'YYYY-MM-DDThh:mm:ss.sss' in UTC."""
    if when.tzinfo is not None:
        when = when.astimezone(_UTC)
    return when.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def winjupos_stamp(when: _dt.datetime) -> str:
    """Datetime (naive = UTC) -> 'YYYY-MM-DD-HHMM_T', nearest 0.1 minute."""
    if when.tzinfo is not None:
        when = when.astimezone(_UTC).replace(tzinfo=None)
    day = when.replace(hour=0, minute=0, second=0, microsecond=0)
    tenths = math.floor((when - day).total_seconds() / 6.0 + 0.5)
    t = day + _dt.timedelta(seconds=6 * tenths)      # carries past midnight
    return "%s_%d" % (t.strftime("%Y-%m-%d-%H%M"), t.second // 6)


def parse_winjupos_stamp(text) -> Optional[_dt.datetime]:
    """First WinJUPOS stamp in ``text`` -> aware UTC datetime, else None."""
    m = WINJUPOS_STAMP_RE.search(str(text or ""))
    if not m:
        return None
    y, mo, d, hh, mi, tenth = (int(g) for g in m.groups())
    try:
        return _dt.datetime(y, mo, d, hh, mi, tzinfo=_UTC) + _dt.timedelta(seconds=6 * tenth)
    except ValueError:
        return None


def strip_winjupos_stamp(name: str) -> str:
    """``name`` without its WinJUPOS stamp and the separators around it."""
    name = name or ""
    m = WINJUPOS_STAMP_RE.search(name)
    if not m:
        return name
    head = name[:m.start()].rstrip("-_ ")
    tail = name[m.end():].lstrip("-_ ")
    return "_".join(p for p in (head, tail) if p)


_ISO_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?)?")
_HMS_RE = re.compile(r"(\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?")


def _header_get(h, key):
    """Card value from a fits.Header or an XISF-style dict header."""
    if h is None:
        return None
    try:
        v = h.get(key) if hasattr(h, "get") else h[key]
    except Exception:
        return None
    if isinstance(v, dict):
        v = v.get("value")
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
        if isinstance(v, dict):
            v = v.get("value")
    return v


def _header_comment(h, key) -> str:
    try:
        return str(h.comments[key])                  # fits.Header
    except Exception:
        pass
    try:
        v = h.get(key)
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        if isinstance(v, dict):
            return str(v.get("comment") or "")
    except Exception:
        pass
    return ""


def header_time(h) -> Optional[float]:
    """DATE-OBS of one frame's header -> Unix seconds (UTC), else None."""
    raw = _header_get(h, "DATE-OBS") or _header_get(h, "DATE-BEG")
    if raw is None:
        return None
    s = str(raw).strip()
    m = _ISO_RE.match(s)
    if not m:
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        t = _dt.datetime(y, mo, d, tzinfo=_UTC)
        if m.group(4) is not None:
            t += _dt.timedelta(hours=int(m.group(4)), minutes=int(m.group(5)),
                               seconds=int(m.group(6) or 0) + float(m.group(7) or 0.0))
        else:
            # Date-only DATE-OBS: the time sits in its own card.
            tv = _header_get(h, "TIME-OBS") or _header_get(h, "UT") or _header_get(h, "UTSTART")
            tm = _HMS_RE.match(str(tv).strip()) if tv else None
            if tm:
                t += _dt.timedelta(hours=int(tm.group(1)), minutes=int(tm.group(2)),
                                   seconds=int(tm.group(3) or 0) + float(tm.group(4) or 0.0))
    except (ValueError, OverflowError):
        return None
    u = _to_unix(t)
    return u if _MIN_UNIX <= u <= _MAX_UNIX else None


def _drop_outliers(t: np.ndarray) -> np.ndarray:
    """NaN any frame time more than 12 h from the median: no capture runs
    that long, so it is a corrupt trailer entry, not a frame."""
    t = np.array(t, dtype=np.float64)
    fin = np.isfinite(t)
    if fin.any():
        med = float(np.median(t[fin]))
        t[fin & (np.abs(t - med) > 12 * 3600.0)] = np.nan
    return t


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

def clean_text(value) -> str:
    """Printable ASCII (FITS strings must be ASCII); '' for placeholder fill."""
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = "".join(ch for ch in s if ch.isprintable()).strip()
    # A field filled with one repeated character ('aaaa...', '????') is a
    # writer's placeholder, not a name.
    if len(s) >= 4 and len(set(s)) == 1:
        return ""
    return s


def target_from_path(path: str) -> str:
    """Planet, Moon or Sun named in the file name, else in the two folders
    above it (SharpCap captures to <date>/<target>/)."""
    if not path:
        return ""
    p = os.path.normpath(str(path))
    names = [os.path.splitext(os.path.basename(p))[0]]
    parent = os.path.dirname(p)
    for _ in range(2):
        base = os.path.basename(parent)
        if not base:
            break
        names.append(base)
        parent = os.path.dirname(parent)
    for name in names:
        for tok in re.split(r"[^a-z]+", name.lower()):
            if tok in _TARGET_TOKENS:
                return _TARGET_TOKENS[tok]
    return ""


def _source_path(source, src) -> str:
    if isinstance(source, str) and source.strip():
        return source.strip()
    if isinstance(source, (list, tuple)) and source and isinstance(source[0], str):
        return source[0]
    p = getattr(src, "path", None) or getattr(getattr(src, "meta", None), "path", None)
    return str(p) if p else ""


def _is_fits_path(path: str) -> bool:
    return str(path).lower().endswith(_FITS_EXTS)


def _read_fits_header(path: str):
    if not path or not _is_fits_path(path):
        return None
    try:
        from astropy.io import fits
        return fits.getheader(path)
    except Exception:
        return None


_MONTHS = {m: i + 1 for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}


def parse_local_date(text) -> Optional[_dt.datetime]:
    """A capture-software date string -> naive datetime (as written, no zone).

    ctime style 'Wed Nov 29 19:56:12 2006' (VfW/AVICap IDIT) is parsed by hand,
    not with strptime, whose %a/%b follow the process locale. Numeric forms:
    'YYYY:MM:DD hh:mm:ss', 'YYYY-MM-DD hh:mm:ss', with '-', ':', '/' or 'T'.
    """
    s = " ".join(str(text or "").replace("\0", " ").split())
    m = re.search(r"([A-Za-z]{3})[a-z]*\.? (\d{1,2}) (\d{1,2}):(\d{2}):(\d{2}) (\d{4})", s)
    if m and m.group(1).lower() in _MONTHS:
        parts = (int(m.group(6)), _MONTHS[m.group(1).lower()], int(m.group(2)),
                 int(m.group(3)), int(m.group(4)), int(m.group(5)))
    else:
        m = re.search(r"(\d{4})[-:/](\d{2})[-:/](\d{2})[ T](\d{2}):(\d{2}):(\d{2})", s)
        if not m:
            return None
        parts = tuple(int(g) for g in m.groups())
    try:
        when = _dt.datetime(*parts)
    except ValueError:
        return None
    return when if _MIN_UNIX <= _to_unix(when) < _MAX_UNIX else None


def avi_info(path: str, max_bytes: int = 1 << 20) -> dict:
    """
    Text chunks and the frame interval from an AVI's RIFF header.

    Walks the header lists before the 'movi' data (within max_bytes): IDIT
    (capture start on the capture PC's clock), ICRD, ISFT (capture software),
    IART, INAM, ICMT, and avih's microseconds per frame. {} for anything that
    is not a RIFF AVI (MP4/MOV/MKV also open through the AVI reader).
    """
    try:
        with open(path, "rb") as f:
            b = f.read(int(max_bytes))
    except (OSError, TypeError, ValueError):
        return {}
    if len(b) < 12 or b[0:4] != b"RIFF" or b[8:12] != b"AVI ":
        return {}
    out: dict = {}

    def walk(off: int, end: int, depth: int) -> None:
        while off + 8 <= end and depth < 4:
            cid = b[off:off + 4]
            size = int.from_bytes(b[off + 4:off + 8], "little")
            body = off + 8
            if cid in (b"LIST", b"RIFF"):
                if b[body:body + 4] == b"movi":
                    return
                walk(body + 4, min(end, body + size), depth + 1)
            elif cid == b"avih" and size >= 4:
                out["us_per_frame"] = int.from_bytes(b[body:body + 4], "little")
            elif cid in (b"IDIT", b"ICRD", b"ISFT", b"IART", b"INAM", b"ICMT"):
                raw = b[body:body + min(size, 512)].split(b"\0", 1)[0]
                out[cid.decode("ascii")] = raw.decode("latin-1").strip()
            off = body + size + (size & 1)

    walk(12, len(b), 0)
    return out


def _card_value(v):
    """A header value as a plain JSON-safe scalar, or None to skip it."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        return clean_text(v) or None
    return None


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def collect_observation(src, source=None, keep_idx=None, *,
                        weights: Optional[dict] = None,
                        epoch_frame: Optional[int] = None) -> dict:
    """
    Capture metadata for one stack, read while ``src`` is open.

    ``keep_idx`` are the stacked frame indices, ``weights`` their quality
    weights ({index: weight}), ``epoch_frame`` the frame every other frame was
    derotated to (None when no derotation ran). Returns a plain, JSON-safe
    dict; missing metadata is recorded in ``notes``, never raised.
    """
    meta = getattr(src, "meta", None)
    n = int(getattr(meta, "frames", 0) or 0)
    path = _source_path(source, src)
    files = list(getattr(meta, "file_list", None) or [])
    keep = np.arange(n, dtype=np.int64) if keep_idx is None else np.asarray(keep_idx, dtype=np.int64).ravel()
    keep = keep[(keep >= 0) & (keep < n)]

    notes: list[str] = []
    cards: dict[str, tuple[Any, str]] = {}
    times = None              # per-frame Unix seconds, NaN = unknown
    times_src = ""
    start = None              # capture start, when per-frame times are missing
    start_src = ""
    local_start = None        # capture start on the capture PC's clock (naive), zone unknown
    local_src = ""

    if getattr(meta, "observer", None) is not None:              # SER
        placeholder = []
        for key, attr, label in (("OBSERVER", "observer", "Observer"),
                                 ("INSTRUME", "instrument", "Instrument"),
                                 ("TELESCOP", "telescope", "Telescope")):
            raw = str(getattr(meta, attr, "") or "")
            val = clean_text(raw)
            if val:
                cards[key] = (val[:68], "From the SER header")
            elif raw.strip():
                placeholder.append(label)
        if placeholder:
            notes.append("SER header %s: placeholder text, not used." % "/".join(placeholder))

        ticks = None
        if getattr(meta, "has_timestamps", False) and hasattr(src, "timestamps_ticks"):
            try:
                ticks = src.timestamps_ticks()
            except Exception:
                ticks = None
        if ticks is not None and len(ticks) == n and n > 0:
            t = _drop_outliers(ser_ticks_to_unix(ticks))
            if np.isfinite(t).any():
                times, times_src = t, "SER frame timestamps"

        utc = ser_ticks_to_unix([int(getattr(meta, "date_utc_ticks", 0) or 0)])[0]
        loc = ser_ticks_to_unix([int(getattr(meta, "date_ticks", 0) or 0)])[0]
        if np.isfinite(utc):
            start, start_src = float(utc), "SER header start time"
        elif np.isfinite(loc):
            # Ticks of the local wall clock: read them as a naive time.
            local_start = _from_unix(float(loc)).replace(tzinfo=None)
            local_src = "SER header DateTime (capture PC clock)"

    elif files:                                                  # image sequence
        first_hdr = None
        try:
            first_hdr = (getattr(src, "_headers", None) or {}).get(0)
        except Exception:
            first_hdr = None
        if first_hdr is None:
            first_hdr = _read_fits_header(files[0])
        if first_hdr is not None:
            for key in _CARRY_KEYS:
                v = _card_value(_header_get(first_hdr, key))
                if v is not None:
                    cards[key] = (v, clean_text(_header_comment(first_hdr, key)) or "From the first frame")
        t0 = header_time(first_hdr)
        tN = header_time(_read_fits_header(files[-1])) if len(files) > 1 else None
        if t0 is not None and tN is not None and n > 1 and tN >= t0:
            # Constant frame rate assumed, as the derotation code does.
            times = np.linspace(t0, tN, n)
            times_src = "first and last frame DATE-OBS, by frame index"
        elif t0 is not None:
            start, start_src = t0, "first frame DATE-OBS"

    elif path.lower().endswith(".avi"):                          # AVI (RIFF)
        info = avi_info(path)
        art = clean_text(info.get("IART", ""))
        if art:
            cards["OBSERVER"] = (art[:68], "From the AVI header (IART)")
        sw = clean_text(info.get("ISFT", ""))
        if sw:
            # 'C:\K3CCDTools\K3CCDTools.exe -AVICAP32- ... Version: 5.0' -> 'K3CCDTools'
            exe = re.match(r"^(?:.*[\\/])?([^\\/]+?)\.exe\b", sw, flags=re.IGNORECASE)
            cards["SWCREATE"] = ((exe.group(1) if exe else sw)[:47], "Capture software (AVI header)")
        for key, label in (("IDIT", "AVI IDIT chunk (capture PC clock)"),
                           ("ICRD", "AVI ICRD chunk (capture PC clock)")):
            t_loc = parse_local_date(info.get(key, ""))
            if t_loc is not None:
                local_start, local_src = t_loc, label
                break

    # The file name: a WinJUPOS stamp is the capture program's own time.
    name_time = None
    stamp_dt = parse_winjupos_stamp(os.path.basename(path)) if path else None
    if stamp_dt is not None:
        name_time = _to_unix(stamp_dt)
    elif times is None and start is None and path:
        try:   # SharpCap folders, EXIF; same rules Lunar Relief uses
            from setiastro.saspro.lunar_solver import capture_time
            t_cap, how = capture_time({}, path)
            if t_cap is not None:
                start, start_src = _to_unix(t_cap), str(how).rstrip(".")
        except Exception:
            pass

    # --- dates ---------------------------------------------------------------
    date_obs = date_end = date_avg = None
    avg_src = ""
    if times is not None:
        fin = times[np.isfinite(times)]
        date_obs, date_end = float(fin.min()), float(fin.max())
        kt = times[keep]
        ok = np.isfinite(kt)
        if (epoch_frame is not None and 0 <= int(epoch_frame) < n
                and np.isfinite(times[int(epoch_frame)])):
            date_avg = float(times[int(epoch_frame)])
            avg_src = "time of frame %d, the derotation reference" % int(epoch_frame)
        elif ok.any():
            w = np.array([max(float((weights or {}).get(int(i), 1.0)), 0.0) for i in keep[ok]])
            if not (w.sum() > 0):
                w = np.ones_like(w)
            base = float(kt[ok].min())
            date_avg = base + float(np.sum(w * (kt[ok] - base)) / np.sum(w))
            avg_src = "quality-weighted mean time of the %d stacked frames" % int(ok.sum())
    else:
        date_obs = start
        if name_time is not None:
            date_avg = name_time
            avg_src = "the source file name"

    # --- WinJUPOS stamp ------------------------------------------------------
    stamp = ""
    when = date_avg if date_avg is not None else date_obs
    if when is not None:
        stamp = winjupos_stamp(_from_unix(when))

    # A local-clock start is kept only when no UTC time is known: it is a
    # record of what the file says, never turned into DATE-OBS here, because
    # the capture computer's UTC offset is not in the file.
    date_loc = local_start if date_obs is None and date_avg is None else None

    obj = cards.get("OBJECT", ("", ""))[0] or target_from_path(path)

    # --- log lines (English, like every SASpro log) ----------------------------
    if date_obs is not None and date_end is not None:
        notes.append("Capture %s to %s UTC (%s)." % (
            fits_time(_from_unix(date_obs))[:19].replace("T", " "),
            fits_time(_from_unix(date_end))[11:19], times_src))
    elif date_obs is not None:
        notes.append("Capture start %s UTC (%s)." % (
            fits_time(_from_unix(date_obs))[:19].replace("T", " "), start_src))
    if date_avg is not None:
        notes.append("Stack time %s UTC: %s." % (fits_time(_from_unix(date_avg))[11:21], avg_src))
    if stamp:
        if date_avg is None:
            notes.append("WinJUPOS stamp %s is the capture START: there are no frame "
                         "timestamps, so the mid-time is unknown." % stamp)
        else:
            notes.append("WinJUPOS stamp %s." % stamp)
        if stamp_dt is not None and winjupos_stamp(stamp_dt) != stamp:
            notes.append("The source name's stamp %s is replaced by the stack time."
                         % winjupos_stamp(stamp_dt))
    elif date_loc is not None:
        notes.append("Capture start %s on the capture computer's clock (%s). Its UTC offset "
                     "is not in the file, so it is written as DATE-LOC, not DATE-OBS, and the "
                     "stack keeps its source name." % (fits_time(date_loc)[:19].replace("T", " "), local_src))
    else:
        notes.append("No capture time found (no SER date or frame timestamps, no time in "
                     "the file name); the stack keeps its source name.")

    return {
        "source_path": path,
        "source_name": os.path.basename(path) if path else "",
        "frames_total": n,
        "frames_kept": int(keep.size),
        "object": obj,
        "cards": [[k, v, c] for k, (v, c) in cards.items()],
        "date_obs": fits_time(_from_unix(date_obs)) if date_obs is not None else "",
        "date_end": fits_time(_from_unix(date_end)) if date_end is not None else "",
        "date_avg": fits_time(_from_unix(date_avg)) if date_avg is not None else "",
        "date_loc": fits_time(date_loc) if date_loc is not None else "",
        "date_loc_src": local_src if date_loc is not None else "",
        "times_src": times_src or start_src,
        "date_avg_src": avg_src,
        "winjupos": stamp,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def stack_title(observation: Optional[dict], base: str, suffix: str = "_stack") -> str:
    """Document / file name for a stack: the source name re-stamped with the
    stack's WinJUPOS time (stamp first), plus ``suffix``. Without a capture
    time it is the source name plus ``suffix``, as before."""
    base = (base or "").strip() or "Stack"
    stamp = (observation or {}).get("winjupos") or ""
    if stamp:
        rest = strip_winjupos_stamp(base)
        base = "%s-%s" % (stamp, rest) if rest else stamp
    if suffix and not base.lower().endswith(suffix.lower()):
        base = base + suffix
    return base


def stack_header(observation: Optional[dict], diag: Optional[dict] = None):
    """FITS header for the stacked image (astropy fits.Header)."""
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyWarning

    obs = observation or {}
    d = diag or {}
    h = fits.Header()
    with warnings.catch_warnings():
        # Long SER strings plus a comment can overflow 80 columns; astropy
        # then trims the comment, which is all we want.
        warnings.simplefilter("ignore", VerifyWarning)

        obj = clean_text(obs.get("object") or "")
        if obj:
            from_header = any(c[0] == "OBJECT" for c in obs.get("cards") or ())
            h["OBJECT"] = (obj, "From the first frame" if from_header else "Target, from the file name")
        for item in obs.get("cards") or ():
            try:
                key, value, comment = item
            except (TypeError, ValueError):
                continue
            if key == "OBJECT":
                continue
            try:
                h[str(key)] = (value, str(comment or ""))
            except Exception:
                continue

        if obs.get("date_obs"):
            h["DATE-OBS"] = (obs["date_obs"], "Capture start (UTC)")
        if obs.get("date_end"):
            h["DATE-END"] = (obs["date_end"], "Capture end (UTC)")
        if obs.get("date_avg"):
            h["DATE-AVG"] = (obs["date_avg"], "Stack epoch (UTC), see HISTORY")
        if obs.get("date_loc"):
            h["DATE-LOC"] = (obs["date_loc"], "Local capture start; UTC offset unknown")
        if obs.get("date_obs") or obs.get("date_avg"):
            h["TIMESYS"] = ("UTC", "Capture times are UTC")
        h["DATE"] = (fits_time(_dt.datetime.now(_UTC)), "Stack creation date (NOT acquisition)")

        kept = d.get("frames_kept", obs.get("frames_kept"))
        total = d.get("frames_total", obs.get("frames_total"))
        if kept:
            h["NCOMBINE"] = (int(kept), "Frames stacked")

        h.add_history("SASpro Planetary Stacker: %s of %s frames" % (kept, total))
        src = clean_text(obs.get("source_name") or "")
        if src:
            h.add_history("Source: %s" % src)
        track = d.get("track_mode")
        if track:
            try:
                h.add_history("Tracking %s, drizzle %gx" % (track, float(d.get("drizzle_scale", 1.0))))
            except Exception:
                h.add_history("Tracking %s" % track)
        if obs.get("times_src") and obs.get("date_obs"):
            h.add_history("Capture times: %s" % obs["times_src"])
        if obs.get("date_avg_src") and obs.get("date_avg"):
            h.add_history("DATE-AVG: %s" % obs["date_avg_src"])
        if obs.get("date_loc_src") and obs.get("date_loc"):
            h.add_history("DATE-LOC: %s" % obs["date_loc_src"])
    return h
