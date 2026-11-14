# src/setiastro/saspro/sep_tiled.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
SEP extraction helpers that keep its two hidden costs off the hot path.

1. sep.extract's cost grows faster than linearly with the objects one call
   finds (3124x2088 frame, 1M stack: 1000 objects 115 ms, 2000 280 ms, 4000
   885 ms, 8000 2944 ms), and on Windows that per-call overhead also contends
   ACROSS processes: stacking's ~3000-object previews ran ~1 s per frame and
   topped out at ~5 frames/s whether 18 or 36 workers ran them. extract_strips
   runs it on overlapping horizontal strips (410 ms at 4000, 797 ms at 8000;
   ~21 frames/s on 36 workers) and keeps each object only in the strip that
   owns its centroid. Below ~2000 objects one call is as fast or faster, so
   extract_auto() only strips a dense field.

   Equivalence, measured: on ordinary fields the strip catalogue matches the
   single call object for object, and its shape scatter (88 objects >0.1% of
   2967) is SEP's own -- the same call run twice scatters 94. In a dense
   cluster sitting on a cut, SEP's deblending is itself unstable (two
   identical calls: median ecc 0.2515 vs 0.2463) and the strip results fall
   inside that spread; the object count matched exactly from a 64 px margin
   up, where a 32 px margin was off by 1-2 of ~430.

2. The pixel stack (set_extract_pixstack) is allocated on every call, and it
   is one PROCESS-GLOBAL value that other SASpro tools rely on. Lowering it
   under a tool running in another GUI thread would make that tool overflow,
   so ensure_pixstack() only ever raises it. A strip's stack is sized to the
   strip, which cannot overflow.

Import-light (numpy only; sep is passed in), so worker processes stay cheap.
"""
from __future__ import annotations

import numpy as np


def ensure_pixstack(sep, n: int) -> None:
    """Raise SEP's process-global pixel stack to at least n. Never lowers."""
    try:
        if int(sep.get_extract_pixstack()) < int(n):
            sep.set_extract_pixstack(int(n))
    except Exception:
        pass


def pixstack_full(e: BaseException) -> bool:
    m = str(e)
    return "pixel buffer full" in m or "set_extract_pixstack" in m


# First-try pixel stack. Big enough for any ordinary frame; an overflow is
# an error, never a silent truncation, so it is retried at full size.
PIXSTACK_FIRST = 1_000_000

# Above this many pixels over threshold, strips win (~2000+ objects).
STRIP_MIN_ABOVE = 50_000


def extract_small_first(sep, sub: np.ndarray, *args, **kw):
    """sep.extract with the stack raised to a small size first and, only on
    overflow, to the whole image -- which cannot overflow. Same catalogue as
    an unlimited stack; never lowers a stack someone else raised."""
    ensure_pixstack(sep, min(PIXSTACK_FIRST, int(sub.size)))
    try:
        return sep.extract(sub, *args, **kw)
    except Exception as e:
        if not pixstack_full(e):
            raise
    ensure_pixstack(sep, int(sub.size))
    return sep.extract(sub, *args, **kw)


def extract_auto(sep, sub: np.ndarray, thresh, *, err=None, **kw):
    """One call for a sparse field, strips for a dense one (see module doc).
    ``thresh`` is relative to ``err`` when err is given, as in sep.extract."""
    level = float(thresh) * (float(err) if np.isscalar(err) and err is not None else 1.0)
    dense = (np.isscalar(err) or err is None) and         int(np.count_nonzero(sub > level)) > STRIP_MIN_ABOVE
    # Strips only pay off with a strip-sized stack. In the GUI process some
    # tool has usually raised it to 20M already, and since it is never
    # lowered, every strip would drag that whole allocation; there one call
    # costs exactly what it always did. Fresh worker processes start at
    # SEP's 300k, which is where the measurement phase runs.
    strip_px = 4 * (256 + 2 * 64) * int(sub.shape[1] if sub.ndim > 1 else 1)
    try:
        small_stack = int(sep.get_extract_pixstack()) <= strip_px
    except Exception:
        small_stack = False
    if dense and small_stack:
        return extract_strips(sep, sub, thresh, err=err, **kw)
    return extract_small_first(sep, sub, thresh, err=err, **kw)


# Catalog fields measured in rows, shifted back to full-image coordinates.
_ROW_FIELDS = ("y", "ymin", "ymax", "ypeak", "ycpeak")
# Per-pixel inputs that must be cut along with the image.
_ROW_KWARGS = ("err", "var", "mask")


def extract_strips(sep, sub: np.ndarray, thresh, *, core: int = 256,
                   margin: int = 64, edge_tol: int = 2, **kw):
    """
    sep.extract(sub, thresh, **kw) over horizontal strips, merged.

    ``sub`` is the background-SUBTRACTED image (subtract the full-frame
    background first, so every strip sees exactly the same pixels the single
    call would). Each strip is ``core`` rows plus ``margin`` above and below.
    An object is kept by the strip whose core holds its centroid.

    Equivalence with one full-frame call rests on every kept object being
    whole in its strip. A strip whose cut edge is touched (within
    ``edge_tol`` rows, which covers the 3x3 detection filter) by an object
    that also reaches into its core is redone with twice the margin, until
    none does or the strip is the whole image. Big stars, halos and blends
    therefore cost a wider strip, never a truncated measurement.
    """
    sub = np.ascontiguousarray(sub, dtype=np.float32)
    H = int(sub.shape[0])
    if H <= core + 2 * margin:
        ensure_pixstack(sep, sub.size)
        return sep.extract(sub, thresh, **kw)

    parts = []
    y0 = 0
    while y0 < H:
        y1 = min(H, y0 + core)
        m = int(margin)
        while True:
            a0 = max(0, y0 - m)
            a1 = min(H, y1 + m)
            strip = sub[a0:a1]
            skw = dict(kw)
            for k in _ROW_KWARGS:
                v = skw.get(k)
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == H:
                    skw[k] = np.ascontiguousarray(v[a0:a1])
            ensure_pixstack(sep, strip.size)
            cat = sep.extract(strip, thresh, **skw)
            if (a0 == 0 and a1 == H) or len(cat) == 0:
                break
            ymin = np.asarray(cat["ymin"]); ymax = np.asarray(cat["ymax"])
            touch = np.zeros(len(cat), dtype=bool)
            if a0 > 0:
                touch |= ymin <= edge_tol
            if a1 < H:
                touch |= ymax >= (a1 - a0 - 1 - edge_tol)
            reaches_core = (ymax >= (y0 - a0)) & (ymin < (y1 - a0))
            if not np.any(touch & reaches_core):
                break
            m *= 2
        if len(cat):
            cy = np.asarray(cat["y"]) + a0
            own = cat[(cy >= y0) & (cy < y1)].copy()
            if a0:
                for f in _ROW_FIELDS:
                    if f in own.dtype.names:
                        own[f] += a0
            parts.append(own)
        y0 = y1

    if not parts:
        return cat[:0]
    return np.concatenate(parts)
