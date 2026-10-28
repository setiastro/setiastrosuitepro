#src/setiastro/saspro/gaia_quad_index.py
#!/usr/bin/env python3
# ======================================================
#   _____      __  _ ___         __
#  / ___/___  / /_(_)   |  _____/ /__________
#  \__ \/ _ \/ __/ / /| | / ___/ __/ ___/ __ \
# ___/ /  __/ /_/ / ___ |(__  ) /_/ /  / /_/ /
#/____/\___/\__/_/_/  |_/____/\__/_/   \____/
#  SASpro Blind Solve Index
#
#  gaia_quad_index.py  —  quad-hash index layers for blind plate solving
#
#  A quad index is a geometric hash of 4-star patterns (see quad_core.py and
#  Lang et al. 2010, AJ 139, 1782). It is what lets SASpro solve a field with
#  NO RA/Dec and NO field-of-view: the pattern itself says where you pointed.
#
#  A quad index is SCALE-BOUND — which four stars are neighbours depends on the
#  angular scale you picked — so the index is a stack of LAYERS, each built at
#  one quad size. One layer reliably solves fields from about 2.5x to 8x its
#  quad size, so layers spaced a factor of 2 apart overlap comfortably.
#
#  These layers are BUILT LOCALLY from the Gaia catalogs the user already
#  downloaded (Astrometry tab, or the XP Spectrum library as a fallback).
#  Nothing extra to download. See gaia_quad_builder.py.
#
#  Layers 2-7 cover 20' to 25deg fields, need stars only to G~13.6, and total
#  about 650 MB. Everything they need lives in gaia_astro_lt15.sqlite, which is
#  part of the "Bright (G < 17)" astrometric tier.
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from setiastro.saspro.quad_core import (
    CODE_HI, CODE_LO, nbins_for_tol, probe_keys,
)

# ── file format ──────────────────────────────────────────────────────────────

SAQ_MAGIC = b"SASQ"
SAQ_VERSION = 1
HEADER_BYTES = 128
ALIGN = 64

# Codes are stored quantised to uint16 over [CODE_LO, CODE_HI] -> 2.3e-5 per
# step, far finer than any tolerance we ever apply.
_CODE_SPAN = CODE_HI - CODE_LO
_U16 = 65535.0

# RA as uint32 over [0,360), Dec as int32 over [-90,90]: 0.0003" resolution.
_RA_SCALE = 4294967296.0 / 360.0
_DEC_SCALE = 2147483647.0 / 90.0

# The bins are sized once, at the largest tolerance we will ever use. The
# per-quad tolerance (which depends on |AB| and the centroid noise) is applied
# afterwards as an exact filter, so a quad with a long baseline gets the full
# benefit of its precision without needing its own binning.
BIN_TOL = 0.010
EPOCH_JYEAR = 2027.0            # proper motions are propagated to this epoch


# ── layer definitions ────────────────────────────────────────────────────────

@dataclass
class QuadLayerDef:
    index:        int
    key:          str
    q_arcmin:     float          # nominal quad size; the band is 0.7x - 1.4x
    gmag_limit:   float
    per_cell:     int = 5
    recommended:  bool = False
    warning:      str = ""
    build_note:   str = ""
    # Measured fraction of the per_cell slots that actually fill. The
    # geometric count assumes every cell finds per_cell stars at the magnitude
    # limit, and the limits were picked so the all-sky total just covers the
    # slots -- but Gaia is clustered, so the galactic plane overflows while
    # high latitudes starve. The shortfall is worst on the finest layers,
    # where a cell either has stars or it does not.
    fill:         float = 1.0

    @property
    def q_lo_arcmin(self) -> float:
        return 0.70 * self.q_arcmin

    @property
    def q_hi_arcmin(self) -> float:
        return 1.40 * self.q_arcmin

    @property
    def cell_arcmin(self) -> float:
        return self.q_arcmin

    # Measured coverage band (see BLIND_SOLVER_DESIGN.md): one layer solves
    # fields from ~2.5x to ~8x its quad size. Quoted conservatively.
    @property
    def field_lo_arcmin(self) -> float:
        return 2.5 * self.q_arcmin

    @property
    def field_hi_arcmin(self) -> float:
        return 8.0 * self.q_arcmin

    @property
    def filename(self) -> str:
        return f"gaia_quads_L{self.index}.saq"

    @property
    def geometric_stars(self) -> int:
        """Stars if every cell filled -- an upper bound, not the outcome."""
        q_deg = self.q_arcmin / 60.0
        return int(self.per_cell * 41252.96 / (q_deg * q_deg))

    @property
    def est_stars(self) -> int:
        return int(self.fill * self.geometric_stars)

    @property
    def est_quads(self) -> int:
        return int(1.6 * self.est_stars)

    @property
    def est_bytes(self) -> int:
        return 20 * self.est_quads + 8 * self.est_stars

    @property
    def label(self) -> str:
        return f"Layer {self.index} — quads {_fmt_ang(self.q_arcmin)}"

    @property
    def field_label(self) -> str:
        return f"{_fmt_ang(self.field_lo_arcmin)} – {_fmt_ang(self.field_hi_arcmin)} fields"


def _fmt_ang(arcmin: float) -> str:
    if arcmin < 60.0:
        return f"{arcmin:.0f}′"
    return f"{arcmin / 60.0:.1f}°"


def fmt_size(nbytes: float) -> str:
    if nbytes >= 1024 ** 3:
        return f"{nbytes / 1024**3:.2f} GB"
    if nbytes >= 1024 ** 2:
        return f"{nbytes / 1024**2:.0f} MB"
    return f"{nbytes / 1024:.0f} KB"


# Ordered coarse -> fine. The G limits come from interpolating the exact row
# counts already hard-coded in gaia_database.py, for 5 index stars per cell.
# Fill factors and build times MEASURED on a real Gaia DR3 build (astrometric
# library complete to G <= 18, 4 reader processes). Layer 0 is extrapolated.
QUAD_LAYER_DEFS: List[QuadLayerDef] = [
    # Layers 8 and 9 are for camera lenses, not telescopes, and are off by
    # default because almost nobody shoots a 20-degree field. They cost
    # essentially nothing: the same G <= 8.0 scan layer 7 already does, over
    # cells four and sixteen times larger, so they finish in seconds and
    # together take about 140 KB.
    #
    # What they do NOT fix: a frame this wide is a gnomonic projection about
    # its own centre, while every index quad was projected about its own A-B
    # midpoint. The two only agree where the frame is locally flat, so matches
    # still come from near the frame centre and verification still works
    # through a similarity transform that cannot represent the warp further
    # out. These layers raise the hit rate on ultra-wide fields; they do not
    # make them reliable. See BLIND_SOLVER_DESIGN.md.
    QuadLayerDef(9, "quads_l9", 1024.0, 8.0, fill=1.0,
                 build_note="seconds",
                 warning="Ultra-wide fields only (43°+) — camera lenses under "
                         "about 20 mm. Solves at this scale are opportunistic, "
                         "not dependable."),
    QuadLayerDef(8, "quads_l8",  512.0, 8.0, fill=1.0,
                 build_note="seconds",
                 warning="Very wide fields only (21°+) — camera lenses, not "
                         "telescopes."),
    QuadLayerDef(7, "quads_l7", 256.0,  8.0, recommended=True,
                 fill=1.01, build_note="seconds"),
    QuadLayerDef(6, "quads_l6", 128.0,  8.0, recommended=True,
                 fill=0.58, build_note="seconds"),
    QuadLayerDef(5, "quads_l5",  64.0,  9.1, recommended=True,
                 fill=0.72, build_note="seconds"),
    QuadLayerDef(4, "quads_l4",  32.0, 10.5, recommended=True,
                 fill=0.65, build_note="under a minute"),
    QuadLayerDef(3, "quads_l3",  16.0, 12.0, recommended=True,
                 fill=0.59, build_note="about a minute"),
    QuadLayerDef(2, "quads_l2",   8.0, 13.6, recommended=True,
                 fill=0.52, build_note="a few minutes"),
    QuadLayerDef(1, "quads_l1",   4.0, 15.4, fill=0.47,
                 build_note="about 6 minutes",
                 warning="Only needed for fields under 20′."),
    # Layer 0 has never been built end to end; its fill is extrapolated from
    # the 0.59 / 0.52 / 0.47 trend of layers 3, 2 and 1.
    QuadLayerDef(0, "quads_l0",   2.0, 17.4, fill=0.43,
                 build_note="roughly 30–45 minutes",
                 warning="Longest build by far, needs about 8 GB of free RAM "
                         "while sorting, and the Faint (G 17–18) astrometric "
                         "tier must be installed. Only needed for fields under "
                         "10′."),
]

LAYER_BY_INDEX: Dict[int, QuadLayerDef] = {d.index: d for d in QUAD_LAYER_DEFS}


# ── library directory (mirrors get_astro_library_dir) ────────────────────────

_SETTINGS_QUAD_LIBRARY_DIR = "gaia_quad_library/custom_dir"


def _default_quad_library_dir() -> Path:
    from PyQt6.QtCore import QStandardPaths
    base = Path(QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.AppDataLocation))
    d = base / "gaia_quad_library"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_quad_library_dir() -> Path:
    """Active blind-solve index directory; honours a user override."""
    from PyQt6.QtCore import QSettings
    custom = QSettings().value(_SETTINGS_QUAD_LIBRARY_DIR, "", type=str)
    if custom:
        p = Path(custom)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass
    return _default_quad_library_dir()


def set_quad_library_dir(path: Path):
    from PyQt6.QtCore import QSettings
    QSettings().setValue(_SETTINGS_QUAD_LIBRARY_DIR, str(path))


def clear_quad_library_dir_override():
    from PyQt6.QtCore import QSettings
    QSettings().remove(_SETTINGS_QUAD_LIBRARY_DIR)


# ── writing ──────────────────────────────────────────────────────────────────

def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def quantize_codes(codes) -> np.ndarray:
    """
    Codes -> uint16 over [CODE_LO, CODE_HI]: 2.3e-5 per step, far finer than
    any tolerance we ever apply. Already-quantised input passes straight
    through, which is what lets the builder quantise at generation time.
    Holding 300M quads as float64 costs 32 bytes each and dominates peak
    memory on the finest layer; 8 bytes each is the difference between
    building and swapping.
    """
    a = np.asarray(codes)
    if a.dtype == np.uint16:
        return np.ascontiguousarray(a.reshape(-1, 4), dtype="<u2")
    q = np.clip((a.astype(np.float64) - CODE_LO) / _CODE_SPAN, 0.0, 1.0)
    return np.rint(q * _U16).astype("<u2").reshape(-1, 4)


def write_saq(path: Path,
              layer: QuadLayerDef,
              star_ra: np.ndarray,
              star_dec: np.ndarray,
              keys: np.ndarray,
              codes: np.ndarray,
              refs: np.ndarray,
              nbins: int,
              epoch: float = EPOCH_JYEAR) -> Path:
    """
    Write one layer. `keys` must already be sorted ascending, with `codes` and
    `refs` in the same order.

    star_ra/dec : float64 degrees
    keys        : uint32, sorted
    codes       : (N,4) float64 in code units
    refs        : (N,2) uint32, indices of A and B into the star table
    """
    path = Path(path)
    n_stars = len(star_ra)
    n_quads = len(keys)

    ra_u = np.clip(np.asarray(star_ra, np.float64) % 360.0 * _RA_SCALE,
                   0, 4294967295.0).astype("<u4")
    dec_i = np.clip(np.asarray(star_dec, np.float64) * _DEC_SCALE,
                    -2147483647.0, 2147483647.0).astype("<i4")

    cq = quantize_codes(codes)

    off_stars = _align(HEADER_BYTES)
    off_keys = _align(off_stars + n_stars * 8)
    off_codes = _align(off_keys + n_quads * 4)
    off_refs = _align(off_codes + n_quads * 8)
    total = _align(off_refs + n_quads * 8)

    hdr = bytearray(HEADER_BYTES)
    struct.pack_into("<4sHH", hdr, 0, SAQ_MAGIC, SAQ_VERSION, layer.index)
    struct.pack_into("<fff", hdr, 8,
                     layer.q_lo_arcmin, layer.q_hi_arcmin, layer.cell_arcmin)
    struct.pack_into("<HH", hdr, 20, layer.per_cell, nbins)
    struct.pack_into("<fffff", hdr, 24,
                     CODE_LO, CODE_HI, BIN_TOL, layer.gmag_limit, float(epoch))
    struct.pack_into("<QQ", hdr, 48, n_stars, n_quads)
    struct.pack_into("<QQQQ", hdr, 64, off_stars, off_keys, off_codes, off_refs)

    tmp = path.with_suffix(".saq.tmp")
    with open(tmp, "wb") as f:
        f.write(bytes(hdr))

        def _put(off, arr):
            f.seek(off)
            f.write(np.ascontiguousarray(arr).tobytes())

        st = np.empty(n_stars, dtype=[("ra", "<u4"), ("dec", "<i4")])
        st["ra"], st["dec"] = ra_u, dec_i
        _put(off_stars, st)
        _put(off_keys, np.asarray(keys, "<u4"))
        _put(off_codes, cq.reshape(-1, 4))
        _put(off_refs, np.asarray(refs, "<u4").reshape(-1, 2))
        f.seek(total - 1)
        f.write(b"\0")

    os.replace(tmp, path)
    return path


# ── reading ──────────────────────────────────────────────────────────────────

def read_saq_header(path: Path) -> Optional[dict]:
    """
    n_stars / n_quads / epoch without memmapping anything, so the tab can show
    what a layer ACTUALLY contains instead of an estimate, the moment it
    exists on disk.
    """
    try:
        with open(path, "rb") as f:
            hdr = f.read(HEADER_BYTES)
        if len(hdr) < HEADER_BYTES:
            return None
        magic, ver, layer_idx = struct.unpack_from("<4sHH", hdr, 0)
        if magic != SAQ_MAGIC:
            return None
        n_stars, n_quads = struct.unpack_from("<QQ", hdr, 48)
        epoch = struct.unpack_from("<f", hdr, 40)[0]
        return dict(version=int(ver), layer=int(layer_idx), n_stars=int(n_stars),
                    n_quads=int(n_quads), epoch=float(epoch))
    except Exception:
        return None


class QuadLayer:
    """
    One memmapped layer. Nothing is loaded until a query touches it, so opening
    the whole stack costs microseconds no matter how large the files are.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            hdr = f.read(HEADER_BYTES)
        magic, ver, layer_idx = struct.unpack_from("<4sHH", hdr, 0)
        if magic != SAQ_MAGIC:
            raise ValueError(f"{self.path.name}: not a SASpro quad index")
        if ver != SAQ_VERSION:
            raise ValueError(f"{self.path.name}: format version {ver}, "
                             f"expected {SAQ_VERSION}")
        self.layer_index = int(layer_idx)
        (self.q_lo_arcmin, self.q_hi_arcmin,
         self.cell_arcmin) = struct.unpack_from("<fff", hdr, 8)
        self.per_cell, self.nbins = struct.unpack_from("<HH", hdr, 20)
        (self.code_lo, self.code_hi, self.bin_tol,
         self.gmag_limit, self.epoch) = struct.unpack_from("<fffff", hdr, 24)
        self.n_stars, self.n_quads = struct.unpack_from("<QQ", hdr, 48)
        o_s, o_k, o_c, o_r = struct.unpack_from("<QQQQ", hdr, 64)

        n_s, n_q = int(self.n_stars), int(self.n_quads)
        self._stars = np.memmap(self.path, dtype=[("ra", "<u4"), ("dec", "<i4")],
                                mode="r", offset=o_s, shape=(n_s,))
        self._keys = np.memmap(self.path, dtype="<u4", mode="r",
                               offset=o_k, shape=(n_q,))
        self._codes = np.memmap(self.path, dtype="<u2", mode="r",
                                offset=o_c, shape=(n_q, 4))
        self._refs = np.memmap(self.path, dtype="<u4", mode="r",
                               offset=o_r, shape=(n_q, 2))

    @property
    def defn(self) -> Optional[QuadLayerDef]:
        return LAYER_BY_INDEX.get(self.layer_index)

    def star_radec(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        s = self._stars[np.asarray(idx, np.int64)]
        return (s["ra"].astype(np.float64) / _RA_SCALE,
                s["dec"].astype(np.float64) / _DEC_SCALE)

    def decode(self, ci) -> np.ndarray:
        return (self._codes[np.asarray(ci, np.int64)].astype(np.float64)
                / _U16) * _CODE_SPAN + CODE_LO

    def query(self, qcode: np.ndarray, tol, chunk: int = 256):
        """
        Candidate quads whose code is within `tol` of each query code
        (Chebyshev, per axis). `tol` may be a scalar or one value per query —
        per-quad tolerance is the whole point, since code precision scales as
        sigma/|AB| and a long baseline deserves a tighter box.

        Returns (qi, ci): indices into qcode and into this layer's quads.
        """
        qcode = np.atleast_2d(np.asarray(qcode, np.float64))
        m = len(qcode)
        if m == 0 or self.n_quads == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        tol = np.broadcast_to(np.asarray(tol, np.float64), (m,))

        out_q, out_c = [], []
        for s in range(0, m, chunk):
            e = min(s + chunk, m)
            sub = qcode[s:e]
            pk = probe_keys(sub, self.nbins).ravel()
            lo = np.searchsorted(self._keys, pk, "left")
            hi = np.searchsorted(self._keys, pk, "right")
            n = hi - lo
            tot = int(n.sum())
            if tot == 0:
                continue
            qi = np.repeat(np.repeat(np.arange(len(sub)), 16), n)
            ci = np.repeat(lo, n) + (np.arange(tot)
                                     - np.repeat(np.cumsum(n) - n, n))
            # a clipped probe can name the same cell twice
            pair = qi.astype(np.int64) * (int(self.n_quads) + 1) + ci
            _, u = np.unique(pair, return_index=True)
            qi, ci = qi[u], ci[u]
            d = np.abs(self.decode(ci) - sub[qi]).max(axis=1)
            keep = d <= tol[s:e][qi]
            if keep.any():
                out_q.append(qi[keep] + s)
                out_c.append(ci[keep])

        if not out_q:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return (np.concatenate(out_q).astype(np.int64),
                np.concatenate(out_c).astype(np.int64))

    def quad_ab(self, ci):
        """RA/Dec of the A and B stars of each candidate quad."""
        r = self._refs[np.asarray(ci, np.int64)]
        ra_a, dec_a = self.star_radec(r[:, 0])
        ra_b, dec_b = self.star_radec(r[:, 1])
        return ra_a, dec_a, ra_b, dec_b

    def close(self):
        """
        Release the memory maps, not merely the references to them.

        Dropping the attribute is not enough. numpy keeps the underlying mmap
        alive behind the array, and Windows refuses to delete or move a file
        while a mapping of it is open -- which is what made Remove fail with
        "[WinError 32] the process cannot access the file because it is being
        used by another process". Close the mapping explicitly.
        """
        for a in ("_stars", "_keys", "_codes", "_refs"):
            arr = getattr(self, a, None)
            mm = getattr(arr, "_mmap", None) if arr is not None else None
            try:
                delattr(self, a)
            except Exception:
                pass
            arr = None
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass


class QuadIndexLibrary:
    """All built layers in the quad library directory."""

    def __init__(self, library_dir: Optional[Path] = None):
        self._dir = Path(library_dir) if library_dir else get_quad_library_dir()
        self._layers: Dict[int, QuadLayer] = {}
        self._open()

    def _open(self):
        for d in QUAD_LAYER_DEFS:
            p = self._dir / d.filename
            if d.index in self._layers or not p.exists():
                continue
            try:
                self._layers[d.index] = QuadLayer(p)
            except Exception as e:
                print(f"[QuadIndexLibrary] could not open {p.name}: {e}")

    def refresh(self):
        self.close()
        try:
            self._dir = get_quad_library_dir()
        except Exception:
            pass
        self._open()

    def close(self):
        for l in self._layers.values():
            l.close()
        self._layers.clear()

    @property
    def directory(self) -> Path:
        return self._dir

    def is_built(self, index: int) -> bool:
        return (self._dir / LAYER_BY_INDEX[index].filename).exists()

    def installed(self) -> List[QuadLayer]:
        return [self._layers[i] for i in sorted(self._layers)]

    def any_installed(self) -> bool:
        return bool(self._layers)

    def layers_for_field(self, field_arcmin: Optional[float]) -> List[QuadLayer]:
        """
        Layers to try, best first.

        With a field size (from the header pixel scale, which
        _estimate_scale_arcsec_from_header already recovers for most files)
        this is usually one or two layers. Without one it is every layer,
        coarsest first — the coarse layers are megabytes and milliseconds, so
        the cheap guesses get made before the expensive ones.
        """
        got = self.installed()
        if not got:
            return []
        if field_arcmin is None or not np.isfinite(field_arcmin) or field_arcmin <= 0:
            return sorted(got, key=lambda l: -l.q_lo_arcmin)

        def rank(l: QuadLayer):
            q = 0.5 * (l.q_lo_arcmin + l.q_hi_arcmin)
            ratio = field_arcmin / max(q, 1e-9)
            inband = 2.0 <= ratio <= 10.0
            return (0 if inband else 1, abs(math.log(ratio / 4.5)))

        return sorted(got, key=rank)


_quad_library: Optional[QuadIndexLibrary] = None


def get_quad_library() -> QuadIndexLibrary:
    global _quad_library
    if _quad_library is None:
        _quad_library = QuadIndexLibrary()
    return _quad_library


def refresh_quad_library():
    """Re-read the directory. Reopens every layer, so never call this
    immediately BEFORE deleting or moving a layer file -- see
    close_quad_library()."""
    global _quad_library
    if _quad_library is not None:
        _quad_library.close()
    _quad_library = QuadIndexLibrary()


def close_quad_library():
    """
    Close the library and leave it closed.

    This is the one to call before deleting or moving a .saq: refresh()
    closes and then immediately reopens, which re-maps the very file you are
    about to touch. The next get_quad_library() reopens lazily.
    """
    global _quad_library
    if _quad_library is not None:
        _quad_library.close()
    _quad_library = None