#src/setiastro/saspro/gaia_quad_builder.py
#!/usr/bin/env python3
# ======================================================
#  gaia_quad_builder.py  —  build blind-solve quad layers LOCALLY
#
#  Builds the quad-hash index described in gaia_quad_index.py straight from the
#  Gaia catalogs the user already downloaded. Nothing extra comes off the wire.
#
#  Source priority:
#    1. Astrometric library  (gaia_astro_*.sqlite)  — preferred, far denser.
#       Layers 2-7 need only G <= 13.6, which lives entirely inside
#       gaia_astro_lt15.sqlite, one file of the "Bright (G < 17)" tier.
#    2. XP spectral library  (gaia_xp_*.sqlite)     — fallback. Reaches
#       G <= 12 with the Ultra-Bright + Bright + Medium tiers, so it can build
#       layers 3-7 (fields 40' and wider) with no astrometric data at all.
#
#  ── READ PATH, AND WHY IT LOOKS LIKE THIS ───────────────────────────────────
#
#  The obvious design — walk the sky in declination stripes, querying
#  `WHERE dec_zone BETWEEN ? AND ?` — is a trap. The distributed tables are
#  WITHOUT ROWID with source_id as the primary key, so ix_zone is a SECONDARY
#  index: SQLite answers a dec_zone query with "SEARCH stars USING INDEX
#  ix_zone", which means a primary-key B-tree lookup for EVERY matching row.
#  That is random access scattered across a multi-GB file. Measured against a
#  single sequential scan of the same data it was 8.7x slower, and that gap
#  widens on a real 16 GB file where the page cache cannot hide it.
#
#  So phase 1 is ONE SEQUENTIAL SCAN per source file, streamed in chunks, with
#  the per-cell selection applied incrementally. Keeping the brightest K per
#  cell is associative — the top-K of a union of per-chunk top-K sets is the
#  global top-K — so chunking costs nothing in correctness.
#
#  Threads do NOT help here and actively hurt (measured 0.3x): sqlite3 releases
#  the GIL inside sqlite3_step, but building the row tuples holds it, so
#  threads just contend. Processes do help, modestly, and only because each
#  worker reduces its slice to a per-cell selection before anything is pickled
#  back. That path is off in a frozen build unless the host app calls
#  multiprocessing.freeze_support() — see WORKERS below.
#
#  Phase 2 forms quads from one KD-tree over unit vectors, so the neighbour
#  search needs no projection and RA wraparound disappears; each quad is then
#  gnomonic-projected about its own A-B midpoint before hashing.
#
#  Pure numpy + sqlite3. No Qt in this module, so it can be driven headlessly
#  from a script or a test; the QThread wrapper lives in gaia_quad_tab.py.
#
#  Written by Franklin Marek
#  www.setiastro.com
# ======================================================

from __future__ import annotations

import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from setiastro.saspro.quad_core import quad_codes, pack_keys, nbins_for_tol
from setiastro.saspro.gaia_quad_index import (
    BIN_TOL, EPOCH_JYEAR, QuadLayerDef, quantize_codes, write_saq,
)

GAIA_REF_EPOCH = 2016.0          # Gaia DR3 astrometry epoch
SCAN_CHUNK = 500_000             # rows pulled from sqlite per step

# Proper motion only matters where a decade of it is a real fraction of a
# quad. At G<=13.6 a brisk 50 mas/yr star has moved 0.5" since epoch 2016,
# which against an 8' quad is 0.1% — far under the code tolerance. Below a 4'
# quad it starts to bite, so only the two finest layers pay to fetch pm.
PM_QUAD_ARCMIN = 4.0

# The cumulative Gaia DR3 count curve now lives in gaia_database, where the
# per-band row counts it is derived from already are, so the builder and the
# plate solver size their work from ONE table rather than two copies that can
# drift. Used here only to estimate progress: counting rows up front would
# mean scanning the table twice.

ProgressCb = Optional[Callable[[float, str], None]]
CancelCb = Optional[Callable[[], bool]]


def expected_rows(gmax: float) -> int:
    from setiastro.saspro.gaia_database import expected_allsky_count
    return expected_allsky_count(gmax)


def default_workers() -> int:
    """
    How many reader processes to use by default.

    Disabled in a frozen build: multiprocessing 'spawn' re-imports the entry
    point, and unless the host called multiprocessing.freeze_support() that
    relaunches the whole GUI. Override with SASPRO_QUAD_WORKERS.
    """
    env = os.environ.get("SASPRO_QUAD_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if getattr(sys, "frozen", False):
        return 1
    n = os.cpu_count() or 1
    return max(1, min(4, n // 2))


# ─────────────────────────────────────────────────────────────────────────────
#  Source catalogues
# ─────────────────────────────────────────────────────────────────────────────

def _band_from_filename(name: str) -> Tuple[float, float]:
    """
    Magnitude band a distributed file covers, from its name.

    Astrometric: gaia_astro_lt15 -> (0,15),  gaia_astro_150_160 -> (15.0,16.0),
                 gaia_astro_gt213 -> (21.3, 99)
    XP:          gaia_xp_lt8 -> (0,8),  gaia_xp_8_10 -> (8,10),
                 gaia_xp_100_105 -> (10.0,10.5)

    The naming is not consistent between the two builders (lt15 means G<15 but
    gt213 means G>21.3), so the rule is: a 3-digit token is tenths, anything
    shorter is whole magnitudes.
    """
    stem = Path(name).stem

    def val(tok: str) -> float:
        return float(tok) / 10.0 if len(tok) >= 3 else float(tok)

    m = re.search(r"_lt(\d+)$", stem)
    if m:
        return 0.0, val(m.group(1))
    m = re.search(r"_gt(\d+)$", stem)
    if m:
        return val(m.group(1)), 99.0
    m = re.search(r"_(\d+)_(\d+)$", stem)
    if m:
        return val(m.group(1)), val(m.group(2))
    return 0.0, 99.0


def _contiguous_depth(bands: List[Tuple[float, float]]) -> float:
    """How faint the installed files reach without a gap, starting from G=0."""
    depth = 0.0
    for lo, hi in sorted(bands):
        if lo <= depth + 1e-6:
            depth = max(depth, hi)
    return depth


def _connect(path: str) -> sqlite3.Connection:
    c = sqlite3.connect(path, check_same_thread=False)
    c.execute("PRAGMA query_only = ON;")
    c.execute("PRAGMA cache_size = -131072;")      # 128 MB page cache
    c.execute("PRAGMA mmap_size = 1073741824;")    # let the OS map 1 GB
    return c


class _Source:
    kind = "?"
    table = "stars"

    def depth(self) -> float:
        raise NotImplementedError

    def files(self, gmax: float) -> List[Path]:
        raise NotImplementedError

    def scan(self, gmax: float, want_pm: bool) -> Iterator[tuple]:
        raise NotImplementedError

    def close(self):
        for c in getattr(self, "_conns", {}).values():
            try:
                c.close()
            except Exception:
                pass
        getattr(self, "_conns", {}).clear()


def _stream(conn, sql, params, want_pm):
    cur = conn.execute(sql, params)
    while True:
        rows = cur.fetchmany(SCAN_CHUNK)
        if not rows:
            return
        # Every selected column is non-NULL (COALESCE handles pm), so this is
        # one C-level cast. Going through dtype=object and per-column astype,
        # which is the natural-looking alternative, is several times slower.
        a = np.asarray(rows, dtype=np.float64)
        if want_pm:
            yield a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        else:
            yield a[:, 0], a[:, 1], a[:, 2], None, None


class AstrometricSource(_Source):
    kind = "astrometric"
    table = "stars"

    def __init__(self, directory: Path,
                 neighborhood_name: str = "gaia_neighborhood.sqlite"):
        self._dir = Path(directory)
        self._conns: Dict[str, sqlite3.Connection] = {}
        self._bands: Dict[str, Tuple[float, float]] = {}
        for p in sorted(self._dir.glob("gaia_astro_*.sqlite")):
            if p.name == neighborhood_name:
                continue
            try:
                self._conns[p.name] = _connect(str(p))
                self._bands[p.name] = _band_from_filename(p.name)
            except Exception as e:
                print(f"[quad-builder] skipping {p.name}: {e}")

    def depth(self) -> float:
        return _contiguous_depth(list(self._bands.values()))

    def files(self, gmax: float) -> List[Path]:
        return [self._dir / n for n, (lo, _) in self._bands.items() if lo <= gmax]

    def scan(self, gmax, want_pm):
        cols = ("ra, dec, phot_g_mean_mag, COALESCE(pmra,0), COALESCE(pmdec,0)"
                if want_pm else "ra, dec, phot_g_mean_mag")
        for name, conn in self._conns.items():
            if self._bands[name][0] > gmax:
                continue
            sql = (f"SELECT {cols} FROM stars WHERE phot_g_mean_mag IS NOT NULL "
                   f"AND phot_g_mean_mag <= ?")
            try:
                yield from _stream(conn, sql, (float(gmax),), want_pm)
            except Exception as e:
                print(f"[quad-builder] scan failed on {name}: {e}")

    def slices(self, gmax: float, nparts: int) -> List[tuple]:
        """Contiguous primary-key ranges, for the parallel reader."""
        out = []
        for p in self.files(gmax):
            try:
                c = _connect(str(p))
                lo, hi = c.execute(
                    "SELECT MIN(source_id), MAX(source_id) FROM stars").fetchone()
                c.close()
            except Exception:
                continue
            if lo is None:
                continue
            # Plain Python ints: a numpy integer binds as 0 in sqlite3,
            # silently, and the query then returns nothing at all.
            edges = [int(v) for v in np.linspace(int(lo), int(hi) + 1, nparts + 1)]
            out += [(str(p), edges[i], edges[i + 1]) for i in range(nparts)]
        return out


class XPSource(_Source):
    kind = "XP spectral"
    table = "sources"

    def __init__(self, directory: Path):
        self._dir = Path(directory)
        self._conns: Dict[str, sqlite3.Connection] = {}
        self._bands: Dict[str, Tuple[float, float]] = {}
        for p in sorted(self._dir.glob("gaia_xp_*.sqlite")):
            try:
                self._conns[p.name] = _connect(str(p))
                self._bands[p.name] = _band_from_filename(p.name)
            except Exception as e:
                print(f"[quad-builder] skipping {p.name}: {e}")

    def depth(self) -> float:
        return _contiguous_depth(list(self._bands.values()))

    def files(self, gmax: float) -> List[Path]:
        return [self._dir / n for n, (lo, _) in self._bands.items() if lo <= gmax]

    def scan(self, gmax, want_pm):
        # The XP tables carry no proper motion; the caller tolerates None.
        for name, conn in self._conns.items():
            if self._bands[name][0] > gmax:
                continue
            sql = ("SELECT ra, dec, phot_g_mean_mag FROM sources "
                   "WHERE phot_g_mean_mag IS NOT NULL AND phot_g_mean_mag <= ?")
            try:
                yield from _stream(conn, sql, (float(gmax),), False)
            except Exception as e:
                print(f"[quad-builder] scan failed on {name}: {e}")


@dataclass
class SourceInfo:
    kind: str
    depth: float
    detail: str


def open_source(astro_dir: Optional[Path] = None,
                xp_dir: Optional[Path] = None) -> Tuple[Optional[_Source], SourceInfo]:
    """Pick the deepest installed catalogue; astrometric wins on a tie."""
    best, info = None, SourceInfo("none", 0.0, "No Gaia catalogue installed.")
    cands: List[_Source] = []
    if astro_dir and Path(astro_dir).exists():
        cands.append(AstrometricSource(Path(astro_dir)))
    if xp_dir and Path(xp_dir).exists():
        cands.append(XPSource(Path(xp_dir)))
    for s in cands:
        d = s.depth()
        if d > info.depth:
            if best is not None:
                best.close()
            best = s
            info = SourceInfo(s.kind, d, f"{s.kind} library, complete to G ≤ {d:.1f}")
        else:
            s.close()
    return best, info


def buildable_layers(depth: float, layers) -> Dict[int, bool]:
    return {d.index: (depth >= d.gmag_limit - 1e-6) for d in layers}


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _radec_to_xyz(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(ra_deg); dec = np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def _chord(theta_rad):
    return 2.0 * np.sin(np.asarray(theta_rad) * 0.5)


def _gnomonic(points_xyz: np.ndarray, centre_xyz: np.ndarray) -> np.ndarray:
    """
    Gnomonic projection of (M,K,3) directions about (M,3) centres, as (M,K,2).
    Done with vectors, not trig: u = unit(z x n0) is east, v = n0 x u is north,
    and x = (p.u)/(p.n0).
    """
    n0 = centre_xyz
    u = np.column_stack([-n0[:, 1], n0[:, 0], np.zeros(len(n0))])
    nu = np.linalg.norm(u, axis=1)
    bad = nu < 1e-12                                   # exactly at a pole
    u[bad] = np.array([1.0, 0.0, 0.0])
    nu[bad] = 1.0
    u /= nu[:, None]
    v = np.cross(n0, u)
    w = np.einsum("mkj,mj->mk", points_xyz, n0)
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    x = np.einsum("mkj,mj->mk", points_xyz, u) / w
    y = np.einsum("mkj,mj->mk", points_xyz, v) / w
    return np.stack([x, y], axis=2)


def _topk_smallest(vals: np.ndarray, valid: np.ndarray, k: int):
    """Per row, the k column indices with the smallest `vals` among `valid`."""
    big = np.iinfo(np.int64).max
    v = np.where(valid, vals, big)
    order = np.argsort(v, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(v, order, axis=1) < big


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1 — choose index stars
# ─────────────────────────────────────────────────────────────────────────────

def _select_index_stars(ra, dec, mag, cell_deg, per_cell):
    """
    Keep the brightest `per_cell` stars in each roughly-equal-area cell.

    Cells are a declination band plus a cos(dec)-corrected RA division — the
    same zone idea the astrometric builder already uses, within ~10% of equal
    area, and with no healpy dependency.
    """
    if len(ra) == 0:
        return np.zeros(0, np.int64)
    jd = np.floor((dec + 90.0) / cell_deg).astype(np.int64)
    band_mid = (jd + 0.5) * cell_deg - 90.0
    nra = np.maximum(1, np.round(360.0 * np.cos(np.radians(band_mid))
                                 / cell_deg).astype(np.int64))
    ir = np.minimum(np.floor((ra % 360.0) / (360.0 / nra)).astype(np.int64),
                    nra - 1)
    key = jd * (1 << 22) + ir

    o = np.lexsort((mag, key))
    ks = key[o]
    new = np.empty(len(ks), bool)
    new[0] = True
    np.not_equal(ks[1:], ks[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, len(ks)))
    rank = np.arange(len(ks)) - np.repeat(starts, counts)
    return o[rank < per_cell]


def _apply_pm(ra, dec, pmra, pmdec, epoch):
    """Propagate to `epoch`. pmra is mu_alpha* (already includes cos dec)."""
    dt = float(epoch) - GAIA_REF_EPOCH
    if dt == 0.0 or pmra is None:
        return ra, dec
    cd = np.maximum(np.cos(np.radians(dec)), 1e-6)
    return ((ra + (pmra * dt / 3.6e6) / cd) % 360.0,
            np.clip(dec + (pmdec * dt / 3.6e6), -90.0, 90.0))


class _Selector:
    """
    Running per-cell top-K over a stream of chunks.

    Correct because the selection is associative: the top-K of a union of
    per-chunk top-K sets is the global top-K. Consolidation is deferred until
    the backlog is several times the final size, so the sort cost stays near
    linear in the rows actually kept.
    """

    def __init__(self, cell_deg: float, per_cell: int, expected: int):
        self.cell = cell_deg
        self.k = per_cell
        self.limit = max(4 * max(expected, 1), 2_000_000)
        self._parts: List[tuple] = []
        self.n = 0
        self.rows_read = 0

    def add(self, ra, dec, mag):
        if len(ra) == 0:
            return
        s = _select_index_stars(ra, dec, mag, self.cell, self.k)
        self._parts.append((ra[s], dec[s], mag[s]))
        self.n += len(s)
        if self.n > self.limit:
            self.consolidate()

    def consolidate(self):
        if len(self._parts) <= 1:
            return
        ra = np.concatenate([p[0] for p in self._parts])
        dec = np.concatenate([p[1] for p in self._parts])
        mag = np.concatenate([p[2] for p in self._parts])
        s = _select_index_stars(ra, dec, mag, self.cell, self.k)
        self._parts = [(ra[s], dec[s], mag[s])]
        self.n = len(s)

    def result(self):
        self.consolidate()
        if not self._parts:
            z = np.zeros(0)
            return z, z, z
        return self._parts[0]


def _scan_slice(task):
    """
    Worker entry point: scan one primary-key slice and return its per-cell
    selection. Reducing inside the worker is what keeps the pickled payload
    small — a raw slice would be far more expensive to ship back than to read.
    """
    (db, lo, hi, gmax, want_pm, cell_deg, per_cell, expected, epoch) = task
    cols = ("ra, dec, phot_g_mean_mag, COALESCE(pmra,0), COALESCE(pmdec,0)"
            if want_pm else "ra, dec, phot_g_mean_mag")
    sel = _Selector(cell_deg, per_cell, expected)
    c = _connect(db)
    try:
        cur = c.execute(
            f"SELECT {cols} FROM stars WHERE source_id >= ? AND source_id < ? "
            f"AND phot_g_mean_mag IS NOT NULL AND phot_g_mean_mag <= ?",
            (int(lo), int(hi), float(gmax)))
        while True:
            rows = cur.fetchmany(SCAN_CHUNK)
            if not rows:
                break
            a = np.asarray(rows, dtype=np.float64)
            sel.rows_read += len(a)
            ra, dec = a[:, 0], a[:, 1]
            if want_pm:
                ra, dec = _apply_pm(ra, dec, a[:, 3], a[:, 4], epoch)
            sel.add(ra, dec, a[:, 2])
    finally:
        c.close()
    ra, dec, mag = sel.result()
    return ra, dec, mag, sel.rows_read


# ─────────────────────────────────────────────────────────────────────────────
#  Build
# ─────────────────────────────────────────────────────────────────────────────

def _phase1(layer, source, cell_deg, epoch, workers, rep, stop):
    """One sequential pass per file (or per primary-key slice, in parallel)."""
    want_pm = layer.q_arcmin <= PM_QUAD_ARCMIN
    expected = layer.est_stars
    est_rows = max(expected, expected_rows(layer.gmag_limit))
    sel = _Selector(cell_deg, layer.per_cell, expected)
    t0 = time.time()

    slices = []
    if workers > 1 and isinstance(source, AstrometricSource):
        slices = source.slices(layer.gmag_limit, workers * 3)

    if slices:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(db, lo, hi, layer.gmag_limit, want_pm, cell_deg,
                  layer.per_cell, expected, epoch) for db, lo, hi in slices]
        done = 0
        try:
            with ProcessPoolExecutor(workers) as ex:
                for ra, dec, mag, nread in ex.map(_scan_slice, tasks):
                    stop()
                    sel.rows_read += nread
                    sel.add(ra, dec, mag)
                    done += 1
                    rep(0.45 * done / len(tasks),
                        f"Reading catalogue ({workers} workers) … "
                        f"{sel.rows_read:,} rows, {sel.n:,} kept")
            return sel.result(), sel.rows_read, time.time() - t0
        except Exception as e:
            # A frozen build without freeze_support(), a sandbox with no
            # /dev/shm, a locked-down policy — any of these. Fall through to
            # the serial path rather than failing the build.
            print(f"[quad-builder] parallel read unavailable ({e}); "
                  f"falling back to a single reader")
            sel = _Selector(cell_deg, layer.per_cell, expected)

    for ra, dec, mag, pmra, pmdec in source.scan(layer.gmag_limit, want_pm):
        stop()
        sel.rows_read += len(ra)
        if want_pm and pmra is not None:
            ra, dec = _apply_pm(ra, dec, pmra, pmdec, epoch)
        sel.add(ra, dec, mag)
        rep(0.45 * min(1.0, sel.rows_read / max(est_rows, 1)),
            f"Reading catalogue … {sel.rows_read:,} rows, {sel.n:,} kept")

    return sel.result(), sel.rows_read, time.time() - t0


def build_layer(layer: QuadLayerDef,
                out_path: Path,
                source: _Source,
                progress: ProgressCb = None,
                cancel: CancelCb = None,
                kn: int = 28,
                kc: int = 24,
                max_b_per_a: int = 2,
                a_chunk: int = 150_000,
                workers: Optional[int] = None,
                epoch: float = EPOCH_JYEAR) -> Dict:
    """
    Build one layer and write it to `out_path`. Returns a stats dict.
    Raises RuntimeError("cancelled") if the cancel callback asks it to stop.
    """
    from scipy.spatial import cKDTree

    t0 = time.time()
    if workers is None:
        workers = default_workers()

    def rep(f, msg):
        if progress:
            progress(max(0.0, min(1.0, f)), msg)

    def stop():
        if cancel and cancel():
            raise RuntimeError("cancelled")

    cell_deg = layer.cell_arcmin / 60.0
    q_lo = math.radians(layer.q_lo_arcmin / 60.0)
    q_hi = math.radians(layer.q_hi_arcmin / 60.0)

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    (star_ra, star_dec, star_mag), n_read, t_read = _phase1(
        layer, source, cell_deg, epoch, workers, rep, stop)

    if len(star_ra) == 0:
        raise RuntimeError("No stars returned by the catalogue — is the right "
                           "magnitude tier installed?")

    # Global brightness order, so an array index IS its brightness rank. The
    # solver ranks its detections the same way; that correspondence is what
    # makes the builder and the solver choose the same C and D.
    o = np.argsort(star_mag, kind="stable")
    star_ra, star_dec, star_mag = star_ra[o], star_dec[o], star_mag[o]
    n_stars = len(star_ra)
    rep(0.45, f"{n_stars:,} index stars from {n_read:,} rows in {t_read:.0f}s "
              f"— building tree")
    stop()

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    xyz = _radec_to_xyz(star_ra, star_dec)
    tree = cKDTree(xyz)
    c_lo, c_hi = _chord(q_lo), _chord(q_hi)

    keys_l, codes_l, refs_l = [], [], []
    n_quads = 0
    nb = nbins_for_tol(BIN_TOL)

    for s in range(0, n_stars, a_chunk):
        stop()
        e = min(s + a_chunk, n_stars)
        rep(0.45 + 0.50 * s / max(n_stars, 1),
            f"Forming quads … {s:,}/{n_stars:,} stars, {n_quads:,} quads")

        ai = np.arange(s, e)
        d, j = tree.query(xyz[ai], k=kn, distance_upper_bound=c_hi, workers=-1)
        # Only pair A with a FAINTER star: each unordered pair appears once.
        ok = (j < n_stars) & (d >= c_lo) & (j > ai[:, None])
        if not ok.any():
            continue
        col, got = _topk_smallest(j, ok, max_b_per_a)
        bj = np.take_along_axis(j, col, axis=1)

        a_rep = np.repeat(ai, max_b_per_a)[got.ravel()]
        b_rep = bj.ravel()[got.ravel()]
        if len(a_rep) == 0:
            continue

        # Circle with AB as diameter: centre is the normalised midpoint,
        # angular radius is half the separation.
        mid = xyz[a_rep] + xyz[b_rep]
        mid /= np.linalg.norm(mid, axis=1)[:, None]
        sep = 2.0 * np.arcsin(np.clip(
            0.5 * np.linalg.norm(xyz[a_rep] - xyz[b_rep], axis=1), 0.0, 1.0))
        r_chord = _chord(sep * 0.5)

        dc, jc = tree.query(mid, k=kc, distance_upper_bound=float(r_chord.max()),
                            workers=-1)
        okc = ((jc < n_stars) & (dc <= r_chord[:, None])
               & (jc != a_rep[:, None]) & (jc != b_rep[:, None]))
        colc, gotc = _topk_smallest(jc, okc, 2)
        cd = np.take_along_axis(jc, colc, axis=1)
        have = gotc.all(axis=1)
        if not have.any():
            continue

        a_rep, b_rep, mid = a_rep[have], b_rep[have], mid[have]
        sids = np.column_stack([a_rep, b_rep, cd[have, 0], cd[have, 1]]).astype(np.int64)

        pts = _gnomonic(xyz[sids], mid)                     # (M,4,2)
        m = len(pts)
        code, idx, good, _ = quad_codes(pts.reshape(-1, 2),
                                        np.arange(4 * m).reshape(m, 4))
        if not good.any():
            continue
        code = code[good]
        ids = sids.reshape(-1)[idx[good]]                   # canonical A,B,C,D

        keys_l.append(pack_keys(code, nb).astype(np.uint32))
        # Quantise NOW, not at write time. Layer 0 produces ~128M quads; held
        # as float64 the code array alone is 4 GB, and the sort below then
        # needs a second copy of it. At uint16 it is 1 GB.
        codes_l.append(quantize_codes(code))
        refs_l.append(ids[:, :2].astype(np.uint32))
        n_quads += len(code)

    if n_quads == 0:
        raise RuntimeError("No quads formed. The catalogue is probably too "
                           "sparse at this magnitude limit for this layer.")

    stop()
    rep(0.95, f"Sorting {n_quads:,} quads")
    # Each array is concatenated, permuted and freed one at a time. Doing it
    # in a single tuple assignment keeps every old array alive until the whole
    # right-hand side is built, which on layer 0 is gigabytes of avoidable
    # peak. star_mag has done its job (it set the brightness order) and goes
    # too.
    del star_mag
    keys = np.concatenate(keys_l); del keys_l
    o = np.argsort(keys, kind="stable")
    tmp = keys[o]; del keys; keys = tmp
    codes = np.concatenate(codes_l); del codes_l
    tmp = codes[o]; del codes; codes = tmp
    refs = np.concatenate(refs_l); del refs_l
    tmp = refs[o]; del refs; refs = tmp
    del o, tmp

    rep(0.98, "Writing index file")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_saq(out_path, layer, star_ra, star_dec, keys, codes, refs,
              nbins=nb, epoch=epoch)

    dt = time.time() - t0
    rep(1.0, f"Layer {layer.index} built — {n_quads:,} quads, "
             f"{out_path.stat().st_size / 1024**2:.0f} MB, {dt:.0f}s")
    return dict(layer=layer.index, n_stars=int(n_stars), n_quads=int(n_quads),
                bytes=int(out_path.stat().st_size), seconds=dt,
                rows_read=int(n_read), read_seconds=t_read,
                workers=int(workers), path=str(out_path))