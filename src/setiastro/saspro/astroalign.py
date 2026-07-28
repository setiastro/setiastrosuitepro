"""
setiastro/saspro/astroalign.py
══════════════════════════════════════════════════════════════════════════════
SASpro Parallel Star Alignment Engine  —  robust matching edition
──────────────────────────────────────────────────────────────────────────────
Drop-in replacement for the upstream astroalign library. Preserves the public
API (find_transform / register / apply_transform / estimate_transform /
matrix_transform / MaxIterError) so all existing call sites keep working.

What changed vs. the previous SASpro rewrite
────────────────────────────────────────────
The previous version reproduced upstream astroalign's pipeline — generate
triangle invariants, ball-query invariant space for candidate triangle pairs,
then RANSAC over whole-triangle hypotheses — accelerated with Numba. It failed
(`MaxIterError: List of matching triangles exhausted…`) whenever detections
between the two frames differed enough: dropped stars, spurious detections,
centroid noise, a rotator/meridian flip, or a mirrored optical train.

This edition borrows the robustness machinery from the in-house Gaia plate
solver and folds it into a layered matcher:

  1. Canonical triangle invariants (FIX).  Vertices are now ordered by the
     length of their opposite side, so vertex *k* of one triangle is the same
     geometric corner as vertex *k* of a similar triangle. The prior kernel
     computed invariants from sorted sides but stored vertices in raw
     KD-tree-neighbour order, so ~2/3 of the point correspondences fed to
     RANSAC were geometrically wrong even for a correctly matched triangle
     pair. Canonical ordering is also the prerequisite for vertex voting.

  2. Vertex voting.  Every matched triangle pair casts three votes — one per
     canonical vertex correspondence. True correspondences accumulate votes;
     coincidental triangle matches scatter theirs. Correspondences are then
     assigned one-to-one by descending vote count. This is dramatically more
     tolerant of spurious triangles than whole-triangle consensus.

  3. Parity-aware similarity RANSAC.  Minimal 2-point samples, testing both
     handedness conventions, so a mirrored frame (reflection) is recovered
     rather than silently returning a garbage transform.

  4. Full-set re-match with iterative tightening.  Once a coarse transform is
     found from the bright subset, ALL control points are projected through it
     and nearest-neighbour matched, refitting and shrinking the tolerance to
     the observed residual scale. The triangle stage therefore only needs to
     find a *rough* transform; the re-match harvests the rest.

  5. Hough translation-vote fallback.  If triangles fail entirely (very few or
     very asymmetric detections), a (dx, dy) translation vote followed by a
     similarity refit still recovers near-translation alignments.

Numba is still used where it pays (parallel invariant generation) and remains
optional — a pure-numpy path runs everything correctly (just single-threaded)
when numba is unavailable.

Return type note: for ordinary (non-mirrored) data find_transform returns a
skimage SimilarityTransform, exactly as before. For mirrored data it returns a
skimage AffineTransform (a SimilarityTransform cannot represent a reflection);
both expose .params / .inverse / .scale / .rotation and work with
matrix_transform() and warp().

Original astroalign algorithm design:
    © 2016 Martin Beroiz (MIT License) — https://github.com/quatrope/astroalign
SASpro changes © Franklin Marek | www.setiastro.com
"""

from __future__ import annotations

__version__ = "2.0.0-saspro"

__all__ = [
    "MIN_MATCHES_FRACTION",
    "MaxIterError",
    "NUM_NEAREST_NEIGHBORS",
    "PIXEL_TOL",
    "apply_transform",
    "estimate_transform",
    "find_transform",
    "matrix_transform",
    "register",
]

import math
import warnings
import numpy as np
from itertools import combinations
from scipy.spatial import cKDTree as _KDTree
from skimage.transform import (
    SimilarityTransform as _SimilarityTransform,
    AffineTransform as _AffineTransform,
)

# ─────────────────────────────────────────────────────────────────────────────
# Numba bootstrap — optional. Pure-numpy path is fully correct without it.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    import numba as _numba  # noqa: F401
    _NUMBA_OK = True
except ImportError:  # pragma: no cover
    _NUMBA_OK = False

    def njit(*args, **kwargs):          # type: ignore[misc]
        def _wrap(fn):
            return fn
        return _wrap if not (args and callable(args[0])) else _wrap(args[0])

    def prange(n):                       # type: ignore[misc]
        return range(n)

# ─────────────────────────────────────────────────────────────────────────────
# Public constants (retained for API compatibility)
# ─────────────────────────────────────────────────────────────────────────────
PIXEL_TOL = 2
"""Final-match pixel tolerance floor. Default: 2"""

MIN_MATCHES_FRACTION = 0.8
"""Retained for API compatibility. Acceptance is now based on the recovered
final correspondence count + residual, not a triangle fraction."""

NUM_NEAREST_NEIGHBORS = 5
"""Retained for API compatibility (legacy KNN-triangle parameter)."""

# ── tunables for the robust matcher ──────────────────────────────────────────
MAX_TRIANGLE_STARS = 70
"""Brightest / spatially-spread control points used for triangle enumeration.
Triangle count is C(n,3), so this bounds cost; 70 → ~55k triangles."""

INVARIANT_TOL = 0.03
"""Nearest-neighbour tolerance in (long/short, mid/short) invariant space."""

MIN_TRIANGLE_SHORT_PX = 6.0
"""Reject triangles whose shortest side is below this (centroid noise dominates)."""

MAX_TRIANGLE_ASPECT = 10.0
"""Reject needle-thin triangles (longest/shortest side above this)."""

MIN_FINAL_MATCHES = 6
"""Minimum recovered correspondences to accept a solution (absolute floor)."""


# ─────────────────────────────────────────────────────────────────────────────
# skimage shims (same lazy-loader pattern as upstream)
# ─────────────────────────────────────────────────────────────────────────────
def estimate_transform(*args, **kwargs):
    from skimage.transform import estimate_transform as _et
    return _et(*args, **kwargs)


def matrix_transform(*args, **kwargs):
    from skimage.transform import matrix_transform as _mt
    return _mt(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers (unchanged from upstream)
# ─────────────────────────────────────────────────────────────────────────────
def _data(image):
    if hasattr(image, "data") and isinstance(image.data, np.ndarray):
        return image.data
    return np.asarray(image)


def _mask(image):
    if hasattr(image, "mask"):
        m = np.asarray(image.mask)
        return m if m.ndim == 2 else np.logical_or.reduce(m, axis=-1)
    return None


def _bw(image):
    return image if image.ndim == 2 else np.mean(image, axis=-1)


def _shape(image):
    if image.ndim == 2:
        return image.shape
    h, w, _ = image.shape
    return h, w


# ─────────────────────────────────────────────────────────────────────────────
# SEP-based star detector (same as upstream _find_sources)
# ─────────────────────────────────────────────────────────────────────────────
def _find_sources(img, detection_sigma=5, min_area=5, mask=None):
    """Return (x, y) star positions sorted by flux descending."""
    import sep
    sep.set_extract_pixstack(20000000)
    image = np.ascontiguousarray(img.astype(np.float32))
    bkg = sep.Background(image, mask=mask)
    thresh = detection_sigma * bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh, minarea=min_area, mask=mask)
    sources.sort(order="flux")
    return np.array([[s["x"], s["y"]] for s in sources[::-1]])


# ─────────────────────────────────────────────────────────────────────────────
# Spatial balancing: keep the brightest points spread across a grid so the
# triangle stage isn't dominated by one bright cluster (e.g. a nebula core).
# Input points are assumed brightness-sorted (descending); "brightest per cell"
# is therefore the earliest-in-array per cell.
# ─────────────────────────────────────────────────────────────────────────────
def _grid_balance(pts, m, w=None, h=None, grid=None):
    """
    Return ~m spatially-spread points from a brightness-sorted array, so the
    triangle stage isn't dominated by one bright cluster. Uses a round-robin
    over grid cells (brightest of each cell first, then the next, …) which
    spreads points across the frame AND always reaches min(m, n) points.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n <= m:
        return pts.copy()
    xy = pts[:, :2]
    if w is None:
        w = float(xy[:, 0].max() - xy[:, 0].min()) or 1.0
        x0 = float(xy[:, 0].min())
    else:
        x0 = 0.0
    if h is None:
        h = float(xy[:, 1].max() - xy[:, 1].min()) or 1.0
        y0 = float(xy[:, 1].min())
    else:
        y0 = 0.0

    if grid is None:
        cols = max(1, int(round(math.sqrt(m * w / max(h, 1.0)))))
        rows = max(1, int(round(m / cols)))
    else:
        rows, cols = grid
    cw = w / cols if cols else w
    rh = h / rows if rows else h

    # bucket point indices into cells, preserving brightness (array) order
    cells: dict = {}
    for idx in range(n):
        cx = int((xy[idx, 0] - x0) / cw) if cw else 0
        cy = int((xy[idx, 1] - y0) / rh) if rh else 0
        cx = min(max(cx, 0), cols - 1)
        cy = min(max(cy, 0), rows - 1)
        cells.setdefault((cx, cy), []).append(idx)

    # round-robin: depth 0 takes the brightest of every occupied cell, etc.
    chosen: list = []
    depth = 0
    keys = list(cells.keys())
    while len(chosen) < m:
        progressed = False
        for key in keys:
            bucket = cells[key]
            if len(bucket) > depth:
                chosen.append(bucket[depth])
                progressed = True
                if len(chosen) >= m:
                    break
        if not progressed:
            break
        depth += 1
    return pts[np.array(chosen[:m], dtype=np.int64)]


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — canonical triangle invariants  (parallel)
# ══════════════════════════════════════════════════════════════════════════════
#
# For three points, order the vertices by the length of the side OPPOSITE each
# one (short→long). Two similar triangles then share the same canonical vertex
# order, so vertex k ↔ vertex k is a geometrically meaningful correspondence.
# Invariant = (longest/shortest, middle/shortest) — invariant to rotation,
# translation, uniform scale, AND reflection (side lengths are preserved).
# ──────────────────────────────────────────────────────────────────────────────
@njit(parallel=True, cache=True, fastmath=True)
def _tri_invariants_kernel(coords, triples, min_short, max_aspect):
    T = triples.shape[0]
    inv = np.zeros((T, 2), dtype=np.float64)
    V = np.zeros((T, 3), dtype=np.int64)
    valid = np.zeros(T, dtype=np.bool_)
    for k in prange(T):
        i0 = triples[k, 0]
        i1 = triples[k, 1]
        i2 = triples[k, 2]
        x0 = coords[i0, 0]; y0 = coords[i0, 1]
        x1 = coords[i1, 0]; y1 = coords[i1, 1]
        x2 = coords[i2, 0]; y2 = coords[i2, 1]
        # side opposite each vertex
        la = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)   # opp v0
        lb = math.sqrt((x0 - x2) ** 2 + (y0 - y2) ** 2)   # opp v1
        lc = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)   # opp v2
        va = i0; vb = i1; vc = i2
        # 3-element sort network (ascending by opposite-side length)
        if lb < la:
            la, lb = lb, la; va, vb = vb, va
        if lc < la:
            la, lc = lc, la; va, vc = vc, va
        if lc < lb:
            lb, lc = lc, lb; vb, vc = vc, vb
        if la < min_short:
            continue
        denom = la if la > 1e-9 else 1e-9
        if lc / denom >= max_aspect:
            continue
        inv[k, 0] = lc / denom
        inv[k, 1] = lb / denom
        V[k, 0] = va; V[k, 1] = vb; V[k, 2] = vc
        valid[k] = True
    return inv, V, valid


def _tri_invariants_numpy(coords, triples, min_short, max_aspect):
    """Vectorised numpy equivalent of the kernel."""
    p0 = coords[triples[:, 0]]
    p1 = coords[triples[:, 1]]
    p2 = coords[triples[:, 2]]
    s0 = np.hypot(p1[:, 0] - p2[:, 0], p1[:, 1] - p2[:, 1])   # opp v0
    s1 = np.hypot(p0[:, 0] - p2[:, 0], p0[:, 1] - p2[:, 1])   # opp v1
    s2 = np.hypot(p0[:, 0] - p1[:, 0], p0[:, 1] - p1[:, 1])   # opp v2
    S = np.column_stack([s0, s1, s2])
    order = np.argsort(S, axis=1)                             # short→long
    Ss = np.take_along_axis(S, order, axis=1)
    V = np.take_along_axis(triples, order, axis=1)
    denom = np.maximum(Ss[:, 0], 1e-9)
    good = (Ss[:, 0] > min_short) & (Ss[:, 2] / denom < max_aspect)
    inv = np.column_stack([Ss[:, 2] / denom, Ss[:, 1] / denom])
    return inv[good], V[good]


from functools import lru_cache


@lru_cache(maxsize=16)
def _triple_index(n):
    """All C(n,3) vertex-index triples as an (T,3) int64 array (cached per n)."""
    return np.array(list(combinations(range(n), 3)), dtype=np.int64)


def _generate_tri_invariants(coords):
    """
    Return (inv, V) for all C(n,3) triangles over *coords*.
    inv : (T,2) float64 invariants ; V : (T,3) int64 canonical vertex indices.
    """
    coords = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
    n = len(coords)
    if n < 3:
        return np.zeros((0, 2)), np.zeros((0, 3), np.int64)
    triples = _triple_index(n)
    if _NUMBA_OK:
        inv, V, valid = _tri_invariants_kernel(
            coords, triples, float(MIN_TRIANGLE_SHORT_PX), float(MAX_TRIANGLE_ASPECT)
        )
        return inv[valid], V[valid]
    return _tri_invariants_numpy(
        coords, triples, float(MIN_TRIANGLE_SHORT_PX), float(MAX_TRIANGLE_ASPECT)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — vertex voting
# ══════════════════════════════════════════════════════════════════════════════
def _vote_correspondences(inv_s, V_s, inv_r, V_r,
                          inv_tol=INVARIANT_TOL, min_votes=2, k=1):
    """
    Match source→target triangles in invariant space; every matched pair casts
    one vote per canonical vertex correspondence. Return one-to-one (s,t) index
    pairs assigned greedily by descending vote count.
    """
    if len(inv_s) == 0 or len(inv_r) == 0:
        return []
    n_tri_r = len(inv_r)
    tree = _KDTree(inv_r)
    dists, idxs = tree.query(inv_s, k=k, distance_upper_bound=inv_tol, workers=-1)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # keep only source-triangle rows that found a finite target-triangle match
    finite = np.isfinite(dists) & (idxs < n_tri_r)
    si_rows, kk_cols = np.where(finite)
    if si_rows.size == 0:
        return []
    ri_rows = idxs[si_rows, kk_cols]

    # three canonical vertex-pair votes per matched triangle pair, vectorised
    src_v = V_s[si_rows]          # (H,3) source star indices in canonical order
    ref_v = V_r[ri_rows]          # (H,3) target star indices in canonical order
    NR = int(V_r.max()) + 1 if len(V_r) else 1     # star-index space for encoding
    keys = (src_v.astype(np.int64) * NR + ref_v.astype(np.int64)).ravel()
    uniq, counts = np.unique(keys, return_counts=True)

    order = np.argsort(-counts)                    # descending vote count
    uniq, counts = uniq[order], counts[order]
    s_idx = uniq // NR
    t_idx = uniq % NR

    used_s, used_r, cand = set(), set(), []
    for si, tj, c in zip(s_idx.tolist(), t_idx.tolist(), counts.tolist()):
        if c < min_votes:
            break
        if si in used_s or tj in used_r:
            continue
        used_s.add(si)
        used_r.add(tj)
        cand.append((si, tj))
    return cand


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — parity-aware similarity RANSAC
# ══════════════════════════════════════════════════════════════════════════════
def _sim_from_2pt(p0, p1, q0, q1, reflect):
    """Exact similarity (optionally reflected) from two correspondences."""
    dp = complex(p1[0] - p0[0], p1[1] - p0[1])
    dq = complex(q1[0] - q0[0], q1[1] - q0[1])
    if abs(dp) < 1e-9:
        return None
    if reflect:
        w = dq / np.conj(dp)
        a, b = w.real, w.imag
        L = np.array([[a, b], [b, -a]])
    else:
        w = dq / dp
        a, b = w.real, w.imag
        L = np.array([[a, -b], [b, a]])
    t = np.array([q0[0], q0[1]]) - L @ np.array([p0[0], p0[1]])
    return L, t


def _fit_similarity_ls(P, Q, reflect):
    """Least-squares similarity (optionally reflected) over n correspondences."""
    n = len(P)
    x, y = P[:, 0], P[:, 1]
    A = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    if reflect:
        # L = [[a, b],[b, -a]] : qx = a·x + b·y + tx ; qy = -a·y + b·x + ty
        A[0::2, 0] = x;   A[0::2, 1] = y;  A[0::2, 2] = 1.0
        A[1::2, 0] = -y;  A[1::2, 1] = x;  A[1::2, 3] = 1.0
    else:
        # L = [[a, -b],[b, a]] : qx = a·x - b·y + tx ; qy = b·x + a·y + ty
        A[0::2, 0] = x;   A[0::2, 1] = -y;  A[0::2, 2] = 1.0
        A[1::2, 0] = y;   A[1::2, 1] = x;   A[1::2, 3] = 1.0
    b[0::2] = Q[:, 0]
    b[1::2] = Q[:, 1]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bb, tx, ty = sol
    L = np.array([[a, bb], [bb, -a]]) if reflect else np.array([[a, -bb], [bb, a]])
    return L, np.array([tx, ty])


def _similarity_ransac(P, Q, tol, iters=400, seed=0):
    """
    RANSAC over correspondences using minimal 2-point samples, trying both
    handedness conventions per sample. Returns (L, t, reflect, inlier_mask).
    """
    n = len(P)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    tol2 = tol * tol
    best_mask = None
    best_reflect = False
    # no point running more trials than there are distinct 2-point samples
    iters = int(min(iters, max(1, n * (n - 1) // 2)))
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        for reflect in (False, True):
            r = _sim_from_2pt(P[i], P[j], Q[i], Q[j], reflect)
            if r is None:
                continue
            L, t = r
            pred = P @ L.T + t
            d2 = np.sum((pred - Q) ** 2, axis=1)
            mask = d2 < tol2
            if best_mask is None or mask.sum() > best_mask.sum():
                best_mask = mask
                best_reflect = reflect
    if best_mask is None or best_mask.sum() < 2:
        return None
    L, t = _fit_similarity_ls(P[best_mask], Q[best_mask], best_reflect)
    return L, t, best_reflect, best_mask


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — full-set re-match with iterative tightening
# ══════════════════════════════════════════════════════════════════════════════
def _rematch_full(all_src, all_ref, L, t, tol0, min_keep=MIN_FINAL_MATCHES, rounds=4):
    """
    Project every source point through (L,t), nearest-neighbour match to target,
    refit + shrink tolerance toward the observed residual scale. Returns
    (src_pts, tgt_pts, L, t, reflect).
    """
    rtree = _KDTree(all_ref)
    reflect = np.linalg.det(L) < 0
    tol = float(tol0)
    pred = all_src @ L.T + t
    dists, idxs = rtree.query(pred, k=1, workers=-1)
    for _ in range(rounds):
        mask = dists < tol
        if int(mask.sum()) < min_keep:
            break
        L, t = _fit_similarity_ls(all_src[mask], all_ref[idxs[mask]], reflect)
        pred = all_src @ L.T + t
        dists, idxs = rtree.query(pred, k=1, workers=-1)
        core = dists[dists < tol]
        if len(core) == 0:
            break
        new_tol = max(1.5, 4.0 * float(np.median(core)))
        if new_tol >= tol:
            break
        tol = new_tol
    mask = dists < tol
    resid = float(np.median(dists[mask])) if mask.any() else float("inf")
    return all_src[mask], all_ref[idxs[mask]], L, t, reflect, resid


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5 — Hough (dx,dy) translation-vote fallback
# ══════════════════════════════════════════════════════════════════════════════
def _hough_translation_match(src, ref, w, h, tol, max_off=0.5):
    """
    Vote on (dx,dy) offsets, take the peak bin + 3×3 neighbourhood as a
    candidate pool, then fit a similarity via RANSAC. Returns
    (L, t, reflect, inlier_mask) or None.
    """
    src = np.asarray(src, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    bin_px = tol * 2.0
    votes: dict = {}
    for i in range(len(src)):
        dx = ref[:, 0] - src[i, 0]
        dy = ref[:, 1] - src[i, 1]
        ok = (np.abs(dx) <= w * max_off) & (np.abs(dy) <= h * max_off)
        if not ok.any():
            continue
        bx = np.round(dx[ok] / bin_px).astype(int)
        by = np.round(dy[ok] / bin_px).astype(int)
        js = np.where(ok)[0]
        for b, c, j in zip(bx, by, js):
            votes.setdefault((int(b), int(c)), []).append((i, int(j)))
    if not votes:
        return None
    bk = max(votes, key=lambda kk: len(votes[kk]))
    merged = list(votes[bk])
    for dbx in (-1, 0, 1):
        for dby in (-1, 0, 1):
            if dbx or dby:
                merged.extend(votes.get((bk[0] + dbx, bk[1] + dby), []))
    if len(merged) < 4:
        return None
    P = np.array([src[i] for i, j in merged], dtype=np.float64)
    Q = np.array([ref[j] for i, j in merged], dtype=np.float64)
    return _similarity_ransac(P, Q, tol * 2.0, iters=300)


# ─────────────────────────────────────────────────────────────────────────────
# Assemble a skimage transform from (L, t). SimilarityTransform for ordinary
# parity (byte-for-byte compatible with prior behaviour), AffineTransform for
# reflected data (a SimilarityTransform cannot hold a reflection).
# ─────────────────────────────────────────────────────────────────────────────
def _to_skimage_transform(L, t, reflect):
    params = np.array([
        [L[0, 0], L[0, 1], t[0]],
        [L[1, 0], L[1, 1], t[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    if reflect:
        return _AffineTransform(matrix=params)
    return _SimilarityTransform(matrix=params)


# ══════════════════════════════════════════════════════════════════════════════
#  Core matcher — orchestrates the stages
# ══════════════════════════════════════════════════════════════════════════════
def _match_controlpoints(src_cp, tgt_cp, w, h, pix_tol=PIXEL_TOL,
                         inv_tol=INVARIANT_TOL, min_final=MIN_FINAL_MATCHES):
    """
    Returns (transform, src_pts, tgt_pts, info) or None.
    info is a dict with 'stage', 'reflect', 'n', 'resid'.
    """
    src_cp = np.ascontiguousarray(np.asarray(src_cp, dtype=np.float64))
    tgt_cp = np.ascontiguousarray(np.asarray(tgt_cp, dtype=np.float64))

    coarse_tol = max(pix_tol * 3.0, 6.0)

    # acceptance count adapts to how many points are actually available, so the
    # minimum (3-star) API case works; never below 3. For the usual case
    # (>= MIN_FINAL_MATCHES points on both sides) this equals MIN_FINAL_MATCHES.
    min_final_eff = max(3, min(min_final, len(src_cp), len(tgt_cp)))

    # bright, spatially-spread subset for triangle enumeration
    src_tri = _grid_balance(src_cp, MAX_TRIANGLE_STARS, w, h)
    tgt_tri = _grid_balance(tgt_cp, MAX_TRIANGLE_STARS, w, h)

    def _finish(rr, stage):
        L, t, reflect, _ = rr
        sp, tp, L, t, reflect, resid = _rematch_full(
            src_cp, tgt_cp, L, t, tol0=coarse_tol, min_keep=min_final_eff
        )
        if len(sp) >= min_final_eff and resid < max(pix_tol * 2.0, 3.0):
            T = _to_skimage_transform(L, t, reflect)
            return (T, sp.astype(np.float32), tp.astype(np.float32),
                    {"stage": stage, "reflect": bool(reflect),
                     "n": int(len(sp)), "resid": float(resid)})
        return None

    # ── Stage A: triangle invariants + vertex voting ─────────────────────────
    inv_s, V_s = _generate_tri_invariants(src_tri)
    inv_r, V_r = _generate_tri_invariants(tgt_tri)

    cand = _vote_correspondences(inv_s, V_s, inv_r, V_r,
                                 inv_tol=inv_tol, min_votes=2)
    if len(cand) < 3:
        cand = _vote_correspondences(inv_s, V_s, inv_r, V_r,
                                     inv_tol=inv_tol * 1.5, min_votes=1)

    if len(cand) >= 3:
        P = src_tri[[i for i, j in cand]]
        Q = tgt_tri[[j for i, j in cand]]
        rr = _similarity_ransac(P, Q, tol=coarse_tol, iters=400)
        if rr is not None:
            out = _finish(rr, "triangle+vote")
            if out is not None:
                return out

    # ── Stage B: Hough translation-vote fallback ─────────────────────────────
    src_h = _grid_balance(src_cp, 150, w, h)
    tgt_h = _grid_balance(tgt_cp, 150, w, h)
    rr = _hough_translation_match(src_h, tgt_h, w, h, tol=max(pix_tol * 4.0, 8.0))
    if rr is not None:
        out = _finish(rr, "hough")
        if out is not None:
            return out

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MaxIterError (same as upstream)
# ─────────────────────────────────────────────────────────────────────────────
class MaxIterError(RuntimeError):
    """Raised when no valid transform is found by any matching stage."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# _MatchTransform — retained for any external callers that import it
# ─────────────────────────────────────────────────────────────────────────────
class _MatchTransform:
    def __init__(self, source, target):
        self.source = np.asarray(source, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)

    def fit(self, data: np.ndarray):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        return estimate_transform("similarity", self.source[s], self.target[d])

    def get_error(self, data: np.ndarray, approx_t) -> np.ndarray:
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d]).reshape(d1, d2)
        return resid.max(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# find_transform  (same signature / return contract as upstream)
# ─────────────────────────────────────────────────────────────────────────────
def find_transform(
    source,
    target,
    max_control_points: int = 50,
    detection_sigma: int = 5,
    min_area: int = 5,
):
    """
    Estimate the transform that maps *source* pixel coords into *target*.

    Parameters
    ----------
    source, target
        2-D NumPy arrays (images) OR iterables of (x, y) control-point pairs.
    max_control_points
        Maximum number of brightest stars to use.
    detection_sigma
        Background-sigma threshold for SEP star detection (images only).
    min_area
        Minimum connected-pixel area for a valid detection (images only).

    Returns
    -------
    T : skimage transform
        Maps source → target. SimilarityTransform for ordinary data;
        AffineTransform when the frames are mirrored (reflection).
    (source_pos, target_pos) : tuple of (N,2) float32 arrays
        Matched star positions in each image.

    Raises
    ------
    TypeError      — unsupported input type
    ValueError     — fewer than 3 stars detected
    MaxIterError   — no transform found by any matching stage
    """
    # figure out frame size for spatial balancing / Hough ranges
    frame_wh = [None, None]

    def _get_controlp(img_or_pts, label):
        # explicit (x,y) list?
        try:
            arr = _data(img_or_pts)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                return np.asarray(arr, dtype=np.float64)[:max_control_points]
        except Exception:
            pass
        try:
            pts = np.asarray(img_or_pts)
            if pts.ndim == 2 and pts.shape[1] == 2:
                return pts.astype(np.float64)[:max_control_points]
        except Exception:
            pass
        # otherwise treat as image
        try:
            img2d = _bw(_data(img_or_pts))
            frame_wh[0] = img2d.shape[1]
            frame_wh[1] = img2d.shape[0]
            mk = _mask(img_or_pts)
            return _find_sources(img2d, detection_sigma=detection_sigma,
                                 min_area=min_area, mask=mk)[:max_control_points]
        except Exception as e:
            raise TypeError(f"Input type for {label} not supported: {e}")

    src_cp = _get_controlp(source, "source")
    tgt_cp = _get_controlp(target, "target")

    if len(src_cp) < 3:
        raise ValueError("Reference stars in source image are less than the minimum value (3).")
    if len(tgt_cp) < 3:
        raise ValueError("Reference stars in target image are less than the minimum value (3).")

    # frame extent (fall back to point bounds if inputs were raw coordinates)
    all_xy = np.vstack([src_cp[:, :2], tgt_cp[:, :2]])
    w = frame_wh[0] or float(all_xy[:, 0].max() - all_xy[:, 0].min()) or 1.0
    h = frame_wh[1] or float(all_xy[:, 1].max() - all_xy[:, 1].min()) or 1.0

    result = _match_controlpoints(src_cp, tgt_cp, w, h, pix_tol=PIXEL_TOL)

    if result is None:
        raise MaxIterError(
            "No acceptable transform found: triangle+vote and Hough "
            "translation fallback both failed. The frames may share too few "
            "real stars, or the detection thresholds differ too much between "
            "them."
        )

    T, src_pts, tgt_pts, info = result
    return T, (src_pts, tgt_pts)


# ─────────────────────────────────────────────────────────────────────────────
# apply_transform  (identical to upstream)
# ─────────────────────────────────────────────────────────────────────────────
def apply_transform(transform, source, target, fill_value=None, propagate_mask=False):
    """
    Apply *transform* to *source*, outputting an image the same shape as *target*.

    Returns (aligned_image, footprint) — footprint is True where no source
    pixel contributes.
    """
    from skimage.transform import warp

    source_data = _data(source)
    target_shape = _data(target).shape

    aligned_image = warp(
        source_data,
        inverse_map=transform.inverse,
        output_shape=target_shape,
        order=3,
        mode="constant",
        cval=float(np.median(source_data)),
        clip=True,
        preserve_range=True,
    )

    footprint = warp(
        np.zeros(_shape(source_data), dtype=np.float32),
        inverse_map=transform.inverse,
        output_shape=target_shape,
        cval=1.0,
    ) > 0.4

    source_mask = _mask(source)
    if source_mask is not None and propagate_mask:
        if source_mask.shape == source_data.shape:
            src_mask_rot = warp(
                source_mask.astype(np.float32),
                inverse_map=transform.inverse,
                output_shape=target_shape,
                cval=1.0,
            ) > 0.4
            footprint = footprint | src_mask_rot

    if fill_value is not None:
        aligned_image[footprint] = fill_value

    return aligned_image, footprint


# ─────────────────────────────────────────────────────────────────────────────
# register  (identical to upstream)
# ─────────────────────────────────────────────────────────────────────────────
def register(source, target, fill_value=None, propagate_mask=False,
             max_control_points=50, detection_sigma=5, min_area=5):
    """Transform *source* to align pixel-to-pixel with *target*."""
    t, __ = find_transform(
        source=source, target=target, max_control_points=max_control_points,
        detection_sigma=detection_sigma, min_area=min_area,
    )
    return apply_transform(t, source, target, fill_value, propagate_mask)


# ─────────────────────────────────────────────────────────────────────────────
# JIT warm-up  (optional — call once at app startup to avoid first-call delay)
# ─────────────────────────────────────────────────────────────────────────────
def warmup_jit():
    """Pre-compile the numba kernel with a tiny synthetic dataset. No-op without numba."""
    if not _NUMBA_OK:
        return
    try:
        rng = np.random.default_rng(42)
        coords = (rng.random((12, 2)) * 1000.0)
        triples = np.array(list(combinations(range(12), 3)), dtype=np.int64)
        _tri_invariants_kernel(np.ascontiguousarray(coords), triples,
                               float(MIN_TRIANGLE_SHORT_PX), float(MAX_TRIANGLE_ASPECT))
    except Exception:
        pass  # warmup failure is non-fatal


# ─────────────────────────────────────────────────────────────────────────────
# Self-test harness:  python astroalign.py
# Generates synthetic star fields under known transforms (rotation, scale,
# translation, reflection) with dropped/spurious stars + centroid noise and
# reports recovered accuracy. No SEP / images required.
# ─────────────────────────────────────────────────────────────────────────────
def _selftest(n_trials=150, seed=123, verbose=True):
    rng = np.random.default_rng(seed)

    def field(n, w=4000, h=3000):
        return np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)])

    def xform(pts, ang, sc, tx, ty, flip):
        th = math.radians(ang)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]]) * sc
        if flip:
            R = R @ np.array([[-1.0, 0.0], [0.0, 1.0]])
        return pts @ R.T + np.array([tx, ty])

    n_ok = 0
    worst = 0.0
    for _ in range(n_trials):
        n = int(rng.integers(20, 80))
        ang = rng.uniform(0, 360)
        sc = rng.uniform(0.85, 1.15)
        tx = rng.uniform(-500, 500)
        ty = rng.uniform(-400, 400)
        flip = bool(rng.random() < 0.5)
        drop = rng.uniform(0, 0.45)
        add = int(rng.uniform(0, 0.6 * n))
        noise = rng.uniform(0, 2.0)

        src = field(n)
        tgt_full = xform(src, ang, sc, tx, ty, flip)
        keep = rng.random(n) > drop
        tgt = tgt_full[keep].copy()
        if noise > 0:
            tgt += rng.normal(0, noise, tgt.shape)
        if add > 0:
            tgt = np.vstack([tgt, field(add)])
        rng.shuffle(tgt)

        try:
            T, (sp, tp) = find_transform(src, tgt, max_control_points=80)
            pred = matrix_transform(src, T.params)
            d, _ = _KDTree(tgt_full).query(pred, k=1)
            med = float(np.median(d))
            if med < 2.0 and len(sp) >= MIN_FINAL_MATCHES:
                n_ok += 1
                worst = max(worst, med)
        except Exception:
            pass

    if verbose:
        print(f"[selftest] numba={'on' if _NUMBA_OK else 'off'}  "
              f"passed {n_ok}/{n_trials} randomized trials "
              f"(rotation/scale/translation/reflection + drop/outliers/noise); "
              f"worst passing median residual {worst:.3f}px")
    return n_ok, n_trials


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    warmup_jit()
    _selftest()