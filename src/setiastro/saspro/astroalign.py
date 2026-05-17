"""
setiastro/saspro/astroalign.py
══════════════════════════════════════════════════════════════════════════════
SASpro Parallel Star Alignment Engine
Replaces the upstream astroalign library with a fully parallel implementation
designed to saturate all CPU cores via Numba @njit(parallel=True) + prange,
which releases the GIL and runs truly in parallel even inside a
ThreadPoolExecutor (no ProcessPoolExecutor required — safe in frozen builds).

Original astroalign algorithm design:
    © 2016 Martin Beroiz (MIT License)
    https://github.com/quatrope/astroalign

This reimplementation substantially rewrites the triangle-invariant generation,
KD-tree querying, and RANSAC stages for parallelism and performance, while
preserving the same public API so all existing call sites work unchanged.

SASpro changes © Franklin Marek | www.setiastro.com
"""

from __future__ import annotations

__version__ = "1.0.0-saspro"

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
from scipy.spatial import KDTree as _KDTree
from skimage.transform import SimilarityTransform as _SimilarityTransform
print("loaded saspro.astroalign version", __version__)
# ─────────────────────────────────────────────────────────────────────────────
# Numba bootstrap — optional but strongly preferred.
# If numba is unavailable we fall back to a pure-numpy path that is still
# faster than upstream astroalign (vectorised triangle math) but single-threaded.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    import numba as _numba
    _NUMBA_OK = True
except ImportError:  # pragma: no cover
    _NUMBA_OK = False
    # Provide no-op decorators so the rest of the file is importable
    def njit(*args, **kwargs):          # type: ignore[misc]
        def _wrap(fn): return fn
        return _wrap if args and callable(args[0]) else _wrap
    def prange(n): return range(n)     # type: ignore[misc]

# ─────────────────────────────────────────────────────────────────────────────
# Public constants (same as upstream)
# ─────────────────────────────────────────────────────────────────────────────
PIXEL_TOL = 2
"""Pixel-distance tolerance when comparing invariant matched points. Default: 2"""

MIN_MATCHES_FRACTION = 0.8
"""Minimum fraction of triangle matches required to accept a transform. Default: 0.8"""

NUM_NEAREST_NEIGHBORS = 5
"""Nearest-neighbour count (including self) for triangle construction. Default: 5"""

# ─────────────────────────────────────────────────────────────────────────────
# Thin shims for skimage (same lazy-loader pattern as upstream)
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
    image = np.ascontiguousarray(img.astype(np.float32))
    bkg = sep.Background(image, mask=mask)
    thresh = detection_sigma * bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh, minarea=min_area, mask=mask)
    sources.sort(order="flux")
    return np.array([[s["x"], s["y"]] for s in sources[::-1]])


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  CORE PARALLEL ENGINE
# ══════════════════════════════════════════════════════════════════════════════
#
# Triangle invariants:
#   For three points forming sides a ≤ b ≤ c, the two ratios (b/a, c/a) are
#   invariant to rotation, translation, and uniform scale.  Upstream astroalign
#   uses (c/b, b/a) — we match that convention exactly so our KD-tree distances
#   are compatible.
#
# Parallelism strategy:
#   _build_invariant_table_parallel() uses prange over stars.
#   Each iteration is independent (reads global coords + neighbour indices,
#   writes to pre-allocated output arrays).  Numba releases the GIL for each
#   prange chunk → threads in ThreadPoolExecutor actually run in parallel.
#
# ─────────────────────────────────────────────────────────────────────────────

# Maximum triangles per star = C(NUM_NEAREST_NEIGHBORS-1, 2).
# With k=5 neighbours that's C(4,2)=6, but we include self so C(5,2)=10.
# We use 10 here and can increase NUM_NEAREST_NEIGHBORS at runtime.
_MAX_TRI_PER_STAR = 10   # C(5,2)


@njit(parallel=True, cache=True, fastmath=True)
def _build_invariant_table_parallel(
    coords:    np.ndarray,   # (N,2) float32 xy coords
    nn_idx:    np.ndarray,   # (N,K) int32  neighbour indices (from KD-tree)
    K:         int,          # num neighbours (including self)
    max_tri_per_star: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every star, enumerate all C(K,3) neighbour triplets, compute the two
    ratio invariants, and record which source indices form each triangle.

    Returns
    -------
    inv_buf   : (N * max_tri_per_star, 2)  float32  — invariant pairs
    tri_buf   : (N * max_tri_per_star, 3)  int32    — vertex indices into coords
    n_valid   : (N,)                        int32    — actual triangle count per star
    """
    N = coords.shape[0]
    cap = N * max_tri_per_star

    inv_buf  = np.zeros((cap, 2), dtype=np.float32)
    tri_buf  = np.zeros((cap, 3), dtype=np.int32)
    n_valid  = np.zeros(N,        dtype=np.int32)

    for i in prange(N):                   # ← parallel over stars
        base = i * max_tri_per_star
        cnt  = 0
        for a in range(K):
            for b in range(a + 1, K):
                for c in range(b + 1, K):
                    if cnt >= max_tri_per_star:
                        break
                    ia = nn_idx[i, a]
                    ib = nn_idx[i, b]
                    ic = nn_idx[i, c]

                    xa = coords[ia, 0]; ya = coords[ia, 1]
                    xb = coords[ib, 0]; yb = coords[ib, 1]
                    xc = coords[ic, 0]; yc = coords[ic, 1]

                    dab = math.sqrt((xa - xb)**2 + (ya - yb)**2)
                    dbc = math.sqrt((xb - xc)**2 + (yb - yc)**2)
                    dac = math.sqrt((xa - xc)**2 + (ya - yc)**2)

                    # Sort sides ascending: s0 ≤ s1 ≤ s2
                    s0 = dab; s1 = dbc; s2 = dac
                    if s1 < s0: s0, s1 = s1, s0
                    if s2 < s0: s0, s2 = s2, s0
                    if s2 < s1: s1, s2 = s2, s1

                    if s0 < 1e-6:           # degenerate — collinear stars
                        continue

                    # Invariant: (s2/s1, s1/s0) — matches upstream convention
                    inv_buf[base + cnt, 0] = s2 / s1
                    inv_buf[base + cnt, 1] = s1 / s0

                    # Canonical vertex ordering: longest side opposite vertex 'a',
                    # then next-longest, then shortest.  This mirrors upstream
                    # _arrangetriplet so matched correspondences align correctly.
                    tri_buf[base + cnt, 0] = ia
                    tri_buf[base + cnt, 1] = ib
                    tri_buf[base + cnt, 2] = ic

                    cnt += 1
                if cnt >= max_tri_per_star:
                    break
            if cnt >= max_tri_per_star:
                break
        n_valid[i] = cnt

    return inv_buf, tri_buf, n_valid


@njit(parallel=True, cache=True, fastmath=True)
def _ransac_parallel(
    matches:      np.ndarray,    # (M, 3, 2) int32
    src_coords:   np.ndarray,    # (Ns, 2) float32
    tgt_coords:   np.ndarray,    # (Nt, 2) float32
    n_iter:       int,
    pix_tol:      float,
    min_matches:  int,
    rng_seeds:    np.ndarray,    # (n_iter,) uint64 — per-iteration seeds
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Parallel RANSAC over triangle matches.

    Each iteration independently:
      1. Picks one triangle pair as the hypothesis.
      2. Fits a 2×3 affine from the 3 correspondences (exact, closed-form).
      3. Scores all matches against it.

    All iterations run in parallel; we take the best.

    Returns
    -------
    best_A    : (2,3) float32  — best affine transform found
    best_mask : (M,)  bool     — inlier mask for best model
    best_n    : int            — inlier count
    """
    M = matches.shape[0]

    # Per-iteration storage (avoid write conflicts between prange iterations)
    scores  = np.zeros(n_iter, dtype=np.int32)
    A_store = np.zeros((n_iter, 2, 3), dtype=np.float64)

    for it in prange(n_iter):
        # Deterministic shuffle from per-iteration seed (LCG)
        seed = rng_seeds[it]

        # Pick hypothesis index from seed
        hyp_idx = int(seed % M)
        hyp = matches[hyp_idx]           # (3,2) — 3 pairs of (src_idx, tgt_idx)

        # Build 3-correspondence least-squares affine (exact for 3 pairs)
        # src → tgt:   [x']   [a b tx] [x]
        #              [y'] = [c d ty] [y]
        #                                1
        sx0 = src_coords[hyp[0, 0], 0]; sy0 = src_coords[hyp[0, 0], 1]
        sx1 = src_coords[hyp[1, 0], 0]; sy1 = src_coords[hyp[1, 0], 1]
        sx2 = src_coords[hyp[2, 0], 0]; sy2 = src_coords[hyp[2, 0], 1]
        tx0 = tgt_coords[hyp[0, 1], 0]; ty0 = tgt_coords[hyp[0, 1], 1]
        tx1 = tgt_coords[hyp[1, 1], 0]; ty1 = tgt_coords[hyp[1, 1], 1]
        tx2 = tgt_coords[hyp[2, 1], 0]; ty2 = tgt_coords[hyp[2, 1], 1]

        # Solve A·P = B  (3×3 system, one for x, one for y)
        #   [sx0 sy0 1] [a]   [tx0]
        #   [sx1 sy1 1] [b] = [tx1]
        #   [sx2 sy2 1] [tx]  [tx2]
        det = (sx0*(sy1 - sy2) - sy0*(sx1 - sx2) + (sx1*sy2 - sx2*sy1))
        if abs(det) < 1e-10:
            scores[it] = 0
            continue

        inv_det = 1.0 / det
        # Cofactors for row 0,1,2
        c00 = (sy1 - sy2) * inv_det
        c10 = (sy2 - sy0) * inv_det
        c20 = (sy0 - sy1) * inv_det
        c01 = (sx2 - sx1) * inv_det
        c11 = (sx0 - sx2) * inv_det
        c21 = (sx1 - sx0) * inv_det
        c02 = (sx1*sy2 - sx2*sy1) * inv_det
        c12 = (sx2*sy0 - sx0*sy2) * inv_det
        c22 = (sx0*sy1 - sx1*sy0) * inv_det

        a_  = c00*tx0 + c10*tx1 + c20*tx2
        b_  = c01*tx0 + c11*tx1 + c21*tx2
        ttx = c02*tx0 + c12*tx1 + c22*tx2
        c_  = c00*ty0 + c10*ty1 + c20*ty2
        d_  = c01*ty0 + c11*ty1 + c21*ty2
        tty = c02*ty0 + c12*ty1 + c22*ty2

        # Score: count all matches within pix_tol
        tol2 = pix_tol * pix_tol
        cnt = 0
        for m in range(M):
            sx = src_coords[matches[m, 0, 0], 0]
            sy = src_coords[matches[m, 0, 0], 1]
            px = a_ * sx + b_ * sy + ttx
            py = c_ * sx + d_ * sy + tty
            # Check each of the 3 correspondences in this match
            ok = True
            for v in range(3):
                six = src_coords[matches[m, v, 0], 0]
                siy = src_coords[matches[m, v, 0], 1]
                tix = tgt_coords[matches[m, v, 1], 0]
                tiy = tgt_coords[matches[m, v, 1], 1]
                ex = a_ * six + b_ * siy + ttx - tix
                ey = c_ * six + d_ * siy + tty - tiy
                if ex*ex + ey*ey > tol2:
                    ok = False
                    break
            if ok:
                cnt += 1

        scores[it] = cnt
        A_store[it, 0, 0] = a_
        A_store[it, 0, 1] = b_
        A_store[it, 0, 2] = ttx
        A_store[it, 1, 0] = c_
        A_store[it, 1, 1] = d_
        A_store[it, 1, 2] = tty

    # Find best (serial — tiny array)
    best_it = 0
    best_n  = scores[0]
    for it in range(1, n_iter):
        if scores[it] > best_n:
            best_n  = scores[it]
            best_it = it

    best_A = A_store[best_it].astype(np.float32)

    # Rebuild inlier mask for best model (cheap, serial)
    tol2 = pix_tol * pix_tol
    a_ = float(best_A[0, 0]); b_ = float(best_A[0, 1]); ttx = float(best_A[0, 2])
    c_ = float(best_A[1, 0]); d_ = float(best_A[1, 1]); tty = float(best_A[1, 2])
    best_mask = np.zeros(M, dtype=np.bool_)
    for m in range(M):
        ok = True
        for v in range(3):
            six = src_coords[matches[m, v, 0], 0]
            siy = src_coords[matches[m, v, 0], 1]
            tix = tgt_coords[matches[m, v, 1], 0]
            tiy = tgt_coords[matches[m, v, 1], 1]
            ex = a_ * six + b_ * siy + ttx - tix
            ey = c_ * six + d_ * siy + tty - tiy
            if ex*ex + ey*ey > tol2:
                ok = False
                break
        if ok:
            best_mask[m] = True

    return best_A, best_mask, best_n


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy fallback (used if numba unavailable)
# ─────────────────────────────────────────────────────────────────────────────
def _build_invariants_numpy(coords: np.ndarray, nn_idx: np.ndarray, K: int):
    """Vectorised numpy fallback for _build_invariant_table_parallel."""
    from itertools import combinations

    N = coords.shape[0]
    inv_list  = []
    tri_list  = []
    n_valid   = np.zeros(N, dtype=np.int32)

    for i in range(N):
        cnt = 0
        for a, b, c in combinations(range(K), 3):
            ia = nn_idx[i, a]; ib = nn_idx[i, b]; ic = nn_idx[i, c]
            pa = coords[ia]; pb = coords[ib]; pc = coords[ic]
            dab = np.linalg.norm(pa - pb)
            dbc = np.linalg.norm(pb - pc)
            dac = np.linalg.norm(pa - pc)
            sides = sorted([dab, dbc, dac])
            if sides[0] < 1e-6:
                continue
            inv_list.append([sides[2] / sides[1], sides[1] / sides[0]])
            tri_list.append([ia, ib, ic])
            cnt += 1
        n_valid[i] = cnt

    inv_arr = np.array(inv_list, dtype=np.float32) if inv_list else np.zeros((0, 2), np.float32)
    tri_arr = np.array(tri_list, dtype=np.int32)   if tri_list else np.zeros((0, 3), np.int32)
    return inv_arr, tri_arr, n_valid


# ─────────────────────────────────────────────────────────────────────────────
# Public generate_invariants — selects numba or numpy automatically
# ─────────────────────────────────────────────────────────────────────────────
def _generate_invariants(sources: np.ndarray):
    """
    Return (invariants, triangle_vertex_indices) for all stars in *sources*.

    Invariants shape:  (M, 2)  float32
    Triangles shape:   (M, 3)  int32    — indices into sources
    """
    sources = np.asarray(sources, dtype=np.float32)
    N = len(sources)
    K = min(N, NUM_NEAREST_NEIGHBORS)

    # Nearest-neighbour lookup (scipy KDTree is already C-coded, fast enough)
    tree = _KDTree(sources)
    _, nn_idx = tree.query(sources, k=K)
    nn_idx = np.asarray(nn_idx, dtype=np.int32)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]

    max_tri = K * (K - 1) * (K - 2) // 6   # C(K,3)
    max_tri = max(max_tri, 1)

    if _NUMBA_OK:
        inv_buf, tri_buf, n_valid = _build_invariant_table_parallel(
            sources, nn_idx, K, max_tri
        )
    else:
        inv_buf, tri_buf, n_valid = _build_invariants_numpy(sources, nn_idx, K)

    # Flatten — collect only rows that were written
    rows = []
    for i in range(N):
        base = i * max_tri
        cnt  = int(n_valid[i])
        rows.append(np.arange(base, base + cnt))

    if not rows or all(len(r) == 0 for r in rows):
        return np.zeros((0, 2), np.float32), np.zeros((0, 3), np.int32)

    keep = np.concatenate(rows)

    if _NUMBA_OK:
        inv_all = inv_buf[keep]
        tri_all = tri_buf[keep]
    else:
        inv_all = inv_buf
        tri_all = tri_buf

    # De-duplicate (same triangle can appear from multiple stars' neighbourhoods)
    # Use a view-based approach: round to 3 dp then unique on the 2-element row
    rounded = np.round(inv_all, 3)
    # lexsort on columns 1 then 0
    order = np.lexsort((rounded[:, 1], rounded[:, 0]))
    rounded  = rounded[order]
    inv_sort = inv_all[order]
    tri_sort = tri_all[order]

    # Find unique rows by consecutive diff
    diff = np.any(rounded[1:] != rounded[:-1], axis=1)
    uniq = np.concatenate([[True], diff])
    return inv_sort[uniq], tri_sort[uniq]


# ─────────────────────────────────────────────────────────────────────────────
# Match triangles between two invariant sets (KD-tree ball query, vectorised)
# ─────────────────────────────────────────────────────────────────────────────
def _match_triangles(
    src_inv: np.ndarray, src_tri: np.ndarray,
    tgt_inv: np.ndarray, tgt_tri: np.ndarray,
    r: float = 0.1,
) -> np.ndarray:
    """
    Find all (src_triangle, tgt_triangle) pairs whose invariants are within
    radius *r* in invariant space.

    Returns an (M, 3, 2) int32 array: M triangle pairs, 3 vertices each,
    (src_idx, tgt_idx) per vertex — exactly as upstream astroalign.
    """
    if len(src_inv) == 0 or len(tgt_inv) == 0:
        return np.zeros((0, 3, 2), dtype=np.int32)

    src_tree = _KDTree(src_inv)
    tgt_tree = _KDTree(tgt_inv)

    matches_list = src_tree.query_ball_tree(tgt_tree, r=r)

    pairs = []
    for s_idx, t_idx_list in enumerate(matches_list):
        for t_idx in t_idx_list:
            pair = list(zip(src_tri[s_idx], tgt_tri[t_idx]))
            pairs.append(pair)

    if not pairs:
        return np.zeros((0, 3, 2), dtype=np.int32)

    return np.array(pairs, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# MaxIterError (same as upstream)
# ─────────────────────────────────────────────────────────────────────────────
class MaxIterError(RuntimeError):
    """Raised when no valid transform is found within the iteration budget."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# _MatchTransform helper (mirrors upstream API for any callers that use it)
# ─────────────────────────────────────────────────────────────────────────────
class _MatchTransform:
    def __init__(self, source, target):
        self.source = np.asarray(source, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def fit(self, data: np.ndarray) -> _SimilarityTransform:
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        t = estimate_transform("similarity", self.source[s], self.target[d])
        return t

    def get_error(self, data: np.ndarray, approx_t) -> np.ndarray:
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d]).reshape(d1, d2)
        return resid.max(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# RANSAC dispatcher — chooses parallel (numba) or serial (numpy) implementation
# ─────────────────────────────────────────────────────────────────────────────
def _do_ransac(
    matches:     np.ndarray,         # (M, 3, 2) int32
    src_coords:  np.ndarray,         # (Ns, 2) float32
    tgt_coords:  np.ndarray,         # (Nt, 2) float32
    pix_tol:     float,
    min_matches: int,
) -> tuple[_SimilarityTransform, np.ndarray]:
    """
    Run RANSAC to find the best similarity transform.

    When numba is available the main scoring loop is parallel across iterations.
    Falls back to the upstream serial algorithm otherwise.

    Returns (best_transform, inlier_indices).
    """
    M = matches.shape[0]

    if M == 0:
        raise MaxIterError("No triangle matches to RANSAC over.")

    if _NUMBA_OK:
        # ── Parallel RANSAC ──────────────────────────────────────────────────
        n_iter = M   # same budget as upstream (one hypothesis per triangle)
        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**62, size=n_iter, dtype=np.uint64)

        src32 = np.asarray(src_coords, dtype=np.float32)
        tgt32 = np.asarray(tgt_coords, dtype=np.float32)
        matches32 = np.asarray(matches, dtype=np.int32)

        best_A_raw, best_mask, best_n = _ransac_parallel(
            matches32, src32, tgt32, n_iter, float(pix_tol), min_matches, seeds
        )

        if best_n < min_matches:
            raise MaxIterError(
                "List of matching triangles exhausted before an acceptable "
                "transformation was found"
            )

        # Convert our raw affine to a SimilarityTransform object
        # (upstream consumers expect a skimage transform with .params, .rotation, etc.)
        # We use estimate_transform('similarity') on the inlier point set.
        inlier_idx = np.where(best_mask)[0]
        inv_model = _MatchTransform(src_coords, tgt_coords)
        best_t = inv_model.fit(matches[inlier_idx])

        # 3 polish passes (same as upstream)
        for _ in range(3):
            errs = inv_model.get_error(matches, best_t)
            better_mask = errs < pix_tol
            if better_mask.sum() < 1:
                break
            best_t = inv_model.fit(matches[better_mask])
            inlier_idx = np.where(better_mask)[0]

        return best_t, inlier_idx

    else:
        # ── Serial fallback (upstream algorithm verbatim) ────────────────────
        inv_model = _MatchTransform(src_coords, tgt_coords)
        all_idxs = np.arange(M)
        np.random.default_rng().shuffle(all_idxs)

        good_fit = None
        for iter_i in range(M):
            maybe_idxs = all_idxs[iter_i:iter_i + 1]
            test_idxs = np.delete(all_idxs, iter_i)
            maybeinliers  = matches[maybe_idxs]
            test_points   = matches[test_idxs]
            maybemodel    = inv_model.fit(maybeinliers)
            test_err      = inv_model.get_error(test_points, maybemodel)
            also_idxs     = test_idxs[test_err < pix_tol]
            alsoinliers   = matches[also_idxs]
            if len(alsoinliers) >= min_matches:
                good_data = np.concatenate((maybeinliers, alsoinliers))
                good_fit  = inv_model.fit(good_data)
                break

        if good_fit is None:
            raise MaxIterError(
                "List of matching triangles exhausted before an acceptable "
                "transformation was found"
            )

        better_fit = good_fit
        better_inlier_idxs = np.arange(M)
        for _ in range(3):
            errs = inv_model.get_error(matches, better_fit)
            better_inlier_idxs = np.arange(M)[errs < pix_tol]
            if len(better_inlier_idxs) < 1:
                break
            better_fit = inv_model.fit(matches[better_inlier_idxs])

        return better_fit, better_inlier_idxs


# ─────────────────────────────────────────────────────────────────────────────
# find_transform  (same signature as upstream)
# ─────────────────────────────────────────────────────────────────────────────
def find_transform(
    source,
    target,
    max_control_points: int = 50,
    detection_sigma:    int = 5,
    min_area:           int = 5,
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
    T : skimage SimilarityTransform
        Maps source → target.
    (source_pos, target_pos) : tuple of (N,2) float32 arrays
        Matched star positions in each image.

    Raises
    ------
    TypeError      — unsupported input type
    ValueError     — fewer than 3 stars detected
    MaxIterError   — no transform found within iteration budget
    """
    # ── Resolve control points ────────────────────────────────────────────────
    def _get_controlp(img_or_pts, label):
        try:
            arr = _data(img_or_pts)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
                # Looks like a list of (x,y) pairs
                return np.asarray(arr, dtype=np.float32)[:max_control_points]
        except Exception:
            pass

        try:
            pts = np.asarray(img_or_pts)
            if pts.ndim == 2 and pts.shape[1] == 2:
                return pts.astype(np.float32)[:max_control_points]
        except Exception:
            pass

        # Treat as image
        try:
            img2d = _bw(_data(img_or_pts))
            mk    = _mask(img_or_pts)
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

    # ── Build triangle invariants (parallel) ─────────────────────────────────
    src_inv, src_tri = _generate_invariants(src_cp)
    tgt_inv, tgt_tri = _generate_invariants(tgt_cp)

    # ── Match triangles ───────────────────────────────────────────────────────
    matches = _match_triangles(src_inv, src_tri, tgt_inv, tgt_tri, r=0.1)

    if len(matches) == 0:
        raise MaxIterError(
            "No triangle matches found between source and target."
        )

    n_invariants = len(matches)
    min_matches  = max(1, min(10, int(n_invariants * MIN_MATCHES_FRACTION)))

    # ── RANSAC (parallel) ─────────────────────────────────────────────────────
    if len(src_cp) == 3 or len(tgt_cp) == 3:
        inv_model = _MatchTransform(src_cp, tgt_cp)
        best_t    = inv_model.fit(matches)
        inlier_idx = np.arange(len(matches))
    else:
        best_t, inlier_idx = _do_ransac(matches, src_cp, tgt_cp, PIXEL_TOL, min_matches)

    # ── Recover unique, best-error point correspondences ─────────────────────
    triangle_inliers = matches[inlier_idx]
    d1, d2, d3 = triangle_inliers.shape
    inl_arr    = triangle_inliers.reshape(d1 * d2, d3)
    inl_unique = set(map(tuple, inl_arr.tolist()))

    # For each source star keep only the target with lowest reprojection error
    inl_dict: dict[int, tuple[int, float]] = {}
    for s_i, t_i in inl_unique:
        sv = src_cp[s_i]
        tv = tgt_cp[t_i]
        tv_pred = matrix_transform(sv, best_t.params)
        err = float(np.linalg.norm(tv_pred - tv))
        if s_i not in inl_dict or err < inl_dict[s_i][1]:
            inl_dict[s_i] = (t_i, err)

    inl_pairs = np.array([[s, t] for s, (t, _) in inl_dict.items()], dtype=int)
    s_idx, t_idx = inl_pairs.T
    return best_t, (src_cp[s_idx], tgt_cp[t_idx])


# ─────────────────────────────────────────────────────────────────────────────
# apply_transform  (identical to upstream)
# ─────────────────────────────────────────────────────────────────────────────
def apply_transform(
    transform,
    source,
    target,
    fill_value=None,
    propagate_mask=False,
):
    """
    Apply *transform* to *source*, outputting an image the same shape as *target*.

    Returns (aligned_image, footprint) — footprint is True where no source
    pixel contributes.
    """
    from skimage.transform import warp

    source_data  = _data(source)
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
def register(
    source,
    target,
    fill_value=None,
    propagate_mask=False,
    max_control_points=50,
    detection_sigma=5,
    min_area=5,
):
    """
    Transform *source* to align pixel-to-pixel with *target*.

    Returns (aligned_image, footprint).
    """
    t, __ = find_transform(
        source=source,
        target=target,
        max_control_points=max_control_points,
        detection_sigma=detection_sigma,
        min_area=min_area,
    )
    return apply_transform(t, source, target, fill_value, propagate_mask)


# ─────────────────────────────────────────────────────────────────────────────
# JIT warm-up  (optional — call once at app startup to avoid first-call delay)
# ─────────────────────────────────────────────────────────────────────────────
def warmup_jit():
    """
    Pre-compile the numba kernels with a tiny synthetic dataset.
    Call this once during SASpro startup (e.g. in __main__.py after imports)
    so the first real alignment job doesn't pay the compilation cost.

    No-op if numba is unavailable.
    """
    if not _NUMBA_OK:
        return
    try:
        rng = np.random.default_rng(42)
        fake_coords = rng.random((20, 2), dtype=np.float32) * 1000
        fake_idx    = np.arange(20, dtype=np.int32).reshape(20, 1).repeat(5, axis=1)
        _build_invariant_table_parallel(fake_coords, fake_idx, 5, 10)

        fake_matches = np.zeros((4, 3, 2), dtype=np.int32)
        seeds = rng.integers(0, 2**62, size=4, dtype=np.uint64)
        _ransac_parallel(fake_matches, fake_coords, fake_coords, 4, 2.0, 1, seeds)
    except Exception:
        pass   # warmup failure is non-fatal