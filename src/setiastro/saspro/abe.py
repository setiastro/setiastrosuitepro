#saspro/abe.py — SASpro Automatic (Dynamic) Background Extraction (ADBE)
# -----------------------------------------------------------------------------
# This module migrates the SASv2 ABE functionality into SASpro with:
#   • Polynomial background model (degree 1–6)
#   • Optional RBF refinement stage (multiquadric) with smoothing
#   • Smart sample-point generation (borders, corners, quartiles) with
#     gradient-descent-to-dim-spot and bright-region avoidance
#   • User-drawn exclusion polygons directly on the preview (image-space)
#   • Non‑destructive preview, commit with undo, optional background doc
#   • Mono and RGB float workflows (expects [0..1] float domain internally)
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ---------------------------------------------------------------------------
# DEBUG: when True, an ADBE run with Multiscale Gradient Refinement enabled
# also pushes the raw median+k*MAD signal-protection mask to a new document,
# so you can eyeball exactly what the threshold is protecting (tune protect_k
# against it). Flip to False for normal use. Only affects the multiscale path.
# ---------------------------------------------------------------------------
DEBUG = False

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from PyQt6.QtCore import (Qt, QSize, QEvent, QPointF, QTimer, QSettings, QByteArray, QObject,
                          QThread, pyqtSignal)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel, QSpinBox,
    QCheckBox, QPushButton, QScrollArea, QWidget, QMessageBox, QComboBox,
    QGroupBox, QApplication, QToolBar, QToolButton, QRadioButton, QDoubleSpinBox,
    QSlider, QProgressBar, QButtonGroup, QSplitter, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QIcon
from PyQt6 import sip

from scipy.interpolate import RBFInterpolator

from .doc_manager import ImageDocument
from setiastro.saspro.legacy.numba_utils import build_poly_terms, evaluate_polynomial
from .autostretch import autostretch as hard_autostretch
from setiastro.saspro.color_space_manager import tag_qimage_with_working_color_space
from setiastro.saspro.widgets.themed_buttons import themed_toolbtn
from setiastro.saspro.abe_multiscale_gradient import multiscale_gradient_correct
from setiastro.saspro.thread_keeper import start_owned_thread
from PyQt6.QtCore import QCoreApplication


def _tr(text: str) -> str:
    return QCoreApplication.translate("ABE", text)


# =============================================================================
#                         Headless ABE Core (poly + RBF)
# =============================================================================

def _downsample_area(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    if cv2 is None:
        return img[::scale, ::scale] if img.ndim == 2 else img[::scale, ::scale, :]
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Speed helpers. Each one returns exactly (bit for bit) what the plain numpy
# expression it stands in for returns; they only skip redundant work. ABE's
# output must not move when these change -- verify with np.array_equal.
# ---------------------------------------------------------------------------
_MEDIAN_FAST_MIN = 1 << 20      # below ~1M elements np.median is already quick
_MEDIAN_CHUNK = 1 << 22         # elements per counting task (16 MB of float32)


def _abe_threads(n_tasks: int) -> int:
    return max(1, min(int(n_tasks), 8, os.cpu_count() or 1))


def _median_exact(a):
    """
    np.median(a) for a large float array, or for a list of arrays taken as one
    concatenated sample -- the same value, several times faster.

    The median depends only on the multiset of values, so any exact selection
    finds the same middle element(s); they are then averaged by the very np.mean
    call np.median makes. A strided sample brackets the middle ranks, one
    threaded counting pass checks the bracket, and only the bracketed ~1% of the
    data is partitioned. Small or non-float input, NaNs, and a missed bracket
    all fall back to np.median itself.
    """
    parts = [np.asarray(p) for p in a] if isinstance(a, (list, tuple)) else [np.asarray(a)]

    def _fallback():
        if len(parts) == 1:
            return np.median(parts[0])
        return np.median(np.concatenate([p.reshape(-1) for p in parts]))

    dt = parts[0].dtype
    n = int(sum(p.size for p in parts))
    if (n < _MEDIAN_FAST_MIN or dt not in (np.float32, np.float64)
            or any(p.dtype != dt for p in parts)):
        return _fallback()
    flats = [p.reshape(-1) for p in parts]
    if n % 2 == 0:
        k_lo, k_hi = n // 2 - 1, n // 2
    else:
        k_lo = k_hi = (n - 1) // 2

    step = max(1, n // 262144)
    samp = np.sort(np.concatenate([f[::step] for f in flats]))
    m = samp.size
    tasks = [(f, s) for f in flats for s in range(0, f.size, _MEDIAN_CHUNK)]

    for widen in (1, 8):
        r = (k_lo + 0.5) * m / n
        margin = widen * (6.0 * np.sqrt(m * 0.25) + 32.0)
        lo = samp[int(max(0, np.floor(r - margin)))]
        hi = samp[int(min(m - 1, np.ceil(r + margin)))]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            break

        def _count(task, lo=lo, hi=hi):
            f, s = task
            ch = f[s:s + _MEDIAN_CHUNK]
            below = ch < lo
            above = ch > hi
            keep = ~(below | above)
            return int(np.count_nonzero(below)), int(np.count_nonzero(above)), ch[keep]

        nt = _abe_threads(len(tasks))
        if nt > 1:
            with ThreadPoolExecutor(nt) as ex:
                res = list(ex.map(_count, tasks))
        else:
            res = [_count(t) for t in tasks]
        c_lo = sum(r_[0] for r_ in res)
        c_hi = sum(r_[1] for r_ in res)
        mid = np.concatenate([r_[2] for r_ in res])
        if c_lo + c_hi + mid.size != n or np.isnan(mid).any():
            break                               # NaNs present -> np.median
        if c_lo <= k_lo and k_hi < c_lo + mid.size:
            mid.partition(sorted({k_lo - c_lo, k_hi - c_lo}))
            if k_hi != k_lo:
                pair = np.array([mid[k_lo - c_lo], mid[k_hi - c_lo]], dtype=dt)
            else:
                pair = np.array([mid[k_lo - c_lo]], dtype=dt)
            return np.mean(pair, axis=0)
    return _fallback()


def _channel_planes(img: np.ndarray, n: int = 3) -> list:
    """Contiguous float32 copies of the first n channels of an HWC image."""
    if cv2 is not None and img.ndim == 3 and img.shape[2] == n == 3 and img.dtype == np.float32:
        return [np.ascontiguousarray(p) for p in cv2.split(img)]
    return [np.array(img[..., c], dtype=np.float32) for c in range(n)]


def _stack_planes(planes: list) -> np.ndarray:
    """np.stack(planes, axis=-1) -- cv2.merge is the same copy, ~2x faster."""
    if (cv2 is not None and len(planes) == 3
            and all(p.dtype == np.float32 and p.ndim == 2 for p in planes)):
        return cv2.merge(list(planes))
    return np.stack(planes, axis=-1)


def _upscale_bg(bg_small: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    oh, ow = out_shape
    if cv2 is None:
        ys = (np.linspace(0, bg_small.shape[0] - 1, oh)).astype(int)
        xs = (np.linspace(0, bg_small.shape[1] - 1, ow)).astype(int)
        if bg_small.ndim == 2:
            return bg_small[ys][:, xs]
        return np.stack([bg_small[..., c][ys][:, xs] for c in range(bg_small.shape[2])], axis=-1)
    if bg_small.ndim == 2:
        return cv2.resize(bg_small, (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    # Per-channel resize exactly as before; only the stacking changed (no
    # second full-size .astype copy).
    planes = [cv2.resize(bg_small[..., c], (ow, oh), interpolation=cv2.INTER_LANCZOS4)
              .astype(np.float32, copy=False) for c in range(bg_small.shape[2])]
    return _stack_planes(planes)


def _fit_poly_on_small(small: np.ndarray, points: np.ndarray, degree: int, patch_size: int = 15) -> np.ndarray:
    H, W = small.shape[:2]
    half = patch_size // 2
    pts = np.asarray(points, dtype=np.int32)
    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)

    A = build_poly_terms(xs.astype(np.float32), ys.astype(np.float32), degree).astype(np.float32)

    if small.ndim == 3 and small.shape[2] == 3:
        bg_small = np.zeros_like(small, dtype=np.float32)
        
        # Batch collect samples: (num_samples, 3)
        # We need N samples. z will be list of (3,) arrays
        
        # Pre-allocate Z: (N, 3)
        Z = np.zeros((len(xs), 3), dtype=np.float32)
        
        for k, (x, y) in enumerate(zip(xs, ys)):
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            y0, y1 = max(0, y - half), min(H, y + half + 1)
            # Efficiently compute median for all channels in this patch
            patch = small[y0:y1, x0:x1, :]
            Z[k] = np.median(patch, axis=(0, 1))

        # Solve once: A is (N, terms), Z is (N, 3) -> coeffs is (terms, 3)
        coeffs_all, *_ = np.linalg.lstsq(A, Z, rcond=None)
        
        # Evaluate per channel
        for c in range(3):
            # coeffs_all[:, c] gives the terms for channel c
            bg_small[..., c] = evaluate_polynomial(H, W, coeffs_all[:, c].astype(np.float32), degree)
            
        return bg_small
    else:
        z = []
        for x, y in zip(xs, ys):
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            y0, y1 = max(0, y - half), min(H, y + half + 1)
            z.append(np.median(small[y0:y1, x0:x1]))
        z = np.asarray(z, dtype=np.float32)
        coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
        return evaluate_polynomial(H, W, coeffs.astype(np.float32), degree)


def _divide_into_quartiles(image: np.ndarray):
    h, w = image.shape[:2]
    hh, ww = h // 2, w // 2
    return {
        "top_left":     (slice(0, hh),     slice(0, ww),     (0, 0)),
        "top_right":    (slice(0, hh),     slice(ww, w),     (ww, 0)),
        "bottom_left":  (slice(hh, h),     slice(0, ww),     (0, hh)),
        "bottom_right": (slice(hh, h),     slice(ww, w),     (ww, hh)),
    }


def _exclude_bright_regions(gray: np.ndarray, exclusion_fraction: float = 0.5) -> np.ndarray:
    flat = gray.ravel()
    thresh = np.percentile(flat, 100 * (1 - exclusion_fraction))
    return (gray < thresh)


def _to_luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float32)


def _gradient_descent_to_dim_spot(
    image: np.ndarray,
    x: int,
    y: int,
    max_iter: int = 60,
    patch_size: int = 15,
    exclusion_mask: np.ndarray | None = None,
    max_offset: int | None = None,
    _lum_cache: np.ndarray | None = None,
) -> tuple[int, int]:
    """
    Slide a sample seed downhill in brightness until it lands on the
    locally dimmest reachable patch — a star- and nebula-free spot
    suitable for background sampling.

    Four things the previous one-pixel-step median descent got wrong:

      1. **Star rejection.** The old score was the patch median. Median is
         robust against a few bright pixels but a dense star field or a
         saturated star PSF whose core fills >50% of the patch still drags
         the median up, so the descent stalls on top of the star. New
         score is the 25th percentile of the patch: stars live in the
         upper tail, the lower quartile tracks sky even when the patch
         straddles a bright source, so the score decreases monotonically
         as the patch slides OFF the star.

      2. **Step size.** 1-pixel neighbour steps meant moving even one
         patch-width away took `patch_size` iterations. Two-stage descent
         now: a coarse pass (step = max(2, patch_size // 3), up to 15
         iters) to escape a nearby star glow quickly, then a fine 1-px
         pass (up to max_iter - 15 iters) to settle on the local minimum.

      3. **Exclusion awareness.** The old descent didn't know about the
         user's drawn exclusion polygons — a seed near the edge of a
         masked-out galaxy could still happily descend INTO the galaxy
         along its brightness gradient. Now, if `exclusion_mask` is
         provided (True = sampling allowed; same convention as the rest
         of ABE), the descent will never step onto an excluded pixel.
         A seed that starts inside the excluded region is returned
         unchanged, and the caller's `allowed()` check drops it.

      4. **Bounded travel.** The old descent could wander arbitrarily far
         from the seed. On an image with a smooth background gradient
         (every real sky image has one — that is the WHOLE POINT of ABE)
         every seed slid downhill toward the dim corner, collapsing the
         sampler's intended spatial tiling into a lopsided cluster and
         starving the polynomial fit of coverage on the bright side. The
         new `max_offset` parameter clamps descent travel to a radius
         around the seed; `_generate_sample_points` sets this to half
         the grid-cell size, so each seed can slide OFF a local star
         but cannot drift into another cell's territory.

    `_lum_cache` lets batch callers (notably `_generate_sample_points`)
    pass in `_to_luminance(image)` once instead of re-computing it on
    every seed. Noticeable speedup when num_points is in the hundreds.
    """
    half = patch_size // 2
    lum = _lum_cache if _lum_cache is not None else (
        _to_luminance(image) if image.ndim == 3
        else image.astype(np.float32, copy=False)
    )
    H, W = lum.shape

    sx0, sy0 = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))

    def _in_bounds(px: int, py: int) -> bool:
        return 0 <= px < W and 0 <= py < H

    def _in_radius(px: int, py: int) -> bool:
        if max_offset is None:
            return True
        return abs(px - sx0) <= max_offset and abs(py - sy0) <= max_offset

    def _mask_ok(px: int, py: int) -> bool:
        if exclusion_mask is None:
            return True
        return bool(exclusion_mask[py, px])

    def patch_score(px: int, py: int) -> float:
        x0, x1 = max(0, px - half), min(W, px + half + 1)
        y0, y1 = max(0, py - half), min(H, py + half + 1)
        # 25th-percentile sky estimate: robust to stars/compact sources.
        return float(np.percentile(lum[y0:y1, x0:x1], 25.0))

    cx, cy = sx0, sy0

    # Seed inside the drawn exclusion → return unchanged. The caller's
    # allowed() check treats this as "drop" (see _generate_sample_points).
    if not _mask_ok(cx, cy):
        return cx, cy

    def _descent_pass(step: int, iters: int) -> None:
        """
        One descent pass at a fixed 8-neighbour step size. Mutates cx, cy
        in the enclosing scope. Stops early when no neighbour improves.
        A neighbour is considered only if it is in-bounds, within the
        allowed max_offset radius from the seed, AND outside the
        exclusion mask — so the descent slides AROUND excluded regions
        and never leaves its grid cell.
        """
        nonlocal cx, cy
        for _ in range(iters):
            cur = patch_score(cx, cy)
            best = (cx, cy)
            best_val = cur
            for dy in (-step, 0, step):
                for dx in (-step, 0, step):
                    if dx == 0 and dy == 0:
                        continue
                    nx = cx + dx
                    ny = cy + dy
                    if (not _in_bounds(nx, ny)
                            or not _in_radius(nx, ny)
                            or not _mask_ok(nx, ny)):
                        continue
                    val = patch_score(nx, ny)
                    if val < best_val:
                        best_val = val
                        best = (nx, ny)
            if best == (cx, cy):
                break
            cx, cy = best

    coarse_step = max(2, patch_size // 3)
    coarse_iters = min(15, max(1, max_iter // 2))
    fine_iters = max(5, max_iter - coarse_iters)

    _descent_pass(coarse_step, coarse_iters)   # escape nearby bright features
    _descent_pass(1, fine_iters)               # settle on the local minimum

    return cx, cy


def _patch_p25_scores(lum: np.ndarray, positions, half: int, memo: dict) -> None:
    """
    memo[(x, y)] = float(np.percentile(lum[patch at (x, y)], 25.0)) for every
    position -- the patch_score of _gradient_descent_to_dim_spot.

    Full-size (2*half+1)^2 patches go through ONE np.percentile call along
    axis 1; per row that is the same partition and the same interpolation as
    the single call, so every value is identical. Edge-clipped patches have
    their own sizes and are scored one at a time, as before.
    """
    H, W = lum.shape
    p = 2 * half + 1
    full = []
    for px, py in positions:
        x0, x1 = max(0, px - half), min(W, px + half + 1)
        y0, y1 = max(0, py - half), min(H, py + half + 1)
        if x1 - x0 == p and y1 - y0 == p:
            full.append((px, py))
        else:
            memo[(px, py)] = float(np.percentile(lum[y0:y1, x0:x1], 25.0))
    if not full:
        return
    xs0 = np.fromiter((q[0] - half for q in full), dtype=np.intp, count=len(full))
    ys0 = np.fromiter((q[1] - half for q in full), dtype=np.intp, count=len(full))
    win = np.lib.stride_tricks.sliding_window_view(lum, (p, p))
    patches = win[ys0, xs0].reshape(len(full), p * p)
    vals = np.percentile(patches, 25.0, axis=1, overwrite_input=True)
    for q, v in zip(full, vals.tolist()):
        memo[q] = float(v)


def _descend_seeds_lockstep(
    lum: np.ndarray,
    seeds,
    patch_size: int = 15,
    exclusion_mask: np.ndarray | None = None,
    max_offset: int | None = None,
    max_iter: int = 60,
) -> list[tuple[int, int]]:
    """
    _gradient_descent_to_dim_spot for many seeds at once, with identical
    results: the same two passes, the same neighbour order and the same strict
    `<` tie-break per seed. Only the work is shared -- every patch score is
    computed once (a position's score never changes), and each iteration's new
    scores for all moving seeds are batched into one np.percentile call.
    """
    H, W = lum.shape
    half = patch_size // 2

    def _mask_ok(px: int, py: int) -> bool:
        if exclusion_mask is None:
            return True
        return bool(exclusion_mask[py, px])

    state = []          # [cx, cy, sx0, sy0] per seed
    start_live = []
    for i, (x, y) in enumerate(seeds):
        sx0, sy0 = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
        state.append([sx0, sy0, sx0, sy0])
        # Seed inside the drawn exclusion -> returned unchanged (the
        # caller's allowed() check drops it), exactly as the single descent.
        if _mask_ok(sx0, sy0):
            start_live.append(i)

    coarse_step = max(2, patch_size // 3)
    coarse_iters = min(15, max(1, max_iter // 2))
    fine_iters = max(5, max_iter - coarse_iters)
    memo: dict = {}

    for step, iters in ((coarse_step, coarse_iters), (1, fine_iters)):
        live = list(start_live)
        for _ in range(iters):
            if not live:
                break
            cands = {}
            need = []
            need_set = set()
            for i in live:
                cx, cy, sx0, sy0 = state[i]
                lst = [(cx, cy)]
                for dy in (-step, 0, step):
                    for dx in (-step, 0, step):
                        if dx == 0 and dy == 0:
                            continue
                        nx = cx + dx
                        ny = cy + dy
                        if not (0 <= nx < W and 0 <= ny < H):
                            continue
                        if max_offset is not None and (abs(nx - sx0) > max_offset
                                                       or abs(ny - sy0) > max_offset):
                            continue
                        if not _mask_ok(nx, ny):
                            continue
                        lst.append((nx, ny))
                cands[i] = lst
                for q in lst:
                    if q not in memo and q not in need_set:
                        need_set.add(q)
                        need.append(q)
            if need:
                _patch_p25_scores(lum, need, half, memo)
            moved = []
            for i in live:
                lst = cands[i]
                best = lst[0]
                best_val = memo[best]
                for q in lst[1:]:
                    val = memo[q]
                    if val < best_val:
                        best_val = val
                        best = q
                if best == lst[0]:
                    continue        # converged for this pass
                state[i][0], state[i][1] = best
                moved.append(i)
            live = moved

    return [(s[0], s[1]) for s in state]


def _generate_sample_points(image, num_points=100, exclusion_mask=None, patch_size=15, rng=None):
    """
    Build a well-tiled set of background sample points for the polynomial
    (and RBF) fit stage.

    The previous sampler walked the four corners + 5 border points + a
    few random picks per image quartile. That left big interior gaps on
    high-aspect fields, over-sampled corners (not where polynomial fits
    need coverage anyway), and the random quartile picks changed run to
    run even with the same seed count.

    The new sampler:

      - **Uniform grid tiling.** The image is divided into an
        aspect-ratio-aware grid sized to hit roughly `num_points` cells
        (wider images get more columns than rows). Each cell contributes
        one seed at its centre, which gives a dense, even coverage —
        exactly what a polynomial fit needs to avoid being pulled around
        by any single bright region.

      - **Exclusion-aware seeding.** A grid cell whose centre falls
        inside the drawn exclusion mask does a short spiral search for
        a valid starting point nearby — a cell straddling the edge of a
        galaxy mask still contributes a point from its sky-side half.
        A cell fully inside the exclusion zone is skipped; nothing in
        that region influences the fit.

      - **Star-robust descent.** Each seed is slid downhill by the
        upgraded `_gradient_descent_to_dim_spot` (lower-percentile score,
        two-stage step size, exclusion-aware). A seed that drops onto a
        star therefore slides off it; a seed near the edge of an
        excluded galaxy slides AWAY from the galaxy, not into it.

      - **Deduplication.** Two adjacent seeds that happen to settle on
        the same dim patch only contribute once, so a crowded local min
        doesn't get oversampled relative to the rest of the frame.

    Returns an (N, 2) int32 array of (x, y) pixel coordinates. Falls back
    to a coarse grid if exclusion is so aggressive nothing survives, so
    the downstream polynomial fit never gets a zero-point input.
    """
    H, W = image.shape[:2]
    border = 10

    # Pre-compute luminance once and share it with every descent call.
    # Avoids an _to_luminance per seed on an RGB input — on a 500 point
    # grid that's hundreds of redundant array allocations.
    lum = (_to_luminance(image) if image.ndim == 3
           else image.astype(np.float32, copy=False))

    def allowed(x: int, y: int) -> bool:
        if exclusion_mask is None:
            return True
        return bool(exclusion_mask[min(max(0, y), H - 1), min(max(0, x), W - 1)])

    # Grid dimensions — aspect-ratio aware so a wide field doesn't get
    # over-sampled vertically. Clamp to at least 2×2 so even a tiny image
    # produces something a polynomial fit can use.
    target = max(4, int(num_points))
    aspect = max(W, 1) / max(H, 1)
    ny = max(2, int(round(np.sqrt(target / max(aspect, 1e-6)))))
    nx = max(2, int(round(ny * aspect)))

    xs = np.linspace(border, W - border - 1, nx, dtype=int)
    ys = np.linspace(border, H - border - 1, ny, dtype=int)

    # Per-seed descent radius: half the cell size (whichever is smaller).
    # This is what keeps the sampler's intended tiling intact even on an
    # image with a strong background gradient — without it, every seed
    # would slide downhill into the dim corner and the fit would starve
    # on the bright side. A seed can still move far enough to escape a
    # local star (patch_size px is a few hops with coarse step), but it
    # cannot leave its own cell.
    cell_w = int((W - 2 * border) / max(nx - 1, 1))
    cell_h = int((H - 2 * border) / max(ny - 1, 1))
    cell_offset = max(patch_size, min(cell_w, cell_h) // 2)

    # Spiral offsets for finding a valid starting point when a cell
    # centre lands inside the exclusion mask. Walks outward in square
    # rings up to ~patch_size/2 — enough to clear a thin mask edge
    # without wandering into the next cell's territory.
    max_ring = max(3, patch_size // 2)
    spiral_offsets: list[tuple[int, int]] = [(0, 0)]
    for r in range(1, max_ring + 1):
        for dx in range(-r, r + 1):
            spiral_offsets.append((dx, -r))
            spiral_offsets.append((dx,  r))
        for dy in range(-r + 1, r):
            spiral_offsets.append((-r, dy))
            spiral_offsets.append(( r, dy))

    seen: set[tuple[int, int]] = set()
    pts: list[tuple[int, int]] = []

    seeds: list[tuple[int, int]] = []
    for sy in ys:
        for sx in xs:
            # Find a valid seed near this cell centre. For cells wholly
            # outside the exclusion, the first offset (0, 0) hits.
            seed = None
            for dx, dy in spiral_offsets:
                px = int(sx) + dx
                py = int(sy) + dy
                if 0 <= px < W and 0 <= py < H and allowed(px, py):
                    seed = (px, py)
                    break
            if seed is None:
                continue   # cell fully inside the exclusion mask
            seeds.append(seed)

    # Star-robust, exclusion-aware, cell-bounded descent of every seed. The
    # exclusion mask is passed through so a seed near a mask edge slides
    # AROUND the excluded region, not into it; the max_offset cap keeps the
    # seed inside its grid cell so the tiling isn't collapsed by a strong
    # background gradient. Same walk as _gradient_descent_to_dim_spot, run
    # for all seeds in lockstep so the patch scores are batched.
    ends = _descend_seeds_lockstep(
        lum, seeds,
        patch_size=patch_size,
        exclusion_mask=exclusion_mask,
        max_offset=cell_offset,
    )

    for nx_, ny_ in ends:
        # Belt-and-suspenders: the descent should never return an
        # excluded pixel, but if a future refactor regresses that or
        # the seed itself started inside the mask, drop the point.
        if not allowed(nx_, ny_):
            continue

        key = (int(nx_), int(ny_))
        if key in seen:
            continue
        seen.add(key)
        pts.append(key)

    if len(pts) == 0:
        # Fallback: coarse exclusion-respecting grid. Keeps the function
        # from ever returning zero points, which would break the polyfit
        # downstream.
        g = int(np.sqrt(max(9, num_points)))
        xs_f = np.linspace(border, W - border - 1, g, dtype=int)
        ys_f = np.linspace(border, H - border - 1, g, dtype=int)
        pts = [(int(x), int(y)) for y in ys_f for x in xs_f if allowed(int(x), int(y))]
        if len(pts) == 0:
            # Last resort — ignore exclusion so the caller gets SOMETHING.
            # This only trips if the entire image is masked out.
            pts = [(int(x), int(y)) for y in ys_f for x in xs_f]

    return np.asarray(pts, dtype=np.int32)

def _fit_rbf_on_small(small: np.ndarray, points: np.ndarray, smooth: float = 0.1, patch_size: int = 15) -> np.ndarray:
    """
    RBF background fit using scipy.interpolate.RBFInterpolator (scipy 1.7+).
    Replaces the deprecated scipy.interpolate.Rbf.
    Runs in float32 throughout — sufficient precision for smooth gradient fitting.
    Hard-capped at 1MP to prevent OOM regardless of downsample factor.
    """
    H, W = small.shape[:2]
    half = patch_size // 2

    # Hard cap: reduce further for RBF only if still too large
    MAX_RBF_PIXELS = 1_000_000
    rbf_scale = 1
    if H * W > MAX_RBF_PIXELS:
        rbf_scale = int(np.ceil(np.sqrt(H * W / MAX_RBF_PIXELS)))
        if cv2 is not None:
            small_rbf = cv2.resize(
                small,
                (max(1, W // rbf_scale), max(1, H // rbf_scale)),
                interpolation=cv2.INTER_AREA
            ).astype(np.float32, copy=False)
        else:
            small_rbf = small[::rbf_scale, ::rbf_scale].astype(np.float32, copy=False)
    else:
        small_rbf = small.astype(np.float32, copy=False)

    H_r, W_r = small_rbf.shape[:2]

    pts = np.asarray(points, dtype=np.int32)
    xs = np.clip(np.round(pts[:, 0] / rbf_scale), 0, W_r - 1).astype(np.float32)
    ys = np.clip(np.round(pts[:, 1] / rbf_scale), 0, H_r - 1).astype(np.float32)

    # RBFInterpolator expects (N, 2) array of (y, x) coordinates
    sample_coords = np.column_stack([ys, xs])  # (N, 2) float32

    # Query grid: (H_r * W_r, 2)
    grid_y, grid_x = np.mgrid[0:H_r, 0:W_r]
    query_coords = np.column_stack([
        grid_y.ravel().astype(np.float32),
        grid_x.ravel().astype(np.float32)
    ])

    def _median_patch(arr2d, x, y):
        x0, x1 = max(0, int(x) - half), min(W_r, int(x) + half + 1)
        y0, y1 = max(0, int(y) - half), min(H_r, int(y) + half + 1)
        return float(np.median(arr2d[y0:y1, x0:x1]))

    if small_rbf.ndim == 3 and small_rbf.shape[2] == 3:
        bg_r = np.zeros((H_r, W_r, 3), dtype=np.float32)

        def _fit_channel(c):
            z = np.array(
                [_median_patch(small_rbf[..., c], x, y) for x, y in zip(xs, ys)],
                dtype=np.float32
            )
            interp = RBFInterpolator(
                sample_coords, z,
                kernel='multiquadric',
                smoothing=float(smooth) * len(z),  # RBFInterpolator smoothing scales with N
                epsilon=1.0
            )
            return interp(query_coords).reshape(H_r, W_r).astype(np.float32)

        # The channels are independent fits; scipy's evaluation kernel releases
        # the GIL, so three threads run them together. Each channel is the very
        # same call as before, so the result is unchanged. A mono image comes
        # in stacked three times: identical channels give identical fits, so
        # one fit is copied.
        if (np.array_equal(small_rbf[..., 0], small_rbf[..., 1])
                and np.array_equal(small_rbf[..., 0], small_rbf[..., 2])):
            plane = _fit_channel(0)
            for c in range(3):
                bg_r[..., c] = plane
        else:
            with ThreadPoolExecutor(_abe_threads(3)) as ex:
                for c, plane in enumerate(ex.map(_fit_channel, range(3))):
                    bg_r[..., c] = plane
        return _upscale_bg(bg_r, (H, W)) if rbf_scale > 1 else bg_r
    else:
        z = np.array(
            [_median_patch(small_rbf, x, y) for x, y in zip(xs, ys)],
            dtype=np.float32
        )
        interp = RBFInterpolator(
            sample_coords, z,
            kernel='multiquadric',
            smoothing=float(smooth) * len(z),
            epsilon=1.0
        )
        bg_r = interp(query_coords).reshape(H_r, W_r).astype(np.float32)
        return _upscale_bg(bg_r, (H, W)) if rbf_scale > 1 else bg_r

def _stretch_plane(p: np.ndarray, target_median: float):
    """One channel of the legacy unlinked stretch, in place on a contiguous
    float32 plane (the per-channel body of the original loop, verbatim)."""
    m0 = float(np.min(p))
    p -= m0
    med = float(_median_exact(p))
    if med != 0.0:
        num = (med - 1.0) * target_median * p
        den = (med * (target_median + p - 1.0) - target_median * p)
        den = np.where(den == 0.0, 1e-6, den)
        np.divide(num, den, out=p)
    return p, m0, med


def _unstretch_plane(p: np.ndarray, m0: float, orig_med: float) -> np.ndarray:
    """One channel of the legacy unstretch, in place on a contiguous plane."""
    ch_med = float(_median_exact(p))
    orig_med = float(orig_med)
    if ch_med != 0.0 and orig_med != 0.0:
        num = (ch_med - 1.0) * orig_med * p
        den = (ch_med * (orig_med + p - 1.0) - orig_med * p)
        den = np.where(den == 0.0, 1e-6, den)
        np.divide(num, den, out=p)
    p += float(m0)
    return p


def _legacy_stretch_unlinked(image: np.ndarray, _channels_identical: bool = False):
    """
    Unlinked midtone stretch used for background FITTING only.

    Each channel is processed as a contiguous plane (strided per-channel
    arithmetic on an HWC array was most of the cost), the three channels in
    parallel threads. `_channels_identical` lets abe_run say "this is a mono
    image stacked three times": channel 0 is then computed once and copied,
    which is exactly what three identical computations would give.
    """
    was_single = False
    img = image
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        was_single = True
        img = img[..., 0] if img.ndim == 3 else img
        planes = [np.array(img, dtype=np.float32, copy=True)]
    elif _channels_identical:
        planes = [np.array(img[..., 0], dtype=np.float32, copy=True)]
    else:
        planes = _channel_planes(img, 3)
    target_median = 0.25

    if len(planes) == 1:
        results = [_stretch_plane(planes[0], target_median)] * 3
    else:
        with ThreadPoolExecutor(_abe_threads(3)) as ex:
            results = list(ex.map(lambda p: _stretch_plane(p, target_median), planes))

    ch_mins: list[float] = [r[1] for r in results]
    ch_meds: list[float] = [r[2] for r in results]
    out = _stack_planes([r[0] for r in results])
    if img.ndim == 3 and img.shape[2] > 3:
        # Extra channels (never used by abe_run) pass through as before.
        out = np.concatenate([out, img[..., 3:].astype(np.float32)], axis=-1)

    # *** NO np.clip here — bright stars legitimately exceed 1.0 in stretch domain ***
    return out, {"mins": ch_mins, "meds": ch_meds, "was_single": was_single}


def _legacy_unstretch_unlinked(image: np.ndarray, state: dict):
    mins = state["mins"]; meds = state["meds"]; was_single = state["was_single"]
    img = np.asarray(image)

    if was_single:
        # Only channel 0 is returned; the other two were never used.
        base = img if img.ndim == 2 else img[..., 0]
        return _unstretch_plane(np.array(base, dtype=np.float32, copy=True), mins[0], meds[0])

    # Contiguous float32 planes; a 2-D / single-channel input stands for
    # three identical channels.
    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        base = img if img.ndim == 2 else img[..., 0]
        planes = [np.array(base, dtype=np.float32, copy=True) for _ in range(3)]
    else:
        planes = _channel_planes(img, 3)

    if (mins[0] == mins[1] == mins[2] and meds[0] == meds[1] == meds[2]
            and np.array_equal(planes[0], planes[1]) and np.array_equal(planes[0], planes[2])):
        # Mono stacked x3: identical inputs give identical outputs -> once.
        p0 = _unstretch_plane(planes[0], mins[0], meds[0])
        planes = [p0, p0, p0]
    else:
        with ThreadPoolExecutor(_abe_threads(3)) as ex:
            planes = list(ex.map(lambda c: _unstretch_plane(planes[c], mins[c], meds[c]), range(3)))
    out = _stack_planes(planes)
    if img.ndim == 3 and img.shape[2] > 3:
        out = np.concatenate([out, img[..., 3:].astype(np.float32)], axis=-1)

    # *** NO np.clip here — let abe_run do the single final rescale ***
    return out

def _anchor_median_linear_rescale(img: np.ndarray, pivot: float, eps: float = 1e-8,
                                  *, _inplace: bool = False) -> np.ndarray:
    """
    Clip-free range normalization anchored at the median (pivot).
    Step 1: lift negatives (shift so min == 0, pivot adjusts accordingly).
    Step 2: compress ceiling if max > 1.0, keeping pivot fixed.
    No hard clipping — all data is preserved via linear mapping.

    `_inplace=True` (abe_run's own scratch arrays) works on `img` itself when
    it is already float32, skipping two full-size copies; values are identical.
    """
    if _inplace and isinstance(img, np.ndarray) and img.dtype == np.float32:
        out = img
    else:
        out = np.asarray(img, dtype=np.float32).copy()

    # Step 1: lift negatives
    mn = float(np.nanmin(out))
    if mn < 0.0:
        out -= mn
        pivot = float(pivot) - mn

    # Step 2: compress ceiling only if needed
    mx = float(np.nanmax(out))
    if not np.isfinite(mx) or mx <= 1.0:
        return out

    p = float(pivot) if np.isfinite(pivot) else 0.0
    denom = max(mx - p, eps)
    a = (1.0 - p) / denom
    # out = p + (out - p) * a, without two full-size temporaries (operand
    # order kept, so even NaN payloads match the expression above)
    np.subtract(out, p, out=out)
    np.multiply(out, a, out=out)
    np.add(p, out, out=out)

    return out.astype(np.float32, copy=False)

def _build_nodata_mask(img: np.ndarray,
                       pad_px: int = 4,
                       zero_tol: float = 1e-6) -> np.ndarray | None:
    """
    Detect "no-data" border pixels that come from registered / rotated frames
    (black triangular wedges after a warp, zero-fill border from cv2.warpAffine
    with BORDER_CONSTANT=0, etc.) and from NaN/inf anywhere in the frame.

    Returned mask follows the ABE sampling convention:
        True  = sample this pixel (good)
        False = DO NOT sample (no-data / infinite)

    `pad_px` dilates the no-data region slightly so the sampler also skips
    the few-pixel resampling fringe at the border where values are still
    "real" but photometrically unreliable (Lanczos ringing, bilinear
    half-pixels of zero bleeding into the first row of real sky). Dropping
    those keeps the polynomial fit from being tugged toward 0 along the
    wedge edge.

    Returns None if there are no no-data pixels at all — lets the caller
    cheap-check and skip the OR work.
    """
    if img is None:
        return None
    a = np.asarray(img)
    if a.ndim == 3:
        # No-data = all channels at/below tol, OR any channel non-finite.
        # ANY-non-finite is stricter than ALL on purpose: if even one
        # channel is NaN there, we don't want the sampler landing on it.
        zero_all = np.all(np.abs(a) <= zero_tol, axis=2)
        nonfinite_any = np.any(~np.isfinite(a), axis=2)
        nodata = zero_all | nonfinite_any
    else:
        nodata = (np.abs(a) <= zero_tol) | ~np.isfinite(a)
    if not nodata.any():
        return None
    if pad_px > 0 and cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * pad_px + 1, 2 * pad_px + 1))
        nodata = cv2.dilate(nodata.astype(np.uint8), k).astype(bool)
    # Convert "nodata" to the sampling mask (True = sample).
    return ~nodata


def abe_run(
    image: np.ndarray,
    degree: int = 2,
    num_samples: int = 100,
    downsample: int = 4,
    patch_size: int = 15,
    use_rbf: bool = True,
    rbf_smooth: float = 0.1,
    exclusion_mask: np.ndarray | None = None,
    return_background: bool = True,
    progress_cb=None,
    legacy_prestretch: bool = True,
    manual_points: np.ndarray | None = None,
    rng=None,
    correction_mode: str = "subtract",
    ignore_nodata: bool = False,
    nodata_pad_px: int = 4,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    ABE core.

    `ignore_nodata` (default False for back-compat): when True, auto-detect
    zero-valued or non-finite border pixels and treat them as excluded
    sample sites. Essential for images coming out of star alignment or
    reprojection, where the rotated frame leaves black wedges at the
    corners — otherwise the polynomial fit tries to interpolate the frame
    down to 0.0 along those wedges and drags the real sky with it. The
    detected mask is OR-ed into whatever `exclusion_mask` the caller
    passed. `nodata_pad_px` dilates the no-data region to also drop the
    few-pixel resampling fringe at the wedge edge.
    """
    if image is None:
        raise ValueError("ABE: image is None")

    img_src = np.asarray(image).astype(np.float32, copy=False)
    mono = (img_src.ndim == 2) or (img_src.ndim == 3 and img_src.shape[2] == 1)

    # Max over the finite pixels. A finite np.max means no NaN and no +inf,
    # and -inf cannot be the max, so it IS the max over the finite pixels --
    # no full-size mask and boolean copy needed in the usual all-finite case.
    _mx = np.max(img_src) if img_src.size else np.float32(np.nan)
    if np.isfinite(_mx):
        work_scale = float(_mx)
    else:
        finite_mask = np.isfinite(img_src)
        work_scale = float(np.max(img_src[finite_mask])) if np.any(finite_mask) else 1.0
        del finite_mask
    if not np.isfinite(work_scale) or work_scale <= 0.0:
        work_scale = 1.0

    img_src_work = img_src / work_scale if work_scale > 1.0 else img_src
    img_rgb = img_src_work if (img_src_work.ndim == 3 and img_src_work.shape[2] == 3) else np.stack(
        [img_src_work.squeeze()] * 3, axis=-1
    )

    # Auto no-data detection. Built on img_rgb (not img_src) so a mono input
    # that we just stacked up to 3 channels is still handled correctly. Done
    # before any stretching so a legacy_prestretch mapping NaN→huge doesn't
    # hide the no-data wedge. Then OR into any user-supplied exclusion mask.
    if ignore_nodata:
        if progress_cb: progress_cb(_tr("Detecting no-data border/wedge…"))
        auto_mask = _build_nodata_mask(img_rgb, pad_px=int(nodata_pad_px))
        if auto_mask is not None:
            if exclusion_mask is None:
                exclusion_mask = auto_mask
            else:
                try:
                    ex = np.asarray(exclusion_mask, dtype=bool)
                    if ex.shape[:2] == auto_mask.shape[:2]:
                        exclusion_mask = ex & auto_mask
                    else:
                        # Shape mismatch — can't merge safely; keep the user's.
                        if progress_cb:
                            progress_cb(_tr("No-data mask shape differs from "
                                        "user exclusion; keeping user mask."))
                except Exception:
                    pass

    # --- Stretch for background FITTING only ---
    stretch_state = None
    img_rgb_stretched = img_rgb
    if legacy_prestretch:
        img_rgb_stretched, stretch_state = _legacy_stretch_unlinked(
            img_rgb, _channels_identical=(mono and img_rgb is not img_src_work))

    orig_med = float(_median_exact(img_rgb_stretched))

    if progress_cb: progress_cb(_tr("Downsampling image…"))
    small = _downsample_area(img_rgb_stretched, downsample)
    mask_small = None
    if exclusion_mask is not None:
        if progress_cb: progress_cb(_tr("Downsampling exclusion mask…"))
        mask_small = _downsample_area(exclusion_mask.astype(np.float32), downsample) >= 0.5
    manual_small = None
    if manual_points is not None:
        mp = np.asarray(manual_points, dtype=np.int32)
        if mp.ndim == 2 and mp.shape[1] == 2 and len(mp) > 0:
            manual_small = _scale_points_to_small(mp, img_rgb.shape[:2], small.shape[:2])
            if mask_small is not None and len(manual_small) > 0:
                keep = []
                Hs, Ws = mask_small.shape[:2]
                for x, y in manual_small:
                    xx = int(np.clip(x, 0, Ws - 1))
                    yy = int(np.clip(y, 0, Hs - 1))
                    if bool(mask_small[yy, xx]):
                        keep.append((xx, yy))
                manual_small = np.asarray(keep, dtype=np.int32) if keep else np.zeros((0, 2), dtype=np.int32)

    # --- Polynomial stage ---
    if degree <= 0:
        if progress_cb: progress_cb(_tr("Degree 0: skipping polynomial stage…"))
        bg_poly_stretched = np.zeros_like(img_rgb_stretched, dtype=np.float32)
    else:
        if progress_cb: progress_cb(_tr("Sampling points (poly stage)…"))
        if manual_small is not None and len(manual_small) > 0:
            pts = manual_small
        else:
            pts = _generate_sample_points(
                small,
                num_points=num_samples,
                exclusion_mask=mask_small,
                patch_size=patch_size,
                rng=rng,
            )
        if progress_cb: progress_cb(_tr("Fitting polynomial (degree {0})…").format(degree))
        bg_poly_small = _fit_poly_on_small(small, pts, degree=degree, patch_size=patch_size)
        if progress_cb: progress_cb(_tr("Upscaling polynomial background…"))
        bg_poly_stretched = _upscale_bg(bg_poly_small, img_rgb_stretched.shape[:2])

    # --- RBF stage ---
    if use_rbf:
        # RBF refines on the poly-subtracted stretched image
        if degree > 0:
            after_poly_stretched = img_rgb_stretched - bg_poly_stretched
            med_after = float(_median_exact(after_poly_stretched))
            after_poly_stretched += (orig_med - med_after)
            after_poly_stretched = _anchor_median_linear_rescale(
                after_poly_stretched, orig_med, _inplace=True)
        else:
            after_poly_stretched = img_rgb_stretched.copy()

        if progress_cb: progress_cb(_tr("Downsampling for RBF stage…"))
        small_rbf = _downsample_area(after_poly_stretched, downsample)

        if progress_cb: progress_cb(_tr("Sampling points (RBF stage)…"))
        if manual_points is not None and len(manual_points) > 0:
            pts_rbf = _scale_points_to_small(
                np.asarray(manual_points, dtype=np.int32),
                img_rgb.shape[:2],
                small_rbf.shape[:2],
            )
            if mask_small is not None and len(pts_rbf) > 0:
                keep = []
                Hs, Ws = small_rbf.shape[:2]
                mask_rbf = mask_small if mask_small.shape[:2] == small_rbf.shape[:2] else \
                    _downsample_area(exclusion_mask.astype(np.float32), downsample) >= 0.5
                for x, y in pts_rbf:
                    xx = int(np.clip(x, 0, Ws - 1))
                    yy = int(np.clip(y, 0, Hs - 1))
                    if bool(mask_rbf[yy, xx]):
                        keep.append((xx, yy))
                pts_rbf = np.asarray(keep, dtype=np.int32) if keep else np.zeros((0, 2), dtype=np.int32)
        else:
            pts_rbf = _generate_sample_points(
                small_rbf,
                num_points=num_samples,
                exclusion_mask=mask_small,
                patch_size=patch_size,
                rng=rng,
            )

        if progress_cb: progress_cb(_tr("Fitting RBF (smooth={0:.3f})…").format(rbf_smooth))
        bg_rbf_small = _fit_rbf_on_small(small_rbf, pts_rbf, smooth=rbf_smooth, patch_size=patch_size)
        if progress_cb: progress_cb(_tr("Upscaling RBF background…"))
        bg_rbf_stretched = _upscale_bg(bg_rbf_small, img_rgb_stretched.shape[:2])

        # Total background in stretched space
        total_bg_stretched = (bg_poly_stretched + bg_rbf_stretched).astype(np.float32, copy=False)
        del bg_rbf_stretched, after_poly_stretched
    else:
        total_bg_stretched = bg_poly_stretched.astype(np.float32)

    # ---------------------------------------------------------------
    # KEY FIX: unstretch the background BEFORE subtracting,
    # then subtract in linear space from the original linear image.
    # This preserves star colors and all linear photometric relationships.
    # ---------------------------------------------------------------
    if legacy_prestretch and stretch_state is not None:
        if progress_cb: progress_cb(_tr("Unstretching background to linear domain…"))
        total_bg_linear = _legacy_unstretch_unlinked(total_bg_stretched, stretch_state)
        total_bg_linear = total_bg_linear.astype(np.float32, copy=False)
    else:
        total_bg_linear = total_bg_stretched

    del img_rgb_stretched, bg_poly_stretched
    if progress_cb: progress_cb(_tr("Applying background correction in linear space…"))
    # median(img_rgb) was computed twice (three times in divide mode); it is
    # one value, so compute it once.
    img_rgb_med = float(_median_exact(img_rgb))
    if correction_mode == "divide":
        # Normalize background to 1.0 at its median so division is stable
        bg_med = float(_median_exact(total_bg_linear))
        bg_med = bg_med if bg_med > 1e-8 else 1e-8
        bg_norm = total_bg_linear / bg_med
        bg_norm = np.where(bg_norm < 1e-8, 1e-8, bg_norm)
        corrected = img_rgb / bg_norm
        del bg_norm
        # Re-center to original median
        orig_linear_med = img_rgb_med
        corr_med = float(_median_exact(corrected))
        if corr_med > 1e-8:
            corrected *= (orig_linear_med / corr_med)
    else:
        corrected = img_rgb - total_bg_linear
        orig_linear_med = img_rgb_med
        corr_med = float(_median_exact(corrected))
        corrected += (orig_linear_med - corr_med)

    corrected = _anchor_median_linear_rescale(corrected, img_rgb_med, _inplace=True)
    corrected = corrected.astype(np.float32, copy=False)
    total_bg_linear = total_bg_linear.astype(np.float32, copy=False)

    # Restore original input scale
    if work_scale > 1.0:
        corrected *= work_scale
        total_bg_linear = total_bg_linear * work_scale

    # Squeeze mono back to 2D (a contiguous copy, so the 3-channel buffer
    # is not kept alive behind a strided view)
    if mono:
        if corrected.ndim == 3:
            corrected = np.ascontiguousarray(corrected[..., 0])
        if total_bg_linear.ndim == 3:
            total_bg_linear = np.ascontiguousarray(total_bg_linear[..., 0])

    if progress_cb: progress_cb(_tr("Ready"))
    if return_background:
        return corrected.astype(np.float32, copy=False), total_bg_linear.astype(np.float32, copy=False)
    return corrected.astype(np.float32, copy=False)



def mtf_style_autostretch(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    def stretch_channel(c):
        med = np.median(c); mad = np.median(np.abs(c - med))
        mad_std = mad * 1.4826
        mn, mx = float(c.min()), float(c.max())
        bp = max(mn, med - sigma * mad_std)
        wp = min(mx, med + 0.5*sigma * mad_std)
        if wp - bp <= 1e-8:
            return np.zeros_like(c, dtype=np.float32)
        out = (c - bp) / (wp - bp)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    if image.ndim == 2:
        return stretch_channel(image.astype(np.float32, copy=False))
    if image.ndim == 3 and image.shape[2] == 3:
        return np.stack([stretch_channel(image[..., i].astype(np.float32, copy=False))
                         for i in range(3)], axis=-1)
    raise ValueError("Unsupported image format for autostretch.")

def _scale_points_to_small(points: np.ndarray, src_hw: tuple[int, int], small_hw: tuple[int, int]) -> np.ndarray:
    """
    Scale image-space points from source resolution into downsampled 'small' resolution.
    points: Nx2 in full-image pixel coords
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    sm_h, sm_w = int(small_hw[0]), int(small_hw[1])

    sx = float(sm_w) / float(max(1, src_w))
    sy = float(sm_h) / float(max(1, src_h))

    out = np.empty_like(pts, dtype=np.int32)
    out[:, 0] = np.clip(np.round(pts[:, 0] * sx), 0, sm_w - 1).astype(np.int32)
    out[:, 1] = np.clip(np.round(pts[:, 1] * sy), 0, sm_h - 1).astype(np.int32)
    return out



# =============================================================================
#                                   UI Dialog
# =============================================================================

def _asfloat32(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)                  # zero-copy view when possible
    return a if a.dtype == np.float32 else a.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Dialog work that runs on a worker thread. Everything here reads a snapshot
# taken on the GUI thread (no widget access), so Preview / Apply / the preview
# stretch never block the window. Results are identical to the old
# synchronous code -- the same calls in the same order.
# -----------------------------------------------------------------------------

def _source_float(src) -> np.ndarray | None:
    """ABEDialog._get_source_float on a captured image array."""
    src = np.asarray(src)
    if src is None or src.size == 0:
        return None
    if np.issubdtype(src.dtype, np.integer):
        scale = float(np.iinfo(src.dtype).max)
        return src.astype(np.float32) / scale
    # float path: preserve full range; abe_run() will normalize internally if needed
    return src.astype(np.float32, copy=False)


def _resample_mask_nn(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize via integer indexing."""
    mh, mw = mask.shape[:2]
    th, tw = out_hw
    if (mh, mw) == (th, tw):
        return mask
    yi = np.linspace(0, mh - 1, th).astype(np.int32)
    xi = np.linspace(0, mw - 1, tw).astype(np.int32)
    return mask[yi][:, xi]


def _blend_mask_float(processed: np.ndarray, src: np.ndarray, mask) -> np.ndarray:
    """
    m*out + (1-m)*src in float [0..1], mono or RGB (ABEDialog's mask blend,
    with the active mask captured up front). mask=None -> processed unchanged.
    """
    if mask is None:
        return processed

    out = processed.astype(np.float32, copy=False)
    src = src.astype(np.float32, copy=False)

    # match HxW
    m = _resample_mask_nn(mask, out.shape[:2])

    # channel reconcile
    if out.ndim == 2 and src.ndim == 3:
        out = out[..., None]
    if src.ndim == 2 and out.ndim == 3:
        src = src[..., None]

    if out.ndim == 3 and out.shape[2] == 3 and m.ndim == 2:
        m = m[..., None]

    blended = (m * out + (1.0 - m) * src).astype(np.float32, copy=False)
    # squeeze back to mono if we expanded
    if blended.ndim == 3 and blended.shape[2] == 1:
        blended = blended[..., 0]
    return np.clip(blended, 0.0, 1.0)


def _prescale_qimage(qimg: QImage, scale):
    """((sw, sh), scaled QImage) exactly as ABEDialog._update_preview_scaled
    would make it at `scale`, or None. Done on the worker so the GUI thread
    only turns it into a QPixmap."""
    if scale is None or qimg is None or qimg.isNull():
        return None
    sw = max(1, int(qimg.width() * scale))
    sh = max(1, int(qimg.height() * scale))
    scaled = qimg.scaled(sw, sh, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return None if scaled.isNull() else ((sw, sh), scaled)


def _render_display_u8(src: np.ndarray, stretch: bool):
    """
    Float preview -> (uint8 backing buffer, tagged QImage). QImage is
    reentrant, so this runs on the worker; only the QPixmap is made on the GUI
    thread. Same autostretch call as before. Values are clipped to [0, 1]
    before the 8-bit cast, so out-of-range pixels no longer wrap around.
    """
    if stretch:
        disp = hard_autostretch(src, target_median=0.5, sigma=2,
                                linked=False, use_24bit=True)
        disp = np.asarray(disp, dtype=np.float32)
    else:
        disp = src

    if disp.ndim == 2 or (disp.ndim == 3 and disp.shape[2] == 1):
        mono = disp if disp.ndim == 2 else disp[..., 0]
        buf = np.ascontiguousarray((np.clip(mono, 0.0, 1.0) * 255.0).astype(np.uint8))
        h, w = buf.shape
        qimg = QImage(buf.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        buf = np.ascontiguousarray((np.clip(disp, 0.0, 1.0) * 255.0).astype(np.uint8))
        h, w, _ = buf.shape
        qimg = QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB888)
    qimg = tag_qimage_with_working_color_space(qimg)
    return buf, qimg


def _abe_dialog_compute(imgf: np.ndarray, job: dict, progress=None, status=None,
                        log=None, cancelled=None):
    """
    The dialog's background run: abe_run plus the optional multiscale stage.
    This is the former ABEDialog._run_abe body; the only changes are that the
    settings come from `job` (a GUI-thread snapshot), debug/log lines go to
    `log(kind, msg)`, and the structure / MAD maps are returned instead of
    being stored on the dialog.

    progress -- status line that may raise to cancel (a checkpoint)
    status   -- status line that never raises (used inside DarkStar callbacks)

    Returns (corrected, bg, structure_map, mad_mask).
    """
    def _dbg(msg):
        if log is not None and job.get("dbg", False):
            log("dbg", msg)

    def _log(msg):
        if log is not None:
            log("log", msg)

    if status is None:
        status = progress

    excl_mask = job.get("excl")
    structure_map = None
    mad_mask = None
    if job.get("dbg", False):
        if excl_mask is None:
            _dbg("_run_abe: excl_mask=None (polynomial + multiscale will "
                 "run on the WHOLE frame, no protected zones).")
        else:
            _dbg(f"_run_abe: excl_mask received, shape={np.asarray(excl_mask).shape}, "
                 f"dtype={np.asarray(excl_mask).dtype}, "
                 f"excluded(False)={int((~np.asarray(excl_mask, dtype=bool)).sum())} px.")

    p = job["abe"]
    seed = int(p["seed"])
    rng = np.random.default_rng(seed if seed >= 0 else None)

    corrected, bg = abe_run(
        imgf,
        degree=p["degree"],
        num_samples=p["num_samples"],
        downsample=p["downsample"],
        patch_size=p["patch_size"],
        use_rbf=p["use_rbf"],
        rbf_smooth=p["rbf_smooth"],
        exclusion_mask=excl_mask,
        return_background=True,
        progress_cb=progress,
        manual_points=p["manual_points"],
        correction_mode=p["correction_mode"],
        rng=rng,
    )

    # --- Stage 2: optional multiscale multiplicative-gradient refinement ---
    # Runs AFTER the normal (additive/smooth) background removal. Removes
    # stars first for a clean estimate, extracts the selected scale band as
    # a multiplicative gradient, and divides it out. Applied to the already-
    # corrected image so both stages compose.
    ms = job.get("ms")
    if ms is not None and corrected is not None:
        try:
            band_lo = int(ms["band_lo"])
            band_hi = int(ms["band_hi"])
            if band_hi < band_lo:
                band_lo, band_hi = band_hi, band_lo
            layers = max(band_hi + 1, 9)
            strength = float(ms["strength_pct"]) / 100.0

            # Signal-protection parameters. The "Disable signal protection"
            # escape hatch overrides the UI values with the equivalent of
            # "no mask at all" (k=inf, no dilation, no blur). This is the
            # right fix for halo-dominated frames where even k=20 still
            #sasprotects — and dilates+blurs — the bright halos, keeping them
            # out of the gradient estimate so the stage can't remove them.
            if bool(ms["disable_protect"]):
                protect_k = float("inf")
                protect_grow = 0
                sig_blur = 0.0
            else:
                protect_k = float(ms["protect_k"])   # MAD units
                protect_grow = int(ms["protect_grow"])
                sig_blur = float(ms["sigblur"])

            grad_smooth = float(ms["smooth"])
            excl_feather_frac = float(ms["feather_pct"]) / 100.0  # % -> fraction; 0 => off
            include_residual = bool(ms["residual"])
            use_darkstar = bool(ms["darkstar"])
            # Refine mode: "rgb" (default) estimates per channel so a
            # colour-selective gradient (bright-blue halo) is actually
            # visible to the band model; "luma" collapses to the pre-fix
            # single-plane estimate for narrowband or achromatic-defect
            # cases where per-channel would amplify noise.
            channels_mode = ms["channels"]

            # The protection mask is built on the STAR-INCLUSIVE image
            # (post-ADBE, pre-star-removal) so stars, star-removal residuals,
            # and real signal are all protected regardless of how well
            # DarkStar worked. The gradient BAND is still estimated on the
            # starless copy below. Snapshot the star-inclusive image now,
            # before DarkStar replaces estimate_from.
            mask_from = corrected

            # Estimate on a starless copy when DarkStar is available/enabled.
            estimate_from = corrected
            if use_darkstar:
                if progress: progress(_tr("Multiscale: removing stars (DarkStar)…"))
                try:
                    from setiastro.saspro.remove_stars import darkstar_starless_from_array
                except Exception as e:
                    import traceback
                    msg = f"Multiscale: DarkStar import FAILED: {e}"
                    if status: status(msg)
                    _log(msg + "\n" + traceback.format_exc())
                    darkstar_starless_from_array = None

                if darkstar_starless_from_array is not None:
                    try:
                        est_in = np.clip(np.asarray(corrected, dtype=np.float32), 0.0, 1.0)

                        def _ds_status(msg):
                            if status:
                                try: status(_tr("Multiscale/DarkStar: {0}").format(msg))
                                except Exception: pass
                            try: _log(f"[Multiscale/DarkStar] {msg}")
                            except Exception: pass

                        def _ds_prog(done, total, stage=""):
                            if status:
                                try: status(_tr("Multiscale/DarkStar: {0} {1}/{2}").format(stage, done, total))
                                except Exception: pass
                            # False asks DarkStar to stop (dialog closed / re-run).
                            return not (cancelled is not None and cancelled())

                        starless, _stars, _mono = darkstar_starless_from_array(
                            est_in,
                            use_gpu=True,
                            progress_cb=_ds_prog,
                            status_cb=_ds_status,
                        )
                        if starless is not None:
                            estimate_from = starless
                            if status: status(_tr("Multiscale: DarkStar starless ready."))
                            _log("[Multiscale] DarkStar star removal succeeded; "
                                 "estimating gradient on starless image.")
                        else:
                            if status: status(_tr("Multiscale: DarkStar returned None; "
                                              "estimating on corrected image."))
                    except Exception as e:
                        import traceback
                        msg = f"Multiscale: DarkStar run FAILED: {e}"
                        if status: status(msg)
                        _log(msg + "\n" + traceback.format_exc())

            # Exclusion polygons protect their region at APPLY time only:
            # the gradient runs everywhere, but these pixels are composited
            # straight from the original. ABE excl_mask is True=sample /
            # False=excluded, so the keep-original region is ~excl_mask.
            protect_blend = None
            if excl_mask is not None:
                try:
                    protect_blend = (~np.asarray(excl_mask, dtype=bool)).astype(np.float32)
                    kept = int(protect_blend.sum())
                    if not protect_blend.any():
                        _dbg("_run_abe/multiscale: protect_blend from excl_mask "
                             "is ALL ZERO (nothing to keep) → setting None.")
                        protect_blend = None
                    else:
                        _dbg(f"_run_abe/multiscale: protect_blend built — "
                             f"shape={protect_blend.shape}, keep-original px={kept} "
                             f"({100.0*kept/protect_blend.size:.2f}%).")
                except Exception as e:
                    _dbg(f"_run_abe/multiscale: protect_blend build FAILED ({e}) → None.")
                    protect_blend = None
            else:
                _dbg("_run_abe/multiscale: excl_mask is None → "
                     "protect_blend=None (galaxy will NOT be protected in refinement).")

            # Capture the multiplicative STRUCTURE map too when the user
            # asked for a structure document (troubleshooting output).
            want_structure = bool(job.get("want_structure", False))
            want_debug = bool(DEBUG)

            _pre = corrected
            _dbg(f"_run_abe/multiscale: CALLING multiscale_gradient_correct — "
                 f"protect_blend_mask={'None' if protect_blend is None else f'{int(np.asarray(protect_blend).sum())} keep px'}, "
                 f"feather_frac={excl_feather_frac:.3f}, channels={channels_mode}, "
                 f"band={band_lo}-{band_hi}, protect_k={protect_k}.")
            _ms_result = multiscale_gradient_correct(
                corrected,
                estimate_from=estimate_from,
                mask_from=mask_from,
                layers=layers,
                base_sigma=1.0,
                band_lo=band_lo,
                band_hi=band_hi,
                include_residual=include_residual,
                strength=strength,
                protect_k=protect_k,
                #sasprotect_grow was previously hard-coded to 6; it's now
                # a UI control (see sp_ms_grow). Lower values are the
                # main lever for halo-dominated frames: the former 6-px
                # dilation grew every red blob 6 px past where k alone
                # caught it, so raising k didn't shrink the protected
                # footprint anywhere near as much as users expected.
                protect_grow=protect_grow,
                protect_blur_px=sig_blur,
                protect_blend_mask=protect_blend,
                gradient_smooth_px=grad_smooth,
                protect_feather_frac=excl_feather_frac,
                channels=channels_mode,
                return_extras=want_structure,
                return_debug=want_debug,
                progress_cb=(lambda m: progress(_tr("Multiscale: {0}").format(m)) if progress else None),
            )

            # Unpack per the (return_extras, return_debug) return shapes:
            #   extras & debug -> (corr, grad, feather, mad)
            #   extras only     -> (corr, grad, feather)
            #   debug only      -> (corr, mad)
            #   neither         ->  corr
            _grad_map = None; _feather_m = None
            if want_structure and want_debug:
                corrected, _grad_map, _feather_m, _mad = _ms_result
                mad_mask = _mad
            elif want_structure:
                corrected, _grad_map, _feather_m = _ms_result
            elif want_debug:
                corrected, _mad = _ms_result
                mad_mask = _mad
            else:
                corrected = _ms_result

            if want_structure:
                try:
                    gm = np.asarray(_grad_map, dtype=np.float32)
                    # Paint the median of the structure map inside the
                    # exclusion (feathered), exactly like a mask blend:
                    #   struct = (1-m)*struct + m*median(struct)
                    # so the protected region shows flat "nothingness"
                    # instead of the raw galaxy structure — making it
                    # visually obvious it was NOT corrected.
                    if _feather_m is not None:
                        m = np.asarray(_feather_m, dtype=np.float32)
                        if m.ndim == 3:
                            m = m[..., 0]
                        if m.shape[:2] != gm.shape[:2]:
                            yi = np.linspace(0, m.shape[0]-1, gm.shape[0]).astype(np.int32)
                            xi = np.linspace(0, m.shape[1]-1, gm.shape[1]).astype(np.int32)
                            m = m[yi][:, xi]
                        m = np.clip(m, 0.0, 1.0)
                        # 3D grad map (per-channel RGB correction) needs
                        # per-plane painting: each channel has its own
                        # median (a strong blue-halo correction has a very
                        # different median in B than in R/G), and the 2D
                        # feather mask would otherwise fail to broadcast
                        # against (H, W, 3).
                        if gm.ndim == 3:
                            for c in range(gm.shape[2]):
                                g_med_c = (float(np.median(gm[..., c]))
                                           if gm[..., c].size else 1.0)
                                gm[..., c] = (gm[..., c] * (1.0 - m)
                                              + g_med_c * m)
                        else:
                            g_med = float(np.median(gm)) if gm.size else 1.0
                            gm = gm * (1.0 - m) + g_med * m
                    structure_map = gm.astype(np.float32, copy=False)
                except Exception:
                    structure_map = None
            # sanity: how much did the stage actually change the image?
            try:
                _diff = float(np.mean(np.abs(corrected.astype(np.float32) - _pre.astype(np.float32))))
                _log(
                    f"[Multiscale] stage applied: band={band_lo}-{band_hi}, "
                    f"strength={strength:.2f}, protect_k={protect_k:.1f}, "
                    f"residual={include_residual}, "
                    f"mean|Δ|={_diff:.6g} "
                    f"(estimate={'starless' if estimate_from is not _pre else 'corrected'})"
                )
            except Exception:
                pass
        except Exception as e:
            import traceback
            msg = f"Multiscale refinement FAILED: {e}"
            if status: status(msg + " — using base ABE result.")
            _log(msg + "\n" + traceback.format_exc())

    return corrected, bg, structure_map, mad_mask


def _abe_dialog_job(job: dict, progress, status, log, cancelled) -> dict:
    """One worker-thread job: 'display' (stretch the preview), 'preview' or 'apply'."""
    kind = job["kind"]
    if kind == "display":
        src = job["src"]
        if job.get("clip_source"):
            # First render of a document: the old _populate_initial_preview.
            src = np.clip(_asfloat32(_source_float(src)), 0.0, 1.0)
        buf, qimg = _render_display_u8(src, job["stretch"])
        return {"kind": kind, "job": job, "src": src, "buf": buf, "qimg": qimg,
                "pre": _prescale_qimage(qimg, job.get("view_scale"))}

    imgf = _source_float(job["src"])
    res = {"kind": kind, "job": job, "corrected": None, "bg": None,
           "structure": None, "mad": None}
    if imgf is None:
        return res
    cached = job.get("cached")
    if cached is not None:
        # Apply right after a Preview with the same settings on the same image:
        # that preview IS this result (the run is deterministic).
        corrected, bg, structure_map, mad_mask = cached
        status(_tr("Using the preview result…"))
    else:
        corrected, bg, structure_map, mad_mask = _abe_dialog_compute(
            imgf, job, progress, status, log, cancelled)
    res.update(corrected=corrected, bg=bg, structure=structure_map, mad=mad_mask)
    if corrected is None:
        return res

    if kind == "preview":
        show = bg if job["preview_bg"] else corrected
        # If previewing the corrected image, honor the active mask
        if not job["preview_bg"]:
            show = _blend_mask_float(show, imgf, job["mask"])
        progress(_tr("Rendering preview…"))
        res["show"] = show
        res["buf"], res["qimg"] = _render_display_u8(show, job["stretch"])
        res["pre"] = _prescale_qimage(res["qimg"], job.get("view_scale"))
    else:
        out = corrected
        src_img = np.asarray(job["src"])
        if out.ndim == 3 and out.shape[2] == 3 and (src_img.ndim == 2 or (src_img.ndim == 3 and src_img.shape[2] == 1)):
            out = out[..., 0]
        out_masked = _blend_mask_float(out, imgf, job["mask"])
        if cached is not None and np.may_share_memory(out_masked, corrected):
            # The document gets its own array, not the dialog's preview buffer.
            out_masked = out_masked.copy()
        res["out_masked"] = out_masked
    return res


class _ABECancelled(BaseException):
    """Raised at a job checkpoint once the dialog cancelled the job.

    BaseException on purpose: the ABE code has `except Exception` fallbacks
    (multiscale, DarkStar) that must not swallow a cancel."""


class _ABEJobThread(QThread):
    """
    Runs one dialog job off the GUI thread. No parent; started through
    thread_keeper.start_owned_thread so the dialog can close mid-run without
    a running QThread ever being destroyed. Does not shadow QThread.finished.
    """
    progress = pyqtSignal(str)
    log = pyqtSignal(str, str)
    result = pyqtSignal(object)
    failed = pyqtSignal(int, str, str)
    ended = pyqtSignal(int)         # always the last thing run() does

    def __init__(self, job: dict, seq: int = 0):
        super().__init__(None)
        self._job = job
        self._cancel = False
        # Lets the dialog ignore a late signal from a job it already replaced:
        # a thread reports isRunning() False before its queued signals land.
        self.seq = int(seq)

    def cancel(self) -> None:
        self._cancel = True
        try:
            self.requestInterruption()
        except RuntimeError:
            pass

    def is_cancelled(self) -> bool:
        return self._cancel

    def _checkpoint(self, msg) -> None:
        if self._cancel:
            raise _ABECancelled()
        self.progress.emit(str(msg))

    def _status(self, msg) -> None:
        if not self._cancel:
            self.progress.emit(str(msg))

    def run(self) -> None:
        job, self._job = self._job, None
        try:
            try:
                res = _abe_dialog_job(job, self._checkpoint, self._status,
                                      self.log.emit, self.is_cancelled)
            except _ABECancelled:
                return
            except Exception as e:
                import traceback
                self.failed.emit(self.seq, str(e), traceback.format_exc())
                return
            if not self._cancel:
                res["seq"] = self.seq
                self.result.emit(res)
        finally:
            self.ended.emit(self.seq)


class _ProtectPreviewDialog(QDialog):
    """
    Live preview of the median + k*MAD signal-protection mask.

    Shows an autostretched, downsampled copy of the image with the protected
    pixels tinted (red overlay) so the user can see exactly what each k value
    catches. The slider is debounced so dragging stays smooth; the mask is
    rebuilt on a short timer after the last movement. OK returns the chosen k.
    """
    def __init__(self, parent, img_small: np.ndarray, *, k_init=3.0,
                 grow=6, blur_px=8.0, exclusion_overlay=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Protection mask preview (median + k·MAD)"))
        self.setModal(True)
        self._img = np.clip(np.asarray(img_small, dtype=np.float32), 0.0, 1.0)
        # Hand-drawn exclusion region (1 inside drawn polygons), same HxW as the
        # preview image, composited in green. None if no polygons were drawn.
        self._excl = None
        if exclusion_overlay is not None:
            try:
                eo = np.asarray(exclusion_overlay, dtype=np.float32)
                if eo.shape[:2] == self._img.shape[:2]:
                    self._excl = np.clip(eo, 0.0, 1.0)
            except Exception:
                self._excl = None
        self._grow = int(grow)
        self._blur_px = float(blur_px)
        self._k = float(k_init)

        # Luminance for the mask + an autostretched RGB base for display.
        if self._img.ndim == 2:
            self._luma = self._img
            base = self._img
        elif self._img.ndim == 3 and self._img.shape[2] == 1:
            self._luma = self._img[..., 0]
            base = self._img[..., 0]
        else:
            self._luma = (0.2126 * self._img[..., 0] + 0.7152 * self._img[..., 1]
                          + 0.0722 * self._img[..., 2]).astype(np.float32)
            base = self._img
        try:
            disp = hard_autostretch(base, target_median=0.5, sigma=2,
                                    linked=False, use_24bit=True)
            disp = np.asarray(disp, dtype=np.float32)
        except Exception:
            disp = np.clip(base, 0.0, 1.0)
        if disp.ndim == 2:
            disp = np.repeat(disp[..., None], 3, axis=2)
        elif disp.ndim == 3 and disp.shape[2] == 1:
            disp = np.repeat(disp, 3, axis=2)
        self._disp8 = np.ascontiguousarray((np.clip(disp, 0, 1) * 255).astype(np.uint8))

        v = QVBoxLayout(self)
        self.lbl_img = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setMinimumSize(QSize(420, 300))
        v.addWidget(self.lbl_img, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel(self.tr("Protect k (MAD):")))
        self.sld = QSlider(Qt.Orientation.Horizontal)
        # 1.0 .. 20.0 in 0.1 steps. The ceiling was raised from 5.0 for
        # halo-dominated frames: when bright reflections stay well above the
        # sky even at k=5, raising k further is what lets protection respond.
        # Beyond ~20 the MAD-sigma framing stops being meaningful; use the
        # "Disable signal protection" switch in the main dialog instead.
        self.sld.setRange(10, 200)
        self.sld.setValue(int(round(self._k * 10)))
        self.sld.setSingleStep(1)
        self.sld.setPageStep(5)
        row.addWidget(self.sld, 1)
        self.lbl_k = QLabel(f"{self._k:.1f}")
        self.lbl_k.setMinimumWidth(32)
        row.addWidget(self.lbl_k)
        v.addLayout(row)

        self.lbl_stat = QLabel("")
        self.lbl_stat.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(self.lbl_stat)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_ok = QPushButton(self.tr("Use this k"))
        self.btn_cancel = QPushButton(self.tr("Cancel"))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addWidget(self.btn_ok); btns.addWidget(self.btn_cancel)
        v.addLayout(btns)

        # Debounce: rebuild the mask ~120 ms after the last slider movement.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._rebuild_mask)
        self.sld.valueChanged.connect(self._on_slider)

        self.resize(560, 460)
        self._rebuild_mask()   # initial render

    def selected_k(self) -> float:
        return self.sld.value() / 10.0

    def _on_slider(self, v):
        self.lbl_k.setText(f"{v/10.0:.1f}")
        self._debounce.start()   # restart the debounce window

    # ---- preview tunables (tweak these two) --------------------------------
    # How much lower than the chosen k the DISPLAYED threshold is drawn. The
    # applied mask is dilated+blurred so it covers more than the bare threshold;
    # showing k - this makes the red footprint read closer to reality. Does NOT
    # change the committed k (slider value / selected_k stay honest).
    PREVIEW_K_FUDGE = 1
    # Opacity of the red protection overlay, 0..1. ~0.9 = near-opaque.
    PREVIEW_OVERLAY_ALPHA = 0.90
    # ------------------------------------------------------------------------

    def _rebuild_mask(self):
        k = self.sld.value() / 10.0
        # Show the RAW median+k*MAD threshold (no dilate/blur): that's the part
        # k actually controls, so moving the slider gives obvious visual
        # feedback. The run-time dilate+blur is a fixed post-process and would
        # only wash out the k response here. The DISPLAYED threshold is nudged
        # down by PREVIEW_K_FUDGE so the footprint approximates the applied,
        # dilated+blurred coverage. Slider value / selected_k() stay honest.
        k_show = max(1.0, k - self.PREVIEW_K_FUDGE)
        med = float(np.median(self._luma))
        mad = float(np.median(np.abs(self._luma - med))) * 1.4826
        if mad <= 0:
            mad = 1e-6
        m = (self._luma > (med + k_show * mad)).astype(np.float32)
        m = np.clip(m, 0.0, 1.0)

        # Composite: base image with a near-opaque red tint where the MAD mask
        # is, then a green tint where the hand-drawn exclusion zones are (both
        # get protected at apply time; green distinguishes the drawn regions).
        out = self._disp8.astype(np.float32)
        alpha = float(self.PREVIEW_OVERLAY_ALPHA)
        a = m[..., None] * alpha
        red = np.array([230.0, 60.0, 60.0], dtype=np.float32)
        out = out * (1.0 - a) + red * a
        if self._excl is not None:
            ag = self._excl[..., None] * alpha
            green = np.array([60.0, 200.0, 90.0], dtype=np.float32)
            out = out * (1.0 - ag) + green * ag
        out = np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))

        h, w, _ = out.shape
        qimg = QImage(out.data, w, h, out.strides[0], QImage.Format.Format_RGB888)
        self._comp8 = out  # keep ref alive
        pm = QPixmap.fromImage(qimg)
        # fit to the label
        target = self.lbl_img.size()
        pm = pm.scaled(target, Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        self.lbl_img.setPixmap(pm)

        frac = float(m.mean()) * 100.0
        if self._excl is not None:
            self.lbl_stat.setText(
                self.tr("k = {0:.1f}  →  {1:.1f}% protected by MAD (red). Drawn exclusions shown in green. Lower k protects more.").format(k, frac)
            )
        else:
            self.lbl_stat.setText(
                self.tr("k = {0:.1f}  →  {1:.1f}% of pixels protected (shown in red). Lower k protects more; higher k protects only the brightest cores.").format(k, frac)
            )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # Re-fit the current composite to the new label size.
        try:
            if getattr(self, "_comp8", None) is not None:
                out = self._comp8
                h, w, _ = out.shape
                qimg = QImage(out.data, w, h, out.strides[0], QImage.Format.Format_RGB888)
                pm = QPixmap.fromImage(qimg).scaled(
                    self.lbl_img.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                self.lbl_img.setPixmap(pm)
        except Exception:
            pass


class ABEDialog(QDialog):
    """
    Non-destructive preview with polygon exclusions and optional RBF stage.
    Apply commits to the document image with undo. Optionally spawns a
    background document containing the extracted gradient.
    """
    def _make_collapsible(self, box: "QGroupBox", collapsed: bool = False,
                          settings_key: str | None = None):
        """
        Turn a QGroupBox into a collapsible section whose TITLE carries a
        disclosure triangle (▾ open / ▸ collapsed). Clicking the title bar
        toggles it; collapsing reclaims all of the group's vertical space so
        the controls column fits an HD monitor.

        Deliberately NOT a checkable groupbox — a checkbox would read as an
        enable/disable switch and mislead users into thinking they can turn
        the whole section off by unchecking it. A disclosure triangle is the
        universal "this only folds" affordance. The group's actual behaviour
        is never gated on fold state (Multiscale has its own 'Enable' box).

        settings_key: if given, the expanded/collapsed state is persisted to
        QSettings under "abe/collapse/<key>" and restored here, so the user's
        folding choices survive closing and reopening the dialog. `collapsed`
        is only the default used when no saved state exists yet.
        """
        # Remember the plain title; we prefix an arrow glyph that we flip.
        base_title = box.title()
        box._base_title = base_title           # stash for the toggler
        box._collapse_key = settings_key

        # Restore saved fold state if we have a key; else use the default.
        expanded_default = not collapsed
        if settings_key:
            saved = QSettings().value(f"abe/collapse/{settings_key}", None)
            if saved is not None:
                try:
                    expanded_default = (str(saved).lower() in ("1", "true", "yes"))
                except Exception:
                    pass
        box._expanded = expanded_default

        def _set_layout_visible(lay, visible):
            if lay is None:
                return
            for i in range(lay.count()):
                it = lay.itemAt(i)
                w = it.widget()
                if w is not None:
                    w.setVisible(visible)
                else:
                    _set_layout_visible(it.layout(), visible)

        def _apply(_box=box):
            exp = bool(_box._expanded)
            arrow = "\u25be " if exp else "\u25b8 "   # ▾ / ▸
            _box.setTitle(arrow + _box._base_title)
            _set_layout_visible(_box.layout(), exp)
            # Shrink to just the title bar when collapsed.
            _box.setMaximumHeight(16777215 if exp else _box.fontMetrics().height() + 18)

        def _toggle(_box=box):
            _box._expanded = not _box._expanded
            _apply(_box)
            key = getattr(_box, "_collapse_key", None)
            if key:
                try:
                    QSettings().setValue(f"abe/collapse/{key}", bool(_box._expanded))
                except Exception:
                    pass

        box._toggle_collapse = _toggle   # exposed for the event filter

        # Click anywhere in the title strip (top ~title-height band) toggles.
        class _TitleClickFilter(QObject):
            def __init__(self, gb):
                super().__init__(gb); self._gb = gb
            def eventFilter(self, obj, ev):
                if ev.type() == QEvent.Type.MouseButtonRelease:
                    # Only the title strip, so clicks on controls still work
                    # even though contents are hidden when collapsed.
                    title_h = self._gb.fontMetrics().height() + 10
                    try:
                        y = ev.position().toPoint().y()
                    except Exception:
                        y = ev.pos().y()
                    if y <= title_h:
                        self._gb._toggle_collapse()
                        return True
                return False

        filt = _TitleClickFilter(box)
        box.installEventFilter(filt)
        box._collapse_filter = filt   # retain against GC
        box.setCursor(Qt.CursorShape.PointingHandCursor)

        _apply(box)
        return box

    def __init__(self, parent, document: ImageDocument):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Automatic (Dynamic) Background Extraction (ADBE)"))

        # IMPORTANT: avoid “attached modal sheet” behavior on some Linux WMs
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        # Non-modal: allow user to switch between images while dialog is open
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass  # older PyQt6 versions

        self._main = parent
        self.doc = document
        self._manual_points: list[QPointF] = []
        # Points to seed on first show (set by open_abe_with_preset before show()).
        # Kept separate from _manual_points so _on_active_doc_changed's clear,
        # which can fire during the show sequence, cannot wipe them pre-seed.
        self._pending_manual_points = None

        self._connected_current_doc_changed = False
        if hasattr(self._main, "currentDocumentChanged"):
            try:
                self._main.currentDocumentChanged.connect(self._on_active_doc_changed)
                self._connected_current_doc_changed = True
            except Exception:
                self._connected_current_doc_changed = False

        self._preview_scale = 1.0
        self._preview_qimg = None
        self._last_preview = None  # backing ndarray for QImage lifetime
        self._overlay = None


        # image-space polygons: list[list[QPointF]] in ORIGINAL IMAGE COORDS
        self._polygons: list[list[QPointF]] = []
        self._drawing_poly: list[QPointF] | None = None
        # Loud exclusion-zone tracing (see _dbg). On while chasing the
        # intermittent "exclusion ignored" bug; flip to False to silence
        # all [ABE-EXCL] lines in one place.
        self._EXCL_DEBUG = False
        self._panning = False
        self._pan_last = None

        # Background jobs (see _ABEJobThread). Preview / Apply run one at a
        # time; a request made while one runs is queued, never dropped.
        self._job_thr = None            # running preview/apply thread
        self._job_kind = None           # "preview" | "apply"
        self._job_seq = 0               # seq of the current preview/apply job
        self._disp_seq = 0              # seq of the current display job
        self._thr_seq = 0
        self._pending_preview = False   # Preview clicked during a run
        self._pending_apply = None      # Apply snapshot waiting for a run
        self._abe_cache = None          # last preview result (Apply reuses it)
        self._disp_thr = None           # running preview-stretch thread
        self._disp_pending = False      # stretch requested while one ran
        self._src_gen = 0               # bumps whenever the preview source changes
        self._initial_src = None        # document image awaiting its first render
        self._fitted_once = False
        self._preview_source_f01 = None 

        # ---------------- Controls ----------------
        self.sp_degree = QSpinBox(); self.sp_degree.setRange(0, 6); self.sp_degree.setValue(2)
        self.sp_samples = QSpinBox(); self.sp_samples.setRange(20, 10000); self.sp_samples.setSingleStep(20); self.sp_samples.setValue(120)
        self.sp_down = QSpinBox()
        self.sp_down.setRange(4, 32)  # minimum 2, not 1
        self.sp_down.setValue(8)
        self.sp_down.setToolTip(
            self.tr("Downsample factor for background fitting.\n"
            "Minimum 4 recommended — factor 1 (full resolution) will\n"
            "cause extreme memory use with RBF refinement enabled.")
        )
        self.sp_patch = QSpinBox(); self.sp_patch.setRange(5, 151); self.sp_patch.setSingleStep(2); self.sp_patch.setValue(15)
        self.chk_use_rbf = QCheckBox(self.tr("Enable RBF refinement (after polynomial)")); self.chk_use_rbf.setChecked(True)
        self.sp_rbf = QSpinBox(); self.sp_rbf.setRange(0, 1000); self.sp_rbf.setValue(100)  # shown as ×0.01 below
        corr_box = QGroupBox(self.tr("Correction Mode"))
        corr_layout = QHBoxLayout(corr_box)
        self.radio_subtract = QRadioButton(self.tr("Subtract"))
        self.radio_divide   = QRadioButton(self.tr("Divide"))
        self.radio_subtract.setChecked(True)
        self.radio_subtract.setToolTip(
            self.tr("Subtract the background model from the image.\n"
            "Standard approach for most deep-sky data.")
        )
        self.radio_divide.setToolTip(
            self.tr("Divide the image by the background model (normalized to 1).\n"
            "Better for images with strong vignetting or multiplicative gradients.")
        )
        corr_layout.addWidget(self.radio_subtract)
        corr_layout.addWidget(self.radio_divide)
        corr_layout.addStretch(1)
        # --- Multiscale multiplicative-gradient refinement (dust motes,
        #     reflections, filter-edge shadows — things poly/RBF can't model) ---
        ms_box = QGroupBox(self.tr("Multiscale Gradient Refinement"))
        ms_layout = QFormLayout(ms_box)

        self.chk_ms_enable = QCheckBox(self.tr("Enable multiscale refinement"))
        self.chk_ms_enable.setChecked(False)
        self.chk_ms_enable.setToolTip(
            self.tr("After the normal background removal, run an extra pass that models\n"
            "SHARP-EDGED MULTIPLICATIVE defects — dust motes, internal reflections,\n"
            "filter-edge shadows — that polynomial/RBF fits cannot represent.\n\n"
            "Removes stars first (DarkStar) so they don't pollute the estimate,\n"
            "decomposes in log space, extracts the selected scale band as the\n"
            "gradient, and divides it out. Best run with Subtract mode above.")
        )
        ms_layout.addRow(self.chk_ms_enable)

        self.chk_ms_darkstar = QCheckBox(self.tr("Remove stars first (DarkStar)"))
        self.chk_ms_darkstar.setChecked(True)
        self.chk_ms_darkstar.setToolTip(
            self.tr("Run CosmicClarity DarkStar to get a starless image for estimating\n"
            "the gradient. Stars are kept in the final output — they just don't\n"
            "contaminate the gradient model. Strongly recommended.")
        )
        ms_layout.addRow(self.chk_ms_darkstar)

        # "Nuclear" escape hatch for frames the signal mask can't handle no
        # matter where k is set — bright halos / reflections / scatter that
        # out-sigma any reasonable threshold. When checked, protection is
        # bypassed entirely (equivalent to passing k = infinity): no pixels
        # are masked, no dilation, no inpaint, no blur. The gradient model
        # sees every pixel, including bright signal, so this will flatten
        # real nebulosity as well as the halos. Only use it when there is
        # no faint extended signal worth preserving — or when a drawn
        # exclusion region protects what you care about at apply time.
        self.chk_ms_disable_protect = QCheckBox(
            self.tr("Disable signal protection (treat all pixels as gradient)")
        )
        self.chk_ms_disable_protect.setChecked(False)
        self.chk_ms_disable_protect.setToolTip(
            self.tr("Bypass the median + k·MAD signal protection entirely. The gradient\n"
            "model then sees every pixel — stars, nebulosity, halos — and\n"
            "divides them all out.\n\n"
            "Use for halo-dominated frames where even Protect k = 20 can't catch\n"
            "the bright reflections, or when you've drawn exclusion polygons\n"
            "around anything you want preserved.\n\n"
            "WARNING: this WILL flatten real nebulosity and faint IFN. Draw\n"
            "exclusion polygons around any extended signal you want to keep,\n"
            "or leave this off and raise Protect k instead.")
        )
        ms_layout.addRow(self.chk_ms_disable_protect)

        self.sp_ms_band_lo = QSpinBox(); self.sp_ms_band_lo.setRange(0, 11); self.sp_ms_band_lo.setValue(6)
        self.sp_ms_band_hi = QSpinBox(); self.sp_ms_band_hi.setRange(0, 11); self.sp_ms_band_hi.setValue(8)
        self.sp_ms_band_lo.setToolTip(
            self.tr("First detail layer treated as gradient. Lower = finer structure.\n"
            "Dust motes / reflections / filter edges live in the mid-to-coarse\n"
            "layers. On a full-res frame, 6-8 is a good starting band; if your\n"
            "image is downscaled, shift the band 1-2 lower.")
        )
        self.sp_ms_band_hi.setToolTip(
            self.tr("Last detail layer (inclusive) treated as gradient. Higher = coarser.\n"
            "Raise toward 9-10 to also catch very broad filter-edge darkening.")
        )
        self.sp_ms_band_lo.setMaximumWidth(56)
        self.sp_ms_band_hi.setMaximumWidth(56)
        band_row = QHBoxLayout()
        band_row.setContentsMargins(0, 0, 0, 0)
        band_row.addWidget(self.sp_ms_band_lo)
        band_row.addWidget(QLabel(self.tr("to")))
        band_row.addWidget(self.sp_ms_band_hi)
        band_row.addStretch(1)
        band_wrap = QWidget(); band_wrap.setLayout(band_row)
        ms_layout.addRow(self.tr("Scale band:"), band_wrap)

        self.chk_ms_residual = QCheckBox(self.tr("Include coarsest residual"))
        self.chk_ms_residual.setChecked(False)
        self.chk_ms_residual.setToolTip(
            self.tr("Fold the coarsest (smooth) residual into the gradient too.\n"
            "Usually leave OFF — the smooth background is already handled by\n"
            "the poly/RBF stage above, and including it here risks double-\n"
            "correcting. Turn ON only if a broad multiplicative falloff remains.")
        )
        ms_layout.addRow(self.chk_ms_residual)

        self.sp_ms_strength = QSpinBox(); self.sp_ms_strength.setRange(0, 100); self.sp_ms_strength.setValue(100)
        self.sp_ms_strength.setSuffix(" %")
        self.sp_ms_strength.setMaximumWidth(80)
        self.sp_ms_strength.setToolTip(
            self.tr("Blend strength of the multiplicative correction.\n"
            "100% applies it fully; lower values ease it in if the full\n"
            "correction over-flattens real large-scale structure.")
        )

        self.sp_ms_protect = QDoubleSpinBox()
        # Ceiling raised from 5.0 → 20.0 for halo-dominated frames: bright
        # reflections can sit 20-30σ above the sky, where even k=5 still
        #sasprotects (and dilates+blurs) a huge disk around them. Values above
        # ~20 stop being meaningful in MAD-sigma units — use the "Disable
        # signal protection" checkbox above for that nuclear case.
        self.sp_ms_protect.setRange(1.0, 20.0)
        self.sp_ms_protect.setSingleStep(0.5)
        self.sp_ms_protect.setDecimals(1)
        self.sp_ms_protect.setValue(3.0)
        self.sp_ms_protect.setMaximumWidth(80)
        self.sp_ms_protect.setToolTip(
            self.tr("Signal-protection threshold: pixels brighter than\n"
            "    median + k · MAD\n"
            "are treated as signal (galaxies, nebulosity), masked out and\n"
            "inpainted so they are NOT modeled as gradient.\n\n"
            "k is in MAD-sigma units:\n"
            "  k = 1-2    protects aggressively — safest for faint signal\n"
            "  k = 3      balanced default\n"
            "  k = 4-5    protects only the brightest cores — lets more\n"
            "             extended structure into the gradient estimate\n"
            "  k = 6-20   needed on halo-dominated frames where bright\n"
            "             reflections stay above the sky at k = 5\n\n"
            "If even k = 20 still protects the halos you want removed,\n"
            "use the 'Disable signal protection' checkbox above.")
        )

        # Primary controls: Strength and Protect-k, two per row.
        ms_grid = QGridLayout()
        ms_grid.setContentsMargins(0, 0, 0, 0)
        ms_grid.setHorizontalSpacing(8)
        ms_grid.addWidget(QLabel(self.tr("Strength:")),     0, 0)
        ms_grid.addWidget(self.sp_ms_strength,              0, 1)
        ms_grid.addWidget(QLabel(self.tr("Protect k:")),    0, 2)
        ms_grid.addWidget(self.sp_ms_protect,               0, 3)
        ms_grid.setColumnStretch(4, 1)
        ms_grid_wrap = QWidget(); ms_grid_wrap.setLayout(ms_grid)
        ms_layout.addRow(ms_grid_wrap)

        self.btn_ms_preview_k = QPushButton(self.tr("Preview protection…"))
        self.btn_ms_preview_k.setToolTip(
            self.tr("Open a live preview of the median + k·MAD protection mask overlaid\n"
            "on a downsampled copy of the image. Drag the slider to see exactly\n"
            "what each k value protects, then apply it back to Protect k.")
        )
        self.btn_ms_preview_k.clicked.connect(self._open_protect_preview)
        ms_layout.addRow(self.btn_ms_preview_k)

        # --- Advanced (collapsed): the three blur knobs most users never touch.
        #     They act on three DIFFERENT things at three different stages:
        #       1. Signal mask blur  -> the median+k*MAD mask, before estimate
        #       2. Smooth gradient map -> the structure map, before dividing
        #       3. Exclusion feather -> the drawn exclusion mask, final composite
        self.sp_ms_sigblur = QSpinBox()
        self.sp_ms_sigblur.setRange(0, 50)
        self.sp_ms_sigblur.setValue(8)
        self.sp_ms_sigblur.setSuffix(" px")
        self.sp_ms_sigblur.setMaximumWidth(90)
        self.sp_ms_sigblur.setToolTip(
            self.tr("Blur applied to the median+k·MAD SIGNAL mask (after dilation),\n"
            "before the gradient is estimated. Softens the hard-edged detections\n"
            "so each star's full footprint is protected (not just its bright\n"
            "core) and the mask edges don't make the inpaint + pyramid ring into\n"
            "fake gradient.\n\n"
            "0 = off (hard mask). 8 px is a good default; raise it to spread the\n"
            "protection further around bright sources. Turn on the DEBUG mask\n"
            "document to see exactly what this is doing.")
        )

        self.sp_ms_smooth = QSpinBox()
        self.sp_ms_smooth.setRange(0, 10)
        self.sp_ms_smooth.setValue(2)
        self.sp_ms_smooth.setSuffix(" px")
        self.sp_ms_smooth.setMaximumWidth(90)
        self.sp_ms_smooth.setToolTip(
            self.tr("Blur applied to the gradient MAP (not the image) before it's\n"
            "divided out. A gradient is smooth by definition, so smoothing the\n"
            "map removes per-pixel noise the band picked up — which gets worse\n"
            "as the lower band number drops toward the fine, noise-dominated\n"
            "layers (3 and below). This strips noise from the correction while\n"
            "leaving your image's own detail untouched.\n\n"
            "0 = off. 2-3 px is a good default; raise it if a low band makes the\n"
            "background look smoothed or the exclusion area look noisy.")
        )

        self.sp_ms_feather = QSpinBox()
        self.sp_ms_feather.setRange(0, 50)
        self.sp_ms_feather.setValue(10)
        self.sp_ms_feather.setSuffix(" %")
        self.sp_ms_feather.setSpecialValueText(self.tr("Off (hard edge)"))
        self.sp_ms_feather.setMaximumWidth(140)
        self.sp_ms_feather.setToolTip(
            self.tr("Feather width for the EXCLUSION-AREA edge, as a PERCENTAGE of the\n"
            "image's short side — the Gaussian blur applied to the protect mask\n"
            "so a drawn exclusion dissolves into the corrected background with\n"
            "no hard line.\n\n"
            "Because it's a percentage, it feathers the same visual amount at\n"
            "any image size (10% ≈ 300 px on a 4500×3000 frame, ≈88 px on an\n"
            "880 px frame). 10% is a good default; lower for a tighter edge.\n"
            "0 = off (hard edge).")
        )

        # How many pixels each threshold hit is dilated out before blurring.
        # Lives here, not above, because it mostly matters for troubleshooting
        # halo-dominated frames: the previous hard-coded 6 iterations grew
        # each red blob ~6 px past where k alone caught it, so raising k
        # chased a target that `grow` kept enlarging. Dropping grow to 1-2
        # often fixes halo frames at a reasonable k; 0 uses the raw threshold.
        # Default stays at 6 for compatibility with the IFN-protecting case.
        self.sp_ms_grow = QSpinBox()
        self.sp_ms_grow.setRange(0, 50)
        self.sp_ms_grow.setValue(6)
        self.sp_ms_grow.setSuffix(" px")
        self.sp_ms_grow.setMaximumWidth(90)
        self.sp_ms_grow.setToolTip(
            self.tr("How far each threshold hit is DILATED outward (in pixels) before\n"
            "the signal mask is blurred. Dilation grows each star / signal\n"
            "detection to cover its full footprint — faint wings, not just the\n"
            "bright core.\n\n"
            "  0     use the raw median+k·MAD threshold with no growth\n"
            "  1-2   tight — good for halo-dominated frames where the mask\n"
            "        is otherwise too generous\n"
            "  6     default — covers typical star footprints\n"
            "  10+   very generous — protect extended structure around hits\n\n"
            "If raising Protect k isn't shrinking the protected region as\n"
            "much as you expect, this is why: each hit still grows outward\n"
            "by this many pixels regardless of k. Lower this first.")
        )

        # Opt-out from per-channel RGB gradient refinement. Default is Full
        # RGB (the fix for colour-selective gradients like bright-blue star
        # halos). Luma Only is the pre-fix behaviour: estimate the gradient
        # once from Rec.709 luminance and divide the same 2D map into R, G
        # and B. Keep it for the cases where per-channel is actually a step
        # backwards:
        #   1. Narrowband (SHO, HOO) where one channel is near-empty — the
        #      log of near-noise is unstable and per-channel can divide
        #      noise structure back into the result.
        #   2. Known-achromatic defects — dust motes, filter-edge shadows,
        #      baffle reflections. The gradient genuinely is wavelength-
        #      independent; luma makes one map with sqrt(3) better SNR.
        #   3. Speed-sensitive previews on huge frames (per-channel is 3x
        #      the inpaint + decompose cost).
        # The two sit next to each other in the Advanced section so the
        # choice is visible without crowding the main controls.
        self.radio_ms_rgb = QRadioButton(self.tr("Full RGB (per-channel)"))
        self.radio_ms_luma = QRadioButton(self.tr("Luma only"))
        self.radio_ms_rgb.setChecked(True)
        self.radio_ms_rgb.setToolTip(
            self.tr("Estimate the multiscale gradient per channel (R, G, B).\n"
            "Required to remove any colour-selective gradient — bright-blue\n"
            "star halos, red light-pollution cast, one-channel filter leaks.")
        )
        self.radio_ms_luma.setToolTip(
            self.tr("Estimate the gradient once from Rec.709 luminance and divide\n"
            "the same 2D map into R, G and B.\n\n"
            "Use for:\n"
            "  • narrowband data (SHO, HOO) where a near-empty channel\n"
            "    would amplify noise in per-channel mode\n"
            "  • achromatic defects (dust motes, filter edges, baffles) —\n"
            "    the gradient is the same shape in all three channels\n"
            "  • faster previews on very large frames (per-channel is 3x\n"
            "    the inpaint + decompose cost)\n\n"
            "Cannot remove colour-selective gradients — those need Full RGB.")
        )
        self._ms_channels_group = QButtonGroup(self)
        self._ms_channels_group.addButton(self.radio_ms_rgb)
        self._ms_channels_group.addButton(self.radio_ms_luma)

        ms_adv_grid = QGridLayout()
        ms_adv_grid.setContentsMargins(0, 0, 0, 0)
        ms_adv_grid.setHorizontalSpacing(8)
        #sasprotect grow sits at the top because it's the first thing to try when
        # k alone isn't shrinking the protected region (halo-dominated frames).
        ms_adv_grid.addWidget(QLabel(self.tr("Protect grow:")),        0, 0)
        ms_adv_grid.addWidget(self.sp_ms_grow,                         0, 1)
        ms_adv_grid.addWidget(QLabel(self.tr("Signal mask blur:")),    1, 0)
        ms_adv_grid.addWidget(self.sp_ms_sigblur,                      1, 1)
        ms_adv_grid.addWidget(QLabel(self.tr("Smooth gradient map:")), 2, 0)
        ms_adv_grid.addWidget(self.sp_ms_smooth,                       2, 1)
        ms_adv_grid.addWidget(QLabel(self.tr("Exclusion feather:")),   3, 0)
        ms_adv_grid.addWidget(self.sp_ms_feather,                      3, 1)
        # Refine mode goes last — this is the opt-out from per-channel RGB.
        # Default is Full RGB; narrowband + achromatic-defect users flip to
        # Luma. Two radios share one row via an HBox so the Advanced grid
        # stays two columns wide and the label aligns with the others.
        _ms_channels_row = QHBoxLayout()
        _ms_channels_row.setContentsMargins(0, 0, 0, 0)
        _ms_channels_row.setSpacing(10)
        _ms_channels_row.addWidget(self.radio_ms_rgb)
        _ms_channels_row.addWidget(self.radio_ms_luma)
        _ms_channels_row.addStretch(1)
        _ms_channels_wrap = QWidget(); _ms_channels_wrap.setLayout(_ms_channels_row)
        ms_adv_grid.addWidget(QLabel(self.tr("Refine mode:")),         4, 0)
        ms_adv_grid.addWidget(_ms_channels_wrap,                       4, 1)
        ms_adv_grid.setColumnStretch(2, 1)

        self._ms_adv_box = QGroupBox(self.tr("Advanced"))
        _ms_adv_lay = QVBoxLayout(self._ms_adv_box)
        _ms_adv_lay.setContentsMargins(8, 4, 8, 4)
        _ms_adv_wrap = QWidget(); _ms_adv_wrap.setLayout(ms_adv_grid)
        _ms_adv_lay.addWidget(_ms_adv_wrap)
        ms_layout.addRow(self._ms_adv_box)

        self.chk_make_bg_doc = QCheckBox(self.tr("Create background document")); self.chk_make_bg_doc.setChecked(False)
        self.chk_preview_bg   = QCheckBox(self.tr("Preview background instead of corrected")); self.chk_preview_bg.setChecked(False)
        self.chk_make_structure_doc = QCheckBox(self.tr("Create multiscale structure document"))
        self.chk_make_structure_doc.setChecked(False)
        self.chk_make_structure_doc.setToolTip(
            self.tr("When Multiscale Gradient Refinement is on, also open the multiplicative\n"
            "STRUCTURE map it divided out (the mote/reflection/filter-edge gradient\n"
            "only — not the smooth poly/RBF background). Useful for troubleshooting:\n"
            "it shows exactly what the multiscale stage thinks the defect is. The\n"
            "regular 'Create background document' still gives the full combined\n"
            "background from the poly/RBF stage.")
        )
        self.cmb_sample_mode = QComboBox()
        self.cmb_sample_mode.addItems(["Auto", "Manual"])

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(-1, 99999)
        self.sp_seed.setValue(42)
        self.sp_seed.setSpecialValueText(self.tr("Random (no seed)"))
        self.sp_seed.setToolTip(
            self.tr("Random seed for sample point placement.\n"
            "-1 = non-deterministic (different each run).\n"
            "Any other value = fully reproducible results.")
        )

        # ── Place Points group ──────────────────────────────────────────
        gb_place = QGroupBox(self.tr("Place Sample Points"))
        place_layout = QVBoxLayout(gb_place)

        # Grid row
        grid_row = QHBoxLayout()
        self.btn_place_grid = QPushButton(self.tr("Place Grid"))
        self.btn_place_grid.setToolTip(
            self.tr("Fill the image with a regular N×N grid of sample points.\n"
            "Switches to Manual mode so you can add/remove points before running.")
        )
        self.sp_grid_size = QSpinBox()
        self.sp_grid_size.setRange(2, 50)
        self.sp_grid_size.setValue(10)
        self.sp_grid_size.setPrefix(self.tr("Grid: "))
        self.sp_grid_size.setSuffix("×" + str(self.sp_grid_size.value()))
        self.sp_grid_size.setToolTip(self.tr("Number of grid points along each axis (N×N total)"))
        self.sp_grid_size.valueChanged.connect(
            lambda v: self.sp_grid_size.setSuffix(f"×{v}")
        )
        self._grid_size_factory = self.sp_grid_size.value()   # for Reset to Defaults
        grid_row.addWidget(self.btn_place_grid)
        grid_row.addWidget(self.sp_grid_size)
        grid_row.addStretch(1)

        # Auto points row
        self.btn_place_auto = QPushButton(self.tr("Show Auto Points"))
        self.btn_place_auto.setToolTip(
            self.tr("Run the automatic gradient-descent sampler and show all chosen points.\n"
            "Switches to Manual mode so you can tweak before running.")
        )

        place_layout.addLayout(grid_row)
        place_layout.addWidget(self.btn_place_auto)

        self.btn_clear_samples = QPushButton(self.tr("Clear Sample Points"))
        self.btn_clear_samples.clicked.connect(self._clear_manual_samples)

        self.btn_place_grid.clicked.connect(self._place_grid_points)
        self.btn_place_auto.clicked.connect(self._place_auto_points)

        # Preview area
        self.preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(QSize(480, 360))
        self.preview_label.setScaledContents(False)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Buttons
        self.btn_preview = QPushButton(self.tr("Preview"))
        self.btn_apply   = QPushButton(self.tr("Apply"))
        self.btn_close   = QPushButton(self.tr("Close"))
        self.btn_clear   = QPushButton(self.tr("Clear Exclusions"))
        # Reset all controls to their shipped defaults. Lives on the same row
        # as Clear Exclusions because both are destructive-ish housekeeping
        # buttons the user reaches for occasionally, not every run — keeping
        # them side-by-side and away from Preview/Apply reduces the chance of
        # a mis-click clobbering tuned settings mid-session.
        self.btn_reset   = QPushButton(self.tr("Reset to Defaults"))
        self.btn_reset.setToolTip(
            self.tr("Restore every ADBE control to its shipped default value "
            "(polynomial degree, sample count, multiscale refinement\n"
            "parameters, output options — everything on this dialog).\n"
            "Does NOT clear your drawn exclusion polygons; use "
            "'Clear Exclusions' for that.")
        )
        self.btn_preview.clicked.connect(self._do_preview)
        self.btn_apply.clicked.connect(self._do_apply)
        self.btn_close.clicked.connect(self.close)
        self.btn_clear.clicked.connect(self._clear_polys)
        self.btn_reset.clicked.connect(self._reset_to_defaults)

        # Layout
        params = QFormLayout()
        params.addRow(self.tr("Polynomial degree:"), self.sp_degree)
        params.addRow(self.tr("# sample points:"),   self.sp_samples)
        params.addRow(self.tr("Downsample factor:"), self.sp_down)
        params.addRow(self.tr("Patch size (px):"),   self.sp_patch)
        params.addRow(self.tr("Sample mode:"), self.cmb_sample_mode)
        params.addRow(self.tr("Random seed:"), self.sp_seed)
        # ── Workflow note ───────────────────────────────────────────────
        note_lbl = QLabel(
            self.tr("💡 Tip: For best results with RGB data, combine R/G/B channels "
            "before running ADBE. Running ADBE on individual mono channels "
            "separately can cause color artifacts at saturated stars after combining.")
        )
        note_lbl.setWordWrap(True)
        note_lbl.setStyleSheet(
            "font-size: 10px; color: palette(window-text); "
            "background: palette(base); "
            "border: 1px solid palette(mid); "
            "border-radius: 3px; padding: 4px;"
        )

        rbf_box = QGroupBox(self.tr("RBF Refinement"))
        rbf_form = QFormLayout()
        rbf_form.addRow(self.chk_use_rbf)
        rbf_form.addRow(self.tr("Smooth (x0.01):"), self.sp_rbf)
        rbf_box.setLayout(rbf_form)

        # Collapsible optional sections — folding these reclaims the vertical
        # space that was pushing the dialog past an HD monitor's height. The
        # two advanced/optional stages start collapsed; RBF (on by default)
        # starts expanded. The groups are also wrapped in a scroll area below
        # as a hard safety net.
        self._make_collapsible(gb_place, collapsed=False, settings_key="place")
        self._make_collapsible(rbf_box,  collapsed=False, settings_key="rbf")
        self._make_collapsible(corr_box, collapsed=True,  settings_key="correction")
        self._make_collapsible(ms_box,   collapsed=True,  settings_key="multiscale")
        if hasattr(self, "_ms_adv_box"):
            self._make_collapsible(self._ms_adv_box, collapsed=True,
                                   settings_key="multiscale_advanced")

        opts = QVBoxLayout()
        opts.addLayout(params)          # degree, samples, downsample, patch, sample mode
        opts.addWidget(note_lbl)
        opts.addWidget(gb_place)        # Place Grid / Show Auto Points — part of sampling setup
        opts.addWidget(self.btn_clear_samples)
        opts.addWidget(rbf_box)         # RBF refinement
        opts.addWidget(corr_box)
        opts.addWidget(ms_box)
        opts.addWidget(self.chk_make_bg_doc)
        opts.addWidget(self.chk_make_structure_doc)
        opts.addWidget(self.chk_preview_bg)

        # Housekeeping row — Clear Exclusions + Reset to Defaults — sits
        # ABOVE Preview/Apply by request. The common run flow is scroll down
        # → Preview → Apply, so Preview/Apply want to be the LAST buttons the
        # eye lands on; Clear/Reset are rarer and shouldn't be between
        # inspection controls and the fire button.
        housekeeping_row = QHBoxLayout()
        housekeeping_row.addWidget(self.btn_clear)
        housekeeping_row.addWidget(self.btn_reset)
        housekeeping_row.addStretch(1)
        opts.addLayout(housekeeping_row)

        # Preview / Apply — the "do the thing" row, kept together and placed
        # directly above the status label so progress/completion messages
        # appear right under the buttons that triggered them.
        action_row = QHBoxLayout()
        action_row.addWidget(self.btn_preview)
        action_row.addWidget(self.btn_apply)
        action_row.addStretch(1)
        opts.addLayout(action_row)

        # ▼ New status label
        self.status_label = QLabel(self.tr("Ready"))
        self.status_label.setWordWrap(True)
        opts.addWidget(self.status_label)

        opts.addStretch(1)

        # ── Drag-to-canvas grip (PI-style "new instance") ────────────────
        # After the final opts.addStretch(1) → pins to the lower-left corner.
        # Deferred import avoids an abe.py <-> shortcuts.py import cycle.
        from setiastro.saspro.shortcuts import PresetDragHandle
        from setiastro.saspro.help_support import make_help_button
        try:
            from setiastro.saspro.resources import abeicon_path
            _icon = QIcon(abeicon_path)
        except Exception:
            _icon = QIcon()

        drag_row = QHBoxLayout()
        drag_row.setContentsMargins(0, 0, 0, 0)
        self.preset_drag_handle = PresetDragHandle(
            "abe",
            self._abe_params,
            icon=_icon,
            tooltip=self.tr(
                "Drag to the canvas to create an ADBE shortcut\n"
                "with these exact settings.\n"
                "Drop directly on an image to apply them headlessly."
            ),
            parent=self,
        )
        drag_row.addWidget(self.preset_drag_handle)
        drag_row.addStretch(1)
        self.btn_help = make_help_button("abe", self)
        drag_row.addWidget(self.btn_help)
        opts.addLayout(drag_row)

        # ⬇️ Right-side stack: toolbar row ABOVE the preview. Wrapped in its
        # own QWidget so the main QSplitter below can treat it as one pane.
        right_container = QWidget()
        right = QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)
        right.addLayout(self._build_toolbar())      # Zoom In / Out / Fit / Autostretch
        right.addWidget(self.preview_scroll, 1)     # Preview below the buttons

        # Wrap the controls column in a scroll area so that even fully expanded
        # it can never exceed the window height on a small display — the user
        # scrolls the controls instead of the dialog growing off-screen.
        opts_container = QWidget()
        opts_container.setLayout(opts)
        self._opts_scroll = QScrollArea()
        self._opts_scroll.setWidgetResizable(True)
        self._opts_scroll.setWidget(opts_container)
        self._opts_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Horizontal scroll bar needs to be ALLOWED now that the user can
        # drag the controls column NARROWER than the inner form. Previously
        # the column was fixed between 320-380 px so no horizontal overflow
        # was possible; with the splitter the user can go anywhere above the
        # minimum, and a long tooltip-driven row could still clip.
        self._opts_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._opts_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Keep a sane minimum (narrower than 320 would clip spinboxes) but
        # DROP the hard maximum — that was the whole reason the user asked
        # for a splitter: when the vertical scrollbar appears it eats enough
        # width to cut off text in the collapsed-header labels and tooltips.
        # The splitter below starts the pane at the previous ~360 px width
        # so existing users see no immediate change, but can drag outward.
        self._opts_scroll.setMinimumWidth(280)

        # Replace the fixed QHBoxLayout with a QSplitter so the user can
        # drag the boundary between the controls column and the preview.
        # Rationale: when deeply-nested collapsibles expand, the controls
        # column grows a vertical scrollbar; that scrollbar steals pixels
        # from the inner form width and clips tooltips / long labels. With
        # a splitter the user can give the column a few more pixels on
        # demand instead of being stuck at the old 380 px ceiling.
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._main_splitter.addWidget(self._opts_scroll)
        self._main_splitter.addWidget(right_container)
        # Collapse-prevention: the preview pane can shrink to its minimum
        # but not vanish entirely (nobody wants a one-pixel preview column
        # from an accidental drag). The controls pane is already protected
        # by its own minimum width above.
        self._main_splitter.setCollapsible(0, False)
        self._main_splitter.setCollapsible(1, False)
        # Stretch factors: when the WINDOW is resized, the preview grows
        # and the controls stay roughly their current width. The user's
        # explicit drag is what moves the boundary.
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        # Initial sizes match the pre-splitter default (~360 px controls,
        # the rest preview). Qt scales these to the actual window width,
        # so the ratio is what matters, not the absolute numbers.
        self._main_splitter.setSizes([360, 1000])
        # Make the handle visibly draggable — the default 1-px line is
        # almost invisible, and users typically don't discover splitters
        # they can't see. 6 px is wide enough for a comfortable grab.
        self._main_splitter.setHandleWidth(6)

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self._main_splitter)

        self._load_settings()
        # the toolbar was built before the settings load: show the real state
        self.btn_autostr.setChecked(self._autostretch_on)
        if self._autostretch_on:
            self.btn_autostr.setToolTip(self.tr("Autostretch preview (On)"))

        self._base_pixmap = None  # clean, scaled image with no overlays
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_label.installEventFilter(self)
        self._install_zoom_filters()
        self._populate_initial_preview()
        self.sp_degree.valueChanged.connect(self._degree_changed) 
        self.sp_degree.valueChanged.connect(self._save_settings)
        self.sp_samples.valueChanged.connect(self._save_settings)
        self.sp_down.valueChanged.connect(self._save_settings)
        self.sp_patch.valueChanged.connect(self._save_settings)
        self.sp_seed.valueChanged.connect(self._save_settings)
        self.chk_use_rbf.toggled.connect(self._save_settings)
        self.sp_rbf.valueChanged.connect(self._save_settings)
        self.radio_subtract.toggled.connect(self._save_settings)
        self.radio_divide.toggled.connect(self._save_settings)
        self.chk_make_bg_doc.toggled.connect(self._save_settings)
        self.chk_make_structure_doc.toggled.connect(self._save_settings)
        self.chk_preview_bg.toggled.connect(self._save_settings)

        self._sample_mode_changed(self.cmb_sample_mode.currentText())

        self.cmb_sample_mode.currentTextChanged.connect(self._sample_mode_changed)
        self.cmb_sample_mode.currentTextChanged.connect(self._save_settings)

        # Multiscale refinement controls — persist on change
        if hasattr(self, "chk_ms_enable"):
            self.chk_ms_enable.toggled.connect(self._save_settings)
            self.chk_ms_darkstar.toggled.connect(self._save_settings)
            self.chk_ms_disable_protect.toggled.connect(self._save_settings)
            self.chk_ms_disable_protect.toggled.connect(self._protect_disabled_changed)
            self.sp_ms_band_lo.valueChanged.connect(self._save_settings)
            self.sp_ms_band_hi.valueChanged.connect(self._save_settings)
            self.chk_ms_residual.toggled.connect(self._save_settings)
            self.sp_ms_strength.valueChanged.connect(self._save_settings)
            self.sp_ms_protect.valueChanged.connect(self._save_settings)
            self.sp_ms_grow.valueChanged.connect(self._save_settings)
            self.sp_ms_smooth.valueChanged.connect(self._save_settings)
            self.sp_ms_feather.valueChanged.connect(self._save_settings)
            self.sp_ms_sigblur.valueChanged.connect(self._save_settings)
            # Refine mode (per-channel RGB vs luma-only). Only one radio
            # needs its signal connected — the two share a QButtonGroup, so
            # the other one's state is always the complement.
            self.radio_ms_rgb.toggled.connect(self._save_settings)

        # Apply initial enabled-state for protection controls (reflects
        # whatever _load_settings restored for chk_ms_disable_protect).
        if hasattr(self, "chk_ms_disable_protect"):
            self._protect_disabled_changed(self.chk_ms_disable_protect.isChecked())

    def _reset_to_defaults(self):
        """
        Restore every control on the dialog to its shipped default. The
        defaults mirror what _load_settings falls back to when a key is
        missing from QSettings — they are the authoritative "fresh install"
        values, not whatever happens to be in the user's registry right
        now.

        Approach: wipe every ``abe/…`` key out of QSettings and then call
        _load_settings. The getters there already encode every default; a
        second source of truth here would just drift. Confirms with the
        user first because this clobbers tuned multiscale params that may
        have taken a while to dial in.

        Does NOT clear drawn exclusion polygons (that is Clear Exclusions'
        job) or touch window geometry, so a reset doesn't yank the dialog
        back to a tiny default size while the user is working.

        Also left alone: manual sample points (user input -- with points
        placed, Sample mode stays Manual, since switching to Auto discards
        them), the preview autostretch and the sections' fold state (view
        preferences, not processing settings).
        """
        if self._job_kind == "apply" or self._pending_apply is not None:
            return                      # an Apply is running on these settings
        reply = QMessageBox.question(
            self, self.tr("Reset ADBE to defaults?"),
            self.tr(
                "Reset every control on this dialog to its default value?\n\n"
                "This affects polynomial degree, sample count, RBF settings, "
                "correction mode, multiscale refinement, and output options.\n\n"
                "Your drawn exclusion polygons will NOT be cleared "
                "(use 'Clear Exclusions' for that), and the preview "
                "window size is unchanged."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        s = QSettings()
        # Enumerate every abe/… key and remove it, so missing-key fallbacks
        # in _load_settings become the only source of values. Using remove()
        # on the group prefix clears nested keys in one shot.
        s.beginGroup("abe")
        for k in s.allKeys():
            if not k.startswith("collapse/"):   # fold state is a view preference
                s.remove(k)
        s.endGroup()
        s.sync()
        keep_stretch = getattr(self, "_autostretch_on", None)
        keep_manual = self._manual_mode() and bool(self._manual_points)
        manual_text = self.cmb_sample_mode.currentText()

        # Block the per-widget valueChanged/toggled signals during the
        # reload so we don't fire _save_settings 20 times (each write would
        # repopulate the key we just cleared with the DEFAULT, which is
        # harmless but wasteful) and so _protect_disabled_changed sees the
        # final state exactly once at the end.
        widgets_to_block = [
            getattr(self, name, None) for name in (
                "sp_degree", "sp_samples", "sp_down", "sp_patch",
                "cmb_sample_mode", "sp_seed",
                "chk_use_rbf", "sp_rbf",
                "radio_subtract", "radio_divide",
                "chk_make_bg_doc", "chk_make_structure_doc", "chk_preview_bg",
                "chk_ms_enable", "chk_ms_darkstar", "chk_ms_disable_protect",
                "sp_ms_band_lo", "sp_ms_band_hi", "chk_ms_residual",
                "sp_ms_strength", "sp_ms_protect", "sp_ms_grow",
                "sp_ms_smooth", "sp_ms_feather", "sp_ms_sigblur",
                "radio_ms_rgb", "radio_ms_luma",
            )
        ]
        prev_blocked = []
        for w in widgets_to_block:
            if w is None:
                continue
            try:
                prev_blocked.append((w, w.blockSignals(True)))
            except Exception:
                pass
        try:
            self._load_settings()
            if keep_stretch is not None:
                self._autostretch_on = keep_stretch
            if keep_manual:
                self.cmb_sample_mode.setCurrentText(manual_text)
            self.sp_grid_size.setValue(self._grid_size_factory)   # not persisted
            # Re-apply the sample-mode-dependent UI gating (clears or
            # restores manual points vs auto, etc.) that
            # _sample_mode_changed normally runs on text-changed.
            try:
                self._sample_mode_changed(self.cmb_sample_mode.currentText())
            except Exception:
                pass
            # Re-apply the protection-disabled greyout state based on the
            # just-reloaded (default: unchecked) checkbox.
            if hasattr(self, "_protect_disabled_changed") and \
                    hasattr(self, "chk_ms_disable_protect"):
                try:
                    self._protect_disabled_changed(
                        self.chk_ms_disable_protect.isChecked())
                except Exception:
                    pass
        finally:
            for w, was in prev_blocked:
                try:
                    w.blockSignals(was)
                except Exception:
                    pass

        # One canonical save at the end so the cleared keys get rewritten
        # with their default values and the next session loads them without
        # waiting for the first change.
        try:
            self._save_settings()
        except Exception:
            pass

        self._set_status(self.tr("Reset to defaults."))

    def _load_settings(self):
        s = QSettings()

        # Core ABE params
        self.sp_degree.setValue(int(s.value("abe/degree", 2)))
        self.sp_samples.setValue(int(s.value("abe/samples", 120)))
        self.sp_down.setValue(int(s.value("abe/downsample", 4)))
        self.sp_patch.setValue(int(s.value("abe/patch_size", 15)))
        sample_mode = str(s.value("abe/sample_mode", "Auto"))
        if self.cmb_sample_mode.findText(sample_mode) >= 0:
            self.cmb_sample_mode.setCurrentText(sample_mode)
        # RBF
        self.chk_use_rbf.setChecked(bool(s.value("abe/use_rbf", True, type=bool)))
        self.sp_rbf.setValue(int(s.value("abe/rbf_smooth_x100", 100)))
        divide = bool(s.value("abe/correction_divide", False, type=bool))
        if hasattr(self, "radio_divide"):
            self.radio_divide.setChecked(divide)
            self.radio_subtract.setChecked(not divide)

        if hasattr(self, "chk_ms_enable"):
            self.chk_ms_enable.setChecked(s.value("abe/ms_enable", False, type=bool))
            self.chk_ms_darkstar.setChecked(s.value("abe/ms_darkstar", True, type=bool))
            # Nuclear "disable protection" default is OFF — a user who turned
            # it on for one halo frame shouldn't quietly inherit it next
            # session on normal data, but if they explicitly saved it on, we
            # honour that.
            self.chk_ms_disable_protect.setChecked(
                s.value("abe/ms_disable_protect", False, type=bool)
            )
            self.sp_ms_band_lo.setValue(int(s.value("abe/ms_band_lo", 6)))
            self.sp_ms_band_hi.setValue(int(s.value("abe/ms_band_hi", 8)))
            self.chk_ms_residual.setChecked(s.value("abe/ms_residual", False, type=bool))
            self.sp_ms_strength.setValue(int(s.value("abe/ms_strength", 100)))
            self.sp_ms_protect.setValue(float(s.value("abe/ms_protect", 3.0)))
            #sasprotect_grow was previously hard-coded to 6. Default to that so
            # existing users see no behavioural change on upgrade.
            self.sp_ms_grow.setValue(int(s.value("abe/ms_grow", 6)))
            self.sp_ms_smooth.setValue(int(s.value("abe/ms_smooth", 2)))
            self.sp_ms_feather.setValue(int(s.value("abe/ms_feather", 10)))
            self.sp_ms_sigblur.setValue(int(s.value("abe/ms_sigblur", 8)))
            # Refine mode: "rgb" (default — per-channel, catches colour-
            # selective gradients) or "luma" (opt-out for narrowband and
            # achromatic defects). Stored as a string so the preset JSON
            # stays readable rather than a boolean nobody can decode later.
            _ms_ch = str(s.value("abe/ms_channels", "rgb")).strip().lower()
            if _ms_ch == "luma":
                self.radio_ms_luma.setChecked(True)
            else:
                self.radio_ms_rgb.setChecked(True)
        
        # Options
        self.chk_make_bg_doc.setChecked(bool(s.value("abe/make_bg_doc", False, type=bool)))
        self.chk_make_structure_doc.setChecked(bool(s.value("abe/make_structure_doc", False, type=bool)))
        self.chk_preview_bg.setChecked(bool(s.value("abe/preview_bg", False, type=bool)))
        self.sp_seed.setValue(int(s.value("abe/seed", 42)))

        # Optional preview prefs
        self._autostretch_on = bool(s.value("abe/preview_autostretch", True, type=bool))


    def _save_settings(self):
        s = QSettings()

        s.setValue("abe/degree", self.sp_degree.value())
        s.setValue("abe/samples", self.sp_samples.value())
        s.setValue("abe/downsample", self.sp_down.value())
        s.setValue("abe/patch_size", self.sp_patch.value())
        s.setValue("abe/sample_mode", self.cmb_sample_mode.currentText())
        s.setValue("abe/correction_divide", self.radio_divide.isChecked())
        if hasattr(self, "chk_ms_enable"):
            s.setValue("abe/ms_enable", self.chk_ms_enable.isChecked())
            s.setValue("abe/ms_darkstar", self.chk_ms_darkstar.isChecked())
            s.setValue("abe/ms_disable_protect", self.chk_ms_disable_protect.isChecked())
            s.setValue("abe/ms_band_lo", self.sp_ms_band_lo.value())
            s.setValue("abe/ms_band_hi", self.sp_ms_band_hi.value())
            s.setValue("abe/ms_residual", self.chk_ms_residual.isChecked())
            s.setValue("abe/ms_strength", self.sp_ms_strength.value())
            s.setValue("abe/ms_protect", self.sp_ms_protect.value())
            s.setValue("abe/ms_grow", self.sp_ms_grow.value())
            s.setValue("abe/ms_smooth", self.sp_ms_smooth.value())
            s.setValue("abe/ms_feather", self.sp_ms_feather.value())
            s.setValue("abe/ms_sigblur", self.sp_ms_sigblur.value())
            s.setValue("abe/ms_channels",
                       "luma" if self.radio_ms_luma.isChecked() else "rgb")
        s.setValue("abe/use_rbf", self.chk_use_rbf.isChecked())
        s.setValue("abe/rbf_smooth_x100", self.sp_rbf.value())
        s.setValue("abe/seed", self.sp_seed.value())
        s.setValue("abe/make_bg_doc", self.chk_make_bg_doc.isChecked())
        s.setValue("abe/make_structure_doc", self.chk_make_structure_doc.isChecked())
        s.setValue("abe/preview_bg", self.chk_preview_bg.isChecked())

        s.setValue("abe/preview_autostretch", bool(getattr(self, "_autostretch_on", False)))

    def _protect_disabled_changed(self, disabled: bool):
        """
        React to the "Disable signal protection" checkbox: grey out every
        control that only makes sense when the median+k·MAD mask is active,
        so the user is not left dragging a slider that silently does nothing.

        The disabled widgets keep their last values — the UI just reflects
        that _run_abe() is going to pass protect_k=inf, protect_grow=0 and
        protect_blur_px=0 instead of the UI values until the box is cleared
        again. The preview button is also disabled: with no mask, there is
        nothing to preview.
        """
        active = not bool(disabled)
        for w in (
            getattr(self, "sp_ms_protect", None),
            getattr(self, "sp_ms_grow", None),
            getattr(self, "sp_ms_sigblur", None),
            getattr(self, "btn_ms_preview_k", None),
        ):
            if w is not None:
                try:
                    w.setEnabled(active)
                except Exception:
                    pass
        # Give the user a line in the status strip so the state is obvious
        # even though three controls are now greyed. Only call when the
        # status label exists (not during __init__ before it is built).
        if hasattr(self, "status_label") and self.status_label is not None:
            if disabled:
                self._set_status(
                    self.tr("Signal protection DISABLED — all pixels, including stars "
                    "and nebulosity, will be treated as gradient.")
                )
            else:
                self._set_status(self.tr("Ready"))

    def _manual_mode(self) -> bool:
        return self.cmb_sample_mode.currentText().lower() == "manual"


    def _sample_mode_changed(self, _text: str):
        manual = self._manual_mode()

        if manual:
            # Switching to Manual: do NOT clear exclusion polygons. They are
            # apply-time protection (the drawn region is composited back from
            # the original in BOTH the polynomial and multiscale stages) and
            # stay valid no matter which sampling mode is active — the sampler
            # mode only governs how background sample POINTS are chosen, which
            # is orthogonal. The only things that should ever drop exclusions
            # are the explicit "Clear Exclusions" button and loading a
            # genuinely different image. Commit any half-drawn polygon first so
            # it isn't stranded, since in Manual mode left-click now places
            # sample points instead of extending the in-progress polygon.
            self._commit_pending_polygon()

        else:
            # Switching to Auto — clear manual sample points (Auto generates
            # its own). Exclusion polygons are left untouched.
            if self._manual_points:
                _log = getattr(self._main, "_log", None)
                if callable(_log):
                    _log(f"[ABE] _sample_mode_changed(Auto): CLEARING {len(self._manual_points)} pts")
                self._manual_points.clear()

        # In manual mode, # sample points is not used for fitting
        self.sp_samples.setEnabled(not manual)

        tip = self.tr(
            "Auto: ADBE picks sample points automatically.\n"
            "Manual: left-click to add sample squares, right-click to remove nearest one."
        )
        self.cmb_sample_mode.setToolTip(tip)
        self.btn_clear_samples.setEnabled(manual and len(self._manual_points) > 0)

        self._redraw_overlay()


    def _clear_manual_samples(self):
        self._manual_points.clear()
        self.btn_clear_samples.setEnabled(False)
        self._redraw_overlay()


    def _manual_points_array(self) -> np.ndarray | None:
        if not self._manual_points:
            return None
        pts = np.array([[int(round(p.x())), int(round(p.y()))] for p in self._manual_points], dtype=np.int32)
        return pts

    def seed_manual_points(self, pts) -> None:
        """
        Public: load manual sample points (full-image pixel coords) into the
        dialog, switch to Manual mode, and redraw so the yellow squares appear.

        Called from _after_show (step 5), AFTER render+fit have built the base
        pixmap. Robust to point shape: accepts [[x,y],...], [(x,y),...], an
        (N,2) array, or a flat [x0,y0,x1,y1,...]. Logs what it seeds instead of
        silently swallowing — a bare except here previously hid every failure.
        """
        log = getattr(self._main, "_log", None)
        try:
            if sip.isdeleted(self):
                return
        except Exception:
            return

        norm = []
        try:
            arr = np.asarray(pts, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2:
                norm = [(float(x), float(y)) for x, y in arr]
            elif arr.ndim == 1 and arr.size >= 2 and arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
                norm = [(float(x), float(y)) for x, y in arr]
        except Exception as e:
            if callable(log):
                log(f"[ABE] seed_manual_points: could not parse points: {e!r}")
            norm = []

        if callable(log):
            log(f"[ABE] seed_manual_points: seeding {len(norm)} point(s)")

        if not norm:
            return

        self._manual_points = [QPointF(x, y) for x, y in norm]

        # Flip to Manual WITHOUT emitting currentTextChanged. Otherwise the
        # connected _sample_mode_changed slot runs against the freshly-seeded
        # list: its Auto branch clears _manual_points outright, and its Manual
        # branch clears polygons — either way it stomps the seed we just set.
        # blockSignals lets us set the combo silently; we then set sp_samples'
        # enabled state by hand (the one side effect we still want).
        if self.cmb_sample_mode.findText("Manual") >= 0:
            _blocked = self.cmb_sample_mode.blockSignals(True)
            try:
                self.cmb_sample_mode.setCurrentText("Manual")
            finally:
                self.cmb_sample_mode.blockSignals(_blocked)
            self.sp_samples.setEnabled(False)   # manual mode → # samples unused

        self.btn_clear_samples.setEnabled(len(self._manual_points) > 0)
        self._redraw_overlay()

    def _find_nearest_manual_point_index(self, img_pt: QPointF, max_dist_px: float = 20.0) -> int:
        if not self._manual_points:
            return -1

        best_idx = -1
        best_d2 = float("inf")
        x0, y0 = float(img_pt.x()), float(img_pt.y())

        for i, p in enumerate(self._manual_points):
            dx = float(p.x()) - x0
            dy = float(p.y()) - y0
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        if best_idx < 0:
            return -1

        return best_idx if best_d2 <= (max_dist_px * max_dist_px) else -1

    def _place_grid_points(self):
        """Generate a regular grid of sample points and switch to manual mode."""
        src = self._get_source_float()
        if src is None:
            return

        H, W = src.shape[:2]
        n = int(self.sp_grid_size.value())
        border = int(self.sp_patch.value()) // 2 + 2

        excl = self._build_exclusion_mask()

        xs = np.linspace(border, W - border - 1, n, dtype=int)
        ys = np.linspace(border, H - border - 1, n, dtype=int)

        new_pts = []
        for y in ys:
            for x in xs:
                # Skip points inside exclusion polygons
                if excl is not None and not excl[int(y), int(x)]:
                    continue
                new_pts.append(QPointF(float(x), float(y)))

        self._manual_points.clear()
        self._manual_points.extend(new_pts)

        # Switch to manual so user can edit
        self.cmb_sample_mode.setCurrentText("Manual")
        self.btn_clear_samples.setEnabled(len(self._manual_points) > 0)
        self._redraw_overlay()
        self._set_status(self.tr("Placed {0} grid points ({1}×{2}). Edit as needed.").format(len(new_pts), n, n))


    def _place_auto_points(self):
        """Run auto gradient-descent sampling and push results into manual points."""
        src = self._get_source_float()
        if src is None:
            return

        self._set_status(self.tr("Running auto sampling with gradient descent…"))
        QApplication.processEvents()

        dwn   = int(self.sp_down.value())
        npts  = int(self.sp_samples.value())
        patch = int(self.sp_patch.value())

        excl = self._build_exclusion_mask()

        # Downsample for speed (same as abe_run does)
        small = _downsample_area(src, dwn)
        mask_small = None
        if excl is not None:
            mask_small = _downsample_area(excl.astype(np.float32), dwn) >= 0.5

        # Run the same auto sampler used internally
        seed = int(self.sp_seed.value())
        rng = np.random.default_rng(seed if seed >= 0 else None)

        pts_small = _generate_sample_points(
            small,
            num_points=npts,
            exclusion_mask=mask_small,
            patch_size=patch,
            rng=rng,
        )

        if pts_small is None or len(pts_small) == 0:
            self._set_status(self.tr("Auto sampling found no valid points."))
            return

        # Scale back up to full image coords
        H, W = src.shape[:2]
        sh, sw = small.shape[:2]
        sx = W / max(1, sw)
        sy = H / max(1, sh)

        self._manual_points.clear()
        for x, y in pts_small:
            fx = float(np.clip(x * sx, 0, W - 1))
            fy = float(np.clip(y * sy, 0, H - 1))
            self._manual_points.append(QPointF(fx, fy))

        # Switch to manual so user can edit
        self.cmb_sample_mode.setCurrentText("Manual")
        self.btn_clear_samples.setEnabled(len(self._manual_points) > 0)
        self._redraw_overlay()
        self._set_status(
            self.tr("Placed {0} auto-sampled points. Right-click to remove, left-click to add.").format(len(self._manual_points))
        )

    def _post_init_fit_and_stretch(self) -> None:
        # No longer used — sequence moved entirely into showEvent
        pass

    def _dbg(self, msg: str) -> None:
        """
        Loud debug trace for the exclusion-zone pipeline. Writes to BOTH:
          - stdout (print, flushed) so it shows in the terminal/console the
            app was launched from, and
          - the main window's log panel (if present),
        so the message is visible no matter where the user is looking.

        Prefixed [ABE-EXCL] so the whole exclusion path can be grepped out of
        a busy log. Kept deliberately chatty while we chase the intermittent
        "exclusion ignored" bug; gate on _EXCL_DEBUG so it can be silenced
        later with one flag instead of deleting call sites.
        """
        if not getattr(self, "_EXCL_DEBUG", True):
            return
        line = f"[ABE-EXCL] {msg}"
        try:
            print(line, flush=True)
        except Exception:
            pass
        try:
            _log = getattr(self._main, "_log", None)
            if callable(_log):
                _log(line)
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        try:
            lbl = getattr(self, "status_label", None)
            if lbl is None:
                return
            # Guard against wrapped C++ object deleted (dialog partially destroyed)
            from PyQt6 import sip
            if sip.isdeleted(lbl):
                return
            lbl.setText(text)
            # Paint just the label. The old processEvents() here re-entered
            # the event loop from inside a running slot; the heavy work is on
            # worker threads now, so there is nothing left to pump for.
            lbl.repaint()
        except RuntimeError:
            pass

    def _build_toolbar(self):
        """
        Toolbar row: Zoom In, Zoom Out, Fit, Autostretch.
        """
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(6)

        self.btn_zoom_in  = themed_toolbtn("zoom-in", self.tr("Zoom In"))
        self.btn_zoom_out = themed_toolbtn("zoom-out", self.tr("Zoom Out"))
        self.btn_fit      = themed_toolbtn("zoom-fit-best", self.tr("Fit to Preview"))

        # Use a plain QToolButton so it is always visible even if the theme icon is missing.
        self.btn_autostr = QToolButton(self)
        self.btn_autostr.setText(self.tr("AutoStretch"))
        self.btn_autostr.setCheckable(True)
        self.btn_autostr.setChecked(bool(getattr(self, "_autostretch_on", False)))
        self.btn_autostr.setToolTip(self.tr("Autostretch preview"))
        self.btn_autostr.setMinimumHeight(24)
        self.btn_autostr.setMinimumWidth(44)

        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_fit.clicked.connect(self.fit_to_preview)
        self.btn_autostr.clicked.connect(self.autostretch_preview)

        # Keep all toolbar buttons together on the left
        bar.addWidget(self.btn_zoom_in)
        bar.addWidget(self.btn_zoom_out)
        bar.addWidget(self.btn_fit)
        bar.addWidget(self.btn_autostr)
        bar.addStretch(1)

        return bar

    # ----- active document change -----
    def _on_active_doc_changed(self, doc):
        if doc is None or getattr(doc, "image", None) is None:
            return

        # CRITICAL: only reset drawn state when the active document actually
        # changes to a DIFFERENT one. The main window re-emits
        # currentDocumentChanged on window activation / focus churn — notably
        # when a modal child dialog opens and then closes (the protection
        # preview and its "computing…" splash) — carrying the SAME document.
        # Without this guard that spurious re-fire wiped the user's exclusion
        # polygons between drawing them and pressing Apply: the exact
        # "draw zones → open protection preview → close it → Apply → zones
        # ignored" bug. Re-activating the current document must be a no-op so
        # drawn exclusions and manual sample points survive.
        if doc is self.doc:
            self._dbg("_on_active_doc_changed: SAME document re-activated "
                      "(likely modal preview closing) — keeping "
                      f"{len(self._polygons)} exclusion polygon(s) and "
                      f"{len(self._manual_points)} sample point(s).")
            return

        _log = getattr(self._main, "_log", None)
        if callable(_log):
            _log(f"[ABE] _on_active_doc_changed: NEW doc — clearing "
                 f"{len(self._polygons)} polygon(s), {len(self._manual_points)} pt(s)")
        self._dbg(f"_on_active_doc_changed: NEW document — clearing "
                  f"{len(self._polygons)} exclusion polygon(s) and "
                  f"{len(self._manual_points)} sample point(s).")
        self.doc = doc
        self._polygons.clear()
        self._drawing_poly = None
        self._manual_points.clear()
        self._preview_source_f01 = None
        # A result for the previous document is of no use any more.
        self._abe_cache = None
        self._pending_preview = False
        if self._job_kind == "preview" and self._job_thr is not None:
            self._job_thr.cancel()
        self._populate_initial_preview()
        self._request_display()

    # ----- data helpers -----
    def _get_source_float(self) -> np.ndarray | None:
        src = np.asarray(self.doc.image)
        if src is None or src.size == 0:
            return None

        if np.issubdtype(src.dtype, np.integer):
            scale = float(np.iinfo(src.dtype).max)
            return src.astype(np.float32) / scale

        # float path: preserve full range; abe_run() will normalize internally if needed
        return src.astype(np.float32, copy=False)

    # ----- preview/applier -----
    def _run_abe(self, excl_mask: np.ndarray | None, progress=None):
        """
        Synchronous run on the current settings (scripts / tests). The dialog
        itself runs this work on a worker thread: see _snapshot_job and
        _abe_dialog_compute, which hold the former body of this method.
        """
        # Cleared each run; set only when multiscale + structure-doc are on.
        self._last_structure_map = None
        self._last_mad_mask = None
        job = self._snapshot_job("preview", excl=excl_mask)
        imgf = _source_float(job["src"]) if job is not None else None
        if imgf is None:
            return None, None
        corrected, bg, self._last_structure_map, self._last_mad_mask = _abe_dialog_compute(
            imgf, job, progress, progress, self._on_job_log)
        return corrected, bg

    # ----- background jobs: snapshot -> worker thread -> result slots -----
    def _snapshot_job(self, kind: str, excl="build") -> dict | None:
        """Everything a preview/apply run needs, read from the widgets now."""
        # A preset's manual points normally land with the first render; if the
        # user runs before that render arrives, seed them now so the run uses them.
        if getattr(self, "_pending_manual_points", None) is not None:
            pts = self._pending_manual_points
            self._pending_manual_points = None
            self.seed_manual_points(pts)
        if isinstance(excl, str):
            self._set_status(self.tr("Building exclusion mask…"))
            excl = self._build_exclusion_mask()
        img = getattr(self.doc, "image", None) if self.doc is not None else None
        if img is None or np.asarray(img).size == 0:
            return None
        abe = {
            "degree": int(self.sp_degree.value()),
            "num_samples": int(self.sp_samples.value()),
            "downsample": int(self.sp_down.value()),
            "patch_size": int(self.sp_patch.value()),
            "use_rbf": bool(self.chk_use_rbf.isChecked()),
            "rbf_smooth": float(self.sp_rbf.value()) * 0.01,
            "seed": int(self.sp_seed.value()),
            "manual_points": self._manual_points_array() if self._manual_mode() else None,
            "correction_mode": "divide" if self.radio_divide.isChecked() else "subtract",
        }
        ms = None
        if getattr(self, "chk_ms_enable", None) is not None and self.chk_ms_enable.isChecked():
            ms = {
                "band_lo": int(self.sp_ms_band_lo.value()),
                "band_hi": int(self.sp_ms_band_hi.value()),
                "strength_pct": float(self.sp_ms_strength.value()),
                "disable_protect": bool(self.chk_ms_disable_protect.isChecked()),
                "protect_k": float(self.sp_ms_protect.value()),
                "protect_grow": int(self.sp_ms_grow.value()),
                "sigblur": float(self.sp_ms_sigblur.value()),
                "smooth": float(self.sp_ms_smooth.value()),
                "feather_pct": float(self.sp_ms_feather.value()),
                "residual": bool(self.chk_ms_residual.isChecked()),
                "darkstar": bool(self.chk_ms_darkstar.isChecked()),
                "channels": "luma" if self.radio_ms_luma.isChecked() else "rgb",
            }
        mask, mid, mname = self._active_mask_layer()
        job = {
            "kind": kind,
            "doc": self.doc,
            "src": img,
            "excl": excl,
            "abe": abe,
            "ms": ms,
            "want_structure": bool(getattr(self, "chk_make_structure_doc", None)
                                   and self.chk_make_structure_doc.isChecked()),
            "dbg": bool(getattr(self, "_EXCL_DEBUG", False)),
            "mask": mask,
            "mask_meta": (mid, mname),
            "preview_bg": bool(self.chk_preview_bg.isChecked()),
            "stretch": bool(getattr(self, "_autostretch_on", False)),
            "view_scale": self._expected_view_scale(np.asarray(img).shape) if kind == "preview" else None,
        }
        if kind == "apply":
            job["make_bg_doc"] = bool(self.chk_make_bg_doc.isChecked())
            job["replay_params"] = self._apply_replay_params()
        job["cached"] = self._cached_result_for(job)
        return job

    def _apply_replay_params(self) -> dict:
        """The replay/metadata params dict of an Apply, read at click time."""
        deg        = int(self.sp_degree.value())
        npts       = int(self.sp_samples.value())
        dwn        = int(self.sp_down.value())
        patch      = int(self.sp_patch.value())
        use_rbf    = bool(self.chk_use_rbf.isChecked())
        rbf_smooth = float(self.sp_rbf.value()) * 0.01
        make_bg_doc = bool(self.chk_make_bg_doc.isChecked())

        params = {
            "degree": deg, "samples": npts, "downsample": dwn,
            "patch": patch, "rbf": use_rbf, "rbf_smooth": rbf_smooth,
            "make_background_doc": make_bg_doc,
            "sample_mode": self.cmb_sample_mode.currentText().lower(),
            "manual_sample_count": len(self._manual_points),
        }

        # Multiscale refinement params (only when enabled, so existing
        # presets stay byte-identical when the feature is off).
        if getattr(self, "chk_ms_enable", None) is not None and self.chk_ms_enable.isChecked():
            ms_params = {
                "enabled": True,
                "darkstar": bool(self.chk_ms_darkstar.isChecked()),
                "band_lo": int(self.sp_ms_band_lo.value()),
                "band_hi": int(self.sp_ms_band_hi.value()),
                "include_residual": bool(self.chk_ms_residual.isChecked()),
                "strength": float(self.sp_ms_strength.value()) / 100.0,
                "protect_k": float(self.sp_ms_protect.value()),
                "protect_grow": int(self.sp_ms_grow.value()),
                "smooth_px": float(self.sp_ms_smooth.value()),
                "feather_pct": float(self.sp_ms_feather.value()),
                "sigblur_px": float(self.sp_ms_sigblur.value()),
            }
            if bool(self.chk_ms_disable_protect.isChecked()):
                ms_params["disable_protect"] = True
            if self.radio_ms_luma.isChecked():
                ms_params["channels"] = "luma"
            params["multiscale"] = ms_params
        return params

    def _cached_result_for(self, job: dict):
        """The last preview's (corrected, bg, structure, mad) if `job` would
        recompute exactly that: same document image object, same settings,
        same exclusion mask. Otherwise None."""
        c = getattr(self, "_abe_cache", None)
        if c is None:
            return None
        try:
            if c["doc"] is not job["doc"] or c["src"] is not job["src"]:
                return None
            if c["ms"] != job["ms"] or c["want_structure"] != job["want_structure"]:
                return None
            a, b = c["abe"], job["abe"]
            for k in a:
                if k == "manual_points":
                    pa, pb = a[k], b[k]
                    if (pa is None) != (pb is None):
                        return None
                    if pa is not None and not np.array_equal(pa, pb):
                        return None
                elif a[k] != b[k]:
                    return None
            ea, eb = c["excl"], job["excl"]
            if (ea is None) != (eb is None):
                return None
            if ea is not None and (ea.shape != eb.shape or not np.array_equal(ea, eb)):
                return None
        except Exception:
            return None
        return (c["corrected"], c["bg"], c["structure"], c["mad"])

    def _start_job_thread(self, job: dict, on_result, on_ended, on_failed) -> "_ABEJobThread":
        self._thr_seq += 1
        thr = _ABEJobThread(job, self._thr_seq)
        thr.progress.connect(self._on_job_progress)
        thr.log.connect(self._on_job_log)
        thr.failed.connect(on_failed)
        thr.result.connect(on_result)
        thr.ended.connect(on_ended)
        start_owned_thread(thr)
        return thr

    def _start_compute(self, kind: str, job: dict | None = None) -> None:
        if job is None:
            job = self._snapshot_job(kind)
        if job is None:
            QMessageBox.information(self, self.tr("No image"), self.tr("No image is loaded in the active document."))
            self._set_status(self.tr("Ready"))
            return
        self._dbg(f"_do_{kind}: excl mask is "
                  f"{'None (NO exclusions)' if job['excl'] is None else 'present'}")
        self._job_kind = kind
        self._set_status(self.tr("Running ABE preview…") if kind == "preview" else self.tr("Running ABE (apply)…"))
        self._job_thr = self._start_job_thread(
            job, self._on_compute_result, self._on_compute_ended,
            self._on_compute_failed)
        self._job_seq = self._job_thr.seq
        self._update_busy_ui()

    def _update_busy_ui(self) -> None:
        try:
            applying = (self._job_kind == "apply") or (self._pending_apply is not None)
            self.btn_apply.setEnabled(not applying)
            self.btn_preview.setEnabled(not applying)
            self.btn_reset.setEnabled(not applying)
        except RuntimeError:
            pass

    def _on_job_progress(self, msg: str) -> None:
        if getattr(self, "_closing", False):
            return
        self._set_status(msg)

    def _on_job_log(self, kind: str, msg: str) -> None:
        if kind == "dbg":
            self._dbg(msg)
            return
        mw = self.parent()
        if mw is not None and hasattr(mw, "_log"):
            try:
                mw._log(msg)
            except Exception:
                pass

    def _on_compute_failed(self, seq: int, msg: str, tb: str) -> None:
        kind = self._job_kind
        self._on_job_log("log", f"[ABE] {kind or 'run'} failed: {msg}\n{tb}")
        if getattr(self, "_closing", False) or seq != self._job_seq:
            return
        self._pending_apply = None
        try:
            self._set_status(self.tr("Error"))
            if kind == "apply":
                QMessageBox.critical(self, self.tr("Apply failed"), msg)
            else:
                QMessageBox.warning(self, self.tr("Preview failed"), msg)
        except RuntimeError:
            pass

    def _on_compute_result(self, res: dict) -> None:
        if getattr(self, "_closing", False) or res.get("seq") != self._job_seq:
            return
        job = res["job"]
        if res.get("corrected") is None:
            self._pending_apply = None
            QMessageBox.information(self, self.tr("No image"), self.tr("No image is loaded in the active document."))
            self._set_status(self.tr("Ready"))
            return
        if job["kind"] == "apply":
            self._finish_apply(res)
            return
        if job["doc"] is not self.doc:
            return          # the user switched documents during the run
        self._last_structure_map = res["structure"]
        self._last_mad_mask = res["mad"]
        self._abe_cache = {
            "doc": job["doc"], "src": job["src"], "abe": job["abe"], "ms": job["ms"],
            "want_structure": job["want_structure"], "excl": job["excl"],
            "corrected": res["corrected"], "bg": res["bg"],
            "structure": res["structure"], "mad": res["mad"],
        }
        self._preview_source_f01 = res["show"]
        self._src_gen += 1
        self._initial_src = None
        if job["stretch"] == bool(getattr(self, "_autostretch_on", False)):
            self._install_display(res["buf"], res["qimg"], res.get("pre"))
        else:
            self._request_display()     # stretch was toggled during the run
        if not self._pending_preview and self._pending_apply is None:
            self._set_status(self.tr("Ready"))

    def _on_compute_ended(self, seq: int) -> None:
        if seq != self._job_seq:
            return              # a job this dialog already replaced
        self._job_thr = None
        self._job_kind = None
        if getattr(self, "_closing", False):
            return
        if self._pending_apply is not None:
            job = self._pending_apply
            self._pending_apply = None
            # The preview that just landed may be exactly this Apply.
            job["cached"] = self._cached_result_for(job)
            self._start_compute("apply", job)
        elif self._pending_preview:
            self._pending_preview = False
            self._start_compute("preview")
        self._update_busy_ui()

    # ----- preview display (stretch + 8-bit) on a worker -----
    def _request_display(self) -> None:
        """Re-render the preview with the current autostretch state, off the
        GUI thread. While one render runs, the newest request waits for it."""
        if getattr(self, "_closing", False):
            return
        if self._disp_thr is not None:
            self._disp_pending = True
            return
        self._disp_pending = False
        if self._preview_source_f01 is not None:
            job = {"kind": "display", "src": self._preview_source_f01, "clip_source": False}
        elif self._initial_src is not None:
            job = {"kind": "display", "src": self._initial_src, "clip_source": True}
        else:
            return
        job["stretch"] = bool(getattr(self, "_autostretch_on", False))
        job["gen"] = self._src_gen
        job["view_scale"] = self._expected_view_scale(np.asarray(job["src"]).shape)
        self._disp_thr = self._start_job_thread(
            job, self._on_display_result, self._on_display_ended,
            self._on_display_failed)
        self._disp_seq = self._disp_thr.seq

    def _on_display_result(self, res: dict) -> None:
        if getattr(self, "_closing", False) or res.get("seq") != self._disp_seq:
            return
        job = res["job"]
        if job["gen"] != self._src_gen:
            return                      # a newer preview replaced the source
        if job.get("clip_source"):
            self._preview_source_f01 = res["src"]
            self._initial_src = None
        if job["stretch"] != bool(getattr(self, "_autostretch_on", False)):
            self._disp_pending = True   # toggled meanwhile: render again
            return
        self._install_display(res["buf"], res["qimg"], res.get("pre"))

    def _on_display_failed(self, seq: int, msg: str, tb: str) -> None:
        self._on_job_log("log", f"[ABE] preview render failed: {msg}\n{tb}")
        if not getattr(self, "_closing", False) and seq == self._disp_seq:
            self._set_status(self.tr("Error"))

    def _on_display_ended(self, seq: int) -> None:
        if seq != self._disp_seq:
            return
        self._disp_thr = None
        if self._disp_pending and not getattr(self, "_closing", False):
            self._request_display()

    def _expected_view_scale(self, shape) -> float | None:
        """The scale an image of `shape` will be shown at when it lands:
        the fit scale for the first image, else the current zoom."""
        try:
            if self._fitted_once:
                return float(self._preview_scale)
            ih, iw = int(shape[0]), int(shape[1])
            vp = self.preview_scroll.viewport()
            vw, vh = max(1, vp.width()), max(1, vp.height())
            if iw == 0 or ih == 0:
                return None
            return max(0.05, min(min(vw / iw, vh / ih), 8.0))
        except Exception:
            return None

    def _install_display(self, buf: np.ndarray, qimg: QImage, pre=None) -> None:
        """Show a rendered preview (GUI thread: only the QPixmap is made here).
        `pre` is the worker's pre-scaled copy, used if the scale still matches."""
        try:
            if sip.isdeleted(self) or sip.isdeleted(self.preview_label):
                return
        except Exception:
            return
        self._last_preview = buf        # backing ndarray for QImage lifetime
        self._preview_qimg = qimg
        if not self._fitted_once:
            # First image: fit to the viewport (this builds the pixmap), then
            # seed preset manual points onto the real preview.
            self._fitted_once = True
            self.fit_to_preview(_prescaled=pre)
            if self._pending_manual_points is not None:
                pts = self._pending_manual_points
                self._pending_manual_points = None
                self.seed_manual_points(pts)
            return
        self._update_preview_scaled(_prescaled=pre)
        self._redraw_overlay()

    def _abe_params(self) -> dict:
        """
        Serialize the live UI into the exact schema apply_abe_via_preset() reads.
        Single source of truth = the headless consumer in abe_preset.py.

        Traps mirrored here:
          • rbf_smooth — sp_rbf is an integer spinbox shown as "×0.01"; the
            consumer wants the already-scaled float (100 -> 1.0), exactly as
            _run_abe()/_do_apply() compute it. Raw widget value inflates 100x.
          • correction_mode — read the radio pair; abe_run() branches
            subtract vs divide on this string.
          • manual_points — mode-dependent key. Only Manual mode with points
            contributes, captured in FULL-IMAGE pixel coords (Nx2), matching
            what _run_abe() hands abe_run(). Absent/empty => headless
            auto-samples, exactly like the dialog. JSON round-trips as
            [[x,y],...]; consumer coerces.
        """
        params = {
            "degree":              int(self.sp_degree.value()),
            "samples":             int(self.sp_samples.value()),
            "downsample":          int(self.sp_down.value()),
            "patch":               int(self.sp_patch.value()),
            "rbf":                 bool(self.chk_use_rbf.isChecked()),
            "rbf_smooth":          float(self.sp_rbf.value()) * 0.01,   # ×0.01 trap
            "seed":                int(self.sp_seed.value()),
            "make_background_doc": bool(self.chk_make_bg_doc.isChecked()),
            "correction_mode":     "divide" if self.radio_divide.isChecked() else "subtract",
        }

        # Mode-dependent: mirror _run_abe()'s gate exactly.
        mp = self._manual_points_array() if self._manual_mode() else None
        if mp is not None and len(mp) > 0:
            params["manual_points"] = mp.tolist()   # [[x,y],...] JSON-safe ints

        # Multiscale refinement — only emitted when enabled, so a preset made
        # with the feature off stays byte-identical to the legacy schema.
        if getattr(self, "chk_ms_enable", None) is not None and self.chk_ms_enable.isChecked():
            ms = {
                "enabled": True,
                "darkstar": bool(self.chk_ms_darkstar.isChecked()),
                "band_lo": int(self.sp_ms_band_lo.value()),
                "band_hi": int(self.sp_ms_band_hi.value()),
                "include_residual": bool(self.chk_ms_residual.isChecked()),
                "strength": float(self.sp_ms_strength.value()) / 100.0,
                "protect_k": float(self.sp_ms_protect.value()),
                "protect_grow": int(self.sp_ms_grow.value()),
                "smooth_px": float(self.sp_ms_smooth.value()),
                "feather_pct": float(self.sp_ms_feather.value()),
                "sigblur_px": float(self.sp_ms_sigblur.value()),
            }
            # Only emit the disable-protection flag when it's actually set,
            # so a preset made the normal way stays schema-compatible with
            # the pre-upgrade consumer. Headless side should treat a missing
            # key as False.
            if bool(self.chk_ms_disable_protect.isChecked()):
                ms["disable_protect"] = True
            # Likewise for the Luma opt-out: emit "channels" only when it
            # differs from the default so a preset built with the normal
            # (per-channel RGB) refinement stays byte-identical to one made
            # on the previous version. Headless side should treat a missing
            # key as "rgb".
            if self.radio_ms_luma.isChecked():
                ms["channels"] = "luma"
            params["multiscale"] = ms

        return params

    def _open_protect_preview(self):
        """
        Open the live median+k*MAD protection-mask preview.

        Mirrors the real pipeline: the mask runs on the ADBE-flattened image,
        so we run the normal background removal FIRST on a downsampled copy and
        build the preview on that result. Previewing on the raw image would show
        the mask protecting the bright side of the still-present large gradient
        instead of the actual stars and signal.

        On OK the chosen k is written back into the Protect-k control.
        """
        # Defence in depth: the button is already greyed when protection is
        # disabled, but a keyboard shortcut or programmatic click could still
        # land here. Previewing a mask that will never be applied would just
        # confuse the user.
        if getattr(self, "chk_ms_disable_protect", None) is not None \
                and self.chk_ms_disable_protect.isChecked():
            QMessageBox.information(
                self, self.tr("Preview protection"),
                self.tr("Signal protection is disabled — there is no mask to preview.\n"
                "Uncheck 'Disable signal protection' to use Protect k.")
            )
            return

        src = self._get_source_float()
        if src is None:
            QMessageBox.information(self, self.tr("Preview protection"), self.tr("No active image."))
            return

        # Downsample for responsiveness (cap the long side ~900 px).
        try:
            h, w = src.shape[:2]
            scale = max(1, int(np.ceil(max(h, w) / 900.0)))
            small = _downsample_area(src, scale) if scale > 1 else src
        except Exception:
            small = src
            scale = 1

        # Hand-drawn exclusion polygons are ALSO protected at apply time (they
        # composite back from the original), so overlay them in the preview too
        # — shown in green to distinguish from the red MAD mask. Downsample the
        # excluded region (~excl mask) to match the preview image.
        excl_overlay = None
        try:
            excl = self._build_exclusion_mask()   # True=sample / False=excluded
            if excl is not None:
                drawn = (~excl).astype(np.float32)  # 1 inside the drawn polygons
                if scale > 1:
                    drawn = _downsample_area(drawn, scale)
                drawn = (drawn >= 0.5).astype(np.float32)
                if drawn.any():
                    excl_overlay = drawn
        except Exception:
            excl_overlay = None

        # Run the SAME ADBE the user has configured, on the small copy, so the
        # preview mask sees the flattened background the real mask will see.
        self._set_status(self.tr("Preview: running ADBE for protection preview…"))

        # Small modal splash — the ADBE pass runs synchronously on the GUI
        # thread and can take a few seconds, so show a "computing" dialog first
        # (and force it to paint) so the UI doesn't look frozen.
        splash = QDialog(self)
        splash.setWindowTitle(self.tr("Protection preview"))
        splash.setModal(True)
        try:
            splash.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        except Exception:
            pass
        _sv = QVBoxLayout(splash)
        _sv.setContentsMargins(20, 18, 20, 18)
        _lbl = QLabel(self.tr("Computing protection preview…\nRunning background extraction."))
        _lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _sv.addWidget(_lbl)
        _pb = QProgressBar()
        _pb.setRange(0, 0)   # indeterminate "busy" bar
        _pb.setTextVisible(False)
        _sv.addWidget(_pb)
        splash.setFixedWidth(300)
        splash.show()
        splash.raise_()
        # Force the splash to actually paint before the blocking compute.
        QApplication.processEvents()
        QApplication.processEvents()

        flattened = small
        try:
            deg        = int(self.sp_degree.value())
            npts       = int(self.sp_samples.value())
            dwn        = int(self.sp_down.value())
            patch      = int(self.sp_patch.value())
            use_rbf    = bool(self.chk_use_rbf.isChecked())
            rbf_smooth = float(self.sp_rbf.value()) * 0.01
            seed       = int(self.sp_seed.value())
            correction_mode = "divide" if self.radio_divide.isChecked() else "subtract"
            rng = np.random.default_rng(seed if seed >= 0 else None)
            # The small image is already downsampled; use a light internal
            # downsample so the fit stays quick but valid.
            flattened = abe_run(
                small,
                degree=deg, num_samples=npts, downsample=max(2, min(dwn, 4)),
                patch_size=patch, use_rbf=use_rbf, rbf_smooth=rbf_smooth,
                exclusion_mask=None, return_background=False,
                correction_mode=correction_mode, rng=rng,
            )
        except Exception as e:
            if hasattr(self._main, "_log"):
                self._main._log(f"[Protect preview] ADBE pass failed ({e}); "
                                f"previewing on raw image.")
            flattened = small
        finally:
            try:
                splash.close()
                splash.deleteLater()
            except Exception:
                pass
            self._set_status(self.tr("Ready"))
            QApplication.processEvents()

        # Pass through the UI's current grow (previously hard-coded to 6).
        # The preview itself still displays the raw threshold with a small
        # fudge for dilation (see _ProtectPreviewDialog.PREVIEW_K_FUDGE), but
        # the dialog records the value for completeness.
        dlg = _ProtectPreviewDialog(self, flattened, k_init=float(self.sp_ms_protect.value()),
                                    grow=int(self.sp_ms_grow.value()),
                                    blur_px=float(self.sp_ms_sigblur.value()),
                                    exclusion_overlay=excl_overlay)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            k = dlg.selected_k()
            # Clamp to the Protect-k spin's current range (1.0..20.0 after the
            # halo-frame bump). The old 5.0 ceiling here silently discarded any
            # higher k the user had chosen in the preview slider.
            k = max(1.0, min(20.0, float(k)))
            self.sp_ms_protect.setValue(k)

    def _degree_changed(self, v: int):
        # Make it clear what 0 means, and default RBF on (can still be unchecked)
        if v == 0:
            self.chk_use_rbf.setChecked(True)
            if hasattr(self, "_set_status"):
                self._set_status(self.tr("Polynomial disabled (degree 0) → RBF-only."))
        else:
            if hasattr(self, "_set_status"):
                self._set_status(self.tr("Ready"))

    def _populate_initial_preview(self):
        """Don't render anything yet — just remember the source. showEvent
        (or a document switch) renders it on a worker via _request_display."""
        img = getattr(self.doc, "image", None) if self.doc is not None else None
        if img is None or np.asarray(img).size == 0:
            return
        self._initial_src = img
        self._src_gen += 1
        # Leave _preview_qimg as None — placeholder shown in showEvent


    def _show_placeholder(self):
        if getattr(self, "_closing", False):
            return
        """Show a 'Computing preview…' message in the preview area while we work."""
        vp = self.preview_scroll.viewport()
        w, h = max(480, vp.width()), max(360, vp.height())

        # Dark background with centered text — matches the typical dark theme
        pm = QPixmap(w, h)
        pm.fill(QColor(30, 30, 30))

        painter = QPainter(pm)
        painter.setPen(QColor(180, 180, 180))
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(
            pm.rect(),
            Qt.AlignmentFlag.AlignCenter,
            self.tr("Computing preview…")
        )
        painter.end()

        self.preview_label.setPixmap(pm)
        self.preview_label.resize(pm.size())
        QApplication.processEvents()

    def _render_preview_from_source(self, stretch: bool = True):
        """
        Render self._preview_source_f01 into QImage/pixmap.
        stretch=True applies autostretch. Called after geometry is known.
        """
        src = getattr(self, "_preview_source_f01", None)
        if src is None:
            return

        if stretch and getattr(self, "_autostretch_on", False):
            disp = hard_autostretch(src, target_median=0.5, sigma=2,
                                    linked=False, use_24bit=True)
            disp = np.asarray(disp, dtype=np.float32)
        else:
            disp = np.clip(src, 0.0, 1.0).astype(np.float32)

        if disp.ndim == 2 or (disp.ndim == 3 and disp.shape[2] == 1):
            mono = disp if disp.ndim == 2 else disp[..., 0]
            buf8 = np.ascontiguousarray((mono * 255.0).astype(np.uint8))
            self._last_preview = np.ascontiguousarray(np.stack([buf8] * 3, axis=-1))
            h, w = buf8.shape
            self._preview_qimg = QImage(buf8.data, w, h, w,
                                        QImage.Format.Format_Grayscale8)
        else:
            buf8 = np.ascontiguousarray((disp * 255.0).astype(np.uint8))
            self._last_preview = buf8
            h, w, _ = buf8.shape
            self._preview_qimg = QImage(buf8.data, w, h, buf8.strides[0],
                                        QImage.Format.Format_RGB888)

        self._preview_qimg = tag_qimage_with_working_color_space(self._preview_qimg)

        self._update_preview_scaled()
        self._redraw_overlay()

    def _do_preview(self):
        try:
            from PyQt6 import sip
            if sip.isdeleted(self) or getattr(self, "_closing", False):
                return
            # Commit any polygon the user just finished drawing but whose
            # mouse-release landed off the label (see
            # _commit_pending_polygon). Done before mask-build so the overlay
            # repaints it as a committed zone and the mask includes it.
            self._dbg(f"_do_preview: ENTER — _polygons={len(self._polygons)}, "
                      f"sample_mode={self.cmb_sample_mode.currentText()}")
            self._commit_pending_polygon()
            if self._job_kind == "apply" or self._pending_apply is not None:
                return          # Apply is on its way; the dialog closes after it
            if self._job_thr is not None:
                # Settings changed mid-run: stop the running preview at its next
                # checkpoint and run again on the settings as they are now.
                self._pending_preview = True
                if self._job_kind == "preview":
                    self._job_thr.cancel()
                self._set_status(self.tr("Preview will restart with the current settings…"))
                return
            self._start_compute("preview")
        except Exception as e:
            self._set_status(self.tr("Error"))
            QMessageBox.warning(self, self.tr("Preview failed"), str(e))

    def _do_apply(self):
        try:
            if getattr(self, "_closing", False):
                return
            # Commit a finished-but-uncommitted polygon first (mouse-release
            # off the label), so Apply never silently runs on the whole frame
            # when the user clearly drew an exclusion. See
            # _commit_pending_polygon.
            self._dbg(f"_do_apply: ENTER — _polygons={len(self._polygons)}, "
                      f"sample_mode={self.cmb_sample_mode.currentText()}")
            self._commit_pending_polygon()
            if self._job_kind == "apply" or self._pending_apply is not None:
                return
            job = self._snapshot_job("apply")
            if job is None:
                QMessageBox.information(self, self.tr("No image"), self.tr("No image is loaded in the active document."))
                self._set_status(self.tr("Ready"))
                return
            if self._job_thr is not None:
                # A preview is running: Apply waits for it (and reuses it when
                # the settings match), instead of being dropped.
                self._pending_apply = job
                self._pending_preview = False
                self._set_status(self.tr("Apply queued — waiting for the running preview…"))
                self._update_busy_ui()
                return
            self._start_compute("apply", job)
        except Exception as e:
            self._set_status(self.tr("Error"))
            QMessageBox.critical(self, self.tr("Apply failed"), str(e))

    def _finish_apply(self, res: dict):
        """GUI-thread tail of Apply: commit the worker's result to the document."""
        try:
            job = res["job"]
            doc = job["doc"]
            bg = res["bg"]
            out_masked = res["out_masked"]
            params = job["replay_params"]
            make_bg_doc = bool(job["make_bg_doc"])

            step_name = (
                f"ABE (deg={params['degree']}, samples={params['samples']}, "
                f"ds={params['downsample']}, patch={params['patch']}, "
                f"rbf={'on' if params['rbf'] else 'off'}, s={params['rbf_smooth']:.3f})"
            )

            # Remember for replay — do this before any close
            mw = self.parent()
            try:
                remember = getattr(mw, "remember_last_headless_command", None)
                if remember is None:
                    remember = getattr(mw, "_remember_last_headless_command", None)
                if callable(remember):
                    remember("abe", params, description="Automatic Background Extraction")
                    if hasattr(mw, "_log"):
                        mw._log(f"[Replay] ABE UI apply stored: command_id='abe', preset_keys={list(params.keys())}")
            except Exception:
                pass

            mid, mname = job["mask_meta"]
            meta = {
                "step_name": "ABE",
                "abe": dict(params),
                "masked": bool(mid),
                "mask_id": mid,
                "mask_name": mname,
                "mask_blend": "m*out + (1-m)*src",
            }

            self._set_status(self.tr("Committing edit…"))
            doc.apply_edit(
                out_masked.astype(np.float32, copy=False),
                step_name=step_name,
                metadata=meta,
            )

            # Background doc — do before close
            if make_bg_doc and bg is not None:
                self._set_status(self.tr("Creating background document…"))
                dm = getattr(mw, "docman", None)
                if dm is not None:
                    base = os.path.splitext(doc.display_name())[0]
                    bg_meta = {
                        "bit_depth": "32-bit floating point",
                        "is_mono": (bg.ndim == 2),
                        "source": "ABE background",
                        "original_header": doc.metadata.get("original_header"),
                    }
                    doc_bg = dm.open_array(bg.astype(np.float32, copy=False), metadata=bg_meta, title=f"{base}_ABE_BG")
                    if hasattr(mw, "_spawn_subwindow_for"):
                        mw._spawn_subwindow_for(doc_bg)

            # Multiscale STRUCTURE document — the multiplicative gradient map the
            # multiscale stage divided out (mote/reflection/filter-edge only).
            # Separate from the full poly/RBF background above; for troubleshooting.
            struct = res["structure"]
            if job["want_structure"] and struct is not None:
                self._set_status(self.tr("Creating multiscale structure document…"))
                dm = getattr(mw, "docman", None)
                if dm is not None:
                    base = os.path.splitext(doc.display_name())[0]
                    # The structure map is a MULTIPLICATIVE gradient with median
                    # ~1.0, so ~half its values exceed 1.0 (and a strong defect
                    # spikes well past it). Dumped as-is, everything above 1.0
                    # clips to white and the bright-side structure is lost. Scale
                    # by its own max into [0,1] purely for viewing — this is a
                    # troubleshooting view, not photometric data, so a display
                    # normalization is exactly right.
                    s_view = np.asarray(struct, dtype=np.float32)
                    s_view = np.nan_to_num(s_view, nan=0.0, posinf=0.0, neginf=0.0)
                    s_max = float(s_view.max()) if s_view.size else 0.0
                    if s_max > 0.0:
                        s_view = s_view / s_max
                    s_view = np.clip(s_view, 0.0, 1.0).astype(np.float32, copy=False)
                    # The normalized map sits jammed against 1.0 (a gradient is
                    # ~flat, so after /max nearly everything is a hair below 1).
                    # Drop it halfway down the histogram so it lands mid-range
                    # and the viewer's autostretch has room on both sides — this
                    # only repositions (keeps the spread), it does not stretch.
                    s_min = float(s_view.min()) if s_view.size else 0.0
                    s_view = (s_view - 0.5 * s_min).astype(np.float32, copy=False)
                    s_meta = {
                        "bit_depth": "32-bit floating point",
                        "is_mono": (s_view.ndim == 2),
                        "source": "ABE multiscale structure (gradient map, "
                                  "normalized to max and offset for display)",
                        "original_header": doc.metadata.get("original_header"),
                    }
                    doc_s = dm.open_array(s_view,
                                          metadata=s_meta, title=f"{base}_ABE_Structure")
                    if hasattr(mw, "_spawn_subwindow_for"):
                        mw._spawn_subwindow_for(doc_s)

            # DEBUG: push the raw median+k*MAD signal-protection mask so you can
            # eyeball what the threshold caught. Controlled by the module-level
            # DEBUG flag; only present when the multiscale stage ran.
            mad = res["mad"]
            if DEBUG and mad is not None:
                self._set_status(self.tr("Creating DEBUG MAD mask document…"))
                dm = getattr(mw, "docman", None)
                if dm is not None:
                    base = os.path.splitext(doc.display_name())[0]
                    mad_view = np.clip(np.nan_to_num(np.asarray(mad, dtype=np.float32)),
                                       0.0, 1.0).astype(np.float32, copy=False)
                    mad_meta = {
                        "bit_depth": "32-bit floating point",
                        "is_mono": (mad_view.ndim == 2),
                        "source": "ABE DEBUG median+k*MAD signal-protection mask",
                        "original_header": doc.metadata.get("original_header"),
                    }
                    doc_mad = dm.open_array(mad_view, metadata=mad_meta,
                                            title=f"{base}_ABE_DEBUG_MADmask")
                    if hasattr(mw, "_spawn_subwindow_for"):
                        mw._spawn_subwindow_for(doc_mad)

            # Restore autostretch on the active view
            prev_autostretch = False
            view = None
            try:
                if hasattr(mw, "mdi") and mw.mdi.activeSubWindow():
                    view = mw.mdi.activeSubWindow().widget()
                    prev_autostretch = bool(getattr(view, "autostretch_enabled", False))
            except Exception:
                pass

            if hasattr(mw, "_log"):
                mw._log(step_name)

            try:
                if view is not None and hasattr(view, "set_autostretch") and callable(view.set_autostretch):
                    view.set_autostretch(prev_autostretch)
            except Exception:
                pass

            # All work is done — set flag so closeEvent knows it's a clean close
            # then defer the actual close so this stack frame fully unwinds first
            self._closing = True
            QTimer.singleShot(0, self.close)

        except Exception as e:
            # Guard: dialog may be partially torn down on some platforms
            try:
                from PyQt6 import sip
                if not sip.isdeleted(self):
                    self._set_status(self.tr("Error"))
                    QMessageBox.critical(self, self.tr("Apply failed"), str(e))
            except Exception:
                pass

    def closeEvent(self, ev):
        self._close_teardown()
        super().closeEvent(ev)

    def done(self, r):
        # Esc ends a QDialog through done(), which never delivers closeEvent:
        # without this it skipped the job cancel, the active-view unhook and
        # the release, and the dialog kept its image until collected.
        self._close_teardown()
        super().done(r)

    def _close_teardown(self):
        if getattr(self, "_close_torn_down", False):
            return
        self._close_torn_down = True
        self._closing = True
        # Stop background jobs. thread_keeper owns the threads, so this window
        # may go away while one is still unwinding; its result is dropped.
        self._pending_apply = None
        self._pending_preview = False
        for _thr in (getattr(self, "_job_thr", None), getattr(self, "_disp_thr", None)):
            if _thr is not None:
                try:
                    _thr.cancel()
                except RuntimeError:
                    pass
        # Release any mouse grab still held from an in-progress polygon drag
        # so it can't outlive the dialog (Qt usually auto-releases on hide,
        # but be explicit).
        try:
            self.preview_label.releaseMouse()
        except Exception:
            pass
        try:
            if self._connected_current_doc_changed and hasattr(self._main, "currentDocumentChanged"):
                self._main.currentDocumentChanged.disconnect(self._on_active_doc_changed)
        except Exception:
            pass
        self._connected_current_doc_changed = False
        try:
            self._save_window_geometry()
        except Exception:
            pass
        try:
            if getattr(self, "_worker", None) is not None:
                try:
                    self._worker.requestInterruption()
                except Exception:
                    pass
            if getattr(self, "_thread", None) is not None:
                self._thread.quit()
                self._thread.wait(500)
        except Exception:
            pass
        try:
            self._release_memory()
        except Exception:
            pass

    def _release_memory(self):
        """Drop preview source / QImage backing buffers on close."""
        if getattr(self, "_memory_released", False):
            return
        self._memory_released = True

        self._preview_source_f01 = None
        self._last_preview = None
        self._preview_qimg = None
        self._overlay = None
        self._base_pixmap = None
        self._abe_cache = None
        self._initial_src = None
        self.doc = None
        self._main = None
        try:
            self.preview_label.clear()
        except Exception:
            pass

        try:
            from setiastro.saspro.free_torch_memory import trim_process_memory
            trim_process_memory()
        except Exception:
            pass

    def _refresh_document_from_active(self):
        """
        Refresh the dialog's document reference to the currently active document.
        This allows reusing the same dialog on different images.
        """
        try:
            main = self.parent()
            if main and hasattr(main, "_active_doc"):
                new_doc = main._active_doc()
                if new_doc is not None and new_doc is not self.doc:
                    self.doc = new_doc
                    # Reset preview state for new document
                    self._preview_source_f01 = None
                    self._last_preview = None
                    self._preview_qimg = None
                    # Clear polygons since they were for old image
                    self._clear_polys()
                    self._manual_points.clear()
        except Exception:
            pass


    # ----- exclusion polygons & mask -----
    def _clear_polys(self):
        self._polygons.clear()
        self._drawing_poly = None
        # ✅ redraw from the clean base
        self._redraw_overlay()

    def _commit_pending_polygon(self) -> bool:
        """
        Fold any in-progress polygon (``_drawing_poly``) into the committed
        list if it has enough points to be a real region (>= 3 vertices).

        This closes the intermittent "my exclusion zone was ignored" bug.
        A polygon only ever reached ``_polygons`` from inside the
        ``obj is self.preview_label`` branch of the event filter (on
        right-click or left-button *release*). If the cursor drifted off the
        label before the button came up — a fast drag that ended past the
        image edge, a release over a scrollbar, a flick that left the widget
        — that release was delivered to a different widget and the commit
        never ran. The polygon then sat in ``_drawing_poly`` forever: still
        PAINTED (as a dashed outline, so the user saw it) but invisible to
        ``_build_exclusion_mask``, which only read ``_polygons``. Result:
        ~1 run in 4, the mask came back empty and refinement ran on the
        whole frame.

        Calling this at mask-build time (and at the top of Preview/Apply)
        guarantees a finished-but-uncommitted polygon is honoured. A
        sub-3-point scrap is discarded, matching the release handler.

        Returns True if a polygon was committed.
        """
        dp = self._drawing_poly
        if dp is not None and len(dp) >= 3:
            # Hand the live list off to _polygons, then null the pending slot
            # so a second call is a no-op and the list isn't aliased/mutated.
            self._polygons.append(dp)
            self._drawing_poly = None
            self._dbg(f"_commit_pending_polygon: committed a pending {len(dp)}-pt "
                      f"polygon; _polygons now has {len(self._polygons)}.")
            try:
                self._redraw_overlay()   # repaint so it shows as committed
            except Exception:
                pass
            return True
        if dp is not None:
            self._dbg(f"_commit_pending_polygon: pending polygon had only "
                      f"{len(dp)} pt(s) (<3) — discarding, NOT committing.")
        return False

    def _image_shape(self) -> tuple[int, int]:
        src = np.asarray(self.doc.image)
        if src.ndim == 2:
            return src.shape[0], src.shape[1]
        return src.shape[0], src.shape[1]

    def _build_exclusion_mask(self) -> np.ndarray | None:
        # Safety net: fold a finished-but-uncommitted polygon in before we
        # read _polygons, so a release that landed off the label can never
        # silently drop the user's last-drawn zone (see
        # _commit_pending_polygon for the full rationale).
        self._commit_pending_polygon()

        self._dbg(f"_build_exclusion_mask: ENTER — _polygons={len(self._polygons)}, "
                  f"_drawing_poly={'None' if self._drawing_poly is None else len(self._drawing_poly)}")

        if not self._polygons:
            self._dbg("_build_exclusion_mask: no committed polygons → "
                      "returning None (NO exclusions).")
            return None

        H, W = self._image_shape()
        self._dbg(f"_build_exclusion_mask: image shape H={H} W={W} "
                  f"(from self.doc.image); cv2={'present' if cv2 is not None else 'MISSING'}")

        # Normalise every stored polygon into an int32 (N,2) array, dropping
        # any that are degenerate (fewer than 3 distinct vertices). QPointF
        # coords are in ORIGINAL IMAGE space; cv2.fillPoly clips to the mask
        # bounds on its own, but we validate here so one bad polygon can't
        # wipe out the whole mask or raise mid-fill.
        valid_polys = []
        for i, poly in enumerate(self._polygons):
            n = 0 if poly is None else len(poly)
            if poly is None or n < 3:
                self._dbg(f"  poly[{i}]: {n} pt(s) (<3) → SKIP")
                continue
            try:
                pts = np.array([[int(round(p.x())), int(round(p.y()))]
                                for p in poly], dtype=np.int32)
            except Exception as e:
                self._dbg(f"  poly[{i}]: coord conversion FAILED ({e}) → SKIP")
                continue
            uniq = np.unique(pts, axis=0).shape[0]
            xmin, ymin = int(pts[:, 0].min()), int(pts[:, 1].min())
            xmax, ymax = int(pts[:, 0].max()), int(pts[:, 1].max())
            in_bounds = (xmax >= 0 and ymax >= 0 and xmin < W and ymin < H)
            self._dbg(f"  poly[{i}]: {n} pts, {uniq} unique, "
                      f"bbox=({xmin},{ymin})-({xmax},{ymax}), "
                      f"in_bounds={in_bounds}")
            # Need at least 3 UNIQUE points or it has zero area and fillPoly
            # paints nothing — treat that as not-a-region rather than a
            # silent no-op.
            if uniq < 3:
                self._dbg(f"  poly[{i}]: <3 unique vertices → SKIP (degenerate)")
                continue
            valid_polys.append(pts)

        if not valid_polys:
            # Polygons existed but none were real regions. Return None (the
            # documented "no exclusions" sentinel) and log it, so a future
            # recurrence is visible instead of silently processing the whole
            # frame as if nothing was drawn.
            self._dbg(f"_build_exclusion_mask: {len(self._polygons)} polygon(s) "
                      f"stored but NONE had >=3 unique vertices → returning None "
                      f"(NO exclusions). This is the bug symptom if you drew real zones.")
            return None

        mask = np.ones((H, W), dtype=np.uint8)
        if cv2 is None:
            # Pure-numpy fallback (expect OpenCV in SASpro). Coarse: fills the
            # polygon's bounding box rather than the exact winding region.
            self._dbg("_build_exclusion_mask: cv2 MISSING — using coarse "
                      "bounding-box fallback fill.")
            for pts in valid_polys:
                minx, maxx = np.clip([pts[:, 0].min(), pts[:, 0].max()], 0, W - 1)
                miny, maxy = np.clip([pts[:, 1].min(), pts[:, 1].max()], 0, H - 1)
                mask[miny:maxy + 1, minx:maxx + 1] = 0
        else:
            # Fill each polygon in its OWN fillPoly call — NOT one call with the
            # whole list. cv2.fillPoly applies an even-odd rule across all
            # polygons passed in a single call, so where two zones OVERLAP the
            # region is covered twice and toggles back to UN-filled — a hole.
            # That silently un-protected exactly the overlap the user drew to
            # be extra-safe (e.g. a second zone over a galaxy core already
            # inside a bigger zone). Filling one at a time makes the result a
            # true UNION: overlaps are set to 0 twice (idempotent), never
            # cancelled. See the overlap test in commit notes.
            for i, pts in enumerate(valid_polys):
                cv2.fillPoly(mask, [pts], 0)  # 0 = excluded; one poly per call

        excluded_px = int((mask == 0).sum())
        pct = 100.0 * excluded_px / float(max(1, H * W))
        self._dbg(f"_build_exclusion_mask: filled {len(valid_polys)} zone(s) → "
                  f"{excluded_px} px excluded ({pct:.2f}%). "
                  f"Convention: True=sample/keep, False=excluded-inside-polygon.")

        # Final guard: if, despite valid polygons, nothing actually got
        # excluded (e.g. every polygon fell entirely outside the image),
        # report None so downstream treats it as "no mask" consistently.
        if excluded_px == 0:
            self._dbg("_build_exclusion_mask: 0 px excluded despite valid "
                      "polygons → returning None. (Polygons likely fell outside "
                      "the image, or coords are in the wrong space.)")
            return None
        self._dbg("_build_exclusion_mask: RETURNING bool mask with "
                  f"{excluded_px} excluded px.")
        return mask.astype(bool)

    # ----- preview rendering helpers -----

    def _set_preview_pixmap(self, arr: np.ndarray):
        if arr is None or arr.size == 0:
            self.preview_label.clear(); self._overlay = None; self._preview_source_f01 = None
            return

        # keep the float source for autostretch toggling (no re-normalization).
        # Synchronous; the dialog's own Preview renders on a worker instead.
        a = _asfloat32(arr)
        self._preview_source_f01 = a  # ← no np.clip here
        self._src_gen += 1
        self._initial_src = None
        buf, qimg = _render_display_u8(a, bool(getattr(self, "_autostretch_on", False)))
        self._install_display(buf, qimg)



    def _update_preview_scaled(self, *, _prescaled=None):
        try:
            if sip.isdeleted(self) or sip.isdeleted(self.preview_label):
                return
        except Exception:
            return

        if self._preview_qimg is None:
            self.preview_label.clear()
            return

        sw = max(1, int(self._preview_qimg.width()  * self._preview_scale))
        sh = max(1, int(self._preview_qimg.height() * self._preview_scale))

        if _prescaled is not None and _prescaled[0] == (sw, sh):
            scaled = _prescaled[1]      # same scaled() call, done on the worker
        else:
            scaled = self._preview_qimg.scaled(
                sw, sh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        if scaled.isNull():
            return

        self._base_pixmap = QPixmap.fromImage(scaled)
        if self._base_pixmap.isNull():
            return

        try:
            if not sip.isdeleted(self.preview_label):
                self.preview_label.setPixmap(self._base_pixmap)
                self.preview_label.resize(self._base_pixmap.size())
        except Exception:
            pass

    def _redraw_overlay(self):
        # Guard against being called during/after close
        try:
            if sip.isdeleted(self) or sip.isdeleted(self.preview_label):
                return
        except Exception:
            return

        pm_base = self._base_pixmap or self.preview_label.pixmap()
        if pm_base is None or pm_base.isNull():
            return

        # start from a fresh copy of the clean base
        composed = QPixmap(pm_base)
        if composed.isNull():
            return

        overlay = QPixmap(pm_base.size())
        if overlay.isNull():
            return
        overlay.fill(Qt.GlobalColor.transparent)

        painter = QPainter(overlay)
        if not painter.isActive():
            return  # paint device invalid — widget being destroyed

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # map image-space polys to label-space
        img_w = self._preview_qimg.width() if self._preview_qimg else 1
        img_h = self._preview_qimg.height() if self._preview_qimg else 1
        # Map image-space -> overlay-canvas space. The overlay is pm_base-sized
        # (the SCALED pixmap), so use the preview scale, NOT the label size:
        # when the scaled image is smaller than the label's 480x360 minimum it
        # gets centered, the label is wider than the pixmap, and lab_w/img_w
        # overshoots — which is exactly why drawn polygons drift from the cursor
        # once the image leaves the viewport edge. _label_to_image_coords divides
        # by this same scale (and subtracts the centering offset), so this makes
        # the draw and the hit-test exact inverses at every zoom level.
        sx = float(self._preview_scale)
        sy = float(self._preview_scale)

        # finalized polygons (green, semi-transparent)
        pen = QPen(QColor(0, 255, 0), 2)
        brush = QColor(0, 255, 0, 60)
        painter.setPen(pen)
        painter.setBrush(brush)
        for poly in self._polygons:
            if len(poly) >= 3:
                mapped = [QPointF(p.x() * sx, p.y() * sy) for p in poly]
                painter.drawPolygon(*mapped)

        # in-progress poly (red dashed)
        if self._drawing_poly and len(self._drawing_poly) >= 2:
            pen2 = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mapped = [QPointF(p.x() * sx, p.y() * sy) for p in self._drawing_poly]
            painter.drawPolyline(*mapped)

        # manual sample squares (yellow outlines)
        if self._manual_points:
            patch = int(max(1, self.sp_patch.value()))
            # On-screen half-size from the patch, but never below a visible
            # floor. At fit-scale on a large image (this file: ~4500px in a
            # ~480px viewport, scale ≈ 0.1) patch*scale collapses to ~0.8px and
            # the squares disappear — which is exactly why preset-seeded points
            # rendered but were invisible while zoomed-in manual placement showed
            # fine. Clamp so points are always visible regardless of zoom.
            MIN_HALF_PX = 4.0
            hx = max(0.5 * patch * sx, MIN_HALF_PX)
            hy = max(0.5 * patch * sy, MIN_HALF_PX)

            pen3 = QPen(QColor(255, 220, 0), 2)
            painter.setPen(pen3)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for p in self._manual_points:
                cx = p.x() * sx
                cy = p.y() * sy
                painter.drawRect(
                    int(round(cx - hx)),
                    int(round(cy - hy)),
                    int(round(2 * hx)),
                    int(round(2 * hy)),
                )


        painter.end()

        p = QPainter(composed)
        if not p.isActive():
            return
        p.drawPixmap(0, 0, overlay)
        p.end()

        try:
            if not sip.isdeleted(self.preview_label):
                self.preview_label.setPixmap(composed)
        except Exception:
            pass

    # ----- zoom/pan + polygon drawing -----
    def eventFilter(self, obj, ev):
        # ---- Mouse wheel zoom handling (Qt6-friendly) ----
        if ev.type() == QEvent.Type.Wheel and (
            obj is self.preview_label
            or obj is self.preview_scroll
            or obj is self.preview_scroll.viewport()
            or obj is self.preview_scroll.horizontalScrollBar()
            or obj is self.preview_scroll.verticalScrollBar()
        ):
            # Always stop the wheel from scrolling the scrollbars/scroll area.
            ev.accept()

            # Anchor at the mouse position in the viewport
            # (even if the event came from a scrollbar)
            vp = self.preview_scroll.viewport()
            anchor_vp = vp.mapFromGlobal(ev.globalPosition().toPoint())

            # Clamp to viewport rect (robust if the event originated on scrollbars)
            r = vp.rect()
            if not r.contains(anchor_vp):
                anchor_vp.setX(max(r.left(),  min(r.right(),  anchor_vp.x())))
                anchor_vp.setY(max(r.top(),   min(r.bottom(), anchor_vp.y())))

            # Smooth trackpad support first, then mouse wheel fallback
            dy = ev.pixelDelta().y()
            if dy != 0:
                abs_dy = abs(dy)
                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)

                if abs_dy <= 3:
                    base_factor = 1.012 if ctrl_down else 1.010
                elif abs_dy <= 10:
                    base_factor = 1.025 if ctrl_down else 1.020
                else:
                    base_factor = 1.040 if ctrl_down else 1.030

                factor = base_factor if dy > 0 else 1.0 / base_factor
            else:
                dy = ev.angleDelta().y()
                if dy == 0:
                    return True

                ctrl_down = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                step = 1.25 if ctrl_down else 1.15
                factor = step if dy > 0 else 1.0 / step

            self._zoom_at(factor, anchor_vp)
            return True

        # ---- Existing polygon drawing on the label ----
        if obj is self.preview_label:
            if ev.type() == QEvent.Type.MouseButtonPress:
                # panning still works the same in either mode
                if ev.buttons() & Qt.MouseButton.MiddleButton or (
                    ev.buttons() & Qt.MouseButton.LeftButton and (ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                ):
                    self._panning = True
                    self._pan_last = ev.position().toPoint()
                    self.preview_label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

                # manual sample-point mode
                if self._manual_mode():
                    if ev.button() == Qt.MouseButton.LeftButton:
                        img_pt = self._label_to_image_coords(ev.position())
                        if img_pt is not None:
                            self._manual_points.append(img_pt)
                            self.btn_clear_samples.setEnabled(True)
                            self._redraw_overlay()
                            return True

                    if ev.button() == Qt.MouseButton.RightButton:
                        img_pt = self._label_to_image_coords(ev.position())
                        if img_pt is not None:
                            idx = self._find_nearest_manual_point_index(img_pt)
                            if idx >= 0:
                                self._manual_points.pop(idx)
                                self.btn_clear_samples.setEnabled(
                                    self._manual_mode() and len(self._manual_points) > 0
                                )
                                self._redraw_overlay()
                            return True

                # existing polygon mode
                if ev.buttons() & Qt.MouseButton.RightButton:
                    if self._drawing_poly and len(self._drawing_poly) >= 3:
                        self._polygons.append(self._drawing_poly)
                    self._drawing_poly = None
                    self._redraw_overlay()
                    return True

                if ev.buttons() & Qt.MouseButton.LeftButton:
                    # clamp=True: polygon drawing clamps out-of-bounds mouse
                    # positions to the image edge instead of ignoring them.
                    # Fixes the "my corner polygon missed the corner" bug —
                    # see _label_to_image_coords docstring for the full story.
                    img_pt = self._label_to_image_coords(ev.position(), clamp=True)
                    if img_pt is not None:
                        if self._drawing_poly is None:
                            self._drawing_poly = [img_pt]
                            # Grab the mouse for the duration of the freehand
                            # drag so the eventual MouseButtonRelease is
                            # ALWAYS delivered back to preview_label — even if
                            # the cursor leaves the widget (off the image
                            # edge, over a scrollbar, off the window). Without
                            # this, a release outside the label never ran the
                            # commit branch and the finished polygon was
                            # stranded in _drawing_poly (the intermittent
                            # "exclusion ignored" bug). _commit_pending_polygon
                            # is the safety net; this is the root-cause fix.
                            try:
                                self.preview_label.grabMouse()
                            except Exception:
                                pass
                        else:
                            self._drawing_poly.append(img_pt)
                        self._redraw_overlay()
                        return True

            elif ev.type() == QEvent.Type.MouseMove:
                if getattr(self, "_panning", False):
                    pos = ev.position().toPoint()
                    delta = pos - (self._pan_last or pos)
                    self._pan_last = pos
                    hsb = self.preview_scroll.horizontalScrollBar()
                    vsb = self.preview_scroll.verticalScrollBar()
                    hsb.setValue(hsb.value() - delta.x())
                    vsb.setValue(vsb.value() - delta.y())
                    return True
                if (not self._manual_mode()) and self._drawing_poly is not None and (ev.buttons() & Qt.MouseButton.LeftButton):
                    # clamp=True: the drag of a freehand polygon follows
                    # the image edge when the cursor wanders outside, so
                    # dragging around a corner encloses the corner instead
                    # of cutting it off at the re-entry point.
                    img_pt = self._label_to_image_coords(ev.position(), clamp=True)
                    if img_pt is not None:
                        self._drawing_poly.append(img_pt)
                        self._redraw_overlay()
                        return True

            elif ev.type() == QEvent.Type.MouseButtonRelease:
                # Always release any mouse grab we took when drawing started,
                # before anything else — grabbing is a global state and must
                # not outlive the drag no matter which branch handles release.
                try:
                    self.preview_label.releaseMouse()
                except Exception:
                    pass

                # finish panning
                if getattr(self, "_panning", False):
                    self._panning = False
                    self._pan_last = None
                    self.preview_label.unsetCursor()
                    return True

                # Close polygon on LEFT mouse release
                if (not self._manual_mode()) and ev.button() == Qt.MouseButton.LeftButton and self._drawing_poly is not None:
                    if len(self._drawing_poly) >= 3:
                        self._polygons.append(self._drawing_poly)
                    self._drawing_poly = None
                    self._redraw_overlay()
                    return True

        return super().eventFilter(obj, ev)

    def _ensure_scale_state(self):
        # internal guard so _zoom_at can be called even if _scale hasn't been set
        if not hasattr(self, "_scale"):
            self._scale = float(self.view.transform().m11()) if not self.view.transform().isIdentity() else 1.0

    def _zoom_at(self, factor: float, anchor_vp) -> None:
        """
        Zoom the preview by 'factor', keeping the content point under 'anchor_vp'
        (a QPoint in viewport coordinates) stationary.
        """
        old_scale = float(self._preview_scale)
        new_scale = max(0.05, min(old_scale * factor, 8.0))
        if abs(new_scale - old_scale) < 1e-6:
            return
        factor = new_scale / old_scale

        # content coordinates (relative to the QLabel) under the cursor BEFORE scaling
        hsb = self.preview_scroll.horizontalScrollBar()
        vsb = self.preview_scroll.verticalScrollBar()
        old_x = hsb.value() + anchor_vp.x()
        old_y = vsb.value() + anchor_vp.y()

        # apply scale
        self._preview_scale = new_scale
        self._update_preview_scaled()
        self._redraw_overlay()

        # desired scroll so the same content point stays under the cursor
        new_x = int(old_x * factor - anchor_vp.x())
        new_y = int(old_y * factor - anchor_vp.y())

        # clamp to valid range
        hsb.setValue(max(hsb.minimum(), min(new_x, hsb.maximum())))
        vsb.setValue(max(vsb.minimum(), min(new_y, vsb.maximum())))


    def zoom_in(self) -> None:
        vp = self.preview_scroll.viewport()
        self._zoom_at(1.25, vp.rect().center())

    def zoom_out(self) -> None:
        vp = self.preview_scroll.viewport()
        self._zoom_at(0.8, vp.rect().center())

    def fit_to_preview(self, *, _prescaled=None) -> None:
        """Set scale so the image fits inside the viewport (keeps aspect)."""
        if self._preview_qimg is None:
            return
        vp = self.preview_scroll.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())
        iw, ih = self._preview_qimg.width(), self._preview_qimg.height()
        if iw == 0 or ih == 0:
            return
        scale = min(vw / iw, vh / ih)
        self._preview_scale = max(0.05, min(scale, 8.0))
        self._update_preview_scaled(_prescaled=_prescaled)
        self._redraw_overlay()

        # center after fit
        hsb = self.preview_scroll.horizontalScrollBar()
        vsb = self.preview_scroll.verticalScrollBar()
        hsb.setValue((hsb.maximum() - hsb.minimum()) // 2)
        vsb.setValue((vsb.maximum() - vsb.minimum()) // 2)

    def _label_to_image_coords(self, posf, clamp: bool = False) -> QPointF | None:
        """
        Convert a mouse position in the preview label into image-space
        coordinates.

        By default, returns None when the position is outside the drawn
        pixmap area — the correct behaviour for click events (a user
        clicking in the empty area beside the image should not be treated
        as clicking on the image edge).

        When `clamp=True`, out-of-bounds positions are snapped to the
        nearest pixmap edge instead of dropped. This is what the
        exclusion-polygon drawing paths want: when the user drags the
        mouse past a corner of the image, we want the polygon to follow
        the IMAGE corner (so a dragged "arc around the corner" actually
        encloses the corner), not to be ignored until the cursor
        re-enters the image. The old drop-on-out-of-bounds behaviour was
        the cause of the "my corner polygon missed the corner" bug: Qt
        kept delivering MouseMove events as the cursor left the label,
        each one was dropped, and the first in-bounds re-entry event got
        appended as a straight-line jump from the last recorded point —
        cutting the corner off entirely.
        """
        if self._preview_qimg is None:
            return None

        img_w = self._preview_qimg.width()
        img_h = self._preview_qimg.height()

        # The scaled pixmap may be smaller than the label if zoomed out —
        # Qt centers it, so we must account for that margin before converting.
        pm = self.preview_label.pixmap()
        if pm is None or pm.isNull():
            return None

        pm_w = pm.width()
        pm_h = pm.height()
        lab_w = self.preview_label.width()
        lab_h = self.preview_label.height()

        off_x = max(0, (lab_w - pm_w) // 2)
        off_y = max(0, (lab_h - pm_h) // 2)

        # Position relative to the pixmap origin (not the label origin)
        px = float(posf.x()) - off_x
        py = float(posf.y()) - off_y

        if clamp:
            # Snap out-of-bounds positions to the nearest pixmap edge so
            # the caller still gets a usable image-space point. The final
            # clamp to [0, img_w-1] below belt-and-suspenders this.
            px = max(0.0, min(px, pm_w - 1.0))
            py = max(0.0, min(py, pm_h - 1.0))
        else:
            # Outside the drawn pixmap area — ignore. Click-style callers
            # (manual sample placement) rely on this to reject stray
            # clicks beside the image.
            if px < 0 or py < 0 or px >= pm_w or py >= pm_h:
                return None

        # Pixmap pixels → image pixels via the preview scale
        x_img = px / max(1e-12, self._preview_scale)
        y_img = py / max(1e-12, self._preview_scale)

        # Clamp to image bounds (also catches float rounding on the pm edge)
        x_img = max(0.0, min(x_img, img_w - 1.0))
        y_img = max(0.0, min(y_img, img_h - 1.0))

        return QPointF(x_img, y_img)

    def _install_zoom_filters(self):
        """Install event filters so Ctrl+Wheel works even when the cursor is over scrollbars."""
        self.preview_scroll.installEventFilter(self)
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_scroll.horizontalScrollBar().installEventFilter(self)
        self.preview_scroll.verticalScrollBar().installEventFilter(self)
        self.preview_label.installEventFilter(self)
        
    def _set_preview_from_float(self, arr: np.ndarray):
        if arr is None or arr.size == 0:
            return
        a = _asfloat32(arr)
        self._preview_source_f01 = a  # ← no np.clip

        src_to_show = (hard_autostretch(self._preview_source_f01, target_median=0.5, sigma=2,
                                        linked=False, use_24bit=True)
                    if getattr(self, "_autostretch_on", False) else self._preview_source_f01)

        if src_to_show.ndim == 2 or (src_to_show.ndim == 3 and src_to_show.shape[2] == 1):
            mono = src_to_show if src_to_show.ndim == 2 else src_to_show[..., 0]
            buf8_mono = (mono * 255.0).astype(np.uint8)               # ← no np.clip
            buf8_mono = np.ascontiguousarray(buf8_mono)
            self._last_preview = np.ascontiguousarray(np.stack([buf8_mono]*3, axis=-1))
            h, w = buf8_mono.shape
            qimg = QImage(buf8_mono.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            buf8 = (src_to_show * 255.0).astype(np.uint8)             # ← no np.clip
            buf8 = np.ascontiguousarray(buf8)
            self._last_preview = buf8
            h, w, _ = buf8.shape
            qimg = QImage(buf8.data, w, h, buf8.strides[0], QImage.Format.Format_RGB888)

        qimg = tag_qimage_with_working_color_space(qimg)
        self._preview_qimg = qimg
        self._update_preview_scaled()
        self._redraw_overlay()

    # --- mask helpers ---------------------------------------------------
    def _active_mask_layer(self):
        """Return (mask_float01, mask_id, mask_name) or (None, None, None)."""
        mid = getattr(self.doc, "active_mask_id", None)
        if not mid: return None, None, None
        layer = getattr(self.doc, "masks", {}).get(mid)
        if layer is None: return None, None, None
        m = np.asarray(getattr(layer, "data", None))
        if m is None or m.size == 0: return None, None, None
        m = m.astype(np.float32, copy=False)
        if m.dtype.kind in "ui":
            m /= float(np.iinfo(m.dtype).max)
        else:
            mx = float(m.max()) if m.size else 1.0
            if mx > 1.0: m /= mx
        return np.clip(m, 0.0, 1.0), mid, getattr(layer, "name", "Mask")

    def _resample_mask_if_needed(self, mask: np.ndarray, out_hw: tuple[int,int]) -> np.ndarray:
        """Nearest-neighbor resize via integer indexing."""
        mh, mw = mask.shape[:2]
        th, tw = out_hw
        if (mh, mw) == (th, tw): return mask
        yi = np.linspace(0, mh - 1, th).astype(np.int32)
        xi = np.linspace(0, mw - 1, tw).astype(np.int32)
        return mask[yi][:, xi]

    def _blend_with_mask_float(self, processed: np.ndarray, src: np.ndarray | None = None) -> np.ndarray:
        """
        m*out + (1-m)*src in float [0..1], mono or RGB.
        If src is None, uses the current document image (float [0..1]).
        """
        mask, _mid, _mname = self._active_mask_layer()
        if mask is None:
            return processed
        if src is None:
            src = self._get_source_float()
        return _blend_mask_float(processed, src, mask)

    def autostretch_preview(self, checked: bool | None = None) -> None:
        """
        Toggle preview autostretch on/off and rebuild the viewport from the
        underlying float preview source (rendered on a worker thread).
        """
        if self._preview_source_f01 is None and self._initial_src is None:
            return

        current = bool(getattr(self, "_autostretch_on", False))
        new_state = (not current) if checked is None else bool(checked)
        self._autostretch_on = new_state

        self._request_display()

        if hasattr(self, "btn_autostr"):
            self.btn_autostr.setChecked(new_state)
            self.btn_autostr.setToolTip(
                self.tr("Autostretch preview (On)") if new_state else self.tr("Autostretch preview")
            )

        self._save_settings()

    def _apply_autostretch_inplace(self, sigma: float = 3.0):
        self.autostretch_preview(True)

    def _restore_window_geometry(self):
        try:
            s = QSettings()
            g = s.value("abe_ui/window_geometry", None)   # ✅ unique, NOT stat_stretch
            if g is not None:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_window_geometry(self):
        try:
            s = QSettings()
            s.setValue("abe_ui/window_geometry", self.saveGeometry())
        except Exception:
            pass        

    def showEvent(self, ev):
        super().showEvent(ev)

        if not getattr(self, "_geom_restored", False):
            self._geom_restored = True

            def _after_show():
                # 1) Restore geometry first so viewport size is correct
                self._restore_window_geometry()
                QApplication.processEvents()

                # 2) Show placeholder so user sees something while we compute
                self._show_placeholder()

                # 3) Render (stretch + 8-bit) on a worker thread. When it lands,
                #    _install_display fits it to the now-correct viewport and
                #    seeds any preset manual points LAST, onto the real preview.
                self._request_display()

            QTimer.singleShot(0, _after_show)
            return

        # Already restored — just refit if shown again
        QTimer.singleShot(0, self.fit_to_preview)