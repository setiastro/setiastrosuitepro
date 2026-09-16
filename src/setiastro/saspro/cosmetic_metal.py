# setiastro/saspro/cosmetic_metal.py
"""
Metal-friendly cosmetic correction (hot/cold pixel rejection).

Why this exists
---------------
The previous GPU cosmetic pass built a (1,C,H,W,9) neighbourhood tensor with
`unfold` and then leaned on `torch.sort`, `torch.median` and `torch.quantile`
per frame. Those three are exactly the kernels Metal (MPS) is worst at — sort
and median are slow-or-CPU-fallback, and `torch.quantile` outright refuses
inputs above 2**24 elements, which a modern sensor plane exceeds. That is what
pegged the GPU at ~100% while barely moving frames.

This rewrite computes the identical statistics with only Metal-native ops:

  * REFERENCE (the value a bad pixel is replaced with, and the centre it's
    compared against): the median of the 8-neighbour ring, via a 19-comparator
    sorting network built from elementwise min/max — no sort, no median.
  * SCALE (how far a pixel must deviate to be "bad"): a single robust global
    sigma per plane (1.4826 · MAD), computed once on the host from numpy. A
    per-pixel MAD over only 8 samples is a wildly noisy, downward-biased scale —
    the old code used it and would false-flag ~1.6% of clean background at a
    nominal 5σ. A global sigma drops that to ~1 pixel in 260k at 5σ.
  * SATURATION quantile: also a single scalar per plane on the host — never a
    device-side `quantile`.

So the only reductions are two cheap scalars per plane, done on the CPU while
the frame is still numpy (the calibration pipeline hands us numpy anyway, so
there is no extra device sync). Everything per-pixel stays on the GPU as
min / max / compare / where.

Correctness notes
-----------------
  * CFA / Bayer frames are de-interleaved into their 4 phase sub-planes and each
    is corrected against SAME-colour neighbours, then re-interleaved. Running a
    3×3 median straight over a mosaic blends R/G/B and is wrong. The pattern
    NAME doesn't matter here — all four patterns just relabel which 2×2 phase is
    which colour; the phase split is identical.
  * Structure protection (protect_sigma > 0): a pixel is NOT corrected if its
    second-most-extreme neighbour is itself beyond protect_sigma — i.e. it isn't
    an isolated spike but part of real structure (a star core, a tight double).
    This preserves stars even when a nearby pixel is genuinely bright.
  * The saturation guard exempts the brightest `sat_quantile` fraction from
    correction, matching the previous behaviour (don't rework saturated cores).
    Set sat_quantile >= 1.0 to disable it.

Public entry point: cosmetic_correct(). Returns the same array type/layout as
the input (numpy in → numpy out; torch tensor in → tensor out on same device).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# 19-comparator sorting network for 8 inputs (min -> i, max -> j).
# Verified by the 0/1 principle: it sorts all 2**8 binary inputs, hence all
# inputs. Used to extract the ring median with elementwise min/max only.
_NET8: Tuple[Tuple[int, int], ...] = (
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
    (0, 1), (2, 3), (4, 5), (6, 7),
    (2, 4), (3, 5),
    (1, 4), (3, 6),
    (1, 2), (3, 4), (5, 6),
)

# 9-comparator sorting network for 5 inputs (verified by the 0/1 principle).
# Used for the cross-median (N,S,E,W,center) in the tiled cosmetic kernel.
_NET5: Tuple[Tuple[int, int], ...] = (
    (0, 1), (2, 3), (0, 2), (1, 4), (0, 1), (2, 3), (1, 2), (3, 4), (2, 3),
)

_EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# dim-0 network reductions for a STACKED tensor (used by torch_rejection's
# tiled cosmetic pass). Only elementwise min/max — the whole point is that Metal
# never sees torch.sort / torch.median.
# ─────────────────────────────────────────────────────────────────────────────
def sort8_along_dim0(t8, torch):
    """Sort a stacked (8, ...) tensor ascending along dim 0, in place, using only
    torch.minimum / torch.maximum. Returns the same tensor. Replaces
    `torch.sort(t8, dim=0).values` on MPS."""
    for i, j in _NET8:
        lo = torch.minimum(t8[i], t8[j])
        hi = torch.maximum(t8[i], t8[j])
        t8[i].copy_(lo)
        t8[j].copy_(hi)
    return t8


def median5_along_dim0(t5, torch):
    """Median of a stacked (5, ...) tensor along dim 0 via a 9-comparator network
    (min/max only). Returns a fresh (...) tensor. Replaces
    `t5.median(dim=0).values` on MPS."""
    for i, j in _NET5:
        lo = torch.minimum(t5[i], t5[j])
        hi = torch.maximum(t5[i], t5[j])
        t5[i].copy_(lo)
        t5[j].copy_(hi)
    return t5[2].clone()


# ─────────────────────────────────────────────────────────────────────────────
# host-side scalars (numpy) — one robust sigma + one saturation value per plane
# ─────────────────────────────────────────────────────────────────────────────
def _plane_scalars_np(plane: np.ndarray, sat_quantile: float) -> Tuple[float, float]:
    """Return (global_sigma, sat_val) for a single 2-D plane, computed on CPU."""
    p = np.asarray(plane, dtype=np.float32)
    med = float(np.median(p))
    mad = float(np.median(np.abs(p - med)))
    gsig = 1.4826 * mad + _EPS
    if sat_quantile is None or sat_quantile >= 1.0:
        sat_val = float("inf")           # guard disabled
    else:
        # np.quantile is a single O(n) partition — cheap, and never the device
        # quantile that Metal rejects.
        sat_val = float(np.quantile(p, sat_quantile))
    return gsig, sat_val


# ─────────────────────────────────────────────────────────────────────────────
# NumPy reference implementation (also the CPU fallback + the tested spec)
# ─────────────────────────────────────────────────────────────────────────────
def _sort8_np(v: List[np.ndarray]) -> List[np.ndarray]:
    v = list(v)
    for i, j in _NET8:
        lo = np.minimum(v[i], v[j])
        hi = np.maximum(v[i], v[j])
        v[i], v[j] = lo, hi
    return v


def _ring_np(plane: np.ndarray):
    P = np.pad(plane, 1, mode="edge")
    H, W = plane.shape
    sl = lambda dy, dx: P[dy:dy + H, dx:dx + W]
    neigh = [sl(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2)]
    center = neigh[4]
    ring = [neigh[k] for k in range(9) if k != 4]
    s = _sort8_np(ring)
    rmed = 0.5 * (s[3] + s[4])
    return center, rmed, s


def _cosmetic_plane_np(plane, hot_sigma, cold_sigma, protect_sigma, gsig, sat_val):
    center, rmed, s = _ring_np(plane)
    hot_m = center > rmed + hot_sigma * gsig
    cold_m = center < rmed - cold_sigma * gsig
    if protect_sigma and protect_sigma > 0:
        hot_m = hot_m & ~(s[6] > rmed + protect_sigma * gsig)
        cold_m = cold_m & ~(s[1] < rmed - protect_sigma * gsig)
    if np.isfinite(sat_val):
        sat_m = plane >= sat_val
        hot_m = hot_m & ~sat_m
        cold_m = cold_m & ~sat_m
    mask = hot_m | cold_m
    return np.where(mask, rmed, plane).astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Torch implementation — identical math, Metal-native ops only
# ─────────────────────────────────────────────────────────────────────────────
def _sort8_torch(v, torch):
    v = list(v)
    for i, j in _NET8:
        lo = torch.minimum(v[i], v[j])
        hi = torch.maximum(v[i], v[j])
        v[i], v[j] = lo, hi
    return v


def _ring_torch(plane, torch):
    F = torch.nn.functional
    # replicate pad (edge) — Metal-safe, unlike some reflect sizes; the 1-px
    # border matters nothing for single-pixel cosmetic detection.
    P = F.pad(plane[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    H, W = plane.shape
    def sl(dy, dx):
        return P[dy:dy + H, dx:dx + W]
    neigh = [sl(dy, dx) for dy in (0, 1, 2) for dx in (0, 1, 2)]
    center = neigh[4]
    ring = [neigh[k] for k in range(9) if k != 4]
    s = _sort8_torch(ring, torch)
    rmed = 0.5 * (s[3] + s[4])
    return center, rmed, s


def _cosmetic_plane_torch(plane, hot_sigma, cold_sigma, protect_sigma,
                          gsig, sat_val, torch):
    center, rmed, s = _ring_torch(plane, torch)
    hot_m = center > rmed + hot_sigma * gsig
    cold_m = center < rmed - cold_sigma * gsig
    if protect_sigma and protect_sigma > 0:
        hot_m = hot_m & ~(s[6] > rmed + protect_sigma * gsig)
        cold_m = cold_m & ~(s[1] < rmed - protect_sigma * gsig)
    if np.isfinite(sat_val):
        sat_m = plane >= float(sat_val)
        hot_m = hot_m & ~sat_m
        cold_m = cold_m & ~sat_m
    mask = hot_m | cold_m
    return torch.where(mask, rmed, plane)


# ─────────────────────────────────────────────────────────────────────────────
# layout helpers
# ─────────────────────────────────────────────────────────────────────────────
def _iter_planes(image: np.ndarray):
    """Yield (index_key, 2-D plane view). Supports 2-D, CHW-3, HWC-3."""
    if image.ndim == 2:
        yield ("mono", None), image
    elif image.ndim == 3 and image.shape[0] == 3:
        for c in range(3):
            yield ("chw", c), image[c]
    elif image.ndim == 3 and image.shape[-1] == 3:
        for c in range(3):
            yield ("hwc", c), image[..., c]
    else:
        yield ("mono", None), image  # best-effort: treat as single plane


def _deinterleave_bayer(a: np.ndarray) -> List[np.ndarray]:
    return [a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]]


def _reinterleave_bayer(subs: List[np.ndarray], shape) -> np.ndarray:
    out = np.empty(shape, dtype=np.float32)
    out[0::2, 0::2], out[0::2, 1::2], out[1::2, 0::2], out[1::2, 1::2] = subs
    return out


# ─────────────────────────────────────────────────────────────────────────────
# device / torch resolution
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_torch_device(torch, device):
    if torch is None:
        try:
            from setiastro.saspro.runtime_torch import import_torch
            torch = import_torch(prefer_cuda=True, prefer_xpu=False,
                                 prefer_dml=True, status_cb=lambda s: None)
        except Exception:
            try:
                import torch as _t
                torch = _t
            except Exception:
                return None, None
    if device is None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                device = torch.device("cuda")
            elif (hasattr(torch, "backends")
                  and hasattr(torch.backends, "mps")
                  and torch.backends.mps.is_available()):
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        except Exception:
            device = torch.device("cpu")
    return torch, device


# ─────────────────────────────────────────────────────────────────────────────
# public entry point
# ─────────────────────────────────────────────────────────────────────────────
def cosmetic_correct(
    image,
    *,
    hot_sigma: float = 5.0,
    cold_sigma: float = 5.0,
    protect_sigma: float = 0.0,
    sat_quantile: float = 0.9995,
    bayer_pattern: Optional[str] = None,
    torch=None,
    device=None,
    use_gpu: bool = True,
):
    """
    Hot/cold pixel cosmetic correction.

    image        : np.ndarray (2-D mono, CHW-3, or HWC-3) OR a torch.Tensor of
                   the same layout. numpy in → numpy out; tensor in → tensor out
                   on the same device.
    hot_sigma/cold_sigma : deviation (in global-sigma units) to flag a bright/dark
                   outlier.
    protect_sigma: >0 shields real structure (stars) from correction; 0 disables.
    sat_quantile : brightest fraction exempted from correction; >=1.0 disables.
    bayer_pattern: 'RGGB'/'BGGR'/'GRBG'/'GBRG' for a CFA mosaic (2-D only) — the
                   frame is corrected per colour phase. None for mono/colour.
    use_gpu      : if False (or no GPU present) the validated numpy path is used.

    Values are assumed to be in SASpro's internal [0,1] range.
    """
    is_torch_in = not isinstance(image, np.ndarray)

    # tensor in: correct on-device (compute the two scalars via a one-time CPU
    # reduction of each plane — bounded, once per plane, never per pixel). The
    # numpy-in path below avoids even that, and is the calibration pipeline's
    # normal path.
    if is_torch_in:
        t = image
        tmod = torch
        if tmod is None:
            import torch as tmod  # type: ignore
        return _correct_tensor(t, tmod, hot_sigma, cold_sigma, protect_sigma,
                               sat_quantile, bayer_pattern)

    arr = np.ascontiguousarray(image, dtype=np.float32)

    run_gpu = False
    if use_gpu:
        torch, device = _resolve_torch_device(torch, device)
        run_gpu = (torch is not None and device is not None
                   and str(device.type) in ("cuda", "mps"))

    def _do_plane(plane2d: np.ndarray) -> np.ndarray:
        gsig, sat_val = _plane_scalars_np(plane2d, sat_quantile)
        if run_gpu:
            t = torch.from_numpy(np.ascontiguousarray(plane2d, np.float32)).to(
                device, dtype=torch.float32)
            out = _cosmetic_plane_torch(t, hot_sigma, cold_sigma, protect_sigma,
                                        gsig, sat_val, torch)
            return out.detach().to("cpu").numpy().astype(np.float32, copy=False)
        return _cosmetic_plane_np(plane2d, hot_sigma, cold_sigma, protect_sigma,
                                  gsig, sat_val)

    # CFA mosaic: per-phase (2-D only; a mosaic is single-plane)
    if bayer_pattern and arr.ndim == 2:
        subs = _deinterleave_bayer(arr)
        corr = [_do_plane(np.ascontiguousarray(sp)) for sp in subs]
        return _reinterleave_bayer(corr, arr.shape)

    if arr.ndim == 2:
        return _do_plane(arr)

    # colour: per channel, preserving layout
    out = np.empty_like(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:      # CHW
        for c in range(3):
            out[c] = _do_plane(np.ascontiguousarray(arr[c]))
        return out
    if arr.ndim == 3 and arr.shape[-1] == 3:     # HWC
        for c in range(3):
            out[..., c] = _do_plane(np.ascontiguousarray(arr[..., c]))
        return out

    return _do_plane(arr)  # unexpected shape: best effort


def _correct_tensor(t, torch, hot_sigma, cold_sigma, protect_sigma,
                    sat_quantile, bayer_pattern):
    def _do(plane_t):
        # one-time CPU reduction for the two scalars; per-pixel work stays on device
        arr = plane_t.detach().to("cpu", dtype=torch.float32).numpy()
        gsig, sat_val = _plane_scalars_np(arr, sat_quantile)
        return _cosmetic_plane_torch(plane_t, hot_sigma, cold_sigma,
                                     protect_sigma, gsig, sat_val, torch)

    if bayer_pattern and t.ndim == 2:
        s00 = _do(t[0::2, 0::2]); s01 = _do(t[0::2, 1::2])
        s10 = _do(t[1::2, 0::2]); s11 = _do(t[1::2, 1::2])
        out = torch.empty_like(t)
        out[0::2, 0::2] = s00; out[0::2, 1::2] = s01
        out[1::2, 0::2] = s10; out[1::2, 1::2] = s11
        return out
    if t.ndim == 2:
        return _do(t)
    if t.ndim == 3 and t.shape[0] == 3:
        return torch.stack([_do(t[c]) for c in range(3)], dim=0)
    if t.ndim == 3 and t.shape[-1] == 3:
        return torch.stack([_do(t[..., c]) for c in range(3)], dim=-1)
    return _do(t)