# pro/torch_rejection.py
from __future__ import annotations
import contextlib
import numpy as np

# Always route through our runtime shim so ALL GPU users share the same backend.
# Nothing heavy happens at import; we only resolve Torch when needed.
from .runtime_torch import import_torch, add_runtime_to_sys_path, best_device

# Algorithms supported by the GPU path here (names match your UI/CPU counterparts)
_SUPPORTED = {
    "Comet Median",
    "Simple Median (No Rejection)",
    "Comet High-Clip Percentile",
    "Comet Lower-Trim (30%)",
    "Comet Percentile (40th)",
    "Simple Average (No Rejection)",
    "Weighted Windsorized Sigma Clipping",
    "Windsorized Sigma Clipping",   # <<< NEW (unweighted)
    "Kappa-Sigma Clipping",
    "Trimmed Mean",
    "Extreme Studentized Deviate (ESD)",
    "Biweight Estimator",
    "Modified Z-Score Clipping",
    "Max Value",
}

# ---------------------------------------------------------------------------
# Lazy Torch resolution (so PyInstaller bootstrap and non-GPU users don’t break)
# ---------------------------------------------------------------------------
_TORCH = None
_DEVICE = None

import warnings

_DML_REDUCE_WARNED = False

def _is_directml_privateuse(dev) -> bool:
    """
    torch-directml uses a PrivateUse backend and the device string usually contains:
      'privateuseone:0'
    The reporter typo'd it as 'privateuserone:0' so we accept both.
    """
    try:
        s = str(dev).lower()
    except Exception:
        return False
    return ("privateuseone" in s) or ("privateuserone" in s)

def _warn_dml_reduce_once():
    global _DML_REDUCE_WARNED
    if _DML_REDUCE_WARNED:
        return
    _DML_REDUCE_WARNED = True
    warnings.warn(
        "DirectML detected (privateuseone). Forcing nan reducers (nanmedian/nanquantile/nanstd) to CPU "
        "to avoid a known torch-directml fallback bug (can return empty tensors).",
        RuntimeWarning
    )

def _cpu_reduce_then_back(torch, x, dim: int, op: str, q: float | None = None):
    """
    Force a reduction to run on CPU (workaround for DirectML broken fallbacks),
    then move the result back to x.device.
    """
    x_cpu = x.detach().to("cpu")

    if op == "nanmedian":
        y = torch.nanmedian(x_cpu, dim=dim).values
    elif op == "nanquantile":
        y = torch.nanquantile(x_cpu, float(q), dim=dim)
    elif op == "nanstd":
        y = torch.nanstd(x_cpu, dim=dim, unbiased=False)
    else:
        raise ValueError(f"Unknown op: {op}")

    return y.to(x.device)



def _get_torch(prefer_cuda: bool = True, prefer_dml: bool = False):
    global _TORCH, _DEVICE
    if _TORCH is not None:
        return _TORCH

    try:
        add_runtime_to_sys_path(lambda *_: None)
    except Exception:
        pass

    # Let runtime_torch install/resolve the right torch stack
    torch = import_torch(prefer_cuda=prefer_cuda, prefer_dml=prefer_dml, status_cb=lambda *_: None)

    # IMPORTANT: honor best_device() (it may return DirectML / privateuseone)
    try:
        _DEVICE = best_device(torch, prefer_cuda=prefer_cuda, prefer_dml=prefer_dml)
    except Exception:
        _DEVICE = None

    _TORCH = torch
    _force_fp32_policy(torch)

    # If best_device failed, fall back conservatively
    if _DEVICE is None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                _DEVICE = torch.device("cuda")
            elif getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
                _DEVICE = torch.device("mps")
            else:
                _DEVICE = torch.device("cpu")
        except Exception:
            _DEVICE = torch.device("cpu")

    return _TORCH



def _device():
    if _DEVICE is not None:
        return _DEVICE
    # Default to CPU if torch is not yet resolved; the first GPU call resolves it.
    return None

def torch_available() -> bool:
    """Return True iff we can import/resolve torch via the runtime shim."""
    try:
        _get_torch(prefer_cuda=True)
        return True
    except Exception:
        return False

def gpu_algo_supported(algo_name: str) -> bool:
    return algo_name in _SUPPORTED

# ---------------------------------------------------------------------------
# Helpers (nan-safe reducers) – assume torch is available *inside* callers
# ---------------------------------------------------------------------------
def _nanmedian(torch, x, dim: int):
    # DirectML workaround: force CPU reducer (avoids broken fallback returning empty tensor)
    if _is_directml_privateuse(getattr(x, "device", None)):
        _warn_dml_reduce_once()
        try:
            y = _cpu_reduce_then_back(torch, x, dim, op="nanmedian")
            if getattr(y, "numel", lambda: 1)() == 0:
                raise RuntimeError("nanmedian returned empty tensor on DirectML")
            return y
        except Exception:
            # fall through to generic fallback below
            pass

    try:
        y = torch.nanmedian(x, dim=dim).values
        # Guard: some broken backends return empty tensor instead of real result
        if getattr(y, "numel", lambda: 1)() == 0:
            raise RuntimeError("nanmedian returned empty tensor")
        return y
    except Exception:
        # Generic safe fallback (sort-based)
        m = torch.isfinite(x)
        x2 = x.clone()
        x2[~m] = float("inf")
        idx = x2.argsort(dim=dim)
        cnt = m.sum(dim=dim).clamp_min(1)
        mid = (cnt - 1) // 2
        gather_idx = idx.gather(dim, mid.unsqueeze(dim))
        return x.gather(dim, gather_idx).squeeze(dim)

def _nanstd(torch, x, dim: int):
    if _is_directml_privateuse(getattr(x, "device", None)):
        _warn_dml_reduce_once()
        try:
            y = _cpu_reduce_then_back(torch, x, dim, op="nanstd")
            if getattr(y, "numel", lambda: 1)() == 0:
                raise RuntimeError("nanstd returned empty tensor on DirectML")
            return y
        except Exception:
            pass

    try:
        y = torch.nanstd(x, dim=dim, unbiased=False)
        if getattr(y, "numel", lambda: 1)() == 0:
            raise RuntimeError("nanstd returned empty tensor")
        return y
    except Exception:
        # Fallback: compute nan-safe variance from sums
        m = torch.isfinite(x)
        cnt = m.sum(dim=dim).clamp_min(1)
        s1 = torch.where(m, x, torch.zeros_like(x)).sum(dim=dim)
        s2 = torch.where(m, x * x, torch.zeros_like(x)).sum(dim=dim)
        mean = s1 / cnt
        var = (s2 / cnt) - mean * mean
        return var.clamp_min(0).sqrt()

def _nanquantile(torch, x, q: float, dim: int):
    if _is_directml_privateuse(getattr(x, "device", None)):
        _warn_dml_reduce_once()
        try:
            y = _cpu_reduce_then_back(torch, x, dim, op="nanquantile", q=float(q))
            if getattr(y, "numel", lambda: 1)() == 0:
                raise RuntimeError("nanquantile returned empty tensor on DirectML")
            return y
        except Exception:
            pass

    try:
        y = torch.nanquantile(x, float(q), dim=dim)
        if getattr(y, "numel", lambda: 1)() == 0:
            raise RuntimeError("nanquantile returned empty tensor")
        return y
    except Exception:
        # Fallback: argsort finite values and take kth
        m = torch.isfinite(x)
        x2 = x.clone()
        x2[~m] = float("inf")
        idx = x2.argsort(dim=dim)
        n = m.sum(dim=dim).clamp_min(1)
        kth = (float(q) * (n - 1)).round().to(torch.long).clamp(min=0)
        gather_idx = idx.gather(dim, kth.unsqueeze(dim))
        return x.gather(dim, gather_idx).squeeze(dim)


def _no_amp_ctx(torch, dev):
    """
    Return a context that disables autocast on this thread for the current device.
    Works across torch 1.13–2.x and CUDA/CPU/MPS. No-ops if unsupported.
    """
    import contextlib
    # PyTorch 2.x unified API
    try:
        ac = getattr(torch, "autocast", None)
        if ac is not None:
            dt = "cuda" if getattr(dev, "type", "") == "cuda" else \
                 "mps"  if getattr(dev, "type", "") == "mps"  else "cpu"
            return ac(device_type=dt, enabled=False)
    except Exception:
        pass
    # Older CUDA AMP API
    try:
        amp = getattr(getattr(torch, "cuda", None), "amp", None)
        if amp and hasattr(amp, "autocast"):
            return amp.autocast(enabled=False)
    except Exception:
        pass
    return contextlib.nullcontext()


# --- add near the top (after imports) ---
def _safe_inference_ctx(torch):
    """
    Return a context manager for inference that won't explode on older or
    backend-variant Torch builds (DirectML/MPS/CPU-only).
    """
    try:
        # Prefer inference_mode if both the API and C++ backend support it
        if getattr(torch, "inference_mode", None) is not None:
            _C = getattr(torch, "_C", None)
            if _C is not None and hasattr(_C, "_InferenceMode"):
                return torch.inference_mode()
    except Exception:
        pass
    # Fallbacks
    if getattr(torch, "no_grad", None) is not None:
        return torch.no_grad()
    import contextlib
    return contextlib.nullcontext()

def _force_fp32_policy(torch):
    try:
        # default dtype for new tensors (does not upcast existing)
        torch.set_default_dtype(torch.float32)
    except Exception:
        pass
    # disable “helpful” lower-precision math
    try:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        # prefer strict fp32 matmul kernels where supported
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

def _soft_outlier_weight(torch, z, threshold: float, mode: str = "quartic"):
    """
    Convert absolute z-score into a soft weight in [0,1].

    z:
      absolute standardized residuals
    threshold:
      values >= threshold get weight 0
    mode:
      "linear"  -> 1 - z/t
      "quartic" -> (1 - (z/t)^2)^2
    """
    t = max(float(threshold), 1e-12)
    u = (z / t).clamp_min(0.0)

    if mode == "linear":
        w = 1.0 - u
    else:
        # smoother falloff, less harsh near center
        w = (1.0 - u * u).clamp_min(0.0)
        w = w * w

    return torch.where(u < 1.0, w, torch.zeros_like(u))

# ---------------------------------------------------------------------------
# GPU Calibration Pipeline — dark sub, flat div, cosmetic correction
# All operate on tensors already on device to avoid round-trips
# ---------------------------------------------------------------------------

def _upload_calibration_frame(torch, dev, arr: np.ndarray):
    """Upload a numpy calibration frame (dark or flat) to GPU as float32 tensor.
    Handles 2D (mono), 3D CHW, and 3D HWC layouts — always returns (C, H, W)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[np.newaxis, :, :]          # (1, H, W)
    elif a.ndim == 3 and a.shape[-1] in (1, 3):
        a = a.transpose(2, 0, 1)          # HWC → CHW
    return torch.from_numpy(a).to(dev, dtype=torch.float32, non_blocking=True)


def calibration_pipeline_gpu(
    light_np: np.ndarray,
    dark_t,                          # (C,H,W) tensor on GPU or None
    flat_t,                          # (C,H,W) tensor on GPU or None
    pedestal: float = 0.0,
    hot_sigma: float = 3.0,
    cold_sigma: float = 3.0,
    apply_cosmetic: bool = True,
    bayer_pattern: str | None = None,
) -> np.ndarray:
    """
    Full per-frame calibration pipeline on GPU:
      1. dark subtraction (with pedestal offset)
      2. flat division
      3. cosmetic correction (two-pass)

    light_np: (H,W) mono or (H,W,C) HWC or (C,H,W) CHW float32
    Returns: same layout as input, float32
    """
    torch = _get_torch(prefer_cuda=True)
    dev   = _device() or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    light_np = np.asarray(light_np, dtype=np.float32)

    # remember input layout so we can restore it on output
    was_2d  = (light_np.ndim == 2)
    was_hwc = (light_np.ndim == 3 and light_np.shape[-1] in (1, 3))

    # normalise to CHW for GPU ops
    if was_2d:
        t = torch.from_numpy(light_np[np.newaxis]).to(dev, non_blocking=True)  # (1,H,W)
    elif was_hwc:
        t = torch.from_numpy(
            light_np.transpose(2, 0, 1)
        ).to(dev, non_blocking=True)                                              # (C,H,W)
    else:
        t = torch.from_numpy(light_np).to(dev, non_blocking=True)               # (C,H,W)

    C, H, W = t.shape

    with _safe_inference_ctx(torch), _no_amp_ctx(torch, dev):

        # ── 1. dark subtraction ──────────────────────────────────────
        if dark_t is not None:
            d = dark_t
            if d.shape[0] == 1 and C > 1:
                d = d.expand(C, H, W)
            t = t - d
            if pedestal != 0.0:
                t = t + pedestal
            t = t.clamp(min=0.0)

        # ── 2. flat division ─────────────────────────────────────────
        if flat_t is not None:
            f = flat_t
            if f.shape[0] == 1 and C > 1:
                f = f.expand(C, H, W)
            # flat already normalised (median=1) by caller
            t = t / f.clamp(min=1e-6)

        # ── 3. cosmetic correction ───────────────────────────────────
        if apply_cosmetic:
            t = _cosmetic_correction_tensor(
                torch, dev, t,
                hot_sigma=hot_sigma,
                cold_sigma=cold_sigma,
                bayer_pattern=bayer_pattern,
            )

    # ── back to CPU, restore original layout ─────────────────────────
    out = t.cpu().numpy()               # (C, H, W)
    if was_2d:
        return out[0]                   # (H, W)
    elif was_hwc:
        return out.transpose(1, 2, 0)  # (H, W, C)
    return out                          # (C, H, W)


# Module-level buffer cache — keyed by (device_str, H, W)
_CC_BUFFERS: dict = {}


def _get_cc_buffers(torch, dev, H, W):
    """Get or create persistent reusable buffers for cosmetic correction."""
    key = (str(dev), H, W)
    if key not in _CC_BUFFERS:
        _CC_BUFFERS[key] = {
            "five":         torch.empty((5, H, W), device=dev, dtype=torch.float32),
            "stacked":      torch.empty((8, H, W), device=dev, dtype=torch.float32),
            "sorted_s":     torch.empty((8, H, W), device=dev, dtype=torch.float32),
            "sort_idx":     torch.empty((8, H, W), device=dev, dtype=torch.int64),
            "m5":           torch.empty((H, W),    device=dev, dtype=torch.float32),
            "hot_thresh":   torch.empty((H, W),    device=dev, dtype=torch.float32),
            "cold_thresh":  torch.empty((H, W),    device=dev, dtype=torch.float32),
            "avg3x3":       torch.empty((H, W),    device=dev, dtype=torch.float32),
            "replacement":  torch.empty((H, W),    device=dev, dtype=torch.float32),
            "ones_k":       torch.ones(1, 1, 3, 3, device=dev, dtype=torch.float32),
        }
    return _CC_BUFFERS[key]


def _clear_cc_buffers():
    """Free all cached cosmetic correction buffers. Call after calibration run."""
    _CC_BUFFERS.clear()


def _cosmetic_correction_tensor(torch, dev, t, hot_sigma, cold_sigma, bayer_pattern=None):
    """
    Two-pass cosmetic correction on a (C,H,W) tensor already on device.
    Uses pre-allocated persistent buffers to avoid per-frame cudaMalloc overhead.
    Stride = 2 for Bayer (same-color neighbors), 1 otherwise.
    """
    import torch.nn.functional as F

    s = 2 if bayer_pattern else 1
    C, H, W = t.shape
    buf = _get_cc_buffers(torch, dev, H, W)
    result = torch.empty_like(t)

    # strided sample for fast avgDev — full median on 46M pixels is slow
    # 1% sample is accurate enough for thresholding
    _stride = max(1, min(H, W) // 100)

    for ci in range(C):
        plane = t[ci]  # (H, W)

        # ── fast avgDev via strided sample ────────────────────────────
        sample  = plane[::_stride, ::_stride]
        med_val = sample.median()
        avg_dev = (sample - med_val).abs().mean()
        avg_dev_f = float(avg_dev)
        del sample

        # ── pad plane once — reused for both m5 and neighbor lookups ──
        pad_amt = s
        p = F.pad(
            plane.unsqueeze(0).unsqueeze(0),
            (pad_amt, pad_amt, pad_amt, pad_amt),
            mode='reflect'
        ).squeeze(0).squeeze(0)
        # p: (H + 2s, W + 2s)

        # ── m5: median of {N, S, E, W, center} ───────────────────────
        buf["five"][0].copy_(p[0:H,                 pad_amt:W+pad_amt])        # N
        buf["five"][1].copy_(p[2*pad_amt:H+2*pad_amt, pad_amt:W+pad_amt])      # S
        buf["five"][2].copy_(p[pad_amt:H+pad_amt,   0:W])                      # W
        buf["five"][3].copy_(p[pad_amt:H+pad_amt,   2*pad_amt:W+2*pad_amt])    # E
        buf["five"][4].copy_(plane)                                             # center
        buf["m5"].copy_(buf["five"].median(dim=0).values)
        m5 = buf["m5"]

        # ── detection thresholds ──────────────────────────────────────
        scale_f = max(avg_dev_f * float(hot_sigma), avg_dev_f)
        torch.add(m5, scale_f, out=buf["hot_thresh"])
        torch.sub(m5, avg_dev_f * float(cold_sigma), out=buf["cold_thresh"])

        hot_candidates = plane > buf["hot_thresh"]
        cold_map       = plane < buf["cold_thresh"]

        # ── star guard: 3x3 neighbor average (stride-1 always) ───────
        plane_4d  = plane.unsqueeze(0).unsqueeze(0)
        plane_pad1 = F.pad(plane_4d, (1, 1, 1, 1), mode='reflect')
        box3x3 = F.conv2d(plane_pad1, buf["ones_k"]).squeeze(0).squeeze(0)
        torch.sub(box3x3, plane, out=buf["avg3x3"])
        buf["avg3x3"].div_(8.0)
        del box3x3, plane_pad1, plane_4d

        star_guard = buf["avg3x3"] < (m5 + avg_dev_f * 0.5)
        hot_map    = hot_candidates & star_guard
        flagged    = hot_map | cold_map

        del hot_candidates, cold_map, star_guard, hot_map

        # ── pad flagged mask using same pad_amt ───────────────────────
        fp = F.pad(
            flagged.float().unsqueeze(0).unsqueeze(0),
            (pad_amt, pad_amt, pad_amt, pad_amt),
            mode='reflect'
        ).squeeze(0).squeeze(0)

        # ── build 8-neighbor stack directly into pre-allocated buffer ─
        # flagged neighbors → inf so they sort to the end
        offsets = [
            (-s, -s), (-s,  0), (-s, +s),
            ( 0, -s),           ( 0, +s),
            (+s, -s), (+s,  0), (+s, +s),
        ]

        for i, (dy, dx) in enumerate(offsets):
            ny0, ny1 = dy + pad_amt, dy + pad_amt + H
            nx0, nx1 = dx + pad_amt, dx + pad_amt + W
            nbr_vals    = p[ny0:ny1, nx0:nx1]
            nbr_flagged = fp[ny0:ny1, nx0:nx1] > 0.5
            torch.where(nbr_flagged, nbr_vals.new_full((), float('inf')),
                        nbr_vals, out=buf["stacked"][i])

        del fp, p

        # ── sort to find median of clean neighbors ────────────────────
        torch.sort(buf["stacked"], dim=0, out=(buf["sorted_s"], buf["sort_idx"]))

        # count clean (non-inf) neighbors per pixel
        valid_count = (buf["stacked"] < float('inf')).sum(dim=0).clamp(min=1)

        mid_lo = ((valid_count - 1) // 2).long()
        mid_hi = (valid_count       // 2).long()

        rep_lo = buf["sorted_s"].gather(0, mid_lo.unsqueeze(0)).squeeze(0)
        rep_hi = buf["sorted_s"].gather(0, mid_hi.unsqueeze(0)).squeeze(0)

        torch.add(rep_lo, rep_hi, out=buf["replacement"])
        buf["replacement"].mul_(0.5)

        # fallback to m5 where all 8 neighbors were also flagged
        all_flagged = valid_count == 0
        torch.where(all_flagged, m5, buf["replacement"], out=buf["replacement"])

        del valid_count, mid_lo, mid_hi, rep_lo, rep_hi, all_flagged

        # ── apply correction only to flagged pixels ───────────────────
        torch.where(flagged, buf["replacement"], plane, out=result[ci])

        del flagged

    return result


def prepare_calibration_frame(
    arr: np.ndarray,
    frame_type: str,          # "dark" or "flat"
    normalize_flat: bool = True,
) -> np.ndarray:
    """
    Prepare a master dark or flat for upload:
    - ensures float32
    - for flat: normalises each channel to median=1, guards zeros
    Returns (C, H, W) float32 numpy array ready for _upload_calibration_frame.
    """
    a = np.asarray(arr, dtype=np.float32)

    # normalise layout to CHW
    if a.ndim == 2:
        a = a[np.newaxis]
    elif a.ndim == 3 and a.shape[-1] in (1, 3):
        a = a.transpose(2, 0, 1)

    if frame_type == "flat" and normalize_flat:
        for ci in range(a.shape[0]):
            band  = a[ci]
            v     = band[np.isfinite(band) & (band > 0)]
            denom = float(np.median(v)) if v.size else 1.0
            denom = denom if (np.isfinite(denom) and denom > 0) else 1.0
            a[ci] = band / denom
        a = np.nan_to_num(a, nan=1.0, posinf=1.0, neginf=1.0)
        a[a == 0] = 1.0

    return a

# ---------------------------------------------------------------------------
# GPU Cosmetic Correction
# ---------------------------------------------------------------------------
def cosmetic_correction_gpu(
    image: np.ndarray,
    hot_sigma: float = 3.0,
    cold_sigma: float = 3.0,
    bayer_pattern: str | None = None,
) -> np.ndarray:
    """
    Two-pass cosmetic correction on GPU.
    bayer_pattern: if set (e.g. "RGGB"), uses stride-2 neighbors so only
                   same-color CFA pixels are compared. Otherwise uses stride-1
                   for debayered/mono images.
    """
    torch = _get_torch(prefer_cuda=True)
    dev = _device() or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    img = np.asarray(image, dtype=np.float32)
    was_gray = (img.ndim == 2)
    if was_gray:
        img = img[:, :, None]

    H, W, C = img.shape
    t = torch.from_numpy(img.transpose(2, 0, 1)).to(dev, dtype=torch.float32)
    result = torch.empty_like(t)

    # stride-2 for Bayer (same-color neighbors), stride-1 for mono/debayered
    s = 2 if bayer_pattern else 1

    with _safe_inference_ctx(torch), _no_amp_ctx(torch, dev):
        for ci in range(C):
            plane = t[ci]  # (H, W)

            med_val = plane.median()
            avg_dev = (plane - med_val).abs().mean()

            # pad by stride amount so border pixels are handled cleanly
            p = torch.nn.functional.pad(
                plane.unsqueeze(0).unsqueeze(0),
                (s, s, s, s), mode='reflect'
            ).squeeze(0).squeeze(0)
            # p: (H+2s, W+2s)

            north  = p[0:H,       s:W+s  ]
            south  = p[2*s:H+2*s, s:W+s  ]
            west   = p[s:H+s,     0:W    ]
            east   = p[s:H+s,     2*s:W+2*s]
            center = plane

            five = torch.stack([north, south, west, east, center], dim=0)
            m5 = five.median(dim=0).values

            scale       = torch.clamp(avg_dev * hot_sigma, min=float(avg_dev))
            hot_thresh  = m5 + scale
            cold_thresh = m5 - avg_dev * cold_sigma

            hot_candidates = plane > hot_thresh
            cold_map       = plane < cold_thresh

            # star guard: 3x3 (or 5x5 for Bayer to cover same stride) neighbor average
            guard_pad = s * 2 if bayer_pattern else 1
            ones_k = torch.ones(1, 1, 3, 3, device=dev, dtype=torch.float32)
            plane_4d  = plane.unsqueeze(0).unsqueeze(0)
            plane_pad = torch.nn.functional.pad(plane_4d, (1, 1, 1, 1), mode='reflect')
            box3x3    = torch.nn.functional.conv2d(plane_pad, ones_k).squeeze(0).squeeze(0)
            avg3x3_neighbors = (box3x3 - plane) / 8.0
            star_guard = avg3x3_neighbors < (m5 + avg_dev * 0.5)
            hot_map    = hot_candidates & star_guard

            flagged = hot_map | cold_map

            # 8 neighbors at stride s
            offsets = [
                (-s, -s), (-s,  0), (-s, +s),
                ( 0, -s),           ( 0, +s),
                (+s, -s), (+s,  0), (+s, +s),
            ]

            flagged_pad = torch.nn.functional.pad(
                flagged.float().unsqueeze(0).unsqueeze(0),
                (s, s, s, s), mode='reflect'
            ).squeeze().bool()

            plane_pad2 = torch.nn.functional.pad(
                plane.unsqueeze(0).unsqueeze(0),
                (s, s, s, s), mode='reflect'
            ).squeeze()

            neighbor_stack = []
            for dy, dx in offsets:
                ny0, ny1 = dy + s, dy + s + H
                nx0, nx1 = dx + s, dx + s + W
                nbr_vals    = plane_pad2[ny0:ny1, nx0:nx1]
                nbr_flagged = flagged_pad[ny0:ny1, nx0:nx1]
                nbr_clean   = torch.where(
                    nbr_flagged,
                    torch.full_like(nbr_vals, float('nan')),
                    nbr_vals
                )
                neighbor_stack.append(nbr_clean)

            stacked     = torch.stack(neighbor_stack, dim=0)
            stacked_inf = torch.where(torch.isnan(stacked),
                                      torch.full_like(stacked, float('inf')), stacked)
            sorted_s, _ = stacked_inf.sort(dim=0)

            valid_count = (~torch.isnan(stacked)).sum(dim=0).clamp(min=1)
            mid_lo = ((valid_count - 1) // 2).long()
            mid_hi = (valid_count       // 2).long()

            rep_lo      = sorted_s.gather(0, mid_lo.unsqueeze(0)).squeeze(0)
            rep_hi      = sorted_s.gather(0, mid_hi.unsqueeze(0)).squeeze(0)
            replacement = (rep_lo + rep_hi) * 0.5
            replacement = torch.where(torch.isinf(replacement), m5, replacement)

            result[ci] = torch.where(flagged, replacement, plane)

    out = result.cpu().numpy().transpose(1, 2, 0)
    if was_gray:
        return out[:, :, 0]
    return out

# ---------------------------------------------------------------------------
# Public GPU reducer – lazy-loads Torch, never decorates at import time
# ---------------------------------------------------------------------------
def torch_reduce_tile(
    ts_np: np.ndarray,
    weights_np: np.ndarray,
    *,
    algo_name: str,
    kappa: float = 2.5,
    iterations: int = 3,
    sigma_low: float = 2.5,
    sigma_high: float = 2.5,
    trim_fraction: float = 0.1,
    esd_threshold: float = 3.0,
    biweight_constant: float = 6.0,
    modz_threshold: float = 3.5,
    comet_hclip_k: float = 1.30,
    comet_hclip_p: float = 25.0,
    ignore_zero_pixels: bool = True,
    forced_reject_mask_np: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns: (tile_result, tile_rej_map)
      tile_result: (th, tw, C) float32
      tile_rej_map: (F, th, tw, C) bool   (collapse C on caller if needed)

    forced_reject_mask_np:
      Optional binary reject mask for this tile.
      Accepted shapes:
        (H, W)       -> broadcast to all frames/channels
        (F, H, W)    -> per-frame mask, broadcast across channels
        (F, H, W, 1) -> per-frame mask, already channel-shaped
        (F, H, W, C) -> full per-frame/per-channel mask
    """
    torch = _get_torch(prefer_cuda=True)
    dev = _device() or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    ts_np = np.asarray(ts_np, dtype=np.float32)
    if ts_np.ndim == 3:
        ts_np = ts_np[..., None]
    F, H, W, C = ts_np.shape

    if H == 0 or W == 0 or C < 1:
        raise ValueError(
            f"torch_reduce_tile received degenerate tile shape={ts_np.shape}. "
            "This usually means a bad edge tile or corrupted frame; "
            "try disabling GPU rejection or reducing chunk size."
        )
    if C < 1:
        raise ValueError(f"torch_reduce_tile received input with C={C} channels (shape={ts_np.shape}). Expected C >= 1.")

    ts = torch.from_numpy(ts_np).to(dev, dtype=torch.float32, non_blocking=True)

    weights_np = np.asarray(weights_np, dtype=np.float32)
    if weights_np.ndim == 0:
        weights_np = np.full((F,), float(weights_np), dtype=np.float32)

    if weights_np.ndim == 1:
        if weights_np.shape[0] != F:
            raise ValueError(f"weights shape {weights_np.shape} does not match F={F}")
        w_np = weights_np.reshape(F, 1, 1, 1)

    elif weights_np.ndim == 2:
        if weights_np.shape == (F, C):
            w_np = weights_np.reshape(F, 1, 1, C)
        else:
            raise ValueError(
                f"Unsupported 2D weights shape {weights_np.shape}. "
                f"Expected (F,C)=({F},{C}) for per-channel weights."
            )

    elif weights_np.ndim == 3:
        if weights_np.shape == (F, H, W):
            w_np = weights_np[..., None]
        else:
            raise ValueError(
                f"Unsupported 3D weights shape {weights_np.shape}. Expected (F,H,W)=({F},{H},{W})."
            )

    elif weights_np.ndim == 4:
        if weights_np.shape == (F, H, W, 1) or weights_np.shape == (F, H, W, C):
            w_np = weights_np
        else:
            raise ValueError(
                f"Unsupported 4D weights shape {weights_np.shape}. "
                f"Expected (F,H,W,1) or (F,H,W,C)=({F},{H},{W},{C})."
            )
    else:
        raise ValueError(f"Unsupported weights ndim={weights_np.ndim} shape={weights_np.shape}")

    w = torch.from_numpy(w_np).to(dev, dtype=torch.float32, non_blocking=False)

    # ---------- NEW: normalize optional forced reject mask ----------
    forced_reject = None
    if forced_reject_mask_np is not None:
        frm = np.asarray(forced_reject_mask_np, dtype=bool)

        if frm.ndim == 2:
            if frm.shape != (H, W):
                raise ValueError(
                    f"forced_reject_mask_np shape {frm.shape} does not match tile (H,W)=({H},{W})"
                )
            frm = np.broadcast_to(frm[None, :, :, None], (F, H, W, C))

        elif frm.ndim == 3:
            if frm.shape != (F, H, W):
                raise ValueError(
                    f"forced_reject_mask_np shape {frm.shape} does not match (F,H,W)=({F},{H},{W})"
                )
            frm = np.broadcast_to(frm[:, :, :, None], (F, H, W, C))

        elif frm.ndim == 4:
            if frm.shape == (F, H, W, 1):
                frm = np.broadcast_to(frm, (F, H, W, C))
            elif frm.shape != (F, H, W, C):
                raise ValueError(
                    f"forced_reject_mask_np shape {frm.shape} does not match "
                    f"(F,H,W,1) or (F,H,W,C)=({F},{H},{W},{C})"
                )
        else:
            raise ValueError(
                f"Unsupported forced_reject_mask_np ndim={frm.ndim} shape={frm.shape}"
            )

        forced_reject = torch.from_numpy(np.array(frm, dtype=bool, order="C")).to(dev, dtype=torch.bool, non_blocking=False)

    algo = algo_name

    # ---------- central validity mask ----------
    valid = torch.isfinite(ts)
    if ignore_zero_pixels:
        valid = valid & (ts != 0.0)
    if forced_reject is not None:
        valid = valid & (~forced_reject)

    # Base forced rejection bookkeeping
    forced_rej_map = (~valid)

    with _safe_inference_ctx(torch), _no_amp_ctx(torch, dev):
        if algo in ("Comet Median", "Simple Median (No Rejection)"):
            x = ts.masked_fill(~valid, float("nan"))
            out = _nanmedian(torch, x, dim=0)
            rej = forced_rej_map
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Comet Percentile (40th)":
            x = ts.masked_fill(~valid, float("nan"))
            out = _nanquantile(torch, x, 0.40, dim=0)
            rej = forced_rej_map
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo_name == "Windsorized Sigma Clipping":
            low = float(sigma_low)
            high = float(sigma_high)
            keep = valid.clone()

            for _ in range(int(iterations)):
                x_iter = ts.masked_fill(~keep, float("nan"))
                med = _nanmedian(torch, x_iter, dim=0)
                sq_dev = torch.where(keep, (ts - med.unsqueeze(0)) ** 2, torch.zeros_like(ts))
                n_kept = keep.sum(dim=0).clamp_min(1).to(ts.dtype)
                std = (sq_dev.sum(dim=0) / n_kept).sqrt().clamp_min(1e-12)
                lo = med - low * std
                hi = med + high * std
                keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))

            rej = ~keep

            ts_winsorized = torch.where(valid & keep, ts, torch.where(valid, med.unsqueeze(0), torch.zeros_like(ts)))
            cnt = valid.sum(dim=0).to(ts.dtype)
            num = torch.where(valid, ts_winsorized, torch.zeros_like(ts)).sum(dim=0)
            x_valid = ts.masked_fill(~valid, float("nan"))
            fallback = _nanmedian(torch, x_valid, dim=0)
            fallback = torch.nan_to_num(fallback, nan=0.0)
            out = torch.where(cnt > 0, num / cnt.clamp_min(1), fallback)

            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Comet Lower-Trim (30%)":
            x = ts.masked_fill(~valid, float("inf"))
            vals, idx = x.sort(dim=0, stable=True)

            n = valid.sum(dim=0)
            k_keep = torch.floor(n.to(torch.float32) * (1.0 - 0.30)).to(torch.long)
            k_keep = k_keep.clamp(min=1)

            arangeF = torch.arange(F, device=dev).view(F, 1, 1, 1).expand_as(vals)
            keep_sorted = arangeF < k_keep.unsqueeze(0).expand_as(vals)
            keep_sorted = keep_sorted & torch.isfinite(vals)

            den = keep_sorted.sum(dim=0)
            num = torch.where(keep_sorted, vals, torch.zeros_like(vals)).sum(dim=0)

            fallback = _nanmedian(torch, ts.masked_fill(~valid, float("nan")), dim=0)
            out = torch.where(den > 0, num / den.clamp_min(1).to(vals.dtype), fallback)

            keep_orig = torch.zeros_like(keep_sorted)
            keep_orig.scatter_(0, idx, keep_sorted)
            rej = ~keep_orig
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Comet High-Clip Percentile":
            x = ts.masked_fill(~valid, float("nan"))
            med = _nanmedian(torch, x, dim=0)
            mad = _nanmedian(torch, (x - med.unsqueeze(0)).abs(), dim=0) + 1e-6
            hi = med + (float(comet_hclip_k) * 1.4826 * mad)
            clipped = torch.where(valid, torch.minimum(ts, hi.unsqueeze(0)), torch.full_like(ts, float("nan")))
            out = _nanquantile(torch, clipped, float(comet_hclip_p) / 100.0, dim=0)
            rej = forced_rej_map
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Simple Average (No Rejection)":
            w_eff = torch.where(valid, w, torch.zeros_like(w))
            ts_eff = torch.where(valid, ts, torch.zeros_like(ts))
            den = w_eff.sum(dim=0)
            num = (ts_eff * w_eff).sum(dim=0)

            x = ts.masked_fill(~valid, float("nan"))
            fallback = _nanmedian(torch, x, dim=0)
            out = torch.where(den > 0, num / den.clamp_min(1e-20), fallback)

            rej = forced_rej_map
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Max Value":
            x = ts.masked_fill(~valid, float("-inf"))
            out = x.max(dim=0).values
            out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
            rej = forced_rej_map
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()


        if algo == "Weighted Windsorized Sigma Clipping":
            low = float(sigma_low)
            high = float(sigma_high)
            keep = valid.clone()

            for _ in range(int(iterations)):
                x_iter = ts.masked_fill(~keep, float("nan"))
                med = _nanmedian(torch, x_iter, dim=0)
                sq_dev = torch.where(keep, (ts - med.unsqueeze(0)) ** 2, torch.zeros_like(ts))
                n_kept = keep.sum(dim=0).clamp_min(1).to(ts.dtype)
                std = (sq_dev.sum(dim=0) / n_kept).sqrt().clamp_min(1e-12)
                lo = med - low * std
                hi = med + high * std
                keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))

            # True Winsorized mean: replace rejected pixels with the converged median
            # (best estimate of true value), not the boundary — boundary still biases.
            ts_winsorized = torch.where(valid & keep, ts, torch.where(valid, med.unsqueeze(0), torch.zeros_like(ts)))

            w_valid = torch.where(valid, w, torch.zeros_like(w))
            num = (ts_winsorized * w_valid).sum(dim=0)
            den = w_valid.sum(dim=0)

            x_valid = ts.masked_fill(~valid, float("nan"))
            fallback = _nanmedian(torch, x_valid, dim=0)
            fallback = torch.nan_to_num(fallback, nan=0.0)
            out = torch.where(den > 1e-20, num / den.clamp_min(1e-20), fallback)

            rej = ~keep
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Extreme Studentized Deviate (ESD)":
            x = ts.masked_fill(~valid, float("nan"))
            keep = valid.clone()

            for _ in range(int(iterations)):
                x_iter = ts.masked_fill(~keep, float("nan"))
                med = _nanmedian(torch, x_iter, dim=0)
                sq_dev = torch.where(keep, (ts - med.unsqueeze(0)) ** 2, torch.zeros_like(ts))
                n_kept = keep.sum(dim=0).clamp_min(1).to(ts.dtype)
                std = (sq_dev.sum(dim=0) / n_kept).sqrt().clamp_min(1e-12)
                z = torch.where(
                    valid,
                    (ts - med.unsqueeze(0)).abs() / std.unsqueeze(0),
                    torch.full_like(ts, float("inf"))
                )
                keep = valid & (z < float(esd_threshold))

            # med here is the converged median of kept pixels — the best unbiased
            # location estimate. Use weighted mean of kept pixels, but where
            # rejection was heavy (>50% of valid frames rejected), trust the
            # converged median more than the kept-pixel mean which is a biased tail.
            w_eff = torch.where(keep, w, torch.zeros_like(w))
            num = torch.where(keep, ts * w_eff, torch.zeros_like(ts)).sum(dim=0)
            den = w_eff.sum(dim=0)

            n_valid = valid.sum(dim=0).to(ts.dtype)
            n_kept_final = keep.sum(dim=0).to(ts.dtype)
            rej_frac = 1.0 - (n_kept_final / n_valid.clamp_min(1))

            # weighted mean of kept pixels
            mean_kept = torch.where(den > 1e-20, num / den.clamp_min(1e-20), med)
            # blend: heavy rejection → trust converged median; light rejection → trust mean
            # threshold at 30% rejection: above that, blend toward median
            blend = (rej_frac - 0.30).clamp(0.0, 0.70) / 0.70  # 0 at <30%, 1 at >100% rej
            out = (1.0 - blend) * mean_kept + blend * med

            fallback = _nanmedian(torch, x, dim=0)
            fallback = torch.nan_to_num(fallback, nan=0.0)
            any_valid = n_valid > 0
            out = torch.where(any_valid, out, fallback)

            rej = ~keep
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Kappa-Sigma Clipping":
            keep = valid.clone()

            for _ in range(int(iterations)):
                x_iter = ts.masked_fill(~keep, float("nan"))
                med = _nanmedian(torch, x_iter, dim=0)
                sq_dev = torch.where(keep, (ts - med.unsqueeze(0)) ** 2, torch.zeros_like(ts))
                n_kept = keep.sum(dim=0).clamp_min(1).to(ts.dtype)
                std = (sq_dev.sum(dim=0) / n_kept).sqrt().clamp_min(1e-12)
                lo = med - float(kappa) * std
                hi = med + float(kappa) * std
                keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))

            w_eff = torch.where(keep, w, torch.zeros_like(w))
            num = torch.where(keep, ts * w_eff, torch.zeros_like(ts)).sum(dim=0)
            den = w_eff.sum(dim=0)

            n_valid = valid.sum(dim=0).to(ts.dtype)
            n_kept_final = keep.sum(dim=0).to(ts.dtype)
            rej_frac = 1.0 - (n_kept_final / n_valid.clamp_min(1))

            x_valid = ts.masked_fill(~valid, float("nan"))
            mean_kept = torch.where(den > 1e-20, num / den.clamp_min(1e-20), med)
            blend = (rej_frac - 0.30).clamp(0.0, 0.70) / 0.70
            out = (1.0 - blend) * mean_kept + blend * med

            fallback = _nanmedian(torch, x_valid, dim=0)
            fallback = torch.nan_to_num(fallback, nan=0.0)
            any_valid = n_valid > 0
            out = torch.where(any_valid, out, fallback)

            rej = ~keep
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Trimmed Mean":
            x = ts.masked_fill(~valid, float("nan"))
            qlo = _nanquantile(torch, x, trim_fraction, dim=0)
            qhi = _nanquantile(torch, x, 1.0 - trim_fraction, dim=0)
            keep = valid & (ts >= qlo.unsqueeze(0)) & (ts <= qhi.unsqueeze(0))
            w_eff = torch.where(keep, w, torch.zeros_like(w))
            den = w_eff.sum(dim=0).clamp_min(1e-20)
            out = (ts.mul(w_eff)).sum(dim=0).div(den)
            rej = ~keep
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Biweight Estimator":
            x = ts
            m = _nanmedian(torch, x.masked_fill(~valid, float("nan")), dim=0)
            mad = _nanmedian(torch, (x - m.unsqueeze(0)).abs().masked_fill(~valid, float("nan")), dim=0) + 1e-12
            u = (x - m.unsqueeze(0)) / (float(biweight_constant) * mad.unsqueeze(0))
            mask = valid & (u.abs() < 1.0)
            w_eff = torch.where(mask, w, torch.zeros_like(w))
            one_minus_u2 = (1 - u**2).clamp_min(0)
            num = ((x - m.unsqueeze(0)) * (one_minus_u2**2) * w_eff).sum(dim=0)
            den = ((one_minus_u2**2) * w_eff).sum(dim=0)
            out = torch.where(den > 0, m + num / den, m)
            rej = ~mask
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Modified Z-Score Clipping":
            x = ts
            med = _nanmedian(torch, x.masked_fill(~valid, float("nan")), dim=0)
            mad = _nanmedian(torch, (x - med.unsqueeze(0)).abs().masked_fill(~valid, float("nan")), dim=0) + 1e-12
            mz = 0.6745 * (x - med.unsqueeze(0)) / mad.unsqueeze(0)
            keep = valid & (mz.abs() < float(modz_threshold))
            w_eff = torch.where(keep, w, torch.zeros_like(w))
            den = w_eff.sum(dim=0).clamp_min(1e-20)
            out = (ts.mul(w_eff)).sum(dim=0).div(den)
            rej = ~keep
            return out.to(dtype=torch.float32).contiguous().cpu().numpy(), rej.cpu().numpy()

        raise NotImplementedError(f"GPU path not implemented for: {algo_name}")