# pro/torch_rejection.py
from __future__ import annotations
import contextlib
import numpy as np

# Always route through our runtime shim so ALL GPU users share the same backend.
# Nothing heavy happens at import; we only resolve Torch when needed.
from .runtime_torch import import_torch, add_runtime_to_sys_path

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

def _get_torch(prefer_cuda: bool = True):
    """
    Resolve and cache the torch module via the SAS runtime shim.
    This may install/repair torch into the per-user runtime if needed.
    """
    global _TORCH, _DEVICE
    if _TORCH is not None:
        return _TORCH

    # In frozen builds, help the process see the runtime site-packages first.
    try:
        add_runtime_to_sys_path(lambda *_: None)
    except Exception:
        pass

    # Import (and if necessary, install) torch using the unified runtime.
    torch = import_torch(prefer_cuda=prefer_cuda, status_cb=lambda *_: None)
    _TORCH = torch

    # Choose the best device once; cheap calls, but cached anyway
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
    try:
        return torch.nanmedian(x, dim=dim).values
    except Exception:
        m = torch.isfinite(x)
        x2 = x.clone()
        x2[~m] = float("inf")
        idx = x2.argsort(dim=dim)
        cnt = m.sum(dim=dim).clamp_min(1)
        mid = (cnt - 1) // 2
        gather_idx = idx.gather(dim, mid.unsqueeze(dim))
        return x.gather(dim, gather_idx).squeeze(dim)

def _nanstd(torch, x, dim: int):
    try:
        return torch.nanstd(x, dim=dim, unbiased=False)
    except Exception:
        m = torch.isfinite(x)
        cnt = m.sum(dim=dim).clamp_min(1)
        s1 = torch.where(m, x, torch.zeros_like(x)).sum(dim=dim)
        s2 = torch.where(m, x * x, torch.zeros_like(x)).sum(dim=dim)
        mean = s1 / cnt
        var = (s2 / cnt) - mean * mean
        return var.clamp_min(0).sqrt()

def _nanquantile(torch, x, q: float, dim: int):
    try:
        return torch.nanquantile(x, q, dim=dim)
    except Exception:
        m = torch.isfinite(x)
        x2 = x.clone()
        x2[~m] = float("inf")
        idx = x2.argsort(dim=dim)
        n = m.sum(dim=dim).clamp_min(1)
        kth = (q * (n - 1)).round().to(torch.long)
        kth = kth.clamp(min=0)
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


# ---------------------------------------------------------------------------
# Public GPU reducer – lazy-loads Torch, never decorates at import time
# ---------------------------------------------------------------------------
def torch_reduce_tile(
    ts_np: np.ndarray,            # (F, th, tw, C) or (F, th, tw) -> treated as C=1
    weights_np: np.ndarray,       # (F,) or (F, th, tw, C)
    *,
    algo_name: str,
    kappa: float = 2.5,
    iterations: int = 3,
    sigma_low: float = 2.5,       # for winsorized
    sigma_high: float = 2.5,      # for winsorized
    trim_fraction: float = 0.1,   # for trimmed mean
    esd_threshold: float = 3.0,   # for ESD
    biweight_constant: float = 6.0,  # for biweight
    modz_threshold: float = 3.5,  # for modified z
    comet_hclip_k: float = 1.30,  # for comet high-clip percentile
    comet_hclip_p: float = 25.0,  # for comet high-clip percentile
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns: (tile_result, tile_rej_map)
      tile_result: (th, tw, C) float32
      tile_rej_map: (F, th, tw, C) bool   (collapse C on caller if needed)
    """
    # Resolve torch on demand, using the SAME backend as the rest of the app.
    torch = _get_torch(prefer_cuda=True)
    dev = _device() or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Normalize shape to 4D float32
    ts_np = np.asarray(ts_np, dtype=np.float32)
    if ts_np.ndim == 3:
        ts_np = ts_np[..., None]
    F, H, W, C = ts_np.shape

    # Host → device
    ts = torch.from_numpy(ts_np).to(dev, dtype=torch.float32, non_blocking=True)

    # Weights broadcast to 4D
    weights_np = np.asarray(weights_np, dtype=np.float32)
    if weights_np.ndim == 1:
        w = torch.from_numpy(weights_np).to(dev, dtype=torch.float32, non_blocking=True).view(F,1,1,1)
    else:
        w = torch.from_numpy(weights_np).to(dev, dtype=torch.float32, non_blocking=True)

    algo = algo_name
    valid = torch.isfinite(ts)

    # Use inference_mode if present; else nullcontext.
    with _safe_inference_ctx(torch), _no_amp_ctx(torch, dev):
        # ---------------- simple, no-rejection reducers ----------------
        if algo in ("Comet Median", "Simple Median (No Rejection)"):
            out = ts.median(dim=0).values
            rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Comet Percentile (40th)":
            out = _nanquantile(torch, ts, 0.40, dim=0)
            rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo_name == "Windsorized Sigma Clipping":
            # Unweighted: mask by k*sigma around median, then plain mean of survivors
            low = float(sigma_low)
            high = float(sigma_high)
            valid = torch.isfinite(ts)
            keep = valid.clone()
            for _ in range(int(iterations)):
                x = ts.masked_fill(~keep, float("nan"))
                med = _nanmedian(torch, x, dim=0)
                std = _nanstd(torch, x, dim=0)
                lo = med - low * std
                hi = med + high * std
                keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))
            # plain (unweighted) mean of kept samples
            kept = torch.where(keep, ts, torch.zeros_like(ts))
            cnt  = keep.sum(dim=0)

            # Fallback to nanmedian where nothing survived
            mask_no = (cnt == 0)
            if mask_no.any():
                x = ts.masked_fill(~torch.isfinite(ts), float("nan"))
                fallback = _nanmedian(torch, x, dim=0)
                kept = kept.clone()
                kept[mask_no] = fallback[mask_no]
                cnt = torch.where(mask_no, torch.ones_like(cnt), cnt)

            out = kept.sum(dim=0) / cnt.clamp_min(1)
            rej  = ~keep
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()


        if algo == "Comet Lower-Trim (30%)":
            n = torch.isfinite(ts).sum(dim=0).clamp_min(1)
            k_keep = torch.floor(n * (1.0 - 0.30)).to(torch.long).clamp(min=1)
            vals, idx = ts.sort(dim=0, stable=True)
            arangeF = torch.arange(F, device=dev).view(F, 1, 1, 1).expand_as(vals)
            keep = arangeF < k_keep.unsqueeze(0).expand_as(vals)
            den = keep.sum(dim=0).clamp_min(1).to(vals.dtype)
            out = (vals * keep).sum(dim=0) / den
            keep_orig = torch.zeros_like(keep)
            keep_orig.scatter_(0, idx, keep)
            rej = ~keep_orig
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Comet High-Clip Percentile":
            med = _nanmedian(torch, ts, dim=0)
            mad = _nanmedian(torch, (ts - med.unsqueeze(0)).abs(), dim=0) + 1e-6
            hi = med + (float(comet_hclip_k) * 1.4826 * mad)
            clipped = torch.minimum(ts, hi.unsqueeze(0))
            out = _nanquantile(torch, clipped, float(comet_hclip_p) / 100.0, dim=0)
            rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Simple Average (No Rejection)":
            num = (ts * w).sum(dim=0)
            den = w.sum(dim=0).clamp_min(1e-20)
            out = (num / den)
            rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Max Value":
            out = ts.max(dim=0).values
            rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        # ---------------- rejection-based reducers ----------------
        if algo == "Kappa-Sigma Clipping":
            keep = valid.clone()
            for _ in range(int(iterations)):
                x = ts.masked_fill(~keep, float("nan"))
                med = _nanmedian(torch, x, dim=0)
                std = _nanstd(torch, x, dim=0)
                lo = med - float(kappa) * std
                hi = med + float(kappa) * std
                keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))
            w_eff = torch.where(keep, w, torch.zeros_like(w))
            den = w_eff.sum(dim=0).clamp_min(1e-20)
            out = (ts.mul(w_eff)).sum(dim=0).div(den)
            rej = ~keep
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Weighted Windsorized Sigma Clipping":
            low = float(sigma_low); high = float(sigma_high)
            keep = valid.clone()
            for _ in range(int(iterations)):
                x = ts.masked_fill(~keep, float("nan"))
                med = _nanmedian(torch, x, dim=0)
                std = _nanstd(torch, x, dim=0)
                lo = med - low * std
                hi = med + high * std
                keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))
            w_eff = torch.where(keep, w, torch.zeros_like(w))
            den = w_eff.sum(dim=0)
            num = (ts * w_eff).sum(dim=0)
            out = torch.empty((H, W, C), dtype=ts.dtype, device=dev)
            mask_no = den <= 0
            if mask_no.any():
                x = ts.masked_fill(~valid, float("nan"))
                out_fallback = _nanmedian(torch, x, dim=0)
                out[mask_no] = out_fallback[mask_no]
            if (~mask_no).any():
                out[~mask_no] = (num[~mask_no] / den[~mask_no])
            rej = ~keep
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Trimmed Mean":
            x = ts.masked_fill(~valid, float("nan"))
            qlo = _nanquantile(torch, x, trim_fraction, dim=0)
            qhi = _nanquantile(torch, x, 1.0 - trim_fraction, dim=0)
            keep = valid & (ts >= qlo.unsqueeze(0)) & (ts <= qhi.unsqueeze(0))
            w_eff = torch.where(keep, w, torch.zeros_like(w))
            den = w_eff.sum(dim=0).clamp_min(1e-20)
            out = (ts.mul(w_eff)).sum(dim=0).div(den)
            rej = ~keep
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        if algo == "Extreme Studentized Deviate (ESD)":
            x = ts.masked_fill(~valid, float("nan"))
            mean = torch.where(torch.isfinite(x), x, torch.zeros_like(x)).nanmean(dim=0)
            std = _nanstd(torch, x, dim=0).clamp_min(1e-12)
            z = (ts - mean.unsqueeze(0)).abs() / std.unsqueeze(0)
            keep = valid & (z < float(esd_threshold))
            w_eff = torch.where(keep, w, torch.zeros_like(w))
            den = w_eff.sum(dim=0).clamp_min(1e-20)
            out = (ts.mul(w_eff)).sum(dim=0).div(den)
            rej = ~keep
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

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
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

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
            assert out.dtype == torch.float32, f"reducer produced {out.dtype}, expected float32"
            return out.contiguous().cpu().numpy(), rej.cpu().numpy()

        raise NotImplementedError(f"GPU path not implemented for: {algo_name}")
