# pro/torch_rejection.py
from __future__ import annotations
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# Full set supported here
_SUPPORTED = {
    "Comet Median",                        # == Simple Median (No Rejection)
    "Simple Median (No Rejection)",
    "Comet High-Clip Percentile",
    "Comet Lower-Trim (30%)",
    "Comet Percentile (40th)",
    "Simple Average (No Rejection)",
    "Weighted Windsorized Sigma Clipping",
    "Kappa-Sigma Clipping",
    "Trimmed Mean",
    "Extreme Studentized Deviate (ESD)",
    "Biweight Estimator",
    "Modified Z-Score Clipping",
    "Max Value",
}

def torch_available() -> bool:
    return _HAS_TORCH

def gpu_algo_supported(algo_name: str) -> bool:
    return algo_name in _SUPPORTED

def _pick_device():
    if not _HAS_TORCH:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------- helpers (nan-safe reducers) ----------

def _nanmedian(x: torch.Tensor, dim: int):
    # torch.nanmedian exists in recent PyTorch; fall back if not available.
    try:
        return torch.nanmedian(x, dim=dim).values
    except Exception:
        m = torch.isfinite(x)
        # Replace NaNs with +inf, take min of absolute deviation trick isn’t robust for median.
        # Fallback: sort with mask.
        x2 = x.clone()
        x2[~m] = float("inf")
        idx = x2.argsort(dim=dim)
        # count finite
        cnt = m.sum(dim=dim).clamp_min(1)
        mid = (cnt - 1) // 2
        # gather the median per-slice
        rng = [slice(None)] * x.ndim
        rng[dim] = mid
        gather_idx = idx.gather(dim, mid.unsqueeze(dim))
        return x.gather(dim, gather_idx).squeeze(dim)

def _nanstd(x: torch.Tensor, dim: int):
    try:
        return torch.nanstd(x, dim=dim, unbiased=False)
    except Exception:
        m = torch.isfinite(x)
        cnt = m.sum(dim=dim).clamp_min(1)
        s1 = torch.where(m, x, torch.zeros_like(x)).sum(dim=dim)
        s2 = torch.where(m, x*x, torch.zeros_like(x)).sum(dim=dim)
        mean = s1 / cnt
        var = (s2 / cnt) - mean * mean
        return var.clamp_min(0).sqrt()

def _nanquantile(x: torch.Tensor, q: float, dim: int):
    # Prefer nanquantile; else manual
    try:
        return torch.nanquantile(x, q, dim=dim)
    except Exception:
        m = torch.isfinite(x)
        # Replace NaNs with +inf so they sort to the end
        x2 = x.clone()
        x2[~m] = float("inf")
        idx = x2.argsort(dim=dim)
        # effective length per slice
        n = m.sum(dim=dim).clamp_min(1)
        kth = (q * (n - 1)).round().to(torch.long)
        kth = kth.clamp(min=0)
        gather_idx = idx.gather(dim, kth.unsqueeze(dim))
        return x.gather(dim, gather_idx).squeeze(dim)


@torch.no_grad()
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
    if not _HAS_TORCH:
        raise RuntimeError("torch not available")

    dev = _pick_device()
    if dev is None:
        raise RuntimeError("no torch device")

    # Normalize shape to 4D
    ts_np = np.asarray(ts_np, dtype=np.float32)
    if ts_np.ndim == 3:
        ts_np = ts_np[..., None]
    F, H, W, C = ts_np.shape

    ts = torch.from_numpy(ts_np).to(dev, non_blocking=True)  # (F,H,W,C)

    # Weights to broadcastable 4D
    weights_np = np.asarray(weights_np, dtype=np.float32)
    if weights_np.ndim == 1:
        w = torch.from_numpy(weights_np).to(dev, non_blocking=True).view(F, 1, 1, 1)
    else:
        w = torch.from_numpy(weights_np).to(dev, non_blocking=True)

    algo = algo_name

    # Valid = non-zero samples (per-channel)
    valid = (ts != 0.0)

    # ---------------- simple, no-rejection reducers ----------------
    if algo in ("Comet Median", "Simple Median (No Rejection)"):
        out = ts.median(dim=0).values
        rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)

        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Comet Percentile (40th)":
        out = _nanquantile(ts, 0.40, dim=0)
        rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Comet Lower-Trim (30%)":
        # keep lowest k across F then mean
        n = torch.isfinite(ts).sum(dim=0).clamp_min(1)  # counts (H,W,C)
        k_keep = torch.floor(n * (1.0 - 0.30)).to(torch.long).clamp(min=1)
        # topk with largest=False gives the lowest values and their indices
        # We need a fixed k across (H,W,C); use per-pixel k via masking.
        # Strategy: sort then take first k per-slice with masking.
        vals, idx = ts.sort(dim=0, stable=True)  # (F,H,W,C) ascending; NaNs to end—ensure no NaN by treating zeros as zeros (valid is separate)
        arangeF = torch.arange(F, device=dev).view(F, 1, 1, 1).expand_as(vals)
        # Build keep mask: rank < k_keep
        rank = arangeF  # since sorted ascending, index is rank
        kk = k_keep.unsqueeze(0).expand_as(vals)
        keep = rank < kk
        # Weighted mean of kept samples (uniform weights)
        den = keep.sum(dim=0).clamp_min(1).to(vals.dtype)
        out = (vals * keep).sum(dim=0) / den
        rej = ~keep.scatter(0, idx, keep)  # produce rejection aligned to original frames
        # The above rej is still in sorted order; need to invert permutation:
        # Build zero mask and scatter 'keep' to original frame order:
        keep_orig = torch.zeros_like(keep)
        keep_orig.scatter_(0, idx, keep)
        rej = ~keep_orig

        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Comet High-Clip Percentile":
        # median & MAD along frames, per pixel/channel
        med = _nanmedian(ts, dim=0)
        mad = _nanmedian((ts - med.unsqueeze(0)).abs(), dim=0) + 1e-6
        hi = med + (float(comet_hclip_k) * 1.4826 * mad)
        clipped = torch.minimum(ts, hi.unsqueeze(0))
        out = _nanquantile(clipped, float(comet_hclip_p) / 100.0, dim=0)
        rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Simple Average (No Rejection)":
        num = (ts * w).sum(dim=0)
        den = w.sum(dim=0).clamp_min(1e-20)
        out = (num / den)
        rej = torch.zeros((F, H, W, C), dtype=torch.bool, device=dev)
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
            med = _nanmedian(x, dim=0)
            std = _nanstd(x, dim=0)
            lo = med - float(kappa) * std
            hi = med + float(kappa) * std
            keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))
        w_eff = torch.where(keep, w, torch.zeros_like(w))
        den = w_eff.sum(dim=0).clamp_min(1e-20)
        out = (ts.mul(w_eff)).sum(dim=0).div(den)
        rej = ~keep
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Weighted Windsorized Sigma Clipping":
        # Iteratively clip outside median ± {low,high}*std (weights only used in final mean; same as CPU)
        low = float(sigma_low)
        high = float(sigma_high)
        keep = valid.clone()
        for _ in range(int(iterations)):
            x = ts.masked_fill(~keep, float("nan"))
            med = _nanmedian(x, dim=0)
            std = _nanstd(x, dim=0)
            lo = med - low * std
            hi = med + high * std
            keep = valid & (ts >= lo.unsqueeze(0)) & (ts <= hi.unsqueeze(0))

        # Weighted average on kept samples; if none, fall back to median of nonzero
        w_eff = torch.where(keep, w, torch.zeros_like(w))
        den = w_eff.sum(dim=0)
        num = (ts * w_eff).sum(dim=0)
        # fallback where den==0 → median(nonzero)
        out = torch.empty((H, W, C), dtype=ts.dtype, device=dev)
        mask_no = den <= 0
        if mask_no.any():
            # median of nonzero/finite
            x = ts.masked_fill(~valid, float("nan"))
            out_fallback = _nanmedian(x, dim=0)
            out[mask_no] = out_fallback[mask_no]
        if (~mask_no).any():
            out[~mask_no] = (num[~mask_no] / den[~mask_no])
        rej = ~keep
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Trimmed Mean":
        # Symmetric trim on both ends by fraction, using quantile window
        x = ts.masked_fill(~valid, float("nan"))
        qlo = _nanquantile(x, trim_fraction, dim=0)
        qhi = _nanquantile(x, 1.0 - trim_fraction, dim=0)
        keep = valid & (ts >= qlo.unsqueeze(0)) & (ts <= qhi.unsqueeze(0))
        w_eff = torch.where(keep, w, torch.zeros_like(w))
        den = w_eff.sum(dim=0).clamp_min(1e-20)
        out = (ts.mul(w_eff)).sum(dim=0).div(den)
        rej = ~keep
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Extreme Studentized Deviate (ESD)":
        x = ts.masked_fill(~valid, float("nan"))
        mean = torch.where(
            torch.isfinite(x),
            x,
            torch.zeros_like(x)
        ).nanmean(dim=0)
        std = _nanstd(x, dim=0)
        std = std.clamp_min(1e-12)
        z = (ts - mean.unsqueeze(0)).abs() / std.unsqueeze(0)
        keep = valid & (z < float(esd_threshold))
        w_eff = torch.where(keep, w, torch.zeros_like(w))
        den = w_eff.sum(dim=0).clamp_min(1e-20)
        out = (ts.mul(w_eff)).sum(dim=0).div(den)
        rej = ~keep
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Biweight Estimator":
        x = ts
        m = _nanmedian(x.masked_fill(~valid, float("nan")), dim=0)
        mad = _nanmedian((x - m.unsqueeze(0)).abs().masked_fill(~valid, float("nan")), dim=0)
        # fallback where MAD=0 → median
        mad = mad + 1e-12
        u = (x - m.unsqueeze(0)) / (float(biweight_constant) * mad.unsqueeze(0))
        mask = valid & (u.abs() < 1.0)
        w_eff = torch.where(mask, w, torch.zeros_like(w))
        one_minus_u2 = (1 - u**2).clamp_min(0)
        num = ((x - m.unsqueeze(0)) * (one_minus_u2**2) * w_eff).sum(dim=0)
        den = ((one_minus_u2**2) * w_eff).sum(dim=0)
        out = torch.where(
            den > 0,
            m + num / den,
            m
        )
        # rej where valid but mask false; also reject zeros
        rej = ~mask
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    if algo == "Modified Z-Score Clipping":
        x = ts
        med = _nanmedian(x.masked_fill(~valid, float("nan")), dim=0)
        mad = _nanmedian((x - med.unsqueeze(0)).abs().masked_fill(~valid, float("nan")), dim=0)
        mad = mad + 1e-12
        mz = 0.6745 * (x - med.unsqueeze(0)) / mad.unsqueeze(0)
        keep = valid & (mz.abs() < float(modz_threshold))
        w_eff = torch.where(keep, w, torch.zeros_like(w))
        den = w_eff.sum(dim=0).clamp_min(1e-20)
        out = (ts.mul(w_eff)).sum(dim=0).div(den)
        rej = ~keep
        return out.contiguous().cpu().numpy(), rej.cpu().numpy()

    raise NotImplementedError(f"GPU path not implemented for: {algo_name}")
