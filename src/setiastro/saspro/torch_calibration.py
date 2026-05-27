# setiastro/saspro/torch_calibration.py
from __future__ import annotations
import numpy as np
from typing import Optional

# Lazy torch — only imported when first needed
_torch = None
_device = None

def _get_torch_and_device():
    global _torch, _device
    if _torch is not None:
        return _torch, _device
    try:
        from setiastro.saspro.runtime_torch import import_torch
        torch = import_torch(
            prefer_cuda=True,
            prefer_xpu=False,
            prefer_dml=True,
            status_cb=lambda s: None,
        )
        # Pick best available device
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            device = torch.device("cuda")
        elif (hasattr(torch, "backends")
              and hasattr(torch.backends, "mps")
              and torch.backends.mps.is_available()):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        _torch = torch
        _device = device
        return _torch, _device
    except Exception:
        return None, None


def gpu_available() -> bool:
    torch, device = _get_torch_and_device()
    if torch is None or device is None:
        return False
    return str(device.type) in ("cuda", "mps")


def _to_tensor(arr: np.ndarray, torch, device):
    """Convert numpy array to float32 torch tensor on device."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return torch.from_numpy(a).to(device, dtype=torch.float32)


def _to_numpy(t) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def subtract_dark_gpu(
    light: np.ndarray,
    dark: np.ndarray,
    pedestal: float = 0.0,
) -> Optional[np.ndarray]:
    """
    GPU dark subtraction: light - dark + pedestal.
    Returns None if GPU unavailable (caller falls back to Numba).
    light/dark: any of (H,W), (C,H,W), (H,W,C) — matched shapes expected.
    """
    torch, device = _get_torch_and_device()
    if torch is None or str(device.type) == "cpu":
        return None

    try:
        t_light = _to_tensor(light, torch, device)
        t_dark  = _to_tensor(dark,  torch, device)

        # broadcast handles (C,H,W) light with (H,W) dark etc.
        result = t_light - t_dark + float(pedestal)
        result = torch.clamp(result, 0.0, float("inf"))

        out = _to_numpy(result)
        return out.reshape(light.shape)
    except Exception:
        return None


def divide_flat_gpu(
    light: np.ndarray,
    flat: np.ndarray,
) -> Optional[np.ndarray]:
    """
    GPU flat division. flat should already be normalized (median=1.0 per channel).
    Returns None if GPU unavailable.
    """
    torch, device = _get_torch_and_device()
    if torch is None or str(device.type) == "cpu":
        return None

    try:
        t_light = _to_tensor(light, torch, device)
        t_flat  = _to_tensor(flat,  torch, device)

        # Guard against zeros/nan in flat
        t_flat = torch.where(
            torch.isfinite(t_flat) & (t_flat > 0),
            t_flat,
            torch.ones_like(t_flat),
        )

        result = t_light / t_flat
        out = _to_numpy(result)
        return out.reshape(light.shape)
    except Exception:
        return None


def cosmetic_correction_gpu(
    image: np.ndarray,
    hot_sigma: float = 5.0,
    cold_sigma: float = 5.0,
    sat_quantile: float = 0.9995,
) -> Optional[np.ndarray]:
    """
    GPU cosmetic correction using unfold for 3×3 neighborhood stats.
    Computes ring median via sorting the 8 neighbors, then MAD sigma.
    Star guard and cold cluster guard are NOT applied here (they require
    conditional branching that's expensive on GPU) — use this for clean
    sensors where hot/cold pixels are isolated.
    Returns None if GPU unavailable or image is not 2D mono.
    """
    torch, device = _get_torch_and_device()
    if torch is None or str(device.type) == "cpu":
        return None

    # Only handle 2D for now — color is per-channel via Numba anyway
    if image.ndim not in (2, 3):
        return None

    try:
        was_2d = (image.ndim == 2)
        if was_2d:
            arr = image[None, None]  # (1,1,H,W)
        else:
            # CHW
            if image.shape[0] == 3:
                arr = image[None]    # (1,3,H,W)
            else:
                return None          # HWC — don't handle here

        t = _to_tensor(arr, torch, device)  # (1,C,H,W)
        _, C, H, W = t.shape

        # unfold: extract 3×3 patches → (1, C, H, W, 9)
        padded = torch.nn.functional.pad(t, (1,1,1,1), mode="reflect")
        patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)
        # patches shape: (1, C, H, W, 3, 3) → flatten last two → (1,C,H,W,9)
        patches = patches.reshape(1, C, H, W, 9)

        # center pixel index is 4 in the 3×3 (row-major: (1,1)=4)
        center = patches[..., 4:5]          # (1,C,H,W,1)

        # ring = all 8 neighbors (exclude center index 4)
        ring = torch.cat([patches[..., :4], patches[..., 5:]], dim=-1)  # (1,C,H,W,8)

        # ring median via sort
        ring_sorted, _ = torch.sort(ring, dim=-1)
        ring_median = (ring_sorted[..., 3] + ring_sorted[..., 4]) / 2.0  # median of 8

        # MAD
        mad = torch.median(torch.abs(ring - ring_median.unsqueeze(-1)), dim=-1).values
        sigma = 1.4826 * mad + 1e-8

        # saturation threshold per channel
        result = t.clone()
        for c in range(C):
            plane = t[0, c]
            sat_val = float(torch.quantile(plane.float(), sat_quantile).item())

            hot_mask  = (center[0, c, :, :, 0] > ring_median[0, c] + hot_sigma  * sigma[0, c])
            cold_mask = (center[0, c, :, :, 0] < ring_median[0, c] - cold_sigma * sigma[0, c])
            sat_mask  = (plane >= sat_val)

            # don't touch saturated pixels
            hot_mask  = hot_mask  & ~sat_mask
            cold_mask = cold_mask & ~sat_mask

            result[0, c] = torch.where(
                hot_mask | cold_mask,
                ring_median[0, c],
                plane,
            )

        out = _to_numpy(result[0])  # (C,H,W) or (1,H,W)
        if was_2d:
            return out[0]   # back to (H,W)
        return out          # (C,H,W)

    except Exception:
        return None