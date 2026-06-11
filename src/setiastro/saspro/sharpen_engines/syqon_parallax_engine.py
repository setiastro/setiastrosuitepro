from __future__ import annotations

# src/setiastro/saspro/sharpen_engines/syqon_parallax_engine.py
#
# SyQon Parallax inference engine for SASpro.
# Three models:
#   correction  — StellarDirectNet aberration correction (boolean)
#   star_reduce — StellarDirectNet star reduction (level 1-6)
#   sharpen     — AstroNAFLiteDeblur sharpening (alpha 0-1)
#
# Tiling:
#   correction / star_reduce — cosine-blend tiling (Axiom pattern)
#   sharpen                  — SyQon tent-window tiling (infer_astro_sharp_lite pattern)

import os
from typing import Callable, Optional, Tuple

import numpy as np

ProgressCB = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _infer_device(torch, *, prefer_cuda: bool = True, prefer_dml: bool = True):
    if prefer_cuda and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_dml and os.name == "nt":
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _pad_reflect(img: np.ndarray, pad: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = img.shape[:2]
    if pad <= 0:
        return img, (h, w)
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    return padded.astype(np.float32, copy=False), (h, w)


def _unpad(img: np.ndarray, orig_hw: tuple[int, int], pad: int) -> np.ndarray:
    h, w = orig_hw
    if pad <= 0:
        return img[:h, :w, :]
    return img[pad:pad + h, pad:pad + w, :]


def _cosine_weights(tile: int, overlap: int) -> np.ndarray:
    """Cosine-blend 2-D weight map (H, W, 1)."""
    if overlap <= 0:
        return np.ones((tile, tile, 1), dtype=np.float32)
    coords = np.arange(tile, dtype=np.float32)
    dist   = np.minimum(coords, coords[::-1])
    ramp   = np.clip(dist / float(overlap), 0.0, 1.0)
    w1d    = np.clip(0.5 - 0.5 * np.cos(np.pi * ramp), 1e-3, 1.0).astype(np.float32)
    return (w1d[:, None] * w1d[None, :])[..., None].astype(np.float32)


def _tent_window_torch(tile: int, torch):
    """Exact SyQon tent window as a torch tensor (1,1,H,W)."""
    coords = torch.linspace(-1.0, 1.0, steps=tile)
    ramp   = torch.clamp(1.0 - torch.abs(coords), min=1e-3)
    window = torch.outer(ramp, ramp)
    window = window / window.max()
    return window.unsqueeze(0).unsqueeze(0)


def _syqon_grid_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    """Exact SyQon compute_grid_positions — guarantees full coverage."""
    if tile_size <= overlap:
        raise ValueError("tile_size must be > overlap")
    if length <= tile_size:
        return [0]
    stride    = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def _pad_tile_np(tile: np.ndarray, tile_size: int) -> np.ndarray:
    """Exact SyQon pad_tile_np — reflect/edge pad undersized tiles."""
    h, w, c = tile.shape
    if h == tile_size and w == tile_size:
        return tile
    pad_h    = tile_size - h
    pad_w    = tile_size - w
    pad_mode = "reflect" if h > 1 and w > 1 else "edge"
    return np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)


# ---------------------------------------------------------------------------
# Generic cosine-blend tiled runner (correction + star_reduce)
# ---------------------------------------------------------------------------

def _run_tiled_cosine(
    infer_fn,
    img: np.ndarray,
    *,
    tile: int,
    overlap: int,
    pad: int,
    progress_cb: Optional[ProgressCB] = None,
    label: str = "",
) -> np.ndarray:
    img_padded, orig_hw = _pad_reflect(img, pad)
    H, W   = img_padded.shape[:2]
    stride = max(tile - overlap, 1)
    ys     = list(range(0, H, stride))
    xs     = list(range(0, W, stride))
    total  = len(ys) * len(xs)
    w2     = _cosine_weights(tile, overlap)

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc   = np.zeros((H, W, 1), dtype=np.float32)
    buf     = np.zeros((tile, tile, 3), dtype=np.float32)
    done    = 0

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            patch  = img_padded[y0:y1, x0:x1, :]
            ph, pw = patch.shape[:2]

            if ph != tile or pw != tile:
                buf.fill(0.0)
                buf[:ph, :pw, :] = patch
                patch_in = buf
            else:
                patch_in = patch

            pred    = np.clip(infer_fn(patch_in), 0.0, 1.0)
            wlocal  = w2[:ph, :pw, :]
            out_acc[y0:y1, x0:x1, :] += pred[:ph, :pw, :] * wlocal
            w_acc  [y0:y1, x0:x1, :] += wlocal

            done += 1
            if callable(progress_cb):
                progress_cb(done, total, f"{label} tile {done}/{total}…")

    np.maximum(w_acc, 1e-8, out=w_acc)
    result = np.clip(out_acc / w_acc, 0.0, 1.0)
    return _unpad(result, orig_hw, pad)


# ---------------------------------------------------------------------------
# Correction inference
# ---------------------------------------------------------------------------

def parallax_correction_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    *,
    tile: int = 512,
    overlap: int = 64,
    pad: int = 96,
    use_gpu: bool = True,
    prefer_dml: bool = True,
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Aberration correction — boolean pass, no level conditioning.
    Returns (corrected_rgb01, info_dict).
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)

    model, config = load_parallax_model(ckpt_path, variant="correction")
    model.to(device).eval()

    def _infer(patch_hwc: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(patch_hwc.transpose(2, 0, 1)[None], dtype=np.float32)).to(device)
        with torch.no_grad():
            out = model(t)
        return out[0].clamp(0.0, 1.0).float().cpu().numpy().transpose(1, 2, 0)

    result = _run_tiled_cosine(
        _infer, _ensure_rgb(img_rgb01),
        tile=tile, overlap=overlap, pad=pad,
        progress_cb=progress_cb, label="[Correction]",
    )

    info = {
        "variant": "correction",
        "device":  str(device),
        "torch_version": getattr(torch, "__version__", None),
        "tile": tile, "overlap": overlap, "pad": pad,
    }
    return result.astype(np.float32, copy=False), info


# ---------------------------------------------------------------------------
# Star reduction inference
# ---------------------------------------------------------------------------

def parallax_star_reduce_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    level: int,
    *,
    tile: int = 512,
    overlap: int = 64,
    pad: int = 96,
    use_gpu: bool = True,
    prefer_dml: bool = True,
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Star reduction at level 1-6, normalised to level/10.
    Returns (reduced_rgb01, info_dict).
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    level  = int(np.clip(level, 1, 6))
    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)

    model, config = load_parallax_model(ckpt_path, variant="star_reduce")
    model.to(device).eval()

    lv_t = torch.tensor([float(level) / 10.0], dtype=torch.float32, device=device)

    def _infer(patch_hwc: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(patch_hwc.transpose(2, 0, 1)[None], dtype=np.float32)).to(device)
        with torch.no_grad():
            out = model(t, lv_t)
        return out[0].clamp(0.0, 1.0).float().cpu().numpy().transpose(1, 2, 0)

    result = _run_tiled_cosine(
        _infer, _ensure_rgb(img_rgb01),
        tile=tile, overlap=overlap, pad=pad,
        progress_cb=progress_cb, label=f"[StarReduce L{level}]",
    )

    info = {
        "variant": "star_reduce",
        "level":   level,
        "device":  str(device),
        "torch_version": getattr(torch, "__version__", None),
        "tile": tile, "overlap": overlap, "pad": pad,
    }
    return result.astype(np.float32, copy=False), info


# ---------------------------------------------------------------------------
# Sharpen inference  — exact SyQon tent-window pipeline
# ---------------------------------------------------------------------------

def parallax_sharpen_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    alpha: float,
    *,
    tile: int = 512,
    overlap: int = 64,
    use_gpu: bool = True,
    prefer_dml: bool = True,
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Sharpening with blend alpha 0.0 (passthrough) → 1.0 (full sharpen).
    Uses SyQon's exact tent-window tiling from infer_astro_sharp_lite.py.
    Returns (sharpened_rgb01, info_dict).
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    alpha  = float(np.clip(alpha, 0.0, 1.0))
    img    = _ensure_rgb(img_rgb01)

    if alpha == 0.0:
        return img.astype(np.float32, copy=False), {"variant": "sharpen", "alpha": 0.0, "skipped": True}

    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)

    model, config = load_parallax_model(ckpt_path, variant="sharpen")
    model.to(device).eval()

    height, width = img.shape[:2]
    xs    = _syqon_grid_positions(width,  tile, overlap)
    ys    = _syqon_grid_positions(height, tile, overlap)
    total = len(xs) * len(ys)
    coords = [(top, left) for top in ys for left in xs]

    # Accumulation tensors on CPU — matches SyQon exactly
    output_tensor = torch.zeros((1, 3, height, width), dtype=torch.float32)
    weight_sum    = torch.zeros((1, 1, height, width), dtype=torch.float32)
    base_window   = _tent_window_torch(tile, torch)

    use_autocast   = device.type in ("cuda", "mps")
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, (top, left) in enumerate(coords, start=1):
        bottom = min(top + tile, height)
        right  = min(left + tile, width)
        tile_h = bottom - top
        tile_w = right - left

        tile_data        = img[top:bottom, left:right, :]
        tile_data_padded = _pad_tile_np(tile_data, tile)

        tile_tensor = (
            torch.from_numpy(tile_data_padded)
            .permute(2, 0, 1).unsqueeze(0).float().to(device)
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                pred_tile = model(tile_tensor)

        pred_tile = pred_tile.float().cpu()

        pred_patch   = pred_tile[0:1][:, :, :tile_h, :tile_w]
        window_patch = base_window[:, :, :tile_h, :tile_w]

        output_tensor[:, :, top:bottom, left:right] += pred_patch * window_patch
        weight_sum   [:, :, top:bottom, left:right] += window_patch

        if callable(progress_cb):
            progress_cb(idx, total, f"[Sharpen] tile {idx}/{total}…")

    # Reconstruct + alpha blend — exact SyQon formula
    reconstructed = output_tensor / torch.clamp(weight_sum, min=1e-6)
    reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).numpy()

    final_output = img + alpha * (reconstructed - img)
    result = np.clip(final_output, 0.0, 1.0).astype(np.float32, copy=False)

    info = {
        "variant": "sharpen",
        "alpha":   alpha,
        "device":  str(device),
        "torch_version": getattr(torch, "__version__", None),
        "tile": tile, "overlap": overlap,
    }
    return result, info


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    else:
        x = x[..., :3]
    return np.clip(x, 0.0, 1.0)


def clear_parallax_models_cache(*, aggressive: bool = False, status_cb=print) -> None:
    status_cb("[SyQon Parallax] No persistent session cache to clear.")
    if aggressive:
        try:
            import gc; gc.collect()
            status_cb("[SyQon Parallax] gc.collect() called")
        except Exception:
            pass
        try:
            from setiastro.saspro.runtime_torch import import_torch
            torch = import_torch(prefer_cuda=True, prefer_xpu=False, prefer_dml=True, status_cb=lambda *_: None)
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                status_cb("[SyQon Parallax] torch.cuda.empty_cache() called")
        except Exception:
            pass