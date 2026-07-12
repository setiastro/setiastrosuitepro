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
# Hybrid Neural-Morphological Star Reduction (levels 7-10)
#
# For star_level > 6, the model is run ONCE at the safe level=5, then a
# circular-structuring-element erosion (escalating with level) is blended
# into regions the level-5 prediction already flagged as stellar via
# |pred_neural - input|. Ported from SyQon's updated Siril script.
# ---------------------------------------------------------------------------

def _get_circular_kernel(size: int, device, torch):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    dist = x ** 2 + y ** 2
    return (dist <= 1.0).float().to(device)


def _erode_circular(img, size: int, torch, F):
    """Grayscale erosion with a circular structuring element. img: (B,C,H,W)."""
    kernel = _get_circular_kernel(size, img.device, torch)
    K = size
    padding = K // 2
    B, C, H, W = img.shape

    x = img.reshape(B * C, 1, H, W)
    patches = F.unfold(x, kernel_size=K, padding=padding)

    k_flat = kernel.reshape(-1, 1)
    large_val = torch.tensor(999.0, device=patches.device, dtype=patches.dtype)
    masked_patches = torch.where(k_flat == 1.0, patches, large_val)

    min_vals, _ = torch.min(masked_patches, dim=1)
    return min_vals.reshape(B, C, H, W)


def _hybrid_star_reduction(model, t, level: int, torch, F):
    """t: (1,3,H,W) float32 in [0,1] (post brightness normalization).
    Runs model once at safe level=5, then blends in morphological erosion."""
    safe_lv = torch.tensor([5.0 / 10.0], dtype=torch.float32, device=t.device)
    pred_neural = model(t, safe_lv)

    diff = torch.abs(pred_neural - t)
    diff_max, _ = torch.max(diff, dim=1, keepdim=True)

    if level == 7:
        mask_raw = torch.clamp((diff_max - 0.008) * 30.0, 0.0, 1.0)
        mask_dilated = F.max_pool2d(mask_raw, kernel_size=3, stride=1, padding=1)
        mask_smooth = F.avg_pool2d(mask_dilated, kernel_size=3, stride=1, padding=1)
        eroded = _erode_circular(t, size=3, torch=torch, F=F)
    elif level == 8:
        mask_raw = torch.clamp((diff_max - 0.006) * 50.0, 0.0, 1.0)
        mask_dilated = F.max_pool2d(mask_raw, kernel_size=3, stride=1, padding=1)
        mask_smooth = F.avg_pool2d(mask_dilated, kernel_size=3, stride=1, padding=1)
        eroded = _erode_circular(t, size=5, torch=torch, F=F)
    elif level == 9:
        mask_raw = torch.clamp((diff_max - 0.005) * 100.0, 0.0, 1.0)
        mask_dilated = F.max_pool2d(mask_raw, kernel_size=5, stride=1, padding=2)
        mask_smooth = F.avg_pool2d(mask_dilated, kernel_size=5, stride=1, padding=2)
        eroded = _erode_circular(_erode_circular(t, size=5, torch=torch, F=F), size=5, torch=torch, F=F)
    else:  # level == 10
        mask_raw = torch.clamp((diff_max - 0.003) * 200.0, 0.0, 1.0)
        mask_dilated = F.max_pool2d(mask_raw, kernel_size=5, stride=1, padding=2)
        mask_smooth = F.avg_pool2d(mask_dilated, kernel_size=5, stride=1, padding=2)
        eroded = t
        for _ in range(5):
            eroded = _erode_circular(eroded, size=5, torch=torch, F=F)

    return t + mask_smooth * (eroded - t)
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
    normalize_brightness: bool = False,
    target_median: float = 0.10,
    progress_cb: Optional[ProgressCB] = None,
    label: str = "",
) -> np.ndarray:
    img_padded, orig_hw = _pad_reflect(img, pad)
    H, W = img_padded.shape[:2]

    ys = _syqon_grid_positions(H, tile, overlap)
    xs = _syqon_grid_positions(W, tile, overlap)
    total = len(ys) * len(xs)
    w2    = _cosine_weights(tile, overlap)

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc   = np.zeros((H, W, 1), dtype=np.float32)
    done    = 0

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            patch  = img_padded[y0:y1, x0:x1, :]
            ph, pw = patch.shape[:2]

            patch_in = _pad_tile_np(patch, tile) if (ph != tile or pw != tile) else patch

            if normalize_brightness:
                # Per-tile median shift — keeps model input in trained brightness
                # range regardless of source image brightness.
                # Floor prevents explosion on near-black background tiles.
                tile_median = float(np.median(patch_in))
                tile_median = max(tile_median, 1e-3)
                scale       = min(target_median / tile_median, target_median / 1e-3)
                patch_norm  = np.clip(patch_in * scale, 0.0, 1.0).astype(np.float32)
                pred_norm   = np.clip(infer_fn(patch_norm), 0.0, 1.0)
                pred        = np.clip(pred_norm / scale, 0.0, 1.0).astype(np.float32)
            else:
                pred = np.clip(infer_fn(patch_in), 0.0, 1.0)

            wlocal = w2[:ph, :pw, :]
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
    mode="classic",
) -> Tuple[np.ndarray, dict]:
    """
    Aberration correction — boolean pass, no level conditioning.
    Returns (corrected_rgb01, info_dict).
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)

    mode = str(mode or "classic").strip().lower()
    model, config = load_parallax_model(ckpt_path, variant="correction", mode=mode)
    model.to(device).eval()

    def _infer(patch_hwc: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(patch_hwc.transpose(2, 0, 1)[None], dtype=np.float32)).to(device)
        with torch.no_grad():
            out = model(t)
        return out[0].clamp(0.0, 1.0).float().cpu().numpy().transpose(1, 2, 0)

    result = _run_tiled_cosine(
        _infer, _ensure_rgb(img_rgb01),
        tile=tile, overlap=overlap, pad=pad,
        normalize_brightness=(mode != "aesthetics"), target_median=0.10,
        progress_cb=progress_cb, label=f"[Correction/{mode}]",
    )

    info = {
        "variant": "correction",
        "mode":    mode,
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
    mode: str = "classic",
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Star reduction at level 1-6, normalised to level/10.
    Returns (reduced_rgb01, info_dict).
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    mode   = str(mode or "classic").strip().lower()
    level  = int(np.clip(level, 1, 10))
    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)
    F      = torch.nn.functional

    model, config = load_parallax_model(ckpt_path, variant="star_reduce", mode=mode)
    model.to(device).eval()

    if mode == "aesthetics":
        # NAFNet 4-channel: RGB + constant level plane.
        # Model trained on level 1-7; clamp caller's 1-10 range into that.
        aesth_level = int(np.clip(level, 1, 7))
        lvl_norm    = (float(aesth_level) - 1.0) / 6.0

        def _infer(patch_hwc: np.ndarray) -> np.ndarray:
            t = torch.from_numpy(np.ascontiguousarray(patch_hwc.transpose(2, 0, 1)[None], dtype=np.float32)).to(device)
            b, _, h, w = t.shape
            lvl_ch = torch.full((b, 1, h, w), lvl_norm, dtype=t.dtype, device=device)
            t4 = torch.cat([t, lvl_ch], dim=1)
            with torch.no_grad():
                out = model(t4)
            return out[0].clamp(0.0, 1.0).float().cpu().numpy().transpose(1, 2, 0)
    else:
        hybrid = level > 6
        if hybrid:
            def _infer(patch_hwc: np.ndarray) -> np.ndarray:
                t = torch.from_numpy(np.ascontiguousarray(patch_hwc.transpose(2, 0, 1)[None], dtype=np.float32)).to(device)
                with torch.no_grad():
                    out = _hybrid_star_reduction(model, t, level, torch, F)
                return out[0].clamp(0.0, 1.0).float().cpu().numpy().transpose(1, 2, 0)
        else:
            lv_t = torch.tensor([float(level) / 10.0], dtype=torch.float32, device=device)

            def _infer(patch_hwc: np.ndarray) -> np.ndarray:
                t = torch.from_numpy(np.ascontiguousarray(patch_hwc.transpose(2, 0, 1)[None], dtype=np.float32)).to(device)
                with torch.no_grad():
                    out = model(t, lv_t)
                return out[0].clamp(0.0, 1.0).float().cpu().numpy().transpose(1, 2, 0)

    result = _run_tiled_cosine(
        _infer, _ensure_rgb(img_rgb01),
        tile=tile, overlap=overlap, pad=pad,
        normalize_brightness=(mode != "aesthetics"), target_median=0.10,
        progress_cb=progress_cb, label=f"[StarReduce/{mode} L{level}]",
    )

    info = {
        "variant": "star_reduce",
        "mode":    mode,
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
    pad: int = 96,
    use_gpu: bool = True,
    prefer_dml: bool = True,
    mode: str = "classic",
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Sharpening with blend alpha 0.0 (passthrough) → 1.0 (full sharpen).
    Uses SyQon's exact tent-window tiling from infer_astro_sharp_lite.py.
    Returns (sharpened_rgb01, info_dict).
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    alpha  = float(np.clip(alpha, 0.0, 2.0))
    img    = _ensure_rgb(img_rgb01)

    if alpha == 0.0:
        return img.astype(np.float32, copy=False), {"variant": "sharpen", "alpha": 0.0, "skipped": True}

    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)

    mode = str(mode or "classic").strip().lower()
    model, config = load_parallax_model(ckpt_path, variant="sharpen", mode=mode)
    model.to(device).eval()

    # AstroNAFLiteDeblur is the only Parallax model using grouped/depthwise
    # convolutions (NAFBlock.conv2, groups=dw_c). On some Turing-generation
    # GPUs (observed: GTX 1650), cuDNN's autotuned algorithm for grouped
    # convs at certain tile sizes (256/512) silently returns all-zero
    # output with NO error — completes normally but produces a black tile.
    # Disabling cudnn.benchmark forces a stable algorithm selection.
    _cudnn_benchmark_prev = None
    if device.type == "cuda":
        _cudnn_benchmark_prev = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False

    # Reflect-pad full image before tiling — same as correction/star_reduce
    img_padded, orig_hw = _pad_reflect(img, pad)
    height, width = img_padded.shape[:2]

    xs    = _syqon_grid_positions(width,  tile, overlap)
    ys    = _syqon_grid_positions(height, tile, overlap)
    total = len(xs) * len(ys)
    coords = [(top, left) for top in ys for left in xs]

    # Accumulation tensors on CPU — matches SyQon exactly
    output_tensor = torch.zeros((1, 3, height, width), dtype=torch.float32)
    weight_sum    = torch.zeros((1, 1, height, width), dtype=torch.float32)
    base_window   = _tent_window_torch(tile, torch)

    # Only CUDA benefits from autocast here (fp16). MPS autocast is unsupported
    # in some torch builds and even when supported, we run fp32 on it anyway —
    # so skip it entirely. CPU (Intel Mac) is fp32, no autocast.
    use_autocast   = (device.type == "cuda")
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32

    for idx, (top, left) in enumerate(coords, start=1):
        bottom = min(top + tile, height)
        right  = min(left + tile, width)
        tile_h = bottom - top
        tile_w = right - left

        tile_data        = img_padded[top:bottom, left:right, :]
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

    # Reconstruct, unpad, then alpha blend against original unpadded image
    reconstructed = output_tensor / torch.clamp(weight_sum, min=1e-6)
    reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
    reconstructed = _unpad(reconstructed, orig_hw, pad)

    final_output = img + alpha * (reconstructed - img)
    result = np.clip(final_output, 0.0, 1.0).astype(np.float32, copy=False)

    if _cudnn_benchmark_prev is not None:
        torch.backends.cudnn.benchmark = _cudnn_benchmark_prev
        
    info = {
        "variant": "sharpen",
        "alpha":   alpha,
        "mode":    mode,
        "device":  str(device),
        "torch_version": getattr(torch, "__version__", None),
        "tile": tile, "overlap": overlap, "pad": pad,
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