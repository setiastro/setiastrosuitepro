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

from setiastro.saspro.runtime_torch import np_to_torch, torch_to_np

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
    # MPS only on Apple Silicon. is_available()/is_built() both report True on
    # Intel Macs with a Metal-capable Radeon, but the backend faults on first use
    # there and torch surfaces it as the misleading "Numpy is not available".
    # Use the shared gate so this picker can't drift from best_device/current_backend.
    from setiastro.saspro.runtime_torch import mps_is_usable
    if mps_is_usable(torch):
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


def _resolve_batch_size(cfg, device, tile: int) -> int:
    """
    Resolve a UI/preset ``batch_size`` value into an integer batch count.

    cfg  : "Auto", "1", "2", "4", "8", or an int
    device: torch.device the model is running on

    CPU: always 1 (batching mostly hurts here).
    DirectML: capped at 2 even if user requests more — batching is fragile
              on DML in current torch-directml builds and OOMs silently.
    Native GPU (CUDA/MPS/XPU): "Auto" picks from a tile-area × VRAM heuristic;
              explicit ints pass through.
    """
    if device is None or getattr(device, "type", "cpu") == "cpu":
        return 1

    dev_str = str(device).lower()
    is_dml  = "privateuseone" in dev_str or "dml" in dev_str

    val = str(cfg).strip()
    if val.lower() == "auto":
        if is_dml:
            return 1
        # Tile-area heuristic — smaller tiles allow bigger batches
        if tile <= 384:
            base = 4
        elif tile <= 640:
            base = 2
        else:
            base = 1
        # Scale up on high-VRAM CUDA cards
        try:
            import torch as _t
            if getattr(device, "type", None) == "cuda":
                gb = _t.cuda.get_device_properties(device).total_memory / (1024.0 ** 3)
                if gb >= 12.0:
                    base = max(base, 4)
                elif gb < 6.0:
                    base = 1
        except Exception:
            pass
        return max(1, base)

    try:
        b = max(1, int(val))
    except Exception:
        b = 1
    if is_dml and b > 2:
        b = 2
    return b

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
    """t: (B,3,H,W) float32 in [0,1] (post brightness normalization).
    Runs model once at safe level=5, then blends in morphological erosion.
    Batch-safe: safe_lv is broadcast to match t's batch dim."""
    b = t.shape[0]
    safe_lv = torch.full((b,), 5.0 / 10.0, dtype=torch.float32, device=t.device)
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
    infer_batch_fn,
    img: np.ndarray,
    *,
    tile: int,
    overlap: int,
    pad: int,
    normalize_brightness: bool = False,
    target_median: float = 0.10,
    progress_cb: Optional[ProgressCB] = None,
    label: str = "",
    batch_size: int = 1,
) -> np.ndarray:
    """
    Batched cosine-blend tiled runner.
    infer_batch_fn: callable taking list[(H,W,3) float32], returning
                    list[(H,W,3) float32] of the same length.
    Streams patches into flush-when-full batches so memory usage stays flat.
    """
    img_padded, orig_hw = _pad_reflect(img, pad)
    H, W = img_padded.shape[:2]

    ys = _syqon_grid_positions(H, tile, overlap)
    xs = _syqon_grid_positions(W, tile, overlap)
    total = len(ys) * len(xs)
    w2    = _cosine_weights(tile, overlap)

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc   = np.zeros((H, W, 1), dtype=np.float32)
    done    = 0

    bs = max(1, int(batch_size))
    batch_meta: list = []      # list of (y0, y1, x0, x1, ph, pw, scale|None)
    batch_patches: list = []

    def _flush():
        nonlocal done
        if not batch_patches:
            return
        preds = infer_batch_fn(batch_patches)
        for meta, pred in zip(batch_meta, preds):
            y0, y1, x0, x1, ph, pw, scale = meta
            pred = np.clip(pred, 0.0, 1.0)
            if scale is not None:
                pred = np.clip(pred / scale, 0.0, 1.0).astype(np.float32)
            wlocal = w2[:ph, :pw, :]
            out_acc[y0:y1, x0:x1, :] += pred[:ph, :pw, :] * wlocal
            w_acc  [y0:y1, x0:x1, :] += wlocal
            done += 1
            if callable(progress_cb):
                progress_cb(done, total, f"{label} tile {done}/{total}…")
        batch_meta.clear()
        batch_patches.clear()

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            patch  = img_padded[y0:y1, x0:x1, :]
            ph, pw = patch.shape[:2]

            patch_in = _pad_tile_np(patch, tile) if (ph != tile or pw != tile) else patch

            scale = None
            if normalize_brightness:
                # Per-tile median shift — keeps model input in trained brightness
                # range regardless of source image brightness.
                # Floor prevents explosion on near-black background tiles.
                tile_median = float(np.median(patch_in))
                tile_median = max(tile_median, 1e-3)
                scale       = min(target_median / tile_median, target_median / 1e-3)
                patch_in    = np.clip(patch_in * scale, 0.0, 1.0).astype(np.float32)

            batch_meta.append((y0, y1, x0, x1, ph, pw, scale))
            batch_patches.append(patch_in)
            if len(batch_patches) >= bs:
                _flush()

    _flush()

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
    batch_size="Auto",
) -> Tuple[np.ndarray, dict]:
    """
    Aberration correction — boolean pass, no level conditioning.
    Returns (corrected_rgb01, info_dict).

    batch_size: "Auto" (default), or an integer 1/2/4/8. See
                _resolve_batch_size for the auto heuristic.
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)

    mode = str(mode or "classic").strip().lower()
    model, config = load_parallax_model(ckpt_path, variant="correction", mode=mode)
    model.to(device).eval()

    bs = _resolve_batch_size(batch_size, device, tile)

    def _infer_batch(patches: list) -> list:
        stacked = np.stack([np.ascontiguousarray(p.transpose(2, 0, 1)) for p in patches])
        t = np_to_torch(stacked, device=device, dtype=torch.float32, torch=torch)
        with torch.no_grad():
            out = model(t)
        out = torch_to_np(out.clamp(0.0, 1.0).float())
        return [out[i].transpose(1, 2, 0) for i in range(out.shape[0])]

    result = _run_tiled_cosine(
        _infer_batch, _ensure_rgb(img_rgb01),
        tile=tile, overlap=overlap, pad=pad,
        normalize_brightness=(mode != "aesthetics"), target_median=0.10,
        progress_cb=progress_cb, label=f"[Correction/{mode}]",
        batch_size=bs,
    )

    info = {
        "variant": "correction",
        "mode":    mode,
        "device":  str(device),
        "batch_size_cfg":    str(batch_size),
        "batch_size_active": int(bs),
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
    batch_size="Auto",
) -> Tuple[np.ndarray, dict]:
    """
    Star reduction at level 1-6 (classic) or 1-7 (aesthetics).
    Returns (reduced_rgb01, info_dict).

    batch_size: "Auto" (default) or int 1/2/4/8.
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import load_parallax_model

    mode   = str(mode or "classic").strip().lower()
    level  = int(np.clip(level, 1, 10))
    torch  = import_torch(prefer_cuda=use_gpu, prefer_xpu=False, prefer_dml=prefer_dml, status_cb=lambda *_: None)

    import numpy as _np
    print("[np-check] numpy:", _np.__version__)
    try:
        torch.from_numpy(_np.zeros(1, dtype=_np.float32)); print("[np-check] from_numpy: OK")
    except Exception as e:
        print("[np-check] from_numpy FAILED:", e)
    print("[np-check] device:", _infer_device(torch))

    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)
    F      = torch.nn.functional

    model, config = load_parallax_model(ckpt_path, variant="star_reduce", mode=mode)
    model.to(device).eval()

    bs = _resolve_batch_size(batch_size, device, tile)

    if mode == "aesthetics":
        # NAFNet 4-channel: RGB + constant level plane.
        # Aesthetics star_reduce model was trained on level 1-7; clamp
        # caller's 1-10 range into that.
        aesth_level = int(np.clip(level, 1, 7))
        lvl_norm    = (float(aesth_level) - 1.0) / 6.0

        def _infer_batch(patches: list) -> list:
            stacked = np.stack([np.ascontiguousarray(p.transpose(2, 0, 1)) for p in patches])
            t = np_to_torch(stacked, device=device, dtype=torch.float32, torch=torch)
            b, _, h, w = t.shape
            lvl_ch = torch.full((b, 1, h, w), lvl_norm, dtype=t.dtype, device=device)
            t4 = torch.cat([t, lvl_ch], dim=1)
            with torch.no_grad():
                out = model(t4)
            out = torch_to_np(out.clamp(0.0, 1.0).float())
            return [out[i].transpose(1, 2, 0) for i in range(out.shape[0])]
    else:
        hybrid = level > 6
        if hybrid:
            def _infer_batch(patches: list) -> list:
                stacked = np.stack([np.ascontiguousarray(p.transpose(2, 0, 1)) for p in patches])
                t = np_to_torch(stacked, device=device, dtype=torch.float32, torch=torch)
                with torch.no_grad():
                    out = _hybrid_star_reduction(model, t, level, torch, F)
                out = torch_to_np(out.clamp(0.0, 1.0).float())
                return [out[i].transpose(1, 2, 0) for i in range(out.shape[0])]
        else:
            lv_scalar = float(level) / 10.0

            def _infer_batch(patches: list) -> list:
                N = len(patches)
                stacked = np.stack([np.ascontiguousarray(p.transpose(2, 0, 1)) for p in patches])
                t = np_to_torch(stacked, device=device, dtype=torch.float32, torch=torch)
                lv = torch.full((N,), lv_scalar, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = model(t, lv)
                out = torch_to_np(out.clamp(0.0, 1.0).float())
                return [out[i].transpose(1, 2, 0) for i in range(out.shape[0])]

    result = _run_tiled_cosine(
        _infer_batch, _ensure_rgb(img_rgb01),
        tile=tile, overlap=overlap, pad=pad,
        normalize_brightness=(mode != "aesthetics"), target_median=0.10,
        progress_cb=progress_cb, label=f"[StarReduce/{mode} L{level}]",
        batch_size=bs,
    )

    info = {
        "variant": "star_reduce",
        "mode":    mode,
        "level":   level,
        "device":  str(device),
        "batch_size_cfg":    str(batch_size),
        "batch_size_active": int(bs),
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
    batch_size="Auto",
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

    # Autocast (fp16) only on CUDA, and NEVER on macOS. On Mac we always run
    # fp32 with no autocast context at all — the MPS autocast path is broken
    # or absent on several configs (notably Intel Mac Pro on Tahoe 26, where
    # the OS falsely reports MPS as available), and torch.amp.autocast
    # validates device_type in its constructor BEFORE honoring enabled=,
    # so even a disabled context with device_type="mps" raises. fp32
    # everywhere on Mac is correct and safe.
    import sys as _sys
    use_autocast   = (device.type == "cuda") and (_sys.platform != "darwin")
    autocast_dtype = torch.float16

    bs = _resolve_batch_size(batch_size, device, tile)

    # Streaming batch: gather up to bs padded tiles, run one forward pass,
    # scatter results back into the accumulator with per-tile trim.
    batch_recs: list = []   # (top, left, bottom, right, tile_h, tile_w, tile_padded_np)
    done = 0

    def _flush_sharpen():
        nonlocal done
        if not batch_recs:
            return
        stacked = np.stack([r[6].transpose(2, 0, 1) for r in batch_recs])  # (B,3,H,W)
        batch_tensor = np_to_torch(stacked, device=device, dtype=torch.float32, torch=torch)
        with torch.no_grad():
            if use_autocast:
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    pred_batch = model(batch_tensor)
            else:
                pred_batch = model(batch_tensor)
        pred_batch = pred_batch.float().cpu()
        for bi, (top, left, bottom, right, tile_h, tile_w, _) in enumerate(batch_recs):
            pred_patch   = pred_batch[bi:bi+1][:, :, :tile_h, :tile_w]
            window_patch = base_window[:, :, :tile_h, :tile_w]
            output_tensor[:, :, top:bottom, left:right] += pred_patch * window_patch
            weight_sum   [:, :, top:bottom, left:right] += window_patch
            done += 1
            if callable(progress_cb):
                progress_cb(done, total, f"[Sharpen] tile {done}/{total}…")
        batch_recs.clear()

    for (top, left) in coords:
        bottom = min(top + tile, height)
        right  = min(left + tile, width)
        tile_h = bottom - top
        tile_w = right - left

        tile_data        = img_padded[top:bottom, left:right, :]
        tile_data_padded = _pad_tile_np(tile_data, tile)

        batch_recs.append((top, left, bottom, right, tile_h, tile_w, tile_data_padded))
        if len(batch_recs) >= bs:
            _flush_sharpen()

    _flush_sharpen()

    # Reconstruct, unpad, then alpha blend against original unpadded image
    reconstructed = output_tensor / torch.clamp(weight_sum, min=1e-6)
    reconstructed = torch_to_np(reconstructed.squeeze(0).permute(1, 2, 0))
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
        "batch_size_cfg":    str(batch_size),
        "batch_size_active": int(bs),
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