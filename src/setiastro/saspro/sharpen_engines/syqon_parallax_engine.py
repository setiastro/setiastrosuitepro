from __future__ import annotations

import os
from typing import Callable, Optional, Tuple
import numpy as np

ProgressCB = Callable[[int, int, str], None]

_SYQON_PARALLAX_SESSION = None
_SYQON_PARALLAX_CKPT = None


def _infer_device(torch, *, prefer_cuda: bool = True, prefer_dml: bool = True):
    if prefer_cuda and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_dml and (os.name == "nt"):
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_parallax_model(
    ckpt_path: str,
    *,
    variant: str = "deblur",   # deblur | star_reduce | star_abcorr
    use_gpu: bool = True,
    prefer_dml: bool = True,
):
    """
    Load a SyQon Parallax sharpening model from a checkpoint file.
    Currently a placeholder — returns the passthrough model until SyQon
    releases the architecture and weights.
    """
    from setiastro.saspro.runtime_torch import import_torch
    from setiastro.saspro.syqon_parallax_model.model import create_parallax_model

    torch = import_torch(
        prefer_cuda=use_gpu,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=lambda *_: None,
    )

    model = create_parallax_model(variant=variant)

    # When real weights are available, load them here:
    # sd = torch.load(ckpt_path, map_location="cpu")
    # model.load_state_dict(sd, strict=False)

    model.eval()
    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)
    model.to(device)

    info = {
        "variant": variant,
        "device": str(device),
        "torch_version": getattr(torch, "__version__", None),
        "torch_file": getattr(torch, "__file__", None),
        "placeholder": True,  # remove when real model is wired
    }
    return model, device, info, torch


def parallax_sharpen_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    *,
    variant: str = "deblur",   # deblur | star_reduce | star_abcorr
    tile: int = 512,
    overlap: int = 64,
    use_gpu: bool = True,
    prefer_dml: bool = True,
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    strength: float = 1.0,
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Run SyQon Parallax sharpening on a float32 RGB image in [0,1].

    Currently a passthrough placeholder — returns input unchanged.
    Tiling, Hanning blending, and AMP logic mirrors syqon_prism_engine
    and will be activated when SyQon releases the model.

    Args:
        img_rgb01:   H×W×3 float32 numpy array in [0,1]
        ckpt_path:   Path to the .pt model file
        variant:     parallax_lite | parallax_standard | parallax_pro
        tile:        Tile size in pixels
        overlap:     Overlap between tiles
        use_gpu:     Prefer GPU inference
        prefer_dml:  Prefer DirectML on Windows
        use_amp:     Use automatic mixed precision (CUDA only)
        amp_dtype:   fp16 or bf16
        strength:    Blend factor [0,1] between original and sharpened
        progress_cb: Optional callback(done, total, stage)

    Returns:
        (sharpened_rgb01, info_dict)
    """
    x = np.asarray(img_rgb01, dtype=np.float32)
    was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    else:
        x = x[..., :3]

    H, W = x.shape[:2]
    tile = int(tile)
    overlap = int(overlap)
    stride = max(tile - overlap, 1)

    model, device, info, torch = load_parallax_model(
        ckpt_path,
        variant=variant,
        use_gpu=use_gpu,
        prefer_dml=prefer_dml,
    )

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc   = np.zeros((H, W, 1), dtype=np.float32)

    wy = np.hanning(tile).astype(np.float32)
    wx = np.hanning(tile).astype(np.float32)
    w2 = (wy[:, None] * wx[None, :]).astype(np.float32)[..., None]

    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    total = len(ys) * len(xs)
    done  = 0

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)

                patch = x[y0:y1, x0:x1, :]
                ph, pw = patch.shape[:2]

                if ph != tile or pw != tile:
                    pad = np.zeros((tile, tile, 3), dtype=np.float32)
                    pad[:ph, :pw, :] = patch
                    patch = pad

                # --- inference (passthrough until real model wired) ---
                chw = patch.transpose(2, 0, 1)
                t = torch.from_numpy(chw[None, ...]).to(device=device, dtype=torch.float32)
                pred_t = model(t)
                pred = pred_t[0].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                # -------------------------------------------------------

                sharp_patch = np.clip(pred, 0.0, 1.0)[:ph, :pw, :]
                wlocal = w2[:ph, :pw, :]

                out_acc[y0:y1, x0:x1, :] += sharp_patch * wlocal
                w_acc[y0:y1, x0:x1, :]  += wlocal

                done += 1
                if callable(progress_cb):
                    progress_cb(done, total, "SyQon Parallax tiles…")

    sharpened = out_acc / np.maximum(w_acc, 1e-8)
    sharpened = np.clip(sharpened, 0.0, 1.0).astype(np.float32)

    # Strength blend
    strength = float(np.clip(strength, 0.0, 1.0))
    result = (1.0 - strength) * x + strength * sharpened
    result = np.clip(result, 0.0, 1.0).astype(np.float32)

    if was_mono:
        return result.mean(axis=2).astype(np.float32), info
    return result, info


def clear_parallax_models_cache(*, aggressive: bool = False, status_cb=print) -> None:
    global _SYQON_PARALLAX_SESSION, _SYQON_PARALLAX_CKPT

    try:
        had_session = _SYQON_PARALLAX_SESSION is not None
        had_ckpt    = _SYQON_PARALLAX_CKPT is not None

        _SYQON_PARALLAX_SESSION = None
        _SYQON_PARALLAX_CKPT    = None

        status_cb(
            f"[SyQon Parallax] Cleared cache "
            f"(session={'yes' if had_session else 'no'}, ckpt={'yes' if had_ckpt else 'no'})"
        )
    except Exception as e:
        try:
            status_cb(f"[SyQon Parallax] Cache clear failed: {type(e).__name__}: {e}")
        except Exception:
            pass

    if not aggressive:
        return

    try:
        import gc
        gc.collect()
        status_cb("[SyQon Parallax] gc.collect() called")
    except Exception:
        pass

    try:
        from setiastro.saspro.runtime_torch import import_torch
        torch = import_torch(
            prefer_cuda=True,
            prefer_xpu=False,
            prefer_dml=True,
            status_cb=lambda *_: None,
        )
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
            status_cb("[SyQon Parallax] torch.cuda.empty_cache() called")
    except Exception:
        pass