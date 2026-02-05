#src/setiastro/saspro/starless_engines/syqon_nafnet_engine.py
from __future__ import annotations

from pathlib import Path
import numpy as np
from typing import Callable, Optional, Tuple
import os

def _infer_device(torch, *, prefer_cuda: bool = True, prefer_dml: bool = True):
    # CUDA
    if prefer_cuda and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return torch.device("cuda")

    # DirectML (Windows)
    if prefer_dml and (os.name == "nt"):
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass

    # MPS (macOS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def _load_state_dict(torch, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], ckpt
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"], ckpt
        if any(
            k.startswith(
                (
                    "intro.", "ending.", "encoders.", "downs.", "middle.", "decoders.", "ups."
                )
            )
            for k in ckpt.keys()
        ):
            return ckpt, {}
    raise RuntimeError("Unsupported checkpoint format (expected state_dict-like dict).")


def _infer_nafnet_cfg_from_sd(sd: dict) -> Tuple[int, int]:
    base_ch = int(sd["intro.weight"].shape[0]) if "intro.weight" in sd else 32

    downs_idx = set()
    for k in sd.keys():
        if k.startswith("downs."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                downs_idx.add(int(parts[1]))

    if downs_idx:
        levels = max(downs_idx) + 1
    else:
        enc_idx = set()
        for k in sd.keys():
            if k.startswith("encoders."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    enc_idx.add(int(parts[1]))
        levels = (max(enc_idx) + 1) if enc_idx else 4

    levels = max(1, min(8, int(levels)))
    return base_ch, levels


def load_nafnet_model(ckpt_path: str, *, use_gpu: bool, prefer_dml: bool):
    from setiastro.saspro.runtime_torch import import_torch

    torch = import_torch(
        prefer_cuda=use_gpu,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=lambda *_: None,
    )

    from setiastro.saspro.syqon_model.model import create_model

    sd, meta = _load_state_dict(torch, ckpt_path)
    base_ch, depth = _infer_nafnet_cfg_from_sd(sd)

    model = create_model(base_ch=base_ch, depth=depth)
    model.load_state_dict(sd, strict=False)
    model.eval()

    device = _infer_device(torch, prefer_cuda=use_gpu, prefer_dml=prefer_dml)
    model.to(device)

    info = {
        "base_ch": base_ch,
        "depth": depth,
        "meta": meta,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
    }
    return model, device, info, torch


def _to_torch_chw(img_chw, device, torch):
    x = np.asarray(img_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("expected CHW float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    t = torch.from_numpy(x[None, ...])
    return t.to(device=device, dtype=torch.float32)


def _predict_tile(
    model,
    t,
    *,
    device,
    use_amp: bool,
    amp_dtype: str,
    info: dict,
    torch,
):
    """
    Run model(t) and return pred as HWC float32 numpy.

    If AMP is enabled and non-finite output occurs, fallback to fp32 for this tile
    and record info["amp_fallback"] / info["amp_reason"].
    """

    def _to_numpy(pred_t):
        # pred_t: 1CHW torch
        pred_np = pred_t[0].detach().to("cpu").numpy().transpose(1, 2, 0)
        return pred_np.astype(np.float32, copy=False)

    # --- AMP path (if requested & supported) ---
    if use_amp and device.type == "cuda":
        dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
        with torch.cuda.amp.autocast(dtype=dtype):
            pred_t = model(t)
        pred = _to_numpy(pred_t)

        if not np.isfinite(pred).all():
            # fallback to fp32 for this tile
            info["amp_fallback"] = True
            info["amp_reason"] = "non-finite output detected under CUDA AMP; reran tile in fp32"
            pred_t = model(t)
            pred = _to_numpy(pred_t)

        return pred

    if use_amp and device.type == "mps":
        # Conservative: autocast enabled, no explicit dtype forcing
        try:
            with torch.autocast(device_type="mps"):
                pred_t = model(t)
            pred = _to_numpy(pred_t)

            if not np.isfinite(pred).all():
                info["amp_fallback"] = True
                info["amp_reason"] = "non-finite output detected under MPS autocast; reran tile in fp32"
                pred_t = model(t)
                pred = _to_numpy(pred_t)

            return pred
        except Exception:
            # If autocast isn't supported in their torch build, just do fp32
            pred_t = model(t)
            pred = _to_numpy(pred_t)
            return pred

    # --- fp32 path ---
    pred_t = model(t)
    pred = _to_numpy(pred_t)

    if not np.isfinite(pred).all():
        raise RuntimeError("Non-finite output detected in fp32 inference.")

    return pred


def nafnet_starless_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    *,
    tile: int = 512,
    overlap: int = 64,
    prefer_cuda: bool = True,
    residual_mode: bool = True,  # True = model predicts stars; starless = x - pred
    use_amp: bool = False,
    amp_dtype: str = "fp16",
    use_gpu: bool = True, prefer_dml: bool = True,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
):
    """
    Input: HWC float32 [0,1], RGB (or mono HxW)
    Output: starless HWC float32 [0,1] and stars_only HWC float32 [0,1]
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

    model, device, info, torch = load_nafnet_model(
        ckpt_path,
        use_gpu=use_gpu,
        prefer_dml=prefer_dml,
    )

    info = dict(info or {})
    info["device"] = str(device)
    info["torch_version"] = getattr(torch, "__version__", None)
    info["torch_file"] = getattr(torch, "__file__", None)
    # AMP config
    use_amp_requested = bool(use_amp)
    amp_dtype = (amp_dtype or "fp16").lower()
    if amp_dtype not in ("fp16", "bf16"):
        amp_dtype = "fp16"

    # only allow AMP where it makes sense
    use_amp_effective = bool(use_amp_requested) and (device.type in ("cuda", "mps"))
    info["use_amp_requested"] = use_amp_requested
    info["use_amp_effective"] = use_amp_effective
    info["amp_dtype"] = amp_dtype

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc = np.zeros((H, W, 1), dtype=np.float32)

    # feather weight
    wy = np.hanning(tile).astype(np.float32)
    wx = np.hanning(tile).astype(np.float32)
    w2 = (wy[:, None] * wx[None, :]).astype(np.float32)
    w2 = w2[..., None]  # tile x tile x 1

    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    total = len(ys) * len(xs)
    done = 0

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

                chw = patch.transpose(2, 0, 1)  # CHW
                t = _to_torch_chw(chw, device, torch)

                pred = _predict_tile(
                    model,
                    t,
                    device=device,
                    use_amp=use_amp_effective,
                    amp_dtype=amp_dtype,
                    info=info,
                    torch=torch,
                )  # HWC float32

                if residual_mode:
                    starless_patch = patch - pred
                    stars_patch = pred
                else:
                    starless_patch = pred
                    stars_patch = patch - pred

                starless_patch = np.clip(starless_patch, 0.0, 1.0).astype(np.float32, copy=False)
                stars_patch = np.clip(stars_patch, 0.0, 1.0).astype(np.float32, copy=False)

                starless_patch = starless_patch[:ph, :pw, :]
                wlocal = w2[:ph, :pw, :]

                out_acc[y0:y1, x0:x1, :] += starless_patch * wlocal
                w_acc[y0:y1, x0:x1, :] += wlocal

                done += 1
                if callable(progress_cb):
                    progress_cb(done, total, "SyQon NAFNet tiles…")

    starless = out_acc / np.maximum(w_acc, 1e-8)
    starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)
    stars_only = np.clip(x - starless, 0.0, 1.0).astype(np.float32, copy=False)

    if was_mono:
        return (
            starless.mean(axis=2).astype(np.float32, copy=False),
            stars_only.mean(axis=2).astype(np.float32, copy=False),
            info,
        )
    return starless, stars_only, info
