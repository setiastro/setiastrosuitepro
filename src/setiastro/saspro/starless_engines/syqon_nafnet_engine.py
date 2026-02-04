#src/setiastro/saspro/starless_engines/syqon_nafnet_engine.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple

def _infer_device(prefer_cuda=True):
    import torch
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    # (optional) MPS support if you want:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _load_state_dict(ckpt_path: str):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        # common patterns
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"], ckpt
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"], ckpt
        # or it might already be a raw sd
        if any(k.startswith(("intro.", "ending.", "encoders.", "downs.", "middle.", "decoders.", "ups.")) for k in ckpt.keys()):
            return ckpt, {}
    raise RuntimeError("Unsupported checkpoint format (expected state_dict-like dict).")

def _infer_nafnet_cfg_from_sd(sd: dict) -> Tuple[int, int]:
    """
    Infer NAFNet config from state_dict produced by our NAFNet class.

    Returns:
        base_ch: intro conv out channels
        levels: number of encoder/decoder levels (len(encoders) == len(downs) == levels)
    """
    base_ch = None
    if "intro.weight" in sd:
        base_ch = int(sd["intro.weight"].shape[0])
    else:
        base_ch = 32  # safe fallback

    # Prefer downs.N.* indices (most robust for our architecture)
    downs_idx = set()
    for k in sd.keys():
        if k.startswith("downs."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                downs_idx.add(int(parts[1]))

    if downs_idx:
        levels = max(downs_idx) + 1
    else:
        # Fallback to encoders.N.*
        enc_idx = set()
        for k in sd.keys():
            if k.startswith("encoders."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    enc_idx.add(int(parts[1]))
        levels = (max(enc_idx) + 1) if enc_idx else 4  # reasonable default

    # Clamp to something sane
    levels = max(1, min(8, int(levels)))

    return base_ch, levels


def load_nafnet_model(ckpt_path: str, prefer_cuda=True):
    import torch
    from setiastro.saspro.syqon_model.model import create_model  # <-- you will place model.py here
    sd, meta = _load_state_dict(ckpt_path)
    base_ch, depth = _infer_nafnet_cfg_from_sd(sd)

    model = create_model(base_ch=base_ch, depth=depth)
    model.load_state_dict(sd, strict=False)
    model.eval()

    device = _infer_device(prefer_cuda=prefer_cuda)
    model.to(device)

    return model, device, {"base_ch": base_ch, "depth": depth, "meta": meta}

def _to_torch_chw(img_chw: np.ndarray, device):
    import torch
    x = np.asarray(img_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("expected CHW float32")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    t = torch.from_numpy(x[None, ...])  # 1CHW
    return t.to(device=device, dtype=torch.float32)

def nafnet_starless_rgb01(
    img_rgb01: np.ndarray,
    ckpt_path: str,
    *,
    tile: int = 512,
    overlap: int = 64,
    prefer_cuda: bool = True,
    residual_mode: bool = True,  # True = model predicts stars; starless = x - pred
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
):
    """
    Input: HWC float32 [0,1], RGB (or mono HxW)
    Output: starless HWC float32 [0,1] and stars_only HWC float32 [0,1]
    """
    import torch

    x = np.asarray(img_rgb01, dtype=np.float32)
    was_mono = (x.ndim == 2) or (x.ndim == 3 and x.shape[2] == 1)

    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    else:
        x = x[..., :3]

    H, W = x.shape[:2]
    tile = int(tile)
    overlap = int(overlap)
    stride = max(tile - overlap, 1)

    model, device, info = load_nafnet_model(ckpt_path, prefer_cuda=prefer_cuda)

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc = np.zeros((H, W, 1), dtype=np.float32)

    # simple feather weight (cosine-ish)
    wy = np.hanning(tile).astype(np.float32)
    wx = np.hanning(tile).astype(np.float32)
    w2 = (wy[:, None] * wx[None, :]).astype(np.float32)
    w2 = w2[..., None]  # H W 1

    # tile loops
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    total = len(ys) * len(xs)
    done = 0

    with torch.no_grad():
        for y0 in ys:
            for x0 in xs:
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)
                # pad to tile
                patch = x[y0:y1, x0:x1, :]
                ph, pw = patch.shape[:2]
                if ph != tile or pw != tile:
                    pad = np.zeros((tile, tile, 3), dtype=np.float32)
                    pad[:ph, :pw, :] = patch
                    patch = pad

                chw = patch.transpose(2, 0, 1)  # CHW
                t = _to_torch_chw(chw, device)
                pred = model(t)  # 1CHW
                pred = pred[0].detach().to("cpu").numpy().transpose(1, 2, 0)  # HWC

                if residual_mode:
                    starless_patch = patch - pred
                    stars_patch = pred
                else:
                    starless_patch = pred
                    stars_patch = patch - pred

                starless_patch = np.clip(starless_patch, 0.0, 1.0).astype(np.float32, copy=False)
                stars_patch = np.clip(stars_patch, 0.0, 1.0).astype(np.float32, copy=False)

                # crop back to valid region
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
        return starless.mean(axis=2).astype(np.float32, copy=False), stars_only.mean(axis=2).astype(np.float32, copy=False), info
    return starless, stars_only, info
