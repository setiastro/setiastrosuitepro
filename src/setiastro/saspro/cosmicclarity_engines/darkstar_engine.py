# src/setiastro/saspro/cosmicclarity_engines/darkstar_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from setiastro.saspro.resources import get_resources

# Optional deps
try:
    import onnxruntime as ort
except Exception:
    ort = None

ProgressCB = Callable[[int, int, str], None]  # (done, total, stage)


# ---------------- Torch import (your existing runtime_torch helper) ----------------

def _get_torch(status_cb=print):
    from setiastro.saspro.runtime_torch import import_torch
    return import_torch(prefer_cuda=True, prefer_xpu=False, status_cb=status_cb)


def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()


def _autocast_context(torch, device) -> Any:
    # Keep your ">= 8.0" rule (match your other CC engines)
    try:
        if hasattr(device, "type") and device.type == "cuda":
            major, minor = torch.cuda.get_device_capability()
            cap = float(f"{major}.{minor}")
            if cap >= 8.0:
                return torch.cuda.amp.autocast()
    except Exception:
        pass
    return _nullcontext()


# ---------------- Models (same topology as your script) ----------------

def _build_darkstar_torch_models(torch):
    import torch.nn as nn

    class RefinementCNN(nn.Module):
        def __init__(self, channels: int = 96):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, channels, 3, padding=1, dilation=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=4, dilation=4), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=8, dilation=8), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=8, dilation=8), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=4, dilation=4), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(channels, 3,      3, padding=1, dilation=1), nn.Sigmoid()
            )
        def forward(self, x): return self.net(x)

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.relu  = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        def forward(self, x):
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            return self.relu(out + x)

    class DarkStarCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(16), ResidualBlock(16), ResidualBlock(16),
            )
            self.encoder2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(32), ResidualBlock(32), ResidualBlock(32),
            )
            self.encoder3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                ResidualBlock(64), ResidualBlock(64),
            )
            self.encoder4 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(128), ResidualBlock(128),
            )
            self.encoder5 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                ResidualBlock(256),
            )

            self.decoder5 = nn.Sequential(
                nn.Conv2d(256 + 128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(128), ResidualBlock(128),
            )
            self.decoder4 = nn.Sequential(
                nn.Conv2d(128 + 64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(64), ResidualBlock(64),
            )
            self.decoder3 = nn.Sequential(
                nn.Conv2d(64 + 32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(32), ResidualBlock(32), ResidualBlock(32),
            )
            self.decoder2 = nn.Sequential(
                nn.Conv2d(32 + 16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(16), ResidualBlock(16), ResidualBlock(16),
            )
            self.decoder1 = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(16), ResidualBlock(16),
                nn.Conv2d(16, 3, 3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            e1 = self.encoder1(x)
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)
            e5 = self.encoder5(e4)

            d5 = self.decoder5(torch.cat([e5, e4], dim=1))
            d4 = self.decoder4(torch.cat([d5, e3], dim=1))
            d3 = self.decoder3(torch.cat([d4, e2], dim=1))
            d2 = self.decoder2(torch.cat([d3, e1], dim=1))
            return self.decoder1(d2)

    class CascadedStarRemovalNetCombined(nn.Module):
        def __init__(self, stage1_path: str, stage2_path: str | None = None):
            super().__init__()
            self.stage1 = DarkStarCNN()
            ckpt1 = torch.load(stage1_path, map_location="cpu")

            # strip "stage1." prefix if present
            if isinstance(ckpt1, dict):
                sd1 = {k[len("stage1."):] : v for k, v in ckpt1.items() if k.startswith("stage1.")}
                if sd1:
                    ckpt1 = sd1
            self.stage1.load_state_dict(ckpt1)

            # refinement exists in your code but currently not used (forward returns coarse)
            self.stage2 = RefinementCNN()
            if stage2_path:
                try:
                    ckpt2 = torch.load(stage2_path, map_location="cpu")
                    if isinstance(ckpt2, dict) and "model_state" in ckpt2:
                        ckpt2 = ckpt2["model_state"]
                    self.stage2.load_state_dict(ckpt2)
                except Exception:
                    pass

            for p in self.stage1.parameters():
                p.requires_grad = False

        def forward(self, x):
            with torch.no_grad():
                coarse = self.stage1(x)
            return coarse

    return CascadedStarRemovalNetCombined


# ---------------- Stretch/unstretch + borders (match your other engines) ----------------

def add_border(image: np.ndarray, border_size: int = 5) -> np.ndarray:
    if image.ndim == 2:
        med = float(np.median(image))
        return np.pad(image, ((border_size, border_size), (border_size, border_size)),
                      mode="constant", constant_values=med)
    if image.ndim == 3 and image.shape[2] == 3:
        meds = np.median(image, axis=(0, 1)).astype(np.float32)
        chans = []
        for c in range(3):
            chans.append(np.pad(image[..., c], ((border_size, border_size), (border_size, border_size)),
                                mode="constant", constant_values=float(meds[c])))
        return np.stack(chans, axis=-1)
    raise ValueError("add_border expects 2D or HxWx3")


def remove_border(image: np.ndarray, border_size: int = 5) -> np.ndarray:
    if image.ndim == 2:
        return image[border_size:-border_size, border_size:-border_size]
    return image[border_size:-border_size, border_size:-border_size, :]


def stretch_image_unlinked_rgb(img_rgb: np.ndarray, target_median: float = 0.25):
    x = img_rgb.astype(np.float32, copy=True)
    orig_min = x.reshape(-1, 3).min(axis=0)  # (3,)
    x = (x - orig_min.reshape(1, 1, 3))
    orig_meds = np.median(x, axis=(0, 1)).astype(np.float32)

    for c in range(3):
        m = float(orig_meds[c])
        if m != 0:
            x[..., c] = ((m - 1) * target_median * x[..., c]) / (
                m * (target_median + x[..., c] - 1) - target_median * x[..., c]
            )
    x = np.clip(x, 0, 1)
    return x, orig_min.astype(np.float32), orig_meds.astype(np.float32)


def unstretch_image_unlinked_rgb(img_rgb: np.ndarray, orig_meds, orig_min):
    x = img_rgb.astype(np.float32, copy=True)
    for c in range(3):
        m_now = float(np.median(x[..., c]))
        m0 = float(orig_meds[c])
        if m_now != 0 and m0 != 0:
            x[..., c] = ((m_now - 1) * m0 * x[..., c]) / (
                m_now * (m0 + x[..., c] - 1) - m0 * x[..., c]
            )
    x = x + orig_min.reshape(1, 1, 3)
    return np.clip(x, 0, 1).astype(np.float32, copy=False)


# ---------------- Chunking & stitch (soft blend like your script) ----------------

def split_image_into_chunks_with_overlap(image: np.ndarray, chunk_size: int, overlap: int):
    H, W = image.shape[:2]
    step = chunk_size - overlap
    out = []
    for i in range(0, H, step):
        for j in range(0, W, step):
            ei = min(i + chunk_size, H)
            ej = min(j + chunk_size, W)
            out.append((image[i:ei, j:ej], i, j))
    return out


def _blend_weights(chunk_size: int, overlap: int):
    if overlap <= 0:
        return np.ones((chunk_size, chunk_size), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    flat = np.ones(max(chunk_size - 2 * overlap, 1), dtype=np.float32)
    v = np.concatenate([ramp, flat, ramp[::-1]])
    w = np.outer(v, v).astype(np.float32)
    return w


def stitch_chunks_soft_blend(
    chunks: list[tuple[np.ndarray, int, int]],
    out_shape: tuple[int, int, int],
    *,
    chunk_size: int,
    overlap: int,
    border_size: int = 5,
) -> np.ndarray:
    H, W, C = out_shape
    out = np.zeros((H, W, C), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    bw_full = _blend_weights(chunk_size, overlap)

    for tile, i, j in chunks:
        th, tw = tile.shape[:2]

        # adaptive inner crop like your script
        top    = 0 if i == 0 else min(border_size, th // 2)
        left   = 0 if j == 0 else min(border_size, tw // 2)
        bottom = 0 if (i + th) >= H else min(border_size, th // 2)
        right  = 0 if (j + tw) >= W else min(border_size, tw // 2)

        inner = tile[top:th-bottom, left:tw-right, :]
        ih, iw = inner.shape[:2]

        rr0 = i + top
        cc0 = j + left
        rr1 = rr0 + ih
        cc1 = cc0 + iw

        bw = bw_full[:ih, :iw].reshape(ih, iw, 1)
        out[rr0:rr1, cc0:cc1, :] += inner * bw
        wsum[rr0:rr1, cc0:cc1, :] += bw

    out = out / np.maximum(wsum, 1e-8)
    return out


# ---------------- Model loading (cached) ----------------

@dataclass
class DarkStarModels:
    device: Any
    is_onnx: bool
    model: Any
    torch: Any | None = None
    chunk_size: int = 512  # used for ONNX fixed shapes


_MODELS_CACHE: dict[tuple[str, bool, bool], DarkStarModels] = {}  # (tag,use_gpu,color)

def load_darkstar_models(*, use_gpu: bool, color: bool, status_cb=print) -> DarkStarModels:
    R = get_resources()

    # you’ll want these resource keys added (see note below)
    if color:
        pth = R.CC_DARKSTAR_COLOR_PTH
        onnx = R.CC_DARKSTAR_COLOR_ONNX
        tag = "cc_darkstar_color"
    else:
        pth = R.CC_DARKSTAR_MONO_PTH
        onnx = R.CC_DARKSTAR_MONO_ONNX
        tag = "cc_darkstar_mono"

    key = (tag, bool(use_gpu), bool(color))
    if key in _MODELS_CACHE:
        return _MODELS_CACHE[key]

    torch = _get_torch(status_cb=status_cb)

    # CUDA torch first
    if use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
        dev = torch.device("cuda")
        status_cb(f"Dark Star: using CUDA ({torch.cuda.get_device_name(0)})")
        Net = _build_darkstar_torch_models(torch)
        net = Net(pth, None).eval().to(dev)
        m = DarkStarModels(device=dev, is_onnx=False, model=net, torch=torch, chunk_size=512)
        _MODELS_CACHE[key] = m
        return m

    # Windows DirectML ONNX
    if use_gpu and ort is not None and ("DmlExecutionProvider" in ort.get_available_providers()):
        if onnx and onnx.strip():
            status_cb("Dark Star: using DirectML (ONNX Runtime)")
            sess = ort.InferenceSession(onnx, providers=["DmlExecutionProvider"])
            # fixed input: [1,3,H,W]
            inp = sess.get_inputs()[0]
            cs = int(inp.shape[2]) if inp.shape and inp.shape[2] else 512
            m = DarkStarModels(device="DirectML", is_onnx=True, model=sess, torch=None, chunk_size=cs)
            _MODELS_CACHE[key] = m
            return m

    # MPS torch
    if use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        status_cb("Dark Star: using MPS")
        Net = _build_darkstar_torch_models(torch)
        net = Net(pth, None).eval().to(dev)
        m = DarkStarModels(device=dev, is_onnx=False, model=net, torch=torch, chunk_size=512)
        _MODELS_CACHE[key] = m
        return m

    # CPU torch
    dev = torch.device("cpu")
    status_cb("Dark Star: using CPU")
    Net = _build_darkstar_torch_models(torch)
    net = Net(pth, None).eval().to(dev)
    m = DarkStarModels(device=dev, is_onnx=False, model=net, torch=torch, chunk_size=512)
    _MODELS_CACHE[key] = m
    return m


# ---------------- Core inference on one HxWx3 image ----------------

def _infer_tile(models: DarkStarModels, tile_rgb: np.ndarray) -> np.ndarray:
    tile_rgb = np.asarray(tile_rgb, np.float32)
    h0, w0 = tile_rgb.shape[:2]

    if models.is_onnx:
        cs = int(models.chunk_size)
        if (h0 != cs) or (w0 != cs):
            pad = np.zeros((cs, cs, 3), np.float32)
            pad[:h0, :w0, :] = tile_rgb
            tile_rgb = pad
        inp = tile_rgb.transpose(2, 0, 1)[None, ...]  # 1,3,H,W
        sess = models.model
        out = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]  # 3,H,W
        out = out.transpose(1, 2, 0)
        return out[:h0, :w0, :].astype(np.float32, copy=False)

    # torch
    torch = models.torch
    dev = models.device
    t = torch.from_numpy(tile_rgb.transpose(2, 0, 1)).unsqueeze(0).to(dev)
    with torch.no_grad(), _autocast_context(torch, dev):
        y = models.model(t)[0].detach().cpu().numpy().transpose(1, 2, 0)
    return y[:h0, :w0, :].astype(np.float32, copy=False)


# ---------------- Public API ----------------

@dataclass
class DarkStarParams:
    use_gpu: bool = True
    chunk_size: int = 512
    overlap_frac: float = 0.125
    mode: str = "unscreen"        # "unscreen" or "additive"
    output_stars_only: bool = False


def darkstar_starremoval_rgb01(
    img_rgb01: np.ndarray,
    *,
    params: DarkStarParams,
    progress_cb: Optional[ProgressCB] = None,
    status_cb=print,
) -> tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    Input : float32 image in [0..1], shape HxWx3 or HxWx1 or HxW
    Output: (starless_rgb01, stars_only_rgb01 or None, was_mono)
    """
    if progress_cb is None:
        progress_cb = lambda done, total, stage: None

    img = np.asarray(img_rgb01, np.float32)
    was_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

    # normalize shape to HxWx3
    if img.ndim == 2:
        img3 = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        ch = img[..., 0]
        img3 = np.stack([ch, ch, ch], axis=-1)
    else:
        img3 = img

    img3 = np.clip(img3, 0.0, 1.0)

    # decide "true RGB" vs "3-channel mono"
    same_rg = np.allclose(img3[...,0], img3[...,1], rtol=0, atol=1e-6)
    same_rb = np.allclose(img3[...,0], img3[...,2], rtol=0, atol=1e-6)
    is_true_rgb = not (same_rg and same_rb)

    models = load_darkstar_models(use_gpu=params.use_gpu, color=is_true_rgb, status_cb=status_cb)

    # border + optional stretch decision (like your script)
    stretch_needed = float(np.median(img3)) < 0.125
    if stretch_needed:
        stretched, orig_min, orig_meds = stretch_image_unlinked_rgb(img3)
    else:
        stretched, orig_min, orig_meds = img3, None, None

    bordered = add_border(stretched, border_size=5)

    # ONNX may force chunk_size
    chunk_size = int(models.chunk_size) if models.is_onnx else int(params.chunk_size)
    overlap = int(round(float(params.overlap_frac) * chunk_size))

    chunks = split_image_into_chunks_with_overlap(bordered, chunk_size=chunk_size, overlap=overlap)
    total = len(chunks)

    out_tiles: list[tuple[np.ndarray, int, int]] = []
    for k, (tile, i, j) in enumerate(chunks, start=1):
        out = _infer_tile(models, tile)
        out_tiles.append((out, i, j))
        progress_cb(k, total, "Dark Star removal")

    starless_b = stitch_chunks_soft_blend(
        out_tiles,
        bordered.shape,
        chunk_size=chunk_size,
        overlap=overlap,
        border_size=5,
    )

    # un-stretch if applied
    if stretch_needed:
        starless_b = unstretch_image_unlinked_rgb(starless_b, orig_meds, orig_min)

    # remove border
    starless = remove_border(starless_b, border_size=5)

    starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)

    stars_only = None
    if params.output_stars_only:
        if params.mode == "additive":
            stars_only = np.clip(img3 - starless, 0.0, 1.0)
        else:  # unscreen
            denom = np.maximum(1.0 - starless, 1e-6)
            stars_only = np.clip((img3 - starless) / denom, 0.0, 1.0).astype(np.float32, copy=False)

    # if input was mono, return mono-ish output? (your CC behavior usually returns HxWx1)
    if was_mono:
        starless = np.mean(starless, axis=2, keepdims=True).astype(np.float32, copy=False)
        if stars_only is not None:
            stars_only = np.mean(stars_only, axis=2, keepdims=True).astype(np.float32, copy=False)

    return starless, stars_only, was_mono
