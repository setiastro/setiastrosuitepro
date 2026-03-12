from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import os
import numpy as np

# Keep these imports because your runtime torch loader is still the right way
# to get CUDA / DirectML / MPS / CPU fallback behavior.
from setiastro.saspro.runtime_torch import _user_runtime_dir, _venv_paths, _check_cuda_in_venv
from setiastro.saspro.resources import get_resources

ProgressCB = Callable[[int, int, str], None]  # (done, total, stage)


# -----------------------------------------------------------------------------
# Resolve model path
# -----------------------------------------------------------------------------

def _resolve_darkstar_model_paths() -> tuple[str, str, str, str]:
    """
    Resolve packaged Dark Star model paths from SAS resources.
    Returns:
        (mono_pth, mono_onnx, color_pth, color_onnx)
    """
    res = get_resources()
    mono_pth   = str(res.CC_DARKSTAR_MONO_PTH)
    mono_onnx  = str(res.CC_DARKSTAR_MONO_ONNX)
    color_pth  = str(res.CC_DARKSTAR_COLOR_PTH)
    color_onnx = str(res.CC_DARKSTAR_COLOR_ONNX)
    return mono_pth, mono_onnx, color_pth, color_onnx


# -----------------------------------------------------------------------------
# Torch import helper
# -----------------------------------------------------------------------------

def _get_torch(*, prefer_cuda: bool, prefer_dml: bool, status_cb=print):
    from setiastro.saspro.runtime_torch import import_torch
    return import_torch(
        prefer_cuda=prefer_cuda,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=status_cb,
    )


def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()


def _autocast_context(torch, device) -> Any:
    """
    Use autocast only where it is reasonably safe/useful.
    Keep your existing CUDA >= 8.0 behavior.
    """
    try:
        if hasattr(device, "type") and device.type == "cuda":
            major, minor = torch.cuda.get_device_capability()
            cap = float(f"{major}.{minor}")
            if cap >= 8.0:
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    return torch.amp.autocast(device_type="cuda")
                return torch.cuda.amp.autocast()

        elif hasattr(device, "type") and device.type == "mps":
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast(device_type="mps")

    except Exception:
        pass

    return _nullcontext()


# -----------------------------------------------------------------------------
# NAFNet model
# -----------------------------------------------------------------------------

def _build_darkstar_nafnet_model(torch):
    import torch.nn as nn
    import torch.nn.functional as F

    class LayerNorm2d(nn.Module):
        def __init__(self, channels: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mean = x.mean(dim=1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return x * self.weight + self.bias

    class SimpleGate(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class NAFBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.norm1 = LayerNorm2d(channels)

            self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
            self.dwconv = nn.Conv2d(
                channels * 2,
                channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels * 2,
                bias=True,
            )
            self.sg = SimpleGate()
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            )
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

            self.norm2 = LayerNorm2d(channels)
            self.ffn1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
            self.ffn2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

            self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.norm1(x)
            y = self.conv1(y)
            y = self.dwconv(y)
            y = self.sg(y)
            y = y * self.sca(y)
            y = self.conv2(y)
            x = x + y * self.beta

            y = self.norm2(x)
            y = self.ffn1(y)
            y = self.sg(y)
            y = self.ffn2(y)
            x = x + y * self.gamma
            return x

    class DarkStarNAFNet(nn.Module):
        def __init__(
            self,
            in_ch: int = 3,
            out_ch: int = 3,
            width: int = 32,
            enc_blk_nums=(2, 4, 6, 8),
            dec_blk_nums=(2, 2, 2, 2),
            middle_blk_num: int = 4,
        ):
            super().__init__()

            self.in_ch = in_ch
            self.out_ch = out_ch
            self.width = width
            self.enc_blk_nums = tuple(enc_blk_nums)
            self.dec_blk_nums = tuple(dec_blk_nums)
            self.middle_blk_num = int(middle_blk_num)

            self.intro = nn.Conv2d(in_ch, width, kernel_size=3, padding=1, bias=True)
            self.ending = nn.Conv2d(width, out_ch, kernel_size=3, padding=1, bias=True)

            self.encoders = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups = nn.ModuleList()

            ch = width
            for num in enc_blk_nums:
                self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
                self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2, bias=True))
                ch *= 2

            self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

            for num in dec_blk_nums:
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(ch, ch * 2, kernel_size=1, bias=True),
                        nn.PixelShuffle(2),
                    )
                )
                ch //= 2
                self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))

            self.padder_size = 2 ** len(enc_blk_nums)

        def check_image_size(self, x: torch.Tensor):
            _, _, h, w = x.size()
            mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
            mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
            if mod_pad_h != 0 or mod_pad_w != 0:
                x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
            return x, mod_pad_h, mod_pad_w

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            inp = x
            x, mod_pad_h, mod_pad_w = self.check_image_size(x)

            x = self.intro(x)

            encs = []
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encs.append(x)
                x = down(x)

            x = self.middle(x)

            for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
                x = up(x)
                x = x + enc_skip
                x = decoder(x)

            x = self.ending(x)

            if mod_pad_h != 0 or mod_pad_w != 0:
                x = x[:, :, :x.shape[2] - mod_pad_h if mod_pad_h > 0 else x.shape[2],
                          :x.shape[3] - mod_pad_w if mod_pad_w > 0 else x.shape[3]]

            if inp.shape[2:] == x.shape[2:]:
                x = x + inp

            return torch.clamp(x, 0.0, 1.0)

    return DarkStarNAFNet

def _extract_state_dict(ckpt):
    """
    Robustly find the actual state dict in a checkpoint.
    """
    if not isinstance(ckpt, dict):
        return ckpt

    preferred_keys = [
        "state_dict",
        "model_state_dict",
        "model_state",
        "model",
        "net",
        "network",
        "params",
        "params_ema",
    ]
    for k in preferred_keys:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]

    # If it already looks like a raw state dict, use it.
    if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    return ckpt


def _strip_common_prefixes(state_dict: dict):
    """
    Remove wrappers such as 'module.' or '_orig_mod.' if present.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = ("module.", "_orig_mod.")
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


# -----------------------------------------------------------------------------
# Stretch / unstretch / borders
# -----------------------------------------------------------------------------

def add_border(image: np.ndarray, border_size: int = 5) -> np.ndarray:
    if image.ndim == 2:
        med = float(np.median(image))
        return np.pad(
            image,
            ((border_size, border_size), (border_size, border_size)),
            mode="constant",
            constant_values=med,
        )
    if image.ndim == 3 and image.shape[2] == 3:
        meds = np.median(image, axis=(0, 1)).astype(np.float32)
        chans = []
        for c in range(3):
            chans.append(
                np.pad(
                    image[..., c],
                    ((border_size, border_size), (border_size, border_size)),
                    mode="constant",
                    constant_values=float(meds[c]),
                )
            )
        return np.stack(chans, axis=-1)
    raise ValueError("add_border expects 2D or HxWx3")


def remove_border(image: np.ndarray, border_size: int = 5) -> np.ndarray:
    if image.ndim == 2:
        return image[border_size:-border_size, border_size:-border_size]
    return image[border_size:-border_size, border_size:-border_size, :]


def stretch_image_mono(img: np.ndarray, target_median: float = 0.25):
    x = img.astype(np.float32, copy=True)
    orig_min = float(np.min(x))
    x = x - orig_min
    orig_med = float(np.median(x))

    if orig_med != 0:
        x = ((orig_med - 1.0) * target_median * x) / (
            orig_med * (target_median + x - 1.0) - target_median * x
        )

    x = np.clip(x, 0, 1)
    return x.astype(np.float32, copy=False), np.float32(orig_min), np.float32(orig_med)


def unstretch_image_mono(img: np.ndarray, orig_med, orig_min):
    x = img.astype(np.float32, copy=True)
    m_now = float(np.median(x))
    m0 = float(orig_med)

    if m_now != 0 and m0 != 0:
        x = ((m_now - 1.0) * m0 * x) / (m_now * (m0 + x - 1.0) - m0 * x)

    x = x + float(orig_min)
    return np.clip(x, 0, 1).astype(np.float32, copy=False)


def stretch_image_unlinked_rgb(img_rgb: np.ndarray, target_median: float = 0.25):
    x = img_rgb.astype(np.float32, copy=True)
    orig_min = x.reshape(-1, 3).min(axis=0)
    x = x - orig_min.reshape(1, 1, 3)
    orig_meds = np.median(x, axis=(0, 1)).astype(np.float32)

    for c in range(3):
        m = float(orig_meds[c])
        if m != 0:
            x[..., c] = ((m - 1) * target_median * x[..., c]) / (
                m * (target_median + x[..., c] - 1) - target_median * x[..., c]
            )

    x = np.clip(x, 0, 1)
    return x.astype(np.float32, copy=False), orig_min.astype(np.float32), orig_meds.astype(np.float32)


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

# -----------------------------------------------------------------------------
# Luma helpers
# -----------------------------------------------------------------------------

_LUMA_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

def _compute_luminance_rec709(img_rgb: np.ndarray) -> np.ndarray:
    img_rgb = np.asarray(img_rgb, np.float32)
    return np.clip(
        img_rgb[..., 0] * _LUMA_REC709[0]
        + img_rgb[..., 1] * _LUMA_REC709[1]
        + img_rgb[..., 2] * _LUMA_REC709[2],
        0.0,
        1.0,
    ).astype(np.float32, copy=False)

def _recombine_luminance_linear_scale(
    target_rgb: np.ndarray,
    new_luma: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    target_rgb = np.asarray(target_rgb, np.float32)
    new_luma = np.asarray(new_luma, np.float32)

    Y = _compute_luminance_rec709(target_rgb)
    scale = new_luma / np.maximum(Y, eps)
    out = target_rgb * scale[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

# -----------------------------------------------------------------------------
# Chunking / stitching
# -----------------------------------------------------------------------------

def split_image_into_chunks_with_overlap(image: np.ndarray, chunk_size: int, overlap: int):
    H, W = image.shape[:2]
    step = max(1, chunk_size - overlap)
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
    chunks,
    out_shape,
    *,
    chunk_size: int,
    overlap: int,
    border_size: int = 5,
):
    if len(out_shape) == 2:
        H, W = out_shape
        C = 1
        is_mono = True
    else:
        H, W, C = out_shape
        is_mono = False

    out = np.zeros((H, W, C), np.float32)
    wsum = np.zeros((H, W, 1), np.float32)
    bw_full = _blend_weights(chunk_size, overlap)

    for tile, i, j in chunks:
        if tile.ndim == 2:
            tile3 = tile[..., None]
        else:
            tile3 = tile

        th, tw = tile3.shape[:2]

        top = 0 if i == 0 else min(border_size, th // 2)
        left = 0 if j == 0 else min(border_size, tw // 2)
        bottom = 0 if (i + th) >= H else min(border_size, th // 2)
        right = 0 if (j + tw) >= W else min(border_size, tw // 2)

        inner = tile3[top:th - bottom, left:tw - right, :]
        ih, iw = inner.shape[:2]

        rr0 = i + top
        cc0 = j + left
        rr1 = rr0 + ih
        cc1 = cc0 + iw

        bw = bw_full[:ih, :iw].reshape(ih, iw, 1)
        out[rr0:rr1, cc0:cc1, :] += inner * bw
        wsum[rr0:rr1, cc0:cc1, :] += bw

    out = out / np.maximum(wsum, 1e-8)

    if is_mono:
        return out[..., 0]
    return out


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

@dataclass
class DarkStarModels:
    device: Any
    is_onnx: bool
    mono_model: Any | None = None
    color_model: Any | None = None
    torch: Any | None = None
    chunk_size: int = 512


_MODELS_CACHE: dict[tuple[str, str], DarkStarModels] = {}


def load_darkstar_models(*, use_gpu: bool, status_cb=print) -> DarkStarModels:
    """
    Runtime version:
      - mono_model: robust mono-trained-triplicated RGB NAFNet
      - color_model: RGB-trained color NAFNet (optional fallback if file missing)
      - torch backends only:
            CUDA
            MPS
            DirectML (torch-directml)
            CPU
      - no ONNX / ORT fallback
    """
    mono_path, mono_onnx_path, color_path, color_onnx_path = _resolve_darkstar_model_paths()

    if not os.path.exists(mono_path):
        raise FileNotFoundError(f"Dark Star mono model not found:\n{mono_path}")

    is_windows = os.name == "nt"

    torch = _get_torch(
        prefer_cuda=bool(use_gpu),
        prefer_dml=bool(use_gpu and is_windows),
        status_cb=status_cb,
    )

    def _load_one_model(dev, path: str, label: str):
        Net = _build_darkstar_nafnet_model(torch)
        net = Net(
            in_ch=3,
            out_ch=3,
            width=32,
            enc_blk_nums=(2, 4, 6, 8),
            dec_blk_nums=(2, 2, 2, 2),
            middle_blk_num=4,
        )

        ckpt = torch.load(path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        sd = _strip_common_prefixes(sd)
        net.load_state_dict(sd, strict=True)
        net = net.eval().to(dev)

        status_cb(f"Dark Star: loaded {label} model from {os.path.basename(path)}")
        return net

    def _build_and_load(dev, backend_id: str, backend_msg: str):
        key = ("darkstar_dual", backend_id)
        if key in _MODELS_CACHE:
            return _MODELS_CACHE[key]

        status_cb(backend_msg)

        mono_model = _load_one_model(dev, mono_path, "mono-triplicated")

        color_model = None
        if os.path.exists(color_path):
            try:
                color_model = _load_one_model(dev, color_path, "color")
            except Exception as e:
                status_cb(f"Dark Star: failed to load color model, will fall back to mono-per-channel. {e}")
                color_model = None
        else:
            status_cb("Dark Star: color model not found, color/hybrid modes will fall back to mono-per-channel.")

        m = DarkStarModels(
            device=dev,
            is_onnx=False,
            mono_model=mono_model,
            color_model=color_model,
            torch=torch,
            chunk_size=512,
        )
        _MODELS_CACHE[key] = m
        return m

    # CUDA
    if use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
        dev = torch.device("cuda")
        return _build_and_load(dev, "cuda", f"Dark Star: using CUDA ({torch.cuda.get_device_name(0)})")

    # MPS
    if use_gpu and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        return _build_and_load(dev, "mps", "Dark Star: using MPS")

    # DirectML torch
    if use_gpu and is_windows:
        try:
            import torch_directml
            dev = torch_directml.device()
            return _build_and_load(dev, "torch_dml", "Dark Star: using DirectML (torch-directml)")
        except Exception:
            pass

    # CPU
    dev = torch.device("cpu")
    return _build_and_load(dev, "cpu", "Dark Star: using CPU")

# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------
def _infer_tile_rgb_with_model(models: DarkStarModels, tile_rgb: np.ndarray, model) -> np.ndarray:
    tile_rgb = np.asarray(tile_rgb, np.float32)
    h0, w0 = tile_rgb.shape[:2]

    torch = models.torch
    dev = models.device

    t = torch.from_numpy(tile_rgb.transpose(2, 0, 1)[None, ...]).to(dev)

    with torch.no_grad(), _autocast_context(torch, dev):
        y = model(t)[0].detach().float().cpu().numpy()

    y = np.transpose(y, (1, 2, 0))
    return y[:h0, :w0, :].astype(np.float32, copy=False)

def _run_rgb_chunked_with_model(
    img_rgb: np.ndarray,
    *,
    model,
    models: DarkStarModels,
    params,
    progress_cb: ProgressCB,
    progress_start: int,
    progress_total: int,
    stage_name: str,
) -> tuple[np.ndarray, int]:
    bordered = add_border(img_rgb, border_size=5)

    chunk_size = int(params.chunk_size)
    overlap = int(round(float(params.overlap_frac) * chunk_size))
    chunks = split_image_into_chunks_with_overlap(bordered, chunk_size=chunk_size, overlap=overlap)

    out_tiles = []
    done = progress_start
    for tile, i, j in chunks:
        out = _infer_tile_rgb_with_model(models, tile, model)
        out_tiles.append((out, i, j))
        done += 1
        progress_cb(done, progress_total, stage_name)

    starless_b = stitch_chunks_soft_blend(
        out_tiles,
        bordered.shape,
        chunk_size=chunk_size,
        overlap=overlap,
        border_size=5,
    )

    starless = remove_border(starless_b, border_size=5)
    starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)
    return starless, done

def _compute_stars_only(original: np.ndarray, starless: np.ndarray, mode: str) -> np.ndarray:
    if mode == "additive":
        return np.clip(original - starless, 0.0, 1.0).astype(np.float32, copy=False)

    denom = np.maximum(1.0 - starless, 1e-6)
    return np.clip((original - starless) / denom, 0.0, 1.0).astype(np.float32, copy=False)

def _run_channel_chunked(
    ch: np.ndarray,
    *,
    models: DarkStarModels,
    params,
    progress_cb: ProgressCB,
    progress_start: int,
    progress_total: int,
    stage_name: str,
) -> tuple[np.ndarray, int]:
    """
    Run a single mono channel through the mono-trained-triplicated Dark Star model:
      mono tile -> triplicate to RGB -> infer with mono_model -> collapse back to mono
    """
    ch = np.asarray(ch, np.float32)
    bordered = add_border(ch, border_size=5)

    chunk_size = int(models.chunk_size) if models.is_onnx else int(params.chunk_size)
    overlap = int(round(float(params.overlap_frac) * chunk_size))

    chunks = split_image_into_chunks_with_overlap(
        bordered,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    out_tiles = []
    done = progress_start

    for tile, i, j in chunks:
        tile = np.asarray(tile, np.float32)

        tile_rgb = np.stack([tile, tile, tile], axis=-1)
        out_rgb = _infer_tile_rgb_with_model(models, tile_rgb, models.mono_model)
        out_mono = out_rgb[..., 0].astype(np.float32, copy=False)

        out_tiles.append((out_mono, i, j))
        done += 1
        progress_cb(done, progress_total, stage_name)

    starless_b = stitch_chunks_soft_blend(
        out_tiles,
        bordered.shape,
        chunk_size=chunk_size,
        overlap=overlap,
        border_size=5,
    )

    starless = remove_border(starless_b, border_size=5)
    starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)
    return starless, done

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

@dataclass
class DarkStarParams:
    use_gpu: bool = True
    chunk_size: int = 512
    overlap_frac: float = 0.125
    mode: str = "unscreen"              # "unscreen" or "additive"
    output_stars_only: bool = False
    processing_path: str = "hybrid_luma_color"  # "mono_per_channel" | "hybrid_luma_color" | "color_only"

def darkstar_starremoval_rgb01(
    img_rgb01: np.ndarray,
    *,
    params: DarkStarParams,
    progress_cb: Optional[ProgressCB] = None,
    status_cb=print,
) -> tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    Input : float32 image in [0..1], shape HxW, HxWx1, or HxWx3
    Output: (starless_rgb01, stars_only_rgb01 or None, was_mono)

    Correct testing behavior for the current Dark Star checkpoint:
      - the checkpoint is a 3-channel NAFNet
      - mono data was trained by triplicating the same mono plane into RGB
      - mono images: run once by triplicating mono -> RGB -> infer -> collapse back to mono
      - color images: split R/G/B, and for each channel:
            mono channel -> triplicate to RGB -> infer -> collapse back to mono
    """
    if progress_cb is None:
        progress_cb = lambda done, total, stage: None

    img = np.asarray(img_rgb01, np.float32)
    was_mono = (img.ndim == 2) or (img.ndim == 3 and img.shape[2] == 1)

    models = load_darkstar_models(use_gpu=params.use_gpu, status_cb=status_cb)

    # -------------------------------------------------------------------------
    # Case 1: pure 2D mono
    # -------------------------------------------------------------------------
    if img.ndim == 2:
        mono = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

        stretch_needed = float(np.median(mono - float(np.min(mono)))) < 0.125
        if stretch_needed:
            stretched, orig_min, orig_med = stretch_image_mono(mono)
        else:
            stretched, orig_min, orig_med = mono, None, None

        # total chunk count for one mono pass
        chunk_size = int(params.chunk_size)
        overlap = int(round(float(params.overlap_frac) * chunk_size))
        total = len(
            split_image_into_chunks_with_overlap(
                add_border(stretched, border_size=5),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        # _run_channel_chunked is expected to take a 2D channel, triplicate it
        # to RGB internally for the RGB-trained NAFNet, then collapse back to mono.
        starless, _done = _run_channel_chunked(
            stretched,
            models=models,
            params=params,
            progress_cb=progress_cb,
            progress_start=0,
            progress_total=total,
            stage_name="Dark Star removal",
        )

        if stretch_needed:
            starless = unstretch_image_mono(starless, orig_med, orig_min)

        starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)

        stars_only = None
        if params.output_stars_only:
            if params.mode == "additive":
                stars_only = np.clip(mono - starless, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                denom = np.maximum(1.0 - starless, 1e-6)
                stars_only = np.clip((mono - starless) / denom, 0.0, 1.0).astype(np.float32, copy=False)

            stars_only = stars_only[..., None]

        return starless[..., None], stars_only, True

    # -------------------------------------------------------------------------
    # Case 2: HxWx1 mono
    # -------------------------------------------------------------------------
    if img.ndim == 3 and img.shape[2] == 1:
        mono = np.clip(img[..., 0], 0.0, 1.0).astype(np.float32, copy=False)

        stretch_needed = float(np.median(mono - float(np.min(mono)))) < 0.125
        if stretch_needed:
            stretched, orig_min, orig_med = stretch_image_mono(mono)
        else:
            stretched, orig_min, orig_med = mono, None, None

        chunk_size = int(params.chunk_size)
        overlap = int(round(float(params.overlap_frac) * chunk_size))
        total = len(
            split_image_into_chunks_with_overlap(
                add_border(stretched, border_size=5),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        starless, _done = _run_channel_chunked(
            stretched,
            models=models,
            params=params,
            progress_cb=progress_cb,
            progress_start=0,
            progress_total=total,
            stage_name="Dark Star removal",
        )

        if stretch_needed:
            starless = unstretch_image_mono(starless, orig_med, orig_min)

        starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)

        stars_only = None
        if params.output_stars_only:
            if params.mode == "additive":
                stars_only = np.clip(mono - starless, 0.0, 1.0).astype(np.float32, copy=False)
            else:
                denom = np.maximum(1.0 - starless, 1e-6)
                stars_only = np.clip((mono - starless) / denom, 0.0, 1.0).astype(np.float32, copy=False)

            stars_only = stars_only[..., None]

        return starless[..., None], stars_only, True

    # -------------------------------------------------------------------------
    # Case 3: HxWx3 color
    # -------------------------------------------------------------------------
    img3 = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)

    stretch_needed = float(np.median(img3 - float(np.min(img3)))) < 0.125
    if stretch_needed:
        stretched, orig_min, orig_meds = stretch_image_unlinked_rgb(img3)
    else:
        stretched, orig_min, orig_meds = img3, None, None

    chunk_size = int(params.chunk_size)
    overlap = int(round(float(params.overlap_frac) * chunk_size))

    path = str(getattr(params, "processing_path", "hybrid_luma_color")).strip().lower()
    if path not in ("mono_per_channel", "hybrid_luma_color", "color_only"):
        path = "hybrid_luma_color"

    if path in ("color_only", "hybrid_luma_color") and models.color_model is None:
        status_cb("Dark Star: color model unavailable, falling back to mono_per_channel")
        path = "mono_per_channel"

    # ------------------------------------------------------------------
    # mono_per_channel
    # ------------------------------------------------------------------
    if path == "mono_per_channel":
        n_chunks = len(
            split_image_into_chunks_with_overlap(
                add_border(stretched[..., 0], border_size=5),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
        total = n_chunks * 3

        out = np.zeros_like(stretched, dtype=np.float32)
        done = 0
        channel_names = ["R", "G", "B"]

        for c in range(3):
            out[..., c], done = _run_channel_chunked(
                stretched[..., c],
                models=models,
                params=params,
                progress_cb=progress_cb,
                progress_start=done,
                progress_total=total,
                stage_name=f"Dark Star removal ({channel_names[c]})",
            )

        starless = out

    # ------------------------------------------------------------------
    # color_only
    # ------------------------------------------------------------------
    elif path == "color_only":
        total = len(
            split_image_into_chunks_with_overlap(
                add_border(stretched, border_size=5),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        starless, _done = _run_rgb_chunked_with_model(
            stretched,
            model=models.color_model,
            models=models,
            params=params,
            progress_cb=progress_cb,
            progress_start=0,
            progress_total=total,
            stage_name="Dark Star color model",
        )

    # ------------------------------------------------------------------
    # hybrid_luma_color
    # ------------------------------------------------------------------
    else:
        n_chunks_rgb = len(
            split_image_into_chunks_with_overlap(
                add_border(stretched, border_size=5),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
        n_chunks_l = len(
            split_image_into_chunks_with_overlap(
                add_border(_compute_luminance_rec709(stretched), border_size=5),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
        total = n_chunks_l + n_chunks_rgb

        # robust mono luminance
        lum = _compute_luminance_rec709(stretched)
        starless_lum, done = _run_channel_chunked(
            lum,
            models=models,
            params=params,
            progress_cb=progress_cb,
            progress_start=0,
            progress_total=total,
            stage_name="Dark Star luminance",
        )

        # color model RGB
        starless_color, done = _run_rgb_chunked_with_model(
            stretched,
            model=models.color_model,
            models=models,
            params=params,
            progress_cb=progress_cb,
            progress_start=done,
            progress_total=total,
            stage_name="Dark Star color model",
        )

        # recombine robust mono luminance into color-model chroma
        starless = _recombine_luminance_linear_scale(starless_color, starless_lum)

    if stretch_needed:
        starless = unstretch_image_unlinked_rgb(starless, orig_meds, orig_min)

    starless = np.clip(starless, 0.0, 1.0).astype(np.float32, copy=False)

    stars_only = None
    if params.output_stars_only:
        stars_only = _compute_stars_only(img3, starless, params.mode)

    return starless, stars_only, False