# src/setiastro/saspro/cosmicclarity_engines/denoise_engine.py
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

import cv2


from setiastro.saspro.resources import get_resources
from setiastro.saspro.runtime_torch import _user_runtime_dir, _venv_paths, _check_cuda_in_venv

warnings.filterwarnings("ignore")

from typing import Callable

ProgressCB = Callable[[int, int], None]  # (done, total)


def _get_torch(*, prefer_cuda: bool, prefer_dml: bool, status_cb=print):
    from setiastro.saspro.runtime_torch import import_torch
    return import_torch(
        prefer_cuda=prefer_cuda,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=status_cb,
    )

def _get_ort(status_cb=print):
    """
    Import onnxruntime AFTER runtime_torch has added runtime site-packages to sys.path.
    """
    try:
        import onnxruntime as ort  # type: ignore
        return ort
    except Exception as e:
        try:
            status_cb(f"CosmicClarity Denoise: onnxruntime not available ({type(e).__name__}: {e})")
        except Exception:
            pass
        return None


def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()


def _autocast_context(torch, device) -> Any:
    """
    Use new torch.amp.autocast('cuda') when available.
    Keep your cap >= 8.0 rule.
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
            # MPS often benefits from autocast in newer torch versions
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast(device_type="mps")

    except Exception:
        pass

    return _nullcontext()


# ============================
# DROP-IN REPLACEMENT METHODS
# for: src/setiastro/saspro/cosmicclarity_engines/denoise_engine.py
# (NAFNet-style model bundle + backend resolution + padded infer like sharpen_engine)
# ============================

# ---- NEW: match sharpen_engine-style NAFNet configs ----
_DENOISE_NAFNET_FULL = dict(
    width=64,
    enc_blk_nums=(2, 4, 6, 8),
    dec_blk_nums=(2, 2, 2, 2),
    middle_blk_num=4,
)

_DENOISE_NAFNET_LITE = dict(
    width=32,
    enc_blk_nums=(2, 4, 6, 8),
    dec_blk_nums=(2, 2, 2, 2),
    middle_blk_num=4,
)

# ---- NEW: dataclass bundle like sharpen ----
@dataclass
class DenoiseModels:
    device: Any
    is_onnx: bool
    mono: Any
    color: Any
    torch: Any | None = None
    variant: str = "full"         # "full" or "lite"
    mono_path: str = ""
    color_path: str = ""

# Cache by (backend_tag, resolved_backend)
_MODELS_CACHE: dict[tuple[str, str], DenoiseModels] = {}
_BACKEND_TAG = "cc_denoise_ai4_nafnet"


def _pad2d_to_multiple(x: np.ndarray, mult: int = 16, mode: str = "reflect") -> tuple[np.ndarray, int, int]:
    """Pad 2D array on bottom/right so H,W are multiples of `mult`."""
    h, w = x.shape
    ph = (mult - (h % mult)) % mult
    pw = (mult - (w % mult)) % mult
    if ph == 0 and pw == 0:
        return x, h, w
    xp = np.pad(x, ((0, ph), (0, pw)), mode=mode)
    return xp, h, w


def _ort_pick_io_names_single_input(session) -> tuple[str, str]:
    """
    Returns: (img_name, out_name) for a 1-input denoise model.
    If exporter ever produces extra inputs, we still pick the rank-4 input as image.
    """
    ins = session.get_inputs()
    out = session.get_outputs()[0].name

    img_name = None
    for i in ins:
        shp = i.shape
        rank = len(shp) if shp is not None else 0
        if rank == 4:
            img_name = i.name
            break

    if img_name is None:
        img_name = ins[0].name

    return img_name, out


def _load_onnx_models(ort, *, lite: bool) -> DenoiseModels:
    prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
    R = get_resources()

    def s(path: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return ort.InferenceSession(path, sess_options=so, providers=prov)

    if lite:
        mono_path  = R.CC_DENOISE_MONO_ONNX_LITE
        color_path = R.CC_DENOISE_COLOR_ONNX_LITE
    else:
        mono_path  = R.CC_DENOISE_MONO_ONNX
        color_path = R.CC_DENOISE_COLOR_ONNX

    mono_sess  = s(mono_path)
    color_sess = s(color_path)

    return DenoiseModels(
        device="DirectML",
        is_onnx=True,
        mono=mono_sess,
        color=color_sess,
        torch=None,
        variant=("lite" if lite else "full"),
        mono_path=str(mono_path),
        color_path=str(color_path),
    )




def _load_torch_models(torch, device, *, lite: bool) -> DenoiseModels:
    import torch.nn as nn

    # ---- NAFNet blocks (same style as sharpen_engine) ----
    class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(dim=1, keepdim=True)
            var  = (x - mean).pow(2).mean(dim=1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return x * self.weight + self.bias

    class SimpleGate(nn.Module):
        def forward(self, x):
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class NAFBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm1 = LayerNorm2d(channels)
            self.conv1 = nn.Conv2d(channels, channels * 2, 1, bias=True)
            self.dwconv = nn.Conv2d(
                channels * 2, channels * 2, 3,
                padding=1, groups=channels * 2, bias=True
            )
            self.sg = SimpleGate()
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, 1, bias=True),
            )
            self.conv2 = nn.Conv2d(channels, channels, 1, bias=True)

            self.norm2 = LayerNorm2d(channels)
            self.ffn1 = nn.Conv2d(channels, channels * 2, 1, bias=True)
            self.ffn2 = nn.Conv2d(channels, channels, 1, bias=True)

            self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
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

    class NAFNetDenoise(nn.Module):
        """
        RGB->RGB denoise. Uses residual-out (y = x + delta) by default like sharpen.
        (If your training exported absolute output, set residual_out=False.)
        """
        def __init__(
            self,
            in_ch=3, out_ch=3, width=32,
            enc_blk_nums=(2, 4, 6, 8),
            dec_blk_nums=(2, 2, 2, 2),
            middle_blk_num=4,
            residual_out=True,
            clamp_out=False,
        ):
            super().__init__()
            self.intro = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)

            self.encoders = nn.ModuleList()
            self.downs    = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.ups      = nn.ModuleList()

            ch = width
            for n in enc_blk_nums:
                self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
                self.downs.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=True))
                ch *= 2

            self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blk_num)])

            for n in dec_blk_nums:
                self.ups.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 1, bias=True), nn.PixelShuffle(2)))
                ch //= 2
                self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

            self.ending = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)

            self.residual_out = bool(residual_out)
            self.clamp_out = bool(clamp_out)

        def forward_delta(self, x):
            x = self.intro(x)
            skips = []
            for enc, down in zip(self.encoders, self.downs):
                x = enc(x); skips.append(x); x = down(x)
            x = self.middle(x)
            for up, dec in zip(self.ups, self.decoders):
                x = up(x); x = x + skips.pop(); x = dec(x)
            return self.ending(x)

        def forward(self, x):
            delta = self.forward_delta(x)
            y = x + delta if self.residual_out else delta
            if self.clamp_out:
                y = y.clamp(0.0, 1.0)
            return y

    R = get_resources()

    cfg = _DENOISE_NAFNET_LITE if lite else _DENOISE_NAFNET_FULL

    def m_naf_rgb(path: str, cfg: dict):
        net = NAFNetDenoise(**cfg, residual_out=True, clamp_out=False)
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        net.load_state_dict(sd)
        net.eval()
        return net.to(device)

    if lite:
        mono_path  = getattr(R, "CC_DENOISE_MONO_PTH_LITE", None)
        color_path = getattr(R, "CC_DENOISE_COLOR_PTH_LITE", None)
    else:
        mono_path  = getattr(R, "CC_DENOISE_MONO_PTH", None)
        color_path = getattr(R, "CC_DENOISE_COLOR_PTH", None)

    if not mono_path or not os.path.exists(mono_path):
        raise RuntimeError("Denoise: MONO model not found for selected variant.")
    if not color_path or not os.path.exists(color_path):
        raise RuntimeError("Denoise: COLOR model not found for selected variant.")

    mono  = m_naf_rgb(mono_path,  cfg)
    color = m_naf_rgb(color_path, cfg)

    return DenoiseModels(
        device=device,
        is_onnx=False,
        mono=mono,
        color=color,
        torch=torch,
        variant=("lite" if lite else "full"),
        mono_path=str(mono_path),
        color_path=str(color_path),
    )


def load_models(use_gpu: bool = True, *, lite: bool = False, status_cb=print) -> DenoiseModels:
    backend_tag = _BACKEND_TAG + ("_lite" if lite else "_full")
    is_windows = (os.name == "nt")

    torch = _get_torch(
        prefer_cuda=bool(use_gpu),
        prefer_dml=bool(use_gpu and is_windows),
        status_cb=status_cb,
    )
    ort = _get_ort(status_cb=status_cb)

    # (CUDA probe unchanged)

    # 1) CUDA
    if use_gpu:
        try:
            cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except Exception:
            cuda_ok = False

        if cuda_ok:
            cache_key = (backend_tag, "cuda")
            if cache_key in _MODELS_CACHE:
                return _MODELS_CACHE[cache_key]

            device = torch.device("cuda")
            models = _load_torch_models(torch, device, lite=lite)
            _MODELS_CACHE[cache_key] = models
            return models

    # 2) MPS
    if use_gpu:
        try:
            mps_ok = bool(hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        except Exception:
            mps_ok = False

        if mps_ok:
            cache_key = (backend_tag, "mps")
            if cache_key in _MODELS_CACHE:
                return _MODELS_CACHE[cache_key]

            device = torch.device("mps")
            models = _load_torch_models(torch, device, lite=lite)
            _MODELS_CACHE[cache_key] = models
            return models

    # 3) Torch-DirectML
    if use_gpu and is_windows:
        try:
            import torch_directml
            cache_key = (backend_tag, "dml_torch")
            if cache_key in _MODELS_CACHE:
                return _MODELS_CACHE[cache_key]

            dml = torch_directml.device()
            _ = (torch.ones(1, device=dml) + 1).to("cpu").item()

            models = _load_torch_models(torch, dml, lite=lite)

            # smoke test (unchanged)
            _MODELS_CACHE[cache_key] = models
            return models

        except Exception as e:
            status_cb(f"CosmicClarity Denoise: DirectML (torch-directml) failed, falling back. {type(e).__name__}: {e}")

    # 4) ORT DirectML
    if use_gpu and ort is not None:
        try:
            prov = ort.get_available_providers()
        except Exception:
            prov = []

        if "DmlExecutionProvider" in prov:
            cache_key = (backend_tag, "dml_ort")
            if cache_key in _MODELS_CACHE:
                return _MODELS_CACHE[cache_key]

            models = _load_onnx_models(ort, lite=lite)
            _MODELS_CACHE[cache_key] = models
            return models

    # 5) CPU
    cache_key = (backend_tag, "cpu")
    if cache_key in _MODELS_CACHE:
        return _MODELS_CACHE[cache_key]

    device = torch.device("cpu")
    models = _load_torch_models(torch, device, lite=lite)
    _MODELS_CACHE[cache_key] = models
    return models

def _infer_chunk_rgb(models: DenoiseModels, model: Any, chunk_rgb: np.ndarray) -> np.ndarray:
    chunk_rgb = np.asarray(chunk_rgb, np.float32)  # HxWx3
    h0, w0 = chunk_rgb.shape[:2]

    # pad each channel together
    ph = (16 - (h0 % 16)) % 16
    pw = (16 - (w0 % 16)) % 16
    if ph or pw:
        chunk_p = np.pad(chunk_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    else:
        chunk_p = chunk_rgb

    if models.is_onnx:
        inp = np.transpose(chunk_p, (2, 0, 1))[None, ...].astype(np.float32)  # (1,3,Hp,Wp)
        name_img, name_out = _ort_pick_io_names_single_input(model)
        out = model.run([name_out], {name_img: inp})[0]  # expect (1,3,Hp,Wp) or similar
        if out.ndim == 4:
            y = out[0]
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
        y = np.transpose(y, (1, 2, 0))  # Hp,Wp,3
        return y[:h0, :w0].astype(np.float32, copy=False)

    torch = models.torch
    dev = models.device
    t = torch.tensor(np.transpose(chunk_p, (2,0,1)), dtype=torch.float32)[None, ...].to(dev)  # (1,3,Hp,Wp)

    with torch.no_grad(), _autocast_context(torch, dev):
        y = model(t)[0].detach().cpu().numpy()  # (3,Hp,Wp)

    y = np.transpose(y, (1,2,0))
    return y[:h0, :w0].astype(np.float32, copy=False)


def _infer_chunk_2d(models: DenoiseModels, model: Any, chunk2d: np.ndarray) -> np.ndarray:
    """
    NAFNet-friendly infer:
      - pad chunk to mult-of-16 (reflect)
      - make (1,3,H,W)
      - run model (torch or onnx)
      - return 2D (cropped back to original h,w), float32
    """
    chunk2d = np.asarray(chunk2d, np.float32)
    chunk_p, h0, w0 = _pad2d_to_multiple(chunk2d, mult=16, mode="reflect")

    if models.is_onnx:
        inp = chunk_p[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,Hp,Wp)
        inp = np.tile(inp, (1, 3, 1, 1))                                # (1,3,Hp,Wp)

        name_img, name_out = _ort_pick_io_names_single_input(model)
        out = model.run([name_out], {name_img: inp})[0]

        # normalize output handling
        if out.ndim == 4:
            y = out[0, 0]
        elif out.ndim == 3:
            y = out[0]
            if y.shape[0] in (1, 3):
                y = y[0]
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")

        return y[:h0, :w0].astype(np.float32, copy=False)

    # torch
    torch = models.torch
    dev = models.device

    t_cpu = torch.tensor(chunk_p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,Hp,Wp)
    t_rgb = t_cpu.repeat(1, 3, 1, 1).to(dev)                                      # (1,3,Hp,Wp)

    with torch.no_grad(), _autocast_context(torch, dev):
        y = model(t_rgb)            # (1,3,Hp,Wp)
        y = y[0, 0].detach().cpu().numpy()

    return y[:h0, :w0].astype(np.float32, copy=False)



# ----------------------------
# Your helpers: luminance/chroma, chunks, borders, stretch
# (paste your existing implementations here)
# ----------------------------
def extract_luminance(image: np.ndarray):
    """
    Input: mono HxW, mono HxWx1, or RGB HxWx3 float32 in [0,1].
    Output: (Y, Cb, Cr) where:
      - Y is HxW
      - Cb/Cr are HxW in [0,1] (with +0.5 offset already applied)
    """
    x = np.asarray(image, dtype=np.float32)

    # Ensure 3-channel
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("extract_luminance expects HxW, HxWx1, or HxWx3")

    # RGB -> YCbCr (BT.601) (same numbers as your sharpen_engine)
    M = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]], dtype=np.float32)

    ycbcr = x @ M.T
    y  = ycbcr[..., 0]
    cb = ycbcr[..., 1] + 0.5
    cr = ycbcr[..., 2] + 0.5
    return y, cb, cr

def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    y = np.asarray(y, np.float32)
    cb = np.asarray(cb, np.float32) - 0.5
    cr = np.asarray(cr, np.float32) - 0.5
    ycbcr = np.stack([y, cb, cr], axis=-1)

    M = np.array([[1.0, 0.0, 1.402],
                  [1.0, -0.344136, -0.714136],
                  [1.0, 1.772, 0.0]], dtype=np.float32)

    rgb = ycbcr @ M.T
    return np.clip(rgb, 0.0, 1.0)


def merge_luminance(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    return ycbcr_to_rgb(np.clip(y, 0, 1), np.clip(cb, 0, 1), np.clip(cr, 0, 1))


def denoise_rgb_with_color_model(img_rgb, models, *, chunk_size=256, overlap=64, progress_cb=None):
    img_rgb = np.asarray(img_rgb, np.float32)
    chunks = split_image_into_chunks_with_overlap(img_rgb, chunk_size=chunk_size, overlap=overlap)

    out = np.zeros_like(img_rgb, dtype=np.float32)
    wts = np.zeros(img_rgb.shape[:2], dtype=np.float32)

    total = len(chunks)
    for idx, (chunk, i, j) in enumerate(chunks):
        den = _infer_chunk_rgb(models, models.color, chunk)  # HxWx3

        h, w = den.shape[:2]
        bh = min(16, h // 2)
        bw = min(16, w // 2)

        y0 = i + bh; y1 = i + h - bh
        x0 = j + bw; x1 = j + w - bw
        if y1 <= y0 or x1 <= x0:
            continue

        inner = den[bh:h-bh, bw:w-bw, :]  # (hi, wi, 3)

        yy0 = max(0, y0); yy1 = min(img_rgb.shape[0], y1)
        xx0 = max(0, x0); xx1 = min(img_rgb.shape[1], x1)
        if yy1 <= yy0 or xx1 <= xx0:
            continue

        sy0 = yy0 - y0; sy1 = sy0 + (yy1 - yy0)
        sx0 = xx0 - x0; sx1 = sx0 + (xx1 - xx0)
        src = inner[sy0:sy1, sx0:sx1, :]

        out[yy0:yy1, xx0:xx1, :] += src
        wts[yy0:yy1, xx0:xx1] += 1.0

        if progress_cb is not None:
            try:
                cont = progress_cb(idx + 1, total)
                if cont is False:
                    raise RuntimeError("Cancelled.")
            except Exception:
                pass

    out /= np.maximum(wts[..., None], 1.0)
    return out


# Function to split an image into chunks with overlap
def split_image_into_chunks_with_overlap(image, chunk_size, overlap):
    height, width = image.shape[:2]
    chunks = []
    step_size = chunk_size - overlap

    for i in range(0, height, step_size):
        for j in range(0, width, step_size):
            end_i = min(i + chunk_size, height)
            end_j = min(j + chunk_size, width)
            if end_i <= i or end_j <= j:
                continue
            chunk = image[i:end_i, j:end_j]
            chunks.append((chunk, i, j))
    return chunks


def blend_images(before, after, amount):
    return (1 - amount) * before + amount * after

def stitch_chunks_ignore_border(chunks, image_shape, border_size: int = 16):
    """
    chunks: list of (chunk, i, j) or (chunk, i, j, is_edge)
    image_shape: (H,W)
    Robust to boundary clipping (prevents 256x256 -> 256x0 broadcasts).
    """
    H, W = image_shape
    stitched = np.zeros((H, W), dtype=np.float32)
    weights  = np.zeros((H, W), dtype=np.float32)

    for entry in chunks:
        if len(entry) == 3:
            chunk, i, j = entry
        else:
            chunk, i, j, _ = entry

        h, w = chunk.shape[:2]
        if h <= 0 or w <= 0:
            continue

        bh = min(border_size, h // 2)
        bw = min(border_size, w // 2)

        # inner region in chunk coords
        y0 = i + bh
        y1 = i + h - bh
        x0 = j + bw
        x1 = j + w - bw

        if y1 <= y0 or x1 <= x0:
            continue

        inner = chunk[bh:h-bh, bw:w-bw]

        # clip destination to image bounds
        yy0 = max(0, y0)
        yy1 = min(H, y1)
        xx0 = max(0, x0)
        xx1 = min(W, x1)

        if yy1 <= yy0 or xx1 <= xx0:
            continue

        # clip source to match clipped destination
        sy0 = yy0 - y0
        sy1 = sy0 + (yy1 - yy0)
        sx0 = xx0 - x0
        sx1 = sx0 + (xx1 - xx0)

        src = inner[sy0:sy1, sx0:sx1]

        stitched[yy0:yy1, xx0:xx1] += src
        weights[yy0:yy1,  xx0:xx1] += 1.0

    stitched /= np.maximum(weights, 1.0)
    return stitched


def replace_border(original_image, processed_image, border_size=16):
    # Ensure the dimensions of both images match
    if original_image.shape != processed_image.shape:
        raise ValueError("Original image and processed image must have the same dimensions.")
    
    # Replace the top border
    processed_image[:border_size, :] = original_image[:border_size, :]
    
    # Replace the bottom border
    processed_image[-border_size:, :] = original_image[-border_size:, :]
    
    # Replace the left border
    processed_image[:, :border_size] = original_image[:, :border_size]
    
    # Replace the right border
    processed_image[:, -border_size:] = original_image[:, -border_size:]
    
    return processed_image

def stretch_image_unlinked(image: np.ndarray, target_median: float = 0.25):
    x = np.asarray(image, np.float32).copy()
    orig_min = float(np.min(x))
    x -= orig_min

    t = float(target_median)

    if x.ndim == 2:
        m0 = float(np.median(x))
        orig_meds = [m0]
        if m0 != 0.0:
            denom = (m0 * (t + x - 1.0) - t * x)
            # avoid divide-by-zero
            x = np.where(np.abs(denom) > 1e-12, ((m0 - 1.0) * t * x) / denom, x)
        return x, orig_min, orig_meds

    orig_meds = [float(np.median(x[..., c])) for c in range(3)]
    for c in range(3):
        m0 = float(orig_meds[c])
        if m0 != 0.0:
            denom = (m0 * (t + x[..., c] - 1.0) - t * x[..., c])
            x[..., c] = np.where(np.abs(denom) > 1e-12, ((m0 - 1.0) * t * x[..., c]) / denom, x[..., c])
    return x, orig_min, orig_meds


def unstretch_image_unlinked(image: np.ndarray, orig_meds, orig_min: float, target_median: float = 0.25):
    y = np.asarray(image, np.float32).copy()
    t = float(target_median)

    def inv(yc: np.ndarray, m0: float) -> np.ndarray:
        # x = y*m0*(t-1) / ( t*(m0 - 1 + y) - y*m0 )
        denom = (t * (m0 - 1.0 + yc) - yc * m0)
        num = (yc * m0 * (t - 1.0))
        return np.where(np.abs(denom) > 1e-12, num / denom, yc)

    if y.ndim == 2:
        m0 = float(orig_meds[0])
        if m0 != 0.0:
            y = inv(y, m0)
        y += float(orig_min)
        return y

    for c in range(3):
        m0 = float(orig_meds[c])
        if m0 != 0.0:
            y[..., c] = inv(y[..., c], m0)

    y += float(orig_min)
    return y


# Backwards-compatible names used by denoise_rgb01()
def stretch_image(image: np.ndarray):
    return stretch_image_unlinked(image)

def unstretch_image(image: np.ndarray, original_medians, original_min: float):
    return unstretch_image_unlinked(image, original_medians, original_min)

def add_border(image, border_size=16):
    if image.ndim == 2:                                # mono
        med = np.median(image)
        return np.pad(image,
                      ((border_size, border_size), (border_size, border_size)),
                      mode="constant",
                      constant_values=med)

    elif image.ndim == 3 and image.shape[2] == 3:       # RGB
        meds = np.median(image, axis=(0, 1)).astype(image.dtype)  # (3,)
        padded = [np.pad(image[..., c],
                         ((border_size, border_size), (border_size, border_size)),
                         mode="constant",
                         constant_values=float(meds[c]))
                  for c in range(3)]
        return np.stack(padded, axis=-1)
    else:
        raise ValueError("add_border expects mono or RGB image.")

def remove_border(image, border_size: int = 16):
    if image.ndim == 2:
        return image[border_size:-border_size, border_size:-border_size]
    return image[border_size:-border_size, border_size:-border_size, :]



def denoise_channel(channel, models: DenoiseModels, *, which="mono",
                    chunk_size=256, overlap=64, progress_cb=None):
    model = models.mono if which == "mono" else models.color
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=chunk_size, overlap=overlap)

    out_chunks = []
    total = len(chunks)

    for idx, (chunk, i, j) in enumerate(chunks):
        den2d = _infer_chunk_2d(models, model, chunk)
        out_chunks.append((den2d, i, j))

        if progress_cb is not None:
            try:
                cont = progress_cb(idx + 1, total)
                if cont is False:
                    raise RuntimeError("Cancelled.")
            except Exception:
                pass


    return stitch_chunks_ignore_border(out_chunks, channel.shape, border_size=16)



def denoise_rgb01(
    img_rgb01: np.ndarray,
    *,
    denoise_strength: float,
    denoise_mode: str = "luminance",
    separate_channels: bool = False,
    color_denoise_strength: Optional[float] = None,
    chunk_size: int = 256, overlap: int = 64,
    use_gpu: bool = True,
    lite: bool = False,          # <--- NEW
    progress_cb=None,
) -> np.ndarray:
    models = load_models(use_gpu=use_gpu, lite=lite)
    # --- NEW: log what we actually loaded ---
    def _log(msg: str):
        if progress_cb is not None:
            try:
                # progress_cb is (done,total)->bool; many callers ignore the numbers
                progress_cb(0, 0)  # harmless "ping" if someone uses it
            except Exception:
                pass
        try:
            print(msg)
        except Exception:
            pass

    try:
        backend = "onnx/DirectML" if models.is_onnx else "torch"
        dev = getattr(models.device, "type", None) or str(models.device)
        _log(
            f"[CC Denoise] variant={getattr(models,'variant','?')} "
            f"(lite={lite}) backend={backend} device={dev} "
            f"mono={os.path.basename(getattr(models,'mono_path',''))} "
            f"color={os.path.basename(getattr(models,'color_path',''))}"
        )
    except Exception:
        pass
    img_rgb01 = np.asarray(img_rgb01, np.float32)

    def _is_triplicated_mono(rgb: np.ndarray, eps: float = 1e-7) -> bool:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return False
        r = rgb[..., 0]; g = rgb[..., 1]; b = rgb[..., 2]
        return (np.max(np.abs(r - g)) <= eps) and (np.max(np.abs(r - b)) <= eps)

    # Detect "real mono" even if caller triplicated it
    mono_input = (img_rgb01.ndim == 2) or (img_rgb01.ndim == 3 and img_rgb01.shape[2] == 1) or _is_triplicated_mono(img_rgb01)

    if mono_input:
        # collapse to 2D
        if img_rgb01.ndim == 2:
            mono = img_rgb01
        else:
            mono = img_rgb01[..., 0]

        # --- keep your stretch logic but in 2D ---
        stretch_needed = (np.median(mono - np.min(mono)) < 0.05)
        if stretch_needed:
            mono_s, original_min, original_meds = stretch_image(mono)
        else:
            mono_s = mono.astype(np.float32, copy=False)
            original_min = float(np.min(mono))
            original_meds = [float(np.median(mono_s))]

        mono_s = add_border(mono_s, border_size=16)

        den_m = denoise_channel(mono_s, models, which="mono",
                                chunk_size=chunk_size, overlap=overlap, progress_cb=progress_cb)
        mono_out = blend_images(mono_s, den_m, denoise_strength)
        mono_out = np.clip(mono_out, 0.0, 1.0)
        if stretch_needed:
            mono_out = unstretch_image(mono_out, original_meds, original_min)
            mn = float(np.min(mono_out))
            mx = float(np.max(mono_out))
            if mn < -1e-3:
                print("UNSTRETCH produced negatives:", mn, mx)            

        mono_out = remove_border(mono_out, border_size=16)
        mono_out = np.clip(mono_out, 0.0, 1.0).astype(np.float32, copy=False)

        # replicate back to RGB for downstream pipeline consistency
        return np.stack([mono_out, mono_out, mono_out], axis=-1)



    stretch_needed = (np.median(img_rgb01 - np.min(img_rgb01)) < 0.05)

    if stretch_needed:
        stretched_core, original_min, original_medians = stretch_image(img_rgb01)
    else:
        stretched_core = img_rgb01.astype(np.float32, copy=False)
        original_min = float(np.min(img_rgb01))
        original_medians = [float(np.median(img_rgb01[..., c])) for c in range(3)]

    stretched = add_border(stretched_core, border_size=16)

    # Process
    if separate_channels or denoise_mode == "separate":
        out_ch = []
        for c in range(3):
            dch = denoise_channel(stretched[..., c], models, which="mono", chunk_size=chunk_size, overlap=overlap, progress_cb=progress_cb)
            out_ch.append(blend_images(stretched[..., c], dch, denoise_strength))
        den = np.stack(out_ch, axis=-1)

    elif denoise_mode == "luminance":
        y, cb, cr = extract_luminance(stretched)
        den_y = denoise_channel(y, models, which="mono", chunk_size=chunk_size, overlap=overlap, progress_cb=progress_cb)
        y2 = blend_images(y, den_y, denoise_strength)
        den = merge_luminance(y2, cb, cr)

    else:
        # full: L via MONO NN, color via COLOR NN (no guided chroma)
        y, cb, cr = extract_luminance(stretched)

        # 1) denoise luminance with mono model
        den_y = denoise_channel(y, models, which="mono", chunk_size=chunk_size, overlap=overlap, progress_cb=progress_cb)
        y2 = blend_images(y, den_y, denoise_strength)

        # 2) denoise RGB with color model, then use it only for chroma (Cb/Cr)
        den_rgb = denoise_rgb_with_color_model(stretched, models, chunk_size=chunk_size, overlap=overlap, progress_cb=progress_cb)

        # Color slider controls how much we take chroma from den_rgb vs original
        cs = denoise_strength if color_denoise_strength is None else float(color_denoise_strength)
        cs = float(np.clip(cs, 0.0, 1.0))

        _y_d, cb_d, cr_d = extract_luminance(den_rgb)  # ignore _y_d; we already did luminance via mono NN
        cb2 = (1.0 - cs) * cb + cs * cb_d
        cr2 = (1.0 - cs) * cr + cs * cr_d

        den = merge_luminance(y2, cb2, cr2)

    den = np.clip(den, 0.0, 1.0)
    if stretch_needed:
        den = unstretch_image(den, original_medians, original_min)

    den = remove_border(den, border_size=16)
    return np.clip(den, 0.0, 1.0).astype(np.float32, copy=False)