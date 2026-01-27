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

try:
    import onnxruntime as ort
except Exception:
    ort = None


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


# ----------------------------
# Model definitions (unchanged)
# ----------------------------
def _load_torch_model(torch, device, ckpt_path: str):
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        def forward(self, x):
            residual = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            out = self.relu(out + residual)
            return out

    class DenoiseCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), ResidualBlock(16))
            self.encoder2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), ResidualBlock(32))
            self.encoder3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=2, dilation=2), nn.ReLU(), ResidualBlock(64))
            self.encoder4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), ResidualBlock(128))
            self.encoder5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=2, dilation=2), nn.ReLU(), ResidualBlock(256))

            self.decoder5 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU(), ResidualBlock(128))
            self.decoder4 = nn.Sequential(nn.Conv2d(128 +  64,  64, 3, padding=1), nn.ReLU(), ResidualBlock(64))
            self.decoder3 = nn.Sequential(nn.Conv2d( 64 +  32,  32, 3, padding=1), nn.ReLU(), ResidualBlock(32))
            self.decoder2 = nn.Sequential(nn.Conv2d( 32 +  16,  16, 3, padding=1), nn.ReLU(), ResidualBlock(16))
            self.decoder1 = nn.Sequential(nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

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

    net = DenoiseCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt.get("model_state_dict", ckpt))
    net.eval()
    return net


# ----------------------------
# Model cache
# ----------------------------
_cached_models: dict[tuple[str, bool], Dict[str, Any]] = {}  # (backend_tag, use_gpu)
_BACKEND_TAG = "cc_denoise_ai3_6"

R = get_resources()


def load_models(use_gpu: bool = True, status_cb=print) -> Dict[str, Any]:
    key = (_BACKEND_TAG, bool(use_gpu))
    if key in _cached_models:
        return _cached_models[key]

    is_windows = (os.name == "nt")

    torch = _get_torch(
        prefer_cuda=bool(use_gpu),
        prefer_dml=bool(use_gpu and is_windows),
        status_cb=status_cb,
    )

    # 1) CUDA
    if use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
        status_cb(f"CosmicClarity Denoise: using CUDA ({torch.cuda.get_device_name(0)})")
        mono_model = _load_torch_model(torch, device, R.CC_DENOISE_PTH)
        models = {"device": device, "is_onnx": False, "mono_model": mono_model, "torch": torch}
        status_cb(f"Denoise backend resolved: "
                f"{'onnx' if models['is_onnx'] else 'torch'} / device={models['device']!r}")        
        _cached_models[key] = models
        return models
    # >>> ADD THIS BLOCK HERE <<<
    # 2) MPS (macOS Apple Silicon)
    if use_gpu and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        status_cb("CosmicClarity Denoise: using MPS")
        mono_model = _load_torch_model(torch, device, R.CC_DENOISE_PTH)
        models = {"device": device, "is_onnx": False, "mono_model": mono_model, "torch": torch}
        status_cb(f"Denoise backend resolved: "
                f"{'onnx' if models['is_onnx'] else 'torch'} / device={models['device']!r}")
        _cached_models[key] = models
        return models
    # >>> END INSERT <<<
    # 2) Torch-DirectML (Windows)
    if use_gpu and is_windows:
        try:
            import torch_directml
            dml = torch_directml.device()
            status_cb("CosmicClarity Denoise: using DirectML (torch-directml)")
            mono_model = _load_torch_model(torch, dml, R.CC_DENOISE_PTH)
            models = {"device": dml, "is_onnx": False, "mono_model": mono_model, "torch": torch}
            status_cb(f"Denoise backend resolved: "
                    f"{'onnx' if models['is_onnx'] else 'torch'} / device={models['device']!r}")                 
            _cached_models[key] = models
            return models
        except Exception:
            pass

    # 3) ORT DirectML fallback
    if use_gpu and ort is not None and ("DmlExecutionProvider" in ort.get_available_providers()):
        status_cb("CosmicClarity Denoise: using DirectML (ONNX Runtime)")
        mono_model = ort.InferenceSession(R.CC_DENOISE_ONNX, providers=["DmlExecutionProvider"])
        models = {"device": "DirectML", "is_onnx": True, "mono_model": mono_model, "torch": None}
        status_cb(f"Denoise backend resolved: "
                f"{'onnx' if models['is_onnx'] else 'torch'} / device={models['device']!r}")             
        _cached_models[key] = models
        return models

    # 4) CPU
    device = torch.device("cpu")
    status_cb("CosmicClarity Denoise: using CPU")
    mono_model = _load_torch_model(torch, device, R.CC_DENOISE_PTH)
    models = {"device": device, "is_onnx": False, "mono_model": mono_model, "torch": torch}
    status_cb(f"Denoise backend resolved: "
            f"{'onnx' if models['is_onnx'] else 'torch'} / device={models['device']!r}")         
    _cached_models[key] = models
    return models


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


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Fast guided filter using boxFilter (edge-preserving, very fast).
    guide and src are HxW float32 in [0,1].
    radius is the neighborhood radius; ksize=(2*radius+1).
    eps is the regularization term.
    """
    r = max(1, int(radius))
    ksize = (2*r + 1, 2*r + 1)

    mean_I  = cv2.boxFilter(guide, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    mean_p  = cv2.boxFilter(src,   ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    mean_Ip = cv2.boxFilter(guide * src, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    cov_Ip  = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)

    q = mean_a * guide + mean_b
    return q



def denoise_chroma(cb: np.ndarray,
                   cr: np.ndarray,
                   strength: float,
                   method: str = "guided",
                   strength_scale: float = 2.0,
                   guide_y: np.ndarray | None = None):
    """
    Fast chroma-only denoise for Cb/Cr in [0,1] float32.
    method: 'guided' (default), 'gaussian', 'bilateral'
    strength_scale: lets chroma smoothing go up to ~2× your slider.
    guide_y: optional luminance guide (Y in [0,1]); required for 'guided' to be best.
    """
    eff = float(np.clip(strength * strength_scale, 0.0, 1.0))
    if eff <= 0.0:
        return cb, cr

    cb = cb.astype(np.float32, copy=False)
    cr = cr.astype(np.float32, copy=False)

    if method == "guided":
        # Need a guide; if not provided, fall back to Gaussian
        if guide_y is not None:
            # radius & eps scale with strength; tuned for strong chroma smoothing but edge-safe
            radius = 2 + int(round(10 * eff))         # ~2..12  (ksize ~5..25)
            eps    = (0.001 + 0.05 * eff) ** 2        # small regularization
            cb_f   = _guided_filter(guide_y, cb, radius, eps)
            cr_f   = _guided_filter(guide_y, cr, radius, eps)
        else:
            method = "gaussian"  # no guide provided → fast fallback

    if method == "gaussian":
        k     = 1 + 2 * int(round(8 * eff))           # 1,3,5,..,17
        sigma = max(0.15, 2.4 * eff)
        cb_f  = cv2.GaussianBlur(cb, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
        cr_f  = cv2.GaussianBlur(cr, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

    if method == "bilateral":
        # Bilateral is decent but slower than Gaussian; guided is preferred for speed/quality.
        d      = 5 + 2 * int(round(6 * eff))          # 5..17
        sigmaC = 25.0 * (0.5 + 3.0 * eff)             # ~12.5..100
        sigmaS = 3.0  * (0.5 + 6.0 * eff)             # ~1.5..21
        cb_f = cv2.bilateralFilter(cb, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)
        cr_f = cv2.bilateralFilter(cr, d=d, sigmaColor=sigmaC, sigmaSpace=sigmaS)

    # Blend (maskless)
    w = eff
    cb_out = (1.0 - w) * cb + w * cb_f
    cr_out = (1.0 - w) * cr + w * cr_f
    return cb_out, cr_out


# Function to split an image into chunks with overlap
def split_image_into_chunks_with_overlap(image, chunk_size, overlap):
    height, width = image.shape[:2]
    chunks = []
    step_size = chunk_size - overlap  # Define how much to step over (overlapping area)

    for i in range(0, height, step_size):
        for j in range(0, width, step_size):
            end_i = min(i + chunk_size, height)
            end_j = min(j + chunk_size, width)
            chunk = image[i:end_i, j:end_j]
            chunks.append((chunk, i, j))  # Return chunk and its position
    return chunks

def blend_images(before, after, amount):
    return (1 - amount) * before + amount * after

def stitch_chunks_ignore_border(chunks, image_shape, border_size: int = 16):
    """
    chunks: list of (chunk, i, j) or (chunk, i, j, is_edge)
    image_shape: (H,W)
    """
    H, W = image_shape
    stitched = np.zeros((H, W), dtype=np.float32)
    weights  = np.zeros((H, W), dtype=np.float32)

    for entry in chunks:
        # accept both 3-tuple and 4-tuple
        if len(entry) == 3:
            chunk, i, j = entry
        else:
            chunk, i, j, _ = entry

        h, w = chunk.shape[:2]
        bh = min(border_size, h // 2)
        bw = min(border_size, w // 2)

        inner = chunk[bh:h-bh, bw:w-bw]
        stitched[i+bh:i+h-bh, j+bw:j+w-bw] += inner
        weights[i+bh:i+h-bh, j+bw:j+w-bw] += 1.0

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

    if x.ndim == 2:
        med = float(np.median(x))
        orig_meds = [med]
        if med != 0:
            x = ((med - 1) * target_median * x) / (med * (target_median + x - 1) - target_median * x)
        return np.clip(x, 0, 1), orig_min, orig_meds

    # 3ch
    orig_meds = [float(np.median(x[..., c])) for c in range(3)]
    for c in range(3):
        m = orig_meds[c]
        if m != 0:
            x[..., c] = ((m - 1) * target_median * x[..., c]) / (
                m * (target_median + x[..., c] - 1) - target_median * x[..., c]
            )
    return np.clip(x, 0, 1), orig_min, orig_meds


def unstretch_image_unlinked(image: np.ndarray, orig_meds, orig_min: float):
    x = np.asarray(image, np.float32).copy()

    if x.ndim == 2:
        m_now = float(np.median(x))
        m0 = float(orig_meds[0])
        if m_now != 0 and m0 != 0:
            x = ((m_now - 1) * m0 * x) / (m_now * (m0 + x - 1) - m0 * x)
        x += float(orig_min)
        return np.clip(x, 0, 1)

    for c in range(3):
        m_now = float(np.median(x[..., c]))
        m0 = float(orig_meds[c])
        if m_now != 0 and m0 != 0:
            x[..., c] = ((m_now - 1) * m0 * x[..., c]) / (
                m_now * (m0 + x[..., c] - 1) - m0 * x[..., c]
            )

    x += float(orig_min)
    return np.clip(x, 0, 1)

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


# ----------------------------
# Channel denoise (paste + keep)
# IMPORTANT: remove print() spam; instead accept an optional progress callback
# ----------------------------
def denoise_channel(channel: np.ndarray, models: Dict[str, Any], *, progress_cb: ProgressCB | None = None) -> np.ndarray:
    device = models["device"]
    is_onnx = models["is_onnx"]
    model = models["mono_model"]

    chunk_size = 256
    overlap = 64
    chunks = split_image_into_chunks_with_overlap(channel, chunk_size=chunk_size, overlap=overlap)

    denoised_chunks = []
    total = len(chunks)

    for idx, (chunk, i, j) in enumerate(chunks):
        original_chunk_shape = chunk.shape

        if is_onnx:
            chunk_input = chunk[np.newaxis, np.newaxis, :, :].astype(np.float32)
            chunk_input = np.tile(chunk_input, (1, 3, 1, 1))
            if chunk_input.shape[2] != chunk_size or chunk_input.shape[3] != chunk_size:
                padded = np.zeros((1, 3, chunk_size, chunk_size), dtype=np.float32)
                padded[:, :, :chunk_input.shape[2], :chunk_input.shape[3]] = chunk_input
                chunk_input = padded

            input_name = model.get_inputs()[0].name
            out = model.run(None, {input_name: chunk_input})[0]
            denoised_chunk = out[0, 0, :original_chunk_shape[0], :original_chunk_shape[1]]

        else:
            torch = models["torch"]
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            chunk_tensor = chunk_tensor.expand(1, 3, chunk_tensor.shape[2], chunk_tensor.shape[3])

            with torch.no_grad(), _autocast_context(torch, device):
                out = model(chunk_tensor).detach().cpu().numpy()  # (1,3,H,W)

            denoised_chunk = out[0, 0, :original_chunk_shape[0], :original_chunk_shape[1]]

        denoised_chunks.append((denoised_chunk, i, j))

        if progress_cb is not None:
            progress_cb(idx + 1, total)

    return stitch_chunks_ignore_border(denoised_chunks, channel.shape, border_size=16)

# ----------------------------
# High-level denoise for a loaded RGB float image (0..1)
# (this is the “engine API” SASpro will call)
# ----------------------------
def denoise_rgb01(
    img_rgb01: np.ndarray,
    *,
    denoise_strength: float,
    denoise_mode: str = "luminance",          # luminance | full | separate
    separate_channels: bool = False,
    color_denoise_strength: Optional[float] = None,
    use_gpu: bool = True,
    progress_cb=None,
) -> np.ndarray:
    """
    Input: float32 RGB [0..1]
    Output: float32 RGB [0..1]
    """
    models = load_models(use_gpu=use_gpu)

    # Determine stretch necessity (keep your logic)
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
            dch = denoise_channel(stretched[..., c], models, progress_cb=progress_cb)
            out_ch.append(blend_images(stretched[..., c], dch, denoise_strength))
        den = np.stack(out_ch, axis=-1)

    elif denoise_mode == "luminance":
        y, cb, cr = extract_luminance(stretched)
        den_y = denoise_channel(y, models, progress_cb=progress_cb)
        y2 = blend_images(y, den_y, denoise_strength)
        den = merge_luminance(y2, cb, cr)

    else:
        # full: L via NN, chroma via guided
        y, cb, cr = extract_luminance(stretched)
        den_y = denoise_channel(y, models, progress_cb=progress_cb)
        y2 = blend_images(y, den_y, denoise_strength)

        cs = denoise_strength if color_denoise_strength is None else color_denoise_strength
        cb2, cr2 = denoise_chroma(cb, cr, strength=cs, method="guided", guide_y=y)
        den = merge_luminance(y2, cb2, cr2)

    # unstretch if needed
    if stretch_needed:
        den = unstretch_image(den, original_medians, original_min)

    den = remove_border(den, border_size=16)
    return np.clip(den, 0.0, 1.0).astype(np.float32, copy=False)
