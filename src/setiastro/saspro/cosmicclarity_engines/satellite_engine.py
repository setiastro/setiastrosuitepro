# src/setiastro/saspro/cosmicclarity_engines/satellite_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from setiastro.saspro.resources import get_resources

# Optional deps
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from skimage.transform import resize as _sk_resize
except Exception:
    _sk_resize = None

ProgressCB = Callable[[int, int], None]  # (done, total)

# ---------- Torch import (updated: CUDA + torch-directml awareness) ----------

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
#   Models (from standalone)
# ----------------------------

def _build_torch_models(torch):
    # Import torch.nn + torchvision lazily, only after torch loads
    import torch.nn as nn

    try:
        from torchvision import models
        from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError(f"torchvision is required for Satellite engine torch backend: {e}")

    class LayerNorm2d(nn.Module):
        def __init__(self, channels: int, eps: float = 1e-6):
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
        def __init__(self, channels: int):
            super().__init__()
            self.norm1 = LayerNorm2d(channels)
            self.conv1 = nn.Conv2d(channels, channels * 2, 1, bias=True)
            self.dwconv = nn.Conv2d(
                channels * 2, channels * 2, 3, padding=1,
                groups=channels * 2, bias=True
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

    class NAFNetSatelliteRemover(nn.Module):
        """
        Satellite removal AI4 NAFNet.
        Residual output: returns x + delta, no clamp.
        """
        def __init__(
            self,
            in_ch: int = 3,
            out_ch: int = 3,
            width: int = 32,
            enc_blk_nums=(2, 4, 6, 8),
            dec_blk_nums=(2, 2, 2, 2),
            middle_blk_num: int = 4,
            residual_out: bool = True,
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
                self.ups.append(
                    nn.Sequential(
                        nn.Conv2d(ch, ch * 2, 1, bias=True),
                        nn.PixelShuffle(2),
                    )
                )
                ch //= 2
                self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

            self.ending = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
            self.residual_out = bool(residual_out)

        def forward(self, x):
            x0 = x
            y = self.intro(x)

            skips = []
            for enc, down in zip(self.encoders, self.downs):
                y = enc(y)
                skips.append(y)
                y = down(y)

            y = self.middle(y)

            for up, dec in zip(self.ups, self.decoders):
                y = up(y)
                y = y + skips.pop()
                y = dec(y)

            delta = self.ending(y)
            return (x0 + delta) if self.residual_out else delta

    class BinaryClassificationCNN(nn.Module):
        def __init__(self, input_channels: int = 3):
            super().__init__()
            self.pre_conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.pre_conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.features = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.features.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.features.fc = nn.Linear(self.features.fc.in_features, 1)

        def forward(self, x):
            x = self.pre_conv1(x)
            x = self.pre_conv2(x)
            return self.features(x)

    class BinaryClassificationCNN2(nn.Module):
        def __init__(self, input_channels: int = 3):
            super().__init__()
            self.pre_conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.pre_conv2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            self.mobilenet.features[0][0] = nn.Conv2d(
                64, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            in_features = self.mobilenet.classifier[-1].in_features
            self.mobilenet.classifier[-1] = nn.Linear(in_features, 1)

        def forward(self, x):
            x = self.pre_conv1(x)
            x = self.pre_conv2(x)
            return self.mobilenet(x)

    # Also return the torchvision transforms helper you used
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

    return nn, NAFNetSatelliteRemover, BinaryClassificationCNN, BinaryClassificationCNN2, tfm


# ----------------------------
#   Loading helpers
# ----------------------------
def _extract_state_dict_from_checkpoint(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt

    for key in ("model_state_dict", "state_dict", "net", "model", "ema_state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            break
    else:
        sd = ckpt

    cleaned = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def _load_model_weights_lenient(torch, nn, model, checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict_from_checkpoint(ckpt)

    msd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in msd and msd[k].shape == v.shape}

    missing = [k for k in msd.keys() if k not in filtered]
    unexpected = [k for k in state_dict.keys() if k not in msd]

    model.load_state_dict(filtered, strict=False)

    if missing:
        print(f"[Satellite] {checkpoint_path}: missing {len(missing)} keys after filtering")
    if unexpected:
        print(f"[Satellite] {checkpoint_path}: unexpected {len(unexpected)} keys ignored")

    return model

# ---------- Satellite model cache + loader (updated for torch-directml + ORT DML) ----------

_SAT_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

def get_satellite_models(resources: Any = None, use_gpu: bool = True, status_cb=print) -> Dict[str, Any]:
    """
    Backend order:
      1) CUDA (PyTorch)
      2) DirectML (torch-directml)        [Windows]
      3) DirectML (ONNX Runtime DML EP)   [Windows]
      4) MPS (PyTorch)                   [macOS]
      5) CPU (PyTorch)

    Cache key includes backend tag, so switching GPU on/off never reuses the wrong backend.
    """
    import os

    if resources is None:
        resources = get_resources()

    p_det1 = resources.CC_SAT_DETECT1_PTH
    p_det2 = resources.CC_SAT_DETECT2_PTH
    p_rem  = resources.CC_SAT_REMOVE_PTH

    o_det1 = resources.CC_SAT_DETECT1_ONNX
    o_det2 = resources.CC_SAT_DETECT2_ONNX
    o_rem  = resources.CC_SAT_REMOVE_ONNX

    is_windows = (os.name == "nt")

    # ORT DirectML availability
    ort_dml_ok = bool(use_gpu) and (ort is not None) and ("DmlExecutionProvider" in ort.get_available_providers())

    # Torch: ask runtime_torch to prefer what we want (CUDA first, DML on Windows)
    torch = None

    torch = _get_torch(
        prefer_cuda=bool(use_gpu),
        prefer_dml=bool(use_gpu and is_windows),
        status_cb=status_cb,
    )

    # Decide backend
    backend = "cpu"

    # 1) CUDA
    if use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
        backend = "cuda"
    else:
        # 2) torch-directml (Windows)
        if use_gpu and is_windows:
            try:
                import torch_directml  # optional
                _ = torch_directml.device()
                backend = "torch_dml"
            except Exception:
                backend = "ort_dml" if ort_dml_ok else "cpu"
        else:
            # 4) MPS (macOS)
            if use_gpu and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                backend = "mps"
            else:
                backend = "cpu"

    key = (p_det1, p_det2, p_rem, backend)
    if key in _SAT_CACHE:
        return _SAT_CACHE[key]

    # ---------------- DirectML via ONNX Runtime ----------------
    if backend == "ort_dml":
        if ort is None:
            raise RuntimeError("onnxruntime not available, cannot use ORT DirectML backend.")
        # sanity: need ONNX paths
        if not (o_det1 and o_det2 and o_rem):
            raise FileNotFoundError("Satellite ONNX model paths are missing in resources.")

        det1 = ort.InferenceSession(o_det1, providers=["DmlExecutionProvider"])
        det2 = ort.InferenceSession(o_det2, providers=["DmlExecutionProvider"])
        rem  = ort.InferenceSession(o_rem,  providers=["DmlExecutionProvider"])

        out = {
            "backend": "ort_dml",
            "detection_model1": det1,
            "detection_model2": det2,
            "removal_model": rem,
            "device": "DirectML",
            "is_onnx": True,
        }
        _SAT_CACHE[key] = out
        status_cb("CosmicClarity Satellite: using DirectML (ONNX Runtime)")
        return out

    # ---------------- Torch backends (CUDA / torch-directml / MPS / CPU) ----------------
    # pick device
    if backend == "cuda":
        device = torch.device("cuda")
        status_cb(f"CosmicClarity Satellite: using CUDA ({torch.cuda.get_device_name(0)})")
    elif backend == "mps":
        device = torch.device("mps")
        status_cb("CosmicClarity Satellite: using MPS")
    elif backend == "torch_dml":
        import torch_directml
        device = torch_directml.device()
        status_cb("CosmicClarity Satellite: using DirectML (torch-directml)")
    else:
        device = torch.device("cpu")
        status_cb("CosmicClarity Satellite: using CPU")

    nn, NAFNetSatelliteRemover, BinaryClassificationCNN, BinaryClassificationCNN2, tfm = _build_torch_models(torch)

    det1 = BinaryClassificationCNN(3).to(device)
    det1 = _load_model_weights_lenient(torch, nn, det1, p_det1, device).eval()

    det2 = BinaryClassificationCNN2(3).to(device)
    det2 = _load_model_weights_lenient(torch, nn, det2, p_det2, device).eval()

    rem = NAFNetSatelliteRemover(
        width=32,
        enc_blk_nums=(2, 4, 6, 8),
        dec_blk_nums=(2, 2, 2, 2),
        middle_blk_num=4,
        residual_out=True,
    ).to(device)
    rem = _load_model_weights_lenient(torch, nn, rem, p_rem, device).eval()

    out = {
        "backend": backend,
        "detection_model1": det1,
        "detection_model2": det2,
        "removal_model": rem,
        "device": device,
        "is_onnx": False,
        "torch": torch,
        "tfm": tfm,
    }
    _SAT_CACHE[key] = out
    return out

# ----------------------------
#   Core processing
# ----------------------------

def _ensure_rgb01(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Input: HxW, HxWx1, or HxWx3; float/uint.
    Output: HxWx3 float32 [0..1], plus is_mono flag (originally mono-like).
    """
    a = np.asarray(img)
    is_mono = (a.ndim == 2) or (a.ndim == 3 and a.shape[2] == 1)

    a = np.nan_to_num(a.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    # normalize if >1
    mx = float(np.max(a)) if a.size else 1.0
    if mx > 1.0:
        a = a / mx

    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {a.shape}")

    a = np.clip(a, 0.0, 1.0)
    return a, is_mono


def _extract_luminance_bt601(rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # BT.601 matrix (matches your standalone)
    M = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]], dtype=np.float32)
    ycbcr = np.dot(rgb01, M.T)
    y  = ycbcr[..., 0]
    cb = ycbcr[..., 1] + 0.5
    cr = ycbcr[..., 2] + 0.5
    return y, cb, cr


def _merge_luminance_bt601(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    y  = np.clip(y,  0, 1).astype(np.float32)
    cb = (np.clip(cb, 0, 1).astype(np.float32) - 0.5)
    cr = (np.clip(cr, 0, 1).astype(np.float32) - 0.5)

    ycbcr = np.stack([y, cb, cr], axis=-1)

    M = np.array([[1.0,  0.0, 1.402],
                  [1.0, -0.344136, -0.714136],
                  [1.0,  1.772, 0.0]], dtype=np.float32)
    rgb = np.dot(ycbcr, M.T)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _split_chunks(img: np.ndarray, chunk: int, overlap: int):
    H, W = img.shape[:2]
    step = chunk - overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1 = min(y0 + chunk, H)
            x1 = min(x0 + chunk, W)
            yield img[y0:y1, x0:x1], y0, x0

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
def stretch_image(image: np.ndarray, *, target_median: float = 0.25):
    return stretch_image_unlinked(image, target_median=float(target_median))

def unstretch_image(image: np.ndarray, original_medians, original_min: float, *, target_median: float = 0.25):
    return unstretch_image_unlinked(image, original_medians, original_min, target_median=float(target_median))


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


def _stitch_ignore_border(chunks, shape_hw3, border: int = 16) -> np.ndarray:
    H, W, C = shape_hw3
    acc = np.zeros((H, W, C), np.float32)
    wgt = np.zeros((H, W, C), np.float32)

    for tile, y0, x0 in chunks:
        th, tw = tile.shape[:2]
        bh = min(border, th // 2)
        bw = min(border, tw // 2)

        inner = tile[bh:th-bh, bw:tw-bw, :]
        acc[y0+bh:y0+th-bh, x0+bw:x0+tw-bw, :] += inner
        wgt[y0+bh:y0+th-bh, x0+bw:x0+tw-bw, :] += 1.0

    return acc / np.maximum(wgt, 1.0)


def _apply_clip_trail_logic(processed: np.ndarray, original: np.ndarray, sensitivity: float) -> np.ndarray:
    # exactly your standalone math
    sattrail_only = original - processed
    mean_val = float(np.mean(sattrail_only))
    clipped = np.clip((sattrail_only - mean_val) * 10.0, 0.0, 1.0)
    mask = np.where(clipped < sensitivity, 0.0, 1.0).astype(np.float32)
    return np.clip(original - mask, 0.0, 1.0)

def _resize_tile_for_detect(tile_rgb01: np.ndarray) -> np.ndarray:
    if _sk_resize is None:
        raise RuntimeError("skimage.transform.resize is required for satellite detection.")
    r = _sk_resize(tile_rgb01, (256, 256, 3), mode="reflect", anti_aliasing=True).astype(np.float32)
    return r

def _pad_tile_to_shape_rgb(tile: np.ndarray, out_h: int, out_w: int) -> tuple[np.ndarray, int, int]:
    tile = np.asarray(tile, np.float32)
    h, w = tile.shape[:2]
    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)

    if pad_h or pad_w:
        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    return tile, h, w

def _torch_detect_batch(tiles_rgb01: list[np.ndarray], models: Dict[str, Any], batch_size: int = 32) -> list[bool]:
    torch = models["torch"]
    device = models["device"]
    det1 = models["detection_model1"]
    det2 = models["detection_model2"]

    resized = [_resize_tile_for_detect(t) for t in tiles_rgb01]
    flags = [False] * len(resized)

    for i in range(0, len(resized), batch_size):
        batch_np = np.stack(resized[i:i+batch_size], axis=0)  # NHWC
        batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)  # NCHW

        x = torch.from_numpy(batch_np)
        if hasattr(device, "type") and device.type == "cuda":
            x = x.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
        else:
            x = x.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            o1 = det1(x).flatten()
        keep1 = (o1 > 0.5)

        if keep1.any():
            idxs = keep1.nonzero(as_tuple=False).flatten()
            with torch.no_grad():
                o2 = det2(x[idxs]).flatten()
            keep2 = (o2 > 0.25).detach().cpu().numpy()

            idxs_cpu = idxs.detach().cpu().numpy()
            for local_j, passed in zip(idxs_cpu, keep2):
                if passed:
                    flags[i + int(local_j)] = True

    return flags

def _torch_remove_batch(
    tiles_rgb01: list[np.ndarray],
    models: Dict[str, Any],
    *,
    padded_h: int,
    padded_w: int,
    batch_size: int = 8,
) -> list[np.ndarray]:
    torch = models["torch"]
    device = models["device"]
    rem = models["removal_model"]

    padded_tiles = []
    orig_shapes = []

    for tile in tiles_rgb01:
        padded, h, w = _pad_tile_to_shape_rgb(tile, padded_h, padded_w)
        padded_tiles.append(padded)
        orig_shapes.append((h, w))

    outs: list[np.ndarray] = []

    for i in range(0, len(padded_tiles), batch_size):
        batch_tiles = padded_tiles[i:i+batch_size]
        batch_np = np.stack(batch_tiles, axis=0)  # NHWC
        batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)  # NCHW

        x = torch.from_numpy(batch_np)
        if hasattr(device, "type") and device.type == "cuda":
            x = x.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
        else:
            x = x.to(device=device, dtype=torch.float32)

        with torch.no_grad(), _autocast_context(torch, device):
            y = rem(x).detach().cpu().numpy()  # NCHW

        y = np.transpose(y, (0, 2, 3, 1))  # NHWC

        for j, out in enumerate(y):
            h, w = orig_shapes[i + j]
            outs.append(np.clip(out[:h, :w, :], 0.0, 1.0).astype(np.float32))

    return outs

# ---------- Torch detection (FIX: tfm expects PIL/ndarray, not Tensor; avoid double ToTensor) ----------

def _torch_detect(tile_rgb01: np.ndarray, models: Dict[str, Any]) -> bool:
    """
    Your tfm = ToTensor()+Resize(256,256). It expects HxWxC numpy in [0..1] (or uint8).
    Do NOT feed it a tensor.
    """
    torch = models["torch"]
    device = models["device"]
    det1 = models["detection_model1"]
    det2 = models["detection_model2"]
    tfm = models["tfm"]

    a = np.asarray(tile_rgb01, np.float32)
    a = np.clip(a, 0.0, 1.0)

    # torchvision transform pipeline
    inp = tfm(a)               # -> Tensor [C,H,W], float32
    inp = inp.unsqueeze(0).to(device)

    with torch.no_grad():
        o1 = float(det1(inp).item())
    if o1 <= 0.5:
        return False

    with torch.no_grad():
        o2 = float(det2(inp).item())
    return (o2 > 0.25)



def _torch_remove(tile_rgb01: np.ndarray, models: Dict[str, Any]) -> np.ndarray:
    torch = models["torch"]
    device = models["device"]
    rem = models["removal_model"]

    tile = np.asarray(tile_rgb01, np.float32)
    h, w = tile.shape[:2]

    # NAFNet downsamples 4 times -> require multiple of 16
    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16

    if pad_h or pad_w:
        tile = np.pad(
            tile,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )

    x = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad(), _autocast_context(torch, device):
        out = rem(x).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

    # crop back to original tile size
    out = out[:h, :w, :]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _onnx_detect(tile_rgb01: np.ndarray, sess) -> bool:
    # Resize to 256x256 like your standalone ONNX path
    if _sk_resize is None:
        raise RuntimeError("skimage.transform.resize is required for ONNX satellite detection path.")
    r = _sk_resize(tile_rgb01, (256, 256, 3), mode="reflect", anti_aliasing=True).astype(np.float32)
    inp = np.transpose(r, (2, 0, 1))[None, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    return bool(out[0] > 0.5)


def _onnx_remove(tile_rgb01: np.ndarray, sess) -> np.ndarray:
    tile = np.asarray(tile_rgb01, np.float32)
    h, w = tile.shape[:2]

    pad_h = (16 - (h % 16)) % 16
    pad_w = (16 - (w % 16)) % 16

    if pad_h or pad_w:
        tile = np.pad(
            tile,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )

    inp = np.transpose(tile, (2, 0, 1))[None, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    pred = np.transpose(out.squeeze(0), (1, 2, 0)).astype(np.float32)

    pred = pred[:h, :w, :]
    return np.clip(pred, 0.0, 1.0)

def satellite_remove_image(
    image: np.ndarray,
    models: Dict[str, Any],
    *,
    mode: str = "full",
    clip_trail: bool = True,
    sensitivity: float = 0.1,
    chunk_size: int = 256,
    overlap: int = 64,
    border_size: int = 16,
    temp_stretch: bool = True,
    target_median: float = 0.25,
    compatibility_mode: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, bool]:
    """
    image: input image (any dtype/shape). Expected to be linear-ish in [0..1] for best behavior.
    Returns: (out_image_same_shape_style, trail_detected_any)
    """
    rgb01, was_mono = _ensure_rgb01(image)
    tm = float(np.clip(target_median, 0.01, 0.50))

    def _should_stretch(x: np.ndarray) -> bool:
        # same idea as denoise engine
        x = np.asarray(x, np.float32)
        return bool(np.median(x - np.min(x)) < 0.05)

    # luminance mode -> process Y only, then merge back
    if mode.lower() == "luminance":
        y, cb, cr = _extract_luminance_bt601(rgb01)

        # --- NEW: temp stretch on luminance only ---
        if bool(temp_stretch):
            stretch_needed = True
        else:
            stretch_needed = _should_stretch(y)

        if stretch_needed:
            y_s, original_min, original_meds = stretch_image(y, target_median=tm)
        else:
            y_s = y.astype(np.float32, copy=False)
            original_min = float(np.min(y))
            original_meds = [float(np.median(y_s))]

        # run network as RGB by repeating Y
        y3 = np.stack([y_s, y_s, y_s], axis=-1)

        out3, detected = _satellite_remove_rgb(
            y3, models,
            clip_trail=clip_trail, sensitivity=sensitivity,
            chunk_size=chunk_size, overlap=overlap, border_size=border_size,
            compatibility_mode=compatibility_mode,
            progress_cb=progress_cb,
        )

        out_y = out3[..., 0]

        # --- NEW: unstretch luminance back ---
        if stretch_needed:
            out_y = unstretch_image(out_y, original_meds, original_min, target_median=tm)

        out_y = np.clip(out_y, 0.0, 1.0).astype(np.float32, copy=False)
        out_rgb = _merge_luminance_bt601(out_y, cb, cr)

    else:
        # --- NEW: temp stretch on full RGB before detection/removal ---
        if bool(temp_stretch):
            stretch_needed = True
        else:
            stretch_needed = _should_stretch(rgb01)

        if stretch_needed:
            rgb_s, original_min, original_meds = stretch_image(rgb01, target_median=tm)
        else:
            rgb_s = rgb01.astype(np.float32, copy=False)
            original_min = float(np.min(rgb01))
            original_meds = [float(np.median(rgb01[..., c])) for c in range(3)]

        out_rgb, detected = _satellite_remove_rgb(
            rgb_s, models,
            clip_trail=clip_trail, sensitivity=sensitivity,
            chunk_size=chunk_size, overlap=overlap, border_size=border_size,
            compatibility_mode=compatibility_mode,
            progress_cb=progress_cb,
        )

        # --- NEW: unstretch back ---
        if stretch_needed:
            out_rgb = unstretch_image(out_rgb, original_meds, original_min, target_median=tm)

        out_rgb = np.clip(out_rgb, 0.0, 1.0).astype(np.float32, copy=False)

    # If original was mono-like, return HxWx1 (matches your SASpro convention for mono docs)
    if (np.asarray(image).ndim == 2) or (np.asarray(image).ndim == 3 and np.asarray(image).shape[2] == 1):
        out_m = out_rgb[..., 0:1].astype(np.float32)
        return out_m, detected

    return out_rgb.astype(np.float32), detected


# ---------- Satellite remove loop (FIX: use correct ONNX sessions, not det1/rem confusion) ----------

def _satellite_remove_rgb(
    rgb01: np.ndarray,
    models: Dict[str, Any],
    *,
    clip_trail: bool,
    sensitivity: float,
    chunk_size: int,
    overlap: int,
    border_size: int,
    compatibility_mode: bool,
    progress_cb: Optional[Callable[[int, int], None]],
) -> Tuple[np.ndarray, bool]:
    is_onnx = bool(models.get("is_onnx", False))

    H, W = rgb01.shape[:2]
    trail_any = False

    all_tiles = list(_split_chunks(rgb01, chunk_size, overlap))
    total = len(all_tiles)
    out_tiles = [None] * total

    # ---------------- ONNX path (leave mostly serial for now) ----------------
    if is_onnx:
        for idx, (tile, y0, x0) in enumerate(all_tiles, start=1):
            orig = tile.astype(np.float32, copy=False)

            det1_sess = models["detection_model1"]
            det2_sess = models["detection_model2"]
            rem_sess  = models["removal_model"]

            d1 = _onnx_detect(orig, det1_sess)
            d2 = _onnx_detect(orig, det2_sess) if d1 else False

            detected = bool(d1 and d2)
            if detected:
                trail_any = True
                pred = _onnx_remove(orig, rem_sess)
                final = _apply_clip_trail_logic(pred, orig, sensitivity) if clip_trail else pred
            else:
                final = orig

            out_tiles[idx - 1] = (final, y0, x0)

            if progress_cb is not None:
                progress_cb(idx, total)

    # ---------------- Torch path (batched) ----------------
    else:
        tiles_only = [tile.astype(np.float32, copy=False) for (tile, _, _) in all_tiles]

        if compatibility_mode:
            detect_bs = 8
            remove_bs = 2
        else:
            detect_bs = 32
            remove_bs = 8

        detected_flags = [False] * total
        done_flags = [False] * total

        # -------- Batched detection with progress --------
        torch = models["torch"]
        device = models["device"]
        det1 = models["detection_model1"]
        det2 = models["detection_model2"]

        resized = [_resize_tile_for_detect(t) for t in tiles_only]

        for i in range(0, len(resized), detect_bs):
            batch_np = np.stack(resized[i:i + detect_bs], axis=0)
            batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)

            x = torch.from_numpy(batch_np)
            if hasattr(device, "type") and device.type == "cuda":
                x = x.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            else:
                x = x.to(device=device, dtype=torch.float32)

            with torch.inference_mode():
                o1 = det1(x).flatten()
            keep1 = (o1 > 0.5)

            local_detected = np.zeros(len(batch_np), dtype=bool)

            if keep1.any():
                idxs = keep1.nonzero(as_tuple=False).flatten()
                with torch.inference_mode():
                    o2 = det2(x[idxs]).flatten()
                keep2 = (o2 > 0.25).detach().cpu().numpy()
                idxs_cpu = idxs.detach().cpu().numpy()

                for local_j, passed in zip(idxs_cpu, keep2):
                    if passed:
                        local_detected[int(local_j)] = True

            # store detection results
            for local_j in range(len(batch_np)):
                global_j = i + local_j
                detected_flags[global_j] = bool(local_detected[local_j])

            # non-detected tiles are fully done now
            for local_j in range(len(batch_np)):
                global_j = i + local_j
                if not detected_flags[global_j]:
                    tile, y0, x0 = all_tiles[global_j]
                    out_tiles[global_j] = (tile.astype(np.float32, copy=False), y0, x0)
                    done_flags[global_j] = True

            if progress_cb is not None:
                progress_cb(sum(done_flags), total)

        detected_indices = [i for i, flag in enumerate(detected_flags) if flag]
        trail_any = bool(detected_indices)

        # -------- Batched removal with progress --------
        if detected_indices:
            padded_tiles = []
            orig_shapes = []

            for idx in detected_indices:
                padded, h, w = _pad_tile_to_shape_rgb(tiles_only[idx], chunk_size, chunk_size)
                padded_tiles.append(padded)
                orig_shapes.append((h, w))

            for batch_start in range(0, len(padded_tiles), remove_bs):
                batch_tiles = padded_tiles[batch_start:batch_start + remove_bs]
                batch_np = np.stack(batch_tiles, axis=0)
                batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)

                x = torch.from_numpy(batch_np)
                if hasattr(device, "type") and device.type == "cuda":
                    x = x.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
                else:
                    x = x.to(device=device, dtype=torch.float32)

                rem = models["removal_model"]
                with torch.inference_mode(), _autocast_context(torch, device):
                    y = rem(x).detach().cpu().numpy()

                y = np.transpose(y, (0, 2, 3, 1))

                for local_j, pred in enumerate(y):
                    src_idx = detected_indices[batch_start + local_j]
                    h, w = orig_shapes[batch_start + local_j]
                    pred = np.clip(pred[:h, :w, :], 0.0, 1.0).astype(np.float32)

                    orig, y0, x0 = all_tiles[src_idx]
                    final = _apply_clip_trail_logic(pred, orig.astype(np.float32, copy=False), sensitivity) if clip_trail else pred
                    out_tiles[src_idx] = (final, y0, x0)
                    done_flags[src_idx] = True

                if progress_cb is not None:
                    progress_cb(sum(done_flags), total)

    out = _stitch_ignore_border(out_tiles, (H, W, 3), border=border_size)

    if border_size > 0:
        out[:border_size, :, :] = rgb01[:border_size, :, :]
        out[-border_size:, :, :] = rgb01[-border_size:, :, :]
        out[:, :border_size, :] = rgb01[:, :border_size, :]
        out[:, -border_size:, :] = rgb01[:, -border_size:, :]

    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    if not trail_any:
        return rgb01.astype(np.float32, copy=False), False

    return out, True