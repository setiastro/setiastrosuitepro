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

def _get_torch(status_cb=print):
    from setiastro.saspro.runtime_torch import import_torch
    return import_torch(prefer_cuda=True, prefer_xpu=False, status_cb=status_cb)

def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()

def _autocast_context(torch, device):
    # mirror your sharpen rule (>= 8.0)
    try:
        if hasattr(device, "type") and device.type == "cuda":
            major, minor = torch.cuda.get_device_capability()
            cap = float(f"{major}.{minor}")
            if cap >= 8.0:
                return torch.cuda.amp.autocast()
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

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        def forward(self, x):
            r = x
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            x = self.relu(x + r)
            return x

    class SatelliteRemoverCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                ResidualBlock(16), ResidualBlock(16),
            )
            self.encoder2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                ResidualBlock(32), ResidualBlock(32),
            )
            self.encoder3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=2, dilation=2), nn.ReLU(),
                ResidualBlock(64), ResidualBlock(64),
            )
            self.encoder4 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                ResidualBlock(128), ResidualBlock(128),
            )
            self.encoder5 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=2, dilation=2), nn.ReLU(),
                ResidualBlock(256), ResidualBlock(256),
            )
            self.decoder5 = nn.Sequential(
                nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU(),
                ResidualBlock(128), ResidualBlock(128),
            )
            self.decoder4 = nn.Sequential(
                nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU(),
                ResidualBlock(64), ResidualBlock(64),
            )
            self.decoder3 = nn.Sequential(
                nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU(),
                ResidualBlock(32), ResidualBlock(32),
            )
            self.decoder2 = nn.Sequential(
                nn.Conv2d(32 + 16, 16, 3, padding=1), nn.ReLU(),
                ResidualBlock(16), ResidualBlock(16),
            )
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

    return nn, SatelliteRemoverCNN, BinaryClassificationCNN, BinaryClassificationCNN2, tfm


# ----------------------------
#   Loading helpers
# ----------------------------
def _load_model_weights_lenient(torch, nn, model, checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    msd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    return model

_SAT_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

def get_satellite_models(resources: Any = None, use_gpu: bool = True, status_cb=print) -> Dict[str, Any]:
    if resources is None:
        resources = get_resources()

    p_det1 = resources.CC_SAT_DETECT1_PTH
    p_det2 = resources.CC_SAT_DETECT2_PTH
    p_rem  = resources.CC_SAT_REMOVE_PTH
    o_det1 = resources.CC_SAT_DETECT1_ONNX
    o_det2 = resources.CC_SAT_DETECT2_ONNX
    o_rem  = resources.CC_SAT_REMOVE_ONNX

    # Decide whether DirectML is available (no torch required for this path)
    want_dml = bool(use_gpu) and (ort is not None) and ("DmlExecutionProvider" in ort.get_available_providers())

    # Prefer torch cuda if torch exists + cuda is available
    torch = None
    want_cuda = False
    if use_gpu:
        try:
            torch = _get_torch(status_cb=status_cb)
            want_cuda = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception:
            torch = None
            want_cuda = False

    backend = "cuda" if want_cuda else ("dml" if want_dml else "cpu")
    key = (p_det1, p_det2, p_rem, backend)
    if key in _SAT_CACHE:
        return _SAT_CACHE[key]

    if backend == "dml":
        # ORT DirectML sessions
        det1 = ort.InferenceSession(o_det1, providers=["DmlExecutionProvider"])
        det2 = ort.InferenceSession(o_det2, providers=["DmlExecutionProvider"])
        rem  = ort.InferenceSession(o_rem,  providers=["DmlExecutionProvider"])
        out = {
            "detection_model1": det1,
            "detection_model2": det2,
            "removal_model": rem,
            "device": "DirectML",
            "is_onnx": True,
        }
        _SAT_CACHE[key] = out
        status_cb("CosmicClarity Satellite: using DirectML (ONNX Runtime)")
        return out

    # Torch required for cuda/cpu path
    if torch is None:
        torch = _get_torch(status_cb=status_cb)

    device = torch.device("cuda" if (use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        status_cb(f"CosmicClarity Satellite: using CUDA ({torch.cuda.get_device_name(0)})")
    else:
        status_cb("CosmicClarity Satellite: using CPU")

    nn, SatelliteRemoverCNN, BinaryClassificationCNN, BinaryClassificationCNN2, tfm = _build_torch_models(torch)

    det1 = BinaryClassificationCNN(3).to(device)
    det1 = _load_model_weights_lenient(torch, nn, det1, p_det1, device).eval()

    det2 = BinaryClassificationCNN2(3).to(device)
    det2 = _load_model_weights_lenient(torch, nn, det2, p_det2, device).eval()

    rem = SatelliteRemoverCNN().to(device)
    rem = _load_model_weights_lenient(torch, nn, rem, p_rem, device).eval()

    out = {
        "detection_model1": det1,
        "detection_model2": det2,
        "removal_model": rem,
        "device": device,
        "is_onnx": False,
        "torch": torch,
        "tfm": tfm,  # keep the composed resize/toTensor transform cached too
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


def _torch_detect(tile_rgb01: np.ndarray, models: Dict[str, Any]) -> bool:
    torch = models["torch"]
    device = models["device"]
    det1 = models["detection_model1"]
    det2 = models["detection_model2"]
    tfm = models["tfm"]

    inp = tfm(tile_rgb01).unsqueeze(0).to(device)

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

    x = torch.from_numpy(tile_rgb01).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad(), _autocast_context(torch, device):
        out = rem(x).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

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
    if _sk_resize is None:
        raise RuntimeError("skimage.transform.resize is required for ONNX satellite removal path.")
    r = _sk_resize(tile_rgb01, (256, 256, 3), mode="reflect", anti_aliasing=True).astype(np.float32)
    inp = np.transpose(r, (2, 0, 1))[None, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
    pred = np.transpose(out.squeeze(0), (1, 2, 0)).astype(np.float32)
    # resize back to original tile size
    pred2 = _sk_resize(pred, tile_rgb01.shape, mode="reflect", anti_aliasing=True).astype(np.float32)
    return np.clip(pred2, 0.0, 1.0)


def satellite_remove_image(
    image: np.ndarray,
    models: Dict[str, Any],
    *,
    mode: str = "full",            # "full" or "luminance"
    clip_trail: bool = True,
    sensitivity: float = 0.1,
    chunk_size: int = 256,
    overlap: int = 64,
    border_size: int = 16,
    progress_cb: Optional[Callable[[int, int], None]] = None,  # (done, total)
) -> Tuple[np.ndarray, bool]:
    """
    image: input image (any dtype/shape). Expected to be linear-ish in [0..1] for best behavior.
    Returns: (out_image_same_shape_style, trail_detected_any)
    """
    rgb01, was_mono = _ensure_rgb01(image)

    # luminance mode -> process Y only, then merge back
    if mode.lower() == "luminance":
        y, cb, cr = _extract_luminance_bt601(rgb01)
        # treat Y as "mono" but we still run the network as RGB by repeating
        y3 = np.stack([y, y, y], axis=-1)
        out3, detected = _satellite_remove_rgb(
            y3, models,
            clip_trail=clip_trail, sensitivity=sensitivity,
            chunk_size=chunk_size, overlap=overlap, border_size=border_size,
            progress_cb=progress_cb,
        )
        out_y = out3[..., 0]
        out_rgb = _merge_luminance_bt601(out_y, cb, cr)
    else:
        out_rgb, detected = _satellite_remove_rgb(
            rgb01, models,
            clip_trail=clip_trail, sensitivity=sensitivity,
            chunk_size=chunk_size, overlap=overlap, border_size=border_size,
            progress_cb=progress_cb,
        )

    # If original was mono-like, return HxWx1 (matches your SASpro convention for mono docs)
    if (np.asarray(image).ndim == 2) or (np.asarray(image).ndim == 3 and np.asarray(image).shape[2] == 1):
        out_m = out_rgb[..., 0:1].astype(np.float32)
        return out_m, detected

    return out_rgb.astype(np.float32), detected


def _satellite_remove_rgb(
    rgb01: np.ndarray,
    models: Dict[str, Any],
    *,
    clip_trail: bool,
    sensitivity: float,
    chunk_size: int,
    overlap: int,
    border_size: int,
    progress_cb: Optional[Callable[[int, int], None]],
) -> Tuple[np.ndarray, bool]:
    det1 = models["detection_model1"]

    rem  = models["removal_model"]
    is_onnx = bool(models.get("is_onnx", False))


    H, W = rgb01.shape[:2]
    trail_any = False

    # chunk loop
    all_tiles = list(_split_chunks(rgb01, chunk_size, overlap))
    total = len(all_tiles)
    out_tiles = []

    for idx, (tile, y0, x0) in enumerate(all_tiles, start=1):
        orig = tile.astype(np.float32, copy=False)

        if is_onnx:
            detected = _onnx_detect(orig, det1)
            if detected:
                trail_any = True
                pred = _onnx_remove(orig, rem)
                final = _apply_clip_trail_logic(pred, orig, sensitivity) if clip_trail else pred
            else:
                final = orig
        else:
            detected = _torch_detect(orig, models)
            if detected:
                trail_any = True
                pred = _torch_remove(orig, models)
                final = _apply_clip_trail_logic(pred, orig, sensitivity) if clip_trail else pred
            else:
                final = orig

        out_tiles.append((final, y0, x0))

        if progress_cb is not None:
            progress_cb(idx, total)

    out = _stitch_ignore_border(out_tiles, (H, W, 3), border=border_size)

    # Replace border like your standalone replace_border() (keeps edges unchanged)
    if border_size > 0:
        out[:border_size, :, :] = rgb01[:border_size, :, :]
        out[-border_size:, :, :] = rgb01[-border_size:, :, :]
        out[:, :border_size, :] = rgb01[:, :border_size, :]
        out[:, -border_size:, :] = rgb01[:, -border_size:, :]

    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    # If no trail detected, return input unmodified (matches your standalone intent)
    if not trail_any:
        return rgb01.astype(np.float32, copy=False), False

    return out, True
