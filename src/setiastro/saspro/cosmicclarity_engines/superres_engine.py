#src/setiastro/saspro/cosmicclarity_engines/superres_engine.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
cv2.setNumThreads(1)
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


from setiastro.saspro.resources import get_resources

def _load_torch_superres_model(torch, device, pth_path: str):
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        def forward(self, x):
            residual = x
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
            out = out + residual
            return self.relu(out)

    class SuperResolutionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(16))
            self.encoder2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(32))
            self.encoder3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True), ResidualBlock(64))
            self.encoder4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(128))
            self.encoder5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True), ResidualBlock(256))

            self.decoder5 = nn.Sequential(nn.Conv2d(256 + 128, 128, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(128))
            self.decoder4 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(64))
            self.decoder3 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(32))
            self.decoder2 = nn.Sequential(nn.Conv2d(32 + 16, 16, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(16))
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

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"SuperRes model not found: {pth_path}")

    model = SuperResolutionCNN().to(device)
    sd = torch.load(pth_path, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


# ----------------------------
# Shared helpers (copy from denoise_engine if you want)
# ----------------------------
def stretch_image(image: np.ndarray):
    original_min = float(np.min(image))
    stretched = image - original_min

    is_single = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)
    target_median = 0.25

    if is_single:
        med = float(np.median(stretched))
        orig_medians = [med]
        if med != 0:
            stretched = ((med - 1) * target_median * stretched) / (med * (target_median + stretched - 1) - target_median * stretched)
    else:
        orig_medians = []
        for c in range(3):
            med = float(np.median(stretched[..., c]))
            orig_medians.append(med)
            if med != 0:
                stretched[..., c] = ((med - 1) * target_median * stretched[..., c]) / (
                    med * (target_median + stretched[..., c] - 1) - target_median * stretched[..., c]
                )

    return np.clip(stretched, 0, 1).astype(np.float32), original_min, orig_medians


def unstretch_image(image: np.ndarray, original_medians, original_min: float):
    is_single = (image.ndim == 2) or (image.ndim == 3 and image.shape[2] == 1)
    if is_single:
        med = float(np.median(image))
        if med != 0 and original_medians[0] != 0:
            image = ((med - 1) * original_medians[0] * image) / (med * (original_medians[0] + image - 1) - original_medians[0] * image)
    else:
        for c in range(3):
            med = float(np.median(image[..., c]))
            if med != 0 and original_medians[c] != 0:
                image[..., c] = ((med - 1) * original_medians[c] * image[..., c]) / (
                    med * (original_medians[c] + image[..., c] - 1) - original_medians[c] * image[..., c]
                )

    image = image + original_min
    return np.clip(image, 0, 1).astype(np.float32)


def add_border(image: np.ndarray, border_size: int = 16):
    if image.ndim == 2:
        med = float(np.median(image))
        return np.pad(image, ((border_size, border_size), (border_size, border_size)), mode="constant", constant_values=med)
    else:
        meds = np.median(image, axis=(0, 1)).astype(np.float32)
        chans = []
        for c in range(image.shape[2]):
            chans.append(np.pad(image[..., c], ((border_size, border_size), (border_size, border_size)),
                                mode="constant", constant_values=float(meds[c])))
        return np.stack(chans, axis=-1)


def remove_border(image: np.ndarray, border_size: int):
    if image.ndim == 2:
        return image[border_size:-border_size, border_size:-border_size]
    return image[border_size:-border_size, border_size:-border_size, :]


def split_image_into_chunks_with_overlap(image: np.ndarray, chunk_size: int = 256, overlap: int = 64):
    h, w = image.shape[:2]
    step = chunk_size - overlap
    chunks = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            end_i = min(i + chunk_size, h)
            end_j = min(j + chunk_size, w)
            patch = image[i:end_i, j:end_j]
            chunks.append((patch, i, j))
    return chunks


def stitch_chunks_ignore_border(chunks, out_hw: Tuple[int, int], border_size: int = 16):
    H, W = out_hw
    stitched = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for patch, i, j in chunks:
        ph, pw = patch.shape[:2]
        b_h = min(border_size, ph // 2)
        b_w = min(border_size, pw // 2)

        inner = patch[b_h:ph - b_h, b_w:pw - b_w]
        ih, iw = inner.shape[:2]

        stitched[i + b_h:i + b_h + ih, j + b_w:j + b_w + iw] += inner
        weight[i + b_h:i + b_h + ih, j + b_w:j + b_w + iw] += 1.0

    return stitched / np.maximum(weight, 1.0)


# ----------------------------
# Model loading (cached)
# ----------------------------
from typing import Dict, Any, Tuple
import os

_cached: dict[tuple[str, int, bool], dict[str, Any]] = {}
_BACKEND_TAG = "cc_superres"

R = get_resources()

def _superres_paths(scale: int) -> tuple[str, str]:
    if scale == 2:
        return (R.CC_SUPERRES_2X_PTH, R.CC_SUPERRES_2X_ONNX)
    if scale == 3:
        return (R.CC_SUPERRES_3X_PTH, R.CC_SUPERRES_3X_ONNX)
    if scale == 4:
        return (R.CC_SUPERRES_4X_PTH, R.CC_SUPERRES_4X_ONNX)
    raise ValueError("scale must be 2, 3, or 4")


def _pick_backend(torch, use_gpu: bool):
    # Prefer: CUDA (PyTorch) -> DML (ONNX) on Windows -> MPS (PyTorch) -> CPU
    if use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
        return ("pytorch", torch.device("cuda"))
    if use_gpu and ort is not None and ("DmlExecutionProvider" in ort.get_available_providers()):
        return ("onnx", "DirectML")
    if use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ("pytorch", torch.device("mps"))
    return ("pytorch", torch.device("cpu"))


def load_superres(scale: int, use_gpu: bool = True, status_cb=print) -> Dict[str, Any]:
    scale = int(scale)
    if scale not in (2, 3, 4):
        raise ValueError("scale must be 2, 3, or 4")

    is_windows = (os.name == "nt")
    torch = _get_torch(prefer_cuda=bool(use_gpu), prefer_dml=bool(use_gpu and is_windows), status_cb=status_cb)

    pth_path, onnx_path = _superres_paths(scale)

    # --- DEBUG (remove later) ---
    cuda_ok = bool(use_gpu) and hasattr(torch, "cuda") and torch.cuda.is_available()
    dml_ok = bool(use_gpu) and (ort is not None) and ("DmlExecutionProvider" in ort.get_available_providers())

    # ---------------------------

    # IMPORTANT: key should include the ACTUAL selected backend/device, not just use_gpu
    # so you can't get stuck reusing CPU from a previous call.
    # We'll decide backend first, then cache.

    # Prefer torch CUDA if available & allowed (same as sharpen)
    if cuda_ok:
        device = torch.device("cuda")
        status_cb(f"CosmicClarity SuperRes: using CUDA ({torch.cuda.get_device_name(0)})")
        key = (_BACKEND_TAG, scale, "cuda")
        if key in _cached:
            return _cached[key]
        model = _load_torch_superres_model(torch, device, pth_path)
        out = {"backend": "pytorch", "device": device, "model": model, "scale": scale, "torch": torch}
        _cached[key] = out
        return out

    # Torch-DirectML (Windows)
    if use_gpu and is_windows:
        try:
            import torch_directml
            dml = torch_directml.device()
            status_cb("CosmicClarity SuperRes: using DirectML (torch-directml)")
            key = (_BACKEND_TAG, scale, "torch_dml")
            if key in _cached:
                return _cached[key]
            model = _load_torch_superres_model(torch, dml, pth_path)
            out = {"backend": "pytorch", "device": dml, "model": model, "scale": scale, "torch": torch}
            _cached[key] = out
            return out
        except Exception:
            pass


    # DirectML ONNX fallback (Windows)
    if dml_ok:
        status_cb("CosmicClarity SuperRes: using DirectML (ONNX Runtime)")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"SuperRes ONNX model not found: {onnx_path}")
        key = (_BACKEND_TAG, scale, "dml")
        if key in _cached:
            return _cached[key]
        sess = ort.InferenceSession(onnx_path, providers=["DmlExecutionProvider"])
        out = {"backend": "onnx", "device": "DirectML", "model": sess, "scale": scale, "torch": None}
        _cached[key] = out
        return out

    # MPS (mac)
    if bool(use_gpu) and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        status_cb("CosmicClarity SuperRes: using MPS")
        key = (_BACKEND_TAG, scale, "mps")
        if key in _cached:
            return _cached[key]
        model = _load_torch_superres_model(torch, device, pth_path)
        out = {"backend": "pytorch", "device": device, "model": model, "scale": scale, "torch": torch}
        _cached[key] = out
        return out

    # CPU
    device = torch.device("cpu")
    status_cb("CosmicClarity SuperRes: using CPU")
    key = (_BACKEND_TAG, scale, "cpu")
    if key in _cached:
        return _cached[key]
    model = _load_torch_superres_model(torch, device, pth_path)
    out = {"backend": "pytorch", "device": device, "model": model, "scale": scale, "torch": torch}
    _cached[key] = out
    return out


def _amp_ok(torch, device) -> bool:
    if not isinstance(device, torch.device) or device.type != "cuda":
        return False
    try:
        props = torch.cuda.get_device_properties(device)
        return props.major >= 8
    except Exception:
        return False


def superres_rgb01(
    img_rgb01: np.ndarray,
    *,
    scale: int,
    use_gpu: bool = True,
    progress_cb=None,  # progress_cb(done:int,total:int)
) -> np.ndarray:
    """
    Input: float32 RGB in [0..1], shape (H,W,3)
    Output: float32 RGB in [0..1], shape (H*scale,W*scale,3)
    """
    scale = int(scale)
    if scale not in (2, 3, 4):
        raise ValueError("scale must be 2, 3, or 4")

    engine = load_superres(scale, use_gpu=use_gpu, status_cb=print)  # or your logger

    # We process each channel independently (matches your current behavior)
    H, W = img_rgb01.shape[:2]
    out_chans = []

    # progress accounting: per-channel chunks
    for c in range(3):
        chan = img_rgb01[..., c].astype(np.float32, copy=False)

        # border + optional stretch
        bordered = add_border(chan, border_size=16)
        if float(np.median(bordered)) < 0.08:
            stretched, orig_min, orig_meds = stretch_image(bordered)
            stretched_applied = True
        else:
            stretched = bordered.astype(np.float32, copy=False)
            stretched_applied = False
            orig_min = float(np.min(bordered))
            orig_meds = [float(np.median(bordered))]

        # bicubic upscale
        h, w = stretched.shape[:2]
        up = cv2.resize(stretched, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # chunk & infer
        chunks = split_image_into_chunks_with_overlap(up, chunk_size=256, overlap=64)
        total = len(chunks)
        done0 = 0

        processed = []
        use_amp = (engine["backend"] == "pytorch") and _amp_ok(engine["torch"], engine["device"])
        dev = engine["device"]
        dev_type = getattr(dev, "type", None)
        for idx, (patch, i, j) in enumerate(chunks):
            ph, pw = patch.shape[:2]

            # build 256x256x3 patch
            patch_in = np.zeros((256, 256, 3), dtype=np.float32)
            patch_in[:ph, :pw, 0] = patch[:ph, :pw]
            patch_in[:ph, :pw, 1] = patch[:ph, :pw]
            patch_in[:ph, :pw, 2] = patch[:ph, :pw]

            if engine["backend"] == "pytorch":
                t = engine["torch"]  # torch module from runtime_torch

                pt = t.from_numpy(patch_in.transpose(2, 0, 1)).unsqueeze(0).to(engine["device"])

                with t.no_grad():
                    if use_amp and dev_type == "cuda":
                        with t.cuda.amp.autocast():
                            out = engine["model"](pt)
                    else:
                        out = engine["model"](pt)
                out_np = out[0].detach().cpu().numpy()   # (C,H,W)
            else:
                # ONNX (DirectML)
                inp = np.expand_dims(patch_in.transpose(2, 0, 1), axis=0).astype(np.float32)
                out_np = engine["model"].run(None, {engine["model"].get_inputs()[0].name: inp})[0].squeeze()

            # output is 3ch grayscale; take first channel
            if out_np.ndim == 3 and out_np.shape[0] == 3:
                out_np = out_np[0]
            elif out_np.ndim == 3 and out_np.shape[-1] == 3:
                out_np = out_np[..., 0]

            out_np = out_np[:ph, :pw].astype(np.float32, copy=False)
            processed.append((out_np, i, j))

            done0 += 1
            if progress_cb is not None:
                # You can interpret as global progress across all channels:
                progress_cb((c * total) + done0, 3 * total)

        # stitch
        stitched = stitch_chunks_ignore_border(processed, up.shape[:2], border_size=16)

        # unstretch if needed
        if stretched_applied:
            stitched = unstretch_image(stitched, orig_meds, orig_min)

        # remove scaled border: 16px border became 16*scale after upscaling
        final_border = int(16 * scale)
        out_chan = remove_border(stitched, border_size=final_border)

        out_chans.append(out_chan)

    out_rgb = np.stack(out_chans, axis=-1)
    return np.clip(out_rgb, 0.0, 1.0).astype(np.float32, copy=False)
