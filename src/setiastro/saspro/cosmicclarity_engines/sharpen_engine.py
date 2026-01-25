# src/setiastro/saspro/cosmicclarity_engines/sharpen_engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any
import os
import numpy as np
from setiastro.saspro.resources import get_resources


# Optional deps used by auto-PSF
try:
    import sep
except Exception:
    sep = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


ProgressCB = Callable[[int, int, str], None]  # (done, total, stage)


# ---------------- Torch model defs (needed for .pth) ----------------
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
    Keep your ">= 8.0" rule, but note: with your NaN reports you may later
    want a runtime policy that can disable autocast per session.
    """
    try:
        if hasattr(device, "type") and device.type == "cuda":
            major, minor = torch.cuda.get_device_capability()
            cap = float(f"{major}.{minor}")
            if cap >= 8.0:
                return torch.cuda.amp.autocast()
    except Exception:
        pass
    return _nullcontext()


def _to_3ch(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (img3, was_mono). img3 is HxWx3 float32."""
    if image.ndim == 2:
        img3 = np.stack([image, image, image], axis=-1)
        return img3, True
    if image.ndim == 3 and image.shape[2] == 1:
        img = image[..., 0]
        img3 = np.stack([img, img, img], axis=-1)
        return img3, True
    return image, False


# Your BT.601 luminance extraction / merge
def extract_luminance_rgb(image_rgb: np.ndarray):
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    if image_rgb.shape[-1] != 3:
        raise ValueError("extract_luminance_rgb expects HxWx3")
    M = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]], dtype=np.float32)
    ycbcr = image_rgb @ M.T
    y = ycbcr[..., 0]
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


# ---------------- Chunking & stitching (your exact behavior) ----------------

def split_image_into_chunks_with_overlap(image2d: np.ndarray, chunk_size: int, overlap: int):
    H, W = image2d.shape
    step = chunk_size - overlap
    out = []
    for i in range(0, H, step):
        for j in range(0, W, step):
            ei = min(i + chunk_size, H)
            ej = min(j + chunk_size, W)
            chunk = image2d[i:ei, j:ej]
            is_edge = (i == 0) or (j == 0) or (i + chunk_size >= H) or (j + chunk_size >= W)
            out.append((chunk, i, j, is_edge))
    return out


def stitch_chunks_ignore_border(chunks, image_shape, border_size: int = 16):
    stitched = np.zeros(image_shape, dtype=np.float32)
    weights  = np.zeros(image_shape, dtype=np.float32)

    for chunk, i, j, _is_edge in chunks:
        h, w = chunk.shape
        bh = min(border_size, h // 2)
        bw = min(border_size, w // 2)
        inner = chunk[bh:h-bh, bw:w-bw]
        stitched[i+bh:i+h-bh, j+bw:j+w-bw] += inner
        weights[i+bh:i+h-bh,  j+bw:j+w-bw] += 1.0

    stitched /= np.maximum(weights, 1.0)
    return stitched


def add_border(image: np.ndarray, border_size: int = 16) -> np.ndarray:
    med = float(np.median(image))
    if image.ndim == 2:
        return np.pad(image, ((border_size, border_size), (border_size, border_size)),
                      mode="constant", constant_values=med)
    return np.pad(image, ((border_size, border_size), (border_size, border_size), (0, 0)),
                  mode="constant", constant_values=med)


def remove_border(image: np.ndarray, border_size: int = 16) -> np.ndarray:
    if image.ndim == 2:
        return image[border_size:-border_size, border_size:-border_size]
    return image[border_size:-border_size, border_size:-border_size, :]


def blend_images(before: np.ndarray, after: np.ndarray, amount: float) -> np.ndarray:
    a = float(np.clip(amount, 0.0, 1.0))
    return (1.0 - a) * before + a * after


# ---------------- Stretch / unstretch (your current logic) ----------------

def stretch_image_unlinked_rgb(image_rgb: np.ndarray, target_median: float = 0.25):
    x = image_rgb.astype(np.float32, copy=True)
    orig_min = float(np.min(x))
    x -= orig_min
    orig_meds = [float(np.median(x[..., c])) for c in range(3)]

    for c in range(3):
        m = orig_meds[c]
        if m != 0:
            x[..., c] = ((m - 1) * target_median * x[..., c]) / (
                m * (target_median + x[..., c] - 1) - target_median * x[..., c]
            )
    x = np.clip(x, 0, 1)
    return x, orig_min, orig_meds


def unstretch_image_unlinked_rgb(image_rgb: np.ndarray, orig_meds, orig_min: float, was_mono: bool):
    x = image_rgb.astype(np.float32, copy=True)
    for c in range(3):
        m_now = float(np.median(x[..., c]))
        m0 = float(orig_meds[c])
        if m_now != 0 and m0 != 0:
            x[..., c] = ((m_now - 1) * m0 * x[..., c]) / (
                m_now * (m0 + x[..., c] - 1) - m0 * x[..., c]
            )
    x += float(orig_min)
    x = np.clip(x, 0, 1)
    if was_mono:
        # match your behavior: return mono with keepdims
        x = np.mean(x, axis=2, keepdims=True)
    return x


# ---------------- Auto PSF per chunk (SEP) ----------------

def measure_psf_fwhm(chunk2d: np.ndarray, default_fwhm: float = 3.0) -> float:
    if sep is None:
        return default_fwhm
    try:
        data = chunk2d.astype(np.float32, copy=False)
        bkg = sep.Background(data)
        sub = data - bkg.back()
        rms = bkg.rms()
        if rms.size == 0:
            return default_fwhm
        objs = sep.extract(sub, 1.5, err=rms)
        fwhms = []
        for o in objs:
            if o["npix"] < 5:
                continue
            sigma = float(np.sqrt(o["a"] * o["b"]))
            fwhm = sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))
            fwhms.append(fwhm)
        return float(np.median(fwhms) * 0.5) if fwhms else default_fwhm
    except Exception:
        return default_fwhm


# ---------------- Model bundle + loading ----------------

@dataclass
class SharpenModels:
    device: Any                 # torch.device or "DirectML"
    is_onnx: bool
    stellar: Any
    ns1: Any
    ns2: Any
    ns4: Any
    ns8: Any
    torch: Any | None = None    # set for torch path


_MODELS_CACHE: dict[tuple[str, bool], SharpenModels] = {}  # (backend_tag, use_gpu)

def load_sharpen_models(use_gpu: bool, status_cb=print) -> SharpenModels:
    backend_tag = "cc_sharpen_ai3_5s"
    key = (backend_tag, bool(use_gpu))
    if key in _MODELS_CACHE:
        return _MODELS_CACHE[key]

    is_windows = (os.name == "nt")

    # ask runtime to prefer DML only on Windows + when user wants GPU
    torch = _get_torch(
        prefer_cuda=bool(use_gpu),                 # still try CUDA first
        prefer_dml=bool(use_gpu and is_windows),   # enable DML install/usage path
        status_cb=status_cb,
    )

    # 1) CUDA
    if use_gpu and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
        status_cb(f"CosmicClarity Sharpen: using CUDA ({torch.cuda.get_device_name(0)})")
        models = _load_torch_models(torch, device)
        _MODELS_CACHE[key] = models
        return models

    # 2) Torch-DirectML (Windows)
    if use_gpu and is_windows:
        try:
            import torch_directml  # provided by torch-directml
            dml = torch_directml.device()
            status_cb("CosmicClarity Sharpen: using DirectML (torch-directml)")
            models = _load_torch_models(torch, dml)
            _MODELS_CACHE[key] = models
            status_cb(f"torch.__version__={getattr(torch,'__version__',None)}; "
                    f"cuda_available={bool(getattr(torch,'cuda',None) and torch.cuda.is_available())}")
            return models
                    
        except Exception:
            pass

    # 3) ONNX Runtime DirectML fallback
    if use_gpu and ort is not None and "DmlExecutionProvider" in ort.get_available_providers():
        status_cb("CosmicClarity Sharpen: using DirectML (ONNX Runtime)")
        models = _load_onnx_models()
        _MODELS_CACHE[key] = models
        return models

    # 4) CPU
    device = torch.device("cpu")
    status_cb("CosmicClarity Sharpen: using CPU")
    models = _load_torch_models(torch, device)
    _MODELS_CACHE[key] = models
    status_cb(f"Sharpen backend resolved: "
            f"{'onnx' if models.is_onnx else 'torch'} / device={models.device!r}")

    return models


def _load_onnx_models() -> SharpenModels:
    assert ort is not None
    prov = ["DmlExecutionProvider"]
    R = get_resources()

    def s(path: str):
        return ort.InferenceSession(path, providers=prov)

    return SharpenModels(
        device="DirectML",
        is_onnx=True,
        stellar=s(R.CC_STELLAR_SHARP_ONNX),
        ns1=s(R.CC_NS1_ONNX),
        ns2=s(R.CC_NS2_ONNX),
        ns4=s(R.CC_NS4_ONNX),
        ns8=s(R.CC_NS8_ONNX),
        torch=None,
    )



def _load_torch_models(torch, device) -> SharpenModels:
    import torch.nn as nn  # comes from runtime torch env

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        def forward(self, x):
            r = x
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            x = self.relu(x + r)
            return x

    class SharpeningCNN(nn.Module):
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

    R = get_resources()

    def m(path: str):
        net = SharpeningCNN()
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval().to(device)
        return net

    return SharpenModels(
        device=device,
        is_onnx=False,
        stellar=m(R.CC_STELLAR_SHARP_PTH),
        ns1=m(R.CC_NS1_PTH),
        ns2=m(R.CC_NS2_PTH),
        ns4=m(R.CC_NS4_PTH),
        ns8=m(R.CC_NS8_PTH),
        torch=torch,
    )



# ---------------- Inference helpers ----------------

def _infer_chunk(models: SharpenModels, model: Any, chunk2d: np.ndarray) -> np.ndarray:
    """Returns 2D float32 (cropped to original chunk shape)."""
    h0, w0 = chunk2d.shape

    if models.is_onnx:
        inp = chunk2d[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,H,W)
        inp = np.tile(inp, (1, 3, 1, 1))                                # (1,3,H,W)
        h, w = inp.shape[2:]
        if (h != 256) or (w != 256):
            pad = np.zeros((1, 3, 256, 256), dtype=np.float32)
            pad[:, :, :h, :w] = inp
            inp = pad
        name_in  = model.get_inputs()[0].name
        name_out = model.get_outputs()[0].name
        out = model.run([name_out], {name_in: inp})[0][0, 0]
        return out[:h0, :w0].astype(np.float32, copy=False)

    # torch path
    torch = models.torch
    dev = models.device
    t = torch.tensor(chunk2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad(), _autocast_context(torch, dev):
        y = model(t.repeat(1, 3, 1, 1)).squeeze().detach().cpu().numpy()[0]
    return y[:h0, :w0].astype(np.float32, copy=False)


# ---------------- Main API ----------------

@dataclass
class SharpenParams:
    mode: str                         # "Both" | "Stellar Only" | "Non-Stellar Only"
    stellar_amount: float             # 0..1
    nonstellar_amount: float          # 0..1
    nonstellar_strength: float        # 1..8 (ignored if auto_detect_psf True)
    sharpen_channels_separately: bool
    auto_detect_psf: bool
    use_gpu: bool


def sharpen_image_array(image: np.ndarray,
                        params: SharpenParams,
                        progress_cb: Optional[ProgressCB] = None,
                        status_cb=print) -> tuple[np.ndarray, bool]:
    """
    Pure in-memory sharpen. Returns (out_image, was_mono).
    """
    if progress_cb is None:
        progress_cb = lambda done, total, stage: None

    img = np.asarray(image)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)

    img3, was_mono = _to_3ch(img)
    img3 = np.clip(img3, 0.0, 1.0)

    models = load_sharpen_models(use_gpu=params.use_gpu, status_cb=status_cb)

    # border & stretch
    bordered = add_border(img3, border_size=16)
    stretch_needed = (np.median(bordered - np.min(bordered)) < 0.08)

    if stretch_needed:
        stretched, orig_min, orig_meds = stretch_image_unlinked_rgb(bordered)
    else:
        stretched, orig_min, orig_meds = bordered, None, None

    # per-channel sharpening option (color only)
    if params.sharpen_channels_separately and (not was_mono):
        out = np.empty_like(stretched)
        for c, label in enumerate(("R", "G", "B")):
            progress_cb(0, 1, f"Sharpening {label} channel")
            out[..., c] = _sharpen_plane(models, stretched[..., c], params, progress_cb)
        sharpened = out
    else:
        # luminance pipeline (works for mono too, since mono is in all 3 chans)
        y, cb, cr = extract_luminance_rgb(stretched)
        y2 = _sharpen_plane(models, y, params, progress_cb)
        sharpened = merge_luminance(y2, cb, cr)

    # unstretch / deborder
    if stretch_needed:
        sharpened = unstretch_image_unlinked_rgb(sharpened, orig_meds, orig_min, was_mono)

    sharpened = remove_border(sharpened, border_size=16)

    # return mono as HxWx1 if it came in mono (matches your CC behavior)
    if was_mono:
        if sharpened.ndim == 3 and sharpened.shape[2] == 3:
            sharpened = np.mean(sharpened, axis=2, keepdims=True).astype(np.float32, copy=False)

    return np.clip(sharpened, 0.0, 1.0), was_mono


def _sharpen_plane(models: SharpenModels,
                   plane: np.ndarray,
                   params: SharpenParams,
                   progress_cb: ProgressCB) -> np.ndarray:
    """
    Sharpen a single 2D plane using your two-stage pipeline.
    """
    plane = np.asarray(plane, np.float32)
    chunks = split_image_into_chunks_with_overlap(plane, chunk_size=256, overlap=64)
    total = len(chunks)

    # Stage 1: stellar
    if params.mode in ("Stellar Only", "Both"):
        out_chunks = []
        for k, (chunk, i, j, is_edge) in enumerate(chunks, start=1):
            y = _infer_chunk(models, models.stellar, chunk)
            blended = blend_images(chunk, y, params.stellar_amount)
            out_chunks.append((blended, i, j, is_edge))
            progress_cb(k, total, "Stellar sharpening")
        plane = stitch_chunks_ignore_border(out_chunks, plane.shape, border_size=16)

        if params.mode == "Stellar Only":
            return plane

        # update chunks for stage 2
        chunks = split_image_into_chunks_with_overlap(plane, chunk_size=256, overlap=64)
        total = len(chunks)

    # Stage 2: non-stellar
    if params.mode in ("Non-Stellar Only", "Both"):
        out_chunks = []
        radii = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
        model_map = {1.0: models.ns1, 2.0: models.ns2, 4.0: models.ns4, 8.0: models.ns8}

        for k, (chunk, i, j, is_edge) in enumerate(chunks, start=1):
            if params.auto_detect_psf:
                fwhm = measure_psf_fwhm(chunk, default_fwhm=3.0)
                r = float(np.clip(fwhm, radii[0], radii[-1]))
            else:
                r = float(np.clip(params.nonstellar_strength, radii[0], radii[-1]))

            idx = int(np.searchsorted(radii, r, side="left"))
            if idx <= 0:
                lo = hi = radii[0]
            elif idx >= len(radii):
                lo = hi = radii[-1]
            else:
                lo, hi = radii[idx-1], radii[idx]

            if lo == hi:
                y = _infer_chunk(models, model_map[lo], chunk)
            else:
                w = (r - lo) / (hi - lo)
                y0 = _infer_chunk(models, model_map[lo], chunk)
                y1 = _infer_chunk(models, model_map[hi], chunk)
                y = (1.0 - w) * y0 + w * y1

            blended = blend_images(chunk, y, params.nonstellar_amount)
            out_chunks.append((blended, i, j, is_edge))
            progress_cb(k, total, "Non-stellar sharpening")

        plane = stitch_chunks_ignore_border(out_chunks, plane.shape, border_size=16)

    return plane

def sharpen_rgb01(
    image_rgb01: np.ndarray,
    *,
    sharpening_mode: str = "Both",
    stellar_amount: float = 0.5,
    nonstellar_amount: float = 0.5,
    nonstellar_strength: float = 3.0,
    auto_detect_psf: bool = True,
    separate_channels: bool = False,
    use_gpu: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb=print,
) -> np.ndarray:
    """
    Backward-compatible API for SASpro CosmicClarityDialogPro.
    Expects/returns float32 RGB in [0,1]. If input is mono, returns HxWx1.
    progress_cb signature: (done, total)  -> UI-friendly.
    """
    # Adapt UI progress_cb(done,total) -> engine progress_cb(done,total,stage)
    if progress_cb is None:
        def _prog(done, total, stage):  # noqa
            return
    else:
        def _prog(done, total, stage):
            try:
                progress_cb(int(done), int(total))
            except Exception:
                pass

    params = SharpenParams(
        mode=str(sharpening_mode),
        stellar_amount=float(stellar_amount),
        nonstellar_amount=float(nonstellar_amount),
        nonstellar_strength=float(nonstellar_strength),
        sharpen_channels_separately=bool(separate_channels),
        auto_detect_psf=bool(auto_detect_psf),
        use_gpu=bool(use_gpu),
    )

    out, _was_mono = sharpen_image_array(
        image_rgb01,
        params=params,
        progress_cb=_prog,
        status_cb=status_cb,
    )
    return np.asarray(out, dtype=np.float32)
