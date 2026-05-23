# src/setiastro/saspro/cosmicclarity_engines/sharpen_engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any
import os
import numpy as np
from setiastro.saspro.resources import get_resources
from setiastro.saspro.runtime_torch import _user_runtime_dir, _venv_paths, _check_cuda_in_venv
import math

# Optional deps used by auto-PSF
try:
    import sep
except Exception:
    sep = None


import sys, time, tempfile, traceback

_DEBUG_SHARPEN = True  # flip False later

_NSCOND_NAFNET = dict(
    width=32,
    enc_blk_nums=(2, 4, 6, 8),
    dec_blk_nums=(2, 2, 2, 2),
    middle_blk_num=4,
)

_SCOND_NAFNET = dict(
    width=32,
    enc_blk_nums=(2, 4, 6, 8),
    dec_blk_nums=(2, 2, 2, 2),
    middle_blk_num=4,
)

# Aberration correction model — wider middle than sharpening (6 vs 4)
_CORRECT_NAFNET = dict(
    width=32,
    enc_blk_nums=(2, 4, 6, 8),
    dec_blk_nums=(2, 2, 2, 2),
    middle_blk_num=6,
)

def _dbg(msg: str, status_cb=print):
    pass


ProgressCB = Callable[[int, int, str], bool]  # True=continue, False=cancel


# ---------------- Torch model defs (needed for .pth) ----------------
def _get_torch(*, prefer_cuda: bool, prefer_dml: bool, status_cb=print):
    from setiastro.saspro.runtime_torch import import_torch
    return import_torch(
        prefer_cuda=prefer_cuda,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=status_cb,
    )

def _get_ort(status_cb=print):
    try:
        import onnxruntime as ort
        return ort
    except Exception as e:
        try:
            status_cb(f"CosmicClarity Sharpen: onnxruntime not available ({type(e).__name__}: {e})")
        except Exception:
            pass
        return None


def _nullcontext():
    from contextlib import nullcontext
    return nullcontext()


def _autocast_context(torch, device) -> Any:
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


# ---------------- Chunking & stitching ----------------

def split_image_into_chunks_with_overlap(image2d: np.ndarray, chunk_size: int, overlap: int):
    H, W = image2d.shape
    step = chunk_size - overlap
    out = []
    for i in range(0, H, step):
        for j in range(0, W, step):
            ei = min(i + chunk_size, H)
            ej = min(j + chunk_size, W)
            if ei <= i or ej <= j:
                continue
            chunk = image2d[i:ei, j:ej]
            is_edge = (i == 0) or (j == 0) or (i + chunk_size >= H) or (j + chunk_size >= W)
            out.append((chunk, i, j, is_edge))
    return out


def stitch_chunks_ignore_border(chunks, image_shape, border_size: int = 16):
    H, W = image_shape
    stitched = np.zeros((H, W), dtype=np.float32)
    weights  = np.zeros((H, W), dtype=np.float32)

    for chunk, i, j, _is_edge in chunks:
        h, w = chunk.shape
        if h <= 0 or w <= 0:
            continue

        bh = min(border_size, h // 2)
        bw = min(border_size, w // 2)

        y0 = i + bh;  y1 = i + h - bh
        x0 = j + bw;  x1 = j + w - bw

        if y1 <= y0 or x1 <= x0:
            continue

        inner = chunk[bh:h-bh, bw:w-bw]

        yy0 = max(0, y0);  yy1 = min(H, y1)
        xx0 = max(0, x0);  xx1 = min(W, x1)

        if yy1 <= yy0 or xx1 <= xx0:
            continue

        sy0 = yy0 - y0;  sy1 = sy0 + (yy1 - yy0)
        sx0 = xx0 - x0;  sx1 = sx0 + (xx1 - xx0)

        src = inner[sy0:sy1, sx0:sx1]
        stitched[yy0:yy1, xx0:xx1] += src
        weights[yy0:yy1,  xx0:xx1] += 1.0

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


# ---------------- Stretch / unstretch ----------------

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
        x = np.mean(x, axis=2, keepdims=True)
    return x


# ---------------- Auto PSF ----------------

def measure_psf_radius(chunk2d: np.ndarray, default_radius: float = 3.0) -> float:
    if sep is None:
        return default_radius
    try:
        data = chunk2d.astype(np.float32, copy=False)
        bkg = sep.Background(data)
        sub = data - bkg.back()
        rms = bkg.rms()
        if rms.size == 0:
            return default_radius
        objs = sep.extract(sub, 1.5, err=rms)
        radii = []
        for o in objs:
            if o["npix"] < 5:
                continue
            sigma = float(np.sqrt(o["a"] * o["b"]))
            fwhm = sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))
            radii.append(fwhm * 0.5)
        return float(np.median(radii)) if radii else default_radius
    except Exception:
        return default_radius


def _is_device_lost_error(err: Exception) -> bool:
    s = f"{type(err).__name__}: {err}".lower()
    needles = ["887a0005", "getdeviceremovedreason", "device removed", "device lost",
               "gpu-apparaat", "onderbroken", "dmlexecutionprovider", "executionprovider.cpp"]
    return any(n in s for n in needles)


def _iter_batch_sizes(initial: int):
    bs = max(1, int(initial))
    yielded = set()
    while bs >= 1:
        if bs not in yielded:
            yielded.add(bs)
            yield bs
        if bs == 1:
            break
        bs = max(1, bs // 2)


def _is_recoverable_batch_error(err: Exception) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    needles = ["out of memory", "cuda", "cudnn", "cublas", "directml", "dml", "mps",
               "allocation", "alloc", "resource", "execution provider", "insufficient memory",
               "bad allocation", "memory", "887a0005", "device removed", "device lost"]
    return any(n in msg for n in needles)


# ---------------- Model bundle ----------------

@dataclass
class SharpenModels:
    device: Any
    is_onnx: bool
    stellar: Any
    ns1: Any
    ns2: Any
    ns4: Any
    ns8: Any
    ns_cond: Any | None = None
    stellar_cond: bool = False
    correct: Any | None = None        # ← aberration correction model (NAFNetCorrect)
    torch: Any | None = None


def _pad2d_to_multiple(x: np.ndarray, mult: int = 16, mode: str = "reflect") -> tuple[np.ndarray, int, int]:
    h, w = x.shape
    ph = (mult - (h % mult)) % mult
    pw = (mult - (w % mult)) % mult
    if ph == 0 and pw == 0:
        return x, h, w
    xp = np.pad(x, ((0, ph), (0, pw)), mode=mode)
    return xp, h, w


_MODELS_CACHE: dict[tuple[str, str], SharpenModels] = {}


def load_sharpen_models(use_gpu: bool, status_cb=print) -> SharpenModels:
    backend_tag = "cc_sharpen_ai4"
    is_windows = (os.name == "nt")
    _dbg(f"ENTER load_sharpen_models(use_gpu={use_gpu})", status_cb=status_cb)

    t0 = time.time()
    try:
        torch = _get_torch(
            prefer_cuda=bool(use_gpu),
            prefer_dml=bool(use_gpu) and (os.name == "nt"),
            status_cb=status_cb,
        )
    except Exception as e:
        _dbg("ERROR in _get_torch:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
             status_cb=status_cb)
        raise
    ort = _get_ort(status_cb=status_cb)

    try:
        rt = _user_runtime_dir()
        vpy = _venv_paths(rt)["python"]
        ok, cuda_tag, err = _check_cuda_in_venv(vpy, status_cb=status_cb)
    except Exception as e:
        _dbg("Runtime CUDA probe FAILED:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
             status_cb=status_cb)

    if use_gpu:
        try:
            cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        except Exception:
            cuda_ok = False

        if cuda_ok:
            cache_key = (backend_tag, "cuda")
            if cache_key in _MODELS_CACHE:
                return _MODELS_CACHE[cache_key]
            try:
                device = torch.device("cuda")
                models = _load_torch_models(torch, device)
                _MODELS_CACHE[cache_key] = models
                return models
            except Exception as e:
                _dbg("CUDA path failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                     status_cb=status_cb)

    if use_gpu:
        try:
            mps_ok = bool(
                hasattr(torch, "backends")
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except Exception:
            mps_ok = False

        if mps_ok:
            cache_key = (backend_tag, "mps")
            if cache_key in _MODELS_CACHE:
                return _MODELS_CACHE[cache_key]
            try:
                device = torch.device("mps")
                models = _load_torch_models(torch, device)
                _MODELS_CACHE[cache_key] = models
                return models
            except Exception as e:
                _dbg("MPS path failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                     status_cb=status_cb)

    if use_gpu and is_windows:
        try:
            import torch_directml
            cache_key = (backend_tag, "dml_torch")
            if cache_key in _MODELS_CACHE:
                return _MODELS_CACHE[cache_key]
            dml = torch_directml.device()
            _ = (torch.ones(1, device=dml) + 1).to("cpu").item()
            models = _load_torch_models(torch, dml)
            _MODELS_CACHE[cache_key] = models
            return models
        except Exception as e:
            _dbg("DirectML (torch-directml) failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                 status_cb=status_cb)

    if use_gpu and ort is not None:
        try:
            prov = ort.get_available_providers()
            if "DmlExecutionProvider" in prov:
                cache_key = (backend_tag, "dml_ort")
                if cache_key in _MODELS_CACHE:
                    return _MODELS_CACHE[cache_key]
                models = _load_onnx_models(ort)
                _MODELS_CACHE[cache_key] = models
                return models
        except Exception as e:
            _dbg("ORT provider check/load failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                 status_cb=status_cb)

    cache_key = (backend_tag, "cpu")
    if cache_key in _MODELS_CACHE:
        return _MODELS_CACHE[cache_key]
    try:
        device = torch.device("cpu")
        models = _load_torch_models(torch, device)
        _MODELS_CACHE[cache_key] = models
        return models
    except Exception as e:
        _dbg("CPU model load failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
             status_cb=status_cb)
        raise


def _load_onnx_models(ort) -> SharpenModels:
    prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
    R = get_resources()

    def s(path: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        return ort.InferenceSession(path, sess_options=so, providers=prov)

    stellar  = s(R.CC_STELLAR_NAF_ONNX)
    img_name, psf_name, out_name = _ort_pick_io_names(stellar)
    if psf_name is not None:
        raise RuntimeError("Stellar ONNX unexpectedly has a PSF input.")
    ns_cond = s(R.CC_NS_COND_NAF_ONNX)

    # Correction model — no ONNX export yet, leave as None for ONNX path
    return SharpenModels(
        device="DirectML",
        is_onnx=True,
        stellar=stellar,
        ns1=None, ns2=None, ns4=None, ns8=None,
        ns_cond=ns_cond,
        stellar_cond=False,
        correct=None,
        torch=None,
    )


def _ort_pick_io_names(session) -> tuple[str, Optional[str], str]:
    ins = session.get_inputs()
    out = session.get_outputs()[0].name
    img_name = None
    psf_name = None
    for i in ins:
        shp  = i.shape
        rank = len(shp) if shp is not None else 0
        if rank == 4:
            img_name = i.name
        elif rank in (1, 2):
            psf_name = i.name
    if img_name is None:
        img_name = ins[0].name
    if len(ins) == 1:
        psf_name = None
    elif psf_name is None:
        psf_name = ins[1].name
    return img_name, psf_name, out


def _load_torch_models(torch, device) -> SharpenModels:
    import torch.nn as nn

    # ---- Shared block definitions ----
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
            self.norm1  = LayerNorm2d(channels)
            self.conv1  = nn.Conv2d(channels, channels * 2, 1, bias=True)
            self.dwconv = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=True)
            self.sg     = SimpleGate()
            self.sca    = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1, bias=True))
            self.conv2  = nn.Conv2d(channels, channels, 1, bias=True)
            self.norm2  = LayerNorm2d(channels)
            self.ffn1   = nn.Conv2d(channels, channels * 2, 1, bias=True)
            self.ffn2   = nn.Conv2d(channels, channels, 1, bias=True)
            self.beta   = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma  = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            y = self.norm1(x); y = self.conv1(y); y = self.dwconv(y)
            y = self.sg(y);    y = y * self.sca(y); y = self.conv2(y)
            x = x + y * self.beta
            y = self.norm2(x); y = self.ffn1(y); y = self.sg(y); y = self.ffn2(y)
            x = x + y * self.gamma
            return x

    # ---- Sharpening model (unchanged) ----
    class NAFNetSharpen(nn.Module):
        def __init__(self, in_ch=3, out_ch=3, width=32,
                     enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                     middle_blk_num=4, residual_out=True, clamp_out=False):
            super().__init__()
            self.intro    = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
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
            self.ending       = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
            self.residual_out = bool(residual_out)
            self.clamp_out    = bool(clamp_out)

        def forward_delta(self, x):
            x = self.intro(x); skips = []
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

    class NAFNetSharpenPSF(NAFNetSharpen):
        def __init__(self, **kw):
            super().__init__(in_ch=4, out_ch=3, **kw)
        def forward(self, x_rgb, psf_t):
            b, _, h, w = x_rgb.shape
            psf_map = psf_t.view(b, 1, 1, 1).expand(b, 1, h, w)
            x4 = torch.cat([x_rgb, psf_map], dim=1)
            delta = self.forward_delta(x4)
            y = x_rgb + delta if self.residual_out else delta
            if self.clamp_out:
                y = y.clamp(0.0, 1.0)
            return y

    # ---- Aberration correction model ----
    # Identical U-Net family but middle_blk_num=6 and trained for PSF correction.
    # Loaded from deep_correct_stellar_AI4.deploy.pth (or .pth).
    class NAFNetCorrect(nn.Module):
        """
        Stellar aberration correction model.
        Residual output: corrected = input + delta.
        middle_blk_num=6 — extra capacity for cross-channel PSF reasoning.
        Accepts RGB input; also handles mono (triplicated channel) transparently.
        """
        def __init__(self, in_ch=3, out_ch=3, width=32,
                     enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                     middle_blk_num=6, residual_out=True):
            super().__init__()
            self.intro    = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
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
            self.ending       = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)
            self.residual_out = bool(residual_out)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x0 = x; y = self.intro(x); skips = []
            for enc, down in zip(self.encoders, self.downs):
                y = enc(y); skips.append(y); y = down(y)
            y = self.middle(y)
            for up, dec in zip(self.ups, self.decoders):
                y = up(y); y = y + skips.pop(); y = dec(y)
            delta = self.ending(y)
            return (x0 + delta) if self.residual_out else delta

    R = get_resources()

    def m_naf_rgb(path: str, cfg: dict):
        net = NAFNetSharpen(**cfg, residual_out=True, clamp_out=False)
        sd  = torch.load(path, map_location="cpu")
        net.load_state_dict(sd)
        net.eval()
        return net.to(device)

    def m_naf_psf(path: str, cfg: dict):
        net = NAFNetSharpenPSF(**cfg, residual_out=True, clamp_out=False)
        sd  = torch.load(path, map_location="cpu")
        net.load_state_dict(sd)
        net.eval()
        return net.to(device)

    def m_naf_correct(path: str, cfg: dict):
        net = NAFNetCorrect(**cfg, residual_out=True)
        # deploy checkpoint wraps weights under "model_state_dict"
        raw = torch.load(path, map_location="cpu")
        sd  = raw.get("model_state_dict", raw)
        net.load_state_dict(sd)
        net.eval()
        return net.to(device)

    # Stellar sharpening model
    stellar_path = getattr(R, "CC_S_PTH", None)
    if not stellar_path or not os.path.exists(stellar_path):
        raise RuntimeError("Sharpen: CC_S_PTH (stellar NAFNet) not found.")
    stellar = m_naf_rgb(stellar_path, _SCOND_NAFNET)

    # Non-stellar PSF-aware model
    ns_path = getattr(R, "CC_NS_PTH", None)
    if not ns_path or not os.path.exists(ns_path):
        raise RuntimeError("Sharpen: CC_NS_PTH (nonstellar PSF NAFNet) not found.")
    ns_cond = m_naf_psf(ns_path, _NSCOND_NAFNET)

    # Aberration correction model (optional — graceful degradation if absent)
    correct = None
    correct_path = getattr(R, "CC_C_PTH", None)
    if correct_path and os.path.exists(correct_path):
        try:
            correct = m_naf_correct(correct_path, _CORRECT_NAFNET)
            _dbg(f"Loaded aberration correction model from {correct_path}", status_cb=print)
        except Exception as e:
            _dbg(f"Failed to load correction model ({correct_path}): {e}", status_cb=print)
            correct = None
    else:
        _dbg("CC_C_PTH not found — aberration correction unavailable.", status_cb=print)

    return SharpenModels(
        device=device,
        is_onnx=False,
        stellar=stellar,
        ns1=None, ns2=None, ns4=None, ns8=None,
        ns_cond=ns_cond,
        stellar_cond=False,
        correct=correct,
        torch=torch,
    )


# ---------------- Inference helpers ----------------

def _chunk_coords_with_overlap(H: int, W: int, chunk_size: int, overlap: int):
    step = chunk_size - overlap
    out = []
    for i in range(0, H, step):
        for j in range(0, W, step):
            ei = min(i + chunk_size, H)
            ej = min(j + chunk_size, W)
            if ei <= i or ej <= j:
                continue
            is_edge = (i == 0) or (j == 0) or (ei >= H) or (ej >= W)
            out.append((i, j, ei, ej, is_edge))
    return out


def _recommended_batch_size(
    models: SharpenModels,
    chunk_size: int,
    *,
    psf: bool = False,
    execution_mode: str = "auto",
    batch_size_override: int = 0,
) -> int:
    if batch_size_override and batch_size_override > 0:
        return int(batch_size_override)
    if execution_mode == "compatibility":
        return 1
    if models.is_onnx:
        return 1
    dev = getattr(models.device, "type", str(models.device)).lower()
    if dev == "cuda":
        if chunk_size <= 256:  return 16 if not psf else 12
        if chunk_size <= 384:  return 8  if not psf else 6
        if chunk_size <= 512:  return 4  if not psf else 3
        return 2
    if dev == "mps":
        if chunk_size <= 256:  return 4 if not psf else 3
        return 2
    return 1


def _encode_psf_log2_0_1(psf: float, psf_min: float = 1.0, psf_max: float = 8.0) -> float:
    t = (math.log2(psf) - math.log2(psf_min)) / (math.log2(psf_max) - math.log2(psf_min))
    return float(np.clip(t, 0.0, 1.0))


def _infer_chunk_psf(models: SharpenModels, model: Any, chunk2d: np.ndarray, psf01: float) -> np.ndarray:
    torch = models.torch
    dev   = models.device
    chunk2d = np.asarray(chunk2d, np.float32)
    t = torch.from_numpy(np.ascontiguousarray(chunk2d)).unsqueeze(0).unsqueeze(0)
    t = t.to(dev, dtype=torch.float32)
    t_rgb = t.expand(-1, 3, -1, -1)
    psf_t = torch.tensor([[float(psf01)]], dtype=torch.float32, device=dev)
    with torch.no_grad(), _autocast_context(torch, dev):
        y = model(t_rgb, psf_t)
        y = y[:, 0].detach().float().cpu().numpy()[0]
    return y.astype(np.float32, copy=False)


def _infer_chunk_psf_onnx(models: SharpenModels, session: Any, chunk2d: np.ndarray, psf01: float) -> np.ndarray:
    inp = np.asarray(chunk2d, np.float32)[np.newaxis, np.newaxis, :, :]
    inp = np.tile(inp, (1, 3, 1, 1))
    name_img, name_psf, name_out = _ort_pick_io_names(session)
    if name_psf is None:
        raise RuntimeError("PSF-conditional ONNX model expected 2 inputs, but only 1 found.")
    psf = np.array([[float(psf01)]], dtype=np.float32)
    out = session.run([name_out], {name_img: inp, name_psf: psf})[0]
    if out.ndim == 4:   y = out[0, 0]
    elif out.ndim == 3: y = out[0]; y = y[0] if y.shape[0] in (1, 3) else y
    else: raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
    return y.astype(np.float32, copy=False)


def _infer_chunks_batched(
    models: SharpenModels,
    model: Any,
    chunks2d: list[np.ndarray],
    batch_size: int,
    *,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
    progress_done_start: int = 0,
    progress_total: int = 0,
) -> list[np.ndarray]:
    if not chunks2d:
        return []

    if progress_total <= 0:
        progress_total = len(chunks2d)

    if models.is_onnx:
        outputs = []
        for idx, ch in enumerate(chunks2d, start=1):
            outputs.append(_infer_chunk(models, model, ch))
            if progress_cb is not None:
                if progress_cb(progress_done_start + idx, progress_total) is False:
                    raise RuntimeError("Cancelled.")
        return outputs

    torch = models.torch
    dev = models.device
    last_err = None

    for bs in _iter_batch_sizes(batch_size):
        try:
            outputs: list[np.ndarray] = [None] * len(chunks2d)  # type: ignore

            done = 0
            for s in range(0, len(chunks2d), bs):
                batch = chunks2d[s:s + bs]

                arr = np.stack(
                    [np.ascontiguousarray(np.asarray(ch, np.float32)) for ch in batch],
                    axis=0
                )[:, None, :, :]  # (B,1,H,W)
                t = torch.from_numpy(arr).to(dev, dtype=torch.float32)
                t_rgb = t.expand(-1, 3, -1, -1)

                with torch.no_grad(), _autocast_context(torch, dev):
                    y = model(t_rgb)
                    y = y[:, 0].detach().float().cpu().numpy()

                for bi, out_ch in enumerate(y):
                    outputs[s + bi] = out_ch.astype(np.float32, copy=False)

                done += len(batch)
                if progress_cb is not None:
                    if progress_cb(progress_done_start + done, progress_total) is False:
                        raise RuntimeError("Cancelled.")

            return outputs

        except Exception as e:
            last_err = e
            if bs == 1 or not _is_recoverable_batch_error(e):
                break

    raise last_err


def _infer_chunks_psf_batched(
    models: SharpenModels,
    model: Any,
    chunks2d: list[np.ndarray],
    psf01_list: list[float],
    batch_size: int,
    *,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
    progress_done_start: int = 0,
    progress_total: int = 0,
) -> list[np.ndarray]:
    if not chunks2d:
        return []

    if progress_total <= 0:
        progress_total = len(chunks2d)

    if models.is_onnx:
        outputs = []
        for idx, (ch, psf01) in enumerate(zip(chunks2d, psf01_list), start=1):
            outputs.append(_infer_chunk_psf_onnx(models, model, ch, psf01))
            if progress_cb is not None:
                if progress_cb(progress_done_start + idx, progress_total) is False:
                    raise RuntimeError("Cancelled.")
        return outputs

    torch = models.torch
    dev = models.device
    last_err = None

    for bs in _iter_batch_sizes(batch_size):
        try:
            outputs: list[np.ndarray] = [None] * len(chunks2d)  # type: ignore

            done = 0
            for s in range(0, len(chunks2d), bs):
                batch       = chunks2d[s:s + bs]
                psf_batch   = psf01_list[s:s + bs]

                arr     = np.stack(
                    [np.ascontiguousarray(np.asarray(ch, np.float32)) for ch in batch],
                    axis=0
                )[:, None, :, :]  # (B,1,H,W)
                psf_arr = np.array([[p] for p in psf_batch], dtype=np.float32)  # (B,1)

                t     = torch.from_numpy(arr).to(dev, dtype=torch.float32)
                t_rgb = t.expand(-1, 3, -1, -1)
                psf_t = torch.from_numpy(psf_arr).to(dev, dtype=torch.float32)

                with torch.no_grad(), _autocast_context(torch, dev):
                    y = model(t_rgb, psf_t)
                    y = y[:, 0].detach().float().cpu().numpy()

                for bi, out_ch in enumerate(y):
                    outputs[s + bi] = out_ch.astype(np.float32, copy=False)

                done += len(batch)
                if progress_cb is not None:
                    if progress_cb(progress_done_start + done, progress_total) is False:
                        raise RuntimeError("Cancelled.")

            return outputs

        except Exception as e:
            last_err = e
            if bs == 1 or not _is_recoverable_batch_error(e):
                break

    raise last_err


def _infer_chunk(models: SharpenModels, model: Any, chunk2d: np.ndarray) -> np.ndarray:
    """Returns 2D float32 (same shape as input chunk)."""
    chunk2d = np.asarray(chunk2d, np.float32)

    if models.is_onnx:
        inp = chunk2d[np.newaxis, np.newaxis, :, :].astype(np.float32)
        inp = np.repeat(inp, 3, axis=1)
        name_img, name_psf, name_out = _ort_pick_io_names(model)
        feeds = {name_img: inp}
        if name_psf is not None:
            feeds[name_psf] = np.array([[0.5]], dtype=np.float32)
        out = model.run([name_out], feeds)[0]
        if out.ndim == 4:   y = out[0, 0]
        elif out.ndim == 3: y = out[0]; y = y[0] if y.shape[0] in (1, 3) else y
        else: raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
        return y.astype(np.float32, copy=False)

    torch = models.torch
    dev   = models.device
    t     = torch.from_numpy(np.ascontiguousarray(chunk2d)).unsqueeze(0).unsqueeze(0)
    t     = t.to(dev, dtype=torch.float32)
    t_rgb = t.expand(-1, 3, -1, -1)

    with torch.no_grad(), _autocast_context(torch, dev):
        y = model(t_rgb)
        y = y[:, 0].detach().float().cpu().numpy()[0]

    return y.astype(np.float32, copy=False)


# ────────────────────────────────────────────────────────────────────────────
# Aberration correction — RGB tiled inference
#
# Operates on a full HxWx3 float32 image using Hanning-weighted blending.
# For mono images (triplicated channels) the correction is purely geometric
# so the output is meaningful and consistent.
# ────────────────────────────────────────────────────────────────────────────

def _correct_rgb_image(
    models: SharpenModels,
    image_rgb: np.ndarray,
    chunk_size: int = 256,
    overlap: int = 64,
    execution_mode: str = "auto",
    batch_size_override: int = 0,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
    progress_done_start: int = 0,
    progress_total: int = 0,
) -> np.ndarray:
    """
    Run the aberration correction model over a full HxWx3 image.
    Returns HxWx3 float32.  Gracefully returns the input unchanged if the
    correction model was not loaded (e.g. pth file missing).
    """
    if models.correct is None:
        _dbg("_correct_rgb_image: correction model not loaded, skipping.", status_cb=print)
        return image_rgb

    torch  = models.torch
    dev    = models.device
    H, W   = image_rgb.shape[:2]
    coords = _chunk_coords_with_overlap(H, W, chunk_size, overlap)
    total  = len(coords)

    if progress_total <= 0:
        progress_total = total

    batch_size = _recommended_batch_size(
        models, chunk_size,
        psf=False,
        execution_mode=execution_mode,
        batch_size_override=batch_size_override,
    )

    # Accumulator (float64 to reduce rounding) + weight map
    accum   = np.zeros((H, W, 3), dtype=np.float64)
    weights = np.zeros((H, W),    dtype=np.float64)

    # Build Hanning window once at the most common shape
    _hann_cache: dict[tuple[int,int], np.ndarray] = {}

    def _hann2d(h: int, w: int) -> np.ndarray:
        if (h, w) not in _hann_cache:
            wy = np.hanning(h).astype(np.float64)
            wx = np.hanning(w).astype(np.float64)
            _hann_cache[(h, w)] = np.outer(wy, wx)
        return _hann_cache[(h, w)]

    last_err = None
    for bs in _iter_batch_sizes(batch_size):
        try:
            accum[:]   = 0.0
            weights[:] = 0.0
            done = 0

            for s in range(0, total, bs):
                batch_coords = coords[s:s + bs]

                # Build batch tensor (B, 3, H_p, W_p) — pad each chunk to mult-of-16
                padded_list, orig_shapes = [], []
                for (i, j, ei, ej, _) in batch_coords:
                    patch = image_rgb[i:ei, j:ej, :3].astype(np.float32, copy=False)
                    ph, pw = patch.shape[:2]
                    # pad to multiple of 16
                    rh = (16 - ph % 16) % 16
                    rw = (16 - pw % 16) % 16
                    if rh or rw:
                        patch = np.pad(patch, ((0, rh), (0, rw), (0, 0)), mode="reflect")
                    padded_list.append(patch)
                    orig_shapes.append((ph, pw))

                # batch only if all same padded shape (group by shape like other infer fns)
                shape_groups: dict[tuple[int,int], list] = {}
                for bi, (patch, (ph, pw)) in enumerate(zip(padded_list, orig_shapes)):
                    shape_groups.setdefault(patch.shape[:2], []).append((bi, patch, ph, pw))

                results: dict[int, np.ndarray] = {}
                for _sh, items in shape_groups.items():
                    arr = np.stack(
                        [np.transpose(it[1], (2, 0, 1)) for it in items], axis=0
                    ).astype(np.float32)   # (B, 3, Hp, Wp)
                    t = torch.from_numpy(arr).to(dev, dtype=torch.float32)
                    with torch.no_grad(), _autocast_context(torch, dev):
                        out = models.correct(t)          # (B, 3, Hp, Wp)
                        out = out.detach().float().cpu().numpy()
                    for k, (bi, _patch, ph, pw) in enumerate(items):
                        results[bi] = np.transpose(out[k, :, :ph, :pw], (1, 2, 0))  # (ph, pw, 3)

                # Accumulate with Hanning weights
                for bi, (i, j, ei, ej, _) in enumerate(batch_coords):
                    ph, pw = orig_shapes[bi]
                    corrected = np.clip(results[bi], 0.0, 1.0).astype(np.float64)
                    w2d = _hann2d(ph, pw)
                    accum[i:i+ph, j:j+pw]   += corrected * w2d[:, :, None]
                    weights[i:i+ph, j:j+pw] += w2d

                done += len(batch_coords)
                if progress_cb is not None:
                    if progress_cb(progress_done_start + done, progress_total) is False:
                        raise RuntimeError("Cancelled.")

            # Normalise
            w_safe = np.maximum(weights, 1e-9)
            result  = (accum / w_safe[:, :, None]).astype(np.float32)
            return np.clip(result, 0.0, 1.0)

        except Exception as e:
            last_err = e
            if bs == 1 or not _is_recoverable_batch_error(e):
                break

    raise last_err


# ---------------- Main API ----------------

@dataclass
class SharpenParams:
    mode: str
    stellar_amount: float
    nonstellar_amount: float
    nonstellar_strength: float
    sharpen_channels_separately: bool
    auto_detect_psf: bool
    use_gpu: bool
    chunk_size: int = 256
    overlap: int = 64
    temp_stretch: bool = False
    target_median: float = 0.25
    execution_mode: str = "auto"
    batch_size_override: int = 0
    # "sharpen_only" | "correct_only" | "correct_sharpen"
    stellar_correct_mode: str = "sharpen_only"


def sharpen_image_array(image: np.ndarray,
                        params: SharpenParams,
                        progress_cb: Optional[ProgressCB] = None,
                        status_cb=print) -> tuple[np.ndarray, bool]:

    def _call_progress(cb, done: int, total: int, stage: str) -> bool:
        try:
            return bool(cb(int(done), int(total), str(stage)))
        except TypeError:
            try:
                return bool(cb(int(done), int(total)))
            except Exception:
                return True
        except Exception:
            return True

    if progress_cb is None:
        def progress_cb(done, total, stage):
            return True
    else:
        _user_cb = progress_cb
        def progress_cb(done, total, stage):
            return _call_progress(_user_cb, done, total, stage)

    img = np.asarray(image)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)

    img3, was_mono = _to_3ch(img)
    img3 = np.clip(img3, 0.0, 1.0)

    stellar_correct_mode = str(getattr(params, "stellar_correct_mode", "sharpen_only")).lower()

    if getattr(params, "execution_mode", "auto") == "compatibility":
        params.batch_size_override = 1
        params.chunk_size = min(int(params.chunk_size), 256)

    def _run_with(models: SharpenModels) -> tuple[np.ndarray, bool]:
        bordered = add_border(img3, border_size=16)

        # ── Aberration correction pre-pass ──────────────────────────────────
        if stellar_correct_mode in ("correct_only", "correct_sharpen"):
            if models.correct is None:
                try:
                    status_cb("Aberration correction model not loaded — skipping correction pass.")
                except Exception:
                    pass
            else:
                try:
                    progress_cb(0, 1, "Aberration correction")
                    bordered = _correct_rgb_image(
                        models,
                        bordered,
                        chunk_size=int(params.chunk_size),
                        overlap=int(params.overlap),
                        execution_mode=getattr(params, "execution_mode", "auto"),
                        batch_size_override=getattr(params, "batch_size_override", 0),
                        progress_cb=lambda d, t: progress_cb(d, t, "Aberration correction"),
                    )
                except Exception as e:
                    try:
                        status_cb(f"Aberration correction failed ({e}); continuing without correction.")
                    except Exception:
                        pass

        # ── Early exit for correct-only mode ────────────────────────────────
        if stellar_correct_mode == "correct_only":
            out = remove_border(bordered, border_size=16)
            if was_mono and out.ndim == 3 and out.shape[2] == 3:
                out = np.mean(out, axis=2, keepdims=True).astype(np.float32, copy=False)
            return np.clip(out, 0.0, 1.0), was_mono

        # ── Standard sharpening (sharpen_only or correct_sharpen) ───────────
        if bool(getattr(params, "temp_stretch", False)):
            stretch_needed = True
        else:
            med_metric = float(np.median(bordered - np.min(bordered)))
            stretch_needed = (med_metric < 0.08)

        if stretch_needed:
            tm = float(np.clip(getattr(params, "target_median", 0.25), 0.01, 0.50))
            stretched, orig_min, orig_meds = stretch_image_unlinked_rgb(bordered, target_median=tm)
        else:
            stretched, orig_min, orig_meds = bordered, None, None

        if params.sharpen_channels_separately and (not was_mono):
            out = np.empty_like(stretched)
            for c, label in enumerate(("R", "G", "B")):
                progress_cb(0, 1, f"Sharpening {label} channel")
                plane_in  = np.clip(stretched[..., c], 0.0, 1.0)
                plane_out = _sharpen_plane(models, plane_in, params, progress_cb)
                out[..., c] = np.clip(plane_out, 0.0, 1.0)
            sharpened = out
        else:
            y, cb_ch, cr = extract_luminance_rgb(stretched)
            y2 = _sharpen_plane(models, y, params, progress_cb)
            sharpened = merge_luminance(y2, cb_ch, cr)

        if stretch_needed:
            sharpened = unstretch_image_unlinked_rgb(sharpened, orig_meds, orig_min, was_mono)

        sharpened = remove_border(sharpened, border_size=16)

        if was_mono:
            if sharpened.ndim == 3 and sharpened.shape[2] == 3:
                sharpened = np.mean(sharpened, axis=2, keepdims=True).astype(np.float32, copy=False)

        return np.clip(sharpened, 0.0, 1.0), was_mono

    try:
        progress_cb(0, 1, "Loading models")
        models = load_sharpen_models(use_gpu=params.use_gpu, status_cb=status_cb)
        out, was_mono = _run_with(models)
        return out, was_mono

    except Exception as e:
        try:
            dev = getattr(models.device, "type", None) or str(models.device)
        except Exception:
            dev = "unknown"

        if params.use_gpu and str(dev).lower() != "cpu" and _is_device_lost_error(e):
            try:
                status_cb(f"CosmicClarity Sharpen: GPU device lost on backend={dev}; retrying on CPU.")
            except Exception:
                pass
            cpu_models = load_sharpen_models(use_gpu=False, status_cb=status_cb)
            out, was_mono = _run_with(cpu_models)
            return out, was_mono

        _dbg("sharpen_image_array ERROR:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
             status_cb=status_cb)
        raise


def _sharpen_plane(models: SharpenModels,
                   plane: np.ndarray,
                   params: SharpenParams,
                   progress_cb: ProgressCB) -> np.ndarray:
    plane = np.asarray(plane, np.float32)

    chunk_size = int(getattr(params, "chunk_size", 256))
    overlap    = int(getattr(params, "overlap", 64))
    chunk_size = max(64, min(1024, chunk_size))
    overlap    = max(0,  min(chunk_size - 1, overlap))

    H, W   = plane.shape
    coords = _chunk_coords_with_overlap(H, W, chunk_size, overlap)
    total  = len(coords)

    if params.mode == "Both":
        total_progress_units = max(1, total * 2)
    else:
        total_progress_units = max(1, total)

    try:
        progress_cb(0, total_progress_units, f"Sharpen start ({total} chunks)")
    except Exception:
        pass

    psf_ref_plane = plane.copy() if params.mode == "Both" else plane

    # ── Stage 1: stellar ────────────────────────────────────────────────────
    if params.mode in ("Stellar Only", "Both"):
        stellar_chunks = [plane[i:ei, j:ej] for (i, j, ei, ej, _) in coords]
        batch_size = _recommended_batch_size(
            models, chunk_size, psf=False,
            execution_mode=getattr(params, "execution_mode", "auto"),
            batch_size_override=getattr(params, "batch_size_override", 0),
        )
        stellar_out = _infer_chunks_batched(
            models, models.stellar, stellar_chunks, batch_size=batch_size,
            progress_cb=lambda done, tot: progress_cb(done, total_progress_units, "Stellar sharpening"),
            progress_done_start=0, progress_total=total,
        )
        out_chunks = []
        for (i, j, ei, ej, is_edge), y in zip(coords, stellar_out):
            chunk   = plane[i:ei, j:ej]
            blended = blend_images(chunk, y, params.stellar_amount)
            out_chunks.append((blended, i, j, is_edge))
        plane = stitch_chunks_ignore_border(out_chunks, plane.shape, border_size=16)

        if params.mode == "Stellar Only":
            try:
                progress_cb(total_progress_units, total_progress_units, "Stellar sharpening")
            except Exception:
                pass
            return plane

    # ── Stage 2: non-stellar ─────────────────────────────────────────────────
    if params.mode in ("Non-Stellar Only", "Both"):
        if models.ns_cond is None:
            raise RuntimeError("Non-stellar sharpen: ns_cond model is required but not loaded.")

        ns_chunks   = [plane[i:ei, j:ej] for (i, j, ei, ej, _) in coords]
        psf01_list  = []
        for (i, j, ei, ej, _), chunk in zip(coords, ns_chunks):
            if params.auto_detect_psf:
                ref = psf_ref_plane[i:ei, j:ej]
                h, w = chunk.shape
                if ref.shape != chunk.shape:
                    ref = ref[:h, :w]
                r = float(np.clip(measure_psf_radius(ref, default_radius=3.0), 1.0, 8.0))
            else:
                r = float(np.clip(params.nonstellar_strength, 1.0, 8.0))
            psf01_list.append(_encode_psf_log2_0_1(r, 1.0, 8.0))

        batch_size   = _recommended_batch_size(
            models, chunk_size, psf=True,
            execution_mode=getattr(params, "execution_mode", "auto"),
            batch_size_override=getattr(params, "batch_size_override", 0),
        )
        start_offset = total if params.mode == "Both" else 0

        ns_out = _infer_chunks_psf_batched(
            models, models.ns_cond, ns_chunks, psf01_list, batch_size=batch_size,
            progress_cb=lambda done, tot: progress_cb(done, total_progress_units, "Non-stellar sharpening"),
            progress_done_start=start_offset, progress_total=total,
        )
        out_chunks = []
        for (i, j, ei, ej, is_edge), y in zip(coords, ns_out):
            chunk   = plane[i:ei, j:ej]
            blended = blend_images(chunk, y, params.nonstellar_amount)
            out_chunks.append((blended, i, j, is_edge))
        plane = stitch_chunks_ignore_border(out_chunks, plane.shape, border_size=16)

    try:
        progress_cb(total_progress_units, total_progress_units, "Sharpen complete")
    except Exception:
        pass

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
    chunk_size: int = 256,
    overlap: int = 64,
    temp_stretch: bool = False,
    target_median: float = 0.25,
    execution_mode: str = "auto",
    batch_size_override: int = 0,
    stellar_correct_mode: str = "sharpen_only",
    progress_cb: Optional[Callable[[int, int], bool]] = None,
    status_cb=print,
) -> np.ndarray:
    if progress_cb is None:
        def _prog(done, total, stage):
            return True
    else:
        def _prog(done, total, stage):
            try:
                return bool(progress_cb(int(done), int(total)))
            except Exception:
                return True

    params = SharpenParams(
        mode=str(sharpening_mode),
        stellar_amount=float(stellar_amount),
        nonstellar_amount=float(nonstellar_amount),
        nonstellar_strength=float(nonstellar_strength),
        sharpen_channels_separately=bool(separate_channels),
        auto_detect_psf=bool(auto_detect_psf),
        use_gpu=bool(use_gpu),
        chunk_size=int(chunk_size),
        overlap=int(overlap),
        temp_stretch=bool(temp_stretch),
        target_median=float(target_median),
        execution_mode=str(execution_mode),
        batch_size_override=int(batch_size_override),
        stellar_correct_mode=str(stellar_correct_mode),
    )

    out, _was_mono = sharpen_image_array(
        image_rgb01,
        params=params,
        progress_cb=_prog,
        status_cb=status_cb,
    )
    return np.asarray(out, dtype=np.float32)


def clear_sharpen_models_cache(*, aggressive: bool = False, status_cb=print) -> None:
    global _MODELS_CACHE
    try:
        n = len(_MODELS_CACHE)
        _MODELS_CACHE.clear()
        status_cb(f"[CC Sharpen] Cleared model cache entries: {n}")
    except Exception as e:
        try:
            status_cb(f"[CC Sharpen] Cache clear failed: {type(e).__name__}: {e}")
        except Exception:
            pass

    if not aggressive:
        return

    try:
        import gc
        gc.collect()
    except Exception:
        pass

    try:
        from setiastro.saspro.runtime_torch import import_torch
        torch = import_torch(prefer_cuda=True, prefer_xpu=False, prefer_dml=True, status_cb=lambda s: None)
        try:
            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                status_cb("[CC Sharpen] torch.cuda.empty_cache() called")
        except Exception:
            pass
        try:
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                    status_cb("[CC Sharpen] torch.mps.empty_cache() called")
        except Exception:
            pass
    except Exception:
        pass