# src/setiastro/saspro/cosmicclarity_engines/sharpen_engine.py
from __future__ import annotations

from dataclasses import dataclass
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

def _dbg(msg: str, status_cb=print):
    pass
    #"""Debug print to terminal; also tries status_cb."""
    #if not _DEBUG_SHARPEN:
    #    return

    #ts = time.strftime("%H:%M:%S")
    #line = f"[SharpenDBG {ts}] {msg}"
    #print(line, flush=True)
    #try:
    #    status_cb(line)
    #except Exception:
    #    pass


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
    """
    Import onnxruntime AFTER runtime_torch has added runtime site-packages to sys.path.
    """
    try:
        import onnxruntime as ort  # type: ignore
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

        y0 = i + bh
        y1 = i + h - bh
        x0 = j + bw
        x1 = j + w - bw

        # empty inner region? skip
        if y1 <= y0 or x1 <= x0:
            continue

        inner = chunk[bh:h-bh, bw:w-bw]

        # clip destination to image bounds
        yy0 = max(0, y0)
        yy1 = min(H, y1)
        xx0 = max(0, x0)
        xx1 = min(W, x1)

        # did clipping erase it?
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
            radius = fwhm * 0.5
            radii.append(radius)
        return float(np.median(radii)) if radii else default_radius
    except Exception:
        return default_radius


# ---------------- Model bundle + loading ----------------

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
    stellar_cond: bool = False          # <-- NEW: stellar uses PSF input?
    torch: Any | None = None


def _pad2d_to_multiple(x: np.ndarray, mult: int = 16, mode: str = "reflect") -> tuple[np.ndarray, int, int]:
    """Pad 2D array on bottom/right so H,W are multiples of `mult`."""
    h, w = x.shape
    ph = (mult - (h % mult)) % mult
    pw = (mult - (w % mult)) % mult
    if ph == 0 and pw == 0:
        return x, h, w
    # reflect is usually safe; edge/constant also OK
    xp = np.pad(x, ((0, ph), (0, pw)), mode=mode)
    return xp, h, w



# Cache by (backend_tag, resolved_backend)
_MODELS_CACHE: dict[tuple[str, str], SharpenModels] = {}  # (backend_tag, resolved)

def load_sharpen_models(use_gpu: bool, status_cb=print) -> SharpenModels:
    backend_tag = "cc_sharpen_ai4"
    is_windows = (os.name == "nt")
    _dbg(f"ENTER load_sharpen_models(use_gpu={use_gpu})", status_cb=status_cb)

    # ---- torch runtime import ----
    t0 = time.time()
    try:
        _dbg("Calling _get_torch(...)", status_cb=status_cb)
        torch = _get_torch(
            prefer_cuda=bool(use_gpu),
            prefer_dml=bool(use_gpu) and (os.name == "nt"),
            status_cb=status_cb,
        )
        _dbg(f"_get_torch OK in {time.time()-t0:.3f}s; torch={getattr(torch,'__version__',None)} file={getattr(torch,'__file__',None)}",
             status_cb=status_cb)
    except Exception as e:
        _dbg("ERROR in _get_torch:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
             status_cb=status_cb)
        raise
    ort = _get_ort(status_cb=status_cb)
    # ---- runtime venv CUDA probe (subprocess) ----
    try:
        rt = _user_runtime_dir()
        vpy = _venv_paths(rt)["python"]
        _dbg(f"Runtime dir={rt} venv_python={vpy}", status_cb=status_cb)
        t1 = time.time()
        ok, cuda_tag, err = _check_cuda_in_venv(vpy, status_cb=status_cb)
        _dbg(f"CUDA probe finished in {time.time()-t1:.3f}s: ok={ok}, torch.version.cuda={cuda_tag}, err={err!r}",
             status_cb=status_cb)
    except Exception as e:
        _dbg("Runtime CUDA probe FAILED:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
             status_cb=status_cb)

    # ---- CUDA branch ----
    if use_gpu:
        _dbg("Checking torch.cuda.is_available() ...", status_cb=status_cb)
        try:
            t2 = time.time()
            cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
            _dbg(f"torch.cuda.is_available()={cuda_ok} (took {time.time()-t2:.3f}s)", status_cb=status_cb)
        except Exception as e:
            _dbg("torch.cuda.is_available() raised:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                 status_cb=status_cb)
            cuda_ok = False

        if cuda_ok:
            cache_key = (backend_tag, "cuda")
            if cache_key in _MODELS_CACHE:
                _dbg("Returning cached CUDA models.", status_cb=status_cb)
                return _MODELS_CACHE[cache_key]

            try:
                _dbg("Creating torch.device('cuda') ...", status_cb=status_cb)
                device = torch.device("cuda")
                _dbg("Querying torch.cuda.get_device_name(0) ...", status_cb=status_cb)
                name = torch.cuda.get_device_name(0)
                _dbg(f"Using CUDA device: {name}", status_cb=status_cb)

                _dbg("Loading torch .pth models (CUDA) ...", status_cb=status_cb)
                t3 = time.time()
                models = _load_torch_models(torch, device)
                _dbg(f"Loaded CUDA models in {time.time()-t3:.3f}s", status_cb=status_cb)

                _MODELS_CACHE[cache_key] = models
                return models
            except Exception as e:
                _dbg("CUDA path failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                     status_cb=status_cb)
                # fall through to DML/ORT/CPU
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ADD THE MPS BLOCK RIGHT HERE (after CUDA, before DirectML)
    # ---- MPS branch (macOS Apple Silicon) ----
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
                _dbg("Returning cached MPS models.", status_cb=status_cb)
                return _MODELS_CACHE[cache_key]

            try:
                device = torch.device("mps")
                _dbg("CosmicClarity Sharpen: using MPS", status_cb=status_cb)

                t_m = time.time()
                models = _load_torch_models(torch, device)
                _dbg(f"Loaded MPS models in {time.time()-t_m:.3f}s", status_cb=status_cb)

                _MODELS_CACHE[cache_key] = models
                return models
            except Exception as e:
                _dbg("MPS path failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                     status_cb=status_cb)
                # fall through
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ---- Torch DirectML branch ----
    if use_gpu and is_windows:
        _dbg("Trying torch-directml path ...", status_cb=status_cb)
        try:
            import torch_directml
            cache_key = (backend_tag, "dml_torch")
            if cache_key in _MODELS_CACHE:
                _dbg("Returning cached DML-torch models.", status_cb=status_cb)
                return _MODELS_CACHE[cache_key]

            t4 = time.time()
            dml = torch_directml.device()            
            _ = (torch.ones(1, device=dml) + 1).to("cpu").item()  # verify DML actually works
            _dbg(f"torch_directml.device() OK in {time.time()-t4:.3f}s", status_cb=status_cb)

            _dbg("Loading torch .pth models (DirectML) ...", status_cb=status_cb)
            t5 = time.time()
            models = _load_torch_models(torch, dml)
            _dbg(f"Loaded DML-torch models in {time.time()-t5:.3f}s", status_cb=status_cb)

            _MODELS_CACHE[cache_key] = models
            return models
        except Exception as e:
            _dbg("DirectML (torch-directml) failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                 status_cb=status_cb)

    # ---- ONNX Runtime DirectML fallback ----
    # ---- ONNX Runtime DirectML fallback ----
    if use_gpu and ort is not None:
        _dbg("Checking ORT providers ...", status_cb=status_cb)
        try:
            prov = ort.get_available_providers()
            _dbg(f"ORT providers: {prov}", status_cb=status_cb)
            if "DmlExecutionProvider" in prov:
                cache_key = (backend_tag, "dml_ort")
                if cache_key in _MODELS_CACHE:
                    _dbg("Returning cached DML-ORT models.", status_cb=status_cb)
                    return _MODELS_CACHE[cache_key]

                _dbg("Loading ONNX models (DML EP) ...", status_cb=status_cb)
                t6 = time.time()
                models = _load_onnx_models(ort)   # <<<< CHANGED
                _dbg(f"Loaded ONNX models in {time.time()-t6:.3f}s", status_cb=status_cb)

                _MODELS_CACHE[cache_key] = models
                return models
        except Exception as e:
            _dbg("ORT provider check/load failed:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                 status_cb=status_cb)


    # ---- CPU fallback ----
    cache_key = (backend_tag, "cpu")
    if cache_key in _MODELS_CACHE:
        _dbg("Returning cached CPU models.", status_cb=status_cb)
        return _MODELS_CACHE[cache_key]

    try:
        _dbg("Falling back to CPU torch models ...", status_cb=status_cb)
        device = torch.device("cpu")
        t7 = time.time()
        models = _load_torch_models(torch, device)
        _dbg(f"Loaded CPU models in {time.time()-t7:.3f}s", status_cb=status_cb)
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

    # NEW AI4 names
    stellar = s(R.CC_STELLAR_NAF_ONNX)        # deep_stellar_sharp_conditional_psf_AI4.onnx (but NOT PSF-aware)
    img_name, psf_name, out_name = _ort_pick_io_names(stellar)
    if psf_name is not None:
        raise RuntimeError("Stellar ONNX unexpectedly has a PSF input; exporter should produce 1-input model.")
    ns_cond = s(R.CC_NS_COND_NAF_ONNX)        # deep_nonstellar_sharp_conditional_psf_AI4.onnx (PSF-aware)

    return SharpenModels(
        device="DirectML",
        is_onnx=True,
        stellar=stellar,
        ns1=None, ns2=None, ns4=None, ns8=None,
        ns_cond=ns_cond,
        stellar_cond=False,
        torch=None,
    )



def _ort_pick_io_names(session) -> tuple[str, Optional[str], str]:
    """
    Returns: (img_name, psf_name_or_None, out_name)

    Robustly identifies which input is image vs PSF based on input shapes/ranks.
    Works even if exporter changes ordering.
    """
    ins = session.get_inputs()
    out = session.get_outputs()[0].name

    img_name = None
    psf_name = None

    for i in ins:
        shp = i.shape  # may contain None/dynamic
        # Heuristic:
        # - Image is NCHW => rank 4
        # - PSF is scalar/batch scalar => rank 1 or 2 (e.g. [1] or [1,1])
        rank = len(shp) if shp is not None else 0
        if rank == 4:
            img_name = i.name
        elif rank in (1, 2):
            psf_name = i.name

    # Fallbacks if shapes are weird/dynamic:
    if img_name is None:
        img_name = ins[0].name
    if len(ins) == 1:
        psf_name = None
    elif psf_name is None:
        # assume 2-input model where second is PSF
        psf_name = ins[1].name

    return img_name, psf_name, out


def _load_torch_models(torch, device) -> SharpenModels:
    import torch.nn as nn  # comes from runtime torch env

    # ---- NEW: NAFNet PSF model defs (must match training) ----
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
            self.dwconv = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=True)
            self.sg = SimpleGate()
            self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1, bias=True))
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

    class NAFNetSharpen(nn.Module):
        def __init__(self, in_ch=3, out_ch=3, width=32,
                    enc_blk_nums=(2,4,6,8), dec_blk_nums=(2,2,2,2),
                    middle_blk_num=4,
                    residual_out=True, clamp_out=False):
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
            # IMPORTANT: stellar model expects RGB->RGB sharpened output
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
            psf_map = psf_t.view(b,1,1,1).expand(b,1,h,w)
            x4 = torch.cat([x_rgb, psf_map], dim=1)
            delta = self.forward_delta(x4)
            y = x_rgb + delta if self.residual_out else delta
            if self.clamp_out:
                y = y.clamp(0.0, 1.0)
            return y


    R = get_resources()

    def m_naf_rgb(path: str, cfg: dict):
        net = NAFNetSharpen(**cfg, residual_out=True, clamp_out=False)
        sd = torch.load(path, map_location="cpu")
        net.load_state_dict(sd)
        net.eval()
        return net.to(device)

    def m_naf_psf(path: str, cfg: dict):
        net = NAFNetSharpenPSF(**cfg, residual_out=True, clamp_out=False)
        sd = torch.load(path, map_location="cpu")
        net.load_state_dict(sd)
        net.eval()
        return net.to(device)

    # --- STELLAR (NOT PSF-aware; new NAFNet) ---
    stellar_path = getattr(R, "CC_S_PTH", None)
    if not stellar_path or not os.path.exists(stellar_path):
        raise RuntimeError("Sharpen: CC_S_PTH (stellar NAFNet) not found.")

    stellar = m_naf_rgb(stellar_path, _SCOND_NAFNET)

    # --- NONSTELLAR (PSF-aware) ---
    ns_cond = None
    ns_path = getattr(R, "CC_NS_PTH", None)
    if ns_path and os.path.exists(ns_path):
        ns_cond = m_naf_psf(ns_path, _NSCOND_NAFNET)
    else:
        raise RuntimeError("Sharpen: CC_NS_PTH (nonstellar PSF NAFNet) not found.")


    return SharpenModels(
        device=device,
        is_onnx=False,
        stellar=stellar,
        ns1=None, ns2=None, ns4=None, ns8=None,
        ns_cond=ns_cond,
        stellar_cond=False,
        torch=torch,
    )

# ---------------- Inference helpers ----------------


def _encode_psf_log2_0_1(psf: float, psf_min: float = 1.0, psf_max: float = 8.0) -> float:
    t = (math.log2(psf) - math.log2(psf_min)) / (math.log2(psf_max) - math.log2(psf_min))
    return float(np.clip(t, 0.0, 1.0))

def _infer_chunk_psf(models: SharpenModels, model: Any, chunk2d: np.ndarray, psf01: float) -> np.ndarray:
    torch = models.torch
    dev = models.device

    chunk_p, h0, w0 = _pad2d_to_multiple(np.asarray(chunk2d, np.float32), mult=16, mode="reflect")
    hp, wp = chunk_p.shape

    t_cpu = torch.tensor(chunk_p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,Hp,Wp)
    t_rgb = t_cpu.repeat(1, 3, 1, 1).to(dev)                                      # (1,3,Hp,Wp)
    psf_t = torch.tensor([[float(psf01)]], dtype=torch.float32, device=dev)        # (1,1)

    with torch.no_grad(), _autocast_context(torch, dev):
        y = model(t_rgb, psf_t)                 # (1,3,Hp,Wp)
        y = y[0, 0].detach().cpu().numpy()      # (Hp,Wp)

    return y[:h0, :w0].astype(np.float32, copy=False)

def _infer_chunk_psf_onnx(models: SharpenModels, session: Any, chunk2d: np.ndarray, psf01: float) -> np.ndarray:
    """ONNX version of PSF-conditional infer. Returns 2D float32."""
    h0, w0 = chunk2d.shape

    chunk_p, h0, w0 = _pad2d_to_multiple(np.asarray(chunk2d, np.float32), mult=16, mode="reflect")
    inp = chunk_p[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,Hp,Wp)
    inp = np.tile(inp, (1, 3, 1, 1))                                # (1,3,Hp,Wp)

    name_img, name_psf, name_out = _ort_pick_io_names(session)
    if name_psf is None:
        raise RuntimeError("PSF-conditional ONNX model expected 2 inputs, but only 1 input was found.")

    psf = np.array([[float(psf01)]], dtype=np.float32)  # (1,1)

    out = session.run([name_out], {name_img: inp, name_psf: psf})[0]

    if out.ndim == 4:
        y = out[0, 0]
    elif out.ndim == 3:
        y = out[0]
        if y.shape[0] in (1, 3):
            y = y[0]
    else:
        raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")

    return y[:h0, :w0].astype(np.float32, copy=False)


def _infer_chunk(models: SharpenModels, model: Any, chunk2d: np.ndarray) -> np.ndarray:
    """Returns 2D float32 (cropped to original chunk shape)."""
    h0, w0 = chunk2d.shape

    if models.is_onnx:
        t0 = time.time()

        # ✅ match torch path: pad to mult-of-16 (NAFNet friendly)
        chunk_p, h0, w0 = _pad2d_to_multiple(np.asarray(chunk2d, np.float32), mult=16, mode="reflect")
        hp, wp = chunk_p.shape

        inp = chunk_p[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1,1,Hp,Wp)
        inp = np.tile(inp, (1, 3, 1, 1))                                # (1,3,Hp,Wp)

        name_img, name_psf, name_out = _ort_pick_io_names(model)
        feeds = {name_img: inp}

        # if someone accidentally passes a 2-input model here, feed a default psf
        if name_psf is not None:
            feeds[name_psf] = np.array([[0.5]], dtype=np.float32)  # neutral-ish

        out = model.run([name_out], feeds)[0]  # expect (1,3,Hp,Wp) or (1,1,Hp,Wp) depending exporter

        # Normalize output handling:
        if out.ndim == 4:
            y = out[0, 0]  # channel 0
        elif out.ndim == 3:
            y = out[0]     # (C,H,W)?? rare
            if y.shape[0] in (1, 3):
                y = y[0]
        else:
            raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")

        y = y[:h0, :w0].astype(np.float32, copy=False)

        if _DEBUG_SHARPEN:
            _dbg(f"ORT infer OK {h0}x{w0} in {time.time()-t0:.3f}s", status_cb=lambda *_: None)
        return y

    # torch path (unchanged)
    torch = models.torch
    dev = models.device

    chunk_p, h0, w0 = _pad2d_to_multiple(chunk2d, mult=16, mode="reflect")

    t0 = time.time()
    if _DEBUG_SHARPEN:
        _dbg(f"Torch infer start chunk={h0}x{w0} padded={chunk_p.shape[0]}x{chunk_p.shape[1]} dev={getattr(dev,'type',dev)}",
             status_cb=lambda *_: None)

    try:
        t_cpu = torch.tensor(chunk_p, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,Hp,Wp)
        if _DEBUG_SHARPEN:
            _dbg("  moving tensor to device ...", status_cb=lambda *_: None)
        t = t_cpu.to(dev)

        if _DEBUG_SHARPEN and hasattr(dev, "type") and dev.type == "cuda":
            _dbg("  cuda.synchronize after .to(dev) ...", status_cb=lambda *_: None)
            torch.cuda.synchronize()

        with torch.no_grad(), _autocast_context(torch, dev):
            if _DEBUG_SHARPEN:
                _dbg("  running model forward ...", status_cb=lambda *_: None)

            y = model(t.repeat(1, 3, 1, 1))        # (1,3,Hp,Wp)
            if _DEBUG_SHARPEN and hasattr(dev, "type") and dev.type == "cuda":
                _dbg("  cuda.synchronize after forward ...", status_cb=lambda *_: None)
                torch.cuda.synchronize()

            y = y[0, 0].detach().cpu().numpy()     # (Hp,Wp)

        out = y[:h0, :w0].astype(np.float32, copy=False)

        if _DEBUG_SHARPEN:
            _dbg(f"Torch infer OK in {time.time()-t0:.3f}s", status_cb=lambda *_: None)
        return out

    except Exception as e:
        if _DEBUG_SHARPEN:
            _dbg("Torch infer ERROR:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                 status_cb=lambda *_: None)
        raise


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
    chunk_size: int = 256
    overlap: int = 64
    temp_stretch: bool = False
    target_median: float = 0.25

def sharpen_image_array(image: np.ndarray,
                        params: SharpenParams,
                        progress_cb: Optional[ProgressCB] = None,
                        status_cb=print) -> tuple[np.ndarray, bool]:
    if progress_cb is None:
        progress_cb = lambda done, total, stage: True

    _dbg("ENTER sharpen_image_array()", status_cb=status_cb)
    t_all = time.time()

    try:
        img = np.asarray(image)
        _dbg(f"Input shape={img.shape} dtype={img.dtype} min={float(np.nanmin(img)):.6f} max={float(np.nanmax(img)):.6f}",
             status_cb=status_cb)

        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
            _dbg("Converted input to float32", status_cb=status_cb)

        img3, was_mono = _to_3ch(img)
        img3 = np.clip(img3, 0.0, 1.0)
        _dbg(f"After _to_3ch: shape={img3.shape} was_mono={was_mono}", status_cb=status_cb)

        # prove progress_cb works
        try:
            progress_cb(0, 1, "Loading models")
        except Exception:
            pass

        models = load_sharpen_models(use_gpu=params.use_gpu, status_cb=status_cb)
        _dbg(f"Models loaded: is_onnx={models.is_onnx} device={models.device!r}", status_cb=status_cb)

        # border & stretch
        bordered = add_border(img3, border_size=16)

        # ✅ NEW: temp stretch override
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
        # per-channel sharpening option (color only)
        if params.sharpen_channels_separately and (not was_mono):
            _dbg("Sharpen per-channel path", status_cb=status_cb)
            out = np.empty_like(stretched)
            for c, label in enumerate(("R", "G", "B")):
                progress_cb(0, 1, f"Sharpening {label} channel")
                out[..., c] = _sharpen_plane(models, stretched[..., c], params, progress_cb)
            sharpened = out
        else:
            _dbg("Sharpen luminance path", status_cb=status_cb)
            y, cb, cr = extract_luminance_rgb(stretched)
            y2 = _sharpen_plane(models, y, params, progress_cb)
            sharpened = merge_luminance(y2, cb, cr)

        # unstretch / deborder
        if stretch_needed:
            _dbg("Unstretching ...", status_cb=status_cb)
            sharpened = unstretch_image_unlinked_rgb(sharpened, orig_meds, orig_min, was_mono)

        sharpened = remove_border(sharpened, border_size=16)

        if was_mono:
            if sharpened.ndim == 3 and sharpened.shape[2] == 3:
                sharpened = np.mean(sharpened, axis=2, keepdims=True).astype(np.float32, copy=False)

        out = np.clip(sharpened, 0.0, 1.0)
        _dbg(f"EXIT sharpen_image_array total_time={time.time()-t_all:.2f}s out_shape={out.shape}", status_cb=status_cb)
        return out, was_mono

    except Exception as e:
        _dbg("sharpen_image_array ERROR:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
             status_cb=status_cb)
        raise


def _sharpen_plane(models: SharpenModels,
                   plane: np.ndarray,
                   params: SharpenParams,
                   progress_cb: ProgressCB) -> np.ndarray:
    plane = np.asarray(plane, np.float32)
    chunk_size = int(getattr(params, "chunk_size", 256))
    overlap = int(getattr(params, "overlap", 64))
    psf_ref_plane = plane  # default: measure PSF from current plane

    # safety
    chunk_size = max(64, min(1024, chunk_size))
    overlap = max(0, min(chunk_size - 1, overlap))

    chunks = split_image_into_chunks_with_overlap(plane, chunk_size=chunk_size, overlap=overlap)

    total = len(chunks)

    # prove we got here
    try:
        progress_cb(0, max(total, 1), f"Sharpen start ({total} chunks)")
    except Exception:
        pass

    _dbg(f"_sharpen_plane: mode={params.mode} total_chunks={total} auto_psf={params.auto_detect_psf} dev={models.device!r}",
         status_cb=lambda *_: None)

    def _every(n: int) -> bool:
        return _DEBUG_SHARPEN and (n == 1 or n % 10 == 0 or n == total)

    # If we're doing Both, lock PSF measurement to the pre-stellar plane
    if params.mode == "Both":
        psf_ref_plane = plane.copy()

    # Stage 1: stellar
    if params.mode in ("Stellar Only", "Both"):
        _dbg("Stage 1: stellar BEGIN", status_cb=lambda *_: None)
        out_chunks = []
        t_stage = time.time()

        for k, (chunk, i, j, is_edge) in enumerate(chunks, start=1):
            t0 = time.time()
            y = _infer_chunk(models, models.stellar, chunk)
            blended = blend_images(chunk, y, params.stellar_amount)
            out_chunks.append((blended, i, j, is_edge))

            if _every(k):
                _dbg(f"  stellar chunk {k}/{total} ({time.time()-t0:.3f}s)", status_cb=lambda *_: None)

            if progress_cb(k, total, "Stellar sharpening") is False:
                _dbg("Stage 1: stellar CANCELLED", status_cb=lambda *_: None)
                return plane

        plane = stitch_chunks_ignore_border(out_chunks, plane.shape, border_size=16)
        _dbg(f"Stage 1: stellar END ({time.time()-t_stage:.2f}s)", status_cb=lambda *_: None)

        if params.mode == "Stellar Only":
            return plane

        # update chunks for stage 2
        chunk_size = int(getattr(params, "chunk_size", 256))
        overlap = int(getattr(params, "overlap", 64))

        # safety
        chunk_size = max(64, min(1024, chunk_size))
        overlap = max(0, min(chunk_size - 1, overlap))

        chunks = split_image_into_chunks_with_overlap(plane, chunk_size=chunk_size, overlap=overlap)

        total = len(chunks)

    # Stage 2: non-stellar (PSF-aware single model ONLY)
    if params.mode in ("Non-Stellar Only", "Both"):
        _dbg("Stage 2: non-stellar BEGIN", status_cb=lambda *_: None)
        if models.ns_cond is None:
            raise RuntimeError("Non-stellar sharpen: ns_cond model is required but not loaded.")

        out_chunks = []
        t_stage = time.time()

        for k, (chunk, i, j, is_edge) in enumerate(chunks, start=1):
            t0 = time.time()

            # determine PSF radius (1..8) then encode to 0..1
            if params.auto_detect_psf:
                # Measure PSF on the corresponding *pre-stellar* data
                h, w = chunk.shape
                ref = psf_ref_plane[i:i+h, j:j+w]
                if ref.shape != chunk.shape:
                    # safety if at borders (should usually match)
                    ref = ref[:h, :w]
                r = measure_psf_radius(ref, default_radius=3.0)
                r = float(np.clip(r, 1.0, 8.0))
            else:
                r = float(np.clip(params.nonstellar_strength, 1.0, 8.0))

            psf01 = _encode_psf_log2_0_1(r, 1.0, 8.0)

            # infer via conditional model
            if models.is_onnx:
                y = _infer_chunk_psf_onnx(models, models.ns_cond, chunk, psf01)
            else:
                y = _infer_chunk_psf(models, models.ns_cond, chunk, psf01)

            blended = blend_images(chunk, y, params.nonstellar_amount)
            out_chunks.append((blended, i, j, is_edge))

            if _every(k):
                _dbg(f"  nonstellar chunk {k}/{total} r={r:.2f} psf01={psf01:.3f} ({time.time()-t0:.3f}s)",
                     status_cb=lambda *_: None)

            if progress_cb(k, total, "Non-stellar sharpening") is False:
                _dbg("Stage 2: non-stellar CANCELLED", status_cb=lambda *_: None)
                return plane

        plane = stitch_chunks_ignore_border(out_chunks, plane.shape, border_size=16)
        _dbg(f"Stage 2: non-stellar END ({time.time()-t_stage:.2f}s)", status_cb=lambda *_: None)


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
    # ✅ NEW
    temp_stretch: bool = False,
    target_median: float = 0.25,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
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

        # ✅ NEW
        temp_stretch=bool(temp_stretch),
        target_median=float(target_median),
    )

    out, _was_mono = sharpen_image_array(
        image_rgb01,
        params=params,
        progress_cb=_prog,
        status_cb=status_cb,
    )
    return np.asarray(out, dtype=np.float32)
