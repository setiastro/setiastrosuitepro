# src/setiastro/saspro/cosmicclarity_engines/satellite_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from setiastro.saspro.resources import get_resources
from setiastro.saspro.runtime_torch import np_to_torch, torch_to_np, mps_is_usable

from setiastro.saspro.cosmicclarity_engines._backend_guard import (
    DeviceLostError,
    EngineCancelled,
    safe_exception_str,
    is_device_lost_error as _is_device_lost_error,
    is_backend_unusable_error as _is_backend_unusable_error,
    is_recoverable_batch_error as _is_recoverable_batch_error,
    iter_batch_sizes as _iter_batch_sizes,
    device_type as _device_type,
    dml_disabled as _dml_disabled,
    disable_dml as _disable_dml,
    prefer_ort_dml as _prefer_ort_dml,
    cap_chunk_size_for_device as _cap_chunk_size_for_device,
    tick as _tick,
    evict_models as _evict_models,
)

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


# ---------------------------------------------------------------------------
# One GPU, one submitter.
#
# Blink's batch satellite pass drives these models from N worker threads. That
# only ever bought us overlapped file I/O -- PyTorch serializes on the default
# CUDA stream anyway -- but the concurrent dispatches are actively harmful:
#
#   * torch-directml: one D3D12 device taking interleaved dispatches from
#     several threads is what produces "The GPU device instance has been
#     suspended. Use GetDeviceRemovedReason..." The adapter is then gone for
#     the life of the process, which is why a failed run poisons every later
#     run until SASpro restarts.
#   * ORT DirectML EP: Run() is not safe to call concurrently on one session.
#   * CUDA: concurrent first-touch of the cuDNN/cuBLAS handles plus
#     pin_memory() from several threads stalls under the GIL -- the "hang with
#     no further console output".
#
# So every GPU submission goes through this lock. Threads still overlap load,
# normalize, stretch, tile and resize (all CPU, and where the wall clock
# actually goes). It is an RLock so a gated helper can call another.
# CPU bundles are not gated -- torch already threads internally there.
# ---------------------------------------------------------------------------
import threading
from contextlib import nullcontext as _nullcontext_cls

_SAT_GPU_LOCK = threading.RLock()

# _SAT_CACHE is mutated by get_satellite_models() and by _evict_models() on
# device loss, from worker threads. Guard it so two threads can't build the
# same bundle at once, or evict one out from under a third.
_SAT_CACHE_LOCK = threading.RLock()


def _gpu_gate(models: Dict[str, Any]):
    """Context manager serializing GPU submission; a no-op on CPU bundles."""
    if _device_type(models.get("device")) == "cpu":
        return _nullcontext_cls()
    return _SAT_GPU_LOCK


# ---------- Torch import (updated: CUDA + torch-directml awareness) ----------

def _get_torch(*, prefer_cuda: bool, prefer_dml: bool, status_cb=print):
    from setiastro.saspro.runtime_torch import import_torch
    return import_torch(
        prefer_cuda=prefer_cuda,
        prefer_xpu=False,
        prefer_dml=prefer_dml,
        status_cb=status_cb,
    )

def _inference_context(torch, status_cb=None):
    if status_cb:
        try:
            status_cb("[Satellite] using torch.no_grad()")
        except Exception:
            pass
    return torch.no_grad()

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
        #from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
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
            # DirectML can't multiply the two non-contiguous strided views that
            # chunk() returns — it fails with a bare "Unspecified error". Slice
            # into contiguous halves instead. No-op cost on CUDA/CPU/MPS.
            c = x.shape[1] // 2
            x1 = x[:, :c].contiguous()
            x2 = x[:, c:].contiguous()
            return x1 * x2

    class DMLSafePixelShuffle(nn.Module):
        """
        PixelShuffle(r) rewritten as reshape+permute. DirectML's native
        PixelShuffle intermittently fails with a bare "Unspecified error";
        the manual form uses only ops DirectML handles. Numerically identical
        to nn.PixelShuffle, so weights load unchanged. Only used on the
        torch-directml device — CUDA/MPS/CPU keep the faster native op.
        """
        def __init__(self, upscale_factor: int):
            super().__init__()
            self.r = int(upscale_factor)

        def forward(self, x):
            r = self.r
            n, c, h, w = x.shape
            oc = c // (r * r)
            # (N, oc, r, r, H, W)
            x = x.reshape(n, oc, r, r, h, w)
            # -> (N, oc, H, r, W, r)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            # -> (N, oc, H*r, W*r)
            return x.reshape(n, oc, h * r, w * r)

    class GlobalAvgPool2d(nn.Module):
        """Drop-in for nn.AdaptiveAvgPool2d(1) — same reasoning as SimpleGate and
        DMLSafePixelShuffle above. Parameterless, so state_dict keys are
        unchanged (sca.1.weight / sca.1.bias)."""
        def forward(self, x):
            return x.mean(dim=(2, 3), keepdim=True)

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
                GlobalAvgPool2d(),
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
            dml_safe: bool = False,
        ):
            super().__init__()
            _Shuffle = DMLSafePixelShuffle if dml_safe else nn.PixelShuffle
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
                        _Shuffle(2),
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
            self.features = models.resnet18(weights=None)
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
            self.mobilenet = models.mobilenet_v2(weights=None)
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
    # Load on CPU first for maximum backend compatibility (mirrors denoise_engine)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict_from_checkpoint(ckpt)

    msd = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in msd and msd[k].shape == v.shape}

    missing = [k for k in msd.keys() if k not in filtered]
    unexpected = [k for k in state_dict.keys() if k not in msd]

    model.load_state_dict(filtered, strict=False)
    model = model.to(device)

    if missing:
        print(f"[Satellite] {checkpoint_path}: missing {len(missing)} keys after filtering")
    if unexpected:
        print(f"[Satellite] {checkpoint_path}: unexpected {len(unexpected)} keys ignored")

    return model

# ---------- Satellite model cache + loader (updated for torch-directml + ORT DML) ----------

_SAT_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

def get_satellite_models(resources: Any = None, use_gpu: bool = True, status_cb=print) -> Dict[str, Any]:
    """
    Thread-safe entry point. Blink's workers (and satellite_remove_image's CPU
    fallback) call this from several threads at once; without the lock two
    threads race to build the same bundle -- double model load, double VRAM,
    and on DirectML two device probes in flight.
    """
    with _SAT_CACHE_LOCK:
        return _get_satellite_models_unlocked(
            resources=resources, use_gpu=use_gpu, status_cb=status_cb
        )


def _get_satellite_models_unlocked(resources: Any = None, use_gpu: bool = True, status_cb=print) -> Dict[str, Any]:
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
        if use_gpu and is_windows and not _dml_disabled():
            if _prefer_ort_dml() and ort_dml_ok:
                # SASPRO_CC_PREFER_ORT_DML=1. ORT's DML EP dispatches one tile
                # at a time with better op coverage, so it sits much further
                # inside the TDR window than a batched torch-directml pass.
                # This knob existed in _backend_guard but was never wired up
                # here, so setting it did nothing.
                status_cb("CosmicClarity Satellite: SASPRO_CC_PREFER_ORT_DML set — "
                          "routing DirectML through ONNX Runtime.")
                backend = "ort_dml"
            else:
                try:
                    import torch_directml
                    dml = torch_directml.device()
                    _ = (torch.ones(1, device=dml) + 1).to("cpu").item()
                    backend = "torch_dml"
                except Exception as e:
                    # The old bare `except Exception:` discarded every diagnostic.
                    status_cb(f"CosmicClarity Satellite: torch-directml unavailable "
                              f"({safe_exception_str(e)})")
                    backend = "ort_dml" if ort_dml_ok else "cpu"
        elif use_gpu and is_windows:
            status_cb("CosmicClarity Satellite: DirectML disabled this session, using CPU.")
            backend = "cpu"
        else:
            # 4) MPS (macOS) — Apple Silicon only. is_available() is True on
            # Intel Macs with a Metal-capable Radeon, but MPS faults on first
            # use there ("Numpy is not available"); the shared gate returns
            # False on Intel so it falls to CPU.
            if use_gpu and mps_is_usable(torch):
                backend = "mps"
            else:
                backend = "cpu"

    key = (p_det1, p_det2, p_rem, backend)
    if key in _SAT_CACHE:
        return _SAT_CACHE[key]

    # Verify model files exist before attempting to load
    missing = []
    for label, path in [("detect1", p_det1), ("detect2", p_det2), ("remove", p_rem)]:
        if not path or not os.path.exists(str(path)):
            missing.append(f"{label}: {path!r}")
    if missing:
        raise FileNotFoundError(
            f"Satellite model file(s) not found:\n" + "\n".join(missing)
        )
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

    # Log the NORMALIZED device tag alongside the backend name. Every DirectML
    # safeguard (batch caps, tile caps, the disable_dml poison pill) keys off
    # device_type() returning "dml", so a user's log should say outright
    # whether those are live on their hardware instead of leaving us to infer
    # it from a crash.
    _dt = _device_type(device)
    _bs = _satellite_batch_sizes({"device": device}, 256, False)
    status_cb(f"CosmicClarity Satellite: backend={backend} device_type={_dt} "
              f"detect_bs={_bs[0]} remove_bs={_bs[1]}")

    nn, NAFNetSatelliteRemover, BinaryClassificationCNN, BinaryClassificationCNN2, tfm = _build_torch_models(torch)

    det1 = BinaryClassificationCNN(3)
    det1 = _load_model_weights_lenient(torch, nn, det1, p_det1, device).eval()

    det2 = BinaryClassificationCNN2(3)
    det2 = _load_model_weights_lenient(torch, nn, det2, p_det2, device).eval()

    rem = NAFNetSatelliteRemover(
        width=32,
        enc_blk_nums=(2, 4, 6, 8),
        dec_blk_nums=(2, 2, 2, 2),
        middle_blk_num=4,
        residual_out=True,
        dml_safe=(backend == "torch_dml"),
    )
    rem = _load_model_weights_lenient(torch, nn, rem, p_rem, device).eval()

    if backend == "torch_dml":
        try:
            x = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
            with torch.no_grad():
                _ = rem(x)
            status_cb("CosmicClarity Satellite: DirectML model smoke test passed")
        except Exception as e:
            import traceback
            status_cb("CosmicClarity Satellite: DirectML smoke test FAILED — "
                      "falling back to CPU. Details:")
            status_cb(traceback.format_exc())
            # Don't hard-fail the whole engine on a DirectML quirk: rebuild on CPU
            # so the user still gets satellite removal, just slower.
            status_cb(f"CosmicClarity Satellite: retrying on CPU "
                      f"({type(e).__name__}: {e})")
            device = torch.device("cpu")
            det1 = _load_model_weights_lenient(torch, nn, BinaryClassificationCNN(3), p_det1, device).eval()
            det2 = _load_model_weights_lenient(torch, nn, BinaryClassificationCNN2(3), p_det2, device).eval()
            rem  = _load_model_weights_lenient(
                torch, nn,
                NAFNetSatelliteRemover(width=32, enc_blk_nums=(2, 4, 6, 8),
                                       dec_blk_nums=(2, 2, 2, 2),
                                       middle_blk_num=4, residual_out=True),
                p_rem, device).eval()
            backend = "cpu"

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

def _satellite_batch_sizes(models: Dict[str, Any], chunk_size: int, compatibility_mode: bool) -> tuple[int, int]:
    """
    Returns (detect_bs, remove_bs). The old fixed 32/8 was tuned for CUDA and
    applied to every backend. On DirectML that is a multi-second dispatch
    against the Windows TDR watchdog — a latency budget, not a VRAM one.
    """
    dev = _device_type(models.get("device"))
    cs = int(chunk_size)

    if dev == "dml":
        return (4, 1)
    if dev in ("mps", "cpu"):
        return (8, 2) if compatibility_mode else (16, 4)
    if compatibility_mode:
        return (8, 2)
    return (32, 8 if cs <= 256 else 4)


def _with_batch_fallback(fn: Callable[[int], Any], initial_bs: int, *, label: str):
    """
    Run fn(batch_size), halving on recoverable errors. fn must be idempotent —
    both callers below reset their output slots on entry.

    A lost device is never retried: the D3D12 device is gone for the life of the
    process and every further dispatch fails.
    """
    last_err = None
    for bs in _iter_batch_sizes(initial_bs):
        try:
            return fn(bs)
        except EngineCancelled:
            raise
        except Exception as e:
            last_err = e
            if _is_device_lost_error(e):
                raise DeviceLostError(safe_exception_str(e)) from e
            if bs == 1 or not _is_recoverable_batch_error(e):
                break
            print(f"[Satellite] {label} failed at batch_size={bs}, retrying smaller. {safe_exception_str(e)}")

    if last_err is not None and _is_device_lost_error(last_err):
        raise DeviceLostError(safe_exception_str(last_err)) from last_err
    raise last_err


def _safe_reflect_mode_for_hw(h: int, w: int) -> str:
    # np.pad(..., mode="reflect") requires dimensions > 1
    if h <= 1 or w <= 1:
        return "edge"
    return "reflect"


def _reflect_pad_full_rgb(image_rgb: np.ndarray, pad: int = 512) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Reflect-pad an HxWx3 image on all sides.
    Returns:
        padded_rgb, (top, bottom, left, right)
    """
    x = np.asarray(image_rgb, dtype=np.float32)
    if pad <= 0:
        return x, (0, 0, 0, 0)

    h, w = x.shape[:2]
    mode = _safe_reflect_mode_for_hw(h, w)

    xp = np.pad(
        x,
        ((pad, pad), (pad, pad), (0, 0)),
        mode=mode,
    )
    return xp.astype(np.float32), (pad, pad, pad, pad)


def _crop_full_rgb_pad(image_rgb: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop back an HxWx3 image using (top, bottom, left, right).
    """
    top, bottom, left, right = pads
    if top == bottom == left == right == 0:
        return np.asarray(image_rgb, dtype=np.float32)

    h, w = image_rgb.shape[:2]
    y0 = top
    y1 = h - bottom if bottom > 0 else h
    x0 = left
    x1 = w - right if right > 0 else w
    return np.asarray(image_rgb[y0:y1, x0:x1, :], dtype=np.float32)


def _crop_full_mask_pad(mask2d: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop back a 2D trail mask using (top, bottom, left, right).
    """
    top, bottom, left, right = pads
    if top == bottom == left == right == 0:
        return np.asarray(mask2d, dtype=bool)

    h, w = mask2d.shape[:2]
    y0 = top
    y1 = h - bottom if bottom > 0 else h
    x0 = left
    x1 = w - right if right > 0 else w
    return np.asarray(mask2d[y0:y1, x0:x1], dtype=bool)

# ----------------------------
#   Core processing
# ----------------------------
def _normalize_for_satellite(img: np.ndarray) -> Tuple[np.ndarray, bool, float, float]:
    """
    Input: HxW, HxWx1, or HxWx3; float/uint with arbitrary calibrated range.
    Output:
      rgb01     : HxWx3 float32 in [0,1]
      is_mono   : whether original input was mono-like
      orig_min  : original minimum value before normalization
      scale     : max(image - orig_min); output can be denormalized with out*scale + orig_min
    """
    a = np.asarray(img)
    is_mono = (a.ndim == 2) or (a.ndim == 3 and a.shape[2] == 1)

    a = np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    orig_min = float(np.min(a)) if a.size else 0.0
    shifted = a - orig_min
    scale = float(np.max(shifted)) if shifted.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    a01 = shifted / scale

    if a01.ndim == 2:
        a01 = np.stack([a01, a01, a01], axis=-1)
    elif a01.ndim == 3 and a01.shape[2] == 1:
        a01 = np.repeat(a01, 3, axis=2)
    elif a01.ndim == 3 and a01.shape[2] >= 3:
        a01 = a01[..., :3]
    else:
        raise ValueError(f"Unsupported image shape: {a.shape}")

    a01 = np.clip(a01, 0.0, 1.0).astype(np.float32)
    return a01, is_mono, orig_min, scale


def _denormalize_from_satellite(rgb01: np.ndarray, orig_min: float, scale: float) -> np.ndarray:
    """
    Undo _normalize_for_satellite() for an RGB image in [0,1].
    """
    out = np.asarray(rgb01, np.float32) * float(scale) + float(orig_min)
    return out.astype(np.float32)


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

def _stitch_mask_ignore_border(mask_chunks, shape_hw, border: int = 16) -> np.ndarray:
    """
    mask_chunks: list of (mask2d_bool, y0, x0)
    OR-stitches masks while ignoring tile borders, matching _stitch_ignore_border().
    """
    H, W = shape_hw
    out = np.zeros((H, W), dtype=bool)

    for tile_mask, y0, x0 in mask_chunks:
        th, tw = tile_mask.shape[:2]
        bh = min(border, th // 2)
        bw = min(border, tw // 2)

        inner = tile_mask[bh:th-bh, bw:tw-bw]
        out[y0+bh:y0+th-bh, x0+bw:x0+tw-bw] |= inner

    return out

def _apply_clip_trail_logic(
    processed: np.ndarray,
    original: np.ndarray,
    sensitivity: float,
    *,
    return_mask: bool = False,
):
    """
    Original standalone clip logic, but now can also return the authoritative binary trail mask.

    processed, original: HxWx3 float32 in [0,1]
    return:
      if return_mask=False -> final_rgb01
      if return_mask=True  -> (final_rgb01, trail_mask_2d_bool)
    """
    sattrail_only = original - processed
    mean_val = float(np.mean(sattrail_only))
    clipped = np.clip((sattrail_only - mean_val) * 10.0, 0.0, 1.0)

    mask3 = np.where(clipped < sensitivity, 0.0, 1.0).astype(np.float32)
    final = np.clip(original - mask3, 0.0, 1.0).astype(np.float32)

    if not return_mask:
        return final

    trail_mask_2d = np.any(mask3 > 0.5, axis=-1)
    return final, trail_mask_2d

def trail_mask_requires_rewrite(sat_mask) -> bool:
    """True if the satellite pass clipped pixels and the calibrated FITS must be rewritten."""
    if sat_mask is None:
        return False
    return bool(np.any(sat_mask))


def _resize_tile_for_detect(tile_rgb01: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile_rgb01, dtype=np.float32)
    if tile.ndim == 3 and tile.shape[0] == 256 and tile.shape[1] == 256 and tile.shape[2] == 3:
        return tile
    if _sk_resize is None:
        raise RuntimeError("skimage.transform.resize is required for satellite detection.")
    r = _sk_resize(tile, (256, 256, 3), mode="reflect", anti_aliasing=True).astype(np.float32)
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

        x = np_to_torch(batch_np, torch=torch)
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
            keep2 = torch_to_np((o2 > 0.25).detach().float()).astype(bool)

            idxs_cpu = torch_to_np(idxs.detach().float()).astype(np.int64)
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

        x = np_to_torch(batch_np, torch=torch)
        if hasattr(device, "type") and device.type == "cuda":
            x = x.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
        else:
            x = x.to(device=device, dtype=torch.float32)

        with torch.no_grad(), _autocast_context(torch, device):
            y = torch_to_np(rem(x).detach().float())  # NCHW

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

    x = np_to_torch(tile, device=device, dtype=torch.float32, torch=torch).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad(), _autocast_context(torch, device):
        out = torch_to_np(rem(x).squeeze(0).detach().float()).transpose(1, 2, 0)

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

# ---------------------------------------------------------------------------
# Fixed-pattern artifact map
#
# Satellite trails are TRANSIENT: a given sensor position is crossed in one
# frame, maybe two. Diffraction spikes and amp glow are PERSISTENT: they sit in
# the same SENSOR position in every frame of a session. So a tile that keeps
# passing both detectors across a whole set is almost certainly not a satellite.
#
# The map is kept at PIXEL resolution in unpadded image coordinates rather than
# keyed on tile indices, because the detect and removal passes do not share a
# tile grid: the removal path reflect-pads the image by 512px before tiling and
# uses the user's overlap (default 64), while the detect path tiles the raw
# image with overlap 0. Identical tile indices in the two passes therefore
# cover completely different pixels. A pixel map is immune to both.
#
# Limits worth knowing: this is stationary in SENSOR coordinates, so it is
# exact for amp glow, good for diffraction spikes under ordinary dithering, and
# degrades under field rotation (alt-az rigs) where spikes sweep across tiles.
# The failure mode is a missed veto, never a false one.
# ---------------------------------------------------------------------------

ARTIFACT_MIN_FRAMES = 5      # below this there is no statistic; do not veto
ARTIFACT_FRAC = 0.8          # flagged in >= this fraction of frames == fixed
ARTIFACT_TILE_FRAC = 0.7     # tile is vetoed when >= this much of it is artifact


_GROUP_EXP_KEYS = ("EXPTIME", "EXPOSURE", "XPOSURE", "EXP-TIME", "EXP_TIME",
                   "ITIME", "INTTIME", "EXP")


def artifact_group_key(header, shape_hw) -> tuple:
    """THE definition of which frames may be pooled for the fixed-pattern
    statistic. Every call site must use this one — there is no second copy.

    Mirrors the Object / Filter / Exposure grouping Blink's tree already
    shows, plus sensor dimensions and instrument.

    Filter is not optional. Amp glow is a property of the sensor AND the
    exposure, so a 20-frame Ha set full of glow pooled with 20 clean Green
    frames puts the glow in 20 of 40 frames -- under any sane persistence
    threshold, so it is NOT recognised as fixed-pattern and all 20 Ha frames
    get flagged as satellite trails. Grouping by shape alone does exactly
    that whenever two filters come off one camera.
    """
    h = header or {}

    def _get(*keys):
        for k in keys:
            try:
                if k in h and h[k] is not None:
                    return h[k]
            except Exception:
                pass
            try:
                v = h.get(k)
                if v is not None:
                    return v
            except Exception:
                pass
        return None

    exp = _get(*_GROUP_EXP_KEYS)
    try:
        s = str(exp).strip().rstrip("sS").strip()
        exp_key = round(float(s), 3)
    except Exception:
        exp_key = "Unknown"

    return (
        tuple(int(v) for v in shape_hw[:2]),
        str(_get("OBJECT") or "Unknown"),
        str(_get("FILTER") or "Unknown"),
        exp_key,
        str(_get("INSTRUME") or ""),
    )


def new_artifact_accumulator(shape_hw) -> np.ndarray:
    """Per-pixel hit counter for one GROUP of frames (one sensor, one shape)."""
    return np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.int32)


def accumulate_artifact_tiles(acc: np.ndarray, tiles) -> None:
    """Add ONE frame's passing tiles. `tiles` is an iterable of
    (y0, x0, h, w) rectangles in UNPADDED image coordinates.

    The frame's tiles are unioned into a boolean hit map first and that is
    added, so a pixel contributes AT MOST 1 per frame. Adding rectangles
    directly would double-count wherever two passing tiles overlap, letting a
    pixel's count exceed the frame total and cross the threshold on a single
    frame -- which matters as soon as a caller scans with a non-zero overlap.
    """
    if acc is None or not tiles:
        return
    H, W = acc.shape[:2]
    hit = np.zeros((H, W), dtype=bool)
    for (y0, x0, h, w) in tiles:
        y0 = max(0, int(y0)); x0 = max(0, int(x0))
        y1 = min(H, y0 + int(h)); x1 = min(W, x0 + int(w))
        if y1 > y0 and x1 > x0:
            hit[y0:y1, x0:x1] = True
    acc += hit


def finalize_artifact_mask(
    acc: np.ndarray,
    n_frames: int,
    *,
    min_frames: int = ARTIFACT_MIN_FRAMES,
    frac: float = ARTIFACT_FRAC,
):
    """
    Returns (mask_or_None, threshold_used).

    None when the group is too small to support the statistic, or when nothing
    is persistent. Callers must treat None as "no veto" and fall back to
    ordinary single-pass behavior.
    """
    n = int(n_frames)
    if acc is None or n < int(min_frames):
        return None, 0
    thr = max(2, int(np.ceil(float(frac) * n)))
    mask = (acc >= thr)
    if not bool(mask.any()):
        return None, thr
    return mask, thr


def tile_is_vetoed(mask, y0: int, x0: int, h: int, w: int,
                   *, frac: float = ARTIFACT_TILE_FRAC) -> bool:
    """True when most of this tile sits inside the fixed-pattern mask.
    Coordinates must be in the SAME frame of reference as `mask`."""
    if mask is None:
        return False
    H, W = mask.shape[:2]
    y0 = max(0, int(y0)); x0 = max(0, int(x0))
    y1 = min(H, y0 + int(h)); x1 = min(W, x0 + int(w))
    if y1 <= y0 or x1 <= x0:
        return False
    sub = mask[y0:y1, x0:x1]
    if sub.size == 0:
        return False
    return bool(float(sub.mean()) >= float(frac))


def artifact_tile_count(mask, chunk: int = 256, frac: float = 0.5) -> int:
    """How many chunk-sized tiles the mask covers. Reporting only."""
    if mask is None:
        return 0
    H, W = mask.shape[:2]
    chunk = max(1, int(chunk))
    n = 0
    for y0 in range(0, H, chunk):
        for x0 in range(0, W, chunk):
            sub = mask[y0:min(H, y0 + chunk), x0:min(W, x0 + chunk)]
            if sub.size and float(sub.mean()) >= float(frac):
                n += 1
    return n


class ArtifactMapBuilder:
    """Accumulates one detection scan and splits its frames into clean vs real.

    Every caller does the same four things after scanning: count tiles per
    group, threshold into a mask, test each frame's tiles against it, and sort
    the frames into "nothing real here" and "these tiles are real". That logic
    lived in three copies -- Blink, the Cosmic Clarity batch, Calibrate_Lights
    -- and the copies drifted, which is how Highlight mode silently stopped
    rejecting fixed-pattern tiles while Fast kept working. One copy, here.

    Callers still own their own loading and threading, which genuinely differ;
    they only hand results in via add() and ask for the verdict via finalize().

    Usage:
        b = ArtifactMapBuilder()
        for item in ...:
            b.add(item_id, header, shape_hw, tiles)     # tiles from
                                                        # satellite_detect_tiles
        clean, real, n_tiles = b.finalize(status_cb=log)

      clean  set of item ids with NO real trail -- never load these again
      real   {item id: [real tiles]} -- hand straight to satellite_remove_image
             as detect_tiles so it runs no detection of its own
    """

    def __init__(self, *, min_frames: int = ARTIFACT_MIN_FRAMES,
                 frac: float = ARTIFACT_FRAC,
                 tile_frac: float = ARTIFACT_TILE_FRAC):
        self._min_frames = int(min_frames)
        self._frac = float(frac)
        self._tile_frac = float(tile_frac)
        self._groups: Dict[tuple, list] = {}      # gkey -> [acc, n]
        self._items: list = []                    # (item_id, gkey, tiles)

    def add(self, item_id, header, shape_hw, tiles) -> tuple:
        """Record one scanned frame. Returns the group key it landed in."""
        gkey = artifact_group_key(header, shape_hw)
        slot = self._groups.get(gkey)
        if slot is None:
            slot = [new_artifact_accumulator(gkey[0]), 0]
            self._groups[gkey] = slot
        accumulate_artifact_tiles(slot[0], tiles)
        slot[1] += 1
        self._items.append((item_id, gkey, list(tiles or [])))
        return gkey

    @property
    def n_groups(self) -> int:
        return len(self._groups)

    def finalize(self, status_cb=None):
        """Returns (clean_ids, real_by_id, n_artifact_tiles)."""
        def _say(msg):
            if status_cb:
                try:
                    status_cb(msg)
                except Exception:
                    pass

        masks: Dict[tuple, Any] = {}
        n_art = 0
        for gkey, (acc, n) in self._groups.items():
            mask, thr = finalize_artifact_mask(
                acc, n, min_frames=self._min_frames, frac=self._frac)
            masks[gkey] = mask
            lbl = f"{gkey[1]}/{gkey[2]}/{gkey[3]}"
            if mask is not None:
                k = artifact_tile_count(mask)
                n_art += k
                _say(f"[{lbl}] {n} frames, threshold {thr}: "
                     f"{k} fixed-pattern tile(s) ignored")
            elif n < self._min_frames:
                _say(f"[{lbl}] only {n} frame(s) — too few for fixed-pattern "
                     f"analysis (needs {self._min_frames})")
            else:
                _say(f"[{lbl}] {n} frames: nothing persistent")

        clean = set()
        real: dict = {}
        for item_id, gkey, tiles in self._items:
            if not tiles:
                clean.add(item_id)
                continue
            mask = masks.get(gkey)
            survivors = tiles if mask is None else [
                t for t in tiles
                if not tile_is_vetoed(mask, t[0], t[1], t[2], t[3],
                                      frac=self._tile_frac)
            ]
            if survivors:
                real[item_id] = survivors
            else:
                clean.add(item_id)
        return clean, real, n_art


def _rects_overlap(ay: int, ax: int, ah: int, aw: int, b) -> bool:
    """Do tile rect A and rect B share any pixel? Both (y0, x0, h, w)."""
    by, bx, bh, bw = b
    return not (ay + ah <= by or by + bh <= ay or ax + aw <= bx or bx + bw <= ax)


def _apply_tile_veto(detected_indices, all_tiles, veto_mask, veto_frac):
    """
    All-or-nothing, deliberately.

    If ANY detected tile survives the veto, keep them ALL: the frame genuinely
    contains a trail, and skipping its other detected tiles would leave a hard
    clipped/unclipped edge part-way along that trail -- a new artifact worse
    than either extreme. Only when EVERY detection sits on known fixed-pattern
    structure is the frame treated as clean.
    """
    if veto_mask is None or not detected_indices:
        return detected_indices
    for i in detected_indices:
        tile, y0, x0 = all_tiles[i]
        th, tw = tile.shape[:2]
        if not tile_is_vetoed(veto_mask, y0, x0, th, tw, frac=veto_frac):
            return detected_indices
    return []


def satellite_detect_tiles(
    image: np.ndarray,
    models: Dict[str, Any],
    *,
    chunk_size: int = 256,
    overlap: int = 0,
    border_size: int = 16,
    temp_stretch: bool = True,
    target_median: float = 0.25,
    compatibility_mode: bool = False,
    stop_after: Optional[int] = None,
    status_cb=print,
) -> list:
    """
    Detection-only pass: det1 + det2 over INTERIOR tiles, no NAFNet removal.

    Returns the list of (y0, x0, h, w) tiles that passed both classifiers, in
    UNPADDED image coordinates.

    `stop_after=N` early-exits once N tiles have passed -- the right choice when
    the caller only wants a yes/no. Pass None to evaluate EVERY tile, which the
    artifact map needs: an early exit would bias the per-pixel counts toward
    whichever tiles happen to be scanned first.

    Raises DeviceLostError when the adapter is gone: the caller MUST abort the
    batch rather than record the frame as clean.
    """
    chunk_size = _cap_chunk_size_for_device(models.get("device"), int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    limit = None if stop_after is None else max(1, int(stop_after))

    rgb01, _was_mono, _orig_min, _scale = _normalize_for_satellite(image)
    H, W = rgb01.shape[:2]

    arr = rgb01.astype(np.float32)
    if bool(temp_stretch) or float(np.median(arr - np.min(arr))) < 0.05:
        arr, _smin, _smeds = stretch_image(arr, target_median=float(target_median))

    interior = [
        (tile, y0, x0)
        for (tile, y0, x0) in _split_chunks(arr, chunk_size, overlap)
        if not _is_border_tile(
            y0, x0, y0 + tile.shape[0], x0 + tile.shape[1], H, W, border_size
        )
    ]
    if not interior:
        return []

    coords = [(int(y0), int(x0), int(t.shape[0]), int(t.shape[1]))
              for (t, y0, x0) in interior]
    tiles_only = [t.astype(np.float32) for (t, _, _) in interior]
    is_onnx = bool(models.get("is_onnx", False))
    detect_bs, _remove_bs = _satellite_batch_sizes(models, chunk_size, compatibility_mode)

    # ---- ONNX (ORT DirectML EP) ----
    if is_onnx:
        det1_sess = models["detection_model1"]
        det2_sess = models["detection_model2"]
        hits: list = []
        with _gpu_gate(models):
            for j, tile in enumerate(tiles_only):
                if _onnx_detect(tile, det1_sess) and _onnx_detect(tile, det2_sess):
                    hits.append(coords[j])
                    if limit is not None and len(hits) >= limit:
                        return hits
        return hits

    # ---- Torch (CUDA / torch-directml / MPS / CPU) ----
    # Resize OUTSIDE the GPU gate: it is pure CPU/skimage work and is exactly
    # what we want the worker threads overlapping with each other.
    resized = [_resize_tile_for_detect(t) for t in tiles_only]

    torch = models["torch"]
    device = models["device"]
    det1 = models["detection_model1"]
    det2 = models["detection_model2"]

    def _detect_pass(bs: int) -> list:
        # Fresh list each call: _with_batch_fallback may re-run this whole pass
        # at a smaller batch size, so it must be idempotent.
        found: list = []
        for i in range(0, len(resized), bs):
            batch_np = np.stack(resized[i:i + bs], axis=0)
            batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)

            x = np_to_torch(batch_np, torch=torch)
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
                keep2 = torch_to_np((o2 > 0.25).detach().float()).astype(bool)
                idxs_cpu = torch_to_np(idxs.detach().float()).astype(np.int64)
                for local_j, passed in zip(idxs_cpu, keep2):
                    if passed:
                        found.append(coords[i + int(local_j)])
                if limit is not None and len(found) >= limit:
                    return found
        return found

    with _gpu_gate(models):
        return _with_batch_fallback(_detect_pass, detect_bs, label="detection")


def satellite_detect_image(
    image: np.ndarray,
    models: Dict[str, Any],
    *,
    min_tiles: int = 1,
    **kwargs,
) -> bool:
    """Yes/no wrapper over satellite_detect_tiles, with the early exit on."""
    tiles = satellite_detect_tiles(
        image, models, stop_after=int(min_tiles), **kwargs
    )
    return len(tiles) >= int(min_tiles)


def satellite_remove_image(
    image: np.ndarray,
    models: Dict[str, Any],
    *,
    status_cb=print,
    **kwargs,
):
    """
    Public entry point. Owns the tile-size cap and the device-loss / CPU-fallback
    policy; _satellite_remove_image_impl does the work against whichever bundle
    it is handed. Every other keyword passes straight through.
    """
    req_chunk = int(kwargs.get("chunk_size", 256))
    req_overlap = int(kwargs.get("overlap", 64))

    def _fit(bundle):
        cs = _cap_chunk_size_for_device(bundle.get("device"), req_chunk)
        out = dict(kwargs)
        out["chunk_size"] = cs
        out["overlap"] = min(req_overlap, max(0, cs - 1))
        return out

    eff = _fit(models)
    if eff["chunk_size"] != req_chunk:
        status_cb(f"[Satellite] tile size reduced {req_chunk} -> {eff['chunk_size']} on "
                  f"backend={_device_type(models.get('device'))} to stay inside the TDR window.")

    try:
        return _satellite_remove_image_impl(image, models, **eff)

    except EngineCancelled:
        raise

    except Exception as e:
        dev = _device_type(models.get("device"))
        hard = _is_device_lost_error(e)        # adapter gone — poison DirectML
        soft = _is_backend_unusable_error(e)   # backend can't run it — fall back only

        if dev != "cpu" and (hard or soft):
            status_cb(f"[Satellite] GPU path failed on backend={dev}: {safe_exception_str(e)}")

            if hard and dev == "dml":
                #sasprocess-wide. Denoise, sharpen and Dark Star must stop handing
                # out DirectML for the rest of this session too.
                _disable_dml("Adapter was removed or hung during satellite removal.",
                             status_cb=status_cb)

            if hard:
                # Another worker thread may be mid-inference against this same
                # bundle; the cache lock keeps the dict mutation itself sane.
                with _SAT_CACHE_LOCK:
                    _evict_models(_SAT_CACHE, models, tag="Satellite", status_cb=status_cb)

            status_cb("[Satellite] retrying on CPU.")
            cpu_models = get_satellite_models(use_gpu=False, status_cb=status_cb)
            return _satellite_remove_image_impl(image, cpu_models, **_fit(cpu_models))

        raise


def _satellite_remove_image_impl(
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
    return_mask: bool = False,
    veto_mask=None,
    veto_frac: float = ARTIFACT_TILE_FRAC,
    detect_tiles=None,
    status_cb=print,
):
    """
    image: input image with arbitrary calibrated range

    detect_tiles: optional list of (y0, x0, h, w) rects, in UNPADDED image
      coordinates, that a previous detection pass already established hold
      REAL trails. When given, the detection models are NOT run at all: only
      tiles overlapping these rects go through NAFNet and everything else is
      passed through untouched. This is the whole point of the two-pass
      design -- pass 1 already answered "where", so pass 2 re-running det1 and
      det2 over every tile would just rediscover it at full cost.

    veto_mask: optional bool array, SAME HxW as `image`, marking fixed-pattern
      structure (diffraction spikes / amp glow) established across a whole set
      by the artifact-map helpers above. A frame whose detections ALL land on
      it is treated as clean and never touched. None disables the veto, which
      is the only correct behavior for a single image.

    Returns:
      default: (out_image_same_shape_style, trail_detected_any)
      if return_mask=True: (out_image_same_shape_style, trail_detected_any, trail_mask_2d_bool)
    """
    rgb01, was_mono, orig_min, scale = _normalize_for_satellite(image)
    tm = float(np.clip(target_median, 0.01, 0.50))

    # Whole-image reflect padding for better edge behavior, like denoise
    full_pad = 512
    rgb01_work, full_pads = _reflect_pad_full_rgb(rgb01, pad=full_pad)

    # Pad the veto mask the SAME way so tile coordinates inside
    # _satellite_remove_rgb (which are in padded space) index it directly.
    # Constant False, not reflect: the pad is synthetic and border tiles are
    # excluded from trail_any anyway.
    veto_work = None
    if veto_mask is not None:
        _vm = np.asarray(veto_mask, dtype=bool)
        if _vm.shape[:2] == rgb01.shape[:2]:
            veto_work = np.pad(
                _vm, ((full_pad, full_pad), (full_pad, full_pad)),
                mode="constant", constant_values=False,
            ) if full_pad > 0 else _vm
        else:
            try:
                status_cb(f"[Satellite] veto mask shape {_vm.shape[:2]} != image "
                          f"{rgb01.shape[:2]} — ignoring it for this frame.")
            except Exception:
                pass

    # Shift known-trail rects into the padded frame the tiler works in.
    force_tiles = None
    if detect_tiles:
        force_tiles = [
            (int(y0) + full_pad, int(x0) + full_pad, int(h), int(w))
            for (y0, x0, h, w) in detect_tiles
        ]

    def _should_stretch(x: np.ndarray) -> bool:
        x = np.asarray(x, np.float32)
        return bool(np.median(x - np.min(x)) < 0.05)

    if mode.lower() == "luminance":
        y, cb, cr = _extract_luminance_bt601(rgb01_work)

        if bool(temp_stretch):
            stretch_needed = True
        else:
            stretch_needed = _should_stretch(y)

        if stretch_needed:
            y_s, stretch_min, stretch_meds = stretch_image(y, target_median=tm)
        else:
            y_s = y.astype(np.float32)
            stretch_min = float(np.min(y))
            stretch_meds = [float(np.median(y_s))]

        y3 = np.stack([y_s, y_s, y_s], axis=-1)

        out3, detected, trail_mask = _satellite_remove_rgb(
            y3,
            models,
            clip_trail=clip_trail,
            sensitivity=sensitivity,
            chunk_size=chunk_size,
            overlap=overlap,
            border_size=border_size,
            compatibility_mode=compatibility_mode,
            progress_cb=progress_cb,
            veto_mask=veto_work,
            veto_frac=veto_frac,
            force_tiles=force_tiles,
        )

        out_y = out3[..., 0]

        if stretch_needed:
            out_y = unstretch_image(out_y, stretch_meds, stretch_min, target_median=tm)

        out_y = np.clip(out_y, 0.0, 1.0).astype(np.float32)
        out_rgb01 = _merge_luminance_bt601(out_y, cb, cr)

        # Crop padded result back to original image extent
        out_rgb01 = _crop_full_rgb_pad(out_rgb01, full_pads)
        trail_mask = _crop_full_mask_pad(trail_mask, full_pads)

    else:
        # FULL-COLOR path must use the padded image too
        if bool(temp_stretch):
            stretch_needed = True
        else:
            stretch_needed = _should_stretch(rgb01_work)

        if stretch_needed:
            rgb_s, stretch_min, stretch_meds = stretch_image(rgb01_work, target_median=tm)
        else:
            rgb_s = rgb01_work.astype(np.float32)
            stretch_min = float(np.min(rgb01_work))
            stretch_meds = [float(np.median(rgb01_work[..., c])) for c in range(3)]

        out_rgb01, detected, trail_mask = _satellite_remove_rgb(
            rgb_s,
            models,
            clip_trail=clip_trail,
            sensitivity=sensitivity,
            chunk_size=chunk_size,
            overlap=overlap,
            border_size=border_size,
            compatibility_mode=compatibility_mode,
            progress_cb=progress_cb,
            veto_mask=veto_work,
            veto_frac=veto_frac,
            force_tiles=force_tiles,
        )

        if stretch_needed:
            out_rgb01 = unstretch_image(out_rgb01, stretch_meds, stretch_min, target_median=tm)

        out_rgb01 = np.clip(out_rgb01, 0.0, 1.0).astype(np.float32)

        # Crop padded result back to original image extent
        out_rgb01 = _crop_full_rgb_pad(out_rgb01, full_pads)
        trail_mask = _crop_full_mask_pad(trail_mask, full_pads)

    # Undo calibrated-image normalization back to original range
    out_rgb = _denormalize_from_satellite(out_rgb01, orig_min, scale)

    # "Clip Satellite Trail to 0.000" must produce a TRUE zero in the output range.
    # The trail is driven to 0.0 in NORMALIZED [0,1] space, but denormalize() adds
    # orig_min back — so clipped pixels land at min(image), not 0.000. Force the
    # authoritative trail-mask pixels to literal 0.0 after denormalization.
    if clip_trail:
        tmask = np.asarray(trail_mask, dtype=bool)
        if tmask.any():
            out_rgb[tmask] = 0.0

    # Preserve original mono-style convention
    src = np.asarray(image)
    if (src.ndim == 2) or (src.ndim == 3 and src.shape[2] == 1):
        out_m = out_rgb[..., 0:1].astype(np.float32)
        if return_mask:
            return out_m, detected, np.asarray(trail_mask, dtype=bool)
        return out_m, detected

    out_rgb = out_rgb.astype(np.float32)
    if return_mask:
        return out_rgb, detected, np.asarray(trail_mask, dtype=bool)
    return out_rgb, detected

def _finish_satellite_tiles(out_tiles, mask_tiles, rgb01, H, W, border_size, trail_any):
    """Stitch, restore the untouched border, and return. Shared by the normal
    detect-then-remove path and the known-tile fast path."""
    out = _stitch_ignore_border(out_tiles, (H, W, 3), border=border_size)
    trail_mask_full = _stitch_mask_ignore_border(mask_tiles, (H, W), border=border_size)

    if border_size > 0:
        out[:border_size, :, :] = rgb01[:border_size, :, :]
        out[-border_size:, :, :] = rgb01[-border_size:, :, :]
        out[:, :border_size, :] = rgb01[:, :border_size, :]
        out[:, -border_size:, :] = rgb01[:, -border_size:, :]

        trail_mask_full[:border_size, :] = False
        trail_mask_full[-border_size:, :] = False
        trail_mask_full[:, :border_size] = False
        trail_mask_full[:, -border_size:] = False

    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    if not trail_any:
        return rgb01.astype(np.float32), False, np.zeros((H, W), dtype=bool)

    return out, True, trail_mask_full


def _remove_known_tiles(all_tiles, detected_indices, out_tiles, mask_tiles, models,
                        *, chunk_size, clip_trail, sensitivity,
                        compatibility_mode, progress_cb, total):
    """NAFNet removal on a KNOWN tile set — no detection models are run.

    Used by the two-pass flow, where pass 1 already established exactly which
    tiles hold real trails.
    """
    is_onnx = bool(models.get("is_onnx", False))
    _detect_bs, remove_bs = _satellite_batch_sizes(models, chunk_size, compatibility_mode)
    done = [0]

    def _store(src_idx, pred):
        orig, y0, x0 = all_tiles[src_idx]
        orig = orig.astype(np.float32)
        if clip_trail:
            final, trail_mask = _apply_clip_trail_logic(
                pred, orig, sensitivity, return_mask=True
            )
        else:
            final = pred
            trail_mask = np.zeros(orig.shape[:2], dtype=bool)
        out_tiles[src_idx] = (final, y0, x0)
        mask_tiles[src_idx] = (trail_mask, y0, x0)
        done[0] += 1

    if is_onnx:
        rem_sess = models["removal_model"]
        with _gpu_gate(models):
            for i in detected_indices:
                orig = all_tiles[i][0].astype(np.float32)
                _store(i, _onnx_remove(orig, rem_sess))
                _tick(progress_cb, total, total)
        return

    torch = models["torch"]
    device = models["device"]
    rem = models["removal_model"]

    padded_tiles = []
    orig_shapes = []
    for idx in detected_indices:
        padded, h, w = _pad_tile_to_shape_rgb(
            all_tiles[idx][0].astype(np.float32), chunk_size, chunk_size
        )
        padded_tiles.append(padded)
        orig_shapes.append((h, w))

    def _remove_pass(bs: int) -> None:
        done[0] = 0
        for batch_start in range(0, len(padded_tiles), bs):
            batch_np = np.stack(padded_tiles[batch_start:batch_start + bs], axis=0)
            batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)

            x = np_to_torch(batch_np, torch=torch)
            if hasattr(device, "type") and device.type == "cuda":
                x = x.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            else:
                x = x.to(device=device, dtype=torch.float32)

            with torch.no_grad(), _autocast_context(torch, device):
                y = torch_to_np(rem(x).detach().float())

            y = np.transpose(y, (0, 2, 3, 1))
            for local_j, pred in enumerate(y):
                src_idx = detected_indices[batch_start + local_j]
                h, w = orig_shapes[batch_start + local_j]
                _store(src_idx, np.clip(pred[:h, :w, :], 0.0, 1.0).astype(np.float32))

            _tick(progress_cb, total, total)

    with _gpu_gate(models):
        _with_batch_fallback(_remove_pass, remove_bs, label="removal")


def _is_border_tile(y0: int, x0: int, y1: int, x1: int, H: int, W: int, border_size: int = 16) -> bool:
    """Returns True if this tile overlaps the image border region."""
    return y0 < border_size or x0 < border_size or y1 > H - border_size or x1 > W - border_size
# ---------- Satellite remove loop (FIX: use correct ONNX sessions, not det1/rem confusion) ----------
def _satellite_remove_rgb(rgb01: np.ndarray, models: Dict[str, Any], **kw):
    """
    GPU gate for the removal pass.

    The ONNX branch interleaves detect+remove per tile, so there is no useful
    place to gate more finely -- serialize the whole pass. The torch branch
    gates its two inference passes individually (see _impl) so the CPU tiling
    and resize work still overlaps across worker threads.
    """
    if bool(models.get("is_onnx", False)):
        with _gpu_gate(models):
            return _satellite_remove_rgb_impl(rgb01, models, **kw)
    return _satellite_remove_rgb_impl(rgb01, models, **kw)


def _satellite_remove_rgb_impl(
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
    veto_mask=None,
    veto_frac: float = ARTIFACT_TILE_FRAC,
    force_tiles=None,
) -> Tuple[np.ndarray, bool, np.ndarray]:
    is_onnx = bool(models.get("is_onnx", False))

    H, W = rgb01.shape[:2]
    trail_any = False

    all_tiles = list(_split_chunks(rgb01, chunk_size, overlap))
    total = len(all_tiles)
    out_tiles = [None] * total
    mask_tiles = [None] * total

    # ── Known-tile fast path ────────────────────────────────────────────────
    # A previous detection pass already told us which tiles hold real trails,
    # so run NO detection here: mark the tiles overlapping those rects, pass
    # every other tile straight through, and go directly to removal.
    if force_tiles is not None:
        detected_indices = []
        for i, (tile, y0, x0) in enumerate(all_tiles):
            th, tw = tile.shape[:2]
            if any(_rects_overlap(y0, x0, th, tw, r) for r in force_tiles):
                detected_indices.append(i)
            else:
                out_tiles[i] = (tile.astype(np.float32), y0, x0)
                mask_tiles[i] = (np.zeros((th, tw), dtype=bool), y0, x0)

        trail_any = any(
            not _is_border_tile(
                all_tiles[i][1], all_tiles[i][2],
                all_tiles[i][1] + all_tiles[i][0].shape[0],
                all_tiles[i][2] + all_tiles[i][0].shape[1],
                H, W, border_size
            )
            for i in detected_indices
        )

        if detected_indices:
            _remove_known_tiles(
                all_tiles, detected_indices, out_tiles, mask_tiles, models,
                chunk_size=chunk_size, clip_trail=clip_trail,
                sensitivity=sensitivity, compatibility_mode=compatibility_mode,
                progress_cb=progress_cb, total=total,
            )
        else:
            _tick(progress_cb, total, total)

        return _finish_satellite_tiles(
            out_tiles, mask_tiles, rgb01, H, W, border_size, trail_any
        )

    # ---------------- ONNX path ----------------
    if is_onnx:
        det1_sess = models["detection_model1"]
        det2_sess = models["detection_model2"]
        rem_sess  = models["removal_model"]

        # Phase 1 — DETECT every tile first, touching no pixels. This used to
        # be one interleaved detect+remove loop, but the all-or-nothing veto
        # needs the whole detection set before it can decide whether this
        # frame's detections are all fixed-pattern structure. Splitting also
        # makes this path behave like the torch path instead of diverging.
        det_flags = [False] * total
        for idx, (tile, y0, x0) in enumerate(all_tiles):
            orig = tile.astype(np.float32)
            out_tiles[idx] = (orig, y0, x0)
            mask_tiles[idx] = (np.zeros(orig.shape[:2], dtype=bool), y0, x0)

            d1 = _onnx_detect(orig, det1_sess)
            det_flags[idx] = bool(d1 and _onnx_detect(orig, det2_sess))

            _tick(progress_cb, idx + 1, total)

        detected_indices = [i for i, f in enumerate(det_flags) if f]
        detected_indices = _apply_tile_veto(
            detected_indices, all_tiles, veto_mask, veto_frac
        )

        # Only interior detections count; border tiles are reflected padding
        # and false-positive constantly.
        trail_any = any(
            not _is_border_tile(
                all_tiles[i][1], all_tiles[i][2],
                all_tiles[i][1] + all_tiles[i][0].shape[0],
                all_tiles[i][2] + all_tiles[i][0].shape[1],
                H, W, border_size
            )
            for i in detected_indices
        )

        # Phase 2 — REMOVE on the surviving detections only.
        # Progress stays pinned at 100% through this phase (the tile count is
        # not known until phase 1 ends); the tick is kept so Cancel still works.
        for i in detected_indices:
            orig, y0, x0 = all_tiles[i]
            orig = orig.astype(np.float32)

            pred = _onnx_remove(orig, rem_sess)
            if clip_trail:
                final, trail_mask = _apply_clip_trail_logic(
                    pred, orig, sensitivity, return_mask=True
                )
            else:
                final = pred
                trail_mask = np.zeros(orig.shape[:2], dtype=bool)

            out_tiles[i] = (final, y0, x0)
            mask_tiles[i] = (trail_mask, y0, x0)

            _tick(progress_cb, total, total)

    # ---------------- Torch path ----------------
    else:
        tiles_only = [tile.astype(np.float32) for (tile, _, _) in all_tiles]

        detect_bs, remove_bs = _satellite_batch_sizes(models, chunk_size, compatibility_mode)

        detected_flags = [False] * total
        done_flags = [False] * total

        torch = models["torch"]
        device = models["device"]
        det1 = models["detection_model1"]
        det2 = models["detection_model2"]
        rem = models["removal_model"]

        resized = [_resize_tile_for_detect(t) for t in tiles_only]

        def _to_device(batch_np):
            x = np_to_torch(batch_np, torch=torch)
            if hasattr(device, "type") and device.type == "cuda":
                return x.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            return x.to(device=device, dtype=torch.float32)

        def _detect_pass(bs: int) -> list[bool]:
            # Reset so a smaller-batch retry starts clean.
            for k in range(total):
                out_tiles[k] = None
                mask_tiles[k] = None
                done_flags[k] = False
            flags = [False] * total

            for i in range(0, len(resized), bs):
                batch_np = np.stack(resized[i:i + bs], axis=0)
                batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)
                x = _to_device(batch_np)

                with torch.no_grad():
                    o1 = det1(x).flatten()
                keep1 = (o1 > 0.5)

                local_detected = np.zeros(len(batch_np), dtype=bool)

                if keep1.any():
                    idxs = keep1.nonzero(as_tuple=False).flatten()
                    with torch.no_grad():
                        o2 = det2(x[idxs]).flatten()
                    keep2 = torch_to_np((o2 > 0.25).detach().float()).astype(bool)
                    idxs_cpu = torch_to_np(idxs.detach().float()).astype(np.int64)
                    for local_j, passed in zip(idxs_cpu, keep2):
                        if passed:
                            local_detected[int(local_j)] = True

                for local_j in range(len(batch_np)):
                    global_j = i + local_j
                    flags[global_j] = bool(local_detected[local_j])
                    if not flags[global_j]:
                        tile, y0, x0 = all_tiles[global_j]
                        out_tiles[global_j] = (tile.astype(np.float32), y0, x0)
                        mask_tiles[global_j] = (np.zeros(tile.shape[:2], dtype=bool), y0, x0)
                        done_flags[global_j] = True

                _tick(progress_cb, sum(done_flags), total)

            return flags

        with _gpu_gate(models):
            detected_flags = _with_batch_fallback(_detect_pass, detect_bs, label="detection")

        detected_indices = [i for i, flag in enumerate(detected_flags) if flag]

        # Drop the frame's detections when EVERY one of them lands on known
        # fixed-pattern structure. All-or-nothing by design — see _apply_tile_veto.
        detected_indices = _apply_tile_veto(
            detected_indices, all_tiles, veto_mask, veto_frac
        )

        # Only count interior tile detections toward trail_any — border tiles have
        # reflected padding content that causes false positives for skip-save logic
        trail_any = any(
            not _is_border_tile(
                all_tiles[i][1], all_tiles[i][2],
                all_tiles[i][1] + all_tiles[i][0].shape[0],
                all_tiles[i][2] + all_tiles[i][0].shape[1],
                H, W, border_size
            )
            for i in detected_indices
        )

        if detected_indices:
            padded_tiles = []
            orig_shapes = []

            for idx in detected_indices:
                padded, h, w = _pad_tile_to_shape_rgb(tiles_only[idx], chunk_size, chunk_size)
                padded_tiles.append(padded)
                orig_shapes.append((h, w))

            def _remove_pass(bs: int) -> None:
                for batch_start in range(0, len(padded_tiles), bs):
                    batch_tiles = padded_tiles[batch_start:batch_start + bs]
                    batch_np = np.stack(batch_tiles, axis=0)
                    batch_np = np.transpose(batch_np, (0, 3, 1, 2)).astype(np.float32)
                    x = _to_device(batch_np)

                    with torch.no_grad(), _autocast_context(torch, device):
                        y = torch_to_np(rem(x).detach().float())

                    y = np.transpose(y, (0, 2, 3, 1))

                    for local_j, pred in enumerate(y):
                        src_idx = detected_indices[batch_start + local_j]
                        h, w = orig_shapes[batch_start + local_j]
                        pred = np.clip(pred[:h, :w, :], 0.0, 1.0).astype(np.float32)

                        orig, y0, x0 = all_tiles[src_idx]
                        orig = orig.astype(np.float32)

                        if clip_trail:
                            final, trail_mask = _apply_clip_trail_logic(
                                pred, orig, sensitivity, return_mask=True
                            )
                        else:
                            final = pred
                            trail_mask = np.zeros(orig.shape[:2], dtype=bool)

                        out_tiles[src_idx] = (final, y0, x0)
                        mask_tiles[src_idx] = (trail_mask, y0, x0)
                        done_flags[src_idx] = True

                    _tick(progress_cb, sum(done_flags), total)

            with _gpu_gate(models):
                _with_batch_fallback(_remove_pass, remove_bs, label="removal")

        # NOTE: a verbatim second copy of the removal loop used to live here
        # (recomputing detected_indices/padded_tiles and re-running NAFNet over
        # every trail tile, without the batch-size fallback or the device-loss
        # handling). It doubled removal inference on every frame that actually
        # had a trail, which is precisely the extra GPU pressure that pushed
        # Highlight mode over the DirectML TDR window. Removed.

    return _finish_satellite_tiles(
        out_tiles, mask_tiles, rgb01, H, W, border_size, trail_any
    )