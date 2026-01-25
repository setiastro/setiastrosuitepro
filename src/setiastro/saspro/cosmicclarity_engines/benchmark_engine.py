# src/setiastro/saspro/cosmicclarity_engines/benchmark_engine.py
from __future__ import annotations

import os, time, json, platform
from pathlib import Path
from typing import Callable, Optional, Literal, Tuple, Dict, Any

import numpy as np
from astropy.io import fits

from numba import njit, prange
import psutil
import cpuinfo

ProgressCB = Callable[[str], None]
CancelCB = Callable[[], bool]
DLProgressCB = Callable[[int, int], None]  # (done_bytes, total_bytes)

from setiastro.saspro.cosmicclarity_engines.sharpen_engine import load_sharpen_models
from setiastro.saspro.resources import get_resources

# -----------------------------
# Paths / cache
# -----------------------------
def benchmark_cache_dir() -> Path:
    # Reuse your runtime dir (same one accel installer uses)
    from setiastro.saspro.runtime_torch import _user_runtime_dir
    rt = Path(_user_runtime_dir())
    d = rt / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    return d

def benchmark_image_path() -> Path:
    return benchmark_cache_dir() / "benchmarkimage.fit"

BENCHMARK_FITS_URL = "https://github.com/setiastro/setiastrosuitepro/releases/download/benchmarkFIT/benchmarkimage.fit"


# -----------------------------
# Download benchmark FITS
# -----------------------------
def download_benchmark_image(
    url: str,
    dst: Optional[Path] = None,
    *,
    status_cb: Optional[ProgressCB] = None,
    progress_cb: Optional[DLProgressCB] = None,
    cancel_cb: Optional[CancelCB] = None,
    timeout: int = 30,
) -> Path:
    """
    Download benchmarkimage.fit from your repo URL into runtime cache.
    Uses streaming download + atomic replace. Supports cancel.
    """
    if dst is None:
        dst = benchmark_image_path()
    dst = Path(dst)

    tmp = dst.with_suffix(dst.suffix + ".part")

    if status_cb:
        status_cb(f"Downloading benchmark image…")

    import requests  # local import keeps startup lighter

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0

        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if cancel_cb and cancel_cb():
                    try:
                        f.close()
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise RuntimeError("Download canceled.")
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)

    # atomic replace
    os.replace(str(tmp), str(dst))

    if status_cb:
        status_cb(f"Benchmark image ready: {dst}")
    return dst

def _get_stellar_model_for_benchmark(use_gpu: bool, status_cb=None):
    """
    Returns (models, backend_tag)
      - models: SharpenModels (torch or onnx)
      - backend_tag: string like 'CUDA', 'CPU', 'DirectML'
    """
    models = load_sharpen_models(use_gpu=use_gpu, status_cb=status_cb or (lambda *_: None))

    if models.is_onnx:
        return models, "DirectML"
    dev = models.device
    if getattr(dev, "type", "") == "cuda":
        return models, "CUDA"
    return models, "CPU"


def torch_benchmark_stellar(
    patches_nchw: np.ndarray,
    *,
    use_gpu: bool,
    progress_cb=None,     # (done,total)->bool
    status_cb=None,
) -> tuple[float, float, str]:
    """
    Torch benchmark using the SAME Stellar model + autocast policy as sharpen_engine.
    """
    status_cb = status_cb or (lambda *_: None)
    models, tag = _get_stellar_model_for_benchmark(use_gpu=use_gpu, status_cb=status_cb)

    if models.is_onnx:
        raise RuntimeError("torch_benchmark_stellar called but models.is_onnx=True")

    torch = models.torch
    device = models.device
    model = models.stellar

    x = torch.from_numpy(patches_nchw).to(device=device, dtype=torch.float32, non_blocking=True)

    # warmup a tiny bit to avoid first-kernel skew
    with torch.no_grad():
        _ = model(x[0:1])
        if device.type == "cuda":
            torch.cuda.synchronize()

    total_ms = 0.0
    n = int(x.shape[0])
    status_cb(f"Benchmarking Stellar model via Torch ({tag})…")

    # IMPORTANT: reuse sharpen_engine autocast policy, not unconditional AMP
    from setiastro.saspro.cosmicclarity_engines.sharpen_engine import _autocast_context

    with torch.no_grad(), _autocast_context(torch, device):
        for i in range(n):
            t0 = time.time()
            _ = model(x[i:i+1])
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_ms += (time.time() - t0) * 1000.0

            if progress_cb and (not progress_cb(i + 1, n)):
                raise RuntimeError("Canceled.")

    return (total_ms / n), total_ms, tag


def onnx_benchmark_stellar(
    patches_nchw: np.ndarray,
    *,
    use_gpu: bool,
    progress_cb=None,
    status_cb=None,
) -> tuple[float, float, str]:
    """
    ONNX benchmark using the SAME provider selection as sharpen_engine (DirectML when available).
    Uses the already-loaded session from load_sharpen_models when is_onnx=True,
    otherwise it can still run ONNX on CPU/CUDA if you want by creating a session.
    """
    status_cb = status_cb or (lambda *_: None)

    # Prefer: if sharpen_engine chose ONNX (DirectML), use that session directly
    models, tag = _get_stellar_model_for_benchmark(use_gpu=use_gpu, status_cb=status_cb)
    if models.is_onnx:
        sess = models.stellar
        provider = "DmlExecutionProvider"
    else:
        # If sharpen_engine chose torch, we can still run ONNX benchmark (Windows only),
        # using your packaged ONNX model path.
        import onnxruntime as ort
        R = get_resources()
        onnx_path = R.CC_STELLAR_SHARP_ONNX

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            provider = "CUDAExecutionProvider"
        else:
            provider = "CPUExecutionProvider"

        sess = ort.InferenceSession(onnx_path, providers=[provider])

    input_name = sess.get_inputs()[0].name
    total_ms = 0.0
    n = int(patches_nchw.shape[0])

    status_cb(f"Benchmarking Stellar model via ONNX ({provider})…")

    for i in range(n):
        patch = patches_nchw[i:i+1].astype(np.float32, copy=False)
        t0 = time.time()
        sess.run(None, {input_name: patch})
        total_ms += (time.time() - t0) * 1000.0

        if progress_cb and (not progress_cb(i + 1, n)):
            raise RuntimeError("Canceled.")

    return (total_ms / n), total_ms, provider

# -----------------------------
# Load + tile image
# -----------------------------
def _load_benchmark_fits(path: Path) -> np.ndarray:
    with fits.open(str(path), memmap=False) as hdul:
        img = hdul[0].data
    if img is None:
        raise RuntimeError("FITS contains no data.")
    img = np.asarray(img, dtype=np.float32)

    # Expect mono 2D; convert to CHW(3,H,W)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=0)
    elif img.ndim == 3:
        # If HWC convert -> CHW
        if img.shape[-1] in (3, 4) and img.shape[0] != 3:
            img = np.transpose(img[..., :3], (2, 0, 1))
        # If already CHW with 3 ok; if 1 channel expand
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
    else:
        raise RuntimeError(f"Unexpected FITS shape: {img.shape}")
    return img


def tile_chw_image(image_chw: np.ndarray, patch_size: int = 256) -> np.ndarray:
    """
    image_chw: (3,H,W) -> patches: (N,3,patch,patch)
    Only full patches (no padding) to match your old behavior.
    """
    c, h, w = image_chw.shape
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            p = image_chw[:, y:y+patch_size, x:x+patch_size]
            if p.shape[1] == patch_size and p.shape[2] == patch_size:
                patches.append(p)
    if not patches:
        raise RuntimeError("No full 256x256 patches found in benchmark image.")
    return np.stack(patches, axis=0).astype(np.float32, copy=False)


# -----------------------------
# CPU microbenchmarks (Numba)
# -----------------------------
@njit
def _mad_cpu_jit(image_array: np.ndarray, median_val: float) -> float:
    return np.median(np.abs(image_array - median_val))

def mad_cpu(image_array: np.ndarray, runs: int = 3) -> list[float]:
    """
    Return ms timings. First run includes JIT compile.
    image_array can be CHW or HW; we flatten it for fairness.
    """
    arr = np.asarray(image_array, dtype=np.float32).ravel()
    times = []
    for _ in range(runs):
        t0 = time.time()
        med = float(np.median(arr))
        _ = _mad_cpu_jit(arr, med)
        times.append((time.time() - t0) * 1000.0)
    return times

@njit(parallel=True)
def _flat_field_jit(image_array: np.ndarray, flat_frame: np.ndarray, median_flat: float) -> np.ndarray:
    out = np.empty_like(image_array)
    n = image_array.size
    for i in prange(n):
        out[i] = image_array[i] / (flat_frame[i] / median_flat)
    return out

def flat_field_correction(image_array: np.ndarray, flat_frame: np.ndarray, runs: int = 3) -> list[float]:
    arr = np.asarray(image_array, dtype=np.float32).ravel()
    flt = np.asarray(flat_frame, dtype=np.float32).ravel()
    times = []
    for _ in range(runs):
        t0 = time.time()
        med = float(np.median(flt))
        _ = _flat_field_jit(arr, flt, med)
        times.append((time.time() - t0) * 1000.0)
    return times


# -----------------------------
# System info
# -----------------------------
def get_system_info() -> dict:
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": cpuinfo.get_cpu_info().get("brand_raw", "Unknown"),
        "RAM": f"{round(psutil.virtual_memory().total / (1024 ** 3), 1)} GB",
        "Python": f"{platform.python_version()}",
    }
    # torch / onnx details (optional)
    try:
        from setiastro.saspro.runtime_torch import add_runtime_to_sys_path
        add_runtime_to_sys_path(status_cb=lambda *_: None)
        import torch
        info["torch.__version__"] = getattr(torch, "__version__", "Unknown")
        info["torch.version.cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
        info["CUDA Available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        info["MPS Available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if info["CUDA Available"]:
            try:
                info["GPU"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception:
        pass

    try:
        import onnxruntime as ort
        info["ONNX Providers"] = ort.get_available_providers()
    except Exception:
        pass

    return info


# -----------------------------
# Torch benchmark (uses your runtime torch)
# -----------------------------
def _load_sharpen_model_for_benchmark(*, use_gpu: bool, status_cb: Optional[ProgressCB] = None):
    """
    Don’t duplicate model code here.
    Reuse your existing Cosmic Clarity sharpen engine model loader if you have one.
    Fallback: load a known .pth from your packaged models.
    """
    # ✅ Prefer: reuse your existing sharpen engine loader (recommended)
    # Example (adjust to your actual function name):
    # from setiastro.saspro.cosmicclarity_engines.sharpen_engine import get_sharpen_models
    # models = get_sharpen_models(use_gpu=use_gpu, status_cb=status_cb)
    # model = models["stellar"] or similar
    # device = models["device"]
    # return model, device

    # If you *don’t* have a sharable loader yet, keep the old network here.
    # But since you said “like Dark Star / satellite / sharpen / denoise”, you probably already do.
    raise RuntimeError("Benchmark model loader not wired yet. Point this at your existing sharpen model loader.")


def torch_benchmark(
    patches_nchw: np.ndarray,
    *,
    use_gpu: bool,
    use_amp: bool,
    progress_cb: Optional[Callable[[int, int], bool]] = None,  # return False to cancel
    status_cb: Optional[ProgressCB] = None,
) -> Tuple[float, float, str]:
    """
    Returns (avg_ms_per_patch, total_ms, backend_string)
    """
    from setiastro.saspro.runtime_torch import add_runtime_to_sys_path
    add_runtime_to_sys_path(status_cb=lambda *_: None)
    import torch

    # Decide device
    backend = "CPU"
    device = torch.device("cpu")
    if use_gpu:
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            device = torch.device("cuda")
            backend = "CUDA"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            backend = "MPS"

    if status_cb:
        status_cb(f"Torch benchmark on {backend}…")

    model, _device = _load_sharpen_model_for_benchmark(use_gpu=use_gpu, status_cb=status_cb)
    # model loader may decide device; honor it
    device = _device if _device is not None else device

    x = torch.from_numpy(patches_nchw).to(device=device, dtype=torch.float32, non_blocking=True)

    total_ms = 0.0
    n = int(x.shape[0])

    # AMP only makes sense on CUDA/MPS; your app already has fp16-broken detection logic
    do_amp = bool(use_amp and device.type in ("cuda", "mps"))

    with torch.no_grad():
        if do_amp and device.type == "cuda":
            ctx = torch.cuda.amp.autocast()
        elif do_amp and device.type == "mps":
            # no torch.mps.amp.autocast officially; keep it off unless you have a known-safe method
            ctx = None
            do_amp = False
        else:
            ctx = None

        if ctx is None:
            for i in range(n):
                t0 = time.time()
                _ = model(x[i:i+1])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                total_ms += (time.time() - t0) * 1000.0
                if progress_cb and (not progress_cb(i + 1, n)):
                    raise RuntimeError("Canceled.")
        else:
            with ctx:
                for i in range(n):
                    t0 = time.time()
                    _ = model(x[i:i+1])
                    torch.cuda.synchronize()
                    total_ms += (time.time() - t0) * 1000.0
                    if progress_cb and (not progress_cb(i + 1, n)):
                        raise RuntimeError("Canceled.")

    return (total_ms / n), total_ms, backend


# -----------------------------
# ONNX benchmark (Windows only)
# -----------------------------
def onnx_benchmark(
    patches_nchw: np.ndarray,
    *,
    onnx_path: Path,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
    status_cb: Optional[ProgressCB] = None,
) -> Tuple[float, float, str]:
    import onnxruntime as ort

    if not Path(onnx_path).exists():
        raise RuntimeError(f"ONNX model not found: {onnx_path}")

    providers = ort.get_available_providers()
    if "DmlExecutionProvider" in providers:
        provider = "DmlExecutionProvider"
    elif "CUDAExecutionProvider" in providers:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    if status_cb:
        status_cb(f"ONNX benchmark provider: {provider}")

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])
    input_name = sess.get_inputs()[0].name

    total_ms = 0.0
    n = int(patches_nchw.shape[0])
    for i in range(n):
        patch = patches_nchw[i:i+1].astype(np.float32, copy=False)
        t0 = time.time()
        sess.run(None, {input_name: patch})
        total_ms += (time.time() - t0) * 1000.0
        if progress_cb and (not progress_cb(i + 1, n)):
            raise RuntimeError("Canceled.")

    return (total_ms / n), total_ms, provider


# -----------------------------
# One-stop runner
# -----------------------------
def run_benchmark(
    *,
    mode: Literal["CPU", "GPU", "Both"] = "Both",
    use_gpu: bool = True,
    use_amp: bool = True,
    benchmark_fits_path: Optional[Path] = None,
    status_cb: Optional[ProgressCB] = None,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
    onnx_model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Returns results dict safe to json.dumps.
    progress_cb signature: (done, total)->bool continue
    """
    if benchmark_fits_path is None:
        benchmark_fits_path = benchmark_image_path()
    benchmark_fits_path = Path(benchmark_fits_path)
    if not benchmark_fits_path.exists():
        raise RuntimeError("Benchmark image not downloaded yet.")

    img_chw = _load_benchmark_fits(benchmark_fits_path)
    patches = tile_chw_image(img_chw, 256)  # (N,3,256,256)

    results: Dict[str, Any] = {}
    results["System Info"] = get_system_info()

    if mode in ("CPU", "Both"):
        if status_cb: status_cb("Running CPU benchmarks…")
        cpu_mad = mad_cpu(img_chw)
        cpu_flat = flat_field_correction(img_chw, img_chw)
        results["CPU MAD (Single Core)"] = {
            "first_ms": float(cpu_mad[0]),
            "avg_ms": float(np.mean(cpu_mad[1:])) if len(cpu_mad) > 1 else float(cpu_mad[0]),
        }
        results["CPU Flat-Field (Multi-Core)"] = {
            "first_ms": float(cpu_flat[0]),
            "avg_ms": float(np.mean(cpu_flat[1:])) if len(cpu_flat) > 1 else float(cpu_flat[0]),
        }

    if mode in ("GPU", "Both"):
        if status_cb: status_cb("Running Torch benchmark…")
        avg_ms, total_ms, backend = torch_benchmark(
            patches, use_gpu=use_gpu, use_amp=use_amp,
            progress_cb=progress_cb, status_cb=status_cb
        )
        results[f"Torch Time ({backend})"] = {"avg_ms": float(avg_ms), "total_ms": float(total_ms)}

        if platform.system() == "Windows":
            if onnx_model_path is None:
                # You should point this at your packaged ONNX model path
                # e.g. from your resources/models directory
                raise RuntimeError("onnx_model_path not provided.")
            if status_cb: status_cb("Running ONNX benchmark…")
            avg_o, total_o, provider = onnx_benchmark(
                patches, onnx_path=Path(onnx_model_path),
                progress_cb=progress_cb, status_cb=status_cb
            )
            results[f"ONNX Time ({provider})"] = {"avg_ms": float(avg_o), "total_ms": float(total_o)}
        else:
            results["ONNX Time"] = "ONNX benchmark only available on Windows."

    return results
