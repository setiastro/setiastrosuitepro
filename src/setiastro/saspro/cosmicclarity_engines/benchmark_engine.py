# src/setiastro/saspro/cosmicclarity_engines/benchmark_engine.py
from __future__ import annotations

import os, time, platform
from pathlib import Path
from typing import Callable, Optional, Literal, Dict, Any

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

from typing import Sequence, Union
from urllib.parse import urlparse, parse_qs
import re

BENCHMARK_URLS = [
    "https://drive.google.com/file/d/1wgp6Ydn8JgF1j9FVnF6PgjyN-6ptJTnK/view?usp=drive_link",
    "https://drive.google.com/file/d/1QhsmuKjvksAMq45M3aHKHylZgZEd8Nh0/view?usp=drive_link",
    "https://github.com/setiastro/setiastrosuitepro/releases/download/benchmarkFIT/benchmarkimage.fit",
]

# Keep for backwards compat (some code imports BENCHMARK_FITS_URL)
BENCHMARK_FITS_URL = BENCHMARK_URLS[-1]


from urllib.parse import urljoin

def _looks_like_html_prefix(b: bytes) -> bool:
    head = (b or b"").lstrip()[:256].lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head

def _parse_gdrive_download_form(html: str) -> tuple[str, dict] | tuple[None, None]:
    """
    Parse the Google Drive virus-scan warning page.
    Extracts:
      - form action URL (often https://drive.usercontent.google.com/download)
      - all hidden input fields needed for the download
    """
    # action="..."
    m = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html)
    if not m:
        return None, None
    action = m.group(1)

    # hidden inputs: <input type="hidden" name="X" value="Y">
    inputs = {}
    for name, val in re.findall(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]*value="([^"]*)"', html):
        inputs[name] = val

    # Some pages omit value="" explicitly; handle name-only hidden inputs too (rare)
    for name in re.findall(r'<input[^>]+type="hidden"[^>]+name="([^"]+)"(?![^>]*value=)', html):
        inputs.setdefault(name, "")

    return action, inputs


def _gdrive_file_id(url: str) -> Optional[str]:
    """
    Extract Google Drive file id from:
      - https://drive.google.com/file/d/<ID>/view
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>&export=download
    """
    try:
        u = urlparse(url)
        if "drive.google.com" not in (u.netloc or ""):
            return None

        # /file/d/<id>/...
        m = re.search(r"/file/d/([^/]+)", u.path or "")
        if m:
            return m.group(1)

        # ?id=<id>
        qs = parse_qs(u.query or "")
        if "id" in qs and qs["id"]:
            return qs["id"][0]
    except Exception:
        pass
    return None


def _gdrive_direct_url(file_id: str) -> str:
    # export=download is essential; confirm token may be appended later if needed
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _gdrive_confirm_token(html: str) -> Optional[str]:
    """
    When Drive shows the "can't scan for viruses" interstitial,
    it includes a confirm token in a download link.
    """
    # Typical patterns include confirm=<TOKEN>
    m = re.search(r"confirm=([0-9A-Za-z_]+)", html)
    if m:
        return m.group(1)
    return None


def _normalize_download_url(url: str) -> tuple[str, Optional[str]]:
    """
    Returns (normalized_url, label_for_logging).
    If Google Drive view/open link, returns a direct uc?export=download&id= URL.
    """
    fid = _gdrive_file_id(url)
    if fid:
        return _gdrive_direct_url(fid), f"Google Drive ({fid})"
    return url, None

def _looks_like_html_prefix(b: bytes) -> bool:
    if not b:
        return False
    head = b.lstrip()[:64].lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head

def _is_probably_valid_fits(path: Path, *, min_bytes: int = 1_000_000) -> bool:
    try:
        if path.stat().st_size < min_bytes:
            return False
        with open(path, "rb") as f:
            first = f.read(80)
        # FITS primary header should start with SIMPLE  =
        if b"SIMPLE" not in first[:20]:
            return False
        return True
    except Exception:
        return False


# -----------------------------
# Download benchmark FITS
# -----------------------------
def download_benchmark_image(
    url: Union[str, Sequence[str], None] = None,
    dst: Optional[Path] = None,
    *,
    status_cb: Optional[ProgressCB] = None,
    progress_cb: Optional[DLProgressCB] = None,
    cancel_cb: Optional[CancelCB] = None,
    timeout: int = 30,
) -> Path:
    """
    Download benchmarkimage.fit into runtime cache.

    url:
      - None -> try BENCHMARK_URLS in order
      - str  -> try that one (but will still handle Drive confirms)
      - list/tuple -> try each in order

    Uses streaming download + atomic replace. Supports cancel.
    """
    if dst is None:
        dst = benchmark_image_path()
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".part")

    # Build candidate list
    if url is None:
        candidates = list(BENCHMARK_URLS)
    elif isinstance(url, (list, tuple)):
        candidates = list(url)
    else:
        candidates = [url]

    import requests  # local import keeps startup lighter

    last_err = None

    for idx, raw in enumerate(candidates, start=1):
        try:
            dl_url, label = _normalize_download_url(raw)
            src_label = label or raw

            if status_cb:
                status_cb(f"Downloading benchmark image… (source {idx}/{len(candidates)}: {src_label})")

            # Use a session so Drive confirm/cookies work reliably
            with requests.Session() as s:
                r = s.get(dl_url, stream=True, timeout=timeout, allow_redirects=True)

                ctype = (r.headers.get("Content-Type") or "").lower()

                # If we got HTML, it’s probably the virus-scan warning page
                if "text/html" in ctype:
                    html = r.text  # reads page into memory (small)
                    r.close()

                    action, params = _parse_gdrive_download_form(html)
                    if action and params:
                        # Submit the "Download anyway" form with the SAME session/cookies
                        r = s.get(action, params=params, stream=True, timeout=timeout, allow_redirects=True)
                        ctype = (r.headers.get("Content-Type") or "").lower()
                    else:
                        raise RuntimeError("Google Drive returned an interstitial HTML page, but download form could not be parsed.")

                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                t_start = time.time()
                t_last = t_start
                done_last = 0
                ema_bps = None

                tmp.parent.mkdir(parents=True, exist_ok=True)

                first_chunk = True
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

                        # SNIFF FIRST BYTES: if it's HTML, abort this source immediately
                        if first_chunk:
                            first_chunk = False
                            if _looks_like_html_prefix(chunk[:256]):
                                raise RuntimeError("Google Drive returned HTML (not the FITS). Link likely requires confirm/permission.")

                        f.write(chunk)
                        done += len(chunk)

                        if progress_cb:
                            progress_cb(done, total)

                        now = time.time()
                        dt = now - t_last
                        if dt >= 0.5:
                            inst_bps = (done - done_last) / max(dt, 1e-9)
                            ema_bps = inst_bps if ema_bps is None else (0.75 * ema_bps + 0.25 * inst_bps)

                            eta = None
                            if total > 0 and ema_bps and ema_bps > 1:
                                eta = (total - done) / ema_bps

                            if status_cb:
                                pct = (done * 100.0 / total) if total > 0 else None
                                if pct is None:
                                    status_cb(f"Downloading… {_fmt_bytes(done)} at {_fmt_bytes(ema_bps)}/s • ETA {_fmt_eta(None)}")
                                else:
                                    status_cb(
                                        f"Downloading… {pct:5.1f}% • {_fmt_bytes(done)}/{_fmt_bytes(total)} "
                                        f"at {_fmt_bytes(ema_bps)}/s • ETA {_fmt_eta(eta)}"
                                    )

                            t_last = now
                            done_last = done

            # atomic replace only after a full success
            os.replace(str(tmp), str(dst))

            # VALIDATE: size + FITS header sanity. If invalid, treat as failure and try next URL.
            if not _is_probably_valid_fits(dst, min_bytes=10_000_000):  # 10MB floor; tune as you like
                raise RuntimeError("Downloaded file is not a valid FITS (too small or missing SIMPLE).")

            if status_cb:
                status_cb(f"Benchmark image ready: {dst}")
            return dst


        except Exception as e:
            last_err = e
            # clean partial
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if status_cb:
                status_cb(f"Source {idx} failed: {e}")

    raise RuntimeError(f"All benchmark download sources failed. Last error: {last_err}")

def _fmt_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(max(0.0, n))
    for u in units:
        if n < 1024.0 or u == units[-1]:
            return f"{n:.1f} {u}" if u != "B" else f"{n:.0f} {u}"
        n /= 1024.0
    return f"{n:.1f} TB"

def _fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0 or not np.isfinite(seconds):
        return "—"
    s = int(seconds + 0.5)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h {m:02d}m"
    if m:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


def _get_stellar_model_for_benchmark(use_gpu: bool, status_cb=None):
    """
    Returns (models, backend_tag)
      - models: SharpenModels (torch or onnx)
      - backend_tag: 'CUDA', 'CPU', 'DirectML', 'MPS', etc.
    """
    models = load_sharpen_models(use_gpu=use_gpu, status_cb=status_cb or (lambda *_: None))

    if models.is_onnx:
        return models, "DirectML"

    dev = models.device
    dev_type = getattr(dev, "type", "")
    if dev_type == "cuda":
        return models, "CUDA"
    if dev_type == "mps":
        return models, "MPS"

    # torch-directml devices don’t have .type == 'dml' typically; handle by string
    if "dml" in str(dev).lower() or "directml" in str(dev).lower():
        return models, "DirectML"

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
    ONNX benchmark:
      - If sharpen_engine selected ONNX (DirectML), reuse that exact session.
      - Otherwise, on Windows, prefer DirectML provider if available (when use_gpu=True),
        then CUDA EP if present, else CPU.
    """
    status_cb = status_cb or (lambda *_: None)

    # Reuse sharpen_engine session if it already chose ONNX (typically DirectML on Windows)
    models, tag = _get_stellar_model_for_benchmark(use_gpu=use_gpu, status_cb=status_cb)
    if models.is_onnx:
        sess = models.stellar
        # Use the provider that the session actually has (best-effort label)
        try:
            provs = sess.get_providers()
            provider = provs[0] if provs else "ONNX"
        except Exception:
            provider = "DmlExecutionProvider"
    else:
        import onnxruntime as ort
        from setiastro.saspro.model_manager import require_model
        onnx_path = require_model("deep_sharp_stellar_cnn_AI3_5s.onnx")

        providers_avail = ort.get_available_providers()

        # Prefer DirectML if possible (Windows) when GPU requested
        providers = []
        if use_gpu and ("DmlExecutionProvider" in providers_avail):
            providers.append("DmlExecutionProvider")
        # If no DML (or user disabled GPU), try CUDA EP if available
        if use_gpu and ("CUDAExecutionProvider" in providers_avail):
            providers.append("CUDAExecutionProvider")
        # Always end with CPU
        providers.append("CPUExecutionProvider")

        # Build session
        sess = ort.InferenceSession(str(onnx_path), providers=providers)

        # Label by what actually got picked
        try:
            provs = sess.get_providers()
            provider = provs[0] if provs else providers[0]
        except Exception:
            provider = providers[0]

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

def _fmt_ms(first_ms: float, avg_ms: float) -> str:
    return f"First: {first_ms:.2f} ms | Avg: {avg_ms:.2f} ms"

def _fmt_gpu(avg_ms: float, total_ms: float) -> str:
    return f"Avg: {avg_ms:.2f} ms | Total: {total_ms:.2f} ms"

def _legacy_results_schema(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal structured results -> legacy/human schema expected by submitter.
    """
    out: Dict[str, Any] = {}

    # --- CPU ---
    cpu_mad = results.get("CPU MAD (Single Core)")
    if isinstance(cpu_mad, dict):
        out["CPU MAD (Single Core)"] = _fmt_ms(cpu_mad.get("first_ms", 0.0), cpu_mad.get("avg_ms", 0.0))
    elif cpu_mad is not None:
        out["CPU MAD (Single Core)"] = str(cpu_mad)

    cpu_flat = results.get("CPU Flat-Field (Multi-Core)")
    if isinstance(cpu_flat, dict):
        out["CPU Flat-Field (Multi-Core)"] = _fmt_ms(cpu_flat.get("first_ms", 0.0), cpu_flat.get("avg_ms", 0.0))
    elif cpu_flat is not None:
        out["CPU Flat-Field (Multi-Core)"] = str(cpu_flat)

    # --- GPU / Torch ---
    # Your internal key looks like "Stellar Torch (CUDA)" or "(CPU)" etc.
    torch_key = next((k for k in results.keys() if k.startswith("Stellar Torch (")), None)
    if torch_key and isinstance(results.get(torch_key), dict):
        backend = torch_key[len("Stellar Torch ("):-1]  # extract tag inside (...)
        t = results[torch_key]
        out[f"GPU Time ({backend})"] = _fmt_gpu(t.get("avg_ms", 0.0), t.get("total_ms", 0.0))

    # --- ONNX ---
    onnx_key = next((k for k in results.keys() if k.startswith("Stellar ONNX")), None)
    if onnx_key is None:
        # you used this on non-windows
        if "Stellar ONNX" in results:
            out["ONNX Time"] = str(results["Stellar ONNX"])
        else:
            out["ONNX Time"] = "ONNX benchmark not run."
    else:
        v = results.get(onnx_key)
        if isinstance(v, dict):
            # keep it under the legacy key name
            out["ONNX Time"] = _fmt_gpu(v.get("avg_ms", 0.0), v.get("total_ms", 0.0))
        else:
            out["ONNX Time"] = str(v)

    # --- System Info ---
    # If you want to match your example more closely, drop Python/torch versions.
    si = results.get("System Info", {})
    if isinstance(si, dict):
        keep = {
            "OS", "CPU", "RAM",
            "CUDA Available", "MPS Available", "ONNX Providers", "GPU"
        }
        out["System Info"] = {k: si[k] for k in keep if k in si}
    else:
        out["System Info"] = si

    return out

def _picked_backend(use_gpu: bool, status_cb=None) -> str:
    models, tag = _get_stellar_model_for_benchmark(use_gpu=use_gpu, status_cb=status_cb)
    return "ONNX" if models.is_onnx else "TORCH"

# -----------------------------
# One-stop runner
# -----------------------------
def run_benchmark(
    *,
    mode: Literal["CPU", "GPU", "Both"] = "Both",
    use_gpu: bool = True,
    benchmark_fits_path: Optional[Path] = None,
    status_cb: Optional[ProgressCB] = None,
    progress_cb: Optional[Callable[[int, int], bool]] = None,
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
        if status_cb: status_cb("Running Stellar model benchmark…")

        picked = _picked_backend(use_gpu=use_gpu, status_cb=status_cb)

        if picked == "TORCH":
            avg_ms, total_ms, backend = torch_benchmark_stellar(
                patches,
                use_gpu=use_gpu,
                progress_cb=progress_cb,
                status_cb=status_cb,
            )
            results[f"Stellar Torch ({backend})"] = {"avg_ms": float(avg_ms), "total_ms": float(total_ms)}

            # Optional: also run ONNX on Windows for comparison
            if platform.system() == "Windows":
                avg_o, total_o, provider = onnx_benchmark_stellar(
                    patches,
                    use_gpu=use_gpu,
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                results[f"Stellar ONNX ({provider})"] = {"avg_ms": float(avg_o), "total_ms": float(total_o)}

        else:
            # Picked ONNX (DirectML). Run ONNX benchmark as the “GPU benchmark”.
            if platform.system() == "Windows":
                avg_o, total_o, provider = onnx_benchmark_stellar(
                    patches,
                    use_gpu=use_gpu,
                    progress_cb=progress_cb,
                    status_cb=status_cb,
                )
                results[f"Stellar ONNX ({provider})"] = {"avg_ms": float(avg_o), "total_ms": float(total_o)}
            else:
                results["Stellar ONNX"] = "ONNX benchmark only available on Windows."

    results = _legacy_results_schema(results)
    return results
