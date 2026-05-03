# saspro/accel_installer.py
from __future__ import annotations
import platform
import subprocess
import sys
import os
from typing import Callable, Optional

from setiastro.saspro.runtime_torch import (
    import_torch, add_runtime_to_sys_path,
    _user_runtime_dir, _venv_paths, _SUPPORTED_PY_MINORS,
)

LogCB = Callable[[str], None]


def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _has_amd_rocm() -> tuple[bool, str]:
    try:
        if platform.system() != "Linux":
            return False, ""
        from setiastro.saspro.runtime_torch import _detect_rocm_arch
        arch = _detect_rocm_arch()
        return (True, arch) if arch else (False, "")
    except Exception:
        return False, ""


def _has_intel_arc() -> bool:
    try:
        sysname = platform.system()
        if sysname == "Windows":
            ps = _run(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'"])
            out = (ps.stdout or "").lower()
            return ("intel" in out) and (
                "arc" in out or "iris xe" in out
                or "a770" in out or "a750" in out or "a580" in out or "a380" in out
            )
        if sysname == "Linux":
            r = _run(["bash", "-lc", "lspci -nn | grep -i 'vga\\|3d'"])
            s = (r.stdout or "").lower()
            return ("intel" in s) and ("arc" in s or "iris xe" in s or "xe" in s)
        return False
    except Exception:
        return False


def _has_nvidia() -> bool:
    try:
        sysname = platform.system()
        if sysname == "Windows":
            ps = _run(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'"])
            out = (ps.stdout or "").lower()
            if "nvidia" in out:
                return True
            w = _run(["wmic", "path", "win32_VideoController", "get", "name"])
            return "nvidia" in (w.stdout or "").lower()
        if sysname == "Linux":
            r = _run(["nvidia-smi", "-L"])
            return "GPU" in (r.stdout or "")
        return False
    except Exception:
        return False


def _nvidia_driver_ok(log_cb: LogCB) -> bool:
    try:
        r = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        drv = (r.stdout or "").strip()
        if not drv:
            log_cb("nvidia-smi not found or driver not detected.")
            return False
        log_cb(f"NVIDIA driver detected: {drv}")
        return True
    except Exception:
        log_cb("Unable to query NVIDIA driver via nvidia-smi.")
        return False


def ensure_torch_installed(
    prefer_gpu: bool,
    log_cb: LogCB,
    preferred_backend: str = "auto",
) -> tuple[bool, Optional[str]]:

    try:
        preferred_backend = (preferred_backend or "auto").lower()
        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        has_nv = _has_nvidia() and platform.system() in ("Windows", "Linux")
        has_intel = (not has_nv) and _has_intel_arc() and platform.system() in ("Windows", "Linux")
        has_amd, amd_arch = ((False, "") if not is_linux else _has_amd_rocm())

        prefer_cuda = False
        prefer_xpu = False
        prefer_rocm = False
        prefer_dml = False

        if preferred_backend == "cpu":
            pass
        elif preferred_backend == "cuda":
            prefer_cuda = prefer_gpu and has_nv
        elif preferred_backend == "xpu":
            prefer_xpu = prefer_gpu and (is_windows or is_linux) and (not has_nv) and has_intel
        elif preferred_backend == "rocm":
            prefer_rocm = prefer_gpu and is_linux and (not has_nv) and has_amd
        elif preferred_backend == "directml":
            prefer_dml = prefer_gpu and is_windows and (not has_nv)
        else:  # auto
            prefer_cuda = prefer_gpu and has_nv
            prefer_xpu = prefer_gpu and (is_windows or is_linux) and (not has_nv) and has_intel
            prefer_rocm = prefer_gpu and is_linux and (not has_nv) and (not has_intel) and has_amd
            prefer_dml = prefer_gpu and is_windows and (not has_nv) and (not prefer_xpu)

        log_cb(
            f"Accel preference='{preferred_backend}' -> "
            f"cuda={prefer_cuda}, xpu={prefer_xpu}, rocm={prefer_rocm}, dml={prefer_dml}"
        )

        torch = import_torch(
            prefer_cuda=prefer_cuda,
            prefer_xpu=prefer_xpu,
            prefer_dml=prefer_dml,
            prefer_rocm=prefer_rocm,
            status_cb=log_cb,
            allow_install=True,
            require_torchaudio=False,
        )

        add_runtime_to_sys_path(status_cb=log_cb)

        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        xpu_ok = bool(hasattr(torch, "xpu") and torch.xpu.is_available())

        rocm_ok = False
        rocm_ver = None
        try:
            rocm_ver = getattr(getattr(torch, "version", None), "hip", None)
            rocm_ok = bool(cuda_ok and rocm_ver)
        except Exception:
            rocm_ok = False

        # Uninstall DML if a real GPU backend won
        _maybe_uninstall_dml = has_nv or xpu_ok or rocm_ok
        if _maybe_uninstall_dml:
            try:
                rt = _user_runtime_dir()
                vpy = _venv_paths(rt)["python"]
                r = subprocess.run(
                    [str(vpy), "-m", "pip", "show", "torch-directml"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                if r.returncode == 0 and r.stdout:
                    log_cb("Non-DML path selected -> uninstalling torch-directml.")
                    subprocess.run(
                        [str(vpy), "-m", "pip", "uninstall", "-y", "torch-directml"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    )
            except Exception:
                pass

        if rocm_ok:
            arch_msg = f" ({amd_arch})" if amd_arch else ""
            log_cb(f"AMD ROCm available{arch_msg}; using ROCm backend (HIP {rocm_ver}).")
            return True, None

        if cuda_ok:
            log_cb("CUDA available; using NVIDIA backend.")
            return True, None

        if xpu_ok:
            try:
                name = None
                if hasattr(torch.xpu, "get_device_name"):
                    name = torch.xpu.get_device_name(0)
                log_cb(f"Intel XPU available{f' ({name})' if name else ''}.")
            except Exception:
                log_cb("Intel XPU available.")
            return True, None

        # No CUDA/XPU/ROCm — evaluate DML on Windows non-NVIDIA
        if is_windows and (not has_nv):
            try:
                import importlib
                importlib.invalidate_caches()
                import torch_directml  # noqa
                log_cb("DirectML detected (already installed).")
            except Exception:
                rt = _user_runtime_dir()
                vpy = _venv_paths(rt)["python"]
                log_cb("Installing torch-directml (Windows fallback)…")
                r = subprocess.run(
                    [str(vpy), "-m", "pip", "install", "--prefer-binary", "torch-directml"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                if r.returncode == 0:
                    try:
                        import importlib
                        importlib.invalidate_caches()
                        import torch_directml  # noqa
                        log_cb("DirectML backend available.")
                    except Exception:
                        log_cb("DirectML import failed after install; staying on CPU.")
                else:
                    log_cb("DirectML install failed; staying on CPU.")

        try:
            import importlib
            _ = importlib.import_module("torch")
        except Exception as e:
            return False, f"PyTorch import failed after install: {e}"

        return True, None

    except Exception as e:
        msg = str(e)
        if "PyTorch C-extension check failed" in msg or "Failed to load PyTorch C extensions" in msg:
            msg += (
                "\n\nHints:\n"
                " • Make sure you are not launching SAS Pro from a folder that contains a 'torch' directory.\n"
                " • If you previously ran a local PyTorch checkout, remove it from PYTHONPATH.\n"
                f" • To force a clean reinstall, delete: "
                f"{os.path.join(str(_user_runtime_dir()), 'venv')} and click Install/Update again."
            )
        return False, msg


def current_backend() -> str:
    try:
        add_runtime_to_sys_path(status_cb=lambda *_: None)
        import importlib
        import platform as _plat
        torch = importlib.import_module("torch")

        try:
            hip_ver = getattr(getattr(torch, "version", None), "hip", None)
        except Exception:
            hip_ver = None

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            if hip_ver:
                try:
                    name = torch.cuda.get_device_name(0)
                except Exception:
                    name = "AMD GPU"
                return f"ROCm ({name}, HIP {hip_ver})"
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA"
            return f"CUDA ({name})"

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                name = None
                if hasattr(torch.xpu, "get_device_name"):
                    name = torch.xpu.get_device_name(0)
            except Exception:
                name = None
            return f"Intel XPU{f' ({name})' if name else ''}"

        cuda_tag = getattr(getattr(torch, "version", None), "cuda", None)
        has_nv = _has_nvidia() and _plat.system() in ("Windows", "Linux")
        if cuda_tag and has_nv:
            return f"CPU (CUDA {cuda_tag} not available — check NVIDIA driver/CUDA runtime)"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"

        if _plat.system() == "Windows" and not has_nv:
            try:
                import torch_directml  # noqa
                return "DirectML"
            except Exception:
                pass

        return "CPU"
    except Exception as e:
        import traceback
        print(f"[current_backend] FAILED: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        return "Not installed"