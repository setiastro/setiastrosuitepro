# saspro/accel_installer.py
from __future__ import annotations
import platform
import subprocess
import sys
import os
from typing import Callable, Optional

from setiastro.saspro.runtime_torch import import_torch, add_runtime_to_sys_path, _user_runtime_dir, _venv_paths

LogCB = Callable[[str], None]

def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def _has_amd_rocm() -> tuple[bool, str]:
    """
    Return (True, gfx_arch) if an AMD ROCm-capable GPU is detected.
    Linux only. Uses runtime_torch._detect_rocm_arch().
    """
    try:
        if platform.system() != "Linux":
            return False, ""
        from setiastro.saspro.runtime_torch import _detect_rocm_arch
        arch = _detect_rocm_arch()
        return (True, arch) if arch else (False, "")
    except Exception:
        return False, ""


def _has_intel_arc() -> bool:
    """
    Return True if the machine appears to have an Intel Arc / Xe (XPU-capable) adapter.
    Windows: CIM/WMIC name sniff.
    Linux: lspci grep.
    macOS: False.
    """
    try:
        sysname = platform.system()
        if sysname == "Windows":
            ps = _run(["powershell","-NoProfile","-Command",
                       "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'"])
            out = (ps.stdout or "").lower()
            # Accept 'arc' or 'iris xe' (dg/xe discrete & some laptops with XPU)
            return ("intel" in out) and ("arc" in out or "iris xe" in out or "a770" in out or "a750" in out or "a580" in out or "a380" in out)
        if sysname == "Linux":
            r = _run(["bash","-lc","lspci -nn | grep -i 'vga\\|3d'"])
            s = (r.stdout or "").lower()
            return ("intel" in s) and ("arc" in s or "iris xe" in s or "xe" in s)
        return False
    except Exception:
        return False

def _has_nvidia() -> bool:
    """
    Return True if the machine *appears* to have an NVIDIA adapter.
    Windows: try PowerShell CIM first (wmic is deprecated), then wmic.
    Linux: use nvidia-smi.
    macOS: always False.
    """
    try:
      sysname = platform.system()
      if sysname == "Windows":
          # Try CIM (preferred)
          ps = _run(["powershell","-NoProfile","-Command",
                     "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';'"])
          out = (ps.stdout or "").lower()
          if "nvidia" in out:
              return True
          # Fallback to wmic (older systems)
          w = _run(["wmic","path","win32_VideoController","get","name"])
          return "nvidia" in (w.stdout or "").lower()
      if sysname == "Linux":
          r = _run(["nvidia-smi","-L"])
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

def ensure_torch_installed(prefer_gpu: bool, log_cb: LogCB, preferred_backend: str = "auto") -> tuple[bool, Optional[str]]:

    try:
        preferred_backend = (preferred_backend or "auto").lower()
        is_windows = platform.system() == "Windows"
        is_linux = platform.system() == "Linux"

        has_nv = _has_nvidia() and platform.system() in ("Windows", "Linux")
        has_intel = (not has_nv) and _has_intel_arc() and platform.system() in ("Windows", "Linux")
        has_amd, amd_arch = ((False, "") if not is_linux else _has_amd_rocm())

        prefer_cuda = prefer_gpu and (preferred_backend in ("auto", "cuda")) and has_nv
        prefer_xpu  = prefer_gpu and (preferred_backend in ("auto", "xpu")) and (not has_nv) and has_intel
        prefer_rocm = (
            prefer_gpu
            and is_linux
            and (preferred_backend in ("auto", "rocm"))
            and (not has_nv)
            and (not has_intel)
            and has_amd
        )

        # DirectML only matters on Windows with no NVIDIA/XPU
        prefer_dml = (
            is_windows
            and (not has_nv)
            and preferred_backend in ("directml", "auto")
            and prefer_gpu
            and (not prefer_xpu)
        )

        if preferred_backend == "cpu":
            prefer_cuda = prefer_xpu = prefer_dml = prefer_rocm = False

        # Install/import torch (ROCm/CUDA/XPU/DML/CPU handled inside runtime_torch)
        torch = import_torch(
            prefer_cuda=prefer_cuda,
            prefer_xpu=prefer_xpu,
            prefer_dml=prefer_dml,
            prefer_rocm=prefer_rocm,
            status_cb=log_cb,
        )

        from setiastro.saspro.runtime_torch import add_runtime_to_sys_path
        add_runtime_to_sys_path(status_cb=log_cb)

        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        xpu_ok  = bool(hasattr(torch, "xpu") and torch.xpu.is_available())

        # ROCm check: in PyTorch ROCm still shows up through torch.cuda, but torch.version.hip is set
        rocm_ok = False
        rocm_ver = None
        try:
            rocm_ver = getattr(getattr(torch, "version", None), "hip", None)
            rocm_ok = bool(cuda_ok and rocm_ver)
        except Exception:
            rocm_ok = False

        # HARD RULES about DirectML:
        # • If NVIDIA exists: never use DML.
        # • If XPU is active: also avoid DML.
        # • If ROCm is active: also avoid DML (Linux anyway, but harmless safety)
        if has_nv:
            _maybe_uninstall_dml = True
        else:
            _maybe_uninstall_dml = xpu_ok or rocm_ok

        if _maybe_uninstall_dml:
            try:
                from setiastro.saspro.runtime_torch import _user_runtime_dir, _venv_paths
                rt = _user_runtime_dir()
                vpy = _venv_paths(rt)["python"]
                r = subprocess.run(
                    [str(vpy), "-m", "pip", "show", "torch-directml"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                if r.returncode == 0 and r.stdout:
                    log_cb("Non-DML path selected → uninstalling torch-directml.")
                    subprocess.run(
                        [str(vpy), "-m", "pip", "uninstall", "-y", "torch-directml"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
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

        # No CUDA/XPU/ROCm ⇒ evaluate DML on Windows non-NVIDIA
        dml_enabled = False
        if is_windows and (not has_nv):
            try:
                import importlib
                importlib.invalidate_caches()
                import torch_directml  # noqa
                dml_enabled = True
                log_cb("DirectML detected (already installed).")
            except Exception:
                from setiastro.saspro.runtime_torch import _user_runtime_dir, _venv_paths
                rt = _user_runtime_dir()
                vpy = _venv_paths(rt)["python"]
                log_cb("Installing torch-directml (Windows fallback)…")
                r = subprocess.run(
                    [str(vpy), "-m", "pip", "install", "--prefer-binary", "torch-directml"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
                if r.returncode == 0:
                    try:
                        import importlib
                        importlib.invalidate_caches()
                        import torch_directml  # noqa
                        dml_enabled = True
                        log_cb("DirectML backend available.")
                    except Exception:
                        dml_enabled = False
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
                f" • To force a clean reinstall, delete: {os.path.join(str(_user_runtime_dir()), 'venv')} and click Install/Update again."
            )
        if "macOS arm64 on Python 3.13" in msg:
            msg += (
                "\n\nmacOS tip:\n"
                " • Install Python 3.12: `brew install python@3.12`\n"
                " • Ensure `/opt/homebrew/bin/python3.12` exists, then relaunch SAS Pro.\n"
            )
        return False, msg

def current_backend() -> str:
    try:
        add_runtime_to_sys_path(status_cb=lambda *_: None)
        import importlib
        import platform as _plat
        torch = importlib.import_module("torch")

        # ROCm appears via torch.cuda in PyTorch, but torch.version.hip is populated
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
    except Exception:
        return "Not installed"
