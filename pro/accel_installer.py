# pro/accel_installer.py
from __future__ import annotations
import platform, subprocess, sys, os
from typing import Callable, Optional
from PyQt6.QtWidgets import QMessageBox
from pro.runtime_torch import import_torch, add_runtime_to_sys_path, _user_runtime_dir, _venv_paths

LogCB = Callable[[str], None]

def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

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


def ensure_torch_installed(prefer_gpu: bool, log_cb: LogCB) -> tuple[bool, Optional[str]]:
    try:
        # Detect platform/GPU
        is_windows = platform.system() == "Windows"
        has_nv = _has_nvidia() and platform.system() in ("Windows", "Linux")

        # Decide whether to try CUDA wheels
        prefer_cuda = prefer_gpu and has_nv
        if prefer_cuda and not _nvidia_driver_ok(log_cb):
            log_cb("CUDA requested but NVIDIA driver not detected/working; CUDA wheels may not initialize.")
        log_cb(f"PyTorch install preference: prefer_cuda={prefer_cuda} (OS={platform.system()})")

        # Install torch (tries CUDA wheels first if prefer_cuda=True; else CPU)
        torch = import_torch(prefer_cuda=prefer_cuda, status_cb=log_cb)
        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())

        # ─────────────────────────────────────────────────────────────
        # HARD RULE: If NVIDIA exists → NO DirectML, ever.
        #   • If CUDA works → done.
        #   • If CUDA doesn't work → stay CPU-only (no DML).
        #   • Also remove any existing torch-directml from this venv.
        # ─────────────────────────────────────────────────────────────
        if has_nv:
            try:
                from pro.runtime_torch import _user_runtime_dir, _venv_paths
                rt = _user_runtime_dir(); vpy = _venv_paths(rt)["python"]
                r = subprocess.run([str(vpy), "-m", "pip", "show", "torch-directml"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if r.returncode == 0 and r.stdout:
                    log_cb("NVIDIA detected → uninstalling torch-directml (hard ban).")
                    subprocess.run([str(vpy), "-m", "pip", "uninstall", "-y", "torch-directml"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except Exception:
                pass

            if cuda_ok:
                log_cb("CUDA available; DML is banned on NVIDIA systems.")
                return True, None

            log_cb("NVIDIA present but CUDA not available → staying on CPU (no DirectML).")
            return True, None

        # ─────────────────────────────────────────────────────────────
        # NO NVIDIA on Windows:
        #   • If CUDA not available (expected), we can use DirectML.
        #   • Try to import; if missing, install once; if install fails, stay CPU.
        # ─────────────────────────────────────────────────────────────
        dml_enabled = False
        if is_windows and (not cuda_ok) and (not has_nv):
            try:
                import importlib; importlib.invalidate_caches()
                import torch_directml  # already present?
                dml_enabled = True
                log_cb("DirectML detected (already installed).")
            except Exception:
                from pro.runtime_torch import _user_runtime_dir, _venv_paths
                rt = _user_runtime_dir(); vpy = _venv_paths(rt)["python"]
                log_cb("Installing torch-directml (Windows non-NVIDIA fallback)…")
                r = subprocess.run([str(vpy), "-m", "pip", "install", "--prefer-binary", "torch-directml"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if r.returncode == 0:
                    try:
                        import importlib; importlib.invalidate_caches()
                        import torch_directml  # noqa
                        dml_enabled = True
                        log_cb("DirectML backend available.")
                    except Exception:
                        dml_enabled = False
                        log_cb("DirectML import failed after install; staying on CPU.")
                else:
                    log_cb("DirectML install failed; staying on CPU.")

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
        import importlib, platform as _plat
        torch = importlib.import_module("torch")

        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            try: name = torch.cuda.get_device_name(0)
            except Exception: name = "CUDA"
            return f"CUDA ({name})"

        cuda_tag = getattr(getattr(torch, "version", None), "cuda", None)
        has_nv = _has_nvidia() and _plat.system() in ("Windows","Linux")

        if cuda_tag and has_nv:
            # built with CUDA but can’t init — driver/runtime mismatch
            return f"CPU (CUDA {cuda_tag} not available — check NVIDIA driver/CUDA runtime)"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"

        # Only show DML when there is NO NVIDIA
        if _plat.system() == "Windows" and not has_nv:
            try:
                import torch_directml  # noqa
                return "DirectML"
            except Exception:
                pass

        return "CPU"
    except Exception:
        return "Not installed"

