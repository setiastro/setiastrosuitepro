# pro/accel_installer.py
from __future__ import annotations
import platform, subprocess, sys
from typing import Callable, Optional

from pro.runtime_torch import import_torch

LogCB = Callable[[str], None]

def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def _has_nvidia() -> bool:
    try:
        sysname = platform.system()
        if sysname == "Windows":
            out = _run(["wmic","path","win32_VideoController","get","name"]).stdout.lower()
            return "nvidia" in out
        if sysname == "Linux":
            out = _run(["nvidia-smi","-L"]).stdout
            return "GPU" in out
        return False
    except Exception:
        return False

def ensure_torch_installed(prefer_gpu: bool, log_cb: LogCB) -> tuple[bool, Optional[str]]:
    """
    Install torch into the per-user venv (runtime_torch) and verify import.
    Returns (ok, message). If ok==True, torch can be imported.
    """
    try:
        # Decide whether to try CUDA wheels
        prefer_cuda = prefer_gpu and _has_nvidia() and platform.system() in ("Windows","Linux")
        torch = import_torch(prefer_cuda=prefer_cuda, status_cb=log_cb)  # <-- uses per-user venv
        # Touch CUDA/MPS to confirm GPU availability (optional)
        _ = getattr(torch, "cuda", None)
        return True, None
    except Exception as e:
        return False, str(e)

def current_backend() -> str:
    try:
        torch = import_torch(prefer_cuda=False, status_cb=lambda *_: None)  # import without forcing CUDA download
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA"
            return f"CUDA ({name})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        return "CPU"
    except Exception:
        return "Not installed"
