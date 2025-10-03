# pro/accel_installer.py
from __future__ import annotations
import platform, subprocess, sys, os
from typing import Callable, Optional

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

def ensure_torch_installed(prefer_gpu: bool, log_cb: LogCB) -> tuple[bool, Optional[str]]:
    """
    Install torch into the per-user venv (runtime_torch) and verify import.
    Returns (ok, message). If ok==True, torch can be imported.
    """
    try:
        # Decide whether to try CUDA wheels
        prefer_cuda = prefer_gpu and _has_nvidia() and platform.system() in ("Windows","Linux")
        log_cb(f"PyTorch install preference: prefer_cuda={prefer_cuda} (OS={platform.system()})")
        torch = import_torch(prefer_cuda=prefer_cuda, status_cb=log_cb)  # <-- uses per-user venv
        # Touch CUDA/MPS to confirm GPU availability (optional)
        _ = getattr(torch, "cuda", None)
        return True, None
    except Exception as e:
        msg = str(e)
        if "PyTorch C-extension check failed" in msg or "Failed to load PyTorch C extensions" in msg:
            msg += (
                "\n\nHints:\n"
                " • Make sure you are not launching SAS Pro from a folder that contains a 'torch' directory.\n"
                " • If you previously ran a local PyTorch checkout, remove it from PYTHONPATH.\n"
                " • You can force a clean reinstall by deleting:\n"
                f"   {os.path.join(str(_user_runtime_dir()), 'venv')}\n"
                "   then clicking Install/Update again."
            )
        return False, msg

def current_backend() -> str:
    """
    Report the detected torch backend without forcing a network install.
    1) Try in-process import (after warming sys.path).
    2) If that fails but the venv exists, probe it via its python (subprocess).
    """
    try:
        # Warm sys.path so a fresh launch can see the venv immediately
        add_runtime_to_sys_path(status_cb=lambda *_: None)

        import importlib
        torch = importlib.import_module("torch")  # may still fail if venv missing
        if hasattr(torch, "cuda") and getattr(torch.cuda, "is_available", lambda: False)():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA"
            return f"CUDA ({name})"
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        return "CPU"
    except Exception:
        # Fall back to probing the venv directly (doesn't depend on current sys.path)
        try:
            rt = _user_runtime_dir()
            paths = _venv_paths(rt)
            vpy = paths["python"]
            if not vpy.exists():
                return "Not installed"

            import json, subprocess, textwrap
            code = textwrap.dedent(
                """
                import json, platform
                try:
                    import torch
                except Exception as e:
                    print(json.dumps({"ok": False, "err": str(e)})); raise SystemExit(0)
                out = {
                    "ok": True,
                    "ver": getattr(torch, "__version__", "?"),
                    "cuda": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
                    "mps":  bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                }
                if out["cuda"]:
                    try:
                        out["dev"] = torch.cuda.get_device_name(0)
                    except Exception:
                        out["dev"] = "CUDA"
                print(json.dumps(out))
                """
            ).strip()
            r = subprocess.run([str(vpy), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if r.returncode == 0:
                info = json.loads(r.stdout.strip().splitlines()[-1])
                if not info.get("ok"):
                    return "Not installed"
                if info.get("cuda"):
                    return f"CUDA ({info.get('dev','CUDA')})"
                if info.get("mps"):
                    return "Apple MPS"
                return "CPU"
        except Exception:
            pass
        return "Not installed"