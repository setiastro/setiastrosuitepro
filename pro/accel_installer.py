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
    try:
        prefer_cuda = prefer_gpu and _has_nvidia() and platform.system() in ("Windows","Linux")
        log_cb(f"PyTorch install preference: prefer_cuda={prefer_cuda} (OS={platform.system()})")

        # Try normal torch first (CUDA on NV, CPU otherwise — your existing logic):
        torch = import_torch(prefer_cuda=prefer_cuda, status_cb=log_cb)

        # If we're on Windows, no CUDA, try to enable DirectML
        dml_enabled = False
        if platform.system() == "Windows":
            try:
                import torch_directml  # already present?
                dml_enabled = True
            except Exception:
                # attempt install of torch-directml into the same venv
                from pro.runtime_torch import _user_runtime_dir, _venv_paths
                rt = _user_runtime_dir()
                vpy = _venv_paths(rt)["python"]
                log_cb("Installing torch-directml (Windows non-NVIDIA path)…")
                r = subprocess.run([str(vpy), "-m", "pip", "install", "--prefer-binary", "torch-directml"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                if r.returncode != 0:
                    log_cb((r.stdout or "")[-4000:])
                else:
                    import importlib; importlib.invalidate_caches()
                    try:
                        import torch_directml  # noqa
                        dml_enabled = True
                    except Exception:
                        dml_enabled = False

        # Touch CUDA/MPS/DML to confirm availability
        _ = getattr(torch, "cuda", None)
        if dml_enabled:
            log_cb("DirectML backend available.")
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
        torch = importlib.import_module("torch")
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try: name = torch.cuda.get_device_name(0)
            except Exception: name = "CUDA"
            return f"CUDA ({name})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        try:
            import torch_directml  # noqa
            # quick probe
            dev = torch_directml.device()
            _ = (torch.ones(1, device=dev) + 1).item()
            return "DirectML"
        except Exception:
            pass
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