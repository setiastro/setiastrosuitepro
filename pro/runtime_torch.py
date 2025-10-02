# pro/runtime_torch.py
from __future__ import annotations
import os, sys, subprocess, platform, shutil, json
from pathlib import Path

import errno

def _is_access_denied(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
    # POSIX: EACCES; Windows: winerror 5
    if getattr(exc, "errno", None) == errno.EACCES:
        return True
    return getattr(exc, "winerror", None) == 5  # ERROR_ACCESS_DENIED

def _access_denied_msg(base_path: Path) -> str:
    return (
        "Access denied while preparing the SASpro runtime at:\n"
        f"  {base_path}\n\n"
        "Possible causes:\n"
        " • A corporate policy blocks writing to %LOCALAPPDATA%.\n"
        " • Security software is sandboxing the app.\n\n"
        "Fixes:\n"
        " 1) Run SASpro once as Administrator (right-click → Run as administrator), or\n"
        " 2) Set an alternate writable folder via environment variable SASPRO_RUNTIME_DIR\n"
        "    (e.g. C:\\Users\\<you>\\SASproRuntime) and relaunch."
    )

def _user_runtime_dir() -> Path:
    # Allow override
    env_override = os.getenv("SASPRO_RUNTIME_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve() / "py311"

    sysname = platform.system()
    if sysname == "Windows":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sysname == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "SASpro" / "runtime" / "py311"


def _venv_paths(rt: Path):
    return {
        "venv": rt / "venv",
        "python": (rt / "venv" / "Scripts" / "python.exe") if platform.system()=="Windows" else (rt / "venv" / "bin" / "python"),
        "site": None,  # we’ll compute once the venv exists
        "marker": rt / "torch_installed.json"
    }

def _site_packages(venv_python: Path) -> Path:
    code = "import site, sys; print([p for p in site.getsitepackages() if 'site-packages' in p][-1])"
    out = subprocess.check_output([str(venv_python), "-c", code], text=True).strip()
    return Path(out)

def _ensure_venv(rt: Path, status_cb=print) -> Path:
    p = _venv_paths(rt)
    if not p["python"].exists():
        try:
            status_cb(f"Setting up SASpro runtime venv at: {p['venv']}")
            p["venv"].mkdir(parents=True, exist_ok=True)
            subprocess.check_call([sys.executable, "-m", "venv", str(p["venv"])])
            subprocess.check_call([str(p["python"]), "-m", "ensurepip", "--upgrade"])
            subprocess.check_call([str(p["python"]), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
        except Exception as e:
            if _is_access_denied(e):
                raise OSError(_access_denied_msg(rt)) from e
            raise
    return p["python"]


import platform

def _install_torch(venv_python: Path, prefer_cuda: bool, status_cb=print):
    torch_pkgs = ["torch"]
    index_args = []
    sysname = platform.system()
    try_cuda = prefer_cuda and sysname in ("Windows","Linux")
    if try_cuda:
        index_args = ["--index-url", "https://download.pytorch.org/whl/cu121"]

    try:
        status_cb("Installing PyTorch (one-time download)…")
        subprocess.check_call([str(venv_python), "-m", "pip", "install", *torch_pkgs, *index_args])
    except subprocess.CalledProcessError as e:
        # pip returned non-zero (not permissions) → try CPU or bubble up
        if try_cuda:
            status_cb("CUDA wheels not available—falling back to CPU-only PyTorch.")
            try:
                subprocess.check_call([str(venv_python), "-m", "pip", "install",
                                       "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])
            except Exception as e2:
                if _is_access_denied(e2):
                    # Permission issue writing into the venv
                    rt = Path(str(venv_python)).parents[1]  # …/runtime/py311
                    raise OSError(_access_denied_msg(rt)) from e2
                raise
        else:
            raise
    except Exception as e:
        if _is_access_denied(e):
            rt = Path(str(venv_python)).parents[1]
            raise OSError(_access_denied_msg(rt)) from e
        raise


def import_torch(prefer_cuda: bool = True, status_cb=print):
    """
    Ensure a per-user venv exists with torch installed; return the imported torch module.
    Never modifies your bundled files. Installs to the per-user runtime dir.
    """
    # Fast path: if torch already bundled/in sys.path, use it
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        pass

    rt = _user_runtime_dir()
    vp = _ensure_venv(rt, status_cb=status_cb)
    site = _site_packages(vp)
    marker = rt / "torch_installed.json"
    
    if not marker.exists():
        try:
            _install_torch(vp, prefer_cuda=prefer_cuda, status_cb=status_cb)
            marker.write_text(json.dumps({"installed": True}), encoding="utf-8")
        except Exception as e:
            if _is_access_denied(e):
                raise OSError(_access_denied_msg(rt)) from e
            raise

    # Add venv site-packages to sys.path for this process
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))

    import importlib
    torch = importlib.import_module("torch")
    return torch
