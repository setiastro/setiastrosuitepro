# pro/runtime_torch.py
from __future__ import annotations
import os, sys, subprocess, platform, shutil, json
from pathlib import Path


import platform
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

            # Use system Python when frozen; otherwise use the running interpreter
            py_cmd = _find_system_python_cmd() if getattr(sys, "frozen", False) else [sys.executable]

            # Clean env (PyInstaller sometimes sets PYTHON* that confuse venv)
            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env.pop("PYTHONPATH", None)

            # ✅ IMPORTANT: split "-m" and "venv" into separate args
            subprocess.check_call(py_cmd + ["-m", "venv", str(p["venv"])], env=env)

            # bootstrap pip & tooling inside the venv
            subprocess.check_call([str(p["python"]), "-m", "ensurepip", "--upgrade"], env=env)
            subprocess.check_call([str(p["python"]), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], env=env)

        except subprocess.CalledProcessError as e:
            # If venv partially created, delete it so the next run is clean
            try:
                if p["venv"].exists():
                    shutil.rmtree(p["venv"], ignore_errors=True)
            finally:
                raise
        except Exception as e:
            if _is_access_denied(e):
                raise OSError(_access_denied_msg(rt)) from e
            raise
    return p["python"]


def _install_torch(venv_python: Path, prefer_cuda: bool, status_cb=print):
    """
    Install torch into the per-user venv.

    Linux behavior (no NVIDIA / CPU path) is now more robust:
      - Try PyPI first
      - If that fails, retry with the official CPU index
      - If that fails, upgrade pip/setuptools/wheel and retry CPU index once more
    """
    import platform
    sysname = platform.system()
    try_cuda = prefer_cuda and sysname in ("Windows", "Linux")

    def _pip(*args):
        subprocess.check_call([str(venv_python), "-m", "pip", *args])

    try:
        if try_cuda:
            # ── CUDA path (unchanged for Windows/Linux with NVIDIA) ───────────────────
            status_cb("Installing PyTorch (one-time download)…")
            _pip("install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121")
        else:
            # ── CPU path ─────────────────────────────────────────────────────────────
            status_cb("Installing PyTorch (CPU)…")
            try:
                # 1) PyPI first (works on many Linux distros)
                _pip("install", "torch")
            except subprocess.CalledProcessError as e1:
                if sysname == "Linux":
                    status_cb("PyPI CPU install failed on Linux → retrying with the official CPU index…")
                    try:
                        # 2) Official CPU index retry
                        _pip("install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
                    except subprocess.CalledProcessError as e2:
                        status_cb("CPU index install also failed → upgrading pip/wheel/setuptools and retrying once…")
                        try:
                            # 3) Upgrade tooling and try CPU index again
                            _pip("install", "--upgrade", "pip", "wheel", "setuptools")
                            _pip("install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
                        except Exception as e3:
                            # Propagate the last error
                            raise e3
                else:
                    # Non-Linux CPU failure → just bubble up (macOS path is handled elsewhere)
                    raise e1

    except subprocess.CalledProcessError as e:
        # Not a permissions error; let caller handle/report details
        # (Permissions issues are handled below)
        raise
    except Exception as e:
        # Map explicit permission denials to our friendly message
        if _is_access_denied(e):
            rt = Path(str(venv_python)).parents[1]  # …/runtime/py311
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

def _find_system_python_cmd() -> list[str]:
    """
    Return a command list that runs a system Python we can use to create a venv.
    On Windows try the py launcher; on POSIX try python3.
    """
    import shutil, platform, subprocess
    if platform.system() == "Darwin":
        for exe in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"):
            if shutil.which(exe) or os.path.exists(exe):
                try:
                    r = subprocess.run([exe, "-c", "import sys; print(sys.version)"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    if r.returncode == 0:
                        return [exe]
                except Exception:
                    pass    
    if platform.system() == "Windows":
        # Try specific versions first via py launcher
        for args in (["py","-3.11"], ["py","-3.10"], ["py","-3"], ["python3"], ["python"]):
            try:
                r = subprocess.run(args + ["-c","import sys; print(sys.version)"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if r.returncode == 0:
                    return args
            except Exception:
                pass
    else:
        for exe in ("python3.11","python3.10","python3"):
            p = shutil.which(exe)
            if p:
                try:
                    r = subprocess.run([p,"-c","import sys; print(sys.version)"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    if r.returncode == 0:
                        return [p]
                except Exception:
                    pass
        # last resort
        p = shutil.which("python")
        if p:
            return [p]
    raise RuntimeError(
        "Could not find a system Python to create the runtime environment.\n"
        "Install Python 3.10+ or set SASPRO_RUNTIME_DIR to a writable path."
    )

def add_runtime_to_sys_path(status_cb=print) -> None:
    """
    If the per-user venv exists, add its site-packages to sys.path so imports
    (like 'import torch') work on a fresh app launch without re-installing.
    No network / install is performed here.
    """
    rt = _user_runtime_dir()
    p  = _venv_paths(rt)
    vpy = p["python"]
    if not vpy.exists():
        return  # venv not created yet

    try:
        site = _site_packages(vpy)
        sp = str(site)
        if sp not in sys.path:
            sys.path.insert(0, sp)
            # Optional: tiny hint for logs
            try: status_cb(f"Added runtime site-packages to sys.path: {sp}")
            except Exception: pass

        # Also add the other common macOS/Unix site-packages in case 'site' returns platlib/purelib variant.
        # This avoids edge cases on different Python builds.
        candidates = [
            site,
            site.parent / "site-packages",
            site.parent / "dist-packages",
        ]
        for c in candidates:
            sc = str(c)
            if c.exists() and sc not in sys.path:
                sys.path.insert(0, sc)

    except Exception:
        # Non-fatal; we'll fall back to the subprocess probe in current_backend()
        return
