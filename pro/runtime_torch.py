# pro/runtime_torch.py  (hardened against shadowing / broken wheels)
from __future__ import annotations
import os, sys, subprocess, platform, shutil, json
from pathlib import Path
import errno
import importlib

import re

def _runtime_base_dir() -> Path:
    """Base folder that may contain multiple versioned runtimes (py39, py310, py311...)."""
    env_override = os.getenv("SASPRO_RUNTIME_DIR")
    if env_override:
        base = Path(env_override).expanduser().resolve()
    else:
        sysname = platform.system()
        if sysname == "Windows":
            base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        elif sysname == "Darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        base = base / "SASpro" / "runtime"
    return base

def _current_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}"

def _discover_existing_runtime_dir() -> Path | None:
    """Return the newest existing runtime dir that already has a venv python."""
    base = _runtime_base_dir()
    if not base.exists():
        return None
    candidates = []
    for p in base.glob("py*"):
        vpy = (p / "venv" / ("Scripts/python.exe" if platform.system()=="Windows" else "bin/python"))
        if vpy.exists():
            # parse pyMAJORMINOR
            m = re.match(r"^py(\d)(\d+)$", p.name)
            ver = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            candidates.append((ver, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]  # newest

def _user_runtime_dir() -> Path:
    """
    Use an existing runtime if we find one; otherwise select a directory for the
    current interpreter version (py39/py310/py311...).
    """
    existing = _discover_existing_runtime_dir()
    return existing or (_runtime_base_dir() / _current_tag())

def _demote_shadow_torch_paths(status_cb=print) -> None:
    """
    If any entry on sys.path contains a plain 'torch' package directory that does
    NOT include the compiled extension (_C.*.pyd / .so), move that entry to the
    end of sys.path so the real wheel in site-packages wins.

    Also handles the common case where the current working directory ('')
    contains a 'torch/' folder (zip of the repo, etc.).
    """
    def has_compiled_ext(torch_dir: Path) -> bool:
        return any(torch_dir.glob("_C.*.pyd")) or any(torch_dir.glob("_C.*.so")) or any(torch_dir.glob("_C.cpython*"))

    moved = []
    new_path = []
    for p in list(sys.path):
        try:
            tp = (Path(p) if p else Path.cwd()) / "torch"
            if tp.is_dir() and not has_compiled_ext(tp):
                # Demote this entry to the end
                moved.append(p or "<cwd>")
                continue
        except Exception:
            pass
        new_path.append(p)

    if moved:
        # Re-append moved items at the end in the same order
        new_path.extend([m if m != "<cwd>" else "" for m in moved])
        sys.path[:] = new_path
        try:
            status_cb(f"Demoted shadowing paths for torch: {', '.join(moved)}")
        except Exception:
            pass


def _torch_sanity_check(status_cb=print):
    """
    Ensure we're importing the wheel (site-packages/dist-packages) and that the
    C-extensions load. Raise RuntimeError with a helpful message if wrong.
    """
    try:
        import torch  # noqa
        tf = getattr(torch, "__file__", "") or ""
        pkg_dir = Path(tf).parent if tf else None

        # 1) Must come from site/dist-packages
        if ("site-packages" not in tf) and ("dist-packages" not in tf):
            raise RuntimeError(
                "Torch was imported from a source directory, not from site-packages:\n"
                f"  torch.__file__ = {tf}\n"
                "A folder named 'torch' is shadowing the real package.\n"
                "Rename/remove that folder, or launch SAS Pro from a different directory."
            )

        # 2) Must have compiled extension present
        has_ext = any(pkg_dir.glob("_C.*.pyd")) or any(pkg_dir.glob("_C.*.so")) or any(pkg_dir.glob("_C.cpython*"))
        if not has_ext:
            raise RuntimeError(
                "Installed torch wheel appears to be missing the compiled extension ('_C').\n"
                f"  package dir = {pkg_dir}\n"
                "This can happen if a source tree shadows the wheel or if the wheel install is corrupt."
            )

        # 3) Force import of torch._C to trip load-time errors early (DLL load, etc.)
        importlib.import_module("torch._C")

    except Exception as e:
        raise RuntimeError(f"PyTorch C-extension check failed: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# Existing functions (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def _is_access_denied(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
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
        "site": None,
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
            py_cmd = _find_system_python_cmd() if getattr(sys, "frozen", False) else [sys.executable]
            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env.pop("PYTHONPATH", None)
            subprocess.check_call(py_cmd + ["-m", "venv", str(p["venv"])], env=env)
            subprocess.check_call([str(p["python"]), "-m", "ensurepip", "--upgrade"], env=env)
            subprocess.check_call([str(p["python"]), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], env=env)
        except subprocess.CalledProcessError:
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
    import platform as _plat
    sysname = _plat.system()
    try_cuda = prefer_cuda and sysname in ("Windows", "Linux")

    def _pip(*args): subprocess.check_call([str(venv_python), "-m", "pip", *args])

    try:
        if try_cuda:
            status_cb("Installing PyTorch (one-time download)…")
            _pip("install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121")
        else:
            status_cb("Installing PyTorch (CPU)…")
            try:
                _pip("install", "torch")
            except subprocess.CalledProcessError as e1:
                if sysname == "Linux":
                    status_cb("PyPI CPU install failed on Linux → retry with official CPU index…")
                    try:
                        _pip("install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
                    except subprocess.CalledProcessError:
                        status_cb("CPU index install also failed → upgrading pip/setuptools/wheel and retrying…")
                        _pip("install", "--upgrade", "pip", "wheel", "setuptools")
                        _pip("install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
                else:
                    raise e1
    except subprocess.CalledProcessError:
        raise
    except Exception as e:
        if _is_access_denied(e):
            rt = Path(str(venv_python)).parents[1]
            raise OSError(_access_denied_msg(rt)) from e
        raise

def import_torch(prefer_cuda: bool = True, status_cb=print):
    """
    Ensure a per-user venv exists with torch installed; return the imported module.
    """
    # Before any attempt, demote shadowing paths (CWD / random folders)
    _demote_shadow_torch_paths(status_cb=status_cb)

    # Fast path: if torch already importable and sane, use it
    try:
        import torch  # noqa
        _torch_sanity_check(status_cb=status_cb)
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

    # Put the venv first and re-demote any shadowers after adding it
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))
    _demote_shadow_torch_paths(status_cb=status_cb)

    # Try import + sanity; if it fails, do a one-time repair on macOS/Windows
    try:
        import torch  # noqa
        _torch_sanity_check(status_cb=status_cb)
        return torch
    except Exception as first_err:
        if platform.system() in ("Darwin", "Windows"):
            try:
                status_cb("Detected broken/shadowed Torch import → attempting repair…")
                subprocess.check_call([str(vp), "-m", "pip", "uninstall", "-y", "torch"])
                subprocess.check_call([str(vp), "-m", "pip", "cache", "purge"])
                subprocess.check_call([str(vp), "-m", "pip", "install", "--no-cache-dir", "torch"])
                importlib.invalidate_caches()
                _demote_shadow_torch_paths(status_cb=status_cb)
                import torch  # noqa
                _torch_sanity_check(status_cb=status_cb)
                return torch
            except Exception:
                # Surface the original (more informative) error
                raise first_err
        raise

def _find_system_python_cmd() -> list[str]:
    import shutil, platform as _plat
    if _plat.system() == "Darwin":
        for exe in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3"):
            if shutil.which(exe) or os.path.exists(exe):
                try:
                    r = subprocess.run([exe, "-c", "import sys; print(sys.version)"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    if r.returncode == 0:
                        return [exe]
                except Exception:
                    pass
    if _plat.system() == "Windows":
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
        p = shutil.which("python")
        if p: return [p]
    raise RuntimeError(
        "Could not find a system Python to create the runtime environment.\n"
        "Install Python 3.10+ or set SASPRO_RUNTIME_DIR to a writable path."
    )

def add_runtime_to_sys_path(status_cb=print) -> None:
    rt = _user_runtime_dir()
    p  = _venv_paths(rt)
    vpy = p["python"]
    if not vpy.exists():
        return
    try:
        site = _site_packages(vpy)
        sp = str(site)
        if sp not in sys.path:
            sys.path.insert(0, sp)
            try: status_cb(f"Added runtime site-packages to sys.path: {sp}")
            except Exception: pass
        # also consider sibling dirs:
        for c in (site, site.parent / "site-packages", site.parent / "dist-packages"):
            sc = str(c)
            if c.exists() and sc not in sys.path:
                sys.path.insert(0, sc)
        # After adding, demote any accidental shadowing paths
        _demote_shadow_torch_paths(status_cb=status_cb)
    except Exception:
        return
