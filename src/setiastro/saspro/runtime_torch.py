# pro/runtime_torch.py  (hardened against shadowing / broken wheels)
from __future__ import annotations
import os
import sys
import subprocess
import platform
import shutil
import json
import time
import errno
import importlib
import re
from pathlib import Path
from contextlib import contextmanager

import platform as _plat
from pathlib import Path as _Path

def _maybe_find_torch_shm_manager(torch_mod) -> str | None:
    # Only Linux wheels include/use this helper binary.
    if _plat.system() != "Linux":
        return None
    try:
        base = _Path(getattr(torch_mod, "__file__", "")).parent
        p = base / "bin" / "torch_shm_manager"
        return str(p) if p.exists() else None
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Paths & runtime selection
# ──────────────────────────────────────────────────────────────────────────────
def _venv_pyver(venv_python: Path) -> tuple[int, int] | None:
    """Return (major, minor) for the venv interpreter, or None if unknown."""
    try:
        out = subprocess.check_output(
            [str(venv_python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True,
        ).strip()
        maj, min_ = out.split(".")
        return int(maj), int(min_)
    except Exception:
        return None

def _tag_for_pyver(maj: int, min_: int) -> str:
    return f"py{maj}{min_}"

def _runtime_base_dir() -> Path:
    """
    Base folder that may contain multiple versioned runtimes (py310, py311, py312...).
    Overridable via SASPRO_RUNTIME_DIR (which points to the parent "runtime" dir).
    """
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
    """
    Return the newest existing runtime dir that already has a venv python,
    using the venv interpreter's REAL version instead of just the folder name.
    """
    base = _runtime_base_dir()
    if not base.exists():
        return None
    candidates: list[tuple[int, int, Path]] = []
    for p in base.glob("py*"):
        vpy = p / "venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
        if not vpy.exists():
            continue
        ver = _venv_pyver(vpy)
        if ver:
            candidates.append((ver[0], ver[1], p))
    if not candidates:
        return None
    candidates.sort()  # pick the highest Python (major, minor)
    return candidates[-1][2]

def _user_runtime_dir() -> Path:
    """
    Use an existing runtime if we find one; otherwise select a directory for the
    current interpreter version (py310/py311/py312...).
    """
    existing = _discover_existing_runtime_dir()
    return existing or (_runtime_base_dir() / _current_tag())

# ──────────────────────────────────────────────────────────────────────────────
# Shadowing & sanity checks
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Shadowing & sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def _is_compiled_torch_dir(d: Path) -> bool:
    """True if 'torch' directory contains the compiled extension files."""
    return any(d.glob("_C.*.pyd")) or any(d.glob("_C.*.so")) or any(d.glob("_C.cpython*"))

def _looks_like_source_tree_torch(d: Path) -> bool:
    """
    True if this is a PyTorch repo / editable install dir (has torch/_C/__init__.py).
    These can *never* satisfy torch._C at runtime.
    """
    return (d / "_C" / "__init__.py").exists()

def _ban_shadow_torch_paths(status_cb=print) -> None:
    """
    Remove (not just demote) any sys.path entries that would cause a source-tree
    import of torch to win over the wheel. Also handles CWD ('') and editable installs.
    """
    keep: list[str] = []
    banned: list[str] = []

    for entry in list(sys.path):
        try:
            base = Path(entry) if entry else Path.cwd()
            td = base / "torch"
            if td.is_dir():
                # (a) repo/editable: has torch/_C/__init__.py  → ban outright
                if _looks_like_source_tree_torch(td):
                    banned.append(entry or "<cwd>")
                    continue
                # (b) any 'torch' dir without compiled _C.*  → ban (cannot work at runtime)
                if not _is_compiled_torch_dir(td):
                    banned.append(entry or "<cwd>")
                    continue
        except Exception:
            # if we can't inspect, keep it
            pass
        keep.append(entry)

    if banned:
        sys.path[:] = keep
        try:
            status_cb("Removed shadowing torch paths: " + ", ".join(banned))
        except Exception:
            pass

_demote_shadow_torch_paths = _ban_shadow_torch_paths

def _purge_bad_torch_from_sysmodules(status_cb=print) -> None:
    """
    If 'torch' is already imported from a shadow location, drop it so we can
    re-import from the wheel after cleaning sys.path.
    """
    try:
        import importlib
        if "torch" in sys.modules:
            mod = sys.modules["torch"]
            tf = getattr(mod, "__file__", "") or ""
            if tf and (("site-packages" not in tf) and ("dist-packages" not in tf)):
                # definitely a shadow import
                for k in list(sys.modules.keys()):
                    if k == "torch" or k.startswith("torch."):
                        sys.modules.pop(k, None)
                status_cb(f"Purged shadowed torch import: {tf}")
        # Always ensure we don't carry a stale extension handle
        sys.modules.pop("torch._C", None)
        importlib.invalidate_caches()
    except Exception:
        pass

def _torch_stack_sanity_check(status_cb=print) -> None:
    """
    Ensure torch imports sanely AND torchvision/torchaudio are importable.
    (Satellite engine requires torchvision; we install torchaudio too for safety.)
    """
    _torch_sanity_check(status_cb=status_cb)

    try:
        import torchvision  # noqa
    except Exception as e:
        raise RuntimeError(f"torchvision import failed: {e}") from e

    try:
        import torchaudio  # noqa
    except Exception as e:
        raise RuntimeError(f"torchaudio import failed: {e}") from e


def _torch_sanity_check(status_cb=print):
    try:
        import torch
        import importlib
        tf = getattr(torch, "__file__", "") or ""
        pkg_dir = Path(tf).parent if tf else None

        # must come from site/dist packages
        if ("site-packages" not in tf) and ("dist-packages" not in tf):
            raise RuntimeError(f"Shadow import: torch.__file__ = {tf}")

        # compiled extension must exist, and 'torch/_C/__init__.py' must NOT
        if not _is_compiled_torch_dir(pkg_dir):
            raise RuntimeError(f"Wheel missing torch._C in {pkg_dir}")
        if (pkg_dir / "_C" / "__init__.py").exists():
            raise RuntimeError(f"Found package folder torch/_C at {pkg_dir/'_C'}, this indicates a source tree.")

        importlib.import_module("torch._C")  # force extension load

        x = torch.ones(1); y = x + 1
        if int(y.item()) != 2:
            raise RuntimeError("Unexpected tensor arithmetic result from torch sanity op.")
    except Exception as e:
        raise RuntimeError(f"PyTorch C-extension check failed: {e}") from e

# ──────────────────────────────────────────────────────────────────────────────
# OS / permissions helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pip_run(venv_python: Path, args: list[str], status_cb=print) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return subprocess.run([str(venv_python), "-m", "pip", *args],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

def _pip_ok(venv_python: Path, args: list[str], status_cb=print) -> bool:
    r = _pip_run(venv_python, args, status_cb=status_cb)
    if r.returncode != 0:
        tail = (r.stdout or "").strip()
        try: status_cb(tail[-4000:])
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
    return r.returncode == 0

def _ensure_numpy(venv_python: Path, status_cb=print) -> None:
    """
    Torch wheels may not pull NumPy; ensure NumPy is present in the SAME venv.
    Safe to call repeatedly.
    """
    def _numpy_present() -> bool:
        code = "import importlib.util; print('OK' if importlib.util.find_spec('numpy') else 'MISS')"
        try:
            out = subprocess.check_output([str(venv_python), "-c", code], text=True).strip()
            return (out == "OK")
        except Exception:
            return False

    if _numpy_present():
        return

    # Keep tools fresh, then install a compatible NumPy (Torch 2.x is fine with NumPy 1.26–2.x)
    _pip_ok(venv_python, ["install", "--upgrade", "pip", "setuptools", "wheel"], status_cb=status_cb)

    # Prefer latest available in [1.26, 3.0)
    if not _pip_ok(venv_python, ["install", "--prefer-binary", "--no-cache-dir", "numpy>=1.26,<3"], status_cb=status_cb):
        # Final fallback to a broadly available pin
        _pip_ok(venv_python, ["install", "--prefer-binary", "--no-cache-dir", "numpy==1.26.*"], status_cb=status_cb)

    # Post-install verification
    if not _numpy_present():
        raise RuntimeError("Failed to install NumPy into the SASpro runtime venv.")


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

# ──────────────────────────────────────────────────────────────────────────────
# Venv creation & site discovery
# ──────────────────────────────────────────────────────────────────────────────

def _venv_paths(rt: Path):
    return {
        "venv": rt / "venv",
        "python": (rt / "venv" / "Scripts" / "python.exe") if platform.system() == "Windows" else (rt / "venv" / "bin" / "python"),
        "marker": rt / "torch_installed.json",
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

            # choose the system python that will back this venv
            py_cmd = _find_system_python_cmd() if getattr(sys, "frozen", False) else [sys.executable]
            # detect its version to ensure the folder tag matches
            out = subprocess.check_output(py_cmd + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
            maj, min_ = map(int, out.split("."))
            desired_tag = _tag_for_pyver(maj, min_)
            if rt.name != desired_tag:
                rt = _runtime_base_dir() / desired_tag
                p = _venv_paths(rt)
                status_cb(f"Adjusted runtime folder to match Python {maj}.{min_}: {rt}")
                p["venv"].mkdir(parents=True, exist_ok=True)

            env = os.environ.copy(); env.pop("PYTHONHOME", None); env.pop("PYTHONPATH", None)
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
    else:
        # venv already exists — verify its interpreter version matches the folder tag
        ver = _venv_pyver(p["python"])
        if ver and rt.name != _tag_for_pyver(*ver):
            status_cb(f"Runtime folder/version mismatch ({rt.name} vs Python {ver[0]}.{ver[1]}). Rebuilding.")
            shutil.rmtree(p["venv"], ignore_errors=True)
            # recreate at the correct tag
            corrected = _runtime_base_dir() / _tag_for_pyver(*ver)
            return _ensure_venv(corrected, status_cb=status_cb)

    return p["python"]

# ──────────────────────────────────────────────────────────────────────────────
# Install locking & version ladder
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _install_lock(rt: Path, timeout_s: int = 600):
    """
    Prevent concurrent partial installs into the same runtime.
    """
    lock = rt / ".install.lock"
    rt.mkdir(parents=True, exist_ok=True)
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise RuntimeError(f"Another install is running (lock: {lock})")
            time.sleep(0.5)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except Exception:
            pass

# coarse but practical ladder by Python minor
_TORCH_VERSION_LADDER: dict[tuple[int, int], list[str]] = {
    (3, 12): ["2.4.*", "2.3.*", "2.2.*"],
    (3, 11): ["2.4.*", "2.3.*", "2.2.*", "2.1.*"],
    (3, 10): ["2.4.*", "2.3.*", "2.2.*", "2.1.*", "1.13.*"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Torch installation with robust fallbacks
# ──────────────────────────────────────────────────────────────────────────────

def _check_cuda_in_venv(venv_python: Path, status_cb=print) -> tuple[bool, str | None, str | None]:
    """
    Run a small script *inside the runtime venv* to see if CUDA is usable.

    Returns (ok, cuda_tag, error_msg)
      • ok       – True if torch imports and torch.cuda.is_available() and a small
                   matmul on device='cuda' succeeds.
      • cuda_tag – value of torch.version.cuda (if available)
      • error_msg – text from any exception or stderr, for logging.
    """
    code = r"""
import json
import sys
try:
    import torch
    info = {
        "cuda_tag": getattr(getattr(torch, "version", None), "cuda", None),
        "has_cuda": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "err": None,
    }
    if info["has_cuda"]:
        # force some real GPU work
        x = torch.rand((256, 256), device="cuda", dtype=torch.float32)
        y = torch.rand((256, 256), device="cuda", dtype=torch.float32)
        _ = (x @ y).sum().item()
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({"cuda_tag": None, "has_cuda": False, "err": str(e)}))
    sys.exit(1)
"""
    r = subprocess.run(
        [str(venv_python), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    out = (r.stdout or "").strip()
    # take last line in case pip noise gets mixed in
    last = out.splitlines()[-1] if out else ""
    try:
        data = json.loads(last) if last else {}
    except Exception as e:
        msg = f"Failed to parse CUDA check output: {e}\nRaw output:\n{out}"
        try:
            status_cb(msg)
        except Exception:
            pass
        return False, None, msg

    ok = bool(data.get("has_cuda"))
    tag = data.get("cuda_tag")
    err = data.get("err")
    return ok, tag, err

def _check_xpu_in_venv(venv_python: Path, status_cb=print) -> tuple[bool, str | None]:
    code = r"""
import json
import sys
try:
    import torch
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if has_xpu:
        x = torch.rand((128, 128), device="xpu")
        y = torch.rand((128, 128), device="xpu")
        _ = (x @ y).sum().item()
    print(json.dumps({"has_xpu": bool(has_xpu)}))
except Exception as e:
    print(json.dumps({"has_xpu": False, "err": str(e)}))
    sys.exit(1)
"""
    r = subprocess.run(
        [str(venv_python), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = (r.stdout or "").strip()
    last = out.splitlines()[-1] if out else ""
    try:
        data = json.loads(last) if last else {}
    except Exception as e:
        msg = f"Failed to parse XPU check output: {e}\nRaw output:\n{out}"
        try: status_cb(msg)
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        return False, msg
    return bool(data.get("has_xpu")), data.get("err")


def _install_torch(venv_python: Path, prefer_cuda: bool, prefer_xpu: bool, status_cb=print):
    """
    Install torch into the per-user venv with best-effort backend detection:
      • macOS arm64 → PyPI (MPS)
      • Win/Linux + (prefer_cuda True) → try CUDA indices in order: cu124, cu121, cu118
      • else → PyPI (CPU), with Linux fallback to official CPU index
    Uses a version ladder when "no matching distribution" occurs.
    """
    import platform as _plat
    INTEL_XPU_INDEX = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"

    def _pip(*args, env=None) -> subprocess.CompletedProcess:
        e = (os.environ.copy() if env is None else env)
        e.pop("PYTHONPATH", None); e.pop("PYTHONHOME", None)
        return subprocess.run([str(venv_python), "-m", "pip", *args],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)

    def _pip_ok(cmd: list[str]) -> bool:
        r = _pip(*cmd)
        if r.returncode != 0:
            # surface tail of pip log for the UI
            tail = (r.stdout or "").strip()
            status_cb(tail[-4000:])
        return r.returncode == 0

    def _pyver() -> tuple[int, int]:
        code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        out = subprocess.check_output([str(venv_python), "-c", code], text=True).strip()
        major, minor = out.split(".")
        return int(major), int(minor)

    sysname = _plat.system()
    machine = _plat.machine().lower()
    py_major, py_minor = _pyver()

    if sysname == "Darwin" and ("arm64" in machine or "aarch64" in machine):
        if py_minor >= 13:
            raise RuntimeError(
                f"PyTorch wheels are not available for macOS arm64 on Python {py_major}.{py_minor}. "
                "Please install Python 3.12 (e.g. `brew install python@3.12`) so SAS Pro can create "
                "its runtime with 3.12 and install the MPS-enabled torch wheel."
            )

    ladder = _TORCH_VERSION_LADDER.get((py_major, py_minor), ["2.4.*", "2.3.*", "2.2.*"])

    status_cb(f"Runtime Python: {py_major}.{py_minor}")

    # Keep venv tools fresh
    _pip_ok(["install", "--upgrade", "pip", "setuptools", "wheel"])

    def _try_series(index_url: str | None, versions: list[str]) -> bool:
        base = ["install", "--prefer-binary", "--no-cache-dir"]
        if index_url:
            base += ["--index-url", index_url]

        # First try "latest trio" from that index
        if _pip_ok(base + ["torch", "torchvision", "torchaudio"]):
            return True

        # Walk the ladder: pin all three to the same version family
        for v in versions:
            if _pip_ok(base + [f"torch=={v}", f"torchvision=={v}", f"torchaudio=={v}"]):
                return True

        return False

    # macOS Apple Silicon → MPS wheels on PyPI
    if sysname == "Darwin" and ("arm64" in machine or "aarch64" in machine):
        status_cb("Installing PyTorch (macOS arm64, MPS)…")
        if not _try_series(None, ladder):
            raise RuntimeError("Failed to find a matching PyTorch wheel for macOS arm64.")
        return

    # Windows/Linux – CUDA first if requested, then CPU
    try_cuda = prefer_cuda and sysname in ("Windows", "Linux")
    cuda_indices = [
        ("cu129", "https://download.pytorch.org/whl/cu129"),
        ("cu128", "https://download.pytorch.org/whl/cu128"),        
        ("cu124", "https://download.pytorch.org/whl/cu124"),
        ("cu121", "https://download.pytorch.org/whl/cu121"),
        ("cu118", "https://download.pytorch.org/whl/cu118"),
    ]

    if try_cuda:
        for tag, url in cuda_indices:
            status_cb(f"Trying PyTorch CUDA wheels: {tag} …")
            if _try_series(url, ladder):
                # Verify the wheel just installed in the *runtime venv*, not the GUI env.
                ok, cuda_tag, err = _check_cuda_in_venv(venv_python, status_cb=status_cb)
                if not ok:
                    status_cb(
                        f"Installed from {tag} but CUDA is not available in the runtime venv "
                        f"(torch.version.cuda={cuda_tag!r}, err={err!r}). "
                        "Uninstalling and trying next…"
                    )
                    _pip_ok(["uninstall", "-y", "torch", "torchvision", "torchaudio"])
                    continue

                status_cb(f"Installed PyTorch CUDA ({tag}; torch.version.cuda={cuda_tag}).")
                return

            status_cb(f"No matching CUDA {tag} wheel for this Python/OS. Trying next…")

        status_cb("Falling back to CPU wheels (no matching CUDA wheel).")
    try_xpu = prefer_xpu and sysname in ("Windows", "Linux")
    if try_xpu:
        status_cb("Trying PyTorch Intel XPU wheels…")
        if _try_series(INTEL_XPU_INDEX, ladder):
            ok, err = _check_xpu_in_venv(venv_python, status_cb=status_cb)
            if ok:
                status_cb("Installed PyTorch Intel XPU (torch.xpu available).")
                return
            else:
                status_cb(f"XPU runtime test failed in venv: {err!r}. Uninstalling and falling back…")
                _pip_ok(["uninstall", "-y", "torch", "torchvision", "torchaudio"])
        else:
            status_cb("No matching Intel XPU wheel for this Python/OS.")
    # CPU path
    status_cb("Installing PyTorch (CPU)…")
    if _try_series(None, ladder):
        return
    if sysname == "Linux":
        status_cb("Retry with official CPU index…")
        if _try_series("https://download.pytorch.org/whl/cpu", ladder):
            return
    raise RuntimeError("Failed to install any compatible PyTorch wheel (CPU or CUDA).")

# ──────────────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────────────

def import_torch(prefer_cuda: bool = True, prefer_xpu: bool = False, status_cb=print):
    """
    Ensure a per-user venv exists with torch installed; return the imported module.
    Hardened against shadow imports, broken wheels, concurrent installs, and partial markers.
    """
    # Before any attempt, demote shadowing paths (CWD / random folders)
    _ban_shadow_torch_paths(status_cb=status_cb)
    _purge_bad_torch_from_sysmodules(status_cb=status_cb)

    add_runtime_to_sys_path(status_cb=lambda *_: None)

    # Fast path: if torch already importable and sane, use it
    try:
        import torch  # noqa
        _torch_stack_sanity_check(status_cb=status_cb)
        return torch
    except Exception:
        pass

    rt = _user_runtime_dir()
    vp = _ensure_venv(rt, status_cb=status_cb)
    site = _site_packages(vp)
    marker = rt / "torch_installed.json"

    try:
        _ensure_numpy(vp, status_cb=status_cb)
    except Exception:
        # Non-fatal; we'll try again if torch complains at runtime
        pass

    # If no marker, perform install under a lock
    if not marker.exists():
        try:
            with _install_lock(rt):
                # Re-check inside lock in case another process finished
                if not marker.exists():
                    _install_torch(vp, prefer_cuda=prefer_cuda, prefer_xpu=prefer_xpu, status_cb=status_cb)
        except Exception as e:
            if _is_access_denied(e):
                raise OSError(_access_denied_msg(rt)) from e
            raise

    # Ensure the venv site is first on sys.path, then demote shadowers again
    if str(site) not in sys.path:
        sys.path.insert(0, str(site))
    _demote_shadow_torch_paths(status_cb=status_cb)

    # Import + sanity. If broken, force a clean repair (all OSes).
    def _force_repair():
        try:
            status_cb("Detected broken/shadowed Torch import → attempting clean repair…")
        except Exception:
            pass

        # remove marker so future launches don't skip install
        try:
            marker.unlink()
        except Exception:
            pass

        subprocess.run([str(vp), "-m", "pip", "uninstall", "-y",
                        "torch", "torchvision", "torchaudio"], check=False)
        subprocess.run([str(vp), "-m", "pip", "cache", "purge"], check=False)
        with _install_lock(rt):
            _install_torch(vp, prefer_cuda=prefer_cuda, prefer_xpu=prefer_xpu, status_cb=status_cb)
        importlib.invalidate_caches()
        _demote_shadow_torch_paths(status_cb=status_cb)


    try:
        _ensure_numpy(vp, status_cb=status_cb)
    except Exception:
        pass

    try:
        import torch  # noqa
        _torch_stack_sanity_check(status_cb=status_cb)
        # write/update marker only when sane
        if not marker.exists():
            pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
            try:
                import torch, torchvision, torchaudio
                marker.write_text(json.dumps({
                    "installed": True,
                    "python": pyver,
                    "when": int(time.time()),
                    "torch": getattr(torch, "__version__", None),
                    "torchvision": getattr(torchvision, "__version__", None),
                    "torchaudio": getattr(torchaudio, "__version__", None),
                }), encoding="utf-8")
            except Exception:
                marker.write_text(json.dumps({"installed": True, "python": pyver, "when": int(time.time())}), encoding="utf-8")

        return torch
    except Exception:
        _force_repair()
        _purge_bad_torch_from_sysmodules(status_cb=status_cb)
        _ban_shadow_torch_paths(status_cb=status_cb)
        import torch  # retry
        _torch_stack_sanity_check(status_cb=status_cb)
        if not marker.exists():
            pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
            try:
                import torch, torchvision, torchaudio
                marker.write_text(json.dumps({
                    "installed": True,
                    "python": pyver,
                    "when": int(time.time()),
                    "torch": getattr(torch, "__version__", None),
                    "torchvision": getattr(torchvision, "__version__", None),
                    "torchaudio": getattr(torchaudio, "__version__", None),
                }), encoding="utf-8")
            except Exception:
                marker.write_text(json.dumps({"installed": True, "python": pyver, "when": int(time.time())}), encoding="utf-8")

        return torch

def _find_system_python_cmd() -> list[str]:
    import platform as _plat
    if _plat.system() == "Darwin":
        # Prefer versions that have PyTorch wheels on arm64.
        candidates = [
            "/opt/homebrew/bin/python3.12",
            "/usr/local/bin/python3.12",
            "/usr/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/usr/local/bin/python3.11",
            "/usr/bin/python3.11",
            "/opt/homebrew/bin/python3.10",
            "/usr/local/bin/python3.10",
            "/usr/bin/python3.10",
            # finally, unversioned fallbacks (may be 3.13 — last resort)
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        for exe in candidates:
            if shutil.which(exe) or os.path.exists(exe):
                try:
                    r = subprocess.run([exe, "-c", "import sys; print(sys.version)"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                    if r.returncode == 0:
                        return [exe]
                except Exception:
                    pass
    if _plat.system() == "Windows":
        for args in (["py","-3.12"], ["py","-3.11"], ["py","-3.10"], ["py","-3"], ["python3"], ["python"]):
            try:
                r = subprocess.run(args + ["-c","import sys; print(sys.version)"],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                if r.returncode == 0:
                    return args
            except Exception:
                pass
    else:
        for exe in ("python3.12","python3.11","python3.10","python3"):
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
        if p:
            return [p]
    raise RuntimeError(
        "Could not find a system Python to create the runtime environment.\n"
        "Install Python 3.10+ or set SASPRO_RUNTIME_DIR to a writable path."
    )

def add_runtime_to_sys_path(status_cb=print) -> None:
    """
    Warm up sys.path so a fresh launch can see the runtime immediately.
    """
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
            try:
                status_cb(f"Added runtime site-packages to sys.path: {sp}")
            except Exception:
                pass
        # also consider sibling dirs:
        for c in (site, site.parent / "site-packages", site.parent / "dist-packages"):
            sc = str(c)
            if c.exists() and sc not in sys.path:
                sys.path.insert(0, sc)
        # After adding, demote any accidental shadowing paths
        _demote_shadow_torch_paths(status_cb=status_cb)
    except Exception:
        return
