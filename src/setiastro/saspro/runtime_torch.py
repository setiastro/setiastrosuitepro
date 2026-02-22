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


def _rt_dbg(msg: str, status_cb=print):
    try:
        status_cb(f"[RT] {msg}")
    except Exception:
        print(f"[RT] {msg}", flush=True)


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
    # HARD POLICY: runtime is always Python 3.12
    return "py312"

def _discover_existing_runtime_dir(status_cb=print) -> Path | None:
    global _RUNTIME_DISCOVERY_LOGGED

    base = _runtime_base_dir()
    if not base.exists():
        return None

    cur_dir = base / _current_tag()  # always py312 now
    cur_vpy = cur_dir / "venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    if cur_vpy.exists():
        # log once only (this is called a lot during startup)
        if not _RUNTIME_DISCOVERY_LOGGED:
            _rt_dbg(f"Found runtime: {cur_dir}", status_cb)
            _RUNTIME_DISCOVERY_LOGGED = True
        return cur_dir

    return None


def _user_runtime_dir(status_cb=print) -> Path:
    global _RUNTIME_DIR_CACHED, _RUNTIME_USERDIR_LOGGED

    if _RUNTIME_DIR_CACHED is None:
        existing = _discover_existing_runtime_dir(status_cb=status_cb)
        _RUNTIME_DIR_CACHED = existing or (_runtime_base_dir() / _current_tag())

    # log once only (avoid startup spam)
    if not _RUNTIME_USERDIR_LOGGED:
        _rt_dbg(f"_user_runtime_dir() -> {_RUNTIME_DIR_CACHED}", status_cb)
        _RUNTIME_USERDIR_LOGGED = True

    return _RUNTIME_DIR_CACHED


def best_device(torch, *, prefer_cuda=True, prefer_dml=False, prefer_xpu=False):
    if prefer_cuda and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_xpu and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    if prefer_dml and platform.system() == "Windows":
        try:
            import torch_directml
            d = torch_directml.device()
            _ = (torch.ones(1, device=d) + 1).item()
            return d
        except Exception:
            pass

    # Only pick MPS if caller wants “gpu-ish” behavior.
    if (prefer_cuda or prefer_xpu or prefer_dml) and getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")



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
    Ensure NumPy exists in the runtime venv AND is ABI-compatible with common
    torch/vision/audio wheels. In practice: enforce numpy<2.

    Safe to call repeatedly.
    """
    def _numpy_present() -> bool:
        code = "import importlib.util; print('OK' if importlib.util.find_spec('numpy') else 'MISS')"
        try:
            out = subprocess.check_output([str(venv_python), "-c", code], text=True).strip()
            return out == "OK"
        except Exception:
            return False

    def _numpy_major() -> int | None:
        code = (
            "import numpy as np\n"
            "v = np.__version__.split('+',1)[0]\n"
            "print(int(v.split('.',1)[0]))\n"
        )
        try:
            out = subprocess.check_output([str(venv_python), "-c", code], text=True).strip()
            return int(out)
        except Exception:
            return None

    # Keep tools fresh
    _pip_ok(venv_python, ["install", "--upgrade", "pip", "setuptools", "wheel"], status_cb=status_cb)

    # 1) If NumPy missing → install safe pin
    if not _numpy_present():
        status_cb("[RT] Installing NumPy (pinning to numpy<2 for torch wheel compatibility)…")
        if not _pip_ok(
            venv_python,
            ["install", "--prefer-binary", "--no-cache-dir", "numpy<2"],
            status_cb=status_cb,
        ):
            # last-ditch fallback (very widely available)
            _pip_ok(
                venv_python,
                ["install", "--prefer-binary", "--no-cache-dir", "numpy==1.26.*"],
                status_cb=status_cb,
            )

    # 2) If NumPy present but major>=2 → downgrade to numpy<2
    maj = _numpy_major()
    if maj is not None and maj >= 2:
        status_cb("[RT] NumPy 2.x detected in runtime venv; downgrading to numpy<2…")
        if not _pip_ok(
            venv_python,
            ["install", "--prefer-binary", "--no-cache-dir", "--force-reinstall", "numpy<2"],
            status_cb=status_cb,
        ):
            _pip_ok(
                venv_python,
                ["install", "--prefer-binary", "--no-cache-dir", "--force-reinstall", "numpy==1.26.*"],
                status_cb=status_cb,
            )

    # Post verification
    if not _numpy_present():
        raise RuntimeError("Failed to install NumPy into the SASpro runtime venv.")
    maj2 = _numpy_major()
    if maj2 is not None and maj2 >= 2:
        raise RuntimeError("NumPy is still 2.x in the SASpro runtime venv after pinning; torch stack may not import.")



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

            if (maj, min_) != (3, 12):
                raise RuntimeError(
                    f"GPU acceleration runtime requires Python 3.12, but found {maj}.{min_}.\n"
                    "Install Python 3.12 and relaunch SAS Pro."
                )

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
        if ver and (ver != (3, 12) or rt.name != "py312"):
            status_cb(f"Runtime venv is Python {ver[0]}.{ver[1]} but policy requires 3.12. Rebuilding py312 runtime.")
            shutil.rmtree(p["venv"], ignore_errors=True)
            return _ensure_venv(_runtime_base_dir() / "py312", status_cb=status_cb)


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
}
# module-level caches
_TORCH_CACHED = None
_RUNTIME_DIR_CACHED: Path | None = None
_RUNTIME_DISCOVERY_LOGGED = False
_RUNTIME_USERDIR_LOGGED = False

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


def _install_torch(venv_python: Path, prefer_cuda: bool, prefer_xpu: bool, prefer_dml: bool, status_cb=print):

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

    def _pyver() -> tuple[int, int]:
        code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        out = subprocess.check_output([str(venv_python), "-c", code], text=True).strip()
        major, minor = out.split(".")
        return int(major), int(minor)

    sysname = _plat.system()
    machine = _plat.machine().lower()
    py_major, py_minor = _pyver()

    def _pip_ok(cmd: list[str]) -> bool:
        r = _pip(*cmd)
        if r.returncode != 0:
            tail = (r.stdout or "").strip()
            status_cb(tail[-4000:])
        return r.returncode == 0

    if (py_major, py_minor) != (3, 12):
        raise RuntimeError(f"Runtime must be Python 3.12 (found {py_major}.{py_minor}).")

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
    
    # NEW: DirectML FIRST (Windows, non-NVIDIA)
    if sysname == "Windows" and prefer_dml:
        status_cb("Installing PyTorch with DirectML (torch-directml)…")

        # Clean slate helps resolver if something already got partially installed
        _pip_ok(["uninstall", "-y", "torch", "torchvision", "torchaudio", "torch-directml"])

        if not _pip_ok(["install", "--prefer-binary", "--no-cache-dir", "torch-directml"]):
            raise RuntimeError("Failed to install torch-directml.")

        # You still need torchvision/torchaudio for your app; let pip resolve compatible versions.
        _pip_ok(["install", "--prefer-binary", "--no-cache-dir", "torchvision", "torchaudio"])

        # Verify import + device creation
        code = "import torch, torch_directml; d=torch_directml.device(); x=torch.tensor([1]).to(d); y=torch.tensor([2]).to(d); print(int((x+y).item()))"
        r = subprocess.run([str(venv_python), "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0 or "3" not in (r.stdout or ""):
            status_cb((r.stdout or "")[-2000:])
            raise RuntimeError("torch-directml installed, but DirectML verification failed.")

        status_cb("Installed DirectML backend successfully.")
        return    
    
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

def _venv_import_probe(venv_python: Path, modname: str) -> tuple[bool, str]:
    """
    Try importing a module INSIDE the runtime venv python.
    Returns (ok, output_or_error_tail).
    """
    code = (
        "import importlib, sys\n"
        f"m=importlib.import_module('{modname}')\n"
        "print('OK', getattr(m,'__version__',None), getattr(m,'__file__',None))\n"
    )
    r = subprocess.run([str(venv_python), "-c", code],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = (r.stdout or "").strip()
    if r.returncode == 0 and out.startswith("OK"):
        return True, out
    return False, out[-4000:] if out else "no output"


def _write_torch_marker(marker: Path, status_cb=print) -> None:
    """
    Create torch_installed.json based on runtime venv imports.
    Safe to call repeatedly.
    """
    rt = marker.parent
    vp = _venv_paths(rt)["python"]

    ok_t, out_t = _venv_import_probe(vp, "torch")
    ok_v, out_v = _venv_import_probe(vp, "torchvision")
    ok_a, out_a = _venv_import_probe(vp, "torchaudio")

    payload = {
        "installed": bool(ok_t),
        "when": int(time.time()),
        "python": None,
        "torch": None,
        "torchvision": None,
        "torchaudio": None,
        "torch_file": None,
        "torchvision_file": None,
        "torchaudio_file": None,
        "probe": {
            "torch": out_t,
            "torchvision": out_v,
            "torchaudio": out_a,
        }
    }

    # get venv python version
    try:
        r = subprocess.run([str(vp), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        if r.returncode == 0:
            payload["python"] = (r.stdout or "").strip()
    except Exception:
        pass

    # parse "OK ver file" lines
    def _parse_ok(s: str):
        # format: "OK <ver> <file>"
        try:
            parts = s.split(" ", 2)
            ver = parts[1] if len(parts) > 1 else None
            f = parts[2] if len(parts) > 2 else None
            return ver, f
        except Exception:
            return None, None

    if ok_t:
        payload["torch"], payload["torch_file"] = _parse_ok(out_t)
    if ok_v:
        payload["torchvision"], payload["torchvision_file"] = _parse_ok(out_v)
    if ok_a:
        payload["torchaudio"], payload["torchaudio_file"] = _parse_ok(out_a)

    try:
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        status_cb(f"[RT] Wrote marker: {marker}")
    except Exception as e:
        status_cb(f"[RT] Failed to write marker {marker}: {e!r}")

def _venv_has_torch_stack(
    vp: Path,
    status_cb=print,
    *,
    require_torchaudio: bool = True
) -> tuple[bool, dict]:
    """
    Definitive check: can the RUNTIME VENV import torch/torchvision/(torchaudio)?
    This does NOT use the frozen app interpreter to decide installation state.
    """
    ok_t, out_t = _venv_import_probe(vp, "torch")
    ok_v, out_v = _venv_import_probe(vp, "torchvision")
    ok_a, out_a = _venv_import_probe(vp, "torchaudio")

    info = {
        "torch": (ok_t, out_t),
        "torchvision": (ok_v, out_v),
        "torchaudio": (ok_a, out_a),
    }

    ok_all = (ok_t and ok_v and ok_a) if require_torchaudio else (ok_t and ok_v)
    return ok_all, info

def _marker_says_ready(
    marker: Path,
    site: Path,
    venv_ver: tuple[int, int] | None,
    *,
    require_torchaudio: bool = True,
    max_age_days: int = 180,
) -> bool:
    """
    Advisory fast-path gate ONLY.

    Returns True if the marker looks sane enough that an in-process import attempt
    is worth trying *without* doing the expensive subprocess venv probes.

    IMPORTANT:
      - This must NOT be used to decide whether to install/uninstall anything.
      - If this returns True and the in-process import fails, we fall back to the
        definitive venv probe (_venv_has_torch_stack).
    """
    try:
        if not marker.exists():
            return False

        raw = marker.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            return False

        if not data.get("installed", False):
            return False

        # Age gate (advisory only).
        when = data.get("when")
        if isinstance(when, (int, float)):
            age_s = max(0.0, time.time() - float(when))
            if age_s > (max_age_days * 86400):
                return False

        # Marker python version should match the RUNTIME VENV python (not the app interpreter).
        py = data.get("python")
        if not (isinstance(py, str) and py.strip()):
            return False

        try:
            maj_s, min_s = py.strip().split(".", 1)
            marker_ver = (int(maj_s), int(min_s))
        except Exception:
            return False

        # If we can't determine venv version, treat marker as unreliable for fast-path.
        if venv_ver is None:
            return False

        if marker_ver != venv_ver:
            return False

        # Check that recorded files (if present) live under the computed site-packages path.
        site_s = str(site)
        tf = data.get("torch_file")
        tvf = data.get("torchvision_file")
        taf = data.get("torchaudio_file")

        def _under_site(p: str | None) -> bool:
            if not p or not isinstance(p, str):
                return False
            return site_s in p

        if not _under_site(tf):
            return False
        if not _under_site(tvf):
            return False
        if require_torchaudio and not _under_site(taf):
            return False

        return True
    except Exception:
        return False

def _cache_file(rt: Path) -> Path:
    return rt / "torch_cache.json"

def _fcache_get(rt: Path) -> dict | None:
    try:
        p = _cache_file(rt)
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8", errors="replace")
        return json.loads(raw) if raw else None
    except Exception:
        return None

def _fcache_set(rt: Path, payload: dict) -> None:
    try:
        p = _cache_file(rt)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass

def _fcache_clear(rt: Path) -> None:
    try:
        p = _cache_file(rt)
        if p.exists():
            p.unlink()
    except Exception:
        pass


# module-level cache (optional but recommended)
# module-level cache (optional but recommended)
_TORCH_CACHED = None


def import_torch(
    prefer_cuda: bool = True,
    prefer_xpu: bool = False,
    prefer_dml: bool = False,
    status_cb=print,
    *,
    require_torchaudio: bool = True,
):
    """
    Ensure a per-user venv exists with torch installed; return the imported torch module.

    ULTRA FAST PATH:
      - Use QSettings cached site-packages (no subprocess at all) and attempt in-process import.

    FAST PATH:
      - If marker looks valid, compute site-packages (1 subprocess) and try in-process imports.
      - If that works, skip expensive subprocess probes.

    SLOW PATH:
      - Probe runtime venv via subprocess (torch/torchvision/torchaudio).
      - Install only if missing, then re-probe.
      - Finally import in-process from venv site-packages.

    NEW RULES:
      - Marker/QSettings are advisory only (fast path gates).
      - If torch/torchvision(/torchaudio) exist in the runtime venv, USE THEM. Do nothing else.
      - Only if missing in the runtime venv should we install.
      - NEVER auto-uninstall user torch/torchvision/torchaudio. No automatic repair.
    """
    global _TORCH_CACHED
    if _TORCH_CACHED is not None:
        return _TORCH_CACHED
    if platform.system() == "Windows" and prefer_cuda:
        # If we are preferring CUDA, do NOT treat DirectML as a co-equal install target.
        # DirectML should only be a fallback after CUDA probe fails.
        prefer_dml = False

    def _write_cache_best_effort(rt: Path, site: Path, venv_ver: tuple[int,int] | None):
        try:
            import torch as _t  # noqa
            import torchvision as _tv  # noqa
            _ta = None
            if require_torchaudio:
                import torchaudio as _ta  # noqa

            _fcache_set(rt, {
                "tag": rt.name,
                "rt_dir": str(rt),
                "site": str(site),
                "python": (f"{venv_ver[0]}.{venv_ver[1]}" if venv_ver else ""),
                "torch": getattr(_t, "__version__", None),
                "torchvision": getattr(_tv, "__version__", None),
                "torchaudio": getattr(_ta, "__version__", None) if _ta else None,
                "when": int(time.time()),
                "require_torchaudio": bool(require_torchaudio),
            })
        except Exception:
            pass

    _rt_dbg(f"sys.frozen={getattr(sys,'frozen',False)}", status_cb)
    _rt_dbg(f"sys.executable={sys.executable}", status_cb)
    _rt_dbg(f"sys.version={sys.version}", status_cb)
    _rt_dbg(f"current_tag={_current_tag()}", status_cb)
    _rt_dbg(f"SASPRO_RUNTIME_DIR={os.getenv('SASPRO_RUNTIME_DIR')!r}", status_cb)

    # Remove obvious shadowing paths (repo folders / cwd torch trees)
    _ban_shadow_torch_paths(status_cb=status_cb)
    _purge_bad_torch_from_sysmodules(status_cb=status_cb)

    # ------------------------------------------------------------
    # Choose runtime + ensure venv exists
    # ------------------------------------------------------------
    rt = _user_runtime_dir(status_cb=status_cb)
    vp = _ensure_venv(rt, status_cb=status_cb)

    # ------------------------------------------------------------
    # ULTRA FAST PATH (runtime-aware): QSettings cache.
    # Now we can compare the cache tag against the RUNTIME tag, not sys.version_info.
    # This stays correct for "app python != runtime venv python" cases.
    # ------------------------------------------------------------

    try:
        qc = _fcache_get(rt)
        if qc:
            site_s = (qc.get("site") or "").strip()
            rt_s   = (qc.get("rt_dir") or "").strip()
            req_ta = bool(qc.get("require_torchaudio", True))
            tag    = (qc.get("tag") or "").strip()

            if (
                tag == rt.name
                and site_s and Path(site_s).exists()
                and rt_s and Path(rt_s).exists()
                and (req_ta == require_torchaudio)
            ):
                status_cb("[RT] File cache hit (runtime tag match); attempting zero-subprocess import.")

                if site_s not in sys.path:
                    sys.path.insert(0, site_s)

                _demote_shadow_torch_paths(status_cb=status_cb)
                _purge_bad_torch_from_sysmodules(status_cb=status_cb)

                import torch  # noqa
                import torchvision  # noqa
                if require_torchaudio:
                    import torchaudio  # noqa

                _TORCH_CACHED = torch
                return torch

    except Exception as e:
        status_cb(f"[RT] File-cache fast-path failed: {type(e).__name__}: {e}. Continuing…")
        _fcache_clear(rt)

    # site-packages path (subprocess but relatively cheap)
    site = _site_packages(vp)
    marker = rt / "torch_installed.json"
    venv_ver = _venv_pyver(vp)

    _rt_dbg(f"venv_ver={venv_ver}", status_cb)
    _rt_dbg(f"rt={rt}", status_cb)
    _rt_dbg(f"venv_python={vp}", status_cb)
    _rt_dbg(f"marker={marker} exists={marker.exists()}", status_cb)
    _rt_dbg(f"site={site}", status_cb)

    # Best-effort ensure numpy in venv (harmless if already there)
    try:
        _ensure_numpy(vp, status_cb=status_cb)
    except Exception:
        pass

    # ------------------------------------------------------------
    # FAST PATH: if marker looks valid, try in-process import NOW.
    # This avoids the 3 subprocess probes on every launch.
    # ------------------------------------------------------------
    try:
        if _marker_says_ready(marker, site, venv_ver, require_torchaudio=require_torchaudio):
            status_cb("[RT] Marker valid; attempting fast in-process import (skipping venv probe).")

            sp = str(site)
            if sp not in sys.path:
                sys.path.insert(0, sp)

            _demote_shadow_torch_paths(status_cb=status_cb)
            _purge_bad_torch_from_sysmodules(status_cb=status_cb)

            import torch  # noqa
            import torchvision  # noqa
            if require_torchaudio:
                import torchaudio  # noqa

            # refresh marker (best-effort)
            try:
                _write_torch_marker(marker, status_cb=status_cb)
            except Exception:
                pass

            _TORCH_CACHED = torch
            _write_cache_best_effort(rt, site, venv_ver)
            return torch

    except Exception as e:
        status_cb(f"[RT] Marker fast-path failed: {type(e).__name__}: {e}. Falling back to full probe…")
        # if marker fast path fails, your cached site-packages may also be stale
        try:
            _fcache_clear(rt)
        except Exception:
            pass

    # ------------------------------------------------------------
    # SLOW PATH: Probe the runtime venv definitively.
    # If it has torch stack, we're DONE (no installs, no repair).
    # ------------------------------------------------------------
    ok_all, info = _venv_has_torch_stack(vp, status_cb=status_cb, require_torchaudio=require_torchaudio)
    status_cb(
        "[RT] venv probe: "
        f"torch={info['torch'][0]} "
        f"torchvision={info['torchvision'][0]} "
        f"torchaudio={info['torchaudio'][0]}"
    )

    if not ok_all:
        missing = []
        if not info["torch"][0]:
            missing.append("torch")
        if not info["torchvision"][0]:
            missing.append("torchvision")
        if require_torchaudio and (not info["torchaudio"][0]):
            missing.append("torchaudio")

        status_cb(f"[RT] Missing in runtime venv: {missing}. Installing…")

        try:
            with _install_lock(rt):
                _install_torch(
                    vp,
                    prefer_cuda=prefer_cuda,
                    prefer_xpu=prefer_xpu,
                    prefer_dml=prefer_dml,
                    status_cb=status_cb,
                )
                _ensure_numpy(vp, status_cb=status_cb)
        except Exception as e:
            if _is_access_denied(e):
                raise OSError(_access_denied_msg(rt)) from e
            raise

        # Re-probe after install
        ok_all, info = _venv_has_torch_stack(vp, status_cb=status_cb, require_torchaudio=require_torchaudio)
        status_cb(
            "[RT] venv re-probe: "
            f"torch={info['torch'][0]} "
            f"torchvision={info['torchvision'][0]} "
            f"torchaudio={info['torchaudio'][0]}"
        )
        if not ok_all:
            msg = "\n".join([f"{k}: ok={ok} :: {out}" for k, (ok, out) in info.items()])
            raise RuntimeError("Torch stack still not importable in runtime venv after install:\n" + msg)

    # Always write/update marker for convenience, but never trust it for decisions.
    try:
        _write_torch_marker(marker, status_cb=status_cb)
    except Exception:
        pass

    # ------------------------------------------------------------
    # Now import torch in-process, but ONLY after putting runtime site first.
    # ------------------------------------------------------------
    sp = str(site)
    if sp not in sys.path:
        sys.path.insert(0, sp)

    _demote_shadow_torch_paths(status_cb=status_cb)
    _purge_bad_torch_from_sysmodules(status_cb=status_cb)

    try:
        import torch  # noqa

        _TORCH_CACHED = torch
        _write_cache_best_effort(rt, site, venv_ver)
        return torch

    except Exception as e:
        # prevent repeatedly hitting a bad cached site path on next launch
        try:
            _fcache_clear(rt)
        except Exception:
            pass

        msg = "\n".join([f"{k}: ok={ok} :: {out}" for k, (ok, out) in info.items()])
        raise RuntimeError(
            "Runtime venv probe says torch stack exists, but in-process import failed.\n"
            "This typically indicates a frozen-stdlib / PyInstaller packaging issue, not a bad torch install.\n\n"
            f"Original error: {type(e).__name__}: {e}\n\n"
            "Runtime venv probe:\n" + msg
        ) from e


def _find_system_python_cmd() -> list[str]:
    """
    Find a SYSTEM Python command suitable for creating the SASpro runtime venv.

    HARD POLICY:
      - Runtime venv must be created with Python 3.12.
      - Do NOT fall back to other minors (3.11/3.13/etc).
      - If 3.12 isn't available, raise with a clear message.

    Returns a list of argv tokens (e.g. ["py","-3.12"] or ["/opt/homebrew/bin/python3.12"]).
    """
    import platform as _plat

    def _is_py312(cmd: list[str]) -> bool:
        try:
            r = subprocess.run(
                cmd + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return r.returncode == 0 and (r.stdout or "").strip() == "3.12"
        except Exception:
            return False

    sysname = _plat.system()

    # macOS: prefer explicit python3.12 locations (Homebrew first)
    if sysname == "Darwin":
        candidates = [
            ["/opt/homebrew/bin/python3.12"],
            ["/usr/local/bin/python3.12"],
            ["/usr/bin/python3.12"],
        ]
        for cmd in candidates:
            exe = cmd[0]
            if os.path.exists(exe) and os.access(exe, os.X_OK) and _is_py312(cmd):
                return cmd
        # also try PATH python3.12 if present
        p = shutil.which("python3.12")
        if p and _is_py312([p]):
            return [p]

        raise RuntimeError(
            "Could not find Python 3.12 to create the SASpro runtime venv on macOS.\n"
            "Install Python 3.12 (Apple Silicon: `brew install python@3.12`) and relaunch."
        )

    # Windows: use the Python Launcher and ONLY 3.12
    if sysname == "Windows":
        for cmd in (["py", "-3.12"],):
            if _is_py312(cmd):
                return cmd

        # As a backup, try an explicit python3.12 on PATH (rare on Windows, but possible)
        p = shutil.which("python3.12")
        if p and _is_py312([p]):
            return [p]

        raise RuntimeError(
            "Could not find Python 3.12 to create the SASpro runtime venv on Windows.\n"
            "Install Python 3.12 from python.org and ensure the Python Launcher is installed,\n"
            "then confirm `py -3.12 --version` works, and relaunch SAS Pro."
        )

    # Linux / other: only accept python3.12
    p = shutil.which("python3.12")
    if p and _is_py312([p]):
        return [p]

    raise RuntimeError(
        "Could not find Python 3.12 to create the SASpro runtime venv.\n"
        "Install Python 3.12 and ensure `python3.12 --version` returns 3.12, then relaunch.\n"
        "Or set SASPRO_RUNTIME_DIR to a writable path (this does not replace the need for Python 3.12)."
    )

def add_runtime_to_sys_path(status_cb=print) -> None:
    """
    Warm up sys.path so a fresh launch can see the runtime immediately.
    """
    rt = _user_runtime_dir(status_cb=status_cb)
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

def prewarm_torch_cache(
    status_cb=print,
    *,
    require_torchaudio: bool = True,
    ensure_venv: bool = True,
    ensure_numpy: bool = False,
    validate_marker: bool = True,
) -> None:
    """
    Build and persist the QSettings cache early (startup), so the first real
    import_torch() call can be zero-subprocess.

    By default this does NOT import torch (keeps startup lighter).
    It only computes runtime rt/vpy/site and writes QSettings.
    """
    try:
        _ban_shadow_torch_paths(status_cb=status_cb)
        _purge_bad_torch_from_sysmodules(status_cb=status_cb)

        rt = _user_runtime_dir(status_cb=status_cb)
        p = _venv_paths(rt)
        vp = p["python"]

        if ensure_venv:
            vp = _ensure_venv(rt, status_cb=status_cb)

        if not vp.exists():
            return

        if ensure_numpy:
            try:
                _ensure_numpy(vp, status_cb=status_cb)
            except Exception:
                pass

        site = _site_packages(vp)
        marker = rt / "torch_installed.json"
        venv_ver = _venv_pyver(vp)

        # Optionally only cache if marker looks valid (recommended),
        # otherwise you may cache a site-packages that doesn't actually contain torch yet.
        if validate_marker:
            if not _marker_says_ready(marker, site, venv_ver, require_torchaudio=require_torchaudio):
                status_cb("[RT] prewarm: marker not valid; skipping QSettings cache write.")
                return

        # IMPORTANT: use runtime tag, not app interpreter tag, for mixed-version scenarios
        cache_tag = rt.name  # e.g. "py312"

        _fcache_set(rt, {
            "tag": rt.name,
            "rt_dir": str(rt),
            "site": str(site),
            "python": (f"{venv_ver[0]}.{venv_ver[1]}" if venv_ver else ""),
            "torch": None,
            "torchvision": None,
            "torchaudio": None,
            "when": int(time.time()),
            "require_torchaudio": bool(require_torchaudio),
        })
        status_cb("[RT] prewarm: file cache written.")
    except Exception as e:
        try:
            status_cb(f"[RT] prewarm failed: {type(e).__name__}: {e}")
        except Exception:
            pass
