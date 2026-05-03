# pro/runtime_torch.py  (index-url strategy; Python 3.12/3.13/3.14 compatible)
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

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_text_kwargs() -> dict:
    import locale
    if platform.system() == "Windows":
        enc = locale.getpreferredencoding(False) or "cp1252"
    else:
        enc = locale.getpreferredencoding(False) or "utf-8"
    return {"text": True, "encoding": enc, "errors": "replace"}


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
            [str(venv_python), "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            **_safe_text_kwargs(),
        ).strip()
        maj, min_ = out.split(".")
        return int(maj), int(min_)
    except Exception:
        return None


def _tag_for_pyver(maj: int, min_: int) -> str:
    return f"py{maj}{min_}"


def _runtime_base_dir() -> Path:
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


# Supported Python minor versions in preference order.
_SUPPORTED_PY_MINORS = [12, 13, 14]


def _discover_existing_runtime_dir(status_cb=print) -> Path | None:
    global _RUNTIME_DISCOVERY_LOGGED

    base = _runtime_base_dir()
    if not base.exists():
        return None

    cur_minor = None  # frozen builds shouldn't lock to build Python version
    if not getattr(sys, "frozen", False):
        cur_minor = sys.version_info.minor if sys.version_info.major == 3 else None

    # If the current interpreter is a supported version, ONLY accept a runtime
    # that matches it exactly. Never fall back to a different Python version's
    # venv — that would mean running py314 SASpro against a py312 torch venv.
    if cur_minor in _SUPPORTED_PY_MINORS:
        tag = _tag_for_pyver(3, cur_minor)
        candidate = base / tag
        vpy = (candidate / "venv" / "Scripts" / "python.exe"
               if platform.system() == "Windows"
               else candidate / "venv" / "bin" / "python")

        if vpy.exists():
            actual_ver = _venv_pyver(vpy)
            if actual_ver == (3, cur_minor):
                if not _RUNTIME_DISCOVERY_LOGGED:
                    _rt_dbg(f"Found runtime: {candidate} (Python {actual_ver[0]}.{actual_ver[1]})", status_cb)
                    _RUNTIME_DISCOVERY_LOGGED = True
                return candidate
            else:
                _rt_dbg(
                    f"Skipping {candidate}: folder tag says py3{cur_minor} but venv "
                    f"reports Python {actual_ver[0]}.{actual_ver[1]}",
                    status_cb,
                )
        # No matching runtime for current Python — return None to trigger fresh install
        return None

    # Current interpreter is not a supported version (shouldn't normally happen).
    # Fall back to scanning all supported versions in preference order.
    for minor in _SUPPORTED_PY_MINORS:
        tag = _tag_for_pyver(3, minor)
        candidate = base / tag
        vpy = (candidate / "venv" / "Scripts" / "python.exe"
               if platform.system() == "Windows"
               else candidate / "venv" / "bin" / "python")

        if not vpy.exists():
            continue

        actual_ver = _venv_pyver(vpy)
        if actual_ver is None:
            _rt_dbg(f"Skipping {candidate}: could not determine venv Python version", status_cb)
            continue

        if actual_ver != (3, minor):
            _rt_dbg(
                f"Skipping {candidate}: folder tag says py3{minor} but venv "
                f"reports Python {actual_ver[0]}.{actual_ver[1]}",
                status_cb,
            )
            continue

        if not _RUNTIME_DISCOVERY_LOGGED:
            _rt_dbg(f"Found runtime: {candidate} (Python {actual_ver[0]}.{actual_ver[1]})", status_cb)
            _RUNTIME_DISCOVERY_LOGGED = True
        return candidate

    return None

def _detect_rocm_arch() -> str:
    """Return AMD ROCm gfx architecture string if detected (Linux only)."""
    try:
        if platform.system() != "Linux":
            return ""
        r = subprocess.run(
            ["rocminfo"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=8, **_safe_text_kwargs(),
        )
        out = r.stdout or ""
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Name:") and "gfx" in s:
                parts = s.split()
                for p in reversed(parts):
                    if p.startswith("gfx"):
                        return p
    except Exception:
        pass
    return ""


def _user_runtime_dir(status_cb=print) -> Path:
    global _RUNTIME_DIR_CACHED, _RUNTIME_USERDIR_LOGGED

    if _RUNTIME_DIR_CACHED is None:
        existing = _discover_existing_runtime_dir(status_cb=status_cb)
        if existing:
            _RUNTIME_DIR_CACHED = existing
        else:
            maj, min_ = sys.version_info.major, sys.version_info.minor
            if maj == 3 and min_ in _SUPPORTED_PY_MINORS:
                tag = _tag_for_pyver(maj, min_)
            else:
                # Current Python not supported — target the oldest supported
                # version as the install target, _ensure_venv will redirect
                # to the correct tag once it finds a supported interpreter.
                tag = _tag_for_pyver(3, _SUPPORTED_PY_MINORS[0])
            _RUNTIME_DIR_CACHED = _runtime_base_dir() / tag

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
    if (prefer_cuda or prefer_xpu or prefer_dml) and \
            getattr(getattr(torch, "backends", None), "mps", None) and \
            torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# CUDA version detection (system-level, no pip)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_cuda_version(status_cb=print) -> tuple[int, int] | None:
    # ---- 1. nvidia-smi ----
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=10, **_safe_text_kwargs(),
        )
        out = r.stdout or ""
        m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
        if m:
            ver = (int(m.group(1)), int(m.group(2)))
            _rt_dbg(f"CUDA {ver[0]}.{ver[1]} detected via nvidia-smi", status_cb)
            return ver
    except Exception:
        pass

    # ---- 2. nvcc ----
    try:
        r = subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=8, **_safe_text_kwargs(),
        )
        out = r.stdout or ""
        m = re.search(r"release\s+([0-9]+)\.([0-9]+)", out, re.I)
        if m:
            ver = (int(m.group(1)), int(m.group(2)))
            _rt_dbg(f"CUDA {ver[0]}.{ver[1]} detected via nvcc", status_cb)
            return ver
    except Exception:
        pass

    # ---- 3. Linux version files ----
    if platform.system() == "Linux":
        for cuda_root in ["/usr/local/cuda", "/usr/cuda"]:
            vj = Path(cuda_root) / "version.json"
            vt = Path(cuda_root) / "version.txt"
            if vj.exists():
                try:
                    data = json.loads(vj.read_text(encoding="utf-8", errors="replace"))
                    vs = (data.get("cuda", {}) or data).get("version", "")
                    if vs:
                        parts = str(vs).split(".")
                        ver = (int(parts[0]), int(parts[1]))
                        _rt_dbg(f"CUDA {ver[0]}.{ver[1]} detected via {vj}", status_cb)
                        return ver
                except Exception:
                    pass
            if vt.exists():
                try:
                    content = vt.read_text(encoding="utf-8", errors="replace")
                    m = re.search(r"([0-9]+)\.([0-9]+)", content)
                    if m:
                        ver = (int(m.group(1)), int(m.group(2)))
                        _rt_dbg(f"CUDA {ver[0]}.{ver[1]} detected via {vt}", status_cb)
                        return ver
                except Exception:
                    pass

    # ---- 4. Windows registry ----
    if platform.system() == "Windows":
        try:
            import winreg
            key_path = r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                i = 0
                best: tuple[int, int] | None = None
                while True:
                    try:
                        sub_name = winreg.EnumKey(key, i)
                        m = re.match(r"v?([0-9]+)\.([0-9]+)", sub_name)
                        if m:
                            candidate = (int(m.group(1)), int(m.group(2)))
                            if best is None or candidate > best:
                                best = candidate
                        i += 1
                    except OSError:
                        break
                if best:
                    _rt_dbg(f"CUDA {best[0]}.{best[1]} detected via Windows registry", status_cb)
                    return best
        except Exception:
            pass

    _rt_dbg("CUDA not detected on this system", status_cb)
    return None


def _cuda_ver_to_cu_tag(cuda_ver: tuple[int, int]) -> str | None:
    """
    Map (major, minor) CUDA version to nearest supported PyTorch cu-tag.
      CUDA 13.x      -> cu130
      CUDA 12.9      -> cu129
      CUDA 12.8      -> cu128
      CUDA 12.6-12.7 -> cu126
      CUDA 12.4-12.5 -> cu124
      CUDA 12.1-12.3 -> cu121
      CUDA 11.8      -> cu118
      CUDA < 11.8    -> None (too old)
    """
    maj, min_ = cuda_ver
    if maj >= 13:
        return "cu130"
    if maj == 12:
        if min_ >= 9:
            return "cu129"
        if min_ >= 8:
            return "cu128"
        if min_ >= 6:
            return "cu126"
        if min_ >= 4:
            return "cu124"
        return "cu121"
    if maj == 11:
        if min_ >= 8:
            return "cu118"
        return None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Version ladder & availability tables
# ──────────────────────────────────────────────────────────────────────────────

# Ordered newest-first — used when iterating CUDA install attempts
_TORCH_VERSION_LADDER_ORDERED = [
    "2.11.0", "2.10.0", "2.9.1", "2.9.0",
    "2.8.0",
    "2.7.0",
    "2.6.0",
    "2.5.1", "2.5.0",
    "2.4.1", "2.4.0",
    "2.3.1", "2.3.0",
]

# torchvision companion version for each torch version
_TORCHVISION_FOR_TORCH: dict[str, str] = {
    "2.11.0": "0.26.0",
    "2.10.0": "0.25.0",
    "2.9.1":  "0.24.1",
    "2.9.0":  "0.24.0",
    "2.8.0":  "0.23.0",
    "2.7.0":  "0.22.0",
    "2.6.0":  "0.21.0",
    "2.5.1":  "0.20.1",
    "2.5.0":  "0.20.0",
    "2.4.1":  "0.19.1",
    "2.4.0":  "0.19.0",
    "2.3.1":  "0.18.1",
    "2.3.0":  "0.18.0",
}

# Which cu-tags have published CUDA wheels for each torch version
_TORCH_CUDA_AVAILABILITY: dict[str, set[str]] = {
    "2.11.0": {"cu128", "cu129", "cu130"},
    "2.10.0": {"cu126", "cu128", "cu129"},
    "2.9.1":  {"cu126", "cu128", "cu130"},
    "2.9.0":  {"cu126", "cu128", "cu130"},
    "2.8.0":  {"cu126", "cu128", "cu129"},
    "2.7.0":  {"cu126", "cu128", "cu129"},
    "2.6.0":  {"cu124", "cu126", "cu128", "cu129"},
    "2.5.1":  {"cu118", "cu121", "cu124", "cu128"},
    "2.5.0":  {"cu118", "cu121", "cu124", "cu128"},
    "2.4.1":  {"cu118", "cu121", "cu124"},
    "2.4.0":  {"cu118", "cu121", "cu124"},
    "2.3.1":  {"cu118", "cu121"},
    "2.3.0":  {"cu118", "cu121"},
}

# Which Python versions have CUDA wheels for each torch version
# cp314 CUDA wheels exist from torch 2.9.x onward (cu126+)
_TORCH_CUDA_PYTHON_AVAILABILITY: dict[str, set[str]] = {
    "2.11.0": {"cp312", "cp313", "cp314"},
    "2.10.0": {"cp312", "cp313", "cp314"},
    "2.9.1":  {"cp312", "cp313", "cp314"},
    "2.9.0":  {"cp312", "cp313", "cp314"},
    "2.8.0":  {"cp312", "cp313"},
    "2.7.0":  {"cp312", "cp313"},
    "2.6.0":  {"cp312", "cp313"},
    "2.5.1":  {"cp312", "cp313"},
    "2.5.0":  {"cp312", "cp313"},
    "2.4.1":  {"cp312"},
    "2.4.0":  {"cp312"},
    "2.3.1":  {"cp312"},
    "2.3.0":  {"cp312"},
}

# Which Python versions have CPU wheels for each torch version (broader)
_TORCH_CPU_PYTHON_AVAILABILITY: dict[str, set[str]] = {
    "2.11.0": {"cp312", "cp313", "cp314"},
    "2.10.0": {"cp312", "cp313", "cp314"},
    "2.9.1":  {"cp312", "cp313", "cp314"},
    "2.9.0":  {"cp312", "cp313", "cp314"},
    "2.8.0":  {"cp312", "cp313", "cp314"},
    "2.7.0":  {"cp312", "cp313"},
    "2.6.0":  {"cp312", "cp313"},
    "2.5.1":  {"cp312", "cp313"},
    "2.5.0":  {"cp312", "cp313"},
    "2.4.1":  {"cp312"},
    "2.4.0":  {"cp312"},
    "2.3.1":  {"cp312"},
    "2.3.0":  {"cp312"},
}

_PYTORCH_WHL_BASE = "https://download.pytorch.org/whl"

# Per-Python-version version ladder for index-url installs (ROCm, XPU, CPU fallback)
_TORCH_VERSION_LADDER: dict[tuple[int, int], list[str]] = {
    (3, 12): ["2.11.*", "2.10.*", "2.9.*", "2.8.*", "2.7.*", "2.6.*", "2.5.*", "2.4.*", "2.3.*"],
    (3, 13): ["2.11.*", "2.10.*", "2.9.*", "2.8.*", "2.7.*", "2.6.*", "2.5.*"],
    (3, 14): ["2.11.*", "2.10.*", "2.9.*", "2.8.*"],
}

_TORCH_COMPAT: dict[str, tuple[str, str]] = {
    "2.11": ("0.26.*", "2.11.*"),
    "2.10": ("0.25.*", "2.10.*"),
    "2.9":  ("0.24.*", "2.9.*"),
    "2.8":  ("0.23.*", "2.8.*"),
    "2.7":  ("0.22.*", "2.7.*"),
    "2.6":  ("0.21.*", "2.6.*"),
    "2.5":  ("0.20.*", "2.5.*"),
    "2.4":  ("0.19.*", "2.4.*"),
    "2.3":  ("0.18.*", "2.3.*"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Platform helpers
# ──────────────────────────────────────────────────────────────────────────────

def _platform_tag(venv_python: Path | None = None) -> str:
    sysname = platform.system()
    machine = platform.machine().lower()
    if sysname == "Windows":
        return "win_amd64"
    if sysname == "Darwin":
        if "arm64" in machine or "aarch64" in machine:
            return "macosx_11_0_arm64"
        return "macosx_10_9_x86_64"
    if "aarch64" in machine or "arm64" in machine:
        return "linux_aarch64"
    return "linux_x86_64"


def _cp_tag(venv_python: Path) -> str | None:
    ver = _venv_pyver(venv_python)
    if ver is None:
        return None
    return f"cp{ver[0]}{ver[1]}"


# ──────────────────────────────────────────────────────────────────────────────
# Shadowing & sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def _is_compiled_torch_dir(d: Path) -> bool:
    return any(d.glob("_C.*.pyd")) or any(d.glob("_C.*.so")) or any(d.glob("_C.cpython*"))


def _looks_like_source_tree_torch(d: Path) -> bool:
    return (d / "_C" / "__init__.py").exists()


def _ban_shadow_torch_paths(status_cb=print) -> None:
    keep: list[str] = []
    banned: list[str] = []
    for entry in list(sys.path):
        try:
            base = Path(entry) if entry else Path.cwd()
            td = base / "torch"
            if td.is_dir():
                if _looks_like_source_tree_torch(td):
                    banned.append(entry or "<cwd>")
                    continue
                if not _is_compiled_torch_dir(td):
                    banned.append(entry or "<cwd>")
                    continue
        except Exception:
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
    try:
        if "torch" in sys.modules:
            mod = sys.modules["torch"]
            tf = getattr(mod, "__file__", "") or ""

            # Detect our own stub by its sentinel version string
            is_stub = (
                not tf and
                getattr(mod, "__version__", "") == "0.0.0+unavailable"
            )

            if is_stub or (tf and ("site-packages" not in tf) and ("dist-packages" not in tf)):
                for k in list(sys.modules.keys()):
                    if k == "torch" or k.startswith("torch."):
                        sys.modules.pop(k, None)
                if is_stub:
                    status_cb("[RT] Purged torch stub from sys.modules before real import")
                else:
                    status_cb(f"[RT] Purged shadowed torch import: {tf}")

        sys.modules.pop("torch._C", None)
        importlib.invalidate_caches()
    except Exception:
        pass


def _torch_sanity_check(status_cb=print):
    try:
        import torch
        tf = getattr(torch, "__file__", "") or ""
        pkg_dir = Path(tf).parent if tf else None

        if ("site-packages" not in tf) and ("dist-packages" not in tf):
            raise RuntimeError(f"Shadow import: torch.__file__ = {tf}")
        if not _is_compiled_torch_dir(pkg_dir):
            raise RuntimeError(f"Wheel missing torch._C in {pkg_dir}")
        if (pkg_dir / "_C" / "__init__.py").exists():
            raise RuntimeError(f"Source tree detected at torch/_C: {pkg_dir / '_C'}")

        importlib.import_module("torch._C")

        x = torch.ones(1)
        if int((x + 1).item()) != 2:
            raise RuntimeError("Unexpected tensor arithmetic result.")

        if hasattr(torch, "inference_mode"):
            try:
                with torch.inference_mode():
                    _ = (torch.ones(1) + 1).item()
            except Exception as e:
                _rt_dbg(f"torch.inference_mode not available in this build ({e}); continuing.", status_cb)

    except Exception as e:
        raise RuntimeError(f"PyTorch C-extension check failed: {e}") from e


# ──────────────────────────────────────────────────────────────────────────────
# OS / permissions helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pip_run(venv_python: Path, args: list[str], status_cb=print) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return subprocess.run(
        [str(venv_python), "-m", "pip", *args],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, **_safe_text_kwargs(),
    )


def _pip_ok(venv_python: Path, args: list[str], status_cb=print) -> bool:
    r = _pip_run(venv_python, args, status_cb=status_cb)
    if r.returncode != 0:
        tail = (r.stdout or "").strip()
        try:
            status_cb(tail[-4000:])
        except Exception:
            pass
    return r.returncode == 0


def _ensure_numpy(venv_python: Path, status_cb=print) -> None:
    def _numpy_state() -> tuple[bool, int | None, str | None]:
        code = (
            "import json, importlib.util\n"
            "spec = importlib.util.find_spec('numpy')\n"
            "if spec is None:\n"
            "    print(json.dumps({'present': False, 'major': None, 'version': None}))\n"
            "else:\n"
            "    import numpy as np\n"
            "    v = str(np.__version__).split('+',1)[0]\n"
            "    try:\n"
            "        major = int(v.split('.',1)[0])\n"
            "    except Exception:\n"
            "        major = None\n"
            "    print(json.dumps({'present': True, 'major': major, 'version': v}))\n"
        )
        try:
            out = subprocess.check_output(
                [str(venv_python), "-c", code], **_safe_text_kwargs(),
            ).strip()
            data = json.loads(out) if out else {}
            return bool(data.get("present")), data.get("major"), data.get("version")
        except Exception:
            return False, None, None

    present, major, version = _numpy_state()

    # For Python 3.14, torch 2.9+ supports numpy 2.x — don't force numpy<2
    # The numpy<2 MINGW experimental build causes warnings and slowdowns
    ver = _venv_pyver(venv_python)
    if ver and ver[1] >= 14:
        if present:
            return  # whatever numpy is installed is fine for cp314
        status_cb("[RT] Installing NumPy for Python 3.14 (no version pin)…")
        _pip_ok(venv_python, ["install", "--prefer-binary", "--no-cache-dir", "numpy"], status_cb=status_cb)
        return

    if present and (major is None or major < 2):
        return  # healthy for py312/313, no-op

    _pip_ok(venv_python, ["install", "--upgrade", "pip", "setuptools", "wheel"], status_cb=status_cb)

    if not present:
        status_cb("[RT] Installing NumPy (pinning to numpy<2 for torch wheel compatibility)…")
        if not _pip_ok(venv_python, ["install", "--prefer-binary", "--no-cache-dir", "numpy<2"], status_cb=status_cb):
            _pip_ok(venv_python, ["install", "--prefer-binary", "--no-cache-dir", "numpy==1.26.*"], status_cb=status_cb)
    elif major is not None and major >= 2:
        status_cb(f"[RT] NumPy {version or '2.x'} detected; downgrading to numpy<2…")
        if not _pip_ok(venv_python, ["install", "--prefer-binary", "--no-cache-dir", "--force-reinstall", "numpy<2"], status_cb=status_cb):
            _pip_ok(venv_python, ["install", "--prefer-binary", "--no-cache-dir", "--force-reinstall", "numpy==1.26.*"], status_cb=status_cb)

    present2, major2, version2 = _numpy_state()
    if not present2:
        raise RuntimeError("Failed to install NumPy into the SASpro runtime venv.")
    if major2 is not None and major2 >= 2:
        raise RuntimeError(f"NumPy is still {version2 or '2.x'} after pinning; torch stack may not import.")


def _is_access_denied(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
    if getattr(exc, "errno", None) == errno.EACCES:
        return True
    return getattr(exc, "winerror", None) == 5


def _access_denied_msg(base_path: Path) -> str:
    return (
        "Access denied while preparing the SASpro runtime at:\n"
        f"  {base_path}\n\n"
        "Possible causes:\n"
        " • A corporate policy blocks writing to %LOCALAPPDATA%.\n"
        " • Security software is sandboxing the app.\n\n"
        "Fixes:\n"
        " 1) Run SASpro once as Administrator (right-click -> Run as administrator), or\n"
        " 2) Set an alternate writable folder via environment variable SASPRO_RUNTIME_DIR\n"
        "    (e.g. C:\\Users\\<you>\\SASproRuntime) and relaunch."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Venv creation & site discovery
# ──────────────────────────────────────────────────────────────────────────────

def _venv_paths(rt: Path):
    return {
        "venv": rt / "venv",
        "python": (rt / "venv" / "Scripts" / "python.exe")
                  if platform.system() == "Windows"
                  else (rt / "venv" / "bin" / "python"),
        "marker": rt / "torch_installed.json",
    }


def _site_packages(venv_python: Path) -> Path:
    code = "import site, sys; print([p for p in site.getsitepackages() if 'site-packages' in p][-1])"
    out = subprocess.check_output(
        [str(venv_python), "-c", code], **_safe_text_kwargs(),
    ).strip()
    return Path(out)


def _ensure_venv(rt: Path, status_cb=print) -> Path:
    p = _venv_paths(rt)
    bootstrap_marker = p["venv"] / ".saspro_bootstrapped"

    if not p["python"].exists():
        try:
            status_cb(f"Setting up SASpro runtime venv at: {p['venv']}")
            p["venv"].mkdir(parents=True, exist_ok=True)

            py_cmd = _find_system_python_cmd() if getattr(sys, "frozen", False) else [sys.executable]
            out = subprocess.check_output(
                py_cmd + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                **_safe_text_kwargs(),
            ).strip()
            maj, min_ = map(int, out.split("."))

            if maj != 3 or min_ not in _SUPPORTED_PY_MINORS:
                raise RuntimeError(
                    f"SASpro runtime requires Python 3.12, 3.13, or 3.14, "
                    f"but found {maj}.{min_}.\n"
                    "Install a supported Python version and relaunch SAS Pro."
                )

            expected_tag = _tag_for_pyver(maj, min_)
            if rt.name != expected_tag:
                correct_rt = _runtime_base_dir() / expected_tag
                status_cb(f"Redirecting venv creation to {correct_rt} (interpreter is {maj}.{min_})")
                # Update the global cache to point at the correct runtime dir
                global _RUNTIME_DIR_CACHED
                _RUNTIME_DIR_CACHED = correct_rt
                return _ensure_venv(correct_rt, status_cb=status_cb)

            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env.pop("PYTHONPATH", None)
            subprocess.check_call(py_cmd + ["-m", "venv", str(p["venv"])], env=env)
            subprocess.check_call([str(p["python"]), "-m", "ensurepip", "--upgrade"], env=env)
            subprocess.check_call(
                [str(p["python"]), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"],
                env=env,
            )
            try:
                bootstrap_marker.write_text(f"{maj}.{min_}", encoding="utf-8")
            except Exception:
                pass

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
        ver = _venv_pyver(p["python"])
        if ver and (ver[0] != 3 or ver[1] not in _SUPPORTED_PY_MINORS):
            status_cb(
                f"Runtime venv is Python {ver[0]}.{ver[1]} which is no longer supported. Rebuilding."
            )
            shutil.rmtree(p["venv"], ignore_errors=True)
            return _ensure_venv(rt, status_cb=status_cb)

        if ver:
            expected_minor = int(rt.name.replace("py3", "")) if rt.name.startswith("py3") else None
            if expected_minor and ver[1] != expected_minor:
                status_cb(
                    f"Runtime venv is Python {ver[0]}.{ver[1]} but folder tag is {rt.name}. Rebuilding."
                )
                shutil.rmtree(p["venv"], ignore_errors=True)
                return _ensure_venv(rt, status_cb=status_cb)

    return p["python"]


# ──────────────────────────────────────────────────────────────────────────────
# Install locking
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _install_lock(rt: Path, timeout_s: int = 600):
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


# ──────────────────────────────────────────────────────────────────────────────
# Module-level caches & ROCm constants
# ──────────────────────────────────────────────────────────────────────────────

_TORCH_CACHED = None
_RUNTIME_DIR_CACHED: Path | None = None
_RUNTIME_DISCOVERY_LOGGED = False
_RUNTIME_USERDIR_LOGGED = False

_ROCM_PYTORCH_NIGHTLY_INDICES: list[tuple[str, str]] = [
    ("rocm7.2", "https://download.pytorch.org/whl/nightly/rocm7.2"),
    ("rocm7.1", "https://download.pytorch.org/whl/nightly/rocm7.1"),
    ("rocm7.0", "https://download.pytorch.org/whl/nightly/rocm7.0"),
    ("rocm6.4", "https://download.pytorch.org/whl/nightly/rocm6.4"),
    ("rocm6.3", "https://download.pytorch.org/whl/nightly/rocm6.3"),
    ("rocm6.2", "https://download.pytorch.org/whl/nightly/rocm6.2"),
]

_ROCM_AMD_GFX_BUCKETS_WITH_TORCH = {
    "gfx1151",
    "gfx1100", "gfx1101", "gfx1102",
    "gfx1200", "gfx1201",
}


# ──────────────────────────────────────────────────────────────────────────────
# Per-venv hardware checks
# ──────────────────────────────────────────────────────────────────────────────

def _check_cuda_in_venv(venv_python: Path, status_cb=print) -> tuple[bool, str | None, str | None]:
    code = r"""
import json, sys
try:
    import torch
    info = {
        "cuda_tag": getattr(getattr(torch, "version", None), "cuda", None),
        "has_cuda": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
        "err": None,
    }
    if info["has_cuda"]:
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
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **_safe_text_kwargs(),
    )
    out = (r.stdout or "").strip()
    last = out.splitlines()[-1] if out else ""
    try:
        data = json.loads(last) if last else {}
    except Exception as e:
        msg = f"Failed to parse CUDA check output: {e}\nRaw:\n{out}"
        try:
            status_cb(msg)
        except Exception:
            pass
        return False, None, msg
    return bool(data.get("has_cuda")), data.get("cuda_tag"), data.get("err")


def _check_xpu_in_venv(venv_python: Path, status_cb=print) -> tuple[bool, str | None]:
    code = r"""
import json, sys
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
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **_safe_text_kwargs(),
    )
    out = (r.stdout or "").strip()
    last = out.splitlines()[-1] if out else ""
    try:
        data = json.loads(last) if last else {}
    except Exception as e:
        msg = f"Failed to parse XPU check output: {e}\nRaw:\n{out}"
        try:
            status_cb(msg)
        except Exception:
            pass
        return False, msg
    return bool(data.get("has_xpu")), data.get("err")


def _detect_rocm_version(status_cb=print) -> str:
    if platform.system() != "Linux":
        return ""

    patterns = [
        re.compile(r"\bHIP(?:\s+runtime)?\s+version\s*:\s*([0-9]+)\.([0-9]+)", re.I),
        re.compile(r"\bROCm(?:\s+version)?\s*[:=]?\s*([0-9]+)\.([0-9]+)", re.I),
        re.compile(r"\brocm[-\s]?([0-9]+)\.([0-9]+)(?:\.[0-9]+)?\b", re.I),
    ]

    def _extract_ver(text: str) -> str:
        for line in (text or "").splitlines():
            for pat in patterns:
                m = pat.search(line.strip())
                if m:
                    return f"{m.group(1)}.{m.group(2)}"
        return ""

    for cmd, timeout_s in [(["hipconfig", "--version"], 6), (["rocminfo", "--support"], 8), (["rocminfo"], 8)]:
        try:
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout_s, **_safe_text_kwargs(),
            )
            ver = _extract_ver(r.stdout or "")
            if ver:
                try:
                    status_cb(f"[RT] Detected ROCm version: {ver} (via {' '.join(cmd)})")
                except Exception:
                    pass
                return ver
        except Exception:
            continue
    return ""


def _ordered_rocm_nightly_indices(detected_rocm_ver: str, status_cb=print) -> list[tuple[str, str]]:
    base = list(_ROCM_PYTORCH_NIGHTLY_INDICES)
    if not detected_rocm_ver:
        return base
    target_label = f"rocm{detected_rocm_ver}"
    preferred = [(l, u) for l, u in base if l == target_label]
    rest = [(l, u) for l, u in base if l != target_label]
    if preferred:
        try:
            status_cb(f"[RT] Prioritizing ROCm nightly index for detected ROCm {detected_rocm_ver}")
        except Exception:
            pass
    return preferred + rest


# ──────────────────────────────────────────────────────────────────────────────
# Core torch install — index-url strategy (no URL probing)
# ──────────────────────────────────────────────────────────────────────────────

def _install_torch(
    venv_python,
    prefer_cuda: bool,
    prefer_xpu: bool,
    prefer_dml: bool,
    prefer_rocm: bool = False,
    status_cb=print,
):
    """
    Install torch/torchvision into the runtime venv using pip --index-url.
    No URL probing — the PyTorch CDN blocks HEAD requests (403).
    pip handles 404/no-match gracefully.

    cp314 + CUDA: wheels exist from torch 2.9.x onward on cu126+.
    """
    sysname = platform.system()
    machine = platform.machine().lower()

    def _pip_install_ok(cmd: list[str]) -> bool:
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        r = subprocess.run(
            [str(venv_python), "-m", "pip", *cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, **_safe_text_kwargs(),
        )
        if r.returncode != 0:
            tail = (r.stdout or "").strip()
            status_cb(tail[-4000:])
        return r.returncode == 0

    # Keep venv tools fresh
    _pip_install_ok(["install", "--upgrade", "pip", "setuptools", "wheel"])

    cp = _cp_tag(venv_python)
    ver = _venv_pyver(venv_python)

    if not cp or not ver:
        raise RuntimeError("Could not determine Python version of the runtime venv.")

    plat = _platform_tag(venv_python)
    status_cb(f"[RT] Runtime: Python {ver[0]}.{ver[1]}, cp={cp}, platform={plat}")

    # ── macOS Apple Silicon → MPS (PyPI, no CUDA) ─────────────────────────────
    if sysname == "Darwin" and ("arm64" in machine or "aarch64" in machine):
        status_cb("Installing PyTorch for macOS arm64 (MPS via PyPI)…")
        ladder = _TORCH_VERSION_LADDER.get((ver[0], ver[1]), ["2.11.*"])
        for tv in ladder:
            if _pip_install_ok(["install", "--prefer-binary", "--no-cache-dir",
                                 f"torch=={tv}", "torchvision"]):
                return
        raise RuntimeError("Failed to find a matching PyTorch wheel for macOS arm64.")

    # ── AMD ROCm (Linux only) ──────────────────────────────────────────────────
    rocm_arch = _detect_rocm_arch() if (prefer_rocm and sysname == "Linux") else ""
    rocm_ver = _detect_rocm_version(status_cb=status_cb) if (prefer_rocm and sysname == "Linux") else ""

    if rocm_arch:
        status_cb(f"ROCm GPU detected: {rocm_arch}" + (f" (ROCm {rocm_ver})" if rocm_ver else ""))
        rocm_candidates: list[tuple[str, str, bool]] = []

        if rocm_arch in _ROCM_AMD_GFX_BUCKETS_WITH_TORCH:
            rocm_candidates.append((
                f"amd-gfx-nightly:{rocm_arch}",
                f"https://rocm.nightlies.amd.com/v2/{rocm_arch}/",
                True,
            ))
        else:
            status_cb(f"ROCm arch {rocm_arch}: no AMD gfx nightly bucket; trying PyTorch nightly ROCm.")

        env_rocm_nightly = (os.getenv("SASPRO_ROCM_NIGHTLY_INDEX") or "").strip()
        if env_rocm_nightly:
            rocm_candidates.insert(0, ("env-rocm-nightly", env_rocm_nightly.rstrip("/"), True))

        for label, url in _ordered_rocm_nightly_indices(rocm_ver, status_cb=status_cb):
            rocm_candidates.append((f"pytorch-nightly:{label}", url, True))

        ladder = _TORCH_VERSION_LADDER.get((ver[0], ver[1]), ["2.11.*"])

        for label, url, use_pre in rocm_candidates:
            status_cb(f"Trying PyTorch ROCm wheels from {label}…")
            base = ["install", "--prefer-binary", "--no-cache-dir", "--index-url", url]
            if use_pre:
                base.append("--pre")

            installed = False
            for v in ladder:
                fam = v.replace(".*", "")
                tv_v, _ = _TORCH_COMPAT.get(fam, (None, None))
                if tv_v and _pip_install_ok(base + [f"torch=={v}", f"torchvision=={tv_v}"]):
                    installed = True
                    break
            if not installed:
                installed = _pip_install_ok(base + ["torch", "torchvision"])

            if installed:
                ok, cuda_tag, err = _check_cuda_in_venv(venv_python, status_cb=status_cb)
                if ok:
                    status_cb(f"Installed PyTorch ROCm from {label} (torch.version.cuda={cuda_tag}).")
                    return
                status_cb(f"ROCm check failed ({cuda_tag!r}, {err!r}). Trying next…")
                _pip_install_ok(["uninstall", "-y", "torch", "torchvision", "torchaudio"])
            else:
                status_cb(f"No compatible PyTorch ROCm wheels from {label}.")

        status_cb("ROCm install attempts failed. Falling back to other backends…")

    # ── Intel XPU ─────────────────────────────────────────────────────────────
    INTEL_XPU_INDEX = "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    if prefer_xpu and sysname in ("Windows", "Linux"):
        status_cb("Trying PyTorch Intel XPU wheels…")
        ladder = _TORCH_VERSION_LADDER.get((ver[0], ver[1]), ["2.11.*"])
        base = ["install", "--prefer-binary", "--no-cache-dir", "--index-url", INTEL_XPU_INDEX]
        installed = False
        for v in ladder:
            fam = v.replace(".*", "")
            tv_v, _ = _TORCH_COMPAT.get(fam, (None, None))
            if tv_v and _pip_install_ok(base + [f"torch=={v}", f"torchvision=={tv_v}"]):
                installed = True
                break
        if not installed:
            installed = _pip_install_ok(base + ["torch", "torchvision"])

        if installed:
            ok, err = _check_xpu_in_venv(venv_python, status_cb=status_cb)
            if ok:
                status_cb("Installed PyTorch Intel XPU.")
                return
            status_cb(f"XPU runtime test failed: {err!r}. Uninstalling and falling back…")
            _pip_install_ok(["uninstall", "-y", "torch", "torchvision", "torchaudio"])
        else:
            status_cb("No matching Intel XPU wheel for this Python/OS.")

    # ── CUDA: index-url installs walking version + cu-tag ladder ──────────────
    if prefer_cuda and sysname in ("Windows", "Linux"):
        cuda_ver = _detect_cuda_version(status_cb=status_cb)

        if cuda_ver is None:
            status_cb("[RT] CUDA not detected on this system; skipping CUDA install.")
        else:
            cu_tag = _cuda_ver_to_cu_tag(cuda_ver)
            if cu_tag is None:
                status_cb(
                    f"[RT] CUDA {cuda_ver[0]}.{cuda_ver[1]} is too old for current PyTorch wheels. "
                    "Falling back to CPU."
                )
            else:
                status_cb(
                    f"[RT] CUDA {cuda_ver[0]}.{cuda_ver[1]} -> {cu_tag}, "
                    f"installing via pip index-url for {cp}…"
                )

                all_cu_tags = ["cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118"]
                cu_tag_order = [cu_tag] + [t for t in all_cu_tags if t != cu_tag]

                installed_cuda = False
                for try_cu in cu_tag_order:
                    index_url = f"{_PYTORCH_WHL_BASE}/{try_cu}"

                    for torch_ver in _TORCH_VERSION_LADDER_ORDERED:
                        avail_cu = _TORCH_CUDA_AVAILABILITY.get(torch_ver, set())
                        avail_cp = _TORCH_CUDA_PYTHON_AVAILABILITY.get(torch_ver, set())
                        if try_cu not in avail_cu or cp not in avail_cp:
                            continue

                        tv_ver = _TORCHVISION_FOR_TORCH.get(torch_ver, "")
                        status_cb(f"[RT] Trying torch=={torch_ver}+{try_cu} via {index_url}…")

                        base = ["install", "--prefer-binary", "--no-cache-dir",
                                "--index-url", index_url]

                        if tv_ver:
                            ok = _pip_install_ok(base + [f"torch=={torch_ver}", f"torchvision=={tv_ver}"])
                        else:
                            ok = _pip_install_ok(base + [f"torch=={torch_ver}", "torchvision"])

                        if not ok:
                            continue

                        cuda_ok, cuda_tag_found, err = _check_cuda_in_venv(venv_python, status_cb=status_cb)
                        if cuda_ok:
                            status_cb(f"[RT] CUDA verified (torch {torch_ver}+{try_cu}, cuda={cuda_tag_found}).")
                            installed_cuda = True
                            break
                        else:
                            status_cb(
                                f"[RT] Installed from {try_cu} but CUDA not active "
                                f"(cuda_tag={cuda_tag_found!r}, err={err!r}). Uninstalling…"
                            )
                            _pip_install_ok(["uninstall", "-y", "torch", "torchvision"])

                    if installed_cuda:
                        return

                if not installed_cuda:
                    status_cb(
                        f"[RT] Could not install a working CUDA torch for {cp}. "
                        "Falling back to CPU."
                    )

    # ── DirectML (Windows, non-NVIDIA) ────────────────────────────────────────
    if sysname == "Windows" and prefer_dml:
        status_cb("Installing PyTorch with DirectML (torch-directml)…")
        _pip_install_ok(["uninstall", "-y", "torch", "torchvision", "torch-directml"])

        if not _pip_install_ok(["install", "--prefer-binary", "--no-cache-dir", "torch-directml"]):
            raise RuntimeError("Failed to install torch-directml.")

        _pip_install_ok(["install", "--prefer-binary", "--no-cache-dir", "torchvision"])

        code = (
            "import torch, torch_directml; d=torch_directml.device(); "
            "x=torch.tensor([1]).to(d); y=torch.tensor([2]).to(d); print(int((x+y).item()))"
        )
        r = subprocess.run(
            [str(venv_python), "-c", code],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **_safe_text_kwargs(),
        )
        if r.returncode != 0 or "3" not in (r.stdout or ""):
            status_cb((r.stdout or "")[-2000:])
            raise RuntimeError("torch-directml installed, but DirectML verification failed.")

        status_cb("Installed DirectML backend successfully.")
        return

    # ── CPU fallback ──────────────────────────────────────────────────────────
    status_cb(f"[RT] Installing PyTorch (CPU) for {cp}…")

    ladder = _TORCH_VERSION_LADDER.get((ver[0], ver[1]), ["2.11.*"])

    cpu_index = f"{_PYTORCH_WHL_BASE}/cpu"
    base_cpu = ["install", "--prefer-binary", "--no-cache-dir", "--index-url", cpu_index]

    for v in ladder:
        fam = v.replace(".*", "")
        tv_v, _ = _TORCH_COMPAT.get(fam, (None, None))
        if tv_v and _pip_install_ok(base_cpu + [f"torch=={v}", f"torchvision=={tv_v}"]):
            status_cb(f"[RT] CPU torch {v} installed via PyTorch CPU index.")
            return
        if _pip_install_ok(base_cpu + [f"torch=={v}", "torchvision"]):
            status_cb(f"[RT] CPU torch {v} installed via PyTorch CPU index.")
            return

    status_cb("[RT] PyTorch CPU index failed; trying PyPI…")
    base_pypi = ["install", "--prefer-binary", "--no-cache-dir"]
    for v in ladder:
        fam = v.replace(".*", "")
        tv_v, _ = _TORCH_COMPAT.get(fam, (None, None))
        if tv_v and _pip_install_ok(base_pypi + [f"torch=={v}", f"torchvision=={tv_v}"]):
            return
        if _pip_install_ok(base_pypi + [f"torch=={v}", "torchvision"]):
            return

    raise RuntimeError(
        f"Failed to install any compatible PyTorch wheel for {cp}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────────────

def _venv_import_probe(venv_python: Path, modname: str) -> tuple[bool, str]:
    code = (
        "import warnings as _w\n"
        "_w.filterwarnings('ignore')\n"
        "import importlib, sys\n"
        f"m=importlib.import_module('{modname}')\n"
        "print('OK', getattr(m,'__version__',None), getattr(m,'__file__',None))\n"
    )
    r = subprocess.run(
        [str(venv_python), "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **_safe_text_kwargs(),
    )
    out = (r.stdout or "").strip()

    if r.returncode == 0:
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("OK"):
                return True, line

    return False, out[-4000:] if out else "no output"


def _write_torch_marker(marker: Path, status_cb=print, *, require_torchaudio: bool = False) -> None:
    rt = marker.parent
    vp = _venv_paths(rt)["python"]

    ok_t, out_t = _venv_import_probe(vp, "torch")
    ok_v, out_v = _venv_import_probe(vp, "torchvision")
    ok_a, out_a = _venv_import_probe(vp, "torchaudio")

    payload = {
        "installed": bool(ok_t),
        "when": int(time.time()),
        "python": None,
        "torch": None, "torchvision": None, "torchaudio": None,
        "torch_file": None, "torchvision_file": None, "torchaudio_file": None,
        "require_torchaudio": bool(require_torchaudio),
        "probe": {"torch": out_t, "torchvision": out_v, "torchaudio": out_a},
    }

    try:
        r = subprocess.run(
            [str(vp), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **_safe_text_kwargs(),
        )
        if r.returncode == 0:
            payload["python"] = (r.stdout or "").strip()
    except Exception:
        pass

    def _parse_ok(s: str):
        try:
            parts = s.split(" ", 2)
            return (parts[1] if len(parts) > 1 else None), (parts[2] if len(parts) > 2 else None)
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
    require_torchaudio: bool = False,
) -> tuple[bool, dict]:
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
    require_torchaudio: bool = False,
    max_age_days: int = 180,
) -> bool:
    try:
        if not marker.exists():
            return False
        raw = marker.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict) or not data.get("installed", False):
            return False

        when = data.get("when")
        if isinstance(when, (int, float)):
            if max(0.0, time.time() - float(when)) > (max_age_days * 86400):
                return False

        py = data.get("python")
        if not (isinstance(py, str) and py.strip()):
            return False
        try:
            maj_s, min_s = py.strip().split(".", 1)
            marker_ver = (int(maj_s), int(min_s))
        except Exception:
            return False
        if venv_ver is None or marker_ver != venv_ver:
            return False

        site_s = str(site)

        def _under_site(p: str | None) -> bool:
            return bool(p and isinstance(p, str) and site_s in p)

        if not _under_site(data.get("torch_file")):
            return False
        if not _under_site(data.get("torchvision_file")):
            return False
        if require_torchaudio and not _under_site(data.get("torchaudio_file")):
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


def import_torch(
    prefer_cuda: bool = True,
    prefer_xpu: bool = False,
    prefer_dml: bool = False,
    prefer_rocm: bool = False,
    status_cb=print,
    *,
    require_torchaudio: bool = False,
    allow_install: bool = False,
):
    """Return the runtime-venv torch module, installing if allow_install=True."""
    global _TORCH_CACHED
    if _TORCH_CACHED is not None:
        return _TORCH_CACHED

    # Conservative ROCm redirect on Linux
    if platform.system() == "Linux" and prefer_cuda and not prefer_rocm:
        try:
            rocm_arch = _detect_rocm_arch()
            if rocm_arch:
                try:
                    import importlib as _il
                    _t = _il.import_module("torch")
                    has_cuda_now = bool(getattr(_t, "cuda", None) and _t.cuda.is_available())
                    hip_now = bool(getattr(getattr(_t, "version", None), "hip", None))
                    if not (has_cuda_now and not hip_now):
                        prefer_cuda = False
                        prefer_rocm = True
                        status_cb(f"[RT] AMD ROCm GPU detected ({rocm_arch}); redirecting to ROCm.")
                except Exception:
                    prefer_cuda = False
                    prefer_rocm = True
                    status_cb(f"[RT] AMD ROCm GPU detected ({rocm_arch}); redirecting to ROCm.")
        except Exception:
            pass

    if platform.system() == "Windows" and prefer_cuda:
        prefer_dml = False

    def _write_cache_best_effort(rt: Path, site: Path, venv_ver: tuple[int, int] | None):
        try:
            import torch as _t
            import torchvision as _tv
            _ta = None
            if require_torchaudio:
                import torchaudio as _ta
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

    def _raise_missing_runtime(info: dict) -> None:
        missing = [k for k in ("torch", "torchvision") if not info[k][0]]
        if require_torchaudio and not info["torchaudio"][0]:
            missing.append("torchaudio")
        if prefer_cuda or prefer_xpu or prefer_dml or prefer_rocm:
            msg = (
                "GPU acceleration runtime is not installed or is incomplete.\n\n"
                f"Missing packages in runtime venv: {', '.join(missing)}\n\n"
                "Please go to Settings -> Preferences and install GPU Acceleration "
                "before running Cosmic Clarity, SyQon, or other hardware accelerated tools."
            )
        else:
            msg = (
                "Hardware acceleration runtime is not installed or is incomplete.\n\n"
                f"Missing packages in runtime venv: {', '.join(missing)}"
            )
        status_cb(f"[RT] {msg}")
        raise RuntimeError(msg)

    _rt_dbg(f"sys.frozen={getattr(sys, 'frozen', False)}", status_cb)
    _rt_dbg(f"sys.executable={sys.executable}", status_cb)
    _rt_dbg(f"sys.version={sys.version}", status_cb)
    _rt_dbg(f"SASPRO_RUNTIME_DIR={os.getenv('SASPRO_RUNTIME_DIR')!r}", status_cb)
    _rt_dbg(
        f"prefs: cuda={prefer_cuda} xpu={prefer_xpu} dml={prefer_dml} rocm={prefer_rocm} "
        f"allow_install={allow_install}",
        status_cb,
    )

    _ban_shadow_torch_paths(status_cb=status_cb)
    _purge_bad_torch_from_sysmodules(status_cb=status_cb)

    rt = _user_runtime_dir(status_cb=status_cb)
    vp = _ensure_venv(rt, status_cb=status_cb)

    # Ultra-fast file cache path
    try:
        qc = _fcache_get(rt)
        if qc:
            site_s = (qc.get("site") or "").strip()
            rt_s = (qc.get("rt_dir") or "").strip()
            req_ta = bool(qc.get("require_torchaudio", False))
            tag = (qc.get("tag") or "").strip()
            if (
                tag == rt.name
                and site_s and Path(site_s).exists()
                and rt_s and Path(rt_s).exists()
                and (req_ta == require_torchaudio)
            ):
                status_cb("[RT] File cache hit; attempting zero-subprocess import.")
                if site_s not in sys.path:
                    sys.path.insert(0, site_s)
                _demote_shadow_torch_paths(status_cb=status_cb)
                _purge_bad_torch_from_sysmodules(status_cb=status_cb)
                import torch
                import torchvision
                if require_torchaudio:
                    import torchaudio
                _TORCH_CACHED = torch
                return torch
    except Exception as e:
        status_cb(f"[RT] File-cache fast-path failed: {type(e).__name__}: {e}. Continuing…")
        _fcache_clear(rt)

    site = _site_packages(vp)
    marker = rt / "torch_installed.json"
    venv_ver = _venv_pyver(vp)

    _rt_dbg(f"venv_ver={venv_ver}", status_cb)
    _rt_dbg(f"rt={rt}", status_cb)
    _rt_dbg(f"venv_python={vp}", status_cb)
    _rt_dbg(f"marker={marker} exists={marker.exists()}", status_cb)
    _rt_dbg(f"site={site}", status_cb)

    try:
        _ensure_numpy(vp, status_cb=status_cb)
    except Exception:
        pass

    # Marker fast path
    try:
        if _marker_says_ready(marker, site, venv_ver, require_torchaudio=require_torchaudio):
            status_cb("[RT] Marker valid; attempting fast in-process import.")
            sp = str(site)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            _demote_shadow_torch_paths(status_cb=status_cb)
            _purge_bad_torch_from_sysmodules(status_cb=status_cb)
            import torch
            import torchvision
            if require_torchaudio:
                import torchaudio
            try:
                _write_torch_marker(marker, status_cb=status_cb)
            except Exception:
                pass
            _TORCH_CACHED = torch
            _write_cache_best_effort(rt, site, venv_ver)
            return torch
    except Exception as e:
        status_cb(f"[RT] Marker fast-path failed: {type(e).__name__}: {e}. Falling back…")
        try:
            _fcache_clear(rt)
        except Exception:
            pass

    # Definitive venv probe
    ok_all, info = _venv_has_torch_stack(vp, status_cb=status_cb, require_torchaudio=require_torchaudio)
    status_cb(
        "[RT] venv probe: "
        f"torch={info['torch'][0]} "
        f"torchvision={info['torchvision'][0]} "
        f"torchaudio={info['torchaudio'][0]}"
    )

    if not ok_all:
        if not allow_install:
            _raise_missing_runtime(info)

        status_cb("[RT] Runtime torch stack missing/incomplete; explicit install allowed. Starting…")
        try:
            _fcache_clear(rt)
        except Exception:
            pass

        with _install_lock(rt):
            ok_all_2, _ = _venv_has_torch_stack(vp, status_cb=status_cb, require_torchaudio=require_torchaudio)
            if not ok_all_2:
                status_cb("[RT] Installing torch stack into runtime venv…")
                _install_torch(
                    vp,
                    prefer_cuda=prefer_cuda,
                    prefer_xpu=prefer_xpu,
                    prefer_dml=prefer_dml,
                    prefer_rocm=prefer_rocm,
                    status_cb=status_cb,
                )

            ok_all_3, info_3 = _venv_has_torch_stack(vp, status_cb=status_cb, require_torchaudio=require_torchaudio)
            status_cb(
                "[RT] post-install probe: "
                f"torch={info_3['torch'][0]} "
                f"torchvision={info_3['torchvision'][0]} "
                f"torchaudio={info_3['torchaudio'][0]}"
            )

            if not ok_all_3:
                missing = [k for k in ("torch", "torchvision") if not info_3[k][0]]
                if require_torchaudio and not info_3["torchaudio"][0]:
                    missing.append("torchaudio")
                msg = (
                    "Hardware acceleration install completed, but runtime is still incomplete.\n\n"
                    f"Missing packages in runtime venv: {', '.join(missing)}\n\n"
                    "Please delete the SASpro runtime folder and try Install/Repair again."
                )
                status_cb(f"[RT] {msg}")
                raise RuntimeError(msg)

            try:
                _write_torch_marker(marker, status_cb=status_cb)
            except Exception:
                pass

        # After a fresh install, skip the in-process import entirely.
        # The first in-process import after a cold install can fail on some cp314
        # builds due to torch.hub side-effect imports that don't work in this
        # process context. The marker is written and the next launch will use
        # the fast marker path which works cleanly.
        status_cb(
            "[RT] Hardware Acceleration installed successfully. "
            "Please restart SASpro to activate GPU acceleration."
        )
        raise RuntimeError(
            "Hardware Acceleration installed successfully.\n\n"
            "Please restart SASpro to activate GPU acceleration.\n\n"
            "This is expected after a fresh installation — your settings have been saved."
        )

    else:
        try:
            _write_torch_marker(marker, status_cb=status_cb)
        except Exception:
            pass

    # Final in-process import (only reached when torch was already installed
    # before this call — not after a fresh install)
    sp = str(site)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    _demote_shadow_torch_paths(status_cb=status_cb)
    _purge_bad_torch_from_sysmodules(status_cb=status_cb)

    try:
        import torch
        import torchvision
        if require_torchaudio:
            import torchaudio
        _torch_sanity_check(status_cb=status_cb)
        _TORCH_CACHED = torch
        _write_cache_best_effort(rt, site, venv_ver)
        return torch
    except Exception as e:
        try:
            _fcache_clear(rt)
        except Exception:
            pass
        ok_all_final, info_final = _venv_has_torch_stack(vp, status_cb=status_cb, require_torchaudio=require_torchaudio)
        msg = "\n".join([f"{k}: ok={ok} :: {out}" for k, (ok, out) in info_final.items()])
        raise RuntimeError(
            "Runtime venv probe says torch stack exists, but in-process import failed.\n"
            "This typically indicates a frozen-stdlib / PyInstaller packaging issue, "
            "shadowed torch import, or broken wheel.\n\n"
            f"Original error: {type(e).__name__}: {e}\n\n"
            f"Probe ok_all={ok_all_final}\n"
            "Runtime venv probe:\n" + msg
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# System Python discovery
# ──────────────────────────────────────────────────────────────────────────────

def _find_system_python_cmd_for_minor(minor: int) -> list[str] | None:
    """Find a system Python 3.minor command. Returns argv list or None."""
    sysname = platform.system()

    def _is_target(cmd: list[str]) -> bool:
        try:
            r = subprocess.run(
                cmd + ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **_safe_text_kwargs(),
            )
            return r.returncode == 0 and (r.stdout or "").strip() == f"3.{minor}"
        except Exception:
            return False

    if sysname == "Windows":
        for c in (["py", f"-3.{minor}"], [f"python3.{minor}"], ["python"]):
            if _is_target(c):
                return c
        p = shutil.which(f"python3.{minor}")
        if p and _is_target([p]):
            return [p]
        return None

    if sysname == "Darwin":
        for path in [
            f"/opt/homebrew/bin/python3.{minor}",
            f"/usr/local/bin/python3.{minor}",
            f"/usr/bin/python3.{minor}",
        ]:
            if os.path.exists(path) and os.access(path, os.X_OK) and _is_target([path]):
                return [path]
        p = shutil.which(f"python3.{minor}")
        if p and _is_target([p]):
            return [p]
        return None

    # Linux
    for name in [f"python3.{minor}", "python3", "python"]:
        p = shutil.which(name)
        if p and _is_target([p]):
            return [p]
    for path in [
        f"/usr/bin/python3.{minor}",
        f"/usr/local/bin/python3.{minor}",
        f"/opt/python3.{minor}/bin/python3.{minor}",
    ]:
        if os.path.isfile(path) and _is_target([path]):
            return [path]
    return None


def _find_system_python_cmd() -> list[str]:
    """
    Find the best available system Python for creating a SASpro runtime venv.
    Preference order: 3.12, 3.13, 3.14.
    """
    # When frozen (PyInstaller), sys.executable is the .exe — never use it as Python.
    # Always search for a real system Python in the frozen case.
    if not getattr(sys, "frozen", False):
        maj, min_ = sys.version_info.major, sys.version_info.minor
        if maj == 3 and min_ in _SUPPORTED_PY_MINORS:
            return [sys.executable]

    for minor in _SUPPORTED_PY_MINORS:
        cmd = _find_system_python_cmd_for_minor(minor)
        if cmd:
            return cmd

    raise RuntimeError(
        "Could not find Python 3.12, 3.13, or 3.14 to create the SASpro runtime venv.\n"
        "Install one of these Python versions and relaunch SAS Pro.\n\n"
        "Windows:  install from python.org and ensure 'py -3.12' (or -3.13/-3.14) works\n"
        "macOS:    brew install python@3.12\n"
        "Linux:    sudo apt install python3.12  (or your distro equivalent)"
    )


def add_runtime_to_sys_path(status_cb=print) -> None:
    rt = _user_runtime_dir(status_cb=status_cb)
    p = _venv_paths(rt)
    vpy = p["python"]
    if not vpy.exists():
        return

    # Verify version match before injecting — prevents stale py312 polluting py314
    ver = _venv_pyver(vpy)
    if ver:
        expected_minor = None
        try:
            expected_minor = int(rt.name[2:]) % 100  # "py314" -> 314 -> 14
        except Exception:
            pass

        if expected_minor is not None and ver[1] != expected_minor:
            _rt_dbg(
                f"add_runtime_to_sys_path: skipping {rt} — venv is Python "
                f"{ver[0]}.{ver[1]} but folder tag implies 3.{expected_minor}",
                status_cb,
            )
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
        for c in (site, site.parent / "site-packages", site.parent / "dist-packages"):
            sc = str(c)
            if c.exists() and sc not in sys.path:
                sys.path.insert(0, sc)
        _demote_shadow_torch_paths(status_cb=status_cb)
    except Exception:
        return


def prewarm_torch_cache(
    status_cb=print,
    *,
    require_torchaudio: bool = False,
    ensure_venv: bool = True,
    ensure_numpy: bool = False,
    validate_marker: bool = True,
) -> None:
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

        if validate_marker:
            if not _marker_says_ready(marker, site, venv_ver, require_torchaudio=require_torchaudio):
                status_cb("[RT] prewarm: marker not valid; skipping cache write.")
                return

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
