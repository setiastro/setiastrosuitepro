from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from importlib import metadata as importlib_metadata
import os
import platform
from pathlib import Path
import sys
from typing import Any

from setiastro.saspro.accel_installer import _has_amd_rocm, _has_intel_arc, _has_nvidia
from setiastro.saspro.runtime_torch import (
    _SUPPORTED_PY_MINORS,
    _find_system_python_cmd_for_minor,
    _runtime_base_dir,
    _user_runtime_dir,
    _venv_paths,
    _venv_pyver,
    add_runtime_to_sys_path,
    is_supported_runtime_python,
    supported_python_version_strings,
    supported_python_versions_text,
)
from setiastro.saspro.versioning import get_app_version


SUPPORT_CRITICAL_PACKAGES = ("numpy", "PyQt6", "psutil", "onnxruntime", "torch")


@dataclass
class DiagnosticsReport:
    generated_at_utc: str
    data: dict[str, Any]
    hints: list[str]
    markdown: str


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


def _app_support_root() -> Path:
    return _runtime_base_dir().parent


def default_report_dir() -> Path:
    return _app_support_root() / "support" / "reports"


def default_report_path(now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return default_report_dir() / f"saspro-diagnostics-{stamp}.md"


def write_report(markdown: str, output_path: str | os.PathLike[str] | None = None) -> Path:
    path = Path(output_path) if output_path else default_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def collect_diagnostics() -> DiagnosticsReport:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    data = {
        "app": _collect_app_info(),
        "system": _collect_system_info(),
        "python": _collect_python_info(),
        "environment": _collect_environment_info(),
        "runtime": _collect_runtime_info(),
        "backend": _collect_backend_info(),
        "dependencies": _collect_dependencies_info(),
        "logs": _collect_log_info(),
    }
    hints = _build_hints(data)
    markdown = _render_markdown(generated_at, data, hints)
    return DiagnosticsReport(
        generated_at_utc=generated_at,
        data=data,
        hints=hints,
        markdown=markdown,
    )


def _collect_app_info() -> dict[str, Any]:
    return {
        "version": get_app_version("setiastrosuitepro"),
        "mode": "packaged" if _is_frozen() else "dev",
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }


def _collect_system_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "platform": platform.platform(),
    }


def _collect_python_info() -> dict[str, Any]:
    version_tuple = (sys.version_info.major, sys.version_info.minor)
    supported_found: dict[str, str] = {}
    for minor in _SUPPORTED_PY_MINORS:
        cmd = _find_system_python_cmd_for_minor(minor)
        if cmd:
            supported_found[f"3.{minor}"] = " ".join(cmd)
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "full_version": sys.version.replace("\n", " "),
        "supported_runtime_versions": supported_python_version_strings(),
        "runtime_python_supported": is_supported_runtime_python(version_tuple),
        "supported_system_pythons": supported_found,
    }


def _summarize_path_var(raw: str, runtime_python: Path | None) -> dict[str, Any]:
    entries = [entry for entry in raw.split(os.pathsep) if entry]
    runtime_dir = str(runtime_python.parent.resolve()) if runtime_python else None
    basenames: list[str] = []
    for entry in entries[:8]:
        name = Path(entry).name or Path(entry).anchor or entry
        if name not in basenames:
            basenames.append(name)
    return {
        "entry_count": len(entries),
        "runtime_python_dir_in_path": bool(runtime_dir and runtime_dir in entries),
        "sample_entries": basenames,
    }


def _collect_environment_info() -> dict[str, Any]:
    runtime_python = _venv_paths(_user_runtime_dir(status_cb=lambda *_: None))["python"]
    env = os.environ
    return {
        "SASPRO_RUNTIME_DIR": env.get("SASPRO_RUNTIME_DIR"),
        "PYTHONPATH_set": bool(env.get("PYTHONPATH")),
        "QT_QPA_PLATFORM": env.get("QT_QPA_PLATFORM"),
        "PATH_summary": _summarize_path_var(env.get("PATH", ""), runtime_python),
    }


def _collect_runtime_info() -> dict[str, Any]:
    base_dir = _runtime_base_dir()
    active_dir = _user_runtime_dir(status_cb=lambda *_: None)
    venv_paths = _venv_paths(active_dir)
    runtime_python = venv_paths["python"]
    runtime_version = _venv_pyver(runtime_python) if runtime_python.exists() else None
    expected_tag = None
    if sys.version_info.major == 3 and sys.version_info.minor in _SUPPORTED_PY_MINORS:
        expected_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    available: list[dict[str, Any]] = []
    if base_dir.exists():
        for child in sorted(base_dir.iterdir()):
            if not child.is_dir() or not child.name.startswith("py"):
                continue
            child_python = _venv_paths(child)["python"]
            available.append(
                {
                    "tag": child.name,
                    "path": str(child),
                    "venv_python_exists": child_python.exists(),
                    "python_version": _venv_pyver(child_python) if child_python.exists() else None,
                }
            )
    return {
        "base_dir": str(base_dir),
        "active_dir": str(active_dir),
        "active_tag": active_dir.name,
        "expected_tag": expected_tag,
        "venv_exists": venv_paths["venv"].exists(),
        "runtime_python": str(runtime_python),
        "runtime_python_exists": runtime_python.exists(),
        "runtime_python_version": runtime_version,
        "available_runtimes": available,
    }
def _probe_torch() -> dict[str, Any]:
    info: dict[str, Any] = {
        "import_ok": False,
        "version": None,
        "file": None,
        "backend": "Not installed",
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device": None,
        "xpu_available": False,
        "xpu_device": None,
        "mps_available": False,
        "rocm_version": None,
        "directml_available": False,
        "import_error": None,
    }
    try:
        add_runtime_to_sys_path(status_cb=lambda *_: None)
        torch = importlib.import_module("torch")
        info["import_ok"] = True
        info["version"] = getattr(torch, "__version__", None)
        info["file"] = getattr(torch, "__file__", None)
        info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        info["rocm_version"] = getattr(getattr(torch, "version", None), "hip", None)
        info["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        info["xpu_available"] = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        info["mps_available"] = bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        )
        if info["cuda_available"]:
            if info["rocm_version"]:
                try:
                    info["cuda_device"] = torch.cuda.get_device_name(0)
                except Exception:
                    info["cuda_device"] = "AMD GPU"
                info["backend"] = f"ROCm ({info['cuda_device']}, HIP {info['rocm_version']})"
            else:
                try:
                    info["cuda_device"] = torch.cuda.get_device_name(0)
                except Exception:
                    info["cuda_device"] = "CUDA"
                info["backend"] = f"CUDA ({info['cuda_device']})"
        elif info["xpu_available"]:
            try:
                if hasattr(torch.xpu, "get_device_name"):
                    info["xpu_device"] = torch.xpu.get_device_name(0)
            except Exception:
                info["xpu_device"] = None
            info["backend"] = f"Intel XPU ({info['xpu_device']})" if info["xpu_device"] else "Intel XPU"
        elif info["mps_available"]:
            info["backend"] = "Apple MPS"
        elif platform.system() == "Windows":
            try:
                importlib.import_module("torch_directml")
                info["directml_available"] = True
                info["backend"] = "DirectML"
            except Exception:
                info["directml_available"] = False
                info["backend"] = "CPU"
        else:
            info["backend"] = "CPU"
    except Exception as exc:
        info["import_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _collect_backend_info() -> dict[str, Any]:
    has_amd, amd_arch = _has_amd_rocm()
    torch_info = _probe_torch()
    return {
        "detected_hardware": {
            "nvidia": _has_nvidia(),
            "intel_arc": _has_intel_arc(),
            "amd_rocm": has_amd,
            "amd_rocm_arch": amd_arch or None,
        },
        "torch": torch_info,
    }


def _collect_dependencies_info() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in SUPPORT_CRITICAL_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None)
            if version is None:
                try:
                    version = importlib_metadata.version(name)
                except Exception:
                    version = None
            out[name] = {
                "present": True,
                "version": version,
            }
        except Exception as exc:
            out[name] = {
                "present": False,
                "version": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return out


def _default_log_dir() -> Path:
    if _is_frozen():
        if sys.platform.startswith("win"):
            return Path(os.path.expandvars("%APPDATA%")) / "SetiAstroSuitePro" / "logs"
        if sys.platform.startswith("darwin"):
            return Path.home() / "Library" / "Logs" / "SetiAstroSuitePro"
        return Path.home() / ".local" / "share" / "SetiAstroSuitePro" / "logs"
    return Path.cwd() / "logs"


def _fallback_logger_dir() -> Path:
    return Path.home() / ".setiastrosuite" / "logs"


def _collect_log_info() -> dict[str, Any]:
    primary_dir = _default_log_dir()
    fallback_dir = _fallback_logger_dir()
    crash_log = primary_dir / "saspro_crash.log"
    app_log = primary_dir / "setiastrosuite.log"
    fallback_log = fallback_dir / "setiastrosuite.log"
    return {
        "primary_log_dir": str(primary_dir),
        "primary_log_dir_exists": primary_dir.exists(),
        "app_log": {"path": str(app_log), "exists": app_log.exists()},
        "fallback_app_log": {"path": str(fallback_log), "exists": fallback_log.exists()},
        "crash_log": {"path": str(crash_log), "exists": crash_log.exists()},
        "report_dir": str(default_report_dir()),
    }


def _build_hints(data: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    py_info = data["python"]
    runtime = data["runtime"]
    backend = data["backend"]
    path_summary = data["environment"]["PATH_summary"]
    torch_info = backend["torch"]
    hardware = backend["detected_hardware"]

    if not py_info["runtime_python_supported"]:
        supported = supported_python_versions_text()
        found = py_info["supported_system_pythons"]
        if found:
            hints.append(
                f"SASpro is running on Python {py_info['version']}, which is outside the supported runtime range ({supported}). "
                f"A supported interpreter is present on this system: {', '.join(sorted(found))}."
            )
        else:
            hints.append(
                f"SASpro is running on Python {py_info['version']}, which is outside the supported runtime range ({supported}), "
                "and no supported system Python was detected."
            )

    expected_tag = runtime["expected_tag"]
    active_tag = runtime["active_tag"]
    runtime_ver = runtime["runtime_python_version"]
    if expected_tag and active_tag != expected_tag:
        hints.append(
            f"Runtime folder mismatch: the active runtime folder is `{active_tag}` but this install expects `{expected_tag}`."
        )
    if runtime["runtime_python_exists"] and runtime_ver:
        version_text = f"{runtime_ver[0]}.{runtime_ver[1]}"
        if active_tag.startswith("py") and active_tag[2:] != f"{runtime_ver[0]}{runtime_ver[1]}":
            hints.append(
                f"Runtime venv mismatch: folder tag `{active_tag}` contains Python {version_text}."
            )
    elif runtime["venv_exists"] and not runtime["runtime_python_exists"]:
        hints.append(
            "The runtime folder exists, but the runtime Python executable is missing. Rebuilding GPU acceleration is likely required."
        )

    if torch_info["import_error"]:
        hints.append(f"PyTorch import failed: {torch_info['import_error']}")
    elif not torch_info["import_ok"]:
        hints.append("PyTorch is not currently importable from the SASpro runtime.")

    if platform.system() == "Windows" and runtime["runtime_python_exists"] and not path_summary["runtime_python_dir_in_path"]:
        hints.append(
            "Windows runtime PATH hint: the SASpro runtime Python directory is not on PATH in this process. "
            "That is usually fine for the report path, but it can explain subprocess/backend launch issues."
        )

    gpu_present = bool(hardware["nvidia"] or hardware["intel_arc"] or hardware["amd_rocm"])
    backend_name = (torch_info["backend"] or "").lower()
    if gpu_present and ("cpu" in backend_name or "not installed" in backend_name):
        hints.append(
            "GPU hardware was detected, but the current PyTorch backend is CPU-only. "
            "That usually means the runtime backend is missing, mismatched, or failed to import."
        )

    return hints


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def _fmt_version_tuple(value: tuple[int, int] | None) -> str:
    if not value:
        return "n/a"
    return f"{value[0]}.{value[1]}"


def _render_markdown(generated_at: str, data: dict[str, Any], hints: list[str]) -> str:
    app = data["app"]
    system = data["system"]
    py_info = data["python"]
    env = data["environment"]
    runtime = data["runtime"]
    backend = data["backend"]
    deps = data["dependencies"]
    logs = data["logs"]
    torch_info = backend["torch"]
    hardware = backend["detected_hardware"]

    lines = [
        "# SASpro Diagnostics Report",
        "",
        f"- Generated (UTC): `{generated_at}`",
        f"- App version: `{app['version']}`",
        f"- Mode: `{app['mode']}`",
        f"- Executable: `{app['executable']}`",
        "",
        "## System",
        "",
        f"- OS: `{system['system']} {system['release']}`",
        f"- Platform: `{system['platform']}`",
        f"- Architecture: `{system['architecture']}`",
        f"- Machine: `{system['machine']}`",
        "",
        "## Python",
        "",
        f"- Python executable: `{py_info['executable']}`",
        f"- Python version: `{py_info['version']}`",
        f"- Supported runtime Python versions: `{supported_python_versions_text()}`",
        f"- Running Python is runtime-supported: `{_fmt_bool(py_info['runtime_python_supported'])}`",
        f"- Supported system Pythons found: `{', '.join(sorted(py_info['supported_system_pythons'])) or 'none detected'}`",
        "",
        "## Environment",
        "",
        f"- `SASPRO_RUNTIME_DIR`: `{env['SASPRO_RUNTIME_DIR'] or 'not set'}`",
        f"- `PYTHONPATH` set: `{_fmt_bool(env['PYTHONPATH_set'])}`",
        f"- `QT_QPA_PLATFORM`: `{env['QT_QPA_PLATFORM'] or 'not set'}`",
        f"- `PATH` entries: `{env['PATH_summary']['entry_count']}`",
        f"- Runtime Python dir present in `PATH`: `{_fmt_bool(env['PATH_summary']['runtime_python_dir_in_path'])}`",
        f"- `PATH` sample: `{', '.join(env['PATH_summary']['sample_entries']) or 'n/a'}`",
        "",
        "## Runtime",
        "",
        f"- Runtime base dir: `{runtime['base_dir']}`",
        f"- Active runtime dir: `{runtime['active_dir']}`",
        f"- Expected runtime tag: `{runtime['expected_tag'] or 'n/a'}`",
        f"- Runtime venv exists: `{_fmt_bool(runtime['venv_exists'])}`",
        f"- Runtime Python: `{runtime['runtime_python']}`",
        f"- Runtime Python exists: `{_fmt_bool(runtime['runtime_python_exists'])}`",
        f"- Runtime Python version: `{_fmt_version_tuple(runtime['runtime_python_version'])}`",
        "",
        "## Backend",
        "",
        f"- GPU hardware detected: `nvidia={_fmt_bool(hardware['nvidia'])}`, `intel_arc={_fmt_bool(hardware['intel_arc'])}`, `amd_rocm={_fmt_bool(hardware['amd_rocm'])}`",
        f"- PyTorch import ok: `{_fmt_bool(torch_info['import_ok'])}`",
        f"- PyTorch version: `{torch_info['version'] or 'n/a'}`",
        f"- PyTorch backend: `{torch_info['backend']}`",
        f"- CUDA available: `{_fmt_bool(torch_info['cuda_available'])}`",
        f"- XPU available: `{_fmt_bool(torch_info['xpu_available'])}`",
        f"- MPS available: `{_fmt_bool(torch_info['mps_available'])}`",
        f"- DirectML available: `{_fmt_bool(torch_info['directml_available'])}`",
    ]
    if torch_info["import_error"]:
        lines.append(f"- PyTorch import error: `{torch_info['import_error']}`")

    lines.extend([
        "",
        "## Dependencies",
        "",
    ])
    for name, dep in deps.items():
        state = dep["version"] if dep["present"] else dep.get("error", "missing")
        lines.append(f"- `{name}`: `{state}`")

    lines.extend([
        "",
        "## Logs",
        "",
        f"- Primary log dir: `{logs['primary_log_dir']}` (exists: `{_fmt_bool(logs['primary_log_dir_exists'])}`)",
        f"- App log: `{logs['app_log']['path']}` (exists: `{_fmt_bool(logs['app_log']['exists'])}`)",
        f"- Fallback app log: `{logs['fallback_app_log']['path']}` (exists: `{_fmt_bool(logs['fallback_app_log']['exists'])}`)",
        f"- Crash log: `{logs['crash_log']['path']}` (exists: `{_fmt_bool(logs['crash_log']['exists'])}`)",
        f"- Default report dir: `{logs['report_dir']}`",
        "",
        "## Likely Issues / Suggested Next Steps",
        "",
    ])
    if hints:
        lines.extend([f"- {hint}" for hint in hints])
    else:
        lines.append("- No obvious support issues were detected from the collected probes.")
    return "\n".join(lines) + "\n"
