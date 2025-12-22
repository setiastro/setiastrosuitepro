# src/setiastro/saspro/versioning.py
from __future__ import annotations
import os
import sys
from pathlib import Path

def _read_pyproject_version(start: Path) -> str | None:
    """
    Walk upward from 'start' looking for pyproject.toml,
    return [tool.poetry].version if found.
    """
    # Python 3.11+: tomllib; Python 3.10: tomli
    try:
        import tomllib as _toml  # type: ignore
    except Exception:
        try:
            import tomli as _toml  # type: ignore
        except Exception:
            _toml = None

    if _toml is None:
        return None

    cur = start
    for _ in range(8):  # don't walk forever
        pp = cur / "pyproject.toml"
        if pp.exists():
            try:
                data = _toml.loads(pp.read_text(encoding="utf-8"))
                v = (
                    data.get("tool", {})
                        .get("poetry", {})
                        .get("version", None)
                )
                if isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:
                return None
        if cur.parent == cur:
            break
        cur = cur.parent
    return None

def get_app_version(dist_name: str = "setiastrosuitepro") -> str:
    """
    Single source of truth for SASpro version.

    Order:
      0) build_info.py (best for PyInstaller frozen builds)
      1) installed distribution metadata (best for pip installs)
      2) pyproject.toml (best for running from source checkout)
      3) safe fallback
    """
    # 0) build_info (PyInstaller-friendly)
    try:
        from ._generated.build_info import APP_VERSION
        if isinstance(APP_VERSION, str) and APP_VERSION.strip() and APP_VERSION.strip() != "0.0.0":
            return APP_VERSION.strip()
    except Exception:
        pass

    # 1) Installed package metadata
    try:
        from importlib.metadata import version as _dist_version
        v = _dist_version(dist_name)
        if v and v != "0.1.0":
            return v
    except Exception:
        pass

    # 2) Source tree pyproject.toml
    here = Path(__file__).resolve()
    v2 = _read_pyproject_version(here.parent)
    if v2:
        return v2

    return "0.0.0"
