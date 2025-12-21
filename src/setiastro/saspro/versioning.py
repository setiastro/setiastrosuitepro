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
      1) installed distribution metadata (best for packaged installs)
      2) pyproject.toml (best for running from source checkout)
      3) safe fallback
    """
    # 1) Installed package metadata (when it matches)
    try:
        from importlib.metadata import version as _dist_version
        v = _dist_version(dist_name)
        # If you want to *avoid* accidentally picking up a stale installed 0.1.0,
        # you can reject that known-bad default:
        if v and v != "0.1.0":
            return v
    except Exception:
        pass

    # 2) Source tree pyproject.toml (walk from this file upward)
    here = Path(__file__).resolve()
    v2 = _read_pyproject_version(here.parent)
    if v2:
        return v2

    # 3) Frozen / unknown: fallback
    return "0.0.0"
