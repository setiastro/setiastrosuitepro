from __future__ import annotations
import os
import sys
from pathlib import Path


def _read_pyproject_version(start: Path) -> str | None:
    """
    Walk upward from 'start' looking for pyproject.toml,
    return [tool.poetry].version if found.
    """
    try:
        import tomllib as _toml  # Python 3.11+
    except Exception:
        try:
            import tomli as _toml  # Python 3.10
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
      1) pyproject.toml (developer/source checkouts)
      2) installed distribution metadata (pip-installed wheel/sdist)
      3) safe fallback
    """
    # 1) Source tree pyproject.toml (walk from this file upward)
    here = Path(__file__).resolve()
    v_src = _read_pyproject_version(here.parent)
    if v_src:
        return v_src

    # 2) Installed package metadata
    try:
        from importlib.metadata import version as _dist_version
        v = _dist_version(dist_name)
        if v and v != "0.1.0":
            return v
    except Exception:
        pass

    # 3) Frozen / unknown: fallback
    return "0.0.0"
