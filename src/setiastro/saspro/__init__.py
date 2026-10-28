"""
Seti Astro Suite Pro - Main Package

This package provides advanced astrophotography tools for image calibration,
stacking, registration, photometry, and visualization.

Important:
- __init__.py must remain import-safe (no UI side effects, no splash, no QApplication).
- Do NOT import setiastro.saspro.__main__ at import time.
"""

__all__ = []

# Re-export commonly used items for convenience (safe imports only)
# ── Lazy re-exports (PEP 562) ──────────────────────────────────────────────
# DocManager / ImageDocument / ImageSubWindow stay importable exactly as
# before -- `from setiastro.saspro import DocManager` still works and the
# class objects are identical to the ones in their defining modules -- but the
# modules behind them are not loaded until a name is actually touched.
#
# They used to be imported eagerly here inside a bare `try/except Exception:
# pass`. Every CLI entry point goes through `import setiastro.saspro.<x>`, and
# Python must execute this file before it can reach any submodule, so the GUI
# stack loaded before argparse had even looked at argv: doc_manager pulls
# QtWidgets/astropy/swap_manager/xisf, subwindow pulls QtGui, matplotlib's
# qtagg backend, layers, autostretch, help_support -> shortcuts -> the command
# registry, imageops.*, and each of those pulls more again. Measured in this
# tree: `solve --help` was 3785 ms and 1570 modules.
#
# The silent `except` mattered too: doc_manager was the FIRST line in the try,
# so anything raising inside it skipped the subwindow import as well -- which
# modules actually loaded depended on where it happened to fail, with no log
# line. __getattr__ below reports the real cause instead, as AttributeError so
# hasattr() and getattr(..., default) behave as they always did.
#
# Keep this file free of module-scope imports of anything that is not stdlib.
# tools/check_cli_imports.py enforces it; tools/import_audit.py measures it.

_LAZY = {
    "DocManager":     ".doc_manager",
    "ImageDocument":  ".doc_manager",
    "ImageSubWindow": ".subwindow",
}
__all__ += list(_LAZY)


def __getattr__(name: str):
    """Import the GUI modules on first attribute access (PEP 562)."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    try:
        value = getattr(importlib.import_module(mod, __name__), name)
    except Exception as exc:
        raise AttributeError(
            f"{__name__}.{name} is unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    globals()[name] = value      # cache; __getattr__ is not consulted again
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
def main():
    """
    Console entrypoint shim.

    IMPORTANT: This must stay import-safe. We import the real entrypoint
    lazily only when the command is executed.
    """
    from .__main__ import main as _main
    return _main()

__all__ += ["main"]
