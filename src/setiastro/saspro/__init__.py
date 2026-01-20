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
try:
    from .doc_manager import DocManager, ImageDocument
    from .subwindow import ImageSubWindow
    __all__ += ["DocManager", "ImageDocument", "ImageSubWindow"]
except Exception:
    pass

def main():
    """
    Console entrypoint shim.

    IMPORTANT: This must stay import-safe. We import the real entrypoint
    lazily only when the command is executed.
    """
    from .__main__ import main as _main
    return _main()

__all__ += ["main"]
