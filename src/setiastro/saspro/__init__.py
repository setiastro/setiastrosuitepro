"""
Seti Astro Suite Pro - Main Package

This package provides advanced astrophotography tools for image calibration,
stacking, registration, photometry, and visualization.

Important:
- __init__.py must remain import-safe (no UI side effects, no splash, no QApplication).
- Do NOT import setiastro.saspro.__main__ from here.
"""

__all__ = []

# Re-export commonly used items for convenience (safe imports only)
try:
    from .doc_manager import DocManager, ImageDocument
    from .subwindow import ImageSubWindow
    __all__ = ["DocManager", "ImageDocument", "ImageSubWindow"]
except Exception:
    # During initial setup or partial installs, some modules may not be available
    __all__ = []
