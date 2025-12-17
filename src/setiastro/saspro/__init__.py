"""
Seti Astro Suite Pro - Main Package

This package provides advanced astrophotography tools for image calibration,
stacking, registration, photometry, and visualization.
"""

# Re-export commonly used items for convenience
try:
    from .doc_manager import DocManager, ImageDocument
    from .subwindow import ImageSubWindow
    __all__ = ["DocManager", "ImageDocument", "ImageSubWindow"]
except ImportError:
    # During initial setup, some modules may not be available
    __all__ = []

