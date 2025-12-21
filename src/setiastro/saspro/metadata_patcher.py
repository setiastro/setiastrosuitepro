"""
Centralized importlib.metadata patching for PyInstaller frozen builds.

This module provides safe fallback implementations for metadata functions
that may fail in frozen (PyInstaller) environments.
"""
import sys
import importlib


def apply_metadata_patches():
    """
    Apply safe fallback patches to importlib.metadata for frozen builds.
    
    This function should be called early in the application startup, before
    any libraries that depend on importlib.metadata are imported.
    """
    if not getattr(sys, "frozen", False):
        # Not a frozen build, no patching needed
        return

    # 1) Attempt to import both metadata modules
    try:
        std_md = importlib.import_module('importlib.metadata')
    except ImportError:
        std_md = None

    try:
        back_md = importlib.import_module('importlib_metadata')
    except ImportError:
        back_md = None

    # 2) Ensure that any "import importlib.metadata" or
    #    "import importlib_metadata" picks up our loaded module
    if std_md:
        sys.modules['importlib.metadata'] = std_md
        setattr(importlib, 'metadata', std_md)
    if back_md:
        sys.modules['importlib_metadata'] = back_md

    # 3) Pick whichever is available for defaults (prefer stdlib)
    meta = std_md or back_md
    if not meta:
        # nothing to patch
        return

    # 4) Save originals
    orig_version = getattr(meta, 'version', None)
    orig_distribution = getattr(meta, 'distribution', None)

    # 5) Define safe fallbacks
    def safe_version(pkg, *args, **kwargs):
        try:
            return orig_version(pkg, *args, **kwargs)
        except Exception:
            return "0.0.0"

    class DummyDist:
        version = "0.0.0"
        metadata = {}

    def safe_distribution(pkg, *args, **kwargs):
        try:
            return orig_distribution(pkg, *args, **kwargs)
        except Exception:
            return DummyDist()

    # 6) Patch both modules (stdlib and back-port) if they exist
    for m in (std_md, back_md):
        if not m:
            continue
        if orig_version:
            m.version = safe_version
        if orig_distribution:
            m.distribution = safe_distribution
