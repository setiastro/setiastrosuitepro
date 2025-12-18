"""
Centralized matplotlib configuration bootstrap.

This module provides a single source of truth for setting up matplotlib's
configuration directory, used by both the main application entry point and
the GUI main window.
"""
import os
import sys


def ensure_mpl_config_dir() -> str:
    """
    Make matplotlib use a known, writable folder.

    Frozen (PyInstaller): <folder-with-exe>/mpl_config
    Dev / IDE:            <repo-folder>/mpl_config

    This matches the pre-warm script that will build the font cache there.
    
    Returns:
        str: Path to the matplotlib configuration directory
    """
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        # We're in pro/ directory, go up one level to project root
        base = os.path.dirname(base)

    mpl_cfg = os.path.join(base, "mpl_config")
    try:
        os.makedirs(mpl_cfg, exist_ok=True)
    except OSError:
        # worst case: let matplotlib pick its default
        return mpl_cfg

    # only set if user / env didn't force something else
    os.environ.setdefault("MPLCONFIGDIR", mpl_cfg)
    return mpl_cfg
