# pro/numba_warmup.py
"""
Background Numba JIT warmup module.

Triggers compilation of frequently-used Numba functions in a background thread
after the main UI has loaded. This ensures the first actual use of these
functions is fast, while not blocking the application startup.
"""
from __future__ import annotations
import threading
import numpy as np
import logging

_warmup_thread: threading.Thread | None = None
_warmup_done = threading.Event()

def _do_warmup():
    """
    Compile critical numba functions with small dummy arrays.
    This runs in a background thread after the UI is visible.
    """
    try:
        # Import numba functions
        from numba_utils import (
            blend_add_numba,
            blend_subtract_numba,
            blend_multiply_numba,
            rescale_image_numba,
            flip_horizontal_numba,
            flip_vertical_numba,
            invert_image_numba,
            apply_flat_division_numba,
        )
        
        # Create small dummy arrays for warmup (32x32 RGB)
        dummy_rgb = np.random.rand(32, 32, 3).astype(np.float32)
        dummy_mono = np.random.rand(32, 32).astype(np.float32)
        
        # Trigger JIT compilation with minimal data
        try:
            _ = blend_add_numba(dummy_rgb, dummy_rgb, 0.5)
        except Exception:
            pass
            
        try:
            _ = blend_subtract_numba(dummy_rgb, dummy_rgb, 0.5)
        except Exception:
            pass
            
        try:
            _ = blend_multiply_numba(dummy_rgb, dummy_rgb, 0.5)
        except Exception:
            pass
            
        try:
            _ = rescale_image_numba(dummy_rgb, 0.5)
        except Exception:
            pass
            
        try:
            _ = flip_horizontal_numba(dummy_rgb)
        except Exception:
            pass
            
        try:
            _ = flip_vertical_numba(dummy_rgb)
        except Exception:
            pass
            
        try:
            _ = invert_image_numba(dummy_rgb)
        except Exception:
            pass
            
        try:
            _ = apply_flat_division_numba(dummy_rgb, dummy_rgb)
        except Exception:
            pass
        
        logging.debug("Numba warmup completed successfully")
        
    except Exception as e:
        logging.debug(f"Numba warmup failed (non-critical): {e}")
    finally:
        _warmup_done.set()


def start_background_warmup():
    """
    Start the Numba warmup in a background thread.
    Call this after the main window is visible.
    """
    global _warmup_thread
    
    if _warmup_thread is not None and _warmup_thread.is_alive():
        return  # Already running
    
    _warmup_thread = threading.Thread(target=_do_warmup, daemon=True, name="NumbaWarmup")
    _warmup_thread.start()


def wait_for_warmup(timeout: float = 5.0) -> bool:
    """
    Wait for the warmup to complete.
    
    Args:
        timeout: Maximum time to wait in seconds.
        
    Returns:
        True if warmup completed, False if timeout.
    """
    return _warmup_done.wait(timeout=timeout)


def is_warmup_done() -> bool:
    """Check if warmup has completed."""
    return _warmup_done.is_set()
