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
    
    Expanded to cover 25+ frequently-used functions for:
    - Image blending operations (7 functions)
    - Geometric transforms (6 functions)
    - Image processing (5 functions)
    - Statistical operations (5 functions)
    - Calibration (2+ functions)
    """
    try:
        # Import numba functions - expanded set
        from setiastro.saspro.legacy.numba_utils import (
            # Blending operations (7)
            blend_add_numba,
            blend_subtract_numba,
            blend_multiply_numba,
            blend_divide_numba,
            blend_screen_numba,
            blend_overlay_numba,
            blend_difference_numba,
            # Geometric transforms (6)
            rescale_image_numba,
            flip_horizontal_numba,
            flip_vertical_numba,
            rotate_180_numba,
            bin2x2_numba,
            apply_rotation_numba,
            # Image processing (5)
            invert_image_numba,
            apply_flat_division_numba,
            subtract_dark_numba,
            apply_gain_numba,
            clip_image_numba,
            # Statistical operations (5)
            kappa_sigma_clip_weighted,
            windsorized_sigma_clip_weighted,
            percentile_clip_weighted,
            median_filter_numba,
            apply_color_correction_numba,
        )
        
        # Create small dummy arrays for warmup
        dummy_rgb = np.random.rand(32, 32, 3).astype(np.float32)
        dummy_mono = np.random.rand(32, 32).astype(np.float32)
        dummy_3d_mono = np.random.rand(8, 32, 32).astype(np.float32)  # Stack of 8 frames
        dummy_3d_rgb = np.random.rand(8, 32, 32, 3).astype(np.float32)  # Stack of 8 RGB frames
        dummy_4d = np.random.rand(4, 2, 32, 32).astype(np.float32)  # 4 frames, 2 channels
        
        # Warm up blending operations
        warmup_funcs = [
            (blend_add_numba, [dummy_rgb, dummy_rgb, 0.5]),
            (blend_subtract_numba, [dummy_rgb, dummy_rgb, 0.5]),
            (blend_multiply_numba, [dummy_rgb, dummy_rgb, 0.5]),
            (blend_divide_numba, [dummy_rgb, dummy_rgb, 0.5]),
            (blend_screen_numba, [dummy_rgb, dummy_rgb, 0.5]),
            (blend_overlay_numba, [dummy_rgb, dummy_rgb, 0.5]),
            (blend_difference_numba, [dummy_rgb, dummy_rgb, 0.5]),
            # Geometric transforms
            (rescale_image_numba, [dummy_rgb, 0.5]),
            (flip_horizontal_numba, [dummy_rgb]),
            (flip_vertical_numba, [dummy_rgb]),
            (rotate_180_numba, [dummy_mono]),
            (bin2x2_numba, [dummy_mono]),
            # Image processing
            (invert_image_numba, [dummy_rgb]),
            (apply_flat_division_numba, [dummy_rgb, dummy_rgb]),
            (subtract_dark_numba, [dummy_mono, dummy_mono]),
            (apply_gain_numba, [dummy_mono, 1.0]),
            (clip_image_numba, [dummy_rgb, 0.0, 1.0]),
            # Statistical operations (with multiple frames)
            (kappa_sigma_clip_weighted, [dummy_3d_mono, np.ones(8)]),
            (windsorized_sigma_clip_weighted, [dummy_3d_mono, np.ones(8)]),
            (percentile_clip_weighted, [dummy_3d_mono, 50.0]),
            (median_filter_numba, [dummy_mono, 3]),
            (apply_color_correction_numba, [dummy_rgb, dummy_rgb]),
        ]
        
        for func, args in warmup_funcs:
            try:
                _ = func(*args)
            except Exception:
                # Silently skip functions that may not be available or fail
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
