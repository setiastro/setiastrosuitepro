#saspro/numba_warmup.py
"""
Background Numba JIT warmup module.

Triggers compilation of frequently-used Numba functions in a background thread
after the main UI has loaded. This ensures the first actual use of these
functions is fast, while not blocking the application startup.

NOTE: This module no longer influences numba's threading-layer selection.
That decision now happens deterministically on the main thread very early
in startup (see numba_bootstrap._force_layer_resolution_on_main_thread, or
the equivalent block in gui_entry.py right after numba_bootstrap imports).
Resolving the layer on the main thread, before any Qt threads exist, is the
only way to prevent the "sometimes TBB, sometimes workqueue" race — numba's
layer selector isn't threadsafe for first-compile races, and whichever
thread (warmup, QThreadPool worker, auto-opened-doc path) hits numba first
decides the layer, non-deterministically.
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
        # Diagnostic: did the main-thread probe in numba_bootstrap already
        # resolve the layer before we got here? If yes, we're hitting a cached
        # decision and the race is impossible. If no, we're the first to touch
        # numba in this process and the resolution happens on THIS thread —
        # which is exactly the race the main-thread probe was supposed to
        # prevent, so flag it loudly.
        try:
            import numba
            try:
                _preresolved = numba.threading_layer()
                logging.info(
                    "[warmup] numba layer was pre-resolved by main thread: %s",
                    _preresolved,
                )
            except (RuntimeError, ValueError):
                logging.warning(
                    "[warmup] numba layer NOT pre-resolved — warmup thread "
                    "will decide. Confirm that numba_bootstrap runs the "
                    "main-thread layer probe before any other numba import."
                )
        except Exception:
            pass

        # NOTE: We used to force NUMBA_THREADING_LAYER=workqueue on Windows
        # here, as a belt-and-suspenders guard against the CPython 3.14 TBB
        # access violation. That guard is now the wrong tool: it killed OMP
        # (which IS threadsafe and doesn't have that bug) and locked Windows
        # users onto a non-threadsafe layer that forced NUMBA_PARALLEL_LOCK
        # to serialize every parallel kernel. Priority steering in
        # numba_bootstrap handles the TBB-avoidance correctly without the
        # OMP collateral damage.

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
            # Geometric transforms
            rescale_image_numba,
            flip_horizontal_numba,
            flip_vertical_numba,
            rotate_180_numba,
            bin2x2_numba,
            # Image processing
            invert_image_numba,
            apply_flat_division_numba,
            subtract_dark,
            # Statistical operations
            kappa_sigma_clip_weighted,
            windsorized_sigma_clip_weighted,
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
            (subtract_dark, [dummy_3d_mono, dummy_mono]),
            # Statistical operations (with multiple frames)
            (kappa_sigma_clip_weighted, [dummy_3d_mono, np.ones(8)]),
            (windsorized_sigma_clip_weighted, [dummy_3d_mono, np.ones(8)]),
        ]

        for func, args in warmup_funcs:
            try:
                _ = func(*args)
            except Exception:
                # Silently skip functions that may not be available or fail
                pass

        logging.debug("Numba warmup completed successfully")

        # Log the resolved numba threading layer. If the main-thread probe
        # ran first (as it should), this will match what was logged there.
        # If a workqueue result shows up here on a machine where TBB is
        # installed, it's the signal that something imported numba before
        # the main-thread probe got to run — check startup order.
        try:
            import numba
            _layer = numba.threading_layer()
            logging.info("Numba threading layer (as seen from warmup): %s", _layer)
            # Pin the resolved layer into numba_utils so the lazy
            # NUMBA_PARALLEL_LOCK decides correctly even if the main-thread
            # probe failed and this warmup is where resolution actually
            # happened. The record_numba_threading_layer call is idempotent,
            # so no harm if the probe already set it.
            try:
                from setiastro.saspro.legacy import numba_utils as _nu
                _nu.record_numba_threading_layer(str(_layer))
            except Exception:
                pass
            if _layer not in ("tbb", "omp"):
                logging.warning(
                    "Numba resolved to '%s' (NOT threadsafe). "
                    "NUMBA_PARALLEL_LOCK will serialize parallel kernels to "
                    "stay safe, which costs ~2-3× in debayer/stretch "
                    "throughput. If TBB is installed on this machine and "
                    "you're still seeing workqueue, the main-thread layer "
                    "probe may have been pre-empted by a concurrent numba "
                    "import — check that numba_bootstrap runs it before any "
                    "QThread, QThreadPool worker, or file-open sidecar can "
                    "touch numba.",
                    _layer,
                )
        except Exception as _e:
            logging.info("Numba threading-layer probe skipped: %r", _e)

        # serialized-warmup: astroalign folded into _do_warmup
        # Run the astroalign JIT warmup in THIS thread, sequentially.
        # Triggering the first parallel=True compile from two threads at
        # once can access-violate in numba's compiler_lock on some
        # Windows/TBB configs, so we do it single-threaded here.
        try:
            from setiastro.saspro.astroalign import warmup_jit as _aa_warmup_jit
            _aa_warmup_jit()
        except Exception:
            pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.warning("Numba warmup failed (non-critical): %r", e)
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