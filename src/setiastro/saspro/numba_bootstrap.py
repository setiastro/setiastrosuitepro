# src/setiastro/saspro/numba_bootstrap.py
"""
Steer Numba away from the TBB threading layer BEFORE Numba is ever imported,
then deterministically resolve the layer on the main thread so startup
threading cannot race the selection.

WHY THIS EXISTS
---------------
On CPython 3.14 (Windows) the TBB threading layer aborts the process with a
native access violation during the first parallel JIT compile — the crash the
handoff traced to astroalign.warmup_jit. That abort is a *structured exception*,
not a Python exception, so it CANNOT be caught with try/except: the guard inside
warmup_jit is inert against it. The only reliable defense is to make sure Numba
never selects TBB on the fragile combo.

Numba resolves its threading layer once, at import time, and (with
THREADING_LAYER left at its 'default') picks the first AVAILABLE layer from
THREADING_LAYER_PRIORITY. So instead of hard-forcing 'workqueue', we set an
explicit priority that front-loads the threadsafe layers:

    Windows ≥ 3.14 (TBB fragile):  omp  ->  workqueue  ->  tbb   (tbb unreachable)
    Windows ≤ 3.13 (TBB healthy):  omp  ->  tbb        ->  workqueue
    Linux / macOS (TBB healthy):   omp  ->  tbb        ->  workqueue

omp and tbb are both threadsafe and keep real parallelism; workqueue is the
always-available pure-C fallback but is NOT threadsafe. On the TBB-fragile combo
(Windows + Python ≥ 3.14) we drop tbb behind the always-available workqueue so
tbb can never be selected (and so can never trigger the 3.14 crash). Everywhere
else tbb is healthy, so we keep both threadsafe layers ahead of workqueue and
only land on workqueue when neither omp nor tbb is installed — which is the one
case the blink loader must serialize around. Either way the app still boots.

WHY THE DETERMINISTIC LAYER PROBE
---------------------------------
Numba's layer selection fires on the first thread that triggers a parallel
compile and is NOT fully serialized against concurrent `dlopen`s or other
threads doing the same resolution. In practice: on the same machine, same
venv, same binary, launches intermittently land on different layers ("TBB
usually, workqueue sometimes") depending on whether a QThreadPool worker,
file-open auto-sidecar, or OpenCV/MKL `dlopen` happens to race the warmup
thread's probe.

To make the selection deterministic we compile one tiny parallel kernel here,
at bootstrap import time, on the main thread, BEFORE any Qt thread exists.
That first compile locks numba's layer choice into the per-process cache.
Every subsequent numba call from any thread sees the already-resolved layer
and the race is impossible. The compile is cached (~30 ms warm, ~500 ms cold
first-ever install), so the startup cost is amortized to effectively zero.

NOTE: numba_warmup formerly hard-pinned NUMBA_THREADING_LAYER=workqueue on ALL
platforms, which overrode this steering and forced the non-threadsafe layer
even where omp/tbb existed (this is what put a Linux user on workqueue). That
pin has been removed; this module is now the single source of truth for layer
selection on every platform.

This must be imported as the very first thing in every process entry point
(gui_entry, the CLI, __main__) — before any module that does `import numba` or
`from numba import ...`. Importing it runs configure() for side effect, then
runs the main-thread layer probe for side effect.

VERIFYING IT WORKED
-------------------
This module logs the resolved layer directly after the probe runs.
numba_warmup._do_warmup() also re-logs it. Read either line in saspro.log:
  - 'omp'       -> best case: threadsafe + parallel
  - 'tbb'       -> also threadsafe + parallel; expected on healthy Linux/macOS
                   and on Windows ≤ 3.13
  - 'workqueue' -> booted safely, but NOT threadsafe; NUMBA_PARALLEL_LOCK will
                   serialize to stay safe (loses ~2-3× in debayer/stretch
                   throughput). Expected only when neither omp nor tbb is
                   installed on this machine, OR when the user's Python is
                   3.14+ on Windows (TBB demoted intentionally).
"""
from __future__ import annotations

import logging
import os
import platform
import sys

# What we set (a priority string), or None if we left Numba's default alone.
# The *resolved* layer is reported by RESOLVED_LAYER below after configure().
SELECTED_PRIORITY: str | None = None

# The actual numba layer that got locked in by the main-thread probe below,
# or None if the probe hasn't run yet or failed. Readable from anywhere via
# `from ...numba_bootstrap import RESOLVED_LAYER`.
RESOLVED_LAYER: str | None = None

# Windows / TBB-fragile priority: TBB demoted BELOW the always-available
# workqueue, so tbb is unreachable and can never be the one that runs.
_SAFE_PRIORITY = "omp workqueue tbb"

# Non-fragile priority: keep BOTH threadsafe layers (omp, then tbb) ahead of the
# non-threadsafe workqueue fallback, so a box only lands on workqueue — and the
# blink loader only has to serialize — when neither omp nor tbb is installed.
_PREFER_SAFE_PRIORITY = "omp tbb workqueue"


def _tbb_is_fragile() -> bool:
    """True on the Python/OS combos where TBB is known to abort at JIT time.

    Currently: Windows + CPython ≥ 3.14, which has the uncatchable TBB
    access violation during the first parallel compile. On Python ≤ 3.13
    (what most Windows users still ship on today) TBB is healthy and
    threadsafe + parallel, so we WANT it in the priority list.

    Widen this (e.g. drop the Python-version check if Linux/macOS 3.14 ends
    up aborting the same way) when evidence demands — the priority demotion
    is safe everywhere, it just isn't needed where TBB is healthy.
    """
    is_windows = platform.system() == "Windows"
    # Previously this function returned True for every Windows regardless of
    # Python version, which demoted TBB on Windows ≤ 3.13 as well — costing
    # most of our Windows users a threadsafe layer they could have had. The
    # version check matches the actual bug surface (CPython 3.14+ on Windows).
    is_3_14_plus = sys.version_info >= (3, 14)
    return is_windows and is_3_14_plus


def configure() -> str | None:
    """Demote TBB via THREADING_LAYER_PRIORITY if this combo needs protection.

    Returns the priority string we set, or None if we left Numba's default in
    place. Safe to call repeatedly and safe on any platform. Never clobbers an
    explicit operator override of either NUMBA_THREADING_LAYER or
    NUMBA_THREADING_LAYER_PRIORITY.
    """
    global SELECTED_PRIORITY

    # An explicit hard layer wins outright (keeps the manual
    # NUMBA_THREADING_LAYER=workqueue workaround and any power-user setting).
    if (os.environ.get("NUMBA_THREADING_LAYER") or "").strip():
        SELECTED_PRIORITY = None
        return None

    # An explicit priority also wins — don't second-guess it.
    existing_prio = (os.environ.get("NUMBA_THREADING_LAYER_PRIORITY") or "").strip()
    if existing_prio:
        SELECTED_PRIORITY = existing_prio
        return existing_prio

    # Set an explicit priority on EVERY platform. Where TBB is fragile
    # (Windows ≥ 3.14) it goes last, behind the always-available workqueue, so
    # tbb is unreachable and can't crash. Everywhere else we front-load the
    # threadsafe layers (omp, then tbb) and leave workqueue as the last resort,
    # so a box only lands on the non-threadsafe workqueue layer when neither
    # omp nor tbb is installed. Leaving numba's own default here would have
    # preferred tbb first, which is fine for safety but skips omp; being
    # explicit also pins sane behavior against any future change to numba's
    # built-in default.
    prio = _SAFE_PRIORITY if _tbb_is_fragile() else _PREFER_SAFE_PRIORITY
    os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = prio
    SELECTED_PRIORITY = prio
    return prio


def _force_layer_resolution_on_main_thread() -> str | None:
    """Resolve numba's threading layer deterministically, NOW, on this thread.

    Compiles one tiny parallel kernel, which forces numba's first-compile
    layer resolution to happen here — on the main thread, at bootstrap time,
    before any Qt thread or QThreadPool worker can race the probe. The
    resolved layer is pinned into numba_utils' NUMBA_PARALLEL_LOCK via
    `record_numba_threading_layer` so the lock makes the right serialize-or-
    not call for every subsequent parallel region.

    This runs on EVERY platform — the probe cost is identical, and the race
    this protects against (concurrent dlopen, QThread first-compile) is
    platform-agnostic. The @cache=True means the compile is paid once per
    SASpro install, not per launch.

    Returns the resolved layer name ('tbb', 'omp', 'workqueue'), or None on
    failure. Updates module globals RESOLVED_LAYER as a side effect.
    """
    global RESOLVED_LAYER

    try:
        import numpy as np
        import numba
        from numba import njit, prange

        @njit(parallel=True, cache=True)
        def _probe(a):
            s = 0.0
            for i in prange(a.shape[0]):
                s += a[i]
            return s

        # Trigger the first parallel compile → numba locks in its layer here,
        # on this thread. Four-element array keeps the compile trivial.
        _probe(np.zeros(4, dtype=np.float32))

        layer = str(numba.threading_layer()).strip().lower()
        RESOLVED_LAYER = layer
        logging.info(
            "[numba_bootstrap] layer resolved deterministically on main "
            "thread: %s (priority was %r)",
            layer, SELECTED_PRIORITY,
        )
        if layer not in ("tbb", "omp"):
            logging.warning(
                "[numba_bootstrap] layer is '%s' — NOT threadsafe. "
                "NUMBA_PARALLEL_LOCK will serialize parallel kernels to stay "
                "safe (~2-3× cost in debayer/stretch). To get a threadsafe "
                "layer: install 'intel-openmp' or 'tbb' into this environment.",
                layer,
            )

        # Pin the resolved layer into numba_utils' lazy lock decision so the
        # very first concurrent parallel region (e.g. the blink loader) does
        # the right thing instead of guessing from sys.platform.
        try:
            from setiastro.saspro.legacy import numba_utils as _nu
            _nu.record_numba_threading_layer(layer)
        except Exception:
            # numba_utils may not be importable at this stage in some builds
            # (circular import, lazy-loaded tool); it's safe to skip — the
            # lazy lock will fall back to its OS-platform heuristic.
            pass

        return layer

    except Exception as e:
        # Any failure here MUST NOT block startup. The warmup thread will
        # resolve the layer later as a secondary path, with whatever race
        # consequences that carries.
        logging.warning(
            "[numba_bootstrap] deterministic layer probe failed: %r. "
            "Layer selection will happen on whichever thread hits numba "
            "first (may be non-deterministic).", e,
        )
        return None


# Configure on import so a bare `import ...numba_bootstrap` is enough.
# Order matters: priority must be set BEFORE the probe triggers numba's
# first-compile layer resolution.
configure()
_force_layer_resolution_on_main_thread()