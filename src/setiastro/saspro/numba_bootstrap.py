# src/setiastro/saspro/numba_bootstrap.py
"""
Steer Numba away from the TBB threading layer BEFORE Numba is ever imported.

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
THREADING_LAYER_PRIORITY. So instead of hard-forcing 'workqueue', we demote TBB
to last in that priority list:

    omp  ->  workqueue  ->  tbb

omp is threadsafe and keeps real parallelism; workqueue is the always-available
pure-C fallback. Because workqueue is always available, tbb is never reached —
it can't be selected, so it can't crash. If omp is present and healthy on this
machine the user gets it (and stays off the not-threadsafe workqueue path that
the blink loader has to serialize around); otherwise they get workqueue and the
app still boots.

This must be imported as the very first thing in every process entry point
(gui_entry, the CLI, __main__) — before any module that does `import numba` or
`from numba import ...`. Importing it runs configure() for side effect.

VERIFYING IT WORKED
-------------------
numba_warmup._do_warmup() logs `Numba threading layer resolved: <layer>` after
a parallel kernel runs. Read that line in saspro.log:
  - 'omp'       -> best case: threadsafe + parallel, blink (#3) unlikely to bite
  - 'workqueue' -> booted safely, but NOT threadsafe; confirm NUMBA_PARALLEL_LOCK
                   covers the blink loader before calling #3 closed
  - 'tbb'       -> the demotion did NOT take effect; investigate before shipping
"""
from __future__ import annotations

import os
import platform
import sys

# What we set (a priority string), or None if we left Numba's default alone.
# The *resolved* layer is logged by numba_warmup, not known here.
SELECTED_PRIORITY: str | None = None

# Priority with TBB demoted to last. workqueue being always-available means
# tbb is unreachable, so it can never be the one that runs.
_SAFE_PRIORITY = "omp workqueue tbb"


def _tbb_is_fragile() -> bool:
    """True on the Python/OS combos where TBB is known to abort at JIT time.

    Widen this (e.g. drop the Windows check) if Linux/macOS 3.14 turns out to
    abort the same way — the priority demotion is safe everywhere, it just
    isn't needed where TBB is healthy.
    """
    maj, minor = sys.version_info.major, sys.version_info.minor
    is_windows = platform.system() == "Windows"
    return is_windows and maj == 3 and minor >= 14


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

    if not _tbb_is_fragile():
        SELECTED_PRIORITY = None
        return None

    os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = _SAFE_PRIORITY
    SELECTED_PRIORITY = _SAFE_PRIORITY
    return _SAFE_PRIORITY


# Configure on import so a bare `import ...numba_bootstrap` is enough.
configure()