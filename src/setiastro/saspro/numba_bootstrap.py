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
THREADING_LAYER_PRIORITY. So instead of hard-forcing 'workqueue', we set an
explicit priority that front-loads the threadsafe layers:

    Windows (TBB fragile):  omp  ->  workqueue  ->  tbb   (tbb unreachable)
    elsewhere:              omp  ->  tbb        ->  workqueue

omp and tbb are both threadsafe and keep real parallelism; workqueue is the
always-available pure-C fallback but is NOT threadsafe. On Windows we drop tbb
behind the always-available workqueue so tbb can never be selected (and so can
never trigger the 3.14 crash). Everywhere else tbb is healthy, so we keep both
threadsafe layers ahead of workqueue and only land on workqueue when neither
omp nor tbb is installed — which is the one case the blink loader must
serialize around. Either way the app still boots.

NOTE: numba_warmup formerly hard-pinned NUMBA_THREADING_LAYER=workqueue on ALL
platforms, which overrode this steering and forced the non-threadsafe layer
even where omp/tbb existed (this is what put a Linux user on workqueue). That
pin is now scoped to Windows; this module is the single source of truth for
layer selection everywhere else.

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

# Windows / TBB-fragile priority: TBB demoted BELOW the always-available
# workqueue, so tbb is unreachable and can never be the one that runs.
_SAFE_PRIORITY = "omp workqueue tbb"

# Non-fragile priority: keep BOTH threadsafe layers (omp, then tbb) ahead of the
# non-threadsafe workqueue fallback, so a box only lands on workqueue — and the
# blink loader only has to serialize — when neither omp nor tbb is installed.
_PREFER_SAFE_PRIORITY = "omp tbb workqueue"


def _tbb_is_fragile() -> bool:
    """True on the Python/OS combos where TBB is known to abort at JIT time.

    Widen this (e.g. drop the Windows check) if Linux/macOS 3.14 turns out to
    abort the same way — the priority demotion is safe everywhere, it just
    isn't needed where TBB is healthy.
    """
    maj, minor = sys.version_info.major, sys.version_info.minor
    is_windows = platform.system() == "Windows"
    return is_windows


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
    # (Windows) it goes last, behind the always-available workqueue, so tbb is
    # unreachable and can't crash. Everywhere else we front-load the threadsafe
    # layers (omp, then tbb) and leave workqueue as the last resort, so a box
    # only lands on the non-threadsafe workqueue layer when neither omp nor tbb
    # is installed. Leaving numba's own default here would have preferred tbb
    # first, which is fine for safety but skips omp; being explicit also pins
    # sane behavior against any future change to numba's built-in default.
    prio = _SAFE_PRIORITY if _tbb_is_fragile() else _PREFER_SAFE_PRIORITY
    os.environ["NUMBA_THREADING_LAYER_PRIORITY"] = prio
    SELECTED_PRIORITY = prio
    return prio


# Configure on import so a bare `import ...numba_bootstrap` is enough.
configure()