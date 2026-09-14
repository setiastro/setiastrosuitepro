# src/setiastro/saspro/stacking_gpu_mem.py
"""Helpers for stacking calibration: session tags, master-flat keys, GPU memory.

Kept as a tiny import-light module so the stacking VRAM / matching rules can be
unit-tested without pulling in stacking_suite (Qt / torch / numba).
"""
from __future__ import annotations

import os
import re
from concurrent.futures import FIRST_COMPLETED, wait


def session_from_manual_keyword(path: str, keyword: str) -> str:
    """Build a session tag from *directory* names using a user keyword.

    Only folder components are searched — never the filename. Otherwise a
    keyword like ``Panel`` matches ``NGC 7822 Panel 1_B_….fits`` and every
    light becomes its own session (which then duplicates master flats in VRAM).
    """
    kw = (keyword or "").strip()
    if not kw or kw.lower() == "default":
        return "Default"

    dirpath = os.path.dirname(os.path.normpath(path))
    parts = [p for p in dirpath.split(os.sep) if p]

    # Prefer folders like NIGHT1 / NIGHT_2 / NIGHT-3 / panel-01 / panel1.
    pat = re.compile(rf"^{re.escape(kw)}\s*[_-]?\s*\d+$", re.IGNORECASE)
    for part in reversed(parts):
        if pat.match(part):
            return part

    kw_low = kw.lower()
    for part in reversed(parts):
        if kw_low in part.lower():
            return part

    return kw


def is_master_flat_key(key: str, *, filter_name: str, image_size: str) -> bool:
    """True if *key* is a master-flat dict key for this filter and size.

    Keys look like ``G (9576x6388) [Default] [G0]``. Matching must not treat
    the letter G as a substring — that hits gain tags ``[G0]`` / ``[G100]``
    on darks and on every other filter's flats.
    """
    prefix = f"{filter_name} ({image_size})"
    k = str(key or "")
    return k == prefix or k.startswith(prefix + " ")


def gpu_flat_cache_key(group_key, flat_path: str | None, *, interactive: bool):
    """Identity used to share one GPU flat tensor across many groups.

    Non-interactive calibration uses the same master flat for every group that
    points at the same path, so the cache key is the path. Interactive
    adjustments are per-group, so those stay unique.
    """
    if not flat_path:
        return None
    if interactive:
        return ("interactive", group_key)
    return ("path", flat_path)


def is_cuda_oom(exc: BaseException) -> bool:
    """True if *exc* is a CUDA / GPU out-of-memory failure."""
    name = type(exc).__name__
    if "OutOfMemory" in name:
        return True
    msg = str(exc).lower()
    if "out of memory" not in msg:
        return False
    return ("cuda" in msg) or ("gpu" in msg) or ("cudnn" in msg)


def reap_completed(pending, on_error=None):
    """Drop finished futures; report failures via *on_error* (or re-raise)."""
    still = []
    for fut in pending:
        if not fut.done():
            still.append(fut)
            continue
        try:
            fut.result()
        except Exception as e:
            if on_error is not None:
                on_error(e)
            else:
                raise
    return still


def submit_bounded(executor, pending, fn, arg, *, max_pending, on_error=None):
    """Submit ``fn(arg)`` without letting more than *max_pending* futures sit queued.

    ``ThreadPoolExecutor`` accepts unlimited ``submit()`` calls. Each queued
    job keeps its arguments alive — for calibrated lights that's a ~200+ MiB
    float32 array per frame, which OOMs a 1000-frame run.
    """
    limit = max(1, int(max_pending))
    pending = list(pending)
    while len(pending) >= limit:
        done, not_done = wait(pending, return_when=FIRST_COMPLETED)
        pending = list(not_done)
        for fut in done:
            try:
                fut.result()
            except Exception as e:
                if on_error is not None:
                    on_error(e)
                else:
                    raise
    pending.append(executor.submit(fn, arg))
    return pending
