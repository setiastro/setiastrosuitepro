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

    Only folder components are searched — never the filename. A keyword that
    also appears in light names would otherwise create one session per file.
    """
    kw = (keyword or "").strip()
    if not kw or kw.lower() == "default":
        return "Default"

    dirpath = os.path.dirname(os.path.normpath(path))
    parts = [p for p in dirpath.split(os.sep) if p]

    # Prefer folders like NIGHT1 / NIGHT_2 / NIGHT-3.
    pat = re.compile(rf"^{re.escape(kw)}\s*[_-]?\s*\d+$", re.IGNORECASE)
    for part in reversed(parts):
        if pat.match(part):
            return part

    kw_low = kw.lower()
    for part in reversed(parts):
        if kw_low in part.lower():
            return part

    return kw


_FILENAME_SET_SEP = r"[\s_\-\.]+"


def set_name_from_filename_keyword(path: str, keyword: str) -> str | None:
    """Build a registration-set name from the *filename* using a user keyword.

    Matches ``Keyword<sep><tag>`` in the file stem only (directories are
    ignored). ``sep`` is one or more spaces, underscores, hyphens, or dots.
    ``tag`` is the following alphanumeric token (``1``, ``01``, ``A``, …).

    Returns ``None`` when the keyword is empty or no match is found so the
    caller can leave the frame in Default.
    """
    kw = (keyword or "").strip()
    if not kw or kw.lower() == "default":
        return None

    stem = os.path.splitext(os.path.basename(path or ""))[0]
    if not stem:
        return None

    pat = re.compile(
        rf"(?:^|{_FILENAME_SET_SEP}){re.escape(kw)}{_FILENAME_SET_SEP}"
        rf"([A-Za-z0-9]+)(?={_FILENAME_SET_SEP}|$)",
        re.IGNORECASE,
    )
    m = pat.search(stem)
    if not m:
        return None
    return f"{kw} {m.group(1)}"


def is_master_flat_key(key: str, *, filter_name: str, image_size: str) -> bool:
    """True if *key* is a master-flat dict key for this filter and size.

    Keys look like ``G (WxH) [session] [G0]``. Matching must not treat the
    letter G as a substring, or it also hits gain tags ``[G0]`` / ``[G100]``.
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
    job keeps its arguments alive, so an unbounded queue of calibrated frames
    can exhaust RAM.
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
