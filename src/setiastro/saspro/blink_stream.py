#src/setiastro/saspro/blink_stream.py
"""Demand-loaded frame cache for Blink's Stream mode.

PreLoad decodes every frame up front and holds it; Stream scans headers only,
then decodes a frame when you look at it. This module is the bit in between:
an LRU of decoded frames under a byte budget, plus a small prefetcher so
playback and arrow-keying do not stall on every step.

Deliberately Qt-free and index-free. Frames are keyed by file path, not list
position, so deleting or reordering rows in the tree cannot alias one frame's
pixels onto another's row -- which is exactly the class of bug that makes a
blink comparator untrustworthy.

The caller supplies ``load_fn(path) -> (array, meta)``. Everything about how a
frame is decoded, stretched and stored stays in BlinkTab; this only decides
what is resident and what gets thrown away.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

DEFAULT_BUDGET_BYTES = 1 << 30      # 1 GiB of decoded frames
DEFAULT_WORKERS = 2                 # enough to hide I/O, few enough to stay calm
DEFAULT_PREFETCH = 3                # frames to run ahead in the travel direction


class FrameCache:
    """LRU of decoded frames with a byte budget and a background prefetcher."""

    def __init__(self, load_fn, budget_bytes=DEFAULT_BUDGET_BYTES,
                 workers=DEFAULT_WORKERS):
        self._load_fn = load_fn
        self._budget = max(1, int(budget_bytes))
        self._lock = threading.RLock()
        self._cache = OrderedDict()      # path -> (array, meta)
        self._bytes = 0
        self._inflight = {}              # path -> Future
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)),
                                        thread_name_prefix="blink-stream")
        self._closed = False
        self.hits = 0
        self.misses = 0

    # -- internals ---------------------------------------------------------

    def _store(self, path, payload):
        array, _meta = payload
        nbytes = int(getattr(array, "nbytes", 0) or 0)
        with self._lock:
            if path in self._cache:
                self._bytes -= int(getattr(self._cache[path][0], "nbytes", 0) or 0)
            self._cache[path] = payload
            self._cache.move_to_end(path)
            self._bytes += nbytes
            # A single frame larger than the whole budget still stays: evicting
            # it would mean never being able to show that frame at all.
            while len(self._cache) > 1 and self._bytes > self._budget:
                _old, dropped = self._cache.popitem(last=False)
                self._bytes -= int(getattr(dropped[0], "nbytes", 0) or 0)

    def _submit(self, path):
        """Start a load if one is not already running. Caller holds no lock."""
        with self._lock:
            if self._closed:
                return None
            fut = self._inflight.get(path)
            if fut is not None:
                return fut
            fut = self._pool.submit(self._run, path)
            self._inflight[path] = fut
            return fut

    def _run(self, path):
        try:
            payload = self._load_fn(path)
            if payload is not None:
                self._store(path, payload)
            return payload
        finally:
            with self._lock:
                self._inflight.pop(path, None)

    def _cancel_pending(self):
        """Drop queued prefetches that have not started.

        A click must not wait behind speculative work. Futures already running
        are left alone -- they will land in the cache and be useful.
        """
        with self._lock:
            for path, fut in list(self._inflight.items()):
                if fut.cancel():
                    self._inflight.pop(path, None)

    # -- public API --------------------------------------------------------

    def peek(self, path):
        """Resident payload, or None. Never blocks, never loads."""
        with self._lock:
            payload = self._cache.get(path)
            if payload is not None:
                self._cache.move_to_end(path)
            return payload

    def get(self, path, prefetch=()):
        """Payload for ``path``, loading it if needed. Blocks.

        Queued-but-unstarted prefetches are cancelled first so an interactive
        click never queues behind speculation. ``prefetch`` is scheduled after
        the wanted frame is in hand.
        """
        payload = self.peek(path)
        if payload is not None:
            with self._lock:
                self.hits += 1
            if prefetch:
                self.prefetch(prefetch)
            return payload

        with self._lock:
            self.misses += 1
        self._cancel_pending()
        fut = self._submit(path)
        if fut is None:
            return None
        try:
            payload = fut.result()
        except Exception:
            raise
        if prefetch:
            self.prefetch(prefetch)
        return payload

    def prefetch(self, paths):
        """Queue background loads for paths not already resident or running."""
        for path in paths:
            if path is None:
                continue
            with self._lock:
                if self._closed or path in self._cache or path in self._inflight:
                    continue
            self._submit(path)

    def invalidate(self, path):
        """Forget one frame (it was renamed, moved or deleted)."""
        with self._lock:
            payload = self._cache.pop(path, None)
            if payload is not None:
                self._bytes -= int(getattr(payload[0], "nbytes", 0) or 0)

    def retain(self, live_paths):
        """Forget every resident frame whose path is no longer in the list.

        Called after rows are deleted or moved out. Keying by path means a
        stale entry can only ever waste memory, never show the wrong pixels
        for a row -- but there is no reason to keep paying for it.
        """
        live = set(live_paths)
        with self._lock:
            stale = [k for k in self._cache if k not in live]
        for path in stale:
            self.invalidate(path)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._bytes = 0

    def shutdown(self):
        with self._lock:
            self._closed = True
            for fut in list(self._inflight.values()):
                fut.cancel()
            self._inflight.clear()
            self._cache.clear()
            self._bytes = 0
        self._pool.shutdown(wait=False)

    @property
    def resident(self):
        with self._lock:
            return len(self._cache)

    @property
    def nbytes(self):
        with self._lock:
            return self._bytes


def prefetch_window(paths, index, direction, count=DEFAULT_PREFETCH):
    """The next ``count`` paths from ``index`` in the travel direction.

    Playback and arrow-keying move one way at a time, so running ahead in that
    direction is worth far more than a symmetric window. Wraps, because Blink's
    next_item/previous_item wrap.
    """
    n = len(paths)
    if n == 0:
        return []
    step = 1 if direction >= 0 else -1
    out = []
    for k in range(1, min(count, n - 1) + 1):
        out.append(paths[(index + k * step) % n])
    return out