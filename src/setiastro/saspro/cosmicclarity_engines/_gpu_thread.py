# src/setiastro/saspro/cosmicclarity_engines/_gpu_thread.py
# SetiAstro Suite Pro -- one long-lived thread for Cosmic Clarity's GPU work.
#
# Copyright (c) 2026 Franklin Marek / SetiAstro

"""
PyTorch keeps cuDNN's execution plans per thread, and under fp16 autocast
building one compiles kernels at run time: ~3.5 s for every input shape, paid
again on every new thread (measured on an RTX 3070 Ti: three fresh threads
running one shape paid 3.7, 3.4 and 3.3 s; the same thread's second call took
0.16 s). Each Execute ran its engine in a fresh worker thread, so every run
paid it, not only the first of the session.

The engines' public entry points (gpu_thread_entry) now run on this one
thread, so plans built by one run -- or by a warm-up (warmup.py) -- serve
every later one. Callers still block until their result is back, exactly as
before; work from two tools at once is taken in turn.
"""

from __future__ import annotations

import functools
import itertools
import queue
import threading
from concurrent.futures import Future

_RUN, _BACKGROUND = 0, 1        # queue priorities: a run goes ahead of queued warm-ups


class _GpuThread:
    def __init__(self):
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()               # FIFO within a priority
        self._lock = threading.Lock()
        self._runs_waiting = 0
        self.thread = threading.Thread(target=self._loop, name="cc-gpu", daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            prio, _n, (fut, fn, args, kwargs) = self._q.get()
            if prio == _RUN:
                with self._lock:
                    self._runs_waiting -= 1
            try:
                if fut.set_running_or_notify_cancel():
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as e:      # EngineCancelled etc.: re-raised in the caller
                        fut.set_exception(e)
                    else:
                        fut.set_result(result)
                        result = None
            finally:
                # Drop this job's references now: the image and its result can
                # be hundreds of MB, and nothing else would free them until the
                # next job arrived.
                fut = fn = args = kwargs = None

    def submit(self, fn, args=(), kwargs=None, *, background: bool = False) -> Future:
        fut: Future = Future()
        prio = _BACKGROUND if background else _RUN
        if prio == _RUN:
            with self._lock:
                self._runs_waiting += 1
        self._q.put((prio, next(self._seq), (fut, fn, tuple(args), dict(kwargs or {}))))
        return fut

    def has_waiting_work(self) -> bool:
        """A run is queued (warm-ups don't count): a warm-up should stop."""
        return self._runs_waiting > 0


_INSTANCE: _GpuThread | None = None
_INSTANCE_LOCK = threading.Lock()


def _gpu_thread() -> _GpuThread:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None or not _INSTANCE.thread.is_alive():
            _INSTANCE = _GpuThread()
        return _INSTANCE


def on_gpu_thread() -> bool:
    inst = _INSTANCE
    return inst is not None and threading.current_thread() is inst.thread


def has_waiting_work() -> bool:
    """True when a run is queued behind the current job: a warm-up stops early."""
    inst = _INSTANCE
    return inst is not None and inst.has_waiting_work()


def submit_background(fn, *args, **kwargs) -> Future:
    """Queue fn on the GPU thread without waiting, behind any run (warm-ups)."""
    return _gpu_thread().submit(fn, args, kwargs, background=True)


def run(fn, *args, **kwargs):
    """Run fn on the GPU thread and wait for it; inline when already there."""
    if on_gpu_thread():
        return fn(*args, **kwargs)
    return _gpu_thread().submit(fn, args, kwargs).result()


def gpu_thread_entry(fn):
    """Decorator for an engine's public entry point (see the module docstring)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return run(fn, *args, **kwargs)
    return wrapper
