# src/setiastro/saspro/thread_keeper.py
"""Ownership for QThread workers that no window should own.

A QThread parented to a dialog is destroyed with the dialog, and a Python-owned
QThread is destroyed when its last reference goes (the window holding it was
closed and collected). If it is still running either way, Qt calls qFatal
("QThread: Destroyed while thread is still running") and SASpro exits silently
with 0xC0000409 and nothing in saspro_crash.log.

Threads started with start_owned_thread() have no parent and are held here until
they finish, so the window that started one can close, or be deleted, at any
point in the run. On app quit, live threads are asked to stop (cancel() when
they have it, plus requestInterruption()) and waited for.
"""
from __future__ import annotations

from PyQt6.QtCore import QCoreApplication, QThread

LIVE_THREADS: set = set()
QUIT_WAIT_MS = 30000          # a cancel normally lands within one unit of work
_quit_hooked = False


def start_owned_thread(thr: QThread) -> None:
    """Start thr and hold it until it finishes; it is deleteLater()'d then."""
    LIVE_THREADS.add(thr)
    thr.finished.connect(lambda t=thr: _release(t))
    _hook_quit()
    thr.start()


def thread_running(thr) -> bool:
    """isRunning() that answers False, not an error, once the object is deleted."""
    if thr is None:
        return False
    try:
        return bool(thr.isRunning())
    except RuntimeError:
        return False


def _release(thr: QThread) -> None:
    LIVE_THREADS.discard(thr)
    try:
        thr.deleteLater()
    except RuntimeError:
        pass


def _hook_quit() -> None:
    global _quit_hooked
    if _quit_hooked:
        return
    app = QCoreApplication.instance()
    if app is None:
        return
    app.aboutToQuit.connect(stop_all_for_quit)
    _quit_hooked = True


def stop_all_for_quit() -> None:
    live = list(LIVE_THREADS)
    for thr in live:
        for stop in ("cancel", "requestInterruption"):
            fn = getattr(thr, stop, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
    for thr in live:
        try:
            thr.wait(QUIT_WAIT_MS)
        except Exception:
            pass
