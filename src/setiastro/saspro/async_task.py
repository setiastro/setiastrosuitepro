# saspro/async_task.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
#
# Run a blocking callable off the GUI thread while keeping the UI responsive.
#
# run_async_blocking(fn, ...) executes fn on a short-lived QThread and spins a
# LOCAL QEventLoop on the calling (GUI) thread until it finishes, then returns
# fn's result -- or re-raises fn's exception -- on the caller's thread. While it
# waits, a modal, indeterminate QProgressDialog keeps the app painting and
# blocks re-entrant interaction with `parent`, so the user can't re-trigger the
# same action or tear the dialog down underneath a running task.
#
# This is for network I/O (SIMBAD, Gaia archive) that otherwise pins the GUI
# thread for the full duration of each request. Two hard rules for `fn`:
#   * It runs on a worker thread, so it MUST NOT touch any Qt widget.
#   * For progress text, pass progress_kw="<kwarg>" and on_progress=<gui fn>:
#     the worker receives a thread-safe emitter under that kwarg, and each
#     emission is marshalled through a queued signal to on_progress (and the
#     dialog label) on the GUI thread. Do NOT hand fn a callback that touches
#     widgets directly.
#
# Control flow is preserved: the calling method still waits for the result
# (linear code stays linear). It keeps the *UI* responsive; it is not
# fire-and-forget. For truly detached work, use a QThread with result signals.

from __future__ import annotations

from PyQt6.QtCore import QThread, QEventLoop, pyqtSignal, Qt
from PyQt6.QtWidgets import QProgressDialog


class AsyncCancelled(Exception):
    """Raised in the caller when the user cancels a run_async_blocking task.

    Note: if you enable cancellable=True inside a bare ``except Exception:``
    retry loop, catch AsyncCancelled first and break -- otherwise the loop will
    treat a cancel as a failed attempt and retry.
    """


# Cancelled tasks can't force-kill a blocked socket, so we detach the worker and
# keep a reference until it unwinds on its own -- otherwise Python/Qt could free
# it mid-call and crash.
_detached: set = set()


class _Worker(QThread):
    progress = pyqtSignal(str)

    def __init__(self, fn, args, kwargs, progress_kw):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = dict(kwargs)
        self.result = None
        self.error: BaseException | None = None
        if progress_kw is not None:
            self._kwargs[progress_kw] = self._emit_progress

    def _emit_progress(self, msg):
        # Called on the worker thread; the signal is delivered queued to the GUI.
        try:
            self.progress.emit(str(msg))
        except Exception:
            pass

    def run(self):
        try:
            self.result = self._fn(*self._args, **self._kwargs)
        except BaseException as exc:   # noqa: BLE001 - marshalled to the caller
            self.error = exc


def run_async_blocking(
    fn,
    *args,
    parent=None,
    title: str = "",
    label: str = "Working\u2026",
    cancellable: bool = True,
    progress_kw: str | None = None,
    on_progress=None,
    **kwargs,
):
    """Run fn(*args, **kwargs) off the GUI thread; return its result or raise.

    parent       : widget the modal progress dialog attaches to.
    title/label  : progress dialog window title / message.
    cancellable  : show a Cancel button; on cancel, raise AsyncCancelled and
                   detach the still-running worker.
    progress_kw  : if set, inject a thread-safe progress emitter under this
                   kwarg name so fn can report status without touching widgets.
    on_progress  : GUI-thread callable invoked (queued) with each progress msg.
    """
    worker = _Worker(fn, args, kwargs, progress_kw)

    loop = QEventLoop()
    # Queued (cross-thread) connection: even if the worker finishes before
    # loop.exec() starts, the quit is posted and exec() returns immediately.
    worker.finished.connect(loop.quit)

    dlg = QProgressDialog(label, "Cancel", 0, 0, parent)   # (0,0) => busy bar
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    if not cancellable:
        dlg.setCancelButton(None)

    cancelled = {"flag": False}
    if cancellable:
        def _on_cancel():
            cancelled["flag"] = True
            loop.quit()
        dlg.canceled.connect(_on_cancel)

    def _on_prog(msg):
        try:
            dlg.setLabelText(msg)
        except Exception:
            pass
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:
                pass
    worker.progress.connect(_on_prog)

    worker.start()
    dlg.show()
    if not worker.isFinished():
        loop.exec()
    try:
        dlg.close()
    except Exception:
        pass

    if cancelled["flag"] and worker.isRunning():
        _detached.add(worker)
        worker.finished.connect(lambda w=worker: _detached.discard(w))
        worker.finished.connect(worker.deleteLater)
        raise AsyncCancelled()

    worker.wait()
    if worker.error is not None:
        raise worker.error
    return worker.result