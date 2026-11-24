# src/setiastro/saspro/script_bridge_host.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
SASpro's side of `setiastrosuitepro --run-script SCRIPT.py` (script_bridge.py).

The single-instance listener hands over the connection; a _Session runs the
script on the GUI thread, as the Scripts menu does, and sends its output back
as it comes. While a script runs, sys.stdout and sys.stderr still write where
they did and also go to the command, and so does ctx.log(). Uncaught
exceptions, run(ctx)'s own or later ones from the script's buttons, go back
as tracebacks and make the exit code 1.

The command ends when run(ctx) has returned and the windows that appeared
during it are closed. One script at a time: another --run-script meanwhile
is turned away (exit code 4) rather than started inside the first one's
dialog.
"""
from __future__ import annotations

import collections
import importlib.util
import itertools
import json
import sys
import threading
import traceback
from pathlib import Path

from PyQt6 import sip
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from setiastro.saspro.ops.scripts import ScriptContext
from setiastro.saspro.script_bridge import (
    EXIT_BUSY, EXIT_FAILED, EXIT_OK, EXIT_USAGE)

_current = None                     # the session that is running a script
_numbers = itertools.count(1)
_WINDOW_POLL_MS = 250
_TRANSIENT = (Qt.WindowType.Popup, Qt.WindowType.ToolTip, Qt.WindowType.Desktop)


def run_from_command(main_window, payload: dict, sock) -> None:
    """Called by the single-instance listener on the GUI thread."""
    _Session(main_window, sock, str(payload.get("path") or "")).start()


class _BridgeContext(ScriptContext):
    """The Scripts menu's ctx, with two differences. ctx.log() also goes to the
    command. And the view is found while SASpro isn't the active application:
    the command usually comes from another program, which has the focus, and
    then the MDI area has no *active* subwindow, only a current one."""

    def __init__(self, app_window, session):
        super().__init__(app_window)
        self._session = session

    def log(self, msg: str):
        try:
            self.app._log(f"[Script] {msg}")
        except Exception:
            pass
        self._session.feed("out", f"{msg}\n")

    def active_subwindow(self):
        try:
            mdi = self.app.mdi
            return mdi.activeSubWindow() or mdi.currentSubWindow()
        except Exception:
            return None


class _Tee:
    """sys.stdout or sys.stderr while a script runs: writes where it did, and to
    the command as well."""

    def __init__(self, inner, session, kind):
        self._inner = inner
        self._session = session
        self._kind = kind

    def write(self, s):
        try:
            if self._inner is not None:
                self._inner.write(s)
        except Exception:
            pass
        if s:
            self._session.feed(self._kind, str(s))
        return len(s) if s else 0

    def flush(self):
        try:
            if self._inner is not None:
                self._inner.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _alive(w) -> bool:
    try:
        return not sip.isdeleted(w)
    except Exception:
        return False


def _is_window(w) -> bool:
    try:
        return w.isWindow() and w.windowType() not in _TRANSIENT
    except Exception:
        return False


class _Session(QObject):
    _wake = pyqtSignal()                    # output from another thread is queued

    def __init__(self, main_window, sock, path: str):
        super().__init__(main_window)
        self._mw = main_window
        self._sock = sock
        self._path = Path(path)
        self._name = self._path.name or "script"
        self._module = f"saspro_cli_script_{next(_numbers)}"
        self._pending = {"out": "", "err": ""}
        self._code = EXIT_OK
        self._windows = []
        self._done = False
        self._mute = False
        self._saved = None
        self._queue = collections.deque()   # (kind, text) from other threads
        self._wake.connect(self._drain)
        self._poll = QTimer(self)
        self._poll.setInterval(_WINDOW_POLL_MS)
        self._poll.timeout.connect(self._check_windows)

    # ---- output -------------------------------------------------------------
    def feed(self, kind: str, text: str) -> None:
        """From any thread."""
        if self._done or self._mute or not text:
            return
        if threading.current_thread() is threading.main_thread():
            self._drain()
            self._on_chunk(kind, text)
        else:
            self._queue.append((kind, text))
            self._wake.emit()

    def _drain(self) -> None:
        """Output the script's threads queued, in order."""
        while self._queue:
            try:
                kind, text = self._queue.popleft()
            except IndexError:
                break
            self._on_chunk(kind, text)

    def _on_chunk(self, kind: str, text: str) -> None:
        """Whole lines go now; the rest waits for its newline or the end."""
        if self._done:
            return
        text = self._pending[kind] + text
        head, nl, tail = text.rpartition("\n")
        self._pending[kind] = tail
        if nl:
            self._send({kind: head + nl})

    def _send(self, msg: dict) -> None:
        sock = self._sock
        try:
            if sock is None or sip.isdeleted(sock) or not sock.isOpen():
                return
            sock.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
            sock.flush()
            # Out now, even while the script keeps the GUI thread busy.
            sock.waitForBytesWritten(200)
        except Exception:
            pass

    # ---- the run --------------------------------------------------------------
    def start(self) -> None:
        global _current
        busy = _current
        if busy is not None and not busy._done:
            self._send({"err": f"SASpro is still running {busy._name} from the command line. "
                               f"Close its window, or wait for it to finish, then run again.\n"})
            self._close(EXIT_BUSY)
            return
        _current = self
        self._hook()
        try:
            from setiastro.saspro.single_instance import raise_window_to_front
            raise_window_to_front(self._mw)
        except Exception:
            pass
        self._log(f"[Scripts] Running {self._path} from the command line")
        before = set(QApplication.topLevelWidgets())
        try:
            fn = self._load()
            if fn is None:
                self._finish(EXIT_USAGE)
                return
            fn(_BridgeContext(self._mw, self))
        except SystemExit as e:
            if e.code not in (None, 0):
                self.feed("err", f"{self._name} ended with sys.exit({e.code!r})\n")
                self._code = EXIT_FAILED
        except Exception:
            self._failed(sys.exc_info(), skip=1)
        if self._code == EXIT_OK:
            # Windows the script opened and left open (show(), not exec()):
            # the script is still running until they are closed.
            self._windows = [w for w in QApplication.topLevelWidgets()
                             if w not in before and _is_window(w) and w.isVisible()]
            if self._windows:
                self._poll.start()
                return
        self._finish(self._code)

    def _load(self):
        """The script's run (or main) function, read from disk now."""
        path = self._path
        if not path.is_file():
            self.feed("err", f"No such script: {path}\n")
            return None
        spec = importlib.util.spec_from_file_location(self._module, path)
        if spec is None or spec.loader is None:
            self.feed("err", f"Can't load {path} as a Python module.\n")
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[self._module] = mod          # dataclasses and pickling look here
        spec.loader.exec_module(mod)
        for name in ("run", "main"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        self.feed("err", f"{self._name} has no run(ctx) or main(ctx).\n")
        return None

    def _check_windows(self) -> None:
        self._windows = [w for w in self._windows if _alive(w) and w.isVisible()]
        if not self._windows:
            self._finish(self._code)

    def _failed(self, exc_info, skip: int = 0) -> None:
        """A traceback to the command (and SASpro's log): the script failed."""
        t, v, tb = exc_info
        for _ in range(skip):
            if tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
        text = "".join(traceback.format_exception(t, v, tb))
        self.feed("err", text if text.endswith("\n") else text + "\n")
        self._code = EXIT_FAILED
        self._log(f"[Scripts] ERROR in {self._name}:\n{text}")

    # ---- streams and the exception hook -------------------------------------
    def _hook(self) -> None:
        out, err, hook = sys.stdout, sys.stderr, sys.excepthook
        self._saved = (out, err, hook)
        self._out = _Tee(out, self, "out")
        self._err = _Tee(err, self, "err")

        def excepthook(t, v, tb):
            # An exception from one of the script's slots, after or during
            # run(ctx). SASpro's own hook still logs it, but not twice here.
            if not self._done:
                self._failed((t, v, tb))
            self._mute = True
            try:
                hook(t, v, tb)
            except Exception:
                pass
            finally:
                self._mute = False

        self._excepthook = excepthook
        sys.stdout, sys.stderr, sys.excepthook = self._out, self._err, excepthook

    def _unhook(self) -> None:
        if self._saved is None:
            return
        out, err, hook = self._saved
        self._saved = None
        # Only what is still ours: something may have wrapped it since.
        if sys.stdout is self._out:
            sys.stdout = out
        if sys.stderr is self._err:
            sys.stderr = err
        if sys.excepthook is self._excepthook:
            sys.excepthook = hook

    # ---- the end --------------------------------------------------------------
    def _finish(self, code: int) -> None:
        global _current
        if self._done:
            return
        self._poll.stop()
        self._unhook()
        self._drain()                            # what the script's threads still queued
        for kind in ("out", "err"):
            rest, self._pending[kind] = self._pending[kind], ""
            if rest:
                self._send({kind: rest})
        self._done = True
        sys.modules.pop(self._module, None)
        if _current is self:
            _current = None
        self._log(f"[Scripts] Finished {self._name} (exit code {code})")
        self._close(code)

    def _close(self, code: int) -> None:
        self._done = True
        self._send({"exit": int(code)})
        sock, self._sock = self._sock, None
        try:
            if sock is not None and not sip.isdeleted(sock):
                if sock.state() == sock.LocalSocketState.UnconnectedState:
                    sock.deleteLater()           # the command has gone already
                else:
                    sock.disconnected.connect(sock.deleteLater)
                    sock.disconnectFromServer()  # after what is still to be written
        except Exception:
            pass
        self.deleteLater()

    def _log(self, msg: str) -> None:
        try:
            self._mw._log(msg)
        except Exception:
            pass
