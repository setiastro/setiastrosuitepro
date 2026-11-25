# src/setiastro/saspro/script_bridge.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
Run a Python script in the SASpro that is already open, from the command line:

    setiastrosuitepro --run-script /path/to/script.py

For programs that write SASpro scripts for the user (SyQon Develop first).
The script is read from disk on every run, so an edit is picked up at once.
Its run(ctx), or main(ctx), is called on SASpro's GUI thread with the same
ctx as the Scripts menu: it works on the image the user is looking at, and
it may open its own PyQt6 dialogs.

The command stays until the script is done: run(ctx) has returned and every
window it opened is closed again (a dialog shown with show() counts as well
as one run with exec()). Until then, everything the script prints and every
ctx.log() line comes back on this command's stdout, a line at a time, and
errors on stderr. An exception, including one raised later by a button in
the script's dialog, comes back as the full traceback on stderr, with the
script's file and line.

Exit codes:
    0  the script finished
    1  the script failed: an exception, or sys.exit() with a non-zero code
    2  usage: no such file, not a .py file, no run(ctx) or main(ctx)
    3  SASpro is not open
    4  SASpro is still running another script from this command
    5  the SASpro that is open is older than 2.0 and can't do this

On the wire (the single-instance connection, single_instance.py) the command
sends one JSON line, {"v": 2, "kind": "run_script", "path": ...}. SASpro
answers with JSON lines {"out": text} and {"err": text}, then {"exit": code}.
SASpro's side is script_bridge_host.py.

This module is imported before anything else on that command, so it uses
the standard library only: no Qt.
"""
from __future__ import annotations

import json
import os
import sys
import time

FLAG = "--run-script"
KIND = "run_script"
# argv hand-offs stay at version 1, so an older SASpro still opens files sent
# by this one; a version-1 SASpro ignores this message, which is how the
# command can tell it is too old.
SCRIPT_PROTOCOL_VERSION = 2

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NOT_RUNNING = 3
EXIT_BUSY = 4
EXIT_TOO_OLD = 5

_HELP = {"-h", "--help", "--h", "-help", "help", "-?", "/?", "/h"}
_BUSY_WAIT_S = 3.0              # a pipe instance that is busy: try again for this long

USAGE = """usage: setiastrosuitepro --run-script SCRIPT.py

Runs SCRIPT.py in the SASpro that is already open, on the image you are
looking at, as the Scripts menu does: its run(ctx), or main(ctx), on the GUI
thread. The script is read from disk each time, and it may open dialogs.

The command stays until the script is done: run(ctx) has returned and any
window it opened is closed. Meanwhile print() and ctx.log() come back on
stdout a line at a time, and errors on stderr, with the full traceback for
an exception. Output is UTF-8.

Exit codes:
  0  the script finished
  1  the script failed (an exception, or sys.exit() with a non-zero code)
  2  usage: no such file, not a .py file, no run(ctx) or main(ctx)
  3  SASpro is not open
  4  SASpro is still running another script from this command
  5  the SASpro that is open is older than 2.0
"""


def wants_run_script(argv) -> bool:
    return any(str(a).lower() == FLAG or str(a).lower().startswith(FLAG + "=")
               for a in (argv or []))


def script_argument(argv):
    """What follows --run-script: the script path, a help word, or None."""
    argv = list(argv or [])
    for i, a in enumerate(argv):
        low = str(a).lower()
        if low.startswith(FLAG + "="):
            return a[len(FLAG) + 1:] or None
        if low == FLAG:
            return argv[i + 1] if i + 1 < len(argv) else None
    return None


def _say(stream, text: str) -> None:
    if stream is None:
        return
    try:
        stream.write(text)
        stream.flush()
    except Exception:
        pass


def _utf8_streams() -> None:
    """Piped output in UTF-8 whatever the locale; a console keeps its own."""
    for s in (sys.stdout, sys.stderr):
        try:
            if s is not None and not s.isatty():
                s.reconfigure(encoding="utf-8", errors="replace")
            elif s is not None:
                s.reconfigure(errors="replace")
        except Exception:
            pass


class _PipeConnection:
    """Windows: the named pipe, as a file."""

    def __init__(self, f):
        self._f = f

    def send(self, data: bytes) -> None:
        self._f.write(data)
        self._f.flush()

    def recv(self) -> bytes:
        try:
            return self._f.read(65536) or b""
        except OSError:                          # broken pipe: SASpro hung up
            return b""

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


class _SocketConnection:
    """macOS and Linux: the AF_UNIX socket."""

    def __init__(self, s):
        self._s = s

    def send(self, data: bytes) -> None:
        self._s.sendall(data)

    def recv(self) -> bytes:
        try:
            return self._s.recv(65536)
        except OSError:
            return b""

    def close(self) -> None:
        try:
            self._s.close()
        except Exception:
            pass


def _connect(name: str):
    """A connection to the open SASpro, or None if none is listening."""
    from setiastro.saspro.single_instance import (
        PROBE_CONNECT_TIMEOUT_MS, _mp_address, _posix_socket_path)
    if sys.platform.startswith("win"):
        address = _mp_address(name)
        deadline = time.monotonic() + _BUSY_WAIT_S
        while True:
            try:
                return _PipeConnection(open(address, "r+b", buffering=0))
            except FileNotFoundError:
                return None
            except OSError as e:
                # ERROR_PIPE_BUSY: SASpro is between connections.
                if getattr(e, "winerror", None) == 231 and time.monotonic() < deadline:
                    time.sleep(0.05)
                    continue
                return None
    import socket
    address = _posix_socket_path(name)
    if not os.path.exists(address):
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(PROBE_CONNECT_TIMEOUT_MS / 1000.0)
        s.connect(address)
        s.settimeout(None)                       # a script can take as long as it likes
    except OSError:
        s.close()
        return None
    return _SocketConnection(s)


def _relay(conn) -> int:
    """Pass SASpro's answer on to stdout and stderr; its exit code."""
    buf = b""
    heard = False
    while True:
        chunk = conn.recv()
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            if not isinstance(msg, dict):
                continue
            heard = True
            if "out" in msg:
                _say(sys.stdout, str(msg["out"]))
            elif "err" in msg:
                _say(sys.stderr, str(msg["err"]))
            elif "exit" in msg:
                try:
                    return int(msg["exit"])
                except (TypeError, ValueError):
                    return EXIT_FAILED
    if not heard:
        _say(sys.stderr, "The SASpro that is open can't run scripts from the command line. "
                         "That needs SASpro 2.0 or later.\n")
        return EXIT_TOO_OLD
    _say(sys.stderr, "SASpro closed before the script finished.\n")
    return EXIT_FAILED


def client_main(argv) -> int:
    """setiastrosuitepro --run-script SCRIPT.py"""
    _utf8_streams()
    arg = script_argument(argv)
    if arg is not None and arg.lower() in _HELP:
        _say(sys.stdout, USAGE)
        return EXIT_OK
    if not arg:
        _say(sys.stderr, USAGE)
        return EXIT_USAGE
    path = os.path.abspath(os.path.expanduser(arg))
    if not os.path.isfile(path):
        _say(sys.stderr, f"No such script: {path}\n")
        return EXIT_USAGE
    if not path.lower().endswith(".py"):
        _say(sys.stderr, f"Not a Python script (.py): {path}\n")
        return EXIT_USAGE

    from setiastro.saspro.single_instance import server_name
    conn = _connect(server_name())
    if conn is None:
        _say(sys.stderr, "SASpro is not open. Start it, open an image, and run this again.\n")
        return EXIT_NOT_RUNNING
    payload = {"v": SCRIPT_PROTOCOL_VERSION, "kind": KIND, "path": path,
               "cwd": os.getcwd(), "pid": os.getpid()}
    try:
        try:
            conn.send(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
        except OSError:
            _say(sys.stderr, "SASpro is not open. Start it, open an image, and run this again.\n")
            return EXIT_NOT_RUNNING
        return _relay(conn)
    except KeyboardInterrupt:
        # Ctrl+C here stops waiting; the script carries on in SASpro.
        return EXIT_FAILED
    finally:
        conn.close()
