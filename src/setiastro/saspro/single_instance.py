# src/setiastro/saspro/single_instance.py
"""
Single-instance handoff for Seti Astro Suite Pro.

When a second SASpro launch happens (e.g. user right-clicks a FITS file ->
"Open with -> SetiAstroSuitePro" while SASpro is already running), the second
process detects the running primary, hands over its argv (file paths, flags),
and exits silently. The already-running primary raises its window and opens
the file(s) through the doc manager.

Behavior:
  * Shared bucket policy, computed per-user:
      - Frozen / pip-installed builds share one server name.
      - Running-from-source checkouts share a separate server name.
      - SASPRO_INSTANCE=<tag> overrides and uses its own named bucket.
  * CLI flags:
      --new-instance     Force this launch to start its own instance.
      --single-instance  Force handoff even if SASPRO_INSTANCE is set.
    Env var:
      SASPRO_NEW_INSTANCE=1   Same as --new-instance.

Design:
  * The initial PROBE (does a primary exist?) uses stdlib
    multiprocessing.connection so it works before QApplication is created,
    with no Qt/event-loop cost on a cold second launch.
  * The LISTENER on the primary side uses QLocalServer so the handler fires
    on the Qt event loop, where it's safe to touch the main window.
  * JSON payload with a protocol version — future-you can bump the version
    and the receiver can fall back gracefully.
  * Stale socket cleanup on Linux/macOS so a crashed primary doesn't block
    the next launch.
"""
from __future__ import annotations

import getpass
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

# Bump when the wire format changes in a non-backward-compatible way.
PROTOCOL_VERSION = 1

# Short probe timeout — we want a snappy "no primary, launch normally".
PROBE_CONNECT_TIMEOUT_MS = 300
PROBE_WRITE_TIMEOUT_MS = 1000


# ───────────────────────────────────────────────────────────────────────────
# Argv parsing
# ───────────────────────────────────────────────────────────────────────────

class ParsedArgv:
    """Result of stripping single-instance flags from argv."""
    __slots__ = ("remaining", "force_new_instance", "force_single_instance")

    def __init__(self, remaining: list[str],
                 force_new_instance: bool,
                 force_single_instance: bool):
        self.remaining = remaining
        self.force_new_instance = force_new_instance
        self.force_single_instance = force_single_instance


def parse_instance_flags(argv: list[str] | None) -> ParsedArgv:
    """
    Strip --new-instance / --single-instance from argv.
    Returns the cleaned argv plus which flags were seen.
    Env var SASPRO_NEW_INSTANCE=1 also counts as --new-instance.
    """
    argv = list(argv or [])
    force_new = False
    force_single = False

    cleaned: list[str] = []
    for tok in argv:
        low = tok.lower()
        if low in ("--new-instance", "--newinstance"):
            force_new = True
            continue
        if low in ("--single-instance", "--singleinstance"):
            force_single = True
            continue
        cleaned.append(tok)

    env_new = os.environ.get("SASPRO_NEW_INSTANCE", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    if env_new:
        force_new = True

    # --single-instance explicitly overrides SASPRO_NEW_INSTANCE
    if force_single:
        force_new = False

    return ParsedArgv(cleaned, force_new, force_single)


# ───────────────────────────────────────────────────────────────────────────
# Server-name policy
# ───────────────────────────────────────────────────────────────────────────

def _running_from_source() -> bool:
    """
    True if we're running from a git checkout / editable install,
    False for PyInstaller-frozen bundles and normal pip installs into
    site-packages.
    """
    # Frozen bundle: always "installed".
    if getattr(sys, "frozen", False):
        return False
    # If this file sits inside a site-packages / dist-packages directory,
    # treat it as installed. Otherwise it's a source checkout.
    here = Path(__file__).resolve()
    parts_lower = [p.lower() for p in here.parts]
    for marker in ("site-packages", "dist-packages"):
        if marker in parts_lower:
            return False
    return True


def _bucket_tag() -> str:
    """
    Choose which "bucket" this process belongs to for the purpose of
    finding its peers. Processes in the same bucket find each other;
    different buckets stay isolated.

    Policy: everything shares one bucket by default — a source launch
    (python setiastrosuitepro.py) and the installed EXE will find each
    other, so right-clicking a FITS and "Open with -> SASpro.exe" hands
    off to whichever SASpro is currently running. For developer isolation,
    use the explicit opt-outs:
       --new-instance   (one-shot: this launch starts its own process)
       SASPRO_NEW_INSTANCE=1   (same, as an env var — set in a dev shell)
       SASPRO_INSTANCE=dev     (give this launch its own named bucket;
                                future launches with the same tag find it)
    """
    # Explicit override always wins — this is the "give me my own bucket" knob.
    override = os.environ.get("SASPRO_INSTANCE", "").strip()
    if override:
        # Sanitize: keep it to something safe for a socket/pipe name.
        safe = "".join(ch for ch in override if ch.isalnum() or ch in "-_")
        if safe:
            return safe
    return "shared"


def _username_safe() -> str:
    try:
        u = getpass.getuser()
    except Exception:
        u = "user"
    return "".join(ch for ch in u if ch.isalnum() or ch in "-_") or "user"


def server_name() -> str:
    """
    Named-pipe / local-socket name shared by all SASpro processes in this
    bucket for this user. Versioned so a future v4 can't mis-handshake with
    a v3 still running.
    """
    return f"SASpro-v3-{_username_safe()}-{_bucket_tag()}"


# On POSIX, QLocalServer / mp.connection sockets live under a filesystem path.
# We standardize on $TMPDIR so stale cleanup is predictable.
def _posix_socket_path(name: str) -> str:
    return str(Path(tempfile.gettempdir()) / name)


# multiprocessing.connection address: on Windows this is a named pipe,
# on POSIX a filesystem path for an AF_UNIX socket.
def _mp_address(name: str):
    if sys.platform.startswith("win"):
        return r"\\.\pipe\{}".format(name)
    return _posix_socket_path(name)


# ───────────────────────────────────────────────────────────────────────────
# Probe side (secondary launch)
# ───────────────────────────────────────────────────────────────────────────

def _build_payload(argv_remaining: list[str], cwd: str) -> bytes:
    """
    Wire payload sent from secondary to primary. We send the raw argv
    (minus instance flags) plus cwd, and let the primary decide what to
    do with it — the primary already has _collect_open_paths() to pull
    file paths out of argv.
    """
    body = {
        "v": PROTOCOL_VERSION,
        "argv": list(argv_remaining),
        "cwd": cwd,
        "pid": os.getpid(),
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _try_handoff_posix(name: str, payload: bytes) -> bool:
    """POSIX handoff via AF_UNIX socket."""
    addr = _posix_socket_path(name)
    if not os.path.exists(addr):
        return False
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(PROBE_CONNECT_TIMEOUT_MS / 1000.0)
        s.connect(addr)
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        # Primary not actually listening — stale socket, clean it up so the
        # next launch doesn't repeat this dance.
        try:
            os.unlink(addr)
        except OSError:
            pass
        s.close()
        return False
    try:
        s.settimeout(PROBE_WRITE_TIMEOUT_MS / 1000.0)
        # Length-prefix with a newline so the server can read until \n.
        s.sendall(payload + b"\n")
        # Optional: wait briefly for an ack so we know the primary read us.
        try:
            s.recv(16)
        except socket.timeout:
            pass
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _try_handoff_windows(name: str, payload: bytes) -> bool:
    """Windows handoff via named pipe."""
    pipe_path = r"\\.\pipe\{}".format(name)
    try:
        # Open in binary mode. If the pipe doesn't exist, this raises FileNotFoundError.
        # If it exists but is busy, we get OSError with winerror 231 — treat as "no primary"
        # for simplicity (very rare in practice; primary accepts fast).
        with open(pipe_path, "r+b", buffering=0) as pipe:
            pipe.write(payload + b"\n")
            pipe.flush()
            try:
                _ = pipe.read(16)
            except OSError:
                pass
        return True
    except (FileNotFoundError, OSError):
        return False


def try_handoff(argv_remaining: list[str]) -> bool:
    """
    Try to hand our argv off to an already-running SASpro primary.
    Returns True if the primary accepted the handoff (caller should exit),
    False if no primary was reachable (caller should become the primary).
    """
    name = server_name()
    payload = _build_payload(argv_remaining, os.getcwd())
    try:
        if sys.platform.startswith("win"):
            return _try_handoff_windows(name, payload)
        return _try_handoff_posix(name, payload)
    except Exception:
        # Any unexpected failure => fall through to becoming the primary.
        # Better to launch a second instance than to crash on startup.
        return False


def maybe_handoff_and_exit(argv: list[str] | None) -> Optional[list[str]]:
    """
    Called at the very top of gui_entry.main().

    Returns:
      * None  — we handed the argv off to an existing primary; caller
                should exit immediately (SystemExit(0)).
      * list  — we are (or will become) the primary. This is the cleaned
                argv with --new-instance / --single-instance stripped;
                caller should use this in place of its original argv.
    """
    parsed = parse_instance_flags(argv)

    # User explicitly wants a new instance — skip the probe entirely.
    if parsed.force_new_instance:
        return parsed.remaining

    if try_handoff(parsed.remaining):
        # Primary accepted. We're done.
        return None

    # No primary (or probe failed). We're the primary now.
    return parsed.remaining


# ───────────────────────────────────────────────────────────────────────────
# Listener side (primary)
# ───────────────────────────────────────────────────────────────────────────

def _cleanup_stale_posix_socket(name: str) -> None:
    """
    On POSIX, a crashed primary leaves its socket file behind, which blocks
    QLocalServer.listen(). If nothing is actually listening on it, remove it.
    """
    if sys.platform.startswith("win"):
        return
    addr = _posix_socket_path(name)
    if not os.path.exists(addr):
        return
    # Probe: can we connect? If yes, a real primary is already running —
    # don't touch the socket. (Caller shouldn't be here in that case, but
    # be defensive.)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.1)
        s.connect(addr)
        s.close()
        return  # live primary — leave the socket alone
    except OSError:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass
    try:
        os.unlink(addr)
    except OSError:
        pass


def install_primary_listener(
    app,
    on_payload: Callable[[dict], None],
    status_cb: Optional[Callable[[str], None]] = None,
):
    """
    Start the QLocalServer on the primary. When another SASpro launch
    hands us a payload, we decode it and call on_payload(dict) on the
    Qt main thread.

    on_payload receives the full decoded JSON dict:
      {"v": 1, "argv": [...], "cwd": "...", "pid": N}

    Caller is responsible for pulling file paths out of argv (reuse
    _collect_open_paths from gui_entry) and raising the main window.

    Returns the QLocalServer instance (keep a reference so it isn't GC'd).
    """
    from PyQt6.QtCore import QTimer
    from PyQt6.QtNetwork import QLocalServer, QLocalSocket

    name = server_name()
    _cleanup_stale_posix_socket(name)

    server = QLocalServer()
    # World-only on POSIX; Windows ignores this.
    try:
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    except Exception:
        pass

    if not server.listen(name):
        # One retry after a forced removeServer — handles the case where
        # our stale check missed it (very old Qt, odd FS state).
        try:
            QLocalServer.removeServer(name)
        except Exception:
            pass
        if not server.listen(name):
            if status_cb:
                status_cb(f"[single-instance] could not listen on {name!r}: "
                          f"{server.errorString()}")
            return None

    if status_cb:
        status_cb(f"[single-instance] primary listening on {name!r}")

    def _drain_socket(sock: QLocalSocket):
        try:
            # Read all available bytes, bounded by a sanity limit.
            sock.waitForReadyRead(500)
            data = bytes(sock.readAll())
            if not data:
                return
            # Payload is one JSON object terminated by \n. If multiple
            # launches piled up on the same socket pass, split on \n.
            for chunk in data.split(b"\n"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    body = json.loads(chunk.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if status_cb:
                        status_cb("[single-instance] bad payload, ignoring")
                    continue

                # Version check: unknown future version — be conservative
                # and ignore rather than acting on fields we don't understand.
                v = body.get("v")
                if not isinstance(v, int) or v > PROTOCOL_VERSION:
                    if status_cb:
                        status_cb(f"[single-instance] unknown protocol v={v!r}, "
                                  "ignoring handoff")
                    continue

                # Ack so the secondary knows we received it.
                try:
                    sock.write(b"OK\n")
                    sock.flush()
                except Exception:
                    pass

                # Route into the main thread's event loop. Even though
                # QLocalServer fires on the GUI thread already, going
                # through singleShot guarantees we're not re-entering any
                # mid-paint / mid-event state.
                QTimer.singleShot(0, lambda b=body: on_payload(b))
        finally:
            try:
                sock.disconnectFromServer()
            except Exception:
                pass

    def _on_new_connection():
        while True:
            sock = server.nextPendingConnection()
            if sock is None:
                break
            # Lifetime: sock is parented to server, auto-cleaned up.
            sock.readyRead.connect(lambda s=sock: _drain_socket(s))
            # If data is already buffered, kick it manually.
            if sock.bytesAvailable() > 0:
                _drain_socket(sock)

    server.newConnection.connect(_on_new_connection)
    return server


# ───────────────────────────────────────────────────────────────────────────
# Window-raise helper for the primary
# ───────────────────────────────────────────────────────────────────────────

def raise_window_to_front(win) -> None:
    """
    Bring an existing main window to the foreground.

    On Windows, SetForegroundWindow is restricted (an app can only steal
    focus from itself or if the user recently interacted with it). The
    AllowSetForegroundWindow(ASFW_ANY) + activateWindow() combo is the
    standard workaround; worst case the window flashes in the taskbar
    instead of popping forward, which is still better than silently
    opening a file in a hidden window.
    """
    try:
        from PyQt6.QtCore import Qt
    except Exception:
        return

    try:
        # Un-minimize if needed.
        state = win.windowState()
        if state & Qt.WindowState.WindowMinimized:
            win.setWindowState(
                (state & ~Qt.WindowState.WindowMinimized)
                | Qt.WindowState.WindowActive
            )

        win.show()
        win.raise_()
        win.activateWindow()
    except Exception:
        pass

    # Windows-specific focus-steal nudge.
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ASFW_ANY = -1
            ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY)
            hwnd = int(win.winId())
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass