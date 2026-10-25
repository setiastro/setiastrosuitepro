# src/setiastro/saspro/__main__.py
from __future__ import annotations
import sys, ctypes
import warnings

# Suppress torch optree version warning — SASpro does not use torch.compile()
# and the C++ pytree backend it guards has no effect on our inference paths.
warnings.filterwarnings(
    "ignore",
    message="optree is installed but the version is too old",
    category=FutureWarning,
    module=r"torch\.utils\._pytree",
)

CLI_SUBCOMMANDS = {
    # wrapper aliases
    "cosmicclarity", "cc",

    # cosmicclarity subcommands
    "sharpen", "denoise", "both", "superres", "satellite",

    # plate solving (headless; blind-capable)
    "solve", "platesolve",

    # other CLI tools you may add
    "benchmark",
    "report",

    # Help and version. These MUST be here, not only in cli.py: everything
    # below this check goes through the single-instance probe first, so
    # "setiastrosuitepro --help" with a window already open would hand the
    # argv to the running instance and exit silently, printing nothing.
    "-h", "--help", "--h", "-help", "help", "-?", "/?", "/h",
    "--version", "version",
}

def _minimize_console_if_owned() -> None:
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    user32   = ctypes.windll.user32
    kernel32.GetConsoleWindow.restype = ctypes.c_void_p
    user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.ShowWindow.restype  = ctypes.c_bool

    hwnd = kernel32.GetConsoleWindow()
    if not hwnd:
        return  # no console (console=False build) — nothing to do

    # Only minimize a console we EXCLUSIVELY own (PyInstaller spawned it for
    # us on a double-click). If cmd / pwsh / VS Code share it, leave it alone.
    # Buffer sized generously; we only care about the ==1 case regardless.
    buf = (ctypes.c_uint * 16)()
    count = kernel32.GetConsoleProcessList(buf, 16)
    if count == 1:
        user32.ShowWindow(hwnd, 7)  # SW_SHOWMINNOACTIVE


def _report_startup_crash(tb: str) -> None:
    """Last-chance crash reporting for failures during GUI startup.

    gui_entry.main() configures file logging only AFTER _bootstrap_imports()
    returns, so anything that dies during the heavy-import phase leaves no log
    entry at all. On a double-click launch _minimize_console_if_owned() has
    also just hidden the console, so the user sees the splash flash and the
    app vanish with nothing to report.

    Write the traceback somewhere findable and show a native MessageBox (via
    ctypes, so it works even when PyQt6 is the thing that failed to import).
    """
    import os, tempfile
    from pathlib import Path

    log_path = None
    try:
        if sys.platform.startswith("win"):
            d = Path(os.path.expandvars("%APPDATA%")) / "SetiAstroSuitePro" / "logs"
        elif sys.platform.startswith("darwin"):
            d = Path.home() / "Library" / "Logs" / "SetiAstroSuitePro"
        else:
            d = Path.home() / ".local" / "share" / "SetiAstroSuitePro" / "logs"
        d.mkdir(parents=True, exist_ok=True)
        log_path = d / "startup-crash.log"
        log_path.write_text(tb, encoding="utf-8")
    except Exception:
        try:
            log_path = Path(tempfile.gettempdir()) / "saspro-startup-crash.log"
            log_path.write_text(tb, encoding="utf-8")
        except Exception:
            log_path = None

    # Always dump to stderr too — covers the `./setiastrosuitepro.exe` from a
    # shared terminal case, where the console was deliberately left visible.
    try:
        print(tb, file=sys.stderr)
    except Exception:
        pass

    try:
        tail = "\r\n".join(tb.strip().splitlines()[-3:])
    except Exception:
        tail = "(no traceback available)"

    body = (
        "Seti Astro Suite Pro failed to start.\r\n\r\n"
        f"{tail}\r\n\r\n"
        + (f"Full details were written to:\r\n{log_path}\r\n\r\n" if log_path else "")
        + "Please send that file to the SetiAstro Discord or open a GitHub issue."
    )

    if sys.platform.startswith("win"):
        # MB_ICONERROR (0x10) | MB_SETFOREGROUND (0x10000) | MB_TOPMOST (0x40000)
        try:
            ctypes.windll.user32.MessageBoxW(
                0, body, "Seti Astro Suite Pro - Startup Failure", 0x50010
            )
        except Exception:
            pass


def entry(argv: list[str] | None = None) -> int:
    # Catch multiprocessing spawn children FIRST — before anything else runs.
    # In a frozen build, every ProcessPool worker re-execs this EXE and ends
    # up back in entry() (the shim's `raise SystemExit(entry())`). This call
    # takes the child over here and never returns, so a worker is guaranteed
    # not to be mistaken for a secondary GUI launch by the handoff probe below.
    import multiprocessing as _mp
    _mp.freeze_support()

    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0].lower() in CLI_SUBCOMMANDS:
        from setiastro.saspro.cli import main as cli_main

        # IMPORTANT: "cc" / "cosmicclarity" are *dispatch aliases*, not actual CLI commands.
        head = argv[0].lower()
        if head in ("cc", "cosmicclarity"):
            argv = argv[1:]  # drop alias so cli.py sees "sharpen|denoise|both|..."
            if not argv:
                argv = ["--help"]  # "python -m ... cc" shows help instead of error

        return int(cli_main(argv))

    # ── Single-instance handoff (earliest possible point) ───────────────────
    # If another SASpro is already running (and the user didn't pass
    # --new-instance / set SASPRO_NEW_INSTANCE=1), hand our argv off and exit
    # HERE — BEFORE importing gui_entry.py. That import triggers the module-
    # level load of PyQt6, numba_bootstrap, and the versioning module, which
    # collectively add ~500ms-1s on a cold secondary launch. For a right-click
    # "Open With -> SASpro" that's the pause the user sees between the console
    # flashing up and the file appearing in the already-running instance.
    #
    # single_instance.py is stdlib-only (socket, json, tempfile, pathlib), so
    # importing it here is virtually free. If the probe succeeds, PyQt6 never
    # loads in this process at all and we exit in ~50ms + Python startup.
    try:
        from setiastro.saspro.single_instance import maybe_handoff_and_exit
        _cleaned = maybe_handoff_and_exit(argv)
        if _cleaned is None:
            # Primary accepted our handoff. Secondary exits silently.
            return 0
        # Strip --new-instance / --single-instance from argv for downstream.
        argv = _cleaned
    except Exception:
        # A probe failure must never block launch — fall through to a normal
        # full-fat start. Rare, but possible on an unusual filesystem or if
        # the socket layer is broken.
        import traceback as _tb
        print("[single-instance] early probe failed, launching normally:")
        _tb.print_exc()

    from setiastro.saspro.gui_entry import main as gui_main
    _minimize_console_if_owned()
    # We already did the single-instance handoff above with the raw (unstripped)
    # argv, so pass _skip_handoff=True to stop gui_entry.main from running its
    # OWN handoff on the cleaned argv — that second call would re-probe the
    # running primary without --new-instance in its view of the args, hand off
    # again, and silently exit the new instance the user actually asked for.
    #
    # Wrapped: gui_entry's own try/except only covers main-window construction,
    # which is well after _bootstrap_imports(). An import-time failure in the
    # bootstrap (a missing VC++ runtime, a quarantined .pyd, a third-party
    # package blowing up at import) propagates all the way out to here, and
    # without this handler the frozen EXE dies with a hidden console and no log.
    # SystemExit re-raises untouched so the normal sys.exit(_app.exec()) path is
    # unaffected; BaseException catches KeyboardInterrupt during a long cold
    # start as well, which is worth logging rather than swallowing.
    try:
        return int(gui_main(argv, _skip_handoff=True))
    except SystemExit:
        raise
    except BaseException:
        import traceback as _tb
        _report_startup_crash(_tb.format_exc())
        return 1

def main(argv: list[str] | None = None) -> int:
    return entry(argv)

if __name__ == "__main__":
    # Frozen ProcessPool workers re-exec this EXE; freeze_support() makes them
    # act as workers instead of launching new SASpro windows. entry() calls
    # freeze_support() itself now (see above) so this is defense-in-depth for
    # the `python -m setiastro.saspro` path specifically.
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(entry())