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

    # other CLI tools you may add
    "benchmark",
    "report",
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
    return int(gui_main(argv))

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