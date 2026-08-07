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

    from setiastro.saspro.gui_entry import main as gui_main
    _minimize_console_if_owned()
    return int(gui_main(argv))

def main(argv: list[str] | None = None) -> int:
    return entry(argv)

if __name__ == "__main__":
    raise SystemExit(entry())
