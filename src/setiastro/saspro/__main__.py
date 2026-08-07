# src/setiastro/saspro/__main__.py
from __future__ import annotations
import sys
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
    import ctypes
    kernel32 = ctypes.windll.kernel32
    hwnd = kernel32.GetConsoleWindow()
    if not hwnd:
        return  # no console (console=False build) — nothing to do
    # Only minimize a console we exclusively own — i.e. one PyInstaller
    # spawned for us. If other PIDs share it (cmd/pwsh/VS Code shell),
    # leave it alone so we don't hijack or un-hide someone's terminal.
    buf = (ctypes.c_uint * 4)()
    count = kernel32.GetConsoleProcessList(buf, 4)
    if count == 1:
        ctypes.windll.user32.ShowWindow(hwnd, 7)  # SW_SHOWMINNOACTIVE

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
