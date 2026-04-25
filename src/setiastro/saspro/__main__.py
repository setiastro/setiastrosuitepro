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
}

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
    return int(gui_main(argv))

def main(argv: list[str] | None = None) -> int:
    return entry(argv)

if __name__ == "__main__":
    raise SystemExit(entry())
