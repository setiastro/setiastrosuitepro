# src/setiastro/saspro/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ───────────────────────────────────────────────────────────────
# GUI dispatch (you already have a main entrypoint elsewhere)
# ───────────────────────────────────────────────────────────────
def _launch_gui(open_paths: list[str] | None = None) -> int:
    """
    Launch SASpro GUI. If open_paths provided, open them on startup.
    Implement this using your existing __main__.py entrypoint / main_window logic.
    """
    from setiastro.saspro.gui_entry import main as gui_main
    return int(gui_main(open_paths) or 0)


def _looks_like_path_token(tok: str) -> bool:
    if not tok or tok.startswith("-"):
        return False
    try:
        p = Path(tok)
        return p.exists()
    except Exception:
        return False


# ───────────────────────────────────────────────────────────────
# Top-level help
# ───────────────────────────────────────────────────────────────
HELP_TOKENS = {"-h", "--help", "--h", "-help", "help", "-?", "/?", "/h"}
VERSION_TOKENS = {"--version", "version"}


def _cc_subcommands() -> list[str]:
    """Read the CC subcommand list off its own parser so it cannot drift."""
    try:
        for a in build_cc_parser()._actions:
            if isinstance(a, argparse._SubParsersAction):
                return sorted(a.choices)
    except Exception:
        pass
    return ["both", "correct", "darkstar", "denoise", "satellite",
            "sharpen", "superres"]


def _version_string() -> str:
    for mod, attrs in (("setiastro.saspro.versioning",
                        ("__version__", "VERSION", "version")),
                       ("setiastro.saspro", ("__version__",))):
        try:
            m = __import__(mod, fromlist=["*"])
            for a in attrs:
                v = getattr(m, a, None)
                if isinstance(v, str) and v:
                    return v
        except Exception:
            pass
    try:
        from importlib.metadata import version
        return version("setiastrosuitepro")
    except Exception:
        return "unknown"


def top_level_help() -> str:
    # Wrapped rather than joined: the list is generated from the CC parser, so
    # it grows every time a subcommand is added and would otherwise run off
    # the page. The whole line is wrapped, label included, because the label
    # takes 13 of the first line's columns.
    import textwrap
    cc = textwrap.fill(f"Subcommands: {', '.join(_cc_subcommands())}",
                       width=78, initial_indent=" " * 15,
                       subsequent_indent=" " * 15).lstrip()
    return f"""Seti Astro Suite Pro \u2014 command line

USAGE
  setiastrosuitepro                          launch the GUI
  setiastrosuitepro IMAGE [IMAGE ...]        launch the GUI with files opened
  setiastrosuitepro <command> [options]      run a command headlessly
  setiastrosuitepro <command> --help         full options for that command

COMMANDS
  solve        Plate-solve images. In-house Gaia DR3 solver (blind or seeded),
               falling back to ASTAP and Astrometry.net. Blind solving needs no
               coordinates at all \u2014 it identifies the field from the star
               pattern using a quad index built from your local Gaia data.
               Can write its result as a .wcs sidecar or an ASTAP-style .ini,
               so it can stand in for ASTAP in imaging software.
               Aliases: platesolve

  cc           Cosmic Clarity image processing, in-process.
               {cc}
               These may be run directly, with or without the leading "cc".
               Aliases: cosmicclarity

  report       Write a diagnostics report (versions, paths, GPU, libraries).
               Attach its output to bug reports.

EXAMPLES
  Solve one image, writing a new FITS:
    setiastrosuitepro solve -i light.fits -o light_solved.fits

  Solve with no coordinates in the header (blind):
    setiastrosuitepro solve -i unknown.fits -o solved.fits

  The header has coordinates but they are WRONG:
    setiastrosuitepro solve -i light.fits --in-place --force-blind

  Solve a whole folder, machine-readable output:
    setiastrosuitepro solve -i ./lights -o ./solved --batch --json

  Write a .wcs sidecar and leave the image untouched:
    setiastrosuitepro solve -i light.fit --wcs-only

  Write an ASTAP-style .ini instead — what N.I.N.A. and other
  ASTAP-driven software read a solve back from:
    setiastrosuitepro solve -i light.fit --ini-only

  Which blind-solve index layers are built:
    setiastrosuitepro solve --index-status

  Denoise an image on the GPU:
    setiastrosuitepro cc denoise -i in.tif -o out.tif --denoise-luma 0.6

  Diagnostics for a bug report:
    setiastrosuitepro report --save

NOTES
  Exit codes are 0 on success, 1 when the work failed, 2 for a usage or I/O
  error, so these commands can be driven from a script.

  Run "setiastrosuitepro <command> --help" for the full option list of any
  command \u2014 this page is only the overview.
"""


def _print_top_level_help(stream=None) -> int:
    (stream or sys.stdout).write(top_level_help())
    return 0


# ───────────────────────────────────────────────────────────────
# Existing CC CLI
# ───────────────────────────────────────────────────────────────
def _add_common_io(p: argparse.ArgumentParser):
    p.add_argument("-i", "--input", required=True, help="Input image path")
    p.add_argument("-o", "--output", required=True, help="Output image path")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True, help="Enable GPU if available")


def _add_chunking_opts(p: argparse.ArgumentParser):
    """
    Chunking controls for tiled inference.
    Defaults match current engines.
    """
    p.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Tile size for processing (e.g. 256, 384, 512). Larger = faster but more VRAM/RAM."
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Tile overlap in pixels to reduce seams (typically chunk_size/4)."
    )


def _add_temp_stretch_opts(p: argparse.ArgumentParser):
    p.add_argument(
        "--temp-stretch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily stretch linear data before AI processing, then unstretch after."
    )
    p.add_argument(
        "--target-median",
        type=float,
        default=0.25,
        help="Target median used for temporary stretch (default: 0.25)."
    )

def _add_darkstar_opts(p: argparse.ArgumentParser):
    p.add_argument(
        "--star-removal-mode",
        dest="darkstar_mode",
        default="unscreen",
        choices=["unscreen", "additive"],
        help="DarkStar extraction mode."
    )
    p.add_argument(
        "--processing-path",
        dest="darkstar_processing_path",
        default="hybrid_luma_color",
        choices=["mono_per_channel", "hybrid_luma_color", "color_only"],
        help="DarkStar processing path."
    )
    p.add_argument(
        "--show-extracted-stars",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also produce/preserve the extracted-stars result when supported."
    )
    p.add_argument(
        "--edge-padding",
        type=int,
        default=64,
        help="Reflect-padding applied before DarkStar to reduce edge artifacts."
    )
    p.add_argument(
        "--compatibility-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DarkStar GPU compatibility mode."
    )


def _progress_print(done: int, total: int) -> bool:
    pct = int(0 if total <= 0 else (100 * done / total))
    print(f"PROGRESS: {pct}%", flush=True)
    return True


def build_cc_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="setiastrosuitepro cc",
        description="SetiAstro Cosmic Clarity (in-process) CLI",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sharpen", help="Sharpen only")
    _add_common_io(p)
    _add_chunking_opts(p)
    _add_temp_stretch_opts(p)
    p.add_argument("--sharpening-mode", default="Both", choices=["Both", "Stellar Only", "Non-Stellar Only"])
    p.add_argument("--stellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-psf", type=float, default=3.0)
    p.add_argument("--auto-psf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sharpen-channels-separately", action="store_true", default=False)
    p.add_argument(
        "--stellar-correct-mode",
        default="sharpen_only",
        choices=["sharpen_only", "correct_only", "correct_sharpen"],
        help="Stellar correction mode: sharpen_only (default), correct_only (aberration correction, no sharpen), "
             "or correct_sharpen (aberration correction then sharpen)."
    )

    p = sub.add_parser("correct", help="Aberration correction only (no sharpening)")
    _add_common_io(p)
    _add_chunking_opts(p)
    _add_temp_stretch_opts(p)

    p = sub.add_parser("denoise", help="Denoise only")
    _add_common_io(p)
    _add_chunking_opts(p)
    _add_temp_stretch_opts(p)
    p.add_argument("--denoise-luma", type=float, default=0.5)
    p.add_argument("--denoise-color", type=float, default=0.5)
    p.add_argument("--denoise-mode", default="full", choices=["full", "luminance"])
    p.add_argument("--separate-channels", action="store_true", default=False)
    p.add_argument("--denoise-walking", action="store_true", default=False,
                   help="Use Walking Noise specialist model instead of standard.")
    p.add_argument("--denoise-lite", action="store_true", default=False,
                   help="Use Lite (smaller/faster) denoise model.")
    
    p = sub.add_parser("both", help="Sharpen then denoise")
    _add_common_io(p)
    _add_chunking_opts(p)
    _add_temp_stretch_opts(p)
    p.add_argument("--sharpening-mode", default="Both", choices=["Both", "Stellar Only", "Non-Stellar Only"])
    p.add_argument("--stellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-psf", type=float, default=3.0)
    p.add_argument("--auto-psf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sharpen-channels-separately", action="store_true", default=False)
    p.add_argument(
        "--stellar-correct-mode",
        default="sharpen_only",
        choices=["sharpen_only", "correct_only", "correct_sharpen"],
        help="Stellar correction mode for the sharpen pass."
    )
    p.add_argument("--denoise-luma", type=float, default=0.5)
    p.add_argument("--denoise-color", type=float, default=0.5)
    p.add_argument("--denoise-mode", default="full", choices=["full", "luminance"])
    p.add_argument("--separate-channels", action="store_true", default=False)
    p.add_argument("--denoise-walking", action="store_true", default=False,
                   help="Use Walking Noise specialist model instead of standard.")
    p.add_argument("--denoise-lite", action="store_true", default=False,
                   help="Use Lite (smaller/faster) denoise model.")
    
    p = sub.add_parser("superres", help="Super resolution")
    _add_common_io(p)
    _add_chunking_opts(p)
    p.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])

    p = sub.add_parser("satellite", help="Satellite trail removal")
    _add_common_io(p)
    _add_chunking_opts(p)
    p.add_argument("--mode", dest="sat_mode", default="full", choices=["full", "luminance"])
    p.add_argument("--clip-trail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sensitivity", type=float, default=0.10)
    p.add_argument("--skip-save", action="store_true", default=False,
                   help="Skip saving output if no satellite trail is detected.")

    p = sub.add_parser("darkstar", help="DarkStar star removal")
    _add_common_io(p)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="DarkStar chunk size."
    )
    p.add_argument(
        "--overlap-frac",
        type=float,
        default=0.125,
        help="DarkStar chunk overlap fraction."
    )
    _add_darkstar_opts(p)

    return ap


def _run_cc(argv: list[str]) -> int:
    # Parse FIRST. argparse exits during parse_args for --help and for a bad
    # option, so importing the engine above this line made `cc --help` pay for
    # loading torch, and made it fail outright on an install where the engine
    # is missing -- printing help is exactly when you can least afford that.
    args = build_cc_parser().parse_args(argv)

    from setiastro.saspro.cosmicclarity_headless import run_cosmicclarity_on_file

    inp = str(Path(args.input))
    out = str(Path(args.output))

    preset = {
        "gpu": bool(args.gpu),
        "chunk_size": int(getattr(args, "chunk_size", 256)),
        "overlap": int(getattr(args, "overlap", 64)),
        "temp_stretch": bool(getattr(args, "temp_stretch", False)),
        "target_median": float(getattr(args, "target_median", 0.25)),
    }

    if args.cmd == "sharpen":
        preset.update({
            "mode": "sharpen",
            "sharpening_mode": args.sharpening_mode,
            "stellar_amount": args.stellar_amount,
            "nonstellar_amount": args.nonstellar_amount,
            "nonstellar_psf": args.nonstellar_psf,
            "auto_psf": bool(args.auto_psf),
            "sharpen_channels_separately": bool(args.sharpen_channels_separately),
            "stellar_correct_mode": str(args.stellar_correct_mode),
        })
    elif args.cmd == "correct":
        preset.update({
            "mode": "sharpen",
            "stellar_correct_mode": "correct_only",
            "sharpening_mode": "Both",
            "stellar_amount": 0.5,
            "nonstellar_amount": 0.5,
            "nonstellar_psf": 3.0,
            "auto_psf": True,
            "sharpen_channels_separately": False,
        })
    elif args.cmd == "denoise":
        preset.update({
            "mode": "denoise",
            "denoise_luma": args.denoise_luma,
            "denoise_color": args.denoise_color,
            "denoise_mode": args.denoise_mode,
            "separate_channels": bool(args.separate_channels),
            "denoise_walking": bool(args.denoise_walking),
            "denoise_lite":    bool(args.denoise_lite),
        })

    elif args.cmd == "both":
        preset.update({
            "mode": "both",
            "sharpening_mode": args.sharpening_mode,
            "stellar_amount": args.stellar_amount,
            "nonstellar_amount": args.nonstellar_amount,
            "nonstellar_psf": args.nonstellar_psf,
            "auto_psf": bool(args.auto_psf),
            "sharpen_channels_separately": bool(args.sharpen_channels_separately),
            "stellar_correct_mode": str(args.stellar_correct_mode),
            "denoise_luma": args.denoise_luma,
            "denoise_color": args.denoise_color,
            "denoise_mode": args.denoise_mode,
            "separate_channels": bool(args.separate_channels),
            "denoise_walking": bool(args.denoise_walking),
            "denoise_lite":    bool(args.denoise_lite),
        })

    elif args.cmd == "superres":
        preset.update({
            "mode": "superres",
            "scale": int(args.scale),
        })
    elif args.cmd == "satellite":
        preset.update({
            "mode": "satellite",
            "sat_mode": args.sat_mode,
            "sat_clip_trail": bool(args.clip_trail),
            "sat_sensitivity": float(args.sensitivity),
            "sat_skip_save": bool(args.skip_save),
        })
    elif args.cmd == "darkstar":
        preset.update({
            "mode": "darkstar",
            "darkstar_mode": str(args.darkstar_mode),
            "processing_path": str(args.darkstar_processing_path),
            "show_extracted_stars": bool(args.show_extracted_stars),
            "chunk_size": int(args.chunk_size),
            "stride": int(args.chunk_size),  # legacy compatibility with replay/UI
            "overlap_frac": float(args.overlap_frac),
            "edge_padding": int(args.edge_padding),
            "compatibility_mode": bool(args.compatibility_mode),
        })
    else:
        raise RuntimeError(f"Unknown cmd: {args.cmd}")

    try:
        run_cosmicclarity_on_file(inp, out, preset, progress_cb=_progress_print)
        return 0
    except RuntimeError as e:
        msg = str(e)
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        return 2


def _run_solve(argv: list[str]) -> int:
    from setiastro.saspro.plate_solve_cli import main as solve_main
    return int(solve_main(argv))


def build_report_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="setiastrosuitepro report",
        description="Generate a SASpro diagnostics report",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="Also save the report to the default support/reports folder.",
    )
    ap.add_argument(
        "--output",
        help="Write the report to an explicit path.",
    )
    return ap


def _run_report(argv: list[str]) -> int:
    args = build_report_parser().parse_args(argv)

    from setiastro.saspro.diagnostics import collect_diagnostics, write_report

    report = collect_diagnostics()
    sys.stdout.write(report.markdown)
    if args.save or args.output:
        path = write_report(report.markdown, output_path=args.output)
        print(f"[report] wrote diagnostics report to: {path}", file=sys.stderr, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        return _launch_gui()

    head = argv[0].lower()

    # Top-level help must be checked BEFORE the CC aliases: "--help" used to
    # reach the Cosmic Clarity parser, which advertised only the CC commands
    # as though they were the whole CLI. "--h" matched nothing at all and fell
    # through to launching the GUI.
    if head in HELP_TOKENS and len(argv) == 1:
        return _print_top_level_help()

    if head in VERSION_TOKENS and len(argv) == 1:
        print(f"Seti Astro Suite Pro {_version_string()}")
        return 0

    if argv and argv[0].lower() in ("cc", "cosmicclarity"):
        return _run_cc(argv[1:] or ["--help"])

    if argv and argv[0].lower() in ("solve", "platesolve"):
        return _run_solve(argv[1:])

    if argv and argv[0].lower() == "report":
        return _run_report(argv[1:])

    if _looks_like_path_token(argv[0]):
        paths = [str(Path(a)) for a in argv if _looks_like_path_token(a)]
        return _launch_gui(open_paths=paths)

    if head in set(_cc_subcommands()) | HELP_TOKENS:
        return _run_cc(argv)

    # A leading dash is a flag, never a file. Opening the GUI on an unknown
    # flag loses the typo silently and looks like the command was accepted.
    if head.startswith("-"):
        print(f"setiastrosuitepro: unknown option: {argv[0]}\n", file=sys.stderr)
        _print_top_level_help(sys.stderr)
        return 2

    return _launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())