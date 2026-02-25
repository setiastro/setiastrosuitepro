# src/setiastro/saspro/cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from setiastro.saspro.cosmicclarity_headless import run_cosmicclarity_on_file


# ───────────────────────────────────────────────────────────────
# GUI dispatch (you already have a main entrypoint elsewhere)
# ───────────────────────────────────────────────────────────────
def _launch_gui(open_paths: list[str] | None = None) -> int:
    """
    Launch SASpro GUI. If open_paths provided, open them on startup.
    Implement this using your existing __main__.py entrypoint / main_window logic.
    """
    from setiastro.saspro.gui_entry import main as gui_main
    return int(gui_main(open_paths) or 0)   # open_paths is argv-style list


def _looks_like_path_token(tok: str) -> bool:
    if not tok or tok.startswith("-"):
        return False
    try:
        p = Path(tok)
        return p.exists()
    except Exception:
        return False


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
    p.add_argument("--sharpening-mode", default="Both", choices=["Both", "Stellar Only", "Non-Stellar Only"])
    p.add_argument("--stellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-psf", type=float, default=3.0)
    p.add_argument("--auto-psf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sharpen-channels-separately", action="store_true", default=False)

    p = sub.add_parser("denoise", help="Denoise only")
    _add_common_io(p)
    _add_chunking_opts(p)
    p.add_argument("--denoise-luma", type=float, default=0.5)
    p.add_argument("--denoise-color", type=float, default=0.5)
    p.add_argument("--denoise-mode", default="full", choices=["full", "luminance"])
    p.add_argument("--separate-channels", action="store_true", default=False)

    p = sub.add_parser("both", help="Sharpen then denoise")
    _add_common_io(p)
    _add_chunking_opts(p)
    p.add_argument("--sharpening-mode", default="Both", choices=["Both", "Stellar Only", "Non-Stellar Only"])
    p.add_argument("--stellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-psf", type=float, default=3.0)
    p.add_argument("--auto-psf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sharpen-channels-separately", action="store_true", default=False)
    p.add_argument("--denoise-luma", type=float, default=0.5)
    p.add_argument("--denoise-color", type=float, default=0.5)
    p.add_argument("--denoise-mode", default="full", choices=["full", "luminance"])
    p.add_argument("--separate-channels", action="store_true", default=False)

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

    return ap


def _run_cc(argv: list[str]) -> int:
    args = build_cc_parser().parse_args(argv)

    inp = str(Path(args.input))
    out = str(Path(args.output))

    # always available (we added chunking to every subcommand)
    preset = {
        "gpu": bool(args.gpu),
        "chunk_size": int(getattr(args, "chunk_size", 256)),
        "overlap": int(getattr(args, "overlap", 64)),
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
        })
    elif args.cmd == "denoise":
        preset.update({
            "mode": "denoise",
            "denoise_luma": args.denoise_luma,
            "denoise_color": args.denoise_color,
            "denoise_mode": args.denoise_mode,
            "separate_channels": bool(args.separate_channels),
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
            "denoise_luma": args.denoise_luma,
            "denoise_color": args.denoise_color,
            "denoise_mode": args.denoise_mode,
            "separate_channels": bool(args.separate_channels),
        })
    elif args.cmd == "superres":
        preset.update({"mode": "superres", "scale": int(args.scale)})
    elif args.cmd == "satellite":
        preset.update({
            "mode": "satellite",
            "sat_mode": args.sat_mode,
            "sat_clip_trail": bool(args.clip_trail),
            "sat_sensitivity": float(args.sensitivity),
        })
    else:
        raise RuntimeError(f"Unknown cmd: {args.cmd}")

    run_cosmicclarity_on_file(inp, out, preset, progress_cb=_progress_print)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        return _launch_gui()

    if argv and argv[0].lower() in ("cc", "cosmicclarity"):
        return _run_cc(argv[1:] or ["--help"])

    if _looks_like_path_token(argv[0]):
        paths = [str(Path(a)) for a in argv if _looks_like_path_token(a)]
        return _launch_gui(open_paths=paths)

    known = {"sharpen", "denoise", "both", "superres", "satellite", "--help", "-h"}
    if argv[0].lower() in known:
        return _run_cc(argv)

    return _launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
