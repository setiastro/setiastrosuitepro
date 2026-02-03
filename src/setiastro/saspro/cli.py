from __future__ import annotations

import argparse
import sys
from pathlib import Path

from setiastro.saspro.cosmicclarity_headless import run_cosmicclarity_on_file


def _add_common_io(p: argparse.ArgumentParser):
    p.add_argument("-i", "--input", required=True, help="Input image path")
    p.add_argument("-o", "--output", required=True, help="Output image path")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True, help="Enable GPU if available")


def _progress_print(done: int, total: int) -> bool:
    # we use (pct,100) in our adapter
    pct = int(0 if total <= 0 else (100 * done / total))
    print(f"PROGRESS: {pct}%", flush=True)
    return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="setiastrosuitepro cc",
        description="SetiAstro Cosmic Clarity (in-process) CLI",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # sharpen
    p = sub.add_parser("sharpen", help="Sharpen only")
    _add_common_io(p)
    p.add_argument("--sharpening-mode", default="Both", choices=["Both", "Stellar Only", "Non-Stellar Only"])
    p.add_argument("--stellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-psf", type=float, default=3.0)
    p.add_argument("--auto-psf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sharpen-channels-separately", action="store_true", default=False)

    # denoise
    p = sub.add_parser("denoise", help="Denoise only")
    _add_common_io(p)
    p.add_argument("--denoise-luma", type=float, default=0.5)
    p.add_argument("--denoise-color", type=float, default=0.5)
    p.add_argument("--denoise-mode", default="full", choices=["full", "luminance"])
    p.add_argument("--separate-channels", action="store_true", default=False)

    # both
    p = sub.add_parser("both", help="Sharpen then denoise")
    _add_common_io(p)
    # sharpen args
    p.add_argument("--sharpening-mode", default="Both", choices=["Both", "Stellar Only", "Non-Stellar Only"])
    p.add_argument("--stellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-amount", type=float, default=0.5)
    p.add_argument("--nonstellar-psf", type=float, default=3.0)
    p.add_argument("--auto-psf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sharpen-channels-separately", action="store_true", default=False)
    # denoise args
    p.add_argument("--denoise-luma", type=float, default=0.5)
    p.add_argument("--denoise-color", type=float, default=0.5)
    p.add_argument("--denoise-mode", default="full", choices=["full", "luminance"])
    p.add_argument("--separate-channels", action="store_true", default=False)

    # superres
    p = sub.add_parser("superres", help="Super resolution")
    _add_common_io(p)
    p.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])

    # satellite
    p = sub.add_parser("satellite", help="Satellite trail removal")
    _add_common_io(p)
    p.add_argument("--mode", dest="sat_mode", default="full", choices=["full", "luminance"])
    p.add_argument("--clip-trail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sensitivity", type=float, default=0.10)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Allow wrapper tokens if someone passes them in anyway
    if argv and argv[0].lower() in ("cc", "cosmicclarity"):
        argv = argv[1:]
        if not argv:
            argv = ["--help"]

    args = build_parser().parse_args(argv)

    inp = str(Path(args.input))
    out = str(Path(args.output))

    # build preset dict using YOUR canonical keys
    preset = {"gpu": bool(args.gpu)}

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


if __name__ == "__main__":
    raise SystemExit(main())
