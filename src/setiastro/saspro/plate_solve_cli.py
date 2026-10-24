# src/setiastro/saspro/plate_solve_cli.py
"""
Headless plate-solving CLI.

    setiastrosuitepro solve -i IMAGE [-o OUT] [options]
    setiastrosuitepro solve -i INDIR -o OUTDIR --batch
    setiastrosuitepro solve --index-status

Runs the same four-pass chain the dialog runs -- in-house Gaia DR3 (with the
blind quad index), then ASTAP seeded and blind, then Astrometry.net -- and
writes the merged WCS into a FITS copy, or back into the original with
--in-place.

Written by Franklin Marek
www.setiastro.com
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

# Image extensions the loader accepts, for --batch directory walks.
IMAGE_EXTS = {".fits", ".fit", ".fts", ".xisf", ".tif", ".tiff",
              ".png", ".jpg", ".jpeg"}

# A QSettings scope of our own. The solver re-reads settings inside its worker
# thread from (organizationName, applicationName), so a plain in-memory shim
# would not survive the hop -- it has to be a real QSettings. Using a separate
# application name keeps every CLI override out of the GUI's own preferences,
# which a user would be rightly annoyed to find rewritten by a batch script.
CLI_SETTINGS_SCOPE = ("SetiAstro", "SASpro-CLI")

# These must match gui_entry.py exactly. Half of SASpro reads its config with
# a bare QSettings(), which resolves against the application identity rather
# than any object we pass around -- the Gaia catalogue folder, the quad index
# folder, and the QStandardPaths default underneath them all. Without setting
# these, a CLI process looks at a different store entirely and concludes the
# user has nothing installed.
QT_ORG = "SetiAstro"
QT_DOMAIN = "setiastrosuite.pro"
QT_APP = "Seti Astro Suite Pro"

SOLVER_PREF = {
    "all": "both",
    "gaia": "gaia_only",
    "astap": "astap_only",
    "astrometry": "astrometry_only",
}


# ── Qt bootstrap ────────────────────────────────────────────────────────────

# The QApplication must outlive every solve. Holding it only in a local would
# let Python drop the last reference and destroy it mid-run -- which surfaces
# as "QWidget: Must construct a QApplication before a QWidget" and an abort,
# some distance from the actual cause.
_APP = None


def _ensure_qapp():
    """
    The solver runs its Gaia pass on a QThread and waits on a QEventLoop, so a
    Q(Core)Application must exist even with no UI. plate_solver imports
    QtWidgets, so this is a QApplication -- which on Linux wants a display it
    will not have on a build server or over ssh. Ask for the offscreen platform
    in that case rather than dying with "could not connect to display".
    """
    from PyQt6.QtCore import QCoreApplication
    from PyQt6.QtWidgets import QApplication

    global _APP
    if _APP is not None:
        return _APP
    existing = QApplication.instance()
    if existing is not None:
        _APP = existing
        return _APP
    if (sys.platform.startswith("linux")
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
            and not os.environ.get("QT_QPA_PLATFORM")):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # Identity BEFORE construction: QSettings and QStandardPaths both key off
    # it, and the GUI's own comment on these lines is "keep these exactly".
    QCoreApplication.setOrganizationName(QT_ORG)
    QCoreApplication.setOrganizationDomain(QT_DOMAIN)
    QCoreApplication.setApplicationName(QT_APP)
    _APP = QApplication([sys.argv[0] if sys.argv else "saspro"])
    return _APP


def _cli_settings(args):
    """
    The user's real configuration, with this run's flags layered on top.

    A scratch scope of its own was the wrong idea: the solver reads the ASTAP
    executable path, the Astrometry.net API key and the library folders from
    the settings object it is handed, so an empty scope told the solver the
    user had configured none of them -- ASTAP "not set", API key "not
    configured", catalogues missing -- on a machine where all three were
    present.

    So: snapshot the real store, apply the command-line overrides to the copy,
    and hand that to the solver. The GUI's own settings are never written to,
    so a batch script cannot leave a user's preferences altered behind it.
    """
    from PyQt6.QtCore import QSettings

    real = QSettings()          # resolves via the identity set in _ensure_qapp
    real.sync()
    s = QSettings(*CLI_SETTINGS_SCOPE)
    # Wipe first: these keys are persisted, and a flag left over from a
    # previous invocation silently changing this one would be a miserable
    # thing to debug from a script.
    s.clear()
    for k in real.allKeys():
        try:
            s.setValue(k, real.value(k))
        except Exception:
            pass

    s.setValue("plate_solver/solver_pref", SOLVER_PREF[args.solver])
    s.setValue("plate_solver/blind_quads", bool(args.blind))
    s.setValue("plate_solver/force_blind", bool(args.force_blind))

    manual = (args.ra is not None and args.dec is not None)
    s.setValue("astap/seed_mode", "manual" if manual else "auto")
    if manual:
        s.setValue("astap/manual_ra", str(args.ra))
        s.setValue("astap/manual_dec", str(args.dec))
    if args.scale is not None:
        s.setValue("astap/manual_scale_arcsec", float(args.scale))

    s.setValue("astap/seed_radius_mode",
               "value" if args.radius is not None else "auto")
    if args.radius is not None:
        s.setValue("astap/seed_radius_value", float(args.radius))
    s.setValue("astap/seed_fov_mode", "compute")
    s.sync()
    return s


# ── reporting ───────────────────────────────────────────────────────────────

def _summarize(hdr, w_px: int, h_px: int) -> dict:
    """Pull the human-facing numbers out of a solved header."""
    from setiastro.saspro.plate_solver import _wcs_from_header_2d

    out: dict = {}
    try:
        import numpy as np
        w = _wcs_from_header_2d(hdr, relax=True)
        cx, cy = (w_px - 1) / 2.0, (h_px - 1) / 2.0
        c = w.pixel_to_world(cx, cy)
        out["ra_deg"] = round(float(c.ra.deg), 6)
        out["dec_deg"] = round(float(c.dec.deg), 6)

        cd = w.pixel_scale_matrix
        det = float(cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0])
        scale = math.sqrt(abs(det)) * 3600.0
        out["scale_arcsec_px"] = round(scale, 4)
        out["fov_w_deg"] = round(w_px * scale / 3600.0, 5)
        out["fov_h_deg"] = round(h_px * scale / 3600.0, 5)
        out["parity"] = "flipped" if det > 0 else "normal"

        xn, yn = w.world_to_pixel_values(out["ra_deg"],
                                         out["dec_deg"] + 10.0 / 3600.0)
        rot = math.degrees(math.atan2(float(xn) - cx, float(yn) - cy)) % 360.0
        out["rotation_deg"] = round(rot, 3)
    except Exception:
        pass

    for key, name in (("SOLVRMS", "rms_arcsec"), ("SOLVP95", "p95_arcsec"),
                      ("SOLVNSTR", "n_stars"), ("FOCALLEN", "focal_len_mm"),
                      ("BLNDODDS", "blind_log_odds"),
                      ("BLNDLAYR", "blind_layer")):
        v = hdr.get(key)
        if v is not None:
            out[name] = float(v) if name != "blind_layer" else v
    if out.get("n_stars") is not None:
        out["n_stars"] = int(out["n_stars"])

    order = hdr.get("A_ORDER", hdr.get("B_ORDER"))
    out["distortion"] = f"SIP-{int(order)}" if order else "linear TAN"

    src = str(hdr.get("SOLVSRC", "")).strip().lower()
    out["seed_grade"] = (src == "quad-seed")
    return out


def _print_summary(res: dict, prefix: str = "") -> None:
    p = prefix
    ra, dec = res.get("ra_deg"), res.get("dec_deg")
    if ra is not None:
        print(f"{p}center     : RA {ra:.5f}  Dec {dec:+.5f}")
    if res.get("scale_arcsec_px"):
        print(f"{p}scale      : {res['scale_arcsec_px']:.4f}\"/px"
              f"   fov {res.get('fov_w_deg', 0):.3f} x "
              f"{res.get('fov_h_deg', 0):.3f} deg")
    if res.get("rotation_deg") is not None:
        print(f"{p}rotation   : {res['rotation_deg']:.2f} deg   "
              f"parity {res.get('parity', '?')}")
    print(f"{p}distortion : {res.get('distortion', '?')}")
    if res.get("rms_arcsec") is not None:
        n = res.get("n_stars")
        px = ""
        if res.get("scale_arcsec_px"):
            px = f" ({res['rms_arcsec'] / res['scale_arcsec_px']:.2f} px)"
        print(f"{p}fit RMS    : {res['rms_arcsec']:.2f}\"{px}"
              + (f" over {n} stars" if n else ""))
    if res.get("blind_log_odds") is not None:
        print(f"{p}blind      : log-odds {res['blind_log_odds']:,.0f}"
              + (f", layer {res.get('blind_layer')}"
                 if res.get("blind_layer") is not None else ""))
    if res.get("seed_grade"):
        print(f"{p}WARNING    : seed-grade solution (quad index placement, not "
              f"refined). Centre and scale are approximate and there are no "
              f"distortion terms.")


# ── one image ───────────────────────────────────────────────────────────────

def _solve_one(path: Path, out_path: Path | None, settings, args) -> dict:
    """Solve a single image. Returns a result dict; never raises."""
    from astropy.io.fits import Header
    from setiastro.saspro.legacy.image_manager import load_image, save_image
    from setiastro.saspro.plate_solver import (
        _solve_numpy_with_fallback, _as_header, _strip_wcs_keys,
        _merge_wcs_into_base_header, _sanitized_header_for_wcs,
        _status_popup_close,
    )

    rec: dict = {"input": str(path), "ok": False}
    try:
        image_data, original_header, bit_depth, is_mono = load_image(str(path))
    except Exception as e:
        rec["error"] = f"cannot read image: {e}"
        return rec
    if image_data is None:
        rec["error"] = "unsupported or unreadable image"
        return rec

    seed_h = (_as_header(original_header)
              if isinstance(original_header, (dict, Header)) else None)
    acq_base = _strip_wcs_keys(seed_h) if isinstance(seed_h, Header) else None

    buf = io.StringIO()
    try:
        # Solver chatter goes to a buffer when quiet, and is replayed only if
        # the solve fails -- which is exactly when a script's operator wants it.
        ctx = contextlib.redirect_stdout(buf) if args.quiet \
            else contextlib.nullcontext()
        with ctx:
            ok, res = _solve_numpy_with_fallback(None, settings,
                                                 image_data, seed_h)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        if args.quiet:
            rec["log"] = buf.getvalue()
        return rec
    finally:
        try:
            _status_popup_close()
        except Exception:
            pass

    if not ok:
        rec["error"] = str(res)
        if args.quiet:
            rec["log"] = buf.getvalue()
        return rec

    if isinstance(acq_base, Header) and isinstance(res, Header):
        hdr_final = _merge_wcs_into_base_header(acq_base, res)
    else:
        hdr_final = (_sanitized_header_for_wcs(res)
                     if isinstance(res, Header) else Header())

    h, w = int(image_data.shape[0]), int(image_data.shape[1])
    rec.update(_summarize(hdr_final, w, h))
    rec["ok"] = True

    if out_path is not None:
        try:
            # Wholesale copy, then drop FILE_PATH. Rebuilding card by card
            # breaks on commentary keywords: hdr["HISTORY"] returns every
            # HISTORY card joined by newlines, and assigning that back raises
            # on the newline -- so a solve that recomputed FOCALLEN would
            # succeed and then fail to save.
            clean = hdr_final.copy()
            for _k in list(clean.keys()):
                if _k.upper() == "FILE_PATH":
                    try:
                        del clean[_k]
                    except Exception:
                        pass
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(
                img_array=image_data,
                filename=str(out_path),
                original_format="fit",
                bit_depth="32-bit floating point",
                original_header=clean,
                is_mono=is_mono,
            )
            rec["output"] = str(out_path)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"solved, but could not save: {e}"
    return rec


# ── parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="setiastrosuitepro solve",
        description="Plate-solve images headlessly (Gaia DR3 blind/seeded, "
                    "ASTAP, Astrometry.net).",
        epilog="Exit codes: 0 all solved, 1 one or more failed, 2 usage/IO "
               "error.",
    )
    ap.add_argument("-i", "--input",
                    help="Input image, or input directory with --batch.")
    ap.add_argument("-o", "--output",
                    help="Output FITS path, or output directory with --batch. "
                         "Omit with --in-place.")
    ap.add_argument("--batch", action="store_true",
                    help="Treat --input as a directory and solve every image "
                         "in it.")
    ap.add_argument("--in-place", action="store_true",
                    help="Write the solved header back over the input file "
                         "instead of to a new one.")

    g = ap.add_argument_group("solver")
    g.add_argument("--solver", choices=sorted(SOLVER_PREF), default="all",
                   help="Which backends to try (default: all, in the usual "
                        "fallback order).")
    g.add_argument("--blind", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use the local quad index when the header carries no "
                        "RA/Dec (default: on).")
    g.add_argument("--force-blind", action="store_true",
                   help="Ignore any RA/Dec in the header and locate the field "
                        "from the star pattern. Use when the coordinates are "
                        "wrong rather than missing.")

    g = ap.add_argument_group("manual seed")
    g.add_argument("--ra", type=float,
                   help="Seed RA in degrees (requires --dec).")
    g.add_argument("--dec", type=float,
                   help="Seed Dec in degrees (requires --ra).")
    g.add_argument("--scale", type=float,
                   help="Pixel scale in arcsec/px.")
    g.add_argument("--radius", type=float,
                   help="ASTAP cone search radius in degrees (default: auto).")

    g = ap.add_argument_group("output")
    g.add_argument("--json", action="store_true",
                   help="Emit the result as JSON on stdout. Implies --quiet.")
    g.add_argument("--quiet", action="store_true",
                   help="Suppress solver progress; show it only on failure.")
    g.add_argument("--index-status", action="store_true",
                   help="Report which blind-solve quad index layers are built, "
                        "then exit.")
    return ap


def _print_index_status() -> int:
    from setiastro.saspro.gaia_quad_index import (
        get_quad_library, refresh_quad_library, QUAD_LAYER_DEFS, _fmt_ang)

    refresh_quad_library()
    lib = get_quad_library()
    built = {l.layer_index: l for l in lib.installed()}
    print(f"quad index folder: {lib.directory}")
    print(f"{'layer':<9}{'quad':>7}{'solves fields':>22}{'state':>10}{'quads':>15}")
    for d in QUAD_LAYER_DEFS:
        lay = built.get(d.index)
        span = f"{_fmt_ang(d.field_lo_arcmin)} - {_fmt_ang(d.field_hi_arcmin)}"
        print(f"{d.label.split('—')[0].strip():<9}{_fmt_ang(d.q_arcmin):>7}"
              f"{span:>22}{'BUILT' if lay else '-':>10}"
              f"{(f'{int(lay.n_quads):,}' if lay else ''):>15}")
    if built:
        lo = min(d.field_lo_arcmin for d in QUAD_LAYER_DEFS
                 if d.index in built)
        hi = max(d.field_hi_arcmin for d in QUAD_LAYER_DEFS
                 if d.index in built)
        print(f"\n{len(built)} of {len(QUAD_LAYER_DEFS)} layers built; "
              f"blind solving covers fields {_fmt_ang(lo)} to {_fmt_ang(hi)} "
              f"wide.")
        return 0
    print("\nNo layers built. Blind solving is unavailable until at least one "
          "is built\n(Gaia DR3 Library -> Blind Solve Index).")
    return 1


# ── entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.json:
        args.quiet = True

    if args.index_status:
        _ensure_qapp()
        from setiastro.saspro.plate_solver import set_status_popup_enabled
        set_status_popup_enabled(False)
        return _print_index_status()

    if not args.input:
        ap.error("-i/--input is required (or use --index-status)")
    if (args.ra is None) != (args.dec is None):
        ap.error("--ra and --dec must be given together")
    if not args.in_place and not args.output:
        ap.error("either -o/--output or --in-place is required")
    if args.in_place and args.output:
        ap.error("--in-place and --output are mutually exclusive")

    src = Path(args.input)
    if not src.exists():
        print(f"ERROR: input does not exist: {src}", file=sys.stderr)
        return 2

    _ensure_qapp()
    from setiastro.saspro.plate_solver import set_status_popup_enabled
    set_status_popup_enabled(False)
    settings = _cli_settings(args)

    # Build the work list.
    jobs: list[tuple[Path, Path | None]] = []
    if args.batch:
        if not src.is_dir():
            print(f"ERROR: --batch needs a directory, got {src}",
                  file=sys.stderr)
            return 2
        outdir = None if args.in_place else Path(args.output)
        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                jobs.append((f, f if args.in_place
                             else outdir / f"{f.stem}_plate_solved.fits"))
        if not jobs:
            print(f"ERROR: no images found in {src}", file=sys.stderr)
            return 2
    else:
        if src.is_dir():
            print("ERROR: input is a directory; pass --batch to solve it",
                  file=sys.stderr)
            return 2
        jobs.append((src, src if args.in_place else Path(args.output)))

    results, failed = [], 0
    for path, out in jobs:
        if not args.json:
            print(f"\n=> {path.name}" if len(jobs) > 1 else f"=> {path.name}")
        rec = _solve_one(path, out, settings, args)
        results.append(rec)
        if rec["ok"]:
            if not args.json:
                _print_summary(rec, prefix="   ")
                if rec.get("output"):
                    print(f"   saved      : {rec['output']}")
        else:
            failed += 1
            if not args.json:
                print(f"   FAILED     : {rec.get('error', 'unknown')}",
                      file=sys.stderr)
                if rec.get("log"):
                    print(rec["log"], file=sys.stderr)

    if args.json:
        json.dump(results if len(results) > 1 else results[0],
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
    elif len(jobs) > 1:
        print(f"\n{len(jobs) - failed}/{len(jobs)} solved.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())