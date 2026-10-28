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

    s.setValue("plate_solver/preference", SOLVER_PREF[args.solver])
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


# ── .wcs sidecar ────────────────────────────────────────────────────────────

# Everything that belongs in a WCS sidecar. The prefixes mirror the list
# _strip_wcs_keys() removes in plate_solver, so the two stay in step: what the
# merge treats as "solver-owned" is exactly what gets written here.
_WCS_PREFIXES = ("CRPIX", "CRVAL", "CDELT", "CROTA", "CD1_", "CD2_", "PC",
                 "CTYPE", "CUNIT", "WCSAXES", "LATPOLE", "LONPOLE", "EQUINOX",
                 "PV1_", "PV2_", "A_", "B_", "AP_", "BP_")
_WCS_EXACT = ("RADESYS", "RADECSYS", "DATE-OBS", "MJD-OBS", "FOCALLEN",
              "SOLVRMS", "SOLVP95", "SOLVNSTR", "SOLVSRC",
              "BLNDODDS", "BLNDLAYR", "BLNDPAR")


def _write_wcs_sidecar(hdr, path: Path, img_w: int, img_h: int) -> str:
    """
    Write a `.wcs` sidecar: a header-only FITS carrying the solution alone.

    This is the same shape ASTAP and Astrometry.net produce, and the same
    shape plate_solver already reads back through fits.getheader() -- so a
    sidecar written here is parseable by SASpro's own ASTAP code path, and by
    anything else that consumes solve-field output.

    IMAGEW/IMAGEH matter more than they look: a header-only FITS has NAXIS=0
    and therefore no record of the frame it describes, so without them a
    reader cannot tell what the pixel coordinates refer to.
    """
    from astropy.io import fits
    from astropy.io.fits import Header

    out = Header()
    for k in hdr.keys():
        ku = str(k).upper()
        if ku in ("", "COMMENT", "HISTORY", "END", "SIMPLE", "BITPIX",
                  "EXTEND") or ku.startswith("NAXIS"):
            continue
        if ku in _WCS_EXACT or ku.startswith(_WCS_PREFIXES):
            try:
                out[k] = hdr[k]
            except Exception:
                pass
    # Normalise the linear terms to a CD matrix. astropy's to_header() emits
    # PC plus CDELT=1.0, which is valid WCS but wrong for a sidecar twice
    # over: ASTAP and solve-field both write CD, and CDELT=1.0 alongside a PC
    # that carries the real scale is the exact shape SASpro's own stacking
    # code treats as a falsified header. Derive CD once and drop the rest.
    try:
        # Same reason as in _write_astap_ini: a bare WCS(hdr) raises on an RGB
        # cube's NAXIS=3 header when SIP terms are present. Here the failure
        # was quieter and worse -- the except below only warned, so the PC and
        # CDELT cards were left in place and no CD was written, which is the
        # falsified-header shape this normalisation exists to prevent.
        from setiastro.saspro.plate_solver import _wcs_from_header_2d
        cd = _wcs_from_header_2d(hdr, relax=True).pixel_scale_matrix
        for _k in list(out.keys()):
            ku = str(_k).upper()
            if ku.startswith("PC") or ku.startswith("CDELT"):
                try:
                    del out[_k]
                except Exception:
                    pass
        out["CD1_1"] = float(cd[0, 0])
        out["CD1_2"] = float(cd[0, 1])
        out["CD2_1"] = float(cd[1, 0])
        out["CD2_2"] = float(cd[1, 1])
    except Exception as e:
        print(f"[solve] could not normalise the sidecar to a CD matrix: {e}")

    if "WCSAXES" not in out:
        out["WCSAXES"] = 2
    out["IMAGEW"] = (int(img_w), "Image width this solution describes [px]")
    out["IMAGEH"] = (int(img_h), "Image height this solution describes [px]")
    out.add_history("Seti Astro Suite Pro: plate solve WCS sidecar")

    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(header=out).writeto(str(path), overwrite=True)
    return str(path)


def _write_astap_ini(path: Path, hdr=None, img_w: int = 0, img_h: int = 0,
                     summary: "dict | None" = None,
                     error: "str | None" = None) -> str:
    """
    Write an ASTAP-style `.ini` result file: plain ``KEY=VALUE`` lines.

    This is the format N.I.N.A. actually reads back from a plate solve -- it
    runs the solver, then parses ``<image-basename>.ini``; the `.wcs` sidecar
    it only deletes afterwards. Writing it here is what lets SASpro stand in
    for ASTAP in N.I.N.A., SGP, Voyager and anything else built against the
    same convention, and it saves a caller from having to parse FITS at all.

    Two details are load-bearing, both learned from how N.I.N.A. parses this:

    * The reader splits each line on the FIRST '=' and builds a dictionary, so
      a repeated key would make it throw. Every key here is written once.
    * When ``PLTSOLVD=T`` it reads CRVAL1/2, CRPIX1/2 and all four CD terms
      WITHOUT checking they exist, so a success line with any of the eight
      missing is worse than a clean failure. If the linear terms cannot be
      recovered this writes a failure instead.

    A failure is worth writing too: without the file a caller can only report
    "no output", while ``PLTSOLVD=F`` plus ERROR carries the real reason.
    """
    lines: list[str] = []

    def _fail(msg: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "PLTSOLVD=F\n"
            f"ERROR={str(msg).replace(chr(10), ' ').strip()}\n",
            encoding="ascii", errors="replace")
        return str(path)

    if error is not None or hdr is None:
        return _fail(error or "no solution")

    # Pull the linear terms. CRVAL/CRPIX come from the WCS object rather than
    # the raw cards so a PC+CDELT header is handled identically to a CD one,
    # and a SIP solution still reports its correct linear anchor.
    try:
        # _wcs_from_header_2d, not a bare WCS(hdr): a colour frame is an RGB
        # cube, so its header says NAXIS=3, and WCSLIB refuses to combine
        # three core axes with SIP ("you must select or reduce these to 2
        # dimensions using the naxis kwarg"). That is every solve from an
        # OSC camera -- which is most of them. The helper exists for exactly
        # this and is what _summarize already uses, so using it here also
        # keeps the .ini and the printed summary derived from one WCS.
        from setiastro.saspro.plate_solver import _wcs_from_header_2d
        w = _wcs_from_header_2d(hdr, relax=True)
        # A header with no WCS does not fail here: astropy hands back a
        # default WCS whose CD is the identity, which is non-singular and
        # would sail through every check below as a "solution" at RA 0 and
        # one degree per pixel. A caller may well slew a mount to whatever
        # this file claims, so an unconfirmed header must never be reported
        # as a success.
        if not w.has_celestial:
            return _fail("no celestial WCS in the solved header")
        crval = [float(v) for v in w.wcs.crval[:2]]
        crpix = [float(v) for v in w.wcs.crpix[:2]]
        cd = w.pixel_scale_matrix
        cd = [[float(cd[0, 0]), float(cd[0, 1])],
              [float(cd[1, 0]), float(cd[1, 1])]]
    except Exception as e:
        return _fail(f"solved, but the linear WCS terms could not be read: {e}")

    vals = crval + crpix + cd[0] + cd[1]
    if not all(math.isfinite(v) for v in vals):
        return _fail("solved, but the WCS contains non-finite terms")
    det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0]
    if abs(det) < 1e-20:
        return _fail("solved, but the CD matrix is singular")

    lines.append("PLTSOLVD=T")
    lines.append(f"CRVAL1={crval[0]!r}")
    lines.append(f"CRVAL2={crval[1]!r}")
    lines.append(f"CRPIX1={crpix[0]!r}")
    lines.append(f"CRPIX2={crpix[1]!r}")
    lines.append(f"CD1_1={cd[0][0]!r}")
    lines.append(f"CD1_2={cd[0][1]!r}")
    lines.append(f"CD2_1={cd[1][0]!r}")
    lines.append(f"CD2_2={cd[1][1]!r}")

    s = summary or {}
    if img_w and img_h:
        lines.append(f"IMAGEW={int(img_w)}")
        lines.append(f"IMAGEH={int(img_h)}")
    for key, name in (("CROTA2", "rotation_deg"),
                      ("SCALE", "scale_arcsec_px"),
                      ("SOLVRMS", "rms_arcsec"),
                      ("SOLVP95", "p95_arcsec")):
        v = s.get(name)
        if v is not None and math.isfinite(float(v)):
            lines.append(f"{key}={float(v)!r}")
    if s.get("n_stars") is not None:
        lines.append(f"SOLVNSTR={int(s['n_stars'])}")
    if s.get("distortion"):
        lines.append(f"DISTORT={s['distortion']}")
    lines.append(f"PARITY={s.get('parity', 'flipped' if det > 0 else 'normal')}")
    # Stated as its own key rather than left to be inferred from the WARNING
    # text: a consumer deciding whether to accept a coarse solution should be
    # matching on a field, not on the wording of a human-readable sentence.
    lines.append(f"SOLVSRC={'quad-seed' if s.get('seed_grade') else 'full-fit'}")

    # A quad-seed result is a real pointing but a coarse one. Say so through
    # WARNING, which is the channel the ASTAP convention already has for
    # "usable, but look at it" -- N.I.N.A. surfaces it to the user and still
    # accepts the solve.
    if s.get("seed_grade"):
        lines.append("WARNING=Seed-grade solution from the blind quad index; "
                     "pointing is good but the scale and rotation are coarse.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii",
                    errors="replace")
    return str(path)


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

def _solve_one(path: Path, out_path: "Path | None", wcs_path: "Path | None",
               settings, args, ini_path: "Path | None" = None) -> dict:
    """
    Solve one image, and write the ASTAP-style `.ini` whatever the outcome.

    The wrapper exists for the failure case: _solve_one_impl() returns early
    from half a dozen places, and a caller driving SASpro as an ASTAP stand-in
    needs the reason on disk rather than an absent file.
    """
    rec = _solve_one_impl(path, out_path, wcs_path, settings, args, ini_path)
    if ini_path is not None and not rec.get("ini"):
        try:
            rec["ini"] = _write_astap_ini(
                ini_path, error=rec.get("error") or "no solution")
        except Exception as e:
            print(f"[solve] could not write {ini_path}: {e}", file=sys.stderr)
    return rec


def _solve_one_impl(path: Path, out_path: "Path | None",
                    wcs_path: "Path | None", settings, args,
                    ini_path: "Path | None" = None) -> dict:
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

    if ini_path is not None:
        # Written before anything else can go wrong: by this point the
        # solution is in hand, and a caller waiting on the .ini should get it
        # even if a later save fails.
        try:
            rec["ini"] = _write_astap_ini(ini_path, hdr_final, w, h, rec)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"solved, but could not write the .ini: {e}"
            return rec

    if wcs_path is not None:
        try:
            rec["wcs"] = _write_wcs_sidecar(hdr_final, wcs_path, w, h)
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"solved, but could not write the .wcs sidecar: {e}"
            return rec

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
    g.add_argument("--stretch-backend", choices=("auto", "numpy", "numba"),
                   default="auto",
                   help="Which kernels the internal stretch uses. 'auto' "
                        "(default) picks numpy for a single image and "
                        "numba for --batch: the solver performs exactly "
                        "one stretch per frame, and numba costs ~700ms to "
                        "import plus JIT compilation to save ~100ms of "
                        "arithmetic, so it only pays off amortised over "
                        "many frames. 'numba' forces the JIT kernels, "
                        "'numpy' forces the plain ones. The solution is "
                        "identical either way.")

    g = ap.add_argument_group("manual seed")
    g.add_argument("--ra", type=float,
                   help="Seed RA in degrees (requires --dec).")
    g.add_argument("--dec", type=float,
                   help="Seed Dec in degrees (requires --ra).")
    g.add_argument("--scale", type=float,
                   help="Pixel scale in arcsec/px.")
    g.add_argument("--radius", type=float,
                   help="ASTAP cone search radius in degrees (default: auto).")

    g = ap.add_argument_group("sidecar")
    g.add_argument("--wcs", action="store_true",
                   help="Also write a .wcs sidecar (header-only FITS) beside "
                        "the output, the way ASTAP and Astrometry.net do. May "
                        "be used on its own to solve WITHOUT rewriting the "
                        "image.")
    g.add_argument("--wcs-only", action="store_true",
                   help="Solve and write ONLY the sidecar, leaving the image "
                        "untouched. Same effect as --wcs with no output, said "
                        "out loud so a script declares its intent and a "
                        "contradictory -o is caught rather than obeyed.")
    g.add_argument("--wcs-path",
                   help="Explicit path for the sidecar. Single-image mode "
                        "only; implies --wcs.")
    g.add_argument("--ini", action="store_true",
                   help="Also write an ASTAP-style .ini result file (plain "
                        "KEY=VALUE: PLTSOLVD, CRVAL/CRPIX, the CD matrix). "
                        "This is what N.I.N.A. and other ASTAP-driven "
                        "software read back, and it needs no FITS parser. "
                        "A failed solve still writes PLTSOLVD=F with the "
                        "reason. May be used on its own.")
    g.add_argument("--ini-only", action="store_true",
                   help="Solve and write ONLY the .ini, leaving the image "
                        "untouched.")
    g.add_argument("--ini-path",
                   help="Explicit path for the .ini. Single-image mode only; "
                        "implies --ini.")

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
    if args.wcs_only:
        if args.output or args.in_place:
            ap.error("--wcs-only writes nothing but the sidecar; drop "
                     "-o/--output and --in-place")
        args.wcs = True
    if args.ini_only:
        if args.output or args.in_place:
            ap.error("--ini-only writes nothing but the .ini; drop "
                     "-o/--output and --in-place")
        args.ini = True
    if args.wcs_path:
        args.wcs = True
        if args.batch:
            ap.error("--wcs-path is for a single image; with --batch the "
                     "sidecars are named after each input")
    if args.ini_path:
        args.ini = True
        if args.batch:
            ap.error("--ini-path is for a single image; with --batch the "
                     ".ini files are named after each input")
    if not args.in_place and not args.output and not args.wcs and not args.ini:
        ap.error("one of -o/--output, --in-place, --wcs or --ini is required")
    if args.in_place and args.output:
        ap.error("--in-place and --output are mutually exclusive")

    src = Path(args.input)
    if not src.exists():
        print(f"ERROR: input does not exist: {src}", file=sys.stderr)
        return 2

    # Which stretch kernels to use. _solve_with_GAIA calls
    # _normalize_for_astap, which performs exactly ONE stretch_color_image
    # per frame -- so for a single image the numba kernels cost ~700 ms of
    # import plus JIT compilation to save on the order of 100 ms of array
    # work. Measured on a 1080x1920x3 frame: 10.3 s with numba, 8.2 s
    # without, and a byte-identical solution (same centre, same four
    # corners, same RMS and pair count).
    #
    # --batch keeps numba, where the import and the JIT amortise across
    # frames; the crossover is somewhere around twenty of them. This
    # matters most for callers that spawn a process per frame -- N.I.N.A.
    # through the ASTAP shim does exactly that, so there is never anything
    # to amortise over.
    #
    # setdefault, so an explicit SASPRO_STRETCH_BACKEND in the environment
    # still wins over the 'auto' default.
    if args.stretch_backend == "auto":
        if not args.batch:
            os.environ.setdefault("SASPRO_STRETCH_BACKEND", "numpy")
    else:
        os.environ["SASPRO_STRETCH_BACKEND"] = args.stretch_backend

    _ensure_qapp()
    from setiastro.saspro.plate_solver import set_status_popup_enabled
    set_status_popup_enabled(False)
    settings = _cli_settings(args)

    # Build the work list.
    def _sidecar_for(inp: Path, outp: "Path | None") -> "Path | None":
        # Beside the output when there is one, else beside the input -- which
        # is what makes `--wcs` alone useful: solve the frame, leave it
        # untouched, drop the solution next to it.
        if not args.wcs:
            return None
        if args.wcs_path:
            return Path(args.wcs_path)
        base = outp if (outp is not None and not args.in_place) else inp
        return base.with_suffix(".wcs")

    def _ini_for(inp: Path, outp: "Path | None") -> "Path | None":
        if not args.ini:
            return None
        if args.ini_path:
            return Path(args.ini_path)
        # Beside the INPUT by default, even when an output is being written.
        # ASTAP's callers look for <input-basename>.ini next to the frame they
        # handed over, so that is where it has to land to be found.
        return inp.with_suffix(".ini")

    jobs: list[tuple[Path, "Path | None", "Path | None", "Path | None"]] = []
    if args.batch:
        if not src.is_dir():
            print(f"ERROR: --batch needs a directory, got {src}",
                  file=sys.stderr)
            return 2
        write_img = args.in_place or bool(args.output)
        outdir = Path(args.output) if (args.output and not args.in_place) else None
        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                if args.in_place:
                    outp = f
                elif outdir is not None:
                    outp = outdir / f"{f.stem}_plate_solved.fits"
                else:
                    outp = None          # --wcs only: do not rewrite images
                jobs.append((f, outp, _sidecar_for(f, outp), _ini_for(f, outp)))
        if not jobs:
            print(f"ERROR: no images found in {src}", file=sys.stderr)
            return 2
    else:
        if src.is_dir():
            print("ERROR: input is a directory; pass --batch to solve it",
                  file=sys.stderr)
            return 2
        outp = src if args.in_place else (Path(args.output) if args.output
                                          else None)
        jobs.append((src, outp, _sidecar_for(src, outp), _ini_for(src, outp)))

    results, failed = [], 0
    for path, out, wcs, ini in jobs:
        if not args.json:
            print(f"\n=> {path.name}" if len(jobs) > 1 else f"=> {path.name}")
        rec = _solve_one(path, out, wcs, settings, args, ini)
        results.append(rec)
        if rec["ok"]:
            if not args.json:
                _print_summary(rec, prefix="   ")
                if rec.get("output"):
                    print(f"   saved      : {rec['output']}")
                if rec.get("wcs"):
                    print(f"   sidecar    : {rec['wcs']}")
                if rec.get("ini"):
                    print(f"   ini        : {rec['ini']}")
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