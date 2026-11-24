# src/setiastro/saspro/plate_solution_check.py
# SetiAstro Suite Pro -- does this image's plate solution actually fit it?
#
# Copyright (c) 2026 Franklin Marek / SetiAstro

"""
SPCC, SSSC and NBExtract look stars up by sky position, so they trust the
image's WCS. A wrong solution does not make them fail: detected stars still
land within a few arcseconds of SOME catalog star by chance, the calibration
runs on those, and the result is quietly wrong.

wcs_match_test() tells the two apart without knowing how deep the catalog or
the image goes. It counts detections with a catalog star within the match
radius, then counts again with every detection shifted a fixed distance in
eight directions, far more than any usable solution is ever off. Those
shifted counts are pure coincidence at this field's own star density, so they
measure chance directly. A right solution beats chance many times over; a
wrong one (another field, a mirror image, a wrong scale) matches at about
chance, however many Gaia groups are installed.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QCoreApplication, QSettings
from PyQt6.QtWidgets import QMessageBox

MATCH_RADIUS_ARCSEC = 3.0      # the radius SPCC and SSSC match with
DECOY_OFFSET_ARCSEC = 60.0     # far past any usable solution's error
MIN_CATALOG_STARS = 15         # fewer in the field: too little to judge
MIN_DETECTIONS = 10
CHANCE_P = 0.01                # this many matches this likely by chance = bad


def _tr(text: str) -> str:
    return QCoreApplication.translate("PlateSolutionCheck", text)


def _unit(ra_deg, dec_deg) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    c = np.cos(dec)
    return np.column_stack([c * np.cos(ra), c * np.sin(ra), np.sin(dec)])


def _shift(ra_deg, dec_deg, east_arcsec: float, north_arcsec: float):
    """Move sky positions by a small offset, in arcsec east and north."""
    dec = np.clip(np.asarray(dec_deg, dtype=np.float64) + north_arcsec / 3600.0,
                  -90.0, 90.0)
    cosd = np.maximum(np.cos(np.radians(dec)), 1e-6)
    ra = np.asarray(ra_deg, dtype=np.float64) + east_arcsec / 3600.0 / cosd
    return np.mod(ra, 360.0), dec


def _chance_p(matched: int, chance: float) -> float:
    """P(at least `matched` coincidences) for a Poisson mean of `chance`."""
    if matched <= 0:
        return 1.0
    if chance <= 0.0:
        return 0.0
    from scipy.special import gammainc       # P(X >= m) = P(m, mean), regularised
    return float(gammainc(matched, chance))


def wcs_match_test(det_ra, det_dec, cat_ra, cat_dec,
                   radius_arcsec: float = MATCH_RADIUS_ARCSEC,
                   offset_arcsec: float = DECOY_OFFSET_ARCSEC) -> dict:
    """
    How many detections sit on a catalog star, against how many would by chance.

    det_*: detected stars' sky positions through the WCS under test.
    cat_*: catalog stars over the detections plus offset_arcsec of margin.

    Returns a dict:
      verdict  "good", "bad", or "unknown" when there is too little to judge
      matched  detections with a catalog star within radius_arcsec
      chance   mean count with the detections shifted (coincidences)
      p        chance of `matched` or more coincidences
      n_det    detections tested
      n_cat    catalog stars inside the detections' own footprint
    """
    from scipy.spatial import cKDTree

    det_ra = np.asarray(det_ra, dtype=np.float64).ravel()
    det_dec = np.asarray(det_dec, dtype=np.float64).ravel()
    cat_ra = np.asarray(cat_ra, dtype=np.float64).ravel()
    cat_dec = np.asarray(cat_dec, dtype=np.float64).ravel()
    out = {"verdict": "unknown", "matched": 0, "chance": 0.0, "p": 1.0,
           "n_det": int(det_ra.size), "n_cat": 0}
    if det_ra.size == 0 or cat_ra.size == 0:
        return out

    # Catalog stars in the field itself (the read carries a margin for the
    # shifted copies). RA is taken relative to the field so 0/360 can't split it.
    ra0 = float(np.median(det_ra))
    d_rel = np.mod(det_ra - ra0 + 180.0, 360.0) - 180.0
    c_rel = np.mod(cat_ra - ra0 + 180.0, 360.0) - 180.0
    n_cat = int(np.count_nonzero(
        (c_rel >= d_rel.min()) & (c_rel <= d_rel.max())
        & (cat_dec >= det_dec.min()) & (cat_dec <= det_dec.max())))

    tree = cKDTree(_unit(cat_ra, cat_dec))
    chord = 2.0 * math.sin(math.radians(radius_arcsec / 3600.0) / 2.0)

    def _count(ra, dec) -> int:
        d, _ = tree.query(_unit(ra, dec), k=1, distance_upper_bound=chord)
        return int(np.count_nonzero(np.isfinite(d)))

    matched = _count(det_ra, det_dec)
    shifted = []
    for k in range(8):
        a = k * math.pi / 4.0
        shifted.append(_count(*_shift(det_ra, det_dec,
                                      offset_arcsec * math.cos(a),
                                      offset_arcsec * math.sin(a))))
    chance = float(np.mean(shifted))
    p = _chance_p(matched, chance)
    out.update(matched=matched, chance=chance, p=p, n_cat=n_cat)

    if n_cat < MIN_CATALOG_STARS or det_ra.size < MIN_DETECTIONS:
        return out
    # Bad: no better than chance, or so close to it that coincidences would
    # be a large share of the stars the calibration then uses.
    out["verdict"] = "bad" if (p > CHANCE_P or matched < 2.0 * chance) else "good"
    return out


def check_gaia_library(sky_coords) -> dict | None:
    """wcs_match_test against the local Gaia library; None when it isn't installed."""
    from setiastro.saspro.gaia_database import get_library
    lib = get_library()
    if not lib.installed_bands():
        return None
    sc = np.asarray(sky_coords, dtype=np.float64)
    ra, dec = sc[:, 0], sc[:, 1]
    pad = (DECOY_OFFSET_ARCSEC + MATCH_RADIUS_ARCSEC) / 3600.0
    dec_min = max(-90.0, float(dec.min()) - pad)
    dec_max = min(90.0, float(dec.max()) + pad)
    cosd = max(1e-6, math.cos(math.radians(max(abs(dec_min), abs(dec_max)))))
    ra_min = float(ra.min()) - pad / cosd
    ra_max = float(ra.max()) + pad / cosd
    if ra_min < 0.0 or ra_max > 360.0:         # the box crosses 0h: read the band
        ra_min, ra_max = 0.0, 360.0
    cat_ra, cat_dec = lib.positions_in_box(ra_min, ra_max, dec_min, dec_max)
    return wcs_match_test(ra, dec, cat_ra, cat_dec)


def _have_blind_index() -> bool:
    try:
        from setiastro.saspro.gaia_quad_index import get_quad_library
        return bool(get_quad_library().any_installed())
    except Exception:
        return False


def ask_about_plate_solution(parent, res: dict, tool: str) -> str:
    """The prompt. Returns "resolve", "continue" or "cancel"."""
    have_index = _have_blind_index()
    if res["matched"] <= 0:
        found = _tr("No Gaia catalog star lines up with any of the {0:,} stars "
                    "found in this image, though {1:,} catalog stars lie in "
                    "this field.").format(res["n_det"], res["n_cat"])
    else:
        found = _tr("Gaia catalog stars line up with only {0:,} of the {1:,} "
                    "stars found in this image: little more than chance alone "
                    "gives in this field (about {2:,.0f})."
                    ).format(res["matched"], res["n_det"], res["chance"])
    text = found + "\n\n" + _tr(
        "The plate solution on this image is most likely inaccurate, so {0} "
        "would calibrate against the wrong stars.").format(tool)
    if have_index:
        text += "\n\n" + _tr(
            "Re-solve it with the Gaia blind solver? It ignores the current "
            "solution and finds the field from the star pattern alone.")
    else:
        text += "\n\n" + _tr(
            "Re-solving it blind needs a Blind Solve Index, which isn't built "
            "yet. Build one on the Gaia DR3 Library dialog, or re-solve the "
            "image in the Plate Solver.")

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(_tr("Plate Solution Looks Wrong"))
    box.setText(text)
    resolve_btn = (box.addButton(_tr("Re-solve (Gaia Blind)"),
                                 QMessageBox.ButtonRole.AcceptRole)
                   if have_index else None)
    continue_btn = box.addButton(_tr("Continue Anyway"),
                                 QMessageBox.ButtonRole.DestructiveRole)
    cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(resolve_btn or cancel_btn)
    box.setEscapeButton(cancel_btn)
    box.exec()
    clicked = box.clickedButton()
    if resolve_btn is not None and clicked is resolve_btn:
        return "resolve"
    if clicked is continue_btn:
        return "continue"
    return "cancel"


def resolve_active_doc(dlg) -> tuple[bool, str]:
    """Gaia blind re-solve of the dialog's active document, in place."""
    from setiastro.saspro.plate_solver import plate_solve_doc_inplace
    dm = getattr(dlg, "doc_manager", None)
    doc = dm.get_active_document() if dm is not None else None
    if doc is None or getattr(doc, "image", None) is None:
        return False, _tr("No active image.")
    parent = getattr(dlg, "main_win", None) or dlg
    try:
        ok, res = plate_solve_doc_inplace(parent, doc, QSettings(), gaia_blind=True)
    except Exception as e:
        return False, str(e)
    return (True, "") if ok else (False, str(res))


def check_dialog_plate_solution(dlg, sky_coords, *, tool: str, log_tag: str) -> str:
    """
    For a calibration dialog that has just put its SEP detections through the
    image's WCS. Returns "ok" to carry on, "stop" to end the step here, or
    "resolved" when the image was just re-solved and the step should start over.
    Headless runs never prompt: a solution that looks wrong stops them.
    """
    try:
        res = check_gaia_library(sky_coords)
    except Exception as e:
        print(f"[{log_tag}] Plate solution check skipped: {e}")
        return "ok"
    if res is None:
        return "ok"                        # no local Gaia library to check against
    print(f"[{log_tag}] Plate solution check: {res['matched']:,} of "
          f"{res['n_det']:,} detections on a Gaia star, chance ~{res['chance']:.1f} "
          f"(p={res['p']:.2g}), {res['n_cat']:,} catalog stars in the field "
          f"-> {res['verdict']}")
    if res["verdict"] != "bad":
        return "ok"
    if getattr(dlg, "_headless", False):
        print(f"[{log_tag}] The plate solution looks wrong; stopping (headless run).")
        return "stop"

    choice = ask_about_plate_solution(dlg, res, tool)
    if choice == "continue":
        print(f"[{log_tag}] Continuing on a plate solution that looks wrong (user's choice).")
        return "ok"
    if choice != "resolve":
        return "stop"
    ok, msg = resolve_active_doc(dlg)
    if ok:
        print(f"[{log_tag}] Re-solved blind; starting the step over.")
        return "resolved"
    QMessageBox.warning(dlg, _tr("Re-solve Failed"),
                        _tr("The Gaia blind solver could not solve this image:\n\n{0}"
                            ).format(msg))
    return "stop"
