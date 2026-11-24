"""XPIXSZ already includes binning; the plate solver must not multiply it again."""
from __future__ import annotations

import pytest

from astropy.io import fits

# SGP bin 2 on the 2.315 um ASI294MM behind a 950 mm scope, as it lands on disk.
SGP_BIN2 = {"XPIXSZ": 4.63, "YPIXSZ": 4.63, "XBINNING": 2, "YBINNING": 2, "FOCALLEN": 950.0}
SCALE_BIN2 = 206.264806 * 4.63 / 950.0          # 1.0053 "/px


def _hdr(cards):
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    return h


def _ps():
    from setiastro.saspro import plate_solver
    return plate_solver


def _wcs_cards(scale_arcsec):
    s = scale_arcsec / 3600.0
    return {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 83.8, "CRVAL2": -5.4,
            "CRPIX1": 500.0, "CRPIX2": 400.0, "CD1_1": -s, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": s}


def test_scale_hint_uses_xpixsz_as_written_on_binned_frames():
    ps = _ps()
    h = _hdr(SGP_BIN2)
    assert ps._estimate_scale_arcsec_from_header(h) == pytest.approx(SCALE_BIN2, rel=1e-6)
    assert ps._compute_scale_arcsec_per_pix(h) == pytest.approx(SCALE_BIN2, rel=1e-6)


def test_native_pixsize1_alone_gets_the_binning():
    # INDI writes the native sensor pixel as PIXSIZE1 beside a binned XPIXSZ.
    ps = _ps()
    h = _hdr({"PIXSIZE1": 2.315, "PIXSIZE2": 2.315, "XBINNING": 2, "YBINNING": 2, "FOCALLEN": 950.0})
    assert ps._header_pixel_um(h) == pytest.approx(4.63)
    assert ps._estimate_scale_arcsec_from_header(h) == pytest.approx(SCALE_BIN2, rel=1e-6)
    both = _hdr({"XPIXSZ": 4.63, "PIXSIZE1": 2.315, "XBINNING": 2, "FOCALLEN": 950.0})
    assert ps._header_pixel_um(both) == pytest.approx(4.63)


def test_focallen_written_back_from_a_binned_solve_is_the_real_one():
    # The old write-back gave 1900 mm here: XPIXSZ (already binned) x XBINNING again.
    ps = _ps()
    base = _hdr(dict(SGP_BIN2, FOCALLEN=900.0))
    out = ps._merge_wcs_into_base_header(base, _hdr(_wcs_cards(SCALE_BIN2)))
    assert float(out["FOCALLEN"]) == pytest.approx(950.0, abs=0.01)
    assert any("900.0 -> 950.0" in str(c) for c in out.get("HISTORY", []))


def test_unbinned_frames_are_unchanged():
    ps = _ps()
    h = _hdr({"XPIXSZ": 3.76, "XBINNING": 1, "FOCALLEN": 303.0})
    assert ps._estimate_scale_arcsec_from_header(h) == pytest.approx(206.264806 * 3.76 / 303.0, rel=1e-6)
    out = ps._merge_wcs_into_base_header(h, _hdr(_wcs_cards(206.264806 * 3.76 / 303.0)))
    assert float(out["FOCALLEN"]) == pytest.approx(303.0, abs=0.01)


@pytest.mark.parametrize("cards, true_scale", [
    (SGP_BIN2, SCALE_BIN2),                                                     # camera-binned
    ({"XPIXSZ": 3.76, "FOCALLEN": 303.0}, 206.264806 * 3.76 / 303.0),           # no binning card
    ({"PIXSIZE1": 2.315, "XBINNING": 2, "YBINNING": 2, "FOCALLEN": 950.0}, SCALE_BIN2),
])
def test_exoplanet_software_binning_doubles_the_hint(cards, true_scale):
    # The exoplanet tool solves a 2x2 binned plane: the hint must be exactly 2x,
    # or the blind solver's +/-30% scale gate throws every candidate away.
    from setiastro.saspro.exoplanet_detector import _rescale_header_for_binning
    ps = _ps()
    binned = _rescale_header_for_binning(_hdr(cards), 2)
    assert ps._estimate_scale_arcsec_from_header(binned) == pytest.approx(2 * true_scale, rel=1e-6)
    # ... and the focal length written back from that solve is still the real one.
    out = ps._merge_wcs_into_base_header(binned, _hdr(_wcs_cards(2 * true_scale)))
    assert float(out["FOCALLEN"]) == pytest.approx(float(cards["FOCALLEN"]), abs=0.01)
