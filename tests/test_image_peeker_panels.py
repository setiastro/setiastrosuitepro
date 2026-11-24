from __future__ import annotations

import math
import os

import numpy as np
import pytest

from astropy.io import fits


def _ipp():
    from setiastro.saspro import image_peeker_pro
    return image_peeker_pro


def _frame(size=600, bg=0.08):
    return np.full((size, size), bg, np.float32)


def _star(img, x0, y0, amp, fwhm, q=1.0, theta=0.7, ss=3):
    """Pixel-integrated elliptical Gaussian; fwhm is the geometric-mean FWHM, q = b/a."""
    H, W = img.shape
    R = int(6 * fwhm / math.sqrt(q)) + 2
    px0, px1 = max(0, int(x0) - R), min(W, int(x0) + R)
    py0, py1 = max(0, int(y0) - R), min(H, int(y0) + R)
    yy, xx = np.mgrid[py0 * ss:py1 * ss, px0 * ss:px1 * ss].astype(np.float64)
    xx = (xx + 0.5) / ss - 0.5 - x0
    yy = (yy + 0.5) / ss - 0.5 - y0
    c, s = math.cos(theta), math.sin(theta)
    fa, fb = fwhm / math.sqrt(q), fwhm * math.sqrt(q)
    u = (xx * c + yy * s) / fa
    v = (-xx * s + yy * c) / fb
    sig = 1.0 / 2.3548
    prof = amp * np.exp(-(u * u + v * v) / (2 * sig * sig))
    img[py0:py1, px0:px1] += prof.reshape(py1 - py0, ss, px1 - px0, ss).mean(axis=(1, 3)).astype(np.float32)


def _field(rng, n, fwhm, snr_lo, snr_hi, q=1.0, size=600, noise=0.002, margin=30):
    img = _frame(size)
    for _ in range(n):
        x0, y0 = rng.uniform(margin, size - margin, 2)
        _star(img, x0, y0, noise * 10 ** rng.uniform(math.log10(snr_lo), math.log10(snr_hi)), fwhm, q)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    return img


def _measure(img, panel=None):
    ipp = _ipp()
    size = img.shape[0]
    panel = panel or size - 100
    return ipp.measure_panel_stars(img, 50, 50, panel)


def test_fwhm_does_not_depend_on_star_brightness():
    # SEP's isophotal 2.3548*sqrt(ab) reads ~0.7x on faint stars and >1.1x on
    # bright ones; panels would differ just by which stars they caught.
    rng = np.random.default_rng(1)
    faint = _measure(_field(rng, 25, 4.0, 15, 30))
    bright = _measure(_field(rng, 25, 4.0, 300, 600))
    assert faint["fwhm_px"] == pytest.approx(4.0, rel=0.05)
    assert bright["fwhm_px"] == pytest.approx(4.0, rel=0.03)


def test_elongated_stars_read_their_eccentricity_and_round_ones_read_low():
    rng = np.random.default_rng(2)
    oval = _measure(_field(rng, 25, 4.0, 30, 400, q=0.8))
    round_ = _measure(_field(rng, 25, 4.0, 8, 20))
    assert oval["ecc"] == pytest.approx(0.6, abs=0.05)
    # The median of each faint round star's own eccentricity reads 0.30 here,
    # from noise alone; the median shape cancels the noise.
    assert round_["ecc"] < 0.2


def test_big_stars_at_long_focal_lengths_are_measured():
    # A stamp sized from the median detection (hot pixels pull it down) left
    # every 10 px star of a real 0.5"/px sub unmeasured.
    rng = np.random.default_rng(3)
    img = _field(rng, 12, 10.0, 40, 300, margin=40)
    for _ in range(25):
        img[rng.integers(0, 600), rng.integers(0, 600)] += 0.5
    r = _measure(img)
    assert r["n_measured"] >= 8
    assert r["fwhm_px"] == pytest.approx(10.0, rel=0.05)


def test_hot_pixels_are_not_stars():
    rng = np.random.default_rng(4)
    img = _field(rng, 15, 4.0, 50, 300)
    clean = _measure(img.copy())["n"]
    for _ in range(40):
        img[rng.integers(60, 540), rng.integers(60, 540)] += 0.3
    r = _measure(img)
    assert abs(r["n"] - clean) <= 1
    assert r["fwhm_px"] == pytest.approx(4.0, rel=0.05)


def test_clipped_stars_are_counted_but_not_measured():
    rng = np.random.default_rng(5)
    img = _field(rng, 20, 4.0, 40, 300)
    for x, y in ((150, 150), (400, 380), (300, 220)):
        _star(img, x, y, 5.0, 4.0)
    np.minimum(img, 1.0, out=img)
    r = _measure(img)
    assert r["n_clipped"] >= 3
    assert r["fwhm_px"] == pytest.approx(4.0, rel=0.05)


def test_nebula_knots_do_not_set_the_panel_fwhm():
    # Real drizzled Soul master: stars 7.7-8.5 px, knots 15-24 px, and the
    # plain median (12.3 px) fell between them.
    rng = np.random.default_rng(6)
    img = _field(rng, 7, 4.0, 60, 300)
    for _ in range(5):
        x0, y0 = rng.uniform(60, 540, 2)
        _star(img, x0, y0, 0.04, 14.0)
    r = _measure(img)
    assert r["fwhm_px"] == pytest.approx(4.0, rel=0.06)


def test_zero_filled_border_is_no_data():
    rng = np.random.default_rng(7)
    img = _field(rng, 20, 4.0, 40, 300)
    img[:, :300] = 0.0
    r = _measure(img)
    assert 0 < r["n"] <= 14
    assert r["fwhm_px"] == pytest.approx(4.0, rel=0.06)


def test_a_grid_is_searched_in_one_sep_call(monkeypatch):
    # Every sep.extract allocates the GUI process's 20M pixel stack (~250 ms).
    ipp = _ipp()
    calls = []
    real = ipp.sep.extract
    monkeypatch.setattr(ipp.sep, "extract", lambda *a, **k: calls.append(1) or real(*a, **k))
    rng = np.random.default_rng(8)
    img = _field(rng, 80, 4.0, 40, 300, size=900)
    xs, ys = ipp._panel_grid(900, 900, 3, 200)
    res = ipp.measure_panels(img, [(x, y) for y in ys for x in xs], 200)
    assert len(res) == 9 and len(calls) == 1
    assert all(r["fwhm_px"] == pytest.approx(4.0, rel=0.08) for r in res if r["fwhm_px"])


def test_panel_tags_name_only_unambiguous_panels():
    tag = _ipp()._panel_tag
    assert [tag(r, c, 3) for r in range(3) for c in range(3)] == \
        ["TL", "T", "TR", "L", "C", "R", "BL", "B", "BR"]
    assert [tag(0, c, 4) for c in range(4)] == ["TL", "", "", "TR"]
    assert tag(2, 2, 5) == "C" and tag(0, 2, 5) == "T" and tag(1, 1, 5) == ""


def test_xpixsz_is_used_as_written_not_multiplied_by_binning():
    # SGP writes XPIXSZ=4.63 with XBINNING=2 on the 2.315 um ASI294MM.
    h = fits.Header()
    h["XPIXSZ"] = 4.63
    h["XBINNING"] = 2
    h["FOCALLEN"] = 950
    o = _ipp()._image_optics({"original_header": h})
    assert o["pixel_um"] == pytest.approx(4.63)
    assert o["focal_mm"] == pytest.approx(950)
    assert o["aperture_mm"] is None and o["scale_arcsec"] is None


def test_native_pixsize1_gets_the_binning():
    # INDI's PIXSIZE1 is the native sensor pixel; only a header without XPIXSZ uses it.
    h = fits.Header()
    h["PIXSIZE1"] = 2.315
    h["XBINNING"] = 2
    o = _ipp()._image_optics({"original_header": h})
    assert o["pixel_um"] == pytest.approx(4.63)
    h["XPIXSZ"] = 4.63
    assert _ipp()._image_optics({"original_header": h})["pixel_um"] == pytest.approx(4.63)


def test_aperture_from_focal_ratio_and_scale_from_the_solve():
    h = fits.Header()
    h["XPIXSZ"] = 3.76
    h["FOCALLEN"] = 303.0
    h["FOCRATIO"] = 4.0
    for k, v in (("CTYPE1", "RA---TAN"), ("CTYPE2", "DEC--TAN"), ("CRVAL1", 10.0), ("CRVAL2", 20.0),
                 ("CRPIX1", 100.0), ("CRPIX2", 100.0), ("CD1_1", -2.56 / 3600), ("CD1_2", 0.0),
                 ("CD2_1", 0.0), ("CD2_2", 2.56 / 3600)):
        h[k] = v
    o = _ipp()._image_optics({"original_header": h})
    assert o["aperture_mm"] == pytest.approx(75.75)
    assert o["scale_arcsec"] == pytest.approx(2.56, rel=1e-6)


def test_no_header_gives_nothing():
    o = _ipp()._image_optics({})
    assert all(o[k] is None for k in ("pixel_um", "focal_mm", "aperture_mm", "scale_arcsec"))


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_dialog_takes_optics_from_the_image_and_does_not_remember_them(qapp, tmp_path):
    from PyQt6.QtCore import QSettings, QEventLoop, QTimer
    ipp = _ipp()

    class Doc:
        pass

    rng = np.random.default_rng(9)
    doc = Doc()
    doc.image = _field(rng, 150, 4.0, 40, 300, size=1200)
    h = fits.Header()
    h["XPIXSZ"] = 2.315
    h["FOCALLEN"] = 950.0
    doc.metadata = {"original_header": h}

    s = QSettings(str(tmp_path / "peek.ini"), QSettings.Format.IniFormat)
    s.setValue("pixel_size_um", 4.8)
    s.setValue("focal_length_mm", 800.0)
    dlg = ipp.ImagePeekerDialogPro(None, doc, s)
    try:
        assert dlg.pixel_size_input.value() == pytest.approx(2.315)   # three decimals, not 2.32
        assert dlg.focal_length_input.value() == pytest.approx(950.0)
        assert "pixel size ÷ focal length" in dlg.scale_value_label.text()

        dlg._refresh_mosaic()
        loop = QEventLoop()
        poll = QTimer()
        poll.timeout.connect(lambda: dlg._stats_shown_key is not None and loop.quit())
        poll.start(50)
        QTimer.singleShot(30000, loop.quit)
        loop.exec()
        poll.stop()
        cells = dlg.preview_pane._panel_cells
        assert len(cells) == 9 and cells[4]["tag"] == "C" and cells[4]["centre"]
        # 4.0 px at 206.265 * 2.315 / 950 = 0.5026 "/px
        big = [float(c["big"].rstrip("″")) for c in cells if c["big"] not in ("", "—")]
        assert big and all(v == pytest.approx(4.0 * 0.5026, rel=0.08) for v in big)
        assert dlg.stats_lbl.text().startswith("FWHM: centre")

        dlg._remember_optics(s)
        assert float(s.value("pixel_size_um")) == pytest.approx(4.8)
        assert float(s.value("focal_length_mm")) == pytest.approx(800.0)
        dlg.focal_length_input.setValue(1000.0)        # typed by the user
        dlg._remember_optics(s)
        assert float(s.value("focal_length_mm")) == pytest.approx(1000.0)
    finally:
        dlg._stats_timer.stop()
        dlg.deleteLater()
