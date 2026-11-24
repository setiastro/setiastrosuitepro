"""Exercise both drizzle passes, real registration, Gaia XP fitting and ImageMM."""
import os
from pathlib import Path
import sqlite3
import zlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest

from setiastro.saspro.blot_spcc import SpectralCalibration
from setiastro.saspro.blot_stack import (
    StackOptions, blot, detector_flags, sampling_geometry, stack,
)
from setiastro.saspro.gaia_database import WL_GRID


def test_detector_thresholds_and_background():
    rng = np.random.default_rng(30)
    model = rng.normal(0, .03, (40, 50)).astype("f4")
    raw = model + 12.
    raw[12, 15:17] += [7., 4.8]
    raw[20, 15] += 4.8
    error = np.ones_like(model)
    flags, = detector_flags(raw[None], None, [(model, error)], [np.eye(3)], 1., StackOptions())
    assert np.array_equal(np.argwhere(flags), [[12, 15], [12, 16]])
    # Bilinear blot must renormalize over surviving coverage.
    model[:] = 4.
    model[8, 8] = np.nan
    transform = np.eye(3)
    transform[0, 2] = .5
    pred, _ = blot(model, error, transform, raw.shape, 1.)
    assert pred[8, 8] == pytest.approx(4.)


def test_automatic_sampling():
    pattern = np.array([[0, 1], [1, 2]])
    scale, widths = sampling_geometry(StackOptions(aperture_mm=100, sampling=1.5),
                                     fits.Header(), pattern, 160)
    assert scale == pytest.approx(1.8165, rel=.001)
    assert widths == pytest.approx([2**-.5, .5, 2**-.5])


def field(tmp_path):
    rng = np.random.default_rng(412)
    size = 176
    xy = rng.uniform(16, size-16, (48, 2))
    xy = np.array([p for i, p in enumerate(xy) if np.all(np.linalg.norm(xy[:i]-p, axis=1) > 12)])
    amp = rng.uniform(.05, .22, len(xy))
    yy, xx = np.mgrid[:size, :size]
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [81, 83]
    wcs.wcs.cdelt = [-.00032, .00032]
    wcs.wcs.crval = [190., 45.]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = wcs.to_header()
    hdr["BAYERPAT"] = "RGGB"
    paths = []
    colors = np.array([[0, 1], [1, 2]])[yy % 2, xx % 2]
    for j, (dx, dy) in enumerate([(0, 0), (.6, 1.1), (-1.1, .4), (1.5, -.8), (-.3, -1.5)]):
        raw = np.empty_like(xx, dtype=np.float32)
        for c, (gain, shift) in enumerate(zip((.65, 1., 1.4), (.9, 0., -.8))):
            channel = np.full(raw.shape, .02 + j*.004 + c*.001)
            for (x, y), peak in zip(xy, amp):
                channel += gain*peak*np.exp(-((xx-x-dx-shift)**2+(yy-y-dy)**2)/(2*1.6**2))
            raw[colors == c] = channel[colors == c]
        raw += rng.normal(0., .0005, raw.shape).astype("f4")
        if j == 0:
            raw[8, 8] = 2.
        path = tmp_path / f"light{j}.fit"
        fits.writeto(path, raw, hdr)
        paths.append(path)
    return paths, xy, wcs


def calibration(tmp_path, xy, wcs):
    library = Path(__file__).resolve().parents[1] / "src/setiastro/data/SASP_data.fits"
    with fits.open(library) as hdus:
        sed = hdus["A0V"].data
        spectrum = np.interp(WL_GRID, sed["WAVELENGTH"]/10., sed["FLUX"]).astype("f4")
    with sqlite3.connect(tmp_path / "gaia_xp_field.sqlite") as db:
        db.execute("CREATE TABLE sources(source_id INTEGER PRIMARY KEY, ra REAL, dec REAL)")
        db.execute("CREATE TABLE spectra(source_id INTEGER PRIMARY KEY, flux_compressed BLOB, flux_error_compressed BLOB)")
        for i, (ra, dec) in enumerate(wcs.all_pix2world(xy, 0)):
            db.execute("INSERT INTO sources VALUES (?,?,?)", (i, ra, dec))
            db.execute("INSERT INTO spectra VALUES (?,?,NULL)", (i, zlib.compress(spectrum.tobytes())))
    return SpectralCalibration(tuple((library, "BAADER_"+c) for c in "RGB"),
                               (library, "100%_QE_CURVE"), (library, "A0V"), tmp_path)


def test_full_chain(tmp_path):
    paths, xy, wcs = field(tmp_path)
    spcc = calibration(tmp_path, xy, wcs)
    out = tmp_path / "master.fit"
    messages = []
    stack(paths, paths[0], out, StackOptions(scale=1.25, iterations=2, save_iterations=True),
          spcc, messages.append)
    assert sum(m.startswith("Registered ") for m in messages) == 5
    assert any("aperture drift R/G" in m for m in messages)
    assert not list(tmp_path.glob(".blot-*"))
    product, hdr = fits.getdata(out, header=True)
    assert product.shape == (3, 220, 220)
    assert np.isfinite(product).all()
    assert "BAYERPAT" not in hdr
    out_wcs = WCS(hdr, naxis=2)
    np.testing.assert_allclose(out_wcs.all_pix2world((xy+.5)*1.25-.5, 0), wcs.all_pix2world(xy, 0))
    snapshots, = tmp_path.glob("imagemm-*")
    seed = np.stack([fits.getdata(snapshots / f"ch{c}-iter000.fits") for c in (1, 2, 3)])
    assert seed[0, 6:15, 6:15].max() < .04  # includes the entire cosmic ray footprint
    apertures = np.column_stack(((xy+.5)*1.25-.5, np.full(len(xy), 7.)))
    for frame in (seed, product):
        ratios = spcc.ratios(frame, apertures, np.ones_like(frame, bool))
        np.testing.assert_allclose(np.nanmedian(ratios, axis=0), 1., atol=.06)
    assert not np.array_equal(seed, product)


def test_cancel_preserves_master(tmp_path):
    paths, _, _ = field(tmp_path)
    master = tmp_path / "existing.fit"
    master.write_bytes(b"original")
    logged = []
    def cancel():
        if any(m.startswith("First drizzle") for m in logged):
            raise InterruptedError
    with pytest.raises(InterruptedError):
        stack(paths, paths[0], master, StackOptions(align_colors=False), log=logged.append, check=cancel)
    assert master.read_bytes() == b"original"
    assert not list(tmp_path.glob(".blot-*"))



def test_missing_coverage_excludes_both_sides_of_mm():
    from setiastro.saspro.blot_stack import refine
    seed = np.ones((1, 23, 27), np.float32)
    cache = np.ones((3, 1, 2, 23, 27), np.float32)
    cache[0, 0, 1, 10, 12] = 0
    cache[1, 0, 1] *= .25
    args = (np.ones(1), np.zeros(1), np.full((3, 1), 3.), np.full((3, 1), .01),
            StackOptions(iterations=2, strength=0), lambda *_: None, lambda: None, lambda *_: None)
    baseline = refine(seed, cache, *args)
    cache[0, 0, 0, 10, 12] = np.nan
    np.testing.assert_allclose(refine(seed, cache, *args), baseline)


def test_gui_stack_action(tmp_path, monkeypatch):
    from PyQt6.QtCore import QCoreApplication, QEventLoop, QSettings, QTimer
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QPushButton
    from setiastro.saspro.blot_stack_dialog import BlotStackDialog
    from setiastro.saspro.stacking_suite import StackingSuiteDialog

    app = QApplication.instance() or QApplication([])
    paths, _, _ = field(tmp_path)
    output = tmp_path / "dialog.fit"
    old_ids = (QCoreApplication.organizationName(), QCoreApplication.applicationName())
    QCoreApplication.setOrganizationName("BlotStackTest")
    QCoreApplication.setApplicationName(tmp_path.name)
    suite = StackingSuiteDialog()
    suite.reference_frame = str(paths[2])
    monkeypatch.setattr(suite, "extract_light_files_from_tree", lambda: {"CFA": paths})
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), ""))
    errors = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _, title, message: errors.append(message))
    visited = []
    def run(dialog):
        dialog.settings = QSettings(str(tmp_path/"settings.ini"), QSettings.Format.IniFormat)
        visited.append(dialog.reference.currentData())
        dialog.spcc.setChecked(False)
        dialog.controls["align_colors"].setChecked(False)
        dialog.controls["iterations"].setValue(1)
        dialog.run_button.click()
        if dialog.worker is not None:
            loop = QEventLoop()
            dialog.worker.finished.connect(loop.quit)
            QTimer.singleShot(60000, loop.quit)
            loop.exec()
        app.processEvents()
    monkeypatch.setattr(BlotStackDialog, "exec", run)
    try:
        action, = [b for b in suite.findChildren(QPushButton) if "Drizzle → Blot" in b.text()]
        action.click()
        assert visited == [str(paths[2])]
        assert not errors
        assert fits.getdata(output).shape == (3, 176, 176)
    finally:
        suite.close()
        suite.deleteLater()
        app.processEvents()
        QSettings().clear()
        QCoreApplication.setOrganizationName(old_ids[0])
        QCoreApplication.setApplicationName(old_ids[1])


def test_scaled_sip_and_cfa_neighbors():
    from setiastro.saspro.blot_stack import scaled_header
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [51., 43.]
    wcs.wcs.cdelt = [-.001, .001]
    wcs.wcs.crval = [20., -30.]
    wcs.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    hdr = wcs.to_header()
    hdr.update(A_ORDER=2, B_ORDER=2, A_2_0=.0003, B_0_2=-.0002)
    points = np.array([[10., 10.], [60., 80.]])
    np.testing.assert_allclose(WCS(scaled_header(hdr, 2.3)).all_pix2world((points+.5)*2.3-.5, 0),
                               WCS(hdr).all_pix2world(points, 0), atol=1e-10)
    pattern = np.array([[0, 1], [1, 2]])
    model = np.zeros((3, 2, 24, 24), np.float32)
    model[:, 1] = 1.
    raw = np.full((1, 24, 24), 8., np.float32)
    raw[0, 10, 10:12] += (6., 4.5)
    flags = detector_flags(raw, pattern, model, [np.eye(3)]*3, 1., StackOptions())
    assert np.array_equal(np.argwhere(flags), [[0, 10, 10], [1, 10, 11]])
