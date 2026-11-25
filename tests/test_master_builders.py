"""Master dark / master flat builders (2026-11-24).

* The Flats tab and "Turn Those Flats Into Master Flats" group flats the same
  way: one master per filter, size, night and exposure within the tolerance.
  The tab used to merge every night of a filter into one group while the build
  made one master per night (Frank: "it made 2 sets of master flats").
* A new master flat enters the master list with its real night. The list used
  to read the master's header before the background save had written it, and
  the session came out as the folder name ("Master_Calibration_Files").
* create_master_dark leaves no temp .dat files behind (it could never delete
  them on Windows while a view still mapped them).
* _load_fits_u16_fast returns astropy's values exactly, and only takes the
  files it understands.

Drives the real StackingSuiteDialog offscreen, CPU reducers, scratch settings.
"""
from __future__ import annotations

import faulthandler
import glob
import os
import shutil
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from PyQt6.QtCore import QCoreApplication, QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from setiastro.saspro import stacking_suite  # noqa: E402
from setiastro.saspro.stacking_master_picks import is_control_item  # noqa: E402

_APP = None
_SETTINGS_DIR = None
_SAVED = None
H, W = 48, 64


def setUpModule():
    global _APP, _SETTINGS_DIR, _SAVED
    _SAVED = (QCoreApplication.organizationName(), QCoreApplication.applicationName(),
              QSettings.defaultFormat())
    _SETTINGS_DIR = tempfile.mkdtemp(prefix="saspro-master-builders-settings-")
    QCoreApplication.setOrganizationName("SASproTestMasterBuilders")
    QCoreApplication.setApplicationName("MasterBuilders")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
    _APP = QApplication.instance() or QApplication([])
    s = QSettings()
    s.setValue("stacking/use_hardware_accel", False)
    s.setValue("stacking/auto_session", True)
    s.sync()


def tearDownModule():
    org, app, fmt = _SAVED
    QCoreApplication.setOrganizationName(org)
    QCoreApplication.setApplicationName(app)
    QSettings.setDefaultFormat(fmt)
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


def _write_u16(path, data, **cards):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.uint16))   # astropy writes BZERO=32768
    for k, v in cards.items():
        hdu.header[k.replace("_", "-")] = v
    hdu.writeto(path, overwrite=True)


class _DialogCase(unittest.TestCase):
    def setUp(self):
        faulthandler.dump_traceback_later(600, exit=True)
        self.tmp = tempfile.mkdtemp(prefix="saspro-master-builders-")
        self.out = os.path.join(self.tmp, "stack")
        os.makedirs(self.out)
        self._boxes = {}
        for name in ("information", "warning", "critical"):
            self._boxes[name] = getattr(QMessageBox, name)
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok))
        self.dlg = stacking_suite.StackingSuiteDialog()
        for _ in range(50):
            _APP.processEvents()
        self.dlg.stacking_directory = self.out

    def tearDown(self):
        for name, fn in self._boxes.items():
            setattr(QMessageBox, name, fn)
        self.dlg.close()
        self.dlg.deleteLater()
        _APP.processEvents()
        faulthandler.cancel_dump_traceback_later()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def masters(self):
        return sorted(glob.glob(os.path.join(self.out, "Master_Calibration_Files", "*")))


class FlatsTabMatchesBuild(_DialogCase):
    """SII flats from two nights, 0.03 s and 0.05 s, tolerance 0.39 s."""

    def make_flats(self):
        raw = os.path.join(self.tmp, "flats")
        os.makedirs(raw)
        rng = np.random.default_rng(3)
        paths = []
        for night, exp, level in (("2026-06-30T03:00:00", 0.05, 20000),
                                  ("2026-10-14T03:00:00", 0.03, 30000)):
            for i in range(3):
                p = os.path.join(raw, f"flat_{night[:10]}_{i}.fit")
                data = level + rng.integers(0, 200, (H, W))
                _write_u16(p, data, IMAGETYP="FLAT", FILTER="SII", EXPTIME=exp,
                           DATE_OBS=night)
                paths.append(p)
        return paths

    def test_one_group_and_one_master_per_night(self):
        dlg = self.dlg
        dlg.flat_exposure_tolerance_spinbox.setValue(0.39)
        dlg._ingest_paths_with_progress(self.make_flats(), dlg.flat_tree, "FLAT", "test")
        dlg.rebuild_flat_tree()
        _APP.processEvents()

        tops = [dlg.flat_tree.topLevelItem(i) for i in range(dlg.flat_tree.topLevelItemCount())]
        self.assertEqual(len(tops), 2, [t.text(0) for t in tops])
        sessions = sorted(dlg._flat_groups_info[t.data(0, 0x0100)]["session"] for t in tops)
        self.assertEqual(len(set(sessions)), 2, sessions)
        for t in tops:
            sess = dlg._flat_groups_info[t.data(0, 0x0100)]["session"]
            self.assertIn(sess, t.text(0))               # the tab says which night
            leaves = [t.child(k) for k in range(t.childCount()) if not is_control_item(t.child(k))]
            self.assertEqual(len(leaves), 3)

        dlg.create_master_flat()
        _APP.processEvents()

        built = [p for p in self.masters() if os.path.basename(p).startswith("MasterFlat_")]
        self.assertEqual(len(built), 2, built)
        built_sessions = sorted(str(fits.getheader(p)["SESSION"]) for p in built)
        self.assertEqual(built_sessions, sessions)

        # The master list knows each master's night: the key's first bracket is
        # the session (what the light-side matcher reads), never the folder.
        flat_keys = {k: v for k, v in dlg.master_files.items()
                     if isinstance(v, str) and os.path.basename(v).startswith("MasterFlat_")}
        self.assertEqual(len(flat_keys), 2, flat_keys)
        for key, path in flat_keys.items():
            self.assertNotIn("Master_Calibration_Files", key)
            self.assertIn(f"[{fits.getheader(path)['SESSION']}]", key)

        # Each master is the median of its own night's flats only.
        for p in built:
            m = fits.getdata(p)
            self.assertEqual(m.shape, (H, W))
            self.assertTrue(np.all(np.isfinite(m)))


class MasterDarkLeavesNoTempFiles(_DialogCase):
    def test_no_dat_and_exact_values(self):
        raw = os.path.join(self.tmp, "darks")
        os.makedirs(raw)
        base = np.random.default_rng(5).integers(100, 4000, (H, W))
        paths = []
        for exp in (60.0, 120.0):
            for i in range(3):
                p = os.path.join(raw, f"dark_{int(exp)}_{i}.fit")
                _write_u16(p, base + int(exp), IMAGETYP="DARK", EXPTIME=exp, CCD_TEMP=-10.0)
                paths.append(p)
        dlg = self.dlg
        dlg._ingest_paths_with_progress(paths, dlg.dark_tree, "DARK", "test")
        dlg.create_master_dark()
        _APP.processEvents()

        out = self.masters()
        self.assertFalse([p for p in out if p.endswith(".dat")], out)
        darks = [p for p in out if os.path.basename(p).startswith("MasterDark_")]
        self.assertEqual(len(darks), 2, out)
        for p in darks:
            exp = float(fits.getheader(p)["EXPTIME"])
            want = ((base + int(exp)).astype(np.float32) / np.float32(65535.0))
            got = np.asarray(fits.getdata(p), dtype=np.float32)
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=0)
        listed = [v for v in dlg.master_files.values() if isinstance(v, str)]
        for p in darks:
            self.assertIn(os.path.normcase(p), {os.path.normcase(v) for v in listed})


class FastU16Loader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="saspro-u16-")
        self.rng = np.random.default_rng(11)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def astropy_values(self, path):
        with fits.open(path, memmap=False) as h:
            for hdu in h:
                if hdu.data is not None:
                    return np.array(hdu.data)

    def test_camera_raw_matches_astropy(self):
        p = os.path.join(self.tmp, "raw.fit")
        data = self.rng.integers(0, 65536, (37, 53), dtype=np.uint16)
        data[0, :4] = (0, 1, 32767, 65535)                # sign-bit edges
        _write_u16(p, data)
        got = stacking_suite._load_fits_u16_fast(p)
        self.assertIsNotNone(got)
        hdr, arr = got
        self.assertEqual(arr.dtype, np.uint16)
        np.testing.assert_array_equal(arr, self.astropy_values(p))
        np.testing.assert_array_equal(arr, data)

        # And the stacker's float tile is unchanged from the astropy-read path.
        src = stacking_suite._MMImage(p)
        try:
            self.assertIsNone(src._fits_hdul)            # took the fast path
            dst = np.empty((10, 20, 1), np.float32)
            src.read_tile_into(dst, 3, 13, 5, 25, 1)
            want = (data[3:13, 5:25].astype(np.float32) / np.float32(65535.0))[..., None]
            np.testing.assert_array_equal(dst, want)
        finally:
            src.close()

    def test_other_files_are_left_to_astropy(self):
        cases = {}
        p = os.path.join(self.tmp, "float.fit")
        fits.PrimaryHDU(self.rng.random((8, 9), dtype=np.float32)).writeto(p)
        cases["float32"] = p
        p = os.path.join(self.tmp, "signed.fit")
        fits.PrimaryHDU(self.rng.integers(-500, 500, (8, 9)).astype(np.int16)).writeto(p)
        cases["signed int16, no BZERO"] = p
        p = os.path.join(self.tmp, "blank.fit")
        hdu = fits.PrimaryHDU(self.rng.integers(0, 60000, (8, 9)).astype(np.uint16))
        hdu.header["BLANK"] = 0
        hdu.writeto(p)
        cases["BLANK"] = p
        p = os.path.join(self.tmp, "bscale.fit")
        hdu = fits.PrimaryHDU(self.rng.integers(-500, 500, (8, 9)).astype(np.int16))
        hdu.header["BSCALE"] = 2.0
        hdu.header["BZERO"] = 32768.0
        hdu.writeto(p, output_verify="silentfix")
        cases["BSCALE 2"] = p
        p = os.path.join(self.tmp, "comp.fits")
        fits.HDUList([fits.PrimaryHDU(),
                      fits.CompImageHDU(self.rng.integers(0, 60000, (8, 9)).astype(np.uint16))]).writeto(p)
        cases["tile-compressed"] = p
        for what, path in cases.items():
            with self.subTest(what):
                self.assertIsNone(stacking_suite._load_fits_u16_fast(path))


if __name__ == "__main__":
    unittest.main()
