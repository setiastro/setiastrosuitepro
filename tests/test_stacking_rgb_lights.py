"""Lights that are already RGB before calibration must keep their geometry.

Reported 2026-11-17 (ddorman58): CR3 -> "Convert to FITS" -> stack failed with
"Reference stars in source image are less than the minimum value (3)" on every
frame. The converted and calibrated frames were fine. Calibration writes a
colour frame as the raw in-memory (H, W, 3) array, so its header reads
NAXIS1=3, NAXIS2=W, NAXIS3=H, and the normalize-canvas probe took that as a
3 x W frame. Every frame was then squashed onto a W x 3 canvas
("Normalize canvas 6024x4020 -> 3x6024") and lost all its stars.

Any light that is RGB before calibration takes this path: converted DSLR raws,
RGB TIFF, RGB FITS. CFA and mono lights stay 2-D through calibration and never
did.

The integration test drives the real ``StackingSuiteDialog.register_images()``
headless. Only star alignment is faked (it records what it is handed and
writes an identity-aligned twin), plus the integrator hand-off and modal
prompts, the same as test_stacking_reuse_aligned.
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
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from PyQt6.QtCore import QCoreApplication, QObject, QSettings, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox, QTreeWidgetItem  # noqa: E402

from setiastro.saspro import stacking_suite  # noqa: E402
from setiastro.saspro.calibration_io import write_calibrated_fast  # noqa: E402

H, W = 120, 180
_RNG = np.random.default_rng(23)
_STARS = [(_RNG.uniform(8, H - 8), _RNG.uniform(8, W - 8), _RNG.uniform(0.2, 0.8))
          for _ in range(45)]

_APP = None
_SETTINGS_DIR = None
_SAVED_APP_IDS = None


def setUpModule():
    """Keep the dialog's QSettings() away from the user's real settings."""
    global _APP, _SETTINGS_DIR, _SAVED_APP_IDS
    _SAVED_APP_IDS = (QCoreApplication.organizationName(),
                      QCoreApplication.applicationName(),
                      QSettings.defaultFormat())
    _SETTINGS_DIR = tempfile.mkdtemp(prefix="saspro-rgb-lights-settings-")
    QCoreApplication.setOrganizationName("SASproTestRGBLights")
    QCoreApplication.setApplicationName("RGBLights")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
    _APP = QApplication.instance() or QApplication([])


def tearDownModule():
    org, app, fmt = _SAVED_APP_IDS
    QCoreApplication.setOrganizationName(org)
    QCoreApplication.setApplicationName(app)
    QSettings.setDefaultFormat(fmt)
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


def _star_rgb(dy=0.0, dx=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lum = 0.02 + 0.002 * _RNG.standard_normal((H, W)).astype(np.float32)
    for y, x, a in _STARS:
        lum += a * np.exp(-(((yy - y - dy) ** 2 + (xx - x - dx) ** 2) / (2 * 1.6 ** 2)))
    rgb = np.stack([lum * 0.8, lum, lum * 0.9], axis=-1)
    return np.clip(rgb, 0, 1).astype(np.float32)          # (H, W, 3)


class _RecordingAlignThread(QObject):
    """Stands in for star_alignment.StarRegistrationThread: records the shape
    of every frame it is handed and writes an identity-aligned twin."""
    progress_update = pyqtSignal(str)
    registration_complete = pyqtSignal(bool, str)
    progress_step = pyqtSignal(int, int)

    seen: dict = {}

    def __init__(self, ref_path, files, out_dir, max_refinement_passes=3,
                 shift_tolerance=0.2, parent_window=None):
        super().__init__()
        self._parent = parent_window
        self.reference = ref_path
        self.original_files = [os.path.normpath(f) for f in files]
        self.output_directory = out_dir
        self.max_refinement_passes = max_refinement_passes
        self.alignment_matrices = {}
        self.transform_deltas = []
        self.drizzle_xforms = {}
        self.reference_image_2d = np.zeros((H, W), np.float32)
        self.file_key_to_current_path = {f: f for f in self.original_files}
        self.done = False

    def isRunning(self):
        return not self.done

    def wait(self, *_a):
        return True

    def start(self):
        QTimer.singleShot(0, self._run)

    def _run(self):
        ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for f in self.original_files:
            data = fits.getdata(f).astype(np.float32)
            type(self).seen[os.path.basename(f)] = data.shape
            if data.ndim == 3 and data.shape[-1] == 3:
                data = np.moveaxis(data, -1, 0)
            self.alignment_matrices[f] = ident
            out = os.path.join(self.output_directory,
                               os.path.splitext(os.path.basename(f))[0] + "_r.fit")
            hdr = fits.Header()
            hdr["DEBAYERED"] = (True, "Color frame")
            fits.PrimaryHDU(data, hdr).writeto(out, overwrite=True)
            self.file_key_to_current_path[f] = out
            self.drizzle_xforms[f] = ("affine", ident)
        n = len(self.original_files)
        self.transform_deltas = [[0.0] * n]
        self.done = True
        self.registration_complete.emit(True, f"Registration complete. Valid frames: {n}/{n}.")


def _write(path, data):
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


class ProbeFrameHWTests(unittest.TestCase):
    """(H, W) from the header, whichever way the planes are stored."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="saspro-probe-hw-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def probe(self, name, data):
        p = os.path.join(self.tmp, name)
        _write(p, data)
        return stacking_suite.StackingSuiteDialog._probe_frame_hw(p)

    def test_mono_or_cfa(self):
        self.assertEqual(self.probe("mono.fit", np.zeros((40, 60), np.float32)), (40, 60))

    def test_channel_first_rgb(self):
        self.assertEqual(self.probe("chw.fit", np.zeros((3, 40, 60), np.float32)), (40, 60))

    def test_calibrated_rgb_as_the_pipeline_writes_it(self):
        """Through the real calibrated-frame writer, not a hand-built layout."""
        p = os.path.join(self.tmp, "cal_c.fit")
        write_calibrated_fast(p, np.zeros((40, 60, 3), np.float32), fits.Header(), False)
        self.assertEqual(fits.getheader(p)["NAXIS1"], 3, "writer layout changed; revisit this test")
        self.assertEqual(stacking_suite.StackingSuiteDialog._probe_frame_hw(p), (40, 60))


class RGBLightsRegistrationTests(unittest.TestCase):

    def setUp(self):
        faulthandler.dump_traceback_later(600, exit=True)  # never hang the suite on a modal
        self.tmp = tempfile.mkdtemp(prefix="saspro-rgb-lights-")
        self.cal = os.path.join(self.tmp, "Calibrated")
        os.makedirs(self.cal)
        self.integrated = None

        s = QSettings()
        s.clear()
        s.setValue("stacking/dir", self.tmp)
        s.setValue("stacking/refinement_passes", 1)
        s.sync()
        _RecordingAlignThread.seen = {}

        def fake_exec(box):
            target = box.defaultButton()
            if target is not None:
                target.click()
            return 0

        def fake_after_align_worker(*_a, **kw):
            # Full-registration flow: on_registration_complete builds this worker.
            self.integrated = [p for lst in kw["light_files"].values() for p in lst]
            return mock.MagicMock()

        for target, attr, new in (
            (stacking_suite, "StarRegistrationThread", _RecordingAlignThread),
            (stacking_suite, "AfterAlignWorker", fake_after_align_worker),
            (QMessageBox, "exec", fake_exec),
        ):
            p = mock.patch.object(target, attr, new)
            p.start()
            self.addCleanup(p.stop)
        self.dlg = None

    def tearDown(self):
        dlg = self.dlg
        if dlg is not None:
            th = getattr(dlg, "post_thread", None)
            if th is not None:
                th.quit()
                th.wait(5000)
            pd = getattr(dlg, "post_progress", None)
            if pd is not None:
                pd.close()
            dlg.close()
            dlg.deleteLater()
            _APP.processEvents()
        faulthandler.cancel_dump_traceback_later()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_calibrated(self, i):
        """A calibrated RGB light exactly as calibrate_lights writes one."""
        h = fits.Header()
        h["IMAGETYP"] = "LIGHT"
        h["EXPTIME"] = 60.0
        h["BAYERPAT"] = "RGGB"          # left over from the raw, as converted frames carry it
        p = os.path.join(self.cal, f"6e8f79_IMG_{i:04d}_c.fit")
        write_calibrated_fast(p, _star_rgb(_RNG.uniform(-2, 2), _RNG.uniform(-2, 2)), h, False)

    def open_dialog(self):
        dlg = stacking_suite.StackingSuiteDialog()
        self.dlg = dlg
        for _ in range(500):
            _APP.processEvents()
            if getattr(dlg, "_cal_scan_stop", None) is None and dlg.reg_tree.topLevelItemCount():
                break
            time.sleep(0.005)
        dlg._ensure_reg_tree_loaded()
        dlg.auto_accept_ref_cb.setChecked(True)   # no reference-review dialog
        dlg._confirm_disk_budget = lambda *a, **k: True

        def capture(aligned_light_files):
            self.integrated = [p for lst in aligned_light_files.values() for p in lst]
            dlg._set_registration_busy(False)

        dlg._start_after_align_worker = capture
        return dlg

    def register(self):
        self.dlg.register_images()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            _APP.processEvents()
            th = getattr(self.dlg, "alignment_thread", None)
            if self.integrated is not None or th is None or not th.isRunning():
                break
            time.sleep(0.005)
        for _ in range(20):
            _APP.processEvents()

    def test_calibrated_rgb_lights_keep_their_size_through_normalization(self):
        for i in range(3):
            self.add_calibrated(i)
        self.open_dialog()

        labels = [self.dlg.reg_tree.topLevelItem(i).text(0)
                  for i in range(self.dlg.reg_tree.topLevelItemCount())]
        self.assertTrue(any(f"{W}x{H}" in t for t in labels),
                        f"Image Registration group should show the frame size {W}x{H}: {labels}")

        self.register()

        normalized = sorted(glob.glob(os.path.join(self.tmp, "Normalized_Images", "*_n.fit")))
        self.assertEqual(len(normalized), 3, "normalization did not write every frame")
        for p in normalized:
            shape = fits.getdata(p).shape
            self.assertEqual(sorted(shape), sorted((H, W, 3)),
                             f"{os.path.basename(p)} came out {shape}, not {H}x{W}x3")
        self.assertEqual(len(_RecordingAlignThread.seen), 3)
        self.assertIsNotNone(self.integrated, "nothing reached the integrator")
        self.assertEqual(len(self.integrated), 3)


class _BatchSettingsOK:
    """Stands in for BatchSettingsDialog: OK with IMAGETYP=LIGHT, camera EXPTIME."""
    def __init__(self, *_a, **_k):
        pass

    def exec(self):
        return 1

    def get_values(self):
        return ("LIGHT", "", "None")


class ConvertToFitsDepthTests(unittest.TestCase):
    """Convert to FITS writes at the source's depth: 16-bit, like any astro
    camera's FITS, for 8/16-bit sources (camera raws load as 16-bit), and
    32-bit float only when the source itself is float. The 32-bit converted
    files from V1.23.9 on were 283 MB for a 24 MP raw. Calibrated frames are
    written elsewhere and stay 32-bit."""

    def setUp(self):
        faulthandler.dump_traceback_later(600, exit=True)
        self.tmp = tempfile.mkdtemp(prefix="saspro-convert-depth-")
        self.out = os.path.join(self.tmp, "converted")
        os.makedirs(self.out)
        p = mock.patch.object(stacking_suite, "BatchSettingsDialog", _BatchSettingsOK)
        p.start()
        self.addCleanup(p.stop)
        self.dlg = stacking_suite.StackingSuiteDialog()

    def tearDown(self):
        self.dlg.close()
        self.dlg.deleteLater()
        _APP.processEvents()
        faulthandler.cancel_dump_traceback_later()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def convert(self, name, data):
        import tifffile
        src = os.path.join(self.tmp, name)
        tifffile.imwrite(src, data)
        tree = self.dlg.conversion_tree
        tree.clear()
        item = QTreeWidgetItem([name, "Pending"])
        item.setData(0, 1000, src)
        tree.addTopLevelItem(item)
        self.dlg.conversion_output_directory = self.out
        self.dlg.convert_all_files()
        out = os.path.join(self.out, os.path.splitext(name)[0] + ".fit")
        self.assertTrue(os.path.exists(out), f"{name} was not converted")
        return out

    def test_16bit_source_converts_to_16bit_fits(self):
        src = (_star_rgb() * 65535).astype(np.uint16)
        out = self.convert("light16.tif", src)
        self.assertEqual(fits.getheader(out)["BITPIX"], 16)
        img, _hdr, _bd, mono = stacking_suite.load_image(out)
        self.assertFalse(mono)
        np.testing.assert_allclose(img, src.astype(np.float32) / 65535.0, atol=1.5 / 65535)

    def test_float_source_stays_32bit_float(self):
        src = _star_rgb()
        out = self.convert("light32.tif", src)
        self.assertEqual(fits.getheader(out)["BITPIX"], -32)
        img, *_ = stacking_suite.load_image(out)
        np.testing.assert_array_equal(img, src)


if __name__ == "__main__":
    unittest.main()
