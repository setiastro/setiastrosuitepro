"""CFA drizzle deposits sparse frames that were normalized like the dense ones.

Reported 2026-11-17 (Ernst): a fine grid over a CFA-drizzled SASpro stack. The
sparse copy each OSC light gets for the drizzle deposit (_n_cfa.fit) was split
off before normalization and written as split -- no per-channel sky match, no
ABE -- while the dense _n.fit that rejection, weights and AFN are measured on
got both. Every position inside a Bayer cell is filled by a different subset
of frames, so frames with different skies average to a different level at
each position: a fixed grid in the signal. Synthetic, noise-free, 16 frames
with a +-25 % sky swing: 13-25 % of the sky peak to peak, and identical for
the axis-aligned and the rotated square drop.

The integration test drives the real ``register_images()`` normalization;
only star alignment and the integrator hand-off are faked, as in
test_stacking_reuse_aligned.
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
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from setiastro.saspro import stacking_suite  # noqa: E402

H, W = 96, 128
_RNG = np.random.default_rng(31)
_STARS = [(_RNG.uniform(6, W - 6), _RNG.uniform(6, H - 6), _RNG.uniform(0.2, 0.8)) for _ in range(40)]
SKY = np.array([0.060, 0.070, 0.065])
GAINS = [1.0, 0.78, 1.22, 0.9, 1.15]          # sky level per frame, frame 0 is the reference

_APP = None
_SETTINGS_DIR = None
_SAVED_APP_IDS = None


def setUpModule():
    """Keep the dialog's QSettings() away from the user's real settings."""
    global _APP, _SETTINGS_DIR, _SAVED_APP_IDS
    _SAVED_APP_IDS = (QCoreApplication.organizationName(),
                      QCoreApplication.applicationName(),
                      QSettings.defaultFormat())
    _SETTINGS_DIR = tempfile.mkdtemp(prefix="saspro-cfa-norm-settings-")
    QCoreApplication.setOrganizationName("SASproTestCFADrizzleNorm")
    QCoreApplication.setApplicationName("CFADrizzleNorm")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
    _APP = QApplication.instance() or QApplication([])


def tearDownModule():
    org, app, fmt = _SAVED_APP_IDS
    QCoreApplication.setOrganizationName(org)
    QCoreApplication.setApplicationName(app)
    QSettings.setDefaultFormat(fmt)
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


def _mosaic(gain, dy, dx):
    """RGGB mosaic of a star field on a sky of SKY * gain."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    lum = np.zeros((H, W))
    for x, y, a in _STARS:
        lum += a * np.exp(-(((yy - y - dy) ** 2 + (xx - x - dx) ** 2) / (2 * 1.4 ** 2)))
    rgb = SKY * gain + lum[..., None]
    bayer = np.zeros((H, W), int)
    bayer[0::2, 1::2] = 1
    bayer[1::2, 0::2] = 1
    bayer[1::2, 1::2] = 2
    mos = np.take_along_axis(rgb, bayer[..., None], axis=2)[..., 0]
    mos += 0.002 * _RNG.standard_normal((H, W))
    return np.clip(mos, 1e-4, 1).astype(np.float32)


class _FakeAlignThread(QObject):
    progress_update = pyqtSignal(str)
    registration_complete = pyqtSignal(bool, str)
    progress_step = pyqtSignal(int, int)

    def __init__(self, ref_path, files, out_dir, max_refinement_passes=3,
                 shift_tolerance=0.2, parent_window=None):
        super().__init__()
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
            if data.ndim == 3 and data.shape[-1] == 3:
                data = np.moveaxis(data, -1, 0)
            out = os.path.join(self.output_directory,
                               os.path.splitext(os.path.basename(f))[0] + "_r.fit")
            fits.PrimaryHDU(data).writeto(out, overwrite=True)
            self.alignment_matrices[f] = ident
            self.file_key_to_current_path[f] = out
            self.drizzle_xforms[f] = ("affine", ident)
        n = len(self.original_files)
        self.transform_deltas = [[0.0] * n]
        self.done = True
        self.registration_complete.emit(True, f"Registration complete. Valid frames: {n}/{n}.")


class SparseFromDenseTests(unittest.TestCase):

    def test_takes_dense_values_at_native_sites_only(self):
        sparse = np.zeros((4, 4, 3), np.float32)
        sparse[0::2, 0::2, 0] = 0.5          # R sites
        dense = np.full((4, 4, 3), 0.25, np.float32)
        out = stacking_suite._cfa_sparse_from_dense(sparse, dense)
        np.testing.assert_array_equal(out[..., 0] != 0, sparse[..., 0] != 0)
        self.assertTrue(np.all(out[0::2, 0::2, 0] == 0.25))
        self.assertFalse(np.any(out[..., 1:]))

    def test_non_finite_becomes_no_data(self):
        sparse = np.ones((2, 2, 3), np.float32)
        dense = np.ones((2, 2, 3), np.float32)
        dense[0, 0, 0] = np.nan
        self.assertEqual(stacking_suite._cfa_sparse_from_dense(sparse, dense)[0, 0, 0], 0.0)

    def test_resampled_frame_has_no_sparse_copy(self):
        self.assertIsNone(stacking_suite._cfa_sparse_from_dense(
            np.ones((4, 4, 3), np.float32), np.ones((8, 8, 3), np.float32)))


class CFADrizzleNormalizationTests(unittest.TestCase):

    def setUp(self):
        faulthandler.dump_traceback_later(600, exit=True)
        self.tmp = tempfile.mkdtemp(prefix="saspro-cfa-norm-")
        self.cal = os.path.join(self.tmp, "Calibrated")
        os.makedirs(self.cal)
        self.integrated = None
        s = QSettings()
        s.clear()
        for k, v in (("stacking/dir", self.tmp), ("stacking/refinement_passes", 1),
                     ("stacking/drizzle_enabled", True), ("stacking/cfa_drizzle", True),
                     ("stacking/debayer_method", "rcd")):
            s.setValue(k, v)
        s.sync()

        def fake_exec(box):
            b = box.defaultButton()
            if b is not None:
                b.click()
            return 0

        def fake_after_align_worker(*_a, **kw):
            self.integrated = [p for lst in kw["light_files"].values() for p in lst]
            return mock.MagicMock()

        def fake_question(*_a, **_k):
            # "CFA Drizzle: Low Sample Count" -> No keeps CFA drizzle on. A
            # static QMessageBox.question never reaches the patched exec, so
            # unstubbed it waits forever on an offscreen modal.
            return QMessageBox.StandardButton.No

        for target, attr, new in (
            (stacking_suite, "StarRegistrationThread", _FakeAlignThread),
            (stacking_suite, "AfterAlignWorker", fake_after_align_worker),
            (QMessageBox, "exec", fake_exec),
            (QMessageBox, "question", fake_question),
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
            dlg.close()
            dlg.deleteLater()
            _APP.processEvents()
        faulthandler.cancel_dump_traceback_later()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sparse_siblings_are_normalized_like_the_dense_frames(self):
        for i, g in enumerate(GAINS):
            h = fits.Header()
            h["IMAGETYP"] = "LIGHT"
            h["FILTER"] = "OSC"
            h["EXPTIME"] = 60.0
            h["BAYERPAT"] = "RGGB"
            dy, dx = (0.0, 0.0) if i == 0 else _RNG.uniform(-2, 2, 2)
            fits.PrimaryHDU(_mosaic(g, dy, dx), h).writeto(
                os.path.join(self.cal, f"abc123_light_{i:02d}_c.fit"))

        dlg = stacking_suite.StackingSuiteDialog()
        self.dlg = dlg
        for _ in range(500):
            _APP.processEvents()
            if getattr(dlg, "_cal_scan_stop", None) is None and dlg.reg_tree.topLevelItemCount():
                break
            time.sleep(0.005)
        dlg._ensure_reg_tree_loaded()
        dlg.auto_accept_ref_cb.setChecked(True)
        dlg._confirm_disk_budget = lambda *a, **k: True
        self.assertTrue(dlg.cfa_drizzle_cb.isChecked(), "CFA drizzle did not switch on")
        dlg.register_images()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            _APP.processEvents()
            th = getattr(dlg, "alignment_thread", None)
            if self.integrated is not None or th is None or not th.isRunning():
                break
            time.sleep(0.005)
        for _ in range(20):
            _APP.processEvents()

        norm = os.path.join(self.tmp, "Normalized_Images")
        sparse_files = sorted(glob.glob(os.path.join(norm, "*_n_cfa.fit")))
        self.assertEqual(len(sparse_files), len(GAINS), "a sparse sibling is missing")
        sky = []
        for sp_path in sparse_files:
            dense = fits.getdata(sp_path.replace("_n_cfa.fit", "_n.fit")).astype(np.float32)
            sparse = fits.getdata(sp_path).astype(np.float32)
            self.assertEqual(dense.shape, sparse.shape)
            sites = sparse != 0
            self.assertGreater(sites.mean(), 0.3)
            np.testing.assert_allclose(sparse[sites], dense[sites], rtol=0, atol=1e-6,
                                       err_msg=f"{os.path.basename(sp_path)} differs from its _n.fit")
            sky.append([np.median(sparse[..., c][sites[..., c]]) for c in range(3)])
        sky = np.array(sky)
        spread = (sky.max(axis=0) - sky.min(axis=0)) / sky.mean(axis=0)
        # unnormalized, these frames differ by up to 44 % (0.78 .. 1.22)
        self.assertTrue(np.all(spread < 0.03), f"sparse sky levels not matched: {spread}")


if __name__ == "__main__":
    unittest.main()
