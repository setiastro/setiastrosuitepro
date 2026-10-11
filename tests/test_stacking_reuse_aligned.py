"""Bug #3 (#126): "Register and Integrate" stacks frames that failed registration,
unaligned and not debayered.

These tests drive the real ``StackingSuiteDialog.register_images()`` headless on
synthetic Seestar-like data (2-D CFA calibrated frames, BAYERPAT=GRBG,
DEBAYERED=F; registered twins are channel-first RGB ``*_n_r.fit`` files in
Aligned_Images). Only three things are replaced:

* the star-alignment thread (``StarRegistrationThread``) by a fake that writes
  a ``*_n_r.fit`` twin for frames that "align", and reports an alignment
  failure (matrix None) or a finalize failure (matrix solved, no output) for
  the frames a test asks for;
* ``QMessageBox.exec``, which clicks the default button of the pre-check
  prompt (the "accept" choice) or a button named by the test;
* the integrator hand-off, which records the frame lists that would be
  stacked instead of stacking them.

Every assertion is about which files reach the integrator, so the tests don't
depend on how the fix is implemented.
"""
from __future__ import annotations

import faulthandler
import os
import re
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

H, W = 64, 96
_RNG = np.random.default_rng(11)
_STARS = [(_RNG.uniform(6, H - 6), _RNG.uniform(6, W - 6), _RNG.uniform(0.3, 0.8))
          for _ in range(30)]
_TAG_RE = re.compile(r"frame([A-Z])_c")

_APP = None
_SETTINGS_DIR = None
_SAVED_APP_IDS = None


def setUpModule():
    """Keep the dialog's QSettings() away from the user's real settings."""
    global _APP, _SETTINGS_DIR, _SAVED_APP_IDS
    _SAVED_APP_IDS = (QCoreApplication.organizationName(),
                      QCoreApplication.applicationName(),
                      QSettings.defaultFormat())
    _SETTINGS_DIR = tempfile.mkdtemp(prefix="saspro-reuse-aligned-settings-")
    QCoreApplication.setOrganizationName("SASproTestReuseAligned")
    QCoreApplication.setApplicationName("ReuseAligned")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)
    _APP = QApplication.instance() or QApplication([])


def tearDownModule():
    org, app, fmt = _SAVED_APP_IDS
    QCoreApplication.setOrganizationName(org)
    QCoreApplication.setApplicationName(app)
    QSettings.setDefaultFormat(fmt)
    shutil.rmtree(_SETTINGS_DIR, ignore_errors=True)


def _star_field(dy=0.0, dx=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = 0.1 + 0.005 * _RNG.standard_normal((H, W)).astype(np.float32)
    for y, x, a in _STARS:
        img += a * np.exp(-(((yy - y - dy) ** 2 + (xx - x - dx) ** 2) / (2 * 1.6 ** 2)))
    return np.clip(img, 0, 1).astype(np.float32)


def _tag(path):
    m = _TAG_RE.search(os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


class _FakeAlignThread(QObject):
    """Stands in for star_alignment.StarRegistrationThread.

    Exposes the attributes on_registration_complete() reads. Frames whose tag
    is in ``align_fail`` get matrix None (the real thread skips their output);
    tags in ``finalize_fail`` get a solved matrix but no written output, so
    file_key_to_current_path keeps the input path, as the real thread does.
    """
    progress_update = pyqtSignal(str)
    registration_complete = pyqtSignal(bool, str)
    progress_step = pyqtSignal(int, int)

    align_fail: set = set()
    finalize_fail: set = set()
    cancel = False
    registered: list = []

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
        cls = type(self)
        n = len(self.original_files)
        if cls.cancel:
            self._parent.request_cancel()
            self.done = True
            self.registration_complete.emit(True, f"Registration complete. Valid frames: 0/{n}.")
            return
        ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        for f in self.original_files:
            tag = _tag(f)
            cls.registered.append(tag)
            if tag in cls.align_fail:
                self.alignment_matrices[f] = None
                continue
            self.alignment_matrices[f] = ident
            if tag in cls.finalize_fail:
                continue
            data = fits.getdata(f).astype(np.float32)
            if data.ndim == 3 and data.shape[-1] == 3:
                data = np.moveaxis(data, -1, 0)
            stem = os.path.splitext(os.path.basename(f))[0]
            out = os.path.join(self.output_directory, stem + "_r.fit")
            hdr = fits.Header()
            hdr["DEBAYERED"] = (True, "Color frame")
            fits.PrimaryHDU(data, hdr).writeto(out, overwrite=True)
            self.file_key_to_current_path[f] = out
            self.drizzle_xforms[f] = ("affine", ident)
        self.transform_deltas = [[0.0] * n]
        self.done = True
        self.registration_complete.emit(True, f"Registration complete. Valid frames: {n}/{n}.")


class ReuseAlignedFramesTests(unittest.TestCase):
    """Frames handed to the integrator after the 'already registered' pre-check."""

    def setUp(self):
        faulthandler.dump_traceback_later(600, exit=True)  # never hang the suite on a modal
        self.tmp = tempfile.mkdtemp(prefix="saspro-reuse-aligned-")
        self.cal = os.path.join(self.tmp, "Calibrated")
        self.ali = os.path.join(self.tmp, "Aligned_Images")
        os.makedirs(self.cal)
        os.makedirs(self.ali)
        self.button = None          # None => default (accept) button
        self.prompts = []
        self.integrated = None

        s = QSettings()
        s.clear()
        s.setValue("stacking/dir", self.tmp)
        s.setValue("stacking/refinement_passes", 1)   # the reported runs were "Fast mode"
        s.sync()

        _FakeAlignThread.align_fail = set()
        _FakeAlignThread.finalize_fail = set()
        _FakeAlignThread.cancel = False
        _FakeAlignThread.registered = []

        def fake_exec(box):
            buttons = box.buttons()
            self.prompts.append((box.windowTitle(), [b.text() for b in buttons]))
            target = box.defaultButton()
            if self.button:
                named = [b for b in buttons if b.text() == self.button]
                if named:
                    target = named[0]
            if target is not None:
                target.click()
            return 0

        def fake_after_align_worker(*_a, **kw):
            # Full-registration flow: on_registration_complete builds this worker.
            self.integrated = [p for lst in kw["light_files"].values() for p in lst]
            return mock.MagicMock()

        for target, attr, new in (
            (stacking_suite, "StarRegistrationThread", _FakeAlignThread),
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

    # ── data ──────────────────────────────────────────────────────────────
    def cal_path(self, tag):
        return os.path.join(self.cal, f"abc123_Light_LP_10s_frame{tag}_c.fit")

    def add_calibrated(self, tag, obj=None):
        h = fits.Header()
        h["IMAGETYP"] = "LIGHT"
        h["FILTER"] = "LP"
        h["EXPTIME"] = 10.0
        h["GAIN"] = 80
        h["XBINNING"] = 1
        h["YBINNING"] = 1
        h["BAYERPAT"] = "GRBG"
        h["DEBAYERED"] = (False, "Mono / CFA")
        if obj:
            h["OBJECT"] = obj
        data = _star_field(_RNG.uniform(-2, 2), _RNG.uniform(-2, 2))
        fits.PrimaryHDU(data, h).writeto(self.cal_path(tag))

    def add_twin(self, tag):
        """A registered counterpart from an earlier run, newer than the calibrated frame."""
        p = os.path.join(self.ali, f"a355e6_abc123_Light_LP_10s_frame{tag}_c_n_r.fit")
        h = fits.Header()
        h["DEBAYERED"] = (True, "Color frame")
        fits.PrimaryHDU(np.stack([_star_field()] * 3, axis=0), h).writeto(p)
        t = os.path.getmtime(self.cal_path(tag)) + 5
        os.utime(p, (t, t))

    # ── dialog ────────────────────────────────────────────────────────────
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
            # integrate_registered_images() → _run_mfdeconv_then_continue() hand-off
            self.integrated = [p for lst in aligned_light_files.values() for p in lst]
            dlg._set_registration_busy(False)

        dlg._start_after_align_worker = capture
        return dlg

    def remove_from_tree(self, tags):
        """Select the frames' rows in the Image Integration tree and click Remove Selected."""
        tree = self.dlg.reg_tree
        tree.clearSelection()
        for i in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(i)
            for j in range(top.childCount()):
                leaf = top.child(j)
                fp = leaf.data(0, stacking_suite.Qt.ItemDataRole.UserRole) or ""
                if _tag(fp) in tags:
                    leaf.setSelected(True)
        self.dlg.clear_tree_selection_registration(tree)

    def click_register_and_integrate(self):
        self.integrated = None
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
        return self.integrated

    # ── assertions ────────────────────────────────────────────────────────
    def assertIntegratedExactly(self, integrated, tags):
        self.assertIsNotNone(integrated, "nothing reached the integrator")
        names = sorted(os.path.relpath(p, self.tmp) for p in integrated)
        self.assertEqual(sorted(_tag(p) for p in integrated), sorted(tags),
                         f"frames handed to the integrator: {names}")
        unregistered = [n for n in names if not n.endswith("_n_r.fit")]
        self.assertEqual(unregistered, [],
                         f"unregistered frames handed to the integrator: {names}")

    # ── tests ─────────────────────────────────────────────────────────────
    def test_register_new_then_integrate_drops_frame_that_failed_alignment(self):
        """Accepting 'Register N New → Integrate All': a new frame that fails
        alignment must not be stacked (today its raw calibrated file is, as-is)."""
        for t in "ABCD":
            self.add_calibrated(t)
        self.add_twin("A")
        self.add_twin("B")
        _FakeAlignThread.align_fail = {"D"}
        self.open_dialog()

        integrated = self.click_register_and_integrate()

        self.assertEqual(self.prompts[0][0], "Partially Registered")
        self.assertEqual(sorted(_FakeAlignThread.registered), ["C", "D"])
        self.assertIntegratedExactly(integrated, ["A", "B", "C"])

    def test_register_new_then_integrate_drops_frame_whose_aligned_output_was_not_written(self):
        """Accepting 'Register N New → Integrate All': a frame whose transform was
        solved but whose aligned output failed to write must not be stacked."""
        for t in "ABCE":
            self.add_calibrated(t)
        self.add_twin("A")
        self.add_twin("B")
        _FakeAlignThread.finalize_fail = {"E"}
        self.open_dialog()

        integrated = self.click_register_and_integrate()

        self.assertEqual(self.prompts[0][0], "Partially Registered")
        self.assertIntegratedExactly(integrated, ["A", "B", "C"])

    def test_register_new_then_integrate_hands_only_debayered_frames_to_integrator(self):
        """Claim (c): nothing that reaches the integrator may still be a 2-D CFA mosaic."""
        for t in "ABCDE":
            self.add_calibrated(t)
        self.add_twin("A")
        self.add_twin("B")
        _FakeAlignThread.align_fail = {"D"}
        _FakeAlignThread.finalize_fail = {"E"}
        self.open_dialog()

        integrated = self.click_register_and_integrate()

        self.assertIsNotNone(integrated, "nothing reached the integrator")
        raw_cfa = []
        for p in integrated:
            hdr = fits.getheader(p)
            if hdr["NAXIS"] != 3 or not hdr.get("DEBAYERED", False):
                raw_cfa.append(f"{os.path.relpath(p, self.tmp)} (NAXIS={hdr['NAXIS']}, "
                               f"BAYERPAT={hdr.get('BAYERPAT')}, "
                               f"DEBAYERED={hdr.get('DEBAYERED')})")
        self.assertEqual(raw_cfa, [], "non-debayered frames handed to the integrator")

    def test_skip_to_integrate_for_default_set_keeps_other_sets_out(self):
        """Multi-set run (frames without OBJECT stay in 'Default', frames with
        OBJECT get their own set). Accepting 'Skip → Integrate' for the Default
        set must stack only that set's registered twins, not the other set's
        unregistered frames."""
        for t in "AB":
            self.add_calibrated(t)
            self.add_twin(t)
        for t in "XY":
            self.add_calibrated(t, obj="M42")
        self.open_dialog()

        integrated = self.click_register_and_integrate()

        self.assertEqual(self.prompts[0][0], "Already Registered")
        self.assertIntegratedExactly(integrated, ["A", "B"])

    def test_rerun_after_cancel_does_not_stack_frame_removed_from_tree(self):
        """Claim (b). Run 1 (Register All) is cancelled; the user then removes a
        frame with Remove Selected and clicks Register and Integrate again. The
        removed frame must not be stacked (today the run reuses the frame list
        queued by the cancelled run)."""
        for t in "ABCD":
            self.add_calibrated(t)
        self.add_twin("A")
        self.add_twin("B")
        self.button = "Register All"
        self.open_dialog()

        _FakeAlignThread.cancel = True
        self.click_register_and_integrate()
        _FakeAlignThread.cancel = False
        self.remove_from_tree({"C"})
        integrated = self.click_register_and_integrate()

        self.assertEqual(len(self.prompts), 2, "the rerun didn't ask again")
        self.assertIsNotNone(integrated, "nothing reached the integrator")
        self.assertNotIn("C", [_tag(p) for p in integrated],
                         f"removed frame stacked: {[os.path.basename(p) for p in integrated]}")

    def test_rerun_after_no_frames_to_stack_does_not_stack_frame_removed_from_tree(self):
        """Like the cancel case, but run 1 (Register All) ends with 'No frames to
        stack; aborting.' because every frame failed alignment."""
        for t in "ABCD":
            self.add_calibrated(t)
        self.add_twin("A")
        self.add_twin("B")
        self.button = "Register All"
        self.open_dialog()

        _FakeAlignThread.align_fail = set("ABCD")
        self.assertIsNone(self.click_register_and_integrate())
        _FakeAlignThread.align_fail = set()
        self.remove_from_tree({"C"})
        integrated = self.click_register_and_integrate()

        self.assertEqual(len(self.prompts), 2, "the rerun didn't ask again")
        self.assertIsNotNone(integrated, "nothing reached the integrator")
        self.assertNotIn("C", [_tag(p) for p in integrated],
                         f"removed frame stacked: {[os.path.basename(p) for p in integrated]}")

    def test_register_all_drops_frame_whose_aligned_output_was_not_written(self):
        """'Register All' button: a finalize failure must not be
        stacked as its unaligned Normalized_Images input."""
        for t in "ABCE":
            self.add_calibrated(t)
        self.add_twin("A")
        self.add_twin("B")
        _FakeAlignThread.finalize_fail = {"E"}
        self.button = "Register All"
        self.open_dialog()

        integrated = self.click_register_and_integrate()

        self.assertEqual(sorted(_FakeAlignThread.registered), ["A", "B", "C", "E"])
        self.assertIntegratedExactly(integrated, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
