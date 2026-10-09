"""Skip Registration and Integrate: per-frame measurement of aligned frames.

The Skip path (``integrate_registered_images``) swaps each calibrated tree frame
for its registered twin in Aligned_Images and measures the twin for star count,
FWHM and weight. SASpro writes those twins as FITS colour with NAXIS3=3, which
astropy returns channel-first, (3, H, W), and with a black warp border.
These tests run the real method on a stub and stop it right after the weights.
"""
from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from setiastro.saspro import stacking_measure_worker as measure_worker  # noqa: E402
from setiastro.saspro.stacking_suite import StackingSuiteDialog  # noqa: E402

H, W = 960, 640
SKY = 0.005
NOISE = 0.0005
STAR_AMP_MAX = 0.04
GROUP = "LP - 10.0s (640x960) [G80]"

# name: (star FWHM px, star amplitude scale, noise seed, warp (degrees, dx, dy))
# "ref" is the registration reference (identity warp, no border). "warped" is the
# same exposure (same stars, seeing and noise) with a warp whose zero fill covers
# its left edge. "soft" and "faint" are worse frames whose left edge holds sky.
FRAMES = {
    "ref": (8.0, 1.0, 11, (0.0, 0.0, 0.0)),
    "warped": (8.0, 1.0, 11, (1.5, 20.0, -7.0)),
    "soft": (13.0, 1.0, 13, (-2.0, -25.0, 6.0)),
    "faint": (8.0, 0.2, 14, (2.5, -25.0, 12.0)),
}


def setUpModule():
    global _app
    _app = QApplication.instance() or QApplication([])


def _sky_rgb(fwhm, amp_scale, noise_seed, n_stars=120):
    """Linear RGB star field (H, W, 3). Same stars for every frame; peak
    amplitudes log-uniform from the noise level up to STAR_AMP_MAX."""
    rng = np.random.default_rng(7)
    img = np.full((H, W, 3), SKY, np.float32)
    sigma = fwhm / 2.3548
    r = int(4 * sigma) + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for _ in range(n_stars):
        cy, cx = rng.uniform(40, H - 40), rng.uniform(40, W - 40)
        amp = float(np.exp(rng.uniform(np.log(NOISE), np.log(STAR_AMP_MAX)))) * amp_scale
        iy, ix = int(cy), int(cx)
        psf = amp * np.exp(-((yy - (cy - iy)) ** 2 + (xx - (cx - ix)) ** 2) / (2 * sigma ** 2))
        img[iy - r:iy + r + 1, ix - r:ix + r + 1, :] += psf[..., None].astype(np.float32)
    noise_rng = np.random.default_rng(noise_seed)
    img += noise_rng.normal(0.0, NOISE, img.shape).astype(np.float32)
    return img


def _cfa_grbg(rgb):
    """2-D GRBG mosaic of an RGB frame, like a calibrated OSC light."""
    cfa = np.empty((H, W), np.float32)
    cfa[0::2, 0::2] = rgb[0::2, 0::2, 1]
    cfa[0::2, 1::2] = rgb[0::2, 1::2, 0]
    cfa[1::2, 0::2] = rgb[1::2, 0::2, 2]
    cfa[1::2, 1::2] = rgb[1::2, 1::2, 1]
    return cfa


def _warp(rgb, deg, dx, dy):
    """Registration-style warp: rotation + shift, zero-filled border."""
    m = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), deg, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = cv2.warpAffine(rgb[..., c], m, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out


def _write_dataset(root, binning=1):
    """Calibrated/<stem>_c.fit (CFA) + Aligned_Images/<hash>_<stem>_c_n_r.fit (3, H, W).
    binning goes into XBINNING/YBINNING; the pixels are the same either way."""
    cal_dir = os.path.join(root, "Calibrated")
    al_dir = os.path.join(root, "Aligned_Images")
    os.makedirs(cal_dir)
    os.makedirs(al_dir)
    tree = []
    for name, (fwhm, amp, seed, (deg, dx, dy)) in FRAMES.items():
        rgb = _sky_rgb(fwhm, amp, seed)
        stem = f"32e542_Light_Pan2_10.0s_LP_{name}"
        h = fits.Header()
        h["EXPTIME"] = 10.0
        h["BAYERPAT"] = "GRBG"
        h["DEBAYERED"] = (False, "Mono / CFA")
        h["XBINNING"] = binning
        h["YBINNING"] = binning
        cal = os.path.join(cal_dir, f"{stem}_c.fit")
        fits.PrimaryHDU(_cfa_grbg(rgb), h).writeto(cal)
        h["DEBAYERED"] = (True, "Color frame")
        aligned = np.transpose(_warp(rgb, deg, dx, dy), (2, 0, 1))
        fits.PrimaryHDU(np.ascontiguousarray(aligned), h).writeto(
            os.path.join(al_dir, f"a355e6_{stem}_c_n_r.fit"))
        tree.append(cal)
    return {GROUP: tree}


class _StopAfterWeights(BaseException):
    pass


class _Settings:
    def __init__(self, values):
        self._v = dict(values)

    def value(self, key, default=None, type=None):
        v = self._v.get(key, default)
        return type(v) if (type is not None and v is not None) else v


class _SkipRun:
    """StackingSuiteDialog stand-in: binds the real dialog methods on demand and
    stubs the UI plumbing. integrate_registered_images stops right after the
    frame weights, at its first drizzle-stamp read."""

    def __init__(self, stacking_dir, tree):
        self.stacking_directory = stacking_dir
        self._tree = tree
        self.light_files = {}
        self.reference_frame = None
        self._reg_current_set = None
        self.settings = _Settings({
            "stacking/drizzle_enabled": False,
            "stacking/save_rejection_layers": False,
            "stacking/autocrop_enabled": False,
        })
        self.log = []

    def __getattr__(self, name):
        raw = inspect.getattr_static(StackingSuiteDialog, name)
        if isinstance(raw, staticmethod):
            return raw.__func__
        if isinstance(raw, classmethod):
            return raw.__get__(None, StackingSuiteDialog)
        if callable(raw):
            return types.MethodType(raw, self)
        return raw

    def tr(self, s):
        return s

    def update_status(self, msg):
        self.log.append(str(msg))

    def _set_registration_busy(self, *a, **k):
        pass

    def _start_exec_monitor(self):
        pass

    def _reset_cancel(self):
        pass

    def _cancelled(self):
        return False

    def _confirm_disk_budget(self, *a, **k):
        return True

    def extract_light_files_from_tree(self, **k):
        self.light_files = {g: list(v) for g, v in self._tree.items()}

    def _read_drizzle_stamp(self, path):
        raise _StopAfterWeights()

    def run(self):
        try:
            self.integrate_registered_images()
        except _StopAfterWeights:
            pass
        return self

    def measured(self):
        """{frame name: (star count, fwhm, normalized weight)} from the weights log.
        Skip logs the aligned twin (_c_n_r), Register the calibrated frame (_c)."""
        pat = re.compile(r"_LP_(\w+?)_c(?:_n_r)?\.fit → StarCount=(\d+),.*?FWHM~([\d.]+),"
                         r".*?Normalized=([\d.]+)")
        out = {}
        for m in pat.finditer("\n".join(self.log)):
            out[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
        return out


class SkipRegistrationMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="saspro-skipmeas-")
        tree = _write_dataset(cls.tmp)
        cls.run_ = _SkipRun(cls.tmp, tree).run()
        cls.m = cls.run_.measured()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_aligned_twins_are_the_frames_integrated(self):
        # Guard: the run reached the weights with every frame swapped for its twin.
        log = "\n".join(self.run_.log)
        self.assertIn(f"Matched {len(FRAMES)}/{len(FRAMES)}", log)
        self.assertEqual(set(self.m), set(FRAMES), log[-2000:])

    def test_every_frame_has_stars(self):
        counts = {k: v[0] for k, v in self.m.items()}
        self.assertTrue(counts and all(c > 0 for c in counts.values()), counts)

    def test_every_frame_has_a_fwhm(self):
        fwhm = {k: v[1] for k, v in self.m.items()}
        self.assertTrue(fwhm and all(f > 0 for f in fwhm.values()), fwhm)

    def test_sharp_frame_measures_smaller_fwhm_than_soft_frame(self):
        self.assertLess(self.m["ref"][1], self.m["soft"][1], self.m)

    def test_star_counts_dont_depend_on_the_warp_border(self):
        # "warped" is "ref" plus a registration warp and its black border.
        ref, warped = self.m["ref"][0], self.m["warped"][0]
        self.assertGreater(min(ref, warped), 0, self.m)
        self.assertLess(max(ref, warped) / min(ref, warped), 1.5, self.m)

    def test_frames_measure_like_register_measures_their_originals(self):
        # Register and Integrate measures the calibrated frame with measure_file.
        # The weights code in integrate_registered_images says the two paths must
        # weight the same frames the same way, so the inputs must agree too.
        cal_dir = os.path.join(self.tmp, "Calibrated")
        for name, (count, fwhm, _w) in self.m.items():
            cal = os.path.join(cal_dir, f"32e542_Light_Pan2_10.0s_LP_{name}_c.fit")
            status, _fp, payload = measure_worker.measure_file(cal, 1, 1)
            self.assertEqual(status, "ok")
            reg_count, reg_fwhm = payload[2], payload[4]
            with self.subTest(frame=name):
                self.assertAlmostEqual(count, reg_count, delta=0.1 * reg_count)
                self.assertAlmostEqual(fwhm, reg_fwhm, delta=0.1 * reg_fwhm)

    def test_weights_dont_fall_back_to_background_proxy(self):
        self.assertNotIn("No stars detected", "\n".join(self.run_.log))


class _RegisterRun(_SkipRun):
    """The same stand-in driving register_images, which stops right after its
    frame weights, when it loads the reference frame."""

    def __init__(self, stacking_dir, tree):
        super().__init__(stacking_dir, tree)
        self.reg_sets = {}
        self.frame_set_of = {}
        self.star_trail_mode = False
        self._reg_queue = None
        self.deleted_calibrated_files = []
        self.reg_tree = types.SimpleNamespace(selectedItems=lambda: [])

    def _precheck_registered_counterparts(self):
        return "register_all"

    def _maybe_warn_cfa_low_frames(self):
        pass

    def _load_image_any(self, path):
        raise _StopAfterWeights()

    def run(self):
        try:
            self.register_images()
        except _StopAfterWeights:
            pass
        return self


class RegisterSkipParityTests(unittest.TestCase):
    """Register and Integrate, and Skip Registration and Integrate on its
    output, log the same star count, FWHM and normalized weight per frame."""
    binning = 1

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="saspro-parity-")
        tree = _write_dataset(cls.tmp, binning=cls.binning)
        runs = []
        for run_cls in (_RegisterRun, _SkipRun):
            run = run_cls(cls.tmp, tree)
            # Same measure_file either way; threads keep the test fast.
            run.settings._v["stacking/measure_use_processes"] = False
            runs.append(run.run())
        cls.reg_run, cls.skip_run = runs
        cls.reg, cls.skip = cls.reg_run.measured(), cls.skip_run.measured()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_both_paths_reached_their_weights(self):
        self.assertEqual(set(self.reg), set(FRAMES), "\n".join(self.reg_run.log)[-2000:])
        self.assertEqual(set(self.skip), set(FRAMES), "\n".join(self.skip_run.log)[-2000:])

    def test_skip_logs_registers_numbers(self):
        for name in FRAMES:
            with self.subTest(frame=name):
                self.assertEqual(self.skip.get(name), self.reg.get(name))


class RegisterSkipParityBinnedTests(RegisterSkipParityTests):
    """2x2-binned frames: Register measures at the set's smallest binning, so
    Skip must measure the originals at that binning too."""
    binning = 2


class _InlinePool:
    """ProcessPoolExecutor stand-in: records each job and runs it inline."""
    submitted = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        type(self).submitted.append((fn, args))
        fut = Future()
        fut.set_result(fn(*args))
        return fut

    def shutdown(self, *a, **k):
        pass


class _BrokenPool(_InlinePool):
    def submit(self, fn, *args):
        raise BrokenProcessPool("worker processes can't start")


class SkipRegistrationProcessPoolTests(unittest.TestCase):
    """The calibrated originals are measured in worker processes, as Register
    and Integrate measures them: measure_file is GIL-bound, so in threads it
    runs about 8x slower (0.20 vs 0.025 s per frame on 3318 Seestar frames)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="saspro-skippool-")
        cls.tree = _write_dataset(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, pool_cls):
        pool_cls.submitted = []
        with mock.patch("concurrent.futures.ProcessPoolExecutor", pool_cls), \
                mock.patch("setiastro.saspro.worker_env.check_process_pools"), \
                mock.patch("setiastro.saspro.worker_env.process_pools_ok",
                           return_value=True):
            return _SkipRun(self.tmp, self.tree).run()

    def test_originals_are_measured_in_worker_processes(self):
        run = self._run(_InlinePool)
        jobs = [(fn, args[0]) for fn, args in _InlinePool.submitted]
        self.assertTrue(all(fn is measure_worker.measure_file for fn, _ in jobs), jobs)
        self.assertEqual(sorted(p for _, p in jobs), sorted(self.tree[GROUP]))
        counts = {k: v[0] for k, v in run.measured().items()}
        self.assertEqual(set(counts), set(FRAMES), counts)
        self.assertTrue(all(c > 0 for c in counts.values()), counts)

    def test_falls_back_to_threads_when_processes_cant_start(self):
        run = self._run(_BrokenPool)
        self.assertIn("using threads instead", "\n".join(run.log))
        counts = {k: v[0] for k, v in run.measured().items()}
        self.assertEqual(set(counts), set(FRAMES), counts)
        self.assertTrue(all(c > 0 for c in counts.values()), counts)


class QuickPreviewChannelFirstTests(unittest.TestCase):
    """Both copies of _quick_preview_any must read FITS colour stored (3, H, W)
    as the same image as (H, W, 3); measure_file in the Register path uses the
    stacking_measure_worker copy."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="saspro-preview-")
        rgb = _sky_rgb(8.0, 1.0, 11)
        cls.chw = os.path.join(cls.tmp, "chw_n_r.fit")
        cls.hwc = os.path.join(cls.tmp, "hwc_n_r.fit")
        fits.PrimaryHDU(np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))).writeto(cls.chw)
        fits.PrimaryHDU(rgb).writeto(cls.hwc)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _dialog_preview(self, path):
        stub = types.SimpleNamespace(
            _bin_from_header_fast_any=lambda fp: (1, 1))
        return StackingSuiteDialog._quick_preview_any(stub, path, 1, 1)

    def _check(self, reader):
        ref = reader(self.hwc)
        self.assertEqual(ref.shape, (H // 2, W // 2))   # control: (H, W, 3) is fine
        got = reader(self.chw)
        self.assertEqual(got.shape, ref.shape)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)

    def test_dialog_preview_reads_channel_first_colour(self):
        self._check(self._dialog_preview)

    def test_measure_worker_preview_reads_channel_first_colour(self):
        self._check(lambda p: measure_worker._quick_preview_any(p, 1, 1))


if __name__ == "__main__":
    unittest.main()
