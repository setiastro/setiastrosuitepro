from __future__ import annotations

import contextlib
import gc
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from setiastro.saspro import stacking_suite  # noqa: E402
from setiastro.saspro.stacking_suite import _MMImage  # noqa: E402

H, W = 10, 8


def _open_all(paths):
    return [_MMImage(p) for p in paths]


def _close_all(images):
    for img in images:
        img.close()
    images.clear()
    gc.collect()


class MMImageReadOnlyMapTests(unittest.TestCase):
    """Integration holds every frame open as an _MMImage, so the FITS memmap
    must be read-only: astropy's default copy-on-write map is charged in full
    against the Windows commit limit and thousands of frames exhausted it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = self._tmp.name
        self.rng = np.random.default_rng(7)
        self.images = []

    def tearDown(self):
        _close_all(self.images)
        self._tmp.cleanup()

    def _path(self, name):
        return os.path.join(self.dir, name)

    def _open(self, path):
        img = _MMImage(path)
        self.images.append(img)
        return img

    def _assert_reads(self, path, expected):
        img = self._open(path)
        full = img.read_full()
        self.assertEqual(full.dtype, np.float32)
        np.testing.assert_array_equal(full, expected)
        np.testing.assert_array_equal(img.read_tile(2, 6, 1, 5), expected[2:6, 1:5])

    def test_float32_color_cube(self):
        data = self.rng.random((3, H, W), dtype=np.float32)
        p = self._path("rgb.fit")
        fits.PrimaryHDU(data).writeto(p)
        self._assert_reads(p, np.moveaxis(data, 0, -1))

    def test_float32_mono(self):
        data = self.rng.random((H, W), dtype=np.float32)
        p = self._path("mono.fit")
        fits.PrimaryHDU(data).writeto(p)
        self._assert_reads(p, data)

    def test_uint8_scaled_by_255(self):
        data = self.rng.integers(0, 255, (H, W)).astype(np.uint8)
        p = self._path("u8.fit")
        fits.PrimaryHDU(data).writeto(p)
        self._assert_reads(p, data.astype(np.float32) / 255.0)

    def test_int16_scaled_by_65535(self):
        data = self.rng.integers(0, 30000, (H, W)).astype(np.int16)
        p = self._path("i16.fit")
        fits.PrimaryHDU(data).writeto(p)
        self._assert_reads(p, data.astype(np.float32) / 65535.0)

    def test_uint16_bzero_scaled_by_65535(self):
        data = self.rng.integers(0, 65535, (H, W)).astype(np.uint16)
        p = self._path("u16.fit")
        fits.PrimaryHDU(data).writeto(p)  # stored as int16 + BZERO=32768
        self._assert_reads(p, data.astype(np.float32) / 65535.0)

    def test_bscale_bzero_left_as_astropy_scales_it(self):
        hdu = fits.PrimaryHDU(self.rng.integers(0, 30000, (H, W)).astype(np.int16))
        hdu.header["BSCALE"] = 0.5
        hdu.header["BZERO"] = 10.0
        p = self._path("bscale.fit")
        hdu.writeto(p)
        expected = fits.getdata(p, memmap=False).astype(np.float32)
        self._assert_reads(p, expected)

    def test_tile_compressed(self):
        data = self.rng.random((H, W), dtype=np.float32)
        p = self._path("comp.fz")
        fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data)]).writeto(p)
        expected = fits.getdata(p, ext=1, memmap=False).astype(np.float32)
        self._assert_reads(p, expected)

    def test_fits_memmap_is_read_only(self):
        p = self._path("ro.fit")
        fits.PrimaryHDU(self.rng.random((H, W), dtype=np.float32)).writeto(p)
        img = self._open(p)
        self.assertFalse(img._fits_data.flags.writeable)

    @unittest.skipUnless(sys.platform == "win32", "commit charge is a Windows concept")
    def test_open_frames_do_not_add_commit(self):
        import psutil

        paths = []
        frame = self.rng.random((2000, 1000), dtype=np.float32)  # 8 MB each
        for i in range(20):
            p = self._path(f"f{i:02d}.fit")
            fits.PrimaryHDU(frame).writeto(p)
            paths.append(p)
        total = sum(os.path.getsize(p) for p in paths)

        proc = psutil.Process()
        gc.collect()
        before = proc.memory_info().private
        self.images.extend(_open_all(paths))
        delta = proc.memory_info().private - before
        # A copy-on-write map charges the whole file (delta ~= total).
        self.assertLess(delta, total * 0.25,
                        f"opening {len(paths)} frames added {delta / 2**20:.1f} MiB "
                        f"of commit for {total / 2**20:.1f} MiB of files")


class ForcedRejectMaskTests(unittest.TestCase):
    """The GPU reducer must not expand the forced-reject mask to (F,H,W,C) on
    the host: at thousands of frames that was the allocation that failed."""

    F, H, W, C = 9, 4, 6, 3
    ALGOS = (
        "Weighted Windsorized Sigma Clipping",
        "Kappa-Sigma Clipping",
        "Simple Median (No Rejection)",
    )

    @classmethod
    def setUpClass(cls):
        from setiastro.saspro import torch_rejection

        try:
            torch_rejection._get_torch()  # never installs: allow_install=False
        except Exception as e:
            raise unittest.SkipTest(f"torch unavailable: {e}")
        cls.tr = torch_rejection

    def setUp(self):
        rng = np.random.default_rng(3)
        F, H, W, C = self.F, self.H, self.W, self.C
        self.ts = rng.normal(0.5, 0.05, (F, H, W, C)).astype(np.float32)
        self.ts[2, 1, 1, :] = 5.0  # outlier
        self.ts[4, 0, :, 0] = 0.0  # no-data pixels
        self.wts = rng.uniform(0.5, 1.5, F).astype(np.float32)
        self.m3 = rng.random((F, H, W)) < 0.2
        self.m2 = rng.random((H, W)) < 0.3

    def _reduce(self, algo, mask, **kw):
        out, rej = self.tr._torch_reduce_tile_impl(
            self.ts, self.wts, algo_name=algo, forced_reject_mask_np=mask, **kw)
        return np.asarray(out), np.asarray(rej)

    def _assert_same(self, algo, mask, reference):
        out, rej = self._reduce(algo, mask)
        ref_out, ref_rej = self._reduce(algo, reference)
        np.testing.assert_array_equal(out, ref_out)
        np.testing.assert_array_equal(rej, ref_rej)

    def _full(self, mask):
        return np.broadcast_to(mask, (self.F, self.H, self.W, self.C)).copy()

    def test_mask_shapes_match_materialised_mask(self):
        F, H, W = self.F, self.H, self.W
        cases = {
            "3D": (self.m3, self._full(self.m3[..., None])),
            "2D": (self.m2, self._full(self.m2[None, :, :, None])),
            "(F,H,W,1)": (self.m3[..., None].copy(), self._full(self.m3[..., None])),
            "3D non-contiguous": (np.zeros((F, H + 3, W), bool)[:, :H, :] | self.m3,
                                  self._full(self.m3[..., None])),
        }
        for algo in self.ALGOS:
            for name, (mask, reference) in cases.items():
                with self.subTest(algo=algo, mask=name):
                    self._assert_same(algo, mask, reference)

    def test_all_false_mask_matches_no_mask(self):
        for algo in self.ALGOS:
            with self.subTest(algo=algo):
                self._assert_same(algo, np.zeros((self.F, self.H, self.W), bool), None)

    def test_mask_is_not_expanded_on_host(self):
        import tracemalloc

        F, H, W, C = 400, 64, 64, 3
        ts = np.full((F, H, W, C), 0.5, np.float32)
        mask = np.zeros((F, H, W), bool)
        expanded = F * H * W * C
        tracemalloc.start()
        try:
            self.tr._torch_reduce_tile_impl(
                ts, np.ones(F, np.float32), algo_name="Simple Median (No Rejection)",
                forced_reject_mask_np=mask, reduce_rej_maps=True)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, expanded // 2,
                        f"peak numpy allocation {peak / 2**20:.1f} MiB; the expanded "
                        f"mask alone is {expanded / 2**20:.1f} MiB")


class IntegrationMemoryErrorFallbackTests(unittest.TestCase):
    """When host RAM runs out mid-integration, the group is re-run through the
    Low RAM reader instead of failing the whole run."""

    ALGO = "Weighted Windsorized Sigma Clipping"

    def setUp(self):
        from PyQt6.QtCore import QSettings

        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        rng = np.random.default_rng(0)
        self.paths = []
        for i in range(4):
            p = os.path.join(self._tmp.name, f"f{i}.fit")
            fits.PrimaryHDU(rng.random((3, 40, 24), dtype=np.float32)).writeto(p)
            self.paths.append(p)
        self.weights = {p: 1.0 for p in self.paths}
        self.low_ram_calls = []
        self.opened = []
        self.stub = SimpleNamespace(
            settings=QSettings(os.path.join(self._tmp.name, "s.ini"),
                               QSettings.Format.IniFormat),
            rejection_algorithm=self.ALGO,
            chunk_height=16, chunk_width=24,  # 3 tiles
            sigma_low=2.5, sigma_high=2.5, kappa=2.5, iterations=3,
            trim_fraction=0.1, esd_threshold=3.0, biweight_constant=6.0,
            modz_threshold=3.5,
            _hw_accel_enabled=lambda: False,
            _dtype=lambda: np.float32,
            _cancelled=lambda: False,
            _normalize_master_stem=str,
            _normal_integration_low_ram=self._low_ram,
        )

    def tearDown(self):
        _close_all(self.opened)
        self._tmp.cleanup()

    def _low_ram(self, **kw):
        kw["open_sources_at_call"] = sum(not s.closed for s in self.opened)
        self.low_ram_calls.append(kw)
        return "LOW_RAM_RESULT", {}, None

    def _recording_mmimage(self):
        test = self

        class Recording(_MMImage):
            closed = False

            def __init__(self, path):
                super().__init__(path)
                test.opened.append(self)

            def close(self):
                self.closed = True
                super().close()

        return Recording

    def _integrate(self):
        return stacking_suite.StackingSuiteDialog.normal_integration_with_rejection(
            self.stub, "g", self.paths, self.weights, status_cb=lambda *_: None)

    @staticmethod
    def _fail_on_second_call(exc):
        calls = []
        real = stacking_suite.windsorized_sigma_clip_weighted_np

        def reducer(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise exc
            return real(*args, **kwargs)

        return reducer

    def _patch(self, name, value):
        p = mock.patch.object(stacking_suite, name, value)
        p.start()
        self.addCleanup(p.stop)

    def _enable_gpu(self, torch_reducer):
        self.stub._hw_accel_enabled = lambda: True
        self._patch("_torch_ok", lambda: True)
        self._patch("_gpu_algo_supported", lambda algo: True)
        self._patch("_rejection_gpu_ok", lambda status_cb=None: True)
        self._patch("_safe_torch_inference_ctx", contextlib.nullcontext)
        self._patch("_torch_reduce_tile", torch_reducer)

    def _assert_low_ram_rerun(self, result):
        self.assertEqual(result, ("LOW_RAM_RESULT", {}, None))
        self.assertEqual(len(self.low_ram_calls), 1)
        kw = self.low_ram_calls[0]
        self.assertEqual(kw["group_key"], "g")
        self.assertEqual(kw["file_list"], self.paths)
        self.assertEqual(kw["frame_weights"], self.weights)
        self.assertIsNone(kw["algo_override"])
        self.assertFalse(kw["collect_per_file_rejections"])

    def test_cpu_memory_error_reruns_with_low_ram_reader(self):
        self._patch("windsorized_sigma_clip_weighted_np",
                    self._fail_on_second_call(MemoryError("simulated")))
        self._assert_low_ram_rerun(self._integrate())

    def test_gpu_memory_error_skips_cpu_fallback(self):
        def torch_oom(*args, **kwargs):
            raise MemoryError("simulated")

        def cpu_reducer(*args, **kwargs):
            raise AssertionError("CPU reduction must not run after a host MemoryError")

        self._enable_gpu(torch_oom)
        self._patch("windsorized_sigma_clip_weighted_np", cpu_reducer)
        self._assert_low_ram_rerun(self._integrate())

    def test_other_gpu_error_still_falls_back_to_cpu(self):
        def torch_fails(*args, **kwargs):
            raise RuntimeError("CUDA out of memory (simulated)")

        self._enable_gpu(torch_fails)
        img, _, _ = self._integrate()
        self.assertEqual(self.low_ram_calls, [])
        self.assertEqual(img.shape, (40, 24, 3))
        self.assertTrue(np.isfinite(img).all())

    def test_sources_closed_before_low_ram_rerun(self):
        self._patch("_MMImage", self._recording_mmimage())
        self._patch("windsorized_sigma_clip_weighted_np",
                    self._fail_on_second_call(MemoryError("simulated")))
        self._integrate()
        self.assertEqual(len(self.opened), len(self.paths))
        self.assertEqual(self.low_ram_calls[0]["open_sources_at_call"], 0)


if __name__ == "__main__":
    unittest.main()
