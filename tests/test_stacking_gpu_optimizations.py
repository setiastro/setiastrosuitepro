from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from setiastro.saspro import torch_rejection as tr
from setiastro.saspro.stacking_gpu_pipeline import TileSession
from setiastro.saspro.torch_drizzle import DrizzleAccumulator


@unittest.skipIf(torch is None, "PyTorch not installed")
class GPUOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.patch_torch = mock.patch.object(tr, "_TORCH", torch)
        self.patch_dev = mock.patch.object(tr, "_DEVICE", torch.device("cpu"))
        self.patch_torch.start()
        self.patch_dev.start()
        self.addCleanup(self.patch_torch.stop)
        self.addCleanup(self.patch_dev.stop)
        rng = np.random.default_rng(10)
        self.stack = rng.normal(0.5, 0.04, (9, 5, 7, 3)).astype(np.float32)
        self.stack[0, 0, 0] = 4.0
        self.stack[:, 1, 2] = 0.0
        self.weights = np.linspace(0.5, 1.5, 9, dtype=np.float32)

    def test_device_sort_matches_native_nan_reducers(self):
        x = torch.tensor([[1., 4., float('nan')], [2., float('nan'), float('nan')],
                          [3., 8., float('nan')], [9., 10., float('nan')]])
        torch.testing.assert_close(tr._sorted_nanquantile(torch, x, .5, 0, median=True),
                                   torch.nanmedian(x, dim=0).values, equal_nan=True)
        for q in (0.0, 0.25, 0.4, 0.5, 1.0):
            torch.testing.assert_close(tr._sorted_nanquantile(torch, x, q, 0),
                                       torch.nanquantile(x, q, dim=0), equal_nan=True)
        expected_std = np.nanstd(x.numpy(), axis=0)
        np.testing.assert_allclose(tr._nanstd_sums(torch, x, 0).numpy(), expected_std,
                                   rtol=1e-6, equal_nan=True)

    def test_directml_helpers_never_call_cpu_roundtrip(self):
        x = torch.tensor([[1., 3.], [5., float('nan')]])
        with mock.patch.object(tr, "_is_directml_privateuse", return_value=True), \
             mock.patch.object(torch, "nanmedian", side_effect=AssertionError("native fallback")), \
             mock.patch.object(torch, "nanquantile", side_effect=AssertionError("native fallback")):
            torch.testing.assert_close(tr._nanmedian(torch, x, 0), torch.tensor([1., 3.]))
            torch.testing.assert_close(tr._nanquantile(torch, x, .25, 0), torch.tensor([2., 3.]))
            torch.testing.assert_close(tr._nanstd(torch, x, 0), torch.tensor([2., 0.]))

    def test_positive_weight_average_does_not_compute_median(self):
        with mock.patch.object(tr, "_nanmedian", side_effect=AssertionError("unexpected median")):
            out, _ = tr.torch_reduce_tile(self.stack, self.weights,
                algo_name="Simple Average (No Rejection)")
        expected = np.average(self.stack, weights=self.weights, axis=0)
        expected[1, 2] = np.nan
        np.testing.assert_allclose(out, expected, rtol=2e-6, equal_nan=True)

    def test_winsorized_center_scale_matches_recomputed_medians(self):
        for frames in (4, 9, 16):
            rng = np.random.default_rng(frames)
            data = rng.normal(.5, .1, (frames, 3, 4, 1)).astype(np.float32)
            data[0] = 4.
            data[:, 0, 0] = np.nan
            x = torch.tensor(data)
            valid = torch.isfinite(x)
            for iterations in (1, 3, 4):
                med = tr._nanmedian(torch, x, 0)
                sigma = tr._mad_sigma(torch, x, med, 0)
                count = valid.sum(dim=0).clamp_min(1)
                for _ in range(iterations):
                    clamped = torch.minimum(torch.maximum(x, (med - 1.5 * sigma)[None]),
                                            (med + 1.5 * sigma)[None])
                    med = tr._nanmedian(torch, clamped, 0)
                    mean = tr._masked_sum(torch, clamped, valid, 0) / count
                    var = tr._masked_sum(torch, (clamped - mean[None])**2, valid, 0) / count
                    sigma = (1.134 * var.clamp_min(0).sqrt()).clamp_min(1e-10)
                actual_med, actual_sigma = tr._winsorized_center_scale(torch, x, valid, iterations)
                torch.testing.assert_close(actual_med, med, equal_nan=True, rtol=0, atol=0)
                torch.testing.assert_close(actual_sigma, sigma, equal_nan=True, rtol=0, atol=0)

    def test_nonpositive_weights_keep_median_fallback(self):
        weights = np.zeros_like(self.weights)
        out, _ = tr.torch_reduce_tile(self.stack, weights,
            algo_name="Simple Average (No Rejection)")
        reference = torch.nanmedian(torch.tensor(self.stack).masked_fill(
            torch.tensor(self.stack) == 0, float('nan')), dim=0).values.numpy()
        np.testing.assert_allclose(out, reference, equal_nan=True)

    def test_prepared_tiles_and_compact_maps_match_all_algorithms(self):
        for algo in sorted(tr._SUPPORTED):
            with self.subTest(algo=algo):
                original, rejection = tr.torch_reduce_tile(self.stack, self.weights, algo_name=algo)
                out, collapsed = tr.torch_reduce_tile(self.stack, self.weights, algo_name=algo,
                    device_tile=torch.from_numpy(self.stack),
                    device_weights=torch.from_numpy(self.weights).view(-1, 1, 1, 1),
                    collapse_rej_channels=True)
                np.testing.assert_array_equal(out, original)
                np.testing.assert_array_equal(collapsed, rejection.any(axis=-1))
                out, (any_map, count) = tr.torch_reduce_tile(self.stack, self.weights,
                    algo_name=algo, reduce_rej_maps=True)
                np.testing.assert_array_equal(out, original)
                np.testing.assert_array_equal(any_map, collapsed.any(axis=0))
                np.testing.assert_array_equal(count, collapsed.sum(axis=0))

    def test_session_results_and_edge_tile(self):
        session = TileSession(torch, torch.device('cpu'), self.weights, self.stack.shape, lambda _: None)
        try:
            for array in (self.stack, self.stack[:, :2]):
                actual = session.reduce(tr.torch_reduce_tile, array, self.weights,
                    session.prepare(array), algo_name="Weighted Windsorized Sigma Clipping",
                    reduce_rej_maps=True)
                expected = tr.torch_reduce_tile(array, self.weights,
                    algo_name="Weighted Windsorized Sigma Clipping", reduce_rej_maps=True)
                np.testing.assert_array_equal(actual[0], expected[0])
                for a, b in zip(actual[1], expected[1]):
                    np.testing.assert_array_equal(a, b)
        finally:
            session.close()

    def test_dev_band_tuner_accepts_optimized_reducer(self):
        from setiastro.saspro.stacking_worker_tune import _time_band

        seconds, throughput, error = _time_band(
            4, 8, 5, 3, 'Weighted Windsorized Sigma Clipping', reps=1)
        self.assertIsNone(error)
        self.assertGreater(seconds, 0)
        self.assertGreater(throughput, 0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_lookahead_slot_reuse(self):
        with mock.patch.object(tr, '_DEVICE', torch.device('cuda')):
            session = TileSession(torch, torch.device('cuda'), self.weights, self.stack.shape, lambda _: None)
            try:
                self.assertTrue(session.pipelined)
                arrays = [self.stack, self.stack * 1.1, self.stack[:, :2]]
                prepared = session.prepare(arrays[0], 0)
                for i, array in enumerate(arrays):
                    next_tile = session.prepare(arrays[i + 1], (i + 1) % 2) if i + 1 < len(arrays) else None
                    actual = session.reduce(tr.torch_reduce_tile, array, self.weights, prepared,
                        algo_name="Weighted Windsorized Sigma Clipping", reduce_rej_maps=True)
                    expected = tr.torch_reduce_tile(array, self.weights,
                        algo_name="Weighted Windsorized Sigma Clipping", reduce_rej_maps=True)
                    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, equal_nan=True)
                    for a, b in zip(actual[1], expected[1]):
                        np.testing.assert_array_equal(a, b)
                    prepared = next_tile
            finally:
                session.close()


@unittest.skipIf(torch is None, "PyTorch not installed")
class DrizzleParityTests(unittest.TestCase):
    def test_projective_kernels_and_sparse_channel_coverage(self):
        from setiastro.saspro.legacy.numba_utils import (
            drizzle_deposit_kernel_mono_proj, drizzle_deposit_color_kernel_proj)
        rng = np.random.default_rng(42)
        transform = np.array([[.99, .03, -.35], [-.02, 1.01, .21],
                              [.0004, -.0003, 1.]], np.float32)
        for color in (False, True):
            image = rng.random((5, 7, 3) if color else (5, 7), dtype=np.float32)
            image[::2, ::2] = 0
            if color:
                image[..., 1][::2] = 0
            shape = (10, 14, 3) if color else (10, 14)
            fn = drizzle_deposit_color_kernel_proj if color else drizzle_deposit_kernel_mono_proj
            for dtype in (np.float32, np.float64):
                for kernel in (0, 1, 2):
                    for pixfrac in (0.0, .3, .9, 1.5):
                        with self.subTest(color=color, kernel=kernel, dtype=dtype, pixfrac=pixfrac):
                            acc = DrizzleAccumulator(torch, torch.device('cpu'), shape, dtype,
                                                      2., pixfrac, kernel, .45)
                            num, cov = np.zeros(shape, dtype), np.zeros(shape, dtype)
                            for weight in (0.7, 1.3):
                                acc.add(image, transform, weight)
                                num, cov = fn(image, transform, num, cov, 2., pixfrac, weight, kernel, .45)
                            np.testing.assert_allclose(acc.numerator.numpy(), num, atol=3e-6, rtol=3e-6)
                            np.testing.assert_allclose(acc.coverage.numpy(), cov, atol=3e-6, rtol=3e-6)
                            expected = np.divide(num, cov, out=np.zeros_like(num), where=cov >= 1e-8).astype(np.float32)
                            np.testing.assert_allclose(acc.finish(), expected, atol=3e-6, rtol=3e-6)
                            acc.close()


@unittest.skipUnless(torch is not None and os.getenv('SASPRO_TEST_DIRECTML') == '1',
                     "Set SASPRO_TEST_DIRECTML=1 to test actual DirectML hardware")
class DirectMLParityTests(unittest.TestCase):
    def test_winsorized_device_results_and_session_match_cpu(self):
        import torch_directml
        from setiastro.saspro.stacking_suite import _rejection_gpu_ok

        rng = np.random.default_rng(9)
        stack = rng.normal(.5, .03, (17, 8, 16, 3)).astype(np.float32)
        stack[:, 1, 2] = 0.
        stack[0, 3, 4] = 4.
        weights = np.ones(17, dtype=np.float32)
        with mock.patch.object(tr, '_TORCH', torch), mock.patch.object(tr, '_DEVICE', torch.device('cpu')):
            expected = tr.torch_reduce_tile(stack, weights, algo_name='Weighted Windsorized Sigma Clipping',
                                             reduce_rej_maps=True)
        with mock.patch.object(tr, '_TORCH', torch), mock.patch.object(tr, '_DEVICE', torch_directml.device()):
            from setiastro.saspro import stacking_suite
            with mock.patch.object(stacking_suite, '_REJ_GPU_VERIFIED', None):
                self.assertTrue(_rejection_gpu_ok())
            session = TileSession.create(weights, stack.shape, lambda _: None)
            try:
                with tr._safe_inference_ctx(torch):
                    actual = session.reduce(tr.torch_reduce_tile, stack, weights, session.prepare(stack),
                        algo_name='Weighted Windsorized Sigma Clipping', reduce_rej_maps=True)
                np.testing.assert_allclose(actual[0], expected[0], rtol=2e-6, atol=2e-6)
                for a, b in zip(actual[1], expected[1]):
                    np.testing.assert_array_equal(a, b)
                self.assertIsNone(DrizzleAccumulator.create((8, 16, 3), np.float32,
                    2., .9, 0, .45, lambda _: None))
            finally:
                session.close()


if __name__ == '__main__':
    unittest.main()
