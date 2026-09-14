from __future__ import annotations

import unittest

import numpy as np

from setiastro.saspro.cosmicclarity_engines.satellite_engine import (
    _resize_tile_for_detect,
    trail_mask_requires_rewrite,
)


class ResizeTileForDetectTests(unittest.TestCase):
    def test_256_tile_is_not_resampled(self):
        tile = np.linspace(0.0, 1.0, 256 * 256 * 3, dtype=np.float32).reshape(256, 256, 3)
        out = _resize_tile_for_detect(tile)
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, tile)
        self.assertTrue(out is tile or np.shares_memory(out, tile))

    def test_smaller_tile_is_resized_to_256(self):
        tile = np.random.rand(80, 120, 3).astype(np.float32)
        out = _resize_tile_for_detect(tile)
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out.dtype, np.float32)


class SatelliteRewriteDecisionTests(unittest.TestCase):
    def test_empty_mask_does_not_need_rewrite(self):
        mask = np.zeros((32, 32), dtype=bool)
        self.assertFalse(trail_mask_requires_rewrite(mask))

    def test_none_mask_does_not_need_rewrite(self):
        self.assertFalse(trail_mask_requires_rewrite(None))

    def test_any_trail_pixel_needs_rewrite(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[3, 4] = True
        self.assertTrue(trail_mask_requires_rewrite(mask))


if __name__ == "__main__":
    unittest.main()
