from __future__ import annotations

import gc
import os
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
