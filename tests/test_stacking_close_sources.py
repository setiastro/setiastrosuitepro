from __future__ import annotations

import gc
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from setiastro.saspro import stacking_suite  # noqa: E402
from setiastro.saspro.stacking_suite import _MMImage  # noqa: E402


def _write_frames(folder, n):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = os.path.join(folder, f"f{i}.fit")
        fits.PrimaryHDU(rng.random((3, 40, 24), dtype=np.float32)).writeto(p)
        paths.append(p)
    return paths


class _CountCollections:
    """Count gc.collect() calls (all threads) while still collecting."""

    def __enter__(self):
        self.calls = 0
        real = gc.collect

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        self._patch = mock.patch.object(gc, "collect", counting)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


class CloseCollectionTests(unittest.TestCase):
    """_MMImage.close() runs a full gc.collect(). Each one walks every tracked
    object in the process, so closing thousands of sources one collection at
    a time held the GIL for ~20 min after a 3,318-frame integration."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = _write_frames(self._tmp.name, 1)[0]

    def tearDown(self):
        gc.collect()
        self._tmp.cleanup()

    def test_closing_a_closed_source_does_not_collect(self):
        # __del__ calls close() again on every source that was closed
        # explicitly; there is nothing left to release then.
        img = _MMImage(self.path)
        img.read_tile(0, 4, 0, 4)
        img.close()
        with _CountCollections() as count:
            img.close()
            del img
        self.assertEqual(count.calls, 0)

    def test_close_still_collects_by_default(self):
        img = _MMImage(self.path)
        with _CountCollections() as count:
            img.close()
        self.assertEqual(count.calls, 1)


class IntegrationCloseTests(unittest.TestCase):
    """normal_integration_with_rejection closes its sources on a background
    thread when it finishes."""

    ALGO = "Weighted Windsorized Sigma Clipping"

    def setUp(self):
        from PyQt6.QtCore import QSettings

        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.stub = SimpleNamespace(
            settings=QSettings(os.path.join(self._tmp.name, "s.ini"),
                               QSettings.Format.IniFormat),
            rejection_algorithm=self.ALGO,
            chunk_height=16, chunk_width=24,
            sigma_low=2.5, sigma_high=2.5, kappa=2.5, iterations=3,
            trim_fraction=0.1, esd_threshold=3.0, biweight_constant=6.0,
            modz_threshold=3.5,
            _hw_accel_enabled=lambda: False,
            _dtype=lambda: np.float32,
            _cancelled=lambda: False,
            _normalize_master_stem=str,
        )

    def tearDown(self):
        gc.collect()
        self._tmp.cleanup()

    def _integrate(self, paths):
        """Integrate, then wait for the close thread to finish."""
        before = set(threading.enumerate())
        img, _, _ = stacking_suite.StackingSuiteDialog.normal_integration_with_rejection(
            self.stub, "g", paths, {p: 1.0 for p in paths},
            status_cb=lambda *_: None)
        for t in set(threading.enumerate()) - before:
            t.join(timeout=30)
            self.assertFalse(t.is_alive())
        return img

    def _collections_for(self, n):
        folder = tempfile.mkdtemp(dir=self._tmp.name)
        paths = _write_frames(folder, n)
        with _CountCollections() as count:
            img = self._integrate(paths)
        self.assertEqual(img.shape, (40, 24, 3))
        return count.calls

    def test_collections_do_not_grow_with_frame_count(self):
        few = self._collections_for(4)
        many = self._collections_for(40)
        self.assertEqual(many, few)

    def test_frames_released_after_close_thread(self):
        # Keeps the guarantee the per-close collection was added for: once
        # the sources are closed, Windows no longer holds the files mapped.
        folder = tempfile.mkdtemp(dir=self._tmp.name)
        paths = _write_frames(folder, 6)
        self._integrate(paths)
        for p in paths:
            os.replace(p, p + ".moved")


if __name__ == "__main__":
    unittest.main()
