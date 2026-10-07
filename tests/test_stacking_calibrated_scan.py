from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from setiastro.saspro.stacking_calibrated_scan import (  # noqa: E402
    CalibratedMeta,
    CalibratedMetaCache,
    ScanEntry,
    list_folder_entries,
    read_calibrated_metas,
    read_fits_headers,
    start_background_read,
    stat_entry,
)


def _write_fits(path, *, compressed=False, **cards):
    hdr = fits.Header()
    for key, value in cards.items():
        hdr[key] = value
    data = np.zeros((8, 10), dtype=np.float32)
    if compressed:
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data=data, header=hdr)])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(data=data, header=hdr)])
    hdul.writeto(path, overwrite=True)


class _TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="saspro_scan_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)


class ReadFitsHeadersTests(_TempDirTest):
    def test_plain_image_uses_primary_for_both(self):
        fp = self.path("a.fit")
        _write_fits(fp, FILTER="Ha", OBJECT="M31")
        primary, science = read_fits_headers(fp)
        self.assertEqual(primary["FILTER"], "Ha")
        self.assertEqual(science["OBJECT"], "M31")
        self.assertEqual(list(primary.keys()), list(fits.getheader(fp, ext=0).keys()))

    def test_compressed_image_science_header_is_extension(self):
        fp = self.path("a.fz")
        _write_fits(fp, compressed=True, FILTER="OIII", OBJECT="NGC7000")
        primary, science = read_fits_headers(fp)
        self.assertNotIn("FILTER", primary)  # same as fits.getheader(fp, ext=0)
        self.assertEqual(science["OBJECT"], "NGC7000")

    def test_opens_file_once(self):
        fp = self.path("a.fit")
        _write_fits(fp, FILTER="L")
        with mock.patch("setiastro.saspro.stacking_calibrated_scan.fits.open",
                        wraps=fits.open) as opened:
            read_fits_headers(fp)
        self.assertEqual(opened.call_count, 1)


class ListFolderEntriesTests(_TempDirTest):
    def test_filters_extensions_and_directories_in_listing_order(self):
        for name in ("b.fit", "a.FITS", "c.xisf", "notes.txt", "d.tif"):
            with open(self.path(name), "wb") as f:
                f.write(b"x")
        os.mkdir(self.path("subdir.fit"))
        entries = list_folder_entries(self.tmp)
        expected = [
            self.path(n) for n in os.listdir(self.tmp)
            if os.path.splitext(n)[1].lower() in (".fit", ".fits", ".xisf", ".tif")
            and os.path.isfile(self.path(n))
        ]
        self.assertEqual([e.path for e in entries], expected)
        self.assertTrue(all(e.mtime_ns and e.size == 1 for e in entries))

    def test_missing_folder_is_empty(self):
        self.assertEqual(list_folder_entries(self.path("nope")), [])

    def test_stat_entry_of_missing_file_has_no_stat(self):
        missing = self.path("gone.fit")
        self.assertEqual(stat_entry(missing), ScanEntry(missing, None, None))


class CalibratedMetaCacheTests(unittest.TestCase):
    META = CalibratedMeta("L", 300.0, "10x8", 100.0, "M31")

    def test_hit_requires_matching_mtime_and_size(self):
        cache = CalibratedMetaCache()
        entry = ScanEntry("x.fit", 5, 10)
        self.assertIsNone(cache.lookup(entry))
        cache.store(entry, self.META)
        self.assertEqual(cache.lookup(entry), self.META)
        self.assertIsNone(cache.lookup(entry._replace(mtime_ns=6)))
        self.assertIsNone(cache.lookup(entry._replace(size=11)))

    def test_entries_without_stat_are_never_cached(self):
        cache = CalibratedMetaCache()
        entry = ScanEntry("x.fit", None, None)
        cache.store(entry, self.META)
        self.assertIsNone(cache.lookup(entry))


class ReadCalibratedMetasTests(unittest.TestCase):
    def setUp(self):
        self.entries = [ScanEntry(f"f{i}.fit", 1, 1) for i in range(50)]
        self.calls = []
        self.lock = threading.Lock()

    def probe(self, path):
        with self.lock:
            self.calls.append(path)
        return CalibratedMeta("L", 1.0, "1x1", None, path)

    def test_probes_misses_once_then_uses_cache(self):
        cache = CalibratedMetaCache()
        first = read_calibrated_metas(self.entries, cache, self.probe, workers=4)
        self.assertEqual(sorted(self.calls), sorted(e.path for e in self.entries))
        second = read_calibrated_metas(self.entries, cache, self.probe, workers=4)
        self.assertEqual(len(self.calls), len(self.entries))
        self.assertEqual(first, second)
        self.assertEqual(first["f7.fit"].obj, "f7.fit")

    def test_threaded_and_inline_reads_agree(self):
        def read(workers):
            return read_calibrated_metas(
                self.entries, CalibratedMetaCache(), self.probe, workers=workers)

        self.assertEqual(read(1), read(4))

    def test_failed_probe_is_left_out(self):
        def probe(path):
            if path == "f3.fit":
                raise OSError("unreadable")
            return self.probe(path)

        metas = read_calibrated_metas(self.entries, CalibratedMetaCache(), probe, workers=4)
        self.assertNotIn("f3.fit", metas)
        self.assertEqual(len(metas), len(self.entries) - 1)

    def test_stop_returns_none_without_reading_everything(self):
        stop = threading.Event()
        stop.set()
        metas = read_calibrated_metas(self.entries, CalibratedMetaCache(), self.probe, stop=stop)
        self.assertIsNone(metas)
        self.assertEqual(self.calls, [])

    def test_progress_ends_at_total(self):
        seen = []
        read_calibrated_metas(self.entries, CalibratedMetaCache(), self.probe, workers=4,
                              on_progress=lambda done, total: seen.append((done, total)))
        self.assertIn((50, 50), seen)
        self.assertLessEqual(len(seen), 21)


class StartBackgroundReadTests(unittest.TestCase):
    def test_fills_cache_then_calls_on_done(self):
        entries = [ScanEntry(f"f{i}.fit", 1, 1) for i in range(20)]
        cache = CalibratedMetaCache()
        done = threading.Event()
        thread = start_background_read(
            entries, cache, lambda p: CalibratedMeta("L", 1.0, "1x1", None, ""),
            stop=threading.Event(), on_done=done.set,
        )
        self.assertTrue(done.wait(10))
        thread.join(10)
        self.assertTrue(all(cache.lookup(e) is not None for e in entries))

    def test_stopped_read_skips_on_done(self):
        stop = threading.Event()
        stop.set()
        done = threading.Event()
        thread = start_background_read(
            [ScanEntry("f.fit", 1, 1)], CalibratedMetaCache(),
            lambda p: CalibratedMeta("L", 1.0, "1x1", None, ""),
            stop=stop, on_done=done.set,
        )
        thread.join(10)
        self.assertFalse(done.is_set())


class StackingSuiteIntegrationTreeTests(unittest.TestCase):
    """Open the real dialog over a small Calibrated/ folder."""

    N_FRAMES = 120

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtCore import QCoreApplication, QSettings
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])
        cls.tmp = tempfile.mkdtemp(prefix="saspro_suite_test_")
        cal = os.path.join(cls.tmp, "Calibrated")
        os.makedirs(cal)
        for i in range(cls.N_FRAMES):
            _write_fits(os.path.join(cal, f"frame_{i:04d}_c.fit"),
                        FILTER="L" if i % 2 else "R", EXPTIME=300.0, GAIN=100,
                        OBJECT="M31")

        # Keep the dialog's QSettings away from the user's real settings.
        cls._saved_names = (QCoreApplication.organizationName(),
                            QCoreApplication.applicationName())
        QCoreApplication.setOrganizationName("SASproTests")
        QCoreApplication.setApplicationName("StackingCalibratedScanTests")
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                          os.path.join(cls.tmp, "settings"))
        settings = QSettings()
        settings.clear()
        settings.setValue("stacking/dir", cls.tmp)
        settings.sync()

        import setiastro.saspro.stacking_suite as stacking_suite
        cls.ss = stacking_suite
        cls._dialog_settings = []

    @classmethod
    def tearDownClass(cls):
        from PyQt6.QtCore import QCoreApplication, QSettings

        # Each dialog's QSettings outlives the test; one with unsaved changes
        # would recreate the settings folder when Python finally destroys it.
        for settings in cls._dialog_settings:
            settings.sync()
        QCoreApplication.setOrganizationName(cls._saved_names[0])
        QCoreApplication.setApplicationName(cls._saved_names[1])
        QSettings.setDefaultFormat(QSettings.Format.NativeFormat)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _open(self):
        from PyQt6.QtCore import Qt

        dlg = self.ss.StackingSuiteDialog()
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._dialog_settings.append(dlg.settings)
        self.addCleanup(dlg.close)
        return dlg

    def _leaf_count(self, dlg):
        tree = dlg.reg_tree
        return sum(tree.topLevelItem(i).childCount() for i in range(tree.topLevelItemCount()))

    def _wait_loaded(self, dlg, timeout=30.0):
        deadline = time.monotonic() + timeout
        while dlg._cal_scan_stop is not None and time.monotonic() < deadline:
            self._app.processEvents()
            time.sleep(0.01)
        self._app.processEvents()
        self.assertIsNone(dlg._cal_scan_stop, "background header scan did not finish")

    def test_open_loads_tree_in_background_with_one_refresh(self):
        calls = []
        original = self.ss.StackingSuiteDialog._refresh_reg_tree_summaries

        def counted(dlg_self):
            calls.append(1)
            return original(dlg_self)

        dialog_cls = self.ss.StackingSuiteDialog
        with mock.patch.object(dialog_cls, "_refresh_reg_tree_summaries", counted):
            dlg = self._open()
            # The constructor only starts the scan; nothing is built yet.
            self.assertIsNotNone(dlg._cal_scan_stop)
            self.assertEqual(self._leaf_count(dlg), 0)
            self._wait_loaded(dlg)

        self.assertEqual(self._leaf_count(dlg), self.N_FRAMES)
        self.assertEqual(dlg.reg_tree.topLevelItemCount(), 2)
        self.assertTrue(all(dlg.reg_tree.topLevelItem(i).isExpanded()
                            for i in range(dlg.reg_tree.topLevelItemCount())))
        self.assertEqual(set(dlg.frame_set_of.values()), {"M31"})
        self.assertLessEqual(len(calls), 3)

    def test_repopulate_rereads_no_unchanged_headers(self):
        dlg = self._open()
        self._wait_loaded(dlg)
        with mock.patch.object(self.ss, "read_fits_headers",
                               wraps=self.ss.read_fits_headers) as reads:
            dlg.gain_tolerance_spin.setValue(dlg.gain_tolerance_spin.value() + 5)
        self.assertEqual(reads.call_count, 0)
        self.assertEqual(self._leaf_count(dlg), self.N_FRAMES)

    def test_tree_consumer_finishes_pending_load(self):
        dlg = self._open()
        self.assertIsNotNone(dlg._cal_scan_stop)
        dlg.extract_light_files_from_tree()
        self.assertIsNone(dlg._cal_scan_stop)
        self.assertEqual(sum(len(v) for v in dlg.light_files.values()), self.N_FRAMES)


if __name__ == "__main__":
    unittest.main()
