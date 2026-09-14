from __future__ import annotations

import unittest

from setiastro.saspro.stacking_gpu_mem import (
    gpu_flat_cache_key,
    is_cuda_oom,
    is_master_flat_key,
    reap_completed,
    session_from_manual_keyword,
    submit_bounded,
)


class SessionFromManualKeywordTests(unittest.TestCase):
    def test_default_and_empty(self):
        path = "/data/raws/panel-01/NGC 7822 Panel 1_B_0085.fits"
        self.assertEqual(session_from_manual_keyword(path, "Default"), "Default")
        self.assertEqual(session_from_manual_keyword(path, ""), "Default")
        self.assertEqual(session_from_manual_keyword(path, "  "), "Default")

    def test_numbered_folder_not_filename(self):
        path = (
            "/mnt/SSD2.5/Astro/NGC7822/raws/panel-01/"
            "NGC 7822 Panel 1_B_2026-03-17_05-34-55_-10.30_60.00s_"
            "gain-0_offset-50_0085.fits"
        )
        self.assertEqual(session_from_manual_keyword(path, "Panel"), "panel-01")

    def test_does_not_use_filename_when_folder_is_ambiguous(self):
        path = (
            "/mnt/SSD2.5/Astro/NGC7822/raws/panel1.satellited/"
            "NGC 7822 Panel 1_B_2026-03-17_05-34-55_-10.30_60.00s_"
            "gain-0_offset-50_0085_satellited.fits"
        )
        self.assertEqual(
            session_from_manual_keyword(path, "Panel"),
            "panel1.satellited",
        )

    def test_night_folder_pattern(self):
        path = "/data/NIGHT_2/lights/frame_001.fits"
        self.assertEqual(session_from_manual_keyword(path, "NIGHT"), "NIGHT_2")

    def test_missing_falls_back_to_keyword(self):
        path = "/data/lights/frame_001.fits"
        self.assertEqual(session_from_manual_keyword(path, "Panel"), "Panel")


class MasterFlatKeyMatchTests(unittest.TestCase):
    """Filter 'G' must not match gain tags like [G0] / [G100]."""

    SIZE = "9576x6388"

    def test_g_matches_only_the_green_flat_key(self):
        self.assertTrue(
            is_master_flat_key(
                "G (9576x6388) [Default] [G0]",
                filter_name="G",
                image_size=self.SIZE,
            )
        )
        self.assertTrue(
            is_master_flat_key(
                "G (9576x6388)",
                filter_name="G",
                image_size=self.SIZE,
            )
        )

    def test_g_does_not_match_gain_suffix_on_other_flats(self):
        for key in (
            "B (9576x6388) [Default] [G0]",
            "R (9576x6388) [Default] [G0]",
            "H (9576x6388) [Default] [G100]",
            "O (9576x6388) [Default] [G100]",
            "S (9576x6388) [Default] [G100]",
        ):
            self.assertFalse(
                is_master_flat_key(key, filter_name="G", image_size=self.SIZE),
                msg=key,
            )

    def test_g_does_not_match_master_dark_gain_tag(self):
        self.assertFalse(
            is_master_flat_key(
                "60s (9576x6388) [-10.0C] [G0]",
                filter_name="G",
                image_size=self.SIZE,
            )
        )

    def test_other_filters_still_match_their_own_flat(self):
        self.assertTrue(
            is_master_flat_key(
                "B (9576x6388) [Default] [G0]",
                filter_name="B",
                image_size=self.SIZE,
            )
        )
        self.assertTrue(
            is_master_flat_key(
                "H (9576x6388) [Default] [G100]",
                filter_name="H",
                image_size=self.SIZE,
            )
        )


class GpuFlatCacheKeyTests(unittest.TestCase):
    def test_reuses_path_when_not_interactive(self):
        gk_a = ("S", "600s", "session-a", "dark.fit", "flat_S.fit")
        gk_b = ("S", "600s", "session-b", "dark.fit", "flat_S.fit")
        self.assertEqual(
            gpu_flat_cache_key(gk_a, "flat_S.fit", interactive=False),
            ("path", "flat_S.fit"),
        )
        self.assertEqual(
            gpu_flat_cache_key(gk_a, "flat_S.fit", interactive=False),
            gpu_flat_cache_key(gk_b, "flat_S.fit", interactive=False),
        )

    def test_is_per_group_when_interactive(self):
        gk_a = ("S", "600s", "session-a", "dark.fit", "flat_S.fit")
        gk_b = ("S", "600s", "session-b", "dark.fit", "flat_S.fit")
        self.assertNotEqual(
            gpu_flat_cache_key(gk_a, "flat_S.fit", interactive=True),
            gpu_flat_cache_key(gk_b, "flat_S.fit", interactive=True),
        )

    def test_none_path(self):
        self.assertIsNone(gpu_flat_cache_key(("S",), None, interactive=False))
        self.assertIsNone(gpu_flat_cache_key(("S",), "", interactive=False))


class CudaOomTests(unittest.TestCase):
    def test_detects_pytorch_oom(self):
        class OutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(is_cuda_oom(OutOfMemoryError("CUDA out of memory.")))
        self.assertTrue(
            is_cuda_oom(
                RuntimeError(
                    "CUDA out of memory. Tried to allocate 234.00 MiB. GPU 0 "
                    "has a total capacity of 23.56 GiB"
                )
            )
        )
        self.assertFalse(is_cuda_oom(ValueError("bad tile")))
        self.assertFalse(is_cuda_oom(RuntimeError("file not found")))


class BoundedSubmitTests(unittest.TestCase):
    def test_never_queues_more_futures_than_the_limit(self):
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        in_flight = 0
        peak = 0
        lock = threading.Lock()

        def work(_):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            time.sleep(0.02)
            with lock:
                in_flight -= 1

        pending = []
        errors = []
        # More workers than the limit: without submit_bounded the pool would
        # run 8 at once and queue the rest while holding every argument.
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i in range(20):
                pending = submit_bounded(
                    ex, pending, work, i, max_pending=3,
                    on_error=errors.append,
                )
            for fut in pending:
                fut.result()
        self.assertEqual(errors, [])
        self.assertLessEqual(peak, 3)

    def test_reap_completed_surfaces_errors_and_drops_done_futures(self):
        from concurrent.futures import ThreadPoolExecutor, wait

        def boom(_):
            raise RuntimeError("disk full")

        def ok(_):
            return 1

        errors = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            pending = [ex.submit(ok, 0), ex.submit(boom, 0)]
            wait(pending)
            still = reap_completed(pending, on_error=errors.append)
        self.assertEqual(still, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("disk full", str(errors[0]))


if __name__ == "__main__":
    unittest.main()
