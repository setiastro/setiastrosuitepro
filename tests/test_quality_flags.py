"""Diagnosis and verdict for Blink Comparator pre-flag."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from setiastro.saspro.stacking_suite import quality_flag_reasons  # noqa: E402


def _frame(**kw):
    base = {
        "ok": True,
        "sky": 1000.0,
        "stars": 200,
        "ecc": 0.20,
        "fwhm": 3.0,
        "obj": "",
        "filt": "",
        "exposure": "",
    }
    base.update(kw)
    return base


def _verdict(row):
    v = row["verdict"]
    return None if not v else v["code"]


def _symptoms(row):
    return [s["code"] for s in row["symptoms"]]


class QualityFlagTests(unittest.TestCase):
    def _set(self, odd, n=9, **common):
        return [_frame(**common) for _ in range(n)] + [_frame(**{**common, **odd})]

    def test_uniform_set_is_clean(self):
        rows = quality_flag_reasons([_frame() for _ in range(8)])
        self.assertTrue(all(not _symptoms(r) and _verdict(r) is None for r in rows))

    def test_small_set_is_left_alone(self):
        rows = quality_flag_reasons([
            _frame(), _frame(), _frame(sky=8000, stars=0, ecc=0.9, fwhm=12),
        ])
        self.assertTrue(all(_verdict(r) is None and not _symptoms(r) for r in rows))

    def test_moderate_sky_is_a_note(self):
        rows = quality_flag_reasons(self._set({"sky": 1500}))
        self.assertEqual(_symptoms(rows[-1]), ["sky_moderate"])
        self.assertIsNone(_verdict(rows[-1]))
        self.assertTrue(all(_verdict(r) is None for r in rows[:-1]))

    def test_extreme_sky_is_flagged(self):
        rows = quality_flag_reasons(self._set({"sky": 3000}))
        self.assertEqual(_verdict(rows[-1]), "sky_extreme")
        self.assertTrue(all(_verdict(r) is None for r in rows[:-1]))

    def test_dimmer_sky_is_not_a_symptom(self):
        rows = quality_flag_reasons(self._set({"sky": 400}))
        self.assertEqual(_symptoms(rows[-1]), [])
        self.assertIsNone(_verdict(rows[-1]))

    def test_few_stars_when_the_field_is_gone(self):
        rows = quality_flag_reasons(self._set({"stars": 4}))
        self.assertEqual(_verdict(rows[-1]), "few_stars")
        self.assertGreaterEqual(rows[-1]["verdict"]["star_ref"], 25)

    def test_a_thinner_star_count_is_not_flagged(self):
        # Half the high end, and even a fifth of it, still shows a star field.
        for stars in (120, 40):
            rows = quality_flag_reasons(self._set({"stars": stars}))
            self.assertNotIn("few_stars", _symptoms(rows[-1]))
            self.assertIsNone(_verdict(rows[-1]))

    def test_clear_frames_survive_a_mostly_cloudy_night(self):
        frames = [_frame(stars=200) for _ in range(4)]
        frames += [_frame(stars=5) for _ in range(8)]
        rows = quality_flag_reasons(frames)
        self.assertTrue(all(_verdict(r) is None for r in rows[:4]))
        self.assertTrue(all(_verdict(r) == "few_stars" for r in rows[4:]))

    def test_broadband_without_stars_flags_the_whole_group(self):
        rows = quality_flag_reasons([
            _frame(stars=2, fwhm=0, ecc=0, filt="L", obj="M42") for _ in range(6)
        ])
        self.assertTrue(all(_verdict(r) == "no_star_field" for r in rows))

    def test_narrowband_without_stars_is_left_alone(self):
        rows = quality_flag_reasons([
            _frame(stars=2, fwhm=0, ecc=0, filt="Ha", obj="M42") for _ in range(6)
        ])
        self.assertTrue(all(_verdict(r) is None for r in rows))

    def test_narrowband_without_stars_and_a_much_brighter_sky(self):
        luma = [_frame(filt="L", obj="M42") for _ in range(8)]
        ha = [_frame(stars=2, sky=3000, fwhm=0, ecc=0, filt="Ha", obj="M42") for _ in range(6)]
        rows = quality_flag_reasons(luma + ha)
        self.assertTrue(all(_verdict(r) is None for r in rows[:8]))
        self.assertTrue(all(_verdict(r) == "no_stars_bright" for r in rows[8:]))

    def test_narrowband_matches_only_a_much_brighter_sky(self):
        luma = [_frame(filt="L", obj="M42") for _ in range(8)]
        ha = [_frame(stars=2, sky=1500, fwhm=0, ecc=0, filt="Ha", obj="M42") for _ in range(6)]
        rows = quality_flag_reasons(luma + ha)
        self.assertTrue(all(_verdict(r) is None for r in rows[8:]))

    def test_star_counts_are_not_compared_across_a_moving_field(self):
        frames = [
            _frame(ra=0.0, dec=0.0, field=1.0) for _ in range(6)
        ]
        frames += [
            _frame(stars=30, ra=10.0, dec=0.0, field=1.0) for _ in range(4)
        ]
        rows = quality_flag_reasons(frames)
        self.assertTrue(all("few_stars" not in _symptoms(r) for r in rows))
        self.assertTrue(all(_verdict(r) is None for r in rows))

    def test_shape_is_not_judged_below_twelve_stars(self):
        frames = [_frame(stars=15) for _ in range(6)]
        frames.append(_frame(stars=8, fwhm=30, ecc=0.9))
        rows = quality_flag_reasons(frames)
        self.assertFalse({"soft", "eccentricity", "trailing"} & set(_symptoms(rows[-1])))
        self.assertIsNone(_verdict(rows[-1]))

    def test_soft_stars_alone(self):
        rows = quality_flag_reasons(self._set({"fwhm": 6.0}))
        self.assertEqual(_verdict(rows[-1]), "soft")

    def test_trailing_replaces_ecc_and_fwhm(self):
        rows = quality_flag_reasons(self._set({"ecc": 0.55, "fwhm": 6.0}))
        self.assertEqual(_verdict(rows[-1]), "trailing")

    def test_safety_net_on_a_homogeneous_night(self):
        frames = [_frame(stars=4) for _ in range(14)]
        frames += [_frame(stars=200) for _ in range(10)]
        rows = quality_flag_reasons(frames)
        self.assertTrue(all(_verdict(r) is None for r in rows))

    def test_only_frame_on_its_sky_loses_the_flag(self):
        frames = [_frame(ra=0.0, dec=0.0, field=1.0) for _ in range(8)]
        frames.append(_frame(fwhm=8.0, ra=5.0, dec=0.0, field=1.0))
        rows = quality_flag_reasons(frames)
        self.assertIsNone(_verdict(rows[-1]))
        self.assertEqual(rows[-1]["note"]["code"], "only_cover")
        self.assertTrue(all(_verdict(r) is None for r in rows[:-1]))

    def test_group_without_stars_is_not_restored_by_coverage(self):
        frames = [
            _frame(stars=2, fwhm=0, ecc=0, filt="L", obj="M42",
                   ra=float(i), dec=0.0, field=1.0)
            for i in range(6)
        ]
        rows = quality_flag_reasons(frames)
        self.assertTrue(all(_verdict(r) == "no_star_field" for r in rows))

    def test_unread_frame_is_not_a_flag_and_not_in_the_reference(self):
        frames = [_frame() for _ in range(6)]
        frames.append({"ok": False, "sky": 1e9, "stars": 0, "ecc": 1, "fwhm": 20})
        rows = quality_flag_reasons(frames)
        self.assertTrue(all(_verdict(r) is None and not _symptoms(r) for r in rows))


if __name__ == "__main__":
    unittest.main()
