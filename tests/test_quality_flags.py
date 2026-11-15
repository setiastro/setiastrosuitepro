"""Grade and suspect verdicts for Blink Comparator pre-flag."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from setiastro.saspro.stacking_suite import quality_grades  # noqa: E402


def _frame(**kw):
    base = {
        "ok": True,
        "sky": 1000.0,
        "stars": 200,
        "ecc": 0.20,
        "fwhm": 3.0,
        "noise": 0.01,
        "patch": 0.06,
        "trail": 0.28,
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


def _grade(row):
    return float(row["grade"])


class QualityGradeTests(unittest.TestCase):
    def _set(self, odd, n=9, **common):
        return [_frame(**common) for _ in range(n)] + [_frame(**{**common, **odd})]

    def test_uniform_set_is_clean_and_high_grade(self):
        rows = quality_grades([_frame() for _ in range(8)])
        self.assertTrue(all(_verdict(r) is None for r in rows))
        self.assertTrue(all(_grade(r) >= 90 for r in rows))

    def test_small_set_is_left_alone(self):
        rows = quality_grades([
            _frame(), _frame(), _frame(sky=8000, stars=0, ecc=0.9, fwhm=12),
        ])
        self.assertTrue(all(_verdict(r) is None for r in rows))
        self.assertTrue(all(not np_isfinite(_grade(r)) for r in rows))

    def test_moderate_sky_lowers_grade_without_flag(self):
        rows = quality_grades(self._set({"sky": 1500}))
        self.assertIn("sky_moderate", _symptoms(rows[-1]))
        self.assertIsNone(_verdict(rows[-1]))
        self.assertLess(_grade(rows[-1]), _grade(rows[0]))

    def test_extreme_sky_is_flagged(self):
        rows = quality_grades(self._set({"sky": 3000}))
        self.assertEqual(_verdict(rows[-1]), "sky_extreme")
        self.assertTrue(all(_verdict(r) is None for r in rows[:-1]))

    def test_dimmer_sky_is_not_penalized(self):
        rows = quality_grades(self._set({"sky": 400}))
        self.assertNotIn("sky_moderate", _symptoms(rows[-1]))
        self.assertIsNone(_verdict(rows[-1]))
        self.assertGreaterEqual(_grade(rows[-1]), _grade(rows[0]) - 1)

    def test_collapsed_star_field_is_flagged(self):
        rows = quality_grades(self._set({"stars": 4}))
        self.assertEqual(_verdict(rows[-1]), "few_stars")
        self.assertLess(_grade(rows[-1]), _grade(rows[0]))

    def test_mildly_thinner_star_count_is_not_flagged(self):
        # Half the high end still shows a field; grade drops, no flag.
        rows = quality_grades(self._set({"stars": 120}))
        self.assertIsNone(_verdict(rows[-1]))
        self.assertLess(_grade(rows[-1]), _grade(rows[0]))

    def test_quarter_of_the_high_end_is_flagged(self):
        rows = quality_grades(self._set({"stars": 40}))
        self.assertEqual(_verdict(rows[-1]), "few_stars")

    def test_clear_frames_survive_a_mostly_cloudy_night(self):
        frames = [_frame(stars=200) for _ in range(4)]
        frames += [_frame(stars=5) for _ in range(8)]
        rows = quality_grades(frames)
        self.assertTrue(all(_verdict(r) is None for r in rows[:4]))
        self.assertTrue(all(_verdict(r) == "few_stars" for r in rows[4:]))

    def test_broadband_without_stars_flags_the_whole_group(self):
        rows = quality_grades([
            _frame(stars=2, fwhm=0, ecc=0, filt="L", obj="M42") for _ in range(6)
        ])
        self.assertTrue(all(_verdict(r) == "no_star_field" for r in rows))
        self.assertTrue(all(_grade(r) == 0 for r in rows))

    def test_narrowband_without_stars_is_left_alone(self):
        rows = quality_grades([
            _frame(stars=2, fwhm=0, ecc=0, filt="Ha", obj="M42") for _ in range(6)
        ])
        self.assertTrue(all(_verdict(r) is None for r in rows))

    def test_narrowband_without_stars_and_a_much_brighter_sky(self):
        luma = [_frame(filt="L", obj="M42") for _ in range(8)]
        ha = [_frame(stars=2, sky=3000, fwhm=0, ecc=0, filt="Ha", obj="M42") for _ in range(6)]
        rows = quality_grades(luma + ha)
        self.assertTrue(all(_verdict(r) is None for r in rows[:8]))
        self.assertTrue(all(_verdict(r) == "no_stars_bright" for r in rows[8:]))

    def test_star_counts_are_not_compared_across_a_moving_field(self):
        frames = [
            _frame(ra=0.0, dec=0.0, field=1.0) for _ in range(6)
        ]
        frames += [
            _frame(stars=30, ra=10.0, dec=0.0, field=1.0) for _ in range(4)
        ]
        rows = quality_grades(frames)
        self.assertTrue(all("few_stars" not in _symptoms(r) for r in rows))
        self.assertTrue(all(_verdict(r) is None for r in rows))

    def test_shape_is_not_judged_below_twelve_stars(self):
        frames = [_frame(stars=15) for _ in range(6)]
        frames.append(_frame(stars=8, fwhm=30, ecc=0.9))
        rows = quality_grades(frames)
        self.assertFalse({"soft", "eccentricity", "trailing"} & set(_symptoms(rows[-1])))

    def test_soft_stars_alone(self):
        rows = quality_grades(self._set({"fwhm": 6.0}))
        self.assertEqual(_verdict(rows[-1]), "soft")

    def test_trailing_replaces_ecc_and_fwhm(self):
        rows = quality_grades(self._set({"ecc": 0.55, "fwhm": 6.0}))
        self.assertEqual(_verdict(rows[-1]), "trailing")

    def test_noisy_brighter_sky_is_flagged(self):
        rows = quality_grades(self._set({"sky": 1600, "noise": 0.02}))
        self.assertEqual(_verdict(rows[-1]), "noisy_sky")

    def test_patchy_sky_is_flagged(self):
        rows = quality_grades(self._set({"patch": 0.22}))
        self.assertEqual(_verdict(rows[-1]), "patchy_sky")

    def test_high_end_trail_flags_eccentricity(self):
        # Median ecc stays normal; only the high-end trail rises.
        rows = quality_grades(self._set({"ecc": 0.22, "trail": 0.55, "fwhm": 3.0}))
        self.assertEqual(_verdict(rows[-1]), "eccentricity")
        self.assertIn("eccentricity", _symptoms(rows[-1]))

    def test_trail_plus_soft_is_trailing(self):
        rows = quality_grades(self._set({"ecc": 0.22, "trail": 0.55, "fwhm": 6.0}))
        self.assertEqual(_verdict(rows[-1]), "trailing")

    def test_only_frame_on_its_sky_loses_the_flag(self):
        frames = [_frame(ra=0.0, dec=0.0, field=1.0) for _ in range(8)]
        frames.append(_frame(fwhm=8.0, ra=5.0, dec=0.0, field=1.0))
        rows = quality_grades(frames)
        self.assertIsNone(_verdict(rows[-1]))
        self.assertEqual(rows[-1]["note"]["code"], "only_cover")

    def test_unread_frame_is_not_a_flag_and_not_in_the_reference(self):
        frames = [_frame() for _ in range(6)]
        frames.append({"ok": False, "sky": 1e9, "stars": 0, "ecc": 1, "fwhm": 20})
        rows = quality_grades(frames)
        self.assertTrue(all(_verdict(r) is None for r in rows))


def np_isfinite(x):
    import math
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    unittest.main()
