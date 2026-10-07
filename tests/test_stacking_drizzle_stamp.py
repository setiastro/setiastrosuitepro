from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

from setiastro.saspro.stacking_suite import StackingSuiteDialog  # noqa: E402

PREFIX = "SASDZORGPATH="
FIRST_CARD_PATH_CHARS = 72 - len(PREFIX)
NORM_DIR = "D:\\stack2\\Normalized_Images\\"
PAN2 = NORM_DIR + "a355e6_32e542_Light_Pan2_10.0s_LP_20260915-"

HOMOGRAPHY = np.array([[1.0001, 0.0027, 40.7],
                       [-0.0023, 0.9998, 123.1],
                       [9.1e-07, -3.5e-07, 1.0]])


def _stamp(hdr, orig_path, matrix=HOMOGRAPHY):
    """Stamp a header the way star_alignment does for an aligned frame."""
    hdr["SASDZ"] = (True, "SASpro drizzle stamp present")
    hdr["SASDZKND"] = ("homography", "Drizzle transform kind")
    hdr["SASDZRFH"] = (1920, "Drizzle reference height (px)")
    hdr["SASDZRFW"] = (1080, "Drizzle reference width (px)")
    hdr["SASDZORG"] = (os.path.basename(orig_path), "Drizzle source original (basename)")
    hdr.add_comment(f"{PREFIX}{orig_path}")
    hdr["SASDZNR"] = 3
    hdr["SASDZNC"] = 3
    for i, v in enumerate(np.asarray(matrix, np.float64).ravel()):
        hdr[f"SASDZ{i:02d}"] = float(v)


class ReadDrizzleStampOriginalPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="saspro-dzstamp-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, orig_path, *, extra_comments=(), stamp=True, comment=True):
        hdr = fits.Header()
        if stamp:
            _stamp(hdr, orig_path)
            if not comment:
                del hdr["COMMENT"]
        for c in extra_comments:
            hdr.add_comment(c)
        out = os.path.join(self.tmp, name)
        fits.PrimaryHDU(np.zeros((4, 4), np.float32), hdr).writeto(out)
        return out

    def _read(self, path):
        return StackingSuiteDialog._read_drizzle_stamp(None, path)

    def test_short_path_unchanged(self):
        orig = r"D:\s\N\a_c_n.fit"
        stamp = self._read(self._write("short.fit", orig))
        self.assertEqual(stamp["orig_path"], orig)
        self.assertEqual(stamp["kind"], "homography")
        self.assertEqual(stamp["ref_shape"], (1920, 1080))
        np.testing.assert_allclose(stamp["matrix"], HOMOGRAPHY)

    def test_path_split_over_two_cards(self):
        orig = PAN2 + "003030_c_n.fit"
        self.assertGreater(len(orig), FIRST_CARD_PATH_CHARS)
        stamp = self._read(self._write("two.fit", orig))
        self.assertEqual(stamp["orig_path"], orig)

    def test_path_split_over_three_or_more_cards(self):
        folders = "\\".join("very_long_folder_name_%02d" % i for i in range(8))
        orig = "D:\\" + folders + "\\frame_0001_c_n.fit"
        self.assertGreater(len(orig) + len(PREFIX), 3 * 72)
        stamp = self._read(self._write("many.fit", orig))
        self.assertEqual(stamp["orig_path"], orig)

    def test_space_at_card_boundary_is_kept(self):
        # astropy strips the trailing space of a card on read; the path must survive it.
        pad = "x" * (FIRST_CARD_PATH_CHARS - len(NORM_DIR) - 1)
        orig = NORM_DIR + pad + " Trunk_21_20.0s_LP_c_n.fit"
        self.assertEqual(orig[FIRST_CARD_PATH_CHARS - 1], " ")
        stamp = self._read(self._write("space.fit", orig))
        self.assertEqual(stamp["orig_path"], orig)

    def test_unrelated_comment_after_path_is_not_swallowed(self):
        orig = PAN2 + "003030_c_n.fit"
        path = self._write("extra.fit", orig, extra_comments=("unrelated note",))
        stamp = self._read(path)
        self.assertEqual(stamp["orig_path"], orig)

    def test_basename_only_when_comment_missing(self):
        orig = PAN2 + "003030_c_n.fit"
        stamp = self._read(self._write("nocomment.fit", orig, comment=False))
        self.assertEqual(stamp["orig_path"], os.path.basename(orig))

    def test_frames_sharing_a_long_prefix_keep_distinct_originals(self):
        a = PAN2 + "003030_c_n.fit"
        b = PAN2 + "003041_c_n.fit"
        self.assertEqual(a[:FIRST_CARD_PATH_CHARS], b[:FIRST_CARD_PATH_CHARS])
        sa = self._read(self._write("a.fit", a))
        sb = self._read(self._write("b.fit", b))
        self.assertEqual((sa["orig_path"], sb["orig_path"]), (a, b))

    def test_unstamped_frame_returns_none(self):
        self.assertIsNone(self._read(self._write("plain.fit", "", stamp=False)))

    def test_long_original_on_disk_is_found(self):
        # Drizzle deposits from the original only if this path exists; otherwise
        # it logs "Original missing" and deposits the aligned pixels instead.
        norm_dir = os.path.join(self.tmp, "Normalized_Images")
        os.makedirs(norm_dir)
        orig = os.path.join(
            norm_dir, "a355e6_32e542_Light_Pan2_10.0s_LP_20260915-003030_c_n.fit")
        self.assertGreater(len(orig), FIRST_CARD_PATH_CHARS)
        fits.PrimaryHDU(np.zeros((4, 4), np.float32)).writeto(orig)
        stamp = self._read(self._write("aligned.fit", orig))
        self.assertTrue(os.path.isfile(stamp["orig_path"]), stamp["orig_path"])


if __name__ == "__main__":
    unittest.main()
