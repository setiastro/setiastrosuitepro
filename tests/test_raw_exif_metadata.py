"""Camera-raw metadata: exposure time (and friends) from the file itself.

CR3 converted with EXPTIME=Unknown no matter what: exifread can't open Canon's
ISO-BMFF container, and the LibRaw fallback read `raw.metadata`, an attribute
rawpy never had. The tests build a minimal synthetic CR3 (ftyp, moov with
Canon's uuid box holding CMT1/CMT2 TIFF streams, mdat) so they run without a
34 MB fixture; the layout matches real CR3s from an EOS Rebel T8i.
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from astropy.io import fits  # noqa: E402

from setiastro.saspro.legacy import image_manager as im  # noqa: E402

_ASCII, _SHORT, _RATIONAL, _SRATIONAL = 2, 3, 5, 10


def _tiff(entries):
    """Little-endian TIFF stream with one IFD. entries: (tag, type, value)."""
    entries = sorted(entries)
    ifd_size = 2 + 12 * len(entries) + 4
    data_off = 8 + ifd_size
    ifd = struct.pack("<H", len(entries))
    data = b""
    for tag, typ, val in entries:
        if typ == _ASCII:
            raw = val.encode("ascii") + b"\0"
            count = len(raw)
        elif typ == _SHORT:
            raw = struct.pack("<H", val)
            count = 1
        elif typ == _RATIONAL:
            raw = struct.pack("<II", *val)
            count = 1
        else:                                   # SRATIONAL
            raw = struct.pack("<ii", *val)
            count = 1
        if len(raw) <= 4:
            ifd += struct.pack("<HHI", tag, typ, count) + raw.ljust(4, b"\0")
        else:
            ifd += struct.pack("<HHII", tag, typ, count, data_off + len(data))
            data += raw + (b"\0" if len(raw) % 2 else b"")
    ifd += struct.pack("<I", 0)
    return b"II*\0" + struct.pack("<I", 8) + ifd + data


def _box(typ, body):
    return struct.pack(">I4s", 8 + len(body), typ) + body


def _cr3(path, cmt2_entries):
    cmt1 = _tiff([(0x010F, _ASCII, "Canon"),
                  (0x0110, _ASCII, "Canon EOS Rebel T8i"),
                  (0x0132, _ASCII, "2026:08:15 02:14:00")])
    canon = (im._CR3_CANON_UUID
             + _box(b"CNCV", b"CanonCR3_001/01.09.00/00.00.00")
             + _box(b"CMT1", cmt1)
             + _box(b"CMT2", _tiff(cmt2_entries)))
    with open(path, "wb") as f:
        f.write(_box(b"ftyp", b"crx \x00\x00\x00\x01crx isom"))
        f.write(_box(b"moov", _box(b"uuid", canon) + _box(b"mvhd", b"\0" * 100)))
        f.write(_box(b"mdat", b"\0" * 256))


_T8I_CMT2 = [
    (0x829A, _RATIONAL, (149, 5)),              # ExposureTime: measured bulb, 29.8 s
    (0x829D, _RATIONAL, (0, 1)),                # FNumber: unknown (telescope)
    (0x8827, _SHORT, 3200),                     # ISOSpeedRatings
    (0x9003, _ASCII, "2026:08:15 02:14:00"),    # DateTimeOriginal
    (0x9201, _SRATIONAL, (-2147483648, 1)),     # ShutterSpeedValue: Canon's "unset"
    (0x920A, _RATIONAL, (0, 1)),                # FocalLength: unknown (telescope)
]


class CR3ExifTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="saspro-cr3-exif-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def header(self, cmt2_entries):
        p = os.path.join(self.tmp, "IMG_7818.CR3")
        _cr3(p, cmt2_entries)
        return im._enrich_header_from_exif(fits.Header(), p)

    def test_exposure_iso_date_and_camera_come_from_the_cr3(self):
        h = self.header(_T8I_CMT2)
        self.assertAlmostEqual(h["EXPTIME"], 29.8)
        self.assertEqual(h["ISO"], 3200)
        self.assertEqual(h["DATE-OBS"], "2026-08-15T02:14:00")
        self.assertEqual(h["INSTRUME"], "Canon EOS Rebel T8i")

    def test_zero_aperture_and_focal_length_are_left_out(self):
        h = self.header(_T8I_CMT2)
        self.assertNotIn("FNUMBER", h)
        self.assertNotIn("FOCALLEN", h)

    def test_apex_shutter_speed_is_the_fallback_not_seconds(self):
        entries = [e for e in _T8I_CMT2 if e[0] not in (0x829A, 0x9201)]
        entries.append((0x9201, _SRATIONAL, (-5, 1)))    # Tv = -5 -> 2**5 = 32 s
        self.assertAlmostEqual(self.header(entries)["EXPTIME"], 32.0)

    def test_canon_unset_shutter_speed_does_not_become_an_exposure(self):
        entries = [e for e in _T8I_CMT2 if e[0] != 0x829A]
        self.assertNotIn("EXPTIME", self.header(entries))

    def test_existing_keys_win(self):
        p = os.path.join(self.tmp, "IMG_7818.CR3")
        _cr3(p, _T8I_CMT2)
        h = fits.Header()
        h["EXPTIME"] = 30.0
        self.assertEqual(im._enrich_header_from_exif(h, p)["EXPTIME"], 30.0)

    def test_not_a_cr3_container_gives_no_tags(self):
        p = os.path.join(self.tmp, "broken.CR3")
        with open(p, "wb") as f:
            f.write(b"\0\0\0\x08junk" + b"\xff" * 64)
        self.assertEqual(im._read_exif_tags(p), {})

    def test_stacking_suite_uses_the_same_reader(self):
        from setiastro.saspro import stacking_suite
        self.assertIs(stacking_suite._enrich_header_from_exif, im._enrich_header_from_exif)


class LibRawFallbackTests(unittest.TestCase):
    """rawpy's `raw.other` fills only what EXIF didn't."""

    @staticmethod
    def raw(**kw):
        import datetime
        other = dict(iso_speed=3200.0, shutter_speed=29.799999237060547, aperture=0.0,
                     focal_length=0.0, timestamp=datetime.datetime(2026, 8, 15, 2, 14))
        other.update(kw)
        return types.SimpleNamespace(other=types.SimpleNamespace(**other))

    def test_fills_gaps_at_float32_precision(self):
        h = im._fill_hdr_from_raw_metadata(self.raw(), fits.Header())
        self.assertEqual(h["EXPTIME"], 29.8)
        self.assertEqual(h["ISO"], 3200)
        self.assertEqual(h["DATE-OBS"], "2026-08-15T02:14:00")
        self.assertNotIn("FNUMBER", h)
        self.assertNotIn("FOCALLEN", h)

    def test_never_overrides_exif(self):
        h = fits.Header()
        h["EXPTIME"] = 30.0
        self.assertEqual(im._fill_hdr_from_raw_metadata(self.raw(), h)["EXPTIME"], 30.0)

    def test_rawpy_without_other_is_a_no_op(self):
        h = im._fill_hdr_from_raw_metadata(types.SimpleNamespace(), fits.Header())
        self.assertEqual(len(h), 0)


if __name__ == "__main__":
    unittest.main()
