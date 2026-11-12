"""HDR composition: saturated cores of a long exposure are rebuilt from a short one."""
from __future__ import annotations

import numpy as np
import pytest

from astropy.io.fits import Header

from setiastro.saspro.hdr_composition import (
    HDRCompositionError,
    compose_hdr,
    crop_header,
    exposure_order_warning,
    exposure_seconds,
    publish_hdr_composition,
)
from setiastro.saspro.hdr_composition_preset import (
    apply_hdr_composition_via_preset,
    hdr_composition_params_from_preset,
    resolve_hdr_docs,
)


def _star(n=96, sky=0.02, amp=0.45) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n]
    r2 = (y - (n / 2.0)) ** 2 + (x - (n / 2.0)) ** 2
    return (sky + amp * np.exp(-r2 / 18.0)).astype(np.float32)


def test_replaces_saturated_core_and_keeps_sky():
    short = _star()
    long = np.minimum(short * 5.0, 0.8).astype(np.float32)
    out, mask, steps = compose_hdr(
        [long, short],
        threshold=0.8,
        growth=1,
        smoothness=2.0,
        names=["long", "short"],
    )
    center = out[48, 48]
    assert center == pytest.approx(float(short[48, 48] * 5.0), abs=0.05)
    assert out[0, 0] == pytest.approx(float(long[0, 0]), abs=1e-3)
    assert mask[48, 48] == pytest.approx(1.0, abs=1e-3)
    assert mask[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert steps[0]["scales"][0] == pytest.approx(5.0, rel=0.02)
    assert steps[0]["offsets"][0] == pytest.approx(0.0, abs=0.02)
    assert out.max() > 1.0


def test_later_frame_does_not_overwrite_recovered_cores():
    short = _star()
    long = np.minimum(short * 5.0, 0.8).astype(np.float32)
    decoy = np.full_like(short, 0.01)
    out, _mask, steps = compose_hdr(
        [long, short, decoy],
        threshold=0.8,
        growth=1,
        smoothness=2.0,
    )
    assert out[48, 48] == pytest.approx(float(short[48, 48] * 5.0), abs=0.05)
    assert steps[1]["scales"][0] is None
    assert steps[1]["replaced"] == pytest.approx(0.0, abs=1e-8)


def test_small_core_stays_fully_replaced_when_the_feather_is_wide():
    y, x = np.mgrid[0:48, 0:48]
    r2 = (y - 24.0) ** 2 + (x - 24.0) ** 2
    short = (0.05 + 0.20 * np.exp(-r2 / 4.0)).astype(np.float32)
    long = np.minimum(short * 6.0, 0.8).astype(np.float32)
    out, mask, _steps = compose_hdr(
        [long, short], threshold=0.8, growth=1, smoothness=7.0,
    )
    assert mask[24, 24] == pytest.approx(1.0, abs=1e-6)
    assert out[24, 24] == pytest.approx(float(short[24, 24] * 6.0), abs=0.05)
    assert out[0, 0] == pytest.approx(float(long[0, 0]), abs=1e-3)


def test_rgb_is_composed_per_channel():
    short = _star()
    short_rgb = np.dstack([short, short * 0.8, short * 0.8]).astype(np.float32)
    long_rgb = np.minimum(np.dstack([
        short * 4.0,
        short * 0.8 * 4.0,
        short * 0.8 * 4.0,
    ]), 0.8).astype(np.float32)
    out, _mask, steps = compose_hdr(
        [long_rgb, short_rgb], threshold=0.8, growth=1, smoothness=2.0,
    )
    assert out.shape == long_rgb.shape
    assert steps[0]["scales"][0] == pytest.approx(4.0, rel=0.03)
    assert out[48, 48, 0] == pytest.approx(float(short[48, 48] * 4.0), abs=0.08)


def test_shape_mismatch_is_rejected():
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.zeros((64, 200), dtype=np.float32)
    with pytest.raises(HDRCompositionError, match="do not line up"):
        compose_hdr([a, b])


def test_one_pixel_difference_uses_the_shared_area():
    short = _star()
    long = np.minimum(short * 5.0, 0.8).astype(np.float32)
    long_wide = np.pad(long, ((0, 0), (0, 1)), mode="edge")
    out, _mask, steps = compose_hdr(
        [long_wide, short],
        threshold=0.8,
        growth=1,
        smoothness=2.0,
        names=["long", "short"],
    )
    assert out.shape == short.shape
    assert out[48, 48] == pytest.approx(float(short[48, 48] * 5.0), abs=0.05)
    assert "overlapping" in steps[0]["size_note"]
    assert steps[0]["crop_origin"] == (0, 0)
    assert steps[0]["crop_size"] == (short.shape[1], short.shape[0])


def test_shifted_crop_lines_up_before_composing():
    short = _star()
    long = np.minimum(short * 5.0, 0.8).astype(np.float32)
    # Extra pixels on the top and left: the content is not at the origin.
    shifted = np.pad(long, ((3, 1), (2, 4)), mode="edge")
    out, _mask, steps = compose_hdr(
        [shifted, short],
        threshold=0.8,
        growth=1,
        smoothness=2.0,
        names=["long", "short"],
    )
    assert out[48, 48] == pytest.approx(float(short[48, 48] * 5.0), abs=0.05)
    assert "Δx=" in steps[0]["size_note"]
    assert steps[0]["crop_origin"] == (2, 3)
    assert steps[0]["crop_size"] == (short.shape[1], short.shape[0])


def test_channel_first_mono_matches_a_plane():
    short = _star()
    long = np.minimum(short * 5.0, 0.8).astype(np.float32)
    out, _mask, steps = compose_hdr(
        [long[None, ...], short],
        threshold=0.8,
        growth=1,
        smoothness=2.0,
    )
    assert out.shape == short.shape
    assert "size_note" not in steps[0]


def test_exposure_seconds_reads_header_and_xisf_keywords():
    class Doc:
        def __init__(self, meta):
            self.metadata = meta

    assert exposure_seconds(Doc({"EXPTIME": "120"})) == 120.0
    assert exposure_seconds(Doc({
        "original_header": {"FITSKeywords": {"EXPOSURE": [{"value": 300}]}},
    })) == 300.0
    assert exposure_seconds(Doc({})) is None


class _Named:
    def __init__(self, name, meta):
        self.metadata = meta
        self._name = name

    def display_name(self):
        return self._name


def test_exposure_order_warning_when_the_longest_is_not_first():
    short = _Named("short", {"EXPTIME": 30})
    long = _Named("long", {"EXPTIME": 300})
    assert exposure_order_warning([long, short]) is None
    warning = exposure_order_warning([short, long])
    assert "long" in warning
    assert "top of the list" in warning


def test_exposure_order_warning_when_a_view_has_no_time():
    long = _Named("long", {"EXPTIME": 300})
    mystery = _Named("mystery", {})
    warning = exposure_order_warning([long, mystery])
    assert "mystery" in warning
    assert "bottom" in warning


def test_crop_header_shifts_crpix_and_leaves_the_original():
    header = Header()
    header["CRPIX1"] = 100.0
    header["CRPIX2"] = 80.0
    header["CRVAL1"] = 10.0
    header["NAXIS1"] = 200
    header["NAXIS2"] = 180
    header["OBJECT"] = "M42"
    out = crop_header(header, 5, 7, 190, 160)
    assert float(out["CRPIX1"]) == pytest.approx(95.0)
    assert float(out["CRPIX2"]) == pytest.approx(73.0)
    assert int(out["NAXIS1"]) == 190
    assert int(out["NAXIS2"]) == 160
    assert float(out["CRVAL1"]) == pytest.approx(10.0)
    assert out["OBJECT"] == "M42"
    assert float(header["CRPIX1"]) == pytest.approx(100.0)


def test_crop_header_shifts_xisf_keywords():
    header = {"FITSKeywords": {
        "CRPIX1": [{"value": 50}],
        "CRPIX2": [{"value": 40}],
        "NAXIS1": [{"value": 100}],
        "NAXIS2": [{"value": 80}],
    }}
    out = crop_header(header, 4, 6, 90, 70)
    assert out["FITSKeywords"]["CRPIX1"][0]["value"] == pytest.approx(46.0)
    assert out["FITSKeywords"]["CRPIX2"][0]["value"] == pytest.approx(34.0)
    assert out["FITSKeywords"]["NAXIS1"][0]["value"] == 90
    assert header["FITSKeywords"]["CRPIX1"][0]["value"] == 50


def test_params_from_preset_clamps_and_parses_titles():
    p = hdr_composition_params_from_preset({
        "threshold": 9.0,
        "growth": -3,
        "smoothness": 100,
        "open_mask": 1,
        "doc_titles": "Long, Short",
    })
    assert p["threshold"] == pytest.approx(1.20)
    assert p["growth"] == 0
    assert p["smoothness"] == pytest.approx(64.0)
    assert p["open_mask"] is True
    assert p["doc_titles"] == ["Long", "Short"]


class _ImgDoc:
    def __init__(self, name, meta, image):
        self.metadata = meta
        self.image = image
        self._name = name

    def display_name(self):
        return self._name


class _FakeDM:
    def __init__(self):
        self.created = []

    def create_document(self, image, metadata=None, name=None):
        doc = _ImgDoc(name or "out", dict(metadata or {}), image)
        self.created.append(doc)
        return doc


class _FakeMain:
    def __init__(self, docs):
        self._docs = list(docs)
        self.doc_manager = _FakeDM()
        self.logs = []
        self.last = None

    def _list_open_docs(self):
        return list(self._docs)

    def _spawn_subwindow_for(self, doc):
        self._docs.append(doc)

    def _log(self, msg):
        self.logs.append(msg)

    def remember_last_headless_command(self, command_id, preset, description=""):
        self.last = {
            "command_id": command_id,
            "preset": dict(preset or {}),
            "description": description,
        }


def test_resolve_hdr_docs_by_titles_and_exposure_fallback():
    short = _ImgDoc("short", {"EXPTIME": 30}, _star())
    long = _ImgDoc("long", {"EXPTIME": 300}, np.minimum(_star() * 5.0, 0.8))
    main = _FakeMain([short, long])

    by_title = resolve_hdr_docs(main, {"doc_titles": ["long", "short"]})
    assert [d.display_name() for d in by_title] == ["long", "short"]

    auto = resolve_hdr_docs(main, {}, target_doc=short)
    assert auto[0] is short
    assert long in auto


def test_apply_hdr_composition_via_preset_publishes_and_remembers():
    short = _star()
    long = np.minimum(short * 5.0, 0.8).astype(np.float32)
    docs = [
        _ImgDoc("long", {"EXPTIME": 300}, long),
        _ImgDoc("short", {"EXPTIME": 30}, short),
    ]
    main = _FakeMain(docs)
    summary, steps = apply_hdr_composition_via_preset(
        main,
        {"threshold": 0.8, "growth": 1, "smoothness": 2.0, "doc_titles": ["long", "short"]},
    )
    assert "short" in summary
    assert steps[0]["scales"][0] == pytest.approx(5.0, rel=0.05)
    assert len(main.doc_manager.created) == 1
    out = main.doc_manager.created[0].image
    assert out[48, 48] == pytest.approx(float(short[48, 48] * 5.0), abs=0.05)
    assert main.last["command_id"] == "hdr_composition"
    assert main.last["preset"]["doc_titles"] == ["long", "short"]


def test_publish_hdr_composition_opens_mask_when_requested():
    short = _star()
    long = np.minimum(short * 5.0, 0.8).astype(np.float32)
    result, mask, steps = compose_hdr([long, short], threshold=0.8, growth=1, smoothness=2.0)
    main = _FakeMain([])
    base = _ImgDoc("long", {}, long)
    summary = publish_hdr_composition(
        main, main.doc_manager, base, result, mask, steps, open_mask=True,
    )
    assert summary is not None
    assert len(main.doc_manager.created) == 2
    assert main.doc_manager.created[1].display_name() == "HDR Composition Mask"
