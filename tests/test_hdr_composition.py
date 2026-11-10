"""HDR composition: saturated cores of a long exposure are rebuilt from a short one."""
from __future__ import annotations

import numpy as np
import pytest

from setiastro.saspro.hdr_composition import (
    HDRCompositionError,
    compose_hdr,
    exposure_seconds,
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
    a = np.zeros((8, 8), dtype=np.float32)
    b = np.zeros((8, 9), dtype=np.float32)
    with pytest.raises(HDRCompositionError):
        compose_hdr([a, b])


def test_exposure_seconds_reads_header_and_xisf_keywords():
    class Doc:
        def __init__(self, meta):
            self.metadata = meta

    assert exposure_seconds(Doc({"EXPTIME": "120"})) == 120.0
    assert exposure_seconds(Doc({
        "original_header": {"FITSKeywords": {"EXPOSURE": [{"value": 300}]}},
    })) == 300.0
    assert exposure_seconds(Doc({})) is None
