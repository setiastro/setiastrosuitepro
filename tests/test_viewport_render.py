from __future__ import annotations

import numpy as np

from setiastro.saspro.imageops.viewport_render import rasterize_for_display


def test_rasterize_for_display_matches_requested_size_not_source():
    src = np.linspace(0.0, 1.0, 800 * 600 * 3, dtype=np.float32).reshape(800, 600, 3)
    out = rasterize_for_display(src, out_w=120, out_h=80)

    assert out.dtype == np.uint8
    assert out.shape == (80, 120, 3)
    assert out.nbytes < src.nbytes / 10


def test_rasterize_for_display_mono_becomes_rgb():
    src = np.full((200, 160), 0.5, dtype=np.float32)
    out = rasterize_for_display(src, out_w=40, out_h=50)

    assert out.shape == (50, 40, 3)
    assert out.dtype == np.uint8
    np.testing.assert_allclose(out[0, 0], [127, 127, 127], atol=1)


def test_rasterize_for_display_crops_source_region():
    src = np.zeros((100, 100), dtype=np.float32)
    src[10:20, 30:40] = 1.0
    out = rasterize_for_display(
        src, out_w=10, out_h=10, src_x=30, src_y=10, src_w=10, src_h=10
    )

    assert out.shape == (10, 10, 3)
    assert int(out.min()) >= 250
