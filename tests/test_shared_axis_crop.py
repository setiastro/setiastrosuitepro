"""Batch crop keeps one pixel rectangle even when a frame is a few pixels smaller."""
from __future__ import annotations

import numpy as np

from setiastro.saspro.crop_dialog_pro import shared_axis_crop


def test_shared_rectangle_matches_the_primary_when_the_frame_is_smaller():
    src = np.arange(6 * 6, dtype=np.float32).reshape(6, 6)
    cropped, matrix = shared_axis_crop(src, (0, 8, 0, 6))
    assert cropped.shape == (6, 8)
    assert np.all(cropped[:, :6] == src)
    assert np.all(cropped[:, 6:] == 0)
    assert matrix[0, 2] == 0
    assert matrix[1, 2] == 0


def test_shared_rectangle_crops_a_larger_frame_at_the_primary_origin():
    src = np.arange(10 * 12, dtype=np.float32).reshape(10, 12)
    cropped, matrix = shared_axis_crop(src, (2, 6, 1, 5))
    assert cropped.shape == (4, 4)
    assert cropped[0, 0] == src[1, 2]
    assert matrix[0, 2] == -2
    assert matrix[1, 2] == -1
