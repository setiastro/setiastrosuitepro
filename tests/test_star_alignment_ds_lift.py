from __future__ import annotations

import cv2
import numpy as np

from setiastro.saspro.star_alignment import (
    _S,
    downscale_affine_2x3_to_ds,
    lift_affine_2x3_from_ds,
)

# 6248 // 3 = 2082 -> the true resize scale is 3.00096, not 3
W, H, DS = 6248, 4176, 3


def _apply(A, pts):
    return pts @ A[:, :2].T + A[:, 2]


def test_S_matches_cv2_resize_pixel_centres():
    S = _S(DS, (W, H))
    # cv2.resize maps x_full = sx * (x_ds + 0.5) - 0.5
    sx, sy = W / (W // DS), H / (H // DS)
    ds_pts = np.array([[0.0, 0.0], [100.25, 37.5], [2081.0, 1391.0]])
    full = np.column_stack([sx * (ds_pts[:, 0] + 0.5) - 0.5, sy * (ds_pts[:, 1] + 0.5) - 0.5])
    np.testing.assert_allclose(_apply(S[:2], full), ds_pts, atol=1e-9)


def test_S_identity_at_ds1():
    np.testing.assert_allclose(_S(1, (W, H)), np.eye(3))


def test_lift_flipped_transform_lands_on_reference():
    # 180° (meridian flip) transform full-res: x' = (W-1) - x, y' = (H-1) - y
    A_full = np.array([[-1.0, 0.0, W - 1.0], [0.0, -1.0, H - 1.0]])
    S = _S(DS, (W, H))

    # Exact DS-grid equivalent: a point in the DS source maps to the DS ref
    A_ds = downscale_affine_2x3_to_ds(A_full, DS, S)
    lifted = lift_affine_2x3_from_ds(A_ds, DS, S)
    np.testing.assert_allclose(lifted, A_full, atol=1e-9)

    # Solve-grid images built the way the aligner builds them must agree with A_ds
    rng = np.random.default_rng(0)
    img = np.zeros((H, W), np.float32)
    for x, y in rng.uniform([200, 200], [W - 200, H - 200], size=(30, 2)):
        cv2.circle(img, (int(x), int(y)), 6, 1.0, -1)
    ref = cv2.warpAffine(img, A_full, (W, H), flags=cv2.INTER_LINEAR)
    src_ds = cv2.resize(img, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    ref_ds = cv2.resize(ref, (W // DS, H // DS), interpolation=cv2.INTER_AREA)
    warped = cv2.warpAffine(src_ds, A_ds, (W // DS, H // DS), flags=cv2.INTER_LINEAR)
    (dx, dy), _ = cv2.phaseCorrelate(ref_ds.astype(np.float64), warped.astype(np.float64))
    assert abs(dx) < 0.1 and abs(dy) < 0.1
