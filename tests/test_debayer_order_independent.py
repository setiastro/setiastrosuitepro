"""Bilinear and edge debayer read only real samples.

They used to fill a sparse array in place from a parallel loop, so a pixel
could average samples its neighbours had just invented: the result changed
from run to run and shifted star centroids between channels.
"""
import numpy as np
import pytest

from setiastro.saspro.legacy import numba_utils as nu

PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
CH = {"R": 0, "G": 1, "B": 2}


def _channel(bp, y, x):
    return CH[bp[2 * (y & 1) + (x & 1)]]


def _box_mean(m, bp, y, x, c):
    H, W = m.shape
    v = [m[yy, xx] for yy in range(y - 1, y + 2) for xx in range(x - 1, x + 2)
         if 0 <= yy < H and 0 <= xx < W and _channel(bp, yy, xx) == c and m[yy, xx] != 0]
    return float(np.mean(np.float64(v))) if v else 0.0


def _reference(m, bp, method):
    H, W = m.shape
    out = np.zeros((H, W, 3))
    for y in range(H):
        for x in range(W):
            for c in range(3):
                if c == _channel(bp, y, x):
                    out[y, x, c] = m[y, x]
                    continue
                if method == "edge":
                    def at(yy, xx):
                        return (m[yy, xx] if 0 <= yy < H and 0 <= xx < W
                                and _channel(bp, yy, xx) == c else 0.0)
                    l, r, t, b = at(y, x - 1), at(y, x + 1), at(y - 1, x), at(y + 1, x)
                    hh, hv = l != 0 and r != 0, t != 0 and b != 0
                    gh, gv = abs(l - r) if hh else 1e6, abs(t - b) if hv else 1e6
                    if gh < gv and hh:
                        out[y, x, c] = 0.5 * (l + r)
                        continue
                    if gv <= gh and hv:
                        out[y, x, c] = 0.5 * (t + b)
                        continue
                out[y, x, c] = _box_mean(m, bp, y, x, c)
    out[m == 0] = 0.0      # clipped pixels stay no-data in every channel
    return out


def _mosaic(h, w, zeros=False, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.05, 0.9, (h, w)).astype(np.float32)
    if zeros:
        m[rng.random(m.shape) < 0.05] = 0.0
    return m


@pytest.mark.parametrize("method", ("bilinear", "edge"))
@pytest.mark.parametrize("bp", PATTERNS)
@pytest.mark.parametrize("shape,zeros", (((24, 30), False), ((25, 31), True), ((8, 8), True)))
def test_matches_a_plain_reference(bp, method, shape, zeros):
    m = _mosaic(*shape, zeros=zeros)
    got = nu.debayer_fits_fast(m, bp, method=method)
    assert np.abs(got - _reference(m, bp, method)).max() < 2e-6


@pytest.mark.parametrize("method", ("bilinear", "edge"))
def test_the_same_frame_always_gives_the_same_result(method):
    m = _mosaic(700, 900, seed=3)
    first = nu.debayer_fits_fast(m, "RGGB", method=method)
    for _ in range(5):
        assert np.array_equal(first, nu.debayer_fits_fast(m, "RGGB", method=method))


@pytest.mark.parametrize("bp", PATTERNS)
def test_a_ramp_comes_back_exactly(bp):
    yy, xx = np.mgrid[0:80, 0:80].astype(np.float32)
    ramp = 0.2 + 0.002 * xx + 0.001 * yy
    got = nu.debayer_fits_fast(ramp, bp, method="bilinear")[4:-4, 4:-4]
    assert np.abs(got - ramp[4:-4, 4:-4, None]).max() < 1e-6


@pytest.mark.parametrize("method", ("bilinear", "edge"))
def test_a_star_stays_put_in_every_channel(method):
    yy, xx = np.mgrid[0:101, 0:101]
    star = (0.1 + 0.8 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / (2 * 1.6 ** 2))).astype(np.float32)
    got = nu.debayer_fits_fast(star, "RGGB", method=method)
    for c in range(3):
        w = np.clip(got[..., c] - 0.1, 0, None)
        assert abs((w * xx).sum() / w.sum() - 50) < 0.01
        assert abs((w * yy).sum() / w.sum() - 50) < 0.01


def test_full_resolution_kernels_agree_with_the_dispatcher():
    m = _mosaic(40, 44, seed=5)
    for bp in PATTERNS:
        fn = getattr(nu, f"debayer_{bp}_fullres_fast")
        assert np.array_equal(fn(m), nu.debayer_fits_fast(m, bp, method="edge", preserve_zeros=False))
