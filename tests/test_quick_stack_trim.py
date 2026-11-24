"""Quick Stack's streamed trimmed mean and its one sky offset per frame.

The trim keeps only a running sum, a count and the two highest / two lowest
samples per pixel, so it is checked against a brute force over every sample.
"""
import os

import numpy as np
import pytest

from setiastro.saspro import quick_stack_worker as qs

RNG = np.random.default_rng(11)
IDENTITY = np.array([[1.0, 0, 0], [0, 1.0, 0]])


def _acc(shape):
    """The accumulators as the thread engine sets them up."""
    A = {n: np.zeros(shape, dt) for n, dt in qs.SharedArrays.SPEC}
    A["hi0"][...] = A["hi1"][...] = -np.inf
    A["lo"][...] = A["lo1"][...] = np.inf
    return A


def _add(A, f, off):
    qs._accumulate_band(f, IDENTITY, off, 0, f.shape[0], A["sumv"], A["cnt"],
                        A["hi0"], A["hi1"], A["lo"], A["lo1"])


def _finish(A):
    out = np.empty(A["sumv"].shape, np.float32)
    qs._finish(A["sumv"], A["cnt"], A["hi0"], A["hi1"], A["lo"], A["lo1"], out,
               qs.MIN_TRIM_FRAMES, qs.FULL_TRIM_FRAMES)
    return out


def _brute(frames, offs):
    """The trimmed mean, sample by sample, of the pixels the kernel can reach
    (identity warp: all but the last row and column)."""
    H, W, C = frames[0].shape
    out = np.zeros((H - 1, W - 1, C))
    for y in range(H - 1):
        for x in range(W - 1):
            for c in range(C):
                rv = [np.float32(f[y, x, c] - o[c]) for f, o in zip(frames, offs)
                      if not (f[y:y + 2, x:x + 2, c] == 0.0).any()]   # bilinear needs all four
                n = len(rv)
                if n == 0:
                    continue
                k = 0 if n < qs.MIN_TRIM_FRAMES else (1 if n < qs.FULL_TRIM_FRAMES else 2)
                out[y, x, c] = np.sort(np.array(rv, np.float64))[k:n - k].mean()
    return out


@pytest.mark.parametrize("F", [4, 9, 21])
def test_the_trim_is_exact(F):
    H, W, C = 9, 11, 3
    frames, offs = [], []
    for _ in range(F):
        f = RNG.normal(0.2, 0.05, (H, W, C)).astype(np.float32)
        f[RNG.random((H, W, C)) < 0.15] = 0.0                # no data: counts differ per pixel
        f[RNG.integers(0, H), RNG.integers(0, W)] = 0.9      # a hot pixel
        frames.append(f)
        offs.append(RNG.uniform(-0.05, 0.05, C).astype(np.float32))
    A = _acc((H, W, C))
    for f, o in zip(frames, offs):
        _add(A, f, o)
    np.testing.assert_allclose(_finish(A)[:-1, :-1], _brute(frames, offs), rtol=2e-6, atol=2e-7)


def test_the_trim_does_not_depend_on_order():
    H, W, C = 5, 5, 3
    frames = [RNG.normal(0.3, 0.1, (H, W, C)).astype(np.float32) for _ in range(12)]
    zero = np.zeros(C, np.float32)
    runs = []
    for order in (range(12), reversed(range(12))):
        A = _acc((H, W, C))
        for i in order:
            _add(A, frames[i], zero)
        runs.append((A, _finish(A)))
    (A1, o1), (A2, o2) = runs
    for n in ("cnt", "hi0", "hi1", "lo", "lo1"):
        assert np.array_equal(A1[n], A2[n]), n
    np.testing.assert_allclose(o1, o2, rtol=1e-6)


def test_the_sky_leaves_no_data_out():
    img = np.full((40, 40, 2), 0.25, np.float32)
    img[..., 1] = 0.125
    img[:24] = 0.0                                       # no data, most of the frame
    img[24, 0] = np.nan
    assert list(qs.channel_sky(img)) == [0.25, 0.125]


# ── end to end, CPU threads ──────────────────────────────────────────────────
def _field(H, W, n, seed, sigma=1.4):
    rng = np.random.default_rng(seed)
    img = rng.normal(0.05, 0.004, (H, W)).astype(np.float32)
    yy, xx = np.mgrid[:H, :W]
    for _ in range(n):
        y, x = rng.uniform(8, H - 8), rng.uniform(8, W - 8)
        img += rng.uniform(0.02, 0.6) * np.exp(-(((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2) / 2)
    return img.astype(np.float32)


def _write(path, img):
    from astropy.io import fits
    fits.PrimaryHDU(img.astype(np.float32), fits.Header({"EXPTIME": 60.0})).writeto(path)
    return str(path)


def test_a_stack_puts_every_frame_on_the_references_sky(tmp_path):
    base = _field(320, 400, 180, 5)
    base += np.float32(0.1) * np.linspace(0, 1, base.shape[1], dtype=np.float32)   # a gradient
    paths = []
    for i in range(7):
        dy, dx = (i % 3) * 2, (i // 3) * 3                   # dither, whole pixels
        img = np.roll(np.roll(base, dy, 0), dx, 1) + np.random.default_rng(100 + i).normal(0, 0.002, base.shape)
        if i == 5:
            img = img + 0.04                                 # a brighter sky
        paths.append(_write(tmp_path / f"f{i}.fits", img))
    res = qs.stack_frames(paths, kind="thread", workers=2, log=lambda s: None)
    assert res.used == 7
    d = dict(res.frames)
    assert d[os.path.normpath(paths[5])]["offset"] == pytest.approx(0.04, abs=0.003)
    assert abs(d[os.path.normpath(paths[1])]["offset"]) < 0.003
    assert "weight" not in d[os.path.normpath(paths[1])]
    hist = "".join(str(h) for h in res.header["HISTORY"])
    assert "trimmed mean" in hist and "one sky offset per frame" in hist and "weight" not in hist
    sky = np.median(res.image[40:-40, 40:-40])
    assert sky == pytest.approx(np.median(base), abs=0.003)      # at the reference's level
