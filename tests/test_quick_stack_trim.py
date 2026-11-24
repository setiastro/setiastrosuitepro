"""Quick Stack's streamed, star-count-weighted trimmed mean and its one sky
offset per frame.

The trim keeps only a running weighted sum, a count, the sum of the weights
and the two highest / two lowest samples per pixel (each with its frame's
weight code), so it is checked against a brute force over every sample.
"""
import os

import numpy as np
import pytest

from setiastro.saspro import quick_stack_worker as qs

RNG = np.random.default_rng(11)
IDENTITY = np.array([[1.0, 0, 0], [0, 1.0, 0]])


def _acc(shape):
    """The accumulators as the thread engine sets them up."""
    return qs.SharedArrays.blank({n: np.empty(shape, dt) for n, dt in qs.SharedArrays.SPEC})


def _add(A, f, off, q=qs.EQUAL_WEIGHT_CODE):
    qs.accumulate(A, f, IDENTITY, off, q, 0, f.shape[0])


def _finish(A):
    out = np.empty(A["sumv"].shape, np.float32)
    qs.finish(A, out)
    return out


def _brute(frames, offs, codes=None, bound=False):
    """The trimmed mean, sample by sample, of the pixels the kernel can reach
    (identity warp: all but the last row and column). The extremes are picked
    by value; the rest are averaged with their frames' weights.

    bound: also the float32 running sum's rounding, as it reaches the result
    (it is taken over every sample, and the dropped ones may be the heavy)."""
    H, W, C = frames[0].shape
    wts = [1.0] * len(frames) if codes is None else [float(qs.WEIGHT_TABLE[q]) for q in codes]
    out = np.zeros((H - 1, W - 1, C))
    err = np.zeros_like(out)
    for y in range(H - 1):
        for x in range(W - 1):
            for c in range(C):
                rv = [(float(np.float32(f[y, x, c] - o[c])), w) for f, o, w in zip(frames, offs, wts)
                      if not (f[y:y + 2, x:x + 2, c] == 0.0).any()]   # bilinear needs all four
                n = len(rv)
                if n == 0:
                    continue
                k = 0 if n < qs.MIN_TRIM_FRAMES else (1 if n < qs.FULL_TRIM_FRAMES else 2)
                kept = sorted(rv)[k:n - k]
                kw = sum(w for _, w in kept)
                out[y, x, c] = sum(v * w for v, w in kept) / kw
                err[y, x, c] = n * np.finfo(np.float32).eps * sum(abs(v) * w for v, w in rv) / kw
    return (out, err) if bound else out


def _frames(F, H=9, W=11, C=3):
    frames, offs = [], []
    for _ in range(F):
        f = RNG.normal(0.2, 0.05, (H, W, C)).astype(np.float32)
        f[RNG.random((H, W, C)) < 0.15] = 0.0                # no data: counts differ per pixel
        f[RNG.integers(0, H), RNG.integers(0, W)] = 0.9      # a hot pixel
        frames.append(f)
        offs.append(RNG.uniform(-0.05, 0.05, C).astype(np.float32))
    return frames, offs


@pytest.mark.parametrize("F", [4, 9, 21])
def test_the_trim_is_exact(F):
    frames, offs = _frames(F)
    A = _acc(frames[0].shape)
    for f, o in zip(frames, offs):
        _add(A, f, o)
    np.testing.assert_allclose(_finish(A)[:-1, :-1], _brute(frames, offs), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("F", [4, 9, 21])
def test_the_weighted_trim_is_exact(F):
    """Each frame its own weight, from 1 star to the table's top: a dropped
    extreme must take its own frame's weight out, not an average one."""
    frames, offs = _frames(F)
    codes = [int(q) for q in RNG.integers(1, 256, F)]
    codes[0], codes[-1] = 1, 255
    A = _acc(frames[0].shape)
    for f, o, q in zip(frames, offs, codes):
        _add(A, f, o, q)
    want, err = _brute(frames, offs, codes, bound=True)
    got = _finish(A)[:-1, :-1]
    assert (np.abs(got - want) <= 2e-7 + err).all(), np.abs(got - want).max()
    # and with weights as one night's frames differ (1 to 1/10), to float32's own precision
    codes = [int(q) for q in RNG.integers(140, 194, F)]
    A = _acc(frames[0].shape)
    for f, o, q in zip(frames, offs, codes):
        _add(A, f, o, q)
    np.testing.assert_allclose(_finish(A)[:-1, :-1], _brute(frames, offs, codes), rtol=3e-6, atol=3e-7)


def test_equal_weights_are_the_plain_trim_bit_for_bit():
    """WEIGHT_TABLE[EQUAL_WEIGHT_CODE] is exactly 1, so a run whose frames
    weigh the same is the trimmed mean Quick Stack gave before weights."""
    assert qs.WEIGHT_TABLE[qs.EQUAL_WEIGHT_CODE] == 1.0
    frames, offs = _frames(17)
    A = _acc(frames[0].shape)
    for f, o in zip(frames, offs):
        _add(A, f, o)
    n = A["cnt"].astype(np.int64)
    s = A["sumv"].astype(np.float64)
    k1, k2 = n >= qs.MIN_TRIM_FRAMES, n >= qs.FULL_TRIM_FRAMES
    s = np.where(k1, s - A["hi0"].astype(np.float64), s)
    s = np.where(k1, s - A["lo"].astype(np.float64), s)
    s = np.where(k2, s - A["hi1"].astype(np.float64), s)
    s = np.where(k2, s - A["lo1"].astype(np.float64), s)
    drop = np.where(k2, 4, np.where(k1, 2, 0))
    plain = np.where(n > 0, s / np.maximum(n - drop, 1), 0.0).astype(np.float32)
    assert np.array_equal(_finish(A), plain)


def test_the_trim_does_not_depend_on_order():
    H, W, C = 5, 5, 3
    frames = [RNG.normal(0.3, 0.1, (H, W, C)).astype(np.float32) for _ in range(12)]
    codes = [int(q) for q in RNG.integers(1, 256, 12)]
    zero = np.zeros(C, np.float32)
    runs = []
    for order in (range(12), reversed(range(12))):
        A = _acc((H, W, C))
        for i in order:
            _add(A, frames[i], zero, codes[i])
        runs.append((A, _finish(A)))
    (A1, o1), (A2, o2) = runs
    for n in ("cnt", "hi0", "hi1", "lo", "lo1", "qhi0", "qhi1", "qlo", "qlo1"):
        assert np.array_equal(A1[n], A2[n]), n
    np.testing.assert_allclose(o1, o2, rtol=1e-6)


def test_the_weight_is_the_star_count():
    """Half the stars, half the weight, to the table's 2.2 % steps."""
    for n in (1, 2, 7, 20, 137, 500, 4321, 60000):
        w = float(qs.WEIGHT_TABLE[qs.weight_code(n)])
        assert abs(w / n - 1.0) <= 0.022, (n, w)
    assert qs.weight_code(0) == qs.weight_code(1) == 1          # no stars: one star's worth, never 0
    assert qs.weight_code(10 ** 7) == 255
    w = lambda n: float(qs.WEIGHT_TABLE[qs.weight_code(n)])     # noqa: E731
    assert w(250) / w(500) == pytest.approx(0.5, abs=0.02)
    assert qs.frame_code(500, False) == qs.EQUAL_WEIGHT_CODE
    assert not qs.run_weighting(qs.MIN_WEIGHT_STARS - 1) and qs.run_weighting(qs.MIN_WEIGHT_STARS)


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
    hist = " ".join(str(h) for h in res.header["HISTORY"])
    assert "trimmed mean" in hist and "one sky offset per frame" in hist
    sky = np.median(res.image[40:-40, 40:-40])
    assert sky == pytest.approx(np.median(base), abs=0.003)      # at the reference's level


# ── star counts and weights ──────────────────────────────────────────────────
@pytest.fixture
def sep_only(monkeypatch):
    from setiastro.saspro import star_alignment as sa
    monkeypatch.setattr(sa, "_GPU_DETECT_SLOT", [None])
    return sa


def test_the_points_are_the_matchers_and_the_count_is_not_capped(sep_only):
    """_detect is _detect_stars_uniform's SEP path: the same points to match,
    but the count is taken before its cap of 25 per cell."""
    sa = sep_only
    cfg = qs.align_settings()
    img = _field(480, 640, 900, 21)
    img[100:140, 200:260] = np.nan                       # no data
    for sigma in (cfg["det_sigma"], 5.0):
        pts, n = qs._detect(img, sigma, cfg)
        qs._seed_detection()
        want = sa._detect_stars_uniform(np.array(img), sigma, cfg["minarea"], grid=(4, 4), max_per_cell=25,
                                        max_total=cfg["limit_stars"], min_fwhm=cfg["min_fwhm"],
                                        max_ellipticity=cfg["max_ellipticity"])
        assert np.array_equal(pts, want)
        assert len(pts) <= 16 * 25 < n                   # the list is full; the count goes on


def _stack_with_clouds(tmp_path, n_stars, cloudy, k=7):
    """k dithered frames; `cloudy` ones lose most of their starlight to cloud."""
    base = _field(320, 400, n_stars, 5)
    sky = np.float32(0.05)
    paths = []
    for i in range(k):
        dy, dx = (i % 3) * 2, (i // 3) * 3
        img = np.roll(np.roll(base, dy, 0), dx, 1)
        if i in cloudy:
            img = sky + (img - sky) * np.float32(0.12)
        img = img + np.random.default_rng(300 + i).normal(0, 0.002, base.shape)
        paths.append(_write(tmp_path / f"f{i}.fits", img))
    lines = []
    res = qs.stack_frames(paths, kind="thread", workers=2, log=lines.append)
    return res, {os.path.basename(p): v for p, v in res.frames}, lines


def test_a_frame_under_cloud_weighs_what_its_stars_say(tmp_path, sep_only):
    res, d, lines = _stack_with_clouds(tmp_path, 260, cloudy={4})
    assert res.used == 7 and res.weighted
    clear = d["f1.fits"]
    assert d["f4.fits"]["stars"] < 0.6 * clear["stars"]
    ratio = d["f4.fits"]["weight"] / clear["weight"]
    assert ratio == pytest.approx(d["f4.fits"]["stars"] / clear["stars"], rel=0.05)
    assert any(s.startswith("Weighted by the stars found in each frame") for s in lines), lines
    assert "weighted by star count" in " ".join(str(h) for h in res.header["HISTORY"])


def test_too_few_stars_in_the_reference_and_frames_weigh_the_same(tmp_path, sep_only):
    res, d, lines = _stack_with_clouds(tmp_path, 16, cloudy=set())
    assert res.used == 7 and not res.weighted
    assert {v["weight"] for v in d.values()} == {1.0}
    assert any(s.startswith("Frames weigh the same") for s in lines), lines
    assert "weighted" not in " ".join(str(h) for h in res.header["HISTORY"])

def test_a_short_card_drops_the_weights_not_the_stack():
    """quick_stack_gpu.plan_memory: weights when they fit, else equal weights on
    the card, else no GPU (the CPU engine then stacks, weighted)."""
    from setiastro.saspro import quick_stack_gpu as g
    N, other, held = 10_000_000, 40_000_000, 50_000_000
    base = N * g.ACC_BYTES + other + g.VRAM_RESERVE
    assert g.plan_memory(base + N * g.WEIGHT_BYTES + 20 * held, N, other, held) == (g.MAX_HELD, True)
    assert g.plan_memory(base + N * g.WEIGHT_BYTES + 2 * held, N, other, held) == (2, True)
    assert g.plan_memory(base + 3 * held, N, other, held) == (3, False)
    with pytest.raises(g.GpuUnavailable):
        g.plan_memory(base + held, N, other, held)
