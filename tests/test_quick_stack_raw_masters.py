"""Quick Stack makes masters from single dark and flat frames, in memory.

Raw darks are grouped by exposure (and size, gain, binning), raw flats by
filter and night; each set stands in the pick lists like a loaded master and
is combined (the lights' trimmed mean) the first time a stack uses it. Flats
are each brought to 1 (per Bayer site) first. Nothing is written to disk.
"""
import os

import numpy as np
import pytest
from astropy.io import fits

from setiastro.saspro import quick_stack_worker as qs

H, W = 96, 128


@pytest.fixture
def rng():
    return np.random.default_rng(3)


@pytest.fixture(autouse=True)
def _clean():
    qs.forget_made()
    yield
    qs.forget_made()


def _write(path, data, **cards):
    hdr = fits.Header()
    for k, v in cards.items():
        hdr[k] = v
    fits.writeto(str(path), data, hdr, overwrite=True)
    return str(path)


def _u16(a):
    return np.clip(np.round(a * 65535), 1, 65535).astype(np.uint16)


def _brute(frames, k_rule=True):
    """The trimmed mean, sample by sample (0 = no data)."""
    st = np.stack(frames).astype(np.float64)
    out = np.zeros(st.shape[1:])
    for idx in np.ndindex(*st.shape[1:]):
        v = np.sort(st[(slice(None),) + idx])
        v = v[v != 0]
        n = len(v)
        if not n:
            continue
        k = 0 if n < qs.MIN_TRIM_FRAMES else (1 if n < qs.FULL_TRIM_FRAMES else 2)
        out[idx] = v[k:n - k].mean()
    return out


def _darks(tmp, rng, n, exp=300.0, gain=100, name="dark", **cards):
    base = 0.01 + rng.normal(0, 0.0004, (H, W))
    out = []
    for i in range(n):
        d = base + rng.normal(0, 0.0003, (H, W))
        if i == 2:
            d[10, 20] = 0.9                                  # a cosmic ray
        out.append(_write(tmp / f"{name}_{i:03d}.fits", _u16(d), IMAGETYP="Dark Frame",
                          EXPTIME=exp, GAIN=gain, **cards))
    return out


def _true_flat(pattern="RGGB"):
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.3 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2)
    gain = {"R": 0.45, "G": 1.0, "B": 0.3}
    g = np.empty((H, W))
    for dy in (0, 1):
        for dx in (0, 1):
            g[dy::2, dx::2] = gain[pattern[2 * dy + dx]]
    return vig * g


def _flats(tmp, rng, n, filt="Ha", date="2026-07-20T11:40:00", levels=None, sub="flats", **cards):
    d = tmp / sub
    d.mkdir(exist_ok=True)
    f = _true_flat()
    out = []
    for i in range(n):
        lv = (levels or [0.5])[i % len(levels or [0.5])]
        a = lv * f * (1 + rng.normal(0, 0.003, (H, W)))
        out.append(_write(d / f"flat_{filt}_{i:03d}.fits", _u16(a), IMAGETYP="Flat Frame",
                          FILTER=filt, EXPTIME=0.1, BAYERPAT="RGGB", **{"DATE-OBS": date}, **cards))
    return out


def _infos(paths):
    return [qs.probe_frame(p) for p in paths]


# ── reading ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("roworder", ["", "TOP-DOWN", "BOTTOM-UP"])
def test_the_quick_read_gives_the_loaders_pixels(tmp_path, rng, roworder):
    a = _u16(rng.uniform(0, 1, (H, W)))
    cards = {"ROWORDER": roworder} if roworder else {}
    p = _write(tmp_path / "d.fits", a, **cards)
    assert qs.u16_plan(p) is not None                       # the quick path is the one tested
    quick, hdr = qs.read_raw_frame(p)
    want, _ = qs.load_master(p)
    assert hdr is None and np.array_equal(quick, want)


def test_a_float_frame_goes_through_the_loader(tmp_path, rng):
    p = _write(tmp_path / "f.fits", rng.uniform(0.1, 0.2, (H, W)).astype(np.float32))
    img, hdr = qs.read_raw_frame(p)
    assert hdr is not None and img.shape == (H, W)


# ── grouping ─────────────────────────────────────────────────────────────────
def test_darks_group_by_exposure_and_flats_by_filter_and_night(tmp_path, rng):
    d300 = _darks(tmp_path, rng, 4, exp=300.0, name="d300")
    d60 = _darks(tmp_path, rng, 3, exp=60.0, name="d60")
    f1 = _flats(tmp_path, rng, 3, "Ha", "2026-07-20T11:40:00", sub="a")
    f2 = _flats(tmp_path, rng, 2, "Ha", "2026-10-12T13:20:00", sub="b")
    f3 = _flats(tmp_path, rng, 2, "OIII", "2026-07-20T11:50:00", sub="c")
    darks = qs.raw_sets(_infos(d300 + d60), "dark")
    flats = qs.raw_sets(_infos(f1 + f2 + f3), "flat")
    assert sorted((d.exposure, len(d.frames)) for d in darks) == [(60.0, 3), (300.0, 4)]
    assert sorted((f.filter, f.date, len(f.frames)) for f in flats) == [
        ("Ha", "2026-07-19", 3), ("Ha", "2026-10-12", 2), ("OIII", "2026-07-19", 2)]
    names = [os.path.basename(m.path) for m in darks + flats]
    assert len(set(names)) == len(names)
    assert "made from 4 dark frames (300 s, gain 100)" in names
    assert all(qs.is_made(m) and not os.path.exists(m.path) for m in darks + flats)


def test_the_flat_from_the_nearest_night_is_picked(tmp_path, rng):
    flats = qs.raw_sets(_infos(_flats(tmp_path, rng, 2, "Ha", "2026-07-20T11:40:00", sub="a")
                               + _flats(tmp_path, rng, 2, "Ha", "2026-10-12T13:20:00", sub="b")), "flat")
    light = qs.FrameInfo(path="x.fits", filter="Ha", exposure=900, width=W, height=H,
                         date="2026-08-01T03:06:40")
    p = qs.plan_masters(light, [], flats)["flat"]
    assert p.state == "auto" and p.master.date == "2026-07-19"
    light.date = "2026-10-01T03:00:00"
    assert qs.plan_masters(light, [], flats)["flat"].master.date == "2026-10-12"


# ── making them ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [4, 9, 17])
def test_a_made_dark_is_the_trimmed_mean_of_its_frames(tmp_path, rng, n):
    paths = _darks(tmp_path, rng, n)
    (m,) = qs.raw_sets(_infos(paths), "dark")
    img, hdr = qs.make_master(m, "dark")
    want = _brute([qs.load_master(p)[0] for p in paths])
    np.testing.assert_allclose(img, want, rtol=1e-6, atol=1e-7)
    assert hdr["IMAGETYP"] == "Master Dark" and hdr["NCOMBINE"] == n
    assert not img.flags.writeable


def test_a_made_flat_brings_each_frame_to_one_per_site(tmp_path, rng):
    paths = _flats(tmp_path, rng, 9, levels=[0.25, 0.5, 0.9])     # a panel at three brightnesses
    (m,) = qs.raw_sets(_infos(paths), "flat")
    img, _ = qs.make_master(m, "flat")
    want = qs.normalize_flat(_true_flat().astype(np.float32), "RGGB")
    got = qs.normalize_flat(img, "RGGB")
    assert float(np.max(np.abs(got / want - 1))) < 0.01            # the vignetting, kept
    for dy in (0, 1):
        for dx in (0, 1):
            assert float(np.median(img[dy::2, dx::2])) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("kind", ["dark", "flat"])
@pytest.mark.parametrize("roworder", ["", "BOTTOM-UP"])
def test_the_16_bit_path_is_the_float_path_to_the_bit(tmp_path, rng, monkeypatch, kind, roworder):
    """16-bit frames are combined straight from their stored samples; the
    result must be what reading them as float first gives: to the bit for one
    frame, to float32 rounding for several (they are summed in the order the
    threads finish, as the lights are)."""
    cards = {"ROWORDER": roworder} if roworder else {}
    paths = (_darks(tmp_path, rng, 7, **cards) if kind == "dark"
             else _flats(tmp_path, rng, 7, levels=[0.3, 0.7], **cards))
    (m,) = qs.raw_sets(_infos(paths), kind)
    one = qs.raw_sets(_infos(paths[:1]), kind)[0]
    quick, quick1 = qs.make_master(m, kind)[0], qs.make_master(one, kind)[0]
    monkeypatch.setattr(qs, "quick_u16", lambda path: False)
    slow, slow1 = qs.make_master(m, kind)[0], qs.make_master(one, kind)[0]
    assert np.array_equal(quick1, slow1)
    np.testing.assert_allclose(quick, slow, rtol=1e-6, atol=0)


def test_frames_that_do_not_fit_are_left_out(tmp_path, rng):
    paths = _darks(tmp_path, rng, 5)
    odd = _write(tmp_path / "dark_odd.fits", _u16(np.full((H, W + 2), 0.01)),
                 IMAGETYP="Dark Frame", EXPTIME=300.0, GAIN=100)
    m = qs.raw_sets(_infos(paths), "dark")[0]
    m.frames = m.frames + (odd, str(tmp_path / "missing.fits"))
    said = []
    img, hdr = qs.make_master(m, "dark", said.append)
    assert hdr["NCOMBINE"] == 5
    assert sum("left out" in s for s in said) == 2, said


def test_made_masters_are_kept_and_reused(tmp_path, rng):
    paths = _darks(tmp_path, rng, 5)
    (m,) = qs.raw_sets(_infos(paths), "dark")
    choice = qs.MasterChoice(m.path, None, "")
    said = []
    qs.ensure_made(choice, [m], said.append)
    assert qs.made_already(m) and sum(s.startswith("Combined 5 dark frames") for s in said) == 1
    qs.ensure_made(choice, [m], said.append)                       # made once
    assert sum(s.startswith("Combined 5 dark frames") for s in said) == 1
    img, _ = qs.load_master(m.path)
    assert img.shape == (H, W)
    qs.forget_made()
    with pytest.raises(qs.MasterMismatch):
        qs.load_master(m.path)


# ── end to end ───────────────────────────────────────────────────────────────
def _stars(tmp, rng, n=6):
    yy, xx = np.mgrid[:H, :W]
    base = np.zeros((H, W))
    for _ in range(60):
        y, x = rng.uniform(6, H - 6), rng.uniform(6, W - 6)
        base += rng.uniform(0.05, 0.5) * np.exp(-(((xx - x) / 1.3) ** 2 + ((yy - y) / 1.3) ** 2) / 2)
    d = tmp / "lights"
    d.mkdir()
    out = []
    for i in range(n):
        dy, dx = i % 3, i // 3
        sky = 0.05 + np.roll(np.roll(base, dy, 0), dx, 1)
        a = 0.01 + sky * _true_flat() + rng.normal(0, 0.001, (H, W))
        out.append(_write(d / f"light_{i}.fits", _u16(a), IMAGETYP="Light Frame", FILTER="Ha",
                          EXPTIME=300.0, GAIN=100, BAYERPAT="RGGB",
                          **{"DATE-OBS": "2026-07-20T03:00:00"}))
    return out


def test_a_stack_with_raw_frames_is_the_stack_with_their_masters(tmp_path, rng):
    """Masters made in memory calibrate exactly as the same masters loaded from files."""
    lights = _stars(tmp_path, rng)
    dset = qs.raw_sets(_infos(_darks(tmp_path, rng, 7)), "dark")
    fset = qs.raw_sets(_infos(_flats(tmp_path, rng, 7, levels=[0.4, 0.6])), "flat")
    frame = qs.probe_frame(lights[0])
    choice, notes = qs.pick_masters(frame, dset, fset)
    assert choice.dark == dset[0].path and choice.flat == fset[0].path
    said = []
    qs.ensure_made(choice, dset + fset, said.append)
    made = qs.stack_frames(lights, masters=choice, kind="thread", workers=2, log=said.append)
    assert any("Calibrating with master dark made from 7 dark frames" in s for s in said), said
    # the same masters, from files
    dpath = _write(tmp_path / "MasterDark.fits", np.array(qs.load_master(choice.dark)[0]))
    fpath = _write(tmp_path / "MasterFlat.fits", np.array(qs.load_master(choice.flat)[0]))
    files = qs.stack_frames(lights, masters=qs.MasterChoice(dpath, fpath, "RGGB"), kind="thread",
                            workers=2, log=lambda s: None)
    np.testing.assert_allclose(made.image, files.image, rtol=1e-6, atol=1e-7)
    assert "made from 7 dark frames" in " ".join(str(h) for h in made.header["HISTORY"])
