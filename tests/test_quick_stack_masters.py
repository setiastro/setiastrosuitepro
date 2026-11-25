"""Quick Stack never fails a stack over a master that does not match the frames.

A master is fitted to the frames once per run: a debayered master on raw
frames gives its mosaic back, a raw master on debayered frames is debayered,
and anything that cannot be made to fit (size, rotation, binning, units,
an unreadable file) is left out with a line in the log saying why.

Files are written to a temp folder and read back through SASpro's own
loaders, so the shapes are what a user's files give.
"""
import numpy as np
import pytest
from astropy.io import fits

from setiastro.saspro import quick_stack_worker as qs

H, W = 128, 160


def _write(path, data, **cards):
    hdr = fits.Header()
    for k, v in cards.items():
        hdr[k] = v
    fits.writeto(str(path), data, hdr, overwrite=True)
    return str(path)


def _colour(path, rgb, **cards):
    """A colour FITS as SASpro and most apps write it: (3,H,W)."""
    return _write(path, np.ascontiguousarray(np.moveaxis(rgb, -1, 0)), **cards)


def _dark(rng, h=H, w=W):
    d = 0.01 + rng.normal(0, 0.0005, (h, w))
    hot = rng.choice(h * w, 40, replace=False)
    d.flat[hot] += rng.uniform(0.02, 0.2, 40)
    return d.astype(np.float32)


def _flat(rng, pattern="RGGB"):
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.3 * (((yy - H / 2) / H) ** 2 + ((xx - W / 2) / W) ** 2)
    gain = {"R": 0.45, "G": 1.0, "B": 0.3}
    g = np.empty((H, W), np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            g[dy::2, dx::2] = gain[pattern[2 * dy + dx]]
    return (0.5 * vig * g * (1 + rng.normal(0, 0.003, (H, W)))).astype(np.float32)


def _debayer(m, pattern, method="bilinear"):
    from setiastro.saspro.legacy.numba_utils import debayer_fits_fast
    return np.asarray(debayer_fits_fast(m, pattern, method=method), np.float32)


def _raw_light(tmp, rng, pattern="RGGB", name="light.fits"):
    raw = np.clip(0.05 + rng.normal(0, 0.002, (H, W)), 0, 1)
    cards = {"BAYERPAT": pattern} if pattern else {}
    return _write(tmp / name, (raw * 65535).astype(np.uint16), **cards)


def _colour_light(tmp, rng, name="rgb_light.fits"):
    return _colour(tmp / name, np.clip(0.05 + rng.normal(0, 0.002, (H, W, 3)), 0, 1).astype(np.float32))


def _load(paths, **choice):
    """(applied MasterChoice, {what: shared pixels}, log lines)."""
    lines = []
    ms = qs.MasterSet.load(qs.MasterChoice(**choice), paths, lines.append)
    try:
        return ms.choice, {w: b.view(ms.shapes[w], np.float32).copy() for w, b in ms.blocks.items()}, lines
    finally:
        ms.close()


@pytest.fixture
def rng():
    return np.random.default_rng(3)


# ── debayered master, raw frames ──────────────────────────────────────────────
@pytest.mark.parametrize("method", ["bilinear", "edge", "vng", "rcd", "ahd"])
@pytest.mark.parametrize("pattern", ["RGGB", "GRBG"])
def test_a_debayered_dark_gives_its_mosaic_back(tmp_path, rng, method, pattern):
    dark = _dark(rng)
    mpath = _colour(tmp_path / "dark_rgb.fits", _debayer(dark, pattern, method))
    used, arrs, lines = _load([_raw_light(tmp_path, rng, pattern)], dark=mpath)
    assert used.dark == mpath, lines
    np.testing.assert_array_equal(arrs["dark"], dark)
    assert "taken back out" in lines[-1]


@pytest.mark.parametrize("method", ["bilinear", "rcd"])
def test_a_debayered_flat_gives_its_mosaic_back(tmp_path, rng, method):
    flat = _flat(rng)
    mpath = _colour(tmp_path / "flat_rgb.fits", _debayer(flat, "RGGB", method))
    used, arrs, lines = _load([_raw_light(tmp_path, rng)], flat=mpath)
    assert used.flat == mpath, lines
    want = flat.copy()
    for dy in (0, 1):
        for dx in (0, 1):                       # each site of the mosaic to its own median
            want[dy::2, dx::2] /= np.median(flat[dy::2, dx::2])
    np.testing.assert_allclose(arrs["flat"], want, rtol=1e-6)


def test_a_mosaic_found_where_the_frames_name_no_pattern(tmp_path, rng):
    """Raw colour frames with no BAYERPAT card load as one plane; a bilinear
    master still shows which samples were measured."""
    dark = _dark(rng)
    mpath = _colour(tmp_path / "dark_rgb.fits", _debayer(dark, "RGGB"))
    used, arrs, _ = _load([_raw_light(tmp_path, rng, pattern="")], dark=mpath)
    np.testing.assert_array_equal(arrs["dark"], dark)


def test_a_colour_image_on_single_plane_frames_is_left_out(tmp_path, rng):
    rgb = np.stack([_dark(rng) for _ in range(3)], axis=-1)       # no mosaic in it
    mpath = _colour(tmp_path / "dark_rgb.fits", rgb)
    used, arrs, lines = _load([_raw_light(tmp_path, rng, pattern="")], dark=mpath)
    assert not used and not arrs
    assert any("not applied" in s for s in lines) and "uncalibrated" in lines[-1]


def test_a_debayered_master_with_another_pattern_says_so(tmp_path, rng):
    dark = _dark(rng)
    mpath = _colour(tmp_path / "dark_rgb.fits", _debayer(dark, "GBRG"))
    used, arrs, lines = _load([_raw_light(tmp_path, rng, "RGGB")], dark=mpath)
    np.testing.assert_array_equal(arrs["dark"], dark)               # its own samples
    assert any("was debayered as GBRG" in s for s in lines)


def test_a_mono_master_saved_as_rgb_is_one_plane(tmp_path, rng):
    dark = _dark(rng)
    mpath = _colour(tmp_path / "dark_grey.fits", np.repeat(dark[..., None], 3, axis=-1))
    used, arrs, _ = _load([_raw_light(tmp_path, rng)], dark=mpath)
    np.testing.assert_array_equal(arrs["dark"], dark)


# ── raw master, debayered frames ──────────────────────────────────────────────
def test_a_raw_master_is_debayered_for_colour_frames(tmp_path, rng):
    dark, flat = _dark(rng), _flat(rng)
    d = _write(tmp_path / "dark.fits", dark, BAYERPAT="RGGB")
    f = _write(tmp_path / "flat.fits", flat, BAYERPAT="RGGB")
    used, arrs, lines = _load([_colour_light(tmp_path, rng)], dark=d, flat=f)
    assert (used.dark, used.flat) == (d, f), lines
    np.testing.assert_array_equal(arrs["dark"], np.moveaxis(_debayer(dark, "RGGB"), -1, 0))
    np.testing.assert_allclose(arrs["flat"],
                               np.moveaxis(qs.normalize_flat(_debayer(flat, "RGGB")), -1, 0), rtol=1e-6)
    assert "debayered as RGGB" in lines[-1]


def test_a_raw_flat_with_no_pattern_is_left_out_for_colour_frames(tmp_path, rng):
    f = _write(tmp_path / "flat.fits", _flat(rng))
    d = _write(tmp_path / "dark.fits", _dark(rng))                  # a dark: one plane for every channel
    used, arrs, lines = _load([_colour_light(tmp_path, rng)], dark=d, flat=f)
    assert used.dark == d and used.flat is None
    assert arrs["dark"].shape == (H, W)
    assert any("no BAYERPAT" in s and "not applied" in s for s in lines)


def test_colour_masters_calibrate_each_channel(tmp_path, rng):
    rgb = np.stack([_dark(rng) for _ in range(3)], axis=-1)
    light = _colour_light(tmp_path, rng)
    used, arrs, _ = _load([light], dark=_colour(tmp_path / "d.fits", rgb))
    img = np.full((H, W, 3), 0.5, np.float32)
    out = qs.calibrate_image(img, arrs["dark"], None, colour=True)
    np.testing.assert_allclose(out, img - rgb, atol=1e-7)


# ── what cannot be made to fit ────────────────────────────────────────────────
@pytest.mark.parametrize("shape, hint", [((W, H), "turned 90°"), ((2 * H, 2 * W), "another binning?"),
                                         ((H - 2, W), "")])
def test_a_master_of_another_size_is_left_out(tmp_path, rng, shape, hint):
    d = _write(tmp_path / "dark.fits", _dark(rng, *shape))
    used, arrs, lines = _load([_raw_light(tmp_path, rng)], dark=d)
    assert not used and not arrs
    note = next(s for s in lines if "not applied" in s)
    assert f"is {shape[1]} x {shape[0]}, the frames are {W} x {H}" in note and hint in note


def test_a_dark_far_brighter_than_the_frames_is_left_out(tmp_path, rng):
    d = _write(tmp_path / "dark.fits", _dark(rng) * 16)               # 12-bit data in other units
    used, arrs, lines = _load([_raw_light(tmp_path, rng)], dark=d)
    assert not used and any("far brighter" in s for s in lines)


def test_an_unreadable_master_is_left_out(tmp_path, rng):
    bad = tmp_path / "dark.fits"
    bad.write_bytes(b"not a fits file at all" * 100)
    flat = _write(tmp_path / "flat.fits", _flat(rng), BAYERPAT="RGGB")
    used, arrs, lines = _load([_raw_light(tmp_path, rng)], dark=str(bad), flat=flat)
    assert used.dark is None and used.flat == flat
    assert any("dark.fits could not be read: not applied" in s for s in lines)


def test_a_frame_that_does_not_fit_the_masters_is_skipped_not_an_error(tmp_path, rng):
    raw = _raw_light(tmp_path, rng)
    ms = qs.MasterSet.load(qs.MasterChoice(dark=_write(tmp_path / "d.fits", _dark(rng))), [raw])
    try:
        img, why = qs.load_frame(_colour_light(tmp_path, rng), ms.descriptor())
        assert img is None and "fitted to raw single-plane frames" in why
        img, _hdr = qs.load_frame(raw, ms.descriptor())
        assert img is not None
    finally:
        qs.forget_masters()
        ms.close()


def test_pick_masters_says_what_size_the_masters_are(tmp_path, rng):
    d = _write(tmp_path / "dark.fits", _dark(rng, W, H))
    frame = qs.probe_frame(_raw_light(tmp_path, rng))
    choice, notes = qs.pick_masters(frame, [qs.probe_frame(d)], [])
    assert not choice
    assert any(f"dark.fits is {H} x {W}, turned 90°" in s for s in notes)


def test_a_colour_fits_stored_pixel_by_pixel_probes_at_its_real_size(tmp_path, rng):
    p = _write(tmp_path / "hwc.fits", np.zeros((H, W, 3), np.float32))   # NAXIS1 = 3
    info = qs.probe_frame(p)
    assert (info.width, info.height) == (W, H)


# ── a whole stack ─────────────────────────────────────────────────────────────
def _star_frames(tmp, dark, n=5, size=320, pattern="RGGB"):
    rng = np.random.default_rng(11)
    k = 70
    xs, ys = rng.uniform(15, size - 15, k), rng.uniform(15, size - 15, k)
    amp = rng.uniform(0.05, 0.4, k)
    yy, xx = np.mgrid[-7:8, -7:8]
    paths = []
    for i in range(n):
        dx, dy = rng.uniform(-5, 5, 2)
        img = 0.05 + dark + rng.normal(0, 0.002, (size, size))
        for x, y, a in zip(xs + dx, ys + dy, amp):
            cx, cy = int(round(x)), int(round(y))
            img[cy - 7:cy + 8, cx - 7:cx + 8] += a * np.exp(
                -((xx + cx - x) ** 2 + (yy + cy - y) ** 2) / (2 * 1.6 ** 2))
        paths.append(_write(tmp / f"light_{i}.fits", (np.clip(img, 0, 1) * 65535).astype(np.uint16),
                            BAYERPAT=pattern, EXPTIME=60.0))
    return paths


def test_a_stack_with_a_debayered_dark_and_a_wrong_size_flat(tmp_path):
    dark = _dark(np.random.default_rng(5), 320, 320)
    paths = _star_frames(tmp_path, dark)
    mdark = _colour(tmp_path / "dark_rgb.fits", _debayer(dark, "RGGB"), EXPTIME=60.0)
    mflat = _write(tmp_path / "flat.fits", np.ones((640, 640), np.float32), BAYERPAT="RGGB")
    lines = []
    r = qs.stack_frames(paths, kind="thread", log=lines.append,
                        masters=qs.MasterChoice(dark=mdark, flat=mflat, bayer="RGGB"))
    assert r.used == len(paths), (lines, r.skipped)
    hist = " ".join(str(v) for v in r.header["HISTORY"])
    assert "master dark dark_rgb.fits (its RGGB mosaic" in hist and "flat.fits" not in hist
    assert any("flat.fits is 640 x 640" in s and "another binning?" in s for s in lines)
