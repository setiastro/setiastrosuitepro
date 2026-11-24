"""Quick Stack: which master each group gets, and what an added file is.

Filters written different ways still match (O3 = OIII, H-alpha = Ha), a flat
with no filter card is matched on its name, a choice made in the window wins
over the automatic one, and a drop of lights, masters, single calibration
frames and bias puts each where it belongs.
"""
import os

import numpy as np
import pytest
from astropy.io import fits

from setiastro.saspro import quick_stack_worker as qs

F = qs.FrameInfo


@pytest.mark.parametrize("name,key", [
    ("Ha", "ha"), ("H-alpha", "ha"), ("Ha 7nm", "ha"), ("Antlia 3nm Ha Pro", "ha"), ("H", "ha"),
    ("O3", "oiii"), ("O-III", "oiii"), ("Baader O-III 6.5nm", "oiii"),
    ("S2", "sii"), ("S-II", "sii"), ("Lum", "l"), ("Red", "r"), ("Astronomik Blue Type 2c", "b"),
    ("Unknown", ""), ("", ""),
])
def test_filter_names_written_different_ways_match(name, key):
    assert qs.filter_key(name) == key


@pytest.mark.parametrize("name", ["L-eXtreme", "Optolong L-eNhance", "Ha+OIII", "UV/IR Cut"])
def test_a_letter_inside_a_longer_name_is_not_a_filter(name):
    assert qs.filter_key(name) not in ("l", "ha", "oiii", "")


@pytest.mark.parametrize("path,key", [
    (r"D:\x\MasterFlat_Ha.fit", "ha"),
    (r"D:\x\masterFlat_BIN-1_6248x4176_FILTER-R_mono.xisf", "r"),
    (r"D:\Flats\OIII\master.fit", "oiii"),
    (r"D:\x\MasterFlat_2025-11-20.fit", ""),
])
def test_filter_from_a_masters_name(path, key):
    assert qs.filter_from_name(path) == key


def _flat(path, filt="Unknown", w=64, h=48):
    return F(path=path, filter=filt, width=w, height=h)


def _light(filt, exp=300.0, w=64, h=48, gain=0.0):
    return F(path=f"light_{filt}.fits", filter=filt, exposure=exp, width=w, height=h, gain=gain)


def test_each_group_gets_its_own_flat():
    flats = [_flat("a.fit", "H-alpha"), _flat("MasterFlat_O3.fit"), _flat("c.fit", "SII")]
    got = {f: qs.plan_masters(_light(f), [], flats)["flat"] for f in ("Ha", "OIII", "SII", "R")}
    assert (got["Ha"].master.path, got["Ha"].state) == ("a.fit", "name")
    assert (got["OIII"].master.path, got["OIII"].state) == ("MasterFlat_O3.fit", "file")
    assert (got["SII"].master.path, got["SII"].state) == ("c.fit", "auto")
    assert got["R"].master is None and (got["R"].state, got["R"].why) == ("none", "filter")


def test_one_flat_still_serves_every_group_and_says_when_its_filter_differs():
    p = qs.plan_masters(_light("OIII"), [], [_flat("f.fit", "Ha")])["flat"]
    assert p.state == "only" and p.master.path == "f.fit"
    assert any("for Ha, the frames are OIII" in n for n in p.notes)


def test_a_choice_made_in_the_window_wins():
    flats = [_flat("ha.fit", "Ha"), _flat("o3.fit", "OIII")]
    darks = [F(path="d300.fit", exposure=300.0, width=64, height=48)]
    p = qs.plan_masters(_light("Ha"), darks, flats, {"flat": "o3.fit", "dark": None})
    assert (p["flat"].master.path, p["flat"].state) == ("o3.fit", "hand")
    assert any("for OIII, the frames are Ha" in n for n in p["flat"].notes)
    assert p["dark"].master is None and p["dark"].state == "hand-none"
    choice, notes = qs.pick_masters(_light("Ha"), darks, flats, {"flat": "o3.fit", "dark": None})
    assert (choice.flat, choice.dark) == ("o3.fit", None)
    assert any("chosen in the Quick Stack window" in n for n in notes)


def test_a_choice_whose_master_was_removed_falls_back_to_auto():
    p = qs.plan_masters(_light("Ha"), [], [_flat("ha.fit", "Ha")], {"flat": "gone.fit"})["flat"]
    assert (p.master.path, p.state) == ("ha.fit", "auto")


def test_a_flat_of_another_size_is_never_picked_automatically():
    p = qs.plan_masters(_light("Ha"), [], [_flat("big.fit", "Ha", 128, 96)])["flat"]
    assert p.master is None and p.why == "size"


# ── what an added file is ─────────────────────────────────────────────────────
def _write(path, data, **cards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = fits.Header()
    for k, v in cards.items():
        hdr[k] = v
    fits.writeto(path, data, hdr, overwrite=True)
    return path


def test_a_drop_sorts_lights_masters_and_leftovers(tmp_path):
    i16 = np.ones((48, 64), np.uint16)
    f32 = np.ones((48, 64), np.float32)
    root = str(tmp_path)
    files = {
        _write(os.path.join(root, "Lights", "Ha", "m31_001.fits"), i16, IMAGETYP="LIGHT", FILTER="Ha"): "light",
        _write(os.path.join(root, "Lights", "Ha", "m31_002.fits"), i16): "light",
        _write(os.path.join(root, "cal", "x.fits"), f32, IMAGETYP="FLAT", FILTER="Ha"): "flat",
        _write(os.path.join(root, "cal", "MasterDark_300s.fits"), f32): "dark",
        _write(os.path.join(root, "Flats", "flat_001.fits"), i16, IMAGETYP="Flat Frame"): "flat-frames",
        _write(os.path.join(root, "Darks 300s", "d_001.fits"), i16): "dark-frames",
        _write(os.path.join(root, "x", "b.fits"), i16, IMAGETYP="BIAS"): "bias",
        _write(os.path.join(root, "x", "df.fits"), i16, IMAGETYP="Dark Flat"): "darkflat",
        _write(os.path.join(root, "Dark Nebula", "img_001.fits"), i16): "light",
        _write(os.path.join(root, "x", "stack.fits"), f32, IMAGETYP="MASTER STACK"): "light",
    }
    for path, want in files.items():
        assert qs.route_file(qs.probe_frame(path)) == want, path


def test_add_masters_trusts_the_user_and_asks_only_when_nothing_says(tmp_path):
    f32 = np.ones((48, 64), np.float32)
    flat = _write(str(tmp_path / "Flats" / "flat_001.fits"), np.ones((48, 64), np.uint16))
    bare = _write(str(tmp_path / "m" / "thing.fits"), f32)
    assert qs.route_file(qs.probe_frame(flat), as_master=True) == "flat"
    assert qs.route_file(qs.probe_frame(bare), as_master=True) == "unknown"
