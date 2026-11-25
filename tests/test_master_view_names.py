"""Opening a run's masters must not look like a second set of masters.

A user stacked R, G and B with the Stacking Suite and found R, G, B, R_1, G_1
and B_1 open. Three things give a master a `_1`:
- a master and its auto-cropped copy got the same window name (OBJECT_FILTER);
- Open Masters opened a master that was already open a second time;
- a re-run saved MasterLight_..._1.fit beside the old master, without saying so.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from setiastro.saspro import doc_manager as dmod  # noqa: E402
from setiastro.saspro.stacking_suite import StackingSuiteDialog as SSD  # noqa: E402


@pytest.mark.parametrize("filt", ["R", "Ha"])
def test_a_master_and_its_copies_open_under_different_names(filt):
    hdr = {"OBJECT": "M31", "FILTER": filt}
    g = f"M31_{filt}_120s_(4144x2822)"
    names = [dmod._derive_display_name_from_header(hdr, path=f"MasterLight_{g}_{tail}.fit")
             for tail in ("40stacked", "40stacked_autocrop", "40stacked_drizzle",
                          "40stacked_drizzle_autocrop")]
    assert names == [f"M31_{filt}", f"M31_{filt}_crop", f"M31_{filt}_drizzle",
                     f"M31_{filt}_drizzle_crop"]


class _Suite:
    """The Suite methods under test, on a stand-in (the dialog itself is heavy)."""
    _path_key = staticmethod(SSD._path_key)
    _dedupe_out_path = SSD._dedupe_out_path
    _saved_masters_note = SSD._saved_masters_note
    _open_view_of = SSD._open_view_of
    _open_saved_masters = SSD._open_saved_masters

    def __init__(self, dm=None, mw=None):
        self.status, self._dm, self._mw = [], dm, mw

    def tr(self, s):
        return s

    def update_status(self, msg):
        self.status.append(msg)

    def _saspro_doc_manager(self):
        return self._dm

    def _saspro_main_window(self):
        return self._mw


def _touch(p):
    open(p, "wb").close()
    return str(p)


def test_a_rerun_master_is_marked_and_explained(tmp_path):
    s = _Suite()
    old = _touch(tmp_path / "MasterLight_M31_R.fit")
    new = _touch(s._dedupe_out_path(old))
    fresh = _touch(s._dedupe_out_path(str(tmp_path / "MasterLight_M31_G.fit")))
    assert os.path.basename(new) == "MasterLight_M31_R_1.fit"
    assert os.path.basename(fresh) == "MasterLight_M31_G.fit"
    lines = s._saved_masters_note([new, fresh]).splitlines()
    assert lines[0] == f"Saved in {tmp_path}:"
    assert "   MasterLight_M31_R_1.fit  *" in lines and "   MasterLight_M31_G.fit" in lines
    assert lines[-1].startswith("* Saved with a number on the end")


def test_no_footnote_when_nothing_was_renamed(tmp_path):
    s = _Suite()
    p = _touch(s._dedupe_out_path(str(tmp_path / "MasterLight_M31_R.fit")))
    note = s._saved_masters_note([p])
    assert "*" not in note and note.endswith("MasterLight_M31_R.fit")


def test_a_long_list_is_cut_short(tmp_path):
    note = _Suite()._saved_masters_note([str(tmp_path / f"m{i}.fit") for i in range(15)])
    assert note.count(".fit") == 12 and note.endswith("… and 3 more")


class _Sub:
    def __init__(self):
        self.raised = 0

    def isMinimized(self):
        return False

    def raise_(self):
        self.raised += 1


def _app(docs, views):
    """A DocManager and main window that know `docs`, `views` {doc: subwindow}."""
    opened = []
    active = []
    dm = SimpleNamespace(all_documents=lambda: list(docs), open_path=opened.append)
    mw = SimpleNamespace(_find_subwindow_for_doc=lambda d: views.get(id(d)),
                         mdi=SimpleNamespace(setActiveSubWindow=active.append),
                         raise_=lambda: None, activateWindow=lambda: None)
    return dm, mw, opened, active


def _doc(path):
    return SimpleNamespace(metadata={"file_path": path})


def test_open_masters_opens_only_what_is_not_open(tmp_path):
    a, b, c = (str(tmp_path / f"MasterLight_{k}.fit") for k in "RGB")
    da = _doc(a.upper().replace("\\", "/"))          # the same file, spelled differently
    db = _doc(b)                                     # listed, but its view was closed
    sw = _Sub()
    dm, mw, opened, active = _app([da, db], {id(da): sw})
    s = _Suite(dm, mw)
    s._open_saved_masters([a, b, c])
    if os.name == "nt":                              # paths compare case-blind on Windows only
        assert opened == [b, c]
        assert s.status == ["🖼 Opened 2 master(s) in SASpro; 1 already open, left as they were."]
    assert active == []                              # new views show themselves


def test_open_masters_twice_brings_them_forward(tmp_path):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # QTimer needs one
    assert app is not None
    a, b = (str(tmp_path / f"MasterLight_{k}.fit") for k in "RG")
    da, db = _doc(a), _doc(b)
    swa, swb = _Sub(), _Sub()
    dm, mw, opened, active = _app([da, db], {id(da): swa, id(db): swb})
    s = _Suite(dm, mw)
    s._open_saved_masters([a, b])
    assert opened == []
    assert active == [swb] and swb.raised == 1
    assert "not opened again" in s.status[-1]
