"""Essentials / All Tools toolbar modes.

Essentials shows the tools of the beginner workflows (plus Cosmic Clarity,
SyQon and RC-Astro) on toolbars of their own, holding the same QActions as
the full toolbars. These tests pin the three things that keep the two sets
apart: Essentials covers every tool the canned workflows use, a button never
moves between the sets, and Workflow Assistant files each tool under its full
toolbar, never under Essentials.
"""
from __future__ import annotations

import os
import re

import pytest

from setiastro.saspro.gui.mixins.toolbar_mixin import ToolbarMixin


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _command_ids_of_attrs():
    """attr -> its command ids (some have aliases), from the reg("cid", self.attr)
    table in toolbar_mixin."""
    import setiastro.saspro.gui.mixins.toolbar_mixin as tm
    src = open(os.path.abspath(tm.__file__), encoding="utf-8").read()
    out = {}
    for cid, attr in re.findall(r'reg\("([\w-]+)",\s*self\.(\w+)\)', src):
        out.setdefault(attr, set()).add(cid)
    return out


def test_essentials_holds_every_tool_of_the_canned_workflows():
    from setiastro.saspro.workflows import _default_canned_workflows

    cid_of = _command_ids_of_attrs()
    attrs = [attr for _n, _k, _t, items in ToolbarMixin._ESSENTIALS for attr, _l in items]
    missing = [a for a in attrs if a not in cid_of]
    assert not missing, f"Essentials entries with no command id: {missing}"
    ess = set().union(*(cid_of[a] for a in attrs))
    beginner = [wf for wf in _default_canned_workflows() if wf.name.startswith("Beginner")]
    assert len(beginner) == 3
    used = {s.command_id for wf in beginner for s in wf.steps if s.command_id}
    assert used <= ess, f"canned workflow tools missing from Essentials: {sorted(used - ess)}"
    assert {"cosmic_clarity", "syqontools", "rcastro"} <= ess


def test_essentials_rows_follow_the_workflow():
    rows = {name: [attr for attr, _l in items] for name, _k, _t, items in ToolbarMixin._ESSENTIALS}
    assert rows["EssentialsLinear"][0] == "act_crop"
    assert rows["EssentialsLinear"][-1] == "act_remove_stars"
    assert rows["EssentialsFinish"][0] == "act_stat_stretch"
    assert rows["EssentialsFinish"][-1] == "act_add_stars"
    placed = {n for row in ToolbarMixin._TOOLBAR_ROWS for n in row}
    assert {"ToolbarMode", "EssentialsLinear", "EssentialsFinish"} <= placed


def test_buttons_never_cross_between_essentials_and_full(qapp):
    from PyQt6.QtWidgets import QMainWindow
    from setiastro.saspro.shortcuts import DraggableToolBar, EssentialsToolBar

    mw = QMainWindow()
    a, b = DraggableToolBar("A", mw), DraggableToolBar("B", mw)
    e, e2 = EssentialsToolBar("E", mw), EssentialsToolBar("E2", mw)
    assert a._accepts_reorder_from(b)                 # full bars trade buttons
    assert not a._accepts_reorder_from(e)             # never from Essentials
    assert not e._accepts_reorder_from(a)             # never into Essentials
    assert not e._accepts_reorder_from(e2)            # nor between Essentials bars
    assert e._accepts_reorder_from(e)                 # reorder along its own bar
    mw.deleteLater()


def test_workflow_catalog_skips_essentials(qapp):
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QMainWindow
    from setiastro.saspro.shortcuts import DraggableToolBar, EssentialsToolBar
    from setiastro.saspro.workflows import _discover_toolbar_categories

    mw = QMainWindow()
    act = QAction("Crop...", mw)
    act.setProperty("command_id", "crop")
    full = DraggableToolBar("Functions", mw)
    full.addAction(act)
    ess = EssentialsToolBar("Essentials: Linear", mw)
    ess.addAction(act)
    cats = _discover_toolbar_categories(mw)
    assert not any("Essentials" in c for c in cats)
    assert [i.command_id for i in cats["Functions"]] == ["crop"]
    mw.deleteLater()

class _Idx:
    def __init__(self, ok):
        self.ok = ok

    def isValid(self):
        return self.ok


class _Popup:
    def __init__(self, visible, highlighted):
        self.visible, self.highlighted = visible, highlighted

    def isVisible(self):
        return self.visible

    def currentIndex(self):
        return _Idx(self.highlighted)

    def model(self):
        return None


class _Completer:
    def __init__(self, popup):
        self._popup = popup

    def popup(self):
        return self._popup

    def completionModel(self):
        return None


class _Edit:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def test_enter_on_a_highlighted_match_runs_it_once(qapp):
    """Enter on a highlighted match reaches the search box first (returnPressed)
    and the completer after it (activated). Only the completer may run it:
    both did, and the tool opened twice -- in the Essentials box and the panel's."""
    from types import SimpleNamespace
    from PyQt6.QtGui import QStandardItem, QStandardItemModel
    from setiastro.saspro.gui.main_window import AstroSuiteProMainWindow as MW

    model = QStandardItemModel()
    model.appendRow(QStandardItem("Pixel Math"))
    ran = []
    mw = SimpleNamespace(_cmd_model=model, _cmd_edit=None, _cmd_completer=None,
                         _run_selected_completion=lambda idx: ran.append(idx.data()))
    edit = _Edit("Pixel Math")

    MW._trigger_first_visible_completion(mw, edit, _Completer(_Popup(True, True)))
    assert ran == []                                  # the completer's activated runs it

    MW._trigger_first_visible_completion(mw, edit, _Completer(_Popup(True, False)))
    assert ran == ["Pixel Math"]                      # nothing highlighted: Enter runs the name
