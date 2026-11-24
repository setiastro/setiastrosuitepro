from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _canned(name):
    from setiastro.saspro.workflows import _default_canned_workflows
    return next(wf for wf in _default_canned_workflows() if wf.name == name)


def _recipe(wf):
    """{lane: [step, ...]}; a Cosmic Clarity step reads cosmic_clarity:<mode>."""
    out = {lane: [] for lane in wf.lanes}
    for s in wf.steps:
        name = s.command_id or s.kind
        out[s.lane].append(f"{name}:{s.preset['mode']}" if "mode" in s.preset else name)
    return out


def _link_names(wf):
    """Sorted (from, to lane or step) of every link, with steps as lane/command."""
    from setiastro.saspro.workflows import workflow_links

    ids = {s.id: f"{s.lane}/{s.command_id or s.kind}" for s in wf.steps}
    return sorted((ids[l.src], l.dst_lane or ids[l.dst]) for l in workflow_links(wf))


# -----------------------------------------------------------------------------
# Canned workflows
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("name, first", [
    ("Beginner OSC Workflow", ["crop"]),
    ("Beginner Mono RGB Workflow", ["crop", "note", "rgb_combine"]),
])
def test_canned_color_splits_after_star_removal_and_ends_in_add_stars(name, first):
    wf = _canned(name)
    assert _recipe(wf) == {
        "Main": first + ["abe", "background_neutral", "white_balance", "cosmic_clarity:sharpen",
                         "cosmic_clarity:denoise", "remove_stars", "split"],
        "Starless": ["stat_stretch", "curves", "selective_color", "texture_clarity"],
        "Stars Only": ["star_stretch"],
        "Combine": ["add_stars"],
    }
    assert _link_names(wf) == [
        ("Main/split", "Starless"),
        ("Main/split", "Stars Only"),
        ("Starless/texture_clarity", "Combine"),
        ("Stars Only/star_stretch", "Combine"),
    ]


def test_mono_rgb_warns_against_adding_l_while_linear():
    wf = _canned("Beginner Mono RGB Workflow")
    main = [s for s in wf.steps if s.lane == "Main"]
    note, combine = main[1], main[2]
    assert note.kind == "note" and combine.command_id == "rgb_combine"
    assert note.note.startswith("Do not add L now")      # the part the graph's note box shows
    assert "Statistical Stretch" in note.note and "Luminance left empty" in combine.note


def test_canned_narrowband_preps_each_filter_then_joins_starless_and_stars():
    wf = _canned("Beginner Mono Narrowband Workflow")
    prep = ["crop", "abe", "cosmic_clarity:sharpen", "cosmic_clarity:denoise", "remove_stars"]
    assert wf.lanes == ["Ha", "OIII", "SII", "Starless", "Stars Only", "Combine"]
    assert _recipe(wf) == {
        "Ha": prep, "OIII": prep, "SII": prep,
        "Starless": ["ppp", "curves", "selective_color", "texture_clarity"],
        "Stars Only": ["nbtorgb"],
        "Combine": ["add_stars"],
    }
    assert _link_names(wf) == sorted(
        [(f"{f}/remove_stars", lane) for f in ("Ha", "OIII", "SII") for lane in ("Starless", "Stars Only")]
        + [("Starless/texture_clarity", "Combine"), ("Stars Only/nbtorgb", "Combine")]
    )


def test_every_canned_step_names_a_registered_command():
    """A command id the toolbar does not register opens nothing and shows as a raw id."""
    import re
    from pathlib import Path

    import setiastro.saspro
    from setiastro.saspro.workflows import _default_canned_workflows

    src = Path(setiastro.saspro.__file__).parent / "gui" / "mixins" / "toolbar_mixin.py"
    registered = set(re.findall(r'\breg\("([a-z0-9_]+)"', src.read_text(encoding="utf-8")))
    assert {"crop", "cosmic_clarity"} <= registered
    used = {s.command_id for wf in _default_canned_workflows() for s in wf.steps if s.kind == "action"}
    assert used <= registered
    assert "cosmicclarity" not in used      # the action itself carries cosmic_clarity


def test_catalog_finds_a_tool_by_its_other_id(qapp):
    from PyQt6.QtGui import QAction, QIcon

    from setiastro.saspro.workflows import WorkflowActionInfo, WorkflowCatalog, WorkflowStep, format_workflow_step_label

    info = WorkflowActionInfo("cosmic_clarity", "Cosmic Clarity UI", "", QIcon(), "Cosmic Clarity", QAction())
    cat = WorkflowCatalog({"cosmic_clarity": info})
    assert cat.get("cosmicclarity") is info
    assert cat.get("nope") is None and cat.get("nope", 5) == 5
    assert format_workflow_step_label(WorkflowStep(command_id="cosmicclarity"), cat) == "Cosmic Clarity UI"


def test_tool_names_lose_the_menu_dots():
    from setiastro.saspro.workflows import _clean_action_text

    assert _clean_action_text("&Crop...") == "Crop"
    assert _clean_action_text("Statistical Stretch…") == "Statistical Stretch"
    assert _clean_action_text("FX") == "FX"
    assert _clean_action_text("") == ""


# -----------------------------------------------------------------------------
# Links: derivation from I/O tags
# -----------------------------------------------------------------------------

def test_only_notes_before_the_consumer_means_the_link_enters_the_lane():
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep, workflow_links

    wf = WorkflowDefinition(
        name="nb", lanes=["main", "stars"],
        steps=[WorkflowStep(id="rs", command_id="remove_stars", lane="main", outputs=["stars_only"]),
               WorkflowStep(id="n", kind="note", lane="stars"),
               WorkflowStep(id="nb", command_id="nbtorgb", lane="stars", inputs=["stars_only"])])
    assert [(l.src, l.dst, l.dst_lane) for l in workflow_links(wf)] == [("rs", "", "stars")]


def test_join_part_way_down_a_lane():
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep, workflow_links

    split = WorkflowStep(id="split", kind="split", lane="main", outputs=["rgb", "lum"])
    merge = WorkflowStep(id="merge", kind="merge", lane="main", inputs=["rgb_done", "lum_done"])
    wf = WorkflowDefinition(
        name="planet",
        lanes=["main", "RGB", "L"],
        steps=[
            WorkflowStep(id="crop", command_id="crop", lane="main", outputs=["c"]),
            split,
            WorkflowStep(id="wb", command_id="white_balance", lane="RGB", inputs=["rgb"], outputs=["rgb_done"]),
            WorkflowStep(id="decon", command_id="convo", lane="L", inputs=["lum"], outputs=["lum_done"]),
            merge,
            WorkflowStep(id="save", command_id="save_as", lane="main", inputs=["final"]),
        ],
    )
    got = {(l.src, l.dst, l.dst_lane) for l in workflow_links(wf)}
    assert got == {
        ("split", "", "RGB"),
        ("split", "", "L"),
        ("wb", "merge", ""),
        ("decon", "merge", ""),
    }


def test_tag_made_earlier_in_the_same_lane_does_not_reach_out():
    from setiastro.saspro.workflows import WorkflowStep, derive_workflow_links

    steps = [
        WorkflowStep(id="a", command_id="crop", lane="main", outputs=["x"]),
        WorkflowStep(id="b", command_id="curves", lane="main", inputs=["x"]),
        WorkflowStep(id="c", command_id="abe", lane="side", outputs=["x"]),
    ]
    assert derive_workflow_links(steps, ["main", "side"]) == []


# -----------------------------------------------------------------------------
# Links: normalization
# -----------------------------------------------------------------------------

def _chain():
    from setiastro.saspro.workflows import WorkflowStep
    return [
        WorkflowStep(id="m1", command_id="crop", lane="main"),
        WorkflowStep(id="m2", command_id="abe", lane="main"),
        WorkflowStep(id="m3", command_id="curves", lane="main"),
        WorkflowStep(id="s1", command_id="remove_stars", lane="side"),
        WorkflowStep(id="s2", command_id="clahe", lane="side"),
    ]


def test_normalize_drops_bad_links_with_reasons():
    from setiastro.saspro.workflows import WorkflowLink, normalize_workflow_links

    steps = _chain()
    lanes = ["main", "side", "empty"]
    links = [
        WorkflowLink(src="m1", dst_lane="side"),     # keep: branch
        WorkflowLink(src="s2", dst="m3"),            # keep: join
        WorkflowLink(src="m1", dst="m2"),            # same lane
        WorkflowLink(src="gone", dst_lane="side"),   # missing source
        WorkflowLink(src="m1", dst_lane="nowhere"),  # missing lane
        WorkflowLink(src="m1", dst_lane="side"),     # duplicate
        WorkflowLink(src="m3", dst_lane="side"),     # m3 -> s1 -> s2 -> m3 loops
        WorkflowLink(src="m2", dst_lane="empty"),    # keep: empty lane is fine
    ]
    dropped = []
    out = normalize_workflow_links(links, steps, lanes, dropped)
    assert [(l.src, l.dst, l.dst_lane) for l in out] == [
        ("m1", "", "side"), ("s2", "m3", ""), ("m2", "", "empty"),
    ]
    assert [r for _l, r in dropped] == ["same_lane", "missing", "missing", "duplicate", "loop"]


def test_links_on_choice_actions_move_to_the_choice():
    from setiastro.saspro.workflows import WorkflowLink, WorkflowStep, normalize_workflow_links

    steps = [
        WorkflowStep(id="t", kind="choice", label_override="Stretch", lane="main"),
        WorkflowStep(id="c1", command_id="curves", parent_id="t", lane="main"),
        WorkflowStep(id="x", command_id="crop", lane="side"),
    ]
    out = normalize_workflow_links([WorkflowLink(src="c1", dst_lane="side")], steps, ["main", "side"])
    assert [(l.src, l.dst_lane) for l in out] == [("t", "side")]


def test_links_round_trip_and_explicit_empty_list_is_kept():
    from setiastro.saspro.workflows import WorkflowDefinition

    wf = _canned("Beginner Mono Narrowband Workflow")
    data = wf.to_dict()
    assert data["schema_version"] == 3
    assert len(data["links"]) == 8
    again = WorkflowDefinition.from_dict(data)
    assert [l.to_dict() for l in again.links] == data["links"]

    data["links"] = []
    assert WorkflowDefinition.from_dict(data).links == []   # user removed them all: stay removed

    del data["links"]                                       # pre-links file: derive once
    assert len(WorkflowDefinition.from_dict(data).links) == 8


def test_link_from_dict_rejects_junk():
    from setiastro.saspro.workflows import WorkflowLink

    assert WorkflowLink.from_dict({"from": "a", "to": "b"}).dst == "b"
    assert WorkflowLink.from_dict({"from": "a", "to_lane": "stars"}).dst_lane == "stars"
    assert WorkflowLink.from_dict({"from": "a", "to": "b", "to_lane": "x"}).dst_lane == ""
    for junk in ({"from": "a"}, {"to": "b"}, {"from": "", "to": "b"}, "nope", None):
        assert WorkflowLink.from_dict(junk) is None


# -----------------------------------------------------------------------------
# Graph structure and placement
# -----------------------------------------------------------------------------

def _placed(wf):
    from setiastro.saspro.workflow_graph import build_workflow_graph, place_workflow_graph
    from setiastro.saspro.workflows import ensure_workflow_links

    ensure_workflow_links(wf)
    g = build_workflow_graph(wf)
    sizes = {k: (180.0, 100.0 if n.kind == "choice" else 36.0) for k, n in g.nodes.items()}
    return place_workflow_graph(g, sizes, reserve_new_lane=True)


def test_branches_start_below_their_fork_and_merges_below_their_feeders():
    wf = _canned("Beginner Mono Narrowband Workflow")
    g = _placed(wf)
    star_removal = [g.nodes[s.id] for s in wf.steps if s.command_id == "remove_stars"]
    assert [n.rank for n in star_removal] == [4, 4, 4]
    assert g.nodes[g.order["Starless"][0]].rank == 5
    assert g.nodes[g.order["Stars Only"][0]].rank == 5
    feeders = [g.nodes[g.order["Starless"][-1]], g.nodes[g.order["Stars Only"][-1]]]
    assert g.nodes[g.order["Combine"][0]].rank == max(n.rank for n in feeders) + 1
    links = [e for e in g.edges if e.link is not None]
    assert sorted(e.label for e in links) == sorted(
        [f"{f} {half}" for f in ("Ha", "OIII", "SII") for half in ("starless", "stars")]
        + ["finished starless", "RGB stars"])


def test_the_gap_grows_with_the_links_that_cross_it():
    from setiastro.saspro.workflow_graph import LINK_GAP, LINK_SPREAD, ROW_GAP

    g = _placed(_canned("Beginner Mono Narrowband Workflow"))
    assert g.row_gap[0] == ROW_GAP                              # lane order only
    assert g.row_gap[4] == LINK_GAP + 5 * LINK_SPREAD           # six links leave the star removals
    assert g.row_gap[8] == LINK_GAP                             # one link drops into Add Stars


def test_lane_steps_never_overlap_and_lanes_never_overlap():
    wf = _canned("Beginner Mono Narrowband Workflow")
    g = _placed(wf)
    for lane, keys in g.order.items():
        nodes = [g.nodes[k] for k in keys]
        for a, b in zip(nodes, nodes[1:]):
            assert a.y + a.h < b.y
    spans = [g.columns[lane] for lane in g.lanes]
    for (x1, w1), (x2, _w2) in zip(spans, spans[1:]):
        assert x1 + w1 < x2
    assert g.new_lane_x >= spans[-1][0] + spans[-1][1]
    for n in g.nodes.values():
        assert 0 <= n.x and n.x + n.w <= g.width
        assert 0 <= n.y and n.y + n.h <= g.height


def test_empty_lane_gets_a_slot_that_links_can_target():
    from setiastro.saspro.workflow_graph import slot_key
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowLink, WorkflowStep

    wf = WorkflowDefinition(
        name="w", lanes=["main", "stars"],
        steps=[WorkflowStep(id="a", command_id="crop", lane="main"),
               WorkflowStep(id="b", command_id="remove_stars", lane="main")],
        links=[WorkflowLink(src="b", dst_lane="stars")],
    )
    g = _placed(wf)
    slot = g.nodes[slot_key("stars")]
    assert slot.is_slot and slot.rank == g.nodes["b"].rank + 1
    assert any(e.dst == slot.key and e.link is not None for e in g.edges)


def test_choice_children_ride_inside_their_card():
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    wf = WorkflowDefinition(
        name="w", lanes=["main"],
        steps=[WorkflowStep(id="t", kind="choice", label_override="Stretch", lane="main"),
               WorkflowStep(id="c1", command_id="curves", parent_id="t", lane="main"),
               WorkflowStep(id="c2", command_id="clahe", parent_id="t", lane="main"),
               WorkflowStep(id="after", command_id="save_as", lane="main")],
        links=[],
    )
    g = _placed(wf)
    assert g.order["main"] == ["t", "after"]
    assert [c.id for c in g.nodes["t"].children] == ["c1", "c2"]


def test_tool_payloads_reads_catalog_drags(qapp):
    from PyQt6.QtCore import QMimeData

    from setiastro.saspro.workflow_graph import has_tool_payload, tool_payloads
    from setiastro.saspro.workflows import WORKFLOW_MIME

    md = QMimeData()
    assert not has_tool_payload(md)
    md.setData(WORKFLOW_MIME, b"curves")
    assert has_tool_payload(md)
    assert tool_payloads(md) == [{"command_id": "curves", "preset": {}, "name": ""}]


# -----------------------------------------------------------------------------
# Editor commands behind the graph
# -----------------------------------------------------------------------------

@pytest.fixture
def editor(qapp, tmp_path):
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QMainWindow, QToolBar

    from setiastro.saspro.workflows import WorkflowDialog

    main = QMainWindow()
    main.get_user_data_dir = lambda: str(tmp_path)   # keep autosave out of the real profile
    bar = QToolBar("Functions", main)
    for cid in ("crop", "abe", "curves", "clahe", "remove_stars", "add_stars", "save_as"):
        act = QAction(cid.replace("_", " ").title(), main)
        act.setProperty("command_id", cid)
        bar.addAction(act)
    main.addToolBar(bar)
    dlg = WorkflowDialog(main)
    dlg.show()
    for _ in range(4):
        qapp.processEvents()
    yield dlg
    dlg.hide()
    dlg.deleteLater()
    main.deleteLater()


def _steps(dlg, lane):
    dlg.sync_steps_from_ui()
    return [s for s in dlg.current_workflow.steps if s.lane == lane and not s.parent_id]


def test_editor_insert_move_link_and_rename(editor, monkeypatch, qapp):
    from setiastro.saspro.workflows import WorkflowDefinition

    editor.load_workflow_definition(WorkflowDefinition(name="t", lanes=["main", "stars"], links=[]))
    editor.graph_insert([{"command_id": "crop"}, {"command_id": "remove_stars"}], "main", 0)
    editor.graph_insert([{"command_id": "curves"}], "main", 1)
    assert [s.command_id for s in _steps(editor, "main")] == ["crop", "curves", "remove_stars"]

    rs = _steps(editor, "main")[2]
    assert editor.graph_link(rs.id, dst_lane="stars") == ""
    assert "already" in editor.graph_link(rs.id, dst_lane="stars")
    assert "same lane" in editor.graph_link(rs.id, dst=_steps(editor, "main")[0].id)

    editor.graph_insert([{"command_id": "clahe"}], "stars", 0)
    clahe = _steps(editor, "stars")[0]
    curves = _steps(editor, "main")[1]
    assert "loop" in editor.graph_link(clahe.id, dst=curves.id)   # curves comes before remove_stars

    # Move curves to the end of main: the join from clahe now makes sense.
    editor.graph_move(curves.id, "main", 2)
    assert [s.command_id for s in _steps(editor, "main")] == ["crop", "remove_stars", "curves"]
    assert editor.graph_link(clahe.id, dst=curves.id) == ""

    # Renaming the target lane keeps the branch pointed at it.
    monkeypatch.setattr("setiastro.saspro.workflows.QInputDialog.getText",
                        lambda *a, **k: ("stars only", True))
    editor.graph_rename_lane("stars")
    assert [(l.src, l.dst_lane) for l in editor.current_workflow.links if l.dst_lane] == [(rs.id, "stars only")]

    # Moving clahe into main makes its join same-lane, so the link goes away.
    editor.graph_move(clahe.id, "main", 3)
    assert all(l.src != clahe.id for l in editor.current_workflow.links)

    saved = editor.current_workflow.to_dict()
    assert {"from": rs.id, "to_lane": "stars only"} in saved["links"]


def test_editor_branch_into_new_lane_and_select_from_graph(editor, monkeypatch, qapp):
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    editor.load_workflow_definition(WorkflowDefinition(
        name="t", lanes=["main"], links=[],
        steps=[WorkflowStep(id="a", command_id="crop", lane="main"),
               WorkflowStep(id="b", command_id="remove_stars", lane="main"),
               WorkflowStep(id="c", command_id="save_as", lane="main")],
    ))
    monkeypatch.setattr("setiastro.saspro.workflows.QInputDialog.getText",
                        lambda *a, **k: ("stars", True))
    editor.graph_branch_new_lane("b")
    assert editor.current_workflow.lanes == ["main", "stars"]
    assert [(l.src, l.dst_lane) for l in editor.current_workflow.links] == [("b", "stars")]

    assert editor.graph_select("c")
    assert editor.graph_current_id() == "c"
    assert editor._current_list().lane_name == "main"

    editor._mark_selected_complete()
    assert editor.graph_completed_ids() == {"c"}

    # The scene rebuilds without errors and shows every step plus the empty lane's slot.
    qapp.processEvents()
    editor._graph.rebuild()
    keys = set(editor._graph.view._nodes)
    assert {"a", "b", "c", "lane:stars"} <= keys


# -----------------------------------------------------------------------------
# The two views
# -----------------------------------------------------------------------------

class _Settings:
    """The two QSettings calls the dialogs make. Holds strings, as the registry does."""

    def __init__(self, **values):
        self.values = {k.replace("__", "/"): v for k, v in values.items()}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


def _main_window(tmp_path, settings=None):
    from PyQt6.QtWidgets import QMainWindow

    main = QMainWindow()
    main.get_user_data_dir = lambda: str(tmp_path)
    main.settings = settings
    return main


def test_graph_is_the_default_view_and_the_list_is_remembered(qapp, tmp_path):
    from setiastro.saspro.workflows import MiniWorkflowDialog, WorkflowDialog

    settings = _Settings(workflow_dialog__graph_open="false")     # left by the old pop-out window
    main = _main_window(tmp_path, settings)

    first = WorkflowDialog(main)
    assert first.view_mode() == "graph"
    assert first.btn_view_graph.isChecked() and not first.btn_view_list.isChecked()
    assert "workflow_dialog/view_mode" not in settings.values     # only a choice is remembered
    first.btn_view_list.click()
    assert first.view_mode() == "list" and first.btn_view_list.isChecked()
    first.deleteLater()

    second = WorkflowDialog(main)
    assert second.view_mode() == "list"
    second._toggle_view_mode()                                     # Ctrl+G
    assert second.view_mode() == "graph"
    assert settings.values["workflow_dialog/view_mode"] == "graph"

    # The runner starts in the graph too, and keeps its own choice.
    wf = _canned("Beginner OSC Workflow")
    mini = MiniWorkflowDialog(main, workflow=wf, catalog=second.catalog)
    assert mini.view_mode() == "graph" and not mini._graph.view.editable
    mini.set_view_mode("list")
    assert settings.values["mini_workflow_dialog/view_mode"] == "list"
    assert settings.values["workflow_dialog/view_mode"] == "graph"
    for w in (second, mini):
        w.deleteLater()
    main.deleteLater()


def test_without_a_graph_the_list_is_the_only_view(qapp, tmp_path, monkeypatch):
    import setiastro.saspro.workflow_graph as wg
    from setiastro.saspro.workflows import WorkflowDialog

    def boom(*a, **k):
        raise RuntimeError("no graph")
    monkeypatch.setattr(wg, "WorkflowGraphPanel", boom)
    main = _main_window(tmp_path)
    dlg = WorkflowDialog(main)
    assert dlg.view_mode() == "list" and not dlg.btn_view_graph.isEnabled()
    dlg.set_view_mode("graph")
    assert dlg.view_mode() == "list"
    dlg.load_workflow_definition(_canned("Beginner OSC Workflow"))
    assert len(_steps(dlg, "Main")) == 8
    dlg.deleteLater()
    main.deleteLater()


def test_palette_details_and_step_buttons(editor, qapp):
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    editor.edt_search.setText("remove")
    tree = editor.tool_tree
    shown = [tree.topLevelItem(i).child(j).data(0, 0x0100)
             for i in range(tree.topLevelItemCount())
             for j in range(tree.topLevelItem(i).childCount())
             if not tree.topLevelItem(i).child(j).isHidden()]
    assert shown == ["remove_stars"]
    editor.edt_search.clear()

    editor.load_workflow_definition(WorkflowDefinition(
        name="View test", lanes=["main"], links=[],
        steps=[WorkflowStep(id="a", command_id="crop", lane="main"),
               WorkflowStep(id="b", command_id="abe", lane="main")],
    ))
    for _ in range(4):
        qapp.processEvents()
    view = editor._graph.view
    assert set(view._nodes) == {"a", "b"}

    assert editor.graph_select("b")
    assert "Abe" in editor.lbl_step_info.text()
    editor.btn_prev.click()
    assert editor.graph_current_id() == "a"
    editor.btn_next.click()
    assert editor.graph_current_id() == "b"
    editor.btn_done.click()
    assert "b" in editor.graph_completed_ids()
    assert view._nodes["b"].done and view._nodes["b"].current

    # Double-clicking a palette tool: below the selected step in the graph, at the end in the list.
    item = next(tree.topLevelItem(i).child(j)
                for i in range(tree.topLevelItemCount())
                for j in range(tree.topLevelItem(i).childCount())
                if tree.topLevelItem(i).child(j).data(0, 0x0100) == "curves")
    assert editor.graph_select("a")
    editor._on_tool_double_clicked(item)
    assert [s.command_id for s in _steps(editor, "main")] == ["crop", "curves", "abe"]
    editor.set_view_mode("list")
    assert editor.graph_select("a")
    editor.tool_tree.setCurrentItem(item)
    editor.btn_add_selected.click()
    assert [s.command_id for s in _steps(editor, "main")] == ["crop", "curves", "abe", "curves"]


def test_graph_rebuilds_only_while_it_is_the_view(editor, qapp):
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    panel = editor._graph
    assert panel.isVisible() and not editor.lanes_scroll.isVisible()
    editor.set_view_mode("list")
    assert not panel.isVisible() and editor.lanes_scroll.isVisible()

    calls = []
    real = panel.rebuild
    panel.rebuild = lambda: (calls.append(1), real())
    editor.load_workflow_definition(WorkflowDefinition(
        name="hidden", lanes=["main"], links=[],
        steps=[WorkflowStep(id="x", command_id="crop", lane="main")]))
    for _ in range(4):
        qapp.processEvents()
    assert calls == []                  # nothing to draw while the list is up

    editor.set_view_mode("graph")
    for _ in range(4):
        qapp.processEvents()
    assert calls and set(panel.view._nodes) == {"x"}


def test_loading_fits_the_new_graph_not_the_old_one(editor, qapp):
    editor.resize(1900, 1000)           # wide enough that no scrollbar comes and goes and refits it
    view = editor._graph.view

    def load(name):
        editor.load_workflow_definition(_canned(name))
        for _ in range(6):
            qapp.processEvents()
        return view._zoom

    wide = load("Beginner Mono Narrowband Workflow")
    assert wide < 0.95 and view.horizontalScrollBar().maximum() == 0
    assert load("Cosmic Clarity Cleanup") == pytest.approx(1.0)
    assert load("Beginner Mono Narrowband Workflow") == pytest.approx(wide)


def test_loading_starts_with_nothing_completed(editor):
    wf = _canned("Beginner OSC Workflow")
    editor.load_workflow_definition(wf)
    editor._mark_selected_complete()
    assert editor.graph_completed_ids()
    editor.load_workflow_definition(wf)            # the same workflow again, same step ids
    assert editor.graph_completed_ids() == set()


def test_esc_in_the_graph_never_closes_the_editor(editor, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    editor.load_workflow_definition(_canned("Beginner OSC Workflow"))
    for _ in range(4):
        qapp.processEvents()
    view = editor._graph.view
    view._select_link(next(e for e in view._edges if e.is_link))
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert editor.isVisible() and view._selected_link is None

    # From anywhere else it closes, and the session is saved as on a normal close.
    assert not editor._autosave_path().exists()
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    assert not editor.isVisible() and editor._autosave_path().exists()


def test_enter_in_a_text_field_presses_no_button(editor, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    editor.load_workflow_definition(_canned("Beginner OSC Workflow"))
    for field in (editor.edt_name, editor.edt_search):
        field.setFocus()
        QTest.keyClick(field, Qt.Key.Key_Return)
    assert len(editor.current_workflow.steps) == 14      # New would have emptied it
    assert editor.isVisible()


def test_link_labels_keep_clear_of_steps_and_each_other(editor, qapp):
    editor.resize(1500, 900)
    editor.load_workflow_definition(_canned("Beginner Mono Narrowband Workflow"))
    for _ in range(6):
        qapp.processEvents()
    view = editor._graph.view
    labels = [e.label_item.rect for e in view._edges if e.label_item is not None]
    assert len(labels) == 8
    for i, a in enumerate(labels):
        assert not any(a.intersects(b) for b in labels[i + 1:])
        assert not any(a.intersects(it.mapRectToScene(it.body())) for it in view._nodes.values())
    # A label is a handle on its link.
    link = next(e for e in view._edges if e.label_item is not None)
    kind, item, _row = view._hit(link.label_item.rect.center())
    assert kind == "link"


def test_hiding_the_tools_gives_the_graph_the_room(editor, qapp):
    editor.resize(1500, 900)
    qapp.processEvents()
    before = editor._graph.width()
    editor.btn_tools.click()
    for _ in range(3):
        qapp.processEvents()
    assert editor.palette_panel.isHidden() and editor.btn_tools.text() == "Show Tools"
    assert editor._graph.width() > before + 100
    editor.btn_tools.click()
    assert not editor.palette_panel.isHidden() and editor.btn_tools.text() == "Hide Tools"


def test_lane_buttons_ask_which_lane(editor, monkeypatch, qapp):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QLineEdit, QMenu, QMessageBox

    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    editor.load_workflow_definition(WorkflowDefinition(
        name="t", lanes=["main", "branch", "stars", "spare"], links=[],
        steps=[WorkflowStep(id="a", command_id="crop", lane="main"),
               WorkflowStep(id="s", command_id="add_stars", lane="stars")]))
    assert editor.graph_select("a")      # main is current; the empty lanes cannot be selected

    asked = []
    monkeypatch.setattr("setiastro.saspro.workflows.QMessageBox.question",
                        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.Yes)
    seen, errors = [], []

    def answer(lane, new_name=None):
        def go():
            dlg = next((w for w in QApplication.topLevelWidgets()
                        if isinstance(w, QDialog) and w.isVisible()
                        and w.windowTitle() in ("Remove Lane", "Rename Lane")), None)
            if dlg is None:
                errors.append("no lane dialog")
                return
            try:
                combo = dlg.findChild(QComboBox)
                seen.append([combo.itemText(i) for i in range(combo.count())])
                combo.setCurrentIndex(combo.findData(lane))
                if new_name is not None:
                    dlg.findChild(QLineEdit).setText(new_name)
                dlg.accept()
            except Exception as e:          # never leave the dialog blocking the test
                errors.append(repr(e))
                dlg.reject()
        QTimer.singleShot(0, go)

    answer("branch")
    editor.btn_remove_lane.click()
    assert errors == []
    assert [t.split()[0] for t in seen[0]] == ["main", "branch", "stars", "spare"]
    assert "(1 step)" in seen[0][0] and "(empty)" in seen[0][1]
    assert editor.current_workflow.lanes == ["main", "stars", "spare"]
    assert asked == []                   # an empty lane goes without a second question

    answer("stars", "stars only")
    editor.btn_rename_lane.click()
    assert errors == []
    assert editor.current_workflow.lanes == ["main", "stars only", "spare"]
    assert [s.lane for s in editor.current_workflow.steps if s.id == "s"] == ["stars only"]

    answer("stars only")
    editor.btn_remove_lane.click()
    assert errors == []
    assert len(asked) == 1 and "1 step" in asked[0]
    assert editor.current_workflow.lanes == ["main", "spare"]
    assert [s.id for s in editor.current_workflow.steps] == ["a"]

    # The graph's lane menu already knows its lane, so it removes that one directly.
    menu = QMenu()
    editor.graph_lane_menu(menu, "spare")
    next(a for a in menu.actions() if a.text() == "Remove Lane…").trigger()
    assert editor.current_workflow.lanes == ["main"]


# -----------------------------------------------------------------------------
# Steps dragged to a height of their own (graph_rows)
# -----------------------------------------------------------------------------

def _osc_like():
    """Frank's TestOSC shape: a long main lane, a short star branch, and a
    combine lane whose only step joins both, so it lands at the very bottom."""
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowLink, WorkflowStep

    main = [WorkflowStep(id=f"m{i}", command_id="curves", lane="main") for i in range(12)]
    stars = [WorkflowStep(id="s0", command_id="star_stretch", lane="stars_only"),
             WorkflowStep(id="s1", command_id="saturation", lane="stars_only")]
    combine = [WorkflowStep(id="add", command_id="add_stars", lane="combine"),
               WorkflowStep(id="save", command_id="save_as", lane="combine")]
    return WorkflowDefinition(
        name="osc", lanes=["main", "stars_only", "combine", "finish"],
        steps=main + stars + combine + [WorkflowStep(id="f0", command_id="crop", lane="finish")],
        links=[WorkflowLink(src="m3", dst_lane="stars_only"),
               WorkflowLink(src="m11", dst="add"),
               WorkflowLink(src="s1", dst="add"),
               WorkflowLink(src="add", dst_lane="finish")])


def test_pinned_step_rises_and_takes_its_downstream_with_it():
    from setiastro.saspro.workflow_graph import natural_rank

    wf = _osc_like()
    g = _placed(wf)
    assert g.nodes["add"].rank == 12                     # below every step that feeds it
    assert (g.nodes["save"].rank, g.nodes["f0"].rank) == (13, 13)

    wf.graph_rows = {"add": 2}
    g = _placed(wf)
    assert g.nodes["add"].rank == 2 and g.nodes["add"].pinned
    assert g.nodes["save"].rank == 3                     # the lane follows it up
    assert g.nodes["f0"].rank == 3                       # and so does a lane branching from it
    assert g.nodes["m11"].rank == 11                     # its feeders stay where they were
    assert natural_rank(wf, "add") == 12

    # A pin cannot lift a step above the step before it in its own lane.
    wf.graph_rows = {"save": 0}
    g = _placed(wf)
    assert g.nodes["save"].rank == g.nodes["add"].rank + 1

    # Rows left empty below a pin keep a step's height, so nothing collapses.
    wf.graph_rows = {"f0": 20}
    g = _placed(wf)
    assert g.nodes["f0"].rank == 20 and len(g.row_h) == 21 and min(g.row_h) > 0


def test_graph_rows_round_trip_and_prune():
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    wf = _osc_like()
    wf.steps.append(WorkflowStep(id="kid", command_id="clahe", lane="main", parent_id="m0"))
    wf.graph_rows = {"add": 2, "gone": 4, "kid": 1}
    data = wf.to_dict()
    assert data["graph_rows"] == {"add": 2}             # missing steps and choice actions dropped
    assert WorkflowDefinition.from_dict(data).graph_rows == {"add": 2}

    data["graph_rows"] = {"add": "3", "save": -1, "f0": True, "s0": "x", "m1": 5}
    assert WorkflowDefinition.from_dict(data).graph_rows == {"add": 3, "m1": 5}
    data.pop("graph_rows")
    assert WorkflowDefinition.from_dict(data).graph_rows == {}
    assert "graph_rows" not in WorkflowDefinition.from_dict(data).to_dict()


def test_drag_a_step_up_its_lane_then_back(editor, qapp):
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QMenu

    from setiastro.saspro.workflow_graph import row_top_y

    editor.load_workflow_definition(_osc_like())
    editor.resize(2200, 1300)
    view = editor._graph.view
    for _ in range(6):
        qapp.processEvents()
    view.set_zoom(0.6)
    qapp.processEvents()
    vp = view.viewport()

    def center(key):
        it = view._nodes[key]
        return it.mapToScene(QPointF(it.w / 2, it.h / 2))

    def drag(a, b):
        a, b = view.mapFromScene(a), view.mapFromScene(b)
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
        for i in range(1, 9):
            QTest.mouseMove(vp, a + (b - a) * i / 8)
            qapp.processEvents()
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
        for _ in range(6):
            qapp.processEvents()

    g = view._graph
    target = QPointF(center("add").x(), row_top_y(g, 4) + g.row_h[4] / 2)
    drag(center("add"), target)
    assert editor.current_workflow.graph_rows == {"add": 4}
    assert [s.id for s in editor.current_workflow.steps if s.lane == "combine"] == ["add", "save"]
    g = view._graph
    assert (g.nodes["add"].rank, g.nodes["save"].rank, g.nodes["f0"].rank) == (4, 5, 5)

    # Its feeders now sit level with or below it, so their links come in from the side.
    m11 = g.nodes["m11"]
    edge = next(e for e in view._edges if e.edge.link is not None and e.edge.src == "m11")
    start = edge.path().pointAtPercent(0.0)
    assert abs(start.x() - (m11.x + m11.w)) < 0.5 and abs(start.y() - (m11.y + m11.h / 2)) < 0.5

    # Auto Position (node menu) undoes it.
    menu = QMenu()
    editor.graph_node_menu(menu, "add")
    next(a for a in menu.actions() if a.text() == "Auto Position").trigger()
    for _ in range(4):
        qapp.processEvents()
    assert editor.current_workflow.graph_rows == {}
    assert view._graph.nodes["add"].rank == 12

    # A step with nothing below it can also be dragged back down to its own
    # row, which drops the pin.
    g = view._graph
    drag(center("f0"), QPointF(center("f0").x(), row_top_y(g, 6) + g.row_h[6] / 2))
    assert editor.current_workflow.graph_rows == {"f0": 6}
    g = view._graph
    drag(center("f0"), QPointF(center("f0").x(), row_top_y(g, 13) + g.row_h[13] / 2))
    assert editor.current_workflow.graph_rows == {}

    # Dragging past another step in the lane still reorders.
    editor.graph_set_row("add", 3)
    for _ in range(4):
        qapp.processEvents()
    g = view._graph
    drag(center("add"), QPointF(center("add").x(), center("save").y() + 30))
    assert [s.id for s in editor.current_workflow.steps if s.lane == "combine"] == ["save", "add"]
    assert editor.current_workflow.graph_rows == {}      # a moved step starts from its automatic row

    # Auto Position All Steps (canvas menu) clears every pin at once.
    editor.graph_set_row("f0", 6)
    editor.graph_set_row("s1", 9)
    menu = QMenu()
    editor.graph_canvas_menu(menu)
    next(a for a in menu.actions() if a.text() == "Auto Position All Steps").trigger()
    assert editor.current_workflow.graph_rows == {}


# -----------------------------------------------------------------------------
# Canned color flows laid out like columns; links that go up; named links
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Beginner OSC Workflow", "Beginner Mono RGB Workflow"])
def test_canned_color_is_no_taller_than_main(name):
    wf = _canned(name)
    g = _placed(wf)
    main_rows = len(g.order["Main"])
    assert max(n.rank for n in g.nodes.values()) == main_rows - 1
    assert [g.nodes[g.order[lane][0]].rank for lane in ("Starless", "Stars Only")] == [0, 0]
    # Add Stars has a free row above it, so no gap in Main is spread for its links.
    add = g.nodes[g.order["Combine"][0]]
    assert add.rank == g.nodes[g.order["Starless"][-1]].rank + 2
    assert len(set(g.row_gap[:-1])) == 1


def _link_paths_clear(view):
    """{link label: steps its drawn path runs through, other than its own two ends}."""
    out = {}
    for e in view._edges:
        if not e.is_link:
            continue
        own = (e.edge.src, e.edge.dst)
        boxes = [it.mapRectToScene(it.body()) for k, it in view._nodes.items() if k not in own]
        path = e.path()
        pts = [path.pointAtPercent(i / 200) for i in range(201)]
        out[e.edge.label] = sum(1 for r in boxes if any(r.contains(p) for p in pts))
    return out


@pytest.mark.parametrize("name", ["Beginner OSC Workflow", "Beginner Mono RGB Workflow",
                                  "Beginner Mono Narrowband Workflow"])
def test_canned_links_never_run_through_a_step(editor, qapp, name):
    editor.resize(1500, 900)
    editor.load_workflow_definition(_canned(name))
    for _ in range(6):
        qapp.processEvents()
    crossed = _link_paths_clear(editor._graph.view)
    assert crossed and not any(crossed.values()), crossed


def test_a_link_up_from_a_split_pill_goes_up_the_gap_between_lanes(editor, qapp):
    editor.resize(1500, 900)
    editor.load_workflow_definition(_canned("Beginner OSC Workflow"))
    for _ in range(6):
        qapp.processEvents()
    from setiastro.saspro.workflow_graph import COL_GAP

    view = editor._graph.view
    g = view._graph
    ups = {e.edge.label: e for e in view._edges if e.is_link and e.edge.label in ("starless", "stars only")}
    for label, lane in (("starless", "Starless"), ("stars only", "Stars Only")):
        x, _w = g.columns[lane]
        gap_x = x - COL_GAP / 2               # the gap just left of the lane it goes to
        path = ups[label].path()
        middle = [path.pointAtPercent(t / 20) for t in range(8, 13)]
        assert all(abs(p.x() - gap_x) < COL_GAP / 2 for p in middle), (label, [p.x() for p in middle], gap_x)


def test_side_route_keeps_the_s_curve_when_it_is_clear():
    """Frank's own TestOSC shape: a link from a full-width step up into the next lane."""
    from PyQt6.QtCore import QPointF, QRectF

    from setiastro.saspro.workflow_graph import boxes_crossed, gap_curve, side_curve

    a, b = QPointF(100, 300), QPointF(200, 100)
    assert boxes_crossed(side_curve(a, b), [QRectF(0, 0, 90, 400)]) == 0
    assert boxes_crossed(side_curve(a, b), [QRectF(110, 200, 40, 30)]) == 1
    path = gap_curve(a, b, 150)
    assert path.pointAtPercent(0.0) == a and path.pointAtPercent(1.0) == b
    assert boxes_crossed(path, [QRectF(160, 150, 100, 30), QRectF(40, 150, 100, 30)]) == 0


def test_link_names_round_trip_and_survive_normalizing():
    from setiastro.saspro.workflows import (
        WorkflowDefinition, WorkflowLink, normalize_workflow_links, workflow_links,
    )

    link = WorkflowLink.from_dict({"from": "a", "to_lane": "x", "label": "  the   starless  "})
    assert link.label == "the starless"
    assert link.to_dict() == {"from": "a", "to_lane": "x", "label": "the starless"}
    assert "label" not in WorkflowLink(src="a", dst="b").to_dict()

    wf = _canned("Beginner OSC Workflow")
    wf.links = normalize_workflow_links(
        [WorkflowLink(src=l.src, dst=l.dst, dst_lane=l.dst_lane, label="half" if l.dst_lane == "Starless" else "")
         for l in workflow_links(wf)],
        wf.steps, wf.lanes)
    again = WorkflowDefinition.from_dict(wf.to_dict())
    assert sorted(l.label for l in again.links) == ["", "", "", "half"]
    assert again.graph_rows == wf.graph_rows and len(again.graph_rows) == 3


def test_graph_shows_a_links_name_else_its_tags():
    from setiastro.saspro.workflow_graph import build_workflow_graph
    from setiastro.saspro.workflows import ensure_workflow_links

    wf = _canned("Beginner OSC Workflow")
    ensure_workflow_links(wf)
    names = sorted(e.label for e in build_workflow_graph(wf).edges if e.link is not None)
    assert names == ["finished starless", "starless", "stars only", "stretched stars"]
    next(l for l in wf.links if l.dst_lane == "Stars Only").label = "just the stars"
    names = sorted(e.label for e in build_workflow_graph(wf).edges if e.link is not None)
    assert names == ["finished starless", "just the stars", "starless", "stretched stars"]


def test_name_a_link_from_its_menu_or_by_double_click(editor, monkeypatch, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QMenu

    editor.resize(1500, 900)
    editor.load_workflow_definition(_canned("Beginner OSC Workflow"))
    for _ in range(6):
        qapp.processEvents()
    view = editor._graph.view

    def edge(label):
        return next(e for e in view._edges if e.is_link and e.edge.label == label)

    answers = []
    monkeypatch.setattr("setiastro.saspro.workflows.QInputDialog.getText",
                        lambda _p, _t, _l, text="": (answers.append(text) or (reply, True)))

    # The menu offers it, and the box starts with what the link shows now.
    menu = QMenu()
    e = edge("stars only")
    editor.graph_link_menu(menu, e.edge.link, e.edge.label)
    texts = [a.text() for a in menu.actions() if a.text()]
    assert texts == ["Rename Link…", "Remove Link"]
    reply = "stars"
    next(a for a in menu.actions() if a.text() == "Rename Link…").trigger()
    for _ in range(4):
        qapp.processEvents()
    assert answers == ["stars only"]
    assert next(l for l in editor.current_workflow.links if l.dst_lane == "Stars Only").label == "stars"
    assert edge("stars").label_item is not None

    # Double-click on the link (its label is a handle too) asks as well.
    reply = "the starless half"
    lab = edge("starless").label_item
    pos = view.mapFromScene(lab.rect.center())
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    for _ in range(6):
        qapp.processEvents()
    assert answers[-1] == "starless"
    assert "the starless half" in [e.edge.label for e in view._edges if e.is_link]

    # The details panel says what it is called.
    split = next(s for s in editor.current_workflow.steps if s.kind == "split")
    from setiastro.saspro.workflows import workflow_link_lines
    lines = workflow_link_lines(editor.current_workflow, split.id, editor.catalog)
    assert any("as <i>the starless half</i>" in line for line in lines), lines

    # OK on the text it already showed changes nothing; blank gives the tags back.
    reply = "finished starless"
    editor.graph_name_link(edge("finished starless").edge.link, "finished starless")
    assert all(l.label == "" for l in editor.current_workflow.links if l.dst_lane == "Combine")
    reply = ""
    editor.graph_name_link(edge("stars").edge.link, "stars")
    for _ in range(4):
        qapp.processEvents()
    assert next(l for l in editor.current_workflow.links if l.dst_lane == "Stars Only").label == ""
    assert edge("stars only").label_item is not None

    # Saved with the workflow.
    saved = editor.current_workflow.to_dict()
    assert sum(1 for l in saved["links"] if l.get("label") == "the starless half") == 1


# -----------------------------------------------------------------------------
# Square links on tracks, and the first-open size
# -----------------------------------------------------------------------------

def _crossings(paths):
    """Proper crossings between the paths (each path a list of polygons)."""
    def segs(polys):
        out = []
        for poly in polys:
            pts = [poly.at(i) for i in range(poly.count())]
            out += [(p, q) for p, q in zip(pts, pts[1:]) if (p - q).manhattanLength() > 1e-6]
        return out

    def cross(o, a, b):
        return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x())

    def hit(s, t):
        (p1, p2), (q1, q2) = s, t
        d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
        d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
        return d1 * d2 < -1e-6 and d3 * d4 < -1e-6

    n = 0
    seg = [segs(p) for p in paths]
    for i in range(len(seg)):
        for j in range(i + 1, len(seg)):
            n += sum(1 for s in seg[i] for t in seg[j] if hit(s, t))
    return n


def test_a_fan_of_links_rides_one_track_each_and_crosses_as_little_as_it_can(editor, qapp):
    from setiastro.saspro.workflow_graph import TRACK_R

    editor.resize(1700, 900)
    editor.load_workflow_definition(_canned("Beginner Mono Narrowband Workflow"))
    for _ in range(6):
        qapp.processEvents()
    view = editor._graph.view
    fan = [e for e in view._edges if e.is_link and e.edge.label.endswith(("starless", "stars"))
           and e.edge.label.split()[0] in ("Ha", "OIII", "SII")]
    assert len(fan) == 6
    tracks = {}
    for e in fan:
        path = e.path()
        ys = [round(path.pointAtPercent(t / 100).y(), 3) for t in range(101)]
        flat = max(set(ys), key=ys.count)            # the horizontal run
        tracks[e.edge.label] = flat
        assert e.label_item.rect.center().y() == pytest.approx(flat)   # the name rides its own track
        assert e.end.x() - 3 * TRACK_R <= e.label_item.rect.right() <= e.end.x()   # beside its target
    assert len(set(tracks.values())) == 6
    # The source nearest the targets runs highest; each source's link to the far target above its near one.
    order = sorted(tracks, key=tracks.get)
    assert order == ["SII stars", "SII starless", "OIII stars", "OIII starless", "Ha stars", "Ha starless"]
    # Three sources, each into both targets: three crossings is the fewest possible.
    assert _crossings([e.path().toSubpathPolygons() for e in fan if not e.edge.label.startswith("SII")]
                      + [e.path().toSubpathPolygons() for e in fan if e.edge.label.startswith("SII")]) == 3


def test_a_longer_link_keeps_its_curve():
    from PyQt6.QtCore import QPointF

    from setiastro.saspro.workflow_graph import stack_tracks, track_curve

    a, b = QPointF(0, 0), QPointF(100, 80)
    path = track_curve(a, b, 30.0)
    assert path.pointAtPercent(0) == a and path.pointAtPercent(1) == b
    assert path.pointAtPercent(0.5).y() == pytest.approx(30.0)
    # Going left mirrors going right.
    links = [("near", QPointF(300, 0), QPointF(100, 90)), ("far", QPointF(500, 0), QPointF(100, 90))]
    assert stack_tracks(links) == ["near", "far"]


def test_first_open_size_fits_a_1080p_screen(editor, monkeypatch):
    from PyQt6.QtCore import QRect

    class Screen:
        def __init__(self, w, h):
            self.r = QRect(0, 0, w, h)

        def availableGeometry(self):
            return self.r

    for (w, h), want in (((2560, 1400), (1700, 900)),       # 1440p: no bigger than on 1080p
                         ((1920, 1032), (1700, 900)),       # 1080p after the taskbar
                         ((1536, 824), (1382, 725))):       # 1080p laptop at 125%
        monkeypatch.setattr(editor, "_screen_for_placement", lambda s=Screen(w, h): s)
        editor._place_first_time()
        # (The dialog's own minimum width can win; it depends on the font.)
        assert editor.width() == max(want[0], editor.minimumSizeHint().width())
        assert editor.height() == want[1]
    assert editor.FIRST_SIZE[0] + 8 <= 1920 and editor.FIRST_SIZE[1] + 32 <= 1032

