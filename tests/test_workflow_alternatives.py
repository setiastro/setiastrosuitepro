from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_legacy_step_has_no_alternatives():
    from setiastro.saspro.workflows import WorkflowStep

    step = WorkflowStep.from_dict({"command_id": "curves", "lane": "main"})
    assert step.alternatives == []
    assert "alternatives" not in step.to_dict()


def test_alternatives_round_trip_and_drop_junk():
    from setiastro.saspro.workflows import WorkflowDefinition, WorkflowStep

    step = WorkflowStep.from_dict({
        "id": "abc123",
        "command_id": "stat_stretch",
        "preset": {"strength": 1},
        "lane": "main",
        "alternatives": [
            {"command_id": "curves", "preset": {"mode": "rgb"}, "label_override": "Curves doux"},
            {"command_id": "   "},
            {"preset": {"x": 1}},
            "nope",
            {"command_id": "clahe"},
        ],
    })
    assert step.command_id == "stat_stretch"
    assert step.preset == {"strength": 1}
    assert [a.command_id for a in step.alternatives] == ["curves", "clahe"]
    assert step.alternatives[0].preset == {"mode": "rgb"}
    assert step.alternatives[0].label_override == "Curves doux"
    assert step.alternatives[1].preset == {}

    again = WorkflowStep.from_dict(step.to_dict())
    assert again.alternatives[0].label_override == "Curves doux"
    assert again.alternatives[1].command_id == "clahe"
    assert again.preset == {"strength": 1}

    wf = WorkflowDefinition.from_dict({
        "name": "Trials",
        "lanes": ["main"],
        "steps": [step.to_dict()],
    })
    assert wf.steps[0].alternatives[0].command_id == "curves"


def test_label_badge_counts_every_choice():
    from setiastro.saspro.workflows import WorkflowAlternative, WorkflowStep, format_workflow_step_label

    plain = WorkflowStep(command_id="stat_stretch", label_override="Statistical Stretch")
    assert format_workflow_step_label(plain, {}) == "Statistical Stretch"

    with_preset = WorkflowStep(
        command_id="stat_stretch",
        label_override="Statistical Stretch",
        preset={"strength": 1},
    )
    assert format_workflow_step_label(with_preset, {}) == "◆ Statistical Stretch"

    trial = WorkflowStep(
        command_id="stat_stretch",
        label_override="Statistical Stretch",
        alternatives=[WorkflowAlternative(command_id="curves")],
    )
    assert format_workflow_step_label(trial, {}) == "Statistical Stretch ▾ 2"

    seeded = WorkflowStep(
        command_id="stat_stretch",
        label_override="Statistical Stretch",
        preset={"strength": 1},
        alternatives=[
            WorkflowAlternative(command_id="curves"),
            WorkflowAlternative(command_id="clahe"),
        ],
    )
    assert format_workflow_step_label(seeded, {}) == "◆ Statistical Stretch ▾ 3"


def test_choices_lead_with_the_primary_tool():
    from setiastro.saspro.workflows import WorkflowAlternative, WorkflowStep, workflow_step_choices

    step = WorkflowStep(
        command_id="stat_stretch",
        label_override="Statistical Stretch",
        preset={"a": 1},
        alternatives=[
            WorkflowAlternative(command_id="curves", label_override="Curves"),
            WorkflowAlternative(command_id="clahe", preset={"clip": 2}),
        ],
    )
    choices = workflow_step_choices(step, {})
    assert [c[0] for c in choices] == ["stat_stretch", "curves", "clahe"]
    assert choices[0][1] == {"a": 1}
    assert choices[0][2] == "Statistical Stretch"
    assert choices[1][2] == "Curves"
    assert choices[2][1] == {"clip": 2}
    assert choices[2][2] == "clahe"


def test_single_choice_opens_without_a_menu():
    from setiastro.saspro.workflows import prompt_workflow_choice

    choice = ("curves", {"a": 1}, "Curves")
    assert prompt_workflow_choice(None, [choice]) == choice
    assert prompt_workflow_choice(None, []) is None


def test_note_step_has_no_choices():
    from setiastro.saspro.workflows import WorkflowStep, format_workflow_step_label, workflow_step_choices

    step = WorkflowStep(kind="note", note="try either")
    assert workflow_step_choices(step, {}) == []
    assert format_workflow_step_label(step, {}) == "📝 Note"


def test_menu_payload_keeps_command_and_preset(qapp):
    from PyQt6.QtWidgets import QMenu

    menu = QMenu()
    act = menu.addAction("Curves")
    act.setData({"command_id": "curves", "preset": {"mode": "rgb"}})
    payload = act.data()
    assert payload["command_id"] == "curves"
    assert dict(payload["preset"]) == {"mode": "rgb"}


def test_item_roles_keep_alternatives(qapp):
    from PyQt6.QtWidgets import QListWidgetItem

    from setiastro.saspro.workflows import (
        WorkflowAlternative,
        WorkflowStep,
        pack_workflow_step_item,
        unpack_workflow_step_item,
    )

    step = WorkflowStep(
        id="step1",
        command_id="stat_stretch",
        note="stretch here",
        lane="stars",
        preset={"a": 1},
        label_override="Stretch",
        alternatives=[WorkflowAlternative(command_id="curves", preset={"mode": "rgb"}, label_override="Curves")],
    )
    item = QListWidgetItem("Stretch")
    pack_workflow_step_item(item, step)
    back = unpack_workflow_step_item(item, lane="stars")
    assert back.id == "step1"
    assert back.command_id == "stat_stretch"
    assert back.note == "stretch here"
    assert back.preset == {"a": 1}
    assert back.label_override == "Stretch"
    assert back.lane == "stars"
    assert len(back.alternatives) == 1
    assert back.alternatives[0].command_id == "curves"
    assert back.alternatives[0].preset == {"mode": "rgb"}
    assert back.alternatives[0].label_override == "Curves"


def test_choice_round_trip_compacts_children_and_strips_title():
    from setiastro.saspro.workflows import WorkflowAlternative, WorkflowDefinition, WorkflowStep

    raw = {
        "name": "Trials",
        "lanes": ["main", "stars"],
        "steps": [
            {
                "id": "child",
                "command_id": "curves",
                "lane": "stars",
                "parent_id": "title",
                "preset": {"mode": "rgb"},
                "alternatives": [{"command_id": "clahe"}],
            },
            {
                "id": "title",
                "kind": "choice",
                "label_override": "Stretch",
                "lane": "main",
                "command_id": "should-drop",
                "alternatives": [{"command_id": "nope"}],
            },
            {"id": "note", "kind": "note", "lane": "main", "parent_id": "title", "note": "keep"},
            {"id": "orphan", "command_id": "crop", "lane": "main", "parent_id": "missing"},
        ],
    }
    wf = WorkflowDefinition.from_dict(raw)
    assert [s.id for s in wf.steps] == ["title", "child", "note", "orphan"]
    title, child, note, orphan = wf.steps
    assert title.kind == "choice"
    assert title.command_id == ""
    assert title.alternatives == []
    assert title.parent_id == ""
    assert "parent_id" not in title.to_dict()
    assert child.parent_id == "title"
    assert child.lane == "main"
    assert child.preset == {"mode": "rgb"}
    assert child.alternatives[0].command_id == "clahe"
    assert note.parent_id == ""
    assert note.note == "keep"
    assert orphan.parent_id == ""
    assert "parent_id" not in orphan.to_dict()

    again = WorkflowDefinition.from_dict(wf.to_dict())
    assert [s.id for s in again.steps] == ["title", "child", "note", "orphan"]
    assert again.steps[1].alternatives[0].command_id == "clahe"


def test_two_choice_groups_stay_separate():
    from setiastro.saspro.workflows import WorkflowStep, normalize_workflow_steps

    steps = [
        WorkflowStep(id="a", kind="choice", label_override="A", lane="main"),
        WorkflowStep(id="b", kind="choice", label_override="B", lane="main"),
        WorkflowStep(id="a1", command_id="curves", parent_id="a", lane="main"),
        WorkflowStep(id="b1", command_id="clahe", parent_id="b", lane="main"),
        WorkflowStep(id="a2", command_id="crop", parent_id="a", lane="main"),
    ]
    out = normalize_workflow_steps(steps)
    assert [s.id for s in out] == ["a", "a1", "a2", "b", "b1"]


def test_choice_menu_does_not_expand_child_alternatives():
    from setiastro.saspro.workflows import (
        WorkflowAlternative,
        WorkflowStep,
        choice_group_entries,
        format_workflow_lane_label,
        format_workflow_step_label,
    )

    title = WorkflowStep(id="t", kind="choice", label_override="Stretch")
    child = WorkflowStep(
        id="c",
        command_id="curves",
        label_override="Curves",
        parent_id="t",
        preset={"mode": "rgb"},
        alternatives=[WorkflowAlternative(command_id="clahe", label_override="CLAHE")],
    )
    assert format_workflow_step_label(title, {}) == "▾ Stretch"
    assert format_workflow_lane_label(child, {}) == "◆ Curves ▾ 2"
    entries = choice_group_entries([child], {})
    assert entries == [("curves", {"mode": "rgb"}, "◆ Curves ▾ 2")]


def test_completion_target_is_the_choice_title():
    from setiastro.saspro.workflows import WorkflowStep, choice_completion_target_id

    title = WorkflowStep(id="t", kind="choice", label_override="Stretch")
    child = WorkflowStep(id="c", command_id="curves", parent_id="t")
    other = WorkflowStep(id="o", command_id="crop")
    steps = [title, child, other]
    assert choice_completion_target_id(title, steps) == "t"
    assert choice_completion_target_id(child, steps) == "t"
    assert choice_completion_target_id(other, steps) == "o"


def test_parent_id_after_drop_rules():
    from setiastro.saspro.workflows import WorkflowStep, parent_id_after_drop

    title = WorkflowStep(id="t", kind="choice", label_override="Stretch", lane="main")
    first = WorkflowStep(id="a", command_id="curves", parent_id="t", lane="main")
    last = WorkflowStep(id="b", command_id="clahe", parent_id="t", lane="main")
    outside = WorkflowStep(id="o", command_id="crop", lane="main")
    steps = [title, first, last, outside]

    assert parent_id_after_drop(steps, "o", "", "t", "below") == "t"
    assert parent_id_after_drop(steps, "o", "", "t", "above") == "t"
    assert parent_id_after_drop(steps, "o", "", "t", "") == "t"
    assert parent_id_after_drop(steps, "o", "", "a", "below") == "t"
    assert parent_id_after_drop(steps, "o", "", "b", "below") == ""
    assert parent_id_after_drop(steps, "a", "t", "b", "below") == "t"
    assert parent_id_after_drop(steps, "a", "t", "t", "above") == "t"


def test_item_roles_keep_parent_id(qapp):
    from PyQt6.QtWidgets import QListWidgetItem

    from setiastro.saspro.workflows import WorkflowStep, pack_workflow_step_item, unpack_workflow_step_item

    step = WorkflowStep(id="c", command_id="curves", parent_id="title", lane="main")
    item = QListWidgetItem("Curves")
    pack_workflow_step_item(item, step)
    back = unpack_workflow_step_item(item, lane="main")
    assert back.parent_id == "title"
