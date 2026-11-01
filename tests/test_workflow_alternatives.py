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
