"""Start panel: Process step by step.

With no image open it asks for images, then opens Workflow Assistant. With
one open (the panel put up over it from View > Start Panel or the Essentials
button) a file prompt reads as if SASpro didn't know about the image, so it
goes straight to Workflow Assistant.
"""
from __future__ import annotations

from types import SimpleNamespace

from setiastro.saspro.start_panel import StartPanel


def _main_window(views):
    calls = []
    mw = SimpleNamespace(
        mdi=SimpleNamespace(subWindowList=lambda: list(views)),
        open_files=lambda: calls.append("open files"),
        act_workflows=SimpleNamespace(trigger=lambda: calls.append("workflows")),
    )
    return mw, calls


def test_no_image_open_asks_for_images_then_opens_workflows():
    mw, calls = _main_window([])
    StartPanel._process_step_by_step(SimpleNamespace(_mw=mw))
    assert calls == ["open files", "workflows"]


def test_image_open_goes_straight_to_workflows():
    mw, calls = _main_window(["a view"])
    StartPanel._process_step_by_step(SimpleNamespace(_mw=mw))
    assert calls == ["workflows"]
