"""Toolbar icon size and the layout from before 2.0 (workspace_layout.py).

2.0 starts everyone at 20 px icons. Putting the 1.x layout back brings its
icon size back: the size it saved, or 24 (1.x's default) when it never saved one.
"""
from __future__ import annotations

import pytest

from setiastro.saspro import workspace_layout as wl


@pytest.fixture
def settings(tmp_path):
    from PyQt6.QtCore import QSettings
    s = QSettings(str(tmp_path / "saspro.ini"), QSettings.Format.IniFormat)
    yield s
    s.sync()


def _as_1x(s, icon_size=None):
    s.setValue("ui/main/state", b"state-from-1x")
    s.setValue("ui/theme", "dark")
    if icon_size is not None:
        s.setValue(wl.ICON_SIZE_KEY, icon_size)
    s.sync()


def test_defaults():
    assert wl.ICON_SIZE_DEFAULT == 20
    assert wl.ICON_SIZE_BEFORE_V2 == 24


def test_icon_size_is_layout():
    assert wl.is_layout_key("toolbar/icon_size")
    assert wl.is_layout_key("Toolbar/icon_size")


@pytest.mark.parametrize("saved, restored", [(None, 24), (24, 24), (32, 32), (20, 20)])
def test_first_2x_start_drops_the_size_and_restore_brings_it_back(settings, saved, restored):
    _as_1x(settings, saved)
    assert wl.start_new_layout_if_due(settings, "2.0.0")
    assert not settings.contains(wl.ICON_SIZE_KEY)          # so the 2.0 default applies
    assert wl.load_saved_layout(settings)
    assert settings.value(wl.ICON_SIZE_KEY, type=int) == restored


def test_restore_replaces_a_size_chosen_in_2x(settings):
    _as_1x(settings, 24)
    wl.start_new_layout_if_due(settings, "2.0.0")
    settings.setValue(wl.ICON_SIZE_KEY, 28)
    assert wl.load_saved_layout(settings)
    assert settings.value(wl.ICON_SIZE_KEY, type=int) == 24


def test_new_install_has_nothing_set_aside(settings):
    assert not wl.start_new_layout_if_due(settings, "2.0.0")
    assert not wl.has_saved_layout(settings)
    assert not settings.contains(wl.ICON_SIZE_KEY)


def test_later_2x_start_leaves_the_size_alone(settings):
    _as_1x(settings, 24)
    wl.start_new_layout_if_due(settings, "2.0.0")
    settings.setValue(wl.ICON_SIZE_KEY, 28)
    assert not wl.start_new_layout_if_due(settings, "2.0.1")
    assert settings.value(wl.ICON_SIZE_KEY, type=int) == 28


def test_1x_start_leaves_the_size_alone(settings):
    _as_1x(settings, 24)
    assert not wl.start_new_layout_if_due(settings, "1.24.8")
    assert settings.value(wl.ICON_SIZE_KEY, type=int) == 24
