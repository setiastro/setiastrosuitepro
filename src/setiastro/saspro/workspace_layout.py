# src/setiastro/saspro/workspace_layout.py
# SetiAstro Suite Pro  ·  Franklin Marek  ·  www.setiastro.com
"""
The saved workspace layout, as something that can be set aside and put back.

SASpro 2.0 starts everyone on the new default layout once. A layout saved by
1.x would otherwise be restored over it, so a returning user would open 2.0
and see exactly what they left. On that one start the 1.x layout is copied to
its own settings group, then cleared, and the window comes up as a new install
does. The welcome window then offers the old layout back; View > Layout does
too, at any later time.

What counts as layout is what Reset to Factory Layout resets: toolbar icons,
their size, order and positions, panel places, panels in the second window,
theme and background. Window size, open images, tool settings, presets,
shortcuts and the Display-Stretch options are not layout and are never touched
here. 2.0 draws toolbar icons at 20 px and 1.x at 24, so a 1.x layout put back
brings its own icon size with it.

No widgets in here. Nothing is removed until the copy has been read back.
"""
from __future__ import annotations

import os

# The version whose first start brings the new layout, and its number. Bump
# the number only to start everyone on a new default layout again.
NEW_LAYOUT_MAJOR = 2
LAYOUT_GENERATION = 2

GENERATION_KEY = "ui/layout_generation"
PENDING_KEY = "ui/layout_choice_pending"
SAVED_GROUP = "saved_layout/before_v2"

# Toolbar icon size: 2.0's default, and 1.x's for a layout that never saved one.
ICON_SIZE_KEY = "toolbar/icon_size"
ICON_SIZE_DEFAULT = 20
ICON_SIZE_BEFORE_V2 = 24

# --upgraderun (settings_sandbox.py) sets this: behave as the first start of
# the new version, whatever this build's version is.
FORCE_ENV = "SASPRO_FORCE_NEW_LAYOUT"

_STATE_KEYS = ("main_window/state", "ui/main/state")

# Compared in lower case: the Windows registry folds "toolbar/..." and
# "Toolbar/..." into one group, the other platforms keep them apart.
_LAYOUT_KEYS = frozenset((
    "ui/main/state",
    "main_window/state",
    "main_window/pre_host_dock_state",
    "ui/theme",
    "ui/custom_background",
    "ui/builtin_background",
    "display/bg_opacity",
    "icon_table/visible",
))
_LAYOUT_PREFIXES = ("toolbar/", "main_window/dock_host/")


def is_layout_key(key: str) -> bool:
    k = str(key).lower()
    return k in _LAYOUT_KEYS or k.startswith(_LAYOUT_PREFIXES)


def layout_keys(settings) -> list:
    return [k for k in settings.allKeys() if is_layout_key(k)]


def has_layout(settings) -> bool:
    """A window layout was saved, by any version."""
    return any(settings.contains(k) for k in _STATE_KEYS)


def has_saved_layout(settings) -> bool:
    """A 1.x layout was set aside and can be put back."""
    return any(settings.contains(f"{SAVED_GROUP}/{k}") for k in _STATE_KEYS)


def choice_pending(settings) -> bool:
    """The old layout was set aside and the user hasn't been asked yet."""
    return has_saved_layout(settings) and settings.value(PENDING_KEY, False, type=bool)


def choice_made(settings) -> None:
    settings.remove(PENDING_KEY)
    settings.sync()


def version_major(version) -> int:
    head = str(version or "").strip().lstrip("vV").split(".", 1)[0]
    return int(head) if head.isdigit() else 0


def new_layout_due(settings, version) -> bool:
    """This is the first start of a version that brings the new layout."""
    if os.environ.get(FORCE_ENV) != "1" and version_major(version) < NEW_LAYOUT_MAJOR:
        return False
    try:
        seen = int(settings.value(GENERATION_KEY, 0) or 0)
    except (TypeError, ValueError):
        seen = 0
    return seen < LAYOUT_GENERATION


def start_new_layout_if_due(settings, version) -> bool:
    """On the first start of 2.0, set a saved 1.x layout aside so the window
    comes up as a new install. True if one was set aside on this start.

    Call before anything reads a layout setting.
    """
    if not new_layout_due(settings, version):
        return False
    had = has_layout(settings)
    if had:
        if not _save_copy(settings):
            return False                 # keep the layout; try again next start
        settings.setValue(PENDING_KEY, True)
    settings.setValue(GENERATION_KEY, LAYOUT_GENERATION)
    settings.sync()
    if had:
        clear_layout(settings)
    return had


def _save_copy(settings) -> bool:
    """Copy every layout key under SAVED_GROUP; True once it reads back."""
    settings.remove(SAVED_GROUP)
    keys = layout_keys(settings)
    for k in keys:
        settings.setValue(f"{SAVED_GROUP}/{k}", settings.value(k))
    if not settings.contains(ICON_SIZE_KEY):
        # Never saved, so 1.x drew its toolbars at its default.
        settings.setValue(f"{SAVED_GROUP}/{ICON_SIZE_KEY}", ICON_SIZE_BEFORE_V2)
    settings.sync()
    if settings.status() != settings.Status.NoError:
        return False
    return all(settings.value(f"{SAVED_GROUP}/{k}") == settings.value(k) for k in keys)


def clear_layout(settings) -> None:
    for k in layout_keys(settings):
        settings.remove(k)
    settings.sync()


def load_saved_layout(settings) -> bool:
    """Make the set-aside layout the current one. The copy is kept, so this
    can be done again later. The caller rebuilds the window from settings."""
    if not has_saved_layout(settings):
        return False
    settings.beginGroup(SAVED_GROUP)
    saved = {k: settings.value(k) for k in settings.allKeys()}
    settings.endGroup()
    clear_layout(settings)
    for k, v in saved.items():
        settings.setValue(k, v)
    settings.sync()
    return True


def describe_saved_layout(settings) -> list:
    """What was set aside, as short facts for the welcome window:
    ("theme", name), ("background", None), ("moved", n), ("hidden", n),
    ("second_window", None)."""
    import json

    def val(key, default=""):
        v = settings.value(f"{SAVED_GROUP}/{key}", default)
        return default if v is None else v

    facts = [("theme", str(val("ui/theme", "system") or "system").lower())]
    if str(val("ui/custom_background")).strip():
        facts.append(("background", None))
    try:
        where = json.loads(str(val("Toolbar/Assignments")) or "{}")
    except ValueError:
        where = {}
    if isinstance(where, dict) and where:
        hidden = sum(1 for t in where.values() if str(t) == "Toolbar/Hidden")
        if len(where) > hidden:
            facts.append(("moved", len(where) - hidden))
        if hidden:
            facts.append(("hidden", hidden))
    if str(val("main_window/dock_host/active", "")).lower() in ("true", "1"):
        facts.append(("second_window", None))
    return facts
