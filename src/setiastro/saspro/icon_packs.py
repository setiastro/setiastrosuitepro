# saspro/icon_packs.py
"""
Optional icon packs: alternate artwork for the application's button icons.

A pack is a folder of image files named after the stock icons in
``setiastro/images`` (``hdr.png``, ``undoicon.png``, ...). Subfolders are only
for the artist's convenience, so a pack may be flat or grouped by category.
Packs may be partial: any icon a pack doesn't provide uses the stock artwork.

Packs are discovered in two places, and a user pack wins over a bundled pack
of the same name:

* bundled: ``setiastro/images/icon_packs/<Pack Name>/``
* user:    ``<SASpro data folder>/icon_packs/<Pack Name>/``

Icons are resolved once and held by widgets for the life of the session, so the
selected pack is read a single time per process and changing it needs a restart.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_PACK = ""  # the stock icons
_SETTINGS_KEY = "ui/icon_pack"
_SWAPPABLE_EXTS = (".png", ".svg")  # interchangeable: both load through QIcon


def _settings():
    # Explicit scope, like the language preference: icons are resolved at import
    # time, before the QApplication identity has been set.
    from PyQt6.QtCore import QSettings
    return QSettings("SetiAstro", "SetiAstroSuitePro")


def get_saved_icon_pack() -> str:
    """Name of the selected pack, or DEFAULT_PACK for the stock icons."""
    return _settings().value(_SETTINGS_KEY, DEFAULT_PACK, type=str) or DEFAULT_PACK


def save_icon_pack(name: str) -> None:
    """Select a pack by name. Takes effect the next time the app starts."""
    _settings().setValue(_SETTINGS_KEY, name or DEFAULT_PACK)


def user_packs_dir() -> Path:
    """Folder for user-installed packs. It is not created until it is needed."""
    # Beside the runtime folder rather than inside it: the runtime gets replaced
    # when it is reinstalled or upgraded, and packs must survive that.
    from PyQt6.QtCore import QStandardPaths
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.GenericDataLocation)
    return Path(base) / "SASpro" / "icon_packs"


def _bundled_packs_dir() -> Path:
    from setiastro.saspro.resources import _get_base_path
    return Path(_get_base_path()) / "images" / "icon_packs"


def available_icon_packs() -> dict[str, Path]:
    """Installed packs as {name: folder}, sorted by name."""
    packs: dict[str, Path] = {}
    for root in (_bundled_packs_dir(), user_packs_dir()):  # user packs override
        try:
            packs.update(
                (entry.name, Path(entry.path))
                for entry in os.scandir(root)
                if entry.is_dir() and not entry.name.startswith(".")
            )
        except OSError:
            continue  # this location doesn't exist
    return dict(sorted(packs.items(), key=lambda kv: kv[0].lower()))


def _index_pack(folder: Path) -> dict[str, str]:
    """Map lowercase icon filenames in a pack, at any depth, to their paths."""
    index: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.lower().endswith(_SWAPPABLE_EXTS):
                index.setdefault(fn.lower(), os.path.join(dirpath, fn))
    return index


@lru_cache(maxsize=1)
def _active_pack_index() -> dict[str, str]:
    try:
        folder = available_icon_packs().get(get_saved_icon_pack())
        return _index_pack(folder) if folder else {}
    except Exception:
        return {}  # icons must never stop the app from starting: use stock


def icon_override(filename: str) -> str | None:
    """
    Path of the active pack's replacement for a stock icon, or None to use stock.

    An exact filename wins; otherwise a .png stands in for a .svg and vice versa.
    """
    index = _active_pack_index()
    if not index:
        return None
    stem, ext = os.path.splitext(filename.lower())
    if ext not in _SWAPPABLE_EXTS:
        return None  # e.g. the .ico/.icns the OS needs for the application icon
    alternates = (e for e in _SWAPPABLE_EXTS if e != ext)
    for candidate in (stem + ext, *(stem + e for e in alternates)):
        if candidate in index:
            return index[candidate]
    return None
