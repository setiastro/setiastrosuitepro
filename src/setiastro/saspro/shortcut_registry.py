#saspro/shortcut_registry.py
"""
Self-describing shortcuts, for the keyboard cheat sheet.

The cheat sheet finds most shortcuts on its own: every QAction in the menus and
toolbars carries its key sequence and its menu text, and that covers the bulk
of SASpro, including keys a user assigns to a command or a script. What it
cannot find on its own is

  * a bare QShortcut -- it has a key but no text, which is how F2 used to show
    up as just "Shortcut";
  * a key handled in keyPressEvent or an eventFilter (Space for the pixel
    readout, Delete on the shortcuts canvas) -- there is no object at all;
  * a mouse or drag gesture.

Those are declared here, next to the code that implements them, so adding one
is one line in the place it already lives and the cheat sheet picks it up:

    add_shortcut(self, "F2", self._rename_document,
                 label="Rename view", context="Image view")

    declare_key("Space", "Hold, then click or drag: probe pixel values",
                "Image view")

    declare_gesture("Shift+drag the view tab onto another view",
                    "Image view", "Apply this image to it as a mask")

Declarations are made at import, so this covers app-wide shortcuts (views,
docks, the shortcuts canvas), whose modules load at startup. A tool dialog's
own keys stay with the tool; its module may not even be imported yet.

Labels are plain English at declaration time and are passed through
QCoreApplication.translate("CheatSheet", ...) when displayed.
"""
from __future__ import annotations

from PyQt6.QtGui import QKeySequence, QShortcut

_KEYS: list = []        # (keys, label, context)
_GESTURES: list = []    # (gesture, context, effect)


def declare_key(keys: str, label: str, context: str) -> None:
    """A key handled somewhere the cheat sheet cannot see (keyPressEvent etc.)."""
    row = (str(keys), str(label), str(context))
    if row not in _KEYS:
        _KEYS.append(row)


def declare_gesture(gesture: str, context: str, effect: str) -> None:
    """A mouse or drag gesture."""
    row = (str(gesture), str(context), str(effect))
    if row not in _GESTURES:
        _GESTURES.append(row)


def add_shortcut(parent, keys, slot, *, label: str, context: str,
                 shortcut_context=None) -> QShortcut:
    """
    QShortcut(keys, parent) wired to `slot`, labelled for the cheat sheet.

    Prefer this over a bare QShortcut: the label is what the cheat sheet shows,
    and it is declared even before the widget that owns the shortcut exists.
    """
    seq = keys if isinstance(keys, QKeySequence) else QKeySequence(keys)
    sc = QShortcut(seq, parent)
    if shortcut_context is not None:
        sc.setContext(shortcut_context)
    sc.activated.connect(slot)
    sc.setProperty("cheat_label", label)
    sc.setProperty("cheat_context", context)
    sc.setWhatsThis(label)
    declare_key(seq.toString(QKeySequence.SequenceFormat.PortableText), label, context)
    return sc


def declared_keys() -> list:
    return list(_KEYS)


def declared_gestures() -> list:
    return list(_GESTURES)
