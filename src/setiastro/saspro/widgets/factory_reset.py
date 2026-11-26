#saspro/widgets/factory_reset.py
"""
Reset a tool to its factory defaults.

A tool snapshots its controls as constructed: after its widgets are built and
before saved settings, a preset or a replay seeds them. Its Reset button then
puts every one of those controls back. The defaults live only in the
constructor, so Reset can never drift from what a first-time user sees.

    from setiastro.saspro.widgets.factory_reset import (
        make_reset_button, snapshot_controls, restore_controls)

    btn_reset = make_reset_button(self)                 # in the button row
    btn_reset.clicked.connect(self._reset_to_factory)
    ...
    self._factory = snapshot_controls(self, exclude=(self.cmb_source,))
    self._load_settings()                               # AFTER the snapshot

    def _reset_to_factory(self):
        restore_controls(self._factory)
        # plus any state that is not a widget (curve points, lists), then
        # save settings if this tool only saves on Apply

What is captured: sliders, spin boxes, combo boxes, checkable buttons and
group boxes, and compound controls that expose value()/setValue() (the
slider-with-edit and CustomSpinBox kinds). Text fields only when passed in
`text_fields`, so a path, an API key or a typed formula is never wiped by
accident.

Never captured: anything in `exclude` (and everything inside it), scroll
bars, progress bars, date/time edits, and controls inside item views, menus
and toolbars. Pass image/view pickers, file and folder paths, and user preset
lists in `exclude`.

Restoring does not block signals: every control announces its change the
normal way, so labels, previews and auto-saved settings follow.
"""
from __future__ import annotations

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import (
    QAbstractButton, QAbstractItemView, QAbstractSlider, QAbstractSpinBox,
    QComboBox, QDateTimeEdit, QDoubleSpinBox, QGroupBox, QLineEdit, QMenu,
    QPlainTextEdit, QProgressBar, QPushButton, QScrollBar, QSpinBox,
    QTextEdit, QToolBar, QWidget,
)

__all__ = ["make_reset_button", "snapshot_controls", "restore_controls"]


def make_reset_button(parent: QWidget | None = None) -> QPushButton:
    """The standard Reset button: same label and tooltip in every tool."""
    btn = QPushButton(QCoreApplication.translate("Common", "Reset"), parent)
    btn.setToolTip(QCoreApplication.translate(
        "Common", "Reset every setting in this tool to its factory default"))
    return btn


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

_SKIP_SUBTREES = (QAbstractItemView, QMenu, QToolBar)


def _kind(w: QWidget, text_ids: set[int]) -> str | None:
    if isinstance(w, (QScrollBar, QProgressBar, QDateTimeEdit)):
        return None
    if isinstance(w, QComboBox):
        return "combo"
    if isinstance(w, QGroupBox):
        return "check" if w.isCheckable() else None
    if isinstance(w, QAbstractButton):
        return "check" if w.isCheckable() else None
    if isinstance(w, QAbstractSlider):
        return "value"
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return "value"
    if isinstance(w, QAbstractSpinBox):
        return None
    if isinstance(w, QLineEdit):
        return "line" if id(w) in text_ids else None
    if isinstance(w, (QPlainTextEdit, QTextEdit)):
        return "plain" if id(w) in text_ids else None
    # compound controls (slider + edit, CustomSpinBox, pyqtgraph SpinBox)
    if callable(getattr(w, "value", None)) and callable(getattr(w, "setValue", None)):
        return "value"
    return None


def _read(w, kind):
    if kind == "combo":
        return (w.currentIndex(), w.currentText())
    if kind == "check":
        return bool(w.isChecked())
    if kind == "line":
        return w.text()
    if kind == "plain":
        return w.toPlainText()
    return w.value()


def snapshot_controls(root: QWidget, *, exclude=(), text_fields=()) -> list:
    """Record the current value of every parameter control under `root`.

    Call it once the UI is fully built and BEFORE saved settings, presets or
    replays are applied. Returns an opaque list for restore_controls().
    """
    excl = {id(w) for w in exclude if w is not None}
    text_ids = {id(w) for w in text_fields if w is not None}
    out: list = []

    def walk(parent):
        for ch in parent.children():
            if not isinstance(ch, QWidget) or id(ch) in excl:
                continue
            if isinstance(ch, _SKIP_SUBTREES):
                continue
            kind = _kind(ch, text_ids)
            if kind is not None:
                try:
                    out.append((ch, kind, _read(ch, kind)))
                except Exception:
                    pass
                if not isinstance(ch, QGroupBox):
                    continue        # a leaf: its inner line edit / slider is its own business
            walk(ch)

    walk(root)
    return out


# --------------------------------------------------------------------------
# restore
# --------------------------------------------------------------------------

def _write(w, kind, v):
    if kind == "combo":
        idx, text = v
        if _editable(w):
            # select the item when there is one (setCurrentText on an editable
            # combo only rewrites the edit text and leaves the old item current)
            j = idx if (0 <= idx < w.count() and w.itemText(idx) == text) else w.findText(text)
            if j >= 0:
                w.setCurrentIndex(j)
            else:
                w.setCurrentText(text)
        elif 0 <= idx < w.count() and w.itemText(idx) == text:
            w.setCurrentIndex(idx)
        else:
            j = w.findText(text)
            if j >= 0:
                w.setCurrentIndex(j)
            elif 0 <= idx < w.count():
                w.setCurrentIndex(idx)
    elif kind == "check":
        w.setChecked(v)
    elif kind == "line":
        if w.text() != v:
            w.setText(v)
            w.editingFinished.emit()      # most tools commit a typed value on editingFinished
    elif kind == "plain":
        if w.toPlainText() != v:
            w.setPlainText(v)
    else:
        w.setValue(v)


def _editable(w) -> bool:
    try:
        return bool(w.isEditable())
    except Exception:
        return False


def _matches(w, kind, v) -> bool:
    try:
        cur = _read(w, kind)
    except Exception:
        return True                       # deleted or unreadable: nothing to do
    if kind == "combo":
        if cur[1] != v[1]:
            return False
        j = w.findText(v[1])
        return cur[0] == v[0] or j == cur[0] or (_editable(w) and j < 0)
    if isinstance(v, float) or isinstance(cur, float):
        try:
            return abs(float(cur) - float(v)) <= 1e-9 * max(1.0, abs(float(v)))
        except Exception:
            return cur == v
    return cur == v


def restore_controls(snapshot: list, *, passes: int = 3) -> None:
    """Put every control in `snapshot` back to its recorded value.

    Signals are left on so each tool reacts as it does to a user change. A
    control that depends on another (a combo that rewrites a slider, a range
    set by a mode) can be overridden while the list is replayed, so the
    replay repeats until every value has stuck.
    """
    for _ in range(max(1, passes)):
        pending = False
        for w, kind, v in snapshot:
            try:
                if _matches(w, kind, v):
                    continue
                _write(w, kind, v)
                pending = True
            except RuntimeError:
                continue                  # widget deleted since the snapshot
            except Exception:
                continue
        if not pending:
            break
