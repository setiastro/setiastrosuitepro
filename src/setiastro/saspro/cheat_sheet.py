#saspro/cheat_sheet.py
"""
Keyboard shortcut cheat sheet.

Built from the running app every time it opens, so it cannot drift out of
date:

  * every QAction in the menu bar, by menu, enabled or not (Undo is disabled
    until there is something to undo, and used to vanish from this list for
    exactly that reason) -- including keys a user assigns to a command or a
    script, since those are set on the same actions;
  * toolbar-only actions;
  * QShortcuts created with shortcut_registry.add_shortcut(), which carry a
    label (a bare QShortcut has only a key, and is skipped rather than shown
    as the meaningless "Shortcut" it used to be);
  * keys and gestures declared with shortcut_registry.declare_key() /
    declare_gesture() next to the code that handles them.

A key bound to two different things in the same place is flagged: Qt treats
such a key as ambiguous and fires neither.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QPushButton, QApplication, QLineEdit, QLabel, QMenu, QToolBar,
    QDockWidget, QMdiSubWindow,
)
from PyQt6.QtGui import QAction, QShortcut, QKeySequence, QColor, QFont
from PyQt6.QtCore import Qt, QCoreApplication, QTimer, QT_TRANSLATE_NOOP

from setiastro.saspro.shortcut_registry import declared_keys, declared_gestures

_SEP = " ▸ "            # "File ▸ Save As ▸ FITS"
_WARN = "⚠ "

# Groups that are live whenever the main window is: menu and toolbar actions,
# and main-window shortcuts. A key used twice across these -- or once here and
# once in an image view, which is active inside the same window -- is ambiguous.
_APP_SCOPE = "app"
_GLOBAL = QT_TRANSLATE_NOOP("CheatSheet", "Global")
_IMAGE_VIEW = QT_TRANSLATE_NOOP("CheatSheet", "Image view")


def _tr(text: str) -> str:
    return QCoreApplication.translate("CheatSheet", text)


def _clean_text(text: str) -> str:
    """Menu text as a reader sees it: no mnemonics, ellipses or line breaks."""
    if not text:
        return ""
    s = str(text).replace("&&", "\x00").replace("&", "").replace("\x00", "&")
    s = s.replace("…", "").replace("...", "")
    s = s.replace("→", "->").replace("←", "<-")
    return " ".join(s.split())


def _key_text(seq) -> str:
    if not isinstance(seq, QKeySequence):
        seq = QKeySequence(str(seq))
    return seq.toString(QKeySequence.SequenceFormat.NativeText).strip()


def _typeable(seq: QKeySequence) -> bool:
    """False for media/browser keys. Qt's standard Close binding, for one,
    includes the Close media key next to Ctrl+F4 and Ctrl+W."""
    try:
        key = seq[0].key()
        k = int(key.value) if hasattr(key, "value") else int(key)
    except Exception:
        return True
    return not (0x01000060 <= k <= 0x0100FFFF)


def _seqs(act: QAction) -> list:
    seqs = list(act.shortcuts() or [])
    if not seqs and not act.shortcut().isEmpty():
        seqs = [act.shortcut()]
    return [s for s in seqs if not s.isEmpty() and _typeable(s)]


def _action_tip(act: QAction) -> str:
    """The longer description, for a row's tooltip; menu text stays the label."""
    try:
        tip = (act.statusTip() or act.toolTip() or "").strip()
    except Exception:
        tip = ""
    return "" if tip.replace("&", "") == (act.text() or "").replace("&", "") else tip


def _action_label(act: QAction) -> str:
    lab = act.property("cheat_label")
    return _clean_text(str(lab)) if lab else _clean_text(act.text())


def _owner_group(obj, main_window) -> str:
    """Where a shortcut lives: an explicit cheat_context, else the dock it sits
    in, else Global for the main window itself."""
    ctx = obj.property("cheat_context")
    if ctx:
        return _tr(str(ctx))
    w = obj.parent()
    while w is not None and w is not main_window:
        if isinstance(w, QMdiSubWindow):
            return _tr(_IMAGE_VIEW)    # e.g. Qt's own Close (Ctrl+F4, Ctrl+W)
        if isinstance(w, QDockWidget):
            return _clean_text(w.windowTitle()) or _tr(_GLOBAL)
        w = w.parent()
    return _tr(_GLOBAL)


@dataclass
class KeyRow:
    keys: str
    label: str
    group: str
    scope: str            # _APP_SCOPE, or the group name for local shortcuts
    ident: object         # the QAction / QShortcut, or the label for declared keys
    conflict: str = ""    # the other bindings, when the key is ambiguous
    menu: bool = False    # came from a menu: keep menu order when shown
    tip: str = ""         # the action's status tip, shown on hover


def collect_keyboard_rows(main_window) -> tuple[list, list]:
    """(rows, group order). Group order is menu bar order, then the rest."""
    rows: list[KeyRow] = []
    order: list[str] = []
    seen_actions: set = set()

    def note_group(g):
        if g not in order:
            order.append(g)

    # 1) Menus, in menu bar order, with the submenu path in the label.
    def walk(menu: QMenu, top: str, path: list):
        for act in menu.actions():
            if act.isSeparator():
                continue
            sub = act.menu()
            if sub is not None:
                walk(sub, top, path + [_clean_text(act.text())])
                continue
            if id(act) in seen_actions:
                continue      # one action reached from two submenus: list it once
            seen_actions.add(id(act))
            if not act.isVisible():
                continue      # a hidden action's shortcut does not fire
            for seq in _seqs(act):
                rows.append(KeyRow(_key_text(seq), _SEP.join(path + [_action_label(act)]),
                                   top, _APP_SCOPE, act, menu=True, tip=_action_tip(act)))
                note_group(top)

    try:
        mb = main_window.menuBar()
        for top_act in mb.actions():
            m = top_act.menu()
            top = _clean_text(top_act.text())
            if m is not None:
                walk(m, top, [])
            else:
                seen_actions.add(id(top_act))
                for seq in _seqs(top_act):
                    rows.append(KeyRow(_key_text(seq), _action_label(top_act),
                                       _tr(_GLOBAL), _APP_SCOPE, top_act))
    except Exception:
        pass

    # 2) Toolbar-only actions.
    for tb in main_window.findChildren(QToolBar):
        for act in tb.actions():
            if id(act) in seen_actions or act.isSeparator() or not act.isVisible():
                continue
            seen_actions.add(id(act))
            for seq in _seqs(act):
                g = _tr("Toolbar")
                rows.append(KeyRow(_key_text(seq), _action_label(act), g, _APP_SCOPE, act,
                                   tip=_action_tip(act)))
                note_group(g)

    # 3) Any other action with a key: main-window actions are app-wide, ones
    #    inside a dock belong to that dock.
    for act in main_window.findChildren(QAction):
        if id(act) in seen_actions or not act.isVisible():
            continue
        # An action fires its shortcut only once it is added to a widget. One
        # that merely exists as a child of the window is dead, and listing it
        # (or flagging a clash with it) would be describing a key that does
        # nothing -- which is exactly what a lone Copy View action once did.
        try:
            if not act.associatedObjects():
                continue
        except Exception:
            pass
        seqs = _seqs(act)
        if not seqs or not _action_label(act):
            continue
        seen_actions.add(id(act))
        g = _owner_group(act, main_window)
        scope = _APP_SCOPE if g == _tr(_GLOBAL) else g
        for seq in seqs:
            rows.append(KeyRow(_key_text(seq), _action_label(act), g, scope, act,
                               tip=_action_tip(act)))

    # 4) Labelled QShortcuts that exist right now (views that are open, docks).
    for sc in main_window.findChildren(QShortcut):
        lab = sc.property("cheat_label")
        if not lab or sc.key().isEmpty():
            continue
        g = _owner_group(sc, main_window)
        scope = _APP_SCOPE if g == _tr(_GLOBAL) else g
        rows.append(KeyRow(_key_text(sc.key()), _tr(str(lab)), g, scope, str(lab)))

    # 5) Declared keys -- covers keyPressEvent handlers, and view shortcuts
    #    before any image is open.
    for keys, label, context in declared_keys():
        g = _tr(context)
        scope = _APP_SCOPE if context == _GLOBAL else g
        rows.append(KeyRow(_key_text(keys), _tr(label), g, scope, label))

    # De-duplicate: the same binding found twice (an open view's QShortcut and
    # its declaration, an action on a menu and a toolbar).
    out, seen = [], set()
    for r in rows:
        k = (r.keys.casefold(), r.label.casefold(), r.group)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    rows = out

    # Ambiguity: one key, two different things, live at the same time.
    def live_scope(r):
        # An image view's keys are live inside the main window, alongside
        # every app-wide action.
        return _APP_SCOPE if r.scope in (_APP_SCOPE, _tr(_IMAGE_VIEW)) else r.scope

    by_key: dict = {}
    for r in rows:
        by_key.setdefault((r.keys.casefold(), live_scope(r)), []).append(r)
    for group_rows in by_key.values():
        idents = {id(r.ident) if not isinstance(r.ident, str) else r.ident.casefold()
                  for r in group_rows}
        labels = {r.label.casefold() for r in group_rows}
        if len(idents) > 1 and len(labels) > 1:
            for r in group_rows:
                others = sorted({f"{o.group}{_SEP}{o.label}" for o in group_rows if o is not r})
                r.conflict = "; ".join(others)

    # Menus first in menu bar order, then Global, Toolbar, Image view, rest A-Z.
    tail = [g for g in (_tr(_GLOBAL), _tr("Toolbar"), _tr(_IMAGE_VIEW))]
    menus = [g for g in order if g not in tail]
    rest = sorted({r.group for r in rows} - set(menus) - set(tail), key=str.casefold)
    group_order = menus + [g for g in tail if any(r.group == g for r in rows)] + rest
    return rows, group_order


def collect_gesture_rows() -> list:
    """(gesture, context, effect), translated, in declaration order."""
    return [(_tr(g), _tr(c), _tr(e)) for g, c, e in declared_gestures()]


class CheatSheetDialog(QDialog):
    """Keyboard shortcuts and mouse gestures, grouped by where they apply."""

    def __init__(self, parent, keyboard_rows, gesture_rows, group_order=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Keyboard Shortcut Cheat Sheet"))
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.resize(860, 640)

        self._keyboard_rows = list(keyboard_rows)
        self._gesture_rows = list(gesture_rows)
        self._group_order = list(group_order or [])

        top = QVBoxLayout(self)

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText(self.tr("Filter by key or action..."))
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        top.addWidget(self.filter_edit)

        tabs = QTabWidget(self)
        self.tree_keys = self._make_tree([self.tr("Shortcut"), self.tr("Action")])
        self.tree_mouse = self._make_tree([self.tr("Gesture"), self.tr("Effect")])
        tabs.addTab(self.tree_keys, self.tr("Keyboard"))
        tabs.addTab(self.tree_mouse, self.tr("Mouse && Drag"))
        top.addWidget(tabs, 1)

        self._fill_keys()
        self._fill_mouse()

        note = QLabel(self.tr(
            "Read from the running app: keys you assign to commands or scripts "
            "appear here too. Keys that only work inside one tool are listed "
            "in that tool."), self)
        note.setWordWrap(True)
        note.setEnabled(False)
        top.addWidget(note)

        btns = QHBoxLayout()
        btns.addStretch(1)
        b_copy = QPushButton(self.tr("Copy"))
        b_copy.clicked.connect(self._copy_all)
        self._b_copy = b_copy
        b_close = QPushButton(self.tr("Close"))
        b_close.clicked.connect(self.accept)
        btns.addWidget(b_copy)
        btns.addWidget(b_close)
        top.addLayout(btns)

    # ---- building -----------------------------------------------------------
    def _make_tree(self, headers) -> QTreeWidget:
        t = QTreeWidget(self)
        t.setColumnCount(2)
        t.setHeaderLabels(headers)
        t.setRootIsDecorated(False)
        t.setUniformRowHeights(True)
        t.setAlternatingRowColors(True)
        t.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        t.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        t.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        return t

    def _group_item(self, tree: QTreeWidget, title: str) -> QTreeWidgetItem:
        it = QTreeWidgetItem(tree, [title])
        f = QFont(it.font(0))
        f.setBold(True)
        it.setFont(0, f)
        it.setFirstColumnSpanned(True)
        it.setFlags(Qt.ItemFlag.ItemIsEnabled)
        it.setExpanded(True)
        return it

    def _groups_in_order(self, groups):
        known = [g for g in self._group_order if g in groups]
        return known + sorted(set(groups) - set(known), key=str.casefold)

    def _fill_keys(self):
        t = self.tree_keys
        by_group: dict = {}
        for r in self._keyboard_rows:
            by_group.setdefault(r.group, []).append(r)
        warn = QColor(217, 130, 43)
        for g in self._groups_in_order(by_group):
            parent = self._group_item(t, g)
            rows = by_group[g]
            # Menu groups keep menu order; the others read best by key.
            if not all(r.menu for r in rows):
                rows = sorted(rows, key=lambda r: (r.keys.casefold(), r.label.casefold()))
            for r in rows:
                it = QTreeWidgetItem(parent, [(_WARN + r.keys) if r.conflict else r.keys, r.label])
                if r.conflict:
                    tip = self.tr("Also assigned to: {0}\nQt treats an ambiguous key as "
                                  "unassigned, so it does nothing.").format(r.conflict)
                    for c in (0, 1):
                        it.setForeground(c, warn)
                        it.setToolTip(c, tip)
                elif r.tip:
                    it.setToolTip(1, r.tip)
        t.expandAll()

    def _fill_mouse(self):
        t = self.tree_mouse
        by_ctx: dict = {}
        for gesture, ctx, effect in self._gesture_rows:
            by_ctx.setdefault(ctx, []).append((gesture, effect))
        for ctx, items in by_ctx.items():
            parent = self._group_item(t, ctx)
            for gesture, effect in items:
                QTreeWidgetItem(parent, [gesture, effect])
        t.expandAll()

    # ---- filter / copy ------------------------------------------------------
    def _apply_filter(self, text: str):
        needle = (text or "").strip().casefold()
        for tree in (self.tree_keys, self.tree_mouse):
            for i in range(tree.topLevelItemCount()):
                grp = tree.topLevelItem(i)
                group_hit = bool(needle) and needle in grp.text(0).casefold()
                any_shown = False
                for j in range(grp.childCount()):
                    ch = grp.child(j)
                    hit = (not needle or group_hit
                           or needle in ch.text(0).casefold()
                           or needle in ch.text(1).casefold())
                    ch.setHidden(not hit)
                    any_shown = any_shown or hit
                grp.setHidden(not any_shown)

    def _copy_all(self):
        """Copy everything to the clipboard as plain text, grouped."""
        lines = ["== " + self.tr("Keyboard") + " =="]
        by_group: dict = {}
        for r in self._keyboard_rows:
            by_group.setdefault(r.group, []).append(r)
        for g in self._groups_in_order(by_group):
            lines.append("")
            lines.append(f"[{g}]")
            for r in by_group[g]:
                flag = "  (" + self.tr("conflict") + ")" if r.conflict else ""
                lines.append(f"  {r.keys:22}  {r.label}{flag}")
        lines.append("")
        lines.append("== " + self.tr("Mouse & Drag") + " ==")
        by_ctx: dict = {}
        for gesture, ctx, effect in self._gesture_rows:
            by_ctx.setdefault(ctx, []).append((gesture, effect))
        for ctx, items in by_ctx.items():
            lines.append("")
            lines.append(f"[{ctx}]")
            for gesture, effect in items:
                lines.append(f"  {gesture:44}  {effect}")
        QApplication.clipboard().setText("\n".join(lines))
        self._b_copy.setText(self.tr("Copied"))
        QTimer.singleShot(1500, lambda: self._b_copy.setText(self.tr("Copy")))


# Legacy alias for backward compatibility
_CheatSheetDialog = CheatSheetDialog
