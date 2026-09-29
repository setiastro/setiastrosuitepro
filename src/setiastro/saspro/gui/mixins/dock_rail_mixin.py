# pro/gui/mixins/dock_rail_mixin.py
"""
Dock rail mixin for AstroSuiteProMainWindow.

Adds thin "activity bar" rails pinned to the left / right / bottom TOOLBAR
areas (which sit *outside* the dock areas), so each rail is the outermost
edge strip. Every rail button is a dock's own toggleViewAction(): checkable,
already wired to visibility, so the button lights when its panel is shown and
dims when hidden — and clicking a dim button brings the panel back. That is
the "little tab to pull it out" with none of the min-size fighting you get
from trying to shrink a native QDockWidget to a sliver: collapsing genuinely
hide()s the dock (min-size problem gone), and the rail is what reopens it.

Each rail also gets a chevron head button that collapses / restores that whole
edge at once, remembering which panels were open so restore brings back the
same subset.

Design notes
------------
* Buttons are assigned to an edge by *intent* (see _DEFAULT_RAIL_PLAN), but a
  dock the user drags to another edge is re-homed live via dockLocationChanged.
* Icons: if you have real QIcons, expose `_rail_icon_for(object_name) -> QIcon`
  on the main window (or drop resource keys into _RAIL_ICON_HINTS) and they win.
  Otherwise each button gets a crisp, DPI-aware initials badge so the rail looks
  finished out of the box with no art assets.
* The checked-state accent reuses your teal (#3dbf9f, same as the Explorer
  active-doc bar) for visual consistency.
* Rails have objectNames, so they participate in saveState()/restoreState()
  like any other toolbar — no extra persistence code needed.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import (
    Qt, QSize, QRect, QPointF, QPoint, QMimeData,
)
from PyQt6.QtGui import (
    QAction, QIcon, QPixmap, QPainter, QPen, QPalette, QPolygonF, QDrag,
)
from PyQt6.QtWidgets import QToolBar, QDockWidget, QApplication, QToolButton


# Fallback glyph size if the toolbar icon-size setting can't be read. The live
# size follows QSettings "toolbar/icon_size" via _rail_icon_size().
_RAIL_ICON_PX = 24

# edge -> Qt toolbar area (where the rail itself lives)
_RAIL_AREAS = {
    "left":   Qt.ToolBarArea.LeftToolBarArea,
    "right":  Qt.ToolBarArea.RightToolBarArea,
    "bottom": Qt.ToolBarArea.BottomToolBarArea,
}

# edge -> Qt dock area (where the panel gets moved when its button is dragged)
_DOCK_AREAS = {
    "left":   Qt.DockWidgetArea.LeftDockWidgetArea,
    "right":  Qt.DockWidgetArea.RightDockWidgetArea,
    "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
}

# Drag payload: the dragged dock's key (objectName or title).
_RAIL_MIME = "application/x-sas-rail-dock"

# Which edge each dock starts on. Keyed by attribute name on the main window;
# missing docks (e.g. resource_monitor_dock == None) are skipped gracefully.
_DEFAULT_RAIL_PLAN = [
    ("left",   "explorer_dock"),
    ("right",  "header_dock"),
    ("right",  "layers_dock"),
    ("right",  "resource_monitor_dock"),
    ("bottom", "console_dock"),
    ("bottom", "status_log_dock"),
    ("bottom", "system_log_dock"),
]

# Optional per-dock label overrides (objectName -> short text or emoji to paint
# when there is no real icon). Leave empty to use auto-initials. This is also
# the natural place to switch a specific panel to a resource-key icon later.
_RAIL_ICON_HINTS: dict[str, str] = {
    # "ExplorerDock": "Ex",
    # "HeaderViewerDock": "H",
}

# Real panel icons (resources.py). Primary lookup is by dock objectName.
_PANEL_ICON_FILES: dict[str, str] = {
    "ExplorerDock":        "explorerpanel",
    "HeaderViewerDock":    "headerpanel",
    "LayersDock":          "layerspanel",
    "ResourceMonitorDock": "monitorpanel",
    "ConsoleDock":         "consolepanel",
    "StatusLogDock":       "stackingpanel",   # "Stacking Log"
    "WindowShelfDock":     "minimizedpanel",  # "Minimized Views"
    # System Log and Command Search have no stable objectName here; matched by
    # title below.
}

# Fallback for docks whose objectName we can't rely on: first title substring
# (lower-cased) that matches wins. Order matters — more specific first so
# "System Monitor" doesn't get swallowed by a bare "system".
_PANEL_ICON_TITLE_HINTS: list[tuple[str, str]] = [
    ("command",    "searchpanel"),
    ("search",     "searchpanel"),
    ("system log", "systempanel"),
    ("minimized",  "minimizedpanel"),
    ("stacking",   "stackingpanel"),
    ("console",    "consolepanel"),
    ("explorer",   "explorerpanel"),
    ("header",     "headerpanel"),
    ("layers",     "layerspanel"),
    ("monitor",    "monitorpanel"),
]


def _edge_for_dock_area(area: "Qt.DockWidgetArea") -> str | None:
    if area == Qt.DockWidgetArea.LeftDockWidgetArea:
        return "left"
    if area == Qt.DockWidgetArea.RightDockWidgetArea:
        return "right"
    if area == Qt.DockWidgetArea.BottomDockWidgetArea:
        return "bottom"
    # TopDockWidgetArea / NoDockWidgetArea / floating -> leave the button put
    return None


def _initials(title: str) -> str:
    """Two-letter badge text from a panel title. 'Header Viewer' -> 'HV'."""
    toks = [t for t in "".join(c if c.isalnum() else " " for c in (title or "")).split() if t]
    if not toks:
        return "•"
    if len(toks) == 1:
        return toks[0][:2].capitalize()
    return (toks[0][0] + toks[1][0]).upper()


class _RailToolBar(QToolBar):
    """A rail that accepts drops of dock buttons dragged from another rail.

    Drops are delegated to the owner (main window) which physically moves the
    panel to this rail's edge; the button then follows via the normal re-home
    path. Non-rail drags fall through to default handling.
    """

    def __init__(self, edge: str, owner, title: str, parent=None):
        super().__init__(title, parent)
        self._rail_edge = edge
        self._rail_owner = owner
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(_RAIL_MIME):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(_RAIL_MIME):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasFormat(_RAIL_MIME):
            key = bytes(md.data(_RAIL_MIME)).decode("utf-8", "ignore")
            try:
                self._rail_owner._on_rail_drop(key, self._rail_edge)
            except Exception:
                pass
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


class _RailButton(QToolButton):
    """A rail button that owns its drag-and-drop. Driven by the dock's
    toggleViewAction via setDefaultAction, so clicking still toggles the panel
    and checked-state/icon/tooltip all come from the action. Press-and-drag
    past the drag threshold starts a QDrag; a plain click falls through to the
    action. Also a drop target, since a drop usually lands on a button.

    The logic lives in the widget (not an event filter) so it can't be silently
    lost when the toolbar recreates or re-homes its buttons.
    """

    def __init__(self, owner, dock_key: str, parent=None):
        super().__init__(parent)
        self._owner = owner
        self._dock_key = dock_key
        self._press: QPoint | None = None
        self.setAcceptDrops(True)
        self.setAutoRaise(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

    # --- drag out ---
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if (self._press is not None
                and (e.buttons() & Qt.MouseButton.LeftButton)
                and (e.position().toPoint() - self._press).manhattanLength()
                    >= QApplication.startDragDistance()):
            self._press = None
            try:
                self._owner._start_rail_drag(self, self._dock_key)
            except Exception:
                pass
            return  # took over the gesture; don't let the button treat it as a click
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._press = None
        super().mouseReleaseEvent(e)

    # --- drop in ---
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(_RAIL_MIME):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(_RAIL_MIME):
            e.acceptProposedAction()
        else:
            super().dragMoveEvent(e)

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasFormat(_RAIL_MIME):
            key = bytes(md.data(_RAIL_MIME)).decode("utf-8", "ignore")
            try:
                edge = self._owner._edge_of_key(self._dock_key)
                if edge:
                    self._owner._on_rail_drop(key, edge)
            except Exception:
                pass
            e.acceptProposedAction()
        else:
            super().dropEvent(e)


class DockRailMixin:
    """Collapsible edge rails built from dock toggleViewAction()s."""

    # ---- public entry point -------------------------------------------------
    def _init_dock_rails(self):
        """Build the left/right/bottom rails and populate them.

        Call once from __init__ AFTER every _init_*_dock() has run and the
        toolbar exists, and BEFORE restore_main_window_state() so a saved
        layout can position the rails.
        """
        if getattr(self, "_dock_rails", None):
            return  # already built

        self._dock_rails: dict[str, QToolBar] = {}
        self._rail_actions: dict[str, QAction] = {}          # objectName -> dock toggle action
        self._rail_buttons: dict[str, _RailButton] = {}      # objectName -> live button
        self._rail_widget_actions: dict[str, QAction] = {}   # objectName -> toolbar widget-action
        self._rail_collapse_actions: dict[str, QAction] = {}  # edge -> chevron action
        self._rail_membership: dict[str, list[str]] = {e: [] for e in _RAIL_AREAS}
        self._rail_snapshot: dict[str, list[str]] = {}       # edge -> names hidden by collapse
        self._rail_busy = False                              # re-entrancy guard
        self._rail_icon_px = self._rail_icon_size()          # follows toolbar/icon_size

        for edge, area in _RAIL_AREAS.items():
            self._make_rail(edge, area)

        for edge, attr in _DEFAULT_RAIL_PLAN:
            dock = getattr(self, attr, None)
            if isinstance(dock, QDockWidget):
                self._assign_dock_to_rail(dock, edge)

        # Sweep: any dock not covered by the plan above still gets a rail
        # button, placed on the rail matching its current edge (falling back
        # to bottom). This auto-covers the shelf ("Minimized Views"), Command
        # Search, and any panel added later without touching this file.
        assigned = {n for names in self._rail_membership.values() for n in names}
        for dock in self._all_dockwidgets():
            key = self._dock_key(dock)
            if key in assigned:
                continue
            edge = self._edge_for_dock(dock) or "bottom"
            self._assign_dock_to_rail(dock, edge)
            assigned.add(key)

        self._refresh_all_rail_collapse_icons()
        self._refresh_rail_visibility()

    def _repaint_dock_rail_icons(self):
        """Rebuild painted glyphs + rail stylesheets. Call this from your theme
        apply path so badges/chevrons track light/dark palette changes."""
        if not getattr(self, "_dock_rails", None):
            return
        for edge, names in self._rail_membership.items():
            rail = self._dock_rails.get(edge)
            if rail is not None:
                rail.setStyleSheet(self._rail_stylesheet(edge))
            for name in names:
                act = self._rail_actions.get(name)
                dock = self._dock_by_object_name(name)
                if act is not None and dock is not None:
                    ic = self._rail_icon(dock)
                    if ic is not None and not ic.isNull():
                        act.setIcon(ic)
        self._refresh_all_rail_collapse_icons()

    def _rail_icon_size(self) -> int:
        """Rail glyph size in px, tied to the shared toolbar/icon_size setting
        (same value the DraggableToolBars use), clamped to the Settings range."""
        s = getattr(self, "settings", None)
        try:
            if s is None:
                from PyQt6.QtCore import QSettings
                s = QSettings()
            n = int(s.value("toolbar/icon_size", 24, type=int))
        except Exception:
            n = 24
        return max(16, min(64, n))

    def _apply_rail_icon_size(self):
        """Re-read toolbar/icon_size and resize every rail (icons + painted
        glyphs). Safe to call before the rails exist. Wired from the main
        window's _apply_toolbar_icon_size so the Settings slider drives both."""
        if not getattr(self, "_dock_rails", None):
            return
        self._rail_icon_px = self._rail_icon_size()
        qs = QSize(self._rail_icon_px, self._rail_icon_px)
        for rail in self._dock_rails.values():
            rail.setIconSize(qs)
        for btn in self._rail_buttons.values():
            try:
                btn.setIconSize(qs)   # addWidget buttons don't inherit rail iconSize
            except Exception:
                pass
        self._repaint_dock_rail_icons()   # repaint badges/chevrons at the new px

    # ---- construction -------------------------------------------------------
    def _make_rail(self, edge: str, area) -> QToolBar:
        rail = _RailToolBar(edge, self, self.tr("Panel rail ({0})").format(edge), self)
        rail.setObjectName(f"DockRail_{edge}")
        rail.setMovable(False)
        rail.setFloatable(False)
        rail.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        rail.setIconSize(QSize(self._rail_icon_px, self._rail_icon_px))
        rail.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        rail.setStyleSheet(self._rail_stylesheet(edge))
        # keep the rail out of the toolbar-area right-click menu
        try:
            rail.toggleViewAction().setVisible(False)
        except Exception:
            pass

        self.addToolBar(area, rail)
        self._dock_rails[edge] = rail

        # collapse/restore head control
        act = QAction(self._chevron_icon(edge, collapsed=False), "", self)
        act.setToolTip(self.tr("Collapse panels"))
        act.triggered.connect(lambda _=False, e=edge: self._toggle_collapse_edge(e))
        rail.addAction(act)
        rail.addSeparator()
        self._rail_collapse_actions[edge] = act
        return rail

    def _assign_dock_to_rail(self, dock: QDockWidget, edge: str):
        """Put (or move) a dock's button onto the rail for `edge`. The button is
        a _RailButton driven by the dock's toggle action; on a move we destroy
        the old button and build a fresh one on the target rail."""
        rail = self._dock_rails.get(edge)
        if rail is None:
            return
        name = dock.objectName() or dock.windowTitle()

        act = self._rail_actions.get(name)
        if act is None:
            act = dock.toggleViewAction()   # same QAction each call; Qt keeps it synced
            act.setToolTip(dock.windowTitle() or name)
            ic = self._rail_icon(dock)
            if ic is not None and not ic.isNull():
                act.setIcon(ic)
            self._rail_actions[name] = act
            # Resolve the edge live (a dock can be re-homed after this connect).
            try:
                dock.visibilityChanged.connect(
                    lambda _v, k=name: self._on_edge_visibility_changed(self._edge_of_key(k))
                )
            except Exception:
                pass
            try:
                dock.dockLocationChanged.connect(
                    lambda a, d=dock: self._on_dock_location_changed(d, a)
                )
            except Exception:
                pass

        # Tear down any existing button for this dock, wherever it lives.
        old_btn = self._rail_buttons.pop(name, None)
        old_wa = self._rail_widget_actions.pop(name, None)
        if old_wa is not None:
            for r in self._dock_rails.values():
                try:
                    r.removeAction(old_wa)
                except Exception:
                    pass
        if old_btn is not None:
            try:
                old_btn.deleteLater()
            except Exception:
                pass

        # membership bookkeeping
        for e in self._rail_membership:
            if name in self._rail_membership[e]:
                self._rail_membership[e].remove(name)
        self._rail_membership.setdefault(edge, [])
        if name not in self._rail_membership[edge]:
            self._rail_membership[edge].append(name)

        # Build a fresh draggable button on the target rail.
        btn = _RailButton(self, name, rail)
        btn.setDefaultAction(act)           # toggle, checked-state, icon, tooltip
        btn.setIconSize(QSize(self._rail_icon_px, self._rail_icon_px))
        wa = rail.addWidget(btn)
        self._rail_buttons[name] = btn
        self._rail_widget_actions[name] = wa

    # ---- live re-homing / state sync ---------------------------------------
    def _on_dock_location_changed(self, dock: QDockWidget, area):
        edge = _edge_for_dock_area(area)
        if edge is None:
            return
        name = dock.objectName() or dock.windowTitle()
        if name in self._rail_membership.get(edge, []):
            return  # already on the right rail
        self._assign_dock_to_rail(dock, edge)
        self._refresh_all_rail_collapse_icons()
        self._refresh_rail_visibility()

    def _on_edge_visibility_changed(self, edge: str):
        if self._rail_busy or not edge:
            return
        self._refresh_edge_collapse_icon(edge)

    # ---- collapse / restore -------------------------------------------------
    def _toggle_collapse_edge(self, edge: str):
        docks = self._docks_for_edge(edge)
        if not docks:
            return
        any_visible = any(d.isVisible() for d in docks)

        self._rail_busy = True
        try:
            if any_visible:
                self._rail_snapshot[edge] = [
                    d.objectName() for d in docks if d.isVisible()
                ]
                for d in docks:
                    if d.isVisible():
                        d.setVisible(False)
            else:
                snap = set(self._rail_snapshot.get(edge, []))
                targets = [d for d in docks if d.objectName() in snap] if snap else []
                if not targets:
                    targets = docks
                for d in targets:
                    d.setVisible(True)
        finally:
            self._rail_busy = False

        self._refresh_edge_collapse_icon(edge)

    # ---- glyph / style refresh ---------------------------------------------
    def _refresh_all_rail_collapse_icons(self):
        for edge in self._dock_rails:
            self._refresh_edge_collapse_icon(edge)

    def _refresh_edge_collapse_icon(self, edge: str):
        act = self._rail_collapse_actions.get(edge)
        if act is None:
            return
        docks = self._docks_for_edge(edge)
        any_visible = any(d.isVisible() for d in docks) if docks else False
        act.setIcon(self._chevron_icon(edge, collapsed=not any_visible))
        act.setToolTip(self.tr("Collapse panels") if any_visible
                       else self.tr("Show panels"))
        act.setEnabled(bool(docks))

    def _refresh_rail_visibility(self):
        # Hide a rail with no panels (just the chevron) -- but while a rail
        # drag is in flight, reveal all three so an emptied edge is still a
        # reachable drop target.
        dragging = getattr(self, "_rail_dragging", False)
        for edge, rail in self._dock_rails.items():
            rail.setVisible(bool(self._rail_membership.get(edge)) or dragging)

    # ---- drag & drop between rails -----------------------------------------
    def _start_rail_drag(self, button, dock_key: str):
        drag = QDrag(button)
        mime = QMimeData()
        mime.setData(_RAIL_MIME, dock_key.encode("utf-8"))
        drag.setMimeData(mime)

        ic = button.icon()
        if ic is not None and not ic.isNull():
            pm = ic.pixmap(QSize(self._rail_icon_px, self._rail_icon_px))
            drag.setPixmap(pm)
            drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))

        # reveal all rails as drop zones for the duration of the drag
        self._rail_dragging = True
        self._refresh_rail_visibility()
        try:
            drag.exec(Qt.DropAction.MoveAction)
        finally:
            self._rail_dragging = False
            try:
                button.setDown(False)   # clear stuck pressed state
            except Exception:
                pass
            self._refresh_rail_visibility()

    def _edge_of_key(self, dock_key: str) -> str | None:
        """Which rail a dock's button currently lives on, or None."""
        for e, names in self._rail_membership.items():
            if dock_key in names:
                return e
        return None

    def _on_rail_drop(self, dock_key: str, target_edge: str):
        dock = self._dock_by_object_name(dock_key)
        if dock is None:
            return
        cur = next((e for e, names in self._rail_membership.items()
                    if dock_key in names), None)
        if cur == target_edge:
            return  # same edge -> nothing to do (reordering not handled)

        self._move_dock_to_edge(dock, target_edge)
        # move usually re-homes the button via dockLocationChanged; make sure.
        if dock_key not in self._rail_membership.get(target_edge, []):
            self._assign_dock_to_rail(dock, target_edge)
        self._refresh_all_rail_collapse_icons()
        self._refresh_rail_visibility()

    def _move_dock_to_edge(self, dock: QDockWidget, edge: str):
        """Physically relocate a panel to `edge`, in whichever window currently
        hosts it, preserving its shown/hidden state."""
        area = _DOCK_AREAS.get(edge)
        if area is None:
            return
        host = None
        for win in (self, getattr(self, "dock_host", None)):
            if win is None:
                continue
            try:
                if win.dockWidgetArea(dock) != Qt.DockWidgetArea.NoDockWidgetArea:
                    host = win
                    break
            except Exception:
                pass
        if host is None:
            host = self

        was_visible = dock.isVisible()
        self._rail_busy = True          # don't let the show/hide churn flip chevrons
        try:
            host.addDockWidget(area, dock)   # docks it here (un-floats if floating)
            dock.setVisible(was_visible)
        finally:
            self._rail_busy = False

    # ---- helpers ------------------------------------------------------------
    def _docks_for_edge(self, edge: str) -> list[QDockWidget]:
        names = self._rail_membership.get(edge, [])
        by_name = self._dock_name_map()
        out = []
        for n in names:
            d = by_name.get(n)
            if isinstance(d, QDockWidget):
                out.append(d)
        return out

    def _all_dockwidgets(self) -> list[QDockWidget]:
        """Every dock panel that currently exists, wherever it's parented
        (main window, or the secondary dock host). Deduped by identity.

        Discovery is by findChildren rather than a fixed name list, so panels
        the DockMixin doesn't enumerate -- the shelf, Command Search, anything
        added later -- are still seen by the rails.
        """
        found: list[QDockWidget] = []
        seen: set[int] = set()

        def _collect(widget):
            if widget is None:
                return
            try:
                for d in widget.findChildren(QDockWidget):
                    if id(d) not in seen:
                        seen.add(id(d))
                        found.append(d)
            except Exception:
                pass

        _collect(self)
        _collect(getattr(self, "dock_host", None))

        # Backstop: anything DockMixin knows about explicitly.
        if hasattr(self, "_all_known_docks"):
            try:
                for d in self._all_known_docks():
                    if isinstance(d, QDockWidget) and id(d) not in seen:
                        seen.add(id(d))
                        found.append(d)
            except Exception:
                pass
        return found

    def _dock_key(self, dock: QDockWidget) -> str:
        # Must match the key _assign_dock_to_rail uses for membership/actions.
        return dock.objectName() or dock.windowTitle()

    def _edge_for_dock(self, dock: QDockWidget) -> str | None:
        """Best-effort current edge for a dock, checking whichever window
        hosts it. None if floating / undockable / area unknown."""
        if dock is None:
            return None
        try:
            if dock.isFloating():
                return None
        except Exception:
            pass
        for win in (self, getattr(self, "dock_host", None)):
            if win is None:
                continue
            try:
                area = win.dockWidgetArea(dock)
            except Exception:
                continue
            edge = _edge_for_dock_area(area)
            if edge is not None:
                return edge
        return None

    def _dock_name_map(self) -> dict[str, QDockWidget]:
        return {self._dock_key(d): d for d in self._all_dockwidgets()}

    def _dock_by_object_name(self, name: str) -> QDockWidget | None:
        return self._dock_name_map().get(name)

    def _rail_icon(self, dock: QDockWidget) -> QIcon | None:
        name = dock.objectName() or ""
        # 1) user-provided real icons win
        hook = getattr(self, "_rail_icon_for", None)
        if callable(hook):
            try:
                ic = hook(name)
                if isinstance(ic, QIcon) and not ic.isNull():
                    return ic
            except Exception:
                pass
        # 2) bundled panel icon from resources.py
        ic = self._panel_icon(dock)
        if ic is not None and not ic.isNull():
            return ic
        # 3) an icon already set on the dock itself
        try:
            if not dock.windowIcon().isNull():
                return dock.windowIcon()
        except Exception:
            pass
        # 4) painted badge fallback
        label = _RAIL_ICON_HINTS.get(name) or _initials(dock.windowTitle() or name)
        return self._badge_icon(label)

    def _panel_icon(self, dock: QDockWidget) -> QIcon | None:
        """Resolve a dock to its bundled rail icon (by objectName, then title
        keyword). Returns None so the caller can fall back to a painted badge."""
        base = _PANEL_ICON_FILES.get(dock.objectName() or "")
        if base is None:
            title = (dock.windowTitle() or "").lower()
            for needle, fname in _PANEL_ICON_TITLE_HINTS:
                if needle in title:
                    base = fname
                    break
        if base is None:
            return None
        try:
            from setiastro.saspro.resources import get_icon_path
            path = get_icon_path(base)
            if path and os.path.exists(path):
                ic = QIcon(path)
                if not ic.isNull():
                    return ic
        except Exception:
            pass
        return None

    def _rail_stylesheet(self, edge: str) -> str:
        accent = "#3dbf9f"
        side = {"left": "border-left",
                "right": "border-right",
                "bottom": "border-bottom"}[edge]
        return f"""
        QToolBar#DockRail_{edge} {{
            background: palette(window);
            border: none;
            spacing: 2px;
            padding: 3px;
        }}
        QToolBar#DockRail_{edge} QToolButton {{
            border: none;
            {side}: 2px solid transparent;
            border-radius: 6px;
            padding: 6px;
            margin: 1px;
        }}
        QToolBar#DockRail_{edge} QToolButton:hover {{
            background: rgba(127, 127, 127, 40);
        }}
        QToolBar#DockRail_{edge} QToolButton:checked {{
            {side}: 2px solid {accent};
            background: rgba(61, 191, 159, 35);
        }}
        QToolBar#DockRail_{edge} QToolButton:checked:hover {{
            background: rgba(61, 191, 159, 60);
        }}
        """

    # ---- painters (DPI-aware) ----------------------------------------------
    def _blank_pixmap(self) -> tuple[QPixmap, float]:
        px = getattr(self, "_rail_icon_px", _RAIL_ICON_PX)
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        pm = QPixmap(int(px * dpr), int(px * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        return pm, dpr

    def _glyph_color(self):
        return self.palette().color(QPalette.ColorRole.WindowText)

    def _badge_icon(self, text: str) -> QIcon:
        px = getattr(self, "_rail_icon_px", _RAIL_ICON_PX)
        pm, _ = self._blank_pixmap()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        f = p.font()
        f.setBold(True)
        f.setPixelSize(int(px * 0.52) if len(text) >= 2 else int(px * 0.62))
        p.setFont(f)
        p.setPen(self._glyph_color())
        p.drawText(QRect(0, 0, px, px), int(Qt.AlignmentFlag.AlignCenter), text)
        p.end()
        return QIcon(pm)

    def _chevron_icon(self, edge: str, collapsed: bool) -> QIcon:
        direction = self._chevron_direction(edge, collapsed)
        px = getattr(self, "_rail_icon_px", _RAIL_ICON_PX)
        pm, _ = self._blank_pixmap()
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(self._glyph_color())
        pen.setWidthF(max(1.4, px * 0.09))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)

        cx = cy = px / 2.0
        a = px * 0.16   # arm half-length along the chevron opening
        b = px * 0.20   # depth of the point

        if direction == "left":
            pts = [(cx + b, cy - 2 * a), (cx - b, cy), (cx + b, cy + 2 * a)]
        elif direction == "right":
            pts = [(cx - b, cy - 2 * a), (cx + b, cy), (cx - b, cy + 2 * a)]
        elif direction == "up":
            pts = [(cx - 2 * a, cy + b), (cx, cy - b), (cx + 2 * a, cy + b)]
        else:  # down
            pts = [(cx - 2 * a, cy - b), (cx, cy + b), (cx + 2 * a, cy - b)]

        p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
        p.end()
        return QIcon(pm)

    @staticmethod
    def _chevron_direction(edge: str, collapsed: bool) -> str:
        # chevron points the way the panels will go when the button is clicked
        if edge == "left":
            return "right" if collapsed else "left"
        if edge == "right":
            return "left" if collapsed else "right"
        return "up" if collapsed else "down"   # bottom