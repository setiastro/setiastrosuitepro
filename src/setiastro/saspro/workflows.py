#src/setiastro/saspro/workflows.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QMimeData, QSize, QTimer
from PyQt6.QtGui import QAction, QBrush, QColor, QCursor, QDrag, QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)
from setiastro.saspro.help_support import make_help_button
WORKFLOW_MIME = "application/x-saspro-workflow-command"
WORKFLOW_SCHEMA_VERSION = 3   # v3: explicit links between lanes


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

def _new_step_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class WorkflowAlternative:
    """Another tool the user can open in place of a step's primary command."""
    command_id: str
    preset: dict = field(default_factory=dict)
    label_override: str = ""

    def to_dict(self) -> dict:
        return {
            "command_id": self.command_id,
            "preset": dict(self.preset or {}),
            "label_override": self.label_override,
        }


def normalize_workflow_alternatives(raw) -> List["WorkflowAlternative"]:
    out: List[WorkflowAlternative] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("command_id") or "").strip()
        if not cid:
            continue
        preset = item.get("preset")
        out.append(WorkflowAlternative(
            command_id=cid,
            preset=dict(preset) if isinstance(preset, dict) else {},
            label_override=str(item.get("label_override") or ""),
        ))
    return out


@dataclass
class WorkflowStep:
    id: str = field(default_factory=_new_step_id)
    command_id: str = ""
    note: str = ""
    enabled: bool = True
    kind: str = "action"   # action, note, split, merge
    lane: str = "main"
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    preset: dict = field(default_factory=dict)   # captured tool settings (from a dragged grip shortcut)
    label_override: str = ""                      # user-renamed display label
    alternatives: List[WorkflowAlternative] = field(default_factory=list)
    parent_id: str = ""                           # set on an action shown under a choice title

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "command_id": self.command_id,
            "note": self.note,
            "enabled": self.enabled,
            "kind": self.kind,
            "lane": self.lane,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "preset": dict(self.preset or {}),
            "label_override": self.label_override,
        }
        if self.alternatives:
            data["alternatives"] = [a.to_dict() for a in self.alternatives]
        if self.parent_id:
            data["parent_id"] = self.parent_id
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowStep":
        _preset = data.get("preset")
        return cls(
            id=str(data.get("id") or _new_step_id()),
            command_id=str(data.get("command_id", "")),
            note=str(data.get("note", "")),
            enabled=bool(data.get("enabled", True)),
            kind=str(data.get("kind", "action")),
            lane=str(data.get("lane", "main")),
            inputs=[str(x) for x in data.get("inputs", []) if str(x).strip()],
            outputs=[str(x) for x in data.get("outputs", []) if str(x).strip()],
            preset=dict(_preset) if isinstance(_preset, dict) else {},
            label_override=str(data.get("label_override", "")),
            alternatives=normalize_workflow_alternatives(data.get("alternatives")),
            parent_id=str(data.get("parent_id") or ""),
        )


@dataclass
class WorkflowLink:
    """A branch or a join between lanes.

    The step `src` hands its result either to the start of lane `dst_lane` (a
    branch) or to the step `dst` part-way down another lane (a join). Exactly
    one of `dst` / `dst_lane` is set. Steps in the same lane are never linked:
    the lane order already connects them.
    """
    src: str
    dst: str = ""
    dst_lane: str = ""

    def to_dict(self) -> dict:
        if self.dst:
            return {"from": self.src, "to": self.dst}
        return {"from": self.src, "to_lane": self.dst_lane}

    @classmethod
    def from_dict(cls, data) -> Optional["WorkflowLink"]:
        if not isinstance(data, dict):
            return None
        src = str(data.get("from") or "").strip()
        dst = str(data.get("to") or "").strip()
        lane = str(data.get("to_lane") or "")
        if not src or not (dst or lane.strip()):
            return None
        return cls(src=src, dst=dst, dst_lane="" if dst else lane)


@dataclass
class WorkflowDefinition:
    name: str
    description: str = ""
    lanes: List[str] = field(default_factory=lambda: ["main"])
    steps: List[WorkflowStep] = field(default_factory=list)
    schema_version: int = WORKFLOW_SCHEMA_VERSION
    # None means the workflow predates links; they are derived from the
    # steps' input/output tags the first time they are needed.
    links: Optional[List[WorkflowLink]] = None
    # Graph rows the user dragged steps to (step id -> row). Layout only:
    # a pinned step ignores how low its links would put it, but still sits
    # below the step before it in its own lane.
    graph_rows: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "lanes": list(self.lanes),
            "steps": [s.to_dict() for s in self.steps],
            "links": [link.to_dict() for link in workflow_links(self)],
        }
        rows = normalize_graph_rows(self.graph_rows, self.steps)
        if rows:
            data["graph_rows"] = rows
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowDefinition":
        # ------------------------------
        # New schema (v2+)
        # ------------------------------
        if isinstance(data, dict) and "lanes" in data:
            lanes = [str(x) for x in data.get("lanes", ["main"]) if str(x).strip()]
            if not lanes:
                lanes = ["main"]

            steps = []
            for s in data.get("steps", []):
                if isinstance(s, dict):
                    step = WorkflowStep.from_dict(s)
                    if not step.lane:
                        step.lane = "main"
                    steps.append(step)
            steps = normalize_workflow_steps(steps)

            raw_links = data.get("links")
            if isinstance(raw_links, list):
                links = [l for l in (WorkflowLink.from_dict(x) for x in raw_links) if l is not None]
            else:
                links = derive_workflow_links(steps, lanes)

            return cls(
                name=str(data.get("name", "Untitled Workflow")),
                description=str(data.get("description", "")),
                lanes=lanes,
                steps=steps,
                schema_version=int(data.get("schema_version", WORKFLOW_SCHEMA_VERSION)),
                links=normalize_workflow_links(links, steps, lanes),
                graph_rows=normalize_graph_rows(data.get("graph_rows"), steps),
            )

        # ------------------------------
        # Legacy schema (phase 1)
        # Convert old flat step list into lane "main"
        # ------------------------------
        steps = []
        for s in data.get("steps", []):
            if isinstance(s, dict) and s.get("command_id"):
                steps.append(
                    WorkflowStep(
                        id=_new_step_id(),
                        command_id=str(s.get("command_id", "")),
                        note=str(s.get("note", "")),
                        enabled=bool(s.get("enabled", True)),
                        kind="action",
                        lane="main",
                        inputs=[],
                        outputs=[],
                    )
                )

        return cls(
            name=str(data.get("name", "Untitled Workflow")),
            description=str(data.get("description", "")),
            lanes=["main"],
            steps=normalize_workflow_steps(steps),
            schema_version=1,
            links=[],
        )


@dataclass
class WorkflowActionInfo:
    command_id: str
    text: str
    status_tip: str
    icon: QIcon
    category: str
    action: QAction


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _clean_action_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("&", "").strip()


def _format_preset_value(v) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        # terse: drop trailing zeros, keep it short
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "…"


def _summarize_preset(preset: dict, max_rows: int = 8) -> str:
    """Compact HTML-ish caption of a step's captured settings for the detail panel."""
    if not isinstance(preset, dict) or not preset:
        return ""
    # De-emphasize routing/plumbing keys that aren't user-facing settings.
    skip = {"family", "auto_run", "args", "op"}
    items = [(k, v) for k, v in preset.items() if k not in skip]
    if not items:
        items = list(preset.items())

    rows = []
    for k, v in items[:max_rows]:
        rows.append(f"{k}: {_format_preset_value(v)}")
    tail = len(items) - max_rows
    body = "; ".join(rows)
    if tail > 0:
        body += f"; <i>+{tail} more…</i>"
    return body


def _main_settings(main) -> object | None:
    return getattr(main, "settings", None)


def _workflows_dir(main) -> Path:
    getter_names = ("get_user_data_dir", "_user_data_dir", "user_data_dir")
    for name in getter_names:
        fn = getattr(main, name, None)
        try:
            if callable(fn):
                p = Path(fn())
                p.mkdir(parents=True, exist_ok=True)
                out = p / "workflows"
                out.mkdir(parents=True, exist_ok=True)
                return out
        except Exception:
            pass

    base = Path.home() / ".setiastro" / "saspro" / "workflows"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _find_action_by_command_id(main, command_id: str) -> Optional[QAction]:
    shortcuts = getattr(main, "shortcuts", None)
    if shortcuts is not None:
        for name in ("get_action", "action", "lookup_action"):
            fn = getattr(shortcuts, name, None)
            if callable(fn):
                try:
                    act = fn(command_id)
                    if isinstance(act, QAction):
                        return act
                except Exception:
                    pass

        for attr in ("_actions", "actions_by_id", "registry", "_registry"):
            mapping = getattr(shortcuts, attr, None)
            if isinstance(mapping, dict):
                act = mapping.get(command_id)
                if isinstance(act, QAction):
                    return act

    try:
        for act in main.findChildren(QAction):
            cid = act.property("command_id")
            if cid == command_id:
                return act
            if act.objectName() == command_id:
                return act
    except Exception:
        pass

    return None

def _discover_script_categories(main) -> Dict[str, List[WorkflowActionInfo]]:
    """
    Build workflow catalog entries from ScriptManager registry.

    Scripts are discovered recursively by ScriptManager.load_registry()
    under SASpro/scripts, including nested folders.
    """
    categories: Dict[str, List[WorkflowActionInfo]] = {}

    sm = _get_script_manager(main)
    if sm is None:
        try:
            print("[WorkflowAssistant] No ScriptManager found on main window")
        except Exception:
            pass
        return categories

    # Ensure registry exists
    try:
        if not getattr(sm, "registry", None):
            sm.load_registry()
    except Exception as e:
        try:
            print(f"[WorkflowAssistant] ScriptManager load_registry failed: {e}")
        except Exception:
            pass
        return categories

    try:
        entries = list(getattr(sm, "registry", []) or [])
    except Exception:
        entries = []

    try:
        print(f"[WorkflowAssistant] Found {len(entries)} script registry entries")
    except Exception:
        pass

    for entry in entries:
        try:
            script_id = str(getattr(entry, "script_id", "") or "").strip()
            name = str(getattr(entry, "name", "") or "").strip()
            group = str(getattr(entry, "group", "") or "").strip()
        except Exception:
            continue

        if not script_id or not name:
            continue

        command_id = f"script:{script_id}"
        category = f"Scripts/{group}" if group else "Scripts"

        act = _find_action_by_command_id(main, command_id)

        if isinstance(act, QAction):
            icon = act.icon()
            status_tip = act.statusTip() or act.toolTip() or f"Run script: {name}"
            bound_action = act
        else:
            icon = QIcon()
            status_tip = f"Run script: {name}"
            bound_action = QAction(name, main)
            bound_action.setProperty("command_id", command_id)
            bound_action.setToolTip(status_tip)

        info = WorkflowActionInfo(
            command_id=command_id,
            text=name,
            status_tip=status_tip,
            icon=icon,
            category=category,
            action=bound_action,
        )

        categories.setdefault(category, []).append(info)

    return categories

def _discover_toolbar_categories(main) -> Dict[str, List[WorkflowActionInfo]]:
    categories: Dict[str, List[WorkflowActionInfo]] = {}

    try:
        toolbars = main.findChildren(QWidget)
    except Exception:
        toolbars = []

    for tb in toolbars:
        if not hasattr(tb, "actions"):
            continue

        try:
            tb_name = tb.windowTitle() or tb.objectName() or "Other"
        except Exception:
            tb_name = "Other"

        if tb_name == "Hidden":
            continue

        items: List[WorkflowActionInfo] = []

        try:
            actions = tb.actions()
        except Exception:
            actions = []

        for act in actions:
            if not isinstance(act, QAction):
                continue
            if act.isSeparator():
                continue

            cid = act.property("command_id")
            if not cid:
                continue

            text = _clean_action_text(act.text())
            items.append(
                WorkflowActionInfo(
                    command_id=str(cid),
                    text=text or str(cid),
                    status_tip=act.statusTip() or act.toolTip() or "",
                    icon=act.icon(),
                    category=tb_name,
                    action=act,
                )
            )

        if items:
            categories[tb_name] = items

    return categories


def _default_canned_workflows() -> List[WorkflowDefinition]:
    return [
        WorkflowDefinition(
            name="Beginner OSC Workflow",
            description="A beginner-friendly basic deep sky OSC workflow.",
            lanes=["main"],
            steps=[
                WorkflowStep(command_id="crop", note="Crop away stacking edges first.", lane="main"),
                WorkflowStep(command_id="pedestal", note="Remove the blank pedestal space below the minimum value in your image.", lane="main"),
                WorkflowStep(command_id="abe", note="Remove gradients while the image is still linear.", lane="main"),
                WorkflowStep(command_id="background_neutral", note="Neutralize the background.", lane="main"),
                WorkflowStep(command_id="white_balance", note="Balance color before stretching.", lane="main"),
                WorkflowStep(command_id="cosmicclarity", note="Apply sharpening then noise reduction.", lane="main"),
                WorkflowStep(command_id="stat_stretch", note="Stretch to non-linear.", lane="main"),
                WorkflowStep(command_id="curves", note="Adjust the contrast.", lane="main"),
                WorkflowStep(command_id="clahe", note="Boost the image depth slightly.", lane="main"),
                WorkflowStep(command_id="save_as", note="Save your result.", lane="main"),
            ],
        ),
        WorkflowDefinition(
            name="Beginner Mono RGB Workflow",
            description="Simple starter workflow for mono RGB channel processing.",
            lanes=["main"],
            steps=[
                WorkflowStep(command_id="crop", note="Crop away stacking edges first.", lane="main"),
                WorkflowStep(command_id="pedestal", note="Remove the blank pedestal space below the minimum value in your image.", lane="main"),
                WorkflowStep(command_id="linear_fit", note="Match the RGB channels before combining.", lane="main"),
                WorkflowStep(command_id="rgb_combine", note="Combine the mono R, G, and B channels into a color image.", lane="main"),
                WorkflowStep(command_id="background_neutral", note="Neutralize the background before color balancing.", lane="main"),
                WorkflowStep(command_id="white_balance", note="Apply white balance to set the color balance.", lane="main"),
                WorkflowStep(command_id="cosmicclarity", note="Apply denoise and/or sharpening with Cosmic Clarity.", lane="main"),
                WorkflowStep(command_id="stat_stretch", note="Stretch the image to non-linear.", lane="main"),
                WorkflowStep(command_id="curves", note="Refine contrast and color.", lane="main"),
                WorkflowStep(command_id="clahe", note="Enhance local contrast if needed.", lane="main"),
                WorkflowStep(command_id="save_as", note="Save your result.", lane="main"),
            ],
        ),
        WorkflowDefinition(
            name="Beginner Mono Narrowband Workflow",
            description="Simple starter workflow for mono narrowband processing with separate starless / stars lanes.",
            lanes=["main", "starless", "stars", "merge"],
            steps=[
                WorkflowStep(command_id="crop", note="Crop away stacking edges first.", lane="main", outputs=["cropped"]),
                WorkflowStep(command_id="pedestal", note="Remove the blank pedestal space below the minimum value in your image.", lane="main", inputs=["cropped"], outputs=["pedestal_fixed"]),
                WorkflowStep(command_id="linear_fit", note="Match channels before combining if needed.", lane="main", inputs=["pedestal_fixed"], outputs=["aligned_channels"]),
                WorkflowStep(command_id="cosmicclarity", note="Optional: light denoise and/or sharpening before star removal. Less is more.", lane="main", inputs=["aligned_channels"], outputs=["prepped_channels"]),
                WorkflowStep(kind="split", note="Split processing into starless and stars-only paths after Remove Stars.", lane="main", outputs=["starless", "stars_only"]),
                WorkflowStep(command_id="remove_stars", note="Remove stars for each master and keep both outputs.", lane="main", inputs=["prepped_channels"], outputs=["starless", "stars_only"]),
                WorkflowStep(kind="note", note="Use your starless masters in this lane.", lane="starless"),
                WorkflowStep(command_id="ppp", note="Build the palette and color image with Perfect Palette Picker.", lane="starless", inputs=["starless"], outputs=["palette_starless"]),
                WorkflowStep(command_id="cosmicclarity", note="Optional cleanup on the starless color image.", lane="starless", inputs=["palette_starless"], outputs=["starless_cc"]),
                WorkflowStep(command_id="curves", note="Refine contrast and color.", lane="starless", inputs=["starless_cc"], outputs=["starless_curved"]),
                WorkflowStep(command_id="clahe", note="Enhance local contrast if needed.", lane="starless", inputs=["starless_curved"], outputs=["starless_final"]),
                WorkflowStep(kind="note", note="Use your stars-only masters in this lane.", lane="stars"),
                WorkflowStep(command_id="nbtorgb", note="Combine your stars-only images into an RGB stars master.", lane="stars", inputs=["stars_only"], outputs=["rgb_stars"]),
                WorkflowStep(command_id="cosmicclarity", note="Optional cleanup on the stars image.", lane="stars", inputs=["rgb_stars"], outputs=["stars_final"]),
                WorkflowStep(kind="merge", note="Merge the starless and stars-only results back together.", lane="merge", inputs=["starless_final", "stars_final"], outputs=["merged"]),
                WorkflowStep(command_id="add_stars", note="Combine your stars and your starless image.", lane="merge", inputs=["starless_final", "stars_final"], outputs=["final"]),
                WorkflowStep(command_id="save_as", note="Save your result.", lane="merge", inputs=["final"]),
            ],
        ),
        WorkflowDefinition(
            name="Cosmic Clarity Cleanup",
            description="Basic sharpening / denoise workflow.",
            lanes=["main"],
            steps=[
                WorkflowStep(command_id="crop", note="Optional crop first.", lane="main"),
                WorkflowStep(command_id="abe", note="Optional gradient cleanup.", lane="main"),
                WorkflowStep(command_id="cosmicclarity", note="Run Cosmic Clarity.", lane="main"),
                WorkflowStep(command_id="curves", note="Refine the result.", lane="main"),
                WorkflowStep(command_id="save_as", note="Save your result.", lane="main"),
            ],
        ),
    ]

def _get_script_manager(main):
    """
    Best-effort lookup for the ScriptManager instance.
    The Scripts menu may work even if the attribute name is not exactly
    'script_manager', so check a few likely locations.
    """
    candidates = [
        getattr(main, "script_manager", None),
        getattr(main, "scripts", None),
        getattr(main, "_script_manager", None),
    ]

    for obj in candidates:
        if obj is not None and hasattr(obj, "registry") and hasattr(obj, "load_registry"):
            return obj

    # Last resort: scan attributes for something that looks like ScriptManager
    try:
        for name in dir(main):
            if name.startswith("_"):
                continue
            obj = getattr(main, name, None)
            if obj is not None and hasattr(obj, "registry") and hasattr(obj, "load_registry"):
                return obj
    except Exception:
        pass

    return None

# -----------------------------------------------------------------------------
# Drag source tree
# -----------------------------------------------------------------------------

class WorkflowToolTree(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setUniformRowHeights(True)

    def mimeData(self, items: List[QTreeWidgetItem]) -> QMimeData:
        md = QMimeData()
        if not items:
            return md
        item = items[0]
        cid = item.data(0, Qt.ItemDataRole.UserRole)
        if cid:
            md.setData(WORKFLOW_MIME, str(cid).encode("utf-8"))
        return md

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is None:
            return
        cid = item.data(0, Qt.ItemDataRole.UserRole)
        if not cid:
            return

        drag = QDrag(self)
        md = QMimeData()
        md.setData(WORKFLOW_MIME, str(cid).encode("utf-8"))
        drag.setMimeData(md)

        icon = item.icon(0)
        if not icon.isNull():
            try:
                drag.setPixmap(icon.pixmap(QSize(24, 24)))
            except Exception:
                pass

        dialog = self.window()
        begin = getattr(dialog, "_begin_drag", None)
        end = getattr(dialog, "_end_drag", None)
        if callable(begin):
            begin()
        try:
            drag.exec(Qt.DropAction.CopyAction)
        finally:
            if callable(end):
                end()


# -----------------------------------------------------------------------------
# Lane list
# -----------------------------------------------------------------------------

class WorkflowLaneList(QListWidget):
    def __init__(self, dialog: "WorkflowDialog", lane_name: str, parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self.lane_name = lane_name

        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setItemDelegate(WorkflowStepDelegate(self))

        self.currentRowChanged.connect(self._notify_selection)
        self.itemSelectionChanged.connect(self._notify_selection_changed)
        self.itemDoubleClicked.connect(self._dialog._run_selected_step)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item is not None:
            self._dialog._dragging_step_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
            self._dialog._dragging_parent_id = str(item.data(Qt.ItemDataRole.UserRole + 11) or "")
        else:
            self._dialog._dragging_step_id = ""
            self._dialog._dragging_parent_id = ""
        self._dialog._begin_drag()
        try:
            super().startDrag(supportedActions)
        finally:
            self._dialog._end_drag()

    def _choice_insert_for(self, target):
        """If the drop lands on a choice or one of its children, attach just after that row."""
        if target is None:
            return "", None
        step = unpack_workflow_step_item(target, self.lane_name)
        row = self.row(target)
        if step.kind == "choice":
            return step.id, row + 1
        if step.kind == "action" and step.parent_id:
            return step.parent_id, row + 1
        return "", None

    def _notify_selection_changed(self):
        self._dialog._on_any_step_selected(self)

    def _notify_selection(self, _row: int):
        self._dialog._on_any_step_selected(self)

    def _shortcut_mime(self):
        try:
            from setiastro.saspro.dnd_mime import MIME_CMD
            return MIME_CMD
        except Exception:
            try:
                from setiastro.saspro.shortcuts import MIME_CMD
                return MIME_CMD
            except Exception:
                return None

    def dragEnterEvent(self, event):
        md = event.mimeData()
        mc = self._shortcut_mime()
        if md.hasFormat(WORKFLOW_MIME) or (mc and md.hasFormat(mc)):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        md = event.mimeData()
        mc = self._shortcut_mime()
        if md.hasFormat(WORKFLOW_MIME) or (mc and md.hasFormat(mc)):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()

        # external tool catalog drop (bare command_id from the tool tree)
        if md.hasFormat(WORKFLOW_MIME):
            try:
                cid = bytes(md.data(WORKFLOW_MIME)).decode("utf-8")
            except Exception:
                cid = ""
            if cid:
                target = self.itemAt(event.position().toPoint())
                parent_id, insert_at = self._choice_insert_for(target)
                self._dialog.add_step_by_command_id(
                    cid, lane_name=self.lane_name, parent_id=parent_id, insert_at=insert_at,
                )
                event.acceptProposedAction()
                return

        # preset grip / desktop-shortcut drop ({command_id, preset})
        mc = self._shortcut_mime()
        if mc and md.hasFormat(mc):
            cid, preset = "", {}
            try:
                from setiastro.saspro.shortcuts import _unpack_cmd_payload
                payload = _unpack_cmd_payload(bytes(md.data(mc)))
            except Exception:
                payload = None

            # A multi-row drag from History Explorer arrives as a bundle
            # payload {command_id:"function_bundle", steps:[...]}. Parse it
            # out into one step per row — exactly as if each row had been
            # dropped individually, mirroring how the canvas explodes it
            # (add_shortcut_from_payload / _add_shortcuts_from_steps).
            _cid0 = ""
            if isinstance(payload, dict):
                _cid0 = str(payload.get("command_id") or payload.get("cid") or "")
            if _cid0 == "function_bundle":
                steps = payload.get("steps")
                if isinstance(steps, list):
                    for st in steps:
                        if not isinstance(st, dict):
                            continue
                        scid = str(st.get("command_id") or st.get("cid") or "")
                        # skip empties and a nested bundle (shouldn't happen)
                        if not scid or scid == "function_bundle":
                            continue
                        _sp = st.get("preset")
                        self._dialog.add_step_by_command_id(
                            scid, lane_name=self.lane_name,
                            preset=(_sp if isinstance(_sp, dict) else {}),
                            name=str(st.get("name") or ""),
                        )
                event.acceptProposedAction()
                return

            drop_name = ""
            if isinstance(payload, dict) and payload.get("command_id"):
                cid = str(payload.get("command_id") or "")
                _p = payload.get("preset")
                preset = _p if isinstance(_p, dict) else {}
                drop_name = str(payload.get("name") or "")
            if cid:
                target = self.itemAt(event.position().toPoint())
                parent_id, insert_at = self._choice_insert_for(target)
                self._dialog.add_step_by_command_id(
                    str(cid), lane_name=self.lane_name,
                    preset=(preset if isinstance(preset, dict) else {}),
                    name=drop_name,
                    parent_id=parent_id,
                    insert_at=insert_at,
                )
                event.acceptProposedAction()
                return

        # internal move between lanes / within lane
        target = self.itemAt(event.position().toPoint())
        target_id = str(target.data(Qt.ItemDataRole.UserRole) or "") if target is not None else ""
        indicator = _drop_indicator_name(self)
        moved_id = getattr(self._dialog, "_dragging_step_id", "")
        old_parent = getattr(self._dialog, "_dragging_parent_id", "")
        super().dropEvent(event)
        self._dialog.apply_internal_drop(moved_id, old_parent, target_id, indicator)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        menu = QMenu(self)

        act_note = menu.addAction("Add Note Step")
        act_split = menu.addAction("Add Split Marker")
        act_merge = menu.addAction("Add Merge Marker")
        act_choice = menu.addAction("Add Choice Group")

        act_rename = None
        act_edit_preset = None
        act_add_alt = None
        menu_remove_alt = None
        menu_edit_alt = None
        if item is not None:
            self.setCurrentItem(item)
            menu.addSeparator()
            act_rename = menu.addAction("Rename…")
            _kind = str(item.data(Qt.ItemDataRole.UserRole + 4) or "action")
            _cid = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
            if _kind == "choice":
                act_add_alt = menu.addAction("Add Alternative…")
            elif _kind == "action" and _cid:
                act_edit_preset = menu.addAction("Edit Preset…")
                act_add_alt = menu.addAction("Add Alternative…")
                alts = normalize_workflow_alternatives(item.data(Qt.ItemDataRole.UserRole + 10))
                if alts:
                    menu_remove_alt = menu.addMenu("Remove Alternative")
                    menu_edit_alt = menu.addMenu("Edit Alternative Preset")
                    catalog = self._dialog.catalog
                    for idx, alt in enumerate(alts):
                        label = _choice_label(alt.command_id, alt.preset, alt.label_override, catalog)
                        act_rm = menu_remove_alt.addAction(label)
                        act_rm.setData(idx)
                        act_ed = menu_edit_alt.addAction(label)
                        act_ed.setData(idx)
            menu.addSeparator()
            act_edit_note = menu.addAction("Edit Note")
            act_edit_ios = menu.addAction("Edit Inputs / Outputs")
            act_remove = menu.addAction("Remove Step")
        else:
            act_edit_note = None
            act_edit_ios = None
            act_remove = None

        chosen = menu.exec(event.globalPos())
        if chosen is None:
            return

        if chosen == act_note:
            self._dialog.add_special_step(kind="note", lane_name=self.lane_name)
        elif chosen == act_split:
            self._dialog.add_special_step(kind="split", lane_name=self.lane_name)
        elif chosen == act_merge:
            self._dialog.add_special_step(kind="merge", lane_name=self.lane_name)
        elif chosen == act_choice:
            self._dialog._add_choice_group(self.lane_name)
        elif item is not None and chosen == act_rename:
            self._dialog._rename_selected_step()
        elif item is not None and act_edit_preset is not None and chosen == act_edit_preset:
            self._dialog._edit_selected_step_preset()
        elif item is not None and act_add_alt is not None and chosen == act_add_alt:
            self._dialog._add_alternative_to_selected()
        elif item is not None and menu_remove_alt is not None and chosen in menu_remove_alt.actions():
            self._dialog._remove_alternative_at(int(chosen.data()))
        elif item is not None and menu_edit_alt is not None and chosen in menu_edit_alt.actions():
            self._dialog._edit_alternative_preset(int(chosen.data()))
        elif item is not None and chosen == act_edit_note:
            self._dialog._edit_selected_step_note()
        elif item is not None and chosen == act_edit_ios:
            self._dialog._edit_selected_step_io()
        elif item is not None and chosen == act_remove:
            self._dialog._remove_selected_step()

def _drop_indicator_name(view) -> str:
    indicator = view.dropIndicatorPosition()
    kind = QAbstractItemView.DropIndicatorPosition
    return {
        kind.AboveItem: "above",
        kind.BelowItem: "below",
        kind.OnItem: "on",
    }.get(indicator, "")


def _derive_step_label(command_id: str, preset: dict, catalog) -> str:
    """Friendly per-step label derived from the preset, falling back to catalog text.
    Used only when a preset-bearing step has no explicit dropped-chip name."""
    base = ""
    try:
        info = catalog.get(command_id) if catalog else None
        if info is not None:
            base = info.text or ""
    except Exception:
        base = ""
    p = preset if isinstance(preset, dict) else {}

    if command_id == "rcastro":
        prod = str(p.get("product", "")).lower()
        return {"bxt": "BlurXTerminator", "sxt": "StarXTerminator",
                "nxt": "NoiseXTerminator"}.get(prod, base or "RC-Astro")
    if command_id in ("cosmic_clarity", "cosmic", "cosmicclarity"):
        mode = str(p.get("mode", "")).lower()
        m = {"sharpen": "Sharpen", "denoise": "Denoise",
             "both": "Sharpen + Denoise", "superres": "Super Res"}.get(mode)
        return f"Cosmic Clarity — {m}" if m else (base or "Cosmic Clarity")
    if command_id == "syqontools":
        fam = str(p.get("family", "")).lower()
        f = {"denoise": "Prism Denoise", "sharpening": "Parallax",
             "starless": "Starless"}.get(fam)
        return f"SyQon — {f}" if f else (base or "SyQon Tools")
    if command_id == "fx":
        eff = str(p.get("effect", "")).replace("_", " ").title()
        return f"FX — {eff}" if eff else (base or "FX")
    if command_id == "remove_stars":
        tool = str(p.get("tool", "")).lower()
        t = {"starnet": "StarNet", "syqon": "SyQon Starless",
             "darkstar": "DarkStar"}.get(tool)
        return f"Remove Stars — {t}" if t else (base or "Remove Stars")

    return base


def _choice_label(command_id: str, preset: dict, label_override: str, catalog) -> str:
    override = str(label_override or "").strip()
    if override:
        return override
    if isinstance(preset, dict) and preset:
        derived = _derive_step_label(command_id, preset, catalog)
        if derived:
            return derived
    info = None
    try:
        info = catalog.get(command_id) if catalog else None
    except Exception:
        info = None
    text = (getattr(info, "text", "") or "") if info is not None else ""
    return text or command_id or "Action"


def format_workflow_step_label(step: WorkflowStep, catalog) -> str:
    if step.kind == "note":
        return "📝 Note"
    if step.kind == "split":
        return "⇢ Split"
    if step.kind == "merge":
        return "⇠ Merge"
    if step.kind == "choice":
        name = str(step.label_override or "").strip() or "Choice"
        return f"▾ {name}"
    base = _choice_label(step.command_id, step.preset, step.label_override, catalog)
    text = f"◆ {base}" if step.preset else base
    if step.alternatives:
        text = f"{text} ▾ {len(step.alternatives) + 1}"
    return text


def format_workflow_lane_label(step: WorkflowStep, catalog) -> str:
    return format_workflow_step_label(step, catalog)


CHILD_ROW_INDENT = 28


class WorkflowStepDelegate(QStyledItemDelegate):
    """Shift a choice's actions to the right, icon included."""

    def paint(self, painter, option, index):
        if index.data(Qt.ItemDataRole.UserRole + 11):
            option = QStyleOptionViewItem(option)
            option.rect = option.rect.adjusted(CHILD_ROW_INDENT, 0, 0, 0)
        super().paint(painter, option, index)


def normalize_workflow_steps(steps: List[WorkflowStep]) -> List[WorkflowStep]:
    """Repair choice links and place each group's children directly under its title."""
    choice_lane = {s.id: s.lane for s in steps if s.kind == "choice"}
    for step in steps:
        if step.kind == "choice":
            step.parent_id = ""
            step.command_id = ""
            step.alternatives = []
        elif step.kind != "action" or step.parent_id not in choice_lane:
            step.parent_id = ""
        else:
            step.lane = choice_lane[step.parent_id]

    children: Dict[str, List[WorkflowStep]] = {}
    for step in steps:
        if step.parent_id:
            children.setdefault(step.parent_id, []).append(step)

    emitted = set()
    out: List[WorkflowStep] = []
    for step in steps:
        if step.id in emitted or step.parent_id:
            continue
        out.append(step)
        emitted.add(step.id)
        if step.kind == "choice":
            for child in children.get(step.id, []):
                if child.id not in emitted:
                    out.append(child)
                    emitted.add(child.id)
    return out


LINK_DROP_REASONS = {
    "missing": "That step no longer exists.",
    "same_lane": "Those steps are in the same lane; the lane order already connects them.",
    "loop": "That link would loop back on itself.",
    "duplicate": "Those steps are already linked.",
}


def _reaches(succ: Dict[str, set], start: str, goal: str) -> bool:
    stack, seen = [start], set()
    while stack:
        node = stack.pop()
        if node == goal:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(succ.get(node, ()))
    return False


def normalize_workflow_links(
    links: Optional[List[WorkflowLink]],
    steps: List[WorkflowStep],
    lanes: List[str],
    dropped: Optional[list] = None,
) -> List[WorkflowLink]:
    """Keep the links that still make sense for these steps and lanes.

    An end on an action inside a choice moves to the choice itself. A link is
    dropped when an end is gone, when both ends share a lane, when it repeats
    another link, or when it would close a loop. Each dropped link is appended
    to `dropped` as (link, reason), with reason a key of LINK_DROP_REASONS.
    """
    by_id = {s.id: s for s in steps}
    owner = {s.id: s.parent_id for s in steps if s.parent_id in by_id}
    lane_set = set(lanes)

    succ: Dict[str, set] = {}
    head: Dict[str, str] = {}
    last: Dict[str, str] = {}
    for s in steps:
        if s.id in owner:
            continue
        if s.lane in last:
            succ.setdefault(last[s.lane], set()).add(s.id)
        else:
            head[s.lane] = s.id
        last[s.lane] = s.id

    def drop(link, reason):
        if dropped is not None:
            dropped.append((link, reason))

    out: List[WorkflowLink] = []
    keys = set()
    for link in links or []:
        src = owner.get(link.src, link.src)
        if src not in by_id:
            drop(link, "missing")
            continue
        src_lane = by_id[src].lane
        if link.dst:
            dst = owner.get(link.dst, link.dst)
            if dst not in by_id:
                drop(link, "missing")
                continue
            if by_id[dst].lane == src_lane:
                drop(link, "same_lane")
                continue
            dst_lane, target = "", dst
        else:
            dst, dst_lane = "", link.dst_lane
            if dst_lane not in lane_set:
                drop(link, "missing")
                continue
            if dst_lane == src_lane:
                drop(link, "same_lane")
                continue
            target = head.get(dst_lane)   # None while that lane is empty
        key = (src, dst, dst_lane)
        if key in keys:
            drop(link, "duplicate")
            continue
        if target is not None and _reaches(succ, target, src):
            drop(link, "loop")
            continue
        keys.add(key)
        out.append(WorkflowLink(src=src, dst=dst, dst_lane=dst_lane))
        if target is not None:
            succ.setdefault(src, set()).add(target)
    return out


def derive_workflow_links(steps: List[WorkflowStep], lanes: List[str]) -> List[WorkflowLink]:
    """Links implied by input/output tags, for workflows saved before links.

    A step that takes a tag made in another lane gets a link from the last step
    there that outputs it. When only notes come before that step in its lane,
    the link enters the lane itself (a branch); otherwise it joins the step.
    Each source feeds a lane once, and a tag the lane already made itself does
    not reach out to other lanes.
    """
    by_lane: Dict[str, List[WorkflowStep]] = {}
    for s in steps:
        if not s.parent_id:
            by_lane.setdefault(s.lane, []).append(s)

    links: List[WorkflowLink] = []
    for lane in lanes:
        seq = by_lane.get(lane, [])
        fed = set()
        for pos, step in enumerate(seq):
            for tag in step.inputs:
                if any(tag in q.outputs for q in seq[:pos]):
                    continue
                for other in lanes:
                    if other == lane:
                        continue
                    makers = [p for p in by_lane.get(other, []) if tag in p.outputs]
                    if not makers or makers[-1].id in fed:
                        continue
                    src = makers[-1]
                    fed.add(src.id)
                    if all(q.kind == "note" for q in seq[:pos]):
                        links.append(WorkflowLink(src=src.id, dst_lane=lane))
                    else:
                        links.append(WorkflowLink(src=src.id, dst=step.id))
    return links


def workflow_links(wf: "WorkflowDefinition") -> List[WorkflowLink]:
    """The workflow's links, derived from I/O tags if it has none yet. Does not modify wf."""
    if wf.links is not None:
        return list(wf.links)
    return normalize_workflow_links(derive_workflow_links(wf.steps, wf.lanes), wf.steps, wf.lanes)


def normalize_graph_rows(rows, steps: List["WorkflowStep"]) -> Dict[str, int]:
    """Keep pins of top-level steps that still exist, as non-negative ints."""
    if not isinstance(rows, dict):
        return {}
    top = {s.id for s in steps if not s.parent_id}
    out: Dict[str, int] = {}
    for key, row in rows.items():
        if str(key) not in top or isinstance(row, bool):
            continue
        try:
            row = int(row)
        except (TypeError, ValueError):
            continue
        if row >= 0:
            out[str(key)] = row
    return out


def ensure_workflow_links(wf: "WorkflowDefinition") -> List[WorkflowLink]:
    """Give wf a links list that matches its current steps and lanes."""
    wf.links = normalize_workflow_links(workflow_links(wf), wf.steps, wf.lanes)
    return wf.links


def workflow_link_lines(wf: "WorkflowDefinition", step_id: str, catalog) -> List[str]:
    """Details-panel lines naming what feeds a step and where its result goes."""
    import html

    by_id = {s.id: s for s in wf.steps}
    step = by_id.get(step_id)
    if step is None or step.parent_id:
        return []
    head: Dict[str, str] = {}
    for s in wf.steps:
        if not s.parent_id:
            head.setdefault(s.lane, s.id)

    def name(s: WorkflowStep) -> str:
        return html.escape(f"{format_workflow_step_label(s, catalog)} ({s.lane})")

    takes, sends = [], []
    for link in wf.links or []:
        if link.src == step_id:
            if link.dst in by_id:
                sends.append(name(by_id[link.dst]))
            elif link.dst_lane:
                sends.append(html.escape(f"start of lane {link.dst_lane}"))
        target = link.dst or head.get(link.dst_lane, "")
        if target == step_id and link.src in by_id:
            takes.append(name(by_id[link.src]))
    lines = []
    if takes:
        lines.append(f"<b>Takes input from:</b> {', '.join(takes)}")
    if sends:
        lines.append(f"<b>Sends result to:</b> {', '.join(sends)}")
    return lines


def parent_id_after_drop(
    steps: List[WorkflowStep],
    moved_id: str,
    old_parent_id: str,
    target_id: str,
    indicator: str,
) -> str:
    """Parent id for an action after an internal move. Empty string detaches it.

    `indicator` is "above", "below", or "on". A missing indicator still counts
    as a drop on the row. The whole choice row accepts the action.
    """
    by_id = {s.id: s for s in steps}
    moved = by_id.get(moved_id)
    target = by_id.get(target_id)
    if moved is None or target is None or moved.kind != "action" or moved.id == target.id:
        return ""
    if indicator not in ("above", "below", "on"):
        indicator = "on"
    choice_ids = {s.id for s in steps if s.kind == "choice"}
    if target.kind == "choice":
        return target.id
    if target.kind == "action" and target.parent_id in choice_ids and target.lane == moved.lane:
        siblings = [
            s for s in steps
            if s.lane == target.lane and s.parent_id == target.parent_id and s.id != moved.id
        ]
        is_last = bool(siblings) and siblings[-1].id == target.id
        if indicator == "below" and is_last and old_parent_id != target.parent_id:
            return ""
        return target.parent_id
    return ""


def choice_group_entries(children: List[WorkflowStep], catalog) -> List[tuple]:
    """One menu entry per child: that child's own command and preset."""
    entries = []
    for child in children:
        if child.kind != "action" or not str(child.command_id or "").strip():
            continue
        entries.append((
            child.command_id,
            dict(child.preset or {}),
            format_workflow_step_label(child, catalog),
        ))
    return entries


def paint_workflow_row(item: QListWidgetItem, is_current: bool) -> None:
    """Choice stays a normal text row, tinted so it is not an action."""
    kind = str(item.data(Qt.ItemDataRole.UserRole + 4) or "")
    font = item.font()
    font.setBold(is_current or kind == "choice")
    item.setFont(font)
    if kind == "choice":
        item.setSizeHint(QSize())
        item.setBackground(QBrush(QColor(36, 58, 54)))
        item.setForeground(QBrush(QColor(215, 243, 234)))
    else:
        item.setBackground(QBrush())
        item.setForeground(QBrush())


def workflow_steps_without(steps: List[WorkflowStep], step_id: str) -> List[WorkflowStep]:
    """Drop one step. Removing a choice also drops the actions under it."""
    drop = {step_id}
    for step in steps:
        if step.id == step_id and step.kind == "choice":
            drop.update(s.id for s in steps if s.parent_id == step.id)
            break
    return [s for s in steps if s.id not in drop]


def choice_completion_target_id(step: WorkflowStep, steps: List[WorkflowStep]) -> str:
    """Completion lives on the choice title, including when a child is selected."""
    if step.parent_id:
        for other in steps:
            if other.id == step.parent_id and other.kind == "choice":
                return other.id
    return step.id


def workflow_step_choices(step: WorkflowStep, catalog) -> List[tuple]:
    """(command_id, preset, label) for the primary tool, then each alternative."""
    if step.kind != "action" or not str(step.command_id or "").strip():
        return []
    choices = [(
        step.command_id,
        dict(step.preset or {}),
        _choice_label(step.command_id, step.preset, step.label_override, catalog),
    )]
    for alt in step.alternatives:
        choices.append((
            alt.command_id,
            dict(alt.preset or {}),
            _choice_label(alt.command_id, alt.preset, alt.label_override, catalog),
        ))
    return choices


def pack_workflow_step_item(item: QListWidgetItem, step: WorkflowStep) -> None:
    item.setData(Qt.ItemDataRole.UserRole, step.id)
    item.setData(Qt.ItemDataRole.UserRole + 1, step.command_id)
    item.setData(Qt.ItemDataRole.UserRole + 2, step.note)
    item.setData(Qt.ItemDataRole.UserRole + 3, bool(step.enabled))
    item.setData(Qt.ItemDataRole.UserRole + 4, step.kind)
    item.setData(Qt.ItemDataRole.UserRole + 5, list(step.inputs))
    item.setData(Qt.ItemDataRole.UserRole + 6, list(step.outputs))
    item.setData(Qt.ItemDataRole.UserRole + 7, False)  # completed, session-only
    item.setData(Qt.ItemDataRole.UserRole + 8, dict(step.preset or {}))
    item.setData(Qt.ItemDataRole.UserRole + 9, step.label_override)
    item.setData(Qt.ItemDataRole.UserRole + 10, [a.to_dict() for a in step.alternatives])
    item.setData(Qt.ItemDataRole.UserRole + 11, step.parent_id)


def unpack_workflow_step_item(item: QListWidgetItem, lane: str) -> WorkflowStep:
    preset = item.data(Qt.ItemDataRole.UserRole + 8)
    return WorkflowStep(
        id=str(item.data(Qt.ItemDataRole.UserRole) or _new_step_id()),
        command_id=str(item.data(Qt.ItemDataRole.UserRole + 1) or ""),
        note=str(item.data(Qt.ItemDataRole.UserRole + 2) or ""),
        enabled=bool(item.data(Qt.ItemDataRole.UserRole + 3)),
        kind=str(item.data(Qt.ItemDataRole.UserRole + 4) or "action"),
        lane=lane,
        inputs=list(item.data(Qt.ItemDataRole.UserRole + 5) or []),
        outputs=list(item.data(Qt.ItemDataRole.UserRole + 6) or []),
        preset=dict(preset) if isinstance(preset, dict) else {},
        label_override=str(item.data(Qt.ItemDataRole.UserRole + 9) or ""),
        alternatives=normalize_workflow_alternatives(item.data(Qt.ItemDataRole.UserRole + 10)),
        parent_id=str(item.data(Qt.ItemDataRole.UserRole + 11) or ""),
    )


def open_choice_group(parent, main, lst, step: WorkflowStep, catalog) -> bool:
    """Menu of a choice title. Empty groups explain themselves and do not count as opened."""
    lane = getattr(lst, "lane_name", "main")
    children = []
    for i in range(lst.count()):
        child = unpack_workflow_step_item(lst.item(i), lane)
        if child.kind == "action" and child.parent_id == step.id:
            children.append(child)
    entries = choice_group_entries(children, catalog)
    if not entries:
        QMessageBox.information(parent, "Choice", "This choice has no alternatives yet.")
        return False
    picked = prompt_workflow_choice(parent, entries)
    if picked is None:
        return False
    command_id, preset, _label = picked
    return open_workflow_command(parent, main, command_id, preset)


def mark_group_complete(lst, item: QListWidgetItem) -> QListWidgetItem:
    """Check the choice title when the selection is the title or one of its children."""
    lane = getattr(lst, "lane_name", "main")
    steps = [unpack_workflow_step_item(lst.item(i), lane) for i in range(lst.count())]
    step = unpack_workflow_step_item(item, lane)
    target_id = choice_completion_target_id(step, steps)
    target_item = item
    for i in range(lst.count()):
        other = lst.item(i)
        sid = str(other.data(Qt.ItemDataRole.UserRole) or "")
        pid = str(other.data(Qt.ItemDataRole.UserRole + 11) or "")
        if sid == target_id:
            other.setData(Qt.ItemDataRole.UserRole + 7, True)
            target_item = other
        elif pid == target_id:
            other.setData(Qt.ItemDataRole.UserRole + 7, False)
    return target_item


def open_workflow_command(parent, main, command_id: str, preset: Optional[dict]) -> bool:
    """Open one tool, seeded with preset when an opener exists."""
    if isinstance(preset, dict) and preset:
        try:
            from setiastro.saspro.shortcuts import _preset_opener_for_command
            opener = _preset_opener_for_command(command_id)
        except Exception:
            opener = None
        if opener is not None:
            try:
                opener(main, dict(preset))
                return True
            except Exception as e:
                QMessageBox.critical(
                    parent, "Step Failed",
                    f"Error opening seeded step:\n\n{command_id}\n\n{e}",
                )
                return False

    action = _find_action_by_command_id(main, command_id)
    if action is None:
        QMessageBox.warning(
            parent, "Action Not Found",
            f"Could not find QAction for command_id:\n{command_id}",
        )
        return False
    try:
        action.trigger()
        return True
    except Exception as e:
        QMessageBox.critical(
            parent, "Step Failed",
            f"Error while opening workflow step:\n\n{command_id}\n\n{e}",
        )
        return False


def prompt_workflow_choice(parent, choices: List[tuple]):
    """Ask which tool to open. A single choice is returned without a menu."""
    if not choices:
        return None
    if len(choices) == 1:
        return choices[0]
    menu = QMenu(parent)
    header = menu.addAction("Choose a tool")
    header.setEnabled(False)
    for command_id, preset, label in choices:
        act = menu.addAction(label or command_id)
        act.setData({"command_id": command_id, "preset": dict(preset or {})})
    chosen = menu.exec(QCursor.pos())
    if chosen is None or chosen is header:
        return None
    payload = chosen.data()
    if not isinstance(payload, dict) or not payload.get("command_id"):
        return None
    preset = payload.get("preset")
    return (str(payload["command_id"]), dict(preset) if isinstance(preset, dict) else {}, chosen.text())


def pick_catalog_command(parent, catalog: Dict[str, "WorkflowActionInfo"]) -> Optional[str]:
    """Searchable catalog picker. Returns a command id, or None if cancelled."""
    if not catalog:
        QMessageBox.information(parent, "Add Alternative", "No tools are available in the catalog.")
        return None

    dlg = QDialog(parent)
    dlg.setWindowTitle("Add Alternative")
    dlg.resize(440, 520)
    lay = QVBoxLayout(dlg)
    search = QLineEdit(dlg)
    search.setPlaceholderText("Search tools...")
    lay.addWidget(search)
    lst = QListWidget(dlg)
    lay.addWidget(lst, 1)
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        parent=dlg,
    )
    ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
    ok_btn.setEnabled(False)
    lay.addWidget(buttons)

    entries = sorted(
        catalog.values(),
        key=lambda info: ((info.category or "").lower(), (info.text or "").lower(), info.command_id),
    )

    def refill():
        needle = search.text().strip().lower()
        lst.clear()
        for info in entries:
            blob = f"{info.category} {info.text} {info.status_tip} {info.command_id}".lower()
            if needle and needle not in blob:
                continue
            caption = f"{info.category} — {info.text}" if info.category else (info.text or info.command_id)
            row = QListWidgetItem(caption)
            row.setData(Qt.ItemDataRole.UserRole, info.command_id)
            if info.status_tip:
                row.setToolTip(info.status_tip)
            if info.icon and not info.icon.isNull():
                row.setIcon(info.icon)
            lst.addItem(row)
        ok_btn.setEnabled(False)

    def on_sel():
        ok_btn.setEnabled(lst.currentItem() is not None)

    search.textChanged.connect(refill)
    lst.itemSelectionChanged.connect(on_sel)
    lst.itemDoubleClicked.connect(dlg.accept)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    refill()
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    row = lst.currentItem()
    if row is None:
        return None
    cid = str(row.data(Qt.ItemDataRole.UserRole) or "").strip()
    return cid or None


# -----------------------------------------------------------------------------
# Graph window plumbing (shared by the editor and the mini runner)
# -----------------------------------------------------------------------------

def _settings_flag(value) -> bool:
    """QSettings hands bools back as bools or as 'true'/'false', depending on the backend."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


class _WorkflowGraphHost:
    """What workflow_graph.WorkflowGraphWindow needs from a workflow dialog.

    The lane lists stay the live state: selection and completion live on their
    items, so the graph reads them and selects through them. That way every
    existing step command (rename, presets, alternatives, remove, open, ...)
    works unchanged when it is started from the graph.

    The graph lives in its own window, made on first use. It is rebuilt only
    while it is showing, and it reopens with the dialog if it was open when
    the dialog last closed.
    """

    graph_editable = False
    _GRAPH_SETTINGS_PREFIX = ""

    def _graph_wf(self) -> WorkflowDefinition:
        raise NotImplementedError

    def _graph_init(self):
        self._graph = None                 # workflow_graph.WorkflowGraphWindow
        self._graph_dirty = True
        self._graph_flush_queued = False
        self._graph_details_queued = False

    def _make_graph_button(self) -> QPushButton:
        btn = QPushButton("Graph View", self)
        btn.setToolTip("Show this workflow as a graph in its own window: lanes, branches, "
                       "joins and progress at a glance (Ctrl+G)")
        btn.setShortcut("Ctrl+G")
        btn.clicked.connect(lambda _c=False: self.open_graph_window())
        return btn

    def open_graph_window(self) -> bool:
        if self._graph is None:
            try:
                from setiastro.saspro.workflow_graph import WorkflowGraphWindow
                self._graph = WorkflowGraphWindow(self, editable=self.graph_editable, parent=self)
            except Exception as e:
                QMessageBox.warning(self, "Graph View", f"The graph view could not be opened:\n{e}")
                return False
            self._graph.sync_palette(getattr(self, "tool_tree", None))
        win = self._graph
        if win.isMinimized():
            win.showNormal()
        else:
            win.show()
        win.raise_()
        win.activateWindow()
        return True

    def _graph_visible(self) -> bool:
        return self._graph is not None and self._graph.isVisible() and not self._graph.isMinimized()

    def showEvent(self, ev):
        super().showEvent(ev)
        if ev.spontaneous() or self.settings is None:
            return
        try:
            reopen = _settings_flag(self.settings.value(f"{self._GRAPH_SETTINGS_PREFIX}/graph_open", False))
        except Exception:
            reopen = False
        if reopen and not self._graph_visible():
            QTimer.singleShot(0, self.open_graph_window)

    def hideEvent(self, ev):
        # Close and Esc both land here. A child window does not hide with its
        # parent, so take the graph down too and remember whether it was up.
        if not ev.spontaneous():
            was_open = self._graph_visible()
            if self.settings is not None:
                try:
                    self.settings.setValue(f"{self._GRAPH_SETTINGS_PREFIX}/graph_open", was_open)
                except Exception:
                    pass
            if self._graph is not None:
                self._graph.hide()
        super().hideEvent(ev)

    def graph_refresh(self):
        """Rebuild the graph soon. Several calls in a row rebuild once."""
        self._graph_dirty = True
        if self._graph_visible() and not self._graph_flush_queued:
            self._graph_flush_queued = True
            QTimer.singleShot(0, self._graph_flush)

    def _graph_flush(self):
        self._graph_flush_queued = False
        if not self._graph_visible():
            return
        if self._graph_dirty:
            self._graph_dirty = False
            self._graph.rebuild()
        else:
            self._graph.restyle()
        self._graph_sync_details()

    def _graph_restyle(self):
        if self._graph_visible() and not self._graph_dirty:
            self._graph.restyle()
        # Some callers restyle first and fill in the details after, so copy
        # the details once the current event is done.
        if self._graph_visible() and not self._graph_details_queued:
            self._graph_details_queued = True
            QTimer.singleShot(0, self._graph_sync_details)

    def _graph_sync_details(self):
        self._graph_details_queued = False
        lbl = getattr(self, "lbl_step_info", None)
        if self._graph_visible() and lbl is not None:
            self._graph.set_details(lbl.text())

    def _graph_request_fit(self):
        if self._graph is not None:
            self._graph.request_fit()

    def _graph_hint(self, text: str):
        if self._graph is not None and text:
            self._graph.show_hint(text)

    # ---- read side ----------------------------------------------------

    def graph_workflow(self) -> WorkflowDefinition:
        wf = self._graph_wf()
        ensure_workflow_links(wf)
        return wf

    def graph_catalog(self) -> Dict[str, WorkflowActionInfo]:
        return self.catalog

    def _graph_items(self):
        for lane, lst in self.lane_lists.items():
            for row in range(lst.count()):
                yield lane, lst, row, lst.item(row)

    def graph_completed_ids(self) -> set:
        return {
            str(item.data(Qt.ItemDataRole.UserRole) or "")
            for _lane, _lst, _row, item in self._graph_items()
            if item.data(Qt.ItemDataRole.UserRole + 7)
        }

    def graph_current_id(self) -> str:
        item = self._current_item()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""

    # ---- commands every host supports ---------------------------------

    def graph_select(self, step_id: str) -> bool:
        for lane, lst, row, item in self._graph_items():
            if str(item.data(Qt.ItemDataRole.UserRole) or "") == step_id:
                self.current_lane_name = lane
                lst.setCurrentRow(row)
                # setCurrentRow is silent when the row was already current.
                self._on_any_step_selected(lst)
                return True
        return False

    def graph_open(self, step_id: str):
        if self.graph_select(step_id):
            self._run_selected_step()

    def graph_open_current(self):
        self._run_selected_step()

    def graph_complete_current(self):
        self._graph_complete_selected()

    def graph_prev(self):
        self._prev_step()

    def graph_next(self):
        self._next_step()

    def graph_add_tool(self, command_id: str):
        pass

    def _graph_complete_selected(self):
        raise NotImplementedError

    def graph_node_menu(self, menu: QMenu, step_id: str):
        if not self.graph_select(step_id):
            return
        menu.addAction("Open Step", self._run_selected_step)
        menu.addAction("Mark Complete", self._graph_complete_selected)

    def graph_lane_menu(self, menu: QMenu, lane: str):
        pass

    def graph_canvas_menu(self, menu: QMenu):
        pass

    def graph_link_menu(self, menu: QMenu, link: WorkflowLink):
        pass

    def graph_set_row(self, step_id: str, row: Optional[int]):
        pass


# -----------------------------------------------------------------------------
# Main dialog
# -----------------------------------------------------------------------------

class WorkflowDialog(_WorkflowGraphHost, QDialog):
    def __init__(self, main, parent=None):
        super().__init__(parent or main)
        self.main = main
        self.settings = _main_settings(main)
        self.workflows_dir = _workflows_dir(main)

        self.setWindowTitle("Workflow Assistant")
        self.resize(1450, 820)

        self.current_workflow = WorkflowDefinition(name="Untitled Workflow")
        self.catalog: Dict[str, WorkflowActionInfo] = {}
        self.canned_workflows = _default_canned_workflows()

        self.lane_lists: Dict[str, WorkflowLaneList] = {}
        self.current_lane_name: Optional[str] = None
        self._dragging_step_id = ""
        self._dragging_parent_id = ""
        self._drag_depth = 0
        self._layout_after_drag = False
        self._layout_select_id: Optional[str] = None
        self._graph_init()

        self._build_ui()
        self._populate_action_catalog()

        self._restore_geometry()
        self.refresh_workflow_view()
        self._restore_autosaved_session()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(splitter, 1)

        # ---------------- Left: tool catalog ----------------
        left = QWidget(self)
        left_lay = QVBoxLayout(left)

        self.edt_search = QLineEdit(self)
        self.edt_search.setPlaceholderText("Search tools...")
        self.edt_search.textChanged.connect(self._refilter_catalog)
        left_lay.addWidget(self.edt_search)

        self.tool_tree = WorkflowToolTree(self)
        self.tool_tree.itemDoubleClicked.connect(self._on_tool_double_clicked)
        left_lay.addWidget(self.tool_tree, 1)

        splitter.addWidget(left)

        # ---------------- Right: lane editor ----------------
        right = QWidget(self)
        right_lay = QVBoxLayout(right)

        form = QFormLayout()
        self.edt_name = QLineEdit(self)
        self.edt_name.setPlaceholderText("Workflow name")
        self.edt_name.textChanged.connect(self._on_name_changed)

        self.edt_description = QTextEdit(self)
        self.edt_description.setPlaceholderText("Workflow description")
        self.edt_description.setFixedHeight(72)
        self.edt_description.textChanged.connect(self._on_description_changed)

        form.addRow("Name:", self.edt_name)
        form.addRow("Description:", self.edt_description)
        right_lay.addLayout(form)

        # lane controls
        lane_row = QHBoxLayout()
        self.btn_add_lane = QPushButton("Add Lane", self)
        self.btn_rename_lane = QPushButton("Rename Lane", self)
        self.btn_remove_lane = QPushButton("Remove Lane", self)

        self.btn_add_lane.clicked.connect(self._add_lane)
        self.btn_rename_lane.clicked.connect(self._rename_current_lane)
        self.btn_remove_lane.clicked.connect(self._remove_current_lane)

        lane_row.addWidget(self.btn_add_lane)
        lane_row.addWidget(self.btn_rename_lane)
        lane_row.addWidget(self.btn_remove_lane)
        lane_row.addStretch(1)
        self.btn_graph = self._make_graph_button()
        lane_row.addWidget(self.btn_graph)
        right_lay.addLayout(lane_row)

        # scrollable lanes
        self.lanes_scroll = QScrollArea(self)
        self.lanes_scroll.setWidgetResizable(True)
        self.lanes_host = QWidget(self)
        self.lanes_layout = QHBoxLayout(self.lanes_host)
        self.lanes_layout.setContentsMargins(4, 4, 4, 4)
        self.lanes_layout.setSpacing(10)
        self.lanes_scroll.setWidget(self.lanes_host)

        self.lbl_step_info = QLabel("Select a step to view details.", self)
        self.lbl_step_info.setWordWrap(True)
        self.lbl_step_info.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_step_info.setMinimumHeight(96)
        self.lbl_step_info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # The details panel scrolls instead of growing, and the user decides
        # how much height it takes from the lanes above it.
        info_scroll = QScrollArea(self)
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        info_scroll.setWidget(self.lbl_step_info)
        self.view_split = QSplitter(Qt.Orientation.Vertical, self)
        self.view_split.addWidget(self.lanes_scroll)
        self.view_split.addWidget(info_scroll)
        self.view_split.setChildrenCollapsible(False)
        self.view_split.setStretchFactor(0, 1)
        self.view_split.setStretchFactor(1, 0)
        self.view_split.setSizes([620, 150])
        right_lay.addWidget(self.view_split, 1)

        # edit buttons
        row1 = QHBoxLayout()
        self.btn_add_selected = QPushButton("Add Selected Tool", self)
        self.btn_add_note = QPushButton("Add Note", self)
        self.btn_add_choice = QPushButton("Add Choice", self)
        self.btn_add_split = QPushButton("Add Split", self)
        self.btn_add_merge = QPushButton("Add Merge", self)
        self.btn_remove = QPushButton("Remove Step", self)
        self.btn_note = QPushButton("Edit Note", self)
        self.btn_io = QPushButton("Edit Inputs / Outputs", self)
        self.btn_clear = QPushButton("Clear Workflow", self)

        self.btn_add_selected.clicked.connect(self._add_selected_tool)
        self.btn_add_note.clicked.connect(lambda: self.add_special_step("note"))
        self.btn_add_choice.clicked.connect(lambda: self._add_choice_group())
        self.btn_add_split.clicked.connect(lambda: self.add_special_step("split"))
        self.btn_add_merge.clicked.connect(lambda: self.add_special_step("merge"))
        self.btn_remove.clicked.connect(self._remove_selected_step)
        self.btn_note.clicked.connect(self._edit_selected_step_note)
        self.btn_io.clicked.connect(self._edit_selected_step_io)
        self.btn_clear.clicked.connect(self._clear_workflow)

        for w in (
            self.btn_add_selected,
            self.btn_add_note,
            self.btn_add_choice,
            self.btn_add_split,
            self.btn_add_merge,
            self.btn_remove,
            self.btn_note,
            self.btn_io,
            self.btn_clear,
        ):
            row1.addWidget(w)
        row1.addStretch(1)
        right_lay.addLayout(row1)

        # run buttons
        row2 = QHBoxLayout()
        self.btn_prev = QPushButton("Previous Step", self)
        self.btn_run = QPushButton("Open Step", self)
        self.btn_done = QPushButton("Mark Complete", self)
        self.btn_next = QPushButton("Next Step", self)
        self.btn_reset_progress = QPushButton("Reset Progress", self)

        self.btn_prev.clicked.connect(self._prev_step)
        self.btn_run.clicked.connect(self._run_selected_step)
        self.btn_done.clicked.connect(self._mark_selected_complete)
        self.btn_next.clicked.connect(self._next_step)
        self.btn_reset_progress.clicked.connect(self._reset_progress)

        row2.addWidget(self.btn_prev)
        row2.addWidget(self.btn_run)
        row2.addWidget(self.btn_done)
        row2.addWidget(self.btn_next)
        row2.addWidget(self.btn_reset_progress)
        row2.addStretch(1)
        right_lay.addLayout(row2)

        splitter.addWidget(right)
        splitter.setSizes([370, 1080])

        # bottom file/canned actions
        bottom = QHBoxLayout()

        self.btn_new = QPushButton("New", self)
        self.btn_load = QPushButton("Load...", self)
        self.btn_save = QPushButton("Save", self)
        self.btn_save_as = QPushButton("Save As...", self)

        self.btn_canned = QToolButton(self)
        self.btn_canned.setText("Add Canned Workflow")
        self.btn_canned.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        canned_menu = QMenu(self.btn_canned)
        for wf in self.canned_workflows:
            act = canned_menu.addAction(wf.name)
            act.triggered.connect(lambda checked=False, wf=wf: self.load_workflow_definition(wf))
        self.btn_canned.setMenu(canned_menu)

        self.btn_new.clicked.connect(self._new_workflow)
        self.btn_load.clicked.connect(self._load_workflow)
        self.btn_save.clicked.connect(self._save_workflow)
        self.btn_save_as.clicked.connect(self._save_workflow_as)

        bottom.addWidget(self.btn_new)
        bottom.addWidget(self.btn_load)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_save_as)
        bottom.addSpacing(16)
        self.btn_run_mini = QPushButton("Run Workflow", self)
        self.btn_run_mini.setStatusTip("Open the compact runner for this workflow")
        self.btn_run_mini.clicked.connect(self._open_mini_runner)

        bottom.addWidget(self.btn_canned)
        bottom.addSpacing(16)
        bottom.addWidget(self.btn_run_mini)
        bottom.addStretch(1)
        bottom.addWidget(make_help_button("workflow_assistant", self))

        root.addLayout(bottom)
    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------
    def _open_mini_runner(self):
        self.sync_steps_from_ui()
        from setiastro.saspro.workflows import show_mini_workflow_dialog
        show_mini_workflow_dialog(self.main)

    def _populate_action_catalog(self):
        self.tool_tree.clear()
        self.catalog.clear()

        categories = _discover_toolbar_categories(self.main)

        # Merge in user scripts discovered recursively from SASpro/scripts
        script_categories = _discover_script_categories(self.main)
        for cat_name, items in script_categories.items():
            categories.setdefault(cat_name, []).extend(items)

        preferred_order = [
            "Functions",
            "Cosmic Clarity",
            "Tools",
            "Geometry",
            "Star Stuff",
            "Masks",
            "What's In My...",
            "Bundles",
            "View",
            "Scripts",
        ]

        category_names = list(categories.keys())

        def _sort_key(cat: str):
            if cat in preferred_order:
                return (0, preferred_order.index(cat), cat.lower())
            if cat.startswith("Scripts/"):
                return (1, 999, cat.lower())
            return (2, 999, cat.lower())

        ordered = sorted(category_names, key=_sort_key)

        for category in ordered:
            parent = QTreeWidgetItem([category])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            parent.setFirstColumnSpanned(True)
            self.tool_tree.addTopLevelItem(parent)

            # sort items nicely inside each category
            items = sorted(
                categories[category],
                key=lambda info: (info.text or "").lower()
            )

            for info in items:
                self.catalog[info.command_id] = info
                item = QTreeWidgetItem(parent, [info.text])
                item.setData(0, Qt.ItemDataRole.UserRole, info.command_id)
                item.setToolTip(0, info.status_tip or info.text)
                if not info.icon.isNull():
                    item.setIcon(0, info.icon)

            parent.setExpanded(True)

    def _refilter_catalog(self):
        needle = self.edt_search.text().strip().lower()

        for i in range(self.tool_tree.topLevelItemCount()):
            parent = self.tool_tree.topLevelItem(i)
            visible_children = 0

            for j in range(parent.childCount()):
                child = parent.child(j)
                text = child.text(0).lower()
                tip = child.toolTip(0).lower()
                show = (not needle) or (needle in text) or (needle in tip)
                child.setHidden(not show)
                if show:
                    visible_children += 1

            parent.setHidden(visible_children == 0)

    # ------------------------------------------------------------------
    # Lane UI
    # ------------------------------------------------------------------

    def _clear_lane_widgets(self):
        while self.lanes_layout.count():
            item = self.lanes_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.lane_lists.clear()

    def _build_lane_widgets(self):
        self._clear_lane_widgets()

        for lane_name in self.current_workflow.lanes:
            card = QWidget(self.lanes_host)
            card.setMinimumWidth(260)
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(4, 4, 4, 4)

            header = QLabel(lane_name, card)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setFrameShape(QFrame.Shape.StyledPanel)
            header.setStyleSheet("font-weight: 600; padding: 6px;")
            card_lay.addWidget(header)

            lst = WorkflowLaneList(self, lane_name=lane_name, parent=card)
            card_lay.addWidget(lst, 1)

            self.lane_lists[lane_name] = lst
            self.lanes_layout.addWidget(card)

        self.lanes_layout.addStretch(1)

    # ------------------------------------------------------------------
    # Workflow model/view sync
    # ------------------------------------------------------------------

    def refresh_workflow_view(self, select_id: Optional[str] = None):
        ensure_workflow_links(self.current_workflow)
        self.graph_refresh()
        done: Dict[str, bool] = {}
        for lst in self.lane_lists.values():
            for i in range(lst.count()):
                item = lst.item(i)
                sid = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if sid:
                    done[sid] = bool(item.data(Qt.ItemDataRole.UserRole + 7))

        self.edt_name.blockSignals(True)
        self.edt_description.blockSignals(True)
        try:
            self.edt_name.setText(self.current_workflow.name)
            self.edt_description.setPlainText(self.current_workflow.description)
        finally:
            self.edt_name.blockSignals(False)
            self.edt_description.blockSignals(False)

        if not self.current_workflow.lanes:
            self.current_workflow.lanes = ["main"]

        self._build_lane_widgets()

        # populate steps into the correct lane list
        by_lane: Dict[str, List[WorkflowStep]] = {lane: [] for lane in self.current_workflow.lanes}
        for step in self.current_workflow.steps:
            lane = step.lane if step.lane in by_lane else "main"
            by_lane.setdefault(lane, []).append(step)

        for lane_name, steps in by_lane.items():
            lst = self.lane_lists.get(lane_name)
            if lst is None:
                continue

            lst.clear()
            for step in steps:
                item = self._make_step_item(step)
                if done.get(step.id):
                    item.setData(Qt.ItemDataRole.UserRole + 7, True)
                lst.addItem(item)

        self.current_lane_name = None
        if select_id:
            for lane_name in self.current_workflow.lanes:
                lst = self.lane_lists.get(lane_name)
                if lst is None:
                    continue
                for i in range(lst.count()):
                    if str(lst.item(i).data(Qt.ItemDataRole.UserRole) or "") == select_id:
                        self.current_lane_name = lane_name
                        lst.setCurrentRow(i)
                        self._refresh_step_visuals()
                        self._update_selected_step_info()
                        return

        for lane_name in self.current_workflow.lanes:
            lst = self.lane_lists.get(lane_name)
            if lst and lst.count() > 0:
                self.current_lane_name = lane_name
                lst.setCurrentRow(0)
                break

        if self.current_lane_name is None:
            self.lbl_step_info.setText("Select a step to view details.")

        self._refresh_step_visuals()

    def _raw_steps_from_ui(self) -> List[WorkflowStep]:
        steps: List[WorkflowStep] = []
        for lane_name in self.current_workflow.lanes:
            lst = self.lane_lists.get(lane_name)
            if lst is None:
                continue
            for i in range(lst.count()):
                steps.append(unpack_workflow_step_item(lst.item(i), lane=lane_name))
        return steps

    def sync_steps_from_ui(self):
        self.current_workflow.steps = normalize_workflow_steps(self._raw_steps_from_ui())
        ensure_workflow_links(self.current_workflow)
        self.graph_refresh()

    def _begin_drag(self):
        self._drag_depth += 1

    def _end_drag(self):
        self._drag_depth = max(0, self._drag_depth - 1)
        if self._drag_depth == 0 and self._layout_after_drag:
            QTimer.singleShot(0, self._flush_layout_after_drag)

    def _flush_layout_after_drag(self):
        if self._drag_depth > 0:
            QTimer.singleShot(0, self._flush_layout_after_drag)
            return
        if not self._layout_after_drag:
            return
        self._layout_after_drag = False
        select_id = self._layout_select_id
        self._layout_select_id = None
        self.refresh_workflow_view(select_id=select_id)

    def apply_internal_drop(self, moved_id: str, old_parent_id: str, target_id: str, indicator: str):
        raw = self._raw_steps_from_ui()
        for step in raw:
            if step.id == moved_id and step.kind == "action":
                step.parent_id = parent_id_after_drop(raw, moved_id, old_parent_id, target_id, indicator)
                break
        self.current_workflow.steps = normalize_workflow_steps(raw)
        # Rebuilding the lane widgets inside dropEvent deletes the list Qt is
        # still dragging, which segfaults. Refresh once the drag has returned.
        self._layout_select_id = moved_id or None
        self._layout_after_drag = True

    def _make_step_item(self, step: WorkflowStep) -> QListWidgetItem:
        label = self._step_display_text(step)
        item = QListWidgetItem(label)

        if step.kind == "action":
            info = self.catalog.get(step.command_id)
            if info and not info.icon.isNull():
                item.setIcon(info.icon)

        pack_workflow_step_item(item, step)
        return item

    def _step_display_text(self, step: WorkflowStep) -> str:
        return format_workflow_lane_label(step, self.catalog)

    def load_workflow_definition(self, wf: WorkflowDefinition):
        self.current_workflow = WorkflowDefinition(
            name=wf.name,
            description=wf.description,
            lanes=list(wf.lanes) if wf.lanes else ["main"],
            steps=[
                WorkflowStep(
                    id=s.id,
                    command_id=s.command_id,
                    note=s.note,
                    enabled=s.enabled,
                    kind=s.kind,
                    lane=s.lane,
                    inputs=list(s.inputs),
                    outputs=list(s.outputs),
                    preset=dict(s.preset or {}),
                    label_override=s.label_override,
                    alternatives=[
                        WorkflowAlternative(
                            command_id=a.command_id,
                            preset=dict(a.preset or {}),
                            label_override=a.label_override,
                        )
                        for a in s.alternatives
                    ],
                    parent_id=s.parent_id,
                )
                for s in wf.steps
            ],
            schema_version=wf.schema_version,
            links=[
                WorkflowLink(src=l.src, dst=l.dst, dst_lane=l.dst_lane)
                for l in workflow_links(wf)
            ],
            graph_rows=dict(getattr(wf, "graph_rows", None) or {}),
        )
        self._graph_request_fit()
        self.refresh_workflow_view()

    # ------------------------------------------------------------------
    # Selection / visuals
    # ------------------------------------------------------------------

    def _current_list(self) -> Optional[WorkflowLaneList]:
        if self.current_lane_name:
            return self.lane_lists.get(self.current_lane_name)

        for lane_name, lst in self.lane_lists.items():
            if lst.currentRow() >= 0:
                self.current_lane_name = lane_name
                return lst
        return None

    def _current_item(self) -> Optional[QListWidgetItem]:
        lst = self._current_list()
        return lst.currentItem() if lst else None

    def _on_any_step_selected(self, lane_list: WorkflowLaneList):
        # If this lane no longer has a valid selection/current item, ignore it.
        if lane_list.currentItem() is None and not lane_list.selectedItems():
            return

        # Clear BOTH selection and current item in all other lane lists.
        for other_name, other_list in self.lane_lists.items():
            if other_list is lane_list:
                continue

            other_list.blockSignals(True)
            try:
                other_list.clearSelection()
                other_list.setCurrentRow(-1)
                other_list.setCurrentItem(None)
                sm = other_list.selectionModel()
                if sm is not None:
                    sm.clearSelection()
                    sm.clearCurrentIndex()
            finally:
                other_list.blockSignals(False)

        self.current_lane_name = lane_list.lane_name
        self._update_selected_step_info()
        self._refresh_step_visuals()

    def _style_step_item(self, item: QListWidgetItem, is_current: bool, completed: bool):
        text = str(item.text())
        if completed and not text.startswith("✓ "):
            item.setText(f"✓ {text}")
        elif not completed and text.startswith("✓ "):
            item.setText(text[2:])

        font = item.font()
        font.setBold(is_current)
        item.setFont(font)
        paint_workflow_row(item, is_current)

    def _refresh_step_visuals(self):
        current_item = self._current_item()
        for lst in self.lane_lists.values():
            for i in range(lst.count()):
                item = lst.item(i)
                completed = bool(item.data(Qt.ItemDataRole.UserRole + 7))
                self._style_step_item(item, item is current_item, completed)
        self._graph_restyle()

    def _update_selected_step_info(self):
        item = self._current_item()
        if item is None:
            self.lbl_step_info.setText("Select a step to view details.")
            return

        kind = str(item.data(Qt.ItemDataRole.UserRole + 4) or "action")
        cid = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
        note = str(item.data(Qt.ItemDataRole.UserRole + 2) or "")
        completed = bool(item.data(Qt.ItemDataRole.UserRole + 7))
        inputs = list(item.data(Qt.ItemDataRole.UserRole + 5) or [])
        outputs = list(item.data(Qt.ItemDataRole.UserRole + 6) or [])

        if kind == "action":
            info = self.catalog.get(cid)
            title = info.text if info else cid
            tip = info.status_tip if info else ""
        elif kind == "note":
            title = "Note Step"
            tip = "Instruction-only step; no tool is launched."
        elif kind == "split":
            title = "Split Marker"
            tip = "Use this to indicate a branch point into multiple lanes."
        elif kind == "choice":
            title = str(item.data(Qt.ItemDataRole.UserRole + 9) or "").strip() or "Choice"
            tip = "Choice group. Open it to pick one of the tools listed under it."
        else:
            title = "Merge Marker"
            tip = "Use this to indicate a merge point where lanes come back together."

        parts = [f"<b>{title}</b>"]
        if self.current_lane_name:
            parts.append(f"<b>Lane:</b> {self.current_lane_name}")
        if tip:
            parts.append(tip)
        if note:
            parts.append(f"<br><b>Note:</b> {note}")
        if inputs:
            parts.append(f"<br><b>Inputs:</b> {', '.join(inputs)}")
        if outputs:
            parts.append(f"<br><b>Outputs:</b> {', '.join(outputs)}")

        preset = item.data(Qt.ItemDataRole.UserRole + 8)
        if isinstance(preset, dict) and preset:
            parts.append(f"<br><b>Preset:</b> {_summarize_preset(preset)}")

        alts = normalize_workflow_alternatives(item.data(Qt.ItemDataRole.UserRole + 10))
        if alts:
            names = [_choice_label(a.command_id, a.preset, a.label_override, self.catalog) for a in alts]
            parts.append(f"<br><b>Alternatives:</b> {', '.join(names)}")

        lst = self._current_list()
        parent_id = str(item.data(Qt.ItemDataRole.UserRole + 11) or "")
        if lst is not None and parent_id:
            for i in range(lst.count()):
                other = lst.item(i)
                if str(other.data(Qt.ItemDataRole.UserRole) or "") == parent_id:
                    group_name = str(other.data(Qt.ItemDataRole.UserRole + 9) or "").strip() or "Choice"
                    completed = bool(other.data(Qt.ItemDataRole.UserRole + 7))
                    parts.append(f"<br><b>Part of:</b> {group_name}")
                    break
        if kind == "choice" and lst is not None:
            sid = str(item.data(Qt.ItemDataRole.UserRole) or "")
            choice_names = []
            for i in range(lst.count()):
                other = lst.item(i)
                if str(other.data(Qt.ItemDataRole.UserRole + 11) or "") != sid:
                    continue
                child = unpack_workflow_step_item(other, self.current_lane_name or "main")
                choice_names.append(format_workflow_step_label(child, self.catalog))
            if choice_names:
                parts.append(f"<br><b>Choices:</b> {', '.join(choice_names)}")

        top_id = parent_id or str(item.data(Qt.ItemDataRole.UserRole) or "")
        for line in workflow_link_lines(self.current_workflow, top_id, self.catalog):
            parts.append(f"<br>{line}")

        parts.append(f"<br><b>Status:</b> {'Completed' if completed else 'Not completed'}")

        self.lbl_step_info.setText("<br>".join(parts))

    # ------------------------------------------------------------------
    # Add/edit/remove steps
    # ------------------------------------------------------------------

    def add_step_by_command_id(self, command_id: str, lane_name: Optional[str] = None,
                               preset: Optional[dict] = None, name: Optional[str] = None,
                               parent_id: str = "", insert_at: Optional[int] = None):
        lane_name = lane_name or self.current_lane_name or (self.current_workflow.lanes[0] if self.current_workflow.lanes else "main")
        if lane_name not in self.current_workflow.lanes:
            self.current_workflow.lanes.append(lane_name)
            self.refresh_workflow_view()

        step = WorkflowStep(
            command_id=command_id, kind="action", lane=lane_name,
            preset=dict(preset) if isinstance(preset, dict) else {},
            parent_id=parent_id,
        )

        # Label priority: explicit dropped chip name > derived-from-preset > catalog default.
        info = self.catalog.get(command_id)
        _catalog_text = info.text if info else ""
        _name = str(name or "").strip()
        if _name and _name != _catalog_text:
            step.label_override = _name
        elif step.preset:
            _lbl = _derive_step_label(command_id, step.preset, self.catalog)
            if _lbl and _lbl != _catalog_text:
                step.label_override = _lbl

        lst = self.lane_lists.get(lane_name)
        if lst is None:
            return

        item = self._make_step_item(step)
        if insert_at is None:
            lst.addItem(item)
        else:
            lst.insertItem(max(0, insert_at), item)
        lst.setCurrentItem(item)
        self.current_lane_name = lane_name
        self.sync_steps_from_ui()
        if parent_id:
            self._layout_select_id = step.id
            self._layout_after_drag = True
            if self._drag_depth == 0:
                QTimer.singleShot(0, self._flush_layout_after_drag)
        else:
            self._refresh_step_visuals()

    def _new_tool_placement(self) -> dict:
        item = self._current_item()
        lst = self._current_list()
        if item is None or lst is None:
            return {}
        step = unpack_workflow_step_item(item, lst.lane_name)
        row = lst.row(item)
        if step.kind == "choice":
            return {"lane_name": lst.lane_name, "parent_id": step.id, "insert_at": row + 1}
        if step.kind == "action" and step.parent_id:
            return {"lane_name": lst.lane_name, "parent_id": step.parent_id, "insert_at": row + 1}
        return {}

    def _add_choice_group(self, lane_name: Optional[str] = None):
        name, ok = QInputDialog.getText(self, "Add Choice", "Choice name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "Add Choice", "Give the choice a name.")
            return
        lane_name = lane_name or self.current_lane_name or (self.current_workflow.lanes[0] if self.current_workflow.lanes else "main")
        if lane_name not in self.current_workflow.lanes:
            self.current_workflow.lanes.append(lane_name)
            self.refresh_workflow_view()
        step = WorkflowStep(kind="choice", label_override=name, lane=lane_name)
        lst = self.lane_lists.get(lane_name)
        if lst is None:
            return
        item = self._make_step_item(step)
        lst.addItem(item)
        lst.setCurrentItem(item)
        self.current_lane_name = lane_name
        self.sync_steps_from_ui()
        self._refresh_step_visuals()
        self._update_selected_step_info()

    def add_special_step(self, kind: str, lane_name: Optional[str] = None):
        lane_name = lane_name or self.current_lane_name or (self.current_workflow.lanes[0] if self.current_workflow.lanes else "main")
        if lane_name not in self.current_workflow.lanes:
            self.current_workflow.lanes.append(lane_name)
            self.refresh_workflow_view()

        default_note = {
            "note": "Add guidance here.",
            "split": "Branch starts here.",
            "merge": "Branches merge here.",
        }.get(kind, "")

        step = WorkflowStep(kind=kind, note=default_note, lane=lane_name)
        lst = self.lane_lists.get(lane_name)
        if lst is None:
            return

        item = self._make_step_item(step)
        lst.addItem(item)
        lst.setCurrentItem(item)
        self.current_lane_name = lane_name
        self.sync_steps_from_ui()
        self._refresh_step_visuals()

    def _on_tool_double_clicked(self, item: QTreeWidgetItem):
        cid = item.data(0, Qt.ItemDataRole.UserRole)
        if cid:
            self.add_step_by_command_id(str(cid), **self._new_tool_placement())

    def _add_selected_tool(self):
        item = self.tool_tree.currentItem()
        if item is None:
            return
        cid = item.data(0, Qt.ItemDataRole.UserRole)
        if cid:
            self.add_step_by_command_id(str(cid), **self._new_tool_placement())

    def _remove_selected_step(self):
        lst = self._current_list()
        if lst is None:
            return
        row = lst.currentRow()
        if row < 0:
            return
        item = lst.item(row)
        kind = str(item.data(Qt.ItemDataRole.UserRole + 4) or "")
        if kind == "choice":
            choice_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
            next_id = ""
            seen = False
            before = ""
            for i in range(lst.count()):
                other = lst.item(i)
                sid = str(other.data(Qt.ItemDataRole.UserRole) or "")
                pid = str(other.data(Qt.ItemDataRole.UserRole + 11) or "")
                if sid == choice_id or pid == choice_id:
                    seen = seen or sid == choice_id
                    continue
                if not seen:
                    before = sid
                elif not next_id:
                    next_id = sid
            self.current_workflow.steps = normalize_workflow_steps(
                workflow_steps_without(self._raw_steps_from_ui(), choice_id)
            )
            self.refresh_workflow_view(select_id=next_id or before or None)
            return
        lst.takeItem(row)
        self.sync_steps_from_ui()
        self._update_selected_step_info()
        self._refresh_step_visuals()

    def _rename_selected_step(self):
        item = self._current_item()
        if item is None:
            return
        current = str(item.data(Qt.ItemDataRole.UserRole + 9) or "")
        kind = str(item.data(Qt.ItemDataRole.UserRole + 4) or "")
        text, ok = QInputDialog.getText(
            self, "Rename Step", "Display label (blank = default):", text=current)
        if not ok:
            return
        if kind == "choice" and not text.strip():
            QMessageBox.warning(self, "Rename Step", "A choice needs a name.")
            return
        item.setData(Qt.ItemDataRole.UserRole + 9, text.strip())
        # rebuild the visible label from model so the ◆/✓ prefixes stay correct
        self.sync_steps_from_ui()
        self.refresh_workflow_view()

    def _edit_selected_step_preset(self):
        item = self._current_item()
        if item is None:
            return
        cid = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
        if not cid:
            return
        current = item.data(Qt.ItemDataRole.UserRole + 8)
        current = dict(current) if isinstance(current, dict) else {}
        try:
            from setiastro.saspro.shortcuts import _open_preset_editor_for_command
        except Exception:
            QMessageBox.information(self, "Edit Preset", "No preset editor is available.")
            return
        result = _open_preset_editor_for_command(self, cid, current)
        if result is None:
            return  # cancelled or no editor for this tool
        item.setData(Qt.ItemDataRole.UserRole + 8, dict(result))
        self.sync_steps_from_ui()
        self.refresh_workflow_view()
        self._update_selected_step_info()

    def _edit_selected_step_note(self):
        item = self._current_item()
        if item is None:
            return
        current = str(item.data(Qt.ItemDataRole.UserRole + 2) or "")
        text, ok = QInputDialog.getMultiLineText(self, "Edit Step Note", "Note:", current)
        if not ok:
            return
        item.setData(Qt.ItemDataRole.UserRole + 2, text)
        self.sync_steps_from_ui()
        self._update_selected_step_info()

    def _edit_selected_step_io(self):
        item = self._current_item()
        if item is None:
            return

        current_inputs = ", ".join(item.data(Qt.ItemDataRole.UserRole + 5) or [])
        current_outputs = ", ".join(item.data(Qt.ItemDataRole.UserRole + 6) or [])

        inputs_text, ok1 = QInputDialog.getText(
            self,
            "Edit Inputs",
            "Inputs (comma-separated):",
            text=current_inputs,
        )
        if not ok1:
            return

        outputs_text, ok2 = QInputDialog.getText(
            self,
            "Edit Outputs",
            "Outputs (comma-separated):",
            text=current_outputs,
        )
        if not ok2:
            return

        inputs = [x.strip() for x in inputs_text.split(",") if x.strip()]
        outputs = [x.strip() for x in outputs_text.split(",") if x.strip()]

        item.setData(Qt.ItemDataRole.UserRole + 5, inputs)
        item.setData(Qt.ItemDataRole.UserRole + 6, outputs)
        self.sync_steps_from_ui()
        self._update_selected_step_info()

    def _clear_workflow(self):
        any_steps = any(lst.count() > 0 for lst in self.lane_lists.values())
        if not any_steps:
            return
        if QMessageBox.question(self, "Clear Workflow", "Remove all steps from the current workflow?") != QMessageBox.StandardButton.Yes:
            return

        self.current_workflow.steps.clear()
        self.current_workflow.lanes = ["main"]
        self.current_workflow.links = []
        self.current_workflow.graph_rows = {}
        self.current_lane_name = None
        self._graph_request_fit()
        self.refresh_workflow_view()

    def _new_workflow(self):
        self.current_workflow = WorkflowDefinition(name="Untitled Workflow", lanes=["main"], links=[])
        self.current_lane_name = None
        self._graph_request_fit()
        self.refresh_workflow_view()

    # ------------------------------------------------------------------
    # Lanes
    # ------------------------------------------------------------------

    def _add_lane(self):
        name, ok = QInputDialog.getText(self, "Add Lane", "Lane name:", text="new_lane")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self.current_workflow.lanes:
            QMessageBox.warning(self, "Lane Exists", f"A lane named '{name}' already exists.")
            return

        self.current_workflow.lanes.append(name)
        self.refresh_workflow_view()
        self.current_lane_name = name

    def _ask_lane(self, title: str, ok_text: str, rename: bool = False):
        """Let the user pick a lane (and, for a rename, type its new name).

        Returns (lane, new_name), or None when cancelled. An empty lane has no
        step to select, so the buttons cannot go by the current selection.
        """
        lanes = list(self.current_workflow.lanes)
        if not lanes:
            return None
        counts = {lane: (lst.count() if lst is not None else 0)
                  for lane, lst in ((lane, self.lane_lists.get(lane)) for lane in lanes)}

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(340)
        form = QFormLayout(dlg)
        combo = QComboBox(dlg)
        for lane in lanes:
            n = counts[lane]
            combo.addItem(f"{lane}   ({'empty' if n == 0 else f'{n} step' + ('s' if n != 1 else '')})", lane)
        start = self.current_lane_name if self.current_lane_name in lanes else lanes[0]
        combo.setCurrentIndex(lanes.index(start))
        form.addRow("Lane:", combo)
        edt = None
        if rename:
            edt = QLineEdit(start, dlg)
            combo.currentIndexChanged.connect(lambda i: (edt.setText(lanes[i]), edt.selectAll()))
            form.addRow("New name:", edt)
            edt.selectAll()
            edt.setFocus()
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=dlg)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        lane = lanes[combo.currentIndex()]
        return lane, (edt.text().strip() if edt is not None else "")

    def _rename_current_lane(self):
        """Rename Lane button: ask which lane, and its new name."""
        if len(self.current_workflow.lanes) == 1:
            self._rename_lane(self.current_workflow.lanes[0])
            return
        picked = self._ask_lane("Rename Lane", "Rename", rename=True)
        if picked is not None:
            self._apply_lane_rename(*picked)

    def _rename_lane(self, lane_name: str):
        """Rename a known lane (graph header, lane menu): ask only for the new name."""
        if lane_name not in self.current_workflow.lanes:
            return
        name, ok = QInputDialog.getText(self, "Rename Lane", "New lane name:", text=lane_name)
        if ok:
            self._apply_lane_rename(lane_name, name)

    def _apply_lane_rename(self, lane_name: str, name: str):
        name = name.strip()
        if not name or name == lane_name:
            return
        if name in self.current_workflow.lanes:
            QMessageBox.warning(self, "Lane Exists", f"A lane named '{name}' already exists.")
            return

        self.sync_steps_from_ui()

        self.current_workflow.lanes = [name if x == lane_name else x for x in self.current_workflow.lanes]
        for step in self.current_workflow.steps:
            if step.lane == lane_name:
                step.lane = name
        for link in self.current_workflow.links or []:
            if link.dst_lane == lane_name:
                link.dst_lane = name

        self.current_lane_name = name
        self.refresh_workflow_view()

    def _remove_current_lane(self):
        """Remove Lane button: ask which lane."""
        if len(self.current_workflow.lanes) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "A workflow must have at least one lane.")
            return
        picked = self._ask_lane("Remove Lane", "Remove…")
        if picked is not None:
            self._remove_lane(picked[0])

    def _remove_lane(self, lane_name: str):
        """Remove a lane and its steps. Asks first when the lane has steps (there is no undo)."""
        if lane_name not in self.current_workflow.lanes:
            return
        if len(self.current_workflow.lanes) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "A workflow must have at least one lane.")
            return

        self.sync_steps_from_ui()
        n = sum(1 for s in self.current_workflow.steps if s.lane == lane_name)
        if n and QMessageBox.question(
            self,
            "Remove Lane",
            f"Remove lane '{lane_name}' and its {n} step{'s' if n != 1 else ''}?",
        ) != QMessageBox.StandardButton.Yes:
            return

        self.current_workflow.lanes = [x for x in self.current_workflow.lanes if x != lane_name]
        self.current_workflow.steps = [s for s in self.current_workflow.steps if s.lane != lane_name]
        if self.current_lane_name not in self.current_workflow.lanes:
            self.current_lane_name = self.current_workflow.lanes[0] if self.current_workflow.lanes else None
        self.refresh_workflow_view()

    # ------------------------------------------------------------------
    # Load/save
    # ------------------------------------------------------------------

    def _load_workflow(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Workflow",
            str(self.workflows_dir),
            "Workflow Files (*.json)",
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            wf = WorkflowDefinition.from_dict(data)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not load workflow:\n\n{e}")
            return

        self.load_workflow_definition(wf)

    def _save_workflow(self):
        name = self.edt_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing Name", "Please give the workflow a name first.")
            return

        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip() or "workflow"
        path = self.workflows_dir / f"{safe}.json"
        self._write_workflow_file(path)

    def _save_workflow_as(self):
        name = self.edt_name.text().strip() or "workflow"
        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip() or "workflow"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workflow As",
            str(self.workflows_dir / f"{safe}.json"),
            "Workflow Files (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        self._write_workflow_file(Path(path))

    def _autosave_path(self) -> Path:
        return self.workflows_dir / "_autosave_last_session.json"

    def _write_workflow_file(self, path: Path):
        self.sync_steps_from_ui()
        self.current_workflow.name = self.edt_name.text().strip() or "Untitled Workflow"
        self.current_workflow.description = self.edt_description.toPlainText().strip()

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.current_workflow.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save workflow:\n\n{e}")
            return

        QMessageBox.information(self, "Workflow Saved", f"Saved workflow to:\n{path}")

    # ------------------------------------------------------------------
    # Running / progress
    # ------------------------------------------------------------------

    def _run_selected_step(self):
        item = self._current_item()
        if item is None:
            return

        kind = str(item.data(Qt.ItemDataRole.UserRole + 4) or "action")
        lane = self.current_lane_name or "main"
        lst = self._current_list()
        step = unpack_workflow_step_item(item, lane=lane)
        if kind == "choice":
            if lst is not None and open_choice_group(self, self.main, lst, step, self.catalog):
                self._refresh_step_visuals()
            return
        if kind != "action":
            QMessageBox.information(
                self,
                "Instruction Step",
                "This step is a note/split/merge marker and does not launch a tool.",
            )
            return

        choices = workflow_step_choices(step, self.catalog)
        if not choices:
            QMessageBox.warning(self, "No Command", "This step has no command ID.")
            return
        picked = prompt_workflow_choice(self, choices)
        if picked is None:
            return
        command_id, preset, _label = picked
        if open_workflow_command(self, self.main, command_id, preset):
            self._refresh_step_visuals()

    def _rewrite_step_item(self, item: QListWidgetItem, step: WorkflowStep) -> None:
        completed = bool(item.data(Qt.ItemDataRole.UserRole + 7))
        pack_workflow_step_item(item, step)
        item.setData(Qt.ItemDataRole.UserRole + 7, completed)
        item.setIcon(QIcon())
        if step.kind == "action":
            info = self.catalog.get(step.command_id)
            if info and not info.icon.isNull():
                item.setIcon(info.icon)
        item.setText(format_workflow_lane_label(step, self.catalog))
        self.sync_steps_from_ui()
        self._update_selected_step_info()
        self._refresh_step_visuals()

    def _step_from_current_item(self) -> tuple[Optional[QListWidgetItem], Optional[WorkflowStep]]:
        item = self._current_item()
        if item is None:
            return None, None
        lst = self._current_list()
        lane = lst.lane_name if lst is not None else (self.current_lane_name or "main")
        return item, unpack_workflow_step_item(item, lane=lane)

    def _add_alternative_to_selected(self):
        item, step = self._step_from_current_item()
        if item is None or step is None:
            return
        if step.kind == "choice":
            cid = pick_catalog_command(self, self.catalog)
            if not cid:
                return
            self._append_choice_child(step, cid)
            return
        if step.kind != "action" or not step.command_id:
            return
        cid = pick_catalog_command(self, self.catalog)
        if not cid:
            return
        step.alternatives.append(WorkflowAlternative(command_id=cid))
        self._rewrite_step_item(item, step)

    def _append_choice_child(self, choice: WorkflowStep, command_id: str):
        lst = self.lane_lists.get(choice.lane)
        if lst is None:
            return
        insert_at = lst.count()
        for i in range(lst.count()):
            sid = str(lst.item(i).data(Qt.ItemDataRole.UserRole) or "")
            pid = str(lst.item(i).data(Qt.ItemDataRole.UserRole + 11) or "")
            if sid == choice.id or pid == choice.id:
                insert_at = i + 1
        self.add_step_by_command_id(
            command_id, lane_name=choice.lane, parent_id=choice.id, insert_at=insert_at,
        )

    def _remove_alternative_at(self, index: int):
        item, step = self._step_from_current_item()
        if item is None or step is None:
            return
        if not (0 <= index < len(step.alternatives)):
            return
        del step.alternatives[index]
        self._rewrite_step_item(item, step)

    def _edit_alternative_preset(self, index: int):
        item, step = self._step_from_current_item()
        if item is None or step is None:
            return
        if not (0 <= index < len(step.alternatives)):
            return
        alt = step.alternatives[index]
        try:
            from setiastro.saspro.shortcuts import _open_preset_editor_for_command
        except Exception:
            QMessageBox.information(self, "Edit Preset", "No preset editor is available.")
            return
        result = _open_preset_editor_for_command(self, alt.command_id, dict(alt.preset or {}))
        if result is None:
            return
        alt.preset = dict(result)
        self._rewrite_step_item(item, step)

    def _mark_selected_complete(self):
        item = self._current_item()
        lst = self._current_list()
        if item is None or lst is None:
            return
        mark_group_complete(lst, item)
        self._refresh_step_visuals()
        self._update_selected_step_info()

    def _reset_progress(self):
        for lst in self.lane_lists.values():
            for i in range(lst.count()):
                lst.item(i).setData(Qt.ItemDataRole.UserRole + 7, False)
        self._refresh_step_visuals()
        self._update_selected_step_info()

    def _ordered_items(self) -> List[tuple[str, WorkflowLaneList, int, QListWidgetItem]]:
        out = []
        for lane_name in self.current_workflow.lanes:
            lst = self.lane_lists.get(lane_name)
            if lst is None:
                continue
            for i in range(lst.count()):
                out.append((lane_name, lst, i, lst.item(i)))
        return out

    def _prev_step(self):
        flat = self._ordered_items()
        cur = self._current_item()
        if not flat or cur is None:
            return
        idx = next((i for i, (_, _, _, item) in enumerate(flat) if item is cur), None)
        if idx is None:
            return
        idx = max(0, idx - 1)
        lane_name, lst, row, _ = flat[idx]
        self.current_lane_name = lane_name
        lst.setCurrentRow(row)

    def _next_step(self):
        flat = self._ordered_items()
        cur = self._current_item()
        if not flat:
            return
        if cur is None:
            lane_name, lst, row, _ = flat[0]
            self.current_lane_name = lane_name
            lst.setCurrentRow(row)
            return

        idx = next((i for i, (_, _, _, item) in enumerate(flat) if item is cur), None)
        if idx is None:
            return
        idx = min(len(flat) - 1, idx + 1)
        lane_name, lst, row, _ = flat[idx]
        self.current_lane_name = lane_name
        lst.setCurrentRow(row)

    # ------------------------------------------------------------------
    # Graph commands (see _WorkflowGraphHost)
    # ------------------------------------------------------------------

    graph_editable = True
    _GRAPH_SETTINGS_PREFIX = "workflow_dialog"

    def _graph_wf(self) -> WorkflowDefinition:
        return self.current_workflow

    def _graph_complete_selected(self):
        self._mark_selected_complete()

    def graph_add_tool(self, command_id: str):
        """Palette double-click: into the selected choice, else just below the selected step."""
        place = self._new_tool_placement()
        item, step = self._step_from_current_item()
        lst = self._current_list()
        if place or step is None or lst is None:
            self.add_step_by_command_id(command_id, **place)
            return
        top_rows = [r for r in range(lst.count())
                    if not str(lst.item(r).data(Qt.ItemDataRole.UserRole + 11) or "")]
        here = lst.row(item)
        self.graph_insert([{"command_id": command_id}], lst.lane_name,
                          sum(1 for r in top_rows if r <= here))

    def _unique_lane_name(self, base: str) -> str:
        lanes = set(self.current_workflow.lanes)
        if base not in lanes:
            return base
        n = 2
        while f"{base} {n}" in lanes:
            n += 1
        return f"{base} {n}"

    def _graph_step_title(self, step_id: str) -> str:
        for s in self.current_workflow.steps:
            if s.id == step_id:
                return format_workflow_step_label(s, self.catalog)
        return step_id

    def graph_node_menu(self, menu: QMenu, step_id: str):
        if not self.graph_select(step_id):
            return
        item, step = self._step_from_current_item()
        if step is None:
            return
        menu.addAction("Open Step", self._run_selected_step)
        menu.addAction("Mark Complete", self._mark_selected_complete)
        menu.addSeparator()
        menu.addAction("Rename…", self._rename_selected_step)
        if step.kind == "choice":
            menu.addAction("Add Alternative…", self._add_alternative_to_selected)
        elif step.kind == "action" and step.command_id:
            menu.addAction("Edit Preset…", self._edit_selected_step_preset)
            menu.addAction("Add Alternative…", self._add_alternative_to_selected)
            if step.alternatives:
                rm = menu.addMenu("Remove Alternative")
                ed = menu.addMenu("Edit Alternative Preset")
                for idx, alt in enumerate(step.alternatives):
                    label = _choice_label(alt.command_id, alt.preset, alt.label_override, self.catalog)
                    rm.addAction(label, lambda _c=False, i=idx: self._remove_alternative_at(i))
                    ed.addAction(label, lambda _c=False, i=idx: self._edit_alternative_preset(i))
        menu.addAction("Edit Note…", self._edit_selected_step_note)
        menu.addAction("Edit Inputs / Outputs…", self._edit_selected_step_io)

        menu.addSeparator()
        top_id = step.parent_id or step.id
        menu.addAction("Branch Into New Lane…",
                       lambda _c=False: self.graph_branch_new_lane(top_id))
        if top_id in self.current_workflow.graph_rows:
            menu.addAction("Auto Position", lambda _c=False: self.graph_set_row(top_id, None))
        own = [l for l in self.current_workflow.links or [] if top_id in (l.src, l.dst)]
        if own:
            sub = menu.addMenu("Remove Link")
            for link in own:
                sub.addAction(self._graph_link_text(link),
                              lambda _c=False, l=link: self.graph_unlink(l))
        menu.addSeparator()
        menu.addAction("Remove Step", self._remove_selected_step)

    def _graph_link_text(self, link: WorkflowLink) -> str:
        src = self._graph_step_title(link.src)
        dst = self._graph_step_title(link.dst) if link.dst else f"lane {link.dst_lane}"
        return f"{src}  →  {dst}"

    def graph_lane_menu(self, menu: QMenu, lane: str):
        lanes = self.current_workflow.lanes
        idx = lanes.index(lane) if lane in lanes else -1
        menu.addAction("Add Note", lambda _c=False: self.add_special_step("note", lane_name=lane))
        menu.addAction("Add Choice…", lambda _c=False: self._add_choice_group(lane))
        menu.addAction("Add Split Marker", lambda _c=False: self.add_special_step("split", lane_name=lane))
        menu.addAction("Add Merge Marker", lambda _c=False: self.add_special_step("merge", lane_name=lane))
        menu.addSeparator()
        menu.addAction("Rename Lane…", lambda _c=False: self._rename_lane(lane))
        act = menu.addAction("Move Lane Left", lambda _c=False: self.graph_move_lane(lane, -1))
        act.setEnabled(idx > 0)
        act = menu.addAction("Move Lane Right", lambda _c=False: self.graph_move_lane(lane, +1))
        act.setEnabled(0 <= idx < len(lanes) - 1)
        menu.addSeparator()
        menu.addAction("Remove Lane…", lambda _c=False: self._remove_lane(lane))

    def graph_canvas_menu(self, menu: QMenu):
        menu.addAction("Add Lane…", self._add_lane)
        if normalize_graph_rows(self.current_workflow.graph_rows, self.current_workflow.steps):
            menu.addAction("Auto Position All Steps", lambda _c=False: self.graph_reset_rows())

    def graph_link_menu(self, menu: QMenu, link: WorkflowLink):
        menu.addAction("Remove Link", lambda _c=False: self.graph_unlink(link))

    def graph_set_row(self, step_id: str, row: Optional[int]):
        """Pin a step to a graph row, or (None) give it back its automatic row."""
        rows = self.current_workflow.graph_rows
        if row is None:
            if rows.pop(step_id, None) is None:
                return
        elif rows.get(step_id) == int(row):
            return
        else:
            rows[step_id] = int(row)
        self.graph_refresh()

    def graph_reset_rows(self):
        if self.current_workflow.graph_rows:
            self.current_workflow.graph_rows = {}
            self.graph_refresh()

    def graph_rename_lane(self, lane: str):
        self._rename_lane(lane)

    def graph_add_lane(self):
        self._add_lane()

    def graph_delete(self, step_id: str):
        if self.graph_select(step_id):
            self._remove_selected_step()

    def _graph_insert_row(self, lst: WorkflowLaneList, index: int, parent_id: str) -> int:
        """List row for the index-th top-level slot of a lane, or just after a choice's group."""
        if parent_id:
            after = None
            for row in range(lst.count()):
                it = lst.item(row)
                if parent_id in (str(it.data(Qt.ItemDataRole.UserRole) or ""),
                                 str(it.data(Qt.ItemDataRole.UserRole + 11) or "")):
                    after = row
            return lst.count() if after is None else after + 1
        top_rows = [row for row in range(lst.count())
                    if not str(lst.item(row).data(Qt.ItemDataRole.UserRole + 11) or "")]
        return top_rows[index] if 0 <= index < len(top_rows) else lst.count()

    def graph_insert(self, payloads: List[dict], lane: str, index: int, parent_id: str = ""):
        """Add dropped tools at a lane position. An empty lane name starts a new lane."""
        if not payloads:
            return
        if not lane:
            self.sync_steps_from_ui()
            lane = self._unique_lane_name("lane")
            self.current_workflow.lanes.append(lane)
            self.refresh_workflow_view()
        lst = self.lane_lists.get(lane)
        if lst is None:
            return
        for n, p in enumerate(payloads):
            row = self._graph_insert_row(lst, index + n, parent_id)
            self.add_step_by_command_id(
                str(p.get("command_id") or ""), lane_name=lane,
                preset=p.get("preset") if isinstance(p.get("preset"), dict) else {},
                name=str(p.get("name") or ""), parent_id=parent_id, insert_at=row,
            )

    def graph_move(self, step_id: str, lane: str, index: int):
        """Move a top-level step (a choice brings its actions) to a lane position."""
        self.sync_steps_from_ui()
        wf = self.current_workflow
        moving = next((s for s in wf.steps if s.id == step_id), None)
        if moving is None or moving.parent_id:
            return
        lane = lane or self._unique_lane_name("lane")
        group_ids = {s.id for s in wf.steps if s.id == step_id or s.parent_id == step_id}
        group = [s for s in wf.steps if s.id in group_ids]
        rest = [s for s in wf.steps if s.id not in group_ids]
        lane_top = [i for i, s in enumerate(rest) if s.lane == lane and not s.parent_id]
        if 0 <= index < len(lane_top):
            pos = lane_top[index]
        else:
            in_lane = [i for i, s in enumerate(rest) if s.lane == lane]
            pos = in_lane[-1] + 1 if in_lane else len(rest)
        new_steps = rest[:pos] + group + rest[pos:]
        if moving.lane == lane and [s.id for s in new_steps] == [s.id for s in wf.steps]:
            return
        for s in group:
            s.lane = lane
        if lane not in wf.lanes:
            wf.lanes.append(lane)
        wf.steps = normalize_workflow_steps(new_steps)
        wf.graph_rows.pop(step_id, None)      # a moved step starts again from its automatic row
        dropped: list = []
        wf.links = normalize_workflow_links(wf.links, wf.steps, wf.lanes, dropped)
        self.refresh_workflow_view(select_id=step_id)
        if dropped:
            n = len(dropped)
            why = {r for _l, r in dropped}
            text = "now in the same lane" if why == {"same_lane"} else "it would no longer make sense"
            self._graph_hint(f"Removed {n} link{'s' if n > 1 else ''}: {text}.")

    def graph_link(self, src: str, dst: str = "", dst_lane: str = "") -> str:
        """Add a branch (dst_lane) or a join (dst). Returns why it was refused, or ""."""
        self.sync_steps_from_ui()
        wf = self.current_workflow
        new = WorkflowLink(src=src, dst=dst, dst_lane="" if dst else dst_lane)
        dropped: list = []
        links = normalize_workflow_links(list(wf.links or []) + [new], wf.steps, wf.lanes, dropped)
        for link, reason in dropped:
            if link is new:
                return LINK_DROP_REASONS.get(reason, "That link is not allowed.")
        wf.links = links
        self.graph_refresh()
        self._update_selected_step_info()
        return ""

    def graph_unlink(self, link: WorkflowLink):
        wf = self.current_workflow
        key = (link.src, link.dst, link.dst_lane)
        wf.links = [l for l in wf.links or [] if (l.src, l.dst, l.dst_lane) != key]
        self.graph_refresh()
        self._update_selected_step_info()

    def graph_branch_new_lane(self, src_id: str, at_end: bool = False):
        """Start a new lane fed by src_id, placed after src's lane (or last)."""
        self.sync_steps_from_ui()
        wf = self.current_workflow
        src = next((s for s in wf.steps if s.id == src_id), None)
        if src is None:
            return
        src = next((s for s in wf.steps if s.id == src.parent_id), src)
        name, ok = QInputDialog.getText(self, "Branch Into New Lane", "New lane name:",
                                        text=self._unique_lane_name("branch"))
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in wf.lanes:
            QMessageBox.warning(self, "Lane Exists", f"A lane named '{name}' already exists.")
            return
        at = len(wf.lanes)
        if not at_end and src.lane in wf.lanes:
            at = wf.lanes.index(src.lane) + 1
        wf.lanes.insert(at, name)
        wf.links = list(wf.links or []) + [WorkflowLink(src=src.id, dst_lane=name)]
        self.refresh_workflow_view(select_id=src.id)
        self._graph_hint(f"Drag tools onto lane '{name}' to fill the branch.")

    def graph_move_lane(self, lane: str, delta: int):
        self.sync_steps_from_ui()
        lanes = self.current_workflow.lanes
        if lane not in lanes:
            return
        i = lanes.index(lane)
        j = i + delta
        if not (0 <= j < len(lanes)):
            return
        lanes[i], lanes[j] = lanes[j], lanes[i]
        self.refresh_workflow_view(select_id=self.graph_current_id() or None)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _on_name_changed(self, text: str):
        self.current_workflow.name = text.strip() or "Untitled Workflow"

    def _on_description_changed(self):
        self.current_workflow.description = self.edt_description.toPlainText().strip()

    # ------------------------------------------------------------------
    # Geometry persistence
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._save_geometry()
        self._autosave_session()
        try:
            if getattr(self.main, "_workflow_dialog", None) is self:
                self.main._workflow_dialog = None
        except Exception:
            pass
        super().closeEvent(event)

    def _autosave_session(self):
        """Silently persist the current workflow so reopening restores it."""
        try:
            self.sync_steps_from_ui()
            self.current_workflow.name = self.edt_name.text().strip() or "Untitled Workflow"
            self.current_workflow.description = self.edt_description.toPlainText().strip()
            path = self._autosave_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.current_workflow.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass   # autosave must never block closing

    def _restore_autosaved_session(self):
        """On open, reload the last session if one was autosaved. Silent no-op otherwise."""
        try:
            path = self._autosave_path()
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
            wf = WorkflowDefinition.from_dict(data)
        except Exception:
            return
        # Only restore if there's actually something to restore (steps or a non-default name).
        has_content = any(getattr(wf, "steps", None) for _ in [0]) if hasattr(wf, "steps") else False
        try:
            has_content = bool(wf.steps)
        except Exception:
            has_content = False
        if not has_content and (not wf.name or wf.name == "Untitled Workflow"):
            return
        self.load_workflow_definition(wf)

    def _restore_geometry(self):
        if self.settings is None:
            return
        try:
            g = self.settings.value("workflow_dialog/geometry")
            if g:
                self.restoreGeometry(g)
            s = self.settings.value("workflow_dialog/view_split")
            if s:
                self.view_split.restoreState(s)
        except Exception:
            pass

    def _save_geometry(self):
        if self.settings is None:
            return
        try:
            self.settings.setValue("workflow_dialog/geometry", self.saveGeometry())
            self.settings.setValue("workflow_dialog/view_split", self.view_split.saveState())
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Mini lane list (read-only drag, execution-only)
# -----------------------------------------------------------------------------

class MiniWorkflowLaneList(QListWidget):
    """
    A read-only (no editing) lane list for MiniWorkflowDialog.
    Double-click opens the step; no drag-drop editing.
    """
    def __init__(self, dialog: "MiniWorkflowDialog", lane_name: str, parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self.lane_name = lane_name

        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setItemDelegate(WorkflowStepDelegate(self))

        self.currentRowChanged.connect(self._notify_selection)
        self.itemSelectionChanged.connect(self._notify_selection_changed)
        self.itemDoubleClicked.connect(self._dialog._run_selected_step)

    def _notify_selection_changed(self):
        self._dialog._on_any_step_selected(self)

    def _notify_selection(self, _row: int):
        self._dialog._on_any_step_selected(self)


# -----------------------------------------------------------------------------
# Mini workflow dialog
# -----------------------------------------------------------------------------

class MiniWorkflowDialog(_WorkflowGraphHost, QDialog):
    """
    A compact, execution-only view of a workflow.

    Shows lanes and steps, lets the user step through with Previous / Open Step /
    Mark Complete / Next Step / Reset Progress.  Mark Complete automatically
    advances to the next step.  Double-clicking a step opens it directly.

    No editing: the full WorkflowDialog handles that.
    """

    _GRAPH_SETTINGS_PREFIX = "mini_workflow_dialog"

    def __init__(self, main, workflow: WorkflowDefinition, catalog: Dict[str, WorkflowActionInfo], parent=None):
        super().__init__(parent or main)
        self.main = main
        self.settings = _main_settings(main)

        self.workflow = workflow
        self.catalog = catalog
        ensure_workflow_links(self.workflow)

        self.lane_lists: Dict[str, MiniWorkflowLaneList] = {}
        self.current_lane_name: Optional[str] = None
        self._graph_init()

        self.setWindowTitle(f"Running: {workflow.name}")
        self.resize(860, 560)

        self._build_ui()
        self._populate()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Workflow name header
        title_row = QHBoxLayout()
        self.lbl_title = QLabel(self.workflow.name, self)
        self.lbl_title.setStyleSheet("font-size: 13px; font-weight: 700; padding: 2px 0;")
        title_row.addWidget(self.lbl_title, 1)
        self.btn_graph = self._make_graph_button()
        title_row.addWidget(self.btn_graph)
        root.addLayout(title_row)

        # Scrollable lanes area
        self.lanes_scroll = QScrollArea(self)
        self.lanes_scroll.setWidgetResizable(True)
        self.lanes_host = QWidget(self)
        self.lanes_layout = QHBoxLayout(self.lanes_host)
        self.lanes_layout.setContentsMargins(4, 4, 4, 4)
        self.lanes_layout.setSpacing(8)
        self.lanes_scroll.setWidget(self.lanes_host)
        root.addWidget(self.lanes_scroll, 1)

        # Step detail label
        self.lbl_step_info = QLabel("Select a step to view details.", self)
        self.lbl_step_info.setWordWrap(True)
        self.lbl_step_info.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_step_info.setMinimumHeight(72)
        self.lbl_step_info.setContentsMargins(6, 4, 6, 4)
        root.addWidget(self.lbl_step_info)

        # Navigation buttons
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀  Previous", self)
        self.btn_run = QPushButton("Open Step", self)
        self.btn_done = QPushButton("✓  Mark Complete", self)
        self.btn_next = QPushButton("Next  ▶", self)
        self.btn_reset = QPushButton("Reset Progress", self)

        self.btn_prev.setMinimumWidth(100)
        self.btn_run.setMinimumWidth(110)
        self.btn_done.setMinimumWidth(130)
        self.btn_next.setMinimumWidth(100)

        self.btn_prev.clicked.connect(self._prev_step)
        self.btn_run.clicked.connect(self._run_selected_step)
        self.btn_done.clicked.connect(self._mark_complete_and_advance)
        self.btn_next.clicked.connect(self._next_step)
        self.btn_reset.clicked.connect(self._reset_progress)

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_run)
        nav.addSpacing(8)
        nav.addWidget(self.btn_done)
        nav.addSpacing(8)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        nav.addWidget(self.btn_reset)
        nav.addSpacing(8)
        nav.addWidget(make_help_button("workflow_assistant", self))
        root.addLayout(nav)

    # ------------------------------------------------------------------
    # Populate lanes from workflow
    # ------------------------------------------------------------------

    def _populate(self):
        # Clear existing lane widgets
        while self.lanes_layout.count():
            item = self.lanes_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.lane_lists.clear()

        lanes = self.workflow.lanes or ["main"]

        for lane_name in lanes:
            card = QWidget(self.lanes_host)
            card.setMinimumWidth(220)
            card_lay = QVBoxLayout(card)
            card_lay.setContentsMargins(4, 4, 4, 4)
            card_lay.setSpacing(4)

            header = QLabel(lane_name, card)
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.setFrameShape(QFrame.Shape.StyledPanel)
            header.setStyleSheet("font-weight: 600; padding: 4px;")
            card_lay.addWidget(header)

            lst = MiniWorkflowLaneList(self, lane_name=lane_name, parent=card)
            card_lay.addWidget(lst, 1)

            self.lane_lists[lane_name] = lst
            self.lanes_layout.addWidget(card)

        self.lanes_layout.addStretch(1)

        # Distribute steps into their lanes
        by_lane: Dict[str, List[WorkflowStep]] = {lane: [] for lane in lanes}
        for step in self.workflow.steps:
            target = step.lane if step.lane in by_lane else lanes[0]
            by_lane[target].append(step)

        for lane_name, steps in by_lane.items():
            lst = self.lane_lists.get(lane_name)
            if lst is None:
                continue
            for step in steps:
                lst.addItem(self._make_item(step))

        # Select first available step
        self.current_lane_name = None
        for lane_name in lanes:
            lst = self.lane_lists.get(lane_name)
            if lst and lst.count() > 0:
                self.current_lane_name = lane_name
                lst.setCurrentRow(0)
                break

        self._graph_request_fit()
        self.graph_refresh()
        self._refresh_visuals()
        self._update_info()

    def _make_item(self, step: WorkflowStep) -> QListWidgetItem:
        label = self._step_label(step)
        item = QListWidgetItem(label)

        if step.kind == "action":
            info = self.catalog.get(step.command_id)
            if info and not info.icon.isNull():
                item.setIcon(info.icon)

        pack_workflow_step_item(item, step)
        return item

    def _step_label(self, step: WorkflowStep) -> str:
        return format_workflow_lane_label(step, self.catalog)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_any_step_selected(self, lane_list: MiniWorkflowLaneList):
        if lane_list.currentItem() is None and not lane_list.selectedItems():
            return

        for other_list in self.lane_lists.values():
            if other_list is lane_list:
                continue
            other_list.blockSignals(True)
            try:
                other_list.clearSelection()
                other_list.setCurrentRow(-1)
                other_list.setCurrentItem(None)
                sm = other_list.selectionModel()
                if sm is not None:
                    sm.clearSelection()
                    sm.clearCurrentIndex()
            finally:
                other_list.blockSignals(False)

        self.current_lane_name = lane_list.lane_name
        self._refresh_visuals()
        self._update_info()

    def _current_list(self) -> Optional[MiniWorkflowLaneList]:
        if self.current_lane_name:
            lst = self.lane_lists.get(self.current_lane_name)
            if lst is not None:
                return lst
        for lst in self.lane_lists.values():
            if lst.currentRow() >= 0:
                self.current_lane_name = lst.lane_name
                return lst
        return None

    def _current_item(self) -> Optional[QListWidgetItem]:
        lst = self._current_list()
        return lst.currentItem() if lst else None

    # ------------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------------

    def _refresh_visuals(self):
        current_item = self._current_item()
        for lst in self.lane_lists.values():
            for i in range(lst.count()):
                item = lst.item(i)
                completed = bool(item.data(Qt.ItemDataRole.UserRole + 7))
                text = item.text()

                if completed and not text.startswith("✓ "):
                    item.setText(f"✓ {text}")
                elif not completed and text.startswith("✓ "):
                    item.setText(text[2:])

                font = item.font()
                font.setBold(item is current_item)
                item.setFont(font)
                paint_workflow_row(item, item is current_item)
        self._graph_restyle()

    def _update_info(self):
        item = self._current_item()
        if item is None:
            self.lbl_step_info.setText("Select a step to view details.")
            return

        kind      = str(item.data(Qt.ItemDataRole.UserRole + 4) or "action")
        cid       = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
        note      = str(item.data(Qt.ItemDataRole.UserRole + 2) or "")
        completed = bool(item.data(Qt.ItemDataRole.UserRole + 7))
        inputs    = list(item.data(Qt.ItemDataRole.UserRole + 5) or [])
        outputs   = list(item.data(Qt.ItemDataRole.UserRole + 6) or [])

        if kind == "action":
            info  = self.catalog.get(cid)
            title = info.text if info else cid
            tip   = (info.status_tip if info else "") or ""
        elif kind == "note":
            title = "Note Step"
            tip   = "Instruction-only step; no tool is launched."
        elif kind == "split":
            title = "Split Marker"
            tip   = "Branch point — processing continues in multiple lanes."
        elif kind == "choice":
            title = str(item.data(Qt.ItemDataRole.UserRole + 9) or "").strip() or "Choice"
            tip   = "Choice group. Open it to pick one of the tools listed under it."
        else:
            title = "Merge Marker"
            tip   = "Merge point — lanes come back together here."

        parts = [f"<b>{title}</b>"]
        if self.current_lane_name:
            parts.append(f"<b>Lane:</b> {self.current_lane_name}")
        if tip:
            parts.append(tip)
        if note:
            parts.append(f"<br><b>Note:</b> {note}")
        if inputs:
            parts.append(f"<b>Inputs:</b> {', '.join(inputs)}")
        if outputs:
            parts.append(f"<b>Outputs:</b> {', '.join(outputs)}")

        preset = item.data(Qt.ItemDataRole.UserRole + 8)
        if isinstance(preset, dict) and preset:
            parts.append(f"<b>Preset:</b> {_summarize_preset(preset)}")

        alts = normalize_workflow_alternatives(item.data(Qt.ItemDataRole.UserRole + 10))
        if alts:
            names = [_choice_label(a.command_id, a.preset, a.label_override, self.catalog) for a in alts]
            parts.append(f"<b>Alternatives:</b> {', '.join(names)}")

        lst = self._current_list()
        parent_id = str(item.data(Qt.ItemDataRole.UserRole + 11) or "")
        if lst is not None and parent_id:
            for i in range(lst.count()):
                other = lst.item(i)
                if str(other.data(Qt.ItemDataRole.UserRole) or "") == parent_id:
                    group_name = str(other.data(Qt.ItemDataRole.UserRole + 9) or "").strip() or "Choice"
                    completed = bool(other.data(Qt.ItemDataRole.UserRole + 7))
                    parts.append(f"<b>Part of:</b> {group_name}")
                    break
        if kind == "choice" and lst is not None:
            sid = str(item.data(Qt.ItemDataRole.UserRole) or "")
            choice_names = []
            for i in range(lst.count()):
                other = lst.item(i)
                if str(other.data(Qt.ItemDataRole.UserRole + 11) or "") != sid:
                    continue
                child = unpack_workflow_step_item(other, self.current_lane_name or "main")
                choice_names.append(format_workflow_step_label(child, self.catalog))
            if choice_names:
                parts.append(f"<b>Choices:</b> {', '.join(choice_names)}")

        top_id = parent_id or str(item.data(Qt.ItemDataRole.UserRole) or "")
        parts.extend(workflow_link_lines(self.workflow, top_id, self.catalog))

        parts.append(f"<b>Status:</b> {'✓ Completed' if completed else 'Not completed'}")

        self.lbl_step_info.setText("<br>".join(parts))

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _ordered_items(self) -> List[tuple]:
        out = []
        for lane_name in (self.workflow.lanes or ["main"]):
            lst = self.lane_lists.get(lane_name)
            if lst is None:
                continue
            for i in range(lst.count()):
                out.append((lane_name, lst, i, lst.item(i)))
        return out

    def _prev_step(self):
        flat = self._ordered_items()
        cur  = self._current_item()
        if not flat or cur is None:
            return
        idx = next((i for i, (_, _, _, it) in enumerate(flat) if it is cur), None)
        if idx is None:
            return
        idx = max(0, idx - 1)
        lane_name, lst, row, _ = flat[idx]
        self.current_lane_name = lane_name
        lst.setCurrentRow(row)

    def _next_step(self):
        flat = self._ordered_items()
        cur  = self._current_item()
        if not flat:
            return
        if cur is None:
            lane_name, lst, row, _ = flat[0]
            self.current_lane_name = lane_name
            lst.setCurrentRow(row)
            return
        idx = next((i for i, (_, _, _, it) in enumerate(flat) if it is cur), None)
        if idx is None:
            return
        idx = min(len(flat) - 1, idx + 1)
        lane_name, lst, row, _ = flat[idx]
        self.current_lane_name = lane_name
        lst.setCurrentRow(row)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _run_selected_step(self):
        item = self._current_item()
        if item is None:
            return

        kind = str(item.data(Qt.ItemDataRole.UserRole + 4) or "action")
        lane = self.current_lane_name or "main"
        lst = self._current_list()
        step = unpack_workflow_step_item(item, lane=lane)
        if kind == "choice":
            if lst is not None:
                open_choice_group(self, self.main, lst, step, self.catalog)
            return
        if kind != "action":
            note = str(item.data(Qt.ItemDataRole.UserRole + 2) or "")
            msg  = note if note else "This step is a marker — no tool to launch."
            QMessageBox.information(self, "Instruction Step", msg)
            return

        choices = workflow_step_choices(step, self.catalog)
        if not choices:
            QMessageBox.warning(self, "No Command", "This step has no command ID.")
            return
        picked = prompt_workflow_choice(self, choices)
        if picked is None:
            return
        command_id, preset, _label = picked
        open_workflow_command(self, self.main, command_id, preset)

    def _mark_complete_and_advance(self):
        item = self._current_item()
        lst = self._current_list()
        if item is None or lst is None:
            return

        target = mark_group_complete(lst, item)
        self._refresh_visuals()
        self._update_info()

        flat = self._ordered_items()
        target_id = str(target.data(Qt.ItemDataRole.UserRole) or "")
        cur_idx = next((i for i, (_, _, _, it) in enumerate(flat) if it is target), None)
        if cur_idx is None:
            return
        nxt = cur_idx
        while nxt + 1 < len(flat):
            _lane, _lst, _row, it = flat[nxt + 1]
            if str(it.data(Qt.ItemDataRole.UserRole + 11) or "") != target_id:
                break
            nxt += 1
        if nxt + 1 < len(flat):
            next_lane, next_lst, next_row, _ = flat[nxt + 1]
            self.current_lane_name = next_lane
            next_lst.setCurrentRow(next_row)
        else:
            self.current_lane_name = lst.lane_name
            lst.setCurrentItem(target)

    def _reset_progress(self):
        for lst in self.lane_lists.values():
            for i in range(lst.count()):
                lst.item(i).setData(Qt.ItemDataRole.UserRole + 7, False)
        self._refresh_visuals()
        self._update_info()

    def _graph_wf(self) -> WorkflowDefinition:
        return self.workflow

    def _graph_complete_selected(self):
        self._mark_complete_and_advance()

    # ------------------------------------------------------------------
    # Geometry persistence
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._save_geometry()
        try:
            if getattr(self.main, "_mini_workflow_dialog", None) is self:
                self.main._mini_workflow_dialog = None
        except Exception:
            pass
        super().closeEvent(event)

    def _restore_geometry(self):
        if self.settings is None:
            return
        try:
            g = self.settings.value("mini_workflow_dialog/geometry")
            if g:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_geometry(self):
        if self.settings is None:
            return
        try:
            self.settings.setValue("mini_workflow_dialog/geometry", self.saveGeometry())
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Public entry points
# -----------------------------------------------------------------------------

def show_workflow_dialog(main):
    dlg = getattr(main, "_workflow_dialog", None)
    if dlg is None:
        dlg = WorkflowDialog(main, parent=main)
        main._workflow_dialog = dlg

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()

def show_mini_workflow_dialog(main):
    full_dlg: Optional[WorkflowDialog] = getattr(main, "_workflow_dialog", None)

    if full_dlg is not None and full_dlg.current_workflow.steps:
        workflow = full_dlg.current_workflow
        catalog  = full_dlg.catalog
        _launch_mini(main, workflow, catalog)
        return

    # Nothing loaded yet — ask the user what they want to do
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QComboBox

    picker = QDialog(main)
    picker.setWindowTitle("Run Workflow")
    lay = QVBoxLayout(picker)

    lay.addWidget(QLabel("No workflow is currently loaded.\nChoose how to get started:"))

    combo = QComboBox(picker)
    combo.addItem("Open Workflow Assistant (build or edit)")
    for wf in _default_canned_workflows():
        combo.addItem(f"Canned: {wf.name}", userData=wf)

    wf_dir = _workflows_dir(main)
    saved_paths = sorted(wf_dir.glob("*.json"))
    for p in saved_paths:
        combo.addItem(f"Saved: {p.stem}", userData=p)

    lay.addWidget(combo)

    btns = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        parent=picker,
    )
    btns.accepted.connect(picker.accept)
    btns.rejected.connect(picker.reject)
    lay.addWidget(btns)

    if picker.exec() != QDialog.DialogCode.Accepted:
        return

    idx = combo.currentIndex()
    data = combo.currentData()

    if idx == 0:
        # Open the full assistant
        show_workflow_dialog(main)
        return

    if isinstance(data, WorkflowDefinition):
        wf = data
    elif isinstance(data, Path):
        try:
            wf = WorkflowDefinition.from_dict(
                json.loads(data.read_text(encoding="utf-8"))
            )
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(main, "Load Failed", f"Could not load workflow:\n\n{e}")
            return
    else:
        return

    # Build a catalog without spinning up the full dialog UI
    categories = _discover_toolbar_categories(main)
    for cat_name, items in _discover_script_categories(main).items():
        categories.setdefault(cat_name, []).extend(items)
    catalog: Dict[str, WorkflowActionInfo] = {}
    for items in categories.values():
        for info in items:
            catalog[info.command_id] = info

    _launch_mini(main, wf, catalog)


def _launch_mini(main, workflow: WorkflowDefinition, catalog: Dict[str, WorkflowActionInfo]):
    existing: Optional[MiniWorkflowDialog] = getattr(main, "_mini_workflow_dialog", None)
    if existing is not None:
        existing.close()
    dlg = MiniWorkflowDialog(main, workflow=workflow, catalog=catalog, parent=main)
    main._mini_workflow_dialog = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()