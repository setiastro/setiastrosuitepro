from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QMimeData, QSize
from PyQt6.QtGui import QAction, QDrag, QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
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
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

WORKFLOW_MIME = "application/x-saspro-workflow-command"
WORKFLOW_SCHEMA_VERSION = 2


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

def _new_step_id() -> str:
    return uuid.uuid4().hex[:12]


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

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "command_id": self.command_id,
            "note": self.note,
            "enabled": self.enabled,
            "kind": self.kind,
            "lane": self.lane,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowStep":
        return cls(
            id=str(data.get("id") or _new_step_id()),
            command_id=str(data.get("command_id", "")),
            note=str(data.get("note", "")),
            enabled=bool(data.get("enabled", True)),
            kind=str(data.get("kind", "action")),
            lane=str(data.get("lane", "main")),
            inputs=[str(x) for x in data.get("inputs", []) if str(x).strip()],
            outputs=[str(x) for x in data.get("outputs", []) if str(x).strip()],
        )


@dataclass
class WorkflowDefinition:
    name: str
    description: str = ""
    lanes: List[str] = field(default_factory=lambda: ["main"])
    steps: List[WorkflowStep] = field(default_factory=list)
    schema_version: int = WORKFLOW_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": WORKFLOW_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "lanes": list(self.lanes),
            "steps": [s.to_dict() for s in self.steps],
        }

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

            return cls(
                name=str(data.get("name", "Untitled Workflow")),
                description=str(data.get("description", "")),
                lanes=lanes,
                steps=steps,
                schema_version=int(data.get("schema_version", WORKFLOW_SCHEMA_VERSION)),
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
            steps=steps,
            schema_version=1,
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

        drag.exec(Qt.DropAction.CopyAction)


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

        self.currentRowChanged.connect(self._notify_selection)
        self.itemSelectionChanged.connect(self._notify_selection_changed)
        self.itemDoubleClicked.connect(self._dialog._run_selected_step)

    def _notify_selection_changed(self):
        self._dialog._on_any_step_selected(self)

    def _notify_selection(self, _row: int):
        self._dialog._on_any_step_selected(self)

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasFormat(WORKFLOW_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        md = event.mimeData()
        if md.hasFormat(WORKFLOW_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()

        # external tool catalog drop
        if md.hasFormat(WORKFLOW_MIME):
            try:
                cid = bytes(md.data(WORKFLOW_MIME)).decode("utf-8")
            except Exception:
                cid = ""
            if cid:
                self._dialog.add_step_by_command_id(cid, lane_name=self.lane_name)
                event.acceptProposedAction()
                return

        # internal move between lanes / within lane
        super().dropEvent(event)
        self._dialog.sync_steps_from_ui()

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        menu = QMenu(self)

        act_note = menu.addAction("Add Note Step")
        act_split = menu.addAction("Add Split Marker")
        act_merge = menu.addAction("Add Merge Marker")

        if item is not None:
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
        elif item is not None and chosen == act_edit_note:
            self._dialog._edit_selected_step_note()
        elif item is not None and chosen == act_edit_ios:
            self._dialog._edit_selected_step_io()
        elif item is not None and chosen == act_remove:
            self._dialog._remove_selected_step()


# -----------------------------------------------------------------------------
# Main dialog
# -----------------------------------------------------------------------------

class WorkflowDialog(QDialog):
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

        self._build_ui()
        self._populate_action_catalog()

        self._restore_geometry()
        self.refresh_workflow_view()

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
        right_lay.addLayout(lane_row)

        # scrollable lanes
        self.lanes_scroll = QScrollArea(self)
        self.lanes_scroll.setWidgetResizable(True)
        self.lanes_host = QWidget(self)
        self.lanes_layout = QHBoxLayout(self.lanes_host)
        self.lanes_layout.setContentsMargins(4, 4, 4, 4)
        self.lanes_layout.setSpacing(10)
        self.lanes_scroll.setWidget(self.lanes_host)
        right_lay.addWidget(self.lanes_scroll, 1)

        self.lbl_step_info = QLabel("Select a step to view details.", self)
        self.lbl_step_info.setWordWrap(True)
        self.lbl_step_info.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_step_info.setMinimumHeight(96)
        right_lay.addWidget(self.lbl_step_info)

        # edit buttons
        row1 = QHBoxLayout()
        self.btn_add_selected = QPushButton("Add Selected Tool", self)
        self.btn_add_note = QPushButton("Add Note", self)
        self.btn_add_split = QPushButton("Add Split", self)
        self.btn_add_merge = QPushButton("Add Merge", self)
        self.btn_remove = QPushButton("Remove Step", self)
        self.btn_note = QPushButton("Edit Note", self)
        self.btn_io = QPushButton("Edit Inputs / Outputs", self)
        self.btn_clear = QPushButton("Clear Workflow", self)

        self.btn_add_selected.clicked.connect(self._add_selected_tool)
        self.btn_add_note.clicked.connect(lambda: self.add_special_step("note"))
        self.btn_add_split.clicked.connect(lambda: self.add_special_step("split"))
        self.btn_add_merge.clicked.connect(lambda: self.add_special_step("merge"))
        self.btn_remove.clicked.connect(self._remove_selected_step)
        self.btn_note.clicked.connect(self._edit_selected_step_note)
        self.btn_io.clicked.connect(self._edit_selected_step_io)
        self.btn_clear.clicked.connect(self._clear_workflow)

        for w in (
            self.btn_add_selected,
            self.btn_add_note,
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
        bottom.addWidget(self.btn_canned)
        bottom.addStretch(1)

        root.addLayout(bottom)

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def _populate_action_catalog(self):
        self.tool_tree.clear()
        self.catalog.clear()

        categories = _discover_toolbar_categories(self.main)
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
        ]
        category_names = list(categories.keys())
        ordered = [c for c in preferred_order if c in categories] + [c for c in category_names if c not in preferred_order]

        for category in ordered:
            parent = QTreeWidgetItem([category])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            parent.setFirstColumnSpanned(True)
            self.tool_tree.addTopLevelItem(parent)

            for info in categories[category]:
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

    def refresh_workflow_view(self):
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
                lst.addItem(self._make_step_item(step))

        # select first available step if any
        self.current_lane_name = None
        for lane_name in self.current_workflow.lanes:
            lst = self.lane_lists.get(lane_name)
            if lst and lst.count() > 0:
                self.current_lane_name = lane_name
                lst.setCurrentRow(0)
                break

        if self.current_lane_name is None:
            self.lbl_step_info.setText("Select a step to view details.")

        self._refresh_step_visuals()

    def sync_steps_from_ui(self):
        steps: List[WorkflowStep] = []

        for lane_name in self.current_workflow.lanes:
            lst = self.lane_lists.get(lane_name)
            if lst is None:
                continue

            for i in range(lst.count()):
                item = lst.item(i)
                step = WorkflowStep(
                    id=str(item.data(Qt.ItemDataRole.UserRole) or _new_step_id()),
                    command_id=str(item.data(Qt.ItemDataRole.UserRole + 1) or ""),
                    note=str(item.data(Qt.ItemDataRole.UserRole + 2) or ""),
                    enabled=bool(item.data(Qt.ItemDataRole.UserRole + 3)),
                    kind=str(item.data(Qt.ItemDataRole.UserRole + 4) or "action"),
                    lane=lane_name,
                    inputs=list(item.data(Qt.ItemDataRole.UserRole + 5) or []),
                    outputs=list(item.data(Qt.ItemDataRole.UserRole + 6) or []),
                )
                steps.append(step)

        self.current_workflow.steps = steps

    def _make_step_item(self, step: WorkflowStep) -> QListWidgetItem:
        label = self._step_display_text(step)
        item = QListWidgetItem(label)

        if step.kind == "action":
            info = self.catalog.get(step.command_id)
            if info and not info.icon.isNull():
                item.setIcon(info.icon)

        item.setData(Qt.ItemDataRole.UserRole, step.id)
        item.setData(Qt.ItemDataRole.UserRole + 1, step.command_id)
        item.setData(Qt.ItemDataRole.UserRole + 2, step.note)
        item.setData(Qt.ItemDataRole.UserRole + 3, bool(step.enabled))
        item.setData(Qt.ItemDataRole.UserRole + 4, step.kind)
        item.setData(Qt.ItemDataRole.UserRole + 5, list(step.inputs))
        item.setData(Qt.ItemDataRole.UserRole + 6, list(step.outputs))
        item.setData(Qt.ItemDataRole.UserRole + 7, False)  # completed

        return item

    def _step_display_text(self, step: WorkflowStep) -> str:
        if step.kind == "note":
            return "📝 Note"
        if step.kind == "split":
            return "⇢ Split"
        if step.kind == "merge":
            return "⇠ Merge"

        info = self.catalog.get(step.command_id)
        return info.text if info else (step.command_id or "Action")

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
                )
                for s in wf.steps
            ],
            schema_version=wf.schema_version,
        )
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

    def _refresh_step_visuals(self):
        current_item = self._current_item()
        for lst in self.lane_lists.values():
            for i in range(lst.count()):
                item = lst.item(i)
                completed = bool(item.data(Qt.ItemDataRole.UserRole + 7))
                self._style_step_item(item, item is current_item, completed)

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
        parts.append(f"<br><b>Status:</b> {'Completed' if completed else 'Not completed'}")

        self.lbl_step_info.setText("<br>".join(parts))

    # ------------------------------------------------------------------
    # Add/edit/remove steps
    # ------------------------------------------------------------------

    def add_step_by_command_id(self, command_id: str, lane_name: Optional[str] = None):
        lane_name = lane_name or self.current_lane_name or (self.current_workflow.lanes[0] if self.current_workflow.lanes else "main")
        if lane_name not in self.current_workflow.lanes:
            self.current_workflow.lanes.append(lane_name)
            self.refresh_workflow_view()

        step = WorkflowStep(command_id=command_id, kind="action", lane=lane_name)
        lst = self.lane_lists.get(lane_name)
        if lst is None:
            return

        item = self._make_step_item(step)
        lst.addItem(item)
        lst.setCurrentItem(item)
        self.current_lane_name = lane_name
        self.sync_steps_from_ui()
        self._refresh_step_visuals()

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
            self.add_step_by_command_id(str(cid))

    def _add_selected_tool(self):
        item = self.tool_tree.currentItem()
        if item is None:
            return
        cid = item.data(0, Qt.ItemDataRole.UserRole)
        if cid:
            self.add_step_by_command_id(str(cid))

    def _remove_selected_step(self):
        lst = self._current_list()
        if lst is None:
            return
        row = lst.currentRow()
        if row < 0:
            return
        lst.takeItem(row)
        self.sync_steps_from_ui()
        self._update_selected_step_info()
        self._refresh_step_visuals()

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
        self.current_lane_name = None
        self.refresh_workflow_view()

    def _new_workflow(self):
        self.current_workflow = WorkflowDefinition(name="Untitled Workflow", lanes=["main"])
        self.current_lane_name = None
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

    def _rename_current_lane(self):
        lane_name = self.current_lane_name or (self.current_workflow.lanes[0] if self.current_workflow.lanes else None)
        if not lane_name:
            return

        name, ok = QInputDialog.getText(self, "Rename Lane", "New lane name:", text=lane_name)
        if not ok:
            return
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

        self.current_lane_name = name
        self.refresh_workflow_view()

    def _remove_current_lane(self):
        lane_name = self.current_lane_name
        if not lane_name:
            return
        if len(self.current_workflow.lanes) <= 1:
            QMessageBox.warning(self, "Cannot Remove", "A workflow must have at least one lane.")
            return

        if QMessageBox.question(
            self,
            "Remove Lane",
            f"Remove lane '{lane_name}' and all of its steps?",
        ) != QMessageBox.StandardButton.Yes:
            return

        self.sync_steps_from_ui()
        self.current_workflow.lanes = [x for x in self.current_workflow.lanes if x != lane_name]
        self.current_workflow.steps = [s for s in self.current_workflow.steps if s.lane != lane_name]
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
        if kind != "action":
            QMessageBox.information(
                self,
                "Instruction Step",
                "This step is a note/split/merge marker and does not launch a tool.",
            )
            return

        command_id = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
        if not command_id:
            QMessageBox.warning(self, "No Command", "This step has no command ID.")
            return

        action = _find_action_by_command_id(self.main, command_id)
        if action is None:
            QMessageBox.warning(self, "Action Not Found", f"Could not find QAction for command_id:\n{command_id}")
            return

        try:
            action.trigger()
            self._refresh_step_visuals()
        except Exception as e:
            QMessageBox.critical(self, "Step Failed", f"Error while opening workflow step:\n\n{command_id}\n\n{e}")

    def _mark_selected_complete(self):
        item = self._current_item()
        if item is None:
            return
        item.setData(Qt.ItemDataRole.UserRole + 7, True)
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
        try:
            if getattr(self.main, "_workflow_dialog", None) is self:
                self.main._workflow_dialog = None
        except Exception:
            pass
        super().closeEvent(event)

    def _restore_geometry(self):
        if self.settings is None:
            return
        try:
            g = self.settings.value("workflow_dialog/geometry")
            if g:
                self.restoreGeometry(g)
        except Exception:
            pass

    def _save_geometry(self):
        if self.settings is None:
            return
        try:
            self.settings.setValue("workflow_dialog/geometry", self.saveGeometry())
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def show_workflow_dialog(main):
    dlg = getattr(main, "_workflow_dialog", None)
    if dlg is None:
        dlg = WorkflowDialog(main, parent=main)
        main._workflow_dialog = dlg

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()