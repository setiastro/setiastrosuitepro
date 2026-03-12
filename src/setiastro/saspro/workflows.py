# src/setiastro/saspro/workflows.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
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


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass
class WorkflowStep:
    command_id: str
    note: str = ""
    enabled: bool = True


@dataclass
class WorkflowDefinition:
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [asdict(s) for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowDefinition":
        steps = []
        for s in data.get("steps", []):
            if isinstance(s, dict) and s.get("command_id"):
                steps.append(
                    WorkflowStep(
                        command_id=str(s.get("command_id", "")),
                        note=str(s.get("note", "")),
                        enabled=bool(s.get("enabled", True)),
                    )
                )
        return cls(
            name=str(data.get("name", "Untitled Workflow")),
            description=str(data.get("description", "")),
            steps=steps,
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
    text = text.replace("&", "").strip()
    return text


def _main_settings(main) -> object | None:
    return getattr(main, "settings", None)


def _workflows_dir(main) -> Path:
    """
    Prefer a user data path if main provides one.
    Otherwise fall back to a sensible local folder under the home directory.
    """
    # Optional future hook if you already have a resource/appdata helper
    getter_names = (
        "get_user_data_dir",
        "_user_data_dir",
        "user_data_dir",
    )
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
    """
    Looks up a QAction using the ShortcutManager first, then falls back to
    scanning actions on the main window.
    """
    # Preferred: ShortcutManager API if present
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

        # common private patterns
        for attr in ("_actions", "actions_by_id", "registry", "_registry"):
            mapping = getattr(shortcuts, attr, None)
            if isinstance(mapping, dict):
                act = mapping.get(command_id)
                if isinstance(act, QAction):
                    return act

    # Fallback: scan all actions attached to main
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
    """
    Discover actions grouped by toolbar membership so the workflow catalog feels
    familiar to users.

    We intentionally ignore separators and actions without a command_id.
    """
    categories: Dict[str, List[WorkflowActionInfo]] = {}

    try:
        toolbars = main.findChildren(QWidget)
    except Exception:
        toolbars = []

    # We only care about objects that behave like QToolBar / your DraggableToolBar
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
            steps=[
                WorkflowStep("crop", "Crop away stacking edges first."),
                WorkflowStep("abe", "Remove gradients while the image is still linear."),
                WorkflowStep("background_neutral", "Neutralize the background."),
                WorkflowStep("white_balance", "Balance color before stretching."),
                WorkflowStep("cosmicclarity", "Apply Sharpening then noise reduction"),
                WorkflowStep("stat_stretch", "Stretch to non-linear."),
                WorkflowStep("curves", "Adjust the contrast."),
                WorkflowStep("clahe", "Boost the image depth slightly."),
                WorkflowStep("save_as", "Save your result."),
            ],
        ),
        WorkflowDefinition(
            name="Beginner Mono Narrowband Workflow",
            description="Simple starter workflow for mono narrowband processing.",
            steps=[
                WorkflowStep("linear_fit", "Match channels before combining if needed."),
                WorkflowStep("remove_stars", "Remove your stars for each master"),
                WorkflowStep("ppp", "Build the palette and color image with Perfect Palette Picker."),
                WorkflowStep("nbtorgb", "Combine your stars only images into an RGB stars master"),
                WorkflowStep("cosmicclarity", "Apply denoise and/or sharpening with Cosmic Clarity."),
                WorkflowStep("curves", "Refine contrast and color."),
                WorkflowStep("clahe", "Enhance local contrast if needed."),
                WorkflowStep("add_stars", "Combine your stars and your starless image"),
                WorkflowStep("save_as", "Save your result."),
            ],
        ),
        WorkflowDefinition(
            name="Cosmic Clarity Cleanup",
            description="Basic sharpening / denoise workflow.",
            steps=[
                WorkflowStep("crop", "Optional crop first."),
                WorkflowStep("abe", "Optional gradient cleanup."),
                WorkflowStep("cosmicclarity", "Run Cosmic Clarity."),
                WorkflowStep("curves", "Refine the result."),
                WorkflowStep("save_as", "Save your result."),
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
# Drag/drop workflow step list
# -----------------------------------------------------------------------------

class WorkflowStepList(QListWidget):
    def __init__(self, dialog: "WorkflowDialog", parent=None):
        super().__init__(parent)
        self._dialog = dialog
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

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
        if md.hasFormat(WORKFLOW_MIME):
            try:
                cid = bytes(md.data(WORKFLOW_MIME)).decode("utf-8")
            except Exception:
                cid = ""
            if cid:
                self._dialog.add_step_by_command_id(cid)
                event.acceptProposedAction()
                return
        super().dropEvent(event)
        self._dialog.sync_steps_from_list()


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
        self.resize(1200, 700)

        self.current_workflow = WorkflowDefinition(name="Untitled Workflow")
        self.current_step_index = -1

        self.catalog: Dict[str, WorkflowActionInfo] = {}
        self.canned_workflows = _default_canned_workflows()

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

        # ---------------- Right: workflow editor/runner ----------------
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

        self.step_list = WorkflowStepList(self, self)
        self.step_list.currentRowChanged.connect(self._on_step_selected)
        self.step_list.itemDoubleClicked.connect(self._run_selected_step)
        right_lay.addWidget(self.step_list, 1)

        self.lbl_step_info = QLabel("Select a step to view details.", self)
        self.lbl_step_info.setWordWrap(True)
        self.lbl_step_info.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_step_info.setMinimumHeight(64)
        right_lay.addWidget(self.lbl_step_info)

        # Buttons row 1: edit workflow
        row1 = QHBoxLayout()
        self.btn_add_selected = QPushButton("Add Selected Tool", self)
        self.btn_remove = QPushButton("Remove Step", self)
        self.btn_note = QPushButton("Edit Note", self)
        self.btn_clear = QPushButton("Clear Workflow", self)

        self.btn_add_selected.clicked.connect(self._add_selected_tool)
        self.btn_remove.clicked.connect(self._remove_selected_step)
        self.btn_note.clicked.connect(self._edit_selected_step_note)
        self.btn_clear.clicked.connect(self._clear_workflow)

        row1.addWidget(self.btn_add_selected)
        row1.addWidget(self.btn_remove)
        row1.addWidget(self.btn_note)
        row1.addWidget(self.btn_clear)
        row1.addStretch(1)
        right_lay.addLayout(row1)

        # Buttons row 2: runner
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
        splitter.setSizes([420, 760])

        # Bottom file/canned actions
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
    # Catalog population
    # ------------------------------------------------------------------

    def _populate_action_catalog(self):
        self.tool_tree.clear()
        self.catalog.clear()

        categories = _discover_toolbar_categories(self.main)

        # Order roughly like your toolbar layout
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
        ordered = [c for c in preferred_order if c in categories] + [
            c for c in category_names if c not in preferred_order
        ]

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

        top_count = self.tool_tree.topLevelItemCount()
        for i in range(top_count):
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

        self.step_list.clear()

        for idx, step in enumerate(self.current_workflow.steps):
            info = self.catalog.get(step.command_id)
            label = info.text if info else step.command_id

            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, step.command_id)
            item.setData(Qt.ItemDataRole.UserRole + 1, step.note)
            item.setData(Qt.ItemDataRole.UserRole + 2, bool(step.enabled))
            item.setData(Qt.ItemDataRole.UserRole + 3, False)  # completed flag

            if info and not info.icon.isNull():
                item.setIcon(info.icon)

            self._style_step_item(item, idx == self.current_step_index, completed=False)
            self.step_list.addItem(item)

        if self.step_list.count() > 0:
            row = self.current_step_index if 0 <= self.current_step_index < self.step_list.count() else 0
            self.step_list.setCurrentRow(row)
        else:
            self.current_step_index = -1
            self.lbl_step_info.setText("Select a step to view details.")

    def sync_steps_from_list(self):
        steps: List[WorkflowStep] = []
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            steps.append(
                WorkflowStep(
                    command_id=str(item.data(Qt.ItemDataRole.UserRole) or ""),
                    note=str(item.data(Qt.ItemDataRole.UserRole + 1) or ""),
                    enabled=bool(item.data(Qt.ItemDataRole.UserRole + 2)),
                )
            )
        self.current_workflow.steps = steps

    def add_step_by_command_id(self, command_id: str):
        info = self.catalog.get(command_id)
        label = info.text if info else command_id

        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, command_id)
        item.setData(Qt.ItemDataRole.UserRole + 1, "")
        item.setData(Qt.ItemDataRole.UserRole + 2, True)
        item.setData(Qt.ItemDataRole.UserRole + 3, False)

        if info and not info.icon.isNull():
            item.setIcon(info.icon)

        self.step_list.addItem(item)
        self.sync_steps_from_list()

        if self.current_step_index < 0:
            self.current_step_index = 0

        self.step_list.setCurrentItem(item)

    def load_workflow_definition(self, wf: WorkflowDefinition):
        self.current_workflow = WorkflowDefinition(
            name=wf.name,
            description=wf.description,
            steps=[WorkflowStep(s.command_id, s.note, s.enabled) for s in wf.steps],
        )
        self.current_step_index = 0 if self.current_workflow.steps else -1
        self.refresh_workflow_view()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

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
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            completed = bool(item.data(Qt.ItemDataRole.UserRole + 3))
            self._style_step_item(item, i == self.current_step_index, completed)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

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
        row = self.step_list.currentRow()
        if row < 0:
            return
        self.step_list.takeItem(row)
        self.sync_steps_from_list()

        if self.step_list.count() == 0:
            self.current_step_index = -1
        elif self.current_step_index >= self.step_list.count():
            self.current_step_index = self.step_list.count() - 1

        self._refresh_step_visuals()

    def _edit_selected_step_note(self):
        row = self.step_list.currentRow()
        if row < 0:
            return
        item = self.step_list.item(row)
        current = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
        text, ok = QInputDialog.getMultiLineText(self, "Edit Step Note", "Note:", current)
        if not ok:
            return
        item.setData(Qt.ItemDataRole.UserRole + 1, text)
        self.sync_steps_from_list()
        self._update_step_info(row)

    def _clear_workflow(self):
        if self.step_list.count() == 0:
            return
        if QMessageBox.question(
            self,
            "Clear Workflow",
            "Remove all steps from the current workflow?",
        ) != QMessageBox.StandardButton.Yes:
            return

        self.current_workflow.steps.clear()
        self.current_step_index = -1
        self.refresh_workflow_view()

    def _new_workflow(self):
        self.current_workflow = WorkflowDefinition(name="Untitled Workflow")
        self.current_step_index = -1
        self.refresh_workflow_view()

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

        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip()
        if not safe:
            safe = "workflow"
        path = self.workflows_dir / f"{safe}.json"
        self._write_workflow_file(path)

    def _save_workflow_as(self):
        name = self.edt_name.text().strip() or "workflow"
        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name).strip()
        if not safe:
            safe = "workflow"

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
        self.sync_steps_from_list()
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

    def _run_selected_step(self):
        row = self.step_list.currentRow()
        if row < 0:
            return

        item = self.step_list.item(row)
        command_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not command_id:
            QMessageBox.warning(self, "No Command", "This step has no command ID.")
            return

        action = _find_action_by_command_id(self.main, command_id)
        if action is None:
            QMessageBox.warning(
                self,
                "Action Not Found",
                f"Could not find QAction for command_id:\n{command_id}",
            )
            return

        try:
            action.trigger()
            self.current_step_index = row
            self._refresh_step_visuals()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Step Failed",
                f"Error while opening workflow step:\n\n{command_id}\n\n{e}",
            )

    def _mark_selected_complete(self):
        row = self.step_list.currentRow()
        if row < 0:
            return
        item = self.step_list.item(row)
        item.setData(Qt.ItemDataRole.UserRole + 3, True)
        self.current_step_index = row
        self._refresh_step_visuals()

    def _reset_progress(self):
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            item.setData(Qt.ItemDataRole.UserRole + 3, False)
        self.current_step_index = 0 if self.step_list.count() else -1
        self._refresh_step_visuals()
        if self.current_step_index >= 0:
            self.step_list.setCurrentRow(self.current_step_index)

    def _prev_step(self):
        if self.step_list.count() == 0:
            return
        row = self.step_list.currentRow()
        if row < 0:
            row = 0
        row = max(0, row - 1)
        self.current_step_index = row
        self.step_list.setCurrentRow(row)
        self._refresh_step_visuals()

    def _next_step(self):
        if self.step_list.count() == 0:
            return
        row = self.step_list.currentRow()
        if row < 0:
            row = 0
        row = min(self.step_list.count() - 1, row + 1)
        self.current_step_index = row
        self.step_list.setCurrentRow(row)
        self._refresh_step_visuals()

    # ------------------------------------------------------------------
    # Selection / metadata
    # ------------------------------------------------------------------

    def _on_step_selected(self, row: int):
        self._update_step_info(row)
        if row >= 0:
            self.current_step_index = row
        self._refresh_step_visuals()

    def _update_step_info(self, row: int):
        if row < 0 or row >= self.step_list.count():
            self.lbl_step_info.setText("Select a step to view details.")
            return

        item = self.step_list.item(row)
        cid = str(item.data(Qt.ItemDataRole.UserRole) or "")
        note = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
        completed = bool(item.data(Qt.ItemDataRole.UserRole + 3))

        info = self.catalog.get(cid)
        title = info.text if info else cid
        tip = info.status_tip if info else ""

        parts = [f"<b>{title}</b>"]
        if tip:
            parts.append(tip)
        if note:
            parts.append(f"<br><b>Note:</b> {note}")
        if completed:
            parts.append("<br><b>Status:</b> Completed")
        else:
            parts.append("<br><b>Status:</b> Not completed")

        self.lbl_step_info.setText("<br>".join(parts))

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