# pro/function_bundle.py
from __future__ import annotations
import json
from typing import Iterable, List
import sys
from PyQt6.QtCore import Qt, QSettings, QByteArray, QMimeData, QSize, QPoint, QEventLoop
from PyQt6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, 
    QPushButton, QSplitter, QMessageBox, QLabel, QAbstractItemView, QDialogButtonBox,
    QApplication, QMenu, QInputDialog, QPlainTextEdit
)
from PyQt6.QtGui import QDrag, QCloseEvent, QCursor, QShortcut, QKeySequence
from PyQt6.QtCore import  QThread
import time
from pro.dnd_mime import MIME_CMD

def _pin_on_top_mac(win: QDialog):
    if sys.platform == "darwin":
        # Float above normal windows, behave like a palette/tool window
        win.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        win.setWindowFlag(Qt.WindowType.Tool, True)
        # Keep showing even when app deactivates (mac-only attribute)
        try:
            win.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow, True)
        except Exception:
            pass

# ---------- pack/unpack helpers (lazy to avoid circular imports) ----------
def _unpack_cmd_safely(raw: bytes):
    try:
        from pro.shortcuts import _unpack_cmd_payload as _unpack
    except Exception:
        _unpack = None
    if _unpack is not None:
        try:
            return _unpack(raw)
        except Exception:
            pass
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _pack_cmd_safely(payload: dict) -> bytes:
    try:
        from pro.shortcuts import _pack_cmd_payload as _pack
    except Exception:
        _pack = None
    if _pack is not None:
        try:
            data = _pack(payload)
            return bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        except Exception:
            pass
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")

# ---------- helpers ----------
def _find_main_window(w: QWidget):
    p = w.parent()
    while p is not None and not (hasattr(p, "doc_manager") or hasattr(p, "docman")):
        p = p.parent()
    return p

def _resolve_doc_and_subwindow(mw, doc_ptr: int):
    if hasattr(mw, "_find_doc_by_id"):
        d, sw = mw._find_doc_by_id(doc_ptr)
        if d is not None:
            return d, sw
    try:
        for sw in mw.mdi.subWindowList():
            vw = sw.widget()
            d = getattr(vw, "document", None)
            if d is not None and id(d) == int(doc_ptr):
                return d, sw
    except Exception:
        pass
    return None, None

def _find_shortcut_canvas(mw: QWidget | None) -> QWidget | None:
    if not mw:
        return None
    canv = getattr(getattr(mw, "shortcuts", None), "canvas", None)
    if canv:
        return canv
    try:
        from pro.shortcuts import ShortcutCanvas
        return mw.findChild(ShortcutCanvas)
    except Exception:
        return None

# =============================  FunctionBundleChip  =============================
class FunctionBundleChip(QWidget):
    """
    Mini, movable chip for a function-bundle. Parent is the ShortcutCanvas.
    - Left-drag: move inside canvas (smooth, clamped)
    - Ctrl+Drag: start external drag with {"command_id":"function_bundle", "steps":[...]}
    - Drop MIME_CMD: append steps (or expand a dropped function_bundle)
    - Double-click: reopen the dialog (event is accepted)
    """
    def __init__(self, panel: "FunctionBundleDialog", name: str, bundle_key: str, parent_canvas: QWidget):
        super().__init__(parent_canvas)
        
        self.setAcceptDrops(True)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)

        self._panel = panel
        self._bundle_key = bundle_key
        self._dragging = False
        self._grab_offset = None

        self.setObjectName("FunctionBundleChip")
        self.setMinimumSize(240, 44)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet("""
            QWidget#FunctionBundleChip {
                background: rgba(34, 34, 38, 240);
                color: #ddd;
                border: 1px solid #666;
                border-radius: 8px;
            }
            QLabel#title { font-weight: 600; padding-left: 10px; padding-top: 6px; }
            QLabel#count { color:#aaa; padding-right: 8px; }
            QLabel#hint  { color:#bbb; font-size:11px; padding: 0 10px 6px 10px; }
        """)

        v = QVBoxLayout(self); v.setContentsMargins(6, 4, 6, 4); v.setSpacing(0)
        top = QHBoxLayout(); top.setContentsMargins(0,0,0,0)
        self._title = QLabel(name); self._title.setObjectName("title")
        self._count = QLabel("(0)"); self._count.setObjectName("count")
        top.addWidget(self._title); top.addStretch(1); top.addWidget(self._count)
        v.addLayout(top)
        self._hint = QLabel("Drop shortcuts to add • Alt+Drag to apply")
        self._hint.setObjectName("hint")
        v.addWidget(self._hint)

        self._sync_count()

    def _sync_count(self):
        self._count.setText(f"({self._panel.step_count()})")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._grab_offset = ev.position()  # QPointF in widget coords
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept(); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if not (ev.buttons() & Qt.MouseButton.LeftButton) or not self._dragging:
            super().mouseMoveEvent(ev); return

        # Alt → start external drag once (matches app gesture)
        if ev.modifiers() & Qt.KeyboardModifier.AltModifier:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._start_external_drag()
            ev.accept(); return

        parent = self.parentWidget()
        if not parent:
            return

        global_top_left = ev.globalPosition() - (self._grab_offset or ev.position())
        tl = parent.mapFromGlobal(global_top_left.toPoint())
        max_x = max(0, parent.width()  - self.width())
        max_y = max(0, parent.height() - self.height())
        x = min(max(0, tl.x()), max_x)
        y = min(max(0, tl.y()), max_y)
        self.move(x, y)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            ev.accept(); return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        try:
            self._panel.showNormal()
            self._panel.raise_()
            self._panel.activateWindow()
        except Exception:
            pass
        ev.accept()

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_CMD):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasFormat(MIME_CMD):
            payload = _unpack_cmd_safely(bytes(md.data(MIME_CMD)))
            if not isinstance(payload, dict) or not payload.get("command_id"):
                e.ignore(); return
            if payload.get("command_id") == "function_bundle":
                steps = payload.get("steps") or []
                self._panel._append_steps(steps)
            else:
                self._panel._append_steps([payload])
            self._sync_count()
            e.acceptProposedAction()
            return
        e.ignore()

    def _start_external_drag(self):
        payload = {"command_id": "function_bundle", "steps": self._panel.current_steps()}
        md = QMimeData()
        md.setData(MIME_CMD, QByteArray(_pack_cmd_safely(payload)))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.setHotSpot(QPoint(self.width() // 2, self.height() // 2))
        drag.exec(Qt.DropAction.CopyAction)

# helper to create/place the chip on the ShortcutCanvas
def _spawn_function_chip_on_canvas(mw: QWidget, panel: "FunctionBundleDialog",
                                   name: str, bundle_key: str) -> FunctionBundleChip | None:
    canvas = _find_shortcut_canvas(mw)
    if not canvas:
        return None
    chip = FunctionBundleChip(panel, name, bundle_key, parent_canvas=canvas)
    # place near cursor, clamped
    pt = canvas.mapFromGlobal(QCursor.pos()) - chip.rect().center()
    pt.setX(max(0, min(pt.x(), canvas.width() - chip.width())))
    pt.setY(max(0, min(pt.y(), canvas.height() - chip.height())))
    chip.move(pt)
    chip.show()
    chip.raise_()
    return chip

def _activate_target_sw(mw, sw):
    try:
        if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not sw:
            mw.mdi.setActiveSubWindow(sw)
        w = getattr(sw, "widget", lambda: None)()
        if w:
            w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    except Exception:
        pass

# =============================  FunctionBundleDialog  =============================
class FunctionBundleDialog(QDialog):
    SETTINGS_KEY = "functionbundles/v1"

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        _pin_on_top_mac(self)
        self.setWindowTitle("Function Bundles")
        self.setModal(False)
        self.resize(920, 560)
        self.setAcceptDrops(True)

        self._settings = QSettings()
        self._bundles: List[dict] = self._load_all()
        if not self._bundles:
            self._bundles = [{"name": "Function Bundle 1", "steps": []}]

        # left: bundles
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.btn_new = QPushButton("New")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_del = QPushButton("Delete")

        # right: steps
        self.steps = QListWidget()
        self.steps.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.steps.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.steps.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.add_hint = QLabel("Drop shortcuts here to add steps")
        self.add_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_hint.setStyleSheet("color:#aaa; padding:6px; border:1px dashed #666; border-radius:6px;")

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_clear  = QPushButton("Clear Steps")
        self.btn_up     = QPushButton("▲ Move Up")
        self.btn_down   = QPushButton("▼ Move Down")

        self.btn_drag_bundle = QPushButton("Drag Bundle")
        self.btn_run_active  = QPushButton("Apply to Active View")
        self.btn_apply_to_vbundle = QPushButton("Apply to View Bundle…")
        self.btn_chip   = QPushButton("Compress to Chip")

        # layout
        left = QVBoxLayout()
        left.addWidget(QLabel("Function Bundles"))
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_new); row.addWidget(self.btn_dup); row.addWidget(self.btn_del)
        left.addLayout(row)

        right = QVBoxLayout()
        right.addWidget(QLabel("Steps"))
        right.addWidget(self.steps, 1)
        right.addWidget(self.add_hint)
        rrow = QHBoxLayout()
        rrow.addWidget(self.btn_up); rrow.addWidget(self.btn_down)
        rrow.addStretch(1)
        rrow.addWidget(self.btn_remove); rrow.addWidget(self.btn_clear)
        right.addLayout(rrow)
        #right.addWidget(self.btn_drag_bundle)
        right.addWidget(self.btn_run_active)
        right.addWidget(self.btn_apply_to_vbundle)
        right.addWidget(self.btn_chip)

        split = QSplitter()
        wl = QWidget(); wl.setLayout(left)
        wr = QWidget(); wr.setLayout(right)
        split.addWidget(wl); split.addWidget(wr)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        root = QHBoxLayout(self)
        root.addWidget(split)

        # wire
        self.list.currentRowChanged.connect(lambda _i: self._refresh_steps_list())
        self.list.customContextMenuRequested.connect(self._bundles_context_menu)
        self.btn_new.clicked.connect(self._new_bundle)
        self.btn_dup.clicked.connect(self._dup_bundle)
        self.btn_del.clicked.connect(self._del_bundle)
        # rename shortcuts
        QShortcut(QKeySequence("F2"), self.list, activated=self._rename_bundle)
        self.list.itemDoubleClicked.connect(lambda _it: self._rename_bundle())

        # step context menu
        self.steps.customContextMenuRequested.connect(self._steps_context_menu)

        self.btn_remove.clicked.connect(self._remove_selected_steps)
        self.btn_clear.clicked.connect(self._clear_steps)
        self.btn_up.clicked.connect(lambda: self._move_steps(-1))
        self.btn_down.clicked.connect(lambda: self._move_steps(+1))

        self.btn_drag_bundle.clicked.connect(self._drag_bundle)
        self.btn_run_active.clicked.connect(self._apply_to_active_view)
        self.btn_apply_to_vbundle.clicked.connect(self._apply_to_view_bundle)
        self.btn_chip.clicked.connect(self._compress_to_chip)

        # populate
        self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(0)

        QShortcut(QKeySequence("Delete"), self.steps, activated=self._remove_selected_steps)
        QShortcut(QKeySequence("Backspace"), self.steps, activated=self._remove_selected_steps)
        QShortcut(QKeySequence("Ctrl+A"), self.steps, activated=self.steps.selectAll)

        # chips per bundle index
        self._chips: dict[int, FunctionBundleChip] = {}

    # ---------- small UI pump ----------
    def _pump_events(self, ms: int = 0):
        """Keep UI responsive between long steps."""
        try:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, ms)
        except Exception:
            pass

    # ---------- CC wait helpers ----------
    def _is_cc_running(self, mw) -> bool:
        # main-window flag
        try:
            if getattr(mw, "_cosmicclarity_headless_running", False):
                return True
        except Exception:
            pass
        # QSettings flag
        try:
            v = QSettings().value("cc/headless_in_progress", False, type=bool)
        except Exception:
            v = bool(QSettings().value("cc/headless_in_progress", False))
        return bool(v)

    def _wait_for_cosmicclarity(self, mw, timeout_ms: int = 2 * 60 * 60 * 1000, poll_ms: int = 50):
        """If CC is running, wait here (processing events) until it finishes."""
        if not self._is_cc_running(mw):
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        except Exception:
            pass
        t0 = time.monotonic()
        while self._is_cc_running(mw) and (time.monotonic() - t0) * 1000 < timeout_ms:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
            QThread.msleep(poll_ms)
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass

    # ---------- persistence ----------
    def _load_all(self) -> List[dict]:
        raw = self._settings.value(self.SETTINGS_KEY, "[]", type=str)
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                out = []
                for b in data:
                    if not isinstance(b, dict): continue
                    nm = (b.get("name") or "Function Bundle").strip()
                    steps = b.get("steps") or []
                    if isinstance(steps, list):
                        out.append({"name": nm, "steps": steps})
                return out
        except Exception:
            pass
        return []

    def _save_all(self):
        try:
            self._settings.setValue(self.SETTINGS_KEY, json.dumps(self._bundles, ensure_ascii=False))
            self._settings.sync()  # <- add this line
        except Exception:
            pass

    # ---------- bundle helpers ----------
    def _current_index(self) -> int:
        i = self.list.currentRow()
        return -1 if i < 0 or i >= len(self._bundles) else i

    def _current_bundle(self) -> dict | None:
        i = self._current_index()
        return None if i < 0 else self._bundles[i]

    def current_steps(self) -> list:
        b = self._current_bundle()
        return [] if not b else list(b.get("steps", []))

    def step_count(self) -> int:
        return len(self.current_steps())

    # ---------- list refresh ----------
    def _refresh_bundle_list(self):
        self.list.clear()
        for b in self._bundles:
            self.list.addItem(QListWidgetItem(b.get("name", "Function Bundle")))

    def _refresh_steps_list(self):
        self.steps.clear()
        for st in self.current_steps():
            self._add_step_item(st)

    def _preset_label(self, preset) -> str:
        """Human-friendly label for the preset shown in the list."""
        if preset is None:
            return ""
        if isinstance(preset, str):
            return f" — {preset}"
        if isinstance(preset, dict):
            # Prefer a human name if present
            name = preset.get("name") or preset.get("label")
            if isinstance(name, str) and name.strip():
                return f" — {name.strip()}"
            # Otherwise a tiny summary like {k1,k2}
            keys = list(preset.keys())
            return f" — {{{', '.join(keys[:3])}{'…' if len(keys)>3 else ''}}}"
        # fallback
        return f" — {str(preset)}"

    def _add_step_item(self, step: dict, at: int | None = None):
        cid = step.get("command_id", "<cmd>")
        preset = step.get("preset", None)
        desc = f"{cid}{self._preset_label(preset)}"
        it = QListWidgetItem(desc)
        it.setData(Qt.ItemDataRole.UserRole, step)
        if at is None:
            self.steps.addItem(it)
        else:
            self.steps.insertItem(at, it)

    def _collect_steps_from_ui(self) -> list:
        out = []
        for i in range(self.steps.count()):
            s = self.steps.item(i).data(Qt.ItemDataRole.UserRole)
            if isinstance(s, dict): out.append(s)
        return out

    def _commit_steps_from_ui(self):
        i = self._current_index()
        if i < 0: return
        self._bundles[i]["steps"] = self._collect_steps_from_ui()
        self._save_all()
        if i in self._chips:
            self._chips[i]._sync_count()
        # refresh visible labels (e.g. after preset edits)
        self._refresh_steps_list()

    # ---------- editing actions ----------
    def _new_bundle(self):
        self._bundles.append({"name": f"Function Bundle {len(self._bundles)+1}", "steps": []})
        self._save_all(); self._refresh_bundle_list()
        self.list.setCurrentRow(self.list.count() - 1)

    def _dup_bundle(self):
        i = self._current_index()
        if i < 0: return
        b = self._bundles[i]
        cp = {"name": f"{b.get('name','Function Bundle')} (copy)", "steps": list(b.get("steps", []))}
        self._bundles.insert(i + 1, cp)
        self._save_all(); self._refresh_bundle_list()
        self.list.setCurrentRow(i + 1)

    def _del_bundle(self):
        i = self._current_index()
        if i < 0: return
        # close any chip for that index
        ch = self._chips.pop(i, None)
        if ch:
            try: ch.setParent(None); ch.deleteLater()
            except Exception: pass
        del self._bundles[i]
        self._save_all(); self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(min(i, self.list.count() - 1))

    def _remove_selected_steps(self):
        rows = sorted({ix.row() for ix in self.steps.selectedIndexes()}, reverse=True)
        for r in rows:
            self.steps.takeItem(r)
        self._commit_steps_from_ui()

    def _clear_steps(self):
        self.steps.clear()
        self._commit_steps_from_ui()

    def _move_steps(self, delta: int):
        if not self.steps.selectedItems():
            return
        items = self.steps.selectedItems()
        rows  = sorted([self.steps.row(it) for it in items])
        for idx in (rows if delta < 0 else reversed(rows)):
            it = self.steps.takeItem(idx)
            new_idx = max(0, min(self.steps.count(), idx + delta))
            self.steps.insertItem(new_idx, it)
            it.setSelected(True)
        self._commit_steps_from_ui()

    def _append_steps(self, steps: Iterable[dict]):
        for st in steps:
            if isinstance(st, dict) and st.get("command_id"):
                self._add_step_item(st)
        self._commit_steps_from_ui()

    # ---------- rename bundle ----------
    def _rename_bundle(self):
        i = self._current_index()
        if i < 0:
            return
        cur = self._bundles[i]
        new_name, ok = QInputDialog.getText(self, "Rename Function Bundle",
                                            "New name:", text=cur.get("name","Function Bundle"))
        if not ok:
            return
        cur["name"] = (new_name or "Function Bundle").strip()
        self._save_all()
        self._refresh_bundle_list()
        self.list.setCurrentRow(i)
        # update chip title if present
        ch = self._chips.get(i)
        if ch:
            ch._title.setText(cur["name"])

    def _bundles_context_menu(self, pos):
        if self.list.count() == 0:
            return
        m = QMenu(self)
        act_ren = m.addAction("Rename…")
        act = m.exec(self.list.mapToGlobal(pos))
        if act is act_ren:
            self._rename_bundle()

    # ---------- step context menu & preset editor ----------
    def _steps_context_menu(self, pos):
        item = self.steps.itemAt(pos)
        if not item:
            return
        m = QMenu(self)
        a_edit  = m.addAction("Edit Preset…")
        a_clear = m.addAction("Clear Preset")
        m.addSeparator()
        a_dup   = m.addAction("Duplicate Step")
        a_rem   = m.addAction("Remove Step")
        act = m.exec(self.steps.mapToGlobal(pos))
        if not act:
            return
        row = self.steps.row(item)
        step = item.data(Qt.ItemDataRole.UserRole) or {}
        if act is a_edit:
            new_preset, ok = self._edit_preset_dialog(step.get("preset", None))
            if ok:
                step["preset"] = new_preset
                item.setData(Qt.ItemDataRole.UserRole, step)
                item.setText(f"{step.get('command_id','<cmd>')}{self._preset_label(new_preset)}")
                self._commit_steps_from_ui()
        elif act is a_clear:
            if "preset" in step:
                step.pop("preset", None)
                item.setData(Qt.ItemDataRole.UserRole, step)
                item.setText(f"{step.get('command_id','<cmd>')}")
                self._commit_steps_from_ui()
        elif act is a_dup:
            self._add_step_item(json.loads(json.dumps(step)), at=row+1)
            self._commit_steps_from_ui()
        elif act is a_rem:
            self.steps.takeItem(row)
            self._commit_steps_from_ui()

    def _edit_preset_dialog(self, current) -> tuple[object, bool]:
        """
        Simple JSON editor for the 'preset' field.
        Accepts dict, string, number, etc. Returns (value, ok).
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Preset")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Edit the preset as JSON (e.g. {\"name\":\"My Preset\", \"strength\": 0.8})"))
        edit = QPlainTextEdit()
        try:
            seed = json.dumps(current, ensure_ascii=False, indent=2)
        except Exception:
            seed = json.dumps(current if current is not None else {}, ensure_ascii=False, indent=2)
        edit.setPlainText(seed)
        v.addWidget(edit, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return current, False
        txt = edit.toPlainText().strip()
        if not txt:
            return None, True
        try:
            val = json.loads(txt)
        except Exception as e:
            QMessageBox.warning(self, "Invalid JSON", f"Could not parse JSON:\n{e}")
            return current, False
        return val, True

    # ---------- DnD into the PANEL (add steps) ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_CMD):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasFormat(MIME_CMD):
            payload = _unpack_cmd_safely(bytes(md.data(MIME_CMD)))
            if not isinstance(payload, dict) or not payload.get("command_id"):
                e.ignore(); return
            if payload.get("command_id") == "function_bundle":
                steps = payload.get("steps") or []
                self._append_steps(steps)
            else:
                self._append_steps([payload])
            e.acceptProposedAction(); return
        e.ignore()

    # ---------- run / export ----------
    def _drag_bundle(self):
        payload = {"command_id": "function_bundle", "steps": self.current_steps()}
        md = QMimeData()
        md.setData(MIME_CMD, QByteArray(_pack_cmd_safely(payload)))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.setHotSpot(self.rect().center())
        drag.exec(Qt.DropAction.CopyAction)

    def _apply_to_active_view(self):
        mw = _find_main_window(self)
        if not mw or not hasattr(mw, "_handle_command_drop"):
            QMessageBox.information(self, "Apply", "Main window not available.")
            return
        sw = mw.mdi.activeSubWindow() if hasattr(mw, "mdi") else None
        if not sw:
            QMessageBox.information(self, "Apply", "No active view.")
            return
        self._apply_steps_to_target_sw(mw, sw, self.current_steps())

    def _apply_to_view_bundle(self):
        mw = _find_main_window(self)
        if not mw:
            QMessageBox.information(self, "Apply", "Main window not available.")
            return

        settings = QSettings()
        settings.sync()  # see latest saved bundles

        raw_v2 = settings.value("viewbundles/v2", "", type=str)
        raw_v1 = settings.value("viewbundles/v1", "", type=str)
        raw = raw_v2 or raw_v1 or "[]"

        try:
            vb_raw = json.loads(raw)
        except Exception:
            vb_raw = []

        # normalize -> [(name, [int_ptr,...])]
        choices = []
        for b in vb_raw:
            if not isinstance(b, dict):
                continue
            name = (b.get("name") or "Bundle").strip()
            ptrs = []
            for x in (b.get("doc_ptrs") or []):
                try:
                    ptrs.append(int(x))
                except Exception:
                    pass
            choices.append((name, ptrs))

        if not choices:
            QMessageBox.information(self, "Apply", "No View Bundles found.")
            return

        # ✅ create the dialog BEFORE using it
        dlg = QDialog(self)
        dlg.setWindowTitle("Apply to View Bundle…")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Select a View Bundle:"))
        lb = QListWidget(); v.addWidget(lb, 1)
        for name, ptrs in choices:
            it = QListWidgetItem(f"{name}  ({len(ptrs)} views)")
            it.setData(Qt.ItemDataRole.UserRole, ptrs)
            lb.addItem(it)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cur = lb.currentItem()
        if not cur:
            return
        ptrs = cur.data(Qt.ItemDataRole.UserRole) or []
        steps = self.current_steps()
        if not steps:
            QMessageBox.information(self, "Apply", "This Function Bundle is empty.")
            return

        # show busy cursor during batch apply
        try: QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        except Exception: pass

        applied = 0
        for p in ptrs:
            _doc, sw = _resolve_doc_and_subwindow(mw, p)
            if sw is None:
                self._pump_events(0)
                continue
            _activate_target_sw(mw, sw)   
            self._apply_steps_to_target_sw(mw, sw, steps)
            applied += 1
            self._wait_for_cosmicclarity(mw)
            self._pump_events(0)

        try: QApplication.restoreOverrideCursor()
        except Exception: pass

        if applied == 0:
            QMessageBox.information(self, "Apply", "No valid targets in the selected bundle.")


    def _apply_steps_to_target_sw(self, mw, sw, steps: list[dict]):
        errors = []
        # busy cursor while running this set
        try: QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        except Exception: pass

        for st in steps:
            _activate_target_sw(mw, sw)   
            try:
                mw._handle_command_drop(st, target_sw=sw)
            except Exception as e:
                errors.append(str(e))

            # 🔸 If CC is (now) running, wait until it’s done before continuing.
            self._wait_for_cosmicclarity(mw)

            # keep UI responsive regardless
            self._pump_events(0)

        try: QApplication.restoreOverrideCursor()
        except Exception: pass

        if errors:
            QMessageBox.warning(self, "Apply", "Some steps failed:\n\n" + "\n".join(errors))


    def _compress_to_chip(self):
        i = self._current_index()
        if i < 0: return
        name = self._bundles[i].get("name", "Function Bundle")

        mw = _find_main_window(self)
        if not mw:
            QMessageBox.information(self, "Compress", "Main window not available."); return

        chip = self._chips.get(i)
        if chip is None or chip.parent() is None:
            chip = _spawn_function_chip_on_canvas(mw, self, name, bundle_key=f"fn-{i}")
            if chip is None:
                QMessageBox.information(self, "Compress", "Shortcut canvas not available."); return
            self._chips[i] = chip
        else:
            chip._title.setText(name)
            chip._sync_count()
            chip.show()
            chip.raise_()
        # keep the panel visible (matches View Bundle behavior)

    def closeEvent(self, e: QCloseEvent):
        super().closeEvent(e)

# ---------- singleton open helper ----------
_dialog_singleton: FunctionBundleDialog | None = None
def show_function_bundles(parent: QWidget | None):
    global _dialog_singleton
    if _dialog_singleton is None:
        _dialog_singleton = FunctionBundleDialog(parent)
        def _clear():
            global _dialog_singleton
            _dialog_singleton = None
        _dialog_singleton.destroyed.connect(_clear)
    _dialog_singleton.show()
    _dialog_singleton.raise_()
    _dialog_singleton.activateWindow()
    return _dialog_singleton
