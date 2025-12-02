# pro/function_bundle.py
from __future__ import annotations
import json
from typing import Iterable, List, Any, Dict
import sys
from PyQt6.QtCore import Qt, QSettings, QByteArray, QMimeData, QSize, QPoint, QEventLoop
from PyQt6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem, QProgressBar,
    QPushButton, QSplitter, QMessageBox, QLabel, QAbstractItemView, QDialogButtonBox,
    QApplication, QMenu, QInputDialog, QPlainTextEdit, QListView
)
from PyQt6.QtGui import QDrag, QCloseEvent, QCursor, QShortcut, QKeySequence
from PyQt6.QtCore import  QThread
import time
from pro.dnd_mime import MIME_CMD
from ops.commands import normalize_cid
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
        
        self._panel = panel
        self._bundle_key = bundle_key     # <── store bundle key for panel lookups
        self._dragging = False
        self._grab_offset = None

        self.setAcceptDrops(True)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # <── allows Delete key

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
            self.setFocus(Qt.FocusReason.MouseFocusReason)  # <── so Delete works
            self._grab_offset = ev.position()  # QPointF in widget coords
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
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
            # Save layout whenever the user finishes a drag
            try:
                self._panel._save_chip_layout()
            except Exception:
                pass
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        try:
            self._panel.showNormal()
            self._panel.raise_()
            self._panel.activateWindow()
        except Exception:
            pass
        ev.accept()

    def contextMenuEvent(self, ev):
        from PyQt6.QtWidgets import QMenu  # already imported at top, but safe

        m = QMenu(self)
        act_del = m.addAction("Delete Chip")
        act = m.exec(ev.globalPos())
        if act is act_del:
            try:
                self._panel._remove_chip_widget(self)
            except Exception:
                pass
        else:
            ev.ignore()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            try:
                self._panel._remove_chip_widget(self)
            except Exception:
                pass
            ev.accept()
            return
        super().keyPressEvent(ev)


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
        from PyQt6.QtWidgets import QApplication

        print(f"[FBChip] _start_external_drag: bundle_key={self._bundle_key}, name={self._title.text()!r}", flush=True)
        QApplication.processEvents()

        payload = {
            "command_id": "function_bundle",
            "steps": self._panel.current_steps(),
            "inherit_target": True,
        }
        print(f"[FBChip]   payload steps={len(payload['steps'])}", flush=True)
        QApplication.processEvents()

        md = QMimeData()
        md.setData(MIME_CMD, QByteArray(_pack_cmd_safely(payload)))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.setHotSpot(QPoint(self.width() // 2, self.height() // 2))

        print("[FBChip]   starting drag.exec(...)", flush=True)
        QApplication.processEvents()
        drag.exec(Qt.DropAction.CopyAction)
        print("[FBChip]   drag.exec finished", flush=True)
        QApplication.processEvents()


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
    CHIP_KEY     = "functionbundles/chips_v1"   # <── new

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

        # ✅ make long step text readable
        self.steps.setWordWrap(True)  # wrap long lines
        self.steps.setTextElideMode(Qt.TextElideMode.ElideRight)  # if still too long, show …
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.steps.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.steps.setResizeMode(QListView.ResizeMode.Adjust)  # recompute item layout on width change
        self.steps.setUniformItemSizes(False)

        self.add_hint = QLabel("Drop shortcuts here to add steps")
        self.add_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_hint.setStyleSheet("color:#aaa; padding:6px; border:1px dashed #666; border-radius:6px;")

        self.btn_edit_preset = QPushButton("Edit Preset…")
        self.btn_edit_preset.setEnabled(False)  # enabled when exactly one step is selected

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

        # controls row under the Steps list
        rrow = QHBoxLayout()
        rrow.addWidget(self.btn_up)
        rrow.addWidget(self.btn_down)

        # center Edit Preset between Move Down and Remove Selected
        rrow.addStretch(1)
        rrow.addWidget(self.btn_edit_preset)
        rrow.addStretch(1)

        # then Remove/Clear on the right
        rrow.addWidget(self.btn_remove)
        rrow.addWidget(self.btn_clear)

        right.addLayout(rrow)

        self.run_status = QLabel("Ready.")
        self.run_status.setStyleSheet("color:#aaa; padding:2px 0;")
        self.run_progress = QProgressBar()
        self.run_progress.setMinimum(0)
        self.run_progress.setMaximum(100)
        self.run_progress.setValue(0)
        self.run_progress.setTextVisible(True)

        prow = QHBoxLayout()
        prow.addWidget(self.run_status, 1)
        prow.addWidget(self.run_progress, 2)
        right.addLayout(prow)

        # right.addWidget(self.btn_drag_bundle)
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

        self.steps.itemSelectionChanged.connect(self._sync_edit_button_enabled)
        self.btn_edit_preset.clicked.connect(self._edit_selected_step_preset)
        QShortcut(QKeySequence("Return"), self.steps, activated=self._edit_selected_step_preset)   # handy
        QShortcut(QKeySequence("Enter"),  self.steps, activated=self._edit_selected_step_preset)

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

        # Restore any chips that were saved in QSettings
        try:
            self._restore_chips_from_settings()
        except Exception:
            pass

    def _save_chip_layout(self):
        """
        Persist current chips and their positions to QSettings so they
        reappear on the canvas next time SASpro is opened.
        """
        try:
            data = []
            for idx, chip in list(self._chips.items()):
                if chip is None or chip.parent() is None:
                    continue
                pos = chip.pos()
                data.append({
                    "index": int(idx),
                    "x": int(pos.x()),
                    "y": int(pos.y()),
                })
            self._settings.setValue(self.CHIP_KEY, json.dumps(data, ensure_ascii=False))
            self._settings.sync()
        except Exception:
            pass

    def _restore_chips_from_settings(self):
        """
        Recreate chips on the ShortcutCanvas from saved layout.
        Called on dialog init.
        """
        mw = _find_main_window(self)
        if not mw:
            return

        raw = self._settings.value(self.CHIP_KEY, "[]", type=str)
        try:
            data = json.loads(raw)
        except Exception:
            data = []

        if not isinstance(data, list):
            return

        for entry in data:
            try:
                idx = int(entry.get("index", -1))
            except Exception:
                continue
            if idx < 0 or idx >= len(self._bundles):
                continue

            name = self._bundles[idx].get("name", "Function Bundle")
            chip = _spawn_function_chip_on_canvas(mw, self, name, bundle_key=f"fn-{idx}")
            if chip is None:
                continue

            # Restore position if provided
            x = entry.get("x")
            y = entry.get("y")
            if isinstance(x, int) and isinstance(y, int):
                chip.move(x, y)

            self._chips[idx] = chip

    def reload_from_settings_after_import(self):
        """
        Reload bundles + chips from QSettings after an external import
        (e.g., shortcuts .sass import).
        """
        try:
            self._bundles = self._load_all()
        except Exception:
            self._bundles = []
        self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(0)

        # Remove existing chips from canvas
        for ch in list(self._chips.values()):
            try:
                ch.setParent(None)
                ch.deleteLater()
            except Exception:
                pass
        self._chips.clear()

        # And recreate them from CHIP_KEY
        try:
            self._restore_chips_from_settings()
        except Exception:
            pass


    def _remove_chip_widget(self, chip: FunctionBundleChip):
        """
        Remove a chip from the canvas and from our registry, without
        deleting the underlying function bundle.
        """
        # Drop from the index → chip dict
        for idx, ch in list(self._chips.items()):
            if ch is chip:
                self._chips.pop(idx, None)
                break

        try:
            chip.setParent(None)
            chip.deleteLater()
        except Exception:
            pass

        self._save_chip_layout()


    def _progress_reset(self):
        try:
            self.run_status.setText("Ready.")
            self.run_progress.setRange(0, 100)
            self.run_progress.setValue(0)
            QApplication.processEvents()
        except Exception:
            pass

    def _progress_set_step(self, idx: int, total: int, label: str):
        """Determinate update for normal steps."""
        try:
            idx = max(0, idx)
            total = max(1, total)
            pct = int(100 * idx / total)
            self.run_status.setText(f"Running step {idx}/{total}: {label}")
            self.run_progress.setRange(0, 100)
            self.run_progress.setValue(pct)
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        except Exception:
            pass

    def _progress_busy(self, label: str):
        """Indeterminate mode for long-running sub-steps (e.g., Cosmic Clarity)."""
        try:
            self.run_status.setText(label)
            self.run_progress.setRange(0, 0)  # indeterminate
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        except Exception:
            pass

    def _step_label(self, st: dict) -> str:
        cid = (st or {}).get("command_id", "<cmd>")
        # If preset has a friendly name/label, include it
        preset = (st or {}).get("preset")
        if isinstance(preset, dict):
            name = preset.get("name") or preset.get("label")
            if isinstance(name, str) and name.strip():
                return f"{cid} — {name.strip()}"
        return str(cid)


    def _sync_edit_button_enabled(self):
        self.btn_edit_preset.setEnabled(len(self.steps.selectedItems()) == 1)

    def _edit_selected_step_preset(self):
        items = self.steps.selectedItems()
        if len(items) != 1:
            return
        it = items[0]
        step = it.data(Qt.ItemDataRole.UserRole) or {}
        new_preset, ok = self._edit_preset_dialog(step.get("preset", None), step)
        if ok:
            step["preset"] = new_preset
            it.setData(Qt.ItemDataRole.UserRole, step)
            it.setText(f"{step.get('command_id','<cmd>')}{self._preset_label(new_preset)}")
            self._commit_steps_from_ui()


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
        it.setToolTip(desc)  # ✅ hover shows full line
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
            try:
                ch.setParent(None)
                ch.deleteLater()
            except Exception:
                pass

        del self._bundles[i]
        self._save_all()
        self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(min(i, self.list.count() - 1))

        # Also update chip layout persistence
        try:
            self._save_chip_layout()
        except Exception:
            pass

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
            new_preset, ok = self._edit_preset_dialog(step.get("preset", None), step)
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

    def _edit_preset_dialog(self, current, step: dict | None = None) -> tuple[object, bool]:
        """
        Prefer the same rich UI editors used by desktop shortcuts.
        - If a bespoke editor exists and user cancels => do NOT open JSON.
        - If no bespoke editor exists => fall back to JSON editor.
        Returns (value, ok).
        """
        # Try to open a command-specific UI if we know the command_id for the selected step
        cmd = None
        if isinstance(step, dict):
            cmd = step.get("command_id")

        try:
            from pro.shortcuts import _open_preset_editor_for_command, _has_preset_editor_for_command
        except Exception:
            _open_preset_editor_for_command = None
            _has_preset_editor_for_command = lambda _c: False  # type: ignore

        # If we have a bespoke UI for this command, use it; cancel means "do nothing"
        if cmd and _has_preset_editor_for_command(cmd) and _open_preset_editor_for_command:
            result = _open_preset_editor_for_command(self, cmd, current if isinstance(current, dict) else {})
            if result is None:
                return current, False   # user cancelled rich UI → don't open JSON
            return result, True         # accepted via rich UI

        # ---- Fallback: JSON editor (only when no bespoke editor exists) ----
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Preset")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Edit the preset as JSON (e.g. {\"name\":\"My Preset\", \"strength\": 0.8})"))
        edit = QPlainTextEdit()
        edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
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
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

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
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        if applied == 0:
            QMessageBox.information(self, "Apply", "No valid targets in the selected bundle.")


    def _apply_steps_to_target_sw(self, mw, sw, steps: list[dict]):
        # local logger
        def _fb(msg: str):
            m = f"[FunctionBundleDialog] {msg}"
            try:
                # main window logger if present
                if hasattr(mw, "_log"):
                    mw._log(m)
            except Exception:
                pass
            try:
                print(m, flush=True)
            except Exception:
                pass

        _fb(f"ENTER _apply_steps_to_target_sw: sw={repr(sw)}, steps={len(steps)}")

        errors = []
        total = len(steps)

        # busy cursor while running this set
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        except Exception:
            pass

        # start fresh
        self._progress_reset()

        for i, st in enumerate(steps, start=1):
            _activate_target_sw(mw, sw)

            label = self._step_label(st)
            self._progress_set_step(i - 1, total, label)

            if not isinstance(st, dict) or not st.get("command_id"):
                _fb(f"  skip step[{i}]: invalid payload={repr(st)}")
                continue

            cid = st.get("command_id")
            if str(cid).lower().startswith("cosmic"):
                _fb(f"  >>> BEGIN CC step[{i}/{total}] cid={cid} payload={repr(st)}")
            else:
                _fb(f"  BEGIN step[{i}/{total}] cid={cid} payload={repr(st)}")

            try:
                mw._handle_command_drop(st, target_sw=sw)

                if str(cid).lower().startswith("cosmic"):
                    _fb(f"  <<< END   CC step[{i}/{total}] cid={cid} OK")
                else:
                    _fb(f"  END   step[{i}/{total}] cid={cid} OK")

            except Exception as e:
                errors.append(str(e))
                if str(cid).lower().startswith("cosmic"):
                    _fb(f"  <<< END   CC step[{i}/{total}] cid={cid} ERROR: {e!r}")
                else:
                    _fb(f"  END   step[{i}/{total}] cid={cid} ERROR: {e!r}")

            self._progress_set_step(i, total, label)
            self._pump_events(0)

        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass

        self.run_status.setText("Done.")
        self.run_progress.setRange(0, 100)
        self.run_progress.setValue(100)

        if errors:
            _fb(f"EXIT with errors: {errors}")
            QMessageBox.warning(
                self,
                "Apply",
                "Some steps failed:\n\n" + "\n".join(errors),
            )
        else:
            _fb("EXIT OK (no errors)")


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
        try:
            self._save_chip_layout()   # <── persist chip presence/pos
        except Exception:
            pass

    def closeEvent(self, e: QCloseEvent):
        super().closeEvent(e)

# ---------- script / command entry point ----------


class FunctionBundleManager:
    """
    Simple QSettings-backed store for Function Bundles.

    This MUST use the same "functionbundles/v1" key that the dialog uses,
    so scripts and UI always see the same bundles.
    """
    SETTINGS_KEY = "functionbundles/v1"

    def __init__(self, app=None):
        # app is unused for now but kept for future (e.g. per-profile settings).
        self._settings = QSettings()

    # ---- low-level ----
    def _load_all(self) -> list[dict]:
        raw = self._settings.value(self.SETTINGS_KEY, "[]", type=str)
        try:
            bundles = json.loads(raw)
        except Exception:
            bundles = []

        if not isinstance(bundles, list):
            return []
        return [b for b in bundles if isinstance(b, dict)]

    # ---- public API ----
    def list_bundles(self) -> list[dict]:
        return self._load_all()

    def get_bundle(self, name: str) -> dict | None:
        if not name:
            return None
        want = name.strip().lower()
        for b in self._load_all():
            n = (b.get("name") or "").strip().lower()
            if n == want:
                return b
        return None


# Optional: cache a single instance per process
_bundle_mgr: FunctionBundleManager | None = None

def get_bundle_manager(app=None) -> FunctionBundleManager:
    """
    Return a process-wide FunctionBundleManager.

    Keeping a single instance avoids re-parsing JSON constantly,
    but still reads from QSettings each time you call list/get.
    """
    global _bundle_mgr
    if _bundle_mgr is None:
        _bundle_mgr = FunctionBundleManager(app)
    return _bundle_mgr

# ---------- script / command entry point ----------
def _normalize_steps_for_hcd(steps: list[Any]) -> list[Dict[str, Any]]:
    """
    Take whatever is stored in the bundle and normalize it into the
    drop-payload shape that MainWindow._handle_command_drop expects:

        {
            "command_id": "<cid>",
            "preset": { ...optional... },
            "on_base": bool,
            ... (other keys passed through as-is)
        }

    This keeps old bundles (with 'id' or 'cid' fields) working.
    """
    out: list[Dict[str, Any]] = []

    for st in steps or []:
        if not isinstance(st, dict):
            continue

        cid = (
            st.get("command_id")
            or st.get("cid")
            or st.get("id")
        )
        if not cid:
            continue

        payload: Dict[str, Any] = {
            "command_id": cid,
        }

        # Preserve preset if present
        if "preset" in st:
            payload["preset"] = st["preset"]

        # Preserve on_base if present
        if "on_base" in st:
            payload["on_base"] = bool(st.get("on_base"))

        # Keep label / description for logging / UI if you want
        if "label" in st:
            payload["label"] = st["label"]

        # Pass through any extra keys you want HCD to see
        for k, v in st.items():
            if k in payload:
                continue
            if k in ("command_id", "cid", "id"):
                continue
            payload[k] = v

        out.append(payload)

    return out



def run_function_bundle_command(ctx, preset: dict | None = None):
    """
    Entry point for CommandSpec(id="function_bundle").

    IMPORTANT:
    This is meant to behave EXACTLY like dropping a Function Bundle
    on a view in the UI. That means we DO NOT iterate steps via
    ctx.run_command; instead we synthesize a single payload with
    command_id='function_bundle' and let MainWindow._handle_command_drop
    do all the work.
    """
    preset = dict(preset or {})

    app = getattr(ctx, "app", None) or getattr(ctx, "main_window", lambda: None)()
    if app is None:
        raise RuntimeError("Function Bundle command requires a GUI main window / ctx.app")

    # --- resolve steps: saved bundle OR inline ---
    bundle_name = preset.get("bundle_name") or preset.get("name")
    steps: list[dict[str, Any]] = list(preset.get("steps") or [])
    inherit = bool(preset.get("inherit_target", True))

    # optional: targets='all_open' or [doc_ptrs], same as HCD branch supports
    targets = preset.get("targets", None)

    if bundle_name and not steps:
        # Use the same bundle store as the Function Bundles dialog
        mgr = get_bundle_manager(app)
        data = mgr.get_bundle(bundle_name)
        if not data:
            raise RuntimeError(f"Function Bundle '{bundle_name}' not found.")
        steps = list(data.get("steps") or [])

    steps = _normalize_steps_for_hcd(steps)

    if not steps:
        try:
            ctx.log("Function Bundle: no steps to run.")
        except Exception:
            pass
        return

    # --- build the same payload the UI uses for a bundle drop ---
    payload: Dict[str, Any] = {
        "command_id": "function_bundle",
        "steps": steps,
        "inherit_target": inherit,
    }
    if targets is not None:
        payload["targets"] = targets

    # If targets were specified, we mimic dropping on the background:
    #   _handle_command_drop(payload, target_sw=None)
    # so the HCD branch fans out to all_open / explicit ptr list.
    if targets is not None:
        target_sw = None
    else:
        # "Normal" script usage: run on the active view, exactly like
        # dragging the bundle chip onto that view.
        try:
            target_sw = ctx.active_subwindow()
        except Exception:
            target_sw = None

    if target_sw is None and targets is None:
        # No active view and no explicit targets – nothing to do.
        raise RuntimeError("Function Bundle: no active view and no explicit targets.")

    # --- delegate to main-window drop handler (single point of truth) ---
    print(
        f"[FunctionBundle] Script call → _handle_command_drop() "
        f"inherit_target={inherit}, targets={targets!r}, steps={len(steps)}",
        flush=True,
    )
    QApplication.processEvents()
    app._handle_command_drop(payload, target_sw=target_sw)
    QApplication.processEvents()


# ---------- singleton open helper ----------
_dialog_singleton: FunctionBundleDialog | None = None
def show_function_bundles(parent: QWidget | None,
                          focus_name: str | None = None,
                          *,
                          auto_spawn_only: bool = False):
    """
    Open (or focus) the Function Bundles dialog.

    If auto_spawn_only=True, ensure the dialog + chips exist,
    but do NOT show the dialog (for startup chip restore).
    """
    global _dialog_singleton
    if _dialog_singleton is None:
        _dialog_singleton = FunctionBundleDialog(parent)
        def _clear():
            global _dialog_singleton
            _dialog_singleton = None
        _dialog_singleton.destroyed.connect(_clear)

    if focus_name:
        ...

    if not auto_spawn_only:
        _dialog_singleton.show()
        _dialog_singleton.raise_()
        _dialog_singleton.activateWindow()
    return _dialog_singleton

def restore_function_bundle_chips(parent: QWidget | None):
    """
    Called at app startup: create the FunctionBundleDialog singleton,
    restore any saved chips onto the ShortcutCanvas, but keep the
    dialog itself hidden.
    """
    try:
        show_function_bundles(parent, auto_spawn_only=True)
    except Exception:
        pass

def export_function_bundles_payload() -> dict:
    """
    Export function bundle definitions + chip layout so they can be embedded
    into a shortcuts .sass file. This works even if the dialog isn't open.
    """
    s = QSettings()
    raw_bundles = s.value(FunctionBundleDialog.SETTINGS_KEY, "[]", type=str)
    raw_chips   = s.value(FunctionBundleDialog.CHIP_KEY, "[]", type=str)

    try:
        bundles = json.loads(raw_bundles)
    except Exception:
        bundles = []
    try:
        chips = json.loads(raw_chips)
    except Exception:
        chips = []

    if not isinstance(bundles, list):
        bundles = []
    if not isinstance(chips, list):
        chips = []

    # `bundles` contains full guts: name + steps (+ presets)
    # `chips` contains chip positions keyed by bundle index
    return {
        "bundles": bundles,
        "chips": chips,
    }

def import_function_bundles_payload(payload: dict, parent: QWidget | None, replace_existing: bool = False):
    """
    Apply imported bundle+chip payload from a .sass file.

    - If replace_existing=True, overwrite existing bundles/chips.
    - If False, append to existing bundles and offset chip indices accordingly.
    """
    if not isinstance(payload, dict):
        return

    new_bundles = payload.get("bundles") or []
    new_chips   = payload.get("chips") or []

    if not isinstance(new_bundles, list):
        new_bundles = []
    if not isinstance(new_chips, list):
        new_chips = []

    s = QSettings()

    if replace_existing:
        bundles = new_bundles
        chips   = new_chips
    else:
        raw_b = s.value(FunctionBundleDialog.SETTINGS_KEY, "[]", type=str)
        raw_c = s.value(FunctionBundleDialog.CHIP_KEY, "[]", type=str)
        try:
            old_bundles = json.loads(raw_b)
        except Exception:
            old_bundles = []
        try:
            old_chips = json.loads(raw_c)
        except Exception:
            old_chips = []

        if not isinstance(old_bundles, list):
            old_bundles = []
        if not isinstance(old_chips, list):
            old_chips = []

        offset = len(old_bundles)
        bundles = old_bundles + new_bundles

        chips = list(old_chips)
        for entry in new_chips:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("index", -1))
            except Exception:
                continue
            if idx < 0:
                continue
            chips.append({
                "index": offset + idx,
                "x": entry.get("x"),
                "y": entry.get("y"),
            })

    try:
        s.setValue(FunctionBundleDialog.SETTINGS_KEY, json.dumps(bundles, ensure_ascii=False))
        s.setValue(FunctionBundleDialog.CHIP_KEY,     json.dumps(chips,    ensure_ascii=False))
        s.sync()
    except Exception:
        pass

    # Refresh any live dialog or, if none, spawn chips from settings
    from typing import cast
    global _dialog_singleton
    if _dialog_singleton is not None:
        try:
            cast(FunctionBundleDialog, _dialog_singleton).reload_from_settings_after_import()
        except Exception:
            pass
    else:
        try:
            restore_function_bundle_chips(parent)
        except Exception:
            pass
