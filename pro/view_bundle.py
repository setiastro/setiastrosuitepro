# pro/view_bundle.py
from __future__ import annotations
import json
import uuid
from typing import Iterable, Optional

from PyQt6.QtCore import Qt, QSettings, QByteArray, QMimeData, QSize, QPoint, QEventLoop
from PyQt6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,QApplication,
    QPushButton, QSplitter, QMessageBox, QLabel, QAbstractItemView, QDialogButtonBox,
    QCheckBox, QFrame, QSizePolicy, QMenu, QInputDialog
)
from PyQt6.QtGui import QDrag, QCloseEvent, QCursor, QShortcut, QKeySequence

from pro.dnd_mime import MIME_CMD, MIME_VIEWSTATE


# ---------- helpers ----------
def _find_main_window(w: QWidget):
    p = w.parent()
    # the main window has either .doc_manager or .docman
    while p is not None and not (hasattr(p, "doc_manager") or hasattr(p, "docman")):
        p = p.parent()
    return p


def _resolve_doc_and_subwindow(mw, doc_ptr):
    """
    Resolve a (doc, sw) pair given the id(ptr) of the document.
    Prefers the main-window helper if available; otherwise, scans open subwindows.
    """
    if hasattr(mw, "_find_doc_by_id"):
        doc, sw = mw._find_doc_by_id(doc_ptr)
        if doc is not None:
            return doc, sw

    # fallback: scan MDI
    try:
        for sw in mw.mdi.subWindowList():
            vw = sw.widget()
            d = getattr(vw, "document", None)
            if d is not None and id(d) == int(doc_ptr):
                return d, sw
    except Exception:
        pass
    return None, None


def _unpack_cmd_safely(raw: bytes):
    """
    Lazy-import the real unpacker to avoid circular imports.
    Fallback to JSON if needed.
    """
    try:
        from pro.shortcuts import _unpack_cmd_payload as _unpack
    except Exception:
        _unpack = None

    if _unpack is not None:
        try:
            return _unpack(raw)
        except Exception:
            pass
    # Fallback: assume JSON
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _pack_cmd_safely(payload: dict) -> bytes:
    """
    Lazy-import the real packer if available, otherwise JSON-encode.
    """
    try:
        from pro.shortcuts import _pack_cmd_payload as _PACK
    except Exception:
        _PACK = None

    if _PACK:
        data = _PACK(payload)
        return bytes(data) if not isinstance(data, (bytes, bytearray)) else data
    return json.dumps(payload).encode("utf-8")


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

def _unwrap_cmd_payload(p: dict) -> dict:
    """
    Some packers wrap as {'command_id': {actual_cmd_dict}, 'preset': {...}}.
    If we see that shape, return the inner dict.
    """
    if isinstance(p, dict):
        cmd = p.get("command_id")
        if isinstance(cmd, dict) and cmd.get("command_id"):
            return dict(cmd)  # copy to avoid aliasing
    return p

# ----------------------------- Bundle Chip -----------------------------
class BundleChip(QWidget):
    """
    A movable chip displayed on the ShortcutCanvas.

    Behaviors:
      - Left-drag: move inside the canvas
      - Ctrl+drag: start external DnD with MIME_CMD payload (command_id="bundle")
      - Drop a view (MIME_VIEWSTATE): add that view to this bundle
      - Drop a shortcut (MIME_CMD): apply that shortcut to all views in the bundle
      - Double-click: re-open the View Bundle dialog (event is accepted so it won't propagate)

    Each chip is bound to ONE bundle via a persistent UUID.
    """
    def __init__(self, panel: "ViewBundleDialog", bundle_uuid: str, name: str,
                 steps: list | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._panel = panel
        self._bundle_uuid = bundle_uuid
        self._name = name
        self._steps = steps or []   # optional future use (not required now)
        self.setAcceptDrops(True)

        self.setObjectName("BundleChip")
        self.setMinimumSize(160, 38)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet("""
            QWidget#BundleChip {
                background: rgba(60, 60, 70, 200);
                border: 1px solid rgba(220,220,220,64);
                border-radius: 8px;
            }
            QLabel#chipTitle {
                padding: 6px 10px 2px 10px;
                color: #e6e6e6;
                font-weight: 600;
            }
            QLabel#chipHint {
                padding: 0 10px 6px 10px;
                color: #bdbdbd;
                font-size: 11px;
            }
            QWidget#BundleChip:hover {
                border-color: rgba(255,255,255,128);
            }
        """)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._title = QLabel(self._name)
        self._title.setObjectName("chipTitle")
        self._hint = QLabel("Drag to move · Ctrl+drag to apply · Drop views/shortcuts here")
        self._hint.setObjectName("chipHint")
        v.addWidget(self._title, 0, Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._hint, 0, Qt.AlignmentFlag.AlignCenter)

        self._press_pos: QPoint | None = None
        self._moving = False

    # --- data binding ---
    @property
    def bundle_uuid(self) -> str:
        return self._bundle_uuid

    def sync_from_panel(self):
        b = self._panel._get_bundle(self._bundle_uuid)
        if b:
            self._name = b.get("name", "Bundle")
            self._title.setText(self._name)

    # --- movement inside canvas / external DnD ---
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            # store where in the chip the user grabbed (widget-local)
            self._grab_offset = ev.position()     # QPointF
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()   # stop propagation to canvas
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if not (ev.buttons() & Qt.MouseButton.LeftButton) or not getattr(self, "_dragging", False):
            super().mouseMoveEvent(ev)
            return

        # Ctrl held → start external DnD once, not repeatedly
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._start_external_drag()
            ev.accept()
            return

        # Anchor the chip to the cursor using GLOBAL coordinates
        parent = self.parentWidget()
        if not parent:
            return

        # where the cursor is globally, minus where we grabbed inside the chip
        global_top_left = ev.globalPosition() - getattr(self, "_grab_offset", ev.position())
        # convert that to the parent’s coordinate system
        top_left = parent.mapFromGlobal(global_top_left.toPoint())

        # clamp inside parent’s rect
        max_x = max(0, parent.width()  - self.width())
        max_y = max(0, parent.height() - self.height())
        x = min(max(0, top_left.x()), max_x)
        y = min(max(0, top_left.y()), max_y)

        self.move(x, y)
        ev.accept()   # don’t let the canvas also handle this drag

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        # reopen the panel and STOP propagation so canvas double-click doesn't fire
        try:
            self._panel.showNormal()
            self._panel.raise_()
            self._panel.activateWindow()
        except Exception:
            pass
        ev.accept()

    def _start_external_drag(self):
        # unchanged from your current version
        payload = {"command_id": "bundle", "steps": self._steps, "bundle_uuid": self._bundle_uuid}
        md = QMimeData()
        md.setData(MIME_CMD, QByteArray(_pack_cmd_safely(payload)))
        drag = QDrag(self)
        drag.setMimeData(md)
        drag.setHotSpot(QPoint(self.width() // 2, self.height() // 2))
        drag.exec(Qt.DropAction.CopyAction)

    # --- accept drops onto the chip ---
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_VIEWSTATE) or e.mimeData().hasFormat(MIME_CMD):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        md = e.mimeData()
        # Add a view to this bundle
        if md.hasFormat(MIME_VIEWSTATE):
            try:
                st = json.loads(bytes(md.data(MIME_VIEWSTATE)).decode("utf-8"))
                doc_ptr = int(st.get("doc_ptr", 0))
                if doc_ptr:
                    self._panel._add_doc_ptrs_to_uuid(self._bundle_uuid, [doc_ptr])
                    # if the panel is showing THIS bundle, refresh its list
                    self._panel._refresh_docs_list_if_current_uuid(self._bundle_uuid)
            except Exception:
                pass
            e.acceptProposedAction()
            return

        # Apply a shortcut to all views in this bundle
        if md.hasFormat(MIME_CMD):
            try:
                payload = _unpack_cmd_safely(bytes(md.data(MIME_CMD)))
                if payload is None:
                    raise ValueError("Unsupported shortcut payload format")
                self._panel._apply_payload_to_bundle(payload, target_uuid=self._bundle_uuid)
                e.acceptProposedAction()
                return
            except Exception as ex:
                QMessageBox.warning(self, "Apply to Bundle", f"Could not parse/execute shortcut:\n{ex}")
        e.ignore()


def spawn_bundle_chip_on_canvas(mw: QWidget, panel: "ViewBundleDialog",
                                bundle_uuid: str, name: str) -> BundleChip | None:
    canvas = _find_shortcut_canvas(mw)
    if not canvas:
        return None

    chip = BundleChip(panel, bundle_uuid, name, parent=canvas)
    chip.resize(190, 46)

    # place near cursor, clamped inside canvas
    pt = canvas.mapFromGlobal(QCursor.pos()) - chip.rect().center()
    pt.setX(max(0, min(pt.x(), canvas.width() - chip.width())))
    pt.setY(max(0, min(pt.y(), canvas.height() - chip.height())))
    chip.move(pt)
    chip.show()
    chip.raise_()
    return chip


# ----------------------------- Select-Views Dialog -----------------------------
class SelectViewsDialog(QDialog):
    """Simple checkbox picker of all open views."""
    def __init__(self, parent: QWidget, choices: list[tuple[str, int]]):
        super().__init__(parent)
        self.setWindowTitle("Add Views to Bundle")
        self.setModal(True)
        self._boxes: list[QCheckBox] = []

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Choose views to add:"))
        v.setSpacing(6)

        box = QVBoxLayout()
        cont = QWidget(); cont.setLayout(box)
        cont.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        for title, ptr in choices:
            cb = QCheckBox(f"{title}")
            cb.setProperty("doc_ptr", int(ptr))
            box.addWidget(cb)
            self._boxes.append(cb)
        box.addStretch(1)
        frame = QFrame(); frame.setLayout(box)
        v.addWidget(frame, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    def selected_ptrs(self) -> list[int]:
        out = []
        for cb in self._boxes:
            if cb.isChecked():
                out.append(int(cb.property("doc_ptr")))
        return out


# ----------------------------- ViewBundleDialog -----------------------------
class ViewBundleDialog(QDialog):
    """
    Pure 'bundle of views' manager.
      • Create many bundles (each with a persistent UUID)
      • Drag a view (from ⧉ tab) → add to bundle
      • Add from list of open views
      • Drop a shortcut (MIME_CMD) onto the bundle/panel/chip → apply to all views in THAT bundle
      • Compress → spawns a small Chip on the ShortcutCanvas that keeps accepting DnD
      • Multiple chips at once (one per bundle)
    """
    SETTINGS_KEY = "viewbundles/v2"  # bumped for uuid

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("View Bundles")
        self.setModal(False)
        self.resize(900, 540)
        self.setAcceptDrops(True)

        self._settings = QSettings()
        self._bundles = self._load_all()  # [{"uuid":str, "name":str, "doc_ptrs":[int,...]}]
        if not self._bundles:
            self._bundles = [{"uuid": self._new_uuid(), "name": "Bundle 1", "doc_ptrs": []}]

        # UI
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # rename UX
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._bundles_context_menu)
        self.list.itemDoubleClicked.connect(lambda _it: self._rename_bundle())
        QShortcut(QKeySequence("F2"), self.list, activated=self._rename_bundle)


        self.docs = QListWidget()
        self.docs.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        self.btn_new = QPushButton("New Bundle")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_del = QPushButton("Delete")
        self.btn_clear = QPushButton("Clear Views")
        self.btn_remove_sel = QPushButton("Remove Selected")
        self.btn_add_from_open = QPushButton("Add from Open…")
        self.btn_compress = QPushButton("Compress to Chip")
        self.drop_hint = QLabel("Drop views here to add • Drop shortcuts here to apply to THIS bundle")
        self.drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_hint.setStyleSheet("color:#aaa; padding:6px; border:1px dashed #666; border-radius:6px;")

        left = QVBoxLayout()
        left.addWidget(QLabel("Bundles"))
        left.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(self.btn_new); row.addWidget(self.btn_dup); row.addWidget(self.btn_del)
        left.addLayout(row)

        right = QVBoxLayout()
        right.addWidget(QLabel("Views in Selected Bundle"))
        right.addWidget(self.docs, 1)

        rrow = QHBoxLayout()
        rrow.addWidget(self.btn_add_from_open)
        rrow.addStretch(1)
        rrow.addWidget(self.btn_remove_sel)
        rrow.addWidget(self.btn_clear)
        right.addLayout(rrow)
        right.addWidget(self.drop_hint)
        right.addWidget(self.btn_compress)

        split = QSplitter()
        wl = QWidget(); wl.setLayout(left)
        wr = QWidget(); wr.setLayout(right)
        split.addWidget(wl); split.addWidget(wr)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

        root = QHBoxLayout(self)
        root.addWidget(split)

        # wiring
        self.btn_new.clicked.connect(self._new_bundle)
        self.btn_dup.clicked.connect(self._dup_bundle)
        self.btn_del.clicked.connect(self._del_bundle)
        self.btn_clear.clicked.connect(self._clear_bundle)
        self.btn_remove_sel.clicked.connect(self._remove_selected)
        self.btn_add_from_open.clicked.connect(self._add_from_open_picker)
        self.btn_compress.clicked.connect(self._compress_to_chip)
        self.list.currentRowChanged.connect(lambda _i: self._refresh_docs_list())

        # populate
        self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(0)

        # chips by uuid
        self._chips: dict[str, BundleChip] = {}  # uuid -> chip widget

    # ---------- persistence ----------
    @staticmethod
    def _new_uuid() -> str:
        return str(uuid.uuid4())

    def _ensure_uuid(self, b: dict):
        if "uuid" not in b or not b["uuid"]:
            b["uuid"] = self._new_uuid()

    def _load_all(self):
        raw = self._settings.value(self.SETTINGS_KEY, "[]", type=str)
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                out = []
                for b in data:
                    if not isinstance(b, dict): continue
                    nm = (b.get("name") or "Bundle").strip()
                    arr = b.get("doc_ptrs") or []
                    arr = [int(x) for x in arr if isinstance(x, (int, str))]
                    u = b.get("uuid") or self._new_uuid()
                    out.append({"uuid": u, "name": nm, "doc_ptrs": arr})
                return out
        except Exception:
            pass
        return []

    def _save_all(self):
        try:
            self._settings.setValue(self.SETTINGS_KEY, json.dumps(self._bundles, ensure_ascii=False))
        except Exception:
            pass

    # ---------- bundle lookups / edits ----------
    def _current_index(self) -> int:
        i = self.list.currentRow()
        if i < 0 or i >= len(self._bundles): return -1
        return i

    def _current_bundle(self) -> Optional[dict]:
        i = self._current_index()
        return None if i < 0 else self._bundles[i]

    def _current_uuid(self) -> Optional[str]:
        b = self._current_bundle()
        return None if not b else b.get("uuid")

    def _get_bundle(self, bundle_uuid: str) -> Optional[dict]:
        for b in self._bundles:
            if b.get("uuid") == bundle_uuid:
                return b
        return None

    def _rename_current_in_list(self, new_name: str):
        i = self._current_index()
        if i < 0: return
        self._bundles[i]["name"] = (new_name or "Bundle").strip()
        self._save_all()
        self._refresh_bundle_list()
        self.list.setCurrentRow(i)
        # sync chip title if exists
        u = self._bundles[i]["uuid"]
        if u in self._chips:
            self._chips[u].sync_from_panel()

    def current_bundle_doc_ptrs(self) -> list[int]:
        b = self._current_bundle()
        return [] if not b else list(b.get("doc_ptrs", []))

    def _set_bundle_ptrs_by_uuid(self, bundle_uuid: str, ptrs: Iterable[int]):
        b = self._get_bundle(bundle_uuid)
        if not b:
            return
        uniq = []
        seen = set()
        for p in ptrs:
            p = int(p)
            if p not in seen:
                seen.add(p); uniq.append(p)
        b["doc_ptrs"] = uniq
        self._save_all()
        # update chip title/count if needed
        if bundle_uuid in self._chips:
            self._chips[bundle_uuid].sync_from_panel()
        # refresh docs if this bundle is selected
        self._refresh_docs_list_if_current_uuid(bundle_uuid)

    def _add_doc_ptrs_to_uuid(self, bundle_uuid: str, ptrs: Iterable[int]):
        b = self._get_bundle(bundle_uuid)
        if not b:
            return
        cur = list(b.get("doc_ptrs", []))
        merged = cur + [int(p) for p in ptrs]
        self._set_bundle_ptrs_by_uuid(bundle_uuid, merged)

    def _set_current_bundle_ptrs(self, ptrs: Iterable[int]):
        u = self._current_uuid()
        if not u: return
        self._set_bundle_ptrs_by_uuid(u, ptrs)

    # ---------- UI refresh ----------
    def _refresh_bundle_list(self):
        self.list.clear()
        for b in self._bundles:
            it = QListWidgetItem(b.get("name", "Bundle"))
            self.list.addItem(it)
        # keep selection reasonable
        if self.list.count() and self.list.currentRow() < 0:
            self.list.setCurrentRow(0)

    # ---------- rename helpers ----------
    def _rename_bundle(self):
        i = self._current_index()
        if i < 0:
            return
        cur = self._bundles[i]
        new_name, ok = QInputDialog.getText(self, "Rename Bundle",
                                            "New name:", text=cur.get("name","Bundle"))
        if not ok:
            return
        self._rename_current_in_list(new_name)

    def _bundles_context_menu(self, pos):
        if self.list.count() == 0:
            return
        # focus the item under cursor (so rename/dup/delete applies to it)
        it = self.list.itemAt(pos)
        if it:
            self.list.setCurrentItem(it)

        m = QMenu(self)
        act_ren = m.addAction("Rename…")
        act_dup = m.addAction("Duplicate")
        act_del = m.addAction("Delete")
        chosen = m.exec(self.list.mapToGlobal(pos))
        if chosen is act_ren:
            self._rename_bundle()
        elif chosen is act_dup:
            self._dup_bundle()
        elif chosen is act_del:
            self._del_bundle()

    def _refresh_docs_list_if_current_uuid(self, bundle_uuid: str):
        if bundle_uuid and bundle_uuid == self._current_uuid():
            self._refresh_docs_list()

    def _refresh_docs_list(self):
        self.docs.clear()
        mw = _find_main_window(self)
        ptrs = self.current_bundle_doc_ptrs()
        for p in ptrs:
            title = f"(unresolved) [{p}]"
            if mw is not None:
                d, sw = _resolve_doc_and_subwindow(mw, p)
                if d is not None:
                    # prefer the subwindow title (has mask glyphs etc.)
                    title = sw.windowTitle() if sw else (getattr(d, "display_name", lambda: "Untitled")())
            it = QListWidgetItem(title)
            it.setData(Qt.ItemDataRole.UserRole, int(p))
            self.docs.addItem(it)

    # ---------- left controls ----------
    def _new_bundle(self):
        b = {"uuid": self._new_uuid(), "name": f"Bundle {len(self._bundles)+1}", "doc_ptrs": []}
        self._bundles.append(b)
        self._save_all(); self._refresh_bundle_list()
        self.list.setCurrentRow(self.list.count() - 1)

    def _dup_bundle(self):
        i = self._current_index()
        if i < 0: return
        b = self._bundles[i]
        cp = {
            "uuid": self._new_uuid(),
            "name": f"{b.get('name','Bundle')} (copy)",
            "doc_ptrs": list(b.get("doc_ptrs", []))
        }
        self._bundles.insert(i + 1, cp)
        self._save_all(); self._refresh_bundle_list()
        self.list.setCurrentRow(i + 1)

    def _del_bundle(self):
        i = self._current_index()
        if i < 0: return
        u = self._bundles[i].get("uuid")
        # remove chip for this bundle, if any
        ch = self._chips.pop(u, None)
        if ch:
            try:
                ch.setParent(None)
                ch.deleteLater()
            except Exception:
                pass
        del self._bundles[i]
        self._save_all(); self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(min(i, self.list.count() - 1))

    # ---------- right controls ----------
    def _clear_bundle(self):
        self._set_current_bundle_ptrs([])

    def _remove_selected(self):
        sel_ptrs = [int(self.docs.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(self.docs.count())
                    if self.docs.item(i).isSelected()]
        if not sel_ptrs: return
        remain = [p for p in self.current_bundle_doc_ptrs() if p not in set(sel_ptrs)]
        self._set_current_bundle_ptrs(remain)

    def _add_from_open_picker(self):
        mw = _find_main_window(self)
        if mw is None:
            QMessageBox.information(self, "Add from Open", "Main window not available.")
            return
        choices: list[tuple[str, int]] = []
        for sw in mw.mdi.subWindowList():
            vw = sw.widget()
            d = getattr(vw, "document", None)
            if d is not None:
                choices.append((sw.windowTitle(), int(id(d))))
        if not choices:
            QMessageBox.information(self, "Add from Open", "No open views.")
            return
        dlg = SelectViewsDialog(self, choices)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            u = self._current_uuid()
            if not u: return
            self._add_doc_ptrs_to_uuid(u, dlg.selected_ptrs())

    def _compress_to_chip(self):
        b = self._current_bundle()
        if not b: return
        u = b["uuid"]; name = b.get("name", "Bundle")

        mw = _find_main_window(self)
        if not mw:
            QMessageBox.information(self, "Compress", "Main window not available.")
            return

        # If a chip for this bundle already exists, just show/raise it
        chip = self._chips.get(u)
        if chip is None or chip.parent() is None:
            chip = spawn_bundle_chip_on_canvas(mw, self, u, name)
            if chip is None:
                QMessageBox.information(self, "Compress", "Shortcut canvas not available.")
                return
            self._chips[u] = chip
        else:
            chip.sync_from_panel()
            chip.show()
            chip.raise_()

        # Do NOT hide the panel automatically; leave it to user preference.
        # If you prefer hiding, uncomment the next line:
        # self.hide()

    # ---------- DnD into the PANEL (applies to CURRENT bundle only) ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME_VIEWSTATE) or e.mimeData().hasFormat(MIME_CMD):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        md = e.mimeData()
        u = self._current_uuid()
        if not u:
            e.ignore(); return

        if md.hasFormat(MIME_VIEWSTATE):
            try:
                st = json.loads(bytes(md.data(MIME_VIEWSTATE)).decode("utf-8"))
                doc_ptr = int(st.get("doc_ptr", 0))
                if doc_ptr:
                    self._add_doc_ptrs_to_uuid(u, [doc_ptr])
            except Exception:
                pass
            e.acceptProposedAction()
            return

        if md.hasFormat(MIME_CMD):
            try:
                payload = _unpack_cmd_safely(bytes(md.data(MIME_CMD)))
                if payload is None:
                    raise ValueError("Unsupported shortcut payload format")
                self._apply_payload_to_bundle(payload, target_uuid=u)
                e.acceptProposedAction()
                return
            except Exception as ex:
                QMessageBox.warning(self, "Apply to Bundle", f"Could not parse/execute shortcut:\n{ex}")
        e.ignore()

    # ---------- applying shortcuts to all views in a bundle ----------
    def _apply_payload_to_bundle(self, payload: dict, target_uuid: Optional[str] = None):
        mw = _find_main_window(self)
        if mw is None or not hasattr(mw, "_handle_command_drop"):
            QMessageBox.information(self, "Apply", "Main window not available.")
            return
        payload = _unwrap_cmd_payload(payload)
        cmd_val = (payload or {}).get("command_id")
        cmd = cmd_val if isinstance(cmd_val, str) else None
        if not cmd:
            QMessageBox.information(self, "Apply", "Invalid shortcut payload.")
            return
        
        # pick doc_ptrs from the target bundle (chip drop) or current selection (panel drop)
        if target_uuid:
            b = self._get_bundle(target_uuid)
            ptrs = [] if not b else list(b.get("doc_ptrs", []))
            
        else:
            ptrs = self.current_bundle_doc_ptrs()
            
        # Apply a Function Bundle (multiple steps) to every view in the bundle
        if cmd == "function_bundle":
            
            # 1) normalize to plain JSON-safe dicts (deep copy)
            try:
                
                steps = json.loads(json.dumps((payload or {}).get("steps") or []))
            except Exception:
                
                steps = list((payload or {}).get("steps") or [])

            # 2) sanity filter: only dict steps with a command_id
            norm_steps = [s for s in steps if isinstance(s, dict) and s.get("command_id")]
            
            if not norm_steps:
                QMessageBox.information(self, "Apply", "This Function Bundle has no usable steps.")
                return

            # 3) apply using the same sequencing as the button
            errors, applied = [], 0
            for ptr in ptrs:
                _doc, sw = _resolve_doc_and_subwindow(mw, ptr)
                
                if sw is None:
                    print(f"    no subwindow found for doc_ptr {ptr}")
                    continue
                try:
                    # activate the target like the button runner does
                    try:
                        
                        if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not sw:
                            mw.mdi.setActiveSubWindow(sw)
                        w = getattr(sw, "widget", lambda: None)()
                        if w:
                            w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
                    except Exception:
                        
                        pass

                    for st in norm_steps:
                        mw._handle_command_drop(st, target_sw=sw)
                        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)
                    applied += 1
                except Exception as e:
                    errors.append(str(e))

            if applied == 0 and not errors:
                QMessageBox.information(self, "Apply", "No valid targets in the bundle.")
            elif errors:
                QMessageBox.warning(self, "Apply", "Applied some steps but some failed:\n\n" + "\n".join(errors))
            return

        # Ignore nested view-bundle shortcuts
        if cmd == "bundle":
            return

        # Single-step shortcut → apply to all docs in the chosen bundle
        errors = []
        applied = 0
        for ptr in ptrs:
            doc, sw = _resolve_doc_and_subwindow(mw, ptr)
            if sw is None:
                continue
            try:
                # 🔸 Make this subwindow active (some commands operate on the active view)
                try:
                    if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not sw:
                        mw.mdi.setActiveSubWindow(sw)
                    w = getattr(sw, "widget", lambda: None)()
                    if w:
                        w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                    # Let the UI/app catch up so the command sees the right active view
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
                except Exception:
                    pass

                mw._handle_command_drop(payload, target_sw=sw)

                # Small pump so commands that spawn work or flip views settle before next doc
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)
                applied += 1
            except Exception as e:
                errors.append(str(e))

        if applied == 0 and not errors:
            QMessageBox.information(self, "Apply", "No valid targets in the bundle.")
        elif errors:
            QMessageBox.warning(self, "Apply", f"Applied to {applied} views, but some failed:\n\n" + "\n".join(errors))

    def closeEvent(self, e: QCloseEvent):
        # keep chips alive; nothing to do
        super().closeEvent(e)


# ----------------------------- singleton open helpers -----------------------------
_dialog_singleton: ViewBundleDialog | None = None

def show_view_bundles(parent: QWidget | None, focus_name: str | None = None):
    """
    Open (or focus) the View Bundles dialog. Optionally set focus to a bundle name.
    """
    global _dialog_singleton
    if _dialog_singleton is None:
        _dialog_singleton = ViewBundleDialog(parent)
        # ensure singleton cleared on destroy
        def _clear():
            global _dialog_singleton
            _dialog_singleton = None
        _dialog_singleton.destroyed.connect(_clear)

    if focus_name:
        # try to select the bundle by name
        for i in range(_dialog_singleton.list.count()):
            if _dialog_singleton.list.item(i).text().strip() == focus_name.strip():
                _dialog_singleton.list.setCurrentRow(i)
                break

    _dialog_singleton.show()
    _dialog_singleton.raise_()
    _dialog_singleton.activateWindow()
    return _dialog_singleton
