# pro/view_bundle.py
from __future__ import annotations
import json
import uuid
import os
from typing import Iterable, Optional
import sys
from PyQt6.QtCore import Qt, QSettings, QByteArray, QMimeData, QSize, QPoint, QEventLoop
from PyQt6.QtWidgets import (
    QDialog, QWidget, QHBoxLayout, QVBoxLayout, QListWidget, QListWidgetItem,QApplication,
    QPushButton, QSplitter, QLabel, QAbstractItemView, QDialogButtonBox,
    QCheckBox, QFrame, QSizePolicy, QMenu, QInputDialog, QFileDialog
)
import traceback
from PyQt6.QtWidgets import QMessageBox as _QMB
from PyQt6.QtGui import QDrag, QCloseEvent, QCursor, QShortcut, QKeySequence
from legacy.image_manager import load_image, save_image
from pro.dnd_mime import MIME_CMD, MIME_VIEWSTATE
from pro.doc_manager import ImageDocument

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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # ← so Delete/Backspace work

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
        self._grab_offset = None
        self._dragging = False

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
            self.setFocus(Qt.FocusReason.MouseFocusReason)  # ← focus for Delete key
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
            # persist chip positions when a drag finishes
            try:
                self._panel._save_chip_layout()
            except Exception:
                pass
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

    def contextMenuEvent(self, ev):
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

        if md.hasUrls():
            paths = []
            for url in md.urls():
                p = url.toLocalFile()
                if not p: continue
                if os.path.isdir(p):
                    for r, d, files in os.walk(p):
                        for f in files:
                            if f.lower().endswith(tuple(x.lower() for x in self._panel._file_exts())):
                                paths.append(os.path.join(r, f))
                else:
                    if p.lower().endswith(tuple(x.lower() for x in self._panel._file_exts())):
                        paths.append(p)
            if paths:
                self._panel._add_files_to_uuid(self._bundle_uuid, paths)
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
                _QMB.warning(self, "Apply to Bundle", f"Could not parse/execute shortcut:\n{ex}")
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
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._boxes: list[QCheckBox] = []

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Choose views to add:"))
        v.setSpacing(6)

        # NEW: "Select all" checkbox
        self._select_all = QCheckBox("Select all open views")
        self._select_all.toggled.connect(self._on_select_all_toggled)
        v.addWidget(self._select_all)

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

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)

    # NEW: handler for the "Select all" checkbox
    def _on_select_all_toggled(self, checked: bool):
        for cb in self._boxes:
            cb.setChecked(checked)

    def selected_ptrs(self) -> list[int]:
        """Return list of doc_ptrs for checked boxes."""
        return [int(cb.property("doc_ptr")) for cb in self._boxes if cb.isChecked()]



class _HeadlessView:
    def __init__(self, doc, mw):
        self.document = doc
        self._mw = mw

    def apply_command(self, command_id: str, preset: dict | None = None):
        # Best-effort fallback if someone calls this directly.
        preset = preset or {}
        apply_to_view = getattr(self._mw, "apply_command_to_view", None)
        if callable(apply_to_view):
            return apply_to_view(self, command_id, preset)
        # If nothing else exists, just no-op (don’t raise a user-facing error).
        return None

class _FakeSubWindow:
    """Headless stand-in that gives _handle_command_drop a .widget() with .document."""
    def __init__(self, view):
        self._view = view
    def widget(self):
        return self._view
    def windowTitle(self):
        # Try to mirror what a real subwindow title would show
        try:
            doc = getattr(self._view, "document", None)
            if doc:
                name = getattr(doc, "display_name", None)
                if callable(name):
                    return name()
                # common fallback attribute(s)
                return getattr(doc, "name", None) or getattr(doc, "filename", None) or "view"
        except Exception:
            pass
        return "view"


def _apply_one_shortcut_to_doc(mw, doc, payload: dict):
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid shortcut payload")

    cid = payload.get("command_id")
    if isinstance(cid, dict):
        payload = cid
        cid = payload.get("command_id")
    if not isinstance(cid, str) or not cid:
        raise RuntimeError("Invalid command id")
    if cid == "bundle":
        return  # ignore nested bundles

    view = _HeadlessView(doc, mw)

    # 1) Primary: same as canvas → ShortcutManager path
    handle = getattr(mw, "_handle_command_drop", None)
    if callable(handle):
        # Pass a fake subwindow whose widget() returns our headless view
        fake_sw = _FakeSubWindow(view)
        handle(payload, target_sw=fake_sw)
        return

    # 2) Secondary: explicit apply-to-view hook
    apply_to_view = getattr(mw, "apply_command_to_view", None)
    if callable(apply_to_view):
        apply_to_view(view, cid, payload.get("preset") or {})
        return

    # 3) Last-resort: let the shim try a no-op-safe apply_command
    view.apply_command(cid, payload.get("preset") or {})



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
    SETTINGS_KEY = "viewbundles/v3"  # bumped for uuid
    CHIP_KEY     = "viewbundles/chips_v1"  # ← new: chip layout    

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        _pin_on_top_mac(self)
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
        # Context menu + double-click niceties on the bundle's treebox/list
        self.docs.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.docs.customContextMenuRequested.connect(self._docs_context_menu)
        self.docs.itemDoubleClicked.connect(self._docs_item_activated)
        self.btn_new = QPushButton("New Bundle")
        self.btn_dup = QPushButton("Duplicate")
        self.btn_del = QPushButton("Delete")
        self.btn_clear = QPushButton("Clear Views")
        self.btn_remove_sel = QPushButton("Remove Selected")
        self.btn_add_from_open = QPushButton("Add from Open…")
        self.btn_add_files = QPushButton("Add Files…")
        self.btn_add_dir   = QPushButton("Add Directory (Recursive)…")        
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
        rrow2 = QHBoxLayout()
        rrow2.addWidget(self.btn_add_files)
        rrow2.addWidget(self.btn_add_dir)
        right.addLayout(rrow2)        
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
        self.btn_add_files.clicked.connect(self._add_files_into_bundle)
        self.btn_add_dir.clicked.connect(self._add_directory_into_bundle)
        # populate
        self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(0)

        # chips by uuid
        self._chips: dict[str, BundleChip] = {}  # uuid -> chip widget

        try:
            self._restore_chips_from_settings()
        except Exception:
            pass

    def _save_chip_layout(self):
        """
        Persist current bundle chips and their positions so they reappear
        on the ShortcutCanvas next time SASpro is opened.
        """
        try:
            data = []
            for uuid, chip in list(self._chips.items()):
                if chip is None or chip.parent() is None:
                    continue
                pos = chip.pos()
                data.append({
                    "uuid": str(uuid),
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
                u = str(entry.get("uuid", "")).strip()
            except Exception:
                continue
            if not u:
                continue

            # must still exist as a bundle
            b = self._get_bundle(u)
            if not b:
                continue

            name = b.get("name", "Bundle")
            chip = spawn_bundle_chip_on_canvas(mw, self, u, name)
            if chip is None:
                continue

            x = entry.get("x")
            y = entry.get("y")
            if isinstance(x, int) and isinstance(y, int):
                chip.move(x, y)

            self._chips[u] = chip

    def _remove_chip_widget(self, chip: BundleChip):
        """
        Remove a chip from the canvas and our uuid→chip registry,
        without deleting the underlying bundle.
        """
        # drop from the mapping
        for u, ch in list(self._chips.items()):
            if ch is chip:
                self._chips.pop(u, None)
                break

        try:
            chip.setParent(None)
            chip.deleteLater()
        except Exception:
            pass

        self._save_chip_layout()


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
                    if not isinstance(b, dict): 
                        continue
                    nm = (b.get("name") or "Bundle").strip()
                    ptrs = [int(x) for x in (b.get("doc_ptrs") or []) if isinstance(x, (int, str))]
                    fps  = [str(p) for p in (b.get("file_paths") or []) if isinstance(p, (str,))]
                    u = b.get("uuid") or self._new_uuid()
                    out.append({"uuid": u, "name": nm, "doc_ptrs": ptrs, "file_paths": fps})
                return out
        except Exception:
            pass
        return []

    def _save_all(self):
        try:
            # ensure keys exist
            for b in self._bundles:
                b.setdefault("doc_ptrs", [])
                b.setdefault("file_paths", [])
            self._settings.setValue(self.SETTINGS_KEY, json.dumps(self._bundles, ensure_ascii=False))
        except Exception:
            pass

    def _file_exts(self):
        return (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fz", ".xisf", ".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def _add_files_into_bundle(self):
        u = self._current_uuid()
        if not u:
            return
        last_dir = QSettings().value("last_opened_folder", "", type=str)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files for Bundle", last_dir,
            "Images (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz *.xisf *.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if not files:
            return
        QSettings().setValue("last_opened_folder", os.path.dirname(files[0]))
        # Dedup in bundle
        self._add_files_to_uuid(u, files)

    def _add_directory_into_bundle(self):
        u = self._current_uuid()
        if not u:
            return
        last_dir = QSettings().value("last_opened_folder", "", type=str)
        directory = QFileDialog.getExistingDirectory(self, "Select Directory for Bundle", last_dir)
        if not directory:
            return
        QSettings().setValue("last_opened_folder", directory)
        exts = tuple(x.lower() for x in self._file_exts())
        found = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(exts):
                    found.append(os.path.join(root, f))
        if not found:
            _QMB.information(self, "Add Directory", "No supported images found recursively.")
            return
        self._add_files_to_uuid(u, found)


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
                # normalize keys
                b.setdefault("doc_ptrs", [])
                b.setdefault("file_paths", [])
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

    def _set_bundle_files_by_uuid(self, bundle_uuid: str, paths: Iterable[str]):
        b = self._get_bundle(bundle_uuid)
        if not b:
            return
        uniq = []
        seen = set()
        for p in paths:
            p = str(p)
            if p not in seen:
                seen.add(p); uniq.append(p)
        b["file_paths"] = uniq
        self._save_all()
        if bundle_uuid in self._chips:
            self._chips[bundle_uuid].sync_from_panel()
        self._refresh_docs_list_if_current_uuid(bundle_uuid)

    def _add_files_to_uuid(self, bundle_uuid: str, paths: Iterable[str]):
        b = self._get_bundle(bundle_uuid)
        if not b:
            return
        cur = list(b.get("file_paths", []))
        merged = cur + [str(p) for p in paths]
        self._set_bundle_files_by_uuid(bundle_uuid, merged)

    def current_bundle_file_paths(self) -> list[str]:
        b = self._current_bundle()
        if not b: return []
        return list(b.get("file_paths", []))

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
        # --- views ---
        for p in self.current_bundle_doc_ptrs():
            title = f"(unresolved) [{p}]"
            if mw is not None:
                d, sw = _resolve_doc_and_subwindow(mw, p)
                if d is not None:
                    title = sw.windowTitle() if sw else (getattr(d, "display_name", lambda: "Untitled")())
            it = QListWidgetItem(f"[view] {title}")
            it.setData(Qt.ItemDataRole.UserRole, int(p))
            it.setData(Qt.ItemDataRole.UserRole + 1, "view")
            self.docs.addItem(it)
        # --- files ---
        for path in self.current_bundle_file_paths():
            it = QListWidgetItem(f"[file] {path}")
            it.setData(Qt.ItemDataRole.UserRole, path)
            it.setData(Qt.ItemDataRole.UserRole + 1, "file")
            self.docs.addItem(it)

    # ---------- list niceties: context menu + double-click ----------
    def _docs_item_kind_and_value(self, it):
        """Return ('view'|'file', value) from a QListWidgetItem."""
        if not it:
            return None, None
        kind = it.data(Qt.ItemDataRole.UserRole + 1)
        val  = it.data(Qt.ItemDataRole.UserRole)
        return kind, val

    def _docs_item_activated(self, it):
        """Double-click action: open file, or focus view."""
        kind, val = self._docs_item_kind_and_value(it)
        if kind == "file":
            self._open_file_in_new_view(str(val))
        elif kind == "view":
            self._focus_view_ptr(int(val))

    def _docs_context_menu(self, pos):
        if self.docs.count() == 0:
            return
        # Focus the item under the cursor so actions apply sensibly
        it = self.docs.itemAt(pos)
        if it:
            it.setSelected(True)

        # Gather selection breakdown
        sel = [self.docs.item(i) for i in range(self.docs.count()) if self.docs.item(i).isSelected()]
        file_items = [s for s in sel if self._docs_item_kind_and_value(s)[0] == "file"]
        view_items = [s for s in sel if self._docs_item_kind_and_value(s)[0] == "view"]
        if not file_items and not view_items:
            return

        m = QMenu(self)
        act_open_files = act_focus_views = None
        if file_items:
            lab = "Open in New View" if len(file_items) == 1 else f"Open {len(file_items)} Files in New Views"
            act_open_files = m.addAction(lab)
        if view_items:
            labv = "Focus View" if len(view_items) == 1 else f"Focus {len(view_items)} Views"
            act_focus_views = m.addAction(labv)

        chosen = m.exec(self.docs.mapToGlobal(pos))
        if chosen is act_open_files:
            for itf in file_items:
                _, path = self._docs_item_kind_and_value(itf)
                self._open_file_in_new_view(str(path))
        elif chosen is act_focus_views:
            for itv in view_items:
                _, ptr = self._docs_item_kind_and_value(itv)
                self._focus_view_ptr(int(ptr))

    def _focus_view_ptr(self, doc_ptr: int):
        mw = _find_main_window(self)
        if mw is None:
            return
        doc, sw = _resolve_doc_and_subwindow(mw, doc_ptr)
        if sw is None:
            return
        try:
            if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not sw:
                mw.mdi.setActiveSubWindow(sw)
            w = getattr(sw, "widget", lambda: None)()
            if w:
                w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        except Exception:
            pass

    def _open_file_in_new_view(self, path: str):
        """Open a bundle-listed file into a brand-new view (no save/overwrite)."""
        mw = _find_main_window(self)
        if mw is None:
            _QMB.information(self, "Open", "Main window not available.")
            return
        try:
            sw = None
            opened_doc = None
            # Prefer docman API if present
            if hasattr(mw, "docman") and hasattr(mw.docman, "open_path"):
                opened_doc = mw.docman.open_path(path)
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 120)
                if opened_doc is not None and hasattr(mw, "_find_doc_by_id"):
                    _doc, sw = mw._find_doc_by_id(id(opened_doc))
            # Fallback to legacy open hook
            if sw is None:
                if hasattr(mw, "_open_image"):
                    mw._open_image(path)
                else:
                    raise RuntimeError("No file-open method found on main window")
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 120)
                # best-effort: find by title tail match
                bn = os.path.basename(path)
                for cand in getattr(mw.mdi, "subWindowList", lambda: [])():
                    if bn in cand.windowTitle():
                        sw = cand
                        break
            # Focus the new subwindow
            if sw is not None:
                if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not sw:
                    mw.mdi.setActiveSubWindow(sw)
                w = getattr(sw, "widget", lambda: None)()
                if w:
                    w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        except Exception as e:
            _QMB.warning(self, "Open", f"Could not open:\n{path}\n\n{e}")

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
        self._save_all()
        self._refresh_bundle_list()
        if self.list.count():
            self.list.setCurrentRow(min(i, self.list.count() - 1))

        # update chip layout persistence
        try:
            self._save_chip_layout()
        except Exception:
            pass


    # ---------- right controls ----------
    def _clear_bundle(self):
        self._set_current_bundle_ptrs([])
        u = self._current_uuid()
        if u: self._set_bundle_files_by_uuid(u, [])

    def _remove_selected(self):
        view_ptrs, file_paths = [], []
        for i in range(self.docs.count()):
            it = self.docs.item(i)
            if not it.isSelected():
                continue
            kind = it.data(Qt.ItemDataRole.UserRole + 1)
            if kind == "view":
                view_ptrs.append(int(it.data(Qt.ItemDataRole.UserRole)))
            elif kind == "file":
                file_paths.append(str(it.data(Qt.ItemDataRole.UserRole)))

        if view_ptrs:
            remain = [p for p in self.current_bundle_doc_ptrs() if p not in set(view_ptrs)]
            self._set_current_bundle_ptrs(remain)
        if file_paths:
            remain = [p for p in self.current_bundle_file_paths() if p not in set(file_paths)]
            u = self._current_uuid()
            if u: self._set_bundle_files_by_uuid(u, remain)

    def _add_from_open_picker(self):
        mw = _find_main_window(self)
        if mw is None:
            _QMB.information(self, "Add from Open", "Main window not available.")
            return
        choices: list[tuple[str, int]] = []
        for sw in mw.mdi.subWindowList():
            vw = sw.widget()
            d = getattr(vw, "document", None)
            if d is not None:
                choices.append((sw.windowTitle(), int(id(d))))
        if not choices:
            _QMB.information(self, "Add from Open", "No open views.")
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
            _QMB.information(self, "Compress", "Main window not available.")
            return

        # If a chip for this bundle already exists, just show/raise it
        chip = self._chips.get(u)
        if chip is None or chip.parent() is None:
            chip = spawn_bundle_chip_on_canvas(mw, self, u, name)
            if chip is None:
                _QMB.information(self, "Compress", "Shortcut canvas not available.")
                return
            self._chips[u] = chip
        else:
            chip.sync_from_panel()
            chip.show()
            chip.raise_()

        # persist chip presence/position
        try:
            self._save_chip_layout()
        except Exception:
            pass


    # ---------- DnD into the PANEL (applies to CURRENT bundle only) ----------
    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasFormat(MIME_VIEWSTATE) or md.hasFormat(MIME_CMD) or md.hasUrls():
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

        if md.hasUrls():
            paths = []
            for url in md.urls():
                p = url.toLocalFile()
                if not p: 
                    continue
                if os.path.isdir(p):
                    for r, d, files in os.walk(p):
                        for f in files:
                            if f.lower().endswith(tuple(x.lower() for x in self._file_exts())):
                                paths.append(os.path.join(r, f))
                else:
                    if p.lower().endswith(tuple(x.lower() for x in self._file_exts())):
                        paths.append(p)
            if paths:
                self._add_files_to_uuid(u, paths)
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
                _QMB.warning(self, "Apply to Bundle", f"Could not parse/execute shortcut:\n{ex}")
        e.ignore()

    # ---------- applying shortcuts to all views in a bundle ----------
    def _apply_payload_to_bundle(self, payload: dict, target_uuid: Optional[str] = None):
        mw = _find_main_window(self)
        if mw is None or not hasattr(mw, "_handle_command_drop"):
            _QMB.information(self, "Apply", "Main window not available.")
            return

        payload = _unwrap_cmd_payload(payload)
        cmd_val = (payload or {}).get("command_id")
        cmd = cmd_val if isinstance(cmd_val, str) else None
        if not cmd:
            _QMB.information(self, "Apply", "Invalid shortcut payload.")
            return
        if cmd == "bundle":
            return  # ignore nested bundles

        # --- gather targets ---
        if target_uuid:
            b = self._get_bundle(target_uuid)
            ptrs = [] if not b else list(b.get("doc_ptrs", []))
            file_paths = [] if not b else list(b.get("file_paths", []))
        else:
            ptrs = self.current_bundle_doc_ptrs()
            file_paths = self.current_bundle_file_paths()

        # --- counters / errors ---
        view_applied = 0
        file_ok = 0
        view_errors: list[str] = []
        file_errors: list[str] = []

        # ---------- Apply to OPEN VIEWS ----------
        if cmd == "function_bundle":
            try:
                steps = json.loads(json.dumps((payload or {}).get("steps") or []))
            except Exception:
                steps = list((payload or {}).get("steps") or [])
            norm_steps = [s for s in steps if isinstance(s, dict) and s.get("command_id")]

            if norm_steps:
                for ptr in ptrs:
                    _doc, sw = _resolve_doc_and_subwindow(mw, ptr)
                    if sw is None:
                        continue
                    try:
                        if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not sw:
                            mw.mdi.setActiveSubWindow(sw)
                        w = getattr(sw, "widget", lambda: None)()
                        if w:
                            w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

                        for st in norm_steps:
                            mw._handle_command_drop(st, target_sw=sw)
                            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)
                        view_applied += 1
                    except Exception as e:
                        view_errors.append(str(e))
            # else: no steps → we’ll still try files below
        else:
            for ptr in ptrs:
                _doc, sw = _resolve_doc_and_subwindow(mw, ptr)
                if sw is None:
                    continue
                try:
                    if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not sw:
                        mw.mdi.setActiveSubWindow(sw)
                    w = getattr(sw, "widget", lambda: None)()
                    if w:
                        w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)

                    mw._handle_command_drop(payload, target_sw=sw)
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)
                    view_applied += 1
                except Exception as e:
                    view_errors.append(str(e))

        # start total with views
        total_applied = view_applied

        # ---------- Apply to FILE PATHS ----------
        if file_paths:
            for p in file_paths:
                try:
                    self._apply_payload_to_single_file(payload, p, overwrite=True, out_dir=None)
                    file_ok += 1
                except Exception as e:
                    tb = traceback.format_exc(limit=6)
                    file_errors.append(f"{os.path.basename(p)}: {e.__class__.__name__}: {e}\n{tb}")

            total_applied += file_ok

        # ---------- Final summary ----------
        if total_applied == 0 and not (view_errors or file_errors):
            _QMB.information(self, "Apply", "No valid targets in the bundle.")
            return

        # If there were any errors, show a detailed mixed summary
        if view_errors or file_errors:
            msg = []
            if view_applied:
                msg.append(f"Applied to {view_applied} open view(s).")
            if file_ok:
                msg.append(f"Applied to {file_ok} file(s).")
            if view_errors:
                msg.append("View errors:\n  " + "\n  ".join(view_errors))
            if file_errors:
                msg.append("File errors:\n  " + "\n  ".join(file_errors))
            _QMB.warning(self, "Apply", "\n\n".join(msg))
            return

        _QMB.information(self, "Apply", f"Finished. Applied to {total_applied} target(s).")
            



    def closeEvent(self, e: QCloseEvent):
        # keep chips alive; nothing to do
        super().closeEvent(e)

    def _path_format_from_ext(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext in ("jpeg",): ext = "jpg"
        return ext or "fits"

    def _resolve_file_target(self, src_path: str, overwrite: bool, out_dir: str | None) -> str:
        return (src_path if overwrite or not out_dir
                else os.path.join(out_dir, os.path.basename(src_path)))

    def _apply_payload_to_single_file(self, payload: dict, path: str,
                                    overwrite: bool = True, out_dir: Optional[str] = None) -> bool:
        """
        Headless batch that avoids DocManager completely:
        - load with legacy I/O
        - wrap in a transient ImageDocument (not added to DocManager)
        - apply shortcuts via the same dispatcher using a FakeSubWindow
        - save with legacy I/O
        """
        mw = _find_main_window(self)
        if mw is None:
            raise RuntimeError("Main window not available")

        # 1) load from disk (no signals, no UI)
        img, header, bit_depth, is_mono = load_image(path)
        if img is None:
            raise RuntimeError(f"Could not load: {path}")

        meta = {
            "file_path": path,
            "original_header": header,
            "bit_depth": bit_depth,
            "is_mono": is_mono,
            "original_format": self._path_format_from_ext(path),
        }
        # transient doc (NOT registered anywhere)
        doc = ImageDocument(img, meta)

        # 2) apply
        pl = _unwrap_cmd_payload(payload) or {}
        cid = pl.get("command_id")
        if not isinstance(cid, str):
            raise RuntimeError("Invalid shortcut payload")

        if cid == "function_bundle":
            steps = [s for s in (pl.get("steps") or []) if isinstance(s, dict) and s.get("command_id")]
            if not steps:
                raise RuntimeError("Function Bundle has no usable steps")
            for st in steps:
                _apply_one_shortcut_to_doc(mw, doc, st)
        elif cid != "bundle":  # ignore nested bundles
            _apply_one_shortcut_to_doc(mw, doc, pl)

        # 3) save back (still no UI)
        target_path = self._resolve_file_target(path, overwrite, out_dir)
        ext = os.path.splitext(target_path)[1].lower().lstrip(".")
        # use legacy writer directly; mirror DocManager’s parameter mapping
        save_image(
            img_array=doc.image,
            filename=target_path,
            original_format=ext,
            bit_depth=doc.metadata.get("bit_depth", "32-bit floating point"),
            original_header=doc.metadata.get("original_header"),
            is_mono=doc.metadata.get("is_mono", getattr(doc.image, "ndim", 2) == 2),
            image_meta=doc.metadata.get("image_meta"),
            file_meta=doc.metadata.get("file_meta"),
        )

        return True


    def _apply_payload_to_single_file_via_ui(self, payload: dict, path: str,
                                            overwrite: bool = True, out_dir: Optional[str] = None) -> bool:
        """
        Your previous UI-based routine, but using docman.open_path(path) (no file picker).
        """
        mw = _find_main_window(self)
        if mw is None:
            raise RuntimeError("Main window not available")

        before = set(getattr(mw.mdi, "subWindowList", lambda: [])())
        opened_sw = None
        opened_doc = None
        try:
            if hasattr(mw, "docman") and hasattr(mw.docman, "open_path"):
                opened_doc = mw.docman.open_path(path)      # no dialogs, emits documentAdded
            elif hasattr(mw, "_open_image"):
                mw._open_image(path)
            else:
                raise RuntimeError("No file-open method found on main window")

            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 150)

            if opened_doc is not None and hasattr(mw, "_find_doc_by_id"):
                _doc, sw = mw._find_doc_by_id(id(opened_doc))
                opened_sw = sw

            if opened_sw is None:
                bn = os.path.basename(path)
                for sw in getattr(mw.mdi, "subWindowList", lambda: [])():
                    if bn in sw.windowTitle():
                        opened_sw = sw
                        break
        except Exception as e:
            raise RuntimeError(f"Open failed: {e}")

        if opened_sw is None:
            raise RuntimeError("Could not resolve newly opened view")

        try:
            if hasattr(mw, "mdi") and mw.mdi.activeSubWindow() is not opened_sw:
                mw.mdi.setActiveSubWindow(opened_sw)
            w = getattr(opened_sw, "widget", lambda: None)()
            if w:
                w.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        except Exception:
            pass

        def _apply_one(st):
            mw._handle_command_drop(st, target_sw=opened_sw)

        pl = _unwrap_cmd_payload(payload) or {}
        if pl.get("command_id") == "function_bundle":
            steps = pl.get("steps") or []
            steps = [s for s in steps if isinstance(s, dict) and s.get("command_id")]
            if not steps:
                raise RuntimeError("Function Bundle has no usable steps")
            for st in steps:
                _apply_one(st)
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)
        elif pl.get("command_id") == "bundle":
            pass
        else:
            _apply_one(pl)
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)

        # save back
        target_path = self._resolve_file_target(path, overwrite, out_dir)
        saved = False
        try:
            vw = getattr(opened_sw, "widget", lambda: None)()
            doc = getattr(vw, "document", None) if vw else None
            if doc and hasattr(doc, "save_to_path"):
                doc.save_to_path(target_path); saved = True
            elif doc and hasattr(doc, "save"):
                try:
                    doc.save(target_path); saved = True
                except Exception:
                    if hasattr(doc, "set_filename"):
                        doc.set_filename(target_path); doc.save(); saved = True
            if not saved and hasattr(mw, "_save_active_document_as"):
                mw._save_active_document_as(target_path); saved = True
            if not saved and hasattr(mw, "_save_document_as") and doc:
                mw._save_document_as(doc, target_path); saved = True
            if not saved and hasattr(mw, "_save_document") and doc:
                mw._save_document(doc); saved = True
            if not saved:
                raise RuntimeError("No save method available")
        finally:
            try:
                opened_sw.close()
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 0)
            except Exception:
                pass

        return True



# ----------------------------- singleton open helpers -----------------------------
_dialog_singleton: ViewBundleDialog | None = None

def show_view_bundles(parent: QWidget | None,
                      focus_name: str | None = None,
                      *,
                      auto_spawn_only: bool = False):
    """
    Open (or focus) the View Bundles dialog. Optionally set focus to a bundle name.

    If auto_spawn_only=True, ensure the dialog + chips exist,
    but do NOT show the dialog (for startup chip restore).
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

    if not auto_spawn_only:
        _dialog_singleton.show()
        _dialog_singleton.raise_()
        _dialog_singleton.activateWindow()
    return _dialog_singleton

def restore_view_bundle_chips(parent: QWidget | None):
    """
    Called at app startup: create the ViewBundleDialog singleton,
    restore any saved chips onto the ShortcutCanvas, but keep the
    dialog itself hidden.
    """
    try:
        show_view_bundles(parent, auto_spawn_only=True)
    except Exception:
        # fail silently; nothing critical here
        pass
