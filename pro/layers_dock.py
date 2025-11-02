#pro.layers_dock.py
from __future__ import annotations
from typing import Optional
import json
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QByteArray, QTimer
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QSlider, QCheckBox,
    QPushButton, QFrame, QMessageBox
)
from PyQt6.QtGui import QIcon, QDragEnterEvent, QDropEvent, QPixmap, QCursor

from pro.dnd_mime import MIME_VIEWSTATE, MIME_MASK
from pro.layers import composite_stack, ImageLayer, BLEND_MODES

# ---------- Small row widget for a layer ----------
class _LayerRow(QWidget):
    changed = pyqtSignal()
    requestDelete = pyqtSignal()
    moveUp = pyqtSignal()
    moveDown = pyqtSignal()

    def __init__(self, name: str, mode: str = "Normal", opacity: float = 1.0,
                 visible: bool = True, parent=None, *, is_base: bool = False):
        super().__init__(parent)
        self._name = name
        self._is_base = bool(is_base)

        v = QVBoxLayout(self); v.setContentsMargins(6, 2, 6, 2)

        # row 1: visibility, name, mode, opacity, reorder/delete
        r1 = QHBoxLayout(); v.addLayout(r1)
        self.chk = QCheckBox(); self.chk.setChecked(visible)
        self.lbl = QLabel(name)
        self.mode = QComboBox(); self.mode.addItems(BLEND_MODES)
        try: self.mode.setCurrentIndex(max(0, BLEND_MODES.index(mode)))
        except Exception: self.mode.setCurrentIndex(0)
        self.sld = QSlider(Qt.Orientation.Horizontal); self.sld.setRange(0, 100); self.sld.setValue(int(round(opacity*100)))
        self.btn_up = QPushButton("↑"); self.btn_up.setFixedWidth(28)
        self.btn_dn = QPushButton("↓"); self.btn_dn.setFixedWidth(28)
        self.btn_x  = QPushButton("✕"); self.btn_x.setFixedWidth(28)

        r1.addWidget(self.chk); r1.addWidget(self.lbl, 1)
        r1.addWidget(self.mode); r1.addWidget(QLabel("Opacity")); r1.addWidget(self.sld, 1)
        r1.addWidget(self.btn_up); r1.addWidget(self.btn_dn); r1.addWidget(self.btn_x)

        # row 2: mask controls (hidden for base)
        r2 = QHBoxLayout(); v.addLayout(r2)
        self.mask_combo = QComboBox(); self.mask_combo.setMinimumWidth(140)
        self.mask_combo.setPlaceholderText("Mask: (none)")
        self.mask_src = QComboBox(); self.mask_src.addItems(["Active Mask", "Luminance"])
        self.mask_invert = QCheckBox("Invert")
        self.btn_clear_mask = QPushButton("Clear"); self.btn_clear_mask.setFixedWidth(52)
        r2.addWidget(QLabel("Mask")); r2.addWidget(self.mask_combo, 1)
        r2.addWidget(self.mask_src); r2.addWidget(self.mask_invert); r2.addWidget(self.btn_clear_mask)

        if self._is_base:
            # Base row is informational only
            for w in (self.chk, self.mode, self.sld, self.btn_up, self.btn_dn, self.btn_x,
                      self.mask_combo, self.mask_src, self.mask_invert, self.btn_clear_mask):
                w.setEnabled(False)
            self.lbl.setStyleSheet("color: palette(mid);")
        else:
            self.chk.stateChanged.connect(self._emit)
            self.mode.currentIndexChanged.connect(self._emit)
            self.sld.valueChanged.connect(self._emit)
            self.mask_combo.currentIndexChanged.connect(self._emit)
            self.mask_src.currentIndexChanged.connect(self._emit)
            self.mask_invert.stateChanged.connect(self._emit)
            self.btn_clear_mask.clicked.connect(self._on_clear_mask)
            self.btn_x.clicked.connect(self.requestDelete.emit)
            self.btn_up.clicked.connect(self.moveUp.emit)
            self.btn_dn.clicked.connect(self.moveDown.emit)

    def _on_clear_mask(self):
        # select the explicit "(none)" entry
        self.mask_combo.setCurrentIndex(0)
        self._emit()

    def _emit(self, *_):
        self.changed.emit()

    def params(self):
        return {
            "visible": self.chk.isChecked(),
            "mode": self.mode.currentText(),
            "opacity": self.sld.value() / 100.0,
            "name": self._name,
            # mask UI state
            "mask_index": self.mask_combo.currentIndex(),
            "mask_src": self.mask_src.currentText(),
            "mask_invert": self.mask_invert.isChecked(),
        }

    def setName(self, name: str):
        self._name = name
        self.lbl.setText(name)

# ---------- The Dock ----------
class LayersDock(QDockWidget):
    def __init__(self, main_window):
        super().__init__("Layers", main_window)
        self.setObjectName("LayersDock")
        self.mw = main_window
        self.docman = main_window.docman
        self._wired_title_sources = set()

        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.timeout.connect(self._apply_list_to_view)
        self._apply_debounce_ms = 100  # tweak 60–150ms as you like

        # UI
        w = QWidget()
        v = QVBoxLayout(w); v.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout(); v.addLayout(top)
        top.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        top.addWidget(self.view_combo, 1)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setAlternatingRowColors(True)
        v.addWidget(self.list, 1)

        # buttons
        row = QHBoxLayout(); v.addLayout(row)
        self.btn_clear = QPushButton("Clear All Layers")
        self.btn_merge = QPushButton("Merge Layers and Push to View")
        self.btn_merge.setToolTip("Flatten the visible layers into the current view and add an undo step.")
        row.addWidget(self.btn_merge)
        row.addStretch(1)
        row.addWidget(self.btn_clear)

        self.setWidget(w)

        # dnd (accept drops from views)
        self.setAcceptDrops(True)

        # signals
        self.view_combo.currentIndexChanged.connect(self._on_pick_view)
        self.btn_clear.clicked.connect(self._clear_layers)

        # keep in sync with MDI/windows
        self.mw.mdi.subWindowActivated.connect(lambda _sw: self._refresh_views())
        self.docman.documentAdded.connect(lambda _d: self._refresh_views())
        self.docman.documentRemoved.connect(lambda _d: self._refresh_views())

        self.btn_merge.clicked.connect(self._merge_and_push)

        # initial
        self._refresh_views()

    # ---------- helpers ----------
    def _mask_choices(self):
        out = []
        for sw in self._all_subwindows():
            title = sw._effective_title() or "Untitled"
            out.append((title, sw.document))
        return out

    def _all_subwindows(self):
        from pro.subwindow import ImageSubWindow
        subs = []
        for sw in self.mw.mdi.subWindowList():
            w = sw.widget()
            if isinstance(w, ImageSubWindow):
                subs.append(w)
        return subs

    def _refresh_views(self):
        subs = self._all_subwindows()
        current = self.current_view()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        for w in subs:
            title = w._effective_title() or "Untitled"
            self.view_combo.addItem(title, userData=w)
        self.view_combo.blockSignals(False)

        if current and current in subs:
            idx = subs.index(current)
            self.view_combo.setCurrentIndex(idx)
        elif subs:
            self.view_combo.setCurrentIndex(0)

        # NEW: listen for future title changes
        self._wire_title_change_listeners(subs)

        self._rebuild_list()


    def _wire_title_change_listeners(self, subs):
        # connect once per subwindow
        for sw in subs:
            if sw in self._wired_title_sources:
                continue
            if hasattr(sw, "viewTitleChanged"):
                try:
                    sw.viewTitleChanged.connect(lambda *_: self._refresh_titles_only())
                except Exception:
                    pass
            self._wired_title_sources.add(sw)

    def _refresh_titles_only(self):
        """Update just the titles in the View dropdown, mask source lists,
        and base-row label, preserving current selection and layer state."""
        subs = self._all_subwindows()
        if not subs:
            return

        # Update the View dropdown text in place
        self.view_combo.blockSignals(True)
        cur_idx = self.view_combo.currentIndex()
        for i, sw in enumerate(subs):
            t = sw._effective_title() or "Untitled"
            if i < self.view_combo.count():
                self.view_combo.setItemText(i, t)
            else:
                self.view_combo.addItem(t, userData=sw)
        self.view_combo.blockSignals(False)
        if 0 <= cur_idx < self.view_combo.count():
            self.view_combo.setCurrentIndex(cur_idx)

        # Update mask choices shown in each row (titles only)
        choices = [(sw._effective_title() or "Untitled", sw.document) for sw in subs]
        docs = [d for _, d in choices]

        for i in range(self.list.count()):
            roww = self.list.itemWidget(self.list.item(i))
            if not isinstance(roww, _LayerRow):
                continue

            # base row label
            if getattr(roww, "_is_base", False):
                vw = self.current_view()
                base_name = vw._effective_title() if (vw and hasattr(vw, "_effective_title")) else "Current View"
                roww.setName(f"Base • {base_name}")
                continue

            # non-base row: update mask combo item texts without changing selection
            if roww.mask_combo.count() > 0:
                # index 0 is "(none)"
                # build a map from doc -> title
                title_for_doc = {doc: title for title, doc in choices}
                for idx in range(1, roww.mask_combo.count()):
                    doc = roww.mask_combo.itemData(idx)
                    if doc in title_for_doc:
                        roww.mask_combo.setItemText(idx, title_for_doc[doc])

    def current_view(self):
        idx = self.view_combo.currentIndex()
        if idx < 0:
            return None
        return self.view_combo.itemData(idx)

    def _on_pick_view(self, _i):
        self._rebuild_list()

    def _rebuild_list(self):
        self.list.clear()
        vw = self.current_view()
        if not vw:
            return

        choices = self._mask_choices()
        docs = [d for _, d in choices]

        for lyr in getattr(vw, "_layers", []):
            raw_name = getattr(lyr, "name", "Layer")
            name = raw_name if isinstance(raw_name, str) else str(raw_name)

            # --- Optional dynamic title sync ---
            try:
                src_doc = getattr(lyr, "src_doc", None)
                # What the document considers its "base" display name
                doc_disp = None
                if src_doc is not None:
                    dn = getattr(src_doc, "display_name", None)
                    doc_disp = dn() if callable(dn) else dn

                # If our stored name is just the base doc name, prefer the current view title
                if src_doc is not None and name == (doc_disp or name):
                    for sw in self._all_subwindows():
                        if getattr(sw, "document", None) is src_doc:
                            t = getattr(sw, "_effective_title", None)
                            if callable(t):
                                t = t()
                            if t:
                                name = t
                            break
            except Exception:
                pass
            mode = getattr(lyr, "mode", "Normal")
            opacity = float(getattr(lyr, "opacity", 1.0))
            visible = bool(getattr(lyr, "visible", True))
            roww = _LayerRow(name, mode, opacity, visible)
            roww.mask_combo.blockSignals(True)
            roww.mask_combo.clear()
            roww.mask_combo.addItem("(none)", userData=None)
            for title, doc in choices:
                roww.mask_combo.addItem(title, userData=doc)
            if getattr(lyr, "mask_doc", None) in docs:
                roww.mask_combo.setCurrentIndex(1 + docs.index(lyr.mask_doc))
            else:
                roww.mask_combo.setCurrentIndex(0)
            roww.mask_src.setCurrentIndex(1 if getattr(lyr, "mask_use_luma", False) else 0)
            roww.mask_invert.setChecked(bool(getattr(lyr, "mask_invert", False)))
            roww.mask_combo.blockSignals(False)
            self._bind_row(roww)
            it = QListWidgetItem(self.list)
            it.setSizeHint(roww.sizeHint())
            self.list.addItem(it)
            self.list.setItemWidget(it, roww)

        base_name = getattr(vw, "_effective_title", None)
        base_name = base_name() if callable(base_name) else "Current View"
        base_label = f"Base • {base_name}"
        base_row = _LayerRow(base_label, "—", 1.0, True, is_base=True)
        itb = QListWidgetItem(self.list)
        itb.setSizeHint(base_row.sizeHint())
        self.list.addItem(itb)
        self.list.setItemWidget(itb, base_row)
        has_layers = bool(getattr(vw, "_layers", []))
        self.btn_merge.setEnabled(has_layers)
        self.btn_clear.setEnabled(has_layers)

    def _layer_count(self) -> int:
        vw = self.current_view()
        return len(getattr(vw, "_layers", [])) if vw else 0

    def _bind_row(self, roww: _LayerRow):
        if getattr(roww, "_is_base", False):
            return
        roww.changed.connect(self._apply_list_to_view_debounced)

        roww.requestDelete.connect(lambda: self._delete_row(roww))
        roww.moveUp.connect(lambda: self._move_row(roww, -1))
        roww.moveDown.connect(lambda: self._move_row(roww, +1))

    def _apply_list_to_view_debounced(self):
        # restart the timer on every slider tick
        self._apply_timer.start(self._apply_debounce_ms)


    def _find_row_index(self, roww: _LayerRow) -> int:
        for i in range(self.list.count()):
            if self.list.itemWidget(self.list.item(i)) is roww:
                return i
        return -1

    def _delete_row(self, roww: _LayerRow):
        vw = self.current_view()
        if not vw:
            return
        idx = self._find_row_index(roww)
        if idx < 0:
            return
        if idx >= self._layer_count():
            return
        vw._layers.pop(idx)
        self.list.takeItem(idx)
        self._apply_list_to_view()

    def _move_row(self, roww: _LayerRow, delta: int):
        vw = self.current_view()
        if not vw:
            return
        i = self._find_row_index(roww)
        if i < 0 or i >= self._layer_count():
            return
        j = i + delta
        if j < 0 or j >= self._layer_count():
            return
        vw._layers[i], vw._layers[j] = vw._layers[j], vw._layers[i]
        self._rebuild_list()
        self._apply_list_to_view()

    def _apply_list_to_view(self):
        vw = self.current_view()
        if not vw:
            return
        n = self._layer_count()
        rows = []
        for i in range(n):
            it = self.list.item(i)
            rows.append(self.list.itemWidget(it))

        for lyr, roww in zip(vw._layers, rows):
            p = roww.params()
            lyr.visible = p["visible"]
            lyr.mode = p["mode"]
            lyr.opacity = float(p["opacity"])
            mi = p["mask_index"]
            if mi is not None and mi > 0:
                doc = roww.mask_combo.itemData(mi)
                lyr.mask_doc = doc
            else:
                lyr.mask_doc = None
            lyr.mask_use_luma = (p["mask_src"] == "Luminance")
            lyr.mask_invert = bool(p["mask_invert"])
        vw._reinstall_layer_watchers()
        vw.apply_layer_stack(vw._layers)

    def _clear_layers(self):
        vw = self.current_view()
        if not vw: return
        vw._layers = []
        vw._reinstall_layer_watchers()
        self._rebuild_list()
        vw.apply_layer_stack([])

    def dragEnterEvent(self, e: QDragEnterEvent):
        md = e.mimeData()
        if md.hasFormat(MIME_VIEWSTATE) or md.hasFormat(MIME_MASK):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QDragEnterEvent):
        self.dragEnterEvent(e)

    def dropEvent(self, e: QDropEvent):
        vw = self.current_view()
        if not vw:
            e.ignore(); return
        md = e.mimeData()
        try:
            if md.hasFormat(MIME_VIEWSTATE):
                st = json.loads(bytes(md.data(MIME_VIEWSTATE)).decode("utf-8"))
                doc_ptr = st.get("doc_ptr")
                if doc_ptr is None:
                    raise RuntimeError("Missing doc_ptr in MIME_VIEWSTATE payload")
                src_doc = self._resolve_doc_ptr(doc_ptr)
                if src_doc is None:
                    raise RuntimeError("Source doc gone")
                layer_name = "Layer"
                src_title = None
                for sw in self._all_subwindows():
                    if getattr(sw, "document", None) is src_doc:
                        t = getattr(sw, "_effective_title", None)
                        if callable(t):
                            t = t()
                        src_title = t or None
                        break

                if src_title:
                    layer_name = src_title
                else:
                    # fallback to document display name
                    dn = getattr(src_doc, "display_name", None)
                    layer_name = dn() if callable(dn) else (dn or "Layer")
                new_layer = ImageLayer(
                    name=layer_name,
                    src_doc=src_doc,
                    visible=True,
                    opacity=1.0,
                    mode="Normal",
                )
                if not hasattr(vw, "_layers") or vw._layers is None:
                    vw._layers = []
                vw._layers.insert(0, new_layer)
                vw._reinstall_layer_watchers()
                self._rebuild_list()
                vw.apply_layer_stack(vw._layers)
                e.acceptProposedAction()
                return

            if md.hasFormat(MIME_MASK):
                payload = json.loads(bytes(md.data(MIME_MASK)).decode("utf-8"))
                doc_ptr = payload.get("mask_doc_ptr")
                if doc_ptr is None:
                    raise RuntimeError("Missing mask_doc_ptr in MIME_MASK payload")
                mask_doc = self._resolve_doc_ptr(doc_ptr)
                if mask_doc is None:
                    raise RuntimeError("Mask doc gone")
                if not getattr(vw, "_layers", None):
                    QMessageBox.information(self, "No Layers", "Add a layer first, then drop a mask onto it.")
                    e.ignore(); return
                sel_row = self.list.currentRow()
                if sel_row < 0:
                    sel_row = 0
                idx = min(sel_row, len(vw._layers) - 1)
                layer = vw._layers[idx]
                layer.mask_doc = mask_doc
                layer.mask_invert = bool(payload.get("invert", False))
                try:
                    layer.mask_feather = float(payload.get("feather", 0.0) or 0.0)
                except Exception:
                    layer.mask_feather = 0.0
                vw._reinstall_layer_watchers()
                self._rebuild_list()
                vw.apply_layer_stack(vw._layers)
                e.acceptProposedAction()
                return
        except Exception as ex:
            print("[LayersDock] drop error:", ex)
        e.ignore()

    def _resolve_doc_ptr(self, ptr: int):
        for d in self.docman.all_documents():
            if id(d) == ptr:
                return d
        return None

    def _merge_and_push(self):
        vw = self.current_view()
        if not vw:
            return

        # No layers? Nothing to do.
        layers = list(getattr(vw, "_layers", []) or [])
        if not layers:
            QMessageBox.information(self, "Layers", "There are no layers to merge.")
            return

        try:
            # Base image from the current view's document
            base_doc = getattr(vw, "document", None)
            if base_doc is None or getattr(base_doc, "image", None) is None:
                QMessageBox.warning(self, "Layers", "No base image available for this view.")
                return

            base_img = base_doc.image
            merged = composite_stack(base_img, layers)
            if merged is None:
                QMessageBox.warning(self, "Layers", "Composite failed (empty result).")
                return

            # Push into the document as an undoable edit
            # (assumes document.apply_edit accepts float [0..1] or handles dtype internally)
            meta = dict(getattr(base_doc, "metadata", {}) or {})
            meta["step_name"] = "Layers Merge"
            base_doc.apply_edit(merged.copy(), metadata=meta, step_name="Layers Merge")

            # Clear layers and update live preview
            vw._layers = []
            vw._reinstall_layer_watchers()
            self._rebuild_list()
            vw.apply_layer_stack([])

            # Nice confirmation
            QMessageBox.information(self, "Layers",
                                    "Merged visible layers and pushed the result to the current view.")
        except Exception as ex:
            print("[LayersDock] merge error:", ex)
            QMessageBox.critical(self, "Layers", f"Merge failed:\n{ex}")

