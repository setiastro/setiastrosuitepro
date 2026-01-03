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

from setiastro.saspro.dnd_mime import MIME_VIEWSTATE, MIME_MASK
from setiastro.saspro.layers import composite_stack, ImageLayer, BLEND_MODES

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
        self.mask_invert = QCheckBox("Invert")
        self.btn_clear_mask = QPushButton("Clear")
        self.btn_clear_mask.setFixedWidth(52)
        r2.addWidget(QLabel("Mask")); r2.addWidget(self.mask_combo, 1)
        r2.addWidget(self.mask_invert); r2.addWidget(self.btn_clear_mask)

        # Extra controls for some blend modes (e.g. Sigmoid)
        self.sig_center_label = None
        self.sig_center = None
        self.sig_strength_label = None
        self.sig_strength = None

        if not self._is_base:
            # row 3: Sigmoid parameters
            r3 = QHBoxLayout(); v.addLayout(r3)

            self.sig_center_label = QLabel("Sigmoid center")
            from PyQt6.QtWidgets import QDoubleSpinBox
            self.sig_center = QDoubleSpinBox()
            self.sig_center.setRange(0.0, 1.0)
            self.sig_center.setSingleStep(0.01)
            self.sig_center.setDecimals(3)
            self.sig_center.setValue(0.5)

            self.sig_strength_label = QLabel("Strength")
            self.sig_strength = QDoubleSpinBox()
            self.sig_strength.setRange(0.1, 50.0)
            self.sig_strength.setSingleStep(0.5)
            self.sig_strength.setDecimals(2)
            self.sig_strength.setValue(10.0)

            r3.addWidget(self.sig_center_label)
            r3.addWidget(self.sig_center)
            r3.addWidget(self.sig_strength_label)
            r3.addWidget(self.sig_strength)
            r3.addStretch(1)

        if self._is_base:
            # Base row is informational only
            for w in (self.chk, self.mode, self.sld, self.btn_up, self.btn_dn, self.btn_x,
                      self.mask_combo, self.mask_invert, self.btn_clear_mask):
                w.setEnabled(False)
            self.lbl.setStyleSheet("color: palette(mid);")
        else:
            self.chk.stateChanged.connect(self._emit)
            self.mode.currentIndexChanged.connect(self._on_mode_changed)
            self.sld.valueChanged.connect(self._emit)
            self.mask_combo.currentIndexChanged.connect(self._emit)
            self.mask_invert.stateChanged.connect(self._emit)
            self.btn_clear_mask.clicked.connect(self._on_clear_mask)
            self.btn_x.clicked.connect(self.requestDelete.emit)
            self.btn_up.clicked.connect(self.moveUp.emit)
            self.btn_dn.clicked.connect(self.moveDown.emit)

            # Sigmoid controls emit change + only show for Sigmoid mode
            if self.sig_center is not None:
                self.sig_center.valueChanged.connect(self._emit)
            if self.sig_strength is not None:
                self.sig_strength.valueChanged.connect(self._emit)

            self.mode.currentIndexChanged.connect(
                lambda _i: self._update_extra_controls(self.mode.currentText())
            )
            # Initial visibility
            self._update_extra_controls(self.mode.currentText())

    def _on_mode_changed(self, _idx: int):
        # Update which extra controls are visible
        self._update_extra_controls(self.mode.currentText())
        # Make our layout recompute height
        lay = self.layout()
        if lay is not None:
            lay.invalidate()
            lay.activate()

        self.adjustSize()
        self.updateGeometry()
        # Tell the dock “something changed”
        self._emit()

    def _update_extra_controls(self, mode_text: str):
        is_sig = (mode_text == "Sigmoid")
        for w in (self.sig_center_label, self.sig_center,
                  self.sig_strength_label, self.sig_strength):
            if w is not None:
                w.setVisible(is_sig)


    def _update_extra_controls(self, mode_text: str):
        is_sig = (mode_text == "Sigmoid")
        for w in (self.sig_center_label, self.sig_center,
                  self.sig_strength_label, self.sig_strength):
            if w is not None:
                w.setVisible(is_sig)

        # Let the layout recompute our preferred height
        self.adjustSize()
        self.updateGeometry()

    def set_sigmoid_params(self, center: float, strength: float):
        if self.sig_center is None or self.sig_strength is None:
            return
        self.sig_center.blockSignals(True)
        self.sig_strength.blockSignals(True)
        self.sig_center.setValue(float(center))
        self.sig_strength.setValue(float(strength))
        self.sig_center.blockSignals(False)
        self.sig_strength.blockSignals(False)
        self._update_extra_controls(self.mode.currentText())


    def _on_clear_mask(self):
        # select the explicit "(none)" entry
        self.mask_combo.setCurrentIndex(0)
        self._emit()

    def _emit(self, *_):
        self.changed.emit()

    def params(self):
        out = {
            "visible": self.chk.isChecked(),
            "mode": self.mode.currentText(),
            "opacity": self.sld.value() / 100.0,
            "name": self._name,
            # mask UI state
            "mask_index": self.mask_combo.currentIndex(),
            "mask_src": "Luminance",
            "mask_invert": self.mask_invert.isChecked(),
        }
        if self.sig_center is not None and self.sig_strength is not None:
            out["sigmoid_center"] = self.sig_center.value()
            out["sigmoid_strength"] = self.sig_strength.value()
        return out

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
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setAlternatingRowColors(True)
        v.addWidget(self.list, 1)

        # buttons
        row = QHBoxLayout(); v.addLayout(row)

        self.btn_clear = QPushButton("Clear All Layers")

        self.btn_merge = QPushButton("Merge → Push to View")
        self.btn_merge.setToolTip("Flatten the visible layers into the current view and add an undo step.")

        self.btn_merge_new = QPushButton("Merge → New Document")
        self.btn_merge_new.setToolTip("Flatten the visible layers into a new document (does not modify the base view).")

        self.btn_merge_sel = QPushButton("Merge Selected → Single Layer")
        self.btn_merge_sel.setToolTip("Merge the selected layers into one raster layer (Photoshop-style).")

        row.addWidget(self.btn_merge)
        row.addWidget(self.btn_merge_new)
        row.addWidget(self.btn_merge_sel)
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
        self.btn_merge_new.clicked.connect(self._merge_to_new_doc)
        self.btn_merge_sel.clicked.connect(self._merge_selected_to_single_layer)

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
        from setiastro.saspro.subwindow import ImageSubWindow
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
            
            roww.mask_invert.setChecked(bool(getattr(lyr, "mask_invert", False)))
            roww.mask_combo.blockSignals(False)
            center = getattr(lyr, "sigmoid_center", 0.5)
            strength = getattr(lyr, "sigmoid_strength", 10.0)
            roww.set_sigmoid_params(center, strength)            
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
        if hasattr(self, "btn_merge_new"):
            self.btn_merge_new.setEnabled(has_layers)    
        has_layers = bool(getattr(vw, "_layers", []))
        self.btn_merge_sel.setEnabled(has_layers)
        self._refresh_row_heights()

    def _selected_layer_indices(self) -> list[int]:
        vw = self.current_view()
        if not vw:
            return []
        n = len(getattr(vw, "_layers", []) or [])
        idxs = []
        for it in self.list.selectedItems():
            r = self.list.row(it)
            if 0 <= r < n:
                idxs.append(r)
        idxs = sorted(set(idxs))
        return idxs

    def _render_stack(self, base_img: np.ndarray, layers: list[ImageLayer]) -> np.ndarray:
        # composite_stack already respects visibility/opacity/modes/masks
        out = composite_stack(base_img, layers)
        return out if out is not None else base_img

    def _open_baked_layer_doc(self, base_doc, arr: np.ndarray, title: str):
        dm = getattr(self.mw, "docman", None)
        if not dm or not hasattr(dm, "open_array"):
            return None
        meta = dict(getattr(base_doc, "metadata", {}) or {})
        meta.update({
            "bit_depth": "32-bit floating point",
            "is_mono": (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)),
            "source": "Layers Merge Selected",
        })
        return dm.open_array(arr.astype(np.float32, copy=False), metadata=meta, title=title)

    def _merge_selected_to_single_layer(self):
        vw = self.current_view()
        if not vw:
            return

        layers = list(getattr(vw, "_layers", []) or [])
        if not layers:
            QMessageBox.information(self, "Layers", "There are no layers to merge.")
            return

        sel = self._selected_layer_indices()
        if len(sel) < 2:
            QMessageBox.information(self, "Layers", "Select two or more layers to merge.")
            return

        try:
            base_doc = getattr(vw, "document", None)
            if base_doc is None or getattr(base_doc, "image", None) is None:
                QMessageBox.warning(self, "Layers", "No base image available for this view.")
                return
            base_img = base_doc.image

            i0, i1 = sel[0], sel[-1]

            # IMPORTANT ASSUMPTION (matches your UI order):
            # vw._layers is top-to-bottom in the list, and "below" means larger index.
            layers_above = layers[:i0]
            layers_sel   = layers[i0:i1+1]
            layers_below = layers[i1+1:]

            # 1) Render what exists directly under the selected range
            under = self._render_stack(base_img, layers_below)

            # 2) Render selected layers on top of that "under" image
            baked = self._render_stack(under, layers_sel)


            # 3) Create a baked raster layer (NO new document)
            merged_layer = ImageLayer(
                name=f"Merged ({len(layers_sel)})",
                src_doc=None,
                pixels=baked.astype(np.float32, copy=False),
                visible=True,
                opacity=1.0,
                mode="Normal",
            )

            # Keep masks off by default; you can also decide to inherit the topmost mask
            merged_layer.mask_doc = None
            merged_layer.mask_use_luma = True
            merged_layer.mask_invert = False

            new_layers = layers_above + [merged_layer] + layers_below
            vw._layers = new_layers

            vw._reinstall_layer_watchers()
            self._rebuild_list()
            vw.apply_layer_stack(vw._layers)

            QMessageBox.information(self, "Layers", f"Merged {len(layers_sel)} layers into a single layer.")
        except Exception as ex:
            print("[LayersDock] merge_selected error:", ex)
            QMessageBox.critical(self, "Layers", f"Merge Selected failed:\n{ex}")


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
        # Also refresh row heights so mode-dependent controls (like Sigmoid)
        # can expand/collapse the row visually.
        self._refresh_row_heights()

    def _refresh_row_heights(self):
        """Update QListWidgetItem size hints to match current row widgets."""
        try:
            for i in range(self.list.count()):
                item = self.list.item(i)
                roww = self.list.itemWidget(item)
                if roww is not None:
                    # Ask the row for an up-to-date size hint
                    item.setSizeHint(roww.sizeHint())
        except Exception as ex:
            print("[LayersDock] _refresh_row_heights error:", ex)



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
            # Sigmoid parameters (if present)
            if "sigmoid_center" in p:
                lyr.sigmoid_center = float(p["sigmoid_center"])
            if "sigmoid_strength" in p:
                lyr.sigmoid_strength = float(p["sigmoid_strength"])
            mi = p["mask_index"]
            if mi is not None and mi > 0:
                doc = roww.mask_combo.itemData(mi)
                lyr.mask_doc = doc
            else:
                lyr.mask_doc = None

            # Force luminance masks only
            lyr.mask_use_luma = True
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
                # Try robust resolution (UIDs/file_path/ptr)
                src_doc = self._resolve_doc_from_state(st)
                if src_doc is None:
                    raise RuntimeError("Source doc gone")
                layer_name = "Layer"
                src_title = None
                for sw in self._all_subwindows():
                    if getattr(sw, "document", None) is src_doc:
                        t = getattr(sw, "_effective_title", None)
                        src_title = t() if callable(t) else t
                        break
                if src_title:
                    layer_name = src_title
                else:
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
                # payload may include doc_uid/base_doc_uid/file_path/mask_doc_ptr
                mask_doc = self._resolve_doc_from_state(payload)
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
        """Legacy path: resolve by Python id() pointer."""
        try:
            for d in self.docman.all_documents():
                if id(d) == ptr:
                    return d
        except Exception:
            pass
        return None

    def _resolve_doc_from_state(self, st):
        """
        Accepts either:
        - dict payload (preferred): may include doc_uid, base_doc_uid, file_path, doc_ptr/mask_doc_ptr
        - int legacy pointer
        Tries, in order: doc_uid → base_doc_uid → legacy ptr → file_path.
        """
        # If called with an int, treat it as a raw pointer
        if isinstance(st, int):
            return self._resolve_doc_ptr(st)

        if not isinstance(st, dict):
            return None

        # 1) Prefer UIDs
        doc_uid = st.get("doc_uid")
        base_uid = st.get("base_doc_uid")
        if doc_uid and hasattr(self.docman, "get_document_by_uid"):
            d = self.docman.get_document_by_uid(doc_uid)
            if d is not None:
                return d
        if base_uid and hasattr(self.docman, "get_document_by_uid"):
            d = self.docman.get_document_by_uid(base_uid)
            if d is not None:
                return d

        # 2) Legacy pointer
        ptr = st.get("doc_ptr") or st.get("mask_doc_ptr")  # mask payloads may use mask_doc_ptr
        if isinstance(ptr, int):
            d = self._resolve_doc_ptr(ptr)
            if d is not None:
                return d

        # 3) Last-ditch: file path match
        fp = (st.get("file_path") or "").strip()
        if fp:
            try:
                for d in self.docman.all_documents():
                    meta = getattr(d, "metadata", {}) or {}
                    if meta.get("file_path") == fp:
                        return d
            except Exception:
                pass

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

    def _merge_to_new_doc(self):
        vw = self.current_view()
        if not vw:
            return

        layers = list(getattr(vw, "_layers", []) or [])
        if not layers:
            QMessageBox.information(self, "Layers", "There are no layers to merge.")
            return

        try:
            base_doc = getattr(vw, "document", None)
            if base_doc is None or getattr(base_doc, "image", None) is None:
                QMessageBox.warning(self, "Layers", "No base image available for this view.")
                return

            base_img = base_doc.image
            merged = composite_stack(base_img, layers)
            if merged is None:
                QMessageBox.warning(self, "Layers", "Composite failed (empty result).")
                return

            # Push as a new document (same pattern as stars-only)
            self._push_merged_as_new_doc(base_doc, merged)

            QMessageBox.information(self, "Layers",
                                    "Merged visible layers and created a new document.")
        except Exception as ex:
            print("[LayersDock] merge_to_new_doc error:", ex)
            QMessageBox.critical(self, "Layers", f"Merge failed:\n{ex}")

    def _push_merged_as_new_doc(self, base_doc, arr: np.ndarray):
        dm = getattr(self.mw, "docman", None)
        if not dm or not hasattr(dm, "open_array"):
            return

        # Derive a friendly title based on the *view title* if possible
        title = None
        try:
            # Use current view title (respects per-view rename)
            vw = self.current_view()
            if vw and hasattr(vw, "_effective_title"):
                base = (vw._effective_title() or "").strip()
            else:
                base = ""

            if not base:
                dn = getattr(base_doc, "display_name", None)
                base = dn() if callable(dn) else (dn or "Untitled")

            suffix = "_merged"
            title = base if base.endswith(suffix) else f"{base}{suffix}"
        except Exception:
            title = "Merged Layers"

        try:
            meta = dict(getattr(base_doc, "metadata", {}) or {})
            meta.update({
                "bit_depth": "32-bit floating point",
                "is_mono": (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 1)),
                "source": "Layers Merge",
                "step_name": "Layers Merge",
            })

            newdoc = dm.open_array(arr.astype(np.float32, copy=False), metadata=meta, title=title)
            if hasattr(self.mw, "_spawn_subwindow_for"):
                self.mw._spawn_subwindow_for(newdoc)
        except Exception as ex:
            print("[LayersDock] _push_merged_as_new_doc error:", ex)
