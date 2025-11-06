# pro/subwindow.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QSize, QEvent, QByteArray, QMimeData, QSettings, QTimer, QRect, QPoint, QMargins
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QToolButton, QHBoxLayout, QMessageBox, QMdiSubWindow, QMenu, QInputDialog, QApplication, QTabWidget, QRubberBand
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QShortcut, QKeySequence, QCursor, QDrag, QGuiApplication
from PyQt6 import sip
import numpy as np
import json
import math
import os


from .autostretch import autostretch   # ← uses pro/imageops/stretch.py

from pro.dnd_mime import MIME_VIEWSTATE, MIME_MASK, MIME_ASTROMETRY, MIME_CMD 
from pro.shortcuts import _unpack_cmd_payload

from .layers import composite_stack, ImageLayer, BLEND_MODES

# --- NEW: simple table model for TableDocument ---
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant

__all__ = ["ImageSubWindow", "TableSubWindow"]

class SimpleTableModel(QAbstractTableModel):
    def __init__(self, rows: list[list], headers: list[str], parent=None):
        super().__init__(parent)
        self._rows = rows
        self._headers = headers

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else (len(self._headers) if self._headers else (len(self._rows[0]) if self._rows else 0))

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            try:
                return str(self._rows[index.row()][index.column()])
            except Exception:
                return ""
        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            try:
                return self._headers[section] if self._headers and 0 <= section < len(self._headers) else f"C{section+1}"
            except Exception:
                return f"C{section+1}"
        else:
            return str(section + 1)


class _DragTab(QLabel):
    """
    Little grab tab you can drag to copy/sync view state.
    - Drag onto MDI background → duplicate view (same document)
    - Drag onto another subwindow → copy zoom/pan/stretch to that view
    """
    def __init__(self, owner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owner = owner
        self._press_pos = None
        self.setText("⧉")
        self.setToolTip(
            "Drag to duplicate/copy view.\n"
            "Hold Shift while dragging to drop this image as a mask onto another view.\n"
            "Hold Ctrl while dragging to copy the astrometric solution (WCS) to another view."
        )

        self.setFixedSize(22, 18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet(
            "QLabel{background:rgba(255,255,255,30); "
            "border:1px solid rgba(255,255,255,60); border-radius:4px;}"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)


    def mouseMoveEvent(self, ev):
        if self._press_pos is None:
            return
        if (ev.position() - self._press_pos).manhattanLength() > 6:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._press_pos = None
            if (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier):
                self.owner._start_mask_drag()
            elif (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier):
                self.owner._start_astrometry_drag()   # ← NEW
            else:
                self.owner._start_viewstate_drag()

    def mouseReleaseEvent(self, ev):
        self._press_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

MASK_GLYPH = "■"
#ACTIVE_PREFIX = "Active View: "
ACTIVE_PREFIX = ""
GLYPHS = "■●◆▲▪▫•◼◻◾◽"

class ImageSubWindow(QWidget):
    aboutToClose = pyqtSignal(object)
    autostretchChanged = pyqtSignal(bool)
    requestDuplicate = pyqtSignal(object)  # document
    layers_changed = pyqtSignal() 
    autostretchProfileChanged = pyqtSignal(str)
    viewTitleChanged = pyqtSignal(object, str)
    activeSourceChanged = pyqtSignal(object)  # None for full, or (x,y,w,h) for ROI


    def __init__(self, document, parent=None):
        super().__init__(parent)
        self._base_document = None
        self.document = document
        self._last_title_for_emit = None

        # ─────────────────────────────────────────────────────────
        # View / render state
        # ─────────────────────────────────────────────────────────
        self._min_scale = 0.02
        self._max_scale = 3.00  # 300%
        self.scale = 0.25
        self._dragging = False
        self._drag_start = QPoint()
        self._autostretch_linked = QSettings().value("display/stretch_linked", False, type=bool)
        self.autostretch_enabled = False
        self.autostretch_target = 0.25
        self.autostretch_sigma = 3.0
        self.autostretch_profile = "normal"
        self.show_mask_overlay = False
        self._mask_overlay_alpha = 0.5   # 0..1
        self._mask_overlay_invert = True
        self._layers: list[ImageLayer] = []
        self.layers_changed.connect(lambda: None)
        self._display_override: np.ndarray | None = None
        self._readout_hint_shown = False

        # pixel readout live-probe state
        self._space_down = False
        self._readout_dragging = False
        self._last_readout = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Title (doc/view) sync
        self._view_title_override = None
        self.document.changed.connect(self._sync_host_title)
        self._sync_host_title()
        self.document.changed.connect(self._refresh_local_undo_buttons)

        # Cached display buffer
        self._buf8 = None         # backing np.uint8 [H,W,3]
        self._qimg_src = None     # QImage wrapping _buf8

        # Keep mask visuals in sync when doc changes
        self.document.changed.connect(self._on_doc_mask_changed)

        # ─────────────────────────────────────────────────────────
        # Preview tabs state
        # ─────────────────────────────────────────────────────────
        self._tabs: QTabWidget | None = None
        self._previews: list[dict] = []  # {"id": int, "name": str, "roi": (x,y,w,h), "arr": np.ndarray}
        self._active_source_kind = "full"  # "full" | "preview"
        self._active_preview_id: int | None = None
        self._next_preview_id = 1

        # Rubber-band / selection for previews
        self._preview_select_mode = False
        self._rubber: QRubberBand | None = None
        self._rubber_origin: QPoint | None = None

        # ─────────────────────────────────────────────────────────
        # UI construction
        # ─────────────────────────────────────────────────────────
        lyt = QVBoxLayout(self)

        # Top row: drag-tab + Preview button
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._drag_tab = _DragTab(self)
        row.addWidget(self._drag_tab, 0, Qt.AlignmentFlag.AlignLeft)

        self._preview_btn = QToolButton(self)
        self._preview_btn.setText("⟂")  # crosshair glyph
        self._preview_btn.setToolTip("Create Preview: click, then drag on the image to define a preview rectangle.")
        self._preview_btn.setCheckable(True)
        self._preview_btn.clicked.connect(self._toggle_preview_select_mode)
        row.addWidget(self._preview_btn, 0, Qt.AlignmentFlag.AlignLeft)
        # — Undo / Redo just for this subwindow —
        self._btn_undo = QToolButton(self)
        self._btn_undo.setText("↶")  # or use an icon
        self._btn_undo.setToolTip("Undo (this view)")
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._on_local_undo)
        row.addWidget(self._btn_undo, 0, Qt.AlignmentFlag.AlignLeft)

        self._btn_redo = QToolButton(self)
        self._btn_redo.setText("↷")
        self._btn_redo.setToolTip("Redo (this view)")
        self._btn_redo.setEnabled(False)
        self._btn_redo.clicked.connect(self._on_local_redo)
        row.addWidget(self._btn_redo, 0, Qt.AlignmentFlag.AlignLeft)
        row.addStretch(1)
        lyt.addLayout(row)

        # QTabWidget that hosts "Full" (real viewer) + any Preview tabs (placeholder widgets)
        self._tabs = QTabWidget(self)
        self._tabs.setTabsClosable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(True)

        # Build the default "Full" tab, which contains the ONE real viewer (scroll+label)
        full_host = QWidget(self)
        full_v = QVBoxLayout(full_host)
        full_v.setContentsMargins(0, 0, 0, 0)

        self.scroll = QScrollArea(full_host)
        self.scroll.setWidgetResizable(False)
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)
        self.scroll.viewport().setMouseTracking(True)
        self.label.setMouseTracking(True)        
        full_v.addWidget(self.scroll)

        # IMPORTANT: add the tab BEFORE connecting signals so currentChanged can't fire early
        self._full_tab_idx = self._tabs.addTab(full_host, "Full")
        self._full_host = full_host
        self._tabs.tabBar().setVisible(False)  # hidden until a preview exists
        lyt.addWidget(self._tabs)

        # Now it’s safe to connect
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs.currentChanged.connect(lambda _=None: self._refresh_local_undo_buttons())

        # DnD + event filters for the single viewer
        self.setAcceptDrops(True)
        self.scroll.viewport().installEventFilter(self)
        self.label.installEventFilter(self)

        # Context menu + shortcuts
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_ctx_menu)
        QShortcut(QKeySequence("F2"), self, activated=self._rename_view)
        QShortcut(QKeySequence("A"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Alt+Shift+A"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.toggle_mask_overlay)

        # Re-render when the document changes
        self.document.changed.connect(lambda: self._render(rebuild=True))
        self._render(rebuild=True)
        QTimer.singleShot(0, self._maybe_announce_readout_help)
        self._refresh_local_undo_buttons()


        # Mask/title adornments
        self._mask_dot_enabled = self._active_mask_array() is not None
        self._active_title_prefix = False
        self._rebuild_title()

        # Track docs used by layer stack (if any)
        self._watched_docs = set()
        self._history_doc = None
        self._install_history_watchers()

    def _install_history_watchers(self):
        # disconnect old
        hd = getattr(self, "_history_doc", None)
        if hd is not None and hasattr(hd, "changed"):
            try: hd.changed.disconnect(self._refresh_local_undo_buttons)
            except Exception: pass

        # resolve new history doc (ROI when on Preview tab, else base)
        new_hd = self._resolve_history_doc()
        self._history_doc = new_hd

        # connect new
        if new_hd is not None and hasattr(new_hd, "changed"):
            try: new_hd.changed.connect(self._refresh_local_undo_buttons)
            except Exception: pass

        # make the buttons correct right now
        self._refresh_local_undo_buttons()

    def _drag_identity_fields(self):
        """
        Returns a dict with identity hints for DnD:
        doc_uid (preferred), base_doc_uid (parent/full), and file_path.
        Falls back gracefully if fields are missing.
        """
        doc = getattr(self, "document", None)
        base = getattr(self, "base_document", None) or doc

        # If DocManager maps preview/ROI views, prefer the true backing doc as base
        dm = getattr(self, "_docman", None)
        try:
            if dm and hasattr(dm, "get_document_for_view"):
                back = dm.get_document_for_view(self)
                if back is not None:
                    base = back
        except Exception:
            pass

        meta = (getattr(doc, "metadata", None) or {})
        base_meta = (getattr(base, "metadata", None) or {})

        return {
            "doc_uid": getattr(doc, "uid", None),
            "base_doc_uid": getattr(base, "uid", None),
            "file_path": meta.get("file_path") or base_meta.get("file_path") or "",
        }


    def _on_local_undo(self):
        doc = self._resolve_history_doc()
        if not doc or not hasattr(doc, "undo"):
            return
        try:
            doc.undo()
            # most ImageDocument implementations emit changed; belt-and-suspenders:
            if hasattr(doc, "changed"): doc.changed.emit()
        except Exception:
            pass
        # repaint and refresh our buttons
        self._render(rebuild=True)
        self._refresh_local_undo_buttons()

    def _on_local_redo(self):
        doc = self._resolve_history_doc()
        if not doc or not hasattr(doc, "redo"):
            return
        try:
            doc.redo()
            if hasattr(doc, "changed"): doc.changed.emit()
        except Exception:
            pass
        self._render(rebuild=True)
        self._refresh_local_undo_buttons()


    def refresh_preview_roi(self, roi_tuple=None):
        """
        Rebuild the active preview pixmap from the parent document’s data.
        If roi_tuple is provided, it's the updated region (x,y,w,h).
        """
        try:
            if not (hasattr(self, "has_active_preview") and self.has_active_preview()):
                return

            # Optional: sanity check that roi matches the current preview
            if roi_tuple is not None:
                cur = self.current_preview_roi()
                if not (cur and tuple(map(int, cur)) == tuple(map(int, roi_tuple))):
                    return  # different preview; no refresh needed

            # Your own method that (re)generates the preview pixmap from the doc
            if hasattr(self, "rebuild_preview_pixmap") and callable(self.rebuild_preview_pixmap):
                self.rebuild_preview_pixmap()
            elif hasattr(self, "_update_preview_layer") and callable(self._update_preview_layer):
                self._update_preview_layer()
            else:
                # Fallback: repaint
                self.update()
        except Exception:
            pass

    def refresh_full(self):
        """Full-image redraw hook for non-ROI updates."""
        try:
            if hasattr(self, "rebuild_image_pixmap") and callable(self.rebuild_image_pixmap):
                self.rebuild_image_pixmap()
            else:
                self.update()
        except Exception:
            pass

    def refresh_preview_region(self, roi):
        """
        roi: (x,y,w,h) in FULL image coords. Rebuild the active Preview tab’s pixmap
        from self.document.image[y:y+h, x:x+w].
        """
        if not (hasattr(self, "has_active_preview") and self.has_active_preview()):
            # No preview active → fall back to full refresh
            if hasattr(self, "refresh_from_document"):
                self.refresh_from_document()
            else:
                self.update()
            return

        try:
            x, y, w, h = map(int, roi)
            arr = self.document.image[y:y+h, x:x+w]
            # Whatever your existing path is to update the preview tab from an ndarray:
            # e.g., self._set_preview_from_array(arr) or self._update_preview_pixmap(arr)
            if hasattr(self, "_set_preview_from_array"):
                self._set_preview_from_array(arr)
            elif hasattr(self, "update_preview_from_array"):
                self.update_preview_from_array(arr)
            else:
                # Fallback: full refresh if you don’t expose a thin setter
                if hasattr(self, "rebuild_active_preview"):
                    self.rebuild_active_preview()
                elif hasattr(self, "refresh_from_document"):
                    self.refresh_from_document()
            self.update()
        except Exception:
            # Safe fallback
            if hasattr(self, "rebuild_active_preview"):
                self.rebuild_active_preview()
            elif hasattr(self, "refresh_from_document"):
                self.refresh_from_document()
            else:
                self.update()


    def _ensure_tabs(self):
        if self._tabs:
            return
        self._tabs = QTabWidget(self)
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(True)

        # Build the default "Full" tab: it contains your scroll+label
        full_host = QWidget(self)
        v = QVBoxLayout(full_host)
        v.setContentsMargins(QMargins(0,0,0,0))
        # Reuse your existing scroll/label as the content of the "Full" tab
        self.scroll = QScrollArea(full_host)
        self.scroll.setWidgetResizable(False)
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)
        v.addWidget(self.scroll)
        self._full_tab_idx = self._tabs.addTab(full_host, "Full")
        self._full_host = full_host    
        self._tabs.tabBar().setVisible(False)  # hidden until a first preview exists

    def _on_tab_close_requested(self, idx: int):
        # Prevent closing "Full"
        if idx == self._full_tab_idx:
            return
        wid = self._tabs.widget(idx)
        prev_id = getattr(wid, "_preview_id", None)

        # Remove model entry
        self._previews = [p for p in self._previews if p["id"] != prev_id]
        # If you closed the active one, fall back to full
        if self._active_preview_id == prev_id:
            self._active_source_kind = "full"
            self._active_preview_id = None
            self._render(True)

        self._tabs.removeTab(idx)
        wid.deleteLater()

        # Hide tabs if no more previews
        if not self._previews:
            self._tabs.tabBar().setVisible(False)

    def _on_tab_changed(self, idx: int):
        if not hasattr(self, "_full_tab_idx"):
            return
        if idx == self._full_tab_idx:
            self._active_source_kind = "full"
            self._active_preview_id = None
            host = getattr(self, "_full_host", None) or self._tabs.widget(idx)  # ← safe
        else:
            wid = self._tabs.widget(idx)
            self._active_source_kind = "preview"
            self._active_preview_id = getattr(wid, "_preview_id", None)
            host = wid

        if host is not None:
            self._move_view_into(host)
        self._install_history_watchers()
        self._render(True)
        self._refresh_local_undo_buttons()


    def _toggle_preview_select_mode(self, on: bool):
        self._preview_select_mode = bool(on)
        self._set_preview_cursor(self._preview_select_mode)
        if self._preview_select_mode:
            mw = self._find_main_window()
            if mw and hasattr(mw, "statusBar"):
                mw.statusBar().showMessage("Preview mode: drag a rectangle on the image to create a preview.", 6000)
        else:
            self._cancel_rubber()

    def _cancel_rubber(self):
        if self._rubber is not None:
            self._rubber.hide()
            self._rubber.deleteLater()
            self._rubber = None
        self._rubber_origin = None
        self._preview_select_mode = False
        self._set_preview_cursor(False)
        if self._preview_btn.isChecked():
            self._preview_btn.setChecked(False)

    def _current_tab_host(self):
        # returns the QWidget inside the current tab
        return self._tabs.widget(self._tabs.currentIndex())

    def _move_view_into(self, host_widget: QWidget):
        """Reparent the single viewer (scroll+label) into host_widget's layout."""
        if self.scroll.parent() is host_widget:
            return
        # take it out of the old parent layout
        try:
            old_layout = self.scroll.parentWidget().layout()
            if old_layout:
                old_layout.removeWidget(self.scroll)
        except Exception:
            pass

        # ensure host has a VBox layout
        lay = host_widget.layout()
        if lay is None:
            from PyQt6.QtWidgets import QVBoxLayout
            lay = QVBoxLayout(host_widget)
            lay.setContentsMargins(0, 0, 0, 0)

        # insert viewer; kill any placeholder child labels if present
        try:
            kids = host_widget.findChildren(QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly)
        except Exception:
            kids = host_widget.findChildren(QLabel)  # recursive fallback
        for ch in list(kids):
            if ch is not self.label:
                ch.deleteLater()

        self.scroll.setParent(host_widget)
        lay.addWidget(self.scroll)
        self.scroll.show()

    def _set_preview_cursor(self, active: bool):
        cur = Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor
        for w in (self, getattr(self, "scroll", None) and self.scroll.viewport(), getattr(self, "label", None)):
            if not w:
                continue
            try:
                w.unsetCursor()          # clear any prior override
                w.setCursor(cur)         # then set desired cursor
            except Exception:
                pass


    def _maybe_announce_readout_help(self):
        """Show the readout hint only once automatically."""
        if self._readout_hint_shown:
            return
        self._announce_readout_help()
        self._readout_hint_shown = True        

    def _announce_readout_help(self):
        mw = self._find_main_window()
        if mw and hasattr(mw, "statusBar"):
            sb = mw.statusBar()
            if sb:
                sb.showMessage("Press Space + Click/Drag to probe pixels (WCS shown if available)", 8000)


      
    def apply_layer_stack(self, layers):
        """
        Rebuild the display override from base document + given layer stack.
        Does not mutate the underlying document.image.
        """
        try:
            base = self.document.image
            if layers:
                comp = composite_stack(base, layers)
                self._display_override = comp
            else:
                self._display_override = None
            self.layers_changed.emit()
            self._render(rebuild=True)
        except Exception as e:
            print("[ImageSubWindow] apply_layer_stack error:", e)      

    # --- add to ImageSubWindow ---
    def _collect_layer_docs(self):
        docs = set()
        for L in getattr(self, "_layers", []):
            d = getattr(L, "src_doc", None)
            if d is not None:
                docs.add(d)
            md = getattr(L, "mask_doc", None)
            if md is not None:
                docs.add(md)
        return docs

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            # only the first time we enter probe mode
            if not self._space_down and not self._readout_hint_shown:
                self._announce_readout_help()
                self._readout_hint_shown = True
            self._space_down = True
            ev.accept()
            return
        super().keyPressEvent(ev)



    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self._space_down = False
            # DO NOT stop _readout_dragging here – mouse release will do that
            ev.accept()
            return
        super().keyReleaseEvent(ev)



    def _sample_image_at_viewport_pos(self, vp_pos: QPoint):
        """
        vp_pos: position in viewport coords (the visible part of the scroll area).
        Returns (x_img_int, y_img_int, sample_dict) or None if OOB.
        sample_dict is always raw float(s), never normalized.
        """
        if self.document is None or self.document.image is None:
            return None

        arr = np.asarray(self.document.image)

        # detect shape
        if arr.ndim == 2:
            h, w = arr.shape
            channels = 1
        elif arr.ndim == 3:
            h, w, channels = arr.shape[:3]
        else:
            return None  # unsupported shape

        # current scroll offsets
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        x_label = hbar.value() + vp_pos.x()
        y_label = vbar.value() + vp_pos.y()

        scale = max(self.scale, 1e-12)
        x_img = x_label / scale
        y_img = y_label / scale

        xi = int(round(x_img))
        yi = int(round(y_img))

        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return None

        # ---- mono cases ----
        if arr.ndim == 2 or channels == 1:
            # pure mono or (H, W, 1)
            if arr.ndim == 2:
                val = float(arr[yi, xi])
            else:
                val = float(arr[yi, xi, 0])
            sample = {"mono": val}
            return (xi, yi, sample)

        # ---- color / 3+ channels ----
        pix = arr[yi, xi]

        # make robust if pix is 1-D
        # expect at least 3 numbers, fallback to repeating R
        r = float(pix[0])
        g = float(pix[1]) if channels > 1 else r
        b = float(pix[2]) if channels > 2 else r

        sample = {"r": r, "g": g, "b": b}
        return (xi, yi, sample)



    def sizeHint(self) -> QSize:
        lbl = getattr(self, "image_label", None) or getattr(self, "label", None)
        sa  = getattr(self, "scroll_area", None) or self.findChild(QScrollArea)
        if lbl and hasattr(lbl, "pixmap") and lbl.pixmap() and not lbl.pixmap().isNull():
            pm = lbl.pixmap()
            # logical pixels (HiDPI-safe)
            dpr  = pm.devicePixelRatioF() if hasattr(pm, "devicePixelRatioF") else 1.0
            pm_w = int(math.ceil(pm.width()  / dpr))
            pm_h = int(math.ceil(pm.height() / dpr))

            # label margins
            lm = lbl.contentsMargins()
            w = pm_w + lm.left() + lm.right()
            h = pm_h + lm.top()  + lm.bottom()

            # scrollarea chrome (frame + reserve bar thickness)
            if sa:
                fw = sa.frameWidth()
                w += fw * 2 + sa.verticalScrollBar().sizeHint().width()
                h += fw * 2 + sa.horizontalScrollBar().sizeHint().height()

            # this widget’s margins
            m = self.contentsMargins()
            w += m.left() + m.right() + 2
            h += m.top()  + m.bottom() + 20

            # tiny safety pad so bars never appear from rounding
            return QSize(w + 2, h + 8)

        return super().sizeHint()

    def _on_layer_source_changed(self):
        # Any source/mask doc changed → recomposite current stack
        try:
            self.apply_layer_stack(self._layers)
        except Exception as e:
            print("[ImageSubWindow] _on_layer_source_changed error:", e)

    def _reinstall_layer_watchers(self):
        # Disconnect old
        for d in list(self._watched_docs):
            try:
                d.changed.disconnect(self._on_layer_source_changed)
            except Exception:
                pass
        # Connect new
        newdocs = self._collect_layer_docs()
        for d in newdocs:
            try:
                d.changed.connect(self._on_layer_source_changed)
            except Exception:
                pass
        self._watched_docs = newdocs


    def toggle_mask_overlay(self):
        self.show_mask_overlay = not self.show_mask_overlay
        self._render(rebuild=True)

    def _rebuild_title(self, *, base: str | None = None):
        sub = self._mdi_subwindow()
        if not sub:
            return
        if base is None:
            base = self._effective_title() or "Untitled"

        title = base
        if self._active_title_prefix:
            title = f"{ACTIVE_PREFIX}{title}"
        if self._mask_dot_enabled:
            title = f"{MASK_GLYPH} {title}"

        # only emit when it actually changes
        if title != sub.windowTitle():
            sub.setWindowTitle(title)
            sub.setToolTip(title)
            if title != self._last_title_for_emit:
                self._last_title_for_emit = title
                # notify listeners (Layers dock etc.)
                try:
                    self.viewTitleChanged.emit(self, title)
                except Exception:
                    pass


    def _strip_decorations(self, title: str) -> tuple[str, bool]:
        had_glyph = False
        while len(title) >= 2 and title[1] == " " and title[0] in GLYPHS:
            title = title[2:]
            had_glyph = True
        if title.startswith("Active View: "):
            title = title[len("Active View: "):]
        return title, had_glyph

    def set_active_highlight(self, on: bool):
        self._is_active_flag = bool(on)
        return
        sub = self._mdi_subwindow()
        if not sub:
            return

        core, had_glyph = self._strip_decorations(sub.windowTitle())

        if on and not getattr(self, "_suppress_active_once", False):
            core = ACTIVE_PREFIX + core
        self._suppress_active_once = False

        # recompose: glyph (from flag), then active prefix, then base/core
        if getattr(self, "_mask_dot_enabled", False):
            core = "■ " + core
        #sub.setWindowTitle(core)
        sub.setToolTip(core)

    def _set_mask_highlight(self, on: bool):
        self._mask_dot_enabled = bool(on)
        self._rebuild_title()

    def _sync_host_title(self):
        # document renamed → rebuild from flags + new base
        self._rebuild_title()



    def base_doc_title(self) -> str:
        """The clean, base title (document display name), no prefixes/suffixes."""
        return self.document.display_name() or "Untitled"

    def _active_mask_array(self):
        """Return the active mask ndarray (H,W) or None."""
        doc = getattr(self, "document", None)
        if not doc:
            return None
        mid = getattr(doc, "active_mask_id", None)
        if not mid:
            return None
        masks = getattr(doc, "masks", {}) or {}
        layer = masks.get(mid)
        if layer is None:
            return None
        data = getattr(layer, "data", None)
        if data is None:
            return None
        import numpy as np
        a = np.asarray(data)
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            return None
        # ensure 0..1 float
        a = a.astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0)
        return a

    def refresh_mask_overlay(self):
        """Recompute the source buffer (incl. red mask tint) and repaint."""
        self._render(rebuild=True)

    def _apply_subwindow_style(self):
        """No-op shim retained for backward compatibility."""
        pass

    def _on_doc_mask_changed(self):
        """Doc changed → refresh highlight and overlay if needed."""
        has_mask = self._active_mask_array() is not None
        self._set_mask_highlight(has_mask)
        if self.show_mask_overlay and has_mask:
            self._render(rebuild=True)
        elif self.show_mask_overlay and not has_mask:
            # overlay was on but mask went away → just redraw to clear
            self._render(rebuild=True)


    # ---------- public API ----------
    def set_autostretch(self, on: bool):
        on = bool(on)
        if on == getattr(self, "autostretch_enabled", False):
            # still rebuild so linked profile changes can reflect immediately if desired
            pass
        self.autostretch_enabled = on
        try:
            self.autostretchChanged.emit(on)
        except Exception:
            pass
        # keep your newer fast-path behavior
        self._recompute_autostretch_and_update()

    def toggle_autostretch(self):
        self.set_autostretch(not self.autostretch_enabled)

    def set_autostretch_target(self, target: float):
        self.autostretch_target = float(target)
        if self.autostretch_enabled:
            self._render(rebuild=True)

    def set_autostretch_sigma(self, sigma: float):
        self.autostretch_sigma = float(sigma)
        if self.autostretch_enabled:
            self._render(rebuild=True)

    def set_autostretch_profile(self, profile: str):
        """'normal' => target=0.25, sigma=3 ; 'hard' => target=0.5, sigma=1"""
        p = (profile or "").lower()
        if p not in ("normal", "hard"):
            p = "normal"
        if p == self.autostretch_profile:
            return
        if p == "hard":
            self.autostretch_target = 0.5
            self.autostretch_sigma  = 2
        else:
            self.autostretch_target = 0.3
            self.autostretch_sigma  = 5
        self.autostretch_profile = p
        if self.autostretch_enabled:
            self._render(rebuild=True)

    def is_hard_autostretch(self) -> bool:
        return self.autostretch_profile == "hard"

    def _mdi_subwindow(self) -> QMdiSubWindow | None:
        w = self.parent()
        while w is not None and not isinstance(w, QMdiSubWindow):
            w = w.parent()
        return w

    def _effective_title(self) -> str:
        # Prefer a per-view override; otherwise doc display name
        return self._view_title_override or self.document.display_name()

    def _show_ctx_menu(self, pos):
        menu = QMenu(self)
        a_view = menu.addAction("Rename View… (F2)")
        a_doc  = menu.addAction("Rename Document…")
        menu.addSeparator()
        a_min  = menu.addAction("Send to Shelf")
        a_clear = menu.addAction("Clear View Name (use doc name)")
        menu.addSeparator()
        a_help = menu.addAction("Show pixel/WCS readout hint")
        menu.addSeparator()
        a_prev = menu.addAction("Create Preview (drag rectangle)")   # ← move BEFORE exec

        act = menu.exec(self.mapToGlobal(pos))

        if act == a_view:
            self._rename_view()
        elif act == a_doc:
            self._rename_document()
        elif act == a_min:
            self._send_to_shelf()
        elif act == a_clear:
            self._view_title_override = None
            self._sync_host_title()
        elif act == a_help:
            self._announce_readout_help()
        elif act == a_prev:
            self._preview_btn.setChecked(True)
            self._toggle_preview_select_mode(True)



    def _send_to_shelf(self):
        sub = self._mdi_subwindow()
        mw  = self._find_main_window()
        if sub and mw and hasattr(mw, "window_shelf"):
            sub.hide()
            mw.window_shelf.add_entry(sub)


    def _rename_view(self):
        current = self._view_title_override or self.document.display_name()
        new, ok = QInputDialog.getText(self, "Rename View", "New view name:", text=current)
        if ok and new.strip():
            self._view_title_override = new.strip()
            self._sync_host_title()  # calls _rebuild_title → emits viewTitleChanged

            # optional: directly ping layers dock (defensive)
            mw = self._find_main_window()
            if mw and hasattr(mw, "layers_dock") and mw.layers_dock:
                try:
                    mw.layers_dock._refresh_titles_only()
                except Exception:
                    pass

    def _rename_document(self):
        current = self.document.display_name()
        new, ok = QInputDialog.getText(self, "Rename Document", "New document name:", text=current)
        if ok and new.strip():
            # store on the doc so Explorer + other views update too
            self.document.metadata["display_name"] = new.strip()
            self.document.changed.emit()  # triggers all listeners
            # If this view had an override equal to the old name, drop it
            if self._view_title_override and self._view_title_override == current:
                self._view_title_override = None
            self._sync_host_title()
            mw = self._find_main_window()
            if mw and hasattr(mw, "layers_dock") and mw.layers_dock:
                try:
                    mw.layers_dock._refresh_titles_only()
                except Exception:
                    pass

    def set_scale(self, s: float):
        self.scale = float(max(self._min_scale, min(s, self._max_scale)))
        self._render()



    # ---- view state API (center in image coords + scale) ----
    #def get_view_state(self) -> dict:
    #    pm = self.label.pixmap()
    #    if pm is None:
    #        return {"scale": self.scale, "center": (0.0, 0.0)}
    #    vp = self.scroll.viewport().size()
    #    hbar = self.scroll.horizontalScrollBar()
    #    vbar = self.scroll.verticalScrollBar()
    #    cx_label = hbar.value() + vp.width() / 2.0
    #    cy_label = vbar.value() + vp.height() / 2.0
    #    return {
    #        "scale": float(self.scale),
    #        "center": (float(cx_label / max(1e-6, self.scale)),
    #                   float(cy_label / max(1e-6, self.scale)))
    #    }

    def _start_viewstate_drag(self):
        """Package view state + robust doc identity into a drag."""
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        state = {
            "doc_ptr": id(self.document),                      # legacy
            "scale": float(self.scale),
            "hval": int(hbar.value()),
            "vval": int(vbar.value()),
            "autostretch": bool(self.autostretch_enabled),
            "autostretch_target": float(self.autostretch_target),
        }
        state.update(self._drag_identity_fields())             # ← NEW: uid + base_uid + file_path

        md = QMimeData()
        md.setData(MIME_VIEWSTATE, QByteArray(json.dumps(state).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(md)
        if self.label.pixmap():
            drag.setPixmap(self.label.pixmap().scaled(
                64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            drag.setHotSpot(QPoint(16, 16))
        drag.exec(Qt.DropAction.CopyAction)


    def _start_mask_drag(self):
        """
        Start a drag that carries 'this document is a mask' to drop targets.
        """
        payload = {
            "mask_doc_ptr": id(self.document),
            "mode": "replace",       # future: "union"/"intersect"/"diff"
            "invert": False,
            "feather": 0.0,          # px
            "name": self.document.display_name(),
        }
        payload.update(self._drag_identity_fields())           # ← add uid/base_uid/file_path

        md = QMimeData()
        md.setData(MIME_MASK, QByteArray(json.dumps(payload).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(md)
        if self.label.pixmap():
            drag.setPixmap(self.label.pixmap().scaled(
                64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            drag.setHotSpot(QPoint(16, 16))
        drag.exec(Qt.DropAction.CopyAction)

    def _start_astrometry_drag(self):
        """
        Start a drag that carries 'copy astrometric solution from this document'.
        We only send a pointer; the main window resolves + copies actual WCS.
        """
        payload = {
            "wcs_from_doc_ptr": id(self.document),
            "name": self.document.display_name(),
        }
        payload.update(self._drag_identity_fields()) 
        md = QMimeData()
        md.setData(MIME_ASTROMETRY, QByteArray(json.dumps(payload).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(md)
        if self.label.pixmap():
            drag.setPixmap(self.label.pixmap().scaled(
                64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            drag.setHotSpot(QPoint(16, 16))
        drag.exec(Qt.DropAction.CopyAction)


    def apply_view_state(self, st: dict):
        try:
            new_scale = float(st.get("scale", self.scale))
        except Exception:
            new_scale = self.scale
        # clamp with new max
        self.scale = max(self._min_scale, min(new_scale, self._max_scale))
        self._render(rebuild=False)

        vp = self.scroll.viewport().size()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        if "hval" in st or "vval" in st:
            # direct scrollbar values (fast path)
            hv = int(st.get("hval", hbar.value()))
            vv = int(st.get("vval", vbar.value()))
            hbar.setValue(hv)
            vbar.setValue(vv)
            return

        # fallback: center in image coordinates
        center = st.get("center")
        if center is None:
            return
        try:
            cx_img, cy_img = float(center[0]), float(center[1])
        except Exception:
            return
        cx_label = cx_img * self.scale
        cy_label = cy_img * self.scale
        hbar.setValue(int(cx_label - vp.width()  / 2.0))
        vbar.setValue(int(cy_label - vp.height() / 2.0))


    # ---- DnD 'view tab' -------------------------------------------------
    def _install_view_tab(self):
        self._view_tab = QToolButton(self)
        self._view_tab.setText("View")
        self._view_tab.setToolTip("Drag onto another window to copy zoom/pan.\n"
                                  "Double-click to duplicate this view.")
        self._view_tab.setCursor(Qt.CursorShape.OpenHandCursor)
        self._view_tab.setAutoRaise(True)
        self._view_tab.move(8, 8)     # pinned near top-left of the subwindow
        self._view_tab.show()

        # start drag on press
        self._view_tab.mousePressEvent = self._viewtab_mouse_press
        # duplicate on double-click
        self._view_tab.mouseDoubleClickEvent = self._viewtab_mouse_double

    def _viewtab_mouse_press(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return QToolButton.mousePressEvent(self._view_tab, ev)

        # build the SAME payload schema used by _start_viewstate_drag()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        state = {
            "doc_ptr": id(self.document),
            "scale": float(self.scale),
            "hval": int(hbar.value()),
            "vval": int(vbar.value()),
            "autostretch": bool(self.autostretch_enabled),
            "autostretch_target": float(self.autostretch_target),
        }
        state.update(self._drag_identity_fields())

        mime = QMimeData()
        mime.setData(MIME_VIEWSTATE, QByteArray(json.dumps(state).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(mime)

        pm = self.label.pixmap()
        if pm:
            drag.setPixmap(pm.scaled(96, 96,
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation))
            drag.setHotSpot(QCursor.pos() - self.mapToGlobal(self._view_tab.pos()))
        drag.exec(Qt.DropAction.CopyAction)

    def _viewtab_mouse_double(self, _ev):
        # ask main window to duplicate this subwindow
        self.requestDuplicate.emit(self)

    # accept view-state drops anywhere in the view
    def dragEnterEvent(self, ev):
        md = ev.mimeData()

        if (md.hasFormat(MIME_VIEWSTATE)
                or md.hasFormat(MIME_ASTROMETRY)
                or md.hasFormat(MIME_MASK)
                or md.hasFormat(MIME_CMD)):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        md = ev.mimeData()

        if (md.hasFormat(MIME_VIEWSTATE)
                or md.hasFormat(MIME_ASTROMETRY)
                or md.hasFormat(MIME_MASK)
                or md.hasFormat(MIME_CMD)):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev):
        md = ev.mimeData()

        # 0) Function/Action command → forward to main window for headless/UI routing
        if md.hasFormat(MIME_CMD):
            try:
                payload = _unpack_cmd_payload(bytes(md.data(MIME_CMD)))
            except Exception:
                ev.ignore(); return
            mw = self._find_main_window()
            sw = self._mdi_subwindow()
            if mw and sw and hasattr(mw, "_handle_command_drop"):
                mw._handle_command_drop(payload, sw)
                ev.acceptProposedAction()
            else:
                ev.ignore()
            return

        # 1) view state (existing)
        if md.hasFormat(MIME_VIEWSTATE):
            try:
                st = json.loads(bytes(md.data(MIME_VIEWSTATE)).decode("utf-8"))
                self.apply_view_state(st)
                ev.acceptProposedAction()
            except Exception:
                ev.ignore()
            return

        # 2) mask (NEW) → forward to main-window handler using this view as target
        if md.hasFormat(MIME_MASK):
            try:
                payload = json.loads(bytes(md.data(MIME_MASK)).decode("utf-8"))
            except Exception:
                ev.ignore(); return
            mw = self._find_main_window()
            sw = self._mdi_subwindow()
            if mw and sw and hasattr(mw, "_handle_mask_drop"):
                mw._handle_mask_drop(payload, sw)
                ev.acceptProposedAction()
            else:
                ev.ignore()
            return

        # 3) astrometry (existing forwarding)
        if md.hasFormat(MIME_ASTROMETRY):
            try:
                payload = json.loads(bytes(md.data(MIME_ASTROMETRY)).decode("utf-8"))
            except Exception:
                ev.ignore(); return
            mw = self._find_main_window()
            sw = self._mdi_subwindow()
            if mw and hasattr(mw, "_on_astrometry_drop") and sw is not None:
                mw._on_astrometry_drop(payload, sw)
                ev.acceptProposedAction()
            else:
                ev.ignore()
            return

        ev.ignore()

    # keep the tab visible if the widget resizes
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, "_view_tab"):
            self._view_tab.raise_()

    def is_autostretch_linked(self) -> bool:
        return bool(self._autostretch_linked)

    def set_autostretch_linked(self, linked: bool):
        linked = bool(linked)
        if self._autostretch_linked == linked:
            return
        self._autostretch_linked = linked
        if self.autostretch_enabled:
            self._recompute_autostretch_and_update()

    def _on_docman_nudge(self, *args):
        # Guard against late signals hitting after destruction/minimize
        try:
            from PyQt6 import sip as _sip
            if _sip.isdeleted(self):
                return
        except Exception:
            pass
        try:
            self._refresh_local_undo_buttons()
        except RuntimeError:
            # Buttons already gone; safe to ignore
            pass
        except Exception:
            pass


    def _recompute_autostretch_and_update(self):
        self._qimg_src = None   # force source rebuild
        self._render(True)

    def set_doc_manager(self, docman):
        self._docman = docman
        try:
            docman.imageRegionUpdated.connect(self._on_doc_region_updated)
            docman.imageRegionUpdated.connect(self._on_docman_nudge)
            if hasattr(docman, "previewRepaintRequested"):
                docman.previewRepaintRequested.connect(self._on_docman_nudge)
        except Exception:
            pass

        base = getattr(self, "base_document", None) or getattr(self, "document", None)
        if base is not None:
            try:
                base.changed.connect(self._on_base_doc_changed)
            except Exception:
                pass
        self._install_history_watchers()

    def _on_base_doc_changed(self):
        # Full-image changes (or unknown) → rebuild our pixmap
        QTimer.singleShot(0, lambda: (self._render(rebuild=True), self._refresh_local_undo_buttons()))

    def _on_doc_region_updated(self, doc, roi_tuple_or_none):
        # Only react if it’s our base doc
        base = getattr(self, "base_document", None) or getattr(self, "document", None)
        if doc is None or base is None or doc is not base:
            return

        # If not on a Preview tab, just refresh.
        if not (getattr(self, "_active_source_kind", None) == "preview"
                and getattr(self, "_active_preview_id", None) is not None):
            QTimer.singleShot(0, lambda: self._render(rebuild=True))
            return

        # We’re on a Preview tab: refresh only if the changed region overlaps our ROI.
        try:
            my_roi = self.current_preview_roi()  # (x,y,w,h) in full-image coords
        except Exception:
            my_roi = None

        if my_roi is None or roi_tuple_or_none is None:
            QTimer.singleShot(0, lambda: self._render(rebuild=True))
            return

        if self._roi_intersects(my_roi, roi_tuple_or_none):
            QTimer.singleShot(0, lambda: self._render(rebuild=True))

    @staticmethod
    def _roi_intersects(a, b):
        ax, ay, aw, ah = map(int, a)
        bx, by, bw, bh = map(int, b)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return False
        return not (ax+aw <= bx or bx+bw <= ax or ay+ah <= by or by+bh <= ay)

    def refresh_from_docman(self):
        #print("[ImageSubWindow] refresh_from_docman called")
        """
        Called by MainWindow when DocManager says the image changed.
        We nuke the cached QImage and rebuild from the current doc proxy
        (which resolves ROI vs full), so the Preview tab repaints correctly.
        """
        try:
            # Invalidate any cached source so _render() fully rebuilds
            if hasattr(self, "_qimg_src"):
                self._qimg_src = None
        except Exception:
            pass
        self._render(rebuild=True)

    # ---------- rendering ----------
    def _render(self, rebuild: bool = False):
        """
        Render the current view.

        Rules:
        - If a Preview is active, FIRST sync that preview's stored arr from the
        DocManager's ROI document (the thing tools actually modify), then render.
        - Never reslice from the parent/full image here.
        - Keep a strong reference to the numpy buffer that backs the QImage.
        """
        # ---------------------------
        # 1) Choose & sync source arr
        # ---------------------------
        base_img = None
        if self._active_source_kind == "preview" and self._active_preview_id is not None:
            src = next((p for p in self._previews if p["id"] == self._active_preview_id), None)
            #print("[ImageSubWindow] _render: preview mode, id =", self._active_preview_id, "src =", src is not None)
            if src is not None:
                # Pull the *edited* ROI image from DocManager, if available
                if hasattr(self, "_docman") and self._docman is not None:
                    #print("[ImageSubWindow] _render: pulling edited ROI from DocManager")
                    try:
                        roi_doc = self._docman.get_document_for_view(self)
                        roi_img = getattr(roi_doc, "image", None)
                        if roi_img is not None:
                            # Replace the preview’s static copy with the edited ROI buffer
                            src["arr"] = np.asarray(roi_img).copy()
                    except Exception:
                        print("[ImageSubWindow] _render: failed to pull edited ROI from DocManager")
                        pass
                base_img = src.get("arr", None)
        else:
            #print("[ImageSubWindow] _render: full image mode")
            base_img = self._display_override if (self._display_override is not None) else (
                getattr(self.document, "image", None)
            )

        if base_img is None:
            self._qimg_src = None
            self.label.clear()
            return

        arr = np.asarray(base_img)

        # ---------------------------------------
        # 2) Normalize dimensionality and dtype
        # ---------------------------------------
        # Scalar → 1x1; 1D → 1xN; (H,W,1) → mono (H,W)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]

        is_mono = (arr.ndim == 2)

        # ---------------------------------------
        # 3) Visualization buffer (float32)
        # ---------------------------------------
        if self.autostretch_enabled:
            if np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                denom = float(max(1, info.max))
                arr_f = (arr.astype(np.float32) / denom)
            else:
                arr_f = arr.astype(np.float32, copy=False)
                mx = float(arr_f.max()) if arr_f.size else 1.0
                if mx > 5.0:  # compress absurdly large ranges
                    arr_f = arr_f / mx

            vis = autostretch(
                arr_f,
                target_median=self.autostretch_target,
                sigma=self.autostretch_sigma,
                linked=(not is_mono and self._autostretch_linked),
                use_16bit=None,
            )
        else:
            vis = arr

        # ---------------------------------------
        # 4) Convert to 8-bit RGB for QImage
        # ---------------------------------------
        if vis.dtype == np.uint8:
            buf8 = vis
        elif vis.dtype == np.uint16:
            buf8 = (vis.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            buf8 = (np.clip(vis.astype(np.float32, copy=False), 0.0, 1.0) * 255.0).astype(np.uint8)

        # Force H×W×3
        if buf8.ndim == 2:
            buf8 = np.stack([buf8] * 3, axis=-1)
        elif buf8.ndim == 3:
            c = buf8.shape[2]
            if c == 1:
                buf8 = np.repeat(buf8, 3, axis=2)
            elif c > 3:
                buf8 = buf8[..., :3]
        else:
            buf8 = np.stack([buf8.squeeze()] * 3, axis=-1)

        # ---------------------------------------
        # 5) Optional mask overlay
        # ---------------------------------------
        if getattr(self, "show_mask_overlay", False):
            m = self._active_mask_array()
            if m is not None:
                if getattr(self, "_mask_overlay_invert", True):
                    m = 1.0 - m
                th, tw = buf8.shape[:2]
                sh, sw = m.shape[:2]
                if (sh, sw) != (th, tw):
                    yi = (np.linspace(0, sh - 1, th)).astype(np.int32)
                    xi = (np.linspace(0, sw - 1, tw)).astype(np.int32)
                    m = m[yi][:, xi]
                a = m.astype(np.float32, copy=False) * float(getattr(self, "_mask_overlay_alpha", 0.35))
                bf = buf8.astype(np.float32, copy=False)
                bf[..., 0] = np.clip(bf[..., 0] + (255.0 - bf[..., 0]) * a, 0.0, 255.0)
                buf8 = bf.astype(np.uint8, copy=False)

        # ---------------------------------------
        # 6) Wrap into QImage (keep buffer alive)
        # ---------------------------------------
        buf8 = np.ascontiguousarray(buf8)
        h, w, _ = buf8.shape
        bytes_per_line = buf8.strides[0]

        self._buf8 = buf8  # keep alive
        try:

            ptr = sip.voidptr(self._buf8.ctypes.data)
            qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        except Exception:
            buf8c = np.array(self._buf8, copy=True, order="C")
            self._buf8 = buf8c
            ptr = sip.voidptr(self._buf8.ctypes.data)
            qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self._qimg_src = qimg
        if qimg is None or qimg.isNull():
            self.label.clear()
            return

        # ---------------------------------------
        # 7) Scale & present
        # ---------------------------------------
        sw = max(1, int(qimg.width() * self.scale))
        sh = max(1, int(qimg.height() * self.scale))
        scaled = qimg.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(QPixmap.fromImage(scaled))
        self.label.resize(scaled.size())


    def has_active_preview(self) -> bool:
        return self._active_source_kind == "preview" and self._active_preview_id is not None

    def current_preview_roi(self) -> tuple[int,int,int,int] | None:
        """
        Returns (x, y, w, h) in FULL image coordinates if a preview tab is active, else None.
        """
        if not self.has_active_preview():
            return None
        src = next((p for p in self._previews if p["id"] == self._active_preview_id), None)
        return None if src is None else tuple(src["roi"])

    def current_preview_name(self) -> str | None:
        if not self.has_active_preview():
            return None
        src = next((p for p in self._previews if p["id"] == self._active_preview_id), None)
        return None if src is None else src["name"]


    # ---------- interaction ----------
    def _zoom_at_anchor(self, factor: float):
        if self._qimg_src is None:
            return
        old_scale = self.scale
        # clamp with new max
        new_scale = max(self._min_scale, min(old_scale * factor, self._max_scale))
        if abs(new_scale - old_scale) < 1e-8:
            return

        vp = self.scroll.viewport()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        # Anchor in viewport coordinates via global cursor (robust)
        try:
            anchor_vp = vp.mapFromGlobal(QCursor.pos())
        except Exception:
            anchor_vp = None

        if (anchor_vp is None) or (not vp.rect().contains(anchor_vp)):
            anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)

        # Current label coords under the anchor
        x_label_pre = hbar.value() + anchor_vp.x()
        y_label_pre = vbar.value() + anchor_vp.y()

        # Convert to image coords at old scale
        xi = x_label_pre / max(old_scale, 1e-12)
        yi = y_label_pre / max(old_scale, 1e-12)

        # Apply scale and redraw (updates label size + scrollbar ranges)
        self.scale = new_scale
        self._render(rebuild=False)

        # Reproject that image point to label coords at new scale
        x_label_post = xi * new_scale
        y_label_post = yi * new_scale

        # Desired scrollbar values to keep point under the cursor
        new_h = int(round(x_label_post - anchor_vp.x()))
        new_v = int(round(y_label_post - anchor_vp.y()))

        # Clamp to valid range
        new_h = max(hbar.minimum(), min(new_h, hbar.maximum()))
        new_v = max(vbar.minimum(), min(new_v, vbar.maximum()))

        # Apply
        hbar.setValue(new_h)
        vbar.setValue(new_v)

    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    def eventFilter(self, obj, ev):
        is_on_view = (obj is self.label) or (obj is self.scroll.viewport())

        # 0) PREVIEW-SELECT MODE: consume mouse events first so earlier branches don't steal them
        if self._preview_select_mode and is_on_view:
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                vp_pos = obj.mapTo(self.scroll.viewport(), ev.pos())
                self._rubber_origin = vp_pos
                if self._rubber is None:
                    self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self.scroll.viewport())
                self._rubber.setGeometry(QRect(self._rubber_origin, QSize(1, 1)))
                self._rubber.show()
                ev.accept(); return True

            if ev.type() == QEvent.Type.MouseMove and self._rubber and self._rubber_origin is not None:
                vp_pos = obj.mapTo(self.scroll.viewport(), ev.pos())
                rect = QRect(self._rubber_origin, vp_pos).normalized()
                self._rubber.setGeometry(rect)
                ev.accept(); return True

            if ev.type() == QEvent.Type.MouseButtonRelease and self._rubber and self._rubber_origin is not None:
                vp_pos = obj.mapTo(self.scroll.viewport(), ev.pos())
                rect = QRect(self._rubber_origin, vp_pos).normalized()
                self._finish_preview_rect(rect)
                ev.accept(); return True
            # If in preview mode but not one of the handled events, let others pass through
            # (no return here)

        # 1) Ctrl + wheel → zoom
        if ev.type() == QEvent.Type.Wheel:
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else 1/1.25
                self._zoom_at_anchor(factor)
                return True
            return False

        # 2) Space+click → start readout
        if ev.type() == QEvent.Type.MouseButtonPress:
            if self._space_down and ev.button() == Qt.MouseButton.LeftButton:
                vp_pos = obj.mapTo(self.scroll.viewport(), ev.pos())
                res = self._sample_image_at_viewport_pos(vp_pos)
                if res is not None:
                    xi, yi, sample = res
                    self._show_readout(xi, yi, sample)
                self._readout_dragging = True
                return True
            return False

        # 3) Space+drag → live readout
        if ev.type() == QEvent.Type.MouseMove:
            if self._readout_dragging:
                vp_pos = obj.mapTo(self.scroll.viewport(), ev.pos())
                res = self._sample_image_at_viewport_pos(vp_pos)
                if res is not None:
                    xi, yi, sample = res
                    self._show_readout(xi, yi, sample)
                return True
            return False

        # 4) Release → stop live readout
        if ev.type() == QEvent.Type.MouseButtonRelease:
            if self._readout_dragging:
                self._readout_dragging = False
                return True
            return False

        return super().eventFilter(obj, ev)


    def _finish_preview_rect(self, vp_rect: QRect):
        # Map viewport rectangle into image coordinates
        if vp_rect.width() < 4 or vp_rect.height() < 4:
            self._cancel_rubber()
            return

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        # Upper-left in label coords
        x_label0 = hbar.value() + vp_rect.left()
        y_label0 = vbar.value() + vp_rect.top()
        x_label1 = hbar.value() + vp_rect.right()
        y_label1 = vbar.value() + vp_rect.bottom()

        s = max(self.scale, 1e-12)

        x0 = int(round(x_label0 / s))
        y0 = int(round(y_label0 / s))
        x1 = int(round(x_label1 / s))
        y1 = int(round(y_label1 / s))

        if x1 <= x0 or y1 <= y0:
            self._cancel_rubber()
            return

        roi = (x0, y0, x1 - x0, y1 - y0)
        self._create_preview_from_roi(roi)
        self._cancel_rubber()

    def _create_preview_from_roi(self, roi: tuple[int,int,int,int]):
        """
        roi: (x, y, w, h) in FULL IMAGE coordinates
        """
        arr = np.asarray(self.document.image)
        H, W = (arr.shape[0], arr.shape[1]) if arr.ndim >= 2 else (0, 0)
        x, y, w, h = roi
        # clamp to image bounds
        x = max(0, min(x, max(0, W-1)))
        y = max(0, min(y, max(0, H-1)))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        crop = arr[y:y+h, x:x+w].copy()  # isolate for preview

        pid = self._next_preview_id
        self._next_preview_id += 1
        name = f"Preview {pid} ({w}×{h})"

        self._previews.append({"id": pid, "name": name, "roi": (x, y, w, h), "arr": crop})

        # Build a tab with a simple QLabel viewer (reuses global rendering through _render)
        host = QWidget(self)
        l = QVBoxLayout(host); l.setContentsMargins(0,0,0,0)
        # For simplicity, we reuse the SAME scroll/label pipeline; the source image is switched in _render
        # but we still want a local label so the tab displays something. Make a tiny label holder:
        holder = QLabel(" ")  # placeholder; we still render into self.label (single view)
        holder.setMinimumHeight(1)
        l.addWidget(holder)

        host._preview_id = pid  # attach id for lookups
        idx = self._tabs.addTab(host, name)
        self._tabs.setCurrentIndex(idx)
        self._tabs.tabBar().setVisible(True)  # show tabs when first preview appears

        # Switch active source and redraw
        self._active_source_kind = "preview"
        self._active_preview_id = pid
        self._render(True)

    def mousePressEvent(self, e):
        # If we're defining a preview ROI, don't start panning here
        if self._preview_select_mode:
            e.ignore()             # let the eventFilter (label/viewport) handle it
            return

        if e.button() == Qt.MouseButton.LeftButton:
            if self._space_down:
                vp = self.scroll.viewport()
                vp_pos = vp.mapFrom(self, e.pos())
                res = self._sample_image_at_viewport_pos(vp_pos)
                if res is not None:
                    xi, yi, sample = res
                    self._show_readout(xi, yi, sample)
                self._readout_dragging = True
                return

            # normal pan mode
            self._dragging = True
            self._drag_start = e.pos()
            return

        super().mousePressEvent(e)


    def _show_readout(self, xi, yi, sample):
        mw = self._find_main_window()
        if mw is None:
            return

        # We want raw float prints, never 16-bit normalized
        r = g = b = None
        k = None

        if isinstance(sample, dict):
            # 1) the clean mono path
            if "mono" in sample:
                try:
                    k = float(sample["mono"])
                except Exception:
                    k = sample["mono"]
            # 2) the clean RGB path
            elif all(ch in sample for ch in ("r", "g", "b")):
                try:
                    r = float(sample["r"])
                    g = float(sample["g"])
                    b = float(sample["b"])
                except Exception:
                    r = sample["r"]; g = sample["g"]; b = sample["b"]
            else:
                # 3) weird dict → just take the first numeric-looking value
                for v in sample.values():
                    try:
                        k = float(v)
                        break
                    except Exception:
                        continue

        elif isinstance(sample, (list, tuple)):
            if len(sample) == 1:
                try:
                    k = float(sample[0])
                except Exception:
                    k = sample[0]
            elif len(sample) >= 3:
                try:
                    r = float(sample[0]); g = float(sample[1]); b = float(sample[2])
                except Exception:
                    r, g, b = sample[0], sample[1], sample[2]

        else:
            # numpy scalar / plain number
            try:
                k = float(sample)
            except Exception:
                k = sample

        msg = f"x={xi}  y={yi}"

        if r is not None and g is not None and b is not None:
            msg += f"   R={r:.6f}  G={g:.6f}  B={b:.6f}"
        elif k is not None:
            msg += f"   K={k:.6f}"
        else:
            # final fallback if everything was weird
            msg += "   K=?"

        # ---- WCS ----
        wcs = self._extract_wcs_from_doc()
        if wcs is not None:
            try:
                world = wcs.pixel_to_world_values(float(xi), float(yi))
                ra_deg, dec_deg = float(world[0]), float(world[1])

                # RA
                ra_h = ra_deg / 15.0
                ra_hh = int(ra_h)
                ra_mm = int((ra_h - ra_hh) * 60.0)
                ra_ss = ((ra_h - ra_hh) * 60.0 - ra_mm) * 60.0

                # Dec
                sign = "+" if dec_deg >= 0 else "-"
                d = abs(dec_deg)
                dec_dd = int(d)
                dec_mm = int((d - dec_dd) * 60.0)
                dec_ss = ((d - dec_dd) * 60.0 - dec_mm) * 60.0

                msg += (
                    f"   RA={ra_hh:02d}:{ra_mm:02d}:{ra_ss:05.2f}"
                    f"  Dec={sign}{dec_dd:02d}:{dec_mm:02d}:{dec_ss:05.2f}"
                )
            except Exception:
                pass

        mw.statusBar().showMessage(msg)



    def _deg_to_hms(self, ra_deg: float) -> str:
        """
        RA in degrees → 'HH:MM:SS.s'
        """
        # RA: 0..360 deg → 0..24 h
        total_seconds = (ra_deg / 15.0) * 3600.0
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60.0
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    def _deg_to_dms(self, dec_deg: float) -> str:
        """
        Dec in degrees → '+DD:MM:SS.s'
        """
        sign = "+" if dec_deg >= 0 else "-"
        d = abs(dec_deg)
        total_seconds = d * 3600.0
        deg = int(total_seconds // 3600)
        arcmin = int((total_seconds % 3600) // 60)
        arcsec = total_seconds % 60.0
        return f"{sign}{deg:02d}:{arcmin:02d}:{arcsec:05.2f}"

    def _extract_wcs_from_doc(self):
        """
        Try to get an astropy WCS object from the current document.
        Priority:
        1) cached on metadata["_astropy_wcs"]
        2) explicit metadata["wcs"] (already a WCS)
        3) metadata["original_header"] / ["fits_header"] / ["header"]
           - FITS Header → WCS(...)
           - XISF-style dict with FITSKeywords → rebuild Header → WCS(...)
        Returns WCS or None.
        """
        doc = getattr(self, "document", None)
        if doc is None:
            return None

        meta = getattr(doc, "metadata", {}) or {}

        # 0) cached
        cached = meta.get("_astropy_wcs")
        if cached is not None:
            return cached

        # 1) explicit WCS stored there
        explicit = meta.get("wcs")
        if explicit is not None:
            # cache for faster reuse
            meta["_astropy_wcs"] = explicit
            return explicit

        # We'll need astropy here
        try:
            from astropy.io.fits import Header
            from astropy.wcs import WCS
        except Exception:
            return None

        # 2) try to find *any* header-like thing
        hdr = (
            meta.get("original_header")
            or meta.get("fits_header")
            or meta.get("header")
        )

        # 2a) it is already an astropy Header
        if isinstance(hdr, Header):
            try:
                w = WCS(hdr, relax=True)
                meta["_astropy_wcs"] = w
                return w
            except Exception:
                return None

        # 2b) XISF-style dict: look for "FITSKeywords"
        if isinstance(hdr, dict):
            fk = hdr.get("FITSKeywords")
            if fk:
                # build a temporary FITS header
                tmp = Header()
                # XISF often stores like: {"CTYPE1": [{"value": "RA---TAN"}], ...}
                # so we need to be defensive
                for key, val in fk.items():
                    # val can be list[dict] or scalar
                    if isinstance(val, list) and val:
                        first = val[0]
                        v = first.get("value")
                        c = first.get("comment", "")
                    else:
                        v = val
                        c = ""
                    if v is None:
                        continue
                    try:
                        tmp[str(key)] = (v, c)
                    except Exception:
                        # fallback: at least set the value
                        tmp[str(key)] = v
                try:
                    w = WCS(tmp, relax=True)
                    meta["_astropy_wcs"] = w
                    return w
                except Exception:
                    return None

        return None
  
    def mouseMoveEvent(self, e):
        # While defining preview ROI, let the eventFilter drive the QRubberBand
        if self._preview_select_mode:
            e.ignore()
            return

        if self._readout_dragging:
            vp = self.scroll.viewport()
            vp_pos = vp.mapFrom(self, e.pos())
            res = self._sample_image_at_viewport_pos(vp_pos)
            if res is not None:
                xi, yi, sample = res
                self._show_readout(xi, yi, sample)
            return

        if self._dragging:
            delta = e.pos() - self._drag_start
            self.scroll.horizontalScrollBar().setValue(
                self.scroll.horizontalScrollBar().value() - delta.x()
            )
            self.scroll.verticalScrollBar().setValue(
                self.scroll.verticalScrollBar().value() - delta.y()
            )
            self._drag_start = e.pos()
            return

        super().mouseMoveEvent(e)


    def mouseReleaseEvent(self, e):
        if self._preview_select_mode:
            e.ignore()   # eventFilter will consume the release to finish the ROI
            return

        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._readout_dragging = False
            return
        super().mouseReleaseEvent(e)


    def closeEvent(self, e):
        mw = self._find_main_window()
        doc = getattr(self, "document", None)

        # If main window is force-closing (global exit accepted), don't ask.
        force = bool(getattr(mw, "_force_close_all", False))

        if not force and doc is not None:
            # Ask only if this doc has edits
            should_warn = False
            if mw and hasattr(mw, "_document_has_edits"):
                should_warn = mw._document_has_edits(doc)
            else:
                # Fallback if called standalone
                try:
                    should_warn = bool(doc.can_undo())
                except Exception:
                    should_warn = False

            if should_warn:
                r = QMessageBox.question(
                    self, "Close Image?",
                    "This image has edits that aren’t applied/saved.\nClose anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if r != QMessageBox.StandardButton.Yes:
                    e.ignore()
                    return

        try:
            if hasattr(self, "_docman") and self._docman is not None:
                self._docman.imageRegionUpdated.disconnect(self._on_doc_region_updated)
                # NEW: also drop the nudge hook(s)
                try:
                    self._docman.imageRegionUpdated.disconnect(self._on_docman_nudge)
                except Exception:
                    pass
                if hasattr(self._docman, "previewRepaintRequested"):
                    try:
                        self._docman.previewRepaintRequested.disconnect(self._on_docman_nudge)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            base = getattr(self, "base_document", None) or getattr(self, "document", None)
            if base is not None:
                base.changed.disconnect(self._on_base_doc_changed)
        except Exception:
            pass

        # proceed with your current teardown
        try:
            # emit your existing signal if you have it
            if hasattr(self, "aboutToClose"):
                self.aboutToClose.emit(doc)
        except Exception:
            pass
        super().closeEvent(e)

    def _resolve_history_doc(self):
        """
        Return the doc whose history we should mutate:
        - If a Preview tab is active → the ROI/proxy doc from DocManager
        - Otherwise → the base/full document
        """
        # Prefer DocManager's ROI-aware mapping if present
        dm = getattr(self, "_docman", None)
        if (self._active_source_kind == "preview"
                and self._active_preview_id is not None
                and dm is not None
                and hasattr(dm, "get_document_for_view")):
            try:
                d = dm.get_document_for_view(self)
                if d is not None:
                    return d
            except Exception:
                pass
        # Fallback to the main doc
        return getattr(self, "document", None)


    def _refresh_local_undo_buttons(self):
        """Enable/disable the local Undo/Redo toolbuttons based on can_undo/can_redo."""
        try:
            doc = self._resolve_history_doc()
            can_u = bool(doc and hasattr(doc, "can_undo") and doc.can_undo())
            can_r = bool(doc and hasattr(doc, "can_redo") and doc.can_redo())
        except Exception:
            can_u = can_r = False

        b_u = getattr(self, "_btn_undo", None)
        b_r = getattr(self, "_btn_redo", None)

        try:
            if b_u: b_u.setEnabled(can_u)
        except RuntimeError:
            return
        except Exception:
            pass
        try:
            if b_r: b_r.setEnabled(can_r)
        except RuntimeError:
            return
        except Exception:
            pass



# --- NEW: TableSubWindow -------------------------------------------------
from PyQt6.QtWidgets import QTableView, QPushButton, QFileDialog

class TableSubWindow(QWidget):
    """
    Lightweight subwindow to render TableDocument (rows/headers) in a QTableView.
    Provides: copy, export CSV, row count display.
    """
    viewTitleChanged = pyqtSignal(object, str)  # to mirror ImageSubWindow emissions (if needed)

    def __init__(self, table_document, parent=None):
        super().__init__(parent)
        self.document = table_document
        self._last_title_for_emit = None

        lyt = QVBoxLayout(self)
        title_row = QHBoxLayout()
        self.title_lbl = QLabel(self.document.display_name())
        title_row.addWidget(self.title_lbl)
        title_row.addStretch(1)

        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._export_csv)
        title_row.addWidget(self.export_btn)
        lyt.addLayout(title_row)

        self.table = QTableView(self)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        lyt.addWidget(self.table, 1)

        rows = getattr(self.document, "rows", [])
        headers = getattr(self.document, "headers", [])
        self._model = SimpleTableModel(rows, headers, self)
        self.table.setModel(self._model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()

        self._sync_host_title()
        #print(f"[TableSubWindow] init rows={self._model.rowCount()} cols={self._model.columnCount()} title='{self.document.display_name()}'")
        # react to doc rename if you add such behavior later
        try:
            self.document.changed.connect(self._on_doc_changed)
        except Exception:
            pass

    def _on_doc_changed(self):
        # if title changes or content updates in future
        self.title_lbl.setText(self.document.display_name())
        self._sync_host_title()

    def _mdi_subwindow(self) -> QMdiSubWindow | None:
        w = self.parent()
        while w is not None and not isinstance(w, QMdiSubWindow):
            w = w.parent()
        return w

    def _sync_host_title(self):
        sub = self._mdi_subwindow()
        if not sub:
            return
        title = self.document.display_name()
        if title != sub.windowTitle():
            sub.setWindowTitle(title)
            sub.setToolTip(title)
            if title != self._last_title_for_emit:
                self._last_title_for_emit = title
                try:
                    self.viewTitleChanged.emit(self, title)
                except Exception:
                    pass

    def _export_csv(self):
        # Prefer already-exported CSV from metadata when available, otherwise prompt
        existing = self.document.metadata.get("table_csv")
        if existing and os.path.exists(existing):
            # Offer to open/save-as that CSV
            dst, ok = QFileDialog.getSaveFileName(self, "Save CSV As…", os.path.basename(existing), "CSV Files (*.csv)")
            if ok and dst:
                try:
                    import shutil
                    shutil.copyfile(existing, dst)
                except Exception as e:
                    QMessageBox.warning(self, "Export CSV", f"Failed to copy CSV:\n{e}")
            return

        # No pre-export → write one from the model
        dst, ok = QFileDialog.getSaveFileName(self, "Export CSV…", "table.csv", "CSV Files (*.csv)")
        if not ok or not dst:
            return
        try:
            import csv
            with open(dst, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                # headers
                cols = self._model.columnCount()
                hdrs = [self._model.headerData(c, Qt.Orientation.Horizontal) for c in range(cols)]
                w.writerow([str(h) for h in hdrs])
                # rows
                rows = self._model.rowCount()
                for r in range(rows):
                    w.writerow([self._model.data(self._model.index(r, c), Qt.ItemDataRole.DisplayRole) for c in range(cols)])
        except Exception as e:
            QMessageBox.warning(self, "Export CSV", f"Failed to export CSV:\n{e}")
