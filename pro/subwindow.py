# pro/subwindow.py
from __future__ import annotations
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QSize, QEvent, QByteArray, QMimeData, QSettings, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QToolButton, QHBoxLayout, QMessageBox, QMdiSubWindow, QMenu, QInputDialog, QApplication
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QShortcut, QKeySequence, QCursor, QDrag
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


    def __init__(self, document, parent=None):
        super().__init__(parent)
        self.document = document
        self._last_title_for_emit = None


        # view state
        self._min_scale = 0.02
        self._max_scale = 3.00     # 300%        
        self.scale = 0.25
        self._dragging = False
        self._drag_start = QPoint()
        self._autostretch_linked = QSettings().value("display/stretch_linked", False, type=bool)
        self.autostretch_enabled = False
        self.autostretch_target = 0.25    # tweakable, e.g. 0.2–0.35 typical
        self.autostretch_sigma   = 3.0    # normal profile
        self.autostretch_profile = "normal"        
        self.show_mask_overlay = False
        self._mask_overlay_alpha = 0.5  # 0..1
        self._mask_overlay_invert = True 
        self._layers: list[ImageLayer] = []   # per-view layer stack
        self.layers_changed.connect(lambda: None)  # placeholder to avoid "unused" warnings        
        self._display_override: np.ndarray | None = None
        self._readout_hint_shown = False
        # keep mask visuals in sync when the doc changes (mask attach/remove etc.)
        self.document.changed.connect(self._on_doc_mask_changed)
        # context menu + shortcuts
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_ctx_menu)
        QShortcut(QKeySequence("F2"), self, activated=self._rename_view)  # quick rename for this view

        self._space_down = False        # ← NEW
        self._readout_dragging = False  # ← NEW: when we’re in “probe while drag” mode
        self._last_readout = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # so we get key events

        # keep the host MDI subwindow title in sync with doc/view name
        self._view_title_override = None  # if set, only this window shows the override
        self.document.changed.connect(self._sync_host_title)
        self._sync_host_title()

        # cached display sources (reused on zoom/pan)
        self._buf8 = None          # np.uint8 [H,W,3] backing memory
        self._qimg_src = None      # QImage built from _buf8

        # ui
        lyt = QVBoxLayout(self)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._drag_tab = _DragTab(self)
        row.addWidget(self._drag_tab, 0, Qt.AlignmentFlag.AlignLeft)
        row.addStretch(1)
        lyt.addLayout(row)       
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(False)
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)
        lyt.addWidget(self.scroll)

        self.setAcceptDrops(True)  # accept other windows’ view states
        #self._install_view_tab()

        self.scroll.viewport().installEventFilter(self)
        self.label.installEventFilter(self)

        # shortcuts (A = toggle autostretch)
        QShortcut(QKeySequence("A"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Alt+Shift+A"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.toggle_mask_overlay)

        # re-render when the document changes
        self.document.changed.connect(lambda: self._render(rebuild=True))
        self._render(rebuild=True)
        QTimer.singleShot(0, self._maybe_announce_readout_help)


        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # so key press/release reach us

        self._mask_dot_enabled = self._active_mask_array() is not None
        self._active_title_prefix = False
        self._rebuild_title()
        self._watched_docs = set()

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
        if on == self.autostretch_enabled:
            return
        self.autostretch_enabled = on
        self.autostretchChanged.emit(on)
        self._render(rebuild=True)

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
        a_help = menu.addAction("Show pixel/WCS readout hint")  # ← NEW
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
        elif act == a_help:               # ← NEW
            self._announce_readout_help()

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
        """Package view state + doc pointer into a drag."""
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

    def set_autostretch(self, on: bool):
        self.autostretch_enabled = bool(on)
        self._recompute_autostretch_and_update()

    def _recompute_autostretch_and_update(self):
        self._qimg_src = None   # force source rebuild
        self._render(True)

    # ---------- rendering ----------
    def _render(self, rebuild: bool = False):
        if rebuild or self._qimg_src is None:
            img = self._display_override if (self._display_override is not None) else self.document.image
            if img is None:
                return
            arr = np.asarray(img)

            # --- Coerce non-image arrays (0-D/1-D) to 2-D so we can display them ---
            # e.g. vector HDU → show as a 1×N strip; scalar → 1×1 pixel
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr[np.newaxis, :]  # 1×N

            # detect mono now so we can ignore "linked" for mono images
            is_mono = (arr.ndim == 2) or (arr.ndim == 3 and arr.shape[2] == 1)

            if self.autostretch_enabled:
                # normalize common integer inputs to [0..1]
                if np.issubdtype(arr.dtype, np.integer):
                    info = np.iinfo(arr.dtype)
                    denom = float(max(1, info.max))
                    arr_f = (arr.astype(np.float32) / denom)
                else:
                    arr_f = arr.astype(np.float32, copy=False)
                    mx = float(arr_f.max()) if arr_f.size else 1.0
                    if mx > 5.0:  # wildly large floats → compress to [0..1]
                        arr_f = arr_f / mx

                vis = autostretch(
                    arr_f,
                    target_median=self.autostretch_target,
                    sigma=self.autostretch_sigma,
                    linked=(not is_mono and self._autostretch_linked),
                    use_16bit=None,
                )
            else:
                # true linear view → ensure we have 3 channels for display
                if arr.ndim == 2:
                    vis = np.stack([arr] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 1:
                    vis = np.repeat(arr, 3, axis=2)
                else:
                    vis = arr

            # convert to 8-bit RGB for QImage
            if vis.dtype == np.uint8:
                buf8 = vis
            elif vis.dtype == np.uint16:
                buf8 = (vis.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
            else:
                buf8 = (np.clip(vis, 0.0, 1.0) * 255.0).astype(np.uint8)

            # --- Final safety: force H×W×3 no matter what came in ---
            if buf8.ndim == 2:
                buf8 = np.stack([buf8] * 3, axis=-1)
            elif buf8.ndim == 3 and buf8.shape[2] == 1:
                buf8 = np.repeat(buf8, 3, axis=2)
            elif buf8.ndim != 3 or buf8.shape[2] < 3:
                # rare/odd cases: pad/truncate to 3 channels
                if buf8.ndim == 3 and buf8.shape[2] > 3:
                    buf8 = buf8[..., :3]
                else:
                    buf8 = np.stack([buf8.squeeze()] * 3, axis=-1)

            # --- MASK OVERLAY (unchanged logic) ---
            if self.show_mask_overlay:
                m = self._active_mask_array()
                if m is not None:
                    if getattr(self, "_mask_overlay_invert", True):
                        m = 1.0 - m
                    th, tw = buf8.shape[:2]
                    sh, sw = m.shape
                    if (sh, sw) != (th, tw):
                        yi = (np.linspace(0, sh - 1, th)).astype(np.int32)
                        xi = (np.linspace(0, sw - 1, tw)).astype(np.int32)
                        m = m[yi][:, xi]
                    a = m * float(getattr(self, "_mask_overlay_alpha", 0.35))
                    bf = buf8.astype(np.float32, copy=False)
                    bf[..., 0] = np.clip(bf[..., 0] + (255.0 - bf[..., 0]) * a, 0.0, 255.0)
                    buf8 = bf.astype(np.uint8, copy=False)
            # --- /MASK OVERLAY ---

            buf8 = np.ascontiguousarray(buf8)
            h, w, _ = buf8.shape
            bytes_per_line = buf8.strides[0]

            self._buf8 = buf8  # keep backing memory alive
            ptr = sip.voidptr(self._buf8.ctypes.data)
            try:
                self._qimg_src = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            except Exception:
                # absolute last-resort guard: build from a copy if direct wrap fails
                buf8c = np.array(self._buf8, copy=True, order="C")
                self._buf8 = buf8c
                ptr = sip.voidptr(self._buf8.ctypes.data)
                self._qimg_src = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # scale and show
        qimg = self._qimg_src
        if qimg is None:
            return

        sw = max(1, int(qimg.width() * self.scale))
        sh = max(1, int(qimg.height() * self.scale))
        scaled = qimg.scaled(
            sw, sh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(QPixmap.fromImage(scaled))
        self.label.resize(scaled.size())


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
        # 1) Ctrl + wheel → zoom
        if ev.type() == QEvent.Type.Wheel:
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                factor = 1.25 if ev.angleDelta().y() > 0 else 1/1.25
                self._zoom_at_anchor(factor)
                return True
            return False

        # 2) Space+click on label / viewport → start readout
        if ev.type() == QEvent.Type.MouseButtonPress:
            if self._space_down and ev.button() == Qt.MouseButton.LeftButton:
                vp_pos = obj.mapTo(self.scroll.viewport(), ev.pos())
                res = self._sample_image_at_viewport_pos(vp_pos)
                if res is not None:
                    xi, yi, sample = res
                    self._show_readout(xi, yi, sample)
                # go into live-probe mode
                self._readout_dragging = True
                self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                return True   # eat event so the label doesn't start selection/pan
            return False

        # 3) While dragging with space held → continuous readout
        if ev.type() == QEvent.Type.MouseMove:
            if self._readout_dragging:
                vp_pos = obj.mapTo(self.scroll.viewport(), ev.pos())
                res = self._sample_image_at_viewport_pos(vp_pos)
                if res is not None:
                    xi, yi, sample = res
                    self._show_readout(xi, yi, sample)
                return True
            return False

        # 4) Mouse release → stop live-probe
        if ev.type() == QEvent.Type.MouseButtonRelease:
            if self._readout_dragging:
                self._readout_dragging = False
                return True
            return False

        return super().eventFilter(obj, ev)


    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            # --- live readout mode ---
            if self._space_down:
                vp = self.scroll.viewport()
                vp_pos = vp.mapFrom(self, e.pos())
                res = self._sample_image_at_viewport_pos(vp_pos)
                if res is not None:
                    xi, yi, sample = res
                    self._show_readout(xi, yi, sample)
                # mark that we’re *not* panning, but we *are* reading out on move
                self._readout_dragging = True
                return

            # --- normal pan mode ---
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
        # --- live readout while dragging with space held ---
        if self._readout_dragging:
            vp = self.scroll.viewport()
            vp_pos = vp.mapFrom(self, e.pos())
            res = self._sample_image_at_viewport_pos(vp_pos)
            if res is not None:
                xi, yi, sample = res
                self._show_readout(xi, yi, sample)
            return

        # --- normal pan ---
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
        if e.button() == Qt.MouseButton.LeftButton:
            # end whichever mode we were in
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

        # proceed with your current teardown
        try:
            # emit your existing signal if you have it
            if hasattr(self, "aboutToClose"):
                self.aboutToClose.emit(doc)
        except Exception:
            pass
        super().closeEvent(e)

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
        print(f"[TableSubWindow] init rows={self._model.rowCount()} cols={self._model.columnCount()} title='{self.document.display_name()}'")
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
