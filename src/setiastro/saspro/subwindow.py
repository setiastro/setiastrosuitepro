# src/setiastro/saspro/subwindow.py
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import weakref
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import (
    QAbstractTableModel,
    QByteArray,
    QEvent,
    QMargins,
    QMimeData,
    QModelIndex,
    QPoint,
    QRect,
    QSettings,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
    QSortFilterProxyModel,
    QSignalBlocker,
)
from PyQt6.QtGui import (
    QCursor,
    QDrag,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QMdiSubWindow,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableView,
    QToolButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from .autostretch import autostretch
from .layers import BLEND_MODES, ImageLayer, composite_stack
from setiastro.saspro.dnd_mime import (
    MIME_ASTROMETRY,
    MIME_CMD,
    MIME_LINKVIEW,
    MIME_MASK,
    MIME_VIEWSTATE,
)
from setiastro.saspro.shortcuts import _unpack_cmd_payload
from setiastro.saspro.widgets.image_utils import ensure_contiguous


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
        self.setToolTip(self.tr(
            "Drag to duplicate/copy view.\n"
            "Hold Alt while dragging to LINK this view with another (live pan/zoom sync).\n"
            "Hold Shift while dragging to drop this image as a mask onto another view.\n"
            "Hold Ctrl while dragging to copy the astrometric solution (WCS) to another view."
        ))

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
            mods = QApplication.keyboardModifiers()
            if (mods & Qt.KeyboardModifier.AltModifier):
                self.owner._start_link_drag()
            elif (mods & Qt.KeyboardModifier.ShiftModifier):
                print("[DragTab] Shift+drag → start_mask_drag() from", id(self.owner))
                self.owner._start_mask_drag()
            elif (mods & Qt.KeyboardModifier.ControlModifier):
                self.owner._start_astrometry_drag()
            else:
                self.owner._start_viewstate_drag()

    def mouseReleaseEvent(self, ev):
        self._press_pos = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

MASK_GLYPH = "■"
#ACTIVE_PREFIX = "Active View: "
ACTIVE_PREFIX = ""
GLYPHS = "■●◆▲▪▫•◼◻◾◽🔗"
LINK_PREFIX = "🔗 "
DECORATION_PREFIXES = (
    LINK_PREFIX,                # "🔗 "
    f"{MASK_GLYPH} ",           # "■ "
    "Active View: ",            # legacy
)

_GLYPH_RE = re.compile(f"[{re.escape(GLYPHS)}]")

def _strip_ui_decorations(title: str) -> str:
    if not title:
        return ""
    s = str(title).strip()

    # Remove common prefix tag(s)
    s = s.replace("[LINK]", "").strip()

    # Remove glyphs anywhere (often used as status markers in titles)
    s = _GLYPH_RE.sub("", s)

    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

from astropy.wcs import WCS as _AstroWCS
from astropy.io.fits import Header as _FitsHeader

def build_celestial_wcs(header) -> _AstroWCS | None:
    """
    Given a FITS-like header or a dict with FITS keywords, return a *2-D celestial*
    astropy.wcs.WCS. Returns None if a sane celestial WCS cannot be recovered.
    Resilient to 3rd axes (RGB/STOKES) and SIP distortions.

    Accepted `header`:
      * astropy.io.fits.Header
      * dict of FITS cards (string->value)
      * dict containing {"FITSKeywords": {NAME: [{value: ..., comment: ...}], ...}}
    """
    if header is None:
        return None

    # (A) If we already got a WCS, try to coerce to celestial
    if isinstance(header, _AstroWCS):
        try:
            wc = getattr(header, "celestial", None)
            return wc if (wc is not None and getattr(wc, "naxis", 2) == 2) else header
        except Exception:
            return header

    # (B) Ensure we have a bona-fide FITS Header
    hdr_obj = None
    if isinstance(header, _FitsHeader):
        hdr_obj = header
    elif isinstance(header, dict):
        # XISF-style: {"FITSKeywords": {"CTYPE1":[{"value":"RA---TAN"}], ...}}
        if "FITSKeywords" in header and isinstance(header["FITSKeywords"], dict):
            from astropy.io.fits import Header
            hdr_obj = Header()
            for k, v in header["FITSKeywords"].items():
                if isinstance(v, list) and v:
                    val = v[0].get("value")
                    com = v[0].get("comment", "")
                    if val is not None:
                        try: hdr_obj[str(k)] = (val, com)
                        except Exception: hdr_obj[str(k)] = val
                elif v is not None:
                    try: hdr_obj[str(k)] = v
                    except Exception as e:
                        import logging
                        logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
        else:
            # Flat dict of FITS-like cards
            from astropy.io.fits import Header
            hdr_obj = Header()
            for k, v in header.items():
                try: hdr_obj[str(k)] = v
                except Exception as e:
                    import logging
                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

    if hdr_obj is None:
        return None

    # (C) Try full WCS first
    try:
        w = _AstroWCS(hdr_obj, relax=True)
        wc = getattr(w, "celestial", None)
        if wc is not None and getattr(wc, "naxis", 2) == 2:
            return wc
        if getattr(w, "has_celestial", False):
            return w.celestial
    except Exception:
        w = None

    # (D) Force a 2-axis interpretation (drop e.g. RGB axis)
    try:
        w2 = _AstroWCS(hdr_obj, relax=True, naxis=2)
        if getattr(w2, "has_celestial", False):
            return w2.celestial
    except Exception:
        pass

    # (E) As a last resort, scrub obvious axis-3 cards and retry
    try:
        hdr2 = hdr_obj.copy()
        for k in ("CTYPE3","CUNIT3","CRVAL3","CRPIX3",
                  "CD3_1","CD3_2","CD3_3","PC3_1","PC3_2","PC3_3"):
            if k in hdr2:
                del hdr2[k]
        w3 = _AstroWCS(hdr2, relax=True, naxis=2)
        if getattr(w3, "has_celestial", False):
            return w3.celestial
    except Exception:
        pass

    return None

def _compute_cropped_wcs(parent_hdr_like, x, y, w, h):
    """
    Build a cropped WCS header from parent_hdr_like and ROI (x,y,w,h).

    IMPORTANT:
    - If the parent header already describes a cropped ROI (NAXIS1/2 already
      equal to w/h, or the ROI is obviously outside the parent NAXIS), we
      *do not* shift CRPIX again. We just return a copy of the parent header,
      marking it as ROI-CROP if needed.
    """
    # Normalize ROI values to ints
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # Same helper as before; safe on dict/FITS Header
    try:
        from astropy.io.fits import Header
    except Exception:
        Header = None

    if Header is not None and isinstance(parent_hdr_like, Header):
        base = {k: parent_hdr_like.get(k) for k in parent_hdr_like.keys()}
    elif isinstance(parent_hdr_like, dict):
        fk = parent_hdr_like.get("FITSKeywords")
        if isinstance(fk, dict) and fk:
            base = {}
            for k, arr in fk.items():
                try:
                    base[k] = (arr or [{}])[0].get("value", None)
                except Exception:
                    pass
        else:
            base = dict(parent_hdr_like)
    else:
        base = {}

    # ------------------------------------------------------------------
    # Detect "already cropped" headers to avoid double-shifting CRPIX.
    # ------------------------------------------------------------------
    nax1 = base.get("NAXIS1")
    nax2 = base.get("NAXIS2")

    if isinstance(nax1, (int, float)) and isinstance(nax2, (int, float)):
        n1 = int(nax1)
        n2 = int(nax2)

        # Case A: parent already has same size as requested ROI,
        # but x,y are non-zero → this smells like ROI-of-ROI.
        if w == n1 and h == n2 and (x != 0 or y != 0):

            base["NAXIS1"], base["NAXIS2"] = n1, n2
            base.setdefault("CROPX", 0)
            base.setdefault("CROPY", 0)
            base.setdefault("SASKIND", "ROI-CROP")
            return base

        # Case B: ROI clearly outside parent dimensions → also treat as
        # "already cropped, don't touch CRPIX".
        if x >= n1 or y >= n2 or x + w > n1 or y + h > n2:

            base["NAXIS1"], base["NAXIS2"] = n1, n2
            base.setdefault("CROPX", 0)
            base.setdefault("CROPY", 0)
            base.setdefault("SASKIND", "ROI-CROP")
            return base

    # ------------------------------------------------------------------
    # Normal behavior: real crop relative to full-frame parent.
    # ------------------------------------------------------------------
    c1, c2 = base.get("CRPIX1"), base.get("CRPIX2")
    if isinstance(c1, (int, float)) and isinstance(c2, (int, float)):
        base["CRPIX1"] = float(c1) - float(x)
        base["CRPIX2"] = float(c2) - float(y)

    base["NAXIS1"], base["NAXIS2"] = w, h
    base["CROPX"], base["CROPY"] = x, y
    base["SASKIND"] = "ROI-CROP"
    return base

def _dnd_dbg_dump_state(tag: str, state: dict):
    try:
        import json
        # Keep it readable but complete
        keys = sorted(state.keys())
        print(f"\n[DNDDBG:{tag}] keys={keys}")
        for k in keys:
            v = state.get(k)
            # avoid huge blobs
            if isinstance(v, (dict, list)) and len(str(v)) > 400:
                print(f"  {k} = <{type(v).__name__} len={len(v)}>")
            else:
                print(f"  {k} = {v!r}")
        # show json size
        raw = json.dumps(state).encode("utf-8")
        print(f"[DNDDBG:{tag}] json_bytes={len(raw)} head={raw[:120]!r}")
    except Exception as e:
        print(f"[DNDDBG:{tag}] dump failed: {e}")

_DEBUG_DND_DUP = False

def _strip_ext_if_filename(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    base, ext = os.path.splitext(s)
    if ext and len(ext) <= 10:
        return base
    return s

class ImageSubWindow(QWidget):
    aboutToClose = pyqtSignal(object)
    autostretchChanged = pyqtSignal(bool)
    requestDuplicate = pyqtSignal(object)  # document
    layers_changed = pyqtSignal() 
    autostretchProfileChanged = pyqtSignal(str)
    viewTitleChanged = pyqtSignal(object, str)
    activeSourceChanged = pyqtSignal(object)  # None for full, or (x,y,w,h) for ROI
    viewTransformChanged = pyqtSignal(float, int, int)
    _registry = weakref.WeakValueDictionary()
    resized = pyqtSignal() 
    replayOnBaseRequested = pyqtSignal(object)


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
        self._link_emit_timer = QTimer(self)
        self._link_emit_timer.setSingleShot(True)
        self._link_emit_timer.setInterval(100)  # tweak 120–250ms to taste
        self._link_emit_timer.timeout.connect(self._emit_view_transform_now)
        self._suppress_link_emit = False  # guard while applying remote updates        
        self._link_squelch = False  # prevents feedback on linked apply
        self._pan_live = False
        self._linked_views = weakref.WeakSet()
        ImageSubWindow._registry[id(self)] = self
        self._link_badge_on = False
        s = getattr(self, "_settings", None) or QSettings()
        self._settings = s
        self._smooth_zoom = s.value("display/smooth_zoom_settle", True, type=bool)



        # whenever we move/zoom, relay to linked peers
        self.viewTransformChanged.connect(self._relay_to_linked)
        # pixel readout live-probe state
        self._space_down = False
        self._readout_dragging = False
        # Pinch gesture state (macOS trackpad)
        self._gesture_zoom_start = None
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
        self._preview_btn.setToolTip(self.tr("Create Preview: click, then drag on the image to define a preview rectangle."))
        self._preview_btn.setCheckable(True)
        self._preview_btn.clicked.connect(self._toggle_preview_select_mode)
        row.addWidget(self._preview_btn, 0, Qt.AlignmentFlag.AlignLeft)
        # — Undo / Redo just for this subwindow —
        self._btn_undo = QToolButton(self)
        self._btn_undo.setText("↶")  # or use an icon
        self._btn_undo.setToolTip(self.tr("Undo (this view)"))
        self._btn_undo.setEnabled(False)
        self._btn_undo.clicked.connect(self._on_local_undo)
        row.addWidget(self._btn_undo, 0, Qt.AlignmentFlag.AlignLeft)

        self._btn_redo = QToolButton(self)
        self._btn_redo.setText("↷")
        self._btn_redo.setToolTip(self.tr("Redo (this view)"))
        self._btn_redo.setEnabled(False)
        self._btn_redo.clicked.connect(self._on_local_redo)
        row.addWidget(self._btn_redo, 0, Qt.AlignmentFlag.AlignLeft)

        self._btn_replay_main = QToolButton(self)
        self._btn_replay_main.setText("⟳")  # pick any glyph you like
        self._btn_replay_main.setToolTip(self.tr(
            "Click: replay the last action on the base image.\n"
            "Arrow: pick a specific past action to replay on the base image."
        ))
        self._btn_replay_main.setEnabled(False)  # enabled only when preview + history

        # Left-click = your existing 'replay last on base'
        self._btn_replay_main.clicked.connect(self._on_replay_last_clicked)

        # NEW: dropdown menu listing all replayable actions
        self._replay_menu = QMenu(self)
        self._btn_replay_main.setMenu(self._replay_menu)
        self._btn_replay_main.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )

        row.addWidget(self._btn_replay_main, 0, Qt.AlignmentFlag.AlignLeft)


        # ── NEW: WCS grid toggle ─────────────────────────────────────────
        self._btn_wcs = QToolButton(self)
        self._btn_wcs.setText("⌗")
        self._btn_wcs.setToolTip(self.tr("Toggle WCS grid overlay (if WCS exists)"))
        self._btn_wcs.setCheckable(True)

        # Start OFF on every new view, regardless of WCS presence or past sessions
        self._show_wcs_grid = False
        self._btn_wcs.setChecked(False)

        self._btn_wcs.toggled.connect(self._on_toggle_wcs_grid)
        row.addWidget(self._btn_wcs, 0, Qt.AlignmentFlag.AlignLeft)
        # ─────────────────────────────────────────────────────────────────

        # ---- Inline view title (shown when the MDI subwindow is maximized) ----
        self._inline_title = QLabel(self)
        self._inline_title.setText("")
        self._inline_title.setToolTip(self.tr("Active view"))
        self._inline_title.setVisible(False)
        self._inline_title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._inline_title.setStyleSheet("""
            QLabel {
                padding-left: 8px;
                padding-right: 6px;
                font-size: 11px;
                color: rgba(255,255,255,0.80);
            }
        """)
        self._inline_title.setSizePolicy(
            self._inline_title.sizePolicy().horizontalPolicy(),
            self._inline_title.sizePolicy().verticalPolicy(),
        )

        # Push everything after this to the far right
        row.addStretch(1)
        row.addWidget(self._inline_title, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

        # (optional) tiny spacing to the edge
        row.addSpacing(6)

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
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)        
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.label)
        self.scroll.viewport().setMouseTracking(True)
        self.label.setMouseTracking(True)        
        full_v.addWidget(self.scroll)

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        for bar in (hbar, vbar):
            bar.valueChanged.connect(self._on_scroll_changed)
            bar.sliderMoved.connect(self._on_scroll_changed)
            bar.actionTriggered.connect(self._on_scroll_changed)

        # IMPORTANT: add the tab BEFORE connecting signals so currentChanged can't fire early
        self._full_tab_idx = self._tabs.addTab(full_host, self.tr("Full"))
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
        QShortcut(QKeySequence("F2"), self, activated=self._rename_document)
        QShortcut(QKeySequence("F3"), self, activated=self._rename_document)
        #QShortcut(QKeySequence("A"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Alt+Shift+A"), self, activated=self.toggle_autostretch)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.toggle_mask_overlay)

        # Re-render when the document changes
        self.document.changed.connect(lambda: self._render(rebuild=True))
        self._render(rebuild=True)
        QTimer.singleShot(0, self._maybe_announce_readout_help)
        self._refresh_local_undo_buttons()
        self._update_replay_button()

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        for bar in (hbar, vbar):
            bar.valueChanged.connect(self._schedule_emit_view_transform)
            bar.sliderMoved.connect(lambda _=None: self._schedule_emit_view_transform())
            bar.actionTriggered.connect(lambda _=None: self._schedule_emit_view_transform())

        # Mask/title adornments
        self._mask_dot_enabled = self._active_mask_array() is not None
        self._active_title_prefix = False
        self._rebuild_title()

        # Track docs used by layer stack (if any)
        self._watched_docs = set()
        self._history_doc = None
        self._install_history_watchers()

        QTimer.singleShot(0, self._install_mdi_state_watch)
        QTimer.singleShot(0, self._update_inline_title_and_buttons)

    def reload_display_settings(self):
        s = getattr(self, "_settings", None) or QSettings()
        self._settings = s

        # re-read any settings that are cached in the view
        self._smooth_zoom = s.value("display/smooth_zoom_settle", True, type=bool)
        self._autostretch_linked = s.value("display/stretch_linked", False, type=bool)

        # if you cache bg opacity or other display keys here, re-read them too
        # self._bg_opacity = s.value("display/bg_opacity", 50, type=int) / 100.0

        # force redraw
        self.update()

    # ----- link drag payload -----
    def _start_link_drag(self):
        """
        Alt + drag from ⧉: start a 'link these two views' drag.
        """
        payload = {
            "source_view_id": id(self),
        }
        # identity hints (not strictly required, but nice to have)
        try:
            payload.update(self._drag_identity_fields())
        except Exception:
            pass

        md = QMimeData()
        md.setData(MIME_LINKVIEW, QByteArray(json.dumps(payload).encode("utf-8")))
        drag = QDrag(self)
        drag.setMimeData(md)
        if self.label.pixmap():
            drag.setPixmap(self.label.pixmap().scaled(
                64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            drag.setHotSpot(QPoint(16, 16))
        drag.exec(Qt.DropAction.CopyAction)

    # ----- link management -----
    def link_to(self, other: "ImageSubWindow"):
        if other is self or other in self._linked_views:
            return

        # Gather the full sets (including each endpoint)
        a_group = set(self._linked_views) | {self}
        b_group = set(other._linked_views) | {other}
        merged = a_group | b_group

        # Clear old badges so we can reapply cleanly
        for v in merged:
            try:
                v._linked_views.discard(v)  # no-op safety
            except Exception:
                pass

        # Fully connect everyone to everyone
        for v in merged:
            v._linked_views.update(merged - {v})
            try:
                v._set_link_badge(True)
            except Exception:
                pass

        # Snap everyone to the initiator’s transform immediately
        try:
            s, h, v = self._current_transform()
            for peer in merged - {self}:
                peer.set_view_transform(s, h, v, from_link=True)
        except Exception:
            pass


    def unlink_from(self, other: "ImageSubWindow"):
        if other in self._linked_views:
            self._linked_views.discard(other)
            other._linked_views.discard(self)
        # clear badge if both are now free
        if not self._linked_views:
            self._set_link_badge(False)
        if not other._linked_views:
            other._set_link_badge(False)

    def unlink_all(self):
        peers = list(self._linked_views)
        for p in peers:
            self.unlink_from(p)

    def _relay_to_linked(self, scale: float, h: int, v: int):
        """
        When this view pans/zooms, nudge all linked peers. Guarded to avoid loops.
        """
        for peer in list(self._linked_views):
            try:
                peer.set_view_transform(scale, h, v, from_link=True)
            except Exception:
                pass

    def _set_link_badge(self, on: bool):
        self._link_badge_on = bool(on)
        self._rebuild_title()

    def _on_scroll_changed(self, *_):
        if self._suppress_link_emit:
            return
        # If we’re actively dragging, emit immediately for realtime follow
        if self._dragging or self._pan_live:
            self._emit_view_transform_now()
        else:
            self._schedule_emit_view_transform()

    def _current_transform(self):
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        return float(self.scale), int(hbar.value()), int(vbar.value())

    def _emit_view_transform(self):
        try:
            h = int(self.scroll.horizontalScrollBar().value())
            v = int(self.scroll.verticalScrollBar().value())
        except Exception:
            h = v = 0
        try:
            self.viewTransformChanged.emit(float(self.scale), h, v)
        except Exception:
            pass

    def _schedule_emit_view_transform(self):
        if self._suppress_link_emit:
            return
        # If we’re in a live pan, don’t debounce—emit now.
        if self._dragging or self._pan_live:
            self._emit_view_transform_now()
        else:
            self._link_emit_timer.start()

    def _emit_view_transform_now(self):
        if self._suppress_link_emit:
            return
        h = self.scroll.horizontalScrollBar().value()
        v = self.scroll.verticalScrollBar().value()
        try:
            self.viewTransformChanged.emit(float(self.scale), int(h), int(v))
        except Exception:
            pass

    # ------------------------------------------------------------
    # MDI maximize handling: show inline title + avoid duplicate buttons
    # ------------------------------------------------------------
    def _install_mdi_state_watch(self):
        sw = self._mdi_subwindow()
        if sw is None:
            return
        # Watch maximize/restore changes on the hosting QMdiSubWindow
        sw.installEventFilter(self)

    def _is_mdi_maximized(self) -> bool:
        sw = self._mdi_subwindow()
        if sw is None:
            return False
        try:
            return sw.isMaximized()
        except Exception:
            return False

    def _set_mdi_minmax_buttons_enabled(self, enabled: bool):
        return  # leave Qt default buttons alone

    def _current_view_title_for_inline(self) -> str:
        # Prefer your already-pretty title (strip decorations if needed).
        try:
            # If you have _current_view_title_for_drag already, reuse it:
            return self._current_view_title_for_drag()
        except Exception:
            pass
        try:
            return (self.windowTitle() or "").strip()
        except Exception:
            return ""

    def _update_inline_title_and_buttons(self):
        maximized = self._is_mdi_maximized()

        # Show inline title only when maximized (optional)
        try:
            self._inline_title.setVisible(maximized)
            if maximized:
                self._inline_title.setText(self._current_view_title_for_inline() or "Untitled")
        except Exception:
            pass

        # IMPORTANT: do NOT change QMdiSubWindow window flags.
        # Leaving them alone restores the default Qt "double button" behavior.

    #------ Replay helpers------
    def _update_replay_button(self):
        """
        Update the 'Replay on main image' button:

        - Enabled only when a Preview/ROI is active.
        - Populates the dropdown menu with all headless-history entries
          from the main window (newest first).
        """
        btn = getattr(self, "_btn_replay_main", None)
        if not btn:
            return

        # Do we have an active preview in this view?
        try:
            has_preview = self.has_active_preview()
        except Exception:
            has_preview = False

        mw = self._find_main_window()
        menu = getattr(self, "_replay_menu", None)

        history = []
        has_history = False

        # Pull history from main window if available
        if mw is not None and hasattr(mw, "get_headless_history"):
            try:
                history = mw.get_headless_history() or []
                has_history = bool(history)
            except Exception:
                history = []
                has_history = False

        # Rebuild the dropdown menu
        if menu is not None:
            menu.clear()
            if has_history:
                # We want newest actions at the *top* of the menu
                for idx_from_end, entry in enumerate(reversed(history)):
                    real_index = len(history) - 1 - idx_from_end  # index into original list

                    cid  = entry.get("command_id", "") or ""
                    desc = entry.get("description") or cid or f"#{real_index+1}"

                    act = menu.addAction(desc)
                    if cid and cid != desc:
                        act.setToolTip(cid)

                    # Capture the index in a default arg so each action gets its own index
                    act.triggered.connect(
                        lambda _chk=False, i=real_index: self._replay_history_index(i)
                    )

        # Also allow left-click "last action" when main window still has a last payload
        has_last = bool(mw and getattr(mw, "_last_headless_command", None))

        enabled = bool(has_preview and (has_history or has_last))
        btn.setEnabled(enabled)


    def _replay_history_index(self, index: int):
        """
        Called when the user selects an entry from the replay dropdown.

        We forward to MainWindow.replay_headless_history_entry_on_base(index, target_sw),
        which reuses the big replay_last_action_on_base() switchboard.
        """
        mw = self._find_main_window()
        if mw is None or not hasattr(mw, "replay_headless_history_entry_on_base"):
            try:
                print("[Replay] _replay_history_index: main window or handler missing")
            except Exception:
                pass
            return

        target_sw = self._mdi_subwindow()

        try:
            mw.replay_headless_history_entry_on_base(index, target_sw=target_sw)
            try:
                print(
                    f"[Replay] _replay_history_index: index={index}, "
                    f"view id={id(self)}, target_sw={id(target_sw) if target_sw else None}"
                )
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"[Replay] _replay_history_index failed: {e}")
            except Exception:
                pass


    def _on_replay_last_clicked(self):
        """
        User clicked the ⟳ button *main area* (not the arrow).

        This still does the old behavior:
        - Emit replayOnBaseRequested(view)
        - Main window then replays the *last* action on the base doc
          for this subwindow (via replay_last_action_on_base).
        """
        # DEBUG: log that the button actually fired
        try:
            roi = None
            if hasattr(self, "has_active_preview") and self.has_active_preview():
                try:
                    roi = self.current_preview_roi()
                except Exception:
                    roi = None
            print(
                f"[Replay] Button clicked in view id={id(self)}, "
                f"has_active_preview={self.has_active_preview() if hasattr(self, 'has_active_preview') else 'n/a'}, "
                f"roi={roi}"
            )
        except Exception:
            pass


        self.replayOnBaseRequested.emit(self)



    def _on_pan_or_zoom_changed(self, *_):
        # Debounce lightly if you want; for now, just emit
        self._emit_view_transform()

    def set_view_transform(self, scale, hval, vval, from_link=False):
        self._suppress_link_emit = True
        try:
            scale = float(max(self._min_scale, min(scale, self._max_scale)))

            scale_changed = (abs(scale - self.scale) > 1e-9)
            if scale_changed:
                self.scale = scale
                self._render(rebuild=False)  # fast present for responsiveness

                # ✅ NEW: schedule the final smooth redraw (same as main zoom path)
                if self._smooth_zoom:
                    self._request_zoom_redraw()

            hbar = self.scroll.horizontalScrollBar()
            vbar = self.scroll.verticalScrollBar()
            hv = int(hval); vv = int(vval)
            if hv != hbar.value():
                hbar.setValue(hv)
            if vv != vbar.value():
                vbar.setValue(vv)
        finally:
            self._suppress_link_emit = False

        if not from_link:
            self._schedule_emit_view_transform()


    def _on_toggle_wcs_grid(self, on: bool):
        self._show_wcs_grid = bool(on)
        QSettings().setValue("display/show_wcs_grid", self._show_wcs_grid)
        self._render(rebuild=True)  # repaint current frame



    def _install_history_watchers(self):
        # disconnect old history doc
        hd = getattr(self, "_history_doc", None)
        if hd is not None and hasattr(hd, "changed"):
            try:
                hd.changed.disconnect(self._on_history_doc_changed)
            except Exception:
                pass
            # in case older builds were wired directly:
            try:
                hd.changed.disconnect(self._refresh_local_undo_buttons)
            except Exception:
                pass

        # resolve new history doc (ROI when on Preview tab, else base)
        new_hd = self._resolve_history_doc()
        self._history_doc = new_hd

        # connect new
        if new_hd is not None and hasattr(new_hd, "changed"):
            try:
                new_hd.changed.connect(self._on_history_doc_changed)
            except Exception:
                pass

        # make the buttons correct right now
        self._refresh_local_undo_buttons()

    def _drag_identity_fields(self) -> dict:
        st = {}

        # existing identity (whatever you already do)
        try:
            doc = getattr(self, "document", None)
            st["doc_ptr"] = id(doc) if doc is not None else None
            st["doc_uid"] = getattr(doc, "uid", None)
            meta = getattr(doc, "metadata", {}) or {}
            st["file_path"] = (meta.get("file_path") or "").strip()
            st["base_doc_uid"] = meta.get("base_doc_uid") or st["doc_uid"]
            st["source_kind"] = meta.get("source_kind") or "full"
        except Exception:
            pass

        # ✅ NEW: add the current user-visible view title
        st["source_view_title"] = self._current_view_title_for_drag()

        # (optional) also include the subwindow title raw, for debugging/forensics
        try:
            sw = self._mdi_subwindow()
            st["source_sw_title_raw"] = (sw.windowTitle() if sw is not None else "")
        except Exception:
            st["source_sw_title_raw"] = ""

        return st


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
        self._full_tab_idx = self._tabs.addTab(full_host, self.tr("Full"))
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

        self._update_replay_button()     

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
        self._update_replay_button() 
        self._emit_view_transform() 
        mw = self._find_main_window()
        if mw is not None and getattr(mw, "_auto_fit_on_resize", False):
            try:
                mw._zoom_active_fit()
            except Exception:
                pass

    def _toggle_preview_select_mode(self, on: bool):
        self._preview_select_mode = bool(on)
        self._set_preview_cursor(self._preview_select_mode)
        if self._preview_select_mode:
            mw = self._find_main_window()
            if mw and hasattr(mw, "statusBar"):
                mw.statusBar().showMessage(self.tr("Preview mode: drag a rectangle on the image to create a preview."), 6000)
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
                sb.showMessage(self.tr("Press Space + Click/Drag to probe pixels (WCS shown if available)"), 8000)


      
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

    def _collect_layer_docs(self):
        """
        Collect unique ImageDocument objects referenced by the layer stack:
        - layer src_doc (if doc-backed)
        - layer mask_doc (if any)
        Raster/baked layers may have src_doc=None; those are ignored.
        Returns a LIST in a stable order (bottom→top traversal order), de-duped.
        """
        out = []
        seen = set()

        layers = getattr(self, "_layers", None) or []
        for L in layers:
            # 1) source doc (may be None for raster/baked layers)
            d = getattr(L, "src_doc", None)
            if d is not None:
                k = id(d)
                if k not in seen:
                    seen.add(k)
                    out.append(d)

            # 2) mask doc (also may be None)
            md = getattr(L, "mask_doc", None)
            if md is not None:
                k = id(md)
                if k not in seen:
                    seen.add(k)
                    out.append(md)

        return out


    def _reinstall_layer_watchers(self):
        """
        Reconnect layer source/mask document watchers to trigger live layer recomposite.
        Safe against:
        - raster/baked layers (src_doc=None)
        - deleted docs / partially-torn-down Qt objects
        - repeated calls
        """
        # Previous watchers
        olddocs = list(getattr(self, "_watched_docs", []) or [])

        # Disconnect old
        for d in olddocs:
            try:
                # Doc may already be deleted or signal gone
                d.changed.disconnect(self._on_layer_source_changed)
            except Exception:
                pass

        # Collect new
        newdocs = self._collect_layer_docs()

        # Connect new
        for d in newdocs:
            try:
                d.changed.connect(self._on_layer_source_changed)
            except Exception:
                pass

        # Store as list (stable)
        self._watched_docs = newdocs



    def toggle_mask_overlay(self):
        self.show_mask_overlay = not self.show_mask_overlay
        self._render(rebuild=True)

    def _rebuild_title(self, *, base: str | None = None):
        sub = self._mdi_subwindow()
        if not sub:
            return

        if base is None:
            base = self._effective_title() or self.tr("Untitled")

        # Strip badges (🔗, ■, etc) AND "Active View:" prefix
        core, _ = self._strip_decorations(base)

        # ALSO strip file extensions if it looks like a filename
        # (this prevents .tiff/.fit coming back via any fallback path)
        try:
            b, ext = os.path.splitext(core)
            if ext and len(ext) <= 10 and not core.endswith("..."):
                core = b
        except Exception:
            pass

        # Build the displayed title with badges
        shown = core
        if getattr(self, "_link_badge_on", False):
            shown = f"{LINK_PREFIX}{shown}"
        if self._mask_dot_enabled:
            shown = f"{MASK_GLYPH} {shown}"

        # Update chrome
        if shown != sub.windowTitle():
            sub.setWindowTitle(shown)
            sub.setToolTip(shown)

        # IMPORTANT: emit ONLY the clean core (no badges, no extensions)
        if core != self._last_title_for_emit:
            self._last_title_for_emit = core
            try:
                self.viewTitleChanged.emit(self, core)
            except Exception:
                pass



    def _strip_decorations(self, title: str) -> tuple[str, bool]:
        had = False
        # loop to remove multiple stacked badges, in any order
        while True:
            changed = False

            # A) explicit multi-char prefixes
            for pref in DECORATION_PREFIXES:
                if title.startswith(pref):
                    title = title[len(pref):]
                    had = changed = True

            # B) generic 1-glyph + space (covers any stray glyph in GLYPHS)
            if len(title) >= 2 and title[1] == " " and title[0] in GLYPHS:
                title = title[2:]
                had = changed = True

            if not changed:
                break

        return title, had


    def set_active_highlight(self, on: bool):
        self._is_active_flag = bool(on)
        return


    def _set_mask_highlight(self, on: bool):
        self._mask_dot_enabled = bool(on)
        self._rebuild_title()

    def _sync_host_title(self):
        # document renamed → rebuild from flags + new base
        self._rebuild_title()



    def base_doc_title(self) -> str:
        """The clean, base title (document display name), no prefixes/suffixes."""
        return self.document.display_name() or self.tr("Untitled")

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

    def _effective_title(self) -> str:
        """
        Returns the *core* title for this view (no UI badges like 🔗/■, and no file extension).
        Badges are added later by _rebuild_title().
        """
        # 1) Prefer per-view override if set
        t = (self._view_title_override or "").strip()

        # 2) Else prefer metadata display_name (what duplicate/rename should set)
        if not t:
            try:
                md = getattr(self.document, "metadata", {}) or {}
                t = (md.get("display_name") or "").strip()
            except Exception:
                t = ""

        # 3) Else fall back to doc.display_name()
        if not t:
            try:
                t = (self.document.display_name() or "").strip()
            except Exception:
                t = ""

        t = t or self.tr("Untitled")

        # Strip UI decorations (🔗, ■, Active View:, etc.)
        try:
            t, _ = self._strip_decorations(t)
        except Exception:
            pass

        # Strip extension if it looks like a filename
        try:
            base, ext = os.path.splitext(t)
            if ext and len(ext) <= 10:
                t = base
        except Exception:
            pass

        return t


    def _show_ctx_menu(self, pos):
        menu = QMenu(self)
        #a_view = menu.addAction(self.tr("Rename View… (F2)"))
        a_doc  = menu.addAction(self.tr("Rename Document…  (F2)"))
        menu.addSeparator()
        a_min  = menu.addAction(self.tr("Send to Shelf"))
        a_clear = menu.addAction(self.tr("Clear View Name (use doc name)"))
        menu.addSeparator()
        a_unlink = menu.addAction(self.tr("Unlink from Linked Views"))   # ← NEW
        menu.addSeparator()
        a_help = menu.addAction(self.tr("Show pixel/WCS readout hint"))
        menu.addSeparator()
        a_prev = menu.addAction(self.tr("Create Preview (drag rectangle)"))
        # --- Mask actions (requested in zoom/context menu) ---
        mw = self._find_main_window()
        vw = self  # this ImageSubWindow is the view
        doc = getattr(vw, "document", None)

        has_mask = bool(doc and getattr(doc, "active_mask_id", None))

        menu.addSeparator()
        menu.addSection(self.tr("Mask"))

        # 1) Toggle overlay (single item: Show/Hide)
        a_mask_overlay = menu.addAction(self.tr("Show Mask Overlay"))
        a_mask_overlay.setCheckable(True)
        a_mask_overlay.setChecked(bool(getattr(vw, "show_mask_overlay", False)))
        a_mask_overlay.setEnabled(has_mask)

        # 2) Invert mask
        a_invert = menu.addAction(self.tr("Invert Mask"))
        a_invert.setEnabled(has_mask)

        # 3) Remove mask
        a_remove = menu.addAction(self.tr("Remove Mask"))
        a_remove.setEnabled(has_mask)

        act = menu.exec(self.mapToGlobal(pos))

        if act == a_doc:
            self._rename_document()
        elif act == a_min:
            self._send_to_shelf()
        elif act == a_clear:
            self._view_title_override = None
            self._sync_host_title()
        elif act == a_unlink:
            self.unlink_all()
        elif act == a_help:
            self._announce_readout_help()
        elif act == a_prev:
            self._preview_btn.setChecked(True)
            self._toggle_preview_select_mode(True)
        # --- Mask dispatch ---
        elif act == a_mask_overlay:
            if mw:
                if a_mask_overlay.isChecked():
                    mw._show_mask_overlay()
                else:
                    mw._hide_mask_overlay()
        elif act == a_invert:
            if mw:
                mw._invert_mask()
        elif act == a_remove:
            if mw:
                mw._remove_mask_menu()


    def _send_to_shelf(self):
        sub = self._mdi_subwindow()
        mw  = self._find_main_window()
        if sub and mw and hasattr(mw, "window_shelf"):
            sub.hide()
            mw.window_shelf.add_entry(sub)

    def _rename_view(self):
        """LEGACY: View rename removed. Keep as shim so older calls don't break."""
        return self._rename_document()


    def _rename_document(self):
        current = self.document.display_name()
        new, ok = QInputDialog.getText(
            self, self.tr("Rename Document"), self.tr("New name:"), text=current
        )
        if not (ok and new.strip()):
            return

        new_name = new.strip()
        if new_name == current:
            return

        self.document.metadata["display_name"] = new_name
        self.document.changed.emit()  # everyone listening updates
        self._sync_host_title()       # update this subwindow title immediately

        mw = self._find_main_window()
        if mw and hasattr(mw, "layers_dock") and mw.layers_dock:
            try:
                mw.layers_dock._refresh_titles_only()
            except Exception:
                pass


    def set_scale(self, s: float):
        # Programmatic scale changes must schedule final smooth redraw
        s = float(max(self._min_scale, min(s, self._max_scale)))
        if abs(s - self.scale) < 1e-9:
            return
        self.scale = s
        self._render()                 # fast present happens here
        self._schedule_emit_view_transform()

        # ✅ NEW: ensure we do the final smooth redraw (same as manual zoom)
        if self._smooth_zoom:
            self._request_zoom_redraw()



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
            "doc_ptr": id(self.document),
            "scale": float(self.scale),
            "hval": int(hbar.value()),
            "vval": int(vbar.value()),
            "autostretch": bool(self.autostretch_enabled),
            "autostretch_target": float(self.autostretch_target),
        }
        state.update(self._drag_identity_fields())

        roi = None
        try:
            if hasattr(self, "has_active_preview") and self.has_active_preview():
                r = self.current_preview_roi()
                if r and len(r) == 4:
                    roi = tuple(map(int, r))
        except Exception:
            roi = None

        if roi:
            state["roi"] = roi
            state["source_kind"] = "roi-preview"
            try:
                pname = self.current_preview_name()
            except Exception:
                pname = None
            if pname:
                state["preview_name"] = str(pname)
        else:
            state["source_kind"] = "full"

        if _DEBUG_DND_DUP:
            _dnd_dbg_dump_state("DRAG_START:dragtab", state)


        md = QMimeData()
        md.setData(MIME_VIEWSTATE, QByteArray(json.dumps(state).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(md)

        pm = self.label.pixmap()
        if pm and not pm.isNull():
            drag.setPixmap(
                pm.scaled(
                    96, 96,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            drag.setHotSpot(QPoint(16, 16))  # optional, but feels nicer

        drag.exec(Qt.DropAction.CopyAction)




    def _start_mask_drag(self):
        """
        Start a drag that carries 'this document is a mask' to drop targets.
        """
        doc = self.document
        if doc is None:
            return

        payload = {
            # New-style field
            "mask_doc_ptr": id(doc),

            # Backward-compat field: many handlers still look for 'doc_ptr'
            "doc_ptr": id(doc),

            "mode": "replace",       # future: "union"/"intersect"/"diff"
            "invert": False,
            "feather": 0.0,          # px
            "name": doc.display_name(),
        }

        # Add identity hints (uids, base uid, file_path)
        payload.update(self._drag_identity_fields())

        md = QMimeData()
        md.setData(MIME_MASK, QByteArray(json.dumps(payload).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(md)
        if self.label.pixmap():
            drag.setPixmap(
                self.label.pixmap().scaled(
                    64, 64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
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
        self._emit_view_transform() 


    # ---- DnD 'view tab' -------------------------------------------------

    def _mdi_subwindow(self):
        """Return the QMdiSubWindow that hosts this view, or None."""
        try:
            from PyQt6.QtWidgets import QMdiSubWindow
            p = self.parent()
            while p is not None:
                if isinstance(p, QMdiSubWindow):
                    return p
                p = p.parent()
        except Exception:
            pass
        return None

    def _current_view_title_for_drag(self) -> str:
        """
        The *actual* user-visible view title (what they renamed to),
        NOT the document/file name.
        """
        title = ""
        try:
            sw = self._mdi_subwindow()
            if sw is not None:
                title = (sw.windowTitle() or "").strip()
        except Exception:
            title = ""

        if not title:
            try:
                title = (self.windowTitle() or "").strip()
            except Exception:
                title = ""

        if not title:
            # absolute fallback
            try:
                title = (self.document.display_name() or "").strip()
            except Exception:
                title = ""

        # Optional: strip [LINK], glyphs, etc if your title includes those
        try:
            title = _strip_ui_decorations(title)
        except Exception:
            pass

        return title or "Untitled"


    def _install_view_tab(self):
        self._view_tab = QToolButton(self)
        self._view_tab.setText(self.tr("View"))
        self._view_tab.setToolTip(self.tr("Drag onto another window to copy zoom/pan.\n"
                                  "Double-click to duplicate this view."))
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

        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        # NEW: capture the *current view title* the user sees
        view_title = self._current_view_display_name()

        state = {
            "doc_ptr": id(self.document),
            "doc_uid": getattr(self.document, "uid", None),   # harmless even if None
            "file_path": (getattr(self.document, "metadata", {}) or {}).get("file_path", ""),
            "scale": float(self.scale),
            "hval": int(hbar.value()),
            "vval": int(vbar.value()),
            "autostretch": bool(self.autostretch_enabled),
            "autostretch_target": float(self.autostretch_target),

            # NEW: this is what we will use for naming duplicates
            "source_view_title": view_title,
        }
        state.update(self._drag_identity_fields())
        if _DEBUG_DND_DUP:
            _dnd_dbg_dump_state("DRAG_START:viewtab", state)
        mime = QMimeData()
        mime.setData(MIME_VIEWSTATE, QByteArray(json.dumps(state).encode("utf-8")))

        drag = QDrag(self)
        drag.setMimeData(mime)

        pm = self.label.pixmap()
        if pm:
            drag.setPixmap(pm.scaled(
                96, 96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
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
                or md.hasFormat(MIME_CMD)
                or md.hasFormat(MIME_LINKVIEW)):
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        md = ev.mimeData()

        if (md.hasFormat(MIME_VIEWSTATE)
                or md.hasFormat(MIME_ASTROMETRY)
                or md.hasFormat(MIME_MASK)
                or md.hasFormat(MIME_CMD)
                or md.hasFormat(MIME_LINKVIEW)):
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

        if md.hasFormat(MIME_LINKVIEW):
            try:
                payload = json.loads(bytes(md.data(MIME_LINKVIEW)).decode("utf-8"))
                sid = int(payload.get("source_view_id"))
            except Exception:
                ev.ignore(); return
            src = ImageSubWindow._registry.get(sid)
            if src is not None and src is not self:
                src.link_to(self)
                ev.acceptProposedAction()
            else:
                ev.ignore()
            return

        ev.ignore()

    def _current_view_display_name(self) -> str:
        """
        Best-effort: the exact title the user sees for THIS subwindow/view.
        Prefer QMdiSubWindow title, fallback to document display_name.
        """
        # 1) QMdiSubWindow title (what user sees)
        try:
            sw = self._mdi_subwindow()
            if sw is not None:
                t = (sw.windowTitle() or "").strip()
                if t:
                    return t
        except Exception:
            pass

        # 2) This widget's own windowTitle (sometimes used)
        try:
            t = (self.windowTitle() or "").strip()
            if t:
                return t
        except Exception:
            pass

        # 3) Document display name fallback
        try:
            d = getattr(self, "document", None)
            if d is not None and hasattr(d, "display_name"):
                t = (d.display_name() or "").strip()
                if t:
                    return t
        except Exception:
            pass

        return "Untitled"


    # keep the tab visible if the widget resizes
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        try:
            self.resized.emit()
        except Exception:
            pass        
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

    def _on_history_doc_changed(self):
        """
        Called when the current history document (full or ROI) changes.
        Ensures the pixmap is rebuilt immediately, including when a
        tool operates on a Preview/ROI doc.
        """
        QTimer.singleShot(0, lambda: (self._render(rebuild=True),
                                      self._refresh_local_undo_buttons()))

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

    def _deg_to_hms(self, ra_deg: float) -> str:
        """RA in degrees → 'HH:MM:SS' (rounded secs, with carry)."""
        ra_h = ra_deg / 15.0
        hh = int(ra_h) % 24
        mmf = (ra_h - hh) * 60.0
        mm = int(mmf)
        ss = int(round((mmf - mm) * 60.0))
        if ss == 60:
            ss = 0; mm += 1
        if mm == 60:
            mm = 0; hh = (hh + 1) % 24
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _deg_to_dms(self, dec_deg: float) -> str:
        """Dec in degrees → '±DD:MM:SS' (rounded secs, with carry)."""
        sign = "+" if dec_deg >= 0 else "-"
        d = abs(dec_deg)
        dd = int(d)
        mf = (d - dd) * 60.0
        mm = int(mf)
        ss = int(round((mf - mm) * 60.0))
        if ss == 60:
            ss = 0; mm += 1
        if mm == 60:
            mm = 0; dd += 1
        return f"{sign}{dd:02d}:{mm:02d}:{ss:02d}"

    def copy_viewport_to_clipboard_image(self) -> QImage | None:
        """
        Returns a QImage of exactly what the user is currently seeing:
        - includes autostretch + mask overlay (because it uses the rendered pixmap)
        - includes current zoom scale + pan position (viewport crop)
        - works for Full tab and Preview tabs (because _pm_src is built from active source)
        """
        # Make sure we have something rendered
        pm = getattr(self, "_pm_src", None)
        if pm is None or pm.isNull():
            # Try to build it once (slow path) if needed
            try:
                self._render(rebuild=True)
                pm = getattr(self, "_pm_src", None)
            except Exception:
                pm = None

        if pm is None or pm.isNull():
            return None

        # We need the *currently displayed* pixmap size, not the source pixmap
        # Because _present_scaled likely sets label pixmap to a scaled version.
        shown_pm = self.label.pixmap()
        if shown_pm is None or shown_pm.isNull():
            # fallback to unscaled source if label doesn't have one
            shown_pm = pm

        vp = self.scroll.viewport()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        # Visible rect in label coordinates:
        # scrollbars are offsets into the label widget.
        x = int(hbar.value())
        y = int(vbar.value())
        w = int(vp.width())
        h = int(vp.height())

        # Clamp crop to pixmap bounds (important when image smaller than viewport)
        px_w = int(shown_pm.width())
        px_h = int(shown_pm.height())

        if px_w <= 0 or px_h <= 0:
            return None

        if x < 0: x = 0
        if y < 0: y = 0
        if x >= px_w or y >= px_h:
            return None

        w = min(w, px_w - x)
        h = min(h, px_h - y)
        if w <= 0 or h <= 0:
            return None

        cropped = shown_pm.copy(QRect(x, y, w, h))
        if cropped.isNull():
            return None

        return cropped.toImage()

    # ---------- rendering ----------
    def _render(self, rebuild: bool = False):
        #print("[ImageSubWindow] _render called, rebuild =", rebuild)
        """
        Render the current view.

        Fast path:
        - rebuild=False: only rescale already-built pixmap/QImage (NO numpy work).
        Slow path:
        - rebuild=True: rebuild visualization (autostretch, 8-bit conversion, overlays),
                        refresh QImage/QPixmap cache, then present.

        Rules:
        - If a Preview is active, FIRST sync that preview's stored arr from the
        DocManager's ROI document (the thing tools actually modify), then render.
        - Never reslice from the parent/full image here.
        - Keep a strong reference to the numpy buffer that backs the QImage.
        """
        # ---- GUARD: widget/label may be deleted but document.changed still fires ----
        try:
            from PyQt6 import sip as _sip
            if _sip.isdeleted(self):
                return
            lbl = getattr(self, "label", None)
            if lbl is None or _sip.isdeleted(lbl):
                return
        except Exception:
            if not hasattr(self, "label"):
                return
        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # FAST PATH: if we're not rebuilding content and we already have a source pixmap,
        # just present scaled (fast). This is the key to smooth zoom.
        # ---------------------------------------------------------------------------
        if (not rebuild) and getattr(self, "_pm_src", None) is not None:
            self._present_scaled(interactive=True)
            return

        # ---------------------------
        # 1) Choose & sync source arr
        # ---------------------------
        base_img = None
        if self._active_source_kind == "preview" and self._active_preview_id is not None:
            src = next((p for p in self._previews if p["id"] == self._active_preview_id), None)
            if src is not None:
                # Pull the *edited* ROI image from DocManager, if available
                if hasattr(self, "_docman") and self._docman is not None:
                    try:
                        roi_doc = self._docman.get_document_for_view(self)
                        roi_img = getattr(roi_doc, "image", None)
                        # IMPORTANT: only copy on rebuild; zoom should not trigger a copy
                        if roi_img is not None:
                            if rebuild or ("arr" not in src) or (src.get("arr") is None):
                                src["arr"] = np.asarray(roi_img).copy()
                    except Exception:
                        print("[ImageSubWindow] _render: failed to pull edited ROI from DocManager")
                base_img = src.get("arr", None)
        else:
            base_img = self._display_override if (self._display_override is not None) else (
                getattr(self.document, "image", None)
            )

        if base_img is None:
            self._qimg_src = None
            self._pm_src = None
            self._pm_src_wcs = None
            self._buf8 = None
            self.label.clear()
            return

        arr = np.asarray(base_img)

        # ---------------------------------------
        # 2) Normalize dimensionality and dtype
        # ---------------------------------------
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
                if mx > 5.0:
                    arr_f = arr_f / mx

            vis = autostretch(
                arr_f,
                target_median=self.autostretch_target,
                sigma=self.autostretch_sigma,
                linked=(not is_mono and self._autostretch_linked),
                use_24bit=None,
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
        # 5) Optional mask overlay (baked into buf8)
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
        if buf8.dtype != np.uint8:
            buf8 = buf8.astype(np.uint8)

        buf8 = ensure_contiguous(buf8)
        h, w, c = buf8.shape
        bytes_per_line = int(w * 3)

        self._buf8 = buf8  # keep alive

        try:
            addr = int(self._buf8.ctypes.data)
            ptr  = sip.voidptr(addr)
            qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            if qimg is None or qimg.isNull():
                raise RuntimeError("QImage null")
        except Exception:
            buf8c = np.array(self._buf8, copy=True, order="C")
            self._buf8 = buf8c
            addr = int(self._buf8.ctypes.data)
            ptr  = sip.voidptr(addr)
            qimg = QImage(ptr, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self._qimg_src = qimg
        if qimg is None or qimg.isNull():
            self._pm_src = None
            self._pm_src_wcs = None
            self.label.clear()
            return

        # Cache unscaled pixmap ONCE per rebuild
        self._pm_src = QPixmap.fromImage(self._qimg_src)

        # Invalidate any cached “WCS baked” pixmap on rebuild
        self._pm_src_wcs = None

        # Present final-quality after rebuild
        self._present_scaled(interactive=False)

        rebuild = False  # done


    def _present_scaled(self, interactive: bool):
        """
        Present the cached source pixmap scaled to current self.scale.

        interactive=True:
        - Fast scaling
        - No WCS draw
        interactive=False:
        - Smooth scaling
        - Optionally draw WCS overlay once
        """
        if getattr(self, "_pm_src", None) is None:
            return

        pm_base = self._pm_src

        sw = max(1, int(pm_base.width() * self.scale))
        sh = max(1, int(pm_base.height() * self.scale))

        mode = Qt.TransformationMode.FastTransformation if interactive else Qt.TransformationMode.SmoothTransformation
        pm_scaled = pm_base.scaled(sw, sh, Qt.AspectRatioMode.KeepAspectRatio, mode)

        # If interactive, skip WCS overlay entirely (this is the biggest speed win)
        if interactive:
            self.label.setPixmap(pm_scaled)
            self.label.resize(pm_scaled.size())
            return

        # Non-interactive: (optionally) draw WCS grid.
        if getattr(self, "_show_wcs_grid", False):
            # Cache a baked WCS pixmap at *this* scale to avoid re-drawing
            # if _present_scaled(False) is called multiple times at same scale.
            cache_key = (sw, sh, float(self.scale))
            if getattr(self, "_pm_src_wcs_key", None) != cache_key or getattr(self, "_pm_src_wcs", None) is None:
                pm_scaled = self._draw_wcs_grid_on_pixmap(pm_scaled)
                self._pm_src_wcs = pm_scaled
                self._pm_src_wcs_key = cache_key
            else:
                pm_scaled = self._pm_src_wcs

        self.label.setPixmap(pm_scaled)
        self.label.resize(pm_scaled.size())


    def _draw_wcs_grid_on_pixmap(self, pm_scaled: QPixmap) -> QPixmap:
        """
        Your existing WCS painter logic, moved to operate on a QPixmap (already scaled).
        Runs ONLY on non-interactive redraw.
        """
        wcs2 = self._get_celestial_wcs()
        if wcs2 is None:
            return pm_scaled

        from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush
        from PyQt6.QtCore import QSettings, QRect
        from astropy.wcs.utils import proj_plane_pixel_scales
        import numpy as _np

        _settings = getattr(self, "_settings", None) or QSettings()
        pref_enabled   = _settings.value("wcs_grid/enabled", True, type=bool)
        pref_mode      = _settings.value("wcs_grid/mode", "auto", type=str)
        pref_step_unit = _settings.value("wcs_grid/step_unit", "deg", type=str)
        pref_step_val  = _settings.value("wcs_grid/step_value", 1.0, type=float)

        if not pref_enabled:
            return pm_scaled

        # Determine full image geometry from the CURRENT SOURCE buffer (not pm_scaled)
        # We can infer W/H from qimg src (original)
        if getattr(self, "_qimg_src", None) is None:
            return pm_scaled
        H_full = int(self._qimg_src.height())
        W_full = int(self._qimg_src.width())

        # Pixel scales/FOV
        px_scales_deg = proj_plane_pixel_scales(wcs2)
        px_deg = float(max(px_scales_deg[0], px_scales_deg[1]))
        fov_deg = px_deg * float(max(W_full, H_full))

        if pref_mode == "fixed":
            step_deg = float(pref_step_val if pref_step_unit == "deg" else (pref_step_val / 60.0))
            step_deg = max(1e-6, min(step_deg, 90.0))
        else:
            nice = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30]
            target_lines = 8
            desired = max(fov_deg / target_lines, px_deg * 100)
            step_deg = min((n for n in nice if n >= desired), default=30)

        # World bounds from corners
        corners = _np.array([[0, 0], [W_full-1, 0], [0, H_full-1], [W_full-1, H_full-1]], dtype=float)
        try:
            ra_c, dec_c = wcs2.pixel_to_world_values(corners[:,0], corners[:,1])
            ra_min = float(_np.nanmin(ra_c));  ra_max = float(_np.nanmax(ra_c))
            dec_min = float(_np.nanmin(dec_c)); dec_max = float(_np.nanmax(dec_c))
            if ra_max - ra_min > 300:
                ra_c_wrapped = _np.mod(ra_c + 180.0, 360.0)
                ra_min = float(_np.nanmin(ra_c_wrapped)); ra_max = float(_np.nanmax(ra_c_wrapped))
                ra_shift = 180.0
            else:
                ra_shift = 0.0
        except Exception:
            ra_min, ra_max, dec_min, dec_max, ra_shift = 0.0, 360.0, -90.0, 90.0, 0.0

        pm = QPixmap(pm_scaled)  # copy so we don’t mutate caller
        p = QPainter(pm)
        pen = QPen(QColor(255, 255, 255, 140))
        pen.setWidth(1)
        p.setPen(pen)

        # Scale factor between full-res image and pm_scaled
        s = float(pm.width()) / float(max(1, W_full))

        Wf, Hf = float(W_full), float(H_full)

        def draw_world_poly(xs_world, ys_world):
            try:
                px, py = wcs2.world_to_pixel_values(xs_world, ys_world)
            except Exception:
                return

            px = _np.asarray(px, dtype=float)
            py = _np.asarray(py, dtype=float)

            ok = _np.isfinite(px) & _np.isfinite(py)
            margin = float(max(Wf, Hf) * 2.0)
            ok &= (px > -margin) & (px < (Wf - 1.0 + margin))
            ok &= (py > -margin) & (py < (Hf - 1.0 + margin))

            for i in range(1, len(px)):
                if not (ok[i-1] and ok[i]):
                    continue
                x0 = float(px[i-1]) * s
                y0 = float(py[i-1]) * s
                x1 = float(px[i])   * s
                y1 = float(py[i])   * s
                if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 2.0e9:
                    continue
                p.drawLine(int(x0), int(y0), int(x1), int(y1))

        ra_samples = _np.linspace(ra_min, ra_max, 512, dtype=float)
        ra_samples_wrapped = _np.mod(ra_samples + ra_shift, 360.0) if ra_shift else ra_samples
        dec_samples = _np.linspace(dec_min, dec_max, 512, dtype=float)

        def _frange(a, b, sstep):
            out = []
            x = a
            while x <= b + 1e-9:
                out.append(x)
                x += sstep
            return out

        def _round_to(x, sstep):
            return sstep * round(x / sstep)

        ra_start  = _round_to(ra_min, step_deg)
        dec_start = _round_to(dec_min, step_deg)

        for dec in _frange(dec_start, dec_max, step_deg):
            dec_arr = _np.full_like(ra_samples_wrapped, dec)
            draw_world_poly(ra_samples_wrapped, dec_arr)

        for ra in _frange(ra_start, ra_max, step_deg):
            ra_arr = _np.full_like(dec_samples, (ra + ra_shift) % 360.0)
            draw_world_poly(ra_arr, dec_samples)

        # Labels
        font = QFont()
        font.setPixelSize(11)
        p.setFont(font)
        text_pen  = QPen(QColor(255, 255, 255, 230))
        box_brush = QBrush(QColor(0, 0, 0, 140))
        p.setPen(text_pen)

        img_w = pm.width()
        img_h = pm.height()

        def _draw_label(x, y, txt, anchor="lt"):
            if not _np.isfinite([x, y]).all():
                return
            fm = p.fontMetrics()
            wtxt = fm.horizontalAdvance(txt) + 6
            htxt = fm.height() + 4

            if anchor == "lt":
                rx, ry = int(x) + 4, int(y) + 3
            elif anchor == "rt":
                rx, ry = int(x) - wtxt - 4, int(y) + 3
            elif anchor == "lb":
                rx, ry = int(x) + 4, int(y) - htxt - 3
            else:
                rx, ry = int(x) - wtxt // 2, int(y) + 3

            rx = max(0, min(rx, img_w - wtxt - 1))
            ry = max(0, min(ry, img_h - htxt - 1))

            rect = QRect(rx, ry, wtxt, htxt)
            p.save()
            p.setBrush(box_brush)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(rect, 4, 4)
            p.restore()
            p.drawText(rect.adjusted(3, 2, -3, -2),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)

        # DEC labels on left edge
        for dec in _frange(dec_start, dec_max, step_deg):
            try:
                x_pix, y_pix = wcs2.world_to_pixel_values((ra_min + ra_shift) % 360.0, dec)
                if not _np.isfinite([x_pix, y_pix]).all():
                    continue
                x_pix = min(max(x_pix, 0.0), Wf - 1.0)
                y_pix = min(max(y_pix, 0.0), Hf - 1.0)
                _draw_label(x_pix * s, y_pix * s, self._deg_to_dms(dec), anchor="lt")
            except Exception:
                pass

        # RA labels on top edge
        for ra in _frange(ra_start, ra_max, step_deg):
            ra_wrapped = (ra + ra_shift) % 360.0
            try:
                x_pix, y_pix = wcs2.world_to_pixel_values(ra_wrapped, dec_min)
                if not _np.isfinite([x_pix, y_pix]).all():
                    continue
                x_pix = min(max(x_pix, 0.0), Wf - 1.0)
                y_pix = min(max(y_pix, 0.0), Hf - 1.0)
                _draw_label(x_pix * s, y_pix * s, self._deg_to_hms(ra_wrapped), anchor="ct")
            except Exception:
                pass

        p.end()
        return pm


    # ---------- interaction ----------
    def _zoom_at_anchor(self, factor: float):
        if getattr(self, "_qimg_src", None) is None and getattr(self, "_pm_src", None) is None:
            return

        old_scale = float(self.scale)
        new_scale = max(self._min_scale, min(old_scale * float(factor), self._max_scale))
        if abs(new_scale - old_scale) < 1e-8:
            return

        vp = self.scroll.viewport()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()

        try:
            anchor_vp = vp.mapFromGlobal(QCursor.pos())
        except Exception:
            anchor_vp = None

        if (anchor_vp is None) or (not vp.rect().contains(anchor_vp)):
            anchor_vp = QPoint(vp.width() // 2, vp.height() // 2)

        x_label_pre = hbar.value() + anchor_vp.x()
        y_label_pre = vbar.value() + anchor_vp.y()

        xi = x_label_pre / max(old_scale, 1e-12)
        yi = y_label_pre / max(old_scale, 1e-12)

        # Apply new scale
        self.scale = new_scale

        # FAST present (no rebuild)
        self._present_scaled(interactive=True)

        # Keep anchor stable
        x_label_post = xi * new_scale
        y_label_post = yi * new_scale

        new_h = int(round(x_label_post - anchor_vp.x()))
        new_v = int(round(y_label_post - anchor_vp.y()))

        new_h = max(hbar.minimum(), min(new_h, hbar.maximum()))
        new_v = max(vbar.minimum(), min(new_v, vbar.maximum()))

        hbar.setValue(new_h)
        vbar.setValue(new_v)

        # Defer one final smooth redraw (and WCS overlay) after the burst
        if self._smooth_zoom:
            self._request_zoom_redraw()


    def _request_zoom_redraw(self):
        if getattr(self, "_zoom_timer", None) is None:
            self._zoom_timer = QTimer(self)
            self._zoom_timer.setSingleShot(True)
            self._zoom_timer.timeout.connect(self._apply_zoom_redraw)

        # 60–120ms feels better than 16ms for “zoom burst collapse”
        # but keep your 16ms if you prefer.
        self._zoom_timer.start(90)


    def _apply_zoom_redraw(self):
        if not getattr(self, "_smooth_zoom", True):
            return
        if getattr(self, "_pm_src", None) is None:
            return
        self._present_scaled(interactive=False)



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



    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    def event(self, e):
        """Override event() to handle native macOS gestures (pinch zoom)."""
        # Handle native gestures (macOS trackpad pinch zoom)
        if e.type() == QEvent.Type.NativeGesture:
            gesture_type = e.gestureType()

            if gesture_type == Qt.NativeGestureType.BeginNativeGesture:
                # Start of pinch gesture - store initial scale
                self._gesture_zoom_start = self.scale
                e.accept()
                return True

            elif gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
                # Ongoing pinch zoom - value() is cumulative scale factor
                # Typical values: -0.5 to +0.5 for moderate pinches
                zoom_delta = e.value()

                # Convert delta to zoom factor
                # Use smaller multiplier for smoother feel (0.5x damping)
                factor = 1.0 + (zoom_delta * 0.5)

                # Apply incremental zoom
                self._zoom_at_anchor(factor)
                e.accept()
                return True

            elif gesture_type == Qt.NativeGestureType.EndNativeGesture:
                # End of pinch gesture - cleanup
                self._gesture_zoom_start = None
                e.accept()
                return True

        # Let parent handle all other events
        return super().event(e)



    def eventFilter(self, obj, ev):
        is_on_view = (obj is self.label) or (obj is self.scroll.viewport())

        # 0) PREVIEW-SELECT MODE: consume mouse events first so earlier branches don't steal them
        if self._preview_select_mode and is_on_view:
            vp = self.scroll.viewport()
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                vp_pos = obj.mapTo(vp, ev.pos())
                self._rubber_origin = vp_pos
                if self._rubber is None:
                    self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, vp)
                self._rubber.setGeometry(QRect(self._rubber_origin, QSize(1, 1)))
                self._rubber.show()
                ev.accept(); return True

            if ev.type() == QEvent.Type.MouseMove and self._rubber is not None and self._rubber_origin is not None:
                vp_pos = obj.mapTo(vp, ev.pos())
                rect = QRect(self._rubber_origin, vp_pos).normalized()
                self._rubber.setGeometry(rect)
                ev.accept(); return True

            if ev.type() == QEvent.Type.MouseButtonRelease and self._rubber is not None and self._rubber_origin is not None:
                vp_pos = obj.mapTo(vp, ev.pos())
                rect = QRect(self._rubber_origin, vp_pos).normalized()
                self._finish_preview_rect(rect)
                ev.accept(); return True
            # don’t swallow unrelated events

        # 1) Ctrl + wheel → zoom
        if ev.type() == QEvent.Type.Wheel:
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Try pixelDelta first (macOS trackpad gives smooth values)
                dy = ev.pixelDelta().y()

                if dy != 0:
                    # Smooth trackpad scrolling: use smaller base factor
                    # Scale proportionally to delta magnitude for natural feel
                    # Typical trackpad deltas are 1-10 pixels per event
                    abs_dy = abs(dy)
                    if abs_dy <= 3:
                        base_factor = 1.01  # Very gentle for tiny movements
                    elif abs_dy <= 10:
                        base_factor = 1.02  # Gentle for small movements
                    else:
                        base_factor = 1.03  # Moderate for larger gestures

                    factor = base_factor if dy > 0 else 1/base_factor
                else:
                    # Traditional mouse wheel: use angleDelta with moderate factor
                    dy = ev.angleDelta().y()
                    if dy == 0:
                        return True
                    # Use 1.15 for mouse wheel (gentler than original 1.25)
                    factor = 1.15 if dy > 0 else 1/1.15
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

        sw = self._mdi_subwindow()
        if sw is not None and obj is sw:
            et = ev.type()
            if et in (QEvent.Type.WindowStateChange, QEvent.Type.Show, QEvent.Type.Resize):
                QTimer.singleShot(0, self._update_inline_title_and_buttons)

        return super().eventFilter(obj, ev)

    def _viewport_pos_to_image_xy(self, vp_pos: QPoint) -> tuple[int, int] | None:
        """
        Convert a point in viewport coordinates to FULL image pixel coordinates.
        Returns None if the point is outside the displayed pixmap (in margins).
        """
        pm = self.label.pixmap()
        if pm is None:
            return None

        # Convert viewport point into label coordinates
        p_label = self.label.mapFrom(self.scroll.viewport(), vp_pos)

        # If label is larger than pixmap, pixmap may be centered inside label.
        pm_w, pm_h = pm.width(), pm.height()
        lbl_w, lbl_h = self.label.width(), self.label.height()

        off_x = max(0, (lbl_w - pm_w) // 2)
        off_y = max(0, (lbl_h - pm_h) // 2)

        px = p_label.x() - off_x
        py = p_label.y() - off_y

        # Outside the drawn pixmap area → clamp
        px = max(0, min(px, pm_w - 1))
        py = max(0, min(py, pm_h - 1))

        s = max(float(self.scale), 1e-12)

        # pixmap pixels -> image pixels (pm = image * scale)
        xi = int(round(px / s))
        yi = int(round(py / s))
        return xi, yi

    def _finish_preview_rect(self, vp_rect: QRect):
        if vp_rect.width() < 4 or vp_rect.height() < 4:
            self._cancel_rubber()
            return

        # Convert the two corners from viewport space to image space
        p0 = self._viewport_pos_to_image_xy(vp_rect.topLeft())
        p1 = self._viewport_pos_to_image_xy(vp_rect.bottomRight())

        if p0 is None or p1 is None:
            # User dragged into margins; you can either cancel or clamp.
            # Cancel is simplest:
            self._cancel_rubber()
            return

        x0, y0 = p0
        x1, y1 = p1

        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        if w < 1 or h < 1:
            self._cancel_rubber()
            return

        self._create_preview_from_roi((x, y, w, h))
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
        name = self.tr("Preview {0} ({1}×{2})").format(pid, w, h)

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
        self._update_replay_button()  
        mw = self._find_main_window()
        if mw is not None and getattr(mw, "_auto_fit_on_resize", False):
            try:
                mw._zoom_active_fit()
            except Exception:
                pass        

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
            self._pan_live = True   
            self._drag_start = e.pos()

            # NEW: emit once at drag start so linked views sync instantly
            self._emit_view_transform()
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
        wcs2 = self._get_celestial_wcs()
        if wcs2 is not None:
            try:
                ra_deg, dec_deg = map(float, wcs2.pixel_to_world_values(float(xi), float(yi)))

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



    # 1) helper to build ROI-adjusted WCS (keeps projection/rotation/CD/PC intact)
    def _wcs_for_roi(self, base_wcs, roi, arr_shape=None):
        # roi = (x, y, w, h) in FULL-image pixel coords
        import numpy as np
        if base_wcs is None or roi is None:
            return base_wcs
        x, y, w, h = map(int, roi)
        wnew = base_wcs.deepcopy()
        # shift reference pixel into the cropped frame
        wnew.wcs.crpix = wnew.wcs.crpix - np.array([float(x), float(y)], dtype=float)
        # tell astropy the new image size for grid/edge computations
        try:
            wnew.array_shape = (h, w)
            wnew.pixel_shape = (w, h)
        except Exception:
            pass
        # prefer 2-D celestial
        try:
            cel = getattr(wnew, "celestial", None)
            if cel is not None and getattr(cel, "naxis", 2) == 2:
                return cel
        except Exception:
            pass
        return wnew


    # 2) make _get_celestial_wcs ROI-aware
    def _get_celestial_wcs(self):
        """
        Return the *correct* celestial WCS for whatever the user is actually
        seeing in this view.

        - On the Full tab: just use the document's WCS / header.
        - On a Preview tab: prefer the ROI backing doc's WCS from DocManager.
          If that's not available, synthesize a cropped header from the base
          header + preview ROI via _compute_cropped_wcs().
        """
        doc = getattr(self, "document", None)
        if doc is None:
            return None

        # -----------------------------
        # FULL IMAGE (no preview active)
        # -----------------------------
        if not self.has_active_preview():
            meta = getattr(doc, "metadata", {}) or {}
            w = meta.get("wcs")
            if isinstance(w, _AstroWCS):
                try:
                    wc = getattr(w, "celestial", None)
                    return wc if (wc is not None and getattr(wc, "naxis", 2) == 2) else w
                except Exception:
                    return w

            hdr = (
                meta.get("original_header")
                or meta.get("fits_header")
                or meta.get("header")
            )
            if hdr is None:
                return None

            w = build_celestial_wcs(hdr)
            if w is not None:
                meta["wcs"] = w
            return w

        # -----------------------------
        # PREVIEW TAB (ROI view)
        # -----------------------------
        roi = self.current_preview_roi()
        if roi is None:
            return None

        # Base document is the full image doc; backing_doc may be the ROI doc
        base_doc = getattr(self, "base_document", None) or doc
        base_meta = getattr(base_doc, "metadata", {}) or {}

        dm = getattr(self, "_docman", None)
        backing_doc = None
        if dm is not None:
            try:
                backing_doc = dm.get_document_for_view(self)
            except Exception:
                backing_doc = None

        # 1) If DocManager has a separate ROI doc for this view, use ITS WCS
        if backing_doc is not None and backing_doc is not base_doc:
            bmeta = getattr(backing_doc, "metadata", {}) or {}
            w = bmeta.get("wcs")
            if isinstance(w, _AstroWCS):
                try:
                    wc = getattr(w, "celestial", None)
                    return wc if (wc is not None and getattr(wc, "naxis", 2) == 2) else w
                except Exception:
                    return w

            hdr = (
                bmeta.get("original_header")
                or bmeta.get("fits_header")
                or bmeta.get("header")
            )
            if hdr is not None:
                w = build_celestial_wcs(hdr)
                if w is not None:
                    bmeta["wcs"] = w
                    return w

        # 2) Fallback: synthesize cropped WCS from base header + ROI
        hdr_full = (
            base_meta.get("original_header")
            or base_meta.get("fits_header")
            or base_meta.get("header")
        )
        if hdr_full is None:
            return None

        cache_key = f"_preview_wcs_{self._active_preview_id}"
        cached = base_meta.get(cache_key)
        if isinstance(cached, _AstroWCS):
            try:
                wc = getattr(cached, "celestial", None)
                return wc if (wc is not None and getattr(wc, "naxis", 2) == 2) else cached
            except Exception:
                pass

        try:
            x, y, w, h = map(int, roi)
            cropped_hdr = _compute_cropped_wcs(hdr_full, x, y, w, h)
            wcs = build_celestial_wcs(cropped_hdr)
        except Exception:
            wcs = None

        if wcs is not None:
            base_meta[cache_key] = wcs
        return wcs


    def _extract_wcs_from_doc(self):
        """
        Try to get an astropy WCS from the current document or a sensible parent.
        Caches the resolved WCS on whichever doc we pulled it from.
        """
        doc = getattr(self, "document", None)
        if doc is None:
            return None

        def _try_on_meta(meta: dict):
            # (1) literal WCS object stored?
            w = meta.get("wcs")
            if isinstance(w, _AstroWCS):
                return w
            # (2) any header-like thing present?
            hdr = meta.get("original_header") or meta.get("fits_header") or meta.get("header")
            return build_celestial_wcs(hdr)

        # 1) current doc (+ cache)
        meta = getattr(doc, "metadata", {}) or {}
        if "_astropy_wcs" in meta:
            return meta["_astropy_wcs"]
        w = _try_on_meta(meta)
        if w is not None:
            meta["_astropy_wcs"] = w
            return w

        # 2) likely parents/sources
        candidates = []

        base = getattr(self, "base_document", None)
        if base is not None and base is not doc:
            candidates.append(base)

        dm = getattr(self, "_docman", None)
        if dm is not None and hasattr(dm, "get_document_for_view"):
            try:
                src = dm.get_document_for_view(self)
                if src is not None and src is not doc and src is not base:
                    candidates.append(src)
            except Exception:
                pass

        src_uid = meta.get("wcs_source_doc_uid") or meta.get("base_doc_uid")
        if src_uid is not None:
            try:
                from setiastro.saspro.doc_manager import DocManager
                reg = getattr(DocManager, "_global_registry", {})
                by_uid = reg.get(src_uid)
                if by_uid and by_uid not in candidates and by_uid is not doc and by_uid is not base:
                    candidates.append(by_uid)
            except Exception:
                pass

        for cand in candidates:
            m = getattr(cand, "metadata", {}) or {}
            if "_astropy_wcs" in m:
                meta["_astropy_wcs"] = m["_astropy_wcs"]
                return m["_astropy_wcs"]
            w = _try_on_meta(m)
            if w is not None:
                m["_astropy_wcs"] = w
                meta["_astropy_wcs"] = w
                return w

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
            self.scroll.horizontalScrollBar().setValue(self.scroll.horizontalScrollBar().value() - delta.x())
            self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().value() - delta.y())
            self._drag_start = e.pos()
            # live emit happens via _on_scroll_changed(), but this is a nice extra nudge:
            self._emit_view_transform_now()
            return

        super().mouseMoveEvent(e)



    def mouseReleaseEvent(self, e):
        if self._preview_select_mode:
            e.ignore(); return
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._pan_live = False        # ← back to debounced mode
            self._readout_dragging = False
            self._emit_view_transform()
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
                    self, self.tr("Close Image?"),
                    self.tr("This image has edits that aren’t applied/saved.\nClose anyway?"),
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
        try:
            self.unlink_all()
        except Exception:
            pass
        try:
            if id(self) in ImageSubWindow._registry:
                ImageSubWindow._registry.pop(id(self), None)
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

# --- NEW: Enhanced TableSubWindow + Plot/Stats --------------------------------
# ----------------------------- helpers ---------------------------------
def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s


def _find_column_candidates(model: TypedTableModel) -> dict[str, list[int]]:
    """
    Build a semantic map of likely science columns from FITS/MAST-style names.
    """
    out = {
        "time": [],
        "flux": [],
        "flux_err": [],
        "wavelength": [],
        "frequency": [],
        "intensity": [],
        "quality": [],
        "cadence": [],
        "background": [],
        "magnitude": [],
    }

    for i, ci in enumerate(model.column_infos()):
        name = _norm_name(ci.name)

        # time-like
        if name in ("time", "bjd", "mjd", "jd", "hjd") or "time" in name:
            out["time"].append(i)

        # flux-like
        if any(k in name for k in (
            "pdcsap_flux", "sap_flux", "det_flux", "sys_rm_flux", "flux", "counts", "rate"
        )):
            out["flux"].append(i)

        # error-like
        if any(k in name for k in (
            "flux_err", "error", "err", "unc", "uncert"
        )):
            out["flux_err"].append(i)

        # wavelength / spectrum axis
        if any(k in name for k in (
            "wavelength", "lambda", "wave", "lam", "angstrom", "nm", "micron", "um"
        )):
            out["wavelength"].append(i)

        # frequency axis
        if any(k in name for k in (
            "frequency", "freq", "hz", "ghz", "mhz"
        )):
            out["frequency"].append(i)

        # intensity / spectral ordinate
        if any(k in name for k in (
            "intensity", "signal", "spectrum", "spec_flux", "flam", "fnu", "net", "value"
        )):
            out["intensity"].append(i)

        if "quality" in name or "qual" in name:
            out["quality"].append(i)

        if "cadence" in name:
            out["cadence"].append(i)

        if "bkg" in name or "background" in name:
            out["background"].append(i)

        if "mag" in name:
            out["magnitude"].append(i)

    return out


def _pick_first_existing(candidates: list[int], valid_numeric: set[int]) -> Optional[int]:
    for c in candidates:
        if c in valid_numeric:
            return c
    return None


def _guess_plot_columns(model: TypedTableModel) -> dict[str, Optional[int]]:
    """
    Best-effort semantic guessing for common astronomy table patterns.
    """
    numeric_cols = {i for i, ci in enumerate(model.column_infos()) if ci.is_numeric}
    sem = _find_column_candidates(model)

    time_col = _pick_first_existing(sem["time"], numeric_cols)

    # Prefer PDCSAP/SAP/DET style columns before generic flux
    flux_ranked = []
    for i in sem["flux"]:
        nm = _norm_name(model.column_infos()[i].name)
        rank = 100
        if "pdcsap_flux" in nm:
            rank = 0
        elif "sap_flux" in nm:
            rank = 1
        elif "det_flux" in nm:
            rank = 2
        elif "sys_rm_flux" in nm:
            rank = 3
        elif "flux" in nm:
            rank = 10
        flux_ranked.append((rank, i))
    flux_ranked.sort()
    flux_col = flux_ranked[0][1] if flux_ranked else None

    flux_err_col = _pick_first_existing(sem["flux_err"], numeric_cols)
    wave_col = _pick_first_existing(sem["wavelength"], numeric_cols)
    freq_col = _pick_first_existing(sem["frequency"], numeric_cols)
    inten_col = _pick_first_existing(sem["intensity"], numeric_cols)
    mag_col = _pick_first_existing(sem["magnitude"], numeric_cols)

    # fallback generic numeric columns
    numeric_list = sorted(numeric_cols)
    if time_col is None and numeric_list:
        time_col = numeric_list[0]
    if flux_col is None and len(numeric_list) >= 2:
        flux_col = numeric_list[1]
    elif flux_col is None and numeric_list:
        flux_col = numeric_list[0]

    if inten_col is None:
        # for spectra, if no explicit intensity, use best flux-like column
        inten_col = flux_col

    return {
        "time": time_col,
        "flux": flux_col,
        "flux_err": flux_err_col,
        "wavelength": wave_col,
        "frequency": freq_col,
        "intensity": inten_col,
        "magnitude": mag_col,
    }


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=bool)
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def _sigma_clip_mask(y: np.ndarray, sigma: float) -> np.ndarray:
    if y.size == 0 or sigma <= 0:
        return np.ones_like(y, dtype=bool)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad <= 0:
        std = np.nanstd(y)
        if not np.isfinite(std) or std <= 0:
            return np.ones_like(y, dtype=bool)
        return np.abs(y - med) <= sigma * std
    robust_sigma = 1.4826 * mad
    return np.abs(y - med) <= sigma * robust_sigma


def _phase_fold(x: np.ndarray, period: float, epoch: float = 0.0) -> np.ndarray:
    if period <= 0:
        return x.copy()
    return ((x - epoch) / period) % 1.0


def _bin_xy(x: np.ndarray, y: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0 or nbins <= 1:
        return x, y
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    which = np.digitize(x, edges) - 1
    xb = []
    yb = []
    for b in range(nbins):
        m = which == b
        if not np.any(m):
            continue
        xb.append(float(np.nanmedian(x[m])))
        yb.append(float(np.nanmedian(y[m])))
    if not xb:
        return x, y
    return np.asarray(xb, dtype=np.float64), np.asarray(yb, dtype=np.float64)

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    # FITS tables often have bytes
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            try:
                return x.decode("latin1", errors="replace")
            except Exception:
                return repr(x)
    # numpy scalars
    try:
        if isinstance(x, (np.generic,)):
            x = x.item()
    except Exception:
        pass
    return str(x)

def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return True
    except Exception:
        pass
    try:
        if isinstance(x, (np.floating, np.integer)):
            xv = x.item()
            if isinstance(xv, float) and (math.isnan(xv) or math.isinf(xv)):
                return True
    except Exception:
        pass
    return False

def _try_float(x: Any) -> Optional[float]:
    if _is_missing(x):
        return None
    # already numeric
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except Exception:
            return None

    s = _safe_str(x).strip()
    if not s:
        return None
    # handle FITS bytes shown like b'...'
    # also allow scientific notation
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def _numeric_fraction(values: List[Any]) -> float:
    if not values:
        return 0.0
    ok = 0
    tot = 0
    for v in values:
        tot += 1
        if _try_float(v) is not None:
            ok += 1
    return ok / max(1, tot)

def _calc_stats_numeric(vals: np.ndarray) -> dict:
    # vals is 1D float, may contain NaNs already removed
    if vals.size == 0:
        return dict(count=0)
    return dict(
        count=int(vals.size),
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        mean=float(np.mean(vals)),
        median=float(np.median(vals)),
        std=float(np.std(vals)),
    )


# ----------------------------- models ----------------------------------

@dataclass
class ColumnInfo:
    name: str
    is_numeric: bool
    numeric_values: Optional[np.ndarray] = None  # aligned by row index with NaNs for missing


class TypedTableModel(QAbstractTableModel):
    """
    Table model with lightweight type inference per column.
    Keeps raw values for display, plus numeric vectors (NaN for missing) for plotting/stats.
    """
    def __init__(self, rows: list, headers: list, parent=None):
        super().__init__(parent)
        self._headers = [str(h) for h in (headers or [])]
        self._rows = list(rows or [])
        self._nrows = len(self._rows)
        self._ncols = len(self._headers) if self._headers else (len(self._rows[0]) if self._rows else 0)

        # normalize headers if missing
        if not self._headers:
            self._headers = [f"COL_{i}" for i in range(self._ncols)]

        # column cache
        self._colinfo: list[ColumnInfo] = []
        self._build_column_info()

    def _build_column_info(self):
        self._colinfo.clear()
        # gather values by col
        for c in range(self._ncols):
            col_vals = []
            for r in range(self._nrows):
                try:
                    col_vals.append(self._rows[r][c])
                except Exception:
                    col_vals.append(None)

            frac = _numeric_fraction(col_vals)
            is_num = (frac >= 0.80) and (self._nrows >= 1)  # pretty permissive
            if is_num:
                nv = np.full((self._nrows,), np.nan, dtype=np.float64)
                for i, v in enumerate(col_vals):
                    f = _try_float(v)
                    if f is not None:
                        nv[i] = f
                self._colinfo.append(ColumnInfo(name=self._headers[c], is_numeric=True, numeric_values=nv))
            else:
                self._colinfo.append(ColumnInfo(name=self._headers[c], is_numeric=False, numeric_values=None))

    # Qt model API
    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._nrows

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._ncols

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._headers):
                return self._headers[section]
        else:
            return str(section + 1)
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r = index.row()
        c = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            try:
                v = self._rows[r][c]
            except Exception:
                v = None
            # show nicer for floats
            f = _try_float(v)
            if f is not None and self._colinfo[c].is_numeric:
                # keep it readable; avoids "1.0000000002" spam
                return f"{f:.10g}"
            return _safe_str(v)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if self._colinfo[c].is_numeric:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        return None

    # convenience
    def column_infos(self) -> list[ColumnInfo]:
        return self._colinfo

    def raw_value(self, row: int, col: int) -> Any:
        try:
            return self._rows[row][col]
        except Exception:
            return None

    def numeric_column(self, col: int) -> Optional[np.ndarray]:
        if 0 <= col < len(self._colinfo) and self._colinfo[col].is_numeric:
            return self._colinfo[col].numeric_values
        return None


class MultiColumnContainsFilterProxy(QSortFilterProxyModel):
    """
    Simple row filter: keep rows where ANY column contains the substring (case-insensitive).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._needle = ""

        # sorting is enabled by view; keep stable sort
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_needle(self, text: str):
        self._needle = (text or "").strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        n = self._needle
        if not n:
            return True

        m = self.sourceModel()
        if m is None:
            return True

        cols = m.columnCount()
        for c in range(cols):
            idx = m.index(source_row, c, source_parent)
            s = m.data(idx, Qt.ItemDataRole.DisplayRole)
            if s is None:
                continue
            if n in str(s).lower():
                return True
        return False


# ----------------------------- stats panel ----------------------------------

class TableStatsPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Stats", parent)
        self.setMinimumWidth(260)
        lay = QFormLayout(self)

        self.lbl_rows = QLabel("—")
        self.lbl_cols = QLabel("—")
        self.lbl_sel_col = QLabel("—")
        self.lbl_type = QLabel("—")
        self.lbl_count = QLabel("—")
        self.lbl_min = QLabel("—")
        self.lbl_max = QLabel("—")
        self.lbl_mean = QLabel("—")
        self.lbl_median = QLabel("—")
        self.lbl_std = QLabel("—")
        self.lbl_unique = QLabel("—")

        lay.addRow("Rows (visible):", self.lbl_rows)
        lay.addRow("Columns:", self.lbl_cols)
        lay.addRow("Selected col:", self.lbl_sel_col)
        lay.addRow("Type:", self.lbl_type)
        lay.addRow("Count:", self.lbl_count)
        lay.addRow("Min:", self.lbl_min)
        lay.addRow("Max:", self.lbl_max)
        lay.addRow("Mean:", self.lbl_mean)
        lay.addRow("Median:", self.lbl_median)
        lay.addRow("Std:", self.lbl_std)
        lay.addRow("Unique (sample):", self.lbl_unique)

    def set_table_shape(self, nrows_visible: int, ncols: int):
        self.lbl_rows.setText(str(nrows_visible))
        self.lbl_cols.setText(str(ncols))

    def set_selection_info(
        self,
        col_name: str,
        is_numeric: bool,
        stats: Optional[dict] = None,
        unique_sample: Optional[int] = None,
    ):
        self.lbl_sel_col.setText(col_name or "—")
        self.lbl_type.setText("numeric" if is_numeric else "text/mixed")

        if is_numeric and stats and stats.get("count", 0) > 0:
            self.lbl_count.setText(str(stats.get("count", "—")))
            self.lbl_min.setText(f"{stats.get('min'):.10g}")
            self.lbl_max.setText(f"{stats.get('max'):.10g}")
            self.lbl_mean.setText(f"{stats.get('mean'):.10g}")
            self.lbl_median.setText(f"{stats.get('median'):.10g}")
            self.lbl_std.setText(f"{stats.get('std'):.10g}")
        else:
            self.lbl_count.setText("—")
            self.lbl_min.setText("—")
            self.lbl_max.setText("—")
            self.lbl_mean.setText("—")
            self.lbl_median.setText("—")
            self.lbl_std.setText("—")

        self.lbl_unique.setText("—" if unique_sample is None else str(unique_sample))


# ----------------------------- plot dialog ----------------------------------
class PlotTableDialog(QDialog):
    def __init__(self, parent, source_model: TypedTableModel, proxy_model: MultiColumnContainsFilterProxy):
        super().__init__(parent)
        self.setWindowTitle("Plot Table Columns")
        self._src = source_model
        self._proxy = proxy_model
        self._semantic = _guess_plot_columns(self._src)

        root = QVBoxLayout(self)

        # ---------------- controls ----------------
        ctrl = QGroupBox("Plot Settings", self)
        form = QFormLayout(ctrl)

        self.cmb_preset = QComboBox(self)
        self.cmb_preset.addItem("Generic X-Y", userData="generic")
        self.cmb_preset.addItem("TESS Light Curve", userData="tess_lc")
        self.cmb_preset.addItem("Phase Folded Light Curve", userData="phase")
        self.cmb_preset.addItem("Spectrum", userData="spectrum")
        self.cmb_preset.addItem("Histogram", userData="hist")
        form.addRow("Preset:", self.cmb_preset)

        self.cmb_x = QComboBox(self)
        self.cmb_y = QComboBox(self)
        self.cmb_yerr = QComboBox(self)
        self.cmb_yerr.addItem("(none)", userData=-1)

        self._num_cols: list[int] = []
        for i, ci in enumerate(self._src.column_infos()):
            if ci.is_numeric:
                self._num_cols.append(i)
                label = ci.name
                self.cmb_x.addItem(label, userData=i)
                self.cmb_y.addItem(label, userData=i)
                self.cmb_yerr.addItem(label, userData=i)

        form.addRow("X:", self.cmb_x)
        form.addRow("Y:", self.cmb_y)
        form.addRow("Y error:", self.cmb_yerr)

        self.chk_sort_x = QCheckBox("Sort by X", self)
        self.chk_sort_x.setChecked(True)
        form.addRow("", self.chk_sort_x)

        self.chk_remove_nan = QCheckBox("Remove NaNs / missing", self)
        self.chk_remove_nan.setChecked(True)
        form.addRow("", self.chk_remove_nan)

        self.chk_line = QCheckBox("Connect points (line)", self)
        self.chk_line.setChecked(False)
        form.addRow("", self.chk_line)

        self.chk_normalize_y = QCheckBox("Normalize Y by median", self)
        self.chk_normalize_y.setChecked(False)
        form.addRow("", self.chk_normalize_y)

        self.chk_invert_y = QCheckBox("Invert Y axis (magnitudes)", self)
        self.chk_invert_y.setChecked(False)
        form.addRow("", self.chk_invert_y)

        self.chk_log_x = QCheckBox("Log X axis", self)
        self.chk_log_y = QCheckBox("Log Y axis", self)
        log_row = QWidget(self)
        log_lay = QHBoxLayout(log_row)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.addWidget(self.chk_log_x)
        log_lay.addWidget(self.chk_log_y)
        log_lay.addStretch(1)
        form.addRow("", log_row)

        self.chk_sigma_clip = QCheckBox("Sigma clip Y", self)
        self.chk_sigma_clip.setChecked(False)
        self.sp_sigma = QDoubleSpinBox(self)
        self.sp_sigma.setRange(0.5, 20.0)
        self.sp_sigma.setSingleStep(0.5)
        self.sp_sigma.setValue(4.0)
        sig_row = QWidget(self)
        sig_lay = QHBoxLayout(sig_row)
        sig_lay.setContentsMargins(0, 0, 0, 0)
        sig_lay.addWidget(self.chk_sigma_clip)
        sig_lay.addWidget(self.sp_sigma)
        sig_lay.addStretch(1)
        form.addRow("", sig_row)

        self.sp_ms = QDoubleSpinBox(self)
        self.sp_ms.setRange(0.5, 20.0)
        self.sp_ms.setSingleStep(0.5)
        self.sp_ms.setValue(3.0)
        form.addRow("Marker size:", self.sp_ms)

        # phase-fold controls
        self.sp_period = QDoubleSpinBox(self)
        self.sp_period.setRange(1e-9, 1e9)
        self.sp_period.setDecimals(8)
        self.sp_period.setValue(1.0)

        self.sp_epoch = QDoubleSpinBox(self)
        self.sp_epoch.setRange(-1e9, 1e9)
        self.sp_epoch.setDecimals(8)
        self.sp_epoch.setValue(0.0)

        self.sp_phase_bins = QSpinBox(self)
        self.sp_phase_bins.setRange(2, 1000)
        self.sp_phase_bins.setValue(100)

        form.addRow("Period:", self.sp_period)
        form.addRow("Epoch:", self.sp_epoch)
        form.addRow("Phase bins:", self.sp_phase_bins)

        # histogram controls
        self.sp_hist_bins = QSpinBox(self)
        self.sp_hist_bins.setRange(5, 5000)
        self.sp_hist_bins.setValue(50)
        form.addRow("Histogram bins:", self.sp_hist_bins)

        root.addWidget(ctrl)

        # ---------------- plot area ----------------
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavToolbar(self.canvas, self)
        root.addWidget(self.toolbar)
        root.addWidget(self.canvas, 1)

        # ---------------- buttons ----------------
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        self.btn_plot = QPushButton("Plot", self)
        self.btn_plot.clicked.connect(self._do_plot)
        btns.addButton(self.btn_plot, QDialogButtonBox.ButtonRole.ActionRole)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.cmb_preset.currentIndexChanged.connect(self._preset_changed)

        self.resize(960, 700)

        self._preset_changed()
        if self.cmb_x.count() > 0 and self.cmb_y.count() > 0:
            self._do_plot()

    def _gather_visible_numeric(self, col: int) -> np.ndarray:
        nv = self._src.numeric_column(col)
        if nv is None:
            return np.array([], dtype=np.float64)

        out = np.empty((self._proxy.rowCount(),), dtype=np.float64)
        out[:] = np.nan

        for pr in range(self._proxy.rowCount()):
            src_idx = self._proxy.mapToSource(self._proxy.index(pr, col))
            sr = src_idx.row()
            if 0 <= sr < nv.shape[0]:
                out[pr] = nv[sr]
        return out

    def _set_combo_to_col(self, combo: QComboBox, col: Optional[int]):
        if col is None:
            return
        idx = combo.findData(col)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _preset_changed(self):
        preset = self.cmb_preset.currentData()

        # defaults
        self.chk_line.setChecked(False)
        self.chk_sort_x.setChecked(True)
        self.chk_normalize_y.setChecked(False)
        self.chk_invert_y.setChecked(False)
        self.chk_log_x.setChecked(False)
        self.chk_log_y.setChecked(False)

        # enable/disable phase/hist controls
        is_phase = (preset == "phase")
        is_hist = (preset == "hist")
        self.sp_period.setEnabled(is_phase)
        self.sp_epoch.setEnabled(is_phase)
        self.sp_phase_bins.setEnabled(is_phase)
        self.sp_hist_bins.setEnabled(is_hist)

        sem = self._semantic

        if preset == "tess_lc":
            self._set_combo_to_col(self.cmb_x, sem["time"])
            self._set_combo_to_col(self.cmb_y, sem["flux"])
            self._set_combo_to_col(self.cmb_yerr, sem["flux_err"])
            self.chk_line.setChecked(False)
            self.chk_normalize_y.setChecked(True)

        elif preset == "phase":
            self._set_combo_to_col(self.cmb_x, sem["time"])
            self._set_combo_to_col(self.cmb_y, sem["flux"])
            self._set_combo_to_col(self.cmb_yerr, sem["flux_err"])
            self.chk_line.setChecked(False)
            self.chk_normalize_y.setChecked(True)

        elif preset == "spectrum":
            xcol = sem["wavelength"] if sem["wavelength"] is not None else sem["frequency"]
            ycol = sem["intensity"] if sem["intensity"] is not None else sem["flux"]
            self._set_combo_to_col(self.cmb_x, xcol)
            self._set_combo_to_col(self.cmb_y, ycol)
            self.chk_line.setChecked(True)

        elif preset == "hist":
            # histogram uses Y column only
            if sem["flux"] is not None:
                self._set_combo_to_col(self.cmb_y, sem["flux"])

        else:  # generic
            if sem["time"] is not None:
                self._set_combo_to_col(self.cmb_x, sem["time"])
            if sem["flux"] is not None:
                self._set_combo_to_col(self.cmb_y, sem["flux"])

    def _apply_common_cleaning(self, x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray]):
        mask = np.ones_like(y, dtype=bool)

        if self.chk_remove_nan.isChecked():
            mask &= _finite_mask(x, y)
            if yerr is not None:
                mask &= np.isfinite(yerr)

        x2 = x[mask]
        y2 = y[mask]
        yerr2 = yerr[mask] if yerr is not None else None

        if self.chk_sigma_clip.isChecked() and x2.size > 0 and y2.size > 0:
            cm = _sigma_clip_mask(y2, float(self.sp_sigma.value()))
            x2 = x2[cm]
            y2 = y2[cm]
            if yerr2 is not None:
                yerr2 = yerr2[cm]

        if self.chk_normalize_y.isChecked() and y2.size > 0:
            med = np.nanmedian(y2)
            if np.isfinite(med) and med != 0:
                y2 = y2 / med
                if yerr2 is not None:
                    yerr2 = yerr2 / abs(med)

        return x2, y2, yerr2

    def _do_plot(self):
        if self.cmb_y.count() == 0:
            QMessageBox.information(self, "Plot", "No numeric columns available to plot.")
            return

        preset = self.cmb_preset.currentData()
        ycol = int(self.cmb_y.currentData())
        xcol = int(self.cmb_x.currentData()) if self.cmb_x.count() > 0 else -1
        yerr_col = int(self.cmb_yerr.currentData())

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ms = float(self.sp_ms.value())

        if preset == "hist":
            y = self._gather_visible_numeric(ycol)
            y = y[np.isfinite(y)]
            if self.chk_sigma_clip.isChecked():
                y = y[_sigma_clip_mask(y, float(self.sp_sigma.value()))]
            if self.chk_normalize_y.isChecked() and y.size > 0:
                med = np.nanmedian(y)
                if np.isfinite(med) and med != 0:
                    y = y / med

            if y.size == 0:
                QMessageBox.warning(self, "Plot", "No valid numeric data for histogram.")
                return

            ax.hist(y, bins=int(self.sp_hist_bins.value()))
            ax.set_xlabel(self.cmb_y.currentText())
            ax.set_ylabel("Count")
            ax.set_title(f"Histogram: {self.cmb_y.currentText()}")

        else:
            if self.cmb_x.count() == 0:
                QMessageBox.warning(self, "Plot", "No numeric X column available.")
                return

            x = self._gather_visible_numeric(xcol)
            y = self._gather_visible_numeric(ycol)
            yerr = self._gather_visible_numeric(yerr_col) if yerr_col >= 0 else None

            if x.size == 0 or y.size == 0:
                QMessageBox.warning(self, "Plot", "No data available.")
                return

            x2, y2, yerr2 = self._apply_common_cleaning(x, y, yerr)

            if x2.size == 0:
                QMessageBox.warning(self, "Plot", "All rows were filtered out.")
                return

            if preset == "phase":
                period = float(self.sp_period.value())
                epoch = float(self.sp_epoch.value())
                x2 = _phase_fold(x2, period, epoch)

                if self.chk_sort_x.isChecked():
                    order = np.argsort(x2)
                    x2 = x2[order]
                    y2 = y2[order]
                    if yerr2 is not None:
                        yerr2 = yerr2[order]

                xb, yb = _bin_xy(x2, y2, int(self.sp_phase_bins.value()))
                ax.plot(x2, y2, "o", alpha=0.35, markersize=max(1.0, ms - 1.0))
                if xb.size > 0:
                    ax.plot(xb, yb, "-", linewidth=1.5)
                ax.set_xlabel("Phase")
                ax.set_ylabel(self.cmb_y.currentText())
                ax.set_title("Phase-Folded Light Curve")

            else:
                if self.chk_sort_x.isChecked():
                    order = np.argsort(x2)
                    x2 = x2[order]
                    y2 = y2[order]
                    if yerr2 is not None:
                        yerr2 = yerr2[order]

                if preset == "spectrum":
                    if yerr2 is not None:
                        ax.errorbar(
                            x2, y2, yerr=yerr2,
                            fmt="-",
                            linewidth=1.0,
                            ecolor="orange"
                        )
                    else:
                        ax.plot(x2, y2, "-", linewidth=1.0)
                    ax.set_title("Spectrum")
                else:
                    if yerr2 is not None:
                        ax.errorbar(
                            x2, y2, yerr=yerr2,
                            fmt="o" if not self.chk_line.isChecked() else "-o",
                            markersize=ms,
                            ecolor="orange"
                        )
                    else:
                        if self.chk_line.isChecked():
                            ax.plot(x2, y2, "-o", markersize=ms)
                        else:
                            ax.plot(x2, y2, "o", markersize=ms)

                ax.set_xlabel(self.cmb_x.currentText())
                ax.set_ylabel(self.cmb_y.currentText())

        if self.chk_log_x.isChecked():
            try:
                ax.set_xscale("log")
            except Exception:
                pass
        if self.chk_log_y.isChecked():
            try:
                ax.set_yscale("log")
            except Exception:
                pass
        if self.chk_invert_y.isChecked():
            ax.invert_yaxis()

        ax.grid(True, which="both", alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

# ----------------------------- TableSubWindow --------------------------------

class TableSubWindow(QWidget):
    """
    Enhanced table view for FITS/astropy tables:
      - Export CSV
      - Copy selected rows as CSV
      - Search/filter rows
      - Stats panel (selected column + visible rows)
      - Plot dialog (numeric columns; optional error bars)
    """
    viewTitleChanged = pyqtSignal(object, str)

    def __init__(self, table_document, parent=None):
        super().__init__(parent)
        self.document = table_document
        self._last_title_for_emit = None

        root = QVBoxLayout(self)

        # --- header row
        header_row = QHBoxLayout()
        self.title_lbl = QLabel(self.document.display_name())
        header_row.addWidget(self.title_lbl)
        header_row.addStretch(1)

        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search / filter rows…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setMaximumWidth(360)
        header_row.addWidget(self.search_edit)

        self.btn_plot = QPushButton(self.tr("Plot…"), self)
        self.btn_copy = QPushButton(self.tr("Copy Rows"), self)
        self.export_btn = QPushButton(self.tr("Export CSV…"), self)

        header_row.addWidget(self.btn_plot)
        header_row.addWidget(self.btn_copy)
        header_row.addWidget(self.export_btn)

        root.addLayout(header_row)

        # --- main splitter: table + stats
        split = QSplitter(self)
        split.setOrientation(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        # table
        table_wrap = QWidget(self)
        table_lay = QVBoxLayout(table_wrap)
        table_lay.setContentsMargins(0, 0, 0, 0)

        self.table = QTableView(self)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        table_lay.addWidget(self.table, 1)

        split.addWidget(table_wrap)

        # stats
        self.stats = TableStatsPanel(self)
        split.addWidget(self.stats)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)

        # data/model
        rows = getattr(self.document, "rows", [])
        headers = getattr(self.document, "headers", [])

        self._src_model = TypedTableModel(rows, headers, self)
        self._proxy = MultiColumnContainsFilterProxy(self)
        self._proxy.setSourceModel(self._src_model)

        self.table.setModel(self._proxy)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.resizeColumnsToContents()

        # connect signals
        self.export_btn.clicked.connect(self._export_csv)
        self.btn_copy.clicked.connect(self._copy_selected_rows_csv)
        self.btn_plot.clicked.connect(self._open_plot_dialog)
        self.search_edit.textChanged.connect(self._on_search_changed)

        # selection -> stats
        sel = self.table.selectionModel()
        if sel is not None:
            sel.selectionChanged.connect(self._update_stats_from_selection)

        self._sync_host_title()

        try:
            self.document.changed.connect(self._on_doc_changed)
        except Exception:
            pass

        # initial stats
        self._update_stats_from_selection()

    # ---- doc/title
    def _on_doc_changed(self):
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

    # ---- search/filter
    def _on_search_changed(self, text: str):
        self._proxy.set_needle(text)
        self._update_stats_from_selection()

    # ---- stats
    def _update_stats_from_selection(self, *_):
        nrows_vis = self._proxy.rowCount()
        ncols = self._src_model.columnCount()
        self.stats.set_table_shape(nrows_vis, ncols)

        # selected column (prefer currentIndex)
        idx = self.table.currentIndex()
        if not idx.isValid():
            # no selection
            self.stats.set_selection_info("—", False, None, None)
            return

        # currentIndex is in proxy coords
        col = idx.column()
        # col name/type from source model (same column index)
        infos = self._src_model.column_infos()
        if not (0 <= col < len(infos)):
            self.stats.set_selection_info("—", False, None, None)
            return

        ci = infos[col]
        if ci.is_numeric and ci.numeric_values is not None:
            # stats over visible rows only
            v = []
            for pr in range(self._proxy.rowCount()):
                sidx = self._proxy.mapToSource(self._proxy.index(pr, col))
                sr = sidx.row()
                if 0 <= sr < ci.numeric_values.shape[0]:
                    v.append(ci.numeric_values[sr])
            arr = np.asarray(v, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            st = _calc_stats_numeric(arr)
            self.stats.set_selection_info(ci.name, True, st, None)
        else:
            # unique sample over visible rows only (capped)
            uniq = set()
            cap = 5000
            for pr in range(min(self._proxy.rowCount(), cap)):
                sidx = self._proxy.mapToSource(self._proxy.index(pr, col))
                sr = sidx.row()
                uniq.add(_safe_str(self._src_model.raw_value(sr, col)))
            self.stats.set_selection_info(ci.name, False, None, len(uniq))

    # ---- copy selected rows
    def _copy_selected_rows_csv(self):
        sel = self.table.selectionModel()
        if sel is None:
            return
        rows = sel.selectedRows()
        if not rows:
            QMessageBox.information(self, "Copy", "No rows selected.")
            return

        # stable order
        proxy_rows = sorted({r.row() for r in rows})
        cols = self._src_model.columnCount()
        headers = [self._src_model.headerData(c, Qt.Orientation.Horizontal) for c in range(cols)]
        headers = [str(h) for h in headers]

        # build CSV text
        out_lines = []
        out_lines.append(",".join([self._csv_escape(h) for h in headers]))

        for pr in proxy_rows:
            # map proxy row -> source row
            sidx0 = self._proxy.mapToSource(self._proxy.index(pr, 0))
            sr = sidx0.row()
            row_vals = []
            for c in range(cols):
                row_vals.append(_safe_str(self._src_model.raw_value(sr, c)))
            out_lines.append(",".join([self._csv_escape(v) for v in row_vals]))

        txt = "\n".join(out_lines)
        QGuiApplication.clipboard().setText(txt)
        QMessageBox.information(self, "Copy", f"Copied {len(proxy_rows)} rows to clipboard as CSV.")

    @staticmethod
    def _csv_escape(s: str) -> str:
        # minimal CSV quoting
        if s is None:
            return ""
        s = str(s)
        if any(ch in s for ch in [",", '"', "\n", "\r"]):
            s = s.replace('"', '""')
            return f'"{s}"'
        return s

    # ---- plot
    def _open_plot_dialog(self):
        dlg = PlotTableDialog(self, self._src_model, self._proxy)
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    # ---- export csv (your original logic, but respects current filtered view)
    def _export_csv(self):
        # Prefer already-exported CSV from metadata when available, otherwise prompt
        existing = getattr(self.document, "metadata", {}).get("table_csv")
        if existing and os.path.exists(existing):
            dst, ok = QFileDialog.getSaveFileName(
                self,
                self.tr("Save CSV As…"),
                os.path.basename(existing),
                self.tr("CSV Files (*.csv)")
            )
            if ok and dst:
                try:
                    import shutil
                    shutil.copyfile(existing, dst)
                except Exception as e:
                    QMessageBox.warning(self, self.tr("Export CSV"), self.tr("Failed to copy CSV:\n{0}").format(e))
            return

        dst, ok = QFileDialog.getSaveFileName(
            self,
            self.tr("Export CSV…"),
            "table.csv",
            self.tr("CSV Files (*.csv)")
        )
        if not ok or not dst:
            return

        try:
            cols = self._src_model.columnCount()
            hdrs = [self._src_model.headerData(c, Qt.Orientation.Horizontal) for c in range(cols)]
            hdrs = [str(h) for h in hdrs]

            with open(dst, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(hdrs)

                # export visible rows ONLY (respects filter)
                for pr in range(self._proxy.rowCount()):
                    sidx0 = self._proxy.mapToSource(self._proxy.index(pr, 0))
                    sr = sidx0.row()
                    row = [_safe_str(self._src_model.raw_value(sr, c)) for c in range(cols)]
                    w.writerow(row)

        except Exception as e:
            QMessageBox.warning(self, self.tr("Export CSV"), self.tr("Failed to export CSV:\n{0}").format(e))