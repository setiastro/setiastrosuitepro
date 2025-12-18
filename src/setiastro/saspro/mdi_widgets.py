# pro/mdi_widgets.py
"""
MDI-related widgets and supporting classes extracted from main file.

Contains:
- MdiArea: Custom QMdiArea with drag-and-drop support
- ViewLinkController: Synchronizes view transforms across linked windows
- ConsoleListWidget: QListWidget with context menu for console output
- QtLogStream: QObject to mirror stdout/stderr to Qt signals
- _DocProxy: Lightweight proxy for ROI/preview document resolution
"""

import json
import weakref
from typing import Optional

from PyQt6.QtWidgets import QMdiArea, QListWidget, QMenu, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QObject

from setiastro.saspro.dnd_mime import (
    MIME_VIEWSTATE, MIME_CMD, MIME_MASK, MIME_ASTROMETRY, MIME_LINKVIEW
)
from setiastro.saspro.shortcuts import _unpack_cmd_payload


class MdiArea(QMdiArea):
    """
    Custom QMdiArea with support for drag-and-drop of:
    - View states (for duplicating views)
    - Commands with presets (from shortcut toolbar)
    - Masks
    - Astrometry data
    - Link view payloads
    """
    backgroundDoubleClicked = pyqtSignal()
    viewStateDropped = pyqtSignal(dict, object)   # (state_dict, target_subwindow or None)
    commandDropped = pyqtSignal(dict, object)     # ({"command_id","preset"}, target_subwindow or None)
    maskDropped = pyqtSignal(dict, object)        # (payload, target_subwindow or None)
    astrometryDropped = pyqtSignal(dict, object)
    linkViewDropped = pyqtSignal(dict, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if (md.hasFormat(MIME_VIEWSTATE)
                or md.hasFormat(MIME_CMD)
                or md.hasFormat(MIME_MASK)
                or md.hasFormat(MIME_ASTROMETRY)
                or md.hasFormat(MIME_LINKVIEW)):
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dropEvent(self, e):
        pos = e.position().toPoint()

        # Map the event position from the MdiArea into the viewport's coords
        vp = self.viewport()
        vp_pos = vp.mapFrom(self, pos) if vp is not None else pos

        # Get subwindows in real z-order (back → front)
        try:
            order_enum = getattr(QMdiArea, "WindowOrder", None)
            subwins = self.subWindowList(order_enum.StackingOrder) if order_enum else self.subWindowList()
        except Exception:
            subwins = self.subWindowList()

        # Pick the visually top-most window under the cursor
        target = None
        for sw in reversed(subwins):  # reversed: front-most first
            if sw.isVisible() and sw.geometry().contains(vp_pos):
                target = sw
                break

        # 1) View-state payload
        if e.mimeData().hasFormat(MIME_VIEWSTATE):
            try:
                raw = bytes(e.mimeData().data(MIME_VIEWSTATE))
                state = json.loads(raw.decode("utf-8"))
            except Exception:
                e.ignore()
                return
            self.viewStateDropped.emit(state, target)
            e.acceptProposedAction()
            return

        # 2) Command + preset payload (from shortcuts)
        if e.mimeData().hasFormat(MIME_CMD):
            try:
                payload = _unpack_cmd_payload(bytes(e.mimeData().data(MIME_CMD)))
            except Exception:
                e.ignore()
                return
            self.commandDropped.emit(payload, target)
            e.acceptProposedAction()
            return

        # 3) Mask payload (from subwindow DnD)
        if e.mimeData().hasFormat(MIME_MASK):
            try:
                raw = bytes(e.mimeData().data(MIME_MASK))
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                e.ignore()
                return
            self.maskDropped.emit(payload, target)
            e.acceptProposedAction()
            return

        # 4) Astrometric payload (from subwindow DnD)
        if e.mimeData().hasFormat(MIME_ASTROMETRY):
            try:
                raw = bytes(e.mimeData().data(MIME_ASTROMETRY))
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                e.ignore()
                return
            self.astrometryDropped.emit(payload, target)
            e.acceptProposedAction()
            return

        # 5) Link view payload
        if e.mimeData().hasFormat(MIME_LINKVIEW):
            try:
                raw = bytes(e.mimeData().data(MIME_LINKVIEW))
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                e.ignore()
                return
            self.linkViewDropped.emit(payload, target)
            e.acceptProposedAction()
            return

        # Fallback
        super().dropEvent(e)

    def mouseDoubleClickEvent(self, event):
        pt = event.position().toPoint() if hasattr(event, "position") else event.pos()
        for sw in self.subWindowList():
            if sw.geometry().contains(pt):
                return super().mouseDoubleClickEvent(event)
        self.backgroundDoubleClicked.emit()
        event.accept()


class ViewLinkController:
    """
    Controller to synchronize view transforms (zoom, scroll) across linked windows.
    
    Views can be assigned to named groups. When one view's transform changes,
    all other views in the same group are updated to match.
    """
    
    def __init__(self, mdi: QMdiArea):
        self.mdi = mdi
        self.groups: dict[str, set] = {}     # name -> set(views)
        self.by_view: dict = {}               # view -> group name
        self._slots: dict = {}                # view -> callable
        self._broadcasting = False

    def attach_view(self, view):
        """Connect a view's transform change signal to the controller."""
        if view in self._slots:
            return
        slot = lambda scale, h, v, vref=view: self._on_view_transform_from(vref, scale, h, v)
        view.viewTransformChanged.connect(slot)
        self._slots[view] = slot

    def detach_view(self, view):
        """Disconnect and remove a view from any group."""
        slot = self._slots.pop(view, None)
        if slot:
            try:
                view.viewTransformChanged.disconnect(slot)
            except Exception:
                pass
        g = self.by_view.pop(view, None)
        if g and g in self.groups:
            self.groups[g].discard(view)
            if not self.groups[g]:
                self.groups.pop(g, None)

    def set_view_group(self, view, name_or_none: Optional[str]):
        """Assign a view to a named group, or remove from groups if None."""
        old = self.by_view.pop(view, None)
        if old and old in self.groups:
            self.groups[old].discard(view)
            if not self.groups[old]:
                self.groups.pop(old, None)
        if name_or_none:
            self.groups.setdefault(name_or_none, set()).add(view)
            self.by_view[view] = name_or_none

    def group_of(self, view) -> Optional[str]:
        """Return the group name for a view, or None."""
        return self.by_view.get(view)

    def _on_view_transform_from(self, src_view, scale: float, hval: float, vval: float):
        """Handle transform change from a source view - broadcast to group."""
        if self._broadcasting:
            return
        g = self.by_view.get(src_view)
        if not g:
            return

        self._broadcasting = True
        try:
            for tgt in tuple(self.groups.get(g, ())):
                if tgt is src_view:
                    continue
                try:
                    # Skip deleted / half-torn-down views
                    from PyQt6 import sip as _sip
                    if _sip.isdeleted(tgt):
                        continue
                except Exception:
                    pass

                hb = tgt.scroll.horizontalScrollBar().value()
                vb = tgt.scroll.verticalScrollBar().value()
                if abs(scale - tgt.scale) < 1e-9 and int(hval) == hb and int(vval) == vb:
                    continue

                try:
                    tgt.set_view_transform(scale, hval, vval, from_link=True)
                except Exception as ex:
                    print("[link] apply failed:", ex)
        finally:
            self._broadcasting = False


class ConsoleListWidget(QListWidget):
    """
    QListWidget with a context menu for console output:
    - Select All
    - Copy Selected
    - Copy All
    - Clear
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

    def _selected_lines(self) -> list[str]:
        return [itm.text() for itm in self.selectedItems()]

    def _all_lines(self) -> list[str]:
        return [self.item(i).text() for i in range(self.count())]

    def _copy_text(self, lines: list[str]):
        if not lines:
            return
        text = "\n".join(lines)
        cb = QApplication.clipboard()
        cb.setText(text)

    def contextMenuEvent(self, event):
        menu = QMenu(self)

        act_select_all = menu.addAction("Select All")
        act_copy_sel = menu.addAction("Copy Selected")
        act_copy_all = menu.addAction("Copy All")
        menu.addSeparator()
        act_clear = menu.addAction("Clear")

        action = menu.exec(event.globalPos())
        if action is None:
            return

        if action is act_select_all:
            self.selectAll()
        elif action is act_copy_sel:
            self._copy_text(self._selected_lines())
        elif action is act_copy_all:
            self._copy_text(self._all_lines())
        elif action is act_clear:
            self.clear()


class QtLogStream(QObject):
    """
    QObject that intercepts writes (e.g., from stdout/stderr) and emits
    them as Qt signals, while still forwarding to the original stream.
    """
    text_emitted = pyqtSignal(str)

    def __init__(self, orig_stream, parent=None):
        super().__init__(parent)
        self._orig = orig_stream

    def write(self, text: str):
        # Still write to the original stream
        try:
            if self._orig is not None:
                self._orig.write(text)
        except Exception:
            pass
        # Mirror into Qt
        if text:
            self.text_emitted.emit(text)

    def flush(self):
        try:
            if self._orig is not None:
                self._orig.flush()
        except Exception:
            pass


class _DocProxy:
    """
    Lightweight proxy that always resolves to the current document
    for a view (ROI when a Preview/ROI tab is active, else base doc).
    All attribute gets/sets forward to the currently-active target.
    """
    __slots__ = ("_dm", "_view_ref", "_base_doc")

    def __init__(self, doc_manager, view, base_doc):
        self._dm = doc_manager
        self._view_ref = weakref.ref(view)
        self._base_doc = base_doc

    def _target(self):
        view = self._view_ref()
        if view is None:
            return self._base_doc
        doc = self._dm.get_document_for_view(view)
        return doc or self._base_doc

    def __getattr__(self, name):
        return getattr(self._target(), name)

    def __setattr__(self, name, value):
        if name in _DocProxy.__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self._target(), name, value)

    def __repr__(self):
        tgt = self._target()
        try:
            dn = tgt.display_name() if hasattr(tgt, "display_name") else "<doc>"
        except Exception:
            dn = "<doc>"
        return f"<DocProxy → {dn}>"


# Role constant for action data
ROLE_ACTION = Qt.ItemDataRole.UserRole + 1
