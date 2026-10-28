#saspro/mdi_widgets.py
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

from PyQt6 import sip
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
                    if sip.isdeleted(tgt):
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

    Shutdown safety
    ---------------
    Third-party libraries (colorama in particular) snapshot whatever
    ``sys.stdout`` happens to be when they initialise, hold that reference
    for the life of the process, and poke at it from an ``atexit`` callback.
    By the time those callbacks run, Qt has already destroyed the C++ side of
    this QObject, so any attribute access that touches Qt raises
    ``RuntimeError: wrapped C/C++ object ... has been deleted``.

    Every file-like method below is therefore written to degrade quietly
    rather than raise: a deleted wrapper simply reports itself as a closed,
    non-tty stream.
    """
    text_emitted = pyqtSignal(str)

    def __init__(self, orig_stream, parent=None):
        super().__init__(parent)
        self._orig = orig_stream

    # ------------------------------------------------------------------
    # Lifetime helpers
    # ------------------------------------------------------------------
    def _alive(self) -> bool:
        """True only while the underlying C++ QObject still exists."""
        try:
            return not sip.isdeleted(self)
        except Exception:
            return False

    def detach(self):
        """
        Drop the reference to the wrapped stream.

        Call this when restoring the real ``sys.stdout``/``sys.stderr`` so a
        lingering reference to this wrapper can't keep writing anywhere.
        """
        try:
            object.__setattr__(self, "_orig", None)
        except Exception:
            pass

    def _stream(self):
        """Return the wrapped stream, or None if unavailable/deleted."""
        if not self._alive():
            return None
        try:
            return self._orig
        except Exception:
            return None

    # ------------------------------------------------------------------
    # File-like protocol — these must never raise
    # ------------------------------------------------------------------
    @property
    def closed(self) -> bool:
        s = self._stream()
        if s is None:
            # No wrapped stream, or our C++ object is gone: treat as closed.
            return True
        try:
            return bool(getattr(s, "closed", True))
        except Exception:
            return True

    def isatty(self) -> bool:
        s = self._stream()
        if s is None:
            return False
        try:
            return bool(s.isatty())
        except Exception:
            return False

    def fileno(self):
        # Delegate to the real stream so things like subprocess(stdout=sys.stdout)
        # keep working. Raising OSError is the documented behaviour for streams
        # with no descriptor, and callers are expected to handle it.
        s = self._stream()
        if s is None:
            raise OSError("QtLogStream has no underlying file descriptor")
        try:
            return s.fileno()
        except Exception as exc:
            raise OSError("QtLogStream has no underlying file descriptor") from exc

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def encoding(self):
        s = self._stream()
        try:
            return getattr(s, "encoding", "utf-8") if s is not None else "utf-8"
        except Exception:
            return "utf-8"

    @property
    def errors(self):
        s = self._stream()
        try:
            return getattr(s, "errors", None) if s is not None else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def write(self, text: str):
        # Still write to the original stream
        s = self._stream()
        if s is not None:
            try:
                if not getattr(s, "closed", False):
                    s.write(text)
            except Exception:
                pass

        # Mirror into Qt — but only while the C++ object is alive. Anything
        # printed during interpreter teardown would otherwise blow up here.
        if text and self._alive():
            try:
                self.text_emitted.emit(text)
            except RuntimeError:
                # Object was deleted between the check and the emit.
                pass
            except Exception:
                pass

        return len(text) if text else 0

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        s = self._stream()
        if s is None:
            return
        try:
            if not getattr(s, "closed", False):
                s.flush()
        except Exception:
            pass

    def close(self):
        # Never close the real stdout/stderr on our behalf; just detach.
        self.detach()


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