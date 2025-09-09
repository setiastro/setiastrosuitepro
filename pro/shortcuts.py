# pro/shortcuts.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Optional
import uuid

from PyQt6.QtCore import (Qt, QPoint, QRect, QMimeData, QSettings, QByteArray,
                          QDataStream, QIODevice, QEvent)
from PyQt6.QtGui import (QAction, QDrag, QIcon, QMouseEvent, QPixmap, QKeyEvent)
from PyQt6.QtWidgets import (QToolBar, QWidget, QToolButton, QMenu, QApplication, QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox,
                             QInputDialog, QMessageBox, QDialog, QSlider,
    QFormLayout, QDialogButtonBox, QDoubleSpinBox, QCheckBox, QLabel, QRubberBand, QRadioButton, QPlainTextEdit, QTabWidget, QLineEdit, QPushButton, QFileDialog)

from PyQt6.QtWidgets import QMdiArea, QMdiSubWindow
from pro.linear_fit import _LinearFitPresetDialog

try:
    import sip
except Exception:
    sip = None
    
from pro.dnd_mime import MIME_VIEWSTATE, MIME_CMD, MIME_MASK, MIME_ACTION

from pathlib import Path
import os  # ← NEW

# Accept these endings (case-insensitive)
OPENABLE_ENDINGS = (
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff",
    ".fits", ".fit",
    ".fits.gz", ".fit.gz", ".fz",
    ".xisf",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".pef",
)




def _is_dead(w) -> bool:
    """True if widget is None or its C++ has been destroyed."""
    if w is None:
        return True
    if sip is not None:
        try:
            return sip.isdeleted(w)
        except Exception:
            return False
    # sip not available: best-effort heuristic
    try:
        _ = w.parent()  # will raise on dead wrappers
        return False
    except RuntimeError:
        return True

# ---------- constants / helpers ----------

SET_KEY_V1 = "Shortcuts/v1"   # legacy (id-less)
SET_KEY_V2 = "Shortcuts/v2"   # new: stores id, label, etc.
SET_KEY = SET_KEY_V2

# Used when dragging a DESKTOP shortcut onto a view for headless run


def _pack_cmd_payload(command_id: str, preset: dict | None = None) -> bytes:
    return json.dumps({"command_id": command_id, "preset": preset or {}}).encode("utf-8")

def _unpack_cmd_payload(b: bytes) -> dict:
    return json.loads(b.decode("utf-8"))


@dataclass
class ShortcutEntry:
    shortcut_id: str
    command_id: str
    x: int
    y: int
    label: str

# ---------- a QToolBar that supports Alt+drag to create shortcuts ----------
class DraggableToolBar(QToolBar):
    """
    Alt/Ctrl/Shift + Left-drag a toolbar button to create a desktop shortcut.
    We hook QToolButton children (not the toolbar itself), because
    mouse events go to the buttons.
    """
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._press_pos: dict[QToolButton, QPoint] = {}
        self._dragging_from: QToolButton | None = None
        self._press_had_mod: dict[QToolButton, bool] = {}
        self._suppress_release: set[QToolButton] = set()

    def _mods_ok(self, mods: Qt.KeyboardModifiers) -> bool:
        return bool(mods & (
            Qt.KeyboardModifier.AltModifier |
            Qt.KeyboardModifier.ControlModifier |
            Qt.KeyboardModifier.ShiftModifier
        ))

    # install/remove our event filter when actions are added/removed
    def actionEvent(self, e):
        super().actionEvent(e)
        t = e.type()
        if t == QEvent.Type.ActionAdded:
            act = e.action()
            btn = self.widgetForAction(act)
            if isinstance(btn, QToolButton):
                btn.installEventFilter(self)
        elif t == QEvent.Type.ActionRemoved:
            act = e.action()
            btn = self.widgetForAction(act)
            if isinstance(btn, QToolButton):
                try:
                    btn.removeEventFilter(self)
                except Exception:
                    pass

    def eventFilter(self, obj, ev):
        if isinstance(obj, QToolButton):
            # L-press: remember start + whether a drag-modifier was held
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._press_pos[obj] = ev.globalPosition().toPoint()
                self._press_had_mod[obj] = self._mods_ok(QApplication.keyboardModifiers())
                return False  # allow normal press visuals

            # Move with L held: if (had-mod at press OR has-mod now) AND moved enough → start drag
            if ev.type() == QEvent.Type.MouseMove and (ev.buttons() & Qt.MouseButton.LeftButton):
                start = self._press_pos.get(obj)
                if start is not None:
                    delta = ev.globalPosition().toPoint() - start
                    if ((self._press_had_mod.get(obj, False) or self._mods_ok(QApplication.keyboardModifiers()))
                        and delta.manhattanLength() > QApplication.startDragDistance()):
                        # find the QAction backing this button
                        act = next((a for a in self.actions() if self.widgetForAction(a) is obj), None)
                        if act:
                            self._start_drag_for_action(act)
                            # eat subsequent release so the action doesn't trigger
                            self._suppress_release.add(obj)
                        # clear press tracking
                        self._press_pos.pop(obj, None)
                        self._press_had_mod.pop(obj, None)
                        return True  # consume the move (prevents click)
                return False

            # Release: if we started a drag, swallow the release so click won't fire
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._press_pos.pop(obj, None)
                self._press_had_mod.pop(obj, None)
                if obj in self._suppress_release:
                    self._suppress_release.discard(obj)
                    return True   # eat release → no click
                return False

        return super().eventFilter(obj, ev)

    def _start_drag_for_action(self, act: QAction):
        act_id = act.property("command_id") or act.objectName()
        if not act_id:
            return
        mime = QMimeData()
        mime.setData(MIME_ACTION, act_id.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(mime)

        pm = act.icon().pixmap(32, 32) if not act.icon().isNull() else QPixmap(32, 32)
        if pm.isNull():
            pm = QPixmap(32, 32); pm.fill(Qt.GlobalColor.darkGray)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)


# ---------- the button that sits on the MDI desktop ----------
class ShortcutButton(QToolButton):
    def __init__(self,
                 manager: "ShortcutManager",
                 sid: str,                 # NEW
                 command_id: str,
                 icon: QIcon,
                 label: str,               # NEW (display text)
                 parent: QWidget):
        super().__init__(parent)
        self._mgr = manager
        self.sid = sid                    # NEW
        self.command_id = command_id
        self.setIcon(icon)
        self.setText(label)               # use label instead of action text
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setIconSize(QPixmap(32, 32).size())
        self.setAutoRaise(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self._dragging = False
        self._press_pos = None
        self._start_geom = None
        self._did_command_drag = False
        self.setToolTip(
            f"{label}\n• Double-click: open\n• Drag: move\n• Alt/Ctrl+Drag onto a view: headless apply"
        )

    # --- Preset helpers (QSettings) -------------------------------------
    def _preset_key(self) -> str:
        # per-instance key
        return f"presets/shortcuts/{self.sid}"

    def _load_preset(self) -> Optional[dict]:
        s = getattr(self._mgr, "settings", QSettings())
        raw = s.value(self._preset_key(), "", type=str) or ""
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        # fallback: legacy per-command preset if instance hasn’t been saved yet
        legacy = s.value(f"presets/{self.command_id}", "", type=str) or ""
        if legacy:
            try:
                return json.loads(legacy)
            except Exception:
                pass
        return None

    def _save_preset(self, preset: Optional[dict]):
        s = getattr(self._mgr, "settings", QSettings())
        if preset is None:
            s.remove(self._preset_key())
        else:
            s.setValue(self._preset_key(), json.dumps(preset))
        s.sync()

    # --- Context menu (run / preset / delete) ----------------------------
    def _context_menu(self, pos):
        m = QMenu(self)
        m.addAction("Run", lambda: self._mgr.trigger(self.command_id))
        m.addSeparator()
        m.addAction("Edit Preset…", self._edit_preset_ui)
        m.addAction("Clear Preset", lambda: self._save_preset(None))
        m.addAction("Rename…", self._rename)                    # ← NEW
        m.addSeparator()
        m.addAction("Delete", self._delete)
        m.exec(self.mapToGlobal(pos))

    def _rename(self):
        current = self.text()
        new_name, ok = QInputDialog.getText(self, "Rename Shortcut", "Name:", text=current)
        if not ok or not new_name.strip():
            return
        self.setText(new_name.strip())
        self._mgr.update_label(self.sid, new_name.strip())   # ← was self.shortcut_id

    def _edit_preset_ui(self):
        """
        Small inline editors per command. Grows as we add more commands.
        Falls back to a simple JSON editor if command unhandled.
        """
        cid = self.command_id
        cur = self._load_preset() or {}

        if cid == "stat_stretch":
            # load current values
            cur = self._load_preset() or {}

            # launch the proper dialog
            dlg = _StatStretchPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                preset = dlg.result_dict()
                self._save_preset(preset)
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return

        if cid == "star_stretch":
            dlg = _StarStretchPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return

        if cid == "linear_fit":
            cur = self._load_preset() or {}
            dlg = _LinearFitPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Linear Fit preset stored on shortcut.")
            return

        if cid == "remove_green":
            dlg = _RemoveGreenPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return

        if cid == "background_neutral":
            dlg = _BackgroundNeutralPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return
        if cid == "white_balance":
            dlg = _WhiteBalancePresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return
        if cid == "wavescale_hdr":
            dlg = _WaveScaleHDRPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return   
        # accept either id for Dark Enhancer
        if cid in ("wavescale_dark_enhance", "wavescale_dark_enhancer"):
            dlg = _WaveScaleDarkEnhancerPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return  
        if cid == "clahe":
            dlg = _CLAHEPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return   
        if cid == "morphology":
            dlg = _MorphologyPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return   
        if cid == "pixel_math":
            dlg = _PixelMathPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return       
        if cid in ("signature_insert", "signature_adder", "signature"):
            dlg = _SignatureInsertPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return   
        if cid == "halo_b_gon":
            dlg = _HaloBGonPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return   
        if cid in ("geom_rescale", "rescale"):
            dlg = _RescalePresetDialog(self, initial=cur or {"factor": 1.0})
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return   
        if cid == "debayer":
            cur = self._load_preset() or {"pattern": "auto"}
            dlg = _DebayerPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Debayer preset stored on shortcut.")
            return        
        if cid == "image_combine":
            dlg = _ImageCombinePresetDialog(self, initial=cur or {
                "mode": "Blend", "opacity": 1.0, "luma_only": False, "output": "replace", "docB_title": ""
            })
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return       
        if cid in ("star_spikes", "diffraction_spikes"):
            dlg = _StarSpikesPresetDialog(self, initial=cur)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._save_preset(dlg.result_dict())
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            return                                             
        # --- Fallback: JSON text prompt (simple & pragmatic) -------------
        raw = json.dumps(cur or {}, indent=2)
        text, ok = QInputDialog.getMultiLineText(self, "Edit Preset (JSON)", "Preset:", raw)
        if ok:
            try:
                preset = json.loads(text or "{}")
                if not isinstance(preset, dict):
                    raise ValueError("Preset must be a JSON object")
                self._save_preset(preset)
                QMessageBox.information(self, "Preset saved", "Preset stored on shortcut.")
            except Exception as e:
                QMessageBox.warning(self, "Invalid JSON", str(e))


    def _start_command_drag(self):
        md = QMimeData()
     
        md.setData(MIME_CMD, _pack_cmd_payload(self.command_id, self._load_preset() or {}))
        drag = QDrag(self)
        drag.setMimeData(md)
        pm = self.icon().pixmap(32, 32)
        if pm.isNull():
            pm = QPixmap(32, 32); pm.fill(Qt.GlobalColor.darkGray)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)
        self._did_command_drag = True

    # --- Mouse handlers --------------------------------------------------
    def _mods_mean_command_drag(self) -> bool:
        # Use ALT only for headless drag so Ctrl/Shift can be used for multiselect
        return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            mods = QApplication.keyboardModifiers()

            if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                self._mgr.toggle_select(self.sid)            # ← was self.shortcut_id
                return

            if self.sid not in self._mgr.selected:           # ← was self.shortcut_id
                self._mgr.select_only(self.sid)

            self._dragging = True
            self._press_pos = e.globalPosition().toPoint()
            self._last_drag_pos = self._press_pos
            self._did_command_drag = False

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._dragging and self._press_pos is not None:
            cur = e.globalPosition().toPoint()
            step = cur - self._last_drag_pos
            if step.manhattanLength() < QApplication.startDragDistance():
                return super().mouseMoveEvent(e)

            # If exactly 1 selected and ALT held → command drag (headless)
            if len(self._mgr.selected) == 1 and self._mods_mean_command_drag():
                self._start_command_drag()
                return

            # Otherwise: move the whole selection by step delta
            self._mgr.move_selected_by(step.x(), step.y())
            self._last_drag_pos = cur
            return

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if self._dragging and e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            if not self._did_command_drag:
                self._mgr.save_shortcuts()  # persist positions after move
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e: QMouseEvent):
        # double-click still runs the action (open dialog)
        self._mgr.trigger(self.command_id)

    def _delete(self):
        self._mgr.delete_by_id(self.sid, persist=True)       # ← was command_id


# ---------- overlay canvas that sits on top of QMdiArea.viewport() ----------
class ShortcutCanvas(QWidget):
    def __init__(self, mgr: "ShortcutManager", parent: QWidget):
        super().__init__(parent)
        self._mgr = mgr
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setGeometry(parent.rect())
        parent.installEventFilter(self)   # keep in sync with viewport size
        
        # NEW: rubber-band selection
        self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rubber_origin = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # to receive Delete/Ctrl+A  
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)      

    def eventFilter(self, obj, ev):
        # keep sized with viewport
        if obj is self.parent() and ev.type() == ev.Type.Resize:
            self.setGeometry(self.parent().rect())
        return super().eventFilter(obj, ev)

    # --- rubber-band selection on empty space ---
    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            local = e.position().toPoint()
            # If click hits no child (shortcut), start rubber-band
            if self.childAt(local) is None:
                self._rubber_origin = local
                self._rubber.setGeometry(QRect(self._rubber_origin, self._rubber_origin))
                self._rubber.show()
                # if no add/toggle mods, clear selection first
                if not (QApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)):
                    self._mgr.clear_selection()
                self.setFocus()
                e.accept()
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._rubber.isVisible() and self._rubber_origin is not None:
            rect = QRect(self._rubber_origin, e.position().toPoint()).normalized()
            self._rubber.setGeometry(rect)
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if self._rubber.isVisible() and self._rubber_origin is not None:
            rect = QRect(self._rubber_origin, e.position().toPoint()).normalized()
            mode = "add" if (QApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)) else "replace"
            self._rubber.hide()
            self._rubber_origin = None
            self._mgr.select_in_rect(rect, mode=mode)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    # --- keyboard: Delete / Backspace / Ctrl+A ---
    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._mgr.delete_selected()
            e.accept(); return
        if e.key() == Qt.Key.Key_A and (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._mgr.select_in_rect(self.rect(), mode="replace")
            e.accept(); return
        super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasFormat(MIME_ACTION):
            self.raise_()
            e.acceptProposedAction()
        elif self._md_has_openable_urls(md):
            e.acceptProposedAction()
        else:
            # IMPORTANT: don't accept command drags here (let subwindow get it)
            e.ignore()



    def _top_subwindow_at(self, vp_pos: QPoint) -> QMdiSubWindow | None:
        # iterate by stacking order: last is front-most
        for sw in reversed(self._mgr.mdi.subWindowList(QMdiArea.SubWindowViewOrder.StackingOrder)):
            if not sw.isVisible():
                continue
            # QMdiSubWindow geometry is in the viewport’s coord space
            if sw.geometry().contains(vp_pos):
                return sw
        return None

    def _forward_command_drop(self, e) -> bool:
        md = e.mimeData()
        if not md.hasFormat(MIME_CMD):
            return False
        sw = self._top_subwindow_at(e.position().toPoint())
        if sw is None:
            return False
        # let manager apply the command to that subwindow/view
        try:
            payload = _unpack_cmd_payload(bytes(md.data(MIME_CMD)))
        except Exception:
            return False
        self._mgr.apply_command_to_subwindow(sw, payload)
        e.acceptProposedAction()
        return True

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME_ACTION) or self._md_has_openable_urls(e.mimeData()):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.lower()                           # restore
        super().dragLeaveEvent(e)

    def dropEvent(self, e):
        md = e.mimeData()

        # 1) route function/preset drops to the front-most subwindow under cursor
        if self._forward_command_drop(e):
            return

        # 2) desktop shortcut creation (MIME_ACTION) → create a button
        if md.hasFormat(MIME_ACTION):
            act_id = bytes(md.data(MIME_ACTION)).decode("utf-8")
            self._mgr.add_shortcut(act_id, e.position().toPoint())
            e.acceptProposedAction()
            self.lower()
            return
    
        # File / folder open
        if self._md_has_openable_urls(md):
            paths = self._collect_openable_files_from_urls(md)
            if paths:
                opener = getattr(self._mgr.mw, "_handle_external_file_drop", None)
                if callable(opener):
                    opener(paths)
                else:
                    dm = getattr(self._mgr.mw, "docman", None)
                    if dm and hasattr(dm, "open_files") and callable(dm.open_files):
                        docs = dm.open_files(paths)
                        try:
                            for d in (docs or []):
                                self._mgr.mw._spawn_subwindow_for(d)
                        except Exception:
                            pass
                    elif dm and hasattr(dm, "open_path") and callable(dm.open_path):
                        for p in paths:
                            doc = dm.open_path(p)
                            if doc is not None:
                                self._mgr.mw._spawn_subwindow_for(doc)
            e.acceptProposedAction()
            return

        e.ignore()  

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        has_sel = bool(self._mgr.selected)
        a_del = menu.addAction("Delete Selected", self._mgr.delete_selected); a_del.setEnabled(has_sel)
        a_clr = menu.addAction("Clear Selection", self._mgr.clear_selection); a_clr.setEnabled(has_sel)
        menu.exec(e.globalPos())

    def mouseDoubleClickEvent(self, e):
        # If user double-clicks empty canvas area, forward to MDI's handler
        if e.button() == Qt.MouseButton.LeftButton:
            local = e.position().toPoint()
            if self.childAt(local) is None:
                try:
                    # Reuse your existing connection: mdi.backgroundDoubleClicked -> open_files
                    self._mgr.mdi.backgroundDoubleClicked.emit()
                except Exception:
                    pass
                e.accept()
                return
        super().mouseDoubleClickEvent(e)

    def _is_openable_path(self, path: str) -> bool:
        return path.lower().endswith(OPENABLE_ENDINGS)

    def _md_has_openable_urls(self, md) -> bool:
        if not md.hasUrls():
            return False
        for u in md.urls():
            if not u.isLocalFile():
                continue
            p = u.toLocalFile()
            if os.path.isdir(p):
                return True  # we'll scan it on drop
            if self._is_openable_path(p):
                return True
        return False

    def _collect_openable_files_from_urls(self, md) -> list[str]:
        files: list[str] = []
        if not md.hasUrls():
            return files
        for u in md.urls():
            if not u.isLocalFile():
                continue
            p = u.toLocalFile()
            if os.path.isdir(p):
                # recurse folder for matching files
                for root, _, names in os.walk(p):
                    for name in names:
                        fp = os.path.join(root, name)
                        if self._is_openable_path(fp):
                            files.append(fp)
            else:
                if self._is_openable_path(p):
                    files.append(p)
        return files


class ShortcutManager:
    def __init__(self, mdi_area, main_window):
        # mdi_area should be your QMdiArea; we attach to its viewport
        self.mdi = mdi_area
        self.mw = main_window
        self.registry: Dict[str, QAction] = {}
        self.canvas = ShortcutCanvas(self, self.mdi.viewport())
        self.canvas.lower()  # keep below subwindows (raise() if you want pinned-on-top)
        self.canvas.show()
        self.widgets: Dict[str, ShortcutButton] = {}
        self.settings = QSettings()  # shared settings store for positions + presets
        self.selected: set[str] = set()  # ← set of shortcut_ids

    # ---- registry ----
    def register_action(self, command_id: str, action: QAction):
        action.setProperty("command_id", command_id)
        if not action.objectName():
            action.setObjectName(command_id)
        self.registry[command_id] = action

    def trigger(self, command_id: str):
        act = self.registry.get(command_id)
        if act:
            act.trigger()

    def _on_widget_destroyed(self, sid: str):
        # Called from QObject.destroyed — never touch the widget, just clean maps
        self.widgets.pop(sid, None)
        self.selected.discard(sid)

    # ---- CRUD for shortcuts --------------------------------------------
    def _default_label_for(self, command_id: str) -> str:
        act = self.registry.get(command_id)
        if not act:
            return command_id
        return (act.text() or act.toolTip() or command_id).strip() or command_id

    def add_shortcut(self,
                     command_id: str,
                     pos: QPoint,
                     *,
                     label: Optional[str] = None,
                     shortcut_id: Optional[str] = None):
        """
        Always creates a NEW instance (multiple per command_id allowed).
        """
        act = self.registry.get(command_id)
        if not act:
            return

        sid = shortcut_id or uuid.uuid4().hex
        lbl = (label or self._default_label_for(command_id)).strip() or command_id

        w = ShortcutButton(self, sid, command_id, act.icon(), lbl, self.canvas)  # ← FIXED SIG
        w.adjustSize()
        w.move(pos)
        w.show()

        # when the C++ object dies, clean maps using the SID
        w.destroyed.connect(lambda _=None, sid=sid: self._on_widget_destroyed(sid))

        self.widgets[sid] = w
        self.save_shortcuts()

    def update_label(self, shortcut_id: str, new_label: str):
        w = self.widgets.get(shortcut_id)
        if w and not _is_dead(w):
            w.setText(new_label.strip())  # in case caller didn't already
        self.save_shortcuts()

    def remove(self, shortcut_id: str):
        if shortcut_id in self.widgets:
            self.widgets.pop(shortcut_id, None)
            self.save_shortcuts()

    # ---- persistence (QSettings JSON blob) ----
    def save_shortcuts(self):
        data = []
        for sid, w in list(self.widgets.items()):
            if _is_dead(w):
                self.widgets.pop(sid, None)
                self.selected.discard(sid)
                continue
            try:
                if not w.isVisible():
                    continue
                p = w.pos()
                data.append({
                    "id": sid,
                    "command_id": w.command_id,
                    "label": w.text(),
                    "x": p.x(),
                    "y": p.y(),
                })
            except RuntimeError:
                self.widgets.pop(sid, None)
                self.selected.discard(sid)

        # Save new format and remove legacy
        self.settings.setValue(SET_KEY_V2, json.dumps(data))
        self.settings.remove(SET_KEY_V1)
        self.settings.sync()

    def load_shortcuts(self):
        # try v2 first
        raw_v2 = self.settings.value(SET_KEY_V2, "", type=str) or ""
        if raw_v2:
            try:
                arr = json.loads(raw_v2)
                for entry in arr:
                    sid = entry.get("id") or uuid.uuid4().hex
                    cid = entry.get("command_id")
                    x = int(entry.get("x", 10))
                    y = int(entry.get("y", 10))
                    label = entry.get("label") or self._default_label_for(cid)
                    self.add_shortcut(cid, QPoint(x, y), label=label, shortcut_id=sid)
                return
            except Exception as e:
                try:
                    self.mw._log(f"Shortcuts v2: failed to load ({e})")
                except Exception:
                    pass

        # migrate v1 (positions only)
        raw_v1 = self.settings.value(SET_KEY_V1, "", type=str) or ""
        if not raw_v1:
            return
        try:
            arr = json.loads(raw_v1)
            for entry in arr:
                cid = entry.get("id") or entry.get("command_id")  # old key was "id" = command_id
                x = int(entry.get("x", 10))
                y = int(entry.get("y", 10))
                # each old entry becomes its own instance
                sid = uuid.uuid4().hex
                label = self._default_label_for(cid)
                self.add_shortcut(cid, QPoint(x, y), label=label, shortcut_id=sid)
            # after migrating, persist as v2
            self.save_shortcuts()
        except Exception as e:
            try:
                self.mw._log(f"Shortcuts v1: failed to migrate ({e})")
            except Exception:
                pass

    def apply_command_to_subwindow(self, subwin, payload: dict):
        """Apply a dragged command to the specific subwindow under the pointer."""
        cmd = payload.get("command_id")
        preset = payload.get("preset", {}) or {}
        if not cmd:
            return
        # Try to get the subwindow’s content widget
        w = getattr(subwin, "widget", None)
        target = w() if callable(w) else w
        # Your app-specific hook to run `cmd` on `target`
        # Prefer a precise API; fall back to main window’s dispatcher.
        if hasattr(target, "apply_command"):
            target.apply_command(cmd, preset)
        elif hasattr(self.mw, "apply_command_to_view"):
            self.mw.apply_command_to_view(target, cmd, preset)
        elif hasattr(self.mw, "run_command"):
            self.mw.run_command(cmd, preset, view=target)
        else:
            # Last resort: activate then trigger global action
            self.mdi.setActiveSubWindow(subwin)
            act = self.registry.get(cmd)
            if act:
                act.trigger()

    def clear(self):
        for sid, w in list(self.widgets.items()):
            try:
                if not _is_dead(w):
                    w.hide()
                    w.deleteLater()
            except RuntimeError:
                pass
        self.widgets.clear()
        self.selected.clear()
        self.settings.setValue(SET_KEY_V2, "[]")
        self.settings.remove(SET_KEY_V1)
        self.settings.sync()
    # ---------- selection ----------
    def _apply_sel_visual(self, sid: str, on: bool):
        w = self.widgets.get(sid)
        if _is_dead(w):
            # Clean up any stale references
            self.widgets.pop(sid, None)
            self.selected.discard(sid)
            return
        try:
            if on:
                w.setStyleSheet("QToolButton { border: 2px solid #4da3ff; border-radius: 6px; padding: 2px; }")
            else:
                w.setStyleSheet("")
        except RuntimeError:
            # C++ object died between get() and call
            self.widgets.pop(sid, None)
            self.selected.discard(sid)

    def clear_selection(self):
        # copy to avoid mutating while iterating
        for sid in list(self.selected):
            self._apply_sel_visual(sid, False)
        self.selected.clear()

    def select_only(self, sid: str):
        self.clear_selection()
        self.selected.add(sid)
        self._apply_sel_visual(sid, True)

    def toggle_select(self, sid: str):
        if sid in self.selected:
            self.selected.remove(sid)
            self._apply_sel_visual(sid, False)
        else:
            self.selected.add(sid)
            self._apply_sel_visual(sid, True)

    def select_in_rect(self, rect: QRect, *, mode: str = "replace"):
        if mode == "replace":
            self.clear_selection()
        for sid, w in list(self.widgets.items()):
            if _is_dead(w):
                self.widgets.pop(sid, None)
                self.selected.discard(sid)
                continue
            if rect.intersects(w.geometry()):
                if sid not in self.selected:
                    self.selected.add(sid)
                    self._apply_sel_visual(sid, True)

    def selected_widgets(self):
        out = []
        for sid in list(self.selected):
            w = self.widgets.get(sid)
            if _is_dead(w):
                self.widgets.pop(sid, None)
                self.selected.discard(sid)
                continue
            out.append(w)
        return out

    def clear(self):
        for sid, w in list(self.widgets.items()):
            try:
                if not _is_dead(w):
                    w.hide()
                    try:
                        w.setParent(None)   # ← detach from canvas immediately
                    except Exception:
                        pass
                    w.deleteLater()
            except RuntimeError:
                pass
        self.widgets.clear()
        self.selected.clear()
        self.settings.setValue(SET_KEY_V2, "[]")
        self.settings.remove(SET_KEY_V1)
        self.settings.sync()
        try:
            self.canvas.update()  # nudge repaint
        except Exception:
            pass


    # ---------- group move / delete ----------
    def _group_bounds(self) -> QRect:
        rect = None
        for w in self.selected_widgets():
            rect = w.geometry() if rect is None else rect.united(w.geometry())
        return rect if rect is not None else QRect()

    def move_selected_by(self, dx: int, dy: int):
        if not self.selected:
            return
        # clamp whole group to canvas bounds so relative spacing stays intact
        group = self._group_bounds()
        vp = self.canvas.rect()
        min_dx = vp.left()  - group.left()
        max_dx = vp.right() - group.right()
        min_dy = vp.top()   - group.top()
        max_dy = vp.bottom()- group.bottom()
        dx = max(min_dx, min(dx, max_dx))
        dy = max(min_dy, min(dy, max_dy))
        if dx == 0 and dy == 0:
            return
        for w in self.selected_widgets():
            g = w.geometry()
            g.translate(dx, dy)
            w.setGeometry(g)

    def delete_by_id(self, sid: str, *, persist: bool = True):
        self.selected.discard(sid)
        w = self.widgets.pop(sid, None)
        if not _is_dead(w):
            try:
                w.hide()
            except RuntimeError:
                pass
            try:
                w.deleteLater()
            except RuntimeError:
                pass
        if persist:
            self.save_shortcuts()

    def delete_selected(self):
        # bulk delete, then persist once
        for sid in list(self.selected):
            self.delete_by_id(sid, persist=False)
        self.selected.clear()
        self.save_shortcuts()

    def remove(self, sid: str):
        # legacy single-remove (kept for callers)
        self.delete_by_id(sid, persist=True)


class _StatStretchPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Stretch — Preset")
        init = dict(initial or {})

        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(0.0, 1.0); self.spin_target.setDecimals(3)
        self.spin_target.setSingleStep(0.01)
        self.spin_target.setValue(float(init.get("target_median", 0.25)))

        self.chk_linked = QCheckBox("Linked RGB channels")
        self.chk_linked.setChecked(bool(init.get("linked", False)))

        self.chk_normalize = QCheckBox("Normalize to [0..1]")
        self.chk_normalize.setChecked(bool(init.get("normalize", False)))

        self.spin_curves = QDoubleSpinBox()
        self.spin_curves.setRange(0.0, 1.0); self.spin_curves.setDecimals(2)
        self.spin_curves.setSingleStep(0.05)
        self.spin_curves.setValue(float(init.get("curves_boost", 0.0 if not init.get("apply_curves") else 0.20)))

        form = QFormLayout(self)
        form.addRow("Target median:", self.spin_target)
        form.addRow("", self.chk_linked)
        form.addRow("", self.chk_normalize)
        form.addRow("Curves boost (0–1):", self.spin_curves)
        form.addRow(QLabel("Curves are applied only if boost > 0."))

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        boost = float(self.spin_curves.value())
        return {
            "target_median": float(self.spin_target.value()),
            "linked": bool(self.chk_linked.isChecked()),
            "normalize": bool(self.chk_normalize.isChecked()),
            "apply_curves": bool(boost > 0.0),
            "curves_boost": boost,
        }
    

class _StarStretchPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Star Stretch — Preset")
        init = dict(initial or {})

        self.spin_amount = QDoubleSpinBox()
        self.spin_amount.setRange(0.0, 8.0); self.spin_amount.setDecimals(2)
        self.spin_amount.setSingleStep(0.05)
        self.spin_amount.setValue(float(init.get("stretch_factor", 5.00)))

        self.spin_sat = QDoubleSpinBox()
        self.spin_sat.setRange(0.0, 2.0); self.spin_sat.setDecimals(2)
        self.spin_sat.setSingleStep(0.05)
        self.spin_sat.setValue(float(init.get("color_boost", 1.00)))

        self.chk_scnr = QCheckBox("Remove Green via SCNR")
        self.chk_scnr.setChecked(bool(init.get("scnr_green", False)))

        form = QFormLayout(self)
        form.addRow("Stretch amount (0–8):", self.spin_amount)
        form.addRow("Color boost (0–2):", self.spin_sat)
        form.addRow("", self.chk_scnr)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "stretch_factor": float(self.spin_amount.value()),   # 0..8
            "color_boost":    float(self.spin_sat.value()),      # 0..2
            "scnr_green":     bool(self.chk_scnr.isChecked()),
        }

class _RemoveGreenPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Remove Green — Preset")
        init = dict(initial or {})

        self.spin_amount = QDoubleSpinBox()
        self.spin_amount.setRange(0.0, 1.0)
        self.spin_amount.setDecimals(2)
        self.spin_amount.setSingleStep(0.05)
        # default full SCNR if nothing saved
        self.spin_amount.setValue(float(init.get("amount", 1.00)))

        form = QFormLayout(self)
        form.addRow("Amount (0–1):", self.spin_amount)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "amount": float(self.spin_amount.value())  # 0..1
        }

class _BackgroundNeutralPresetDialog(QDialog):
    """
    Preset UI for Background Neutralization:
      • Mode: Auto (default) or Rectangle
      • Rect (normalized): x, y, w, h in [0..1]
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Background Neutralization — Preset")
        init = dict(initial or {})

        # Mode radios
        self.radio_auto = QRadioButton("Auto (50×50 finder)")
        self.radio_rect = QRadioButton("Rectangle (normalized coords)")
        mode = (init.get("mode") or "auto").lower()
        if mode == "rect":
            self.radio_rect.setChecked(True)
        else:
            self.radio_auto.setChecked(True)

        # Rect spinboxes (normalized 0..1)
        rn = init.get("rect_norm") or [0.40, 0.60, 0.08, 0.06]
        self.spin_x = QDoubleSpinBox(); self._cfg_norm_box(self.spin_x, rn[0])
        self.spin_y = QDoubleSpinBox(); self._cfg_norm_box(self.spin_y, rn[1])
        self.spin_w = QDoubleSpinBox(); self._cfg_norm_box(self.spin_w, rn[2])
        self.spin_h = QDoubleSpinBox(); self._cfg_norm_box(self.spin_h, rn[3])

        form = QFormLayout(self)
        form.addRow(self.radio_auto)
        form.addRow(self.radio_rect)
        form.addRow("x (0..1):", self.spin_x)
        form.addRow("y (0..1):", self.spin_y)
        form.addRow("w (0..1):", self.spin_w)
        form.addRow("h (0..1):", self.spin_h)

        # Enable/disable rect fields based on mode
        self.radio_auto.toggled.connect(self._update_enabled)
        self.radio_rect.toggled.connect(self._update_enabled)
        self._update_enabled()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _cfg_norm_box(self, box: QDoubleSpinBox, val: float):
        box.setRange(0.0, 1.0)
        box.setDecimals(3)
        box.setSingleStep(0.01)
        try:
            box.setValue(float(val))
        except Exception:
            box.setValue(0.0)

    def _update_enabled(self):
        on = self.radio_rect.isChecked()
        for w in (self.spin_x, self.spin_y, self.spin_w, self.spin_h):
            w.setEnabled(on)

    def result_dict(self) -> dict:
        if self.radio_auto.isChecked():
            return {"mode": "auto"}
        # sanitize/cap in [0,1]
        x = max(0.0, min(1.0, float(self.spin_x.value())))
        y = max(0.0, min(1.0, float(self.spin_y.value())))
        w = max(0.0, min(1.0, float(self.spin_w.value())))
        h = max(0.0, min(1.0, float(self.spin_h.value())))
        # ensure at least a 1e-6 nonzero footprint so integer rounding later doesn't zero-out
        if w == 0.0: w = 1e-6
        if h == 0.0: h = 1e-6
        return {"mode": "rect", "rect_norm": [x, y, w, h]}
    
class _WhiteBalancePresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("White Balance — Preset")
        init = dict(initial or {})

        v = QVBoxLayout(self)

        # Mode
        row = QHBoxLayout()
        row.addWidget(QLabel("Mode:"))
        self.mode = QComboBox()
        self.mode.addItems(["Star-Based", "Manual", "Auto"])
        m = (init.get("mode") or "star").lower()
        if m == "manual": self.mode.setCurrentText("Manual")
        elif m == "auto": self.mode.setCurrentText("Auto")
        else: self.mode.setCurrentText("Star-Based")
        row.addWidget(self.mode); row.addStretch()
        v.addLayout(row)

        # Star options
        self.grp_star = QGroupBox("Star-Based")
        sv = QGridLayout(self.grp_star)
        self.spin_thr = QDoubleSpinBox(); self.spin_thr.setRange(0.5, 200.0); self.spin_thr.setDecimals(1)
        self.spin_thr.setSingleStep(0.5); self.spin_thr.setValue(float(init.get("threshold", 50.0)))
        self.chk_reuse = QCheckBox("Reuse cached detections"); self.chk_reuse.setChecked(bool(init.get("reuse_cached_sources", True)))
        sv.addWidget(QLabel("Threshold (σ):"), 0, 0); sv.addWidget(self.spin_thr, 0, 1)
        sv.addWidget(self.chk_reuse, 1, 0, 1, 2)
        v.addWidget(self.grp_star)

        # Manual options
        self.grp_manual = QGroupBox("Manual")
        gv = QGridLayout(self.grp_manual)
        self.r = QDoubleSpinBox(); self._cfg_gain(self.r, float(init.get("r_gain", 1.0)))
        self.g = QDoubleSpinBox(); self._cfg_gain(self.g, float(init.get("g_gain", 1.0)))
        self.b = QDoubleSpinBox(); self._cfg_gain(self.b, float(init.get("b_gain", 1.0)))
        gv.addWidget(QLabel("Red gain:"), 0, 0); gv.addWidget(self.r, 0, 1)
        gv.addWidget(QLabel("Green gain:"), 1, 0); gv.addWidget(self.g, 1, 1)
        gv.addWidget(QLabel("Blue gain:"), 2, 0); gv.addWidget(self.b, 2, 1)
        v.addWidget(self.grp_manual)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addWidget(btns)

        self.mode.currentTextChanged.connect(self._refresh)
        self._refresh()

    def _cfg_gain(self, box: QDoubleSpinBox, val: float):
        box.setRange(0.5, 1.5); box.setDecimals(3); box.setSingleStep(0.01); box.setValue(val)

    def _refresh(self):
        t = self.mode.currentText()
        self.grp_star.setVisible(t == "Star-Based")
        self.grp_manual.setVisible(t == "Manual")

    def result_dict(self) -> dict:
        t = self.mode.currentText()
        if t == "Manual":
            return {"mode": "manual", "r_gain": float(self.r.value()), "g_gain": float(self.g.value()), "b_gain": float(self.b.value())}
        if t == "Auto":
            return {"mode": "auto"}
        return {"mode": "star", "threshold": float(self.spin_thr.value()), "reuse_cached_sources": bool(self.chk_reuse.isChecked())}


class _WaveScaleHDRPresetDialog(QDialog):
    """
    Preset UI for WaveScale HDR:
      • n_scales (2..10)
      • compression_factor (0.10..5.00)
      • mask_gamma (0.10..10.00)
      • decay_rate (0.10..1.00)
      • optional dim_gamma (enable to store; omit to use auto)
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("WaveScale HDR — Preset")
        init = dict(initial or {})

        form = QFormLayout(self)

        self.sp_scales = QSpinBox()
        self.sp_scales.setRange(2, 10)
        self.sp_scales.setValue(int(init.get("n_scales", 5)))

        self.dp_comp = QDoubleSpinBox()
        self.dp_comp.setRange(0.10, 5.00)
        self.dp_comp.setDecimals(2)
        self.dp_comp.setSingleStep(0.05)
        self.dp_comp.setValue(float(init.get("compression_factor", 1.50)))

        self.dp_gamma = QDoubleSpinBox()
        self.dp_gamma.setRange(0.10, 10.00)
        self.dp_gamma.setDecimals(2)
        self.dp_gamma.setSingleStep(0.05)
        # matches slider default of 500 → 5.00
        self.dp_gamma.setValue(float(init.get("mask_gamma", 5.00)))

        self.dp_decay = QDoubleSpinBox()
        self.dp_decay.setRange(0.10, 1.00)
        self.dp_decay.setDecimals(2)
        self.dp_decay.setSingleStep(0.05)
        self.dp_decay.setValue(float(init.get("decay_rate", 0.50)))

        # Optional dim gamma
        row_dim = QHBoxLayout()
        self.chk_dim = QCheckBox("Use custom dim γ")
        self.dp_dim = QDoubleSpinBox()
        self.dp_dim.setRange(0.10, 6.00)
        self.dp_dim.setDecimals(2)
        self.dp_dim.setSingleStep(0.05)
        self.dp_dim.setValue(float(init.get("dim_gamma", 2.00)))
        if "dim_gamma" in init:
            self.chk_dim.setChecked(True)
        self.dp_dim.setEnabled(self.chk_dim.isChecked())
        self.chk_dim.toggled.connect(self.dp_dim.setEnabled)
        row_dim.addWidget(self.chk_dim)
        row_dim.addWidget(self.dp_dim, 1)

        form.addRow("Number of scales:", self.sp_scales)
        form.addRow("Coarse compression:", self.dp_comp)
        form.addRow("Mask gamma:", self.dp_gamma)
        form.addRow("Decay rate:", self.dp_decay)
        form.addRow("Dimming:", row_dim)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        out = {
            "n_scales": int(self.sp_scales.value()),
            "compression_factor": float(self.dp_comp.value()),
            "mask_gamma": float(self.dp_gamma.value()),
            "decay_rate": float(self.dp_decay.value()),
        }
        if self.chk_dim.isChecked():
            out["dim_gamma"] = float(self.dp_dim.value())  # you said you'll add this param
        return out

class _WaveScaleDarkEnhancerPresetDialog(QDialog):
    """
    Preset UI for WaveScale Dark Enhancer:
      • n_scales (2–10)
      • boost_factor (0.10–10.00)
      • mask_gamma (0.10–10.00)
      • iterations (1–10)
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("WaveScale Dark Enhancer — Preset")
        init = dict(initial or {})

        form = QFormLayout(self)

        self.sp_scales = QSpinBox(); self.sp_scales.setRange(2, 10); self.sp_scales.setValue(int(init.get("n_scales", 6)))
        self.dp_boost  = QDoubleSpinBox(); self.dp_boost.setRange(0.10, 10.00); self.dp_boost.setDecimals(2); self.dp_boost.setSingleStep(0.05)
        self.dp_boost.setValue(float(init.get("boost_factor", 5.00)))
        self.dp_gamma  = QDoubleSpinBox(); self.dp_gamma.setRange(0.10, 10.00); self.dp_gamma.setDecimals(2); self.dp_gamma.setSingleStep(0.05)
        self.dp_gamma.setValue(float(init.get("mask_gamma", 1.00)))
        self.sp_iters  = QSpinBox(); self.sp_iters.setRange(1, 10); self.sp_iters.setValue(int(init.get("iterations", 2)))

        form.addRow("Number of scales:", self.sp_scales)
        form.addRow("Boost factor:", self.dp_boost)
        form.addRow("Mask gamma:", self.dp_gamma)
        form.addRow("Iterations:", self.sp_iters)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel,
                                parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "n_scales": int(self.sp_scales.value()),
            "boost_factor": float(self.dp_boost.value()),
            "mask_gamma": float(self.dp_gamma.value()),
            "iterations": int(self.sp_iters.value()),
        }

class _CLAHEPresetDialog(QDialog):
    """
    Preset UI for CLAHE:
      • clip_limit (0.1–4.0)
      • tile (1–32)  → used as (tile, tile)
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("CLAHE — Preset")
        init = dict(initial or {})

        form = QFormLayout(self)

        self.dp_clip = QDoubleSpinBox()
        self.dp_clip.setRange(0.10, 4.00)
        self.dp_clip.setDecimals(2)
        self.dp_clip.setSingleStep(0.10)
        self.dp_clip.setValue(float(init.get("clip_limit", 2.00)))

        self.sp_tile = QSpinBox()
        self.sp_tile.setRange(1, 32)
        self.sp_tile.setValue(int(init.get("tile", 8)))

        form.addRow("Clip limit:", self.dp_clip)
        form.addRow("Tile size:", self.sp_tile)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "clip_limit": float(self.dp_clip.value()),
            "tile": int(self.sp_tile.value()),
        }

class _MorphologyPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Morphology — Preset")
        init = dict(initial or {})

        form = QFormLayout(self)

        self.op = QComboBox()
        self.op.addItems(["Erosion", "Dilation", "Opening", "Closing"])
        op = (init.get("operation","erosion") or "erosion").lower()
        idx = {"erosion":0,"dilation":1,"opening":2,"closing":3}.get(op,0)
        self.op.setCurrentIndex(idx)

        self.k = QSpinBox(); self.k.setRange(1,31); self.k.setSingleStep(2)
        kv = int(init.get("kernel", 3)); self.k.setValue(kv if kv%2==1 else kv+1)

        self.it = QSpinBox(); self.it.setRange(1,10); self.it.setValue(int(init.get("iterations",1)))

        form.addRow("Operation:", self.op)
        form.addRow("Kernel size (odd):", self.k)
        form.addRow("Iterations:", self.it)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        op = ["erosion","dilation","opening","closing"][self.op.currentIndex()]
        k  = int(self.k.value());  k = k if k%2==1 else k+1
        it = int(self.it.value())
        return {"operation": op, "kernel": k, "iterations": it}

class _PixelMathPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Pixel Math — Preset")
        init = dict(initial or {})
        v = QVBoxLayout(self)
        self.rb_single = QRadioButton("Single"); self.rb_single.setChecked(init.get("mode","single")=="single")
        self.rb_rgb    = QRadioButton("Per-channel"); self.rb_rgb.setChecked(init.get("mode","single")=="rgb")
        row = QHBoxLayout(); row.addWidget(self.rb_single); row.addWidget(self.rb_rgb); row.addStretch(1)
        v.addLayout(row)
        self.ed_single = QPlainTextEdit(); self.ed_single.setPlaceholderText("expr"); self.ed_single.setPlainText(init.get("expr",""))
        v.addWidget(self.ed_single)
        self.tabs = QTabWidget(); 
        self.ed_r, self.ed_g, self.ed_b = QPlainTextEdit(), QPlainTextEdit(), QPlainTextEdit()
        for ed, name, key in ((self.ed_r,"Red","expr_r"),(self.ed_g,"Green","expr_g"),(self.ed_b,"Blue","expr_b")):
            w = QWidget(); lay = QVBoxLayout(w); ed.setPlainText(init.get(key,"")); lay.addWidget(ed); self.tabs.addTab(w, name)
        v.addWidget(self.tabs)
        self.rb_single.toggled.connect(lambda on: (self.ed_single.setVisible(on), self.tabs.setVisible(not on)))
        self.rb_single.toggled.emit(self.rb_single.isChecked())
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addWidget(btns)
    def result_dict(self) -> dict:
        if self.rb_single.isChecked():
            return {"mode":"single","expr":self.ed_single.toPlainText().strip()}
        return {"mode":"rgb","expr_r":self.ed_r.toPlainText().strip(),
                "expr_g":self.ed_g.toPlainText().strip(),"expr_b":self.ed_b.toPlainText().strip()}

class _SignatureInsertPresetDialog(QDialog):
    """
    Preset editor for Signature / Insert.
    Keeps the PNG path + placement so users can drag a shortcut and re-apply.
    """
    POS_KEYS = [
        ("Top-Left", "top_left"),
        ("Top-Center", "top_center"),
        ("Top-Right", "top_right"),
        ("Middle-Left", "middle_left"),
        ("Center", "center"),
        ("Middle-Right", "middle_right"),
        ("Bottom-Left", "bottom_left"),
        ("Bottom-Center", "bottom_center"),
        ("Bottom-Right", "bottom_right"),
    ]

    def __init__(self, parent, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Signature / Insert – Preset")
        self.setMinimumWidth(520)

        init = dict(initial or {})
        v = QVBoxLayout(self)

        tip = QLabel("Tip: For transparent signatures, use a PNG and “Load from File”. "
                     "Views are RGB, so alpha is not preserved.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#e0b000;")
        v.addWidget(tip)

        grid = QGridLayout()

        # File path
        grid.addWidget(QLabel("Signature file (PNG/JPG/TIF):"), 0, 0)
        self.ed_path = QLineEdit(init.get("file_path", ""))
        b_browse = QPushButton("Browse…")
        def _pick():
            fp, _ = QFileDialog.getOpenFileName(self, "Select signature image",
                                                "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
            if fp: self.ed_path.setText(fp)
        b_browse.clicked.connect(_pick)
        grid.addWidget(self.ed_path, 0, 1)
        grid.addWidget(b_browse, 0, 2)

        # Position
        grid.addWidget(QLabel("Position:"), 1, 0)
        self.cb_pos = QComboBox()
        for text, key in self.POS_KEYS:
            self.cb_pos.addItem(text, userData=key)
        want = init.get("position", "bottom_right")
        idx = max(0, next((i for i,(_,k) in enumerate(self.POS_KEYS) if k == want), 0))
        self.cb_pos.setCurrentIndex(idx)
        grid.addWidget(self.cb_pos, 1, 1)

        # Margins
        grid.addWidget(QLabel("Margin X (px):"), 2, 0)
        self.sp_mx = QSpinBox(); self.sp_mx.setRange(0, 5000); self.sp_mx.setValue(int(init.get("margin_x", 20)))
        grid.addWidget(self.sp_mx, 2, 1)

        grid.addWidget(QLabel("Margin Y (px):"), 3, 0)
        self.sp_my = QSpinBox(); self.sp_my.setRange(0, 5000); self.sp_my.setValue(int(init.get("margin_y", 20)))
        grid.addWidget(self.sp_my, 3, 1)

        # Scale / Opacity / Rotation
        grid.addWidget(QLabel("Scale (%)"), 4, 0)
        self.sp_scale = QSpinBox(); self.sp_scale.setRange(10, 800); self.sp_scale.setValue(int(init.get("scale", 100)))
        grid.addWidget(self.sp_scale, 4, 1)

        grid.addWidget(QLabel("Opacity (%)"), 5, 0)
        self.sp_op = QSpinBox(); self.sp_op.setRange(0, 100); self.sp_op.setValue(int(init.get("opacity", 100)))
        grid.addWidget(self.sp_op, 5, 1)

        grid.addWidget(QLabel("Rotation (°)"), 6, 0)
        self.sp_rot = QSpinBox(); self.sp_rot.setRange(-180, 180); self.sp_rot.setValue(int(init.get("rotation", 0)))
        grid.addWidget(self.sp_rot, 6, 1)

        # Auto affix
        self.cb_affix = QCheckBox("Auto-affix after placement")
        self.cb_affix.setChecked(bool(init.get("auto_affix", True)))
        grid.addWidget(self.cb_affix, 7, 0, 1, 2)

        v.addLayout(grid)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def result_dict(self) -> dict:
        return {
            "file_path": self.ed_path.text().strip(),
            "position":  self.cb_pos.currentData(),
            "margin_x":  int(self.sp_mx.value()),
            "margin_y":  int(self.sp_my.value()),
            "scale":     int(self.sp_scale.value()),
            "opacity":   int(self.sp_op.value()),
            "rotation":  int(self.sp_rot.value()),
            "auto_affix": bool(self.cb_affix.isChecked()),
        }

class _HaloBGonPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Halo-B-Gon Preset")
        v = QVBoxLayout(self)
        g = QGridLayout(); v.addLayout(g)

        g.addWidget(QLabel("Reduction:"), 0, 0)
        self.sl = QSlider(Qt.Orientation.Horizontal); self.sl.setRange(0,3); self.sl.setValue(int((initial or {}).get("reduction",0)))
        self.lab = QLabel(["Extra Low","Low","Medium","High"][self.sl.value()])
        self.sl.valueChanged.connect(lambda v: self.lab.setText(["Extra Low","Low","Medium","High"][int(v)]))
        g.addWidget(self.sl, 0, 1); g.addWidget(self.lab, 0, 2)

        self.cb = QCheckBox("Linear data"); self.cb.setChecked(bool((initial or {}).get("linear",False)))
        g.addWidget(self.cb, 1, 1)

        row = QHBoxLayout(); v.addLayout(row)
        ok = QPushButton("OK"); ok.clicked.connect(self.accept)
        ca = QPushButton("Cancel"); ca.clicked.connect(self.reject)
        row.addStretch(1); row.addWidget(ok); row.addWidget(ca)

    def result_dict(self) -> dict:
        return {"reduction": int(self.sl.value()), "linear": bool(self.cb.isChecked())}

class _RescalePresetDialog(QDialog):
    """
    Preset dialog for Geometry → Rescale.
    Stores: {"factor": float} where factor ∈ [0.10, 10.00].
    """
    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        self.setWindowTitle("Rescale Preset")
        self._initial = initial or {}

        from PyQt6.QtWidgets import QFormLayout, QDoubleSpinBox, QDialogButtonBox

        lay = QVBoxLayout(self)
        form = QFormLayout()

        self.spn_factor = QDoubleSpinBox(self)
        self.spn_factor.setDecimals(2)
        self.spn_factor.setRange(0.10, 10.00)
        self.spn_factor.setSingleStep(0.05)
        self.spn_factor.setValue(float(self._initial.get("factor", 1.0)))
        form.addRow("Scaling factor:", self.spn_factor)

        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self.resize(320, 120)

    def result_dict(self):
        return {"factor": float(self.spn_factor.value())}

class _ImageCombinePresetDialog(QDialog):
    def __init__(self, parent, initial: dict):
        super().__init__(parent); self.setWindowTitle("Image Combine Preset")
        mode = QComboBox(); mode.addItems(["Average","Add","Subtract","Blend","Multiply","Divide","Screen","Overlay","Difference"])
        mode.setCurrentText(initial.get("mode", "Blend"))
        alpha = QSlider(Qt.Orientation.Horizontal); alpha.setRange(0,100); alpha.setValue(int(100*float(initial.get("opacity",1.0))))
        luma = QCheckBox("Luminance only"); luma.setChecked(bool(initial.get("luma_only", False)))
        out_rep = QRadioButton("Replace A"); out_new = QRadioButton("Create new"); (out_new if initial.get("output")=="new" else out_rep).setChecked(True)
        from PyQt6.QtWidgets import QLineEdit
        other = QLineEdit(initial.get("docB_title","")); other.setPlaceholderText("Optional: exact title of B")

        form = QFormLayout()
        form.addRow("Mode:", mode)
        form.addRow("Opacity:", alpha)
        form.addRow("", luma)
        form.addRow("Output:", None)
        h = QHBoxLayout(); h.addWidget(out_rep); h.addWidget(out_new); h.addStretch(1)
        form.addRow("", QLabel(""))
        root = QVBoxLayout(self); root.addLayout(form); root.addLayout(h)
        form.addRow("Other source (title):", other)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        root.addWidget(btns)
        self._mode, self._alpha, self._luma, self._rep, self._other = mode, alpha, luma, out_rep, other

    def result_dict(self):
        return {
            "mode": self._mode.currentText(),
            "opacity": self._alpha.value()/100.0,
            "luma_only": self._luma.isChecked(),
            "output": "replace" if self._rep.isChecked() else "new",
            "docB_title": self._other.text().strip(),
        }

class _StarSpikesPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Diffraction Spikes Preset")
        v = QVBoxLayout(self)
        g = QGridLayout(); v.addLayout(g)
        ini = dict(initial or {})

        row = 0
        def dspin(mini, maxi, step, key, default):
            sp = QDoubleSpinBox(); sp.setRange(mini, maxi); sp.setSingleStep(step); sp.setValue(float(ini.get(key, default)))
            return sp

        def ispin(mini, maxi, step, key, default):
            sp = QSpinBox(); sp.setRange(mini, maxi); sp.setSingleStep(step); sp.setValue(int(ini.get(key, default)))
            return sp

        self.flux_min  = dspin(0.0, 999999.0, 10.0, "flux_min", 30.0);   g.addWidget(QLabel("Flux Min:"), row,0); g.addWidget(self.flux_min, row,1); row+=1
        self.flux_max  = dspin(1.0, 999999.0, 50.0, "flux_max", 300.0);  g.addWidget(QLabel("Flux Max:"), row,0); g.addWidget(self.flux_max, row,1); row+=1
        self.bmin      = dspin(0.1, 999.0,   0.5,  "bscale_min", 10.0);  g.addWidget(QLabel("Boost Min:"), row,0); g.addWidget(self.bmin, row,1); row+=1
        self.bmax      = dspin(0.1, 999.0,   0.5,  "bscale_max", 30.0);  g.addWidget(QLabel("Boost Max:"), row,0); g.addWidget(self.bmax, row,1); row+=1
        self.smin      = dspin(0.1, 999.0,   0.1,  "shrink_min", 1.0);   g.addWidget(QLabel("Shrink Min:"), row,0); g.addWidget(self.smin, row,1); row+=1
        self.smax      = dspin(0.1, 999.0,   0.1,  "shrink_max", 5.0);   g.addWidget(QLabel("Shrink Max:"), row,0); g.addWidget(self.smax, row,1); row+=1
        self.dth       = dspin(0.0, 100.0,   0.1,  "detect_thresh", 5.0);g.addWidget(QLabel("Detect Threshold:"), row,0); g.addWidget(self.dth, row,1); row+=1
        self.radius    = dspin(1.0, 512.0,   1.0,  "radius", 128.0);     g.addWidget(QLabel("Pupil Radius:"), row,0); g.addWidget(self.radius, row,1); row+=1
        self.obstr     = dspin(0.0, 0.99,    0.01, "obstruction", 0.2);  g.addWidget(QLabel("Obstruction:"), row,0); g.addWidget(self.obstr, row,1); row+=1
        self.vanes     = ispin(2,   8,       1,    "num_vanes", 4);      g.addWidget(QLabel("Num Vanes:"), row,0); g.addWidget(self.vanes, row,1); row+=1
        self.vwidth    = dspin(0.0, 50.0,    0.5,  "vane_width", 4.0);   g.addWidget(QLabel("Vane Width:"), row,0); g.addWidget(self.vwidth, row,1); row+=1
        self.rotdeg    = dspin(0.0, 360.0,   1.0,  "rotation", 0.0);     g.addWidget(QLabel("Rotation (°):"), row,0); g.addWidget(self.rotdeg, row,1); row+=1
        self.boost     = dspin(0.1, 10.0,    0.1,  "color_boost", 1.5);  g.addWidget(QLabel("Spike Boost:"), row,0); g.addWidget(self.boost, row,1); row+=1
        self.blur      = dspin(0.1, 10.0,    0.1,  "blur_sigma", 2.0);   g.addWidget(QLabel("PSF Blur Sigma:"), row,0); g.addWidget(self.blur, row,1); row+=1

        self.jwst = QCheckBox("JWST Pupil"); self.jwst.setChecked(bool(ini.get("jwst", False)))
        g.addWidget(self.jwst, row, 0, 1, 2); row += 1

        rowbox = QHBoxLayout(); v.addLayout(rowbox)
        ok = QPushButton("OK"); ca = QPushButton("Cancel")
        ok.clicked.connect(self.accept); ca.clicked.connect(self.reject)
        rowbox.addStretch(1); rowbox.addWidget(ok); rowbox.addWidget(ca)

    def result_dict(self) -> dict:
        return {
            "flux_min": float(self.flux_min.value()),
            "flux_max": float(self.flux_max.value()),
            "bscale_min": float(self.bmin.value()),
            "bscale_max": float(self.bmax.value()),
            "shrink_min": float(self.smin.value()),
            "shrink_max": float(self.smax.value()),
            "detect_thresh": float(self.dth.value()),
            "radius": float(self.radius.value()),
            "obstruction": float(self.obstr.value()),
            "num_vanes": int(self.vanes.value()),
            "vane_width": float(self.vwidth.value()),
            "rotation": float(self.rotdeg.value()),
            "color_boost": float(self.boost.value()),
            "blur_sigma": float(self.blur.value()),
            "jwst": bool(self.jwst.isChecked()),
        }
    
class _DebayerPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Debayer — Preset")
        init = dict(initial or {})
        self.combo = QComboBox(self)
        self.combo.addItems(["auto", "RGGB", "BGGR", "GRBG", "GBRG"])
        want = str(init.get("pattern", "auto")).upper()
        idx = max(0, self.combo.findText(want, Qt.MatchFlag.MatchFixedString))
        self.combo.setCurrentIndex(idx)

        lay = QVBoxLayout(self)
        row = QHBoxLayout(); row.addWidget(QLabel("Bayer pattern:")); row.addWidget(self.combo, 1)
        lay.addLayout(row)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def result_dict(self) -> dict:
        return {"pattern": self.combo.currentText().upper()}
