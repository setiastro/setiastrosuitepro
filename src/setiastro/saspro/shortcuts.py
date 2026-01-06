# pro/shortcuts.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional
import uuid

from PyQt6.QtCore import (Qt, QPoint, QRect, QMimeData, QSettings, QByteArray,
                          QDataStream, QIODevice, QEvent, QSize)
from PyQt6.QtGui import (QAction, QDrag, QIcon, QMouseEvent, QPixmap, QKeyEvent, QKeyEvent, QCursor, QKeySequence)
from PyQt6.QtWidgets import (QToolBar, QWidget, QToolButton, QMenu, QApplication, QVBoxLayout, QHBoxLayout, QComboBox, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox,
                             QInputDialog, QMessageBox, QDialog, QSlider,
    QFormLayout, QDialogButtonBox, QDoubleSpinBox, QCheckBox, QLabel, QRubberBand, QRadioButton, QPlainTextEdit, QTabWidget, QLineEdit, QPushButton, QFileDialog)

from PyQt6.QtWidgets import QMdiArea, QMdiSubWindow
# _LinearFitPresetDialog loaded on demand (see line ~334)



try:
    from PyQt6 import sip
except Exception:
    sip = None
    
from setiastro.saspro.dnd_mime import MIME_VIEWSTATE, MIME_CMD, MIME_MASK, MIME_ACTION

from pathlib import Path
import os  # ← NEW

SASS_KIND   = "sas.shortcuts"
SASS_VER    = 1

TOOLBAR_REORDER_MIME = "application/x-saspro-toolbar-reorder"
# Accept these endings (case-insensitive)
OPENABLE_ENDINGS = (
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff",
    ".fits", ".fit",
    ".fits.gz", ".fit.gz", ".fz",
    ".xisf",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef",
)

_ICONS = None

def _get_icons():
    """Lazy-load icons so shortcuts.py can be imported early without circular deps."""
    global _ICONS
    if _ICONS is not None:
        return _ICONS

    # Find where get_icons() lives in your project and import it here.
    # Try a couple common locations; keep the first that exists in your tree.
    try:
        from setiastro.saspro.resources import get_icons as _gi
    except Exception:
        _gi = None
        
    _ICONS = _gi() if _gi else None
    return _ICONS


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
KEYBINDS_KEY = "Keybinds/v1"   # JSON dict: {command_id: "Ctrl+Alt+S"}

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
        self._settings_key: str | None = None
        self.setAcceptDrops(True)

    def _mods_ok(self, mods: Qt.KeyboardModifiers) -> bool:
        return bool(mods & (
            Qt.KeyboardModifier.AltModifier |
            Qt.KeyboardModifier.ControlModifier |
            Qt.KeyboardModifier.ShiftModifier
        ))

    # NEW: called by main window / mixin
    def setSettingsKey(self, key: str):
        self._settings_key = str(key)

    def _settings(self):
        mw = self.window()
        s = getattr(mw, "settings", None)
        if s is None:
            from PyQt6.QtCore import QSettings
            s = QSettings()
        return s

    def _action_id(self, act: QAction) -> str | None:
        cid = act.property("command_id") or act.objectName()
        return str(cid) if cid else None

    def _hidden_key(self) -> str | None:
        if not self._settings_key:
            return None
        return f"{self._settings_key}/Hidden"

    def _load_hidden_set(self) -> set[str]:
        k = self._hidden_key()
        if not k:
            return set()
        s = self._settings()
        raw = s.value(k, [], type=list)
        # QSettings sometimes returns strings instead of list depending on backend
        if isinstance(raw, str):
            return {raw}
        return {str(x) for x in (raw or [])}

    def _save_hidden_set(self, hidden: set[str]):
        k = self._hidden_key()
        if not k:
            return
        s = self._settings()
        s.setValue(k, sorted(hidden))

    def _set_action_hidden(self, act: QAction, hide: bool):
        cid = self._action_id(act)
        if not cid:
            return
        hidden = self._load_hidden_set()
        if hide:
            hidden.add(cid)
            act.setVisible(False)
        else:
            hidden.discard(cid)
            act.setVisible(True)
        self._save_hidden_set(hidden)

    def apply_hidden_state(self):
        """Call after toolbar is populated / order restored."""
        hidden = self._load_hidden_set()
        if not hidden:
            return
        for act in self.actions():
            cid = self._action_id(act)
            if cid and cid in hidden:
                act.setVisible(False)


    def _persist_order(self):
        """Persist current action order (by command_id/objectName) to QSettings."""
        if not self._settings_key:
            return

        # Prefer main-window settings if available
        mw = self.window()
        s = getattr(mw, "settings", None)
        if s is None:
            from PyQt6.QtCore import QSettings
            s = QSettings()

        ids: list[str] = []
        for act in self.actions():
            cid = act.property("command_id") or act.objectName()
            if cid:
                ids.append(str(cid))

        s.setValue(self._settings_key, ids)


    def _is_locked(self) -> bool:
        """Check if toolbar icon movement is locked globally."""
        s = self._settings()
        # Default to False (unlocked)
        return s.value("UI/ToolbarLocked", False, type=bool)

    def _set_locked(self, locked: bool):
        """Set the global lock state."""
        s = self._settings()
        s.setValue("UI/ToolbarLocked", locked)

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
            # RIGHT CLICK → show "Create Desktop Shortcut"
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.RightButton:
                act = self._find_action_for_button(obj)
                if act:
                    self._show_toolbutton_context_menu(obj, act, ev.globalPosition().toPoint())
                    return True  # consume
                return False

            # Keyboard/trackpad context menu event
            if ev.type() == QEvent.Type.ContextMenu:
                act = self._find_action_for_button(obj)
                if act:
                    self._show_toolbutton_context_menu(obj, act, ev.globalPos())
                    return True
                return False            

            # L-press: remember start + whether a drag-modifier was held
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._press_pos[obj] = ev.globalPosition().toPoint()
                self._press_had_mod[obj] = self._mods_ok(QApplication.keyboardModifiers())
                return False  # allow normal press visuals

            # Move with L held:
            if ev.type() == QEvent.Type.MouseMove and (ev.buttons() & Qt.MouseButton.LeftButton):
                start = self._press_pos.get(obj)
                if start is not None:
                    delta = ev.globalPosition().toPoint() - start
                    if delta.manhattanLength() > QApplication.startDragDistance():
                        mods_now = QApplication.keyboardModifiers()
                        had_mod  = self._press_had_mod.get(obj, False)
                        
                        # CASE 1: had/has modifiers → create desktop shortcut / function-bundle drag (existing behavior)
                        if had_mod or self._mods_ok(mods_now):
                            act = self._find_action_for_button(obj)
                            if act:
                                self._start_drag_for_action(act)
                                self._suppress_release.add(obj)
                            self._press_pos.pop(obj, None)
                            self._press_had_mod.pop(obj, None)
                            return True  # consume
                        else:
                            # CASE 2: plain drag (no modifiers) → reorder within this toolbar
                            # CHECK LOCK STATE FIRST
                            if self._is_locked():
                                # Lock is active: DO NOT start drag.
                                # Should we consume the event? 
                                # If we consume it, the button won't feel "pressed" anymore if the user keeps dragging?
                                # Actually, if we just return False, standard QToolButton behavior applies (it might think it's being pressed).
                                # However, we want to prevent the *reorder* logic.
                                # So simply doing nothing here is enough to prevent the reorder drag from starting.
                                
                                # But we might want to let the user know, or just silently fail distinctively?
                                # Silently failing distinctively is what the user asked for (prevent involuntary move).
                                # If we return False, the button keeps tracking the mouse, which is fine (it won't click unless released inside).
                                return False 

                            self._start_reorder_drag_for_button(obj)
                            self._suppress_release.add(obj)
                            self._press_pos.pop(obj, None)
                            self._press_had_mod.pop(obj, None)
                            return True  # consume

                return False

            # Release: if we started any drag, swallow the release so click won't fire
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

        mods = QApplication.keyboardModifiers()
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)

        md = QMimeData()
        if alt:
            # Put BOTH payloads on the drag:
            # 1) MIME_CMD → lets you drop directly on a View or Function Bundle chip
            #    (use per-command preset if available, else empty dict)
            s = QSettings()
            raw = s.value(f"presets/{act_id}", "", type=str) or ""
            try:
                preset = json.loads(raw) if raw else {}
            except Exception:
                preset = {}
            md.setData(MIME_CMD, _pack_cmd_payload(act_id, preset))

            # 2) MIME_ACTION → canvas still interprets this to create a desktop shortcut
            md.setData(MIME_ACTION, act_id.encode("utf-8"))
        else:
            # Ctrl/Shift (legacy): only create a desktop shortcut
            md.setData(MIME_ACTION, act_id.encode("utf-8"))

        drag = QDrag(self)
        drag.setMimeData(md)
        pm = act.icon().pixmap(32, 32) if not act.icon().isNull() else QPixmap(32, 32)
        if pm.isNull():
            pm = QPixmap(32, 32); pm.fill(Qt.GlobalColor.darkGray)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)

    def _start_reorder_drag_for_button(self, btn: QToolButton):
        """
        Start a drag whose only purpose is to reorder actions within THIS toolbar.
        No presets, no desktop shortcuts.
        """
        act = self._find_action_for_button(btn)
        if act is None:
            return

        md = QMimeData()
        # Tag this as an internal toolbar-reorder drag
        md.setData(TOOLBAR_REORDER_MIME, b"1")

        drag = QDrag(btn)
        drag.setMimeData(md)

        pm = act.icon().pixmap(32, 32) if not act.icon().isNull() else QPixmap(32, 32)
        if pm.isNull():
            pm = QPixmap(32, 32)
            pm.fill(Qt.GlobalColor.darkGray)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.MoveAction)


    def _find_action_for_button(self, btn: QToolButton) -> QAction | None:
        # Find the QAction that owns this toolbutton
        for a in self.actions():
            if self.widgetForAction(a) is btn:
                return a
        return None

    def dragEnterEvent(self, e):
        # Accept toolbar-reorder drags from any DraggableToolBar
        if e.mimeData().hasFormat(TOOLBAR_REORDER_MIME):
            src = e.source()
            if isinstance(src, QToolButton) and isinstance(src.parent(), DraggableToolBar):
                e.acceptProposedAction()
                return
        super().dragEnterEvent(e)

    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(TOOLBAR_REORDER_MIME):
            src = e.source()
            if isinstance(src, QToolButton) and isinstance(src.parent(), DraggableToolBar):
                e.acceptProposedAction()
                return
        super().dragMoveEvent(e)

    def dropEvent(self, e):
        if e.mimeData().hasFormat(TOOLBAR_REORDER_MIME):
            src_btn = e.source()
            if isinstance(src_btn, QToolButton):
                src_tb = src_btn.parent()
                if isinstance(src_tb, DraggableToolBar):
                    # Find the QAction that belongs to the dragged button
                    src_act = src_tb._find_action_for_button(src_btn)
                    if src_act is None:
                        e.ignore()
                        return

                    pos = e.position().toPoint()
                    target_act = self.actionAt(pos)

                    # Remove from source toolbar and insert into this one
                    src_tb.removeAction(src_act)
                    if target_act is None:
                        self.addAction(src_act)
                    else:
                        self.insertAction(target_act, src_act)

                    # Persist order for both toolbars
                    self._persist_order()
                    if src_tb is not self:
                        src_tb._persist_order()
                        # Also persist the cross-toolbar assignment
                        self._update_assignment_for_action(src_act)

                    e.acceptProposedAction()
                    return

        super().dropEvent(e)


    def _update_assignment_for_action(self, act: QAction):
        """
        Persist that this action now belongs to this toolbar (cross-toolbar move).
        Stored as: Toolbar/Assignments → JSON {command_id: settings_key}.
        """
        if not self._settings_key:
            return

        cid = act.property("command_id") or act.objectName()
        if not cid:
            return

        mw = self.window()
        s = getattr(mw, "settings", None)
        if s is None:
            s = QSettings()

        raw = s.value("Toolbar/Assignments", "", type=str) or ""
        try:
            mapping = json.loads(raw) if raw else {}
        except Exception:
            mapping = {}

        mapping[str(cid)] = self._settings_key
        s.setValue("Toolbar/Assignments", json.dumps(mapping))


    def _add_shortcut_for_action(self, act: QAction):
        # Resolve command id
        act_id = act.property("command_id") or act.objectName()
        if not act_id:
            return
        # Find ShortcutManager on the main window
        mw = self.window()
        mgr = getattr(mw, "shortcuts", None)
        mdi = getattr(mw, "mdi", None)
        if mgr is None or mdi is None:
            return
        # Map current cursor pos (global) into the viewport
        gpos = QCursor.pos()
        vp   = mdi.viewport()
        pos  = vp.mapFromGlobal(gpos)
        # Clamp into viewport rect (center if way out of bounds)
        rect = vp.rect()
        if not rect.contains(pos):
            pos = rect.center()
        mgr.add_shortcut(str(act_id), pos)

    def _show_toolbutton_context_menu(self, btn: QToolButton, act: QAction, gpos: QPoint):
        m = QMenu(btn)

        m.addAction(self.tr("Create Desktop Shortcut"), lambda: self._add_shortcut_for_action(act))

        # Hide this icon
        cid = self._action_id(act)
        if cid:
            m.addSeparator()
            m.addAction(self.tr("Hide this icon"), lambda: self.window()._hide_action_to_hidden_toolbar(act))


        # (Optional) teach users about Alt+Drag:
        m.addSeparator()
        tip = m.addAction(self.tr("Tip: Alt+Drag to create"))
        tip.setEnabled(False)

        m.exec(gpos)


    def contextMenuEvent(self, ev):
        # Right-click on empty toolbar area
        m = QMenu(self)

        # 1. Lock/Unlock Action
        is_locked = self._is_locked()
        act_lock = m.addAction(self.tr("Lock Toolbar Icons"))
        act_lock.setCheckable(True)
        act_lock.setChecked(is_locked)
        
        def _toggle_lock(checked):
            self._set_locked(checked)
            
        act_lock.triggered.connect(_toggle_lock)
        
        m.addSeparator()

        # Submenu listing hidden actions for this toolbar
        sub = m.addMenu(self.tr("Show hidden…"))

        mw = self.window()
        tb_hidden = getattr(mw, "_hidden_toolbar", lambda: None)()
        any_hidden = False
        if tb_hidden:
            for act in tb_hidden.actions():
                # Skip separators
                if act.isSeparator():
                    continue
                any_hidden = True
                sub.addAction(act.text() or (act.property("command_id") or act.objectName() or "item"),
                              lambda a=act: mw._unhide_action_from_hidden_toolbar(a))

        if not any_hidden:
            sub.setEnabled(False)

        m.addSeparator()
        m.addAction(self.tr("Reset hidden icons"), self._reset_hidden_icons)

        m.exec(ev.globalPos())

    def _reset_hidden_icons(self):
        mw = self.window()
        tb_hidden = getattr(mw, "_hidden_toolbar", lambda: None)()
        if not tb_hidden:
            return
        # copy list because we'll mutate
        acts = [a for a in tb_hidden.actions() if not a.isSeparator()]
        for a in acts:
            mw._unhide_action_from_hidden_toolbar(a)



_PRESET_UI_IDS = {
    "stat_stretch","star_stretch","crop","curves","ghs","abe","graxpert",
    "remove_stars","cosmic_clarity","cosmic","cosmicclarity",
    "convo","convolution","deconvolution","convo_deconvo",
    "linear_fit","wavescale_hdr","wavescale_dark_enhance","wavescale_dark_enhancer",
    "remove_green","star_align","background_neutral","white_balance","clahe",
    "morphology","pixel_math","rgb_align","signature_insert","signature_adder",
    "signature","halo_b_gon","geom_rescale","rescale","debayer","image_combine",
    "star_spikes","diffraction_spikes", "multiscale_decomp","geom_rotate_any",
}

def _has_preset_editor_for_command(command_id: str) -> bool:
    """Return True if we have a bespoke UI for this command_id."""
    return command_id in _PRESET_UI_IDS

# ---- Shared preset editor helper for other modules (e.g. Function Bundles) ----
def _open_preset_editor_for_command(parent, command_id: str, initial: dict | None):
    """
    Open the same command-specific preset editor UIs used by ShortcutButton.
    Returns a dict on success (OK), or None if cancelled / no editor available.
    """
    cur = initial or {}

    # Keep each branch self-contained with local imports to avoid heavy module churn.
    if command_id == "stat_stretch":
        dlg = _StatStretchPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "star_stretch":
        dlg = _StarStretchPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "crop":
        from setiastro.saspro.shortcuts import _CropPresetDialog
        dlg = _CropPresetDialog(parent, initial=cur or {
            "mode": "margins",
            "margins": {"top": 0, "right": 0, "bottom": 0, "left": 0},
            "create_new_view": False
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "geom_rotate_any":
        dlg = _GeomRotateAnyPresetDialog(parent, initial=cur or {
            "angle_deg": 0.0,
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "curves":
        dlg = _CurvesPresetDialog(parent, initial=cur or {"shape":"linear","amount":0.5,"mode":"K (Brightness)"})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "ghs":
        dlg = _GHSPresetDialog(parent, initial=cur or {
            "alpha":1.0,"beta":1.0,"gamma":1.0,"pivot":0.5,"lp":0.0,"hp":0.0,"channel":"K (Brightness)"
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "abe":
        dlg = _ABEPresetDialog(parent, initial=cur or {
            "degree":2, "samples":120, "downsample":6, "patch":15, "rbf":True, "rbf_smooth":1.0, "make_background_doc":False
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "graxpert":
        from setiastro.saspro.graxpert_preset import GraXpertPresetDialog
        dlg = GraXpertPresetDialog(parent, initial=cur or {"smoothing":0.10,"gpu":True})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "remove_stars":
        from setiastro.saspro.remove_stars_preset import RemoveStarsPresetDialog
        dlg = RemoveStarsPresetDialog(parent, initial=cur or {"tool":"starnet","linear":True})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id in ("cosmic_clarity","cosmic","cosmicclarity"):
        from setiastro.saspro.cosmicclarity_preset import _CosmicClarityPresetDialog
        dlg = _CosmicClarityPresetDialog(parent, initial=cur or {
            "mode":"sharpen","gpu":True,"create_new_view":False,"sharpening_mode":"Both",
            "auto_psf":True,"nonstellar_psf":3.0,"stellar_amount":0.50,"nonstellar_amount":0.50,
            "denoise_luma":0.50,"denoise_color":0.50,"denoise_mode":"full","separate_channels":False,"scale":2
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id in ("convo","convolution","deconvolution","convo_deconvo"):
        from setiastro.saspro.convo_preset import ConvoPresetDialog
        dlg = ConvoPresetDialog(parent, initial=cur or {
            "op":"convolution","radius":5.0,"kurtosis":2.0,"aspect":1.0,"rotation":0.0,"strength":1.0
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "linear_fit":
        from setiastro.saspro.linear_fit import _LinearFitPresetDialog
        dlg = _LinearFitPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "wavescale_hdr":
        from setiastro.saspro.wavescale_hdr_preset import WaveScaleHDRPresetDialog
        dlg = WaveScaleHDRPresetDialog(parent, initial=cur or {"n_scales":5,"compression_factor":1.5,"mask_gamma":5.0})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id in ("wavescale_dark_enhance","wavescale_dark_enhancer"):
        from setiastro.saspro.wavescalede_preset import WaveScaleDSEPresetDialog
        dlg = WaveScaleDSEPresetDialog(parent, initial=cur or {
            "n_scales":6,"boost_factor":5.0,"mask_gamma":1.0,"iterations":2
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "multiscale_decomp":
        from setiastro.saspro.multiscale_decomp import _MultiScaleDecompPresetDialog
        dlg = _MultiScaleDecompPresetDialog(parent, initial=cur or {
            "layers": 4,
            "base_sigma": 1.0,
            "linked_rgb": True,
            "layers_cfg": [],
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "remove_green":
        dlg = _RemoveGreenPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "star_align":
        from setiastro.saspro.star_alignment_preset import StarAlignmentPresetDialog
        dlg = StarAlignmentPresetDialog(parent, initial=cur or {"ref_mode":"active","overwrite":False,"downsample":2})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "background_neutral":
        dlg = _BackgroundNeutralPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "white_balance":
        dlg = _WhiteBalancePresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "clahe":
        dlg = _CLAHEPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "morphology":
        dlg = _MorphologyPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "pixel_math":
        dlg = _PixelMathPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "rgb_align":
        dlg = _RGBAlignPresetDialog(parent, initial=cur or {"model":"homography","new_doc":True})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id in ("signature_insert","signature_adder","signature"):
        dlg = _SignatureInsertPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "halo_b_gon":
        dlg = _HaloBGonPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id in ("geom_rescale","rescale"):
        dlg = _RescalePresetDialog(parent, initial=cur or {"factor":1.0})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "debayer":
        dlg = _DebayerPresetDialog(parent, initial=cur or {"pattern":"auto"})
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id == "image_combine":
        dlg = _ImageCombinePresetDialog(parent, initial=cur or {
            "mode":"Blend","opacity":1.0,"luma_only":False,"output":"replace","docB_title":""
        })
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    if command_id in ("star_spikes","diffraction_spikes"):
        dlg = _StarSpikesPresetDialog(parent, initial=cur)
        return dlg.result_dict() if dlg.exec() == QDialog.DialogCode.Accepted else None

    # Unknown / no bespoke UI
    return None


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
        m.addAction(self.tr("Run"), lambda: self._mgr.trigger(self.command_id))
        m.addSeparator()
        m.addAction(self.tr("Edit Preset…"), self._edit_preset_ui)
        m.addAction(self.tr("Clear Preset"), lambda: self._save_preset(None))
        m.addAction(self.tr("Rename…"), self._rename)                    # ← NEW
        m.addSeparator()
        m.addAction(self.tr("Delete"), self._delete)
        m.exec(self.mapToGlobal(pos))

    def _rename(self):
        current = self.text()
        new_name, ok = QInputDialog.getText(self, self.tr("Rename Shortcut"), self.tr("Name:"), text=current)
        if not ok or not new_name.strip():
            return
        self.setText(new_name.strip())
        self._mgr.update_label(self.sid, new_name.strip())   # ← was self.shortcut_id

    def _edit_preset_ui(self):
        cid = self.command_id
        cur = self._load_preset() or {}
        result = _open_preset_editor_for_command(self, cid, cur)
        if result is not None:
            self._save_preset(result)
            QMessageBox.information(self, self.tr("Preset saved"), self.tr("Preset stored on shortcut."))
            return

        # Fallback: JSON editor
        raw = json.dumps(cur or {}, indent=2)
        text, ok = QInputDialog.getMultiLineText(self, self.tr("Edit Preset (JSON)"), self.tr("Preset:"), raw)
        if ok:
            try:
                preset = json.loads(text or "{}")
                if not isinstance(preset, dict):
                    raise ValueError(self.tr("Preset must be a JSON object"))
                self._save_preset(preset)
                QMessageBox.information(self, self.tr("Preset saved"), self.tr("Preset stored on shortcut."))
            except Exception as e:
                QMessageBox.warning(self, self.tr("Invalid JSON"), str(e))


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


def _open_view_bundles_from_canvas(w):
    try:
        from setiastro.saspro.view_bundle import show_view_bundles
        mw = _find_main_window(w)
        show_view_bundles(mw)
    except Exception:
        pass

def _open_function_bundles_from_canvas(w):
    try:
        from setiastro.saspro.function_bundle import show_function_bundles
        mw = _find_main_window(w)
        show_function_bundles(mw)
    except Exception:
        pass

def _find_main_window(w):
    p = w.parent()
    while p is not None and not (hasattr(p, "doc_manager") or hasattr(p, "docman")):
        p = p.parent()
    return p

# ---------- overlay canvas that sits on top of QMdiArea.viewport() ----------
class ShortcutCanvas(QWidget):
    def __init__(self, mgr: "ShortcutManager", parent: QWidget):
        super().__init__(parent)
        self._mgr = mgr
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")        
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
        if md.hasFormat(MIME_ACTION) or md.hasFormat(MIME_CMD) or self._md_has_openable_urls(md):
            self.raise_()
            e.acceptProposedAction()
        else:
            e.ignore()

    def _top_subwindow_at(self, vp_pos: QPoint) -> QMdiSubWindow | None:
        # Use the correct enum for PyQt6, fall back gracefully if unavailable
        try:
            order_enum = QMdiArea.WindowOrder  # PyQt6
            swlist = self._mgr.mdi.subWindowList(order_enum.StackingOrder)
        except Exception:
            # Fallback for older bindings
            swlist = self._mgr.mdi.subWindowList()

        # Iterate from front-most to back-most (StackingOrder is typically back->front)
        for sw in reversed(swlist):
            if not sw.isVisible():
                continue
            # QMdiSubWindow geometry is in the viewport's coordinate space
            if sw.geometry().contains(vp_pos):
                return sw
        return None

    def _forward_command_drop(self, e) -> bool:
        from PyQt6.QtWidgets import QApplication
        md = e.mimeData()
        if not md.hasFormat(MIME_CMD):
            return False
        sw = self._top_subwindow_at(e.position().toPoint())
        if sw is None:

            return False
        try:
            raw = bytes(md.data(MIME_CMD))
            payload = _unpack_cmd_payload(raw)  # your existing helper

        except Exception as ex:

            return False
        self._mgr.apply_command_to_subwindow(sw, payload)
        e.acceptProposedAction()
        return True


    def dragMoveEvent(self, e):
        if e.mimeData().hasFormat(MIME_ACTION) or e.mimeData().hasFormat(MIME_CMD) or self._md_has_openable_urls(e.mimeData()):
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
            self.lower()
            return

        # 2) command-only drops (no MIME_ACTION) → create a shortcut with preset
        #    This is used by History Explorer Alt+drag.
        if md.hasFormat(MIME_CMD) and not md.hasFormat(MIME_ACTION):
            try:
                raw = bytes(md.data(MIME_CMD))
                payload = _unpack_cmd_payload(raw)
            except Exception:
                payload = None

            if isinstance(payload, dict) and payload.get("command_id"):
                self._mgr.add_shortcut_from_payload(payload, e.position().toPoint())
                e.acceptProposedAction()
                self.lower()
                return

        # 3) desktop shortcut creation (MIME_ACTION) → create a button (no preset)
        if md.hasFormat(MIME_ACTION):
            act_id = bytes(md.data(MIME_ACTION)).decode("utf-8")
            self._mgr.add_shortcut(act_id, e.position().toPoint())
            e.acceptProposedAction()
            self.lower()
            return

        # 4) File / folder open (unchanged)
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
            self.lower() 
            return
        self.lower()
        e.ignore()  

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        has_sel = bool(self._mgr.selected)
        a_del = menu.addAction(self.tr("Delete Selected"), self._mgr.delete_selected); a_del.setEnabled(has_sel)
        a_clr = menu.addAction(self.tr("Clear Selection"), self._mgr.clear_selection); a_clr.setEnabled(has_sel)
        menu.addSeparator()
        a_vb = menu.addAction(self.tr("View Bundles…"), lambda: _open_view_bundles_from_canvas(self))
        a_fb = menu.addAction(self.tr("Function Bundles…"), lambda: _open_function_bundles_from_canvas(self))
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

        # Ensure action shortcuts work even if focus is in child widgets / MDI
        action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)

        self.registry[command_id] = action

        # Apply saved keybind if present
        kb = self._load_keybinds().get(command_id)
        if kb:
            action.setShortcut(QKeySequence(kb))

    def find_keybind_conflicts(self) -> dict[str, list[str]]:
        # returns {"Ctrl+Alt+K": ["script:...", "stat_stretch"], ...}
        conflicts = {}
        for cid, act in self.registry.items():
            ks = act.shortcut().toString()
            if not ks:
                continue
            conflicts.setdefault(ks, []).append(cid)
        return {k:v for k,v in conflicts.items() if len(v) > 1}

    def trigger(self, command_id: str):
        act = self.registry.get(command_id)
        if act:
            act.trigger()

    def _on_widget_destroyed(self, sid: str):
        # Called from QObject.destroyed — never touch the widget, just clean maps
        self.widgets.pop(sid, None)
        self.selected.discard(sid)

    def _collect_live_items(self) -> list[dict]:
        """
        Collect visible shortcut widgets into a serializable list.

        For each shortcut we also try to inline its preset (if any) so that
        function bundles and other per-instance presets can be exported.
        """
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
                item = {
                    "id": sid,
                    "command_id": getattr(w, "command_id", None),
                    "label": w.text(),
                    "x": int(p.x()),
                    "y": int(p.y()),
                }

                # Try to inline per-instance preset
                preset = None
                try:
                    if hasattr(w, "_load_preset"):
                        preset = w._load_preset()
                except Exception:
                    preset = None

                if isinstance(preset, dict) and preset:
                    item["preset"] = preset

                data.append(item)
            except RuntimeError:
                self.widgets.pop(sid, None)
                self.selected.discard(sid)

        # Debug: summarize what we collected
        try:
            summary = []
            for it in data:
                cid = it.get("command_id")
                has_preset = "preset" in it
                summary.append(f"{cid!r} preset={has_preset}")
            self._debug(f"_collect_live_items: {len(data)} item(s): " + ", ".join(summary))
        except Exception:
            pass

        return data

    def _export_function_bundles_for_shortcuts(self) -> dict | None:
        """
        Ask pro.function_bundle for function bundle defs + chip layout
        so we can embed them into the .sass export.
        """
        try:
            from setiastro.saspro.function_bundle import export_function_bundles_payload
        except Exception:
            return None
        try:
            fb = export_function_bundles_payload()
            if isinstance(fb, dict):
                return fb
        except Exception:
            pass
        return None

    def _import_function_bundles_for_shortcuts(self, payload: dict | None, replace_existing: bool):
        """
        Restore function bundle defs + chips after a .sass import.
        """
        if not isinstance(payload, dict):
            return
        try:
            from setiastro.saspro.function_bundle import import_function_bundles_payload
        except Exception:
            return
        try:
            mw = getattr(self, "mw", None)
            import_function_bundles_payload(payload, mw, replace_existing=replace_existing)
        except Exception:
            pass


    def _debug(self, msg: str):
        """Best-effort debug logging for shortcuts."""
        try:
            # Prefer main window log if available
            if hasattr(self.mw, "_log") and callable(self.mw._log):
                self.mw._log(f"[Shortcuts] {msg}")
                return
        except Exception:
            pass
        # Fallback to stdout
        try:
            print(f"[Shortcuts] {msg}")
        except Exception:
            pass


    # ---------- New: export/import ----------
    def export_to_file(self, file_path: str) -> tuple[bool, str]:
        try:
            fp = self._ensure_ext(file_path, ".sass")

            items = self._collect_live_items()
            payload = {
                "kind": SASS_KIND,
                "version": SASS_VER,
                "exported_at": int(time.time()),
                "items": items,
            }

            # NEW: include function bundles + chip layout if available
            fb_payload = self._export_function_bundles_for_shortcuts()
            if fb_payload is not None:
                payload["function_bundles"] = fb_payload

            Path(fp).write_text(json.dumps(payload, indent=2), encoding="utf-8")

            # optional debug
            try:
                self._debug(f"export_to_file → {fp}, shortcuts={len(items)}, "
                            f"fb_bundles={len((fb_payload or {}).get('bundles', []))}")
            except Exception:
                pass

            return True, fp
        except Exception as e:
            try:
                self._debug(f"export_to_file FAILED: {e}")
            except Exception:
                pass
            return False, str(e)

    def import_from_file(self, file_path: str, *, replace_existing: bool = False) -> tuple[bool, str]:
        try:
            txt = Path(file_path).read_text(encoding="utf-8")
            obj = json.loads(txt)

            fb_payload = None

            # Basic validation (accepts legacy raw arrays too)
            if isinstance(obj, dict) and obj.get("kind") == SASS_KIND:
                items = obj.get("items", [])
                fb_payload = obj.get("function_bundles")
            elif isinstance(obj, list):
                # legacy: straight array of items (no function bundle info)
                items = obj
            else:
                return False, "File is not a SAS shortcuts file."

            # optional debug
            try:
                self._debug(
                    f"import_from_file ← {file_path}, items={len(items)}, "
                    f"has_fb={isinstance(fb_payload, dict)}, replace_existing={replace_existing}"
                )
            except Exception:
                pass

            if replace_existing:
                self.clear()  # clears both UI + settings keys, keeps manager ready

            # Build widgets (shortcuts) as before
            for it in items:
                cid = it.get("command_id")
                if not cid:
                    continue
                sid    = it.get("id") or uuid.uuid4().hex
                x      = int(it.get("x", 10))
                y      = int(it.get("y", 10))
                label  = it.get("label") or self._default_label_for(cid)

                self.add_shortcut(cid, QPoint(x, y), label=label, shortcut_id=sid)

                # If you also inline per-instance presets for normal shortcuts,
                # you can restore them here as well (omitted here for brevity).

            # Persist shortcuts to QSettings
            self.save_shortcuts()

            # NEW: Restore function bundle definitions + chips
            self._import_function_bundles_for_shortcuts(fb_payload, replace_existing=replace_existing)

            return True, "OK"
        except Exception as e:
            try:
                self._debug(f"import_from_file FAILED: {e}")
            except Exception:
                pass
            return False, str(e)

    def _icon_for_command(self, command_id: str, act: QAction | None) -> QIcon:
        # 1) Prefer the QAction icon (works for built-in tools and scripts if set)
        if act is not None:
            try:
                ico = act.icon()
                if ico is not None and not ico.isNull():
                    return ico
            except Exception:
                pass

            # 2) Optional: if the action carries a script_icon_path property, use it
            try:
                p = act.property("script_icon_path")
                if isinstance(p, str) and p.strip() and Path(p).exists():
                    return QIcon(p.strip())
            except Exception:
                pass

        # 3) Fallback for scripts: use the generic SCRIPT icon
        if isinstance(command_id, str) and command_id.startswith("script:"):
            try:
                ic = _get_icons()
                if ic is not None and hasattr(ic, "SCRIPT"):
                    v = ic.SCRIPT
                    return v if isinstance(v, QIcon) else QIcon(str(v))
            except Exception:
                pass

        return QIcon()


    # ---------- utils ----------
    def _ensure_ext(self, path: str, ext: str) -> str:
        p = Path(path)
        if p.suffix.lower() != ext.lower():
            p = p.with_suffix(ext)
        return str(p)

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

        ico = self._icon_for_command(command_id, act)
        w = ShortcutButton(self, sid, command_id, ico, lbl, self.canvas)
        w.adjustSize()
        w.move(pos)
        w.show()

        # when the C++ object dies, clean maps using the SID
        w.destroyed.connect(lambda _=None, sid=sid: self._on_widget_destroyed(sid))

        self.widgets[sid] = w
        self.save_shortcuts()

    def add_shortcut_from_payload(self, payload: dict, pos: QPoint):
        """
        Create a desktop shortcut from a full command payload
        (e.g. drag from History Explorer with command_id + preset).
        """
        if not isinstance(payload, dict):
            return

        cid = payload.get("command_id") or payload.get("cid")
        if not isinstance(cid, str) or not cid:
            return

        preset = payload.get("preset") or {}
        if not isinstance(preset, dict):
            try:
                preset = dict(preset)
            except Exception:
                preset = {}

        # Normal shortcut creation
        sid = uuid.uuid4().hex
        label = self._default_label_for(cid)
        self.add_shortcut(cid, pos, label=label, shortcut_id=sid)

        # Attach preset at instance-level (same mechanism as context menu)
        w = self.widgets.get(sid)
        if w and not _is_dead(w):
            try:
                w._save_preset(preset)
            except Exception:
                pass

        # Persist layout + presets
        self.save_shortcuts()


    def update_label(self, shortcut_id: str, new_label: str):
        w = self.widgets.get(shortcut_id)
        if w and not _is_dead(w):
            w.setText(new_label.strip())  # in case caller didn't already
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


    def _load_keybinds(self) -> dict:
        raw = self.settings.value(KEYBINDS_KEY, "", type=str) or ""
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _save_keybinds(self, d: dict):
        self.settings.setValue(KEYBINDS_KEY, json.dumps(d))
        self.settings.sync()

    def set_keybind(self, command_id: str, keyseq: str | None):
        """
        keyseq: e.g. "Ctrl+Alt+K". If None/empty => clear binding.
        Applies immediately if action is registered.
        """
        d = self._load_keybinds()
        if keyseq and keyseq.strip():
            d[command_id] = keyseq.strip()
        else:
            d.pop(command_id, None)
        self._save_keybinds(d)

        act = self.registry.get(command_id)
        if act is not None:
            if keyseq and keyseq.strip():
                act.setShortcut(QKeySequence(keyseq.strip()))
                act.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            else:
                act.setShortcut(QKeySequence())

    def apply_command_to_subwindow(self, subwin, payload):
        """Apply a dragged command (or bundle) to the specific subwindow."""
        from PyQt6.QtWidgets import QApplication

        # --- normalize payload to a dict ---
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = json.loads(payload.decode("utf-8"))
            except Exception:
                print("[Shortcuts] apply_command_to_subwindow: invalid bytes payload", flush=True)
                QApplication.processEvents()
                return
        if not isinstance(payload, dict):
            print(f"[Shortcuts] apply_command_to_subwindow: non-dict payload {type(payload)}", flush=True)
            QApplication.processEvents()
            return

        print(f"[Shortcuts] apply_command_to_subwindow: subwin={subwin}, payload={payload!r}", flush=True)
        QApplication.processEvents()

        # --- flatten accidental nesting:
        cid = payload.get("command_id")
        if isinstance(cid, dict):
            payload = cid
            cid = payload.get("command_id")

        if not isinstance(cid, str) and isinstance(payload.get("command_id"), dict):
            payload = payload["command_id"]
            cid = payload.get("command_id")

        if not isinstance(cid, str) or not cid:
            print("[Shortcuts] apply_command_to_subwindow: no valid command_id", flush=True)
            QApplication.processEvents()
            return

        # --- function bundle handling ---
        if cid == "function_bundle":
            steps = payload.get("steps")
            inherit_target = bool(payload.get("inherit_target", True))
            print(f"[Shortcuts] function_bundle: {len(steps or [])} step(s), inherit_target={inherit_target}", flush=True)
            QApplication.processEvents()

            # If explicit steps are present (chip DnD / history replay payload),
            # run them INLINE via the normal command path (same as FunctionBundleDialog).
            if isinstance(steps, list) and steps:
                for i, st in enumerate(steps, start=1):
                    try:
                        scid = st.get("command_id")
                    except Exception:
                        scid = None
                    print(f"[Shortcuts]   inline step {i}/{len(steps)} → {scid!r}", flush=True)
                    QApplication.processEvents()
                    # Reuse the same target subwindow for each step
                    self.apply_command_to_subwindow(subwin, st)
                return

            # No inline steps → this is a true 'function_bundle' command (e.g. from Scripts),
            # so delegate to the central executor.
            try:
                from setiastro.saspro.function_bundle import run_function_bundle_command
                print("[Shortcuts] function_bundle: using run_function_bundle_command (no inline steps)", flush=True)
                QApplication.processEvents()

                try:
                    self.mdi.setActiveSubWindow(subwin)
                except Exception:
                    pass

                # Pass the whole payload as cfg so things like bundle_key, etc., are available.
                cfg = dict(payload)
                try:
                    run_function_bundle_command(self.mw, preset=payload.get("preset") or None, cfg=cfg)
                except TypeError:
                    # older signature: (ctx, cfg)
                    run_function_bundle_command(self.mw, cfg)
                print("[Shortcuts] function_bundle: run_function_bundle_command finished", flush=True)
                QApplication.processEvents()
                return
            except Exception as ex:
                print(f"[Shortcuts] function_bundle: FAILED in central executor: {ex!r}", flush=True)
                QApplication.processEvents()
                return

        # --- primary path (unchanged) ---
        mw = self.mw
        try:
            if hasattr(mw, "_handle_command_drop"):
                print(f"[Shortcuts] forwarding cid={cid!r} to _handle_command_drop", flush=True)
                QApplication.processEvents()
                mw._handle_command_drop(payload, target_sw=subwin)
                return
        except Exception as ex:
            print(f"[Shortcuts] _handle_command_drop raised: {ex!r}, falling through", flush=True)
            QApplication.processEvents()

        # --- secondary paths (unchanged) ---
        w = getattr(subwin, "widget", None)
        target = w() if callable(w) else w
        preset = payload.get("preset") or {}

        if hasattr(target, "apply_command"):
            print(f"[Shortcuts] target.apply_command for cid={cid!r}", flush=True)
            QApplication.processEvents()
            target.apply_command(cid, preset)
            return
        if hasattr(mw, "apply_command_to_view"):
            print(f"[Shortcuts] mw.apply_command_to_view for cid={cid!r}", flush=True)
            QApplication.processEvents()
            mw.apply_command_to_view(target, cid, preset)
            return
        if hasattr(mw, "run_command"):
            print(f"[Shortcuts] mw.run_command for cid={cid!r}", flush=True)
            QApplication.processEvents()
            mw.run_command(cid, preset, view=target)
            return

        print(f"[Shortcuts] fallback QAction trigger for cid={cid!r}", flush=True)
        QApplication.processEvents()
        self.mdi.setActiveSubWindow(subwin)
        act = self.registry.get(cid if isinstance(cid, str) else str(cid))
        if act:
            act.trigger()


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

    def clear_selection(self):
        """Clear current selection highlight without deleting shortcuts."""
        # Remove highlight from all currently selected items
        for sid in list(self.selected):
            try:
                self._apply_sel_visual(sid, False)
            except Exception:
                pass
        self.selected.clear()

        # Nudge repaint (optional but helps)
        try:
            self.canvas.update()
        except Exception:
            pass


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
    """
    Preset editor for headless replay: command_id="stat_stretch"

    Keys supported:
      target_median: float
      linked: bool
      normalize: bool
      apply_curves: bool
      curves_boost: float            # 0..1
      blackpoint_sigma: float        # 0..1 (matches your slider/100)
      no_black_clip: bool
      hdr_compress: bool
      hdr_amount: float              # 0..1
      hdr_knee: float                # 0..1
      luma_only: bool
      luma_mode: str                 # e.g. "rec709"
    """
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Stretch — Preset")
        init = dict(initial or {})

        # --- Target median ---
        self.spin_target = QDoubleSpinBox()
        self.spin_target.setRange(0.0, 1.0)
        self.spin_target.setDecimals(3)
        self.spin_target.setSingleStep(0.01)
        self.spin_target.setValue(float(init.get("target_median", 0.25)))

        # --- Linked / Normalize ---
        self.chk_linked = QCheckBox("Linked RGB channels")
        self.chk_linked.setChecked(bool(init.get("linked", False)))

        self.chk_normalize = QCheckBox("Normalize to [0..1]")
        self.chk_normalize.setChecked(bool(init.get("normalize", False)))

        # --- Curves ---
        self.chk_curves = QCheckBox("Apply Curves")
        self.chk_curves.setChecked(bool(init.get("apply_curves", False)))

        self.spin_curves = QDoubleSpinBox()
        self.spin_curves.setRange(0.0, 1.0)
        self.spin_curves.setDecimals(2)
        self.spin_curves.setSingleStep(0.05)
        self.spin_curves.setValue(float(init.get("curves_boost", 0.0)))
        self.spin_curves.setEnabled(self.chk_curves.isChecked())
        self.chk_curves.toggled.connect(self.spin_curves.setEnabled)

        # --- Blackpoint sigma ---
        self.spin_bp = QDoubleSpinBox()
        self.spin_bp.setRange(0.0, 1.0)
        self.spin_bp.setDecimals(2)
        self.spin_bp.setSingleStep(0.05)
        self.spin_bp.setValue(float(init.get("blackpoint_sigma", 0.0)))

        self.chk_no_black_clip = QCheckBox("No black clip")
        self.chk_no_black_clip.setChecked(bool(init.get("no_black_clip", False)))

        # --- HDR compress ---
        self.chk_hdr = QCheckBox("HDR compression")
        self.chk_hdr.setChecked(bool(init.get("hdr_compress", False)))

        self.spin_hdr_amt = QDoubleSpinBox()
        self.spin_hdr_amt.setRange(0.0, 1.0)
        self.spin_hdr_amt.setDecimals(2)
        self.spin_hdr_amt.setSingleStep(0.05)
        self.spin_hdr_amt.setValue(float(init.get("hdr_amount", 0.0)))

        self.spin_hdr_knee = QDoubleSpinBox()
        self.spin_hdr_knee.setRange(0.0, 1.0)
        self.spin_hdr_knee.setDecimals(2)
        self.spin_hdr_knee.setSingleStep(0.05)
        self.spin_hdr_knee.setValue(float(init.get("hdr_knee", 0.5)))

        def _set_hdr_enabled(on: bool):
            on = bool(on)
            self.spin_hdr_amt.setEnabled(on)
            self.spin_hdr_knee.setEnabled(on)

        _set_hdr_enabled(self.chk_hdr.isChecked())
        self.chk_hdr.toggled.connect(_set_hdr_enabled)

        # --- Luma only ---
        self.chk_luma_only = QCheckBox("Luma-only mode")
        self.chk_luma_only.setChecked(bool(init.get("luma_only", False)))

        self.cmb_luma = QComboBox()
        # keep in sync with your tool’s supported modes
        self.cmb_luma.addItems(["rec709", "avg", "hsp", "max"])
        init_mode = str(init.get("luma_mode", "rec709") or "rec709")
        idx = self.cmb_luma.findText(init_mode)
        if idx >= 0:
            self.cmb_luma.setCurrentIndex(idx)

        self.cmb_luma.setEnabled(self.chk_luma_only.isChecked())
        self.chk_luma_only.toggled.connect(self.cmb_luma.setEnabled)

        # --- Layout ---
        form = QFormLayout()

        form.addRow("Target median:", self.spin_target)
        form.addRow("", self.chk_linked)
        form.addRow("", self.chk_normalize)

        form.addRow("", QLabel("— Tone shaping —"))
        form.addRow("", self.chk_curves)
        form.addRow("Curves boost (0–1):", self.spin_curves)

        form.addRow("", QLabel("— Blackpoint / HDR —"))
        form.addRow("Blackpoint σ (0–1):", self.spin_bp)
        form.addRow("", self.chk_no_black_clip)

        form.addRow("", self.chk_hdr)
        form.addRow("HDR amount (0–1):", self.spin_hdr_amt)
        form.addRow("HDR knee (0–1):", self.spin_hdr_knee)

        form.addRow("", QLabel("— Luma mode —"))
        form.addRow("", self.chk_luma_only)
        form.addRow("Luma mode:", self.cmb_luma)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        self.setLayout(form)

    def result_dict(self) -> dict:
        hdr_on = bool(self.chk_hdr.isChecked())
        curves_on = bool(self.chk_curves.isChecked())
        luma_on = bool(self.chk_luma_only.isChecked())

        return {
            "target_median": float(self.spin_target.value()),
            "linked": bool(self.chk_linked.isChecked()),
            "normalize": bool(self.chk_normalize.isChecked()),

            "apply_curves": curves_on,
            "curves_boost": float(self.spin_curves.value()) if curves_on else 0.0,

            "blackpoint_sigma": float(self.spin_bp.value()),
            "no_black_clip": bool(self.chk_no_black_clip.isChecked()),

            "hdr_compress": hdr_on,
            "hdr_amount": float(self.spin_hdr_amt.value()) if hdr_on else 0.0,
            "hdr_knee": float(self.spin_hdr_knee.value()) if hdr_on else 0.0,

            "luma_only": luma_on,
            "luma_mode": str(self.cmb_luma.currentText()) if luma_on else "rec709",
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

        # Local labels so there’s no external dependency.
        MODE_LABELS = {
            "avg": "Average neutral (G → min(avg(R,B), G))",
            "max": "Average neutral MAX (G → min(max(R,B), G))",
            "min": "Average neutral MIN (G → min(min(R,B), G))",
        }
        MODE_INDEX = {"avg": 0, "max": 1, "min": 2}

        # Amount
        self.spin_amount = QDoubleSpinBox()
        self.spin_amount.setRange(0.0, 1.0)
        self.spin_amount.setDecimals(2)
        self.spin_amount.setSingleStep(0.05)
        self.spin_amount.setValue(float(init.get("amount", 1.00)))  # default full SCNR

        # Mode
        self.combo_mode = QComboBox()
        self.combo_mode.addItem(MODE_LABELS["avg"], userData="avg")
        self.combo_mode.addItem(MODE_LABELS["max"], userData="max")
        self.combo_mode.addItem(MODE_LABELS["min"], userData="min")
        init_mode = str(init.get("mode", init.get("neutral_mode", "avg"))).lower()
        self.combo_mode.setCurrentIndex(MODE_INDEX.get(init_mode, 0))

        # Preserve lightness
        self.cb_preserve = QCheckBox("Preserve lightness")
        self.cb_preserve.setChecked(bool(init.get("preserve_lightness", init.get("preserve", True))))

        # Layout
        form = QFormLayout(self)
        form.addRow("Amount (0–1):", self.spin_amount)
        form.addRow("Neutral mode:", self.combo_mode)
        form.addRow("", self.cb_preserve)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "amount": float(self.spin_amount.value()),                 # 0..1
            "mode": self.combo_mode.currentData() or "avg",            # "avg" | "max" | "min"
            "preserve_lightness": bool(self.cb_preserve.isChecked()),  # True/False
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
      • clip_limit (0.10–4.00)
      • tile_px (8–512 px)  → converted to OpenCV tileGridSize based on image size
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

        self.sp_tile_px = QSpinBox()
        self.sp_tile_px.setRange(8, 512)
        self.sp_tile_px.setSingleStep(8)

        # Support both old and new in the UI:
        if "tile_px" in init:
            self.sp_tile_px.setValue(int(init.get("tile_px", 128)))
        else:
            # legacy tile count → rough px guess; keeps old presets "reasonable"
            legacy_tile = int(init.get("tile", 8))
            legacy_tile = max(2, min(legacy_tile, 128))
            # Heuristic: convert tile count to a "typical" px size assuming ~2048 min dim
            px_guess = int(round(2048 / float(legacy_tile)))
            px_guess = max(8, min(px_guess, 512))
            # snap to step 8
            px_guess = int(round(px_guess / 8)) * 8
            self.sp_tile_px.setValue(px_guess)

        form.addRow("Clip limit:", self.dp_clip)
        form.addRow("Tile size (px):", self.sp_tile_px)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "clip_limit": float(self.dp_clip.value()),
            "tile_px": int(self.sp_tile_px.value()),
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

from setiastro.saspro.curves_preset import list_custom_presets, _norm_mode

class _CurvesPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Curves — Preset")
        init = dict(initial or {})

        # --- Mode ---------------------------------------------------------
        self.mode = QComboBox()
        self.mode.addItems(["K (Brightness)", "R", "G", "B", "L*", "a*", "b*", "Chroma", "Saturation"])
        want = (init.get("mode") or "K (Brightness)").strip()
        self.mode.setCurrentIndex(max(0, self.mode.findText(want)))

        # --- Shape --------------------------------------------------------
        self.shape = QComboBox()
        self.shape.addItem("Linear", "linear")
        self.shape.addItem("S-curve (mild)", "s_mild")
        self.shape.addItem("S-curve (medium)", "s_med")
        self.shape.addItem("S-curve (strong)", "s_strong")
        self.shape.addItem("Lift shadows", "lift_shadows")
        self.shape.addItem("Crush shadows", "crush_shadows")
        self.shape.addItem("Fade blacks", "fade_blacks")
        self.shape.addItem("Highlight roll-off", "rolloff_highlights")
        self.shape.addItem("Flatten contrast", "flatten")
        self.shape.addItem("Custom points", "custom")
        self.shape.setCurrentIndex(max(0, self.shape.findData((init.get("shape") or "linear").lower())))

        # --- Amount (ignored if custom) -----------------------------------
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0.0, 1.0); self.amount.setDecimals(2)
        self.amount.setSingleStep(0.05)
        self.amount.setValue(float(init.get("amount", 0.50)))

        # --- Custom points (normalized "x,y; x,y; ...") -------------------
        self.points = QLineEdit()
        self.points.setPlaceholderText("points_norm: x,y; x,y; ...  (0..1)  e.g. 0,0; 0.25,0.15; 0.75,0.85; 1,1")
        if isinstance(init.get("points_norm"), (list, tuple)) and init["points_norm"]:
            s = "; ".join(f"{float(x):.6g},{float(y):.6g}" for x, y in init["points_norm"])
            self.points.setText(s)

        # ===================== Custom Presets picker ======================
        self.preset_picker = QComboBox()
        self.btn_load = QPushButton("Load custom → fields")

        # populate & enable/disable based on availability
        self._rebuild_customs()
        self.btn_load.clicked.connect(self._load_selected_preset_into_fields)

        # wrap the load-row in a QWidget so we can hide/show the whole row
        load_row = QHBoxLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.addWidget(self.btn_load)
        self._row_custom_controls = QWidget(self)
        self._row_custom_controls.setLayout(load_row)

        # layout (use explicit labels so they can be hidden with the row)
        form = QFormLayout(self)
        form.addRow(QLabel("Mode:", self), self.mode)
        form.addRow(QLabel("Shape:", self), self.shape)
        form.addRow(QLabel("Amount (0–1):", self), self.amount)
        form.addRow(QLabel("Custom points:", self), self.points)

        self._lbl_custom_picker = QLabel("Custom presets:", self)
        form.addRow(self._lbl_custom_picker, self.preset_picker)
        form.addRow(QLabel("", self), self._row_custom_controls)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

        # enable/disable + show/hide depending on shape
        def _update_enabled():
            custom = (self.shape.currentData() == "custom")
            self.points.setEnabled(custom)
            self.amount.setEnabled(custom)

            # show/hide the custom presets UI as requested
            self._set_custom_picker_visible(custom)

        self.shape.currentIndexChanged.connect(_update_enabled)
        _update_enabled()
    # ---------------------------------------------------------------------

    def _set_custom_picker_visible(self, visible: bool):
        """Show/hide the custom presets picker + load row."""
        for w in (self._lbl_custom_picker, self.preset_picker, self._row_custom_controls):
            w.setVisible(bool(visible))

    def _rebuild_customs(self):
        """Refresh the list from QSettings and (de)activate picker/load."""
        self.preset_picker.clear()
        customs = list_custom_presets()
        if not customs:
            self.preset_picker.addItem("(No custom presets saved)", userData=None)
            self.preset_picker.setEnabled(False)
            self.btn_load.setEnabled(False)
            return
        self.preset_picker.setEnabled(True)
        self.btn_load.setEnabled(True)
        for p in sorted(customs, key=lambda d: d.get("name", "").lower()):
            self.preset_picker.addItem(p.get("name", "(unnamed)"), userData=p)

    def _load_selected_preset_into_fields(self):
        p = self.preset_picker.currentData()
        if not isinstance(p, dict):
            return
        # mode
        want = _norm_mode(p.get("mode"))
        idx = self.mode.findText(want)
        if idx >= 0:
            self.mode.setCurrentIndex(idx)
        # switch to custom
        j = self.shape.findData("custom")
        if j >= 0:
            self.shape.setCurrentIndex(j)
        # points → text
        pts = p.get("points_norm") or []
        if isinstance(pts, (list, tuple)) and pts:
            s = "; ".join(f"{float(x):.6g},{float(y):.6g}" for x, y in pts)
            self.points.setText(s)

    # -------------------- parsing & result -------------------------------
    def _parse_points_text(self) -> list[tuple[float, float]]:
        txt = (self.points.text() or "").strip()
        if not txt:
            return []
        s = txt.replace("\n", ";").replace("\r", ";")
        parts = [p.strip() for p in s.split(";") if p.strip()]
        out: list[tuple[float, float]] = []
        for part in parts:
            p = part.replace(",", " ").split()
            if len(p) != 2:
                continue
            try:
                x = float(p[0]); y = float(p[1])
            except ValueError:
                continue
            out.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))

        if out:
            if all(abs(x - 0.0) > 1e-6 for x, _ in out): out.insert(0, (0.0, 0.0))
            if all(abs(x - 1.0) > 1e-6 for x, _ in out): out.append((1.0, 1.0))
            out = sorted(out, key=lambda t: t[0])
            cleaned, lastx = [], -1.0
            for x, y in out:
                if x <= lastx: x = min(1.0, lastx + 1e-4)
                cleaned.append((x, y)); lastx = x
            out = cleaned
        return out

    def result_dict(self) -> dict:
        mode  = _norm_mode(self.mode.currentText())
        shape = self.shape.currentData() or "linear"
        amt   = float(self.amount.value())
        d = {"mode": mode, "shape": shape, "amount": amt}
        if shape == "custom":
            pts = self._parse_points_text()
            if pts:
                d["points_norm"] = pts
            else:
                d["shape"] = "linear"
                d.pop("points_norm", None)
        return d

class _GHSPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Universal Hyperbolic Stretch — Preset")
        init = dict(initial or {})

        self.mode = QComboBox()
        self.mode.addItems(["K (Brightness)", "R", "G", "B"])
        want = (init.get("channel") or "K (Brightness)").strip()
        i = self.mode.findText(want); self.mode.setCurrentIndex(max(0, i))

        def _mk_spin(minv, maxv, step, val, dec=2):
            s = QDoubleSpinBox(); s.setRange(minv, maxv); s.setDecimals(dec); s.setSingleStep(step); s.setValue(val); return s

        self.alpha = _mk_spin(0.02, 10.0, 0.02, float(init.get("alpha", 1.00)))
        self.beta  = _mk_spin(0.02, 10.0, 0.02, float(init.get("beta",  1.00)))
        self.gamma = _mk_spin(0.01,  5.0, 0.01, float(init.get("gamma", 1.00)))
        self.pivot = _mk_spin(0.00,  1.0, 0.01, float(init.get("pivot", 0.50)))
        self.lp    = _mk_spin(0.00,  1.0, 0.01, float(init.get("lp",    0.00)))
        self.hp    = _mk_spin(0.00,  1.0, 0.01, float(init.get("hp",    0.00)))

        form = QFormLayout(self)
        form.addRow("Channel:", self.mode)
        form.addRow("α (0.02–10):", self.alpha)
        form.addRow("β (0.02–10):", self.beta)
        form.addRow("γ (0.01–5):",  self.gamma)
        form.addRow("Pivot (0–1):", self.pivot)
        form.addRow("LP (0–1):",    self.lp)
        form.addRow("HP (0–1):",    self.hp)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "channel": self.mode.currentText(),
            "alpha": float(self.alpha.value()),
            "beta":  float(self.beta.value()),
            "gamma": float(self.gamma.value()),
            "pivot": float(self.pivot.value()),
            "lp":    float(self.lp.value()),
            "hp":    float(self.hp.value()),
        }        

class _ABEPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("ABE — Preset")
        p = dict(initial or {})
        form = QFormLayout(self)

        self.degree  = QSpinBox(); self.degree.setRange(1, 6);  self.degree.setValue(int(p.get("degree", 2)))
        self.samples = QSpinBox(); self.samples.setRange(20, 100000); self.samples.setSingleStep(20); self.samples.setValue(int(p.get("samples", 120)))
        self.down    = QSpinBox(); self.down.setRange(1, 64); self.down.setValue(int(p.get("downsample", 6)))
        self.patch   = QSpinBox(); self.patch.setRange(5, 151); self.patch.setSingleStep(2); self.patch.setValue(int(p.get("patch", 15)))
        self.rbf     = QCheckBox("Enable RBF"); self.rbf.setChecked(bool(p.get("rbf", True)))
        self.smooth  = QDoubleSpinBox(); self.smooth.setRange(0.0, 10.0); self.smooth.setDecimals(3); self.smooth.setSingleStep(0.01); self.smooth.setValue(float(p.get("rbf_smooth", 1.0)))
        self.mk_bg   = QCheckBox("Also create background document"); self.mk_bg.setChecked(bool(p.get("make_background_doc", False)))

        form.addRow("Polynomial degree:", self.degree)
        form.addRow("# samples:",         self.samples)
        form.addRow("Downsample:",        self.down)
        form.addRow("Patch size (px):",   self.patch)
        form.addRow(self.rbf)
        form.addRow("RBF smooth:",        self.smooth)
        form.addRow(self.mk_bg)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "degree": int(self.degree.value()),
            "samples": int(self.samples.value()),
            "downsample": int(self.down.value()),
            "patch": int(self.patch.value()),
            "rbf": bool(self.rbf.isChecked()),
            "rbf_smooth": float(self.smooth.value()),
            "make_background_doc": bool(self.mk_bg.isChecked()),
            # exclusion polygons: intentionally unsupported here
        }        
class _CropPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Crop Preset")
        init = dict(initial or {})
        mode = str(init.get("mode", "margins")).lower()
        margins = dict(init.get("margins", {}))

        lay = QVBoxLayout(self)
        form = QFormLayout()

        # --- Mode + help button row --------------------------------------
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["margins", "rect_norm", "quad_norm"])
        self.cmb_mode.setCurrentText(mode)
        # Per-item tooltips
        self.cmb_mode.setItemData(0, "Crop by pixel margins from each edge.", Qt.ItemDataRole.ToolTipRole)
        self.cmb_mode.setItemData(1, "Axis-aligned rectangle in 0..1 normalized coords (optional rotation).", Qt.ItemDataRole.ToolTipRole)
        self.cmb_mode.setItemData(2, "Four corners (TL,TR,BR,BL) in 0..1 normalized coords for perspective/keystone.", Qt.ItemDataRole.ToolTipRole)

        # Tiny "?" button
        self.btn_mode_help = QToolButton()
        self.btn_mode_help.setText("?")
        self.btn_mode_help.setToolTip("What do these modes mean?")
        self.btn_mode_help.setFixedWidth(24)
        self.btn_mode_help.clicked.connect(self._show_mode_help)

        # Put combo + help button on one row for the form
        mode_row = QWidget(self)
        mode_row_lay = QHBoxLayout(mode_row)
        mode_row_lay.setContentsMargins(0, 0, 0, 0)
        mode_row_lay.addWidget(self.cmb_mode, 1)
        mode_row_lay.addWidget(self.btn_mode_help, 0)
        form.addRow("Mode:", mode_row)
        # -----------------------------------------------------------------

        # Margins UI
        self.top = QSpinBox(); self.right = QSpinBox(); self.bottom = QSpinBox(); self.left = QSpinBox()
        for sb in (self.top, self.right, self.bottom, self.left):
            sb.setRange(0, 1_000_000)
        self.top.setValue(int(margins.get("top", 0)))
        self.right.setValue(int(margins.get("right", 0)))
        self.bottom.setValue(int(margins.get("bottom", 0)))
        self.left.setValue(int(margins.get("left", 0)))

        self.cb_new = QCheckBox("Create new view")
        self.cb_new.setChecked(bool(init.get("create_new_view", False)))
        self.le_title = QLineEdit(init.get("title", "Crop"))

        form.addRow("Top (px):", self.top)
        form.addRow("Right (px):", self.right)
        form.addRow("Bottom (px):", self.bottom)
        form.addRow("Left (px):", self.left)
        form.addRow("", self.cb_new)
        form.addRow("New view title:", self.le_title)
        lay.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _show_mode_help(self):
        current = self.cmb_mode.currentText()
        txt = (
            "<b>Crop modes</b><br><br>"
            "<b>margins</b> — Crop by pixel offsets from each image edge.<br>"
            "• <i>top/right/bottom/left</i> are in pixels.<br><br>"
            "<b>rect_norm</b> — Axis-aligned rectangle (optionally rotated) expressed in normalized 0..1 units.<br>"
            "• Schema: { mode:'rect_norm', rect:{ x, y, w, h, angle_deg } }<br>"
            "• x,y: top-left; w,h: size; angle_deg: CCW rotation around center (optional).<br><br>"
            "<b>quad_norm</b> — Arbitrary 4-corner crop in normalized 0..1 units (perspective/keystone).<br>"
            "• Schema: { mode:'quad_norm', quad:[[xTL,yTL],[xTR,yTR],[xBR,yBR],[xBL,yBL]] }<br>"
            "• Order: TL, TR, BR, BL. (0,0)=top-left, (1,1)=bottom-right."
        )
        # Small extra hint for the selected item
        if current == "rect_norm":
            txt += "<br><br><i>Tip:</i> Use rect_norm for regular boxes; add a small angle when needed."
        elif current == "quad_norm":
            txt += "<br><br><i>Tip:</i> Use quad_norm when the box edges aren’t parallel (keystone or tilt)."

        QMessageBox.information(self, "Crop modes help", txt)

    def result_dict(self) -> dict:
        return {
            "mode": self.cmb_mode.currentText(),
            "margins": {
                "top": int(self.top.value()),
                "right": int(self.right.value()),
                "bottom": int(self.bottom.value()),
                "left": int(self.left.value()),
            },
            "create_new_view": bool(self.cb_new.isChecked()),
            "title": self.le_title.text().strip() or "Crop",
        }

class _RGBAlignPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("RGB Align — Preset")
        init = dict(initial or {})
        v = QVBoxLayout(self)

        # ── model row ───────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Alignment model:"))
        self.cb_model = QComboBox()
        # include EDGE first
        self.cb_model.addItems(["edge", "homography", "affine", "poly3", "poly4"])
        want = init.get("model", "edge").lower()
        idx = max(0, self.cb_model.findText(want, Qt.MatchFlag.MatchFixedString))
        self.cb_model.setCurrentIndex(idx)
        row.addWidget(self.cb_model, 1)
        v.addLayout(row)

        # ── SEP sigma ───────────────────────────────────────
        sep_row = QHBoxLayout()
        sep_row.addWidget(QLabel("SEP sigma:"))
        self.sb_sigma = QSpinBox()
        self.sb_sigma.setRange(1, 10)
        self.sb_sigma.setValue(int(init.get("sep_sigma", 3)))
        self.sb_sigma.setToolTip("Detection threshold (σ) for EDGE mode.\n"
                                 "Higher = fewer stars. Only used when model = EDGE.")
        sep_row.addWidget(self.sb_sigma)
        v.addLayout(sep_row)

        # ── create new ──────────────────────────────────────
        self.chk_new = QCheckBox("Create new document")
        self.chk_new.setChecked(bool(init.get("new_doc", True)))
        v.addWidget(self.chk_new)

        # ── buttons ─────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def result_dict(self) -> dict:
        return {
            "model": self.cb_model.currentText().lower(),   # "edge" / "homography" / ...
            "sep_sigma": int(self.sb_sigma.value()),        # <-- new
            "new_doc": bool(self.chk_new.isChecked()),
        }

class _GeomRotateAnyPresetDialog(QDialog):
    def __init__(self, parent=None, initial: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Arbitrary Rotation — Preset")
        init = dict(initial or {})

        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-360.0, 360.0)
        self.spin_angle.setDecimals(2)
        self.spin_angle.setSingleStep(0.25)
        self.spin_angle.setValue(float(init.get("angle_deg", init.get("angle", 0.0))))

        form = QFormLayout(self)
        form.addRow("Angle (degrees):", self.spin_angle)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def result_dict(self) -> dict:
        return {
            "angle_deg": float(self.spin_angle.value()),
        }
