from __future__ import annotations
from PyQt6.QtCore import Qt, QSize, QPointF, QEvent, QMimeData
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel,
    QScrollArea, QWidget, QMessageBox, QSlider, QListWidgetItem, QApplication,
    QInputDialog, QAbstractItemView, QMenu
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QMouseEvent, QDrag
from PyQt6 import sip
import numpy as np
import json

from .autostretch import autostretch
from .dnd_mime import MIME_CMD
from setiastro.saspro.swap_manager import get_swap_manager

from setiastro.saspro.widgets.themed_buttons import themed_toolbtn


# ---------- helpers ----------
def _pack_cmd_payload(command_id: str, preset: dict | None = None) -> bytes:
    return json.dumps({"command_id": command_id, "preset": preset or {}}).encode("utf-8")

# Map human step names → command_id used by replay_last_action_on_base
_NAME_TO_COMMAND_ID = {
    # Background neutralization / WB
    "background neutralization": "background_neutral",
    "background neutralisation": "background_neutral",
    "background neutral": "background_neutral",
    "white balance": "white_balance",

    # Simple tools with no real preset
    "pedestal removal": "pedestal",
    "pedestal": "pedestal",
    "linear fit": "linear_fit",

    # Stretching / tone tools
    "statistical stretch": "stat_stretch",
    "stat stretch": "stat_stretch",
    "star stretch": "star_stretch",
    "curves": "curves",
    "ghs": "ghs",
    "generalized hyperbolic stretch": "ghs",
    "levels": "levels",
    "histogram transform": "levels",
    "histogram transformation": "levels",

    # Background/gradient
    "abe": "abe",
    "automatic background extraction": "abe",
    "adbe": "abe",
    "graxpert": "graxpert",

    # Star / color tools
    "remove stars": "remove_stars",
    "remove green": "remove_green",

    # Convolution / deconvolution
    "convo / deconvo": "convo",
    "convolution / deconvolution": "convo",
    "convolution": "convo",
    "deconvolution": "convo",

    # Wavescale tools
    "wavescale hdr": "wavescale_hdr",
    "wave scale hdr": "wavescale_hdr",
    "wavescale dark enhancer": "wavescale_dark_enhance",
    "wave scale dark enhancer": "wavescale_dark_enhance",
    "dark structure enhance": "wavescale_dark_enhance",

    # Other image-processing tools
    "clahe": "clahe",
    "morphology": "morphology",
    "pixel math": "pixel_math",
    "pixelmath": "pixel_math",
    "halo-b-gon": "halo_b_gon",
    "halo b gon": "halo_b_gon",
    "aberration ai": "aberrationai",
    "cosmic clarity": "cosmic_clarity",
    "rgb → mono": "rgb_to_mono",
    "rgb to mono": "rgb_to_mono",
    "mono → rgb": "mono_to_rgb",
    "mono to rgb": "mono_to_rgb",
    "swap r ↔ b": "swap_rb",
    "swap r/b": "swap_rb",    
}


def _norm_preset(p) -> dict:
    """Best effort: turn whatever we get into a dict."""
    if not p:
        return {}
    if isinstance(p, dict):
        return dict(p)
    try:
        return dict(p)
    except Exception:
        return {}


def _extract_cmd_payload_from_meta(meta: dict | None) -> dict | None:
    """
    Best-effort: pull a (command_id, preset) payload out of a history meta dict.

    We look in three places, in order:

      1) Embedded payloads (headless_payload / replay_payload / cmd_payload)
      2) Explicit command_id / cid + preset directly on metadata
      3) Inference from step_name + preset or special tool keys
    """
    if not isinstance(meta, dict):
        return None
    # --- 0) Doc-local replay marker (most reliable; authoritative) ---
    rp = meta.get("_replay")
    if isinstance(rp, dict):
        cid = _reconcile_cid(
            rp.get("command_id"),
            rp.get("step_name") or meta.get("step_name"),
            bool(rp.get("cid_explicit")),
        )
        if cid:
            return {"command_id": str(cid), "preset": _norm_preset(rp.get("preset"))}
        # Marker exists but unresolved → this step just isn't replayable.
        # Do NOT fall through to loose/leaked keys.
        return None
        # else fall through to the legacy heuristics below
    # --- 1) Embedded payload dicts ----------------------------
    for key in ("headless_payload", "replay_payload", "cmd_payload"):
        p = meta.get(key)
        if isinstance(p, dict):
            cid = p.get("command_id") or p.get("cid")
            if cid:
                return {
                    "command_id": str(cid),
                    "preset": _norm_preset(p.get("preset")),
                }

    # --- 2) Direct command_id + preset on metadata (leak-guarded) ----
    cid = meta.get("command_id") or meta.get("cid")
    if cid:
        cid_l = str(cid).strip().lower()
        name_cid = _command_id_for_step_label(
            meta.get("step_name") or meta.get("name") or ""
        )
        if name_cid == cid_l:
            # Row's own name confirms the loose id → trust it and its preset.
            preset = meta.get("preset") or meta.get("preset_dict") or {}
            return {"command_id": cid_l, "preset": _norm_preset(preset)}
        if name_cid:
            # Name disagrees → the loose id leaked from a neighbouring op
            # (self-identifying tools leave command_id/preset in metadata,
            # which the next op inherits). Trust the row's name; the loose
            # preset belonged to the other op.
            return {"command_id": str(name_cid), "preset": {}}
        # No name mapping to confirm the loose id → unreliable leftover.
        # Never trust it (that's exactly how SCNR/GHS became phantom 'curves').
        return None

    # --- 3) Heuristics for tools that only store preset + step_name ----
    preset = _norm_preset(meta.get("preset"))
    step_name_raw = meta.get("step_name") or meta.get("name")
    step_name = str(step_name_raw or "").strip().lower()

    inferred_cid = None

    if step_name:
        # Normalize a bit: underscores / hyphens → spaces
        base = step_name.replace("_", " ").replace("-", " ")

        # Exact match first
        inferred_cid = _NAME_TO_COMMAND_ID.get(base)

        # Fuzzy: allow things like "Remove Green (mask=ON)"
        if inferred_cid is None:
            for key, val in _NAME_TO_COMMAND_ID.items():
                if key in base:
                    inferred_cid = val
                    break

    # Fallback: look for tool-specific keys in metadata
    if inferred_cid is None:
        for k in meta.keys():
            k_norm = str(k).lower()

            if k_norm in ("remove_green", "remove green"):
                inferred_cid = "remove_green"
                break
            if k_norm in ("stat_stretch", "statistical_stretch"):
                inferred_cid = "stat_stretch"
                break
            if k_norm in ("ghs", "generalized hyperbolic stretch"):
                inferred_cid = "ghs"
                break
            if k_norm in ("abe", "automatic background extraction"):
                inferred_cid = "abe"
                break
            if k_norm in ("wavescale_hdr", "wave_scale_hdr"):
                inferred_cid = "wavescale_hdr"
                break
            if k_norm in ("wavescale_dark_enhance", "dark_structure_enhance"):
                inferred_cid = "wavescale_dark_enhance"
                break
            if k_norm in ("pixel_math", "pixelmath"):
                inferred_cid = "pixel_math"
                break
            if k_norm in ("halo_b_gon", "halo b gon"):
                inferred_cid = "halo_b_gon"
                break
            if k_norm in ("aberrationai", "aberration_ai"):
                inferred_cid = "aberrationai"
                break
            if k_norm in ("cosmic_clarity",):
                inferred_cid = "cosmic_clarity"
                break
            if k_norm in ("convo", "convolution", "deconvolution"):
                inferred_cid = "convo"
                break

    if inferred_cid is None:
        # Nothing we know how to replay
        return None

    return {
        "command_id": str(inferred_cid),
        "preset": preset,
    }

# Map human step names → command_id used by replay_last_action_on_base
_NAME_TO_COMMAND_ID = {
    "pedestal removal": "pedestal",
    "pedestal": "pedestal",

    "statistical stretch": "stat_stretch",
    "stat stretch": "stat_stretch",

    "curves": "curves",

    "remove green": "remove_green",

    "background neutralization": "background_neutral",
    "background neutralisation": "background_neutral",
    "background neutral": "background_neutral",
    "bn": "background_neutral",

    "white balance": "white_balance",

    "convo/deconvo": "convo",
    "convolution / deconvolution": "convo",
    "convolution": "convo",
    "deconvolution": "convo",

    "ghs": "ghs",
    "generalized hyperbolic stretch": "ghs",

    "automatic background extraction": "abe",
    "abe": "abe",

    "graxpert": "graxpert",

    "remove stars": "remove_stars",

    "star stretch": "star_stretch",

    "wavescale hdr": "wavescale_hdr",
    "wave scale hdr": "wavescale_hdr",

    "wavescale dark enhancer": "wavescale_dark_enhance",
    "dark structure enhance": "wavescale_dark_enhance",

    "clahe": "clahe",

    "morphology": "morphology",

    "pixel math": "pixel_math",
    "pixelmath": "pixel_math",

    "halo-b-gon": "halo_b_gon",
    "halo b gon": "halo_b_gon",

    "aberration ai": "aberrationai",

    "cosmic clarity": "cosmic_clarity",

    "linear fit": "linear_fit",
}


def _norm_step_label(label: str) -> str:
    """Normalize a human label like 'Statistical Stretch (target=0.25, unlinked)'."""
    s = str(label or "").strip().lower()
    if not s:
        return ""
    # Drop decorations like '(...)' or ' - extra'
    for sep in ("(", "[", " - "):
        idx = s.find(sep)
        if idx > 0:
            s = s[:idx]
    return " ".join(s.split())


_KNOWN_CIDS_CACHE = None

def _known_cids() -> set:
    """Canonical cids from the local table plus the shared alias table."""
    global _KNOWN_CIDS_CACHE
    if _KNOWN_CIDS_CACHE is None:
        s = set(_CANONICAL_CIDS)
        try:
            from setiastro.saspro.command_ids import COMMAND_ID_ALIASES
            s |= set(COMMAND_ID_ALIASES.values())
        except Exception:
            pass
        _KNOWN_CIDS_CACHE = s
    return _KNOWN_CIDS_CACHE


def _command_id_for_step_label(label: str) -> str | None:
    """Map a history step label to a canonical command_id.

    Order:
      1) local table, exact on the pre-decoration base and the full label
      2) command_ids.normalize_command_id() on base / full / any '(...)' text
         — the shared alias table is richer (e.g. 'scnr (remove green)',
         'hyperbolic stretch') and is the single source of truth
      3) fuzzy: longest known display-name that appears in the full label
    """
    raw = str(label or "").strip().lower()
    if not raw:
        return None
    full = " ".join(raw.split())
    base = _norm_step_label(label)

    # 1) local table, exact
    for cand in (base, full):
        if cand and cand in _NAME_TO_COMMAND_ID:
            return _NAME_TO_COMMAND_ID[cand]

    # 2) shared alias table — the meaningful name is often inside '(...)'
    try:
        from setiastro.saspro.command_ids import normalize_command_id
    except Exception:
        normalize_command_id = None

    if normalize_command_id is not None:
        import re as _re
        cands = [base, full]
        m = _re.search(r"\(([^)]+)\)", raw)      # e.g. 'scnr (remove green)' → 'remove green'
        if m:
            cands.append(" ".join(m.group(1).split()))
        known = _known_cids()
        for cand in cands:
            if not cand:
                continue
            norm = normalize_command_id(cand)
            if norm in known:
                return norm

    # 3) fuzzy substring, longest first; skip tiny keys to avoid false hits
    for key in sorted(_NAME_TO_COMMAND_ID, key=len, reverse=True):
        if len(key) >= 3 and key in full:
            return _NAME_TO_COMMAND_ID[key]
    return None

ROLE_IS_OP = Qt.ItemDataRole.UserRole + 1   # marks true operation rows (not "Current Image")

_CANONICAL_CIDS = set(_NAME_TO_COMMAND_ID.values())


def _pack_bundle_payload(steps) -> bytes:
    return json.dumps(
        {"command_id": "function_bundle", "steps": list(steps or [])}
    ).encode("utf-8")


def _reconcile_cid(cid_guess, step_name, explicit: bool = False):
    """Return a canonical command_id or None.

    Trust an explicit (tool-supplied) cid outright. For a name-derived guess,
    accept it only if it's a known canonical id (or a script/geom id),
    otherwise map the human step label through the display-name table.
    """
    cid_guess = str(cid_guess or "").strip().lower()
    if explicit and cid_guess:
        return cid_guess
    if cid_guess in _CANONICAL_CIDS or cid_guess.startswith(("script:", "geom_")):
        return cid_guess
    return _command_id_for_step_label(step_name or cid_guess)

def _payloads_from_headless_history(main_window, undo_entries):
    """
    Use the main window's headless history to get presets for each undo entry.

    We walk FORWARD through _headless_history so that:
      - repeated operations get the right preset in order
      - commands on other documents are skipped automatically.
    Returns a list[len(undo_entries)] of payload dicts or None.
    """
    n = len(undo_entries)
    payloads = [None] * n

    if main_window is None or not hasattr(main_window, "get_headless_history"):
        return payloads

    try:
        hist = list(main_window.get_headless_history()) or []
    except Exception:
        return payloads

    if not hist:
        return payloads

    H = len(hist)
    h_idx = 0

    for i, (_img, meta, name) in enumerate(undo_entries):
        label = name or (meta or {}).get("step_name") or ""
        cid = _command_id_for_step_label(label)
        if not cid:
            continue
        cid = cid.strip().lower()

        # Scan forward in global history until we find the next entry with this cid.
        while h_idx < H:
            entry = hist[h_idx]
            h_idx += 1
            entry_cid = str(entry.get("command_id", "")).strip().lower()
            if entry_cid != cid:
                continue

            preset = entry.get("preset") or {}
            if not isinstance(preset, dict):
                try:
                    preset = dict(preset)
                except Exception:
                    preset = {}
            payloads[i] = {"command_id": cid, "preset": preset}
            break

    return payloads

# Shared utilities
from setiastro.saspro.widgets.image_utils import to_float01 as _to_float01


def _mk_qimage_rgb8(float01: np.ndarray) -> tuple[QImage, np.ndarray]:
    """Make a QImage (RGB888) and return it along with the backing uint8 buffer to keep alive."""
    f = float01
    if f.ndim == 2:
        f = np.stack([f] * 3, axis=-1)
    elif f.ndim == 3 and f.shape[2] == 1:
        f = np.repeat(f, 3, axis=2)
    buf8 = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    buf8 = np.ascontiguousarray(buf8)
    h, w, _ = buf8.shape
    bpl = buf8.strides[0]
    ptr = sip.voidptr(buf8.ctypes.data)
    qimg = QImage(ptr, w, h, bpl, QImage.Format.Format_RGB888)
    return qimg, buf8


def _extract_undo_entries(doc):
    # Prefer the public getter we just added
    if hasattr(doc, "get_undo_stack"):
        return list(doc.get_undo_stack())

    # Fallbacks if needed
    for attr in ("_undo_stack", "undo_stack"):
        stack = getattr(doc, attr, None)
        if stack is None:
            continue
        out = []
        for item in stack:
            if isinstance(item, (list, tuple)):
                if len(item) >= 3:
                    # item[0] is now swap_id (str) or image (ndarray)
                    sid_or_img, meta, name = item[0], item[1] or {}, item[2] or "Unnamed"
                elif len(item) == 2:
                    sid_or_img, meta = item
                    meta = meta or {}
                    name = meta.get("step_name", "Unnamed")
                else:
                    continue
                out.append((sid_or_img, meta, str(name)))

        if out:
            return out
    return []


class HistoryListWidget(QListWidget):
    """
    QListWidget that drags replayable steps onto a view.

    Selection → drag behaviour (native Qt, file-manager style):
      • Ctrl/Shift-click to select multiple rows, then drag any selected row →
        a single MIME_CMD carrying a function_bundle payload (all steps, in
        row order). Dropped on a view, the whole recipe runs on that image.
      • Drag a single selected row → a MIME_CMD for that one command.
      • Press on an unselected row / empty space → normal selection (rubber
        band, Ctrl/Shift extend). No modifier key required to drag.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)

    def startDrag(self, supportedActions):
        # Qt calls this once the press-on-selection passes the drag threshold,
        # with the current selection intact — no manual tracking needed.
        steps = self.selected_replay_steps()          # ROLE_IS_OP rows, row order
        if len(steps) >= 2:
            self._start_bundle_drag(steps)
            return
        if len(steps) == 1:
            self._start_drag(steps[0])
            return
        # Nothing replayable selected (e.g. only the "Current Image" row):
        # fall back to the row under the cursor if it's single-draggable.
        it = self.currentItem()
        if it is not None:
            p = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(p, dict) and p.get("command_id"):
                self._start_drag(p)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press_pos = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._press_pos is not None and (e.buttons() & Qt.MouseButton.LeftButton):
            delta = e.position().toPoint() - self._press_pos
            if delta.manhattanLength() >= QApplication.startDragDistance():
                mods = QApplication.keyboardModifiers()
                if mods & Qt.KeyboardModifier.AltModifier:
                    sel = self.selected_replay_steps()
                    if len(sel) >= 2:
                        self._start_bundle_drag(sel)
                        self._press_pos = None
                        return
                    item = self.itemAt(self._press_pos)
                    if item is not None:
                        payload = item.data(Qt.ItemDataRole.UserRole)
                        if isinstance(payload, dict) and payload.get("command_id"):
                            self._start_drag(payload)
                            self._press_pos = None
                            return
                    item = self.itemAt(self._press_pos)
                    if item is not None:
                        payload = item.data(Qt.ItemDataRole.UserRole)
                        if isinstance(payload, dict) and payload.get("command_id"):
                            self._start_drag(payload)
                            self._press_pos = None
                            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent):
        self._press_pos = None
        super().mouseReleaseEvent(e)

    def _start_drag(self, payload: dict):
        cid = payload.get("command_id")
        preset = payload.get("preset") or {}
        if not cid:
            return

        md = QMimeData()
        md.setData(MIME_CMD, _pack_cmd_payload(cid, preset))

        drag = QDrag(self)
        drag.setMimeData(md)
        pm = QPixmap(32, 32)
        pm.fill(Qt.GlobalColor.darkGray)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)

    def _iter_selected_op_payloads(self):
        for row in range(self.count()):
            it = self.item(row)
            if not it.isSelected():
                continue
            if it.data(ROLE_IS_OP) is not True:
                continue
            p = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(p, dict) and p.get("command_id"):
                yield {"command_id": p["command_id"], "preset": dict(p.get("preset") or {})}

    def selected_replay_steps(self) -> list[dict]:
        return list(self._iter_selected_op_payloads())

    def all_replay_steps(self) -> list[dict]:
        out = []
        for row in range(self.count()):
            it = self.item(row)
            if it.data(ROLE_IS_OP) is not True:
                continue
            p = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(p, dict) and p.get("command_id"):
                out.append({"command_id": p["command_id"], "preset": dict(p.get("preset") or {})})
        return out

    def _start_bundle_drag(self, steps: list[dict]):
        md = QMimeData()
        md.setData(MIME_CMD, _pack_bundle_payload(steps))
        drag = QDrag(self)
        drag.setMimeData(md)
        pm = QPixmap(40, 40)
        pm.fill(Qt.GlobalColor.darkCyan)
        drag.setPixmap(pm)
        drag.setHotSpot(pm.rect().center())
        drag.exec(Qt.DropAction.CopyAction)

class HistoryExplorerDialog(QDialog):
    def __init__(self, document, parent=None):
        super().__init__(parent)
        self.setWindowTitle("History Explorer")
        self.setWindowFlag(Qt.WindowType.Window, True)
        import platform
        if platform.system() == "Darwin":
            self.setWindowFlag(Qt.WindowType.Tool, True)  
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.doc = document

        self.setMinimumSize(700, 500)
        layout = QVBoxLayout(self)

        self.history_list = HistoryListWidget(self)
        self.history_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )        
        layout.addWidget(self.history_list)
        self.history_instructions = QLabel(
            "Double-click any row to view the image at that step  •  "
            "Drag a step/steps onto the canvas to create a function shortcut(s)  •  "
            "Select multiple (or all) steps and use Create Function Bundle to "
            "bundle them into a replayable function  •  "
            "Drag a step/steps onto the Workflow Assistant to create a Workflow  •  "
        )
        self.history_instructions.setWordWrap(True)
        layout.addWidget(self.history_instructions)
        # ---- Fetch undo stack ----
        self.undo_entries = _extract_undo_entries(self.doc)  # list[(sid_or_img, meta, name)]
        self.items: list[tuple[object, dict, str]] = []
        # Headless command presets aligned to each undo entry
        mw = self._find_main_window()
        self._history_payloads = _payloads_from_headless_history(mw, self.undo_entries)


        # DEBUG: log what's in the undo stack
        mw = self._find_main_window()
        log = getattr(mw, "_log", None)
        if log:
            try:
                log(
                    f"[HistoryExplorer] doc id={id(self.doc)} has "
                    f"{len(self.undo_entries)} undo entries"
                )
                for idx, (img, meta, name) in enumerate(self.undo_entries):
                    mk = list((meta or {}).keys())
                    payload_meta = _extract_cmd_payload_from_meta(meta or {})
                    payload_hist = (
                        self._history_payloads[idx]
                        if 0 <= idx < len(self._history_payloads)
                        else None
                    )
                    payload = payload_hist or payload_meta
                    cid_dbg = None
                    if payload:
                        cid_dbg = payload.get("command_id") or payload.get("cid")
                    src = "hist" if payload_hist else ("meta" if payload_meta else "-")
                    log(
                        f"[HistoryExplorer] undo[{idx}] name='{name}', "
                        f"step_name='{(meta or {}).get('step_name')}', "
                        f"meta_keys={mk}, replayable={bool(payload)}, "
                        f"cid={cid_dbg}, src={src}"
                    )

                cm = getattr(self.doc, "metadata", {}) or {}
                mk = list(cm.keys())
                payload_meta = _extract_cmd_payload_from_meta(cm)
                payload_hist_last = None
                for p in reversed(self._history_payloads):
                    if p:
                        payload_hist_last = p
                        break
                payload = payload_hist_last or payload_meta
                cid_dbg = None
                if payload:
                    cid_dbg = payload.get("command_id") or payload.get("cid")
                src = "hist" if payload_hist_last else ("meta" if payload_meta else "-")
                log(
                    f"[HistoryExplorer] current image: "
                    f"meta_keys={mk}, replayable={bool(payload)}, "
                    f"cid={cid_dbg}, src={src}"
                )
            except Exception:
                pass


        # ---- Build rows ----
        # We want:
        #  1. Original Image (oldest snapshot)
        #  2. State after 1st op  → label = undo[0].name
        #  3. State after 2nd op  → label = undo[1].name
        #  ...
        #  N+1. State after Nth op (current image) → label = undo[N-1].name
        #  N+2. Current Image

        # 1) Original Image (if any undo entries exist)
        row_index = 0
        if self.undo_entries:
            orig_src, orig_meta, _ = self.undo_entries[0]
            item = QListWidgetItem("1. Original Image")
            self.history_list.addItem(item)
            self.items.append((orig_src, orig_meta, "Original Image"))
            row_index += 1

        # 2) Per-operation states
        n = len(self.undo_entries)
        for op_idx in range(n):
            op_name = self.undo_entries[op_idx][2] or f"Step {op_idx + 1}"

            if op_idx + 1 < n:
                src, meta, _ = self.undo_entries[op_idx + 1]
            else:
                # Last operation → use current image + metadata
                src = getattr(self.doc, "image", None)
                meta = getattr(self.doc, "metadata", {}) or {}

            # 1) Prefer the doc-local replay marker on THIS op's own undo tuple
            op_meta = self.undo_entries[op_idx][1] or {}
            payload = _extract_cmd_payload_from_meta(op_meta)

            # 2) Fallback: aligned headless-history preset
            if payload is None and 0 <= op_idx < len(self._history_payloads):
                payload = self._history_payloads[op_idx]

            # 3) Last resort: infer from the after-state metadata (legacy)
            if payload is None:
                payload = _extract_cmd_payload_from_meta(meta)

            is_replayable = payload is not None

            label = f"{row_index + 1}. {op_name}"
            if is_replayable:
                label += "  ⟲"

            item = QListWidgetItem(label)
            if is_replayable:
                item.setData(Qt.ItemDataRole.UserRole, payload)
                item.setData(ROLE_IS_OP, True)                
                item.setToolTip("Replayable step. Alt+Drag to drop onto a view or desktop.")
            self.history_list.addItem(item)

            self.items.append((src, meta, op_name))
            row_index += 1


        # 3) Final "Current Image" row
        cur_img = getattr(self.doc, "image", None)
        cur_meta = getattr(self.doc, "metadata", {}) or {}

        # Prefer the replay marker on the last op's undo tuple
        cur_payload = None
        if self.undo_entries:
            cur_payload = _extract_cmd_payload_from_meta(self.undo_entries[-1][1] or {})
        if cur_payload is None:
            for p in reversed(self._history_payloads):
                if p:
                    cur_payload = p
                    break
        if cur_payload is None:
            cur_payload = _extract_cmd_payload_from_meta(cur_meta)

        cur_replay = cur_payload is not None

        label = f"{row_index + 1}. Current Image"
        if cur_replay:
            label += "  ⟲"
        cur_item = QListWidgetItem(label)
        if cur_replay:
            cur_item.setData(Qt.ItemDataRole.UserRole, cur_payload)
            cur_item.setToolTip("Replayable step. Alt+Drag to drop onto a view or desktop.")
        self.history_list.addItem(cur_item)
        self.items.append((cur_img, cur_meta, "Current Image"))


        self.history_list.itemDoubleClicked.connect(self._open_preview)

        row = QHBoxLayout()
        self.btn_make_bundle = QPushButton("Create Function Bundle")
        self.btn_make_bundle.setToolTip(
            "Turn the selected replayable steps (or all of them) into a Function "
            "Bundle you can drop on any other image.\n"
            "Tip: Alt+Drag a multi-selection straight onto a view to run the recipe there."
        )
        self.btn_make_bundle.clicked.connect(self._create_bundle_from_selection)

        self.btn_open_bundles = QPushButton("Open Function Bundles…")
        self.btn_open_bundles.clicked.connect(self._open_function_bundles)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)

        row.addWidget(self.btn_make_bundle)
        row.addWidget(self.btn_open_bundles)
        row.addStretch(1)
        row.addWidget(btn_close)
        layout.addLayout(row)

    def _open_preview(self, item):
        row = self.history_list.row(item)
        if 0 <= row < len(self.items):
            src, meta, name = self.items[row]
            if src is None:
                QMessageBox.warning(self, "Preview", "No image stored for this step.")
                return
            pv = HistoryImagePreview(src, meta, self.doc, parent=self)
            pv.setWindowTitle(item.text())
            pv.show()
            mw = self._find_main_window()
            if mw and hasattr(mw, "_log"):
                mw._log(f"History: preview opened → {item.text()}")
        else:
            QMessageBox.warning(self, "Preview", "Invalid selection.")

    def _create_bundle_from_selection(self):
        steps = self.history_list.selected_replay_steps()
        used_selection = bool(steps)
        if not steps:
            steps = self.history_list.all_replay_steps()
        if not steps:
            QMessageBox.information(
                self, "Function Bundle",
                "No replayable steps were found in this history."
            )
            return

        try:
            base_name = self.doc.display_name()
        except Exception:
            base_name = "History"
        default_name = f"Recipe from {base_name}"

        name, ok = QInputDialog.getText(
            self, "Create Function Bundle",
            f"Create a bundle from {len(steps)} "
            f"{'selected ' if used_selection else ''}step(s).\n\nBundle name:",
            text=default_name,
        )
        if not ok:
            return

        try:
            from setiastro.saspro.function_bundle import add_function_bundle
            final = add_function_bundle(name or default_name, steps, parent=self)
        except Exception as e:
            QMessageBox.critical(self, "Function Bundle", f"Could not create bundle:\n{e}")
            return

        mw = self._find_main_window()
        if mw and hasattr(mw, "_log"):
            mw._log(f"History: created Function Bundle '{final}' with {len(steps)} step(s).")

        if QMessageBox.question(
            self, "Function Bundle",
            f"Created '{final}' with {len(steps)} step(s).\n\nOpen Function Bundles now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            self._open_function_bundles(focus_name=final)

    def _open_function_bundles(self, focus_name: str | None = None):
        try:
            from setiastro.saspro.function_bundle import show_function_bundles
            show_function_bundles(self._find_main_window() or self, focus_name=focus_name)
        except Exception as e:
            QMessageBox.critical(self, "Function Bundles", f"Could not open:\n{e}")

    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p


class HistoryImagePreview(QWidget):
    """
    Preview a single history entry with zoom/pan, optional display autostretch,
    compare vs current, and restore.
    """
    def __init__(self, image_source: object, metadata: dict, document, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.doc = document
        self.metadata = metadata or {}
        
        # Resolve image source (ndarray or swap_id)
        self.image_data = None
        if isinstance(image_source, str):
            # It's a swap ID
            sm = get_swap_manager()
            loaded = sm.load_state(image_source)
            if loaded is not None:
                self.image_data = loaded
            else:
                # Failed to load
                self.image_data = None
        else:
            # Assume it's an ndarray
            self.image_data = image_source
            
        if self.image_data is None:
            # Fallback placeholder?
            self.image_data = np.zeros((100, 100, 3), dtype=np.float32)

        self.zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._autostretch_on = False

        self._qimg_src = None
        self._buf8 = None

        # UI
        self.label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.scroll = QScrollArea(widgetResizable=False)
        self.scroll.setWidget(self.label)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.label.installEventFilter(self)

        # controls
        self.btn_stretch = QPushButton("Toggle AutoStretch")
        self.btn_stretch.clicked.connect(self._toggle_autostretch)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self._fit_to_view)

        self.btn_1to1 = QPushButton("1:1")
        self.btn_1to1.clicked.connect(lambda: self._set_zoom(1.0))

        self.btn_compare = QPushButton("Compare to Current…")
        self.btn_compare.clicked.connect(self._open_compare)

        self.btn_restore = QPushButton("Restore This Version")
        self.btn_restore.clicked.connect(self._restore)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 800)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(lambda v: self._set_zoom(v/100.0))

        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_stretch)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_fit)
        ctrl.addWidget(self.btn_1to1)
        ctrl.addWidget(self.slider)
        ctrl.addWidget(self.btn_compare)
        ctrl.addWidget(self.btn_restore)

        lay = QVBoxLayout(self)
        lay.addWidget(self.scroll, 1)
        lay.addLayout(ctrl)

        self._rebuild_source()
        self._fit_to_view()

    # data → qimage
    def _make_vis(self) -> np.ndarray:
        f = _to_float01(self.image_data)
        if f is None:
            return None
        if self._autostretch_on:
            try:
                return np.clip(autostretch(f, target_median=0.25, linked=False), 0, 1)
            except Exception:
                pass
        return np.clip(f, 0, 1)

    def _rebuild_source(self):
        vis = self._make_vis()
        if vis is None:
            self.label.clear(); self._qimg_src = None; self._buf8 = None
            return
        self._qimg_src, self._buf8 = _mk_qimage_rgb8(vis)
        self._update_scaled()

    # zoom/pan
    def _set_zoom(self, z: float):
        self.zoom = float(max(0.05, min(z, 8.0)))
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.zoom * 100))
        self.slider.blockSignals(False)
        self._update_scaled()

    def _fit_to_view(self):
        if self._qimg_src is None:
            return
        vp = self.scroll.viewport().size()
        if self._qimg_src.width() == 0 or self._qimg_src.height() == 0:
            return
        s = min(vp.width() / self._qimg_src.width(), vp.height() / self._qimg_src.height())
        self._set_zoom(max(0.05, s))

    def _update_scaled(self):
        if self._qimg_src is None:
            return
        sw = max(1, int(self._qimg_src.width()  * self.zoom))
        sh = max(1, int(self._qimg_src.height() * self.zoom))
        scaled = self._qimg_src.scaled(sw, sh, Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(QPixmap.fromImage(scaled))
        self.label.resize(scaled.size())

    # actions
    def _toggle_autostretch(self):
        self._autostretch_on = not self._autostretch_on
        self._rebuild_source()

    def _open_compare(self):
        cur = getattr(self.doc, "image", None)
        if cur is None:
            QMessageBox.warning(self, "Compare", "No current image to compare.")
            return

        win = QWidget(self, Qt.WindowType.Window)
        win.setWindowTitle("Compare with Current")
        win.resize(900, 700)

        v = QVBoxLayout(win)

        help_lbl = QLabel(
            "Left-click + drag: move before/after divider  •  "
            "Middle-click + drag, or hold Space + left-click + drag: pan image  •  "
            "Ctrl+wheel: zoom"
        )
        help_lbl.setWordWrap(True)
        v.addWidget(help_lbl)

        self.slider_widget = ComparisonSlider(self.image_data, cur, parent=win)
        v.addWidget(self.slider_widget, 1)

        bar = QHBoxLayout()
        b_out = QPushButton("Zoom Out"); b_in = QPushButton("Zoom In")
        b_fit = QPushButton("Fit"); b_1  = QPushButton("1:1")
        b_st  = QPushButton("Toggle AutoStretch")
        b_out.clicked.connect(self.slider_widget.zoom_out)
        b_in.clicked.connect(self.slider_widget.zoom_in)
        b_fit.clicked.connect(self.slider_widget.fit_to_view)
        b_1.clicked.connect(lambda: self.slider_widget.set_zoom(1.0))
        b_st.clicked.connect(self.slider_widget.toggle_autostretch)

        bar.addWidget(b_out); bar.addWidget(b_in); bar.addWidget(b_fit); bar.addWidget(b_1)
        bar.addStretch(1); bar.addWidget(b_st)
        v.addLayout(bar)

        win.show()
        self.slider_widget.setFocus()
        mw = self._find_main_window()
        if mw and hasattr(mw, "_log"):
            mw._log("History: opened Compare with Current.")

    def _restore(self):
        try:
            # Prefer a method that records step name if available
            if hasattr(self.doc, "set_image"):
                self.doc.set_image(self.image_data.copy(), {"step_name": "Restored from History"})
            elif hasattr(self.doc, "update_image"):
                self.doc.update_image(self.image_data.copy(), {"step_name": "Restored from History"})
            else:
                QMessageBox.critical(self, "Restore", "Document does not support setting image.")
                return
            mw = self._find_main_window()
            if mw and hasattr(mw, "_log"):
                mw._log("History: restored image from history.")
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Restore failed", str(e))

    def _find_main_window(self):
        p = self.parent()
        while p is not None and not hasattr(p, "docman"):
            p = p.parent()
        return p

    # input
    def eventFilter(self, obj, ev):
        if obj is self.scroll.viewport():
            if ev.type() == QEvent.Type.Wheel:
                if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    if self._qimg_src is None or self.label.pixmap() is None:
                        return True
                    factor = 1.25 if ev.angleDelta().y() > 0 else 1/1.25
                    pos_vp = ev.position()
                    pos_lb = self.label.mapFrom(self.scroll.viewport(), pos_vp.toPoint())
                    old = self.label.pixmap().size()
                    rel_x = pos_lb.x() / max(1, old.width())
                    rel_y = pos_lb.y() / max(1, old.height())
                    self._set_zoom(self.zoom * factor)
                    new = self.label.pixmap().size()
                    hbar = self.scroll.horizontalScrollBar()
                    vbar = self.scroll.verticalScrollBar()
                    hbar.setValue(int(rel_x * new.width()  - self.scroll.viewport().width()/2))
                    vbar.setValue(int(rel_y * new.height() - self.scroll.viewport().height()/2))
                    return True
                return False

        if obj is self.scroll.viewport() or obj is self.label:
            if ev.type() == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = True
                self._pan_start = ev.position()
                self.scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
            if ev.type() == QEvent.Type.MouseMove and self._panning:
                d = ev.position() - self._pan_start
                hbar = self.scroll.horizontalScrollBar()
                vbar = self.scroll.verticalScrollBar()
                hbar.setValue(hbar.value() - int(d.x()))
                vbar.setValue(vbar.value() - int(d.y()))
                self._pan_start = ev.position()
                return True
            if ev.type() == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton:
                self._panning = False
                self.scroll.viewport().setCursor(Qt.CursorShape.ArrowCursor)
                return True

        return super().eventFilter(obj, ev)

class ComparisonSlider(QWidget):
    """Before/after slider with Ctrl+wheel zoom, Fit, 1:1, optional display autostretch.

    Panning:
      - Middle-click drag pans the image.
      - Hold Space and left-click drag pans the image (left-click alone
        without Space still moves the before/after divider).
    """
    def __init__(self, before_image: np.ndarray, after_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.before = np.asarray(before_image)
        self.after  = np.asarray(after_image)
        self.zoom = 1.0
        self.autostretch_on = False
        self.slider_pos = 0.5

        self.pan_x = 0.0
        self.pan_y = 0.0

        self._space_down = False
        self._panning = False
        self._pan_last_pos = None

        self._q_before = None; self._buf_before = None
        self._q_after  = None; self._buf_after  = None
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._rebuild()

    def _mk_vis(self, a: np.ndarray) -> np.ndarray:
        f = _to_float01(a)
        if self.autostretch_on:
            try:
                return np.clip(autostretch(f, target_median=0.25, linked=False), 0, 1)
            except Exception:
                pass
        return np.clip(f, 0, 1)

    def _rebuild(self):
        qb, bb = _mk_qimage_rgb8(self._mk_vis(self.before))
        qa, ba = _mk_qimage_rgb8(self._mk_vis(self.after))
        self._q_before, self._buf_before = qb, bb
        self._q_after,  self._buf_after  = qa, ba

    # public controls
    def set_zoom(self, z: float):
        self.zoom = float(max(0.05, min(z, 8.0))); self.update()
    def zoom_in(self):  self.set_zoom(self.zoom * 1.25)
    def zoom_out(self): self.set_zoom(self.zoom / 1.25)
    def fit_to_view(self):
        if not self._q_before: return
        W,H = self.width(), self.height()
        iw,ih = self._q_before.width(), self._q_before.height()
        if iw==0 or ih==0: return
        s = min(W/iw, H/ih)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.set_zoom(max(0.05, s))

    def toggle_autostretch(self):
        self.autostretch_on = not self.autostretch_on
        self._rebuild(); self.update()

    # painting & input
    def paintEvent(self, _ev):
        if not self._q_before or not self._q_after:
            return
        p = QPainter(self)
        W,H = self.width(), self.height()
        iw, ih = self._q_before.width(), self._q_before.height()
        if iw==0 or ih==0: return
        s = min(W/iw, H/ih) * self.zoom
        tw, th = int(iw*s), int(ih*s)
        b = self._q_before.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        a = self._q_after.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
        ox = (W - b.width()) // 2 + int(self.pan_x)
        oy = (H - b.height()) // 2 + int(self.pan_y)
        cut = int(W * self.slider_pos)

        p.save(); p.setClipRect(0, 0, cut, H); p.drawImage(ox, oy, b); p.restore()
        p.save(); p.setClipRect(cut, 0, W-cut, H); p.drawImage(ox, oy, a); p.restore()

        p.setPen(Qt.GlobalColor.red); p.drawLine(cut, 0, cut, H)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton or (
            ev.button() == Qt.MouseButton.LeftButton and self._space_down
        ):
            self._panning = True
            self._pan_last_pos = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if ev.button() == Qt.MouseButton.LeftButton:
            self._set_div(ev.position().x())

    def mouseMoveEvent(self, ev):
        if self._panning and self._pan_last_pos is not None:
            delta = ev.position() - self._pan_last_pos
            self.pan_x += delta.x()
            self.pan_y += delta.y()
            self._pan_last_pos = ev.position()
            self.update()
            return

        if ev.buttons() & Qt.MouseButton.LeftButton:
            self._set_div(ev.position().x())

    def mouseReleaseEvent(self, ev):
        if self._panning and (
            ev.button() == Qt.MouseButton.MiddleButton
            or ev.button() == Qt.MouseButton.LeftButton
        ):
            self._panning = False
            self._pan_last_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space_down else Qt.CursorShape.ArrowCursor)

    def _set_div(self, x):
        self.slider_pos = min(max(x / max(1, self.width()), 0.0), 1.0); self.update()

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(self.zoom * (1.25 if ev.angleDelta().y() > 0 else 0.8))
            ev.accept()
        else:
            ev.ignore()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self._space_down = True
            if not self._panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self._space_down = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        super().keyReleaseEvent(ev)