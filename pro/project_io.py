# pro/project_io.py
from __future__ import annotations
import io, os, json, time, zipfile, uuid, pickle
from typing import Any, Dict, List, Tuple
import numpy as np
from PyQt6.QtWidgets import QMdiSubWindow
from PyQt6.QtCore import QPoint

# ---------- helpers ----------
def _np_save_to_bytes(arr: np.ndarray, *, compress: bool = True) -> bytes:
    """
    If compress=True -> np.savez_compressed (zlib)
    If compress=False -> np.save (raw .npy)
    """
    bio = io.BytesIO()
    arr = arr.astype(np.float32, copy=False)
    if compress:
        np.savez_compressed(bio, img=arr)
    else:
        np.save(bio, arr)
    return bio.getvalue()


def _np_load_from_bytes(data: bytes) -> np.ndarray:
    """
    Reads both npz-with-{'img'} and raw npy.
    Detect format by magic header.
    """
    # ZIP magic for .npz
    if data[:4] == b'PK\x03\x04':
        bio = io.BytesIO(data)
        with np.load(bio, allow_pickle=False) as z:
            return z["img"].astype(np.float32, copy=False)
    # .npy magic: \x93NUMPY
    bio = io.BytesIO(data)
    arr = np.load(bio, allow_pickle=False)
    return arr.astype(np.float32, copy=False)


def _now_iso() -> str:
    try:
        import datetime as _dt
        return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return ""

def _json_sanitize(obj):
    """
    Make arbitrary metadata JSON-serializable.
    - numpy scalars/arrays -> lists or float/int
    - objects we can't encode -> string repr
    """
    import numpy as _np
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, _np.ndarray):
        # avoid massive JSON; store shape/dtype only
        return {"__nd__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    # numpy scalar
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    # astropy header or others -> repr
    try:
        return repr(obj)
    except Exception:
        return str(type(obj))

# ---------- main IO ----------
# pro/project_io.py
import zipfile


class ProjectWriter:
    VERSION = 1

    @staticmethod
    def write(path: str, *, docs: list, shortcuts=None, mdi=None, compress: bool = True, shelf=None):
      
        """
        Write a .sas project.
        compress=False → much faster saves (bigger file).
        """
        docs = list(docs or [])
        id_map = {doc: uuid.uuid4().hex for doc in docs}

        # --- UI / subwindow geometry ---
        ui = {"views": [], "active_doc_id": None}
        minimized_set = set()
        saved_state = {}
        if shelf is not None:
            try:
                minimized_set = set(getattr(shelf, "_item2sub", {}).values())
                saved_state = getattr(shelf, "_saved_state", {})
            except Exception:
                pass

        if mdi is not None:
            try:
                active_sw = mdi.activeSubWindow()
            except Exception:
                active_sw = None

            for sw in getattr(mdi, "subWindowList", lambda: [])():
                try:
                    view = sw.widget()
                    doc = getattr(view, "document", None)
                    if doc not in id_map:
                        continue

                    is_min = sw in minimized_set
                    # choose the rectangle we want to persist
                    rect = None
                    was_max = False
                    if is_min:
                        st = saved_state.get(sw, {})
                        if isinstance(st.get("geom"), QRect):
                            rect = QRect(st["geom"])
                        was_max = bool(st.get("max", False))
                    if rect is None:
                        # normal/maximized windows
                        was_max = was_max or bool(sw.isMaximized())
                        rect = sw.normalGeometry() if was_max else sw.geometry()

                    ui["views"].append({
                        "doc_id": id_map[doc],
                        "x": rect.x(), "y": rect.y(),
                        "w": rect.width(), "h": rect.height(),
                        "minimized": bool(is_min),
                        "was_max": bool(was_max),
                    })
                    if sw is active_sw and not is_min:
                        ui["active_doc_id"] = id_map[doc]
                except Exception:
                    pass

        # --- Shortcuts dump ---
        sc_dump = []
        if shortcuts is not None:
            for sid, w in list(getattr(shortcuts, "widgets", {}).items()):
                try:
                    if hasattr(w, "isVisible") and not w.isVisible():
                        continue
                    p = w.pos()
                    try:
                        preset = w._load_preset()
                    except Exception:
                        preset = None
                    sc_dump.append({
                        "id": sid,
                        "command_id": w.command_id,
                        "label": w.text(),
                        "x": p.x(), "y": p.y(),
                        "preset": preset or None,
                    })
                except Exception:
                    continue

        # --- Manifest + zip mode ---
        manifest = {
            "version": ProjectWriter.VERSION,
            "created": _now_iso(),
            "doc_count": len(docs),
        }
        zip_mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED

        with zipfile.ZipFile(path, "w", compression=zip_mode, allowZip64=True) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
            z.writestr("ui.json", json.dumps(ui, indent=2))
            z.writestr("shortcuts.json", json.dumps(sc_dump, indent=2))

            # per-document payloads
            cur_ext = "npz" if compress else "npy"
            hist_ext = cur_ext

            for doc in docs:
                doc_id = id_map[doc]
                base = f"views/{doc_id}"
                meta = dict(getattr(doc, "metadata", {}) or {})
                meta.setdefault("display_name", doc.display_name())
                safe_meta = _json_sanitize(meta)

                z.writestr(f"{base}/meta.json", json.dumps(safe_meta, indent=2))

                # current image
                if getattr(doc, "image", None) is not None:
                    z.writestr(f"{base}/current.{cur_ext}", _np_save_to_bytes(doc.image, compress=compress))

                # history stacks
                undo_list = []
                for i, (img, m, name) in enumerate(getattr(doc, "_undo", []) or []):
                    fname = f"history/undo_{i:04d}.{hist_ext}"
                    undo_list.append({
                        "name": name or "Edit",
                        "meta": _json_sanitize(m or {}),
                        "file": fname
                    })
                    z.writestr(f"{base}/{fname}", _np_save_to_bytes(img, compress=compress))

                redo_list = []
                for i, (img, m, name) in enumerate(getattr(doc, "_redo", []) or []):
                    fname = f"history/redo_{i:04d}.{hist_ext}"
                    redo_list.append({
                        "name": name or "Edit",
                        "meta": _json_sanitize(m or {}),
                        "file": fname
                    })
                    z.writestr(f"{base}/{fname}", _np_save_to_bytes(img, compress=compress))

                z.writestr(
                    f"{base}/history/stack.json",
                    json.dumps({"undo": undo_list, "redo": redo_list}, indent=2),
                )



class ProjectReader:
    def __init__(self, main_window):
        self.mw = main_window
        # Prefer the new name, fall back to the old one if present
        self.dm = getattr(main_window, "doc_manager", None) or getattr(main_window, "dm", None)
        self.sc = getattr(main_window, "shortcuts", None)

    def read(self, path: str):
        if self.dm is None:
            raise RuntimeError("No DocManager available")

        if not zipfile.is_zipfile(path):
            LegacyProjectReader(self.mw).read(path)
            return

        with zipfile.ZipFile(path, "r") as z:
            # Ensure we have a ShortcutManager and restore shortcuts ONCE
            if not getattr(self.mw, "shortcuts", None):
                from pro.doc_manager import ShortcutManager
                self.mw.shortcuts = ShortcutManager(self.mw.mdi, self.mw)
            self.sc = self.mw.shortcuts

            try:
                self._restore_shortcuts(z)     # ← do this ONCE, before docs
            except Exception:
                pass

            doc_id_map = {}
            # now iterate docs only
            for name in z.namelist():
                if not name.startswith("views/") or not name.endswith("/meta.json"):
                    continue
                base = os.path.dirname(name)
                doc_id = base.split("/")[1]

                # meta
                try:
                    meta = json.loads(z.read(f"{base}/meta.json").decode("utf-8"))
                except Exception:
                    meta = {}

                # current (try npz then npy)
                img = None
                try:
                    img = _np_load_from_bytes(z.read(f"{base}/current.npz"))
                except Exception:
                    try:
                        img = _np_load_from_bytes(z.read(f"{base}/current.npy"))
                    except Exception:
                        img = None

                disp = meta.get("display_name") or "Untitled"
                doc = self.dm.create_document(img, metadata=meta, name=disp)
                doc_id_map[doc_id] = doc

                # history
                try:
                    stack = json.loads(z.read(f"{base}/history/stack.json").decode("utf-8"))
                except Exception:
                    stack = {"undo": [], "redo": []}

                undo_tuples = []
                for entry in stack.get("undo", []):
                    fname = entry.get("file")
                    try:
                        arr = _np_load_from_bytes(z.read(f"{base}/{fname}"))
                        undo_tuples.append((arr, entry.get("meta") or {}, entry.get("name") or "Edit"))
                    except Exception:
                        continue
                doc._undo = undo_tuples

                redo_tuples = []
                for entry in stack.get("redo", []):
                    fname = entry.get("file")
                    try:
                        arr = _np_load_from_bytes(z.read(f"{base}/{fname}"))
                        redo_tuples.append((arr, entry.get("meta") or {}, entry.get("name") or "Edit"))
                    except Exception:
                        continue
                doc._redo = redo_tuples

            # restore UI geometry/minimized
            try:
                ui = json.loads(z.read("ui.json").decode("utf-8"))
                self._restore_ui(ui, doc_id_map)
            except Exception:
                # fallback: at least open all docs
                for doc in doc_id_map.values():
                    try:
                        self.mw._spawn_subwindow_for(doc)
                    except Exception:
                        pass

            # After everything is created, bring the canvas to the front & ready
            self._post_restore_shortcuts()

    def _post_restore_shortcuts(self):
        """Ensure the shortcuts canvas is interactive and on top after restore."""
        sc = getattr(self, "sc", None)
        if not sc:
            return
        try:
            c = sc.canvas
            # make sure we're parented and sized to the viewport
            if c.parent() is not sc.mdi.viewport():
                c.setParent(sc.mdi.viewport())
            c.setGeometry(sc.mdi.viewport().rect())
            # ensure it will actually receive mouse/keyboard events
            c.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            c.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            c.raise_()
            c.show()
            # Wake up the child buttons (some styles need a nudge)
            for w in list(sc.widgets.values()):
                try:
                    w.setEnabled(True)
                    w.show()
                    w.raise_()
                except Exception:
                    pass
        except Exception:
            pass

    # ---------- helpers ----------
    def _restore_shortcuts(self, z: zipfile.ZipFile):
        if not self.sc:
            return
        data = []
        try:
            data = json.loads(z.read("shortcuts.json").decode("utf-8"))
        except Exception:
            return

        # Clear existing canvas
        try:
            self.sc.clear()
        except Exception:
            pass

        # Recreate
        for entry in data:
            cid = entry.get("command_id")
            sid = entry.get("id") or uuid.uuid4().hex
            label = entry.get("label") or cid
            x = int(entry.get("x", 10)); y = int(entry.get("y", 10))
            w = self.sc.add_shortcut(
                    cid,
                    QPoint(x, y),
                    label=label,
                    shortcut_id=sid,
                )
            # move exact
            try:
                w = self.sc.widgets.get(sid)
                if w:
                    w.move(x, y)
            except Exception:
                pass
            # preset
            preset = entry.get("preset")
            if preset is not None:
                try:
                    w = self.sc.widgets.get(sid)
                    if w:
                        w._save_preset(preset)
                except Exception:
                    pass
        # persist
        try:
            self.sc.save_shortcuts()
        except Exception:
            pass

    def _restore_ui(self, ui: dict, id_map: dict):
        views = ui.get("views", [])
        active_id = ui.get("active_doc_id")
        active_sw = None
        shelf = getattr(self.mw, "window_shelf", None)

        for v in views:
            doc = id_map.get(v.get("doc_id"))
            if not doc:
                continue

            sw = self.mw._spawn_subwindow_for(doc)

            # geometry from project
            x = int(v.get("x", sw.x()))
            y = int(v.get("y", sw.y()))
            w = int(v.get("w", sw.width()))
            h = int(v.get("h", sw.height()))
            was_max = bool(v.get("was_max", False))
            is_min = bool(v.get("minimized", False))

            try:
                if not is_min:
                    # normal path: restore rect / max
                    if was_max:
                        sw.showMaximized()
                    else:
                        sw.setWindowState(Qt.WindowState.WindowNoState)
                        sw.setGeometry(QRect(x, y, w, h))
                        sw.showNormal()
                else:
                    # minimized path: seed shelf saved state with our rect/max and add entry
                    if shelf is not None:
                        try:
                            # ensure saved state exists for this subwindow
                            shelf._saved_state[sw] = {"geom": QRect(x, y, w, h), "max": was_max}
                        except Exception:
                            pass
                        sw.setWindowState(Qt.WindowState.WindowNoState)
                        sw.setGeometry(QRect(x, y, w, h))
                        sw.hide()
                        try:
                            shelf.add_entry(sw)
                        except Exception:
                            # fallback: capture-then-add
                            try:
                                shelf.pre_capture_state(sw)
                                shelf.add_entry(sw)
                            except Exception:
                                pass
                    else:
                        # no shelf available; just hide at the geometry so it’s not intrusive
                        sw.setGeometry(QRect(x, y, w, h))
                        sw.hide()
            except Exception:
                pass

            if v.get("doc_id") == active_id and not is_min:
                active_sw = sw

        if active_sw:
            self.mw.mdi.setActiveSubWindow(active_sw)

class LegacyProjectReader:
    """
    Reads SASv2 pickle projects and coerces them into SASpro documents.
    This class is completely separate so SASpro loading behavior is unchanged.
    """
    def __init__(self, main_window):
        self.mw = main_window
        self.dm = getattr(main_window, "doc_manager", None) or getattr(main_window, "dm", None)
        self.sc = getattr(main_window, "shortcuts", None)

    def read(self, path: str):
        import pickle
        import numpy as np

        if self.dm is None:
            raise RuntimeError("No DocManager available")

        with open(path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Not a SASpro project and failed to parse legacy SASv2 pickle: {e}")

        images: dict       = data.get("images") or {}
        meta_by_slot: dict = data.get("metadata") or {}
        slot_names: dict   = data.get("slot_names") or {}
        undo_by_slot: dict = data.get("undo_stacks") or {}
        redo_by_slot: dict = data.get("redo_stacks") or {}
        masks: dict        = data.get("masks") or {}
        current_slot       = data.get("current_slot", None)

        # Legacy projects had no shortcuts; ensure manager exists but don't restore anything
        if not getattr(self.mw, "shortcuts", None):
            try:
                from pro.doc_manager import ShortcutManager
                self.mw.shortcuts = ShortcutManager(self.mw.mdi, self.mw)
            except Exception:
                pass
        self.sc = getattr(self.mw, "shortcuts", None)

        _log = getattr(self.mw, "update_status", None)

        doc_for_slot = {}
        active_sw = None
        first_sw = None

        for slot, arr in sorted(images.items(), key=lambda kv: kv[0]):
            # Convert to array
            try:
                img = np.asarray(arr)
            except Exception:
                if _log: _log(f"Skipping slot {slot}: unreadable image payload.")
                continue

            # Skip empty/tiny (≤ 10×10)
            if self._is_tiny_or_empty(img):
                if _log: _log(f"Skipping slot {slot}: empty/tiny image (≤ 10×10).")
                continue

            # Normalize dtype
            if img.dtype != np.float32:
                img = img.astype(np.float32, copy=False)

            disp = slot_names.get(slot) or f"Slot {slot}"
            meta = dict(meta_by_slot.get(slot, {}) or {})
            meta.setdefault("source", "SASv2")
            if slot in masks and masks[slot] is not None:
                meta["legacy_mask_present"] = True

            doc = self.dm.create_document(img, metadata=meta, name=disp)

            # Attach legacy mask (in-memory only)
            try:
                if slot in masks and masks[slot] is not None:
                    setattr(doc, "_legacy_mask", np.asarray(masks[slot]))
            except Exception:
                pass

            # Undo/Redo
            doc._undo = self._coerce_legacy_stack(undo_by_slot.get(slot) or [])
            doc._redo = self._coerce_legacy_stack(redo_by_slot.get(slot) or [])

            doc_for_slot[slot] = doc

            # Open subwindow
            try:
                sw = self.mw._spawn_subwindow_for(doc)
                if first_sw is None:
                    first_sw = sw
                if current_slot is not None and slot == current_slot:
                    active_sw = sw
            except Exception:
                pass

        if not doc_for_slot:
            if _log: _log("No non-empty slots found in legacy project.")
            return

        try:
            self.mw.mdi.setActiveSubWindow(active_sw or first_sw)
        except Exception:
            pass

    @staticmethod
    def _is_tiny_or_empty(img) -> bool:
        import numpy as _np
        if img is None:
            return True
        if not isinstance(img, _np.ndarray):
            return True
        if img.ndim < 2:
            return True
        h, w = img.shape[:2]
        return (h <= 10 or w <= 10)

    def _coerce_legacy_stack(self, stack_list):
        import numpy as np
        out = []
        for entry in stack_list:
            try:
                if isinstance(entry, tuple):
                    arr = entry[0]
                    meta = entry[1] if len(entry) >= 2 and isinstance(entry[1], dict) else {}
                    name = entry[2] if len(entry) >= 3 and isinstance(entry[2], str) else "Edit"
                else:
                    arr = entry
                    meta, name = {}, "Edit"
                arr = np.asarray(arr).astype(np.float32, copy=False)
                out.append((arr, meta, name))
            except Exception:
                continue
        return out            
