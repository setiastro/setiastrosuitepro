# pro/doc_manager.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal
import os
import numpy as np


from legacy.image_manager import load_image as legacy_load_image, save_image as legacy_save_image

def _normalize_ext(ext: str) -> str:
    e = ext.lower().lstrip(".")
    if e == "jpeg": return "jpg"
    if e == "tiff": return "tif"
    if e in ("fit", "fits"): return e
    return e

_ALLOWED_DEPTHS = {
    "png":  {"8-bit"},
    "jpg":  {"8-bit"},
    "fits": {"32-bit floating point"},
    "fit":  {"32-bit floating point"},
    "tif":  {"8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"},
    "xisf": {"16-bit", "32-bit unsigned", "32-bit floating point"},
}

class ImageDocument(QObject):
    changed = pyqtSignal()

    def __init__(self, image: np.ndarray, metadata: dict | None = None, parent=None):
        super().__init__(parent)
        self.image = image
        self.metadata = dict(metadata or {})
        self.mask = None
        self._undo: list[tuple[np.ndarray, dict, str]] = []
        self._redo: list[tuple[np.ndarray, dict, str]] = []        
        self.masks: dict[str, MaskLayer] = {}
        self.active_mask_id: str | None = None

    # --- history helpers (NEW) ---
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def last_undo_name(self) -> str | None:
        return self._undo[-1][2] if self._undo else None

    def last_redo_name(self) -> str | None:
        return self._redo[-1][2] if self._redo else None

    def add_mask(self, layer: MaskLayer, make_active: bool = True):
        self.masks[layer.id] = layer
        if make_active:
            self.active_mask_id = layer.id

    def remove_mask(self, mask_id: str):
        self.masks.pop(mask_id, None)
        if self.active_mask_id == mask_id:
            self.active_mask_id = None

    def get_active_mask(self):
        return self.masks.get(self.active_mask_id) if self.active_mask_id else None

    def apply_edit(self, new_image: np.ndarray, metadata: dict | None = None, step_name: str = "Edit"):
        """
        Push current state to undo (with step_name), clear redo, set new image/metadata, emit changed.
        """
        if self.image is not None:
            self._undo.append((self.image.copy(), self.metadata.copy(), step_name))
            self._redo.clear()
        if metadata:
            self.metadata.update(metadata)
        # also carry the step name onto the new “current” state for reference
        self.metadata.setdefault("step_name", step_name)
        self.image = new_image
        self.changed.emit()

    def undo(self) -> str | None:
        if not self._undo:
            return None
        prev_img, prev_meta, name = self._undo.pop()
        # push current to redo with same name
        self._redo.append((self.image.copy(), self.metadata.copy(), name))
        self.image = prev_img
        self.metadata = prev_meta
        self.changed.emit()
        return name

    def redo(self) -> str | None:
        if not self._redo:
            return None
        nxt_img, nxt_meta, name = self._redo.pop()
        self._undo.append((self.image.copy(), self.metadata.copy(), name))
        self.image = nxt_img
        self.metadata = nxt_meta
        self.changed.emit()
        return name

    # existing methods unchanged below...
    def set_image(self, img: np.ndarray, metadata: dict | None = None, step_name: str = "Edit"):
        """
        Treat set_image as an editing operation that records history.
        (History previews and “Restore from History” call this.)
        """
        self.apply_edit(img, metadata or {}, step_name=step_name)

    def display_name(self) -> str:
        p = self.metadata.get("file_path")
        return os.path.basename(p) if p else "Untitled"
    
    # --- Add to ImageDocument (public history helpers) -------------------

    def get_undo_stack(self):
        """
        Oldest → newest *before* current image.
        Returns [(img, meta, name), ...]
        """
        out = []
        for img, meta, name in self._undo:
            out.append((img, meta or {}, name or "Unnamed"))
        return out

    def display_name(self) -> str:
        # Prefer an explicit display name if set
        dn = self.metadata.get("display_name")
        if dn:
            return dn
        p = self.metadata.get("file_path")
        return os.path.basename(p) if p else "Untitled"


def _dm_json_sanitize(obj):
    """Tiny, local JSON sanitizer: keeps size small & avoids numpy/astropy weirdness."""
    import numpy as _np
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _dm_json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dm_json_sanitize(x) for x in obj]
    # numpy array → small placeholder
    try:
        if isinstance(obj, _np.ndarray):
            return {"__nd__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
        # numpy scalar
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    try:
        return repr(obj)
    except Exception:
        return str(type(obj))


def _snapshot_header_for_metadata(meta: dict):
    """
    If meta contains a header under common keys, add a JSON-safe snapshot at
    meta["__header_snapshot__"] so viewers/project IO never choke.
    """
    if not isinstance(meta, dict):
        return
    if "__header_snapshot__" in meta:
        return

    hdr = (meta.get("original_header")
           or meta.get("fits_header")
           or meta.get("header"))

    if hdr is None:
        return

    snap = None

    # Try astropy Header (without hard dependency)
    try:
        from astropy.io.fits import Header  # type: ignore
    except Exception:
        Header = None  # type: ignore

    try:
        if Header is not None and isinstance(hdr, Header):
            cards = []
            for k in hdr.keys():
                try:
                    val = hdr[k]
                except Exception:
                    val = ""
                try:
                    cmt = hdr.comments[k] if hasattr(hdr, "comments") else ""
                except Exception:
                    cmt = ""
                cards.append([str(k), _dm_json_sanitize(val), str(cmt)])
            snap = {"format": "fits-cards", "cards": cards}
        elif isinstance(hdr, dict):
            # Already a dict-like header (e.g., XISF style)
            snap = {"format": "dict",
                    "items": {str(k): _dm_json_sanitize(v) for k, v in hdr.items()}}
        else:
            # Last resort string
            snap = {"format": "repr", "text": repr(hdr)}
    except Exception:
        try:
            snap = {"format": "repr", "text": str(hdr)}
        except Exception:
            snap = None

    if snap:
        meta["__header_snapshot__"] = snap

class DocManager(QObject):
    documentAdded = pyqtSignal(object)   # ImageDocument
    documentRemoved = pyqtSignal(object) # ImageDocument

    def __init__(self, image_manager=None, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self._docs: list[ImageDocument] = []
        self._active_doc: ImageDocument | None = None
        self._mdi: "QMdiArea | None" = None  # type: ignore


    # --- File -> Document ---
    def open_path(self, path: str) -> "ImageDocument":
        img, header, bit_depth, is_mono = legacy_load_image(path)
        if img is None:
            raise IOError(f"Could not load: {path}")

        ext = os.path.splitext(path)[1].lower().lstrip('.')
        meta = {
            "file_path": path,
            "original_header": header,                 # keep raw
            "bit_depth": bit_depth,
            "is_mono": is_mono,
            "original_format": _normalize_ext(ext),
        }
        # add snapshot so viewers/project IO are safe
        _snapshot_header_for_metadata(meta)

        doc = ImageDocument(img, meta)
        self._docs.append(doc)
        self.documentAdded.emit(doc)
        return doc
    
    # --- Slot -> Document ---
    def open_from_slot(self, slot_idx: int | None = None) -> "ImageDocument | None":
        if not self.image_manager:
            return None

        if slot_idx is None:
            slot_idx = getattr(self.image_manager, "current_slot", 0)

        img = self.image_manager.get_image_for_slot(slot_idx)
        if img is None:
            return None

        meta = {}
        try:
            meta = dict(self.image_manager._metadata.get(slot_idx, {}))
        except Exception:
            pass

        meta.setdefault("file_path", f"Slot {slot_idx}")
        meta.setdefault("bit_depth", "32-bit floating point")
        meta.setdefault("is_mono", (img.ndim == 2))
        meta.setdefault("original_header", meta.get("original_header"))  # whatever SASv2 had
        meta.setdefault("original_format", "fits")

        _snapshot_header_for_metadata(meta)

        doc = ImageDocument(img, meta)
        self._docs.append(doc)
        self.documentAdded.emit(doc)
        return doc
    
    # --- Save ---
    def _infer_bit_depth_for_format(self, img: np.ndarray, ext: str, current_bit_depth: str | None) -> str:
        # Previous heuristic fallback (used only if no override provided).
        if ext in ("png", "jpg"):
            return "8-bit"
        if ext in ("fits", "fit"):
            return "32-bit floating point"
        if ext == "tif":
            if current_bit_depth in _ALLOWED_DEPTHS["tif"]:
                return current_bit_depth
            return "16-bit" if np.issubdtype(img.dtype, np.floating) else "8-bit"
        if ext == "xisf":
            return current_bit_depth if current_bit_depth in _ALLOWED_DEPTHS["xisf"] else "32-bit floating point"
        return "32-bit floating point"

    def save_document(self, doc: ImageDocument, path: str, bit_depth_override: str | None = None):
        ext = _normalize_ext(os.path.splitext(path)[1])
        img = doc.image

        # Decide bit depth (override → fallback inference)
        if bit_depth_override:
            allowed = _ALLOWED_DEPTHS.get(ext, set())
            if allowed and bit_depth_override not in allowed:
                # Guardrail: if someone passes an invalid choice, fallback
                bit_depth = next(iter(allowed))
            else:
                bit_depth = bit_depth_override
        else:
            bit_depth = self._infer_bit_depth_for_format(img, ext, doc.metadata.get("bit_depth"))

        # For integer encodes, clip to [0..1] to avoid wrap/overflow
        needs_clip = ext in ("png", "jpg", "tif") and bit_depth in ("8-bit", "16-bit", "32-bit unsigned")
        img_to_save = np.clip(img, 0.0, 1.0) if needs_clip else img

        legacy_save_image(
            img_array=img_to_save,
            filename=path,
            original_format=ext,
            bit_depth=bit_depth,
            original_header=doc.metadata.get("original_header"),
            is_mono=doc.metadata.get("is_mono", img.ndim == 2),
            image_meta=doc.metadata.get("image_meta"),
            file_meta=doc.metadata.get("file_meta"),
        )

        # Update metadata with the user’s choice
        doc.metadata["file_path"] = path
        doc.metadata["original_format"] = ext
        doc.metadata["bit_depth"] = bit_depth
        doc.changed.emit()

    def duplicate_document(self, source_doc: ImageDocument, new_name: str | None = None) -> ImageDocument:
        img_copy = source_doc.image.copy() if source_doc.image is not None else None
        meta = dict(source_doc.metadata or {})

        # Give it a nice display name
        base = source_doc.display_name()
        dup_title = new_name or f"{base}_duplicate"
        meta["display_name"] = dup_title

        # Fresh document (empty undo/redo)
        dup = ImageDocument(img_copy, meta, parent=self.parent())
        self._docs.append(dup)
        self.documentAdded.emit(dup)
        return dup

    #def open_array(self, arr, metadata: dict | None = None, title: str | None = None) -> ImageDocument:
    #    import numpy as np
    ##    if arr is None:
    #        raise ValueError("open_array: arr is None")
    #    img = np.asarray(arr)
    #    if img.dtype != np.float32:
    #        img = img.astype(np.float32, copy=False)

    #    meta = dict(metadata or {})
    #    meta.setdefault("bit_depth", "32-bit floating point")
    #    meta.setdefault("is_mono", img.ndim == 2)
    #    meta.setdefault("original_header", meta.get("original_header"))
    #    meta.setdefault("original_format", meta.get("original_format", "fits"))
    #    if title:
    #        meta.setdefault("display_name", title)

    #    doc = ImageDocument(img, meta, parent=self.parent())
    #    self._docs.append(doc)
    #    self.documentAdded.emit(doc)
    #    return doc

    # convenient aliases used by your tool code
    def open_array(self, img: np.ndarray, metadata: dict | None = None, title: str | None = None) -> "ImageDocument":
        meta = dict(metadata or {})
        if title:
            meta["display_name"] = title
        # normalize a few expected fields if missing
        try:
            if "is_mono" not in meta and isinstance(img, np.ndarray):
                meta["is_mono"] = (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1))
        except Exception:
            pass
        meta.setdefault("bit_depth", meta.get("bit_depth", "32-bit floating point"))

        _snapshot_header_for_metadata(meta)

        doc = ImageDocument(img, meta, parent=self.parent())
        self._docs.append(doc)
        self.documentAdded.emit(doc)
        return doc

    # (optional alias for old code)
    open_numpy = open_array
    create_document = open_array

    def create_document(self, image, metadata: dict | None = None, name: str | None = None) -> ImageDocument:
        return self.open_array(image, metadata=metadata, title=name)

    def close_document(self, doc):
        if doc in self._docs:
            self._docs.remove(doc)
            self.documentRemoved.emit(doc)

    # --- Active-document helpers (NEW) ---------------------------------
    def all_documents(self):
        return list(self._docs)

    def _find_main_window(self):
        from PyQt6.QtWidgets import QMainWindow, QApplication
        w = self.parent()
        while w is not None and not isinstance(w, QMainWindow):
            w = w.parent()
        if w:
            return w
        for tlw in QApplication.topLevelWidgets():
            if isinstance(tlw, QMainWindow):
                return tlw
        return None

    def set_active_document(self, doc: ImageDocument | None):
        # only track docs we know about
        if doc is not None and doc not in self._docs:
            return
        self._active_doc = doc

    def set_mdi_area(self, mdi):
        """Call this once from MainWindow after MDI is created."""
        self._mdi = mdi
        try:
            mdi.subWindowActivated.connect(self._on_subwindow_activated)
        except Exception:
            pass

    def _on_subwindow_activated(self, sw):
        # Map active subwindow -> its ImageDocument
        doc = None
        try:
            if sw is not None:
                # most code keeps the document on the widget
                w = sw.widget()
                doc = getattr(w, "document", None) or getattr(sw, "document", None)
        except Exception:
            doc = None
        self.set_active_document(doc)

    def get_active_document(self):
        """
        Return the currently active document if known. Fallbacks:
        - Ask MDI directly if available
        - Last opened doc
        """
        # 1) If we already know it, use it.
        if self._active_doc is not None:
            return self._active_doc if (self._active_doc in self._docs) else None

        # 2) Try asking MDI directly
        try:
            if self._mdi is not None:
                sw = self._mdi.activeSubWindow()
                if sw is not None:
                    w = sw.widget()
                    doc = getattr(w, "document", None) or getattr(sw, "document", None)
                    if doc is not None:
                        self._active_doc = doc
                        return doc
        except Exception:
            pass

        # 3) Fallback: last doc
        return self._docs[-1] if self._docs else None

    def update_active_document(self, updated_image, metadata=None, step_name: str = "Edit"):
        """
        Apply an edit to the active document (records undo/redo).
        """
        import numpy as np
        doc = self.get_active_document()
        if doc is None:
            raise RuntimeError("No active document")
        img = np.asarray(updated_image)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        doc.apply_edit(img, dict(metadata or {}), step_name)

    # Back-compat/aliases so tools can call any of these:
    def update_image(self, updated_image, metadata=None, step_name: str = "Edit"):
        self.update_active_document(updated_image, metadata, step_name)

    def set_image(self, img, metadata=None, step_name: str = "Edit"):
        self.update_active_document(img, metadata, step_name)

    def apply_edit_to_active(self, img, step_name: str = "Edit", metadata=None):
        self.update_active_document(img, metadata, step_name)
