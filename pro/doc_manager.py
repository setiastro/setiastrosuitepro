# pro/doc_manager.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox
import os
import numpy as np
from legacy.xisf import XISF as XISFReader
from astropy.io import fits  # local import; optional dep
from legacy.image_manager import load_image as legacy_load_image, save_image as legacy_save_image
from legacy.image_manager import list_fits_extensions, load_fits_extension

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

class TableDocument(QObject):
    changed = pyqtSignal()

    def __init__(self, rows: list[list], headers: list[str], metadata: dict | None = None, parent=None):
        super().__init__(parent)
        self.rows = rows              # list of list (2D) for QAbstractTableModel
        self.headers = headers        # list of column names
        self.metadata = dict(metadata or {})
        self._undo = []
        self._redo = []

    def display_name(self) -> str:
        dn = self.metadata.get("display_name")
        if dn:
            return dn
        p = self.metadata.get("file_path")
        return os.path.basename(p) if p else "Untitled Table"

    def can_undo(self) -> bool: return False
    def can_redo(self) -> bool: return False
    def last_undo_name(self) -> str | None: return None
    def last_redo_name(self) -> str | None: return None
    def undo(self) -> str | None: return None
    def redo(self) -> str | None: return None

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

    # in class ImageDocument
    def apply_edit(self, new_image: np.ndarray, metadata: dict | None = None, step_name: str = "Edit"):
        """
        Smart edit:
        - If this is an ROI view (has _roi_info), paste back into parent and emit region update.
        - Else: push history on self and emit full-image update.
        """
        import numpy as np
        #print(f"[ImageDocument] apply_edit called: step_name={step_name}")

        # ------ ROI-aware branch (auto-pasteback) ------
        roi_info = getattr(self, "_roi_info", None)
        if roi_info:
            parent = roi_info.get("parent_doc")
            roi    = roi_info.get("roi")
            if isinstance(parent, ImageDocument) and (getattr(parent, "image", None) is not None) and roi:
                x, y, w, h = map(int, roi)

                img = np.asarray(new_image)
                if img.dtype != np.float32:
                    img = img.astype(np.float32, copy=False)

                base = np.asarray(parent.image)
                if img.shape[:2] != (h, w):
                    raise ValueError(f"Edited preview shape {img.shape[:2]} does not match ROI {(h, w)}")

                # shape reconciliation
                if base.ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
                    img = img[..., 0]
                if base.ndim == 3 and img.ndim == 2:
                    img = np.repeat(img[..., None], base.shape[2], axis=2)

                new_full = base.copy()
                new_full[y:y+h, x:x+w] = img

                # push onto the PARENT’s history
                if metadata:
                    parent.metadata.update(metadata)
                parent.metadata.setdefault("step_name", step_name)
                parent._undo.append((parent.image.copy(), parent.metadata.copy(), step_name))
                parent._redo.clear()
                parent.image = new_full
                parent.changed.emit()

                # notify views about the region that changed
                dm = getattr(self, "_doc_manager", None) or getattr(parent, "_doc_manager", None)
                try:
                    if dm is not None and hasattr(dm, "imageRegionUpdated"):
                        dm.imageRegionUpdated.emit(parent, (x, y, w, h))
                        #print(f"[DocManager] Emitted imageRegionUpdated for ROI: {(x, y, w, h)}")
                      
                except Exception:
                    print(f"[DocManager] Failed to emit imageRegionUpdated for ROI.")
                    pass
                return  # done

        # ------ Normal (full-image) branch ------
        if self.image is not None:
            self._undo.append((self.image.copy(), self.metadata.copy(), step_name))
            self._redo.clear()
        if metadata:
            self.metadata.update(metadata)
        self.metadata.setdefault("step_name", step_name)

        img = np.asarray(new_image)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        self.image = img
        self.changed.emit()

        # full-image repaint hint to views
        dm = getattr(self, "_doc_manager", None)
        try:
            if dm is not None and hasattr(dm, "imageRegionUpdated"):
                dm.imageRegionUpdated.emit(self, None)
        except Exception:
            pass


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

def _safe_str(x) -> str:
    try:
        return str(x)
    except Exception:
        try:
            return repr(x)
        except Exception:
            return "<unrepr>"

def _fits_table_to_csv(hdu, out_csv_path: str, max_rows: int = 250000):
    """
    Convert a FITS (Bin)Table HDU to CSV. Returns the CSV path.
    Limits to max_rows to avoid giant dumps.
    """
    try:
        data = hdu.data
        if data is None:
            raise RuntimeError("No table data")

        # Astropy table→numpy recarray is fine; iterate to strings
        rec = np.asarray(data)
        nrows = int(rec.shape[0]) if rec.ndim >= 1 else 0
        if nrows == 0:
            # write headers only
            names = [str(n) for n in (getattr(data, "names", None) or [])]
            with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
                if names:
                    f.write(",".join(names) + "\n")
            return out_csv_path

        # Column names (fallback to numeric if missing)
        names = list(getattr(data, "names", [])) or [f"C{i+1}" for i in range(rec.shape[1] if rec.ndim == 2 else len(rec.dtype.names or []))]

        import csv
        with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([_safe_str(n) for n in names])

            # Decide how to iterate rows depending on structured vs 2D numeric
            if rec.dtype.names:  # structured/record array
                for ri in range(min(nrows, max_rows)):
                    row = rec[ri]
                    w.writerow([_safe_str(row[name]) for name in rec.dtype.names])
            else:
                # plain 2D numeric table
                if rec.ndim == 1:
                    for ri in range(min(nrows, max_rows)):
                        w.writerow([_safe_str(rec[ri])])
                else:
                    for ri in range(min(nrows, max_rows)):
                        w.writerow([_safe_str(x) for x in rec[ri]])

        return out_csv_path
    except Exception as e:
        raise

def _fits_table_to_rows_headers(hdu, max_rows: int = 500000) -> tuple[list[list], list[str]]:
    """
    Convert a FITS (Bin)Table/Table HDU to (rows, headers).
    Truncates to max_rows for safety.
    """
    data = hdu.data
    if data is None:
        return [], []
    rec = np.asarray(data)
    # Column names
    names = list(getattr(data, "names", [])) or (
        list(rec.dtype.names) if rec.dtype.names else [f"C{i+1}" for i in range(rec.shape[1] if rec.ndim == 2 else 1)]
    )
    rows = []
    nrows = int(rec.shape[0]) if rec.ndim >= 1 else 0
    nrows = min(nrows, max_rows)
    if rec.dtype.names:  # structured array
        for ri in range(nrows):
            row = rec[ri]
            rows.append([_safe_str(row[name]) for name in rec.dtype.names])
    else:
        # numeric 2D/1D table
        if rec.ndim == 1:
            for ri in range(nrows):
                rows.append([_safe_str(rec[ri])])
        else:
            for ri in range(nrows):
                rows.append([_safe_str(x) for x in rec[ri]])
    return rows, [str(n) for n in names]


_shown_raw_preview_paths: set[str] = set()
_raw_preview_boxes: list[QMessageBox] = []  # prevent GC while shown

def _show_raw_preview_warning_nonmodal(path: str):
    parent = QApplication.activeWindow()
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("RAW preview loaded")
    box.setText(
        "Linear RAW decoding failed for:\n"
        f"{path}\n\n"
        "Showing the camera’s embedded JPEG preview instead "
        "(8-bit, non-linear). Some processing tools may be limited."
    )
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.setWindowModality(Qt.WindowModality.NonModal)  # ← fix here
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

    _raw_preview_boxes.append(box)
    box.finished.connect(lambda _=None, b=box: _raw_preview_boxes.remove(b))
    box.show()

def maybe_warn_raw_preview(path: str, header):
    if not header or not bool(header.get("RAW_PREV", False)):
        return
    if path in _shown_raw_preview_paths:
        return
    _shown_raw_preview_paths.add(path)
    QTimer.singleShot(0, lambda p=path: _show_raw_preview_warning_nonmodal(p))

_np = np

class _RoiViewDocument(ImageDocument):
    def __init__(self, parent_doc: ImageDocument, roi: tuple[int,int,int,int], name_suffix: str = " (Preview)"):
        x, y, w, h = roi
        meta = dict(parent_doc.metadata or {})
        base = parent_doc.display_name()
        meta["display_name"] = f"{base}{name_suffix}"
        meta.setdefault("image_meta", {})
        meta["image_meta"] = dict(meta["image_meta"], readonly=True, view_kind="roi-preview")

        super().__init__(_np.zeros((max(1,h), max(1,w), 3), dtype=_np.float32), meta, parent=parent_doc.parent())

        self._parent_doc = parent_doc
        self._roi = ( x, y, w, h )
        

        # NEW: transient preview overlay for this ROI (None means "show parent slice")
        self._preview_override: _np.ndarray | None = None

        self._pundo: list[tuple[_np.ndarray, dict, str]] = []  # (img, meta, name)
        self._predo: list[tuple[_np.ndarray, dict, str]] = []  # (img, meta, name)

    @property
    def image(self):
        p = self._parent_doc
        if p is None or getattr(p, "image", None) is None:
            return None
        x, y, w, h = self._roi
        # If a preview override exists, show it; else show the live parent slice
        return self._preview_override if self._preview_override is not None else p.image[y:y+h, x:x+w]

    @image.setter
    def image(self, _val):
        # ignore: writes should use DocManager(update/commit) paths
        pass

    # --- helper to snapshot what's currently visible in the Preview
    def _current_preview_copy(self) -> _np.ndarray:
        img = self.image  # property: returns override or parent slice
        if img is None:
            return _np.zeros((1,1), dtype=_np.float32)
        return _np.asarray(img, dtype=_np.float32).copy()

    # === KEEP YOUR WORKING BODY; only 3 added lines are marked "NEW" ===
    def apply_edit(self, new_image, metadata=None, step_name="Edit"):

        x, y, w, h = self._roi
        img = np.asarray(new_image, dtype=np.float32, copy=False)
        base = self._parent_doc.image
        if base is not None:
            if base.ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
                img = img[..., 0]
            if base.ndim == 3 and img.ndim == 2:
                img = np.repeat(img[..., None], base.shape[2], axis=2)
        if img.shape[:2] != (h, w):
            raise ValueError(f"Preview edit shape {img.shape[:2]} != ROI {(h, w)}")

        # NEW: push current visible preview to local undo before overriding
        self._pundo.append((self._current_preview_copy(), dict(self.metadata), step_name))
        self._predo.clear()

        self._preview_override = img
        if metadata:
            self.metadata.update(metadata)
        self.metadata.setdefault("step_name", step_name)

        self.changed.emit()  # let listeners know the ROI doc changed

        # repaint-only nudge (unchanged)
        dm = getattr(self, "_doc_manager", None)
        if dm is not None and hasattr(dm, "previewRepaintRequested"):
            try:
                dm.previewRepaintRequested.emit(self._parent_doc, self._roi)
            except Exception:
                pass

        dm = getattr(self, "_doc_manager", None)
        if dm is not None:
            vw = dm._active_view_widget()
            if vw is not None:
                try:
                    if hasattr(vw, "refresh_from_docman") and callable(vw.refresh_from_docman):
                        vw.refresh_from_docman()
                    else:
                        vw._render()
                except Exception:
                    pass


    def _parent(self):
        return getattr(self, "_parent_doc", None)

    def can_undo(self) -> bool:
        return bool(self._pundo)

    def can_redo(self) -> bool:
        return bool(self._predo)

    def last_undo_name(self) -> str | None:
        return self._pundo[-1][2] if self._pundo else None

    def last_redo_name(self) -> str | None:
        return self._predo[-1][2] if self._predo else None

    def undo(self) -> str | None:
        if not self._pundo:
            return None
        # move current → redo; pop undo → current
        curr = self._current_preview_copy()
        self._predo.append((curr, dict(self.metadata), self._pundo[-1][2]))

        prev_img, prev_meta, name = self._pundo.pop()
        self._preview_override = prev_img
        self.metadata = dict(prev_meta)

        try: self.changed.emit()
        except Exception: pass

        dm = getattr(self, "_doc_manager", None)
        if dm is not None and hasattr(dm, "previewRepaintRequested"):
            try: dm.previewRepaintRequested.emit(self._parent_doc, self._roi)
            except Exception: pass
        return name

    def redo(self) -> str | None:
        if not self._predo:
            return None
        # move current → undo; pop redo → current
        curr = self._current_preview_copy()
        self._pundo.append((curr, dict(self.metadata), self._predo[-1][2]))

        nxt_img, nxt_meta, name = self._predo.pop()
        self._preview_override = nxt_img
        self.metadata = dict(nxt_meta)

        try: self.changed.emit()
        except Exception: pass

        dm = getattr(self, "_doc_manager", None)
        if dm is not None and hasattr(dm, "previewRepaintRequested"):
            try: dm.previewRepaintRequested.emit(self._parent_doc, self._roi)
            except Exception: pass
        return name



class LiveViewDocument(QObject):
    """
    Drop-in proxy that mirrors an ImageDocument API but always resolves
    via DocManager + view to the ROI-aware document (if a Preview tab is active).
    Reads: delegate to current resolved doc.
    Writes: use DocManager.update_active_document(...) so ROI is pasted back.
    """
    changed = pyqtSignal()

    def __init__(self, doc_manager: "DocManager", view, base_doc: "ImageDocument"):
        super().__init__(parent=base_doc.parent())
        self._dm = doc_manager
        self._view = view              # ImageSubWindow widget
        self._base = base_doc          # true ImageDocument

        # Bridge base document change signals (ROI wrappers rarely emit)
        try:
            base_doc.changed.connect(self.changed.emit)
        except Exception:
            pass

    # ---- core resolver ----
    def _current(self):
        try:
            d = self._dm.get_document_for_view(self._view)
            return d or self._base
        except Exception:
            return self._base

    # ---- common API surface (reads) ----
    @property
    def image(self):
        d = self._current()
        return getattr(d, "image", None)

    @property
    def metadata(self):
        d = self._current()
        return getattr(d, "metadata", {}) or {}

    def display_name(self):
        d = self._current()
        if hasattr(d, "display_name"):
            try:
                return d.display_name()
            except Exception:
                pass
        return self._base.display_name() if hasattr(self._base, "display_name") else "Untitled"

    # Mask access stays consistent with whichever doc is current
    def get_active_mask(self):
        d = self._current()
        if hasattr(d, "get_active_mask"):
            try:
                return d.get_active_mask()
            except Exception:
                return None
        return None

    @property
    def masks(self):
        d = self._current()
        return getattr(d, "masks", {})

    @property
    def active_mask_id(self):
        d = self._current()
        return getattr(d, "active_mask_id", None)

    # ---- writes route through DocManager so ROI is honored ----
    def apply_edit(self, new_image, metadata=None, step_name="Edit"):
        #print("[LiveViewDocument] apply_edit called, routing via DocManager")
        self._dm.update_active_document(new_image, dict(metadata or {}), step_name)

    # ---- history helpers (optional pass-throughs) ----
    def can_undo(self):
        d = self._current()
        return bool(getattr(d, "can_undo", lambda: False)())

    def can_redo(self):
        d = self._current()
        return bool(getattr(d, "can_redo", lambda: False)())

    def last_undo_name(self):
        d = self._current()
        return getattr(d, "last_undo_name", lambda: None)()

    def last_redo_name(self):
        d = self._current()
        return getattr(d, "last_redo_name", lambda: None)()

    def undo(self):
        d = self._current()
        return getattr(d, "undo", lambda: None)()

    def redo(self):
        d = self._current()
        return getattr(d, "redo", lambda: None)()

    # ---- generic fallback so existing attributes keep working ----
    def __getattr__(self, name):
        # Prefer the current resolved doc, then base_doc
        d = object.__getattribute__(self, "_current")()
        if hasattr(d, name):
            return getattr(d, name)
        return getattr(self._base, name)


class DocManager(QObject):
    documentAdded = pyqtSignal(object)   # ImageDocument
    documentRemoved = pyqtSignal(object) # ImageDocument
    imageRegionUpdated = pyqtSignal(object, object)  # (doc, roi_tuple_or_None)
    previewRepaintRequested = pyqtSignal(object, object)

    def __init__(self, image_manager=None, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self._roi_doc_cache = {} 
        self._docs: list[ImageDocument] = []
        self._active_doc: ImageDocument | None = None
        self._mdi: "QMdiArea | None" = None  # type: ignore
        self.imageRegionUpdated.connect(self._invalidate_roi_cache)

        def _do_preview_repaint(doc, roi):
            vw = self._active_view_widget()
            if vw is not None:
                try:
                    if hasattr(vw, "refresh_from_docman") and callable(vw.refresh_from_docman):
                        vw.refresh_from_docman(doc=doc, roi=roi)
                    else:
                        vw._render()
                except Exception:
                    pass
        self.previewRepaintRequested.connect(_do_preview_repaint)

    def get_document_for_view(self, view):
        """
        Given an ImageSubWindow widget, return either:
        - the full base ImageDocument
        - or a cached ROI-wrapper doc if a Preview/ROI tab is active
        Works with both old (has_active_preview/current_preview_roi) and
        new (_active_roi_tuple) view APIs. Falls back to view.document.
        """
        # 1) Resolve a base document from the view
        base = (
            getattr(view, "base_document", None)
            or getattr(view, "_base_document", None)
            or getattr(view, "document", None)
        )
        if base is None:
            return None

        # 2) Try to discover an ROI (support both APIs)
        roi = None
        try:
            if hasattr(view, "has_active_preview") and callable(view.has_active_preview):
                if view.has_active_preview():
                    # preferred old API
                    try:
                        roi = view.current_preview_roi()  # (x,y,w,h)
                    except Exception:
                        roi = None
        except Exception:
            pass

        if roi is None:
            # new API candidate
            for attr in ("_active_roi_tuple", "current_roi_tuple", "selected_roi", "roi"):
                try:
                    fn = getattr(view, attr, None)
                    if callable(fn):
                        r = fn()
                        if r and len(r) == 4:
                            roi = r
                            break
                except Exception:
                    pass

        # 3) If no ROI, return the base doc
        if not roi:
            return base

        # 4) Cache and return a lightweight ROI view doc
        try:
            x, y, w, h = map(int, roi)
            key = (id(base), id(view), (x, y, w, h))
            roi_doc = self._roi_doc_cache.get(key)
            if roi_doc is None:
                roi_doc = self._build_roi_document(base, (x, y, w, h))
                self._roi_doc_cache[key] = roi_doc
            return roi_doc
        except Exception:
            # If anything about ROI construction fails, fall back
            return base

    def _invalidate_roi_cache(self, parent_doc, roi_tuple):
        """Drop cached ROI docs that overlap an updated region of parent_doc."""
        if not roi_tuple:
            # full-image change -> drop all for this parent
            dead = [k for k in self._roi_doc_cache.keys() if k[0] == id(parent_doc)]
        else:
            px, py, pw, ph = roi_tuple
            def _overlaps(a, b):
                ax, ay, aw, ah = a; bx, by, bw, bh = b
                return not (ax+aw <= bx or bx+bw <= ax or ay+ah <= by or by+bh <= ay)
            dead = []
            for (parent_id, _view_id, aroi), _doc in list(self._roi_doc_cache.items()):
                if parent_id != id(parent_doc):
                    continue
                if _overlaps(aroi, (px, py, pw, ph)):
                    dead.append((parent_id, _view_id, aroi))
        for k in dead:
            self._roi_doc_cache.pop(k, None)


    def _register_doc(self, doc):
        import weakref
        # Only ImageDocument needs the backref; tables can ignore it.
        if hasattr(doc, "image") or hasattr(doc, "apply_edit"):
            try:
                doc._doc_manager = weakref.proxy(self)   # avoid cycles
            except Exception:
                doc._doc_manager = self                  # fallback
        self._docs.append(doc)
        self.documentAdded.emit(doc)

    def _build_roi_document(self, base_doc, roi):
        #print("[DocManager] Building ROI view document")
        doc = _RoiViewDocument(base_doc, roi, name_suffix=" (Preview)")
        try:
            import weakref
            doc._doc_manager = weakref.proxy(self)
        except Exception:
            doc._doc_manager = self

        # Repaint the active view on ROI preview changes, but DO NOT invalidate cache.
        try:
            #print("[DocManager] Connecting ROI view document change signal")
            import weakref
            dm_ref = weakref.ref(self)
            roi_tuple = tuple(map(int, roi))

            def _on_roi_changed():
                dm = dm_ref()
                if dm is None:
                    return
                vw = dm._active_view_widget()
                if vw is not None:
                    try:
                        if hasattr(vw, "refresh_from_docman") and callable(vw.refresh_from_docman):
                            vw.refresh_from_docman(doc=doc, roi=roi_tuple)
                        else:
                            vw._render()
                    except Exception:
                        pass

            doc.changed.connect(_on_roi_changed)
            #print("[DocManager] ROI view document change signal connected")
        except Exception:
            print("[DocManager] Failed to connect ROI view document change signal")
            pass

        return doc


    def wrap_document_for_view(self, view, base_doc: ImageDocument) -> LiveViewDocument:
        """Return a live, ROI-aware proxy for this view."""
        return LiveViewDocument(self, view, base_doc)

    def open_path(self, path: str):
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        norm_ext = _normalize_ext(ext)

        lower_path = path.lower()
        is_fits = lower_path.endswith((".fit", ".fits", ".fit.gz", ".fits.gz", ".fz"))
        is_xisf = (norm_ext == "xisf")

        primary_doc = None
        created_any = False

        # ---------- 1) Try the universal loader first (ALL formats) ----------
        img = header = bit_depth = is_mono = None
        try:
            img, header, bit_depth, is_mono = legacy_load_image(path)
        except Exception as e:
            print(f"[DocManager] legacy_load_image failed (non-fatal if FITS/XISF): {e}")
        maybe_warn_raw_preview(path, header)
        if img is not None:
            meta = {
                "file_path": path,
                "original_header": header,
                "bit_depth": bit_depth,
                "is_mono": is_mono,
                "original_format": norm_ext,
            }
            _snapshot_header_for_metadata(meta)
            primary_doc = ImageDocument(img, meta)
            self._register_doc(primary_doc)
            created_any = True


        # ---------- 2) FITS: enumerate HDUs (tables + extra images + ICC) ----------
        if is_fits:
            try:
                with fits.open(path, memmap=True) as hdul:
                    base = os.path.basename(path)


                    for i, hdu in enumerate(hdul):
                        name_up = (getattr(hdu, "name", "PRIMARY") or "PRIMARY").upper()
                        if primary_doc is not None and (i == 0 or name_up == "PRIMARY"):

                            continue

                        ext_hdr = hdu.header
                        try:
                            en = str(ext_hdr.get("EXTNAME", "")).strip()
                            ev = ext_hdr.get("EXTVER", None)
                            extname = f"{en}[{int(ev)}]" if (en and isinstance(ev, (int, np.integer))) else (en or "")
                        except Exception:
                            extname = ""

                        # --- Tables → TableDocument ---
                        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                            key_str = extname or f"HDU{i}"
                            nice = key_str
                            #print(f"[DocManager] HDU {i}: {type(hdu).__name__} '{nice}' → Table")

                            # Optional CSV export
                            csv_name = f"{os.path.splitext(path)[0]}_{key_str}.csv".replace(" ", "_")
                            try:
                                _ = _fits_table_to_csv(hdu, csv_name)
                            except Exception as e_csv:
                                print(f"[DocManager] Table CSV export failed ({nice}): {e_csv}")
                                csv_name = None

                            # Build in-app table
                            try:
                                rows, headers = _fits_table_to_rows_headers(hdu, max_rows=500000)
                                tmeta = {
                                    "file_path": f"{path}::{key_str}",
                                    "original_header": ext_hdr,
                                    "original_format": "fits",
                                    "display_name": f"{base} {key_str} (Table)",
                                    "doc_type": "table",
                                    "table_csv": csv_name if (csv_name and os.path.exists(csv_name)) else None,
                                }
                                _snapshot_header_for_metadata(tmeta)
                                tdoc = TableDocument(rows, headers, tmeta, parent=self.parent())
                                self._register_doc(tdoc)
                                try: tdoc.changed.emit()
                                except Exception: pass
                                created_any = True
                                #print(f"[DocManager] Added TableDocument: rows={len(rows)} cols={len(headers)} title='{tdoc.display_name()}'")
                            except Exception as e_tab:
                                print(f"[DocManager] Table HDU {nice} → in-app view failed: {e_tab}")
                            continue  # IMPORTANT: don’t treat a table as an image

                        # --- Not a table: ICC or image ---
                        if hdu.data is None:
                            #print(f"[DocManager] HDU {i} '{extname or f'HDU{i}'}' has no data — noted as aux")
                            continue

                        arr = np.asanyarray(hdu.data)
                        en_up = (extname or "").upper()
                        is_probable_icc = ("ICC" in en_up or "PROFILE" in en_up)

                        # ICC ONLY if name suggests ICC/profile AND data is 1-D uint8
                        if arr.ndim == 1 and arr.dtype == np.uint8 and is_probable_icc:
                            try:
                                icc_path = f"{os.path.splitext(path)[0]}_{extname or f'HDU{i}'}_.icc".replace(" ", "_")
                                with open(icc_path, "wb") as f:
                                    f.write(arr.tobytes())
                                #print(f"[DocManager] Extracted ICC profile → {icc_path}")
                                created_any = True
                                continue
                            except Exception as e_icc:
                                print(f"[DocManager] ICC export failed: {e_icc} — will try as image")

                        # Otherwise: treat as image doc
                        try:
                            if arr.dtype.kind in "ui":
                                a = arr.astype(np.float32, copy=False) / np.float32(np.iinfo(arr.dtype).max)
                            else:
                                a = arr.astype(np.float32, copy=False)
                            ext_depth = "32-bit floating point"
                            ext_mono = bool(a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1))
                            key_str = extname or f"HDU {i}"
                            disp = f"{base} {key_str}"

                            aux_meta = {
                                "file_path": f"{path}::{key_str}",
                                "original_header": ext_hdr,
                                "bit_depth": ext_depth,
                                "is_mono": bool(ext_mono),
                                "original_format": "fits",
                                "image_meta": {"derived_from": path, "layer": key_str, "readonly": True},
                                "display_name": disp,
                            }
                            _snapshot_header_for_metadata(aux_meta)
                            aux_doc = ImageDocument(a, aux_meta)
                            self._register_doc(aux_doc)
                            try: aux_doc.changed.emit()
                            except Exception: pass
                            created_any = True

                        except Exception as e_img:
                            print(f"[DocManager] FITS HDU {i} image build failed: {e_img}")
            except Exception as _e:
                print(f"[DocManager] FITS HDU enumeration failed: {_e}")

        # ---------- 3) XISF: create primary if needed, then enumerate extras ----------
        if is_xisf:
            try:
                # helpers
                def _bit_depth_from_dtype(dt: np.dtype) -> str:
                    dt = np.dtype(dt)
                    if dt == np.float32: return "32-bit floating point"
                    if dt == np.float64: return "64-bit floating point"
                    if dt == np.uint8:   return "8-bit"
                    if dt == np.uint16:  return "16-bit"
                    if dt == np.uint32:  return "32-bit unsigned"
                    return "32-bit floating point"

                def _to_float32_01(arr: np.ndarray) -> np.ndarray:
                    a = np.asarray(arr)
                    if a.dtype == np.float32:
                        return a
                    if a.dtype.kind in "iu":
                        return (a.astype(np.float32) / np.iinfo(a.dtype).max).clip(0.0, 1.0)
                    return a.astype(np.float32, copy=False)

                xisf = XISFReader(path)
                metas = xisf.get_images_metadata() or []
                base = os.path.basename(path)

                # If legacy did NOT create a primary, build image #0 now
                if primary_doc is None and len(metas) >= 1:
                    try:
                        arr0 = xisf.read_image(0, data_format="channels_last")
                        arr0_f32 = _to_float32_01(arr0)
                        bd0 = _bit_depth_from_dtype(metas[0].get("dtype", arr0.dtype))
                        is_mono0 = (arr0_f32.ndim == 2) or (arr0_f32.ndim == 3 and arr0_f32.shape[2] == 1)

                        # Friendly label for #0
                        label0 = metas[0].get("id") or "Image[0]"
                        md0 = {
                            "file_path": f"{path}::XISF[0]",
                            "original_header": metas[0],  # will be sanitized
                            "bit_depth": bd0,
                            "is_mono": is_mono0,
                            "original_format": "xisf",
                            "image_meta": {"derived_from": path, "layer_index": 0, "readonly": True},
                            "display_name": f"{base} {label0}",
                        }
                        _snapshot_header_for_metadata(md0)
                        primary_doc = ImageDocument(arr0_f32, md0)
                        self._register_doc(primary_doc)
                        try: primary_doc.changed.emit()
                        except Exception: pass
                        created_any = True

                    except Exception as e0:
                        print(f"[DocManager] XISF primary (index 0) open failed: {e0}")

                # Add images 1..N-1 as siblings (even if primary came from legacy)
                for i in range(1, len(metas)):
                    try:
                        m = metas[i]
                        arr = xisf.read_image(i, data_format="channels_last")
                        arr_f32 = _to_float32_01(arr)

                        bd = _bit_depth_from_dtype(m.get("dtype", arr.dtype))
                        is_mono_i = (arr_f32.ndim == 2) or (arr_f32.ndim == 3 and arr_f32.shape[2] == 1)

                        # Friendly label: prefer id, else EXTNAME/EXTVER in FITSKeywords, else index
                        label = m.get("id") or None
                        if not label:
                            try:
                                fk = m.get("FITSKeywords", {})
                                en = (fk.get("EXTNAME") or [{}])[0].get("value", "")
                                ev = (fk.get("EXTVER")  or [{}])[0].get("value", "")
                                if en:
                                    label = f"{en}[{ev}]" if ev else en
                            except Exception:
                                pass
                        if not label:
                            label = f"Image[{i}]"

                        md = {
                            "file_path": f"{path}::XISF[{i}]",
                            "original_header": m,  # snapshot; sanitized below
                            "bit_depth": bd,
                            "is_mono": is_mono_i,
                            "original_format": "xisf",
                            "image_meta": {"derived_from": path, "layer_index": i, "readonly": True},
                            "display_name": f"{base} {label}",
                        }
                        _snapshot_header_for_metadata(md)

                        sib = ImageDocument(arr_f32, md)
                        self._register_doc(sib)
                        try: sib.changed.emit()
                        except Exception: pass
                        created_any = True

                    except Exception as _e:
                        print(f"[DocManager] XISF image {i} skipped: {_e}")
            except Exception as _e:
                print(f"[DocManager] XISF open/enumeration failed: {_e}")

        # ---------- 4) Return sensible doc or raise ----------
        if primary_doc is not None:
            return primary_doc
        if created_any:
            return self._docs[-1]  # e.g., a table-only FITS or extra XISF image

        raise IOError(f"Could not load: {path}")
    
    # --- Subwindow / ROI awareness -------------------------------------
    def _active_subwindow(self):
        """Return the active QMdiSubWindow (if any)."""
        if self._mdi is None:
            return None
        try:
            return self._mdi.activeSubWindow()
        except Exception:
            return None

    def _active_view_widget(self):
        """Return the active view widget (ImageSubWindow or TableSubWindow)."""
        sw = self._active_subwindow()
        if not sw:
            return None
        try:
            return sw.widget()
        except Exception:
            return None

    def _active_preview_roi(self):
        """
        Returns (x,y,w,h) if the active view is an ImageSubWindow with a selected Preview tab.
        Else returns None.
        """
        #print("[DocManager] Checking for active preview ROI")
        vw = self._active_view_widget()
        if vw and hasattr(vw, "has_active_preview") and vw.has_active_preview():
            try:
                return vw.current_preview_roi()
            except Exception:
                return None
        return None

    def get_active_image(self, prefer_preview: bool = True):
        """
        Unified read: returns the ndarray a tool should operate on.
        If a Preview tab is active and prefer_preview=True, return that crop.
        Otherwise return the full document image.
        """
        doc = self.get_active_document()
        if doc is None or doc.image is None:
            return None
        roi = self._active_preview_roi() if prefer_preview else None
        if roi is None:
            return doc.image
        x, y, w, h = roi
        return doc.image[y:y+h, x:x+w]


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
        self._register_doc(doc)
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
        self._register_doc(dup)
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
        self._register_doc(doc)
        return doc

    # (optional alias for old code)
    open_numpy = open_array


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
        if doc is not None and doc not in self._docs:
            return
        # ensure backref for legacy docs
        if doc is not None and not hasattr(doc, "_doc_manager"):
            try:
                import weakref
                doc._doc_manager = weakref.proxy(self)
            except Exception:
                doc._doc_manager = self
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
        Return the active document-like object.
        If a Preview tab is selected on the active ImageSubWindow, return a lightweight
        _RoiViewDocument so tools that READ get the crop transparently.
        Otherwise return the real ImageDocument.
        """
        # Prefer cached (if set and still valid)
        if self._active_doc is not None and self._active_doc in self._docs:
            base_doc = self._active_doc
        else:
            # Ask MDI
            base_doc = None
            try:
                if self._mdi is not None:
                    sw = self._mdi.activeSubWindow()
                    if sw is not None:
                        w = sw.widget()
                        base_doc = getattr(w, "document", None) or getattr(sw, "document", None)
                        if base_doc is not None:
                            self._active_doc = base_doc
            except Exception:
                pass
            if base_doc is None:
                base_doc = self._docs[-1] if self._docs else None

        # If no doc or doc doesn’t have an image (e.g., a TableDocument), just return it.
        if base_doc is None or not isinstance(base_doc, ImageDocument) or base_doc.image is None:
            return base_doc

        # Check whether the active view is on a Preview tab
        vw = self._active_view_widget()  # uses _mdi
        if vw and hasattr(vw, "has_active_preview") and vw.has_active_preview():
            try:
                roi = vw.current_preview_roi()  # (x,y,w,h) in FULL coords
            except Exception:
                roi = None
            if roi:
                try:
                    name_suffix = f" (Preview {vw.current_preview_name() or ''})"
                    doc = self._build_roi_document(base_doc, roi)
                    # optional: update display name suffix if you want it
                    try:
                        doc.metadata["display_name"] = f"{base_doc.display_name()}{name_suffix}"
                    except Exception:
                        pass
                    return doc
                except Exception:
                    # If anything fails, fall back to full doc
                    return base_doc

        # No preview selected → return the real document
        return base_doc


    def update_active_document(self, updated_image, metadata=None, step_name: str = "Edit"):

        view_doc = self.get_active_document()
        if view_doc is None:
            raise RuntimeError("No active document")

        img = np.asarray(updated_image)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        if isinstance(view_doc, _RoiViewDocument):
            #print("[DocManager] update_active_document: updating ROI doc")
            # Update ONLY the preview
            view_doc.apply_edit(img, dict(metadata or {}), step_name)

            # 🔔 Force the active ImageSubWindow to repaint the Preview tab
            vw = self._active_view_widget()
            if vw is not None:
                try:
                    #print("[DocManager] update_active_document: refreshing active view for ROI doc")
                    # Prefer a public slot if you have one; fall back to _render().
                    if hasattr(vw, "refresh_from_docman") and callable(vw.refresh_from_docman):
                        vw.refresh_from_docman(doc=view_doc, roi=getattr(view_doc, "_roi", None))
                        #print("[DocManager] refresh_from_docman called")
                    else:
                        vw._render()  # fires immediately
                        #print("[DocManager] _render() called")
                except Exception:
                    #print("[DocManager] Error occurred while refreshing active view")
                    pass
            return

        # Full image path (unchanged)
        if isinstance(view_doc, ImageDocument):
            view_doc.apply_edit(img, dict(metadata or {}), step_name)
            try:
                self.imageRegionUpdated.emit(view_doc, None)
            except Exception:
                pass
        else:
            raise RuntimeError("Active document is not an image")


    # Back-compat/aliases so tools can call any of these:
    def update_image(self, updated_image, metadata=None, step_name: str = "Edit"):
        self.update_active_document(updated_image, metadata, step_name)

    def set_image(self, img, metadata=None, step_name: str = "Edit"):
        self.update_active_document(img, metadata, step_name)

    def apply_edit_to_active(self, img, step_name: str = "Edit", metadata=None):
        self.update_active_document(img, metadata, step_name)
