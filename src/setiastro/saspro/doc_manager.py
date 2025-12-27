# saspro/doc_manager.py
from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox
import os
import numpy as np
from setiastro.saspro.xisf import XISF as XISFReader
from astropy.io import fits  # local import; optional dep
from setiastro.saspro.legacy.image_manager import load_image as legacy_load_image, save_image as legacy_save_image
from setiastro.saspro.legacy.image_manager import list_fits_extensions, load_fits_extension
import uuid
from setiastro.saspro.legacy.image_manager import attach_wcs_to_metadata  # or wherever you put it
from astropy.wcs import WCS  # only if not already imported in this module
from setiastro.saspro.debug_utils import debug_dump_metadata

# Memory utilities for lazy loading and caching
try:
    from setiastro.saspro.memory_utils import get_thumbnail_cache, LazyImage
except ImportError:
    get_thumbnail_cache = None
    LazyImage = None

from setiastro.saspro.swap_manager import get_swap_manager
from setiastro.saspro.widgets.image_utils import ensure_contiguous
from typing import Any

# --- WCS DEBUGGING ------------------------------------------------------
_DEBUG_WCS = False  # flip to False when you’re done debugging

def _debug_log_wcs_context(context: str, meta_or_hdr):
    """
    Tiny helper to print key WCS bits:
      - NAXIS1/2
      - CRPIX1/2
      - CRVAL1/2
      - CDELT / CD if present
    Works if you pass either a metadata dict or a FITS-like header dict.
    """
    if not _DEBUG_WCS:
        return

    # Try to resolve a header from a metadata dict
    hdr = None
    if isinstance(meta_or_hdr, dict):
        # metadata dict with possible header keys
        hdr = (meta_or_hdr.get("original_header")
               or meta_or_hdr.get("fits_header")
               or meta_or_hdr.get("header"))
        if hdr is None:
            # maybe you passed the header dict directly
            hdr = meta_or_hdr
    else:
        hdr = meta_or_hdr

    if hdr is None:
        print(f"[WCS DEBUG] {context}: no header found")
        return

    # Normalize dict-like header
    if hasattr(hdr, "keys"):  # astropy Header or dict
        try:
            keys = list(hdr.keys())
        except Exception:
            keys = []
    else:
        print(f"[WCS DEBUG] {context}: header is non-mapping type {type(hdr)}")
        return

    def _get(k, default=None):
        try:
            return hdr.get(k, default)
        except Exception:
            try:
                return hdr[k]
            except Exception:
                return default

    naxis1 = _get("NAXIS1")
    naxis2 = _get("NAXIS2")
    crpix1 = _get("CRPIX1")
    crpix2 = _get("CRPIX2")
    crval1 = _get("CRVAL1")
    crval2 = _get("CRVAL2")

    cd11   = _get("CD1_1")
    cd12   = _get("CD1_2")
    cd21   = _get("CD2_1")
    cd22   = _get("CD2_2")
    cdelt1 = _get("CDELT1")
    cdelt2 = _get("CDELT2")

    print(f"[WCS DEBUG] {context}:")
    print(f"  NAXIS1={naxis1}  NAXIS2={naxis2}")
    print(f"  CRPIX1={crpix1}  CRPIX2={crpix2}")
    print(f"  CRVAL1={crval1}  CRVAL2={crval2}")
    if any(v is not None for v in (cd11, cd12, cd21, cd22)):
        print(f"  CD = [[{cd11}, {cd12}], [{cd21}, {cd22}]]")
    if cdelt1 is not None or cdelt2 is not None:
        print(f"  CDELT1={cdelt1}  CDELT2={cdelt2}")
    print("")

_DEBUG_UNDO = False  # set True while chasing the GraXpert crash


def _debug_log_undo(context: str, **info):
    """
    Lightweight logger for undo/redo/update activity.
    Safe: never raises, even if repr() is weird.
    """
    if not _DEBUG_UNDO:
        return
    try:
        bits = []
        for k, v in info.items():
            try:
                s = str(v)
            except Exception:
                try:
                    s = repr(v)
                except Exception:
                    s = f"<unrepr {type(v)}>"
            bits.append(f"{k}={s}")
        print(f"[UNDO DEBUG] {context}: " + ", ".join(bits))
    except Exception as e:
        # Last-resort safety – don't let logging itself kill us
        try:
            print(f"[UNDO DEBUG] {context}: <logging failed: {e}>")
        except Exception:
            pass

from setiastro.saspro.file_utils import _normalize_ext

def _normalize_image_01(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an image to [0,1] in-place-ish:

      1. If min < 0  → shift so min becomes 0.
      2. Then if max > 1 → divide by max.

    NaNs/Infs are ignored when computing min/max.
    Returns float32 array.
    """
    if arr is None:
        return arr

    a = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        # completely bogus; give back zeros
        return np.zeros_like(a, dtype=np.float32)

    # Step 1: shift up if we have negatives
    min_val = a[finite].min()
    if min_val < 0.0:
        a = a - min_val
        finite = np.isfinite(a)

    # Step 2: scale down if we exceed 1
    max_val = a[finite].max()
    if max_val > 1.0 and max_val > 0.0:
        a = a / max_val

    return a

_ALLOWED_DEPTHS = {
    "png":  {"8-bit"},
    "jpg":  {"8-bit"},
    "fits": ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
    "fit":  ["8-bit", "16-bit", "32-bit unsigned", "32-bit floating point"],
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
        # _undo / _redo now store tuples: (swap_id: str, metadata: dict, step_name: str)
        self._undo: list[tuple[str, dict, str]] = []
        self._redo: list[tuple[str, dict, str]] = []        
        self.masks: dict[str, np.ndarray] = {}
        self.active_mask_id: str | None = None
        self.uid = uuid.uuid4().hex  # stable identity for DnD, layers, masks, etc.

        # NEW: operation log — list of simple dicts
        # Each entry: {
        #   "id": str,
        #   "step": str,
        #   "params": dict,
        #   "roi": (x,y,w,h) | None,
        #   "source": "full" | "roi",
        #   "ts": float
        # }
        self._op_log: list[dict] = []
        
        # Track unsaved changes explicitly
        self.dirty: bool = False
        
        # Copy-on-write support: if this document shares image data with another,
        # _cow_source holds reference to the source. On first write (apply_edit),
        # we copy the image data and clear _cow_source.
        self._cow_source: 'ImageDocument | None' = None
    # --- history helpers (NEW) ---
    # --- operation log helpers (NEW) -----------------------------------
    def record_operation(
        self,
        step_name: str,
        params: dict | None = None,
        roi: tuple[int, int, int, int] | None = None,
        source: str = "full",
    ) -> str:
        """
        Append a param-record for this edit. This is *lightweight* metadata
        used for replaying ROI recipes etc; it does NOT affect undo/redo.
        """
        import time as _time
        op_id = uuid.uuid4().hex
        entry = {
            "id": op_id,
            "step": step_name or "Edit",
            "params": _dm_json_sanitize(params or {}),
            "roi": tuple(roi) if roi else None,
            "source": str(source or "full"),
            "ts": float(_time.time()),
        }
        self._op_log.append(entry)
        return op_id

    def get_operation_log(self) -> list[dict]:
        """Return a copy of the operation log (for UI / replay)."""
        return list(self._op_log)

    def clear_operation_log(self):
        """Clear the operation log (does not touch pixel history)."""
        self._op_log.clear()


    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def last_undo_name(self) -> str | None:
        return self._undo[-1][2] if self._undo else None

    def last_redo_name(self) -> str | None:
        return self._redo[-1][2] if self._redo else None


    def add_mask(self, mask: Any, mask_id: str | None = None, make_active: bool = True) -> str:
        """
        Store a mask on this document.

        - `mask` can be a numpy array or any mask-like object.
        - If `mask_id` is None, a random UUID is generated.
        - Returns the mask_id used.
        """
        if mask_id is None:
            mask_id = getattr(mask, "id", None) or uuid.uuid4().hex

        # If it's an array, normalize to float32; otherwise just store as-is.
        try:
            arr = np.asarray(mask, dtype=np.float32)
            self.masks[mask_id] = arr
        except Exception:
            self.masks[mask_id] = mask

        if make_active:
            self.active_mask_id = mask_id

        return mask_id

    def remove_mask(self, mask_id: str):
        self.masks.pop(mask_id, None)
        if self.active_mask_id == mask_id:
            self.active_mask_id = None

    def get_active_mask(self):
        return self.masks.get(self.active_mask_id) if self.active_mask_id else None

    def close(self):
        """
        Explicit cleanup of swap files.
        """
        sm = get_swap_manager()
        # Clean up undo stack
        for swap_id, _, _ in self._undo:
            sm.delete_state(swap_id)
        self._undo.clear()
        
        # Clean up redo stack
        for swap_id, _, _ in self._redo:
            sm.delete_state(swap_id)
        self._redo.clear()

    def __del__(self):
        # Fallback cleanup if close() wasn't called (though explicit close is better)
        try:
            self.close()
        except Exception:
            pass


    # in class ImageDocument
    def apply_edit(self, new_image: np.ndarray, metadata: dict | None = None, step_name: str = "Edit"):
        """
        Smart edit:
        - If this is an ROI view (has _roi_info), paste back into parent and emit region update.
        - Else: push history on self and emit full-image update.
        - IMPORTANT: merge metadata without nuking FITS/WCS headers.
        """
        import numpy as np

        def _merge_meta(old_meta: dict | None, new_meta: dict | None, step_name: str):
            """
            Merge new_meta into old_meta but preserve critical header fields
            unless they are explicitly overridden with non-None values.
            """
            old = dict(old_meta or {})
            incoming = dict(new_meta or {})

            critical_keys = (
                "original_header",
                "fits_header",
                "wcs_header",
                "file_meta",
                "image_meta",
            )

            # Preserve critical keys unless caller *deliberately* overrides
            for k in critical_keys:
                if k in incoming:
                    if incoming[k] is not None:
                        old[k] = incoming[k]
                # if not in incoming → leave old value alone

            # Merge all remaining keys normally
            for k, v in incoming.items():
                if k in critical_keys:
                    continue
                old[k] = v

            if step_name:
                old.setdefault("step_name", step_name)
            return old

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
                    parent.metadata = _merge_meta(parent.metadata, metadata, step_name)
                else:
                    parent.metadata.setdefault("step_name", step_name)
                
                sm = get_swap_manager()
                sid = sm.save_state(parent.image)
                if sid:
                    parent._undo.append((sid, parent.metadata.copy(), step_name))
                
                for old_sid, _, _ in parent._redo:
                    sm.delete_state(old_sid)
                parent._redo.clear()
                
                parent.image = new_full
                parent.dirty = True
                parent.changed.emit()

                dm = getattr(self, "_doc_manager", None) or getattr(parent, "_doc_manager", None)
                try:
                    if dm is not None and hasattr(dm, "imageRegionUpdated"):
                        dm.imageRegionUpdated.emit(parent, (x, y, w, h))
                except Exception:
                    print(f"[DocManager] Failed to emit imageRegionUpdated for ROI.")
                return  # done

        # ------ Normal (full-image) branch ------

        # Copy-on-write
        if self._cow_source is not None and self.image is not None:
            self.image = self.image.copy()
            self._cow_source = None
        
        if self.image is not None:
            # snapshot current image + metadata for undo
            try:
                curr = np.asarray(self.image, dtype=np.float32)
                curr = ensure_contiguous(curr)
                
                sm = get_swap_manager()
                sid = sm.save_state(curr)
                
                _debug_log_undo(
                    "ImageDocument.apply_edit.snapshot",
                    doc_id=id(self),
                    name=getattr(self, "display_name", lambda: "<no-name>")(),
                    curr_shape=getattr(curr, "shape", None),
                    undo_len_before=len(self._undo),
                    redo_len_before=len(self._redo),
                    step_name=step_name,
                    swap_id=sid
                )
                if sid:
                    self._undo.append((sid, self.metadata.copy(), step_name))
            except Exception as e:
                print(f"[ImageDocument] apply_edit: failed to snapshot current image for undo: {e}")
            
            # Clear redo stack and delete files
            sm = get_swap_manager()
            for old_sid, _, _ in self._redo:
                sm.delete_state(old_sid)
            self._redo.clear()

        # --- header-safe metadata merge ---
        if metadata:
            self.metadata = _merge_meta(self.metadata, metadata, step_name)
        else:
            self.metadata.setdefault("step_name", step_name)

        # normalize new image
        img = np.asarray(new_image, dtype=np.float32)
        if img.size == 0:
            raise ValueError("apply_edit: new image is empty")

        img = ensure_contiguous(img)

        _debug_log_undo(
            "ImageDocument.apply_edit.apply",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            new_shape=getattr(img, "shape", None),
            undo_len_after=len(self._undo),
            redo_len_after=len(self._redo),
            step_name=step_name,
        )

        self.image = img
        self.dirty = True
        self.changed.emit()

        dm = getattr(self, "_doc_manager", None)
        try:
            if dm is not None and hasattr(dm, "imageRegionUpdated"):
                dm.imageRegionUpdated.emit(self, None)
        except Exception:
            pass



    def undo(self) -> str | None:
        # Extra-safe: if stack is empty, bail early.
        if not self._undo:
            _debug_log_undo(
                "ImageDocument.undo.empty_stack",
                doc_id=id(self),
                name=getattr(self, "display_name", lambda: "<no-name>")(),
            )
            return None

        _debug_log_undo(
            "ImageDocument.undo.entry",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            undo_len=len(self._undo),
            redo_len=len(self._redo),
            top_step=self._undo[-1][2] if self._undo else None,
        )

        # Pop with an extra guard in case something cleared _undo between
        # the check above and this call (re-entrancy / threading).
        try:
            prev_sid, prev_meta, name = self._undo.pop()
        except IndexError:
            _debug_log_undo(
                "ImageDocument.undo.pop_index_error",
                doc_id=id(self),
                name=getattr(self, "display_name", lambda: "<no-name>")(),
                undo_len=len(self._undo),
                redo_len=len(self._redo),
            )
            return None

        # Load previous image from swap
        sm = get_swap_manager()
        prev_img = sm.load_state(prev_sid)
        
        # We can delete the swap file now that we have it in RAM
        # (unless we want to keep it for some reason, but standard undo consumes the state)
        sm.delete_state(prev_sid)

        if prev_img is None:
             print(f"[ImageDocument] undo: failed to load swap state {prev_sid}")
             return None

        # Normalize previous image before using it
        try:
            prev_arr = np.asarray(prev_img, dtype=np.float32)
            if prev_arr.size == 0:
                raise ValueError("undo: previous image is empty")
            prev_arr = np.ascontiguousarray(prev_arr)
        except Exception as e:
            print(f"[ImageDocument] undo: invalid prev_img in stack ({type(prev_img)}): {e}")
            # Put it back so we don't corrupt history further? 
            # Actually if load failed we are in trouble.
            return None

        _debug_log_undo(
            "ImageDocument.undo.normalized_prev",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            prev_shape=getattr(prev_arr, "shape", None),
            prev_dtype=getattr(prev_arr, "dtype", None),
            step_name=name,
            meta_step=prev_meta.get("step_name", None) if isinstance(prev_meta, dict) else None,
        )

        # Snapshot current state for redo (best-effort)
        curr_img = self.image
        curr_meta = self.metadata
        try:
            if curr_img is not None:
                curr_arr = np.asarray(curr_img, dtype=np.float32)
                curr_arr = np.ascontiguousarray(curr_arr)
                
                # Save to swap for Redo
                sid = sm.save_state(curr_arr)
                if sid:
                    self._redo.append((sid, dict(curr_meta), name))
            else:
                # Handle None image? Should not happen usually
                pass
        except Exception as e:
            print(f"[ImageDocument] undo: failed to snapshot current image for redo: {e}")

        _debug_log_undo(
            "ImageDocument.undo.before_apply",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            curr_shape=getattr(curr_img, "shape", None) if curr_img is not None else None,
            curr_dtype=getattr(curr_img, "dtype", None) if curr_img is not None else None,
        )

        self.image = prev_arr
        self.metadata = dict(prev_meta or {})
        self.dirty = True
        try:
            self.changed.emit()
        except Exception:
            pass

        _debug_log_undo(
            "ImageDocument.undo.after_apply",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            new_shape=getattr(self.image, "shape", None),
            new_dtype=getattr(self.image, "dtype", None),
            undo_len=len(self._undo),
            redo_len=len(self._redo),
        )
        return name


    def redo(self) -> str | None:
        if not self._redo:
            return None

        _debug_log_undo(
            "ImageDocument.redo.entry",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            redo_len=len(self._redo),
            undo_len=len(self._undo),
            top_step=self._redo[-1][2] if self._redo else None,
        )

        nxt_sid, nxt_meta, name = self._redo.pop()

        # Load next image from swap
        sm = get_swap_manager()
        nxt_img = sm.load_state(nxt_sid)
        sm.delete_state(nxt_sid)

        if nxt_img is None:
            print(f"[ImageDocument] redo: failed to load swap state {nxt_sid}")
            return None

        # Normalize next image before using it
        try:
            nxt_arr = np.asarray(nxt_img, dtype=np.float32)
            if nxt_arr.size == 0:
                raise ValueError("redo: next image is empty")
            nxt_arr = np.ascontiguousarray(nxt_arr)
        except Exception as e:
            print(f"[ImageDocument] redo: invalid nxt_img in stack ({type(nxt_img)}): {e}")
            return None
            
        _debug_log_undo(
            "ImageDocument.redo.normalized_next",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            nxt_shape=getattr(nxt_arr, "shape", None),
            nxt_dtype=getattr(nxt_arr, "dtype", None),
            step_name=name,
            meta_step=nxt_meta.get("step_name", None) if isinstance(nxt_meta, dict) else None,
        )
        curr_img = self.image
        curr_meta = self.metadata
        try:
            if curr_img is not None:
                curr_arr = np.asarray(curr_img, dtype=np.float32)
                curr_arr = np.ascontiguousarray(curr_arr)
                
                # Save current to swap for Undo
                sid = sm.save_state(curr_arr)
                if sid:
                    self._undo.append((sid, dict(curr_meta), name))
            else:
                pass
        except Exception as e:
            print(f"[ImageDocument] redo: failed to snapshot current image for undo: {e}")
        _debug_log_undo(
            "ImageDocument.redo.before_apply",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            curr_shape=getattr(curr_img, "shape", None) if curr_img is not None else None,
            curr_dtype=getattr(curr_img, "dtype", None) if curr_img is not None else None,
        )
        self.image = nxt_arr
        self.metadata = dict(nxt_meta or {})
        self.dirty = True
        try:
            self.changed.emit()
        except Exception:
            pass

        _debug_log_undo(
            "ImageDocument.redo.after_apply",
            doc_id=id(self),
            name=getattr(self, "display_name", lambda: "<no-name>")(),
            new_shape=getattr(self.image, "shape", None),
            new_dtype=getattr(self.image, "dtype", None),
            undo_len=len(self._undo),
            redo_len=len(self._redo),
        )

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
        Returns [(swap_id, meta, name), ...]
        """
        out = []
        for sid, meta, name in self._undo:
            out.append((sid, meta or {}, name or "Unnamed"))
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

def _compute_cropped_wcs(parent_hdr_like: dict | "fits.Header",
                         x: int, y: int, w: int, h: int):
    """
    Returns a plain dict WCS header reflecting a pure pixel crop by (x,y,w,h).
    Keeps CD/CDELT/PC/CRVAL as-is and shifts CRPIX by (+/-) the crop offset.
    Also sets NAXIS1/2 to (w,h) and records custom CROPX/CROPY.
    """
    try:
        from astropy.io.fits import Header  # type: ignore
    except Exception:
        Header = None  # type: ignore

    # Normalize to a dict of key->value (no comments needed for the drag payload)
    if Header is not None and isinstance(parent_hdr_like, Header):
        base = {k: parent_hdr_like.get(k) for k in parent_hdr_like.keys()}
    elif isinstance(parent_hdr_like, dict):
        # If it’s an XISF-like dict, try to pull a FITSKeywords block first
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

    # Shift CRPIX by the crop offset (ROI origin is (x,y) in full-image pixels)
    crpix1 = base.get("CRPIX1")
    crpix2 = base.get("CRPIX2")
    if isinstance(crpix1, (int, float)) and isinstance(crpix2, (int, float)):
        new_crpix1 = float(crpix1) - float(x)
        new_crpix2 = float(crpix2) - float(y)
        base["CRPIX1"] = new_crpix1
        base["CRPIX2"] = new_crpix2
    else:
        new_crpix1 = crpix1
        new_crpix2 = crpix2

    # Update image size keys
    base["NAXIS1"] = int(w)
    base["NAXIS2"] = int(h)

    # Optional helpful tags
    base["CROPX"] = int(x)
    base["CROPY"] = int(y)
    base["SASKIND"] = "ROI-CROP"

    # DEBUG: show how CRPIX changed for this crop
    if _DEBUG_WCS:
        print(f"[WCS DEBUG] _compute_cropped_wcs: roi=({x},{y},{w},{h})")
        print(f"  CRPIX1: {crpix1} -> {new_crpix1}")
        print(f"  CRPIX2: {crpix2} -> {new_crpix2}")
        print("")

    return base

import logging

log = logging.getLogger(__name__)

def _pick_header_for_save(meta: dict) -> fits.Header | None:
    """
    Choose the best header to write to disk.

    Priority:
        1. 'wcs_header'      – if you stash a solved header here
        2. 'fits_header'     – common name after ASTAP / plate solve
        3. 'original_header' – whatever came from disk
        4. 'header'          – older code paths
    """
    if not isinstance(meta, dict):
        return None

    for key in ("wcs_header", "fits_header", "original_header", "header"):
        hdr = meta.get(key)
        if isinstance(hdr, fits.Header):
            log.debug("[_pick_header_for_save] using %s (%d cards)", key, len(hdr))
            return hdr

    log.debug("[_pick_header_for_save] no fits.Header found in metadata; "
              "will let legacy_save_image fall back.")
    return None

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
        self._roi_info = {"parent_doc": parent_doc, "roi": tuple(self._roi)}
        self.metadata["_roi_bounds"] = tuple(self._roi)
        imi = dict(self.metadata.get("image_meta") or {})
        imi.update({"roi": tuple(self._roi), "view_kind": "roi-preview"})
        self.metadata["image_meta"] = imi

        # build and store an ROI-shifted WCS header snapshot to use if detached
        try:
            phdr = (parent_doc.metadata.get("original_header")
                    or parent_doc.metadata.get("fits_header")
                    or parent_doc.metadata.get("header"))
            rx, ry, rw, rh = self._roi
            roi_wcs = _compute_cropped_wcs(phdr, rx, ry, rw, rh)
            self.metadata["roi_wcs_header"] = roi_wcs  # plain dict, drop-in safe

            # 🔴 KEY FIX: for a standalone ROI doc, treat this cropped WCS
            # as the "original_header" so view-drops / duplicates inherit it.
            if phdr is not None:
                # optional: preserve the full parent header
                self.metadata.setdefault("parent_full_header", phdr)
            self.metadata["original_header"] = roi_wcs
            try:
                from .doc_manager import _snapshot_header_for_metadata  # if you move it, adjust import
            except Exception:
                _snapshot_header_for_metadata = None

            try:
                if _snapshot_header_for_metadata is not None:
                    _snapshot_header_for_metadata(self.metadata)
            except Exception:
                pass

            # DEBUG: log parent vs ROI WCS
            if _DEBUG_WCS:
                base_name = parent_doc.display_name() if hasattr(parent_doc, "display_name") else "<parent>"
                print(f"[WCS DEBUG] _RoiViewDocument.__init__: parent='{base_name}' roi={self._roi}")
                _debug_log_wcs_context("  parent_header", phdr)
                _debug_log_wcs_context("  roi_header", self.metadata)
        except Exception as e:
            if _DEBUG_WCS:
                print(f"[WCS DEBUG] _RoiViewDocument.__init__ exception: {e}")
            pass
        self.metadata["image_meta"] = imi
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


    def commit_to_parent(self, new_image: _np.ndarray | None = None,
                        metadata: dict | None = None, step_name: str = "Edit"):
        """
        Paste current preview (or provided new_image) back into the parent image
        with proper undo and region repaint.
        """
        parent = getattr(self, "_parent_doc", None)
        if parent is None or parent.image is None:
            return

        x, y, w, h = self._roi
        # choose source
        src = new_image
        if src is None:
            src = self._preview_override if self._preview_override is not None else parent.image[y:y+h, x:x+w]

        img = _np.asarray(src, dtype=_np.float32, copy=False)
        base = parent.image

        # channel reconciliation
        if base.ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]
        if base.ndim == 3 and img.ndim == 2:
            img = _np.repeat(img[..., None], base.shape[2], axis=2)
        if img.shape[:2] != (h, w):
            raise ValueError(f"Commit shape {img.shape[:2]} does not match ROI {(h, w)}")

        # push undo on parent and paste
        parent._undo.append((base.copy(), parent.metadata.copy(), step_name))
        parent._redo.clear()
        if metadata: parent.metadata.update(metadata)
        parent.metadata.setdefault("step_name", step_name)

        new_full = base.copy()
        new_full[y:y+h, x:x+w] = img
        parent.image = new_full
        try: parent.changed.emit()
        except Exception as e:
            import logging
            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")

        # notify region update + repaint
        dm = getattr(self, "_doc_manager", None) or getattr(parent, "_doc_manager", None)
        if dm is not None:
            try: dm.imageRegionUpdated.emit(parent, (x, y, w, h))
            except Exception as e:
                import logging
                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")


    # --- helper to snapshot what's currently visible in the Preview
    def _current_preview_copy(self) -> _np.ndarray:
        img = self.image  # property: returns override or parent slice
        if img is None:
            return _np.zeros((1, 1), dtype=_np.float32)
        arr = _np.asarray(img, dtype=_np.float32)
        return _np.ascontiguousarray(arr)

    # === KEEP YOUR WORKING BODY; only 3 added lines are marked "NEW" ===
    def apply_edit(self, new_image, metadata=None, step_name="Edit"):
        x, y, w, h = self._roi
        img = np.asarray(new_image, dtype=np.float32, copy=False)
        base = self._parent_doc.image

        _debug_log_undo(
            "_RoiViewDocument.apply_edit.entry",
            roi=(x, y, w, h),
            parent_id=id(self._parent_doc) if self._parent_doc is not None else None,
            roi_doc_id=id(self),
            new_shape=getattr(img, "shape", None),
            step_name=step_name,
            pundo_len=len(self._pundo),
            predo_len=len(self._predo),
        )
        if base is not None:
            if base.ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
                img = img[..., 0]
            if base.ndim == 3 and img.ndim == 2:
                img = np.repeat(img[..., None], base.shape[2], axis=2)
        if img.shape[:2] != (h, w):
            raise ValueError(f"Preview edit shape {img.shape[:2]} != ROI {(h, w)}")
        
        img = np.ascontiguousarray(img) 

        # snapshot current visible preview for local undo
        self._pundo.append((self._current_preview_copy(), dict(self.metadata), step_name))
        self._predo.clear()

        self._preview_override = img
        _debug_log_undo(
            "_RoiViewDocument.apply_edit.after",
            roi=(x, y, w, h),
            preview_shape=getattr(self._preview_override, "shape", None),
            pundo_len=len(self._pundo),
            predo_len=len(self._predo),
            step_name=step_name,
        )

        if metadata:
            self.metadata.update(metadata)
        self.metadata.setdefault("step_name", step_name)

        # 1) notify ROI listeners (e.g. the main window via _on_roi_changed)
        try:
            self.changed.emit()
        except Exception:
            pass

        # 2) optionally: tell DocManager "ROI preview changed" using base doc + ROI
        dm = getattr(self, "_doc_manager", None)
        if dm is not None and hasattr(dm, "previewRepaintRequested"):
            try:
                dm.previewRepaintRequested.emit(self._parent_doc, self._roi)
            except Exception:
                pass



    def _parent(self):
        return getattr(self, "_parent_doc", None)

    def can_undo(self) -> bool:
        # Prefer local preview history if present
        if self._pundo:
            return True
        # Otherwise mirror parent’s history
        p = getattr(self, "_parent_doc", None)
        if p is not None and hasattr(p, "can_undo"):
            try:
                return bool(p.can_undo())
            except Exception:
                return False
        return False

    def can_redo(self) -> bool:
        if self._predo:
            return True
        p = getattr(self, "_parent_doc", None)
        if p is not None and hasattr(p, "can_redo"):
            try:
                return bool(p.can_redo())
            except Exception:
                return False
        return False

    def last_undo_name(self) -> str | None:
        if self._pundo:
            return self._pundo[-1][2]
        p = getattr(self, "_parent_doc", None)
        if p is not None and hasattr(p, "last_undo_name"):
            try:
                return p.last_undo_name()
            except Exception:
                return None
        return None

    def last_redo_name(self) -> str | None:
        if self._predo:
            return self._predo[-1][2]
        p = getattr(self, "_parent_doc", None)
        if p is not None and hasattr(p, "last_redo_name"):
            try:
                return p.last_redo_name()
            except Exception:
                return None
        return None

    def undo(self) -> str | None:
        # --- Case 1: ROI-local preview history ---
        if self._pundo:
            _debug_log_undo(
                "_RoiViewDocument.undo.local.entry",
                roi=self._roi,
                roi_doc_id=id(self),
                pundo_len=len(self._pundo),
                predo_len=len(self._predo),
            )
            # move current → redo; pop undo → current
            curr = self._current_preview_copy()
            self._predo.append((curr, dict(self.metadata), self._pundo[-1][2]))

            prev_img, prev_meta, name = self._pundo.pop()
            self._preview_override = prev_img
            self.metadata = dict(prev_meta)
            _debug_log_undo(
                "_RoiViewDocument.undo.local.apply",
                roi=self._roi,
                new_preview_shape=getattr(prev_img, "shape", None),
                pundo_len=len(self._pundo),
                predo_len=len(self._predo),
                name=name,
            )
            try:
                self.changed.emit()
            except Exception:
                pass

            dm = getattr(self, "_doc_manager", None)
            if dm is not None and hasattr(dm, "previewRepaintRequested"):
                try:
                    dm.previewRepaintRequested.emit(self._parent_doc, self._roi)
                except Exception:
                    pass
            return name

        # --- Case 2: no ROI-local history → delegate to parent ---
        parent = getattr(self, "_parent_doc", None)
        if parent is None or not hasattr(parent, "undo"):
            return None
        _debug_log_undo(
            "_RoiViewDocument.undo.parent.entry",
            roi=self._roi,
            roi_doc_id=id(self),
            parent_id=id(parent),
            parent_undo_len=len(getattr(parent, "_undo", [])),
            parent_redo_len=len(getattr(parent, "_redo", [])),
        )

        name = parent.undo()

        # After parent changes, clear override so we show the new parent slice
        self._preview_override = None

        try:
            self.changed.emit()
        except Exception:
            pass

        dm = getattr(self, "_doc_manager", None) or getattr(parent, "_doc_manager", None)
        if dm is not None and hasattr(dm, "previewRepaintRequested"):
            try:
                dm.previewRepaintRequested.emit(parent, self._roi)
            except Exception:
                pass
        return name

    def redo(self) -> str | None:
        # --- Case 1: ROI-local preview history ---
        if self._predo:
            # move current → undo; pop redo → current
            curr = self._current_preview_copy()
            self._pundo.append((curr, dict(self.metadata), self._predo[-1][2]))

            nxt_img, nxt_meta, name = self._predo.pop()
            self._preview_override = nxt_img
            self.metadata = dict(nxt_meta)

            try:
                self.changed.emit()
            except Exception:
                pass

            dm = getattr(self, "_doc_manager", None)
            if dm is not None and hasattr(dm, "previewRepaintRequested"):
                try:
                    dm.previewRepaintRequested.emit(self._parent_doc, self._roi)
                except Exception:
                    pass
            return name

        # --- Case 2: delegate to parent’s redo ---
        parent = getattr(self, "_parent_doc", None)
        if parent is None or not hasattr(parent, "redo"):
            return None

        name = parent.redo()

        # Parent changed → reset override and repaint
        self._preview_override = None

        try:
            self.changed.emit()
        except Exception:
            pass

        dm = getattr(self, "_doc_manager", None) or getattr(parent, "_doc_manager", None)
        if dm is not None and hasattr(dm, "previewRepaintRequested"):
            try:
                dm.previewRepaintRequested.emit(parent, self._roi)
            except Exception:
                pass
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
        _debug_log_undo(
            "LiveViewDocument.undo.call",
            live_id=id(self),
            resolved_type=type(d).__name__ if d is not None else None,
            resolved_id=id(d) if d is not None else None,
            is_roi=isinstance(d, _RoiViewDocument),
            has_undo=getattr(d, "can_undo", lambda: False)(),
        )
        return getattr(d, "undo", lambda: None)()

    def redo(self):
        d = self._current()
        _debug_log_undo(
            "LiveViewDocument.redo.call",
            live_id=id(self),
            resolved_type=type(d).__name__ if d is not None else None,
            resolved_id=id(d) if d is not None else None,
            is_roi=isinstance(d, _RoiViewDocument),
            has_redo=getattr(d, "can_redo", lambda: False)(),
        )
        return getattr(d, "redo", lambda: None)()


    # ---- generic fallback so existing attributes keep working ----
    def __getattr__(self, name):
        # Prefer the current resolved doc, then base_doc
        d = object.__getattribute__(self, "_current")()
        if hasattr(d, name):
            return getattr(d, name)
        return getattr(self._base, name)

def _xisf_meta_to_fits_header(m: dict) -> fits.Header | None:
    """
    Best-effort: pull common WCS keys out of XISF FITSKeywords into a fits.Header.
    Returns None if nothing usable found.
    """
    fk = m.get("FITSKeywords", {}) if isinstance(m, dict) else {}
    if not fk:
        return None

    want = (
        "WCSAXES", "CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2",
        "CRVAL1", "CRVAL2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2",
        "A_ORDER", "B_ORDER"
    )

    hdr = fits.Header()
    found = False
    for k in want:
        vlist = fk.get(k)
        if vlist and isinstance(vlist, list) and vlist[0].get("value") is not None:
            hdr[k] = vlist[0]["value"]
            found = True

    # also pull SIP coeffs if present
    for k, vlist in fk.items():
        if k.startswith(("A_", "B_", "AP_", "BP_")) and vlist and vlist[0].get("value") is not None:
            try:
                hdr[k] = float(vlist[0]["value"])
                found = True
            except Exception:
                pass

    return hdr if found else None

DEBUG_SAVE_DOCUMENT = False

def debug_dump_metadata_print(meta: dict, context: str = ""):
    if DEBUG_SAVE_DOCUMENT:
        print(f"\n===== METADATA DUMP ({context}) =====")
        if not isinstance(meta, dict):
            print("  (not a dict) ->", type(meta))
            print("====================================")
            return

        keys = sorted(str(k) for k in meta.keys())
        print("  keys:", ", ".join(keys))

        for key in keys:
            val = meta[key]
            if isinstance(val, fits.Header):
                print(f"  {key}: fits.Header with {len(val.cards)} cards")
            else:
                print(f"  {key}: {val!r} ({type(val).__name__})")

        print("===== END METADATA DUMP ({}) =====".format(context))

class DocManager(QObject):
    documentAdded = pyqtSignal(object)   # ImageDocument
    documentRemoved = pyqtSignal(object) # ImageDocument
    imageRegionUpdated = pyqtSignal(object, object)  # (doc, roi_tuple_or_None)
    previewRepaintRequested = pyqtSignal(object, object)
    
    activeBaseChanged = pyqtSignal(object)  # emits ImageDocument | None

    def __init__(self, image_manager=None, parent=None):
        super().__init__(parent)
        self.image_manager = image_manager
        self._roi_doc_cache = {} 
        self._docs: list[ImageDocument] = []
        self._active_doc: ImageDocument | None = None
        self._mdi: "QMdiArea | None" = None  # type: ignore
        def _on_region_updated(doc, roi):
            vw = self._active_view_widget()
            if vw is not None:
                try:
                    if hasattr(vw, "refresh_from_docman") and callable(vw.refresh_from_docman):
                        vw.refresh_from_docman(doc=doc, roi=roi)
                    else:
                        vw._render()
                except Exception:
                    pass

        self.imageRegionUpdated.connect(_on_region_updated)
        self._by_uid = {}
        self._focused_base_doc: ImageDocument | None = None  # <— NEW

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

    def get_document_by_uid(self, uid: str):
        return self._by_uid.get(uid)


    def _register_doc(self, doc):
        import weakref
        # Only ImageDocument needs the backref; tables can ignore it.
        if hasattr(doc, "image") or hasattr(doc, "apply_edit"):
            try:
                doc._doc_manager = weakref.proxy(self)   # avoid cycles
            except Exception:
                doc._doc_manager = self                  # fallback
            self._docs.append(doc)
            if hasattr(doc, "uid"):
                self._by_uid[doc.uid] = doc
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
                        # IMPORTANT: use the *parent* doc here, not the ROI wrapper
                        base = getattr(doc, "_parent_doc", None) or doc
                        if hasattr(vw, "refresh_from_docman") and callable(vw.refresh_from_docman):
                            vw.refresh_from_docman(doc=base, roi=roi_tuple)
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

    def commit_active_preview_to_parent(self, metadata: dict | None = None, step_name: str = "Edit"):
        doc = self.get_active_document()
        if isinstance(doc, _RoiViewDocument):
            doc.commit_to_parent(None, metadata=metadata or {}, step_name=step_name)
            # after commit, force an immediate view repaint
            vw = self._active_view_widget()
            if vw is not None:
                try:
                    if hasattr(vw, "refresh_from_docman") and callable(vw.refresh_from_docman):
                        vw.refresh_from_docman(doc=doc._parent_doc, roi=None)
                    else:
                        vw._render()
                except Exception:
                    pass

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
        meta = None
        try:
            # NEW: prefer metadata-aware return
            out = legacy_load_image(path, return_metadata=True)
            if out and len(out) == 5:
                img, header, bit_depth, is_mono, meta = out
            else:
                img, header, bit_depth, is_mono = out
        except TypeError:
            # legacy_load_image older signature → fall back
            try:
                img, header, bit_depth, is_mono = legacy_load_image(path)
            except Exception as e:
                print(f"[DocManager] legacy_load_image failed (non-fatal if FITS/XISF): {e}")
        except Exception as e:
            print(f"[DocManager] legacy_load_image failed (non-fatal if FITS/XISF): {e}")

        maybe_warn_raw_preview(path, header)

        if img is not None:
            if meta is None:
                meta = {
                    "file_path": path,
                    "original_header": header,
                    "bit_depth": bit_depth,
                    "is_mono": is_mono,
                    "original_format": norm_ext,
                }

                # NEW: attach WCS even for old loader
                meta = attach_wcs_to_metadata(meta, header)

            _snapshot_header_for_metadata(meta)

            img = _normalize_image_01(img)
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
                                except Exception as e:
                                    import logging
                                    logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
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
                                a = arr.astype(np.float32, copy=False)  # NO normalization
                                # optional: if you want to record original scale:
                                ext_depth = f"{arr.dtype.itemsize*8}-bit {'unsigned' if arr.dtype.kind=='u' else 'signed'}"
                            else:
                                a = arr.astype(np.float32, copy=False)  # floats preserved
                                ext_depth = "32-bit floating point" if arr.dtype == np.float32 else "64-bit floating point"

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

                            # NEW: attach WCS from this HDU header
                            aux_meta = attach_wcs_to_metadata(aux_meta, ext_hdr)

                            _snapshot_header_for_metadata(aux_meta)
                            a = _normalize_image_01(a)
                            aux_doc = ImageDocument(a, aux_meta)

                            self._register_doc(aux_doc)
                            try: aux_doc.changed.emit()
                            except Exception as e:
                                import logging
                                logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
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

                def _to_float32_preserve(arr: np.ndarray) -> np.ndarray:
                    a = np.asarray(arr)
                    return a if a.dtype == np.float32 else a.astype(np.float32, copy=False)

                xisf = XISFReader(path)
                metas = xisf.get_images_metadata() or []
                base = os.path.basename(path)

                # If legacy did NOT create a primary, build image #0 now
                if primary_doc is None and len(metas) >= 1:
                    try:
                        arr0 = xisf.read_image(0, data_format="channels_last")
                        arr0_f32 = _to_float32_preserve(arr0)
                        arr0_f32 = _normalize_image_01(arr0_f32)
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
                        # NEW: attach WCS if possible
                        hdr0 = _xisf_meta_to_fits_header(metas[0])
                        if hdr0 is not None:
                            md0 = attach_wcs_to_metadata(md0, hdr0)

                        _snapshot_header_for_metadata(md0)
                        primary_doc = ImageDocument(arr0_f32, md0)
                        self._register_doc(primary_doc)
                        try: primary_doc.changed.emit()
                        except Exception as e:
                            import logging
                            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
                        created_any = True

                    except Exception as e0:
                        print(f"[DocManager] XISF primary (index 0) open failed: {e0}")

                # Add images 1..N-1 as siblings (even if primary came from legacy)
                for i in range(1, len(metas)):
                    try:
                        m = metas[i]
                        arr = xisf.read_image(i, data_format="channels_last")
                        arr_f32 = _to_float32_preserve(arr)
                        arr_f32 = _normalize_image_01(arr_f32)

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
                        hdri = _xisf_meta_to_fits_header(m)
                        if hdri is not None:
                            md = attach_wcs_to_metadata(md, hdri)

                        _snapshot_header_for_metadata(md)
                        sib = ImageDocument(arr_f32, md)
                        self._register_doc(sib)
                        try: sib.changed.emit()
                        except Exception as e:
                            import logging
                            logging.debug(f"Exception suppressed: {type(e).__name__}: {e}")
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

    def save_document(
        self,
        doc: "ImageDocument",
        path: str,
        bit_depth: str | None = None,
        *,
        bit_depth_override: str | None = None,
    ):
        """
        Save the given ImageDocument to 'path'.

        bit_depth_override:
            New-style explicit choice from a dialog.

        bit_depth:
            Legacy positional argument; still honored if override is None.
        """
        ext = _normalize_ext(os.path.splitext(path)[1])
        img = doc.image
        meta = doc.metadata or {}

        # ── MASSIVE DEBUG: show everything we know coming in ───────────────
        debug_dump_metadata_print(meta, context="save_document: BEFORE HEADER PICK")

        # --- Decide final bit depth ---------------------------------------
        requested = bit_depth_override or bit_depth or meta.get("bit_depth")

        if requested:
            allowed = _ALLOWED_DEPTHS.get(ext, set())
            if allowed and requested not in allowed:
                print(f"[save_document] Requested bit depth {requested!r} "
                      f"not in allowed {allowed}, falling back to first.")
                final_bit_depth = next(iter(allowed))
            else:
                final_bit_depth = requested
        else:
            final_bit_depth = self._infer_bit_depth_for_format(
                img, ext, meta.get("bit_depth")
            )



        # --- Clip if needed for integer encodes ---------------------------
        needs_clip = (
            ext in ("png", "jpg", "jpeg", "tif", "tiff")
            and final_bit_depth in ("8-bit", "16-bit", "32-bit unsigned")
        )
        if needs_clip:
            print("[save_document] Clipping image to [0,1] for integer encode.")
        img_to_save = np.clip(img, 0.0, 1.0) if needs_clip else img

        # --- PICK THE HEADER EXPLICITLY -----------------------------------
        # Priority:
        #   1) wcs_header
        #   2) fits_header
        #   3) original_header
        #   4) header
        effective_header = None
        for key in ("original_header", "fits_header", "wcs_header", "header"):
            val = meta.get(key)
            if isinstance(val, fits.Header):
                effective_header = val

                break

        #if effective_header is None:
        #    print("[save_document] WARNING: No fits.Header in metadata, "
        #          "legacy_save_image will pick a default header.")
        #else:
        #    # Print first few cards so we can confirm we have the SIP stuff
        #    print("[save_document] effective_header preview (first 25 cards):")
        #    for i, card in enumerate(effective_header.cards):
        #        if i >= 25:
        #            print("  ... (truncated)")
        #            break
        #        print(f"  {card.keyword:8s} = {card.value!r}")

        # ── Call the legacy saver ─────────────────────────────────────────


        legacy_save_image(
            img_array=img_to_save,
            filename=path,
            original_format=ext,
            bit_depth=final_bit_depth,
            original_header=effective_header,
            is_mono=meta.get("mono", img.ndim == 2),
            image_meta=meta.get("image_meta"),
            file_meta=meta.get("file_meta"),
            wcs_header=meta.get("wcs_header"),
        )

        # ── Update metadata in memory to match what we just wrote ─────────
        meta["file_path"] = path
        meta["original_format"] = ext
        meta["bit_depth"] = final_bit_depth

        if isinstance(effective_header, fits.Header):
            meta["original_header"] = effective_header

            # If you have this helper, keep it; if not, you can skip it
            try:
                _snapshot_header_for_metadata(meta)
            except Exception as e:
                print("[save_document] _snapshot_header_for_metadata error:", e)

        doc.metadata = meta

        # reset dirty flag
        if hasattr(doc, "dirty"):
            doc.dirty = False

        if hasattr(doc, "changed"):
            doc.changed.emit()

    def duplicate_document(self, source_doc: ImageDocument, new_name: str | None = None) -> ImageDocument:
        # DEBUG: log the source doc WCS before we touch anything
        if _DEBUG_WCS:
            try:
                name = source_doc.display_name() if hasattr(source_doc, "display_name") else "<src>"
            except Exception:
                name = "<src>"
            
            _debug_log_wcs_context("  source.metadata", getattr(source_doc, "metadata", {}))

        # COPY-ON-WRITE: Share the source image instead of copying immediately.
        # The duplicate's apply_edit will copy when it first modifies the image.
        # This saves memory when duplicates are created but not modified.
        img_ref = source_doc.image  # Shared reference, no copy

        meta = dict(source_doc.metadata or {})
        base = source_doc.display_name()
        dup_title = (new_name or f"{base}_duplicate")
        # 🚫 strip any lingering emojis / link markers
        dup_title = dup_title.replace("🔗", "").strip()
        meta["display_name"] = dup_title

        # Remove anything that makes the view look "linked/preview"
        imi = dict(meta.get("image_meta") or {})
        for k in ("readonly", "view_kind", "derived_from", "layer", "layer_index", "linked"):
            imi.pop(k, None)
        meta["image_meta"] = imi
        for k in list(meta.keys()):
            if k.startswith("_roi_") or k.endswith("_roi") or k == "roi":
                meta.pop(k, None)

        # NOTE: we intentionally DO NOT remove "roi_wcs_header" or "original_header"
        # so that a ROI doc keeps its cropped WCS in the duplicate.

        # Safe bit depth / mono flags
        meta.setdefault("original_format", meta.get("original_format", "fits"))
        if isinstance(img_ref, np.ndarray):
            meta["is_mono"] = (img_ref.ndim == 2 or (img_ref.ndim == 3 and img_ref.shape[2] == 1))

        _snapshot_header_for_metadata(meta)

        dup = ImageDocument(img_ref, meta, parent=self.parent())
        # Mark this duplicate as sharing image data with source
        dup._cow_source = source_doc
        self._register_doc(dup)

        # DEBUG: log the duplicate doc WCS
        if _DEBUG_WCS:
            try:
                dname = dup.display_name()
            except Exception:
                dname = "<dup>"

            _debug_log_wcs_context("  duplicate.metadata", dup.metadata)

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
            try:
                if hasattr(doc, "uid"):
                    self._by_uid.pop(doc.uid, None)
            except Exception:
                pass
            
            # Cleanup swap files
            if hasattr(doc, "close"):
                try:
                    doc.close()
                except Exception as e:
                    print(f"[DocManager] Failed to close document {doc}: {e}")
                    
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

    def _base_from_subwindow(self, sw):
        """Best-effort: unwrap to the base ImageDocument bound to a subwindow."""
        if sw is None:
            return None
        try:
            w = sw.widget()
            base = (getattr(w, "base_document", None)
                    or getattr(w, "_base_document", None)
                    or getattr(w, "document", None))
            # unwrap ROI wrappers if any
            p = getattr(base, "_parent_doc", None)
            return p if isinstance(p, ImageDocument) else base
        except Exception:
            return None

    def _on_subwindow_activated(self, sw):
        # existing logic (keep it)
        doc = None
        try:
            if sw is not None:
                w = sw.widget()
                doc = getattr(w, "document", None) or getattr(sw, "document", None)
        except Exception:
            doc = None
        self.set_active_document(doc)

        # NEW: compute focused *base* doc and emit change only when different
        new_base = self._base_from_subwindow(sw)
        if new_base is not self._focused_base_doc:
            self._focused_base_doc = new_base
            try:
                self.activeBaseChanged.emit(new_base)
            except Exception:
                pass

    def get_focused_base_document(self):
        """
        Returns the last *activated* subwindow's base ImageDocument (sticky),
        ignoring hover/preview wrappers.
        """
        return self._focused_base_doc

    def get_active_document(self):
        """
        Return the active document-like object.
        If a Preview tab is selected on the active ImageSubWindow, return a cached
        _RoiViewDocument so tools and the Preview tab share the same instance.
        Otherwise return the real ImageDocument.
        
        IMPORTANT: Always check the currently active MDI subwindow first,
        as that's what the user expects to be the "active" document.
        """
        base_doc = None
        
        # ALWAYS check the MDI active subwindow first - this is the source of truth
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
        
        # Fallback to cached value only if MDI lookup failed
        if base_doc is None:
            if self._active_doc is not None and self._active_doc in self._docs:
                base_doc = self._active_doc
            else:
                base_doc = self._docs[-1] if self._docs else None

        # Non-image docs just pass through
        if base_doc is None or not isinstance(base_doc, ImageDocument) or base_doc.image is None:
            return base_doc

        # ✅ ROI-aware, CACHED preview doc
        vw = self._active_view_widget()
        if vw and hasattr(vw, "has_active_preview") and vw.has_active_preview():
            try:
                roi_doc = self.get_document_for_view(vw)  # <-- uses _roi_doc_cache
                if isinstance(roi_doc, _RoiViewDocument):
                    try:
                        name_suffix = f" (Preview {vw.current_preview_name() or ''})"
                        roi_doc.metadata["display_name"] = f"{base_doc.display_name()}{name_suffix}"
                    except Exception:
                        pass
                return roi_doc
            except Exception:
                return base_doc

        return base_doc



    def update_active_document(
        self,
        updated_image,
        metadata=None,
        step_name: str = "Edit",
        doc=None,   # 👈 NEW optional parameter
    ):

        # Prefer explicit doc if given; otherwise fall back to "active"
        view_doc = doc or self.get_active_document()

        # DEBUG: Trace why LinearFit might fail
        # print(f"[DocManager] update_active_document target: {view_doc}, type: {type(view_doc).__name__}")

        # NEW: Unwrap proxy objects (_DocProxy / LiveViewDocument)
        tname = type(view_doc).__name__
        if "LiveViewDocument" in tname:
            try:
                view_doc = view_doc._current()
            except Exception:
                pass
        elif "_DocProxy" in tname:
            try:
                view_doc = view_doc._target()
            except Exception:
                pass

        if view_doc is None:
            raise RuntimeError("No active document")

        old_img = getattr(view_doc, "image", None)
        old_shape = getattr(old_img, "shape", None)

        img = np.asarray(updated_image)
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        _debug_log_undo(
            "DocManager.update_active_document.entry",
            step_name=step_name,
            view_doc_type=type(view_doc).__name__,
            view_doc_id=id(view_doc),
            is_roi=isinstance(view_doc, _RoiViewDocument),
            old_shape=old_shape,
            new_shape=getattr(img, "shape", None),
        )

        # --- Extract operation parameters (if any) from metadata --------
        md = dict(metadata or {})
        op_params = md.pop("__op_params__", None)

        # If this is an ROI view doc, keep track of where this happened
        roi_tuple = None
        source_kind = "full"
        if isinstance(view_doc, _RoiViewDocument):
            roi_tuple = getattr(view_doc, "_roi", None)
            source_kind = "roi"

        # --- ROI preview branch: only update preview, no parent paste ----
        if isinstance(view_doc, _RoiViewDocument):
            # Update ONLY the preview; view repaint is driven by signals
            view_doc.apply_edit(img, md, step_name)

            # Record operation on the ROI doc itself
            if hasattr(view_doc, "record_operation"):
                try:
                    view_doc.record_operation(
                        step_name=step_name,
                        params=op_params,
                        roi=roi_tuple,
                        source=source_kind,
                    )
                except Exception:
                    pass
            _debug_log_undo(
                "DocManager.update_active_document.roi_after",
                step_name=step_name,
                view_doc_id=id(view_doc),
                roi=getattr(view_doc, "_roi", None),
                pundo_len=len(getattr(view_doc, "_pundo", [])),
                predo_len=len(getattr(view_doc, "_predo", [])),
            )
            return

        # --- Full image branch ------------------------------------------
        if isinstance(view_doc, ImageDocument):
            view_doc.apply_edit(img, md, step_name)
            try:
                self.imageRegionUpdated.emit(view_doc, None)
            except Exception:
                pass

            _debug_log_undo(
                "DocManager.update_active_document.full_after",
                step_name=step_name,
                view_doc_id=id(view_doc),
                undo_len=len(getattr(view_doc, "_undo", [])),
                redo_len=len(getattr(view_doc, "_redo", [])),
                final_shape=getattr(view_doc.image, "shape", None),
            )
            # Record operation on the full document
            if hasattr(view_doc, "record_operation"):
                try:
                    view_doc.record_operation(
                        step_name=step_name,
                        params=op_params,
                        roi=None,
                        source=source_kind,
                    )
                    
                except Exception:
                    pass
        else:
            raise RuntimeError("Active document is not an image")

    def get_active_operation_log(self) -> list[dict]:
        """
        Return the operation log for the *currently active* document-like
        (full image or ROI-preview). Empty list if none.
        """
        doc = self.get_active_document()
        if doc is None:
            return []
        get_log = getattr(doc, "get_operation_log", None)
        if callable(get_log):
            try:
                return get_log()
            except Exception:
                return []
        return []



    # Back-compat/aliases so tools can call any of these:
    def update_image(self, updated_image, metadata=None, step_name: str = "Edit"):
        self.update_active_document(updated_image, metadata, step_name)

    def set_image(self, img, metadata=None, step_name: str = "Edit"):
        self.update_active_document(img, metadata, step_name)

    def apply_edit_to_active(self, img, step_name: str = "Edit", metadata=None):
        self.update_active_document(img, metadata, step_name)
